"""Sudoku solver with a hand-written ReLU MLP inside a fixed envelope.

Input: 81 cells x 9 digits one-hot (an empty cell is all zeros). Output: 81 x 9 logits, argmax per cell. A puzzle counts
as solved when every cell's argmax equals the solution. Puzzles are generated from random solved grids by removing
clues in random order while `propagate` (naked singles, then hidden singles in rows, columns and boxes, repeated for
`rounds` rounds) still solves the puzzle, so every puzzle has a unique solution reachable by singles alone.
"""

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
CELLS = 81
DIGITS = 9
ROW = np.arange(CELLS) // 9
COL = np.arange(CELLS) % 9
BOX = (ROW // 3) * 3 + COL // 3
UNITS = {"row": ROW, "col": COL, "box": BOX}
SAME = {name: (unit[:, None] == unit[None, :]) for name, unit in UNITS.items()}
PEERS = (SAME["row"] | SAME["col"] | SAME["box"]) & ~np.eye(CELLS, dtype=bool)


def naked_step(c: np.ndarray) -> np.ndarray:
    fixed = c * (c.sum(axis=2, keepdims=True) == 1)
    blocked = np.einsum("pq,bqd->bpd", PEERS.astype(np.int32), fixed) > 0
    return c & ~blocked


def hidden_step(c: np.ndarray, same: np.ndarray) -> np.ndarray:
    others = np.einsum("pq,bqd->bpd", (same & ~np.eye(CELLS, dtype=bool)).astype(np.int32), c)
    single = c & (others == 0)
    keep = single.sum(axis=2, keepdims=True) == 0
    return np.where(keep, c, single)


def propagate(c: np.ndarray, rounds: int) -> np.ndarray:
    c = c.astype(bool)
    for _ in range(rounds):
        c = naked_step(c)
        for name in ["row", "col", "box"]:
            c = hidden_step(c, SAME[name])
    return c


def candidates_from_clues(grids: np.ndarray) -> np.ndarray:
    c = np.ones((len(grids), CELLS, DIGITS), dtype=bool)
    given = grids > 0
    c[given] = np.eye(DIGITS, dtype=bool)[grids[given] - 1]
    return c


def random_solutions(*, rng: np.random.Generator, count: int) -> np.ndarray:
    r, k = np.arange(9)[:, None], np.arange(9)[None, :]
    base = (3 * (r % 3) + r // 3 + k) % 9
    grids = []
    for _ in range(count):
        rows = np.concatenate([b * 3 + rng.permutation(3) for b in rng.permutation(3)])
        cols = np.concatenate([b * 3 + rng.permutation(3) for b in rng.permutation(3)])
        grids.append(rng.permutation(9)[base[rows][:, cols]] + 1)
    return np.array(grids).reshape(count, CELLS)


def generate_puzzles(*, rng: np.random.Generator, count: int, rounds: int) -> tuple[np.ndarray, np.ndarray]:
    solutions = random_solutions(rng=rng, count=count)
    puzzles = solutions.copy()
    orders = np.argsort(rng.random((count, CELLS)), axis=1)
    target = np.eye(DIGITS, dtype=bool)[solutions - 1]
    for step in range(CELLS):
        trial = puzzles.copy()
        trial[np.arange(count), orders[:, step]] = 0
        solved = (propagate(candidates_from_clues(trial), rounds) == target).all(axis=(1, 2))
        puzzles = np.where(solved[:, None], trial, puzzles)
    return puzzles, solutions


def encode(puzzles: np.ndarray) -> np.ndarray:
    x = np.zeros((len(puzzles), CELLS, DIGITS), dtype=np.float32)
    given = puzzles > 0
    x[given] = np.eye(DIGITS, dtype=np.float32)[puzzles[given] - 1]
    return x.reshape(len(puzzles), CELLS * DIGITS)


@register_target
class Sudoku(Target):
    id = "sudoku"
    domain = "algorithmic"

    def __init__(self, *, layers: int = 64, width: int = 1536, puzzles: int = 500, rounds: int = 7):
        self.layers = layers
        self.width = width
        self.puzzles = puzzles
        self.rounds = rounds

    def description(self) -> str:
        n = CELLS * DIGITS
        spec = mlp_spec_text(inputs=n, outputs=n, layers=self.layers, width=self.width)
        return f"""\
## Task
Solve 9x9 Sudoku puzzles: fill every empty cell so each row, column and 3x3 box holds the digits 1 to 9 once.

## Network
A feed-forward ReLU network.
Input x [{n}]: cell c (c = 9 * row + column, rows and columns 0 to 8, top-left first) holding digit d (1 to 9) sets
x[9 * c + (d - 1)] = 1. Empty cells have all 9 entries 0. Givens are always consistent.
{spec}
Output y [{n}]: the network's answer for cell c is the digit d whose logit y[9 * c + (d - 1)] is largest (ties go to the
smaller digit).

## Scoring
Fresh random puzzles, each with a unique solution that can be reached by repeatedly placing naked singles (a cell with
one possible digit) and hidden singles (a digit with one possible cell in a row, column or box); none needs guessing or
backtracking. Every puzzle is solved by at most {self.rounds} rounds of: place all naked singles, then hidden singles
in rows, then in columns, then in boxes. Puzzles have roughly 22 to 30 givens. A puzzle counts as solved when all 81 cells are correct. The score
is the fraction of {self.puzzles} puzzles solved."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        n = CELLS * DIGITS
        return mlp_shapes(inputs=n, outputs=n, layers=self.layers, width=self.width)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        puzzles, solutions = generate_puzzles(rng=rng, count=self.puzzles, rounds=self.rounds)
        y = mlp_forward(weights=weights, x=encode(puzzles), layers=self.layers).reshape(len(puzzles), CELLS, DIGITS)
        answer = y.argmax(axis=2) + 1
        empty = puzzles == 0
        return Metrics(
            values={
                "solved_rate": float((answer == solutions).all(axis=1).mean()),
                "cell_accuracy": float((answer == solutions).mean()),
                "empty_cell_accuracy": float((answer == solutions)[empty].mean()),
                "mean_givens": float((~empty).sum(axis=1).mean()),
                "puzzles": float(self.puzzles),
            },
            primary="solved_rate",
        )
