"""Sudoku with any unique-solution puzzle, scored per cell with a penalty for wrong digits so abstaining beats guessing.

Puzzles are minimal: start from a random solved grid and remove clues in random order whenever a backtracking count
still finds exactly one solution, so no clue can be removed without losing uniqueness. About 40% are solvable by
singles, 20% need pairs or box-line reductions, and 40% need search (200 puzzles, hidden seed 0).

A cell is filled with digit d when y[d] > 0.5 and every other logit of that cell is < 0.5; anything else is blank.
Per puzzle: cell_score = (correct empty cells - WRONG_CELL_PENALTY * wrong empty cells) / empty cells. Givens are not
scored. Primary: max(0, mean cell_score).
"""

import numpy as np

from wsc_bench.tasks import sudoku_techniques as techniques
from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text
from wsc_bench.tasks.sudoku import BOX, CELLS, COL, DIGITS, ROW, candidates_from_clues, encode, propagate

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
WRONG_CELL_PENALTY = 2
FIXPOINT_ROUNDS = 60
PEER_LIST = [[q for q in range(CELLS) if q != p and (ROW[q] == ROW[p] or COL[q] == COL[p] or BOX[q] == BOX[p])]
             for p in range(CELLS)]


def count_solutions(*, grid: list[int], limit: int) -> int:
    """Backtracking count, branching on the open cell with fewest options, stopping at `limit`."""
    best, best_options = -1, None
    for p in range(CELLS):
        if grid[p]:
            continue
        used = {grid[q] for q in PEER_LIST[p]}
        options = [d for d in range(1, 10) if d not in used]
        if best_options is None or len(options) < len(best_options):
            best, best_options = p, options
            if len(options) <= 1:
                break
    if best_options is None:
        return 1
    total = 0
    for d in best_options:
        grid[best] = d
        total += count_solutions(grid=grid, limit=limit - total)
        grid[best] = 0
        if total >= limit:
            break
    return total


def minimal_puzzle(*, rng: np.random.Generator, solution: np.ndarray) -> np.ndarray:
    grid = [int(v) for v in solution]
    for p in rng.permutation(CELLS):
        keep, grid[p] = grid[p], 0
        if count_solutions(grid=grid, limit=2) != 1:
            grid[p] = keep
    return np.array(grid)


def generate_minimal_puzzles(*, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    solutions = techniques.random_grids(rng=rng, count=count)
    return np.array([minimal_puzzle(rng=rng, solution=s) for s in solutions]), solutions


def difficulty_buckets(*, puzzles: np.ndarray, solutions: np.ndarray) -> dict[str, np.ndarray]:
    target = np.eye(DIGITS, dtype=bool)[solutions - 1]
    start = candidates_from_clues(puzzles)
    singles = (propagate(start, FIXPOINT_ROUNDS) == target).all(axis=(1, 2))
    logic = (techniques.propagate(start, rounds=FIXPOINT_ROUNDS) == target).all(axis=(1, 2))
    return {"singles": singles, "pairs_box_line": logic & ~singles, "search": ~logic}


def read_cells(y: np.ndarray) -> np.ndarray:
    """Digit per cell (1 to 9), or 0 for blank: exactly one logit above 0.5 and all others below 0.5."""
    above, below = y > 0.5, y < 0.5
    filled = (above.sum(axis=2) == 1) & ((above | below).all(axis=2))
    return np.where(filled, y.argmax(axis=2) + 1, 0)


@register_target
class SudokuAny(Target):
    id = "sudoku-any"
    domain = "algorithmic"

    def __init__(self, *, layers: int = 64, width: int = 1536, puzzles: int = 200):
        self.layers = layers
        self.width = width
        self.puzzles = puzzles

    def description(self) -> str:
        n = CELLS * DIGITS
        spec = mlp_spec_text(inputs=n, outputs=n, layers=self.layers, width=self.width)
        return f"""\
## Task
Solve 9x9 Sudoku puzzles: fill every empty cell so each row, column and 3x3 box holds the digits 1 to 9 once.
The network should handle any puzzle with a unique solution, from easy to the hardest known puzzles.

## Network
A feed-forward ReLU network.
Input x [{n}]: cell c (c = 9 * row + column, rows and columns 0 to 8, top-left first) holding digit d (1 to 9) sets
x[9 * c + (d - 1)] = 1. Empty cells have all 9 entries 0. Givens are always consistent.
{spec}
Output y [{n}]: cell c is filled with digit d when y[9 * c + (d - 1)] > 0.5 and the other 8 logits of cell c are all
< 0.5. Any other pattern leaves the cell blank.

## Scoring
Fresh random puzzles, each with a unique solution and minimal: they are made from random solved grids by removing
clues while exactly one solution remains, until no further clue can be removed. They span every difficulty. Some
fall to naked and hidden singles alone, some need further logical techniques, and many need guessing or search.
Puzzles have roughly 20 to 28 givens. Only the empty cells are scored. For each puzzle:
    cell_score = (correct cells - {WRONG_CELL_PENALTY} * wrong cells) / empty cells
A blank cell scores 0 and a wrong digit costs {WRONG_CELL_PENALTY}, so leave a cell blank unless you are sure of it.
The score is the mean cell_score over {self.puzzles} puzzles (floored at 0)."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        n = CELLS * DIGITS
        return mlp_shapes(inputs=n, outputs=n, layers=self.layers, width=self.width)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        puzzles, solutions = generate_minimal_puzzles(rng=rng, count=self.puzzles)
        y = mlp_forward(weights=weights, x=encode(puzzles), layers=self.layers).reshape(len(puzzles), CELLS, DIGITS)
        cells = read_cells(y)
        empty = puzzles == 0
        correct = ((cells == solutions) & empty).sum(axis=1)
        wrong = ((cells > 0) & (cells != solutions) & empty).sum(axis=1)
        score = (correct - WRONG_CELL_PENALTY * wrong) / empty.sum(axis=1)
        solved = correct == empty.sum(axis=1)
        values = {
            "cell_score": float(max(0.0, score.mean())),
            "solved_rate": float(solved.mean()),
            "correct_cell_rate": float(correct.sum() / empty.sum()),
            "wrong_cell_rate": float(wrong.sum() / empty.sum()),
            "puzzles_with_wrong_cells": float((wrong > 0).mean()),
            "mean_givens": float((~empty).sum(axis=1).mean()),
            "puzzles": float(self.puzzles),
        }
        for name, mask in difficulty_buckets(puzzles=puzzles, solutions=solutions).items():
            values[f"share_{name}"] = float(mask.mean())
            values[f"solved_rate_{name}"] = float(solved[mask].mean()) if mask.any() else 0.0
            values[f"cell_score_{name}"] = float(score[mask].mean()) if mask.any() else 0.0
        return Metrics(values=values, primary="cell_score")
