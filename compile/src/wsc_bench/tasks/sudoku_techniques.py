"""Sudoku puzzles that need solving techniques beyond singles, but never guessing, inside a fixed ReLU MLP envelope.

Same input and output encoding and the same full-grid scoring as `sudoku`. The puzzle class differs: singles-only
propagation (naked and hidden singles, repeated to a fixpoint) leaves every puzzle unsolved, while `propagate` with the
technique set below solves it within `rounds` rounds, so every puzzle has a unique solution reachable without search.

Technique set: naked singles, hidden singles, naked pairs, hidden pairs, pointing (a digit confined to one row or
column inside a box leaves the rest of that row or column) and claiming (a digit confined to one box inside a row or
column leaves the rest of that box). A round is three steps, one per unit type (rows, then columns, then boxes). A step
detects every deduction of that unit type from the same candidate state and applies them together; the row and column
steps also apply pointing and claiming between boxes and rows (or columns).

All states reached from a puzzle's givens are consistent with its solution, so the detectors below may assume that
(for example, a unit never holds two naked pairs that share a digit).
"""

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text
from wsc_bench.tasks.sudoku import CELLS, DIGITS, SAME, UNITS, candidates_from_clues, encode

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
UNIT_TYPES = ["row", "col", "box"]
TECHNIQUES = frozenset({"naked_single", "hidden_single", "naked_pair", "hidden_pair", "box_line"})
SINGLES = frozenset({"naked_single", "hidden_single"})
UNIT_CELLS = {t: np.argsort(UNITS[t], kind="stable").reshape(9, 9) for t in UNIT_TYPES}  # unit -> its cells in order
BITS = 2 ** np.arange(9)
NOT_SELF = ~np.eye(9, dtype=bool)
CANDIDATE_BATCH_FACTOR = 13  # candidate grids per missing puzzle in one batch; at 5 rounds about 1 in 11 qualifies


def twins(*, codes: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """codes, eligible [..., 9]: which entries share their code with another eligible entry of the same group."""
    same = (codes[..., :, None] == codes[..., None, :]) & eligible[..., :, None] & eligible[..., None, :] & NOT_SELF
    return same.any(axis=-1)


def box_line_eliminations(c: np.ndarray, *, line: str) -> np.ndarray:
    """Pointing and claiming between boxes and rows (line='row') or columns (line='col'), all from the same state."""
    g = c.reshape(len(c), 3, 3, 3, 3, DIGITS)  # batch, band, row in band, stack, column in stack, digit
    if line == "col":
        g = g.transpose(0, 3, 4, 1, 2, 5)
    segment = g.sum(axis=4)  # batch, band, line in band, box in band, digit: candidates in one box-line segment
    pointing = segment.sum(axis=2, keepdims=True) == segment  # the box's copies of the digit all lie on this line
    claiming = segment.sum(axis=3, keepdims=True) == segment  # the line's copies of the digit all lie in this box
    hit = (pointing.sum(axis=3, keepdims=True) - pointing > 0) | (claiming.sum(axis=2, keepdims=True) - claiming > 0)
    elim = np.broadcast_to(hit[:, :, :, :, None, :], g.shape)
    if line == "col":
        elim = elim.transpose(0, 3, 4, 1, 2, 5)
    return elim.reshape(c.shape)


def technique_step(c: np.ndarray, *, unit_type: str, techniques: frozenset[str] = TECHNIQUES) -> np.ndarray:
    """One step: every enabled deduction for units of `unit_type`, detected from the same state, applied together.

    Naked singles and naked pairs lock cells: a cell with one candidate, or two cells of a unit with the same two
    candidates. Other cells of the unit lose the locked digits. Hidden singles and hidden pairs lock digits: a digit
    with one place in the unit, or two digits with the same two places. Those places lose every other digit."""
    g = c[:, UNIT_CELLS[unit_type]]  # batch, unit, member, digit
    size = g.sum(axis=3)
    count = g.sum(axis=2)
    locked_cell = np.zeros(size.shape, dtype=bool)
    if "naked_single" in techniques:
        locked_cell |= size == 1
    if "naked_pair" in techniques:
        locked_cell |= twins(codes=g @ BITS, eligible=size == 2)
    locked_digit = np.zeros(count.shape, dtype=bool)
    if "hidden_single" in techniques:
        locked_digit |= count == 1
    if "hidden_pair" in techniques:
        locked_digit |= twins(codes=g.transpose(0, 1, 3, 2) @ BITS, eligible=count == 2)
    naked_digits = (g & locked_cell[:, :, :, None]).any(axis=2)
    hidden_cell = (g & locked_digit[:, :, None, :]).any(axis=3)
    keep = ~(naked_digits[:, :, None, :] & ~locked_cell[:, :, :, None])
    keep &= locked_digit[:, :, None, :] | ~hidden_cell[:, :, :, None]
    out = c.copy()
    out[:, UNIT_CELLS[unit_type]] = g & keep
    if "box_line" in techniques and unit_type != "box":
        out &= ~box_line_eliminations(c, line=unit_type)
    return out


def technique_round(c: np.ndarray, *, techniques: frozenset[str] = TECHNIQUES) -> np.ndarray:
    for unit_type in UNIT_TYPES:
        c = technique_step(c, unit_type=unit_type, techniques=techniques)
    return c


def propagate(c: np.ndarray, *, rounds: int, techniques: frozenset[str] = TECHNIQUES) -> np.ndarray:
    """`rounds` technique rounds. A grid that is solved or stops changing is frozen, which never changes the result."""
    c = c.astype(bool).copy()
    active = np.arange(len(c))
    for _ in range(rounds):
        if len(active) == 0:
            break
        nxt = technique_round(c[active], techniques=techniques)
        moving = (nxt != c[active]).any(axis=(1, 2)) & ~(nxt.sum(axis=2) == 1).all(axis=1)
        c[active] = nxt
        active = active[moving]
    return c


EYE_CELLS = np.eye(CELLS, dtype=bool)
PEERS_F = ((SAME["row"] | SAME["col"] | SAME["box"]) & ~EYE_CELLS).astype(np.float32)
OTHERS_F = {t: (SAME[t] & ~EYE_CELLS).astype(np.float32) for t in UNIT_TYPES}


def over_cells(m: np.ndarray, x: np.ndarray) -> np.ndarray:
    """m [81, 81] applied over the cell axis of x [batch, 81, k]."""
    b, n, k = x.shape
    return (m @ x.transpose(1, 0, 2).reshape(n, b * k)).reshape(n, b, k).transpose(1, 0, 2)


def singles_round(c: np.ndarray) -> np.ndarray:
    """One round of `wsc_bench.tasks.sudoku.propagate`: naked singles, then hidden singles in rows, columns, boxes."""
    f = c.astype(np.float32)
    c = c & ~(over_cells(PEERS_F, f * (f.sum(axis=2, keepdims=True) == 1)) > 0)
    for t in UNIT_TYPES:
        single = c & (over_cells(OTHERS_F[t], c.astype(np.float32)) == 0)
        c = np.where(single.any(axis=2, keepdims=True), single, c)
    return c


def singles_fixpoint(c: np.ndarray) -> np.ndarray:
    c = c.astype(bool).copy()
    active = np.arange(len(c))
    while len(active):
        nxt = singles_round(c[active])
        moving = (nxt != c[active]).any(axis=(1, 2))
        c[active] = nxt
        active = active[moving]
    return c


def random_grids(*, rng: np.random.Generator, count: int) -> np.ndarray:
    """Random solved grids: give the most constrained open cell a random candidate, propagate singles, and restart a
    grid from empty on contradiction."""
    c = np.ones((count, CELLS, DIGITS), dtype=bool)
    grids = np.zeros((count, CELLS), dtype=np.int64)
    open_rows = np.arange(count)
    while len(open_rows):
        cc = singles_fixpoint(c[open_rows])
        size = cc.sum(axis=2)
        dead = (size == 0).any(axis=1)
        solved = (size == 1).all(axis=1) & ~dead
        grids[open_rows[solved]] = cc[solved].argmax(axis=2) + 1
        cc[dead] = True
        live = np.flatnonzero(~dead & ~solved)
        cell = np.where(size[live] > 1, size[live], DIGITS + 1).argmin(axis=1)
        pick = (cc[live, cell] * rng.random((len(live), DIGITS))).argmax(axis=1)
        cc[live, cell] = np.eye(DIGITS, dtype=bool)[pick]
        c[open_rows] = cc
        open_rows = open_rows[~solved]
    return grids


def solves(*, candidates: np.ndarray, solutions: np.ndarray) -> np.ndarray:
    return (candidates == np.eye(DIGITS, dtype=bool)[solutions - 1]).all(axis=(1, 2))


def remove_clues(*, rng: np.random.Generator, solutions: np.ndarray, rounds: int) -> np.ndarray:
    """Remove clues in random order, keeping each removal while `rounds` technique rounds still solve the grid."""
    count = len(solutions)
    puzzles = solutions.copy()
    orders = np.argsort(rng.random((count, CELLS)), axis=1)
    for step in range(CELLS):
        trial = puzzles.copy()
        trial[np.arange(count), orders[:, step]] = 0
        ok = solves(candidates=propagate(candidates_from_clues(trial), rounds=rounds), solutions=solutions)
        puzzles = np.where(ok[:, None], trial, puzzles)
    return puzzles


def generate_puzzles(*, rng: np.random.Generator, count: int, rounds: int) -> tuple[np.ndarray, np.ndarray]:
    """Clue-minimal puzzles for the technique propagator that singles-only propagation cannot finish."""
    puzzles, solutions = [], []
    while (missing := count - sum(len(p) for p in puzzles)) > 0:
        grids = random_grids(rng=rng, count=missing * CANDIDATE_BATCH_FACTOR)
        p = remove_clues(rng=rng, solutions=grids, rounds=rounds)
        needs_technique = ~solves(candidates=singles_fixpoint(candidates_from_clues(p)), solutions=grids)
        puzzles.append(p[needs_technique])
        solutions.append(grids[needs_technique])
    return np.concatenate(puzzles)[:count], np.concatenate(solutions)[:count]


@register_target
class SudokuTechniques(Target):
    id = "sudoku-techniques"
    domain = "algorithmic"

    def __init__(self, *, layers: int = 80, width: int = 2048, puzzles: int = 300, rounds: int = 5):
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
Fresh random puzzles, each with a unique solution that needs no guessing or backtracking but cannot be reached with
singles alone: repeatedly placing naked singles (a cell with one possible digit) and hidden singles (a digit with one
possible cell in a row, column or box) until nothing changes always leaves some cell open. Every puzzle is solved by at
most {self.rounds} rounds of candidate elimination with these techniques:
- naked single: a cell with one candidate removes that digit from the other cells of its unit
- hidden single: a digit with one possible cell in a unit becomes that cell's only candidate
- naked pair: two cells of a unit with the same two candidates remove both digits from the unit's other cells
- hidden pair: two digits whose only possible cells in a unit are the same two cells clear every other candidate
  from those two cells
- pointing: if a digit's candidates inside a box all lie in one row (or column), it is removed from the rest of that
  row (or column)
- claiming: if a digit's candidates inside a row (or column) all lie in one box, it is removed from the rest of that
  box
A round is three steps: rows, then columns, then boxes. Each step finds every naked single, hidden single, naked pair
and hidden pair within units of that type from the same candidates and applies them together; the row step also
applies pointing and claiming between boxes and rows, the column step between boxes and columns. Puzzles have
roughly 22 to 30 givens. A puzzle counts as solved when all 81 cells are correct. The score is the fraction of
{self.puzzles} puzzles solved."""

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
