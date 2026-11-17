"""Exact empty-circumcircle complex of 16 integer points, from a dense ReLU network under a cap on nonzero entries.

Input: 16 distinct points with integer coordinates 0..15, four bits per coordinate (128 inputs). Output: one value
per index triple i < j < k (560 triples, lexicographic order). A triple is in the complex iff its three points are not
collinear and no OTHER point lies strictly inside their circumcircle (open disk: cocircular points on the circle do
not disqualify a triangle, so degenerate grid inputs keep several overlapping triangles).

Primary score "vertex_score": for each point, 1 if the predicted set of complex triples containing it equals the true
set exactly, else 0; averaged over the 16 points and then over cases (Astra's per vertex incident set credit).

Exactness: the lifted incircle determinant of (a, b, c, d), taken relative to d, has entries dx, dy <= 15 in absolute
value and z = dx^2 + dy^2 <= 450, so each of its 6 terms is <= 15 * 15 * 450 = 101,250 and the whole determinant is
<= 607,500 < 2^24 = 16,777,216. Every intermediate of the direct float32 evaluation stays an exact integer.

Size estimates (rough, for resizing against the 500,000 nonzero cap; no reference network was built). Astra sized
the 12 point version (220 triples, 495 quadruples) at ~58k nonzeros for the product, sign, orientation and emptiness
blocks and 72-85k with preprocessing; the 16 point version has 560 triples and 1,820 quadruples.
  Naive: one independent incircle test per (triple, other point) = 560 * 13 = 7,280 tests (12 points: 1,980). Each test
    builds the translated lifted entries and expands a 3x3 determinant of products of small integers; even at only
    ~100 nonzero entries per test (far less than exact bit-level multiplication needs) that is 7,280 * 100 = 728,000.
  Shared determinants (Astra): q_i = x_i^2 + y_i^2, c_ij = x_i y_j - y_i x_j, A_ijk = q_i c_jk - q_j c_ik + q_k c_ij.
    Each four point incircle determinant is a signed sum of four shared A's. Scaling Astra's 58k by the quadruple count
    1,820 / 495 = 3.7 gives ~213k for the main blocks, plus ~50k for the 560 A's, orientations and 560 * 13 -> 1 AND
    reductions, plus 16 point preprocessing and per-vertex wiring: about 290,000 to 350,000 (fits under 500,000).
Dense weights are 24 layers of width 3,072 (see LAYERS and WIDTH).
"""

from itertools import combinations

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
POINTS = 16
GRID = 16  # coordinates 0..GRID - 1
TRIPLES = np.array(list(combinations(range(POINTS), 3)), dtype=np.int64)  # (560, 3), lexicographic
COORD_BITS = 4
INPUT_SIZE = 2 * COORD_BITS * POINTS
OUTPUT_SIZE = len(TRIPLES)
LAYERS = 24
WIDTH = 2048


def sample_points(*, rng: np.random.Generator, count: int) -> np.ndarray:
    """(count, POINTS, 2) integer points, distinct within each case, uniform over the GRID x GRID lattice, in random
    order; the rare draw with all points collinear is rejected."""
    out = []
    while len(out) < count:
        cells = rng.permutation(GRID * GRID)[:POINTS]
        pts = np.stack([cells // GRID, cells % GRID], axis=-1).astype(np.int64)
        if complex_bits(pts[None]).any():
            out.append(pts)
    return np.stack(out)


def encode(points: np.ndarray) -> np.ndarray:
    """Per point: x bits LSB first, then y bits LSB first."""
    shifts = np.arange(COORD_BITS)
    bits = (points[..., None] >> shifts) & 1  # (cases, 16, 2, 4)
    return bits.reshape(len(points), INPUT_SIZE).astype(np.float32)


def complex_bits(points: np.ndarray) -> np.ndarray:
    """(cases, 560) bool: exact integer test. Triple in the complex iff non-collinear and no point strictly inside."""
    a, b, c = (points[:, TRIPLES[:, k], :] for k in range(3))  # (cases, 560, 2)
    orient = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    d = points[:, None, :, :]  # (cases, 1, 16, 2)
    rows = []
    for p in (a, b, c):
        rel = p[:, :, None, :] - d  # (cases, 560, 16, 2)
        rows.append((rel[..., 0], rel[..., 1], rel[..., 0] ** 2 + rel[..., 1] ** 2))
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = rows
    det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    inside = det * orient[:, :, None] > 0  # det > 0 means inside when (a, b, c) is counterclockwise
    return (orient != 0) & ~inside.any(axis=2)


INCIDENCE = np.stack([(TRIPLES == p).any(axis=1) for p in range(POINTS)])  # (16, 560): triple t touches point p


def triangle_scores(*, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per case: the mean over the 16 points of 1 if the predicted set of triples touching that point equals the true
    set exactly, else 0 (Astra's per vertex incident set score)."""
    wrong = (predicted != truth)[:, None, :] & INCIDENCE[None]  # (cases, 16, 560)
    return (~wrong.any(axis=2)).mean(axis=1)


@register_target
class EmptyCircumcircle(Target):
    id = "empty-circumcircle"
    domain = "algorithmic"

    def __init__(self, *, cases: int = 200):
        self.cases = cases

    def description(self) -> str:
        spec = mlp_spec_text(inputs=INPUT_SIZE, outputs=OUTPUT_SIZE, layers=LAYERS, width=WIDTH)
        return f"""\
## Task
Given {POINTS} distinct points with integer coordinates in 0..{GRID - 1}, decide for every triple of points whether the
triangle they form has an empty circumcircle. A triple of point indices (i, j, k) with i < j < k belongs to the
complex iff (1) the three points are not collinear, and (2) no OTHER one of the {POINTS - 3} remaining points lies
strictly inside their circumcircle. The open disk is what counts: a point exactly on the circle (cocircular) does NOT
exclude the triple, so when four or more points are cocircular all the triangles among them with no point strictly
inside stay in the complex. Collinear triples are never in the complex. Exact integer geometry is the reference.

## Network
A feed-forward ReLU network.
Input x [{INPUT_SIZE}], every entry 0.0 or 1.0: point p (p = 0..{POINTS - 1}, in the given order) uses x[{2 * COORD_BITS}p .. {2 * COORD_BITS}p + {2 * COORD_BITS - 1}]:
the first {COORD_BITS} entries are the bits of its x coordinate, least significant bit first (x = sum of 2^k * bit k), the
next {COORD_BITS} are the bits of its y coordinate the same way. Coordinates are in 0 to {GRID - 1}. Points are distinct and
not all collinear. The order of the points is random.
{spec}
Output y [{OUTPUT_SIZE}]: one value per triple, in lexicographic order of (i, j, k) with 0 <= i < j < k < {POINTS}:
y[0] is (0, 1, 2), y[1] is (0, 1, 3), ... y[{POINTS - 3}] is (0, 1, {POINTS - 1}), then (0, 2, 3) and so on, ending with
y[{OUTPUT_SIZE - 1}] = (13, 14, 15). A triple is predicted "in the complex" when y > 0.5 and "not in the complex" otherwise.

## Scoring
Each evaluation draws {self.cases} fresh random point sets (distinct lattice points, uniform over the {GRID} x {GRID}
grid, rejecting only sets with all points collinear, so collinear and cocircular configurations occur naturally).
Scoring is per vertex. For each point p, its incident set is the set of complex triples that contain p. The point
earns 1 when the network's predicted incident set (triples containing p with y > 0.5) is exactly the true incident set,
and 0 otherwise. The case score is the mean over the {POINTS} points, and the final score is the mean case score over
the {self.cases} sets. A point that is in no triple of the complex is correct only if no triple containing it is predicted."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return mlp_shapes(inputs=INPUT_SIZE, outputs=OUTPUT_SIZE, layers=LAYERS, width=WIDTH)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        points = sample_points(rng=rng, count=self.cases)
        truth = complex_bits(points)
        predicted = mlp_forward(weights=weights, x=encode(points), layers=LAYERS) > 0.5
        score = triangle_scores(predicted=predicted, truth=truth)
        exact = (predicted == truth).all(axis=1)
        values = {
            "vertex_score": float(score.mean()),
            "exact_rate": float(exact.mean()),
            "triple_accuracy": float((predicted == truth).mean()),
            "recall": float((predicted & truth).sum() / max(truth.sum(), 1)),
            "false_triple_rate": float((predicted & ~truth).sum() / max(predicted.sum(), 1)),
            "mean_true_triples": float(truth.sum(axis=1).mean()),
            "cases": float(self.cases),
        }
        return Metrics(values=values, primary="vertex_score")
