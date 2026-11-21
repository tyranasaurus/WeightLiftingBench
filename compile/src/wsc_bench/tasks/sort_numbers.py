"""Sort n real numbers with a hand-written ReLU MLP: the editor chooses the architecture inside a fixed envelope.

Network: x [n] -> Linear -> ReLU -> ... -> Linear -> y [n] (wsc_bench.tasks.relu_mlp). The editor picks the effective
architecture by which units it uses; unused units stay zero. Scoring: fresh random lists from a mix of distributions
(wide and narrow reals, integers with many duplicates, already sorted, reversed, constant). A list counts as sorted
when every output is within `tolerance` of the true ascending order. Forward pass runs in float32.
"""

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
LIST_KINDS = ["wide_reals", "narrow_reals", "duplicate_ints", "sorted", "reversed", "constant"]
MAX_ABS_VALUE = 1000.0


def sample_lists(*, rng: np.random.Generator, count: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    kinds = rng.integers(len(LIST_KINDS), size=count)
    wide = rng.uniform(-MAX_ABS_VALUE, MAX_ABS_VALUE, size=(count, n))
    narrow = rng.uniform(-1.0, 1.0, size=(count, n))
    ints = rng.integers(0, 4, size=(count, n)).astype(np.float64)
    constant = np.repeat(rng.uniform(-MAX_ABS_VALUE, MAX_ABS_VALUE, size=(count, 1)), n, axis=1)
    x = np.select(
        [kinds[:, None] == k for k in range(4)] + [kinds[:, None] == 5],
        [wide, narrow, ints, np.sort(wide, axis=1), constant],
        default=-np.sort(-wide, axis=1),  # kind 4: reversed
    )
    return x.astype(np.float32), kinds


@register_target
class SortNumbers(Target):
    id = "sort-numbers"
    domain = "algorithmic"

    def __init__(self, *, n: int = 10, layers: int = 12, width: int = 64, lists: int = 10_000, tolerance: float = 0.01):
        self.n = n
        self.layers = layers
        self.width = width
        self.lists = lists
        self.tolerance = tolerance

    def description(self) -> str:
        spec = mlp_spec_text(inputs=self.n, outputs=self.n, layers=self.layers, width=self.width)
        return f"""\
## Network
A feed-forward ReLU network that must sort {self.n} real numbers in ascending order.
Input x is a float32 vector of {self.n} numbers. Output y must be the same numbers sorted ascending: y[0] smallest.
{spec}

## Scoring
Inputs are any {self.n} real numbers with |x| <= {MAX_ABS_VALUE:g}, including duplicates, ties, already sorted, reversed and
constant lists, and both very wide and very narrow ranges. A list counts as correct when every output is within
{self.tolerance:g} of the true sorted value. The score is the fraction of {self.lists} fresh random lists that are correct."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return mlp_shapes(inputs=self.n, outputs=self.n, layers=self.layers, width=self.width)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        x, kinds = sample_lists(rng=rng, count=self.lists, n=self.n)
        y = mlp_forward(weights=weights, x=x, layers=self.layers).astype(np.float64)
        error = np.abs(y - np.sort(x.astype(np.float64), axis=1)).max(axis=1)
        correct = error <= self.tolerance
        values = {
            "sorted_rate": float(correct.mean()),
            "mean_max_abs_error": float(error.mean()),
            "worst_abs_error": float(error.max()),
            "lists": float(self.lists),
        }
        for k, kind in enumerate(LIST_KINDS):
            values[f"sorted_rate_{kind}"] = float(correct[kinds == k].mean())
        return Metrics(values=values, primary="sorted_rate")
