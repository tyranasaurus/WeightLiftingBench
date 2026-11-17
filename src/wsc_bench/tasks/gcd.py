"""Greatest common divisor of two 10 bit integers with a hand-written ReLU MLP: a and b in binary -> gcd(a, b) in binary.

Both operands are drawn from [1, 1023]. Uniform pairs mostly have gcd 1 (a constant output of 1 would score about
0.6), so a scoring sample mixes uniform pairs with pairs that share a factor and pairs where one operand divides the
other. The envelope is deep enough for an unrolled Euclid or binary GCD, and wide enough for a one-hot lookup of each
operand (see tests/test_gcd.py for a construction that uses the latter).
"""

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
BITS = 10
MAX_VALUE = (1 << BITS) - 1
PAIR_KINDS = ["uniform", "shared_factor", "multiple"]
PAIR_KIND_SHARES = [0.4, 0.5, 0.1]


def to_bits(values: np.ndarray, width: int = BITS) -> np.ndarray:
    """Integers to bits, least significant first."""
    return ((values.astype(np.int64)[:, None] >> np.arange(width)) & 1).astype(np.float32)


def sample_pairs(*, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """count pairs (a, b), each in [1, 1023], plus the kind index of each pair."""
    counts = np.floor(np.array(PAIR_KIND_SHARES) * count).astype(int)
    counts[0] += count - counts.sum()
    kinds = np.repeat(np.arange(len(PAIR_KINDS)), counts)
    uniform = rng.integers(1, MAX_VALUE + 1, size=(count, 2))
    g = rng.integers(2, MAX_VALUE // 2 + 1, size=count)  # a = g x, b = g y, so the gcd is a multiple of g >= 2
    shared = g[:, None] * (1 + (rng.random((count, 2)) * (MAX_VALUE // g)[:, None]).astype(np.int64))
    small = rng.integers(1, MAX_VALUE // 2 + 1, size=count)  # b divides a, or a = b
    multiple = np.stack([small * (1 + (rng.random(count) * (MAX_VALUE // small)).astype(np.int64)), small], axis=1)
    pairs = np.select([kinds[:, None] == k for k in range(len(PAIR_KINDS))], [uniform, shared, multiple])
    swap = rng.random(count) < 0.5
    pairs[swap] = pairs[swap][:, ::-1]
    return pairs[:, 0], pairs[:, 1], kinds


@register_target
class Gcd10(Target):
    id = "gcd-10bit"
    domain = "algorithmic"

    def __init__(self, *, layers: int = 8, width: int = 2048, samples: int = 1000):
        self.layers = layers
        self.width = width
        self.samples = samples

    def description(self) -> str:
        spec = mlp_spec_text(inputs=2 * BITS, outputs=BITS, layers=self.layers, width=self.width)
        return f"""\
## Task
Return the greatest common divisor gcd(a, b) of two integers 1 <= a, b <= {MAX_VALUE} (that is, below 2^{BITS}).
The gcd is an integer between 1 and {MAX_VALUE}.

## Network
A feed-forward ReLU network.
Input x [{2 * BITS}]: x[0:{BITS}] is a in binary and x[{BITS}:{2 * BITS}] is b in binary, each least significant bit first, each bit 0.0 or 1.0.
{spec}
Output y [{BITS}]: gcd(a, b) in binary, least significant bit first. Output bit i is read as 1 when y[i] > 0.5, else 0.

## Scoring
Each evaluation draws {self.samples:,} fresh pairs (a, b): uniformly random pairs mixed with pairs that share a common
factor of at least 2 and pairs where one operand is a multiple of the other (or equal to it). The score is the fraction
of pairs whose {BITS} output bits equal gcd(a, b) exactly."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return mlp_shapes(inputs=2 * BITS, outputs=BITS, layers=self.layers, width=self.width)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        a, b, kinds = sample_pairs(rng=rng, count=self.samples)
        assert a.min() >= 1 and b.min() >= 1 and a.max() <= MAX_VALUE and b.max() <= MAX_VALUE
        x = np.concatenate([to_bits(a), to_bits(b)], axis=1)
        truth_value = np.gcd(a, b)
        truth = to_bits(truth_value) > 0.5
        bit_correct = (mlp_forward(weights=weights, x=x, layers=self.layers) > 0.5) == truth
        exact = bit_correct.all(axis=1)
        values = {"exact": float(exact.mean()), "bit_accuracy": float(bit_correct.mean()),
                  "always_one_exact": float((truth_value == 1).mean()), "tested_pairs": float(len(a))}
        values |= {f"{name}_exact": float(exact[kinds == i].mean()) for i, name in enumerate(PAIR_KINDS)}
        return Metrics(values=values, primary="exact")
