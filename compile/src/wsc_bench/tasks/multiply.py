"""Multiply two unsigned 32-bit integers with a hand-written ReLU MLP: 64 bits in (a then b), 64 bits out (a * b).

The input domain has 2^64 points, so no lookup table fits. Scoring draws fresh pairs per seed from a mix of operand
distributions (uniform, small, near 2^32, powers of two, one operand zero). The envelope is shallow on purpose: a
ripple carry or schoolbook unroll needs far more layers than it allows (see tests/test_multiply.py for a
construction that fits).
"""

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
OPERAND_BITS = 32
PRODUCT_BITS = 2 * OPERAND_BITS
PAIR_KINDS = ["uniform", "small", "near_max", "powers_of_two", "one_zero"]
# share of each kind in a sample; one_zero stays rare because a constant all-zero output already gets it right
PAIR_KIND_SHARES = [0.6, 0.1, 0.1, 0.18, 0.02]


def to_bits(values: np.ndarray, width: int) -> np.ndarray:
    """Unsigned integers (uint64) to bits, most significant first."""
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    return ((values.astype(np.uint64)[:, None] >> shifts) & np.uint64(1)).astype(np.float32)


def sample_pairs(*, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """count pairs (a, b) as uint64 arrays plus the kind index of each pair."""
    counts = np.floor(np.array(PAIR_KIND_SHARES) * count).astype(int)
    counts[0] += count - counts.sum()
    kinds = np.repeat(np.arange(len(PAIR_KINDS)), counts)
    top = 2**OPERAND_BITS
    uniform = rng.integers(0, top, size=(count, 2), dtype=np.uint64)
    small = rng.integers(1, 2 ** rng.integers(1, 17, size=(count, 2)), dtype=np.uint64)
    near_max = np.uint64(top - 1) - rng.integers(0, 2**12, size=(count, 2), dtype=np.uint64)
    powers = np.uint64(1) << rng.integers(0, OPERAND_BITS, size=(count, 2), dtype=np.uint64)
    one_zero = uniform.copy()
    one_zero[np.arange(count), rng.integers(0, 2, size=count)] = 0
    pairs = np.select([kinds[:, None] == k for k in range(len(PAIR_KINDS))],
                      [uniform, small, near_max, powers, one_zero]).astype(np.uint64)
    return pairs[:, 0], pairs[:, 1], kinds


@register_target
class Multiply32(Target):
    id = "multiply-32"
    domain = "algorithmic"

    def __init__(self, *, layers: int = 6, width: int = 1536, samples: int = 1000):
        self.layers = layers
        self.width = width
        self.samples = samples

    def description(self) -> str:
        spec = mlp_spec_text(inputs=PRODUCT_BITS, outputs=PRODUCT_BITS, layers=self.layers, width=self.width)
        return f"""\
## Task
Return the exact product a * b of two unsigned 32-bit integers 0 <= a, b < 2^32. The product is an unsigned
64-bit integer 0 <= a * b < 2^64.

## Network
A feed-forward ReLU network.
Input x [{PRODUCT_BITS}]: x[0:32] is a in binary and x[32:64] is b in binary, each most significant bit first, each bit 0.0 or 1.0.
{spec}
Output y [{PRODUCT_BITS}]: a * b in binary, most significant bit first. Output bit i is read as 1 when y[i] > 0.5, else 0.

## Scoring
Each evaluation draws {self.samples:,} fresh random pairs (a, b): uniformly random 32-bit values mixed with small
numbers, values near 2^32, powers of two and pairs where one operand is zero. The score is the fraction of pairs
whose 64 output bits equal a * b exactly."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return mlp_shapes(inputs=PRODUCT_BITS, outputs=PRODUCT_BITS, layers=self.layers, width=self.width)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        a, b, kinds = sample_pairs(rng=rng, count=self.samples)
        x = np.concatenate([to_bits(a, OPERAND_BITS), to_bits(b, OPERAND_BITS)], axis=1)
        truth = to_bits(a * b, PRODUCT_BITS).astype(bool)  # uint64 product of 32-bit operands is exact
        got = mlp_forward(weights=weights, x=x, layers=self.layers) > 0.5
        exact = (got == truth).all(axis=1)
        per_kind = {f"exact_match_{name}": float(exact[kinds == k].mean()) for k, name in enumerate(PAIR_KINDS)
                    if (kinds == k).any()}
        return Metrics(
            values={"exact_match": float(exact.mean()), "bit_accuracy": float((got == truth).mean()), **per_kind,
                    "tested_pairs": float(len(a))},
            primary="exact_match",
        )
