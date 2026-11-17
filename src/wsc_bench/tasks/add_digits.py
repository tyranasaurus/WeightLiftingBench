"""Add two 10-digit decimal integers with a hand-written ReLU MLP (the AdderBoard spec as a one-reply compile task).

Input is 20 one-hot digits (a then b, least significant digit first); output is the 11 digits of a + b, each as 10
logits. The sample is rich in long carry chains, since a network that only handles short chains scores near zero.
"""

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
IN_DIGITS = 10
OUT_DIGITS = IN_DIGITS + 1
IN_DIM = 2 * IN_DIGITS * 10
OUT_DIM = OUT_DIGITS * 10
KINDS = ["uniform", "carry_chain", "nines", "short"]
KIND_SHARES = [0.3, 0.5, 0.1, 0.1]


def digits_of(values: np.ndarray, count: int) -> np.ndarray:
    """int64 values to decimal digits, least significant first, shape (n, count)."""
    return (values[:, None] // 10 ** np.arange(count, dtype=np.int64)) % 10


def one_hot(digits: np.ndarray) -> np.ndarray:
    """(n, k) digits to (n, 10 k) float32, digit j of the number at columns 10 j .. 10 j + 9."""
    return np.eye(10, dtype=np.float32)[digits].reshape(len(digits), -1)


def digits_to_int(d: np.ndarray) -> np.ndarray:
    return (d * 10 ** np.arange(d.shape[1], dtype=np.int64)).sum(axis=1)


def sample_digits(*, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """count pairs of digit arrays (n, 10), least significant first, plus the kind index of each pair."""
    counts = np.floor(np.array(KIND_SHARES) * count).astype(int)
    counts[0] += count - counts.sum()
    kinds = np.repeat(np.arange(len(KINDS)), counts)
    a = rng.integers(0, 10, size=(count, IN_DIGITS))
    b = rng.integers(0, 10, size=(count, IN_DIGITS))
    pos = np.arange(IN_DIGITS)[None, :]
    # carry_chain: digits start .. start + length - 1 all sum to 9 (propagate), fed by a generate at start - 1 (or by
    # nothing, for the half of chains that start at digit 0, where a carry only enters if digit 0 itself generates)
    start = rng.integers(1, IN_DIGITS, size=count)[:, None]
    length = rng.integers(2, IN_DIGITS + 1, size=count)[:, None]
    low = (rng.random(count) < 0.5)[:, None]
    start = np.where(low, 0, start)
    chain = (pos >= start) & (pos < start + length)
    gen = ((pos == start - 1) & ~low) | (low & (pos == 0))  # low chains get their carry-in from a generate at digit 0
    is_chain = (kinds == 1)[:, None]
    a = np.where(is_chain & gen, np.maximum(a, 5), a)  # a >= 5 so some b makes the digit sum reach 10
    chain_b = np.where(chain & ~gen, 9 - a, np.where(gen, np.maximum(b, 10 - a), b))
    b = np.where(is_chain, chain_b, b)
    # nines: a = 10^k - 1, b = 1 or a couple of random low digits
    k = rng.integers(1, IN_DIGITS + 1, size=(count, 1))
    nines = np.where(pos < k, 9, 0)
    small = np.where(rng.random((count, 1)) < 0.5, (pos == 0) * 1, rng.integers(0, 10, size=a.shape) * (pos < 2))
    a = np.where((kinds == 2)[:, None], nines, a)
    b = np.where((kinds == 2)[:, None], small, b)
    # short: numbers with fewer digits
    la = rng.integers(1, IN_DIGITS + 1, size=(count, 1))
    lb = rng.integers(1, IN_DIGITS + 1, size=(count, 1))
    a = np.where((kinds == 3)[:, None] & (pos >= la), 0, a)
    b = np.where((kinds == 3)[:, None] & (pos >= lb), 0, b)
    return a, b, kinds


@register_target
class AddDigits10(Target):
    id = "add-10digit"
    domain = "algorithmic"

    def __init__(self, *, layers: int = 8, width: int = 512, samples: int = 2000):
        self.layers = layers
        self.width = width
        self.samples = samples

    def description(self) -> str:
        spec = mlp_spec_text(inputs=IN_DIM, outputs=OUT_DIM, layers=self.layers, width=self.width)
        return f"""\
## Task
Add two non-negative integers a and b with up to {IN_DIGITS} decimal digits each (0 <= a, b < 10^{IN_DIGITS}). The sum
a + b has up to {OUT_DIGITS} decimal digits. Shorter numbers are padded with leading zeros.

## Network
A feed-forward ReLU network.
Input x [{IN_DIM}]: the {IN_DIGITS} digits of a, then the {IN_DIGITS} digits of b, each digit as a one-hot block of 10 values,
least significant digit first. So x[10*j + d] = 1.0 when the digit of a at position j (position 0 is the ones
digit, position 1 the tens digit, and so on) equals d, and x[{IN_DIGITS * 10} + 10*j + d] = 1.0 when the digit of b at position j
equals d. All other entries are 0.0, so every block has exactly one 1.0.
{spec}
Output y [{OUT_DIM}]: the {OUT_DIGITS} digits of a + b, least significant digit first, each as a block of 10 logits.
y[10*j : 10*j + 10] are the logits for digit j of the sum (digit {OUT_DIGITS - 1} is 0 or 1). Digit j is read as the
argmax of its block.

## Scoring
Each evaluation draws {self.samples:,} fresh pairs (a, b): uniformly random {IN_DIGITS}-digit numbers mixed with pairs built to
produce long carry chains (a carry rippling across many digits, as in 999999999 + 1), numbers made of nines, and
numbers with fewer digits. The score is the fraction of pairs where all {OUT_DIGITS} output digits are correct."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return mlp_shapes(inputs=IN_DIM, outputs=OUT_DIM, layers=self.layers, width=self.width)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        a, b, kinds = sample_digits(rng=rng, count=self.samples)
        x = np.concatenate([one_hot(a), one_hot(b)], axis=1)
        truth = digits_of(digits_to_int(a) + digits_to_int(b), OUT_DIGITS)
        logits = mlp_forward(weights=weights, x=x, layers=self.layers).reshape(len(a), OUT_DIGITS, 10)
        got = logits.argmax(axis=2)
        exact = (got == truth).all(axis=1)
        per_kind = {f"exact_match_{name}": float(exact[kinds == k].mean()) for k, name in enumerate(KINDS)
                    if (kinds == k).any()}
        return Metrics(
            values={"exact_match": float(exact.mean()), "digit_accuracy": float((got == truth).mean()), **per_kind,
                    "tested_pairs": float(len(a))},
            primary="exact_match",
        )
