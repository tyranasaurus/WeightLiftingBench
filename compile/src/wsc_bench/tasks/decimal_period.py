"""Period of the decimal expansion of 1/n (Project Euler 26): the multiplicative order of 10 mod n, as a sparse ReLU net.

Input n has 2^31 <= n < 2^32, gcd(n, 10) = 1 and every prime factor below 4096. The output is ord_n(10) in 32 bits.
Long division needs up to about 2^31 steps, so the network has to compute the period from the factorization. Weights
use the sparse entry-list envelope of n_queens_sparse (layer, row, col, val plus dense biases). About 8% of 32-bit
numbers are 4096-smooth, so roughly 1.4e8 inputs are valid and a lookup table cannot fit the cap.

Sampling: three quarters of each evaluation are uniform over valid n (rejection from uniform 32-bit draws with
vectorized trial division). The other quarter is n = s * m with s a special factor (9, 27, 487^2, high powers of 3 or
7; 3 and 487 are the primes where ord_{p^2}(10) = ord_p(10)) and m drawn uniformly so that n stays valid and in range.
"""

from functools import cache
from math import gcd

import numpy as np
from scipy import sparse

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.multiply import to_bits
from wsc_bench.tasks.registry import register_target

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
BITS = 32
LOW = 2**31
HIGH = 2**32
SMOOTH_BOUND = 4096
LAYERS = 64
WIDTH = 2**18  # 262,144 units per hidden layer
MAX_ENTRIES = 10_000_000
BATCH = 50
OVERSAMPLED_SHARE = 0.25
SPECIAL_FACTORS = [9, 27, 487**2, 9 * 487**2, 27 * 487, *(3**k for k in range(5, 18)), *(7**k for k in range(4, 11)),
                   3**6 * 7**4, 3**10 * 7**3]


def primes_below(limit: int) -> list[int]:
    sieve = np.ones(limit, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return [int(p) for p in np.flatnonzero(sieve)]


ODD_PRIMES = [p for p in primes_below(SMOOTH_BOUND) if p not in (2, 5)]  # the 562 primes allowed in n


def is_valid(values: np.ndarray) -> np.ndarray:
    """Whether each int64 value is coprime to 10 and has every prime factor below SMOOTH_BOUND."""
    rest = values.copy()
    keep = (rest % 2 != 0) & (rest % 5 != 0)
    for p in ODD_PRIMES:
        divisible = keep & (rest % p == 0)
        while divisible.any():
            rest[divisible] //= p
            divisible = rest % p == 0
    return keep & (rest == 1)


def sample_valid(*, rng: np.random.Generator, count: int, low: int, high: int) -> np.ndarray:
    """count values uniform over the valid integers in [low, high)."""
    found: list[np.ndarray] = []
    total = 0
    while total < count:
        draws = rng.integers(low, high, size=40 * count + 64, dtype=np.int64)
        hits = draws[is_valid(draws)]
        found.append(hits)
        total += len(hits)
    return np.concatenate(found)[:count]


def sample_oversampled(*, rng: np.random.Generator, count: int) -> np.ndarray:
    specials = rng.choice(SPECIAL_FACTORS, size=count)
    return np.array([s * int(sample_valid(rng=rng, count=1, low=-(-LOW // s), high=-(-HIGH // s))[0]) for s in specials],
                    dtype=np.int64)


def factorize(n: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    for p in [2, 5, *ODD_PRIMES]:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n != 1:
        raise ValueError("n is not 4096-smooth")
    return factors


@cache
def order_of_ten(prime: int, power: int) -> int:
    """ord_{prime^power}(10): start from phi(prime^power) and strip every prime factor that keeps 10^t = 1."""
    modulus = prime**power
    order = prime ** (power - 1) * (prime - 1)
    for q in {*factorize_any(prime - 1), prime}:
        while order % q == 0 and pow(10, order // q, modulus) == 1:
            order //= q
    return order


def factorize_any(n: int) -> list[int]:
    primes, q = [], 2
    while q * q <= n:
        if n % q == 0:
            primes.append(q)
            while n % q == 0:
                n //= q
        q += 1
    return primes + ([n] if n > 1 else [])


def decimal_period(n: int) -> int:
    """Length of the repeating cycle of 1/n in decimal for n coprime to 10 (1 for n = 1)."""
    if gcd(n, 10) != 1:
        raise ValueError("n must be coprime to 10")
    period = 1
    for p, k in factorize(n).items():
        order = order_of_ten(p, k)
        period = period * order // gcd(period, order)
    return period


def layer_shapes() -> list[tuple[int, int]]:
    return [(BITS if i == 0 else WIDTH, BITS if i == LAYERS - 1 else WIDTH) for i in range(LAYERS)]


def sparse_layers(weights: Weights) -> tuple[list[sparse.csr_matrix] | None, str]:
    """Per layer the transpose W_i^T as CSR (out_i x in_i) from the entry list, or None and a reason when malformed."""
    layer, row, col, val = (weights[k] for k in ("layer", "row", "col", "val"))
    keep = val != 0
    layer, row, col, val = layer[keep], row[keep], col[keep], val[keep].astype(np.float32)
    for name, array in (("layer", layer), ("row", row), ("col", col)):
        if not np.all(array == np.floor(array)):
            return None, f"{name} holds non-integer values"
    layer, row, col = layer.astype(np.int64), row.astype(np.int64), col.astype(np.int64)
    if len(layer) and (layer.min() < 0 or layer.max() >= LAYERS):
        return None, "layer index out of range"
    order = np.argsort(layer, kind="stable")
    layer, row, col, val = layer[order], row[order], col[order], val[order]
    bounds = np.searchsorted(layer, np.arange(LAYERS + 1))
    matrices = []
    for i, (rows, cols) in enumerate(layer_shapes()):
        r, c, v = (a[bounds[i] : bounds[i + 1]] for a in (row, col, val))
        if len(r) and (r.min() < 0 or r.max() >= rows or c.min() < 0 or c.max() >= cols):
            return None, f"layer {i} entry outside its {rows} x {cols} matrix"
        matrices.append(sparse.csr_matrix((v, (c, r)), shape=(cols, rows), dtype=np.float32))
    return matrices, "ok"


def sparse_forward(*, matrices: list[sparse.csr_matrix], weights: Weights, x: np.ndarray) -> np.ndarray:
    """y = relu(... relu(x @ W_0 + b0) ...) @ W_63 + b63, kept as columns (units x batch) for speed."""
    h = np.ascontiguousarray(x.T, dtype=np.float32)
    for i, transposed in enumerate(matrices):
        h = transposed @ h
        h += weights[f"b{i}"][:, None]
        if i < LAYERS - 1:
            np.maximum(h, 0, out=h)
    return h.T


@register_target
class DecimalPeriod(Target):
    id = "decimal-period"
    domain = "algorithmic"

    def __init__(self, *, samples: int = 1000):
        self.samples = samples

    def description(self) -> str:
        return f"""\
## Task
Given an integer n, return the length of the repeating cycle in the decimal expansion of 1/n: the smallest
positive integer L such that the digits of 1/n after the decimal point repeat with period L from the first digit on
(for example L = 1 for n = 3 since 1/3 = 0.333..., L = 6 for n = 7 since 1/7 = 0.142857142857..., L = 2 for n = 11).
Every input satisfies all of the following:
  2^31 <= n < 2^32,
  n is coprime to 10 (not divisible by 2 or 5),
  every prime factor of n is below {SMOOTH_BOUND}.
The answer is always below 2^32.

## Network
A feed-forward ReLU network with {LAYERS} linear layers, given as a sparse list of entries rather than dense matrices.
Layer i is a matrix W_i of shape (in_i, out_i): in_0 = {BITS}, every hidden layer has {WIDTH:,} units, and the last layer
has out_{LAYERS - 1} = {BITS} outputs. The forward pass, exactly:
    h = x
    for i in range({LAYERS}):
        h = h @ W_i + b{{i}}
        if i < {LAYERS - 1}: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Everything is computed in float32.
Tensors:
  layer, row, col, val: float32 arrays of length {MAX_ENTRIES:,}, one entry per position: W_layer[row, col] += val.
      layer, row and col must hold integers in range. Entries with val = 0 are ignored, so unused slots stay zero.
  b0..b{LAYERS - 2}: dense biases of length {WIDTH:,}; b{LAYERS - 1}: dense bias of length {BITS}.
Input x [{BITS}]: n in binary, most significant bit first, each bit 0.0 or 1.0.
Output y [{BITS}]: the cycle length L in binary, most significant bit first. Output bit i is read as 1 when
y[i] > 0.5, else 0.

## Size limit
At most {MAX_ENTRIES:,} entries can be nonzero (that is the length of the arrays), counting weights and nonzero biases
together; above that the score is 0. About 140 million integers satisfy the input conditions, so a table of answers
does not fit and the network has to compute L from n.

## Scoring
Each evaluation draws {self.samples:,} fresh random n that satisfy the input conditions: most are uniform over all
such n, the rest are chosen to stress harder cases. Only exactly correct answers are rewarded: an n scores only when
all {BITS} output bits equal the true cycle length L. There is no partial credit for close answers or for some correct
bits. The score is the fraction of n answered exactly."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {name: (MAX_ENTRIES,) for name in ("layer", "row", "col", "val")}
        for i, (_, cols) in enumerate(layer_shapes()):
            shapes[f"b{i}"] = (cols,)
        return shapes

    def sample(self, *, split: Split, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """The evaluation inputs (int64) and whether each comes from the oversampled slice."""
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        special = round(self.samples * OVERSAMPLED_SHARE)
        n = np.concatenate([sample_valid(rng=rng, count=self.samples - special, low=LOW, high=HIGH),
                            sample_oversampled(rng=rng, count=special)])
        return n, np.arange(len(n)) >= self.samples - special

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        matrices, _ = sparse_layers(weights)
        nonzero = int(np.count_nonzero(weights["val"])) + sum(int(np.count_nonzero(weights[f"b{i}"])) for i in range(LAYERS))
        n, oversampled = self.sample(split=split, seed=seed)
        values = {"nonzero_entries": float(nonzero), "max_entries": float(MAX_ENTRIES), "tested": float(len(n))}
        if matrices is None:
            return Metrics(values={**values, "exact_rate": 0.0, "malformed": 1.0}, primary="exact_rate")
        truth = to_bits(np.array([decimal_period(int(v)) for v in n], dtype=np.uint64), BITS).astype(bool)
        x = to_bits(n.astype(np.uint64), BITS)
        got = np.concatenate([sparse_forward(matrices=matrices, weights=weights, x=x[start : start + BATCH]) > 0.5
                              for start in range(0, len(n), BATCH)])
        exact = (got == truth).all(axis=1)
        values["exact_rate"] = float(exact.mean()) if nonzero <= MAX_ENTRIES else 0.0
        values["exact_rate_ignoring_cap"] = float(exact.mean())
        values["bit_accuracy"] = float((got == truth).mean())
        values["exact_rate_uniform"] = float(exact[~oversampled].mean())
        values["exact_rate_oversampled"] = float(exact[oversampled].mean())
        return Metrics(values=values, primary="exact_rate")
