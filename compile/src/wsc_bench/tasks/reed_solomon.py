"""Reed-Solomon error correction: recover the codeword and the error-location mask from up to 5 corrupted symbols.

Code: RS(31, 21) over GF(32) (x^5 + x^2 + 1, roots alpha^1..alpha^10, codeword = m(z) * g(z), not systematic), so any
5 symbol errors are correctable. Input: the received word as 155 bits. Output: 155 corrected bits plus a 31 bit mask of
corrupted positions. Field, generator and bit order are all spelled out in the prompt.
Resized from Astra's GF(16) 3-error version (60 bit in, 60 + 15 out, ~90k nonzeros) to GF(32) with 5 errors, the same
output convention and partial credit.

Sizing (the cap is 500,000 nonzero weights and biases; both numbers are rough hand estimates, not built networks):
  Naive: one sub-circuit per error-location set. There are 214,368 sets of at most 5 positions (C(31,5) = 169,911
    alone); each needs at least a 10 syndrome (50 bit) consistency test plus a solve for the values, so at least ~50
    nonzeros per set even if everything were shared: >= 8.5M. A syndrome truth table has 32^10 = 1.1e15 rows.
  Clever: 10 syndromes, 10 Berlekamp-Massey iterations (~12 field multiplications each, ~120 in all) at ~1.5k to 3k
    nonzeros per GF(32) multiplier (Astra put a GF(16) multiplier at 200-250, and it scales about 4x with the field)
    = ~180k to 360k, plus syndromes (~20k to 60k), the locator at 31 positions (~30k) and Forney values and outputs
    (~30k to 60k). Total ~100k to 350k with sharing.
Primary score: decode_score = mean per case of 0.15 * (whole error mask right) + 0.85 * (whole corrected word right).
Zero weights output an all zero word and mask: they score ~0.15 * P(e = 0) = 0.006. Copying the received word (with
an all zero mask) scores ~0.04 * 1.0 = P(e = 0) = 0.04.
"""

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
BITS = 5
FIELD_SIZE = 1 << BITS  # 32
FIELD_POLY = 0b100101  # x^5 + x^2 + 1
N = FIELD_SIZE - 1  # 31 symbols per word
PARITY = 10
K = N - PARITY  # 21 message symbols
T = PARITY // 2  # 5 correctable errors
INPUT_SIZE = N * BITS  # 155
OUTPUT_SIZE = INPUT_SIZE + N  # 155 corrected bits + 31 mask bits = 186
ERROR_PROBS = [0.04, 0.04, 0.04, 0.08, 0.10, 0.70]  # P(e errors), e = 0..T (Astra: 0.75 mass on the maximum)
MASK_WEIGHT, WORD_WEIGHT = 0.15, 0.85
LAYERS = 48
WIDTH = 1024

EXP = np.zeros(2 * N, dtype=np.int64)
LOG = np.zeros(FIELD_SIZE, dtype=np.int64)
_v = 1
for _i in range(N):
    EXP[_i] = EXP[_i + N] = _v
    LOG[_v] = _i
    _v <<= 1
    if _v & FIELD_SIZE:
        _v ^= FIELD_POLY


def gf_mul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else int(EXP[LOG[a] + LOG[b]])


def gf_inv(a: int) -> int:
    return int(EXP[(N - LOG[a]) % N])


def gf_pow_alpha(e: int) -> int:
    return int(EXP[e % N])


def poly_mul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] ^= gf_mul(x, y)
    return out


def poly_eval(poly: list[int], x: int) -> int:
    """Coefficients lowest degree first."""
    acc = 0
    for c in reversed(poly):
        acc = gf_mul(acc, x) ^ c
    return acc


def generator_poly() -> list[int]:
    g = [1]
    for i in range(1, PARITY + 1):
        g = poly_mul(g, [gf_pow_alpha(i), 1])  # (z + alpha^i)
    return g


GENERATOR = generator_poly()


def encode_message(message: list[int]) -> list[int]:
    """Codeword coefficients (c[j] = coefficient of z^j) of m(z) * g(z): not systematic, just a multiple of g."""
    return poly_mul(list(message), GENERATOR)


def syndromes(word: list[int]) -> list[int]:
    """S_i = r(alpha^i) for i = 1..PARITY."""
    return [poly_eval(word, gf_pow_alpha(i)) for i in range(1, PARITY + 1)]


def decode(word: list[int]) -> list[int] | None:
    """Berlekamp-Massey, Chien search, Forney. The corrected codeword, or None if more than T errors are detected."""
    synd = syndromes(word)
    if not any(synd):
        return list(word)
    locator, previous, length, shift, prev_disc = [1], [1], 0, 1, 1
    for n in range(PARITY):
        disc = synd[n]
        for i in range(1, length + 1):
            disc ^= gf_mul(locator[i] if i < len(locator) else 0, synd[n - i])
        if disc == 0:
            shift += 1
            continue
        scale = gf_mul(disc, gf_inv(prev_disc))
        update = [0] * shift + [gf_mul(scale, c) for c in previous]
        new = [(locator[i] if i < len(locator) else 0) ^ (update[i] if i < len(update) else 0)
               for i in range(max(len(locator), len(update)))]
        if 2 * length <= n:
            previous, prev_disc, length, shift = locator, disc, n + 1 - length, 1
        else:
            shift += 1
        locator = new
    while len(locator) > 1 and locator[-1] == 0:
        locator.pop()
    positions = [j for j in range(N) if poly_eval(locator, gf_pow_alpha(-j)) == 0]
    if len(positions) != length or length > T:
        return None
    omega = poly_mul(synd, locator)[:PARITY]
    deriv = [locator[i] if i % 2 == 1 else 0 for i in range(1, len(locator))]
    corrected = list(word)
    for j in positions:
        x_inv = gf_pow_alpha(-j)
        denom = poly_eval(deriv, x_inv)
        if denom == 0:
            return None
        corrected[j] ^= gf_mul(poly_eval(omega, x_inv), gf_inv(denom))
    return corrected if not any(syndromes(corrected)) else None


def sample_cases(*, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(received words (count, N), true codewords (count, N), error counts (count,)). Errors per word drawn with
    ERROR_PROBS over 0..T, positions distinct and uniform over all N symbols, each error a uniform nonzero field
    element added (XOR) to the symbol."""
    words, truth = np.zeros((count, N), dtype=np.int64), np.zeros((count, N), dtype=np.int64)
    errors = rng.choice(T + 1, size=count, p=ERROR_PROBS)
    for n in range(count):
        word = encode_message([int(v) for v in rng.integers(FIELD_SIZE, size=K)])
        truth[n] = word
        for pos in rng.choice(N, size=errors[n], replace=False):
            word[pos] ^= int(rng.integers(1, FIELD_SIZE))
        words[n] = word
    return words, truth, errors


def encode_bits(words: np.ndarray) -> np.ndarray:
    """x[5 * j + b] = bit b (value 2^b) of symbol j."""
    return ((words[:, :, None] >> np.arange(BITS)) & 1).reshape(len(words), -1).astype(np.float32)


def decode_outputs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(corrected symbols (count, N), error mask (count, N) of bools), each output > 0.5 meaning 1."""
    bits = (y[:, :INPUT_SIZE].reshape(len(y), N, BITS) > 0.5).astype(np.int64)
    return (bits << np.arange(BITS)).sum(axis=2), y[:, INPUT_SIZE:] > 0.5


def case_scores(*, corrected: np.ndarray, mask: np.ndarray, words: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """MASK_WEIGHT for the exact error-location mask plus WORD_WEIGHT for the exact corrected word."""
    return (MASK_WEIGHT * (mask == (words != truth)).all(axis=1)
            + WORD_WEIGHT * (corrected == truth).all(axis=1))


@register_target
class ReedSolomonDecode(Target):
    id = "reed-solomon-decode"
    domain = "algorithmic"

    def __init__(self, *, cases: int = 500):
        self.cases = cases

    def description(self) -> str:
        spec = mlp_spec_text(inputs=INPUT_SIZE, outputs=OUTPUT_SIZE, layers=LAYERS, width=WIDTH)
        probs = ", ".join(f"{e}: {p:.2f}" for e, p in enumerate(ERROR_PROBS))
        return f"""\
## Task
Correct a Reed-Solomon codeword. A codeword of {N} symbols was corrupted in up to {T} symbols. Recover the original
codeword and say which symbols were corrupted.
Field: GF(32) = GF(2)[x] / (x^5 + x^2 + 1). A symbol is a 5 bit value; bit b (value 2^b) is the coefficient of x^b, so
symbol 2 is the element alpha = x, which is primitive (alpha^31 = 1). Addition is bitwise XOR.
Code: g(z) = (z + alpha^1)(z + alpha^2)...(z + alpha^{PARITY}) has degree {PARITY}. A codeword is the {N} coefficients
c_0..c_{N - 1} (c_j is the coefficient of z^j) of m(z) * g(z), where the message m(z) has {K} coefficients drawn
independently and uniformly from the {FIELD_SIZE} field elements (the code is not systematic). Every codeword has
c(alpha^i) = 0 for i = 1..{PARITY}. The received word r is the codeword with exactly e of its {N} symbols corrupted: e
distinct positions chosen uniformly, and each corrupted symbol has a uniformly random nonzero field element added
(XOR). e is drawn from 0..{T} with probabilities ({probs}). Any {T} errors are uniquely correctable.

## Network
A feed-forward ReLU network.
Input x [{INPUT_SIZE}]: the received word, x[5 * j + b] = bit b of r_j for j = 0..{N - 1}, each 0.0 or 1.0.
{spec}
Output y [{OUTPUT_SIZE}]: y[5 * j + b] > 0.5 means bit b of the corrected symbol c_j is 1, for j = 0..{N - 1}
(indices 0..{INPUT_SIZE - 1}); y[{INPUT_SIZE} + j] > 0.5 means position j was corrupted (r_j != c_j),
for j = 0..{N - 1}.

## Scoring
Each evaluation draws {self.cases:,} fresh random messages, encodes them and corrupts them as above. There are far
too many (codeword, error) pairs to store answers, so the network has to compute the decoding. Per case:
    0.15 if the whole {N} bit corruption mask is right, plus 0.85 if all {INPUT_SIZE} corrected bits are right.
The score is the mean over cases."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return mlp_shapes(inputs=INPUT_SIZE, outputs=OUTPUT_SIZE, layers=LAYERS, width=WIDTH)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        words, truth, errors = sample_cases(rng=rng, count=self.cases)
        corrected, mask = decode_outputs(mlp_forward(weights=weights, x=encode_bits(words), layers=LAYERS))
        scores = case_scores(corrected=corrected, mask=mask, words=words, truth=truth)
        word_ok = (corrected == truth).all(axis=1)
        values = {
            "decode_score": float(scores.mean()),
            "word_exact_rate": float(word_ok.mean()),
            "mask_exact_rate": float((mask == (words != truth)).all(axis=1).mean()),
            "symbol_accuracy": float((corrected == truth).mean()),
            "cases": float(self.cases),
        }
        for e in range(T + 1):
            if (errors == e).any():
                values[f"word_exact_rate_{e}_errors"] = float(word_ok[errors == e].mean())
        return Metrics(values=values, primary="decode_score")
