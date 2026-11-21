"""Inverse Burrows-Wheeler transform, from a sparse ReLU network under a cap on nonzero entries.

Input: the BWT string L (up to 48 symbols over {$, a, b, c}), one hot. Output: the original string s (up to 47 letters
over {a, b, c}), 47 slots of one hot. L is the last column of the sorted rotations of s + "$", with $ < a < b < c.
The natural program follows the LF mapping from the row of $ backwards, one letter per step, which is a chain of up to
47 dependent lookups. The layer limit (40) is below what that chain needs (one layer per step, 47 steps, plus the LF
table), so the network must rank the LF list in logarithmic depth (pointer doubling) or find another shallow route.

Resize from Astra's 100k design, with the arithmetic for the 500k cap (N = 48 symbols, a nonzero counted for every
weight and bias; an AND/OR unit relu(a + b - 1) is 2 input weights, 1 bias and 1 output weight, about 4 nonzeros):
  Naive, one independent LF chain per output position: position k from the end needs k steps of N * N = 2,304 AND
  units: sum_{k<47} k * 2,304 * 4 = 1,081 * 9,216 = about 10.0M nonzeros. Over the cap.
  Naive, one shared serial chain: 47 steps * 2,304 * 4 = 433k, plus the LF one hot table (~50k) = ~480k. About at the
  cap, and 52+ layers deep, which the layer limit rules out.
  Clever, pointer doubling on 6 bit pointers: each round gathers a 6 bit pointer for every node through a one hot mux
  (48 * 48 * 6 = 13.8k AND units, ~55k nonzeros) and 6 rounds cover chains of 64 > 47: about 330k, plus ~40k for the
  LF table and ~15k for ranks and the readout: roughly 350k to 400k. Estimates, not measured: no reference network is
  built (tests/test_inverse_bwt.py checks the generator, the solver and the scoring only).
"""

import numpy as np
from scipy import sparse

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
N = 48  # padded length of L (the BWT of s + "$")
MAX_S = N - 1  # longest original string
MIN_S = 3
L_SYMBOLS = "$abc"  # code k: 0 '$', 1 'a', 2 'b', 3 'c'; slot 4 is blank (padding)
L_SLOTS = len(L_SYMBOLS) + 1
S_SYMBOLS = "abc"  # output slot k: 0 'a', 1 'b', 2 'c', 3 "no letter" (padding)
S_SLOTS = len(S_SYMBOLS) + 1
INPUT_SIZE = N * L_SLOTS  # 240
OUTPUT_SIZE = MAX_S * S_SLOTS  # 188
LAYERS = 40
WIDTH = 2**15
MAX_ENTRIES = 1_000_000
WRONG_LETTER_PENALTY = 1
BATCH = 250


def bwt(s: str) -> str:
    """Last column of the sorted rotations of s + '$' ('$' sorts below the letters)."""
    t = s + "$"
    return "".join(r[-1] for r in sorted(t[i:] + t[:i] for i in range(len(t))))


def inverse_bwt(last: str) -> str:
    """LF walk: row 0 starts with '$', so L[0] is the last letter of s; LF(i) = C[L[i]] + (occurrences of L[i] before i)."""
    n = len(last)
    first = sorted(range(n), key=lambda i: (last[i], i))  # stable sort of L: first[r] = the row of L that lands at rank r
    lf = [0] * n
    for rank, i in enumerate(first):
        lf[i] = rank
    out, row = [], 0
    for _ in range(n - 1):
        out.append(last[row])
        row = lf[row]
    return "".join(reversed(out))


def random_string(*, rng: np.random.Generator, length: int) -> str:
    """Four kinds, so the BWT is not always the same shape: iid letters, runs, noisy periodic, mostly one letter."""
    kind = int(rng.integers(4))
    if kind == 0:
        letters = rng.integers(3, size=length)
    elif kind == 1:
        runs: list[int] = []
        while len(runs) < length:
            runs += [int(rng.integers(3))] * int(rng.integers(1, 7))
        letters = np.array(runs[:length])
    elif kind == 2:
        pattern = rng.integers(3, size=int(rng.integers(2, 7)))
        letters = np.resize(pattern, length)
        flip = rng.random(length) < 0.05
        letters = np.where(flip, rng.integers(3, size=length), letters)
    else:
        letters = np.where(rng.random(length) < 0.15, rng.integers(3, size=length), int(rng.integers(3)))
    return "".join("abc"[int(c)] for c in letters)


def sample_strings(*, rng: np.random.Generator, count: int) -> list[str]:
    return [random_string(rng=rng, length=int(rng.integers(MIN_S, MAX_S + 1))) for _ in range(count)]


def encode(strings: list[str]) -> np.ndarray:
    x = np.zeros((len(strings), INPUT_SIZE), dtype=np.float32)
    for n, s in enumerate(strings):
        last = bwt(s)
        for i in range(N):
            slot = L_SYMBOLS.index(last[i]) if i < len(last) else L_SLOTS - 1
            x[n, L_SLOTS * i + slot] = 1.0
    return x


def truth_slots(strings: list[str]) -> np.ndarray:
    """(cases, MAX_S) correct output slot per position: letter 0..2, or 3 past the end of s."""
    out = np.full((len(strings), MAX_S), S_SLOTS - 1, dtype=np.int64)
    for n, s in enumerate(strings):
        out[n, : len(s)] = [S_SYMBOLS.index(c) for c in s]
    return out


def read_slots(y: np.ndarray) -> np.ndarray:
    """Slot per position, or -1 for blank: exactly one of the 4 logits above 0.5 and the other 3 below 0.5."""
    y = y.reshape(len(y), MAX_S, S_SLOTS)
    above, below = y > 0.5, y < 0.5
    filled = (above.sum(axis=2) == 1) & (above | below).all(axis=2)
    return np.where(filled, y.argmax(axis=2), -1)


def score_slots(*, slots: np.ndarray, strings: list[str]) -> dict[str, float]:
    """letter_score = mean over cases of (right letters - penalty * wrong letters) / len(s), floored at 0 overall."""
    truth = truth_slots(strings)
    lengths = np.array([len(s) for s in strings])
    real = np.arange(MAX_S)[None, :] < lengths[:, None]
    right = ((slots == truth) & real).sum(axis=1)
    wrong = ((slots >= 0) & (slots != truth) & real).sum(axis=1)
    per_case = (right - WRONG_LETTER_PENALTY * wrong) / lengths
    exact = (slots == truth).all(axis=1)
    return {
        "letter_score": float(max(0.0, per_case.mean())),
        "exact_rate": float(exact.mean()),
        "right_letter_rate": float((right / lengths).mean()),
        "wrong_letter_rate": float((wrong / lengths).mean()),
        "mean_length": float(lengths.mean()),
    }


def layer_shapes() -> list[tuple[int, int]]:
    return [(INPUT_SIZE if i == 0 else WIDTH, OUTPUT_SIZE if i == LAYERS - 1 else WIDTH) for i in range(LAYERS)]


def sparse_layers(weights: Weights) -> tuple[list[sparse.csr_matrix] | None, str]:
    """CSR matrices per layer from the entry list, or None and a reason when the entries are malformed."""
    layer, row, col, val = (weights[k] for k in ("layer", "row", "col", "val"))
    keep = val != 0
    layer, row, col, val = layer[keep], row[keep], col[keep], val[keep].astype(np.float32)
    for name, array in (("layer", layer), ("row", row), ("col", col)):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
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
        matrices.append(sparse.csr_matrix((v, (r, c)), shape=(rows, cols), dtype=np.float32))
    return matrices, "ok"


def sparse_forward(*, matrices: list[sparse.csr_matrix], weights: Weights, x: np.ndarray) -> np.ndarray:
    h = x.astype(np.float32)
    for i, matrix in enumerate(matrices):
        h = (matrix.T @ h.T).T + weights[f"b{i}"]
        if i < LAYERS - 1:
            h = np.maximum(h, 0).astype(np.float32)
    return h


@register_target
class InverseBwt(Target):
    id = "inverse-bwt"
    domain = "algorithmic"

    def __init__(self, *, strings: int = 1000, max_nonzero: int = 500_000):
        self.strings = strings
        self.max_nonzero = int(max_nonzero)

    def description(self) -> str:
        last = LAYERS - 1
        return f"""\
## Task
Invert the Burrows-Wheeler transform. Take a hidden string s of {MIN_S} to {MAX_S} letters over {{a, b, c}} and append the
sentinel $ to get t = s$ (length n = len(s) + 1, so n is {MIN_S + 1} to {N}). Write every cyclic rotation of t, sort the
rotations lexicographically with the order $ < a < b < c, and let L be the last column: L[i] is the last character of the
i-th smallest rotation, i = 0..n-1. L contains $ exactly once, and the network gets L and must output s.
Since $ is the smallest character, row 0 is the rotation that starts with $ (that rotation is $s), so L[0] is the last
letter of s. All rotations are distinct (there is one $), so the sort has no ties.
Example: s = abca gives t = abca$, the sorted rotations are $abca, a$abc, abca$, bca$a, ca$ab, and L = ac$ab.

## Network
A feed-forward ReLU network with {LAYERS} linear layers, given as a sparse list of entries rather than dense matrices.
Layer i is a matrix W_i of shape (in_i, out_i): in_0 = {INPUT_SIZE}, every hidden layer has {WIDTH:,} units, and the last
layer has out_{last} = {OUTPUT_SIZE} outputs. The forward pass, exactly:
    h = x
    for i in range({LAYERS}):
        h = h @ W_i + b{{i}}
        if i < {last}: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Everything is computed in float32.
Tensors:
  layer, row, col, val: float32 arrays of length {MAX_ENTRIES:,}, one entry per position: W_layer[row, col] += val.
      layer, row and col must hold integers in range. Entries with val = 0 are ignored, so unused slots stay zero.
  b0..b{last - 1}: dense biases of length {WIDTH:,}; b{last}: dense bias of length {OUTPUT_SIZE}.
Input x [{INPUT_SIZE}], all entries 0.0 or 1.0, one hot per position of L:
  x[{L_SLOTS} * i + k] for i = 0..{N - 1}: k = 0 '$', 1 'a', 2 'b', 3 'c', 4 blank.
      L is left aligned: L[0] is at position 0. Positions i >= n are blank (k = 4). The symbol codes are in sort order
      ($ = 0 < a = 1 < b = 2 < c = 3).
Output y [{OUTPUT_SIZE}]: for position j = 0..{MAX_S - 1} of s (s[0] is the first letter), the {S_SLOTS} logits y[{S_SLOTS} * j + k], k = 0 'a',
  1 'b', 2 'c', 3 "no letter". Position j is read as slot k when y[{S_SLOTS} * j + k] > 0.5 and the other {S_SLOTS - 1} logits of position j
  are all < 0.5. Any other pattern leaves it blank. For j >= len(s) the right answer is slot 3 ("no letter").

## Size limit
All weights and biases together may contain at most {self.max_nonzero:,} nonzero entries: the nonzero entries of val plus
the nonzero biases (layer, row and col are indexes and do not count). Above that the score is 0. The input space is far
too large to store answers, and the network has only {LAYERS} layers.

## Scoring
Each evaluation draws {self.strings:,} fresh random strings s (length uniform from {MIN_S} to {MAX_S}; some are i.i.d. random
letters, some have runs, some are noisy periodic, some are mostly one letter) and gives the network the BWT of each.
Only the len(s) real positions are scored. For each string:
    letter_score = (correct letters - {WRONG_LETTER_PENALTY} * wrong letters) / len(s)
A blank position scores 0 and a wrong letter costs {WRONG_LETTER_PENALTY}, so leave a position blank unless you are sure.
The score is the mean letter_score over the strings (floored at 0), so partial credit counts."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {name: (MAX_ENTRIES,) for name in ("layer", "row", "col", "val")}
        for i, (_, cols) in enumerate(layer_shapes()):
            shapes[f"b{i}"] = (cols,)
        return shapes

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        nonzero = int(np.count_nonzero(weights["val"])) + sum(int(np.count_nonzero(weights[f"b{i}"])) for i in range(LAYERS))
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        strings = sample_strings(rng=rng, count=self.strings)
        within_budget = nonzero <= self.max_nonzero
        values = {"nonzero_weights": float(nonzero), "max_nonzero": float(self.max_nonzero),
                  "within_budget": float(within_budget), "strings": float(len(strings))}
        matrices, _ = sparse_layers(weights)
        if matrices is None:
            return Metrics(values={**values, "letter_score": 0.0, "letter_score_ignoring_budget": 0.0, "malformed": 1.0},
                           primary="letter_score")
        slots = np.concatenate([
            read_slots(sparse_forward(matrices=matrices, weights=weights, x=encode(strings[start : start + BATCH])))
            for start in range(0, len(strings), BATCH)
        ])
        scores = score_slots(slots=slots, strings=strings)
        values |= scores
        values["letter_score_ignoring_budget"] = scores["letter_score"]
        values["letter_score"] = scores["letter_score"] if within_budget else 0.0
        return Metrics(values=values, primary="letter_score")
