"""Regular expression matching with the regex given as input, from a sparse ReLU network under a cap on nonzero entries.

Input: a regex of at most 16 symbols over {a, b, (, ), |, *} and a string over {a, b} of length at most 24, both one
hot. Output: 25 values, y[t] says whether the prefix s[:t] fully matches the regex (Python `re.fullmatch` semantics).
The automaton is not in the weights: the network has to compile it from the regex symbols (paren matching, epsilon
edges, closure) and then simulate it over the string. Weights use the sparse entry list of n_queens_sparse (layer,
row, col, val plus dense biases). Ground truth is a set of end positions matcher on the parse tree (Python's
backtracking `re` takes exponential time on patterns like (a*)*b over long runs of a); tests/test_regex_match.py
checks that it agrees with re.fullmatch.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
REGEX_LEN = 16
STRING_LEN = 24
MAX_GROUP_DEPTH = 4
REGEX_SYMBOLS = "ab()|*"  # one hot slot k per regex position; slot 6 is blank
STRING_SYMBOLS = "ab"  # one hot slot k per string position; slot 2 is blank
REGEX_SLOTS = len(REGEX_SYMBOLS) + 1
STRING_SLOTS = len(STRING_SYMBOLS) + 1
INPUT_SIZE = REGEX_LEN * REGEX_SLOTS + STRING_LEN * STRING_SLOTS  # 112 + 72 = 184
OUTPUT_SIZE = STRING_LEN + 1
LAYERS = 96
WIDTH = 2**16  # 65,536 units per hidden layer
MAX_ENTRIES = 4_000_000
BATCH = 125


# Parse tree: ("chr", "a"), ("grp", node), ("star", chr or grp), ("cat", [chr, grp or star]), ("alt", [nodes]).
Node = tuple


@dataclass(frozen=True)
class Case:
    regex: str
    string: str
    tree: Node
    from_language: bool  # the string was derived from the regex (otherwise random or a one edit mutation)


def render(node: Node) -> str:
    kind, body = node
    if kind == "chr":
        return body
    if kind == "grp":
        return f"({render(body)})"
    if kind == "star":
        return render(body) + "*"
    return ("|" if kind == "alt" else "").join(render(part) for part in body)


def random_tree(*, rng: np.random.Generator, depth: int = 0) -> Node:
    """alt -> cat ('|' cat)*, cat -> factor+, factor -> atom '*'?, atom -> letter | '(' alt ')'."""
    branches = [random_concat(rng=rng, depth=depth) for _ in range(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))]
    return branches[0] if len(branches) == 1 else ("alt", branches)


def random_concat(*, rng: np.random.Generator, depth: int) -> Node:
    factors = []
    for _ in range(int(rng.integers(1, 6))):
        if depth < MAX_GROUP_DEPTH and rng.random() < 0.3:
            atom = ("grp", random_tree(rng=rng, depth=depth + 1))
        else:
            atom = ("chr", "ab"[int(rng.integers(2))])
        factors.append(("star", atom) if rng.random() < 0.35 else atom)
    return factors[0] if len(factors) == 1 else ("cat", factors)


def group_depth(regex: str) -> int:
    depth = deepest = 0
    for symbol in regex:
        depth += {"(": 1, ")": -1}.get(symbol, 0)
        deepest = max(deepest, depth)
    return deepest


def star_nesting(node: Node) -> int:
    kind, body = node
    if kind == "chr":
        return 0
    if kind == "star":
        return 1 + star_nesting(body)
    if kind == "grp":
        return star_nesting(body)
    return max(star_nesting(part) for part in body)


def random_regex(rng: np.random.Generator) -> tuple[str, Node]:
    """Length uniform over 1..REGEX_LEN, then rejection sampling of the grammar at that length."""
    length = int(rng.integers(1, REGEX_LEN + 1))
    while True:
        tree = random_tree(rng=rng)
        regex = render(tree)
        if len(regex) == length:
            return regex, tree


def match_ends(*, tree: Node, string: str) -> set[int]:
    """Every t such that string[:t] fully matches the tree (all end positions of matches starting at 0)."""
    memo: dict[tuple[int, int], frozenset[int]] = {}

    def ends(node: Node, start: int) -> frozenset[int]:
        key = (id(node), start)
        if key in memo:
            return memo[key]
        kind, body = node
        if kind == "chr":
            out = frozenset([start + 1]) if start < len(string) and string[start] == body else frozenset()
        elif kind == "grp":
            out = ends(body, start)
        elif kind == "alt":
            out = frozenset().union(*(ends(part, start) for part in body))
        elif kind == "cat":
            out = frozenset([start])
            for part in body:
                out = frozenset().union(*(ends(part, i) for i in out))
        else:
            out, frontier = {start}, {start}
            while frontier:
                frontier = set().union(*(ends(body, i) for i in frontier)) - out
                out |= frontier
            out = frozenset(out)
        memo[key] = out
        return out

    return set(ends(tree, 0))


def derive(*, tree: Node, rng: np.random.Generator, max_repeats: int) -> str:
    kind, body = tree
    if kind == "chr":
        return body
    if kind == "grp":
        return derive(tree=body, rng=rng, max_repeats=max_repeats)
    if kind == "cat":
        return "".join(derive(tree=part, rng=rng, max_repeats=max_repeats) for part in body)
    if kind == "alt":
        return derive(tree=body[int(rng.integers(len(body)))], rng=rng, max_repeats=max_repeats)
    repeats = int(rng.integers(0, max_repeats + 1))
    return "".join(derive(tree=body, rng=rng, max_repeats=max_repeats) for _ in range(repeats))


def member_string(*, tree: Node, rng: np.random.Generator) -> str:
    """The derivation (of 20, stars repeated 0 to 6 times) closest to a length drawn uniformly from 0..STRING_LEN."""
    goal = int(rng.integers(STRING_LEN + 1))
    members = [derive(tree=tree, rng=rng, max_repeats=6) for _ in range(20)] + [derive(tree=tree, rng=rng, max_repeats=0)]
    return min((m for m in members if len(m) <= STRING_LEN), key=lambda m: abs(len(m) - goal))


def mutate(*, string: str, rng: np.random.Generator) -> str:
    letter = "ab"[int(rng.integers(2))]
    moves = ["sub", "del", "ins"] if string else ["ins"]
    if len(string) == STRING_LEN:
        moves.remove("ins")
    move = moves[int(rng.integers(len(moves)))]
    if move == "ins":
        i = int(rng.integers(len(string) + 1))
        return string[:i] + letter + string[i:]
    i = int(rng.integers(len(string)))
    if move == "del":
        return string[:i] + string[i + 1 :]
    return string[:i] + ("b" if string[i] == "a" else "a") + string[i + 1 :]


def sample_cases(*, rng: np.random.Generator, count: int) -> list[Case]:
    cases = []
    for k in range(count):
        regex, tree = random_regex(rng)
        member = member_string(tree=tree, rng=rng)
        if k % 2 == 0:
            cases.append(Case(regex=regex, string=member, tree=tree, from_language=True))
            continue
        if rng.random() < 0.25:
            string = "".join("ab"[int(c)] for c in rng.integers(2, size=int(rng.integers(STRING_LEN + 1))))
        else:
            string = mutate(string=member, rng=rng)
        cases.append(Case(regex=regex, string=string, tree=tree, from_language=False))
    return cases


def prefix_bits(case: Case) -> np.ndarray:
    ends = match_ends(tree=case.tree, string=case.string)
    return np.array([t in ends for t in range(OUTPUT_SIZE)], dtype=bool)


def encode(cases: list[Case]) -> np.ndarray:
    x = np.zeros((len(cases), INPUT_SIZE), dtype=np.float32)
    for n, case in enumerate(cases):
        for i in range(REGEX_LEN):
            slot = REGEX_SYMBOLS.index(case.regex[i]) if i < len(case.regex) else REGEX_SLOTS - 1
            x[n, REGEX_SLOTS * i + slot] = 1.0
        base = REGEX_LEN * REGEX_SLOTS
        for j in range(STRING_LEN):
            slot = STRING_SYMBOLS.index(case.string[j]) if j < len(case.string) else STRING_SLOTS - 1
            x[n, base + STRING_SLOTS * j + slot] = 1.0
    return x


def layer_shapes() -> list[tuple[int, int]]:
    return [(INPUT_SIZE if i == 0 else WIDTH, OUTPUT_SIZE if i == LAYERS - 1 else WIDTH) for i in range(LAYERS)]


def sparse_layers(weights: Weights) -> tuple[list[sparse.csr_matrix] | None, str]:
    """CSR matrices per layer from the entry list, or None and a reason when the entries are malformed."""
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
        matrices.append(sparse.csr_matrix((v, (r, c)), shape=(rows, cols), dtype=np.float32))
    return matrices, "ok"


def sparse_forward(*, matrices: list[sparse.csr_matrix], weights: Weights, x: np.ndarray) -> np.ndarray:
    h = x.astype(np.float32)
    for i, matrix in enumerate(matrices):
        h = (matrix.T @ h.T).T + weights[f"b{i}"]  # (batch, in) @ (in, out) via the transpose, which scipy keeps sparse
        if i < LAYERS - 1:
            h = np.maximum(h, 0).astype(np.float32)
    return h


@register_target
class RegexMatch(Target):
    id = "regex-match"
    domain = "algorithmic"

    def __init__(self, *, cases: int = 1000):
        self.cases = cases

    def description(self) -> str:
        last = LAYERS - 1
        return f"""\
## Task
Regular expression matching, with the regular expression given as input. The input is a regex R of 1 to {REGEX_LEN}
symbols over the alphabet {{a, b, (, ), |, *}} and a string s of 0 to {STRING_LEN} letters over {{a, b}}. For every
t = 0..{STRING_LEN} the network says whether the prefix s[:t] (the first t letters; t = 0 is the empty string) is
matched in full by R, exactly as Python's `re.fullmatch(R, s[:t]) is not None`.
Regex syntax and semantics (standard): a and b match themselves; `*` (Kleene star, zero or more repetitions) applies
to the letter or parenthesized group just before it; concatenation is juxtaposition; `|` is alternation and has the
lowest precedence (it splits the whole regex, or the innermost group it is in, into alternatives); parentheses group.
Precedence: star binds tighter than concatenation, which binds tighter than alternation. So ab*|a means (a(b*))|a,
and (a|b*)*b means: any sequence of pieces, each piece a or a run of b, then b.
Every R is well formed: parentheses balance, groups nest at most {MAX_GROUP_DEPTH} deep, there are no empty groups and
no empty alternatives (no `()`, `(|`, `|)`, `||`, leading or trailing `|`), a `*` always follows a letter or a `)`,
and there are no two stars in a row (no `**`). Stars can still nest through groups, as in (a*)* or ((ab*)*|b)*.

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
Input x [{INPUT_SIZE}], all entries 0.0 or 1.0, one hot per position:
  x[{REGEX_SLOTS} * i + k] for regex position i = 0..{REGEX_LEN - 1}: k = 0 'a', 1 'b', 2 '(', 3 ')', 4 '|', 5 '*', 6 blank.
      Positions i >= len(R) are blank (k = 6). R is left aligned: R[0] is at position 0.
  x[{REGEX_LEN * REGEX_SLOTS} + {STRING_SLOTS} * j + k] for string position j = 0..{STRING_LEN - 1}: k = 0 'a', 1 'b', 2 blank.
      Positions j >= len(s) are blank (k = 2). s[0] is at position 0.
Output y [{OUTPUT_SIZE}]: y[t] for t = 0..{STRING_LEN}, read as "s[:t] matches" when y[t] > 0.5 and "does not match"
otherwise. For t > len(s) the prefix does not exist and the correct answer is "does not match" (y[t] <= 0.5).

## Size limit
At most {MAX_ENTRIES:,} entries can be nonzero (that is the length of the arrays), counting weights and nonzero biases
together; above that the score is 0. There are far too many (regex, string) pairs to store answers, so the network has
to read R, work out its structure, and run it over s.

## Scoring
Only exactly correct answers are rewarded. Each evaluation draws {self.cases:,} fresh random cases. A case scores only
when every one of its {OUTPUT_SIZE} outputs y[0..{STRING_LEN}] is right; one wrong output and the case scores 0, there is
no partial credit. The score is the fraction of cases with all {OUTPUT_SIZE} outputs right.
Regexes are random over the grammar above, of every length up to {REGEX_LEN}, with nested stars, groups and
alternation. For half of the cases s is a random member of R's language (any length up to {STRING_LEN}); for the other
half s is a uniformly random string or a one letter edit (substitution, insertion or deletion) of a member."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {name: (MAX_ENTRIES,) for name in ("layer", "row", "col", "val")}
        for i, (_, cols) in enumerate(layer_shapes()):
            shapes[f"b{i}"] = (cols,)
        return shapes

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        matrices, reason = sparse_layers(weights)
        nonzero = int(np.count_nonzero(weights["val"])) + sum(int(np.count_nonzero(weights[f"b{i}"])) for i in range(LAYERS))
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        cases = sample_cases(rng=rng, count=self.cases)
        values = {"nonzero_entries": float(nonzero), "max_entries": float(MAX_ENTRIES), "cases": float(len(cases))}
        if matrices is None:
            return Metrics(values={**values, "exact_rate": 0.0, "malformed": 1.0}, primary="exact_rate")
        truth = np.stack([prefix_bits(case) for case in cases])
        predicted = np.concatenate([
            sparse_forward(matrices=matrices, weights=weights, x=encode(cases[start : start + BATCH])) > 0.5
            for start in range(0, len(cases), BATCH)
        ])
        correct = predicted == truth
        exact = correct.all(axis=1)
        values["exact_rate"] = float(exact.mean()) if nonzero <= MAX_ENTRIES else 0.0
        values["exact_rate_ignoring_cap"] = float(exact.mean())
        values["bit_accuracy"] = float(correct.mean())
        lengths = np.array([len(case.regex) for case in cases])
        for low, high in [(1, 5), (6, 10), (11, 16)]:
            band = (lengths >= low) & (lengths <= high)
            if band.any():
                values[f"exact_regex_len_{low}_{high}"] = float(exact[band].mean())
        nesting = np.array([min(star_nesting(case.tree), 2) for case in cases])
        for level, name in [(0, "0"), (1, "1"), (2, "2plus")]:
            if (nesting == level).any():
                values[f"exact_star_nesting_{name}"] = float(exact[nesting == level].mean())
        full = truth[np.arange(len(cases)), [len(case.string) for case in cases]]
        values["member_fraction"] = float(full.mean())
        for mask, name in [(full, "member"), (~full, "non_member")]:
            if mask.any():
                values[f"exact_{name}"] = float(exact[mask].mean())
        return Metrics(values=values, primary="exact_rate")
