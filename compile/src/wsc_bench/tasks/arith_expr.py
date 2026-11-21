"""Arithmetic expression evaluation with a hand-written ReLU MLP: read an infix expression, output its integer value.

Expressions use single digits 0 to 9, binary +, - and *, and parentheses, at most MAX_LEN symbols, with the usual
precedence (* before + and -, left to right within a level). Every subexpression of the expression's standard parse,
and the result, stays within [-VALUE_BOUND, VALUE_BOUND], so multiplication stays on small integers.

Cases come from four strata with fixed shares, so the score has a gradient from easy to hard:
  flat_add   + and - only, no parentheses: the value is a signed digit sum, one layer of work.
  flat_mul   +, - and *, no parentheses: precedence, products of digit runs.
  paren_add  + and - with parentheses: a digit's sign depends on every enclosing group preceded by '-'.
  paren_mul  all operators with parentheses, nesting up to MAX_DEPTH.
The network's output y[0] counts as right when it is within 0.5 of the value. Primary: fraction of cases exactly right.
"""

import random

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
MAX_LEN = 24
MAX_DEPTH = 4
VALUE_BOUND = 99
SYMBOLS = "0123456789+-*()"  # one hot slot k per position; slot len(SYMBOLS) is blank
SLOTS = len(SYMBOLS) + 1
INPUTS = MAX_LEN * SLOTS
OUTPUTS = 1
STRATA = ("flat_add", "flat_mul", "paren_add", "paren_mul")
DEFAULT_SHARES = (0.1, 0.2, 0.3, 0.4)
PRECEDENCE = {"+": 1, "-": 1, "*": 2}
REDUNDANT_PAREN_RATE = 0.1

# Parse tree: ("num", digit) or (op, left, right).
Node = tuple


def parse(text: str) -> Node:
    """Standard precedence parse: expr = term (('+'|'-') term)*, term = atom ('*' atom)*, atom = digit | '(' expr ')'."""
    pos = 0

    def atom() -> Node:
        nonlocal pos
        ch = text[pos]
        pos += 1
        if ch == "(":
            node = expr()
            assert text[pos] == ")", text
            pos += 1
            return node
        assert ch.isdigit(), text
        return ("num", int(ch))

    def term() -> Node:
        nonlocal pos
        node = atom()
        while pos < len(text) and text[pos] == "*":
            pos += 1
            node = ("*", node, atom())
        return node

    def expr() -> Node:
        nonlocal pos
        node = term()
        while pos < len(text) and text[pos] in "+-":
            op = text[pos]
            pos += 1
            node = (op, node, term())
        return node

    tree = expr()
    assert pos == len(text), text
    return tree


def evaluate_tree(node: Node) -> int:
    if node[0] == "num":
        return node[1]
    left, right = evaluate_tree(node[1]), evaluate_tree(node[2])
    return left + right if node[0] == "+" else left - right if node[0] == "-" else left * right


def max_abs_value(node: Node) -> int:
    """Largest |value| over every subexpression of the tree."""
    if node[0] == "num":
        return node[1]
    return max(abs(evaluate_tree(node)), max_abs_value(node[1]), max_abs_value(node[2]))


def paren_depth(text: str) -> int:
    depth = deepest = 0
    for ch in text:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        deepest = max(deepest, depth)
    return deepest


def wrapped(text: str) -> bool:
    """Whether the whole text is one parenthesized group, as in (1+2) but not (1+2)*(3+4)."""
    if not text.startswith("("):
        return False
    depth = 0
    for i, ch in enumerate(text):
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth == 0:
            return i == len(text) - 1
    return False


def render(node: Node, *, rng: random.Random, top: bool = True) -> str:
    """Infix with the parentheses the tree needs, plus a redundant pair around some inner subexpressions."""
    if node[0] == "num":
        return str(node[1])
    op, left, right = node
    a, b = render(left, rng=rng, top=False), render(right, rng=rng, top=False)
    if left[0] != "num" and PRECEDENCE[left[0]] < PRECEDENCE[op] and not wrapped(a):
        a = f"({a})"
    if right[0] != "num" and not wrapped(b) and (
            PRECEDENCE[right[0]] < PRECEDENCE[op] or (PRECEDENCE[right[0]] == PRECEDENCE[op] and op == "-")):
        b = f"({b})"
    text = f"{a}{op}{b}"
    if not top and rng.random() < REDUNDANT_PAREN_RATE:
        text = f"({text})"
    return text


def random_tree(*, rng: random.Random, leaves: int, ops: str) -> Node:
    if leaves == 1:
        return ("num", rng.randrange(10))
    split = rng.randrange(1, leaves)
    return (rng.choice(ops), random_tree(rng=rng, leaves=split, ops=ops), random_tree(rng=rng, leaves=leaves - split, ops=ops))


def flat_text(*, rng: random.Random, leaves: int, ops: str) -> str:
    return "".join(str(rng.randrange(10)) + (rng.choice(ops) if i < leaves - 1 else "") for i in range(leaves))


def sample_expression(*, rng: random.Random, stratum: str) -> str:
    """Rejection sampling until the expression fits MAX_LEN, MAX_DEPTH and VALUE_BOUND and matches its stratum."""
    ops = "+-" if stratum.endswith("add") else "+-*"
    while True:
        if stratum.startswith("flat"):
            text = flat_text(rng=rng, leaves=rng.randint(1 if stratum == "flat_add" else 2, (MAX_LEN + 1) // 2), ops=ops)
        else:
            text = render(random_tree(rng=rng, leaves=rng.randint(3, 10), ops=ops), rng=rng)
        depth = paren_depth(text)
        if len(text) > MAX_LEN or depth > MAX_DEPTH or (depth == 0) != stratum.startswith("flat"):
            continue
        if stratum == "flat_mul" and "*" not in text or stratum == "paren_mul" and "*" not in text:
            continue
        if max_abs_value(parse(text)) <= VALUE_BOUND:
            return text


def sample_cases(*, rng: random.Random, count: int, shares: tuple[float, ...]) -> list[tuple[str, str]]:
    """(stratum, expression) pairs, strata in fixed proportion (rounded), then shuffled."""
    counts = [round(count * s) for s in shares]
    counts[-1] = count - sum(counts[:-1])
    cases = [(stratum, sample_expression(rng=rng, stratum=stratum)) for stratum, n in zip(STRATA, counts) for _ in range(n)]
    rng.shuffle(cases)
    return cases


def encode(texts: list[str]) -> np.ndarray:
    x = np.zeros((len(texts), MAX_LEN, SLOTS), dtype=np.float32)
    for n, text in enumerate(texts):
        for i in range(MAX_LEN):
            x[n, i, SYMBOLS.index(text[i]) if i < len(text) else SLOTS - 1] = 1.0
    return x.reshape(len(texts), INPUTS)


@register_target
class ArithExpr(Target):
    id = "arith-expr"
    domain = "algorithmic"

    def __init__(self, *, layers: int = 64, width: int = 1536, cases: int = 1000, shares: tuple[float, ...] = DEFAULT_SHARES):
        self.layers = layers
        self.width = width
        self.cases = cases
        self.shares = tuple(float(s) for s in shares)
        assert len(self.shares) == len(STRATA) and abs(sum(self.shares) - 1) < 1e-9

    def description(self) -> str:
        spec = mlp_spec_text(inputs=INPUTS, outputs=OUTPUTS, layers=self.layers, width=self.width)
        pct = {s: f"{100 * share:g}%" for s, share in zip(STRATA, self.shares)}
        return f"""\
## Task
Evaluate integer arithmetic expressions. An expression E is a string of at most {MAX_LEN} symbols over
0 1 2 3 4 5 6 7 8 9 + - * ( ). Every number is a single digit 0 to 9; there are no multi-digit numbers. +, - and * are
binary operators (there is no unary minus, so E never starts with '-' and '-' never follows '(' or another operator).
Parentheses group, nest at most {MAX_DEPTH} deep, and always enclose at least one operator (no "(7)"), though some pairs
are redundant, as in 2+(3*4). Standard precedence: * binds tighter than + and -, and operators of the same level apply
left to right, so 8-3-2 = 3, 2+3*4 = 14, 9-2*(3-5) = 13. E is well formed. The network outputs E's integer value,
exactly as Python's eval(E).
Every subexpression of E's standard parse, and E's value, lies between -{VALUE_BOUND} and {VALUE_BOUND} (for example 9*9*2
never appears, since 9*9*2 = 162).

## Network
A feed-forward ReLU network.
Input x [{INPUTS}]: {SLOTS} entries per position, starting at x[{SLOTS} * i] for position i = 0..{MAX_LEN - 1}:
  x[{SLOTS} * i + k] = 1 where k is the index of E[i] in "{SYMBOLS}" (digits 0..9 are k = 0..9, then + - * ( ) are
  k = 10..14); the other {SLOTS - 1} entries are 0. Positions i >= len(E) are blank: x[{SLOTS} * i + {SLOTS - 1}] = 1.
  E is left aligned: E[0] is at position 0.
{spec}
Output y [{OUTPUTS}]: y[0] is read as the value of E. It counts as right when |y[0] - value| < 0.5.

## Scoring
Each evaluation draws {self.cases:,} fresh random expressions; the score is the fraction answered right. They come in
four kinds, in fixed proportions:
  {pct['flat_add']} only + and -, no parentheses (for example 7-3+9-8).
  {pct['flat_mul']} +, - and *, no parentheses, at least one * (for example 4*5-6*3+2).
  {pct['paren_add']} only + and -, with parentheses (for example 6-(4-(7+2))+1).
  {pct['paren_mul']} +, - and *, with parentheses, at least one * (for example (3-5)*(2+4)-9*(8-7)).
Lengths range up to {MAX_LEN} symbols."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return mlp_shapes(inputs=INPUTS, outputs=OUTPUTS, layers=self.layers, width=self.width)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = random.Random(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        cases = sample_cases(rng=rng, count=self.cases, shares=self.shares)
        texts = [text for _, text in cases]
        truth = np.array([evaluate_tree(parse(t)) for t in texts], dtype=np.float64)
        y = mlp_forward(weights=weights, x=encode(texts), layers=self.layers)[:, 0].astype(np.float64)
        right = np.abs(y - truth) < 0.5
        values = {"exact_rate": float(right.mean()), "cases": float(len(cases)),
                  "mean_abs_error": float(np.nan_to_num(np.abs(y - truth), nan=1e9, posinf=1e9).mean())}
        strata = np.array([s for s, _ in cases])
        for stratum in STRATA:
            mask = strata == stratum
            if mask.any():
                values[f"exact_{stratum}"] = float(right[mask].mean())
        depths = np.array([min(paren_depth(t), 3) for t in texts])
        for depth, name in [(1, "1"), (2, "2"), (3, "3plus")]:
            if (depths == depth).any():
                values[f"exact_paren_depth_{name}"] = float(right[depths == depth].mean())
        lengths = np.array([len(t) for t in texts])
        for low, high in [(1, 8), (9, 16), (17, MAX_LEN)]:
            band = (lengths >= low) & (lengths <= high)
            if band.any():
                values[f"exact_len_{low}_{high}"] = float(right[band].mean())
        return Metrics(values=values, primary="exact_rate")
