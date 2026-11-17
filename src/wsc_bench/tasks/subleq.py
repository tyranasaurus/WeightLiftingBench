"""One step of a SUBLEQ computer (one-instruction machine, as in "Can You Train a Computer?") as a ReLU MLP.

State: 16 memory cells of signed 8-bit integers plus a 4-bit program counter, 132 bits. One step reads the instruction
(a, b, c) at the program counter, subtracts M[a] from M[b] with int8 wraparound and branches on the result. The input
domain is 2^132 states, so no lookup table fits. Scoring draws fresh states per seed from a mix of kinds (uniform,
operands with a == b, results of exactly 0, int8 overflow, instructions that overwrite themselves, instructions that
wrap past address 15); the metric is the fraction whose whole new state is exactly right. A construction that fits:
tests/test_subleq.py (8 linear maps).
"""

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
CELLS = 16
CELL_BITS = 8
PC_BITS = 4
STATE_BITS = CELLS * CELL_BITS + PC_BITS
KINDS = ["uniform", "a_equals_b", "zero_result", "overflow", "self_modify", "pc_wrap"]
KIND_SHARES = [0.4, 0.1, 0.15, 0.15, 0.1, 0.1]


def to_bits(values: np.ndarray, width: int) -> np.ndarray:
    """Integers (any sign, two's complement) to bits, most significant first."""
    shifts = np.arange(width - 1, -1, -1)
    return ((values.astype(np.int64)[..., None] >> shifts) & 1).astype(np.float32)


def encode_state(mem: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """mem [n, 16] int8-range values, pc [n] -> [n, 132] bits: cell 0..15, 8 bits each, then the pc."""
    return np.concatenate([to_bits(mem, CELL_BITS).reshape(len(mem), -1), to_bits(pc, PC_BITS)], axis=1)


def step(mem: np.ndarray, pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The exact SUBLEQ step on a batch: mem [n, 16] in [-128, 127], pc [n] in [0, 16)."""
    rows = np.arange(len(mem))
    a, b, c = (mem[rows, (pc + i) % CELLS].astype(np.int64) for i in range(3))
    a_addr, b_addr, c_addr = a % CELLS, b % CELLS, c % CELLS
    result = ((mem[rows, b_addr].astype(np.int64) - mem[rows, a_addr] + 128) % 256) - 128
    new_mem = mem.copy()
    new_mem[rows, b_addr] = result
    return new_mem, np.where(result <= 0, c_addr, (pc + 3) % CELLS)


def sample_states(*, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """count random states (mem [count, 16], pc [count]) plus the kind index of each."""
    counts = np.floor(np.array(KIND_SHARES) * count).astype(int)
    counts[0] += count - counts.sum()
    kinds = np.repeat(np.arange(len(KINDS)), counts)
    mem = rng.integers(-128, 128, size=(count, CELLS))
    pc = rng.integers(0, CELLS, size=count)

    def with_address(address: int) -> int:  # a random int8 whose value mod 16 is `address`
        return address + 16 * int(rng.integers(-8, 8))

    for i, kind in enumerate(kinds):
        m, p = mem[i], int(pc[i])
        triple = [(p + j) % CELLS for j in range(3)]
        data_cells = [q for q in range(CELLS) if q not in triple]
        if KINDS[kind] == "pc_wrap":
            pc[i] = p = int(rng.integers(13, 16))
        elif KINDS[kind] == "a_equals_b":
            m[(p + 1) % CELLS] = with_address(int(m[p] % CELLS))
        elif KINDS[kind] in ("zero_result", "overflow"):
            a_addr, b_addr = (int(q) for q in rng.choice(data_cells, size=2, replace=False))
            m[p], m[(p + 1) % CELLS] = with_address(a_addr), with_address(b_addr)
            if KINDS[kind] == "zero_result":
                m[b_addr] = m[a_addr]
            elif rng.random() < 0.5:  # negative minus positive, below -128
                m[a_addr] = rng.integers(1, 128)
                m[b_addr] = rng.integers(-128, m[a_addr] - 128)
            else:  # positive minus negative, above 127
                m[b_addr] = rng.integers(1, 128)
                m[a_addr] = rng.integers(-128, m[b_addr] - 127)
        elif KINDS[kind] == "self_modify":  # b points into the instruction itself
            m[(p + 1) % CELLS] = with_address(int(rng.choice(triple)))
    return mem, pc, kinds


@register_target
class SubleqStep(Target):
    id = "subleq-step"
    domain = "algorithmic"

    def __init__(self, *, layers: int = 12, width: int = 1536, samples: int = 1000):
        self.layers = layers
        self.width = width
        self.samples = samples

    def description(self) -> str:
        spec = mlp_spec_text(inputs=STATE_BITS, outputs=STATE_BITS, layers=self.layers, width=self.width)
        return f"""\
## Task
Compute one step of a SUBLEQ computer (a one-instruction machine). The machine has 16 memory cells M[0..15], each a signed
8-bit integer (two's complement, range -128..127, arithmetic wraps around modulo 256), and a program counter pc in 0..15.
One step does exactly this, using the memory as it is before the step:
    a = M[pc]; b = M[(pc + 1) mod 16]; c = M[(pc + 2) mod 16]
    a, b and c are used as cell addresses modulo 16 (-1 means cell 15, 17 means cell 1).
    r = M[b mod 16] - M[a mod 16], wrapped into -128..127   (if a and b address the same cell, r = 0)
    M[b mod 16] = r
    if r <= 0: pc = c mod 16, otherwise pc = (pc + 3) mod 16
All other cells are unchanged. The instruction cells may themselves be the cell that is written.

## Network
A feed-forward ReLU network.
Input x [{STATE_BITS}]: the state before the step. x[8*i : 8*i + 8] is M[i] for i = 0..15, as 8 bits of its two's complement
form, most significant bit first (the first bit has weight -128 and is 1 exactly for negative values). x[128:132] is pc in
binary, most significant bit first. Every bit is 0.0 or 1.0.
{spec}
Output y [{STATE_BITS}]: the state after the step, in the same layout as the input (y[8*i : 8*i + 8] is the new M[i], y[128:132]
is the new pc). Output bit i is read as 1 when y[i] > 0.5, else 0.

## Worked example
pc = 2, M[2] = 5, M[3] = 7, M[4] = 9, M[5] = 20, M[7] = -3, every other cell 0. Then a = 5, b = 7, c = 9, and
M[7] = -3 - 20 = -23 <= 0, so pc = 9. After the step M[7] = -23 and pc = 9; the rest is unchanged.
M[7] before is 11111101 (-3), after is 11101001 (-23); pc before is 0010, after is 1001.
Wraparound: with M[b] = -100 and M[a] = 100, r = -200 wraps to 56, which is > 0, so pc becomes (pc + 3) mod 16.

## Scoring
Each evaluation draws {self.samples:,} fresh random states: uniformly random memories and counters mixed with states where
a and b address the same cell, states where the result is exactly 0, int8 overflow cases, instructions whose b address
points into the instruction itself, and counters near 15 so the instruction wraps past cell 15. The score is the fraction
of states whose all {STATE_BITS} output bits equal the correct new state exactly."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return mlp_shapes(inputs=STATE_BITS, outputs=STATE_BITS, layers=self.layers, width=self.width)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        mem, pc, kinds = sample_states(rng=rng, count=self.samples)
        x = encode_state(mem, pc)
        new_mem, new_pc = step(mem, pc)
        truth = encode_state(new_mem, new_pc).astype(bool)
        got = mlp_forward(weights=weights, x=x, layers=self.layers) > 0.5
        match = got == truth
        exact = match.all(axis=1)
        per_kind = {f"exact_match_{name}": float(exact[kinds == k].mean()) for k, name in enumerate(KINDS)
                    if (kinds == k).any()}
        return Metrics(
            values={"exact_match": float(exact.mean()), "bit_accuracy": float(match.mean()),
                    "pc_exact": float(match[:, -PC_BITS:].all(axis=1).mean()),
                    "memory_exact": float(match[:, :-PC_BITS].all(axis=1).mean()),
                    **per_kind, "tested_states": float(len(mem))},
            primary="exact_match",
        )
