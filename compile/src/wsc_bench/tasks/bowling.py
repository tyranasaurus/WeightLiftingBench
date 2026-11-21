"""Ten-pin bowling score with a hand-written ReLU MLP: the rolls of one complete game in, the final score out.

Input: up to 21 rolls, roll i one-hot over pins 0..10 at x[11 * i + pins]; rolls after the game ends are all zeros.
Output: one number, read as the score after rounding to the nearest integer. The hard part is alignment: which roll
starts each frame depends on every earlier strike, and strike and spare bonuses look ahead one or two rolls.
"""

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
MAX_ROLLS = 21
PINS = 11
INPUTS = MAX_ROLLS * PINS
GAME_KINDS = ["uniform", "strike_heavy", "near_perfect", "spares", "open"]
# per kind: probability the first ball of a frame is a strike, and that a non-strike frame is a spare
KIND_ODDS = {"uniform": (None, None), "strike_heavy": (0.5, 0.5), "near_perfect": (0.9, 0.5), "spares": (0.1, 0.8),
             "open": (0.0, 0.0)}


def score_game(rolls: list[int]) -> int:
    total, i = 0, 0
    for _ in range(10):
        if rolls[i] == 10:
            total += 10 + rolls[i + 1] + rolls[i + 2]
            i += 1
        elif rolls[i] + rolls[i + 1] == 10:
            total += 10 + rolls[i + 2]
            i += 2
        else:
            total += rolls[i] + rolls[i + 1]
            i += 2
    return total


def roll_frame(*, rng: np.random.Generator, kind: str) -> tuple[int, int | None]:
    strike_p, spare_p = KIND_ODDS[kind]
    if strike_p is None:
        first = int(rng.integers(0, 11))
    else:
        first = 10 if rng.random() < strike_p else int(rng.integers(0, 10))
    if first == 10:
        return first, None
    if spare_p is None:
        return first, int(rng.integers(0, 11 - first))
    return first, 10 - first if rng.random() < spare_p else int(rng.integers(0, 10 - first))


def sample_game(*, rng: np.random.Generator, kind: str) -> list[int]:
    rolls = []
    for _ in range(9):
        first, second = roll_frame(rng=rng, kind=kind)
        rolls += [first] if second is None else [first, second]
    first, second = roll_frame(rng=rng, kind=kind)
    if second is None:  # strike in the tenth: two bonus balls, the second from a fresh rack only after a strike
        bonus1, bonus2 = roll_frame(rng=rng, kind=kind)
        rolls += [10, bonus1, bonus2 if bonus2 is not None else roll_frame(rng=rng, kind=kind)[0]]
    elif first + second == 10:
        rolls += [first, second, roll_frame(rng=rng, kind=kind)[0]]
    else:
        rolls += [first, second]
    return rolls


def encode(games: list[list[int]]) -> np.ndarray:
    x = np.zeros((len(games), INPUTS), dtype=np.float32)
    for g, rolls in enumerate(games):
        for i, pins in enumerate(rolls):
            x[g, PINS * i + pins] = 1
    return x


@register_target
class Bowling(Target):
    id = "bowling"
    domain = "algorithmic"

    def __init__(self, *, layers: int = 32, width: int = 1536, games: int = 10_000):
        self.layers = layers
        self.width = width
        self.games = games

    def description(self) -> str:
        spec = mlp_spec_text(inputs=INPUTS, outputs=1, layers=self.layers, width=self.width)
        return f"""\
## Task
Score a complete game of ten-pin bowling from its rolls, using the standard rules:
- A game has 10 frames. A frame is two rolls, or one roll if the first knocks down all 10 pins (a strike).
- Open frame: the frame scores its pins. Spare (two rolls total 10): 10 plus the pins of the next roll.
- Strike: 10 plus the pins of the next two rolls.
- Tenth frame: a strike earns two bonus rolls and a spare earns one; bonus rolls only count as those bonuses.
The score is 0 to 300. Example: 12 strikes score 300; 21 rolls of 5 score 150; 20 rolls of 0 score 0.

## Network
A feed-forward ReLU network.
Input x [{INPUTS}]: roll i (i = 0 is the first ball of the game) knocking down p pins sets x[{PINS} * i + p] = 1, for p = 0..10.
A game has 11 to 21 rolls; positions after the last roll are all zeros.
{spec}
Output y [1]: the score. It counts as correct when round(y[0]) equals the true score.

## Scoring
Fresh random complete games, mixed across uniform pin counts, strike-heavy games, near-perfect games, spare-heavy games
and open games with no strikes or spares. The score is the fraction of {self.games} games scored exactly."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return mlp_shapes(inputs=INPUTS, outputs=1, layers=self.layers, width=self.width)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = np.random.default_rng(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        kinds = rng.integers(len(GAME_KINDS), size=self.games)
        games = [sample_game(rng=rng, kind=GAME_KINDS[k]) for k in kinds]
        truth = np.array([score_game(g) for g in games])
        y = mlp_forward(weights=weights, x=encode(games), layers=self.layers)[:, 0].astype(np.float64)
        correct = np.rint(y) == truth
        values = {"exact_match": float(correct.mean()), "mean_abs_error": float(np.abs(y - truth).mean()),
                  "games": float(self.games)}
        for k, kind in enumerate(GAME_KINDS):
            values[f"exact_match_{kind}"] = float(correct[kinds == k].mean())
        return Metrics(values=values, primary="exact_match")
