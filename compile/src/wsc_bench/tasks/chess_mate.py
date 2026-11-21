"""Chess mate in one with a hand-written ReLU MLP: board planes in, one logit per (from, to) square pair out.

Positions come from seeded random self-play: the first position (after ply 10) where the side to move has a mating
move. Black-to-move positions are mirrored so White is always to move. The answer counts when the argmax (from, to)
is a legal move that checkmates (any promotion piece). A known deterministic program (try every legal move, keep the
ones that checkmate) scores 100%; no reference network has been built for this shape.
"""

import csv
import random
from functools import cache
from pathlib import Path

import chess
import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights
from wsc_bench.tasks.registry import register_target
from wsc_bench.tasks.relu_mlp import mlp_forward, mlp_shapes, mlp_spec_text

HIDDEN_SPLIT_SEED_OFFSET = 1_000_000_000
PIECE_ORDER = "PNBRQKpnbrqk"
INPUTS = 12 * 64 + 4 + 8
OUTPUTS = 64 * 64
MIN_PLY = 10
MAX_PLY = 300
LICHESS_MATE_IN_ONE = Path(__file__).resolve().parents[1] / "data" / "lichess_mate_in_one.csv"
SOURCES = ["random-play", "lichess"]


@cache
def lichess_rows() -> list[dict]:
    """Rated mate-in-one puzzles from real games (scripts/build_lichess_mate1.py), White to move."""
    with LICHESS_MATE_IN_ONE.open() as f:
        return list(csv.DictReader(f))


def mating_moves(board: chess.Board) -> list[chess.Move]:
    mates = []
    for move in board.legal_moves:
        if board.gives_check(move):
            board.push(move)
            if board.is_checkmate():
                mates.append(move)
            board.pop()
    return mates


def sample_positions(*, rng: random.Random, count: int) -> list[chess.Board]:
    boards = []
    while len(boards) < count:
        board = chess.Board()
        while not board.is_game_over() and board.ply() < MAX_PLY:
            if board.ply() >= MIN_PLY and mating_moves(board):
                boards.append(board.mirror() if board.turn == chess.BLACK else board.copy())
                break
            board.push(rng.choice(list(board.legal_moves)))
    return boards


def encode(board: chess.Board) -> np.ndarray:
    x = np.zeros(INPUTS, dtype=np.float32)
    for square, piece in board.piece_map().items():
        x[PIECE_ORDER.index(piece.symbol()) * 64 + square] = 1
    rights = [chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8]
    for i, rook_square in enumerate(rights):
        x[768 + i] = float(bool(board.castling_rights & rook_square))
    if board.ep_square is not None and board.has_legal_en_passant():
        x[772 + chess.square_file(board.ep_square)] = 1
    return x


@register_target
class ChessMateInOne(Target):
    id = "chess-mate-in-one"
    domain = "games"

    def __init__(self, *, layers: int = 8, width: int = 4096, positions: int = 500, source: str = "random-play"):
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        self.source = source
        self.layers = layers
        self.width = width
        self.positions = positions

    def description(self) -> str:
        spec = mlp_spec_text(inputs=INPUTS, outputs=OUTPUTS, layers=self.layers, width=self.width)
        return f"""\
## Task
Find a move that checkmates in one in a legal chess position. White is always to move and always has at least one
mating move. Standard chess rules apply.

## Network
A feed-forward ReLU network.
Squares are numbered s = 8 * rank + file with a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.
Input x [{INPUTS}]:
  x[64 * k + s] = 1 when square s holds piece k, with k indexing "{PIECE_ORDER}"
  (0 to 5: White pawn, knight, bishop, rook, queen, king; 6 to 11: the same for Black).
  x[768..771]: castling rights White king side, White queen side, Black king side, Black queen side (1 = available).
  x[772 + f] = 1 when an en passant capture onto file f (a = 0) is legal this move.
{spec}
Output y [{OUTPUTS}]: y[64 * from + to] is the logit of moving the piece on square `from` to square `to`. The network's move
is the argmax. A promotion is read as whichever promotion piece makes the move mate, if any.

## Scoring
Positions come from random games: the first position in a game where the side to move can mate in one (colors swapped
so White is to move). They are messy middlegames and endgames with about 20 pieces on average. The score is the
fraction of {self.positions} fresh positions where the argmax move is legal and checkmates."""

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return mlp_shapes(inputs=INPUTS, outputs=OUTPUTS, layers=self.layers, width=self.width)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        rng = random.Random(seed + (HIDDEN_SPLIT_SEED_OFFSET if split == "hidden" else 0))
        if self.source == "lichess":
            return self.evaluate_lichess(weights=weights, split=split, rng=rng)
        boards = sample_positions(rng=rng, count=self.positions)
        y = mlp_forward(weights=weights, x=np.stack([encode(b) for b in boards]), layers=self.layers)
        picks = y.argmax(axis=1)
        mates = legal = checks = 0
        for board, pick in zip(boards, picks):
            frm, to = divmod(int(pick), 64)
            moves = [m for m in board.legal_moves if m.from_square == frm and m.to_square == to]
            legal += bool(moves)
            checks += any(board.gives_check(m) for m in moves)
            mates += any(m in set(mating_moves(board)) for m in moves)
        n = len(boards)
        return Metrics(
            values={"mate_rate": mates / n, "legal_rate": legal / n, "check_rate": checks / n, "positions": float(n)},
            primary="mate_rate",
        )

    def evaluate_lichess(self, *, weights: Weights, split: Split, rng: random.Random) -> Metrics:
        """Up to `positions` puzzles from the split, sampled evenly across rating bands, scored per band."""
        pool = [row for row in lichess_rows() if row["split"] == split]
        bands = sorted({row["band"] for row in pool}, key=lambda b: min(int(r["rating"]) for r in pool if r["band"] == b))
        per_band = self.positions // len(bands)
        rows = [r for band in bands for r in rng.sample([r for r in pool if r["band"] == band],
                                                         min(per_band, sum(r["band"] == band for r in pool)))]
        boards = [chess.Board(row["fen"]) for row in rows]
        y = mlp_forward(weights=weights, x=np.stack([encode(b) for b in boards]), layers=self.layers)
        mated = []
        for board, pick in zip(boards, y.argmax(axis=1)):
            frm, to = divmod(int(pick), 64)
            moves = [m for m in board.legal_moves if m.from_square == frm and m.to_square == to]
            mated.append(any(m in set(mating_moves(board)) for m in moves))
        mated = np.array(mated)
        values = {"mate_rate": float(mated.mean()), "positions": float(len(rows))}
        for band in bands:
            in_band = np.array([r["band"] == band for r in rows])
            values[f"mate_rate_{band}"] = float(mated[in_band].mean())
        return Metrics(values=values, primary="mate_rate")
