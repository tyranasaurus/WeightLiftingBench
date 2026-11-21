"""Produce weights.npz and network.json for a hand-designed skin-color based person detector.

Strategy
========
1. Per-pixel skin indicator from standard YCrCb-style bounds (7 conditions
   combined with a clip-based soft-AND so near-zero signal survives).
2. Pool to a grid of cell densities.
3. Compare the max cell density against the overall average: a face yields
   high peak density but low overall average, whereas a brown wall or sandy
   background has high peak AND high average — allowing its rejection.

All weights are derived from first principles (standard YCrCb skin colour
bounds and the max-minus-mean heuristic); no fitting to training data.
"""

import numpy as np
import json
import sys
from pathlib import Path


def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = {}

    conds = [
        (0.5,   -0.419, -0.081, -0.022),    # Cr > 0.522
        (-0.5,   0.419,  0.081,  0.178),    # Cr < 0.678
        (-0.169,-0.331,  0.5,    0.198),    # Cb > 0.302
        (0.169,  0.331, -0.5,   -0.002),    # Cb < 0.498
        (0.299,  0.587,  0.114, -0.07),     # Y  > 0.07
        (1.0,   -1.0,    0.0,   -0.012),    # R  > G + 0.012
        (0.0,    1.0,   -1.0,   -0.012),    # G  > B + 0.012
    ]
    s = 20.0
    n_conds = len(conds)

    conv_a_W = np.zeros((2 * n_conds, 3, 1, 1), dtype=np.float32)
    conv_a_b = np.zeros(2 * n_conds, dtype=np.float32)
    for i, (wR, wG, wB, b) in enumerate(conds):
        for ch, offset in [(2 * i, 0.0), (2 * i + 1, 1.0)]:
            conv_a_W[ch, 0, 0, 0] = s * wR
            conv_a_W[ch, 1, 0, 0] = s * wG
            conv_a_W[ch, 2, 0, 0] = s * wB
            conv_a_b[ch] = s * b - offset
    weights['conv_a.W'] = conv_a_W
    weights['conv_a.b'] = conv_a_b

    threshold = 5.5
    conv_b_W = np.zeros((1, 2 * n_conds, 1, 1), dtype=np.float32)
    conv_b_b = np.zeros(1, dtype=np.float32)
    for i in range(n_conds):
        conv_b_W[0, 2 * i,     0, 0] = +1.0
        conv_b_W[0, 2 * i + 1, 0, 0] = -1.0
    conv_b_b[0] = -threshold
    weights['conv_b.W'] = conv_b_W
    weights['conv_b.b'] = conv_b_b

    a = 4.0
    bcoef = 16.0
    beta = 1.1

    dense_max_W = np.zeros((2, 1), dtype=np.float32)
    dense_max_b = np.zeros(2, dtype=np.float32)
    dense_max_W[1, 0] = a
    dense_max_b[0] = beta
    weights['dense_max.W'] = dense_max_W
    weights['dense_max.b'] = dense_max_b

    dense_avg_W = np.zeros((2, 1), dtype=np.float32)
    dense_avg_b = np.zeros(2, dtype=np.float32)
    dense_avg_W[1, 0] = -bcoef
    weights['dense_avg.W'] = dense_avg_W
    weights['dense_avg.b'] = dense_avg_b

    network = {
        "layers": [
            {"name": "conv_a",    "op": "conv",    "inputs": ["image"],     "out": 2 * n_conds, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "relu_a",    "op": "relu",    "inputs": ["conv_a"]},
            {"name": "conv_b",    "op": "conv",    "inputs": ["relu_a"],    "out": 1,           "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "relu_b",    "op": "relu",    "inputs": ["conv_b"]},
            {"name": "pool_cell", "op": "avgpool", "inputs": ["relu_b"],    "size": [12, 12],   "stride": [12, 12]},
            {"name": "pool_max",  "op": "maxpool", "inputs": ["pool_cell"], "size": [8, 8],     "stride": [8, 8]},
            {"name": "pool_avg",  "op": "avgpool", "inputs": ["pool_cell"], "size": [8, 8],     "stride": [8, 8]},
            {"name": "flat_max",  "op": "reshape", "inputs": ["pool_max"],  "shape": [1]},
            {"name": "flat_avg",  "op": "reshape", "inputs": ["pool_avg"],  "shape": [1]},
            {"name": "dense_max", "op": "dense",   "inputs": ["flat_max"],  "out": 2},
            {"name": "dense_avg", "op": "dense",   "inputs": ["flat_avg"],  "out": 2},
            {"name": "scores",    "op": "add",     "inputs": ["dense_max", "dense_avg"]},
        ],
    }
    (out_dir / "network.json").write_text(json.dumps(network, indent=2))
    np.savez(out_dir / "weights.npz", **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
