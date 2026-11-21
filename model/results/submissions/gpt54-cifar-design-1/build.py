import json
import sys
from pathlib import Path

import numpy as np

GRID = 8
C1_OUT = 18
PROTO_OUT = 40
IDENTITY_LAYERS = 48

CH = {
    "blue": 0,
    "green": 1,
    "red": 2,
    "yellow": 3,
    "brown": 4,
    "cyan": 5,
    "dark": 6,
    "bright": 7,
    "hpos": 8,
    "hneg": 9,
    "vpos": 10,
    "vneg": 11,
    "d1pos": 12,
    "d1neg": 13,
    "d2pos": 14,
    "d2neg": 15,
    "lappos": 16,
    "lapneg": 17,
}

GROUPS = {
    "warm": ["red", "yellow", "brown"],
    "cool": ["blue", "cyan"],
    "horiz": ["hpos", "hneg"],
    "vert": ["vpos", "vneg"],
    "diag": ["d1pos", "d1neg", "d2pos", "d2neg"],
    "edge": ["hpos", "hneg", "vpos", "vneg", "d1pos", "d1neg", "d2pos", "d2neg"],
    "lap": ["lappos", "lapneg"],
    "texture": ["hpos", "hneg", "vpos", "vneg", "d1pos", "d1neg", "d2pos", "d2neg", "lappos", "lapneg"],
}


def rect(r0, r1, c0, c1):
    m = np.zeros((GRID, GRID), dtype=np.float32)
    m[r0:r1, c0:c1] = 1.0
    return m


def union(*parts):
    m = np.zeros((GRID, GRID), dtype=np.float32)
    for p in parts:
        m = np.maximum(m, p)
    return m


def norm_mask(m):
    s = float(np.abs(m).sum())
    return m if s == 0.0 else (m / s).astype(np.float32)


MASKS = {
    "all": norm_mask(np.ones((GRID, GRID), dtype=np.float32)),
    "upper": norm_mask(rect(0, 3, 0, 8)),
    "upper2": norm_mask(rect(0, 4, 0, 8)),
    "lower": norm_mask(rect(5, 8, 0, 8)),
    "lower2": norm_mask(rect(4, 8, 0, 8)),
    "center": norm_mask(rect(2, 6, 2, 6)),
    "center_small": norm_mask(rect(3, 5, 3, 5)),
    "center_wide": norm_mask(rect(2, 6, 1, 7)),
    "center_tall": norm_mask(rect(1, 7, 2, 6)),
    "top_center": norm_mask(rect(0, 3, 2, 6)),
    "upper_center": norm_mask(rect(1, 4, 2, 6)),
    "lower_center": norm_mask(rect(4, 7, 2, 6)),
    "bottom_center": norm_mask(rect(5, 8, 2, 6)),
    "middle": norm_mask(rect(3, 5, 0, 8)),
    "side_mid": norm_mask(union(rect(2, 6, 0, 2), rect(2, 6, 6, 8))),
    "top_wide": norm_mask(rect(0, 4, 1, 7)),
    "bottom_wide": norm_mask(rect(4, 8, 1, 7)),
}


def gauss5(sigma=1.0):
    coords = np.arange(-2, 3, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma)).astype(np.float32)
    return g / g.sum()


def normalize_kernel(k, scale=1.0):
    k = np.asarray(k, dtype=np.float32)
    s = float(np.abs(k).sum())
    return k if s == 0.0 else (k * (scale / s)).astype(np.float32)


def add_map_weights(W, row, channel_or_group, mask_name, scale):
    names = GROUPS.get(channel_or_group, [channel_or_group])
    part = (np.float32(scale) / np.float32(len(names))) * MASKS[mask_name].reshape(-1)
    for name in names:
        start = CH[name] * GRID * GRID
        W[row, start:start + GRID * GRID] += part


def build_conv1():
    A = gauss5(1.0)
    luma = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    coords = np.arange(-2, 3, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)
    g = np.exp(-(x * x + y * y) / 2.0).astype(np.float32)

    gx = normalize_kernel(x * g, 2.4)
    gy = normalize_kernel(y * g, 2.4)
    gd1 = normalize_kernel((x + y) * g / np.sqrt(2.0), 2.4)
    gd2 = normalize_kernel((x - y) * g / np.sqrt(2.0), 2.4)
    log = normalize_kernel(((x * x + y * y) - 2.0) * g, 2.0)

    W = np.zeros((C1_OUT, 3, 5, 5), dtype=np.float32)
    b = np.zeros((C1_OUT,), dtype=np.float32)

    def color(name, coeffs, bias):
        W[CH[name]] = np.asarray(coeffs, dtype=np.float32)[:, None, None] * A[None, :, :]
        b[CH[name]] = np.float32(bias)

    color("blue", [-0.60, -0.40, 1.00], -0.02)
    color("green", [-0.60, 1.00, -0.50], -0.03)
    color("red", [1.00, -0.70, -0.40], -0.03)
    color("yellow", [0.60, 0.60, -1.00], -0.18)
    color("brown", [0.90, 0.10, -0.80], -0.16)
    color("cyan", [-0.90, 0.60, 0.60], -0.15)
    W[CH["dark"]] = -luma[:, None, None] * A[None, :, :]
    b[CH["dark"]] = np.float32(0.55)
    W[CH["bright"]] = luma[:, None, None] * A[None, :, :]
    b[CH["bright"]] = np.float32(-0.45)

    for name, kernel in [
        ("hpos", gy),
        ("hneg", -gy),
        ("vpos", gx),
        ("vneg", -gx),
        ("d1pos", gd1),
        ("d1neg", -gd1),
        ("d2pos", gd2),
        ("d2neg", -gd2),
        ("lappos", log),
        ("lapneg", -log),
    ]:
        W[CH[name]] = luma[:, None, None] * kernel[None, :, :]

    return W, b


def build_abslog():
    W = np.zeros((10, C1_OUT * GRID * GRID), dtype=np.float32)
    b = np.array([-0.55, -0.55, -0.50, -0.55, -0.60, -0.60, -0.55, -0.60, -0.55, -0.55], dtype=np.float32)

    specs = {
        0: [("blue", "upper2", 1.5), ("cyan", "upper", 0.5), ("bright", "upper2", 0.6), ("edge", "top_center", 0.6), ("diag", "center", 0.4), ("green", "lower2", -0.7), ("warm", "all", -0.6), ("brown", "all", -0.4), ("blue", "lower2", -0.2), ("horiz", "middle", -0.2)],
        1: [("dark", "lower2", 0.8), ("horiz", "lower2", 0.9), ("bright", "center", 0.6), ("edge", "center_small", 0.6), ("blue", "upper2", -0.5), ("green", "all", -0.4), ("vert", "center_tall", 0.3), ("brown", "all", -0.2)],
        2: [("edge", "upper_center", 0.6), ("diag", "center", 0.7), ("bright", "center", 0.5), ("green", "all", 0.4), ("blue", "upper", 0.4), ("edge", "center_small", 0.5), ("horiz", "lower2", -0.3), ("dark", "lower2", -0.2)],
        3: [("warm", "center", 0.7), ("brown", "center", 0.5), ("texture", "center", 0.8), ("dark", "upper_center", 0.7), ("diag", "top_center", 0.4), ("blue", "upper2", -0.4), ("green", "all", -0.2), ("horiz", "lower_center", -0.2)],
        4: [("green", "lower2", 0.9), ("brown", "center", 0.6), ("vert", "lower2", 0.7), ("edge", "center_wide", 0.4), ("blue", "upper2", 0.2), ("warm", "center", 0.2), ("dark", "lower2", 0.2)],
        5: [("warm", "center", 0.6), ("texture", "center", 0.8), ("dark", "lower_center", 0.6), ("edge", "center_wide", 0.4), ("brown", "center", 0.4), ("green", "all", -0.1), ("blue", "upper2", -0.1), ("horiz", "lower_center", 0.3)],
        6: [("green", "center", 1.1), ("dark", "center", 0.5), ("texture", "center", 0.8), ("bright", "center", 0.4), ("green", "all", 0.4), ("blue", "all", -0.2), ("horiz", "lower2", -0.2), ("brown", "all", 0.1)],
        7: [("brown", "center", 0.8), ("vert", "lower2", 0.8), ("edge", "center_wide", 0.5), ("green", "lower2", 0.6), ("dark", "center", 0.3), ("blue", "upper2", -0.1), ("horiz", "center_wide", 0.3)],
        8: [("blue", "lower2", 1.3), ("bright", "lower2", 0.5), ("horiz", "middle", 0.9), ("dark", "lower_center", 0.6), ("vert", "center_tall", 0.2), ("green", "all", -0.4), ("warm", "all", -0.2), ("blue", "upper2", -0.2)],
        9: [("dark", "lower2", 0.8), ("horiz", "center_wide", 0.7), ("vert", "center_tall", 0.8), ("edge", "center_wide", 0.5), ("bright", "center", 0.4), ("blue", "upper2", -0.3), ("green", "all", -0.3), ("brown", "all", -0.1)],
    }

    for row, terms in specs.items():
        for channel_or_group, mask_name, scale in terms:
            add_map_weights(W, row, channel_or_group, mask_name, scale)
    return W, b


PROTO_SPECS = [
    (0.75, [("blue", "upper2", 1.0), ("bright", "upper_center", 0.7), ("edge", "top_center", 0.6), ("diag", "center", 0.6), ("green", "lower2", -0.5), ("brown", "lower2", -0.4)]),
    (0.70, [("blue", "upper2", 0.9), ("horiz", "top_wide", 0.7), ("vert", "upper_center", 0.4), ("bright", "center", 0.4), ("blue", "lower2", -0.5), ("green", "all", -0.3)]),
    (0.65, [("cyan", "upper", 0.6), ("blue", "upper2", 0.7), ("dark", "upper_center", 0.5), ("edge", "upper_center", 0.6), ("warm", "all", -0.4)]),
    (0.70, [("bright", "upper2", 0.6), ("diag", "center", 0.7), ("edge", "center_small", 0.7), ("green", "lower2", -0.4), ("horiz", "middle", -0.3)]),

    (0.75, [("blue", "lower2", 1.0), ("horiz", "middle", 0.8), ("dark", "lower_center", 0.6), ("edge", "lower_center", 0.5), ("green", "all", -0.4)]),
    (0.70, [("blue", "lower2", 0.9), ("bright", "lower2", 0.6), ("vert", "center_tall", 0.4), ("horiz", "middle", 0.7), ("warm", "all", -0.3)]),
    (0.65, [("blue", "lower2", 0.8), ("dark", "center_wide", 0.6), ("edge", "center_wide", 0.5), ("green", "lower2", -0.5)]),
    (0.70, [("blue", "lower2", 0.9), ("bright", "upper2", 0.4), ("horiz", "middle", 0.8), ("blue", "upper2", -0.4), ("green", "all", -0.3)]),

    (0.75, [("dark", "lower2", 0.8), ("horiz", "lower2", 0.8), ("bright", "center", 0.5), ("edge", "center_small", 0.6), ("blue", "upper2", -0.5), ("green", "all", -0.4)]),
    (0.65, [("edge", "center_small", 0.8), ("horiz", "center_wide", 0.7), ("dark", "lower2", 0.6), ("vert", "center_tall", 0.3), ("texture", "center", -0.4)]),
    (0.65, [("red", "center", 0.6), ("dark", "lower2", 0.7), ("edge", "center_small", 0.6), ("blue", "upper2", -0.4), ("green", "all", -0.3)]),
    (0.65, [("blue", "center", 0.5), ("bright", "center", 0.5), ("horiz", "center_wide", 0.7), ("dark", "lower2", 0.6), ("vert", "center_tall", 0.3), ("green", "all", -0.3)]),

    (0.60, [("edge", "upper_center", 0.7), ("diag", "center", 0.8), ("bright", "center_small", 0.5), ("blue", "upper", 0.4), ("horiz", "lower2", -0.4)]),
    (0.55, [("edge", "center_small", 0.8), ("green", "all", 0.4), ("brown", "middle", 0.4), ("blue", "upper2", 0.3), ("vert", "center_tall", -0.2)]),
    (0.55, [("diag", "center", 0.8), ("blue", "upper2", 0.5), ("dark", "center_small", 0.4), ("green", "all", 0.3), ("dark", "lower2", -0.3)]),
    (0.55, [("bright", "center_small", 0.6), ("red", "center_small", 0.3), ("blue", "upper", 0.3), ("green", "all", 0.3), ("edge", "center_small", 0.7), ("horiz", "middle", -0.2)]),

    (0.70, [("texture", "center", 0.8), ("warm", "center", 0.7), ("dark", "upper_center", 0.8), ("diag", "top_center", 0.5), ("green", "all", -0.3)]),
    (0.65, [("brown", "center", 0.7), ("dark", "upper_center", 0.7), ("edge", "top_center", 0.5), ("texture", "center", 0.6), ("blue", "upper2", -0.2)]),
    (0.65, [("warm", "center", 0.7), ("dark", "center_small", 0.5), ("diag", "top_center", 0.6), ("texture", "center", 0.7), ("horiz", "lower_center", -0.3)]),
    (0.60, [("bright", "center", 0.5), ("texture", "center", 0.8), ("dark", "upper_center", 0.7), ("brown", "center", 0.5), ("green", "all", -0.2)]),

    (0.70, [("texture", "center", 0.8), ("warm", "center", 0.6), ("dark", "lower_center", 0.8), ("horiz", "lower_center", 0.5), ("diag", "top_center", -0.2)]),
    (0.65, [("brown", "center", 0.6), ("dark", "lower_center", 0.7), ("edge", "center_wide", 0.6), ("texture", "center", 0.7), ("green", "all", -0.2)]),
    (0.60, [("bright", "center", 0.5), ("texture", "center", 0.7), ("dark", "lower_center", 0.7), ("warm", "center", 0.6), ("blue", "upper2", -0.2)]),
    (0.60, [("edge", "center_wide", 0.7), ("dark", "center_small", 0.5), ("brown", "center", 0.5), ("horiz", "lower_center", 0.5), ("green", "all", -0.2)]),

    (0.70, [("green", "lower2", 0.8), ("brown", "center", 0.7), ("vert", "lower2", 0.8), ("edge", "center_wide", 0.4), ("blue", "upper2", 0.2)]),
    (0.65, [("green", "all", 0.5), ("brown", "center", 0.7), ("vert", "lower2", 0.8), ("diag", "center", 0.4), ("warm", "center", 0.3)]),
    (0.60, [("green", "lower2", 0.7), ("brown", "center", 0.6), ("edge", "center_small", 0.4), ("vert", "lower2", 0.7), ("texture", "center", -0.2)]),
    (0.60, [("blue", "upper2", 0.3), ("green", "lower2", 0.7), ("brown", "center", 0.6), ("vert", "lower2", 0.7), ("horiz", "middle", -0.2)]),

    (0.75, [("green", "center", 1.0), ("texture", "center", 0.8), ("dark", "center", 0.6), ("bright", "center", 0.4), ("blue", "all", -0.2)]),
    (0.70, [("green", "center", 0.9), ("green", "all", 0.5), ("texture", "center", 0.7), ("lappos", "center", 0.6), ("horiz", "lower2", -0.2)]),
    (0.65, [("green", "center", 0.8), ("brown", "center", 0.5), ("texture", "center", 0.7), ("dark", "center_small", 0.5), ("blue", "upper2", -0.2)]),
    (0.65, [("green", "center", 0.8), ("bright", "center", 0.5), ("edge", "center_small", 0.5), ("texture", "center", 0.7), ("brown", "all", 0.2)]),

    (0.70, [("brown", "center", 0.8), ("vert", "lower2", 0.8), ("edge", "center_wide", 0.5), ("green", "lower2", 0.6), ("diag", "top_center", -0.2)]),
    (0.65, [("brown", "center", 0.7), ("horiz", "center_wide", 0.5), ("vert", "lower2", 0.7), ("green", "lower2", 0.6), ("dark", "center", 0.4)]),
    (0.60, [("bright", "center", 0.4), ("brown", "center", 0.7), ("vert", "lower2", 0.8), ("edge", "center_wide", 0.5), ("blue", "upper2", 0.2)]),
    (0.60, [("brown", "center", 0.8), ("dark", "center_small", 0.4), ("vert", "lower2", 0.7), ("green", "all", 0.4), ("texture", "center", -0.2)]),

    (0.75, [("dark", "lower2", 0.8), ("horiz", "center_wide", 0.8), ("vert", "center_tall", 0.8), ("edge", "center_wide", 0.5), ("blue", "upper2", -0.4)]),
    (0.70, [("bright", "center", 0.5), ("dark", "lower2", 0.7), ("vert", "center_tall", 0.9), ("horiz", "center_wide", 0.7), ("green", "all", -0.3)]),
    (0.65, [("red", "center", 0.4), ("dark", "lower2", 0.7), ("vert", "center_tall", 0.8), ("edge", "center_wide", 0.6), ("horiz", "middle", 0.3)]),
    (0.65, [("yellow", "center", 0.4), ("dark", "lower2", 0.7), ("horiz", "center_wide", 0.8), ("vert", "center_tall", 0.7), ("bright", "upper2", 0.3)]),
]


def build_proto():
    W = np.zeros((PROTO_OUT, C1_OUT * GRID * GRID), dtype=np.float32)
    b = np.zeros((PROTO_OUT,), dtype=np.float32)
    for row, (bias, terms) in enumerate(PROTO_SPECS):
        b[row] = np.float32(-bias)
        for channel_or_group, mask_name, scale in terms:
            add_map_weights(W, row, channel_or_group, mask_name, scale)
    return W, b


def build_proto_out():
    W = np.zeros((10, PROTO_OUT), dtype=np.float32)
    b = np.zeros((10,), dtype=np.float32)

    mapping = {
        0: range(0, 4),
        8: range(4, 8),
        1: range(8, 12),
        2: range(12, 16),
        3: range(16, 20),
        5: range(20, 24),
        4: range(24, 28),
        6: range(28, 32),
        7: range(32, 36),
        9: range(36, 40),
    }
    for cls, cols in mapping.items():
        for col in cols:
            W[cls, col] = np.float32(1.1)

    W[0, 4:8] -= np.float32(0.20)
    W[8, 0:4] -= np.float32(0.20)
    W[1, 36:40] -= np.float32(0.25)
    W[9, 8:12] -= np.float32(0.20)
    W[3, 20:24] -= np.float32(0.10)
    W[5, 16:20] -= np.float32(0.10)
    W[4, 32:36] -= np.float32(0.15)
    W[7, 24:28] -= np.float32(0.15)
    W[2, 0:4] -= np.float32(0.10)
    W[2, 8:12] -= np.float32(0.10)

    return W, b


def build_network():
    layers = [
        {"name": "c1", "op": "conv", "inputs": ["image"], "out": C1_OUT, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "pavg", "op": "avgpool", "inputs": ["r1"], "size": [4, 4], "stride": [4, 4]},
        {"name": "pmax", "op": "maxpool", "inputs": ["r1"], "size": [4, 4], "stride": [4, 4]},
        {"name": "abslog", "op": "dense", "inputs": ["pavg"], "out": 10},
        {"name": "n1", "op": "norm", "inputs": ["pmax"]},
        {"name": "proto", "op": "dense", "inputs": ["n1"], "out": PROTO_OUT},
        {"name": "pr0", "op": "relu", "inputs": ["proto"]},
    ]

    prev = "pr0"
    for i in range(IDENTITY_LAYERS):
        dense = f"id{i}"
        act = f"ir{i}"
        layers.append({"name": dense, "op": "dense", "inputs": [prev], "out": PROTO_OUT})
        layers.append({"name": act, "op": "relu", "inputs": [dense]})
        prev = act

    layers.extend([
        {"name": "protolog", "op": "dense", "inputs": [prev], "out": 10},
        {"name": "logits", "op": "add", "inputs": ["abslog", "protolog"]},
    ])
    return {"layers": layers}


def build_answer():
    weights = {}

    W, b = build_conv1()
    weights["c1.W"] = W
    weights["c1.b"] = b

    W, b = build_abslog()
    weights["abslog.W"] = W
    weights["abslog.b"] = b

    W, b = build_proto()
    weights["proto.W"] = W
    weights["proto.b"] = b

    eye = np.eye(PROTO_OUT, dtype=np.float32)
    zero = np.zeros((PROTO_OUT,), dtype=np.float32)
    for i in range(IDENTITY_LAYERS):
        weights[f"id{i}.W"] = eye.copy()
        weights[f"id{i}.b"] = zero.copy()

    W, b = build_proto_out()
    weights["protolog.W"] = W
    weights["protolog.b"] = b

    return build_network(), weights


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net, weights = build_answer()
    (out / "network.json").write_text(json.dumps(net))
    np.savez(out / "weights.npz", **weights)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
