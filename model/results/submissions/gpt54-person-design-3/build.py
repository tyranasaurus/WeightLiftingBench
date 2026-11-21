import json
import sys
from pathlib import Path

import numpy as np


LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
VP, VN, HP, HN, D1P, D1N, D2P, D2N, LPP, LPN, BRI, DARK, BLUE, GREEN, WARM, SKIN = range(16)


def gray(mask):
    mask = np.asarray(mask, dtype=np.float32)
    return (LUMA[:, None, None] * mask[None, :, :]).astype(np.float32)


def set_center_rgb(W, out_c, coeffs):
    W[out_c, :, 1, 1] = np.asarray(coeffs, dtype=np.float32)


def flat_index(c, y, x):
    return c * 64 + y * 8 + x


def add_map(v, c, m, scale):
    m = np.asarray(m, dtype=np.float32)
    for y in range(8):
        for x in range(8):
            v[flat_index(c, y, x)] += np.float32(scale) * m[y, x]


def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    network = {
        "layers": [
            {"name": "c1", "op": "conv", "inputs": ["image"], "out": 16, "kernel": [3, 3], "stride": 1, "padding": 1},
            {"name": "a1", "op": "relu", "inputs": ["c1"]},
            {"name": "p1", "op": "avgpool", "inputs": ["a1"], "size": [12, 12], "stride": [12, 12]},
            {"name": "f1", "op": "reshape", "inputs": ["p1"], "shape": [1024]},
            {"name": "out", "op": "dense", "inputs": ["f1"], "out": 2},
        ]
    }

    c1_W = np.zeros((16, 3, 3, 3), dtype=np.float32)
    c1_b = np.zeros(16, dtype=np.float32)

    v = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float32) / 3.0
    h = v.T.copy()
    d1 = np.array([[-2.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]], dtype=np.float32) / 4.0
    d2 = np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]], dtype=np.float32) / 4.0
    lap = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32) / 4.0

    c1_W[VP] = gray(v)
    c1_W[VN] = gray(-v)
    c1_W[HP] = gray(h)
    c1_W[HN] = gray(-h)
    c1_W[D1P] = gray(d1)
    c1_W[D1N] = gray(-d1)
    c1_W[D2P] = gray(d2)
    c1_W[D2N] = gray(-d2)
    c1_W[LPP] = gray(lap)
    c1_W[LPN] = gray(-lap)

    set_center_rgb(c1_W, BRI, [0.299, 0.587, 0.114])
    set_center_rgb(c1_W, DARK, [-0.299, -0.587, -0.114])
    c1_b[DARK] = np.float32(0.55)

    set_center_rgb(c1_W, BLUE, [-0.45, -0.45, 1.0])
    c1_b[BLUE] = np.float32(-0.10)

    set_center_rgb(c1_W, GREEN, [-0.40, 1.0, -0.40])
    c1_b[GREEN] = np.float32(-0.08)

    set_center_rgb(c1_W, WARM, [0.90, 0.15, -0.55])
    c1_b[WARM] = np.float32(-0.22)

    set_center_rgb(c1_W, SKIN, [1.10, 0.25, -0.75])
    c1_b[SKIN] = np.float32(-0.46)

    yy, xx = np.mgrid[0:8, 0:8].astype(np.float32)
    d = np.sqrt((yy - 3.5) ** 2 + (xx - 3.5) ** 2)
    center = np.exp(-(d ** 2) / 7.0).astype(np.float32)
    core = (0.40 + 0.80 * center).astype(np.float32)
    wide = (0.25 + 0.55 * center).astype(np.float32)
    top = np.clip(1.10 - yy / 3.0, 0.0, 1.10).astype(np.float32)
    middle = (((yy >= 2) & (yy <= 5) & (xx >= 2) & (xx <= 5)).astype(np.float32))
    border = (((yy < 1) | (yy > 6) | (xx < 1) | (xx > 6)).astype(np.float32))

    person = np.zeros(1024, dtype=np.float32)

    for c in (VP, VN, HP, HN):
        add_map(person, c, core, 0.90)
        add_map(person, c, middle, 0.25)
    for c in (D1P, D1N, D2P, D2N):
        add_map(person, c, wide, 0.55)
    for c in (LPP, LPN):
        add_map(person, c, wide, 0.45)

    add_map(person, BRI, core, -0.55)
    add_map(person, DARK, core, 0.45)
    add_map(person, BLUE, top, 0.55)
    add_map(person, BLUE, core, 0.12)
    add_map(person, GREEN, core, 0.05)
    add_map(person, WARM, core, -0.45)
    add_map(person, WARM, border, -0.25)
    add_map(person, SKIN, middle, 0.55)
    add_map(person, SKIN, border, -0.25)

    out_W = np.zeros((2, 1024), dtype=np.float32)
    out_b = np.array([0.25, -0.25], dtype=np.float32)
    out_W[1] = person
    out_W[0] = -person

    np.savez(
        out_dir / "weights.npz",
        **{
            "c1.W": c1_W,
            "c1.b": c1_b,
            "out.W": out_W,
            "out.b": out_b,
        },
    )
    (out_dir / "network.json").write_text(json.dumps(network, indent=2) + "\n")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
