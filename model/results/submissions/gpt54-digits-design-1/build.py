from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

TAU = 2.0 * math.pi
R = math.radians
EPS = 1e-6

# Prototype counts per class
M28 = 20   # 4 styles x 5 slants
M40 = 40   # 4 styles x 5 slants x 2 widths

SLANTS = (-0.18, -0.09, 0.0, 0.09, 0.18)
WIDTHS = (1.55, 1.95)
FINE_WIDTH = 1.75

# Hand-tuned branch strengths
S28 = 0.70
S14_RAW = 0.70
S14_ORI = 1.35
S7_ORI = 0.55
S7_TOP = 0.75
S7_BOT = 0.75
S7_LEFT = 0.28
S7_RIGHT = 0.28


# ---------- generic geometry ----------
def grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    return x, y


GX28, GY28 = grid(28)


def poly(*pts):
    return ("p", tuple((float(x), float(y)) for x, y in pts))


def ell(cx, cy, rx, ry, a0=0.0, a1=TAU, n=64):
    return ("e", float(cx), float(cy), float(rx), float(ry), float(a0), float(a1), int(n))


def ellipse_points(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float, n: int):
    if a1 < a0:
        a1 += TAU
    ang = np.linspace(a0, a1, n, dtype=np.float32)
    return [(float(cx + rx * math.cos(a)), float(cy + ry * math.sin(a))) for a in ang]


def component_points(spec):
    if spec[0] == "p":
        return list(spec[1])
    _, cx, cy, rx, ry, a0, a1, n = spec
    return ellipse_points(cx, cy, rx, ry, a0, a1, n)


def dist_to_segment(X: np.ndarray, Y: np.ndarray, p0, p1) -> np.ndarray:
    x0, y0 = p0
    x1, y1 = p1
    vx, vy = x1 - x0, y1 - y0
    denom = vx * vx + vy * vy + 1e-9
    t = ((X - x0) * vx + (Y - y0) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    return np.sqrt((X - px) ** 2 + (Y - py) ** 2)


def transform_points(points, slant: float, size: int = 28):
    box = 20.0
    offset = (size - box) / 2.0
    out = []
    for x, y in points:
        x = x + slant * (y - 0.5)
        xx = offset + x * (box - 1.0)
        yy = offset + y * (box - 1.0)
        out.append((xx, yy))
    return out


def polyline_image(points, width: float, slant: float) -> np.ndarray:
    pts = transform_points(points, slant)
    d = np.full((28, 28), 1e9, np.float32)
    for a, b in zip(pts[:-1], pts[1:]):
        d = np.minimum(d, dist_to_segment(GX28, GY28, a, b))
    return np.exp(-0.5 * (d / width) ** 2).astype(np.float32)


def render_style(components, width: float, slant: float) -> np.ndarray:
    img = np.zeros((28, 28), np.float32)
    for comp in components:
        img = np.maximum(img, polyline_image(component_points(comp), width=width, slant=slant))
    return np.clip(img, 0.0, 1.0)


# ---------- digit style library ----------
def style_library():
    return {
        0: [
            [ell(0.50, 0.50, 0.25, 0.38)],
            [ell(0.50, 0.50, 0.22, 0.40)],
            [ell(0.50, 0.50, 0.29, 0.34)],
            [poly((0.50, 0.10), (0.72, 0.18), (0.80, 0.50), (0.70, 0.82), (0.50, 0.90), (0.28, 0.82), (0.20, 0.50), (0.30, 0.18), (0.50, 0.10))],
        ],
        1: [
            [poly((0.50, 0.12), (0.50, 0.88))],
            [poly((0.36, 0.26), (0.50, 0.12), (0.50, 0.88))],
            [poly((0.34, 0.22), (0.50, 0.12), (0.50, 0.88)), poly((0.34, 0.88), (0.66, 0.88))],
            [poly((0.42, 0.16), (0.56, 0.12), (0.56, 0.88))],
        ],
        2: [
            [ell(0.50, 0.30, 0.24, 0.16, R(195), R(15), 36), poly((0.72, 0.34), (0.28, 0.70), (0.78, 0.78))],
            [poly((0.28, 0.26), (0.72, 0.26), (0.64, 0.44), (0.36, 0.66), (0.22, 0.80), (0.78, 0.80))],
            [poly((0.26, 0.30), (0.44, 0.18), (0.68, 0.24), (0.72, 0.38), (0.54, 0.54), (0.30, 0.74), (0.78, 0.78))],
            [poly((0.30, 0.22), (0.70, 0.28), (0.68, 0.42), (0.50, 0.56), (0.28, 0.74), (0.76, 0.78))],
        ],
        3: [
            [poly((0.28, 0.24), (0.58, 0.20), (0.74, 0.34), (0.56, 0.48), (0.76, 0.64), (0.58, 0.80), (0.28, 0.76))],
            [ell(0.47, 0.31, 0.23, 0.15, R(195), R(20), 30), ell(0.49, 0.67, 0.25, 0.17, R(170), R(340), 36)],
            [poly((0.28, 0.22), (0.70, 0.22), (0.54, 0.50), (0.72, 0.78), (0.28, 0.78))],
            [poly((0.30, 0.22), (0.62, 0.20), (0.74, 0.34), (0.58, 0.50), (0.74, 0.64), (0.62, 0.82), (0.34, 0.78))],
        ],
        4: [
            [poly((0.26, 0.52), (0.60, 0.12)), poly((0.60, 0.12), (0.60, 0.88)), poly((0.22, 0.52), (0.74, 0.52))],
            [poly((0.30, 0.16), (0.30, 0.54), (0.72, 0.54)), poly((0.62, 0.12), (0.62, 0.88))],
            [poly((0.24, 0.58), (0.60, 0.18)), poly((0.60, 0.18), (0.60, 0.88)), poly((0.22, 0.58), (0.74, 0.58))],
            [poly((0.30, 0.20), (0.52, 0.20), (0.52, 0.56)), poly((0.22, 0.56), (0.74, 0.56)), poly((0.64, 0.12), (0.64, 0.88))],
        ],
        5: [
            [poly((0.74, 0.22), (0.28, 0.22), (0.28, 0.48), (0.58, 0.48), (0.74, 0.58), (0.62, 0.78), (0.30, 0.76))],
            [poly((0.74, 0.22), (0.30, 0.22), (0.30, 0.46), (0.56, 0.46), (0.74, 0.58), (0.70, 0.74), (0.50, 0.82), (0.28, 0.74))],
            [poly((0.74, 0.22), (0.34, 0.22), (0.30, 0.46), (0.60, 0.50), (0.72, 0.64), (0.60, 0.80), (0.28, 0.76))],
            [poly((0.72, 0.22), (0.30, 0.22), (0.28, 0.44), (0.52, 0.46), (0.72, 0.62), (0.64, 0.80), (0.34, 0.78))],
        ],
        6: [
            [poly((0.66, 0.20), (0.48, 0.22), (0.34, 0.38), (0.30, 0.56), (0.38, 0.76), (0.58, 0.78), (0.68, 0.62), (0.56, 0.46), (0.36, 0.50))],
            [poly((0.64, 0.18), (0.44, 0.24), (0.30, 0.40), (0.30, 0.62), (0.44, 0.80), (0.64, 0.76), (0.70, 0.58), (0.60, 0.44), (0.40, 0.44))],
            [poly((0.62, 0.16), (0.40, 0.30), (0.32, 0.52), (0.40, 0.74), (0.58, 0.80), (0.70, 0.66), (0.58, 0.50), (0.38, 0.52))],
            [poly((0.64, 0.18), (0.46, 0.24), (0.34, 0.40), (0.32, 0.60), (0.42, 0.78), (0.58, 0.78), (0.68, 0.64), (0.62, 0.48), (0.42, 0.50))],
        ],
        7: [
            [poly((0.22, 0.22), (0.78, 0.22), (0.40, 0.86))],
            [poly((0.22, 0.24), (0.78, 0.24), (0.60, 0.44), (0.44, 0.86)), poly((0.34, 0.48), (0.60, 0.48))],
            [poly((0.22, 0.24), (0.78, 0.24), (0.64, 0.36), (0.52, 0.56), (0.46, 0.86))],
            [poly((0.28, 0.22), (0.76, 0.24), (0.54, 0.86))],
        ],
        8: [
            [ell(0.50, 0.33, 0.20, 0.16), ell(0.50, 0.68, 0.23, 0.20)],
            [ell(0.50, 0.33, 0.18, 0.15), ell(0.50, 0.68, 0.21, 0.18)],
            [ell(0.50, 0.34, 0.19, 0.15), ell(0.50, 0.68, 0.20, 0.18), poly((0.44, 0.48), (0.56, 0.52))],
            [ell(0.50, 0.34, 0.22, 0.15), ell(0.50, 0.68, 0.24, 0.18)],
        ],
        9: [
            [ell(0.48, 0.34, 0.22, 0.19), poly((0.62, 0.46), (0.68, 0.84))],
            [ell(0.50, 0.34, 0.20, 0.18), poly((0.60, 0.44), (0.72, 0.60), (0.68, 0.82), (0.50, 0.78))],
            [ell(0.50, 0.32, 0.19, 0.17), poly((0.60, 0.42), (0.70, 0.54), (0.72, 0.74), (0.58, 0.86))],
            [ell(0.49, 0.34, 0.20, 0.18), poly((0.58, 0.44), (0.72, 0.56), (0.72, 0.76), (0.60, 0.86))],
        ],
    }


STYLES = style_library()


# ---------- feature extraction used to build weights ----------
def make_orientation_filters() -> np.ndarray:
    coords = np.arange(-2, 3, dtype=np.float32)
    Y, X = np.meshgrid(coords, coords)
    filters = []
    for theta in np.linspace(0.0, math.pi, 8, endpoint=False, dtype=np.float32):
        u = X * np.cos(theta) + Y * np.sin(theta)
        v = -X * np.sin(theta) + Y * np.cos(theta)
        line = np.exp(-0.5 * (v / 0.70) ** 2) * np.exp(-0.5 * (u / 1.60) ** 2)
        bg = 0.58 * np.exp(-0.5 * ((X / 1.55) ** 2 + (Y / 1.55) ** 2))
        k = line - bg
        k = k - k.mean()
        k = k / (np.sqrt((k * k).sum()) + 1e-6)
        filters.append(k.astype(np.float32))
    return np.stack(filters, axis=0)[:, None, :, :]


ORI_W = make_orientation_filters()
ORI_B = np.zeros((ORI_W.shape[0],), np.float32)


def conv_batch(x: np.ndarray, W: np.ndarray, b: np.ndarray, padding: int = 0) -> np.ndarray:
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho = h + 2 * padding - kh + 1
    wo = w + 2 * padding - kw + 1
    cols = np.stack([xp[:, :, dy:dy + ho, dx:dx + wo] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return (out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]).astype(np.float32)


def avgpool2(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5)).astype(np.float32)


def avgpool4(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 4, 4, w // 4, 4).mean(axis=(3, 5)).astype(np.float32)


def norm_batch(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=(2, 3), keepdims=True)
    var = ((x - mu) ** 2).mean(axis=(2, 3), keepdims=True)
    return ((x - mu) / np.sqrt(var + 1e-5)).astype(np.float32)


# ---------- masks and weight construction ----------
def support_mask(size: int) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    X, Y = np.meshgrid(t, t)
    return np.exp(-((np.abs(X) / 0.90) ** 8 + (np.abs(Y) / 0.90) ** 8)).astype(np.float32)


def top_mask(size: int) -> np.ndarray:
    y = np.arange(size, dtype=np.float32)
    if size == 7:
        w = np.array([1.0, 1.0, 1.0, 0.55, 0.0, 0.0, 0.0], np.float32)
    else:
        c = (size - 1) * 0.5
        w = np.clip((c + 0.6 - y) / (c + 0.6), 0.0, 1.0).astype(np.float32)
    return np.repeat(w[:, None], size, axis=1)


def bottom_mask(size: int) -> np.ndarray:
    return top_mask(size)[::-1].copy()


def left_mask(size: int) -> np.ndarray:
    return top_mask(size).T.copy()


def right_mask(size: int) -> np.ndarray:
    return left_mask(size)[:, ::-1].copy()


SUP28 = support_mask(28)
SUP14 = support_mask(14)
SUP7 = support_mask(7)
TOP7 = SUP7 * top_mask(7)
BOT7 = SUP7 * bottom_mask(7)
LEFT7 = SUP7 * left_mask(7)
RIGHT7 = SUP7 * right_mask(7)


def dense_from_features(features: np.ndarray, scale: float, support: np.ndarray, part: np.ndarray | None = None):
    base = features * support[None, None, :, :]
    masked = base if part is None else base * part[None, None, :, :]
    flat_base = base.reshape(len(base), -1)
    flat_masked = masked.reshape(len(masked), -1)
    denom = (flat_base * flat_base).sum(axis=1, keepdims=True) + EPS
    W = (scale * flat_masked / denom).astype(np.float32)
    b = np.zeros((len(features),), np.float32)
    return W, b


# ---------- prototype generation ----------
def build_prototypes(style_count: int = 4):
    fine, main = [], []
    for digit in range(10):
        styles = STYLES[digit][:style_count]
        for style in styles:
            for slant in SLANTS:
                fine.append(render_style(style, width=FINE_WIDTH, slant=slant))
                for width in WIDTHS:
                    main.append(render_style(style, width=width, slant=slant))
    fine = np.stack(fine, axis=0)[:, None, :, :].astype(np.float32)
    main = np.stack(main, axis=0)[:, None, :, :].astype(np.float32)
    assert fine.shape[0] == 10 * M28
    assert main.shape[0] == 10 * M40
    return fine, main


# ---------- network and weights ----------
def build_network():
    layers = [
        {"name": "n28", "op": "norm", "inputs": ["image"]},
        {"name": "p14", "op": "avgpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]},
        {"name": "n14", "op": "norm", "inputs": ["p14"]},
        {"name": "p7", "op": "avgpool", "inputs": ["image"], "size": [4, 4], "stride": [4, 4]},
        {"name": "n7", "op": "norm", "inputs": ["p7"]},
        {"name": "c1", "op": "conv", "inputs": ["image"], "out": int(ORI_W.shape[0]), "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "op14", "op": "avgpool", "inputs": ["r1"], "size": [2, 2], "stride": [2, 2]},
        {"name": "o14", "op": "norm", "inputs": ["op14"]},
        {"name": "op7", "op": "avgpool", "inputs": ["r1"], "size": [4, 4], "stride": [4, 4]},
        {"name": "o7", "op": "norm", "inputs": ["op7"]},
        {"name": "d28", "op": "dense", "inputs": ["n28"], "out": 10 * M28},
        {"name": "r28", "op": "reshape", "inputs": ["d28"], "shape": [1, 10, M28]},
        {"name": "m28", "op": "maxpool", "inputs": ["r28"], "size": [1, M28], "stride": [1, M28]},
        {"name": "s28", "op": "reshape", "inputs": ["m28"], "shape": [10]},
        {"name": "d14r", "op": "dense", "inputs": ["n14"], "out": 10 * M40},
        {"name": "r14r", "op": "reshape", "inputs": ["d14r"], "shape": [1, 10, M40]},
        {"name": "m14r", "op": "maxpool", "inputs": ["r14r"], "size": [1, M40], "stride": [1, M40]},
        {"name": "s14r", "op": "reshape", "inputs": ["m14r"], "shape": [10]},
        {"name": "d14o", "op": "dense", "inputs": ["o14"], "out": 10 * M40},
        {"name": "r14o", "op": "reshape", "inputs": ["d14o"], "shape": [1, 10, M40]},
        {"name": "m14o", "op": "maxpool", "inputs": ["r14o"], "size": [1, M40], "stride": [1, M40]},
        {"name": "s14o", "op": "reshape", "inputs": ["m14o"], "shape": [10]},
        {"name": "d7ow", "op": "dense", "inputs": ["o7"], "out": 10 * M40},
        {"name": "r7ow", "op": "reshape", "inputs": ["d7ow"], "shape": [1, 10, M40]},
        {"name": "m7ow", "op": "maxpool", "inputs": ["r7ow"], "size": [1, M40], "stride": [1, M40]},
        {"name": "s7ow", "op": "reshape", "inputs": ["m7ow"], "shape": [10]},
        {"name": "d7ot", "op": "dense", "inputs": ["o7"], "out": 10 * M40},
        {"name": "r7ot", "op": "reshape", "inputs": ["d7ot"], "shape": [1, 10, M40]},
        {"name": "m7ot", "op": "maxpool", "inputs": ["r7ot"], "size": [1, M40], "stride": [1, M40]},
        {"name": "s7ot", "op": "reshape", "inputs": ["m7ot"], "shape": [10]},
        {"name": "d7ob", "op": "dense", "inputs": ["o7"], "out": 10 * M40},
        {"name": "r7ob", "op": "reshape", "inputs": ["d7ob"], "shape": [1, 10, M40]},
        {"name": "m7ob", "op": "maxpool", "inputs": ["r7ob"], "size": [1, M40], "stride": [1, M40]},
        {"name": "s7ob", "op": "reshape", "inputs": ["m7ob"], "shape": [10]},
        {"name": "d7rl", "op": "dense", "inputs": ["n7"], "out": 10 * M40},
        {"name": "r7rl", "op": "reshape", "inputs": ["d7rl"], "shape": [1, 10, M40]},
        {"name": "m7rl", "op": "maxpool", "inputs": ["r7rl"], "size": [1, M40], "stride": [1, M40]},
        {"name": "s7rl", "op": "reshape", "inputs": ["m7rl"], "shape": [10]},
        {"name": "d7rr", "op": "dense", "inputs": ["n7"], "out": 10 * M40},
        {"name": "r7rr", "op": "reshape", "inputs": ["d7rr"], "shape": [1, 10, M40]},
        {"name": "m7rr", "op": "maxpool", "inputs": ["r7rr"], "size": [1, M40], "stride": [1, M40]},
        {"name": "s7rr", "op": "reshape", "inputs": ["m7rr"], "shape": [10]},
        {"name": "sumall", "op": "add", "inputs": ["s28", "s14r", "s14o", "s7ow", "s7ot", "s7ob", "s7rl", "s7rr"]},
    ]
    return {"layers": layers}


def build_weights():
    fine, main = build_prototypes()

    fine28 = norm_batch(fine)
    main14_raw = norm_batch(avgpool2(main))
    main7_raw = norm_batch(avgpool4(main))
    ori = np.maximum(conv_batch(main, ORI_W, ORI_B, padding=2), 0.0).astype(np.float32)
    main14_ori = norm_batch(avgpool2(ori))
    main7_ori = norm_batch(avgpool4(ori))

    weights = {
        "c1.W": ORI_W.astype(np.float32),
        "c1.b": ORI_B.astype(np.float32),
    }

    weights["d28.W"], weights["d28.b"] = dense_from_features(fine28, S28, SUP28)
    weights["d14r.W"], weights["d14r.b"] = dense_from_features(main14_raw, S14_RAW, SUP14)
    weights["d14o.W"], weights["d14o.b"] = dense_from_features(main14_ori, S14_ORI, SUP14)
    weights["d7ow.W"], weights["d7ow.b"] = dense_from_features(main7_ori, S7_ORI, SUP7)
    weights["d7ot.W"], weights["d7ot.b"] = dense_from_features(main7_ori, S7_TOP, SUP7, TOP7 / (SUP7 + 1e-6))
    weights["d7ob.W"], weights["d7ob.b"] = dense_from_features(main7_ori, S7_BOT, SUP7, BOT7 / (SUP7 + 1e-6))
    weights["d7rl.W"], weights["d7rl.b"] = dense_from_features(main7_raw, S7_LEFT, SUP7, LEFT7 / (SUP7 + 1e-6))
    weights["d7rr.W"], weights["d7rr.b"] = dense_from_features(main7_raw, S7_RIGHT, SUP7, RIGHT7 / (SUP7 + 1e-6))
    return weights


def main(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = build_network()
    weights = build_weights()
    with (out / "network.json").open("w") as f:
        json.dump(net, f)
    np.savez(out / "weights.npz", **weights)


if __name__ == "__main__":
    main(sys.argv[1])
