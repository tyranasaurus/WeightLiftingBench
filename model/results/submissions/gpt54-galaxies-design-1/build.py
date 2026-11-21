import json
import math
import sys
from pathlib import Path

import numpy as np


TAU = 2.0 * math.pi
WARM = np.array([1.00, 0.82, 0.46], np.float32)
BLUE = np.array([-0.20, 0.15, 1.00], np.float32)
NEUTRAL = np.array([0.55, 0.58, 0.66], np.float32)


def grid(n: int):
    ax = (np.arange(n, dtype=np.float32) + 0.5 - n / 2.0) / (n / 2.0)
    y, x = np.meshgrid(ax, ax, indexing="ij")
    return x, y


def rot_coords(x, y, theta, dx=0.0, dy=0.0):
    xt = x - dx
    yt = y - dy
    c = np.float32(math.cos(theta))
    s = np.float32(math.sin(theta))
    xr = c * xt + s * yt
    yr = -s * xt + c * yt
    return xr, yr


def ellipse(n, size, q, theta=0.0, dx=0.0, dy=0.0, alpha=1.4):
    x, y = grid(n)
    xr, yr = rot_coords(x, y, theta, dx, dy)
    rs = (xr / (size + 1e-4)) ** 2 + (yr / (size * q + 1e-4)) ** 2
    return np.exp(-(rs ** (alpha / 2.0))).astype(np.float32)


def ring(n, ro, ri, q=1.0, theta=0.0):
    return np.maximum(ellipse(n, ro, q, theta, alpha=1.5) - ellipse(n, ri, q, theta, alpha=1.5), 0).astype(np.float32)


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def spiral_field(n, size, q, theta, pitch, chirality, width, one_arm=False):
    x, y = grid(n)
    xr, yr0 = rot_coords(x, y, theta)
    yr = yr0 / (q + 1e-4)
    r = np.sqrt(xr * xr + yr * yr) + 1e-3
    phi = np.arctan2(yr, xr)
    env = np.exp(-((r / (size + 1e-4)) ** 1.35)).astype(np.float32)
    gate = 1.0 / (1.0 + np.exp(-(r - size * 0.14) / (size * 0.05 + 1e-4)))
    phase = phi - chirality * pitch * np.log(r / (size * 0.12 + 0.03) + 1.0)
    a1 = np.exp(-0.5 * (wrap_angle(phase) / width) ** 2)
    if one_arm:
        a2 = np.exp(-0.5 * (wrap_angle(phase + math.pi) / (width * 1.25)) ** 2)
        out = a1 - 0.62 * a2
    else:
        a2 = np.exp(-0.5 * (wrap_angle(phase + math.pi) / width) ** 2)
        out = a1 + a2 - 0.52
    return (env * gate * out).astype(np.float32)


def add_rgb(out, field, color, scale):
    out += field[..., None] * (color[None, None, :] * np.float32(scale))


def smooth_feature_template(kind, n, size, theta):
    t = np.zeros((n, n, 3), np.float32)
    if kind == 0:  # round smooth
        pos = ellipse(n, size, 1.0, theta, alpha=1.3)
        mid = ellipse(n, size * 0.96, 0.72, theta, alpha=1.3)
        outer = ring(n, size * 1.16, size * 0.58, 1.0, theta)
        core = ellipse(n, size * 0.20, 1.0, theta, alpha=2.0)
        add_rgb(t, pos, WARM, 1.00)
        add_rgb(t, core, WARM, 0.28)
        add_rgb(t, mid, WARM, -0.34)
        add_rgb(t, outer, BLUE, -0.18)
    elif kind == 1:  # intermediate ellipse
        pos = ellipse(n, size, 0.66, theta, alpha=1.28)
        rounder = ellipse(n, size * 0.98, 1.0, theta, alpha=1.30)
        thinner = ellipse(n, size * 1.00, 0.28, theta, alpha=1.22)
        outer = ring(n, size * 1.14, size * 0.56, 0.82, theta)
        core = ellipse(n, size * 0.18, 1.0, theta, alpha=2.0)
        add_rgb(t, pos, WARM, 1.02)
        add_rgb(t, core, WARM, 0.18)
        add_rgb(t, rounder, WARM, -0.28)
        add_rgb(t, thinner, WARM, -0.26)
        add_rgb(t, outer, BLUE, -0.14)
    elif kind == 2:  # cigar smooth
        pos = ellipse(n, size, 0.24, theta, alpha=1.20)
        thick = ellipse(n, size * 0.98, 0.56, theta, alpha=1.28)
        thin = ellipse(n, size * 1.02, 0.10, theta, alpha=1.05)
        core = ellipse(n, size * 0.16, 1.0, theta, alpha=2.0)
        add_rgb(t, pos, WARM, 1.08)
        add_rgb(t, core, WARM, 0.10)
        add_rgb(t, thick, WARM, -0.38)
        add_rgb(t, thin, BLUE, -0.18)
    elif kind == 3:  # edge-on without bulge
        line = ellipse(n, size, 0.09, theta, alpha=1.02)
        thick = ellipse(n, size * 1.06, 0.28, theta, alpha=1.55)
        core = ellipse(n, size * 0.18, 1.0, theta, alpha=2.0)
        add_rgb(t, line, BLUE, 1.08)
        add_rgb(t, thick, NEUTRAL, -0.40)
        add_rgb(t, core, WARM, -0.30)
    elif kind == 4:  # edge-on with bulge
        line = ellipse(n, size, 0.10, theta, alpha=1.02)
        thick = ellipse(n, size * 1.06, 0.28, theta, alpha=1.55)
        core = ellipse(n, size * 0.20, 1.0, theta, alpha=2.0)
        add_rgb(t, line, BLUE, 0.84)
        add_rgb(t, thick, NEUTRAL, -0.32)
        add_rgb(t, core, WARM, 0.58)
    else:
        raise ValueError(kind)
    return t


def spiral_feature_template(kind, n, size, theta, chirality):
    t = np.zeros((n, n, 3), np.float32)
    core = ellipse(n, size * 0.18, 1.0, theta, alpha=2.0)
    circ = ellipse(n, size * 0.98, 1.0, theta, alpha=1.35)
    if kind == 0:  # tight spiral
        arm = spiral_field(n, size, 0.88, theta, 4.3, chirality, 0.30)
        add_rgb(t, arm, BLUE, 1.20)
        add_rgb(t, circ, WARM, -0.26)
        add_rgb(t, core, WARM, 0.10)
    elif kind == 1:  # loose spiral
        arm = spiral_field(n, size, 0.94, theta, 2.05, chirality, 0.48)
        add_rgb(t, arm, BLUE, 1.14)
        add_rgb(t, circ, WARM, -0.22)
        add_rgb(t, core, WARM, 0.06)
    elif kind == 2:  # bar
        bar = ellipse(n, size * 0.42, 0.10, theta, alpha=1.0)
        edge = ellipse(n, size * 0.78, 0.09, theta, alpha=1.02)
        rounder = ellipse(n, size * 0.36, 1.0, theta, alpha=1.3)
        add_rgb(t, bar, WARM, 1.10)
        add_rgb(t, edge, BLUE, -0.26)
        add_rgb(t, rounder, WARM, -0.24)
    else:
        raise ValueError(kind)
    return t


def local_feature_template(kind, n, size, theta, style):
    t = np.zeros((n, n, 3), np.float32)
    core = ellipse(n, size * 0.18, 1.0, theta, alpha=2.0)
    circ = ellipse(n, size * 0.98, 1.0, theta, alpha=1.35)
    if kind == 0:  # disturbed
        tail = spiral_field(n, size, 0.92, theta, 2.2 + 0.35 * style, 1.0 if style == 0 else -1.0, 0.34 + 0.06 * style, one_arm=True)
        cl1 = ellipse(n, size * 0.11, 1.0, 0.0, dx=0.24 * math.cos(theta), dy=0.18 * math.sin(theta + 0.7), alpha=2.0)
        cl2 = ellipse(n, size * 0.11, 1.0, 0.0, dx=-0.22 * math.cos(theta + 0.5), dy=-0.17 * math.sin(theta), alpha=2.0)
        add_rgb(t, tail, BLUE, 1.18)
        add_rgb(t, cl1 - 0.72 * cl2, BLUE, 0.92)
        add_rgb(t, core, WARM, 0.18)
        add_rgb(t, circ, WARM, -0.18)
    elif kind == 1:  # merger
        sep = size * (0.95 if style == 0 else 1.28)
        dx = 0.5 * sep * math.cos(theta)
        dy = 0.5 * sep * math.sin(theta)
        c1 = ellipse(n, size * 0.17, 1.0, 0.0, dx=dx, dy=dy, alpha=2.0)
        c2 = ellipse(n, size * 0.17, 1.0, 0.0, dx=-dx, dy=-dy, alpha=2.0)
        bridge = ellipse(n, size * 0.34, 0.16, theta, alpha=1.6)
        add_rgb(t, c1 + c2, WARM, 1.14)
        add_rgb(t, bridge, BLUE, 0.34)
        add_rgb(t, core, WARM, -0.98)
        add_rgb(t, circ, NEUTRAL, -0.14)
    else:
        raise ValueError(kind)
    return t


def to_weight(img, gain):
    w = img.transpose(2, 0, 1).astype(np.float32)
    w -= w.reshape(3, -1).mean(axis=1)[:, None, None]
    nrm = float(np.sqrt(np.sum(w.astype(np.float64) ** 2)))
    if nrm < 1e-8:
        nrm = 1.0
    w *= np.float32(gain / nrm)
    return w


def make_smooth_bank(n, sizes, angles, gain):
    variants = len(sizes) * len(angles)
    kernels = []
    for kind in range(5):
        for size in sizes:
            for theta in angles:
                kernels.append(to_weight(smooth_feature_template(kind, n, size, theta), gain[kind]))
    return np.stack(kernels, axis=0), np.zeros((5 * variants,), np.float32), variants


def make_spiral_bank(n, sizes, angles, gain):
    variants = len(sizes) * len(angles) * 2
    kernels = []
    for kind in range(3):
        for size in sizes:
            for theta in angles:
                for chirality in (1.0, -1.0):
                    kernels.append(to_weight(spiral_feature_template(kind, n, size, theta, chirality), gain[kind]))
    return np.stack(kernels, axis=0), np.zeros((3 * variants,), np.float32), variants


def make_local_bank(n, sizes, angles, gain):
    variants = len(sizes) * len(angles) * 2
    kernels = []
    for kind in range(2):
        for size in sizes:
            for theta in angles:
                for style in (0, 1):
                    kernels.append(to_weight(local_feature_template(kind, n, size, theta, style), gain[kind]))
    return np.stack(kernels, axis=0), np.zeros((2 * variants,), np.float32), variants


def network_dict(v16, v32, vl):
    return {
        "layers": [
            {"name": "p16", "op": "avgpool", "inputs": ["image"], "size": [8, 8], "stride": [8, 8]},
            {"name": "c16", "op": "conv", "inputs": ["p16"], "out": 5 * v16, "kernel": [16, 16], "stride": 1, "padding": 0},
            {"name": "u16", "op": "relu", "inputs": ["c16"]},
            {"name": "r16", "op": "reshape", "inputs": ["u16"], "shape": [1, 5, v16]},
            {"name": "m16", "op": "maxpool", "inputs": ["r16"], "size": [1, v16], "stride": [1, v16]},
            {"name": "f16", "op": "reshape", "inputs": ["m16"], "shape": [5]},
            {"name": "s16", "op": "dense", "inputs": ["f16"], "out": 10},
            {"name": "p32", "op": "avgpool", "inputs": ["image"], "size": [4, 4], "stride": [4, 4]},
            {"name": "n32", "op": "norm", "inputs": ["p32"]},
            {"name": "c32", "op": "conv", "inputs": ["n32"], "out": 3 * v32, "kernel": [32, 32], "stride": 1, "padding": 0},
            {"name": "u32", "op": "relu", "inputs": ["c32"]},
            {"name": "r32", "op": "reshape", "inputs": ["u32"], "shape": [1, 3, v32]},
            {"name": "m32", "op": "maxpool", "inputs": ["r32"], "size": [1, v32], "stride": [1, v32]},
            {"name": "f32", "op": "reshape", "inputs": ["m32"], "shape": [3]},
            {"name": "s32", "op": "dense", "inputs": ["f32"], "out": 10},
            {"name": "cl", "op": "conv", "inputs": ["n32"], "out": 2 * vl, "kernel": [24, 24], "stride": 1, "padding": 0},
            {"name": "ul", "op": "relu", "inputs": ["cl"]},
            {"name": "pl", "op": "maxpool", "inputs": ["ul"], "size": [9, 9], "stride": [9, 9]},
            {"name": "rl", "op": "reshape", "inputs": ["pl"], "shape": [1, 2, vl]},
            {"name": "ml", "op": "maxpool", "inputs": ["rl"], "size": [1, vl], "stride": [1, vl]},
            {"name": "fl", "op": "reshape", "inputs": ["ml"], "shape": [2]},
            {"name": "sl", "op": "dense", "inputs": ["fl"], "out": 10},
            {"name": "scores", "op": "add", "inputs": ["s16", "s32", "sl"]},
        ]
    }


def build(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    angles = [TAU * i / 8.0 for i in range(8)]
    sizes16 = [0.22, 0.34, 0.48]
    sizes32 = [0.14, 0.24, 0.36]
    sizes_local = [0.16, 0.26]

    c16_W, c16_b, v16 = make_smooth_bank(16, sizes16, angles, gain=np.array([1.0, 1.0, 1.0, 1.05, 1.05], np.float32))
    c32_W, c32_b, v32 = make_spiral_bank(32, sizes32, angles, gain=np.array([1.05, 1.00, 1.00], np.float32))
    cl_W, cl_b, vl = make_local_bank(24, sizes_local, angles, gain=np.array([1.10, 1.12], np.float32))

    s16_W = np.array([
        [-0.50, -0.40, -0.30, -0.20, -0.20],
        [-0.40, -0.30, -0.20, -0.10, -0.10],
        [ 1.20, -0.40, -0.80, -0.40, -0.30],
        [-0.30,  1.00, -0.50, -0.30, -0.20],
        [-0.40, -0.30,  1.10, -0.40, -0.30],
        [-0.50, -0.40, -0.20, -0.20, -0.10],
        [-0.60, -0.40, -0.20, -0.30, -0.20],
        [-0.60, -0.40, -0.20, -0.30, -0.20],
        [-0.30, -0.30, -0.80,  1.10, -0.60],
        [-0.20, -0.20, -0.40,  0.70,  1.00],
    ], np.float32)
    s16_b = np.array([0.00, 0.00, 0.18, 0.10, 0.02, 0.00, 0.00, 0.00, 0.02, 0.08], np.float32)

    s32_W = np.array([
        [ 0.20,  0.50, -0.20],
        [ 0.00,  0.00, -0.10],
        [-0.60, -0.60, -0.30],
        [-0.40, -0.40, -0.20],
        [-0.20, -0.20, -0.10],
        [ 0.50,  0.30,  1.10],
        [ 1.20, -0.50, -0.40],
        [-0.50,  1.20, -0.20],
        [-0.20, -0.20, -0.10],
        [-0.10, -0.10, -0.10],
    ], np.float32)
    s32_b = np.zeros((10,), np.float32)

    sl_W = np.array([
        [ 1.30, -0.40],
        [ 0.20,  1.55],
        [-0.30, -0.30],
        [-0.20, -0.20],
        [-0.10, -0.10],
        [-0.10, -0.10],
        [-0.10, -0.10],
        [ 0.10, -0.10],
        [-0.10, -0.10],
        [-0.10, -0.10],
    ], np.float32)
    sl_b = np.zeros((10,), np.float32)

    net = network_dict(v16, v32, vl)
    (out_dir / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(out_dir / "weights.npz", **{
        "c16.W": c16_W,
        "c16.b": c16_b,
        "s16.W": s16_W,
        "s16.b": s16_b,
        "c32.W": c32_W,
        "c32.b": c32_b,
        "s32.W": s32_W,
        "s32.b": s32_b,
        "cl.W": cl_W,
        "cl.b": cl_b,
        "sl.W": sl_W,
        "sl.b": sl_b,
    })


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
