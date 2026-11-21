import json
import sys
from pathlib import Path

import numpy as np

K = 15
R0 = (K - 1) / 2.0
YY, XX = np.mgrid[0:K, 0:K].astype(np.float32)
XX -= R0
YY -= R0
RR = np.sqrt(XX * XX + YY * YY) + 1e-6
PHI = np.arctan2(YY, XX)
GROUPS = ["round", "ellipse", "cigar", "edge0", "edge1", "merger", "disturb", "tight", "loose", "barred"]
G2I = {g: i for i, g in enumerate(GROUPS)}


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def gauss(sx, sy=None, theta=0.0, x0=0.0, y0=0.0):
    if sy is None:
        sy = sx
    dx = XX - x0
    dy = YY - y0
    ct = np.cos(theta)
    st = np.sin(theta)
    u = dx * ct + dy * st
    v = -dx * st + dy * ct
    return np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2)).astype(np.float32)


def shape_from(p, bg=6.8):
    p = p.astype(np.float32)
    b = gauss(bg)
    p = p - b * (p.sum() / (b.sum() + 1e-9))
    n = np.sqrt(np.sum(p * p)) + 1e-9
    return (p / n).astype(np.float32)


def round_pat(s):
    return shape_from(gauss(s))


def ellipse_pat(su, sv, theta):
    return shape_from(gauss(su, sv, theta))


def line_pat(theta, su, sv, bulge=0.0):
    return shape_from(gauss(su, sv, theta) + bulge * gauss(1.35))


def two_blob_pat(theta, d, s):
    ct = np.cos(theta)
    st = np.sin(theta)
    p = gauss(s, x0=0.5 * d * ct, y0=0.5 * d * st)
    p += gauss(s, x0=-0.5 * d * ct, y0=-0.5 * d * st)
    p += 0.18 * gauss(2.8)
    return shape_from(p)


def lopsided_pat(theta, d, s1, s2):
    ct = np.cos(theta)
    st = np.sin(theta)
    p = 0.65 * gauss(s1)
    p += 1.00 * gauss(s2, x0=d * ct, y0=d * st)
    p += 0.35 * gauss(1.7 * s2, x0=-0.55 * d * ct, y0=-0.55 * d * st)
    return shape_from(p)


def spiral_pat(a, hand, phase, sigma, r0, rw, bulge=0.0, bar=0.0):
    target = hand * a * np.log(RR + 1.7) + phase
    d1 = wrap(PHI - target)
    d2 = wrap(PHI - target - np.pi)
    arms = np.exp(-0.5 * (d1 / sigma) ** 2) + np.exp(-0.5 * (d2 / sigma) ** 2)
    env = np.exp(-0.5 * ((RR - r0) / rw) ** 2) * (1.0 - np.exp(-0.5 * (RR / 1.35) ** 2))
    p = arms * env + bulge * gauss(1.55)
    if bar:
        p += bar * gauss(3.6, 0.85, phase)
    return shape_from(p)


def rgb_to_lc_weights():
    W = np.zeros((2, 3, 1, 1), np.float32)
    W[0, :, 0, 0] = np.array([1.0, 1.0, 1.0], np.float32) / 3.0
    W[1, :, 0, 0] = np.array([-0.55, 0.0, 0.55], np.float32)
    b = np.zeros(2, np.float32)
    return W, b


def pack_filter(pat, color):
    W = np.zeros((2, K, K), np.float32)
    W[0] = pat
    W[1] = color * pat
    return W


def make_bank(scale):
    feats, meta = [], []

    def add(group, pat, color):
        feats.append(pack_filter(pat, color))
        meta.append(group)

    if scale == 32:
        round_sizes = [2.0, 3.0]
        ell = (3.5, 2.2)
        cig = (4.8, 1.35)
        edge = (5.4, 0.95)
        merger_d, merger_s = 4.8, 1.35
        disturb_d, disturb_s1, disturb_s2 = 3.8, 2.5, 1.55
        tight = dict(a=2.35, sigma=0.23, r0=4.8, rw=2.4, bulge=0.42, bar=0.0)
        loose = dict(a=1.05, sigma=0.30, r0=4.9, rw=2.8, bulge=0.18, bar=0.0)
        barred = dict(a=1.35, sigma=0.28, r0=4.7, rw=2.5, bulge=0.24, bar=0.88)
    else:
        round_sizes = [1.35, 2.0]
        ell = (2.25, 1.55)
        cig = (3.05, 0.95)
        edge = (3.45, 0.65)
        merger_d, merger_s = 3.25, 0.95
        disturb_d, disturb_s1, disturb_s2 = 2.8, 1.75, 1.10
        tight = dict(a=2.35, sigma=0.24, r0=3.0, rw=1.7, bulge=0.36, bar=0.0)
        loose = dict(a=1.00, sigma=0.31, r0=3.0, rw=2.0, bulge=0.14, bar=0.0)
        barred = dict(a=1.25, sigma=0.29, r0=3.0, rw=1.8, bulge=0.21, bar=0.72)

    for s in round_sizes:
        add("round", round_pat(s), -0.22)

    for theta in np.linspace(0.0, np.pi, 8, endpoint=False):
        add("ellipse", ellipse_pat(ell[0], ell[1], theta), -0.14)
        add("cigar", ellipse_pat(cig[0], cig[1], theta), -0.09)
        add("edge0", line_pat(theta, edge[0], edge[1], 0.0), 0.16)
        add("edge1", line_pat(theta, edge[0], edge[1], 0.78), -0.05)
        add("merger", two_blob_pat(theta, merger_d, merger_s), 0.05)
        add("disturb", lopsided_pat(theta, disturb_d, disturb_s1, disturb_s2), 0.10)

    for hand in (-1.0, 1.0):
        for phase in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            add("tight", spiral_pat(hand=hand, phase=phase, **tight), 0.14)
            add("loose", spiral_pat(hand=hand, phase=phase, **loose), 0.16)
            add("barred", spiral_pat(hand=hand, phase=phase, **barred), 0.14)

    W = np.stack(feats).astype(np.float32)
    b = np.zeros(len(feats), np.float32)
    return W, b, meta


def mix_weights(meta):
    W = np.zeros((len(GROUPS), len(meta), 1, 1), np.float32)
    b = np.zeros(len(GROUPS), np.float32)
    for g in GROUPS:
        idx = [i for i, m in enumerate(meta) if m == g]
        s = 1.0 / np.sqrt(float(len(idx)))
        for i in idx:
            W[G2I[g], i, 0, 0] = s
    return W, b


def pad_weights(offset):
    W = np.zeros((20, 10, 1, 1), np.float32)
    b = np.zeros(20, np.float32)
    for i in range(10):
        W[offset + i, i, 0, 0] = 1.0
    return W, b


def dense_weights():
    coef = np.array([
        [-0.35, -0.30, -0.20, -0.15, -0.15,  0.70,  1.60, -0.05,  0.15,  0.00],
        [-0.35, -0.20, -0.15, -0.10, -0.10,  1.70,  0.45, -0.05,  0.00, -0.05],
        [ 0.05,  0.35,  0.00, -0.25, -0.25, -0.35, -0.35, -0.35, -0.35, -0.30],
        [-0.05,  1.40,  0.30, -0.15, -0.10, -0.20, -0.20, -0.25, -0.25, -0.25],
        [-0.10,  0.40,  1.55,  0.20,  0.10, -0.20, -0.20, -0.25, -0.25, -0.15],
        [-0.20, -0.20, -0.10, -0.10, -0.10,  0.10,  0.05,  0.20,  0.35,  1.80],
        [-0.10, -0.15, -0.05, -0.05, -0.05, -0.10, -0.10,  1.75,  0.15, -0.35],
        [-0.10, -0.15, -0.05, -0.05, -0.05, -0.10, -0.05,  0.00,  1.70,  0.10],
        [-0.10, -0.10,  0.20,  1.80, -0.25, -0.15, -0.10, -0.20, -0.20, -0.20],
        [-0.10, -0.10,  0.10, -0.15,  1.80, -0.15, -0.10, -0.20, -0.20, -0.10],
    ], dtype=np.float32)
    b = np.array([-0.05, -0.05, 0.02, 0.00, 0.00, -0.05, 0.00, 0.00, 0.00, 0.00], np.float32)
    branch = {
        0: np.array([1.00, 1.00, 1.00, 1.05, 1.05, 1.20, 1.15, 1.30, 1.25, 1.35], np.float32),
        1: np.array([1.10, 1.08, 1.02, 0.92, 0.92, 0.82, 0.82, 0.72, 0.78, 0.72], np.float32),
    }
    W = np.zeros((10, 20), np.float32)
    for out in range(10):
        for g in range(10):
            W[out, g] = coef[out, g] * branch[0][g]
            W[out, 10 + g] = coef[out, g] * branch[1][g]
    return W, b


def build(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw32_W, raw32_b, meta32 = make_bank(32)
    raw16_W, raw16_b, meta16 = make_bank(16)
    mix32_W, mix32_b = mix_weights(meta32)
    mix16_W, mix16_b = mix_weights(meta16)
    rgb_W, rgb_b = rgb_to_lc_weights()
    pad32_W, pad32_b = pad_weights(0)
    pad16_W, pad16_b = pad_weights(10)
    score_W, score_b = dense_weights()

    layers = [
        {"name": "p32", "op": "avgpool", "inputs": ["image"], "size": [4, 4], "stride": [4, 4]},
        {"name": "d32", "op": "conv", "inputs": ["p32"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "b32", "op": "conv", "inputs": ["d32"], "out": int(raw32_W.shape[0]), "kernel": [K, K], "stride": 1, "padding": K // 2},
        {"name": "r32", "op": "relu", "inputs": ["b32"]},
        {"name": "g32", "op": "conv", "inputs": ["r32"], "out": 10, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "m32a", "op": "maxpool", "inputs": ["g32"], "size": [2, 2], "stride": [2, 2]},
        {"name": "m32b", "op": "maxpool", "inputs": ["m32a"], "size": [2, 2], "stride": [2, 2]},
        {"name": "m32c", "op": "maxpool", "inputs": ["m32b"], "size": [2, 2], "stride": [2, 2]},
        {"name": "m32d", "op": "maxpool", "inputs": ["m32c"], "size": [2, 2], "stride": [2, 2]},
        {"name": "m32e", "op": "maxpool", "inputs": ["m32d"], "size": [2, 2], "stride": [2, 2]},
        {"name": "p16", "op": "avgpool", "inputs": ["image"], "size": [8, 8], "stride": [8, 8]},
        {"name": "d16", "op": "conv", "inputs": ["p16"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "b16", "op": "conv", "inputs": ["d16"], "out": int(raw16_W.shape[0]), "kernel": [K, K], "stride": 1, "padding": K // 2},
        {"name": "r16", "op": "relu", "inputs": ["b16"]},
        {"name": "g16", "op": "conv", "inputs": ["r16"], "out": 10, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "m16a", "op": "maxpool", "inputs": ["g16"], "size": [2, 2], "stride": [2, 2]},
        {"name": "m16b", "op": "maxpool", "inputs": ["m16a"], "size": [2, 2], "stride": [2, 2]},
        {"name": "m16c", "op": "maxpool", "inputs": ["m16b"], "size": [2, 2], "stride": [2, 2]},
        {"name": "m16d", "op": "maxpool", "inputs": ["m16c"], "size": [2, 2], "stride": [2, 2]},
        {"name": "pad32", "op": "conv", "inputs": ["m32e"], "out": 20, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "pad16", "op": "conv", "inputs": ["m16d"], "out": 20, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "feat", "op": "add", "inputs": ["pad32", "pad16"]},
        {"name": "flat", "op": "reshape", "inputs": ["feat"], "shape": [20]},
        {"name": "scores", "op": "dense", "inputs": ["flat"], "out": 10},
    ]
    (out / "network.json").write_text(json.dumps({"layers": layers}, indent=2))

    np.savez(
        out / "weights.npz",
        **{
            "d32.W": rgb_W,
            "d32.b": rgb_b,
            "b32.W": raw32_W,
            "b32.b": raw32_b,
            "g32.W": mix32_W,
            "g32.b": mix32_b,
            "d16.W": rgb_W,
            "d16.b": rgb_b,
            "b16.W": raw16_W,
            "b16.b": raw16_b,
            "g16.W": mix16_W,
            "g16.b": mix16_b,
            "pad32.W": pad32_W,
            "pad32.b": pad32_b,
            "pad16.W": pad16_W,
            "pad16.b": pad16_b,
            "scores.W": score_W,
            "scores.b": score_b,
        }
    )


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
