import json
import sys
from pathlib import Path

import numpy as np

POOL = 32
KERNEL = 28
OUT_POS = POOL - KERNEL + 1
VARIANTS = 16
FAMILY_NAMES = [
    "disturbed_lopsided",
    "disturbed_arc",
    "merger_close",
    "merger_wide",
    "round_smooth",
    "oval_smooth",
    "cigar_smooth",
    "barred_face",
    "barred_inclined",
    "tight_face",
    "tight_inclined",
    "loose_face",
    "loose_inclined",
    "edge_no_bulge",
    "edge_with_bulge",
]
FAMILIES = len(FAMILY_NAMES)

coords = np.arange(KERNEL, dtype=np.float32) - np.float32((KERNEL - 1) / 2)
Y, X = np.meshgrid(coords, coords, indexing="ij")
PI = np.float32(np.pi)


def rot(theta):
    theta = np.float32(theta)
    ct = np.cos(theta).astype(np.float32)
    st = np.sin(theta).astype(np.float32)
    xr = ct * X + st * Y
    yr = -st * X + ct * Y
    return xr.astype(np.float32), yr.astype(np.float32)


def gauss_xy(x, y, sx, sy, cx=0.0, cy=0.0):
    sx = np.float32(max(0.05, float(sx)))
    sy = np.float32(max(0.05, float(sy)))
    cx = np.float32(cx)
    cy = np.float32(cy)
    return np.exp(np.float32(-0.5) * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2)).astype(np.float32)


def gauss_round(s, cx=0.0, cy=0.0):
    return gauss_xy(X, Y, s, s, cx, cy)


def normalize_map(t):
    t = t.astype(np.float32)
    t = t - np.float32(np.mean(t, dtype=np.float64))
    n = float(np.sqrt(np.sum(t * t, dtype=np.float64)))
    if not np.isfinite(n) or n < 1e-8:
        n = 1.0
    return (t / np.float32(n)).astype(np.float32)


def compose(pos, bg):
    pos = pos.astype(np.float32)
    bg = bg.astype(np.float32)
    a = np.float32(np.sum(pos, dtype=np.float64) / (np.sum(bg, dtype=np.float64) + 1e-8))
    return normalize_map(pos - a * bg)


def stack_kernel(lum, warm=None, warm_weight=0.0):
    lum = normalize_map(lum)
    if warm is None or warm_weight == 0.0:
        warm_arr = np.zeros_like(lum, dtype=np.float32)
    else:
        warm_arr = np.float32(warm_weight) * normalize_map(warm)
    k = np.stack([lum, warm_arr], axis=0).astype(np.float32)
    n = float(np.sqrt(np.sum(k * k, dtype=np.float64)))
    if not np.isfinite(n) or n < 1e-8:
        n = 1.0
    return (k / np.float32(n)).astype(np.float32)


def wrap_angle(a):
    return ((a + PI) % (2 * PI) - PI).astype(np.float32)


def sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def dog_round(scale, ratio):
    g1 = gauss_round(scale)
    g2 = gauss_round(scale * ratio)
    return compose(g1, g2)


def dog_ell(theta, scale, q, ratio):
    xr, yr = rot(theta)
    g1 = gauss_xy(xr, yr, scale, scale * q)
    g2 = gauss_xy(xr, yr, scale * ratio, scale * q * ratio)
    return compose(g1, g2)


def warm_ell(theta, scale, q):
    xr, yr = rot(theta)
    core = gauss_xy(xr, yr, 0.45 * scale + 0.7, 0.45 * scale * q + 0.7)
    outer = gauss_xy(xr, yr, 1.15 * scale + 0.8, 1.15 * scale * q + 0.8)
    return compose(core, outer)


def merger(theta, scale, sep_mul, bridge_mul=0.28, tail_mul=0.22):
    xr, yr = rot(theta)
    d = np.float32(sep_mul * scale)
    s = np.float32(0.28 * scale + 0.60)
    pos = (
        1.00 * gauss_xy(xr, yr, s, s, -0.5 * d, 0.0)
        + 0.92 * gauss_xy(xr, yr, 1.05 * s, 0.95 * s, 0.5 * d, 0.0)
        + bridge_mul * gauss_xy(xr, yr, 0.55 * scale + 0.75, 0.16 * scale + 0.40)
        + tail_mul * gauss_xy(xr, yr, 0.48 * scale + 0.70, 0.18 * scale + 0.36, 0.95 * d, 0.30 * scale)
    )
    bg = gauss_round(1.85 * scale + 0.9)
    return compose(pos, bg)


def disturbed_lopsided(theta, scale):
    xr, yr = rot(theta)
    pos = (
        0.95 * gauss_xy(xr, yr, 0.40 * scale + 0.72, 0.34 * scale + 0.70)
        + 0.82 * gauss_xy(xr, yr, 0.28 * scale + 0.58, 0.24 * scale + 0.52, -0.82 * scale, 0.16 * scale)
        + 0.48 * gauss_xy(xr, yr, 0.50 * scale + 0.64, 0.17 * scale + 0.38, 1.00 * scale, -0.16 * scale)
        + 0.34 * gauss_xy(xr, yr, 0.24 * scale + 0.52, 0.24 * scale + 0.52, 0.35 * scale, 0.62 * scale)
    )
    bg = gauss_round(1.90 * scale + 0.9)
    return compose(pos, bg)


def one_arm_mask(theta, scale, pitch_deg=42.0, q=0.9):
    xr, yr = rot(theta)
    q = np.float32(max(0.25, float(q)))
    yi = yr / q
    r = np.sqrt(xr * xr + yi * yi).astype(np.float32) + np.float32(1e-6)
    phi = np.arctan2(yi, xr).astype(np.float32)
    pitch = np.deg2rad(np.float32(pitch_deg)).astype(np.float32)
    target = np.log1p(r / np.float32(0.52 * scale + 1e-6)).astype(np.float32) / np.tan(pitch)
    d = wrap_angle(phi - target)
    width = np.float32(0.18 * scale + 0.70)
    arm = np.exp(np.float32(-0.5) * ((r * d) / width) ** 2).astype(np.float32)
    arm *= sigmoid((r - np.float32(0.40 * scale)) / np.float32(0.10 * scale + 0.35))
    arm *= np.exp(np.float32(-0.5) * ((r - np.float32(1.18 * scale)) / np.float32(0.54 * scale + 0.65)) ** 2).astype(np.float32)
    return arm.astype(np.float32)


def disturbed_arc(theta, scale):
    xr, yr = rot(theta)
    arm = one_arm_mask(theta, scale, pitch_deg=44.0, q=0.82)
    pos = (
        0.58 * gauss_xy(xr, yr, 0.30 * scale + 0.65, 0.30 * scale + 0.65)
        + 0.98 * arm
        + 0.36 * gauss_xy(xr, yr, 0.22 * scale + 0.54, 0.22 * scale + 0.54, 1.02 * scale, 0.02 * scale)
    )
    bg = gauss_round(1.85 * scale + 0.95)
    return compose(pos, bg)


def spiral_parts(theta, scale, pitch_deg, q=1.0, hand=1.0, bar=0.0, bulge=0.45, arm_gain=1.0):
    xr, yr = rot(theta)
    q = np.float32(max(0.25, float(q)))
    yi = yr / q
    r = np.sqrt(xr * xr + yi * yi).astype(np.float32) + np.float32(1e-6)
    phi = np.arctan2(yi, xr).astype(np.float32)
    pitch = np.deg2rad(np.float32(pitch_deg)).astype(np.float32)
    target = np.float32(hand) * np.log1p(r / np.float32(0.48 * scale + 1e-6)).astype(np.float32) / np.tan(pitch)
    width = np.float32(0.18 * scale + 0.68)
    arm = np.zeros_like(r, dtype=np.float32)
    for off in (0.0, float(np.pi)):
        d = wrap_angle(phi - (target + np.float32(off)))
        arm = np.maximum(arm, np.exp(np.float32(-0.5) * ((r * d) / width) ** 2).astype(np.float32))
    arm *= sigmoid((r - np.float32(0.36 * scale)) / np.float32(0.10 * scale + 0.35))
    arm *= np.exp(np.float32(-0.5) * ((r - np.float32(1.15 * scale)) / np.float32(0.55 * scale + 0.72)) ** 2).astype(np.float32)
    bulge_map = gauss_xy(xr, yr, 0.25 * scale + 0.72, max(0.58, q * (0.25 * scale + 0.72)))
    disk_map = gauss_xy(xr, yr, 0.90 * scale + 0.78, max(0.60, q * (0.78 * scale + 0.72)))
    bar_map = gauss_xy(xr, yr, 0.78 * scale + 0.78, 0.13 * scale + 0.58) if bar > 0.0 else np.zeros_like(arm)
    lum_pos = np.float32(arm_gain) * arm + np.float32(bulge) * bulge_map + 0.14 * disk_map + np.float32(bar) * 0.80 * bar_map
    bg = gauss_round(1.86 * scale + 0.95)
    lum = compose(lum_pos, bg)
    warm = 0.92 * bulge_map + np.float32(bar) * 0.32 * bar_map - 0.62 * arm - 0.08 * disk_map
    return lum.astype(np.float32), normalize_map(warm)


def edge_no_bulge(theta, scale, q):
    xr, yr = rot(theta)
    sx = np.float32(1.06 * scale + 0.62)
    sy = np.float32(q * scale + 0.34)
    pos = (
        1.00 * gauss_xy(xr, yr, sx, sy)
        + 0.30 * gauss_xy(xr, yr, 0.54 * scale + 0.62, 0.78 * sy)
        - 0.16 * gauss_round(0.34 * scale + 0.62)
    )
    bg = gauss_round(1.88 * scale + 0.95)
    lum = compose(pos, bg)
    warm = normalize_map(-0.70 * gauss_xy(xr, yr, 0.82 * scale + 0.62, 0.90 * sy) + 0.18 * gauss_round(0.70 * scale + 0.62))
    return lum, warm


def edge_with_bulge(theta, scale, q):
    xr, yr = rot(theta)
    sx = np.float32(1.04 * scale + 0.62)
    sy = np.float32(q * scale + 0.36)
    bulge = gauss_round(0.34 * scale + 0.72)
    pos = (
        0.92 * gauss_xy(xr, yr, sx, sy)
        + 0.30 * gauss_xy(xr, yr, 0.52 * scale + 0.62, 0.80 * sy)
        + 0.72 * bulge
    )
    bg = gauss_round(1.90 * scale + 0.95)
    lum = compose(pos, bg)
    warm = normalize_map(0.95 * bulge - 0.24 * gauss_xy(xr, yr, 0.92 * scale + 0.62, 0.95 * sy))
    return lum, warm


def family_templates(name):
    kernels, biases = [], []
    if name == "disturbed_lopsided":
        scales = [2.5, 3.4, 4.6, 6.0]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                kernels.append(stack_kernel(disturbed_lopsided(a, s), None, 0.0))
                biases.append(-4.0)
    elif name == "disturbed_arc":
        scales = [2.6, 3.6, 4.8, 6.2]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                kernels.append(stack_kernel(disturbed_arc(a, s), None, 0.0))
                biases.append(-4.0)
    elif name == "merger_close":
        scales = [2.4, 3.4, 4.6, 6.0]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                kernels.append(stack_kernel(merger(a, s, sep_mul=1.25, bridge_mul=0.34, tail_mul=0.18), None, 0.0))
                biases.append(-4.0)
    elif name == "merger_wide":
        scales = [2.6, 3.7, 4.9, 6.4]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                kernels.append(stack_kernel(merger(a, s, sep_mul=1.65, bridge_mul=0.22, tail_mul=0.24), None, 0.0))
                biases.append(-4.0)
    elif name == "round_smooth":
        scales = [1.7, 2.4, 3.4, 4.8]
        ratios = [1.8, 2.2, 2.8, 3.5]
        for s in scales:
            for r in ratios:
                lum = dog_round(s, r)
                warm = warm_ell(0.0, s, 1.0)
                kernels.append(stack_kernel(lum, warm, 0.25))
                biases.append(-3.5)
    elif name == "oval_smooth":
        scales = [2.3, 3.4, 4.8, 6.2]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                lum = dog_ell(a, s, q=0.70, ratio=2.25)
                warm = warm_ell(a, s, 0.70)
                kernels.append(stack_kernel(lum, warm, 0.22))
                biases.append(-3.5)
    elif name == "cigar_smooth":
        scales = [2.7, 3.8, 5.2, 6.8]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                lum = dog_ell(a, s, q=0.34, ratio=2.35)
                warm = warm_ell(a, s, 0.34)
                kernels.append(stack_kernel(lum, warm, 0.18))
                biases.append(-3.5)
    elif name == "barred_face":
        scales = [3.6, 5.0]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                for hand in (-1.0, 1.0):
                    lum, warm = spiral_parts(a, s, pitch_deg=23.0, q=1.0, hand=hand, bar=1.0, bulge=0.40, arm_gain=0.90)
                    kernels.append(stack_kernel(lum, warm, 0.30))
                    biases.append(-3.8)
    elif name == "barred_inclined":
        scales = [3.8, 5.2]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                for hand in (-1.0, 1.0):
                    lum, warm = spiral_parts(a, s, pitch_deg=24.0, q=0.68, hand=hand, bar=1.0, bulge=0.44, arm_gain=0.86)
                    kernels.append(stack_kernel(lum, warm, 0.28))
                    biases.append(-3.8)
    elif name == "tight_face":
        scales = [3.6, 5.0]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                for hand in (-1.0, 1.0):
                    lum, warm = spiral_parts(a, s, pitch_deg=16.0, q=1.0, hand=hand, bar=0.0, bulge=0.56, arm_gain=0.90)
                    kernels.append(stack_kernel(lum, warm, 0.26))
                    biases.append(-3.8)
    elif name == "tight_inclined":
        scales = [3.8, 5.2]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                for hand in (-1.0, 1.0):
                    lum, warm = spiral_parts(a, s, pitch_deg=17.0, q=0.68, hand=hand, bar=0.0, bulge=0.60, arm_gain=0.85)
                    kernels.append(stack_kernel(lum, warm, 0.25))
                    biases.append(-3.8)
    elif name == "loose_face":
        scales = [3.6, 5.0]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                for hand in (-1.0, 1.0):
                    lum, warm = spiral_parts(a, s, pitch_deg=31.0, q=1.0, hand=hand, bar=0.0, bulge=0.34, arm_gain=1.02)
                    kernels.append(stack_kernel(lum, warm, 0.28))
                    biases.append(-3.8)
    elif name == "loose_inclined":
        scales = [3.8, 5.2]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                for hand in (-1.0, 1.0):
                    lum, warm = spiral_parts(a, s, pitch_deg=31.5, q=0.68, hand=hand, bar=0.0, bulge=0.36, arm_gain=0.96)
                    kernels.append(stack_kernel(lum, warm, 0.26))
                    biases.append(-3.8)
    elif name == "edge_no_bulge":
        scales = [3.8, 5.0, 6.4, 7.8]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                lum, warm = edge_no_bulge(a, s, q=0.12)
                kernels.append(stack_kernel(lum, warm, 0.16))
                biases.append(-3.6)
    elif name == "edge_with_bulge":
        scales = [3.8, 5.0, 6.4, 7.8]
        angles = np.linspace(0.0, np.pi, 4, endpoint=False)
        for s in scales:
            for a in angles:
                lum, warm = edge_with_bulge(a, s, q=0.12)
                kernels.append(stack_kernel(lum, warm, 0.20))
                biases.append(-3.6)
    else:
        raise ValueError(name)
    if len(kernels) != VARIANTS:
        raise ValueError(f"{name} produced {len(kernels)} templates, expected {VARIANTS}")
    return kernels, biases


def build_templates():
    kernels, biases = [], []
    for name in FAMILY_NAMES:
        k, b = family_templates(name)
        kernels.extend(k)
        biases.extend(b)
    W = np.stack(kernels, axis=0).astype(np.float32)
    b = np.array(biases, dtype=np.float32)
    return W, b


def evidence_weights():
    W = np.zeros((10, FAMILIES), dtype=np.float32)
    b = np.zeros((10,), dtype=np.float32)

    # 0 disturbed
    W[0, 0] = 1.00
    W[0, 1] = 0.95
    W[0, 2] = 0.22
    W[0, 3] = 0.18
    W[0, 11] = 0.18
    W[0, 12] = 0.14
    W[0, 4] = -0.30
    W[0, 5] = -0.24
    W[0, 6] = -0.14
    W[0, 13] = -0.18
    W[0, 14] = -0.12
    b[0] = -0.20

    # 1 merging
    W[1, 0] = 0.24
    W[1, 1] = 0.20
    W[1, 2] = 1.00
    W[1, 3] = 0.96
    W[1, 4] = -0.22
    W[1, 5] = -0.18
    W[1, 13] = -0.12
    W[1, 14] = -0.10
    b[1] = -0.20

    # 2 round smooth
    W[2, 4] = 1.00
    W[2, 5] = 0.20
    W[2, 6] = -0.20
    W[2, 7] = -0.18
    W[2, 8] = -0.18
    W[2, 9] = -0.18
    W[2, 10] = -0.18
    W[2, 11] = -0.18
    W[2, 12] = -0.18
    W[2, 13] = -0.22
    W[2, 14] = -0.16
    b[2] = -0.10

    # 3 in-between round smooth
    W[3, 4] = 0.22
    W[3, 5] = 1.00
    W[3, 6] = 0.14
    W[3, 7] = -0.14
    W[3, 8] = -0.14
    W[3, 9] = -0.14
    W[3, 10] = -0.14
    W[3, 11] = -0.14
    W[3, 12] = -0.14
    W[3, 13] = -0.12
    W[3, 14] = -0.08
    b[3] = -0.10

    # 4 cigar smooth
    W[4, 5] = 0.16
    W[4, 6] = 1.00
    W[4, 13] = -0.42
    W[4, 14] = -0.24
    W[4, 7] = -0.12
    W[4, 8] = -0.12
    W[4, 9] = -0.12
    W[4, 10] = -0.12
    W[4, 11] = -0.12
    W[4, 12] = -0.12
    b[4] = -0.10

    # 5 barred spiral
    W[5, 0] = 0.08
    W[5, 1] = 0.10
    W[5, 7] = 1.00
    W[5, 8] = 0.98
    W[5, 9] = 0.18
    W[5, 10] = 0.16
    W[5, 11] = -0.16
    W[5, 12] = -0.14
    W[5, 4] = -0.16
    W[5, 5] = -0.16
    W[5, 6] = -0.12
    W[5, 13] = -0.10
    W[5, 14] = -0.10
    b[5] = -0.05

    # 6 tight spiral
    W[6, 0] = 0.06
    W[6, 1] = 0.06
    W[6, 7] = 0.18
    W[6, 8] = 0.18
    W[6, 9] = 1.00
    W[6, 10] = 0.98
    W[6, 11] = -0.10
    W[6, 12] = -0.08
    W[6, 4] = -0.14
    W[6, 5] = -0.14
    W[6, 6] = -0.10
    W[6, 13] = -0.10
    W[6, 14] = -0.10
    b[6] = -0.05

    # 7 loose spiral
    W[7, 0] = 0.16
    W[7, 1] = 0.18
    W[7, 7] = -0.12
    W[7, 8] = -0.10
    W[7, 9] = -0.18
    W[7, 10] = -0.16
    W[7, 11] = 1.00
    W[7, 12] = 0.98
    W[7, 4] = -0.12
    W[7, 5] = -0.12
    W[7, 6] = -0.10
    W[7, 13] = -0.10
    W[7, 14] = -0.10
    b[7] = -0.05

    # 8 edge without bulge
    W[8, 4] = -0.18
    W[8, 5] = -0.12
    W[8, 6] = -0.24
    W[8, 7] = -0.10
    W[8, 8] = -0.10
    W[8, 9] = -0.10
    W[8, 10] = -0.10
    W[8, 11] = -0.10
    W[8, 12] = -0.10
    W[8, 13] = 1.00
    W[8, 14] = -0.42
    b[8] = -0.08

    # 9 edge with bulge
    W[9, 4] = -0.08
    W[9, 5] = 0.08
    W[9, 6] = -0.16
    W[9, 7] = -0.10
    W[9, 8] = -0.10
    W[9, 9] = -0.10
    W[9, 10] = -0.10
    W[9, 11] = -0.10
    W[9, 12] = -0.10
    W[9, 13] = -0.26
    W[9, 14] = 1.00
    b[9] = -0.08

    return W, b


def score_weights():
    W = np.eye(10, dtype=np.float32)
    b = np.array([0.00, 0.02, 0.12, 0.10, -0.18, 0.08, 0.06, 0.10, 0.04, 0.06], dtype=np.float32)
    return W, b


def network_json():
    return {
        "layers": [
            {"name": "pool", "op": "avgpool", "inputs": ["image"], "size": [4, 4], "stride": [4, 4]},
            {"name": "prep", "op": "conv", "inputs": ["pool"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "prep_norm", "op": "norm", "inputs": ["prep"]},
            {"name": "templates", "op": "conv", "inputs": ["prep_norm"], "out": FAMILIES * VARIANTS, "kernel": [KERNEL, KERNEL], "stride": 1, "padding": 0},
            {"name": "templates_relu", "op": "relu", "inputs": ["templates"]},
            {"name": "spatial_max", "op": "maxpool", "inputs": ["templates_relu"], "size": [OUT_POS, OUT_POS], "stride": [OUT_POS, OUT_POS]},
            {"name": "family_grid", "op": "reshape", "inputs": ["spatial_max"], "shape": [FAMILIES, 1, VARIANTS]},
            {"name": "family_max", "op": "maxpool", "inputs": ["family_grid"], "size": [1, VARIANTS], "stride": [1, VARIANTS]},
            {"name": "family", "op": "reshape", "inputs": ["family_max"], "shape": [FAMILIES]},
            {"name": "evidence", "op": "dense", "inputs": ["family"], "out": 10},
            {"name": "evidence_relu", "op": "relu", "inputs": ["evidence"]},
            {"name": "scores", "op": "dense", "inputs": ["evidence_relu"], "out": 10},
        ]
    }


def build(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prep_w = np.array([
        [[[1.0 / 3.0]], [[1.0 / 3.0]], [[1.0 / 3.0]]],
        [[[1.0]], [[0.0]], [[-1.0]]],
    ], dtype=np.float32)
    prep_b = np.zeros((2,), dtype=np.float32)
    tpl_w, tpl_b = build_templates()
    ev_w, ev_b = evidence_weights()
    sc_w, sc_b = score_weights()

    weights = {
        "prep.W": prep_w,
        "prep.b": prep_b,
        "templates.W": tpl_w,
        "templates.b": tpl_b,
        "evidence.W": ev_w,
        "evidence.b": ev_b,
        "scores.W": sc_w,
        "scores.b": sc_b,
    }

    with (out / "network.json").open("w") as f:
        json.dump(network_json(), f)
    np.savez(out / "weights.npz", **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
