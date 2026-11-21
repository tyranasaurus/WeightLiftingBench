import json
import sys
from pathlib import Path

import numpy as np

IMG = 28
DRAW_SCALE = 10.0
BASE_STROKE_PX = 1.75
BASE_SIGMA = BASE_STROKE_PX / DRAW_SCALE
NORM_EPS = 1e-5
EDGE_FILTERS = 8
JITTER_VARIANTS = 4
TRANSFORM_COUNT = 8
RAW7_BG = 0.16
RAW4_BG = 0.12

BRANCH_SCALE_14 = 1.20
BRANCH_SCALE_7 = 0.95
BRANCH_SCALE_RAW7 = 1.00
BRANCH_SCALE_28 = 0.65
BRANCH_SCALE_EDGE = 0.75
BRANCH_SCALE_QUAD = 0.60

# (shear_x_from_y, sx, sy, dx, dy, stroke_scale)
TRANSFORMS = [
    (0.00, 1.00, 1.00, 0.00, 0.00, 1.00),
    (-0.18, 1.00, 1.00, 0.00, 0.00, 1.00),
    (0.18, 1.00, 1.00, 0.00, 0.00, 1.00),
    (0.00, 0.93, 1.08, 0.00, 0.00, 0.95),
    (0.00, 1.08, 0.93, 0.00, 0.00, 1.08),
    (-0.07, 0.98, 1.02, 0.00, -0.03, 0.90),
    (0.07, 1.02, 0.98, 0.00, 0.03, 1.12),
    (0.00, 0.97, 1.03, 0.00, 0.00, 1.22),
]

FINE_VARIANTS = 8  # 4 base styles × 2 deterministic jitters
EDGE_VARIANTS = 8
QUAD_VARIANTS = 4
COARSE_PER_CLASS = 4 * JITTER_VARIANTS * TRANSFORM_COUNT

FINAL_BIAS = np.array([
    0.02, 0.06, 0.00, 0.00, 0.02,
    0.00, 0.04, 0.08, -0.12, -0.03,
], dtype=np.float32)


def pts(data):
    return np.asarray(data, dtype=np.float32)


def closed(poly):
    poly = pts(poly)
    if len(poly) and np.any(poly[0] != poly[-1]):
        poly = np.concatenate([poly, poly[:1]], axis=0)
    return poly


def ellipse_poly(cx, cy, rx, ry, n=72):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    poly = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)
    return closed(poly)


DIGIT_STYLES = {
    0: [
        [ellipse_poly(0.00, 0.00, 0.56, 0.82)],
        [ellipse_poly(0.00, 0.00, 0.48, 0.86)],
        [ellipse_poly(0.00, 0.00, 0.64, 0.74)],
        [closed([
            [-0.18, -0.90], [0.22, -0.90], [0.54, -0.60], [0.60, -0.10],
            [0.52, 0.50], [0.24, 0.88], [-0.20, 0.90], [-0.52, 0.58],
            [-0.60, 0.06], [-0.54, -0.54],
        ])],
    ],
    1: [
        [pts([[-0.02, -0.86], [0.00, 0.88]])],
        [pts([[-0.28, -0.56], [-0.08, -0.86], [0.02, 0.88]])],
        [pts([[-0.18, -0.84], [0.06, -0.88], [0.02, 0.76], [-0.24, 0.88]])],
        [pts([[-0.16, -0.86], [0.16, 0.88]])],
    ],
    2: [
        [pts([
            [-0.54, -0.46], [-0.26, -0.82], [0.20, -0.82], [0.56, -0.52],
            [0.48, -0.14], [0.18, 0.18], [-0.14, 0.42], [-0.42, 0.66],
            [-0.56, 0.88], [0.56, 0.88],
        ])],
        [pts([
            [-0.46, -0.70], [-0.06, -0.86], [0.34, -0.72], [0.46, -0.42],
            [0.30, -0.10], [-0.10, 0.20], [-0.42, 0.46], [-0.56, 0.88],
            [0.54, 0.88],
        ])],
        [pts([
            [-0.48, -0.56], [-0.18, -0.84], [0.18, -0.86], [0.50, -0.60],
            [0.52, -0.30], [0.26, 0.02], [-0.10, 0.28], [-0.44, 0.54],
            [-0.58, 0.86], [0.58, 0.86],
        ])],
        [pts([
            [-0.50, -0.34], [-0.24, -0.70], [0.18, -0.84], [0.54, -0.58],
            [0.44, -0.16], [0.10, 0.18], [-0.24, 0.46], [-0.54, 0.76],
            [-0.54, 0.88], [0.56, 0.88],
        ])],
    ],
    3: [
        [pts([
            [-0.44, -0.78], [-0.06, -0.86], [0.34, -0.78], [0.56, -0.46],
            [0.28, -0.10], [0.02, 0.00], [0.30, 0.08], [0.58, 0.42],
            [0.34, 0.78], [-0.06, 0.88], [-0.44, 0.76],
        ])],
        [pts([
            [-0.30, -0.84], [0.18, -0.84], [0.56, -0.52], [0.26, -0.10],
            [0.00, 0.00], [0.28, 0.12], [0.60, 0.50], [0.24, 0.86],
            [-0.26, 0.86],
        ])],
        [pts([
            [-0.38, -0.78], [0.34, -0.78], [0.06, -0.18], [0.36, -0.06],
            [0.58, 0.52], [0.00, 0.82], [-0.40, 0.78],
        ])],
        [pts([
            [-0.22, -0.84], [0.26, -0.78], [0.50, -0.54], [0.26, -0.20],
            [-0.04, -0.04], [0.28, 0.12], [0.52, 0.46], [0.28, 0.78],
            [-0.20, 0.86],
        ])],
    ],
    4: [
        [pts([[-0.46, 0.10], [0.18, -0.84], [0.18, 0.88]]), pts([[-0.46, 0.10], [0.50, 0.10]])],
        [pts([[-0.32, -0.36], [0.20, -0.84], [0.20, 0.86]]), pts([[-0.52, 0.08], [0.52, 0.08]])],
        [pts([[-0.52, -0.02], [0.26, -0.82]]), pts([[0.26, -0.82], [0.26, 0.88]]), pts([[-0.50, 0.10], [0.54, 0.10]])],
        [pts([[-0.44, -0.10], [0.14, -0.82], [0.30, 0.16]]), pts([[0.30, -0.82], [0.30, 0.88]]), pts([[-0.44, 0.16], [0.54, 0.16]])],
    ],
    5: [
        [pts([
            [0.50, -0.82], [-0.34, -0.82], [-0.46, -0.18], [0.18, -0.18],
            [0.52, 0.10], [0.38, 0.60], [0.00, 0.88], [-0.46, 0.72],
        ])],
        [pts([
            [0.46, -0.82], [-0.26, -0.82], [-0.40, -0.34], [-0.06, -0.22],
            [0.26, -0.18], [0.54, 0.14], [0.34, 0.62], [-0.04, 0.86],
            [-0.46, 0.78],
        ])],
        [pts([
            [0.54, -0.80], [-0.40, -0.80], [-0.40, -0.18], [0.16, -0.18],
            [0.52, 0.18], [0.18, 0.84], [-0.42, 0.72],
        ])],
        [pts([
            [0.46, -0.70], [0.16, -0.84], [-0.26, -0.74], [-0.42, -0.34],
            [-0.04, -0.12], [0.30, -0.04], [0.54, 0.26], [0.26, 0.70],
            [-0.20, 0.88], [-0.48, 0.68],
        ])],
    ],
    6: [
        [pts([
            [-0.22, -0.82], [-0.50, -0.22], [-0.46, 0.36], [-0.14, 0.84],
            [0.32, 0.82], [0.56, 0.46], [0.40, 0.08], [0.06, -0.10],
            [-0.32, 0.02], [-0.44, 0.38], [-0.18, 0.74], [0.22, 0.74],
        ])],
        [pts([
            [-0.12, -0.86], [-0.42, -0.38], [-0.52, 0.10], [-0.38, 0.60],
            [-0.02, 0.88], [0.40, 0.78], [0.56, 0.40], [0.36, 0.02],
            [-0.02, -0.08], [-0.34, 0.12], [-0.34, 0.46], [-0.02, 0.70],
            [0.28, 0.62],
        ])],
        [pts([
            [-0.04, -0.84], [-0.34, -0.52], [-0.54, -0.02], [-0.44, 0.46],
            [-0.16, 0.84], [0.26, 0.82], [0.52, 0.52], [0.46, 0.12],
            [0.16, -0.08], [-0.20, 0.00], [-0.40, 0.26], [-0.26, 0.62],
            [0.10, 0.70],
        ])],
        [pts([
            [-0.28, -0.78], [-0.52, -0.28], [-0.54, 0.26], [-0.30, 0.72],
            [0.12, 0.88], [0.50, 0.62], [0.52, 0.22], [0.26, -0.02],
            [-0.10, -0.02], [-0.36, 0.22], [-0.32, 0.50], [0.00, 0.70],
            [0.30, 0.58],
        ])],
    ],
    7: [
        [pts([[-0.54, -0.82], [0.56, -0.82], [0.00, 0.88]])],
        [pts([[-0.46, -0.82], [0.54, -0.82], [0.18, -0.04], [0.04, 0.88]])],
        [pts([[-0.52, -0.82], [0.56, -0.82], [0.02, 0.88]]), pts([[-0.18, -0.10], [0.28, 0.12]])],
        [pts([[-0.38, -0.80], [0.54, -0.84], [-0.08, 0.88]])],
    ],
    8: [
        [ellipse_poly(0.00, -0.40, 0.38, 0.32), ellipse_poly(0.00, 0.42, 0.44, 0.36)],
        [ellipse_poly(0.00, -0.38, 0.32, 0.28), ellipse_poly(0.00, 0.36, 0.36, 0.32)],
        [ellipse_poly(0.00, -0.44, 0.44, 0.30), ellipse_poly(0.00, 0.40, 0.48, 0.34)],
        [pts([
            [-0.18, -0.80], [0.18, -0.80], [0.40, -0.56], [0.18, -0.24],
            [-0.18, -0.08], [0.18, 0.06], [0.44, 0.42], [0.20, 0.82],
            [-0.20, 0.82], [-0.46, 0.42], [-0.18, 0.06], [0.16, -0.12],
            [-0.16, -0.28], [-0.42, -0.56], [-0.18, -0.80],
        ])],
    ],
    9: [
        [ellipse_poly(0.02, -0.34, 0.44, 0.42), pts([[0.34, -0.04], [0.44, 0.42], [0.22, 0.88]])],
        [ellipse_poly(-0.02, -0.28, 0.38, 0.38), pts([[0.24, -0.04], [0.22, 0.88]])],
        [ellipse_poly(0.04, -0.36, 0.46, 0.40), pts([[0.38, -0.06], [0.52, 0.34], [0.30, 0.86], [-0.08, 0.90]])],
        [closed([
            [-0.22, -0.72], [0.20, -0.82], [0.54, -0.54], [0.50, -0.12],
            [0.16, 0.12], [-0.24, 0.02], [-0.48, -0.30], [-0.34, -0.64],
        ]), pts([[0.34, 0.00], [0.42, 0.46], [0.16, 0.90]])],
    ],
}


def pixel_grid(size=IMG, scale=DRAW_SCALE):
    xs = (np.arange(size, dtype=np.float32) + 0.5 - size / 2.0) / scale
    ys = (np.arange(size, dtype=np.float32) + 0.5 - size / 2.0) / scale
    return np.meshgrid(xs, ys)


GX, GY = pixel_grid()


def transform_poly(poly, tr):
    shear, sx, sy, dx, dy, _ = tr
    x = poly[:, 0]
    y = poly[:, 1]
    xx = sx * x + shear * y + dx
    yy = sy * y + dy
    out = np.stack([xx, yy], axis=1).astype(np.float32)
    return np.clip(out, -0.98, 0.98)


def smooth_jitter(arr, closed_curve):
    if len(arr) < 3:
        return arr
    if closed_curve:
        return (np.roll(arr, 1, axis=0) + 2.0 * arr + np.roll(arr, -1, axis=0)) / 4.0
    out = arr.copy()
    out[1:-1] = (arr[:-2] + 2.0 * arr[1:-1] + arr[2:]) / 4.0
    out[0] = (2.0 * arr[0] + arr[1]) / 3.0
    out[-1] = (2.0 * arr[-1] + arr[-2]) / 3.0
    return out


def jitter_poly(poly, rng, scale):
    poly = poly.astype(np.float32, copy=True)
    is_closed = len(poly) >= 2 and np.all(poly[0] == poly[-1])
    core = poly[:-1] if is_closed else poly
    if len(core) >= 2:
        delta = rng.uniform(-scale, scale, size=core.shape).astype(np.float32)
        if not is_closed:
            delta[0] *= 0.35
            delta[-1] *= 0.35
        delta = smooth_jitter(delta, is_closed)
        core = core + delta
    if is_closed:
        core = np.concatenate([core, core[:1]], axis=0)
    return np.clip(core, -0.98, 0.98).astype(np.float32)


def jitter_style(style, digit, style_idx, variant_idx):
    if variant_idx == 0:
        return [poly.copy() for poly in style]
    rng = np.random.default_rng(10000 + 1000 * digit + 100 * style_idx + variant_idx)
    shear = rng.uniform(-0.08, 0.08)
    bowx = rng.uniform(-0.05, 0.05)
    bowy = rng.uniform(-0.04, 0.04)
    wigx = rng.uniform(-0.03, 0.03)
    wigy = rng.uniform(-0.03, 0.03)
    sx = 1.0 + rng.uniform(-0.04, 0.04)
    sy = 1.0 + rng.uniform(-0.04, 0.04)
    dx = rng.uniform(-0.03, 0.03)
    dy = rng.uniform(-0.03, 0.03)
    out = []
    for poly in style:
        p = jitter_poly(poly, rng, 0.03 if (len(poly) >= 2 and np.all(poly[0] == poly[-1])) else 0.05)
        x = p[:, 0]
        y = p[:, 1]
        xx = sx * (x + shear * y + bowx * (y * y - 0.30) + wigx * np.sin(np.pi * y)) + dx
        yy = sy * (y + bowy * (x * x - 0.20) + wigy * np.sin(np.pi * x)) + dy
        q = np.stack([xx, yy], axis=1).astype(np.float32)
        q = np.clip(q, -0.98, 0.98)
        if len(poly) >= 2 and np.all(poly[0] == poly[-1]):
            q[-1] = q[0]
        out.append(q)
    return out


def draw_polyline(img, poly, sigma):
    if len(poly) < 2:
        return img
    sigma2 = np.float32(sigma * sigma)
    layer = np.zeros_like(img)
    for i in range(len(poly) - 1):
        p0 = poly[i]
        p1 = poly[i + 1]
        vx = np.float32(p1[0] - p0[0])
        vy = np.float32(p1[1] - p0[1])
        vv = np.float32(vx * vx + vy * vy)
        if vv < 1e-8:
            dx = GX - p0[0]
            dy = GY - p0[1]
            d2 = dx * dx + dy * dy
        else:
            t = ((GX - p0[0]) * vx + (GY - p0[1]) * vy) / vv
            t = np.clip(t, 0.0, 1.0)
            px = p0[0] + t * vx
            py = p0[1] + t * vy
            dx = GX - px
            dy = GY - py
            d2 = dx * dx + dy * dy
        seg = np.exp(-0.5 * d2 / sigma2).astype(np.float32)
        layer = np.maximum(layer, seg)
    return np.maximum(img, layer)


def render_style(style, tr):
    img = np.zeros((IMG, IMG), dtype=np.float32)
    sigma = BASE_SIGMA * np.float32(tr[5])
    for poly in style:
        img = draw_polyline(img, transform_poly(poly, tr), sigma)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def downsample_mean(arr, block):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        h, w = arr.shape
        return arr.reshape(h // block, block, w // block, block).mean(axis=(1, 3), dtype=np.float32)
    c, h, w = arr.shape
    return arr.reshape(c, h // block, block, w // block, block).mean(axis=(2, 4), dtype=np.float32)


def norm_channels(arr):
    arr = np.asarray(arr, dtype=np.float32)
    added = False
    if arr.ndim == 2:
        arr = arr[None, :, :]
        added = True
    mu = arr.mean(axis=(1, 2), keepdims=True, dtype=np.float32)
    cen = arr - mu
    var = (cen * cen).mean(axis=(1, 2), keepdims=True, dtype=np.float32)
    out = cen / np.sqrt(var + np.float32(NORM_EPS)).astype(np.float32)
    return out[0] if added else out


def weight_norm(feature, scale):
    feat = norm_channels(feature).astype(np.float32)
    return (feat.reshape(-1) * (np.float32(scale) / np.float32(feat.size))).astype(np.float32)


def weight_raw(feature, scale, bg):
    feat = np.asarray(feature, dtype=np.float32)
    vec = feat.reshape(-1) - np.float32(bg)
    return (vec * (np.float32(scale) / np.float32(feat.size))).astype(np.float32)


def render_bar_kernel(angle_rad, length=1.45, width=0.14, size=7):
    xs = (np.arange(size, dtype=np.float32) + 0.5 - size / 2.0) / (size / 2.0)
    ys = (np.arange(size, dtype=np.float32) + 0.5 - size / 2.0) / (size / 2.0)
    gx, gy = np.meshgrid(xs, ys)
    c = np.float32(np.cos(angle_rad))
    s = np.float32(np.sin(angle_rad))
    v = gx * c + gy * s
    u = -gx * s + gy * c
    d = np.abs(u)
    env = np.exp(-0.5 * (d / np.float32(width)) ** 2).astype(np.float32)
    env *= np.exp(-0.5 * (v / np.float32(length / 2.2)) ** 8).astype(np.float32)
    env = env - env.mean(dtype=np.float32)
    env = env / np.sqrt((env * env).sum(dtype=np.float32) + np.float32(1e-8)).astype(np.float32)
    return env.astype(np.float32)


def conv_same_single(img, kernels):
    img = np.asarray(img, dtype=np.float32)
    kernels = np.asarray(kernels, dtype=np.float32)
    o, kh, kw = kernels.shape
    pad_h = kh // 2
    pad_w = kw // 2
    xp = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w))).astype(np.float32)
    out = np.zeros((o, img.shape[0], img.shape[1]), dtype=np.float32)
    for oy in range(img.shape[0]):
        ys = oy
        ye = oy + kh
        for ox in range(img.shape[1]):
            patch = xp[ys:ye, ox:ox + kw]
            out[:, oy, ox] = (kernels * patch[None, :, :]).sum(axis=(1, 2), dtype=np.float32)
    return out


def relu(x):
    return np.maximum(x, 0.0).astype(np.float32)


def edge_feature(img, edge_kernels):
    e = conv_same_single(norm_channels(img), edge_kernels)
    e = relu(e)
    e = downsample_mean(e, 4)
    e = norm_channels(e)
    return e.astype(np.float32)


def build_weights():
    edge_angles = [
        -3.0 * np.pi / 8.0,
        -np.pi / 4.0,
        -np.pi / 8.0,
        0.0,
        np.pi / 8.0,
        np.pi / 4.0,
        3.0 * np.pi / 8.0,
        np.pi / 2.0,
    ]
    edge_kernels = np.stack([render_bar_kernel(a) for a in edge_angles], axis=0).astype(np.float32)

    coarse14 = []
    coarse7 = []
    raw7 = []
    fine28 = []
    edgew = []
    quadw = []

    for digit in range(10):
        # large synthetic coarse template bank: 4 base styles × 4 deterministic jitters × 8 global transforms
        for style_idx, style in enumerate(DIGIT_STYLES[digit]):
            for variant_idx in range(JITTER_VARIANTS):
                jstyle = jitter_style(style, digit, style_idx, variant_idx)
                for tr in TRANSFORMS:
                    img = render_style(jstyle, tr)
                    coarse14.append(weight_norm(downsample_mean(img, 2), BRANCH_SCALE_14))
                    coarse7.append(weight_norm(downsample_mean(img, 4), BRANCH_SCALE_7))
                    raw7.append(weight_raw(downsample_mean(img, 4), BRANCH_SCALE_RAW7, RAW7_BG))

        # fine full-resolution branch: identity transform with the first two jitter variants
        for style_idx, style in enumerate(DIGIT_STYLES[digit]):
            for variant_idx in (0, 1):
                img = render_style(jitter_style(style, digit, style_idx, variant_idx), TRANSFORMS[0])
                fine28.append(weight_norm(img, BRANCH_SCALE_28))
                edgew.append(weight_norm(edge_feature(img, edge_kernels), BRANCH_SCALE_EDGE))

        # coarse occupancy prior from the four base styles at 4x4 resolution
        for style in DIGIT_STYLES[digit]:
            img = render_style(style, TRANSFORMS[0])
            quadw.append(weight_raw(downsample_mean(img, 7), BRANCH_SCALE_QUAD, RAW4_BG))

    coarse14 = np.stack(coarse14, axis=0).astype(np.float32)
    coarse7 = np.stack(coarse7, axis=0).astype(np.float32)
    raw7 = np.stack(raw7, axis=0).astype(np.float32)
    fine28 = np.stack(fine28, axis=0).astype(np.float32)
    edgew = np.stack(edgew, axis=0).astype(np.float32)
    quadw = np.stack(quadw, axis=0).astype(np.float32)

    weights = {
        'd14.W': coarse14,
        'd14.b': np.zeros((10 * COARSE_PER_CLASS,), dtype=np.float32),
        'd7.W': coarse7,
        'd7.b': np.zeros((10 * COARSE_PER_CLASS,), dtype=np.float32),
        'r7.W': raw7,
        'r7.b': np.zeros((10 * COARSE_PER_CLASS,), dtype=np.float32),
        'd28.W': fine28,
        'd28.b': np.zeros((10 * FINE_VARIANTS,), dtype=np.float32),
        'edge.W': edge_kernels[:, None, :, :].astype(np.float32),
        'edge.b': np.zeros((EDGE_FILTERS,), dtype=np.float32),
        'edense.W': edgew,
        'edense.b': np.zeros((10 * EDGE_VARIANTS,), dtype=np.float32),
        'qdense.W': quadw,
        'qdense.b': np.zeros((10 * QUAD_VARIANTS,), dtype=np.float32),
        'out.W': np.eye(10, dtype=np.float32),
        'out.b': FINAL_BIAS.astype(np.float32),
    }
    return weights


def build_network():
    return {
        'layers': [
            {'name': 'n28', 'op': 'norm', 'inputs': ['image']},
            {'name': 'a14', 'op': 'avgpool', 'inputs': ['image'], 'size': [2, 2], 'stride': [2, 2]},
            {'name': 'n14', 'op': 'norm', 'inputs': ['a14']},
            {'name': 'a7', 'op': 'avgpool', 'inputs': ['image'], 'size': [4, 4], 'stride': [4, 4]},
            {'name': 'n7', 'op': 'norm', 'inputs': ['a7']},

            {'name': 'd14', 'op': 'dense', 'inputs': ['n14'], 'out': 10 * COARSE_PER_CLASS},
            {'name': 'd7', 'op': 'dense', 'inputs': ['n7'], 'out': 10 * COARSE_PER_CLASS},
            {'name': 'r7', 'op': 'dense', 'inputs': ['a7'], 'out': 10 * COARSE_PER_CLASS},
            {'name': 'coarse_add', 'op': 'add', 'inputs': ['d14', 'd7', 'r7']},
            {'name': 'coarse_r', 'op': 'reshape', 'inputs': ['coarse_add'], 'shape': [10, COARSE_PER_CLASS, 1]},
            {'name': 'coarse_m', 'op': 'maxpool', 'inputs': ['coarse_r'], 'size': [COARSE_PER_CLASS, 1], 'stride': [COARSE_PER_CLASS, 1]},
            {'name': 'coarse_s', 'op': 'reshape', 'inputs': ['coarse_m'], 'shape': [10]},

            {'name': 'd28', 'op': 'dense', 'inputs': ['n28'], 'out': 10 * FINE_VARIANTS},
            {'name': 'fine_r', 'op': 'reshape', 'inputs': ['d28'], 'shape': [10, FINE_VARIANTS, 1]},
            {'name': 'fine_m', 'op': 'maxpool', 'inputs': ['fine_r'], 'size': [FINE_VARIANTS, 1], 'stride': [FINE_VARIANTS, 1]},
            {'name': 'fine_s', 'op': 'reshape', 'inputs': ['fine_m'], 'shape': [10]},

            {'name': 'edge', 'op': 'conv', 'inputs': ['n28'], 'out': EDGE_FILTERS, 'kernel': [7, 7], 'stride': 1, 'padding': 3},
            {'name': 'edge_relu', 'op': 'relu', 'inputs': ['edge']},
            {'name': 'edge_pool', 'op': 'avgpool', 'inputs': ['edge_relu'], 'size': [4, 4], 'stride': [4, 4]},
            {'name': 'edge_norm', 'op': 'norm', 'inputs': ['edge_pool']},
            {'name': 'edense', 'op': 'dense', 'inputs': ['edge_norm'], 'out': 10 * EDGE_VARIANTS},
            {'name': 'edge_r', 'op': 'reshape', 'inputs': ['edense'], 'shape': [10, EDGE_VARIANTS, 1]},
            {'name': 'edge_m', 'op': 'maxpool', 'inputs': ['edge_r'], 'size': [EDGE_VARIANTS, 1], 'stride': [EDGE_VARIANTS, 1]},
            {'name': 'edge_s', 'op': 'reshape', 'inputs': ['edge_m'], 'shape': [10]},

            {'name': 'quad', 'op': 'avgpool', 'inputs': ['image'], 'size': [7, 7], 'stride': [7, 7]},
            {'name': 'qdense', 'op': 'dense', 'inputs': ['quad'], 'out': 10 * QUAD_VARIANTS},
            {'name': 'quad_r', 'op': 'reshape', 'inputs': ['qdense'], 'shape': [10, QUAD_VARIANTS, 1]},
            {'name': 'quad_m', 'op': 'maxpool', 'inputs': ['quad_r'], 'size': [QUAD_VARIANTS, 1], 'stride': [QUAD_VARIANTS, 1]},
            {'name': 'quad_s', 'op': 'reshape', 'inputs': ['quad_m'], 'shape': [10]},

            {'name': 'sum', 'op': 'add', 'inputs': ['coarse_s', 'fine_s', 'edge_s', 'quad_s']},
            {'name': 'out', 'op': 'dense', 'inputs': ['sum'], 'out': 10},
        ]
    }


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = build_network()
    weights = build_weights()
    with (out / 'network.json').open('w') as f:
        json.dump(net, f, separators=(',', ':'))
    np.savez(out / 'weights.npz', **weights)
    method = (
        "This network is a hand-built synthetic-template classifier. I typed four stroke sketches for each digit from general knowledge, "
        "rendered them analytically, then expanded them with a fixed library of deterministic jitters and slant/aspect/stroke transforms. "
        "The main score is a per-class max over many multiscale template matches (14x14 normalized, 7x7 normalized, and 7x7 raw occupancy), "
        "with smaller full-resolution, oriented-edge, and 4x4 occupancy branches added as priors. All weights come directly from typed geometry, "
        "fixed formulas, and a fixed random seed scheme; there is no fitting, averaging over the training set, or gradient descent.\n"
    )
    (out / 'method.md').write_text(method)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/app/out')
