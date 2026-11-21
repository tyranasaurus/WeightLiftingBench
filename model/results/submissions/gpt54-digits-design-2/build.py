import json
import math
import sys
from pathlib import Path

import numpy as np

IMAGE_SIZE = 28
CENTER = (IMAGE_SIZE - 1) / 2.0
EPS = 1e-5
BOX_W = 18.0
BOX_H = 20.0

FINE_SLANTS = (-0.28, -0.14, 0.0, 0.14, 0.28)
FINE_THICKNESSES = (1.15, 1.65)
FINE_X_SCALES = (0.94, 1.00, 1.06)
FINE_Y_SCALES = (0.96, 1.04)

COARSE_SLANTS = (-0.30, -0.15, 0.0, 0.15, 0.30)
COARSE_THICKNESSES = (1.10, 1.60)
COARSE_X_SCALES = (0.88, 1.00, 1.12)
COARSE_Y_SCALES = (0.88, 1.00, 1.12)

FINE_WEIGHT = 1.00 / 784.0
MID_WEIGHT = 0.90 / 196.0
COARSE_WEIGHT = 0.35 / 49.0


def poly(*pts):
    return np.array(pts, dtype=np.float32)


def bezier(p0, p1, p2, p3, n=28):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    omt = 1.0 - t
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    p3 = np.asarray(p3, dtype=np.float32)
    return (omt ** 3) * p0 + 3 * (omt ** 2) * t * p1 + 3 * omt * (t ** 2) * p2 + (t ** 3) * p3


def ellipse(cx, cy, rx, ry, n=60, a0=0.0, a1=2 * math.pi, rot=0.0):
    a = np.linspace(a0, a1, n, dtype=np.float32)
    x = rx * np.cos(a)
    y = ry * np.sin(a)
    if rot:
        cr, sr = math.cos(rot), math.sin(rot)
        xr = cr * x - sr * y
        yr = sr * x + cr * y
        x, y = xr, yr
    return np.stack([cx + x, cy + y], axis=1).astype(np.float32)


def transform_points(pts, slant=0.0, sx=1.0, sy=1.0):
    pts = np.asarray(pts, dtype=np.float32).copy()
    x = (pts[:, 0] - 0.5) * sx
    y = (pts[:, 1] - 0.5) * sy
    x = x + slant * y
    pts[:, 0] = x + 0.5
    pts[:, 1] = y + 0.5
    return pts


def to_pixels(pts):
    pts = np.asarray(pts, dtype=np.float32)
    out = np.empty_like(pts)
    out[:, 0] = CENTER + (pts[:, 0] - 0.5) * BOX_W
    out[:, 1] = CENTER + (pts[:, 1] - 0.5) * BOX_H
    return out


def segment_distance(gx, gy, p0, p1):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    vx, vy = x1 - x0, y1 - y0
    denom = vx * vx + vy * vy + 1e-8
    t = ((gx - x0) * vx + (gy - y0) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    return np.sqrt((gx - px) ** 2 + (gy - py) ** 2)


def shift_image(img, dx, dy):
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = xx - dx
    sy = yy - dy
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = sx - x0
    wy = sy - y0

    def gather(xi, yi):
        good = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out = np.zeros((h, w), dtype=np.float32)
        out[good] = img[yi[good], xi[good]]
        return out

    a = gather(x0, y0)
    b = gather(x1, y0)
    c = gather(x0, y1)
    d = gather(x1, y1)
    return ((1 - wx) * (1 - wy) * a + wx * (1 - wy) * b + (1 - wx) * wy * c + wx * wy * d).astype(np.float32)


def center_of_mass(img):
    mass = float(img.sum())
    if mass <= 1e-8:
        return CENTER, CENTER
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    cy = float((img * yy).sum() / mass)
    cx = float((img * xx).sum() / mass)
    return cx, cy


def render(strokes, thickness):
    gy, gx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float32)
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for stroke in strokes:
        pts = to_pixels(stroke)
        for i in range(len(pts) - 1):
            d = segment_distance(gx, gy, pts[i], pts[i + 1])
            img = np.maximum(img, np.exp(-0.5 * (d / thickness) ** 2).astype(np.float32))
    cx, cy = center_of_mass(img)
    img = shift_image(img, CENTER - cx, CENTER - cy)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def blur3(img):
    xp = np.pad(img, ((1, 1), (1, 1))).astype(np.float32)
    return (
        xp[:-2, :-2] + 2 * xp[:-2, 1:-1] + xp[:-2, 2:] +
        2 * xp[1:-1, :-2] + 4 * xp[1:-1, 1:-1] + 2 * xp[1:-1, 2:] +
        xp[2:, :-2] + 2 * xp[2:, 1:-1] + xp[2:, 2:]
    ).astype(np.float32) / 16.0


def avgpool2d(img, size):
    h, w = img.shape
    assert h % size == 0 and w % size == 0
    return img.reshape(h // size, size, w // size, size).mean(axis=(1, 3), dtype=np.float32).astype(np.float32)


def normalize_template(img):
    x = img.astype(np.float32)
    x = x - x.mean()
    x = x / np.sqrt((x * x).mean() + EPS)
    return x.astype(np.float32)


def preprocess(img, pool=None):
    x = blur3(img)
    if pool is not None:
        x = avgpool2d(x, pool)
    return normalize_template(x)


def zero_styles():
    return [
        [ellipse(0.50, 0.50, 0.28, 0.42, rot=0.00)],
        [ellipse(0.50, 0.50, 0.25, 0.45, rot=0.06)],
        [ellipse(0.50, 0.50, 0.31, 0.38, rot=-0.08)],
        [ellipse(0.50, 0.48, 0.27, 0.43, rot=0.10)],
        [ellipse(0.50, 0.50, 0.29, 0.40, rot=0.14)],
        [ellipse(0.50, 0.50, 0.24, 0.46, rot=-0.12)],
    ]


def one_styles():
    return [
        [poly((0.50, 0.12), (0.50, 0.88))],
        [poly((0.34, 0.24), (0.50, 0.12), (0.50, 0.88))],
        [poly((0.36, 0.23), (0.50, 0.12), (0.50, 0.82), (0.38, 0.88))],
        [poly((0.44, 0.16), (0.54, 0.12), (0.56, 0.88))],
        [poly((0.42, 0.12), (0.48, 0.18), (0.48, 0.88), (0.36, 0.88))],
        [poly((0.34, 0.18), (0.48, 0.12), (0.52, 0.88), (0.62, 0.88))],
    ]


def two_styles():
    return [
        [
            bezier((0.18, 0.28), (0.28, 0.08), (0.65, 0.08), (0.78, 0.26), 24),
            bezier((0.78, 0.26), (0.78, 0.38), (0.45, 0.52), (0.22, 0.78), 24),
            poly((0.22, 0.78), (0.82, 0.78)),
        ],
        [
            bezier((0.22, 0.24), (0.38, 0.08), (0.72, 0.14), (0.76, 0.32), 24),
            bezier((0.76, 0.32), (0.64, 0.46), (0.40, 0.56), (0.20, 0.76), 24),
            bezier((0.20, 0.76), (0.42, 0.80), (0.60, 0.80), (0.82, 0.78), 18),
        ],
        [poly((0.18, 0.26), (0.30, 0.14), (0.70, 0.16), (0.78, 0.28), (0.22, 0.78), (0.82, 0.78))],
        [
            bezier((0.20, 0.22), (0.36, 0.12), (0.70, 0.12), (0.78, 0.30), 24),
            bezier((0.78, 0.30), (0.62, 0.52), (0.32, 0.58), (0.26, 0.74), 24),
            bezier((0.26, 0.74), (0.44, 0.80), (0.62, 0.80), (0.82, 0.80), 18),
        ],
        [poly((0.18, 0.24), (0.34, 0.12), (0.74, 0.16), (0.72, 0.32), (0.30, 0.68), (0.22, 0.82), (0.82, 0.82))],
        [
            bezier((0.18, 0.26), (0.34, 0.10), (0.70, 0.10), (0.76, 0.26), 24),
            poly((0.76, 0.26), (0.62, 0.46), (0.26, 0.70), (0.18, 0.82), (0.82, 0.82)),
        ],
    ]


def three_styles():
    return [
        [
            bezier((0.20, 0.24), (0.40, 0.10), (0.76, 0.12), (0.72, 0.40), 24),
            bezier((0.72, 0.40), (0.66, 0.50), (0.46, 0.52), (0.34, 0.52), 16),
            bezier((0.34, 0.52), (0.52, 0.50), (0.82, 0.58), (0.72, 0.82), 24),
            bezier((0.72, 0.82), (0.56, 0.90), (0.32, 0.86), (0.20, 0.74), 18),
        ],
        [poly((0.22, 0.24), (0.42, 0.16), (0.72, 0.20), (0.58, 0.50), (0.74, 0.56), (0.74, 0.76), (0.54, 0.86), (0.24, 0.76))],
        [
            bezier((0.18, 0.24), (0.36, 0.08), (0.72, 0.10), (0.68, 0.34), 24),
            bezier((0.68, 0.34), (0.62, 0.48), (0.46, 0.50), (0.34, 0.50), 16),
            bezier((0.34, 0.50), (0.54, 0.50), (0.80, 0.60), (0.74, 0.80), 24),
            bezier((0.74, 0.80), (0.60, 0.92), (0.34, 0.90), (0.20, 0.78), 18),
        ],
        [
            bezier((0.24, 0.24), (0.54, 0.14), (0.74, 0.18), (0.70, 0.42), 24),
            bezier((0.70, 0.42), (0.64, 0.52), (0.44, 0.54), (0.30, 0.54), 18),
            bezier((0.30, 0.54), (0.52, 0.54), (0.84, 0.60), (0.72, 0.80), 24),
            bezier((0.72, 0.80), (0.54, 0.90), (0.34, 0.88), (0.22, 0.76), 18),
        ],
        [poly((0.20, 0.22), (0.42, 0.16), (0.72, 0.20), (0.56, 0.48), (0.74, 0.56), (0.72, 0.82), (0.50, 0.88), (0.24, 0.80))],
        [
            bezier((0.20, 0.22), (0.44, 0.10), (0.74, 0.16), (0.66, 0.36), 24),
            poly((0.42, 0.50), (0.62, 0.50)),
            bezier((0.32, 0.52), (0.56, 0.52), (0.82, 0.62), (0.70, 0.82), 24),
            bezier((0.70, 0.82), (0.52, 0.90), (0.30, 0.86), (0.20, 0.74), 18),
        ],
    ]


def four_styles():
    return [
        [poly((0.68, 0.12), (0.68, 0.88)), poly((0.22, 0.58), (0.78, 0.58)), poly((0.26, 0.58), (0.64, 0.12))],
        [poly((0.66, 0.12), (0.66, 0.88)), poly((0.22, 0.54), (0.76, 0.54)), poly((0.24, 0.54), (0.60, 0.12))],
        [poly((0.24, 0.46), (0.60, 0.12), (0.60, 0.88)), poly((0.24, 0.46), (0.78, 0.46))],
        [poly((0.70, 0.12), (0.66, 0.88)), poly((0.24, 0.56), (0.78, 0.56)), poly((0.26, 0.56), (0.64, 0.16))],
        [poly((0.34, 0.14), (0.34, 0.52)), poly((0.34, 0.52), (0.76, 0.52)), poly((0.62, 0.12), (0.62, 0.88))],
        [poly((0.56, 0.12), (0.30, 0.48), (0.72, 0.48), (0.72, 0.88)), poly((0.56, 0.12), (0.72, 0.30))],
    ]


def five_styles():
    return [
        [poly((0.78, 0.18), (0.28, 0.18), (0.24, 0.48)), bezier((0.24, 0.48), (0.38, 0.42), (0.70, 0.44), (0.76, 0.70), 24), bezier((0.76, 0.70), (0.64, 0.90), (0.30, 0.90), (0.18, 0.76), 20)],
        [poly((0.76, 0.20), (0.30, 0.20), (0.26, 0.46)), bezier((0.26, 0.46), (0.42, 0.46), (0.70, 0.46), (0.76, 0.60), 20), bezier((0.76, 0.60), (0.70, 0.86), (0.36, 0.90), (0.20, 0.78), 20)],
        [poly((0.78, 0.18), (0.28, 0.18), (0.26, 0.50), (0.62, 0.50)), bezier((0.62, 0.50), (0.84, 0.52), (0.82, 0.84), (0.44, 0.86), 24), bezier((0.44, 0.86), (0.28, 0.86), (0.20, 0.82), (0.16, 0.76), 12)],
        [poly((0.74, 0.18), (0.32, 0.18), (0.28, 0.46)), bezier((0.28, 0.46), (0.36, 0.40), (0.62, 0.42), (0.72, 0.54), 18), bezier((0.72, 0.54), (0.84, 0.74), (0.66, 0.88), (0.30, 0.84), 20)],
        [bezier((0.78, 0.18), (0.34, 0.14), (0.24, 0.30), (0.26, 0.50), 24), bezier((0.26, 0.50), (0.42, 0.46), (0.70, 0.46), (0.76, 0.62), 20), bezier((0.76, 0.62), (0.72, 0.88), (0.34, 0.90), (0.18, 0.78), 22)],
        [poly((0.76, 0.18), (0.32, 0.18), (0.28, 0.44)), poly((0.28, 0.44), (0.62, 0.44)), bezier((0.62, 0.44), (0.86, 0.48), (0.84, 0.82), (0.42, 0.86), 24), bezier((0.42, 0.86), (0.28, 0.86), (0.20, 0.82), (0.16, 0.76), 12)],
    ]


def six_styles():
    return [
        [bezier((0.68, 0.18), (0.44, 0.10), (0.22, 0.34), (0.28, 0.62), 28), ellipse(0.48, 0.64, 0.25, 0.22, n=52)],
        [bezier((0.66, 0.18), (0.36, 0.10), (0.22, 0.36), (0.32, 0.60), 28), ellipse(0.48, 0.64, 0.27, 0.21, n=52)],
        [bezier((0.70, 0.20), (0.48, 0.08), (0.24, 0.30), (0.30, 0.56), 28), ellipse(0.48, 0.64, 0.24, 0.23, n=52), poly((0.30, 0.56), (0.54, 0.56))],
        [bezier((0.62, 0.18), (0.36, 0.12), (0.20, 0.40), (0.30, 0.64), 28), ellipse(0.50, 0.66, 0.25, 0.20, n=52)],
        [bezier((0.70, 0.18), (0.40, 0.08), (0.18, 0.36), (0.30, 0.58), 28), ellipse(0.50, 0.66, 0.26, 0.22, n=52)],
        [bezier((0.64, 0.18), (0.34, 0.12), (0.22, 0.42), (0.34, 0.64), 28), ellipse(0.52, 0.68, 0.23, 0.19, n=48)],
    ]


def seven_styles():
    return [
        [poly((0.20, 0.18), (0.82, 0.18), (0.34, 0.86))],
        [poly((0.18, 0.20), (0.80, 0.20), (0.56, 0.48), (0.34, 0.86))],
        [poly((0.20, 0.18), (0.80, 0.18), (0.34, 0.84)), poly((0.38, 0.44), (0.64, 0.44))],
        [poly((0.24, 0.18), (0.78, 0.18), (0.44, 0.86))],
        [poly((0.28, 0.18), (0.76, 0.18), (0.40, 0.88))],
        [poly((0.20, 0.20), (0.82, 0.20), (0.58, 0.42), (0.42, 0.88)), poly((0.36, 0.48), (0.62, 0.48))],
    ]


def eight_styles():
    return [
        [ellipse(0.50, 0.34, 0.22, 0.18, n=44), ellipse(0.50, 0.68, 0.28, 0.22, n=52)],
        [ellipse(0.50, 0.36, 0.24, 0.20, n=46), ellipse(0.50, 0.68, 0.24, 0.20, n=46)],
        [ellipse(0.50, 0.34, 0.20, 0.17, n=42), ellipse(0.50, 0.70, 0.29, 0.23, n=54)],
        [ellipse(0.50, 0.36, 0.23, 0.19, n=44), ellipse(0.50, 0.68, 0.26, 0.21, n=48)],
        [ellipse(0.50, 0.32, 0.18, 0.16, n=40), ellipse(0.50, 0.70, 0.26, 0.22, n=52)],
        [ellipse(0.50, 0.35, 0.22, 0.19, n=44), ellipse(0.50, 0.68, 0.22, 0.19, n=44)],
    ]


def nine_styles():
    return [
        [ellipse(0.50, 0.36, 0.26, 0.22, n=52), poly((0.66, 0.46), (0.68, 0.88))],
        [ellipse(0.48, 0.34, 0.24, 0.20, n=48), bezier((0.68, 0.40), (0.74, 0.58), (0.66, 0.78), (0.48, 0.88), 26)],
        [ellipse(0.52, 0.36, 0.25, 0.21, n=50), poly((0.70, 0.44), (0.70, 0.84))],
        [ellipse(0.50, 0.34, 0.23, 0.19, n=46), bezier((0.64, 0.42), (0.76, 0.56), (0.70, 0.80), (0.54, 0.88), 26)],
        [ellipse(0.50, 0.34, 0.25, 0.21, n=50), bezier((0.66, 0.44), (0.76, 0.56), (0.72, 0.72), (0.58, 0.84), 26)],
        [ellipse(0.50, 0.36, 0.24, 0.20, n=48), poly((0.68, 0.42), (0.66, 0.86))],
    ]


def base_styles():
    return {
        0: zero_styles(),
        1: one_styles(),
        2: two_styles(),
        3: three_styles(),
        4: four_styles(),
        5: five_styles(),
        6: six_styles(),
        7: seven_styles(),
        8: eight_styles(),
        9: nine_styles(),
    }


def build_branch_templates(styles, *, slants, thicknesses, x_scales=(1.0,), y_scales=(1.0,), pool=None):
    bank = []
    per_class = None
    for digit in range(10):
        digit_templates = []
        for base in styles[digit]:
            for slant in slants:
                for sx in x_scales:
                    for sy in y_scales:
                        transformed = [transform_points(stroke, slant=slant, sx=sx, sy=sy) for stroke in base]
                        for thickness in thicknesses:
                            img = render(transformed, thickness=thickness)
                            digit_templates.append(preprocess(img, pool=pool))
        if per_class is None:
            per_class = len(digit_templates)
        assert len(digit_templates) == per_class
        bank.extend(digit_templates)
    return np.stack(bank, axis=0).reshape(10 * per_class, -1).astype(np.float32), per_class


def diag_scale(scale):
    return (scale * np.eye(10, dtype=np.float32)).astype(np.float32)


def build_network(out_dir: Path):
    styles = base_styles()
    fine_templates, fine_per_class = build_branch_templates(
        styles,
        slants=FINE_SLANTS,
        thicknesses=FINE_THICKNESSES,
        x_scales=FINE_X_SCALES,
        y_scales=FINE_Y_SCALES,
        pool=None,
    )
    mid_templates, mid_per_class = build_branch_templates(
        styles,
        slants=COARSE_SLANTS,
        thicknesses=COARSE_THICKNESSES,
        x_scales=COARSE_X_SCALES,
        y_scales=COARSE_Y_SCALES,
        pool=2,
    )
    coarse_templates, coarse_per_class = build_branch_templates(
        styles,
        slants=COARSE_SLANTS,
        thicknesses=COARSE_THICKNESSES,
        x_scales=COARSE_X_SCALES,
        y_scales=COARSE_Y_SCALES,
        pool=4,
    )

    blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    blur_w = blur[None, None, :, :]
    blur_b = np.zeros((1,), dtype=np.float32)

    net = {
        "layers": [
            {"name": "blur", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [3, 3], "stride": 1, "padding": 1},
            {"name": "norm28", "op": "norm", "inputs": ["blur"]},
            {"name": "fine_match", "op": "dense", "inputs": ["norm28"], "out": 10 * fine_per_class},
            {"name": "fine_group", "op": "reshape", "inputs": ["fine_match"], "shape": [10, 1, fine_per_class]},
            {"name": "fine_best", "op": "maxpool", "inputs": ["fine_group"], "size": [1, fine_per_class], "stride": [1, fine_per_class]},
            {"name": "fine_scores", "op": "reshape", "inputs": ["fine_best"], "shape": [10]},

            {"name": "pool14", "op": "avgpool", "inputs": ["blur"], "size": [2, 2], "stride": [2, 2]},
            {"name": "norm14", "op": "norm", "inputs": ["pool14"]},
            {"name": "mid_match", "op": "dense", "inputs": ["norm14"], "out": 10 * mid_per_class},
            {"name": "mid_group", "op": "reshape", "inputs": ["mid_match"], "shape": [10, 1, mid_per_class]},
            {"name": "mid_best", "op": "maxpool", "inputs": ["mid_group"], "size": [1, mid_per_class], "stride": [1, mid_per_class]},
            {"name": "mid_scores", "op": "reshape", "inputs": ["mid_best"], "shape": [10]},

            {"name": "pool7", "op": "avgpool", "inputs": ["blur"], "size": [4, 4], "stride": [4, 4]},
            {"name": "norm7", "op": "norm", "inputs": ["pool7"]},
            {"name": "coarse_match", "op": "dense", "inputs": ["norm7"], "out": 10 * coarse_per_class},
            {"name": "coarse_group", "op": "reshape", "inputs": ["coarse_match"], "shape": [10, 1, coarse_per_class]},
            {"name": "coarse_best", "op": "maxpool", "inputs": ["coarse_group"], "size": [1, coarse_per_class], "stride": [1, coarse_per_class]},
            {"name": "coarse_scores", "op": "reshape", "inputs": ["coarse_best"], "shape": [10]},

            {"name": "fine_scale", "op": "dense", "inputs": ["fine_scores"], "out": 10},
            {"name": "mid_scale", "op": "dense", "inputs": ["mid_scores"], "out": 10},
            {"name": "coarse_scale", "op": "dense", "inputs": ["coarse_scores"], "out": 10},
            {"name": "scores", "op": "add", "inputs": ["fine_scale", "mid_scale", "coarse_scale"]},
        ]
    }

    weights = {
        "blur.W": blur_w,
        "blur.b": blur_b,
        "fine_match.W": fine_templates,
        "fine_match.b": np.zeros((10 * fine_per_class,), dtype=np.float32),
        "mid_match.W": mid_templates,
        "mid_match.b": np.zeros((10 * mid_per_class,), dtype=np.float32),
        "coarse_match.W": coarse_templates,
        "coarse_match.b": np.zeros((10 * coarse_per_class,), dtype=np.float32),
        "fine_scale.W": diag_scale(FINE_WEIGHT),
        "fine_scale.b": np.zeros((10,), dtype=np.float32),
        "mid_scale.W": diag_scale(MID_WEIGHT),
        "mid_scale.b": np.zeros((10,), dtype=np.float32),
        "coarse_scale.W": diag_scale(COARSE_WEIGHT),
        "coarse_scale.b": np.zeros((10,), dtype=np.float32),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "network.json").open("w") as f:
        json.dump(net, f, separators=(",", ":"))
    np.savez(out_dir / "weights.npz", **weights)


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    build_network(target)
