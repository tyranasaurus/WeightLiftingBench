import json
import math
import sys
from pathlib import Path

import numpy as np


LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
]
H = W = 28
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)


# ---------- geometry ----------

def capsule_mask(x1, y1, x2, y2, r):
    vx = x2 - x1
    vy = y2 - y1
    t = ((X - x1) * vx + (Y - y1) * vy) / (vx * vx + vy * vy + 1e-9)
    t = np.clip(t, 0.0, 1.0)
    px = x1 + t * vx
    py = y1 + t * vy
    return (((X - px) ** 2 + (Y - py) ** 2) <= r * r).astype(np.float32)


def ellipse_mask(cx, cy, rx, ry, ang=0.0):
    ca = math.cos(ang)
    sa = math.sin(ang)
    xx = (X - cx) * ca + (Y - cy) * sa
    yy = -(X - cx) * sa + (Y - cy) * ca
    return (((xx / rx) ** 2 + (yy / ry) ** 2) <= 1.0).astype(np.float32)


def disk_mask(cx, cy, r):
    return (((X - cx) ** 2 + (Y - cy) ** 2) <= r * r).astype(np.float32)


def union(*masks):
    out = np.zeros((H, W), np.float32)
    for m in masks:
        out = np.maximum(out, m)
    return out


def subtract(a, b):
    return np.clip(a - b, 0.0, 1.0)


def shift(mask, dx, dy):
    dx = int(dx)
    dy = int(dy)
    out = np.zeros_like(mask)
    sx0 = max(0, -dx)
    sx1 = min(W, W - dx)
    sy0 = max(0, -dy)
    sy1 = min(H, H - dy)
    tx0 = max(0, dx)
    tx1 = min(W, W + dx)
    ty0 = max(0, dy)
    ty1 = min(H, H + dy)
    out[ty0:ty1, tx0:tx1] = mask[sy0:sy1, sx0:sx1]
    return out


def dilate(mask, r=2):
    out = np.zeros_like(mask)
    rr = int(math.ceil(r))
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            if dx * dx + dy * dy <= r * r + 1e-6:
                out = np.maximum(out, shift(mask, dx, dy))
    return out


def bilinear_sample(img, xx, yy):
    x0 = np.floor(xx).astype(np.int32)
    y0 = np.floor(yy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xx - x0
    wy = yy - y0
    out = np.zeros_like(xx, dtype=np.float32)

    def add(xi, yi, w):
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if np.any(ok):
            out[ok] += img[yi[ok], xi[ok]] * w[ok]

    add(x0, y0, (1.0 - wx) * (1.0 - wy))
    add(x1, y0, wx * (1.0 - wy))
    add(x0, y1, (1.0 - wx) * wy)
    add(x1, y1, wx * wy)
    return out


def affine(img, angle_deg=0.0, scale=1.0, center=(14.0, 14.0)):
    ang = math.radians(angle_deg)
    ca = math.cos(ang)
    sa = math.sin(ang)
    cx, cy = center
    xx = (ca * (X - cx) + sa * (Y - cy)) / scale + cx
    yy = (-sa * (X - cx) + ca * (Y - cy)) / scale + cy
    return bilinear_sample(img, xx, yy)


def blur3(img, n=1):
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
    k /= k.sum()
    for _ in range(n):
        p = np.pad(img, 1, mode="edge")
        out = np.zeros_like(img)
        for dy in range(3):
            for dx in range(3):
                out += k[dy, dx] * p[dy:dy + H, dx:dx + W]
        img = out
    return img


def normalize_image(img):
    mu = float(img.mean())
    sd = float(img.std())
    return (img - mu) / math.sqrt(sd * sd + 1e-5)


# ---------- synthetic hand models ----------

def palm():
    return union(
        ellipse_mask(14.0, 18.0, 5.4, 6.8, -0.03),
        capsule_mask(14.0, 23.0, 14.0, 28.0, 2.8),
    )


def finger(base, tip, r=1.35):
    return capsule_mask(base[0], base[1], tip[0], tip[1], r)


def thumb(base, tip, r=1.5):
    return capsule_mask(base[0], base[1], tip[0], tip[1], r)


def hand_models():
    finger_bases = [(9.8, 13.5), (12.4, 12.8), (15.1, 12.8), (17.8, 13.4)]
    letters = {}
    strokes = {}

    m = union(palm(), *(finger(finger_bases[i], tip, 1.55) for i, tip in enumerate([(9.4, 9.6), (12.0, 9.0), (14.7, 9.0), (17.3, 9.7)])), thumb((18.0, 18.0), (21.2, 14.5), 1.7)); letters['A'] = m; strokes['A'] = [((16.0, 15.0), (19.5, 13.2), 1.0)]
    m = union(ellipse_mask(13.8, 18.2, 4.7, 6.5, -0.02), capsule_mask(13.8, 22.5, 13.8, 28.0, 2.5), finger(finger_bases[0], (9.2, 3.8), 1.2), finger(finger_bases[1], (12.0, 2.8), 1.2), finger(finger_bases[2], (14.8, 3.0), 1.2), finger(finger_bases[3], (17.6, 4.1), 1.2), thumb((17.2, 17.5), (10.0, 16.0), 1.25)); letters['B'] = m; strokes['B'] = [((17.0, 16.0), (10.5, 16.0), 0.8)]
    outer = union(ellipse_mask(14.0, 14.0, 8.5, 10.0, -0.05), capsule_mask(10.5, 20.0, 10.0, 25.0, 2.0), capsule_mask(18.0, 20.0, 17.5, 23.0, 2.0)); inner = ellipse_mask(15.8, 14.0, 5.2, 6.8, -0.05); m = subtract(outer, inner); m = subtract(m, capsule_mask(21.0, 14.5, 24.5, 14.5, 4.0)); letters['C'] = m; strokes['C'] = []
    m = union(palm(), finger(finger_bases[1], (13.2, 3.0), 1.1), finger(finger_bases[0], (10.2, 11.0), 1.45), finger(finger_bases[2], (15.0, 11.2), 1.45), finger(finger_bases[3], (17.3, 12.0), 1.45), thumb((18.0, 17.0), (12.8, 11.6), 1.25)); m = subtract(m, disk_mask(13.5, 13.0, 2.6)); letters['D'] = m; strokes['D'] = [((18.0, 17.0), (12.8, 11.6), 0.8)]
    m = union(palm(), finger(finger_bases[0], (9.7, 10.0), 1.8), finger(finger_bases[1], (12.3, 9.2), 1.8), finger(finger_bases[2], (15.0, 9.3), 1.8), finger(finger_bases[3], (17.6, 10.2), 1.8), thumb((18.0, 15.5), (11.3, 13.2), 1.55)); letters['E'] = m; strokes['E'] = [((17.5, 15.0), (11.0, 13.4), 1.0)]
    m = union(ellipse_mask(14.0, 17.5, 5.1, 6.9, -0.03), capsule_mask(14.0, 23.0, 14.0, 28.0, 2.7), finger(finger_bases[1], (12.2, 4.0), 1.1), finger(finger_bases[2], (15.0, 4.4), 1.1), finger(finger_bases[3], (17.8, 5.3), 1.1), finger(finger_bases[0], (10.3, 10.8), 1.15), thumb((18.0, 17.0), (12.2, 9.8), 1.25)); m = subtract(m, disk_mask(13.6, 11.2, 2.4)); letters['F'] = m; strokes['F'] = [((18.0, 17.0), (12.2, 9.8), 0.7)]
    m = union(ellipse_mask(10.0, 17.0, 4.5, 6.0, -0.05), capsule_mask(9.2, 22.0, 9.2, 28.0, 2.5), finger((10.2, 13.2), (22.5, 11.5), 1.4), thumb((10.8, 16.2), (19.2, 17.7), 1.25)); letters['G'] = m; strokes['G'] = []
    m = union(ellipse_mask(10.2, 17.0, 4.6, 6.0, -0.05), capsule_mask(9.2, 22.0, 9.2, 28.0, 2.5), finger((10.0, 12.8), (22.6, 11.6), 1.28), finger((10.0, 15.0), (22.3, 14.1), 1.22), thumb((11.5, 17.3), (17.0, 18.8), 1.1)); letters['H'] = m; strokes['H'] = []
    m = union(palm(), finger(finger_bases[3], (17.8, 4.0), 1.18), finger(finger_bases[0], (10.0, 10.2), 1.55), finger(finger_bases[1], (12.5, 9.7), 1.55), finger(finger_bases[2], (15.0, 9.8), 1.55), thumb((18.0, 16.6), (13.8, 14.2), 1.3)); letters['I'] = m; strokes['I'] = [((17.0, 15.8), (14.0, 14.2), 0.7)]
    m = union(palm(), finger(finger_bases[1], (12.7, 3.5), 1.1), finger(finger_bases[2], (16.8, 5.8), 1.1), finger(finger_bases[0], (10.1, 10.8), 1.45), finger(finger_bases[3], (17.5, 11.0), 1.45), thumb((18.0, 16.8), (14.5, 10.8), 1.15)); letters['K'] = m; strokes['K'] = [((18.0, 16.8), (14.5, 10.8), 0.7)]
    m = union(palm(), finger(finger_bases[1], (13.0, 3.2), 1.1), finger(finger_bases[0], (10.0, 10.4), 1.45), finger(finger_bases[2], (15.0, 10.2), 1.45), finger(finger_bases[3], (17.5, 11.0), 1.45), thumb((18.0, 16.8), (23.8, 17.0), 1.35)); letters['L'] = m; strokes['L'] = []
    m = union(palm(), finger(finger_bases[0], (9.8, 9.6), 1.65), finger(finger_bases[1], (12.2, 9.0), 1.65), finger(finger_bases[2], (14.8, 9.0), 1.65), finger(finger_bases[3], (17.1, 9.8), 1.65), thumb((17.8, 16.0), (10.2, 15.3), 1.05)); letters['M'] = m; strokes['M'] = [((17.6, 15.8), (9.8, 15.2), 0.8)]
    m = union(palm(), finger(finger_bases[0], (10.0, 9.7), 1.62), finger(finger_bases[1], (12.4, 9.0), 1.62), finger(finger_bases[2], (15.0, 9.2), 1.62), finger(finger_bases[3], (17.2, 10.0), 1.62), thumb((17.9, 15.9), (12.0, 15.4), 1.15)); letters['N'] = m; strokes['N'] = [((17.4, 15.6), (11.8, 15.2), 0.7)]
    m = union(ellipse_mask(14.0, 16.0, 6.7, 8.7, -0.05), capsule_mask(14.0, 22.0, 14.0, 28.0, 2.5)); m = subtract(m, ellipse_mask(14.2, 15.5, 3.7, 4.9, -0.05)); letters['O'] = m; strokes['O'] = []
    base_p = union(palm(), finger(finger_bases[1], (12.7, 3.5), 1.1), finger(finger_bases[2], (16.8, 5.8), 1.1), finger(finger_bases[0], (10.1, 10.8), 1.45), finger(finger_bases[3], (17.5, 11.0), 1.45), thumb((18.0, 16.8), (14.5, 10.8), 1.15)); letters['P'] = affine(shift(base_p, 0, 1), 65.0, 1.0, center=(13.0, 16.0)); strokes['P'] = []
    base_q = union(ellipse_mask(10.0, 17.0, 4.5, 6.0, -0.05), capsule_mask(9.2, 22.0, 9.2, 28.0, 2.5), finger((10.2, 13.2), (22.5, 11.5), 1.4), thumb((10.8, 16.2), (19.2, 17.7), 1.25)); letters['Q'] = affine(shift(base_q, 0, 1), 78.0, 1.0, center=(11.0, 16.0)); strokes['Q'] = []
    m = union(palm(), finger(finger_bases[1], (12.8, 4.0), 1.05), finger(finger_bases[2], (15.2, 4.6), 1.05), finger(finger_bases[0], (10.1, 10.6), 1.45), finger(finger_bases[3], (17.5, 10.9), 1.45), thumb((18.0, 16.7), (13.6, 13.0), 1.15), capsule_mask(12.7, 7.0, 15.0, 9.5, 0.8)); letters['R'] = m; strokes['R'] = [((12.7, 7.0), (15.0, 9.5), 0.5)]
    m = union(palm(), finger(finger_bases[0], (10.0, 9.9), 1.65), finger(finger_bases[1], (12.4, 9.2), 1.65), finger(finger_bases[2], (14.9, 9.4), 1.65), finger(finger_bases[3], (17.3, 10.2), 1.65), thumb((18.2, 15.0), (10.6, 12.0), 1.45)); letters['S'] = m; strokes['S'] = [((18.0, 14.8), (10.8, 12.3), 1.0)]
    m = union(palm(), finger(finger_bases[0], (10.0, 10.0), 1.62), finger(finger_bases[1], (12.4, 9.3), 1.62), finger(finger_bases[2], (14.8, 9.5), 1.62), finger(finger_bases[3], (17.2, 10.3), 1.62), thumb((11.4, 12.0), (21.8, 12.0), 1.2), capsule_mask(14.0, 7.5, 14.0, 10.5, 0.8)); letters['T'] = m; strokes['T'] = [((11.4, 12.0), (21.8, 12.0), 0.8), ((14.0, 7.5), (14.0, 10.5), 0.5)]
    m = union(palm(), finger(finger_bases[1], (12.8, 3.8), 1.12), finger(finger_bases[2], (15.0, 4.1), 1.12), finger(finger_bases[0], (10.1, 10.7), 1.45), finger(finger_bases[3], (17.4, 10.9), 1.45), thumb((18.0, 16.7), (13.8, 13.6), 1.15)); letters['U'] = m; strokes['U'] = []
    m = union(palm(), finger(finger_bases[1], (11.7, 3.8), 1.08), finger(finger_bases[2], (16.2, 4.4), 1.08), finger(finger_bases[0], (10.1, 10.7), 1.45), finger(finger_bases[3], (17.4, 10.9), 1.45), thumb((18.0, 16.7), (13.8, 13.6), 1.15)); letters['V'] = m; strokes['V'] = []
    m = union(palm(), finger(finger_bases[0], (9.7, 5.4), 1.05), finger(finger_bases[1], (12.6, 3.7), 1.05), finger(finger_bases[2], (15.5, 4.3), 1.05), finger(finger_bases[3], (18.0, 5.8), 1.05), thumb((18.0, 16.7), (13.8, 13.6), 1.15)); letters['W'] = m; strokes['W'] = []
    m = union(palm(), finger(finger_bases[1], (13.0, 5.8), 1.15), capsule_mask(13.0, 5.8, 11.7, 4.7, 0.95), finger(finger_bases[0], (10.0, 10.6), 1.5), finger(finger_bases[2], (15.0, 10.2), 1.5), finger(finger_bases[3], (17.4, 10.8), 1.5), thumb((18.0, 16.7), (13.7, 13.2), 1.15)); letters['X'] = m; strokes['X'] = [((13.0, 5.8), (11.7, 4.7), 0.5)]
    m = union(palm(), finger(finger_bases[3], (17.8, 4.3), 1.15), finger(finger_bases[0], (10.0, 10.0), 1.55), finger(finger_bases[1], (12.4, 9.7), 1.55), finger(finger_bases[2], (14.8, 9.9), 1.55), thumb((17.8, 16.8), (23.8, 18.8), 1.35)); letters['Y'] = m; strokes['Y'] = []
    return letters, strokes


MODELS, STROKES = hand_models()
ROW_WEIGHT = np.where(Y < 20.0, 1.0, np.maximum(0.45, 1.0 - 0.09 * (Y - 20.0))).astype(np.float32)
COL_WEIGHT = (0.78 + 0.22 * np.clip(1.0 - ((X - 13.5) / 13.5) ** 2, 0.0, 1.0)).astype(np.float32)
SPATIAL_WEIGHT = ROW_WEIGHT * COL_WEIGHT


# ---------- first-layer features ----------

RAW_POS = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
RAW_NEG = -RAW_POS
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 4.0
SY = SX.T
D1 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], np.float32) / 4.0
D2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], np.float32) / 4.0
FEATURE_FILTERS = np.stack([RAW_POS, RAW_NEG, SX, -SX, SY, -SY, D1, -D1, D2, -D2]).astype(np.float32)


def conv_same_single(img, k):
    kh, kw = k.shape
    p = kh // 2
    pad = np.pad(img, p, mode="constant")
    out = np.zeros_like(img)
    for dy in range(kh):
        for dx in range(kw):
            out += k[dy, dx] * pad[dy:dy + H, dx:dx + W]
    return out


def feature_maps(img_norm):
    return np.maximum(0.0, np.stack([conv_same_single(img_norm, k) for k in FEATURE_FILTERS], axis=0))


# ---------- rendering ----------

def render_gray(letter, angle=0.0, scale=1.0, invert=False):
    mask = MODELS[letter]
    mask_t = affine(mask, angle_deg=angle, scale=scale, center=(14.0, 14.0))
    support = np.clip(blur3(dilate(mask_t, 2), 1) * 1.2, 0.0, 1.0)
    img = np.full((H, W), 0.50, np.float32)
    img = img + 0.28 * mask_t
    for (a, b, r) in STROKES[letter]:
        line = affine(capsule_mask(a[0], a[1], b[0], b[1], r), angle_deg=angle, scale=scale, center=(14.0, 14.0))
        img = np.where(line > 0.35, 0.34, img)
    # softly darken holes / gaps by reusing the mask background and blur
    img = blur3(img, 2)
    if invert:
        img = 1.0 - img
    return img.astype(np.float32), support.astype(np.float32)


# ---------- template creation ----------

def make_gray_template(letter, angle, scale, invert):
    img, support = render_gray(letter, angle=angle, scale=scale, invert=invert)
    x = normalize_image(img)
    ctx = np.clip(blur3(support, 1), 0.0, 1.0) * SPATIAL_WEIGHT
    t = x * ctx
    n = float(np.sqrt(np.sum(t * t)) + 1e-8)
    w = (t / n).astype(np.float32)
    self_resp = float(np.sum(x * w))
    return w, self_resp


def make_feature_template(letter, angle, scale, invert):
    img, support = render_gray(letter, angle=angle, scale=scale, invert=invert)
    x = normalize_image(img)
    feats = feature_maps(x)
    ctx = np.clip(blur3(support, 1), 0.0, 1.0) * SPATIAL_WEIGHT
    t = feats * ctx[None, :, :]
    n = float(np.sqrt(np.sum(t * t)) + 1e-8)
    w = (t / n).astype(np.float32)
    self_resp = float(np.sum(feats * w))
    return w, self_resp


# ---------- network build ----------

def build(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    angles = [-5.0, 0.0, 5.0]
    scales = [0.93, 1.0, 1.07]
    polarity = [False, True]

    gray_templates = []
    feat_templates = []
    gray_bias = []
    feat_bias = []
    owner = []

    for cls, letter in enumerate(LETTERS):
        for inv in polarity:
            for ang in angles:
                for sc in scales:
                    gw, gs = make_gray_template(letter, ang, sc, inv)
                    fw, fs = make_feature_template(letter, ang, sc, inv)
                    gray_templates.append(gw)
                    feat_templates.append(fw)
                    gray_bias.append(-0.58 * gs)
                    feat_bias.append(-0.66 * fs)
                    owner.append(cls)

    gray_templates = np.stack(gray_templates, axis=0).astype(np.float32)[:, None, :, :]
    feat_templates = np.stack(feat_templates, axis=0).astype(np.float32)
    gray_bias = np.array(gray_bias, np.float32)
    feat_bias = np.array(feat_bias, np.float32)
    owner = np.array(owner, np.int64)

    m = len(owner)
    net = {
        "layers": [
            {"name": "n0", "op": "norm", "inputs": ["image"]},
            {"name": "g0", "op": "conv", "inputs": ["n0"], "out": int(m), "kernel": [28, 28], "stride": 1, "padding": 6},
            {"name": "gp", "op": "maxpool", "inputs": ["g0"], "size": [13, 13], "stride": [13, 13]},
            {"name": "gr", "op": "relu", "inputs": ["gp"]},
            {"name": "gs", "op": "reshape", "inputs": ["gr"], "shape": [int(m)]},
            {"name": "f0", "op": "conv", "inputs": ["n0"], "out": 10, "kernel": [3, 3], "stride": 1, "padding": 1},
            {"name": "fr", "op": "relu", "inputs": ["f0"]},
            {"name": "t0", "op": "conv", "inputs": ["fr"], "out": int(m), "kernel": [28, 28], "stride": 1, "padding": 6},
            {"name": "tp", "op": "maxpool", "inputs": ["t0"], "size": [13, 13], "stride": [13, 13]},
            {"name": "tr", "op": "relu", "inputs": ["tp"]},
            {"name": "ts", "op": "reshape", "inputs": ["tr"], "shape": [int(m)]},
            {"name": "dg", "op": "dense", "inputs": ["gs"], "out": 24},
            {"name": "df", "op": "dense", "inputs": ["ts"], "out": 24},
            {"name": "out", "op": "add", "inputs": ["dg", "df"]},
        ]
    }

    dg_W = np.zeros((24, m), np.float32)
    df_W = np.zeros((24, m), np.float32)
    for j, cls in enumerate(owner.tolist()):
        dg_W[cls, j] = 0.75
        df_W[cls, j] = 1.10

    weights = {
        "g0.W": gray_templates,
        "g0.b": gray_bias,
        "f0.W": FEATURE_FILTERS[:, None, :, :].astype(np.float32),
        "f0.b": np.zeros((10,), np.float32),
        "t0.W": feat_templates.astype(np.float32),
        "t0.b": feat_bias,
        "dg.W": dg_W,
        "dg.b": np.zeros((24,), np.float32),
        "df.W": df_W,
        "df.b": np.zeros((24,), np.float32),
    }

    (out / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(out / "weights.npz", **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
