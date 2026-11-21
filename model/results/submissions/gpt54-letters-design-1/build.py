from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
KERNEL = 22
SCALES = [0.92, 1.00, 1.08]
OCC_THR = 0.15
EDGE_THR = 0.08
OCC_GAIN = 2.0
RING_GAIN = 0.85
EDGE_GAIN = 1.45


def conv2d_valid(x: np.ndarray, w: np.ndarray, b: np.ndarray, padding: int = 0) -> np.ndarray:
    c, h, ww = x.shape
    o, c2, k, k2 = w.shape
    assert c == c2 and k == k2
    xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="constant")
    ho = h + 2 * padding - k + 1
    wo = ww + 2 * padding - k + 1
    out = np.zeros((o, ho, wo), np.float32)
    for oy in range(ho):
        for ox in range(wo):
            patch = xp[:, oy:oy + k, ox:ox + k]
            out[:, oy, ox] = b + np.sum(w * patch[None, :, :, :], axis=(1, 2, 3))
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def norm_chw(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=(1, 2), keepdims=True)
    v = ((x - mu) ** 2).mean(axis=(1, 2), keepdims=True)
    return (x - mu) / np.sqrt(v + 1e-5)


def blur2d(x: np.ndarray) -> np.ndarray:
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.0
    y = np.pad(x, ((1, 1), (1, 1)), mode="constant")
    out = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * y[dy:dy + x.shape[0], dx:dx + x.shape[1]]
    return out


def dilate2d(x: np.ndarray, steps: int = 1) -> np.ndarray:
    out = x.copy()
    for _ in range(steps):
        y = np.pad(out, ((1, 1), (1, 1)), mode="constant")
        nbs = [y[dy:dy + out.shape[0], dx:dx + out.shape[1]] for dy in range(3) for dx in range(3)]
        out = np.maximum.reduce(nbs)
    return out


def normalize_map(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(np.abs(x)))
    return x / s if s > 1e-8 else x * 0.0


def bilinear_sample(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    h, w = img.shape
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = y - y0
    wx = x - x0
    y0c = np.clip(y0, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    y1c = np.clip(y1, 0, h - 1)
    x1c = np.clip(x1, 0, w - 1)
    v00 = img[y0c, x0c]
    v01 = img[y0c, x1c]
    v10 = img[y1c, x0c]
    v11 = img[y1c, x1c]
    valid = (y >= 0) & (y <= h - 1) & (x >= 0) & (x <= w - 1)
    out = ((1 - wy) * (1 - wx) * v00 + (1 - wy) * wx * v01 + wy * (1 - wx) * v10 + wy * wx * v11)
    return out * valid.astype(np.float32)


def transform_mask(mask: np.ndarray, scale: float = 1.0, angle_deg: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    h, w = mask.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y = yy - cy - shift[0]
    x = xx - cx - shift[1]
    a = np.deg2rad(-angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    xr = (ca * x - sa * y) / scale
    yr = (sa * x + ca * y) / scale
    src_x = xr + cx
    src_y = yr + cy
    return bilinear_sample(mask, src_y, src_x)


def chunk_mask(mask: np.ndarray, k: int = KERNEL) -> np.ndarray:
    start = (mask.shape[0] - k) // 2
    return mask[start:start + k, start:start + k]


def draw_palm(mask, X, Y, cx, cy, rx, ry, angle_deg=0.0):
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    dx = X - cx
    dy = Y - cy
    xr = ca * dx + sa * dy
    yr = -sa * dx + ca * dy
    q = np.sqrt((xr / max(rx, 1e-6)) ** 2 + (yr / max(ry, 1e-6)) ** 2)
    return np.maximum(mask, np.clip(1.0 - q, 0.0, 1.0))


def draw_capsule(mask, X, Y, x0, y0, x1, y1, r):
    vx = x1 - x0
    vy = y1 - y0
    den = vx * vx + vy * vy + 1e-9
    t = ((X - x0) * vx + (Y - y0) * vy) / den
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    d = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
    field = np.clip(1.0 - d / max(r, 1e-6), 0.0, 1.0)
    return np.maximum(mask, field)


def draw_finger(mask, X, Y, base_x, base_y, length, angle_deg, width, bend=0.0, frac=0.55):
    a = np.deg2rad(angle_deg)
    x1 = base_x + length * np.sin(a)
    y1 = base_y - length * np.cos(a)
    if abs(bend) < 1e-6:
        return draw_capsule(mask, X, Y, base_x, base_y, x1, y1, width)
    mlen = length * frac
    xm = base_x + mlen * np.sin(a)
    ym = base_y - mlen * np.cos(a)
    a2 = np.deg2rad(angle_deg + bend)
    x2 = xm + (length - mlen) * np.sin(a2)
    y2 = ym - (length - mlen) * np.cos(a2)
    mask = draw_capsule(mask, X, Y, base_x, base_y, xm, ym, width)
    mask = draw_capsule(mask, X, Y, xm, ym, x2, y2, width * 0.95)
    return mask


def make_canvas(size=28, supersample=6):
    hi = size * supersample
    xs = (np.arange(hi) + 0.5) / supersample
    ys = (np.arange(hi) + 0.5) / supersample
    X, Y = np.meshgrid(xs, ys)
    return np.zeros((hi, hi), np.float32), X, Y, supersample


def finish(mask, supersample):
    size = mask.shape[0] // supersample
    out = mask.reshape(size, supersample, size, supersample).mean((1, 3))
    return np.clip(out, 0.0, 1.0)


def base_mask(name: str) -> np.ndarray:
    m, X, Y, s = make_canvas()
    if name == 'A':
        m = draw_palm(m, X, Y, 14.5, 18.5, 6.2, 6.8, -6)
        for bx in [10.0, 12.0, 14.0, 16.0]:
            m = draw_finger(m, X, Y, bx, 15.5, 3.2, 6, 1.8, bend=-65)
        m = draw_finger(m, X, Y, 9.0, 18.6, 6.2, 232, 1.9, bend=15, frac=0.5)
    elif name == 'B':
        m = draw_palm(m, X, Y, 14.1, 18.8, 5.3, 7.0, 0)
        for bx, ln in zip([10.2, 12.0, 13.8, 15.6], [8.0, 9.0, 9.0, 8.2]):
            m = draw_finger(m, X, Y, bx, 17.0, ln, -1, 1.45)
        m = draw_finger(m, X, Y, 9.3, 19.5, 5.1, 115, 1.55, bend=-20, frac=0.5)
    elif name == 'C':
        m = np.maximum(m, np.clip(1.0 - np.sqrt((((np.cos(np.deg2rad(-18)) * (X - 13.8) + np.sin(np.deg2rad(-18)) * (Y - 15.5)) / 8.0) ** 2) + (((-np.sin(np.deg2rad(-18)) * (X - 13.8) + np.cos(np.deg2rad(-18)) * (Y - 15.5)) / 9.8) ** 2)), 0.0, 1.0))
        inner = np.zeros_like(m)
        inner = draw_palm(inner, X, Y, 16.4, 15.5, 5.4, 6.9, -18)
        blocker = draw_palm(np.zeros_like(m), X, Y, 18.9, 15.1, 3.6, 9.0, -18)
        m = m * (1.0 - np.clip(inner, 0.0, 1.0))
        m = m * (1.0 - np.clip(blocker, 0.0, 1.0))
        m = draw_capsule(m, X, Y, 9.2, 21.0, 16.4, 23.2, 2.0)
    elif name == 'D':
        m = draw_palm(m, X, Y, 14.0, 19.0, 4.7, 5.8, 0)
        m = draw_finger(m, X, Y, 14.0, 16.5, 10.0, 0, 1.55)
        for bx in [11.7, 13.5, 15.3]:
            m = draw_finger(m, X, Y, bx, 17.0, 3.3, 10, 1.55, bend=-75)
        m = draw_finger(m, X, Y, 10.6, 19.1, 5.0, 55, 1.5, bend=25, frac=0.5)
        hole = draw_palm(np.zeros_like(m), X, Y, 12.5, 18.0, 1.75, 1.75, 0)
        m = m * (1.0 - hole)
    elif name == 'E':
        m = draw_palm(m, X, Y, 14.4, 18.8, 6.1, 6.8, -4)
        for bx, ln, ang in [(10.4, 4.0, 8), (12.3, 4.2, 3), (14.1, 4.0, -2), (15.8, 3.6, -6)]:
            m = draw_finger(m, X, Y, bx, 16.1, ln, ang, 1.75, bend=-95, frac=0.45)
        m = draw_finger(m, X, Y, 9.9, 20.0, 3.8, 110, 1.7, bend=-30, frac=0.5)
    elif name == 'F':
        m = draw_palm(m, X, Y, 14.0, 18.6, 5.0, 6.3, 0)
        for bx, ln, ang in [(12.5, 7.7, -12), (14.3, 8.2, 0), (16.1, 7.8, 10)]:
            m = draw_finger(m, X, Y, bx, 16.8, ln, ang, 1.35)
        m = draw_finger(m, X, Y, 10.9, 17.6, 4.0, 18, 1.35, bend=-80)
        m = draw_finger(m, X, Y, 10.3, 19.0, 4.4, 48, 1.35, bend=15, frac=0.5)
        hole = draw_palm(np.zeros_like(m), X, Y, 11.8, 17.4, 1.35, 1.35, 0)
        m = m * (1.0 - hole)
    elif name == 'G':
        m = draw_palm(m, X, Y, 10.8, 17.8, 4.6, 5.4, -68)
        m = draw_finger(m, X, Y, 11.0, 15.8, 9.8, 84, 1.45)
        m = draw_finger(m, X, Y, 10.3, 18.8, 6.0, 55, 1.45, frac=0.5)
        for bx in [8.6, 9.9, 11.2]:
            m = draw_finger(m, X, Y, bx, 18.4, 2.6, 15, 1.55, bend=-80)
    elif name == 'H':
        m = draw_palm(m, X, Y, 10.7, 18.4, 4.8, 5.5, -68)
        m = draw_finger(m, X, Y, 10.2, 15.9, 9.2, 84, 1.35)
        m = draw_finger(m, X, Y, 10.3, 17.6, 8.6, 84, 1.35)
        m = draw_finger(m, X, Y, 10.0, 19.0, 4.0, 55, 1.3, frac=0.5)
        for bx in [8.8, 10.0]:
            m = draw_finger(m, X, Y, bx, 19.1, 2.4, 20, 1.45, bend=-85)
    elif name == 'I':
        m = draw_palm(m, X, Y, 14.0, 18.8, 5.4, 6.3, 0)
        for bx in [10.8, 12.7, 14.4]:
            m = draw_finger(m, X, Y, bx, 16.4, 3.2, 6, 1.6, bend=-70)
        m = draw_finger(m, X, Y, 17.0, 16.8, 9.2, -4, 1.45)
        m = draw_finger(m, X, Y, 10.0, 19.4, 4.4, 118, 1.5, bend=-25, frac=0.5)
    elif name == 'K':
        m = draw_palm(m, X, Y, 13.8, 19.3, 4.7, 5.8, 0)
        m = draw_finger(m, X, Y, 13.1, 17.2, 9.0, -18, 1.4)
        m = draw_finger(m, X, Y, 14.8, 17.1, 8.2, 16, 1.4)
        m = draw_finger(m, X, Y, 10.2, 19.6, 7.2, 58, 1.45, frac=0.5)
        for bx in [11.6, 13.2]:
            m = draw_finger(m, X, Y, bx, 18.3, 2.5, 5, 1.45, bend=-80)
    elif name == 'L':
        m = draw_palm(m, X, Y, 13.8, 19.0, 4.7, 6.0, 0)
        for bx in [11.6, 13.2, 14.8]:
            m = draw_finger(m, X, Y, bx, 17.0, 2.8, 6, 1.45, bend=-80)
        m = draw_finger(m, X, Y, 15.8, 16.8, 9.2, -1, 1.45)
        m = draw_finger(m, X, Y, 10.2, 19.6, 7.8, 90, 1.55, frac=0.5)
    elif name == 'M':
        m = draw_palm(m, X, Y, 14.4, 18.8, 6.0, 7.0, -2)
        for bx in [11.0, 13.0, 15.0, 17.0]:
            m = draw_finger(m, X, Y, bx, 16.6, 4.2, 3, 1.7, bend=-78)
        m = draw_finger(m, X, Y, 11.2, 20.1, 5.8, 102, 1.3, bend=-28, frac=0.5)
    elif name == 'N':
        m = draw_palm(m, X, Y, 14.2, 18.8, 5.9, 6.8, -2)
        for bx, ln in [(11.5, 4.2), (13.7, 4.0), (15.8, 3.4), (17.2, 3.0)]:
            m = draw_finger(m, X, Y, bx, 16.5, ln, 3, 1.65, bend=-82)
        m = draw_finger(m, X, Y, 12.0, 20.0, 4.9, 100, 1.3, bend=-24, frac=0.5)
    elif name == 'O':
        m = draw_palm(m, X, Y, 14.0, 16.0, 6.7, 8.2, 0)
        hole = draw_palm(np.zeros_like(m), X, Y, 14.0, 16.0, 3.2, 4.0, 0)
        m = m * (1.0 - hole)
        m = draw_capsule(m, X, Y, 11.0, 22.0, 17.0, 22.0, 2.2)
    elif name == 'P':
        m = draw_palm(m, X, Y, 13.0, 14.8, 4.8, 5.7, 12)
        m = draw_finger(m, X, Y, 12.2, 16.0, 8.6, 145, 1.35)
        m = draw_finger(m, X, Y, 13.9, 15.7, 8.0, 112, 1.35)
        m = draw_finger(m, X, Y, 10.0, 13.4, 6.4, 45, 1.35, frac=0.5)
        for bx in [12.0, 13.5]:
            m = draw_finger(m, X, Y, bx, 13.7, 2.6, 20, 1.45, bend=-80)
    elif name == 'Q':
        m = draw_palm(m, X, Y, 13.0, 14.8, 4.8, 5.7, 12)
        m = draw_finger(m, X, Y, 12.6, 15.0, 9.0, 135, 1.45)
        m = draw_finger(m, X, Y, 10.2, 13.8, 5.8, 40, 1.45, frac=0.5)
        for bx in [11.8, 13.0, 14.3]:
            m = draw_finger(m, X, Y, bx, 13.7, 2.4, 15, 1.5, bend=-82)
    elif name == 'R':
        m = draw_palm(m, X, Y, 13.8, 19.2, 4.6, 5.8, 0)
        m = draw_finger(m, X, Y, 13.2, 16.9, 8.8, -8, 1.35)
        m = draw_finger(m, X, Y, 14.9, 17.3, 8.2, 10, 1.35)
        m = draw_finger(m, X, Y, 10.4, 19.4, 4.6, 108, 1.35, frac=0.5)
        m = draw_capsule(m, X, Y, 13.8, 13.0, 14.9, 17.0, 1.05)
        for bx in [11.8, 13.3]:
            m = draw_finger(m, X, Y, bx, 18.2, 2.2, 4, 1.4, bend=-85)
    elif name == 'S':
        m = draw_palm(m, X, Y, 14.4, 18.7, 6.0, 6.8, -2)
        for bx in [10.7, 12.6, 14.5, 16.3]:
            m = draw_finger(m, X, Y, bx, 16.2, 3.6, 6, 1.75, bend=-80)
        m = draw_finger(m, X, Y, 10.5, 19.0, 5.7, 70, 1.55, bend=18, frac=0.5)
    elif name == 'T':
        m = draw_palm(m, X, Y, 14.1, 18.7, 5.9, 6.8, -2)
        for bx in [11.0, 12.8, 14.6, 16.3]:
            m = draw_finger(m, X, Y, bx, 16.4, 3.5, 6, 1.7, bend=-82)
        m = draw_finger(m, X, Y, 11.4, 18.1, 6.0, 92, 1.2, frac=0.5)
    elif name == 'U':
        m = draw_palm(m, X, Y, 13.9, 19.2, 4.5, 5.8, 0)
        m = draw_finger(m, X, Y, 13.2, 17.0, 8.8, -2, 1.35)
        m = draw_finger(m, X, Y, 14.9, 17.0, 8.8, 2, 1.35)
        m = draw_finger(m, X, Y, 10.4, 19.4, 4.4, 106, 1.35, frac=0.5)
        for bx in [11.6, 15.7]:
            m = draw_finger(m, X, Y, bx, 18.0, 2.2, 6, 1.35, bend=-86)
    elif name == 'V':
        m = draw_palm(m, X, Y, 13.9, 19.2, 4.5, 5.8, 0)
        m = draw_finger(m, X, Y, 13.0, 17.0, 9.0, -14, 1.35)
        m = draw_finger(m, X, Y, 15.1, 17.0, 8.9, 12, 1.35)
        m = draw_finger(m, X, Y, 10.4, 19.4, 4.4, 106, 1.35, frac=0.5)
        for bx in [11.6, 15.8]:
            m = draw_finger(m, X, Y, bx, 18.0, 2.2, 6, 1.35, bend=-86)
    elif name == 'W':
        m = draw_palm(m, X, Y, 14.0, 19.4, 4.6, 5.7, 0)
        for bx, ang in [(12.1, -12), (14.0, 0), (15.9, 12)]:
            m = draw_finger(m, X, Y, bx, 17.2, 8.6, ang, 1.3)
        m = draw_finger(m, X, Y, 10.5, 19.6, 4.0, 108, 1.3, frac=0.5)
        m = draw_finger(m, X, Y, 17.3, 18.4, 2.4, 12, 1.25, bend=-84)
    elif name == 'X':
        m = draw_palm(m, X, Y, 14.0, 19.0, 5.1, 6.2, 0)
        m = draw_finger(m, X, Y, 14.5, 16.8, 8.2, -6, 1.45, bend=42, frac=0.62)
        for bx in [11.5, 13.2, 16.3]:
            m = draw_finger(m, X, Y, bx, 17.3, 2.8, 6, 1.5, bend=-82)
        m = draw_finger(m, X, Y, 10.3, 19.4, 4.6, 112, 1.4, bend=-18, frac=0.5)
    elif name == 'Y':
        m = draw_palm(m, X, Y, 13.7, 19.0, 4.7, 5.9, 0)
        for bx in [11.8, 13.5, 15.2]:
            m = draw_finger(m, X, Y, bx, 17.0, 2.6, 6, 1.4, bend=-82)
        m = draw_finger(m, X, Y, 16.7, 16.9, 8.4, 10, 1.35)
        m = draw_finger(m, X, Y, 10.4, 19.2, 7.5, 98, 1.35, frac=0.5)
    else:
        raise ValueError(name)
    return finish(m, s)


def feature_frontend_weights() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 4.0
    ky = kx.T
    kd1 = (kx + ky) / np.sqrt(2.0)
    kd2 = (kx - ky) / np.sqrt(2.0)
    blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.0

    w0 = np.zeros((10, 1, 3, 3), np.float32)
    b0 = np.zeros((10,), np.float32)
    w0[0, 0] = -np.ones((3, 3), np.float32) / 9.0
    w0[1, 0] = np.ones((3, 3), np.float32) / 9.0
    b0[0] = -OCC_THR
    b0[1] = -OCC_THR
    for i, kk in [(2, kx), (3, -kx), (4, ky), (5, -ky), (6, kd1), (7, -kd1), (8, kd2), (9, -kd2)]:
        w0[i, 0] = kk
        b0[i] = -EDGE_THR

    w1 = np.zeros((10, 10, 3, 3), np.float32)
    b1 = np.zeros((10,), np.float32)
    for i in range(10):
        w1[i, i] = blur
    return w0, b0, w1, b1


def apply_frontend(mask: np.ndarray, w0: np.ndarray, b0: np.ndarray, w1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    img = 0.8 - 0.6 * mask
    x = norm_chw(img[None].astype(np.float32))
    y = relu(conv2d_valid(x, w0, b0, padding=1))
    z = conv2d_valid(y, w1, b1, padding=1)
    return z


def make_match_filter(mask: np.ndarray, w0: np.ndarray, b0: np.ndarray, w1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    feat = apply_frontend(mask, w0, b0, w1, b1)
    occ = feat[0] + feat[1]
    vx = feat[2] + feat[3]
    hy = feat[4] + feat[5]
    d1 = feat[6] + feat[7]
    d2 = feat[8] + feat[9]

    occ_c = chunk_mask(occ)
    vx_c = chunk_mask(vx)
    hy_c = chunk_mask(hy)
    d1_c = chunk_mask(d1)
    d2_c = chunk_mask(d2)
    mask_c = chunk_mask(mask)
    ring = np.clip(dilate2d(mask_c, 1) - mask_c, 0.0, 1.0)
    ring = blur2d(ring)

    edge_total = float(vx_c.sum() + hy_c.sum() + d1_c.sum() + d2_c.sum())
    if edge_total < 1e-8:
        edge_total = 1.0

    f = np.zeros((10, KERNEL, KERNEL), np.float32)
    occ_w = OCC_GAIN * normalize_map(occ_c) - RING_GAIN * normalize_map(ring)
    f[0] = occ_w
    f[1] = occ_w
    f[2] = f[3] = EDGE_GAIN * (vx_c / edge_total)
    f[4] = f[5] = EDGE_GAIN * (hy_c / edge_total)
    f[6] = f[7] = EDGE_GAIN * (d1_c / edge_total)
    f[8] = f[9] = EDGE_GAIN * (d2_c / edge_total)
    return f


def build_network_and_weights() -> tuple[dict, dict[str, np.ndarray]]:
    w0, b0, w1, b1 = feature_frontend_weights()
    base = {lab: base_mask(lab) for lab in LABELS}

    protos = []
    for ci, lab in enumerate(LABELS):
        for scale in SCALES:
            protos.append((ci, lab, scale, make_match_filter(transform_mask(base[lab], scale=scale), w0, b0, w1, b1)))

    m = len(protos)
    wm = np.zeros((m, 10, KERNEL, KERNEL), np.float32)
    bm = np.zeros((m,), np.float32)
    for i, (_ci, _lab, _sc, filt) in enumerate(protos):
        wm[i] = filt
        bm[i] = 0.0

    wd = np.zeros((24, m), np.float32)
    bd = np.zeros((24,), np.float32)
    for i, (ci, _lab, _sc, _filt) in enumerate(protos):
        wd[ci, i] = 1.0

    net = {"layers": [
        {"name": "norm0", "op": "norm", "inputs": ["image"]},
        {"name": "feat0", "op": "conv", "inputs": ["norm0"], "out": 10, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "feat1", "op": "relu", "inputs": ["feat0"]},
        {"name": "feat2", "op": "conv", "inputs": ["feat1"], "out": 10, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "match0", "op": "conv", "inputs": ["feat2"], "out": m, "kernel": [KERNEL, KERNEL], "stride": 1, "padding": 0},
        {"name": "match1", "op": "relu", "inputs": ["match0"]},
        {"name": "pool", "op": "maxpool", "inputs": ["match1"], "size": [7, 7], "stride": [7, 7]},
        {"name": "scores", "op": "dense", "inputs": ["pool"], "out": 24},
    ]}
    weights = {
        "feat0.W": w0,
        "feat0.b": b0,
        "feat2.W": w1,
        "feat2.b": b1,
        "match0.W": wm,
        "match0.b": bm,
        "scores.W": wd,
        "scores.b": bd,
    }
    return net, weights


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    net, weights = build_network_and_weights()
    (out_dir / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(out_dir / "weights.npz", **weights)


if __name__ == "__main__":
    build(Path(sys.argv[1]))
