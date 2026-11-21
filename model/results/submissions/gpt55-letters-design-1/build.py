import json, sys, math
from pathlib import Path
import numpy as np

# Data-free builder for a hand-designed ASL classifier.
# All numbers below are analytic drawing parameters for simple hand/sign shapes.
IMG = 28
POOL = 2
CROP = IMG // POOL
EPS = 1e-5
NCLASS = 24
VARIANTS = 128
POLARITIES = 2
MIRRORS = 2
TPER = VARIANTS * POLARITIES * MIRRORS
MIRROR_CLASSES = {0, 1, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def soft_cap(d, r, s=0.035):
    return np.clip(0.5 + (r - d) / s, 0.0, 1.0)


def capsule(X, Y, x1, y1, x2, y2, r, s=0.035):
    vx, vy = x2 - x1, y2 - y1
    vv = vx * vx + vy * vy + 1e-9
    t = ((X - x1) * vx + (Y - y1) * vy) / vv
    t = np.clip(t, 0.0, 1.0)
    px = x1 + t * vx
    py = y1 + t * vy
    d = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
    return soft_cap(d, r, s)


def ellipse(X, Y, cx, cy, rx, ry, ang=0.0, s=0.075):
    ca, sa = math.cos(ang), math.sin(ang)
    x = X - cx
    y = Y - cy
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    e = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    return np.clip(0.5 + (1.0 - e) / s, 0.0, 1.0)


def add(m, p):
    return np.maximum(m, p)


def cut(m, p, strength=1.0):
    return m * (1.0 - strength * np.clip(p, 0.0, 1.0))


def grid(dx=0.0, dy=0.0, scale=1.0, rot=0.0):
    a = (np.arange(IMG, dtype=np.float32) + 0.5 - IMG / 2.0) / (IMG / 2.0)
    X, Y = np.meshgrid(a, a)
    x = X - dx
    y = Y - dy
    ca, sa = math.cos(rot), math.sin(rot)
    # inverse transform from image coordinates to canonical sign coordinates
    return (ca * x + sa * y) / scale, (-sa * x + ca * y) / scale


FX = [-0.24, -0.08, 0.08, 0.235]
FW = [0.086, 0.094, 0.090, 0.080]
FL = [0.76, 0.88, 0.81, 0.68]


def palm(X, Y, narrow=0.0):
    m = np.zeros_like(X)
    m = add(m, ellipse(X, Y, 0.00, 0.36, 0.42 - 0.04 * narrow, 0.47, 0.02))
    m = add(m, capsule(X, Y, -0.11, 0.69, -0.02, 1.16, 0.22 - 0.03 * narrow))
    return m


def finger(m, cre, X, Y, i, angle=0.0, length=1.0, basey=0.08, width=1.0):
    x0 = FX[i]
    L = FL[i] * length
    x1, y1 = x0, basey
    x2 = x0 + math.sin(angle) * L
    y2 = basey - math.cos(angle) * L
    r = FW[i] * width
    m = add(m, capsule(X, Y, x1, y1, x2, y2, r, 0.032))
    m = add(m, ellipse(X, Y, x2, y2, r * 1.05, r * 1.16, angle, 0.10))
    return m, cre


def curled(m, cre, X, Y, i, low=0.0, tight=0.0):
    x = FX[i]
    # Curled fingertips sit high across the top of a fist in the real signs.
    y = -0.28 + 0.10 * low
    m = add(m, ellipse(X, Y, x, y, 0.120 - 0.010 * tight, 0.150 - 0.015 * tight, -0.12 + 0.07 * i, 0.075))
    m = add(m, capsule(X, Y, x - 0.070, y - 0.02, x + 0.060, y + 0.150, 0.076, 0.035))
    cre = add(cre, capsule(X, Y, x + 0.065, y - 0.075, x + 0.045, y + 0.140, 0.014, 0.022))
    return m, cre


def thumb_across(m, cre, X, Y, high=0.0, long=1.0):
    m = add(m, capsule(X, Y, -0.39, 0.33 - 0.04 * high, 0.18 * long, 0.45 - 0.02 * high, 0.105, 0.035))
    m = add(m, ellipse(X, Y, 0.20 * long, 0.45 - 0.02 * high, 0.11, 0.085, 0.2, 0.08))
    cre = add(cre, capsule(X, Y, -0.30, 0.28 - 0.03 * high, 0.08 * long, 0.38, 0.016, 0.020))
    return m, cre


def thumb_side(m, cre, X, Y, up=0.0):
    m = add(m, capsule(X, Y, -0.40, 0.34, -0.72, 0.16 - 0.10 * up, 0.105, 0.035))
    m = add(m, ellipse(X, Y, -0.74, 0.15 - 0.10 * up, 0.11, 0.085, -0.55, 0.08))
    cre = add(cre, capsule(X, Y, -0.38, 0.25, -0.65, 0.13 - 0.07 * up, 0.015, 0.020))
    return m, cre


def thumb_vertical(m, cre, X, Y):
    m = add(m, capsule(X, Y, -0.39, 0.32, -0.37, -0.10, 0.100, 0.035))
    m = add(m, ellipse(X, Y, -0.37, -0.12, 0.095, 0.105, 0.0, 0.08))
    cre = add(cre, capsule(X, Y, -0.30, 0.23, -0.31, -0.04, 0.014, 0.020))
    return m, cre


def fist(X, Y, kind=0):
    m = palm(X, Y, 0.05)
    cre = np.zeros_like(X)
    for i in range(4):
        m, cre = curled(m, cre, X, Y, i, low=0.15 if kind in (2, 3) else 0.0, tight=0.25 if kind == 2 else 0.0)
    if kind == 0:      # A: thumb along the side
        m, cre = thumb_vertical(m, cre, X, Y)
    elif kind == 1:    # S: thumb across the front
        m, cre = thumb_across(m, cre, X, Y, high=0.0, long=1.05)
    elif kind == 2:    # E: curled fingers held lower, thumb high
        m, cre = thumb_across(m, cre, X, Y, high=1.0, long=0.85)
    elif kind == 3:    # M: thumb mostly hidden below three fingers
        m, cre = thumb_across(m, cre, X, Y, high=-0.4, long=0.65)
        cre = add(cre, capsule(X, Y, -0.18, -0.06, -0.18, 0.20, 0.018, 0.020))
        cre = add(cre, capsule(X, Y, -0.02, -0.06, -0.02, 0.20, 0.018, 0.020))
    elif kind == 4:    # N: thumb hidden below two fingers
        m, cre = thumb_across(m, cre, X, Y, high=-0.2, long=0.72)
        cre = add(cre, capsule(X, Y, -0.10, -0.08, -0.10, 0.22, 0.020, 0.020))
    elif kind == 5:    # T: thumb pokes between first two fingers
        m, cre = thumb_across(m, cre, X, Y, high=0.2, long=0.55)
        m = add(m, ellipse(X, Y, -0.15, -0.04, 0.095, 0.155, 0.0, 0.08))
        cre = add(cre, capsule(X, Y, -0.07, -0.14, -0.08, 0.15, 0.018, 0.020))
    return m, cre


def ring_shape(X, Y, closed=True):
    # Curved hand for O/C, with a hollow centre and wrist.
    outer = ellipse(X, Y, 0.04, 0.18, 0.48, 0.63, 0.03, 0.08)
    inner = ellipse(X, Y, 0.14, 0.17, 0.28, 0.37, 0.02, 0.08)
    if closed:
        gap = np.zeros_like(X)
    else:
        gap = ellipse(X, Y, 0.46, 0.10, 0.22, 0.36, 0.0, 0.08)
    m = outer * (1.0 - 0.98 * inner) * (1.0 - 0.95 * gap)
    m = add(m, capsule(X, Y, -0.16, 0.66, -0.08, 1.13, 0.20, 0.035))
    cre = ellipse(X, Y, 0.14, 0.17, 0.30, 0.39, 0.02, 0.05)
    if not closed:
        cre = cre * (1.0 - gap)
    return np.clip(m, 0, 1), np.clip(cre, 0, 1)


def horizontal_hand(X, Y, two=False, down=False):
    m = np.zeros_like(X)
    cre = np.zeros_like(X)
    # canonical points right; optional down is made by using a vertical finger path
    if not down:
        m = add(m, ellipse(X, Y, -0.40, 0.28, 0.34, 0.38, -0.12, 0.08))
        m = add(m, capsule(X, Y, -0.82, 0.45, -0.35, 0.42, 0.18, 0.035))
        m = add(m, capsule(X, Y, -0.25, 0.02, 0.78, 0.00, 0.075, 0.030))
        if two:
            m = add(m, capsule(X, Y, -0.27, 0.14, 0.76, 0.13, 0.073, 0.030))
            cre = add(cre, capsule(X, Y, -0.18, 0.08, 0.72, 0.07, 0.012, 0.018))
        m = add(m, capsule(X, Y, -0.31, 0.26, 0.23, 0.30, 0.090, 0.030))
    else:
        m = add(m, ellipse(X, Y, -0.18, -0.28, 0.36, 0.34, 0.20, 0.08))
        m = add(m, capsule(X, Y, -0.58, -0.34, -0.92, -0.48, 0.18, 0.035))
        m = add(m, capsule(X, Y, -0.10, -0.06, 0.10, 0.86, 0.075, 0.030))
        if two:
            m = add(m, capsule(X, Y, -0.24, -0.04, -0.03, 0.84, 0.073, 0.030))
            cre = add(cre, capsule(X, Y, -0.17, -0.04, 0.04, 0.78, 0.012, 0.018))
        m = add(m, capsule(X, Y, -0.28, -0.10, 0.02, 0.45, 0.090, 0.030))
    return m, cre


def render_mask(cls, X, Y, shape=0.0):
    m = np.zeros_like(X)
    cre = np.zeros_like(X)
    if cls == 0:
        return fist(X, Y, 0)
    if cls == 1:  # B
        m = palm(X, Y, -0.05)
        for i, a in enumerate([-0.05, -0.015, 0.015, 0.045]):
            m, cre = finger(m, cre, X, Y, i, a, 1.03, basey=0.07, width=0.98)
        for x in [-0.16, 0.00, 0.16]:
            cre = add(cre, capsule(X, Y, x, -0.67, x, 0.02, 0.012, 0.018))
        m, cre = thumb_across(m, cre, X, Y, high=0.8, long=0.70)
    elif cls == 2:
        return ring_shape(X, Y, closed=False)
    elif cls == 3:  # D
        m = palm(X, Y, 0.05)
        m, cre = finger(m, cre, X, Y, 0, 0.02, 1.05, basey=0.06, width=0.95)
        for i in [1, 2, 3]:
            m, cre = curled(m, cre, X, Y, i, low=0.20, tight=0.2)
        ring = ellipse(X, Y, -0.06, 0.20, 0.25, 0.25, 0.0, 0.08)
        hole = ellipse(X, Y, -0.04, 0.20, 0.13, 0.14, 0.0, 0.08)
        m = add(m, ring * (1 - 0.9 * hole))
        cre = add(cre, hole)
        m, cre = thumb_side(m, cre, X, Y, up=-0.2)
    elif cls == 4:
        return fist(X, Y, 2)
    elif cls == 5:  # F
        m = palm(X, Y, 0.02)
        for i, a in zip([1, 2, 3], [-0.04, 0.03, 0.09]):
            m, cre = finger(m, cre, X, Y, i, a, 0.98, basey=0.06, width=0.95)
        ring = ellipse(X, Y, -0.27, 0.19, 0.20, 0.21, -0.2, 0.08)
        hole = ellipse(X, Y, -0.25, 0.18, 0.095, 0.11, -0.2, 0.08)
        m = add(m, ring * (1 - 0.95 * hole))
        cre = add(cre, hole)
        m = add(m, capsule(X, Y, -0.42, 0.34, -0.28, 0.21, 0.090, 0.030))
    elif cls == 6:
        return horizontal_hand(X, Y, two=False, down=False)
    elif cls == 7:
        return horizontal_hand(X, Y, two=True, down=False)
    elif cls == 8:  # I
        m = palm(X, Y, 0.05)
        for i in [0, 1, 2]:
            m, cre = curled(m, cre, X, Y, i, low=0.05, tight=0.15)
        m, cre = finger(m, cre, X, Y, 3, 0.07, 0.98, basey=0.08, width=0.86)
        m, cre = thumb_across(m, cre, X, Y, high=0.3, long=0.85)
    elif cls == 9:  # K
        m = palm(X, Y, 0.02)
        m, cre = finger(m, cre, X, Y, 0, -0.23, 1.00, basey=0.08, width=0.93)
        m, cre = finger(m, cre, X, Y, 1, 0.17, 1.03, basey=0.08, width=0.93)
        for i in [2, 3]:
            m, cre = curled(m, cre, X, Y, i, low=0.10, tight=0.1)
        m = add(m, capsule(X, Y, -0.36, 0.36, 0.03, 0.02, 0.095, 0.030))
        cre = add(cre, capsule(X, Y, -0.20, 0.20, 0.02, 0.02, 0.014, 0.020))
    elif cls == 10:  # L
        m = palm(X, Y, 0.06)
        m, cre = finger(m, cre, X, Y, 0, 0.00, 1.08, basey=0.06, width=0.96)
        for i in [1, 2, 3]:
            m, cre = curled(m, cre, X, Y, i, low=0.12, tight=0.05)
        m, cre = thumb_side(m, cre, X, Y, up=0.1)
    elif cls == 11:
        return fist(X, Y, 3)
    elif cls == 12:
        return fist(X, Y, 4)
    elif cls == 13:
        return ring_shape(X, Y, closed=True)
    elif cls == 14:  # P
        m, cre = horizontal_hand(X, Y, two=True, down=True)
        m = add(m, capsule(X, Y, -0.28, -0.13, 0.18, 0.14, 0.090, 0.030))
    elif cls == 15:  # Q
        return horizontal_hand(X, Y, two=False, down=True)
    elif cls == 16:  # R
        m = palm(X, Y, 0.02)
        m = add(m, capsule(X, Y, -0.13, 0.08, 0.09, -0.68, 0.086, 0.030))
        m = add(m, capsule(X, Y, 0.06, 0.08, -0.05, -0.67, 0.086, 0.030))
        cre = add(cre, capsule(X, Y, -0.05, -0.36, 0.05, -0.18, 0.014, 0.020))
        for i in [2, 3]:
            m, cre = curled(m, cre, X, Y, i, low=0.12, tight=0.1)
        m, cre = thumb_across(m, cre, X, Y, high=0.4, long=0.65)
    elif cls == 17:
        return fist(X, Y, 1)
    elif cls == 18:
        return fist(X, Y, 5)
    elif cls == 19:  # U
        m = palm(X, Y, 0.04)
        # two close upright fingers
        m, cre = finger(m, cre, X, Y, 0, -0.03, 1.03, basey=0.07, width=0.95)
        m, cre = finger(m, cre, X, Y, 1, 0.02, 1.00, basey=0.07, width=0.95)
        for i in [2, 3]:
            m, cre = curled(m, cre, X, Y, i, low=0.12, tight=0.1)
        cre = add(cre, capsule(X, Y, -0.16, -0.65, -0.02, 0.04, 0.012, 0.018))
        m, cre = thumb_across(m, cre, X, Y, high=0.4, long=0.70)
    elif cls == 20:  # V
        m = palm(X, Y, 0.04)
        m, cre = finger(m, cre, X, Y, 0, -0.27, 1.02, basey=0.08, width=0.92)
        m, cre = finger(m, cre, X, Y, 1, 0.20, 1.02, basey=0.08, width=0.92)
        for i in [2, 3]:
            m, cre = curled(m, cre, X, Y, i, low=0.12, tight=0.1)
        m, cre = thumb_across(m, cre, X, Y, high=0.4, long=0.68)
    elif cls == 21:  # W
        m = palm(X, Y, 0.02)
        for i, a in zip([0, 1, 2], [-0.25, 0.02, 0.23]):
            m, cre = finger(m, cre, X, Y, i, a, 1.00, basey=0.08, width=0.90)
        m, cre = curled(m, cre, X, Y, 3, low=0.15, tight=0.1)
        m, cre = thumb_across(m, cre, X, Y, high=0.35, long=0.68)
    elif cls == 22:  # X
        m = palm(X, Y, 0.06)
        m = add(m, capsule(X, Y, -0.20, 0.09, -0.18, -0.34, 0.090, 0.030))
        m = add(m, capsule(X, Y, -0.18, -0.34, 0.06, -0.24, 0.084, 0.030))
        m = add(m, ellipse(X, Y, 0.07, -0.24, 0.090, 0.080, 0.45, 0.08))
        cre = add(cre, capsule(X, Y, -0.14, -0.30, 0.02, -0.24, 0.014, 0.020))
        for i in [1, 2, 3]:
            m, cre = curled(m, cre, X, Y, i, low=0.12, tight=0.1)
        m, cre = thumb_across(m, cre, X, Y, high=0.2, long=0.75)
    elif cls == 23:  # Y
        m = palm(X, Y, 0.06)
        for i in [0, 1, 2]:
            m, cre = curled(m, cre, X, Y, i, low=0.10, tight=0.1)
        m, cre = finger(m, cre, X, Y, 3, 0.18, 1.02, basey=0.08, width=0.86)
        m, cre = thumb_side(m, cre, X, Y, up=-0.1)
    return np.clip(m, 0, 1), np.clip(cre, 0, 1)


def make_image(cls, dx, dy, sc, rot, shape=0.0, polarity=0, mirror=0):
    X, Y = grid(dx, dy, sc, rot)
    if mirror:
        X = -X
    mask, crease = render_mask(cls, X, Y, shape)
    # The foreground hand region has one sign; hollows and crease/shadow lines
    # inside the hand have the opposite sign.  This makes O/F/D-like holes
    # evidence only when the centre is background-like, not filled like a fist.
    img = 1.25 * mask - 0.55 * crease
    if polarity:
        img = -img
    return img.astype(np.float32)


def filters():
    F = []
    z = np.zeros((3, 3), np.float32)
    k = z.copy(); k[1, 1] = 1.0; F.append(k)
    F.append(np.ones((3, 3), np.float32) / 9.0)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 4.0
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32) / 4.0
    F += [sx, sy, (sx + sy) / math.sqrt(2), (sx - sy) / math.sqrt(2)]
    F.append(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32) / 4.0)
    F.append(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32) / 8.0)
    return np.stack([f.astype(np.float32) for f in F]).astype(np.float32)


FILT = filters()
NF = FILT.shape[0]


def raw_features(img):
    # Positive support mask (and its negative-polarity counterpart) rather than a
    # zero-mean picture: this asks whether the expected hand/finger region is
    # brighter or darker than the image mean without penalizing arbitrary
    # background outside the hand.
    x = img.astype(np.float32)
    x = x / np.sqrt((x * x).mean() + EPS)
    return x.reshape(-1).astype(np.float32)


def features(img):
    x = img.astype(np.float32)
    x = (x - x.mean()) / np.sqrt(((x - x.mean()) ** 2).mean() + EPS)
    xp = np.pad(x, ((1, 1), (1, 1)), mode='constant')
    out = np.zeros((NF, IMG, IMG), np.float32)
    # cross-correlation, matching blocks.py
    for dy in range(3):
        for dx in range(3):
            out += FILT[:, dy, dx, None, None] * xp[dy:dy + IMG, dx:dx + IMG][None, :, :]
    out = np.maximum(out, 0.0)
    out = out.reshape(NF, CROP, POOL, CROP, POOL).mean(axis=(2, 4))
    mu = out.mean(axis=(1, 2), keepdims=True)
    out = (out - mu) / np.sqrt(((out - mu) ** 2).mean(axis=(1, 2), keepdims=True) + EPS)
    v = out.reshape(-1)
    v = v / np.sqrt(np.mean(v * v) + EPS)
    return v.astype(np.float32)


def variant_params(cls, j):
    # deterministic low-discrepancy-ish jitter, not selected using data
    if j == 0:
        return 0.0, 0.0, 1.0, 0.0, 0.0
    a = (1103515245 * (j + 17 * cls) + 12345) & 0x7fffffff
    b = (1103515245 * (a + 97) + 12345) & 0x7fffffff
    c = (1103515245 * (b + 193) + 12345) & 0x7fffffff
    d = (1103515245 * (c + 389) + 12345) & 0x7fffffff
    e = (1103515245 * (d + 769) + 12345) & 0x7fffffff
    u = [a / 0x7fffffff, b / 0x7fffffff, c / 0x7fffffff, d / 0x7fffffff, e / 0x7fffffff]
    # Horizontal/down signs tolerate a little more vertical/angle variation.
    horiz = cls in (6, 7, 14, 15)
    dx = (u[0] - 0.5) * (0.28 if not horiz else 0.34)
    dy = (u[1] - 0.5) * (0.24 if not horiz else 0.30)
    sc = 0.84 + 0.30 * u[2]
    rot = (u[3] - 0.5) * (0.28 if not horiz else 0.36)
    shp = u[4] - 0.5
    return dx, dy, sc, rot, shp


def make_template_weights():
    raw_size = IMG * IMG
    edge_size = NF * CROP * CROP
    Wr = np.zeros((NCLASS * TPER, raw_size), np.float32)
    We = np.zeros((NCLASS * TPER, edge_size), np.float32)
    b = np.zeros((NCLASS * TPER,), np.float32)
    row = 0
    for cls in range(NCLASS):
        for j in range(VARIANTS):
            dx, dy, sc, rot, shp = variant_params(cls, j)
            for pol in range(POLARITIES):
                for mir in range(MIRRORS):
                    img = make_image(cls, dx, dy, sc, rot, shp, pol, mir if cls in MIRROR_CLASSES else 0)
                    Wr[row] = raw_features(img)
                    We[row] = features(img)
                    row += 1
    Wr *= (1.0 / math.sqrt(float(raw_size)))
    We *= (1.0 / math.sqrt(float(edge_size)))
    return Wr, We, b


def network():
    return {"layers": [
        {"name": "inorm", "op": "norm", "inputs": ["image"]},
        {"name": "feat", "op": "conv", "inputs": ["inorm"], "out": int(NF), "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "frelu", "op": "relu", "inputs": ["feat"]},
        {"name": "fpool", "op": "avgpool", "inputs": ["frelu"], "size": [2, 2], "stride": [2, 2]},
        {"name": "fnorm", "op": "norm", "inputs": ["fpool"]},
        {"name": "tmpl_raw", "op": "dense", "inputs": ["inorm"], "out": int(NCLASS * TPER)},
        {"name": "tmpl_edge", "op": "dense", "inputs": ["fnorm"], "out": int(NCLASS * TPER)},
        {"name": "tmpl", "op": "add", "inputs": ["tmpl_raw", "tmpl_edge"]},
        {"name": "byclass", "op": "reshape", "inputs": ["tmpl"], "shape": [int(NCLASS), int(TPER), 1]},
        {"name": "classmax", "op": "maxpool", "inputs": ["byclass"], "size": [int(TPER), 1], "stride": [int(TPER), 1]},
        {"name": "flat", "op": "reshape", "inputs": ["classmax"], "shape": [int(NCLASS)]},
        {"name": "score", "op": "dense", "inputs": ["flat"], "out": int(NCLASS)}
    ]}


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = network()
    Wr, We, btmpl = make_template_weights()
    Wfeat = FILT[:, None, :, :].astype(np.float32)
    bfeat = np.zeros((NF,), np.float32)
    Wscore = np.eye(NCLASS, dtype=np.float32)
    bscore = np.zeros((NCLASS,), np.float32)
    weights = {
        'feat.W': Wfeat, 'feat.b': bfeat,
        'tmpl_raw.W': Wr, 'tmpl_raw.b': btmpl,
        'tmpl_edge.W': We, 'tmpl_edge.b': btmpl.copy(),
        'score.W': Wscore, 'score.b': bscore,
    }
    (out / 'network.json').write_text(json.dumps(net, separators=(',', ':')))
    np.savez(out / 'weights.npz', **weights)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/app/out')
