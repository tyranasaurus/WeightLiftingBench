#!/usr/bin/env python3
"""build.py OUT_DIR  -- writes OUT_DIR/network.json and OUT_DIR/weights.npz.

A digit classifier whose weights are written from knowledge, not fitted to data.

Idea
----
1. A fixed filter bank (formulas only): one Gaussian blur of the ink and six oriented
   ridge detectors (negative second derivative of an elongated Gaussian), followed by relu,
   a 3x3 stride-1 max pool (+-1 px local tolerance) and a 2x2 average pool.  This turns an image
   into a 7 x 13 x 13 map of "ink here" and "stroke of orientation theta here", smooth enough to
   tolerate small deformations.
2. Templates: digits are drawn as pen strokes along paths typed from general knowledge of how
   each digit is written (several styles each), with hand-chosen ranges of slant, rotation,
   width-to-height ratio, pen width and a smooth random warp, and are then normalised exactly as
   the task describes MNIST (fit in 20x20 keeping proportions, centre of mass at the centre).
   Each drawing is passed through the same filter bank and becomes ONE template, normalised by
   itself alone (zero mean, unit norm), so a dot product with it is a normalised correlation.
3. The template layer is a conv whose kernel covers the whole feature map, with padding 1 so each
   template is also tried at +-1 cell (+-2 pixels) shifts; max over shifts, then max over the
   templates of each class gives the class score (a nearest-template rule).

No training data is read; nothing is fitted.  Randomness uses a fixed seed.
"""
import json
import sys
import numpy as np

SEED = 20261119
N_PER_CLASS = 800           # templates per class
K_FEAT = 11                 # filter size of the fixed filter bank
N_ORIENT = 6                # ridge orientations (every 30 degrees)
SIGMA_BLUR = 1.5            # blur of the intensity channel (pixels)
SIGMA_U, SIGMA_V = 1.3, 2.2  # ridge detector: across-stroke / along-stroke Gaussian widths
GAIN_INT, GAIN_RIDGE = 1.5, 0.8  # hand-chosen relative weights of the two feature groups

# ============================================================================ curve helpers
def arc(cx, cy, rx, ry, a0, a1, n=None):
    if n is None:
        n = max(8, int(abs(a1 - a0) / 6))
    a = np.deg2rad(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def line(p0, p1, n=None):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if n is None:
        n = max(4, int(np.linalg.norm(p1 - p0) / 0.03))
    t = np.linspace(0, 1, n)[:, None]
    return p0 + t * (p1 - p0)


def spline(pts, n=None):
    """Catmull-Rom spline through the given points (open curve)."""
    P = np.asarray(pts, float)
    if len(P) == 2:
        return line(P[0], P[1], n)
    if n is None:
        n = max(8, int(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)) / 0.03))
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    per = max(2, n // (len(P) - 1))
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t = np.linspace(0, 1, per, endpoint=(i == len(P) - 2))[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


def cat(*parts):
    return np.vstack(parts)


def U(rng, a, b):
    return float(rng.uniform(a, b))


# ============================================================================ digit stroke models
# Nominal box: x in [0,1] rightwards, y in [0,1] downwards.
# Arc angles: 0 = right, 90 = bottom, 180 = left, 270 = top (increasing = clockwise on screen).

def digit0(rng):
    style = rng.choice(2, p=[0.8, 0.2])  # oval, oval pinched at the top
    rx, ry = U(rng, 0.18, 0.42), U(rng, 0.40, 0.48)
    cx = U(rng, 0.47, 0.53)
    a0 = U(rng, 230, 310)
    pts = arc(cx, 0.5, rx, ry, a0, a0 + 360 + U(rng, 0, 50))
    if style == 1:
        f = 1 - U(rng, 0.2, 0.5) * np.clip((0.5 - pts[:, 1]) / 0.5, 0, 1) ** 2
        pts = np.stack([cx + (pts[:, 0] - cx) * f, pts[:, 1]], 1)
    return [pts]


def digit1(rng):
    style = rng.choice(3, p=[0.68, 0.22, 0.10])  # plain, with flag, with flag and base
    bow = U(rng, -0.05, 0.05)
    stem = spline([(0.5, 0.02), (0.5 + bow, 0.5), (0.5, 0.98)])
    strokes = []
    if style >= 1:
        f = U(rng, 0.10, 0.26)
        strokes.append(line((0.5 - f, 0.02 + f * U(rng, 0.7, 1.4)), (0.5, 0.02)))
    strokes.append(stem)
    if style == 2:
        b = U(rng, 0.12, 0.22)
        strokes.append(line((0.5 - b, 0.98), (0.5 + b, 0.98)))
    return strokes


def digit2(rng):
    style = rng.choice(4, p=[0.4, 0.22, 0.23, 0.15])  # plain, cursive loop, S-shaped diagonal, Z-like
    cx, cy = U(rng, 0.42, 0.58), U(rng, 0.17, 0.3)
    rx, ry = U(rng, 0.18, 0.36), cy - 0.03
    xb, yb = U(rng, 0.0, 0.24), U(rng, 0.86, 0.98)
    xe, ye = U(rng, 0.6, 1.0), yb + U(rng, -0.08, 0.03)
    if style == 3:
        top = spline([(U(rng, 0.1, 0.25), U(rng, 0.1, 0.22)), (cx, U(rng, 0.02, 0.08)), (U(rng, 0.78, 0.95), U(rng, 0.12, 0.3))])
        return [cat(top, line(top[-1], (xb, yb)), line((xb, yb), (xe, ye)))]
    a0, a1 = U(rng, 150, 240), U(rng, 10, 80)
    head = arc(cx, cy, rx, ry, a0, 360 + a1)
    end = head[-1]
    if style == 1:
        r = U(rng, 0.06, 0.12)
        lx, ly = xb + r + 0.02, yb - r - 0.02
        body = spline([end, (0.5 * (end[0] + lx + r), 0.5 * (end[1] + ly - r)), (lx + 0.9 * r, ly + 0.2 * r), (lx, ly + r),
                       (lx - r, ly), (lx, ly - r), (lx + r, ly + 0.1 * r), (lx + 2 * r, yb), (xe, ye)])
        return [cat(head, body)]
    d = np.array([xb, yb]) - end
    nrm = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)
    if style == 2:
        b = U(rng, 0.04, 0.11) * (1 if rng.random() < 0.5 else -1)
        diag = spline([end, end + 0.3 * d + b * nrm, end + 0.7 * d - b * nrm, (xb, yb)])
    else:
        diag = spline([end, end + 0.5 * d + U(rng, -0.04, 0.07) * nrm, (xb, yb)])
    if rng.random() < 0.5:  # rounded bottom-left corner
        return [cat(head, diag, spline([(xb, yb), (xb + 0.08, yb + 0.01), (xe, ye)]))]
    return [cat(head, diag, line((xb, yb), (xe, ye)))]


def digit3(rng):
    style = rng.choice(2, p=[0.78, 0.22])  # round top, flat top
    xj, yj = U(rng, 0.3, 0.56), U(rng, 0.42, 0.54)   # middle junction
    cy1 = yj / 2 + 0.015
    rx1, ry1 = U(rng, 0.2, 0.34), cy1 - 0.03
    cy2 = (yj + 0.97) / 2
    rx2, ry2 = U(rng, 0.24, 0.36), 0.97 - cy2
    cx1, cx2 = U(rng, 0.46, 0.56), U(rng, 0.44, 0.56)
    bot = arc(cx2, cy2, rx2, ry2, U(rng, 225, 265), 360 + U(rng, 125, 175))
    if style == 0:
        top = arc(cx1, cy1, rx1, ry1, U(rng, 180, 235), 360 + U(rng, 90, 135))
    else:
        top = spline([(U(rng, 0.1, 0.28), U(rng, 0.03, 0.12)), (U(rng, 0.72, 0.9), U(rng, 0.02, 0.08))])
    return [cat(top, line(top[-1], (xj, yj)), line((xj, yj), bot[0]), bot)]


def digit4(rng):
    style = rng.choice(3, p=[0.5, 0.35, 0.15])  # open, closed, open with a curved left stroke
    yc = U(rng, 0.48, 0.72)
    xs = U(rng, 0.55, 0.8)
    xr = U(rng, 0.78, 1.0)
    x2 = U(rng, 0.0, 0.18)
    stem = line((xs + U(rng, -0.04, 0.04), U(rng, 0.0, 0.25)), (xs + U(rng, -0.12, 0.06), 0.98))
    yr = yc + U(rng, -0.03, 0.03)
    if style == 0:
        x1 = U(rng, 0.15, 0.42)
        left = cat(line((x1, U(rng, 0.02, 0.1)), (x2, yc)), line((x2, yc), (xr, yr)))
    elif style == 1:
        top = (stem[0][0] + U(rng, -0.04, 0.03), stem[0][1] + U(rng, 0, 0.05))
        left = cat(line(top, (x2, yc)), line((x2, yc), (xr, yr)))
    else:
        x1 = U(rng, 0.15, 0.4)
        left = spline([(x1, 0.03), (x1 - 0.5 * (x1 - x2), yc - 0.25), (x2 + 0.02, yc - 0.03), (x2 + 0.1, yc), (xr, yr)])
    return [left, stem]


def digit5(rng):
    style = rng.choice(3, p=[0.5, 0.3, 0.2])  # bar+vertical+flat-topped bowl, S-like body with separate bar, round bowl
    xl, xr = U(rng, 0.08, 0.3), U(rng, 0.6, 0.98)
    ybar = U(rng, 0.03, 0.1)
    bar = line((xr, ybar + U(rng, -0.02, 0.04)), (xl, ybar))
    yj = U(rng, 0.36, 0.55)                 # junction of the vertical and the bowl
    xj = xl + U(rng, -0.2, 0.04)            # the vertical often leans left on its way down
    vert = line((xl + U(rng, -0.02, 0.04), ybar), (xj, yj))
    if style == 2:
        cx, cy = U(rng, 0.42, 0.56), U(rng, 0.64, 0.74)
        rx, ry = U(rng, 0.26, 0.42), 0.97 - cy
        bowl = arc(cx, cy, rx, ry, U(rng, 215, 255), 360 + U(rng, 120, 170))
        return [cat(bar, vert, line(vert[-1], bowl[0]), bowl)]
    bx, by = U(rng, 0.78, 0.98), U(rng, 0.62, 0.78)   # rightmost point of the bowl
    xe, ye = U(rng, 0.02, 0.3), U(rng, 0.8, 0.95)     # end of the tail at the lower left
    bowl = spline([(xj, yj), (xj + U(rng, 0.1, 0.3), yj - U(rng, 0.0, 0.05)), (bx, by), (0.5 * (bx + xe) + 0.05, 0.97), (xe, ye)])
    if style == 0:
        return [cat(bar, vert, bowl)]
    body = spline([(xl + U(rng, 0.0, 0.06), ybar + 0.03), (0.5 * (xl + xj), 0.5 * (ybar + yj) + U(rng, -0.03, 0.03)), (xj, yj)])
    return [bar, cat(body, bowl)]


def digit6(rng):
    style = rng.choice(3, p=[0.55, 0.3, 0.15])  # curved descent, straight descent, top curls right
    xt = U(rng, 0.5, 0.9)
    cx, cy = U(rng, 0.4, 0.56), U(rng, 0.64, 0.76)
    rx, ry = U(rng, 0.2, 0.36), 0.97 - cy
    lx = cx - rx
    if style == 0:
        desc = spline([(xt, 0.03), (0.5 * (xt + lx + 0.15), U(rng, 0.1, 0.18)), (lx + U(rng, 0.06, 0.16), 0.36), (lx + 0.02, 0.56), (lx, cy)])
    elif style == 1:
        desc = spline([(xt, 0.03), (0.5 * (xt + lx), 0.5 * (0.03 + cy) + U(rng, -0.05, 0.05)), (lx, cy)])
    else:
        desc = spline([(xt - 0.15, U(rng, 0.08, 0.16)), (xt, 0.03), (0.5 * (xt + lx + 0.1), 0.15), (lx + 0.08, 0.38), (lx, cy)])
    loop = arc(cx, cy, rx, ry, 180, -180 - U(rng, 0, 40))
    return [cat(desc, loop)]


def digit7(rng):
    style = rng.choice(4, p=[0.45, 0.22, 0.13, 0.2])  # plain, crossbar, tick at the start, bar hooks down at the left
    xl, yl = U(rng, 0.02, 0.22), U(rng, 0.04, 0.16)
    xr, yr = U(rng, 0.74, 0.98), U(rng, 0.02, 0.12)
    xb = U(rng, 0.12, 0.6)
    end, start = np.array([xb, 0.97]), np.array([xr, yr])
    mid = 0.5 * (start + end)
    d = end - start
    nrm = np.array([d[1], -d[0]]) / np.linalg.norm(d)
    diag = spline([start, mid + U(rng, -0.08, 0.04) * nrm, end])
    if style == 3:
        bar = spline([(xl, yl + U(rng, 0.1, 0.22)), (xl + 0.04, yl + 0.02), (0.5 * (xl + xr), yl - 0.02 + U(rng, -0.03, 0.03)), (xr, yr)])
    else:
        bar = spline([(xl, yl), (0.5 * (xl + xr), 0.5 * (yl + yr) + U(rng, -0.04, 0.03)), (xr, yr)])
    strokes = []
    if style == 2:
        strokes.append(line((xl - 0.01, yl + U(rng, 0.08, 0.15)), (xl, yl)))
    strokes.append(cat(bar, diag))
    if style == 1:
        ym = U(rng, 0.42, 0.62)
        xm = start[0] + (end[0] - start[0]) * (ym - start[1]) / (end[1] - start[1])
        w = U(rng, 0.1, 0.22)
        strokes.append(line((xm - w, ym), (xm + w, ym)))
    return strokes


def digit8(rng):
    style = rng.choice(4, p=[0.35, 0.25, 0.2, 0.2])  # closed figure-eight, open at the top right, two stacked loops, S plus closing diagonal
    ht = U(rng, 0.34, 0.54)                      # height of the top loop
    at, ab = U(rng, 0.1, 0.28), U(rng, 0.14, 0.34)
    dx = U(rng, -0.1, 0.1)                       # bottom loop shifted sideways
    if style <= 1:
        t0 = U(rng, 0.1, 0.35) * np.pi if style == 1 else 0.0
        t1 = 2 * np.pi + (0.0 if style == 1 else U(rng, 0, 0.5))
        t = np.linspace(t0, t1, 160)
        A = 0.5 * (at + ab) - 0.5 * (ab - at) * np.cos(t)
        R = 0.5 - (0.5 - ht) * np.cos(t)
        x = 0.5 + A * np.sin(2 * t) + dx * 0.5 * (1 - np.cos(t))
        return [np.stack([x, ht - R * np.cos(t)], 1)]
    cy1, ry1 = ht / 2 + 0.015, ht / 2 - 0.025
    cy2, ry2 = ht + (1 - ht) / 2 - 0.01, (1 - ht) / 2 - 0.02
    if style == 2:
        top = arc(0.5, cy1, at, ry1, 90, 90 + 360 + U(rng, 0, 30))
        bot = arc(0.5 + dx, cy2, ab, ry2, 270, 270 + 360 + U(rng, 0, 30))
        return [top, bot]
    top = arc(0.5, cy1, at, ry1, 320, 90)              # from the top right, over the top, down to the crossing
    bot = arc(0.5 + dx, cy2, ab, ry2, 270, 270 + 320)  # clockwise round the bottom loop
    closing = spline([bot[-1], (0.5 + 0.5 * dx, ht), top[0]])
    return [cat(top, bot), closing]


def digit9(rng):
    style = rng.choice(3, p=[0.45, 0.35, 0.2])  # straight stem, hooked stem, stem from a loop left open
    cx = U(rng, 0.42, 0.56)
    rx, ry = U(rng, 0.16, 0.32), U(rng, 0.16, 0.3)
    cy = ry + 0.03
    a0 = U(rng, -15, 25)
    if style == 2:
        loop = arc(cx, cy, rx, ry, a0, a0 - 360 + U(rng, 20, 60))
        start = np.array([cx + rx * np.cos(np.deg2rad(a0)), cy + ry * np.sin(np.deg2rad(a0))])
    else:
        loop = arc(cx, cy, rx, ry, a0, a0 - 360 - U(rng, 0, 25))
        start = loop[-1]
    if style == 1:
        stem = spline([start, (start[0] + 0.03, 0.6), (start[0] - 0.04, 0.84), (start[0] - U(rng, 0.15, 0.38), 0.97)])
    else:
        xb = U(rng, 0.3, 0.8)
        stem = spline([start, (0.5 * (start[0] + xb) + U(rng, -0.04, 0.04), 0.5 * (start[1] + 0.98)), (xb, 0.98)])
    if style == 2:
        return [loop, stem]
    return [cat(loop, stem)]


DIGITS = [digit0, digit1, digit2, digit3, digit4, digit5, digit6, digit7, digit8, digit9]


# ============================================================================ global variation
def warp(strokes, rng, amp=0.03):
    """Smooth random displacement field (three low-frequency waves) applied to all points."""
    waves = []
    for _ in range(3):
        a = U(rng, 0, amp)
        th, ph = U(rng, 0, 2 * np.pi), U(rng, 0, 2 * np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        k = 2 * np.pi / U(rng, 0.5, 1.3)
        kv = np.array([np.cos(ph), np.sin(ph)]) * k
        waves.append((a, u, kv, U(rng, 0, 2 * np.pi)))
    out = []
    for s in strokes:
        d = np.zeros_like(s)
        for a, u, kv, phase in waves:
            d += a * np.sin(s @ kv + phase)[:, None] * u
        out.append(s + d)
    return out


def transform(strokes, shear, rot, aspect):
    """Width scaling, shear (positive leans right) and rotation (degrees) about the box centre."""
    c, s = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    out = []
    for p in strokes:
        q = (p - 0.5) * np.array([aspect, 1.0])
        q = np.stack([q[:, 0] - shear * q[:, 1], q[:, 1]], 1)
        q = np.stack([c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]], 1)
        out.append(q + 0.5)
    return out


def sample_params(rng, cls):
    """Slant, rotation, width-to-height and pen width, ranges chosen from general knowledge."""
    shear = float(np.clip(rng.normal(0.15, 0.25), -0.4, 0.8))
    if cls in (1, 8):
        shear = float(np.clip(rng.normal(0.2, 0.3), -0.45, 0.9))
    rot = U(rng, -8, 8)
    aspect = U(rng, 0.55, 1.2)
    width = U(rng, 1.6, 4.8)
    return shear, rot, aspect, width


# ============================================================================ rendering
HI = 3            # hi-res pixels per final pixel
BOX = 20 * HI     # nominal box side in hi-res pixels
CAN = 120         # hi-res canvas side
_YS, _XS = np.mgrid[0:CAN, 0:CAN]
_PIX = np.stack([_XS.ravel() + 0.5, _YS.ravel() + 0.5], 1).astype(np.float64)


def raster(strokes, width_px):
    """Draw the strokes with a round pen (width in final-scale pixels) on the hi-res canvas."""
    w = width_px * HI
    dmin = np.full(len(_PIX), 1e9)
    off = (CAN - BOX) / 2
    for s in strokes:
        P = s * BOX + off
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        P = P[np.concatenate([[True], seg > 1e-6])]
        if len(P) < 2:
            continue
        lo, hi = P.min(0) - w / 2 - 2, P.max(0) + w / 2 + 2
        idx = np.where((_PIX[:, 0] >= lo[0]) & (_PIX[:, 0] <= hi[0]) & (_PIX[:, 1] >= lo[1]) & (_PIX[:, 1] <= hi[1]))[0]
        if len(idx) == 0:
            continue
        q = _PIX[idx]
        A, AB = P[:-1], P[1:] - P[:-1]
        L2 = (AB ** 2).sum(1) + 1e-12
        best = np.full(len(q), 1e9)
        for j0 in range(0, len(A), 64):
            a, ab, l2 = A[j0:j0 + 64], AB[j0:j0 + 64], L2[j0:j0 + 64]
            t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(2) / l2[None], 0, 1)
            proj = a[None] + t[:, :, None] * ab[None]
            best = np.minimum(best, np.sqrt(((q[:, None, :] - proj) ** 2).sum(2)).min(1))
        dmin[idx] = np.minimum(dmin[idx], best)
    return np.clip(w / 2 - dmin.reshape(CAN, CAN) + 0.5, 0, 1)


def resample_matrix(n_in, n_out):
    """Area-averaging resampling matrix (n_out, n_in)."""
    M = np.zeros((n_out, n_in))
    f = n_in / n_out
    for i in range(n_out):
        a, b = i * f, (i + 1) * f
        for j in range(int(np.floor(a)), min(int(np.ceil(b)), n_in)):
            M[i, j] = max(0.0, min(b, j + 1) - max(a, j))
        M[i] /= f
    return M


def mnist_normalise(ink):
    """Crop to the ink, scale so the larger side is 20 px keeping proportions, centre of mass to the centre of 28x28."""
    m = ink > 0.5
    if not m.any():
        return np.zeros((28, 28))
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    crop = ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = resample_matrix(h, nh) @ crop @ resample_matrix(w, nw).T
    tot = small.sum()
    cy = (small.sum(1) * np.arange(nh)).sum() / tot
    cx = (small.sum(0) * np.arange(nw)).sum() / tot
    oy, ox = int(round(13.5 - cy)), int(round(13.5 - cx))
    oy, ox = min(max(oy, 0), 28 - nh), min(max(ox, 0), 28 - nw)
    out = np.zeros((28, 28))
    out[oy:oy + nh, ox:ox + nw] = small
    return np.clip(out, 0, 1)


def make_digit(cls, rng):
    strokes = DIGITS[cls](rng)
    strokes = warp(strokes, rng, amp=0.04)
    shear, rot, aspect, width = sample_params(rng, cls)
    strokes = transform(strokes, shear, rot, aspect)
    return mnist_normalise(raster(strokes, width))


# ============================================================================ fixed filter bank (formulas)
def filter_bank():
    """Returns W (1 + N_ORIENT, 1, K, K): a Gaussian blur and N_ORIENT oriented ridge detectors."""
    r = K_FEAT // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * SIGMA_BLUR ** 2))
    g /= g.sum()
    filters = [GAIN_INT * g]
    ridge = []
    for k in range(N_ORIENT):
        th = np.pi * k / N_ORIENT
        u = x * np.cos(th) + y * np.sin(th)      # across the stroke
        v = -x * np.sin(th) + y * np.cos(th)     # along the stroke
        env = np.exp(-u ** 2 / (2 * SIGMA_U ** 2) - v ** 2 / (2 * SIGMA_V ** 2))
        f = (1.0 / SIGMA_U ** 2 - u ** 2 / SIGMA_U ** 4) * env   # minus second derivative across
        f -= f.mean()                                             # no response to flat regions
        ridge.append(f)
    # scale so that an ideal straight bar 3 px wide (unit ink) gives response 1 at its centre
    bar = (np.abs(x) <= 1.5).astype(np.float64)                  # vertical bar, matches k = 0
    resp = (ridge[0] * bar).sum()
    for f in ridge:
        filters.append(GAIN_RIDGE * f / resp)
    return np.stack(filters)[:, None].astype(np.float64)


# ============================================================================ network blocks (same maths as the scorer)
def conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
                     for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def pool(x, size, stride, reduce):
    n, c, h, w = x.shape
    (sh, sw), (th, tw) = size, stride
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if reduce == "max" else win.mean(0)


FEAT_SIDE = 13  # 28 -> (3x3 max, stride 1) 26 -> (2x2 average) 13


def features(images, W):
    """images (N, 1, 28, 28) -> (N, 7, 13, 13): filter bank, relu, 3x3 max (local +-1 px tolerance), 2x2 average."""
    out = []
    for i in range(0, len(images), 250):
        f = conv(images[i:i + 250], W, np.zeros(W.shape[0]), 1, K_FEAT // 2)
        f = pool(np.maximum(f, 0), [3, 3], [1, 1], "max")
        out.append(pool(f, [2, 2], [2, 2], "avg"))
    return np.concatenate(out)


# ============================================================================ build
def main(out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    Wf = filter_bank()
    n_feat = Wf.shape[0]

    T = 10 * N_PER_CLASS
    S = FEAT_SIDE
    templates = np.zeros((T, n_feat * S * S), dtype=np.float64)
    t = 0
    for cls in range(10):
        imgs = np.stack([make_digit(cls, rng) for _ in range(N_PER_CLASS)])[:, None]
        F = features(imgs, Wf).reshape(N_PER_CLASS, -1)
        F = F - F.mean(axis=1, keepdims=True)                 # each template: zero mean ...
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)  # ... and unit norm, by itself alone
        templates[t:t + N_PER_CLASS] = F
        t += N_PER_CLASS
        print(f"class {cls}: {N_PER_CLASS} templates", flush=True)

    layers = [
        {"name": "feat", "op": "conv", "inputs": ["image"], "out": n_feat, "kernel": [K_FEAT, K_FEAT], "stride": 1, "padding": K_FEAT // 2},
        {"name": "feat_relu", "op": "relu", "inputs": ["feat"]},
        {"name": "feat_local", "op": "maxpool", "inputs": ["feat_relu"], "size": [3, 3], "stride": [1, 1]},
        {"name": "feat_pool", "op": "avgpool", "inputs": ["feat_local"], "size": [2, 2], "stride": [2, 2]},
        {"name": "tmpl", "op": "conv", "inputs": ["feat_pool"], "out": T, "kernel": [S, S], "stride": 1, "padding": 1},
        {"name": "shift_max", "op": "maxpool", "inputs": ["tmpl"], "size": [3, 3], "stride": [3, 3]},
        {"name": "by_class", "op": "reshape", "inputs": ["shift_max"], "shape": [1, 10, N_PER_CLASS]},
        {"name": "class_max", "op": "maxpool", "inputs": ["by_class"], "size": [1, N_PER_CLASS], "stride": [1, N_PER_CLASS]},
        {"name": "scores", "op": "reshape", "inputs": ["class_max"], "shape": [10]},
    ]
    weights = {
        "feat.W": Wf.astype(np.float32),
        "feat.b": np.zeros(n_feat, np.float32),
        "tmpl.W": templates.reshape(T, n_feat, S, S).astype(np.float32),
        "tmpl.b": np.zeros(T, np.float32),
    }
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)
    print("total weights:", sum(v.size for v in weights.values()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
