#!/usr/bin/env python3
"""Builds a hand-designed digit classifier.  No file is read, nothing is trained or fitted.

Architecture
  image -> 8 directional derivative filters (Sobel, written from the formula) -> relu
        -> 1x1 conv sharpening the direction tuning (g_k - |g_perp|) -> relu
        -> Gaussian pooling to a 7x7 grid (formula) -> piecewise-linear square-root compression
        -> length normalisation of the feature vector (norm layer applied to [f, -f])
        -> four overlapping quadrants of the grid, each matched against the same quadrant of every template
           (dense layer; score = f.w - |w|^2/2, i.e. minus half the squared distance up to a constant)
        -> the quadrant scores are added into halves and the whole image
        -> for the whole image, each half and each quadrant: the best template of each class (maxpool)
        -> the nine "best" scores of each class are added.

Templates
  Every template is one synthetic digit: a pen path typed from general knowledge of how the ten digits
  are written (control points -> spline -> stroke of some thickness), placed the way the task statement
  describes (20 pixel box, centre of mass at the centre of the frame).  Each drawing is pushed through the
  same feature formulas and becomes one row of the dense layers, normalised by its own length only.
  Nothing is averaged, fitted, selected or thresholded across drawings.
"""
import sys, json
from pathlib import Path
import numpy as np

SEED = 20261120
PER_CLASS = 15000          # templates per class
NDIR = 8                   # gradient directions
GRID = 7                   # pooled grid
POOL_K, POOL_S, POOL_P = 12, 4, 4
POOL_SIGMA = 1.8           # sqrt(2) * interval / pi  for a sampling interval of 4 pixels
BREAKS = [0.0, 0.01, 0.04, 0.09, 0.16, 0.25]   # knots of the square-root interpolation (squares of 0, .1, .2, ...)
NORM_EPS = 1e-5
HALF = [np.array([1, 1, 1, 0.5, 0, 0, 0.0]), np.array([0, 0, 0, 0.5, 1, 1, 1.0])]   # the two halves of the grid share the middle line

f32 = np.float32


# ----------------------------------------------------------------------------------------------
# helpers for the pen-path models (everything is vectorised over a batch of B drawings)
# ----------------------------------------------------------------------------------------------
class R:
    """Random helper: every call returns an array of B numbers."""
    def __init__(self, rng, B):
        self.rng, self.B = rng, B
        self.slant = None          # a digit may choose its own slant per drawing

    def u(self, lo, hi):
        return self.rng.uniform(lo, hi, self.B)

    def tri(self, lo, hi):
        return lo + (hi - lo) * 0.5 * (self.rng.uniform(0.0, 1.0, self.B) + self.rng.uniform(0.0, 1.0, self.B))

    def n(self, sd):
        return sd * self.rng.standard_normal(self.B)

    def coin(self, p):
        return self.rng.uniform(0.0, 1.0, self.B) < p


def pts(*xy):
    """pts((x0,y0),(x1,y1),...) with each coordinate an array (B,) or a float -> (B, n, 2)."""
    B = max(np.size(c) for p in xy for c in p)
    out = np.empty((B, len(xy), 2))
    for i, (x, y) in enumerate(xy):
        out[:, i, 0] = x
        out[:, i, 1] = y
    return out


def lerp(a, b, t):
    return a + (b - a) * t


def ell(cx, cy, a, b, deg):
    """Point of an ellipse; with y pointing down, 0 = right, 90 = bottom, 180 = left, -90 = top."""
    r = np.deg2rad(deg)
    return (cx + a * np.cos(r), cy + b * np.sin(r))


def spline(P, nsub):
    """Centripetal Catmull-Rom spline through control points P (B, n, 2) -> (B, (n-1)*nsub+1, 2)."""
    Pp = np.concatenate([2 * P[:, :1] - P[:, 1:2], P, 2 * P[:, -1:] - P[:, -2:-1]], 1)
    P0, P1, P2, P3 = Pp[:, :-3], Pp[:, 1:-2], Pp[:, 2:-1], Pp[:, 3:]

    def kd(a, b):
        return (np.sqrt(((a - b) ** 2).sum(-1)) + 1e-6) ** 0.5

    t0 = np.zeros(P0.shape[:2])
    t1 = t0 + kd(P0, P1)
    t2 = t1 + kd(P1, P2)
    t3 = t2 + kd(P2, P3)
    u = np.linspace(0.0, 1.0, nsub, endpoint=False)
    t = t1[..., None] + (t2 - t1)[..., None] * u                      # (B, seg, nsub)
    e = lambda a: a[..., None, None]
    q = lambda a: a[:, :, None, :]
    tt = t[..., None]
    A1 = (e(t1) - tt) / e(t1 - t0) * q(P0) + (tt - e(t0)) / e(t1 - t0) * q(P1)
    A2 = (e(t2) - tt) / e(t2 - t1) * q(P1) + (tt - e(t1)) / e(t2 - t1) * q(P2)
    A3 = (e(t3) - tt) / e(t3 - t2) * q(P2) + (tt - e(t2)) / e(t3 - t2) * q(P3)
    B1 = (e(t2) - tt) / e(t2 - t0) * A1 + (tt - e(t0)) / e(t2 - t0) * A2
    B2 = (e(t3) - tt) / e(t3 - t1) * A2 + (tt - e(t1)) / e(t3 - t1) * A3
    C = (e(t2) - tt) / e(t2 - t1) * B1 + (tt - e(t1)) / e(t2 - t1) * B2
    C = C.reshape(C.shape[0], -1, 2)
    return np.concatenate([C, P[:, -1:]], 1)


def stroke(P, corners=(), nsub=10, jit=None, rng=None):
    """One pen stroke through control points P (B, n, 2); the tangent breaks at the indices in `corners`.
    Returns (B, m, 3): x, y and a flag that is 1 where ink runs on to the next point."""
    if jit is not None:
        P = P + jit * rng.standard_normal(P.shape)
    cuts = [0] + sorted(corners) + [P.shape[1] - 1]
    parts = [spline(P[:, a:b + 1], nsub) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    Q = np.concatenate(parts, 1)
    flag = np.ones(Q.shape[:2])
    flag[:, -1] = 0.0
    return np.concatenate([Q, flag[..., None]], -1)


def alt(pick, A, Bq):
    """Per drawing: stroke Bq where pick is true, else stroke A (the shorter one is padded with its last point)."""
    m = max(A.shape[1], Bq.shape[1])

    def pad(Q):
        if Q.shape[1] == m:
            return Q
        tail = np.repeat(Q[:, -1:], m - Q.shape[1], 1)
        return np.concatenate([Q, tail], 1)
    return np.where(pick[:, None, None], pad(Bq), pad(A))


# ----------------------------------------------------------------------------------------------
# the ten digits as pen paths.  Coordinates: x to the right, y downwards, the digit about 1 unit tall
# and roughly 0.6 - 0.7 wide.  All numbers below are typed from general knowledge of handwriting.
# Each function returns a list of strokes (B, m, 3).
# ----------------------------------------------------------------------------------------------
J = 0.022   # independent wobble of every control point (fraction of the digit height)


def digit0(r):
    # one closed oval, written anticlockwise from near the top; the ends may overlap or leave a gap
    n = 14
    t = np.linspace(0.0, 1.0, n)[None, :]
    th0 = np.deg2rad(-90.0 + r.u(-35, 45))[:, None]
    over = np.deg2rad(r.u(-22, 45))[:, None]
    th = th0 - t * (2 * np.pi + over)
    w = r.u(0.50, 0.92)[:, None]
    spiral = r.u(-0.28, 0.28)[:, None]
    egg = r.u(-0.22, 0.22)[:, None]
    p = r.u(1.7, 2.8)[:, None]
    c, s = np.cos(th), np.sin(th)
    sx = np.sign(c) * np.abs(c) ** (2.0 / p)
    sy = np.sign(s) * np.abs(s) ** (2.0 / p)
    rad = 1.0 + spiral * (t - 0.5)
    x = 0.5 * w * rad * sx * (1.0 + egg * s)
    y = 0.5 * rad * sy
    return [stroke(np.stack([x, y], -1), nsub=8, jit=J * 0.8, rng=r.rng)]


def digit1(r):
    # a stem; sometimes a short flag up to the top of the stem; sometimes also a base bar
    B = r.B
    style = r.rng.uniform(0, 1, B)
    flag = style > 0.66
    base = style > 0.87
    wob = np.where(flag, 0.012, 0.025)
    stem = pts((0.0, 0.0), (wob * r.n(1.0), 0.33), (wob * r.n(1.0), 0.67), (0.6 * wob * r.n(1.0), 1.0))
    out = [stroke(stem, nsub=16)]
    flen = r.u(0.15, 0.40)                      # the flag is short and clearly falls away from the top
    fang = np.deg2rad(r.u(40, 70))
    fa = np.where(flag, flen * np.cos(fang), 0.0)
    fb = np.where(flag, flen * np.sin(fang), 0.0)
    r.slant = np.where(base, r.tri(-0.12, 0.25), np.where(flag, r.tri(-0.15, 0.35), r.tri(-0.35, 0.95)))
    fl = pts((-fa, fb), (-0.5 * fa + r.n(0.01) * flag, 0.5 * fb), (0.0, 0.0))
    out.append(stroke(fl, nsub=8))
    bl = np.where(base, r.u(0.14, 0.36), 0.0)
    br = np.where(base, r.u(0.14, 0.36), 0.0)
    bx = stem[:, 3, 0]
    ba = pts((bx - bl, 1.0 + r.n(0.015) * base), (bx + br, 1.0 + r.n(0.015) * base))
    out.append(stroke(ba, nsub=14))
    return out


def digit2(r):
    # arc over the top, diagonal down to the bottom left, then the base to the right;
    # at the bottom left either a sharp corner or a small loop
    loop = r.coin(0.45)
    curl = r.coin(0.3)
    xs, ys = r.u(0.0, 0.15), r.u(0.16, 0.36)
    S0 = (xs + np.where(curl, r.u(0.03, 0.10), 0.0), ys + np.where(curl, r.u(0.04, 0.13), 0.0))
    S = (xs, ys)
    T = (r.u(0.24, 0.42), r.u(0.0, 0.03))
    xr, yr = r.u(0.56, 0.72), r.u(0.16, 0.32)
    Rr = (xr, yr)
    cx, cy = r.u(0.0, 0.13), r.u(0.88, 1.0)
    bulge = r.u(-0.05, 0.10)
    M1 = (lerp(xr, cx, 0.3) + bulge, lerp(yr, cy, 0.33))
    M2 = (lerp(xr, cx, 0.68) + 0.4 * bulge, lerp(yr, cy, 0.68))
    ex, ey = r.u(0.62, 0.88), r.u(0.84, 1.0)
    wav = r.u(-0.05, 0.07)
    Bm = (lerp(cx, ex, 0.5), lerp(cy, ey, 0.5) + wav)
    a = stroke(pts(S0, S, T, Rr, M1, M2, (cx, cy), Bm, (ex, ey)), corners=(6,), nsub=9, jit=J, rng=r.rng)
    L2 = (cx - r.u(0.03, 0.09), cy - r.u(0.05, 0.11))
    L3 = (cx + r.u(-0.01, 0.07), cy - r.u(0.13, 0.22))
    L4 = (cx + r.u(0.15, 0.26), cy - r.u(0.03, 0.10))
    B2 = (r.u(0.40, 0.52), np.minimum(cy + r.u(-0.02, 0.06), 1.0))
    b = stroke(pts(S0, S, T, Rr, M1, M2, (cx, cy), L2, L3, L4, B2, (ex, ey)), nsub=7, jit=J, rng=r.rng)
    return [alt(loop, a, b)]


def digit3(r):
    # two bowls open to the left, meeting in a cusp; the top either round or a flat bar with a corner
    flat = r.coin(0.3)
    S = (r.u(0.0, 0.13), np.where(flat, r.u(0.0, 0.05), r.u(0.07, 0.2)))
    T = (r.u(0.24, 0.40), np.where(flat, r.u(-0.02, 0.03), r.u(0.0, 0.02)))
    xr1 = r.u(0.54, 0.70)
    R1 = (xr1, np.where(flat, r.u(-0.02, 0.05), r.u(0.14, 0.26)))
    R1b = (xr1 - r.u(0.0, 0.08), np.where(flat, 0.2, 0.33) + r.n(0.02))
    mx, my = r.u(0.20, 0.40), r.u(0.38, 0.50)
    N1 = (mx + r.u(0.14, 0.24), my + r.u(0.0, 0.05))
    xr2 = r.u(0.58, 0.76)
    R2 = (xr2, r.u(0.64, 0.78))
    R3 = (xr2 - r.u(0.06, 0.14), r.u(0.88, 0.95))
    Bt = (r.u(0.26, 0.42), r.u(0.97, 1.0))
    E = (r.u(-0.06, 0.10), r.u(0.78, 0.93))
    P = pts(S, T, R1, R1b, (mx, my), N1, R2, R3, Bt, E)
    a = stroke(P, corners=(4,), nsub=9, jit=J, rng=r.rng)
    b = stroke(P, corners=(2, 4), nsub=9, jit=J, rng=r.rng)
    return [alt(flat, a, b)]


def digit4(r):
    # stroke 1: down on the left then the crossbar to the right; stroke 2: the long vertical.
    # open top (the usual handwriting) or closed top (stroke 1 starts at the top of the vertical)
    closed = r.coin(0.25)
    roundc = r.coin(0.3) & ~closed
    yb = r.u(0.45, 0.68)
    xd = r.u(0.40, 0.58)
    ytop = np.where(closed, r.u(-0.04, 0.10), r.u(-0.04, 0.25))
    Ao = (r.u(0.04, 0.24), r.u(0.0, 0.12))
    Bc = (r.u(0.0, 0.11), yb)
    A = (np.where(closed, xd + r.n(0.02), Ao[0]), np.where(closed, ytop + r.u(0.0, 0.05), Ao[1]))
    bow = r.u(-0.03, 0.07)
    Am = (lerp(A[0], Bc[0], 0.55) - bow * (~closed), lerp(A[1], Bc[1], 0.55))
    C = (np.maximum(r.u(0.62, 0.84), xd + 0.06), yb + r.u(-0.06, 0.05))
    P1 = pts(A, Am, Bc, C)
    sharp = stroke(P1, corners=(2,), nsub=12, jit=J, rng=r.rng)
    smooth = stroke(P1, nsub=12, jit=J, rng=r.rng)
    D = (xd, ytop)
    E = (xd + r.u(-0.08, 0.05), 1.0)
    Dm = (lerp(D[0], E[0], 0.5) + r.n(0.015), lerp(D[1], E[1], 0.5))
    v = stroke(pts(D, Dm, E), nsub=18, jit=J * 0.7, rng=r.rng)
    return [alt(roundc, sharp, smooth), v]


def digit5(r):
    # down on the left, a bowl open to the left below, and the bar on top (attached or written separately)
    xv = r.u(0.08, 0.26)
    V0 = (xv, r.u(0.0, 0.06))
    yv = r.u(0.34, 0.50)
    V1 = (xv + r.u(-0.10, 0.03), yv)
    B1 = (r.u(0.36, 0.50), yv - r.u(0.02, 0.10))
    xr = r.u(0.56, 0.74)
    B2 = (xr, r.u(0.60, 0.74))
    B2b = (xr - r.u(0.05, 0.14), r.u(0.87, 0.95))
    B3 = (r.u(0.26, 0.42), r.u(0.97, 1.0))
    B4 = (r.u(-0.06, 0.10), r.u(0.80, 0.95))
    P5 = pts(V0, V1, B1, B2, B2b, B3, B4)
    body = alt(r.coin(0.25), stroke(P5, corners=(1,), nsub=10, jit=J, rng=r.rng), stroke(P5, nsub=10, jit=J, rng=r.rng))
    attached = r.coin(0.6)
    dx = np.where(attached, 0.0, r.u(-0.06, 0.08))
    dy = np.where(attached, 0.0, r.u(-0.06, 0.05))
    bs = (V0[0] + dx, V0[1] + dy)
    be = (r.u(0.56, 0.82), V0[1] + dy + r.u(-0.10, 0.03))
    bm = (lerp(bs[0], be[0], 0.5), lerp(bs[1], be[1], 0.5) + r.n(0.015))
    bar = stroke(pts(bs, bm, be), nsub=12, jit=J * 0.6, rng=r.rng)
    return [body, bar]


def digit6(r):
    # a stem from the top right curving down the left side into a closed loop at the bottom
    a, b = r.u(0.19, 0.33), r.u(0.16, 0.28)
    cx, cy = a + r.u(0.0, 0.05), 1.0 - b
    hook = r.coin(0.3)
    sx = r.u(0.38, 0.74)
    S = (sx, r.u(0.0, 0.03))
    S0 = (sx + np.where(hook, r.u(0.05, 0.16), 0.0), S[1] + np.where(hook, r.u(0.02, 0.10), 0.0))
    xl = cx - a
    bend = r.u(0.40, 0.95)
    P1 = (lerp(sx, xl, bend) + r.n(0.02), lerp(0.0, cy, 0.42))
    sweep = r.u(305, 390)
    shrink = r.u(0.0, 0.22)
    cps = [S0, S, P1]
    for i in range(9):
        f = i / 8.0
        k = 1.0 - shrink * f
        cps.append(ell(cx, cy, a * k, b * k, 180.0 - f * sweep))
    return [stroke(pts(*cps), nsub=8, jit=J * 0.8, rng=r.rng)]


def digit7(r):
    # a bar along the top, then a long diagonal down; sometimes a tick at the start of the bar,
    # sometimes a short bar across the middle of the diagonal
    serif = r.coin(0.25)
    cross = r.coin(0.16)
    roundc = r.coin(0.2)
    A = (r.u(0.0, 0.12), r.u(0.0, 0.10))
    A0 = (A[0] + np.where(serif, r.u(-0.02, 0.04), 0.0), A[1] + np.where(serif, r.u(0.08, 0.20), 0.0))
    Bx, By = r.u(0.58, 0.76), r.u(0.0, 0.08)
    Am = (lerp(A[0], Bx, 0.5), lerp(A[1], By, 0.5) + r.u(-0.04, 0.04))
    Cx = r.u(0.14, 0.52)
    bul = r.u(-0.04, 0.09)
    D1 = (lerp(Bx, Cx, 0.35) + bul, 0.35 + 0.65 * By)
    D2 = (lerp(Bx, Cx, 0.7) + 0.6 * bul, 0.7 + 0.3 * By)
    P = pts(A0, A, Am, (Bx, By), D1, D2, (Cx, 1.0))
    sharp = stroke(P, corners=(1, 3), nsub=10, jit=J, rng=r.rng)
    smooth = stroke(P, corners=(1,), nsub=10, jit=J, rng=r.rng)
    yc = r.u(0.45, 0.62)
    xc = lerp(Bx, Cx, yc) + 0.5 * bul
    hl = np.where(cross, r.u(0.13, 0.25), 0.0)
    hr = np.where(cross, r.u(0.13, 0.25), 0.0)
    sl = r.n(0.03) * cross
    bar = stroke(pts((xc - hl, yc + sl), (xc + hr, yc - sl)), nsub=10)
    return [alt(roundc, sharp, smooth), bar]


def digit8(r):
    # one stroke: from the upper right over the top, down through the waist, round the lower loop,
    # back up through the waist and round to the start
    b1 = r.u(0.16, 0.30)
    b2 = 0.5 - b1
    a1 = r.u(0.15, 0.26)
    a2 = r.u(0.18, 0.31)
    cxm = np.maximum(a1, a2) + 0.02
    c1 = cxm + r.u(-0.03, 0.06)
    c2 = cxm + r.u(-0.05, 0.03)
    X = (0.5 * (c1 + c2) + r.n(0.02), 2 * b1 + r.n(0.015))
    cy1, cy2 = b1, 2 * b1 + b2
    phi = r.u(-80, -15)
    S = ell(c1, cy1, a1, b1, phi)
    gap = r.u(-25, 30)
    Eend = ell(c1, cy1, a1 * r.u(0.85, 1.1), b1 * r.u(0.85, 1.1), phi + gap)
    cps = [S,
           ell(c1, cy1, a1, b1, -105), ell(c1, cy1, a1, b1, -150), ell(c1, cy1, a1, b1, 165),
           ell(c1, cy1, a1 * 0.8, b1 * 0.9, 125),
           X,
           ell(c2, cy2, a2 * 0.85, b2 * 0.9, -40), ell(c2, cy2, a2, b2, 5), ell(c2, cy2, a2, b2, 50),
           ell(c2, cy2, a2, b2, 95), ell(c2, cy2, a2, b2, 140), ell(c2, cy2, a2, b2, 185),
           ell(c2, cy2, a2 * 0.85, b2 * 0.9, 225),
           X,
           ell(c1, cy1, a1 * 0.8, b1 * 0.9, 55), ell(c1, cy1, a1, b1, 10),
           Eend]
    return [stroke(pts(*cps), nsub=7, jit=J * 0.7, rng=r.rng)]


def digit9(r):
    # a loop at the top written anticlockwise from the upper right, closed on the right side,
    # then the tail straight down or hooked to the left at the bottom
    a, b = r.u(0.17, 0.31), r.u(0.16, 0.28)
    cx, cy = a + 0.02, b
    hookd = r.coin(0.4)
    cps = [ell(cx, cy, a, b, np.where(hookd, r.u(-45, 0), r.u(-75, 0)))]
    for ang in (-100, -145, 170, 125, 85, 45):
        cps.append(ell(cx, cy, a, b, ang))
    jx = cx + a + r.u(-0.03, 0.05)
    jy = b - r.u(0.0, 0.16)
    cps.append((jx, jy))                                    # index 7: the pen turns back down here
    ex = jx + r.u(-0.22, 0.05)
    T1s = (lerp(jx, ex, 0.3) + r.u(0.0, 0.04), lerp(jy, 1.0, 0.45))
    hk1 = (jx - r.u(0.05, 0.18), r.u(0.93, 1.0))
    hk2 = (jx - r.u(0.20, 0.42), r.u(0.84, 0.95))
    T1 = (np.where(hookd, jx - r.u(-0.02, 0.04), T1s[0]), np.where(hookd, r.u(0.6, 0.75), T1s[1]))
    T2 = (np.where(hookd, hk1[0], lerp(jx, ex, 0.75)), np.where(hookd, hk1[1], lerp(jy, 1.0, 0.8)))
    T3 = (np.where(hookd, hk2[0], ex), np.where(hookd, hk2[1], 1.0))
    cps += [T1, T2, T3]
    return [stroke(pts(*cps), corners=(7,), nsub=9, jit=J * 0.8, rng=r.rng)]


DIGITS = [digit0, digit1, digit2, digit3, digit4, digit5, digit6, digit7, digit8, digit9]
# slant range (shear, x per unit of height; positive leans to the right), typed per class
SLANT = {0: (-0.20, 0.45), 1: (-0.35, 0.95),   # digit 1 sets its own (see digit1)
         2: (-0.20, 0.50), 3: (-0.20, 0.50), 4: (-0.20, 0.50),
         5: (-0.20, 0.50), 6: (-0.20, 0.50), 7: (-0.25, 0.45), 8: (-0.20, 0.50), 9: (-0.20, 0.50)}


def draw_paths(digit, B, rng):
    """Pen paths of B random drawings of one digit, in pixel units about their own middle:
    Q (B, m, 2), ink flags (B, m), pen widths (B,), and the random helper."""
    r = R(rng, B)
    P = np.concatenate(DIGITS[digit](r), 1)
    x, y, flag = P[..., 0].copy(), P[..., 1].copy(), P[..., 2]
    x -= 0.5 * (x.max(1, keepdims=True) + x.min(1, keepdims=True))
    y -= 0.5 * (y.max(1, keepdims=True) + y.min(1, keepdims=True))
    col = lambda v: v[:, None]
    # global handwriting variations: smooth bending, width, slant, a little rotation
    ax, ay = col(r.n(0.025)), col(r.n(0.025))
    fx, fy = col(r.u(0.5, 1.3)), col(r.u(0.5, 1.3))
    px, py = col(r.u(0, 2 * np.pi)), col(r.u(0, 2 * np.pi))
    x2 = x + ax * np.sin(2 * np.pi * fx * y + px)
    y2 = y + ay * np.sin(2 * np.pi * fy * x + py)
    wide = col(np.exp(r.tri(np.log(0.75), np.log(1.20))))
    lo, hi = SLANT[digit]
    sh = col(r.tri(lo, hi) if r.slant is None else r.slant)
    rot = col(np.deg2rad(r.n(4.0)))
    x3 = x2 * wide - sh * y2
    y3 = y2
    x4 = x3 * np.cos(rot) - y3 * np.sin(rot)
    y4 = x3 * np.sin(rot) + y3 * np.cos(rot)
    # pen width in pixels of the final image: mostly 1.5 - 3.6, sometimes bold
    bold = r.coin(0.15)
    w = np.where(bold, r.u(3.6, 5.2), r.u(1.5, 3.6))
    # fit the inked bounding box (path box grown by the pen width) into 20 pixels, keeping proportions
    ext = np.maximum(x4.max(1) - x4.min(1), y4.max(1) - y4.min(1))
    sc = (20.0 - w) / np.maximum(ext, 1e-6)
    Q = np.stack([x4 * sc[:, None], y4 * sc[:, None]], -1)
    return Q, flag, w, r


def render(Q, flag, w, soft, cx, cy):
    """Ink image of a pen path: full ink within w/2 of the path (exact distance to the line segments),
    falling off linearly over `soft` pixels."""
    g = np.arange(28, dtype=f32)
    ax = (Q[:, :-1, 0] + cx[:, None]).astype(f32)
    ay = (Q[:, :-1, 1] + cy[:, None]).astype(f32)
    fl = flag[:, :-1].astype(f32)
    dx = (Q[:, 1:, 0] - Q[:, :-1, 0]).astype(f32) * fl          # a jump between two strokes counts as a point
    dy = (Q[:, 1:, 1] - Q[:, :-1, 1]).astype(f32) * fl
    inv = 1.0 / (dx * dx + dy * dy + f32(1e-12))
    rx = g - ax[:, :, None]                                      # (B, m, 28) pixel minus segment start, x
    ry = g - ay[:, :, None]
    dot = (ry * dy[:, :, None])[:, :, :, None] + (rx * dx[:, :, None])[:, :, None, :]      # (B, m, 28, 28)
    t = np.clip(dot * inv[:, :, None, None], 0.0, 1.0)
    d2 = (ry * ry)[:, :, :, None] + (rx * rx)[:, :, None, :] - t * (2.0 * dot - t / inv[:, :, None, None])
    d = np.sqrt(np.maximum(d2.min(1), 0.0))
    return np.clip(0.5 + (w[:, None, None].astype(f32) * 0.5 - d) / soft[:, None, None].astype(f32), 0.0, 1.0)


def make_images(digit, B, rng):
    """B random drawings of one digit as 28 x 28 images with the centre of mass at the centre of the frame."""
    Q, flag, w, r = draw_paths(digit, B, rng)
    soft = r.u(0.9, 1.6)
    # the centre of the frame is 13.5 in pixel-index units; placing by whole pixels leaves up to half a
    # pixel either way, and the range below also allows for a "centre" counted as pixel 14
    tx, ty = 13.5 + r.u(-0.5, 1.0), 13.5 + r.u(-0.5, 1.0)
    seg = np.sqrt(((Q[:, 1:] - Q[:, :-1]) ** 2).sum(-1))
    wt = np.concatenate([seg, seg[:, -1:]], 1) * flag + 1e-9
    mx = (Q[..., 0] * wt).sum(1) / wt.sum(1)
    my = (Q[..., 1] * wt).sum(1) / wt.sum(1)
    img = render(Q, flag, w, soft, tx - mx, ty - my)
    # whole-pixel correction so that the centre of mass of the ink itself sits on the target
    g = np.arange(28, dtype=np.float64)
    tot = img.sum((1, 2), dtype=np.float64) + 1e-9
    comx = (img.sum(1, dtype=np.float64) * g).sum(1) / tot
    comy = (img.sum(2, dtype=np.float64) * g).sum(1) / tot
    sx = np.rint(tx - comx).astype(int)
    sy = np.rint(ty - comy).astype(int)
    for i in np.nonzero((sx != 0) | (sy != 0))[0]:
        img[i] = np.roll(img[i], (sy[i], sx[i]), (0, 1))
    return img


# ----------------------------------------------------------------------------------------------
# the feature formulas (the same numbers go into the network's first layers)
# ----------------------------------------------------------------------------------------------
def conv1_weights():
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
    sy = sx.T
    W = np.zeros((NDIR, 1, 3, 3))
    for k in range(NDIR):
        a = 2 * np.pi * k / NDIR
        W[k, 0] = np.cos(a) * sx + np.sin(a) * sy
    return W


def conv2_weights():
    # h_k = g_k - |g_perp| with |g_perp| = relu(g_{k+2}) + relu(g_{k-2}): each gradient is shared between the
    # two nearest of the eight directions (no response 45 degrees away or further)
    W = np.zeros((NDIR, NDIR, 1, 1))
    for k in range(NDIR):
        W[k, k, 0, 0] = 1.0
        W[k, (k + 2) % NDIR, 0, 0] = -1.0
        W[k, (k - 2) % NDIR, 0, 0] = -1.0
    return W


def gauss1d():
    c = (POOL_K - 1) / 2.0
    g = np.exp(-((np.arange(POOL_K) - c) ** 2) / (2 * POOL_SIGMA ** 2))
    return g / g.sum()


def sqrt_slopes():
    b = np.array(BREAKS)
    knots = np.concatenate([b, [(np.sqrt(b[-1]) + 0.1) ** 2]])
    slopes = np.diff(np.sqrt(knots)) / np.diff(knots)
    return np.concatenate([[slopes[0]], np.diff(slopes)])       # slope, then the change of slope at each knot


def quadrants():
    """For each quadrant: (row indices, column indices, weights (4, 4)); the weights of the four add up to 1 everywhere."""
    out = []
    for a in range(2):
        for b in range(2):
            rows, cols = np.nonzero(HALF[a])[0], np.nonzero(HALF[b])[0]
            out.append((rows, cols, np.outer(HALF[a][rows], HALF[b][cols])))
    return out


def features(img):
    """img (N, 28, 28) float32 -> (N, NDIR, GRID, GRID) float64: what the network holds after its norm layer."""
    N = img.shape[0]
    W1 = conv1_weights().astype(f32)
    xp = np.pad(img.astype(f32), ((0, 0), (1, 1), (1, 1)))
    g = np.zeros((N, NDIR, 28, 28), dtype=f32)
    for dy in range(3):
        for dx in range(3):
            patch = xp[:, dy:dy + 28, dx:dx + 28]
            for k in range(NDIR):
                if W1[k, 0, dy, dx] != 0:
                    g[:, k] += W1[k, 0, dy, dx] * patch
    g = np.maximum(g, 0)
    h = np.empty_like(g)
    for k in range(NDIR):
        h[:, k] = g[:, k] - g[:, (k + 2) % NDIR] - g[:, (k - 2) % NDIR]
    h = np.maximum(h, 0).astype(np.float64)
    g1 = gauss1d()
    M = np.zeros((GRID, 28 + 2 * POOL_P))
    for i in range(GRID):
        M[i, i * POOL_S:i * POOL_S + POOL_K] = g1
    M = M[:, POOL_P:POOL_P + 28]                                # (7, 28): the Gaussian pooling is separable
    p = np.matmul(np.matmul(M, h), M.T)                         # (N, NDIR, 7, 7)
    sl = sqrt_slopes()
    out = np.zeros_like(p)
    for j, bj in enumerate(BREAKS):
        out += sl[j] * np.maximum(p - bj, 0)
    ms = (out ** 2).mean((1, 2, 3), keepdims=True)              # each drawing is scaled by its own length only
    return out / np.sqrt(ms + NORM_EPS)


def template_rows(fe):
    """fe (N, NDIR, 7, 7) -> list of four (N, NDIR*16) arrays, one per quadrant."""
    rows = []
    for (ri, ci, wq) in quadrants():
        v = fe[:, :, ri][:, :, :, ci] * np.sqrt(wq)
        rows.append(v.reshape(fe.shape[0], -1))
    return rows


# ----------------------------------------------------------------------------------------------
def build(out_dir, per_class=PER_CLASS, verbose=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    DQ = NDIR * 16
    K = 10 * per_class
    Wq = [np.zeros((K, DQ), dtype=f32) for _ in range(4)]
    bq = [np.zeros(K, dtype=f32) for _ in range(4)]
    chunk = 100
    for d in range(10):
        rng = np.random.default_rng([SEED, d])
        for s in range(0, per_class, chunk):
            n = min(chunk, per_class - s)
            rows = template_rows(features(make_images(d, n, rng)))
            for q in range(4):
                Wq[q][d * per_class + s:d * per_class + s + n] = rows[q]
                bq[q][d * per_class + s:d * per_class + s + n] = -0.5 * (rows[q] ** 2).sum(1)
        if verbose:
            print("digit", d, "done", flush=True)

    nb = len(BREAKS)
    W4 = np.zeros((NDIR * nb, NDIR, 1, 1))
    b4 = np.zeros(NDIR * nb)
    W5 = np.zeros((2 * NDIR, NDIR * nb, 1, 1))
    sl = sqrt_slopes()
    for j in range(nb):
        for k in range(NDIR):
            W4[j * NDIR + k, k, 0, 0] = 1.0
            b4[j * NDIR + k] = -BREAKS[j]
            W5[k, j * NDIR + k, 0, 0] = sl[j]              # f
            W5[NDIR + k, j * NDIR + k, 0, 0] = -sl[j]      # -f, so that the norm layer divides by the length of f
    W3 = np.zeros((NDIR, NDIR, POOL_K, POOL_K))
    g1 = gauss1d()
    for k in range(NDIR):
        W3[k, k] = np.outer(g1, g1)
    weights = {
        "c1.W": conv1_weights(), "c1.b": np.zeros(NDIR),
        "c2.W": conv2_weights(), "c2.b": np.zeros(NDIR),
        "c3.W": W3, "c3.b": np.zeros(NDIR),
        "c4.W": W4, "c4.b": b4,
        "c5.W": W5, "c5.b": np.zeros(2 * NDIR),
    }
    layers = [
        {"name": "c1", "op": "conv", "inputs": ["image"], "out": NDIR, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "c2", "op": "conv", "inputs": ["r1"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "r2", "op": "relu", "inputs": ["c2"]},
        {"name": "c3", "op": "conv", "inputs": ["r2"], "out": NDIR, "kernel": [POOL_K, POOL_K], "stride": POOL_S, "padding": POOL_P},
        {"name": "c4", "op": "conv", "inputs": ["c3"], "out": NDIR * nb, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "r4", "op": "relu", "inputs": ["c4"]},
        {"name": "c5", "op": "conv", "inputs": ["r4"], "out": 2 * NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "flat", "op": "reshape", "inputs": ["c5"], "shape": [1, 2 * NDIR * GRID * GRID, 1]},
        {"name": "unit", "op": "norm", "inputs": ["flat"]},
        {"name": "feat", "op": "reshape", "inputs": ["unit"], "shape": [2 * NDIR, GRID, GRID]},
    ]
    for q, (ri, ci, wq) in enumerate(quadrants()):
        Ws = np.zeros((DQ, 2 * NDIR, GRID, GRID))
        for k in range(NDIR):
            for i in range(4):
                for j in range(4):
                    Ws[k * 16 + i * 4 + j, k, ri[i], ci[j]] = np.sqrt(wq[i, j])
        weights["sel%d.W" % q], weights["sel%d.b" % q] = Ws, np.zeros(DQ)
        weights["m%d.W" % q], weights["m%d.b" % q] = Wq[q], bq[q]
        layers.append({"name": "sel%d" % q, "op": "conv", "inputs": ["feat"], "out": DQ, "kernel": [GRID, GRID], "stride": 1, "padding": 0})
        layers.append({"name": "m%d" % q, "op": "dense", "inputs": ["sel%d" % q], "out": K})
    layers += [
        {"name": "top", "op": "add", "inputs": ["m0", "m1"]},
        {"name": "bottom", "op": "add", "inputs": ["m2", "m3"]},
        {"name": "left", "op": "add", "inputs": ["m0", "m2"]},
        {"name": "right", "op": "add", "inputs": ["m1", "m3"]},
        {"name": "whole", "op": "add", "inputs": ["top", "bottom"]},
    ]
    best = []
    for nm in ["whole", "top", "bottom", "left", "right", "m0", "m1", "m2", "m3"]:
        layers.append({"name": nm + "_bank", "op": "reshape", "inputs": [nm], "shape": [1, 10, per_class]})
        layers.append({"name": nm + "_best", "op": "maxpool", "inputs": [nm + "_bank"], "size": [1, per_class], "stride": [1, per_class]})
        best.append(nm + "_best")
    layers.append({"name": "total", "op": "add", "inputs": best})
    layers.append({"name": "scores", "op": "reshape", "inputs": ["total"], "shape": [10]})
    weights = {k: np.asarray(v).astype(f32) for k, v in weights.items()}
    (out_dir / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
    np.savez(out_dir / "weights.npz", **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
