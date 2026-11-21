"""Writes network.json and weights.npz for a hand-designed digit classifier.  Usage: python3 build.py OUT_DIR

Nothing is read from disk and nothing is trained or fitted.  Every weight comes from one of two sources:
  * formulas: Gaussian-derivative filters, a direction-sharpening rule, a Gaussian pooling filter, a piecewise-linear
    square-root curve;
  * drawings: each digit is drawn by this program as pen strokes along paths typed from general knowledge of how
    people write digits (an oval for 0, a bar for 1, hook + diagonal + base for 2, ...), with random variation of the
    proportions, slant, stroke width and pen wobble (fixed seed).  Every drawing is turned into ONE template row of
    the last dense layer, normalised by itself alone (its own mean and its own standard deviation).
The network scores a class by its best-matching template (a maximum over that class's templates).
"""
import json
import os
import sys

import numpy as np

SEED = 20240607
PER_CLASS = 18000          # drawings (templates) per class
rng = np.random.default_rng(SEED)
WREL = 0.15                # pen width of the current batch of drawings, in units of the digit height (set in generate)


# ====================================================================== small helpers for typing pen paths
def U(B, a, b):
    return rng.uniform(a, b, B)


def coin(B, p):
    return rng.random(B) < p


def P(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    return np.stack([x, y], -1)


def S(*pts):
    return np.stack(pts, 1)


def lerp(a, b, t):
    t = np.asarray(t, float)
    if t.ndim == 1:
        t = t[:, None]
    return a + (b - a) * t


def ell(cx, cy, rx, ry, a):
    """Point on an ellipse; angle in degrees, 0 = right, 90 = bottom (y runs downward), 180 = left, 270 = top."""
    a = np.deg2rad(a)
    return P(cx + rx * np.cos(a), cy + ry * np.sin(a))


def ellr(cx, cy, rx, ry, a, phi):
    """Point on an ellipse whose long axis is tilted by phi degrees (rising to the right)."""
    a = np.deg2rad(a); phi = np.deg2rad(phi)
    dx = rx * np.cos(a); dy = ry * np.sin(a)
    return P(cx + dx * np.cos(phi) + dy * np.sin(phi), cy - dx * np.sin(phi) + dy * np.cos(phi))


# ====================================================================== the ten digits as pen strokes
# x runs to the right, y runs downward, a digit is about 1 unit tall.  Each function returns a list of strokes;
# a stroke is an array (B, K, 2) of pen positions that a smooth curve is passed through.  B drawings at a time,
# each with its own random proportions.

def style0(B):
    """An oval drawn counter-clockwise from the top; the ends may leave a gap, overlap, curl in or stick out."""
    rx = U(B, 0.2, 0.5); ry = 0.5
    a0 = U(B, -130, -40); sweep = U(B, 330, 430)
    e2 = U(B, 0, 0.07); p2 = U(B, 0, 180); e3 = U(B, 0, 0.06); p3 = U(B, 0, 120)
    sp = U(B, -0.12, 0.12)
    u = rng.random(B)
    curl = np.where(u < 0.2, U(B, 0.1, 0.35), 0.0) - np.where(u > 0.8, U(B, 0.1, 0.4), 0.0)
    lead = np.where(coin(B, 0.2), U(B, 0.08, 0.3), 0.0)
    n = 14; pts = []
    for i in range(n):
        t = i / (n - 1)
        a = a0 - sweep * t
        r = (1 + e2 * np.cos(np.deg2rad(2 * (a - p2))) + e3 * np.cos(np.deg2rad(3 * (a - p3)))) * (1 + sp * (t - 0.5))
        if i == 0: r = r * (1 + lead)
        if i == n - 1: r = r * (1 - curl)
        pts.append(ell(0.0, 0.0, rx * r, ry * r, a))
    return [S(*pts)]


def style1(B, flag=False, base=False):
    """A single bar; optionally a short flag rising to its top, and a foot."""
    top = P(np.zeros(B), np.zeros(B)); bot = P(U(B, -0.03, 0.03), np.ones(B))
    mid = P(U(B, -0.04, 0.04), U(B, 0.4, 0.6))
    st = [S(top, mid, bot)]
    if flag:
        fy = U(B, 0.12, 0.36)
        st.append(S(P(-fy * U(B, 0.5, 1.0), fy), top))
    if base:
        st.append(S(bot + P(-U(B, 0.15, 0.42), U(B, -0.03, 0.03)), bot + P(U(B, 0.15, 0.42), U(B, -0.03, 0.03))))
    return st


def style2(B, kind):
    """Top hook, diagonal down to the bottom left, base to the right; the turn is a corner, a round bend or a loop.
    The hook may be narrower than the base and may curl in at its start; the base may sag and flick up at its end."""
    W = U(B, 0.55, 1.1)
    rty = U(B, 0.16, 0.3); rx = 0.5 * W * U(B, 0.55, 1.0)
    cx = rx + (W - 2 * rx) * U(B, 0.0, 0.7)
    a0 = U(B, 125, 225); a1 = U(B, 375, 425)
    top = [ell(cx, rty, rx, rty, a0 + (a1 - a0) * i / 4) for i in range(5)]
    xbl = W * U(B, -0.05, 0.2)
    corner = P(xbl, np.ones(B))
    mid = lerp(top[-1], corner, U(B, 0.45, 0.6)) + P(W * U(B, -0.08, 0.08), 0.0)
    bend = P(W * U(B, 0.85, 1.2), 1 + U(B, -0.25, 0.08))
    bmid = lerp(corner, bend, U(B, 0.4, 0.6)) + P(0.0, U(B, -0.03, 0.1))
    if kind == 'corner':
        return [S(*top, mid, corner), S(corner, bmid, bend)]
    if kind == 'round':
        return [S(*top, mid, corner, bmid, bend)]
    la = U(B, 0.07, 0.24); lb = la * U(B, 0.25, 0.7); phi = U(B, 0.0, 45.0)
    cph, sph = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))
    lcx = xbl + la * cph - lb * sph; lcy = 1 - la * sph - lb * cph
    lp = [ellr(lcx, lcy, la, lb, a, phi) for a in (75, 170, 265, 325)]
    bend = P(W * U(B, 0.85, 1.25), 1 + U(B, -0.22, 0.22))
    bm2 = lerp(lp[-1], bend, U(B, 0.45, 0.6)) + P(0.0, U(B, -0.03, 0.08))
    return [S(*top, mid, *lp, bm2, bend)]


def style3(B, kind):
    """Two bowls open to the left, meeting in a point in the middle; the top may be a flat bar with a diagonal."""
    W = U(B, 0.5, 0.85)
    yj = U(B, 0.42, 0.54); xj = W * U(B, 0.3, 0.55)
    nub = W * (U(B, 0.0, 0.1) if kind == 'smooth' else U(B, 0.0, 0.18))
    J = P(xj - nub, yj + U(B, -0.02, 0.02))
    rt = W * U(B, 0.78, 1.0); ryt = yj / 2
    ryb = (1 - yj) / 2; ycb = yj + ryb
    xe = W * U(B, -0.15, 0.2); a1 = U(B, 495, 545)
    bot = [ell(xj, ycb, W - xj, ryb, a) for a in (300, 330, 360, 405, 450)]
    bot += [ell(xj, ycb, xj - xe, ryb, 450 + (a1 - 450) * f) for f in (0.5, 1.0)]
    if kind == 'flat':
        s0 = P(W * U(B, 0.0, 0.15), U(B, 0.0, 0.06)); s1 = P(rt, U(B, -0.03, 0.06))
        dm = lerp(s1, J, 0.5) + P(W * U(B, -0.05, 0.08), 0.0)
        return [S(s0, s1), S(s1, dm, J), S(J, *bot)]
    xs = W * U(B, 0.0, 0.2); a0 = U(B, 190, 235)
    topb = [ell(xj, ryt, xj - xs, ryt, a0), ell(xj, ryt, xj - xs, ryt, (a0 + 270) / 2)]
    topb += [ell(xj, ryt, rt - xj, ryt, a) for a in (270, 315, 360, 405, 438)]
    if kind == 'smooth':
        return [S(*topb, J, *bot)]
    return [S(*topb, J), S(J, *bot)]


def style4(B, kind):
    """Left stroke down to a cross bar, bar to the right, and a tall stem through the bar; open or closed top.
    What makes it a 4 and not a 9: an open top stays visibly open, and the bar visibly crosses the stem."""
    W = U(B, 0.6, 1.0)
    xs = W * U(B, 0.5, 0.78); yb = U(B, 0.45, 0.72)
    if kind == 'closed':
        ext = np.maximum(W * U(B, 0.18, 0.45), 0.5 * WREL + 0.12)
    else:
        gap = np.maximum(W * U(B, 0.25, 0.6), WREL + 0.12)
        xs = np.maximum(xs, gap - 0.05 * W)
        ext = np.maximum(W * U(B, 0.02, 0.42), 0.5 * WREL + U(B, 0.0, 0.08))
    st = P(xs + U(B, -0.04, 0.08), U(B, -0.03, 0.22)); sb = P(xs + U(B, -0.05, 0.05), np.ones(B))
    sm = lerp(st, sb, 0.5) + P(U(B, -0.03, 0.03), 0.0)
    bl = P(np.zeros(B), yb)
    br = P(xs + ext, yb + U(B, -0.1, 0.04))
    if kind == 'closed':
        lt = st + P(U(B, -0.03, 0.03), U(B, 0.0, 0.05))
    else:
        lt = P(xs - gap, U(B, 0.0, 0.18))
    lm = lerp(lt, bl, 0.5) + P(U(B, -0.05, 0.05), 0.0)
    stem = S(st, sm, sb)
    if kind == 'round':
        c1 = lerp(lt, bl, 0.8) + P(U(B, -0.03, 0.02), 0.0)
        c2 = lerp(bl, br, 0.22) + P(0.0, U(B, -0.01, 0.03))
        return [S(lt, lm, c1, c2, br), stem]
    return [S(lt, lm, bl), S(bl, br), stem]


def style5(B, kind):
    """Top bar (flat, or a long flag rising to the right), short down stroke on the left, then a bowl open to the left."""
    W = U(B, 0.55, 0.95)
    tl = P(W * U(B, 0.08, 0.35), U(B, 0.0, 0.08))
    tr = P(W * U(B, 0.72, 1.12), U(B, -0.1, 0.07))
    fl = W * U(B, 0.5, 1.2); fa = np.deg2rad(U(B, 0.0, 32.0))       # a long flag rising to the right, for half of them
    tr = np.where(coin(B, 0.5)[:, None], tl + P(fl * np.cos(fa), -fl * np.sin(fa)), tr)
    V = P(W * U(B, 0.03, 0.3), U(B, 0.27, 0.5))            # the down stroke may be quite short
    ytop = V[:, 1] - U(B, 0.0, 0.06)
    ryb = (1 - ytop) / 2; ycb = ytop + ryb
    xcb = W * U(B, 0.42, 0.6); rr = W * U(B, 0.88, 1.0) - xcb
    xe = W * U(B, -0.1, 0.2); a1 = U(B, 470, 525)
    bowl = [ell(xcb, ycb, rr, ryb, a) for a in (280, 320, 360, 405, 450)]
    bowl += [ell(xcb, ycb, xcb - xe, ryb, 450 + (a1 - 450) * f) for f in (0.5, 1.0)]
    vm = lerp(tl, V, 0.5) + P(U(B, -0.03, 0.03), 0.0)
    if kind == 'onestroke':
        return [S(tr, lerp(tr, tl, 0.5) + P(0.0, U(B, -0.03, 0.03)), tl, vm, V, *bowl)]
    off = P(W * U(B, -0.06, 0.1), U(B, -0.07, 0.04)) * coin(B, 0.5)[:, None]
    bar = S(tl + off, tr + off * 0.5)
    if kind == 'angular':
        return [bar, S(tl, vm, V), S(V, *bowl)]
    return [bar, S(tl, vm, V, *bowl)]


def style6(B, hook=False):
    """A stem from the top right curving down the left side into a closed loop at the bottom."""
    W = U(B, 0.5, 0.85)
    ry = U(B, 0.16, 0.31); rx = 0.5 * W * U(B, 0.78, 1.0)
    xc = rx + W * U(B, 0.0, 0.06); yc = 1 - ry
    top = P(W * U(B, 0.4, 1.05), np.zeros(B))
    left = ell(xc, yc, rx, ry, 180.0)
    mid = lerp(top, left, U(B, 0.45, 0.6)) - P(W * U(B, -0.03, 0.14), 0.0)
    aend = -180 + U(B, -45, 40); sh = U(B, 0.7, 1.0)
    loop = [ell(xc, yc, rx, ry, a) for a in (135, 90, 45, 0, -45, -90)]
    loop += [ell(xc, yc, rx * (1 + sh) / 2, ry * (1 + sh) / 2, -90 + (aend + 90) * 0.5), ell(xc, yc, rx * sh, ry * sh, aend)]
    pre = [top + P(W * U(B, 0.06, 0.25), U(B, 0.02, 0.12))] if hook else []
    return [S(*pre, top, mid, left, *loop)]


def style7(B, sharp=True, serif=False, cross=False, foot=False):
    """Top bar and a diagonal down to the bottom; optional tick at the start of the bar and optional cross bar."""
    W = U(B, 0.55, 0.95)
    l = P(np.zeros(B), U(B, 0.0, 0.14)); c = P(W, U(B, 0.0, 0.1)); b = P(W * U(B, 0.1, 0.7), np.ones(B))
    bm = lerp(l, c, 0.5) + P(0.0, U(B, -0.05, 0.05))
    dm = lerp(c, b, 0.5) + P(W * U(B, -0.12, 0.1), 0.0)
    ft = [b + P(W * U(B, 0.05, 0.18), U(B, -0.04, 0.02))] if foot else []
    st = [S(l, bm, c), S(c, dm, b, *ft)] if sharp else [S(l, bm, c, dm, b, *ft)]
    if serif:
        st.append(S(l + P(U(B, -0.02, 0.03), U(B, 0.08, 0.35)), l))
    if cross:
        q = lerp(c, b, U(B, 0.42, 0.62)) + P(dm[:, 0] - 0.5 * (c[:, 0] + b[:, 0]), 0.0) * 0.8
        st.append(S(q - P(W * U(B, 0.15, 0.4), U(B, -0.04, 0.04)), q + P(W * U(B, 0.15, 0.38), U(B, -0.04, 0.04))))
    return st


def style8(B, start_tail=False):
    """One stroke: over the top loop to the left, down through the crossing, round the bottom loop, back up.
    Either loop may be narrow; the ends at the top right may leave a gap, overlap or stick out; the stroke may
    begin with a long tail coming in from the upper right."""
    W = U(B, 0.4, 0.85)
    yx = U(B, 0.36, 0.56)
    f1 = U(B, 0.35, 1.0); f2 = U(B, 0.35, 1.0); fs = U(B, 0.8, 1.0) / np.maximum(f1, f2)
    rxt = 0.5 * W * f1 * fs; rxb = 0.5 * W * f2 * fs
    xct = 0.5 * W + W * U(B, -0.08, 0.1); xcb = 0.5 * W + W * U(B, -0.08, 0.06)
    ryt = yx / 2; ryb = (1 - yx) / 2; yct = ryt; ycb = yx + ryb
    X = P(0.5 * (xct + xcb) + W * U(B, -0.04, 0.04), yx)
    a_s = U(B, -80, -5); gap = U(B, -45, 60)
    tail = np.where(coin(B, 0.3), U(B, 0.1, 0.7), 0.0); lead = np.where(coin(B, 0.3), U(B, 0.1, 0.6), 0.0)
    if start_tail:
        a_s = U(B, -75, -25); gap = U(B, -20, 40); lead = 0.0
    a_e = a_s + gap
    pts = [ell(xct, yct, rxt * (1 + lead), ryt * (1 + 0.3 * lead), a_s)]
    if start_tail:
        pts = [pts[0] + P(W * U(B, 0.2, 0.75), -U(B, -0.05, 0.18))] + pts
    pts += [ell(xct, yct, rxt, ryt, a) for a in (-100, -140, -180, -222)]
    pts += [X]
    pts += [ell(xcb, ycb, rxb, ryb, a) for a in (-42, 0, 45, 90, 135, 180, 222)]
    pts += [X + P(U(B, -0.015, 0.015), U(B, -0.015, 0.015))]
    pts += [ell(xct, yct, rxt, ryt, 42.0), ell(xct, yct, rxt, ryt, 42 + (a_e - 42) * 0.5)]
    pts += [ell(xct, yct, rxt * (1 + tail), ryt * (1 + tail), a_e)]
    return [S(*pts)]


def style9(B, kind):
    """A loop at the top and a tail down its right side; the tail is straight, leaning, or hooked to the left.
    The loop may be egg-shaped or flat-sided, and may stay open towards the top of the tail."""
    W = U(B, 0.5, 0.85)
    ry = U(B, 0.18, 0.31); rx = 0.5 * W * U(B, 0.7, 1.0); xc = rx; yc = ry
    e2 = U(B, 0, 0.08); p2 = U(B, 0, 180); e3 = U(B, 0, 0.08); p3 = U(B, 0, 120)

    def lp(a):
        r = 1 + e2 * np.cos(np.deg2rad(2 * (a - p2))) + e3 * np.cos(np.deg2rad(3 * (a - p3)))
        return ell(xc, yc, rx * r, ry * r, a)
    xs = xc + rx * U(B, 0.88, 1.06)
    if kind == 'hook':
        sb = P(xs + W * U(B, -0.75, -0.3), 1 - U(B, 0.0, 0.08))
        s2 = P(xs + W * U(B, -0.12, 0.03), U(B, 0.8, 0.9)); s3 = lerp(s2, sb, 0.45) + P(0.0, U(B, 0.04, 0.1))
        tailpts = [s2, s3, sb]
    else:
        sb = P(xs + W * U(B, -0.55, 0.1), np.ones(B))
        tailpts = [sb]
    if kind == 'leaf':
        # the loop meets the tail in a point at the top right and rejoins it lower down; its left end is blunt
        st = P(xs + U(B, -0.02, 0.05), U(B, 0.0, 0.08))
        J = lerp(st, sb, U(B, 0.36, 0.58)) + P(W * U(B, -0.03, 0.03), 0.0)
        T = st + P(U(B, -0.04, 0.02) - np.where(coin(B, 0.35), W * U(B, 0.08, 0.3), 0.0), U(B, 0.0, 0.08))
        L = P(xs - W * U(B, 0.5, 0.95), T[:, 1] + (J[:, 1] - T[:, 1]) * U(B, 0.3, 0.75))
        h = (J[:, 1] - T[:, 1]) * U(B, 0.12, 0.3)
        tm = lerp(T, L, U(B, 0.4, 0.6)) - P(0.0, U(B, 0.03, 0.16))
        bm = lerp(L, J, U(B, 0.4, 0.6)) + P(0.0, U(B, 0.0, 0.14))
        return [S(T, tm, L - P(0.0, h), L + P(0.0, h), bm, J), S(st, J, sb)]
    if kind == 'cw':
        a_s = U(B, 15, 70)
        loop = [lp(a_s + (360 - a_s) * i / 8) for i in range(9)]
        sm = lerp(loop[-1], tailpts[0], 0.5) + P(W * U(B, -0.03, 0.08), 0.0)
        return [S(*loop, sm, *tailpts)]
    a_s = U(B, -85, 10); a_e = -360 + U(B, -30, 35)
    loop = [lp(a_s + (a_e - a_s) * i / 8) for i in range(9)]
    st = P(xs, yc - ry * U(B, 0.2, 1.3))
    sm = lerp(st, tailpts[0], 0.5) + P(W * U(B, -0.04, 0.08), 0.0)
    return [S(*loop), S(st, sm, *tailpts)]


# class -> list of (share of the class's drawings, function, settings, (mean, spread) of the slant[, rotation])
SL = (0.15, 0.22)
STYLES = {
    0: [(1.0, style0, {}, SL, (12.0, 14.0, -15.0, 45.0))],
    1: [(0.7, style1, {}, (0.3, 0.3)), (0.17, style1, dict(flag=True), (0.12, 0.18)),
        (0.13, style1, dict(flag=True, base=True), (0.12, 0.18))],
    2: [(0.3, style2, dict(kind='corner'), SL), (0.25, style2, dict(kind='round'), SL), (0.45, style2, dict(kind='loop'), SL)],
    3: [(0.5, style3, dict(kind='cusp'), SL), (0.25, style3, dict(kind='smooth'), SL), (0.25, style3, dict(kind='flat'), SL)],
    4: [(0.55, style4, dict(kind='open'), SL), (0.3, style4, dict(kind='round'), SL), (0.15, style4, dict(kind='closed'), SL)],
    5: [(0.35, style5, dict(kind='angular'), SL), (0.4, style5, dict(kind='smooth'), SL), (0.25, style5, dict(kind='onestroke'), SL)],
    6: [(0.7, style6, {}, SL), (0.3, style6, dict(hook=True), SL)],
    7: [(0.25, style7, dict(sharp=True), SL), (0.25, style7, dict(sharp=False), SL), (0.16, style7, dict(sharp=True, serif=True), SL),
        (0.11, style7, dict(sharp=True, cross=True), SL), (0.05, style7, dict(sharp=False, cross=True), SL),
        (0.06, style7, dict(sharp=True, serif=True, cross=True), SL), (0.06, style7, dict(sharp=True, foot=True), SL),
        (0.06, style7, dict(sharp=False, foot=True), SL)],
    8: [(0.65, style8, {}, SL), (0.35, style8, dict(start_tail=True), SL)],
    9: [(0.35, style9, dict(kind='ccw'), SL), (0.2, style9, dict(kind='cw'), SL), (0.2, style9, dict(kind='hook'), SL),
        (0.25, style9, dict(kind='leaf'), SL)],
}


# ====================================================================== from pen positions to a 28 x 28 image
def spline(C, m=5):
    """Centripetal Catmull-Rom curve through the pen positions C (B, K, 2); m points per span."""
    B, K, _ = C.shape
    Pm = np.concatenate([2 * C[:, :1] - C[:, 1:2], C, 2 * C[:, -1:] - C[:, -2:-1]], 1)
    d = np.sqrt(np.sqrt(((Pm[:, 1:] - Pm[:, :-1]) ** 2).sum(-1)) + 1e-9) + 1e-6
    out = []
    u = (np.arange(m) / m)[None, :, None]
    for i in range(K - 1):
        p0, p1, p2, p3 = (Pm[:, i + j][:, None, :] for j in range(4))
        t1 = d[:, i][:, None, None]; t2 = t1 + d[:, i + 1][:, None, None]; t3 = t2 + d[:, i + 2][:, None, None]
        t = t1 + (t2 - t1) * u
        a1 = (t1 - t) / t1 * p0 + t / t1 * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / t2 * a1 + t / t2 * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(C[:, -1:])
    return np.concatenate(out, 1)


def make_segments(strokes, slant, B, rot=(0.0, 5.0, -12.0, 12.0), jit=0.015):
    """Pen wobble, a smooth random bend of the whole drawing, then slant / width / small rotation.
    Returns the start and end points of all line segments, each (B, S, 2)."""
    polys = [spline(C + rng.normal(0, jit, C.shape)) for C in strokes]
    allp = np.concatenate(polys, 1)
    c = 0.5 * (allp.min(1) + allp.max(1))
    wf = []
    strength = U(B, 0.7, 1.6)
    for _ in range(4):
        g = rng.normal(0, 1, (B, 2)); g = g / np.sqrt((g * g).sum(1, keepdims=True)) * U(B, 0.25, 1.2)[:, None]
        dv = rng.normal(0, 1, (B, 2)); dv = dv / np.sqrt((dv * dv).sum(1, keepdims=True)) * (strength * U(B, 0.0, 0.05))[:, None]
        wf.append((g, U(B, 0, 2 * np.pi), dv))
    sm, ss = slant
    sh = np.clip(rng.normal(sm, ss, B), sm - 2.2 * ss, sm + 2.2 * ss)
    ax = np.exp(np.clip(rng.normal(0, 0.13, B), -0.3, 0.3))
    th = np.deg2rad(np.clip(rng.normal(rot[0], rot[1], B), rot[2], rot[3]))
    out = []
    for p in polys:
        q = p - c[:, None, :]
        disp = 0
        for g, ph, dv in wf:
            phase = q[..., 0] * g[:, None, 0] + q[..., 1] * g[:, None, 1]
            disp = disp + np.sin(2 * np.pi * phase + ph[:, None])[..., None] * dv[:, None, :]
        q = q + disp
        x = ax[:, None] * q[..., 0] - sh[:, None] * q[..., 1]
        y = q[..., 1]
        ct, sn = np.cos(th)[:, None], np.sin(th)[:, None]
        out.append(np.stack([ct * x - sn * y, sn * x + ct * y], -1))
    A = np.concatenate([p[:, :-1] for p in out], 1)
    E = np.concatenate([p[:, 1:] for p in out], 1)
    return A, E


GY, GX = np.meshgrid(np.arange(28, dtype=np.float32), np.arange(28, dtype=np.float32), indexing='ij')
GX = GX.ravel(); GY = GY.ravel()


def raster(A, E, w):
    """Ink = 1 within half the stroke width of the pen path, falling linearly to 0 over one pixel."""
    D = E - A
    inv = (1.0 / ((D * D).sum(-1) + 1e-12)).astype(np.float32)
    dx = GX[None, :, None] - A[:, None, :, 0]
    dy = GY[None, :, None] - A[:, None, :, 1]
    t = dx * D[:, None, :, 0]
    t += dy * D[:, None, :, 1]
    t *= inv[:, None, :]
    np.clip(t, 0.0, 1.0, out=t)
    dx -= t * D[:, None, :, 0]
    dy -= t * D[:, None, :, 1]
    dx *= dx; dy *= dy; dx += dy
    dist = np.sqrt(dx.min(-1))
    return np.clip((0.5 * w + 0.5).astype(np.float32)[:, None] - dist, 0.0, 1.0)


def pen_widths(B):
    """Stroke width in pixels: 1.5 to 5, mostly 2 to 4."""
    return 1.5 + 3.5 * rng.beta(2.0, 3.0, B)


def render(A, E, w, chunk=64):
    """As the task describes the images: the drawing is scaled so its ink fits a 20-pixel box, and placed so that
    its centre of mass is at the centre of the 28 x 28 frame (to within half a pixel)."""
    B = len(A)
    off = rng.uniform(-0.5, 0.5, (B, 2))
    lo = np.minimum(A.min(1), E.min(1)); hi = np.maximum(A.max(1), E.max(1))
    sc = (20.0 - w) / np.maximum((hi - lo).max(1), 1e-6)
    ctr = 0.5 * (lo + hi)
    A = (A - ctr[:, None]) * sc[:, None, None] + 13.5
    E = (E - ctr[:, None]) * sc[:, None, None] + 13.5
    out = np.empty((B, 784), np.float32)
    gx, gy = GX.astype(np.float64), GY.astype(np.float64)
    for i in range(0, B, chunk):
        a, e, ww = A[i:i + chunk], E[i:i + chunk], w[i:i + chunk]
        im = raster(a.astype(np.float32), e.astype(np.float32), ww).astype(np.float64)
        m = im.sum(1)
        cm = np.stack([(im * gx).sum(1) / m, (im * gy).sum(1) / m], -1)
        sh = (13.5 + off[i:i + chunk] - cm)[:, None, :]
        out[i:i + chunk] = raster((a + sh).astype(np.float32), (e + sh).astype(np.float32), ww)
    return out.reshape(B, 28, 28)


def generate(cls, n):
    """n drawings of digit cls, its styles in fixed shares."""
    styles = STYLES[cls]
    counts = [int(round(s[0] * n)) for s in styles]
    counts[0] += n - sum(counts)
    ims = []
    global WREL
    for s, k in zip(styles, counts):
        if k > 0:
            w = pen_widths(k)
            WREL = w / (20.0 - w)
            A, E = make_segments(s[1](k, **s[2]), s[3], k, *s[4:])
            ims.append(render(A, E, w))
    return np.concatenate(ims, 0)


# ====================================================================== the fixed feature layers (formulas only)
NDIR = 8
GRID = 9                       # 9 x 9 pooling positions, 3 pixels apart
DIM = NDIR * GRID * GRID       # 648 numbers describe an image
SIG_G, SIG_P = 1.0, 2.0        # scale of the gradient filter and of the pooling window, in pixels


def gradient_filters():
    """7 x 7 Gaussian-derivative filters: brightness slope along x and along y."""
    u = np.arange(7) - 3.0
    g = np.exp(-u ** 2 / (2 * SIG_G ** 2)); g /= g.sum()
    d = u * g / SIG_G ** 2
    return np.outer(g, d), np.outer(d, g)          # [dy, dx] : d/dx , d/dy


def pool_filter():
    u = np.arange(12) - 5.5
    g = np.exp(-u ** 2 / (2 * SIG_P ** 2))
    return g / g.sum()


# Square-root-like curve built from straight pieces.  REF is the pooled response of one long full-contrast edge
# through the middle of a pooling window, 1 / (sqrt(2 pi) SIG_P), worked out from the filter formulas.
REF = 1.0 / (np.sqrt(2 * np.pi) * SIG_P)
KNOTS = np.array([0.0, REF / 64, REF / 16, REF / 4, REF])
VALS = np.array([0.0, 0.125, 0.25, 0.5, 1.0, 2.0])         # the curve at the knots and at 4 REF: sqrt(x / REF)
_sl = np.diff(VALS) / np.diff(np.append(KNOTS, 4 * REF))
COEF = np.append(_sl[0], np.diff(_sl))                    # curve(x) = sum_j COEF[j] * relu(x - KNOTS[j])
ANG = np.arange(NDIR) * (2 * np.pi / NDIR)


def features(imgs):
    """The same arithmetic as layers c1 ... c5 of the network, for drawings (B, 28, 28) -> (B, 648), float64."""
    B = len(imgs)
    kx, ky = gradient_filters()
    xp = np.pad(imgs.astype(np.float64), ((0, 0), (3, 3), (3, 3)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (7, 7), axis=(1, 2)).reshape(B, 784, 49)
    gx = win @ kx.ravel(); gy = win @ ky.ravel()
    a = np.maximum(np.cos(ANG)[None, :, None] * gx[:, None, :] + np.sin(ANG)[None, :, None] * gy[:, None, :], 0)
    r = np.maximum(a - np.roll(a, 2, axis=1) - np.roll(a, -2, axis=1), 0).reshape(B, NDIR, 28, 28)
    g = pool_filter()
    G = np.zeros((GRID, 28))
    for i in range(GRID):
        for k in range(12):
            j = 3 * i - 4 + k
            if 0 <= j < 28:
                G[i, j] = g[k]
    p = np.einsum('iy,bkyx->bkix', G, r)
    p = np.einsum('bkix,jx->bkij', p, G)
    out = np.zeros_like(p)
    for cj, tj in zip(COEF, KNOTS):
        out += cj * np.maximum(p - tj, 0)
    return out.reshape(B, DIM)


def build(out_dir, per_class=PER_CLASS):
    kx, ky = gradient_filters()
    W1 = np.stack([np.cos(t) * kx + np.sin(t) * ky for t in ANG])[:, None]            # (8, 1, 7, 7)
    W2 = np.zeros((NDIR, NDIR, 1, 1))
    for k in range(NDIR):
        W2[k, k] = 1.0; W2[k, (k + 2) % NDIR] = -1.0; W2[k, (k - 2) % NDIR] = -1.0
    g = pool_filter()
    W3 = np.zeros((NDIR, NDIR, 12, 12))
    for k in range(NDIR):
        W3[k, k] = np.outer(g, g)
    nk = len(KNOTS)
    W4 = np.zeros((NDIR * nk, NDIR, 1, 1)); b4 = np.zeros(NDIR * nk)
    W5 = np.zeros((NDIR, NDIR * nk, 1, 1))
    for k in range(NDIR):
        for j in range(nk):
            W4[k * nk + j, k] = 1.0; b4[k * nk + j] = -KNOTS[j]; W5[k, k * nk + j] = COEF[j]

    # one template per drawing, each normalised by itself alone
    Wd = np.empty((10 * per_class, DIM), np.float32)
    for c in range(10):
        ims = generate(c, per_class)
        for i in range(0, per_class, 500):
            f = features(ims[i:i + 500])
            mu = f.mean(1, keepdims=True)
            sd = np.sqrt(((f - mu) ** 2).mean(1, keepdims=True) + 1e-5)
            Wd[c * per_class + i: c * per_class + i + len(f)] = (f - mu) / sd / DIM
    weights = {
        'c1.W': W1, 'c1.b': np.zeros(NDIR), 'c2.W': W2, 'c2.b': np.zeros(NDIR), 'c3.W': W3, 'c3.b': np.zeros(NDIR),
        'c4.W': W4, 'c4.b': b4, 'c5.W': W5, 'c5.b': np.zeros(NDIR), 'match.W': Wd, 'match.b': np.zeros(10 * per_class),
    }
    net = {'layers': [
        {'name': 'c1', 'op': 'conv', 'inputs': ['image'], 'out': NDIR, 'kernel': [7, 7], 'stride': 1, 'padding': 3},
        {'name': 'r1', 'op': 'relu', 'inputs': ['c1']},
        {'name': 'c2', 'op': 'conv', 'inputs': ['r1'], 'out': NDIR, 'kernel': [1, 1], 'stride': 1, 'padding': 0},
        {'name': 'r2', 'op': 'relu', 'inputs': ['c2']},
        {'name': 'c3', 'op': 'conv', 'inputs': ['r2'], 'out': NDIR, 'kernel': [12, 12], 'stride': 3, 'padding': 4},
        {'name': 'c4', 'op': 'conv', 'inputs': ['c3'], 'out': NDIR * nk, 'kernel': [1, 1], 'stride': 1, 'padding': 0},
        {'name': 'r4', 'op': 'relu', 'inputs': ['c4']},
        {'name': 'c5', 'op': 'conv', 'inputs': ['r4'], 'out': NDIR, 'kernel': [1, 1], 'stride': 1, 'padding': 0},
        {'name': 'vec', 'op': 'reshape', 'inputs': ['c5'], 'shape': [1, DIM, 1]},
        {'name': 'unit', 'op': 'norm', 'inputs': ['vec']},
        {'name': 'match', 'op': 'dense', 'inputs': ['unit'], 'out': 10 * per_class},
        {'name': 'byclass', 'op': 'reshape', 'inputs': ['match'], 'shape': [1, 10, per_class]},
        {'name': 'best', 'op': 'maxpool', 'inputs': ['byclass'], 'size': [1, per_class], 'stride': [1, per_class]},
        {'name': 'scores', 'op': 'reshape', 'inputs': ['best'], 'shape': [10]},
    ]}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'network.json'), 'w') as f:
        json.dump(net, f, indent=1)
    np.savez(os.path.join(out_dir, 'weights.npz'), **{k: v.astype(np.float32) for k, v in weights.items()})


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
