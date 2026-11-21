#!/usr/bin/env python3
"""Writes network.json and weights.npz for a hand-designed handwritten-digit classifier.
Usage: python3 build.py OUT_DIR.   Reads no files: every number comes from this source.

Templates are digits drawn from typed pen paths (stroke models written from general knowledge of how each digit
is written), with randomly drawn slant, rotation, width, pen size and small shape jitter (fixed seed).  Each
rendered template is mapped through the network's own fixed feature layers (oriented-gradient filters from
formulas) and normalised by itself alone; the network scores an input by normalised correlation with every
template and takes the best match per class."""
import sys, os, json
"""Stroke models of handwritten digits typed from general knowledge, and an MNIST-style renderer.
Coordinates: x to the right, y downward, digit height about 1 (y from 0 at the top to 1 at the bottom)."""
import numpy as np


def catmull(pts, n=10):
    """Centripetal-ish Catmull-Rom spline through the given points (uniform parametrisation)."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        t = np.linspace(0, 1, n * max(1, len(P) - 1) + 1)[:, None]
        return P[0] * (1 - t) + P[-1] * t
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        a = 2 * p1
        b = p2 - p0
        c = 2 * p0 - 5 * p1 + 4 * p2 - p3
        d = -p0 + 3 * p1 - 3 * p2 + p3
        out.append(0.5 * (a + b * t + c * t ** 2 + d * t ** 3))
    out.append(P[-1:])
    return np.vstack(out)


def arc(cx, cy, rx, ry, a0, a1, n=None, egg=0.0):
    """Points on an ellipse; angles in degrees, 0 = right, 90 = down (y points down)."""
    if n is None:
        n = max(8, int(abs(a1 - a0) / 6))
    a = np.radians(np.linspace(a0, a1, n))
    s = np.sin(a)
    return np.stack([cx + rx * np.cos(a) * (1 + egg * s), cy + ry * s], 1)


def line(*pts):
    return catmull(pts, n=2) if len(pts) == 2 else np.vstack([catmull([pts[i], pts[i + 1]], n=2) for i in range(len(pts) - 1)])


def U(rng, a, b):
    return rng.uniform(a, b)


def jit(rng, pts, s):
    P = np.asarray(pts, float)
    return P + rng.normal(0, s, P.shape)


# ---------------------------------------------------------------- digit styles
# each style: f(rng) -> list of strokes, each an (n, 2) array of pen positions

def d0(rng):
    rx = U(rng, 0.27, 0.5)
    egg = U(rng, -0.25, 0.25)
    a0 = U(rng, -130, -50)
    sweep = U(rng, 345, 390) * (1 if rng.random() < 0.3 else -1)
    pts = arc(0, 0.5, rx, 0.5, a0, a0 + sweep, n=60, egg=egg)
    # gentle wobble of the loop: radial modulation with two low harmonics
    k = np.linspace(0, 2 * np.pi * sweep / 360, len(pts))
    m = 1 + U(rng, -0.06, 0.06) * np.cos(2 * k + U(rng, 0, 6.3)) + U(rng, -0.04, 0.04) * np.cos(3 * k + U(rng, 0, 6.3))
    c = np.array([0, 0.5])
    pts = c + (pts - c) * m[:, None]
    return [pts]


def d0_dented(rng):
    """0 whose closing end curls inward or leaves a small tail at the top."""
    rx = U(rng, 0.27, 0.45)
    a0 = U(rng, -100, -70)
    pts = arc(0, 0.5, rx, 0.5, a0, a0 - 360, n=60, egg=U(rng, -0.2, 0.2))
    tail = np.array([pts[-1], pts[-1] + [U(rng, -0.05, 0.12), U(rng, 0.08, 0.2)]])
    return [np.vstack([pts, catmull(tail, 4)])]


def d0_open(rng):
    """0 left open where the pen started: the end of the loop stops short of the start."""
    rx = U(rng, 0.27, 0.48)
    a0 = U(rng, -80, -20)                  # start at the top right
    pts = arc(0, 0.5, rx, 0.5, a0, a0 - U(rng, 310, 340), n=55, egg=U(rng, -0.2, 0.2))
    return [pts]


def d1(rng):
    c = U(rng, -0.06, 0.06)
    return [catmull([[0, 0], [c, 0.5], [0, 1]], 12)]


def d1_flag(rng):
    fx, fy = U(rng, -0.25, -0.1), U(rng, 0.1, 0.25)
    pts = [catmull([[fx, fy], [0, 0]], 4), catmull([[0, 0], [U(rng, -0.03, 0.03), 0.5], [0, 1]], 12)]
    strokes = [np.vstack(pts)]
    if rng.random() < 0.4:
        w = U(rng, 0.12, 0.25)
        strokes.append(line([-w, 1], [w, 1]))
    return strokes


def d2(rng):
    w = U(rng, 0.22, 0.32)
    top = U(rng, 0.0, 0.08)
    cx = U(rng, -0.05, 0.05)
    start = [-w + U(rng, 0, 0.08), U(rng, 0.12, 0.3)]
    wb = w * U(rng, 0.8, 1.25)             # the corner and base need not line up with the top hook
    pts = [start, [cx - 0.12, top + 0.04], [cx + 0.05, top], [cx + w - 0.05, top + 0.07], [cx + w, 0.24],
           [cx + w * 0.6, 0.45], [-0.05, 0.7 + U(rng, -0.05, 0.05)], [-wb, 0.98]]
    up = catmull(jit(rng, pts, 0.02), 10)
    corner = up[-1]
    bx = wb * U(rng, 0.8, 1.3)
    base = catmull([corner, [0.0, 0.97 + U(rng, -0.03, 0.02)], [bx, 0.98 + U(rng, -0.06, 0.03)]], 8)
    return [np.vstack([up, base])]


def d2_loop(rng):
    w = U(rng, 0.22, 0.32)
    start = [-w + U(rng, 0, 0.08), U(rng, 0.12, 0.3)]
    pts = [start, [-0.12, 0.04], [0.05, 0.0], [w - 0.05, 0.07], [w, 0.24], [w * 0.6, 0.45], [-0.05, 0.68]]
    lp = U(rng, 0.07, 0.19)  # loop radius
    lx = -w + lp
    pts += [[lx - lp * 0.3, 0.98 - 2 * lp], [lx - lp, 0.98 - lp * 0.6], [lx - lp * 0.2, 0.99],
            [lx + lp, 0.98 - lp * 0.9], [lx + lp * 0.2, 0.98 - lp * 1.6], [lx - lp * 0.6, 0.98 - lp * 0.6],
            [lx + lp * 0.8, 0.99], [0.1, 0.98], [w + U(rng, -0.05, 0.08), 0.95 + U(rng, -0.05, 0.03)]]
    return [catmull(jit(rng, pts, 0.015), 8)]


def d2_z(rng):
    w = U(rng, 0.22, 0.32)
    p = jit(rng, [[-w, 0.06], [w * 0.4, 0.0], [w, 0.08]], 0.02)
    top = catmull(p, 6)
    diag = catmull([top[-1], [0.0, 0.55], [-w, 1.0]], 8)
    base = line([-w, 1.0], [w + U(rng, -0.05, 0.08), 0.99])
    return [np.vstack([top, diag, base])]


def d3(rng):
    w = U(rng, 0.22, 0.3)
    mx = U(rng, -0.22, 0.05)
    my = U(rng, 0.4, 0.52)
    wt = w * U(rng, 0.7, 1.0)              # half width of the upper bump
    sx = (w - wt) * U(rng, 0.0, 1.0)       # a narrow upper bump sits to the right
    up = [[-wt + 0.02 + sx, U(rng, 0.0, 0.22)], [-0.05 + sx, 0.0], [wt * 0.7 + sx, 0.04], [wt * 0.95 + sx, my * 0.42],
          [wt * 0.5 + sx, my * 0.82], [mx, my]]
    lo = [[mx, my], [w * 0.6, my + 0.08], [w, my + (1 - my) * 0.5], [w * 0.7, 0.92], [0.0, 1.0], [-w, U(rng, 0.8, 0.92)]]
    a = catmull(jit(rng, up, 0.015), 8)
    b = catmull(jit(rng, lo, 0.015), 8)
    if rng.random() < 0.3:  # small inward curl at the middle
        b = np.vstack([[mx - 0.08, my + 0.01], b])
    return [np.vstack([a, b])]


def d3_flat(rng):
    w = U(rng, 0.22, 0.3)
    my = U(rng, 0.38, 0.48)
    top = line([-w, 0.0], [w, 0.0])
    diag = catmull([[w, 0.0], [w * 0.3, my * 0.6], [-0.05, my]], 6)
    lo = [[-0.05, my], [w * 0.6, my + 0.06], [w, my + (1 - my) * 0.5], [w * 0.7, 0.92], [0.0, 1.0], [-w, U(rng, 0.8, 0.92)]]
    return [np.vstack([top, diag, catmull(jit(rng, lo, 0.015), 8)])]


def d4_open(rng):
    by = U(rng, 0.5, 0.7)          # height of the cross bar
    sx = U(rng, 0.05, 0.2)         # x of the vertical stem
    lx = U(rng, -0.35, -0.22)      # left end of the bar
    tx = lx + U(rng, 0.05, 0.25)   # top of the left stroke
    a = np.vstack([catmull(jit(rng, [[tx, 0.0], [(tx + lx) / 2, by / 2], [lx, by]], 0.01), 6),
                   line([lx, by], [sx + U(rng, 0.05, 0.2), by + U(rng, -0.05, 0.03)])])
    st = U(rng, -0.02, 0.25)
    b = catmull([[sx + U(rng, -0.02, 0.05), st], [sx, 0.6], [sx - U(rng, 0, 0.05), 1.0]], 6)
    return [a, b]


def d4_closed(rng):
    by = U(rng, 0.55, 0.7)
    sx = U(rng, 0.05, 0.2)
    lx = U(rng, -0.35, -0.22)
    a = np.vstack([line([sx + U(rng, -0.02, 0.02), 0.0], [lx, by]),
                   line([lx, by], [sx + U(rng, 0.05, 0.2), by + U(rng, -0.04, 0.02)])])
    b = line([sx, 0.0], [sx - U(rng, 0, 0.05), 1.0])
    return [a, b]


def d4_cup(rng):
    """4 written like a cup with a long right stem (open top)."""
    by = U(rng, 0.45, 0.62)
    sx = U(rng, 0.12, 0.25)
    lx = U(rng, -0.3, -0.15)
    a = catmull(jit(rng, [[lx + U(rng, -0.05, 0.05), 0.0], [lx, by * 0.6], [lx + 0.1, by], [sx, by - U(rng, 0, 0.1)],
                          [sx + U(rng, 0.0, 0.1), by - U(rng, 0.0, 0.15)]], 0.01), 7)
    b = catmull([[sx + U(rng, -0.03, 0.05), U(rng, 0.0, 0.1)], [sx, 0.6], [sx - U(rng, 0, 0.06), 1.0]], 6)
    return [a, b]


def d5(rng):
    w = U(rng, 0.22, 0.3)
    vx = -w + U(rng, 0.0, 0.12)
    vy = U(rng, 0.3, 0.5)                      # where the vertical stroke turns into the belly
    vert = line([vx + U(rng, 0, 0.08), U(rng, 0.0, 0.05)], [vx, vy])
    by = U(rng, -0.02, 0.06)                   # belly bulge
    ex, ey = -w - U(rng, -0.08, 0.05), U(rng, 0.75, 0.95)   # end of the belly (lower left)
    belly = catmull(jit(rng, [[vx, vy], [0.0, vy - 0.05 + by], [w * 0.9, vy + 0.1 + by], [w + U(rng, -0.03, 0.05), vy + (1 - vy) * 0.55],
                              [w * 0.55, 0.93], [-0.05, 1.0], [ex, ey]], 0.02), 8)
    fl = U(rng, 0.0, 0.05)
    fx0 = vx + U(rng, 0, 0.06) + (U(rng, 0.03, 0.1) if rng.random() < 0.3 else 0)
    flag = line([fx0, fl], [fx0 + U(rng, 0.25, 0.6), fl + U(rng, -0.06, 0.03)])
    return [np.vstack([vert, belly]), flag]


def d5_round(rng):
    """5 whose vertical and belly flow as one curve (more like an S)."""
    w = U(rng, 0.22, 0.3)
    vy = U(rng, 0.35, 0.48)
    body = catmull(jit(rng, [[-w * 0.6, 0.02], [-w * 0.8, vy * 0.5], [-w * 0.7, vy], [0.05, vy - 0.08], [w, vy + 0.2],
                             [w * 0.8, 0.88], [0.0, 1.0], [-w, U(rng, 0.8, 0.95)]], 0.02), 8)
    flag = line([-w * 0.6, 0.02], [w + U(rng, -0.1, 0.08), U(rng, -0.02, 0.06)])
    return [body, flag]


def d5_bigbelly(rng):
    """short vertical, short flag, large round belly that curls back to the left."""
    w = U(rng, 0.24, 0.32)
    vx = -w * U(rng, 0.45, 0.8)
    vy = U(rng, 0.28, 0.42)
    a0 = U(rng, 220, 245)                       # the belly leaves the vertical at its upper-left
    rx = w * U(rng, 0.9, 1.1)
    ry = (1 - vy) / (1 - np.sin(np.radians(a0)))
    cx, cy = vx - rx * np.cos(np.radians(a0)), 1 - ry
    vert = line([vx + U(rng, 0, 0.06), 0.02], [vx, vy])
    belly = arc(cx, cy, rx, ry, a0, a0 + U(rng, 245, 280), n=40)
    flag = line([vx + U(rng, 0, 0.04), 0.02], [vx + U(rng, 0.25, 0.55), U(rng, -0.03, 0.06)])
    return [np.vstack([vert, belly]), flag]


def d6(rng):
    r = U(rng, 0.22, 0.32)          # loop radius
    ly = 1 - r * U(rng, 1.0, 1.1)   # loop centre height
    rx = r * U(rng, 0.9, 1.15)
    tx = U(rng, 0.05, 0.3)
    stem = catmull(jit(rng, [[tx, 0.0], [tx * 0.3 - 0.12, 0.18], [-rx * 0.95, ly - r * 0.5]], 0.015), 7)
    # loop: from its left side down, across the bottom, up the right side, over the top, back to the left
    lp = arc(0, ly, rx, r, 180, 180 - 330 + U(rng, -30, 20), n=45)
    return [np.vstack([stem, lp])]


def d6_straight(rng):
    r = U(rng, 0.21, 0.3)
    ly = 1 - r
    rx = r * U(rng, 0.9, 1.15)
    tx = U(rng, 0.0, 0.25)
    a = line([tx, 0.0], [-rx, ly])
    lp = arc(0, ly, rx, r, 180, 180 - 330 + U(rng, -30, 20), n=45)
    return [np.vstack([a, lp])]


def d7(rng):
    w = U(rng, 0.27, 0.4)
    ty = U(rng, 0.0, 0.03)
    left = [-w, ty + U(rng, -0.03, 0.03)]
    right = [w, ty]
    bx = U(rng, -0.15, 0.1)
    mid = [(w + bx) / 2 + U(rng, -0.05, 0.05), 0.5]
    strokes = []
    top = line(left, right)
    if rng.random() < 0.25:  # small hook at the start
        top = np.vstack([line([left[0] + U(rng, -0.03, 0.03), ty + U(rng, 0.08, 0.15)], left), top])
    stem = catmull([right, mid, [bx, 1.0]], 8)
    strokes.append(np.vstack([top, stem]))
    if rng.random() < 0.2:  # cross bar
        cy = U(rng, 0.45, 0.6)
        cx = (w + bx) / 2 + (bx - w) * (cy - 0.5) / 1.0
        strokes.append(line([cx - U(rng, 0.12, 0.2), cy], [cx + U(rng, 0.12, 0.2), cy + U(rng, -0.03, 0.03)]))
    return strokes


def d7_curved(rng):
    """7 with a curved top-right corner and a stem that bends."""
    w = U(rng, 0.27, 0.4)
    bx = U(rng, -0.15, 0.1)
    pts = [[-w, U(rng, 0.0, 0.08)], [0.0, 0.0], [w * 0.8, 0.02], [w, 0.1], [w * 0.6, 0.4], [(w + bx) / 2 - 0.05, 0.7], [bx, 1.0]]
    return [catmull(jit(rng, pts, 0.015), 8)]


def d8(rng):
    tr = U(rng, 0.1, 0.24)      # top loop half width
    br = U(rng, 0.13, 0.32)     # bottom loop half width
    my = U(rng, 0.38, 0.52)     # crossing height
    tx = U(rng, -0.03, 0.1)     # top loop sideways offset
    bx = U(rng, -0.1, 0.03)     # bottom loop sideways offset
    cx = (tx + bx) / 2
    pts = [[tx + tr * 0.8, 0.08], [tx, 0.0], [tx - tr, 0.1], [tx - tr * 0.8, my * 0.7], [cx, my], [bx + br, my + (1 - my) * 0.35],
           [bx + br * 0.9, 0.88], [bx, 1.0], [bx - br * 0.9, 0.88], [bx - br, my + (1 - my) * 0.35], [cx, my], [tx + tr * 0.8, my * 0.7],
           [tx + tr, 0.15], [tx + tr * 0.3, 0.0]]
    if rng.random() < 0.25:      # top loop left open
        pts = pts[:-2] + [[tx + tr * U(rng, 0.9, 1.2), U(rng, 0.1, 0.25)]]
    return [catmull(jit(rng, pts, 0.02), 8)]


def d8_circles(rng):
    tr = U(rng, 0.14, 0.24)
    br = U(rng, 0.2, 0.3)
    my = U(rng, 0.4, 0.5)
    a = arc(U(rng, -0.02, 0.08), my / 2, tr, my / 2, -90, 270, n=40)
    b = arc(U(rng, -0.06, 0.02), (1 + my) / 2, br, (1 - my) / 2, -90, 270, n=40)
    return [a, b]


def d9(rng):
    r = U(rng, 0.2, 0.3)
    ry = U(rng, 0.2, 0.27)
    sx = r * U(rng, 0.85, 1.05)
    loop = arc(0, ry, r, ry, U(rng, 10, 40), U(rng, 10, 40) - 360 - U(rng, 0, 30), n=45)
    stem = catmull([[sx, ry - U(rng, 0, 0.1)], [sx - U(rng, 0, 0.05), 0.6], [sx - U(rng, 0, 0.1), 1.0]], 8)
    return [np.vstack([loop, stem])]


def d9_curved(rng):
    r = U(rng, 0.2, 0.3)
    ry = U(rng, 0.2, 0.27)
    sx = r * U(rng, 0.85, 1.05)
    loop = arc(0, ry, r, ry, U(rng, 10, 40), U(rng, 10, 40) - 360 - U(rng, 0, 30), n=45)
    stem = catmull([[sx, ry], [sx, 0.65], [sx - 0.08, 0.92], [sx - U(rng, 0.2, 0.35), 1.0 - U(rng, 0, 0.08)]], 8)
    return [np.vstack([loop, stem])]


def d9_open(rng):
    """9 whose loop is not closed at the upper left."""
    r = U(rng, 0.2, 0.3)
    ry = U(rng, 0.2, 0.27)
    sx = r * U(rng, 0.85, 1.05)
    loop = arc(0, ry, r, ry, U(rng, 120, 200), U(rng, -10, 20) - 360 + 180, n=40)[::-1]
    stem = catmull([[sx, ry - U(rng, 0, 0.1)], [sx - U(rng, 0, 0.05), 0.6], [sx - U(rng, 0, 0.12), 1.0]], 8)
    return [np.vstack([loop, stem])]


STYLES = {
    0: [(d0, 0.6), (d0_dented, 0.2), (d0_open, 0.2)],
    1: [(d1, 0.7), (d1_flag, 0.3)],
    2: [(d2, 0.55), (d2_loop, 0.35), (d2_z, 0.1)],
    3: [(d3, 0.8), (d3_flat, 0.2)],
    4: [(d4_open, 0.45), (d4_closed, 0.25), (d4_cup, 0.3)],
    5: [(d5, 0.5), (d5_round, 0.25), (d5_bigbelly, 0.25)],
    6: [(d6, 0.7), (d6_straight, 0.3)],
    7: [(d7, 0.75), (d7_curved, 0.25)],
    8: [(d8, 0.75), (d8_circles, 0.25)],
    9: [(d9, 0.55), (d9_curved, 0.3), (d9_open, 0.15)],
}


# ---------------------------------------------------------------- rendering

def densify(stroke, step=0.25):
    """Resample a polyline so that consecutive points are at most `step` apart (in its own units)."""
    out = [stroke[0]]
    for a, b in zip(stroke[:-1], stroke[1:]):
        L = np.hypot(*(b - a))
        k = max(1, int(np.ceil(L / step)))
        t = np.arange(1, k + 1)[:, None] / k
        out.append(a + (b - a) * t)
    return np.vstack([np.atleast_2d(o) for o in out])


YY, XX = np.mgrid[0:28, 0:28].astype(float)
PIX = np.stack([XX.ravel(), YY.ravel()], 1)


def render_px(strokes, r):
    """strokes in pixel coordinates; ink = coverage-like ramp of the distance to the pen path."""
    d = np.full(784, 1e9)
    for s in strokes:
        A, B = s[:-1], s[1:]
        AB = B - A
        L2 = (AB ** 2).sum(1)
        L2[L2 == 0] = 1e-12
        AP = PIX[:, None, :] - A[None]
        t = np.clip((AP * AB[None]).sum(2) / L2[None], 0, 1)
        q = A[None] + t[..., None] * AB[None]
        dd = np.sqrt(((PIX[:, None, :] - q) ** 2).sum(2)).min(1)
        d = np.minimum(d, dd)
    return np.clip(r + 0.5 - d, 0, 1).reshape(28, 28)


def make(strokes, shear, rot, xscale, r):
    """Apply the global writing transform, then MNIST-style normalisation: fit into 20 x 20 keeping proportions,
    then centre of mass at the centre of the frame."""
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    T = []
    for st in strokes:
        p = np.asarray(st, float).copy()
        p[:, 0] *= xscale
        p[:, 0] = p[:, 0] - shear * (p[:, 1] - 0.5)
        p = p @ np.array([[c, s], [-s, c]])
        T.append(p)
    allp = np.vstack(T)
    lo, hi = allp.min(0), allp.max(0)
    size = max(hi - lo)
    sc = (20.0 - 2 * r) / size
    ctr = (lo + hi) / 2
    T = [densify((p - ctr) * sc + 13.5, 0.4) for p in T]
    img = render_px(T, r)
    m = img.sum()
    cx, cy = (img * XX).sum() / m, (img * YY).sum() / m
    T = [p + [13.5 - cx, 13.5 - cy] for p in T]
    return render_px(T, r)


def sample(rng, digit):
    styles = STYLES[digit]
    w = np.array([p for _, p in styles]); w = w / w.sum()
    f = styles[rng.choice(len(styles), p=w)][0]
    strokes = f(rng)
    if digit == 1:
        shear = U(rng, -0.35, 0.75)
    else:
        shear = U(rng, -0.25, 0.65)
    rot = U(rng, -14, 14)
    xscale = U(rng, 0.7, 1.35)
    r = U(rng, 0.7, 2.0)
    return make(strokes, shear, rot, xscale, r)


# ---------------------------------------------------------------- the network

def gauss1d(sigma, radius):
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return x, g / g.sum()


NDIR = 8           # gradient directions, 45 degrees apart (with polarity)
SIG_GRAD = 1.0     # Gaussian scale of the gradient filters, in pixels
SIG_BLUR = 0.8     # blur of the pooled maps, in 14 x 14 cells
TPC = 3000         # templates per class
FILL_STEPS = 44    # flood-fill steps from the border (one pixel, 4-connected, per step)
K_INK = 3.0        # ink above 1/3 blocks the fill
HOLE_GAIN = 1.6    # weight of the hole agreement term: -HOLE_GAIN * |h_in - h_template|^2
RAY_STEPS = 22     # steps of ray propagation (one pixel per step) for the "ink seen in this direction" maps
CONC_WEIGHT = 1.0  # weight of the concavity-code correlation relative to the orientation correlation
# directions: (dy, dx) offset of the pixel the ray comes from
RAYS = [(0, -1), (0, 1), (-1, 0), (1, 0)]   # ink seen to the left, right, above, below
# concavity codes: background pixels with ink seen (+1) / not seen (-1) / either (0) in directions L, R, U, D
CODES = [(-1, 1, 1, 1),    # bay open to the left
         (1, -1, 1, 1),    # bay open to the right
         (1, 1, -1, 1),    # bay open upward
         (1, 1, 1, -1),    # bay open downward
         (1, 1, 1, 1),     # enclosed in all four directions
         (-1, 1, 1, -1),   # open to the left and down
         (1, -1, 1, -1),   # open to the right and down
         (-1, 1, -1, 1),   # open to the left and up
         (1, -1, -1, 1)]   # open to the right and up
NCODE = len(CODES)


def layer_weights():
    W = {}
    # orientation branch ------------------------------------------------------------
    # 1. directional derivatives of a Gaussian: channel k responds to brightness increasing along u_k
    x, g = gauss1d(SIG_GRAD, 3)
    gx = np.outer(g, x * g) / (SIG_GRAD ** 2)           # d/dx of the smoothed image (correlation form)
    gy = np.outer(x * g, g) / (SIG_GRAD ** 2)
    step = (np.arange(-3, 4) >= 0).astype(float)        # a unit step edge gives a peak response of 1
    peak = (gx * step[None, :]).sum()
    gx, gy = gx / peak, gy / peak
    th = np.arange(NDIR) * 2 * np.pi / NDIR
    W["grad.W"] = np.stack([np.cos(t) * gx + np.sin(t) * gy for t in th])[:, None]
    W["grad.b"] = np.zeros(NDIR)
    # 2. sharpen orientation tuning: s_k = c_k - (c_{k+2} + c_{k-2})  (the two perpendicular directions)
    S = np.zeros((NDIR, NDIR, 1, 1))
    for k in range(NDIR):
        S[k, k] = 1.0
        S[k, (k + 2) % NDIR] = -1.0
        S[k, (k - 2) % NDIR] = -1.0
    W["sharp.W"] = S
    W["sharp.b"] = np.zeros(NDIR)
    # 3. blur of the pooled maps, channel by channel
    _, b = gauss1d(SIG_BLUR, 2)
    B = np.zeros((NDIR, NDIR, 5, 5))
    for k in range(NDIR):
        B[k, k] = np.outer(b, b)
    W["blur.W"] = B
    W["blur.b"] = np.zeros(NDIR)
    # hole branch: flood fill of the background from the frame border ---------------------
    # tensor channels: 0 = reached by the fill, 1 = ink (carried along unchanged)
    W["h_ones.W"] = np.array([0.0, 1.0]).reshape(2, 1, 1, 1)
    W["h_ones.b"] = np.array([1.0, 0.0])
    sd = np.zeros((2, 2, 3, 3)); sd[0, 0] = -1.0; sd[1, 1, 1, 1] = 1.0
    W["h_seed.W"] = sd                                   # channel 0: 9 - number of in-frame neighbours (> 0 only on the border)
    W["h_seed.b"] = np.array([9.0, 0.0])
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], float)
    sa = np.zeros((2, 2, 3, 3)); sa[0, 0] = -cross; sa[1, 1, 1, 1] = 1.0
    sc = np.zeros((2, 2, 1, 1)); sc[0, 0] = -1.0; sc[0, 1] = -K_INK; sc[1, 1] = 1.0
    for t in range(FILL_STEPS):
        W[f"f{t}a.W"], W[f"f{t}a.b"] = sa, np.array([1.0, 0.0])      # 1 - (reached in the 4-neighbourhood)
        W[f"f{t}c.W"], W[f"f{t}c.b"] = sc, np.array([1.0, 0.0])      # reached' = 1 - not_near_reached - K*ink
    W["hole.W"] = np.array([-2.0, -K_INK]).reshape(1, 2, 1, 1)       # hole = background never reached
    W["hole.b"] = np.array([1.0])
    W["hblur.W"] = np.outer(b, b).reshape(1, 1, 5, 5)
    W["hblur.b"] = np.zeros(1)
    # concavity branch: channels 2d, 2d+1 hold y_d and relu(y_d - 1) so that seen_d = min(1, y_d) = ch(2d) - ch(2d+1);
    # channel 8 carries the ink.  One step: y_d(p) = seen_d(p - off_d) + K * ink(p - off_d)
    ri = np.zeros((9, 1, 1, 1)); ri[8, 0] = 1.0
    W["r_init.W"], W["r_init.b"] = ri, np.zeros(9)
    rs = np.zeros((9, 9, 3, 3)); rb = np.zeros(9)
    for d, (oy, ox) in enumerate(RAYS):
        for o in (2 * d, 2 * d + 1):
            rs[o, 2 * d, 1 + oy, 1 + ox] = 1.0
            rs[o, 2 * d + 1, 1 + oy, 1 + ox] = -1.0
            rs[o, 8, 1 + oy, 1 + ox] = K_INK
        rb[2 * d + 1] = -1.0
    rs[8, 8, 1, 1] = 1.0
    for t in range(RAY_STEPS):
        W[f"r{t}.W"], W[f"r{t}.b"] = rs, rb
    # codes: fuzzy AND (relu(sum of terms - (n - 1))) of background (1 - K*ink) and seen / not seen in each direction
    cw = np.zeros((NCODE, 9, 1, 1)); cb = np.zeros(NCODE)
    for c, code in enumerate(CODES):
        n = 1
        cw[c, 8] = -K_INK; cb[c] = 1.0
        for d, sgn in enumerate(code):
            if sgn == 0:
                continue
            n += 1
            cw[c, 2 * d] = sgn; cw[c, 2 * d + 1] = -sgn
            if sgn < 0:
                cb[c] += 1.0
        cb[c] -= n - 1
    W["codes.W"], W["codes.b"] = cw, cb
    _, b2 = gauss1d(1.0, 2)
    CB = np.zeros((NCODE, NCODE, 5, 5))
    for k in range(NCODE):
        CB[k, k] = np.outer(b2, b2)
    W["cblur.W"], W["cblur.b"] = CB, np.zeros(NCODE)
    return W


FEAT_LAYERS = [
    {"name": "grad", "op": "conv", "inputs": ["image"], "out": NDIR, "kernel": [7, 7], "stride": 1, "padding": 3},
    {"name": "grad_r", "op": "relu", "inputs": ["grad"]},
    {"name": "sharp", "op": "conv", "inputs": ["grad_r"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "sharp_r", "op": "relu", "inputs": ["sharp"]},
    {"name": "pool", "op": "maxpool", "inputs": ["sharp_r"], "size": [2, 2], "stride": [2, 2]},
    {"name": "blur", "op": "conv", "inputs": ["pool"], "out": NDIR, "kernel": [5, 5], "stride": 1, "padding": 2},
    {"name": "h_ones", "op": "conv", "inputs": ["image"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "h_seed", "op": "conv", "inputs": ["h_ones"], "out": 2, "kernel": [3, 3], "stride": 1, "padding": 1},
]
prev = "h_seed"
for _t in range(FILL_STEPS):
    FEAT_LAYERS += [
        {"name": f"f{_t}a", "op": "conv", "inputs": [prev], "out": 2, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": f"f{_t}b", "op": "relu", "inputs": [f"f{_t}a"]},
        {"name": f"f{_t}c", "op": "conv", "inputs": [f"f{_t}b"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": f"f{_t}d", "op": "relu", "inputs": [f"f{_t}c"]},
    ]
    prev = f"f{_t}d"
FEAT_LAYERS += [
    {"name": "hole", "op": "conv", "inputs": [prev], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "hole_r", "op": "relu", "inputs": ["hole"]},
    {"name": "hpool", "op": "avgpool", "inputs": ["hole_r"], "size": [2, 2], "stride": [2, 2]},
    {"name": "hblur", "op": "conv", "inputs": ["hpool"], "out": 1, "kernel": [5, 5], "stride": 1, "padding": 2},
]
FEAT_LAYERS += [{"name": "r_init", "op": "conv", "inputs": ["image"], "out": 9, "kernel": [1, 1], "stride": 1, "padding": 0}]
prev = "r_init"
for _t in range(RAY_STEPS):
    FEAT_LAYERS += [
        {"name": f"r{_t}", "op": "conv", "inputs": [prev], "out": 9, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": f"r{_t}r", "op": "relu", "inputs": [f"r{_t}"]},
    ]
    prev = f"r{_t}r"
FEAT_LAYERS += [
    {"name": "codes", "op": "conv", "inputs": [prev], "out": NCODE, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "codes_r", "op": "relu", "inputs": ["codes"]},
    {"name": "cpool", "op": "avgpool", "inputs": ["codes_r"], "size": [2, 2], "stride": [2, 2]},
    {"name": "cblur", "op": "conv", "inputs": ["cpool"], "out": NCODE, "kernel": [5, 5], "stride": 1, "padding": 2},
]
FEAT_DIM = NDIR * 14 * 14
HOLE_DIM = 14 * 14
CONC_DIM = NCODE * 14 * 14


def head_layers(T):
    return [
        {"name": "flat", "op": "reshape", "inputs": ["blur"], "shape": [1, 1, FEAT_DIM]},
        {"name": "unit", "op": "norm", "inputs": ["flat"]},
        {"name": "match", "op": "dense", "inputs": ["unit"], "out": T},
        {"name": "hmatch", "op": "dense", "inputs": ["hblur"], "out": T},
        {"name": "cflat", "op": "reshape", "inputs": ["cblur"], "shape": [1, 1, CONC_DIM]},
        {"name": "cunit", "op": "norm", "inputs": ["cflat"]},
        {"name": "cmatch", "op": "dense", "inputs": ["cunit"], "out": T},
        {"name": "total", "op": "add", "inputs": ["match", "hmatch", "cmatch"]},
        {"name": "byclass", "op": "reshape", "inputs": ["total"], "shape": [1, 10, T // 10]},
        {"name": "best", "op": "maxpool", "inputs": ["byclass"], "size": [1, T // 10], "stride": [1, T // 10]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]},
    ]


# numpy forward pass (same arithmetic as the scorer's interpreter)
def _conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def _pool(x, size, stride, red):
    n, c, h, w = x.shape
    (sh, sw), (th, tw) = size, stride
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if red == "max" else win.mean(0)


def forward(layers, W, imgs, keep):
    # float64 arithmetic on the stored (float32-rounded) weights: reproducible whatever the BLAS threading
    v = {"image": imgs.astype(np.float32).astype(np.float64)}
    W32 = {k: a.astype(np.float32).astype(np.float64) for k, a in W.items()}
    for L in layers:
        ins = [v[i] for i in L["inputs"]]
        x, op, nm = ins[0], L["op"], L["name"]
        if op == "conv":
            v[nm] = _conv(x, W32[nm + ".W"], W32[nm + ".b"], L["stride"], L["padding"])
        elif op == "relu":
            v[nm] = np.maximum(x, 0)
        elif op in ("maxpool", "avgpool"):
            v[nm] = _pool(x, L["size"], L["stride"], op[:3])
        elif op == "add":
            v[nm] = sum(ins)
        elif op == "reshape":
            v[nm] = x.reshape((len(x),) + tuple(L["shape"]))
        elif op == "dense":
            v[nm] = x.reshape(len(x), -1) @ W32[nm + ".W"].T + W32[nm + ".b"]
        elif op == "norm":
            mu = x.mean(axis=(2, 3), keepdims=True)
            v[nm] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        # drop what is no longer needed
        for k in list(v):
            if k not in keep and k != nm and k != "image" and not any(k in M["inputs"] for M in layers[layers.index(L) + 1:]):
                del v[k]
    return {k: v[k] for k in keep}


def templates(seed=12345):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((10 * TPC, 1, 28, 28))
    for k in range(10):
        for j in range(TPC):
            imgs[k * TPC + j, 0] = sample(rng, k)
    return imgs


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    W = layer_weights()
    imgs = templates()
    F, H, C = [], [], []
    for i in range(0, len(imgs), 500):
        o = forward(FEAT_LAYERS, W, imgs[i:i + 500], ["blur", "hblur", "cblur"])
        F.append(o["blur"].reshape(len(o["blur"]), -1).astype(np.float64))
        H.append(o["hblur"].reshape(len(o["hblur"]), -1).astype(np.float64))
        C.append(o["cblur"].reshape(len(o["cblur"]), -1).astype(np.float64))
    F, H, C = np.concatenate(F), np.concatenate(H), np.concatenate(C)
    C = C - C.mean(1, keepdims=True)
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    # the input's code vector has length sqrt(CONC_DIM) after the norm layer; scale so that a perfect code match is
    # worth CONC_WEIGHT times a perfect orientation match (sqrt(FEAT_DIM))
    W["cmatch.W"] = C * CONC_WEIGHT * np.sqrt(FEAT_DIM / CONC_DIM)
    W["cmatch.b"] = np.zeros(len(C))
    # each template is normalised by itself alone: zero mean, unit length
    F = F - F.mean(1, keepdims=True)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    T = len(F)
    W["match.W"] = F
    W["match.b"] = np.zeros(T)
    # hole agreement: -g |h - h_t|^2 = 2g h.h_t - g |h_t|^2 - g |h|^2 (the last term is the same for every template)
    W["hmatch.W"] = 2 * HOLE_GAIN * H
    W["hmatch.b"] = -HOLE_GAIN * (H ** 2).sum(1)
    net = {"layers": FEAT_LAYERS + head_layers(T)}
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
    return imgs


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
