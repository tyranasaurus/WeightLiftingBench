"""Builds a hand-designed traffic-sign classifier (no training data is read).

Every class is drawn from geometric primitives (discs, rings, rounded triangles, strokes, polygons)
typed from general knowledge of German road-sign designs.  Each drawing is rendered at several sizes,
aspect ratios and pictogram scales; every rendering becomes one correlation template, normalised by
itself alone.  The network is: per-channel norm of the image -> one 32x32 conv holding all templates
(padding gives +-3 px shifts) -> max over shifts -> max over the variants of each class.

usage: python3 build.py OUT_DIR
"""
import json
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------------------------
# colours (typical camera RGB of sign paints, chosen by hand)
WHITE = (0.85, 0.85, 0.85)
RED = (0.70, 0.12, 0.15)
BLACK = (0.08, 0.08, 0.08)
BLUE = (0.10, 0.25, 0.62)
YELLOW = (0.95, 0.66, 0.10)
GRAY = (0.40, 0.40, 0.40)
DGRAY = (0.25, 0.25, 0.25)
LRED = (0.85, 0.12, 0.10)
AMBER = (0.95, 0.72, 0.10)
GREEN = (0.10, 0.62, 0.42)

# ----------------------------------------------------------------------------------------------
# shapes: (bbox, fn) with bbox = (u0, u1, v0, v1) and fn(U, V) -> bool mask.  u right, v down.


def disc(cx, cy, r):
    return (cx - r, cx + r, cy - r, cy + r), lambda U, V: (U - cx) ** 2 + (V - cy) ** 2 <= r * r


def ellipse(cx, cy, rx, ry):
    return (cx - rx, cx + rx, cy - ry, cy + ry), lambda U, V: ((U - cx) / rx) ** 2 + ((V - cy) / ry) ** 2 <= 1


def ring(cx, cy, r0, r1):
    def f(U, V):
        d = (U - cx) ** 2 + (V - cy) ** 2
        return (d <= r1 * r1) & (d >= r0 * r0)
    return (cx - r1, cx + r1, cy - r1, cy + r1), f


def rect(u0, u1, v0, v1):
    return (u0, u1, v0, v1), lambda U, V: (U >= u0) & (U <= u1) & (V >= v0) & (V <= v1)


def poly(pts):
    P = np.asarray(pts, float)

    def f(U, V):
        inside = np.zeros(U.shape, bool)
        n = len(P)
        for k in range(n):
            x1, y1 = P[k]
            x2, y2 = P[(k + 1) % n]
            if y1 == y2:
                continue
            c = (y1 > V) != (y2 > V)
            xi = x1 + (V - y1) * (x2 - x1) / (y2 - y1)
            inside ^= c & (U < xi)
        return inside
    return (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()), f


def stroke(pts, w):
    P = np.asarray(pts, float)
    r2 = (w / 2) ** 2

    def f(U, V):
        m = np.zeros(U.shape, bool)
        if len(P) == 1:
            return (U - P[0, 0]) ** 2 + (V - P[0, 1]) ** 2 <= r2
        for (px, py), (qx, qy) in zip(P[:-1], P[1:]):
            dx, dy = qx - px, qy - py
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else np.clip(((U - px) * dx + (V - py) * dy) / L2, 0, 1)
            m |= (U - px - t * dx) ** 2 + (V - py - t * dy) ** 2 <= r2
        return m
    h = w / 2
    return (P[:, 0].min() - h, P[:, 0].max() + h, P[:, 1].min() - h, P[:, 1].max() + h), f


def union(*shapes):
    bb = np.array([s[0] for s in shapes])

    def f(U, V):
        m = np.zeros(U.shape, bool)
        for _, g in shapes:
            m |= g(U, V)
        return m
    return (bb[:, 0].min(), bb[:, 1].max(), bb[:, 2].min(), bb[:, 3].max()), f


def shift(shape, du, dv):
    (u0, u1, v0, v1), f = shape
    return (u0 + du, u1 + du, v0 + dv, v1 + dv), lambda U, V: f(U - du, V - dv)


def mirror(shape):
    (u0, u1, v0, v1), f = shape
    return (-u1, -u0, v0, v1), lambda U, V: f(-U, V)


def arc(cx, cy, rx, ry, a0, a1, n=None):
    """points on an ellipse arc, angles in degrees, y down (90 = bottom, 270 = top)."""
    if n is None:
        n = max(4, int(abs(a1 - a0) / 7.5) + 1)
    a = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in a]


def superellipse(cx, cy, rx, ry, p, n=64):
    t = np.linspace(0, 2 * np.pi, n + 1)
    c, s = np.cos(t), np.sin(t)
    return [(cx + rx * np.sign(a) * abs(a) ** (2 / p), cy + ry * np.sign(b) * abs(b) ** (2 / p)) for a, b in zip(c, s)]


def sd_convex(U, V, P):
    """exact signed distance to a convex polygon (negative inside)."""
    P = np.asarray(P, float)
    g = P.mean(0)
    n = len(P)
    smax = np.full(U.shape, -np.inf)
    dmin = np.full(U.shape, np.inf)
    for k in range(n):
        (px, py), (qx, qy) = P[k], P[(k + 1) % n]
        dx, dy = qx - px, qy - py
        L = np.hypot(dx, dy)
        nx, ny = dy / L, -dx / L
        if nx * (g[0] - px) + ny * (g[1] - py) > 0:
            nx, ny = -nx, -ny
        smax = np.maximum(smax, (U - px) * nx + (V - py) * ny)
        t = np.clip(((U - px) * dx + (V - py) * dy) / (L * L), 0, 1)
        dmin = np.minimum(dmin, np.hypot(U - px - t * dx, V - py - t * dy))
    return np.where(smax <= 0, smax, dmin)


def tri_vertices(gx, gy, rho, up=True):
    s = 1 if up else -1
    return [(gx, gy - s * 2 * rho), (gx + np.sqrt(3) * rho, gy + s * rho), (gx - np.sqrt(3) * rho, gy + s * rho)]


def rounded_tri(gx, gy, rho, rc, inset, up=True):
    P = tri_vertices(gx, gy, rho - rc, up)
    return (-0.6, 0.6, -0.6, 0.6), lambda U, V: sd_convex(U, V, P) - rc <= -inset


def octagon(apothem, inset):
    R = apothem / np.cos(np.pi / 8)
    P = [(R * np.cos(np.pi / 8 + k * np.pi / 4), R * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)]
    return (-0.6, 0.6, -0.6, 0.6), lambda U, V: sd_convex(U, V, P) <= -inset


def diamond(half, inset):
    return (-0.6, 0.6, -0.6, 0.6), lambda U, V: (np.abs(U) + np.abs(V) - half) / np.sqrt(2) <= -inset


# ----------------------------------------------------------------------------------------------
# glyphs: centre lines in a box of height 1 (y down), width gw; stroke width sw (units of height)


def glyph(ch, gw, sw):
    x0, x1, y0, y1 = sw / 2, gw - sw / 2, sw / 2, 1 - sw / 2
    xm, rx = gw / 2, (gw - sw) / 2
    if ch == '0':
        return [superellipse(xm, 0.5, rx, 0.5 - y0, 2.5)]
    if ch == '1':
        xs = gw - sw / 2
        return [[(xs, y0), (xs, y1)], [(xs, y0), (xs - 0.30, y0 + 0.24)]]
    if ch == '2':
        return [arc(xm, 0.29, rx, 0.29 - y0, 195, 390) + [(x0, y1), (x1, y1)]]
    if ch == '3':
        yj = 0.46
        return [arc(xm, (y0 + yj) / 2, 0.9 * rx, (yj - y0) / 2, 200, 450) + arc(xm, (yj + y1) / 2, rx, (y1 - yj) / 2, 270, 520)]
    if ch == '5':
        cy = 0.68
        return [[(x1, y0), (x0 + 0.02, y0), (x0 + 0.01, 0.47)] + arc(xm, cy, rx, y1 - cy, 215, 520)]
    if ch == '6':
        cy = 0.665
        return [superellipse(xm, cy, rx, y1 - cy, 2.0), arc(xm, 0.62, rx, 0.62 - y0, 180, 285)]
    if ch == '7':
        return [[(x0, y0), (x1, y0), (x0 + 0.3 * (x1 - x0), y1)]]
    if ch == '8':
        yj = 0.465
        return [superellipse(xm, (y0 + yj) / 2, 0.88 * rx, (yj - y0) / 2, 2.0),
                superellipse(xm, (yj + y1) / 2, rx, (y1 - yj) / 2, 2.0)]
    if ch == 'S':
        yj = 0.48
        return [arc(xm, (y0 + yj) / 2, rx, (yj - y0) / 2, 340, 90) + arc(xm, (yj + y1) / 2, rx, (y1 - yj) / 2, 270, 520)]
    if ch == 'T':
        return [[(0, y0), (gw, y0)], [(xm, y0), (xm, y1)]]
    if ch == 'O':
        return [superellipse(xm, 0.5, rx, 0.5 - y0, 2.2)]
    if ch == 'P':
        yb = 0.56
        rbx, rby = 0.5 * (x1 - x0), (yb - y0) / 2
        return [[(x0, y1), (x0, y0), (x1 - rbx, y0)] + arc(x1 - rbx, y0 + rby, rbx, rby, 270, 450) + [(x0, yb)]]
    raise ValueError(ch)


def text(s, cx, cy, h, gw, sw, gap, one=0.55):
    adv = [gw * one if ch == '1' else gw for ch in s]
    total = sum(adv) + gap * (len(s) - 1)
    x = -total / 2
    shapes = []
    for ch, a in zip(s, adv):
        g = glyph(ch, a, sw) if ch != '1' else glyph('1', a, sw)
        for line in g:
            pts = [(cx + (x + px) * h, cy + (py - 0.5) * h) for px, py in line]
            shapes.append(stroke(pts, sw * h))
        x += a + gap
    return union(*shapes)


# ----------------------------------------------------------------------------------------------
# pictograms


def car_rear(uc, vb, s):
    """car seen from behind, width s, bottom of wheels at vb."""
    vc = vb - 0.36 * s
    return union(poly([(uc - 0.30 * s, vc - 0.36 * s), (uc + 0.30 * s, vc - 0.36 * s), (uc + 0.42 * s, vc - 0.08 * s), (uc - 0.42 * s, vc - 0.08 * s)]),
                 rect(uc - 0.5 * s, uc + 0.5 * s, vc - 0.10 * s, vc + 0.20 * s),
                 rect(uc - 0.46 * s, uc - 0.24 * s, vc + 0.18 * s, vc + 0.36 * s),
                 rect(uc + 0.24 * s, uc + 0.46 * s, vc + 0.18 * s, vc + 0.36 * s))


def truck_rear(uc, vb, s):
    vc = vb - 0.36 * s
    return union(rect(uc - 0.5 * s, uc + 0.5 * s, vc - 0.62 * s, vc + 0.20 * s),
                 rect(uc - 0.46 * s, uc - 0.22 * s, vc + 0.18 * s, vc + 0.36 * s),
                 rect(uc + 0.22 * s, uc + 0.46 * s, vc + 0.18 * s, vc + 0.36 * s))


def truck_side():
    """lorry in side view facing left, centred."""
    return union(rect(-0.10, 0.26, -0.15, 0.06),
                 poly([(-0.26, 0.07), (-0.26, -0.03), (-0.22, -0.09), (-0.12, -0.09), (-0.12, 0.07)]),
                 rect(-0.26, 0.26, 0.04, 0.08),
                 disc(-0.19, 0.10, 0.042), disc(0.09, 0.10, 0.042), disc(0.18, 0.10, 0.042))


def end_band():
    d = np.array([1.0, -1.0]) / np.sqrt(2)
    n = np.array([1.0, 1.0]) / np.sqrt(2)
    lines = []
    for o in (-0.08, -0.04, 0.0, 0.04, 0.08):
        p, q = o * n - 0.6 * d, o * n + 0.6 * d
        lines.append(stroke([tuple(p), tuple(q)], 0.018))
    return union(*lines)


def person(ux, vh, s=1.0):
    """bold walking figure, head centre at (ux, vh), size factor s."""
    neck = (ux, vh + 0.05 * s)
    hip = (ux - 0.01 * s, vh + 0.19 * s)
    return union(disc(ux, vh, 0.04 * s),
                 stroke([neck, hip], 0.075 * s),
                 stroke([hip, (hip[0] - 0.07 * s, hip[1] + 0.08 * s), (hip[0] - 0.08 * s, hip[1] + 0.17 * s)], 0.05 * s),
                 stroke([hip, (hip[0] + 0.04 * s, hip[1] + 0.09 * s), (hip[0] + 0.08 * s, hip[1] + 0.17 * s)], 0.05 * s),
                 stroke([(neck[0], neck[1] + 0.03 * s), (neck[0] - 0.07 * s, neck[1] + 0.10 * s), (neck[0] - 0.08 * s, neck[1] + 0.15 * s)], 0.04 * s),
                 stroke([(neck[0], neck[1] + 0.03 * s), (neck[0] + 0.06 * s, neck[1] + 0.09 * s), (neck[0] + 0.09 * s, neck[1] + 0.13 * s)], 0.04 * s))


def arrow_head(tip, direction, length, half):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    t = np.asarray(tip, float)
    b = t - length * d
    return poly([tuple(t), tuple(b + half * n), tuple(b - half * n)])


# triangle sign geometry (side 1, corner radius RC): bbox centred at the origin
RHO = 1 / (2 * np.sqrt(3))
RC = 0.05
TRI_W = 2 * (np.sqrt(3) * (RHO - RC) + RC)
TRI_H = 3 * RHO - RC
TRI_G = (RHO - RC) / 2          # centroid sits this far below the bbox centre
TRI_STRETCH = 1.10              # crops are roughly square, so triangles appear stretched vertically


def picto(k):
    """bold black pictogram of warning sign class k (coordinates: triangle side 1, origin at bbox centre;
    the inner white triangle spans v from -0.23 to 0.29 and is 0.60 wide at its base)."""
    if k == 11:      # priority at next junction: broad upright bar with pointed top, crossed by a bar
        return union(poly([(-0.055, 0.25), (0.055, 0.25), (0.045, -0.06), (0.0, -0.13), (-0.045, -0.06)]),
                     rect(-0.13, 0.13, 0.01, 0.08))
    if k == 18:      # "!"
        return union(poly([(-0.042, -0.13), (0.042, -0.13), (0.026, 0.13), (-0.026, 0.13)]), rect(-0.035, 0.035, 0.17, 0.24))
    if k == 19:      # bend to the left: thick line up from the bottom, curving to the left at the top
        return union(stroke([(0.06, 0.27), (0.06, 0.06)] + arc(-0.03, 0.06, 0.09, 0.10, 0, -90) + [(-0.10, -0.04)], 0.085),
                     arrow_head((-0.16, -0.04), (-1, 0), 0.07, 0.065))
    if k == 20:
        return mirror(picto(19))
    if k == 21:      # double bend, first to the left: stepped thick line, lower part right of centre
        return union(stroke([(0.05, 0.27), (0.05, 0.13), (-0.05, 0.05), (-0.05, -0.10)], 0.08),
                     arrow_head((-0.05, -0.17), (0, -1), 0.07, 0.065))
    if k == 22:      # bumpy road
        return union(rect(-0.22, 0.22, 0.20, 0.26), poly(arc(-0.10, 0.205, 0.075, 0.07, 180, 360)),
                     poly(arc(0.10, 0.205, 0.075, 0.07, 180, 360)))
    if k == 23:      # slippery road: car above two wide skid marks
        car = union(rect(-0.11, 0.11, -0.05, 0.04), poly([(-0.08, -0.05), (0.08, -0.05), (0.055, -0.11), (-0.055, -0.11)]),
                    rect(-0.11, -0.065, 0.04, 0.075), rect(0.065, 0.11, 0.04, 0.075))
        tr1 = stroke([(-0.06, 0.10), (-0.13, 0.15), (-0.06, 0.20), (-0.14, 0.26)], 0.035)
        tr2 = stroke([(0.07, 0.10), (0.0, 0.15), (0.08, 0.20), (0.0, 0.26)], 0.035)
        return union(car, tr1, tr2)
    if k == 24:      # road narrows on the right
        return union(stroke([(-0.075, -0.03), (-0.075, 0.27)], 0.055),
                     stroke([(0.10, 0.27), (0.10, 0.12), (0.04, 0.04), (0.04, -0.07)], 0.055))
    if k == 25:      # road works: worker leaning left with a shovel, heap on the right
        man = union(disc(0.0, -0.10, 0.04),
                    stroke([(-0.01, -0.06), (-0.05, 0.08)], 0.075),
                    stroke([(-0.05, 0.08), (-0.11, 0.16), (-0.12, 0.25)], 0.05),
                    stroke([(-0.05, 0.08), (0.0, 0.16), (0.02, 0.25)], 0.05),
                    stroke([(-0.02, -0.03), (-0.10, 0.08), (-0.16, 0.21)], 0.035),
                    poly([(-0.19, 0.20), (-0.14, 0.20), (-0.15, 0.26), (-0.20, 0.26)]))
        heap = poly(arc(0.13, 0.265, 0.10, 0.09, 180, 360))
        return union(man, heap)
    if k == 26:
        return union(disc(0, -0.08, 0.05), disc(0, 0.06, 0.05), disc(0, 0.20, 0.05))  # coloured separately
    if k == 27:      # pedestrian
        return person(0.0, -0.10, 1.0)
    if k == 28:      # children: two figures side by side
        return union(person(-0.08, -0.04, 0.85), person(0.08, -0.08, 0.95))
    if k == 29:      # cyclist: two large wheels, rider above
        w1, w2 = (-0.115, 0.18), (0.115, 0.18)
        bike = union(ring(w1[0], w1[1], 0.055, 0.085), ring(w2[0], w2[1], 0.055, 0.085),
                     stroke([w1, (-0.01, 0.18), (0.06, 0.07), (-0.07, 0.07), (-0.01, 0.18)], 0.028),
                     stroke([(0.06, 0.07), w2], 0.028), stroke([(0.05, 0.04), (0.08, 0.03)], 0.028))
        rider = union(disc(0.05, -0.12, 0.036), stroke([(-0.05, 0.05), (0.02, -0.08)], 0.065),
                      stroke([(0.02, -0.06), (0.06, 0.03)], 0.04),
                      stroke([(-0.05, 0.05), (0.02, 0.11), (-0.01, 0.17)], 0.045))
        return union(bike, rider)
    if k == 30:      # snowflake: six-armed star
        c = (0.0, 0.09)
        parts = []
        for a in (90, 30, 150):
            t = np.radians(a)
            d = np.array([np.cos(t), np.sin(t)])
            p, q = np.array(c) - 0.13 * d, np.array(c) + 0.13 * d
            parts.append(stroke([tuple(p), tuple(q)], 0.038))
            for sgn in (1, -1):
                b = np.array(c) + sgn * 0.08 * d
                for rot in (45, -45):
                    r = np.radians(rot)
                    e = sgn * np.array([d[0] * np.cos(r) - d[1] * np.sin(r), d[0] * np.sin(r) + d[1] * np.cos(r)])
                    parts.append(stroke([tuple(b), tuple(b + 0.05 * e)], 0.026))
        return union(*parts)
    if k == 31:      # deer leaping up to the left: head and antlers upper left, hind legs to the lower right
        return union(disc(-0.03, -0.09, 0.035),
                     stroke([(-0.04, -0.11), (-0.06, -0.16)], 0.025), stroke([(-0.02, -0.11), (-0.01, -0.16)], 0.025),
                     stroke([(-0.03, -0.07), (-0.01, -0.01)], 0.05),
                     stroke([(-0.01, -0.01), (0.10, 0.11)], 0.075),
                     stroke([(0.09, 0.11), (0.16, 0.17), (0.21, 0.24)], 0.035),
                     stroke([(0.08, 0.13), (0.11, 0.20), (0.14, 0.26)], 0.035),
                     stroke([(0.0, 0.01), (-0.09, 0.05), (-0.12, 0.01)], 0.035),
                     stroke([(0.01, 0.03), (-0.07, 0.10), (-0.10, 0.08)], 0.03))
    raise ValueError(k)


def blue_arrows(k):
    W = 0.13
    if k == 35:
        return union(rect(-W / 2, W / 2, -0.15, 0.37), arrow_head((0, -0.37), (0, -1), 0.24, 0.18))
    if k == 33:
        return union(stroke([(-0.12, 0.37), (-0.12, -0.02)] + arc(0.03, -0.02, 0.15, 0.15, 180, 270) + [(0.10, -0.17)], W),
                     arrow_head((0.34, -0.17), (1, 0), 0.22, 0.17))
    if k == 34:
        return mirror(blue_arrows(33))
    if k == 36:
        w = 0.11
        return union(rect(-0.06 - w / 2, -0.06 + w / 2, -0.15, 0.37), arrow_head((-0.06, -0.37), (0, -1), 0.22, 0.16),
                     stroke([(-0.06, 0.22)] + arc(0.10, 0.22, 0.16, 0.17, 180, 270) + [(0.14, 0.05)], w),
                     arrow_head((0.35, 0.05), (1, 0), 0.20, 0.15))
    if k == 37:
        return mirror(blue_arrows(36))
    if k == 38:
        return union(stroke([(-0.25, -0.25), (0.08, 0.08)], W), arrow_head((0.27, 0.27), (1, 1), 0.27, 0.19))
    if k == 39:
        return mirror(blue_arrows(38))
    if k == 40:
        parts = []
        R = 0.25
        for c in (90, 210, 330):            # arc centres, mathematical angles (y up), counter-clockwise travel
            a0, a1 = c - 38, c + 28
            pts = [(R * np.cos(np.radians(a)), -R * np.sin(np.radians(a))) for a in np.linspace(a0, a1, 10)]
            parts.append(stroke(pts, 0.075))
            te = np.radians(a1 + 22)
            tip = (R * np.cos(te), -R * np.sin(te))
            dirv = (-np.sin(te), -np.cos(te))
            parts.append(arrow_head(tip, dirv, 0.13, 0.085))
        return union(*parts)
    raise ValueError(k)


# ----------------------------------------------------------------------------------------------
# sign designs: list of (shape, colour, is_content); plus geometry


def design(k):
    """returns dict(W, H, stretch, anchor, layers, interior); layers are (shape, colour, is_content)."""
    circ = dict(W=1.0, H=1.0, stretch=1.0, anchor=(0.0, 0.0))
    if k in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16):
        L = [(disc(0, 0, 0.5), RED, False), (disc(0, 0, 0.375), WHITE, False)]
        if k <= 8:
            s = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}[k]
            if len(s) == 2:
                L.append((text(s, 0, 0, 0.40, 0.58, 0.15, 0.10), BLACK, True))
            else:
                L.append((text(s, 0, 0, 0.40, 0.48, 0.13, 0.07), BLACK, True))
        elif k == 9:
            L += [(car_rear(-0.135, 0.08, 0.23), RED, True), (car_rear(0.135, 0.08, 0.23), BLACK, True)]
        elif k == 10:
            L += [(truck_rear(-0.13, 0.10, 0.22), RED, True), (car_rear(0.135, 0.10, 0.23), BLACK, True)]
        elif k == 16:
            L.append((truck_side(), BLACK, True))
        return dict(circ, layers=L, interior=disc(0, 0, 0.35), base=WHITE)
    if k in (6, 32, 41, 42):
        L = [(disc(0, 0, 0.5), WHITE, False)]
        if k == 6:
            L.append((text('80', 0, 0, 0.40, 0.58, 0.13, 0.10), GRAY, True))
        elif k == 41:
            L += [(car_rear(-0.135, 0.08, 0.23), GRAY, True), (car_rear(0.135, 0.08, 0.23), GRAY, True)]
        elif k == 42:
            L += [(truck_rear(-0.13, 0.10, 0.22), GRAY, True), (car_rear(0.135, 0.10, 0.23), GRAY, True)]
        b = end_band()
        clip = disc(0, 0, 0.49)
        L.append(((b[0], lambda U, V, f=b[1], g=clip[1]: f(U, V) & g(U, V)), BLACK, True))
        return dict(circ, layers=L, interior=disc(0, 0, 0.42), base=WHITE)
    if k == 17:
        return dict(circ, layers=[(disc(0, 0, 0.5), RED, False),
                                  (rect(-0.37, 0.37, -0.095, 0.095), WHITE, True)], interior=disc(0, 0, 0.42), base=RED)
    if 33 <= k <= 40:
        return dict(circ, layers=[(disc(0, 0, 0.5), BLUE, False),
                                  (blue_arrows(k), WHITE, True)], interior=disc(0, 0, 0.42), base=BLUE)
    if k == 14:
        L = [(octagon(0.5, 0.0), RED, False),
             (text('STOP', 0, 0, 0.31, 0.56, 0.18, 0.10, one=1.0), WHITE, True)]
        return dict(circ, layers=L, interior=octagon(0.42, 0.0), base=RED)
    if k == 12:
        L = [(diamond(0.5, 0.0), WHITE, False), (diamond(0.29, 0.0), YELLOW, True)]
        return dict(circ, layers=L, interior=diamond(0.44, 0.0), base=WHITE)
    if k == 13:
        g = -TRI_G
        L = [(rounded_tri(0, g, RHO, RC, 0.0, up=False), RED, False),
             (rounded_tri(0, g, RHO, RC, 0.145, up=False), WHITE, False)]
        return dict(W=TRI_W, H=TRI_H, stretch=TRI_STRETCH, anchor=(0.0, g), layers=L, interior=None, base=WHITE)
    # warning triangles
    g = TRI_G
    L = [(rounded_tri(0, g, RHO, RC, 0.0), RED, False), (rounded_tri(0, g, RHO, RC, 0.115), WHITE, False)]
    if k == 26:
        L += [(disc(0, -0.08, 0.05), LRED, True), (disc(0, 0.06, 0.05), AMBER, True), (disc(0, 0.20, 0.05), GREEN, True)]
    else:
        L.append((picto(k), BLACK, True))
    return dict(W=TRI_W, H=TRI_H, stretch=TRI_STRETCH, anchor=(0.0, 0.09), layers=L,
                interior=rounded_tri(0, g, RHO, RC, 0.14), base=WHITE)


# ----------------------------------------------------------------------------------------------
# rendering

RES, EXT = 560, 0.56
TEX = 2 * EXT / RES
AX = -EXT + (np.arange(RES) + 0.5) * TEX


def render_canvas(des, c, with_content=True, content_only=False):
    """with content_only, the pictogram is drawn on an endless field of the sign's surface colour (no border)."""
    rgb = np.zeros((RES, RES, 3))
    alpha = np.zeros((RES, RES))
    if content_only:
        rgb[:] = des['base']
        alpha[:] = 1.0
    au, av = des['anchor']
    for (bbox, fn), col, content in des['layers']:
        if (content and not with_content) or (content_only and not content):
            continue
        u0, u1, v0, v1 = bbox
        if content:
            u0, u1 = au + (u0 - au) * c, au + (u1 - au) * c
            v0, v1 = av + (v0 - av) * c, av + (v1 - av) * c
        i0 = max(int(np.floor((u0 + EXT) / TEX)) - 1, 0)
        i1 = min(int(np.ceil((u1 + EXT) / TEX)) + 1, RES)
        j0 = max(int(np.floor((v0 + EXT) / TEX)) - 1, 0)
        j1 = min(int(np.ceil((v1 + EXT) / TEX)) + 1, RES)
        if i1 <= i0 or j1 <= j0:
            continue
        U, V = np.meshgrid(AX[i0:i1], AX[j0:j1])
        if content:
            U, V = au + (U - au) / c, av + (V - av) / c
        m = fn(U, V)
        rgb[j0:j1, i0:i1][m] = col
        alpha[j0:j1, i0:i1][m] = 1.0
    return rgb, alpha


def render_mask(des):
    """canvas holding only the interior (content) region of the sign, as alpha."""
    rgb = np.zeros((RES, RES, 3))
    if des['interior'] is None:
        return rgb, np.zeros((RES, RES))
    _, fn = des['interior']
    U, V = np.meshgrid(AX, AX)
    return rgb, fn(U, V).astype(float)


SS = 8
CTR = 15.5


def sample(canvas, des, D, aspect):
    """render the sign with bbox width D*sqrt(aspect) px, centred in a 32x32 image; returns premultiplied rgb, alpha."""
    rgb, alpha = canvas
    pw = D * np.sqrt(aspect)
    ph = D * des['H'] / des['W'] * des['stretch'] / np.sqrt(aspect)
    xs = (np.arange(32 * SS) + 0.5) / SS - 0.5
    u = (xs - CTR) / pw * des['W']
    v = (xs - CTR) / ph * des['H']
    iu = np.floor((u + EXT) / TEX).astype(int)
    iv = np.floor((v + EXT) / TEX).astype(int)
    okx = (iu >= 0) & (iu < RES)
    oky = (iv >= 0) & (iv < RES)
    iu, iv = np.clip(iu, 0, RES - 1), np.clip(iv, 0, RES - 1)
    ok = oky[:, None] & okx[None, :]
    a = alpha[iv[:, None], iu[None, :]] * ok
    c = rgb[iv[:, None], iu[None, :]] * a[..., None]
    a = a.reshape(32, SS, 32, SS).mean((1, 3))
    c = c.reshape(32, SS, 32, SS, 3).mean((1, 3)).transpose(2, 0, 1)
    return c, a


def gauss_blur(x, sigma):
    r = int(np.ceil(3 * sigma))
    t = np.arange(-r, r + 1)
    g = np.exp(-t ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    pad = [(0, 0)] * (x.ndim - 2) + [(r, r), (r, r)]
    xp = np.pad(x, pad)
    H, W = x.shape[-2:]
    y = sum(g[i] * xp[..., i:i + H, :] for i in range(2 * r + 1))
    y = sum(g[i] * y[..., :, i:i + W] for i in range(2 * r + 1))
    return y


# opponent colour axes: lightness, red-green, yellow-blue (orthonormal)
OPP = np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]], float)
OPP /= np.linalg.norm(OPP, axis=1, keepdims=True)
SIGMA_FRAME, SIGMA_CONTENT = 0.8, 1.0
SIGMA_SURROUND = 3.0             # scale separating the coarse pictogram (frame) from its detail (content)
PLAIN_BIAS = 2.0                 # content score credited to classes whose interior is plain (nothing drawn)


def blurred_colour(c_pm, a, sigma):
    return gauss_blur(c_pm, sigma) / np.maximum(gauss_blur(a, sigma), 1e-6)


def local_mean(col, m):
    """normalised Gaussian average of a drawing inside mask m."""
    return gauss_blur(col * m, SIGMA_SURROUND) / np.maximum(gauss_blur(m, SIGMA_SURROUND), 1e-6)


def budgets(TL, TC):
    """separate unit budgets for the lightness part and the colour part (colour dropped if the drawing has none);
    returned in normalised-RGB space, None if flat."""
    out = np.zeros((3,) + TL.shape)
    nl, nc = np.linalg.norm(TL), np.linalg.norm(TC)
    if nl < 1e-6 and nc < 1e-6:
        return None
    if nl >= 1e-6:
        out[0] = TL / nl
    if nc > 0.05 * max(nl, 1e-6):
        out[1:] = TC / nc
    return np.tensordot(OPP.T, out, 1)


def frame_template(col, m):
    """sign surface: lightness relative to the sign's own mean (shape of light and dark parts) and colour
    relative to neutral (the image's per-channel normalisation makes its average colour neutral)."""
    O = np.tensordot(OPP, col, 1)
    return budgets(m * (O[0] - (O[0] * m).sum() / m.sum()), m * O[1:])


COLOUR_CONTENT = (9, 10, 12, 26)  # classes whose pictogram colour carries information beyond its lightness


def content_template(col, m, colour):
    """pictogram detail inside the interior mask m (col already band-passed): zero-mean in every axis.
    White-on-colour and black-on-white symbols are matched on lightness only."""
    O = np.tensordot(OPP, col, 1)
    O = m * (O - ((O * m).sum((1, 2)) / m.sum())[:, None, None])
    return budgets(O[0], O[1:] if colour else 0 * O[1:])


SIZES = 19.0 * (27.0 / 19.0) ** (np.arange(7) / 6)
ASPECTS = (0.92, 1.0, 1.087)
CONTENT = (0.92, 1.08)
NCLS = 43
PAD = 3


def variants():
    return [(c, D, a) for c in CONTENT for D in SIZES for a in ASPECTS]


def build_templates():
    """per class and variant: a frame template (the sign with its pictogram blurred to a coarse blob, whole sign
    area) and a content template (the pictogram's detail = drawing minus its coarse blur, interior only)."""
    V = variants()
    WF = np.zeros((NCLS * len(V), 3, 32, 32), np.float32)
    WC = np.zeros((NCLS * len(V), 3, 32, 32), np.float32)
    bias_f = np.zeros(NCLS * len(V), np.float32)
    bias_c = np.zeros(NCLS * len(V), np.float32)
    for k in range(NCLS):
        des = design(k)
        frame = render_canvas(des, 1.0, with_content=False)
        full = {c: render_canvas(des, c, content_only=True) for c in CONTENT}
        inner = render_mask(des)
        for j, (c, D, a) in enumerate(V):
            idx = k * len(V) + j
            cp, al = sample(frame, des, D, a)
            colf = blurred_colour(cp, al, SIGMA_FRAME)
            _, mi = sample(inner, des, D, a)
            t = None
            if mi.sum() > 0:
                cp, alc = sample(full[c], des, D, a)
                colc = blurred_colour(cp, alc, SIGMA_CONTENT)
                low = local_mean(colc, mi)
                colf = colf + mi * (low - np.asarray(des['base'])[:, None, None])
                t = content_template(colc - low, mi, k in COLOUR_CONTENT)
            if t is None:
                bias_c[idx] = PLAIN_BIAS
            else:
                WC[idx] = t
            WF[idx] = frame_template(colf, al)
    return WF, WC, bias_f, bias_c


def network(nv):
    K = NCLS * nv
    P = 2 * PAD + 1
    return {"layers": [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "frame", "op": "conv", "inputs": ["n0"], "out": K, "kernel": [32, 32], "stride": 1, "padding": PAD},
        {"name": "content", "op": "conv", "inputs": ["n0"], "out": K, "kernel": [32, 32], "stride": 1, "padding": PAD + 1},
        {"name": "content_local", "op": "maxpool", "inputs": ["content"], "size": [3, 3], "stride": [1, 1]},
        {"name": "both", "op": "add", "inputs": ["frame", "content_local"]},
        {"name": "mshift", "op": "maxpool", "inputs": ["both"], "size": [P, P], "stride": [P, P]},
        {"name": "byclass", "op": "reshape", "inputs": ["mshift"], "shape": [NCLS, nv, 1]},
        {"name": "mvar", "op": "maxpool", "inputs": ["byclass"], "size": [nv, 1], "stride": [nv, 1]},
        {"name": "scores", "op": "reshape", "inputs": ["mvar"], "shape": [NCLS]},
    ]}


def main(out):
    os.makedirs(out, exist_ok=True)
    WF, WC, bias_f, bias_c = build_templates()
    nv = len(variants())
    net = network(nv)
    weights = {"frame.W": WF, "frame.b": bias_f, "content.W": WC, "content.b": bias_c}
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(net, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **weights)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
