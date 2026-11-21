"""Build a hand-designed ASL fingerspelling network.  No data is read: every number
comes from formulas (Gaussian derivatives), from an anatomical hand shape model
typed from general knowledge of the ASL alphabet, and from hand-chosen settings.

Usage:  python3 build.py OUT_DIR
"""
import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# signed distance primitives in the hand frame (x right, y up, wrist at origin,
# lengths in units of palm width)
# --------------------------------------------------------------------------


def sdf(p, X, Y):
    k = p[0]
    if k == "cap":
        _, ax, ay, bx, by, r = p
        vx, vy = bx - ax, by - ay
        wx, wy = X - ax, Y - ay
        L2 = vx * vx + vy * vy
        t = np.clip((wx * vx + wy * vy) / (L2 if L2 > 1e-12 else 1e-12), 0.0, 1.0)
        dx, dy = wx - t * vx, wy - t * vy
        return np.sqrt(dx * dx + dy * dy) - r
    if k == "box":
        _, cx, cy, hw, hh, r = p
        qx = np.abs(X - cx) - (hw - r)
        qy = np.abs(Y - cy) - (hh - r)
        ax, ay = np.maximum(qx, 0.0), np.maximum(qy, 0.0)
        return np.sqrt(ax * ax + ay * ay) + np.minimum(np.maximum(qx, qy), 0.0) - r
    if k == "varc":
        _, cx, cy, R, t0, t1, a0, a1 = p
        dx, dy = X - cx, Y - cy
        d = np.sqrt(dx * dx + dy * dy)
        ang = np.degrees(np.arctan2(dy, dx))
        th = t0 + t1 * np.cos(np.radians(ang - 180.0))
        u0, u1 = a0 % 360.0, a1 % 360.0
        a = ang % 360.0
        ins = (a >= u0) & (a <= u1) if u0 <= u1 else (a >= u0) | (a <= u1)
        rad = np.abs(d - R) - 0.5 * th
        e = []
        for aa in (a0, a1):
            ar = np.radians(aa)
            ex, ey = cx + R * np.cos(ar), cy + R * np.sin(ar)
            te = t0 + t1 * np.cos(np.radians(aa - 180.0))
            e.append(np.sqrt((X - ex) ** 2 + (Y - ey) ** 2) - 0.5 * te)
        return np.where(ins, rad, np.minimum(e[0], e[1]))
    if k == "ring":
        _, cx, cy, rx, ry, th = p
        dx, dy = (X - cx) / rx, (Y - cy) / ry
        e = np.sqrt(dx * dx + dy * dy) + 1e-9
        gx, gy = dx / (rx * e), dy / (ry * e)
        g = np.sqrt(gx * gx + gy * gy) + 1e-9
        return np.abs((e - 1.0) / g) - 0.5 * th
    raise ValueError(k)


def cap(ax, ay, bx, by, r):
    return ("cap", ax, ay, bx, by, r)


def disk(cx, cy, r):
    return ("cap", cx, cy, cx, cy, r)


def box(cx, cy, hw, hh, r):
    return ("box", cx, cy, hw, hh, r)


# --------------------------------------------------------------------------
# anatomy constants (general knowledge of hand proportions, palm width = 1)
# --------------------------------------------------------------------------
FX = [-0.36, -0.12, 0.12, 0.36]          # index, middle, ring, pinky knuckle x
FL = [0.92, 1.02, 0.94, 0.74]            # finger lengths from the knuckle line
FR = [0.115, 0.120, 0.110, 0.095]        # finger radii
KY = 1.00                                # knuckle line height
PALM = box(0.0, 0.82, 0.48, 0.30, 0.24)     # upper palm, full width (all fingers out)
PALM_LO = cap(0.0, 0.75, 0.0, 0.20, 0.36)   # lower palm tapers toward the wrist
PALM_N = box(0.0, 0.80, 0.40, 0.30, 0.24)   # narrower: curled fingers fold the hand in
PALM_NLO = cap(0.0, 0.72, 0.0, 0.20, 0.32)
FIST = box(0.0, 0.56, 0.50, 0.52, 0.42)     # a fist is rounded, not boxy
WRIST = cap(0.0, 0.30, 0.0, -1.90, 0.38)


def finger(i, spread=0.0, length=1.0, bend=0.0, base=None, fat=1.0):
    """Two tapered capsule segments for an extended finger."""
    x0, y0 = (FX[i], KY) if base is None else base
    a = np.radians(90.0 + spread)
    L = FL[i] * length
    mx, my = x0 + 0.55 * L * np.cos(a), y0 + 0.55 * L * np.sin(a)
    a2 = a + np.radians(bend)
    tx, ty = mx + 0.45 * L * np.cos(a2), my + 0.45 * L * np.sin(a2)
    r = FR[i] * fat
    return [cap(x0, y0, mx, my, r), cap(mx, my, tx, ty, 0.88 * r)]


def chain(pts, r):
    return [cap(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                r * (1.0 - 0.10 * i / max(1, len(pts) - 2)))
            for i in range(len(pts) - 1)]


class Shape:
    def __init__(self):
        self.parts = []      # (primitive, intensity weight, in bbox, draws a crease)
        self.holes = []

    def add(self, p, w=1.0, bb=True, crease=True):
        self.parts.append((p, w, bb, crease))

    def many(self, ps, w=1.0, bb=True):
        for p in ps:
            self.add(p, w, bb)

    def wrist(self, p=None):
        self.add(WRIST if p is None else p, 0.60, False, False)

    def palm(self, narrow=False):
        if narrow:
            self.add(PALM_N); self.add(PALM_NLO)
        else:
            self.add(PALM); self.add(PALM_LO)

    def knuckles(self, idx=(0, 1, 2, 3), r=0.14, y=1.14):
        for i in idx:
            self.add(disk(FX[i], y, r))

    def hole(self, cx, cy, r):
        self.holes.append(disk(cx, cy, r))


def curled_thumb_across(sh, y=0.62):
    """Thumb folded across the palm: no silhouette change, but a real crease."""
    sh.add(cap(-0.46, y - 0.10, 0.20, y + 0.06, 0.13))


# --------------------------------------------------------------------------
# the 24 letters, drawn from general knowledge of the ASL alphabet
# --------------------------------------------------------------------------
def make_shape(letter, pose):
    sp = (0.85, 1.18)[pose]          # finger-spread variant
    ln = (1.00, 0.95)[pose]          # finger-length variant
    s = Shape()
    L = letter

    if L == "A":                     # fist, thumb straight up along the index side
        s.add(FIST)
        s.knuckles(r=0.13, y=1.02)
        s.add(cap(-0.52, 0.45, -0.55, 1.34, 0.15))
        s.wrist()
    elif L == "B":                   # four fingers up together, thumb across palm
        s.palm(narrow=False)
        for i in range(4):
            s.many(finger(i, spread=(i - 1.5) * 1.5 * sp, length=ln, fat=1.14))
        curled_thumb_across(s)
        s.wrist()
    elif L == "C":                   # cupped hand: a crescent opening to the right
        s.add(("varc", 0.0, 0.10, 0.95, 0.44, 0.17, 42.0, 318.0))
        s.add(cap(0.70, 0.72, 0.80, 0.84, 0.16))      # rounded fingertip end
        s.add(cap(0.70, -0.53, 0.80, -0.64, 0.14))    # thumb end
        a = np.radians(248.0)
        s.wrist(cap(1.00 * np.cos(a), 0.10 + 1.00 * np.sin(a),
                    2.60 * np.cos(a), 0.10 + 2.60 * np.sin(a), 0.36))
    elif L == "D":                   # index up, others curled onto thumb: a loop
        s.add(box(0.0, 0.55, 0.42, 0.50, 0.34))
        s.many(finger(0, spread=2.0 * sp, length=ln))
        s.add(("ring", 0.16, 0.66, 0.36, 0.31, 0.17))
        s.add(cap(-0.44, 0.36, -0.10, 0.44, 0.13))
        s.hole(0.16, 0.66, 0.20)
        s.wrist()
    elif L == "E":                   # fingers curled tight over a tucked thumb
        s.add(box(0.0, 0.54, 0.53, 0.50, 0.42))
        s.knuckles(r=0.12, y=0.98)
        s.add(cap(-0.46, 0.26, 0.22, 0.32, 0.13))
        s.wrist()
    elif L == "F":                   # index+thumb ring, three fingers fanned up
        s.add(box(0.0, 0.52, 0.46, 0.48, 0.30))
        for i, a in ((1, 2.0), (2, -11.0), (3, -26.0)):
            s.many(finger(i, spread=a * sp, length=ln))
        s.add(("ring", -0.12, 0.66, 0.30, 0.27, 0.15))
        s.hole(-0.12, 0.66, 0.17)
        s.wrist()
    elif L == "G":                   # index and thumb parallel, pointing sideways
        s.add(box(0.0, 0.56, 0.46, 0.52, 0.36))
        s.many(finger(0, spread=0.0, length=1.05 * ln, base=(-0.28, KY)))
        s.add(cap(-0.70, 0.50, -0.74, 1.28, 0.14))
        s.knuckles(idx=(1, 2, 3), r=0.12, y=1.00)
        s.wrist()
    elif L == "H":                   # index+middle together, pointing sideways
        s.add(box(0.0, 0.56, 0.44, 0.52, 0.34))
        for i in (0, 1):
            s.many(finger(i, spread=0.0, length=1.05 * ln))
        s.knuckles(idx=(2, 3), r=0.12, y=1.00)
        curled_thumb_across(s, y=0.55)
        s.wrist()
    elif L == "I":                   # pinky alone up
        s.add(FIST)
        s.knuckles(idx=(0, 1, 2), r=0.13, y=1.02)
        s.many(finger(3, spread=-7.0 * sp, length=ln))
        curled_thumb_across(s)
        s.wrist()
    elif L == "K":                   # index+middle in a V, thumb up between them
        s.palm(narrow=True)
        s.many(finger(0, spread=13.0 * sp, length=ln))
        s.many(finger(1, spread=-9.0 * sp, length=ln))
        s.add(cap(-0.46, 0.40, -0.26, 1.18, 0.14))
        s.knuckles(idx=(2, 3), r=0.12, y=1.00)
        s.wrist()
    elif L == "L":                   # index up, thumb straight out: a right angle
        s.palm(narrow=True)
        s.many(finger(0, spread=1.0, length=ln))
        s.add(cap(-0.44, 0.46, -1.26, 0.52, 0.15))
        s.knuckles(idx=(1, 2, 3), r=0.12, y=1.00)
        s.wrist()
    elif L == "M":                   # three fingers draped over the thumb
        s.add(box(0.0, 0.56, 0.53, 0.52, 0.42))
        s.knuckles(idx=(0, 1, 2), r=0.17, y=1.00)
        s.knuckles(idx=(3,), r=0.12, y=0.96)
        s.add(cap(-0.30, 0.34, 0.46, 0.44, 0.13))
        s.wrist()
    elif L == "N":                   # two fingers draped over the thumb
        s.add(box(0.0, 0.56, 0.53, 0.52, 0.42))
        s.knuckles(idx=(0, 1), r=0.18, y=1.02)
        s.knuckles(idx=(2, 3), r=0.12, y=0.96)
        s.add(cap(-0.30, 0.34, 0.46, 0.44, 0.13))
        s.wrist()
    elif L == "O":                   # all fingers meet the thumb: a closed ring
        s.add(("ring", 0.0, 0.78, 0.78, 0.82, 0.55))
        s.add(cap(-0.55, 0.20, -0.30, 0.02, 0.20))
        s.wrist(cap(-0.20, 0.10, -0.30, -1.90, 0.36))
    elif L == "P":                   # K turned downward
        s.palm(narrow=True)
        s.many(finger(0, spread=15.0 * sp, length=ln))
        s.many(finger(1, spread=-14.0 * sp, length=0.80 * ln))
        s.add(cap(-0.46, 0.40, -0.28, 1.10, 0.14))
        s.knuckles(idx=(2, 3), r=0.12, y=1.00)
        s.wrist()
    elif L == "Q":                   # G turned downward
        s.add(box(0.0, 0.56, 0.46, 0.52, 0.36))
        s.many(finger(0, spread=0.0, length=1.00 * ln, base=(-0.28, KY)))
        s.add(cap(-0.70, 0.50, -0.74, 1.24, 0.14))
        s.knuckles(idx=(1, 2, 3), r=0.12, y=1.00)
        s.wrist()
    elif L == "R":                   # index and middle crossed
        s.palm(narrow=True)
        s.many(finger(0, spread=-7.0, length=ln, bend=-4.0))
        s.many(finger(1, spread=9.0, length=ln, bend=4.0))
        s.knuckles(idx=(2, 3), r=0.12, y=1.00)
        curled_thumb_across(s)
        s.wrist()
    elif L == "S":                   # fist, thumb lying across the front
        s.add(FIST)
        s.knuckles(r=0.13, y=1.02)
        s.add(cap(-0.56, 0.74, 0.26, 0.86, 0.15))
        s.wrist()
    elif L == "T":                   # fist, thumb tip between index and middle
        s.add(FIST)
        s.knuckles(r=0.13, y=1.02)
        s.add(cap(-0.50, 0.50, -0.25, 1.22, 0.14))
        s.wrist()
    elif L == "U":                   # index+middle up, touching
        s.palm(narrow=True)
        for i in (0, 1):
            s.many(finger(i, spread=0.0, length=ln))
        s.knuckles(idx=(2, 3), r=0.12, y=1.00)
        curled_thumb_across(s)
        s.wrist()
    elif L == "V":                   # index+middle up, spread apart
        s.palm(narrow=True)
        s.many(finger(0, spread=16.0 * sp, length=ln))
        s.many(finger(1, spread=-14.0 * sp, length=ln))
        s.knuckles(idx=(2, 3), r=0.12, y=1.00)
        curled_thumb_across(s)
        s.wrist()
    elif L == "W":                   # three fingers up and spread
        s.palm(narrow=False)
        for i, a in ((0, 18.0), (1, 0.0), (2, -18.0)):
            s.many(finger(i, spread=a * sp, length=ln))
        s.knuckles(idx=(3,), r=0.12, y=1.00)
        curled_thumb_across(s, y=0.55)
        s.wrist()
    elif L == "X":                   # index bent into a hook
        s.add(FIST)
        s.knuckles(idx=(1, 2, 3), r=0.13, y=1.02)
        s.many(chain([(-0.36, KY), (-0.37, 1.52), (-0.16, 1.66), (0.02, 1.44)], 0.115))
        curled_thumb_across(s)
        s.wrist()
    elif L == "Y":                   # thumb and pinky out, middle three curled
        s.palm(narrow=True)
        s.many(finger(3, spread=-30.0 * sp, length=ln))
        s.add(cap(-0.44, 0.40, -1.16, 0.96, 0.15))
        s.knuckles(idx=(0, 1, 2), r=0.13, y=1.02)
        s.wrist()
    else:
        raise ValueError(L)
    return s


LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
# base orientation of the whole hand in the image (degrees, 0 = fingers up).
# G/H point sideways, P/Q point down; the rest point up.
BASE_ROT = {c: 0.0 for c in LETTERS}
BASE_ROT["G"] = -90.0
BASE_ROT["H"] = -90.0
BASE_ROT["P"] = 142.0
BASE_ROT["Q"] = 152.0
# letters whose pose is least constrained get a wider rotation sweep
WIDE = {"P", "Q", "C"}

# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------
SS = 3                      # supersampling factor
N = 28
TARGET = 24.0               # hand bbox long side, in pixels
SCALES = (0.86, 1.00, 1.13)
ROTS = (-12.0, -4.0, 4.0, 12.0)
ROTS_WIDE = (-26.0, -9.0, 9.0, 26.0)
MIRRORS = (False, True)
POSES = (0, 1)


def union_sdf(sh, X, Y, only_bbox=False):
    d = None
    for p, w, bb, cr in sh.parts:
        if only_bbox and not bb:
            continue
        v = sdf(p, X, Y)
        d = v if d is None else np.minimum(d, v)
    for h in sh.holes:
        d = np.maximum(d, -sdf(h, X, Y))
    return d


def hand_frame(sh, rot, mirror):
    """Return (scale, bcx, bcy, rot, mirror) fitting the hand bbox to TARGET."""
    gx = np.arange(-2.9, 2.91, 0.05)
    gy = np.arange(-2.2, 3.31, 0.05)
    X, Y = np.meshgrid(gx, gy)
    d = union_sdf(sh, X, Y, only_bbox=True)
    ys, xs = np.where(d < 0.0)
    px, py = X[ys, xs], Y[ys, xs]
    if mirror:
        px = -px
    a = np.radians(rot)
    rx = px * np.cos(a) - py * np.sin(a)
    ry = px * np.sin(a) + py * np.cos(a)
    bw = rx.max() - rx.min()
    bh = ry.max() - ry.min()
    sc = TARGET / max(bw, bh)
    return sc, 0.5 * (rx.max() + rx.min()), 0.5 * (ry.max() + ry.min())


def render(sh, rot, mirror, scale_mul):
    """Return (mask, part-boundary map) on the 28x28 grid."""
    sc0, bcx, bcy = hand_frame(sh, rot, mirror)
    sc = sc0 * scale_mul
    # supersampled pixel centres -> rotated frame -> hand frame
    q = (np.arange(N * SS) + 0.5) / SS - 0.5          # pixel coordinate 0..27.5
    C, R = np.meshgrid(q, q)
    px = (C - 13.5) / sc + bcx
    py = -(R - 13.5) / sc + bcy
    a = np.radians(-rot)
    hx = px * np.cos(a) - py * np.sin(a)
    hy = px * np.sin(a) + py * np.cos(a)
    if mirror:
        hx = -hx
    d = union_sdf(sh, hx, hy)
    mask = (d < 0.0).astype(np.float64)
    # weighted mask: wrist counts less than the hand itself
    wmap = np.zeros_like(mask)
    for p, w, bb, cr in sh.parts:
        inside = sdf(p, hx, hy) < 0.0
        wmap = np.where(inside, np.maximum(wmap, w), wmap)
    for h in sh.holes:
        wmap = np.where(sdf(h, hx, hy) < 0.0, 0.0, wmap)
    mask = mask * wmap
    # part boundaries: the outline, plus creases where separate parts meet.
    # a crease only shows near the surface; arcs buried deep inside the hand
    # (e.g. the underside of a knuckle bump) are not visible.
    tau = 0.70 / sc                                   # ~0.7 pixel wide ridge
    bnd = np.exp(-(d / tau) ** 2)
    depth = np.clip((d + 0.30) / 0.14, 0.0, 1.0)
    for p, w, bb, cr in sh.parts:
        if not cr:
            continue
        v = sdf(p, hx, hy)
        bnd = np.maximum(bnd, np.exp(-(v / tau) ** 2) * depth)
    bnd = bnd * np.clip(1.0 - d / (2.0 / sc), 0.0, 1.0)

    def down(A):
        return A.reshape(N, SS, N, SS).mean(axis=(1, 3))

    return down(mask), down(bnd)


# --------------------------------------------------------------------------
# small filter helpers (formulas only)
# --------------------------------------------------------------------------


def gauss(sig, k):
    ax = np.arange(k) - (k - 1) / 2.0
    X, Y = np.meshgrid(ax, ax)
    g = np.exp(-(X * X + Y * Y) / (2.0 * sig * sig))
    return g / g.sum()


def dgauss(sig, k):
    ax = np.arange(k) - (k - 1) / 2.0
    X, Y = np.meshgrid(ax, ax)
    g = np.exp(-(X * X + Y * Y) / (2.0 * sig * sig))
    gx = -X / (sig * sig) * g
    gy = -Y / (sig * sig) * g
    gx /= np.sqrt((gx ** 2).sum())
    gy /= np.sqrt((gy ** 2).sum())
    return gx, gy


def unit_zero_mean(A):
    B = A - A.mean()
    n = np.sqrt((B ** 2).sum())
    return B / (n if n > 1e-12 else 1.0)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
SAT_A = 1.0            # saturation knee, in standard deviations
SAT_K = 0.0            # hard clip: makes the map quasi-binary, so a match is
                       # scored on shape agreement, not on one strong feature
DOG_SIG = 3.0          # centre-surround scale: keeps finger/crease structure,
DOG_K = 15             # rejects the smooth, uneven background
BLUR_IMG = 0.70
BLUR_BND = 0.90
BLUR_EDGE = 0.60
EDGE_SIG = (1.0, 2.0)
EDGE_K = 7
PAD = 6
STRIDE = 2
NPOS = (N + 2 * PAD - N) // STRIDE + 1     # 7 shifts: -6..6


def saturate(x):
    return x - (1.0 - SAT_K) * np.maximum(x - SAT_A, 0.0) \
             + (1.0 - SAT_K) * np.maximum(-x - SAT_A, 0.0)


def norm_t(A):
    B = A - A.mean()
    return B / (np.sqrt((B ** 2).mean()) + 1e-6)


def blur_zero(A, g):
    """Gaussian blur with zero padding, exactly as the conv block does it."""
    k = g.shape[0]
    r = k // 2
    P = np.pad(A, r, mode="constant")
    out = np.zeros_like(A)
    for dy in range(k):
        for dx in range(k):
            out += g[dy, dx] * P[dy:dy + A.shape[0], dx:dx + A.shape[1]]
    return out


def front(mask):
    """The network's shared front end, applied to a drawing."""
    t = blur_zero(mask - mask.mean(), gauss(BLUR_IMG, 5))
    return norm_t(saturate(norm_t(t)))


def tpl_sil(mask):
    """Stops before the centre-surround, so solid-vs-hollow is preserved."""
    return unit_zero_mean(front(mask))


def tpl_dog(mask):
    t = front(mask)
    t = t - blur_zero(t, gauss(DOG_SIG, DOG_K))
    return unit_zero_mean(norm_t(saturate(norm_t(t))))


def tpl_edge(bnd):
    b = blur_zero(bnd - bnd.mean(), gauss(BLUR_BND, 5))
    b = norm_t(saturate(norm_t(b)))
    return unit_zero_mean(b)


def variant_list(letter):
    rots = ROTS_WIDE if letter in WIDE else ROTS
    out = []
    for r in rots:
        for sc in SCALES:
            for m in MIRRORS:
                for p in POSES:
                    out.append((BASE_ROT[letter] + r, m, sc, p))
    return out


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    nv = len(variant_list("A"))
    assert all(len(variant_list(c)) == nv for c in LETTERS)
    K = len(LETTERS)

    # channel order per bank: class, variant, [polarity]
    WS = np.zeros((K * nv * 2, 1, N, N), np.float32)
    WD = np.zeros((K * nv * 2, 1, N, N), np.float32)
    WB = np.zeros((K * nv, 1, N, N), np.float32)
    for k, letter in enumerate(LETTERS):
        for v, (rot, mir, scm, pose) in enumerate(variant_list(letter)):
            mask, bnd = render(make_shape(letter, pose), rot, mir, scm)
            tS, tD, tB = tpl_sil(mask), tpl_dog(mask), tpl_edge(bnd)
            for pol, sgn in ((0, 1.0), (1, -1.0)):
                WS[k * nv * 2 + 2 * v + pol, 0] = sgn * tS
                WD[k * nv * 2 + 2 * v + pol, 0] = sgn * tD
            WB[k * nv + v, 0] = tB

    layers, W = [], {}
    BANKS = {"S": WS, "D": WD, "B": WB}

    def conv(name, inp, out, k, stride, pad, w, b=None):
        layers.append({"name": name, "op": "conv", "inputs": [inp], "out": out,
                       "kernel": [k, k], "stride": stride, "padding": pad})
        W[name + ".W"] = np.asarray(w, np.float32)
        W[name + ".b"] = np.zeros(out, np.float32) if b is None else np.asarray(b, np.float32)

    def simple(name, op, inp, **kw):
        d = {"name": name, "op": op, "inputs": [inp] if isinstance(inp, str) else inp}
        d.update(kw)
        layers.append(d)

    def bank(tag, src, pol):
        """best shift -> best polarity -> best variant, per class"""
        conv("c" + tag, src, K * nv * pol, N, STRIDE, PAD, BANKS[tag])
        simple("r" + tag, "relu", "c" + tag)
        simple("p" + tag, "maxpool", "r" + tag, size=[NPOS, NPOS], stride=[NPOS, NPOS])
        src2 = "p" + tag
        if pol == 2:
            simple("o" + tag, "reshape", src2, shape=[K * nv, 2, 1])
            simple("x" + tag, "maxpool", "o" + tag, size=[2, 1], stride=[2, 1])
            src2 = "x" + tag
        simple("s" + tag, "reshape", src2, shape=[K, nv, 1])
        simple("m" + tag, "maxpool", "s" + tag, size=[nv, 1], stride=[nv, 1])
        simple("f" + tag, "reshape", "m" + tag, shape=[K])

    def clip_block(tag, src):
        """norm -> hard clip to +-SAT_A -> norm, as three tiny layers."""
        simple(tag + "i", "norm", src)
        conv(tag + "c", tag + "i", 4, 1, 1, 0,
             np.array([1., -1., 1., -1.]).reshape(4, 1, 1, 1),
             np.array([0., 0., -SAT_A, -SAT_A]))
        simple(tag + "r", "relu", tag + "c")
        conv(tag, tag + "r", 1, 1, 1, 0,
             np.array([1., -1., -(1. - SAT_K), (1. - SAT_K)]).reshape(1, 4, 1, 1))
        simple("n" + tag, "norm", tag)
        return "n" + tag

    # ---- shared front end: denoise, normalise illumination, clip outliers ----
    simple("n0", "norm", "image")
    conv("sm", "n0", 1, 5, 1, 2, gauss(BLUR_IMG, 5)[None, None])
    nS = clip_block("S0", "sm")

    # ---- branch S: overall silhouette (solid vs. hollow), either polarity ----
    bank("S", nS, 2)

    # ---- branch D: centre-surround contrast, either polarity ----
    conv("dbl", nS, 1, DOG_K, 1, DOG_K // 2, gauss(DOG_SIG, DOG_K)[None, None])
    conv("dng", "dbl", 1, 1, 1, 0, -np.ones((1, 1, 1, 1), np.float32))
    simple("dog", "add", [nS, "dng"])
    simple("nD0", "norm", "dog")
    nD = clip_block("T", "nD0")
    bank("D", nD, 2)

    # ---- branch B: polarity-free edges vs. outline + finger creases ----
    gxs, gys = [], []
    for sig in EDGE_SIG:
        gx, gy = dgauss(sig, EDGE_K)
        gxs += [gx, -gx]
        gys += [gy, -gy]
    kern = np.stack([gxs[0], gxs[1], gys[0], gys[1],
                     gxs[2], gxs[3], gys[2], gys[3]])[:, None]
    conv("grad", nS, 8, EDGE_K, 1, EDGE_K // 2, kern)
    simple("rg", "relu", "grad")
    conv("emag", "rg", 1, 1, 1, 0, np.ones((1, 8, 1, 1), np.float32))
    conv("eblur", "emag", 1, 5, 1, 2, gauss(BLUR_EDGE, 5)[None, None])
    simple("nB0", "norm", "eblur")
    nB = clip_block("U", "nB0")
    bank("B", nB, 1)

    # ---- hand-set readout: three complementary views, weighted alike ----
    for tag in ("S", "D", "B"):
        layers.append({"name": "d" + tag, "op": "dense", "inputs": ["f" + tag], "out": K})
        W["d" + tag + ".W"] = np.eye(K, dtype=np.float32)
        W["d" + tag + ".b"] = np.zeros(K, np.float32)
    simple("score", "add", ["dS", "dD", "dB"])

    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **W)
    print("variants/class", nv, "templates", K * nv,
          "weights", sum(int(v.size) for v in W.values()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
