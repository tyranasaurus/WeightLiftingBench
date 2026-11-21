"""Builds a hand-designed traffic-sign classifier network for 32x32 GTSRB-style images.

Everything is drawn from general knowledge of German road signs (shapes, colours,
pictograms, digits as strokes).  No training data is read; no statistics of any
images are used.  Each template is normalised by itself alone.

Run:  python3 build.py OUT_DIR   ->  OUT_DIR/network.json, OUT_DIR/weights.npz
"""
import json
import sys

import numpy as np

# ---------------------------------------------------------------- constants
KS = 28          # conv kernel canvas for template banks
PAD = 2          # conv padding -> 9x9 position search (+-4 px)
S = 4            # supersampling factor for rasterisation
CH = 8           # feature channels
# channel indices of the feature stack F
C_RG, C_RB, C_BR, C_BG, C_YEL, C_GRAY, C_INK, C_BRI = range(8)

SCALES = [22.0, 24.5, 27.0]          # sign diameter in px, 3 variants
NSCALES = [22.5, 25.0, 27.5]         # scales for the number bank
ROTS = [-0.25, 0.0, 0.25]            # rotation variants for digits
PROTS = [-0.35, 0.0, 0.35]          # rotation variants for pictograms (signs are often rotated)


# ---------------------------------------------------------------- rasteriser
def grid(canvas, D):
    """supersample coordinate grids in sign units (x right, y down, sign centre 0)"""
    n = canvas * S
    xs = ((np.arange(n) + 0.5) / S - canvas / 2.0) / D
    ys = ((np.arange(n) + 0.5) / S - canvas / 2.0) / D
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def raster(canvas, D, shape):
    """raster one shape -> bool array (canvas*S, canvas*S) in sign units"""
    X, Y = grid(canvas, D)
    return _shape(X, Y, shape)


def _shape(X, Y, sh):
    kind = sh[0]
    if kind == "disc":
        _, cx, cy, r = sh
        return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    if kind == "ring":
        _, cx, cy, ri, ro = sh
        d = (X - cx) ** 2 + (Y - cy) ** 2
        return (d <= ro * ro) & (d >= ri * ri)
    if kind == "rect":
        _, x0, y0, x1, y1 = sh
        return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    if kind == "poly":
        _, pts = sh
        return _poly(X, Y, pts)
    if kind == "stroke":
        _, pts, w = sh
        return _stroke(X, Y, pts, w)
    if kind == "sub":
        _, a, b = sh
        return _shape(X, Y, a) & ~_shape(X, Y, b)
    if kind == "union":
        out = np.zeros_like(X, bool)
        for s in sh[1:]:
            out |= _shape(X, Y, s)
        return out
    raise ValueError(kind)


def _poly(X, Y, pts):
    inside = np.zeros_like(X, bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cond = ((y1 > Y) != (y2 > Y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (Y - y1) / (y2 - y1) + x1
        inside ^= cond & (X < xint)
    return inside


def _stroke(X, Y, pts, w):
    m = np.zeros_like(X, bool)
    hw = w / 2.0
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 == 0:
            d2 = (X - x1) ** 2 + (Y - y1) ** 2
        else:
            t = np.clip(((X - x1) * dx + (Y - y1) * dy) / L2, 0, 1)
            px, py = x1 + t * dx, y1 + t * dy
            d2 = (X - px) ** 2 + (Y - py) ** 2
        m |= d2 <= hw * hw
    # round caps at the ends
    for (x1, y1) in (pts[0], pts[-1]):
        m |= (X - x1) ** 2 + (Y - y1) ** 2 <= hw * hw
    return m


def _ss_mean(m):
    """(n,n) bool/float -> (n/S,n/S) float coverage in [0,1]"""
    n = m.shape[0] // S
    a = m.astype(np.float32)
    return a.reshape(n, S, n, S).mean(axis=(1, 3))


# ---------------------------------------------------------------- rotation
def _rot_pts(pts, a):
    ca, sa = np.cos(a), np.sin(a)
    return [(x * ca - y * sa, x * sa + y * ca) for (x, y) in pts]


def rot_shape(sh, a):
    k = sh[0]
    if k == "disc":
        (cx, cy) = _rot_pts([(sh[1], sh[2])], a)[0]
        return ("disc", cx, cy, sh[3])
    if k == "ring":
        (cx, cy) = _rot_pts([(sh[1], sh[2])], a)[0]
        return ("ring", cx, cy, sh[3], sh[4])
    if k == "rect":
        x0, y0, x1, y1 = sh[1:]
        pts = _rot_pts([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], a)
        return ("poly", pts)
    if k == "poly":
        return ("poly", _rot_pts(sh[1], a))
    if k == "stroke":
        return ("stroke", _rot_pts(sh[1], a), sh[2])
    if k == "sub":
        return ("sub", rot_shape(sh[1], a), rot_shape(sh[2], a))
    if k == "union":
        return ("union",) + tuple(rot_shape(s, a) for s in sh[1:])
    raise ValueError(k)


def rot_parts(p, a):
    q = Parts()
    for (ch, val, shape, clip) in p.parts:
        q.add(ch, val, rot_shape(shape, a),
              rot_shape(clip, a) if clip is not None else None)
    return q


# ---------------------------------------------------------------- templates
class Parts(object):
    """collected drawing parts of one template"""

    def __init__(self):
        self.parts = []     # (channel, value, shape, clip_shape_or_None)
        self.support = None  # shape or list of shapes (union)

    def add(self, ch, val, shape, clip=None):
        self.parts.append((ch, val, shape, clip))

    def supp(self, *shapes):
        self.support = shapes[0] if len(shapes) == 1 else ("union",) + tuple(shapes)


def make_kernel(canvas, D, parts, normalise=True):
    """Parts -> (CH, canvas, canvas) float kernel.

    The support is the pattern footprint itself.  The kernel is zero-mean over
    that footprint (so it is orthogonal to any flat/constant background) and
    unit L2 there; entries outside are exactly zero.
    """
    maps = np.zeros((CH, canvas * S, canvas * S), np.float32)
    for (ch, val, shape, clip) in parts.parts:
        m = raster(canvas, D, shape)
        if clip is not None:
            m &= raster(canvas, D, clip)
        maps[ch] += val * m
    k = np.stack([_ss_mean(maps[c]) for c in range(CH)])
    if not normalise:
        return k
    sup = np.abs(k).sum(0) > 0.03
    supp = np.repeat(sup[None], CH, 0)
    flat = k.reshape(CH, -1)
    supp3 = supp.reshape(CH, -1)
    vals = flat[supp3]
    if len(vals) > 0 and vals.std() > 0:
        flat[supp3] = vals - vals.mean()
        vals = flat[supp3]
        nrm = np.sqrt((vals ** 2).sum())
        if nrm > 0:
            flat[supp3] = vals / nrm
    return flat.reshape(CH, canvas, canvas) * supp


# ---------------------------------------------------------------- sign shapes
# outer geometry, in sign units (D = diameter)
def ring_annulus():
    return ("ring", 0, 0, 0.385, 0.50)


def ring_inner(r=0.35):
    return ("disc", 0, 0, r)


def tri_up_pts():
    return [(0, -0.48), (-0.46, 0.48), (0.46, 0.48)]


def tri_down_pts():
    return [(0, 0.48), (-0.46, -0.48), (0.46, -0.48)]


def tri_up_outer():
    return ("poly", tri_up_pts() + tri_up_pts()[:1])


def tri_band(pts, t):
    """band along the edges of a closed triangle: outer poly minus inner (inset) poly"""
    cx = sum(p[0] for p in pts) / 3.0
    cy = sum(p[1] for p in pts) / 3.0
    inner = []
    for (x, y) in pts:
        # push vertex toward centroid so that edge moves in by ~t
        vx, vy = x - cx, y - cy
        L = np.hypot(vx, vy)
        f = max(0.05, 1.0 - t / L * 1.55)
        inner.append((cx + vx * f, cy + vy * f))
    return ("sub", ("poly", pts), ("poly", inner))


def tri_up_inner():
    return tri_band(tri_up_pts(), 0.135)[2]


def tri_down_inner():
    return tri_band(tri_down_pts(), 0.175)[2]


def diamond_pts():
    return [(0, -0.50), (-0.50, 0), (0, 0.50), (0.50, 0)]


def oct_poly():
    pts = []
    for a in range(8):
        th = np.pi / 8 + a * np.pi / 4
        pts.append((0.47 * np.cos(th), 0.47 * np.sin(th)))
    return pts


# ---------------------------------------------------------------- family templates
def fam_ring():
    p = Parts()
    p.add(C_RG, 1.2, ring_annulus())
    p.add(C_RB, 0.6, ring_annulus())
    p.add(C_RG, -1.2, ring_inner(0.34))       # interior must NOT be red
    p.add(C_YEL, -2.5, ring_annulus())
    p.add(C_BR, -1.2, ring_annulus())        # not yellow
    p.add(C_GRAY, 0.35, ring_inner(0.34))
    p.add(C_GRAY, -0.30, ring_annulus())
    p.supp(("disc", 0, 0, 0.56))
    return p


def fam_discR():
    p = Parts()
    d = ("disc", 0, 0, 0.50)
    p.add(C_RG, 1.0, d)
    p.add(C_RB, 0.6, d)
    p.add(C_YEL, -2.5, d)
    p.add(C_BR, -1.2, d)
    p.add(C_RG, -1.0, ("ring", 0, 0, 0.50, 0.56))
    p.add(C_GRAY, -0.35, d)
    p.add(C_GRAY, -0.8, ("disc", 0, 0, 0.30))    # centre is red, not white
    p.add(C_BRI, -0.7, ("sub", ("poly", [(0.95 * x, 0.95 * y) for (x, y) in oct_poly()]),
                        ("poly", [(0.80 * x, 0.80 * y) for (x, y) in oct_poly()])))  # no white rim
    p.supp(("disc", 0, 0, 0.56))
    return p


def fam_oct():
    p = Parts()
    oct_ = ("poly", oct_poly())
    oct_in = ("poly", [(0.85 * x, 0.85 * y) for (x, y) in oct_poly()])
    p.add(C_RG, 1.0, oct_)
    p.add(C_RB, 0.6, oct_)
    p.add(C_YEL, -2.5, oct_)
    p.add(C_BR, -1.2, oct_)
    p.add(C_RG, -1.0, ("sub", ("disc", 0, 0, 0.56), oct_))
    p.add(C_GRAY, -0.8, ("disc", 0, 0, 0.30))    # centre is red, not white
    p.add(C_BRI, 0.7, ("sub", ("poly", [(0.97 * x, 0.97 * y) for (x, y) in oct_poly()]),
                       ("poly", [(0.80 * x, 0.80 * y) for (x, y) in oct_poly()])))
    p.supp(("disc", 0, 0, 0.56))
    return p


def fam_triUp():
    p = Parts()
    band = tri_band(tri_up_pts(), 0.105)
    inner = tri_up_inner()
    p.add(C_RG, 1.0, band)
    p.add(C_RB, 0.55, band)
    p.add(C_YEL, -2.5, band)
    p.add(C_BR, -1.2, band)
    p.add(C_GRAY, 0.60, inner)
    p.add(C_GRAY, -0.45, band)
    p.supp(("poly", [(1.09 * x, 1.07 * y) for (x, y) in tri_up_pts()]))
    return p


def fam_triDown():
    p = Parts()
    band = tri_band(tri_down_pts(), 0.150)
    inner = tri_down_inner()
    p.add(C_RG, 1.0, band)
    p.add(C_RB, 0.55, band)
    p.add(C_YEL, -2.5, band)
    p.add(C_BR, -1.2, band)
    p.add(C_RG, -1.5, ("disc", 0, -0.44, 0.15))    # no red at the top centre (a ring has)
    p.add(C_GRAY, 0.60, inner)
    p.add(C_GRAY, -0.45, band)
    p.supp(("poly", [(1.10 * x, 1.08 * y) for (x, y) in tri_down_pts()]))
    return p


def fam_diamond():
    p = Parts()
    dia = ("poly", diamond_pts())
    dia_in = ("poly", [(0.80 * x, 0.80 * y) for (x, y) in diamond_pts()])
    rim = ("sub", ("poly", [(0.94 * x, 0.94 * y) for (x, y) in diamond_pts()]),
           ("poly", [(0.72 * x, 0.72 * y) for (x, y) in diamond_pts()]))
    p.add(C_YEL, 1.2, dia)
    p.add(C_RG, -2.5, dia)
    p.add(C_BR, -1.2, dia)
    p.add(C_GRAY, 0.15, dia_in)
    p.add(C_BRI, 0.5, rim)
    p.supp(("poly", [(1.10 * x, 1.08 * y) for (x, y) in diamond_pts()]))
    return p


def _corners(r=0.51, cr=0.11):
    """four blobs at the corners of a square of half-side 0.36 (outside a disc)"""
    q = 0.36
    return ("union", ("disc", -q, -q, cr), ("disc", q, -q, cr),
            ("disc", -q, q, cr), ("disc", q, q, cr))


def fam_discB():
    p = Parts()
    p.add(C_BR, 1.0, ("disc", 0, 0, 0.49))
    p.add(C_BG, 0.5, ("disc", 0, 0, 0.49))
    p.add(C_RG, -1.5, ("disc", 0, 0, 0.49))     # no red
    p.add(C_YEL, -1.5, ("disc", 0, 0, 0.49))    # no yellow
    p.add(C_BR, -1.0, ("ring", 0, 0, 0.50, 0.56))
    p.add(C_BR, -1.4, _corners())               # no blue in square corners
    p.add(C_GRAY, -0.20, ("disc", 0, 0, 0.49))
    p.add(C_BRI, -0.6, ("disc", 0, 0, 0.38))     # a blue disc is not bright white
    p.supp(("disc", 0, 0, 0.56))
    return p


def fam_squareB():
    p = Parts()
    sq = ("rect", -0.44, -0.44, 0.44, 0.44)
    p.add(C_BR, 1.0, sq)
    p.add(C_BG, 0.5, sq)
    p.add(C_RG, -1.5, sq)                        # no red
    p.add(C_YEL, -1.5, sq)                       # no yellow
    p.add(C_BR, -1.2, ("sub", ("rect", -0.52, -0.52, 0.52, 0.52), ("rect", -0.47, -0.47, 0.47, 0.47)))
    p.add(C_BR, 1.4, _corners())                # blue square must have blue corners
    p.add(C_GRAY, -0.20, sq)
    p.add(C_BRI, -0.6, ("rect", -0.34, -0.34, 0.34, 0.34))   # not bright white
    p.supp(("rect", -0.56, -0.56, 0.56, 0.56))
    return p


def fam_discW():
    p = Parts()
    p.add(C_GRAY, 1.0, ("disc", 0, 0, 0.42))
    p.add(C_BR, -2.0, ("disc", 0, 0, 0.44))
    p.add(C_RG, -3.0, ("disc", 0, 0, 0.50))     # no red anywhere over the disc
    p.add(C_RG, -1.0, ("ring", 0, 0, 0.50, 0.56))
    p.add(C_YEL, -1.5, ("disc", 0, 0, 0.42))
    p.supp(("disc", 0, 0, 0.52))
    return p


FAMILIES = {
    "fring": (fam_ring, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]),
    "fdiscR": (fam_discR, [17]),
    "foct": (fam_oct, [14]),
    "ftriUp": (fam_triUp, [11] + list(range(18, 32))),
    "ftriDown": (fam_triDown, [13]),
    "fdiamond": (fam_diamond, [12]),
    "fdiscB": (fam_discB, [33, 34, 35, 38, 39, 40]),
    "fsquareB": (fam_squareB, [36, 37]),
    "fdiscW": (fam_discW, [6, 32, 41, 42]),
}


# ---------------------------------------------------------------- digit font
# strokes in a unit box (x in [0,1] width, y in [0,1] height, y down)
DIGITS = {
    "0": ([(0.5 + 0.30 * np.cos(a), 0.5 + 0.44 * np.sin(a)) for a in
           np.linspace(0, 2 * np.pi, 18)], True),
    "1": ([(0.34, 0.24), (0.52, 0.08), (0.52, 0.92)], False),
    "2": ([(0.15, 0.32), (0.24, 0.12), (0.50, 0.05), (0.74, 0.18), (0.72, 0.36),
           (0.32, 0.66), (0.14, 0.92), (0.86, 0.90)], False),
    "3": ([(0.16, 0.18), (0.44, 0.06), (0.70, 0.17), (0.55, 0.38), (0.75, 0.58),
           (0.66, 0.85), (0.34, 0.94), (0.12, 0.80)], False),
    "5": ([(0.80, 0.08), (0.24, 0.10), (0.21, 0.46), (0.54, 0.38), (0.80, 0.58),
           (0.68, 0.87), (0.30, 0.94), (0.11, 0.77)], False),
    "6": ([(0.72, 0.08), (0.32, 0.26), (0.20, 0.58), (0.36, 0.90), (0.66, 0.90),
           (0.80, 0.64), (0.64, 0.45), (0.36, 0.45), (0.22, 0.60)], False),
    "7": ([(0.16, 0.10), (0.84, 0.10), (0.40, 0.92)], False),
    "8": ([(0.50, 0.06), (0.69, 0.18), (0.57, 0.34), (0.76, 0.58), (0.65, 0.87),
           (0.50, 0.94), (0.35, 0.87), (0.24, 0.58), (0.43, 0.34), (0.31, 0.18)], True),
}


def number_strokes(num, h, rot=0.0):
    """return list of (pts_in_sign_units, width) for a multi-digit number, height h (D units)"""
    w = 0.58 * h
    gap = 0.06 * h
    W = len(num) * w + (len(num) - 1) * gap
    out = []
    ca, sa = np.cos(rot), np.sin(rot)
    for k, ch_ in enumerate(num):
        xoff = -W / 2.0 + k * (w + gap) + w / 2.0
        pts, closed = DIGITS[ch_]
        q = []
        for (u, v) in pts:
            x = xoff + (u - 0.5) * w
            y = (v - 0.5) * h
            q.append((x * ca - y * sa, x * sa + y * ca))
        if closed:
            q = q + q[:1]
        out.append((q, 0.19 * h))
    return out


def add_ink_parts(p, strokes, val=-1.0, ch=C_GRAY):
    for (pts, w) in strokes:
        p.add(ch, val, ("stroke", pts, w))


# ---------------------------------------------------------------- pictograms
# inner regions
SUP_RING = ("disc", 0, 0, 0.36)
SUP_TRI = ("poly", [(0, -0.29), (-0.32, 0.38), (0.32, 0.38)])
WHITE_TRI = SUP_TRI


def pic_blankdisc():
    """class 15: empty white disc"""
    p = Parts()
    for (v, c, sh) in _discW_excl():
        p.add(c, v, sh)
    p.add(C_GRAY, 0.7, ("disc", 0, 0, 0.32))
    p.add(C_INK, -2.0, ("disc", 0, 0, 0.32))   # no dark pictogram inside
    return p


def car_strokes(x0, y0, s):
    """simple side-view car; returns list of shapes (ink)"""
    roof = ("poly", [(x0 - 0.07 * s, y0 - 0.02 * s), (x0 - 0.035 * s, y0 - 0.12 * s),
                     (x0 + 0.055 * s, y0 - 0.12 * s), (x0 + 0.095 * s, y0 - 0.02 * s)])
    body = ("rect", x0 - 0.13 * s, y0 - 0.02 * s, x0 + 0.13 * s, y0 + 0.07 * s)
    wheels = ("union", ("disc", x0 - 0.075 * s, y0 + 0.075 * s, 0.045 * s),
              ("disc", x0 + 0.075 * s, y0 + 0.075 * s, 0.045 * s))
    return [roof, body, wheels]


def truck_strokes(x0, y0, s):
    """truck facing left; cab at left, box at right"""
    box = ("rect", x0 + 0.03 * s, y0 - 0.17 * s, x0 + 0.28 * s, y0 + 0.05 * s)
    cab = ("poly", [(x0 - 0.13 * s, y0 + 0.05 * s), (x0 - 0.13 * s, y0 - 0.06 * s),
                    (x0 - 0.05 * s, y0 - 0.10 * s), (x0 + 0.03 * s, y0 - 0.07 * s),
                    (x0 + 0.03 * s, y0 + 0.05 * s)])
    wheels = ("union",
              ("disc", x0 - 0.075 * s, y0 + 0.075 * s, 0.042 * s),
              ("disc", x0 + 0.10 * s, y0 + 0.075 * s, 0.042 * s),
              ("disc", x0 + 0.22 * s, y0 + 0.075 * s, 0.042 * s))
    return [box, cab, wheels]


def pic_nopass():
    """class 9: black car left, red car right"""
    p = Parts()
    for sh in car_strokes(-0.115, 0.01, 0.60):
        p.add(C_GRAY, -1.0, sh)
    for sh in car_strokes(0.115, 0.01, 0.66):
        p.add(C_RG, 1.1, sh)          # right car is RED
        p.add(C_GRAY, -0.7, sh)
    p.add(C_GRAY, 0.5, ("sub", ("disc", 0, 0, 0.34), ("disc", 0, 0, 0.30)))
    p.supp(SUP_RING)
    return p


def pic_nopass_truck():
    """class 10: black car left, black truck right"""
    p = Parts()
    for sh in car_strokes(-0.135, 0.035, 0.52):
        p.add(C_GRAY, -1.0, sh)
    for sh in truck_strokes(0.075, -0.01, 0.60):
        p.add(C_GRAY, -1.0, sh)
    p.add(C_GRAY, 0.45, ("sub", ("disc", 0, 0, 0.34), ("disc", 0, 0, 0.30)))
    p.supp(SUP_RING)
    return p


def pic_notrucks():
    """class 16: single black truck"""
    p = Parts()
    for sh in truck_strokes(-0.02, 0.0, 0.78):
        p.add(C_GRAY, -1.0, sh)
    p.add(C_GRAY, 0.45, ("sub", ("disc", 0, 0, 0.34), ("disc", 0, 0, 0.30)))
    p.supp(SUP_RING)
    return p


# --- warning pictograms (ink shapes in the inner triangle box
#     x in [-0.28,0.28], y in [-0.24,0.35], sign units) ---
def wp_exclaim():
    p = Parts()
    add_ink_parts(p, [([(0.0, -0.24), (0.0, 0.05)], 0.11)])
    p.add(C_GRAY, -1.0, ("disc", 0, 0.155, 0.062))
    p.add(C_GRAY, 0.5, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_bend(dirleft):
    """L-shaped stroke: shaft on the right, arm at the top pointing left (bend left)"""
    sgn = 1.0 if dirleft else -1.0     # bend left -> shaft right, arm points left
    p = Parts()
    add_ink_parts(p, [([(sgn * 0.145, 0.33), (sgn * 0.145, 0.075), (sgn * -0.06, 0.075)], 0.10)])
    p.add(C_GRAY, 0.5, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_double():
    p = Parts()
    add_ink_parts(p, [([(-0.09, 0.33), (-0.09, 0.16), (0.09, 0.02), (0.09, -0.16)], 0.10)])
    p.add(C_GRAY, 0.5, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_bumpy():
    p = Parts()
    car = car_strokes(0.0, -0.13, 0.42)
    for sh in car:
        p.add(C_GRAY, -1.0, sh)
    add_ink_parts(p, [([(-0.16, 0.30), (-0.09, 0.20), (-0.02, 0.30)], 0.055),
                      ([(0.02, 0.30), (0.09, 0.20), (0.16, 0.30)], 0.055),
                      ([(-0.16, 0.30), (0.16, 0.30)], 0.045)])
    p.add(C_GRAY, 0.5, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_slippery():
    p = Parts()
    car = car_strokes(0.02, -0.19, 0.40)
    for sh in car:
        p.add(C_GRAY, -1.0, sh)
    add_ink_parts(p, [([(-0.14, 0.34), (-0.145, 0.26), (-0.09, 0.18), (-0.085, 0.10), (-0.03, 0.02)], 0.04),
                      ([(0.04, 0.34), (0.045, 0.26), (0.10, 0.18), (0.095, 0.10), (0.04, 0.02)], 0.04)])
    p.add(C_GRAY, 0.5, WHITE_TRI)
    return p


def wp_narrows():
    p = Parts()
    add_ink_parts(p, [([(0.0, -0.24), (0.0, 0.20)], 0.055),
                      ([(0.115, -0.24), (0.115, 0.20)], 0.05)])
    p.add(C_GRAY, 0.5, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_works():
    p = Parts()
    p.add(C_GRAY, -1.0, ("disc", -0.07, -0.26, 0.055))
    add_ink_parts(p, [([(-0.05, -0.19), (-0.26, 0.22)], 0.085),        # body, diagonal down-left
                      ([(-0.26, 0.22), (-0.14, 0.34)], 0.06),          # leg
                      ([(-0.02, -0.12), (0.09, 0.02)], 0.05),          # arm
                      ([(0.09, 0.02), (0.17, 0.18)], 0.045)])          # shovel
    p.add(C_GRAY, -1.0, ("poly", [(0.11, 0.34), (0.30, 0.34), (0.28, 0.20), (0.14, 0.20)]))  # mound
    p.add(C_GRAY, 0.5, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_lights():
    p = Parts()
    p.add(C_GRAY, -1.0, ("ring", 0, -0.04, 0.13, 0.165))
    for yy in (-0.145, -0.05, 0.045):
        p.add(C_GRAY, -1.0, ("disc", 0, yy, 0.045))
    p.add(C_GRAY, -1.0, ("rect", -0.02, 0.14, 0.02, 0.34))
    p.add(C_GRAY, 0.5, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_pedestrian():
    p = Parts()
    p.add(C_GRAY, -1.0, ("disc", 0.02, -0.215, 0.05))
    add_ink_parts(p, [([(0.02, -0.16), (-0.01, 0.02)], 0.07),
                      ([(-0.01, 0.02), (0.10, 0.30)], 0.06),
                      ([(-0.01, 0.02), (-0.11, 0.28)], 0.06),
                      ([(0.02, -0.14), (0.13, -0.02)], 0.05)])
    for i, xx in enumerate((-0.26, -0.13, 0.0)):
        p.add(C_GRAY, -1.0, ("rect", xx, 0.30 + (i % 2) * 0.0, xx + 0.09, 0.345))
    p.add(C_GRAY, 0.45, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_children():
    p = Parts()
    p.add(C_GRAY, -1.0, ("disc", -0.11, -0.225, 0.06))
    add_ink_parts(p, [([(-0.10, -0.14), (-0.12, 0.04)], 0.075),
                      ([(-0.12, 0.04), (-0.03, 0.32)], 0.06),
                      ([(-0.12, 0.04), (-0.20, 0.30)], 0.06),
                      ([(-0.10, -0.12), (0.02, -0.02)], 0.05)])
    p.add(C_GRAY, -1.0, ("disc", 0.10, -0.10, 0.042))
    add_ink_parts(p, [([(0.10, -0.05), (0.08, 0.10)], 0.065),
                      ([(0.08, 0.10), (0.15, 0.32)], 0.055),
                      ([(0.08, 0.10), (0.01, 0.30)], 0.055),
                      ([(0.10, -0.04), (0.19, 0.04)], 0.045)])
    p.add(C_GRAY, 0.45, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_cycle():
    p = Parts()
    p.add(C_GRAY, -1.0, ("ring", -0.125, 0.20, 0.075, 0.10))
    p.add(C_GRAY, -1.0, ("ring", 0.125, 0.20, 0.075, 0.10))
    add_ink_parts(p, [([(-0.125, 0.11), (0.02, 0.10)], 0.05),
                      ([(0.02, 0.10), (0.125, 0.11)], 0.05),
                      ([(0.02, 0.10), (0.0, -0.01)], 0.05),
                      ([(0.0, -0.01), (-0.09, 0.0)], 0.05),
                      ([(0.0, -0.01), (0.07, -0.03)], 0.045),
                      ([(-0.09, 0.0), (-0.125, 0.11)], 0.04),
                      ([(0.07, -0.03), (0.125, 0.11)], 0.04)])
    p.add(C_GRAY, 0.5, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_snow():
    p = Parts()
    add_ink_parts(p, [([(0.0, -0.28), (0.0, 0.19)], 0.055)])
    for a in np.radians([60, 120]):
        dx, dy = np.cos(a), np.sin(a)
        add_ink_parts(p, [([(-0.17 * dx, -0.15 * dy - 0.045), (0.17 * dx, 0.15 * dy - 0.045)], 0.05)])
    p.add(C_GRAY, 0.5, WHITE_TRI)
    return p


def wp_deer():
    p = Parts()
    p.add(C_GRAY, -1.0, ("disc", 0.03, -0.24, 0.065))                   # head, top centre
    add_ink_parts(p, [([(0.0, -0.18), (-0.10, -0.04)], 0.06),           # neck down-left
                      ([(-0.14, -0.02), (0.14, -0.02), (0.14, 0.10), (-0.14, 0.10)], None)][:1] if False else
                      [([(-0.15, -0.02), (0.15, -0.02)], 0.10),        # back
                      ([(-0.15, 0.08), (0.13, 0.08)], 0.10),           # belly
                      ([(-0.12, 0.06), (-0.24, 0.28)], 0.045),          # front leg
                      ([(-0.07, 0.08), (-0.15, 0.32)], 0.045),
                      ([(0.09, 0.06), (0.22, 0.26)], 0.045),            # hind leg
                      ([(0.14, 0.08), (0.26, 0.32)], 0.045),
                      ([(0.15, -0.04), (0.25, -0.16)], 0.04)])          # tail up
    p.add(C_GRAY, 0.5, WHITE_TRI)
    p.supp(SUP_TRI)
    return p


def wp_cross():
    """class 11: very thick X filling the triangle"""
    p = Parts()
    add_ink_parts(p, [([(-0.26, -0.19), (0.26, 0.34)], 0.13),
                      ([(0.26, -0.19), (-0.26, 0.34)], 0.13)])
    p.add(C_GRAY, 0.35, WHITE_TRI)
    return p


WARNING_PICS = {
    11: wp_cross, 18: wp_exclaim,
    19: lambda: wp_bend(True), 20: lambda: wp_bend(False), 21: wp_double,
    22: wp_bumpy, 23: wp_slippery, 24: wp_narrows, 25: wp_works, 26: wp_lights,
    27: wp_pedestrian, 28: wp_children, 29: wp_cycle, 30: wp_snow, 31: wp_deer,
}


# --- blue arrow pictograms (white marks; drawn as ink then sign-flipped) ---
def arrow_parts(pts_list, widths, extra_shapes=(), flip=True):
    p = Parts()
    v = -1.0 if flip else 1.0
    for (pts, w) in zip(pts_list, widths):
        add_ink_parts(p, [(pts, w)], val=v)
    for sh in extra_shapes:
        p.add(C_GRAY, v, sh)
    return p


def pic_turn(dirright):
    sgn = 1.0 if dirright else -1.0
    p = Parts()
    p.add(C_BR, -0.9, _corners())                # no blue in corners (it's a disc)
    shaft = [(sgn * -0.21, 0.34), (sgn * -0.21, 0.05)]
    arc = [(sgn * -0.21, 0.05), (sgn * -0.17, -0.10), (sgn * -0.07, -0.17), (sgn * 0.05, -0.17)]
    head = ("poly", [(sgn * 0.03, -0.28), (sgn * 0.03, -0.06), (sgn * 0.26, -0.17)])
    add_ink_parts(p, [(shaft, 0.11), (arc, 0.11)], val=1.0)
    p.add(C_GRAY, 1.0, head)
    p.add(C_GRAY, -0.45, ("disc", 0, 0, 0.34))
    p.supp(("disc", 0, 0, 0.38))
    return p


def pic_ahead():
    p = Parts()
    p.add(C_BR, -0.9, _corners())                # no blue in corners (it's a disc)
    add_ink_parts(p, [([(-0.055, 0.34), (-0.055, -0.06)], 0.10),
                      ([(0.055, 0.34), (0.055, -0.06)], 0.10)], val=1.0)
    p.add(C_GRAY, 1.0, ("poly", [(-0.13, -0.06), (0.13, -0.06), (0.0, -0.32)]))
    p.add(C_GRAY, -0.45, ("disc", 0, 0, 0.34))
    p.supp(("disc", 0, 0, 0.38))
    return p


def pic_keep(dirright):
    sgn = 1.0 if dirright else -1.0
    p = Parts()
    p.add(C_BR, -0.9, _corners())                # no blue in corners (it's a disc)
    add_ink_parts(p, [([(sgn * -0.19, 0.33), (sgn * 0.09, 0.02), (sgn * 0.09, -0.10)], 0.10)], val=1.0)
    p.add(C_GRAY, 1.0, ("poly", [(sgn * 0.0, -0.09), (sgn * 0.19, -0.09), (sgn * 0.095, -0.30)]))
    p.add(C_GRAY, -0.45, ("disc", 0, 0, 0.34))
    p.supp(("disc", 0, 0, 0.38))
    return p


def pic_roundabout():
    p = Parts()
    p.add(C_BR, -0.9, _corners())                # no blue in corners (it's a disc)
    r = 0.185
    for a0 in np.radians([-90, 30, 150]):
        a1 = a0 + np.radians(105)
        pts = [(r * np.cos(a0 + t), r * np.sin(a0 + t)) for t in np.linspace(0, np.radians(105), 10)]
        add_ink_parts(p, [(pts, 0.06)], val=1.0)
        hx, hy = r * np.cos(a1), r * np.sin(a1)
        tx, ty = -np.sin(a1), np.cos(a1)
        p.add(C_GRAY, 1.0, ("poly", [(hx - 0.02 * tx - 0.05 * tx, hy - 0.02 * ty - 0.05 * ty),
                                     (hx - 0.02 * tx + 0.05 * tx, hy - 0.02 * ty + 0.05 * ty),
                                     (hx + 0.085 * np.cos(a1), hy + 0.085 * np.sin(a1))]))
    p.add(C_GRAY, -0.45, ("disc", 0, 0, 0.34))
    p.supp(("disc", 0, 0, 0.38))
    return p


def pic_square_arrows(dirright):
    sgn = 1.0 if dirright else -1.0
    p = Parts()
    p.add(C_BR, 1.2, _corners())                 # blue square corners
    add_ink_parts(p, [([(-0.11, 0.33), (-0.11, -0.04)], 0.09)], val=1.0)
    p.add(C_GRAY, 1.0, ("poly", [(-0.20, -0.05), (-0.02, -0.05), (-0.11, -0.27)]))
    add_ink_parts(p, [([(sgn * 0.10, 0.33), (sgn * 0.10, 0.10)], 0.09),
                      ([(sgn * 0.10, 0.10), (sgn * 0.23, 0.10)], 0.09)], val=1.0)
    p.add(C_GRAY, 1.0, ("poly", [(sgn * 0.155, -0.01), (sgn * 0.155, 0.21), (sgn * 0.30, 0.10)]))
    p.add(C_GRAY, -0.40, ("rect", -0.36, -0.36, 0.36, 0.36))
    p.supp(("rect", -0.42, -0.42, 0.42, 0.42))
    return p


BLUE_PICS = {
    33: lambda: pic_turn(True), 34: lambda: pic_turn(False), 35: pic_ahead,
    38: lambda: pic_keep(True), 39: lambda: pic_keep(False), 40: pic_roundabout,
    36: lambda: pic_square_arrows(True), 37: lambda: pic_square_arrows(False),
}


# --- white-disc family pictograms ---
def stripes(n, t, clip_r=0.40):
    out = []
    for i in range(n):
        x0 = (i - (n - 1) / 2.0) * 0.115
        out.append((("stroke", [(x0 - 0.30, 0.30), (x0 + 0.30, -0.30)], t), ("disc", 0, 0, clip_r)))
    return out


def _discW_excl():
    """white-disc context: no red / blue / yellow over the disc"""
    return [(-3.0, C_RG, ("disc", 0, 0, 0.44)), (-2.0, C_BR, ("disc", 0, 0, 0.44)),
            (-1.5, C_YEL, ("disc", 0, 0, 0.44))]


def pic_end80():
    p = Parts()
    for (v, c, sh) in _discW_excl():
        p.add(c, v, sh)
    add_ink_parts(p, number_strokes("80", 0.34))
    for (sh, clip) in stripes(4, 0.035):
        p.add(C_GRAY, -0.85, sh, clip=clip)
    p.add(C_GRAY, 0.55, ("disc", 0, 0, 0.38))
    p.supp(("disc", 0, 0, 0.42))
    return p


def pic_endall():
    p = Parts()
    for (v, c, sh) in _discW_excl():
        p.add(c, v, sh)
    for (sh, clip) in stripes(5, 0.035):
        p.add(C_GRAY, -1.3, sh, clip=clip)
    p.add(C_GRAY, 0.55, ("disc", 0, 0, 0.38))
    p.supp(("disc", 0, 0, 0.42))
    return p


def pic_endnopass():
    p = Parts()
    for (v, c, sh) in _discW_excl():
        p.add(c, v, sh)
    for (sh, clip) in stripes(2, 0.05):
        p.add(C_GRAY, -1.0, sh, clip=clip)
    p.add(C_GRAY, 0.55, ("disc", 0, 0, 0.38))
    p.supp(("disc", 0, 0, 0.42))
    return p


def pic_endnopass_truck():
    p = Parts()
    for (v, c, sh) in _discW_excl():
        p.add(c, v, sh)
    for (sh, clip) in stripes(2, 0.05):
        p.add(C_GRAY, -0.9, sh, clip=clip)
    for sh in truck_strokes(-0.16, -0.02, 0.42):
        p.add(C_GRAY, -0.9, sh, clip=("disc", 0, 0, 0.40))
    for sh in car_strokes(0.16, 0.10, 0.34):
        p.add(C_GRAY, -0.9, sh, clip=("disc", 0, 0, 0.40))
    p.add(C_GRAY, 0.55, ("disc", 0, 0, 0.38))
    p.supp(("disc", 0, 0, 0.42))
    return p


def pic_diamond():
    """class 12: yellow diamond interior"""
    p = Parts()
    dia = ("poly", [(0.82 * x, 0.82 * y) for (x, y) in diamond_pts()])
    p.add(C_YEL, 1.2, dia)
    p.add(C_RG, -2.0, dia)
    p.add(C_BR, -0.8, dia)
    p.add(C_GRAY, 0.15, dia)
    return p


def pic_yield_inner():
    """class 13: white inner inverted triangle with red border"""
    inner = tri_down_inner()
    band = tri_band(tri_down_pts(), 0.150)
    p = Parts()
    p.add(C_GRAY, 1.0, inner)
    p.add(C_RG, -0.9, inner)      # interior not red
    p.add(C_RG, 0.7, band)        # border is red
    p.add(C_RG, -1.2, ("disc", 0, -0.44, 0.15))   # no red at the top centre
    return p


# --- stop / no-entry marks ---
def pic_stopbar():
    """class 17: white horizontal bar on a red disc"""
    bar = ("rect", -0.31, -0.075, 0.31, 0.075)
    barexp = ("rect", -0.35, -0.115, 0.35, 0.115)
    p = Parts()
    p.add(C_BRI, 1.2, bar)
    p.add(C_GRAY, 1.0, bar)
    p.add(C_RG, 1.2, ("disc", 0, 0, 0.36))    # the interior itself must be red
    p.add(C_RG, -1.0, bar)
    p.add(C_YEL, -1.0, ("disc", 0, 0, 0.40))
    p.add(C_INK, -0.8, bar)      # no black ink over the bar
    return p


def _letter(ch, x0, y0, h):
    """tiny block letter, returns shapes in sign units; letter box w=0.55h"""
    w = 0.55 * h
    t = 0.16 * h
    def R(a, b, c, d):  # rect by corners (in letter units)
        return ("rect", x0 + a * w, y0 + b * h, x0 + c * w, y0 + d * h)
    if ch == "S":
        return [R(0.1, 0.05, 0.9, 0.25), R(0.1, 0.05, 0.3, 0.5),
                R(0.1, 0.42, 0.9, 0.6), R(0.7, 0.5, 0.9, 0.95), R(0.1, 0.78, 0.9, 0.95)]
    if ch == "T":
        return [R(0.05, 0.05, 0.95, 0.25), R(0.4, 0.25, 0.6, 0.95)]
    if ch == "O":
        return [("ring", x0 + 0.5 * w, y0 + 0.5 * h, 0.22 * h, 0.38 * h)]
    if ch == "P":
        return [R(0.1, 0.05, 0.3, 0.95), R(0.1, 0.05, 0.9, 0.25),
                R(0.6, 0.25, 0.9, 0.6), R(0.1, 0.42, 0.9, 0.6)]
    return []


def pic_stoptext():
    p = Parts()
    p.add(C_RG, 1.2, ("disc", 0, 0, 0.34))    # the interior itself must be red
    p.add(C_YEL, -1.0, ("disc", 0, 0, 0.40))
    h = 0.29
    w = 0.55 * h
    total = 4 * w + 3 * 0.032
    band = ("rect", -total / 2.0 - 0.03, -h / 2.0 - 0.03, total / 2.0 + 0.03, h / 2.0 + 0.03)
    letters = []
    x = -total / 2.0
    for ch_ in "STOP":
        for sh in _letter(ch_, x, -h / 2.0, h):
            p.add(C_BRI, 1.2, sh)
            letters.append(sh)
        x += w + 0.032
    p.add(C_BRI, -0.9, ("sub", band, ("union",) + tuple(letters)))   # gaps are red, not white
    p.add(C_INK, -0.8, band)     # no black ink in the letter band
    return p
    p.supp(("disc", 0, 0, 0.40))
    return p


# ---------------------------------------------------------------- numbers
NUMBERS = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}


def number_pic(num, white_disc=False, rot=0.0, D=None):
    h = 0.42 if len(num) == 2 else 0.35
    p = Parts()
    add_ink_parts(p, number_strokes(num, h, rot))
    p.add(C_GRAY, 0.5, ("disc", 0, 0, 0.33))
    p.supp(("disc", 0, 0, 0.36))
    return p


def number_variants(num):
    out = []
    for D in NSCALES:
        for rot in ROTS:
            out.append((D, number_pic(num, rot=rot)))
    return out


def n80w_variants():
    out = []
    for D in NSCALES:
        for rot in ROTS:
            p = Parts()
            add_ink_parts(p, number_strokes("80", 0.36, rot))
            p.add(C_GRAY, 0.30, ("disc", 0, 0, 0.38))
            p.supp(("disc", 0, 0, 0.40))
            out.append((D, p))
    return out


NUM_LOGICALS = ["n" + NUMBERS[c] for c in sorted(NUMBERS)] + ["n80w"]


def number_bank_parts(name):
    if name == "n80w":
        return n80w_variants()
    num = name[1:]
    return number_variants(num)


# ---------------------------------------------------------------- composites
def merge(a, b):
    p = Parts()
    p.parts = a.parts + b.parts
    p.support = ("union", a.support, b.support)
    return p


def composite_parts(cls):
    fam_of = {}
    for fname, (_, members) in FAMILIES.items():
        for m in members:
            fam_of[m] = fname
    fname = fam_of[cls]
    fam_fn = FAMILIES[fname][0]
    if cls in NUMBERS:
        return lambda D: merge(fam_fn(), number_pic(NUMBERS[cls]))
    if cls == 6:
        return lambda D: merge(fam_fn(), pic_end80())
    if cls == 32:
        return lambda D: merge(fam_fn(), pic_endall())
    if cls == 41:
        return lambda D: merge(fam_fn(), pic_endnopass())
    if cls == 42:
        return lambda D: merge(fam_fn(), pic_endnopass_truck())
    if cls == 12:
        return lambda D: merge(fam_fn(), pic_diamond())
    if cls == 13:
        return lambda D: merge(fam_fn(), pic_yield_inner())
    if cls == 14:
        return lambda D: merge(fam_fn(), pic_stoptext())
    if cls == 17:
        return lambda D: merge(fam_fn(), pic_stopbar())
    if cls in WARNING_PICS:
        return lambda D: merge(fam_fn(), WARNING_PICS[cls]())
    if cls in BLUE_PICS:
        return lambda D: merge(fam_fn(), BLUE_PICS[cls]())
    if cls == 9:
        return lambda D: merge(fam_fn(), pic_nopass())
    if cls == 10:
        return lambda D: merge(fam_fn(), pic_nopass_truck())
    if cls == 15:
        return lambda D: merge(fam_fn(), pic_blankdisc())
    if cls == 16:
        return lambda D: merge(fam_fn(), pic_notrucks())
    raise ValueError(cls)


# picto logicals (name -> parts fn)
PICO_LOGICALS = (
    ["w%d" % c for c in sorted(WARNING_PICS)]
    + ["p9", "p10", "p15", "p16"]
    + ["a%d" % c for c in sorted(BLUE_PICS)]
    + ["e6", "e32", "e41", "e42"]
    + ["stopbar", "stoptext", "p12", "p13"]
)


def pico_parts(name):
    if name.startswith("w"):
        return lambda D: WARNING_PICS[int(name[1:])]()
    if name == "p9":
        return lambda D: pic_nopass()
    if name == "p10":
        return lambda D: pic_nopass_truck()
    if name == "p15":
        return lambda D: pic_blankdisc()
    if name == "p16":
        return lambda D: pic_notrucks()
    if name.startswith("a"):
        return lambda D: BLUE_PICS[int(name[1:])]()
    if name == "e6":
        return lambda D: pic_end80()
    if name == "e32":
        return lambda D: pic_endall()
    if name == "e41":
        return lambda D: pic_endnopass()
    if name == "e42":
        return lambda D: pic_endnopass_truck()
    if name == "stopbar":
        return lambda D: pic_stopbar()
    if name == "stoptext":
        return lambda D: pic_stoptext()
    if name == "p12":
        return lambda D: pic_diamond()
    if name == "p13":
        return lambda D: pic_yield_inner()
    raise ValueError(name)


PIC_OF_CLASS = {}   # class -> picto logical name
for c in WARNING_PICS:
    PIC_OF_CLASS[c] = "w%d" % c
PIC_OF_CLASS.update({9: "p9", 10: "p10", 15: "p15", 16: "p16",
                     6: "e6", 32: "e32", 41: "e41", 42: "e42",
                     14: "stoptext", 17: "stopbar",
                     12: "p12", 13: "p13"})
for c in BLUE_PICS:
    PIC_OF_CLASS[c] = "a%d" % c


# ---------------------------------------------------------------- global bank
def global_parts():
    gd = 24.5  # assumed sign size for fixed-position features
    r = 7.5 / gd * 2.0   # ~0.30 * D in sign units... r_px/D = 7.5/24.5
    r = 7.5 / 24.5
    g1 = Parts()  # red inside disc
    g1.add(C_RG, 1.0, ("disc", 0, 0, r))
    g1.supp(("disc", 0, 0, r))
    g2 = Parts()  # ink inside disc
    g2.add(C_INK, 1.0, ("disc", 0, 0, r))
    g2.supp(("disc", 0, 0, r))
    g3 = Parts()  # white horizontal bar
    bw, bh = 9.0 / gd, 3.0 / gd
    g3.add(C_BRI, 1.0, ("rect", -bw, -bh, bw, bh))
    g3.add(C_BRI, -0.8, ("sub", ("disc", 0, 0, 11.0 / gd), ("rect", -bw - 0.02, -bh - 0.02, bw + 0.02, bh + 0.02)))
    g3.supp(("disc", 0, 0, 11.0 / gd))
    return [("g_red_in", g1), ("g_ink_in", g2), ("g_bar", g3)]


# ---------------------------------------------------------------- fusion table
def build_fusion():
    fuse = {}   # feature name -> {class: weight}

    def add(f, c, w):
        fuse.setdefault(f, {})
        fuse[f][c] = fuse[f].get(c, 0.0) + w

    fam_members = {f: FAMILIES[f][1] for f in FAMILIES}
    fam_of = {}
    for f, ms in fam_members.items():
        for m in ms:
            fam_of[m] = f

    # family evidence + cross-family penalty
    for f, ms in fam_members.items():
        for c in range(43):
            add(f, c, 1.0 if c in ms else -0.40)
    # extra penalties between look-alike families (chosen from sign knowledge)
    ALIKE = [("fdiscW", "fring"), ("fdiscW", "ftriUp"), ("fdiscW", "fdiscB"),
             ("fdiscW", "fsquareB"), ("fdiscW", "foct"), ("fdiscW", "fdiscR"),
             ("foct", "fdiscR"), ("fdiscR", "fring"), ("foct", "fring"),
             ("fdiamond", "fsquareB"), ("fdiscB", "fsquareB"),
             ("ftriUp", "ftriDown"), ("fdiscW", "fdiamond")]
    for (fa, fb) in ALIKE:
        for c in fam_members[fa]:
            add(fb, c, -0.15)
        for c in fam_members[fb]:
            add(fa, c, -0.15)

    # numbers carry the within-ring discrimination: strong weight
    for cls, num in NUMBERS.items():
        nm = "n" + num
        for c in range(43):
            if c == cls:
                add(nm, c, 1.45)
            elif c in NUMBERS:
                add(nm, c, -0.08)
    add("n80w", 6, 0.90)
    add("n80w", 5, -0.30)
    add("n80", 6, -0.30)

    # pictogram weights: fine pictograms carry the discrimination; the
    # simple 'empty interior' pictograms are weak on purpose (they are broad)
    SIMPLE = (12, 13, 14, 17, 15)      # diamond / yield / stop / no-entry / blank
    for c, pic in PIC_OF_CLASS.items():
        for c2 in range(43):
            if c2 == c:
                w = 0.80 if c in SIMPLE else (1.30 if c in (6, 32, 41, 42) else
                                              (1.05 if c == 11 else
                                               (1.30 if c in (18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                              27, 28, 29, 30, 31) else 1.25)))
                add(pic, c2, w)
            elif c2 in PIC_OF_CLASS and fam_of.get(c2) == fam_of.get(c):
                add(pic, c2, -0.06)
    # extra rivalry between look-alike pictograms
    BLUE = [33, 34, 35, 36, 37, 38, 39, 40]
    for i, a in enumerate(BLUE):
        for b in BLUE[i + 1:]:
            add(PIC_OF_CLASS[a], b, -0.18)
            add(PIC_OF_CLASS[b], a, -0.18)
    for a, b in [(19, 20), (19, 21), (20, 21), (22, 23), (27, 28), (18, 26), (18, 30),
                 (41, 42), (32, 41), (32, 6), (41, 6), (9, 10), (15, 16), (6, 5),
                 (11, 25), (24, 26), (30, 11)]:
        if a in PIC_OF_CLASS and b in PIC_OF_CLASS:
            add(PIC_OF_CLASS[a], b, -0.22)
            add(PIC_OF_CLASS[b], a, -0.22)

    # composites: supportive but secondary
    for c in range(43):
        cn = "c%d" % c
        for c2 in range(43):
            if c2 == c:
                add(cn, c2, 0.45 if c in (6, 32, 41, 42) else 0.70)
            elif fam_of.get(c2) == fam_of.get(c):
                add(cn, c2, -0.04)

    # global features.  g_ink_in separates 'content' signs from blank ones.
    for c in range(43):
        add("g_red_in", c, -0.10)
    add("g_red_in", 9, 0.65)
    add("g_red_in", 10, -0.20)
    for c in list(NUMBERS) + [9, 10, 16, 32, 42]:
        add("g_ink_in", c, 0.9)
    add("g_ink_in", 6, 0.6)
    add("g_ink_in", 15, -1.6)
    add("g_ink_in", 41, -0.5)
    add("g_bar", 17, 0.55)
    add("g_bar", 14, -0.25)

    biases = {c: 0.0 for c in range(43)}
    return fuse, biases


# ---------------------------------------------------------------- assembly
def build_all():
    fam_names = list(FAMILIES.keys())
    # bank A: families, V=3
    kA = []
    for f in fam_names:
        fn = FAMILIES[f][0]
        for D in SCALES:
            kA.append((D, fn()))
    # bank B: numbers, V=9
    kB = []
    for nm in NUM_LOGICALS:
        for (D, p) in number_bank_parts(nm):
            kB.append((D, p))
    # bank C: pictos + composites, V = 3 scales x 3 rotations
    kC = []
    pic_fns = {n: pico_parts(n) for n in PICO_LOGICALS}
    comp_fns = {c: composite_parts(c) for c in range(43)}
    for n in PICO_LOGICALS:
        for D in SCALES:
            for a in PROTS:
                kC.append((D, rot_parts(pic_fns[n](D), a) if a else pic_fns[n](D)))
    for c in range(43):
        for D in SCALES:
            for a in PROTS:
                kC.append((D, rot_parts(comp_fns[c](D), a) if a else comp_fns[c](D)))

    WA = np.stack([make_kernel(KS, D, p) for (D, p) in kA])
    WB = np.stack([make_kernel(KS, D, p) for (D, p) in kB])
    WC = np.stack([make_kernel(KS, D, p) for (D, p) in kC])
    # global bank: canvas 32, D fixed, L1-normalised (mean-type features)
    gk = []
    for (name, p) in global_parts():
        k = make_kernel(32, 24.5, p, normalise=False)
        pos = np.clip(k, 0, None).sum()
        k = k / max(pos, 1e-6)
        gk.append(k)
    WD = np.stack(gk)

    fuse, biases = build_fusion()

    def dense_W(features):
        W = np.zeros((43, len(features)), np.float32)
        for j, f in enumerate(features):
            for c, w in fuse.get(f, {}).items():
                W[c, j] = w
        return W

    dA = dense_W(fam_names)
    dB = dense_W(NUM_LOGICALS)
    dC = dense_W(PICO_LOGICALS + ["c%d" % c for c in range(43)])
    dD = dense_W([n for (n, _) in global_parts()])

    bA = np.zeros(43, np.float32)
    for c, v in biases.items():
        bA[c] = v

    layers = [
        {"name": "nimg", "op": "norm", "inputs": ["image"]},
        {"name": "gray", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "gnorm", "op": "norm", "inputs": ["gray"]},
        {"name": "neg", "op": "conv", "inputs": ["gnorm"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "ink", "op": "relu", "inputs": ["neg"]},
        {"name": "bri", "op": "relu", "inputs": ["gnorm"]},
        {"name": "rd", "op": "conv", "inputs": ["image"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "rpos", "op": "relu", "inputs": ["rd"]},
        {"name": "rn", "op": "norm", "inputs": ["rpos"]},
        {"name": "yc", "op": "conv", "inputs": ["rpos"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "ypos", "op": "relu", "inputs": ["yc"]},
        {"name": "yn", "op": "norm", "inputs": ["ypos"]},
        {"name": "bd", "op": "conv", "inputs": ["image"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "bpos", "op": "relu", "inputs": ["bd"]},
        {"name": "bn", "op": "norm", "inputs": ["bpos"]},
        {"name": "fr", "op": "conv", "inputs": ["rn"], "out": 8, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "fyc", "op": "conv", "inputs": ["yn"], "out": 8, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "fb", "op": "conv", "inputs": ["bn"], "out": 8, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "fy", "op": "conv", "inputs": ["yn"], "out": 8, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "fg", "op": "conv", "inputs": ["gnorm"], "out": 8, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "fi", "op": "conv", "inputs": ["ink"], "out": 8, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "fbr", "op": "conv", "inputs": ["bri"], "out": 8, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "F", "op": "add", "inputs": ["fr", "fb", "fy", "fyc", "fg", "fi", "fbr"]},
        {"name": "cA", "op": "conv", "inputs": ["F"], "out": int(WA.shape[0]), "kernel": [KS, KS], "stride": 1, "padding": PAD},
        {"name": "rA", "op": "reshape", "inputs": ["cA"], "shape": [len(fam_names), len(SCALES), 81]},
        {"name": "mA", "op": "maxpool", "inputs": ["rA"], "size": [len(SCALES), 81], "stride": [1, 1]},
        {"name": "vA", "op": "reshape", "inputs": ["mA"], "shape": [len(fam_names)]},
        {"name": "cB", "op": "conv", "inputs": ["F"], "out": int(WB.shape[0]), "kernel": [KS, KS], "stride": 1, "padding": PAD},
        {"name": "rB", "op": "reshape", "inputs": ["cB"], "shape": [len(NUM_LOGICALS), 9, 81]},
        {"name": "mB", "op": "maxpool", "inputs": ["rB"], "size": [9, 81], "stride": [1, 1]},
        {"name": "vB", "op": "reshape", "inputs": ["mB"], "shape": [len(NUM_LOGICALS)]},
        {"name": "cC", "op": "conv", "inputs": ["F"], "out": int(WC.shape[0]), "kernel": [KS, KS], "stride": 1, "padding": PAD},
        {"name": "rC", "op": "reshape", "inputs": ["cC"], "shape": [len(kC) // (len(SCALES) * len(PROTS)), len(SCALES) * len(PROTS), 81]},
        {"name": "mC", "op": "maxpool", "inputs": ["rC"], "size": [len(SCALES) * len(PROTS), 81], "stride": [1, 1]},
        {"name": "vC", "op": "reshape", "inputs": ["mC"], "shape": [len(kC) // (len(SCALES) * len(PROTS))]},
        {"name": "cD", "op": "conv", "inputs": ["F"], "out": int(WD.shape[0]), "kernel": [32, 32], "stride": 1, "padding": 0},
        {"name": "vD", "op": "reshape", "inputs": ["cD"], "shape": [int(WD.shape[0])]},
        {"name": "dA", "op": "dense", "inputs": ["vA"], "out": 43},
        {"name": "dB", "op": "dense", "inputs": ["vB"], "out": 43},
        {"name": "dC", "op": "dense", "inputs": ["vC"], "out": 43},
        {"name": "dD", "op": "dense", "inputs": ["vD"], "out": 43},
        {"name": "scores", "op": "add", "inputs": ["dA", "dB", "dC", "dD"]},
    ]

    weights = {
        "gray.W": np.full((1, 3, 1, 1), 1.0 / 3.0, np.float32), "gray.b": np.zeros(1, np.float32),
        "neg.W": np.full((1, 1, 1, 1), -1.0, np.float32), "neg.b": np.zeros(1, np.float32),
        "rd.W": np.array([[[[1.0]], [[-1.0]], [[0.0]]],    # R-G
                          [[[1.0]], [[0.0]], [[-1.0]]],    # R-B
                          [[[-1.0]], [[1.0]], [[0.0]]]], np.float32),  # G-R
        "rd.b": np.zeros(3, np.float32),
        "yc.W": np.array([[[[-1.0]], [[1.0]], [[-1.0]]]], np.float32),  # (R-B)-(R-G)-(G-R): yellow only
        "yc.b": np.full(1, -0.04, np.float32),  # margin: red leaks ~0.02 into yellow
        "bd.W": np.array([[[[-1.0]], [[0.0]], [[1.0]]],   # B-R
                          [[[0.0]], [[-1.0]], [[1.0]]]], np.float32),  # B-G
        "bd.b": np.zeros(2, np.float32),
        "fr.W": np.zeros((8, 3, 1, 1), np.float32), "fr.b": np.zeros(8, np.float32),
        "fb.W": np.zeros((8, 2, 1, 1), np.float32), "fb.b": np.zeros(8, np.float32),
        "fy.W": np.zeros((8, 1, 1, 1), np.float32), "fy.b": np.zeros(8, np.float32),
        "fyc.W": np.zeros((8, 1, 1, 1), np.float32), "fyc.b": np.zeros(8, np.float32),
        "fg.W": np.zeros((8, 1, 1, 1), np.float32), "fg.b": np.zeros(8, np.float32),
        "fi.W": np.zeros((8, 1, 1, 1), np.float32), "fi.b": np.zeros(8, np.float32),
        "fbr.W": np.zeros((8, 1, 1, 1), np.float32), "fbr.b": np.zeros(8, np.float32),
        "cA.W": WA.astype(np.float32), "cA.b": np.full(len(WA), -5.0, np.float32),
        "cB.W": WB.astype(np.float32), "cB.b": np.full(len(WB), -8.0, np.float32),
        "cC.W": WC.astype(np.float32), "cC.b": np.full(len(WC), -6.0, np.float32),
        "cD.W": WD.astype(np.float32), "cD.b": np.zeros(len(WD), np.float32),
        "dA.W": dA, "dA.b": bA,
        "dB.W": dB, "dB.b": np.zeros(43, np.float32),
        "dC.W": dC, "dC.b": np.zeros(43, np.float32),
        "dD.W": dD, "dD.b": np.zeros(43, np.float32),
    }
    weights["fr.W"][0, 0] = 1.0
    weights["fr.W"][1, 1] = 1.0
    weights["fb.W"][2, 0] = 1.0
    weights["fb.W"][3, 1] = 1.0
    weights["fy.W"][4, 0] = 1.0
    weights["fyc.W"][4, 0] = 1.0
    weights["fg.W"][5, 0] = 1.0
    weights["fi.W"][6, 0] = 1.0
    weights["fbr.W"][7, 0] = 1.0
    return layers, weights


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    layers, weights = build_all()
    import os
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump({"layers": layers}, f)
    np.savez(os.path.join(out, "weights.npz"), **weights)
    tot = sum(int(np.prod(v.shape)) for v in weights.values())
    print("wrote network.json and weights.npz; total weights %d" % tot)


if __name__ == "__main__":
    main()
