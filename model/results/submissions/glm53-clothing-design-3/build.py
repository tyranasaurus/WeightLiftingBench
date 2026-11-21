#!/usr/bin/env python3
"""Hand-designed network for 28x28 greyscale clothing classification.

python3 build.py NEW_FOLDER   ->  writes NEW_FOLDER/network.json and NEW_FOLDER/weights.npz

Every weight comes from formulas in this file: Gaussian blurs and Gaussian
derivatives, boxes, rings, bars, polygons and a coarse occupancy grid, all
encoding general knowledge of how the ten classes look:

  tee       wide shoulders + short sleeves, narrow body below the sleeves
  trouser   waistband, two legs with a dark gap between them, empty sides
  pullover  wide all the way down (long sleeves at the sides), round dark collar
  dress     thin straps, narrow bodice, flared skirt, empty sides at the waist
  coat      shoulders + long sleeves, lapel V, centre front seam, split hem
  shirt     shoulders + long sleeves, collar points, button placket, buttons
  sandal    sparse, diagonal sole and straps, empty top and bottom rows
  sneaker   solid horizontal band in the middle rows, bright sole, empty top
            and bottom rows
  bag       thin handle arc over a dark gap, rectangular body to the bottom
  boot      tall shaft on the right, heel at the right, toe to the left

The occupancy branch (maxpool over 4x4 cells, then a per-cell threshold
implemented as a 1x1 conv and a relu) measures what fraction of a region
contains item, which works equally for bright and dark photos.  Detector
thresholds (THRS) and the rule-table weights (SCORE) are typed in by hand.
The output biases are the only derived numbers: each is chosen so that a
typical synthetic line drawing of that class (rendered by render_probes from
the same shape models) scores zero.  No dataset is read; a fixed seed makes
the build deterministic.
"""
import json
import sys
from pathlib import Path

import numpy as np

H = W = 28
YY = np.arange(H)[:, None] * np.ones((1, W))
XX = np.ones((H, 1)) * np.arange(W)[None, :]

# ------------------------------------------------------------------ drawing
def rect(y0, y1, x0, x1):
    m = np.zeros((H, W))
    y0, y1 = max(int(y0), 0), min(int(y1), H - 1)
    x0, x1 = max(int(x0), 0), min(int(x1), W - 1)
    if y1 >= y0 and x1 >= x0:
        m[y0:y1 + 1, x0:x1 + 1] = 1.0
    return m

def disk(cy, cx, r):
    return (((YY - cy) ** 2 + (XX - cx) ** 2) <= r * r).astype(float)

def ring(cy, cx, r0, r1, upper_only=True):
    rr = (YY - cy) ** 2 + (XX - cx) ** 2
    m = (rr >= r0 * r0) & (rr <= r1 * r1)
    if upper_only:
        m &= (YY <= cy)
    return m.astype(float)

def bar(p0, p1, w):
    (y0, x0), (y1, x1) = p0, p1
    vy, vx = y1 - y0, x1 - x0
    L2 = vy * vy + vx * vx + 1e-9
    t = np.clip(((YY - y0) * vy + (XX - x0) * vx) / L2, 0, 1)
    d2 = (YY - y0 - t * vy) ** 2 + (XX - x0 - t * vx) ** 2
    return (d2 <= (w / 2.0) ** 2).astype(float)

def tri(p0, p1, p2):
    (y0, x0), (y1, x1), (y2, x2) = p0, p1, p2

    def side(py, px, ay, ax, by, bx):
        return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

    s0 = side(YY, XX, y0, x0, y1, x1)
    s1 = side(YY, XX, y1, x1, y2, x2)
    s2 = side(YY, XX, y2, x2, y0, x0)
    inside = ((s0 >= 0) & (s1 >= 0) & (s2 >= 0)) | ((s0 <= 0) & (s1 <= 0) & (s2 <= 0))
    return inside.astype(float)

def poly(pts):
    m = np.zeros((H, W))
    for i in range(1, len(pts) - 1):
        m += tri(pts[0], pts[i], pts[i + 1])
    return np.clip(m, 0, 1)

# ---------------------------------------------------------------- feat layers
KS = 11
NS = 5                      # smooth branch: raw, B1, B2, blob+, blob-
S_RAW, S_B1, S_B2, S_BLOBP, S_BLOBM = range(NS)
NE = 8                      # rectified derivative branch
E_DXP, E_DXM, E_DYP, E_DYM, E_DUP, E_DUM, E_DVP, E_DVM = range(NE)
NCELL = 2                   # occupancy branch channels (from the raw cell max)
CELL_T = [0.12]             # occupancy threshold
CELL_TS = [0.09]            # strong-emptiness threshold
CELL_CAP = [0.06, 0.50]     # saturation caps (empty channel uncapped)


def gk(size, sigma):
    k = size // 2
    y, x = np.mgrid[-k:k + 1, -k:k + 1].astype(float)
    w = np.exp(-(x ** 2 + y ** 2) / (2 * sigma * sigma))
    return w / w.sum()


def smooth_kernels():
    K = np.zeros((NS, KS, KS))
    K[S_RAW, KS // 2, KS // 2] = 1.0
    K[S_B1] = gk(KS, 1.0)
    K[S_B2] = gk(KS, 2.0)
    blob = gk(KS, 1.0) - gk(KS, 3.0)
    K[S_BLOBP] = 8.0 * blob
    K[S_BLOBM] = -8.0 * blob
    return K


def deriv_kernels():
    K = np.zeros((NE, KS, KS))
    sd = 1.2
    y, x = np.mgrid[-(KS // 2):KS // 2 + 1, -(KS // 2):KS // 2 + 1].astype(float)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sd * sd))

    def half(w):  # positive half sums to 0.5, so a unit step gives response 1
        s = w[w > 0].sum()
        return w * (0.5 / s) if s > 0 else w

    dx = half(x * g)
    dy = half(y * g)
    du = half((x + y) * g)
    dv = half((y - x) * g)
    E = 2.0
    K[E_DXP], K[E_DXM] = E * dx, -E * dx
    K[E_DYP], K[E_DYM] = E * dy, -E * dy
    K[E_DUP], K[E_DUM] = E * du, -E * du
    K[E_DVP], K[E_DVM] = E * dv, -E * dv
    return K


def cellt_weights():
    """1x1 conv on the pooled cell map -> 2 channels from the RAW source:
    0 = relu(rawcell - .12)  occupied     1 = relu(.09 - rawcell)  strongly empty"""
    W = np.zeros((NCELL, NS, 1, 1))
    b = np.zeros(NCELL)
    W[0, S_RAW, 0, 0] = 1.0;  b[0] = -CELL_T[0]
    W[1, S_RAW, 0, 0] = -1.0; b[1] = CELL_TS[0]
    return W, b


# ---------------------------------------------------------------- detectors
CLASSES = ("tee", "trouser", "pullover", "dress", "coat", "sandal",
           "shirt", "sneaker", "bag", "boot")
C = {n: i for i, n in enumerate(CLASSES)}


def cells(r0, r1, c0, c1):
    m = np.zeros((7, 7))
    m[r0:r1 + 1, c0:c1 + 1] = 1.0
    return m


# occupancy-channel indices in cellSat
OCC, EMPS = 0, 1         # raw-cell occupied / strongly empty

# Hand-chosen thresholds (mean-excess units for cell detectors, mean
# brightness for templates, mean rectified-derivative energy for edges).
THRS = {
    "cTop": .18, "cTopE": .25, "cTopC": .18, "cTopS": .15, "cTopSE": .20,
    "cArcTop": .10, "cUarm": .15, "cUarmE": .20, "cWaistSide": .18, "cWaistSideE": .22, "cSkirt": .15, "cBotRow": .15,
    "cBotRowE": .40, "cMidFrac": .30, "cShaftR": .25, "cShaftL": .25,
    "cHeel": .18, "cToe": .18, "cSoleRow": .45, "neck": .15,
    "handleindark": .10, "coatPanels": .03, "teeT": .30, "flare": .30, 
    "collarpts": .18, "placketB": .35, "knitE": .25, "soleE": .40,
    "topedgeE": .55, "diagA": .58, "gapE": .45,
}

DETS = [
    # ---- occupancy-grid detectors (fraction of cells in a region with item) ----
    dict(name="cTop", branch="C", ch=OCC, sup=cells(0, 1, 0, 6)),
    dict(name="cTopE", branch="C", ch=EMPS, sup=cells(0, 1, 0, 6)),
    dict(name="cTopC", branch="C", ch=OCC, sup=cells(0, 1, 2, 4)),
    dict(name="cTopS", branch="C", ch=OCC, sup=cells(0, 1, 1, 1) + cells(0, 1, 5, 5)),
    dict(name="cTopSE", branch="C", ch=EMPS, sup=cells(0, 1, 1, 1) + cells(0, 1, 5, 5)),
    dict(name="cArcTop", branch="C", ch=OCC, sup=cells(0, 0, 3, 3)),
    dict(name="cUarm", branch="C", ch=OCC, sup=cells(5, 5, 0, 0) + cells(5, 5, 6, 6) + cells(6, 6, 0, 0) + cells(6, 6, 6, 6)),
    dict(name="cUarmE", branch="C", ch=EMPS, sup=cells(5, 5, 0, 0) + cells(5, 5, 6, 6) + cells(6, 6, 0, 0) + cells(6, 6, 6, 6)),
    dict(name="cWaistSide", branch="C", ch=OCC, sup=cells(2, 3, 0, 0) + cells(2, 3, 6, 6)),
    dict(name="cWaistSideE", branch="C", ch=EMPS, sup=cells(2, 3, 0, 0) + cells(2, 3, 6, 6)),
    dict(name="cSkirt", branch="C", ch=OCC, sup=cells(5, 6, 1, 1) + cells(5, 6, 5, 5)),
    dict(name="cBotRow", branch="C", ch=OCC, sup=cells(6, 6, 0, 6)),
    dict(name="cBotRowE", branch="C", ch=EMPS, sup=cells(6, 6, 0, 6)),
    dict(name="cMidFrac", branch="C", ch=OCC, sup=cells(2, 4, 0, 6)),
    dict(name="cShaftR", branch="C", ch=OCC, sup=cells(0, 1, 4, 4)),
    dict(name="cShaftL", branch="C", ch=OCC, sup=cells(0, 1, 2, 2)),
    dict(name="cHeel", branch="C", ch=OCC, sup=cells(3, 4, 6, 6)),
    dict(name="cToe", branch="C", ch=OCC, sup=cells(3, 4, 0, 0)),
    dict(name="cSoleRow", branch="C", ch=OCC, sup=cells(4, 5, 0, 6)),
    dict(name="neck", branch="S", ch=S_BLOBM,
         parts=[(disk(2.5, 14, 2.6), +1)], postar=4),
    dict(name="handleindark", branch="S", ch=S_B2, parts=[(rect(5, 8, 12, 16), -1)],
         negtar=1, dark=True),
    # ---- shape templates (raw blur units) ----
    dict(name="teeT", branch="S", ch=S_B1,
         parts=[(rect(2, 9, 3, 25) + rect(10, 23, 9, 18) + rect(24, 27, 9, 18), +1),
                (rect(11, 23, 0, 8), -1), (rect(11, 23, 19, 27), -1)], postar=1, negtar=[0.5, 0.5]),
    dict(name="flare", branch="S", ch=S_B1,
         parts=[(poly([(15, 10), (15, 17), (26, 22), (26, 5)]), +1),
                (rect(7, 14, 0, 7), -1), (rect(7, 14, 20, 27), -1)], postar=1, negtar=[0.5, 0.5]),
    dict(name="coatPanels", branch="S", ch=S_B2,
         parts=[(rect(9, 20, 4, 8) + rect(9, 20, 17, 21), +1), (rect(9, 20, 12, 15), -1)],
         postar=1, negtar=1.15),
    dict(name="collarpts", branch="S", ch=S_B1,
         parts=[(tri((0, 10), (0, 13), (3, 12)) + tri((0, 15), (0, 18), (3, 16)), +1),
                (rect(1, 2, 13, 15), -1)], postar=1, negtar=0.45),
    # ---- edge-energy detectors (rectified Gaussian derivatives) ----
    dict(name="placketB", branch="E",
         parts=[(E_DXP, rect(7, 12, 11, 13) + rect(16, 21, 11, 13), +1),
                (E_DXM, rect(7, 12, 15, 17) + rect(16, 21, 15, 17), +1)], postar=4),
    dict(name="soleE", branch="E",
         parts=[(E_DYP, rect(14, 18, 2, 26), +1), (E_DYM, rect(18, 22, 2, 26), +1)], postar=4),
    dict(name="topedgeE", branch="E",
         parts=[(E_DYP, rect(7, 11, 3, 25), +1), (E_DYM, rect(10, 13, 3, 25), +1)], postar=4),
    dict(name="diagA", branch="E",
         parts=[(E_DUP, rect(6, 20, 3, 25), +1), (E_DUM, rect(6, 20, 3, 25), +1)], postar=5),
    dict(name="knitE", branch="E",
         parts=[(E_DXP, rect(7, 20, 5, 10) + rect(7, 20, 18, 23), +1),
                (E_DXM, rect(7, 20, 5, 10) + rect(7, 20, 18, 23), +1)], postar=3),
    dict(name="gapE", branch="E",
         parts=[(E_DXM, rect(12, 26, 11, 14), +1), (E_DXP, rect(12, 26, 14, 17), +1)], postar=4),
]
DIDX = {d["name"]: i for i, d in enumerate(DETS)}
ND = len(DETS)


def det_row(spec):
    """One dense row (flattened) from a spec."""
    if spec["branch"] == "C":
        w = np.zeros((NCELL, 7, 7))
        w[spec["ch"]] = spec["sup"] / spec["sup"].sum() * 10.0   # saturated units
        return w.reshape(-1)
    if spec["branch"] == "E":
        nch = NE
        parts = spec["parts"]
    else:
        nch = NS
        ch0 = spec["ch"]
        parts = [(ch0, arr, sign) for (arr, sign) in spec["parts"]]
    pos = np.zeros((nch, H, W))
    negs = []
    for ch, arr, sign in parts:
        if sign > 0:
            pos[ch] += arr
        else:
            negs.append((ch, arr))
    w = np.zeros((nch, H, W))
    P = pos.sum()
    pt = spec.get("postar", 1.0)
    if P > 0 and pt:
        w += pos * (pt / P)
    nts = spec.get("negtar", 0.0)
    if isinstance(nts, (int, float)):
        nts = [nts]
    for gi, (ch, arr) in enumerate(negs):
        t = nts[gi] if gi < len(nts) else (nts[-1] if nts else 0.0)
        s = arr.sum()
        if s > 0 and t:
            w[ch] -= arr * (t / s)
    return w.reshape(-1)


# ------------------------------------------------------- synthetic drawings
def outline(v, amt):
    m = v > 0.02
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    out = v.copy()
    out[m & ~e] += amt
    return out


def pr_tee(rng, oy, ox):
    b = rng.uniform(0.3, 1.0)
    wide = int(rng.integers(-1, 2))
    w = int(rng.integers(0, 3))
    v = rect(2 + oy, 9 + oy, 3 + w + ox, 24 - w + ox) * b             # shoulders + sleeves
    v += rect(10 + oy, 23 + oy, 9 + ox + wide, 18 + ox - wide) * b    # body
    v += rect(24 + oy, 26 + oy, 9 + ox + wide, 18 + ox - wide) * min(b + 0.15, 1.0)
    v -= disk(2.0 + oy, 14 + ox, 2.6) * b * 0.55                      # neck hole
    if rng.random() < 0.3:                                            # printed design
        v += rect(11 + oy, 14 + oy, 11 + ox, 17 + ox) * b * 0.5
    return v


def pr_trouser(rng, oy, ox):
    b = rng.uniform(0.3, 1.0)
    v = rect(0 + oy, 3 + oy, 9 + ox, 19 + ox) * b
    v += rect(4 + oy, 26 + oy, 8 + ox, 12 + ox) * b
    v += rect(4 + oy, 26 + oy, 16 + ox, 20 + ox) * b
    v -= rect(1 + oy, 7 + oy, 12 + ox, 13 + ox) * b * 0.4             # fly seam
    v += rect(25 + oy, 26 + oy, 8 + ox, 20 + ox) * 0.15
    return v


def pr_pullover(rng, oy, ox):
    b = rng.uniform(0.3, 1.0)
    w = int(rng.integers(0, 3))
    v = rect(1 + oy, 23 + oy, 2 + w + ox, 25 - w + ox) * b
    v += rect(24 + oy, 26 + oy, 7 + ox, 20 + ox) * b                  # hem centre
    if rng.random() < 0.7:                                            # cuffs reach the corners
        v += rect(24 + oy, 26 + oy, 2 + w + ox, 6 + ox) * b
        v += rect(24 + oy, 26 + oy, 21 + ox, 25 - w + ox) * b
    v -= disk(2.0 + oy, 14 + ox, 3.4) * b * 0.6                       # round collar
    v += rect(24 + oy, 26 + oy, 7 + ox, 20 + ox) * 0.15               # hem band
    for x in range(4, 24, 3):                                         # knit texture
        v -= rect(3 + oy, 23 + oy, x + ox, x + ox) * b * rng.uniform(0.22, 0.34)
    return v


def pr_dress(rng, oy, ox):
    b = rng.uniform(0.3, 1.0)
    v = rect(0 + oy, 3 + oy, 9 + ox, 18 + ox) * b                    # bodice top / straps
    v += rect(4 + oy, 14 + oy, 9 + ox, 18 + ox) * b                   # bodice
    v += poly([(15 + oy, 9 + ox), (15 + oy, 18 + ox), (26 + oy, 22 + ox), (26 + oy, 5 + ox)]) * b
    v += rect(25 + oy, 26 + oy, 6 + ox, 21 + ox) * 0.12
    if rng.random() < 0.6:
        v -= rect(15 + oy, 16 + oy, 9 + ox, 18 + ox) * b * 0.3        # waist seam
    return v


def pr_coat(rng, oy, ox):
    b = rng.uniform(0.35, 1.0)
    w = int(rng.integers(0, 2))
    v = rect(0 + oy, 23 + oy, 4 + w + ox, 24 - w + ox) * b            # body
    v += rect(24 + oy, 26 + oy, 8 + ox, 20 + ox) * b                  # hem centre
    if rng.random() < 0.4:                                            # coat sides reach low
        v += rect(24 + oy, 26 + oy, 4 + w + ox, 9 + ox) * b
        v += rect(24 + oy, 26 + oy, 18 + ox, 24 - w + ox) * b
    v += bar((1 + oy, 10 + ox), (7 + oy, 14 + ox), 2.2) * b * 0.6     # lapels
    v += bar((1 + oy, 18 + ox), (7 + oy, 14 + ox), 2.2) * b * 0.6
    v -= rect(2 + oy, 5 + oy, 13 + ox, 15 + ox) * b * 0.55            # dark V inside lapels
    v -= rect(5 + oy, 21 + oy, 13 + ox, 14 + ox) * b * 0.45           # centre seam
    v -= rect(21 + oy, 26 + oy, 12 + ox, 16 + ox) * b * 1.0           # split hem (black gap)
    v += rect(9 + oy, 20 + oy, 4 + ox, 8 + ox) * b * 0.14            # front panels
    v += rect(9 + oy, 20 + oy, 17 + ox, 21 + ox) * b * 0.14
    return v


def pr_shirt(rng, oy, ox):
    b = rng.uniform(0.3, 1.0)
    body = b * 0.12 if rng.random() < 0.2 else b                      # outline-style variant
    w = int(rng.integers(0, 3))
    v = rect(1 + oy, 23 + oy, 2 + w + ox, 25 - w + ox) * body
    v += rect(24 + oy, 26 + oy, 7 + ox, 20 + ox) * body               # hem centre
    if rng.random() < 0.4:                                            # cuffs reach the corners
        v += rect(24 + oy, 26 + oy, 2 + w + ox, 6 + ox) * b
        v += rect(24 + oy, 26 + oy, 21 + ox, 25 - w + ox) * b
    v += tri((0 + oy, 10 + ox), (0 + oy, 13 + ox), (3 + oy, 12 + ox)) * b * 0.5
    v += tri((0 + oy, 15 + ox), (0 + oy, 18 + ox), (3 + oy, 16 + ox)) * b * 0.5
    v -= rect(1 + oy, 2 + oy, 13 + ox, 15 + ox) * body * 0.6          # gap between points
    v += rect(5 + oy, 23 + oy, 13 + ox, 15 + ox) * b * 0.25           # placket
    for yy in (8, 12, 16, 20):
        v += disk(yy + oy, 14 + ox, 1.2) * b * 0.5                    # buttons
    v += rect(24 + oy, 26 + oy, 2 + ox, 26 + ox) * b * 0.15           # cuffs/hem
    return v


def pr_sandal(rng, oy, ox):
    b = rng.uniform(0.35, 1.0)
    v = bar((21 + oy, 3 + ox), (7 + oy, 23 + ox), 5.5) * b            # diagonal sole
    v += disk(20 + oy, 6 + ox, 3.5) * b                               # heel blob
    v += disk(6 + oy, 21 + ox, 3.0) * b                               # toe blob
    if rng.random() < 0.7:
        v += bar((13 + oy, 9 + ox), (19 + oy, 19 + ox), 1.4) * b
    if rng.random() < 0.7:
        v += bar((9 + oy, 13 + ox), (15 + oy, 23 + ox), 1.4) * b
    return v


def pr_sneaker(rng, oy, ox):
    b = rng.uniform(0.3, 1.0)
    v = rect(13 + oy, 17 + oy, 1 + ox, 26 + ox) * b                   # toe box
    v += rect(11 + oy, 16 + oy, 8 + ox, 25 + ox) * b                  # upper
    v += rect(10 + oy, 13 + oy, 15 + ox, 24 + ox) * b                 # ankle
    v += rect(18 + oy, 21 + oy, 1 + ox, 26 + ox) * min(b + 0.2, 1.0)  # sole
    v -= disk(10.5 + oy, 20 + ox, 2.2) * b * 0.5                      # ankle opening
    v += rect(20 + oy, 21 + oy, 21 + ox, 26 + ox) * 0.15
    for yy0, xx0 in ((11, 11), (12, 14), (13, 17)):
        v += bar((yy0 + oy, xx0 + ox), (yy0 + 2 + oy, xx0 + 3 + ox), 1.1) * 0.3
    return v


def pr_bag(rng, oy, ox):
    b = rng.uniform(0.3, 1.0)
    if rng.random() < 0.7:                                            # portrait handbag
        r0 = rng.uniform(7.0, 9.5)
        v = ring(11 + oy, 14 + ox, r0, r0 + 1.5) * b                  # handle arc
        bw = int(rng.integers(0, 3))
        v += rect(11 + oy, 27 + oy, 2 + bw + ox, 25 - bw + ox) * b     # body
        v += rect(11 + oy, 12 + oy, 2 + bw + ox, 25 - bw + ox) * min(b + 0.2, 1.0)
        v -= rect(16 + oy, 17 + oy, 3 + bw + ox, 24 - bw + ox) * b * 0.4        # flap line
        v -= rect(20 + oy, 21 + oy, 3 + bw + ox, 24 - bw + ox) * b * 0.3        # pocket line
        v -= rect(13 + oy, 25 + oy, 14 + ox, 15 + ox) * b * 0.15
        v += rect(26 + oy, 27 + oy, 3 + ox, 25 + ox) * 0.15
    else:                                                             # landscape
        v = bar((5 + oy, 2 + ox), (5 + oy, 25 + ox), 1.6) * b
        v += bar((8 + oy, 2 + ox), (8 + oy, 25 + ox), 1.6) * b
        v += rect(9 + oy, 26 + oy, 0 + ox, 27 + ox) * b
        v -= rect(14 + oy, 15 + oy, 1 + ox, 26 + ox) * b * 0.35
        v -= rect(18 + oy, 19 + oy, 1 + ox, 26 + ox) * b * 0.25
        v += rect(25 + oy, 26 + oy, 0 + ox, 27 + ox) * 0.15
    return v


def pr_boot(rng, oy, ox):
    b = rng.uniform(0.3, 1.0)
    lean = int(rng.integers(-1, 2))
    v = rect(1 + oy, 11 + oy, 13 + ox + lean, 24 + ox) * b            # shaft
    v += rect(12 + oy, 19 + oy, 1 + ox, 26 + ox) * b                  # foot
    v += rect(19 + oy, 23 + oy, 1 + ox, 26 + ox) * min(b + 0.35, 1.0)  # sole
    v += rect(13 + oy, 19 + oy, 21 + ox, 26 + ox) * b * 0.3           # heel column
    v -= rect(12 + oy, 18 + oy, 2 + ox, 8 + ox) * b * 0.25
    return v


PROBES = [pr_tee, pr_trouser, pr_pullover, pr_dress, pr_coat,
          pr_sandal, pr_shirt, pr_sneaker, pr_bag, pr_boot]


def render_probes(n_per=30, seed=20261119, stress=False):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c, gen in enumerate(PROBES):
        for _ in range(n_per):
            if stress:
                oy, ox = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
            else:
                oy, ox = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
            v = np.clip(gen(rng, oy, ox), 0, 1)
            v = outline(v, 0.22)
            v += rng.normal(0, 0.015, (H, W))
            if stress:
                v = v * rng.uniform(0.45, 0.85)
            X.append(np.clip(v, 0, 1))
            y.append(c)
    return np.asarray(X, dtype=np.float32)[:, None], np.asarray(y)


# ------------------------------------------------------------- tiny forward
def conv_forward(x, K, b, stride=1, pad=0):
    n, c, h, w = x.shape
    o, ci, kh, kw = K.shape
    assert ci == c
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho = (x.shape[2] - kh) // stride + 1
    wo = (x.shape[3] - kw) // stride + 1
    out = np.zeros((n, o, ho * wo))
    for dy in range(kh):
        for dx in range(kw):
            sl = x[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
            sl = sl.reshape(n, c, ho * wo)
            kk = K[:, :, dy, dx]
            out += np.einsum('nck,oc->nok', sl, kk)
    out = out.reshape(n, o, ho, wo) + b[None, :, None, None]
    return out


def maxpool_forward(x, sh, sw, th, tw):
    n, c, h, w = x.shape
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    out = np.full((n, c, ho, wo), -np.inf)
    for dy in range(sh):
        for dx in range(sw):
            out = np.maximum(out, x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw])
    return out


def forward(net, weights, x):
    vals = {"image": x}
    for L in net["layers"]:
        ins = [vals[i] for i in L["inputs"]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = conv_forward(ins[0], weights[name + ".W"], weights[name + ".b"],
                                      L.get("stride", 1), L.get("padding", 0))
        elif op == "dense":
            vals[name] = ins[0].reshape(len(ins[0]), -1) @ weights[name + ".W"].T + weights[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op == "maxpool":
            (sh, sw), (th, tw) = L["size"], L["stride"]
            vals[name] = maxpool_forward(ins[0], sh, sw, th, tw)
        elif op == "add":
            vals[name] = sum(ins)
        else:
            raise ValueError(op)
    return vals[net["layers"][-1]["name"]]


# ------------------------------------------------------------ rule table
SCORE = {
    # The four top garments share a common evidence base (cTopS, cUarm, cTop,
    # cTopC, cWaistSide, cSkirt, cBotRow); class-unique evidence decides.
    "tee": [("cTopS", 2.5), ("cUarm", -4.0), ("cUarmE", 3.2), ("cWaistSideE", 2.0),
            ("cTop", 1.2), ("cTopC", 0.8), ("cSkirt", 0.8), ("cBotRow", 0.8),
            ("teeT", 1.5), ("neck", 0.4), ("cTopSE", -1.5), ("coatPanels", -0.4),
            ("placketB", -0.6), ("knitE", -0.8), ("collarpts", -0.8),
            ("gapE", -1.5), ("cTopE", -2.0), ("cBotRowE", -1.5), ("flare", -1.0),
            ("diagA", -0.5), ("handleindark", -1.0),
            ("cMidFrac", -0.4)],
    "trouser": [("gapE", 2.2), ("cTopSE", 2.0), ("cWaistSideE", 2.0), ("cUarmE", 2.2),
                ("cTopC", 1.0), ("cTop", 1.0), ("cBotRow", 1.0),
                ("cArcTop", 0.8), ("cTopS", -2.5), ("cUarm", -2.2),
                ("cSkirt", -1.0), ("cWaistSide", -1.2), ("cTopE", -1.0),
                ("cBotRowE", -1.5), ("flare", -1.2), ("topedgeE", -1.0),
                ("diagA", -0.8), ("soleE", -1.0), ("collarpts", -0.5), ("neck", -0.5), ("placketB", -0.4),
                ("knitE", -0.5), ("cMidFrac", -0.6), ("handleindark", -1.2)],
    "pullover": [("cTopS", 2.5), ("cUarm", 3.5), ("cWaistSide", 0.5),
                 ("cTop", 1.2), ("cTopC", 0.8), ("cSkirt", 0.6), ("cBotRow", 0.8),
                 ("neck", 1.0), ("knitE", 1.0), ("cUarmE", -3.0),
                 ("gapE", -1.6), ("collarpts", -1.0),
                 ("placketB", -1.0), ("teeT", -1.4), ("flare", -1.0),
                 ("cTopE", -2.0), ("cBotRowE", -1.6), ("cTopSE", -1.5),
                 ("diagA", -0.5), ("handleindark", -1.2), ("cMidFrac", 0.3)],
    "dress": [("flare", 2.5), ("cWaistSideE", 2.8), ("cUarmE", 2.8), ("cUarm", -3.0),
              ("cTopSE", 2.2), ("cSkirt", 1.2), ("cTopC", 1.2), ("cTop", 1.0),
              ("cBotRow", 1.0), ("cArcTop", 1.0), ("cTopS", -1.2),
              ("gapE", -2.0), ("cTopE", -1.2), ("cBotRowE", -1.4),
              ("cWaistSide", -1.5), ("topedgeE", -1.2), ("diagA", -0.8),
              ("soleE", -0.8), ("teeT", -1.0), ("placketB", -0.6),
              ("collarpts", -0.5), ("cMidFrac", -0.5),
              ("neck", -0.6), ("handleindark", -2.0), ("knitE", -0.4)],
    "coat": [("cTopS", 2.5), ("cUarm", 2.5), ("cWaistSide", 0.5),
             ("cTop", 1.0), ("cTopC", 0.8), ("cSkirt", 0.6), ("cBotRow", 0.6),
             ("gapE", 1.7), ("coatPanels", 1.2), ("collarpts", -0.6), ("cUarmE", -1.6),
             ("flare", -1.2), ("cTopE", -2.0), ("cBotRowE", -1.4),
             ("teeT", -1.4), ("topedgeE", -0.8), ("diagA", -0.6),
             ("cTopSE", -1.2), ("neck", -0.5),
             ("knitE", -0.4), ("cMidFrac", 0.3)],
    "sandal": [("cTopE", 1.6), ("cBotRowE", 2.2), ("diagA", 2.6), ("cTopSE", 0.8),
               ("cWaistSideE", 1.0), ("cTopS", -0.8), ("cBotRow", -0.6),
               ("cUarm", -0.4), ("cTop", -0.6), ("cTopC", -0.5),
               ("gapE", -0.5), ("cArcTop", -0.6), ("cSkirt", -0.4),
               ("flare", -0.4), ("placketB", -0.3), ("cWaistSide", -0.4),
               ("topedgeE", -0.4), ("cMidFrac", -0.3), ("cHeel", -0.3)],
    "shirt": [("cTopS", 2.5), ("cWaistSide", 0.8),
              ("cTop", 1.2), ("cTopC", 0.8), ("cSkirt", 0.8), ("cBotRow", 0.8),
              ("placketB", 2.0), ("collarpts", 1.0), ("knitE", -2.0),
              ("cUarmE", -2.4), ("cUarm", 4.5), ("cTopE", -2.0), ("cBotRowE", -1.6),
              ("flare", -0.8), ("topedgeE", -0.6), ("teeT", -1.4),
              ("cTopSE", -1.5), ("diagA", -0.4),
              ("handleindark", -1.2), ("neck", -0.4)],
    "sneaker": [("cTopE", 2.6), ("cBotRowE", 2.2), ("cMidFrac", 1.8), ("soleE", 1.6),
                ("cSoleRow", 0.6), ("cHeel", 0.5), ("cToe", 0.5), ("cTopSE", 1.0),
                ("diagA", -1.5), ("cTop", -2.2), ("cTopS", -2.0),
                ("cUarm", -1.2), ("cBotRow", -1.8), ("gapE", -1.2),
                ("cWaistSide", -0.8), ("flare", -1.0), ("cArcTop", -1.0),
                ("cTopC", -1.5), ("cSkirt", -0.8), ("neck", -0.4),
                ("cShaftR", -1.2), ("knitE", -0.4)],
    "bag": [("handleindark", 2.0), ("topedgeE", 2.0), ("cWaistSide", 2.2),
            ("cBotRow", 1.8), ("cTopSE", 2.0), ("cMidFrac", 1.0),
            ("cSkirt", 0.8), ("neck", -0.6), ("cTopS", -1.8), ("gapE", -1.2), ("coatPanels", -0.4),
            ("flare", -1.6), ("cTopE", -1.2), ("cBotRowE", -1.8),
            ("diagA", -0.8), ("soleE", -0.8), ("placketB", -0.5),
            ("collarpts", -0.6), ("cWaistSideE", -1.5), ("teeT", -0.8),
            ("cTop", -1.0),
            ("knitE", -0.6), ("neck", -0.4)],
    "boot": [("cShaftR", 2.6), ("cShaftL", -2.0), ("cHeel", 0.8), ("soleE", 1.0),
             ("cSoleRow", 0.8), ("cTopSE", 0.8), ("cMidFrac", 0.6),
             ("cToe", 0.5), ("cTop", 0.8), ("cBotRowE", -1.2),
             ("diagA", -1.2), ("cTopS", -2.2), ("cUarm", -1.4),
             ("gapE", -1.2), ("flare", -1.0), ("topedgeE", -1.0),
             ("cSkirt", -0.8), ("knitE", -0.4), ("placketB", -0.4)],
}






# ---------------------------------------------------------------- assemble
def branch_rows(branch):
    nch = {"C": NCELL * 49, "S": NS * H * W, "E": NE * H * W}[branch]
    rows = np.zeros((ND, nch))
    b = np.zeros(ND)
    for i, spec in enumerate(DETS):
        if spec["branch"] != branch:
            continue
        rows[i] = det_row(spec)
        b[i] = -THRS[spec["name"]]
    return rows, b


def build():
    featSW = smooth_kernels()[:, None, :, :].copy()
    featDW = deriv_kernels()[:, None, :, :].copy()
    cellW, cellB = cellt_weights()
    zb = np.zeros

    rowsC, bC = branch_rows("C")
    rowsS, bS = branch_rows("S")
    rowsE, bE = branch_rows("E")

    net = {"layers": [
        {"name": "featS", "op": "conv", "inputs": ["image"], "out": NS,
         "kernel": [KS, KS], "stride": 1, "padding": KS // 2},
        {"name": "cellM", "op": "maxpool", "inputs": ["featS"], "size": [4, 4], "stride": [4, 4]},
        {"name": "cellT", "op": "conv", "inputs": ["cellM"], "out": NCELL,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "cellR", "op": "relu", "inputs": ["cellT"]},
        {"name": "cellHi", "op": "conv", "inputs": ["cellR"], "out": NCELL,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "cellHiR", "op": "relu", "inputs": ["cellHi"]},
        {"name": "cellHiN", "op": "conv", "inputs": ["cellHiR"], "out": NCELL,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "cellSat", "op": "add", "inputs": ["cellR", "cellHiN"]},
        {"name": "featD", "op": "conv", "inputs": ["image"], "out": NE,
         "kernel": [KS, KS], "stride": 1, "padding": KS // 2},
        {"name": "featR", "op": "relu", "inputs": ["featD"]},
        {"name": "detC", "op": "dense", "inputs": ["cellSat"], "out": ND},
        {"name": "detS", "op": "dense", "inputs": ["featS"], "out": ND},
        {"name": "detE", "op": "dense", "inputs": ["featR"], "out": ND},
        {"name": "det", "op": "add", "inputs": ["detC", "detS", "detE"]},
        {"name": "detr", "op": "relu", "inputs": ["det"]},
        {"name": "score", "op": "dense", "inputs": ["detr"], "out": 10},
    ]}
    capW = np.zeros((NCELL, NCELL, 1, 1))
    capb = np.zeros(NCELL)
    negW = np.zeros((NCELL, NCELL, 1, 1))
    for i in range(NCELL):
        capW[i, i, 0, 0] = 1.0
        capb[i] = -CELL_CAP[i]
        negW[i, i, 0, 0] = -1.0

    weights = {
        "featS.W": featSW.astype(np.float32), "featS.b": zb(NS).astype(np.float32),
        "featD.W": featDW.astype(np.float32), "featD.b": zb(NE).astype(np.float32),
        "cellT.W": cellW.astype(np.float32), "cellT.b": cellB.astype(np.float32),
        "cellHi.W": capW.astype(np.float32), "cellHi.b": capb.astype(np.float32),
        "cellHiN.W": negW.astype(np.float32), "cellHiN.b": zb(NCELL).astype(np.float32),
        "detC.W": rowsC.astype(np.float32), "detC.b": bC.astype(np.float32),
        "detS.W": rowsS.astype(np.float32), "detS.b": bS.astype(np.float32),
        "detE.W": rowsE.astype(np.float32), "detE.b": bE.astype(np.float32),
    }

    # score weights from the rule table
    scoreW = np.zeros((10, ND))
    for cname, entries in SCORE.items():
        for dname, w in entries:
            if dname not in DIDX:
                raise KeyError(dname)
            scoreW[C[cname], DIDX[dname]] += w

    # output biases: each is chosen so that a typical synthetic line drawing
    # of that class scores zero (the only numbers derived from the drawings).
    Xp, yp = render_probes()
    FS = conv_forward(Xp, featSW, zb(NS), 1, KS // 2)
    CM = maxpool_forward(FS, 4, 4, 4, 4)
    CT = conv_forward(CM, cellW, cellB, 1, 0)
    CR = np.maximum(CT, 0)
    CHR = np.maximum(conv_forward(CR, capW, capb, 1, 0), 0)
    CS = CR + conv_forward(CHR, negW, zb(NCELL), 1, 0)   # CR - relu(CR-cap) = min(excess, cap)
    FD = conv_forward(Xp, featDW, zb(NE), 1, KS // 2)
    FR = np.maximum(FD, 0)
    detraw = (CS.reshape(len(Xp), -1) @ rowsC.T
              + FS.reshape(len(Xp), -1) @ rowsS.T
              + FR.reshape(len(Xp), -1) @ rowsE.T)
    detr = np.maximum(detraw + bC + bS + bE, 0)
    s0 = detr @ scoreW.T
    scoreb = np.zeros(10)
    for c in range(10):
        scoreb[c] = -s0[yp == c, c].mean()
    # self-check: on the synthetic drawings, each class's rule should win
    for c in range(10):
        win = np.bincount(np.argmax(s0[yp == c] + scoreb, 1), minlength=10)
        print(f"self-check {CLASSES[c]:8s}: own-rule wins {win[c]}/{(yp == c).sum()}")

    weights["score.W"] = scoreW.astype(np.float32)
    weights["score.b"] = scoreb.astype(np.float32)
    return net, weights, (Xp, yp, detraw)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    net, weights, _ = build()
    (out / "network.json").write_text(json.dumps(net))
    np.savez(out / "weights.npz", **weights)


if __name__ == "__main__":
    main()
