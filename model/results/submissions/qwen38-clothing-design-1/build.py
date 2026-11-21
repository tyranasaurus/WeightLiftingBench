"""Build weights for a hand-designed clothing/shoe/bag classifier.

All numbers come from formulas and hand-typed shape knowledge; no data is read.

Architecture
  c1   5x5 conv bank, 14 channels from formulas: intensity, blur, four
       oriented edge pairs, vertical/horizontal dark-gap (2nd derivative,
       zero sum), dark-hole and thin-bright detectors.
  n1   norm  -> per-image contrast normalisation (brightness invariance)
  r1   relu
  clip the foreground channel to a binary presence map with
       min(x, 0.5*max) = x - relu(x - 0.5*max)   (dense/add/relu, identities)
       so every occupied cell carries exactly the same known gain Gref and
       every shape threshold becomes an exact ratio.
  branches: 14x14 binary presence, 7x7 fill, row/column occupancy, global
       fill, channel maxima (gain references), 7x7 and 14x14 unclipped maps
       for the gap/edge channels; each read by one dense layer into a shared
       hidden vector (add), relu, then a dense readout.
  hidden units = hand-drawn silhouette templates (4 geometric variants per
       class) + semantic part detectors; readout = hand-written rule table.
"""
import json
import sys

import numpy as np

H = 128  # hidden units

# ----------------------------------------------------------------------------
# stage 1: conv bank kernels (5x5), from formulas
# ----------------------------------------------------------------------------
K = 5
yy, xx = np.mgrid[0:K, 0:K].astype(float) - 2.0
G = np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.3 ** 2))


def nrm(a):
    s = np.abs(a).sum()
    return a / s if s > 0 else a


KERNELS = []
F = np.zeros((K, K)); F[2, 2] = 1.0
KERNELS.append(F)                                        # 0 intensity
KERNELS.append(np.ones((K, K)) / (K * K))                # 1 blur
KERNELS.append(nrm(xx * G))                              # 2 vertical edge +
KERNELS.append(nrm(-xx * G))                             # 3 vertical edge -
KERNELS.append(nrm(yy * G))                              # 4 horizontal edge +
KERNELS.append(nrm(-yy * G))                             # 5 horizontal edge -
KERNELS.append(nrm((xx + yy) * G))                       # 6 diag edge +
KERNELS.append(nrm(-(xx + yy) * G))                      # 7 diag edge -
KERNELS.append(nrm((xx - yy) * G))                       # 8 diag edge +
KERNELS.append(nrm(-(xx - yy) * G))                      # 9 diag edge -
hgap = np.array([0.3, 0.6, -1.8, 0.6, 0.3])             # dark line, zero sum
roww = np.array([0.5, 0.75, 1.0, 0.75, 0.5])
KERNELS.append(nrm(np.outer(roww, hgap)))                # 10 vertical dark gap
KERNELS.append(nrm(np.outer(hgap, roww)))                # 11 horizontal dark gap
hole = np.where(np.maximum(np.abs(xx), np.abs(yy)) == 2, 1.0 / 16.0, 0.0)
hole[np.abs(xx) <= 1] = -1.0 / 9.0                       # 12 dark hole/blob
KERNELS.append(hole)
KERNELS.append(-hole)                                    # 13 thin bright line
NC = len(KERNELS)
CW = np.zeros((NC, 1, K, K))
for i, k in enumerate(KERNELS):
    CW[i, 0] = k
CB = np.zeros(NC)

# ----------------------------------------------------------------------------
# branches and feature indices
# ----------------------------------------------------------------------------
B1, B2, B3, B4, B5, B6, B7, B8, B9 = range(9)
SZ = {B1: NC * 196, B2: NC * 49, B3: NC * 28, B4: NC * 28, B5: NC, B6: NC,
      B7: NC * 49, B8: NC * 196, B9: NC}


def i14(ch, r, c):   # binary presence 14x14
    return B1, ch * 196 + r * 14 + c


def i7f(ch, r, c):   # fill 7x7 (of binary map)
    return B2, ch * 49 + r * 7 + c


def irow(ch, y):
    return B3, ch * 28 + y


def icol(ch, x):
    return B4, ch * 28 + x


def igm(ch):
    return B5, ch


def igx(ch):         # unclipped channel maxima (gain references)
    return B6, ch


def i7u(ch, r, c):   # unclipped 7x7 means (gap channels)
    return B7, ch * 49 + r * 7 + c


def i14u(ch, r, c):  # unclipped 14x14 maxima (gap channels)
    return B8, ch * 196 + r * 14 + c


def igref(ch):       # binary gain reference = 0.5 * max of channel
    return B9, ch


GR = 1  # blurred foreground channel used for geometry


class Units:
    def __init__(self):
        self.terms, self.thr, self.b0, self.names = [], [], [], []

    def add(self, name, terms, thr=0.0, b0=0.0):
        d = {}
        for t in terms:
            key = (t[0], t[1])
            d[key] = d.get(key, 0.0) + t[2]
        self.terms.append(d)
        self.thr.append(thr)
        self.b0.append(b0)
        self.names.append(name)
        return len(self.terms) - 1


U = Units()


def rect(bidx, ch, r0, r1, c0, c1, w, per=1):
    n = (r1 - r0 + 1) * (c1 - c0 + 1)
    f = lambda r, c: {B1: i14, B2: i7f, B7: i7u, B8: i14u}[bidx](ch, r, c)
    return [f(r, c) + (w / n / per,) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def m14(ch, r0, r1, c0, c1, w):
    return rect(B1, ch, r0, r1, c0, c1, w)


def m7f(ch, r0, r1, c0, c1, w):
    return rect(B2, ch, r0, r1, c0, c1, w)


def m7u(ch, r0, r1, c0, c1, w):
    return rect(B7, ch, r0, r1, c0, c1, w)


def m14u(ch, r0, r1, c0, c1, w):
    return rect(B8, ch, r0, r1, c0, c1, w)


def gref(w):
    return [igref(GR) + (w,)]


# ----------------------------------------------------------------------------
# silhouette templates, typed from general knowledge of the classes
# '#'=+1 expected item, '+'=+0.5, '.'=-0.5 item must not be here, ' '=ignore
# ----------------------------------------------------------------------------
TPL = {
    0: [  # t-shirt / top: short sleeves, straight body
        "...###..###...",
        "..##########..",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        "...########...",
        "...########...",
        "...########...",
        "...########...",
        "...########...",
        "...########...",
        "...########...",
        "...++++++++..."],
    1: [  # trouser: narrow, two legs with a centre gap (gap must stay empty)
        "....######....",
        "....######....",
        "....######....",
        "....######....",
        "....######....",
        "....###.##....",
        "....###.##....",
        "....###.##....",
        "....###.##....",
        "....###.##....",
        "....###.##....",
        "....###.##....",
        "....###.##....",
        "....+++.++...."],
    2: [  # pullover: long sleeves down the sides, small round neck
        "...###..###...",
        "..##########..",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".++++++++++++.",
        ".............."],
    3: [  # dress: narrow bodice, flaring skirt
        "....######....",
        "....######....",
        "....######....",
        "....######....",
        "....######....",
        "...########...",
        "...########...",
        "...########...",
        "..##########..",
        "..##########..",
        "..##########..",
        ".############.",
        ".############.",
        ".++++++++++++."],
    4: [  # coat: wide, long sleeves, full length, lapel V
        "....##..##....",
        "...###..###...",
        "..####..####..",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".++++++++++++.",
        ".............."],
    5: [  # sandal: flat, open, straps over a sole (toe left)
        "..............",
        "..............",
        "..........+...",
        ".........++...",
        ".......++++...",
        "......+++++...",
        "...++++++++...",
        ".+++++++++++..",
        "##############",
        "##############",
        ".++.........++",
        "..............",
        "..............",
        ".............."],
    6: [  # shirt: collar, long sleeves, straight body
        "...###..###...",
        "..####..####..",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".++++++++++++.",
        ".............."],
    7: [  # sneaker: solid closed upper on a thick sole (toe left)
        "..............",
        "..............",
        "..............",
        "........++....",
        ".......++++...",
        ".....+++++++..",
        "...++++++++++.",
        "++++++++++++++",
        "##############",
        "##############",
        ".++++++++++++.",
        "..............",
        "..............",
        ".............."],
    8: [  # bag: handle loop over a solid rectangular body
        "......++......",
        ".....+..+.....",
        ".....+..+.....",
        "..++++++++++..",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".############.",
        ".++++++++++++.",
        ".............."],
    9: [  # ankle boot: vertical shaft on one side over a foot (shaft right)
        "..............",
        ".........++...",
        "........++++..",
        "........++++..",
        "........++++..",
        "........+++++.",
        ".......++++++.",
        "......+++++++.",
        "..++++++++++++",
        "##############",
        "##############",
        ".++++++++++++.",
        "..............",
        ".............."],
}
for _k, _t in TPL.items():
    assert len(_t) == 14 and all(len(r) == 14 for r in _t), (_k, [len(r) for r in _t])
POS = {"#": 1.0, "+": 0.5}
NEG = {"#": 0.0, "+": 0.0, ".": -0.5, " ": 0.0}

VARIANTS = {
    0: ["base", "sy0.9", "sy1.1", "sx0.9"],
    1: ["base", "sx0.85", "sx1.2", "sy0.95"],
    2: ["base", "sy0.9", "sy1.1", "sx0.9"],
    3: ["base", "sx0.85", "strap", "sy0.9"],
    4: ["base", "sy0.9", "open", "sx0.9"],
    5: ["base", "mir", "sy0.8", "sy1.25"],
    6: ["base", "sy0.9", "sy1.1", "sx0.9"],
    7: ["base", "mir", "sy0.85", "sy1.2"],
    8: ["base", "sy0.8", "sy1.1", "sx0.9"],
    9: ["base", "mir", "sy0.85", "sy1.2"],
}
VCENTER = {0: 6.5, 1: 6.5, 2: 6.5, 3: 6.5, 4: 6.5, 5: 8.0, 6: 6.5, 7: 8.0, 8: 7.0, 9: 6.0}


def transform(t, mode, ctr):
    g = [list(row) for row in t]
    if mode == "mir":
        g = [row[::-1] for row in g]
    elif mode.startswith("sy"):
        s = float(mode[2:])
        g = [list(t[min(13, max(0, int(round(ctr + (r - ctr) / s))))]) for r in range(14)]
    elif mode == "open":  # coat worn open: dark panel gap down the centre
        g = [list(row) for row in t]
        for r in range(2, 12):
            for c in (6, 7):
                g[r][c] = "."
    elif mode == "strap":  # sleeveless dress: bodice reduced to two straps
        g = [list(row) for row in t]
        for r in range(0, 4):
            for c in (3, 6, 7, 10):
                g[r][c] = "."
    elif mode.startswith("sx"):
        s = float(mode[2:])
        g = [list("".join(row[min(13, max(0, int(round(6.5 + (c - 6.5) / s))))]
                        for c in range(14))) for row in t]
    return g


TPL_UNITS = {}
for k in range(10):
    TPL_UNITS[k] = []
    for mode in VARIANTS[k]:
        g = transform(TPL[k], mode, VCENTER[k])
        terms, spos = [], 0.0
        for r in range(14):
            for c in range(14):
                if POS.get(g[r][c], 0.0):
                    terms.append(i14(GR, r, c) + (POS[g[r][c]],))
                    spos += POS[g[r][c]]
        # normalise by template mass: unit output ~= matched fraction - thr
        for r in range(14):
            for c in range(14):
                ch = g[r][c]
                if POS.get(ch, 0.0):
                    terms.append(i14(GR, r, c) + (POS[ch] / spos - POS[ch],))
                elif NEG.get(ch, 0.0):
                    terms.append(i14(GR, r, c) + (NEG[ch] * 3.0 / spos,))
        TPL_UNITS[k].append(U.add(f"tpl{k}_{mode}", terms, thr=0.45))

# ----------------------------------------------------------------------------
# semantic part detectors; every threshold is an exact ratio:
# binary-map features are in units of gref, gap channels reference their own
# unclipped maximum.
# ----------------------------------------------------------------------------
rowmean = [irow(GR, y) + (1.0 / 28,) for y in range(28)]
colmean = [icol(GR, x) + (1.0 / 28,) for x in range(28)]
P = {}
P["tall"] = U.add("height_tall", rowmean + gref(-0.80))
P["short"] = U.add("height_short", [(t[0], t[1], -t[2]) for t in rowmean] + gref(0.60))
P["nottall"] = U.add("height_nottall", [(t[0], t[1], -t[2]) for t in rowmean] + gref(0.88))
P["narrow"] = U.add("width_narrow", [(t[0], t[1], -t[2]) for t in colmean] + gref(0.60))
P["wide"] = U.add("width_wide", colmean + gref(-0.88))
P["fillhi"] = U.add("fill_high", [igm(GR) + (1.0,)] + gref(-0.55))
P["filllo"] = U.add("fill_low", [igm(GR) + (-1.0,)] + gref(0.48))
# trouser leg gap / coat front opening on the centre column.  A dark line
# between two bright sides gives ~2x the response of a plain step edge, so
# the threshold sits between them.
P["split"] = U.add("vg_split_low", m7u(10, 4, 6, 3, 3, 1.0) + [igx(10) + (-0.35,)])
P["opentop"] = U.add("vg_open_top", m7u(10, 1, 3, 3, 3, 1.0) + [igx(10) + (-0.35,)])
# bag handle: empty hole at top centre under a present arc, solid body below,
# no sleeve mass at the top corners
P["handle"] = U.add("handle", m14(GR, 1, 2, 6, 7, -1.0) + gref(0.55)
                    + m14(GR, 0, 1, 5, 8, 1.0) + gref(-0.35)
                    + m7f(GR, 4, 6, 1, 5, 1.0) + gref(-0.75)
                    + m14(GR, 2, 4, 0, 2, -1.0)
                    + m14(GR, 2, 4, 11, 13, -1.0))
# shoe sole separation line, solid sole below, nothing above
P["sole"] = U.add("sole_line", m7u(11, 4, 4, 1, 5, 1.0) + [igx(11) + (-0.35,)]
                  + m7f(GR, 5, 5, 1, 5, 0.8) + gref(-0.60)
                  + [(irow(GR, y) + (-1.0 / 6,)) for y in range(0, 6)] + gref(0.20))
# strappy open upper (sandal): empty mid band over a present sole band, and no
# solid shaft column (boots)
P["strappy"] = U.add("strappy", m7f(GR, 2, 4, 1, 5, -1.0) + gref(0.50)
                     + m7f(GR, 4, 4, 1, 5, 1.0) + gref(-0.60)
                     + m14(GR, 2, 4, 8, 12, -1.0) + gref(0.30))
P["solid"] = U.add("solid_upper", m7f(GR, 2, 4, 2, 5, 1.0) + gref(-0.65))
# boot shaft: SOLID wide column above a foot pointing away
P["shaftR"] = U.add("shaft_r", m14(GR, 2, 6, 9, 12, 1.0) + gref(-0.70)
                    + m14(GR, 9, 11, 0, 4, 0.5) + gref(-0.20)
                    + m14(GR, 0, 4, 0, 5, -0.5))
P["shaftL"] = U.add("shaft_l", m14(GR, 2, 6, 1, 4, 1.0) + gref(-0.70)
                    + m14(GR, 9, 11, 9, 13, 0.5) + gref(-0.20)
                    + m14(GR, 0, 4, 8, 13, -0.5))
# solid toe box of a sneaker
P["toe"] = U.add("toe_solid", m14(GR, 6, 8, 2, 6, 1.0) + gref(-0.80))
# sleeve length
P["slshort"] = U.add("sleeve_short", m14(GR, 2, 5, 0, 2, 0.5) + m14(GR, 2, 5, 11, 13, 0.5)
                     + m14(GR, 7, 10, 0, 2, -0.5) + m14(GR, 7, 10, 11, 13, -0.5)
                     + gref(-0.30))
P["sllong"] = U.add("sleeve_long", m14(GR, 6, 10, 0, 2, 0.5) + m14(GR, 6, 10, 11, 13, 0.5)
                    + gref(-0.55))
# collar V: top centre row present (collar rim) over an empty notch one row
# deeper; round necks are only one row deep so they stay below threshold
P["collar"] = U.add("collar_v", m14(GR, 0, 0, 6, 7, 1.0) + m14(GR, 1, 2, 6, 7, -1.0)
                    + gref(-0.15))
# clutch bag: solid body with a solid top band, no handle hole
P["clutch"] = U.add("clutch", m7f(GR, 4, 6, 1, 5, 1.0) + gref(-0.80)
                    + m14(GR, 0, 2, 1, 5, 1.0) + gref(-0.60))
# dress flare: wide hem, pinched waist
P["flare"] = U.add("flare", m14(GR, 10, 13, 1, 2, 0.5) + m14(GR, 10, 13, 11, 12, 0.5)
                   + m14(GR, 4, 7, 0, 1, -0.5) + m14(GR, 4, 7, 12, 13, -0.5)
                   + gref(-0.25))

# ----------------------------------------------------------------------------
# readout: hand-written rule table over templates + parts
# ----------------------------------------------------------------------------
RULE = {
    0: {"tpl": 4.0, "slshort": 1.2, "sllong": -0.8, "collar": -0.8, "opentop": -0.7,
        "handle": -0.9, "flare": -0.6, "split": -0.6, "short": -1.0},
    1: {"tpl": 4.0, "narrow": 1.6, "split": 1.4, "tall": 0.6, "handle": -1.0,
        "sllong": -0.8, "slshort": -0.8, "flare": -1.0},
    2: {"tpl": 4.0, "sllong": 1.2, "slshort": -0.7, "collar": -0.7, "opentop": -0.9,
        "handle": -0.9, "flare": -0.6, "split": -0.4},
    3: {"tpl": 4.0, "flare": 1.8, "filllo": 0.6, "narrow": 0.6, "sllong": -0.7, "slshort": -0.5,
        "handle": -0.7, "split": -0.6, "opentop": -0.4},
    4: {"tpl": 4.0, "opentop": 1.2, "split": 0.6, "sllong": 1.0, "collar": 0.6,
        "flare": -0.8, "handle": -0.7, "slshort": -0.6},
    5: {"tpl": 4.0, "short": 1.4, "strappy": 1.3, "solid": -1.0, "shaftR": -0.5,
        "shaftL": -1.0, "sole": 0.3, "filllo": 0.4, "toe": -0.8, "solid": -1.4, "strappy": 1.5},
    6: {"tpl": 4.0, "collar": 1.4, "sllong": 0.8, "slshort": -0.4, "opentop": 0.5,
        "flare": -0.5, "handle": -0.6},
    7: {"tpl": 4.0, "short": 1.4, "solid": 1.6, "sole": 0.8, "strappy": -1.2,
        "shaftR": -0.6, "shaftL": -0.6, "toe": 0.8},
    8: {"tpl": 4.0, "handle": 1.6, "clutch": 1.4, "wide": 0.6, "fillhi": 0.8, "tall": 0.5, "sllong": -1.0,
        "slshort": -1.0, "flare": -0.6, "split": -0.5, "short": -0.5},
    9: {"tpl": 4.0, "shaftR": 1.5, "shaftL": 1.5, "nottall": 0.6, "strappy": -1.2,
        "solid": -0.3, "short": -0.6},
}
BIAS = {k: -1.0 for k in range(10)}

# ----------------------------------------------------------------------------
# assemble
# ----------------------------------------------------------------------------
nu = len(U.terms)
assert nu <= H, (nu, H)
W = {b: np.zeros((H, SZ[b])) for b in SZ}
THR = np.zeros(H)
B0 = np.zeros(H)
for u, (d, thr, b0) in enumerate(zip(U.terms, U.thr, U.b0)):
    for (br, idx), w in d.items():
        W[br][u, idx] = w
    THR[u] = thr
    B0[u] = b0
# template thresholds are in units of the binary gain reference (branch B9)
for u in range(nu):
    if THR[u]:
        W[B9][u, GR] -= THR[u]

ROUT = np.zeros((10, H))
for k in range(10):
    r = RULE[k]
    for tu in TPL_UNITS[k]:
        ROUT[k, tu] = r["tpl"]
    for pname, w in r.items():
        if pname != "tpl":
            ROUT[k, P[pname]] = w

E = np.eye(SZ[B1], dtype=np.float32)
E3 = np.eye(SZ[B3], dtype=np.float32)
GCOL = np.zeros((SZ[B1], NC)); GCOL[:, 0] = -0.5          # clip level 0.5*max(F)
GCOL3 = np.zeros((SZ[B3], NC)); GCOL3[:, 0] = -0.5
GCOL4 = np.zeros((SZ[B4], NC)); GCOL4[:, 0] = -0.5

layers = [
    {"name": "c1", "op": "conv", "inputs": ["image"], "out": NC, "kernel": [K, K],
     "stride": 1, "padding": 2},
    {"name": "n1", "op": "norm", "inputs": ["c1"]},
    {"name": "r1", "op": "relu", "inputs": ["n1"]},
    # binary presence map: min(x, 0.5*max) = x - relu(x - 0.5*max)
    {"name": "p6", "op": "maxpool", "inputs": ["r1"], "size": [28, 28], "stride": [28, 28]},
    {"name": "f6", "op": "reshape", "inputs": ["p6"], "shape": [SZ[B6]]},
    {"name": "q1", "op": "maxpool", "inputs": ["r1"], "size": [2, 2], "stride": [2, 2]},
    {"name": "q1f", "op": "reshape", "inputs": ["q1"], "shape": [SZ[B1]]},
    {"name": "g1", "op": "dense", "inputs": ["f6"], "out": SZ[B1]},
    {"name": "s1", "op": "add", "inputs": ["q1f", "g1"]},
    {"name": "t1", "op": "relu", "inputs": ["s1"]},
    {"name": "n1d", "op": "dense", "inputs": ["t1"], "out": SZ[B1]},
    {"name": "b1f", "op": "add", "inputs": ["q1f", "n1d"]},
    {"name": "b14", "op": "reshape", "inputs": ["b1f"], "shape": [NC, 14, 14]},
    {"name": "q3", "op": "maxpool", "inputs": ["r1"], "size": [1, 28], "stride": [1, 28]},
    {"name": "q3f", "op": "reshape", "inputs": ["q3"], "shape": [SZ[B3]]},
    {"name": "g3", "op": "dense", "inputs": ["f6"], "out": SZ[B3]},
    {"name": "s3", "op": "add", "inputs": ["q3f", "g3"]},
    {"name": "t3", "op": "relu", "inputs": ["s3"]},
    {"name": "n3", "op": "dense", "inputs": ["t3"], "out": SZ[B3]},
    {"name": "f3", "op": "add", "inputs": ["q3f", "n3"]},
    {"name": "q4", "op": "maxpool", "inputs": ["r1"], "size": [28, 1], "stride": [28, 1]},
    {"name": "q4f", "op": "reshape", "inputs": ["q4"], "shape": [SZ[B4]]},
    {"name": "g4", "op": "dense", "inputs": ["f6"], "out": SZ[B4]},
    {"name": "s4", "op": "add", "inputs": ["q4f", "g4"]},
    {"name": "t4", "op": "relu", "inputs": ["s4"]},
    {"name": "n4", "op": "dense", "inputs": ["t4"], "out": SZ[B4]},
    {"name": "f4", "op": "add", "inputs": ["q4f", "n4"]},
    # pooled branches
    {"name": "p2", "op": "avgpool", "inputs": ["b14"], "size": [2, 2], "stride": [2, 2]},
    {"name": "f2", "op": "reshape", "inputs": ["p2"], "shape": [SZ[B2]]},
    {"name": "p5", "op": "avgpool", "inputs": ["b14"], "size": [14, 14], "stride": [14, 14]},
    {"name": "f5", "op": "reshape", "inputs": ["p5"], "shape": [SZ[B5]]},
    {"name": "p9", "op": "maxpool", "inputs": ["b14"], "size": [14, 14], "stride": [14, 14]},
    {"name": "f9", "op": "reshape", "inputs": ["p9"], "shape": [SZ[B9]]},
    {"name": "p7", "op": "avgpool", "inputs": ["r1"], "size": [4, 4], "stride": [4, 4]},
    {"name": "f7", "op": "reshape", "inputs": ["p7"], "shape": [SZ[B7]]},
    {"name": "f8", "op": "reshape", "inputs": ["q1"], "shape": [SZ[B8]]},
    {"name": "d1", "op": "dense", "inputs": ["q1f"], "out": H},   # placeholder replaced
    {"name": "d2", "op": "dense", "inputs": ["f2"], "out": H},
    {"name": "d3", "op": "dense", "inputs": ["f3"], "out": H},
    {"name": "d4", "op": "dense", "inputs": ["f4"], "out": H},
    {"name": "d5", "op": "dense", "inputs": ["f5"], "out": H},
    {"name": "d6", "op": "dense", "inputs": ["f6"], "out": H},
    {"name": "d7", "op": "dense", "inputs": ["f7"], "out": H},
    {"name": "d8", "op": "dense", "inputs": ["f8"], "out": H},
    {"name": "d9", "op": "dense", "inputs": ["f9"], "out": H},
    {"name": "hs", "op": "add", "inputs": ["d1", "d2", "d3", "d4", "d5", "d6", "d7",
                                           "d8", "d9"]},
    {"name": "hh", "op": "relu", "inputs": ["hs"]},
    {"name": "out", "op": "dense", "inputs": ["hh"], "out": 10},
]
# d1 reads the *binary* presence map
layers[[L["name"] for L in layers].index("d1")]["inputs"] = ["b1f"]

weights = {
    "c1.W": CW.astype(np.float32), "c1.b": CB.astype(np.float32),
    "g1.W": GCOL.astype(np.float32), "g1.b": np.zeros(SZ[B1], np.float32),
    "n1d.W": (-E).astype(np.float32), "n1d.b": np.zeros(SZ[B1], np.float32),
    "g3.W": GCOL3.astype(np.float32), "g3.b": np.zeros(SZ[B3], np.float32),
    "n3.W": (-E3).astype(np.float32), "n3.b": np.zeros(SZ[B3], np.float32),
    "g4.W": GCOL4.astype(np.float32), "g4.b": np.zeros(SZ[B4], np.float32),
    "n4.W": (-E3).astype(np.float32), "n4.b": np.zeros(SZ[B4], np.float32),
    "d1.W": W[B1].astype(np.float32), "d1.b": B0.astype(np.float32),
    "d2.W": W[B2].astype(np.float32), "d2.b": np.zeros(H, np.float32),
    "d3.W": W[B3].astype(np.float32), "d3.b": np.zeros(H, np.float32),
    "d4.W": W[B4].astype(np.float32), "d4.b": np.zeros(H, np.float32),
    "d5.W": W[B5].astype(np.float32), "d5.b": np.zeros(H, np.float32),
    "d6.W": W[B6].astype(np.float32), "d6.b": np.zeros(H, np.float32),
    "d7.W": W[B7].astype(np.float32), "d7.b": np.zeros(H, np.float32),
    "d8.W": W[B8].astype(np.float32), "d8.b": np.zeros(H, np.float32),
    "d9.W": W[B9].astype(np.float32), "d9.b": np.zeros(H, np.float32),
    "out.W": ROUT.astype(np.float32),
    "out.b": np.array([BIAS[k] for k in range(10)], np.float32),
}


def main(dest):
    import os
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "network.json"), "w") as f:
        json.dump({"layers": layers}, f)
    np.savez(os.path.join(dest, "weights.npz"), **weights)
    total = sum(int(np.prod(v.shape)) for v in weights.values())
    print(f"wrote {dest}: {len(layers)} layers, {total} weights, {nu} hidden units used")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
