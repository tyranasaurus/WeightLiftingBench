"""Hand-designed sign-language letter network.

Every number comes from formulas (Gaussian-derivative ridge / blob filters) or from
a typed shape model of the 24 ASL hand shapes (which side the thumb is on, how many
finger stripes sit on top of the fist, where holes / hooks / crossed fingers appear).
No data is read and nothing is fitted.

Pipeline
--------
norm(image)                 per-image contrast normalisation
c1 conv 9x9   12 signed filters: ridges at 4 orientations x2 polarities,
              difference-of-Gaussian blob x2, wide Gaussian x2
r1 relu
c2 conv 1x1   polarity-agnostic energies R0,R45,R90,R135 + BLOB + MASS0
p1 maxpool 2  28x28 -> 14x14
c3 conv 7x7   FV,FH,FD1,FD2 (elongated ridge integration = finger evidence),
              MASS (blurred blob), BODY (edge density),
              HOLE  (narrow centre-surround of edge density: any valley / hole),
              HOLEW (wide centre-surround: only gaps wider than ~3 px)
r3 relu
reg conv 14x14  mean of each channel over coarse boxes -> position features
patc conv 7x7   sliding configuration detectors (stripe counts, ring, cross,
                hook, L-corner)
patm maxpool 8  "pattern present anywhere"
feat add, fv reshape -> interpretable feature vector
sc dense 24     hand-written rule table over those features
"""
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------- filters ----


def gauss(t, s):
    return np.exp(-(t ** 2) / (2.0 * s * s))


def ridge(theta_deg, su=1.3, sv=3.0, n=9):
    """Bright-bar (ridge) detector: bar long axis rotated theta from vertical."""
    th = np.deg2rad(theta_deg)
    t = np.arange(n) - (n - 1) / 2.0
    x, y = np.meshgrid(t, t)              # x: column, y: row
    u = x * np.cos(th) - y * np.sin(th)   # across the bar
    v = x * np.sin(th) + y * np.cos(th)   # along the bar
    g2 = (u * u - su * su) / (su ** 4) * gauss(u, su)     # d2/du2 of gaussian
    r = -g2 * gauss(v, sv)
    return r / np.abs(r).max()


def blob(n=9, s1=1.5, s2=3.2):
    t = np.arange(n) - (n - 1) / 2.0
    x, y = np.meshgrid(t, t)
    r2 = np.sqrt(x * x + y * y)
    d = gauss(r2, s1) - 0.55 * gauss(r2, s2)
    return d / np.abs(d).max()


def smooth(n=9, s=3.5):
    t = np.arange(n) - (n - 1) / 2.0
    x, y = np.meshgrid(t, t)
    d = gauss(np.sqrt(x * x + y * y), s)
    return d / d.max()


def g2d(n, s):
    t = np.arange(n) - (n - 1) / 2.0
    x, y = np.meshgrid(t, t)
    return gauss(np.sqrt(x * x + y * y), s)


# ------------------------------------------------------- template helpers ----
CH = dict(FV=0, FH=1, FD1=2, FD2=3, MASS=4, BODY=5, HOLE=6, HOLEW=7)
NC = 8


def blank7():
    return np.zeros((NC, 7, 7))


def rect(a, ch, y0, y1, x0, x1, w):
    a[CH[ch], y0:y1 + 1, x0:x1 + 1] += w


def soft(a, s):
    """small separable smoothing for shift tolerance"""
    n = 2 * int(np.ceil(3 * s)) + 1
    k = gauss(np.arange(n) - n // 2, s)
    k /= k.sum()
    for c in range(a.shape[0]):
        for ax in (0, 1):
            a[c] = np.apply_along_axis(lambda m: np.convolve(m, k, "same"), ax, a[c])
    return a


def normalise(a):
    p = a[a > 0].sum()
    if p > 0:
        a /= p
    return a


def adjust(t):
    """BODY lights up for any hand -> drop it from the logic; MASS is only
    weakly discriminative -> halve it; absence conditions carry most of the
    discrimination -> boost negatives."""
    t[CH["BODY"]] = 0.0
    t[CH["MASS"]] *= 0.5
    t[t < 0] *= 1.5
    return t


# ------------------------------------------- sliding configuration patterns --
# 7x7 windows on the 14x14 grid: stripe counting and special shapes.
L = {}
L["V1"] = [("FV", 0, 6, 2, 4, 1.), ("FV", 0, 6, 0, 1, -1.), ("FV", 0, 6, 5, 6, -1.)]
L["V2"] = [("FV", 0, 6, 1, 2, 1.), ("FV", 0, 6, 4, 5, 1.), ("FV", 0, 6, 3, 3, -1.),
           ("HOLEW", 0, 6, 3, 3, .9), ("FV", 0, 6, 0, 0, -.7), ("FV", 0, 6, 6, 6, -.7)]
L["V3"] = [("FV", 0, 6, 1, 1, 1.), ("FV", 0, 6, 3, 3, 1.), ("FV", 0, 6, 5, 5, 1.),
           ("HOLEW", 0, 6, 2, 2, .8), ("HOLEW", 0, 6, 4, 4, .8), ("FV", 0, 6, 0, 0, -.6),
           ("FV", 0, 6, 2, 2, -.8), ("FV", 0, 6, 4, 4, -.8), ("FV", 0, 6, 6, 6, -.6)]
L["V4"] = [("FV", 0, 6, 0, 0, 1.), ("FV", 0, 6, 2, 2, 1.), ("FV", 0, 6, 4, 4, 1.),
           ("FV", 0, 6, 6, 6, 1.), ("HOLE", 0, 6, 1, 1, .6), ("HOLE", 0, 6, 3, 3, .6),
           ("HOLE", 0, 6, 5, 5, .6), ("FV", 0, 6, 1, 1, -.8), ("FV", 0, 6, 3, 3, -.8),
           ("FV", 0, 6, 5, 5, -.8)]
L["H1"] = [("FH", 2, 4, 0, 6, 1.), ("FH", 0, 1, 0, 6, -1.), ("FH", 5, 6, 0, 6, -1.)]
L["H2"] = [("FH", 1, 1, 0, 6, 1.), ("FH", 3, 3, 0, 6, 1.), ("FH", 5, 5, 0, 6, 1.),
           ("HOLEW", 2, 2, 0, 6, .7), ("HOLEW", 4, 4, 0, 6, .7), ("FH", 0, 0, 0, 6, -.6),
           ("FH", 2, 2, 0, 6, -.8), ("FH", 4, 4, 0, 6, -.8), ("FH", 6, 6, 0, 6, -.6)]
L["CROSS"] = [("FD1", 1, 5, 1, 5, .8), ("FD2", 1, 5, 1, 5, .8), ("FV", 1, 5, 1, 5, -.4),
              ("HOLE", 0, 1, 3, 3, .4), ("HOLE", 5, 6, 3, 3, .4), ("HOLE", 3, 3, 0, 1, .4),
              ("HOLE", 3, 3, 5, 6, .4), ("HOLE", 3, 3, 3, 3, -.5)]
L["RING"] = [("MASS", 0, 1, 0, 6, 1.), ("MASS", 5, 6, 0, 6, 1.), ("MASS", 2, 4, 0, 1, 1.),
             ("MASS", 2, 4, 5, 6, 1.), ("HOLEW", 2, 4, 2, 4, 1.2), ("MASS", 2, 4, 2, 4, -.6),
             ("HOLEW", 0, 1, 2, 4, -1.), ("HOLEW", 5, 6, 2, 4, -1.),
             ("HOLEW", 2, 4, 0, 1, -1.), ("HOLEW", 2, 4, 5, 6, -1.)]
L["HOOK"] = [("FV", 3, 6, 2, 4, 1.), ("FH", 1, 2, 2, 5, 1.), ("FV", 0, 1, 2, 4, -.8),
             ("FV", 3, 6, 0, 1, -.7), ("FV", 3, 6, 5, 6, -.7)]
L["LCOR"] = [("FV", 0, 3, 1, 3, 1.), ("FH", 4, 6, 3, 6, 1.), ("FV", 4, 6, 0, 1, -.5),
             ("FH", 0, 2, 4, 6, -.5)]
PATS = ["V1", "V2", "V3", "V4", "H1", "H2", "RING", "CROSS", "HOOK", "LCOR"]

# ------------------------------------------------------------- rule table ----
# Region features: channel mean over a coarse box.  3x3 grid regions
# (top/mid/bot x left/centre/right) plus finer bands: FT/HT/HTW = vertical
# stripe / narrow gap / wide gap energy in the top strip per column band,
# FB = stripes in the bottom strip (downward fingers), FR/HR = horizontal
# stripe / gap energy on the right side per row band.  PAT_* = configuration
# detector present anywhere.
BX = dict(B0=(2, 5), B1=(5, 8), B2=(8, 11), B3=(11, 13))
RY = dict(R0=(2, 5), R1=(5, 8), R2=(8, 11), R3=(11, 13))
REG = {}
for r in "TMB":
    y0, y1 = dict(T=(0, 4), M=(4, 9), B=(9, 13))[r]
    for k in "LCR":
        x0, x1 = dict(L=(0, 4), C=(4, 9), R=(9, 13))[k]
        for c in ["FV", "FH", "FD", "HOLE", "MASS"]:
            REG[f"{c}_{r}{k}"] = (c, y0, y1, x0, x1)
REG["HOLEW_MC"] = ("HOLEW", 4, 9, 4, 9)
REG["HOLEW_BC"] = ("HOLEW", 8, 12, 5, 10)
for b in "0123":
    x0, x1 = BX["B" + b]
    REG[f"FT_B{b}"] = ("FV", 0, 6, x0, x1)
    REG[f"HT_B{b}"] = ("HOLE", 0, 6, x0, x1)
    REG[f"HTW_B{b}"] = ("HOLEW", 0, 6, x0, x1)
    REG[f"FB_B{b}"] = ("FV", 8, 13, x0, x1)
    y0, y1 = RY["R" + b]
    REG[f"FR_R{b}"] = ("FH", y0, y1, 8, 13)
    REG[f"HR_R{b}"] = ("HOLE", y0, y1, 8, 13)
FEATS = list(REG) + [f"PAT_{p}" for p in PATS]

# per-feature-family divisors so every feature enters the rule table on a
# comparable ~0..1 scale (typed from the shape model, not from data)
SCALE = dict(FV=5., FH=3.5, FD=8., MASS=10., HOLEW=7., HOLE=8., HTW=7., HT=8.,
             FT=5., FB=5., FR=3.5, HR=8.,
             PAT_V1=6., PAT_V2=6., PAT_V3=6., PAT_V4=6., PAT_H1=3., PAT_H2=5.,
             PAT_RING=2., PAT_CROSS=10., PAT_HOOK=2.5, PAT_LCOR=4.)

RULE = {
    "A": dict(PAT_V4=.5, MASS_MC=.7, MASS_TC=.3, FV_MR=.6, FH_MC=-.5, HTW_B1=-.4,
              HTW_B2=-.4),
    "B": dict(FT_B0=.6, FT_B1=.7, FT_B2=.7, FT_B3=.6, MASS_TC=.5, PAT_V2=-.5,
              MASS_MC=-.4, HTW_B1=-.5, HTW_B2=-.5),
    "C": dict(FD_TC=1.0, FH_ML=.9, HT_B0=.7, FH_TC=.5, FH_BC=.5, PAT_H2=.3,
              HOLE_MR=.6, MASS_MC=-.6, FT_B1=-.6, FT_B2=-.6, MASS_MR=-.5),
    "D": dict(FT_B2=.8, FT_B1=.3, HOLEW_BC=.7, HR_R2=.5, PAT_H2=.4, FD_MC=.3,
              FT_B0=-.6, PAT_V2=-.4, MASS_BC=-.3),
    "E": dict(MASS_MC=.9, MASS_TC=.5, FH_BL=.3, FH_MC=-.6, FV_TC=-.7, PAT_V2=-.5,
              PAT_V3=-.5, FT_B1=-.4, FT_B2=-.4, HTW_B1=-.4, HTW_B2=-.4),
    "F": dict(FT_B1=.5, FT_B2=.5, FT_B3=.6, HTW_B1=.4, HTW_B2=.4, HOLEW_BC=.5,
              PAT_H2=.3, FT_B0=-.5),
    "G": dict(FR_R0=1.2, FR_R1=.5, FH_MR=.7, MASS_ML=.7, FH_MC=.3, HR_R1=.4,
              FT_B1=-.6, FT_B2=-.6, FB_B1=-.5),
    "H": dict(FR_R0=.7, FR_R1=.8, FH_MR=.5, MASS_ML=.7, PAT_H1=.7, FH_MC=.3,
              FT_B1=-.6, FT_B2=-.6, FB_B1=-.5),
    "I": dict(FT_B0=.9, FV_ML=.5, MASS_MC=.5, MASS_BC=.2, FT_B1=-.8, FT_B2=-.8,
              FT_B3=-.6, FH_MR=-.6, FR_R1=-.5, PAT_V4=-.4),
    "K": dict(FT_B1=.6, FT_B2=.6, FV_MC=.6, HTW_B1=.4, HTW_B2=.4, MASS_BC=.3,
              FT_B0=-.5, FT_B3=-.5, HOLEW_BC=-.5),
    "L": dict(FT_B1=.8, FR_R2=1.0, FR_R3=.5, PAT_LCOR=.8, MASS_BC=.3, FT_B2=-.6,
              FR_R0=-.5, FT_B0=-.4),
    "M": dict(PAT_V3=.8, FV_TC=.5, MASS_MC=.8, FH_MC=-.6, FT_B0=-.4, FT_B3=-.4,
              HTW_B1=-.4, HTW_B2=-.4),
    "N": dict(PAT_V2=.8, FV_TC=.5, MASS_MC=.8, FH_MC=-.6, FT_B0=-.5, FT_B3=-.5,
              HTW_B1=-.4, HTW_B2=-.4),
    "O": dict(HOLEW_MC=1.0, FD_MC=.5, PAT_H2=.4, MASS_TC=.3, MASS_BC=.3,
              MASS_MC=-.6, FT_B1=-.6, FT_B2=-.6, FH_ML=-.4, MASS_MR=.4,
              HOLE_MR=-.5, FD_TC=-.3),
    "P": dict(FR_R0=.9, FB_B2=1.0, MASS_ML=.6, FH_MC=.3, FR_R1=-.6, FT_B1=-.6,
              FT_B2=-.5, FR_R2=-.4),
    "Q": dict(FB_B1=1.2, MASS_TC=.8, MASS_MC=.4, FT_B1=-.6, FT_B2=-.6, MASS_BC=-.5,
              FH_BC=-.4, FD_TC=-.4, HT_B0=-.3),
    "R": dict(FT_B1=.6, FT_B2=.6, FD_TC=.6, FD_MC=.3, MASS_BC=.3, PAT_V2=.3,
              FT_B0=-.5, FT_B3=-.5, FH_MC=-.4, FR_R2=-.4, PAT_LCOR=-.4),
    "S": dict(FH_MC=.9, MASS_MC=.7, PAT_V4=.4, FV_TC=.3, FV_MR=-.4, HTW_B1=-.4,
              HTW_B2=-.4),
    "T": dict(FR_R1=1.2, FH_MR=.6, MASS_ML=.7, PAT_H1=.5, PAT_LCOR=.4, FR_R0=-.8,
              FR_R2=-.5, FT_B1=-.4, FT_B2=-.4, FV_TC=.5),
    "U": dict(FT_B1=.9, FT_B2=.6, MASS_BC=.4, HTW_B1=-.6, HTW_B2=-.6, FT_B0=-.7,
              FT_B3=-.7, MASS_MC=-.4, PAT_V2=-.4),
    "V": dict(FT_B1=.7, FT_B2=.7, HTW_B1=.8, HTW_B2=.8, MASS_BC=.3, FT_B0=-.5,
              FT_B3=-.5, MASS_MC=-.3),
    "W": dict(FT_B0=.6, FT_B1=.5, FT_B2=.5, FT_B3=.5, HTW_B0=.5, HTW_B1=.5,
              HTW_B2=.5, PAT_V2=.5, FV_MC=-.4, MASS_MC=-.3),
    "X": dict(FH_TC=.6, PAT_HOOK=.6, FT_B2=.4, MASS_BC=.3, FD_MC=-.4, FT_B0=-.5,
              FT_B3=-.5, HTW_B1=-.5, HTW_B2=-.5),
    "Y": dict(FT_B0=.8, FV_ML=.6, FR_R1=.8, FR_R2=.5, FH_MR=.5, MASS_MC=.4,
              FT_B1=-.5, FT_B2=-.6),
}
CLASSES = "A B C D E F G H I K L M N O P Q R S T U V W X Y".split()
BIAS = {c: 0.0 for c in CLASSES}

# ------------------------------------------------------------------ build ----


def build():
    # ---- c1 filters
    f = []
    for th in (0, 45, 90, 135):
        r = ridge(th)
        f += [r, -r]
    b = blob()
    f += [b, -b]
    s = smooth()
    f += [s, -s]
    c1W = np.stack(f)[:, None, :, :].astype(np.float64)          # (12,1,9,9)
    c1b = np.zeros(12)

    # ---- c2 : energies
    c2W = np.zeros((6, 12, 1, 1))
    for i in range(4):
        c2W[i, 2 * i, 0, 0] = 1.0
        c2W[i, 2 * i + 1, 0, 0] = 1.0
    # blob channels run ~7x hotter than ridge channels on hands -> attenuate
    c2W[4, 8, 0, 0] = 0.15
    c2W[4, 9, 0, 0] = 0.15
    c2W[5, 10, 0, 0] = 0.15
    c2W[5, 11, 0, 0] = 0.15
    c2b = np.zeros(6)

    # ---- c3 : finger / mass / hole integration on 14x14
    c3W = np.zeros((8, 6, 7, 7))
    yy, xx = np.mgrid[0:7, 0:7]
    dy, dx = yy - 3, xx - 3
    linev = gauss(dy, 2.2)
    across = np.where(dx == 0, 1.0, np.where(np.abs(dx) == 1, -0.35, 0.0))
    c3W[0, 0] = linev * across                                    # FV from R0
    c3W[1, 2] = (linev * across).T                                # FH from R90
    for sgn in (2, 3):
        k = np.zeros((7, 7))
        for t in range(-3, 4):
            py, px = (-t, t) if sgn == 2 else (t, t)
            k[py + 3, px + 3] = gauss(t, 2.2)
            for o in ((1, 1), (-1, -1)):
                qy, qx = py + o[0], px + o[1]
                if 0 <= qy < 7 and 0 <= qx < 7:
                    k[qy, qx] -= 0.35 * gauss(t, 2.2)
        c3W[sgn, sgn] = k                            # FD1 from R45, FD2 from R135
    c3W[4, 4] = g2d(7, 2.0)
    c3W[4, 5] = 0.8 * g2d(7, 2.5)
    for c in range(4):
        c3W[5, c] = 0.5 * g2d(7, 2.2)
    # HOLE / HOLEW: centre-surround of edge density -> valleys, gaps, holes
    for c in range(4):
        c3W[6, c] = 0.9 * g2d(7, 2.4) - 1.2 * g2d(7, 0.9)
        c3W[7, c] = 0.9 * g2d(7, 3.2) - 1.2 * g2d(7, 1.8)
    # equalise dynamic range: every output channel kernel gets L1 norm 2
    for o in range(c3W.shape[0]):
        c3W[o] /= np.abs(c3W[o]).sum() / 2.0
    c3b = np.zeros(8)

    # ---- reg: mean of each channel over its box
    nf = len(FEATS)
    regW = np.zeros((nf, NC, 14, 14))
    for i, name in enumerate(REG):
        c, y0, y1, x0, x1 = REG[name]
        area = (y1 - y0 + 1) * (x1 - x0 + 1)
        if c == "FD":
            regW[i, CH["FD1"], y0:y1 + 1, x0:x1 + 1] = 0.5 / area
            regW[i, CH["FD2"], y0:y1 + 1, x0:x1 + 1] = 0.5 / area
        else:
            regW[i, CH[c], y0:y1 + 1, x0:x1 + 1] = 1.0 / area
    regb = np.zeros(nf)

    # ---- patc / patm: sliding configuration detectors, max over positions
    patW = np.zeros((nf, NC, 7, 7))
    for j, pname in enumerate(PATS):
        p = blank7()
        for (ch, y0, y1, x0, x1, ww) in L[pname]:
            rect(p, ch, y0, y1, x0, x1, ww)
        patW[len(REG) + j] = normalise(soft(adjust(p), 0.5))
    patb = np.zeros(nf)

    # ---- dense rule table
    scW = np.zeros((24, nf))
    for ci, cname in enumerate(CLASSES):
        for fname, w in RULE[cname].items():
            scW[ci, FEATS.index(fname)] = w / SCALE[fname.split("_")[0] if not
                                                    fname.startswith("PAT_")
                                                    else fname.split("_")[0] + "_" +
                                                    fname.split("_")[1]]
        # (no row re-centering: a constant per row would couple the score to the
        #  total feature activity of the image)
    scb = np.array([BIAS[c] for c in CLASSES])

    layers = [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "c1", "op": "conv", "inputs": ["n0"], "out": 12, "kernel": [9, 9],
         "stride": 1, "padding": 4},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "c2", "op": "conv", "inputs": ["r1"], "out": 6, "kernel": [1, 1],
         "stride": 1, "padding": 0},
        {"name": "p1", "op": "maxpool", "inputs": ["c2"], "size": [2, 2],
         "stride": [2, 2]},
        {"name": "c3", "op": "conv", "inputs": ["p1"], "out": 8, "kernel": [7, 7],
         "stride": 1, "padding": 3},
        {"name": "r3", "op": "relu", "inputs": ["c3"]},
        {"name": "reg", "op": "conv", "inputs": ["r3"], "out": nf, "kernel": [14, 14],
         "stride": 1, "padding": 0},
        {"name": "patc", "op": "conv", "inputs": ["r3"], "out": nf, "kernel": [7, 7],
         "stride": 1, "padding": 0},
        {"name": "patm", "op": "maxpool", "inputs": ["patc"], "size": [8, 8],
         "stride": [8, 8]},
        {"name": "feat", "op": "add", "inputs": ["reg", "patm"]},
        {"name": "fv", "op": "reshape", "inputs": ["feat"], "shape": [nf]},
        {"name": "sc", "op": "dense", "inputs": ["fv"], "out": 24},
    ]
    weights = {
        "c1.W": c1W.astype(np.float32), "c1.b": c1b.astype(np.float32),
        "c2.W": c2W.astype(np.float32), "c2.b": c2b.astype(np.float32),
        "c3.W": c3W.astype(np.float32), "c3.b": c3b.astype(np.float32),
        "reg.W": regW.astype(np.float32), "reg.b": regb.astype(np.float32),
        "patc.W": patW.astype(np.float32), "patc.b": patb.astype(np.float32),
        "sc.W": scW.astype(np.float32), "sc.b": scb.astype(np.float32),
    }
    return {"layers": layers}, weights


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/app/out"
    net, w = build()
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "network.json"), "w") as fh:
        json.dump(net, fh)
    np.savez(os.path.join(out, "weights.npz"), **w)
    print("wrote", out, sum(int(v.size) for v in w.values()), "weights")
