"""Builds a hand-designed CIFAR-style 10-class network.  python3 build.py OUT_DIR

Nothing here is learned or fitted: every weight is written from a formula or from the hand-typed
tables below (colour thresholds, filter shapes, and a table of how strongly each class tends to show
each scene property).  No files are read.

Pipeline
  1. "px"   9x9 conv (padding 3, output 30x30 = image rows/cols 1..30): colour differences, luminance,
            forward differences, long horizontal / vertical line detectors, dark-blob (wheel) detector.
  2. "pxr"  1x1 conv: soft thresholds (ramps) on those quantities, two relu units per ramp.
  3. "pxf"  1x1 conv: per-pixel soft indicators = ramps and ANDs of ramps
            (blue, green, warm/brown, vivid red, vivid yellow, neutral, white, smooth, busy, lines, wheel, sky).
  4. avgpool 6x6 -> 5x5 grid of "fraction of the cell that is X".
  5. a second small branch (9x9 valid conv) measures diagonal straight edges and coarse vertical vs horizontal
     luminance change, averaged over the image; a third runs horizontal / vertical line detectors on the
     2x2-averaged image, so only edges that stay straight for ~18 pixels count.
  6. dense layers: soft cut points on region fractions -> soft bins -> naive-Bayes log-likelihoods typed by hand.
"""
import json, sys, os
import numpy as np

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
WL = np.array([0.30, 0.59, 0.11])          # luminance weights for R, G, B

# ----------------------------------------------------------------------------------------------
# helpers for building linear filters on the RGB image
# ----------------------------------------------------------------------------------------------
class Filters:
    """Collects channels of one conv layer on the image; each channel = (kernel (3,k,k), bias)."""
    def __init__(self, k):
        self.k, self.c, self.names, self.ker, self.bias = k, k // 2, [], [], []

    def new(self, name, bias=0.0):
        self.names.append(name); self.ker.append(np.zeros((3, self.k, self.k))); self.bias.append(bias)
        return self.ker[-1]

    def lum(self, ker, oy, ox, w):            # add w * luminance at offset (oy, ox) from the centre pixel
        ker[:, self.c + oy, self.c + ox] += w * WL

    def rgb(self, ker, w3):                     # colour combination at the centre pixel
        ker[:, self.c, self.c] += np.asarray(w3, float)

    def signed(self, name, fill):               # two channels: relu(f) and relu(-f)
        kp = self.new(name + "+"); fill(kp)
        kn = self.new(name + "-"); fill(kn); kn *= -1

    def arrays(self):
        return np.stack(self.ker).astype(np.float64), np.array(self.bias, float)


class Stage:
    """Builds 1x1-conv / dense stages from named inputs.
    ramp(name, expr, t0, t1): u = (expr - t0) / (t1 - t0) gives two relu units u and u - 1, so that
    (unit1 - unit2) = clip(u, 0, 1): 0 at t0, 1 at t1 (t1 < t0 gives a falling ramp)."""
    def __init__(self, in_names):
        self.idx = {n: i for i, n in enumerate(in_names)}
        self.rows, self.bias, self.names = [], [], []

    def unit(self, name, expr, bias):
        w = np.zeros(len(self.idx))
        for k, v in expr.items():
            w[self.idx[k]] += v
        self.rows.append(w); self.bias.append(bias); self.names.append(name)

    def ramp(self, name, expr, t0, t1):
        s = 1.0 / (t1 - t0)
        e = {k: v * s for k, v in expr.items()}
        self.unit(name + "/a", e, -t0 * s)
        self.unit(name + "/b", e, -t0 * s - 1.0)

    def passthrough(self, name, expr):
        self.unit(name, expr, 0.0)

    def all_of(self, name, ramps, extra=None):
        """relu(sum of ramp outputs - (n - 1)): a soft AND of n ramps (just the ramp when n = 1)."""
        e = {}
        for r in ramps:
            e[r + "/a"] = e.get(r + "/a", 0) + 1.0
            e[r + "/b"] = e.get(r + "/b", 0) - 1.0
        self.unit(name, e, -(len(ramps) - 1.0))

    def arrays(self):
        return np.stack(self.rows), np.array(self.bias, float)


def add(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def scale(d, s):
    return {k: v * s for k, v in d.items()}


# ----------------------------------------------------------------------------------------------
# 1. pixel stage on the image: 9x9 kernels, padding 3 -> 30x30 output, centre pixel = image (y+1, x+1)
#    (all taps that form a difference are inside the image, so zero padding never fakes an edge)
# ----------------------------------------------------------------------------------------------
def pixel_filters():
    F = Filters(9)
    k = F.new("L"); F.rgb(k, WL)
    for nm, w3 in [("rg", (1, -1, 0)), ("gr", (-1, 1, 0)), ("gb", (0, 1, -1)), ("bg", (0, -1, 1)),
                   ("br", (-1, 0, 1)), ("rb", (1, 0, -1))]:
        k = F.new(nm); F.rgb(k, w3)
    F.signed("gx", lambda k: (F.lum(k, 0, 1, 1), F.lum(k, 0, 0, -1)))
    F.signed("gy", lambda k: (F.lum(k, 1, 0, 1), F.lum(k, 0, 0, -1)))

    def hline(k):   # vertical change, averaged along a 9-pixel horizontal run: long horizontal edges
        for ox in range(-4, 5):
            F.lum(k, 1, ox, 1 / 9); F.lum(k, -1, ox, -1 / 9)

    def vline(k):
        for oy in range(-4, 5):
            F.lum(k, oy, 1, 1 / 9); F.lum(k, oy, -1, -1 / 9)
    F.signed("H", hline)
    F.signed("V", vline)

    centre = [(y, x) for y in range(-4, 5) for x in range(-4, 5) if y * y + x * x <= 4]
    ring = [(y, x) for y in range(-4, 5) for x in range(-4, 5) if 8 <= y * y + x * x <= 18]
    k = F.new("blob")      # ring brighter than the centre: a dark round blob of diameter ~4-5 px
    for y, x in ring: F.lum(k, y, x, 1 / len(ring))
    for y, x in centre: F.lum(k, y, x, -1 / len(centre))
    k = F.new("Lc")
    for y, x in centre: F.lum(k, y, x, 1 / len(centre))
    return F


# signed colour expressions in terms of the relu'd channels
RG = {"rg": 1, "gr": -1}; GR = scale(RG, -1)
GB = {"gb": 1, "bg": -1}; BG = scale(GB, -1)
BR = {"br": 1, "rb": -1}; RB = scale(BR, -1)
SAT = {n: 0.5 for n in ("rg", "gr", "gb", "bg", "br", "rb")}     # = max(R,G,B) - min(R,G,B)
ABSRG = {"rg": 1, "gr": 1}
GSUM = {"gx+": 1, "gx-": 1, "gy+": 1, "gy-": 1}
ABSH = {"H+": 1, "H-": 1}
ABSV = {"V+": 1, "V-": 1}
LUM = {"L": 1}


def pixel_ramps(a_names):
    S = Stage(a_names)
    S.passthrough("L", LUM)
    S.passthrough("sat", SAT)
    S.ramp("blue1", BR, 0.03, 0.09)          # B clearly above R
    S.ramp("blue2", BG, -0.06, -0.02)        # and B not below G (excludes greens / cyan-greens)
    S.ramp("notblue", BR, 0.07, 0.03)
    S.ramp("cool", BR, -0.03, 0.01)          # neutral or cool white (overcast sky), not beige
    S.ramp("skyL", LUM, 0.55, 0.65)
    S.ramp("skyL2", LUM, 0.40, 0.50)
    S.ramp("notbright", LUM, 0.50, 0.40)
    S.ramp("grn1", GR, -0.03, 0.02)          # G at least about R
    S.ramp("grn2", GB, 0.03, 0.08)           # and clearly above B: vegetation
    S.ramp("warm1", RB, 0.06, 0.12)          # R above B
    S.ramp("warm2", GB, -0.04, 0.0)          # G not below B
    S.ramp("warm3", RG, 0.32, 0.26)          # and not a pure red: browns, tans, beiges, fur, earth
    S.ramp("vr1", RG, 0.28, 0.38)            # vivid red / orange paint
    S.ramp("vr2", RB, 0.28, 0.38)
    S.ramp("vy1", RB, 0.35, 0.45)            # vivid yellow paint
    S.ramp("vy2", GB, 0.35, 0.45)
    S.ramp("vy3", ABSRG, 0.20, 0.15)
    S.ramp("neutral", SAT, 0.10, 0.06)       # greys, white, black
    S.ramp("lowsat", SAT, 0.15, 0.11)
    S.ramp("light", LUM, 0.62, 0.72)
    S.ramp("smooth", GSUM, 0.06, 0.035)      # almost no change to the neighbours
    S.ramp("busy", GSUM, 0.08, 0.14)         # strong local change (texture or edge)
    S.ramp("hline", ABSH, 0.12, 0.20)
    S.ramp("vline", ABSV, 0.12, 0.20)
    S.ramp("blob", {"blob": 1}, 0.08, 0.16)
    S.ramp("blobdark", {"Lc": 1}, 0.32, 0.22)
    return S


PIXEL_FEATURES = [  # name, ramps that must all hold
    ("blue", ["blue1", "blue2"]),
    ("green", ["grn1", "grn2"]),
    ("warm", ["warm1", "warm2", "warm3"]),
    ("vred", ["vr1", "vr2"]),
    ("vyel", ["vy1", "vy2", "vy3"]),
    ("neutral", ["neutral"]),
    ("white", ["lowsat", "light"]),
    ("smooth", ["smooth"]),
    ("busy", ["busy"]),
    ("hline", ["hline"]),
    ("vline", ["vline"]),
    ("wheel", ["blob", "blobdark"]),
    ("skyb", ["smooth", "blue1", "blue2"]),
    ("skyg", ["smooth", "cool", "skyL", "notblue"]),
    ("skybb", ["smooth", "blue1", "blue2", "skyL2"]),   # bright blue sky (water is usually darker)
    ("dblue", ["blue1", "blue2", "notbright"]),         # darker blue: water, or deep sky
]


def pixel_features(b_names):
    S = Stage(b_names)
    S.passthrough("L", {"L": 1})
    S.passthrough("sat", {"sat": 1})
    for name, ramps in PIXEL_FEATURES:
        S.all_of(name, ramps)
    return S


# ----------------------------------------------------------------------------------------------
# shape branch: 9x9 valid conv (24x24, image rows/cols 4..27)
# ----------------------------------------------------------------------------------------------
def shape_filters():
    F = Filters(9)

    def d1(k):   # "/" lines: change along the (1,1) normal, averaged along 7 pixels of the (-1,1) direction
        for t in range(-3, 4):
            F.lum(k, -t + 1, t + 1, 1 / 7); F.lum(k, -t - 1, t - 1, -1 / 7)

    def d2(k):   # "\" lines
        for t in range(-3, 4):
            F.lum(k, t + 1, t - 1, 1 / 7); F.lum(k, t - 1, t + 1, -1 / 7)

    def cgy(k):  # coarse vertical change: mean of 3x3 below minus 3x3 above
        for oy in (1, 2, 3):
            for ox in (-1, 0, 1):
                F.lum(k, oy, ox, 1 / 9); F.lum(k, -oy, ox, -1 / 9)

    def cgx(k):
        for ox in (1, 2, 3):
            for oy in (-1, 0, 1):
                F.lum(k, oy, ox, 1 / 9); F.lum(k, oy, -ox, -1 / 9)
    for nm, f in [("D1", d1), ("D2", d2), ("cgy", cgy), ("cgx", cgx)]:
        F.signed(nm, f)
    return F


def long_line_filters():
    """On the 2x2-averaged (16x16) image: the same horizontal / vertical line detectors as above, so a
    full response needs an edge that stays straight for about 18 full-size pixels (car bodies, hulls,
    truck boxes, buildings, horizons; rarely animal outlines or fur)."""
    F = Filters(9)

    def hline(k):
        for ox in range(-4, 5):
            F.lum(k, 1, ox, 1 / 9); F.lum(k, -1, ox, -1 / 9)

    def vline(k):
        for oy in range(-4, 5):
            F.lum(k, oy, 1, 1 / 9); F.lum(k, oy, -1, -1 / 9)
    F.signed("H", hline)
    F.signed("V", vline)
    return F


def long_line_ramps(names):
    S = Stage(names)
    S.ramp("hlong", ABSH, 0.12, 0.20)
    S.ramp("vlong", ABSV, 0.12, 0.20)
    return S


def long_line_features(names):
    S = Stage(names)
    S.all_of("hlong", ["hlong"])
    S.all_of("vlong", ["vlong"])
    return S


def shape_ramps(names):
    S = Stage(names)
    S.ramp("dline1", {"D1+": 1, "D1-": 1}, 0.12, 0.20)
    S.ramp("dline2", {"D2+": 1, "D2-": 1}, 0.12, 0.20)
    S.passthrough("acgy", {"cgy+": 1, "cgy-": 1})
    S.passthrough("acgx", {"cgx+": 1, "cgx-": 1})
    return S


def shape_features(names):
    S = Stage(names)
    S.all_of("dline1", ["dline1"])
    S.all_of("dline2", ["dline2"])
    S.passthrough("acgy", {"acgy": 1})
    S.passthrough("acgx", {"acgx": 1})
    return S


# ----------------------------------------------------------------------------------------------
# image-level quantities on the 5x5 grid (cell r covers image rows 1+6r .. 6+6r)
# ----------------------------------------------------------------------------------------------
G = 5
TOP = [(0, c) for c in range(G)]
BOT = [(G - 1, c) for c in range(G)]
CEN = [(r, c) for r in range(1, G - 1) for c in range(1, G - 1)]
BORDER = [(r, c) for r in range(G) for c in range(G) if (r, c) not in CEN]
ALL = [(r, c) for r in range(G) for c in range(G)]
BL = [(3, 0), (3, 1), (4, 0), (4, 1)]
BR_ = [(3, 3), (3, 4), (4, 3), (4, 4)]


def region(ch, cells, w=1.0):
    """{(branch, channel, r, c): weight} = w * mean of channel over the cells."""
    return {("main", ch, r, c): w / len(cells) for r, c in cells}


def shape(ch, w=1.0):
    return {("shape", ch, 0, 0): w}


def longline(ch, w=1.0):
    return {("long", ch, 0, 0): w}


# Quantities binned into low / mid / high by two soft cut points, each cut a ramp (lo, hi).
QUANTITIES = [
    # name,          expression,                                              cut 1,         cut 2
    ("sky_top",      add(region("skyb", TOP), region("skyg", TOP)),          (0.15, 0.35), (0.50, 0.75)),
    ("blue_top",     region("blue", TOP),                                     (0.10, 0.30), (0.45, 0.70)),
    ("dblue_bot",    region("dblue", BOT),                                    (0.08, 0.25), (0.40, 0.65)),
    ("green_bot",    region("green", BOT),                                    (0.08, 0.25), (0.35, 0.60)),
    ("green_top",    region("green", TOP),                                    (0.08, 0.25), (0.35, 0.60)),
    ("warm_cen",     region("warm", CEN),                                     (0.15, 0.35), (0.50, 0.75)),
    ("warm_all",     region("warm", ALL),                                     (0.15, 0.35), (0.50, 0.75)),
    ("vivid_cen",    add(region("vred", CEN), region("vyel", CEN)),          (0.01, 0.05), (0.10, 0.20)),
    ("neutral_all",  region("neutral", ALL),                                  (0.15, 0.35), (0.50, 0.75)),
    ("smooth_bd",    region("smooth", BORDER),                                (0.30, 0.50), (0.65, 0.85)),
    ("busy_all",     region("busy", ALL),                                     (0.12, 0.25), (0.35, 0.55)),
    ("busy_top",     region("busy", TOP),                                     (0.12, 0.25), (0.35, 0.55)),
    ("hlong",        longline("hlong"),                                       (0.04, 0.10), (0.15, 0.25)),
    ("vlong",        longline("vlong"),                                       (0.03, 0.07), (0.10, 0.18)),
    ("dlines",       add(shape("dline1"), shape("dline2")),                                          (0.04, 0.09), (0.13, 0.22)),
    ("lum",          region("L", ALL),                                        (0.25, 0.35), (0.50, 0.60)),
    ("top_minus_bot", add(region("L", TOP), region("L", BOT, -1.0)),         (-0.04, 0.02), (0.08, 0.16)),
    ("gray_bot",     region("neutral", BOT),                                  (0.15, 0.35), (0.55, 0.80)),
    ("white_cen",    region("white", CEN),                                    (0.04, 0.12), (0.22, 0.40)),
    ("smooth_bot",   region("smooth", BOT),                                   (0.30, 0.50), (0.65, 0.90)),
    ("sky_bot",      add(region("skybb", BOT), region("skyg", BOT)),         (0.15, 0.35), (0.50, 0.75)),
    ("layered",      add(shape("acgy"), shape("acgx", -1.25)),               (-0.015, 0.0), (0.012, 0.03)),
]

# Binary conjunctions: AND of two soft conditions, each condition = ramp of a quantity (lo, hi).
CONJUNCTIONS = [
    ("water_below_sky", (region("dblue", BOT), 0.20, 0.45), (add(region("L", TOP), region("L", BOT, -1.0)), 0.02, 0.08)),
    ("green_bokeh",     (region("smooth", BORDER), 0.50, 0.75), (region("green", BORDER), 0.25, 0.50)),
    ("plain_bg_warm",   (region("smooth", BORDER), 0.50, 0.75), (region("warm", CEN), 0.30, 0.55)),
    ("wheels",          (region("wheel", BL), 0.02, 0.06), (region("wheel", BR_), 0.02, 0.06)),
    ("animal_on_grass", (region("green", BOT), 0.25, 0.50), (region("warm", CEN), 0.25, 0.50)),
    ("brown_under_sky", (add(region("skyb", TOP), region("skyg", TOP)), 0.30, 0.60), (region("warm", CEN), 0.15, 0.35)),
]

# How strongly each class shows each quantity, typed from general knowledge of such photos.
# Letters: L very low, l low, m medium, h high, H very high.  Order of classes:
#                 plane car bird cat deer dog frog horse ship truck
LEVELS = {
    "sky_top":       "H m m l l l L m H m",
    "blue_top":      "h l l L L L L l h l",
    "dblue_bot":     "l L l L L L L L H L",
    "green_bot":     "l l m L m m m m L l",
    "green_top":     "L l m L m l m m l l",
    "warm_cen":      "L l m h h h h h L l",
    "warm_all":      "L l m m h m H m L l",
    "vivid_cen":     "l h l L L L l L m h",
    "neutral_all":   "h m m m l m l l m m",
    "smooth_bd":     "H m h m l m l m m m",
    "busy_all":      "L m m m h m H m l m",
    "busy_top":      "L m m m h m h m l m",
    "hlong":         "m h l l l l L m h h",
    "vlong":         "l m l m l l L l m h",
    "dlines":        "m m l m l m l m m m",
    "lum":           "h m m m l m l m m m",
    "top_minus_bot": "m m m m m m m m h m",
    "gray_bot":      "h h l m L m l l m h",
    "white_cen":     "h m l m L m L l h h",
    "smooth_bot":    "H l m m L m l l m l",
    "sky_bot":       "h l m l L l L L l L",
    "layered":       "m m l l m l l m h m",
    # binary conjunctions
    "water_below_sky": "l L l L L L L L H L",
    "green_bokeh":     "L L h L l L l l L L",
    "plain_bg_warm":   "L l h m l m l l L L",
    "wheels":          "l h l l m l l m l m",
    "animal_on_grass": "L L m L H m m H L L",
    "brown_under_sky": "l l H L l L L m l l",
}
# soft bins (low, mid, high) for each letter; kept mild because the cut points are educated guesses
THREE_BIN = {"L": (0.70, 0.20, 0.10), "l": (0.50, 0.33, 0.17), "m": (0.30, 0.40, 0.30),
             "h": (0.17, 0.33, 0.50), "H": (0.10, 0.20, 0.70)}
TWO_BIN = {"L": 0.05, "l": 0.10, "m": 0.18, "h": 0.32, "H": 0.50}


def build():
    layers, W = [], {}

    def conv(name, inp, Wt, b, k, pad):
        layers.append({"name": name, "op": "conv", "inputs": [inp], "out": int(Wt.shape[0]),
                       "kernel": [k, k], "stride": 1, "padding": pad})
        W[name + ".W"], W[name + ".b"] = Wt, b

    def relu(name, inp):
        layers.append({"name": name, "op": "relu", "inputs": [inp]})

    def dense(name, inp, Wt, b):
        layers.append({"name": name, "op": "dense", "inputs": [inp], "out": int(Wt.shape[0])})
        W[name + ".W"], W[name + ".b"] = Wt, b

    # main pixel branch
    FA = pixel_filters(); ka, ba = FA.arrays()
    conv("px", "image", ka, ba, 9, 3); relu("px_r", "px")
    SB = pixel_ramps(FA.names); wb, bb = SB.arrays()
    conv("pxr", "px_r", wb[:, :, None, None], bb, 1, 0); relu("pxr_r", "pxr")
    SC = pixel_features(SB.names); wc, bc = SC.arrays()
    conv("pxf", "pxr_r", wc[:, :, None, None], bc, 1, 0); relu("pxf_r", "pxf")
    layers.append({"name": "grid", "op": "avgpool", "inputs": ["pxf_r"], "size": [6, 6], "stride": [6, 6]})
    main_names = SC.names

    # shape branch
    FS = shape_filters(); ks, bs = FS.arrays()
    conv("sh", "image", ks, bs, 9, 0); relu("sh_r", "sh")
    SR = shape_ramps(FS.names); wr, br = SR.arrays()
    conv("shr", "sh_r", wr[:, :, None, None], br, 1, 0); relu("shr_r", "shr")
    SF = shape_features(SR.names); wf, bf = SF.arrays()
    conv("shf", "shr_r", wf[:, :, None, None], bf, 1, 0); relu("shf_r", "shf")
    layers.append({"name": "shpool", "op": "avgpool", "inputs": ["shf_r"], "size": [24, 24], "stride": [24, 24]})
    shape_names = SF.names

    # long straight lines at half resolution
    layers.append({"name": "half", "op": "avgpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]})
    FL = long_line_filters(); kl, bl = FL.arrays()
    conv("ll", "half", kl, bl, 9, 3); relu("ll_r", "ll")
    LR = long_line_ramps(FL.names); wl, bl2 = LR.arrays()
    conv("llr", "ll_r", wl[:, :, None, None], bl2, 1, 0); relu("llr_r", "llr")
    LF = long_line_features(LR.names); wlf, blf = LF.arrays()
    conv("llf", "llr_r", wlf[:, :, None, None], blf, 1, 0); relu("llf_r", "llf")
    layers.append({"name": "llpool", "op": "avgpool", "inputs": ["llf_r"], "size": [14, 14], "stride": [14, 14]})
    long_names = LF.names

    # image-level ramps: each is a pair of units over the two pooled branches
    branches = {"main": ("grid", main_names, G), "shape": ("shpool", shape_names, 1), "long": ("llpool", long_names, 1)}

    def flat_index(key):
        br_, ch, r, c = key
        _, names, g = branches[br_]
        return br_, names.index(ch) * g * g + r * g + c

    ramps = []   # (name, expr, t0, t1)
    for name, expr, c1, c2 in QUANTITIES:
        ramps.append((name + "#1", expr, *c1)); ramps.append((name + "#2", expr, *c2))
    for name, (e1, a0, a1), (e2, b0, b1) in CONJUNCTIONS:
        ramps.append((name + "#a", e1, a0, a1)); ramps.append((name + "#b", e2, b0, b1))
    nr = len(ramps)
    Wq = {b: np.zeros((2 * nr, len(n) * g * g)) for b, (_, n, g) in branches.items()}
    bq = np.zeros(2 * nr)
    for i, (name, expr, t0, t1) in enumerate(ramps):
        s = 1.0 / (t1 - t0)
        for key, v in expr.items():
            br_, j = flat_index(key)
            Wq[br_][2 * i, j] += v * s
            Wq[br_][2 * i + 1, j] += v * s
        bq[2 * i], bq[2 * i + 1] = -t0 * s, -t0 * s - 1.0
    for b, (pooled, _, _) in branches.items():
        dense("q_" + b, pooled, Wq[b], bq if b == "main" else np.zeros(2 * nr))
    layers.append({"name": "q", "op": "add", "inputs": ["q_" + b for b in branches]})
    relu("q_r", "q")

    # soft indicators: one per quantity cut, one per conjunction
    rid = {r[0]: i for i, r in enumerate(ramps)}
    phi_names, rows, bias = [], [], []

    def phi(name, parts, b):
        w = np.zeros(2 * nr)
        for rn in parts:
            w[2 * rid[rn]] += 1; w[2 * rid[rn] + 1] -= 1
        phi_names.append(name); rows.append(w); bias.append(b)
    for name, *_ in QUANTITIES:
        phi(name + "#1", [name + "#1"], 0.0); phi(name + "#2", [name + "#2"], 0.0)
    for name, *_ in CONJUNCTIONS:
        phi(name, [name + "#a", name + "#b"], -1.0)
    dense("phi", "q_r", np.stack(rows), np.array(bias)); relu("phi_r", "phi")

    # naive-Bayes readout from the hand-typed levels
    K = len(CLASSES); Wo = np.zeros((K, len(phi_names))); bo = np.zeros(K)
    pid = {n: i for i, n in enumerate(phi_names)}
    for name, *_ in QUANTITIES:
        lv = LEVELS[name].split()
        for k in range(K):
            pl, pm, ph = np.log(THREE_BIN[lv[k]])
            bo[k] += pl
            Wo[k, pid[name + "#1"]] += pm - pl
            Wo[k, pid[name + "#2"]] += ph - pm
    for name, *_ in CONJUNCTIONS:
        lv = LEVELS[name].split()
        for k in range(K):
            p = TWO_BIN[lv[k]]
            bo[k] += np.log(1 - p)
            Wo[k, pid[name]] += np.log(p) - np.log(1 - p)
    dense("out", "phi_r", Wo, bo)
    return {"layers": layers}, W, phi_names


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    net, W, _ = build()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(net, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **{k: np.asarray(v, np.float32) for k, v in W.items()})


if __name__ == "__main__":
    main()
