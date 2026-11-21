"""Hand-designed network for 32x32 colour photo classification (10 classes).

Every number here comes from a formula (Gaussian derivatives, Gabors, rings,
sectors) or from general knowledge about what these objects and scenes look
like.  No training data is read, no fitting of any kind is performed.

Structure
  colour branch   : hinge features of (R,G,B) -> 18 soft scene/material indicators
  texture branch  : |Gaussian-derivative / Gabor| energies at two scales
  shape branch    : |ring / bar / blob / line| detectors built from geometry
  trunk           : the branches are concatenated (expand-to-common-width + add)
  pooling         : 4x4 grid means, global mean, global max, 2x2 grid max
  readout         : named region aggregates -> hand-written linear evidence
                    plus a hand-written threshold-rule engine (ReLU logic)
"""
import sys, json
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
layers, weights = [], {}


def layer(name, op, inputs, **kw):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(kw)
    layers.append(d)
    return name


def conv(name, inp, w, b, stride=1, padding=0):
    o, i, kh, kw = w.shape
    layer(name, "conv", [inp], out=o, kernel=[kh, kw], stride=stride, padding=padding)
    weights[name + ".W"] = np.asarray(w, np.float64)
    weights[name + ".b"] = np.asarray(b, np.float64)
    return name


def dense(name, inp, w, b):
    layer(name, "dense", [inp], out=w.shape[0])
    weights[name + ".W"] = np.asarray(w, np.float64)
    weights[name + ".b"] = np.asarray(b, np.float64)
    return name


def relu(name, inp):
    return layer(name, "relu", [inp])


def pool(name, inp, kind, size, stride):
    return layer(name, kind, [inp], size=list(size), stride=list(stride))


def add(name, *ins):
    return layer(name, "add", list(ins))


def reshape(name, inp, shape):
    return layer(name, "reshape", [inp], shape=list(shape))


def eye_expand(name, inp, nin, nout, offset):
    """1x1 conv placing nin channels into slots offset..offset+nin-1 of nout."""
    w = np.zeros((nout, nin, 1, 1))
    for i in range(nin):
        w[offset + i, i, 0, 0] = 1.0
    return conv(name, inp, w, np.zeros(nout))


# ---------------------------------------------------------------- colour branch
# luminance weights (standard Rec.601)
LW = np.array([0.299, 0.587, 0.114])

# c1: signed photometric quantities and their negations, so that after ReLU
# every hinge |q| = relu(q) + relu(-q) is available to the next 1x1 conv.
c1spec = [
    ([1, 0, 0], 0.0),    # 0 R
    ([0, 1, 0], 0.0),    # 1 G
    ([0, 0, 1], 0.0),    # 2 B
    (LW, 0.0),           # 3 L
    (LW, -0.45),         # 4 L-0.45
    (-LW, 0.45),         # 5 0.45-L
    ([-1, 0, 1], 0.0),   # 6 B-R
    ([1, 0, -1], 0.0),   # 7 R-B
    ([-1, 1, 0], 0.0),   # 8 G-R
    ([1, -1, 0], 0.0),   # 9 R-G
    ([0, 1, -1], 0.0),   # 10 G-B
    ([0, -1, 1], 0.0),   # 11 B-G
    ([-0.5, -0.5, 1], 0.0),   # 12 B-(R+G)/2
    ([0.5, 0.5, -1], 0.0),    # 13 (R+G)/2-B
    ([1, -2, 1], 0.0),   # 14 R+B-2G
    ([-1, 2, -1], 0.0),  # 15 2G-R-B
    ([1, 1, -2], 0.0),   # 16 R+G-2B
    ([-1, -1, 2], 0.0),  # 17 2B-R-G
    ([2, -1, -1], 0.0),  # 18 2R-G-B
    ([-2, 1, 1], 0.0),   # 19 G+B-2R
    (LW, -0.25),         # 20 L-0.25
    (-LW, 0.25),         # 21 0.25-L
    (LW, -0.70),         # 22 L-0.70
    (-LW, 0.70),         # 23 0.70-L
    # gates: min(a, b) = a - relu(a-b), so these hinges let a colour difference
    # be conjunctively limited by a luminance range (blue AND not-dark, etc.)
    (LW + np.array([-1, 0, 1]), -0.60),   # 24 (B-R)+L-0.60
    (-(LW + np.array([-1, 0, 1])), 0.60), # 25
    (np.array([-1, 0, 1]) - LW, 0.45),    # 26 (B-R)-L+0.45
    (-(np.array([-1, 0, 1]) - LW), -0.45),# 27
    (np.array([1, -1, 0]) - LW, 0.55),    # 28 (R-G)-L+0.55
    (-(np.array([1, -1, 0]) - LW), -0.55),# 29
    (np.array([-1, 1, 0]) + LW, -0.35),   # 30 (G-R)+L-0.35
    (-(np.array([-1, 1, 0]) + LW), 0.35), # 31
    (np.array([0, 1, -1]) + LW, -0.35),   # 32 (G-B)+L-0.35
    (-(np.array([0, 1, -1]) + LW), 0.35), # 33
]
NC1 = len(c1spec)
w = np.zeros((NC1, 3, 1, 1)); b = np.zeros(NC1)
for i, (coef, bias) in enumerate(c1spec):
    w[i, :, 0, 0] = coef; b[i] = bias
conv("c1", "image", w, b)
relu("r1", "c1")

# named vectors over the 24 relu channels
def q(**kw):
    v = np.zeros(NC1)
    for k, c in kw.items():
        v[int(k[1:])] = c
    return v

R_ = q(_0=1); G_ = q(_1=1); B_ = q(_2=1); L_ = q(_3=1)
U = q(_6=1, _7=-1)          # B-R
V = q(_8=1, _9=-1)          # G-R
T = q(_10=1, _11=-1)        # G-B
CH = q(_6=1, _7=1, _8=1, _9=1, _10=1, _11=1)   # 2*(max-min)
L45 = q(_4=1, _5=-1)        # L-0.45
L25 = q(_20=1, _21=-1)      # L-0.25
L70 = q(_22=1, _23=-1)      # L-0.70
AL45 = q(_4=1, _5=1)        # |L-0.45|
ARG = q(_8=1, _9=1)         # |R-G|
# minima of two colour differences: min(a,b) = a - relu(a-b)
M_G = q(_8=1, _9=-1, _6=-1)          # min(G-R, G-B)   greenness
M_BR = q(_9=1, _8=-1, _14=-1)        # min(R-G, G-B)   brownness (R>G>B)
M_R = q(_9=1, _8=-1, _11=-1)         # min(R-G, R-B)   redness
M_Y = q(_7=1, _6=-1, _9=-1)          # min(R-B, G-B)   yellowness (R,G > B)
M_B = q(_6=1, _7=-1, _8=-1)          # min(B-R, B-G)   blueness
# conjunctive gates (colour difference limited by a luminance range)
WAT_M = q(_6=1, _7=-1, _24=-1)       # min(B-R, 0.60-L)  blue and not bright
SKY_M = q(_6=1, _7=-1, _26=-1)       # min(B-R, L-0.45)  blue and not dark
TAN_M = q(_9=1, _8=-1, _28=-1)       # min(R-G, L-0.55)  warm and bright
GD1 = q(_8=1, _9=-1, _30=-1)         # min(G-R, 0.35-L)  green and dark
GD2 = q(_10=1, _11=-1, _32=-1)       # min(G-B, 0.35-L)

# 18 soft indicators, each scaled so a canonical example reaches about 1.
IND = [
    ("sky_blue",   6*SKY_M),
    ("sky_white",  4*L70 - 1.2*CH),
    ("water_blue", 5*WAT_M),
    ("water_gray", -4*CH + 5*U - 4*AL45),
    ("green_veg",  6*M_G + 0.7*T),
    ("green_dark", 6*GD1 + 3*GD2),
    ("brown",      6*M_BR + 0.7*q(_9=1, _8=-1)),
    ("tan",        6*M_BR + 4*TAN_M),
    ("gray_mid",   -2.5*CH - 3*AL45),
    ("dark",       -4*L25),
    ("bright",     3*L70),
    ("red",        4*M_R),
    ("yellow",     3*M_Y - 3*ARG),
    ("saturated",  2*CH),
    ("achrom",     -2.5*CH),
    ("blue_dom",   4*M_B),
    ("yellowgreen", 4*M_Y - 2*ARG),
    ("white_obj",  4*L70 - 2*CH),
]
K = len(IND)
w = np.zeros((2*K, NC1, 1, 1)); b = np.zeros(2*K)
for i, (_, vec) in enumerate(IND):
    w[i, :, 0, 0] = vec
    w[K+i, :, 0, 0] = vec
    b[i] = 0.0; b[K+i] = -1.0
# every named vector is an exact combination of relu channels (the photometric
# offsets live inside the hinges), so the scalars written in IND are the biases.
OFF = [-0.15, 0.17, -0.10, 0.45, -0.05, -0.10, -0.05, -0.10, 0.70, -0.12,
       0.0, -0.15, -0.20, -0.50, 0.50, -0.10, -0.15, -0.40]
for i, o in enumerate(OFF):
    b[i] += o; b[K+i] += o
conv("c2", "r1", w, b)
relu("r2", "c2")
w = np.zeros((K, 2*K, 1, 1))
for i in range(K):
    w[i, i, 0, 0] = 1.0; w[i, K+i, 0, 0] = -1.0
conv("ccap", "r2", w, np.zeros(K))          # indicators clipped to [0,1]

# raw photometric maps (kept alongside the indicators)
w = np.zeros((4, NC1, 1, 1))
for i in range(4):
    w[i, i, 0, 0] = 1.0
conv("praw", "r1", w, np.zeros(4))
# --------------------------------------------------------------- texture branch
def grid(k):
    a = np.arange(k) - (k - 1) / 2.0
    return np.meshgrid(a, a, indexing="xy")     # X, Y


def norm_pos(f):
    """zero-mean, then scale so a unit-contrast aligned pattern gives ~1."""
    f = f - f.mean()
    s = f[f > 0].sum()
    return f / s if s > 1e-9 else f


X5, Y5 = grid(5)
R5 = np.hypot(X5, Y5)


def dgauss(sig, axis):
    g = np.exp(-(X5**2 + Y5**2) / (2 * sig**2))
    return -(X5 if axis == "x" else Y5) / sig**2 * g


def log(sig):
    return -(R5**2 - 2 * sig**2) / sig**4 * np.exp(-R5**2 / (2 * sig**2))


def gabor(sig, lam, th):
    xr = X5 * np.cos(th) + Y5 * np.sin(th)
    return np.exp(-(X5**2 + Y5**2) / (2 * sig**2)) * np.cos(2 * np.pi * xr / lam)


TEX = [dgauss(0.7, "x"), dgauss(0.7, "y"), dgauss(1.3, "x"), dgauss(1.3, "y"),
       log(0.9), log(1.6)]
for th in (0, np.pi/4, np.pi/2, 3*np.pi/4):
    TEX.append(gabor(1.0, 3.0, th))
for th in (0, np.pi/4, np.pi/2, 3*np.pi/4):
    TEX.append(gabor(1.8, 6.0, th))
TEX += [dgauss(2.0, "x"), dgauss(2.0, "y")]
TEX = [2.5 * norm_pos(f) for f in TEX]
GT = len(TEX)
TEXN = ["tex_dx_fine", "tex_dy_fine", "tex_dx_mid", "tex_dy_mid",
        "tex_blob_fine", "tex_blob_mid",
        "tex_g0_fine", "tex_g45_fine", "tex_g90_fine", "tex_g135_fine",
        "tex_g0_coarse", "tex_g45_coarse", "tex_g90_coarse", "tex_g135_coarse",
        "tex_dx_coarse", "tex_dy_coarse"]
w = np.zeros((2*GT, 3, 5, 5)); b = np.zeros(2*GT)
for i, f in enumerate(TEX):
    for c in range(3):
        w[i, c] = LW[c] * f
        w[GT+i, c] = -LW[c] * f
conv("t1", "image", w, b, stride=1, padding=2)
relu("tr", "t1")
w = np.zeros((GT, 2*GT, 1, 1))
for i in range(GT):
    w[i, i, 0, 0] = 1.0; w[i, GT+i, 0, 0] = 1.0
conv("tabs", "tr", w, np.zeros(GT))         # oriented energy |response|

# ----------------------------------------------------------------- shape branch
X7, Y7 = grid(7)
R7 = np.hypot(X7, Y7)


def ring(r0, w, hub=0.0, rhub=1.1, outer=0.0):
    f = -np.exp(-((R7 - r0) / w)**2)
    if hub:
        f = f + hub * np.exp(-(R7 / rhub)**2)
    if outer:
        f = f + outer * (R7 >= r0 + 1.6)
    return f


def blob(sig, surround=0.35, rs=3.2, ws=1.2):
    return -np.exp(-(R7 / sig)**2) + surround * np.exp(-((R7 - rs) / ws)**2)


def step(axis, sign):
    """bright on one side, dark on the other, averaged along the edge."""
    c = Y7 if axis == "h" else X7
    f = np.clip(-c, -1, 1) * sign
    taper = np.exp(-(X7 if axis == "h" else Y7)**2 / (2 * 2.6**2))
    return f * taper


def line_dark(axis, th=0.0, width=0.9, flank=2.2, fw=1.0, along=2.6):
    xr = X7 * np.cos(th) + Y7 * np.sin(th)
    yr = -X7 * np.sin(th) + Y7 * np.cos(th)
    core = -np.exp(-(xr / width)**2)
    fl = 0.5 * (np.exp(-((xr - flank) / fw)**2) + np.exp(-((xr + flank) / fw)**2))
    return (core + fl) * np.exp(-(yr / along)**2)


def bar_dark(axis, width=1.5, flank=3.0, fw=1.0, along=2.6):
    c, o = (X7, Y7) if axis == "v" else (Y7, X7)
    core = -np.exp(-(c / width)**2)
    fl = 0.5 * (np.exp(-((c - flank) / fw)**2) + np.exp(-((c + flank) / fw)**2))
    return (core + fl) * np.exp(-(o / along)**2)


def corner():
    f = np.zeros((7, 7))
    f[(X7 <= -1) & (Y7 <= -1)] = 1.0
    f[(X7 >= 1) | (Y7 >= 1)] = -0.35
    return f


def dog7(sig):
    return -(R7**2 - 2 * sig**2) / sig**4 * np.exp(-R7**2 / (2 * sig**2))


SHP = [
    ("wheel",      ring(2.4, 0.9, hub=0.7, outer=0.25)),
    ("darkblob",   blob(1.6)),
    ("brightblob", -blob(1.6)),
    ("hstep",      step("h", 1)),      # bright above, dark below
    ("vstep",      step("v", 1)),      # bright left, dark right
    ("vline",      line_dark("v", 0.0)),
    ("hline",      line_dark("h", np.pi/2)),
    ("dline45",    line_dark("d", np.pi/4)),
    ("dline135",   line_dark("d", 3*np.pi/4)),
    ("bigblob",    dog7(2.2)),
    ("brightring", -ring(2.4, 0.9, hub=0.7, outer=0.25)),
    ("vbar",       bar_dark("v")),
    ("hbar",       bar_dark("h")),
    ("corner",     corner()),
]
SHP = [(n, norm_pos(f)) for n, f in SHP]
NS = len(SHP)
SHPN = [n for n, _ in SHP]
w = np.zeros((2*NS, 3, 7, 7)); b = np.zeros(2*NS)
for i, (_, f) in enumerate(SHP):
    for c in range(3):
        w[i, c] = LW[c] * f
        w[NS+i, c] = -LW[c] * f
conv("s1", "image", w, b, stride=1, padding=3)
relu("sr", "s1")     # signed half-wave channels: f and -f stay separate

# ---------------------------------------------------------------------- trunk
CH_IND = [n for n, _ in IND]
CH_RAW = ["raw_R", "raw_G", "raw_B", "raw_L"]
CH_ALL = CH_IND + TEXN + [n + "_p" for n in SHPN] + [n + "_n" for n in SHPN] + CH_RAW
NC = len(CH_ALL)
oI = 0
oT = oI + K
oS = oT + GT
oR = oS + 2*NS
assert oR + 4 == NC
eye_expand("eI", "ccap", K, NC, oI)
eye_expand("eT", "tabs", GT, NC, oT)
eye_expand("eS", "sr", 2*NS, NC, oS)
eye_expand("eR", "praw", 4, NC, oR)
add("trunk", "eI", "eT", "eS", "eR")

# ---------------------------------------------------------------------- pooling
pool("pa", "trunk", "avgpool", [8, 8], [8, 8])      # (NC,4,4) region means
pool("pm", "trunk", "avgpool", [32, 32], [32, 32])  # (NC,1,1) global mean
pool("px", "trunk", "maxpool", [32, 32], [32, 32])  # (NC,1,1) global max
pool("pq", "trunk", "maxpool", [16, 16], [16, 16])  # (NC,2,2) quadrant max
reshape("fa", "pa", [NC*16])
reshape("fm", "pm", [NC])
reshape("fx", "px", [NC])
reshape("fq", "pq", [NC*4])
# ------------------------------------------------------- region aggregate layer
# A 4x4 grid of 8px cells describes the layout of every map.  Regions are
# hand-defined averages over that grid; 'all' and 'max' use the global pools.
rr, cc = np.mgrid[0:4, 0:4]


def mk(mask):
    ij = np.argwhere(mask)
    return [(int(a), int(b), 1.0 / len(ij)) for a, b in ij]


REGION = {
    "top": mk(rr == 0), "top2": mk(rr <= 1), "bot": mk(rr == 3), "bot2": mk(rr >= 2),
    "mid": mk((rr == 1) | (rr == 2)), "center": mk((rr >= 1) & (rr <= 2) & (cc >= 1) & (cc <= 2)),
    "ring": mk(~((rr >= 1) & (rr <= 2) & (cc >= 1) & (cc <= 2))),
    "left": mk(cc <= 1), "right": mk(cc >= 2),
    "tl": mk((rr <= 1) & (cc <= 1)), "tr": mk((rr <= 1) & (cc >= 2)),
    "bl": mk((rr >= 2) & (cc <= 1)), "br": mk((rr >= 2) & (cc >= 2)),
    "topmid": mk((rr == 0) & (cc >= 1) & (cc <= 2)),
    "botmid": mk((rr == 3) & (cc >= 1) & (cc <= 2)),
    "ctop": mk((rr == 1) & (cc >= 1) & (cc <= 2)),
    "cbot": mk((rr == 2) & (cc >= 1) & (cc <= 2)),
    "grid": mk(np.ones((4, 4), bool)),
}
for r in range(4):
    REGION["row%d" % r] = mk(rr == r)
for c in range(4):
    REGION["col%d" % c] = mk(cc == c)
for r in range(4):
    for c in range(4):
        REGION["c%d%d" % (r, c)] = [(r, c, 1.0)]
QMAX = {"tl": 0, "tr": 1, "bl": 2, "br": 3}

AGG = []          # (name, [(channel, region, coef)], const)


def ag(name, *terms, const=0.0):
    AGG.append((name, list(terms), const))


def t(ch, reg, k=1.0):
    return (ch, reg, k)


def sky(reg, k=1.0):
    return [t("sky_blue", reg, k), t("sky_white", reg, k)]


def water(reg, k=1.0):
    return [t("water_blue", reg, k), t("water_gray", reg, 0.4 * k)]


def fine(reg, k=1.0):
    return [t(n, reg, k / 3.0) for n in ("tex_dx_fine", "tex_dy_fine", "tex_blob_fine")]


def midtex(reg, k=1.0):
    return [t(n, reg, k / 3.0) for n in ("tex_dx_mid", "tex_dy_mid", "tex_blob_mid")]


def coarse(reg, k=1.0):
    return [t(n, reg, k / 6.0) for n in ("tex_g0_coarse", "tex_g45_coarse", "tex_g90_coarse",
                                         "tex_g135_coarse", "tex_dx_coarse", "tex_dy_coarse")]


def dxall(reg, k=1.0):
    return [t(n, reg, k / 3.0) for n in ("tex_dx_fine", "tex_dx_mid", "tex_dx_coarse")]


def dyall(reg, k=1.0):
    return [t(n, reg, k / 3.0) for n in ("tex_dy_fine", "tex_dy_mid", "tex_dy_coarse")]


# ---- scene layout
ag("sky_top", *sky("top")); ag("sky_top2", *sky("top2")); ag("sky_all", *sky("all"))
ag("sky_ring", *sky("ring")); ag("sky_center", *sky("center"))
ag("skyblue_all", t("sky_blue", "all")); ag("skywhite_all", t("sky_white", "all"))
ag("skyblue_top", t("sky_blue", "top")); ag("skywhite_top", t("sky_white", "top"))
ag("skyblue_ring", t("sky_blue", "ring")); ag("skyblue_ring_center", t("sky_blue", "ring"), t("sky_blue", "center", -1))
ag("skywhite_ring", t("sky_white", "ring"))
for r in range(4):
    ag("sky_row%d" % r, *sky("row%d" % r))
    ag("water_row%d" % r, *water("row%d" % r))
    ag("green_row%d" % r, t("green_veg", "row%d" % r))
ag("water_bot", *water("bot")); ag("water_bot2", *water("bot2")); ag("water_all", *water("all"))
ag("water_ring", *water("ring")); ag("water_center", *water("center"))
ag("waterblue_all", t("water_blue", "all")); ag("waterblue_bot", t("water_blue", "bot"))
ag("waterb_all", t("water_blue", "all"), t("water_gray", "all", 0.5))
ag("waterb_bot", t("water_blue", "bot"), t("water_gray", "bot", 0.5))
ag("waterb_bot2", t("water_blue", "bot2"), t("water_gray", "bot2", 0.5))
ag("waterb_top", t("water_blue", "top"), t("water_gray", "top", 0.5))
ag("water_layout", t("water_blue", "bot2"), t("water_gray", "bot2", 0.5),
   t("water_blue", "top", -1), t("water_gray", "top", -0.5))
ag("sky_bot", *sky("bot")); ag("sky_bot2", *sky("bot2"))
ag("watergray_all", t("water_gray", "all"))
ag("green_all", t("green_veg", "all")); ag("green_top", t("green_veg", "top"))
ag("green_bot", t("green_veg", "bot")); ag("green_bot2", t("green_veg", "bot2"))
ag("green_ring", t("green_veg", "ring")); ag("green_center", t("green_veg", "center"))
ag("gdark_all", t("green_dark", "all")); ag("gdark_ring", t("green_dark", "ring"))
ag("gdark_top", t("green_dark", "top")); ag("gdark_bot", t("green_dark", "bot"))
ag("gray_all", t("gray_mid", "all")); ag("gray_ring", t("gray_mid", "ring"))
ag("gray_bot", t("gray_mid", "bot")); ag("gray_top", t("gray_mid", "top"))
ag("gray_center", t("gray_mid", "center"))
ag("brown_all", t("brown", "all")); ag("brown_center", t("brown", "center"))
ag("brown_ring", t("brown", "ring")); ag("brown_bot", t("brown", "bot"))
ag("tan_all", t("tan", "all")); ag("tan_center", t("tan", "center")); ag("tan_ring", t("tan", "ring"))
ag("dark_all", t("dark", "all")); ag("dark_center", t("dark", "center")); ag("dark_ring", t("dark", "ring"))
ag("bright_all", t("bright", "all")); ag("bright_ring", t("bright", "ring"))
ag("achrom_all", t("achrom", "all")); ag("achrom_center", t("achrom", "center"))
ag("sat_all", t("saturated", "all")); ag("sat_center", t("saturated", "center")); ag("sat_ring", t("saturated", "ring"))
ag("red_all", t("red", "all")); ag("red_center", t("red", "center"))
ag("yellow_all", t("yellow", "all")); ag("yellow_center", t("yellow", "center"))
ag("yg_all", t("yellowgreen", "all")); ag("yg_center", t("yellowgreen", "center")); ag("yg_ring", t("yellowgreen", "ring"))
ag("blue_all", t("blue_dom", "all")); ag("blue_center", t("blue_dom", "center"))
ag("white_all", t("white_obj", "all")); ag("white_center", t("white_obj", "center")); ag("white_ring", t("white_obj", "ring"))
ag("L_all", t("raw_L", "all")); ag("L_top", t("raw_L", "top")); ag("L_bot", t("raw_L", "bot"))
ag("L_center", t("raw_L", "center")); ag("L_ring", t("raw_L", "ring"))
ag("R_all", t("raw_R", "all")); ag("G_all", t("raw_G", "all")); ag("B_all", t("raw_B", "all"))
ag("R_center", t("raw_R", "center")); ag("G_center", t("raw_G", "center")); ag("B_center", t("raw_B", "center"))
ag("sky_minus_water", *sky("all"), *water("all", -1.0))
ag("green_minus_brown", t("green_veg", "all"), t("brown", "all", -1), t("tan", "all", -0.5))
ag("L_center_ring", t("raw_L", "center"), t("raw_L", "ring", -1))
ag("sat_center_ring", t("saturated", "center"), t("saturated", "ring", -1))
ag("green_center_ring", t("green_veg", "center"), t("green_veg", "ring", -1))
ag("brown_center_ring", t("brown", "center"), t("brown", "ring", -1), t("tan", "center", 0.5), t("tan", "ring", -0.5))
ag("sky_center_ring", *sky("center"), *sky("ring", -1.0))
ag("L_top_bot", t("raw_L", "top"), t("raw_L", "bot", -1))
ag("blue_top_bot", t("blue_dom", "top"), t("blue_dom", "bot", -1))
ag("green_top_bot", t("green_veg", "top"), t("green_veg", "bot", -1))
ag("green_bot_top", t("green_veg", "bot"), t("green_veg", "top", -1))
ag("dark_center_ring", t("dark", "center"), t("dark", "ring", -1))
ag("white_center_ring", t("white_obj", "center"), t("white_obj", "ring", -1))
ag("achrom_center_ring", t("achrom", "center"), t("achrom", "ring", -1))
# ---- texture
ag("texfine_all", *fine("all")); ag("texfine_center", *fine("center")); ag("texfine_ring", *fine("ring"))
ag("texfine_top", *fine("top")); ag("texfine_bot", *fine("bot"))
ag("texmid_all", *midtex("all")); ag("texmid_center", *midtex("center")); ag("texmid_ring", *midtex("ring"))
ag("texcoarse_all", *coarse("all")); ag("texcoarse_center", *coarse("center")); ag("texcoarse_ring", *coarse("ring"))
ag("tex_aniso_all", *dxall("all"), *dyall("all", -1.0))
ag("tex_aniso_center", *dxall("center"), *dyall("center", -1.0))
ag("tex_aniso_ring", *dxall("ring"), *dyall("ring", -1.0))
ag("tex_hstruct", t("tex_g90_fine", "all"), t("tex_g90_coarse", "all"))
ag("tex_vstruct", t("tex_g0_fine", "all"), t("tex_g0_coarse", "all"))
ag("tex_hstruct_ring", t("tex_g90_fine", "ring"), t("tex_g90_coarse", "ring"))
ag("tex_vstruct_ring", t("tex_g0_fine", "ring"), t("tex_g0_coarse", "ring"))
ag("tex_fine_coarse", *fine("all"), *coarse("all", -1.0))
ag("tex_center_ring", *fine("center"), *fine("ring", -1.0))
ag("tex_mid_center_ring", *midtex("center"), *midtex("ring", -1.0))
ag("tex_smooth_all", *fine("all", -1.0), *midtex("all", -1.0), const=0.25)
ag("tex_smooth_ring", *fine("ring", -1.0), *midtex("ring", -1.0), const=0.25)
ag("tex_smooth_top", *fine("top", -1.0), *midtex("top", -1.0), const=0.25)
ag("tex_blob_all", t("tex_blob_fine", "all"), t("tex_blob_mid", "all"))
ag("tex_blob_center", t("tex_blob_fine", "center"), t("tex_blob_mid", "center"))
# ---- shape
ag("wheel_max", t("wheel_p", "max")); ag("wheel_bot2", t("wheel_p", "bot2"))
ag("wheel_bot", t("wheel_p", "bot")); ag("wheel_center", t("wheel_p", "center"))
ag("wheel_bl", t("wheel_p", "qmaxbl")); ag("wheel_br", t("wheel_p", "qmaxbr"))
ag("wheel_tl", t("wheel_p", "qmaxtl")); ag("wheel_tr", t("wheel_p", "qmaxtr"))
ag("wheel_all", t("wheel_p", "all"))
ag("darkblob_max", t("darkblob_p", "max")); ag("darkblob_center", t("darkblob_p", "center"))
ag("darkblob_all", t("darkblob_p", "all"))
ag("brightblob_max", t("brightblob_p", "max")); ag("brightblob_center", t("brightblob_p", "center"))
ag("brightblob_all", t("brightblob_p", "all"))
ag("hstep_max", t("hstep_p", "max")); ag("hstep_n_max", t("hstep_n", "max"))
ag("hstep_top", t("hstep_p", "top")); ag("hstep_mid", t("hstep_p", "mid")); ag("hstep_bot", t("hstep_p", "bot"))
for r in range(4):
    ag("hstep_row%d" % r, t("hstep_p", "row%d" % r))
    ag("hstepn_row%d" % r, t("hstep_n", "row%d" % r))
ag("hstep_all", t("hstep_p", "all")); ag("hstepn_all", t("hstep_n", "all"))
ag("hstep_bal", t("hstep_p", "all"), t("hstep_n", "all", -1))
ag("vstep_max", t("vstep_p", "max")); ag("vstep_n_max", t("vstep_n", "max"))
ag("vstep_bal", t("vstep_p", "all"), t("vstep_n", "all", -1))
ag("vline_max", t("vline_p", "max")); ag("vline_n_max", t("vline_n", "max"))
ag("vline_bot2", t("vline_p", "bot2")); ag("vline_center", t("vline_p", "center"))
ag("vline_all", t("vline_p", "all")); ag("vline_bal", t("vline_p", "all"), t("vline_n", "all", -1))
ag("hline_max", t("hline_p", "max")); ag("hline_n_max", t("hline_n", "max"))
ag("hline_mid", t("hline_p", "mid")); ag("hline_all", t("hline_p", "all"))
ag("hline_bal", t("hline_p", "all"), t("hline_n", "all", -1))
ag("dline_max", t("dline45_p", "max", 0.5), t("dline135_p", "max", 0.5))
ag("dline_all", t("dline45_p", "all", 0.5), t("dline135_p", "all", 0.5))
ag("bigblob_max", t("bigblob_p", "max")); ag("bignblob_max", t("bigblob_n", "max"))
ag("bigblob_all", t("bigblob_p", "all")); ag("bignblob_all", t("bigblob_n", "all"))
ag("brightring_max", t("brightring_p", "max")); ag("brightring_all", t("brightring_p", "all"))
ag("vbar_max", t("vbar_p", "max")); ag("vbar_center", t("vbar_p", "center")); ag("vbar_all", t("vbar_p", "all"))
ag("hbar_max", t("hbar_p", "max")); ag("hbar_center", t("hbar_p", "center")); ag("hbar_all", t("hbar_p", "all"))
ag("corner_max", t("corner_p", "max")); ag("corner_all", t("corner_p", "all"))
ag("corner_n_max", t("corner_n", "max"))
ag("wheel_vs_blob", t("wheel_p", "max"), t("darkblob_p", "max", -1))
ag("vline_vs_blob", t("vline_p", "max"), t("darkblob_p", "max", -1))
ag("hline_vs_vline", t("hline_p", "max"), t("vline_p", "max", -1))
ag("ring_vs_blob", t("brightring_p", "max"), t("darkblob_p", "max", -1))
# ---- object colour / extent summaries
ag("obj_warm", t("brown", "center"), t("tan", "center"), t("red", "center", 0.5), t("yellow", "center", 0.5))
ag("obj_green", t("green_veg", "center"), t("yellowgreen", "center"))
ag("obj_achrom", t("gray_mid", "center"), t("achrom", "center", 0.5))
ag("obj_dark", t("dark", "center"))
ag("obj_white", t("white_obj", "center"))
ag("obj_blue", t("blue_dom", "center"))
ag("obj_sat", t("saturated", "center"))
ag("bg_skygreen", *sky("ring"), t("green_veg", "ring"))
ag("bg_water", *water("ring"))
ag("bg_gray", t("gray_mid", "ring"), t("achrom", "ring", 0.3))
ag("bg_green", t("green_veg", "ring"), t("green_dark", "ring", 0.5))
ag("bg_brown", t("brown", "ring"), t("tan", "ring"))
ag("bg_dark", t("dark", "ring"))
ag("urban", t("gray_mid", "ring"), t("corner_p", "max", 0.4), t("green_veg", "ring", -1), *sky("ring", -0.7))
ag("indoor", t("brown", "ring", 0.6), t("tan", "ring", 0.6), t("gray_mid", "ring", 0.4),
   *sky("ring", -1.0), t("green_veg", "ring", -1.0), t("water_blue", "ring", -1.0))
ag("foliage", t("green_dark", "all"), t("green_veg", "all", 0.5), *fine("all", 0.5))
ag("grassland", t("green_veg", "bot2"), *sky("top", 0.5), t("green_dark", "all", -0.5))

ag("tex_smooth_center", *fine("center", -1.0), *midtex("center", -1.0), const=0.25)
ag("tex_smooth_rel", *fine("all"), *fine("ring", -1.0))
ag("tex_center_rel", *fine("center"), *fine("all", -1.0))
ag("tex_mid_smooth_rel", *midtex("all"), *midtex("ring", -1.0))
ag("tex_ring_vs_center", *fine("ring"), *fine("center", -1.0))
ag("tex_smooth_bot", *fine("bot", -1.0), *midtex("bot", -1.0), const=0.25)
ag("hstep_ctop", t("hstep_p", "ctop")); ag("hstepn_ctop", t("hstep_n", "ctop"))
ag("hstep_cbot", t("hstep_p", "cbot")); ag("hstepn_cbot", t("hstep_n", "cbot"))
ag("sky_ring_center", *sky("ring"), *sky("center", -1.0))
ag("green_ring_center", t("green_veg", "ring"), t("green_veg", "center", -1))
ag("brown_ring_center", t("brown", "ring"), t("brown", "center", -1))
ag("warm_center", t("brown", "center"), t("tan", "center"), t("yellow", "center", 0.6))
ag("cool_center", t("blue_dom", "center"), t("gray_mid", "center", 0.5))
ag("obj_extent", t("dark", "center"), t("brown", "center"), t("tan", "center"),
   t("saturated", "center", 0.5), t("white_obj", "center", 0.5), t("gray_mid", "center", 0.5))
ag("bg_plain", *sky("ring"), t("green_veg", "ring"), *water("ring"), t("gray_mid", "ring", 0.5))
ag("bg_frac_ring", *fine("ring", -1.0), const=0.25)
ag("wheel_low", t("wheel_p", "qmaxbl"), t("wheel_p", "qmaxbr"))
ag("wheel_high", t("wheel_p", "qmaxtl"), t("wheel_p", "qmaxtr"))
ag("vline_low", t("vline_p", "qmaxbl"), t("vline_p", "qmaxbr"))
ag("hline_up", t("hline_p", "qmaxtl"), t("hline_p", "qmaxtr"))
ag("bright_top", t("bright", "top")); ag("dark_bot", t("dark", "bot"))
ag("sat_ring", t("saturated", "ring"))
ag("texfine_max", *fine("max"))
ag("texmid_max", *midtex("max"))
ag("hstep_maxall", t("hstep_p", "max"), t("hstep_n", "max", -1))
ag("white_ring_center", t("white_obj", "ring"), t("white_obj", "center", -1))
ag("gray_ring_center", t("gray_mid", "ring"), t("gray_mid", "center", -1))
ag("L_ring_center", t("raw_L", "ring"), t("raw_L", "center", -1))
ag("green_all_bot", t("green_veg", "all"), t("green_veg", "bot2", 0.5))
ag("sky_water_top", *sky("top"), *water("top", -1.0))
ag("notsky", *sky("all", -1.0), const=1.0)
ag("notgreen", t("green_veg", "all", -1.0), const=1.0)
ag("notwater", *water("all", -1.0), const=1.0)

NA = len(AGG)
ANAME = [a[0] for a in AGG]
AIDX = {n: i for i, n in enumerate(ANAME)}
WA = np.zeros((NA, NC * 16)); WM = np.zeros((NA, NC)); WX = np.zeros((NA, NC)); WQ = np.zeros((NA, NC * 4))
BA = np.zeros(NA)
for i, (name, terms, const) in enumerate(AGG):
    BA[i] = const
    for ch, reg, k in terms:
        c = CH_ALL.index(ch)
        if reg == "all":
            WM[i, c] += k
        elif reg == "max":
            WX[i, c] += k
        elif reg.startswith("qmax"):
            WQ[i, c * 4 + QMAX[reg[4:]]] += k
        else:
            for r, cj, wgt in REGION[reg]:
                WA[i, c * 16 + r * 4 + cj] += k * wgt
dense("aggA", "fa", WA, BA)
dense("aggM", "fm", WM, np.zeros(NA))
dense("aggX", "fx", WX, np.zeros(NA))
dense("aggQ", "fq", WQ, np.zeros(NA))
add("agg", "aggA", "aggM", "aggX", "aggQ")
# ============================================================ hand-written readout
# Class order: 0 airplane 1 automobile 2 bird 3 cat 4 deer 5 dog 6 frog 7 horse
#              8 ship 9 truck
# Each class accumulates evidence from scene layout, object colour and object
# shape/texture.  Thresholds are hand-chosen from what these scenes look like.
EVTAB = {
 0: (-0.9, "skyblue_ring 1.0 sky_ring 0.5 sky_ring_center 0.8 tex_smooth_ring 0.6 obj_achrom 0.5 "
           "white_center 0.3 hline_max 0.3 dline_max 0.3 texcoarse_center 0.2 L_top 0.2 "
           "water_layout -1.0 green_ring -0.8 brown_center -0.6 wheel_max -0.5 texfine_center -0.3 "
           "sat_center -0.3 gdark_ring -0.4"),
 1: (-1.0, "wheel_max 1.5 wheel_bot2 0.8 gray_ring 0.5 urban 0.4 sat_center 0.5 hline_max 0.3 "
           "hbar_max 0.3 red_center 0.3 blue_center 0.3 sky_top 0.2 hstepn_ctop 0.3 "
           "green_ring -0.6 water_all -0.6 white_center -0.4 bigblob_max -0.3 brown_center -0.4 "
           "gdark_ring -0.4 yg_center -0.3"),
 2: (-0.7, "green_ring 0.4 obj_warm 0.5 brown_center 0.3 yg_center 0.3 tex_center_ring 0.7 "
           "texfine_center 0.4 tex_smooth_ring 0.4 darkblob_max 0.2 dline_max 0.3 hline_max 0.2 "
           "skyblue_ring 0.2 water_layout -0.6 wheel_max -0.6 obj_achrom -0.1 bigblob_max -0.3 "
           "sky_ring_center -0.3 green_center_ring -0.3"),
 3: (-0.6, "indoor 0.5 obj_achrom 0.6 gray_center 0.4 achrom_center 0.3 texfine_all 0.5 "
           "texfine_center 0.4 darkblob_max 0.3 dark_center 0.3 corner_max 0.2 bg_brown 0.2 "
           "sky_ring -1.0 green_ring -0.8 water_all -0.8 wheel_max -0.6 green_bot2 -0.4 "
           "sat_center -0.3 bigblob_max -0.2"),
 4: (-1.0, "gdark_ring 1.0 gdark_all 0.5 brown_center 0.8 brown_center_ring 0.6 green_ring 0.4 "
           "foliage 0.3 vline_max 0.3 tan_center 0.2 texfine_ring 0.3 sky_ring -0.6 water_all -0.5 "
           "green_bot2 -0.4 white_center -0.4 wheel_max -0.4 bigblob_max -0.3 gray_ring -0.3"),
 5: (-0.7, "brown_center 0.5 tan_center 0.5 white_center 0.4 obj_warm 0.4 texfine_all 0.5 "
           "texfine_center 0.4 green_ring 0.2 darkblob_max 0.3 indoor 0.3 bg_brown 0.3 white_all 0.2 "
           "sky_ring -0.8 water_all -0.7 wheel_max -0.6 obj_achrom -0.3 gdark_ring -0.4 yg_center -0.3"),
 6: (-1.1, "yg_center 1.2 green_center 0.7 obj_green 0.7 green_all 0.5 green_center_ring 0.8 "
           "texfine_center 0.3 tex_blob_center 0.3 darkblob_max 0.2 yg_all 0.3 sky_ring -0.9 "
           "water_all -0.6 obj_achrom -0.6 white_center -0.4 wheel_max -0.4 brown_center -0.2 "
           "skyblue_ring -0.5"),
 7: (-1.1, "green_bot2 1.0 grassland 0.5 brown_center 0.7 obj_warm 0.4 sky_top 0.4 vline_bot2 0.4 "
           "bigblob_max 0.3 white_center 0.3 brown_all 0.3 hline_max 0.2 gdark_ring -0.7 "
           "water_all -0.5 gray_ring -0.4 wheel_max -0.5 texfine_center -0.2 yg_center -0.3"),
 8: (-1.0, "water_layout 1.3 waterb_all 0.5 water_ring 0.5 hstep_mid 0.4 sky_top 0.3 hline_max 0.15 "
           "brightblob_center 0.2 white_center 0.2 tex_hstruct_ring 0.2 green_ring -0.8 "
           "brown_center -0.5 wheel_max -0.6 obj_warm -0.3 gdark_ring -0.5 gray_ring -0.2"),
 9: (-1.6, "wheel_max 1.4 wheel_bot2 0.8 white_center 0.3 bigblob_max 0.2 bright_all 0.15 hbar_max 0.25 "
           "vbar_max 0.2 gray_ring 0.4 urban 0.4 corner_max 0.2 sky_top 0.2 texfine_center -0.4 "
           "green_ring -0.6 water_all -0.6 sat_center -0.5 brown_center -0.3 red_center -0.3 "
           "gdark_ring -0.4 sky_ring -0.5 texfine_all -0.3"),
}
NEV = 10
WEV = np.zeros((NEV, NA)); BEV = np.zeros(NEV)
for k, (bias, spec) in EVTAB.items():
    BEV[k] = bias
    tok = spec.split()
    for j in range(0, len(tok), 2):
        WEV[k, AIDX[tok[j]]] += float(tok[j+1])
dense("ev", "agg", WEV, BEV)

# ---- threshold conditions: capped hinges clip((a-thr)/width, 0, 1)
CONDS, CIDX = [], {}


def cd(agg, op, thr, width):
    key = (agg, op, thr, width)
    if key not in CIDX:
        CIDX[key] = len(CONDS); CONDS.append(key)
    return CIDX[key]


def C(agg, op, thr, width=0.18):
    return cd(agg, op, thr, width)


SKY_HI = C("sky_ring", ">", 0.40, 0.20); SKY_VHI = C("sky_ring", ">", 0.60, 0.20)
SKYB_HI = C("skyblue_ring", ">", 0.30, 0.20); SKYB_RC = C("skyblue_ring_center", ">", 0.12, 0.12)
SKY_LO = C("sky_ring", "<", 0.15, 0.12); SKYTOP_HI = C("sky_top", ">", 0.45, 0.20)
WAT_HI = C("water_layout", ">", 0.12, 0.12); WAT_VHI = C("water_layout", ">", 0.30, 0.15)
WATALL_HI = C("waterb_all", ">", 0.20, 0.15); WAT_LO = C("water_layout", "<", 0.04, 0.06)
WATALL_VHI = C("waterb_all", ">", 0.45, 0.15); SKYBOT_HI = C("sky_bot", ">", 0.35, 0.18)
GRN_HI = C("green_ring", ">", 0.22, 0.15); GRN_LO = C("green_ring", "<", 0.10, 0.08)
GRNALL_HI = C("green_all", ">", 0.25, 0.15); GRNBOT_HI = C("green_bot2", ">", 0.30, 0.18)
GDK_HI = C("gdark_ring", ">", 0.16, 0.13); GDK_LO = C("gdark_ring", "<", 0.10, 0.08)
GRNC_HI = C("green_center", ">", 0.25, 0.15); GRNCR_HI = C("green_center_ring", ">", 0.06, 0.08)
BRN_HI = C("brown_center", ">", 0.22, 0.14); WARM_HI = C("warm_center", ">", 0.28, 0.16)
YG_HI = C("yg_center", ">", 0.15, 0.12); YG_VHI = C("yg_center", ">", 0.30, 0.13)
ACH_HI = C("obj_achrom", ">", 0.28, 0.16); WHT_HI = C("white_center", ">", 0.25, 0.14)
SAT_HI = C("sat_center", ">", 0.25, 0.14); DRK_HI = C("dark_center", ">", 0.22, 0.13)
RED_HI = C("red_center", ">", 0.22, 0.13); BLU_HI = C("blue_center", ">", 0.22, 0.13)
SMR = C("tex_smooth_rel", ">", 0.015, 0.030); RGH_ALL = C("texfine_all", ">", 0.10, 0.07)
RGH_C = C("texfine_center", ">", 0.10, 0.07); TEXCR = C("tex_center_ring", ">", 0.02, 0.03)
SMC = C("tex_center_rel", "<", -0.015, 0.030); HSTR = C("tex_hstruct_ring", ">", 0.08, 0.06)
HSTP_HI = C("hstep_mid", ">", 0.20, 0.10)
WHL = C("wheel_max", ">", 0.16, 0.08); WHL_V = C("wheel_max", ">", 0.22, 0.08)
WHL_LO = C("wheel_max", "<", 0.10, 0.06); WHLB = C("wheel_bot2", ">", 0.06, 0.05)
BIGB = C("bigblob_max", ">", 0.35, 0.12); HLIN = C("hline_max", ">", 0.26, 0.10)
VLIN = C("vline_max", ">", 0.24, 0.10); HSTP = C("hstep_mid", ">", 0.15, 0.08)
DBLB = C("darkblob_max", ">", 0.30, 0.10); CORN = C("corner_max", ">", 0.28, 0.10)
HBAR = C("hbar_max", ">", 0.26, 0.10); VBAR = C("vbar_max", ">", 0.26, 0.10)
BRBL = C("brightblob_max", ">", 0.35, 0.12)
URB = C("urban", ">", 0.15, 0.15); INDR = C("indoor", ">", 0.20, 0.15)
GRAY_HI = C("gray_ring", ">", 0.22, 0.16); SKYRC = C("sky_ring_center", ">", 0.15, 0.12)

RULES = []


def ru(conds, k, w):
    RULES.append((list(conds), k, w))


# aeroplane: sky around a smooth, rigid, achromatic object
ru([SKYB_HI, SMR, WAT_LO], 0, 2.0); ru([SKYB_RC, SMR, WAT_LO], 0, 1.3)
ru([SKY_VHI, ACH_HI, WAT_LO], 0, 1.2); ru([SKY_HI, HLIN, GRN_LO], 0, 0.8)
ru([SKYB_HI, SKYBOT_HI], 0, 1.8); ru([SKYB_HI, WAT_LO, GRN_LO, ACH_HI], 0, 1.5); ru([SKYB_RC, WAT_LO, GRN_LO], 0, 1.2)
ru([SKYTOP_HI, ACH_HI, WAT_LO, GRN_LO], 0, 1.0)
ru([SKY_HI, RGH_C], 0, -0.9); ru([SKY_HI, BRN_HI], 0, -1.0)
ru([SKY_HI, WHL], 0, -0.8); ru([SKY_HI, WHT_HI, BIGB], 0, -0.6)
# ship: blue water under a horizon, sky above
ru([SKYTOP_HI, WAT_HI], 8, 2.4); ru([WAT_VHI, GRN_LO], 8, 1.8)
ru([WATALL_HI, HSTP], 8, 1.4); ru([WAT_HI, BRBL], 8, 0.9)
ru([WAT_HI, HSTR], 8, 0.7); ru([SKYTOP_HI, HSTP_HI, SMR, GRN_LO, WHL_LO], 8, 1.5)
ru([WATALL_VHI, SMR, GRN_LO, WHL_LO], 8, 1.2); ru([SKYBOT_HI], 8, -1.2)
ru([WATALL_HI, WHL], 8, -1.5); ru([WATALL_HI, BRN_HI, GRN_HI], 8, -1.0)
# bird: small warm textured object on a smooth or green background
ru([GRN_HI, TEXCR, WARM_HI], 2, 1.6); ru([SMR, TEXCR, SKY_LO, WARM_HI], 2, 1.4)
ru([SKYB_HI, BRN_HI], 2, 1.2); ru([GRN_HI, RGH_C, DBLB, WARM_HI], 2, 0.9)
ru([SKY_HI, RGH_C, ACH_HI], 2, -0.8); ru([WAT_HI], 2, -1.0)
ru([WHL], 2, -1.2); ru([YG_VHI, GRNC_HI], 2, -1.0)
ru([BIGB, WHT_HI], 2, -0.6); ru([GRN_HI, TEXCR, SKY_LO], 2, 1.2); ru([SMR, TEXCR, SKY_LO, DBLB], 2, 1.2)
ru([SMR, TEXCR, WAT_LO], 2, 1.0)
ru([GRN_HI, TEXCR, GDK_HI], 2, -0.5)
# frog: yellow-green bumpy body, no sky
ru([YG_HI, GRNALL_HI, SKY_LO], 6, 2.4); ru([GRNC_HI, GRNCR_HI, SKY_LO], 6, 1.7)
ru([YG_VHI, RGH_C], 6, 1.1); ru([GRNALL_HI, DBLB, SKY_LO, YG_HI], 6, 0.8)
ru([SKY_HI], 6, -1.5); ru([WATALL_HI], 6, -1.0)
ru([WHL], 6, -1.2); ru([ACH_HI, GRN_LO], 6, -1.0); ru([BRN_HI, GRN_LO], 6, -0.8)
ru([GRN_HI, GRNCR_HI, RGH_C, SKY_LO], 6, 1.2)
ru([GRNBOT_HI, GRNCR_HI, SKYTOP_HI], 6, -0.7)
# deer: brown body in dark foliage
ru([GDK_HI, BRN_HI], 4, 2.2); ru([GRN_HI, BRN_HI, GDK_HI], 4, 1.4)
ru([GDK_HI, WARM_HI, VLIN], 4, 1.0); ru([GRN_HI, BRN_HI, SKY_LO, WAT_LO], 4, 0.8)
ru([GRNBOT_HI, GDK_LO, BRN_HI], 4, -1.0); ru([SKY_HI], 4, -1.2)
ru([WHL], 4, -1.2); ru([WHT_HI], 4, -0.8); ru([WATALL_HI], 4, -1.0)
ru([BRN_HI, WARM_HI, VLIN, SKY_LO, WAT_LO], 4, 1.2)
ru([WARM_HI, VLIN, BIGB, SKY_LO, WAT_LO], 4, 0.8)
ru([YG_HI, GRNC_HI], 4, -0.8)
# horse: big brown/white body on bright grass
ru([GRNBOT_HI, BRN_HI, GDK_LO], 7, 2.2); ru([GRNBOT_HI, WARM_HI, SKYTOP_HI], 7, 1.4)
ru([GRNBOT_HI, BIGB, GDK_LO], 7, 1.0); ru([GRNBOT_HI, VLIN], 7, 0.9)
ru([GRN_HI, WHT_HI, GDK_LO], 7, 0.8); ru([GDK_HI], 7, -1.2)
ru([WHL], 7, -1.2); ru([WATALL_HI], 7, -1.0)
ru([SKY_LO, GRN_LO], 7, -0.8); ru([BRN_HI, GRN_HI, SMC], 7, 0.7)
ru([WARM_HI, VLIN, BIGB, GRN_LO, SKY_LO], 7, 0.9)
ru([YG_HI, GRNC_HI], 7, -0.7); ru([SKY_HI], 7, -1.0); ru([SKYBOT_HI], 7, -0.6)
# cat: fur texture, achromatic/dark body, indoor, no sky/green/water
ru([SKY_LO, GRN_LO, WAT_LO, RGH_ALL, ACH_HI], 3, 1.8)
ru([SKY_LO, GRN_LO, WAT_LO, RGH_ALL, DRK_HI], 3, 1.2)
ru([INDR, RGH_C, DBLB, ACH_HI], 3, 1.0); ru([SKY_LO, GRN_LO, RGH_ALL, CORN], 3, 0.6)
ru([SKY_HI], 3, -1.5); ru([GRN_HI], 3, -1.1); ru([WATALL_HI], 3, -1.2)
ru([WHL], 3, -1.2); ru([GRNBOT_HI], 3, -0.8); ru([SAT_HI], 3, -0.6)
ru([SKY_LO, GRN_LO, WARM_HI], 3, -0.6); ru([YG_HI], 3, -0.8); ru([HSTP_HI, SMR], 3, -0.8)
ru([GRNCR_HI], 3, -0.5); ru([VLIN, WARM_HI], 3, -0.6); ru([URB, HLIN], 3, -0.6)
# dog: fur texture, warm/white body, indoor or grass
ru([SKY_LO, GRN_LO, WAT_LO, RGH_ALL, WARM_HI], 5, 1.8)
ru([SKY_LO, GRN_LO, WAT_LO, RGH_ALL, WHT_HI], 5, 1.2)
ru([GRN_HI, WARM_HI, RGH_C, GDK_LO, SKY_LO], 5, 1.0); ru([INDR, RGH_C, WARM_HI], 5, 0.8)
ru([SKY_HI], 5, -1.3); ru([WATALL_HI], 5, -1.2); ru([WHL], 5, -1.2)
ru([GDK_HI], 5, -0.9); ru([SKY_LO, GRN_LO, ACH_HI], 5, -0.6)
ru([GRNBOT_HI, WARM_HI, BIGB], 5, -0.7); ru([YG_HI, GRNC_HI], 5, -0.8); ru([GRNCR_HI], 5, -0.7)
ru([DBLB, RGH_ALL, SKY_LO, WARM_HI], 5, 0.6); ru([VLIN, WARM_HI, GRN_LO], 5, -0.5)
# automobile: wheels on grey, colourful low body
ru([WHL, GRAY_HI, GRN_LO], 1, 1.8); ru([WHL, SAT_HI], 1, 1.6)
ru([WHLB, URB], 1, 1.1); ru([WHL, HLIN], 1, 0.8)
ru([WHL, RED_HI], 1, 0.9); ru([WHL, BLU_HI], 1, 0.7)
ru([WHL, WHT_HI, BIGB], 1, -1.2); ru([GRN_HI], 1, -1.0)
ru([WATALL_HI], 1, -1.2); ru([SKY_VHI], 1, -1.0); ru([GDK_HI], 1, -0.8)
ru([WHL, GRAY_HI], 1, 1.2); ru([WHL, SKY_LO, GRN_LO, WAT_LO], 1, 1.0)
ru([WHL_LO, GRAY_HI, SAT_HI, HLIN], 1, 0.7)
ru([SKY_LO, GRN_LO, WAT_LO, RGH_ALL], 1, -0.5)
# truck: wheels plus a big flat bright box
ru([WHL, WHT_HI, BIGB], 9, 2.2); ru([WHL, BIGB, SMC], 9, 1.4)
ru([WHLB, BRBL, URB], 9, 1.1); ru([WHL, HBAR], 9, 0.8)
ru([WHL, VBAR], 9, 0.7); ru([WHL, CORN], 9, 0.6)
ru([WHL, SKY_LO, BIGB], 9, 0.7); ru([WHL, SAT_HI], 9, -1.0)
ru([GRN_HI], 9, -1.0); ru([WATALL_HI], 9, -1.2)
ru([SKY_VHI], 9, -0.8); ru([GDK_HI], 9, -0.8)
ru([SKY_HI], 9, -1.2); ru([SKYBOT_HI], 9, -0.8)
ru([RGH_ALL, SKY_LO, GRN_LO, WAT_LO], 9, -0.8)

MC = len(CONDS)
WC = np.zeros((2*MC, NA)); BC = np.zeros(2*MC)
for i, (agg, op, thr, width) in enumerate(CONDS):
    s = 1.0 / width if op == ">" else -1.0 / width
    WC[i, AIDX[agg]] = s; WC[MC+i, AIDX[agg]] = s
    BC[i] = -s * thr; BC[MC+i] = -s * thr - 1.0
dense("cond", "agg", WC, BC)
relu("condr", "cond")
WCAP = np.zeros((MC, 2*MC))
for i in range(MC):
    WCAP[i, i] = 1.0; WCAP[i, MC+i] = -1.0
dense("condcap", "condr", WCAP, np.zeros(MC))

NR = len(RULES)
WR = np.zeros((NR, MC)); BR = np.zeros(NR)
for i, (cs, k, w) in enumerate(RULES):
    for c in cs:
        WR[i, c] += 1.0
    BR[i] = -(len(cs) - 1)
dense("rule", "condcap", WR, BR)
relu("ruler", "rule")
WRO = np.zeros((10, NR))
for i, (cs, k, w) in enumerate(RULES):
    WRO[k, i] += w
dense("ruleout", "ruler", WRO, np.zeros(10))
add("scores", "ev", "ruleout")

# ------------------------------------------------------------------------ save
import os
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": layers}, f)
np.savez(os.path.join(OUT, "weights.npz"),
         **{k: v.astype(np.float32) for k, v in weights.items()})
print("layers", len(layers), "weights", sum(v.size for v in weights.values()),
      "aggregates", NA, "conditions", MC, "rules", NR)
