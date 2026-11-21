#!/usr/bin/env python3
"""
build.py -- writes network.json and weights.npz for a hand-designed CIFAR-10 classifier.

Usage: python3 build.py OUT_FOLDER

Nothing is read from disk.  Every number below is either a formula (Gaussian derivative,
Laplacian of Gaussian, opponent-colour transforms) or a constant I chose from general
knowledge of what the ten classes look like (sky is bright and blue, grass is green, a wheel
is a dark disc near the bottom, legs are thin dark vertical bars, painted vehicles have
vivid colours and long straight edges, fur and frog skin are textured, ...).

Architecture
  image (3,32,32)
   |- col : conv 1x1 -> 12 colour channels (luminance, darkness, +-blue/yellow, +-green/magenta,
   |        +-R-G, +-G-B, +-B-R) -> relu -> avgpool 8x8 (ca): a 4x4 grid
   |- edg : conv 9x9 on luminance -> 22 edge/blob channels -> relu -> avgpool 6x6 (ea): a 4x4
            grid, and maxpool 6x24 (er): the maximum of each channel over 4 horizontal bands
            (the 9x9 kernel without padding gives 24x24 maps, so no border artefacts)
  concept layer: one dense layer per pooled grid, added together, then relu.
                 Each hidden unit is a named, hand-written detector such as "sky at the top",
                 "water at the bottom", "wheel", "legs", "horizon line", "vivid paint".
  output: dense 10, a hand-written table of how much each concept speaks for each class.
"""
import sys, json, os
import numpy as np

# ----------------------------------------------------------------------------------------
# 1. Filters from formulas
# ----------------------------------------------------------------------------------------
K = 9                                  # edge kernel size
LUM = np.array([0.30, 0.59, 0.11])     # luminance weights for R, G, B


def grid(n=K):
    r = n // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    return yy, xx


def deriv_filter(theta_deg, s_u, s_v):
    """First Gaussian derivative along direction theta (s_u across the edge, s_v along it).
    Normalised so that an ideal unit step edge gives |response| = 1 at the edge."""
    yy, xx = grid()
    t = np.deg2rad(theta_deg)
    u = xx * np.cos(t) + yy * np.sin(t)
    v = -xx * np.sin(t) + yy * np.cos(t)
    f = -u * np.exp(-u ** 2 / (2 * s_u ** 2) - v ** 2 / (2 * s_v ** 2))
    f -= f.mean()
    step = (u > 0).astype(float) + 0.5 * (u == 0)
    f /= abs((f * step).sum())
    return f


def log_filter(sigma):
    """Laplacian of Gaussian. Normalised so a dark disc (value -1, radius sqrt(2)*sigma) gives +1."""
    yy, xx = grid()
    r2 = xx ** 2 + yy ** 2
    f = (r2 - 2 * sigma ** 2) / sigma ** 4 * np.exp(-r2 / (2 * sigma ** 2))
    f -= f.mean()
    disc = -(r2 <= 2 * sigma ** 2).astype(float)
    f /= (f * disc).sum()
    return f


def bar_filter(vertical, s_along=2.5):
    """Thin dark line detector: centre column -1, columns at distance 2 +0.5 (sums to zero),
    weighted by a Gaussian along the line.  A 1-pixel dark line of depth 1 gives +1."""
    yy, xx = grid()
    along, across = (yy, xx) if vertical else (xx, yy)
    prof = np.zeros_like(across)
    prof[across == 0] = -1.0
    prof[np.abs(across) == 2] = 0.5
    g = np.exp(-along ** 2 / (2 * s_along ** 2))
    f = prof * g
    f /= -(f[across == 0]).sum()
    return f


# ---- edge conv (3 -> E channels, 9x9, no padding) ----
edge_names, edge_filters = [], []


def add_edge(name, f):
    edge_names.append(name)
    edge_filters.append(f)


for th in (0, 45, 90, 135):     # theta = gradient direction; 0 = vertical edges, 90 = horizontal edges
    f = deriv_filter(th, 1.0, 1.0)
    add_edge(f"fine{th}p", f)
    add_edge(f"fine{th}n", -f)
for th in (0, 45, 90, 135):     # elongated: respond to long straight edges
    f = deriv_filter(th, 1.0, 2.5)
    add_edge(f"long{th}p", f)
    add_edge(f"long{th}n", -f)
f = log_filter(0.8)
add_edge("logf_p", f)           # fine dark dots / texture
add_edge("logf_n", -f)          # fine bright dots / texture
f = log_filter(2.0)
add_edge("logc_p", f)           # dark blob of radius ~3 px (wheel, eye, shadow)
add_edge("logc_n", -f)          # bright blob
add_edge("vbar", bar_filter(True))      # thin dark vertical line (legs, masts, trunks)
add_edge("hbar", bar_filter(False))     # thin dark horizontal line
E = len(edge_names)
EIDX = {n: i for i, n in enumerate(edge_names)}

W_edg = np.zeros((E, 3, K, K))
for i, f in enumerate(edge_filters):
    for c in range(3):
        W_edg[i, c] = f * LUM[c]
b_edg = np.zeros(E)

# ---- colour conv (3 -> 12 channels, 1x1) ----
color_specs = [
    ("L", [0.30, 0.59, 0.11], 0.0),          # luminance
    ("D", [-0.30, -0.59, -0.11], 1.0),       # darkness = 1 - luminance
    ("BYp", [-0.5, -0.5, 1.0], 0.0),         # blue minus yellow (sky, water)
    ("BYn", [0.5, 0.5, -1.0], 0.0),          # yellow / brown / olive
    ("GMp", [-0.5, 1.0, -0.5], 0.0),         # green minus magenta (grass, leaves)
    ("GMn", [0.5, -1.0, 0.5], 0.0),
    ("RGp", [1.0, -1.0, 0.0], 0.0),          # red minus green (red, orange, brown)
    ("RGn", [-1.0, 1.0, 0.0], 0.0),
    ("GBp", [0.0, 1.0, -1.0], 0.0),
    ("GBn", [0.0, -1.0, 1.0], 0.0),
    ("BRp", [-1.0, 0.0, 1.0], 0.0),
    ("BRn", [1.0, 0.0, -1.0], 0.0),
]
C = len(color_specs)
CIDX = {s[0]: i for i, s in enumerate(color_specs)}
W_col = np.zeros((C, 3, 1, 1))
b_col = np.zeros(C)
for i, (n, w, b) in enumerate(color_specs):
    W_col[i, :, 0, 0] = w
    b_col[i] = b

# ----------------------------------------------------------------------------------------
# 2. Concept layer: hand-written detectors over the pooled grids
# ----------------------------------------------------------------------------------------
# grid cells are (row, col), row 0 = top; the band grid ("er") has cells (row, 0)
ALL = [(r, c) for r in range(4) for c in range(4)]
TOP = [(0, c) for c in range(4)]
ROW1 = [(1, c) for c in range(4)]
ROW2 = [(2, c) for c in range(4)]
BOT = [(3, c) for c in range(4)]
MID = ROW1 + ROW2
CENTER = [(r, c) for r in (1, 2) for c in (1, 2)]
BORDER = [x for x in ALL if x not in CENTER]
LOWER = ROW2 + BOT
# band cells
B_TOP, B_ROW1, B_ROW2, B_BOT = [(0, 0)], [(1, 0)], [(2, 0)], [(3, 0)]
B_LOWER = B_ROW2 + B_BOT
B_MID = B_ROW1 + B_ROW2

# branch name -> (channels, grid height, grid width)
BR = {"ca": (C, 4, 4), "ea": (E, 4, 4), "er": (E, 4, 1)}

units = []      # (name, terms, bias); terms = list of (branch, channel_name, cells, weight)
ramps = {}      # name -> (unit_lo_name, unit_hi_name) for saturating units


def unit(name, terms, bias=0.0):
    units.append((name, terms, float(bias)))


def signed_pair(name, terms, bias=0.0):
    """Two relu units +f and -f so the output layer can use the signed value f linearly."""
    unit(name + "+", terms, bias)
    unit(name + "-", [(b, ch, cells, -w) for (b, ch, cells, w) in terms], -bias)


def ramp(name, terms, lo, hi):
    """relu(f - lo) - relu(f - hi): a unit that rises from f=lo and saturates at f=hi."""
    unit(name + "@lo", terms, -lo)
    unit(name + "@hi", terms, -hi)
    ramps[name] = (name + "@lo", name + "@hi")


# --- helpers that build term lists -----------------------------------------------------
def col_signed(pos, neg, cells, w):
    """signed colour opponent value (mean over cells) = relu(+) - relu(-)"""
    return [("ca", pos, cells, w), ("ca", neg, cells, -w)]


def L(cells, w):
    return [("ca", "L", cells, w)]


def BY(cells, w):
    return col_signed("BYp", "BYn", cells, w)


def GM(cells, w):
    return col_signed("GMp", "GMn", cells, w)


def RG(cells, w):
    return col_signed("RGp", "RGn", cells, w)


def SAT(cells, w):
    """|R-G| + |G-B| + |B-R| averaged over cells"""
    return [("ca", ch, cells, w) for ch in ("RGp", "RGn", "GBp", "GBn", "BRp", "BRn")]


def EDGE(cells, w, kinds=("fine0", "fine45", "fine90", "fine135")):
    """edge energy: sum over orientations of |derivative| (average pooled)"""
    return [("ea", k + s, cells, w) for k in kinds for s in ("p", "n")]


def LONG(cells, w, kinds=("long0", "long45", "long90", "long135")):
    return [("ea", k + s, cells, w) for k in kinds for s in ("p", "n")]


def TEX(cells, w):
    return [("ea", "logf_p", cells, w), ("ea", "logf_n", cells, w)]


def BANDMAX(ch, cells, w):
    return [("er", ch, cells, w)]


# ---- A. signed linear features (templates) --------------------------------------------
# scale factors chosen so a clear case gives roughly +-0.5 .. 1
signed_pair("L_all", L(ALL, 3.0), -0.45 * 3.0)                 # bright image vs dark image
signed_pair("L_topbot", L(TOP, 3.0) + L(BOT, -3.0))             # brighter top than bottom (sky over ground)
signed_pair("L_bot", L(BOT, 3.0), -0.45 * 3.0)                 # bright bottom (sky below a plane) vs dark bottom
signed_pair("BY_all", BY(ALL, 4.0))                            # blue image
signed_pair("BY_top", BY(TOP, 4.0))
signed_pair("BY_bot", BY(BOT, 4.0))
signed_pair("BY_center", BY(CENTER, 4.0))
signed_pair("GM_all", GM(ALL, 5.0))                            # green image
signed_pair("GM_bot", GM(BOT, 5.0))
signed_pair("GM_center", GM(CENTER, 5.0))
signed_pair("RG_all", RG(ALL, 4.0))                            # reddish / brown image
signed_pair("RG_center", RG(CENTER, 4.0))
signed_pair("SAT_all", SAT(ALL, 2.0), -0.4 * 2.0)               # colourful vs grey image
signed_pair("E_all", EDGE(ALL, 4.0), -0.12 * 4.0)               # edge energy (scale uncertain: used with small weights)
signed_pair("E_cb", EDGE(CENTER, 6.0) + EDGE(BORDER, -6.0))     # object edges vs background edges
signed_pair("E_top", EDGE(TOP, 4.0), -0.10 * 4.0)               # edges in the top row (tall object) vs empty sky
signed_pair("TEX_all", TEX(ALL, 6.0), -0.06 * 6.0)              # fine texture (scale uncertain: small weights)
# Orientation contrasts are self-centring: for isotropic content the two sums are equal.
# With four 45-degree-spaced derivative channels an axis-aligned edge excites the two diagonal
# channels (0.67 each) more than a diagonal edge excites the two axis channels, so
# "axis-aligned" = diagonal-channel sum minus axis-channel sum.
signed_pair("HV", EDGE(ALL, 20.0, ("fine45", "fine135")) + EDGE(ALL, -20.0, ("fine0", "fine90")))
# horizontal long edges vs vertical long edges (hull, wings, car body vs. tall truck, trees, legs)
signed_pair("HdomV", LONG(ALL, 15.0, ("long90",)) + LONG(ALL, -15.0, ("long0",)))

# ---- B. thresholded concepts ---------------------------------------------------------
# sky at the top: bluish, bright, smooth (saturates so a very blue sky does not dominate)
ramp("sky_top", BY(TOP, 4.0) + L(TOP, 2.0) + EDGE(TOP, -3.0), 1.2, 2.4)
# pale / white / overcast sky at the top: bright and smooth whatever the hue
ramp("bright_smooth_top", L(TOP, 3.0) + EDGE(TOP, -6.0), 1.95, 2.7)
# sky at the bottom too (plane in flight): bluish / bright, smooth
ramp("sky_bot", BY(BOT, 4.0) + L(BOT, 2.0) + EDGE(BOT, -3.0), 1.2, 2.4)
# water at the bottom: bluish and not too bright
ramp("water_bot", BY(BOT, 4.0) + L(BOT, -2.0), -0.6, 0.6)
# blue everywhere (plane in sky, boat on sea)
ramp("blue_all", BY(ALL, 5.0), 0.25, 1.5)
# grass / foliage at the bottom
ramp("grass_bot", GM(BOT, 5.0), 0.15, 1.2)
# green everywhere
ramp("green_all", GM(ALL, 5.0), 0.1, 1.0)
# grey road / pavement at the bottom: unsaturated, mid-bright, smooth
unit("road_bot", SAT(BOT, -6.0) + EDGE(BOT, -9.0) + L(BOT, 1.5), bias=0.9)
# brown object in the centre: red > green and yellowish, saturating so red paint is not "very brown"
ramp("brown_center", RG(CENTER, 3.0) + BY(CENTER, -3.0), 0.3, 1.3)
# vivid red / orange object
ramp("red_center", RG(CENTER, 3.0), 1.05, 2.1)
# vivid yellow / orange (very un-blue) object
ramp("yellow_center", BY(CENTER, -3.0), 1.2, 2.4)
# grey / white object in the centre
unit("grey_center", SAT(CENTER, -2.0) + L(CENTER, 0.6), bias=0.3 - 0.6 * 0.4)
# dark object in the centre
unit("dark_center", L(CENTER, -3.0), bias=0.35 * 3.0)
# textured centre (fur, frog skin, feathers)
unit("tex_center", TEX(CENTER, 6.0) + EDGE(CENTER, -1.0), bias=-0.25)
# smooth image overall (plane in empty sky)
unit("smooth_all", EDGE(ALL, -6.0), bias=0.08 * 6.0)
# vivid paint: the centre cells themselves are strongly saturated (painted body fills the middle)
ramp("vivid", SAT(CENTER, 1.0), 0.7, 1.4)


# Band contrast: the maximum of a relu'd detector channel in one horizontal band minus its maximum
# in a reference band of the same image.  The noise floor of a maximum scales with the image's
# texture level, which is similar in both bands, so the difference is self-centring.
def BANDDIFF(ch, bands, ref_bands, w=2.0):
    return BANDMAX(ch, bands, w) + BANDMAX(ch, ref_bands, -w)


# wheel: dark round blobs in the bottom band but not in the top band
ramp("wheel", BANDDIFF("logc_p", B_BOT, B_TOP), 0.1, 0.8)
# legs: thin dark vertical bars in the lower half but not in the upper half
ramp("legs", BANDDIFF("vbar", B_LOWER, [B_TOP[0], B_ROW1[0]]), 0.1, 0.8)
# horizon: a long horizontal edge in a middle band, stronger than anything in the top band
ramp("horizon1", BANDDIFF("long90p", B_ROW1, B_TOP) + BANDDIFF("long90n", B_ROW1, B_TOP), 0.2, 1.2)
ramp("horizon2", BANDDIFF("long90p", B_ROW2, B_TOP) + BANDDIFF("long90n", B_ROW2, B_TOP), 0.2, 1.2)
# bright blob in the middle bands, stronger than in the top band (white plane / boat / pale face)
ramp("bright_blob", BANDDIFF("logc_n", B_MID, B_TOP), 0.1, 0.8)
# rectangular man-made structure: long axis-aligned edges beat long diagonal ones (symmetric, so
# isotropic content gives zero)
unit("boxy", LONG(ALL, 25.0, ("long0", "long90")) + LONG(ALL, -25.0, ("long45", "long135")), bias=-0.05)
# plain background: little edge energy in the border cells
unit("plain_border", EDGE(BORDER, -6.0), bias=0.08 * 6.0)

H = len(units)
UIDX = {u[0]: i for i, u in enumerate(units)}

W_dense = {b: np.zeros((H, int(np.prod(shape)))) for b, shape in BR.items()}
b_hidden = np.zeros(H)
for i, (name, terms, bias) in enumerate(units):
    b_hidden[i] = bias
    for (branch, ch, cells, w) in terms:
        nch, gh, gw = BR[branch]
        chi = CIDX[ch] if branch == "ca" else EIDX[ch]
        for (r, c) in cells:
            W_dense[branch][i, chi * gh * gw + r * gw + c] += w / len(cells)

# ----------------------------------------------------------------------------------------
# 3. Output table: how much each concept speaks for each class
# ----------------------------------------------------------------------------------------
#          classes: 0 plane, 1 car, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
T = {
    #                    plane  car  bird  cat  deer  dog  frog horse ship truck
    "L_all":            [ 1.0,  0.0,  0.0, -0.1, -0.3,  0.0, -0.6,  0.0,  0.3,  0.2],
    "L_topbot":         [ 0.0,  0.4,  0.3, -0.2,  0.4, -0.2, -0.5,  0.5,  0.5,  0.3],
    "L_bot":            [ 0.8, -0.2,  0.2,  0.0, -0.3,  0.0, -0.3, -0.2, -0.5,  0.0],
    "BY_all":           [ 1.2,  0.0,  0.4, -0.3, -0.8, -0.5, -0.8, -0.8,  1.5,  0.0],
    "BY_top":           [ 0.6,  0.2,  0.5, -0.3,  0.3, -0.3, -0.6,  0.4,  0.8,  0.3],
    "BY_bot":           [ 0.8, -0.3, -0.2, -0.3, -0.5, -0.4, -0.6, -0.6,  1.0, -0.3],
    "BY_center":        [ 0.5, -0.2,  0.1, -0.3, -0.6, -0.4, -0.7, -0.6,  0.6, -0.2],
    "GM_all":           [-1.0, -0.3,  0.6,  0.0,  1.0,  0.2,  0.8,  0.8, -1.0, -0.5],
    "GM_bot":           [-0.6,  0.2,  0.4,  0.0,  0.7,  0.5,  0.3,  1.0, -1.0, -0.3],
    "GM_center":        [-0.5, -0.4,  0.2, -0.2,  0.3, -0.1,  0.6,  0.0, -0.6, -0.4],
    "RG_all":           [-0.8,  0.2,  0.2,  0.4,  0.8,  0.6,  0.4,  0.8, -0.8,  0.2],
    "RG_center":        [-0.5,  0.3,  0.2,  0.3,  0.6,  0.5,  0.3,  0.6, -0.5,  0.2],
    "SAT_all":          [-0.3,  0.6,  0.2, -0.5,  0.0, -0.3,  0.0,  0.0, -0.2,  0.5],
    "E_all":            [-0.8,  0.2,  0.2,  0.6,  0.3,  0.6,  0.4,  0.0, -0.3,  0.2],
    "E_cb":             [ 0.5,  0.3,  0.8, -0.2, -0.3, -0.1, -0.5,  0.0,  0.3,  0.0],
    "E_top":            [-0.5, -0.3, -0.3,  0.3,  0.2,  0.3,  0.3, -0.3, -0.5,  0.6],
    "TEX_all":          [-0.6, -0.5,  0.3,  0.6,  0.3,  0.6,  0.6,  0.0, -0.5, -0.4],
    "HV":               [ 0.3,  0.8, -0.4,  0.0, -0.3,  0.0, -0.5, -0.3,  0.3,  0.8],
    "HdomV":            [ 0.5,  0.4,  0.0,  0.0, -0.2,  0.0, -0.2,  0.2,  0.5, -0.4],
    "sky_top":          [ 0.8,  0.0,  0.7, -0.5, -0.2, -0.5, -1.0,  0.5,  0.8,  0.3],
    "bright_smooth_top":[ 0.6,  0.1,  0.3, -0.2, -0.2, -0.2, -0.6,  0.4,  0.5,  0.3],
    "sky_bot":          [ 1.2, -0.3,  0.5, -0.5, -0.5, -0.5, -0.8, -0.5, -0.5, -0.3],
    "water_bot":        [-0.5, -0.8,  0.0, -0.5, -0.5, -0.5, -0.5, -0.5,  1.5, -0.8],
    "blue_all":         [ 0.8, -0.3,  0.3, -0.5, -0.8, -0.5, -1.0, -0.8,  1.0, -0.3],
    "grass_bot":        [-0.6,  0.1,  0.4, -0.2,  0.7,  0.5,  0.3,  1.0, -1.0, -0.4],
    "green_all":        [-0.6, -0.5,  0.5, -0.3,  0.6,  0.0,  0.5,  0.3, -0.8, -0.5],
    "road_bot":         [ 0.4,  1.0, -0.3,  0.1, -0.5,  0.1, -0.5, -0.5, -1.0,  1.0],
    "brown_center":     [-0.8,  0.0,  0.2,  0.3,  1.0,  0.8,  0.2,  1.0, -0.8,  0.0],
    "red_center":       [-0.5,  0.8,  0.3, -0.5, -0.5, -0.5, -0.3, -0.5, -0.3,  0.8],
    "yellow_center":    [-0.5,  0.5,  0.2, -0.3, -0.2, -0.2,  0.1, -0.2, -0.5,  0.6],
    "grey_center":      [ 0.6,  0.3,  0.0,  0.4, -0.5,  0.1, -0.5, -0.3,  0.5,  0.3],
    "dark_center":      [-0.6,  0.2,  0.1,  0.5, -0.2,  0.0,  0.5,  0.2,  0.0,  0.0],
    "tex_center":       [-0.6, -0.5,  0.3,  0.8,  0.3,  0.8,  0.8,  0.0, -0.5, -0.5],
    "smooth_all":       [ 1.0, -0.3, -0.3, -0.5, -0.5, -0.5, -0.5, -0.3,  0.3, -0.3],
    "vivid":            [-0.3,  1.0,  0.2, -0.5, -0.3, -0.5,  0.0, -0.3, -0.2,  0.8],
    "wheel":            [-0.3,  1.0, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3,  0.8],
    "legs":             [-0.5, -0.5,  0.3,  0.0,  1.0,  0.2, -0.3,  1.0, -0.2, -0.3],
    "horizon1":         [ 0.3,  0.3, -0.2, -0.3,  0.0, -0.3, -0.3,  0.2,  0.5,  0.2],
    "horizon2":         [ 0.2,  0.3, -0.2, -0.3,  0.0, -0.3, -0.3,  0.2,  0.6,  0.2],
    "bright_blob":      [ 0.5,  0.2,  0.3,  0.1, -0.2,  0.2, -0.4,  0.2,  0.3,  0.1],
    "boxy":             [ 0.0,  0.6, -0.4, -0.3, -0.3, -0.3, -0.5, -0.3,  0.3,  0.6],
    "plain_border":     [ 0.8,  0.2,  0.8, -0.3, -0.3, -0.3, -0.5,  0.0,  0.5,  0.0],
}
# Class offsets, set by hand from the two calibration checks.  With all offsets at zero, cat and
# dog took 59% of all predictions (their texture cues are on for most photos) while horse and
# plane were almost never predicted; with offsets of about one unit the picture inverted.  These
# values interpolate between those two observations so that each class is predicted about
# equally often.
#             plane  car  bird   cat  deer   dog  frog horse ship truck
CLASS_BIAS = [ 1.1,  0.6,  0.6, -0.45, 0.5, -0.4, 0.15,  0.5,  0.3,  0.0]

W_out = np.zeros((10, H))
for name, row in T.items():
    assert len(row) == 10, name
    row = np.array(row, dtype=np.float64)
    if name + "+" in UIDX:              # signed linear feature: +f and -f units
        W_out[:, UIDX[name + "+"]] = row
        W_out[:, UIDX[name + "-"]] = -row
    elif name in ramps:                 # saturating unit: relu(f-lo) - relu(f-hi)
        lo, hi = ramps[name]
        W_out[:, UIDX[lo]] = row
        W_out[:, UIDX[hi]] = -row
    else:
        W_out[:, UIDX[name]] = row
# Equal evidence budget: scale every class row to the same total absolute weight, so that
# classes with many correlated cues (sky, blue, water for plane and ship) cannot stack up more
# score than classes with few cues (car, truck).
row_l1 = np.abs(W_out).sum(axis=1)
W_out *= (row_l1.mean() / row_l1)[:, None]
b_out = np.array(CLASS_BIAS, dtype=np.float64)
for name, _, _ in units:
    base = name.rstrip("+-").replace("@lo", "").replace("@hi", "")
    assert base in T, f"unit {name} has no output row"

# ----------------------------------------------------------------------------------------
# 4. Write the network
# ----------------------------------------------------------------------------------------
layers = [
    {"name": "col", "op": "conv", "inputs": ["image"], "out": C, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "rcol", "op": "relu", "inputs": ["col"]},
    {"name": "ca", "op": "avgpool", "inputs": ["rcol"], "size": [8, 8], "stride": [8, 8]},
    {"name": "edg", "op": "conv", "inputs": ["image"], "out": E, "kernel": [K, K], "stride": 1, "padding": 0},
    {"name": "redg", "op": "relu", "inputs": ["edg"]},
    {"name": "ea", "op": "avgpool", "inputs": ["redg"], "size": [6, 6], "stride": [6, 6]},
    {"name": "er", "op": "maxpool", "inputs": ["redg"], "size": [6, 24], "stride": [6, 24]},
    {"name": "d_ca", "op": "dense", "inputs": ["ca"], "out": H},
    {"name": "d_ea", "op": "dense", "inputs": ["ea"], "out": H},
    {"name": "d_er", "op": "dense", "inputs": ["er"], "out": H},
    {"name": "hsum", "op": "add", "inputs": ["d_ca", "d_ea", "d_er"]},
    {"name": "h", "op": "relu", "inputs": ["hsum"]},
    {"name": "out", "op": "dense", "inputs": ["h"], "out": 10},
]

weights = {
    "col.W": W_col, "col.b": b_col,
    "edg.W": W_edg, "edg.b": b_edg,
    "d_ca.W": W_dense["ca"], "d_ca.b": b_hidden,      # the concept thresholds live in d_ca.b
    "d_ea.W": W_dense["ea"], "d_ea.b": np.zeros(H),
    "d_er.W": W_dense["er"], "d_er.b": np.zeros(H),
    "out.W": W_out, "out.b": b_out,
}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **{k: v.astype(np.float32) for k, v in weights.items()})
    total = sum(int(np.prod(v.shape)) for v in weights.values())
    print(f"wrote {out}/network.json and weights.npz: {H} concept units, {total} weights")
