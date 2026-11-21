"""Builds a hand-designed CIFAR-10-style classifier. Every weight comes from
formulas (Gaussians, derivative-of-Gaussians, difference-of-Gaussians, ring
templates) or from hand-written tables describing what the 10 classes look like
in general (sky is blue, grass is green, tyres are dark rings, deer carry
antlers, ...).  No data is read, nothing is fitted, no statistics of training
images or labels are used.

Pipeline:
  image -> opponent channels (intensity I, red-green A, blue-yellow Bb)
        -> Gaussian blur
        -> oriented derivative-of-Gaussian edges, 3 colour chans x 4 orients
           x 2 signs, avg- and max-pooled onto an 8x8 grid
        -> 3x3 fine-scale edges on I (fur/feather texture), avg-pooled 8x8
        -> DoG blob detectors at 3 scales x 2 signs (eyes, heads, bodies)
        -> annulus (ring) detectors at 2 radii x 2 signs (wheels/tyres)
        -> 4x4 block means of the blurred opponent channels (block colours)
  Scene branch: the 8x8 maps are concatenated channel-wise (1x1 conv channel
  placement + add) into a canvas; 26 hand-written predicates read regions of it
  (sky-blue top, green bottom, brown core, horizon, smooth core, ...).
  Presence branch: band maxima (top/mid/bottom thirds) of the max-pooled blob,
  ring and strong-edge maps; 19 thresholded predicates ("a wheel somewhere in
  the lower third", "a strong horizontal edge across the middle", ...).
  Scoring: deviation matching against hand-written class prototypes.
  score_k = 2 * sum_i r_i * p_ki * (x_i - pbar_i), where p_ki is the typical
  value of predicate i for class k (written from general knowledge), pbar_i its
  average over classes and r_i the reliability of that predicate.  All classes
  tie on a perfectly average image, so a class wins only where the image
  deviates from average in the way its prototype predicts; features that sit at
  their average level for every photo (generic busyness) move nothing.
"""
import json
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# filter construction (formulas only)
# ----------------------------------------------------------------------------


def g1(k, s):
    """normalised 1d gaussian"""
    x = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-x * x / (2.0 * s * s))
    return g / g.sum()


def d1(k, s):
    """1d derivative of gaussian, unit L1"""
    x = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-x * x / (2.0 * s * s))
    d = -x * g
    return d / np.abs(d).sum()


def l1(K):
    return K / np.abs(K).sum()


def oriented(k, s):
    """4 oriented derivative-of-gaussian kernels, unit L1 each.
    order: o0 = d/dx (vertical edges), o1 = 45deg, o2 = d/dy (horizontal edges),
    o3 = 135deg."""
    g, d = g1(k, s), d1(k, s)
    Kx = np.outer(g, d)
    Ky = np.outer(d, g)
    K45 = (Kx + Ky) / np.sqrt(2.0)
    K135 = (Ky - Kx) / np.sqrt(2.0)
    return [l1(K) for K in (Kx, K45, Ky, K135)]


def dog(k, sc, ss):
    """difference of gaussians (blob detector), zero DC, unit L1"""
    Gc = np.outer(g1(k, sc), g1(k, sc))
    Gs = np.outer(g1(k, ss), g1(k, ss))
    return l1(Gc - Gs)


def ring(k, r):
    """annulus template (wheel), zero DC, unit L1"""
    x = np.arange(k) - (k - 1) / 2.0
    X, Y = np.meshgrid(x, x)
    rho = np.hypot(X, Y)
    K = np.zeros((k, k))
    K[np.abs(rho - r) <= 1.15] = 1.0
    K[rho < r - 1.15] = -0.6
    K[(rho > r + 1.15) & (rho <= r + 3.5)] = -0.35
    K = K - K.mean()
    return l1(K)


# ----------------------------------------------------------------------------
# layer bookkeeping
# ----------------------------------------------------------------------------
L = []
W = {}
SH = {"image": (3, 32, 32)}  # channel counts of each tensor


def add_layer(name, op, inputs, out_shape, **settings):
    L.append({"name": name, "op": op, "inputs": inputs, **settings})
    SH[name] = out_shape


def conv(name, inp, k, out, pad=None):
    add_layer(name, "conv", [inp], (out, None, None), out=out, kernel=[k, k],
              stride=1, padding=(k // 2 if pad is None else pad))
    W[name + ".W"] = np.zeros((out, SH[inp][0], k, k))
    W[name + ".b"] = np.zeros(out)


def relu(name, inp):
    add_layer(name, "relu", [inp], SH[inp])


def pool(name, inp, kind, size, stride):
    add_layer(name, kind, [inp], (SH[inp][0], None, None), size=size, stride=stride)


# ----------------------------------------------------------------------------
# stage 1: opponent channels + blur
# ----------------------------------------------------------------------------
conv("opp", "image", 1, 3, pad=0)
W["opp.W"][0, :, 0, 0] = [1 / 3, 1 / 3, 1 / 3]      # I
W["opp.W"][1, :, 0, 0] = [1.0, -1.0, 0.0]           # A = R-G
W["opp.W"][2, :, 0, 0] = [-0.5, -0.5, 1.0]          # Bb = B - (R+G)/2

G5 = np.outer(g1(5, 1.0), g1(5, 1.0))
conv("blur", "opp", 5, 3)
for i in range(3):
    W["blur.W"][i, i] = G5

# ----------------------------------------------------------------------------
# stage 2: oriented edges (5x5 derivative of gaussian, sigma 1) on blurred opp
# ----------------------------------------------------------------------------
OR5 = oriented(5, 1.0)  # o0 vert, o1 45, o2 horiz, o3 135
conv("ged", "blur", 5, 24)
for c in range(3):          # 0=I 1=A 2=Bb
    for o in range(4):
        for s in range(2):  # 0=positive, 1=negative
            W["ged.W"][c * 8 + o * 2 + s, c] = (1.0 - 2.0 * s) * OR5[o]
relu("gedr", "ged")
pool("ec", "gedr", "avgpool", [4, 4], [4, 4])   # (24,8,8) edge energy per block
pool("em", "gedr", "maxpool", [4, 4], [4, 4])   # (24,8,8) strongest edge per block

# ----------------------------------------------------------------------------
# stage 3: fine 3x3 edges on intensity (texture: fur, feathers, water sparkle)
# ----------------------------------------------------------------------------
g3 = np.array([1.0, 2.0, 1.0]) / 4.0
d3 = np.array([-1.0, 0.0, 1.0]) / 2.0
S3 = [l1(K) for K in (np.outer(g3, d3),
                      (np.outer(g3, d3) + np.outer(d3, g3)) / np.sqrt(2),
                      np.outer(d3, g3),
                      (np.outer(d3, g3) - np.outer(g3, d3)) / np.sqrt(2))]
conv("sbl", "opp", 3, 8)
for o in range(4):
    for s in range(2):
        W["sbl.W"][o * 2 + s, 0] = (1.0 - 2.0 * s) * S3[o]
relu("sblr", "sbl")
pool("sc", "sblr", "avgpool", [4, 4], [4, 4])   # (8,8,8)

# ----------------------------------------------------------------------------
# stage 4: blob detectors (DoG) at 3 scales x 2 signs, on blurred intensity
# ----------------------------------------------------------------------------
DOGS = [(9, 1.0, 2.0), (13, 2.0, 4.0), (19, 3.5, 7.0)]
for j, (k, sc, ss) in enumerate(DOGS):
    K = dog(k, sc, ss)
    conv(f"dog{j}", "blur", k, 2)
    W[f"dog{j}.W"][0, 0] = K        # bright blob
    W[f"dog{j}.W"][1, 0] = -K       # dark blob
    relu(f"dog{j}r", f"dog{j}")
    pool(f"d{j}m", f"dog{j}r", "maxpool", [4, 4], [4, 4])   # (2,8,8)

# ----------------------------------------------------------------------------
# stage 5: ring detectors (wheels) at 2 radii x 2 signs
# ----------------------------------------------------------------------------
RINGS = [(7, 2.5), (11, 4.5)]
for j, (k, r) in enumerate(RINGS):
    K = ring(k, r)
    conv(f"rng{j}", "blur", k, 2)
    W[f"rng{j}.W"][0, 0] = K        # bright ring
    W[f"rng{j}.W"][1, 0] = -K       # dark ring (tyre)
    relu(f"rng{j}r", f"rng{j}")
    pool(f"r{j}m", f"rng{j}r", "maxpool", [4, 4], [4, 4])   # (2,8,8)

# ----------------------------------------------------------------------------
# stage 6: block colors (means of blurred opponent channels over 4x4 blocks),
# split into positive/negative saturation parts
# ----------------------------------------------------------------------------
pool("col", "blur", "avgpool", [4, 4], [4, 4])               # (3,8,8)
conv("colneg", "col", 1, 3, pad=0)
for i in range(3):
    W["colneg.W"][i, i, 0, 0] = -1.0
relu("colp", "col")      # (I, Ap, Bbp)
relu("coln", "colneg")   # (0, An, Bbn)

# ----------------------------------------------------------------------------
# canvas: place every branch into its own channels of one (71,8,8) tensor
# ----------------------------------------------------------------------------
NC = 71
# (source, target offset, list of source channels to place)
PLACES = [("ec", 0, range(24)), ("em", 24, range(24)), ("sc", 48, range(8)),
          ("d0m", 56, range(2)), ("d1m", 58, range(2)), ("d2m", 60, range(2)),
          ("r0m", 62, range(2)), ("r1m", 64, range(2)),
          ("colp", 66, range(3)),
          ("coln", 69, [1, 2])]   # coln ch0 is relu(-I) = 0, drop it
xnames = []
for src, off, chans in PLACES:
    xn = src + "X"
    conv(xn, src, 1, NC, pad=0)
    for t, i in enumerate(chans):
        W[xn + ".W"][off + t, i, 0, 0] = 1.0
    xnames.append(xn)
add_layer("canvas", "add", xnames, (NC, 8, 8))

# canvas channel indices
CH = {}
for c, cn in enumerate(["I", "A", "Bb"]):
    for o in range(4):
        for s, sn in enumerate(["p", "n"]):
            CH[f"ec_{cn}_o{o}_{sn}"] = c * 8 + o * 2 + s
            CH[f"em_{cn}_o{o}_{sn}"] = 24 + c * 8 + o * 2 + s
for o in range(4):
    for s, sn in enumerate(["p", "n"]):
        CH[f"sc_o{o}_{sn}"] = 48 + o * 2 + s
CH.update({"dSm_b": 56, "dSm_d": 57, "dMd_b": 58, "dMd_d": 59,
           "dLg_b": 60, "dLg_d": 61, "rSm_b": 62, "rSm_d": 63,
           "rLg_b": 64, "rLg_d": 65,
           "cI": 66, "cAp": 67, "cBp": 68, "cAn": 69, "cBn": 70})

# ----------------------------------------------------------------------------
# predicate layer: hand-written linear evidence scores over the canvas
# parts: (channel, row0,row1, col0,col1, coefficient); the row is the mean over
# the region times the coefficient. relu then keeps the positive part.
# ----------------------------------------------------------------------------
TOP, UP, MID, LOW, BOT = (0, 4), (0, 3), (3, 5), (4, 8), (5, 8)
CORE = (2, 6)
MIDC = (1, 7)
ALL = (0, 8)

PREDS = [
    # ---- background / colour (bounded, well calibrated: lean on these) ----
    ("skyblue_top", 0.0, [("cBp", *TOP, *ALL, 4.0)]),
    ("blue_corners", 0.0, [("cBp", 0, 2, 0, 2, 4.0), ("cBp", 0, 2, 6, 8, 4.0),
                           ("cBp", 6, 8, 0, 2, 4.0), ("cBp", 6, 8, 6, 8, 4.0)]),
    ("grassbot", 0.0, [("cAn", *LOW, *ALL, 4.0), ("cBn", *LOW, *ALL, 2.4)]),
    ("green_all", 0.0, [("cAn", *ALL, *ALL, 4.0), ("cBn", *ALL, *ALL, 2.4)]),
    ("green_core", 0.0, [("cAn", 2, 6, 2, 6, 4.0)]),
    ("warm_all", 0.0, [("cAp", *ALL, *ALL, 4.0), ("cBn", *ALL, *ALL, 2.0)]),
    ("brown_core", 0.0, [("cBn", 2, 6, 2, 6, 2.8), ("cAp", 2, 6, 2, 6, 2.0)]),
    ("brown_low", 0.0, [("cBn", *LOW, *MIDC, 4.0)]),
    ("red_core", -0.7, [("cAp", 2, 6, 2, 6, 4.0)]),
    ("yellow_core", 0.0, [("cAp", 2, 6, 2, 6, 1.2), ("cBn", 2, 6, 2, 6, 4.0)]),
    ("bright_core", 0.0, [("cI", 2, 6, 2, 6, 1.5)]),
    # relative brightness comparisons (scale-robust: relu of a difference)
    ("core_dark_rel", 0.0, [("cI", *ALL, *ALL, 1.5), ("cI", 2, 6, 2, 6, -1.5)]),
    ("core_bright_rel", 0.0, [("cI", 2, 6, 2, 6, 1.5), ("cI", *ALL, *ALL, -1.5)]),
    ("dark_bot_rel", 0.0, [("cI", *ALL, *ALL, 1.5), ("cI", *BOT, *ALL, -1.5)]),
    ("bright_top_rel", 0.0, [("cI", *TOP, *ALL, 1.5), ("cI", *ALL, *ALL, -1.5)]),
    # ---- edges (avg-pooled energy) ----
    ("vlow", 0.0, [("ec_I_o0_p", *LOW, *MIDC, 12.0), ("ec_I_o0_n", *LOW, *MIDC, 12.0)]),
    ("vmid", 0.0, [("ec_I_o0_p", 2, 6, 1, 7, 12.0), ("ec_I_o0_n", 2, 6, 1, 7, 12.0)]),
    ("vsides", 0.0, [("ec_I_o0_p", 2, 6, 0, 2, 12.0), ("ec_I_o0_n", 2, 6, 0, 2, 12.0),
                     ("ec_I_o0_p", 2, 6, 6, 8, 12.0), ("ec_I_o0_n", 2, 6, 6, 8, 12.0)]),
    ("hmid", 0.0, [("ec_I_o2_p", 3, 6, 0, 8, 12.0), ("ec_I_o2_n", 3, 6, 0, 8, 12.0)]),
    ("htop", 0.0, [("ec_I_o2_p", 0, 4, 0, 8, 12.0), ("ec_I_o2_n", 0, 4, 0, 8, 12.0)]),
    ("dtop", 0.0, [("ec_I_o1_p", 0, 4, 1, 7, 12.0), ("ec_I_o1_n", 0, 4, 1, 7, 12.0),
                   ("ec_I_o3_p", 0, 4, 1, 7, 12.0), ("ec_I_o3_n", 0, 4, 1, 7, 12.0)]),
    ("dmid", 0.0, [("ec_I_o1_p", 2, 6, 1, 7, 12.0), ("ec_I_o1_n", 2, 6, 1, 7, 12.0),
                   ("ec_I_o3_p", 2, 6, 1, 7, 12.0), ("ec_I_o3_n", 2, 6, 1, 7, 12.0)]),
    ("horizon", -0.55, [("ec_I_o2_p", 3, 5, 0, 8, 24.0), ("ec_I_o2_n", 3, 5, 0, 8, 24.0)]),
    # ---- fine texture ----
    # centre busier than the border (close-up fur/feathers)
    ("fine_contrast_core", 0.0, [(f"sc_o{o}_{s}", 2, 6, 2, 6, 12.0)
                                 for o in range(4) for s in "pn"]
                                + [(f"sc_o{o}_{s}", 0, 2, 0, 8, -12.0)
                                   for o in range(4) for s in "pn"]
                                + [(f"sc_o{o}_{s}", 6, 8, 0, 8, -12.0)
                                   for o in range(4) for s in "pn"]
                                + [(f"sc_o{o}_{s}", 2, 6, 0, 2, -12.0)
                                   for o in range(4) for s in "pn"]
                                + [(f"sc_o{o}_{s}", 2, 6, 6, 8, -12.0)
                                   for o in range(4) for s in "pn"]),
    # top smoother than the image average, core smoother than the image average
    ("tex_top_smooth", 0.0, [(f"sc_o{o}_{s}", *ALL, *ALL, 12.0)
                             for o in range(4) for s in "pn"]
                            + [(f"sc_o{o}_{s}", 0, 4, 0, 8, -12.0)
                               for o in range(4) for s in "pn"]),
    ("tex_core_smooth", 0.0, [(f"sc_o{o}_{s}", *ALL, *ALL, 12.0)
                              for o in range(4) for s in "pn"]
                             + [(f"sc_o{o}_{s}", 2, 6, 2, 6, -12.0)
                                for o in range(4) for s in "pn"]),
]

NPRED = len(PREDS)
CIN = NC * 64

featW = np.zeros((NPRED, CIN))
featb = np.zeros(NPRED)
for i, (name, bias, parts) in enumerate(PREDS):
    for ch, r0, r1, c0, c1, coef in parts:
        base = CH[ch] * 64
        n = (r1 - r0) * (c1 - c0)
        for r in range(r0, r1):
            for c in range(c0, c1):
                featW[i, base + r * 8 + c] += coef / n
    featb[i] = bias

# ----------------------------------------------------------------------------
# rule table: predicate -> class scores
# classes: 0 airplane 1 automobile 2 bird 3 cat 4 deer 5 dog 6 frog 7 horse
#          8 ship 9 truck
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# hand-written class prototypes: for every predicate, how strongly it shows in
# a typical image of each class, from general knowledge of the classes
# (sky is blue, grass is green, tyres are dark rings, fur is fine texture, ...).
# classes: 0 airplane 1 automobile 2 bird 3 cat 4 deer 5 dog 6 frog 7 horse
#          8 ship 9 truck
# ----------------------------------------------------------------------------
PROTO = {
    # ---- colour / scene (bounded, hand-reasoned levels) ----
    "skyblue_top":      [0.35, 0.10, 0.40, 0.03, 0.08, 0.03, 0.06, 0.08, 0.28, 0.10],
    "blue_corners":     [0.30, 0.04, 0.28, 0.02, 0.04, 0.02, 0.04, 0.04, 0.15, 0.04],
    "green_all":        [0.04, 0.06, 0.14, 0.07, 0.40, 0.07, 0.55, 0.40, 0.06, 0.06],
    "green_core":       [0.02, 0.02, 0.04, 0.04, 0.10, 0.04, 0.35, 0.10, 0.04, 0.02],
    "grassbot":         [0.03, 0.05, 0.12, 0.10, 0.50, 0.10, 0.35, 0.50, 0.03, 0.05],
    "brown_core":       [0.03, 0.06, 0.15, 0.20, 0.45, 0.35, 0.08, 0.45, 0.06, 0.06],
    "brown_low":        [0.02, 0.04, 0.10, 0.14, 0.35, 0.25, 0.08, 0.35, 0.08, 0.04],
    "warm_all":         [0.06, 0.10, 0.12, 0.35, 0.20, 0.30, 0.10, 0.20, 0.06, 0.10],
    "red_core":         [0.03, 0.25, 0.06, 0.01, 0.00, 0.01, 0.00, 0.00, 0.00, 0.15],
    "yellow_core":      [0.04, 0.08, 0.10, 0.01, 0.01, 0.01, 0.02, 0.01, 0.03, 0.12],
    "bright_core":      [0.30, 0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.22, 0.28],
    "core_dark_rel":    [0.04, 0.04, 0.08, 0.08, 0.08, 0.12, 0.08, 0.12, 0.08, 0.16],
    "core_bright_rel":  [0.16, 0.08, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.12, 0.16],
    "dark_bot_rel":     [0.02, 0.08, 0.04, 0.08, 0.08, 0.08, 0.12, 0.08, 0.25, 0.08],
    "bright_top_rel":   [0.22, 0.10, 0.14, 0.04, 0.06, 0.04, 0.04, 0.06, 0.18, 0.10],
    "horizon":          [0.06, 0.03, 0.01, 0.01, 0.03, 0.01, 0.01, 0.03, 0.30, 0.03],
    "tex_top_smooth":   [0.22, 0.06, 0.10, 0.01, 0.03, 0.01, 0.01, 0.03, 0.18, 0.06],
    "tex_core_smooth":  [0.14, 0.18, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.10, 0.18],
    "fine_contrast_core": [0.02, 0.02, 0.08, 0.18, 0.08, 0.15, 0.12, 0.08, 0.02, 0.02],
    # ---- scene edges ----
    "hmid":             [0.20, 0.30, 0.12, 0.15, 0.15, 0.15, 0.08, 0.15, 0.30, 0.30],
    "htop":             [0.20, 0.15, 0.08, 0.08, 0.08, 0.08, 0.04, 0.08, 0.12, 0.25],
    "dtop":             [0.08, 0.04, 0.25, 0.22, 0.36, 0.12, 0.08, 0.08, 0.04, 0.04],
    "dmid":             [0.04, 0.04, 0.20, 0.12, 0.12, 0.12, 0.08, 0.12, 0.04, 0.04],
    "vlow":             [0.08, 0.30, 0.08, 0.15, 0.38, 0.20, 0.12, 0.18, 0.08, 0.30],
    "vmid":             [0.08, 0.30, 0.08, 0.12, 0.15, 0.15, 0.08, 0.15, 0.18, 0.30],
    "vsides":           [0.20, 0.15, 0.08, 0.08, 0.08, 0.08, 0.04, 0.08, 0.25, 0.20],
    # ---- presence ----
    "wheels_bot":       [0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40],
    "wheels_mid":       [0.00, 0.22, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.18],
    "eyes_mid":         [0.04, 0.08, 0.30, 0.45, 0.18, 0.45, 0.50, 0.18, 0.04, 0.08],
    "eyes_top":         [0.02, 0.04, 0.20, 0.28, 0.12, 0.28, 0.45, 0.12, 0.02, 0.04],
    "bigdark_mid":      [0.12, 0.18, 0.04, 0.12, 0.25, 0.35, 0.04, 0.62, 0.35, 0.38],
    "bigbright_mid":    [0.28, 0.12, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.18, 0.28],
    "middark_mid":      [0.08, 0.16, 0.12, 0.25, 0.25, 0.45, 0.18, 0.46, 0.28, 0.28],
    "midbright_mid":    [0.12, 0.12, 0.12, 0.12, 0.08, 0.12, 0.08, 0.08, 0.12, 0.18],
    "smalldark_mid":    [0.04, 0.08, 0.18, 0.35, 0.18, 0.25, 0.22, 0.18, 0.04, 0.08],
    "darkblob_bot":     [0.02, 0.22, 0.02, 0.12, 0.18, 0.18, 0.18, 0.18, 0.38, 0.22],
    "bigbright_bot":    [0.04, 0.12, 0.02, 0.02, 0.08, 0.02, 0.04, 0.02, 0.14, 0.18],
    "vedges_bot":       [0.04, 0.35, 0.04, 0.20, 0.40, 0.28, 0.12, 0.24, 0.08, 0.35],
    "vedges_mid":       [0.08, 0.30, 0.08, 0.12, 0.18, 0.18, 0.08, 0.18, 0.22, 0.38],
    "hedges_mid":       [0.30, 0.35, 0.08, 0.12, 0.12, 0.12, 0.08, 0.12, 0.45, 0.35],
    "hedges_top":       [0.25, 0.15, 0.08, 0.08, 0.06, 0.08, 0.04, 0.06, 0.12, 0.25],
    "dedges_top":       [0.16, 0.04, 0.25, 0.25, 0.34, 0.16, 0.08, 0.08, 0.04, 0.04],
    "dedges_mid":       [0.08, 0.04, 0.20, 0.12, 0.12, 0.12, 0.08, 0.12, 0.04, 0.04],
    "cedges_mid":       [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.28, 0.08],
    "ringbright_bot":   [0.00, 0.12, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.08],
}

# reliability: 1 = bounded colour statistic with a trusted scale, lower = template
# response whose scale on real photos is less certain
REL = {
    "skyblue_top": 1.0, "blue_corners": 1.0, "green_all": 1.0, "green_core": 0.9,
    "grassbot": 1.0, "brown_core": 1.0, "brown_low": 1.0, "warm_all": 0.6,
    "red_core": 0.9, "yellow_core": 0.9, "bright_core": 1.0,
    "core_dark_rel": 1.0, "core_bright_rel": 1.0, "dark_bot_rel": 1.0,
    "bright_top_rel": 1.0, "horizon": 0.8,
    "tex_top_smooth": 0.8, "tex_core_smooth": 0.8, "fine_contrast_core": 0.6,
    "hmid": 0.6, "htop": 0.6, "dtop": 0.6, "dmid": 0.5, "vlow": 0.6,
    "vmid": 0.6, "vsides": 0.6,
    "wheels_bot": 0.8, "wheels_mid": 0.7, "eyes_mid": 0.6, "eyes_top": 0.6,
    "bigdark_mid": 0.6, "bigbright_mid": 0.6, "middark_mid": 0.5,
    "midbright_mid": 0.3, "smalldark_mid": 0.5, "darkblob_bot": 0.5,
    "bigbright_bot": 0.3, "vedges_bot": 0.6, "vedges_mid": 0.6,
    "hedges_mid": 0.7, "hedges_top": 0.6, "dedges_top": 0.6, "dedges_mid": 0.4,
    "cedges_mid": 0.6, "ringbright_bot": 0.3,
}


def centroid_weights(names):
    """Deviation matching: score_k = 2 * sum_i r_i p_ki (x_i - pbar_i).
    Every class ties on the average image, so a class can only win through
    the ways this image deviates from a typical one, in the direction its
    prototype predicts.  Features sitting at their average level (e.g. the
    generic busyness of every photo) move no class at all."""
    Wout = np.zeros((10, len(names)))
    bout = np.zeros(10)
    for j, name in enumerate(names):
        p = np.array(PROTO[name], dtype=np.float64)
        r = REL[name]
        pbar = p.mean()
        Wout[:, j] = 2.0 * r * p
        bout -= 2.0 * r * p * pbar
    return Wout, bout


scene_names = [p[0] for p in PREDS]
for n in scene_names:
    assert n in PROTO and n in REL, f"scene predicate {n} has no prototype"
outW, outb = centroid_weights(scene_names)

add_layer("feat", "dense", ["canvas"], (NPRED,), out=NPRED)
W["feat.W"] = featW
W["feat.b"] = featb
relu("featr", "feat")
add_layer("scoreA", "dense", ["featr"], (10,), out=10)
W["scoreA.W"] = outW
W["scoreA.b"] = outb

# ----------------------------------------------------------------------------
# presence branch: band maxima (top = rows 0-3, mid = rows 2-5, bottom = rows 4-7)
# of the per-block maxpooled ring / blob / strong-edge maps.  A max over a band
# is not diluted, so "there is a wheel somewhere in the lower half" survives.
# ----------------------------------------------------------------------------
for nm, src in [("bR0", "r0m"), ("bR1", "r1m"), ("bD0", "d0m"),
                ("bD1", "d1m"), ("bD2", "d2m"), ("bE", "em")]:
    pool(nm, src, "maxpool", [4, 8], [2, 8])          # (C, 3, 1)

NCP = 34
POFF = [("bR0", 0, 2), ("bR1", 2, 2), ("bD0", 4, 2), ("bD1", 6, 2),
        ("bD2", 8, 2), ("bE", 10, 24)]
pnames = []
for src, off, cin in POFF:
    xn = src + "X"
    conv(xn, src, 1, NCP, pad=0)
    for i in range(cin):
        W[xn + ".W"][off + i, i, 0, 0] = 1.0
    pnames.append(xn)
add_layer("pres", "add", pnames, (NCP, 3, 1))

PCH = {"rSm_b": 0, "rSm_d": 1, "rLg_b": 2, "rLg_d": 3, "dSm_b": 4, "dSm_d": 5,
       "dMd_b": 6, "dMd_d": 7, "dLg_b": 8, "dLg_d": 9}
for c, cn in enumerate(["I", "A", "Bb"]):
    for o in range(4):
        for s, sn in enumerate(["p", "n"]):
            PCH[f"em_{cn}_o{o}_{sn}"] = 10 + c * 8 + o * 2 + s

# (name, bias, [(channel, band, coef)]); band 0 top, 1 mid, 2 bottom.
# thresholds calibrated analytically: a matched blob/ring of contrast 0.3-0.5
# responds ~0.1-0.2 after unit-L1 normalisation, clutter ~0.03-0.08.
PREDS_P = [
    ("wheels_bot", -0.7, [("rSm_d", 2, 8.0), ("rLg_d", 2, 4.0),
                        ("dSm_d", 2, 3.0), ("dMd_d", 2, 2.0)]),
    ("wheels_mid", -0.55, [("rSm_d", 1, 6.0)]),
    ("eyes_mid", -1.2, [("dSm_b", 1, 8.0)]),
    ("eyes_top", -0.9, [("dSm_b", 0, 6.0)]),
    ("bigdark_mid", -0.55, [("dLg_d", 1, 5.0)]),
    ("bigbright_mid", -0.55, [("dLg_b", 1, 5.0)]),
    ("middark_mid", -0.6, [("dMd_d", 1, 5.0)]),
    ("midbright_mid", -0.6, [("dMd_b", 1, 5.0)]),
    ("smalldark_mid", -0.7, [("dSm_d", 1, 5.0)]),
    ("darkblob_bot", -0.6, [("dMd_d", 2, 5.0)]),
    ("bigbright_bot", -0.5, [("dLg_b", 2, 4.0)]),
    ("vedges_bot", -0.8, [("em_I_o0_p", 2, 4.0), ("em_I_o0_n", 2, 4.0)]),
    ("vedges_mid", -0.8, [("em_I_o0_p", 1, 4.0), ("em_I_o0_n", 1, 4.0)]),
    ("hedges_mid", -0.8, [("em_I_o2_p", 1, 4.0), ("em_I_o2_n", 1, 4.0)]),
    ("hedges_top", -0.8, [("em_I_o2_p", 0, 4.0), ("em_I_o2_n", 0, 4.0)]),
    ("dedges_top", -0.8, [("em_I_o1_p", 0, 4.0), ("em_I_o1_n", 0, 4.0),
                          ("em_I_o3_p", 0, 4.0), ("em_I_o3_n", 0, 4.0)]),
    ("dedges_mid", -0.8, [("em_I_o1_p", 1, 4.0), ("em_I_o1_n", 1, 4.0),
                          ("em_I_o3_p", 1, 4.0), ("em_I_o3_n", 1, 4.0)]),
    ("cedges_mid", -0.7, [("em_A_o2_p", 1, 3.0), ("em_A_o2_n", 1, 3.0),
                          ("em_Bb_o2_p", 1, 3.0), ("em_Bb_o2_n", 1, 3.0)]),
    ("ringbright_bot", -0.9, [("rSm_b", 2, 6.0)]),
]
NPP = len(PREDS_P)

featPW = np.zeros((NPP, NCP * 3))
featPb = np.zeros(NPP)
for i, (name, bias, parts) in enumerate(PREDS_P):
    for ch, band, coef in parts:
        featPW[i, PCH[ch] * 3 + band] += coef
    featPb[i] = bias

pres_names = [p[0] for p in PREDS_P]
for n in pres_names:
    assert n in PROTO and n in REL, f"presence predicate {n} has no prototype"
outPW, outPb = centroid_weights(pres_names)

add_layer("featP", "dense", ["pres"], (NPP,), out=NPP)
W["featP.W"] = featPW
W["featP.b"] = featPb
relu("featPr", "featP")
add_layer("scoreP", "dense", ["featPr"], (10,), out=10)
W["scoreP.W"] = outPW
W["scoreP.b"] = outPb

# final scores = scene branch + presence branch
add_layer("scores", "add", ["scoreA", "scoreP"], (10,))

# ----------------------------------------------------------------------------
# write answer files
# ----------------------------------------------------------------------------
def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/out")
    out.mkdir(parents=True, exist_ok=True)
    (out / "network.json").write_text(json.dumps({"layers": L}))
    np.savez(out / "weights.npz", **{k: v.astype(np.float32) for k, v in W.items()})
    print(f"wrote {out}/network.json and {out}/weights.npz "
          f"({sum(int(np.prod(v.shape)) for v in W.values()):,} weights, "
          f"{NPRED} scene + {NPP} presence predicates)")


if __name__ == "__main__":
    main()
