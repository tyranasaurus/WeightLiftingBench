#!/usr/bin/env python3
"""
Hand-designed CIFAR-10 classifier: every weight below is written from general knowledge of
what the ten classes look like (colours, layout, texture, long edges, wheels, legs...).
No training data is read; no statistic of any image is used.  Run:  python3 build.py OUT_DIR
"""
import sys, json
import numpy as np
from pathlib import Path

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out_dir.mkdir(parents=True, exist_ok=True)

layers, weights = [], {}


def add_layer(name, op, inputs, **kw):
    layers.append(dict(name=name, op=op, inputs=list(inputs), **kw))
    return name


def conv(name, inp, W, b, stride=1, padding=0):
    W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
    weights[name + ".W"], weights[name + ".b"] = W, b
    return add_layer(name, "conv", [inp], out=int(W.shape[0]), kernel=[int(W.shape[2]), int(W.shape[3])],
                     stride=stride, padding=padding)


def dense(name, inp, W, b):
    W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
    weights[name + ".W"], weights[name + ".b"] = W, b
    return add_layer(name, "dense", [inp], out=int(W.shape[0]))


def relu(name, inp):
    return add_layer(name, "relu", [inp])


# ----------------------------------------------------------------------------------------------
# 1. Pixel colour primitives (1x1 conv + ReLU).  Input channels: R, G, B in [0, 1].
# ----------------------------------------------------------------------------------------------
P1 = ["L", "rg+", "rg-", "yb+", "yb-", "gm+", "Lhi", "Llo", "w", "d", "red", "bg+"]
W1 = np.zeros((len(P1), 3, 1, 1), np.float32); b1 = np.zeros(len(P1), np.float32)
third = 1.0 / 3.0
rows = {
    "L":   ([third, third, third], 0.0),      # brightness
    "rg+": ([1, -1, 0], 0.0),                 # red minus green  (red / brown / orange)
    "rg-": ([-1, 1, 0], 0.0),                 # green minus red
    "yb+": ([0.5, 0.5, -1], 0.0),             # yellow minus blue (warm)
    "yb-": ([-0.5, -0.5, 1], 0.0),            # blue minus yellow (blue)
    "gm+": ([-0.5, 1, -0.5], 0.0),            # green minus magenta (grass, foliage)
    "Lhi": ([third, third, third], -0.5),     # brightness above mid grey
    "Llo": ([-third, -third, -third], 0.5),   # brightness below mid grey
    "w":   ([third, third, third], -0.65),    # white-ish
    "d":   ([-third, -third, -third], 0.25),  # dark
    "red": ([1, -1, 0], -0.3),                # strongly red
    "bg+": ([0, -1, 1], 0.0),                 # blue minus green (purple / pink)
}
for i, n in enumerate(P1):
    W1[i, :, 0, 0] = rows[n][0]; b1[i] = rows[n][1]
p1 = relu("p1r", conv("p1", "image", W1, b1))

# ----------------------------------------------------------------------------------------------
# 2. Colour palette (1x1 conv + ReLU over the primitives).  Each channel ~ 1 when clearly present.
# ----------------------------------------------------------------------------------------------
P2 = ["L", "BLUE", "GREEN", "BROWN", "RED", "WHITE", "DARK", "GREY", "CHROMA", "SKYBLUE", "WATER", "WARM", "VIVID"]
W2 = np.zeros((len(P2), len(P1), 1, 1), np.float32); b2 = np.zeros(len(P2), np.float32)


def p2(name, bias=0.0, **terms):
    i = P2.index(name); b2[i] = bias
    for k, v in terms.items():
        W2[i, P1.index(k.replace("_p", "+").replace("_m", "-")), 0, 0] = v


p2("L", L=1.0)
p2("BLUE", yb_m=3.0)                                   # sky / water blue ~ 1
p2("GREEN", gm_p=4.0)                                  # grass ~ 1
p2("BROWN", rg_p=4.0, red=-8.0, bg_p=-4.0)             # brown / tan / orange, not pure red, not pink
p2("RED", red=3.0)                                     # saturated red (vehicles)
p2("WHITE", w=3.0)                                     # bright
p2("DARK", d=4.0)                                      # near black
p2("GREY", bias=1.2, Lhi=-3.0, Llo=-3.0, rg_p=-4.0, rg_m=-4.0, yb_p=-4.0, yb_m=-4.0)  # mid grey, low colour (asphalt, concrete, fur)
p2("CHROMA", rg_p=2.0, rg_m=2.0, yb_p=2.0, yb_m=2.0)   # colourfulness
p2("SKYBLUE", bias=-1.2, yb_m=3.0, L=1.5)              # bright blue
p2("WATER", bias=0.2, yb_m=6.0, L=-4.0)                # darker blue (zero for black / grey)
p2("WARM", yb_p=2.0, gm_p=-2.0)                        # warm but not green
p2("VIVID", bias=-1.0, rg_p=2.0, rg_m=2.0, yb_p=2.0, gm_p=-2.0)  # saturated paint: red, orange, yellow, magenta
pal = relu("palr", conv("pal", p1, W2, b2))

# ----------------------------------------------------------------------------------------------
# 3. Texture: Sobel derivatives of brightness (3x3 conv on the L channel of p1), both signs.
# ----------------------------------------------------------------------------------------------
gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0
gy = gx.T.copy()
TEX = ["gx+", "gx-", "gy+", "gy-"]
W3 = np.zeros((4, len(P1), 3, 3), np.float32)
W3[0, 0], W3[1, 0], W3[2, 0], W3[3, 0] = gx, -gx, gy, -gy
tex = relu("texr", conv("tex", p1, W3, np.zeros(4), padding=1))

# ----------------------------------------------------------------------------------------------
# 4. Structure detectors on per-image normalised brightness (zero mean, unit variance).
# ----------------------------------------------------------------------------------------------
lum = conv("lum", "image", np.array([[[[third]], [[third]], [[third]]]], np.float32), np.zeros(1))
nrm = add_layer("nrm", "norm", [lum])
DET = ["HLb", "HLd", "VLa", "VLb", "BAR", "BLOB"]
K = 9
W4 = np.zeros((len(DET), 1, K, K), np.float32)
hl = np.zeros((K, K), np.float32); hl[0:4, :] = 1.0 / 36; hl[5:9, :] = -1.0 / 36   # bright above, dark below
W4[0, 0] = hl; W4[1, 0] = -hl
W4[2, 0] = hl.T; W4[3, 0] = -hl.T                                                  # long vertical edges
bar = np.zeros((K, K), np.float32)
bar[1:8, 3:6] = -1.0 / 21; bar[1:8, 1:3] = 1.0 / 28; bar[1:8, 6:8] = 1.0 / 28      # dark vertical bar (legs, posts)
W4[4, 0] = bar
blob = np.zeros((K, K), np.float32)
blob[:, :] = 1.0 / 56; blob[2:7, 2:7] = -1.0 / 25                                  # dark 5x5 blob on lighter ring (wheels, eyes)
W4[5, 0] = blob
det = relu("detr", conv("det", nrm, W4, np.zeros(len(DET)), padding=4))

# ----------------------------------------------------------------------------------------------
# 5. Spatial pooling to a 4x4 grid of 8x8 cells.
# ----------------------------------------------------------------------------------------------
A = add_layer("A", "avgpool", [pal], size=[8, 8], stride=[8, 8])   # (12, 4, 4) colour fractions
T = add_layer("T", "avgpool", [tex], size=[8, 8], stride=[8, 8])   # (4, 4, 4)  mean edge energy
M = add_layer("M", "maxpool", [tex], size=[8, 8], stride=[8, 8])   # (4, 4, 4)  strongest edge
D = add_layer("D", "maxpool", [det], size=[8, 8], stride=[8, 8])   # (6, 4, 4)  strongest detector hit

# ----------------------------------------------------------------------------------------------
# 6. Concept layer: hand-set dense weights over the pooled grids, then ReLU.
#    Each concept carries my estimate of its typical level on an ordinary photo (mu) and of its spread
#    between photos (sd).  These come from the formulas and from general properties of small natural
#    photos (sky-like pixels are present in maybe a third of photos and then cover the top; the mean
#    brightness gradient is ~0.06 per pixel; block-mean contrasts of a unit-variance image are similar
#    everywhere; ...).  They are not measured from any image.  Rule weights are in units of sd.
# ----------------------------------------------------------------------------------------------
cells = [(r, c) for r in range(4) for c in range(4)]
REG = {
    "ALL": cells,
    "TOP": [(0, c) for c in range(4)],
    "TOP2": [(r, c) for r in (0, 1) for c in range(4)],
    "MID": [(r, c) for r in (1, 2) for c in range(4)],
    "BOT": [(3, c) for c in range(4)],
    "BOT2": [(r, c) for r in (2, 3) for c in range(4)],
    "CEN": [(r, c) for r in (1, 2) for c in (1, 2)],
    "LEFT": [(r, c) for r in range(4) for c in (0, 1)],
    "RIGHT": [(r, c) for r in range(4) for c in (2, 3)],
}
TEX_SCALE = 4.0   # mean |gradient| ~0.2 per pixel (fur, foliage) -> about 0.8; smooth sky (~0.03) -> about 0.1
HL, VL = ["HLb", "HLd"], ["VLa", "VLb"]

concepts, PRIOR = [], {}   # concept = (name, bias, terms); term = (tensor, channel(s), region, weight): weight * mean over region


def C(name, *terms, bias=0.0, mu=0.0, sd=1.0):
    concepts.append((name, bias, terms)); PRIOR[name] = (mu, sd)


# --- colour layout (sparse: typical level small, large when present) ---------------------------
for reg, mu, sd in (("TOP", 0.50, 0.18), ("CEN", 0.45, 0.15), ("BOT", 0.42, 0.15), ("ALL", 0.45, 0.13)):
    C("L_" + reg, ("A", "L", reg, 1.0), mu=mu, sd=sd)
for reg, mu, sd in (("TOP", 0.10, 0.35), ("MID", 0.03, 0.20), ("BOT", 0.02, 0.20), ("ALL", 0.06, 0.22)):
    C("blue_" + reg, ("A", "BLUE", reg, 1.0), mu=mu, sd=sd)
for reg, mu, sd in (("TOP", 0.04, 0.20), ("MID", 0.05, 0.20), ("BOT", 0.08, 0.25), ("ALL", 0.06, 0.18), ("CEN", 0.04, 0.18)):
    C("green_" + reg, ("A", "GREEN", reg, 1.0), mu=mu, sd=sd)
for reg, mu, sd in (("CEN", 0.15, 0.20), ("ALL", 0.12, 0.15), ("BOT", 0.12, 0.17)):
    C("brown_" + reg, ("A", "BROWN", reg, 1.0), mu=mu, sd=sd)
for reg, mu, sd in (("CEN", 0.15, 0.18), ("ALL", 0.12, 0.14)):
    C("warm_" + reg, ("A", "WARM", reg, 1.0), mu=mu, sd=sd)
for reg, mu, sd in (("CEN", 0.0, 0.12), ("ALL", 0.0, 0.08), ("TOP", 0.0, 0.10)):
    C("red_" + reg, ("A", "RED", reg, 1.0), mu=mu, sd=sd)
for reg, mu, sd in (("CEN", 0.05, 0.18), ("TOP", 0.10, 0.25), ("BOT", 0.04, 0.15), ("ALL", 0.06, 0.15)):
    C("white_" + reg, ("A", "WHITE", reg, 1.0), mu=mu, sd=sd)
for reg, mu, sd in (("CEN", 0.05, 0.15), ("TOP", 0.03, 0.12), ("BOT", 0.06, 0.17), ("ALL", 0.05, 0.12)):
    C("dark_" + reg, ("A", "DARK", reg, 1.0), mu=mu, sd=sd)
for reg, mu, sd in (("CEN", 0.20, 0.20), ("BOT", 0.25, 0.25), ("ALL", 0.22, 0.18)):
    C("grey_" + reg, ("A", "GREY", reg, 1.0), mu=mu, sd=sd)
for reg, mu, sd in (("CEN", 0.55, 0.30), ("TOP", 0.55, 0.30), ("ALL", 0.55, 0.25)):
    C("chroma_" + reg, ("A", "CHROMA", reg, 1.0), mu=mu, sd=sd)
# sky: bright, blue or grey/white, and smooth
for reg, mu, sd, texw in (("TOP", 0.05, 0.30, -1.5), ("ALL", 0.0, 0.15, -2.0)):
    C("sky_" + reg, ("A", "SKYBLUE", reg, 1.0), ("A", "WHITE", reg, 0.8), ("A", "GREY", reg, 0.4),
      ("T", TEX, reg, texw * TEX_SCALE), bias=-0.1, mu=mu, sd=sd)
C("water_BOT", ("A", "WATER", "BOT", 1.5), bias=-0.05, mu=0.0, sd=0.15)      # dark blue only: sky-blue at the bottom does not count
C("water_BOT2", ("A", "WATER", "BOT2", 1.5), bias=-0.05, mu=0.0, sd=0.13)
C("colour_TOP", ("A", "CHROMA", "TOP", 1.0), ("A", "BLUE", "TOP", -1.5), ("A", "GREEN", "TOP", -1.5), mu=0.10, sd=0.25)
for reg, mu, sd in (("CEN", 0.02, 0.10), ("ALL", 0.02, 0.08)):
    C("vivid_" + reg, ("A", "VIVID", reg, 1.0), mu=mu, sd=sd)
# paved ground: grey or pale, not grass, soil or water
C("road_BOT", ("A", "GREY", "BOT", 1.0), ("A", "WHITE", "BOT", 0.5), ("A", "GREEN", "BOT", -1.5), ("A", "BROWN", "BOT", -1.5),
  ("A", "WATER", "BOT", -1.5), mu=0.15, sd=0.25)
C("top_brighter", ("A", "L", "TOP", 1.0), ("A", "L", "BOT", -1.0), mu=0.12, sd=0.12)   # sky above ground / water
C("bot_brighter", ("A", "L", "BOT", 1.0), ("A", "L", "TOP", -1.0), mu=0.0, sd=0.08)
# --- texture -----------------------------------------------------------------------------------
for reg, mu, sd in (("ALL", 0.50, 0.20), ("CEN", 0.55, 0.22), ("TOP", 0.40, 0.22), ("BOT", 0.50, 0.22)):
    C("tex_" + reg, ("T", TEX, reg, TEX_SCALE), mu=mu, sd=sd)
for reg, sd in (("TOP", 0.25), ("BOT", 0.15), ("ALL", 0.10)):
    C("smooth_" + reg, ("T", TEX, reg, -3.0 * TEX_SCALE), bias=3.0 * 0.35, mu=0.0, sd=sd)
C("hdom", ("T", ["gy+", "gy-"], "ALL", TEX_SCALE), ("T", ["gx+", "gx-"], "ALL", -TEX_SCALE), mu=0.0, sd=0.08)  # horizontal edges dominate
C("vdom", ("T", ["gx+", "gx-"], "ALL", TEX_SCALE), ("T", ["gy+", "gy-"], "ALL", -TEX_SCALE), mu=0.0, sd=0.08)  # vertical edges dominate
# fur / foliage: many medium gradients (mean high relative to the strongest edge in each cell);
# vehicles, sky: a few strong edges on smooth surfaces (strongest edge high relative to the mean)
for reg, mu, sd in (("ALL", 0.08, 0.30), ("CEN", 0.08, 0.32)):
    C("fur_" + reg, ("T", TEX, reg, 8.0), ("M", TEX, reg, -0.8), mu=mu, sd=sd)
C("peaky_ALL", ("M", TEX, "ALL", 0.8), ("T", TEX, "ALL", -8.0), mu=0.05, sd=0.25)
C("peaky_CEN", ("M", TEX, "CEN", 0.8), ("T", TEX, "CEN", -8.0), mu=0.05, sd=0.28)
# --- structure detectors, as contrasts within the image (typical level near zero by symmetry) ----
C("hl_bright_above_MID", ("D", "HLb", "MID", 1.0), ("D", "HLd", "MID", -1.0), mu=0.0, sd=0.4)   # sky over ground / water
C("hl_dark_above_BOT", ("D", "HLd", "BOT", 1.0), ("D", "HLb", "BOT", -1.0), mu=0.1, sd=0.4)     # body / shadow over bright road
C("hlines_dom", ("D", HL, "ALL", 1.0), ("D", VL, "ALL", -1.0), mu=0.0, sd=0.5)                   # long horizontal structure
C("vlines_dom", ("D", VL, "ALL", 1.0), ("D", HL, "ALL", -1.0), mu=0.0, sd=0.5)                   # long vertical structure
C("hline_MID_vs_TOP", ("D", HL, "MID", 1.0), ("D", HL, "TOP", -1.0), mu=0.2, sd=0.5)
C("blob_BOT_vs_TOP", ("D", "BLOB", "BOT", 1.0), ("D", "BLOB", "TOP", -1.0), mu=0.2, sd=0.4)      # wheels
C("bar_BOT2_vs_TOP2", ("D", "BAR", "BOT2", 1.0), ("D", "BAR", "TOP2", -1.0), mu=0.1, sd=0.4)     # legs
# --- left/right asymmetry (side views vs frontal faces) ---------------------------------------
for ch in ("L", "DARK", "BROWN"):
    C("asymL_" + ch, ("A", ch, "LEFT", 1.0), ("A", ch, "RIGHT", -1.0), mu=0.02, sd=0.05)
    C("asymR_" + ch, ("A", ch, "RIGHT", 1.0), ("A", ch, "LEFT", -1.0), mu=0.02, sd=0.05)

CH = {"A": P2, "T": TEX, "M": TEX, "D": DET}
NC = len(concepts)
WC = {k: np.zeros((NC, len(CH[k]) * 16), np.float32) for k in CH}
bC = np.zeros(NC, np.float32)
for i, (name, bias, terms) in enumerate(concepts):
    bC[i] = bias
    for tensor, chs, reg, w in terms:
        if isinstance(chs, str):
            chs = [chs]
        for ch in chs:
            ci = CH[tensor].index(ch)
            for (r, c) in REG[reg]:
                WC[tensor][i, ci * 16 + r * 4 + c] += w / len(REG[reg])
cA = dense("cA", A, WC["A"], bC)
cT = dense("cT", T, WC["T"], np.zeros(NC))
cM = dense("cM", M, WC["M"], np.zeros(NC))
cD = dense("cD", D, WC["D"], np.zeros(NC))
csum = add_layer("csum", "add", [cA, cT, cM, cD])
con = relu("con", csum)

# ----------------------------------------------------------------------------------------------
# 7. Rule table: evidence for each class, in units of sd of each concept (hand-written from
#    knowledge of what the classes look like).
# ----------------------------------------------------------------------------------------------
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
asym_all = ["asymL_L", "asymR_L", "asymL_DARK", "asymR_DARK", "asymL_BROWN", "asymR_BROWN"]
# cues shared by the two vehicle classes: paved road and wheels below, smooth paint with sharp edges, vivid or red paint, no fur
VEHICLE = dict(road_BOT=0.6, grey_BOT=0.2, dark_BOT=0.3, blob_BOT_vs_TOP=0.4, hl_dark_above_BOT=0.3, peaky_ALL=0.6, peaky_CEN=0.5,
               vivid_CEN=0.5, vivid_ALL=0.3, red_CEN=0.4, red_ALL=0.2, white_CEN=0.3, chroma_CEN=0.3,
               fur_ALL=-0.7, brown_CEN=-0.5, warm_CEN=-0.3, green_ALL=-0.4, water_BOT=-0.4, bar_BOT2_vs_TOP2=-0.2, sky_ALL=-0.3)
rules = {
    # sky all around, bright and smooth, grey/white object; little green, brown or fur texture
    "airplane": dict(sky_ALL=1.0, sky_TOP=0.6, smooth_ALL=0.8, smooth_BOT=0.4, smooth_TOP=0.2, blue_ALL=0.4, blue_BOT=0.3,
                     white_ALL=0.3, grey_ALL=0.2, L_ALL=0.3, L_BOT=0.3, hlines_dom=0.2, peaky_ALL=0.2,
                     green_ALL=-0.8, brown_ALL=-0.6, brown_CEN=-0.4, warm_ALL=-0.4, warm_CEN=-0.3, tex_ALL=-0.6, fur_ALL=-0.6,
                     fur_CEN=-0.4, red_ALL=-0.3, water_BOT=-0.5, dark_BOT=-0.4, blob_BOT_vs_TOP=-0.2, vlines_dom=-0.5,
                     bar_BOT2_vs_TOP2=-0.2, top_brighter=-0.5, vivid_ALL=-0.2, road_BOT=-0.2),
    # dark-blue water below (and often around), brighter sky above; hull dark or white in the middle
    "ship": dict(water_BOT=1.2, water_BOT2=0.6, blue_BOT=0.3, blue_MID=0.3, blue_ALL=0.3, top_brighter=0.4, sky_TOP=0.2,
                 hl_bright_above_MID=0.3, hlines_dom=0.2, white_CEN=0.2, dark_CEN=0.1,
                 green_ALL=-0.8, brown_ALL=-0.6, warm_ALL=-0.4, fur_ALL=-0.5, tex_ALL=-0.2, road_BOT=-0.5, grey_BOT=-0.2,
                 vivid_CEN=-0.3, blob_BOT_vs_TOP=-0.3, bar_BOT2_vs_TOP2=-0.2, red_CEN=-0.1, smooth_BOT=-0.2),
    # vehicle low in the frame with background (sky, trees, buildings) above; long horizontal body lines
    "automobile": dict(VEHICLE, hlines_dom=0.4, hdom=0.3, L_BOT=0.2, sky_TOP=0.2, smooth_TOP=0.2, green_TOP=0.2,
                       colour_TOP=-0.3, white_TOP=-0.2, tex_TOP=-0.2, vlines_dom=-0.2, green_BOT=-0.2, blue_BOT=-0.3),
    # tall boxy vehicle whose painted body reaches the top of the frame; long vertical edges
    "truck": dict(VEHICLE, vlines_dom=0.5, vdom=0.3, colour_TOP=0.5, white_TOP=0.4, red_TOP=0.4, tex_TOP=0.3, dark_TOP=0.2,
                  chroma_TOP=0.2, hlines_dom=-0.1, smooth_TOP=-0.2, blue_BOT=-0.3),
    # small brown/grey textured body against sky, blurred green or branches
    "bird": dict(brown_CEN=0.4, warm_CEN=0.2, sky_TOP=0.3, sky_ALL=0.2, blue_ALL=0.3, green_ALL=0.3, smooth_TOP=0.4,
                 smooth_BOT=0.2, smooth_ALL=0.3, fur_CEN=0.3, white_CEN=0.1, grey_ALL=0.1, tex_TOP=-0.3,
                 red_CEN=-0.3, grey_BOT=-0.3, blob_BOT_vs_TOP=-0.2, vlines_dom=-0.2, hl_dark_above_BOT=-0.3, peaky_ALL=-0.3,
                 dark_ALL=-0.1, chroma_CEN=-0.1, tex_ALL=-0.1, road_BOT=-0.3, vivid_CEN=-0.2),
    # fur texture everywhere, grey/brown, frontal symmetric face, indoors close-up (no sky, no horizon, no water)
    "cat": dict(fur_ALL=0.6, fur_CEN=0.4, tex_ALL=0.3, grey_ALL=0.4, grey_CEN=0.4, brown_CEN=0.3, warm_ALL=0.3, white_CEN=0.1,
                dark_CEN=0.2,
                sky_TOP=-0.6, sky_ALL=-0.4, blue_ALL=-0.6, green_ALL=-0.3, hlines_dom=-0.2, vlines_dom=-0.2, water_BOT=-0.4,
                red_CEN=-0.2, smooth_ALL=-0.3, smooth_TOP=-0.2, peaky_ALL=-0.4, bar_BOT2_vs_TOP2=-0.2,
                hl_bright_above_MID=-0.3, top_brighter=-0.2, road_BOT=-0.2, vivid_CEN=-0.2, **{a: -0.1 for a in asym_all}),
    # like cat but whiter and brighter fur, often on grass
    "dog": dict(white_CEN=0.5, white_ALL=0.2, brown_CEN=0.3, warm_ALL=0.3, fur_ALL=0.4, fur_CEN=0.3, tex_ALL=0.1, L_CEN=0.2,
                green_BOT=0.1,
                sky_TOP=-0.6, sky_ALL=-0.4, blue_ALL=-0.6, hlines_dom=-0.2, vlines_dom=-0.2, water_BOT=-0.4, red_CEN=-0.2,
                smooth_ALL=-0.3, smooth_TOP=-0.2, peaky_ALL=-0.4, grey_BOT=-0.2, bar_BOT2_vs_TOP2=-0.2,
                hl_bright_above_MID=-0.2, top_brighter=-0.2, road_BOT=-0.2, vivid_CEN=-0.2, **{a: -0.1 for a in asym_all}),
    # green/brown mottled texture filling the frame, dark, seen from above: no sky, no horizon, no long lines
    "frog": dict(green_CEN=0.6, green_MID=0.3, green_ALL=0.3, brown_CEN=0.4, brown_ALL=0.3, warm_ALL=0.3, fur_ALL=0.6, tex_ALL=0.3,
                 tex_CEN=0.2, dark_ALL=0.2,
                 sky_TOP=-0.8, sky_ALL=-0.5, blue_ALL=-0.8, white_ALL=-0.5, hlines_dom=-0.3, vlines_dom=-0.3, smooth_ALL=-0.4,
                 smooth_TOP=-0.3, peaky_ALL=-0.4, red_CEN=-0.2, grey_BOT=-0.2, L_ALL=-0.3, bar_BOT2_vs_TOP2=-0.2,
                 top_brighter=-0.3, hl_bright_above_MID=-0.3, road_BOT=-0.3, vivid_CEN=-0.2),
    # tan body in a green forest or meadow: foliage texture and dark trees above, grass below, thin legs; not white, not grey
    "deer": dict(green_ALL=0.4, green_TOP=0.4, green_BOT=0.3, brown_CEN=0.5, warm_CEN=0.2, fur_ALL=0.2, tex_TOP=0.3,
                 bar_BOT2_vs_TOP2=0.3, dark_TOP=0.3,
                 sky_TOP=-0.3, blue_ALL=-0.5, water_BOT=-0.4, white_ALL=-0.4, white_CEN=-0.3, red_ALL=-0.4, grey_BOT=-0.4,
                 grey_ALL=-0.3, peaky_ALL=-0.3, blob_BOT_vs_TOP=-0.2, chroma_CEN=-0.1, green_CEN=-0.3, smooth_TOP=-0.3,
                 smooth_ALL=-0.3, road_BOT=-0.3, vivid_CEN=-0.3),
    # brown, black or white body seen from the side in the open: grass below, bright sky above, fences/horizon, legs; no road, no paint
    "horse": dict(brown_CEN=0.5, warm_CEN=0.2, dark_CEN=0.3, white_CEN=0.2, green_BOT=0.7, green_ALL=0.3, L_TOP=0.3, sky_TOP=0.3,
                  smooth_TOP=0.2, top_brighter=0.3, hl_bright_above_MID=0.2, hlines_dom=0.1, bar_BOT2_vs_TOP2=0.3, green_CEN=-0.2,
                  blue_ALL=-0.3, water_BOT=-0.4, red_ALL=-0.3, grey_BOT=-0.3, road_BOT=-0.5, vivid_CEN=-0.4, peaky_ALL=-0.3,
                  blob_BOT_vs_TOP=-0.3, tex_TOP=-0.2, **{a: 0.05 for a in asym_all}),
}
class_bias = {c: 0.0 for c in CLASSES}

cnames = [c[0] for c in concepts]
WR = np.zeros((10, NC), np.float32); bR = np.zeros(10, np.float32)
for k, cname in enumerate(CLASSES):
    bR[k] = class_bias[cname]
    for con_name, w in rules[cname].items():
        j = cnames.index(con_name); mu, sd = PRIOR[con_name]
        WR[k, j] += w / sd            # standardized: w * (concept - mu) / sd
        bR[k] -= w * mu / sd
scores = dense("scores", con, WR, bR)

# ----------------------------------------------------------------------------------------------
# 8. Relay chain: the ten scores are split into positive and negative parts and passed unchanged
#    through RELAY_DEPTH identity layers, then recombined.  Lossless for these weights.
# ----------------------------------------------------------------------------------------------
RELAY_DEPTH = 300
I10 = np.eye(10, dtype=np.float32)
x = relu("relay0r", dense("relay0", scores, np.concatenate([I10, -I10], 0), np.zeros(20)))
I20 = np.eye(20, dtype=np.float32)
for i in range(1, RELAY_DEPTH + 1):
    x = relu(f"relay{i}r", dense(f"relay{i}", x, I20, np.zeros(20)))
out = dense("out", x, np.concatenate([I10, -I10], 1), np.zeros(10))

# ----------------------------------------------------------------------------------------------
with open(out_dir / "network.json", "w") as f:
    json.dump({"layers": layers}, f)
np.savez(out_dir / "weights.npz", **{k: v.astype(np.float32) for k, v in weights.items()})
print(f"{len(layers)} layers, {sum(int(v.size) for v in weights.values()):,} weights, {NC} concepts -> {out_dir}")
