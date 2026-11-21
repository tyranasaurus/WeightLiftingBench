"""Writes a hand-designed CIFAR-style 10-class network: python3 build.py OUT_DIR

No data is read. Every number below is typed by hand from general knowledge of
what photos of the ten classes look like (colours of sky, water, grass, fur,
paint; how smooth or textured those surfaces are; where they sit in a frame).

Pipeline
  1. conv 7x7 (valid) on the image: colour primitives (centre tap only), Sobel
     gradients and 7-pixel-long line detectors, each followed by a ReLU ramp.
  2. two 1x1 conv layers turn the ramps into 13 soft per-pixel "conditions"
     (blue, green, warm, white, gray, dark, vivid red, vivid yellow, edge,
     smooth, horizontal line, vertical line, bright-and-smooth), each in [0, 1].
  3. a dense layer scores hand-written scene prototypes: every prototype says
     which "material" (sky, water, grass, fur, car paint, ...) fills each of 8
     image regions; each material is a hand-typed table of how often a pixel of
     it meets each condition.  The score is the Bernoulli naive-Bayes log
     likelihood of the region's condition fractions under that table.
  4. max over the prototypes of each class -> 10 class scores.
"""
import sys, json
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

layers, weights = [], {}

# ----------------------------------------------------------------------------
# 1. first conv layer: 7x7 valid -> 26 x 26
# ----------------------------------------------------------------------------
K, C = 7, 3                      # kernel size, centre index
H1 = 32 - K + 1                  # 26
conv1 = []                       # (name, kernel (3,K,K), bias)

PRIM = {  # linear colour primitives (coefficients on r, g, b)
    "br": (-1, 0, 1), "bg": (0, -1, 1), "gb": (0, 1, -1), "gr": (-1, 1, 0),
    "rb": (1, 0, -1), "rg": (1, -1, 0), "L": (1 / 3, 1 / 3, 1 / 3),
}


def color_kernel(coef):
    k = np.zeros((3, K, K)); k[:, C, C] = coef; return k


def add_ramp(name, kernel, t0, t1):
    """two channels whose difference is clip((x - t0) / (t1 - t0), 0, 1)."""
    s = 1.0 / (t1 - t0)
    conv1.append((name + "_a", kernel * s, -t0 * s))
    conv1.append((name + "_b", kernel * s, -t0 * s - 1.0))


# colour ramps: (name, primitive, t0, t1)
RAMPS = [
    ("br1", "br", 0.03, 0.08),    # bluish: b above r
    ("bg1", "bg", -0.06, -0.02),  # b not far below g
    ("Llo", "L", 0.15, 0.25),     # not dark
    ("gb1", "gb", 0.03, 0.08),    # greenish: g above b
    ("gr1", "gr", -0.04, 0.00),   # g not below r
    ("rb1", "rb", 0.04, 0.10),    # warm: r above b
    ("rg1", "rg", -0.02, 0.02),   # r not below g
    ("Lhi", "L", 0.55, 0.70),     # bright
    ("Ldk", "L", 0.10, 0.18),     # (1 - this) = dark
    ("Lmid", "L", 0.45, 0.60),    # fairly bright (for sky-like smooth pixels)
    ("rg2", "rg", 0.22, 0.32),    # vivid red
    ("rb2", "rb", 0.20, 0.30),
    ("rb3", "rb", 0.30, 0.42),    # vivid yellow / orange
    ("gb2", "gb", 0.20, 0.30),
]
for nm, pr, t0, t1 in RAMPS:
    add_ramp(nm, color_kernel(PRIM[pr]), t0, t1)

# rectified pairwise colour differences (sum/2 = max - min = saturation)
for pr in ("rg", "gr", "gb", "bg", "br", "rb"):
    conv1.append(("abs_" + pr, color_kernel(PRIM[pr]), 0.0))

# Sobel gradients of brightness (derivative per pixel)
L3 = np.full(3, 1 / 3)
gx = np.zeros((3, K, K)); gy = np.zeros((3, K, K))
for oy in (-1, 0, 1):
    for ox in (-1, 0, 1):
        w = (2 if oy == 0 else 1) * ox / 8.0
        gx[:, C + oy, C + ox] += L3 * w
        w = (2 if ox == 0 else 1) * oy / 8.0
        gy[:, C + oy, C + ox] += L3 * w
for nm, k in (("gxp", gx), ("gxn", -gx), ("gyp", gy), ("gyn", -gy)):
    conv1.append((nm, k, 0.0))

# straight 7-pixel line detectors: central difference averaged along the line
hl = np.zeros((3, K, K)); vl = np.zeros((3, K, K))
for o in range(-3, 4):
    hl[:, C + 1, C + o] += L3 / 14.0; hl[:, C - 1, C + o] -= L3 / 14.0
    vl[:, C + o, C + 1] += L3 / 14.0; vl[:, C + o, C - 1] -= L3 / 14.0
for nm, k in (("hp", hl), ("hn", -hl), ("vp", vl), ("vn", -vl)):
    add_ramp(nm, k, 0.03, 0.06)

names1 = [c[0] for c in conv1]
idx1 = {n: i for i, n in enumerate(names1)}
weights["c1.W"] = np.stack([c[1] for c in conv1]).astype(np.float32)
weights["c1.b"] = np.array([c[2] for c in conv1], np.float32)
layers.append({"name": "c1", "op": "conv", "inputs": ["image"], "out": len(conv1), "kernel": [K, K], "stride": 1, "padding": 0})
layers.append({"name": "r1", "op": "relu", "inputs": ["c1"]})


# ----------------------------------------------------------------------------
# 2./3. 1x1 conv layers building the soft conditions
# ----------------------------------------------------------------------------
def up(n):  # linear expression of a ramp from layer-1 channels
    return {n + "_a": 1.0, n + "_b": -1.0}


def comb(*terms):
    out = {}
    for coef, d in terms:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + coef * v
    return out


def one_by_one(name, inp, rows, idx_in):
    W = np.zeros((len(rows), len(idx_in), 1, 1)); b = np.zeros(len(rows))
    for o, (_, d, bias) in enumerate(rows):
        for k, v in d.items():
            W[o, idx_in[k], 0, 0] += v
        b[o] = bias
    weights[name + ".W"] = W.astype(np.float32); weights[name + ".b"] = b.astype(np.float32)
    layers.append({"name": name, "op": "conv", "inputs": [inp], "out": len(rows), "kernel": [1, 1], "stride": 1, "padding": 0})
    layers.append({"name": name + "r", "op": "relu", "inputs": [name]})
    return {r[0]: i for i, r in enumerate(rows)}


sat = {"abs_" + p: 0.5 for p in ("rg", "gr", "gb", "bg", "br", "rb")}
grad = {n: 1.0 for n in ("gxp", "gxn", "gyp", "gyn")}
rows2 = [
    ("BLUE", comb((1, up("br1")), (1, up("bg1")), (1, up("Llo"))), -2.0),
    ("GREEN", comb((1, up("gb1")), (1, up("gr1"))), -1.0),
    ("WARM", comb((1, up("rb1")), (1, up("rg1"))), -1.0),
    ("DARK", comb((-1, up("Ldk"))), 1.0),
    ("VIVRED", comb((1, up("rg2")), (1, up("rb2"))), -1.0),
    ("VIVYEL", comb((1, up("rb3")), (1, up("gb2"))), -1.0),
    ("HLINE", comb((1, up("hp")), (1, up("hn"))), 0.0),
    ("VLINE", comb((1, up("vp")), (1, up("vn"))), 0.0),
    ("Lhi", up("Lhi"), 0.0),
    ("Lmid", up("Lmid"), 0.0),
    ("Llo", up("Llo"), 0.0),
    ("sat_a", comb((1 / 0.05, sat)), -1.0),            # saturation ramp 0.05 -> 0.10
    ("sat_b", comb((1 / 0.05, sat)), -2.0),
    ("edge_a", comb((1 / 0.04, grad)), -1.0),          # edge ramp 0.04 -> 0.08
    ("edge_b", comb((1 / 0.04, grad)), -2.0),
    ("smo_a", comb((1 / 0.013, grad)), -0.012 / 0.013),  # smooth ramp 0.012 -> 0.025
    ("smo_b", comb((1 / 0.013, grad)), -0.012 / 0.013 - 1.0),
]
idx2 = one_by_one("c2", "r1", rows2, idx1)

COND = ["BLUE", "GREEN", "WARM", "WHITE", "GRAY", "DARK", "VIVRED", "VIVYEL", "EDGE", "SMOOTH", "HLINE", "VLINE", "BRSMOOTH"]
satup = {"sat_a": 1.0, "sat_b": -1.0}
rows3 = []
for c in COND:
    if c == "WHITE":
        rows3.append((c, comb((1, {"Lhi": 1.0}), (-1, satup)), 0.0))
    elif c == "GRAY":
        rows3.append((c, comb((1, {"Llo": 1.0}), (-1, {"Lhi": 1.0}), (-1, satup)), 0.0))
    elif c == "EDGE":
        rows3.append((c, {"edge_a": 1.0, "edge_b": -1.0}, 0.0))
    elif c == "SMOOTH":
        rows3.append((c, {"smo_a": -1.0, "smo_b": 1.0}, 1.0))
    elif c == "BRSMOOTH":   # AND(smooth, fairly bright)
        rows3.append((c, {"smo_a": -1.0, "smo_b": 1.0, "Lmid": 1.0}, 0.0))
    else:
        rows3.append((c, {c: 1.0}, 0.0))
one_by_one("c3", "c2r", rows3, idx2)
NC = len(COND)

# ----------------------------------------------------------------------------
# 4. materials and scene prototypes (all hand-typed guesses)
# ----------------------------------------------------------------------------
# share of pixels of each material expected to meet each condition, in COND order:
#              BLUE GREEN WARM WHITE GRAY DARK VRED VYEL EDGE SMOOTH HLINE VLINE BRSMOOTH
MAT = {
    "sky":      (.75, .00, .01, .07, .03, .00, .00, .00, .04, .80, .02, .01, .75),
    "haze":     (.20, .00, .06, .40, .20, .00, .00, .00, .04, .80, .02, .01, .70),
    "water":    (.50, .03, .03, .06, .20, .06, .00, .00, .25, .30, .10, .02, .20),
    "grass":    (.00, .55, .25, .00, .03, .03, .00, .01, .35, .15, .05, .04, .03),
    "foliage":  (.02, .40, .18, .02, .06, .25, .00, .00, .50, .10, .07, .08, .02),
    "dirt":     (.01, .04, .55, .05, .18, .04, .01, .02, .30, .25, .05, .03, .12),
    "road":     (.05, .03, .20, .05, .45, .10, .00, .00, .25, .35, .07, .04, .10),
    "indoor":   (.05, .03, .40, .15, .20, .15, .02, .01, .30, .35, .10, .10, .22),
    "urban":    (.07, .05, .25, .15, .30, .12, .02, .01, .45, .22, .15, .18, .10),
    "furbrown": (.00, .02, .75, .02, .03, .12, .03, .02, .30, .25, .07, .07, .06),
    "furdark":  (.02, .02, .15, .02, .20, .55, .00, .00, .25, .40, .06, .06, .02),
    "furlight": (.03, .01, .30, .30, .30, .03, .00, .00, .30, .35, .07, .07, .18),
    "furmix":   (.02, .02, .40, .12, .25, .15, .01, .00, .45, .18, .08, .08, .06),
    "metal":    (.12, .01, .06, .35, .30, .06, .03, .01, .35, .35, .14, .09, .25),
    "carbody":  (.12, .02, .20, .12, .22, .20, .14, .03, .45, .30, .18, .10, .14),
    "carlow":   (.03, .03, .12, .05, .30, .35, .03, .00, .45, .20, .15, .10, .05),
    "truck":    (.10, .04, .20, .18, .22, .15, .11, .06, .50, .25, .18, .18, .12),
    "frog":     (.01, .20, .45, .03, .18, .10, .01, .03, .55, .10, .07, .07, .03),
    "bird":     (.03, .03, .40, .10, .22, .18, .02, .02, .40, .25, .08, .08, .08),
    "blur":     (.05, .30, .35, .05, .12, .05, .00, .01, .08, .60, .02, .02, .25),
    "litter":   (.01, .15, .45, .03, .18, .12, .00, .01, .55, .10, .07, .07, .02),
    "night":    (.00, .00, .05, .00, .05, .80, .00, .00, .08, .80, .02, .02, .00),
}
GENERIC = np.array([.10, .10, .35, .10, .15, .10, .02, .02, .35, .30, .10, .10, .15])
SMOOTHING = 0.30   # every material is blended 30% toward a generic photo: real regions are mixed
# effective number of independent pixels per region, per condition: colour is trusted most,
# texture and line rates (harder to guess) less
N_EFF = np.array([8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 2.5, 2.5, 4.0])

# regions on the 26 x 26 map: 4 horizontal bands x (centre, sides)
BANDS = [(0, 6), (6, 13), (13, 20), (20, 26)]
CENTRE = (7, 19)
region = np.zeros((H1, H1), int)
for bi, (y0, y1) in enumerate(BANDS):
    for y in range(y0, y1):
        for x in range(H1):
            region[y, x] = 2 * bi + (0 if CENTRE[0] <= x < CENTRE[1] else 1)
NR = 8
rsize = np.bincount(region.ravel(), minlength=NR)


def mix(spec):
    """'sky' or 'metal .6 sky .4' -> condition-rate vector"""
    t = spec.split()
    if len(t) == 1:
        v = np.array(MAT[t[0]], float)
    else:
        parts = [(t[i], float(t[i + 1])) for i in range(0, len(t), 2)]
        tot = sum(w for _, w in parts)
        v = sum(np.array(MAT[m], float) * w for m, w in parts) / tot
    v = (1 - SMOOTHING) * v + SMOOTHING * GENERIC
    return np.clip(v, 0.01, 0.99)


# each prototype: class, prior weight within class, 4 bands of (centre, sides)
PROTOS = [
    # airplane
    (0, .45, [("sky", "sky"), ("metal .5 sky .5", "sky .7 metal .3"), ("metal .6 sky .4", "sky .7 metal .3"), ("sky", "sky")]),
    (0, .30, [("haze", "haze"), ("metal .5 haze .5", "haze .7 metal .3"), ("metal .6 haze .4", "haze .7 metal .3"), ("haze", "haze")]),
    (0, .25, [("sky .5 haze .5", "sky .5 haze .5"), ("metal .6 haze .4", "haze .6 metal .4"), ("metal .7 road .3", "metal .5 road .5"), ("road .5 grass .5", "road .5 grass .5")]),
    # automobile
    (1, .50, [("urban .5 foliage .3 haze .2", "urban .5 foliage .3 haze .2"), ("carbody .7 urban .3", "carbody .5 urban .5"), ("carbody", "carbody .7 urban .3"), ("carlow .6 road .4", "carlow .5 road .5")]),
    (1, .20, [("haze", "haze"), ("carbody .6 haze .4", "carbody .4 haze .6"), ("carbody", "carbody .6 haze .4"), ("carlow .4 haze .6", "carlow .4 haze .6")]),
    (1, .30, [("carbody .6 urban .4", "carbody .4 urban .6"), ("carbody", "carbody"), ("carbody", "carbody"), ("carlow", "carlow .6 road .4")]),
    # bird
    (2, .35, [("foliage", "foliage"), ("bird .5 foliage .5", "foliage"), ("bird .5 foliage .5", "foliage"), ("foliage", "foliage")]),
    (2, .35, [("blur", "blur"), ("bird .5 blur .5", "blur"), ("bird .6 blur .4", "blur"), ("blur .6 bird .4", "blur")]),
    (2, .20, [("sky .5 haze .5", "sky .5 haze .5"), ("bird .4 sky .3 haze .3", "sky .5 haze .5"), ("bird .4 sky .3 haze .3", "sky .5 haze .5"), ("sky .4 haze .4 water .2", "sky .4 haze .4 water .2")]),
    (2, .10, [("haze .5 foliage .5", "haze .5 foliage .5"), ("bird .6 foliage .4", "foliage .6 haze .4"), ("bird .6 dirt .4", "dirt .5 grass .5"), ("dirt .5 grass .5", "dirt .5 grass .5")]),
    # cat
    (3, .60, [("indoor", "indoor"), ("furmix .4 furbrown .2 furlight .2 indoor .2", "indoor .6 furmix .4"), ("furmix .4 furbrown .2 furlight .2 indoor .2", "indoor .6 furmix .4"), ("indoor .5 furmix .5", "indoor")]),
    (3, .40, [("furmix .5 indoor .5", "indoor .6 furmix .4"), ("furmix .5 furlight .3 furbrown .2", "furmix .5 indoor .5"), ("furmix .5 furlight .3 furbrown .2", "furmix .5 indoor .5"), ("furmix .6 indoor .4", "indoor .5 furmix .5")]),
    # deer
    (4, .50, [("foliage", "foliage"), ("furbrown .5 foliage .5", "foliage"), ("furbrown .5 foliage .5", "foliage .7 grass .3"), ("grass .5 foliage .3 dirt .2", "grass .5 foliage .3 dirt .2")]),
    (4, .30, [("foliage .5 haze .5", "foliage .5 haze .5"), ("furbrown .5 grass .5", "grass .6 foliage .4"), ("furbrown .5 grass .5", "grass"), ("grass", "grass")]),
    (4, .20, [("foliage .5 dirt .5", "foliage .5 dirt .5"), ("furbrown .5 dirt .5", "dirt .5 foliage .5"), ("furbrown .5 dirt .5", "dirt"), ("dirt", "dirt")]),
    # dog
    (5, .45, [("indoor", "indoor"), ("furbrown .4 furmix .2 furlight .2 furdark .2", "indoor .6 furbrown .4"), ("furbrown .4 furmix .2 furlight .2 furdark .2", "indoor .6 furbrown .4"), ("indoor .5 furbrown .5", "indoor")]),
    (5, .30, [("foliage .5 urban .5", "foliage .5 urban .5"), ("furbrown .3 furmix .3 furlight .2 furdark .2", "grass .5 foliage .5"), ("furbrown .3 furmix .3 furlight .2 furdark .2", "grass"), ("grass", "grass")]),
    (5, .25, [("furbrown .5 indoor .5", "indoor .6 furbrown .4"), ("furbrown .5 furlight .3 furdark .2", "furbrown .5 indoor .5"), ("furbrown .5 furlight .3 furdark .2", "furbrown .5 indoor .5"), ("furbrown .6 indoor .4", "indoor .5 furbrown .5")]),
    # frog
    (6, .60, [("litter", "litter"), ("frog .7 litter .3", "litter"), ("frog .8 litter .2", "litter"), ("litter .6 frog .4", "litter")]),
    (6, .25, [("foliage .5 grass .5", "foliage .5 grass .5"), ("frog .7 foliage .3", "foliage .5 grass .5"), ("frog .8 grass .2", "grass .5 foliage .5"), ("grass .6 frog .4", "grass")]),
    (6, .15, [("night", "night"), ("frog .6 night .4", "night"), ("frog .7 night .3", "night"), ("night .6 frog .4", "night")]),
    # horse
    (7, .50, [("foliage .5 sky .25 haze .25", "foliage .5 sky .25 haze .25"), ("furbrown .4 furdark .2 foliage .4", "foliage .6 furbrown .4"), ("furbrown .4 furdark .2 grass .4", "grass .6 furbrown .4"), ("grass .7 furdark .3", "grass")]),
    (7, .30, [("urban .5 foliage .5", "urban .5 foliage .5"), ("furbrown .4 furdark .2 urban .4", "urban .6 furbrown .4"), ("furbrown .4 furdark .2 dirt .4", "dirt .6 furbrown .4"), ("dirt .7 furdark .3", "dirt")]),
    (7, .20, [("furbrown .5 furdark .3 foliage .2", "foliage .5 furbrown .5"), ("furbrown .6 furdark .4", "furbrown .5 foliage .5"), ("furbrown .6 furdark .4", "furbrown .5 grass .5"), ("furbrown .5 grass .5", "grass")]),
    # ship
    (8, .60, [("sky .5 haze .5", "sky .5 haze .5"), ("metal .5 sky .25 haze .25", "sky .35 haze .35 metal .3"), ("metal .6 water .4", "water .6 metal .4"), ("water", "water")]),
    (8, .25, [("sky .5 haze .5", "sky .5 haze .5"), ("metal .4 urban .3 haze .3", "urban .5 haze .5"), ("metal .6 water .4", "water .5 urban .5"), ("water", "water")]),
    (8, .15, [("haze .5 metal .5", "haze .6 metal .4"), ("metal .6 truck .4", "metal .5 haze .5"), ("metal .5 truck .5", "metal .5 water .5"), ("water .6 metal .4", "water")]),
    # truck
    (9, .50, [("haze .4 sky .3 urban .3", "haze .4 sky .3 urban .3"), ("truck .8 haze .2", "truck .6 haze .4"), ("truck", "truck .7 urban .3"), ("carlow .5 road .5", "carlow .4 road .6")]),
    (9, .30, [("truck .7 haze .3", "truck .5 haze .5"), ("truck", "truck"), ("truck", "truck"), ("carlow .6 truck .4", "carlow .5 road .5")]),
    (9, .20, [("haze .4 sky .3 foliage .3", "haze .4 sky .3 foliage .3"), ("truck .8 foliage .2", "truck .6 foliage .4"), ("truck", "truck .7 dirt .3"), ("carlow .5 dirt .5", "dirt .6 grass .4")]),
]
P = max(sum(1 for p in PROTOS if p[0] == k) for k in range(10))
NOUT = 10 * P
Wd = np.zeros((NOUT, NC * H1 * H1)); bd = np.full(NOUT, -1e4)
slot = [0] * 10
for k, prior, layout in PROTOS:
    o = k * P + slot[k]; slot[k] += 1
    rates = np.zeros((NR, NC))
    for bi, (cen, side) in enumerate(layout):
        rates[2 * bi] = mix(cen); rates[2 * bi + 1] = mix(side)
    logit = np.log(rates / (1 - rates))
    per_pix = N_EFF[:, None, None] / rsize[region][None] * logit[region].transpose(2, 0, 1)   # (NC, 26, 26)
    Wd[o] = per_pix.ravel()
    bd[o] = (N_EFF[None] * np.log(1 - rates)).sum() + np.log(prior)
weights["proto.W"] = Wd.astype(np.float32); weights["proto.b"] = bd.astype(np.float32)
layers.append({"name": "proto", "op": "dense", "inputs": ["c3r"], "out": NOUT})
layers.append({"name": "proto3", "op": "reshape", "inputs": ["proto"], "shape": [10, P, 1]})
layers.append({"name": "best", "op": "maxpool", "inputs": ["proto3"], "size": [P, 1], "stride": [P, 1]})
layers.append({"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]})

(OUT / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
np.savez(OUT / "weights.npz", **weights)
print("wrote", OUT, sum(v.size for v in weights.values()), "weights")
