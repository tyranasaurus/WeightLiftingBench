"""Data-free builder for a hand-designed CIFAR-10 network.

Every number here comes from formulas (Gabor / Laplacian / ring / bar shape
models) or from a rule table written by hand from general knowledge of what
the ten classes look like.  No training data is read.

Architecture (no concat op exists, so branches are merged by ADDing their
10-way score vectors):
  S branch : 1x1 colour primitives -> relu -> 1x1 semantic colour channels
             (sky / water / vegetation / road-grey / white / dark / brown,
             built as linear thresholds = soft ANDs of colour properties)
             -> relu -> avgpool 4x4 -> dense(10)
  E branch : 7x7 Gabor bank (6 orientations x 2 scales, +/- for abs) plus
             fine Laplacian, dark-ring (wheel) and dark-vertical-bar (leg)
             detectors -> relu -> 1x1 abs-combine -> relu
             -> avgpool 4x4 -> dense(10)   (texture / horizontal structure)
             -> maxpool 4x4 -> dense(10)   (peak shape presence)
  final    : add of the three score vectors -> (10,)
"""
import json
import sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ---------------------------------------------------------------- primitives
LUM = np.array([0.299, 0.587, 0.114], np.float64)

P1_NAMES = ["R", "G", "B", "L", "gn", "bl", "rd", "RG", "GR", "GB", "BG", "RB", "BR"]
P1_W = {
    "R": [1, 0, 0], "G": [0, 1, 0], "B": [0, 0, 1], "L": LUM.tolist(),
    "gn": [-0.5, 1, -0.5], "bl": [-0.5, -0.5, 1], "rd": [1, -0.5, -0.5],
    "RG": [1, -1, 0], "GR": [-1, 1, 0], "GB": [0, 1, -1],
    "BG": [0, -1, 1], "RB": [1, 0, -1], "BR": [-1, 0, 1],
}
# p3 outputs (after relu): raw + saturation + luminance hinges + R-B, R-G
P3_NAMES = ["R", "G", "B", "L", "gn", "bl", "rd", "sat", "Lhi", "Llo", "RB", "RG"]
# semantic colour channels of the S branch
S_NAMES = ["sky", "water", "veg", "gray", "white", "dark", "brown",
           "R", "G", "B", "L", "gn", "bl", "rd", "sat"]
# edge channels of the E branch
ORIENTS = [0, 30, 60, 90, 120, 150]
E_NAMES = (["H%d" % a for a in ORIENTS] + ["F%d" % a for a in ORIENTS]
           + ["TEX", "RING", "VBAR", "FUR"])

layers = []
weights = {}


def conv(name, inp, out, k, stride, pad, W, b):
    layers.append({"name": name, "op": "conv", "inputs": [inp], "out": out,
                   "kernel": [k, k], "stride": stride, "padding": pad})
    weights[name + ".W"] = np.asarray(W, np.float32)
    weights[name + ".b"] = np.asarray(b, np.float32)


def simple(name, op, inp, **kw):
    d = {"name": name, "op": op, "inputs": [inp]}
    d.update(kw)
    layers.append(d)


def dense(name, inp, nin, W, b):
    layers.append({"name": name, "op": "dense", "inputs": [inp], "out": 10})
    weights[name + ".W"] = np.asarray(W, np.float32).reshape(10, nin)
    weights[name + ".b"] = np.asarray(b, np.float32)


# ---------------------------------------------------------------- S branch
W = np.zeros((len(P1_NAMES), 3, 1, 1))
for i, n in enumerate(P1_NAMES):
    W[i, :, 0, 0] = P1_W[n]
conv("p1", "image", len(P1_NAMES), 1, 1, 0, W, np.zeros(len(P1_NAMES)))
simple("p2", "relu", "p1")

W = np.zeros((len(P3_NAMES), len(P1_NAMES), 1, 1))
b = np.zeros(len(P3_NAMES))
idx = {n: i for i, n in enumerate(P1_NAMES)}
for i, n in enumerate(P3_NAMES):
    if n in idx:
        W[i, idx[n], 0, 0] = 1.0
    elif n == "sat":
        for d in ["RG", "GR", "GB", "BG", "RB", "BR"]:
            W[i, idx[d], 0, 0] = 0.5
    elif n == "Lhi":
        W[i, idx["L"], 0, 0] = 1.0; b[i] = -0.5
    elif n == "Llo":
        W[i, idx["L"], 0, 0] = -1.0; b[i] = 0.5
conv("p3", "p2", len(P3_NAMES), 1, 1, 0, W, b)
simple("p4", "relu", "p3")

# semantic channels: linear threshold + gain, then relu  (soft AND of colours)
j = {n: i for i, n in enumerate(P3_NAMES)}
W = np.zeros((len(S_NAMES), len(P3_NAMES), 1, 1))
b = np.zeros(len(S_NAMES))


def setch(name, terms, bias):
    i = S_NAMES.index(name)
    for t, c in terms.items():
        W[i, j[t], 0, 0] = c
    b[i] = bias


setch("sky",   {"bl": 6.0, "L": 4.0}, -3.2)           # bright AND blue
setch("water", {"bl": 6.0, "L": -1.2}, -0.04)         # blue, not too bright
setch("veg",   {"gn": 3.0}, -0.24)                    # green dominant
setch("gray",  {"sat": -2.5, "Lhi": -1.2, "Llo": -1.2}, 0.4)   # low-sat mid-L
setch("white", {"L": 2.5, "sat": -4.0}, -1.5)         # bright AND unsaturated
setch("dark",  {"L": -2.0}, 0.56)                     # dark
setch("brown", {"RB": 2.4, "RG": 1.5, "bl": -1.5}, -0.36)      # warm, R>G>B
for n in ["R", "G", "B", "L", "gn", "bl", "rd", "sat"]:
    setch(n, {n: 1.0}, 0.0)
conv("s1", "p4", len(S_NAMES), 1, 1, 0, W, b)
simple("s2", "relu", "s1")


def add_clip(prefix, inp, caps):
    """Fixed saturating hinge: (relu(x) - relu(x-cap))/cap  ->  [0,1]."""
    n = len(caps)
    W = np.zeros((2 * n, n, 1, 1)); b = np.zeros(2 * n)
    for i, c in enumerate(caps):
        W[2 * i, i, 0, 0] = 1.0
        W[2 * i + 1, i, 0, 0] = 1.0; b[2 * i + 1] = -c
    conv(prefix + "_ca", inp, 2 * n, 1, 1, 0, W, b)
    simple(prefix + "_cr", "relu", prefix + "_ca")
    W = np.zeros((n, 2 * n, 1, 1))
    for i, c in enumerate(caps):
        W[i, 2 * i, 0, 0] = 1.0 / c; W[i, 2 * i + 1, 0, 0] = -1.0 / c
    conv(prefix + "_cb", prefix + "_cr", n, 1, 1, 0, W, np.zeros(n))
    return prefix + "_cb"


S_CAPS = [1, 1, 1, 0.5, 1, 0.5, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
s_bounded = add_clip("s", "s2", S_CAPS)
simple("s_avg", "avgpool", s_bounded, size=[8, 8], stride=[8, 8])
simple("s_flat", "reshape", "s_avg", shape=[len(S_NAMES) * 16])

# ---------------------------------------------------------------- E branch
def gabor(theta_deg, lam, sigma, gain=2.0):
    t = np.deg2rad(theta_deg)
    ys, xs = np.mgrid[-3:4, -3:4].astype(np.float64)
    xr = xs * np.cos(t) + ys * np.sin(t)
    yr = -xs * np.sin(t) + ys * np.cos(t)
    g = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xr / lam)
    return g / np.abs(g).sum() * gain    # unit step-edge response ~ gain/2


def log_kernel(sigma):
    ys, xs = np.mgrid[-3:4, -3:4].astype(np.float64)
    r2 = xs ** 2 + ys ** 2
    g = (r2 / sigma ** 2 - 2.0) * np.exp(-r2 / (2 * sigma ** 2))
    return g / np.abs(g).sum() * 2.0


def ring_kernel():
    ys, xs = np.mgrid[-3:4, -3:4].astype(np.float64)
    r = np.sqrt(xs ** 2 + ys ** 2)
    g = (0.6 * np.exp(-r ** 2 / 2.0)
         - 1.0 * np.exp(-(r - 2.5) ** 2 / (2 * 0.9 ** 2))
         + 0.5 * np.exp(-r ** 2 / (2 * 3.5 ** 2)))
    return g / np.abs(g).sum() * 2.0


def vbar_kernel():
    ys, xs = np.mgrid[-3:4, -3:4].astype(np.float64)
    g = (0.5 * np.exp(-xs ** 2 / (2 * 2.5 ** 2))
         - 1.0 * np.exp(-xs ** 2 / (2 * 0.9 ** 2))) * np.exp(-ys ** 2 / (2 * 3.0 ** 2))
    return g / np.abs(g).sum() * 2.0


e1_filters = []   # (kernel2d, sign)
for lam, sig, gain in [(5.5, 2.0, 2.0), (3.0, 1.2, 3.0)]:
    for a in ORIENTS:
        g = gabor(a, lam, sig, gain)
        e1_filters.append((g, +1)); e1_filters.append((g, -1))
tl = log_kernel(0.8)
e1_filters.append((tl, +1)); e1_filters.append((tl, -1))
e1_filters.append((ring_kernel(), +1))
e1_filters.append((vbar_kernel(), +1))

W = np.zeros((len(e1_filters), 3, 7, 7))
for o, (g, s) in enumerate(e1_filters):
    for c in range(3):
        W[o, c] = s * g * LUM[c]
conv("e1", "image", len(e1_filters), 7, 1, 3, W, np.zeros(len(e1_filters)))
simple("e2", "relu", "e1")

# abs-combine Gabor pairs and Laplacian pair; pass ring / bar through
W = np.zeros((len(E_NAMES), len(e1_filters), 1, 1))
for k in range(12):
    W[k, 2 * k, 0, 0] = 1.0; W[k, 2 * k + 1, 0, 0] = 1.0
W[12, 24, 0, 0] = 1.0; W[12, 25, 0, 0] = 1.0     # TEX
W[13, 26, 0, 0] = 1.0                            # RING
W[14, 27, 0, 0] = 1.0                            # VBAR
for k in range(12, 24):                          # FUR: isotropic fine texture
    W[15, k, 0, 0] = 1.0 / 6.0
conv("e3", "e2", len(E_NAMES), 1, 1, 0, W, np.zeros(len(E_NAMES)))
simple("e4", "relu", "e3")
E_CAPS = [1] * 12 + [0.5, 1, 1, 1]
e_bounded = add_clip("e", "e4", E_CAPS)
simple("e_avg", "avgpool", e_bounded, size=[8, 8], stride=[8, 8])
simple("e_avg_flat", "reshape", "e_avg", shape=[len(E_NAMES) * 16])
simple("e_max", "maxpool", e_bounded, size=[8, 8], stride=[8, 8])
simple("e_max_flat", "reshape", "e_max", shape=[len(E_NAMES) * 16])

# ---------------------------------------------------------------- readout rules
# spatial masks over the 4x4 grid, each normalised to sum 1 (region average)
def mask(rows):
    m = np.zeros((4, 4))
    for r, vals in rows.items():
        m[r, :] = vals
    return m / m.sum()

MASKS = {
    "top":    mask({0: 1.0, 1: 0.5}),
    "bot":    mask({2: 0.5, 3: 1.0}),
    "ctr":    mask({1: np.array([0, 1, 1, 0]), 2: np.array([0, 1, 1, 0])}),
    "all":    np.ones((4, 4)) / 16.0,
    "mid":    mask({1: 1.0, 2: 1.0}),
    "toprow": mask({0: 1.0}),
    "botrow": mask({3: 1.0}),
}

# rule = (class, channel, region, magnitude); hand-written from class knowledge
S_RULES = [
    # airplane: bright sky (blue or overcast grey) above, pale smooth fuselage,
    # tarmac below, horizontal wings (not a vertical box)
    (0, "sky", "top", 2.0), (0, "gray", "top", 0.5), (0, "white", "ctr", 1.5),
    (0, "gray", "ctr", 0.3), (0, "gray", "bot", 0.4),
    (0, "veg", "all", -1.2), (0, "water", "bot", -1.2), (0, "brown", "ctr", -0.5),
    # automobile: saturated compact body, grey road, wheels low, smooth paint
    (1, "sat", "ctr", 1.2), (1, "rd", "ctr", 0.6), (1, "bl", "ctr", 0.4),
    (1, "gray", "bot", 0.6), (1, "white", "ctr", 0.3), (1, "sky", "top", -0.4),
    (1, "veg", "all", -0.6), (1, "water", "all", -0.8), (1, "brown", "ctr", -0.8),
    # bird: small object, textured foliage everywhere, no legs / wheels
    (2, "veg", "all", 1.2), (2, "sky", "top", 0.4), (2, "brown", "ctr", 0.1),
    (2, "dark", "ctr", 0.2), (2, "water", "bot", -0.6), (2, "white", "ctr", -0.3),
    # cat: fur face close-up, neutral/warm indoor; NOT green/blue/saturated
    (3, "brown", "ctr", 0.8), (3, "gray", "ctr", 0.5), (3, "dark", "ctr", 0.4),
    (3, "white", "ctr", 0.2), (3, "sky", "top", -0.5), (3, "veg", "all", -1.5),
    (3, "water", "all", -1.5), (3, "sat", "ctr", -1.0),
    # deer: tan body, grass / forest, thin legs
    (4, "brown", "ctr", 1.0), (4, "veg", "bot", 1.0), (4, "veg", "all", 0.6),
    (4, "sky", "top", -0.3), (4, "water", "all", -0.8), (4, "gray", "bot", -0.4),
    # dog: fur full body, legs visible, varied indoor/outdoor
    (5, "brown", "ctr", 0.6), (5, "white", "ctr", 0.3), (5, "dark", "ctr", 0.4),
    (5, "gray", "ctr", 0.3), (5, "sky", "top", 0.3), (5, "veg", "bot", 0.4),
    (5, "veg", "all", -0.5), (5, "water", "all", -1.5), (5, "sat", "ctr", -0.8),
    # frog: green smooth body filling the centre
    (6, "veg", "ctr", 2.0), (6, "gn", "ctr", 0.8), (6, "G", "ctr", 0.5),
    (6, "brown", "ctr", 0.4), (6, "veg", "bot", 0.6), (6, "sky", "top", -0.6),
    (6, "water", "all", -0.5), (6, "white", "ctr", -0.5),
    # horse: large brown body, grass, sky, strong legs
    (7, "brown", "ctr", 1.0), (7, "veg", "bot", 0.9), (7, "veg", "all", 0.2),
    (7, "sky", "top", 0.6), (7, "dark", "ctr", 0.4), (7, "white", "ctr", 0.3),
    (7, "water", "all", -0.8), (7, "gray", "bot", -0.4),
    # ship: blue water below, horizon, pale smooth hull
    (8, "water", "bot", 2.2), (8, "water", "all", 0.6), (8, "sky", "top", 0.8),
    (8, "white", "ctr", 1.0), (8, "gray", "ctr", 0.4), (8, "veg", "all", -1.0),
    (8, "brown", "ctr", -0.5),
    # truck: tall pale box, vertical sides, wheels, road, smooth panels
    (9, "white", "ctr", 1.0), (9, "gray", "ctr", 0.3), (9, "sat", "ctr", 0.5),
    (9, "gray", "bot", 0.4), (9, "sky", "top", 0.4), (9, "veg", "all", -0.6),
    (9, "water", "all", -0.8), (9, "brown", "ctr", -0.8),
]
S_BIAS = [-1.0, -1.0, -1.0, -1.0, -1.1, -1.0, -1.1, -1.2, -1.2, -1.0]

# FUR (isotropic fine texture) separates furry animals (+) from smooth
# vehicles (-); coarse horizontal energy marks horizon / wings / box tops.
EAVG_RULES = [
    (0, "H90", "mid", 0.8), (0, "H0", "ctr", -0.4),
    (0, "FUR", "ctr", -1.0), (0, "FUR", "top", -0.6),
    (1, "H90", "bot", 0.4), (1, "FUR", "ctr", -1.5),
    (2, "FUR", "all", 3.0),
    (3, "FUR", "ctr", 2.5),
    (4, "FUR", "ctr", 1.0),
    (5, "FUR", "ctr", 1.5),
    (6, "FUR", "ctr", -1.5),
    (7, "H90", "mid", 0.3), (7, "FUR", "ctr", 1.0),
    (8, "H90", "mid", 1.6), (8, "H90", "bot", 0.6), (8, "FUR", "ctr", -1.0),
    (8, "FUR", "bot", 0.3),
    (9, "H90", "mid", 0.8), (9, "H0", "ctr", 0.6), (9, "FUR", "ctr", -1.5),
]
EMAX_RULES = [
    (0, "RING", "bot", -0.3),
    (1, "RING", "bot", 0.5),
    (2, "RING", "bot", -0.3), (2, "VBAR", "bot", -0.3),
    (3, "RING", "bot", -0.3), (3, "VBAR", "bot", -0.5),
    (4, "VBAR", "bot", 0.4), (4, "RING", "bot", -0.3),
    (5, "VBAR", "bot", 0.5), (5, "RING", "bot", -0.3),
    (6, "RING", "bot", -0.3), (6, "VBAR", "bot", -0.3),
    (7, "VBAR", "bot", 0.4), (7, "RING", "bot", -0.3),
    (8, "H90", "mid", 0.4), (8, "RING", "bot", -0.3),
    (9, "RING", "bot", 0.5), (9, "VBAR", "ctr", 0.3),
]


def build_readout(names, rules, bias):
    W = np.zeros((10, len(names), 4, 4))
    ci = {n: i for i, n in enumerate(names)}
    for cls, ch, reg, mag in rules:
        W[cls, ci[ch]] += mag * MASKS[reg]
    return W.reshape(10, -1), np.array(bias, np.float64)


Ws, bs = build_readout(S_NAMES, S_RULES, S_BIAS)
dense("s_head", "s_flat", len(S_NAMES) * 16, Ws, bs)
Wa, ba = build_readout(E_NAMES, EAVG_RULES, [0.0] * 10)
dense("e_avg_head", "e_avg_flat", len(E_NAMES) * 16, Wa, ba)
Wm, bm = build_readout(E_NAMES, EMAX_RULES, [0.0] * 10)
dense("e_max_head", "e_max_flat", len(E_NAMES) * 16, Wm, bm)

layers.append({"name": "scores", "op": "add",
               "inputs": ["s_head", "e_avg_head", "e_max_head"]})

with open(OUT + "/network.json", "w") as f:
    json.dump({"layers": layers}, f, indent=1)
np.savez(OUT + "/weights.npz", **weights)
print("wrote", len(layers), "layers,",
      sum(v.size for v in weights.values()), "weights")
