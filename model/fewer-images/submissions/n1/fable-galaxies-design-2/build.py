"""Writes network.json and weights.npz for a hand-designed galaxy classifier.
Every number comes from formulas and hand-chosen constants in this file; no data is read.
Usage: python3 build.py OUT_DIR
"""
import json, sys, os
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
H = 128
W = {}
layers = []

# ---- c1: 3x3 binomial blur + channel mix, minus a brightness floor ----------------------------
# The survey viewer draws sky at about asinh(20*0.03)/sqrt(20) = 0.127; a floor of 0.2 removes sky and noise.
FLOOR = 0.2
blur = np.outer([1, 2, 1], [1, 2, 1]) / 16.0
c1 = np.zeros((3, 3, 3, 3))
c1[0, :, :, :] = blur / 3.0      # luminance (mean of red, green, blue)
c1[1, 2, :, :] = blur            # blue channel (g band)
c1[2, 0, :, :] = blur            # red channel (z band)
W["c1.W"] = c1
W["c1.b"] = np.full(3, -FLOOR)
layers.append({"name": "c1", "op": "conv", "inputs": ["image"], "out": 3, "kernel": [3, 3], "stride": 1, "padding": 1})
layers.append({"name": "s", "op": "relu", "inputs": ["c1"]})

# ---- d1: windowed fluxes and angular moments about the image centre ---------------------------
yy, xx = np.mgrid[0:H, 0:H].astype(np.float64)
dx, dy = xx - 63.5, yy - 63.5
r2 = dx * dx + dy * dy
th = np.arctan2(dy, dx)
gw = lambda sig: np.exp(-r2 / (2.0 * sig * sig))
SIG_CORE, SIG_BIG, SIG_COL, SIG_ASY = 3.0, 18.0, 12.0, 13.0
NF = 21
d1 = np.zeros((NF, 3, H, H))
d1[0, 0] = gw(SIG_CORE)                      # core flux
d1[1, 0] = gw(SIG_BIG)                       # wide flux
d1[2, 1] = gw(SIG_COL)                       # blue flux
d1[3, 2] = gw(SIG_COL)                       # red flux
for k in range(8):                           # m=2 moment (elongation), 8 phases
    d1[4 + k, 0] = gw(SIG_BIG) * np.cos(2 * th - k * np.pi / 4)
taper = r2 / (r2 + 36.0)                     # ignore the core so small centring errors do not count
for k in range(8):                           # m=1 moment (lopsidedness), 8 phases
    d1[12 + k, 0] = gw(SIG_ASY) * taper * np.cos(th - k * np.pi / 4)
d1[20, 0] = gw(SIG_ASY)                      # flux in the lopsidedness window
W["d1.W"] = d1.reshape(NF, -1)
W["d1.b"] = np.zeros(NF)
layers.append({"name": "d1", "op": "dense", "inputs": ["s"], "out": NF})
layers.append({"name": "r1", "op": "relu", "inputs": ["d1"]})

# ---- d2: build triples (a, b, 0); norm over each triple gives a function of a/b only ----------
KAPPA = 8.0 / np.pi                          # mean of sum_k relu(cos(alpha - k*pi/4))
d2 = np.zeros((12, NF))
d2[0, 4:12] = 1.0 / KAPPA; d2[1, 1] = 1.0    # elongation e = |m2| / flux  = (1-q)/(1+q) for ellipses
d2[3, 0] = 1.0;            d2[4, 1] = 1.0    # compactness = core flux / wide flux
d2[6, 2] = 0.5;            d2[7, 3] = 1.0    # colour = (blue / red) / 2
d2[9, 12:20] = 1.0 / KAPPA; d2[10, 20] = 1.0 # lopsidedness = |m1| / flux
W["d2.W"] = d2
W["d2.b"] = np.zeros(12)
layers.append({"name": "d2", "op": "dense", "inputs": ["r1"], "out": 12})
layers.append({"name": "tri", "op": "reshape", "inputs": ["d2"], "shape": [4, 3, 1]})
layers.append({"name": "ratio", "op": "norm", "inputs": ["tri"]})


def g(t):
    """Value the norm op gives at the first position of the triple (t, 1, 0)."""
    m = (1.0 + t) / 3.0
    var = ((t - m) ** 2 + (1 - m) ** 2 + m ** 2) / 3.0
    return (t - m) / np.sqrt(var)


# ---- d3 / d4: soft thresholds on the four ratios and a hand-typed table of log-likelihoods ----
# each feature has three states (low, mid, high) separated by two soft steps given as (t_lo, t_hi)
STEPS = [
    [(0.09, 0.15), (0.33, 0.43)],    # elongation: round | in-between | elongated (q about 0.79 and 0.45)
    [(0.09, 0.13), (0.22, 0.32)],    # compactness: very extended | extended | compact
    [(0.36, 0.44), (0.485, 0.565)],  # colour/2: red | neutral | blue (blue/red about 0.8 and 1.05)
    [(0.10, 0.20), (0.25, 0.40)],    # lopsidedness: symmetric | some | strong
]
# classes: 0 disturbed, 1 merging, 2 round, 3 in-between, 4 cigar, 5 barred, 6 tight, 7 loose, 8 edge-on no bulge, 9 edge-on bulge
PRIOR = [0.061, 0.104, 0.149, 0.114, 0.019, 0.115, 0.103, 0.148, 0.080, 0.106]
TABLE = [
    # elongation (low, mid, high)
    [(.40, .48, .12), (.35, .45, .20), (.88, .11, .01), (.15, .77, .08), (.01, .14, .85),
     (.45, .50, .05), (.40, .50, .10), (.40, .50, .10), (.01, .06, .93), (.01, .14, .85)],
    # compactness (very extended, extended, compact)
    [(.20, .50, .30), (.25, .45, .30), (.03, .27, .70), (.05, .35, .60), (.10, .50, .40),
     (.30, .55, .15), (.25, .55, .20), (.25, .50, .25), (.40, .50, .10), (.45, .45, .10)],
    # colour (red, neutral, blue)
    [(.40, .40, .20), (.40, .40, .20), (.80, .18, .02), (.75, .22, .03), (.60, .33, .07),
     (.35, .50, .15), (.30, .50, .20), (.12, .48, .40), (.25, .45, .30), (.65, .30, .05)],
    # lopsidedness (low, mid, high)
    [(.50, .30, .20), (.20, .35, .45), (.85, .12, .03), (.82, .14, .04), (.80, .15, .05),
     (.78, .17, .05), (.78, .17, .05), (.70, .22, .08), (.78, .16, .06), (.80, .15, .05)],
]
TRUST = [1.0, 0.7, 0.7, 1.0]         # how much each cue is trusted

d3 = np.zeros((16, 12)); b3 = np.zeros(16)
d4 = np.zeros((10, 16)); b4 = np.log(np.array(PRIOR))
for f in range(4):
    for j in range(10):
        b4[j] += TRUST[f] * np.log(TABLE[f][j][0])
    for sidx, (tlo, thi) in enumerate(STEPS[f]):
        xlo, xhi = g(tlo), g(thi)
        u = 4 * f + 2 * sidx
        d3[u, 3 * f] = 1.0;     b3[u] = -xlo
        d3[u + 1, 3 * f] = 1.0; b3[u + 1] = -xhi
        for j in range(10):
            p = TABLE[f][j]
            gain = TRUST[f] * (np.log(p[sidx + 1]) - np.log(p[sidx])) / (xhi - xlo)
            d4[j, u] = gain
            d4[j, u + 1] = -gain
W["d3.W"], W["d3.b"] = d3, b3
W["d4.W"], W["d4.b"] = d4, b4
layers.append({"name": "d3", "op": "dense", "inputs": ["ratio"], "out": 16})
layers.append({"name": "r3", "op": "relu", "inputs": ["d3"]})
layers.append({"name": "d4", "op": "dense", "inputs": ["r3"], "out": 10})

with open(os.path.join(out_dir, "network.json"), "w") as fh:
    json.dump({"layers": layers}, fh, indent=1)
np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
