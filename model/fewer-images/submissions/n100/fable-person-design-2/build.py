"""Writes network.json and weights.npz from formulas only (no data read). Usage: python3 build.py OUT_DIR"""
import sys, json, numpy as np
out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
W = {}
L = []
def layer(name, op, inputs, **kw):
    L.append(dict(name=name, op=op, inputs=inputs, **kw))

# ---- per-pixel colour rules (1x1 conv): each channel is the violation of one linear rule, relu(-c/margin)
m = 0.03
rules = [  # (coefficients on R, G, B, constant) with c >= 0 meaning "consistent"
    # skin cone from general knowledge: hue about 6..36 degrees, saturation 0.15..0.65, not dark
    ((-0.1, 1.0, -0.9), 0.0, m),      # (G-B) >= 0.1 (R-B)
    ((0.6, -1.0, 0.4), 0.0, m),       # (G-B) <= 0.6 (R-B)
    ((0.85, 0.0, -1.0), 0.0, m),      # R-B >= 0.15 R
    ((-0.35, 0.0, 1.0), 0.0, m),      # R-B <= 0.65 R
    ((1.0, 0.0, 0.0), -0.3, m),       # R >= 0.3
    # snow-like: bright, neutral to bluish (not warm white)
    ((0.0, 1.0, 0.0), -0.7, 0.08),    # G >= 0.7
    ((1.0, 0.0, 0.0), -0.6, 0.08),    # R >= 0.6
    ((-1.0, 0.0, 1.0), 0.03, 0.04),   # R - B <= 0.03
    ((1.0, 0.0, -1.0), 0.3, 0.08),    # B - R <= 0.3
]
f1W = np.zeros((len(rules), 3, 1, 1)); f1b = np.zeros(len(rules))
for k, (c, c0, mg) in enumerate(rules):
    f1W[k, :, 0, 0] = -np.array(c) / mg; f1b[k] = -c0 / mg
W["f1.W"], W["f1.b"] = f1W, f1b
layer("f1", "conv", ["image"], out=len(rules), kernel=[1, 1], stride=1, padding=0)
layer("f1r", "relu", ["f1"])
skW = np.zeros((1, len(rules), 1, 1)); skW[0, :5] = -1.0
snW = np.zeros((1, len(rules), 1, 1)); snW[0, 5:] = -1.0
W["sk.W"], W["sk.b"] = skW, np.ones(1)
W["sn.W"], W["sn.b"] = snW, np.ones(1)
layer("sk", "conv", ["f1r"], out=1, kernel=[1, 1], stride=1, padding=0); layer("skr", "relu", ["sk"])
layer("sn", "conv", ["f1r"], out=1, kernel=[1, 1], stride=1, padding=0); layer("snr", "relu", ["sn"])
layer("sk48", "avgpool", ["skr"], size=[2, 2], stride=[2, 2])
layer("sn48", "avgpool", ["snr"], size=[2, 2], stride=[2, 2])

# ---- compact skin blobs (face / limb sized): centre disc mean minus surrounding ring mean, threshold 0.55
def blob_kernel(k, r):
    h = k // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]; d = np.sqrt(yy ** 2 + xx ** 2)
    cen = d <= r; sur = (d > r) & (d <= 2 * r)
    return cen / cen.sum() - sur / sur.sum()
THR = 0.55
bS = np.stack([blob_kernel(11, 1.5), blob_kernel(11, 2.5)])[:, None]
bL = np.stack([blob_kernel(27, 4.0), blob_kernel(27, 6.5)])[:, None]
W["bS.W"], W["bS.b"] = bS, -THR * np.ones(2)
W["bL.W"], W["bL.b"] = bL, -THR * np.ones(2)
layer("bS", "conv", ["sk48"], out=2, kernel=[11, 11], stride=1, padding=5); layer("bSr", "relu", ["bS"])
layer("bSm", "maxpool", ["bSr"], size=[16, 48], stride=[16, 48])          # top / middle / bottom band maxima
layer("bL", "conv", ["sk48"], out=2, kernel=[27, 27], stride=2, padding=13); layer("bLr", "relu", ["bL"])
layer("bLm", "maxpool", ["bLr"], size=[8, 24], stride=[8, 24])
band = np.array([1.0, 1.0, 0.5])          # heads sit in the upper part; floors, tables and ground at the bottom
dW = np.zeros((2, 6)); dW[1] = 10.0 * np.tile(band, 2)
W["dS.W"], W["dS.b"] = dW, np.zeros(2)
W["dL.W"], W["dL.b"] = dW.copy(), np.zeros(2)
layer("dS", "dense", ["bSm"], out=2)
layer("dL", "dense", ["bLm"], out=2)

# ---- too much skin-coloured area (wood, sand, warm light) counts against; also carries the prior bias
layer("gf", "avgpool", ["sk48"], size=[48, 48], stride=[48, 48])
W["gfd.W"], W["gfd.b"] = np.ones((1, 1)), -0.25 * np.ones(1)
layer("gfd", "dense", ["gf"], out=1); layer("gfr", "relu", ["gfd"])
W["dG.W"], W["dG.b"] = np.array([[0.0], [-6.0]]), np.array([0.0, -0.5])
layer("dG", "dense", ["gfr"], out=2)

# ---- snow filling the lower half of the picture (ski / snowboard scenes) counts for a person
layer("snb", "avgpool", ["sn48"], size=[24, 48], stride=[24, 48])           # (1, 2, 1): top half, bottom half
W["snd.W"], W["snd.b"] = np.array([[0.0, 1.0]]), -0.35 * np.ones(1)
layer("snd", "dense", ["snb"], out=1); layer("sndr", "relu", ["snd"])
W["dN.W"], W["dN.b"] = np.array([[0.0], [8.0]]), np.zeros(2)
layer("dN", "dense", ["sndr"], out=2)

layer("scores", "add", ["dS", "dL", "dG", "dN"])
import os
os.makedirs(out_dir, exist_ok=True)
json.dump({"layers": L}, open(os.path.join(out_dir, "network.json"), "w"), indent=1)
np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
