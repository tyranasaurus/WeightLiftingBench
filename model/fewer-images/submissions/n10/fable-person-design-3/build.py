"""Writes network.json and weights.npz for a hand-designed person / no-person network.
Every number comes from formulas and hand-chosen constants below (general knowledge of skin colour); no data is read."""
import sys, json, os
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
L, W = [], {}

def conv(name, inp, w, b, stride=1, padding=0):
    w = np.asarray(w, np.float32); b = np.asarray(b, np.float32)
    L.append({"name": name, "op": "conv", "inputs": [inp], "out": int(w.shape[0]), "kernel": [int(w.shape[2]), int(w.shape[3])], "stride": stride, "padding": padding})
    W[name + ".W"], W[name + ".b"] = w, b

def dense(name, inp, w, b):
    w = np.asarray(w, np.float32); b = np.asarray(b, np.float32)
    L.append({"name": name, "op": "dense", "inputs": [inp], "out": int(w.shape[0])})
    W[name + ".W"], W[name + ".b"] = w, b

def relu(name, inp): L.append({"name": name, "op": "relu", "inputs": [inp]})
def pool(name, inp, op, size, stride): L.append({"name": name, "op": op, "inputs": [inp], "size": list(size), "stride": list(stride)})

# 1. colour at half resolution (JPEG chroma is unreliable at single pixels)
pool("p48", "image", "avgpool", (2, 2), (2, 2))

# 2. skin-colour cone, written as homogeneous linear inequalities on R, G, B (so it is brightness invariant).
#    With a = R-G and b = G-B, skin of all tones has R > G > B, hue about 8..30 degrees, saturation about 0.18..0.68.
#    Each row is the VIOLATION of one inequality (positive = outside the cone); last row passes R through.
forms = [[-0.90, 1.0, 0.0],    # R-G >= 0.10 R
         [0.20, -1.2, 1.0],    # G-B >= 0.2 (R-G)   (not pink / magenta)
         [-1.0, 2.0, -1.0],    # G-B <= R-G         (hue <= 30 deg: not yellow, wood, sand)
         [0.32, 0.0, -1.0],    # R-B <= 0.68 R      (not saturated orange / red objects)
         [-0.82, 0.0, 1.0],    # R-B >= 0.18 R      (not grey / white)
         [1.0, 0.0, 0.0]]      # R itself (brightness)
conv("c1", "p48", np.array(forms).reshape(6, 3, 1, 1), np.zeros(6))
relu("r1", "c1")

# 3. soft skin membership x = 5 (R - 0.15) - 25 * (sum of violations); S = clip(x, 0, 1) = relu(x) - relu(x - 1)
row = np.array([-25.0] * 5 + [5.0])
conv("c2", "r1", np.stack([row, row]).reshape(2, 6, 1, 1), [-0.75, -1.75])
relu("r2", "c2")
conv("S48", "r2", np.array([1.0, -1.0]).reshape(1, 2, 1, 1), [0.0])
pool("S24", "S48", "avgpool", (2, 2), (2, 2))
pool("S12", "S24", "avgpool", (2, 2), (2, 2))

# 4. compact skin blob = centre-surround (difference of Gaussians) on the skin map, at three pyramid levels.
#    Balanced DoG, sigma 1.0 / 2.5: a straight edge of a big skin-coloured area answers about 0.21, a corner 0.35,
#    a limb-like bar 0.42, an isolated disc 0.59 -> threshold 0.30 keeps blobs and limbs, drops walls and tables.
r = 6
yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
def gauss(s):
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * s * s)); return g / g.sum()
dog = gauss(1.0) - gauss(2.5)
THETA, GAIN = 0.30, 10.0
outs = []
for lvl, size in (("S48", 48), ("S24", 24), ("S12", 12)):
    n = "b" + lvl[1:]
    conv(n, lvl, dog.reshape(1, 1, 13, 13), [-THETA], padding=r)
    relu(n + "r", n)
    pool(n + "m", n + "r", "maxpool", (size, size), (size, size))
    dense(n + "d", n + "m", [[0.0], [GAIN]], [0.0, 0.0])
    outs.append(n + "d")

# 5. scene check: if a large share of the photo is skin-coloured (warm room, food, wood), blobs are less trustworthy.
pool("favg", "S48", "avgpool", (48, 48), (48, 48))
conv("fc", "favg", np.ones((1, 1, 1, 1)), [-0.35])
relu("fr", "fc")
dense("fd", "fr", [[0.0], [-4.0]], [0.0, -0.2])   # -0.2: needs a total blob excess above 0.02 to say "person"
outs.append("fd")

L.append({"name": "scores", "op": "add", "inputs": outs})

json.dump({"layers": L}, open(os.path.join(out, "network.json"), "w"), indent=1)
np.savez(os.path.join(out, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
