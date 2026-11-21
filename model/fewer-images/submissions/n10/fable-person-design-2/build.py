"""Hand-designed person / no-person network. Every number comes from this file (no data is read).

Idea: a per-pixel skin-colour gate (linear ratio rules in RGB, typed from general knowledge of skin tones:
red > green > blue, hue roughly 5-32 degrees, moderate saturation, not dark), followed by centre-surround
"isolated skin blob" detectors (an upright ellipse of skin with a non-skin ring around it) at six scales,
a global max over positions per scale, a hand-set soft threshold per scale, and a vote count.
"""
import sys, json
from pathlib import Path
import numpy as np

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out_dir.mkdir(parents=True, exist_ok=True)
layers, W = [], {}


def conv(name, src, w, b, stride=1, padding=0):
    w = np.asarray(w, np.float32); b = np.asarray(b, np.float32)
    layers.append({"name": name, "op": "conv", "inputs": [src], "out": int(w.shape[0]),
                   "kernel": [int(w.shape[2]), int(w.shape[3])], "stride": stride, "padding": padding})
    W[name + ".W"], W[name + ".b"] = w, b
    return name


def dense(name, src, w, b):
    w = np.asarray(w, np.float32); b = np.asarray(b, np.float32)
    layers.append({"name": name, "op": "dense", "inputs": [src], "out": int(w.shape[0])})
    W[name + ".W"], W[name + ".b"] = w, b
    return name


def relu(name, src):
    layers.append({"name": name, "op": "relu", "inputs": [src]}); return name


def pool(name, src, op, size, stride):
    layers.append({"name": name, "op": op, "inputs": [src], "size": list(size), "stride": list(stride)}); return name


# ---- 1. skin gate: violations of linear ratio rules (rows are coefficients of R, G, B, constant)
rules = [
    (-1.0, 1.12, 0.0, 0.0),    # R/G >= 1.12  (redder than green)
    (1.0, -1.8, 0.0, 0.0),     # R/G <= 1.8   (not pure red / orange paint)
    (0.0, -1.0, 1.0, 0.0),     # G >= B
    (0.0, 1.0, -1.85, 0.0),    # G/B <= 1.85  (not too yellow / saturated)
    (-1.15, 2.15, -1.0, 0.0),  # (G-B) <= 1.15 (R-G): hue below about 32 degrees (wood, sand, straw are yellower)
    (-1.0, 0.0, 0.0, 0.30),    # R >= 0.30    (not dark)
]
w1 = np.zeros((len(rules), 3, 1, 1)); b1 = np.zeros(len(rules))
for i, (r, g, b, c) in enumerate(rules):
    w1[i, :, 0, 0] = (r, g, b); b1[i] = c
conv("viol", "image", w1, b1); relu("viol_r", "viol")
GATE_SLOPE = 20.0  # membership = 1 - 20 * total violation, clipped at 0
conv("skin", "viol_r", -GATE_SLOPE * np.ones((1, len(rules), 1, 1)), [1.0]); relu("skin_r", "skin")

# ---- 2. pyramid of the skin map
pool("s48", "skin_r", "avgpool", (2, 2), (2, 2))
pool("s24", "s48", "avgpool", (2, 2), (2, 2))
pool("s12", "s24", "avgpool", (2, 2), (2, 2))


# ---- 3. centre-surround blob kernels: upright ellipse of skin, ring of non-skin (neck/body sector below ignored)
def blob_kernel(ax, ay, k=11):
    c = k // 2
    dy, dx = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float64)
    e = np.sqrt((dx / ax) ** 2 + (dy / ay) ** 2)
    centre = e <= 1.0
    ring = (e > 1.5) & (e <= 2.2) & ~((dy > 0) & (np.abs(dx) < 0.8 * ax))
    ker = np.zeros((k, k))
    ker[centre] = 1.0 / centre.sum()
    ker[ring] = -1.0 / ring.sum()
    return ker


kers = np.stack([blob_kernel(1.5, 2.2), blob_kernel(2.0, 2.8)])[:, None]  # (2, 1, 11, 11)
LO, WIDTH = 0.35, 0.30   # a blob response of 0.35 counts as nothing, 0.65 as a full vote
THETA = 1.0              # votes needed (summed over the six scales) to call "person"
votes = []
for lvl, n in (("s48", 48), ("s24", 24), ("s12", 12)):
    conv("blob_" + lvl, lvl, kers, [0.0, 0.0], padding=5); relu("blobr_" + lvl, "blob_" + lvl)
    pool("max_" + lvl, "blobr_" + lvl, "maxpool", (n, n), (n, n))
    dense("x_" + lvl, "max_" + lvl, np.eye(2) / WIDTH, [-LO / WIDTH] * 2); relu("xr_" + lvl, "x_" + lvl)
    dense("t_" + lvl, "xr_" + lvl, -np.eye(2), [1.0, 1.0]); relu("tr_" + lvl, "t_" + lvl)
    votes.append(dense("v_" + lvl, "tr_" + lvl, [[0.0, 0.0], [-1.0, -1.0]], [0.0, 2.0]))  # class 1: sum of clip(x, 0, 1)
layers.append({"name": "votes", "op": "add", "inputs": votes})
dense("scores", "votes", np.eye(2), [0.0, -THETA])

(out_dir / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
np.savez(out_dir / "weights.npz", **W)
