"""Hand-designed person / no-person network.  Every number below is typed by hand from general knowledge
(published skin-colour rules, face proportions); nothing is read from or fitted to any image.

Pipeline
  v, m   : 1x1 convs + relu turn RGB into two soft colour maps in [0, 1]:
             skin  (R > G >= B, moderate saturation, hue between red and orange, bright enough)
             sky   (clearly blue, bright)
           Each map is relu(1 - 10 * sum of rule violations), so it is 1 inside the colour cone and fades outside.
  pA..pC : average-pool pyramid (cells of 2, 4 and 8 pixels).
  bA..bC : at each pyramid level two centre-surround "face blob" templates on the skin map: a Gaussian centre
           (the face, slightly taller than wide) minus a ring over the top and both sides (hair / background must
           not be skin; the row below is left free for neck and arms).  Response is in [-1, 1]; relu(resp - 0.50),
           then a global max gives "strongest isolated skin blob" for face widths of roughly 4 to 36 pixels.
  g      : global fractions of skin-coloured and sky-coloured pixels; a photo that is mostly skin-coloured
           (wood, food, sand) or mostly sky counts against "person".
  out    : person score = sum of blob evidence (finest level at half weight) - 3*relu(skin fraction - 0.25)
           - 1.5*relu(sky fraction - 0.40) - 0.20
           no-person score = 0.
"""
import json
import sys
from pathlib import Path

import numpy as np

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out_dir.mkdir(parents=True, exist_ok=True)
f32 = np.float32

# ---------------------------------------------------------------- colour rules (1x1 convs)
# rows: (coefficient of R, of G, of B, constant); each is a violation amount that should be <= 0
skin_rules = [
    (-1.0, 1.0, 0.0, 0.05),     # R - G >= 0.05      (red dominates green)
    (1.0, -1.0, 0.0, -0.38),    # R - G <= 0.38      (not pure red)
    (0.0, -1.0, 1.0, -0.03),    # B <= G + 0.03      (not purple / pink-blue)
    (-1.0, 0.0, 0.0, 0.35),     # R >= 0.35          (not too dark)
    (-0.62, 1.0, -0.38, 0.0),   # (G-B) <= 0.62(R-B) (hue below about 37 degrees: not yellow)
    (0.30, 0.0, -1.0, 0.0),     # B >= 0.30 R        (saturation below 0.70)
    (-0.85, 0.0, 1.0, 0.0),     # B <= 0.85 R        (saturation above 0.15: not grey / white)
]
sky_rules = [
    (1.0, 0.0, -1.0, 0.10),     # B - R >= 0.10
    (0.0, 1.0, -1.0, -0.03),    # G <= B + 0.03
    (0.0, 0.0, -1.0, 0.45),     # B >= 0.45
]
rules = skin_rules + sky_rules
nv = len(rules)
vW = np.zeros((nv, 3, 1, 1), f32)
vb = np.zeros((nv,), f32)
for i, (r, g, b, c) in enumerate(rules):
    vW[i, :, 0, 0] = (r, g, b)
    vb[i] = c

GAIN = 10.0
mW = np.zeros((2, nv, 1, 1), f32)
mW[0, :len(skin_rules), 0, 0] = -GAIN
mW[1, len(skin_rules):, 0, 0] = -GAIN
mb = np.ones((2,), f32)


# ---------------------------------------------------------------- face-blob templates
def blob_kernel(ring, sigma):
    """7x7 kernel: Gaussian centre (sum 1) minus a ring over the top and both sides (sum 1)."""
    K = np.zeros((7, 7), np.float64)
    cen = np.zeros((7, 7), np.float64)
    sur = np.zeros((7, 7), np.float64)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if max(abs(dy), abs(dx)) < ring:
                cen[dy + 3, dx + 3] = np.exp(-(dx * dx) / (2 * sigma * sigma) - (dy * dy) / (2 * (1.25 * sigma) ** 2))
            elif max(abs(dy), abs(dx)) == ring and dy < ring:
                sur[dy + 3, dx + 3] = 1.0
    K = cen / cen.sum() - sur / sur.sum()
    return K


THETA = 0.50
bW = np.zeros((2, 2, 7, 7), f32)
bW[0, 0] = blob_kernel(2, 0.8)     # face about 2-3 cells wide
bW[1, 0] = blob_kernel(3, 1.3)     # face about 3.5-4.5 cells wide
bb = np.full((2,), -THETA, f32)

dW = np.zeros((2, 2), f32)
dW[1, :] = 1.0                     # blob evidence votes for "person"
db = np.zeros((2,), f32)

# ---------------------------------------------------------------- global colour fractions
gW = np.zeros((2, 2, 1, 1), f32)
gW[0, 0, 0, 0] = 1.0
gW[1, 1, 0, 0] = 1.0
gb = np.array([-0.25, -0.40], f32)
dGW = np.zeros((2, 2), f32)
dGW[1, 0] = -3.0                   # photo mostly skin-coloured: wood, food, sand
dGW[1, 1] = -1.5                   # photo mostly sky
dGb = np.array([0.0, -0.20], f32)  # evidence needed before saying "person"

weights = {"v.W": vW, "v.b": vb, "m.W": mW, "m.b": mb, "gc.W": gW, "gc.b": gb, "dG.W": dGW, "dG.b": dGb}
layers = [
    {"name": "v", "op": "conv", "inputs": ["image"], "out": nv, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "vr", "op": "relu", "inputs": ["v"]},
    {"name": "m", "op": "conv", "inputs": ["vr"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "mr", "op": "relu", "inputs": ["m"]},
    {"name": "pA", "op": "avgpool", "inputs": ["mr"], "size": [2, 2], "stride": [2, 2]},
    {"name": "pB", "op": "avgpool", "inputs": ["pA"], "size": [2, 2], "stride": [2, 2]},
    {"name": "pC", "op": "avgpool", "inputs": ["pB"], "size": [2, 2], "stride": [2, 2]},
]
for lev, side in (("A", 48), ("B", 24), ("C", 12)):
    layers += [
        {"name": "b" + lev, "op": "conv", "inputs": ["p" + lev], "out": 2, "kernel": [7, 7], "stride": 1, "padding": 3},
        {"name": "b" + lev + "r", "op": "relu", "inputs": ["b" + lev]},
        {"name": "b" + lev + "m", "op": "maxpool", "inputs": ["b" + lev + "r"], "size": [side, side], "stride": [side, side]},
        {"name": "d" + lev, "op": "dense", "inputs": ["b" + lev + "m"], "out": 2},
    ]
    weights["b" + lev + ".W"] = bW.copy()
    weights["b" + lev + ".b"] = bb.copy()
    weights["d" + lev + ".W"] = dW.copy() * (0.5 if lev == "A" else 1.0)   # specks of 4-6 pixels are weak evidence
    weights["d" + lev + ".b"] = db.copy()
layers += [
    {"name": "g", "op": "avgpool", "inputs": ["pC"], "size": [12, 12], "stride": [12, 12]},
    {"name": "gc", "op": "conv", "inputs": ["g"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "gr", "op": "relu", "inputs": ["gc"]},
    {"name": "dG", "op": "dense", "inputs": ["gr"], "out": 2},
    {"name": "out", "op": "add", "inputs": ["dA", "dB", "dC", "dG"]},
]

(out_dir / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
np.savez(out_dir / "weights.npz", **{k: v.astype(f32) for k, v in weights.items()})
