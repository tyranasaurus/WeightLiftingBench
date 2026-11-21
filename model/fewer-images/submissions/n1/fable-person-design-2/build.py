"""Writes network.json and weights.npz for a hand-designed person / no-person network.
Every number is typed from general knowledge of skin colour and face-sized blobs; nothing is fitted and no file is read."""
import json, sys, numpy as np
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
L, W = [], {}

# 1) Skin-colour membership: a cone in RGB given by linear limits (R > G > B, hue about 3-39 degrees,
#    moderate saturation, not too dark). Each row is one violated-limit amount: w . (R, G, B) + b > 0 means "outside".
viol = [
    ([-1.00, 1.00, 0.00], 0.05),   # R must exceed G by 0.05
    ([0.05, -1.00, 0.95], 0.00),   # hue lower limit: (G-B) >= 0.05 (R-B)
    ([-0.65, 1.00, -0.35], 0.00),  # hue upper limit: (G-B) <= 0.65 (R-B)
    ([-0.85, 0.00, 1.00], 0.00),   # saturation at least 0.15
    ([0.28, 0.00, -1.00], 0.00),   # saturation at most 0.72
    ([-1.00, 0.00, 0.00], 0.33),   # R at least 0.33 (not too dark)
]
K = len(viol)
W["c1.W"] = np.array([v[0] for v in viol], np.float32).reshape(K, 3, 1, 1)
W["c1.b"] = np.array([v[1] for v in viol], np.float32)
L += [{"name": "c1", "op": "conv", "inputs": ["image"], "out": K, "kernel": [1, 1], "stride": 1, "padding": 0},
      {"name": "r1", "op": "relu", "inputs": ["c1"]}]
GAIN = 10.0                        # a violation of 0.1 switches the pixel off completely
W["c2.W"] = np.full((1, K, 1, 1), -GAIN, np.float32)
W["c2.b"] = np.array([1.0], np.float32)
L += [{"name": "c2", "op": "conv", "inputs": ["r1"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
      {"name": "skin", "op": "relu", "inputs": ["c2"]}]

# 2) Compact skin blobs at four sizes: centre 3x3 mean minus the mean of the outer ring of a 7x7 window,
#    on the skin map at full, 1/2, 1/4 and 1/8 resolution. A full blob gives about 1, a straight edge of a large
#    skin-coloured surface (wood, sand) at most 0.5, the inside of a large surface 0. Keep only the part above 0.5.
ker = np.zeros((7, 7), np.float32)
ker[:, :] = -1.0 / 24.0
ker[1:6, 1:6] = 0.0
ker[2:5, 2:5] = 1.0 / 9.0
levels = [("s1", None, 96, 0.5), ("s2", 2, 48, 1.0), ("s4", 4, 24, 1.0), ("s8", 8, 12, 1.0)]
ev = []
for name, f, size, wt in levels:
    src = "skin"
    if f is not None:
        L.append({"name": name + "_dn", "op": "avgpool", "inputs": ["skin"], "size": [f, f], "stride": [f, f]})
        src = name + "_dn"
    W[name + "_blob.W"] = (wt * ker).reshape(1, 1, 7, 7).astype(np.float32)
    W[name + "_blob.b"] = np.array([-0.5 * wt], np.float32)
    L += [{"name": name + "_blob", "op": "conv", "inputs": [src], "out": 1, "kernel": [7, 7], "stride": 1, "padding": 3},
          {"name": name + "_pos", "op": "relu", "inputs": [name + "_blob"]},
          {"name": name + "_max", "op": "maxpool", "inputs": [name + "_pos"], "size": [size, size], "stride": [size, size]}]
    ev.append(name + "_max")

# 3) Penalty when a large share of the picture is skin coloured (wooden rooms, food close-ups, sepia photos).
L.append({"name": "frac", "op": "avgpool", "inputs": ["skin"], "size": [96, 96], "stride": [96, 96]})
W["f1.W"] = np.array([1.0], np.float32).reshape(1, 1, 1, 1); W["f1.b"] = np.array([-0.30], np.float32)
W["f2.W"] = np.array([-2.0], np.float32).reshape(1, 1, 1, 1); W["f2.b"] = np.array([0.0], np.float32)
L += [{"name": "f1", "op": "conv", "inputs": ["frac"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
      {"name": "f1r", "op": "relu", "inputs": ["f1"]},
      {"name": "f2", "op": "conv", "inputs": ["f1r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0}]

# 4) Decision: person when the summed blob evidence minus the penalty exceeds a hand-chosen threshold.
THETA = 0.28
L.append({"name": "evidence", "op": "add", "inputs": ev + ["f2"]})
W["out.W"] = np.array([[0.0], [10.0]], np.float32)
W["out.b"] = np.array([0.0, -10.0 * THETA], np.float32)
L.append({"name": "out", "op": "dense", "inputs": ["evidence"], "out": 2})

(out / "network.json").write_text(json.dumps({"layers": L}, indent=1))
np.savez(out / "weights.npz", **W)
