"""Hand-written person / no-person network. Every number is typed from general knowledge; no data is read."""
import json, sys, os
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)

layers = [
    {"name": "c1", "op": "conv", "inputs": ["image"], "out": 5, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "r1", "op": "relu", "inputs": ["c1"]},
    {"name": "c2", "op": "conv", "inputs": ["r1"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "r2", "op": "relu", "inputs": ["c2"]},
    {"name": "p1", "op": "avgpool", "inputs": ["r2"], "size": [4, 4], "stride": [4, 4]},
    {"name": "c3", "op": "conv", "inputs": ["p1"], "out": 2, "kernel": [5, 5], "stride": 1, "padding": 2},
    {"name": "r3", "op": "relu", "inputs": ["c3"]},
    {"name": "p2", "op": "maxpool", "inputs": ["r3"], "size": [24, 24], "stride": [24, 24]},
    {"name": "fc", "op": "dense", "inputs": ["p2"], "out": 2},
]

W = {}
# c1: violations of a skin-colour rule (R > G > B, moderate differences, not dark).
c1 = np.zeros((5, 3, 1, 1), np.float32)
b1 = np.zeros(5, np.float32)
c1[0, :, 0, 0] = [-1, 1, 0]; b1[0] = 0.06    # R-G too small
c1[1, :, 0, 0] = [1, -1, 0]; b1[1] = -0.30   # R-G too large (pure red objects)
c1[2, :, 0, 0] = [0, -1, 1]; b1[2] = 0.02    # G-B too small
c1[3, :, 0, 0] = [0, 1, -1]; b1[3] = -0.25   # G-B too large (yellow / orange objects)
c1[4, :, 0, 0] = [-1, 0, 0]; b1[4] = 0.30    # too dark
W["c1.W"], W["c1.b"] = c1, b1
# c2: skin-ness = 1 - weighted violations, clipped at 0 by the relu.
c2 = np.zeros((1, 5, 1, 1), np.float32)
c2[0, :, 0, 0] = [-12, -12, -12, -12, -6]
W["c2.W"], W["c2.b"] = c2, np.array([1.0], np.float32)
# c3: centre-surround blob detectors on the 24 x 24 skin map (a face or arm is a compact skin blob,
# while wood, sand or warm lighting fill large areas and cancel out).
c3 = np.zeros((2, 1, 5, 5), np.float32)
c3[0, 0] = -1.0 / 16.0
c3[0, 0, 1:4, 1:4] = 1.0 / 9.0               # 3x3 centre minus 5x5 ring
c3[1, 0, 1:4, 1:4] = -1.0 / 8.0
c3[1, 0, 2, 2] = 1.0                         # single cell minus its 3x3 ring
W["c3.W"], W["c3.b"] = c3, np.zeros(2, np.float32)
# fc: person if the strongest skin blobs are contrasted enough.
fc = np.zeros((2, 2), np.float32)
fc[1] = [1.0, 1.0]
W["fc.W"], W["fc.b"] = fc, np.array([0.0, -0.45], np.float32)

with open(os.path.join(out, "network.json"), "w") as f:
    json.dump({"layers": layers}, f, indent=1)
np.savez(os.path.join(out, "weights.npz"), **W)
