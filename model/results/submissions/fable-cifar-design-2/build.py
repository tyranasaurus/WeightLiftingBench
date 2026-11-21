"""Hand-designed colour/edge layout network for 10-class 32x32 photos.
Every number below is typed from general knowledge of the classes; nothing is read from data.
Usage: python3 build.py OUT_FOLDER
"""
import json, sys, os
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

# ---------------- layer c1: 3x3 conv, no padding -> 11 channels (30x30) ----------------
# 0 R-G, 1 G-R, 2 G-B, 3 B-G, 4 B-R, 5 R-B (each rectified), 6 luminance,
# 7 +gx, 8 -gx, 9 +gy, 10 -gy  (Sobel/8 derivatives of luminance)
W1 = np.zeros((11, 3, 3, 3)); b1 = np.zeros(11)
R, G, B = 0, 1, 2
for o, (p, n) in enumerate([(R, G), (G, R), (G, B), (B, G), (B, R), (R, B)]):
    W1[o, p, 1, 1] = 1.0; W1[o, n, 1, 1] = -1.0
W1[6, :, 1, 1] = 1.0 / 3.0
sob = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8.0   # d/dx (responds to vertical edges)
for c in range(3):
    W1[7, c] = sob / 3.0; W1[8, c] = -sob / 3.0
    W1[9, c] = sob.T / 3.0; W1[10, c] = -sob.T / 3.0

# ---------------- layer c2: 1x1 conv -> 12 soft "material" maps ----------------
NAMES = ["blue", "green", "red", "warm", "white", "dark", "grey", "hedge", "vedge", "smooth", "sat", "lum"]
W2 = np.zeros((12, 11, 1, 1)); b2 = np.zeros(12)
def setw(o, d, bias=0.0):
    for k, v in d.items(): W2[o, k, 0, 0] = v
    b2[o] = bias
setw(0, {4: 2, 5: -2, 2: -2, 3: 2}, -0.16)              # blue: 4*(B - (R+G)/2 - 0.04)   sky, water
setw(1, {1: 2.5, 0: -2.5, 2: 2.5, 3: -2.5}, -0.10)      # green: 5*(G - (R+B)/2 - 0.02)  vegetation
setw(2, {0: 2, 1: -2, 5: 2, 4: -2}, -0.48)              # red: 4*(R - (G+B)/2 - 0.12)    paint
setw(3, {5: 3, 4: -3}, -0.18)                           # warm: 3*(R - B - 0.06)         fur, earth
setw(4, {6: 4, 0: -3, 1: -3, 2: -3, 3: -3, 4: -3, 5: -3}, -2.6)   # white: 4*(lum - 0.65 - 1.5*sat)
setw(5, {6: -4}, 1.0)                                   # dark: 4*(0.25 - lum)
setw(6, {0: -4, 1: -4, 2: -4, 3: -4, 4: -4, 5: -4}, 0.96)         # grey: 8*(0.12 - sat)
setw(7, {9: 6, 10: 6})                                  # horizontal-edge energy
setw(8, {7: 6, 8: 6})                                   # vertical-edge energy
setw(9, {7: -16, 8: -16, 9: -16, 10: -16}, 0.8)         # smooth: 16*(0.05 - |gx| - |gy|)
setw(10, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})          # saturation*2
setw(11, {6: 1})                                        # luminance

# ---------------- readout: hand-typed evidence table ----------------
# classes: 0 plane, 1 car, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
# each entry: (top row, middle row, bottom row) of a 3x3 grid; a single number means all rows.
T = {
 "blue":   [(2, 1.5, 1.5), (0, -0.3, -0.5), (0.5, 0.3, 0.2), -1, (0, -0.5, -1), -1, -1.5, (0.5, 0, -1), (1.5, 1.5, 2.5), (0.5, 0, -0.5)],
 "green":  [(-1, -1, -0.3), -0.5, 1, -0.5, (1.5, 1.5, 2), (0, 0, 0.5), 1, (0.5, 1, 2), -1.5, -0.5],
 "red":    [-0.5, (0, 2.5, 1), -0.5, 0, -0.5, 0, 0, 0, (-0.5, 0, 0.5), (0.5, 2.5, 1)],
 "warm":   [-1.5, 0, 0.3, 1.0, 0.8, 1.0, 1.2, 0.8, -1.5, 0],
 "white":  [(1, 0.5, 0.5), 0.2, 0, 0.2, -0.5, 0.2, -0.5, (0.3, 0, 0), (0.5, 0.3, -0.5), (0.5, 0.5, 0)],
 "dark":   [-1, (0, 0.2, 1), 0, 0.5, 0.2, 0.5, 0.5, 0.2, (-0.3, -0.3, 0.2), (0, 0.2, 0.8)],
 "grey":   [1, (0.8, 0.8, 1.2), 0, 0.3, -0.8, 0.2, -0.8, 0, 0.5, (0.8, 0.8, 1.2)],
 "hedge":  [(0, 0.5, 0), (0, 1, 1), -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, (0, 1, 0.5), (0.5, 0.8, 0.8)],
 "vedge":  [-0.8, 0.3, -0.3, 0, 0.3, 0, 0, (0, 0.3, 0.5), (0.2, 0, -0.3), 1.2],
 "smooth": [1.5, 0, 1.0, -0.3, -0.2, -0.3, -0.8, 0, 1.0, -0.3],
 "sat":    [0] * 10,
 "lum":    [0] * 10,
}
Wd = np.zeros((10, 12, 3, 3)); bd = np.zeros(10)
for f, name in enumerate(NAMES):
    for k in range(10):
        v = T[name][k]
        rows = v if isinstance(v, tuple) else (v, v, v)
        for r in range(3):
            Wd[k, f, r, :] = rows[r] / 9.0
# centre the evidence over classes so that no feature favours every class at once
Wd -= Wd.mean(axis=0, keepdims=True)
Wd = Wd.reshape(10, 12 * 9)

net = {"layers": [
    {"name": "c1", "op": "conv", "inputs": ["image"], "out": 11, "kernel": [3, 3], "stride": 1, "padding": 0},
    {"name": "r1", "op": "relu", "inputs": ["c1"]},
    {"name": "c2", "op": "conv", "inputs": ["r1"], "out": 12, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "r2", "op": "relu", "inputs": ["c2"]},
    {"name": "pool", "op": "avgpool", "inputs": ["r2"], "size": [10, 10], "stride": [10, 10]},
    {"name": "fc", "op": "dense", "inputs": ["pool"], "out": 10},
]}
with open(os.path.join(out_dir, "network.json"), "w") as fh:
    json.dump(net, fh, indent=1)
f32 = np.float32
np.savez(os.path.join(out_dir, "weights.npz"), **{
    "c1.W": W1.astype(f32), "c1.b": b1.astype(f32),
    "c2.W": W2.astype(f32), "c2.b": b2.astype(f32),
    "fc.W": Wd.astype(f32), "fc.b": bd.astype(f32)})
