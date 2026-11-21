"""Writes network.json and weights.npz for a hand-designed 32x32 photo classifier.
Reads no files. Every number is typed from general knowledge of the classes:
soft colour indicators (sky/water blue, vegetation green, warm fur/earth, paint red, bright, dark)
and Sobel edge energies, averaged on a 4x4 grid, then a hand-typed rule table per class."""
import sys, json
from pathlib import Path
import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)

# ---- layer 1: 3x3 conv, 10 channels: 6 colour indicators (centre tap only) + 4 signed Sobel edges on luminance
W1 = np.zeros((10, 3, 3, 3), np.float64); b1 = np.zeros(10, np.float64)
def colour(ch, rgb, bias, gain):
    W1[ch, :, 1, 1] = np.array(rgb) * gain; b1[ch] = bias * gain
colour(0, [-0.5, -0.5, 1.0], -0.02, 4.0)        # blue: sky, water
colour(1, [-0.5, 1.0, -0.5], -0.02, 5.0)        # green: grass, foliage
colour(2, [1.0, 0.0, -1.0], -0.05, 3.0)         # warm: brown, tan, orange, yellow
colour(3, [1.0, -1.0, 0.0], -0.05, 4.0)         # red: paint
colour(4, [1 / 3, 1 / 3, 1 / 3], -0.70, 3.3)    # bright: white sky, white paint
colour(5, [-1 / 3, -1 / 3, -1 / 3], 0.30, 3.3)  # dark: shadow, tyres, windows
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64) / 4.0   # responds to vertical edges
sy = sx.T                                                              # responds to horizontal edges
for ch, k in ((6, sx), (7, -sx), (8, sy), (9, -sy)):
    W1[ch] = 8.0 * k[None] / 3.0; b1[ch] = -0.02 * 8.0

# ---- rule table: class -> feature -> (weights for the 4 rows top..bottom, side-column factor, centre-column factor)
R = lambda v: [v, v, v, v]
T = {
 0: dict(blue=([1.2, 1.0, 1.0, 1.0], 1.2, 0.8), green=(R(-1.0), 1, 1), warm=(R(-0.8), 1, 1), red=(R(-0.3), 1, 1),
         bright=([0.6, 0.5, 0.5, 0.5], 1.2, 0.6), dark=(R(-0.3), 1, 1), vedge=(R(-0.6), 1.3, 0.5),
         hedge=([-0.4, -0.2, -0.2, -0.4], 1.3, 0.3), bias=0.0),                       # airplane: smooth sky all round
 1: dict(blue=([-0.3, -0.3, -0.4, -0.5], 1, 1), green=(R(-0.6), 1, 1), warm=(R(-0.1), 1, 1),
         red=([0.2, 0.8, 0.8, 0.2], 0.7, 1.3), bright=([0.1, 0.1, 0.0, 0.0], 1, 1), dark=([-0.1, 0.1, 0.4, 0.5], 1, 1),
         vedge=([0.0, 0.1, 0.2, 0.1], 1, 1), hedge=([0.0, 0.4, 0.6, 0.5], 1, 1), bias=0.3),  # car: low, long, dark underside
 2: dict(blue=([0.4, 0.4, 0.3, 0.2], 1, 1), green=([0.4, 0.4, 0.4, 0.3], 1, 1), warm=([0.1, 0.2, 0.2, 0.1], 1, 1),
         red=(R(-0.3), 1, 1), bright=(R(0.1), 1, 1), dark=(R(-0.1), 1, 1), vedge=(R(-0.4), 1, 1), hedge=(R(-0.4), 1, 1),
         bias=0.1),                                                                    # bird: small thing, blurred backdrop
 3: dict(blue=(R(-0.8), 1, 1), green=(R(-0.7), 1, 1), warm=([0.3, 0.4, 0.4, 0.3], 1, 1), red=(R(0.1), 1, 1),
         bright=(R(0.0), 1, 1), dark=(R(0.2), 1, 1), vedge=(R(0.2), 0.5, 1.5), hedge=(R(0.1), 1, 1), bias=0.3),  # cat: indoor
 4: dict(blue=(R(-0.8), 1, 1), green=([1.0, 0.9, 0.9, 1.0], 1.2, 0.8), warm=([0.2, 0.5, 0.5, 0.3], 0.7, 1.3),
         red=(R(-0.2), 1, 1), bright=(R(-0.4), 1, 1), dark=(R(0.2), 1, 1), vedge=([0.0, 0.0, 0.1, 0.1], 1, 1),
         hedge=(R(-0.2), 1, 1), bias=0.0),                                             # deer: tan body in green/brown woods
 5: dict(blue=(R(-0.7), 1, 1), green=([-0.2, -0.2, 0.0, 0.2], 1, 1), warm=([0.4, 0.5, 0.5, 0.4], 0.7, 1.3),
         red=(R(0.1), 1, 1), bright=([0.1, 0.3, 0.3, 0.1], 0.5, 1.5), dark=(R(0.0), 1, 1), vedge=(R(0.0), 1, 1),
         hedge=(R(0.0), 1, 1), bias=0.2),                                              # dog: warm or white fur in centre
 6: dict(blue=(R(-1.2), 1, 1), green=(R(0.6), 1, 1), warm=(R(0.7), 1, 1), red=(R(-0.3), 1, 1), bright=(R(-0.5), 1, 1),
         dark=(R(0.2), 1, 1), vedge=(R(-0.2), 1, 1), hedge=(R(0.0), 1, 1), bias=0.0),  # frog: olive/yellow-brown close-up
 7: dict(blue=([0.5, 0.1, -0.4, -0.6], 1, 1), green=([-0.1, 0.2, 0.7, 1.0], 1, 1), warm=([0.1, 0.5, 0.5, 0.3], 0.7, 1.3),
         red=(R(0.1), 1, 1), bright=([0.2, 0.0, -0.2, -0.2], 1, 1), dark=([0.0, 0.2, 0.2, 0.0], 1, 1),
         vedge=([0.0, 0.1, 0.4, 0.5], 1, 1), hedge=(R(-0.1), 1, 1), bias=0.0),         # horse: sky above, grass below, legs
 8: dict(blue=([0.6, 0.8, 1.3, 1.5], 1.1, 0.9), green=(R(-0.8), 1, 1), warm=(R(-0.6), 1, 1), red=(R(-0.2), 1, 1),
         bright=([0.3, 0.3, 0.0, -0.2], 1, 1), dark=([-0.2, 0.0, 0.1, 0.1], 1, 1), vedge=([0.1, 0.2, 0.0, -0.3], 1, 1),
         hedge=([-0.2, 0.5, 0.6, 0.0], 1, 1), bias=0.0),                               # ship: water below, hull line in middle
 9: dict(blue=([0.0, -0.3, -0.5, -0.6], 1, 1), green=(R(-0.5), 1, 1), warm=([0.0, 0.1, 0.1, 0.0], 1, 1),
         red=([0.3, 0.6, 0.6, 0.2], 1, 1), bright=([0.5, 0.3, 0.1, 0.0], 1, 1), dark=([-0.1, 0.0, 0.2, 0.4], 1, 1),
         vedge=([0.3, 0.5, 0.5, 0.3], 1, 1), hedge=([0.3, 0.4, 0.5, 0.4], 1, 1), bias=0.1),  # truck: tall boxy, fills frame
}
CH = dict(blue=[0], green=[1], warm=[2], red=[3], bright=[4], dark=[5], vedge=[6, 7], hedge=[8, 9])

W2 = np.zeros((10, 10, 4, 4), np.float64); b2 = np.zeros(10, np.float64)
for k, rules in T.items():
    b2[k] = rules["bias"]
    for feat, chans in CH.items():
        rows, side, centre = rules[feat]
        for r in range(4):
            for c in range(4):
                for ch in chans:
                    W2[k, ch, r, c] = rows[r] * (centre if c in (1, 2) else side) / 16.0

net = {"layers": [
    {"name": "feat", "op": "conv", "inputs": ["image"], "out": 10, "kernel": [3, 3], "stride": 1, "padding": 1},
    {"name": "feat_relu", "op": "relu", "inputs": ["feat"]},
    {"name": "grid", "op": "avgpool", "inputs": ["feat_relu"], "size": [8, 8], "stride": [8, 8]},
    {"name": "scores", "op": "dense", "inputs": ["grid"], "out": 10},
]}
(out / "network.json").write_text(json.dumps(net, indent=1))
np.savez(out / "weights.npz", **{"feat.W": W1.astype(np.float32), "feat.b": b1.astype(np.float32),
                                 "scores.W": W2.reshape(10, -1).astype(np.float32), "scores.b": b2.astype(np.float32)})
