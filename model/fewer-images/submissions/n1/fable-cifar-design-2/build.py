"""Writes network.json and weights.npz for a hand-designed 10-class photo classifier.
Every number comes from this file: formula filters (colour-opponent axes, Sobel derivatives) and a
hand-typed table of what each class's scene usually looks like (sky, water, grass, road, fur colours).
No data is read, nothing is fitted.  Usage: python3 build.py OUT_DIR
"""
import json, sys, os
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------- layer 1: 3x3 conv, 10 channels
# 0/1  +-(luminance - 0.5)        2/3  +-(R - G)        4/5  +-(B - (R+G)/2)
# 6/7  +-d(luminance)/dx (Sobel)  8/9  +-d(luminance)/dy (Sobel)
W1 = np.zeros((10, 3, 3, 3)); b1 = np.zeros(10)
lum = np.array([1 / 3, 1 / 3, 1 / 3]); rg = np.array([1.0, -1.0, 0.0]); by = np.array([-0.5, -0.5, 1.0])
for k, v in enumerate([lum, rg, by]):
    W1[2 * k, :, 1, 1] = v; W1[2 * k + 1, :, 1, 1] = -v
b1[0], b1[1] = -0.5, 0.5
sob_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
sob_y = sob_x.T
for c in range(3):
    W1[6, c] = sob_x / 3; W1[7, c] = -sob_x / 3
    W1[8, c] = sob_y / 3; W1[9, c] = -sob_y / 3

# ---------------------------------------------------------------- layer 2: 1x1 conv, 5 feature maps
# L, RG, BY (signed again), Ex = |dL/dx| (vertical edges), Ey = |dL/dy| (horizontal edges),
# each minus a hand-chosen "ordinary photo" level so that 0 means nothing special.
NEUTRAL = np.array([-0.02, 0.01, -0.04, 0.03, 0.03])   # photos are slightly dark and warm; mild edge energy
SIGMA = np.array([0.20, 0.07, 0.12, 0.02, 0.02])       # hand-chosen spread of each feature
W2 = np.zeros((5, 10, 1, 1)); b2 = -NEUTRAL.copy()
W2[0, 0] = 1; W2[0, 1] = -1
W2[1, 2] = 1; W2[1, 3] = -1
W2[2, 4] = 1; W2[2, 5] = -1
W2[3, 6] = 1; W2[3, 7] = 1
W2[4, 8] = 1; W2[4, 9] = 1

# ---------------------------------------------------------------- readout: hand-typed scene table
# After 7x7 average pooling the image is a 4x4 grid of cells.  For each class I type how far each
# feature is expected to sit from the ordinary-photo level in each part of the picture.
L, RG, BY, EX, EY = range(5)
ROWS = {"top": [0], "upper": [1], "lower": [2], "bottom": [3], "mid": [1, 2], "all": [0, 1, 2, 3]}
COLS = {"centre": [1, 2], "sides": [0, 3], "all": [0, 1, 2, 3]}

def table(entries):
    d = np.zeros((5, 4, 4))
    for rows, cols, ch, val in entries:
        for r in ROWS[rows]:
            for c in COLS[cols]:
                d[ch, r, c] = val
    return d

CLASS = {}
# airplane: pale blue or white sky all round, smooth border, a light horizontal body in the middle
CLASS[0] = table([
    ("all", "all", L, 0.10), ("all", "all", BY, 0.08), ("all", "all", RG, -0.02),
    ("top", "all", L, 0.15), ("top", "all", BY, 0.12), ("top", "all", RG, -0.03),
    ("mid", "sides", L, 0.12), ("mid", "sides", BY, 0.10),
    ("mid", "centre", L, 0.05), ("mid", "centre", BY, 0.05),
    ("bottom", "all", L, 0.08), ("bottom", "all", BY, 0.07),
    ("top", "all", EX, -0.02), ("top", "all", EY, -0.02), ("bottom", "all", EX, -0.015), ("bottom", "all", EY, -0.015),
    ("mid", "sides", EX, -0.015), ("mid", "sides", EY, -0.01), ("mid", "centre", EY, 0.01),
])
# automobile: grey road below, coloured (often red) body in the middle, long horizontal body lines
CLASS[1] = table([
    ("top", "all", L, -0.03), ("top", "all", BY, 0.01),
    ("mid", "centre", RG, 0.04), ("mid", "centre", BY, 0.01), ("mid", "sides", BY, 0.02),
    ("bottom", "all", L, 0.03), ("bottom", "all", BY, 0.04), ("bottom", "all", RG, 0.0),
    ("mid", "all", EY, 0.015), ("lower", "centre", EY, 0.02), ("mid", "centre", EX, 0.01),
    ("bottom", "all", EX, -0.005),
])
# bird: soft blurred background (sky or foliage), small darker object in the middle
CLASS[2] = table([
    ("all", "all", L, 0.02), ("all", "all", RG, -0.015), ("all", "all", BY, -0.01),
    ("top", "all", BY, 0.02), ("top", "all", L, 0.04),
    ("mid", "centre", L, -0.03),
    ("top", "all", EX, -0.01), ("top", "all", EY, -0.01), ("mid", "sides", EX, -0.01), ("mid", "sides", EY, -0.01),
    ("bottom", "all", EY, -0.005),
])
# cat: indoor warm greys and browns, animal fills the frame, few long horizontal lines
CLASS[3] = table([
    ("all", "all", RG, 0.03), ("all", "all", BY, -0.04), ("all", "all", L, -0.03),
    ("mid", "centre", L, 0.01), ("mid", "centre", RG, 0.035),
    ("all", "all", EY, -0.005), ("mid", "centre", EX, 0.005),
])
# deer: green-brown woodland or meadow all round, brown body in the middle, rather dark
CLASS[4] = table([
    ("all", "all", L, -0.05), ("all", "all", RG, -0.02), ("all", "all", BY, -0.08),
    ("mid", "centre", RG, 0.02), ("mid", "centre", BY, -0.07), ("mid", "centre", L, -0.04),
    ("bottom", "all", RG, -0.03), ("bottom", "all", BY, -0.09),
    ("lower", "centre", EX, 0.005),
])
# dog: like cat but more often outdoors and lighter; pale or tan body in the middle
CLASS[5] = table([
    ("all", "all", RG, 0.025), ("all", "all", BY, -0.045), ("all", "all", L, 0.0),
    ("mid", "centre", L, 0.05), ("mid", "centre", RG, 0.03),
    ("bottom", "all", RG, 0.0), ("bottom", "all", BY, -0.06),
])
# frog: close-up, green/olive/brown everywhere, very little blue, dark
CLASS[6] = table([
    ("all", "all", L, -0.08), ("all", "all", RG, -0.02), ("all", "all", BY, -0.12),
    ("mid", "centre", RG, -0.03), ("mid", "centre", BY, -0.13), ("mid", "centre", L, -0.06),
    ("all", "all", EY, -0.003),
])
# horse: bright sky or trees above, dark brown body in the middle, grass below, legs = vertical edges low down
CLASS[7] = table([
    ("top", "all", L, 0.08), ("top", "all", BY, 0.0), ("top", "all", RG, -0.02),
    ("upper", "all", L, 0.0), ("upper", "all", BY, -0.04),
    ("mid", "centre", L, -0.08), ("mid", "centre", RG, 0.04), ("mid", "centre", BY, -0.06),
    ("bottom", "all", L, -0.02), ("bottom", "all", RG, -0.04), ("bottom", "all", BY, -0.10),
    ("lower", "centre", EX, 0.015), ("bottom", "centre", EX, 0.012),
])
# ship: bright sky above, darker blue water below, horizon and hull = horizontal edges in the middle
CLASS[8] = table([
    ("top", "all", L, 0.12), ("top", "all", BY, 0.10), ("top", "all", RG, -0.02),
    ("upper", "all", L, 0.06), ("upper", "all", BY, 0.07),
    ("lower", "all", L, -0.02), ("lower", "all", BY, 0.06), ("lower", "all", RG, -0.02),
    ("bottom", "all", L, -0.08), ("bottom", "all", BY, 0.10), ("bottom", "all", RG, -0.03),
    ("mid", "all", EY, 0.012), ("top", "all", EX, -0.012), ("top", "all", EY, -0.012),
    ("bottom", "all", EX, -0.008),
])
# truck: tall boxy body filling the frame, often white or red, sky above, grey road below
CLASS[9] = table([
    ("top", "all", L, 0.07), ("top", "all", BY, 0.04),
    ("mid", "centre", L, 0.05), ("mid", "centre", RG, 0.03), ("mid", "centre", BY, 0.0),
    ("mid", "sides", L, 0.02), ("mid", "sides", BY, 0.02),
    ("bottom", "all", L, -0.01), ("bottom", "all", BY, 0.03),
    ("mid", "centre", EX, 0.02), ("mid", "all", EY, 0.015), ("upper", "centre", EY, 0.02),
])

# Diagonal-Gaussian (naive Bayes style) readout typed by hand: score = sum d*f/s^2 - d^2/(2 s^2)
inv_var = (1.0 / SIGMA ** 2)[:, None, None]
Wd = np.zeros((10, 5 * 4 * 4)); bd = np.zeros(10)
for k in range(10):
    d = CLASS[k]
    Wd[k] = (d * inv_var).reshape(-1)
    bd[k] = -0.5 * float((d * d * inv_var).sum())
SCALE = 1.0 / 16.0     # cells in one feature map are strongly correlated; one global factor, no effect on argmax
Wd *= SCALE; bd *= SCALE

net = {"layers": [
    {"name": "c1", "op": "conv", "inputs": ["image"], "out": 10, "kernel": [3, 3], "stride": 1, "padding": 0},
    {"name": "r1", "op": "relu", "inputs": ["c1"]},
    {"name": "c2", "op": "conv", "inputs": ["r1"], "out": 5, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "p", "op": "avgpool", "inputs": ["c2"], "size": [7, 7], "stride": [7, 7]},
    {"name": "out", "op": "dense", "inputs": ["p"], "out": 10},
]}
with open(os.path.join(out_dir, "network.json"), "w") as f:
    json.dump(net, f, indent=1)
np.savez(os.path.join(out_dir, "weights.npz"), **{
    "c1.W": W1.astype(np.float32), "c1.b": b1.astype(np.float32),
    "c2.W": W2.astype(np.float32), "c2.b": b2.astype(np.float32),
    "out.W": Wd.astype(np.float32), "out.b": bd.astype(np.float32),
})
