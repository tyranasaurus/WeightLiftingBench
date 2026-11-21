#!/usr/bin/env python3
"""Build a hand-designed CIFAR-10 classifier network.

Every number here comes from general knowledge about the ten classes
(handed-down filter design formulas: Gabor bars, Gaussian derivative
edges, double-opponent colour blobs; part detectors like 'dark disc with
textured ring' for a wheel; and a rule table mapping semantic channels
and spatial regions to class scores). No data of any kind is read.

Architecture
------------
image (3,32,32)
  c1  conv 3->16, 5x5, pad 2   : luminance bars (4 orients), fine steps,
                                 colour-opponent blobs (RG, YB, white,
                                 black, green, blue, warm/tan, red),
                                 coarse horizontal bar, coarse sky/ground
  n1  norm ; r1 relu ; p1 maxpool 2x2      -> (16,16,16)
  c2  conv 16->16, 3x3, pad 1  : semantic part/region detectors
                                 (wheel, bright blob, horizontal & vertical
                                 structure, fine structure, organic/diagonal
                                 texture, sky, foliage, warm coat, light
                                 region, dark mass, red region, sky-over-
                                 ground layering, local contrast, water,
                                 neutral/gray)
  n2  norm ; r2 relu ; p2 maxpool 2x2      -> (16,8,8)
  three readout branches (spatial pyramid), all dense -> (10,), summed:
  d8  dense on p2 (16*8*8=1024)
  a4  avgpool 2x2 -> (16,4,4) ; d4 dense (256)
  a1  avgpool 8x8 -> (16,1,1) ; d1 dense (16)
  out add [d8, d4, d1]
"""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- stage 1
xs = np.arange(-2.0, 3.0)
Y, X = np.meshgrid(xs, xs, indexing="ij")   # Y grows downward
LUMA = np.array([0.299, 0.587, 0.114])


def znorm(w):
    """zero mean, unit L2 norm"""
    w = w - w.mean()
    return w / (np.linalg.norm(w) + 1e-12)


def bar(theta_deg, lam, sigma):
    """even-symmetric bar (line) filter; theta=0 -> vertical bar"""
    t = np.radians(theta_deg)
    u = X * np.cos(t) + Y * np.sin(t)
    return np.cos(2 * np.pi * u / lam) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))


def dog_blob(a):
    """double-opponent colour blob; a = per-channel centre weight vector"""
    gc = np.exp(-(X ** 2 + Y ** 2) / (2 * 0.8 ** 2))
    gs = np.exp(-(X ** 2 + Y ** 2) / (2 * 1.7 ** 2))
    k = gc.sum() / gs.sum()
    core = gc - k * gs
    a = np.asarray(a, dtype=float)
    a = a / np.linalg.norm(a)
    return np.stack([a[c] * core for c in range(3)])


W1 = np.zeros((16, 3, 5, 5))
# 0-3: luminance bars, four orientations
for i, th in enumerate([0, 45, 90, 135]):
    W1[i] = np.stack([LUMA[c] * znorm(bar(th, 4.0, 1.2)) for c in range(3)])
# 4: fine vertical step, bright-above positive (sky/ground, object/shadow)
p4 = znorm((-Y) * np.exp(-Y ** 2 / (2 * 0.9 ** 2)) * np.exp(-X ** 2 / (2 * 1.4 ** 2)))
W1[4] = np.stack([LUMA[c] * p4 for c in range(3)])
# 5: fine horizontal step, bright-right positive
p5 = znorm((X) * np.exp(-X ** 2 / (2 * 0.9 ** 2)) * np.exp(-Y ** 2 / (2 * 1.4 ** 2)))
W1[5] = np.stack([LUMA[c] * p5 for c in range(3)])
# 6-13: colour-opponent blobs
blobs = [
    (1, -1, 0),            # 6  red-vs-green
    (0.5, 0.5, -1),        # 7  yellow/sand-vs-blue
    (1, 1, 1),             # 8  white/bright blob
    (-1, -1, -1),          # 9  dark blob
    (-0.5, 1, -0.5),       # 10 green
    (-0.4, -0.4, 1),       # 11 blue
    (1, 0.3, -1),          # 12 warm (tan/brown/skin/fur)
    (1, -0.8, -0.8),       # 13 red
]
for i, a in enumerate(blobs):
    W1[6 + i] = znorm(dog_blob(a))
# 14: coarse horizontal bar (large horizontal structures: horizon, vehicle sides)
W1[14] = np.stack([LUMA[c] * znorm(bar(90, 8.0, 2.0)) for c in range(3)])
# 15: coarse bright-above-dark-below gradient (sky over ground)
p15 = znorm((-Y) * np.exp(-(X ** 2 + Y ** 2) / (2 * 2.0 ** 2)))
W1[15] = np.stack([LUMA[c] * p15 for c in range(3)])

# ---------------------------------------------------------------- stage 2
def PAT(kind):
    if kind == "C":
        p = np.zeros((3, 3)); p[1, 1] = 1.0
    elif kind == "U":
        p = np.ones((3, 3))
    elif kind == "RING":
        p = np.ones((3, 3)); p[1, 1] = 0.0
    elif kind == "G3":
        p = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 4.0
    else:
        raise ValueError(kind)
    return p


S2 = [
    # 0 wheel: dark disc centre, bright/textured ring, not dark around
    [(9, "C", 2.0), (8, "RING", 0.6), (0, "RING", 0.35), (1, "RING", 0.35),
     (2, "RING", 0.35), (3, "RING", 0.35), (9, "RING", -0.6)],
    # 1 bright round blob (headlight, white ball, bright small object)
    [(8, "C", 2.0), (0, "RING", 0.3), (2, "RING", 0.3), (9, "C", -0.6)],
    # 2 horizontal structure (vehicle sides, horizon, ground planes)
    [(2, "U", 1.0), (14, "U", 0.8), (4, "U", 0.5), (0, "C", -0.4)],
    # 3 vertical structure (posts, legs, trunks, tall objects)
    [(0, "U", 1.0), (5, "U", 0.6), (2, "C", -0.4)],
    # 4 fine structure / edge density (objects, fur, busy texture)
    [(i, "G3", 1.0) for i in range(6)],
    # 5 organic / diagonal / curved texture (animals, foliage)
    [(1, "U", 1.0), (3, "U", 1.0), (0, "U", 0.4), (2, "U", 0.4),
     (6, "U", 0.4), (7, "U", 0.4)],
    # 6 sky: blue + bright + bright-above
    [(11, "U", 1.2), (8, "U", 0.5), (15, "U", 0.6)],
    # 7 green foliage / grass
    [(10, "U", 1.2), (0, "U", 0.25), (1, "U", 0.25), (2, "U", 0.25), (3, "U", 0.25)],
    # 8 warm coat / bare ground (tan, brown, fur, skin)
    [(12, "U", 1.2), (7, "U", 0.4)],
    # 9 light region (white/gray/bright)
    [(8, "U", 1.2), (15, "U", 0.3)],
    # 10 dark mass (shadows, black fur, dark hulls)
    [(9, "U", 1.3)],
    # 11 red region
    [(13, "U", 1.2), (6, "U", 0.4)],
    # 12 sky-over-ground layering (large scale bright above dark)
    [(15, "U", 1.2), (4, "U", 0.5)],
    # 13 local contrast / object presence (any blob or strong edge)
    [(8, "C", 1.0), (9, "C", 1.0), (0, "C", 0.5), (2, "C", 0.5)],
    # 14 water: blue + horizontal ripples + glints
    [(11, "U", 0.8), (2, "U", 0.6), (8, "U", 0.5), (4, "U", 0.3)],
    # 15 neutral / achromatic region (man-made: gray, silver, white, black)
    [(8, "G3", 0.8), (9, "G3", 0.8), (10, "C", -0.5), (11, "C", -0.5),
     (12, "C", -0.5), (13, "C", -0.5), (6, "C", -0.4), (7, "C", -0.4)],
]

W2 = np.zeros((16, 16, 3, 3))
for o, terms in enumerate(S2):
    for ch, kind, w in terms:
        W2[o, ch] += w * PAT(kind)
    W2[o] = W2[o] / (np.linalg.norm(W2[o]) + 1e-12)

# ---------------------------------------------------------------- readout
# spatial profiles on the 8x8 grid (after p2)
yy = np.arange(8)[:, None] * np.ones((1, 8))
xx = np.arange(8)[None, :] * np.ones((8, 1))
CY = np.clip(yy - 3.5, -1.5, 1.5)      # rows 0-2 map to -1.5, rows 5-7 to +1.5
CX = np.clip(xx - 3.5, -1.5, 1.5)


def prof8(name):
    if name == "U":
        return np.ones((8, 8))
    if name == "TOP":
        return (yy <= 2).astype(float)
    if name == "MID":
        return ((yy >= 3) & (yy <= 5)).astype(float)
    if name == "BOT":
        return (yy >= 5).astype(float)
    if name == "CEN":
        return np.exp(-((yy - 3.5) ** 2 + (xx - 3.5) ** 2) / (2 * 2.0 ** 2))
    if name == "TOPCEN":
        return (yy <= 2).astype(float) * np.exp(-(CX ** 2) / (2 * 1.8 ** 2))
    if name == "BOTCEN":
        return (yy >= 5).astype(float) * np.exp(-(CX ** 2) / (2 * 2.0 ** 2))
    if name == "SIDE":
        return (np.abs(xx - 3.5) >= 2).astype(float)
    if name == "CENW":
        return np.exp(-((yy - 3.5) ** 2 + (xx - 3.5) ** 2) / (2 * 3.2 ** 2))
    raise ValueError(name)


COARSE = {"U": "CENW", "TOP": "TOP", "MID": "MID", "BOT": "BOT",
          "CEN": "CEN", "TOPCEN": "TOPCEN", "BOTCEN": "BOTCEN",
          "SIDE": "SIDE", "CENW": "CENW"}


def prof4(name):
    y = np.arange(4)[:, None] * np.ones((1, 4))
    x = np.arange(4)[None, :] * np.ones((4, 1))
    if name == "U":
        return np.ones((4, 4))
    if name == "TOP":
        return (y <= 1).astype(float)
    if name == "MID":
        return ((y >= 1) & (y <= 2)).astype(float)
    if name == "BOT":
        return (y >= 2).astype(float)
    if name == "CEN":
        return np.exp(-((y - 1.5) ** 2 + (x - 1.5) ** 2) / (2 * 1.2 ** 2))
    if name == "TOPCEN":
        return (y <= 1).astype(float) * np.exp(-((x - 1.5) ** 2) / (2 * 1.0 ** 2))
    if name == "BOTCEN":
        return (y >= 2).astype(float) * np.exp(-((x - 1.5) ** 2) / (2 * 1.0 ** 2))
    if name == "SIDE":
        return (np.abs(x - 1.5) >= 1).astype(float)
    if name == "CENW":
        return np.exp(-((y - 1.5) ** 2 + (x - 1.5) ** 2) / (2 * 2.0 ** 2))
    raise ValueError(name)


# per-class rules: (semantic channel, profile, weight) -- from general
# knowledge of what each class looks like and where in the frame it sits.
# "U" (uniform) profiles are forbidden here: they vanish under the zero-sum
# DC removal; global scene gist belongs to the GLOB table below.
RULES = {
    0: [(6, "TOP", 1.2), (9, "CEN", 0.9), (2, "MID", 0.8), (4, "CEN", 0.6),
        (15, "CEN", 0.5), (13, "CEN", 0.4), (1, "CEN", 0.3), (12, "TOP", 0.4),
        (10, "CEN", -0.3), (3, "SIDE", -0.3), (14, "BOT", -0.6),
        (0, "BOT", -0.4), (7, "CENW", -0.3), (8, "CENW", -0.2)],
    1: [(0, "BOT", 1.6), (0, "BOTCEN", 0.7), (2, "MID", 1.0), (15, "CEN", 0.7),
        (13, "CEN", 0.5), (11, "CEN", 0.4), (10, "CEN", 0.5), (4, "CEN", 0.4),
        (6, "TOP", 0.4), (12, "TOP", 0.3), (1, "CEN", 0.3), (9, "CEN", 0.3),
        (3, "SIDE", 0.3),
        (7, "CENW", -0.4), (8, "CENW", -0.3), (14, "BOT", -0.4),
        (5, "CENW", -0.3)],
    2: [(6, "TOP", 0.9), (7, "CENW", 0.6), (7, "SIDE", 0.5), (12, "TOP", 0.3),
        (5, "CEN", 0.6), (10, "CEN", 0.6), (13, "CEN", 0.4), (4, "CEN", 0.4),
        (8, "CEN", 0.5), (9, "CEN", 0.3), (11, "CEN", 0.3), (2, "CEN", 0.3),
        (3, "CEN", 0.3),
        (0, "BOT", -0.8), (14, "BOT", -0.4), (1, "BOT", -0.3),
        (15, "CENW", -0.2)],
    3: [(8, "CEN", 0.8), (4, "CEN", 0.7), (5, "CEN", 0.6), (5, "TOPCEN", 0.6),
        (10, "CEN", 0.5), (9, "CEN", 0.4), (15, "CEN", 0.4), (13, "CEN", 0.3),
        (11, "CEN", 0.2),
        (6, "TOP", -0.6), (12, "TOP", -0.4), (0, "BOT", -0.6),
        (14, "BOT", -0.3), (7, "SIDE", -0.4)],
    4: [(8, "CEN", 1.1), (8, "CENW", 0.4), (7, "SIDE", 0.9), (7, "TOP", 0.5),
        (4, "CEN", 0.5), (5, "CEN", 0.5), (3, "BOT", 0.4), (10, "CEN", 0.4),
        (10, "CENW", 0.3), (13, "CEN", 0.3), (6, "TOP", 0.3), (9, "CEN", 0.2),
        (0, "BOT", -0.6), (14, "BOT", -0.3), (2, "MID", -0.2),
        (15, "CENW", -0.3), (11, "CENW", -0.2)],
    5: [(8, "CEN", 0.9), (4, "CEN", 0.7), (9, "CEN", 0.6), (10, "CEN", 0.5),
        (5, "CEN", 0.6), (13, "CEN", 0.3), (15, "CEN", 0.3), (8, "CENW", 0.4),
        (7, "CENW", 0.3), (11, "CEN", 0.2), (2, "CEN", 0.2),
        (6, "TOP", -0.4), (12, "TOP", -0.3), (0, "BOT", -0.6),
        (14, "BOT", -0.3), (7, "SIDE", -0.3)],
    6: [(7, "SIDE", 1.3), (7, "TOP", 0.7), (14, "BOT", 0.8), (10, "CEN", 0.7),
        (4, "CEN", 0.4), (5, "CEN", 0.4), (13, "CEN", 0.3), (10, "CENW", 0.3),
        (6, "TOP", -0.5), (12, "TOP", -0.4), (0, "BOT", -0.3),
        (8, "CEN", -0.5), (15, "CENW", -0.3)],
    7: [(8, "CENW", 0.9), (8, "CEN", 0.5), (2, "MID", 0.7), (3, "BOT", 0.6),
        (4, "CEN", 0.5), (7, "BOT", 0.5), (6, "TOP", 0.5), (12, "TOP", 0.3),
        (9, "CENW", 0.4), (10, "CEN", 0.4), (5, "CEN", 0.4), (13, "CEN", 0.3),
        (0, "BOT", -0.5), (14, "BOT", -0.4), (11, "CENW", -0.2),
        (15, "CENW", -0.3)],
    8: [(14, "BOT", 1.7), (6, "TOP", 1.0), (2, "MID", 0.9), (12, "TOP", 0.7),
        (9, "MID", 0.7), (15, "MID", 0.5), (10, "MID", 0.5), (4, "MID", 0.4),
        (13, "MID", 0.3), (11, "MID", 0.3),
        (0, "BOTCEN", -0.4), (7, "SIDE", -0.4), (8, "CENW", -0.3),
        (3, "CENW", -0.2), (5, "CENW", -0.2)],
    9: [(0, "BOT", 1.5), (2, "CENW", 0.9), (15, "CENW", 0.7), (9, "CENW", 0.6),
        (11, "CEN", 0.5), (10, "CENW", 0.6), (4, "TOP", 0.6), (6, "TOP", 0.4),
        (12, "TOP", 0.3), (3, "SIDE", 0.4), (13, "CEN", 0.4), (1, "CEN", 0.3),
        (4, "CEN", 0.4),
        (7, "SIDE", -0.4), (8, "CENW", -0.3), (14, "BOT", -0.4),
        (5, "CENW", -0.2)],
}

# global (1x1) per-channel scene-gist weights: absolute cues that survive
# even when a feature covers the whole frame (sky, water, forest green).
GLOB = {
    0: {6: 1.6, 12: 0.6, 9: 0.6, 2: 0.4, 15: 0.4, 4: 0.2,
        14: -0.9, 7: -0.5, 8: -0.3, 10: -0.3},
    1: {0: 0.9, 2: 0.6, 15: 0.6, 4: 0.3, 13: 0.3, 11: 0.3, 10: 0.2,
        6: 0.3, 12: 0.2, 7: -0.4, 8: -0.3, 14: -0.3},
    2: {7: 0.7, 6: 0.7, 5: 0.3, 12: 0.3, 10: 0.2, 13: 0.2, 8: 0.2,
        0: -0.5, 15: -0.3, 14: -0.3, 2: -0.2},
    3: {4: 0.6, 8: 0.5, 5: 0.3, 10: 0.3, 13: 0.2, 15: 0.2, 11: 0.2,
        6: -0.6, 7: -0.4, 12: -0.5, 14: -0.3, 2: -0.3, 3: -0.2},
    4: {8: 0.9, 7: 1.0, 10: 0.4, 4: 0.3, 5: 0.3, 3: 0.2,
        6: -0.3, 15: -0.4, 9: -0.3, 2: -0.3, 14: -0.3, 0: -0.3},
    5: {4: 0.8, 8: 0.6, 9: 0.3, 10: 0.4, 5: 0.3, 13: 0.2, 7: 0.3,
        6: -0.3, 12: -0.3, 2: -0.2, 15: -0.2},
    6: {7: 1.8, 10: 0.5, 14: 0.4, 4: 0.3, 5: 0.3,
        8: -0.7, 6: -0.6, 12: -0.6, 15: -0.5, 9: -0.5, 2: -0.3, 3: -0.2,
        11: -0.2},
    7: {8: 1.1, 7: 0.5, 2: 0.5, 12: 0.4, 3: 0.3, 6: 0.3, 9: 0.3, 4: 0.2,
        5: 0.2, 0: -0.4, 15: -0.3, 14: -0.3, 11: -0.2},
    8: {14: 1.6, 12: 0.9, 6: 0.8, 2: 0.5, 9: 0.4, 15: 0.4, 10: 0.2, 4: 0.2,
        7: -0.5, 8: -0.5, 5: -0.2, 0: -0.2, 11: -0.2},
    9: {0: 0.8, 2: 0.6, 15: 0.6, 9: 0.4, 10: 0.4, 11: 0.4, 4: 0.3, 6: 0.3,
        12: 0.2, 13: 0.2, 7: -0.3, 8: -0.3, 14: -0.3, 5: -0.2},
}

W8 = np.zeros((10, 16, 8, 8))
for c, rules in RULES.items():
    for s, pn, w in rules:
        W8[c, s] += w * prof8(pn)

# DC removal, folded into the dense weights: each per-(class, channel)
# spatial profile is made zero-sum over positions, so the readouts respond
# only to where a feature stands out relative to the rest of the same image.
W8 -= W8.mean(axis=(2, 3), keepdims=True)

# balance the classes: give every class the same total absolute template
# mass, so no class wins merely by having larger weights on generic
# structure; only the *pattern* discriminates.
mass = np.abs(W8).sum(axis=(1, 2, 3))
target = float(np.exp(np.mean(np.log(mass))))
scale = target / mass
W8 *= scale[:, None, None, None]

# coarse 4x4 branch: same rules, coarse profiles, half strength
W4 = np.zeros((10, 16, 4, 4))
for c, rules in RULES.items():
    for s, pn, w in rules:
        W4[c, s] += 0.5 * scale[c] * w * prof4(COARSE[pn])
W4 -= W4.mean(axis=(2, 3), keepdims=True)

W1d = np.zeros((10, 16))
for c, gw in GLOB.items():
    for s, w in gw.items():
        W1d[c, s] = w
gmass = np.abs(W1d).sum(axis=1)
W1d *= (3.0 / gmass)[:, None]

# d1 sees absolute global means; after norm+relu+maxpool+avgpool a typical
# channel mean is ~ACT1 on any structured image, so centre each class there.
ACT1 = 0.8
b1 = -(W1d.sum(axis=1) * ACT1)

weights = {
    "c1.W": W1.astype(np.float32), "c1.b": np.zeros(16, np.float32),
    "c2.W": W2.astype(np.float32), "c2.b": np.zeros(16, np.float32),
    "d8.W": W8.reshape(10, -1).astype(np.float32),
    "d8.b": np.zeros(10, np.float32),
    "d4.W": W4.reshape(10, -1).astype(np.float32),
    "d4.b": np.zeros(10, np.float32),
    "d1.W": W1d.astype(np.float32), "d1.b": b1.astype(np.float32),
}
np.savez(os.path.join(OUT, "weights.npz"), **weights)

# ---------------------------------------------------------------- layers
net = {"layers": [
    {"name": "c1", "op": "conv", "inputs": ["image"], "out": 16,
     "kernel": [5, 5], "stride": 1, "padding": 2},
    {"name": "n1", "op": "norm", "inputs": ["c1"]},
    {"name": "r1", "op": "relu", "inputs": ["n1"]},
    {"name": "p1", "op": "maxpool", "inputs": ["r1"], "size": [2, 2],
     "stride": [2, 2]},
    {"name": "c2", "op": "conv", "inputs": ["p1"], "out": 16,
     "kernel": [3, 3], "stride": 1, "padding": 1},
    {"name": "n2", "op": "norm", "inputs": ["c2"]},
    {"name": "r2", "op": "relu", "inputs": ["n2"]},
    {"name": "p2", "op": "maxpool", "inputs": ["r2"], "size": [2, 2],
     "stride": [2, 2]},
    {"name": "a4", "op": "avgpool", "inputs": ["p2"], "size": [2, 2],
     "stride": [2, 2]},
    {"name": "a1", "op": "avgpool", "inputs": ["p2"], "size": [8, 8],
     "stride": [8, 8]},
    {"name": "d8", "op": "dense", "inputs": ["p2"], "out": 10},
    {"name": "d4", "op": "dense", "inputs": ["a4"], "out": 10},
    {"name": "d1", "op": "dense", "inputs": ["a1"], "out": 10},
    {"name": "out", "op": "add", "inputs": ["d8", "d4", "d1"]},
]}
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump(net, f, indent=1)

print("wrote", os.path.join(OUT, "network.json"), "and weights.npz;",
      "total weights:", sum(v.size for v in weights.values()))
