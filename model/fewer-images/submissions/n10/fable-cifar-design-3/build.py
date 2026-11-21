"""Hand-designed network for 10-class 32x32 photo classification. Every number is typed by hand from general
knowledge of colours / scenes of the classes; no data is read and nothing is fitted."""
import sys, json, os
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
R, G, B = 0, 1, 2

# ---- layer A: 3x3 conv (no padding) + relu: colour-opponent parts, brightness parts, Sobel edges of luminance ----
NA = 17
AW = np.zeros((NA, 3, 3, 3), np.float32); Ab = np.zeros(NA, np.float32)
def centre(u, r=0.0, g=0.0, b=0.0, bias=0.0):
    AW[u, R, 1, 1], AW[u, G, 1, 1], AW[u, B, 1, 1], Ab[u] = r, g, b, bias
centre(0, r=-1, b=1)                 # d1 = relu(B-R)
centre(1, r=1, b=-1)                 # d2 = relu(R-B)
centre(2, r=-1, g=1)                 # d3 = relu(G-R)
centre(3, r=1, g=-1)                 # d4 = relu(R-G)
centre(4, g=1, b=-1)                 # d5 = relu(G-B)
centre(5, g=-1, b=1)                 # d6 = relu(B-G)
centre(6, 1/3, 1/3, 1/3)             # V  = brightness
centre(7, -1/3, -1/3, -1/3, 0.75)    # relu(0.75 - V)
centre(8, 1/3, 1/3, 1/3, -0.75)      # relu(V - 0.75)
centre(9, -1/3, -1/3, -1/3, 0.30)    # relu(0.30 - V)
centre(10, r=-1, g=2, b=-1, bias=-0.10)   # m_b = relu((B-R) - 2(B-G) - 0.10)
centre(11, r=2, g=-1, b=-1, bias=-0.06)   # m_g = relu((G-B) - 2(G-R) - 0.06)
centre(12, r=-3, g=4, b=-1)               # m_w = relu((R-B) - 4(R-G))
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0   # Sobel, luminance derivative per pixel
for ch in range(3):
    AW[13, ch] = sx / 3; AW[14, ch] = -sx / 3          # +-gx  (vertical edges)
    AW[15, ch] = sx.T / 3; AW[16, ch] = -sx.T / 3      # +-gy  (horizontal edges)

# ---- layer B: 1x1 conv + relu: soft colour names and edge classes; clip01(z) = relu(z) - relu(z-1) ----
NB = 17
BW = np.zeros((NB, NA, 1, 1), np.float32); Bb = np.zeros(NB, np.float32)
def unit(u, terms, bias):
    for k, v in terms.items(): BW[u, k, 0, 0] = v
    Bb[u] = bias
def pair(u, terms, gain, thr):      # clip01(gain * (x - thr)) as two relus
    t = {k: gain * v for k, v in terms.items()}
    unit(u, t, -gain * thr); unit(u + 1, t, -gain * thr - 1.0)
pair(0, {0: 1, 1: -1, 10: -1}, 10, 0.04)          # blue : min(B-R, 2(B-G)+0.10)   sky, water
pair(2, {4: 1, 5: -1, 11: -1}, 10, 0.04)          # green: min(G-B, 2(G-R)+0.06)   vegetation
pair(4, {1: 1, 0: -1, 12: -1}, 8, 0.06)           # warm : min(R-B, 4(R-G))        brown, tan, orange, fur, soil
pair(6, {3: 1, 4: -2, 5: -2}, 6, 0.15)            # red  : (R-G) - 2|G-B|          paint
allc = {k: -5.0 for k in range(6)}                # -10 * chroma (chroma = sum of d / 2)
unit(8, {**allc, 7: -4.0}, 1.0)                   # white: neutral and bright
unit(9, {**allc, 8: -4.0, 9: -4.0}, 1.0)          # grey : neutral, mid brightness
unit(10, {6: -6.0}, 2.0); unit(11, {6: -6.0}, 1.0)  # dark : clip01(2 - 6V)
pair(12, {15: 1, 16: 1}, 12, 0.04)                # eh: strong horizontal edge
pair(14, {13: 1, 14: 1}, 12, 0.04)                # ev: strong vertical edge
unit(16, {13: -12, 14: -12, 15: -12, 16: -12}, 1.0)  # flat: no edges

# ---- layer C: 1x1 linear: the 10 maps ----
MAPS = ["blue", "green", "warm", "red", "white", "grey", "dark", "eh", "ev", "flat"]
NM = len(MAPS)
CW = np.zeros((NM, NB, 1, 1), np.float32); Cb = np.zeros(NM, np.float32)
for m, (a, b) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7), (8, None), (9, None), (10, 11), (12, 13), (14, 15), (16, None)]):
    CW[m, a, 0, 0] = 1.0
    if b is not None: CW[m, b, 0, 0] = -1.0

# ---- region layer: 6x6 grid of cell means -> region means (top, bottom, centre, border) + conjunctions ----
REG = ["T", "Bt", "Ctr", "Brd"]
masks = {r: np.zeros((6, 6), np.float32) for r in REG}
masks["T"][0:2, :] = 1; masks["Bt"][4:6, :] = 1; masks["Ctr"][1:5, 1:5] = 1; masks["Brd"][:] = 1; masks["Brd"][1:5, 1:5] = 0
for r in REG: masks[r] /= masks[r].sum()
NF = NM * 4
CONJ = ["sky_all", "grass_sky", "green_all", "white_all", "object_on_smooth"]
NR = NF + len(CONJ)
RW = np.zeros((NR, NM, 6, 6), np.float32); Rb = np.zeros(NR, np.float32)
def fi(m, r): return MAPS.index(m) * 4 + REG.index(r)
for m in MAPS:
    for r in REG: RW[fi(m, r), MAPS.index(m)] = masks[r]
def conj(name, plus, minus=(), bias=-1.0):
    u = NF + CONJ.index(name)
    for m, r in plus: RW[u, MAPS.index(m)] += masks[r]
    for m, r in minus: RW[u, MAPS.index(m)] -= masks[r]
    Rb[u] = bias
conj("sky_all", [("blue", "T"), ("blue", "Bt")])
conj("grass_sky", [("green", "Bt"), ("blue", "T"), ("white", "T")])
conj("green_all", [("green", "T"), ("green", "Bt")])
conj("white_all", [("white", "T"), ("white", "Bt")])
conj("object_on_smooth", [("flat", "Brd")], [("flat", "Ctr")], bias=-0.15)

# ---- output: hand-typed evidence table. Columns: plane, auto, bird, cat, deer, dog, frog, horse, ship, truck ----
E = {
 ("blue", "T"):   [ 2.0,  0.3,  0.6, -1.0, -0.6, -0.8, -1.2,  0.3,  1.8,  0.5],
 ("blue", "Bt"):  [ 1.5, -0.5,  0.5, -0.8, -0.8, -0.8, -1.0, -0.8,  2.5, -0.6],
 ("blue", "Ctr"): [ 0.5,  0.3,  0.0, -0.5, -0.8, -0.6, -0.6, -0.6,  0.6,  0.3],
 ("blue", "Brd"): [ 1.0, -0.2,  0.5, -0.5, -0.5, -0.5, -0.8, -0.2,  1.0,  0.0],
 ("green", "T"):  [-1.0, -0.3,  0.8, -0.5,  1.5,  0.0,  0.8,  0.8, -1.2, -0.3],
 ("green", "Bt"): [-0.6, -0.6,  0.6, -0.3,  1.8,  0.5,  0.8,  1.8, -1.2, -0.5],
 ("green", "Ctr"):[-0.8, -0.3,  0.5, -0.3,  0.8, -0.2,  1.5,  0.2, -1.0, -0.5],
 ("green", "Brd"):[-0.5, -0.3,  0.6, -0.2,  1.2,  0.2,  0.5,  1.0, -1.0, -0.3],
 ("warm", "T"):   [-1.0, -0.5,  0.3,  0.8,  0.5,  0.8,  1.0,  0.3, -1.0, -0.5],
 ("warm", "Bt"):  [-0.8, -0.5,  0.3,  0.8,  0.6,  0.8,  1.0,  0.5, -1.0, -0.3],
 ("warm", "Ctr"): [-1.0, -0.5,  0.3,  0.6,  0.8,  0.9,  0.8,  1.0, -0.8, -0.5],
 ("warm", "Brd"): [-0.6, -0.4,  0.2,  0.6,  0.3,  0.6,  0.8,  0.2, -0.8, -0.3],
 ("red", "Ctr"):  [-0.3,  2.0,  0.0,  0.0, -0.5,  0.0,  0.2,  0.0,  0.0,  2.0],
 ("red", "Brd"):  [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.3,  0.0,  0.0,  0.0],
 ("white", "T"):  [ 1.2,  0.3,  0.3,  0.0, -0.6,  0.0, -0.2,  0.2,  0.5,  0.8],
 ("white", "Bt"): [ 0.8,  0.0,  0.2,  0.2, -0.6,  0.2,  0.2, -0.4, -0.3,  0.2],
 ("white", "Ctr"):[ 0.2,  0.3, -0.2,  0.2, -0.6,  0.5, -0.4, -0.2,  0.6,  0.6],
 ("white", "Brd"):[ 0.8,  0.0,  0.3,  0.0, -0.5,  0.0,  0.2, -0.2,  0.2,  0.4],
 ("grey", "T"):   [ 0.6,  0.4,  0.2,  0.2, -0.5,  0.0, -0.4, -0.2,  0.5,  0.4],
 ("grey", "Bt"):  [ 0.3,  1.5,  0.0,  0.2, -0.6,  0.1, -0.3, -0.2,  0.4,  1.3],
 ("grey", "Ctr"): [ 0.5,  0.6,  0.0,  0.5, -0.5,  0.0, -0.3, -0.4,  0.3,  0.5],
 ("grey", "Brd"): [ 0.4,  0.5,  0.1,  0.2, -0.4,  0.0, -0.2, -0.2,  0.3,  0.4],
 ("dark", "T"):   [-0.6,  0.2,  0.0,  0.4,  0.5,  0.3,  0.3,  0.0, -0.6, -0.3],
 ("dark", "Bt"):  [-0.5,  0.8,  0.0,  0.2,  0.2,  0.1,  0.2,  0.0,  0.2,  0.5],
 ("dark", "Ctr"): [ 0.0,  0.3,  0.2,  0.3,  0.0,  0.1,  0.0,  0.5,  0.0,  0.2],
 ("dark", "Brd"): [-0.5,  0.2,  0.0,  0.4,  0.4,  0.3,  0.3,  0.0, -0.3, -0.2],
 ("eh", "Ctr"):   [ 0.5,  1.2, -0.5, -0.8, -0.6, -0.8, -0.6, -0.3,  1.0,  1.2],
 ("eh", "Brd"):   [-0.5,  0.5, -0.3,  0.0,  0.0,  0.0,  0.0,  0.0,  0.3,  0.5],
 ("ev", "Ctr"):   [-0.8,  0.3, -0.2,  0.3,  0.2,  0.2, -0.3,  0.6, -0.3,  1.2],
 ("ev", "Brd"):   [-0.8,  0.2, -0.2,  0.3,  0.3,  0.2,  0.0,  0.3, -0.5,  0.6],
 ("flat", "Brd"): [ 1.5, -0.3,  0.8, -0.2, -0.6, -0.2,  0.3, -0.5,  0.6, -0.6],
 ("flat", "Ctr"): [ 0.3, -0.3,  0.3,  0.0,  0.0,  0.0,  0.2, -0.3,  0.0, -0.6],
}
EC = {
 "sky_all":          [ 1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.3,  0.0],
 "grass_sky":        [ 0.3,  0.0,  0.0,  0.0,  0.6,  0.0,  0.0,  1.5,  0.0,  0.2],
 "green_all":        [ 0.0,  0.0,  0.5,  0.0,  0.8,  0.0,  0.6, -0.3,  0.0,  0.0],
 "white_all":        [ 0.5,  0.4,  0.3,  0.2,  0.0,  0.2,  0.5,  0.0,  0.3,  0.5],
 "object_on_smooth": [ 1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.3,  0.0,  0.3,  0.0],
}
OW = np.zeros((10, NR), np.float32); Ob = np.zeros(10, np.float32)
for (m, r), row in E.items(): OW[:, fi(m, r)] = row
for name, row in EC.items(): OW[:, NF + CONJ.index(name)] = row

net = {"layers": [
    {"name": "a", "op": "conv", "inputs": ["image"], "out": NA, "kernel": [3, 3], "stride": 1, "padding": 0},
    {"name": "a_r", "op": "relu", "inputs": ["a"]},
    {"name": "b", "op": "conv", "inputs": ["a_r"], "out": NB, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "b_r", "op": "relu", "inputs": ["b"]},
    {"name": "c", "op": "conv", "inputs": ["b_r"], "out": NM, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "cells", "op": "avgpool", "inputs": ["c"], "size": [5, 5], "stride": [5, 5]},
    {"name": "regions", "op": "dense", "inputs": ["cells"], "out": NR},
    {"name": "regions_r", "op": "relu", "inputs": ["regions"]},
    {"name": "scores", "op": "dense", "inputs": ["regions_r"], "out": 10},
]}
with open(os.path.join(out_dir, "network.json"), "w") as f: json.dump(net, f, indent=1)
np.savez(os.path.join(out_dir, "weights.npz"), **{
    "a.W": AW, "a.b": Ab, "b.W": BW, "b.b": Bb, "c.W": CW, "c.b": Cb,
    "regions.W": RW.reshape(NR, -1), "regions.b": Rb, "scores.W": OW, "scores.b": Ob})
