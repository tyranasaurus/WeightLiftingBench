"""Writes network.json and weights.npz for a hand-designed 32x32 photo classifier.
Every number below is typed from general knowledge of the classes (sky is blue and smooth, grass is green,
roads are grey, fur is brown/tan, ...). No data is read and nothing is fitted."""
import json, sys, os
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
# class order: airplane automobile bird cat deer dog frog horse ship truck

# ---- colour branch, layer A: soft per-pixel indicators  sat(x) = relu(x) - relu(x - 1) --------------------
t = 1.0 / 3.0
IND = [  # name, (wR, wG, wB), constant, ramp width
    ("blue1",   (-1, 0, 1), -0.04, 0.10),   # bluer than red
    ("blue2",   (0, -1, 1),  0.03, 0.08),   # at least about as blue as green
    ("bright",  (t, t, t),  -0.35, 0.20),
    ("vbright", (t, t, t),  -0.68, 0.15),
    ("dark",    (-t, -t, -t), 0.26, 0.12),
    ("green1",  (0, 1, -1), -0.04, 0.10),   # greener than blue
    ("green2",  (-1, 1, 0),  0.02, 0.08),   # at least about as green as red (includes yellow-green grass)
    ("warm1",   (1, 0, -1), -0.08, 0.12),   # R > G > B: earth, fur, wood
    ("warm2",   (1, -1, 0), -0.01, 0.06),
    ("red1",    (1, -1, 0), -0.15, 0.15),   # saturated red paint
    ("red2",    (1, 0, -1), -0.15, 0.15),
]
ABS = [(1, -1, 0), (-1, 1, 0), (0, 1, -1), (0, -1, 1), (-1, 0, 1), (1, 0, -1)]  # their sum = 2 * (max - min)
nA = 2 * len(IND) + len(ABS)
caW, cab = np.zeros((nA, 3, 1, 1)), np.zeros(nA)
pos = {}
for j, (name, w, c, width) in enumerate(IND):
    pos[name] = j
    for u in (0, 1):
        caW[2 * j + u, :, 0, 0] = np.array(w, float) / width
        cab[2 * j + u] = c / width - u
for j, w in enumerate(ABS):
    caW[2 * len(IND) + j, :, 0, 0] = w

# ---- colour branch, layer B: materials = soft ANDs of indicators ------------------------------------------
MATS = ["sky", "dblue", "green", "brown", "red", "white", "gray", "dark"]
cbW, cbb = np.zeros((len(MATS), nA, 1, 1)), np.zeros(len(MATS))
def put(m, name, sign):
    j = pos[name]
    cbW[MATS.index(m), 2 * j, 0, 0] += sign
    cbW[MATS.index(m), 2 * j + 1, 0, 0] -= sign
def lowsat(m):  # adds (1 - (max-min)/0.12) to the unit
    cbW[MATS.index(m), 2 * len(IND):, 0, 0] -= 1.0 / 0.24
for n in ("blue1", "blue2", "bright"): put("sky", n, 1)
cbb[MATS.index("sky")] = -2                                   # blue and bright
put("dblue", "blue1", 1); put("dblue", "blue2", 1); put("dblue", "bright", -1)
cbb[MATS.index("dblue")] = -1                                 # blue and not bright (water, dusk sky)
put("green", "green1", 1); put("green", "green2", 1); cbb[MATS.index("green")] = -1
put("brown", "warm1", 1); put("brown", "warm2", 1); put("brown", "red1", -1); cbb[MATS.index("brown")] = -1
put("red", "red1", 1); put("red", "red2", 1); cbb[MATS.index("red")] = -1
put("white", "vbright", 1); lowsat("white")                   # very bright and unsaturated
put("gray", "vbright", -1); put("gray", "dark", -1); lowsat("gray"); cbb[MATS.index("gray")] = 1
put("dark", "dark", 1)

# ---- colour readout: rule table typed by hand: evidence for each class when a material fills a band --------
#            plane  auto  bird   cat  deer   dog  frog horse  ship truck
TABLE = {
 "sky":   {"top": [ 2.2, -0.3,  0.8, -1.5, -0.5, -1.2, -2.0,  0.3,  1.5,  0.5],
           "mid": [ 2.5, -0.6,  0.9, -1.0, -1.0, -1.0, -1.5, -0.8,  1.2, -0.3],
           "bot": [ 2.0, -1.0,  0.6, -1.0, -1.2, -1.0, -1.5, -1.5,  2.2, -1.2],
           "ctr": [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]},
 "dblue": {"top": [ 0.6, -0.4,  0.0, -0.5, -0.5, -0.5, -0.6, -0.5,  1.0, -0.4],
           "mid": [ 0.6, -0.2,  0.2, -0.5, -0.6, -0.5, -0.6, -0.6,  1.6, -0.4],
           "bot": [ 0.3, -0.3,  0.3, -0.6, -0.8, -0.6, -0.8, -0.9,  3.0, -0.6],
           "ctr": [ 0.0,  0.4,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.3]},
 "green": {"top": [-1.5,  0.3,  1.0, -0.5,  1.5,  0.2,  0.8,  1.0, -1.2,  0.0],
           "mid": [-1.5, -0.6,  1.0, -0.5,  1.5,  0.2,  1.5,  0.6, -1.5, -0.9],
           "bot": [-0.4, -0.3,  0.7, -0.5,  2.0,  0.8,  1.2,  2.0, -2.0, -0.7],
           "ctr": [ 0.0,  0.0,  0.0,  0.0, -0.3, -0.2,  0.8, -0.5,  0.0,  0.0]},
 "brown": {"top": [-1.8, -0.2,  0.5,  1.0,  0.8,  0.9,  1.2,  0.5, -1.8, -0.2],
           "mid": [-1.8,  0.0,  0.5,  1.1,  0.9,  1.0,  1.2,  0.9, -1.8,  0.0],
           "bot": [-1.5,  0.0,  0.4,  1.0,  0.8,  0.9,  1.2,  0.9, -1.8,  0.0],
           "ctr": [ 0.0,  0.0,  0.0,  0.2,  0.5,  0.3,  0.0,  0.6,  0.0,  0.0]},
 "red":   {"top": [-0.5,  2.2, -0.6, -0.2, -1.0, -0.4, -0.8, -0.6,  0.2,  2.2],
           "mid": [-0.5,  2.2, -0.6, -0.2, -1.0, -0.4, -0.8, -0.6,  0.2,  2.2],
           "bot": [-0.5,  2.2, -0.6, -0.2, -1.0, -0.4, -0.8, -0.6,  0.2,  2.2],
           "ctr": [ 0.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.3]},
 "white": {"top": [ 1.0,  0.2,  0.2,  0.0, -0.8,  0.0, -1.0,  0.2,  0.5,  1.0],
           "mid": [ 0.6,  0.3,  0.0,  0.4, -0.8,  0.5, -0.8, -0.2,  0.2,  0.8],
           "bot": [ 0.4,  0.5,  0.0,  0.4, -0.7,  0.4, -0.5, -0.6, -0.8,  0.3],
           "ctr": [ 0.2,  0.2,  0.0,  0.2,  0.0,  0.4,  0.0,  0.0,  0.2,  0.3]},
 "gray":  {"top": [ 1.0,  0.3,  0.3,  0.0, -0.8, -0.3, -0.8, -0.4,  0.8,  0.4],
           "mid": [ 0.8,  0.6,  0.0,  0.4, -0.8, -0.2, -0.6, -0.6,  0.6,  0.6],
           "bot": [ 0.7,  1.5, -0.3,  0.3, -1.0,  0.0, -0.6, -0.5,  0.5,  1.5],
           "ctr": [ 0.3,  0.3,  0.0,  0.2,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]},
 "dark":  {"top": [-1.0,  0.2,  0.0,  0.6,  0.4,  0.4,  0.5,  0.2, -0.6,  0.0],
           "mid": [-0.8,  0.3,  0.0,  0.5,  0.3,  0.4,  0.5,  0.5, -0.4,  0.2],
           "bot": [-0.8,  1.0,  0.0,  0.4,  0.2,  0.3,  0.5,  0.2, -0.3,  0.8],
           "ctr": [ 0.0,  0.0,  0.0,  0.2,  0.0,  0.2,  0.0,  0.5,  0.0,  0.0]},
}
GAIN_COLOUR = 2.0
dcW = np.zeros((10, len(MATS), 4, 4))
for mi, m in enumerate(MATS):
    for r in range(4):
        band = "top" if r == 0 else ("bot" if r == 3 else "mid")
        for c in range(4):
            dcW[:, mi, r, c] += np.array(TABLE[m][band]) / 16.0
            if r in (1, 2) and c in (1, 2):
                dcW[:, mi, r, c] += np.array(TABLE[m]["ctr"]) / 4.0
dcW *= GAIN_COLOUR
dcW -= dcW.mean(axis=0, keepdims=True)          # no feature favours all classes at once
dcW = dcW.reshape(10, -1)

# ---- edge branch: Sobel derivatives of brightness, both signs, pooled on a 3 x 3 grid ----------------------
GAIN_EDGE = 12.0
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8.0
ceW = np.zeros((4, 3, 3, 3))
for i in range(3):
    ceW[0, i], ceW[1, i], ceW[2, i], ceW[3, i] = sx / 3, -sx / 3, sx.T / 3, -sx.T / 3
ceW *= GAIN_EDGE
#                  plane  auto  bird   cat  deer   dog  frog horse  ship truck
BORDER_TEXTURE = [-1.5,  0.5, -0.8,  0.5,  0.0,  0.4,  0.6,  0.0, -0.6,  0.9]   # busy surround vs smooth sky / blur
TOP_TEXTURE    = [-0.4,  0.2, -0.2,  0.3,  0.0,  0.3,  0.3, -0.2, -0.5,  0.4]
HORIZONTALNESS = [ 0.6,  0.7, -0.3, -0.3, -0.4, -0.3,  0.0, -0.5,  1.0,  0.0]   # horizontal minus vertical edges, middle band
deW = np.zeros((10, 4, 3, 3))
for r in range(3):
    for c in range(3):
        if not (r == 1 and c == 1):
            deW[:, :, r, c] += np.array(BORDER_TEXTURE)[:, None] / 8.0
        if r == 0:
            deW[:, :, r, c] += np.array(TOP_TEXTURE)[:, None] / 3.0
        if r == 1:
            deW[:, 2:, r, c] += np.array(HORIZONTALNESS)[:, None] / 3.0   # channels 2,3: horizontal edges (d/dy)
            deW[:, :2, r, c] -= np.array(HORIZONTALNESS)[:, None] / 3.0   # channels 0,1: vertical edges (d/dx)
deW -= deW.mean(axis=0, keepdims=True)
deW = deW.reshape(10, -1)

net = {"layers": [
    {"name": "ca", "op": "conv", "inputs": ["image"], "out": nA, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "ra", "op": "relu", "inputs": ["ca"]},
    {"name": "cb", "op": "conv", "inputs": ["ra"], "out": len(MATS), "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "rb", "op": "relu", "inputs": ["cb"]},
    {"name": "pc", "op": "avgpool", "inputs": ["rb"], "size": [8, 8], "stride": [8, 8]},
    {"name": "dc", "op": "dense", "inputs": ["pc"], "out": 10},
    {"name": "ce", "op": "conv", "inputs": ["image"], "out": 4, "kernel": [3, 3], "stride": 1, "padding": 0},
    {"name": "re", "op": "relu", "inputs": ["ce"]},
    {"name": "pe", "op": "avgpool", "inputs": ["re"], "size": [10, 10], "stride": [10, 10]},
    {"name": "de", "op": "dense", "inputs": ["pe"], "out": 10},
    {"name": "scores", "op": "add", "inputs": ["dc", "de"]},
]}
with open(os.path.join(out_dir, "network.json"), "w") as f:
    json.dump(net, f, indent=1)
f32 = lambda a: np.asarray(a, np.float32)
np.savez(os.path.join(out_dir, "weights.npz"), **{
    "ca.W": f32(caW), "ca.b": f32(cab), "cb.W": f32(cbW), "cb.b": f32(cbb),
    "dc.W": f32(dcW), "dc.b": f32(np.zeros(10)), "ce.W": f32(ceW), "ce.b": f32(np.zeros(4)),
    "de.W": f32(deW), "de.b": f32(np.zeros(10))})
