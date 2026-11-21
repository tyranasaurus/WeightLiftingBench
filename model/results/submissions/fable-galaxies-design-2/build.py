"""Hand-built galaxy classifier. Every number comes from the formulas and hand-chosen constants below; no data is read.
Usage: python3 build.py OUT_DIR   (writes OUT_DIR/network.json and OUT_DIR/weights.npz)"""
import json, sys, os
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
layers, W = [], {}
R = 64                      # working resolution after 2x2 averaging
CEN = (R - 1) / 2.0         # image centre in working pixels


def add_layer(name, op, inputs, **kw):
    layers.append(dict(name=name, op=op, inputs=inputs, **kw)); return name


def conv(name, inp, w, b, stride=1, padding=0):
    w = np.asarray(w, np.float32); W[name + ".W"] = w; W[name + ".b"] = np.asarray(b, np.float32)
    return add_layer(name, "conv", [inp], out=int(w.shape[0]), kernel=[int(w.shape[2]), int(w.shape[3])], stride=stride, padding=padding)


def dense(name, inp, w, b):
    w = np.asarray(w, np.float32); W[name + ".W"] = w; W[name + ".b"] = np.asarray(b, np.float32)
    return add_layer(name, "dense", [inp], out=int(w.shape[0]))


# ---- 1. half resolution, smoothed luminance and smoothed (red - blue) -------------------------------------------
add_layer("p2", "avgpool", ["image"], size=[2, 2], stride=[2, 2])
b1 = np.array([1, 4, 6, 4, 1], np.float64) / 16.0
B5 = np.outer(b1, b1)                                   # binomial blur
w = np.zeros((2, 3, 5, 5))
w[0, :] = B5 / 3.0                                      # luminance
w[1, 0] = B5; w[1, 2] = -B5                             # red minus blue
conv("sm", "p2", w, [0, 0], padding=2)

# ---- 2. sky level of this image = darkest 16x16 block mean of the luminance -------------------------------------
conv("nl", "p2", np.full((1, 3, 1, 1), -1 / 3.0), [0])
add_layer("nb", "avgpool", ["nl"], size=[16, 16], stride=[8, 8])       # (1,7,7) of minus block means
add_layer("mb", "maxpool", ["nb"], size=[7, 7], stride=[7, 7])         # (1,1,1) = -sky
T_ABS = 0.04                                            # isophote: this far above the sky (hand-chosen, several times the smoothed noise)
w = np.zeros((2 * R * R, 1)); w[:R * R, 0] = 1.0
b = np.zeros(2 * R * R); b[:R * R] = -T_ABS
dense("bd", "mb", w, b)
add_layer("bm", "reshape", ["bd"], shape=[2, R, R])
add_layer("ex", "add", ["sm", "bm"])                    # [luminance - sky - T, red - blue]

# ---- 3. footprint mask and the connected piece of it that contains the image centre ----------------------------
KM = 50.0                                               # mask goes 0 -> 1 over 0.02 of brightness
conv("mk1", "ex", [[[[KM]], [[0]]], [[[KM]], [[0]]]], [0, -1])
add_layer("mk1r", "relu", ["mk1"])
conv("mm", "mk1r", [[[[1.0]], [[-1.0]]]], [-1.0])       # mask - 1
yy, xx = np.mgrid[0:R, 0:R]
dx, dy = xx - CEN, yy - CEN
rr = np.hypot(dx, dy)
seed = (rr <= 3.2).astype(np.float64)                   # small disc at the image centre
dense("sd", "mb", np.zeros((R * R, 1)), seed.ravel())
add_layer("sdr", "reshape", ["sd"], shape=[1, R, R])
add_layer("s0a", "add", ["mm", "sdr"])
cur = add_layer("s0", "relu", ["s0a"])
for t in range(16):                                     # geodesic dilation: grow the seed inside the mask, 2 px per step
    conv(f"gp{t}", cur, [[[[1.0]]]], [0], padding=2)
    add_layer(f"gx{t}", "maxpool", [f"gp{t}"], size=[5, 5], stride=[1, 1])
    add_layer(f"ga{t}", "add", [f"gx{t}", "mm"])
    cur = add_layer(f"gs{t}", "relu", [f"ga{t}"])

# ---- 4. gate brightness and colour by the central piece: channels [G, S, D+, D-] -------------------------------
w = np.zeros((4, 2, 1, 1)); w[0, 0] = 1; w[2, 1] = 1; w[3, 1] = -1
conv("e4", "ex", w, [0, 0, 0, 0])
conv("s4", cur, np.ones((4, 1, 1, 1)), [-1, 0, -1, -1])
add_layer("g4a", "add", ["e4", "s4"])
add_layer("g4", "relu", ["g4a"])

# ---- 5. moments: second moments along 16 directions (mask and flux weighted), first moments along 16 directions ---
NT = 16
w = np.zeros((5 * NT, 4, R, R))
for k in range(NT):
    th = np.pi * k / NT
    u2 = (dx * np.cos(th) + dy * np.sin(th)) ** 2
    w[k, 1] = u2; w[NT + k, 1] = -u2                    # mask S
    w[2 * NT + k, 0] = u2; w[3 * NT + k, 0] = -u2       # flux G
    ph = 2 * np.pi * k / NT
    w[4 * NT + k, 1] = dx * np.cos(ph) + dy * np.sin(ph)
conv("mo", "g4", w, np.zeros(5 * NT))
add_layer("mor", "reshape", ["mo"], shape=[1, 5, NT])
add_layer("pm", "maxpool", ["mor"], size=[1, NT], stride=[1, NT])   # [maxS, -minS, maxG, -minG, area*offset]
w = np.zeros((3, 4, R, R)); w[0, 1] = 1; w[1, 0] = 1; w[2, 2] = 1; w[2, 3] = -1
conv("sc", "g4", w, [0, 0, 0])                          # [area, flux, red-minus-blue flux]

# ---- 6. ratios by the norm trick: a channel holding [x, -x, K*y, -K*y] is normalised to about sqrt(2)*x/(K*y) ----
KD = 100.0
NC = 5
w1 = np.zeros((NC * 4, 5)); w2 = np.zeros((NC * 4, 3))


def put(wm, c, j, src, val):
    sgn = [1, -1]
    for q in range(2): wm[4 * c + j + q, src] += sgn[q] * val


put(w1, 0, 0, 1, -1.0); put(w1, 0, 2, 0, KD)            # q2S = minS / maxS
put(w1, 1, 0, 3, -1.0); put(w1, 1, 2, 2, KD)            # q2G = minG / maxG
put(w1, 2, 0, 0, 1.0);  put(w2, 2, 2, 0, KD * 100.0)    # sz  = maxS / (100 area) = (semi-major)^2 / 400
put(w2, 3, 0, 2, 1.0);  put(w2, 3, 2, 1, KD)            # col = (red - blue flux) / luminance flux
put(w1, 4, 0, 4, 1.0);  put(w2, 4, 2, 0, KD * 10.0)     # off = centroid offset / 10 px
dense("dq1", "pm", w1, np.zeros(NC * 4))
dense("dq2", "sc", w2, np.zeros(NC * 4))
add_layer("dq", "add", ["dq1", "dq2"])
add_layer("dqr", "reshape", ["dq"], shape=[NC, 1, 4])
add_layer("dn", "norm", ["dqr"])

# ---- 7. hand-written rule table: hinge units on the features, then class scores = minus the penalties ----------
FS = KD / np.sqrt(2.0)
feat = {"q2S": (0, FS), "q2G": (1, FS), "sz": (2, FS), "col": (3, FS), "off": (4, 10 * FS)}   # (channel, scale)
units = [  # (name, {feature: coefficient}, constant)   unit = relu(sum coeff*feature + constant)
    ("not_round",   {"q2S": -0.5, "q2G": -0.5}, 0.72),   # mean squared axis ratio below 0.72 (b/a < 0.85)
    ("too_round",   {"q2S": 0.5, "q2G": 0.5}, -0.75),    # too round for an in-between galaxy
    ("too_long",    {"q2S": -0.5, "q2G": -0.5}, 0.22),   # b/a below ~0.47: too elongated for in-between / face-on disc
    ("cig_round",   {"q2S": 1.0}, -0.25),                # cigar: b/a between ~0.32 and 0.5
    ("cig_thin",    {"q2S": -1.0}, 0.10),
    ("eo8_round",   {"q2S": 1.0}, -0.14),                # edge-on: footprint b/a below ~0.37
    ("eo9_round",   {"q2S": 1.0}, -0.18),
    ("blue_for_E",  {"col": -1.0}, 0.22),                # red-sequence light has red > blue
    ("red_for_8",   {"col": 1.0}, -0.25),
    ("blue_for_9",  {"col": -1.0}, 0.20),
    ("big_for_3",   {"sz": 1.0}, -0.12),                 # in-between smooth galaxies are small in the frame
    ("small_for_S", {"sz": -1.0}, 0.12),                 # spirals are resolved, larger discs
    ("red_for_S",   {"col": 1.0}, -0.30),
    ("no_bar",      {"q2S": -1.0, "q2G": 1.0}, 0.08),    # bar: bright inner light more elongated than the footprint
    ("blue_for_6",  {"col": -1.0}, 0.10),                # tight spirals: bigger red bulge
    ("red_for_7",   {"col": 1.0}, -0.18),                # loose spirals: blue
    ("offcentre",   {"off": 1.0, "sz": -4.0}, -1.5),     # footprint centroid away from the image centre
    ("centred",     {"off": -1.0, "sz": 4.0}, 2.5),
    ("big_for_4",   {"sz": 1.0}, -0.30),
    ("blue_for_1",  {"col": -1.0}, 0.10),
]
H = len(units)
w = np.zeros((H, NC * 4)); b = np.zeros(H)
for i, (_, coef, const) in enumerate(units):
    for f, cv in coef.items():
        ch, sc = feat[f]; w[i, 4 * ch] += cv * sc
    b[i] = const
dense("h1", "dn", w, b)
add_layer("h1r", "relu", ["h1"])
ui = {u[0]: i for i, u in enumerate(units)}
rules = {  # class: ({unit: penalty weight}, bias)
    0: ({"centred": 1.5, "red_for_7": 8}, -0.8),
    1: ({"centred": 1.5, "blue_for_1": 4}, -0.5),
    2: ({"not_round": 20, "blue_for_E": 10, "offcentre": 1}, 0.0),
    3: ({"too_round": 20, "too_long": 20, "big_for_3": 15, "offcentre": 1}, 0.0),
    4: ({"cig_round": 20, "cig_thin": 30, "blue_for_E": 10, "big_for_4": 10, "offcentre": 1}, -1.5),
    5: ({"too_long": 20, "small_for_S": 15, "red_for_S": 10, "no_bar": 15, "offcentre": 1}, 0.0),
    6: ({"too_long": 20, "small_for_S": 15, "red_for_S": 10, "blue_for_6": 8, "offcentre": 1}, -0.3),
    7: ({"too_long": 20, "small_for_S": 15, "red_for_S": 10, "red_for_7": 8, "offcentre": 1}, -0.3),
    8: ({"eo8_round": 30, "red_for_8": 8, "offcentre": 1}, 0.0),
    9: ({"eo9_round": 30, "blue_for_9": 8, "offcentre": 1}, 0.0),
}
w = np.zeros((10, H)); b = np.zeros(10)
for k, (pen, bias) in rules.items():
    for u, pv in pen.items(): w[k, ui[u]] = -pv
    b[k] = bias
dense("out", "h1r", w, b)

json.dump({"layers": layers}, open(os.path.join(out_dir, "network.json"), "w"))
np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
