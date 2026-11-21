"""Hand-designed CIFAR-10-style classifier. Every weight is produced by formulas
and by a hand-written rule table encoding general knowledge about the ten
classes (sky/water/vegetation/road layout, wheels, fur vs smooth metal, white
boxes, thin legs, rings). No data is read; the script is self-contained."""
import json
import os
import sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

K = 5
C2 = (K - 1) / 2.0


def grid():
    yy, xx = np.mgrid[0:K, 0:K]
    return yy - C2, xx - C2


def gabor(theta_deg, phase, sigma=1.1, lam=3.0, gamma=0.5):
    dy, dx = grid()
    t = np.deg2rad(theta_deg)
    xp = dx * np.cos(t) + dy * np.sin(t)
    yp = -dx * np.sin(t) + dy * np.cos(t)
    g = np.exp(-(xp ** 2 + gamma ** 2 * yp ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xp / lam + phase)
    g -= g.mean()
    g /= (np.sqrt((g ** 2).sum()) + 1e-12)
    return g


def gauss(sigma=1.6):
    dy, dx = grid()
    g = np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def delta(dy0, dx0):
    m = np.zeros((K, K)); m[int(dy0 + C2), int(dx0 + C2)] = 1.0
    return m


def ones():
    return np.ones((K, K)) / (K * K)


def mean32():
    return np.ones((K, K)) / 32.0


def hrow():
    m = np.zeros((K, K)); m[int(C2), :] = 1.0 / K
    return m


def vcol():
    m = np.zeros((K, K)); m[:, int(C2)] = 1.0 / K
    return m


def diag1():
    m = np.zeros((K, K))
    for i in range(K): m[i, i] = 1.0 / K
    return m


def diag2():
    m = np.zeros((K, K))
    for i in range(K): m[i, K - 1 - i] = 1.0 / K
    return m


LUM = np.array([0.299, 0.587, 0.114])

# ------------------------------------------------------------------ stage 1
S1, s1_W, s1_b = [], [], []


def add_s1(name, w3, b=0.0):
    S1.append(name); s1_W.append(np.asarray(w3, float)); s1_b.append(b)


for o in range(8):
    th = o * 22.5
    add_s1(f"Lb{o}", np.stack([gabor(th, 0.0) * LUM[c] for c in range(3)]))
    add_s1(f"Ld{o}", np.stack([gabor(th, np.pi) * LUM[c] for c in range(3)]))
    add_s1(f"Le{o}", np.stack([gabor(th, -np.pi / 2) * LUM[c] for c in range(3)]))
    add_s1(f"Lf{o}", np.stack([gabor(th, np.pi / 2) * LUM[c] for c in range(3)]))
for o in range(4):
    th = o * 45.0
    add_s1(f"RGc{o}", np.stack([gabor(th, 0.0, 1.6, 4.5), -gabor(th, 0.0, 1.6, 4.5), np.zeros((K, K))]) * 1.2)
    add_s1(f"RGs{o}", np.stack([gabor(th, -np.pi / 2, 1.6, 4.5), gabor(th, np.pi / 2, 1.6, 4.5), np.zeros((K, K))]) * 1.2)
    add_s1(f"BYc{o}", np.stack([-0.5 * gabor(th, 0.0, 1.6, 4.5), -0.5 * gabor(th, 0.0, 1.6, 4.5), gabor(th, 0.0, 1.6, 4.5)]) * 1.2)
    add_s1(f"BYs{o}", np.stack([-0.5 * gabor(th, -np.pi / 2, 1.6, 4.5), -0.5 * gabor(th, -np.pi / 2, 1.6, 4.5), gabor(th, -np.pi / 2, 1.6, 4.5)]) * 1.2)

g = gauss(1.6)
add_s1("dogOn", np.stack([g * LUM[c] for c in range(3)]) - np.stack([gauss(2.4) * LUM[c] for c in range(3)]))
add_s1("dogOff", np.stack([gauss(2.4) * LUM[c] for c in range(3)]) - np.stack([g * LUM[c] for c in range(3)]))
add_s1("bL", np.stack([g * LUM[c] for c in range(3)]))
add_s1("bRG", np.stack([g, -g, np.zeros((K, K))]))
add_s1("bGR", np.stack([-g, g, np.zeros((K, K))]))
add_s1("bBY", np.stack([-0.5 * g, -0.5 * g, g]))
add_s1("bYB", np.stack([0.5 * g, 0.5 * g, -g]))
add_s1("bGM", np.stack([-0.5 * g, g, -0.5 * g]))
add_s1("bMG", np.stack([0.5 * g, -g, 0.5 * g]))
add_s1("bRB", np.stack([g, np.zeros((K, K)), -g]))
add_s1("bBR", np.stack([-g, np.zeros((K, K)), g]))
add_s1("hi", np.stack([g * LUM[c] for c in range(3)]), b=-0.72)
add_s1("sh", np.stack([-g * LUM[c] for c in range(3)]), b=0.32)

I1 = {n: i for i, n in enumerate(S1)}
N1 = len(S1)

# ------------------------------------------------------------------ stage 2 (base detectors)
S2, s2_terms, s2_bias = [], [], []


def ch2(name, terms, bias=0.0):
    S2.append(name); s2_terms.append(terms); s2_bias.append(bias)


def T(n, pat, g=1.0):
    return (n, pat, g)


RING1 = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
RING2 = [(0, 2), (2, 0), (0, -2), (-2, 0), (1, 2), (2, 1), (-1, 2), (-2, 1),
         (1, -2), (2, -1), (-1, -2), (-2, -1), (2, 2), (2, -2), (-2, 2), (-2, -2)]


def ring(barchar, offs):
    terms = []
    n = len(offs)
    for (dy, dx) in offs:
        phi = np.degrees(np.arctan2(dy, dx)) % 180.0
        o = int(round(phi / 22.5)) % 8
        terms.append(T(f"{barchar}{o}", delta(dy, dx), 1.0 / np.sqrt(n)))
    return terms


LB = [f"Lb{o}" for o in range(8)] + [f"Ld{o}" for o in range(8)] + [f"Le{o}" for o in range(8)] + [f"Lf{o}" for o in range(8)]
OPP = ["bRG", "bGR", "bBY", "bYB", "bGM", "bMG"]

ch2("ring_d1", ring("Ld", RING1))
ch2("ring_d2", ring("Ld", RING2))
ch2("ring_b1", ring("Lb", RING1))
ch2("ring_b2", ring("Lb", RING2))
ch2("line_h", [T("Le4", hrow()), T("Lf4", hrow())])
ch2("line_v", [T("Le0", vcol()), T("Lf0", vcol())])
ch2("line_d1", [T("Le2", diag1()), T("Lf2", diag1())])
ch2("line_d2", [T("Le6", diag2()), T("Lf6", diag2())])
ch2("texture", [T(n, mean32(), 1.0) for n in LB])
ch2("edgeh_d", [T("Le4", ones()), T("Lf4", ones())])
ch2("edgev_d", [T("Le0", ones()), T("Lf0", ones())])
ch2("elong_h", [T("Le4", hrow()), T("Lf4", hrow()), T("Le0", ones(), -0.5), T("Lf0", ones(), -0.5)])
ch2("elong_v", [T("Le0", vcol()), T("Lf0", vcol()), T("Le4", ones(), -0.5), T("Lf4", ones(), -0.5)])
ch2("blue", [T("bBY", delta(0, 0))])
ch2("green", [T("bGM", delta(0, 0))])
ch2("red", [T("bRG", delta(0, 0)), T("bRB", delta(0, 0))])
ch2("yellow", [T("bYB", delta(0, 0)), T("bRG", delta(0, 0), 0.5)])
ch2("warm", [T("bRB", delta(0, 0))])
ch2("chroma", [T(n, delta(0, 0)) for n in OPP])
ch2("achrom_b", [T("bL", delta(0, 0), 1.0)] + [T(n, delta(0, 0), -0.8) for n in OPP], bias=-0.35)
ch2("achrom_m", [T("bL", delta(0, 0), 1.0)] + [T(n, delta(0, 0), -0.8) for n in OPP], bias=-0.15)
ch2("bright", [T("bL", delta(0, 0))], bias=-0.55)
ch2("dark", [T("sh", delta(0, 0))])
ch2("highlight", [T("hi", delta(0, 0))])
ch2("smooth", [T(n, mean32(), -1.0) for n in LB], bias=0.45)
ch2("rough", [T(n, mean32(), 1.0) for n in LB], bias=-0.90)

I2 = {n: i for i, n in enumerate(S2)}
N2 = len(S2)
s2_W = np.zeros((N2, N1, K, K))
for oi, terms in enumerate(s2_terms):
    sl = np.zeros((N1, K, K))
    for (n, pat, gg) in terms:
        sl[I1[n]] += pat * gg
    s2_W[oi] = sl
s2_b = np.array(s2_bias, float)

# ------------------------------------------------------------------ stage 3 (1x1 combos over stage 2)
S3, s3_terms, s3_bias = [], [], []


def ch3(name, terms, bias=0.0):
    S3.append(name); s3_terms.append(terms); s3_bias.append(bias)


def U(n, g=1.0):
    return (n, g)


ch3("foliage", [U("green", 1.0), U("rough", 0.3)], bias=-0.10)
ch3("fur", [U("rough", 0.6), U("warm", 0.8), U("green", -1.0), U("blue", -1.0)], bias=-0.15)
ch3("sky", [U("blue", 1.0), U("bright", 0.4), U("smooth", 0.5)], bias=-0.15)
ch3("water", [U("blue", 1.0), U("bright", -0.5), U("smooth", 0.4)], bias=-0.08)
ch3("road", [U("achrom_m", 0.8), U("smooth", 0.5), U("highlight", -0.5)], bias=-0.10)
ch3("metal", [U("highlight", 0.5), U("line_h", 0.6), U("line_v", 0.4), U("smooth", 0.4)], bias=-0.10)
ch3("wheel", [U("ring_d2", 1.0), U("dark", 0.6)], bias=-0.15)
ch3("white_p", [U("achrom_b", 1.0), U("smooth", 0.5)], bias=-0.10)
ch3("green_sm", [U("green", 1.0), U("smooth", 0.5)], bias=-0.08)
ch3("dark_blob", [U("ring_d1", 1.0), U("dark", 0.8)], bias=-0.15)
ch3("warm_bg", [U("warm", 0.8), U("achrom_m", 0.3), U("smooth", 0.4)], bias=-0.15)
ch3("hbound", [U("line_h", 1.5)], bias=-0.30)
ch3("vbound", [U("line_v", 1.5)], bias=-0.30)
for p in ["elong_h", "elong_v", "green", "chroma", "line_v", "line_h", "achrom_m", "achrom_b", "warm", "dark", "smooth", "rough"]:
    ch3(p, [U(p, 1.0)])

I3 = {n: i for i, n in enumerate(S3)}
N3 = len(S3)
s3_W = np.zeros((N3, N2, 1, 1))
for oi, terms in enumerate(s3_terms):
    for (n, gg) in terms:
        s3_W[oi, I2[n], 0, 0] += gg
s3_b = np.array(s3_bias, float)

# ------------------------------------------------------------------ graph
layers = []
layers.append({"name": "s1", "op": "conv", "inputs": ["image"], "out": N1, "kernel": [K, K], "stride": 1, "padding": 2})
layers.append({"name": "r1", "op": "relu", "inputs": ["s1"]})
layers.append({"name": "p1", "op": "maxpool", "inputs": ["r1"], "size": [2, 2], "stride": [2, 2]})
layers.append({"name": "s2", "op": "conv", "inputs": ["p1"], "out": N2, "kernel": [K, K], "stride": 1, "padding": 2})
layers.append({"name": "r2", "op": "relu", "inputs": ["s2"]})
layers.append({"name": "s3", "op": "conv", "inputs": ["r2"], "out": N3, "kernel": [1, 1], "stride": 1, "padding": 0})
layers.append({"name": "r3", "op": "relu", "inputs": ["s3"]})
layers.append({"name": "h2p", "op": "avgpool", "inputs": ["r3"], "size": [4, 4], "stride": [4, 4]})
layers.append({"name": "h2f", "op": "reshape", "inputs": ["h2p"], "shape": [N3 * 16]})
layers.append({"name": "h2d", "op": "dense", "inputs": ["h2f"], "out": 10})
layers.append({"name": "h3p", "op": "avgpool", "inputs": ["r3"], "size": [16, 16], "stride": [16, 16]})
layers.append({"name": "h3f", "op": "reshape", "inputs": ["h3p"], "shape": [N3]})
layers.append({"name": "h3d", "op": "dense", "inputs": ["h3f"], "out": 10})
layers.append({"name": "out", "op": "add", "inputs": ["h2d", "h3d"]})

O = np.ones((4, 4))
TOP = O.copy(); TOP[2:, :] = 0
BOT = O.copy(); BOT[:2, :] = 0
MID = O.copy(); MID[0, :] = 0; MID[3, :] = 0
CTR = O.copy(); CTR[0, :] = 0; CTR[3, :] = 0; CTR[:, 0] = 0; CTR[:, 3] = 0
MASKS = {"ALL": O, "TOP": TOP, "BOT": BOT, "MID": MID, "CTR": CTR}

W_h2 = np.zeros((10, N3 * 16)); B_h2 = np.zeros(10)
W_h3 = np.zeros((10, N3)); B_h3 = np.zeros(10)


def rule(cls, head, chan, mask, w):
    ci = I3[chan]
    if head == "H3":
        W_h3[cls, ci] += w
        return
    M = MASKS[mask]
    for r in range(4):
        for c in range(4):
            if M[r, c]:
                W_h2[cls, ci * 16 + r * 4 + c] += w


# airplane: smooth bright sky above, slim bright body, no water/vegetation/wheels
rule(0, "H2", "sky", "TOP", 1.6); rule(0, "H2", "sky", "MID", 0.6)
rule(0, "H2", "smooth", "TOP", 0.6); rule(0, "H2", "white_p", "MID", 0.8)
rule(0, "H2", "water", "BOT", -1.6); rule(0, "H2", "foliage", "ALL", -0.7)
rule(0, "H2", "green", "ALL", -0.6); rule(0, "H2", "fur", "ALL", -0.8)
rule(0, "H2", "wheel", "BOT", -0.6); rule(0, "H2", "warm", "ALL", -0.4)
rule(0, "H3", "sky", "ALL", 0.5); rule(0, "H3", "smooth", "ALL", 0.3)
# automobile: wheels + grey road below, saturated glossy body, not a tall white box
rule(1, "H2", "wheel", "BOT", 1.6); rule(1, "H2", "road", "BOT", 1.2)
rule(1, "H2", "chroma", "MID", 0.6); rule(1, "H2", "metal", "MID", 0.8)
rule(1, "H2", "white_p", "TOP", -0.7); rule(1, "H2", "water", "BOT", -0.7)
rule(1, "H2", "foliage", "ALL", -0.5); rule(1, "H2", "fur", "ALL", -0.6)
rule(1, "H2", "sky", "TOP", -0.3); rule(1, "H3", "wheel", "ALL", 0.5)
# bird: uniform sky/field backdrop all around, small dark textured body, no wheels/panels
rule(2, "H2", "sky", "ALL", 0.7); rule(2, "H2", "smooth", "ALL", 0.5)
rule(2, "H2", "dark", "CTR", 0.4); rule(2, "H2", "fur", "CTR", 0.8)
rule(2, "H2", "wheel", "BOT", -0.5); rule(2, "H2", "white_p", "ALL", -0.2)
rule(2, "H2", "foliage", "ALL", 0.3); rule(2, "H3", "sky", "ALL", 0.3)
# cat: close-up fur, warm/grey, indoor warm backdrop, no sky/water/vegetation
rule(3, "H2", "fur", "CTR", 1.4); rule(3, "H2", "fur", "ALL", 0.5)
rule(3, "H2", "warm", "CTR", 0.6); rule(3, "H2", "achrom_m", "CTR", 0.5)
rule(3, "H2", "warm_bg", "TOP", 0.6); rule(3, "H2", "warm_bg", "BOT", 0.4)
rule(3, "H2", "foliage", "ALL", -0.7); rule(3, "H2", "sky", "ALL", -0.7)
rule(3, "H2", "water", "ALL", -0.7); rule(3, "H2", "wheel", "BOT", -0.4)
rule(3, "H3", "fur", "ALL", 0.4)
# deer: vegetation around, tan warm body, not smooth/white
rule(4, "H2", "foliage", "ALL", 1.0); rule(4, "H2", "foliage", "BOT", 0.6)
rule(4, "H2", "warm", "MID", 1.0); rule(4, "H2", "fur", "MID", 0.6)
rule(4, "H2", "wheel", "BOT", -0.6); rule(4, "H2", "white_p", "ALL", -0.5)
rule(4, "H2", "smooth", "ALL", -0.4); rule(4, "H2", "sky", "ALL", -0.3)
rule(4, "H2", "green", "CTR", -0.6)
# dog: fur, often white, often outdoor grass below
rule(5, "H2", "fur", "CTR", 1.2); rule(5, "H2", "fur", "ALL", 0.5)
rule(5, "H2", "achrom_b", "CTR", 0.7); rule(5, "H2", "warm", "CTR", 0.4)
rule(5, "H2", "foliage", "BOT", 0.5); rule(5, "H2", "sky", "ALL", -0.6)
rule(5, "H2", "water", "ALL", -0.6); rule(5, "H2", "wheel", "BOT", -0.4)
rule(5, "H3", "fur", "ALL", 0.4)
# frog: green close-up blob, dark wet surround, big dark eyes
rule(6, "H2", "green", "CTR", 1.6); rule(6, "H2", "foliage", "ALL", 0.6)
rule(6, "H2", "dark", "BOT", 0.5); rule(6, "H2", "dark_blob", "MID", 0.8); rule(6, "H2", "dark", "ALL", 0.4)
rule(6, "H2", "sky", "ALL", -0.7); rule(6, "H2", "wheel", "BOT", -0.5)
rule(6, "H2", "white_p", "ALL", -0.6); rule(6, "H2", "smooth", "ALL", -0.4)
rule(6, "H3", "green", "ALL", 0.4)
# horse: grass below and around, warm body, not smooth/white
rule(7, "H2", "foliage", "BOT", 1.4); rule(7, "H2", "foliage", "ALL", 0.6)
rule(7, "H2", "fur", "MID", 0.9); rule(7, "H2", "warm", "MID", 0.6)
rule(7, "H2", "wheel", "BOT", -0.6); rule(7, "H2", "white_p", "ALL", -0.5)
rule(7, "H2", "smooth", "ALL", -0.4); rule(7, "H3", "foliage", "ALL", 0.4)
# ship: water below and mid, sky above, bright hull
rule(8, "H2", "water", "BOT", 2.0); rule(8, "H2", "water", "MID", 1.2)
rule(8, "H2", "sky", "TOP", 0.9); rule(8, "H2", "smooth", "BOT", 0.7)
rule(8, "H2", "white_p", "MID", 0.6); rule(8, "H2", "foliage", "ALL", -0.6)
rule(8, "H2", "wheel", "BOT", -0.6); rule(8, "H2", "fur", "ALL", -0.6)
rule(8, "H3", "water", "ALL", 0.7)
# truck: tall white/flat box, wheels, road
rule(9, "H2", "white_p", "MID", 1.2); rule(9, "H2", "white_p", "TOP", 0.5)
rule(9, "H2", "wheel", "BOT", 1.0); rule(9, "H2", "road", "BOT", 1.0); rule(9, "H2", "sky", "TOP", -0.8)
rule(9, "H2", "smooth", "MID", 0.5); rule(9, "H2", "smooth", "TOP", 0.5); rule(9, "H2", "foliage", "ALL", -0.5)
rule(9, "H2", "fur", "ALL", -0.6); rule(9, "H2", "chroma", "MID", -0.3)
rule(9, "H3", "white_p", "ALL", 0.5)

BIAS = [0.0] * 10
B_h2 += np.array(BIAS, float)

weights = {
    "s1.W": np.stack(s1_W).astype(np.float32), "s1.b": np.array(s1_b, np.float32),
    "s2.W": s2_W.astype(np.float32), "s2.b": s2_b.astype(np.float32),
    "s3.W": s3_W.astype(np.float32), "s3.b": s3_b.astype(np.float32),
    "h2d.W": W_h2.astype(np.float32), "h2d.b": B_h2.astype(np.float32),
    "h3d.W": W_h3.astype(np.float32), "h3d.b": B_h3.astype(np.float32),
}

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": layers}, f)
np.savez(os.path.join(OUT, "weights.npz"), **weights)
print("wrote", OUT, "s1", N1, "s2", N2, "s3", N3, "weights", sum(int(np.prod(v.shape)) for v in weights.values()))
