"""Hand-crafted galaxy classifier.

Pipeline (see /app/out/method.md):
  image (3, 128, 128)
  -> conv 1x1: derive L=(R+G+B)/3, Y=R-B, B'=B-R
  -> norm: per-channel zero-mean / unit-std over spatial positions
  -> relu: keeps 'above the mean' pixels (galaxy pixels)
  -> avgpool 2x2 stride 2 -> (3, 64, 64)
  -> conv 64x64 difference templates -> (K, 1, 1)
  -> relu, reshape, dense K -> 10 class scores.

Templates come from closed-form Gaussian formulas as differences:
 - A/D/E: center-vs-surround DoG, bandpass for round blobs of a given sigma
 - B/F:   moderate elongation at angle theta minus elongation at theta+90
 - C/G/H: thin shape at angle theta minus thin at theta+90
 - I:     two-blob at angle theta minus one central blob

Dense weights encode class prototypes:
 - 2/3  : compact / medium round yellow blob, no blue disk
 - 4    : moderate yellow elongation (cigar), not thin
 - 5    : yellow bar + blue disk (strong F on yellow)
 - 6/7  : round blue, tight (smaller E) vs loose (larger E)
 - 8    : thin + blue
 - 9    : thin + yellow central bulge
 - 1    : two-blob pair (merging)
Nothing comes from the data: all numbers are from the code itself.
"""
import json, sys
from pathlib import Path
import numpy as np

if len(sys.argv) < 2:
    print("usage: python3 build.py NEW_FOLDER"); sys.exit(1)

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)

SIZE = 64
CENTER = (SIZE - 1) / 2.0


def gauss(sx, sy, angle_deg=0.0, cx=CENTER, cy=CENTER, size=SIZE):
    rows, cols = np.mgrid[:size, :size].astype(np.float64)
    dx = cols - cx
    dy = rows - cy
    t = np.deg2rad(angle_deg)
    xp = dx * np.cos(t) + dy * np.sin(t)
    yp = -dx * np.sin(t) + dy * np.cos(t)
    return np.exp(-0.5 * ((xp / sx) ** 2 + (yp / sy) ** 2))


def unit(g):
    s = g.sum()
    return g / s if s > 0 else g


def dog(sigma, scale=2.5):
    return unit(gauss(sigma, sigma)) - unit(gauss(sigma * scale, sigma * scale))


def elong_diff(sx, sy, angle):
    return unit(gauss(sx, sy, angle)) - unit(gauss(sx, sy, angle + 90.0))


def pair_diff(sigma, sep, angle):
    tr = np.deg2rad(angle)
    dx, dy = sep * np.cos(tr), sep * np.sin(tr)
    g1 = gauss(sigma, sigma, 0, cx=CENTER + dx, cy=CENTER + dy)
    g2 = gauss(sigma, sigma, 0, cx=CENTER - dx, cy=CENTER - dy)
    return unit(g1 + g2) - unit(gauss(sigma * 1.3, sigma * 1.3))


templates = []

A_SIGMAS = [2.5, 4.0, 6.0, 9.0, 13.0]
for sigma in A_SIGMAS:
    w = np.zeros((3, SIZE, SIZE)); w[0] = dog(sigma)
    templates.append(("A", w))

B_ANGLES = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
for a in B_ANGLES:
    w = np.zeros((3, SIZE, SIZE)); w[0] = elong_diff(8.0, 2.5, a)
    templates.append(("B", w))

for a in B_ANGLES:
    w = np.zeros((3, SIZE, SIZE)); w[0] = elong_diff(14.0, 1.0, a)
    templates.append(("C", w))

D_SIGMAS = [3.0, 5.0, 8.0]
for sigma in D_SIGMAS:
    w = np.zeros((3, SIZE, SIZE)); w[1] = dog(sigma)
    templates.append(("D", w))

E_SIGMAS = [3.0, 5.0, 8.0, 12.0]
for sigma in E_SIGMAS:
    w = np.zeros((3, SIZE, SIZE)); w[2] = dog(sigma)
    templates.append(("E", w))

F_ANGLES = [0, 45, 90, 135]
for a in F_ANGLES:
    w = np.zeros((3, SIZE, SIZE)); w[1] = elong_diff(8.0, 2.5, a)
    templates.append(("F", w))

for a in F_ANGLES:
    w = np.zeros((3, SIZE, SIZE)); w[1] = elong_diff(14.0, 1.0, a)
    templates.append(("G", w))

for a in F_ANGLES:
    w = np.zeros((3, SIZE, SIZE)); w[2] = elong_diff(14.0, 1.0, a)
    templates.append(("H", w))

# I template: use a wider separation to match typical merger separations in 64x64
for a in F_ANGLES:
    w = np.zeros((3, SIZE, SIZE)); w[0] = pair_diff(3.5, 11.0, a)
    templates.append(("I", w))

K = len(templates)

W_tmpl = np.stack([t[1] for t in templates], 0).astype(np.float32)
b_tmpl = np.zeros(K, dtype=np.float32)

W_mix = np.zeros((3, 3, 1, 1), np.float32)
W_mix[0, :, 0, 0] = [0.333, 0.333, 0.333]
W_mix[1, :, 0, 0] = [1.0,   0.0,   -1.0]
W_mix[2, :, 0, 0] = [-1.0,  0.0,    1.0]
b_mix = np.zeros(3, np.float32)


W_fc = np.zeros((10, K), np.float32)
b_fc = np.zeros(10, np.float32)


def grp(prefix):
    return [i for i, (n, _) in enumerate(templates) if n == prefix]


A = grp("A"); B = grp("B"); C = grp("C")
D = grp("D"); E = grp("E"); F = grp("F")
G = grp("G"); H = grp("H"); I = grp("I")


def add_w(cls, ids, v):
    for i in ids: W_fc[cls, i] += v


# Class 0 (disturbed): weak default + some elongation / pair evidence
add_w(0, I, 0.3)
add_w(0, B, 0.1)
add_w(0, [A[3], A[4]], 0.1)
b_fc[0] = 0.1

# Class 1 (merging): strong two-blob signal
add_w(1, I, 2.5)
add_w(1, [A[0], A[1]], -0.4)
add_w(1, D, -0.2)

# Class 2 (round smooth): compact yellow, NO blue
add_w(2, [A[0], A[1]], 0.5)
add_w(2, [A[2]], -0.3)
add_w(2, [A[3], A[4]], -0.4)
add_w(2, [D[0], D[1]], 0.3)
add_w(2, [D[2]], -0.2)
add_w(2, E, -1.3)
add_w(2, B, -0.3)
add_w(2, C, -0.5)
add_w(2, H, -0.5)
add_w(2, G, -0.3)
add_w(2, I, -0.5)

# Class 3 (in-between round smooth): medium round, NOT blue, NOT very small
add_w(3, [A[0]], -0.3)
add_w(3, [A[1], A[2]], 0.5)
add_w(3, [A[3]], 0.1)
add_w(3, [A[4]], -0.2)
add_w(3, [D[0]], -0.2)
add_w(3, [D[1], D[2]], 0.3)
add_w(3, E, -1.0)
add_w(3, C, -0.5)
add_w(3, B, -0.2)
add_w(3, H, -0.5)
add_w(3, G, -0.3)
add_w(3, I, -0.4)

# Class 4 (cigar smooth): moderate yellow elongation
add_w(4, B, 0.6)
add_w(4, F, 1.8)
add_w(4, C, -0.3)
add_w(4, G, -0.5)
add_w(4, E, -1.0)
add_w(4, H, -0.5)
add_w(4, [A[0], A[1]], -0.3)
add_w(4, I, -0.3)
b_fc[4] = 0.15

# Class 5 (barred spiral): yellow bar + blue disk
add_w(5, F, 1.2)
add_w(5, B, 0.3)
add_w(5, [E[1], E[2], E[3]], 0.7)
add_w(5, C, -0.3)
add_w(5, H, -0.3)
add_w(5, G, -0.4)
add_w(5, I, -0.3)

# Class 6 (unbarred tight spiral): medium round blue, no bar, smaller than class 7
add_w(6, [A[1], A[2]], 0.3)
add_w(6, [E[1]], 1.2)
add_w(6, [E[2]], 0.5)
add_w(6, [E[3]], -0.3)
add_w(6, F, -0.5)
add_w(6, G, -0.3)
add_w(6, H, -0.4)
add_w(6, C, -0.3)
add_w(6, [D[1], D[2]], -0.3)
add_w(6, I, -0.3)

# Class 7 (unbarred loose spiral): large / extended blue, not compact
add_w(7, [A[2], A[3]], 0.3)
add_w(7, [A[0]], -0.3)
add_w(7, [E[0], E[1]], -0.3)
add_w(7, [E[2]], 0.6)
add_w(7, [E[3]], 1.5)
add_w(7, F, -0.5)
add_w(7, G, -0.3)
add_w(7, H, -0.4)
add_w(7, C, -0.3)
add_w(7, [D[0], D[1], D[2]], -0.3)
add_w(7, I, -0.3)

# Class 8 (edge-on without bulge): thin + blue preferred
add_w(8, C, 0.6)
add_w(8, H, 1.8)
add_w(8, G, -0.5)
add_w(8, A[:3], -0.3)
add_w(8, D, -0.3)
add_w(8, F, -0.2)
add_w(8, I, -0.3)

# Class 9 (edge-on with bulge): thin + yellow central bulge
add_w(9, C, 0.6)
add_w(9, G, 1.8)
add_w(9, H, -0.5)
add_w(9, E, -0.3)
add_w(9, A[:3], -0.2)
add_w(9, F, -0.1)
add_w(9, I, -0.3)


np.savez(out / "weights.npz",
         **{"mix.W": W_mix, "mix.b": b_mix,
            "tmpl.W": W_tmpl, "tmpl.b": b_tmpl,
            "fc.W": W_fc, "fc.b": b_fc})


net = {"layers": [
    {"name": "mix",  "op": "conv",    "inputs": ["image"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "n1",   "op": "norm",    "inputs": ["mix"]},
    {"name": "r1",   "op": "relu",    "inputs": ["n1"]},
    {"name": "p1",   "op": "avgpool", "inputs": ["r1"], "size": [2, 2], "stride": [2, 2]},
    {"name": "tmpl", "op": "conv",    "inputs": ["p1"], "out": K, "kernel": [SIZE, SIZE], "stride": 1, "padding": 0},
    {"name": "r2",   "op": "relu",    "inputs": ["tmpl"]},
    {"name": "flat", "op": "reshape", "inputs": ["r2"], "shape": [K]},
    {"name": "fc",   "op": "dense",   "inputs": ["flat"], "out": 10},
]}
(out / "network.json").write_text(json.dumps(net))
