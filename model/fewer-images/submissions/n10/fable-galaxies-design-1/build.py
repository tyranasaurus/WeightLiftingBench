"""Writes network.json and weights.npz for a hand-designed galaxy classifier. Reads no files.
Every number comes from formulas and hand-chosen constants (general knowledge of galaxy classes and of the
Legacy Survey viewer's asinh colour stretch, whose sky level is asinh(0.6)/sqrt(20) = 0.127)."""
import sys, json, os
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

N = 32                                   # working grid after 4x4 average pooling
yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
cx = cy = (N - 1) / 2.0                  # galaxy sits at the image centre
dx, dy = xx - cx, yy - cy
r = np.sqrt(dx * dx + dy * dy)
th = np.arctan2(dy, dx)
win = lambda s: np.exp(-r * r / (2.0 * s * s))
W6, W15 = win(6.0), win(1.5)             # whole-galaxy window and core window (in 4-pixel units)

# ---- pixel stage: luminance above sky, blue excess, red excess -------------------------------------------
c1W = np.zeros((3, 3, 1, 1)); c1b = np.zeros(3)
c1W[0, :, 0, 0] = 1.0 / 3.0; c1b[0] = -0.16          # luminance minus (sky 0.127 + noise margin)
c1W[1, 0, 0, 0], c1W[1, 2, 0, 0], c1b[1] = -1.0, 1.0, -0.03   # blue minus red  (g band bright: star forming)
c1W[2, 0, 0, 0], c1W[2, 2, 0, 0], c1b[2] = 1.0, -1.0, -0.03   # red minus blue  (z band bright: old stars)

def place(t, ch):
    v = np.zeros((3, N, N)); v[ch] = t
    return v.reshape(-1)

# ---- rotation-invariant angular harmonics: 16 phases each, modulus taken by a max over phases -------------
K = 16
rows = []
for k in range(K):                                   # m = 2: elongation, (1-q)/(1+q) for axis ratio q
    ph = np.pi * k / K
    rows.append(place(W6 * (r * r / (r * r + 1.0)) * np.cos(2 * (th - ph)), 0))
for k in range(K):                                   # m = 4: thinness (about e2^2 for ellipses, near e2 for lines)
    ph = 0.5 * np.pi * k / K
    rows.append(place(W6 * (r * r / (r * r + 2.0)) * np.cos(4 * (th - ph)), 0))
for k in range(K):                                   # m = 1: lopsidedness (second nucleus, tidal distortion)
    ph = 2 * np.pi * k / K
    rows.append(place(W6 * (r / (r + 1.0)) * np.cos(th - ph), 0))
dMW = np.stack(rows); dMb = np.zeros(3 * K)

# ---- hand-typed class prototypes in ratio-feature space (each feature divided by the windowed flux T) ------
#            e2    e4    a1    conc  blue  red
proto = np.array([
    [0.20, 0.06, 0.20, 0.35, 0.12, 0.15],   # 0 disturbed
    [0.25, 0.08, 0.30, 0.35, 0.10, 0.20],   # 1 merging
    [0.04, 0.02, 0.05, 0.55, 0.00, 0.40],   # 2 round smooth
    [0.20, 0.04, 0.05, 0.50, 0.00, 0.40],   # 3 in-between smooth
    [0.40, 0.14, 0.05, 0.45, 0.02, 0.35],   # 4 cigar smooth
    [0.15, 0.04, 0.08, 0.35, 0.12, 0.15],   # 5 barred spiral
    [0.10, 0.03, 0.07, 0.35, 0.10, 0.18],   # 6 tight spiral
    [0.15, 0.05, 0.12, 0.28, 0.25, 0.05],   # 7 loose spiral
    [0.60, 0.40, 0.08, 0.25, 0.15, 0.10],   # 8 edge-on, no bulge
    [0.50, 0.25, 0.06, 0.40, 0.02, 0.35],   # 9 edge-on, bulge
])
scale = np.array([0.08, 0.08, 0.10, 0.15, 0.10, 0.15])          # hand-chosen tolerance of each feature
prior = np.array([1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873], dtype=np.float64)
prior /= prior.sum()                                              # known class frequencies of Galaxy10 DECaLS
gain = proto / scale ** 2                                         # nearest-prototype linear discriminant
offset = -0.5 * (proto ** 2 / scale ** 2).sum(1) + np.log(prior)  # multiplies the flux T, so scores stay scale-free

tT, tC, tB, tR = place(W6, 0), place(W15, 0), place(W6, 1), place(W6, 2)
dSW = offset[:, None] * tT[None] + gain[:, 3:4] * tC[None] + gain[:, 4:5] * tB[None] + gain[:, 5:6] * tR[None]
dSb = 1e-3 * prior                                                # fallback when nothing is above the sky
dRW = gain[:, [0, 1, 2]].copy(); dRb = np.zeros(10)

layers = [
    {"name": "pool", "op": "avgpool", "inputs": ["image"], "size": [4, 4], "stride": [4, 4]},
    {"name": "c1", "op": "conv", "inputs": ["pool"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "r1", "op": "relu", "inputs": ["c1"]},
    {"name": "dS", "op": "dense", "inputs": ["r1"], "out": 10},
    {"name": "dM", "op": "dense", "inputs": ["r1"], "out": 3 * K},
    {"name": "mr", "op": "reshape", "inputs": ["dM"], "shape": [3, K, 1]},
    {"name": "mx", "op": "maxpool", "inputs": ["mr"], "size": [K, 1], "stride": [K, 1]},
    {"name": "dR", "op": "dense", "inputs": ["mx"], "out": 10},
    {"name": "scores", "op": "add", "inputs": ["dS", "dR"]},
]
with open(os.path.join(out_dir, "network.json"), "w") as f:
    json.dump({"layers": layers}, f, indent=1)
f32 = lambda a: np.asarray(a, dtype=np.float32)
np.savez(os.path.join(out_dir, "weights.npz"), **{
    "c1.W": f32(c1W), "c1.b": f32(c1b), "dS.W": f32(dSW), "dS.b": f32(dSb),
    "dM.W": f32(dMW), "dM.b": f32(dMb), "dR.W": f32(dRW), "dR.b": f32(dRb)})
