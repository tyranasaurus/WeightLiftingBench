"""Hand-designed galaxy classifier. Every number comes from formulas and hand-chosen rule tables in this file.
Usage: python3 build.py OUT_DIR   (reads no files)

Pipeline
  c0   1x1 conv + relu: sky-subtracted luminance, red and blue  (sky = hand-chosen "near black" level)
  d1   dense: centre-windowed moment functionals written as comparisons  a - t*b  (ratio > threshold tests)
         elongation  E = (Mxx-Myy)cos2th + 2Mxy sin2th over Mxx+Myy, for 12 angles and 6 thresholds
         centroid offset from the frame centre, for 12 directions and 2 thresholds
         colour (red - blue)/luminance, mean radius, central surface brightness
  d2   dense: sum over angles (acts as an OR), and the pair (s, s-1) so relu(s)-relu(s-1) is a 0..1 step
  d3   dense: additive hand-typed rule table over the thermometer codes + log class frequencies
"""
import sys, json, numpy as np
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)

N = 128
T0 = 0.16                                   # sky level: clearly above a near-black background
yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
x = xx - 63.5; y = yy - 63.5
r2 = x * x + y * y; r = np.sqrt(r2)
def gw(s): return np.exp(-r2 / (2.0 * s * s))

# ---------------- c0: [L - t0, R - t0, B - t0]
c0W = np.zeros((3, 3, 1, 1)); c0b = np.full(3, -T0)
c0W[0, :, 0, 0] = 1.0 / 3.0; c0W[1, 0, 0, 0] = 1.0; c0W[2, 2, 0, 0] = 1.0

# ---------------- thresholds chosen by hand
E_T = [0.12, 0.25, 0.42, 0.58, 0.72, 0.84]   # (1-q^2)/(1+q^2): q = 0.89, 0.77, 0.64, 0.52, 0.40, 0.29
O_T = [2.0, 4.0]                             # centroid offset, pixels
C_T = [-0.05, 0.08, 0.20, 0.33]              # (red - blue) / luminance
S_T = [3.5, 5.0, 7.0, 9.5, 12.0]             # flux-weighted mean radius, pixels
B_T = [0.25, 0.45, 0.65]                     # central surface brightness above sky
NTH = 12
KE, KO, KC, KS, KB = 20.0, 2.0, 5.0, 2.0, 50.0

rows, bias = [], []
def add_row(l=None, rd=None, bl=None, b=0.0):
    w = np.zeros((3, N, N))
    if l is not None: w[0] = l
    if rd is not None: w[1] = rd
    if bl is not None: w[2] = bl
    rows.append(w.reshape(-1)); bias.append(b)

w12, w14, w8, w16 = gw(12.0), gw(14.0), gw(8.0), gw(16.0)
# elongation: 6 thresholds x 12 angles
for t in E_T:
    for k in range(NTH):
        th = np.pi * k / NTH
        add_row(l=KE * w12 * (((x * x - y * y) * np.cos(2 * th) + 2 * x * y * np.sin(2 * th)) - t * r2) / 100.0)
# offset: 2 thresholds x 12 directions
for t in O_T:
    for k in range(NTH):
        ph = 2 * np.pi * k / NTH
        add_row(l=KO * w14 * ((x * np.cos(ph) + y * np.sin(ph)) - t))
n_or = len(rows)
# colour, size, central brightness: pairs (z, z - 1)
for t in C_T:
    for sh in (0.0, -1.0): add_row(l=-KC * t * w8, rd=KC * w8, bl=-KC * w8, b=sh)
for t in S_T:
    for sh in (0.0, -1.0): add_row(l=KS * w16 * (r - t), b=sh)
disk = (r <= 1.6).astype(np.float64); disk /= disk.sum()
for t in B_T:
    for sh in (0.0, -1.0): add_row(l=KB * disk, b=-KB * t + sh)
d1W = np.stack(rows); d1b = np.array(bias)
n1 = len(rows); n_pass = n1 - n_or

# ---------------- d2: OR over angles -> (s, s-1); pass-through for the rest
n_grp = len(E_T) + len(O_T)
n2 = 2 * n_grp + n_pass
d2W = np.zeros((n2, n1)); d2b = np.zeros(n2)
for g in range(n_grp):
    d2W[2 * g, g * NTH:(g + 1) * NTH] = 1.0
    d2W[2 * g + 1, g * NTH:(g + 1) * NTH] = 1.0; d2b[2 * g + 1] = -1.0
for i in range(n_pass):
    d2W[2 * n_grp + i, n_or + i] = 1.0

# ---------------- d3: rule table. Each row lists a class's score in each bin of the feature (0 = typical).
#        class:  0 disturbed, 1 merging, 2 round, 3 in-between, 4 cigar, 5 barred, 6 tight, 7 loose, 8 edge-on no bulge, 9 edge-on bulge
TAB_E = [[-1.0, -0.3, 0.0, -0.3, -1.0, -2.0, -3.0],
         [-1.2, -0.3, 0.0, -0.2, -0.8, -2.0, -3.5],
         [0.0, -0.7, -2.5, -4.0, -5.0, -5.0, -5.0],
         [-2.5, -0.8, 0.0, -0.5, -2.0, -4.0, -5.0],
         [-5.0, -4.0, -2.0, -0.3, 0.0, -0.8, -2.5],
         [-0.7, 0.0, -0.2, -1.0, -2.5, -4.0, -5.0],
         [-0.5, 0.0, -0.3, -1.2, -2.5, -4.0, -5.0],
         [-0.8, 0.0, -0.1, -0.6, -1.8, -3.0, -4.5],
         [-5.0, -5.0, -3.5, -2.0, -0.7, 0.0, 0.0],
         [-5.0, -4.0, -2.5, -1.0, -0.2, 0.0, -0.3]]
TAB_O = [[-0.6, 0.0, 0.0], [-1.5, 0.0, 0.3], [0.0, -0.8, -1.8], [0.0, -0.8, -1.8], [0.0, -0.8, -1.8],
         [0.0, -0.8, -1.8], [0.0, -0.8, -1.8], [0.0, -0.4, -1.2], [0.0, -0.5, -1.5], [0.0, -0.8, -1.8]]
TAB_C = [[0.0, 0.0, -0.3, -0.8, -1.5],      # blue ... red
         [-1.0, -0.5, 0.0, 0.0, -0.2],
         [-1.8, -1.0, -0.4, 0.0, 0.0],
         [-1.8, -1.0, -0.4, 0.0, 0.0],
         [-1.6, -0.8, -0.3, 0.0, -0.1],
         [-1.0, -0.3, 0.0, -0.3, -1.0],
         [-0.3, 0.0, 0.0, -0.6, -1.5],
         [0.0, 0.0, -0.3, -1.0, -2.0],
         [0.0, 0.0, -0.3, -1.2, -2.0],
         [-2.0, -1.2, -0.4, 0.0, 0.0]]
TAB_S = [[-2.0, -1.0, -0.3, 0.0, 0.0, 0.0],  # small ... large
         [-2.5, -1.2, -0.3, 0.0, 0.0, 0.0],
         [0.0, 0.0, -0.3, -1.0, -1.8, -2.5],
         [-0.3, 0.0, -0.2, -0.8, -1.5, -2.2],
         [-0.5, 0.0, 0.0, -0.5, -1.2, -2.0],
         [-2.5, -1.5, -0.5, 0.0, 0.0, -0.2],
         [-2.5, -1.5, -0.4, 0.0, 0.0, -0.2],
         [-2.5, -1.5, -0.5, 0.0, 0.0, 0.0],
         [-2.5, -1.5, -0.5, 0.0, 0.0, 0.0],
         [-1.5, -0.5, 0.0, 0.0, -0.3, -0.8]]
TAB_B = [[0.0, 0.0, -0.4, -0.8],             # faint ... bright centre
         [-0.8, -0.2, 0.0, 0.0],
         [-1.5, -0.5, 0.0, 0.0],
         [-1.5, -0.4, 0.0, 0.0],
         [-0.8, 0.0, 0.0, -0.3],
         [-1.5, -0.4, 0.0, 0.0],
         [-1.2, -0.2, 0.0, -0.2],
         [-0.3, 0.0, -0.3, -0.8],
         [0.0, -0.2, -1.2, -2.2],
         [-1.5, -0.3, 0.0, 0.0]]
# class frequencies of the public Galaxy10 DECaLS catalogue (general knowledge of the data set)
COUNTS = np.array([1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873], dtype=np.float64)

d3W = np.zeros((10, n2)); d3b = np.log(COUNTS / COUNTS.sum())
col = 0
for tab in (TAB_E, TAB_O, TAB_C, TAB_S, TAB_B):
    tab = np.array(tab)
    nth = tab.shape[1] - 1
    d3b += tab[:, 0]
    for j in range(nth):
        inc = tab[:, j + 1] - tab[:, j]
        d3W[:, col] = inc; d3W[:, col + 1] = -inc
        col += 2
assert col == n2

layers = [
    {"name": "c0", "op": "conv", "inputs": ["image"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "r0", "op": "relu", "inputs": ["c0"]},
    {"name": "d1", "op": "dense", "inputs": ["r0"], "out": int(n1)},
    {"name": "r1", "op": "relu", "inputs": ["d1"]},
    {"name": "d2", "op": "dense", "inputs": ["r1"], "out": int(n2)},
    {"name": "r2", "op": "relu", "inputs": ["d2"]},
    {"name": "d3", "op": "dense", "inputs": ["r2"], "out": 10},
]
(out / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
f32 = lambda a: np.asarray(a, dtype=np.float32)
np.savez(out / "weights.npz", **{"c0.W": f32(c0W), "c0.b": f32(c0b), "d1.W": f32(d1W), "d1.b": f32(d1b),
                                  "d2.W": f32(d2W), "d2.b": f32(d2b), "d3.W": f32(d3W), "d3.b": f32(d3b)})
