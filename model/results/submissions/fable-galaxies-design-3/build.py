"""Hand-designed galaxy morphology network. Every number comes from formulas and hand-chosen rules below.
No training data, no fitting. Usage: python3 build.py OUT_DIR"""
import sys, json, os
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
W = {}
layers = []

# ---- constants from general knowledge -------------------------------------------------------------------
# Legacy Survey viewer stretch: pixel = asinh(Q*I)/sqrt(Q) with Q = 20 and sky offset I = 0.03 -> sky level
BG = float(np.arcsinh(20 * 0.03) / np.sqrt(20.0))      # about 0.127
T_LOW = BG + 0.03      # faint isophote, a few grey levels above the sky
T_HIGH = BG + 0.20     # bright isophote (cores, bulges, nuclei)
NPH = 16               # phases used to read the amplitude of an azimuthal Fourier mode
CN = 20.0              # weight of the flux entry in the divisive normalisation

# ---- layer 1: blur + sky-subtracted maps ----------------------------------------------------------------
ax = np.arange(5) - 2.0
g1 = np.exp(-ax ** 2 / 2.0); g1 /= g1.sum()
g = np.outer(g1, g1)
w = np.zeros((4, 3, 5, 5)); b = np.zeros(4)
w[0, :, :, :] = g / 3.0; b[0] = -T_LOW      # luminance above faint isophote
w[1, :, :, :] = g / 3.0; b[1] = -T_HIGH     # luminance above bright isophote
w[2, 0] = g; b[2] = -T_LOW                  # red (z band) above faint isophote
w[3, 2] = g; b[3] = -T_LOW                  # blue (g band) above faint isophote
W["pre.W"], W["pre.b"] = w, b
layers.append({"name": "pre", "op": "conv", "inputs": ["image"], "out": 4, "kernel": [5, 5], "stride": 1, "padding": 2})
layers.append({"name": "pre_r", "op": "relu", "inputs": ["pre"]})

# ---- layer 2: whole-image moments about the centre -------------------------------------------------------
yy, xx = np.mgrid[0:128, 0:128]
x = xx - 63.5; y = yy - 63.5
r = np.sqrt(x * x + y * y); phi = np.arctan2(y, x)
win = np.exp(-r ** 2 / (2 * 20.0 ** 2))     # wide central window
win8 = np.exp(-r ** 2 / (2 * 8.0 ** 2))
win6 = np.exp(-r ** 2 / (2 * 6.0 ** 2))
psi = 2 * np.pi * np.arange(NPH) / NPH

# phase groups: [m=2 low, m=4 low, m=1 low, m=2 high, m=1 high], NPH phases each
wp = np.zeros((5 * NPH, 4, 128, 128))
for gi, (m, ch) in enumerate([(2, 0), (4, 0), (1, 0), (2, 1), (1, 1)]):
    for k in range(NPH):
        wp[gi * NPH + k, ch] = win * np.cos(m * phi - psi[k])
W["featp.W"], W["featp.b"] = wp, np.zeros(5 * NPH)
layers.append({"name": "featp", "op": "conv", "inputs": ["pre_r"], "out": 5 * NPH, "kernel": [128, 128], "stride": 1, "padding": 0})
layers.append({"name": "featp_g", "op": "reshape", "inputs": ["featp"], "shape": [1, 5, NPH]})
layers.append({"name": "amp_p", "op": "maxpool", "inputs": ["featp_g"], "size": [1, NPH], "stride": [1, NPH]})
layers.append({"name": "amp", "op": "reshape", "inputs": ["amp_p"], "shape": [5]})   # A2, A4, A1, B2, B1

# scalar sums: F0, Fc8, H0, Fc6, Fr6, Fb6
ws = np.zeros((6, 4, 128, 128))
ws[0, 0] = win; ws[1, 0] = win8; ws[2, 1] = win; ws[3, 0] = win6; ws[4, 2] = win6; ws[5, 3] = win6
W["feats.W"], W["feats.b"] = ws, np.zeros(6)
layers.append({"name": "feats", "op": "conv", "inputs": ["pre_r"], "out": 6, "kernel": [128, 128], "stride": 1, "padding": 0})

# ---- layer 3: divisive normalisation with the norm op -----------------------------------------------------
# Three groups, each a vector [CN*flux, f1, f2, ...] followed by its negative: norm then returns 3*f/sqrt(sum sq),
# which is (3/CN) * f/flux to within a few percent.
G = 18
def idx(grp, pos): return grp * G + pos
va = np.zeros((3 * G, 5)); vs = np.zeros((3 * G, 6))
def put(mat, grp, pos, src, val):
    mat[idx(grp, pos), src] = val; mat[idx(grp, pos + 9), src] = -val
put(vs, 0, 0, 0, CN)   # CN*F0
put(va, 0, 1, 0, 1.0)  # A2
put(va, 0, 2, 1, 1.0)  # A4
put(va, 0, 3, 2, 1.0)  # A1
put(vs, 0, 4, 1, 1.0)  # Fc8
put(vs, 0, 5, 2, 1.0)  # H0
put(vs, 1, 0, 2, CN)   # CN*H0
put(va, 1, 1, 3, 1.0)  # B2
put(va, 1, 2, 4, 1.0)  # B1
put(vs, 2, 0, 3, CN)   # CN*Fc6
put(vs, 2, 1, 4, 1.0)  # Fr6
put(vs, 2, 2, 5, 1.0)  # Fb6
W["va.W"], W["va.b"] = va, np.zeros(3 * G)
W["vs.W"], W["vs.b"] = vs, np.zeros(3 * G)
layers.append({"name": "va", "op": "dense", "inputs": ["amp"], "out": 3 * G})
layers.append({"name": "vs", "op": "dense", "inputs": ["feats"], "out": 3 * G})
layers.append({"name": "vsum", "op": "add", "inputs": ["va", "vs"]})
layers.append({"name": "vgrid", "op": "reshape", "inputs": ["vsum"], "shape": [3, 1, G]})
layers.append({"name": "vnorm", "op": "norm", "inputs": ["vgrid"]})
layers.append({"name": "u", "op": "reshape", "inputs": ["vnorm"], "shape": [3 * G]})

# ---- layer 4: hand-written soft rules ---------------------------------------------------------------------
S = CN / 3.0   # ratio = S * u
E, E4, A1, K8, HF = idx(0, 1), idx(0, 2), idx(0, 3), idx(0, 4), idx(0, 5)
GH, HE, HA = idx(1, 0), idx(1, 1), idx(1, 2)
FR, FB = idx(2, 1), idx(2, 2)
# each ramp: (coefficients on u, constant, a, b) -> clip((z - a)/(b - a), 0, 1)
ramps = [
    ({E: S}, 0.0, 0.06, 0.14),                 # 0 not round      (ellipticity (1-q)/(1+q): q below about 0.85)
    ({E: S}, 0.0, 0.25, 0.38),                 # 1 elongated      (q below about 0.5)
    ({E: S}, 0.0, 0.45, 0.58),                 # 2 thin           (q below about 0.3)
    ({K8: S}, 0.0, 0.50, 0.80),                # 3 small in the frame
    ({FR: S, FB: -S}, 0.0, 0.05, 0.45),        # 4 red / yellow colour
    ({HF: S}, 0.0, 0.05, 0.30),                # 5 has a bright core
    ({A1: S}, 0.0, 0.08, 0.25),                # 6 lopsided at the faint isophote
    ({HA: S}, 0.0, 0.15, 0.40),                # 7 bright light off centre (second nucleus)
    ({HE: S, E: -S}, 0.0, 0.05, 0.20),         # 8 bright part more elongated than the whole (bar)
    ({E: S, HE: -S, GH: 2.0 / 3.0}, -2.0, 0.10, 0.30),  # 9 bright part rounder than the whole (bulge), needs a core
    ({E4: S}, 0.0, 0.20, 0.40),                # 10 strong m=4: needle-like
]
nr = len(ramps)
hw = np.zeros((2 * nr, 3 * G)); hb = np.zeros(2 * nr)
for j, (co, const, a, bb) in enumerate(ramps):
    s = 1.0 / (bb - a)
    for k, v in co.items():
        hw[2 * j, k] = s * v; hw[2 * j + 1, k] = s * v
    hb[2 * j] = s * (const - a); hb[2 * j + 1] = s * (const - bb)
W["hid.W"], W["hid.b"] = hw, hb
layers.append({"name": "hid", "op": "dense", "inputs": ["u"], "out": 2 * nr})
layers.append({"name": "hid_r", "op": "relu", "inputs": ["hid"]})

# evidence table: rows = ramps, columns = classes
#        dist  merg  round inbtw cigar bar   tight loose e-nob e-bul
bias = [-0.8, -0.3,  0.0, -2.5, -5.0,  1.0,  1.0,  1.2, -4.0, -4.0]
table = [
    [0.3,  0.3, -4.5,  3.0,  2.0,  0.0,  0.0,  0.0,  1.5,  2.0],   # not round
    [-0.3, -0.3, -1.5, -2.0, 3.0, -1.5, -1.5, -1.5,  3.0,  3.5],   # elongated
    [-1.5, -1.5, 0.0, -2.5, -1.5, -2.0, -2.0, -2.0,  2.0,  0.0],   # thin
    [-0.5, 0.0,  0.0,  1.5,  1.5, -1.0, -1.0, -0.5, -1.0, -0.3],   # small
    [-0.5, 0.3,  1.5,  0.0,  0.0,  0.3, -0.2, -1.2, -1.2,  1.0],   # red
    [-0.3, 0.5,  1.5, -0.3, -0.3,  0.3,  0.3, -0.8, -1.2,  0.5],   # bright core
    [1.5,  2.0, -1.0, -0.8, -0.5, -0.5, -0.5,  0.3, -0.5, -0.5],   # lopsided
    [0.5,  2.0, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3],   # second nucleus
    [0.0,  0.0, -0.5,  0.0,  0.0,  1.5, -0.3,  0.0,  0.0,  0.0],   # bar
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  1.0],   # bulge
    [0.0,  0.0,  0.0,  0.0, -0.5,  0.0,  0.0,  0.0,  1.0,  0.3],   # needle
]
ow = np.zeros((10, 2 * nr))
for j in range(nr):
    for c in range(10):
        ow[c, 2 * j] = table[j][c]; ow[c, 2 * j + 1] = -table[j][c]
W["out.W"], W["out.b"] = ow, np.array(bias, dtype=np.float64)
layers.append({"name": "out", "op": "dense", "inputs": ["hid_r"], "out": 10})

with open(os.path.join(out_dir, "network.json"), "w") as f:
    json.dump({"layers": layers}, f, indent=1)
np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
