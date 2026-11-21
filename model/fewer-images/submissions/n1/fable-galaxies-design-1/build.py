"""Hand-designed galaxy classifier: writes network.json and weights.npz. Reads no files.
Every number below is typed by hand from general knowledge of galaxy morphology and of the image format.
Usage: python3 build.py OUT_FOLDER"""
import sys, os, json
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------- geometry (64 x 64 grid after 2x2 averaging)
G = 64
ii = np.arange(G)
X = np.repeat((2.0 * ii - 63.0)[None, :], G, 0)      # offset from the image centre, in original pixels
Y = np.repeat((2.0 * ii - 63.0)[:, None], G, 1)
R2 = X * X + Y * Y
frame = np.ones((G, G)); frame[3:-3, 3:-3] = 0.0; frame /= frame.sum()   # border frame = sky estimate


def dcfree(w):
    """Make a linear functional blind to a constant sky level (sky taken as the mean of the border frame)."""
    return w - w.sum() * frame


def g(u):
    """What the norm layer returns for the group [a, -a, s, -s] at position 0, with u = a / s."""
    return np.sqrt(2.0) * u / np.sqrt(1.0 + u * u)


WINS = [4.0, 7.0, 12.0, 20.0]      # Gaussian window sigmas (original pixels); the best-matched one is selected
NW, NA = len(WINS), 16             # windows, projection angles
SC = 10.0                          # overall scale before norm (keeps the 1e-5 epsilon negligible)

W = {}
# ---------------------------------------------------------------- layer 1: luminance and red-minus-blue, 64 x 64
pre = np.zeros((2, 3, 2, 2))
pre[0, :, :, :] = 1.0 / 12.0       # mean of R, G, B over each 2 x 2 block
pre[1, 0, :, :] = 0.25             # R
pre[1, 2, :, :] = -0.25            # minus B
W["pre.W"], W["pre.b"] = pre, np.zeros(2)

# ---------------------------------------------------------------- linear measurements (window moments)
# lin: per window [M0, Tn, C, M0s], then central surface brightness
NLIN = NW * 4 + 1
lin = np.zeros((NLIN, 2, G, G))
proj = np.zeros((NW * 2 * NA, 2, G, G))
for j, w in enumerate(WINS):
    Wg = np.exp(-R2 / (2 * w * w))
    ws = w / 2.5
    Ws = np.exp(-R2 / (2 * ws * ws))
    lin[j * 4 + 0, 0] = dcfree(Wg)                    # M0: windowed flux
    lin[j * 4 + 1, 0] = dcfree(Wg * R2 / (w * w))     # Tn: windowed trace of second moments / w^2
    lin[j * 4 + 2, 1] = dcfree(Wg)                    # C: windowed red-minus-blue flux
    lin[j * 4 + 3, 0] = dcfree(Ws)                    # M0s: flux in a 2.5x smaller window
    for k in range(NA):
        ph = 2 * np.pi * k / NA
        # quadrupole (ellipticity) projected on angle ph; the max over angles is its magnitude
        proj[(2 * j) * NA + k, 0] = dcfree(Wg * (np.cos(ph) * (X * X - Y * Y) + np.sin(ph) * 2 * X * Y) / (w * w))
        # dipole (centroid offset) projected on angle ph
        proj[(2 * j + 1) * NA + k, 0] = dcfree(Wg * (np.cos(ph) * X + np.sin(ph) * Y) / w)
Wc = np.exp(-R2 / (2 * 2.5 * 2.5)); Wc /= Wc.sum()
lin[NW * 4, 0] = dcfree(Wc)                           # central surface brightness above sky
W["lin.W"], W["lin.b"] = lin.reshape(NLIN, -1), np.zeros(NLIN)
W["proj.W"], W["proj.b"] = proj.reshape(NW * 2 * NA, -1), np.zeros(NW * 2 * NA)

# ---------------------------------------------------------------- ratios via norm: groups [a, -a, s, -s]
NR = 5                                               # ratios per window: size match, ellipticity, offset, colour, concentration
RHO0, OFF_S, COL_S, CONC0 = 0.45, 0.15, 0.30, 0.46
ra = np.zeros((NW * NR * 4, NLIN))
rb = np.zeros((NW * NR * 4, NW * 2))


def put(M, ch, col, a=0.0, s=0.0):
    b = ch * 4
    M[b + 0, col] += SC * a; M[b + 1, col] -= SC * a
    M[b + 2, col] += SC * s; M[b + 3, col] -= SC * s


for j in range(NW):
    m0, tn, cc, m0s = j * 4, j * 4 + 1, j * 4 + 2, j * 4 + 3
    c0 = j * NR
    put(ra, c0 + 0, tn, a=1.0); put(ra, c0 + 0, m0, a=-RHO0, s=1.0)          # (Tn/M0 - 0.45)
    put(ra, c0 + 1, tn, a=-0.5, s=1.0); put(rb, c0 + 1, 2 * j, a=1.0)        # (|E|/Tn - 0.5)
    put(rb, c0 + 2, 2 * j + 1, a=1.0 / OFF_S); put(ra, c0 + 2, m0, s=1.0)    # centroid offset / (0.15 w)
    put(ra, c0 + 3, cc, a=1.0 / COL_S); put(ra, c0 + 3, m0, s=1.0)           # colour (R-B)/L / 0.3
    put(ra, c0 + 4, m0s, a=1.0 / CONC0); put(ra, c0 + 4, m0, a=-1.0, s=1.0)  # concentration / 0.46 - 1
W["ra.W"], W["ra.b"] = ra, np.zeros(len(ra))
W["rb.W"], W["rb.b"] = rb, np.zeros(len(rb))

# ---------------------------------------------------------------- hand rule table: typical range of each feature per class
# classes: 0 disturbed, 1 merging, 2 round, 3 in-between, 4 cigar, 5 barred, 6 tight spiral, 7 loose spiral,
#          8 edge-on no bulge, 9 edge-on with bulge
E_RNG = [(0.10, 0.70), (0.15, 0.75), (0.00, 0.13), (0.20, 0.55), (0.55, 0.75), (0.08, 0.50), (0.05, 0.55), (0.08, 0.60), (0.78, 1.00), (0.62, 0.88)]
OFF_RNG = [(0.06, 0.30), (0.15, 0.60), (0.0, 0.06), (0.0, 0.06), (0.0, 0.06), (0.0, 0.08), (0.0, 0.08), (0.0, 0.12), (0.0, 0.10), (0.0, 0.10)]
COL_RNG = [(-0.10, 0.35), (-0.20, 0.30), (0.22, 0.60), (0.22, 0.60), (0.20, 0.60), (0.05, 0.35), (0.00, 0.30), (-0.30, 0.15), (-0.10, 0.30), (0.22, 0.65)]
CONC_RNG = [(0.20, 0.85), (0.20, 0.85), (0.46, 0.85), (0.46, 0.85), (0.42, 0.85), (0.40, 0.75), (0.40, 0.75), (0.20, 0.48), (0.20, 0.48), (0.46, 0.85)]
CSB_RNG = [(0.15, 0.90), (0.15, 0.90), (0.55, 0.90), (0.55, 0.90), (0.45, 0.90), (0.45, 0.90), (0.45, 0.90), (0.20, 0.75), (0.10, 0.55), (0.55, 0.90)]
FEATS = [  # (ranges, transform to the norm-output domain, weight per unit in that domain)
    (E_RNG, lambda v: g(v - 0.5), 16.0),
    (OFF_RNG, lambda v: g(v / OFF_S), 3.0),
    (COL_RNG, lambda v: g(v / COL_S), 2.5),
    (CONC_RNG, lambda v: g(v / CONC0 - 1.0), 1.5),
    (CSB_RNG, lambda v: v, 5.0),
]
CLASS_BIAS = [-1.5, -1.0, 0.0, 0.0, -1.0, -0.3, -0.2, 0.0, 0.0, 0.0]
SIZE_BIAS = [
    [-0.3, -0.3, 0.5, 0.4, 0.0, -0.6, -0.6, -0.3, -0.3, -0.5],   # smallest window: small galaxy
    [0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.1, -0.1, 0.0, 0.0, 0.3, 0.3, 0.3, 0.2, 0.2],
    [0.2, 0.3, -0.4, -0.2, 0.0, 0.3, 0.4, 0.3, 0.3, 0.3],        # largest window: large galaxy
]
P_MATCH = 60.0                    # penalty on a window whose size does not match the galaxy

NF = len(FEATS)
PER = 2 + 10 * NF * 2
NH = NW * PER
hn_W = np.zeros((NH, NW * NR * 4)); hn_b = np.zeros(NH)
hc_W = np.zeros((NH, NLIN))
sc_W = np.zeros((NW * 10, NH)); sc_b = np.zeros(NW * 10)
for j in range(NW):
    h0 = j * PER
    src = lambda r: (j * NR + r) * 4          # position 0 of ratio channel r of window j
    hn_W[h0 + 0, src(0)] = 1.0; hn_W[h0 + 1, src(0)] = -1.0          # |size mismatch|
    for c in range(10):
        sc_b[j * 10 + c] = CLASS_BIAS[c] + SIZE_BIAS[j][c]
        sc_W[j * 10 + c, h0 + 0] = -P_MATCH; sc_W[j * 10 + c, h0 + 1] = -P_MATCH
        for f, (rng, tr, wt) in enumerate(FEATS):
            lo, hi = rng[c]
            u = h0 + 2 + (c * NF + f) * 2
            if f < 4:
                hn_W[u, src(f + 1)] = -1.0; hn_b[u] = tr(lo)          # relu(lo - x)
                hn_W[u + 1, src(f + 1)] = 1.0; hn_b[u + 1] = -tr(hi)  # relu(x - hi)
            else:
                hc_W[u, NW * 4] = -1.0; hn_b[u] = lo
                hc_W[u + 1, NW * 4] = 1.0; hn_b[u + 1] = -hi
            sc_W[j * 10 + c, u] = -wt; sc_W[j * 10 + c, u + 1] = -wt
W["hn.W"], W["hn.b"] = hn_W, hn_b
W["hc.W"], W["hc.b"] = hc_W, np.zeros(NH)
W["sc.W"], W["sc.b"] = sc_W, sc_b

layers = [
    {"name": "pre", "op": "conv", "inputs": ["image"], "out": 2, "kernel": [2, 2], "stride": 2, "padding": 0},
    {"name": "lin", "op": "dense", "inputs": ["pre"], "out": NLIN},
    {"name": "proj", "op": "dense", "inputs": ["pre"], "out": NW * 2 * NA},
    {"name": "projr", "op": "reshape", "inputs": ["proj"], "shape": [1, NW * 2, NA]},
    {"name": "projm", "op": "maxpool", "inputs": ["projr"], "size": [1, NA], "stride": [1, NA]},
    {"name": "ra", "op": "dense", "inputs": ["lin"], "out": NW * NR * 4},
    {"name": "rb", "op": "dense", "inputs": ["projm"], "out": NW * NR * 4},
    {"name": "rsum", "op": "add", "inputs": ["ra", "rb"]},
    {"name": "rres", "op": "reshape", "inputs": ["rsum"], "shape": [NW * NR, 2, 2]},
    {"name": "rn", "op": "norm", "inputs": ["rres"]},
    {"name": "hn", "op": "dense", "inputs": ["rn"], "out": NH},
    {"name": "hc", "op": "dense", "inputs": ["lin"], "out": NH},
    {"name": "hsum", "op": "add", "inputs": ["hn", "hc"]},
    {"name": "hrelu", "op": "relu", "inputs": ["hsum"]},
    {"name": "sc", "op": "dense", "inputs": ["hrelu"], "out": NW * 10},
    {"name": "scr", "op": "reshape", "inputs": ["sc"], "shape": [1, NW, 10]},
    {"name": "scm", "op": "maxpool", "inputs": ["scr"], "size": [NW, 1], "stride": [NW, 1]},
    {"name": "out", "op": "reshape", "inputs": ["scm"], "shape": [10]},
]
with open(os.path.join(out_dir, "network.json"), "w") as fh:
    json.dump({"layers": layers}, fh, indent=1)
np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
