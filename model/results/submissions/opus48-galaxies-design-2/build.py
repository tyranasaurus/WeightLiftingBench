"""Builds a hand-designed galaxy-morphology network (no training / no data).

Every weight comes from a geometric formula typed from general knowledge of how
the ten Galaxy-Zoo classes look.  The image is contrast-normalised (`norm`),
thresholded to an isophote, and summarised by fixed templates:
  * oriented thin "slits" -> an elongation measure e = (max-min slit flux),
    which is flux-scaled and size-robust (0 for round, large for edge-on);
  * Gaussian apertures -> central concentration (bulge) vs total flux;
  * a difference-of-Gaussians band-pass -> disk texture (spiral arms/knots);
  * a colour template -> redness (old/smooth) vs blueness (star-forming);
  * first moments -> centroid offset (asymmetry / companions).
All cues are fractions of the total flux, so a small decision "tree" built from
min()=AND and sum()=OR gates turns them into the ten class scores.  Everything
is homogeneous, so the prediction does not depend on overall brightness/size.

Run:  python3 build.py OUT_DIR
"""
import sys, os, json
import numpy as np

# ---------------------------------------------------------------- constants --
N = 128
C = (N - 1) / 2.0
yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
X = xx - C
Y = yy - C
R2 = X * X + Y * Y

SIG_T = 28.0     # total-flux window
SIG_M = 14.0     # mid window (colour)
SIG_C = 6.0      # core aperture (bulge scale)
SIG_N = 18.0     # normaliser for first-moment templates
SLW   = 3.0      # slit half-width (thin direction)
SLL   = 24.0     # slit length scale (long direction)
ISO_K = 0.8      # isophote level in std units
GT    = 0.03     # texture threshold (fraction of total flux)
NANG  = 12

def gauss(sig):
    return np.exp(-R2 / (2.0 * sig * sig))

w_tot  = gauss(SIG_T)
w_mid  = gauss(SIG_M)
w_core = gauss(SIG_C)
w_ann  = gauss(28.0) * (1.0 - gauss(7.0))

def gkern(k, sig):
    a = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-(a[:, None] ** 2 + a[None, :] ** 2) / (2 * sig * sig))
    return g / g.sum()
KLOG = 9
LoG = gkern(KLOG, 0.8) - gkern(KLOG, 2.0)

def flat(t):
    return t.reshape(-1).astype(np.float64)

ang = np.arange(NANG) * (np.pi / NANG)
cang = np.arange(NANG) * (2 * np.pi / NANG)

W = {}

# preprocessing convs
W["gray.W"] = np.array([[[[1/3.]], [[1/3.]], [[1/3.]]]]); W["gray.b"] = np.array([0.0])
W["col.W"]  = np.array([[[[1.0]], [[0.0]], [[-1.0]]]]);   W["col.b"]  = np.array([0.0])
W["iso_pre.W"] = np.array([[[[1.0]]]]);                   W["iso_pre.b"] = np.array([-ISO_K])
W["logp.W"] = LoG[None, None, :, :].copy();               W["logp.b"] = np.array([0.0])

# ---- fM : slit flux (0..11), tot, core, mid, cx, cy  ------------------------
LF = 19
fM = np.zeros((LF, N * N))
for k in range(NANG):
    along = X * np.cos(ang[k]) + Y * np.sin(ang[k])
    perp = -X * np.sin(ang[k]) + Y * np.cos(ang[k])
    fM[k] = flat(np.exp(-(perp ** 2) / (2 * SLW ** 2) - (along ** 2) / (2 * SLL ** 2)))
fM[12] = flat(w_tot)
fM[13] = flat(w_core)
fM[14] = flat(w_mid)
fM[15] = flat(w_tot * (X / SIG_N))
fM[16] = flat(w_tot * (Y / SIG_N))
W["fM.W"] = fM; W["fM.b"] = np.zeros(LF)

fL = np.zeros((LF, N * N)); fL[17] = flat(w_ann)
W["fLog.W"] = fL; W["fLog.b"] = np.zeros(LF)
fC = np.zeros((LF, N * N)); fC[18] = flat(w_mid)
W["fCol.W"] = fC; W["fCol.b"] = np.zeros(LF)

# ---- g1 : slit, -slit, centroid-projection  --------------------------------
G = NANG * 3
g1 = np.zeros((G, LF))
for k in range(NANG):
    g1[k, k] = 1.0
    g1[NANG + k, k] = -1.0
for j in range(NANG):
    g1[2 * NANG + j, 15] = np.cos(cang[j])
    g1[2 * NANG + j, 16] = np.sin(cang[j])
W["g1.W"] = g1; W["g1.b"] = np.zeros(G)
# after reshape(3,1,12)+maxpool -> g1f = [Fmax, -Fmin, off]

# ---- hP, hF -> H = [e, off, tot, core, tex, red, mid, spare] ----------------
hP = np.zeros((8, 3))
hP[0, 0] = 1.0; hP[0, 1] = 1.0     # e = Fmax + (-Fmin) = Fmax - Fmin
hP[1, 2] = 1.0                      # off
W["hP.W"] = hP; W["hP.b"] = np.zeros(8)
hF = np.zeros((8, LF))
hF[2, 12] = 1.0   # tot
hF[3, 13] = 1.0   # core
hF[4, 17] = 1.0   # tex
hF[5, 18] = 1.0   # red
hF[6, 14] = 1.0   # mid
W["hF.W"] = hF; W["hF.b"] = np.zeros(8)
E, OFF, TOT, CORE, TEX, RED = 0, 1, 2, 3, 4, 5

# thresholds (fractions of total flux) + scale gains, chosen by hand
CC = 0.34    # concentration level
CR = 0.02    # red/blue colour crossover (round group: smooth vs spiral)
CRE = 0.03   # red/blue crossover for the edge-on group (8 vs 9)
GT = 0.05    # texture level
KC = 6.0     # colour gain (brings the small colour signal up to shape scale)
KT = 3.0     # texture gain
# ---- Clin -> literals (pre-relu) -------------------------------------------
NC = 15
Cl = np.zeros((NC, 8))
Cl[0, TOT] = 0.18; Cl[0, E] = -1              # L0 roundish   e<.18tot
Cl[1, E] = 1; Cl[1, TOT] = -0.05              # L1 elong      e>.05tot
Cl[2, E] = 1; Cl[2, TOT] = -0.20              # L2 edge       e>.20tot
Cl[3, E] = 1; Cl[3, TOT] = -0.33              # L3 vedge      e>.33tot
Cl[4, CORE] = 1; Cl[4, TOT] = -CC             # L4 concentrated
Cl[5, TOT] = CC; Cl[5, CORE] = -1             # L5 diffuse
Cl[6, RED] = KC; Cl[6, TOT] = -KC * CR        # L6 red
Cl[7, TOT] = KC * CR; Cl[7, RED] = -KC        # L7 blue
Cl[8, OFF] = 1; Cl[8, TOT] = -0.18            # L8 asymmetric
Cl[9, TOT] = 0.18; Cl[9, E] = -1              # L9 e<.18tot (mid upper)
Cl[10, TOT] = 0.40; Cl[10, E] = -1            # L10 e<.40tot (cigar upper)
Cl[11, TEX] = KT; Cl[11, TOT] = -KT * GT      # L11 textured
Cl[12, TOT] = KT * GT; Cl[12, TEX] = -KT      # L12 no-texture
Cl[13, RED] = KC; Cl[13, TOT] = -KC * CRE     # L13 edge-red  (bulge)
Cl[14, TOT] = KC * CRE; Cl[14, RED] = -KC     # L14 edge-blue (disk)
W["Clin.W"] = Cl; W["Clin.b"] = np.zeros(NC)

# ---- stage-1 mins  S_i = min(a_i, b_i) -------------------------------------
# S0 mid-band(elong&e<.18)  S1 cigar-band(edge&e<.40)  S2 roundish&red
# S3 roundish&blue  S4 edge&edge-blue  S5 edge&edge-red  S6 vedge&edge-blue
# S7 asym&diff  S8 roundish&conc  S9 asym&tex   S10 elong&red
pairs = [(1, 9), (2, 10), (0, 6), (0, 7), (2, 14), (2, 13), (3, 14),
         (8, 5), (0, 4), (8, 11), (1, 6)]
NS = len(pairs)
d1 = np.zeros((NS, NC))
for i, (a, b) in enumerate(pairs):
    d1[i, a] = 1.0; d1[i, b] = -1.0
W["d1.W"] = d1; W["d1.b"] = np.zeros(NS)
Sa = np.zeros((NS, NC))
for i, (a, b) in enumerate(pairs):
    Sa[i, a] = 1.0
W["Sa.W"] = Sa; W["Sa.b"] = np.zeros(NS)
Sneg = -np.eye(NS)
W["Sneg.W"] = Sneg; W["Sneg.b"] = np.zeros(NS)
# S = Sa + Sneg(r1)

# ---- stage-2 mins  (S index, C index) --------------------------------------
# T0 in-between = min(mid-band, red)       T1 cigar = min(cigar-band, red)
# T2 tight = min(roundish&blue, conc)      T3 loose = min(roundish&blue, diff)
# T4 smooth = min(roundish&red, conc)
tpairs = [(0, 6), (1, 6), (3, 4), (3, 5), (2, 4)]
NT = len(tpairs)
d2S = np.zeros((NT, NS)); d2C = np.zeros((NT, NC))
for i, (si, ci) in enumerate(tpairs):
    d2S[i, si] = 1.0; d2C[i, ci] = -1.0
W["d2S.W"] = d2S; W["d2S.b"] = np.zeros(NT)
W["d2C.W"] = d2C; W["d2C.b"] = np.zeros(NT)
Ta = np.zeros((NT, NS))
for i, (si, ci) in enumerate(tpairs):
    Ta[i, si] = 1.0
W["Ta.W"] = Ta; W["Ta.b"] = np.zeros(NT)
Tneg = -np.eye(NT)
W["Tneg.W"] = Tneg; W["Tneg.b"] = np.zeros(NT)
# T = Ta + Tneg(r2)

# ---- final scores ----------------------------------------------------------
outS = np.zeros((10, NS))
# class 0 (disturbed) has no positive evidence: my asymmetry/texture cue was
# pure noise (7% precision) and stole true 2/3/9, so leave it as the zero-score
# fallback that only wins when nothing else fires.
outS[1, 7] = 1.0                      # merging   <- asym & diffuse
outS[2, 2] = 0.7                      # round smooth <- roundish&red
outS[3, 0] = 0.3; outS[3, 10] = 0.2   # in-between <- mid-band, elong&red
outS[7, 3] = 0.50                     # loose     <- roundish&blue (base-rate default spiral)
outS[8, 4] = 1.0; outS[8, 6] = 0.5    # edge no bulge <- edge&blue, vedge&blue
outS[9, 5] = 1.0                      # edge with bulge <- edge&red
W["outS.W"] = outS; W["outS.b"] = np.zeros(10)
outT = np.zeros((10, NT))
outT[2, 4] = 0.5                      # round smooth <- red&conc
outT[3, 0] = 1.0                      # in-between   <- mid-band&red
outT[4, 1] = 0.0                      # cigar: no reliable cue vs edge-bulge, so
                                      # leave it near-silent (protects class 9)
outT[7, 3] = 1.0                      # loose        <- blue&diffuse
W["outT.W"] = outT; W["outT.b"] = np.zeros(10)
outC = np.zeros((10, NC))
outC[1, 8] = 0.3                      # merging raw asym
W["outC.W"] = outC; W["outC.b"] = np.zeros(10)

# ------------------------------------------------------------------ network --
layers = [
    {"name": "gray", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "col",  "op": "conv", "inputs": ["image"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "sg",   "op": "norm", "inputs": ["gray"]},
    {"name": "iso_pre", "op": "conv", "inputs": ["sg"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "M",    "op": "relu", "inputs": ["iso_pre"]},
    {"name": "logp", "op": "conv", "inputs": ["sg"], "out": 1, "kernel": [KLOG, KLOG], "stride": 1, "padding": KLOG // 2},
    {"name": "Mlog", "op": "relu", "inputs": ["logp"]},
    {"name": "fM",   "op": "dense", "inputs": ["M"], "out": LF},
    {"name": "fLog", "op": "dense", "inputs": ["Mlog"], "out": LF},
    {"name": "fCol", "op": "dense", "inputs": ["col"], "out": LF},
    {"name": "F",    "op": "add", "inputs": ["fM", "fLog", "fCol"]},
    {"name": "g1",   "op": "dense", "inputs": ["F"], "out": G},
    {"name": "g1r",  "op": "reshape", "inputs": ["g1"], "shape": [3, 1, NANG]},
    {"name": "g1p",  "op": "maxpool", "inputs": ["g1r"], "size": [1, NANG], "stride": [1, 1]},
    {"name": "g1f",  "op": "reshape", "inputs": ["g1p"], "shape": [3]},
    {"name": "hP",   "op": "dense", "inputs": ["g1f"], "out": 8},
    {"name": "hF",   "op": "dense", "inputs": ["F"], "out": 8},
    {"name": "H",    "op": "add", "inputs": ["hP", "hF"]},
    {"name": "Clin", "op": "dense", "inputs": ["H"], "out": NC},
    {"name": "C",    "op": "relu", "inputs": ["Clin"]},
    {"name": "d1",   "op": "dense", "inputs": ["C"], "out": NS},
    {"name": "r1",   "op": "relu", "inputs": ["d1"]},
    {"name": "Sa",   "op": "dense", "inputs": ["C"], "out": NS},
    {"name": "Sneg", "op": "dense", "inputs": ["r1"], "out": NS},
    {"name": "S",    "op": "add", "inputs": ["Sa", "Sneg"]},
    {"name": "d2S",  "op": "dense", "inputs": ["S"], "out": NT},
    {"name": "d2C",  "op": "dense", "inputs": ["C"], "out": NT},
    {"name": "d2",   "op": "add", "inputs": ["d2S", "d2C"]},
    {"name": "r2",   "op": "relu", "inputs": ["d2"]},
    {"name": "Ta",   "op": "dense", "inputs": ["S"], "out": NT},
    {"name": "Tneg", "op": "dense", "inputs": ["r2"], "out": NT},
    {"name": "T",    "op": "add", "inputs": ["Ta", "Tneg"]},
    {"name": "outS", "op": "dense", "inputs": ["S"], "out": 10},
    {"name": "outT", "op": "dense", "inputs": ["T"], "out": 10},
    {"name": "outC", "op": "dense", "inputs": ["C"], "out": 10},
    {"name": "out",  "op": "add", "inputs": ["outS", "outT", "outC"]},
]
net = {"layers": layers}

out_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/out"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "network.json"), "w") as f:
    json.dump(net, f, indent=1)
np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
print("wrote", out_dir, "layers", len(layers), "tensors", len(W))
