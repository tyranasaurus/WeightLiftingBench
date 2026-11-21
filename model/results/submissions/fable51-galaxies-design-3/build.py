#!/usr/bin/env python3
"""
Hand-designed galaxy morphology network: writes network.json and weights.npz.

Every number below comes from formulas and from general knowledge of galaxy morphology and of the
DESI Legacy Survey image rendering (sky level ~0.14 in the 0..1 images, 0.52"/px, galaxy centred).
Nothing is fitted to the training images.

Pipeline (ops: conv / dense / relu / maxpool / avgpool / add / reshape):
  1. luminance above sky, Gaussian-smoothed (cascade of 3x3 binomial blurs); red and blue above sky
  2. soft isophote masks at 4 surface-brightness levels (each a pair of ReLUs)
  3. measurements as fixed dense layers:
       - windowed area of each mask
       - elongation of each mask: amplitude of its cos(2*phi) angular moment, taken as the max over 16
         directions (reshape + maxpool); for an ellipse of axis ratio q this equals area*(1-q)/(1+q)
       - red and blue flux inside the low mask (ReLU gating), core excluded
       - small-scale structure: positive difference-of-Gaussians inside the mask, at two scales
         (knots at half resolution, arms at quarter resolution)
       - central aperture fluxes and the peak brightness
       - 180-degree rotational asymmetry of the mid mask
       - secondary nucleus detector (compact bright peak with extended light around it)
  4. soft comparisons  A - t*B > 0  (ratio thresholds written as linear inequalities), turned into
     0..1 indicators by pairs of ReLUs
  5. soft logic (AND / AND-NOT of indicators) and a hand-written rule table for the 10 class scores
"""
import sys, json, numpy as np
from pathlib import Path

H = W = 128
CY = CX = 63.5
BG_R, BG_G, BG_B = 0.138, 0.140, 0.145      # sky level per channel in the rendering
BG_L = (BG_R + BG_G + BG_B) / 3.0

# isophote levels above sky (in the 0..1 stretched units of the images)
T_LO, T_MID, T_HI, T_VHI = 0.06, 0.17, 0.38, 0.55
K_STEP = 50.0                                 # mask edge softness: 1 / 0.02 brightness units

layers = []
weights = {}


def add_layer(name, op, inputs, **kw):
    L = {"name": name, "op": op, "inputs": inputs}
    L.update(kw)
    layers.append(L)
    return name


def conv(name, inp, Wt, b, stride=1, padding=0):
    Wt = np.asarray(Wt, np.float64)
    b = np.asarray(b, np.float64).reshape(-1)
    weights[name + ".W"] = Wt
    weights[name + ".b"] = b
    return add_layer(name, "conv", [inp], out=int(Wt.shape[0]), kernel=[int(Wt.shape[2]), int(Wt.shape[3])],
                     stride=stride, padding=padding)


def dense(name, inp, Wt, b):
    Wt = np.asarray(Wt, np.float64)
    b = np.asarray(b, np.float64).reshape(-1)
    weights[name + ".W"] = Wt
    weights[name + ".b"] = b
    return add_layer(name, "dense", [inp], out=int(Wt.shape[0]))


def relu(name, inp):
    return add_layer(name, "relu", [inp])


def gauss2d(size, sigma):
    r = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-0.5 * (r / sigma) ** 2)
    k = np.outer(g, g)
    return k / k.sum()


def step_map(name, inp, thr, width, extra_bias=0.0):
    """Per-pixel soft step((x - thr)/width) in 0..1 for a (1,h,w) map, plus extra_bias."""
    Wq = np.array([[[[1.0 / width]]], [[[1.0 / width]]]])
    bq = np.array([-thr / width, -thr / width - 1.0])
    conv(name + "_s", inp, Wq, bq)
    relu(name + "_r", name + "_s")
    return conv(name, name + "_r", np.array([[[[1.0]], [[-1.0]]]]), [extra_bias])


def window(r, r_in, r_out):
    """1 inside r_in, linear ramp to 0 at r_out."""
    return np.clip((r_out - r) / (r_out - r_in), 0.0, 1.0)


# coordinate grids at full, half and quarter resolution
def grid(n):
    y, x = np.mgrid[:n, :n].astype(np.float64)
    c = (n - 1) / 2.0
    return np.hypot(y - c, x - c), np.arctan2(y - c, x - c)


RR, PHI = grid(H)
RR2, _ = grid(H // 2)
RR4, _ = grid(H // 4)

# ----------------------------------------------------------------------------------------------
# 1. luminance and colour channels
# ----------------------------------------------------------------------------------------------
conv("lum", "image", np.array([[[[1 / 3]], [[1 / 3]], [[1 / 3]]]]), [-BG_L])             # L - sky
conv("rb", "image", np.array([[[[1.0]], [[0.0]], [[0.0]]], [[[0.0]], [[0.0]], [[1.0]]]]), [-BG_R, -BG_B])
binom = np.array([1, 2, 1], np.float64)
k3 = np.outer(binom, binom) / 16.0
prev = "lum"
for i in range(3):                                                                       # sigma^2 = 1.5
    prev = conv(f"blur{i + 1}", prev, k3[None, None], [0.0], padding=1)
LS = prev                                                                                # smoothed L - sky

# ----------------------------------------------------------------------------------------------
# 2. soft isophote masks  M_t = relu(k (L - t)) - relu(k (L - t) - 1)
# ----------------------------------------------------------------------------------------------
levels = [T_LO, T_MID, T_HI, T_VHI]
Wm = np.zeros((8, 1, 1, 1)); bm = np.zeros(8)
for i, t in enumerate(levels):
    Wm[2 * i, 0, 0, 0] = K_STEP;     bm[2 * i] = -K_STEP * t
    Wm[2 * i + 1, 0, 0, 0] = K_STEP; bm[2 * i + 1] = -K_STEP * t - 1.0
# faint levels from the well-smoothed luminance (noise), bright levels from the lightly smoothed one
# (sigma^2 = 0.5) so that small bright bars keep their elongation
conv("mstep", LS, Wm, bm)
relu("mstep_r", "mstep")
Wc = np.zeros((4, 8, 1, 1))
for i in range(2):
    Wc[i, 2 * i, 0, 0] = 1.0
    Wc[i, 2 * i + 1, 0, 0] = -1.0
conv("masks_lo", "mstep_r", Wc, np.zeros(4))  # (4,128,128): M_lo, M_mid, 0, 0
conv("mstep_hi", "blur1", Wm, bm)
relu("mstep_hi_r", "mstep_hi")
Wc2 = np.zeros((4, 8, 1, 1))
for i in range(2, 4):
    Wc2[i, 2 * i, 0, 0] = 1.0
    Wc2[i, 2 * i + 1, 0, 0] = -1.0
conv("masks_hi", "mstep_hi_r", Wc2, np.zeros(4))  # (4,128,128): 0, 0, M_hi, M_vhi
add_layer("masks", "add", ["masks_lo", "masks_hi"])  # (4,128,128): M_lo, M_mid, M_hi, M_vhi in [0,1]

# ----------------------------------------------------------------------------------------------
# 3. measurement vector V.  Every galaxy-dependent sum is taken inside three concentric windows
#    (small / medium / large); the rule logic later picks the window that matches the galaxy's size,
#    so field stars far from a small galaxy do not pollute its measurements.
# ----------------------------------------------------------------------------------------------
WINS = [(12.0, 16.0), (22.0, 28.0), (36.0, 48.0)]
NW = len(WINS)
WIN = [window(RR, a, b) for a, b in WINS]
WIN2 = [window(RR2, a / 2, b / 2) for a, b in WINS]
WIN4 = [window(RR4, a / 4, b / 4) for a, b in WINS]
INNER = [(6.0, 9.0), (10.0, 13.0), (16.0, 20.0)]            # inner (bar / bulge) region for each size class
WIN_IN = [window(RR, a, b) for a, b in INNER]

slot = {}
n = 0
NLEV = 5                                         # lo, mid, hi, vhi isophote; mid isophote inside an inner window
for w in range(NW):
    for lev in range(NLEV):
        slot[f"E{lev}w{w}"] = n; n += 1          # elongation amplitude (= area*(1-q)/(1+q) for an ellipse)
    for lev in range(NLEV):
        slot[f"A{lev}w{w}"] = n; n += 1          # windowed mask area (pixels)
    slot[f"CRw{w}"] = n; n += 1                  # red flux inside the low mask (core excluded)
    slot[f"CBw{w}"] = n; n += 1                  # blue flux inside the low mask
    slot[f"SKw{w}"] = n; n += 1                  # knot structure (half resolution)
    slot[f"SAw{w}"] = n; n += 1                  # arm structure (quarter resolution)
    slot[f"ASw{w}"] = n; n += 1                  # 180-degree asymmetry of the mid mask (pixels)
slot["F1"] = n; n += 1                           # aperture flux r < 2.5
slot["F2"] = n; n += 1                           # aperture flux r < 6
slot["PK"] = n; n += 1                           # peak smoothed luminance near the centre
slot["NUC"] = n; n += 1                          # secondary nucleus indicator (0..1)
NV = n

# 3a. shape: angular moments of each mask in each window, max over 16 directions
NANG = 16
Wang = np.zeros((NW * NLEV * NANG, 4, H, W))
for w in range(NW):
    for m in range(NLEV):
        chan = 1 if m == 4 else m
        win = WIN_IN[w] if m == 4 else WIN[w]
        for k in range(NANG):
            th = np.pi * k / NANG
            Wang[(w * NLEV + m) * NANG + k, chan] = win * np.cos(2 * PHI - 2 * th)
dense("ang", "masks", Wang.reshape(NW * NLEV * NANG, -1), np.zeros(NW * NLEV * NANG))
add_layer("ang_r", "reshape", ["ang"], shape=[1, NW * NLEV, NANG])
add_layer("ang_max", "maxpool", ["ang_r"], size=[1, NANG], stride=[1, NANG])   # (1,15,1)
WE = np.zeros((NV, NW * NLEV))
for w in range(NW):
    for m in range(NLEV):
        WE[slot[f"E{m}w{w}"], w * NLEV + m] = 1.0
dense("v_shape", "ang_max", WE, np.zeros(NV))

# areas
Warea = np.zeros((NV, 4, H, W))
for w in range(NW):
    for m in range(NLEV):
        Warea[slot[f"A{m}w{w}"], 1 if m == 4 else m] = WIN_IN[w] if m == 4 else WIN[w]
dense("v_area", "masks", Warea.reshape(NV, -1), np.zeros(NV))

# 3b. colour inside the low mask: gated R and B = relu(x + M_lo - 1), core (r<3) excluded
Wg = np.zeros((2, 4, 1, 1)); Wg[0, 0, 0, 0] = 1.0; Wg[1, 0, 0, 0] = 1.0
conv("mlo2", "masks", Wg, [-1.0, -1.0])
add_layer("rb_gate", "add", ["rb", "mlo2"])
relu("rb_g", "rb_gate")
Wcol = np.zeros((NV, 2, H, W))
for w in range(NW):
    Wcol[slot[f"CRw{w}"], 0] = WIN[w] * (RR > 3.0)
    Wcol[slot[f"CBw{w}"], 1] = WIN[w] * (RR > 3.0)
dense("v_col", "rb_g", Wcol.reshape(NV, -1), np.zeros(NV))

# 3c. structure: positive DoG inside the mask
#   knots: half resolution, sigma 0.6 vs 1.5 (1.2 vs 3 px), inside the mid mask, r > 5 px
add_layer("lum2", "avgpool", ["lum"], size=[2, 2], stride=[2, 2])                      # (1,64,64) L - sky
Wsel = np.zeros((1, 4, 1, 1)); Wsel[0, 1, 0, 0] = 1.0
conv("mmid", "masks", Wsel, [0.0])                                                     # M_mid
add_layer("mmid2", "avgpool", ["mmid"], size=[2, 2], stride=[2, 2])                    # (1,64,64)
Wsel_mh = np.zeros((1, 4, 1, 1)); Wsel_mh[0, 1, 0, 0] = 1.0; Wsel_mh[0, 2, 0, 0] = -1.0
conv("mmid_nohi", "masks", Wsel_mh, [0.0])                                             # M_mid - M_hi: disc, not core
add_layer("mmid_nohi2", "avgpool", ["mmid_nohi"], size=[2, 2], stride=[2, 2])          # (1,64,64)
dog_k = gauss2d(5, 0.6) - gauss2d(5, 1.5)
conv("dogk", "lum2", dog_k[None, None], [-0.015 - 1.0], padding=2)                      # DoG - bias - 1
add_layer("dogk_gate", "add", ["dogk", "mmid_nohi2"])
relu("dogk_g", "dogk_gate")
Wsk = np.zeros((NV, 1, H // 2, W // 2))
for w in range(NW):
    Wsk[slot[f"SKw{w}"], 0] = WIN2[w] * (RR2 > 2.5)
dense("v_knots", "dogk_g", Wsk.reshape(NV, -1), np.zeros(NV))
#   arms: quarter resolution, sigma 0.5 vs 1.5 (2 vs 6 px), inside the low mask, r > 8 px
add_layer("lum4", "avgpool", ["lum2"], size=[2, 2], stride=[2, 2])                     # (1,32,32)
Wsel0 = np.zeros((1, 4, 1, 1)); Wsel0[0, 0, 0, 0] = 1.0; Wsel0[0, 2, 0, 0] = -1.0
conv("mlo_nohi", "masks", Wsel0, [0.0])                                                # M_lo - M_hi
add_layer("mlo4", "avgpool", ["mlo_nohi"], size=[4, 4], stride=[4, 4])                 # (1,32,32)
dog_a = gauss2d(7, 0.5) - gauss2d(7, 1.5)
conv("doga", "lum4", dog_a[None, None], [-0.01 - 1.0], padding=3)
add_layer("doga_gate", "add", ["doga", "mlo4"])
relu("doga_g", "doga_gate")
Wsa = np.zeros((NV, 1, H // 4, W // 4))
for w in range(NW):
    Wsa[slot[f"SAw{w}"], 0] = WIN4[w] * (RR4 > 2.0)
dense("v_arms", "doga_g", Wsa.reshape(NV, -1), np.zeros(NV))

# 3d. aperture fluxes and peak brightness of the smoothed luminance
relu("ls_r", LS)
Wap = np.zeros((NV, 1, H, W))
Wap[slot["F1"], 0] = (RR < 2.5).astype(np.float64)
Wap[slot["F2"], 0] = (RR < 6.0).astype(np.float64)
dense("v_ap", "ls_r", Wap.reshape(NV, -1), np.zeros(NV))
add_layer("pk_pool", "maxpool", ["ls_r"], size=[8, 8], stride=[8, 8])                  # (1,16,16)
Wpk4 = np.zeros((4, 1, 16, 16))
for i, (a, b) in enumerate([(7, 7), (7, 8), (8, 7), (8, 8)]):
    Wpk4[i, 0, a, b] = 1.0
dense("pk4", "pk_pool", Wpk4.reshape(4, -1), np.zeros(4))
add_layer("pk4_r", "reshape", ["pk4"], shape=[1, 1, 4])
add_layer("pk_max", "maxpool", ["pk4_r"], size=[1, 4], stride=[1, 4])                  # (1,1,1)
Wpkv = np.zeros((NV, 1)); Wpkv[slot["PK"], 0] = 1.0
dense("v_pk", "pk_max", Wpkv, np.zeros(NV))

# 3e. 180-degree asymmetry of the mid mask at half resolution (one mirror pair per output)
h2 = H // 2
pos = [(i, j) for i in range(h2) for j in range(h2) if WIN2[-1][i, j] > 0 and i < h2 // 2]
Wd = np.zeros((2 * len(pos), 1, h2, h2))
for k, (i, j) in enumerate(pos):
    mi, mj = h2 - 1 - i, h2 - 1 - j
    Wd[2 * k, 0, i, j] = 1.0;  Wd[2 * k, 0, mi, mj] = -1.0
    Wd[2 * k + 1, 0, i, j] = -1.0;  Wd[2 * k + 1, 0, mi, mj] = 1.0
dense("asym_d", "mmid2", Wd.reshape(2 * len(pos), -1), np.zeros(2 * len(pos)))
relu("asym_r", "asym_d")
Wa = np.zeros((NV, 2 * len(pos)))
for w in range(NW):
    for k, (i, j) in enumerate(pos):
        Wa[slot[f"ASw{w}"], 2 * k] = Wa[slot[f"ASw{w}"], 2 * k + 1] = 2.0 * 4.0 * WIN2[w][i, j]   # full-res pixel units
dense("v_asym", "asym_r", Wa, np.zeros(NV))

# 3f. secondary nucleus: compact peak (DoG > 0.04) AND bright (L > 0.30) AND extended light around it
#     (mid-mask area in a 22x22 px box > 100 px), at half resolution, max over the annulus 8..30 px
add_layer("ls2", "avgpool", ["ls_r"], size=[2, 2], stride=[2, 2])                      # (1,64,64)
conv("dog_raw", "lum2", dog_k[None, None], [0.0], padding=2)
conv("mbox", "mmid2", np.ones((1, 1, 11, 11)) * 4.0, [0.0], padding=5)
step_map("nuc_pk", "dog_raw", 0.04, 0.02)
step_map("nuc_br", "ls2", 0.30, 0.03)
step_map("nuc_ex", "mbox", 100.0, 20.0, extra_bias=-2.0)
add_layer("nuc_sum", "add", ["nuc_pk", "nuc_br", "nuc_ex"])
relu("nuc_r", "nuc_sum")
idx = np.argwhere((RR2 > 4.0) & (RR2 < 15.0))
Wsel2 = np.zeros((len(idx), 1, H // 2, W // 2))
for k, (a, b) in enumerate(idx):
    Wsel2[k, 0, a, b] = 1.0
dense("nuc_sel", "nuc_r", Wsel2.reshape(len(idx), -1), np.zeros(len(idx)))
add_layer("nuc_rs", "reshape", ["nuc_sel"], shape=[1, 1, len(idx)])
add_layer("nuc_max", "maxpool", ["nuc_rs"], size=[1, len(idx)], stride=[1, len(idx)])
Wnv = np.zeros((NV, 1)); Wnv[slot["NUC"], 0] = 1.0
dense("v_nuc", "nuc_max", Wnv, np.zeros(NV))

add_layer("V", "add", ["v_shape", "v_area", "v_col", "v_knots", "v_arms", "v_ap", "v_pk", "v_asym", "v_nuc"])

# ----------------------------------------------------------------------------------------------
# 4. soft comparisons: indicator_j = step((a_j . V - t_j) / eps_j)
# ----------------------------------------------------------------------------------------------
comps = []


def cmp(name, coefs, thr, eps):
    comps.append((name, {slot[k]: v for k, v in coefs.items()}, thr, eps))


def q2rho(q):
    return (1 - q) / (1 + q)


QS = {1: 0.76, 15: 0.70, 2: 0.60, 3: 0.48, 4: 0.30}     # axis-ratio boundaries
for w in range(NW):
    for lev, L in enumerate("emcvi"):                    # low / mid / high / very high isophote / inner region
        for k, q in QS.items():
            cmp(f"{L}{k}w{w}", {f"E{lev}w{w}": 1, f"A{lev}w{w}": -q2rho(q)}, 0, 2 if L in "cv" else 4)
    cmp(f"coreokw{w}", {f"A2w{w}": 1}, 25, 8)             # high isophote large enough to carry a shape
    cmp(f"vcoreokw{w}", {f"A3w{w}": 1}, 8, 4)             # brightest isophote: bars are small blobs here
    cmp(f"iokw{w}", {f"A4w{w}": 1}, 20, 8)
    cmp(f"redw{w}", {f"CRw{w}": 1, f"CBw{w}": -1.30}, 0, 1.5)
    cmp(f"vredw{w}", {f"CRw{w}": 1, f"CBw{w}": -1.70}, 0, 1.5)
    cmp(f"bluew{w}", {f"CBw{w}": 1, f"CRw{w}": -1.00}, 0, 1.5)
    cmp(f"vbluew{w}", {f"CBw{w}": 1, f"CRw{w}": -1.25}, 0, 1.5)
    cmp(f"strw{w}", {f"SAw{w}": 1, f"A0w{w}": -0.0003}, 0, 0.04)    # arm-scale structure per low-mask area
    cmp(f"str2w{w}", {f"SAw{w}": 1, f"A0w{w}": -0.0006}, 0, 0.04)
    cmp(f"knotsw{w}", {f"SKw{w}": 1, f"A1w{w}": -0.0012}, 0, 0.05)
    cmp(f"asymw{w}", {f"ASw{w}": 1, f"A1w{w}": -0.40}, 0, 6)
    cmp(f"asym2w{w}", {f"ASw{w}": 1, f"A1w{w}": -0.75}, 0, 6)
    cmp(f"concw{w}", {f"A2w{w}": -1, f"A0w{w}": 0.10}, 0, 4)        # high isophote area < 10% of low
    cmp(f"flatw{w}", {f"A2w{w}": 1, f"A0w{w}": -0.25}, 0, 4)        # high isophote area > 25% of low
cmp("full_lo", {"A0w2": 1}, 2600, 300)                   # low mask fills the large window
cmp("full_mid", {"A1w2": 1}, 2600, 300)
cmp("s_small", {"A0w1": -1}, -500, 60)                   # size classes: small from the medium window's low area
cmp("s_big", {"A0w2": 1}, 1500, 200)                     # (less star contamination), big from the large window
cmp("big", {"A0w2": 1}, 1500, 200)
cmp("small", {"A0w2": -1}, -350, 60)
cmp("tiny", {"A0w2": -1}, -160, 40)
cmp("bright", {"PK": 1}, 0.70, 0.05)
cmp("faint", {"PK": -1}, -0.55, 0.05)
cmp("nuc", {"NUC": 1}, 0.5, 0.3)

J = len(comps)
Wcmp = np.zeros((2 * J, NV)); bcmp = np.zeros(2 * J)
for j, (name, coefs, thr, eps) in enumerate(comps):
    for s, c in coefs.items():
        Wcmp[2 * j, s] = c / eps
        Wcmp[2 * j + 1, s] = c / eps
    bcmp[2 * j] = -thr / eps
    bcmp[2 * j + 1] = -thr / eps - 1.0
dense("cmp", "V", Wcmp, bcmp)
relu("cmp_r", "cmp")
Wbin = np.zeros((J, 2 * J))
for j in range(J):
    Wbin[j, 2 * j] = 1.0; Wbin[j, 2 * j + 1] = -1.0
dense("bin", "cmp_r", Wbin, np.zeros(J))
relu("bin_r", "bin")
IND = {name: j for j, (name, *_rest) in enumerate(comps)}


# ----------------------------------------------------------------------------------------------
# 5. soft logic layers and the rule table
# ----------------------------------------------------------------------------------------------
def logic_layer(name, inp, in_index, terms):
    """terms: list of (name, {input_name: coef}, bias); output = relu(W x + b)."""
    Wl = np.zeros((len(terms), len(in_index))); bl = np.zeros(len(terms))
    for i, (tname, coefs, bias) in enumerate(terms):
        for src, c in coefs.items():
            Wl[i, in_index[src]] = c
        bl[i] = bias
    dense(name, inp, Wl, bl)
    relu(name + "_r", name)
    return name + "_r", {tname: i for i, (tname, *_r) in enumerate(terms)}


def merge(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    return out


# layer A: window selection.  X_eff = X(w0)*small + X(w1)*medium + X(w2)*big, each product a soft AND;
#          the outer-shape indicators of the big window fall back to the mid / high isophote when the
#          low mask fills the window.  The pieces are summed in layer B.
PER_WINDOW = [f"{L}{k}" for L in "emcvi" for k in QS] + ["coreok", "vcoreok", "iok", "red", "vred", "blue", "vblue",
                                                         "str", "str2", "knots", "asym", "asym2", "conc", "flat"]
GLOBAL = ["full_lo", "full_mid", "s_small", "s_big", "big", "small", "tiny", "bright", "faint", "nuc"]
termsA = [(g, {g: 1.0}, 0.0) for g in GLOBAL]
pieces = {}
for X in PER_WINDOW:
    ps = [f"{X}#0", f"{X}#1", f"{X}#2"]
    if X.startswith("e"):
        # outer shape: for small galaxies the faint isophote is blurred round by the PSF, so use the mid
        # isophote; medium and big galaxies use faint OR mid (OR = sum minus overlap); when the faint
        # mask fills the big window fall back to the mid, then the high isophote
        # soft OR under a soft size weight s:  AND(e,s) + AND(m,s) - AND(e,m,s)
        k = X[1:]
        termsA.append((ps[0], {f"m{k}w0": 1.0, "s_small": 1.0}, -1.0))                                 # small: mid
        termsA.append((ps[1], {f"e{k}w1": 1.0, "s_small": -1.0, "s_big": -1.0}, 0.0))                  # medium: e
        ps.append(f"{X}#m1"); termsA.append((ps[-1], {f"m{k}w1": 1.0, "s_small": -1.0, "s_big": -1.0}, 0.0))          # medium: m
        ps.append(f"{X}#or1"); termsA.append((ps[-1], {f"e{k}w1": 1.0, f"m{k}w1": 1.0, "s_small": -1.0, "s_big": -1.0}, -1.0))
        termsA.append((ps[2], {f"e{k}w2": 1.0, "s_big": 1.0, "full_lo": -1.0}, -1.0))                  # big (not full): e
        ps.append(f"{X}#m2"); termsA.append((ps[-1], {f"m{k}w2": 1.0, "s_big": 1.0, "full_lo": -1.0}, -1.0))          # big: m
        ps.append(f"{X}#or2"); termsA.append((ps[-1], {f"e{k}w2": 1.0, f"m{k}w2": 1.0, "s_big": 1.0, "full_lo": -1.0}, -2.0))
        ps.append(f"{X}#3"); termsA.append((ps[-1], {f"m{k}w2": 1.0, "full_lo": 1.0, "full_mid": -1.0}, -1.0))
        ps.append(f"{X}#4"); termsA.append((ps[-1], {f"c{k}w2": 1.0, "full_mid": 1.0}, -1.0))
    else:
        termsA.append((ps[0], {f"{X}w0": 1.0, "s_small": 1.0}, -1.0))                  # X(w0) AND small
        termsA.append((ps[1], {f"{X}w1": 1.0, "s_small": -1.0, "s_big": -1.0}, 0.0))   # X(w1) AND medium
        termsA.append((ps[2], {f"{X}w2": 1.0, "s_big": 1.0}, -1.0))                    # X(w2) AND big
    pieces[X] = ps
outA, IA = logic_layer("logicA", "bin_r", IND, termsA)


def X(name, s=1.0):
    """effective (window-selected) indicator as a linear combination of its pieces"""
    return {p: (-s if "#or" in p else s) for p in pieces[name]}


# layer B: effective indicators and derived morphological terms
termsB = [(g, {g: 1.0}, 0.0) for g in GLOBAL]
termsB += [(n_, X(n_), 0.0) for n_ in PER_WINDOW]
termsB += [
    ("round", X("e1", -1.0), 1.0),                                    # q > 0.8
    ("moder", merge(X("e1"), X("e2", -1.0)), 0.0),                    # 0.6 < q < 0.8
    ("elong", merge(X("e2"), X("e3", -1.0)), 0.0),                    # 0.45 < q < 0.6
    ("thin", X("e3"), 0.0),                                           # q < 0.45
    ("vthin", X("e4"), 0.0),                                          # q < 0.30
    ("notthin", X("e3", -1.0), 1.0),
    ("notelong", X("e2", -1.0), 1.0),
    # bars: an elongated inner region (high isophote, brightest isophote, or the mid isophote inside the
    # inner window) in a galaxy whose outer (faint or mid) isophote is not elongated
    # (a merging pair also has an elongated inner region, so bars additionally require symmetry)
    ("bar", merge(X("c3"), X("coreok"), X("e3", -1.0), X("asym", -1.0)), -1.0),        # core q<0.48, outer q>0.48
    ("bar_w", merge(X("c2"), X("coreok"), X("e2", -1.0), X("asym", -1.0)), -1.0),      # core q<0.6, outer q>0.6
    ("barv", merge(X("v3"), X("vcoreok"), X("e3", -1.0), X("asym", -1.0)), -1.0),      # brightest isophote q<0.48
    ("barv_w", merge(X("v2"), X("vcoreok"), X("e2", -1.0), X("asym", -1.0)), -1.0),    # brightest isophote q<0.6
    ("bari", merge(X("i3"), X("iok"), X("e2", -1.0), X("asym", -1.0)), -1.0),          # inner region q<0.48
    ("bari_w", merge(X("i2"), X("iok"), X("e2", -1.0), X("asym", -1.0)), -1.0),        # inner region q<0.6
    ("notvcore", X("vcoreok", -1.0), 1.0),                            # no saturated core: no bright bulge
    ("notbright", {"bright": -1.0}, 1.0),
    ("eo_bulge", merge(X("e3"), X("coreok"), X("c2", -1.0)), -1.0),   # thin outer, measurable rounder core
    ("eo_nobulge", merge(X("e3"), X("c3")), -1.0),                    # thin outer and thin core
    ("eo_nocore", merge(X("e3"), X("coreok", -1.0)), 0.0),            # thin outer and no bright core at all
    ("redsmooth", merge(X("red"), X("str", -1.0)), 0.0),
    ("bluestr", merge(X("blue"), X("str")), -1.0),
    ("sym", X("asym", -1.0), 1.0),
    ("notred", X("red", -1.0), 1.0),
    ("notblue", X("blue", -1.0), 1.0),
    ("notstr", X("str", -1.0), 1.0),
    ("notsmall", {"small": -1.0}, 1.0),
    ("merge_pair", merge({"nuc": 1.0}, X("asym")), -1.0),
    ("midcol", merge(X("red", -1.0), X("blue", -1.0)), 1.0),          # neither red nor blue
]
outB, IB = logic_layer("logicB", outA, IA, termsB)

rules = {
    0: {"asym": 2.0, "blue": 0.6, "knots": 0.3, "faint": 0.5, "nuc": -2.0, "red": -1.5, "thin": -1.2, "vthin": -1.0},
    1: {"merge_pair": 2.2, "asym": 1.4, "asym2": 1.2, "nuc": 0.8, "thin": -0.8, "faint": -0.3},
    2: {"round": 2.0, "redsmooth": 1.2, "red": 0.8, "notstr": 0.6, "sym": 0.6, "bright": 0.4, "conc": -0.4, "bar": -1.5, "bar_w": -0.8,
        "barv": -1.0, "barv_w": -0.5, "bari": -1.0, "bari_w": -0.5, "blue": -1.0, "asym": -1.3, "nuc": -0.8, "elong": -1.5,
        "thin": -2.0, "str2": -0.8, "knots": -0.5},
    3: {"moder": 2.0, "elong": 1.2, "redsmooth": 1.2, "red": 0.8, "notstr": 0.8, "sym": 0.5, "conc": -0.4, "blue": -1.5, "asym": -1.3,
        "nuc": -0.8, "thin": -1.5, "str2": -0.8, "round": -0.8, "bar": -1.0, "bari": -0.8, "knots": -0.5},
    4: {"elong": 1.0, "thin": 1.2, "vthin": -1.2, "red": 0.8, "small": 0.8, "tiny": 0.4, "notstr": 0.5, "sym": 0.4,
        "blue": -1.0, "big": -1.5, "asym": -1.0, "str": -0.8, "round": -2.0, "moder": -1.0, "c4": -0.6},
    5: {"bar": 1.5, "bar_w": 2.0, "barv": 1.5, "barv_w": 1.5, "bari": 0.5, "bari_w": 0.5, "notthin": 0.5, "str": 0.4,
        "thin": -2.0, "asym": -0.6, "nuc": -0.6},
    6: {"str": 1.0, "str2": 0.3, "knots": 0.2, "midcol": 0.8, "round": 0.5, "moder": 0.9, "elong": 0.5, "bright": 0.8,
        "vcoreok": 0.6, "conc": 0.6, "sym": 0.4, "notsmall": 0.4, "thin": -1.5, "vred": -0.6, "asym": -1.2, "nuc": -0.6,
        "bar": -1.0, "bar_w": -0.8, "barv": -0.6, "bari": -1.0, "bari_w": -0.8, "vblue": -0.8},
    7: {"blue": 1.4, "vblue": 0.6, "bluestr": 0.8, "str": 0.6, "knots": 0.6, "faint": 0.4, "notbright": 0.4, "notvcore": 0.3,
        "big": 0.4, "round": 0.3, "moder": 0.3, "conc": 0.3, "red": -1.5, "thin": -1.0, "vthin": -1.5, "small": -0.8,
        "tiny": -0.8, "bar": -0.8, "bar_w": -1.0, "bari": -0.8, "bari_w": -1.0, "nuc": -0.4, "asym": -0.6},
    8: {"thin": 1.2, "vthin": 1.8, "eo_nobulge": 0.8, "eo_nocore": 0.8, "blue": 0.8, "faint": 0.5, "flat": 0.5, "notvcore": 0.6,
        "red": -0.8, "vred": -0.5, "eo_bulge": -0.8, "conc": -0.5, "bright": -0.3, "round": -2.0, "moder": -1.5},
    9: {"thin": 1.2, "vthin": 1.2, "eo_bulge": 0.8, "red": 1.0, "vred": 0.4, "bright": 0.5, "vcoreok": 0.6, "conc": 0.5,
        "blue": -1.0, "vblue": -0.6, "eo_nobulge": -0.5, "eo_nocore": -0.5, "flat": -0.5, "round": -2.0, "moder": -1.5},
}
bias = np.array([-1.3, -1.2, 0.0, -0.2, -1.3, -0.6, -0.5, -0.3, -0.2, -0.2])
Ws = np.zeros((10, len(IB)))
for k, rule in rules.items():
    for name, c in rule.items():
        Ws[k, IB[name]] = c
dense("scores", outB, Ws, bias)

# ----------------------------------------------------------------------------------------------
# write
# ----------------------------------------------------------------------------------------------
out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
(out / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
np.savez(out / "weights.npz", **{k: v.astype(np.float32) for k, v in weights.items()})
total = sum(int(np.prod(v.shape)) for v in weights.values())
print(f"wrote {len(layers)} layers, {total:,} weights to {out}")
