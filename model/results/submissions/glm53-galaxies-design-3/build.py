"""Builds a hand-designed galaxy classifier. Reads no files; all weights come
from formulas (Gaussian kernels, polar ring/sector masks, Fourier tables) and
hand-chosen rule constants based on general galaxy-morphology knowledge.

python3 build.py OUT_DIR -> OUT_DIR/network.json, OUT_DIR/weights.npz
"""
import sys, json, math
import numpy as np
from pathlib import Path

NR, NT = 12, 24          # radial rings, azimuthal samples
NR1 = NR + 1             # texture rings incl. corner sky reference
CLIP = 1.6               # polar-map clip (clean sinusoid peaks reach ~1.41)
RC = 3.5 * 1.24 ** np.arange(NR)   # ring radii in original pixels (max ~37)

# ---------------------------------------------------------------- plumbing
L, W = [], {}

def conv(name, inp, out, k, stride, pad, w, b):
    L.append({"name": name, "op": "conv", "inputs": [inp], "out": out,
              "kernel": [k, k], "stride": stride, "padding": pad})
    W[name + ".W"] = np.asarray(w, np.float32).reshape(out, -1, k, k)
    W[name + ".b"] = np.zeros(out, np.float32) + b

def conv1(name, inp, w, b=0.0):
    w = np.asarray(w, np.float32)
    conv(name, inp, w.shape[0], 1, 1, 0, w.reshape(w.shape[0], -1, 1, 1),
         np.zeros(w.shape[0], np.float32) + b)

def dense(name, inp, out, w, b):
    L.append({"name": name, "op": "dense", "inputs": [inp], "out": out})
    W[name + ".W"] = np.asarray(w, np.float32)
    W[name + ".b"] = np.asarray(b, np.float32)

def rel(name, inp):
    L.append({"name": name, "op": "relu", "inputs": [inp]})

def addl(name, *inps):
    L.append({"name": name, "op": "add", "inputs": list(inps)})

def reshp(name, inp, shape):
    L.append({"name": name, "op": "reshape", "inputs": [inp], "shape": shape})

def norml(name, inp):
    L.append({"name": name, "op": "norm", "inputs": [inp]})

def pool(name, inp, kind, size, stride):
    L.append({"name": name, "op": kind, "inputs": [inp],
              "size": list(size), "stride": list(stride)})

def gauss2d(k, sigma):
    t = (np.arange(k) - (k - 1) / 2.0) / sigma
    g = np.exp(-t * t / 2.0)
    ker = np.outer(g, g)
    return ker / ker.sum()

# coordinate grids -----------------------------------------------------------
# A1 (64x64) pixel (y,x) sits at original coords (2y-63.5, 2x-63.5).
yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
CY, CX = 2.0 * yy - 63.5, 2.0 * xx - 63.5
RR = np.hypot(CX, CY)
TH = np.arctan2(CY, CX)
# pooled 32x32 maps: pixel center at 4y-62
pyy, pxx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
PRR = np.hypot(4.0 * pxx - 62.0, 4.0 * pyy - 62.0)

def ring_masks(res):
    """radial Gaussian masks on a grid; res=64 -> A1 grid, 32 -> pooled grid"""
    if res == 64:
        rr = RR
    else:
        rr = PRR
    masks = []
    for i in range(NR):
        sr = min(7.0, max(1.3, 0.30 * RC[i])) if res == 64 else min(9.0, max(2.0, 0.55 * RC[i]))
        m = np.exp(-((rr - RC[i]) / sr) ** 2)
        masks.append(m / m.sum())
    out = np.stack(masks)
    if res == 32:                    # corner annulus r~66..88: sky reference
        cm = np.exp(-((rr - 74.0) / 9.0) ** 2)
        out = np.concatenate([out, (cm / cm.sum())[None, :]], axis=0)
    return out.reshape(out.shape[0], -1)

THJ = np.arange(NT) * 2 * np.pi / NT

# ------------------------------------------------ input conditioning
conv1("lum", "image", [[0.34, 0.33, 0.33]])              # luminance
conv1("col", "image", [[0.55, -0.28, -0.27]])            # red minus blue/green
norml("nlum", "lum")
norml("ncol", "col")

# ------------------------------------------------ luminance pyramid
conv("b0", "nlum", 1, 5, 1, 2, gauss2d(5, 1.2), 0)       # 128, sigma~1.2
G9 = gauss2d(9, 2.0)
conv("a1", "b0", 1, 9, 2, 4, G9, 0)                      # 64
conv("a2", "a1", 1, 9, 2, 4, G9, 0)                      # 32
conv("a3", "a2", 1, 9, 2, 4, G9, 0)                      # 16
conv("ca1", "ncol", 1, 9, 2, 4, G9, 0)                   # colour, 64

# ------------------------------------------------ bandpass texture
conv("g3a", "b0", 1, 5, 1, 2, gauss2d(5, 2.6), 0)        # sigma~2.9
conv("g8a", "g3a", 1, 9, 1, 4, gauss2d(9, 5.5), 0)       # sigma~6.2
conv1("ng3", "g3a", [[-1]])
addl("bp1", "b0", "ng3")                                 # fine bandpass
conv1("ng8", "g8a", [[-1]])
addl("bp2", "g3a", "ng8")                                # coarse bandpass
for tag, src in (("1", "bp1"), ("2", "bp2")):
    conv1("pm" + tag, src, [[1], [-1]])                  # [x; -x]
    rel("pm" + tag + "r", "pm" + tag)
    conv1("abp" + tag, "pm" + tag + "r", [[1, 1]])       # |bandpass|
pool("t1p", "abp1", "avgpool", [4, 4], [4, 4])           # 32x32
pool("t2p", "abp2", "avgpool", [4, 4], [4, 4])

# ------------------------------------------------ polar ring map + harmonics
# Each ring channel holds NT azimuthal samples plus one "scale" slot
# K_i = 5*(ring mean + 0.7).  The channel norm then divides every sample by
# ~K_i, giving the ring's shape relative to its own brightness (with a
# floor), while faint stars and sky noise stay small.  A sample is then
# clipped to +-CLIP so bright star spikes cannot dominate.
GAM = 5.0; FLO = 0.7; CLIP2 = 1.25
Pw = np.zeros((NR * (NT + 1), 4096))
for i in range(NR):
    sr = min(7.0, max(1.3, 0.30 * RC[i]))
    st = max(1.1, RC[i] * (2 * math.pi / NT) * 0.8)
    for j in range(NT):
        dt = np.angle(np.exp(1j * (TH - THJ[j])))
        w = np.exp(-((RR - RC[i]) / sr) ** 2 - ((RR * dt) / st) ** 2)
        Pw[i * (NT + 1) + j] = (w / w.sum()).ravel()
    rm = np.exp(-((RR - RC[i]) / sr) ** 2)          # radial mask (sums to ~1)
    Pw[i * (NT + 1) + NT] = (GAM * rm / rm.sum()).ravel()
dense("pol", "a1", NR * (NT + 1), Pw,
      np.concatenate([np.zeros(NR * NT), GAM * FLO * np.ones(NR)]))
reshp("polr", "pol", [NR, NT + 1, 1])
norml("poln", "polr")                     # divides each ring by ~K_i

# second branch: same samples, per-ring STD normalisation (amplitude-
# equalised).  Phase-coherence features (spirals, bars) are read from this
# branch; magnitude features come from the K-normalised branch.
Pw2 = Pw.reshape(NR, NT + 1, 4096)[:, :NT, :].reshape(NR * NT, 4096)
dense("pol2", "a1", NR * NT, Pw2, np.zeros(NR * NT))
reshp("pol2r", "pol2", [NR, NT, 1])
norml("poln2", "pol2r")

# clip each sample to +-CLIP (suppresses star spikes)
Wc = np.zeros((4 * NR, NR, 1, 1)); bc = np.zeros(4 * NR)
Wd = np.zeros((NR, 4 * NR, 1, 1))
for i in range(NR):
    Wc[4 * i, i] = 1; Wc[4 * i + 1, i] = -1
    Wc[4 * i + 2, i] = 1; bc[4 * i + 2] = -CLIP
    Wc[4 * i + 3, i] = -1; bc[4 * i + 3] = -CLIP
    Wd[i, 4 * i] = 1; Wd[i, 4 * i + 1] = -1
    Wd[i, 4 * i + 2] = -1; Wd[i, 4 * i + 3] = 1
for tag, src, bb in (("a", "poln", bc), ("b", "poln2", np.where(bc != 0, -CLIP2, 0.0))):
    conv1("pc" + tag, src, Wc, bb)
    rel("pc" + tag + "r", "pc" + tag)
    conv1("pclip" + tag, "pc" + tag + "r", Wd)           # clipped ring maps

Hw = np.zeros((8 * NR, NR * (NT + 1)))
Hw2 = np.zeros((8 * NR, NR * NT))
for i in range(NR):
    for m in range(1, 5):
        cs = np.cos(m * THJ) * 2 / NT; sn = np.sin(m * THJ) * 2 / NT
        Hw[i * 8 + (m - 1) * 2,     i * (NT + 1):i * (NT + 1) + NT] = cs
        Hw[i * 8 + (m - 1) * 2 + 1, i * (NT + 1):i * (NT + 1) + NT] = sn
        Hw2[i * 8 + (m - 1) * 2,     i * NT:(i + 1) * NT] = cs
        Hw2[i * 8 + (m - 1) * 2 + 1, i * NT:(i + 1) * NT] = sn
dense("harm", "pclipa", 8 * NR, Hw, np.zeros(8 * NR))    # magnitude branch
dense("harm2", "pclipb", 8 * NR, Hw2, np.zeros(8 * NR))  # phase branch

# sky-gradient plane subtraction for m=1: a planar background contributes
# C1_i ~ g * r_i; estimate g from the two outermost (mostly sky) rings.
RBAR = (RC[10] + RC[11]) / 2
def dip_terms(ids, m, c, sc=1.0, h2=False):
    fslot = s_h2 if h2 else s_h
    t = [("sgn", fslot(i, m, c), sc) for i in ids]
    for i in ids:                       # remove r-proportional plane part
        f = RC[i] / RBAR / 2.0
        t += [("sgn", fslot(10, m, c), -sc * f), ("sgn", fslot(11, m, c), -sc * f)]
    return t

# ------------------------------------------------ radial profiles
RM64 = ring_masks(64)
dense("rmean", "a1", NR, RM64, np.zeros(NR))             # ring mean light
reshp("rmeanr", "rmean", [1, NR, 1])
norml("mn", "rmeanr")                                    # normalised profile
dense("crmean", "ca1", NR, RM64, np.zeros(NR))           # colour ring means
RM32 = ring_masks(32)
dense("t1r", "t1p", NR1, RM32, np.zeros(NR1))            # texture ring means + corner
dense("t2r", "t2p", NR1, RM32, np.zeros(NR1))

# ------------------------------------------------ assemble feature vector F
def expand(name, inp, wmat, nslots, off):
    dense(name, inp, nslots, wmat, np.zeros(nslots))
THOFF = NR + 16 * NR                                    # 204: texture slot base
NF = THOFF + 3 * NR1                                    # 243
E_MN = np.zeros((NF, NR))
E_HA = np.zeros((NF, 8 * NR)); E_H2 = np.zeros((NF, 8 * NR))
E_T1 = np.zeros((NF, NR1)); E_T2 = np.zeros((NF, NR1)); E_CM = np.zeros((NF, NR))
for i in range(NR):
    E_MN[i, i] = 1                     # slots 0..11   Mn_i
    E_CM[THOFF + 2 * NR1 + i, i] = 1   # colour ring means
for i in range(NR1):
    E_T1[THOFF + i, i] = 1             # T1 rings 0..11, corner at 12
    E_T2[THOFF + NR1 + i, i] = 1
for k in range(8 * NR):
    E_HA[NR + k, k] = 1                # slots 12..107: magnitude harmonics
    E_H2[NR + 8 * NR + k, k] = 1       # slots 108..203: phase harmonics
expand("f_mn", "mn", E_MN, NF, 0)
expand("f_ha", "harm", E_HA, NF, 0)
expand("f_h2", "harm2", E_H2, NF, 0)
expand("f_t1", "t1r", E_T1, NF, 0)
expand("f_t2", "t2r", E_T2, NF, 0)
expand("f_cm", "crmean", E_CM, NF, 0)
addl("F", "f_mn", "f_ha", "f_h2", "f_t1", "f_t2", "f_cm")

# feature slot helpers
def s_mn(i):  return i
def s_h(i, m, c):  return NR + i * 8 + (m - 1) * 2 + c        # magnitude branch
def s_h2(i, m, c):  return NR + 8 * NR + i * 8 + (m - 1) * 2 + c  # phase branch
def s_t1(i):  return THOFF + i                 # i=12 is the corner sky slot
def s_t2(i):  return THOFF + NR1 + i
def s_cm(i):  return THOFF + 2 * NR1 + i

# ------------------------------------------------ decision stack: derived features
dense("d1", "F", 2 * NF, np.concatenate([np.eye(NF), -np.eye(NF)]), np.zeros(2 * NF))
rel("d1r", "d1")                       # u1 = [F+ ; F-]

D2, D2i = [], {}
def d2row(name, terms, bias=0.0):
    row = np.zeros(2 * NF)
    for kind, slot, coef in terms:
        if kind == "sgn":
            row[slot] += coef; row[NF + slot] -= coef
        elif kind == "abs":
            row[slot] += coef; row[NF + slot] += coef
        elif kind == "pos":
            row[slot] += coef
        elif kind == "neg":
            row[NF + slot] += coef
    D2i[name] = len(D2); D2.append((row, bias))

for i in range(NR):                                   # shifted radial profile
    d2row("mns%d" % i, [("sgn", s_mn(i), 1.0)], 0.3)
def grp(name, ids, m, sc):
    t = []
    for i in ids:
        t += [("abs", s_h(i, m, 0), sc), ("abs", s_h(i, m, 1), sc)]
    d2row(name, t)
grp("E2n", range(0, 4), 2, 0.25)                      # elongation, inner
grp("E2m", range(4, 8), 2, 0.25)                      # mid
grp("E2o", range(8, 12), 2, 0.25)                     # outer
grp("E2x", range(5, 10), 2, 0.20)                     # disk radii
for i in range(NR):                                   # plane-subtracted dipoles
    d2row("d1c%d" % i, dip_terms([i], 1, 0))
    d2row("d1s%d" % i, dip_terms([i], 1, 1))
grp("D3g", range(3, 10), 3, 1.0 / 14)
grp("D4g", range(3, 10), 4, 1.0 / 14)
d2row("barA",  [("sgn", s_h2(i, 2, 0), 1.0) for i in range(1, 6)])
d2row("barB",  [("sgn", s_h2(i, 2, 1), 1.0) for i in range(1, 6)])
d2row("outA",  [("sgn", s_h2(i, 2, 0), 1.0) for i in range(6, 11)])
d2row("outB",  [("sgn", s_h2(i, 2, 1), 1.0) for i in range(6, 11)])
d2row("axisA", [("sgn", s_h2(i, 2, 0), 1.0) for i in range(2, 11)])
d2row("axisB", [("sgn", s_h2(i, 2, 1), 1.0) for i in range(2, 11)])
d2row("d1A",   dip_terms(range(3, 8), 1, 0, h2=True))
d2row("d1B",   dip_terms(range(3, 8), 1, 1, h2=True))
d2row("d1oA",  dip_terms(range(6, 11), 1, 0, h2=True))
d2row("d1oB",  dip_terms(range(6, 11), 1, 1, h2=True))
SPC = [("t", 0.45, range(2, 11)), ("m", 0.85, range(3, 11)), ("l", 1.3, range(4, 11))]
for tag, k, ids in SPC:                               # spiral pitch projections,
    lr = np.log(RC[list(ids)])                        # orthogonal to fixed phase
    lrc = lr - lr.mean()
    def orth(v):
        v = v - v.mean()
        v = v - (v * lrc).sum() / (lrc * lrc).sum() * lrc
        return v / np.sqrt((v * v).sum())          # unit-norm residual basis
    ca = orth(np.cos(k * lr))
    sa = orth(np.sin(k * lr))
    for sg in (1, -1):
        ta, tb = [], []
        for j, i in enumerate(ids):
            ta += [("sgn", s_h2(i, 2, 0), ca[j]), ("sgn", s_h2(i, 2, 1), sg * sa[j])]
            tb += [("sgn", s_h2(i, 2, 1), ca[j]), ("sgn", s_h2(i, 2, 0), -sg * sa[j])]
        d2row("sp%sA%d" % (tag, sg), ta)
        d2row("sp%sB%d" % (tag, sg), tb)
d2row("CONC",  [("sgn", s_mn(i), s) for i, s in
                ((0, 1/3.), (1, 1/3.), (2, 1/3.), (3, -1/3.), (4, -1/3.), (5, -1/3.))])
d2row("CONCc", [("sgn", s_mn(i), s) for i, s in ((0, .5), (1, .5), (2, -.5), (3, -.5))])
d2row("TX1", [("sgn", s_t1(i), 1.0) for i in range(3, 10)] +
             [("sgn", s_t1(12), -7.0)])
d2row("TX2", [("sgn", s_t2(i), 1.0) for i in range(3, 10)] +
             [("sgn", s_t2(12), -7.0)])
d2row("TXc", [("sgn", s_t1(i), 1.0) for i in range(0, 3)] +
             [("sgn", s_t1(12), -3.0)])
for i in range(NR):                                   # per-ring elongation
    d2row("e2r%d" % i, [("abs", s_h(i, 2, 0), 1.0), ("abs", s_h(i, 2, 1), 1.0)], -0.30)
d2row("CRED", [("sgn", s_cm(i), s) for i, s in
               ((0, 1/3.), (1, 1/3.), (2, 1/3.), (7, -1/3.), (8, -1/3.), (9, -1/3.))])
d2row("SKYT", [("sgn", s_t1(i), 0.5) for i in (10, 11)])
ND = len(D2)
d2W = np.stack([r for r, b in D2]); d2b = np.array([b for r, b in D2])
dense("d2", "d1r", ND, d2W, d2b)
dense("d3", "d2", 2 * ND, np.concatenate([np.eye(ND), -np.eye(ND)]), np.zeros(2 * ND))
rel("d3r", "d3")                       # u3 = [v+ ; v-]

# ------------------------------------------------ indicators (d4)
D4, D4i = [], {}
def d4row(name, terms, bias=0.0):
    row = np.zeros(2 * ND)
    for kind, nm, coef in terms:
        j = D2i[nm]
        if kind == "sgn":
            row[j] += coef; row[ND + j] -= coef
        elif kind == "abs":
            row[j] += coef; row[ND + j] += coef
        elif kind == "pos":
            row[j] += coef
    D4i[name] = len(D4); D4.append((row, bias))
for nm in D2i:                                        # signed passthrough
    d4row("P_" + nm, [("sgn", nm, 1.0)])
for nm in D2i:                                        # alias plain names
    D4i[nm] = D4i["P_" + nm]
for nm in ("barA", "barB", "axisA", "axisB", "d1A", "d1B"):
    d4row("Q_" + nm, [("abs", nm, 1.0)])
d4row("BAR",  [("abs", "barA", 1.0), ("abs", "barB", 1.0)])
d4row("BARX", [("abs", "barA", 1.0), ("abs", "barB", 1.0),
               ("abs", "outA", -0.8), ("abs", "outB", -0.8)])
d4row("AXIS", [("abs", "axisA", 1.0), ("abs", "axisB", 1.0)])
d4row("DBL",  [("abs", "d1A", 1.0), ("abs", "d1B", 1.0)])
d4row("DBLo", [("abs", "d1oA", 1.0), ("abs", "d1oB", 1.0)])
for nm, ids in (("D1i", range(0, 4)), ("D1m", range(4, 8)),
                ("D1o", range(8, 12)), ("D1x", range(1, 7))):
    sc = 1.0 / len(list(ids))
    d4row(nm, sum([[("abs", "d1c%d" % i, sc), ("abs", "d1s%d" % i, sc)]
                   for i in ids], []))
for tag, k, ids in SPC:
    for sg in (1, -1):
        d4row("SP%s%d" % (tag, sg),
              [("abs", "sp%sA%d" % (tag, sg), 1.0), ("abs", "sp%sB%d" % (tag, sg), 1.0)])
    d4row("SP%s" % tag, [("abs", "sp%sA1" % tag, 1.0), ("abs", "sp%sB1" % tag, 1.0),
                         ("abs", "sp%sA-1" % tag, 1.0), ("abs", "sp%sB-1" % tag, 1.0)])
d4row("EXT", [("pos", "mns%d" % i, (i - 3) / 8.0) for i in range(4, NR)])
d4row("ACT", [("pos", "mns%d" % i, 1.0 / 12.0) for i in range(NR)])
d4row("NEL", [("pos", "e2r%d" % i, 1.0) for i in range(2, 11)])
d4row("NELW", [("pos", "e2r%d" % i, (i - 1) / 8.0) for i in range(2, 11)])
d4row("BARCON", [("pos", "e2r%d" % i, 1.0) for i in (2, 3, 4)] +
                [("pos", "e2r%d" % i, -1.5) for i in (0, 1)])
NE = len(D4)
d4W = np.stack([r for r, b in D4]); d4b = np.array([b for r, b in D4])
dense("d4", "d3r", NE, d4W, d4b)
dense("d5", "d4", 2 * NE, np.concatenate([np.eye(NE), -np.eye(NE)]), np.zeros(2 * NE))
rel("d5r", "d5")                       # u5 = [w+ ; w-]

# ------------------------------------------------ gates (d6) and clamping
G, Gi = [], {}
def gate(name, terms, slope, thr):
    row = np.zeros(2 * NE)
    for nm, coef in terms:
        j = D4i[nm]
        row[j] += coef; row[NE + j] -= coef      # signed value
    G.append((row, -slope * thr, slope))
    Gi[name] = len(G) - 1
def hi(name, nm, slope, thr):  gate(name, [(nm, 1.0)], slope, thr)
def lo(name, nm, slope, thr):  gate(name, [(nm, -1.0)], slope, -thr)

hi("gE2mL2", "E2m", 4, 0.15)                               # at least mildly flat
hi("gE2mHI", "E2m", 4, 0.60);  lo("gE2mLO", "E2m", 5, 0.15)
hi("gE2nHI", "E2n", 4, 0.25)
hi("gE2oHI", "E2o", 4, 0.65);  lo("gE2oLO", "E2o", 4, 0.20)
hi("gE2xHI", "E2x", 4, 0.75)
gate("gAXIS", [("AXIS", 1.0), ("E2x", 2.5)], 1.0, 10.0)
gate("gAXISv", [("AXIS", 1.0), ("E2x", 2.5)], 1.0, 12.0)
hi("gBAR",   "BAR", 1.5, 4.2)
hi("gDBL",   "DBL", 1.5, 1.8); hi("gDBLo", "DBLo", 1.5, 1.3)
hi("gSPt",   "SPt", 1.5, 3.2); hi("gSPm", "SPm", 1.5, 3.5); hi("gSPl", "SPl", 1.5, 3.5)
hi("gD1m",   "D1m", 4, 0.32);  hi("gD1o", "D1o", 4, 0.60)
gate("gIRR",  [("D3g", 1.0), ("D4g", 1.0)], 3, 0.35)
hi("gCONC",  "CONC", 2, 0.85); hi("gCONCv", "CONC", 2, 1.20)
lo("gCONClo", "CONC", 2, 0.60); hi("gCONCc", "CONCc", 3, 0.75)
hi("gEXTl",  "EXT", 2, 0.28);  lo("gEXTs", "EXT", 3, 0.10)
hi("gTX",    "TX1", 3, 0.30);  hi("gTXv", "TX1", 3, 0.65)
hi("gRED",   "CRED", 1.2, 2.2)
gate("gFLAT", [("E2m", 1.0), ("E2o", 0.6), ("E2x", 1.0)], 2.0, 1.7)
gate("gTHIN",  [("E2n", 1.0), ("E2m", 1.0)], 2.5, 1.2)
hi("gNEL",   "NEL", 1.0, 3.5)
NG = len(G)
gW = np.stack([r for r, b, s in G])
gb = np.array([b for r, b, s in G])
gs = np.array([s for r, b, s in G])
dense("d6", "d5r", NG, gW * gs[:, None], gb)           # slope folded into W
rel("d6r", "d6")                                       # open gates
dense("d6c", "d6r", 2 * NG,
      np.concatenate([np.eye(NG), np.eye(NG)]),
      np.concatenate([np.zeros(NG), -np.ones(NG)]))
rel("d6cr", "d6c")
dense("d7a", "d6cr", NG,
      np.concatenate([np.eye(NG), -np.eye(NG)], axis=1), np.zeros(NG))  # gates in [0,1]

# ------------------------------------------------ class scores (final layer)
CL = ["disturbed", "merging", "round", "inbetween", "cigar",
      "barred", "tight", "loose", "edgeon", "edgeonbulge"]
def sc(gate_names_pos, gate_names_neg, bias):
    r = np.zeros(NG)
    for gname, wgt in gate_names_pos:
        r[Gi[gname]] += wgt
    for gname, wgt in gate_names_neg:
        r[Gi[gname]] -= wgt
    return r, bias
S = []
S.append(sc([("gIRR", .5), ("gD1o", .5), ("gD1m", .45), ("gCONClo", .4),
             ("gTX", .35), ("gDBL", .35)],                       # disturbed
            [("gSPt", .5), ("gSPm", .4), ("gSPl", .4), ("gBAR", .6), ("gAXIS", .7),
             ("gE2mHI", .5), ("gE2xHI", .5)], 0.10))
S.append(sc([("gDBL", .6), ("gDBLo", .5), ("gD1m", .5), ("gD1o", .5),
             ("gIRR", .4), ("gTX", .35), ("gEXTl", .25)],       # merging
            [("gSPt", .5), ("gSPm", .4), ("gSPl", .4), ("gBAR", .5), ("gAXIS", .7),
             ("gE2mHI", .5)], 0.05))
S.append(sc([("gE2mLO", .8), ("gE2oLO", .6), ("gCONC", .35)],   # round smooth
            [("gE2mL2", .5), ("gE2mHI", .8), ("gSPt", .6), ("gSPm", .5), ("gSPl", .5),
             ("gBAR", .4), ("gTX", .45), ("gD1m", .45), ("gD1o", .45), ("gIRR", .4),
             ("gAXIS", .7), ("gDBL", .45), ("gDBLo", .35), ("gD1o", .5),
             ("gNEL", .5), ("gE2xHI", .5)], 0.30))
S.append(sc([("gE2mL2", .45), ("gCONC", .35)],                  # in-between smooth
            [("gE2mLO", .8), ("gE2mHI", .8), ("gNEL", .6), ("gAXIS", .5),
             ("gDBL", .35), ("gDBLo", .3), ("gD1o", .35),
             ("gSPt", .5), ("gSPm", .5), ("gSPl", .5), ("gBAR", .4), ("gTX", .45),
             ("gD1m", .4), ("gIRR", .4)], 0.55))
S.append(sc([("gE2mHI", .8), ("gE2xHI", .5), ("gNEL", .45), ("gE2oHI", .4),
             ("gTHIN", .3), ("gCONC", .25)],                     # cigar
            [("gAXIS", .7), ("gTX", .4), ("gSPt", .4), ("gBAR", .3),
             ("gE2mLO", .5)], -0.10))
S.append(sc([("gSPt", .55), ("gSPm", .35), ("gBAR", .45), ("gTX", .3)],  # barred spiral
            [("gAXIS", .6), ("gE2mHI", .5), ("gE2xHI", .4), ("gD1m", .3),
             ("gIRR", .3)], 0.10))
S.append(sc([("gSPt", .85), ("gSPm", .4), ("gTX", .3)],          # tight spiral
            [("gSPl", .45), ("gBAR", .45), ("gAXIS", .5), ("gE2mHI", .35),
             ("gDBL", .3)], 0.10))
S.append(sc([("gSPl", .95), ("gSPm", .4), ("gTX", .3), ("gEXTl", .2)],  # loose spiral
            [("gBAR", .35), ("gAXIS", .5), ("gIRR", .35)], 0.10))
S.append(sc([("gAXIS", 1.0), ("gTHIN", .6), ("gE2xHI", .45), ("gNEL", .4),
             ("gE2mHI", .35), ("gFLAT", .3), ("gE2oHI", .3), ("gTX", .25)],  # edge-on
            [("gCONCv", .5), ("gCONCc", .4), ("gRED", .3), ("gBAR", .4),
             ("gSPt", .3)], -0.50))
S.append(sc([("gAXIS", 1.0), ("gCONCc", .45), ("gCONCv", .35), ("gCONC", .35),
             ("gRED", .3), ("gE2xHI", .35), ("gE2mHI", .3), ("gFLAT", .3)],  # edge-on+bulge
            [("gTHIN", .45), ("gBAR", .35), ("gSPt", .3),
             ("gCONClo", .3)], -0.55))
sW = np.stack([r for r, b in S]); sb = np.array([b for r, b in S])
dense("d8", "d7a", 10, sW, sb)                          # class scores, shape (10,)

# ------------------------------------------------ output
def main(outdir):
    d = Path(outdir); d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "weights.npz", **W)
    with open(d / "network.json", "w") as f:
        json.dump({"layers": L}, f)
    tot = sum(int(v.size) for v in W.values())
    print("layers:", len(L), " weights:", tot)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
