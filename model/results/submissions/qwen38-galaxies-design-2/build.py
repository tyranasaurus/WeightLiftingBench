"""Build a hand-designed galaxy-morphology classifier network + weights.

Everything here comes from formulas and general astronomical knowledge of the ten
morphology classes (smooth round / intermediate / cigar, barred & unbarred spirals,
edge-on with/without bulge, disturbed, merging): a Gaussian/DoG pyramid, regional
pooling grids, second-moment shape descriptors (linear functionals of the pooled
grids), and a hand-written rule table (semantic features -> concepts -> class
scores).  No data is read; no fitting is performed.

Usage: python3 build.py OUT_DIR
"""
import json
import sys
import numpy as np
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

# ----------------------------------------------------------------------------
# kernel helpers (pure formulas)
# ----------------------------------------------------------------------------

def g1(sig, k):
    x = np.arange(k) - k // 2
    g = np.exp(-0.5 * (x / sig) ** 2)
    return g / g.sum()

def dg1(sig, k):
    x = np.arange(k) - k // 2
    g = -x / sig ** 2 * np.exp(-0.5 * (x / sig) ** 2)
    return g / g[g > 0].sum()

def sep(a, b):
    return np.outer(a, b)

K5, K7 = 5, 7
BIN5 = sep(g1(1.12, K5), g1(1.12, K5))
BIN7 = sep(g1(1.12, K7), g1(1.12, K7))
BIN3 = sep(g1(0.72, 3), g1(0.72, 3))
DGX = sep(dg1(1.3, K7), g1(1.3, K7))
DGY = sep(g1(1.3, K7), dg1(1.3, K7))
D7 = np.zeros((K7, K7)); D7[K7 // 2, K7 // 2] = 1.0
BP5 = D7 - BIN7
D3 = np.zeros((3, 3)); D3[1, 1] = 1.0
BP3 = D3 - BIN3

layers = []
W = {}

def conv(name, inp, out, k, w, b=None, stride=1, padding=0):
    layers.append({"name": name, "op": "conv", "inputs": [inp], "out": out,
                   "kernel": [k, k], "stride": stride, "padding": padding})
    w = np.asarray(w, np.float64)
    assert w.shape == (out, w.shape[1], k, k), (name, w.shape)
    W[name + ".W"] = w
    W[name + ".b"] = np.zeros(out) if b is None else np.asarray(b, np.float64)

def pool(name, inp, op, size, stride):
    layers.append({"name": name, "op": op, "inputs": [inp], "size": [size, size],
                   "stride": [stride, stride]})

def simple(name, op, inp):
    layers.append({"name": name, "op": op, "inputs": [inp]})

# --- stage 0 -----------------------------------------------------------------
simple("nr", "norm", "image")
conv("lum", "nr", 1, 1, np.array([1 / 3.0, 1 / 3.0, 1 / 3.0]).reshape(1, 3, 1, 1))
conv("red", "nr", 1, 1, np.array([0.5, 0.0, -0.5]).reshape(1, 3, 1, 1))
conv("redn", "red", 1, 1, np.array([-1.0]).reshape(1, 1, 1, 1))
conv("m0", "lum", 1, K5, BIN5.reshape(1, 1, K5, K5), padding=2)
simple("n", "norm", "m0")
simple("rn", "relu", "n")

# --- pyramid -----------------------------------------------------------------
prev = "n"
for lev in range(4):
    t = "t%d" % lev
    wk = np.zeros((3, 1, K7, K7))
    wk[0, 0] = BIN7
    wk[1, 0] = DGX
    wk[2, 0] = DGY
    conv(t, prev, 3, K7, wk, padding=3)
    mx = np.zeros((8, 3, 1, 1))
    mx[0, 1] = 1; mx[1, 2] = 1; mx[2, 1] = -1; mx[3, 2] = -1
    c = 0.70710678
    mx[4, 1] = c; mx[4, 2] = c; mx[5, 1] = -c; mx[5, 2] = -c
    mx[6, 1] = c; mx[6, 2] = -c; mx[7, 1] = -c; mx[7, 2] = c
    conv("mix%d" % lev, t, 8, 1, mx)
    simple("r%d" % lev, "relu", "mix%d" % lev)
    ak = np.zeros((4, 8, 1, 1))
    ak[0, :, 0, 0] = [-1, 1, -1, 1, 0, 0, 0, 0]
    ak[1, :, 0, 0] = [1, -1, 1, -1, 0, 0, 0, 0]
    ak[2, :, 0, 0] = [0, 0, 0, 0, -1, -1, 1, 1]
    ak[3, :, 0, 0] = [0, 0, 0, 0, 1, 1, -1, -1]
    conv("an%d" % lev, "r%d" % lev, 4, 1, ak)
    simple("ra%d" % lev, "relu", "an%d" % lev)
    conv("bp%d" % lev, prev, 1, K7, BP5.reshape(1, 1, K7, K7), padding=3)
    simple("rb%d" % lev, "relu", "bp%d" % lev)
    conv("s%d" % lev, t, 1, 1, np.array([1.0, 0, 0]).reshape(1, 3, 1, 1))
    pool("m%d" % (lev + 1), "s%d" % lev, "avgpool", 2, 2)
    prev = "m%d" % (lev + 1)
conv("bp4", prev, 1, 3, BP3.reshape(1, 1, 3, 3), padding=1)
simple("rb4", "relu", "bp4")
# star-safe, thin-localised, noise-gated body map for shape moments:
# fine+mid band-pass, minus a star gate (fine band-pass peaks), minus noise floor
pool("rb1d", "rb1", "avgpool", 2, 2)
pool("rb0m", "rb0", "maxpool", 4, 4)
conv("rb0mn", "rb0m", 1, 1, np.array([[-1.2]]).reshape(1, 1, 1, 1))
layers.append({"name": "mBpre", "op": "add",
               "inputs": ["rb1d", "rb2", "rb0mn"]})
conv("mBpre2", "mBpre", 1, 1, np.array([[1.0]]).reshape(1, 1, 1, 1), b=[-0.10])
simple("mB", "relu", "mBpre2")

# ----------------------------------------------------------------------------
# pooling into grids 16x16, 8x8, 4x4, 2x2 -> one canvas tensor (CTOT, 20, 20)
# ----------------------------------------------------------------------------
SZMAP = {"n": 128, "rn": 128, "red": 128, "redn": 128,
         "rb0": 128, "ra0": 128,
         "rb1": 64, "ra1": 64,
         "rb2": 32, "ra2": 32, "m2": 32, "rb1d": 32, "rb0m": 32, "mB": 32,
         "an0": 128, "an1": 64,
         "rb3": 16, "ra3": 16, "m3": 16,
         "rb4": 8, "m4": 8}

P16 = [("mB16", "mB", "avgpool", 2),
       ("b116", "rb1", "avgpool", 4),
       ("b216", "rb2", "avgpool", 2)]
P8 = [("n88", "n", "avgpool", 16),
      ("b088", "rb0", "avgpool", 16),
      ("b188", "rb1", "avgpool", 8),
      ("b288", "rb2", "avgpool", 4),
      ("b388", "rb3", "avgpool", 2),
      ("m288", "m2", "avgpool", 4),
      ("m388", "m3", "avgpool", 2),
      ("r88", "red", "avgpool", 16),
      ("mB88", "mB", "avgpool", 4)]
P4 = [("n44mx", "n", "maxpool", 32),
      ("b044mx", "rb0", "maxpool", 32),
      ("b144mx", "rb1", "maxpool", 16),
      ("b244", "rb2", "avgpool", 8),
      ("b244mx", "rb2", "maxpool", 8),
      ("b344mx", "rb3", "maxpool", 4),
      ("b344mn", "rb3", "avgpool", 4),
      ("m244mx", "m2", "maxpool", 8),
      ("m244mn", "m2", "avgpool", 8),
      ("m344mx", "m3", "maxpool", 4),
      ("m444", "m4", "avgpool", 2),
      ("r44mx", "red", "maxpool", 32),
      ("rn44mx", "redn", "maxpool", 32)]
P2 = [("n22mx", "n", "maxpool", 64),
      ("n22pos", "rn", "avgpool", 64),
      ("b022mx", "rb0", "maxpool", 64),
      ("b022mn", "rb0", "avgpool", 64),
      ("b122mx", "rb1", "maxpool", 32),
      ("b122mn", "rb1", "avgpool", 32),
      ("b222mx", "rb2", "maxpool", 16),
      ("b222mn", "rb2", "avgpool", 16),
      ("b322mn", "rb3", "avgpool", 8),
      ("b422mn", "rb4", "avgpool", 4),
      ("r22mn", "red", "avgpool", 64),
      ("a022", "an0", "avgpool", 64),
      ("a122", "an1", "avgpool", 32)]

chan = {}
c = 0
for nm, _, _, _ in P16:
    chan[nm] = c; c += 1
C16 = c
for nm, _, _, _ in P8:
    chan[nm] = c; c += 1
C8 = c
for nm, _, _, _ in P4:
    chan[nm] = c; c += 1
C4 = c
for nm, _, _, _ in P2:
    chan[nm] = c; c += 4 if nm[0] == "a" else 1
CTOT = c

OFF16, OFF8, OFF4, OFF2 = 2, 6, 8, 9
CAN = 20

def pool_and_slot(specs, tag, nchan_of):
    adds = []
    for (nm, src, op, win) in specs:
        pool("p_" + nm, src, op, win, win)
        nc = nchan_of(nm)
        h = SZMAP[src] // win
        p = (CAN - h) // 2
        wslot = np.zeros((CTOT, nc, 1, 1))
        for j in range(nc):
            wslot[chan[nm] + j, j, 0, 0] = 1.0
        conv("c_" + nm, "p_" + nm, CTOT, 1, wslot, padding=p)
        adds.append("c_" + nm)
    layers.append({"name": "add" + tag, "op": "add", "inputs": adds})
    return "add" + tag

a16 = pool_and_slot(P16, "16", lambda nm: 1)
a8 = pool_and_slot(P8, "8", lambda nm: 1)
a4 = pool_and_slot(P4, "4", lambda nm: 1)
a2 = pool_and_slot(P2, "2", lambda nm: 4 if nm[0] == "a" else 1)

emb = []
for gi, (src, lo, hi) in enumerate([(a16, 0, C16), (a8, C16, C8),
                                    (a4, C8, C4), (a2, C4, CTOT)]):
    we = np.zeros((CTOT, CTOT, 1, 1))
    for j in range(lo, hi):
        we[j, j, 0, 0] = 1.0
    conv("emb%d" % gi, src, CTOT, 1, we)
    emb.append("emb%d" % gi)
layers.append({"name": "feat", "op": "add", "inputs": emb})
layers.append({"name": "flat", "op": "reshape", "inputs": ["feat"],
               "shape": [CTOT * CAN * CAN]})

# ----------------------------------------------------------------------------
# canvas indexing
# ----------------------------------------------------------------------------
def idx(ch, r, q):
    return ch * CAN * CAN + r * CAN + q

def g16(nm, i, j): return idx(chan[nm], OFF16 + i, OFF16 + j)
def g8(nm, i, j):  return idx(chan[nm], OFF8 + i, OFF8 + j)
def g4(nm, i, j):  return idx(chan[nm], OFF4 + i, OFF4 + j)
def g2(nm, i, j):  return idx(chan[nm], OFF2 + i, OFF2 + j)
def g2ch(nm, k, i, j): return idx(chan[nm] + k, OFF2 + i, OFF2 + j)

C4_8 = [(3, 3), (3, 4), (4, 3), (4, 4)]
R1_8 = [(i, j) for i in range(2, 6) for j in range(2, 6) if (i, j) not in C4_8]
R2_8 = [(i, j) for i in range(1, 7) for j in range(1, 7)
        if not (2 <= i <= 5 and 2 <= j <= 5)]
OUT8 = [(i, j) for i in range(8) for j in range(8)
        if not (1 <= i <= 6 and 1 <= j <= 6)]
C4_4 = [(1, 1), (1, 2), (2, 1), (2, 2)]
R12_4 = [(i, j) for i in range(4) for j in range(4) if (i, j) not in C4_4]
ALL2 = [(0, 0), (0, 1), (1, 0), (1, 1)]

def M(nm, cells, gf, w=1.0):
    return [(gf(nm, i, j), w / len(cells)) for (i, j) in cells]

def M2ch(nm, k, w=1.0):
    return [(g2ch(nm, k, i, j), w / 4.0) for (i, j) in ALL2]

# ----------------------------------------------------------------------------
# dense 1: semantic features
# ----------------------------------------------------------------------------
SEM = []

def sem(name, bias, terms):
    SEM.append((name, bias, terms))

# flux / profile
sem("f_pos", 0.0, M2ch("n22pos", 0))
sem("f_pk", 0.0, M2ch("n22mx", 0))
sem("c2", 0.0, M("n88", C4_8, g8))
sem("c4", 0.0, M("m288", C4_8, g8))
sem("c8", 0.0, M("m388", C4_8, g8))
sem("c16", 0.0, M("m444", C4_4, g4))
sem("r1l", 0.0, M("n88", R1_8, g8))
sem("r2l", 0.0, M("n88", R2_8, g8))
sem("cc1", 0.0, [(None, 1.0, "c2"), (None, -2.2, "r1l")])
sem("ccpk", 0.0, [(None, 1.0, "f_pk"), (None, -2.0, "c2")])
sem("big", 0.0, [(None, 1.0, "c16"), (None, -0.40, "c8")])
sem("small", 0.0, [(None, 1.0, "c8"), (None, -3.0, "c16")])
# texture (spiral arms / knots)
sem("t1c", 0.0, M("b188", C4_8, g8))
sem("t1r1", 0.0, M("b188", R1_8, g8))
sem("t1r2", 0.0, M("b188", R2_8, g8))
sem("t1o", 0.0, M("b188", OUT8, g8))
sem("t2r1", 0.0, M("b288", R1_8, g8))
sem("t2r2", 0.0, M("b288", R2_8, g8))
sem("t0r", 0.0, M("b088", R1_8 + R2_8, g8))
sem("t0c", 0.0, M("b088", C4_8, g8))
sem("t1tot", 0.0, M2ch("b122mn", 0))
sem("t2tot", 0.0, M2ch("b222mn", 0))
sem("sp", -0.024, [(None, 3.0, "t1r1"), (None, 3.0, "t1r2"), (None, 2.1, "t2r1"),
                   (None, 2.1, "t2r2"), (None, 2.4, "t16r1"), (None, 2.4, "t16r2"),
                   (None, 3.0, "t16r0"), (None, -0.9, "t1c"), (None, -0.6, "t16c")])
sem("spF", -0.06, [(None, 1.0, "t1r2"), (None, 1.0, "t1o"), (None, 1.0, "t2r2"),
                   (None, -0.9, "t1r1")])
sem("spN", -0.03, [(None, 1.0, "t1r1"), (None, 1.0, "t2r1"), (None, -1.1, "t1r2"),
                   (None, -1.1, "t2r2")])
sem("kno", 0.0, [(None, 1.0, "t0r"), (None, -0.6, "t0c")])
# second moments of the star-safe body map (16x16 and 8x8 grids)
def moment_rows(tag, nm, gf, n, cent=None):
    if cent is None:
        cells = [(i, j) for i in range(n) for j in range(n)]
        cc = (n - 1) / 2.0
    else:
        cells = [(i, j) for i in range(cent[0], cent[1])
                 for j in range(cent[0], cent[1])]
        cc = (cent[0] + cent[1] - 1) / 2.0
    wxx = [(gf(nm, i, j), (j - cc) ** 2) for (i, j) in cells]
    wyy = [(gf(nm, i, j), (i - cc) ** 2) for (i, j) in cells]
    wxy = [(gf(nm, i, j), (i - cc) * (j - cc)) for (i, j) in cells]
    wt = [(gf(nm, i, j), (i - cc) ** 2 + (j - cc) ** 2) for (i, j) in cells]
    wf = [(gf(nm, i, j), 1.0) for (i, j) in cells]
    sem("Mxx" + tag, 0.0, wxx)
    sem("Myy" + tag, 0.0, wyy)
    sem("dMp" + tag, 0.0, wxx + [(a, -w) for a, w in wyy])
    sem("dMn" + tag, 0.0, wyy + [(a, -w) for a, w in wxx])
    sem("cMp" + tag, 0.0, wxy)
    sem("cMn" + tag, 0.0, [(a, -w) for a, w in wxy])
    sem("T" + tag, 0.0, wt)
    sem("F" + tag, 0.0, wf)

moment_rows("16", "mB16", g16, 16)
moment_rows("8", "mB88", g8, 8)
moment_rows("16c", "mB16", g16, 16, cent=(4, 12))
# stars / spikes
sem("shp", 0.0, M2ch("b022mx", 0, 1.0) + M2ch("b222mx", 0, -2.5))
for lv in (0, 1):
    for k, sfx in enumerate(["h", "v", "d", "e"]):
        sem("a%d%s" % (lv, sfx), 0.0, M2ch("a%d22" % lv, k))
sem("h_dv1", 0.0, [(None, 1.0, "a1h"), (None, -1.0, "a1v")])
sem("h_dv1n", 0.0, [(None, -1.0, "a1h"), (None, 1.0, "a1v")])
sem("h_d41", 0.0, [(None, 1.0, "a1d"), (None, -1.0, "a1e")])
sem("h_d41n", 0.0, [(None, -1.0, "a1d"), (None, 1.0, "a1e")])
sem("min1", 0.0, [(None, 1.0, "a1h"), (None, 1.0, "a1v"), (None, -1.0, "h_dv1"),
                  (None, -1.0, "h_dv1n")])
sem("min45", 0.0, [(None, 1.0, "a1d"), (None, 1.0, "a1e"), (None, -1.0, "h_d41"),
                   (None, -1.0, "h_d41n")])
# redness
sem("rc", 0.0, M("r88", C4_8, g8))
sem("rr", 0.0, M("r88", R1_8 + R2_8, g8))
# bulge: central mid-band-pass excess over the ring (per-cell soft peaks)
for (i, j) in C4_8:
    sem("bc_%d%d" % (i, j), -0.6, [(g8("b288", i, j), 1.0)])
for (i, j) in R1_8:
    sem("br_%d%d" % (i, j), -0.6, [(g8("b288", i, j), 1.0)])
# asymmetry on the coarse luminance grid (32px cells)
pairs4 = []
for i in range(4):
    for j in range(4):
        pi, pj = 3 - i, 3 - j
        if (i, j) < (pi, pj):
            pairs4.append(((i, j), (pi, pj)))
for (a, b) in pairs4:
    sem("as4_%d%d" % (a[0], a[1]), -0.6, [(g4("m244mn", a[0], a[1]), 1.0),
                                      (g4("m244mn", b[0], b[1]), -1.0)])
# tails / large-scale debris off centre
sem("tailsr", 0.0, M("b388", R2_8 + OUT8, g8))
sem("tails", -0.12, [(None, 1.0, "tailsr"), (None, -0.25, "t3c")])
sem("t3c", 0.0, M("b388", C4_8, g8))
# fine-ring texture (8px cells) for small galaxies
D2_16 = [(i, j) for i in range(16) for j in range(16)
         if max(abs(i - 7.5), abs(j - 7.5)) <= 3.5
         and max(abs(i - 7.5), abs(j - 7.5)) >= 1.5]
D3_16 = [(i, j) for i in range(16) for j in range(16)
         if max(abs(i - 7.5), abs(j - 7.5)) <= 5.5
         and max(abs(i - 7.5), abs(j - 7.5)) >= 3.5]
C1_16 = [(i, j) for i in range(16) for j in range(16)
         if max(abs(i - 7.5), abs(j - 7.5)) <= 1.5]
D1_16 = [(i, j) for i in range(16) for j in range(16)
         if 0.5 <= max(abs(i - 7.5), abs(j - 7.5)) < 1.5]
sem("t16r1", 0.0, M("b116", D2_16, g16))
sem("t16r2", 0.0, M("b116", D3_16, g16))
sem("t16c", 0.0, M("b116", C1_16, g16))
sem("t16r0", 0.0, M("b116", D1_16, g16))
# wide companions (merging): coarse band-pass peaks off centre
for (i, j) in R12_4:
    sem("p3_%d%d" % (i, j), -0.9, [(g4("b344mx", i, j), 1.0)])
# close double nucleus: diagonal imbalance of mid-band-pass peaks in the centre
for (nm, tag) in [("b244mx", "c2"), ("b144mx", "c1")]:
    for (i, j) in C4_4:
        w = 1.0 if (i, j) in [(1, 1), (2, 2)] else -1.0
        sem("dc%s_%d%d" % (tag, i, j), 0.0, [(g4(nm, i, j), w)])
        sem("dc%s_%d%dn" % (tag, i, j), 0.0, [(g4(nm, i, j), -w)])

sem("n_a1h", 0.0, [(None, -1.0, "a1h")])
sem("n_a1d", 0.0, [(None, -1.0, "a1d")])
semidx = {nm: k for k, (nm, _, _) in enumerate(SEM)}
D1 = len(SEM)
w1 = np.zeros((D1, CTOT * CAN * CAN))
b1 = np.zeros(D1)
for k, (nm, bias, terms) in enumerate(SEM):
    b1[k] = bias
    for t in terms:
        if t[0] is None:
            w1[k, semidx[t[2]]] += t[1]
        else:
            w1[k, t[0]] += t[1]
layers.append({"name": "d1", "op": "dense", "inputs": ["flat"], "out": D1})
W["d1.W"], W["d1.b"] = w1, b1
simple("a_sem", "relu", "d1")

# ----------------------------------------------------------------------------
# dense 2: concepts
# ----------------------------------------------------------------------------
CON = []

def con(name, bias, terms):
    CON.append((name, bias, terms))

for nm in ["f_pos", "f_pk", "c2", "c4", "c8", "c16", "cc1", "ccpk", "big",
           "small", "t1c", "t1r1", "t1r2", "t1o", "t2r1", "t2r2", "t0r",
           "t1tot", "t2tot", "sp", "spF", "spN", "kno", "shp", "rc", "rr",
           "tails", "Mxx16", "Myy16", "T16", "F16", "Mxx8", "Myy8", "T8", "F8",
           "T16c", "F16c", "min1", "min45"]:
    con(nm, 0.0, [(nm, 1.0)])
# elongation / roundness / thinness from moments
def sc(terms, f):
    return [(n, w * f) for n, w in terms]

def Eterms(tag, w=1.0):
    return [("dMp" + tag, w), ("dMn" + tag, w), ("cMp" + tag, 2 * w),
            ("cMn" + tag, 2 * w)]

def ELterms(tag, k):
    return Eterms(tag) + [("T" + tag, -k)]

def THterms(tag, k):
    """thin: minor-axis second moment (T-E)/2 below k*F."""
    return [("F" + tag, 2 * k), ("T" + tag, -1.0), ("dMp" + tag, 1.0),
            ("dMn" + tag, 1.0), ("cMp" + tag, 2.0), ("cMn" + tag, 2.0)]

def TKterms(tag, k):
    """thick: minor-axis second moment above k*F."""
    return [("T" + tag, 1.0), ("dMp" + tag, -1.0), ("dMn" + tag, -1.0),
            ("cMp" + tag, -2.0), ("cMn" + tag, -2.0), ("F" + tag, -2 * k)]

for tag in ("16", "8", "16c"):
    con("E" + tag, 0.0, Eterms(tag))
for tag, ks in [("16", (0.2, 0.4, 0.6)), ("8", (0.2, 0.4, 0.6)),
                ("16c", (0.4, 0.6))]:
    for k in ks:
        con("EL%s_%d" % (tag, int(k * 10)), 0.0, ELterms(tag, k))
    for k in (0.25, 0.40):
        con("TH%s_%d" % (tag, int(k * 100)), 0.0, THterms(tag, k))
    for k in (0.30, 0.45):
        con("TK%s_%d" % (tag, int(k * 100)), 0.0, TKterms(tag, k))
def RDterms(tag):
    return [("T" + tag, 0.25)] + Eterms(tag, -1.0)

for tag in ("16", "8"):
    con("RD" + tag, 0.0, RDterms(tag))
con("BARm", 0.0, sc(Eterms("16c"), 0.04) + [("T16c", -0.018)]
    + sc(Eterms("16"), -0.02) + [("T16", 0.009)])
con("EOraw", 0.0, sc(THterms("16", 0.25), 0.3) + sc(THterms("8", 0.25), 0.2)
    + sc(ELterms("16", 0.4), 0.04) + sc(ELterms("8", 0.4), 0.02))
con("CIGc", 0.0, sc(ELterms("16", 0.4), 0.04) + sc(ELterms("8", 0.4), 0.04)
    + sc(TKterms("16", 0.30), 0.3) + sc(TKterms("8", 0.30), 0.2)
    + sc(RDterms("16"), -0.5) + sc(RDterms("8"), -0.3))
con("BULGE", -0.2, [("bc_%d%d" % (i, j), 1.0) for (i, j) in C4_8]
    + [("br_%d%d" % (i, j), -1.0) for (i, j) in R1_8])
con("ASYM4", -1.2, [("as4_%d%d" % (a[0], a[1]), 1.0) for (a, b) in pairs4])
con("MGcnt", -1.2, [("p3_%d%d" % (i, j), 1.0) for (i, j) in R12_4])
con("MGclose", -0.5, [("dc%s_%d%d" % (t, i, j), 1.0) for (t, (i, j)) in
                      [("c2", p) for p in C4_4] + [("c1", p) for p in C4_4]]
    + [("dc%s_%d%dn" % (t, i, j), -1.0) for (t, (i, j)) in
       [("c2", p) for p in C4_4] + [("c1", p) for p in C4_4]]
    + [("F16c", -0.15)])
con("SMOOTH", -0.02, [("f_pos", 0.8), ("t1tot", -1.4), ("t2tot", -1.4),
                    ("tailsr", -3.0)])
con("crossU", -0.5, [("min1", 1.0), ("min45", 1.0)])
con("starU", -0.6, [("shp", 1.0)])
con("REDC", -0.05, [("rr", 1.0)])
con("BLUEC", -0.03, [("rr", -1.0)])
con("TXTc", 0.0, [("t1c", 1.0)])
for _nm, _t in [("EOraw2", 0.035), ("EL2", 0.02), ("ELbig", 0.06),
              ("ELvbig", 0.12)]:
    con(_nm, 0.0, [("a1h", 1.0), ("h_a1h", 1.0), ("a1d", 1.0), ("h_a1d", 1.0),
                   ("F16", 0.0)]) if False else con(
        _nm, -4.0 * _t, [("a1h", 1.0), ("n_a1h", 1.0), ("a1d", 1.0),
                         ("n_a1d", 1.0)])


conidx = {nm: k for k, (nm, _, _) in enumerate(CON)}
D2 = len(CON)
w2 = np.zeros((D2, D1))
b2 = np.zeros(D2)
for k, (nm, bias, terms) in enumerate(CON):
    b2[k] = bias
    for (snm, w) in terms:
        w2[k, semidx[snm]] += w
layers.append({"name": "d2", "op": "dense", "inputs": ["a_sem"], "out": D2})
W["d2.W"], W["d2.b"] = w2, b2
simple("a_con", "relu", "d2")

# ----------------------------------------------------------------------------
# dense 3: class scores
# ----------------------------------------------------------------------------
CLS = [
    (-0.2, [("ASYM4", 10.0), ("tails", 25.0), ("MGclose", 2.0), ("MGcnt", -6.0),
            ("SMOOTH", -40.0), ("sp", -20.0), ("EOraw2", -2.0), ("ELvbig", -8.0)]),
    (-0.1, [("MGcnt", 6.0), ("MGclose", 4.0), ("ASYM4", 4.0), ("SMOOTH", -40.0),
            ("EOraw2", -2.0), ("sp", -10.0), ("ELvbig", -8.0)]),
    (0.3, [("SMOOTH", 40.0), ("EL2", -3.0), ("ELbig", -6.0), ("ELvbig", -8.0),
           ("EOraw2", -5.0), ("sp", -80.0), ("REDC", 20.0), ("crossU", -4.0)]),
    (0.1, [("SMOOTH", 40.0), ("EL2", 4.0), ("ELbig", -6.0), ("ELvbig", -8.0),
           ("EOraw2", -5.0), ("sp", -80.0), ("REDC", 14.0), ("crossU", -4.0)]),
    (-0.4, [("SMOOTH", 40.0), ("ELbig", 8.0), ("ELvbig", -8.0), ("EOraw2", -8.0),
            ("sp", -80.0), ("REDC", 16.0), ("crossU", -4.0)]),
    (0.0, [("sp", 80.0), ("SMOOTH", -30.0), ("EOraw2", -3.0), ("MGcnt", -6.0),
           ("ccpk", 1.0), ("spN", 10.0), ("ELvbig", -8.0)]),
    (0.0, [("sp", 80.0), ("spN", 40.0), ("spF", -30.0), ("BLUEC", 30.0),
           ("ccpk", 0.5), ("EOraw2", -3.0), ("MGcnt", -6.0), ("ELvbig", -8.0)]),
    (0.2, [("sp", 80.0), ("spF", 40.0), ("spN", -30.0), ("BLUEC", 35.0),
           ("kno", 20.0), ("EOraw2", -3.0), ("MGcnt", -6.0), ("ELvbig", -8.0)]),
    (0.0, [("EOraw2", 5.0), ("ELvbig", 10.0), ("BULGE", -3.0), ("BLUEC", 10.0),
           ("sp", -30.0), ("SMOOTH", -20.0), ("crossU", -8.0), ("starU", -6.0)]),
    (0.1, [("EOraw2", 5.0), ("ELvbig", 10.0), ("BULGE", 3.0), ("sp", -30.0),
           ("SMOOTH", -20.0), ("crossU", -8.0), ("starU", -6.0)]),
]
w3 = np.zeros((10, D2))
b3 = np.zeros(10)
for k, (bias, terms) in enumerate(CLS):
    b3[k] = bias
    for (cnm, w) in terms:
        w3[k, conidx[cnm]] += w
layers.append({"name": "scores", "op": "dense", "inputs": ["a_con"], "out": 10})
W["scores.W"], W["scores.b"] = w3, b3

# ----------------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "network.json").open("w") as f:
    json.dump({"layers": layers}, f)
np.savez(OUT / "weights.npz", **{k: v.astype(np.float32) for k, v in W.items()})
print("layers:", len(layers), "weights:", sum(v.size for v in W.values()),
      "dense1:", D1, "dense2:", D2, "channels:", CTOT)
