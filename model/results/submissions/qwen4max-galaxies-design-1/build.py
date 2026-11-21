"""Hand-designed galaxy morphology network.

Every number here comes from analytic formulae (Gaussian weights, ellipses,
logarithmic spirals, Fourier modes, bars, double nuclei) and from hand-chosen
thresholds.  No training data is read.

Architecture
------------
1. Two 1x1 convs make a red-weighted luminance map and a blue-weighted map.
2. Three "functional" convs (kernel 128, stride 128) evaluate a bank of
   hand-designed linear measurements at the image centre: compensated radial
   apertures, traceless quadrupoles at four scales, angular Fourier modes in
   three annuli, logarithmic-spiral matched filters at four pitch angles,
   double-nucleus matched filters, and three-band aperture photometry.
3. Orientation is marginalised with max-pooling over the orientation block.
4. Brightness is divided out with a norm trick: for a pair (a, b) the group
   [a, -a, b, -b] has zero mean and std sqrt((a^2+b^2)/2), so after norm the
   first entry is sqrt(2)*(a/b)/sqrt(1+(a/b)^2) -- a monotone, bounded encoding
   of the ratio a/b that needs no division op.
5. A hand-written fuzzy rule table (clipped ramps, min = AND, max = OR, built
   from dense + relu + add) turns ~30 morphological features into 10 scores.
"""
import sys, json, math
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
LAYERS = []
WEIGHTS = {}


def layer(name, op, inputs, **kw):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(kw)
    LAYERS.append(d)
    return name


def conv(name, inp, W, b, stride=1, padding=0):
    W = np.asarray(W, np.float64)
    b = np.asarray(b, np.float64)
    o, c, kh, kw = W.shape
    WEIGHTS[name + ".W"] = W.astype(np.float32)
    WEIGHTS[name + ".b"] = b.astype(np.float32)
    return layer(name, "conv", [inp], out=o, kernel=[kh, kw], stride=stride, padding=padding)


def dense(name, inp, W, b):
    W = np.asarray(W, np.float64)
    b = np.asarray(b, np.float64)
    WEIGHTS[name + ".W"] = W.astype(np.float32)
    WEIGHTS[name + ".b"] = b.astype(np.float32)
    return layer(name, "dense", [inp], out=W.shape[0])


def relu(name, inp):
    return layer(name, "relu", [inp])


def add(name, *ins):
    return layer(name, "add", list(ins))


def reshape(name, inp, shape):
    return layer(name, "reshape", [inp], shape=list(shape))


def norm(name, inp):
    return layer(name, "norm", [inp])


def pool(name, inp, size, stride, kind="max"):
    return layer(name, kind + "pool", [inp], size=list(size), stride=list(stride))


# ---------------------------------------------------------------- grid
N = 128
CEN = (N - 1) / 2.0
yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
X = xx - CEN
Y = yy - CEN
R = np.hypot(X, Y)
TH = np.arctan2(Y, X)


def zs(k):
    """make a kernel exactly zero-sum so a constant sky contributes nothing"""
    k = np.asarray(k, np.float64)
    return k - k.mean()


def gauss(s):
    return np.exp(-0.5 * (R / s) ** 2)


def annulus(r0, frac=0.45):
    """smooth radial window peaked at r0 with fractional width frac"""
    w = np.exp(-0.5 * ((R - r0) / (frac * r0)) ** 2)
    return w


def orient_block(fun, n_or, period=math.pi):
    """n_or orientations spanning `period` (the symmetry period of the pattern)"""
    return [fun(2 * math.pi * j / (n_or * (2 * math.pi / period)) if False else period * j / n_or)
            for j in range(n_or)]


# ---------------------------------------------------------------- 1. luminance
# channel order of the input is R, G, B (z, r, g rendered as red, green, blue)
conv("lum", "image", np.array([[[[0.45]], [[0.40]], [[0.15]]]]), [0.0])
conv("blu", "image", np.array([[[[0.15]], [[0.35]], [[0.50]]]]), [0.0])

# ---------------------------------------------------------------- 2. measurement bank
# channel layout of fL is recorded here so later layers can address it
CH = {}
KL = []


def put(tag, kern):
    CH[tag] = len(KL)
    KL.append(np.asarray(kern, np.float64))


def put_block(tag, kerns):
    CH[tag] = (len(KL), len(kerns))
    for k in kerns:
        KL.append(np.asarray(k, np.float64))


# --- 2a. compensated radial apertures (surface brightness inside scale s)
PROF_S = [2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 27.0, 40.0]
BG = np.exp(-0.5 * ((R - 52.0) / 6.0) ** 2)   # sky annulus, away from the galaxy
for s in PROF_S:
    g = gauss(s)
    put("prof%g" % s, zs(g - (g.sum() / BG.sum()) * BG))
_g10 = gauss(10.0)
put("sb", zs(_g10 - (_g10.sum() / BG.sum()) * BG) / _g10.sum())

# --- 2b. traceless quadrupole + trace moment at four scales
SHAPE_S = [3.0, 6.0, 11.0, 20.0]
NOR = 16
for s in SHAPE_S:
    w = gauss(s)
    qs = []
    for j in range(NOR):
        a = math.pi * j / NOR
        u = X * math.cos(a) + Y * math.sin(a)
        qs.append(zs(w * (u * u - 0.5 * R * R)) / (s * s))
    put_block("q%g" % s, qs)
    m = w * R * R
    put("m%g" % s, zs(m - (m.sum() / BG.sum()) * BG) / (s * s))

# --- 2c. angular Fourier modes in three annuli (lopsidedness, oval, boxy)
FOUR_R = [6.0, 12.0, 22.0]
FOUR_M = [1, 2, 4]
NFO = 8
for r0 in FOUR_R:
    w = annulus(r0)
    put("af%g" % r0, zs(w - (w.sum() / BG.sum()) * BG) / w.sum())
    for m in FOUR_M:
        ks = []
        for j in range(NFO):
            ph = (2 * math.pi / m) * j / NFO
            ks.append(zs(w * np.cos(m * (TH - ph))) / w.sum())
        put_block("f%d_%g" % (m, r0), ks)

# --- 2d. logarithmic-spiral matched filters (pitch angle = arm winding)
SPW = annulus(15.0, 0.62)
put("afsp", zs(SPW - (SPW.sum() / BG.sum()) * BG) / SPW.sum())
PITCH = [12.0, 22.0, 35.0, 90.0]
NSP = 16
LR = np.log(np.maximum(R, 1.0) / 10.0)
for p in PITCH:
    wind = 0.0 if p >= 89 else LR / math.tan(math.radians(p))
    ks = []
    for j in range(NSP):
        ph = math.pi * j / NSP
        ks.append(zs(SPW * np.cos(2 * (TH - wind - ph))) / SPW.sum())
    put_block("sp%g" % p, ks)

# --- 2e. double-nucleus matched filters (mergers)
SIG = 3.5
DBL_D = [8.0, 14.0, 22.0]


def blob(cx, cy, s=SIG):
    return np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / (s * s))


_gb = blob(0, 0, SIG)
put("sgn", zs(_gb - (_gb.sum() / BG.sum()) * BG) / _gb.sum())
for d in DBL_D:
    ss = math.sqrt(SIG * SIG + d * d)
    g1 = blob(0, 0, ss)
    put("sg%g" % d, zs(g1 - (g1.sum() / BG.sum()) * BG) / g1.sum())
    ks = []
    for j in range(8):
        a = math.pi * j / 8
        cx, cy = d * math.cos(a), d * math.sin(a)
        t = blob(cx, cy) + blob(-cx, -cy)
        ks.append(zs(t - (t.sum() / BG.sum()) * BG) / t.sum())
    put_block("pair%g" % d, ks)
    ks = []
    for j in range(12):
        a = 2 * math.pi * j / 12
        cx, cy = d * math.cos(a), d * math.sin(a)
        t = blob(0, 0) + blob(cx, cy)
        ks.append(zs(t - (t.sum() / BG.sum()) * BG) / t.sum())
    put_block("off%g" % d, ks)

# ---------------------------------------------------------------- build the convs
WL = np.stack(KL)[:, None, :, :]
conv("fL", "lum", WL, np.zeros(len(KL)), stride=N, padding=0)
NL = len(KL)

# --- 2f. three-band aperture photometry (colour of core / disk / outskirts)
KC = []
CC = {}
COLW = [("cc", gauss(4.0)), ("cm", annulus(10.0, 0.5)), ("co", annulus(21.0, 0.5))]
for tag, w in COLW:
    for b in range(3):
        CC["%s%d" % (tag, b)] = len(KC)
        k = np.zeros((3, N, N))
        k[b] = zs(w - (w.sum() / BG.sum()) * BG) / w.sum()
        KC.append(k)
conv("fC", "image", np.stack(KC), np.zeros(len(KC)), stride=N, padding=0)

# ---------------------------------------------------------------- 3. blue clumpiness
# difference-of-Gaussians high pass, half resolution
g1 = np.exp(-0.5 * (np.hypot(np.mgrid[0:9, 0:9][1] - 4, np.mgrid[0:9, 0:9][0] - 4) / 1.5) ** 2)
g2 = np.exp(-0.5 * (np.hypot(np.mgrid[0:9, 0:9][1] - 4, np.mgrid[0:9, 0:9][0] - 4) / 5.0) ** 2)
dog = (g1 / g1.sum() - g2 / g2.sum())[None, None]
conv("hp", "blu", dog, [0.0], stride=2, padding=4)
relu("hpp", "hp")
conv("nhp", "hp", np.array([[[[-1.0]]]]), [0.0])
relu("hpn", "nhp")
add("hpa", "hpp", "hpn")

# radial windows on the 64x64 grid (radius = half of the 128 radius)
y2, x2 = np.mgrid[0:64, 0:64].astype(np.float64)
R2 = np.hypot(x2 - 31.5, y2 - 31.5)
KW = []
for tag, w in [("cl_in", np.exp(-0.5 * (R2 / 6.0) ** 2)),
               ("cl_mid", np.exp(-0.5 * ((R2 - 12.0) / 5.0) ** 2)),
               ("cl_sky", np.exp(-0.5 * ((R2 - 26.0) / 4.0) ** 2))]:
    KW.append((w / w.sum())[None])
conv("clump", "hpa", np.stack(KW), np.zeros(3), stride=64, padding=0)

# ---------------------------------------------------------------- 4. 180 deg asymmetry
pool("a32", "lum", [4, 4], [4, 4], "avg")
P = np.zeros((1024, 1024))
for i in range(32):
    for j in range(32):
        P[(31 - i) * 32 + (31 - j), i * 32 + j] = 1.0
dense("flip", "a32", P, np.zeros(1024))
reshape("a32r", "flip", [1, 32, 32])
conv("na32", "a32", np.array([[[[-1.0]]]]), [0.0])
add("dif", "a32r", "na32")
conv("ndif", "dif", np.array([[[[-1.0]]]]), [0.0])
relu("rd1", "dif")
relu("rd2", "ndif")
add("adif", "rd1", "rd2")
y3, x3 = np.mgrid[0:32, 0:32].astype(np.float64)
R3 = np.hypot(x3 - 15.5, y3 - 15.5)
wa = (R3 < 10.0).astype(np.float64)
wb = ((R3 > 13.0) & (R3 < 15.6)).astype(np.float64)
ka = (wa / wa.sum() - wb / wb.sum())[None]
conv("asym", "adif", ka[None], [0.0], stride=32, padding=0)

# ---------------------------------------------------------------- 5. orientation max
SCAL = (["prof%g" % s for s in PROF_S] + ["sb"] + ["m%g" % s for s in SHAPE_S]
        + ["af%g" % r for r in FOUR_R] + ["afsp"] + ["sgn"] + ["sg%g" % d for d in DBL_D])
BLKS = (["q%g" % s for s in SHAPE_S]
        + ["f%d_%g" % (m, r) for r in FOUR_R for m in FOUR_M]
        + ["sp%g" % p for p in PITCH]
        + ["pair%g" % d for d in DBL_D] + ["off%g" % d for d in DBL_D])
NB = len(BLKS)
WB = np.zeros((16 * NB, NL))
bB = np.full(16 * NB, -1e6)
for i, t in enumerate(BLKS):
    st, cnt = CH[t]
    for j in range(cnt):
        WB[16 * i + j, st + j] = 1.0
        bB[16 * i + j] = 0.0
dense("dB", "fL", WB, bB)
reshape("dBr", "dB", [1, 1, 16 * NB])
pool("omv", "dBr", [1, 16], [1, 16], "max")

# ---------------------------------------------------------------- 6. ratio features
# each ratio is encoded as sqrt(2)*(a/b)/sqrt(1+(a/b)^2) via the [a,-a,b,-b] norm
RAT = []


def ratio(name, a, b):
    """a, b: lists of (source, tag, coefficient); source in s=scalar, b=blockmax,
    c=colour conv, k=clump conv, y=asym conv"""
    RAT.append((name, a, b))


for s in SHAPE_S:
    ratio("q2_%g" % s, [("s", "m%g" % s, 1.0), ("b", "q%g" % s, -2.0)],
          [("s", "m%g" % s, 1.0), ("b", "q%g" % s, 2.0)])
for r0 in FOUR_R:
    for m in FOUR_M:
        ratio("f%d_%g" % (m, r0), [("b", "f%d_%g" % (m, r0), 1.0)],
              [("s", "af%g" % r0, 1.0)])
for p in PITCH:
    ratio("sp%g" % p, [("b", "sp%g" % p, 1.0)], [("s", "afsp", 1.0)])
for d in DBL_D:
    ratio("pair%g" % d, [("b", "pair%g" % d, 1.0), ("s", "sgn", -1.0)],
          [("s", "sgn", 1.0)])
    ratio("off%g" % d, [("b", "off%g" % d, 1.0), ("s", "sgn", -1.0)],
          [("s", "sgn", 1.0)])
for s1, s2 in [(2, 8), (3, 12), (5, 18), (8, 27), (18, 40)]:
    ratio("c%g_%g" % (s1, s2), [("s", "prof%g" % s1, 1.0)], [("s", "prof%g" % s2, 1.0)])
for tag, _ in COLW:
    ratio("col_" + tag, [("c", tag + "2", 1.0), ("c", tag + "0", -1.0)],
          [("c", tag + "2", 1.0), ("c", tag + "0", 1.0)])
ratio("cl_in", [("k", "cl_in", 1.0), ("k", "cl_sky", -1.0)], [("s", "sb", 1.0)])
ratio("cl_mid", [("k", "cl_mid", 1.0), ("k", "cl_sky", -1.0)], [("s", "sb", 1.0)])
ratio("asym", [("y", "asym", 1.0)], [("s", "sb", 1.0)])

K = len(RAT)
FI = {n: i for i, (n, _, _) in enumerate(RAT)}
WS = np.zeros((4 * K, NL))
WO = np.zeros((4 * K, NB))
WC = np.zeros((4 * K, len(KC)))
WK = np.zeros((4 * K, 3))
WY = np.zeros((4 * K, 1))
KIDX = {"cl_in": 0, "cl_mid": 1, "cl_sky": 2}
for k, (name, a, b) in enumerate(RAT):
    for j, (sgn, terms) in enumerate([(1.0, a), (-1.0, a), (1.0, b), (-1.0, b)]):
        for src, tag, co in terms:
            v = sgn * co
            if src == "s":
                WS[4 * k + j, CH[tag]] += v
            elif src == "b":
                WO[4 * k + j, BLKS.index(tag)] += v
            elif src == "c":
                WC[4 * k + j, CC[tag]] += v
            elif src == "k":
                WK[4 * k + j, KIDX[tag]] += v
            elif src == "y":
                WY[4 * k + j, 0] += v
dense("dS", "fL", WS, np.zeros(4 * K))
dense("dO", "omv", WO, np.zeros(4 * K))
dense("dC", "fC", WC, np.zeros(4 * K))
dense("dK", "clump", WK, np.zeros(4 * K))
dense("dY", "asym", WY, np.zeros(4 * K))
add("V", "dS", "dO", "dC", "dK", "dY")
reshape("Vr", "V", [K, 1, 4])
norm("Vn", "Vr")
Wsel = np.zeros((K, 4 * K))
for k in range(K):
    Wsel[k, 4 * k] = 1.0
dense("feat", "Vn", Wsel, np.zeros(K))

# ---------------------------------------------------------------- 7. hand-written rules
# Normalised [0,1] membership functions built from relu hinges:
#   rise(x;lo,hi) = [relu(x-lo)-relu(x-hi)]/(hi-lo)
#   fall(x;lo,hi) = [relu(hi-x)-relu(lo-x)]/(hi-lo)
#   bump(x;lo,mid,hi) = triangle peaking at 1 at mid
# A membership is a signed sum of hinges; the score table adds them with hand weights.
H = []          # (terms, offset) per relu row
MEM = {}        # name -> list of (hinge row, coefficient)


def _row(terms, off):
    H.append((terms, off))
    return len(H) - 1


def rise(name, terms, lo, hi):
    w = 1.0 / (hi - lo)
    MEM[name] = [(_row(terms, -lo), w), (_row(terms, -hi), -w)]


def fall(name, terms, lo, hi):
    w = 1.0 / (hi - lo)
    MEM[name] = [(_row([(t, -c) for t, c in terms], hi), w),
                 (_row([(t, -c) for t, c in terms], lo), -w)]


def bump(name, terms, lo, mid, hi):
    a, b = 1.0 / (mid - lo), 1.0 / (hi - mid)
    MEM[name] = [(_row(terms, -lo), a), (_row(terms, -mid), -(a + b)), (_row(terms, -hi), b)]


def one(t):
    return [(t, 1.0)]


E = one("q2_11")
EI = one("q2_3")
fall("thin", E, 0.08, 0.26)
fall("vthin", E, 0.02, 0.12)
rise("round", E, 0.72, 0.95)
bump("cigar", E, 0.08, 0.22, 0.42)
bump("inbet", E, 0.30, 0.55, 0.80)
rise("armsT", one("sp12"), 0.08, 0.30)
rise("armsM", one("sp22"), 0.08, 0.30)
rise("armsL", one("sp35"), 0.08, 0.30)
rise("odd", one("asym"), 0.15, 0.50)
rise("clum", one("cl_mid"), 0.08, 0.30)
rise("lop", one("f1_22"), 0.08, 0.30)
rise("dbl", one("pair14"), -0.30, 0.10)
rise("dbl2", one("off14"), -0.50, 0.00)
rise("red", [("col_cm", -1.0)], 0.15, 0.50)
rise("blue", one("col_co"), -0.15, 0.10)
rise("conc", one("c3_12"), 0.25, 0.60)
rise("bulge", [("q2_11", 1.0), ("q2_3", -1.0)], 0.03, 0.20)
rise("bar", [("q2_3", 1.0), ("q2_11", -1.0)], 0.05, 0.25)
rise("ext", one("c18_40"), 0.50, 0.85)
rise("f4in", one("f4_6"), 0.05, 0.25)

WH = np.zeros((len(H), K))
bH = np.zeros(len(H))
for i, (terms, off) in enumerate(H):
    for t, c in terms:
        WH[i, FI[t]] += c
    bH[i] = off
dense("hinge", "feat", WH, bH)
relu("hr", "hinge")

ARM = [("armsT", 1 / 3.0), ("armsM", 1 / 3.0), ("armsL", 1 / 3.0)]
DBL = [("dbl", 0.6), ("dbl2", 0.4)]
SMOOTH = [("red", 0.8), ("conc", 0.5), ("odd", -1.5), ("clum", -1.0), ("lop", -0.8), ("armsT",-0.0),
          ("thin", -1.5), ("bar", -0.5)]
SC = {
    2: [("round", 2.2)] + SMOOTH + [(n, -2.5 * c) for n, c in ARM],
    3: [("inbet", 2.2)] + SMOOTH + [(n, -2.5 * c) for n, c in ARM],
    4: [("cigar", 2.2), ("red", 0.8), ("conc", 0.5), ("odd", -1.5), ("clum", -1.0),
        ("thin", -1.0), ("vthin", -1.5)] + [(n, -1.0 * c) for n, c in ARM],
    8: [("thin", 3.0), ("vthin", 1.2), ("bulge", -1.5), ("conc", -0.5), ("blue", 0.3),
        ("odd", -1.0)],
    9: [("thin", 3.0), ("bulge", 2.0), ("conc", 0.6), ("red", 0.3), ("odd", -1.0)],
    5: [(n, 1.6 * c) for n, c in ARM] + [("bar", 1.2), ("f4in", 0.4), ("thin", -1.2),
                                         ("odd", -0.8)],
    6: [("armsT", 1.8), ("armsM", 1.0), ("conc", 1.0), ("red", 0.6), ("armsL", -0.9),
        ("ext", -0.5), ("thin", -4.5), ("odd", -1.0), ("bar", -0.8)],
    7: [("armsL", 1.6), ("armsM", 0.8), ("blue", 0.7), ("ext", 0.5), ("conc", -0.6),
        ("thin", -4.5), ("odd", -0.8), ("bar", -0.8)],
    0: [("odd", 1.8), ("clum", 1.2), ("lop", 1.0), ("round", -0.8)] + [(n, -1.2 * c) for n, c in DBL],
    1: [(n, 1.8 * c) for n, c in DBL] + [("odd", 0.9), ("clum", 0.7), ("lop", 0.5),
                                          ("round", -0.8)],
}
BIAS = {0: -0.2, 1: -0.2, 2: 0.0, 3: 0.0, 4: -0.5, 5: -0.1, 6: -0.1, 7: -0.1, 8: -0.2, 9: -0.2}
WO10 = np.zeros((10, len(H)))
bO10 = np.zeros(10)
for k in range(10):
    for nm, w in SC[k]:
        for row, c in MEM[nm]:
            WO10[k, row] += w * c
    bO10[k] = BIAS[k]
dense("scores", "hr", WO10, bO10)

# ---------------------------------------------------------------- save
import os
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": LAYERS}, f)
np.savez(os.path.join(OUT, "weights.npz"), **WEIGHTS)
print("layers", len(LAYERS), "weights", sum(v.size for v in WEIGHTS.values()), "features", K)
