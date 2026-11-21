"""build.py - hand-designed galaxy morphology classifier.

All weights are analytic (Gaussians, moments, log-spiral templates, DCT patterns)
or hand-chosen decision constants. No training data is read.

Design:
  Stage A  sky removal (min of 16x16 block means), channel mixing, per-image
           standardization (norm), threshold maps at 1/2/4 sigma.
  Stage B  global linear feature kernels (full-image convs, stride 128): windowed
           light sums, ring masses, sector masses, second moments, log-spiral
           arm templates, bar templates; plus flip-asymmetry and DoG clump maps.
  Stage C  "whiten" modules: features are broadcast onto orthogonal DCT patterns
           in a 16x16 map, normed (this divides by the group RMS), and projected
           back - turning feature *ratios* into bounded quantities.
  Stage D  derived evidence (max-pooled amplitudes, elongations) and a
           hand-written Galaxy-Zoo-style decision rule.
"""
import sys, json
import numpy as np

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."

# ------------------------------------------------------------------ kernels
N = 128
CEN = (N - 1) / 2.0
yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
X = xx - CEN
Y = yy - CEN
R = np.hypot(X, Y)
R2 = X * X + Y * Y
TH = np.arctan2(Y, X)


def win_gauss(sig):
    G = np.exp(-R2 / (2.0 * sig * sig))
    return G / G.sum()


def win_ring(r1, r2, taper=2.0):
    A = np.clip((R - r1) / taper, 0, 1) * np.clip((r2 - R) / taper, 0, 1)
    s = A.sum()
    return A / s if s > 0 else A * 0.0


def sector_win(r1, r2, j, nsec):
    ang = (TH + np.pi) / (2.0 * np.pi) * nsec
    fj = np.floor(ang).astype(int) % nsec
    A = (fj == j) * np.clip((R - r1) / 2.0, 0, 1) * np.clip((r2 - R) / 2.0, 0, 1)
    s = A.sum()
    return A / s if s > 0 else A * 0.0


def mom_kernels(sig):
    """a = <x^2-y^2>, b = <2xy>, t = <x^2+y^2> in units of sig, window G(sig)."""
    G = win_gauss(sig)
    xh, yh = X / sig, Y / sig
    return G * (xh * xh - yh * yh), G * (2.0 * xh * yh), G * (xh * xh + yh * yh)


def spiral_kernel(r1, r2, pitch_deg, phase, hand):
    A = np.clip((R - r1) / 3.0, 0, 1) * np.clip((r2 - R) / 3.0, 0, 1)
    w = 1.0 / np.tan(np.radians(pitch_deg))
    rmid = 0.5 * (r1 + r2)
    T = A * np.cos(2.0 * (TH - phase - hand * w * np.log(np.maximum(R, 1.0) / rmid)))
    T = T - T.mean()
    n = np.sqrt((T * T).sum())
    return T / n if n > 0 else T * 0.0


def bar_kernel(ang_deg):
    """Pure elongated ridge through the centre (bar / cigar core / edge-on
    plane all respond; the decision rule separates them with scale contrasts)."""
    psi = np.radians(ang_deg)
    along = X * np.cos(psi) + Y * np.sin(psi)
    perp = -X * np.sin(psi) + Y * np.cos(psi)
    T = np.exp(-perp ** 2 / (2.0 * 3.0 ** 2)) * np.exp(-along ** 2 / (2.0 * 9.0 ** 2))
    T = T - T.mean()
    n = np.sqrt((T * T).sum())
    return T / n if n > 0 else T * 0.0


def dog_kernel(k, s1, s2):
    o = np.arange(k) - k // 2
    dy, dx = np.meshgrid(o, o, indexing="ij")
    g1 = np.exp(-(dy ** 2 + dx ** 2) / (2.0 * s1 ** 2)); g1 /= g1.sum()
    g2 = np.exp(-(dy ** 2 + dx ** 2) / (2.0 * s2 ** 2)); g2 /= g2.sum()
    return g1 - g2


def grid_windows(size, step, sigs):
    """Gaussian windows evaluated on a strided grid of a smaller map whose
    pixel (u,v) corresponds to original coordinate step*u - CEN."""
    u = np.arange(size)
    gx = step * u - CEN
    gy = step * u - CEN
    GX, GY = np.meshgrid(gx, gy, indexing="xy")  # GX[v,u]? careful: rows=y
    GX, GY = np.meshgrid(gx, gy)  # GX[y,x] = gx[x], GY[y,x]=gy[y]
    out = []
    for s in sigs:
        G = np.exp(-(GX ** 2 + GY ** 2) / (2.0 * s * s))
        out.append(G / G.sum())
    return out


# ------------------------------------------------------------------ layers
L = []
W = {}


def layer(name, op, inputs, **set_):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(set_)
    L.append(d)
    return name


def conv(name, src, kern, stride=1, pad=0, bias=None):
    kern = np.asarray(kern, dtype=np.float64)
    layer(name, "conv", [src], out=int(kern.shape[0]), kernel=[int(kern.shape[2]), int(kern.shape[3])],
          stride=stride, padding=pad)
    W[name + ".W"] = kern.astype(np.float32)
    b = np.zeros(kern.shape[0]) if bias is None else np.asarray(bias, dtype=np.float64)
    W[name + ".b"] = b.astype(np.float32)


def dense(name, src, M, bias=None):
    M = np.asarray(M, dtype=np.float64)
    layer(name, "dense", [src], out=int(M.shape[0]))
    W[name + ".W"] = M.astype(np.float32)
    b = np.zeros(M.shape[0]) if bias is None else np.asarray(bias, dtype=np.float64)
    W[name + ".b"] = b.astype(np.float32)


def relul(name, src):
    return layer(name, "relu", [src])


def reshp(name, src, shape):
    return layer(name, "reshape", [src], shape=shape)


# ------------------------------------------------------------------ stage A
# channel mix: display R=G(z band), G=r band, B=g band.
conv("Lch", "image", np.array([0.25, 0.45, 0.30]).reshape(1, 3, 1, 1))   # luminance
conv("Cgz", "image", np.array([-1.0, 0.0, 1.0]).reshape(1, 3, 1, 1))     # g - z
conv("Crz", "image", np.array([-1.0, 1.0, 0.0]).reshape(1, 3, 1, 1))     # r - z


def sky_remove(ch, out):
    """Luminance sky = min of 16x16 block means (sky is the darkest thing)."""
    layer(ch + "p", "avgpool", [ch], size=[16, 16], stride=[16, 16])        # (1,8,8)
    conv(ch + "n", ch + "p", -np.ones((1, 1, 1, 1)))                        # -blocks
    layer(ch + "s", "maxpool", [ch + "n"], size=[8, 8], stride=[8, 8])      # (1,1,1) = -sky
    dense(ch + "b", ch + "s", np.ones((16384, 1)))                          # broadcast
    reshp(ch + "br", ch + "b", [1, 128, 128])
    layer(out, "add", [ch, ch + "br"])                                      # value - sky


def border_remove(ch, out):
    """Colour sky = mean over an outer annulus of the frame (the min-block
    trick fails for colour: a red galaxy's centre is the most negative block)."""
    conv(ch + "bs", ch, win_ring(52.0, 64.0, 3.0).reshape(1, 1, 128, 128))  # (1,1,1)
    conv(ch + "bn", ch + "bs", -np.ones((1, 1, 1, 1)))                      # -border mean
    dense(ch + "bb", ch + "bn", np.ones((16384, 1)))
    reshp(ch + "br2", ch + "bb", [1, 128, 128])
    layer(out, "add", [ch, ch + "br2"])                                     # colour - sky colour


sky_remove("Lch", "BL")
border_remove("Cgz", "BCg")
border_remove("Crz", "BCr")

layer("NL", "norm", ["BL"])                            # per-image standardised luminance
conv("sh05", "NL", np.ones((1, 1, 1, 1)), bias=[-0.7]); relul("M05", "sh05")  # >= 0.7 sigma
conv("sh1", "NL", np.ones((1, 1, 1, 1)), bias=[-1.0]); relul("M1", "sh1")   # >= 1 sigma
conv("sh2", "NL", np.ones((1, 1, 1, 1)), bias=[-2.0]); relul("M2", "sh2")   # >= 2 sigma
conv("sh4", "NL", np.ones((1, 1, 1, 1)), bias=[-4.0]); relul("M4", "sh4")   # >= 4 sigma

# ------------------------------------------------------------------ stage B: raw features
# luminance light sums (raw, sky-removed)
ks_BL = [win_gauss(8.0), win_gauss(16.0), win_gauss(32.0), win_ring(6, 20), win_ring(16, 40)]
conv("cBL", "BL", np.stack(ks_BL)[:, None, :, :])                       # (5,1,128,128)

# M1 (>=1 sigma): light sums (for asym normalisation) + sectors
ks_M1 = [win_gauss(16.0), win_gauss(32.0)]
for j in range(8):
    ks_M1.append(sector_win(6, 20, j, 8))
for j in range(8):
    ks_M1.append(sector_win(16, 40, j, 8))
conv("cM1", "M1", np.stack(ks_M1)[:, None, :, :])                       # (18,1,128,128)

# M2 (>=2 sigma): light sums + moments at core scales
ks_M2 = [win_gauss(8.0), win_gauss(16.0), win_gauss(32.0)]
for sig in (8.0, 16.0):
    ks_M2 += list(mom_kernels(sig))
conv("cM2", "M2", np.stack(ks_M2)[:, None, :, :])                       # (9,1,128,128)

# M05 (>=0.7 sigma): body map - ring profile + outer moments (star-tapered window)
def win_gauss_cut(sig, rcut):
    G = np.exp(-R2 / (2.0 * sig * sig)) * np.clip((rcut - R) / 6.0, 0, 1)
    return G / G.sum()


def mom_kernels_cut(sig, rcut):
    G = win_gauss_cut(sig, rcut)
    xh, yh = X / sig, Y / sig
    return G * (xh * xh - yh * yh), G * (2.0 * xh * yh), G * (xh * xh + yh * yh)


ks_M05 = [win_gauss_cut(24.0, 58.0),
          win_ring(0, 6), win_ring(6, 14), win_ring(14, 26), win_ring(26, 42)]
ks_M05 += list(mom_kernels_cut(24.0, 58.0))
conv("cM05", "M05", np.stack(ks_M05)[:, None, :, :])                    # (8,1,128,128)

# M4 (>=4 sigma): nucleus moments + nucleus light (tight window)
ks_M4 = list(mom_kernels(16.0)) + [win_gauss(8.0)]
conv("cM4", "M4", np.stack(ks_M4)[:, None, :, :])                       # (4,1,128,128)

# colour sums
ks_C = [win_gauss(8.0), win_gauss(16.0), win_ring(6, 20), win_ring(16, 40)]
conv("cCg", "BCg", np.stack(ks_C)[:, None, :, :])                       # (4,1,128,128)
conv("cCr", "BCr", win_gauss(16.0)[None, None, :, :])                   # (1,1,128,128)

# spiral templates on M1: 2 annuli x 4 pitches x 2 hands x 8 phases
PITCHES = (10.0, 20.0, 32.0, 45.0)
ANN = ((6.0, 16.0), (16.0, 30.0))
ks_T, tpl_meta = [], []
for ai, (r1, r2) in enumerate(ANN):
    for p in PITCHES:
        for hand in (+1.0, -1.0):
            for kph in range(8):
                ph = np.pi * kph / 8.0
                ks_T.append(spiral_kernel(r1, r2, p, ph, hand))
                tpl_meta.append((ai, p, hand, ph))
conv("cT1", "M1", np.stack(ks_T)[:, None, :, :])                        # (128,1,128,128)

# bar templates on M2: 8 rotations
ks_B = [bar_kernel(a) for a in np.arange(8) * 22.5]
conv("cBar", "M2", np.stack(ks_B)[:, None, :, :])                       # (8,1,128,128)

# flip asymmetry on M1 pooled to 32x32
layer("A4", "avgpool", ["M1"], size=[4, 4], stride=[4, 4])               # (1,32,32)
u = np.arange(32)
flip = np.zeros((1024, 1024))
for p in range(1024):
    y, x = p // 32, p % 32
    flip[p, (31 - y) * 32 + (31 - x)] = 1.0
dense("A4f", "A4", flip)
reshp("A4fr", "A4f", [1, 32, 32])
conv("A4n", "A4", -np.ones((1, 1, 1, 1)))
layer("Adiff", "add", ["A4fr", "A4n"])
relul("Adp", "Adiff")
conv("Adn", "Adiff", -np.ones((1, 1, 1, 1)))
relul("Adm", "Adn")
layer("Aabs", "add", ["Adp", "Adm"])                                     # |flip - orig|
aw = grid_windows(32, 4, [16.0, 32.0])
conv("cAs", "Aabs", np.stack(aw)[:, None, :, :])                         # (2,1,32,32) -> (2,1,1)

# DoG clump energy maps from BL
conv("hp1", "BL", dog_kernel(7, 1.0, 3.5)[None, None, :, :], stride=2, pad=3)   # (1,64,64)
relul("hp1p", "hp1")
conv("hp2", "BL", dog_kernel(21, 2.5, 8.0)[None, None, :, :], stride=4, pad=10) # (1,32,32)
relul("hp2p", "hp2")
w64 = grid_windows(64, 2, [16.0, 32.0])
w32 = grid_windows(32, 4, [16.0, 32.0])
conv("chp1", "hp1p", np.stack(w64)[:, None, :, :])                       # k=64 s=64
conv("chp2", "hp2p", np.stack(w32)[:, None, :, :])                       # k=32 s=32

# ------------------------------------------------------------------ raw gather
RAW_SRC = [
    ("cBL", ["l8", "l16", "l32", "lrin", "lrout"]),
    ("cM1", ["fm16", "fm32"]
          + ["ci%d" % j for j in range(8)] + ["co%d" % j for j in range(8)]),
    ("cM2", ["f8m2", "f16m2", "f32m2", "a8", "b8", "t8", "a16", "b16", "t16"]),
    ("cM05", ["f24m05", "r0", "r6", "r14", "r26", "a24", "b24", "t24"]),
    ("cM4", ["a4", "b4", "t4", "f8m4"]),
    ("cCg", ["cg8", "cg16", "cgrin", "cgrout"]),
    ("cCr", ["cr16"]),
    ("cT1", ["tp%d" % i for i in range(128)]),
    ("cBar", ["br%d" % i for i in range(8)]),
    ("cAs", ["as16", "as32"]),
    ("chp1", ["h1a", "h1b"]),
    ("chp2", ["h2a", "h2b"]),
]
KRAW = sum(len(n) for _, n in RAW_SRC)
RAWIDX, pos = {}, 0
for _, names in RAW_SRC:
    for nm in names:
        RAWIDX[nm] = pos
        pos += 1
routers = []
for lname, names in RAW_SRC:
    reshp(lname + "rs", lname, [len(names)])
    Rt = np.zeros((KRAW, len(names)))
    for i, nm in enumerate(names):
        Rt[RAWIDX[nm], i] = 1.0
    dense(lname + "rt", lname + "rs", Rt)
    routers.append(lname + "rt")
layer("RAW", "add", routers)          # (KRAW,)

# ------------------------------------------------------------------ whitening
NP = 256
_jj = np.arange(NP)


def dct_col(j):
    return np.sqrt(2.0 / NP) * np.cos(np.pi * j * (_jj + 0.5) / NP)


# verify orthonormality of the pattern bank (pure math check)
_Pchk = np.stack([dct_col(j) for j in range(1, 33)], 1)
assert np.allclose(_Pchk.T @ _Pchk, np.eye(32), atol=1e-10)
assert np.allclose(_Pchk.sum(0), 0, atol=1e-10)

# slot bookkeeping ------------------------------------------------------
SLOTS = {}


def newslot(nm):
    SLOTS[nm] = len(SLOTS)
    return SLOTS[nm]


L1 = (["ua8", "ub8", "ut8", "ua16", "ub16", "ut16", "ua24", "ub24", "ut24",
       "ua4", "ub4", "ut4", "p0", "p6", "p14", "p26"]
      + ["ci%d" % j for j in range(8)] + ["co%d" % j for j in range(8)]
      + ["tp%d" % i for i in range(128)] + ["br%d" % i for i in range(8)]
      + ["col8", "col16", "colrin", "colrout", "colrz", "as16n", "as32n",
         "cl1a", "cl1b", "cl2a", "cl2b", "sz8", "sz16", "sz24", "coredm",
         "snr16"])
L2 = ["ms8", "ms16", "ms24", "ms4", "m1i", "m2i", "m1o", "m2o"] \
     + ["tm%d%d" % (a, p) for a in range(2) for p in range(4)] \
     + ["brmax", "brnegmin"]
L3 = ["e8", "e16", "e24", "e4", "arm_tight", "arm_loose", "armmax"]
for nm in L1 + L2 + L3:
    newslot(nm)
K = len(SLOTS)


def whiten(mod, src, sel, floor, outs):
    """Broadcast selected scalars onto DCT patterns, norm, project back.
    Result: outs[i] = sel[i] / sqrt(sum(sel^2) + floor^2)."""
    m = len(sel)
    if src == "RAW":
        Sel = np.zeros((m, KRAW))
        for i, nm in enumerate(sel):
            Sel[i, RAWIDX[nm]] = 1.0
    else:
        Sel = np.zeros((m, K))
        for i, nm in enumerate(sel):
            Sel[i, SLOTS[nm]] = 1.0
    dense(mod + "_s", src, Sel)
    P = np.stack([dct_col(j + 1) for j in range(m + 1)], 1)   # (256, m+1)
    dense(mod + "_b", mod + "_s", P[:, :m], bias=floor * P[:, m])
    reshp(mod + "_br", mod + "_b", [1, 16, 16])
    layer(mod + "_n", "norm", [mod + "_br"])
    PK = np.zeros((m, 1, 16, 16))
    for i in range(m):
        PK[i, 0] = P[:, i].reshape(16, 16)
    conv(mod + "_p", mod + "_n", PK, stride=16)
    reshp(mod + "_pr", mod + "_p", [m])
    Rt = np.zeros((K, m))
    for i, nm in enumerate(outs):
        if nm in SLOTS:          # second pair outputs ("x") are discarded
            Rt[SLOTS[nm], i] = 1.0 / 16.0
    dense(mod + "_r", mod + "_pr", Rt)
    return mod + "_r"


def sel_dense(mod, src, names, wts=None):
    M = np.zeros((len(names), K))
    for i, nm in enumerate(names):
        M[i, SLOTS[nm]] = 1.0 if wts is None else wts[i]
    dense(mod + "_s", src, M)
    return mod + "_s"


def maxpool_vals(mod, src, names, wts=None, sign=1.0):
    """Arrange selected scalars spatially and take the max (sign=-1 gives -min)."""
    m = len(names)
    sel_dense(mod, src, names, wts)
    reshp(mod + "_sr", mod + "_s", [1, 1, m])
    if sign < 0:
        conv(mod + "_ng", mod + "_sr", -np.ones((1, 1, 1, 1)))
        srcm = mod + "_ng"
    else:
        srcm = mod + "_sr"
    layer(mod + "_m", "maxpool", [srcm], size=[1, m], stride=[1, m])
    reshp(mod + "_mr", mod + "_m", [1])
    return mod + "_mr"


def route_scalar(mod, srcm, out, w=1.0):
    Rt = np.zeros((K, 1))
    Rt[SLOTS[out], 0] = w
    dense(mod + "_r", srcm, Rt)
    return mod + "_r"


# ---- level 1: whiten raw groups
r1 = []
r1.append(whiten("wmo8", "RAW", ["a8", "b8", "t8"], 0.10, ["ua8", "ub8", "ut8"]))
r1.append(whiten("wmo16", "RAW", ["a16", "b16", "t16"], 0.10, ["ua16", "ub16", "ut16"]))
r1.append(whiten("wmo24", "RAW", ["a24", "b24", "t24"], 0.12, ["ua24", "ub24", "ut24"]))
r1.append(whiten("wmo4", "RAW", ["a4", "b4", "t4"], 0.10, ["ua4", "ub4", "ut4"]))
r1.append(whiten("wring", "RAW", ["r0", "r6", "r14", "r26"], 0.08,
                 ["p0", "p6", "p14", "p26"]))
r1.append(whiten("wseci", "RAW", ["ci%d" % j for j in range(8)], 0.05,
                 ["ci%d" % j for j in range(8)]))
r1.append(whiten("wseco", "RAW", ["co%d" % j for j in range(8)], 0.05,
                 ["co%d" % j for j in range(8)]))
# spiral templates: whiten per (annulus, pitch) group so each pitch's best
# phase/hand is measured relative to that group's own spread
for ai in range(2):
    for pi in range(4):
        idxs = [i for i, (a, p, h, ph) in enumerate(tpl_meta)
                if a == ai and PITCHES.index(p) == pi]
        r1.append(whiten("wtpl%d%d" % (ai, pi), "RAW",
                         ["tp%d" % i for i in idxs], 0.20,
                         ["tp%d" % i for i in idxs]))
r1.append(whiten("wbar", "RAW", ["br%d" % i for i in range(8)], 0.10,
                 ["br%d" % i for i in range(8)]))
# colour pairs (colour sum, light sum) -> bounded colour index
r1.append(whiten("wcol8", "RAW", ["cg8", "l8"], 0.03, ["col8", "x"]))
r1.append(whiten("wcol16", "RAW", ["cg16", "l16"], 0.03, ["col16", "x"]))
r1.append(whiten("wcolri", "RAW", ["cgrin", "lrin"], 0.03, ["colrin", "x"]))
r1.append(whiten("wcolro", "RAW", ["cgrout", "lrout"], 0.03, ["colrout", "x"]))
r1.append(whiten("wcolrz", "RAW", ["cr16", "l16"], 0.03, ["colrz", "x"]))
# asymmetry pairs (asym sum, light sum)
r1.append(whiten("was16", "RAW", ["as16", "fm16"], 0.03, ["as16n", "x"]))
r1.append(whiten("was32", "RAW", ["as32", "fm32"], 0.03, ["as32n", "x"]))
# clump energy pairs (DoG energy, light sum)
r1.append(whiten("wcl1a", "RAW", ["h1a", "l16"], 0.02, ["cl1a", "x"]))
r1.append(whiten("wcl1b", "RAW", ["h1b", "l32"], 0.02, ["cl1b", "x"]))
r1.append(whiten("wcl2a", "RAW", ["h2a", "l16"], 0.02, ["cl2a", "x"]))
r1.append(whiten("wcl2b", "RAW", ["h2b", "l32"], 0.02, ["cl2b", "x"]))
# size pairs (second moment, flux)
r1.append(whiten("wsz8", "RAW", ["t8", "f8m2"], 0.10, ["sz8", "x"]))
r1.append(whiten("wsz16", "RAW", ["t16", "f16m2"], 0.10, ["sz16", "x"]))
r1.append(whiten("wsz24", "RAW", ["t24", "f24m05"], 0.12, ["sz24", "x"]))
# core dominance (M4 nucleus light vs M2 core light)
r1.append(whiten("wcore", "RAW", ["f8m4", "f8m2"], 0.10, ["coredm", "x"]))
# significance (core light alone; gates noise-driven evidence)
r1.append(whiten("wsnr", "RAW", ["f16m2"], 0.40, ["snr16", "x"]))
layer("FEAT1", "add", r1)            # (K,)

# ---- level 2: derived amplitudes
r2 = []
for sc, msv in (("8", "ms8"), ("16", "ms16"), ("24", "ms24"), ("4", "ms4")):
    # 8 directional projections of (ua, ub)
    M = np.zeros((8, K))
    for k in range(8):
        psi = np.pi * k / 8.0
        M[k, SLOTS["ua" + sc]] = np.cos(2 * psi)
        M[k, SLOTS["ub" + sc]] = np.sin(2 * psi)
    dense("mp" + sc + "_p", "FEAT1", M)
    reshp("mp" + sc + "_pr", "mp" + sc + "_p", [1, 1, 8])
    layer("mp" + sc + "_m", "maxpool", ["mp" + sc + "_pr"], size=[1, 8], stride=[1, 8])
    reshp("mp" + sc + "_mr", "mp" + sc + "_m", [1])
    r2.append(route_scalar("mp" + sc + "r", "mp" + sc + "_mr", msv))

# sector Fourier amplitudes: theta_j = -pi + (j+0.5)*pi/4
thsec = np.array([-np.pi + (j + 0.5) * np.pi / 4.0 for j in range(8)])


def fourier_amp(mod, src, base, out, mult):
    nph = 8 if mult == 1 else 4
    M = np.zeros((nph, K))
    for k in range(nph):
        phi = np.pi * k / nph
        for j in range(8):
            M[k, SLOTS["%s%d" % (base, j)]] = np.cos(mult * (thsec[j] - phi))
    dense(mod + "_p", src, M)
    reshp(mod + "_pr", mod + "_p", [1, 1, nph])
    layer(mod + "_m", "maxpool", [mod + "_pr"], size=[1, nph], stride=[1, nph])
    reshp(mod + "_mr", mod + "_m", [1])
    return route_scalar(mod + "r", mod + "_mr", out)


r2.append(fourier_amp("fm1i", "FEAT1", "ci", "m1i", 1))
r2.append(fourier_amp("fm2i", "FEAT1", "ci", "m2i", 2))
r2.append(fourier_amp("fm1o", "FEAT1", "co", "m1o", 1))
r2.append(fourier_amp("fm2o", "FEAT1", "co", "m2o", 2))

# template maxima per (annulus, pitch): max over 4 phases x 2 hands
tpl_by_ap = {}
for ti, (ai, p, hand, ph) in enumerate(tpl_meta):
    tpl_by_ap.setdefault((ai, PITCHES.index(p)), []).append("tp%d" % ti)
for (ai, pi), names in sorted(tpl_by_ap.items()):
    mv = maxpool_vals("tmx%d%d" % (ai, pi), "FEAT1", names)
    r2.append(route_scalar("tmxr%d%d" % (ai, pi), mv, "tm%d%d" % (ai, pi)))

# bar anisotropy: max and -min over 8 rotations
mv = maxpool_vals("brmx", "FEAT1", ["br%d" % i for i in range(8)])
r2.append(route_scalar("brmxr", mv, "brmax"))
mv = maxpool_vals("brmn", "FEAT1", ["br%d" % i for i in range(8)], sign=-1.0)
r2.append(route_scalar("brmnr", mv, "brnegmin"))
layer("FEAT2", "add", ["FEAT1"] + r2)   # (K,)

# ---- level 3: elongation ratios, arm aggregates
r3 = []
for sc in ("8", "16", "24", "4"):
    r3.append(whiten("we" + sc, "FEAT2", ["ms" + sc, "ut" + sc], 0.05,
                     ["e" + sc, "x"]))
tight_names = ["tm%d%d" % (a, p) for a in range(2) for p in (0, 1)]
loose_names = ["tm%d%d" % (a, p) for a in range(2) for p in (2, 3)]
mv = maxpool_vals("armt", "FEAT2", tight_names)
r3.append(route_scalar("armtr", mv, "arm_tight"))
mv = maxpool_vals("arml", "FEAT2", loose_names)
r3.append(route_scalar("armlr", mv, "arm_loose"))
mv = maxpool_vals("armx", "FEAT2", tight_names + loose_names)
r3.append(route_scalar("armxr", mv, "armmax"))
layer("FEAT3", "add", ["FEAT2"] + r3)   # (K,)

# ------------------------------------------------------------------ scoring
# Hand-written decision rule mirroring the Galaxy Zoo tree.  Elongation
# features obey e = epsilon/sqrt(1+epsilon^2) with epsilon=(1-q^2)/(1+q^2):
# q=0.7 -> e~0.32 (round/in-between edge), q=0.55 -> e~0.50,
# q=0.45 -> e~0.55 (in-between/cigar edge), q=0.3 -> e~0.64 (edge-on).
# Constants come from these physical values, not from data.
FEAT = "FEAT3"

# evidence units: relu(sum(w*feat) - c); conjunctions are linear gates
EVDEFS = [
    # edge-on: thin at core AND body scales AND extended AND well detected
    ("eoX",  {"e24": 3.0, "e16": 3.0, "e8": 3.0, "sz24": 5.0, "snr16": 2.5}, 8.8),
    ("eoX2", {"e24": 3.5, "e16": 3.5}, 4.4),          # very flattened
    ("bulge9", {"e24": 3.0, "e16": 3.0, "col8": -4.0, "sz24": 2.0}, 4.9),
    ("e4loEO", {"e4": -5.0, "e16": 3.0, "e24": 3.0}, 1.2),
    ("armTplT", {"arm_tight": 4.0}, 1.80),
    ("armTplL", {"arm_loose": 4.0}, 1.80),
    ("red8",   {"col8": -4.0}, 0.30),                 # red centre
    ("red16",  {"col16": -4.0}, 0.20),                # red overall
    ("blue",   {"col16": 4.0}, 0.20),                 # blue disk
    ("clumpF", {"cl1a": 5.0}, 0.40),                  # fine clumps
    # arms: azimuthal m=2 mode after subtracting the ellipticity a smooth
    # elongated body produces (an ellipse gives m2i ~ 2.2*e16)
    ("arm2",  {"m2i": 2.2, "m2o": 2.2, "e16": -3.4, "e24": -3.4}, 0.55),
    ("m2in",   {"m2i": 1.5}, 0.78),                   # strong m=2 -> loose winding
    ("m2inLO", {"m2i": -2.5}, -1.50),                 # weak m=2 -> tight winding
    ("m2out_e", {"m2o": 2.5}, 1.00),
    ("m1o_e",  {"m1o": 2.5, "snr16": 3.0}, 4.65),     # lopsided + signal
    ("m1i_e",  {"m1i": 2.5, "snr16": 3.0}, 4.30),
    ("asyH",   {"as16n": 4.0, "snr16": 3.0}, 4.50),   # strong asymmetry (merger)
    ("asyM",   {"as16n": 4.0, "snr16": 3.0}, 4.15),   # moderate (disturbed)
    ("asy32H", {"as32n": 4.0, "snr16": 3.0}, 4.60),
    ("elongPen", {"e16": 2.5}, 1.00),                 # penalty for arm classes
    ("notSmallE", {"sz24": 4.0}, 1.60),               # extended body (sz24>0.4)
    ("barE",   {"e8": 3.0, "e16": -3.5}, -0.10),      # core more bar-like than body
    ("barA",   {"brmax": 2.5, "brnegmin": 2.5, "e8": 2.0, "e24": -2.5}, 1.45),
    ("roundE",  {"e16": -5.0}, -1.50),                # e16 < 0.30
    ("midE",    {"e16": 2.5}, 0.75),                  # e16 > 0.30
    ("midEhi",  {"e16": 2.5}, 1.35),                  # e16 > 0.54
    ("cigE",    {"e16": 2.5, "e24": 1.0}, 1.55),      # elongated at both scales
    ("smallE",  {"sz24": -4.0}, -2.20),               # compact body
    ("extE",    {"sz24": 4.0}, 2.40),                 # extended body
    ("coreHi",  {"coredm": 3.0}, 1.35),               # bulge-dominated
    ("coreLo",  {"coredm": -3.0}, -1.20),             # disk-dominated
    ("coreLoEO", {"coredm": -3.0, "e16": 4.0, "e24": 4.0, "e8": 3.0}, 4.2),
    ("e4lo",    {"e4": -5.0}, -2.00),                 # round nucleus
    ("e4hi",    {"e4": 5.0}, 2.00),                   # elongated nucleus
    ("smoothC", {"cl2a": -6.0}, -0.90),               # little medium structure
    ("smoothT", {"cl1a": -6.0}, -0.60),               # little fine structure
]

E = len(EVDEFS)
ME = np.zeros((E, K))
BE = np.zeros(E)
for i, (nm, terms, c) in enumerate(EVDEFS):
    for f, w in terms.items():
        ME[i, SLOTS[f]] += w
    BE[i] = -c
dense("evp", FEAT, ME, BE)
relul("evr", "evp")

# class scores: linear in evidence (+ small direct feature terms)
CLDEFS = {
 0: ({"asyM": 2.4, "m1o_e": 1.0, "clumpF": 1.6, "blue": 0.6, "arm2": -1.0,
      "eoX": -2.0, "red16": -0.5, "cigE": -1.5, "midEhi": -1.0},
     {"as16n": 1.0, "m1o": 0.5}, -0.10),
 1: ({"asyH": 2.2, "asy32H": 1.5, "m1o_e": 0.6, "m1i_e": 1.0, "clumpF": 0.4,
      "blue": 0.4, "arm2": -1.0, "eoX": -2.0},
     {"as16n": 1.2, "as32n": 0.8}, -0.15),
 2: ({"smoothC": 1.4, "smoothT": 1.0, "roundE": 1.3, "red16": 0.6, "red8": 0.4,
      "asyM": -0.8, "clumpF": -0.6, "armTplT": -0.6,
      "eoX": -2.0, "eoX2": -2.0, "arm2": -1.2, "clumpF": -0.6,
      "midE": -1.0, "cigE": -1.6},
     {"e16": -1.0}, 0.10),
 3: ({"smoothC": 1.4, "smoothT": 1.0, "midE": 1.8, "midEhi": -1.4, "smallE": 0.3,
      "asyM": -0.8, "clumpF": -0.6, "armTplL": -0.8, "armTplT": -0.5,
      "red16": 0.5, "eoX": -2.0, "eoX2": -2.0, "arm2": -1.2,
      "cigE": -1.4, "roundE": -1.0},
     {"e16": 1.2}, 0.0),
 4: ({"smoothC": 0.8, "smoothT": 0.9, "cigE": 2.4, "smallE": 0.4, "asyM": -0.6,
      "red16": 0.5, "eoX": -2.2, "eoX2": -2.2, "extE": -1.5, "arm2": -1.0},
     {"e16": 1.6, "e24": 0.8, "sz24": -2.5}, -0.60),
 5: ({"arm2": 2.0, "barE": 2.0, "barA": 2.0, "armTplT": 0.8, "armTplL": 0.8,
      "eoX": -3.0, "eoX2": -3.0, "elongPen": -1.5, "asyH": -0.8},
     {"brmax": 0.6, "brnegmin": 0.6, "e8": 1.2}, 0.0),
 6: ({"arm2": 2.0, "m2inLO": 0.8, "blue": 0.4, "armTplT": 1.6, "barE": -1.3,
      "barA": -1.0, "eoX": -3.0, "eoX2": -3.0, "elongPen": -2.5, "asyH": -0.8},
     {"m2i": -0.5}, 0.0),
 7: ({"arm2": 2.0, "m2in": 1.2, "m2out_e": 0.4, "armTplL": 1.5, "barE": -1.3,
      "barA": -1.0, "eoX": -3.0, "eoX2": -3.0, "elongPen": -2.5, "asyH": -0.8,
      "extE": 0.4},
     {}, 0.0),
 8: ({"eoX": 2.4, "eoX2": 1.2, "coreLoEO": 0.8, "e4hi": 0.6, "arm2": -1.2,
      "asyH": -2.2, "asyM": -1.5,
      "barA": -1.2, "coreHi": -1.2, "notSmallE": 0.7},
     {"e24": 0.6, "e16": 0.5, "sz24": 1.2, "col8": -0.2}, 0.0),
 9: ({"eoX": 2.4, "eoX2": 1.2, "bulge9": 2.2, "coreHi": 0.8, "e4loEO": 0.5,
      "arm2": -1.2, "barA": -1.2, "coreLo": -1.0, "asyH": -2.2, "asyM": -1.5,
      "notSmallE": 0.5},
     {"e24": 0.7, "e16": 0.5, "sz24": 0.8, "col8": -0.8}, 0.0),
}

W1 = np.zeros((10, K))
B1 = np.zeros(10)
W2 = np.zeros((10, E))
for k, (evs, feats, b) in CLDEFS.items():
    for nm, w in evs.items():
        W2[k, [i for i, (n, _, _) in enumerate(EVDEFS) if n == nm][0]] += w
    for nm, w in feats.items():
        W1[k, SLOTS[nm]] += w
    B1[k] = b
dense("sc1", FEAT, W1, B1)
dense("sc2", "evr", W2)
layer("scores", "add", ["sc1", "sc2"])       # (10,)

# ------------------------------------------------------------------ output
import os
os.makedirs(OUTDIR, exist_ok=True)
with open(os.path.join(OUTDIR, "network.json"), "w") as f:
    json.dump({"layers": L}, f)
json.dump(SLOTS, open(os.path.join(OUTDIR, "slots.json"), "w"))
np.savez(os.path.join(OUTDIR, "weights.npz"),
         **{k: np.asarray(v, dtype=np.float32) for k, v in W.items()})
print("wrote", OUTDIR, "layers:", len(L), "weights:", sum(int(np.prod(v.shape)) for v in W.values()))
