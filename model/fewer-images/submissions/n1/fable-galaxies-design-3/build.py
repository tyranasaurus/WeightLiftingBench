#!/usr/bin/env python3
"""Hand-built galaxy morphology network.  python3 build.py OUT_DIR

Writes OUT_DIR/network.json and OUT_DIR/weights.npz.  Reads no files: every weight below comes from a
formula, a geometric construction, or a rule table typed by hand from general knowledge of galaxies and
of how the Legacy Survey viewer renders its images.  Nothing is trained or fitted.

Outline of the computation (all of it conv / dense / relu / pool / add / reshape):
  1. luminance minus sky, resampled on a polar grid about the frame centre (dense layer of Gaussian blobs);
     red-minus-blue resampled on the same grid.
  2. soft thresholds at 6 absolute surface-brightness levels and 3 levels relative to the central peak;
     along every ray the first run of above-threshold samples is kept ("star-shaped" mask of the central
     object), so unconnected field stars and galaxies are ignored.
  3. the isophote radius as a function of direction for every level; opposite directions are combined by
     min / max (the min ignores a neighbour on one side, max - min measures lopsidedness).
  4. scalar measurements: semi-major / semi-minor axis per level, mirror asymmetry of the stack of
     isophotes (twist), lopsidedness, colour of outskirts and core, bright secondary blobs, peak brightness.
  5. every measurement is a ratio; it is read out as "thermometer" bits [numerator - t * denominator > 0],
     and the class scores are sums of hand-typed log-probability tables over those bits (naive-Bayes form)
     plus the class frequencies expected for a Galaxy-Zoo style sample.
"""
import sys, json, math
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ----------------------------------------------------------------------------------------------------------
# constants from the viewer's rendering formula:  rgb = v * arcsinh(Q*I) / (sqrt(Q)*I),  v = flux*scale + m,
# I = mean of v over the three bands, Q = 20, m = 0.03.  So the mean of R,G,B equals arcsinh(Q*I)/sqrt(Q),
# and blank sky (flux 0) renders at arcsinh(Q*m)/sqrt(Q) in every channel.
Q_STRETCH, M_PED = 20.0, 0.03
SKY = math.asinh(Q_STRETCH * M_PED) / math.sqrt(Q_STRETCH)          # 0.1272

N_IMG = 128
CEN = 63.5                                    # the catalogue position falls between the four central pixels
NTH = 72                                      # directions, 5 degrees apart
NH = NTH // 2                                 # 36: directions per half turn
# radii in pixels: 0.5 px steps to 9.5, 1 px steps to 23, 2 px steps to 42, 4 px steps to 88 (frame corner: 90)
RAD = np.array([0.5 * q for q in range(20)] + list(range(10, 24)) + list(range(24, 44, 2)) + list(range(44, 92, 4)),
               dtype=np.float64)
NR = len(RAD)                                 # 56


def ring_geometry(radii):
    edges = np.concatenate([[0.0], 0.5 * (radii[1:] + radii[:-1]), [radii[-1] + 0.5 * (radii[-1] - radii[-2])]])
    return edges[1:] - edges[:-1], math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


DR, AREA = ring_geometry(RAD)                 # radial length and ring area represented by each sample
# colour is measured on the whole-pixel radii only, from a 2x2 binned map (same smoothing for colour and light)
CIDX = [i for i, r in enumerate(RAD) if abs(r - round(r)) < 1e-9]
CRAD = RAD[CIDX]
NC = len(CIDX)                                # 46
CDR, CAREA = ring_geometry(CRAD)
# dark length (px, at most 2 per sample) that a sample adds to a ray's deficit; 2 px of darkness ends the ray
DEF_W = np.minimum(DR, 2.0) / 2.0
N_CEN = int(np.sum(RAD <= 2.0))               # samples within 2 px of the centre

# surface-brightness levels (in units of the rendered luminance above sky)
ABS_LEVELS = [0.045, 0.08, 0.13, 0.20, 0.29, 0.40]      # roughly a factor 1.8 in flux between neighbours
REL_DROPS = [0.10, 0.20, 0.32]                           # below the central peak (0.22 = one e-fold in flux)
REL_FLOOR = ABS_LEVELS[1]
NA, NREL = len(ABS_LEVELS), len(REL_DROPS)
K = NA + NREL                                 # 9 levels; index 0..5 absolute (A1..A6), 6..8 relative (R1..R3)
# width of each soft threshold: about the brightness change per pixel of a galaxy profile at that level
# (profiles are compressed near the sky by the arcsinh stretch and steepest in the bright, logarithmic part)
RAMP_W = [0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.08]
KR2 = NA + 1                                  # the level "peak - 0.20"
GATE_C = 4.0
R_NEAR = 30.0                                 # px: neighbourhood searched for a bright second nucleus

layers, weights = [], {}


def L(name, op, inputs, **kw):
    d = {"name": name, "op": op, "inputs": list(inputs)}
    d.update(kw)
    layers.append(d)
    return name


def conv1(name, inp, W, b):
    W = np.asarray(W, dtype=np.float64)
    weights[name + ".W"] = W.reshape(W.shape[0], W.shape[1], 1, 1).astype(np.float32)
    weights[name + ".b"] = np.asarray(b, dtype=np.float64).astype(np.float32)
    return L(name, "conv", [inp], out=int(W.shape[0]), kernel=[1, 1], stride=1, padding=0)


def dense(name, inp, W, b):
    W = np.asarray(W)
    weights[name + ".W"] = W.astype(np.float32)
    weights[name + ".b"] = np.asarray(b, dtype=np.float64).astype(np.float32)
    return L(name, "dense", [inp], out=int(W.shape[0]))


def relu(name, inp):
    return L(name, "relu", [inp])


def add(name, inps):
    return L(name, "add", inps)


def reshape(name, inp, shape):
    return L(name, "reshape", [inp], shape=list(shape))


def maxpool(name, inp, size, stride=None):
    return L(name, "maxpool", [inp], size=list(size), stride=list(stride or size))


def avgpool(name, inp, size, stride=None):
    return L(name, "avgpool", [inp], size=list(size), stride=list(stride or size))


# ----------------------------------------------------------------------------------------------------------
# 1. polar resampling
def polar_matrix(n_pix, scale, sig_min, radii, dr):
    """Rows: (radius i, direction t) in the order i*NTH + t.  Columns: pixels of an n_pix x n_pix map whose
    pixel size is `scale` original pixels.  Each row is a normalised Gaussian blob centred on the sample."""
    Wm = np.zeros((len(radii) * NTH, n_pix * n_pix), dtype=np.float32)
    dth = 2 * math.pi / NTH
    for i, r in enumerate(radii):
        sig = max(0.8, 0.4 * dr[i], 0.4 * r * dth)          # original pixels
        sig_m = max(sig_min, sig / scale)                    # pixels of this map
        half = int(math.ceil(3.5 * sig_m)) + 1
        for t in range(NTH):
            th = t * dth
            x = CEN + r * math.cos(th)
            y = CEN + r * math.sin(th)
            if not (-0.5 <= x <= N_IMG - 0.5 and -0.5 <= y <= N_IMG - 0.5):
                continue                                     # outside the frame: reads as blank sky
            xm = (x - (scale - 1) / 2.0) / scale
            ym = (y - (scale - 1) / 2.0) / scale
            x0, y0 = int(round(xm)), int(round(ym))
            xs = np.arange(max(0, x0 - half), min(n_pix, x0 + half + 1))
            ys = np.arange(max(0, y0 - half), min(n_pix, y0 + half + 1))
            if len(xs) == 0 or len(ys) == 0:
                continue
            gx = np.exp(-0.5 * ((xs - xm) / sig_m) ** 2)
            gy = np.exp(-0.5 * ((ys - ym) / sig_m) ** 2)
            blob = np.outer(gy, gx)
            s = blob.sum()
            if s <= 0:
                continue
            blob /= s
            row = i * NTH + t
            cols = (ys[:, None] * n_pix + xs[None, :]).ravel()
            Wm[row, cols] = blob.ravel().astype(np.float32)
    return Wm


conv1("lum", "image", [[1 / 3.0, 1 / 3.0, 1 / 3.0]], [-SKY])                 # luminance above sky
dense("polL", "lum", polar_matrix(N_IMG, 1, 0.8, RAD, DR), np.zeros(NR * NTH))
reshape("P", "polL", [NR, 2, NH])                                             # channel = radius, (half turn, direction)

_pc = polar_matrix(N_IMG // 2, 2, 0.8, CRAD, CDR)
conv1("dif", "image", [[1.0, 0.0, -1.0]], [0.0])                             # red minus blue (z band minus g band)
avgpool("dif2", "dif", [2, 2])
dense("polD", "dif2", _pc, np.zeros(NC * NTH))
reshape("Dp", "polD", [NC, 2, NH])
avgpool("lum2", "lum", [2, 2])
dense("polF", "lum2", _pc, np.zeros(NC * NTH))
reshape("Fp", "polF", [NC, 2, NH])

# ----------------------------------------------------------------------------------------------------------
# 2. central peak and thresholds
Wc = np.zeros((N_CEN, NR))
for q in range(N_CEN):
    Wc[q, q] = 1.0
conv1("cen", "P", Wc, np.zeros(N_CEN))
reshape("cen_r", "cen", [1, 2 * N_CEN, NH])
maxpool("L0", "cen_r", [2 * N_CEN, NH])                                       # brightest smoothed value within 2 px

dense("rt1", "L0", np.ones((NREL, 1)), [-d - REL_FLOOR for d in REL_DROPS])  # peak - drop - floor
relu("rt1r", "rt1")
Wt = np.zeros((NREL * NTH, NREL))
for q in range(NREL):
    Wt[q * NTH:(q + 1) * NTH, q] = 1.0
dense("rt2", "rt1r", Wt, np.full(NREL * NTH, REL_FLOOR))                      # max(peak - drop, floor), broadcast
reshape("Tmap", "rt2", [NREL, 2, NH])

NRAMP = 2 * K * NR


def ramp_idx(k, i, u):
    return (k * NR + i) * 2 + u


Wa, ba, Wb = np.zeros((NRAMP, NR)), np.zeros(NRAMP), np.zeros((NRAMP, NREL))
for k in range(K):
    for i in range(NR):
        for u in range(2):
            c = ramp_idx(k, i, u)
            Wa[c, i] = 1.0 / RAMP_W[k]
            ba[c] = 0.5 - u
            if k < NA:
                ba[c] -= ABS_LEVELS[k] / RAMP_W[k]
            else:
                Wb[c, k - NA] = -1.0 / RAMP_W[k]
conv1("thr_a", "P", Wa, ba)
conv1("thr_b", "Tmap", Wb, np.zeros(NRAMP))
add("thr", ["thr_a", "thr_b"])
relu("ramp", "thr")                                                           # s = ramp[.,0] - ramp[.,1] in [0, 1]

# star-shaped masks: M(k, i) = relu( s(i) - deficit(i) ), deficit(i) = sum_{i' < i} DEF_W(i') (1 - s(i'));
# two pixels of darkness along a ray end it, so only the object at the centre is kept
NM = K * NR
IDX_S, IDX_G1, IDX_G2 = NM, NM + NR, NM + 2 * NR
NSTAR = NM + 3 * NR
Ws, bs = np.zeros((NSTAR, NRAMP)), np.zeros(NSTAR)
for k in range(K):
    for i in range(NR):
        o = k * NR + i
        Ws[o, ramp_idx(k, i, 0)] = 1.0
        Ws[o, ramp_idx(k, i, 1)] = -1.0
        for i2 in range(i):
            Ws[o, ramp_idx(k, i2, 0)] = DEF_W[i2]
            Ws[o, ramp_idx(k, i2, 1)] = -DEF_W[i2]
        bs[o] = -DEF_W[:i].sum()
for i in range(NR):
    # the plain threshold map s at level R2, and "a real gap lies between here and the centre":
    # gap = clip(2 * deficit - 1.5, 0, 1), i.e. 1.5 to 2.5 px of darkness nearer the centre
    Ws[IDX_S + i, ramp_idx(KR2, i, 0)] = 1.0
    Ws[IDX_S + i, ramp_idx(KR2, i, 1)] = -1.0
    for o, off in ((IDX_G1 + i, 1.5), (IDX_G2 + i, 2.5)):
        for i2 in range(i):
            Ws[o, ramp_idx(KR2, i2, 0)] = -2.0 * DEF_W[i2]
            Ws[o, ramp_idx(KR2, i2, 1)] = 2.0 * DEF_W[i2]
        bs[o] = 2.0 * DEF_W[:i].sum() - off
conv1("star_pre", "ramp", Ws, bs)
relu("M", "star_pre")

# ----------------------------------------------------------------------------------------------------------
# 2b. bumps along rays.  In linear flux a smooth galaxy (any Sersic profile, any bulge + disc sum, seen through
# any seeing) falls off convexly along every ray once outside the core: F(r) lies below the chord joining
# F(r - d) and F(r + d).  Spiral arms, rings and the ends of bars put local bumps on the rays.
# Linear flux is recovered from the rendered luminance by inverting the arcsinh stretch with a
# piecewise-linear sinh; a bump counts when it stands above the chord by more than BUMP_K of the local flux.
KNOTS = [-0.05 + 0.05 * q for q in range(20)]            # luminance above sky at the hinge points
SAT_L = 0.62                                             # above this a red galaxy starts to clip in the red channel
BUMP_K, BUMP_EPS = 0.08, 0.006
R_BUMP_MIN = 5.0


def lin_flux(lp):
    return math.sinh(math.sqrt(Q_STRETCH) * (lp + SKY)) / Q_STRETCH - M_PED


_fk = [lin_flux(x) for x in KNOTS + [KNOTS[-1] + 0.05]]
_slopes = [(_fk[q + 1] - _fk[q]) / 0.05 for q in range(len(KNOTS))]
HINGE = [_slopes[0]] + [_slopes[q] - _slopes[q - 1] for q in range(1, len(KNOTS))]   # F = F(k0) + sum HINGE_j relu(x - k_j)
F_BASE = _fk[0]
NKN = len(KNOTS)
NFH = (NKN + 1) * NR


def fh_idx(j, i):
    return j * NR + i


Wfh, bfh = np.zeros((NFH, NR)), np.zeros(NFH)
for i in range(NR):
    for j in range(NKN):
        Wfh[fh_idx(j, i), i] = 1.0
        bfh[fh_idx(j, i)] = -KNOTS[j]
    Wfh[fh_idx(NKN, i), i] = 1.0
    bfh[fh_idx(NKN, i)] = -SAT_L
conv1("fh_pre", "P", Wfh, bfh)
relu("fh", "fh_pre")


def bump_step(r):
    return 2.0 if r < 24 else (4.0 if r < 44 else 8.0)


BUMP = []                                                 # (i, i_inner, i_outer, weight_inner, weight_outer)
for i, r in enumerate(RAD):
    d = bump_step(r)
    if r < R_BUMP_MIN or r + d > RAD[-1]:
        continue
    im = int(np.argmin(np.abs(RAD - (r - d))))
    ip = int(np.argmin(np.abs(RAD - (r + d))))
    d1, d2 = r - RAD[im], RAD[ip] - r
    BUMP.append((i, im, ip, d2 / (d1 + d2), d1 / (d1 + d2)))
NBUMP = len(BUMP)
NARM = NBUMP + NR
Waf, Wam, bam = np.zeros((NARM, NFH)), np.zeros((NARM, NSTAR)), np.zeros(NARM)
for a, (i, im, ip, wm, wp) in enumerate(BUMP):
    for j in range(NKN):
        Waf[a, fh_idx(j, i)] += HINGE[j] * (1.0 - BUMP_K)
        Waf[a, fh_idx(j, im)] -= HINGE[j] * wm
        Waf[a, fh_idx(j, ip)] -= HINGE[j] * wp
    Waf[a, fh_idx(NKN, im)] = -50.0                       # ignore rays whose inner point is near saturation
    Wam[a, 0 * NR + ip] = GATE_C                          # whole stencil inside the faint envelope of the object
    bam[a] = -BUMP_K * F_BASE - BUMP_EPS - GATE_C
for i in range(NR):
    a = NBUMP + i                                         # flux inside the envelope, for normalisation
    for j in range(NKN):
        Waf[a, fh_idx(j, i)] = HINGE[j]
    Wam[a, 0 * NR + i] = GATE_C
    bam[a] = F_BASE - GATE_C
conv1("arm_f", "fh", Waf, bam)
conv1("arm_m", "M", Wam, np.zeros(NARM))
add("arm_pre", ["arm_f", "arm_m"])
relu("arm", "arm_pre")
Was = np.zeros((2, NARM))
for a, (i, im, ip, wm, wp) in enumerate(BUMP):
    Was[0, a] = AREA[i]
Was[1, NBUMP:] = AREA
conv1("arm_s", "arm", Was, np.zeros(2))                                       # (2, 2, 36): bump flux, flux, per ray
Wasm = np.zeros((2 * NTH, 2 * NTH))
for t in range(NTH):
    for dt, wgt in zip(range(-2, 3), (1 / 9.0, 2 / 9.0, 3 / 9.0, 2 / 9.0, 1 / 9.0)):
        Wasm[t, (t + dt) % NTH] = wgt                     # smooth the bump flux over +-10 degrees
    Wasm[NTH + t, NTH + t] = 1.0
dense("arm_sm", "arm_s", Wasm, np.zeros(2 * NTH))
reshape("arm_r", "arm_sm", [2, 2, NH])
avgpool("armavg", "arm_r", [2, NH])                                           # mean over all rays
maxpool("arm_mx0", "arm_r", [2, 1], [1, 1])
avgpool("armmax", "arm_mx0", [1, NH])                                         # mean over ray pairs of the larger
# 2 * armavg - armmax = mean over ray pairs of the SMALLER bump flux: bumps present on both sides of the centre
# (a pair of arms, a ring, two bar ends); a star or companion on one side does not count.

# ----------------------------------------------------------------------------------------------------------
# 3. isophote radius functions rho_k(direction)
Wr = np.zeros((K, NSTAR))
for k in range(K):
    Wr[k, k * NR:(k + 1) * NR] = DR
conv1("rho", "M", Wr, np.zeros(K))                                            # (K, 2, 36)

Wsm = np.zeros((K * NTH, K * NTH))                                            # circular [1 2 1]/4 over direction
for k in range(K):
    for t in range(NTH):
        for dt, wgt in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            Wsm[k * NTH + t, k * NTH + (t + dt) % NTH] = wgt
dense("rho_sm", "rho", Wsm, np.zeros(K * NTH))
reshape("rho_s", "rho_sm", [K, 2, NH])
maxpool("rho_max", "rho_s", [2, 1], [1, 1])                                   # (K, 1, 36)
avgpool("rho_avg", "rho_s", [2, 1], [1, 1])
Wsa, Wsb = np.zeros((2 * K, K)), np.zeros((2 * K, K))
for k in range(K):
    Wsa[k, k] = 2.0
    Wsb[k, k] = -1.0                                                          # min = 2*avg - max
    Wsb[K + k, k] = 1.0                                                       # max
conv1("sym_a", "rho_avg", Wsa, np.zeros(2 * K))
conv1("sym_b", "rho_max", Wsb, np.zeros(2 * K))
add("sym", ["sym_a", "sym_b"])                                                # (2K, 1, 36): rho_min_k, rho_max_k

# semi-major a_k = max over direction of rho_min_k, semi-minor b_k = min over direction
Wg = np.zeros((2 * K, 2 * K))
for k in range(K):
    Wg[k, k] = 1.0
    Wg[K + k, k] = -1.0
conv1("ang", "sym", Wg, np.zeros(2 * K))
maxpool("ab", "ang", [1, NH])                                                 # (2K,1,1): a_k, -b_k

# mirror asymmetry of the stack of isophotes: for each candidate axis n (2.5 degree steps) compare every
# smoothed radius function with its reflection; an aligned set of ellipses has an axis with no residual,
# a bar inside a disc or isophotes twisted by spiral arms do not.
MIR_LEVELS = list(range(NA))
NML = len(MIR_LEVELS)
g = np.exp(-0.5 * (np.arange(-7, 8) / 2.4) ** 2)
g /= g.sum()
Sm = np.zeros((NH, NH))
for j in range(NH):
    for d, wgt in zip(range(-7, 8), g):
        Sm[j, (j + d) % NH] += wgt
Wmir = np.zeros((NH * NML * NH, 2 * K * NH), dtype=np.float32)
for n in range(NH):
    for a, k in enumerate(MIR_LEVELS):
        for j in range(NH):
            o = (n * NML + a) * NH + j
            Wmir[o, k * NH:(k + 1) * NH] = Sm[j] - Sm[(n - j) % NH]
dense("mir", "sym", Wmir, np.zeros(NH * NML * NH))
relu("mir_r", "mir")
Wms = np.zeros((NH, NH * NML * NH), dtype=np.float32)
for n in range(NH):
    Wms[n, n * NML * NH:(n + 1) * NML * NH] = -1.0
dense("mir_sum", "mir_r", Wms, np.zeros(NH))                                  # minus the residual of each axis
reshape("mir_v", "mir_sum", [1, 1, NH])
maxpool("chi", "mir_v", [1, NH])                                              # minus the smallest residual

# ----------------------------------------------------------------------------------------------------------
# 4. gated sums: colour of regions, areas, secondary blobs
REGIONS = ("all", "out", "core")
NG = 3 * 3 * NC + 2 * NR + K
G_XENV, G_XNEAR, G_AREA = 9 * NC, 9 * NC + NR, 9 * NC + 2 * NR
WgM, WgF, WgD, bg = np.zeros((NG, NSTAR)), np.zeros((NG, NC)), np.zeros((NG, NC)), np.zeros(NG)


def region_mask(reg, i):
    """list of (channel of M, coefficient) giving the region's mask at radius index i of the fine grid"""
    if reg == "all":
        return [(1 * NR + i, 1.0)]                                   # inside the A2 isophote
    if reg == "out":
        return [(0 * NR + i, 1.0), (3 * NR + i, -1.0)]               # between the A1 and A4 isophotes
    return [(KR2 * NR + i, 1.0)]                                     # inside the R2 isophote


for a, reg in enumerate(REGIONS):
    for ci, i in enumerate(CIDX):
        for u in range(3):                                            # +colour, -colour, light
            o = (a * 3 + u) * NC + ci
            if u == 0:
                WgD[o, ci] = 1.0
            elif u == 1:
                WgD[o, ci] = -1.0
            else:
                WgF[o, ci] = 1.0
            for ch, cf in region_mask(reg, i):
                WgM[o, ch] = GATE_C * cf
            bg[o] = -GATE_C
for i in range(NR):
    for o, env in ((G_XENV + i, True), (G_XNEAR + i, False)):
        WgM[o, IDX_S + i] = 1.0
        WgM[o, IDX_G1 + i] = 1.0
        WgM[o, IDX_G2 + i] = -1.0
        bg[o] = -1.0
        if env:
            WgM[o, 0 * NR + i] = 1.0                                  # inside the faint envelope of the object
            bg[o] = -2.0
for k in range(K):
    WgM[G_AREA + k, k * NR:(k + 1) * NR] = AREA                       # area of each mask (already >= 0)
conv1("g_m", "M", WgM, bg)
conv1("g_f", "Fp", WgF, np.zeros(NG))
conv1("g_d", "Dp", WgD, np.zeros(NG))
add("g_pre", ["g_m", "g_f", "g_d"])
relu("g", "g_pre")

GS = {}
names = []
for reg in REGIONS:
    names += ["G_" + reg, "F_" + reg]
names += ["X_env", "X_near"] + ["A_%d" % k for k in range(K)]
for q, nm in enumerate(names):
    GS[nm] = q
NGS = len(names)
Wgs = np.zeros((NGS, NG))
for a, reg in enumerate(REGIONS):
    Wgs[GS["G_" + reg], (a * 3 + 0) * NC:(a * 3 + 1) * NC] = CAREA
    Wgs[GS["G_" + reg], (a * 3 + 1) * NC:(a * 3 + 2) * NC] = -CAREA
    Wgs[GS["F_" + reg], (a * 3 + 2) * NC:(a * 3 + 3) * NC] = CAREA
Wgs[GS["X_env"], G_XENV:G_XENV + NR] = AREA
Wgs[GS["X_near"], G_XNEAR:G_XNEAR + NR] = AREA * (RAD <= R_NEAR)
for k in range(K):
    Wgs[GS["A_%d" % k], G_AREA + k] = 1.0
conv1("gs", "g", Wgs, np.zeros(NGS))
avgpool("gsum", "gs", [2, NH])                                                # mean over the 72 directions
# mean over direction pairs of the larger of the two opposite rays: (gmax - gsum) is then the part of a sum
# that is NOT matched on the opposite side of the centre.  A companion nucleus sits on one side only,
# whereas a pair of spiral arms or a ring puts equal light on both sides and cancels.
maxpool("gs_mx", "gs", [2, 1], [1, 1])
avgpool("gmax", "gs_mx", [1, NH])

# ----------------------------------------------------------------------------------------------------------
# 5. thermometer bits.  Sources of scalars: ab (2K), chi (1), sym (2K*36), gsum (NGS), L0 (1)
SRC = {"ab": 2 * K, "chi": 1, "sym": 2 * K * NH, "gsum": NGS, "gmax": NGS, "L0": 1, "armavg": 2, "armmax": 2}
bit_rows = []            # each: (dict source -> vector, bias, gain)
bit_index = {}


def new_bit(name, terms, bias, gain):
    """bit = clip(gain * (sum of terms + bias), 0, 1); terms: list of (source, index or vector, coefficient)"""
    vecs = {}
    for src, idx, cf in terms:
        v = vecs.setdefault(src, np.zeros(SRC[src]))
        if np.isscalar(idx):
            v[idx] += cf
        else:
            v += cf * np.asarray(idx, dtype=np.float64)
    bit_index[name] = len(bit_rows)
    bit_rows.append((vecs, bias, gain))


def A_(k):                     # semi-major axis of level k
    return ("ab", k, 1.0)


def B_(k, cf=1.0):             # semi-minor axis of level k (ab stores -b)
    return ("ab", K + k, -cf)


def sum_min(k):                # sum over the 36 directions of rho_min_k
    v = np.zeros(2 * K * NH)
    v[k * NH:(k + 1) * NH] = 1.0
    return v


def sum_max(k):
    v = np.zeros(2 * K * NH)
    v[(K + k) * NH:(K + k + 1) * NH] = 1.0
    return v


Q_T = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.87]
COL_T = [-0.03, 0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18]
SA_T = [10.0, 14.0, 19.0, 26.0, 36.0, 50.0]
CHI_T = [0.03, 0.05, 0.07, 0.09, 0.12, 0.16]
ASY_T = [0.05, 0.08, 0.12, 0.17, 0.23, 0.30]
ISL_T = [0.08, 0.25, 0.60]
L0_T = [0.30, 0.40, 0.50, 0.60, 0.70]
QR2_T = [0.30, 0.40, 0.50, 0.60, 0.75]
LAM_T = [0.10, 0.15, 0.20, 0.27, 0.35]
ARM_T = [0.003, 0.006, 0.012, 0.025, 0.05]
BAR_OUT_T = [0.45 + 0.05 * q for q in range(10)]
BAR_IN_T = [0.10 + 0.05 * q for q in range(14)]
BAR_DELTAS = [0.15, 0.25, 0.35]
BAR_LEVELS = [2, 3, 4, 5, 6, 7, 8]

FEATS = {}                      # feature -> (valid bit name or None, [threshold bit names])


def thermo(feat, valid, bits):
    FEATS[feat] = (valid, bits)


G_SHAPE, G_COL, G_SUM, G_AREA_GAIN, G_L0 = 8.0, 30.0, 0.5, 1.0, 200.0
new_bit("v_a1", [A_(0)], -3.0, G_SHAPE)
new_bit("v_a2", [A_(1)], -3.0, G_SHAPE)
new_bit("v_r2", [A_(KR2)], -1.5, G_SHAPE)
for k in BAR_LEVELS:
    new_bit("v_bar%d" % k, [A_(k)], -4.0, G_SHAPE)
for nm, k in (("QA1", 0), ("QA2", 1)):
    for t in Q_T:
        new_bit("%s>%g" % (nm, t), [B_(k), ("ab", k, -t)], 0.0, G_SHAPE)
    thermo(nm, "v_a%d" % (k + 1), ["%s>%g" % (nm, t) for t in Q_T])
for t in SA_T:
    new_bit("SA>%g" % t, [A_(0)], -t, 4.0)
thermo("SA", "v_a1", ["SA>%g" % t for t in SA_T])
for nm, reg in (("COLO", "out"), ("COLI", "core")):
    new_bit("v_" + nm, [("gsum", GS["F_" + reg], 1.0)], -2.0, G_AREA_GAIN)
    for t in COL_T:
        new_bit("%s>%g" % (nm, t), [("gsum", GS["G_" + reg], 1.0), ("gsum", GS["F_" + reg], -3.0 * t)], 0.0, G_COL)
    thermo(nm, "v_" + nm, ["%s>%g" % (nm, t) for t in COL_T])
chi_den = sum(sum_min(k) for k in MIR_LEVELS)
new_bit("v_CHI", [("sym", chi_den, 1.0)], -150.0, G_SUM)
for t in CHI_T:
    new_bit("CHI>%g" % t, [("chi", 0, -2.0), ("sym", chi_den, -t)], 0.0, G_SUM)     # chi stores minus the residual
thermo("CHI", "v_CHI", ["CHI>%g" % t for t in CHI_T])
for t in ASY_T:
    new_bit("ASY>%g" % t, [("sym", sum_max(0), 1.0 - t), ("sym", sum_min(0), -1.0 - t)], 0.0, G_SUM)
thermo("ASY", "v_a1", ["ASY>%g" % t for t in ASY_T])
for nm, src in (("ISLE", "X_env"), ("ISLN", "X_near")):
    for t in ISL_T:
        new_bit("%s>%g" % (nm, t), [("gmax", GS[src], 1.0), ("gsum", GS[src], -1.0), ("gsum", GS["A_%d" % KR2], -t)],
                -10.0 * t, G_AREA_GAIN)
    thermo(nm, None, ["%s>%g" % (nm, t) for t in ISL_T])
for t in L0_T:
    new_bit("L0>%g" % t, [("L0", 0, 1.0)], -t, G_L0)
thermo("L0", None, ["L0>%g" % t for t in L0_T])
for t in QR2_T:
    new_bit("QR2>%g" % t, [B_(KR2), ("ab", KR2, -t)], 0.0, G_SHAPE)
thermo("QR2", "v_r2", ["QR2>%g" % t for t in QR2_T])
for t in LAM_T:
    new_bit("LAM>%g" % t, [A_(KR2), ("ab", 0, -t)], 0.0, G_SHAPE)
thermo("LAM", "v_r2", ["LAM>%g" % t for t in LAM_T])
new_bit("v_ARM", [("armavg", 1, 1.0)], -1.0, 2.0)
for t in ARM_T:
    new_bit("ARM>%g" % t, [("armavg", 0, 2.0), ("armmax", 0, -1.0), ("armavg", 1, -t)], 0.0, 20.0)
thermo("ARM", "v_ARM", ["ARM>%g" % t for t in ARM_T])
for t in BAR_OUT_T:
    new_bit("BO>%.2f" % t, [B_(0), ("ab", 0, -t)], 0.0, G_SHAPE)
for k in BAR_LEVELS:
    for t in BAR_IN_T:
        new_bit("BI%d>%.2f" % (k, t), [B_(k), ("ab", k, -t)], 0.0, G_SHAPE)

NB = len(bit_rows)
Wth = {s: np.zeros((2 * NB, n)) for s, n in SRC.items()}
bth = np.zeros(2 * NB)
for q, (vecs, bias, gain) in enumerate(bit_rows):
    for s, v in vecs.items():
        Wth[s][2 * q] = gain * v
        Wth[s][2 * q + 1] = gain * v
    bth[2 * q] = gain * bias
    bth[2 * q + 1] = gain * bias - 1.0
dense("th_ab", "ab", Wth["ab"], bth)
dense("th_chi", "chi", Wth["chi"], np.zeros(2 * NB))
dense("th_sym", "sym", Wth["sym"], np.zeros(2 * NB))
dense("th_gs", "gsum", Wth["gsum"], np.zeros(2 * NB))
dense("th_gm", "gmax", Wth["gmax"], np.zeros(2 * NB))
dense("th_l0", "L0", Wth["L0"], np.zeros(2 * NB))
dense("th_aa", "armavg", Wth["armavg"], np.zeros(2 * NB))
dense("th_am", "armmax", Wth["armmax"], np.zeros(2 * NB))
add("th_pre", ["th_ab", "th_chi", "th_sym", "th_gs", "th_gm", "th_l0", "th_aa", "th_am"])
relu("th", "th_pre")

# hidden layer: every bit, plus the bar conjunctions
# [outer isophote rounder than t] and [inner isophote k flatter than t - delta] and [inner isophote resolved]
conj = [(k, t, d) for d in BAR_DELTAS for k in BAR_LEVELS for t in BAR_OUT_T]
NHID = NB + len(conj)
Wh, bh = np.zeros((NHID, 2 * NB)), np.zeros(NHID)
for q in range(NB):
    Wh[q, 2 * q], Wh[q, 2 * q + 1] = 1.0, -1.0


def put(row, bitname, cf):
    q = bit_index[bitname]
    Wh[row, 2 * q] += cf
    Wh[row, 2 * q + 1] -= cf


for c, (k, t, d) in enumerate(conj):
    row = NB + c
    put(row, "BO>%.2f" % t, 1.0)
    put(row, "BI%d>%.2f" % (k, t - d), -1.0)
    put(row, "v_bar%d" % k, 1.0)
    bh[row] = -1.0
dense("hid_pre", "th", Wh, bh)
relu("hid", "hid_pre")

Wbar, bbar = np.zeros((2 * len(BAR_DELTAS), NHID)), np.zeros(2 * len(BAR_DELTAS))
for c, (k, t, d) in enumerate(conj):
    q = BAR_DELTAS.index(d)
    Wbar[2 * q, NB + c] = 1.0
    Wbar[2 * q + 1, NB + c] = 1.0
for q in range(len(BAR_DELTAS)):
    bbar[2 * q + 1] = -1.0
dense("bar_pre", "hid", Wbar, bbar)
relu("bar", "bar_pre")

# ----------------------------------------------------------------------------------------------------------
# 6. rule tables.  Rows: probability of each bin of a measurement for each class, typed by hand.
# classes: 0 disturbed, 1 merging, 2 round smooth, 3 in-between smooth, 4 cigar smooth, 5 barred spiral,
#          6 tight spiral, 7 loose spiral, 8 edge-on no bulge, 9 edge-on with bulge
PRIOR = [0.061, 0.104, 0.149, 0.114, 0.019, 0.115, 0.103, 0.148, 0.080, 0.106]

# Only the rows of the eight host classes (2..9) are used for shape / colour measurements; the merging and
# disturbed classes borrow the best host (see below) and have rows of their own only for the interaction tables.
T_Q = {  # axis ratio of an outer isophote; bins <.2 .2-.25 .25-.3 .3-.35 .35-.4 .4-.45 .45-.5 .5-.6 .6-.7 .7-.8 .8-.87 >.87
    # smooth classes follow the Galaxy Zoo roundness answers; discs follow random inclinations, with arms and
    # bars hard to see when strongly inclined; edge-on discs are thin, thicker when a bulge is present
    2: [.001, .001, .001, .001, .002, .004, .005, .015, .04, .15, .33, .45],
    3: [.001, .002, .004, .01, .02, .04, .07, .25, .30, .22, .07, .013],
    4: [.03, .07, .17, .22, .20, .14, .08, .06, .02, .006, .002, .002],
    5: [.002, .003, .005, .01, .02, .04, .06, .15, .20, .21, .15, .15],
    6: [.003, .007, .02, .04, .06, .08, .09, .18, .17, .15, .10, .10],
    7: [.004, .006, .012, .025, .04, .06, .075, .17, .19, .18, .12, .118],
    8: [.33, .27, .20, .10, .05, .025, .012, .008, .003, .001, .001, .001],
    9: [.15, .24, .27, .18, .09, .04, .015, .008, .003, .001, .001, .001],
}
T_SA = {  # semi-major axis of the faintest isophote, px; bins <10 10-14 14-19 19-26 26-36 36-50 >50
    2: [.10, .30, .33, .20, .05, .015, .005],
    3: [.07, .25, .33, .23, .09, .02, .01],
    4: [.03, .17, .30, .30, .14, .05, .01],
    5: [.02, .10, .23, .30, .22, .10, .03],
    6: [.03, .15, .27, .28, .17, .08, .02],
    7: [.03, .14, .25, .28, .18, .09, .03],
    8: [.01, .05, .15, .27, .27, .17, .08],
    9: [.01, .04, .14, .27, .28, .18, .08],
}
# Colour index = sum(R - B) / sum(R + G + B - 3 sky).  With the viewer's band scales (g 6.0, r 3.4, z 2.2) a
# red-sequence galaxy (g-r 0.8..1.0, r-z 0.7) comes out at +0.13..+0.25 and a star-forming disc
# (g-r 0.35..0.6, r-z 0.3..0.45) at -0.15..0.0; early-type spiral discs and dusty edge-on discs lie between.
T_COLO = {  # colour index of the outskirts; bins <-.03 -.03-0 0-.03 .03-.06 .06-.09 .09-.12 .12-.15 .15-.18 >.18
    2: [.005, .01, .02, .04, .09, .17, .25, .22, .195],
    3: [.005, .01, .025, .05, .10, .18, .25, .21, .17],
    4: [.01, .02, .04, .08, .13, .19, .22, .18, .13],
    5: [.13, .16, .19, .18, .14, .10, .05, .03, .02],
    6: [.17, .19, .20, .17, .12, .08, .04, .02, .01],
    7: [.38, .22, .16, .10, .06, .04, .02, .01, .01],
    8: [.12, .13, .15, .16, .15, .12, .08, .05, .04],
    9: [.02, .03, .06, .10, .15, .20, .20, .14, .10],
}
T_COLI = {  # colour index of the core (bright cores can clip in the red channel and look whiter)
    2: [.005, .01, .02, .05, .10, .17, .24, .22, .185],
    3: [.005, .01, .02, .05, .10, .17, .24, .22, .185],
    4: [.005, .01, .02, .05, .10, .17, .24, .22, .185],
    5: [.02, .03, .06, .10, .16, .20, .20, .14, .09],
    6: [.03, .05, .09, .14, .18, .19, .16, .10, .06],
    7: [.10, .13, .17, .18, .16, .12, .08, .04, .02],
    8: [.06, .08, .13, .17, .18, .15, .11, .07, .05],
    9: [.01, .02, .04, .08, .14, .20, .22, .17, .12],
}
T_CHI = {  # mirror asymmetry of the isophote stack; bins <.03 .03-.05 .05-.07 .07-.09 .09-.12 .12-.16 >.16
    # small face-on spirals have nearly circular isophotes whatever their arms do, so a low value says little;
    # a high value means twisted isophotes: open arms or a bar
    2: [.38, .33, .16, .07, .03, .02, .01],
    3: [.38, .33, .16, .07, .03, .02, .01],
    4: [.33, .32, .18, .09, .04, .02, .02],
    5: [.18, .24, .20, .14, .11, .08, .05],
    6: [.26, .28, .20, .12, .07, .04, .03],
    7: [.18, .23, .19, .14, .11, .08, .07],
    8: [.25, .27, .20, .12, .08, .05, .03],
    9: [.25, .27, .20, .12, .08, .05, .03],
}
T_ASY = {  # lopsidedness of the faintest isophote; bins <.05 .05-.08 .08-.12 .12-.17 .17-.23 .23-.3 >.3
    0: [.08, .14, .20, .20, .16, .12, .10],
    1: [.05, .08, .13, .18, .20, .18, .18],
    2: [.30, .30, .20, .10, .05, .03, .02],
    3: [.30, .30, .20, .10, .05, .03, .02],
    4: [.30, .30, .20, .10, .05, .03, .02],
    5: [.24, .28, .22, .13, .07, .04, .02],
    6: [.25, .29, .22, .12, .07, .03, .02],
    7: [.17, .24, .23, .16, .10, .06, .04],
    8: [.28, .28, .20, .11, .06, .04, .03],
    9: [.28, .28, .20, .11, .06, .04, .03],
}
T_ISLE = {  # one-sided bright second blob inside the envelope, area relative to the core; bins <.08 .08-.25 .25-.6 >.6
    0: [.86, .06, .05, .03], 1: [.45, .12, .18, .25], 2: [.94, .03, .02, .01], 3: [.94, .03, .02, .01],
    4: [.94, .03, .02, .01], 5: [.92, .04, .025, .015], 6: [.92, .04, .025, .015], 7: [.88, .06, .04, .02],
    8: [.94, .03, .02, .01], 9: [.94, .03, .02, .01],
}
T_ISLN = {  # one-sided bright second blob within 30 px
    0: [.78, .08, .08, .06], 1: [.30, .12, .20, .38], 2: [.82, .07, .06, .05], 3: [.82, .07, .06, .05],
    4: [.82, .07, .06, .05], 5: [.82, .07, .06, .05], 6: [.82, .07, .06, .05], 7: [.78, .08, .08, .06],
    8: [.82, .07, .06, .05], 9: [.82, .07, .06, .05],
}
T_L0 = {  # central brightness above sky; bins <.3 .3-.4 .4-.5 .5-.6 .6-.7 >.7
    2: [.02, .08, .20, .30, .25, .15],
    3: [.02, .07, .18, .30, .26, .17],
    4: [.02, .08, .20, .30, .25, .15],
    5: [.06, .17, .27, .26, .16, .08],
    6: [.05, .15, .26, .27, .18, .09],
    7: [.15, .25, .27, .19, .10, .04],
    8: [.15, .27, .28, .18, .09, .03],
    9: [.03, .10, .22, .30, .23, .12],
}
_qr2_faceon = [.01, .03, .08, .18, .38, .32]
T_QR2 = {  # axis ratio of the isophote at peak - 0.20; bins <.3 .3-.4 .4-.5 .5-.6 .6-.75 >.75
    # used to tell a thin disc (flat at every level) from a disc with a bulge (round in the middle)
    2: _qr2_faceon, 3: _qr2_faceon, 5: _qr2_faceon, 6: _qr2_faceon, 7: _qr2_faceon,
    4: [.10, .25, .30, .20, .12, .03],
    8: [.40, .30, .17, .08, .04, .01],
    9: [.07, .17, .25, .24, .20, .07],
}
T_LAM = {  # length of the bright core / length of the faintest isophote; bins <.1 .1-.15 .15-.2 .2-.27 .27-.35 >.35
    # a compact bright bulge inside a long faint disc gives a small ratio; a pure exponential disc about 0.25-0.35
    2: [.05, .15, .28, .30, .16, .06],
    3: [.05, .15, .28, .30, .16, .06],
    4: [.05, .15, .28, .30, .16, .06],
    5: [.12, .25, .27, .20, .11, .05],
    6: [.12, .25, .27, .20, .11, .05],
    7: [.04, .12, .22, .28, .21, .13],
    8: [.02, .08, .20, .30, .25, .15],
    9: [.15, .30, .27, .17, .08, .03],
}
T_ARM = {  # two-sided bump flux / flux of the object; bins <.003 .003-.006 .006-.012 .012-.025 .025-.05 >.05
    # smooth galaxies have none; many small spirals show none either, so "none" is only mild evidence
    2: [.86, .07, .04, .02, .007, .003], 3: [.86, .07, .04, .02, .007, .003], 4: [.86, .07, .04, .02, .007, .003],
    5: [.45, .14, .14, .13, .09, .05], 6: [.45, .14, .14, .13, .09, .05], 7: [.40, .14, .15, .14, .10, .07],
    8: [.60, .14, .11, .08, .05, .02], 9: [.65, .13, .10, .07, .035, .015],
}
T_BAR = {  # an inner isophote flatter than the outer one by: <.15, .15-.25, .25-.35, >.35
    # seeing only ever makes inner isophotes rounder, and ellipticals keep or increase their flattening outward,
    # so a clearly flatter inner isophote means a bar (or the inner part of open arms)
    2: [.96, .025, .01, .005], 3: [.96, .025, .01, .005], 4: [.97, .02, .005, .005],
    5: [.42, .20, .19, .19], 6: [.82, .10, .05, .03], 7: [.75, .12, .08, .05],
    8: [.92, .04, .02, .02], 9: [.92, .04, .02, .02],
}
# "Merging" and "disturbed" are not shapes of their own: Galaxy Zoo asks about them separately from the shape
# questions, so such a galaxy is some ordinary host (round, spiral, edge-on ...) plus signs of interaction.
# Their score is therefore: the best host explanation of the shape / colour measurements (weighted by how
# often each host type occurs among such systems) plus the evidence from the interaction measurements.
GENERIC = {"QA1": (T_Q, 0.5), "QA2": (T_Q, 0.5), "SA": (T_SA, 0.4), "COLO": (T_COLO, 1.0), "COLI": (T_COLI, 0.5),
           "CHI": (T_CHI, 0.7), "L0": (T_L0, 0.5), "QR2": (T_QR2, 0.5), "LAM": (T_LAM, 0.5), "ARM": (T_ARM, 1.0)}
SPECIFIC = {"ASY": (T_ASY, 0.7), "ISLE": (T_ISLE, 1.0), "ISLN": (T_ISLN, 0.6)}
HOSTS = [2, 3, 4, 5, 6, 7, 8, 9]
HOST_MIX = {1: [.10, .15, .02, .08, .12, .33, .08, .12],     # host types of the central galaxy of a merger
            0: [.12, .22, .03, .08, .13, .27, .05, .10]}     # ... of a disturbed galaxy
MODS = [1, 0]
NOUT = 10 + len(MODS) * len(HOSTS)


def logtab(tab):
    n = len(next(iter(tab.values())))
    out = np.zeros((10, n))
    for c in tab:
        p = np.asarray(tab[c], dtype=np.float64)
        out[c] = np.log(p / p.sum())
    return out


def rows_for(c, generic):
    """output rows that receive class c's table for a generic / specific measurement"""
    if not generic:
        return [c]
    if c not in HOSTS:
        return []
    return [c] + [10 + m * len(HOSTS) + HOSTS.index(c) for m in range(len(MODS))]


Wo, bo = np.zeros((NOUT, NHID)), np.zeros(NOUT)
bo[:10] = np.log(np.asarray(PRIOR) / np.sum(PRIOR))
for m, mod in enumerate(MODS):
    bo[10 + m * len(HOSTS):10 + (m + 1) * len(HOSTS)] = np.log(np.asarray(HOST_MIX[mod]) / np.sum(HOST_MIX[mod]))
for group, generic in ((GENERIC, True), (SPECIFIC, False)):
    for feat, (tab, wt) in group.items():
        valid, bits = FEATS[feat]
        lt = logtab(tab)
        assert lt.shape[1] == len(bits) + 1, feat
        for c in range(10):
            for row in rows_for(c, generic):
                if valid is None:
                    bo[row] += wt * lt[c, 0]
                else:
                    Wo[row, bit_index[valid]] += wt * lt[c, 0]
                for j, nm in enumerate(bits):
                    Wo[row, bit_index[nm]] += wt * (lt[c, j + 1] - lt[c, j])
# nothing bright at the very centre: the catalogue position sits between two objects, or on a faint smudge
EMPTY_BONUS = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0])
Wo[:10, bit_index["v_a1"]] -= EMPTY_BONUS
bo[:10] += EMPTY_BONUS
dense("out_a", "hid", Wo, bo)

lb = logtab(T_BAR)
Wob, bob = np.zeros((NOUT, 2 * len(BAR_DELTAS))), np.zeros(NOUT)
for c in HOSTS:
    for row in rows_for(c, True):
        bob[row] = lb[c, 0]
        for q in range(len(BAR_DELTAS)):
            Wob[row, 2 * q] = lb[c, q + 1] - lb[c, q]
            Wob[row, 2 * q + 1] = -(lb[c, q + 1] - lb[c, q])
dense("out_b", "bar", Wob, bob)
add("pre_scores", ["out_a", "out_b"])

Wsel = np.zeros((len(MODS) * len(HOSTS), NOUT))
for q in range(len(MODS) * len(HOSTS)):
    Wsel[q, 10 + q] = 1.0
dense("host_sel", "pre_scores", Wsel, np.zeros(len(MODS) * len(HOSTS)))
reshape("host_r", "host_sel", [len(MODS), 1, len(HOSTS)])
maxpool("host_best", "host_r", [1, len(HOSTS)])                                # best host explanation
Wmx = np.zeros((10, len(MODS)))
for m, mod in enumerate(MODS):
    Wmx[mod, m] = 1.0
dense("mod_scores", "host_best", Wmx, np.zeros(10))
Wcl = np.zeros((10, NOUT))
for c in range(10):
    Wcl[c, c] = 1.0
dense("cls_scores", "pre_scores", Wcl, np.zeros(10))
add("scores", ["cls_scores", "mod_scores"])

# ----------------------------------------------------------------------------------------------------------
import os
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": layers}, f, indent=1)
np.savez(os.path.join(OUT, "weights.npz"), **weights)
print("layers:", len(layers), " weights:", sum(int(v.size) for v in weights.values()))
