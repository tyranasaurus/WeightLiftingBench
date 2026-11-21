"""Hand-built galaxy morphology network.  python3 build.py OUT_DIR  writes OUT_DIR/network.json and OUT_DIR/weights.npz.

Every weight below comes from a formula or a hand-chosen constant (binomial smoothing kernels, ray/wedge integration
weights, thresholds, rule tables).  Nothing is read from disk and nothing is fitted.

Pipeline (all built from conv / dense / relu / pool / add / reshape / norm):
  1. luminance minus the sky pedestal, binomial smoothing, per-image noise estimate (mean |high-pass|)
  2. fuzzy sky-threshold mask at half resolution and a centre-seeded flood fill (iterated 3x3 max-dilation
     clipped by the mask) -> the connected component that contains the image centre
  3. the smoothed luminance gated by that component (space-to-depth layout so a 64x64 mask gates 128x128 pixels)
  4. isophote masks at one absolute and several peak-relative levels; ray sums (radial extents in 64 directions)
     and areas of each isophote; chord min / max / lopsidedness / m=2 direction per level
  5. difference-of-Gaussians ridge map (arms, bars, clumps), second-nucleus blob strength, colours
  6. ratios through `norm` (a channel laid out as [x, -x, K*D, -K*D] is normalised to x / D)
  7. hand-written fuzzy rules -> ten class scores
"""
import sys, json
import numpy as np

BG = 0.125            # sky pedestal of the viewer's arcsinh mapping: arcsinh(20*0.03)/sqrt(20) = 0.127 (minus a little for noise)
NRAY = 64             # ray directions (32 chords)
NIT = 28              # reconstruction iterations at 64x64 (reach ~56 full-resolution pixels from the centre)
KDOM = 16.0           # dominance factor of the denominators in the norm-ratio trick
GP = 8                # positions per ratio group
LEVEL_F = [0.25, 0.40, 0.55, 0.70]     # peak-relative isophote levels (level 0 is the absolute outer isophote)
NLEV = 1 + len(LEVEL_F)
T0, W0 = 0.04, 0.015  # outer isophote: 0.04 above sky, soft ramp of +-0.015
WREL = 0.015          # half-width of the soft ramp of the relative isophotes
GAMMA, WG = 0.7, 0.005    # a pixel block belongs to isophote T if it is connected to the centre at smoothed level >= GAMMA*T
TAU_R, DEL_R = 0.012, 0.012   # ridge (difference of Gaussians) threshold and ramp
KN_MASK, KN_RIDGE = 5.0, 3.5  # noise multiples that may raise the mask / ridge thresholds
N_FLOOR = 0.02        # ... only when that noise multiple exceeds this floor (mask) / TAU_R (ridge)
R_CEN = 6.0           # central region (full-resolution pixels): defines the central peak height H
R_CORE = 5.5          # nucleus region excluded from the ridge count
RP_IN, RP_OUT = 3.5, 24.0     # half-resolution radii: central zone (central peak-ness) / limit of the second-nucleus search
RHO_LEVELS, KW = (0, 1), 10   # isophotes on which the bulge shape index rho is measured; chord offset (x 5.625 degrees)
DA_LO, DA_HI = 0.02, 0.04     # soft threshold on the dome height when measuring dome area
F_PEAK = 0.45         # a second nucleus must reach this fraction of the central peak height (smoothed, half res)
F_CONN = 0.15         # ... and be joined to the centre at a level of at least this fraction of the central peak height
TP, DP = 0.02, 0.02   # peak-ness ramp used to blank compact sources (stars, nuclei) out of the ridge count
C_OFF = 0.1           # offset that keeps gated colour planes positive (removed again in the sums)
R_CCEN, R_COUT = 2.2, 3.5             # half-resolution radii of the "central" and "outer" colour regions


# ---- rule table constants (hand-chosen; see method.md) ---------------------------------------------------------
# each pair is (value where the soft indicator is 0, value where it is 1)
Z_CI = (-0.11, -0.04)     # colour index (blue-red)/(sum): red early types ~ -0.15, star-forming discs ~ 0
Z_RF = (0.010, 0.040)     # fraction of the galaxy area covered by ridges (arms)
Z_RA = (0.24, 0.36)       # area of the 0.25-peak isophote / area of the outer isophote (unused: weight 0)
Z_A0 = (11.0, 21.0)       # semi-major axis of the outer isophote in pixels: resolved arms need a big galaxy
Z_LOP = (0.03, 0.08)      # lopsidedness of the 0.25-peak isophote
Z_H = (0.75, 0.50)        # low central surface brightness: not an elliptical
ZW = (1.0, 1.0, 0.0, 0.8, 0.3, 0.4)   # weights of the six cues
Z_HALF, Z_SHIFT = 0.6, 0.0
L_CI = (-0.04, 0.03)      # loose arms: bluer,
L_RF = (0.02, 0.06)       # more ridge area,
L_LOP = (0.04, 0.10)      # more lopsided,
L_H = (0.80, 0.55)        # weaker nucleus
LW = (0.8, 0.6, 0.5, 0.5)
L_HALF, L_SHIFT = 0.6, 0.0
M_PO = (0.03, 0.07)       # dome height of a second nucleus (smoothed luminance units)
M_AK, M_AD = 36.0, (0.0, 1.5)     # dome area (half-res px) minus M_AK * dome height: a Gaussian point source gives area ~ 20-31 x height,
                                  # a companion galaxy 44 x or more (worked out on rendered point sources / Sersic profiles)
QB_W = (0.4, 0.3, 0.3)    # body axis ratio = weighted mean over the outer, 0.25-peak and 0.40-peak isophotes
Q_ROUND = (0.80, 0.88)    # body axis ratio: round above, in-between below
Q_CIGAR = (0.56, 0.46)    # cigar below
Q_EDGE = (0.38, 0.28)     # thin disc below
A_BIG = (18.0, 30.0)      # cigars are small; long thin things are edge-on discs
A_INV = (4.0, 2.0)        # nothing at the image centre
B_Q = (0.35, 0.55)        # axis ratio of the 0.7-peak isophote: a bare thin disc stays thin, a bulge is round
B_RS = (0.34, 0.22)       # length of the 0.7-peak isophote / outer isophote: a bare exponential disc gives >~0.3, a bulge less
B_RHO = (1.25, 1.08)      # shape index rho (see build): thin ellipse ~1.7, diamond (exponential disc) ~1.3, central bump ~1.0
B_MIX = (0.4, 0.4, 0.2)   # bulge indicator = weighted mean of the three cues
BAR_DQ, BAR_TW, BAR_LO, BAR_HI = 0.10, 0.1, 0.10, 0.35
D_LOP = (0.25, 0.45)
W_MAIN, W_SUB, W_BAR, W_BARX, W_MERGE, W_DIST, W_INV = 3.0, 2.0, 2.5, 1.0, 4.0, 3.5, 6.0
BIAS = [0.0, 0.5, 0.2, 0.0, -1.0, 0.0, 0.0, 0.2, -0.1, 0.0]


class Net:
    def __init__(self):
        self.layers, self.w, self.sh = [], {}, {"image": (3, 128, 128)}

    def _add(self, name, op, inputs, oshape, **kw):
        assert name not in self.sh, name
        L = {"name": name, "op": op, "inputs": list(inputs)}
        L.update(kw)
        self.layers.append(L)
        self.sh[name] = tuple(int(v) for v in oshape)
        return name

    def conv(self, name, inp, W, b=None, stride=1, padding=0):
        W = np.asarray(W, np.float64)
        o, i, k, k2 = W.shape
        c, h, w = self.sh[inp]
        assert i == c and k == k2, (name, W.shape, self.sh[inp])
        b = np.zeros(o) if b is None else np.asarray(b, np.float64).reshape(o)
        self.w[name + ".W"], self.w[name + ".b"] = W.astype(np.float32), b.astype(np.float32)
        ho, wo = (h + 2 * padding - k) // stride + 1, (w + 2 * padding - k) // stride + 1
        return self._add(name, "conv", [inp], (o, ho, wo), out=int(o), kernel=[int(k), int(k)], stride=int(stride), padding=int(padding))

    def conv1(self, name, inp, M, b=None, padding=0):
        M = np.asarray(M, np.float64)
        return self.conv(name, inp, M[:, :, None, None], b, 1, padding)

    def dense(self, name, inp, W, b=None):
        W = np.asarray(W, np.float64)
        o, i = W.shape
        assert i == int(np.prod(self.sh[inp])), (name, W.shape, self.sh[inp])
        b = np.zeros(o) if b is None else np.asarray(b, np.float64).reshape(o)
        self.w[name + ".W"], self.w[name + ".b"] = W.astype(np.float32), b.astype(np.float32)
        return self._add(name, "dense", [inp], (o,), out=int(o))

    def relu(self, name, inp):
        return self._add(name, "relu", [inp], self.sh[inp])

    def norm(self, name, inp):
        return self._add(name, "norm", [inp], self.sh[inp])

    def pool(self, name, inp, kind, size, stride=None):
        stride = size if stride is None else stride
        c, h, w = self.sh[inp]
        ho, wo = (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1
        return self._add(name, kind, [inp], (c, ho, wo), size=[int(size[0]), int(size[1])], stride=[int(stride[0]), int(stride[1])])

    def add(self, name, inps):
        for i in inps:
            assert self.sh[i] == self.sh[inps[0]], (name, [self.sh[j] for j in inps])
        return self._add(name, "add", inps, self.sh[inps[0]])

    def reshape(self, name, inp, shape):
        assert int(np.prod(shape)) == int(np.prod(self.sh[inp])), (name, shape, self.sh[inp])
        return self._add(name, "reshape", [inp], shape, shape=[int(v) for v in shape])

    def size(self, name):
        return int(np.prod(self.sh[name]))


# ----------------------------------------------------------------------------------------------------------------
# formula helpers
def binom(k):
    v = np.array([1.0])
    for _ in range(k - 1):
        v = np.convolve(v, [0.5, 0.5])
    return np.outer(v, v)


def conv2_full(a, b):
    out = np.zeros((a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1))
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            out[i:i + a.shape[0], j:j + a.shape[1]] += a * b[i, j]
    return out


def s2d(m):
    """(H, W) full-resolution map -> (4, H/2, W/2) with phase p = 2*dy + dx."""
    return np.stack([m[dy::2, dx::2] for dy in (0, 1) for dx in (0, 1)])


def radius_map(n=128):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    c = (n - 1) / 2.0
    return np.hypot(x - c, y - c), np.arctan2(y - c, x - c)


def ray_weights(nray=NRAY, n=128, ss=4):
    """w[k] integrates a map along direction k:  sum_x w[k, x] M(x) ~ radial extent of M in a wedge around angle
    2 pi k / nray.  Triangular angular window (half-width dphi, widened to ~one pixel near the centre so the weights
    stay smooth), 1 / (2 pi r) radial weight shared out over the rays, ss x ss sub-pixel sampling."""
    c = (n - 1) / 2.0
    sub = (np.arange(ss) + 0.5) / ss - 0.5
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    dphi = 2 * np.pi / nray
    ang = np.arange(nray) * dphi
    w = np.zeros((nray, n, n))
    for sy in sub:
        for sx in sub:
            dx, dy = x + sx - c, y + sy - c
            r = np.maximum(np.hypot(dx, dy), 0.25)
            phi = np.arctan2(dy, dx)
            hw = np.minimum(np.maximum(dphi, 1.0 / r), np.pi)
            dd = np.abs((ang[:, None, None] - phi[None] + np.pi) % (2 * np.pi) - np.pi)
            ker = np.maximum(0.0, 1.0 - dd / hw[None])
            ker /= ker.sum(0, keepdims=True)
            w += ker * (nray / (2 * np.pi * r))[None] / (ss * ss)
    return w


# ----------------------------------------------------------------------------------------------------------------
class Scal:
    """Registry of raw scalars (entries of flattened layer outputs) and linear expressions over them."""

    def __init__(self):
        self.src = {}          # raw name -> (layer, index)

    def reg(self, layer, names):
        for i, nm in enumerate(names):
            if nm is not None:
                assert nm not in self.src, nm
                self.src[nm] = (layer, i)


def lin_layers(net, prefix, exprs, scal):
    """exprs: list of dicts {raw_name: coef, '1': const}.  Builds one dense per source layer plus an add, returning a
    layer whose output vector holds the expressions."""
    by = {}
    const = np.zeros(len(exprs))
    for r, e in enumerate(exprs):
        for nm, cf in e.items():
            if nm == "1":
                const[r] += cf
                continue
            layer, idx = scal.src[nm]
            by.setdefault(layer, np.zeros((len(exprs), net.size(layer))))[r, idx] += cf
    outs = []
    for n_, (layer, M) in enumerate(sorted(by.items())):
        outs.append(net.dense(f"{prefix}_{layer}", layer, M, const if n_ == 0 else None))
    return outs[0] if len(outs) == 1 else net.add(prefix, outs)


def build(with_rules=True):
    net = Net()
    sc = Scal()
    b5, b3 = binom(5), binom(3)
    rmap, _ = radius_map()
    r64, _ = radius_map(64)
    s2dk = np.zeros((4, 1, 2, 2))
    for p in range(4):
        s2dk[p, 0, p // 2, p % 2] = 1.0

    def clip_pair(prefix, src, Wx, bx, Wb, nmap):
        """soft threshold maps  clip(Wx.src + bx + Wb.[H, s_mask, s_ridge], 0, 1)  ->  conv, broadcast, add, relu.
        Returns the relu layer holding pairs (u, u-1); the caller differences them."""
        Wx, bx, Wb = np.asarray(Wx, float), np.asarray(bx, float), np.asarray(Wb, float)
        W2 = np.repeat(Wx, 2, 0)
        b2 = np.repeat(bx, 2) - np.tile([0.0, 1.0], len(bx))
        net.conv1(prefix + "_x", src, W2, b2)
        net.dense(prefix + "_b", "hs", np.repeat(np.repeat(Wb, 2, 0), nmap, 0))
        net.reshape(prefix + "_br", prefix + "_b", (2 * len(bx), 64, 64))
        net.add(prefix + "_a", [prefix + "_x", prefix + "_br"])
        return net.relu(prefix + "_r", prefix + "_a")

    def pair_diff(n, reps=1, bias=0.0):
        """matrix that turns n (u, u-1) relu pairs into n clipped values, each repeated `reps` times."""
        M = np.zeros((n * reps, 2 * n))
        for c in range(n):
            for r_ in range(reps):
                M[c * reps + r_, 2 * c], M[c * reps + r_, 2 * c + 1] = 1.0, -1.0
        return M, np.full(n * reps, bias)

    # ---- 1. luminance, smoothing, noise ------------------------------------------------------------------------
    net.conv1("lum", "image", np.full((1, 3), 1 / 3.0), [-BG])                       # X = L - sky
    net.conv("xs", "lum", b5[None, None], padding=2)                                 # sigma^2 = 1
    hpk = -b5.copy(); hpk[2, 2] += 1.0
    net.conv("hp", "lum", np.stack([hpk, -hpk])[:, None], padding=2)                 # +-(X - smooth X)
    net.relu("hpr", "hp")
    net.pool("hpm", "hpr", "avgpool", [128, 128])                                    # (2,1,1): mean|hp| = sum
    # white-noise bookkeeping: sigma_L = mean|hp| / (sqrt(2/pi) |hpk|); smoothed-noise sigmas follow from kernel norms
    sig_per_n = 1.0 / (np.sqrt(2 / np.pi) * np.sqrt((hpk ** 2).sum()))
    box = np.full((2, 2), 0.25)
    b3up = np.zeros((5, 5)); b3up[::2, ::2] = b3                                     # 64-res 3x3 binomial seen at full res
    g64 = np.sqrt((conv2_full(conv2_full(b5, box), b3up) ** 2).sum())                # noise gain of the half-res image
    dk = conv2_full(conv2_full(b5, b5), b5)
    dk = -dk; dk[4:9, 4:9] += b5
    gdog = np.sqrt((dk ** 2).sum())                                                  # noise gain of the DoG map
    net.dense("sn", "hpm", [[KN_MASK * g64 * sig_per_n] * 2, [KN_RIDGE * gdog * sig_per_n] * 2], [-N_FLOOR, -TAU_R])
    net.relu("snr", "sn")                                                            # [s_mask, s_ridge] >= 0
    sc.reg("snr", ["s_mask", "s_ridge"])

    # ---- 2. grayscale reconstruction from the centre at half resolution -----------------------------------------
    # R(x) = the highest level at which x is connected to the central block = lim m <- min(X, dilate(m))
    net.pool("x64", "xs", "avgpool", [2, 2])
    net.conv("x64s", "x64", b3[None, None], padding=1)
    net.relu("x64p", "x64s")
    seedk = np.zeros((1, 1, 61, 61)); seedk[0, 0, 30, 30] = 1.0
    net.conv("seed", "x64p", seedk)                                                  # central 4x4 block
    net.conv1("ff0_c", "seed", [[1.0], [1.0]], padding=31)                           # (2,66,66), zero outside the block
    net.conv1("qmap", "x64p", [[0.0], [-1.0]])
    prev = "ff0_c"
    for i in range(1, NIT + 1):
        net.pool(f"ff{i}_d", prev, "maxpool", [3, 3], [1, 1])
        net.add(f"ff{i}_a", [f"ff{i}_d", "qmap"])
        net.relu(f"ff{i}_r", f"ff{i}_a")                                             # [D, relu(D - X)]
        if i < NIT:
            prev = net.conv1(f"ff{i}_c", f"ff{i}_r", [[1.0, -1.0], [1.0, -1.0]], padding=1)   # min(D, X), re-padded
    net.conv1("r64", f"ff{NIT}_r", [[1.0, -1.0]])                                    # reconstruction R, (1,64,64)

    # ---- 3. central peak height H ---------------------------------------------------------------------------------
    net.conv("xs2d", "xs", s2dk, stride=2)                                           # full-res smoothed X, (4,64,64)
    cen = s2d((rmap < R_CEN).astype(float))
    net.dense("hc_k", "snr", np.zeros((4 * 4096, 2)), cen.reshape(-1) - 1.0)
    net.reshape("hc_kr", "hc_k", (4, 64, 64))
    net.add("hc_a", ["xs2d", "hc_kr"])
    net.relu("hc_r", "hc_a")
    net.pool("h_mp", "hc_r", "maxpool", [64, 64])
    net.reshape("h_rs", "h_mp", (1, 4, 1))
    net.pool("h_mx", "h_rs", "maxpool", [4, 1])                                      # H: central peak height above sky
    sc.reg("h_mx", ["H"])
    assert lin_layers(net, "hs", [{"H": 1.0}, {"s_mask": 1.0}, {"s_ridge": 1.0}], sc) == "hs"   # [H, s_mask, s_ridge]

    # ---- 4. isophote masks (connected to the centre), ray sums ---------------------------------------------------
    Wx, bx, Wb, Wgx, bgx, Wgb = [], [], [], [], [], []
    for j in range(NLEV):
        hw = W0 if j == 0 else WREL
        for p in range(4):
            Wx.append(np.eye(4)[p] / (2 * hw))
            bx.append(0.5 - (T0 / (2 * hw) if j == 0 else 0.0))
            Wb.append([0.0, -1 / (2 * hw), 0.0] if j == 0 else [-LEVEL_F[j - 1] / (2 * hw), 0.0, 0.0])
        Wgx.append([1 / (2 * WG)])
        bgx.append(0.5 - (GAMMA * T0 / (2 * WG) if j == 0 else 0.0))
        Wgb.append([0.0, -GAMMA / (2 * WG), 0.0] if j == 0 else [-GAMMA * LEVEL_F[j - 1] / (2 * WG), 0.0, 0.0])
    clip_pair("lv", "xs2d", Wx, bx, Wb, 4096)
    net.conv1("lv_raw", "lv_r", *pair_diff(NLEV * 4))                                # raw soft masks, (NLEV*4,64,64)
    Wgx.append([1 / (2 * WG)]); bgx.append(0.5); Wgb.append([-F_CONN / (2 * WG), 0.0, 0.0])   # gate for a second nucleus
    clip_pair("gt", "r64", Wgx, bgx, Wgb, 4096)
    Mg, bg_ = pair_diff(NLEV + 1, reps=4, bias=-1.0)
    net.conv1("gt_m1", "gt_r", Mg[:NLEV * 4], bg_[:NLEV * 4])                        # (connectivity gate) - 1
    M0, _ = pair_diff(NLEV + 1)
    net.conv1("cb", "gt_r", M0[:1])                                                  # level-0 component, half res
    net.add("lv_g", ["lv_raw", "gt_m1"])
    net.relu("lv_m", "lv_g")                                                         # isophote masks, (NLEV*4,64,64)

    rw = ray_weights()
    rw4 = np.stack([s2d(rw[k_]) for k_ in range(NRAY)]).reshape(NRAY, 4 * 4096)      # (NRAY, 4*4096)
    nout = NLEV * (NRAY + 1)
    Wr = np.zeros((nout, NLEV * 4 * 4096), np.float32)
    names = []
    for j in range(NLEV):
        Wr[j * (NRAY + 1): j * (NRAY + 1) + NRAY, j * 4 * 4096:(j + 1) * 4 * 4096] = rw4
        Wr[j * (NRAY + 1) + NRAY, j * 4 * 4096:(j + 1) * 4 * 4096] = 1.0
        names += [f"E{j}_{k_}" for k_ in range(NRAY)] + [f"A{j}"]
    net.dense("rays", "lv_m", Wr)
    sc.reg("rays", names)

    # chords: S_k = E_k + E_{k+32};  d_k = E_k - E_{k+32};  M_k = S_k - |d_k| = 2 min(E_k, E_{k+32})
    nh = NRAY // 2
    ex = []
    for j in range(NLEV):
        ex += [{f"E{j}_{k_}": 1.0, f"E{j}_{k_ + nh}": 1.0} for k_ in range(nh)]
        ex += [{f"E{j}_{k_}": 1.0, f"E{j}_{k_ + nh}": -1.0} for k_ in range(nh)]
        ex += [{f"E{j}_{k_}": -1.0, f"E{j}_{k_ + nh}": 1.0} for k_ in range(nh)]
    ch1 = lin_layers(net, "ch1", ex, sc)
    net.relu("ch1r", ch1)
    nm = []
    for j in range(NLEV):
        nm += [f"S{j}_{k_}" for k_ in range(nh)] + [f"dp{j}_{k_}" for k_ in range(nh)] + [f"dn{j}_{k_}" for k_ in range(nh)]
    sc.reg("ch1r", nm)
    ex = []
    for j in range(NLEV):
        S = [{f"S{j}_{k_}": 1.0} for k_ in range(nh)]
        Mn = [{f"S{j}_{k_}": 1.0, f"dp{j}_{k_}": -1.0, f"dn{j}_{k_}": -1.0} for k_ in range(nh)]
        ex += S + [{a: -v for a, v in e.items()} for e in S] + Mn + [{a: -v for a, v in e.items()} for e in Mn]
    # W_k = S_{k+KW} + S_{k-KW}: the two chords at +-56 degrees from direction k.  Its minimum over k sits on the minor
    # axis; divided by twice the minor-axis chord it is ~1.7 for a thin ellipse or strip and ~1.0-1.3 when a central
    # bulge (or a diamond-shaped outline) fattens the middle.
    for j in RHO_LEVELS:
        ex += [{f"S{j}_{(k_ + KW) % nh}": -1.0, f"S{j}_{(k_ - KW) % nh}": -1.0} for k_ in range(nh)]
    ch2 = lin_layers(net, "ch2", ex, sc)
    net.reshape("ch2r", ch2, (NLEV * 4 + len(RHO_LEVELS), nh, 1))
    net.pool("ch2m", "ch2r", "maxpool", [nh, 1])                                     # per level: Smax, -Smin, Mmax, -Mmin
    nm = []
    for j in range(NLEV):
        nm += [f"Smax{j}", f"nSmin{j}", f"Mmax{j}", f"nMmin{j}"]
    nm += [f"nWmin{j}" for j in RHO_LEVELS]
    sc.reg("ch2m", nm)

    # ---- 5a. difference of Gaussians (ridges, sharpness) --------------------------------------------------------
    net.conv("xs2", "xs", b5[None, None], padding=2)
    net.conv("nxs3", "xs2", -b5[None, None], padding=2)
    net.add("dog", ["xs", "nxs3"])                                                   # G(s2=1) - G(s2=3) of the luminance
    net.conv("dog4", "dog", s2dk, stride=2)
    net.reshape("dg_r", "dog4", (1, 4, 4096))
    net.pool("dg_m", "dg_r", "maxpool", [4, 1])
    net.reshape("dg64", "dg_m", (1, 64, 64))                                         # max DoG in each 2x2 block

    # ---- 5b. local peaks at half resolution: P = min over 8 directions of the drop at ~4 px ------------------------
    offs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
    k = np.zeros((8, 1, 5, 5))
    for i, (dy, dx) in enumerate(offs):
        k[i, 0, 2 + dy, 2 + dx] = 1.0
    for i, (dy, dx) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        k[4 + i, 0, 2 + dy, 2 + dx] = 0.5
        k[4 + i, 0, 2 + 2 * dy, 2 + 2 * dx] = 0.5
    k[:, 0, 2, 2] = -1.0
    net.conv("pk_u", "x64s", k, padding=2)                                           # neighbour minus centre, 8 directions
    net.reshape("pk_ur", "pk_u", (1, 8, 4096))
    net.pool("pk_um", "pk_ur", "maxpool", [8, 1])
    net.reshape("pk_us", "pk_um", (1, 64, 64))
    net.conv1("pk_n", "pk_us", [[-1.0], [-1.0 / DP], [-1.0 / DP]], [0.0, -TP / DP, -TP / DP - 1.0])
    net.relu("pk", "pk_n")                                                           # ch0: P;  ch1-ch2: clip pair of P
    net.conv1("np_c", "pk", [[0.0, 1.0, -1.0]], padding=1)
    net.pool("np", "np_c", "maxpool", [3, 3], [1, 1])                                # dilated strong peaks, (1,64,64)

    # ---- 5c. second nucleus: h-dome D = X - R (height above the level at which a spot joins the centre) -----------
    # counted where the spot joins the centre at >= F_CONN*H and reaches >= F_PEAK*H
    net.conv1("r64n", "r64", [[-1.0]])
    net.add("dome", ["x64p", "r64n"])
    clip_pair("hg", "x64s", [[1 / 0.02]], [0.5], [[-F_PEAK / 0.02, 0.0, 0.0]], 4096)
    net.conv1("po_h", "hg_r", [[1.0, -1.0], [0.0, 0.0]], [-1.0, 0.0])
    net.conv1("po_d", "dome", [[1.0], [0.0]])
    net.conv1("po_p", "pk", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    net.conv1("po_c", "gt_r", np.vstack([M0[NLEV], 0 * M0[0]]), [-1.0, 0.0])
    far = (r64 > RP_OUT).astype(float)
    cen64 = (r64 < RP_IN).astype(float)
    net.dense("po_k", "hs", np.zeros((2 * 4096, 3)), np.concatenate([-far.reshape(-1), cen64.reshape(-1) - 1.0]))
    net.reshape("po_kr", "po_k", (2, 64, 64))
    net.add("po_a", ["po_d", "po_p", "po_c", "po_h", "po_kr"])
    net.relu("po_g", "po_a")
    net.pool("pk_o", "po_g", "maxpool", [64, 64])                                    # [dome of a second nucleus, central peak-ness]
    sc.reg("pk_o", ["Po", "Pc"])
    # area of the domes (half-resolution pixels with D above ~DA): a companion galaxy is broad, a star is narrow
    dd = DA_HI - DA_LO
    net.conv1("ad_x", "dome", [[1 / dd], [1 / dd]], [-DA_LO / dd, -DA_LO / dd - 1.0])
    net.relu("ad_r", "ad_x")
    net.conv1("ad_d", "ad_r", [[1.0, -1.0]])
    net.conv1("ad_c", "gt_r", M0[NLEV:NLEV + 1], [-1.0])
    net.dense("ad_k", "hs", np.zeros((4096, 3)), -far.reshape(-1))
    net.reshape("ad_kr", "ad_k", (1, 64, 64))
    net.add("ad_a", ["ad_d", "ad_c", "ad_kr"])
    net.relu("ad_g", "ad_a")
    net.pool("ad_s", "ad_g", "avgpool", [64, 64])
    sc.reg("ad_s", ["ADm"])

    # ---- 5d. ridges: arms, bars, thin discs --------------------------------------------------------------------------
    clip_pair("rg", "dog4", np.eye(4) / DEL_R, [-TAU_R / DEL_R] * 4, [[0.0, 0.0, -1 / DEL_R]] * 4, 4096)
    net.conv1("rg", "rg_r", *pair_diff(4))                                           # soft ridge indicator, (4,64,64)
    net.conv1("rg_c", "cb", [[1.0]] * 4, [-1.0] * 4)
    net.conv1("rg_n", "np", [[-1.0]] * 4)
    core = s2d((rmap < R_CORE).astype(float))
    net.dense("rg_k", "hs", np.zeros((4 * 4096, 3)), -core.reshape(-1))
    net.reshape("rg_kr", "rg_k", (4, 64, 64))
    net.add("rg_g", ["rg", "rg_c", "rg_n", "rg_kr"])
    net.relu("rg_gr", "rg_g")                                                        # ridge & component & not peak/core
    net.pool("rg_s", "rg_gr", "avgpool", [64, 64])
    sc.reg("rg_s", [f"RAp{p}" for p in range(4)])
    RA = {f"RAp{p}": 4096.0 for p in range(4)}                                       # ridge area in pixels

    # ---- 5e. colour ---------------------------------------------------------------------------------------------
    k = np.zeros((3, 3, 2, 2))
    for c in range(3):
        k[c, c] = 0.25
    net.conv("c64", "image", k, [-BG + C_OFF] * 3, stride=2)                          # colours - sky + offset, half res
    net.conv1("c_c", "cb", [[2.0]] * 3, [-2.0] * 3)
    net.add("c_a", ["c64", "c_c"])
    net.relu("c_g", "c_a")
    regs = [np.ones((64, 64)), (r64 < R_CCEN).astype(float), (r64 >= R_COUT).astype(float)]
    Wc = np.zeros((9, 3 * 4096)); Wcb = np.zeros((3, 4096))
    for g, m in enumerate(regs):
        for c in range(3):
            Wc[g * 3 + c, c * 4096:(c + 1) * 4096] = m.reshape(-1)
        Wcb[g] = m.reshape(-1)
    net.dense("c_s", "c_g", Wc)
    net.dense("cb_s", "cb", Wcb)
    sc.reg("c_s", [f"{c}{g}raw" for g in ("all", "cen", "out") for c in "RGB"])
    sc.reg("cb_s", [f"CB{g}" for g in ("all", "cen", "out")])
    colsum = {g: {c: {f"{c}{g}raw": 1.0, f"CB{g}": -C_OFF} for c in "RGB"} for g in ("all", "cen", "out")}

    # ---- 6. ratios through norm ---------------------------------------------------------------------------------
    def neg(e):
        return {a: -v for a, v in e.items()}

    def plus(*es):
        out = {}
        for e in es:
            for a, v in e.items():
                out[a] = out.get(a, 0.0) + v
        return out

    def times(e, f):
        return {a: v * f for a, v in e.items()}

    groups, feat_src = [], {}            # each group: (numerators, denominator); feat name -> (group, slot)

    def group(nums, den):
        """nums: list of (feature_name, expr) (max 3); den: expr or None (None -> plain unit-vector normalisation)."""
        assert len(nums) <= 3
        g = len(groups)
        groups.append(([e for _, e in nums], den))
        for i, (nm_, _) in enumerate(nums):
            feat_src[nm_] = (g, i)

    ang2 = 2 * np.pi * np.arange(nh) / nh
    for j in range(NLEV):
        sumS = {f"S{j}_{k_}": 1.0 for k_ in range(nh)}
        lop = plus({f"dp{j}_{k_}": 1.0 for k_ in range(nh)}, {f"dn{j}_{k_}": 1.0 for k_ in range(nh)})
        cc = {f"S{j}_{k_}": float(np.cos(ang2[k_])) for k_ in range(nh)}
        ss = {f"S{j}_{k_}": float(np.sin(ang2[k_])) for k_ in range(nh)}
        group([(f"qS{j}", {f"nSmin{j}": -1.0})], {f"Smax{j}": 1.0})
        group([(f"qM{j}", {f"nMmin{j}": -1.0})], {f"Mmax{j}": 1.0})
        group([(f"lop{j}", lop), (f"e2c{j}", cc), (f"e2s{j}", ss)], sumS)
        group([(f"uc{j}", cc), (f"us{j}", ss)], None)
    for j in RHO_LEVELS:
        group([(f"rho{j}", {f"nWmin{j}": -0.5})], {f"nSmin{j}": -1.0})
    group([(f"rs{j}", {f"Smax{j}": 1.0}) for j in (1, 2, 3)], {"Smax0": 1.0})
    group([("rs4", {"Smax4": 1.0})], {"Smax0": 1.0})
    group([(f"ra{j}", {f"A{j}": 1.0}) for j in (1, 2, 3)], {"A0": 1.0})
    group([("ra4", {"A4": 1.0}), ("rf", RA)], {"A0": 1.0})
    for g in ("all", "cen", "out"):
        cs = colsum[g]
        group([(f"ci_{g}", plus(cs["B"], neg(cs["R"])))], plus(cs["R"], cs["G"], cs["B"]))

    ex = []
    for nums, den in groups:
        pos = nums + [{}] * (3 - len(nums))
        ex += pos + [neg(e) for e in pos]
        ex += [times(den, KDOM), times(den, -KDOM)] if den is not None else [{}, {}]
    grp = lin_layers(net, "grp", ex, sc)
    net.reshape("grp_r", grp, (len(groups), GP, 1))
    net.norm("grp_n", "grp_r")
    names = [None] * (len(groups) * GP)
    scale = {}
    for nm_, (g, i) in feat_src.items():
        names[g * GP + i] = "n_" + nm_
        scale[nm_] = KDOM / 2.0 if groups[g][1] is not None else 0.5
    sc.reg("grp_n", names)
    feats = {nm_: {"n_" + nm_: scale[nm_]} for nm_ in feat_src}                      # ratio features as expressions
    for raw in ("Smax0", "H", "Po", "Pc", "A0", "s_mask"):
        feats[raw] = {raw: 1.0}
    feats["AD"] = {"ADm": 4096.0}
    if not with_rules:
        return net, sc, feats
    rules(net, sc, feats)
    return net, sc, feats


class Rules:
    """Stages of relu units over named quantities.  `cur` maps a name to a linear expression over registered scalars."""

    def __init__(self, net, sc, cur):
        self.net, self.sc, self.cur, self.n = net, sc, dict(cur), 0
        self.units, self.after = {}, []

    def expand(self, e):
        out = {}
        for nm, cf in e.items():
            if nm == "1":
                out["1"] = out.get("1", 0.0) + cf
                continue
            for a, v in self.cur[nm].items():
                out[a] = out.get(a, 0.0) + cf * v
        return out

    def unit(self, name, e):
        self.units[name] = self.expand(e)

    def ramp(self, name, e, lo, hi):
        """name = clip((e - lo) / (hi - lo), 0, 1): 0 at lo, 1 at hi (lo > hi gives a falling ramp)."""
        u = {a: v / (hi - lo) for a, v in self.expand(e).items()}
        u["1"] = u.get("1", 0.0) - lo / (hi - lo)
        self.units[name + "~a"] = u
        ub = dict(u); ub["1"] -= 1.0
        self.units[name + "~b"] = ub
        self.after.append((name, {name + "~a": 1.0, name + "~b": -1.0}))

    def let_after(self, name, e):
        self.after.append((name, e))

    def stage(self, keep=()):
        self.n += 1
        p = f"r{self.n}"
        names, ex = list(self.units), list(self.units.values())
        for nm in keep:
            e = self.expand({nm: 1.0})
            names += [nm + "~p", nm + "~n"]
            ex += [e, {a: -v for a, v in e.items()}]
        lay = lin_layers(self.net, p, ex, self.sc)
        self.net.relu(p + "r", lay)
        self.sc.reg(p + "r", [f"{p}:{nm}" for nm in names])
        cur = {nm: {f"{p}:{nm}": 1.0} for nm in self.units}
        for nm in keep:
            cur[nm] = {f"{p}:{nm}~p": 1.0, f"{p}:{nm}~n": -1.0}
        self.cur, after = cur, self.after
        self.units, self.after = {}, []
        for name, e in after:
            self.cur[name] = self.expand(e)


def rules(net, sc, feats):
    R = Rules(net, sc, feats)
    carry = ["rs1", "rs2", "rs3", "rs4", "ra1", "ra2", "rf", "lop0", "lop1", "lop2", "ci_all", "ci_cen", "ci_out",
             "Po", "Pc", "H", "Smax0", "AD", "rho0", "rho1"]
    # ---- stage 1: axis ratio of each isophote = max(chord-sum version, opposite-ray-min version); twists ------------
    for j in range(NLEV):
        R.unit(f"mq{j}", {f"qM{j}": 1.0, f"qS{j}": -1.0})
        R.let_after(f"q{j}", {f"qS{j}": 1.0, f"mq{j}": 1.0})
    for a, b in ((1, 2),):
        for comp in ("uc", "us"):
            R.unit(f"tw{a}{b}{comp}p", {f"{comp}{a}": 1.0, f"{comp}{b}": -1.0})
            R.unit(f"tw{a}{b}{comp}n", {f"{comp}{a}": -1.0, f"{comp}{b}": 1.0})
        R.let_after(f"tw{a}{b}", {f"tw{a}{b}{c}{sg}": 0.5 for c in ("uc", "us") for sg in "pn"})
    R.stage(keep=carry + [f"qS{j}" for j in range(NLEV)])
    R.cur["a0"] = R.expand({"Smax0": 0.5})
    R.cur["qbody"] = R.expand({"q0": QB_W[0], "q1": QB_W[1], "q2": QB_W[2]})
    R.cur["ci_mix"] = R.expand({"ci_all": 0.5, "ci_out": 0.5})

    # ---- stage 2: soft indicators ----------------------------------------------------------------------------------
    R.unit("mn01", {"q0": 1.0, "q1": -1.0})
    R.let_after("qthin", {"q0": 1.0, "mn01": -1.0})
    # spiral-versus-smooth cues
    R.ramp("z_ci", {"ci_mix": 1.0}, *Z_CI)
    R.ramp("z_rf", {"rf": 1.0}, *Z_RF)
    R.ramp("z_ra", {"ra1": 1.0}, *Z_RA)
    R.ramp("z_a0", {"a0": 1.0}, *Z_A0)
    R.ramp("z_lop", {"lop1": 1.0}, *Z_LOP)
    R.ramp("z_h", {"H": 1.0}, *Z_H)
    # loose-versus-tight cues
    R.ramp("l_ci", {"ci_mix": 1.0}, *L_CI)
    R.ramp("l_rf", {"rf": 1.0}, *L_RF)
    R.ramp("l_lop", {"lop1": 1.0}, *L_LOP)
    R.ramp("l_h", {"H": 1.0}, *L_H)
    # second nucleus
    R.ramp("m_h", {"Po": 1.0}, *M_PO)
    R.ramp("m_a", {"AD": 1.0, "Po": -M_AK}, *M_AD)
    # shape of the body
    R.ramp("rnd", {"qbody": 1.0}, *Q_ROUND)
    R.ramp("elo", {"qbody": 1.0}, *Q_CIGAR)
    R.ramp("big", {"a0": 1.0}, *A_BIG)
    R.ramp("inv", {"a0": 1.0}, *A_INV)
    # bulge of an edge-on disc: compact inner isophote, or inner isophote rounder than the disc
    R.ramp("b_rs", {"rs4": 1.0}, *B_RS)
    R.ramp("b_q", {"q4": 1.0}, *B_Q)
    R.unit("mrho", {"rho0": 1.0, "rho1": -1.0})
    R.let_after("rhomin", {"rho0": 1.0, "mrho": -1.0})
    # bar: inner isophotes more elongated than the outer one
    for j in (1, 2, 3):
        R.unit(f"bq{j}", {"q0": 1.0, f"q{j}": -1.0, "1": -BAR_DQ})
    # lopsidedness
    R.ramp("d_lop", {"lop0": 1.0, "lop1": 1.0}, *D_LOP)
    R.stage(keep=["tw12", "q0", "rho0"])

    # ---- stage 3: combined evidence ----------------------------------------------------------------------------------
    R.ramp("edge", {"qthin": 1.0}, *Q_EDGE)
    cues = ("z_ci", "z_rf", "z_ra", "z_a0", "z_lop", "z_h")
    z = {c: 2 * w for c, w in zip(cues, ZW)}
    z["1"] = -sum(ZW) + Z_SHIFT
    R.ramp("sp", z, -Z_HALF, Z_HALF)
    cues = ("l_ci", "l_rf", "l_lop", "l_h")
    lz = {c: 2 * w for c, w in zip(cues, LW)}
    lz["1"] = -sum(LW) + L_SHIFT
    R.ramp("loose", lz, -L_HALF, L_HALF)
    R.ramp("b_rho", {"rhomin": 1.0}, *B_RHO)
    R.ramp("merge", {"m_h": 1.0, "m_a": 1.0}, 1.0, 2.0)
    R.ramp("bar", {"bq1": 1.0, "bq2": 1.0, "bq3": 1.0, "tw12": -BAR_TW}, BAR_LO, BAR_HI)
    R.stage(keep=["rnd", "elo", "big", "inv", "d_lop", "b_rs", "b_q"])
    R.cur["bulge"] = R.expand({"b_q": B_MIX[0], "b_rs": B_MIX[1], "b_rho": B_MIX[2]})

    # ---- class scores: each class pays a penalty for every condition it fails ---------------------------------------
    def pen(*terms):
        out = {}
        for cf, nm in terms:
            if nm.startswith("!"):                        # penalty cf * (1 - x)
                out["1"] = out.get("1", 0.0) - cf
                out[nm[1:]] = out.get(nm[1:], 0.0) + cf
            else:                                         # penalty cf * x
                out[nm] = out.get(nm, 0.0) - cf
        return out

    W = W_MAIN
    sco = [
        pen((W_DIST, "!d_lop"), (W_MERGE, "merge"), (W, "edge")),                                    # 0 disturbed
        pen((W_MERGE, "!merge"), (-W_INV, "inv")),                                                   # 1 merging
        pen((W, "edge"), (W, "sp"), (W, "!rnd")),                                                    # 2 round smooth
        pen((W, "edge"), (W, "sp"), (W, "rnd"), (W, "elo")),                                         # 3 in-between smooth
        pen((W, "sp"), (W, "!elo"), (W_SUB, "big")),                                                 # 4 cigar smooth
        pen((W, "edge"), (W, "!sp"), (W_BAR, "!bar")),                                               # 5 barred spiral
        pen((W, "edge"), (W, "!sp"), (W_SUB, "loose"), (W_BARX, "bar")),                             # 6 tight spiral
        pen((W, "edge"), (W, "!sp"), (W_SUB, "!loose"), (W_BARX, "bar")),                            # 7 loose spiral
        pen((W, "!edge"), (W_SUB, "bulge")),                                                         # 8 edge-on, no bulge
        pen((W, "!edge"), (W_SUB, "!bulge")),                                                        # 9 edge-on, bulge
    ]
    for c in range(10):
        sco[c]["1"] = sco[c].get("1", 0.0) + BIAS[c]
    out = lin_layers(net, "score", [R.expand(e) for e in sco], sc)
    assert out == net.layers[-1]["name"]


def finish(net, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": net.layers}, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **net.w)


if __name__ == "__main__":
    net = build()[0]
    finish(net, sys.argv[1] if len(sys.argv) > 1 else ".")
