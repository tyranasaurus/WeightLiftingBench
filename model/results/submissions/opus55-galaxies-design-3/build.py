"""Builds a hand-designed galaxy-morphology network (no training, no data).

Usage: python3 build.py OUT_DIR   -> writes OUT_DIR/network.json and OUT_DIR/weights.npz

Every weight below comes from a formula: smoothing / derivative stencils, coordinate maps (x, y, x^2-y^2, 2xy,
polar angle), a 180-degree rotation permutation, piecewise-linear approximations of log2 and other
1-D functions, and a hand-written rule table with hand-chosen thresholds.
"""
import json
import sys
from pathlib import Path

import numpy as np

F32 = np.float32


# ----------------------------------------------------------------------------------------------
# minimal network builder
# ----------------------------------------------------------------------------------------------
class Net:
    def __init__(self):
        self.layers, self.weights = [], {}
        self.shape = {"image": (3, 128, 128)}
        self.count = 0

    def _nm(self, p):
        self.count += 1
        return f"{p}{self.count}"

    def _add_layer(self, name, op, inputs, oshape, **kw):
        assert name not in self.shape
        L = {"name": name, "op": op, "inputs": list(inputs)}
        L.update(kw)
        self.layers.append(L)
        self.shape[name] = tuple(int(v) for v in oshape)
        return name

    def conv(self, x, W, b=None, stride=1, pad=0, name=None):
        W = np.asarray(W, np.float64)
        o, i, k, k2 = W.shape
        C, H, Wd = self.shape[x]
        assert i == C and k == k2, (W.shape, self.shape[x])
        b = np.zeros(o) if b is None else np.asarray(b, np.float64).reshape(o)
        Ho, Wo = (H + 2 * pad - k) // stride + 1, (Wd + 2 * pad - k) // stride + 1
        name = name or self._nm("cv")
        self.weights[name + ".W"] = W.astype(F32)
        self.weights[name + ".b"] = b.astype(F32)
        return self._add_layer(name, "conv", [x], (o, Ho, Wo), out=int(o), kernel=[int(k), int(k)],
                               stride=int(stride), padding=int(pad))

    def lin(self, x, M, b=None, pad=0, name=None):
        """1x1 convolution with channel matrix M (out, in)."""
        M = np.asarray(M, np.float64)
        return self.conv(x, M[:, :, None, None], b, 1, pad, name)

    def dense(self, x, W, b=None, name=None):
        W = np.asarray(W, np.float64)
        n_in = int(np.prod(self.shape[x]))
        assert W.shape[1] == n_in, (W.shape, self.shape[x])
        b = np.zeros(W.shape[0]) if b is None else np.asarray(b, np.float64)
        name = name or self._nm("fc")
        self.weights[name + ".W"] = W.astype(F32)
        self.weights[name + ".b"] = b.astype(F32)
        return self._add_layer(name, "dense", [x], (W.shape[0],), out=int(W.shape[0]))

    def relu(self, x, name=None):
        return self._add_layer(name or self._nm("re"), "relu", [x], self.shape[x])

    def pool(self, x, op, size, stride, name=None):
        C, H, W = self.shape[x]
        (sh, sw), (th, tw) = size, stride
        return self._add_layer(name or self._nm(op[:2]), op, [x], (C, (H - sh) // th + 1, (W - sw) // tw + 1),
                               size=[int(sh), int(sw)], stride=[int(th), int(tw)])

    def maxpool(self, x, size, stride, name=None):
        return self.pool(x, "maxpool", size, stride, name)

    def avgpool(self, x, size, stride, name=None):
        return self.pool(x, "avgpool", size, stride, name)

    def add(self, *xs, name=None):
        assert len(xs) >= 2 and all(self.shape[v] == self.shape[xs[0]] for v in xs)
        return self._add_layer(name or self._nm("ad"), "add", list(xs), self.shape[xs[0]])

    def reshape(self, x, shape, name=None):
        assert int(np.prod(shape)) == int(np.prod(self.shape[x]))
        return self._add_layer(name or self._nm("rs"), "reshape", [x], tuple(shape), shape=[int(v) for v in shape])

    def norm(self, x, name=None):
        assert len(self.shape[x]) == 3
        return self._add_layer(name or self._nm("nm"), "norm", [x], self.shape[x])

    def activation_floats(self):
        return sum(int(np.prod(s)) for k, s in self.shape.items() if k != "image")

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "network.json").write_text(json.dumps({"layers": self.layers}, indent=1))
        np.savez(out / "weights.npz", **self.weights)


# ----------------------------------------------------------------------------------------------
# scalar feature bus: named scalars living in vector tensors; linear maps, relu, norm ratios,
# piecewise-linear 1-D functions
# ----------------------------------------------------------------------------------------------
class Bus:
    def __init__(self, net):
        self.net = net
        self.loc = {}

    def register(self, tensor, names):
        for i, n in enumerate(names):
            if n is not None:
                assert n not in self.loc, n
                self.loc[n] = (tensor, i)

    def _lin_tensor(self, rows):
        """rows: list of (dict feat->coef, bias). Returns tensor name of the dense (+add) output."""
        net = self.net
        srcs = []
        for d, _ in rows:
            for f in d:
                t = self.loc[f][0]
                if t not in srcs:
                    srcs.append(t)
        if not srcs:
            srcs = [next(iter(self.loc.values()))[0]]
        parts = []
        for si, s in enumerate(srcs):
            n_in = int(np.prod(net.shape[s]))
            W = np.zeros((len(rows), n_in))
            b = np.zeros(len(rows))
            for o, (d, bias) in enumerate(rows):
                for f, c in d.items():
                    t, i = self.loc[f]
                    if t == s:
                        W[o, i] += c
                if si == 0:
                    b[o] = bias
            parts.append(net.dense(s, W, b))
        return parts[0] if len(parts) == 1 else net.add(*parts)

    def lin(self, items):
        """items: list of (name, dict, bias)."""
        t = self._lin_tensor([(d, b) for _, d, b in items])
        self.register(t, [n for n, _, _ in items])
        return t

    def relu(self, items):
        t = self._lin_tensor([(d, b) for _, d, b in items])
        r = self.net.relu(t)
        self.register(r, [n for n, _, _ in items])
        return r

    def norm_groups(self, groups):
        """groups: list of list of (name, dict, bias). For each group with entries a_k, outputs
        name = a_k / sqrt(sum_k a_k^2)  (exactly, up to the 1e-5 epsilon)."""
        m = max(len(g) for g in groups)
        rows, names = [], []
        for g in groups:
            for k in range(m):
                if k < len(g):
                    n, d, b = g[k]
                    rows.append((d, b))
                    rows.append(({f: -c for f, c in d.items()}, -b))
                    names += [n, None]
                else:
                    rows += [({}, 0.0), ({}, 0.0)]
                    names += [None, None]
        t = self._lin_tensor(rows)
        G = len(groups)
        r = self.net.reshape(t, (G, 1, 2 * m))
        nr = self.net.norm(r)
        f = self.net.reshape(nr, (G * 2 * m,))
        # norm output = a*sqrt(m)/|a| ; rescale with a dense so that outputs are a/|a|
        sc = np.zeros((G * 2 * m, G * 2 * m))
        for i in range(G * 2 * m):
            sc[i, i] = 1.0 / np.sqrt(m)
        out = self.net.dense(f, sc, np.zeros(G * 2 * m))
        self.register(out, names)
        return out

    def pl(self, items):
        """items: list of (name, src_feature, knots, values). Piecewise-linear interpolation through
        (knots, values), constant outside [knots[0], knots[-1]]."""
        hid_rows, hid_names, outs = [], [], []
        for name, src, knots, vals in items:
            knots = np.asarray(knots, float)
            vals = np.asarray(vals, float)
            slopes = np.diff(vals) / np.diff(knots)
            coefs = np.concatenate([[slopes[0]], np.diff(slopes), [-slopes[-1]]])
            hn = []
            for j, k in enumerate(knots):
                hname = f"_h_{name}_{j}"
                hid_rows.append((hname, {src: 1.0}, -k))
                hn.append(hname)
            outs.append((name, dict(zip(hn, coefs)), vals[0]))
        # hidden relu units relu(x - k_j); value at x<=k0 is vals[0]
        self.relu(hid_rows)
        return self.lin(outs)

    def log2(self, items, lo=1.0, hi=2.0 ** 30, per_octave=2):
        """items: list of (name, src). log2 of max(src, lo) (lo=1 -> 0)."""
        n = int(round(np.log2(hi / lo) * per_octave))
        knots = lo * 2.0 ** (np.arange(n + 1) / per_octave)
        return self.pl([(nm, s, knots, np.log2(knots)) for nm, s in items])

    def ramp(self, items):
        """items: (name, dict, a, b): clip((x-a)/(b-a), 0, 1) of the linear combination x."""
        rows = []
        for name, d, a, b in items:
            s = 1.0 / (b - a)
            rows.append((f"_r1_{name}", {f: c * s for f, c in d.items()}, -a * s))
            rows.append((f"_r2_{name}", {f: c * s for f, c in d.items()}, -b * s))
        self.relu(rows)
        return self.lin([(name, {f"_r1_{name}": 1.0, f"_r2_{name}": -1.0}, 0.0) for name, _, _, _ in items])

    def max(self, items):
        """items: list of (name, [features]); name = max of the features (via reshape + maxpool)."""
        K = max(len(f) for _, f in items)
        rows = []
        for _, fs in items:
            fs = list(fs) + [fs[0]] * (K - len(fs))
            rows += [({f: 1.0}, 0.0) for f in fs]
        t = self._lin_tensor(rows)
        r = self.net.reshape(t, (len(items), 1, K))
        m = self.net.maxpool(r, [1, K], [1, K])
        v = self.net.reshape(m, (len(items),))
        self.register(v, [n for n, _ in items])
        return v

    def collect(self, names, tensor_name=None):
        """Gather named features into one vector (in this order)."""
        t = self.lin([(n + "~", {n: 1.0}, 0.0) for n in names])
        return t


# ----------------------------------------------------------------------------------------------
# filters and maps from formulas
# ----------------------------------------------------------------------------------------------
B5 = np.array([1, 4, 6, 4, 1], float) / 16.0
B3 = np.array([1, 2, 1], float) / 4.0


def polyphase_kernel(K):
    """Full-resolution kernel K (odd size <= 5) expressed on the 4 polyphase channels of a 2x2
    space-to-depth layout (channel 2a+b holds full-res pixel (2Y+a, 2X+b)). Returns (4, 4, 3, 3)."""
    r = K.shape[0] // 2
    W = np.zeros((4, 4, 3, 3))
    for a in range(2):
        for b in range(2):
            for a2 in range(2):
                for b2 in range(2):
                    for DY in (-1, 0, 1):
                        for DX in (-1, 0, 1):
                            dy, dx = 2 * DY + a2 - a, 2 * DX + b2 - b
                            if abs(dy) <= r and abs(dx) <= r:
                                W[2 * a + b, 2 * a2 + b2, DY + 1, DX + 1] += K[dy + r, dx + r]
    return W


def ridge_kernels():
    """Negative second directional derivatives -D_n for ridges along 0, 45, 90, 135 degrees
    (x = column to the right, y = row downwards) of a 3x3-binomial-smoothed map, as 5x5 kernels."""
    S = np.outer(B3, B3)
    dxx = np.zeros((3, 3)); dxx[1] = [1, -2, 1]
    dyy = dxx.T.copy()
    dxy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4.0

    def full(k):
        out = np.zeros((5, 5))
        for i in range(3):
            for j in range(3):
                out[i:i + 3, j:j + 3] += k[i, j] * S
        return out

    Ixx, Iyy, Ixy = full(dxx), full(dyy), full(dxy)
    ks = []
    for al in np.radians([0, 45, 90, 135]):
        s, c = np.sin(al), np.cos(al)
        D = s * s * Ixx - 2 * s * c * Ixy + c * c * Iyy
        ks.append(-D)
    return np.array(ks)  # (4, 5, 5)


RIDGE_ANG = np.radians([0, 45, 90, 135])

# isophote levels (sky-subtracted luminance) for the full-resolution level masks
LEVELS = [0.05, 0.10, 0.18, 0.30, 0.45, 0.62]
T_SEG = 0.05
T_CORE = [0.35, 0.55]
RAMP_W = 0.02


def build_features(net, bus):
    # ---------------- stage 0: space-to-depth luminance (full res, 4 phases) + 2x2-averaged colours
    W0 = np.zeros((7, 3, 2, 2))
    for a in range(2):
        for b in range(2):
            W0[2 * a + b, :, a, b] = 1.0 / 3.0
    for c in range(3):
        W0[4 + c, c] = 0.25
    c0 = net.conv("image", W0, None, stride=2, pad=0, name="poly")  # (7,64,64)

    # ---------------- sky: minimum over the 8x8 grid of 16x16-pixel block means (per image)
    blk = net.avgpool(c0, [8, 8], [8, 8], name="sky_blocks")  # (7,8,8)
    M = np.zeros((4, 7))
    M[0, :4] = -0.25
    M[1, 4] = M[2, 5] = M[3, 6] = -1.0
    nb = net.lin(blk, M, name="sky_neg")
    negsky = net.maxpool(nb, [8, 8], [8, 8], name="negsky")  # (4,1,1) = -sky
    Wb = np.zeros((7 * 4096, 4))
    for ch in range(7):
        Wb[ch * 4096:(ch + 1) * 4096, 0 if ch < 4 else ch - 3] = 1.0
    skymap = net.reshape(net.dense(negsky, Wb, name="skymap_fc"), (7, 64, 64), name="skymap")
    b0 = net.add(c0, skymap, name="desky")  # sky-subtracted

    # ---------------- smoothing: full-res binomial 5x5 (polyphase), 64-res luminance, smoothed colours
    W1 = np.zeros((8, 7, 3, 3))
    PK = polyphase_kernel(np.outer(B5, B5))
    W1[:4, :4] = PK
    W1[4, :4] = PK.mean(axis=0)
    for c in range(3):
        W1[5 + c, 4 + c] = np.outer(B3, B3)
    sm = net.conv(b0, W1, None, 1, 1, name="smooth")  # (8,64,64)
    bus.net_tensors = {"smooth": sm}

    # ---------------- threshold maps: r = relu(1 - (x - t)/w); T - 1 = -r is 0 above t + w and < 0 below
    nL = len(LEVELS)
    thr = [(4, t) for t in LEVELS] + [(p, t) for t in LEVELS for p in range(4)]
    nT = len(thr)
    MT = np.zeros((nT, 8))
    bT = np.zeros(nT)
    for i, (ch, t) in enumerate(thr):
        MT[i, ch] = -1.0 / RAMP_W
        bT[i] = t / RAMP_W + 1.0
    thr_r = net.relu(net.lin(sm, MT, bT, name="thr_lin"), name="thr_relu")  # (30,64,64)
    Sel = np.zeros((nL, nT))
    for j in range(nL):
        Sel[j, j] = -1.0
    tm1 = net.lin(thr_r, Sel, name="tm1_64")  # (6,64,64): T_j - 1 at 64-res

    # central seeds (constant maps from a zero-weight dense with a patterned bias)
    seed2 = np.zeros((64, 64))
    seed2[31:33, 31:33] = 1.0
    seed4 = np.zeros((64, 64))
    seed4[30:34, 30:34] = 1.0
    seeds = [seed2, seed2, seed2, seed4, seed4, seed4]
    cm = net.reshape(net.dense(negsky, np.zeros((nL * 4096, 4)), np.concatenate([p.ravel() for p in seeds]),
                               name="seed_fc"), (nL, 64, 64), name="seed")

    # geodesic dilation of the seed inside each level's threshold map: X <- maxpool(pad(X)) + (T - 1)
    def flood(x, tm, nch, p, n, name):
        for it in range(n):
            pd = net.lin(x, np.eye(nch), None, pad=p, name=f"{name}_pad{it}")
            mp = net.maxpool(pd, [2 * p + 1, 2 * p + 1], [1, 1], name=f"{name}_mp{it}")
            x = net.add(mp, tm, name=f"{name}_ad{it}")
        return x

    x = net.add(cm, tm1, name="fl_x0")
    x = flood(x, tm1, nL, 1, 8, "flA")
    E6 = np.eye(nL)
    xf = net.lin(x, E6[0:3], name="fl_xf")
    tmf = net.lin(tm1, E6[0:3], name="fl_tmf")
    xb = net.lin(x, E6[3:6], name="fl_xb")
    tmb = net.lin(tm1, E6[3:6], name="fl_tmb")
    xf = flood(xf, tmf, 3, 2, 10, "flF")   # faint levels reach 8 + 20 = 28 cells (56 px) from the centre
    xb = flood(xb, tmb, 3, 2, 3, "flB")    # bright levels reach 8 + 6 = 14 cells (28 px)
    ccf = net.relu(xf, name="cc_faint")    # (3,64,64) connected components of levels 0..2
    ccb = net.relu(xb, name="cc_bright")   # (3,64,64) connected components of levels 3..5
    seg = net.lin(ccf, E6[0:1, 0:3], name="segment")

    # ---------------- full-res level masks: M_j = relu(CC_j + T128_j - 1)  (ch = 4*j + phase)
    Bf = np.zeros((4 * nL, 3))
    Bb = np.zeros((4 * nL, 3))
    for j in range(nL):
        for p in range(4):
            (Bf if j < 3 else Bb)[4 * j + p, j % 3] = 1.0
    Sel2 = np.zeros((4 * nL, nT))
    for i in range(4 * nL):
        Sel2[i, nL + i] = -1.0
    lm = net.relu(net.add(net.lin(ccf, Bf, name="ccf_b"), net.lin(ccb, Bb, name="ccb_b"),
                          net.lin(thr_r, Sel2, name="tm1_128"), name="lev_add"), name="levmask")

    # coordinates of full-res pixels for each polyphase cell, in units of 8 px, centre 63.5
    Y, X = np.mgrid[0:64, 0:64]
    xs, ys = np.zeros((4, 64, 64)), np.zeros((4, 64, 64))
    for a in range(2):
        for b in range(2):
            ys[2 * a + b] = (2 * Y + a - 63.5) / 8.0
            xs[2 * a + b] = (2 * X + b - 63.5) / 8.0
    mom_maps = [np.ones_like(xs), xs * xs + ys * ys, xs * xs - ys * ys, 2 * xs * ys, xs, ys]
    mnames = ["A", "s", "u", "v", "mx", "my"]
    Wm = np.zeros((nL * 6, nL * 4 * 4096))
    for j in range(nL):
        for k, mp in enumerate(mom_maps):
            Wm[j * 6 + k, j * 4 * 4096:(j + 1) * 4 * 4096] = mp.ravel()
    mom = net.dense(lm, Wm, name="moments")
    bus.register(mom, [f"{n}{j}" for j in range(nL) for n in mnames])

    # central brightness: mean of the 2x2 and 6x6 central full-res pixels of the smoothed map
    Wc = np.zeros((2, 8, 64, 64))
    full = np.zeros((128, 128))
    full[63:65, 63:65] = 1.0 / 4
    full6 = np.zeros((128, 128))
    full6[61:67, 61:67] = 1.0 / 36
    for a in range(2):
        for b in range(2):
            Wc[0, 2 * a + b] = full[a::2, b::2]
            Wc[1, 2 * a + b] = full6[a::2, b::2]
    cen = net.dense(sm, Wc.reshape(2, -1), name="centre")
    bus.register(cen, ["pc2", "pc6"])

    # ---------------- 64-res masks: per-level mean over phases
    Ml = np.zeros((nL, 4 * nL))
    for j in range(nL):
        Ml[j, 4 * j:4 * j + 4] = 0.25
    m64 = net.lin(lm, Ml, name="mask64")  # (6,64,64)

    # ---------------- colours in the inner region (level 3) and the outer region (level 1 minus level 3)
    jin = 2
    Mx = np.zeros((6, 8))
    for k in range(3):
        Mx[k, 5 + k] = 1.0
        Mx[3 + k, 5 + k] = 1.0
    cx_ = net.lin(sm, Mx, np.full(6, -1.0), name="col_src")
    Mr = np.zeros((6, nL))
    Mr[0:3, jin] = 1.0
    Mr[3:6, 0] = 1.0
    Mr[3:6, jin] = -1.0
    cr_ = net.lin(m64, Mr, None, name="col_reg")
    colg = net.relu(net.add(cx_, cr_, name="col_add"), name="col_gate")
    Wcs = np.zeros((6, 6 * 4096))
    for k in range(6):
        Wcs[k, k * 4096:(k + 1) * 4096] = 1.0
    cs = net.dense(colg, Wcs, name="col_sums")
    bus.register(cs, ["R_in", "G_in", "B_in", "R_out", "G_out", "B_out"])

    # ---------------- 180-degree rotational asymmetry of the segment-masked 64-res luminance
    ls64 = net.lin(sm, np.eye(8)[4:5], name="ls64")
    pa_ = net.lin(ls64, np.ones((1, 1)), np.full(1, -1.0), name="asy_l")
    cc1 = net.lin(ccf, E6[1:2, 0:3], name="cc1")
    P = net.relu(net.add(pa_, cc1, name="asy_add"), name="asy_P")  # luminance inside the level-1 component
    # rotation about 5 centres (image centre and +-1 full-res px in x or y); asymmetry = min over centres
    centres = [(31.5, 31.5), (31.0, 31.5), (32.0, 31.5), (31.5, 31.0), (31.5, 32.0)]
    rows_d, sumD = [], np.zeros((len(centres), 4096))
    groups_v = []
    for v, (cy, cx) in enumerate(centres):
        seen = set()
        idx = []
        for y in range(64):
            for x in range(64):
                i = y * 64 + x
                y2, x2 = int(round(2 * cy - y)), int(round(2 * cx - x))
                j = y2 * 64 + x2 if (0 <= y2 < 64 and 0 <= x2 < 64) else -1
                if i == j or i in seen:
                    continue
                seen.add(i)
                if j >= 0:
                    seen.add(j)
                idx.append((i, j))
        groups_v.append(len(idx))
        for i, j in idx:
            r = np.zeros(4096)
            r[i] += 1.0
            if j >= 0:
                r[j] -= 1.0
            rows_d.append(r)
            sumD[v] += r
    Wd = np.array(rows_d)  # (n_pairs_total, 4096): D = P_i - P_j for each unordered pair
    dpos = net.relu(net.dense(P, Wd, name="asy_diff"), name="asy_relu")
    Ws = np.zeros((len(centres), Wd.shape[0]))
    o = 0
    for v, n in enumerate(groups_v):
        Ws[v, o:o + n] = 2.0
        o += n
    s1 = net.dense(dpos, Ws, name="asy_sum")           # 2 sum relu(D) per centre
    s2 = net.dense(P, np.vstack([np.ones((1, 4096)), -sumD]), name="asy_tot")  # flux, -sum D per centre
    bus.register(s1, [f"asy_pos{v}" for v in range(len(centres))])
    bus.register(s2, ["flux"] + [f"asy_negD{v}" for v in range(len(centres))])

    # ---------------- ridge orientation field (3 scales) inside the disk annulus (level 1 minus level 5)
    RK = ridge_kernels()
    KG = 10.0
    ann = net.lin(m64, np.eye(nL)[0:1] - np.eye(nL)[4:5], name="annulus")  # (1,64,64)
    src, msk, res = ls64, ann, 64
    for sc in range(3):
        Wr = RK[:, None, :, :] * (2.0 ** sc) ** 2  # scale-normalised second derivative
        rr = net.conv(src, Wr, np.full(4, -KG), 1, 2, name=f"ridge{sc}")
        mm = net.lin(msk, np.full((4, 1), KG), name=f"ridge_m{sc}")
        E = net.relu(net.add(rr, mm, name=f"ridge_a{sc}"), name=f"ridgeE{sc}")  # (4,res,res)
        yy, xx = np.mgrid[0:res, 0:res]
        cc = (res - 1) / 2.0
        th = np.arctan2(yy - cc, xx - cc)
        Wz = np.zeros((3, 4, res, res))
        for k, al in enumerate(RIDGE_ANG):
            Wz[0, k] = np.cos(2 * (al - th))
            Wz[1, k] = np.sin(2 * (al - th))
            Wz[2, k] = 1.0
        z = net.dense(E, Wz.reshape(3, -1), name=f"ridgeZ{sc}")
        bus.register(z, [f"zre{sc}", f"zim{sc}", f"zet{sc}"])
        if sc < 2:
            src = net.avgpool(src, [2, 2], [2, 2], name=f"ls_down{sc}")
            msk = net.avgpool(msk, [2, 2], [2, 2], name=f"ann_down{sc}")
            res //= 2

    # ---------------- companions: bright pixels (levels 3..5) inside the segment but not connected to the
    # central component, within 28 px of the centre (radial taper 24..32 px)
    Mall = net.relu(net.add(net.lin(seg, np.ones((3, 1)), name="seg3"), net.lin(tm1, E6[3:6], name="tm1b"),
                            name="ball_add"), name="bright_all")  # (3,64,64)
    Y, X = np.mgrid[0:64, 0:64]
    rr = np.sqrt((2 * Y + 0.5 - 63.5) ** 2 + (2 * X + 0.5 - 63.5) ** 2)
    win = np.clip((32.0 - rr) / 8.0, 0, 1).ravel()
    Wk = np.zeros((3, 3 * 4096))
    for k in range(3):
        Wk[k, k * 4096:(k + 1) * 4096] = 4.0 * win  # in full-res px
    bus.register(net.dense(Mall, Wk, name="ball_area"), ["ball3", "ball4", "ball5"])
    bus.register(net.dense(ccb, Wk, name="core_area"), ["core3", "core4", "core5"])
    return {"seg": seg, "levmask": lm, "m64": m64, "smooth": sm, "ccf": ccf, "ccb": ccb, "P": P}


def q_of_shat(sh):
    """axis ratio q of a uniform ellipse from shat = s/sqrt(s^2+u^2+v^2) = 1/sqrt(1+e^2)."""
    e = np.sqrt(np.maximum(1.0 / sh ** 2 - 1.0, 0.0))
    e = np.minimum(e, 1.0)
    return np.sqrt((1 - e) / (1 + e))


def build_derived(net, bus):
    nL = len(LEVELS)
    # log2 of extensive quantities (areas in px, second moments in px^2*px^2, colour sums, fluxes)
    items = []
    for j in range(nL):
        items.append((f"LA{j}", f"A{j}"))
    bus.log2(items, lo=1.0, hi=2.0 ** 15)
    bus.lin([(f"s{j}p", {f"s{j}": 64.0}, 1.0) for j in range(nL)])
    bus.log2([(f"Ls{j}", f"s{j}p") for j in range(nL)], lo=1.0, hi=2.0 ** 28)
    # shape: shat, uhat, vhat for each level  (norm trick)
    bus.norm_groups([[(f"sh{j}", {f"s{j}": 1.0}, 0.0), (f"uh{j}", {f"u{j}": 1.0}, 0.0), (f"vh{j}", {f"v{j}": 1.0}, 0.0)]
                     for j in range(nL)])
    # axis ratio and 0.5*log2(1-e^2) as 1-D functions of shat
    qs = np.concatenate([np.linspace(0.04, 0.3, 14), np.linspace(0.32, 1.0, 35)])
    es = (1 - qs ** 2) / (1 + qs ** 2)
    shs = 1 / np.sqrt(1 + es ** 2)
    order = np.argsort(shs)
    kn, qv = shs[order], qs[order]
    hl = np.log2(2 * qv / (1 + qv ** 2))
    pl = []
    for j in range(nL):
        pl.append((f"q{j}", f"sh{j}", kn, qv))
        pl.append((f"hl{j}", f"sh{j}", kn, hl))
    bus.pl(pl)
    # multi-component index log2 J = log2(2 pi) + log2 s + 0.5 log2(1-e^2) - 2 log2 A   (0 for a centred ellipse)
    bus.lin([(f"LJ{j}", {f"Ls{j}": 1.0, f"hl{j}": 1.0, f"LA{j}": -2.0}, np.log2(2 * np.pi)) for j in range(nL)])

    # colours: log2 ratio of blue (g) to red (z) light, inner region and outer disk
    bus.log2([(f"L{c}", c) for c in ["R_in", "G_in", "B_in", "R_out", "G_out", "B_out"]], lo=1.0, hi=2.0 ** 13, per_octave=4)
    # asymmetry numerator  sum_half |D| = 2 sum relu(D) - sum D
    # sum over pairs |D| = 2 sum relu(D) - sum D, minimised over the 5 rotation centres
    bus.lin([(f"_asn{v}", {f"asy_pos{v}": 1.0, f"asy_negD{v}": 1.0}, 0.0) for v in range(5)] + [("flux1", {"flux": 1.0}, 1.0)])
    bus.lin([(f"_asneg{v}", {f"_asn{v}": -1.0}, 0.0) for v in range(5)])
    bus.max([("_asmn", [f"_asneg{v}" for v in range(5)])])
    bus.lin([("asy_num", {"_asmn": -1.0}, 1.0)])
    bus.log2([("Lasy", "asy_num"), ("Lflux", "flux1")], lo=1.0, hi=2.0 ** 13, per_octave=4)
    # chirality and radial/tangential balance of the ridge field, per scale: Im/Et and Re/Et via norm
    groups = []
    for sc in range(3):
        groups.append([(f"zi{sc}", {f"zim{sc}": 1.0}, 0.0), (f"_zie{sc}", {f"zet{sc}": 1.0}, 0.01)])
        groups.append([(f"zr{sc}", {f"zre{sc}": 1.0}, 0.0), (f"_zre{sc}", {f"zet{sc}": 1.0}, 0.01)])
    # companions: bright area not connected to the central component, beyond a 32 px allowance (stars),
    # relative to the central component's area
    allow = {3: 40.0, 4: 24.0, 5: 12.0}  # px of unconnected bright light tolerated (field stars)
    bus.relu([(f"offa{j}", {f"ball{j}": 1.0, f"core{j}": -1.0}, -allow[j]) for j in (3, 4, 5)])
    for j in (3, 4, 5):
        groups.append([(f"offr{j}", {f"offa{j}": 1.0}, 0.0), (f"_offb{j}", {f"core{j}": 1.0}, 16.0)])
    bus.norm_groups(groups)
    bus.ramp([(f"vcore{j}", {f"core{j}": 1.0}, 12.0, 32.0) for j in (3, 4, 5)])
    bus.relu([(f"offg{j}", {f"offr{j}": 1.0, f"vcore{j}": 2.0}, -2.0) for j in (3, 4, 5)])
    # |eps| of each level = sqrt(1 - shat^2)
    shk = np.linspace(0.70, 1.0, 31)
    bus.pl([(f"em{j}", f"sh{j}", shk, np.sqrt(np.maximum(1 - shk ** 2, 0))) for j in range(nL)])
    # validity of each level (enough pixels for a shape): 0 below 2^4.5 px, 1 above 2^5.5 px
    bus.ramp([(f"val{j}", {f"LA{j}": 1.0}, 4.5, 5.5) for j in range(nL)])

    # bar score of inner levels j=2,3,4 against the outer disk (level 1):
    #   |eps_j - eps_1| + |eps_j| - |eps_1|   (0 for an aligned, rounder bulge; >0 for misaligned / more elongated)
    phis = np.radians(np.arange(16) * 22.5)
    proj = []
    for j in (3, 4):
        for k, ph in enumerate(phis):
            proj.append((f"_pj{j}_{k}", {f"uh{j}": np.cos(ph), f"uh1": -np.cos(ph), f"vh{j}": np.sin(ph), f"vh1": -np.sin(ph)}, 0.0))
    bus.lin(proj)
    bus.max([(f"dv{j}", [f"_pj{j}_{k}" for k in range(16)]) for j in (3, 4)])
    # gate by validity: relu(B_j - 3 (1 - val_j))
    bus.relu([(f"bj{j}", {f"dv{j}": 1.0, f"em{j}": 1.0, "em1": -1.0, f"val{j}": 3.0}, -3.0) for j in (3, 4)])
    # misalignment of the faintest isophote with the next one: |eps0 - eps1| - ||eps0| - |eps1||  (>= 0)
    bus.lin([(f"_p01_{k}", {"uh0": np.cos(ph), "uh1": -np.cos(ph), "vh0": np.sin(ph), "vh1": -np.sin(ph)}, 0.0)
             for k, ph in enumerate(phis)])
    bus.max([("dv01", [f"_p01_{k}" for k in range(16)])])
    bus.relu([("_em01p", {"em0": 1.0, "em1": -1.0}, 0.0), ("_em01n", {"em1": 1.0, "em0": -1.0}, 0.0)])
    bus.relu([("mis01", {"dv01": 1.0, "_em01p": -1.0, "_em01n": -1.0}, 0.0)])
    bus.lin([("q0a", {"q0": 1.0, "mis01": 2.0}, 0.0)])
    # robust outer axis ratios: qt = min(q0a,q1,q2) (thinnest aligned outer isophote), qo = median(q0,q1,q2)
    bus.relu([("_d01", {"q0": 1.0, "q1": -1.0}, 0.0), ("_e01", {"q0a": 1.0, "q1": -1.0}, 0.0)])
    bus.relu([("_d2", {"q1": 1.0, "_d01": 1.0, "q2": -1.0}, 0.0),       # min(max(q0,q1), q2) = q1 + d01 - d2
              ("_e3", {"q0a": 1.0, "_e01": -1.0, "q2": -1.0}, 0.0)])    # min(q0a,q1,q2) = min(q0a,q1) - e3
    bus.relu([("_d4", {"q1": 1.0, "q0": -1.0, "_d01": 2.0, "_d2": -1.0}, 0.0)])
    bus.lin([("qt", {"q0a": 1.0, "_e01": -1.0, "_e3": -1.0}, 0.0),
             ("qo", {"q0": 1.0, "_d01": -1.0, "_d4": 1.0}, 0.0)])
    # aligned outer elongation: |(eps_0 + eps_1)/2| (misaligned faint isophotes, e.g. companions, cancel)
    proj = []
    for k, ph in enumerate(phis):
        proj.append((f"_pa_{k}", {"uh0": 0.5 * np.cos(ph), "uh1": 0.5 * np.cos(ph), "vh0": 0.5 * np.sin(ph), "vh1": 0.5 * np.sin(ph)}, 0.0))
    bus.lin(proj)
    bus.max([("ea", [f"_pa_{k}" for k in range(16)])])
    # |eps~| = e / sqrt(1+e^2) -> axis ratio of a uniform ellipse
    qg = np.linspace(0.04, 1.0, 49)
    eg = (1 - qg ** 2) / (1 + qg ** 2)
    mg = eg / np.sqrt(1 + eg ** 2)
    o = np.argsort(mg)
    bus.pl([("qa", "ea", mg[o], qg[o])])
    # edge-on bulge: inner levels rounder than the thinnest outer isophote: q_j - qt for j=3,4,5 (gated)
    bus.relu([(f"bq{j}", {f"q{j}": 1.0, "qt": -1.0, f"val{j}": 2.0}, -2.0) for j in (3, 4, 5)])
    # multi-component index at bright levels (gated)
    bus.relu([(f"mj{j}", {f"LJ{j}": 1.0, f"val{j}": 3.0}, -3.0) for j in (3, 4, 5)])
    # chirality magnitude per scale
    bus.relu([(f"_cp{sc}", {f"zi{sc}": 1.0}, 0.0) for sc in range(3)] + [(f"_cn{sc}", {f"zi{sc}": -1.0}, 0.0) for sc in range(3)])
    bus.lin([(f"ca{sc}", {f"_cp{sc}": 1.0, f"_cn{sc}": 1.0}, 0.0) for sc in range(3)])
    bus.max([("BAR", ["bj3", "bj4"]), ("BQ", ["bq3", "bq4", "bq5"]), ("MULT", ["mj3", "mj4", "mj5"]),
             ("CHIR", ["ca0", "ca1", "ca2"]), ("OFFC", ["offg3", "offg4", "offg5"])])
    bus.lin([("CIo", {"LB_out": 1.0, "LR_out": -1.0}, 0.0), ("CIi", {"LB_in": 1.0, "LR_in": -1.0}, 0.0),
             ("ASY", {"Lasy": 1.0, "Lflux": -1.0}, 0.0), ("ZR", {"zr0": 1.0, "zr1": 1.0}, 0.0),
             ("BP", {"LA4": 1.0, "LA1": -1.0}, 0.0)])


# fuzzy predicates: name -> (feature combination, value where predicate is 0, value where it is 1)
PREDICATES = {
    "THIN":  ({"qt": 1.0}, 0.34, 0.24),     # thinnest aligned outer isophote: edge-on disk
    "ELONG": ({"qt": 1.0}, 0.52, 0.40),     # cigar-like elongation
    "ROUND": ({"qo": 1.0}, 0.70, 0.82),     # completely round
    "EBUL":  ({"BQ": 1.0}, 0.06, 0.16),     # bright isophotes rounder than the disk: bulge
    "BARP":  ({"BAR": 1.0}, 0.30, 0.60),    # bright isophotes more elongated / twisted than the disk: bar
    "CHP":   ({"CHIR": 1.0}, 0.03, 0.09),   # net spiral handedness of the ridge field
    "BLUE":  ({"CIo": 1.0}, -0.05, 0.20),   # outer colour: red sequence ~ -0.25, green valley ~ +0.17, blue ~ +0.5
                                            # (narrowed towards 0 for JPEG chroma desaturation)
    "ASYP":  ({"ASY": 1.0}, -2.6, -1.6),    # strong 180-degree asymmetry (A 0.16 .. 0.33)
    "ASYL":  ({"ASY": 1.0}, -4.0, -3.0),    # mild asymmetry (A 0.06 .. 0.12)
    "MULP":  ({"MULT": 1.0}, 0.35, 0.75),   # central bright component is not a centred ellipse (dumbbell)
    "OFFP":  ({"OFFC": 1.0}, 0.15, 0.40),   # extended bright companion near the centre
    "CORE":  ({"pc2": 1.0}, 0.50, 0.75),    # bright (saturated) nucleus
    "BIG":   ({"LA1": 1.0}, 9.5, 11.0),     # isophotal area 720 .. 2050 px
    "RAD":   ({"ZR": 1.0}, 0.80, 1.15),     # ridge field radial (smooth profile) rather than tangential (arms)
    "IRRO":  ({"LJ1": 0.5, "LJ2": 0.5}, 0.25, 0.60),  # outer isophotes not a centred ellipse
    "BULP":  ({"BP": 1.0}, -4.0, -2.5),     # bright (0.45) component large relative to the disk (0.10)
}
PRED_ORDER = list(PREDICATES)

# Hand-typed class prototypes from general knowledge of the Galaxy Zoo classes: the value each fuzzy predicate
# typically takes for a clear member of the class. Scores are -sum_f TRUST_f (p_f - PROTO_cf)^2 with one shared
# trust per predicate, so the p_f^2 terms are common to all classes and the score is linear in p.
#      THIN ELONG ROUND EBUL BARP  CHP BLUE ASYP ASYL MULP OFFP CORE  BIG  RAD IRRO BULP
PROTO = {
    0: [0.05, 0.20, 0.30, 0.20, 0.30, 0.30, 0.90, 0.60, 0.90, 0.40, 0.20, 0.30, 0.40, 0.40, 0.60, 0.40],  # disturbed
    1: [0.15, 0.40, 0.25, 0.30, 0.30, 0.15, 0.30, 0.70, 1.00, 0.70, 0.60, 0.90, 0.60, 0.60, 0.70, 0.50],  # merging
    2: [0.00, 0.00, 0.90, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.05, 1.00, 0.30, 0.90, 0.10, 0.60],  # round smooth
    3: [0.00, 0.15, 0.05, 0.20, 0.10, 0.05, 0.10, 0.05, 0.15, 0.10, 0.05, 1.00, 0.20, 0.90, 0.10, 0.60],  # in-between
    4: [0.20, 0.90, 0.00, 0.10, 0.10, 0.05, 0.40, 0.05, 0.20, 0.10, 0.05, 0.80, 0.20, 0.80, 0.10, 0.40],  # cigar
    5: [0.00, 0.10, 0.40, 0.20, 0.70, 0.45, 0.65, 0.10, 0.60, 0.30, 0.10, 1.00, 0.80, 0.45, 0.30, 0.50],  # barred
    6: [0.00, 0.10, 0.60, 0.20, 0.20, 0.35, 0.55, 0.05, 0.50, 0.15, 0.10, 1.00, 0.60, 0.60, 0.20, 0.70],  # tight
    7: [0.00, 0.10, 0.40, 0.20, 0.25, 0.55, 0.85, 0.15, 0.70, 0.20, 0.10, 0.80, 0.80, 0.35, 0.30, 0.30],  # loose
    8: [0.90, 1.00, 0.00, 0.10, 0.10, 0.15, 0.60, 0.05, 0.40, 0.10, 0.05, 0.50, 0.80, 0.60, 0.10, 0.30],  # edge-on
    9: [0.60, 0.95, 0.00, 0.80, 0.10, 0.10, 0.15, 0.05, 0.40, 0.20, 0.05, 0.95, 0.50, 0.70, 0.15, 0.50],  # edge+bulge
}
TRUST = [2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.2, 1.2, 0.5, 1.0, 1.0, 0.5, 0.5, 0.7, 0.6, 0.3]
OFFSET = {4: -1.2}  # cigar-shaped smooth galaxies are rare in Galaxy Zoo


def build_classifier(net, bus):
    bus.ramp([(n, d, a, b) for n, (d, a, b) in PREDICATES.items()])
    lam = np.asarray(TRUST, float)
    rows = []
    for c in range(10):
        mu = np.asarray(PROTO[c], float)
        w = {f: 2 * lam[i] * mu[i] for i, f in enumerate(PRED_ORDER)}
        rows.append((f"score{c}", w, -float(np.sum(lam * mu ** 2)) + OFFSET.get(c, 0.0)))
    return bus.lin(rows)


def build():
    net = Net()
    bus = Bus(net)
    maps = build_features(net, bus)
    build_derived(net, bus)
    out = build_classifier(net, bus)
    assert net.layers[-1]["name"] == out and net.shape[out] == (10,)
    return net, bus, maps


if __name__ == "__main__":
    net, bus, maps = build()
    net.save(sys.argv[1] if len(sys.argv) > 1 else ".")
