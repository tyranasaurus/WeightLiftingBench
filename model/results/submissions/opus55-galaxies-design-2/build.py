#!/usr/bin/env python3
"""Hand-designed galaxy-morphology network (no training, no data).

Usage: python3 build.py OUT_DIR   -> writes OUT_DIR/network.json and OUT_DIR/weights.npz

Every number below is a formula or a hand-chosen constant (see method.md).
"""
import sys, os, json
import numpy as np

# --------------------------------------------------------------------------------------------
# graph builder
# --------------------------------------------------------------------------------------------
class Net:
    def __init__(self):
        self.layers, self.w, self.shape, self.k = [], {}, {"image": (3, 128, 128)}, 0
        self.tags = {}

    def _n(self, p):
        self.k += 1
        return f"{p}{self.k}"

    def _add(self, d, shp):
        self.layers.append(d)
        self.shape[d["name"]] = tuple(int(v) for v in shp)
        return d["name"]

    def tag(self, name, x):
        self.tags[name] = x
        return x

    def conv(self, x, W, b=None, stride=1, pad=0):
        W = np.asarray(W, np.float64)
        o, i, k, k2 = W.shape
        C, H, Wd = self.shape[x]
        assert C == i and k == k2, (x, self.shape[x], W.shape)
        b = np.zeros(o) if b is None else np.broadcast_to(np.asarray(b, np.float64), (o,)).copy()
        n = self._n("conv")
        self.w[n + ".W"], self.w[n + ".b"] = W, b
        return self._add(dict(name=n, op="conv", inputs=[x], out=o, kernel=[k, k], stride=stride, padding=pad),
                         (o, (H + 2 * pad - k) // stride + 1, (Wd + 2 * pad - k) // stride + 1))

    def dense(self, x, W, b=None):
        W = np.asarray(W, np.float64)
        nin = int(np.prod(self.shape[x]))
        assert W.ndim == 2 and W.shape[1] == nin, (x, W.shape, self.shape[x])
        b = np.zeros(W.shape[0]) if b is None else np.broadcast_to(np.asarray(b, np.float64), (W.shape[0],)).copy()
        n = self._n("fc")
        self.w[n + ".W"], self.w[n + ".b"] = W, b
        return self._add(dict(name=n, op="dense", inputs=[x], out=W.shape[0]), (W.shape[0],))

    def relu(self, x):
        return self._add(dict(name=self._n("relu"), op="relu", inputs=[x]), self.shape[x])

    def norm(self, x):
        assert len(self.shape[x]) == 3
        return self._add(dict(name=self._n("norm"), op="norm", inputs=[x]), self.shape[x])

    def pool(self, x, op, size, stride=None):
        stride = size if stride is None else stride
        C, H, W = self.shape[x]
        return self._add(dict(name=self._n(op[:3]), op=op, inputs=[x], size=list(size), stride=list(stride)),
                         (C, (H - size[0]) // stride[0] + 1, (W - size[1]) // stride[1] + 1))

    def maxpool(self, x, size, stride=None):
        return self.pool(x, "maxpool", size, stride)

    def avgpool(self, x, size, stride=None):
        return self.pool(x, "avgpool", size, stride)

    def add(self, *xs):
        s = self.shape[xs[0]]
        assert len(xs) >= 2 and all(self.shape[v] == s for v in xs), [self.shape[v] for v in xs]
        return self._add(dict(name=self._n("add"), op="add", inputs=list(xs)), s)

    def reshape(self, x, shp):
        shp = tuple(int(v) for v in shp)
        assert int(np.prod(shp)) == int(np.prod(self.shape[x])), (self.shape[x], shp)
        return self._add(dict(name=self._n("rs"), op="reshape", inputs=[x], shape=list(shp)), shp)

    def size(self, x):
        return int(np.prod(self.shape[x]))

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "network.json"), "w") as f:
            json.dump({"layers": self.layers}, f)
        np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in self.w.items()})


# --------------------------------------------------------------------------------------------
# small-vector toolkit: every vector is a flat (n,) tensor; helpers build dense/relu/norm stacks
# --------------------------------------------------------------------------------------------
class Vec:
    """Symbolic linear expressions over the entries of a flat tensor (for building dense rows)."""

    def __init__(self, net, x):
        self.net, self.x, self.n = net, x, net.size(x)

    def e(self, i):
        r = np.zeros(self.n + 1)
        r[i] = 1.0
        return r  # last entry = constant term

    def one(self):
        r = np.zeros(self.n + 1)
        r[-1] = 1.0
        return r


def lin(net, x, rows):
    """rows: list of (n+1,) arrays (coefficients + constant). Returns dense layer output."""
    R = np.array(rows, np.float64)
    return net.dense(x, R[:, :-1], R[:, -1])


def pwl_hidden(knots):
    return list(knots)


def concat(net, parts):
    """parts: list of (tensor, list_of_rows). Builds one dense per tensor into a common vector and adds them."""
    total = sum(len(r) for _, r in parts)
    outs, off = [], 0
    for x, rows in parts:
        n = net.size(x)
        W = np.zeros((total, n))
        b = np.zeros(total)
        for j, r in enumerate(rows):
            W[off + j] = r[:n]
            b[off + j] = r[n]
        off += len(rows)
        outs.append(net.dense(x, W, b))
    return outs[0] if len(outs) == 1 else net.add(*outs)


def combine(net, parts):
    """parts: list of (tensor, rows) with equal row counts; returns sum over parts of their linear maps."""
    outs = []
    for x, rows in parts:
        R = np.array(rows, np.float64)
        outs.append(net.dense(x, R[:, :-1], R[:, -1]))
    return outs[0] if len(outs) == 1 else net.add(*outs)


def ratio(net, x, groups, lam=100.0):
    """groups: list of (den_row, [num_rows]) linear expressions over the flat tensor x (rows have n+1 entries).
    Returns a flat tensor with num/den for every numerator, in order (den must be > 0).
    Uses the norm layer on [lam*D, -lam*D, u1, -u1, ...]: out_u * lam*sqrt(2/N) = (u/D)/sqrt(1+sum u^2/(lam D)^2)."""
    J = max(len(nums) for _, nums in groups)
    N = 2 + 2 * J
    rows = []
    for den, nums in groups:
        rows.append(lam * den)
        rows.append(-lam * den)
        for u in nums:
            rows.append(u)
            rows.append(-u)
        for _ in range(J - len(nums)):
            rows.append(np.zeros_like(den))
            rows.append(np.zeros_like(den))
    v = lin(net, x, rows)
    v = net.reshape(v, (len(groups), 1, N))
    v = net.norm(v)
    v = net.reshape(v, (len(groups) * N,))
    W = np.zeros((sum(len(n) for _, n in groups), len(groups) * N))
    k = 0
    for g, (den, nums) in enumerate(groups):
        for j in range(len(nums)):
            W[k, g * N + 2 + 2 * j] = lam * np.sqrt(2.0 / N)
            k += 1
    return net.dense(v, W)


def pwl_layer(net, x, funcs, extra_rows=()):
    """funcs: list of (row, knots, values): f(row . [x,1]) piecewise linear through (knots, values), flat outside.
    extra_rows: linear expressions passed through unchanged (must be >= 0 ... we pass them as +/- pair).
    Returns flat tensor [f_1, ..., f_F, extra_1, ...]."""
    hid, outW = [], []
    nF = len(funcs) + len(extra_rows)
    cols = []  # (out index, coefficient) per hidden unit
    const = np.zeros(nF)
    for fi, (row, knots, vals) in enumerate(funcs):
        knots = np.asarray(knots, np.float64)
        vals = np.asarray(vals, np.float64)
        slopes = np.diff(vals) / np.diff(knots)
        slopes = np.concatenate([slopes, [0.0]])
        prev = 0.0
        const[fi] = vals[0]
        for kk, z in enumerate(knots):
            c = slopes[kk] - prev
            prev = slopes[kk]
            if c != 0.0:
                r = row.copy()
                r[-1] -= z
                hid.append(r)
                cols.append((fi, c))
    for ei, row in enumerate(extra_rows):
        hid.append(row.copy())
        cols.append((len(funcs) + ei, 1.0))
        hid.append(-row.copy())
        cols.append((len(funcs) + ei, -1.0))
    h = net.relu(lin(net, x, hid))
    W = np.zeros((nF, len(hid)))
    for j, (o, c) in enumerate(cols):
        W[o, j] += c
    return net.dense(h, W, const)


# --------------------------------------------------------------------------------------------
# geometry helpers (full-resolution pixel coordinates; centre of the frame is (63.5, 63.5))
# phase layout: P[c*4 + a*2 + b, i, j] = image[c, 2i+a, 2j+b]
# --------------------------------------------------------------------------------------------
CEN = 63.5


def phase_coords():
    """Returns X, Y arrays of shape (4, 64, 64): full-res coordinates (relative to centre) of each phase entry."""
    i = np.arange(64)
    X = np.zeros((4, 64, 64))
    Y = np.zeros((4, 64, 64))
    for a in range(2):
        for b in range(2):
            X[a * 2 + b] = (2 * i[None, :] + b) - CEN
            Y[a * 2 + b] = (2 * i[:, None] + a) - CEN
    return X, Y


def gauss1d(sig, rad):
    d = np.arange(-rad, rad + 1)
    g = np.exp(-0.5 * (d / sig) ** 2)
    return g / g.sum()


def phase_conv_kernel(k2d):
    """k2d: (2R+1, 2R+1) full-res kernel with R <= 2. Returns (4, 4, 3, 3) phase-space kernel:
    out phase (a,b) at (i,j) = sum k2d[dy, dx] * in(2i+a+dy, 2j+b+dx)."""
    R = k2d.shape[0] // 2
    K = np.zeros((4, 4, 3, 3))
    for a in range(2):
        for b in range(2):
            for a2 in range(2):
                for b2 in range(2):
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            dy, dx = 2 * di + a2 - a, 2 * dj + b2 - b
                            if abs(dy) <= R and abs(dx) <= R:
                                K[a * 2 + b, a2 * 2 + b2, di + 1, dj + 1] = k2d[dy + R, dx + R]
    return K


# --------------------------------------------------------------------------------------------
# constants chosen by hand (see method.md)
# --------------------------------------------------------------------------------------------
KAPPA = 2.0        # outer isophote: background + KAPPA * (mean |high-pass| noise level of the quietest block)
F_MID = 0.25       # mid isophote: 25% of the way from background to the central peak (display units)
F_IN = 0.60        # inner isophote: 60% of the way to the peak
DELTA = 0.02       # soft-mask transition width (display units, 0..1)
N_ITER = 30        # region-growing steps (3x3 at half resolution: 1 half-res px per step)
N_ITER_MID = 20    # steps for the mid-isophote region
SIG_S = 1.0        # full-resolution smoothing (px)
NFLOOR = 0.015     # minimum outer-isophote height above background (display units)


def stage_front(net):
    """phase split, smoothing, background/noise/peak estimates, thresholds, region growing, masks."""
    T = {}
    # ---- phase split: (3,128,128) -> (12,64,64) exactly
    W0 = np.zeros((12, 3, 2, 2))
    for c in range(3):
        for a in range(2):
            for b in range(2):
                W0[c * 4 + a * 2 + b, c, a, b] = 1.0
    P0 = net.conv("image", W0, stride=2)
    # ---- smoothing (Gaussian sigma=1 full-res px, 5x5) and high-pass of luminance
    g = gauss1d(SIG_S, 2)
    Kg = phase_conv_kernel(np.outer(g, g))
    Kd = phase_conv_kernel(np.pad(np.ones((1, 1)), 2))  # identity
    W1 = np.zeros((20, 12, 3, 3))
    for c in range(3):
        W1[c * 4:(c + 1) * 4, c * 4:(c + 1) * 4] = Kg
        W1[12:16, c * 4:(c + 1) * 4] += (Kd - Kg) / 3.0
        W1[16:20, c * 4:(c + 1) * 4] -= (Kd - Kg) / 3.0
    S1 = net.relu(net.conv(P0, W1, pad=1))
    T["S1"] = S1
    # ---- half-res maps: R64, G64, B64, |HP|64
    Wc = np.zeros((4, 20, 1, 1))
    for c in range(3):
        Wc[c, c * 4:(c + 1) * 4] = 0.25
    Wc[3, 12:20] = 0.25
    C1 = net.conv(S1, Wc)
    T["C1"] = C1
    # ---- background (min over 8x8 half-res blocks of block means) and noise (same for |HP|)
    C2 = net.avgpool(C1, [8, 8])
    C3 = net.maxpool(net.conv(C2, -np.eye(4)[:, :, None, None]), [8, 8])  # (4,1,1) = -[bgR,bgG,bgB,n]
    # ---- central peak of half-res luminance (max over central 4x4)
    Wp = np.zeros((16, 4, 64, 64))
    for k, (i, j) in enumerate([(i, j) for i in range(30, 34) for j in range(30, 34)]):
        Wp[k, 0:3, i, j] = 1.0 / 3.0
    Pk = net.maxpool(net.reshape(net.dense(C1, Wp.reshape(16, -1)), (1, 4, 4)), [4, 4])
    # ---- scalar vector  [bgR, bgG, bgB, n, P, relu(a_mid-b_mid), relu(a_in-b_in)]
    # parts in terms of C3 (= -bgR,-bgG,-bgB,-n) and Pk
    rows_c3 = []
    rows_p = []
    def r4(v):
        return np.array(list(v) + [0.0])
    bgL_c3 = r4([-1 / 3, -1 / 3, -1 / 3, 0])
    n_c3 = r4([0, 0, 0, -1])
    rows_c3 += [r4([-1, 0, 0, 0]), r4([0, -1, 0, 0]), r4([0, 0, -1, 0]), n_c3, np.zeros(5)]
    rows_p += [np.zeros(2)] * 4 + [np.array([1.0, 0.0])]
    # a_mid - b_mid = bgL + F_MID (P - bgL) - (bgL + 2 KAPPA n) = F_MID P - F_MID bgL - 2 KAPPA n
    rows_c3.append(-F_MID * bgL_c3 - 2 * KAPPA * n_c3)
    rows_p.append(np.array([F_MID, 0.0]))
    rows_c3.append(-F_IN * bgL_c3 - 3 * KAPPA * n_c3)
    rows_p.append(np.array([F_IN, 0.0]))
    rows_c3.append(-KAPPA * n_c3 + np.array([0, 0, 0, 0, NFLOOR]))    # relu(NFLOOR - KAPPA n)
    rows_p.append(np.zeros(2))
    Hs = net.relu(combine(net, [(C3, rows_c3), (Pk, rows_p)]))
    T["Hs"] = Hs
    # linear expressions over Hs (7 entries + const)
    def hrow(**kw):
        r = np.zeros(9)
        names = ["bgR", "bgG", "bgB", "n", "P", "dmid", "din", "nfl"]
        for k, v in kw.items():
            r[names.index(k)] = v
        return r
    bgL = hrow(bgR=1 / 3, bgG=1 / 3, bgB=1 / 3)
    t_out = bgL + KAPPA * hrow(n=1) + hrow(nfl=1)          # = bgL + max(KAPPA n, NFLOOR)
    t_mid = bgL + 2 * KAPPA * hrow(n=1) + hrow(dmid=1)
    t_in = bgL + 3 * KAPPA * hrow(n=1) + hrow(din=1)
    T["thr"] = dict(bgL=bgL, t_out=t_out, t_mid=t_mid, t_in=t_in, bgR=hrow(bgR=1), bgG=hrow(bgG=1), bgB=hrow(bgB=1),
                    n=hrow(n=1), P=hrow(P=1))
    # ---- half-res luminance smoothed again (Gaussian sigma=1 half-res px) for region growing
    g2 = gauss1d(1.0, 2)
    Wl = np.zeros((2, 4, 5, 5))
    for c in range(3):
        Wl[0, c] = np.outer(g2, g2) / 3.0 / DELTA
    L64s = net.conv(C1, Wl, pad=2)
    # ---- region growing from a central seed disk at a given isophote level
    ii, jj = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    disk = (((ii - 31.5) ** 2 + (jj - 31.5) ** 2) <= 2.5 ** 2).astype(np.float64)

    def grow(t, n_iter):
        """returns (1,64,64) region connected to the seed where smoothed half-res luminance > t, dilated 1 px.
        R' = clamp(sum3x3(R) + K (M - 1), 0, 1) with M = clamp((L - t)/DELTA, 0, 1)."""
        WE = np.zeros((2 * 4096, 8))
        bE = np.zeros(2 * 4096)
        WE[:4096] = -t[:8] / DELTA
        bE[:4096] = -t[8] / DELTA
        bE[4096:] = disk.ravel()
        E64 = net.reshape(net.dense(Hs, WE, bE), (2, 64, 64))
        D64 = net.add(L64s, E64)  # [ (L-t)/DELTA, disk ]
        Ws = np.zeros((3, 2, 1, 1))
        Ws[0, 0] = 1
        Ws[1, 0] = 1
        Ws[2, 1] = 1
        st = net.relu(net.conv(D64, Ws, b=[0, -1, 0]))  # [relu(d), relu(d-1), disk]
        K = 9.0
        for it in range(n_iter):
            Wi = np.zeros((3, 3, 3, 3))
            bi = np.array([-K, -K - 1, 0]) - np.array([0.02, 0.02, 0.0])   # margin: float32 M is not exactly 1
            if it == 0:
                for o in (0, 1):
                    Wi[o, 2] = 1.0            # seed disk (5x5 sum not needed: 3x3 sum of disk)
                    Wi[o, 0, 1, 1] += K       # + K M
                    Wi[o, 1, 1, 1] -= K
                Wi[2, 0, 1, 1] = 1            # M = relu(d) - relu(d-1)
                Wi[2, 1, 1, 1] = -1
            else:
                for o in (0, 1):
                    Wi[o, 0] = 1.0            # R = c0 - c1
                    Wi[o, 1] = -1.0
                    Wi[o, 2, 1, 1] += K       # + K M
                Wi[2, 2, 1, 1] = 1
            st = net.relu(net.conv(st, Wi, b=bi, pad=1))
        Wr = np.zeros((1, 3, 1, 1))
        Wr[0, 0] = 1
        Wr[0, 1] = -1
        return net.maxpool(net.conv(st, Wr, pad=1), [3, 3], [1, 1])

    Rdil = grow(t_out, N_ITER)
    Rmid = grow(t_mid, N_ITER_MID)
    T["Rdil"] = Rdil
    T["Rmid"] = Rmid
    # ---- full-res soft masks at three isophote levels (phase layout), gated by the grown region
    Wg = np.zeros((24, 20, 1, 1))
    for l in range(3):
        for ph in range(4):
            for c in range(3):
                Wg[l * 8 + ph, c * 4 + ph] = 1.0 / 3.0 / DELTA
                Wg[l * 8 + 4 + ph, c * 4 + ph] = 1.0 / 3.0 / DELTA
    G1 = net.conv(S1, Wg)
    WE = np.zeros((24, 4096, 8))
    bE = np.zeros((24, 4096))
    for l, t in enumerate([t_out, t_mid, t_in]):
        for ph in range(4):
            WE[l * 8 + ph] = -t[:8] / DELTA
            bE[l * 8 + ph] = -t[8] / DELTA
            WE[l * 8 + 4 + ph] = -t[:8] / DELTA
            bE[l * 8 + 4 + ph] = -t[8] / DELTA - 1.0
    E12 = net.reshape(net.dense(Hs, WE.reshape(-1, 8), bE.ravel()), (24, 64, 64))
    G2 = net.relu(net.add(G1, E12))
    Wm = np.zeros((12, 24, 1, 1))
    for l in range(3):
        for ph in range(4):
            Wm[l * 4 + ph, l * 8 + ph] = 1
            Wm[l * 4 + ph, l * 8 + 4 + ph] = -1
    G4 = net.conv(G2, Wm, b=-np.ones(12))
    Wro = np.zeros((12, 1, 1, 1))
    Wro[0:4] = 1.0
    Wrm = np.zeros((12, 1, 1, 1))
    Wrm[4:12] = 1.0
    R12 = net.add(net.conv(Rdil, Wro), net.conv(Rmid, Wrm))   # outer level: outer region; mid & inner: mid region
    Wmask = net.relu(net.add(G4, R12))  # (12,64,64): level*4 + phase
    T["W"] = Wmask
    # ---- grayscale reconstruction by dilation from the central seed: R <- min(dilate3x3(R), L).
    # Keeps all light reachable from the nucleus along non-increasing paths (disk, arms, bars) and
    # flattens separate local domes (stars, knots, companions) to their saddle level.
    Wl1 = np.zeros((1, 4, 1, 1))
    Wl1[0, :3] = 1.0 / 3.0
    L64 = net.conv(C1, Wl1)
    negL = net.conv(C1, -Wl1)
    dmap = net.reshape(net.dense(Hs, np.zeros((4096, 8)), -disk.ravel()), (1, 64, 64))   # -disk
    R = net.add(L64, net.conv(net.relu(net.add(L64, dmap)), -np.ones((1, 1, 1, 1))))      # L - relu(L - disk)
    for it in range(N_ITER):
        D = net.maxpool(net.conv(R, np.ones((1, 1, 1, 1)), pad=1), [3, 3], [1, 1])
        R = net.add(D, net.conv(net.relu(net.add(D, negL)), -np.ones((1, 1, 1, 1))))      # min(D, L)
    T["Rgray"] = R
    T["L64"] = L64
    # ---- masked flux images (phase layout), background subtracted
    groups = [("R", "ann"), ("G", "ann"), ("B", "ann"), ("R", "in"), ("G", "in"), ("B", "in"),
              ("L", "all"), ("L", "ann"), ("L", "in"), ("C", "all")]
    nG = len(groups)
    Wx = np.zeros((nG * 4, 20, 1, 1))
    Ww = np.zeros((nG * 4, 12, 1, 1))
    WE = np.zeros((nG * 4, 4096, 8))
    bE = np.zeros((nG * 4, 4096))
    bgrow = dict(R=T["thr"]["bgR"], G=T["thr"]["bgG"], B=T["thr"]["bgB"], L=bgL, C=bgL)
    # min(L_s, Rgray) per phase = L_s - relu(L_s - Rgray)
    Wls = np.zeros((4, 20, 1, 1))
    for ph in range(4):
        for c in range(3):
            Wls[ph, c * 4 + ph] = 1.0 / 3.0
    Ls4 = net.conv(S1, Wls)
    nR4 = net.conv(R, -np.ones((4, 1, 1, 1)))
    ex4 = net.relu(net.add(Ls4, nR4))
    for gi, (ch, reg) in enumerate(groups):
        for ph in range(4):
            o = gi * 4 + ph
            if ch in ("L", "C"):
                for c in range(3):
                    Wx[o, c * 4 + ph] = 1.0 / 3.0
            else:
                Wx[o, "RGB".index(ch) * 4 + ph] = 1.0
            if reg == "ann":
                Ww[o, 0 * 4 + ph] = 1
                Ww[o, 1 * 4 + ph] = -1
            elif reg == "in":
                Ww[o, 1 * 4 + ph] = 1
            else:
                Ww[o, 0 * 4 + ph] = 1
            WE[o] = -bgrow[ch][:8]
            bE[o] = -bgrow[ch][8] - 1.0
    Hx = net.conv(S1, Wx)
    Hw = net.conv(Wmask, Ww)
    HE = net.reshape(net.dense(Hs, WE.reshape(-1, 8), bE.ravel()), (nG * 4, 64, 64))
    Wex = np.zeros((nG * 4, 4, 1, 1))                  # subtract relu(L_s - Rgray) for the cleaned group
    gC = groups.index(("C", "all"))
    for ph in range(4):
        Wex[gC * 4 + ph, ph] = -1.0
    Hc = net.conv(ex4, Wex)
    Hf = net.relu(net.add(Hx, Hw, HE, Hc))
    T["Hf"] = Hf
    T["groups"] = groups
    return T


# --------------------------------------------------------------------------------------------
# named scalar features: registry that gathers linear combinations from several flat tensors
# --------------------------------------------------------------------------------------------
class Reg:
    def __init__(self, net):
        self.net, self.loc = net, {}

    def put(self, tensor, names):
        assert self.net.size(tensor) == len(names), (self.net.shape[tensor], len(names))
        for i, n in enumerate(names):
            if n is not None:
                self.loc[n] = (tensor, i)
        return tensor

    def lin(self, exprs, names=None):
        """exprs: list of dicts {feature_name: coef, '1': const}. Returns flat tensor (registered if names)."""
        tensors = []
        for e in exprs:
            for nm in e:
                if nm != "1" and self.loc[nm][0] not in tensors:
                    tensors.append(self.loc[nm][0])
        assert tensors
        parts = []
        for t in tensors:
            n = self.net.size(t)
            rows = []
            for e in exprs:
                r = np.zeros(n + 1)
                for nm, c in e.items():
                    if nm != "1" and self.loc[nm][0] == t:
                        r[self.loc[nm][1]] += c
                rows.append(r)
            parts.append((t, rows))
        for j, e in enumerate(exprs):
            parts[0][1][j][-1] += e.get("1", 0.0)
        out = combine(self.net, parts)
        if names:
            self.put(out, names)
        return out

    def ratio(self, groups, names, lam=100.0):
        """groups: list of (den_expr, [num_exprs]); names: flat list of output names (num/den)."""
        flat = []
        idx = []
        for den, nums in groups:
            idx.append((len(flat), [len(flat) + 1 + j for j in range(len(nums))]))
            flat.append(den)
            flat.extend(nums)
        x = self.lin(flat)
        n = len(flat)
        def e(i):
            r = np.zeros(n + 1)
            r[i] = 1.0
            return r
        out = ratio(self.net, x, [(e(d), [e(u) for u in us]) for d, us in idx], lam=lam)
        return self.put(out, names)

    def pwl(self, funcs, names, extra=()):
        """funcs: list of (expr, knots, values). extra: list of (expr) passed linearly. Registers names."""
        exprs = [f[0] for f in funcs] + list(extra)
        x = self.lin(exprs)
        n = len(exprs)
        def e(i):
            r = np.zeros(n + 1)
            r[i] = 1.0
            return r
        out = pwl_layer(self.net, x, [(e(i), f[1], f[2]) for i, f in enumerate(funcs)],
                        extra_rows=[e(len(funcs) + i) for i in range(len(extra))])
        return self.put(out, names)

    def normvec(self, groups, names):
        """groups: list of lists of exprs (same length L). Each group -> [v, -v for v] normed; output v/rms*sqrt(?)
        Returns for each group the normalised entries v_i / sqrt(mean over the 2L entries of v^2) (registered)."""
        L = max(len(g) for g in groups)
        flat = []
        for g in groups:
            for v in g:
                flat.append(v)
                flat.append({k: -c for k, c in v.items()})
            for _ in range(L - len(g)):
                flat.append({"1": 0.0})
                flat.append({"1": 0.0})
        x = self.lin(flat)
        x = self.net.reshape(x, (len(groups), 1, 2 * L))
        x = self.net.norm(x)
        x = self.net.reshape(x, (len(groups) * 2 * L,))
        W = np.zeros((sum(len(g) for g in groups), len(groups) * 2 * L))
        k = 0
        for gi, g in enumerate(groups):
            for j in range(len(g)):
                W[k, gi * 2 * L + 2 * j] = 1.0
                k += 1
        out = self.net.dense(x, W)
        return self.put(out, names)


def F(**kw):
    """shorthand for a linear expression: F(a=1, b=-2, one=3)"""
    d = {}
    for k, v in kw.items():
        d["1" if k == "one" else k] = v
    return d


def E(*pairs, c=0.0):
    """E(('name', coef), ...) with names containing any characters"""
    d = {n: v for n, v in pairs}
    if c:
        d["1"] = c
    return d


LEVELS = ["o", "m", "i"]   # outer / mid / inner isophote


def q_of_out_t(ot):
    """axis ratio from out_t = sqrt(3)/sqrt(1+eps^2) where eps = |e|/t."""
    eps = np.sqrt(np.clip(3.0 / ot ** 2 - 1.0, 0, 1))
    return np.sqrt((1 - eps) / (1 + eps))


def stage_shape(net, reg, T):
    """second moments of the three isophote masks -> centred axis ratios, orientations, sizes."""
    X, Y = phase_coords()
    s = 1.0 / 32.0
    rows = []
    names = []
    for l, L in enumerate(LEVELS):
        for nm, f in [("m0", np.ones_like(X)), ("mx", X * s), ("my", Y * s), ("mxx", (X * s) ** 2),
                      ("myy", (Y * s) ** 2), ("mxy", X * Y * s * s)]:
            Wr = np.zeros((12, 64, 64))
            Wr[l * 4:(l + 1) * 4] = f
            rows.append(Wr.ravel())
            names.append(f"{nm}_{L}")
    mom = net.dense(T["W"], np.array(rows))
    reg.put(mom, names)
    # normalised moments (units of 32 px)
    groups, onames = [], []
    for L in LEVELS:
        den = E((f"m0_{L}", 1.0), c=1.0)
        nums = [E((f"mx_{L}", 1)), E((f"my_{L}", 1)), E((f"mxx_{L}", 1), (f"myy_{L}", 1)),
                E((f"mxx_{L}", 1), (f"myy_{L}", -1)), E((f"mxy_{L}", 2))]
        groups.append((den, nums))
        onames += [f"ux_{L}", f"uy_{L}", f"t_{L}", f"e1_{L}", f"e2_{L}"]
    reg.ratio(groups, onames, lam=1000.0)
    # squares of centroid offsets (PWL on [-1.5, 1.5] in units of 32 px)
    kn = np.linspace(-1.5, 1.5, 193)
    funcs, fn = [], []
    for L in LEVELS:
        for nm, ex in [("sxx", E((f"ux_{L}", 1))), ("syy", E((f"uy_{L}", 1))),
                       ("spp", E((f"ux_{L}", 1), (f"uy_{L}", 1))), ("smm", E((f"ux_{L}", 1), (f"uy_{L}", -1)))]:
            funcs.append((ex, kn, kn ** 2))
            fn.append(f"{nm}_{L}")
    reg.pwl(funcs, fn)
    # centred moments, scaled by 1e3 for the norm layer
    S = 1e3
    cen = []
    cn = []
    for L in LEVELS:
        cen.append(E((f"t_{L}", S), (f"sxx_{L}", -S), (f"syy_{L}", -S)))
        cen.append(E((f"e1_{L}", S), (f"sxx_{L}", -S), (f"syy_{L}", S)))
        cen.append(E((f"e2_{L}", S), (f"spp_{L}", -S / 2), (f"smm_{L}", S / 2)))
        cn += [f"tc_{L}", f"e1c_{L}", f"e2c_{L}"]
    reg.lin(cen, cn)
    # ellipticity via norm: out_t = sqrt3/sqrt(1+eps^2)
    reg.normvec([[E((f"tc_{L}", 1)), E((f"e1c_{L}", 1)), E((f"e2c_{L}", 1))] for L in LEVELS],
                sum([[f"ot_{L}", f"oe1_{L}", f"oe2_{L}"] for L in LEVELS], []))
    # orientation unit vector (times sqrt2)
    reg.normvec([[E((f"e1c_{L}", 1)), E((f"e2c_{L}", 1))] for L in LEVELS],
                sum([[f"n1_{L}", f"n2_{L}"] for L in LEVELS], []))
    # axis ratios via PWL of out_t; empty masks (out_t ~ 0) map to q = 1
    qk = np.concatenate([np.linspace(0.0, 0.2, 9), np.linspace(0.25, 1.0, 16)])
    epsk = (1 - qk ** 2) / (1 + qk ** 2)
    otk = np.sqrt(3.0) / np.sqrt(1 + epsk ** 2)
    knots = np.concatenate([[0.0, 1.0], otk])
    vals = np.concatenate([[1.0, 0.0], qk])
    # log2 of mask areas (PWL through powers of two)
    ak = 2.0 ** np.arange(0, 15)
    funcs = [(E((f"ot_{L}", 1)), knots, vals) for L in LEVELS]
    funcs += [(E((f"m0_{L}", 1)), ak, np.log2(ak)) for L in LEVELS]
    reg.pwl(funcs, [f"q_{L}" for L in LEVELS] + [f"la_{L}" for L in LEVELS])
    # multiplicity: K = 4 pi <r^2>_c / area equals (1+q^2)/q for a filled ellipse; dumbbells / separate
    # components give more. mult = K / ((1+q^2)/q)
    reg.ratio([(E((f"m0_{L}", 1.0), c=1.0), [E((f"tc_{L}", 4 * np.pi * 1024.0 / S))]) for L in LEVELS],
              [f"K_{L}" for L in LEVELS], lam=100.0)
    qg = np.concatenate([[0.05, 0.1, 0.15], np.linspace(0.2, 1.0, 17)])
    reg.pwl([(E((f"q_{L}", 1.0)), qg, (1 + qg ** 2) / qg) for L in LEVELS], [f"gq_{L}" for L in LEVELS])
    reg.ratio([(E((f"gq_{L}", 1.0)), [E((f"K_{L}", 1.0))]) for L in LEVELS], [f"mult_{L}" for L in LEVELS], lam=100.0)


def stage_flux(net, reg, T):
    """sums of masked, background-subtracted fluxes -> colours and light concentration."""
    groups = T["groups"]
    rows, names = [], []
    for gi, (ch, rg) in enumerate(groups):
        Wr = np.zeros((len(groups) * 4, 64, 64))
        Wr[gi * 4:(gi + 1) * 4] = 1.0
        rows.append(Wr.ravel())
        names.append(f"f{ch}_{rg}")
    reg.put(net.dense(T["Hf"], np.array(rows)), names)
    c0 = 0.5
    reg.ratio([(E(("fR_ann", 1), c=c0), [E(("fB_ann", 1)), E(("fG_ann", 1))]),
               (E(("fR_in", 1), c=c0), [E(("fB_in", 1)), E(("fG_in", 1))]),
               (E(("fL_all", 1), c=c0), [E(("fL_in", 1)), E(("fL_ann", 1))])],
              ["BR_ann", "GR_ann", "BR_in", "GR_in", "conc", "frac_ann"], lam=100.0)


def rot180_matrix(n):
    """dense matrix mapping an (n,n) map to its 180-degree rotation about the map centre."""
    idx = np.arange(n * n).reshape(n, n)
    M = np.zeros((n * n, n * n))
    M[idx.ravel(), idx[::-1, ::-1].ravel()] = 1.0
    return M


def stage_asym(net, reg, T):
    """rotational asymmetry (flux and shape) at half resolution, minimised over 5x5 centre shifts."""
    # half-res masked luminance (all) and outer mask
    gi = [g for g in T["groups"]].index(("L", "all"))
    Wg = np.zeros((1, net.shape[T["Hf"]][0], 1, 1))
    Wg[0, gi * 4:(gi + 1) * 4] = 0.25
    G64 = net.conv(T["Hf"], Wg)
    Wm = np.zeros((1, 12, 1, 1))
    Wm[0, 0:4] = 0.25
    M64 = net.conv(T["W"], Wm)
    out = []
    for src, nm in [(G64, "flux"), (M64, "shape")]:
        rot = net.reshape(net.dense(src, rot180_matrix(64)), (1, 64, 64))
        # 25 shifted copies of the rotated map (shift by s half-res px <=> centre moved by s/2)
        Ws = np.zeros((50, 1, 5, 5))
        Wo = np.zeros((50, 1, 1, 1))
        k = 0
        for dy in range(5):
            for dx in range(5):
                Ws[k, 0, dy, dx] = -1.0
                Ws[25 + k, 0, dy, dx] = 1.0
                Wo[k, 0] = 1.0
                Wo[25 + k, 0] = -1.0
                k += 1
        d = net.relu(net.add(net.conv(rot, Ws, pad=2), net.conv(src, Wo)))
        tot = net.avgpool(d, [64, 64])                       # (50,1,1) mean |diff| parts
        Wc = np.zeros((25, 50, 1, 1))
        for k in range(25):
            Wc[k, k] = -4096.0
            Wc[k, 25 + k] = -4096.0
        neg = net.conv(tot, Wc)                              # -sum |G - rot_k(G)|
        mn = net.maxpool(net.reshape(neg, (1, 25, 1)), [25, 1])   # -min_k
        # total of the source map
        tot0 = net.dense(src, np.ones((1, 4096)))
        out.append((mn, tot0, nm))
    for mn, tot0, nm in out:
        reg.put(net.reshape(mn, (1,)), [f"negA_{nm}"])
        reg.put(tot0, [f"sum_{nm}"])
    # A = min_k sum|G - G180_k| / (2 sum G)
    reg.ratio([(E(("sum_flux", 2), c=1.0), [E(("negA_flux", -1))]),
               (E(("sum_shape", 2), c=1.0), [E(("negA_shape", -1))])], ["A_flux", "A_shape"], lam=10.0)
    T["G64"] = G64


def stage_peaks(net, reg, T):
    """strongest extended secondary nucleus inside the grown region, outside the central 3 half-res px.
    nucleus score map = (L - max of ring r=3) - 1.5 (L - max of ring r=1): point sources (stars, knots) have
    most of their contrast already at r=1 and score low; extended nuclei score high."""
    C1 = T["C1"]
    Wl = np.zeros((1, 4, 1, 1))
    Wl[0, :3] = 1.0 / 3.0
    L64 = net.conv(C1, Wl)
    def ringmax(rad):
        offs = [(dy, dx) for dy in range(-rad, rad + 1) for dx in range(-rad, rad + 1) if max(abs(dy), abs(dx)) == rad]
        Wr = np.zeros((len(offs), 1, 2 * rad + 1, 2 * rad + 1))
        for k, (dy, dx) in enumerate(offs):
            Wr[k, 0, dy + rad, dx + rad] = 1.0
        ring = net.conv(L64, Wr, pad=rad)
        return net.reshape(net.maxpool(net.reshape(ring, (1, len(offs), 4096)), [len(offs), 1]), (1, 64, 64))
    r3 = ringmax(4)
    r1 = ringmax(1)
    ii, jj = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    rr = np.sqrt((ii - 31.5) ** 2 + (jj - 31.5) ** 2)
    outside = (rr > 3.0).astype(np.float64)
    cst = net.reshape(net.dense(T["Hs"], np.zeros((4096, 8)), (outside - 2.0).ravel()), (1, 64, 64))  # outside - 2
    # score = (L - r3) - 1.5 |L - r1|: must be a local maximum (slopes and plateau edges give L < r1, which is
    # penalised) and must not be point-like (a sharp peak has a large L - r1)
    nr1 = net.conv(r1, -np.ones((1, 1, 1, 1)))
    d1 = net.add(L64, nr1)                                           # L - r1
    a1 = net.relu(net.conv(d1, np.array([1.0, -1.0]).reshape(2, 1, 1, 1)))   # [relu(d1), relu(-d1)]
    sc_a = net.conv(a1, -1.5 * np.ones((1, 2, 1, 1)))                # -1.5 |L - r1|
    sc_3 = net.conv(r3, -np.ones((1, 1, 1, 1)))
    gate = net.relu(net.add(L64, sc_3, sc_a, T["Rmid"], cst))   # relu(score + R - 1 + outside - 1)
    pk = net.maxpool(gate, [64, 64])
    reg.put(net.reshape(pk, (1,)), ["peak2"])
    Hs = T["Hs"]
    reg.put(Hs, ["h_bgR", "h_bgG", "h_bgB", "h_n", "h_P", "h_dmid", "h_din", "h_nfl"])
    reg.ratio([(E(("h_P", 1), ("h_bgR", -1 / 3), ("h_bgG", -1 / 3), ("h_bgB", -1 / 3), c=0.02),
                [E(("peak2", 1))])], ["s2"], lam=10.0)


QBINS = [1.0, 0.88, 0.76, 0.64, 0.52, 0.40]
NPA = 8
P2 = [0.0, 2.5, -2.5, 5.0, -5.0, 8.0, -8.0, 12.0, -12.0]   # log-spiral radial frequencies for m=2
P13 = [0.0, 4.0, -4.0]                                      # for m=1 and m=3
MODES = [(2, p) for p in P2] + [(1, p) for p in P13] + [(3, p) for p in P13]


def bins():
    out = [(1.0, 0.0)]
    for q in QBINS[1:]:
        for k in range(NPA):
            out.append((q, k * np.pi / NPA))
    return out


def deproj(X, Y, q, th):
    xp = X * np.cos(th) + Y * np.sin(th)
    yp = (-X * np.sin(th) + Y * np.cos(th)) / q
    r = np.sqrt(xp ** 2 + yp ** 2)
    phi = np.arctan2(yp, xp)
    return r, phi


def stage_spiral(net, reg, T):
    """log-spiral (Kalnajs / 2DFFT-style) decomposition of the galaxy light (outer mask, centre tapered) in a
    bank of deprojected frames, plus the m=2 bar mode of the mid-isophote region. Among the frames consistent
    with the measured outer shape, the one with the least power is kept (best deprojection)."""
    X, Y = phase_coords()
    nH = net.shape[T["Hf"]][0]
    gi_all = T["groups"].index(("C", "all"))
    gi_in = T["groups"].index(("L", "in"))
    sel = np.zeros((8, nH, 1, 1))
    for ph in range(4):
        sel[ph, gi_all * 4 + ph] = 1.0
        sel[4 + ph, gi_in * 4 + ph] = 1.0
    Gd = net.conv(T["Hf"], sel[:4])
    Gi = net.conv(T["Hf"], sel[4:])
    B = bins()
    nb = len(B)
    rows_d, rows_i = [], [np.ones(16384)]
    for bi_, (q, th) in enumerate(B):
        r, phi = deproj(X, Y, q, th)
        w = np.clip((r - 2.0) / 2.0, 0, 1)            # taper the centre (bulge, pixel singularity)
        lr = np.log(np.maximum(r, 1e-3))
        rows_d.append(w.ravel())                      # per-frame m=0 normaliser
        for (m, p) in MODES:
            ang = m * phi + p * lr
            rows_d.append((w * np.cos(ang)).ravel())
            rows_d.append((w * np.sin(ang)).ravel())
        wi = np.clip(r - 0.5, 0, 1)
        rows_i.append((wi * np.cos(2 * phi)).ravel())
        rows_i.append((wi * np.sin(2 * phi)).ravel())
    Ad = net.dense(Gd, np.array(rows_d))
    Ai = net.dense(Gi, np.array(rows_i))
    nm = len(MODES)
    per = 1 + 2 * nm
    # amplitudes |A| ~ max over 8 rotations of (c cos a + s sin a)
    al = np.arange(8) * np.pi / 4
    Wd = np.zeros((nb * nm * 8, nb * per))
    for b in range(nb):
        for k in range(nm):
            for j, a in enumerate(al):
                o = (b * nm + k) * 8 + j
                Wd[o, b * per + 1 + 2 * k] = np.cos(a)
                Wd[o, b * per + 2 + 2 * k] = np.sin(a)
    ampd = net.maxpool(net.reshape(net.dense(Ad, Wd), (1, nb * nm, 8)), [1, 8])
    ampd = net.reshape(ampd, (nb * nm,))
    # inner bar mode: face-on frame -> |A|; inclined frames -> max over a in [-90,90] (excludes the
    # minor-axis oval produced by deprojecting a round bulge)
    Wi = np.zeros((nb * 8, 1 + nb * 2))
    for b, (q, th) in enumerate(B):
        for j, a in enumerate(al):
            aa = a if q == 1.0 else (-np.pi / 2 + j * np.pi / 7)
            Wi[b * 8 + j, 1 + b * 2] = np.cos(aa)
            Wi[b * 8 + j, 2 + b * 2] = np.sin(aa)
    ampi = net.reshape(net.maxpool(net.reshape(net.dense(Ai, Wi), (1, nb, 8)), [1, 8]), (nb,))
    nmd = []
    for b in range(nb):
        nmd += [f"A00_{b}"] + [None] * (2 * nm)
    reg.put(Ad, nmd)
    reg.put(Ai, ["A00_i"] + [None] * (nb * 2))
    an = [f"amp_{b}_{k}" for b in range(nb) for k in range(nm)]
    reg.put(ampd, an)
    bn = [f"bar_{b}" for b in range(nb)]
    reg.put(ampi, bn)
    groups = [(E((f"A00_{b}", 1), c=0.5), [E((f"amp_{b}_{k}", 1)) for k in range(nm)]) for b in range(nb)]
    groups.append((E(("A00_i", 1), c=0.5), [E((n, 1)) for n in bn]))
    reg.ratio(groups, [f"r{n}" for n in an] + [f"r{n}" for n in bn], lam=100.0)
    # ---- frame gates from the outer-isophote shape
    def step(expr, z0, z1):
        return (expr, [z0, z1], [0.0, 1.0])
    funcs, gn = [], []
    hw = 0.5 * (QBINS[0] - QBINS[1]) + 0.02
    for j, q in enumerate(QBINS):
        if j < len(QBINS) - 1:
            funcs.append(step(E(("q_o", 1)), q - hw - 0.01, q - hw + 0.01))      # q_o > q - hw
        else:
            funcs.append((E(("q_o", 1)), [-1.0, -0.99], [0.0, 1.0]))            # always on (lowest frame)
        gn.append(f"gq_lo_{j}")
        if j > 0:
            funcs.append(step(E(("q_o", 1)), q + hw - 0.01, q + hw + 0.01))      # q_o > q + hw
            gn.append(f"gq_hi_{j}")
    c0 = np.cos(2 * np.deg2rad(11.25 + 8.0))
    for k in range(NPA):
        th2 = 2 * k * np.pi / NPA
        ex = E(("n1_o", np.cos(th2) / np.sqrt(2)), ("n2_o", np.sin(th2) / np.sqrt(2)))
        funcs.append(step(ex, c0 - 0.015, c0 + 0.015))
        gn.append(f"gpa_{k}")
    reg.pwl(funcs, gn)
    gates = []
    for b, (q, th) in enumerate(B):
        j = QBINS.index(q)
        gq = E((f"gq_lo_{j}", 1))
        if j > 0:
            gq[f"gq_hi_{j}"] = -1.0
            k = int(round(th / (np.pi / NPA)))
            gq[f"gpa_{k}"] = 1.0
            gq["1"] = -1.0
        gates.append(gq)
    reg.pwl([(g, [0.0, 1.0], [0.0, 1.0]) for g in gates], [f"gate_{b}" for b in range(nb)])
    # ---- spectra: min over active frames  (inactive frames get +2), clipped to <= 1
    ex = []
    for k in range(nm):
        for b in range(nb):
            ex.append(E((f"ramp_{b}_{k}", -1), (f"gate_{b}", 2), c=-2.0))   # -(s + 2(1-g))
    for b in range(nb):
        ex.append(E((f"rbar_{b}", -1), (f"gate_{b}", 2), c=-2.0))
    gsp = reg.lin(ex)
    mx = net.maxpool(net.reshape(gsp, (1, nm + 1, nb)), [1, nb])       # = -min
    reg.put(net.reshape(mx, (nm + 1,)), [f"nS_{m}_{p}" for (m, p) in MODES] + ["nS_bar"])
    reg.pwl([(E((f"nS_{m}_{p}", -1.0)), [0.0, 1.0], [0.0, 1.0]) for (m, p) in MODES] +
            [(E(("nS_bar", -1.0)), [0.0, 1.0], [0.0, 1.0])],
            [f"S_{m}_{p}" for (m, p) in MODES] + ["S_bar"])
    reg.put(net.dense(net.relu(reg.lin([E((f"gate_{b}", 1)) for b in range(nb)])), np.ones((1, nb))), ["gsum"])


def reg_max(net, reg, groups, names):
    """groups: list of lists of exprs (equal length after padding with very negative constants); max of each."""
    L = max(len(g) for g in groups)
    ex = []
    for g in groups:
        ex += g + [{"1": -1e3}] * (L - len(g))
    x = reg.lin(ex)
    m = net.maxpool(net.reshape(x, (1, len(groups), L)), [1, L])
    return reg.put(net.reshape(m, (len(groups),)), names)


def reg_vnorm(net, reg, pairs, names, n_ang=16):
    """|(a, b)| ~ max over n_ang directions of a cos + b sin (error <= 1-cos(pi/n_ang))."""
    ex = []
    for a, b in pairs:
        for k in range(n_ang):
            al = 2 * np.pi * k / n_ang
            d = {}
            for kk, v in a.items():
                d[kk] = d.get(kk, 0.0) + v * np.cos(al)
            for kk, v in b.items():
                d[kk] = d.get(kk, 0.0) + v * np.sin(al)
            ex.append(d)
    x = reg.lin(ex)
    m = net.maxpool(net.reshape(x, (1, len(pairs), n_ang)), [1, n_ang])
    return reg.put(net.reshape(m, (len(pairs),)), names)


def stage_texture(net, reg, T):
    """CAS-style clumpiness: positive residual of the half-res luminance above a sigma=1.5 half-res blur,
    above a noise allowance, inside the disk annulus, relative to the annulus flux."""
    g = gauss1d(1.5, 3)
    k2 = np.outer(g, g)
    Wt = -k2[None, None].copy()
    Wt[0, 0, 3, 3] += 1.0
    res = net.conv(T["Rgray"], Wt, pad=3)
    # annulus mask at half res: mean over phases of (W_out - W_mid); gate: relu(res - n + ann - 1)
    Wm = np.zeros((1, 12, 1, 1))
    Wm[0, 0:4] = 0.25
    Wm[0, 4:8] = -0.25
    ann = net.conv(T["W"], Wm)
    thr = T["thr"]
    nmap = net.reshape(net.dense(T["Hs"], np.tile(-thr["n"][:8], (4096, 1)), np.full(4096, -thr["n"][8] - 1.0)),
                       (1, 64, 64))
    cl = net.relu(net.add(res, ann, nmap))
    reg.put(net.dense(cl, np.ones((1, 4096))), ["clump_sum"])
    gi = T["groups"].index(("L", "ann"))
    Wg = np.zeros((1, net.shape[T["Hf"]][0] * 4096))
    Wg[0, gi * 4 * 4096:(gi + 1) * 4 * 4096] = 0.25
    reg.put(net.dense(T["Hf"], Wg), ["ann_sum64"])
    reg.ratio([(E(("ann_sum64", 1), c=0.5), [E(("clump_sum", 1))])], ["clump"], lam=10.0)


def stage_derived(net, reg):
    s = 1.0 / np.sqrt(3.0)
    reg_vnorm(net, reg, [(E(("oe1_o", s)), E(("oe2_o", s))), (E(("oe1_m", s)), E(("oe2_m", s))),
                         (E(("oe1_i", s)), E(("oe2_i", s))),
                         (E(("oe1_m", s), ("oe1_o", -s)), E(("oe2_m", s), ("oe2_o", -s))),
                         (E(("oe1_i", s), ("oe1_o", -s)), E(("oe2_i", s), ("oe2_o", -s)))],
              ["chi_o", "chi_m", "chi_i", "dchi_m", "dchi_i"])
    # chirality excess of m=2 power: inclination / oval leakage is symmetric in +-p, spiral arms are not
    pp = [2.5, 5.0, 8.0, 12.0]
    reg.pwl([(E((f"S_2_{p}", 1.0), (f"S_2_{-p}", -1.0)), [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) for p in pp],
            [f"ax_{p}" for p in pp])
    A = lambda p: E((f"ax_{p}", 1.0))
    reg_max(net, reg, [[A(2.5), A(5.0), A(8.0), A(12.0)], [A(8.0), A(12.0)], [A(2.5), A(5.0)],
                       [E((f"S_{m}_{p}", 1.0)) for m in (1, 3) for p in P13],
                       [E(("q_o", 1.0)), E(("q_i", 1.0))], [E(("q_o", -1.0)), E(("q_m", -1.0))],
                       [E(("q_o", 1.0)), E(("q_m", 1.0))]],
            ["arm", "armT", "armL", "odd", "q_max_oi", "neg_q_e", "q_max_om"])
    reg.lin([E(("dchi_m", 1), ("chi_m", 1), ("chi_o", -1)), E(("dchi_i", 1), ("chi_i", 1), ("chi_o", -1))],
            ["bar_m", "bar_i"])


def T_(name_or_expr, x0, x1, v0, v1):
    ex = E((name_or_expr, 1.0)) if isinstance(name_or_expr, str) else name_or_expr
    return (ex, [x0, x1], [v0, v1])


# soft decision tree (Galaxy Zoo style). Each node logit z is a sum of clipped linear terms of features.
QE = E(("neg_q_e", -1.0))               # flattest of outer / mid isophotes (edge-on test)
NODES = {
    # merging: an extended second nucleus, and/or strong rotational asymmetry
    "M": ([T_("s2", 0.10, 0.30, -1.5, 1.5), T_("mult_m", 1.15, 1.60, -0.8, 1.2), T_("mult_i", 1.15, 1.60, -0.5, 1.0),
           T_("A_flux", 0.20, 0.50, -0.5, 0.8), T_("A_shape", 0.25, 0.50, -0.5, 0.5)], -0.3),
    # disturbed: asymmetric / irregular light
    "D": ([T_("A_flux", 0.15, 0.40, -1.5, 1.5), T_("A_shape", 0.20, 0.45, -1.0, 1.0), T_("clump", 0.0, 0.10, -0.5, 0.5),
           T_("BR_ann", 0.90, 1.30, -0.5, 0.5)], -0.8),
    # edge-on disk: very flat isophote
    "E": ([T_(QE, 0.20, 0.36, 2.0, -2.0)], 0.0),
    # bulge (for edge-on): inner isophote rounder than the flattest one, compact bright centre, red light
    "B": ([T_(E(("q_max_oi", 1.0), ("neg_q_e", 1.0)), 0.06, 0.25, -1.5, 1.5), T_(E(("la_o", 1.0), ("la_i", -1.0)), 2.5, 4.5, -1.0, 1.0),
           T_("BR_ann", 0.80, 1.10, 1.0, -1.0)], 0.0),
    # featured (disk with structure) vs smooth
    "F": ([T_("arm", 0.02, 0.08, -1.5, 1.5), T_("BR_ann", 0.78, 1.02, -1.5, 1.5),
           T_("clump", 0.0, 0.06, -0.5, 1.0), T_("la_o", 9.0, 12.0, -0.5, 0.5)], 0.0),
    # roundness of smooth galaxies
    "R": ([T_("q_max_om", 0.70, 0.86, -1.5, 1.5)], 0.0),
    "C": ([T_(QE, 0.38, 0.52, 1.5, -1.5)], 0.0),
    # bar: inner isophote more elongated than / misaligned with the disk
    "Bar": ([T_("bar_i", 0.25, 0.55, -1.0, 1.0), T_("bar_m", 0.20, 0.45, -0.5, 0.5), T_("S_bar", 0.05, 0.15, -0.5, 0.5)], -0.6),
    # tight (vs loose) arms
    "Ti": ([T_("BR_ann", 0.98, 1.26, 1.0, -1.0), T_(E(("armT", 1.0), ("armL", -1.0)), -0.04, 0.04, -0.5, 0.5),
            T_("clump", 0.01, 0.05, 0.5, -0.5), T_("arm", 0.04, 0.12, 0.5, -0.5)], 0.0),
}
# class paths: list of (node, required sign)
PATHS = {
    1: [("M", +1)],
    0: [("M", -1), ("D", +1)],
    8: [("M", -1), ("D", -1), ("E", +1), ("B", -1)],
    9: [("M", -1), ("D", -1), ("E", +1), ("B", +1)],
    2: [("M", -1), ("D", -1), ("E", -1), ("F", -1), ("R", +1)],
    3: [("M", -1), ("D", -1), ("E", -1), ("F", -1), ("R", -1), ("C", -1)],
    4: [("M", -1), ("D", -1), ("E", -1), ("F", -1), ("C", +1)],
    5: [("M", -1), ("D", -1), ("E", -1), ("F", +1), ("Bar", +1)],
    6: [("M", -1), ("D", -1), ("E", -1), ("F", +1), ("Bar", -1), ("Ti", +1)],
    7: [("M", -1), ("D", -1), ("E", -1), ("F", +1), ("Bar", -1), ("Ti", -1)],
}
VCAP = 3.0


def stage_decision(net, reg):
    funcs, tn = [], []
    for nd, (terms, bias) in NODES.items():
        for j, t in enumerate(terms):
            funcs.append(t)
            tn.append(f"term_{nd}_{j}")
    reg.pwl(funcs, tn)
    zex = []
    for nd, (terms, bias) in NODES.items():
        d = {f"term_{nd}_{j}": 1.0 for j in range(len(terms))}
        d["1"] = bias
        zex.append(d)
    reg.lin(zex, [f"z_{nd}" for nd in NODES])
    vf, vn = [], []
    for nd in NODES:
        vf.append((E((f"z_{nd}", 1.0)), [-VCAP, 0.0], [VCAP, 0.0]))   # violation of z > 0
        vn.append(f"vpos_{nd}")
        vf.append((E((f"z_{nd}", 1.0)), [0.0, VCAP], [0.0, VCAP]))    # violation of z < 0
        vn.append(f"vneg_{nd}")
    reg.pwl(vf, vn)
    sc = []
    for k in range(10):
        d = {}
        for nd, sg in PATHS[k]:
            d[f"vpos_{nd}" if sg > 0 else f"vneg_{nd}"] = -1.0
        sc.append(d)
    return reg.lin(sc, [f"score_{k}" for k in range(10)])


def build():
    net = Net()
    T = stage_front(net)
    reg = Reg(net)
    stage_shape(net, reg, T)
    stage_flux(net, reg, T)
    stage_asym(net, reg, T)
    stage_peaks(net, reg, T)
    stage_spiral(net, reg, T)
    stage_texture(net, reg, T)
    stage_derived(net, reg)
    out = stage_decision(net, reg)
    assert net.layers[-1]["name"] == out and net.shape[out] == (10,)
    return net, reg, T


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    net, reg, T = build()
    net.save(out_dir)
