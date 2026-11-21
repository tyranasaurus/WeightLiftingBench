"""Writes a hand-designed galaxy classifier: python3 build.py OUT_DIR  -> OUT_DIR/network.json, OUT_DIR/weights.npz
Every weight comes from formulas and hand-chosen constants in this file (no data is read)."""
import json, os, sys
import numpy as np

S = 128
CEN = 63.5
YY, XX = np.mgrid[:S, :S].astype(np.float64)
DX, DY = XX - CEN, YY - CEN
RR = np.hypot(DX, DY)


class Net:
    def __init__(self):
        self.layers, self.w, self.shape, self.n = [], {}, {"image": (3, S, S)}, 0

    def _nm(self, p):
        self.n += 1
        return f"{p}{self.n}"

    def _add(self, L, shape):
        self.layers.append(L); self.shape[L["name"]] = tuple(shape); return L["name"]

    def conv(self, x, W, b=None, pad=0, stride=1, p="c"):
        W = np.asarray(W, np.float64)
        if W.ndim == 2: W = W[:, :, None, None]
        o, i, k, _ = W.shape
        c, h, w = self.shape[x]
        assert c == i, (x, c, i)
        b = np.zeros(o) if b is None else np.broadcast_to(np.asarray(b, np.float64), (o,))
        nm = self._nm(p)
        self.w[nm + ".W"], self.w[nm + ".b"] = W.astype(np.float32), np.array(b, np.float32)
        ho, wo = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
        return self._add({"name": nm, "op": "conv", "inputs": [x], "out": o, "kernel": [k, k], "stride": stride, "padding": pad}, (o, ho, wo))

    def dense(self, x, W, b=None, p="d"):
        W = np.asarray(W, np.float64)
        n_in = int(np.prod(self.shape[x]))
        assert W.shape[1] == n_in, (x, W.shape, n_in)
        b = np.zeros(W.shape[0]) if b is None else np.asarray(b, np.float64)
        nm = self._nm(p)
        self.w[nm + ".W"], self.w[nm + ".b"] = W.astype(np.float32), b.astype(np.float32)
        return self._add({"name": nm, "op": "dense", "inputs": [x], "out": W.shape[0]}, (W.shape[0],))

    def relu(self, x):
        return self._add({"name": self._nm("r"), "op": "relu", "inputs": [x]}, self.shape[x])

    def pool(self, x, size, stride, op="maxpool"):
        c, h, w = self.shape[x]
        ho, wo = (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1
        return self._add({"name": self._nm("p"), "op": op, "inputs": [x], "size": list(size), "stride": list(stride)}, (c, ho, wo))

    def add(self, *xs):
        assert all(self.shape[x] == self.shape[xs[0]] for x in xs), [self.shape[x] for x in xs]
        return self._add({"name": self._nm("a"), "op": "add", "inputs": list(xs)}, self.shape[xs[0]])

    def reshape(self, x, shape):
        assert np.prod(shape) == np.prod(self.shape[x])
        return self._add({"name": self._nm("s"), "op": "reshape", "inputs": [x], "shape": list(shape)}, shape)

    def norm(self, x):
        return self._add({"name": self._nm("n"), "op": "norm", "inputs": [x]}, self.shape[x])

    # ---- helpers -------------------------------------------------------------------------------
    def n_of(self, x):
        return int(np.prod(self.shape[x]))

    def const_map(self, src1, values):
        """A constant tensor (any shape) produced by a dense layer with zero weights on a 1-element input."""
        values = np.asarray(values, np.float64)
        d = self.dense(src1, np.zeros((values.size, self.n_of(src1))), values.ravel(), p="k")
        return self.reshape(d, values.shape)

    def pick(self, x, chans, scale=1.0, bias=0.0, pad=0):
        c = self.shape[x][0]
        W = np.zeros((len(chans), c))
        for o, ch in enumerate(chans): W[o, ch] = scale
        return self.conv(x, W, np.full(len(chans), bias), pad=pad)

    def box(self, x, k):
        c = self.shape[x][0]
        p = self.conv(x, np.eye(c), pad=k // 2)
        a = self.pool(p, [k, 1], [1, 1], "avgpool")
        return self.pool(a, [1, k], [1, 1], "avgpool")

    def dilate(self, x, k):
        c = self.shape[x][0]
        p = self.conv(x, np.eye(c), pad=k // 2)
        a = self.pool(p, [k, 1], [1, 1])
        return self.pool(a, [1, k], [1, 1])


class Bus:
    """Collects named scalars from several tensors into one vector via dense layers and an add."""
    def __init__(self, net):
        self.net, self.items, self.names = net, [], []

    def put(self, src, rows, names):
        """rows: (k, n_in(src)) weight rows producing k scalars."""
        self.items.append((src, np.asarray(rows, np.float64), None))
        self.names += list(names)

    def build(self):
        n = len(self.names); outs = []; off = 0
        for src, rows, bias in self.items:
            W = np.zeros((n, self.net.n_of(src)))
            W[off:off + len(rows)] = rows
            b = np.zeros(n)
            if bias is not None: b[off:off + len(rows)] = bias
            outs.append(self.net.dense(src, W, b, p="bus")); off += len(rows)
        v = outs[0] if len(outs) == 1 else self.net.add(*outs)
        return v, {nm: i for i, nm in enumerate(self.names)}


# ====================================================================================================
# hand-chosen constants
LEVELS = [0.05, 0.18, 0.38, 0.62]   # isophote levels above the sky, in display units (0..1)
GAIN = 50.0                         # mask transition width 1/GAIN
CONN_ITERS = 12                     # 32x32 connectivity flood: each step grows 4 full-res px
CORE_ITERS = 7                      # 64x64 level-3 flood: each step grows 4 full-res px
NR, NT = 24, 32                     # log-polar rings and angles
R0, R1 = 2.5, 56.0                  # ring radii range (full-res px)
PEDESTAL = 0.10                     # display level subtracted before the polar analysis
NPH = 16                            # phases for amplitude-by-max
PGRID = [0.0, 2.0, 3.5, 5.0, 7.0, 9.5, 13.0]   # radial wavenumbers for m=2 (pitch = atan(2/p))

H2 = 64                             # working resolution (2x2-averaged input)
c2 = (H2 - 1) / 2
Y2, X2 = np.mgrid[:H2, :H2].astype(np.float64)
DX2, DY2 = 2 * (X2 - c2), 2 * (Y2 - c2)       # offsets from the centre in full-res pixels
RR2 = np.hypot(DX2, DY2)


def ring_geometry():
    du = np.log(R1 / R0) / (NR - 1)
    r = R0 * np.exp(du * np.arange(NR))
    th = 2 * np.pi * np.arange(NT) / NT
    return r, th, du


def polar_matrix():
    """Rows (NR*NT): ring samples minus their ring mean; then NR rows: ring means. Input: 64x64 map."""
    r, th, du = ring_geometry()
    dth = 2 * np.pi / NT
    Wt = np.zeros((NR, NT, H2 * H2))
    for i, ri in enumerate(r):
        sr = max(1.2, 0.6 * ri * du); st = max(1.2, 0.6 * ri * dth)
        for j, t in enumerate(th):
            ex, ey = DX2 - ri * np.cos(t), DY2 - ri * np.sin(t)
            rad = ex * np.cos(t) + ey * np.sin(t)
            tan = -ex * np.sin(t) + ey * np.cos(t)
            g = np.exp(-0.5 * ((rad / sr) ** 2 + (tan / st) ** 2))
            g[g < 1e-4] = 0
            Wt[i, j] = (g / g.sum()).ravel()
    mean = Wt.mean(1)
    return np.concatenate([(Wt - mean[:, None, :]).reshape(NR * NT, H2 * H2), mean])


def upsample_matrix(n_in, reps):
    """Nearest-neighbour 2x upsampling of a (1,n,n) map to `reps` channels of (2n,2n)."""
    n2 = 2 * n_in
    U = np.zeros((n2 * n2, n_in * n_in))
    yy, xx = np.mgrid[:n2, :n2]
    U[np.arange(n2 * n2), ((yy // 2) * n_in + xx // 2).ravel()] = 1.0
    return np.tile(U, (reps, 1))


def build():
    net = Net()
    K = len(LEVELS)
    img2 = net.pool("image", [2, 2], [2, 2], "avgpool")                  # (3,64,64)
    X = net.conv(img2, [[1 / 3, 1 / 3, 1 / 3], [1, 0, -1]], p="lum")     # L, D=R-B
    # ---- sky level: minimum of 16x16 block means (32x32 full-res) ----------------------------------
    blk = net.pool(X, [16, 16], [16, 16], "avgpool")
    nbg = net.pool(net.conv(blk, [[-1.0, 0.0]]), [4, 4], [4, 4])        # (1,1,1) = -sky
    bgm = net.reshape(net.dense(nbg, np.r_[np.ones(H2 * H2), np.zeros(H2 * H2)][:, None]), [2, H2, H2])
    Xb = net.add(X, bgm)                                                 # L - sky, D
    Ls = net.pool(net.pick(Xb, [0], pad=1), [3, 3], [1, 1], "avgpool")   # smoothed L - sky
    # ---- isophote masks: level 0 from the smoothed map, others from the plain map -------------------
    lv = np.array(LEVELS)
    Wa = np.zeros((2 * K, 1)); ba = np.zeros(2 * K)
    Wa[0, 0] = Wa[K, 0] = GAIN; ba[0] = -GAIN * lv[0]; ba[K] = -GAIN * lv[0] - 1
    Wb = np.zeros((2 * K, 2)); bb = np.zeros(2 * K)
    for k in range(1, K):
        Wb[k, 0] = Wb[K + k, 0] = GAIN; bb[k] = -GAIN * lv[k]; bb[K + k] = -GAIN * lv[k] - 1
    rz = net.relu(net.add(net.conv(Ls, Wa, ba), net.conv(Xb, Wb, bb)))
    mm1 = net.conv(rz, np.hstack([np.eye(K), -np.eye(K)]), -np.ones(K))  # clip(z,0,1) - 1, (K,64,64)
    # ---- connectivity of levels 0,1,2 on a 32x32 grid (fraction of each 4x4 full-res block) ---------------
    f32 = net.pool(net.pick(mm1, [0, 1, 2], 1.0, 1.0), [2, 2], [2, 2], "avgpool")   # (3,32,32)
    z32 = net.relu(net.conv(f32, np.vstack([5.0 * np.eye(3), 5.0 * np.eye(3)]), [-0.75] * 3 + [-1.75] * 3))  # ramp 0.15..0.35
    m32 = net.conv(z32, np.hstack([np.eye(3), -np.eye(3)]), -np.ones(3))                  # mask32 - 1
    seed32 = (np.hypot(*(np.mgrid[:32, :32] - 15.5)) <= 0.8).astype(float)
    s = net.relu(net.add(m32, net.const_map(nbg, np.stack([seed32] * 3))))
    for _ in range(CONN_ITERS):
        s = net.relu(net.add(net.dilate(s, 3), m32))
    c32 = net.dilate(s, 3)                                               # connected regions, grown by 1 cell
    Ms = []
    for k in range(3):
        ck = net.reshape(net.dense(net.pick(c32, [k]), upsample_matrix(32, 1)), [1, H2, H2])
        Ms.append(net.relu(net.add(net.pick(mm1, [k]), ck)))             # level-k mask inside its connected region
    # ---- level 3 (bright core): flood from the centre on the 64x64 grid (5x5 steps = 4 full-res px) -------
    m3 = net.pick(mm1, [3])
    seed2 = (RR2 <= 3.2).astype(float)
    sc = net.relu(net.add(m3, net.const_map(nbg, seed2[None])))
    for _ in range(CORE_ITERS):
        sc = net.relu(net.add(net.dilate(sc, 5), m3))
    Ms.append(sc)
    bus = Bus(net)
    # ---- mask moments (full-res pixel units) ------------------------------------------------------------
    NA = 16
    al = np.pi * np.arange(NA) / NA
    for k in range(K):
        rowsS = np.zeros((2, NA, H2 * H2))
        for j, a in enumerate(al):
            pr = 4 * ((DX2 * np.cos(a) + DY2 * np.sin(a)) ** 2).ravel()
            rowsS[0, j] = pr; rowsS[1, j] = -pr
        Sd = net.dense(Ms[k], rowsS.reshape(2 * NA, H2 * H2), p="mom")
        lam = net.pool(net.reshape(Sd, [2, 1, NA]), [1, NA], [1, NA])   # (2,1,1): lmax, -lmin
        bus.put(lam, np.eye(2), [f"lmax{k}", f"nlmin{k}"])
        bus.put(Ms[k], [4 * np.ones(H2 * H2), 4 * DX2.ravel(), 4 * DY2.ravel()], [f"area{k}", f"mx{k}", f"my{k}"])
    # ---- companions: level-3 cores not connected to the central core -----------------------------------
    notc = net.conv(net.dilate(sc, 3), [[-1.0]])
    near = (RR2 < 26).astype(float).ravel() * 4
    for nm, k in (("comp2", 2), ("comp3", 0)):        # inside the level-2 envelope (close pair) / the level-0 region
        inside = net.relu(net.add(m3, Ms[k]))
        bus.put(net.relu(net.add(inside, notc)), [near], [nm])
    # ---- log-polar analysis of (L - pedestal) inside the region --------------------------------------
    Lm = net.relu(net.add(net.pick(Xb, [0]), net.conv(Ms[0], [[1.0]], [-1.0 - PEDESTAL])))
    P = net.dense(Lm, polar_matrix(), p="polar")                        # NR*NT pattern values, NR ring means
    r, th, du = ring_geometry()
    u = np.log(r / R0)
    temps, tnames = [], []
    ph = 2 * np.pi * np.arange(NPH) / NPH
    def add_t(name, m, p, ringw):
        for a in ph:
            T = np.zeros(NR * NT + NR)
            T[:NR * NT] = (ringw[:, None] * np.cos(m * th[None, :] + p * u[:, None] - a)).ravel()
            temps.append(T)
        tnames.append(name)
    allr = np.ones(NR)
    for m in (1, 2, 3):
        for p in PGRID:
            pm = p * m / 2.0
            add_t(f"F{m}_{p}", m, pm, allr)
            if p > 0: add_t(f"F{m}_-{p}", m, -pm, allr)
    add_t("F4_0", 4, 0.0, allr)
    for i in range(NR):
        e = np.zeros(NR); e[i] = 1
        add_t(f"R2_{i}", 2, 0.0, e)
        add_t(f"R1_{i}", 1, 0.0, e)
    nt = len(tnames)
    Fo = net.dense(P, np.array(temps), p="four")
    amp = net.pool(net.reshape(Fo, [nt, 1, NPH]), [1, NPH], [1, NPH])  # (nt,1,1)
    bus.put(amp, np.eye(nt), tnames)
    mrow = np.zeros((NR, NR * NT + NR)); mrow[:, NR * NT:] = np.eye(NR)
    bus.put(P, mrow, [f"mu{i}" for i in range(NR)])
    # ---- colour in the unsaturated part of the level-1 region ----------------------------------------
    zs = net.relu(net.conv(Xb, [[25.0, 0.0], [25.0, 0.0]], [-25 * 0.77, -25 * 0.77 - 1]))
    rg = net.relu(net.add(Ms[1], net.conv(zs, [[-1.0, 1.0]])))           # region in [0,1]
    X3 = net.conv(Xb, [[0, 1], [0, -1], [1, 0]])                         # D, -D, L-sky
    G3 = net.relu(net.add(X3, net.conv(rg, [[1.0], [1.0], [1.0]], [-1.0, -1.0, -1.0])))
    one = np.ones(H2 * H2); zero = np.zeros(H2 * H2)
    bus.put(G3, [np.r_[one, -one, zero], np.r_[zero, zero, one]], ["colD", "colL"])
    vec, idx = bus.build()
    return net, vec, idx


def gfun(t):
    """Output of a norm group [a, -a, b, -b] at entry a, for t = a / b."""
    t = np.asarray(t, np.float64)
    return np.sqrt(2) * t / np.sqrt(1 + t * t)


class Head:
    """Stage A: dense+relu on the bus (signed splits, ramps); stage B: ratio groups through norm;
    stage C: fuzzy memberships (ramps), ANDs and class evidence."""
    def __init__(self, net, vec, idx):
        self.net, self.vec, self.idx = net, vec, idx
        self.A, self.G = [], []          # A: (name, {bus: coef}, bias); G: (name, num{A:coef}, den{A:coef}, den_const)
        self.ramps, self.units, self.scores = [], [], {}

    def a(self, name, terms, bias=0.0):
        self.A.append((name, terms, bias))

    def signed(self, name, terms):
        self.a(name + "+", terms); self.a(name + "-", {k: -v for k, v in terms.items()})

    def group(self, name, num, den, const=0.0):
        self.G.append((name, num, den, const))

    def build_ab(self):
        net = self.net
        nA = len(self.A); nb = net.n_of(self.vec)
        W = np.zeros((nA, nb)); b = np.zeros(nA)
        aidx = {}
        for i, (nm, terms, bias) in enumerate(self.A):
            for k, v in terms.items(): W[i, self.idx[k]] += v
            b[i] = bias; aidx[nm] = i
        Ar = net.relu(net.dense(self.vec, W, b, p="stA"))
        nG = len(self.G)
        W2 = np.zeros((4 * nG, nA)); b2 = np.zeros(4 * nG)
        for g, (nm, num, den, const) in enumerate(self.G):
            for k, v in num.items(): W2[4 * g, aidx[k]] += v; W2[4 * g + 1, aidx[k]] -= v
            for k, v in den.items(): W2[4 * g + 2, aidx[k]] += v; W2[4 * g + 3, aidx[k]] -= v
            b2[4 * g + 2] = const; b2[4 * g + 3] = -const
        Bn = net.norm(net.reshape(net.dense(Ar, W2, b2, p="stB"), [nG, 1, 4]))
        self.gv = net.reshape(Bn, [4 * nG])
        self.gidx = {nm: 4 * g for g, (nm, _, _, _) in enumerate(self.G)}

    # memberships: linear expression over g features, ramp from lo to hi -> [0,1]
    def ramp(self, name, expr, lo, hi):
        self.ramps.append((name, expr, lo, hi))

    def unit(self, name, terms, bias=0.0):
        """terms over memberships: name -> coef; a name prefixed '~' means (1 - membership)."""
        self.units.append((name, terms, bias))

    def build_c(self, classes=10):
        net = self.net
        nR = len(self.ramps); ng = net.n_of(self.gv)
        W = np.zeros((2 * nR, ng)); b = np.zeros(2 * nR)
        for i, (nm, expr, lo, hi) in enumerate(self.ramps):
            sc = 1.0 / (hi - lo)
            for k, v in expr.items():
                W[2 * i, self.gidx[k]] += v * sc; W[2 * i + 1, self.gidx[k]] += v * sc
            b[2 * i] = -lo * sc; b[2 * i + 1] = -lo * sc - 1
        R = net.relu(net.dense(self.gv, W, b, p="ramp"))
        ridx = {nm: i for i, (nm, _, _, _) in enumerate(self.ramps)}
        nU = len(self.units)
        W2 = np.zeros((nU, 2 * nR)); b2 = np.zeros(nU)
        for u, (nm, terms, bias) in enumerate(self.units):
            b2[u] = bias
            for k, v in terms.items():
                neg = k.startswith("~"); i = ridx[k.lstrip("~")]
                if neg:
                    b2[u] += v; W2[u, 2 * i] -= v; W2[u, 2 * i + 1] += v
                else:
                    W2[u, 2 * i] += v; W2[u, 2 * i + 1] -= v
        U = net.relu(net.dense(R, W2, b2, p="unit"))
        uidx = {nm: i for i, (nm, _, _) in enumerate(self.units)}
        W3 = np.zeros((classes, nU)); b3 = np.zeros(classes)
        for c, terms in self.scores.items():
            for k, v in terms.items():
                if k == "bias": b3[c] += v
                else: W3[c, uidx[k]] += v
        self.uidx = uidx
        return net.dense(U, W3, b3, p="score")


def spec(h):
    """Hand-written feature definitions, thresholds and class evidence."""
    K = len(LEVELS)
    for k in range(K):
        h.a(f"lmax{k}", {f"lmax{k}": 1.0}); h.a(f"lmin{k}", {f"nlmin{k}": -1.0}); h.a(f"area{k}", {f"area{k}": 1.0})
    h.signed("colD", {"colD": 1.0}); h.a("colL", {"colL": 1.0})
    def D(m, p, c=1.0):
        return {f"F{m}_{p}": c, f"F{m}_-{p}": -c}
    def merge(*ds):
        out = {}
        for d in ds:
            for k, v in d.items(): out[k] = out.get(k, 0.0) + v
        return out
    pos = [p for p in PGRID if p > 0]
    Sx = merge(*[D(2, p) for p in pos if p >= 3.5], *[D(1, p, 0.5) for p in pos], *[D(3, p, 0.5) for p in pos])
    h.signed("S", Sx)
    h.signed("T", merge(D(2, 5.0, 0.5), D(2, 7.0), D(2, 9.5), D(2, 13.0)))
    h.signed("Lo", merge(D(2, 2.0), D(2, 3.5), D(2, 5.0, 0.5)))
    for i in range(NR):
        h.a(f"na{i}a", {f"mu{i}": 1 / 0.05}); h.a(f"na{i}b", {f"mu{i}": 1 / 0.05}, -1.0)
    na = merge(*[{f"na{i}a": 1.0, f"na{i}b": -1.0} for i in range(NR)])
    for nm in ("F1_0.0", "comp2", "comp3"):
        h.a(nm, {nm: 1.0})
    for k in range(3):                 # |centroid moment| ~ sum of 8 rectified projections / 2.51
        for j in range(8):
            a = 2 * np.pi * j / 8
            h.a(f"off{k}_{j}", {f"mx{k}": np.cos(a), f"my{k}": np.sin(a)})
    h.a("odd", {f"F{m}_{p}": 1.0 for m in (1, 3) for p in PGRID} | {f"F{m}_-{p}": 1.0 for m in (1, 3) for p in PGRID if p > 0})
    # ---- ratio groups (t = num / den) ----
    for k in range(K):
        h.group(f"q{k}", {f"lmin{k}": 1.0}, {f"lmax{k}": 1.0})          # t = q^2
    h.group("col", {"colD+": 1.0, "colD-": -1.0}, {"colL": 0.3})       # t = colour / 0.3
    h.group("A0", {"area0": 1.0}, {}, 400.0)
    h.group("A1", {"area1": 1.0}, {}, 200.0)
    h.group("A3", {"area3": 1.0}, {}, 20.0)
    h.group("A2", {"area2": 1.0}, {}, 40.0)
    for k in range(3):
        h.group(f"off{k}", {f"off{k}_{j}": 1 / 2.51 for j in range(8)}, {f"area{k}": 3.0})   # t = offset / 3 px
    h.group("c20", {"area2": 1.0}, {"area0": 0.06})                    # t = (A2/A0)/0.06
    h.group("c31", {"area3": 1.0}, {"area1": 0.08})                    # t = (A3/A1)/0.08
    h.group("c10", {"area1": 1.0}, {"area0": 0.45})                    # t = (A1/A0)/0.45
    h.group("irr", {"odd": 1.0 / 26}, {k: 0.1 * v for k, v in na.items()})
    h.group("sp", {"S+": 1.0, "S-": 1.0}, {k: 0.3 * v for k, v in na.items()})
    h.group("tt", {"T+": 1.0, "T-": 1.0}, {"Lo+": 1.0, "Lo-": 1.0})
    h.group("lop", {"F1_0.0": 1.0}, {k: 0.3 * v for k, v in na.items()})
    h.group("comp3", {"comp3": 1.0}, {}, 10.0)
    h.group("comp2", {"comp2": 1.0}, {}, 30.0)
    h.build_ab()
    # ---- memberships (thresholds in physical units, mapped through gfun) ----
    gq = lambda q: float(gfun(q * q))
    def rq(name, k, qa, qb): h.ramp(name, {f"q{k}": 1.0}, gq(qa), gq(qb))
    def rr(name, g, scale, a, b): h.ramp(name, {g: 1.0}, float(gfun(a / scale)), float(gfun(b / scale)))
    rq("q1hi", 1, 0.68, 0.80); rq("q2hi", 2, 0.70, 0.82)
    rq("q1mid", 1, 0.40, 0.52); rq("q1low", 1, 0.27, 0.36); rq("q0low", 0, 0.24, 0.32); rq("q1el", 1, 0.38, 0.48)
    h.ramp("bulge", {"q3": 1.0, "q1": -1.0}, 0.06, 0.20)
    h.ramp("bar3", {"q1": 1.0, "q3": -1.0}, 0.10, 0.26)
    h.ramp("bar2", {"q1": 1.0, "q2": -1.0}, 0.10, 0.26)
    rr("a3ok", "A3", 20, 10, 25); rr("a2ok", "A2", 40, 20, 50)
    rr("irr", "irr", 0.1, 0.04, 0.12)
    rr("red", "col", 0.3, 0.0, 0.12); rr("blue", "col", 0.3, -0.15, 0.0)        # blue ramp used inverted
    rr("diffuse", "c20", 0.06, 0.03, 0.08)                              # used inverted: A2/A0 < 0.03
    rr("bulgy", "c31", 0.08, 0.05, 0.12)
    rr("off0", "off0", 3.0, 3.0, 7.0); rr("off1", "off1", 3.0, 2.5, 6.0); rr("off2", "off2", 3.0, 2.0, 5.0)
    rr("expd", "c10", 0.45, 0.36, 0.48)
    rr("sp", "sp", 0.3, 0.12, 0.35)
    rr("tight", "tt", 1.0, 0.7, 1.4)
    rr("big", "A0", 400, 400, 1000); rr("small", "A0", 400, 150, 400)
    rr("comp", "comp2", 30, 4, 16); rr("comp2", "comp3", 10, 8, 30)
    rr("lop", "lop", 0.3, 0.15, 0.45)
    # ---- units ----
    U = h.unit
    U("ROUND1", {"q1hi": 1}); U("ROUND2", {"q2hi": 1})
    U("MID1", {"q1mid": 1, "q1hi": -1})
    U("CIG1", {"q1low": 1, "q1mid": -1})
    U("THIN1", {"~q1low": 1}); U("THIN0", {"~q0low": 1}); U("ELONG", {"~q1el": 1})
    U("BULGE", {"bulge": 1, "a3ok": 1}, -1.0); U("NOBULGE", {"~bulge": 1})
    U("BAR3", {"bar3": 1, "a3ok": 1}, -1.0); U("BAR2", {"bar2": 1, "a2ok": 1}, -1.0)
    U("IRR", {"irr": 1}); U("IRRNR", {"irr": 1, "~red": 1}, -1.0)
    U("DIFFUSE", {"~diffuse": 1, "small": 1}, -1.0);
    U("OFF0", {"off0": 1}); U("OFF1", {"off1": 1}); U("OFF2", {"off2": 1})
    U("NOCORE", {"~a3ok": 1, "small": 1, "~red": 1}, -2.0); U("BULGY", {"bulgy": 1}); U("EXPD", {"expd": 1})
    U("RED", {"red": 1}); U("BLUE", {"~blue": 1})
    U("SP", {"sp": 1}); U("TIGHT", {"tight": 1}); U("LOOSE", {"~tight": 1})
    U("SPTIGHT", {"sp": 1, "tight": 1}, -1.0); U("SPLOOSE", {"sp": 1, "~tight": 1}, -1.0)
    U("BIG", {"big": 1}); U("SMALL", {"~small": 1})
    U("COMP", {"comp": 1}); U("COMP2", {"comp2": 1}); U("LOP", {"lop": 1})
    U("BLUENOSP", {"~blue": 1, "~sp": 1}, -1.0)
    U("BIGBLUE", {"big": 1, "~blue": 1}, -1.0)
    U("DISK", {"small": 1, "~red": 1, "q1low": 1}, -2.0)
    U("THIN_NOB", {"~q1low": 1, "~bulge": 1}, -1.0); U("THIN_BLUE", {"~q1low": 1, "~blue": 1}, -1.0)
    U("THIN_BIG", {"~q1low": 1, "big": 1}, -1.0)
    U("EL_BULGE", {"~q1el": 1, "bulge": 1, "a3ok": 1}, -2.0); U("EL_RED", {"~q1el": 1, "red": 1}, -1.0)
    # ---- class evidence ----
    off_pen = {"OFF0": -0.7, "OFF1": -0.7, "OFF2": -0.5}
    h.scores = {
        0: {"bias": 0.0, "BLUENOSP": 1.0, "NOCORE": 1.5, "DIFFUSE": 0.8, "IRRNR": 0.8, "LOP": 0.5, "OFF0": 0.3, "RED": -0.5, "COMP2": 0.3},
        1: {"bias": 0.0, "COMP": 2.5, "COMP2": 0.8, "LOP": 1.0, "OFF0": 1.0, "OFF1": 1.2, "OFF2": 0.8},
        2: {"bias": 0.5, "ROUND1": 2.0, "ROUND2": 0.8, "RED": 1.0, "BLUE": -1.0, "SP": -2.5, "BIG": -0.7, "EXPD": -0.5,
            "COMP": -1.5, "LOP": -1.5, "DIFFUSE": -1.0, "NOCORE": -1.0, **off_pen},
        3: {"bias": 0.5, "MID1": 2.5, "RED": 1.0, "BLUE": -1.0, "SP": -2.5, "BIG": -0.7, "EXPD": -0.3,
            "COMP": -1.5, "LOP": -1.5, "DIFFUSE": -1.0, "NOCORE": -1.0, **off_pen},
        4: {"bias": -0.5, "CIG1": 2.0, "ELONG": 0.5, "RED": 0.5, "SMALL": 0.5, "SP": -2.5, "THIN1": -1.0, "BULGE": -1.0, "LOP": -1.5, "BLUE": -1.0, **off_pen},
        5: {"bias": 0.0, "SP": 1.0, "BAR3": 1.2, "BAR2": 1.2, "DISK": 0.8, "ROUND1": 0.3, "NOCORE": -1.0, **off_pen},
        6: {"bias": 0.0, "SP": 0.8, "SPTIGHT": 1.0, "DISK": 1.2, "BULGY": 0.6, "EXPD": 0.3, "BLUE": -0.3, "DIFFUSE": -1.0, "NOCORE": -1.0, **off_pen},
        7: {"bias": 0.0, "SP": 1.0, "SPLOOSE": 1.0, "DISK": 0.6, "BLUE": 0.5, "EXPD": 0.3, "BULGY": -0.5, "IRR": 0.3, "NOCORE": -0.5, **off_pen},
        8: {"bias": 0.1, "THIN1": 2.0, "THIN0": 1.0, "THIN_NOB": 1.5, "THIN_BLUE": 1.0, "THIN_BIG": 0.5, "COMP": -1.5, "LOP": -0.5, **off_pen},
        9: {"bias": 0.0, "THIN1": 2.0, "ELONG": 1.0, "THIN0": 0.5, "EL_BULGE": 1.5, "EL_RED": 0.9, "COMP": -1.5, "LOP": -0.5, **off_pen},
    }
    return h.build_c()


def save(net, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump({"layers": net.layers}, f)
    np.savez(os.path.join(out, "weights.npz"), **net.w)


def make():
    net, vec, idx = build()
    h = Head(net, vec, idx)
    out = spec(h)
    return net, h, out


if __name__ == "__main__":
    net, h, out = make()
    save(net, sys.argv[1] if len(sys.argv) > 1 else ".")
