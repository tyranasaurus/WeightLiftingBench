"""Hand-designed galaxy morphology network.

Every weight comes from an analytic formula (Gaussian kernels, polar resampling
grids, Fourier harmonics, radial masks) or from a rule table typed by hand from
general knowledge of galaxy morphology.  No training data is read.
"""
import json, os, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."


class Net:
    def __init__(self):
        self.layers = []
        self.w = {}
        self.shapes = {"image": (3, 128, 128)}

    def _rec(self, name, op, inputs, shp, **st):
        L = {"name": name, "op": op, "inputs": inputs}
        L.update(st)
        self.layers.append(L)
        self.shapes[name] = shp
        return name

    def conv(self, name, inp, W, b, stride=1, padding=0):
        W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
        o, c, kh, kw = W.shape
        _, h, w = self.shapes[inp]
        ho = (h + 2 * padding - kh) // stride + 1
        wo = (w + 2 * padding - kw) // stride + 1
        self.w[name + ".W"] = W; self.w[name + ".b"] = b
        return self._rec(name, "conv", [inp], (o, ho, wo), out=o, kernel=[kh, kw],
                         stride=stride, padding=padding)

    def dense(self, name, inp, W, b):
        W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
        n_in = int(np.prod(self.shapes[inp]))
        assert W.shape[1] == n_in, (name, W.shape, n_in)
        self.w[name + ".W"] = W; self.w[name + ".b"] = b
        return self._rec(name, "dense", [inp], (W.shape[0],), out=int(W.shape[0]))

    def relu(self, name, inp):
        return self._rec(name, "relu", [inp], self.shapes[inp])

    def pool(self, name, inp, size, stride, kind="max"):
        c, h, w = self.shapes[inp]
        ho = (h - size[0]) // stride[0] + 1
        wo = (w - size[1]) // stride[1] + 1
        return self._rec(name, kind + "pool", [inp], (c, ho, wo), size=list(size), stride=list(stride))

    def add(self, name, *ins):
        return self._rec(name, "add", list(ins), self.shapes[ins[0]])

    def reshape(self, name, inp, shape):
        return self._rec(name, "reshape", [inp], tuple(shape), shape=list(shape))

    def norm(self, name, inp):
        return self._rec(name, "norm", [inp], self.shapes[inp])

    def save(self):
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "network.json"), "w") as f:
            json.dump({"layers": self.layers}, f)
        np.savez_compressed(os.path.join(OUT, "weights.npz"), **self.w)


# ---------------------------------------------------------------- helpers
def gauss(k, sigma):
    a = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-0.5 * (a / sigma) ** 2)
    g = np.outer(g, g)
    return (g / g.sum()).astype(np.float32)


YY, XX = np.mgrid[0:128, 0:128].astype(np.float32)
CX = CY = 63.5
RR = np.sqrt((XX - CX) ** 2 + (YY - CY) ** 2)
TH = np.arctan2(YY - CY, XX - CX)


def ring(r0, r1):
    return ((RR >= r0) & (RR < r1)).astype(np.float32)


def disk(r1):
    return (RR < r1).astype(np.float32)


def mean_row(mask, scale=1.0):
    """dense row computing scale * mean of the map over mask"""
    m = mask.astype(np.float32)
    return (scale * m / m.sum()).reshape(1, -1)


def sum_row(mask, scale=1.0):
    return (scale * mask.astype(np.float32)).reshape(1, -1)


def wrow(mask, w, scale=1.0):
    """mean over mask of (map * w): normalised by the mask area, not the weighted sum"""
    m = mask.astype(np.float32)
    return (scale * m * w / m.sum()).reshape(1, -1)


net = Net()

# ---------------------------------------------------------------- stage 1: colour -> light
# channel 0: luminance (r,z weighted for S/N on red galaxies, g for young stars)
# channel 1: warm = R - B  (background is grey, so this is nearly sky-free)
Wpre = np.array([[0.30, 0.45, 0.25], [1.0, 0.0, -1.0]], np.float32).reshape(2, 3, 1, 1)
net.conv("pre", "image", Wpre, np.zeros(2, np.float32), padding=0)

# luminance only, for the smoothing ladder
Wlum = np.array([[1.0, 0.0]], np.float32).reshape(1, 2, 1, 1)
net.conv("lum", "pre", Wlum, np.zeros(1, np.float32))

net.conv("gA", "lum", gauss(5, 1.0).reshape(1, 1, 5, 5), np.zeros(1, np.float32), padding=2)
net.conv("gB", "gA", gauss(5, 1.0).reshape(1, 1, 5, 5), np.zeros(1, np.float32), padding=2)
net.conv("gC", "gB", gauss(7, 1.5).reshape(1, 1, 7, 7), np.zeros(1, np.float32), padding=3)

# normalised shape map (removes sky offset and brightness scale)
net.norm("nC", "gC")

# high pass for texture:  lum - gB   (sky offset cancels)
net.conv("hp", "gB", np.array([[-1.0]], np.float32).reshape(1, 1, 1, 1), np.zeros(1, np.float32))
net.conv("hp2", "lum", np.array([[1.0]], np.float32).reshape(1, 1, 1, 1), np.zeros(1, np.float32))
net.add("hpt", "hp", "hp2")
net.relu("hpos", "hpt")
net.conv("hneg", "hpt", np.array([[-1.0]], np.float32).reshape(1, 1, 1, 1), np.zeros(1, np.float32))
net.relu("hnegr", "hneg")
net.add("habs", "hpos", "hnegr")

# ---------------------------------------------------------------- stage 2: cartesian measurements
# Every source map writes into its own slots of one shared measurement vector;
# the vectors are summed with `add` (there is no concat op).
M = 48
SLOTS = {}


def slot(name):
    SLOTS[name] = len(SLOTS)
    return SLOTS[name]


for nm in ["bgC", "f6", "f14", "f28", "f44", "bgD", "d6", "d14", "d28", "d44",
           "mxx", "myy", "mxy", "sx", "sy", "f36",
           "ixx", "iyy", "ixy", "isx", "isy", "f10",
           "oxx", "oyy", "oxy", "f1036",
           "rnum", "warm14", "warm30", "lum14r", "lum30r", "bgr",
           "cl14", "cl30", "cl48", "noise", "noise2", "clfar",
           "wam", "lum36r", "cl36", "f60", "f4", "f49", "extra1", "extra2", "extra3", "extra4"]:
    slot(nm)
assert len(SLOTS) <= M


def rows_for(specs):
    """specs: list of (slot, rowvector) -> (M, n_in) matrix"""
    return specs


def build_dense(name, inp, n_in, specs):
    W = np.zeros((M, n_in), np.float32)
    b = np.zeros(M, np.float32)
    for s, row in specs:
        W[SLOTS[s]] = row.reshape(-1)
    return net.dense(name, inp, W, b)


X8 = (XX - CX) / 8.0
Y8 = (YY - CY) / 8.0
R32 = RR / 32.0

m36 = disk(36); m10 = disk(10); mout = ring(10, 36)
A36 = m36.sum(); A10 = m10.sum(); AO = mout.sum()
mx2 = (X8 ** 2 * m36).sum() / A36
my2 = (Y8 ** 2 * m36).sum() / A36
ix2 = (X8 ** 2 * m10).sum() / A10
iy2 = (Y8 ** 2 * m10).sum() / A10
ox2 = (X8 ** 2 * mout).sum() / AO
oy2 = (Y8 ** 2 * mout).sum() / AO

# --- from normalised smooth map nC
sp = []
sp.append(("bgC", mean_row(ring(52, 90))))
for nm, mk in [("f6", disk(6)), ("f14", ring(6, 14)), ("f28", ring(14, 28)), ("f44", ring(28, 44)), ("f60", ring(44, 60))]:
    sp.append((nm, mean_row(mk)))
sp.append(("f36", mean_row(m36)))
sp.append(("f10", mean_row(m10)))
sp.append(("f4", mean_row(disk(4))))
sp.append(("f49", mean_row(ring(4, 9))))
sp.append(("f1036", mean_row(mout)))
sp.append(("mxx", wrow(m36, X8 ** 2)))
sp.append(("myy", wrow(m36, Y8 ** 2)))
sp.append(("mxy", wrow(m36, X8 * Y8)))
sp.append(("sx", wrow(m36, X8)))
sp.append(("sy", wrow(m36, Y8)))
sp.append(("ixx", wrow(m10, X8 ** 2)))
sp.append(("iyy", wrow(m10, Y8 ** 2)))
sp.append(("ixy", wrow(m10, X8 * Y8)))
sp.append(("isx", wrow(m10, X8)))
sp.append(("isy", wrow(m10, Y8)))
sp.append(("oxx", wrow(mout, X8 ** 2)))
sp.append(("oyy", wrow(mout, Y8 ** 2)))
sp.append(("oxy", wrow(mout, X8 * Y8)))
sp.append(("rnum", wrow(disk(44), R32)))
build_dense("mC", "nC", 128 * 128, sp)

# --- raw maps: colour and texture (sky cancels in warm and in the high pass)
sp = [("warm14", mean_row(disk(14))), ("warm30", mean_row(ring(14, 30))), ("wam", mean_row(disk(30)))]
build_dense("mW", "pre", 2 * 128 * 128, [(s, np.concatenate([np.zeros_like(r), r], axis=1)) for s, r in sp])

sp = [("lum14r", mean_row(disk(14))), ("lum30r", mean_row(ring(14, 30))),
      ("lum36r", mean_row(disk(36))), ("bgr", mean_row(ring(52, 90)))]
build_dense("mR", "gC", 128 * 128, sp)

sp = [("cl14", mean_row(disk(14))), ("cl30", mean_row(ring(14, 30))), ("cl36", mean_row(disk(36))),
      ("cl48", mean_row(ring(30, 48))), ("clfar", mean_row(ring(48, 64))),
      ("noise", mean_row(ring(48, 90))), ("noise2", mean_row(ring(40, 90)))]
build_dense("mH", "habs", 128 * 128, sp)

net.add("cart", "mC", "mW", "mR", "mH")

# ---------------------------------------------------------------- stage 3: log-polar resampling
# A fixed bilinear resampling onto (rho = ln r, theta) turns a change of galaxy
# size into a translation along rho and a rotation into a translation along theta.
NR, NT, NW = 26, 48, 8
R0, R1 = 3.5, 50.0
DLR = np.log(R1 / R0) / (NR - 1)
RJ = R0 * np.exp(DLR * np.arange(NR))
TK = 2 * np.pi * np.arange(NT) / NT
NC = NT + NW                      # wrap columns so convs can cross theta = 0

# sub-cell sampling points (average over the cell, not a point sample)
SUB_R = np.array([-0.32, 0.0, 0.32])
SUB_T = np.array([-0.32, 0.0, 0.32])


def bilin_rows(r, t, shape=(128, 128)):
    """rows of a dense matrix that sample map at polar (r,t), bilinear, 3x3 subcells"""
    H, W = shape
    out = []
    for dr in SUB_R:
        for dt in SUB_T:
            rr = r * np.exp(dr * DLR)
            tt = t + dt * (2 * np.pi / NT)
            x = CX + rr * np.cos(tt)
            y = CY + rr * np.sin(tt)
            x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
            fx = x - x0; fy = y - y0
            for iy, wy in [(0, 1 - fy), (1, fy)]:
                for ix, wx in [(0, 1 - fx), (1, fx)]:
                    yy = np.clip(y0 + iy, 0, H - 1); xx = np.clip(x0 + ix, 0, W - 1)
                    wgt = wy * wx / 9.0
                    out.append((yy * W + xx, wgt))
    return out


Wlp = np.zeros((NR * NC, 128 * 128), np.float32)
for j in range(NR):
    for k in range(NC):
        t = TK[k % NT]
        idx, wgt = zip(*bilin_rows(RJ[j], t))
        row = np.zeros(128 * 128, np.float32)
        np.add.at(row, np.array(idx), np.array(wgt, np.float32))
        Wlp[j * NC + k] = row
net.dense("lp", "gA", Wlp, np.zeros(NR * NC, np.float32))
net.reshape("lpR", "lp", (1, NR, NC))
net.norm("lpN", "lpR")

# ---------------------------------------------------------------- stage 4: polar measurements
LP = 352
LS = {}


def lslot(nm):
    LS[nm] = len(LS)
    return LS[nm]


for j in range(NR):
    lslot(f"sb{j}")
lslot("lbg")
for m in (1, 2, 3, 4):
    for j in range(NR):
        lslot(f"A{m}_{j}")
for m in (1, 2, 3):
    for j in range(NR):
        lslot(f"c{m}_{j}"); lslot(f"s{m}_{j}")
for j in range(NR):
    lslot(f"asy{j}")
for j in range(NR):
    lslot(f"sbraw{j}")
assert len(LS) <= LP, len(LS)


def lp_dense(name, specs, n_in=None):
    n_in = int(np.prod(net.shapes["lpN"])) if n_in is None else n_in
    W = np.zeros((LP, n_in), np.float32); b = np.zeros(LP, np.float32)
    for s, row in specs:
        W[LS[s]] = np.asarray(row, np.float32).reshape(-1)
    return net.dense(name, "lpN", W, b)


def theta_mean(j):
    r = np.zeros((1, NR * NC), np.float32)
    r[0, j * NC:j * NC + NT] = 1.0 / NT
    return r


# radial surface-brightness profile, sky subtracted using the outermost rings
outer = np.mean([theta_mean(j) for j in range(NR - 4, NR)], axis=0)
sp = [("lbg", outer)]
for j in range(NR):
    sp.append((f"sb{j}", theta_mean(j) - outer))
    sp.append((f"sbraw{j}", theta_mean(j)))
lp_dense("lpSB", sp)

# angular harmonics: 8 phase projections per (radius, m); max over phase = amplitude
sp = []
for j in range(NR):
    for m in (1, 2, 3, 4):
        for l in range(8):
            psi = 2 * np.pi * l / 8
            row = np.zeros((1, NR * NC), np.float32)
            row[0, j * NC:j * NC + NT] = (2.0 / NT) * np.cos(m * TK - psi)
            sp.append((f"A{m}_{j}", row)) if l == 0 else None
# the amplitude needs its own tensor path (max over phase), so build it separately
Wf = np.zeros((NR * 4 * 8, NR * NC), np.float32)
for j in range(NR):
    for mi, m in enumerate((1, 2, 3, 4)):
        for l in range(8):
            psi = 2 * np.pi * l / 8
            Wf[(j * 4 + mi) * 8 + l, j * NC:j * NC + NT] = (2.0 / NT) * np.cos(m * TK - psi)
net.dense("lpF", "lpN", Wf, np.zeros(NR * 4 * 8, np.float32))
net.reshape("lpFR", "lpF", (1, NR * 4, 8))
net.pool("lpFM", "lpFR", [1, 8], [1, 8], "max")
net.reshape("lpFA", "lpFM", (NR * 4,))
Wa = np.zeros((LP, NR * 4), np.float32)
for j in range(NR):
    for mi, m in enumerate((1, 2, 3, 4)):
        Wa[LS[f"A{m}_{j}"], j * 4 + mi] = 1.0
net.dense("lpA", "lpFA", Wa, np.zeros(LP, np.float32))

# complex coefficients for m = 1, 2 (phase drift -> spiral pitch angle)
sp = []
for m in (1, 2, 3):
    for j in range(NR):
        rc = np.zeros((1, NR * NC), np.float32); rs = np.zeros((1, NR * NC), np.float32)
        rc[0, j * NC:j * NC + NT] = (2.0 / NT) * np.cos(m * TK)
        rs[0, j * NC:j * NC + NT] = (2.0 / NT) * np.sin(m * TK)
        sp.append((f"c{m}_{j}", rc)); sp.append((f"s{m}_{j}", rs))
lp_dense("lpCS", sp)

# 180-degree asymmetry: |P(theta) - P(theta + pi)|
sel = np.zeros((NR * NT, NR * NC), np.float32)
shf = np.zeros((NR * NT, NR * NC), np.float32)
for j in range(NR):
    for k in range(NT):
        sel[j * NT + k, j * NC + k] = 1.0
        shf[j * NT + k, j * NC + (k + NT // 2) % NT] = -1.0
net.dense("p48", "lpN", sel, np.zeros(NR * NT, np.float32))
net.dense("psh", "lpN", shf, np.zeros(NR * NT, np.float32))
net.reshape("p48R", "p48", (1, NR, NT))
net.reshape("pshR", "psh", (1, NR, NT))
net.add("pdif", "p48R", "pshR")
net.relu("pdpos", "pdif")
net.conv("pdneg", "pdif", np.array([[-1.0]], np.float32).reshape(1, 1, 1, 1), np.zeros(1, np.float32))
net.relu("pdnegr", "pdneg")
net.add("pdabs", "pdpos", "pdnegr")
net.reshape("pdabsF", "pdabs", (NR * NT,))
Wasy = np.zeros((LP, NR * NT), np.float32)
for j in range(NR):
    Wasy[LS[f"asy{j}"], j * NT:(j + 1) * NT] = 1.0 / NT
net.dense("lpAsy", "pdabsF", Wasy, np.zeros(LP, np.float32))

net.add("polar", "lpSB", "lpA", "lpCS", "lpAsy")

# ---------------------------------------------------------------- arithmetic gadgets
# relu-only piecewise-linear approximations of 1/x and x*y, on geometric knot grids.
def recip_knots(t0=0.005, q=1.25, K=42):
    t = t0 * q ** np.arange(K + 1)
    s = np.empty(K)
    s[0] = -1.0 / (t[0] * t[1])
    for k in range(1, K):
        s[k] = (1.0 / t[k]) * (1.0 / t[k - 1] - 1.0 / t[k + 1])
    return t, s, 1.0 / t[0]


def square_knots(t0=0.01, q=1.25, K=36):
    t = t0 * q ** np.arange(K + 1)
    s = np.empty(K)
    s[0] = t[1]
    for k in range(1, K):
        s[k] = t[k + 1] - t[k - 1]
    return t, s, t[0]


RT, RS, RC = recip_knots()
QT, QS, QC = square_knots()
KR, KQ = len(RS), len(QS)


def recip_layer(tag, src, idxs):
    """1/x for the entries of src listed in idxs (x clamped positive first)"""
    n = int(np.prod(net.shapes[src]))
    D = len(idxs)
    W = np.zeros((D, n), np.float32); b = np.zeros(D, np.float32)
    for i, ix in enumerate(idxs):
        W[i, ix] = 1.0
    d = net.dense(tag + "_d", src, W, b)
    dr = net.relu(tag + "_dr", d)
    W2 = np.zeros((D * KR, D), np.float32); b2 = np.zeros(D * KR, np.float32)
    for i in range(D):
        for k in range(KR):
            W2[i * KR + k, i] = 1.0
            b2[i * KR + k] = -RT[k]
    h = net.dense(tag + "_h", dr, W2, b2)
    hr = net.relu(tag + "_hr", h)
    W3 = np.zeros((D, D * KR), np.float32); b3 = np.full(D, RC, np.float32)
    for i in range(D):
        for k in range(KR):
            W3[i, i * KR + k] = RS[k]
    o = net.dense(tag + "_o", hr, W3, b3)
    return net.relu(tag, o)


def prod_layer(tag, src, pairs):
    """pairs: (idx_a, idx_b[, scale_a, scale_b]); output = sa*sb*a*b"""
    n = int(np.prod(net.shapes[src]))
    P = len(pairs)
    W = np.zeros((2 * P, n), np.float32); b = np.zeros(2 * P, np.float32)
    for i, pr in enumerate(pairs):
        ia, ib = pr[0], pr[1]
        sa = pr[2] if len(pr) > 2 else 1.0
        sb = pr[3] if len(pr) > 3 else 1.0
        W[2 * i, ia] = sa; W[2 * i, ib] = sb
        W[2 * i + 1, ia] = sa; W[2 * i + 1, ib] = -sb
    uv = net.dense(tag + "_uv", src, W, b)
    NV = 2 * KQ + 2
    W2 = np.zeros((2 * P * NV, 2 * P), np.float32); b2 = np.zeros(2 * P * NV, np.float32)
    for v in range(2 * P):
        base = v * NV
        W2[base, v] = 1.0; W2[base + 1, v] = -1.0
        for k in range(KQ):
            W2[base + 2 + 2 * k, v] = 1.0; b2[base + 2 + 2 * k] = -QT[k]
            W2[base + 3 + 2 * k, v] = -1.0; b2[base + 3 + 2 * k] = -QT[k]
    h = net.dense(tag + "_h", uv, W2, b2)
    hr = net.relu(tag + "_hr", h)
    W3 = np.zeros((P, 2 * P * NV), np.float32); b3 = np.zeros(P, np.float32)
    for i in range(P):
        for v, sgn in [(2 * i, 0.25), (2 * i + 1, -0.25)]:
            base = v * NV
            W3[i, base] = sgn * QC; W3[i, base + 1] = sgn * QC
            for k in range(KQ):
                W3[i, base + 2 + 2 * k] = sgn * QS[k]
                W3[i, base + 3 + 2 * k] = sgn * QS[k]
    return net.dense(tag, hr, W3, b3)


def phase_max(tag, src, pairs, nph=16):
    """max over phase of a*cos+b*sin = sqrt(a^2+b^2), via maxpool"""
    n = int(np.prod(net.shapes[src]))
    P = len(pairs)
    W = np.zeros((P * nph, n), np.float32); b = np.zeros(P * nph, np.float32)
    for i, (ia, ib) in enumerate(pairs):
        for l in range(nph):
            ph = 2 * np.pi * l / nph
            W[i * nph + l, ia] = np.cos(ph); W[i * nph + l, ib] = np.sin(ph)
    d = net.dense(tag + "_d", src, W, b)
    r = net.reshape(tag + "_r", d, (1, P, nph))
    m = net.pool(tag + "_m", r, [1, nph], [1, nph], "max")
    return net.reshape(tag, m, (P,))

# ---------------------------------------------------------------- stage 5: diagnostics
def make_vec(tag, exprs, sources):
    """exprs: list of (name, {source_slot: coef}); one dense per source, summed with add"""
    NV = len(exprs)
    names = {e[0]: i for i, e in enumerate(exprs)}
    outs = []
    for src, sd in sources:
        n = int(np.prod(net.shapes[src]))
        W = np.zeros((NV, n), np.float32); b = np.zeros(NV, np.float32)
        for i, (nm, terms) in enumerate(exprs):
            for s, c in terms.items():
                if s in sd:
                    W[i, sd[s]] += c
        outs.append(net.dense(tag if len(sources) == 1 else tag + "_" + src, src, W, b))
    if len(outs) > 1:
        net.add(tag, *outs)
    return names


def join(tag, *srcs):
    """concatenate vectors (no concat op: place each into a shared width, then add)"""
    ns = [int(np.prod(net.shapes[s])) for s in srcs]
    tot = sum(ns); off = 0; outs = []; idx = {}
    for s, n in zip(srcs, ns):
        W = np.zeros((tot, n), np.float32)
        for i in range(n):
            W[off + i, i] = 1.0
        outs.append(net.dense(tag + "_" + s, s, W, np.zeros(tot, np.float32)))
        idx[s] = (off, n); off += n
    net.add(tag, *outs)
    return idx


A4 = float(disk(4).sum()); A49 = float(ring(4, 9).sum())
A6 = float(disk(6).sum()); A14 = float(disk(14).sum()); A28 = float(disk(28).sum())
A44 = float(disk(44).sum()); A36 = float(disk(36).sum()); A10 = float(disk(10).sum())
AO = float(ring(10, 36).sum()); A14r = float(ring(6, 14).sum())
A28r = float(ring(14, 28).sum()); A44r = float(ring(28, 44).sum())
A30 = float(disk(30).sum()); A30r = float(ring(14, 30).sum())

JIN = list(range(2, 10))     # r ~ 4.4 .. 9.6 px
JOUT = list(range(11, 21))   # r ~ 12 .. 33 px
JFAR = list(range(21, 26))   # r ~ 37 .. 50 px
JTW = [8, 10, 12, 14, 16, 18]

E1 = []
def e1(nm, **t): E1.append((nm, t))

e1("bgC", bgC=1.0)
for nm, sl in [("F6", "f6"), ("F14", "f14"), ("F28", "f28"), ("F44", "f44"),
               ("F36", "f36"), ("F10", "f10"), ("F1036", "f1036"), ("F4", "f4"), ("F49", "f49")]:
    e1(nm, **{sl: 1.0, "bgC": -1.0})
e1("D", mxx=1.0, myy=-1.0, bgC=-(mx2 - my2))
e1("U", mxy=2.0)
e1("T", mxx=1.0, myy=1.0, bgC=-(mx2 + my2))
e1("Di", ixx=1.0, iyy=-1.0, bgC=-(ix2 - iy2))
e1("Ui", ixy=2.0)
e1("Ti", ixx=1.0, iyy=1.0, bgC=-(ix2 + iy2))
e1("Do", oxx=1.0, oyy=-1.0, bgC=-(ox2 - oy2))
e1("Uo", oxy=2.0)
e1("To", oxx=1.0, oyy=1.0, bgC=-(ox2 + oy2))
meanRR44 = float((RR * disk(44)).sum() / disk(44).sum())
e1("RN", rnum=32.0, bgC=-meanRR44)
e1("F44d", f6=A6 / A44, f14=A14r / A44, f28=A28r / A44, f44=A44r / A44, bgC=-1.0)
e1("W30", wam=1.0)
e1("W14", warm14=1.0)
e1("L30", lum14r=A14 / A30, lum30r=A30r / A30, bgr=-1.0)
e1("L14", lum14r=1.0, bgr=-1.0)
e1("L36", lum36r=1.0, bgr=-1.0)
for nm, sl in [("CL14", "cl14"), ("CL30", "cl30"), ("CL36", "cl36"), ("CL48", "cl48"),
               ("CLF", "clfar"), ("NZ", "noise"), ("NZ2", "noise2")]:
    e1(nm, **{sl: 1.0})

def avg(pref, js):
    return {f"{pref}{j}": 1.0 / len(js) for j in js}

e1("SBIN", **avg("sb", JIN)); e1("SBOUT", **avg("sb", JOUT)); e1("SBFAR", **avg("sb", JFAR))
for m in (1, 2, 3, 4):
    e1(f"A{m}IN", **avg(f"A{m}_", JIN)); e1(f"A{m}OUT", **avg(f"A{m}_", JOUT))
e1("ASYIN", **avg("asy", JIN)); e1("ASYOUT", **avg("asy", JOUT)); e1("ASYFAR", **avg("asy", JFAR))
FLW = {f"sb{j}": RJ[j] ** 2 / 1000.0 for j in range(NR)}
e1("FLTOT", **FLW)
e1("RHO1N", **{f"sb{j}": (j - 13.0) * RJ[j] ** 2 / 1000.0 for j in range(NR)})
e1("RHO2N", **{f"sb{j}": (j - 13.0) ** 2 * RJ[j] ** 2 / 1000.0 for j in range(NR)})
for j in range(NR):
    e1(f"sb{j}", **{f"sb{j}": 1.0})
for m in (2, 3):
    for j in range(8, 23):
        e1(f"c{m}_{j}", **{f"c{m}_{j}": 1.0}); e1(f"s{m}_{j}", **{f"s{m}_{j}": 1.0})
v1 = make_vec("v1", E1, [("cart", SLOTS), ("polar", LS)])

# ---- floor the radial profile at a small fraction of the map std, then re-measure it
NSB = 23
Wsf = np.zeros((NSB, int(np.prod(net.shapes["v1"]))), np.float32)
for j in range(NSB):
    Wsf[j, v1[f"sb{j}"]] = 1.0
sbf = net.dense("sbf", "v1", Wsf, np.full(NSB, -0.03, np.float32))
sbfr = net.relu("sbfr", sbf)
Wpr = np.zeros((3, NSB), np.float32)
for j in range(NSB):
    w = RJ[j] ** 2 / 1000.0
    Wpr[0, j] = w; Wpr[1, j] = (j - 13.0) * w; Wpr[2, j] = (j - 13.0) ** 2 * w
prof = net.dense("prof", sbfr, Wpr, np.zeros(3, np.float32))

v0i = join("v0", "v1", prof)
W0 = {k: v0i["v1"][0] + i for k, i in v1.items()}
W0.update({"FLTOT2": v0i[prof][0], "RHO1N2": v0i[prof][0] + 1, "RHO2N2": v0i[prof][0] + 2})

# ---- round 1: reciprocals
DEN1 = ["T", "Ti", "To", "F36", "F44", "F14", "F49", "F44d", "L30", "L36", "NZ",
        "SBIN", "SBOUT", "A2OUT", "ASYFAR", "FLTOT2", "F28", "F10", "A1OUT", "SBFAR"]
r1 = recip_layer("r1", "v0", [W0[d] for d in DEN1])
R1 = {d: i for i, d in enumerate(DEN1)}

w1i = join("w1", "v0", r1)
V1O, R1O = w1i["v0"][0], w1i[r1][0]
W1 = {k: V1O + i for k, i in W0.items()}
W1.update({("r_" + k): R1O + i for k, i in R1.items()})

# ---- round 2: products -> dimensionless ratios and harmonic cross/dot terms
P2 = []
def p2(nm, a, b, sa=1.0, sb=1.0): P2.append((nm, (W1[a], W1[b], sa, sb)))

p2("dn", "D", "r_T"); p2("un", "U", "r_T")
p2("dni", "Di", "r_Ti"); p2("uni", "Ui", "r_Ti")
p2("dno", "Do", "r_To"); p2("uno", "Uo", "r_To")
p2("cin", "F10", "r_F36", A10 / A36)
p2("cout", "F28", "r_F44", A28r / A44r)
p2("c614", "F6", "r_F14", A6 / A14r)
p2("dip", "F4", "r_F49", A4 / A49)
p2("size", "RN", "r_F44d", 1.0 / 12.0)
p2("color", "W30", "r_L30")
p2("clump", "CL36", "r_L36")
p2("clumpSN", "CL36", "r_NZ")
p2("snr", "L30", "r_NZ")
p2("a2in", "A2IN", "r_SBIN"); p2("a2out", "A2OUT", "r_SBOUT")
p2("a1out", "A1OUT", "r_SBOUT"); p2("a3out", "A3OUT", "r_SBOUT"); p2("a4out", "A4OUT", "r_SBOUT")
p2("a42", "A4OUT", "r_A2OUT")
p2("asyin", "ASYIN", "r_SBIN"); p2("asyout", "ASYOUT", "r_SBOUT")
p2("asysn", "ASYOUT", "r_ASYFAR")
p2("osig", "SBOUT", "r_ASYFAR")
p2("rho1", "RHO1N2", "r_FLTOT2"); p2("rho2", "RHO2N2", "r_FLTOT2")
p2("a2io", "A2IN", "r_A2OUT")
p2("c1028", "F10", "r_F28", A10 / A28r)
p2("sbio", "SBIN", "r_SBOUT")
p2("sbfar", "SBFAR", "r_SBOUT")
p2("a1in", "A1IN", "r_SBIN")
p2("clout", "CL48", "r_NZ")
p2("a32", "A3OUT", "r_A2OUT")
for m in (2, 3):
    for j in JTW:
        p2(f"x{m}_{j}", f"c{m}_{j}", f"s{m}_{j+2}")
        p2(f"y{m}_{j}", f"s{m}_{j}", f"c{m}_{j+2}")
        p2(f"u{m}_{j}", f"c{m}_{j}", f"c{m}_{j+2}")
        p2(f"v{m}_{j}", f"s{m}_{j}", f"s{m}_{j+2}")
q2 = prod_layer("q2", "w1", [p[1] for p in P2])
Q2 = {p[0]: i for i, p in enumerate(P2)}

# ---- round 3: ellipticity amplitudes, harmonic phase drift, profile width
ell3 = phase_max("ell", "q2", [(Q2["dn"], Q2["un"]), (Q2["dni"], Q2["uni"]),
                               (Q2["dno"], Q2["uno"])], nph=16)
w2i = join("w2", "q2", ell3)
Q2O, E3O = w2i["q2"][0], w2i[ell3][0]
W2 = {k: Q2O + i for k, i in Q2.items()}
W2.update({"e": E3O, "ei": E3O + 1, "eo": E3O + 2})

E3 = []
def e3(nm, **t): E3.append((nm, t))

for k in Q2:
    e3(k, **{k: 1.0})
e3("e", e=1.0); e3("ei", ei=1.0); e3("eo", eo=1.0)
for m in (2, 3):
    e3(f"S{m}", **{f"x{m}_{j}": 1.0 for j in JTW}, **{f"y{m}_{j}": -1.0 for j in JTW})
    e3(f"C{m}", **{f"u{m}_{j}": 1.0 for j in JTW}, **{f"v{m}_{j}": 1.0 for j in JTW})
e3("eio_d", ei=1.0, eo=-1.0)
v3 = make_vec("v3", E3, [("w2", W2)])


def abs_vec(tag, src, idxs):
    n = int(np.prod(net.shapes[src])); D = len(idxs)
    W = np.zeros((2 * D, n), np.float32)
    for i, ix in enumerate(idxs):
        W[2 * i, ix] = 1.0; W[2 * i + 1, ix] = -1.0
    d = net.dense(tag + "_d", src, W, np.zeros(2 * D, np.float32))
    r = net.relu(tag + "_r", d)
    W2m = np.zeros((D, 2 * D), np.float32)
    for i in range(D):
        W2m[i, 2 * i] = 1.0; W2m[i, 2 * i + 1] = 1.0
    return net.dense(tag, r, W2m, np.zeros(D, np.float32))


ab3 = abs_vec("ab3", "v3", [v3["S2"], v3["C2"], v3["S3"], v3["C3"]])
R23 = phase_max("r23", "v3", [(v3["S2"], v3["C2"]), (v3["S3"], v3["C3"])], nph=16)
w3i = join("w3", "v3", ab3, R23)
V3O, AB3O, R23O = w3i["v3"][0], w3i[ab3][0], w3i[R23][0]
W3 = {k: V3O + i for k, i in v3.items()}
W3.update({"aS2": AB3O, "aC2": AB3O + 1, "aS3": AB3O + 2, "aC3": AB3O + 3,
           "R2": R23O, "R3": R23O + 1})

# small floors so the reciprocal never sees ~0
Wf = np.zeros((3, int(np.prod(net.shapes["w3"]))), np.float32)
for i, k in enumerate(["eo", "R2", "R3"]):
    Wf[i, W3[k]] = 1.0
v3b = net.dense("v3b", "w3", Wf, np.array([0.06, 0.02, 0.02], np.float32))
r3 = recip_layer("r3", "v3b", [0, 1, 2])
w3j = join("w3j", "w3", r3)
W3b = dict(W3); W3b.update({"r_eo": w3j[r3][0], "r_R2": w3j[r3][0] + 1, "r_R3": w3j[r3][0] + 2})

P3 = []
def p3(nm, a, b, sa=1.0, sb=1.0): P3.append((nm, (W3b[a], W3b[b], sa, sb)))

p3("sn2", "aS2", "r_R2"); p3("cn2", "C2", "r_R2")
p3("sn3", "aS3", "r_R3"); p3("cn3", "C3", "r_R3")
p3("rho1sq", "rho1", "rho1")
p3("eio", "ei", "r_eo")
p3("ei2", "ei", "ei"); p3("eo2", "eo", "eo")
q3 = prod_layer("q3", "w3j", [p[1] for p in P3])
Q3 = {p[0]: i for i, p in enumerate(P3)}

# ---------------------------------------------------------------- stage 6: bounded indicators
def ind_layer(tag, src, conds):
    """trapezoid indicators in [0,1]: 4 hinges per condition"""
    n = int(np.prod(net.shapes[src])); nc = len(conds)
    Wi = np.zeros((4 * nc, n), np.float32); bi = np.zeros(4 * nc, np.float32)
    Wb = np.zeros((nc, 4 * nc), np.float32); bb = np.zeros(nc, np.float32)
    for i, (nm, spec) in enumerate(conds):
        kind, ix, t, w = spec
        if kind == "s":
            ts = [t, t + w, 1e6, 1e6]; ws = [1.0, -1.0, 0.0, 0.0]
        elif kind == "b":
            ts = [t - w, t, 1e6, 1e6]; ws = [-1.0, 1.0, 0.0, 0.0]; bb[i] = 1.0
        else:
            lo, hi = t, t + w
            r = min(0.35 * (hi - lo), 0.08) if hi > lo else 0.05
            ts = [lo, lo + r, hi, hi + r]; ws = [1.0, -1.0, -1.0, 1.0]
            w = r
        for j in range(4):
            Wi[4 * i + j, ix] = 1.0 / w; bi[4 * i + j] = -ts[j] / w
            Wb[i, 4 * i + j] = ws[j]
    h = net.dense(tag + "_h", src, Wi, bi)
    hr = net.relu(tag + "_r", h)
    return net.dense(tag, hr, Wb, bb), {nm: i for i, (nm, _) in enumerate(conds)}


w4i = join("w4", "q2", "q3", ell3, "r23")
W4 = {k: w4i["q2"][0] + i for k, i in Q2.items()}
W4.update({k: w4i["q3"][0] + i for k, i in Q3.items()})
W4.update({"e": w4i[ell3][0], "ei": w4i[ell3][0] + 1, "eo": w4i[ell3][0] + 2,
           "r2": w4i["r23"][0], "r3": w4i["r23"][0] + 1})
DGN = ["cin", "cout", "c614", "dip", "size", "color", "clump", "clumpSN", "snr", "a2in",
       "a2out", "a1out", "a3out", "a4out", "a42", "asyin", "asyout", "asysn", "osig",
       "rho1", "rho2", "a2io", "c1028", "sbio", "sbfar", "a1in", "clout", "a32",
       "sn2", "cn2", "sn3", "cn3", "eio", "e", "ei", "eo", "r2", "r3"]
E4 = [(k, {k: 1.0}) for k in DGN]
E4.append(("rhow", {"rho2": 1.0, "rho1sq": -1.0}))
dg = make_vec("dg", E4, [("w4", W4)])

# significance gates: outer-disk light, and coherent m=2 / m=3 phase signal
GATE = [("g_osig", ("s", dg["osig"], 1.5, 2.5)),
        ("g_r2", ("s", dg["r2"], 0.12, 0.30)),
        ("g_r3", ("s", dg["r3"], 0.10, 0.25)),
        ("g_ei", ("s", dg["ei"], 0.30, 0.15)),
        ("g_eo", ("s", dg["eo"], 0.62, 0.15)),
        ("g_a2in", ("s", dg["a2in"], 0.45, 0.20)),
        ("g_a2out", ("s", dg["a2out"], 0.70, 0.30))]
gates, GI = ind_layer("gate", "dg", GATE)
w5i = join("w5", "dg", gates)
DG = {k: w5i["dg"][0] + i for k, i in dg.items()}
DG.update({k: w5i[gates][0] + i for k, i in GI.items()})

# outer-region harmonics are only meaningful where there is outer disk light
GATED = ["a2out", "a4out", "a3out", "a42", "a32", "a1out", "asyout"]
qg = prod_layer("qg", "w5", [(DG[k], DG["g_osig"]) for k in GATED])
w6i = join("w6", "w5", qg)
D6 = {k: w6i["w5"][0] + i for k, i in DG.items()}
D6.update({("G_" + k): w6i[qg][0] + i for i, k in enumerate(GATED)})

COND = []
def step(nm, d, t, w): COND.append((nm, ("s", D6[d], t, w)))
def below(nm, d, t, w): COND.append((nm, ("b", D6[d], t - w, w)))
def bin(nm, d, lo, hi): COND.append((nm, ("n", D6[d], lo, hi - lo)))

# --- global shape
below("round", "e", 0.17, 0.11); bin("mild", "e", 0.16, 0.32); bin("modE", "e", 0.32, 0.50)
bin("fat", "e", 0.50, 0.68); bin("thin", "e", 0.68, 0.84); step("vthin", "e", 0.84, 0.07)
step("edge", "e", 0.88, 0.06); step("edge2", "e", 0.93, 0.04)
below("notedge", "e", 0.80, 0.10)
below("ei_round", "ei", 0.26, 0.14); bin("ei_mod", "ei", 0.25, 0.50); bin("ei_thin", "ei", 0.50, 0.75)
step("ei_edge", "ei", 0.78, 0.10)
bin("eo_mod", "eo", 0.30, 0.55); step("eo_thin", "eo", 0.70, 0.12); step("eo_edge", "eo", 0.86, 0.08)
below("bar_lo", "eio", 0.80, 0.15); bin("bar_no", "eio", 0.86, 1.20)
step("bar_hi", "eio", 1.22, 0.25); step("bar_hi2", "eio", 1.55, 0.4)
below("bulge_in", "eio", 0.88, 0.18)
# --- concentration / profile
step("cin_hi", "cin", 0.38, 0.12); step("cin_vhi", "cin", 0.52, 0.12); below("cin_lo", "cin", 0.24, 0.10)
step("c614_hi", "c614", 0.65, 0.25); below("c614_lo", "c614", 0.40, 0.15)
step("cout_hi", "cout", 3.5, 2.0); below("cout_lo", "cout", 2.0, 1.0)
below("dip_lo", "dip", 0.30, 0.12); below("dip_vlo", "dip", 0.20, 0.08); step("dip_hi", "dip", 0.6, 0.3)
below("size_s", "size", 0.68, 0.22); bin("size_m", "size", 0.65, 1.1); step("size_l", "size", 1.15, 0.45)
step("size_xl", "size", 1.8, 0.6)
step("rhow_hi", "rhow", 38, 12); below("rhow_lo", "rhow", 26, 10)
below("rho1_s", "rho1", -3.0, 2.5); step("rho1_l", "rho1", 3.0, 3.0)
# --- colour
step("red", "color", 0.14, 0.10); step("vred", "color", 0.26, 0.10)
below("blue", "color", 0.04, 0.10); below("vblue", "color", -0.08, 0.10)
# --- texture / asymmetry
below("smooth", "clumpSN", 1.18, 0.14); step("clumpy", "clumpSN", 1.35, 0.25)
step("vclumpy", "clumpSN", 1.75, 0.4); step("clumpy2", "clumpSN", 1.22, 0.15)
below("asy_lo", "G_asyout", 0.13, 0.08); step("asy_hi", "G_asyout", 0.24, 0.16)
step("asy_vhi", "G_asyout", 0.42, 0.25)
below("asyin_lo", "asyin", 0.13, 0.08); step("asyin_hi", "asyin", 0.22, 0.16)
below("a1_lo", "G_a1out", 0.11, 0.07); step("a1_hi", "G_a1out", 0.19, 0.14)
step("a1_vhi", "G_a1out", 0.34, 0.2)
step("a1in_hi", "a1in", 0.18, 0.14)
step("clout_hi", "clout", 1.22, 0.25); step("sbfar_hi", "sbfar", 0.06, 0.08)
# --- harmonics
step("a2in_hi", "a2in", 0.50, 0.22); step("a2in_vhi", "a2in", 0.80, 0.3)
below("a2out_lo", "G_a2out", 0.35, 0.18); step("a2out_hi", "G_a2out", 0.85, 0.3)
step("a2out_vhi", "G_a2out", 1.25, 0.3)
step("a42_hi", "G_a42", 0.45, 0.15); step("a42_vhi", "G_a42", 0.60, 0.15)
below("a42_lo", "G_a42", 0.24, 0.12)
step("a3_hi", "G_a3out", 0.14, 0.12); step("a4_hi", "G_a4out", 0.14, 0.12)
step("a32_hi", "G_a32", 0.35, 0.25)
step("a2io_hi", "a2io", 1.25, 0.35); below("a2io_lo", "a2io", 0.85, 0.25)
step("sbio_hi", "sbio", 6.0, 5.0); step("sbio_vhi", "sbio", 14.0, 10.0)
# --- winding (spiral pitch from the m=2 / m=3 phase drift per log radius)
below("nowind", "sn2", 0.20, 0.14); step("wind", "sn2", 0.32, 0.22); step("wind_hi", "sn2", 0.62, 0.25)
below("cn2_neg", "cn2", 0.15, 0.30); below("cn2_vneg", "cn2", -0.25, 0.35)
step("cn2_pos", "cn2", 0.55, 0.30); step("cn2_vpos", "cn2", 0.85, 0.12)
step("wind3", "sn3", 0.35, 0.25); below("cn3_neg", "cn3", 0.2, 0.35)
# --- reliability
step("snr_hi", "snr", 12, 12); below("snr_lo", "snr", 5, 3)
step("osig_hi", "osig", 3, 3); below("osig_lo", "osig", 1.2, 1.0)

ind, IND = ind_layer("ind", "w6", COND)
wsi = join("ws", "ind", gates)
IX = {k: wsi["ind"][0] + i for k, i in IND.items()}
IX.update({k: wsi[gates][0] + i for k, i in GI.items()})

# ---------------------------------------------------------------- stage 7: hand-written rule table
ANDS = [("merg_core", "asyin_hi", "dip_lo"), ("merg_core2", "a1in_hi", "dip_lo"),
        ("dist_outer", "asy_hi", "asyin_lo"), ("dist_lops", "a1_hi", "asyin_lo"),
        ("bar_real", "bar_hi", "g_ei"), ("bar_real2", "bar_hi2", "g_ei"),
        ("bar_arm", "bar_hi", "clumpy2"), ("bar_str", "bar_hi", "a2in_hi"),
        ("edge_nobulge", "edge", "bar_no"), ("edge_bulge", "edge", "bulge_in"),
        ("bulge_edge", "bulge_in", "g_eo"), ("barlow_edge", "bar_lo", "g_eo"),
        ("eiround_edge", "ei_round", "g_eo"),
        ("tight_arm", "cn2_neg", "g_r2"), ("tight_arm2", "cn2_vneg", "g_r2"),
        ("loose_arm", "cn2_pos", "g_r2"), ("veryloose", "cn2_vpos", "g_r2"),
        ("nowind_g", "nowind", "g_r2"), ("wind_g", "wind", "g_r2"),
        ("windhi_g", "wind_hi", "g_r2"), ("wind3_g", "wind3", "g_r3"),
        ("cn3neg_g", "cn3_neg", "g_r3"),
        ("smooth_round", "round", "smooth"), ("smooth_red", "smooth", "red"),
        ("spiral_disk", "clumpy", "notedge"), ("edge_thin", "edge", "a42_hi"),
        ("merg_broad", "cin_lo", "asy_hi"),
        ("a2io_hi_g", "a2io_hi", "g_a2in"),
        ("a2io_lo_g", "a2io_lo", "g_a2out"), ("clumpy_blue", "clumpy", "blue"),
        ("asy_clumpy", "asy_hi", "clumpy"), ("ei_edge_nb", "ei_edge", "bar_no")]
qa = prod_layer("qand", "ws", [(IX[a], IX[b]) for _, a, b in ANDS])
ANDI = {nm: i for i, (nm, a, b) in enumerate(ANDS)}
wii = join("wi", "ws", qa)
IX2 = {k: wii["ws"][0] + i for k, i in IX.items()}
IX2.update({k: wii[qa][0] + i for k, i in ANDI.items()})

ANDS2 = [("tight_wind", "tight_arm", "wind_hi"), ("loose_wind", "loose_arm", "wind"),
         ("tight_nobar", "tight_arm", "bar_no"), ("bar_arm2", "bar_real", "clumpy2"),
         ("edge_bulge2", "edge_bulge", "cin_hi"), ("merg_dip", "merg_core", "asy_hi"),
         ("dist_clumpy", "dist_outer", "clumpy"), ("smooth_ei", "smooth", "bar_no"),
         ("smooth_noarm", "smooth", "nowind_g"),
         ("bar_notedge", "bar_real", "notedge"),
         ("loose_clump", "clumpy", "loose_arm")]
qa2 = prod_layer("qand2", "wi", [(IX2[a], IX2[b]) for _, a, b in ANDS2])
ANDI2 = {nm: i for i, (nm, a, b) in enumerate(ANDS2)}
wia = join("wia", "wi", qa2)
FI = {k: wia["wi"][0] + i for k, i in IX2.items()}
FI.update({k: wia[qa2][0] + i for k, i in ANDI2.items()})

# score contributions, typed from general knowledge of what each class looks like
RULES = {
 "round": {2: 3.0, 3: -1.0, 4: -2.5, 8: -3.0, 9: -2.5, 5: -0.5},
 "mild": {2: 1.0, 3: 2.0, 4: -0.5, 8: -2.0, 9: -1.5},
 "modE": {3: 1.5, 4: 1.0, 2: -1.0, 5: 0.5, 8: -1.0, 9: -0.5},
 "fat": {4: 2.0, 3: 0.5, 5: 0.8, 2: -2.0, 8: -0.5},
 "thin": {4: 1.5, 8: 1.0, 9: 1.0, 5: 0.5, 2: -3.0, 3: -1.0},
 "vthin": {8: 2.5, 9: 2.5, 4: 0.5, 2: -3.0, 3: -2.0},
 "edge": {8: 2.5, 9: 2.5, 4: -1.0, 2: -3.0, 3: -2.5},
 "edge2": {8: 1.5, 9: 1.5, 4: -1.5},
 "notedge": {8: -1.5, 9: -1.5},
 "ei_mod": {9: 1.0, 5: 0.3},
 "bar_real": {5: 2.5, 8: -1.0, 9: -1.0, 2: -0.8, 3: -0.5},
 "bar_real2": {5: 1.5, 8: -1.0, 9: -1.5},
 "bulge_edge": {9: 2.0, 8: -0.8, 5: -1.0},
 "barlow_edge": {9: 1.5, 8: -0.5, 5: -1.0},
 "eiround_edge": {9: 1.5, 8: -1.0, 4: -0.5},
 "tight_arm": {6: 2.5, 7: -0.3, 8: -0.8, 9: -0.8, 2: -0.5, 3: -0.5},
 "tight_arm2": {6: 1.5, 7: -0.5},
 "loose_arm": {7: 1.0, 6: -0.5, 8: +0.3, 9: +0.3},
 "veryloose": {8: 0.5, 9: 0.5, 2: 0.3, 3: 0.3, 4: 0.3, 6: -0.8, 7: -0.3},
 "nowind_g": {2: 0.8, 3: 0.8, 4: 0.8, 8: 1.0, 9: 1.0, 5: 0.5, 6: -0.8, 7: -0.8},
 "wind_g": {6: 1.0, 7: 0.8, 5: 0.3},
 "windhi_g": {6: 1.5, 7: 0.3},
 "wind3_g": {7: 0.8, 6: 0.3},
 "cn3neg_g": {6: 0.8, 7: 0.3},
 "a2io_hi_g": {5: 1.5, 9: -0.5, 8: -0.5},
 "a2io_lo_g": {9: 1.2, 8: 0.5, 5: -0.8},
 "tight_wind": {6: 1.5},
 "loose_wind": {7: 1.0},
 "tight_nobar": {6: 1.0},
 "bar_arm2": {5: 1.5},
 "edge_bulge2": {9: 1.0},
 "merg_dip": {1: 1.5},
 "dist_clumpy": {0: 0.8},
 "smooth_ei": {2: 0.5, 3: 0.5, 4: 0.5},
 "ei_edge_nb": {8: 1.0},
 "ei_thin": {8: 0.8, 4: 0.5, 9: -0.3},
 "ei_edge": {8: 2.0, 9: -0.5, 4: 0.5},
 "eo_thin": {8: 1.0, 9: 1.0, 4: 0.8},
 "eo_edge": {8: 1.5, 9: 1.5},
 "bar_no": {8: 0.8, 2: 0.5, 3: 0.5, 6: 0.5, 7: 0.5, 5: -0.8},
 "cin_hi": {2: 1.2, 3: 0.8, 9: 1.0, 6: 0.5, 7: -0.8, 1: -0.8},
 "cin_vhi": {2: 1.2, 3: 0.5, 9: 0.8, 7: -1.0, 1: -1.0},
 "cin_lo": {7: 1.0, 1: 1.0, 0: 0.5, 2: -1.2, 8: 0.3},
 "c614_hi": {2: 1.0, 3: 0.5, 9: 0.5},
 "c614_lo": {7: 0.8, 1: 0.5, 0: 0.3},
 "cout_hi": {2: 0.8, 3: 0.5, 4: 0.5},
 "cout_lo": {7: 0.5, 0: 0.5, 1: 0.5, 8: 0.3},
 "dip_lo": {1: 2.0, 0: 0.5, 2: -1.0},
 "dip_vlo": {1: 1.5, 2: -1.0},
 "dip_hi": {2: 0.8, 3: 0.5, 9: 0.5, 1: -1.0},
 "size_s": {2: 0.5, 3: 0.3, 6: 0.3, 0: -0.3, 1: -0.5, 8: -0.3},
 "size_xl": {0: 0.5, 1: 0.5, 7: 0.3},
 "rhow_hi": {2: 1.0, 3: 0.8, 4: 0.8, 9: 0.3, 7: -0.5},
 "rhow_lo": {7: 0.5, 6: 0.3, 5: 0.3, 2: -0.5},
 "rho1_s": {2: 0.3, 6: 0.3},
 "red": {2: 1.5, 3: 1.2, 4: 1.2, 9: 0.8, 7: -1.0, 6: -0.3},
 "vred": {2: 1.2, 3: 0.8, 4: 0.8, 7: -1.2, 0: -0.5},
 "blue": {7: 1.2, 6: 0.8, 5: 0.5, 0: 0.5, 8: 0.5, 2: -1.5, 3: -1.0, 4: -0.8},
 "vblue": {7: 1.0, 0: 0.8, 1: 0.5, 2: -1.5, 3: -1.0, 4: -1.0},
 "smooth": {2: 2.0, 3: 1.8, 4: 1.8, 6: 0.3, 7: -1.5, 0: -1.5, 1: -1.5, 5: -0.8},
 "clumpy2": {7: 0.8, 6: 0.5, 5: 0.5, 0: 0.5},
 "clumpy": {7: 1.5, 5: 1.0, 6: 0.8, 0: 1.0, 1: 0.8, 2: -1.5, 3: -1.2, 4: -1.0},
 "vclumpy": {7: 1.2, 0: 1.2, 1: 1.2, 2: -1.5, 3: -1.2, 4: -1.0},
 "asy_lo": {2: 1.2, 3: 1.0, 4: 1.0, 6: 0.5, 8: 0.5, 9: 0.5, 0: -1.5, 1: -1.2},
 "asy_hi": {0: 2.0, 1: 1.5, 7: 0.3, 2: -1.2, 3: -0.8, 4: -0.8},
 "asy_vhi": {0: 1.5, 1: 1.5, 2: -1.5, 3: -1.0},
 "asyin_lo": {0: -0.8, 2: 0.5, 3: 0.5, 6: 0.3},
 "asyin_hi": {1: 1.8, 0: 0.8, 2: -0.8, 3: -0.5},
 "a1_lo": {2: 1.0, 3: 0.8, 4: 0.8, 8: 0.5, 9: 0.5, 0: -1.2, 1: -0.8},
 "a1_hi": {0: 1.8, 1: 1.2, 7: 0.3, 2: -1.0, 3: -0.6},
 "a1_vhi": {0: 1.5, 1: 1.5},
 "a1in_hi": {1: 1.2, 0: 0.5},
 "clout_hi": {0: 1.2, 1: 0.8, 7: 0.3},
 "sbfar_hi": {0: 0.8, 1: 0.8, 2: -0.5},
 "a2in_hi": {5: 1.2, 4: 0.5, 8: 0.3, 9: 0.3, 1: 0.3},
 "a2in_vhi": {5: 1.0, 8: 0.8, 9: 0.5, 1: 0.5},
 "a2out_lo": {2: 1.0, 3: 0.3, 6: 0.3, 7: 0.3, 8: -1.0, 9: -0.8, 4: -0.5},
 "a2out_hi": {4: 1.0, 8: 1.2, 9: 1.2, 5: 0.5, 2: -1.0},
 "a2out_vhi": {8: 1.5, 9: 1.5, 4: 0.3, 2: -1.5, 3: -0.8},
 "a42_hi": {8: 1.5, 9: 1.2, 4: 0.3, 2: -0.8, 3: -0.5},
 "a42_vhi": {8: 1.5, 9: 1.0, 4: -0.5},
 "a42_lo": {2: 0.8, 3: 0.5, 6: 0.3, 7: 0.3, 8: -1.0, 9: -0.8},
 "a3_hi": {7: 1.0, 6: 0.3, 0: 0.3, 2: -0.5},
 "a4_hi": {7: 0.8, 5: 0.3, 0: 0.3},
 "a32_hi": {7: 0.8, 0: 0.3},
 "sbio_hi": {2: 0.5, 9: 0.5, 6: 0.3},
 "sbio_vhi": {2: 0.5, 9: 0.3, 7: -0.3},
 "snr_lo": {2: 0.8, 3: 0.5, 6: 0.3, 0: -0.5, 1: -0.5, 7: -0.3},
 "osig_lo": {8: -0.8, 9: -0.5, 7: -0.5, 0: -0.5, 2: +0.5, 3: 0.3},
 "merg_core": {1: 2.5, 0: 0.5},
 "merg_core2": {1: 2.0},
 "dist_outer": {0: 2.0, 1: -0.8},
 "dist_lops": {0: 1.8, 1: -0.5},
 "bar_arm": {5: 1.5},
 "bar_str": {5: 1.5},
 "edge_nobulge": {8: 1.5, 9: -1.0},
 "edge_bulge": {9: 2.0, 8: -1.0},
 "tight_arm": {6: 2.0},
 "tight_arm2": {6: 1.5},
 "loose_arm": {7: 1.2, 6: -0.5},
 "loose_clump": {7: 1.2},
 "smooth_round": {2: 1.5},
 "smooth_red": {2: 0.8, 3: 0.8, 4: 0.8},
 "spiral_disk": {5: 0.5, 6: 0.8, 7: 0.8},
 "edge_thin": {8: 1.0, 9: 1.0},
 "merg_broad": {1: 1.2, 0: 0.5},
 "smooth_noarm": {2: 0.8, 3: 0.8, 4: 0.8, 8: 0.5, 9: 0.5},
 "bar_notedge": {5: 1.0},
 "tight_nobar": {6: 1.0},
 "clumpy_blue": {7: 0.8, 6: 0.5, 0: 0.3},
 "asy_clumpy": {0: 0.8, 1: 0.8},
}
PRIOR = np.array([0.0, 0.0, 0.4, 0.2, -1.0, 0.0, 0.0, 0.2, -0.3, 0.0], np.float32)
NF = len(FI)
Wr = np.zeros((10, NF), np.float32)
for nm, vec in RULES.items():
    assert nm in FI, nm
    for k, v in vec.items():
        Wr[k, FI[nm]] += v
net.dense("scores", "wia", Wr, PRIOR)

net.save()
print("layers:", len(net.layers), "weights:", sum(int(np.prod(v.shape)) for v in net.w.values()))
