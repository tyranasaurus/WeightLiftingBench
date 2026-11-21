"""Hand-designed galaxy morphology network.

Every number here comes from analytic formulas (Gaussian / DoG kernels, polar
annuli, angular Fourier modes, logarithmic-spiral matched filters, second
moments) and from hand-chosen thresholds.  No training data is read.

Structure
  image -> luminance / colour -> 2x2 average -> light smoothing + DoG
        -> dense "polar Fourier" banks (rings x angular modes) at two scales
        -> hand-written decision circuit (dense + relu, zero bias everywhere,
           so the whole network is positively homogeneous of degree 1 in the
           image intensity => predictions do not depend on exposure depth).
"""
import sys, json, os
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

layers = []
W = {}


def layer(name, op, inputs, **kw):
    L = {"name": name, "op": op, "inputs": inputs}
    L.update(kw)
    layers.append(L)
    return name


def conv(name, src, out, k, stride=1, padding=0, w=None, b=None):
    layer(name, "conv", [src], out=out, kernel=[k, k], stride=stride, padding=padding)
    W[name + ".W"] = np.asarray(w, np.float64)
    W[name + ".b"] = np.zeros(out) if b is None else np.asarray(b, np.float64)


def dense(name, src, out, w=None):
    layer(name, "dense", [src], out=out)
    W[name + ".W"] = np.asarray(w, np.float64)
    W[name + ".b"] = np.zeros(out)


# ---------------------------------------------------------------- geometry
N = 64                      # working resolution (128 / 2)
CEN = (N - 1) / 2.0         # 31.5
yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
DX = xx - CEN
DY = yy - CEN
RHO = np.sqrt(DX ** 2 + DY ** 2)
TH = np.arctan2(DY, DX)
LR = np.log(np.maximum(RHO, 0.35))

# radial windows: Gaussian in ln(rho), centres 1.8 .. 30 (=> 3.6 .. 60 original px)
J = 22
RC = np.exp(np.linspace(np.log(1.2), np.log(30.0), J))
SL = 0.115                  # width in ln(rho)
G = np.exp(-(LR[None, :, :] - np.log(RC)[:, None, None]) ** 2 / (2 * SL ** 2))
GZ = G.reshape(J, -1).sum(1)

AP = np.clip((34.0 - RHO) / 4.0, 0, 1)          # photometric aperture
BG = np.clip((RHO - 32.0) / 4.0, 0, 1)          # sky reference (frame corners)
BGZ = BG.sum()


def bgsub(mask):
    """remove the sky pedestal, keeping the functional linear in the image"""
    return mask - (mask.sum() / BGZ) * BG


def flat(m):
    return np.asarray(m, np.float64).reshape(-1)


# ---------------------------------------------------------------- front end
def gauss(k, s):
    a = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-(a[:, None] ** 2 + a[None, :] ** 2) / (2 * s * s))
    return g / g.sum()


def place(name, src, ch, nch, val=1.0):
    w = np.zeros((nch, 1, 1, 1)); w[ch, 0, 0, 0] = val
    conv(name, src, nch, 1, w=w)


conv("lum", "image", 1, 1, w=np.ones((1, 3, 1, 1)) / 3.0)
conv("red", "image", 1, 1, w=np.array([1.0, 0.0, -1.0]).reshape(1, 3, 1, 1))
layer("pL", "avgpool", ["lum"], size=[2, 2], stride=[2, 2])
layer("pC", "avgpool", ["red"], size=[2, 2], stride=[2, 2])
conv("smA", "pL", 1, 5, padding=2, w=gauss(5, 0.85)[None, None])   # sigma .85
conv("smB", "smA", 1, 5, padding=2, w=gauss(5, 1.40)[None, None])  # sigma 1.64
# stack -> [0, Ls, C, -Lb]
place("eL", "smA", 1, 4)
place("eB", "smB", 3, 4, -1.0)
place("eC", "pC", 2, 4)
layer("stk", "add", ["eL", "eB", "eC"])
# dogc -> [DoG, -DoG, Ls, C, -C]
wd = np.zeros((5, 4, 1, 1))
wd[0, 1] = 1; wd[0, 3] = 1
wd[1, 1] = -1; wd[1, 3] = -1
wd[2, 1] = 1
wd[3, 2] = 1
wd[4, 2] = -1
conv("dogc", "stk", 5, 1, w=wd)
layer("rl5", "relu", ["dogc"])
# mix -> [|DoG|, Ls, C, DoG]
wm = np.zeros((4, 5, 1, 1))
wm[0, 0] = 1; wm[0, 1] = 1
wm[1, 2] = 1
wm[2, 3] = 1; wm[2, 4] = -1
wm[3, 0] = 1; wm[3, 1] = -1
conv("mix", "rl5", 4, 1, w=wm)
CH = {"tex": 0, "L": 1, "C": 2, "H": 3}     # channel of each map inside "mix"

# ---------------------------------------------------------------- polar banks
MS = (1, 2, 3, 4)
idx = {}
rows = []


def addrow(name, mask, chan="L"):
    v = np.zeros(4 * N * N)
    v[CH[chan] * N * N:(CH[chan] + 1) * N * N] = flat(mask)
    idx[name] = len(rows)
    rows.append(v)


for j in range(J):
    addrow(f"P{j}", bgsub(G[j]) / GZ[j])
for m in MS:
    for j in range(J):
        addrow(f"C{m}_{j}", G[j] * np.cos(m * TH) / GZ[j])
        addrow(f"S{m}_{j}", G[j] * np.sin(m * TH) / GZ[j])


def ap(mask):
    return bgsub(AP * mask)


addrow("F", ap(np.ones_like(RHO)))
addrow("TT", ap(RHO ** 2))
addrow("QC", ap(RHO ** 2 * np.cos(2 * TH)))
addrow("QS", ap(RHO ** 2 * np.sin(2 * TH)))
addrow("CX", ap(RHO * np.cos(TH)))
addrow("CY", ap(RHO * np.sin(TH)))
IN = (RHO < 8.0).astype(float)
OU = ((RHO >= 10.0) & (RHO < 34.0)).astype(float)
for tag, msk in (("i", IN), ("o", OU)):
    addrow("F" + tag, ap(msk))
    addrow("T" + tag, ap(msk * RHO ** 2))
    addrow("QC" + tag, ap(msk * RHO ** 2 * np.cos(2 * TH)))
    addrow("QS" + tag, ap(msk * RHO ** 2 * np.sin(2 * TH)))
addrow("BGm", BG / BGZ)
ZONES = [(0, 3), (3, 8), (8, 16), (16, 30)]
for z, (a, b) in enumerate(ZONES):
    addrow(f"ZL{z}", bgsub(((RHO >= a) & (RHO < b)).astype(float)))
NL = len(rows)

# same rings on the DoG (band pass) map: arms / knots / dust lanes
for j in range(J):
    addrow(f"hP{j}", G[j] / GZ[j], "H")
for m in MS:
    for j in range(J):
        addrow(f"hC{m}_{j}", G[j] * np.cos(m * TH) / GZ[j], "H")
        addrow(f"hS{m}_{j}", G[j] * np.sin(m * TH) / GZ[j], "H")
NH = len(rows) - NL

# texture profile  mean|DoG| per ring
for j in range(J):
    addrow(f"TX{j}", G[j] / GZ[j], "tex")

# colour: summed (z-g) in the same radial zones, sky subtracted
for z, (a, b) in enumerate(ZONES):
    addrow(f"CZ{z}", bgsub(((RHO >= a) & (RHO < b)).astype(float)), "C")
addrow("CBG", BG / BGZ, "C")

NF = len(rows)
dense("feat", "mix", NF, w=np.stack(rows))

# ---------------------------------------------------------------- head
# Declarative circuit builder.  Every unit is a linear form over the previous
# stage's rectified outputs and every bias is zero, so the whole network is
# positively homogeneous of degree 1 in the image intensity: predictions do not
# depend on exposure depth.  Every structural threshold is referred to the same
# rho-weighted flux R1, which makes the thresholds dimensionless (scale free)
# and keeps all feature families in commensurate units.
class Circ:
    def __init__(self, names):
        self.base = {n: {i: 1.0} for i, n in enumerate(names)}
        self.nin = len(names)
        self.rows, self.new = [], {}

    def row(self, expr):
        r = {}
        for nm, c in expr.items():
            if c == 0:
                continue
            for i, v in self.base[nm].items():
                r[i] = r.get(i, 0.0) + c * v
        return r

    def add(self, name, expr):
        i = len(self.rows)
        self.rows.append(self.row(expr))
        self.new[name] = {i: 1.0}
        return i

    def carry(self, name, expr):
        ip = self.add(name + "+", expr)
        im = self.add(name + "-", {k: -v for k, v in expr.items()})
        self.new[name] = {ip: 1.0, im: -1.0}

    def end(self, tag, relu=True):
        n = len(self.rows)
        M = np.zeros((n, self.nin))
        for i, r in enumerate(self.rows):
            for j, v in r.items():
                M[i, j] = v
        dense("f" + tag, self.src, n, w=M)
        if relu:
            layer("r" + tag, "relu", ["f" + tag])
            self.src = "r" + tag
        else:
            self.src = "f" + tag
        self.nin = n
        self.base, self.rows, self.new = self.new, [], {}
        return n


names = [None] * NF
for k, v in idx.items():
    names[v] = k
cir = Circ(names)
cir.src = "feat"

# rho-weighted ring moments: good SNR, still sensitive to the outer disk
W1 = RC * GZ
W0 = GZ
ZG = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9], 2: [10, 11, 12, 13, 14],
      3: [15, 16, 17, 18, 19], 4: [20, 21]}
R1 = {f"P{j}": float(W1[j]) for j in range(J)}
F0 = {f"P{j}": float(W0[j]) for j in range(J)}


def zsum(pfx, z, w):
    return {f"{pfx}{j}": float(w[j]) for j in ZG[z]}


def zc(m, z, h=False, w=W1):
    p = "h" if h else ""
    return ({f"{p}C{m}_{j}": float(w[j]) for j in ZG[z]},
            {f"{p}S{m}_{j}": float(w[j]) for j in ZG[z]})


CSUM = float(np.exp(-(np.log(8.0) - np.log(RC)) ** 2 / (2 * SL ** 2)).sum())
K = 8
PHI = np.arange(K) * 2 * np.pi / K
KN = 2.5464790894703255
groups = {}


def _proj(ca, sa, k):
    e = {}
    for nm, c in ca.items():
        e[nm] = e.get(nm, 0.0) + c * np.cos(PHI[k])
    for nm, c in sa.items():
        e[nm] = e.get(nm, 0.0) + c * np.sin(PHI[k])
    return e


def amp(name, ca, sa):
    for k in range(K):
        cir.add(f"{name}#{k}", _proj(ca, sa, k))
    groups[name] = ("amp", K)


def exc(name, ca, sa, ref):
    """sum_k relu(projection_k - ref): exactly zero while |complex| <= ref"""
    for k in range(K):
        e = _proj(ca, sa, k)
        for nm, c in ref.items():
            e[nm] = e.get(nm, 0.0) - c
        cir.add(f"{name}#{k}", e)
    groups[name] = ("exc", K)


def hinge(name, pos, neg):
    e = dict(pos)
    for nm, c in neg.items():
        e[nm] = e.get(nm, 0.0) - c
    cir.add(name, e)


def sc(d, s):
    return {k: v * s for k, v in d.items()}


def merge(*ds):
    e = {}
    for d in ds:
        for k, v in d.items():
            e[k] = e.get(k, 0.0) + v
    return e


# ================= stage 1: projections and elementary hinges ==============
Qc = {f"C2_{j}": float(W1[j]) for j in range(J)}
Qs = {f"S2_{j}": float(W1[j]) for j in range(J)}
amp("Qall", Qc, Qs)
amp("Cen", {"CX": 1.0}, {"CY": 1.0})
ET = (0.08, 0.16, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.93)
for t in ET:
    exc(f"eA{t}", Qc, Qs, sc(R1, t))
ZT = (0.02, 0.05, 0.10, 0.18, 0.30)
HT = (0.004, 0.010, 0.022, 0.045)
AT = (0.008, 0.02, 0.045, 0.09)
WT = (0.003, 0.008, 0.018)
for z in range(5):
    ca, sa = zc(2, z)
    amp(f"Qz{z}", ca, sa)
    for t in ZT:
        exc(f"eZ{z}t{t}", ca, sa, sc(R1, t))
    ha, hs = zc(2, z, True)
    for t in HT:
        exc(f"h2z{z}t{t}", ha, hs, sc(R1, t))
    for h in (False, True):
        c1, s1 = zc(1, z, h)
        for t in AT:
            exc(f"{'h' if h else 'l'}1z{z}t{t}", c1, s1, sc(R1, t))
    for m in (3, 4):
        c3, s3 = zc(m, z, True)
        for t in HT:
            exc(f"h{m}z{z}t{t}", c3, s3, sc(R1, t))
        c4, s4 = zc(m, z, False)
        for t in (0.02, 0.06, 0.14):
            exc(f"l{m}z{z}t{t}", c4, s4, sc(R1, t))

for a, b in ((0, 2), (1, 3), (2, 4)):
    ca, sa = zc(2, a); cb, sb = zc(2, b)
    amp(f"A{a}{b}p", merge(ca, cb), merge(sa, sb))
    amp(f"A{a}{b}m", merge(sc(ca, -1), cb), merge(sc(sa, -1), sb))


def hann(lo, hi):
    x = (np.log(RC) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return np.where((x > 0) & (x < 1), 0.5 - 0.5 * np.cos(2 * np.pi * x), 0.0)


WIND = []
for m, (lo, hi), ks, tag in ((2, (2.5, 24.0), (3.5, 7.0, 11.0, 16.0), "w"),
                             (2, (3.5, 16.0), (5.0, 10.0), "m"),
                             (3, (2.5, 24.0), (7.0, 14.0), "t")):
    w = hann(lo, hi) * W1
    sw = w.sum()
    pref = {f"P{j}": float(w[j]) for j in range(J)}
    for k in ks:
        for sgn in (1, -1):
            kk = sgn * k
            cs, sn = np.cos(kk * np.log(RC)), np.sin(kk * np.log(RC))
            Wc, Ws = float((w * cs).sum()), float((w * sn).sum())
            ec, es = {}, {}
            for j in range(J):
                if w[j] == 0:
                    continue
                ec[f"hC{m}_{j}"] = w[j] * cs[j] - Wc * w[j] / sw
                ec[f"hS{m}_{j}"] = w[j] * sn[j] - Ws * w[j] / sw
                es[f"hS{m}_{j}"] = w[j] * cs[j] - Wc * w[j] / sw
                es[f"hC{m}_{j}"] = -w[j] * sn[j] + Ws * w[j] / sw
            nm = f"W{tag}{m}_{kk:+.0f}"
            for t in WT:
                exc(f"{nm}t{t}", ec, es, sc(pref, t))
            WIND.append((nm, m, tag, kk))

# radial flux profile (concentration / bulge / ring), same R1 units
for z in range(5):
    rz = zsum("P", z, W1)
    for t in (0.03, 0.08, 0.16, 0.30, 0.50):
        hinge(f"rz{z}t{t}", rz, sc(R1, t))
for z in range(5):
    fz = zsum("P", z, W0)
    for t in (0.05, 0.15, 0.35):
        hinge(f"fz{z}t{t}", fz, sc(F0, t))
        hinge(f"fz{z}lo{t}", sc(F0, t), fz)
RISE = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
for a, b in RISE:
    hinge(f"rise{a}{b}", zsum("P", b, W0), sc(zsum("P", a, W0), 1.25))
PZ = {z: {f"P{j}": 1.0 / len(ZG[z]) for j in ZG[z]} for z in range(5)}
RISEP = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
for a, b in RISEP:
    hinge(f"rp{a}{b}", PZ[b], sc(PZ[a], 1.25))
for t in (0.4, 0.9, 1.6, 2.6, 4.0):
    exc(f"cen{t}", {"CX": 1.0}, {"CY": 1.0}, sc(F0, t / CSUM))
CLT = (-0.35, -0.15, 0.05, 0.25, 0.45)
for z in range(4):
    for t in CLT:
        hinge(f"cl{z}_{t}", {f"CZ{z}": 1.0}, {f"ZL{z}": t})
# size: <rho> = R1 / F0
SZT = [2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 17.0, 22.0]
for t in SZT:
    hinge(f"sz{t}", R1, sc(F0, t))
for j in (4, 8, 12, 16):
    for t in (0.10, 0.20, 0.40):
        hinge(f"tx{j}t{t}", {f"TX{j}": float(W0[j])}, {f"P{j}": t * float(W0[j])})
# references carried forward
cir.carry("R1", R1)
cir.carry("F0", F0)
for z in range(5):
    cir.carry(f"Rz{z}", zsum("P", z, W1))
n1 = cir.end("1")

# ================= stage 2: amplitude / excess sums ========================
for nm, (kind, kk) in groups.items():
    cir.add(nm, {f"{nm}#{k}": (1.0 / KN if kind == "amp" else 1.0) for k in range(kk)})
for a, b in ((0, 2), (1, 3), (2, 4)):
    e = {f"A{a}{b}p#{k}": 1.0 / KN for k in range(K)}
    e.update({f"A{a}{b}m#{k}": -1.0 / KN for k in range(K)})
    cir.carry(f"Al{a}{b}", e)
for i in range(1, len(SZT) - 1):
    t0, t1, t2 = SZT[i - 1], SZT[i], SZT[i + 1]
    cir.add(f"bump{t1}", {f"sz{t0}": 1.0, f"sz{t1}": -2.0, f"sz{t2}": 1.0})
cir.carry("R1", {"R1": 1.0})
cir.carry("F0", {"F0": 1.0})
for z in range(5):
    cir.carry(f"Rz{z}", {f"Rz{z}": 1.0})
PASS1 = ([f"cl{z}_{t}" for z in range(4) for t in CLT]
         + [f"sz{t}" for t in SZT]
         + [f"rise{a}{b}" for a, b in RISE]
         + [f"rp{a}{b}" for a, b in RISEP]
         + [f"fz{z}{s}{t}" for z in range(5) for s in ("t", "lo") for t in (0.05, 0.15, 0.35)]
         + [f"tx{j}t{t}" for j in (4, 8, 12, 16) for t in (0.10, 0.20, 0.40)]
         + [f"rz{z}t{t}" for z in range(5) for t in (0.03, 0.08, 0.16, 0.30, 0.50)])
for nm in PASS1:
    cir.add(nm, {nm: 1.0})
n2 = cir.end("2")
# from here on the references are the carried signals, not the raw rings
R1 = {"R1": 1.0}
F0 = {"F0": 1.0}

# ================= stage 3: morphological concepts =========================
def fam(pfx, zs, ts, h=True):
    return merge(*[{f"{pfx}z{z}t{t}": 1.0 for z in zs} for t in ts])


WHI = merge(*[{f"W{tag}{m}_{sgn*k:+.0f}t0.008": 1.0
               for tag, m, ks in (("w", 2, (11, 16)), ("m", 2, (10,)), ("t", 3, (14,)))
               for k in ks} for sgn in (1, -1)])
WLO = merge(*[{f"W{tag}{m}_{sgn*k:+.0f}t0.008": 1.0
               for tag, m, ks in (("w", 2, (3.5, 7)), ("m", 2, (5,)), ("t", 3, (7,)))
               for k in ks} for sgn in (1, -1)])
ARMS_RAW = merge(fam("h3", (1, 2, 3), (0.010, 0.022)),
                 fam("h4", (2, 3), (0.022,)), fam("h2", (2, 3, 4), (0.022, 0.045)),
                 sc(merge(WHI, WLO), 0.6))
SM_RAW = merge(fam("h3", (1, 2, 3), (0.004, 0.010)),
               fam("h4", (1, 2, 3), (0.010,)), fam("h2", (2, 3, 4), (0.022,)),
               fam("h1", (1, 2, 3), (0.008, 0.02)), fam("l1", (1, 2, 3), (0.008, 0.02)),
               sc(merge(WHI, WLO), 0.6))
INNER = merge({"eZ0t0.02": 1.0, "eZ1t0.02": 1.0, "eZ1t0.05": 0.8, "eZ0t0.05": 0.6})
OUTER = merge({"eZ3t0.18": 1.0, "eZ4t0.18": 1.0, "eZ3t0.3": 1.0, "eZ2t0.3": 0.6})
ASYM_RAW = merge(fam("l1", (1, 2, 3), (0.02, 0.045)), fam("h1", (1, 2, 3), (0.02,)))
IRR_RAW = merge(fam("h3", (1, 2, 3), (0.010, 0.022)), fam("h4", (1, 2, 3), (0.010, 0.022)))
DBL_RAW = sc({f"rp{a}{b}": 1.0 for a, b in RISEP}, 800.0)
TW_RAW = merge({"Al02": -1.0, "Al13": -1.0, "Al24": -0.6})

CON = {}
def concept(name, expr, norm):
    cir.add(name, expr)
    CON[name] = norm

concept("ROUND", merge(sc(R1, 0.12), sc({"eA0.08": 1.0, "eA0.16": 1.0}, -1.0)), 0.12)
concept("MILD", merge({"eA0.08": 1.0, "eA0.16": 0.6, "eA0.25": -2.5}, sc(R1, -0.004)), 0.55)
concept("CIGAR", merge({"eA0.35": 1.0, "eA0.45": 0.6, "eA0.65": -2.2}, sc(R1, -0.004)), 0.55)
concept("THIN", merge({"eA0.55": 0.6, "eA0.65": 1.0, "eA0.75": 1.5, "eA0.85": 2.0}, sc(R1, -0.010)), 1.8)
concept("SMOOTH", merge(sc(R1, 0.030), sc(SM_RAW, -1.0)), 0.030)
concept("ARMS", merge(ARMS_RAW, sc(R1, -0.012)), 0.12)
concept("WINDT", merge(WHI, sc(WLO, -0.6), sc(R1, -0.002)), 0.10)
concept("WINDL", merge(WLO, sc(WHI, -0.6), sc(R1, -0.002)), 0.10)
concept("BAR", merge(INNER, sc(ARMS_RAW, 0.35), sc(R1, -0.030)), 0.25)
concept("BULGE", merge(OUTER, sc(INNER, -0.5), sc(R1, -0.006)), 0.06)
concept("ASYM", merge(ASYM_RAW, sc(R1, -0.010)), 0.12)
concept("IRR", merge(IRR_RAW, sc(merge(WHI, WLO), -0.4), sc(R1, -0.004)), 0.10)
concept("DOUBLE", merge(DBL_RAW, sc(R1, -0.030)), 0.15)
concept("TWIST", merge(TW_RAW, sc(R1, -0.004)), 0.05)
concept("RED", merge(sc({f"cl{z}_{t}": 1.0 for z in range(4) for t in (0.25, 0.45)}, 1.0),
                     sc(F0, -0.02)), 0.30)
concept("BLUE", merge(sc({f"cl{z}_{t}": -1.0 for z in range(4) for t in (-0.35, -0.15)}, 1.0),
                      sc(F0, 0.06)), 0.30)
concept("BIG", merge({"sz13.0": 1.0, "sz17.0": 1.0, "sz22.0": 0.5}, sc(F0, -0.02)), 0.4)
concept("SMALL", merge(sc(F0, 0.10), sc({"sz4.0": 1.0, "sz5.0": 1.0, "sz6.5": 0.6}, -1.0)), 0.10)
# pass the elementary elongation / profile excesses through to the score stage
for t in ET:
    cir.add(f"eA{t}", {f"eA{t}": 1.0})
for nm in ("rz0t0.03", "rz0t0.08", "rz1t0.08", "rz2t0.16", "rz3t0.16", "rz4t0.08",
           "fz0t0.35", "fz0lo0.05", "cen1.6", "cen2.6"):
    cir.add(nm, {nm: 1.0})
cir.carry("R1", {"R1": 1.0})
n3 = cir.end("3")

# ================= stage 4: normalised, saturating evidence ================
for nm, nrm in CON.items():
    v = {nm: 1.0 / nrm}
    cir.add(nm + "n", v)                                   # ~O(1)
    cir.add(nm + "c", merge(sc({nm: 1.0}, 1.0 / nrm), sc(R1, -1.6)))
# capped evidence = 1.6 - relu(1.6 - x)  (in units of the concept's own scale)
n4 = cir.end("4")

# ================= stage 5: class scores ===================================
SC = {
    0: {"ASYM": 1.0, "IRR": 0.8, "TWIST": 0.5, "ARMS": 0.25, "DOUBLE": -0.7,
        "SMOOTH": -0.5, "THIN": -0.4, "WINDT": -0.3},
    1: {"DOUBLE": 1.1, "ASYM": 0.7, "IRR": 0.5, "TWIST": 0.3, "SMOOTH": -0.5,
        "THIN": -0.3, "WINDT": -0.3, "BAR": -0.3},
    2: {"SMOOTH": 1.3, "ROUND": 1.1, "ARMS": -0.9, "ASYM": -0.8, "IRR": -0.6,
        "DOUBLE": -0.5, "THIN": -0.5, "MILD": -0.3, "RED": 0.25},
    3: {"SMOOTH": 1.2, "MILD": 1.1, "ARMS": -0.9, "ASYM": -0.8, "IRR": -0.6,
        "DOUBLE": -0.4, "THIN": -0.6, "ROUND": -0.2, "RED": 0.2},
    4: {"SMOOTH": 1.1, "CIGAR": 1.2, "ARMS": -0.8, "ASYM": -0.7, "IRR": -0.5,
        "THIN": -0.9, "BULGE": -0.5, "DOUBLE": -0.4, "RED": 0.15},
    5: {"BAR": 1.2, "ARMS": 0.7, "WINDL": 0.3, "ASYM": -0.5, "THIN": -0.7,
        "DOUBLE": -0.5, "SMOOTH": -0.2, "BULGE": -0.3},
    6: {"ARMS": 0.9, "WINDT": 1.2, "SMOOTH": 0.3, "BAR": -0.7, "ASYM": -0.5,
        "THIN": -0.6, "DOUBLE": -0.4, "IRR": -0.2},
    7: {"ARMS": 0.9, "WINDL": 1.2, "IRR": 0.25, "BAR": -0.7, "ASYM": -0.4,
        "THIN": -0.6, "DOUBLE": -0.4, "WINDT": -0.3, "BLUE": 0.2},
    8: {"THIN": 1.5, "BULGE": -1.1, "ARMS": -0.3, "ASYM": -0.3, "SMOOTH": -0.2,
        "DOUBLE": -0.3, "BLUE": 0.15},
    9: {"THIN": 1.5, "BULGE": 1.1, "ARMS": -0.3, "ASYM": -0.3, "SMOOTH": -0.2,
        "DOUBLE": -0.3, "RED": 0.1},
}
AUX = {"RED", "BLUE", "BIG", "SMALL"}
SC[8]["BIG"] = 0.25; SC[9]["BIG"] = 0.2
SC[2]["SMALL"] = 0.2; SC[3]["SMALL"] = 0.1
rows = []
for k in range(10):
    e = {}
    for nm, w0 in SC[k].items():
        w = w0 * (6.0 if nm in AUX else 1.0)
        # capped, normalised evidence
        e[nm + "n"] = e.get(nm + "n", 0.0) + w
        e[nm + "c"] = e.get(nm + "c", 0.0) - w
    rows.append(cir.row(e))
M = np.zeros((10, cir.nin))
for i, r in enumerate(rows):
    for j, v in r.items():
        M[i, j] = v
dense("score", cir.src, 10, w=M)

json.dump({"layers": layers}, open(os.path.join(OUT, "network.json"), "w"))
np.savez(os.path.join(OUT, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
print("features:", NF, "stages:", n1, n2, n3, n4, "layers:", len(layers),
      "weights:", sum(v.size for v in W.values()))
