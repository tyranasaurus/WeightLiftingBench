"""Hand-designed person / no-person network.

Every number here comes from formulas (colour-space definitions, Gaussian and
centre-surround kernels, trapezoid membership functions) or from general
knowledge about how people look in photographs.  No training data is read.
"""
import json, sys, os
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
LAYERS = []
W = {}


SH = {"image": (3, 96, 96)}


def layer(name, op, inputs, **kw):
    L = {"name": name, "op": op, "inputs": inputs if isinstance(inputs, list) else [inputs]}
    L.update(kw)
    LAYERS.append(L)
    s = SH[L["inputs"][0]]
    if op == "conv":
        k, st, p = L["kernel"][0], L.get("stride", 1), L.get("padding", 0)
        SH[name] = (L["out"], (s[1] + 2 * p - k) // st + 1, (s[2] + 2 * p - k) // st + 1)
    elif op == "dense":
        SH[name] = (L["out"],)
    elif op in ("maxpool", "avgpool"):
        a, b = L["size"], L["stride"]
        SH[name] = (s[0], (s[1] - a[0]) // b[0] + 1, (s[2] - a[1]) // b[1] + 1)
    elif op == "reshape":
        SH[name] = tuple(L["shape"])
    else:
        SH[name] = s
    return name


def conv(name, src, out, k, stride=1, pad=0, w=None, b=None):
    layer(name, "conv", src, out=out, kernel=[k, k], stride=stride, padding=pad)
    W[name + ".W"] = np.asarray(w, np.float32).reshape(out, -1, k, k)
    W[name + ".b"] = np.asarray(b, np.float32).reshape(out)


def dense(name, src, out, w, b):
    layer(name, "dense", src, out=out)
    W[name + ".W"] = np.asarray(w, np.float32)
    W[name + ".b"] = np.asarray(b, np.float32)


def relu(name, src):
    return layer(name, "relu", src)


def pool(name, src, size, stride=None, kind="avg"):
    if stride is None:
        stride = size
    return layer(name, kind + "pool", src, size=[size, size], stride=[stride, stride])


def add(name, srcs):
    return layer(name, "add", srcs)


def reshape(name, src, shape):
    return layer(name, "reshape", src, shape=shape)


def zeros(o, c, k):
    return np.zeros((o, c, k, k), np.float64)


def affine(coef, scale, offset):
    """rows for x = (coef . rgb + offset) / scale  ->  (coef/scale, offset/scale)"""
    return np.asarray(coef, float) / scale, offset / scale


def trap_args(x_coef, a, b, c, d, off=0.0):
    """Rows of a 1x1 conv + relu that later combine into a unit-height trapezoid.

    T(x) = (relu(x-a) - relu(x-b) - relu(x-c) + relu(x-d)) / (b-a)
    x = x_coef . (R,G,B).  Returns 4 (coef, bias) pairs and the combining weights.
    """
    rows = [(x_coef, off - a), (x_coef, off - b), (x_coef, off - c), (x_coef, off - d)]
    comb = np.array([1.0, -1.0, -1.0, 1.0]) / (b - a)
    return rows, comb


CR = np.array([0.5, -0.418688, -0.081312])
CB = np.array([-0.168736, -0.331264, 0.5])
YV = np.array([0.299, 0.587, 0.114])
RG = np.array([1.0, -1.0, 0.0])
GB = np.array([0.0, 1.0, -1.0])
QV = np.array([1.0, -2.0, 1.0])          # R-2G+B : reddish (+) vs yellowish (-)
# D = (R-2G+B) - 0.1*(R-B): "pinkness" scaled by intensity.  Skin is pink
# (R-G about twice G-B); wood, fur, sand, bread, cardboard are yellow.
DV = np.array([0.9, -2.0, 1.1])
ONE = np.array([1.0, 1.0, 1.0]) / 3.0

# ---------------------------------------------------------------- stage 1
# Per-pixel colour membership maps at full 96x96 resolution.
# Skin locus taken from the standard YCbCr skin cluster plus the sign of
# R-2G+B (reddish skin vs. yellowish bread / sand / wood).
rowsA, biasA = [], []
comb = {}
for tag, coef, abcd, off in [("cr", CR, (0.510, 0.545, 0.645, 0.685), 0.5),
                             ("cb", -CB, (-0.525, -0.480, -0.355, -0.310), -0.5),
                             ("q", QV, (-0.030, 0.015, 0.280, 0.380), 0.0),
                             ("d", DV, (-0.010, 0.012, 0.200, 0.300), 0.0)]:
    r, c = trap_args(coef, *abcd, off=off)
    comb[tag] = (len(rowsA), c)
    for coef_, b_ in r:
        rowsA.append(coef_); biasA.append(b_)
rowsA += [YV, -YV]
biasA += [0.0, 0.12]                      # last two: Y and relu(0.12 - Y)
nA = len(rowsA)
conv("cA", "image", nA, 1, w=np.array(rowsA)[:, :, None, None], b=np.array(biasA))
relu("rA", "cA")


def trow(tag, scale=1.0):
    v = np.zeros(nA)
    i, c = comb[tag]
    v[i:i + 4] = scale * c
    return v


Tcr, Tcb, Tq = trow("cr"), trow("cb"), trow("q")
Td = trow("d")
Yrow = np.zeros(nA); Yrow[nA - 2] = 1.0
Drow = np.zeros(nA); Drow[nA - 1] = 1.0        # relu(0.12-Y)

# soft-AND of the three colour tests; constants go in the BIAS, never in a
# coefficient slot (index nA-2 is the luminance channel, not a constant).
v_skin = (0.34 * Tcr + 0.33 * Tcb + 0.33 * Td) / 0.40
v_skin[nA - 1] -= 0.5 / 0.12                  # fade out very dark pixels
b_skin = -0.60 / 0.40
v_warm = (0.55 * Tcr + 0.45 * Tq) / 0.55
b_warm = -0.45 / 0.55
v_dark = -Yrow / 0.26
b_dark = 0.30 / 0.26

B = np.stack([v_skin, v_skin, v_dark, v_dark, v_warm, v_warm, Yrow])
bb = np.array([b_skin, b_skin - 1.0, b_dark, b_dark - 1.0, b_warm, b_warm - 1.0, 0.0])
conv("cB", "rA", 7, 1, w=B[:, :, None, None], b=bb)
relu("rB", "cB")
C = np.zeros((4, 7))
C[0, 0], C[0, 1] = 1, -1                  # skin  in [0,1]
C[1, 2], C[1, 3] = 1, -1                  # dark  in [0,1]
C[2, 4], C[2, 5] = 1, -1                  # warm  in [0,1]
C[3, 6] = 1                               # luminance
conv("cC", "rB", 4, 1, w=C[:, :, None, None], b=np.zeros(4))

# pyramid of the fine maps
pool("sk48", "cC", 2)                     # 4 x 48 x 48
pool("sk24", "sk48", 2)                   # 4 x 24 x 24
pool("sk12", "sk24", 2)                   # 4 x 12 x 12
pool("rgb24", "image", 4)                 # 3 x 24 x 24 : scene colours
pool("rgb48", "image", 2)                 # 3 x 48 x 48

# ---------------------------------------------------------------- stage 2
# Membership maps.  Texture is the key nuisance variable: skin is smooth,
# food / fur / foliage / printed fabric are not.  texn in [0,1].
SEL_S = np.array([1.0, 0.0, 0.0, 0.0])
SEL_W = np.array([0.0, 0.0, 1.0, 0.0])
SEL_Y = np.array([0.0, 0.0, 0.0, 1.0])
for nm, sel in [("s48", SEL_S), ("w48", SEL_W), ("y48", SEL_Y)]:
    conv(nm, "sk48", 1, 1, w=sel[None, :, None, None], b=[0.0])
pool("s24", "s48", 2)

LAP = np.array([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]) / 4.0
conv("lap", "y48", 2, 3, pad=1, w=np.stack([LAP, -LAP])[:, None], b=[0.0, 0.0])
relu("rlap", "lap")
conv("tex48", "rlap", 1, 1, w=np.ones((1, 2, 1, 1)), b=[0.0])
pool("tex24", "tex48", 2)
# texn = clip01(tex / 0.08): 0.08 is the typical |Laplacian| of busy texture
conv("txp", "tex48", 2, 1, w=np.full((2, 1, 1, 1), 1.0 / 0.08), b=[0.0, -1.0])
relu("txpr", "txp")
conv("txn48", "txpr", 1, 1, w=np.array([1.0, -1.0])[None, :, None, None], b=[0.0])
pool("txn24", "txn48", 2)


def lin1(name, src, w, b=0.0):
    conv(name, src, 1, 1, w=np.full((1, 1, 1, 1), w), b=[b])
    return name


# cs = smooth skin (Lukasiewicz AND of skin and NOT-texture)
lin1("ntxn", "txn48", -1.0)
add("csp", ["s48", "ntxn"])
relu("cs48", "csp")
pool("cs24", "cs48", 2)
pool("cs12", "cs24", 2)
# fd = textured warm material: food, fur, bread crust, dry grass
lin1("ptxn", "txn48", 1.0, -1.0)
add("fdp", ["w48", "ptxn"])
relu("fd48", "fdp")
pool("fd24", "fd48", 2)
pool("fd12", "fd24", 2)
# face tissue: skin minus dark minus texture
conv("mbase", "sk48", 1, 1, w=np.array([1.0, -0.60, 0.0, 0.0])[None, :, None, None], b=[0.0])
lin1("mtex", "tex48", -3.0)
add("m48", ["mbase", "mtex"])
pool("m24", "m48", 2)
pool("m12", "m24", 2)
# luminance-only face tissue (black-and-white photos, unusual skin tones)
conv("gp48", "y48", 1, 1, w=np.full((1, 1, 1, 1), 2.0), b=[-0.90])
relu("gp48r", "gp48")
conv("gn48", "y48", 1, 1, w=np.full((1, 1, 1, 1), -2.0), b=[0.90])
relu("gn48r", "gn48")
conv("gma", "gp48r", 2, 1, w=np.array([1.0, 0.0])[None, :, None, None], b=[0.0, 0.0])
conv("gmb", "gn48r", 2, 1, w=np.array([0.0, 1.0])[None, :, None, None], b=[0.0, 0.0])
add("gm48", ["gma", "gmb"])
pool("gm24", "gm48", 2)
pool("gm12", "gm24", 2)

# vegetation and white porcelain / paper at 48
VG = np.array([-1.0, 2.0, -1.0]) * 6.0
BRc, BRb = affine(YV, 0.15, -0.72)
rows48 = [VG, VG, BRc, RG, -RG, GB, -GB, YV]
b48 = [0.0, -1.0, BRb, 0.0, 0.0, 0.0, 0.0, 0.0]
conv("k48a", "rgb48", 8, 1, w=np.array(rows48)[:, :, None, None], b=np.array(b48))
relu("k48ar", "k48a")
K48 = np.zeros((4, 8))
K48[0, 0], K48[0, 1] = 1, -1
K48[1] = K48[0]
K48[2, 2] = 1.0
K48[2, 3:7] = -1.0 / 0.10
K48[3] = K48[2]
conv("k48b", "k48ar", 4, 1, w=K48[:, :, None, None], b=np.array([0., 0., 0., -1.]))
relu("k48br", "k48b")
K48C = np.zeros((2, 4)); K48C[0, 0] = 1; K48C[1, 2], K48C[1, 3] = 1, -1
conv("kw48", "k48br", 2, 1, w=K48C[:, :, None, None], b=np.zeros(2))
conv("vg48", "kw48", 1, 1, w=np.array([1.0, 0.0])[None, :, None, None], b=[0.0])
conv("wh48", "kw48", 1, 1, w=np.array([0.0, 1.0])[None, :, None, None], b=[0.0])
for nm in ("vg", "wh"):
    pool(nm + "24", nm + "48", 2)
    pool(nm + "12", nm + "24", 2)

# oriented edge energy (long straight structure = rooms, vehicles, furniture)
SX = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
conv("sob", "y48", 4, 3, pad=1, w=np.stack([SX, -SX, SX.T, -SX.T])[:, None], b=[0.0] * 4)
relu("rsob", "sob")
conv("ex48", "rsob", 1, 1, w=np.array([1., 1., 0., 0.])[None, :, None, None], b=[0.0])
conv("ey48", "rsob", 1, 1, w=np.array([0., 0., 1., 1.])[None, :, None, None], b=[0.0])
pool("ex24", "ex48", 2)
pool("ey24", "ey48", 2)

# Two-channel face input: [smooth skin, non-skin tissue].  Keeping the skin
# channel non-negative matters -- on a signed map a bright blob with a dark
# surround would look like a face even when no skin is present at all.
conv("dk48", "sk48", 1, 1, w=np.array([0.0, 0.60, 0.0, 0.0])[None, :, None, None], b=[0.0])
lin1("tx3", "tex48", 3.0)
add("dn48", ["dk48", "tx3"])
for tag, src in (("fm", "cs48"), ("fa", "fd48")):
    conv(tag + "a", src, 2, 1, w=np.array([1.0, 0.0])[None, :, None, None], b=[0.0, 0.0])
    conv(tag + "b", "dn48", 2, 1, w=np.array([0.0, 1.0])[None, :, None, None], b=[0.0, 0.0])
    add(tag + "48", [tag + "a", tag + "b"])
    pool(tag + "24", tag + "48", 2)
    pool(tag + "12", tag + "24", 2)

# ---------------------------------------------------------------- stage 3
# Analytic head / face templates and centre-surround blob templates.
# Head geometry is the textbook one: an oval ~1.25x taller than wide, two dark
# eyes at (+-0.22W, -0.10W), a darker mouth at +0.30W, hair above, non-skin
# around.  Memberships are smooth so the template does not alias on coarse grids.
def grid(k):
    t = np.arange(k) - (k - 1) / 2.0
    return np.meshgrid(t, t, indexing="xy")


def _c01(v):
    return np.clip(v, 0.0, 1.0)


def face_parts(k, D):
    X, Y = grid(k)
    a, b = 0.5 * D, 0.62 * D
    rho = np.sqrt((X / a) ** 2 + (Y / b) ** 2)
    oval = _c01((1.0 - rho) / 0.28)
    rho2 = np.sqrt((X / (a + 0.38 * D)) ** 2 + (Y / (b + 0.38 * D)) ** 2)
    ring = _c01((1.0 - rho2) / 0.30) * (1.0 - oval)
    hair = (_c01((-Y - 0.45 * b) / (0.45 * b)) * _c01((0.80 * D - np.abs(X)) / (0.25 * D))
            * (1.0 - oval))
    eye = np.exp(-(((np.abs(X) - 0.22 * D) / (0.135 * D)) ** 2 +
                   ((Y + 0.10 * D) / (0.105 * D)) ** 2)) * oval
    mouth = np.exp(-((X / (0.19 * D)) ** 2 + ((Y - 0.30 * D) / (0.075 * D)) ** 2)) * oval
    return oval, ring, hair, eye, mouth


def face_kernel(k, D, eyes=True):
    """Two channels: [smooth skin, non-skin tissue].  Each is zero-sum, so a
    flat field of either kind gives no response."""
    oval, ring, hair, eye, mouth = face_parts(k, D)
    ew, mw = (1.0, 0.5) if eyes else (0.0, 0.0)
    K0 = oval - ew * eye - mw * mouth - 0.55 * ring
    K1 = (2.2 if eyes else 0.0) * eye + (0.8 if eyes else 0.0) * mouth + 0.8 * hair
    K0 = K0 - K0.mean()
    K1 = K1 - K1.mean()
    cs_id = np.maximum(0.85 * (oval - eye - 0.7 * mouth), 0.0)
    dn_id = 0.10 + 0.38 * eye + 0.25 * mouth + 0.35 * hair
    r = float((K0 * cs_id).sum() + (K1 * dn_id).sum())
    return np.stack([K0, K1]) / max(r, 1e-6)


def blob_kernel(k, D):
    """Isolated disk of diameter D: disk mean minus surrounding annulus mean."""
    X, Y = grid(k)
    r = np.sqrt(X ** 2 + Y ** 2)
    disk = _c01((0.5 * D + 0.5 - r) / 1.0)
    ann = _c01((r - 0.60 * D) / 0.8) * _c01((1.10 * D - r) / 0.8)
    K = disk / disk.sum() - ann / max(ann.sum(), 1e-6)
    return K / max(float((K * disk).sum()), 1e-6)


GROUPS = {}
LEVEL = {}


def tmpl(name, src, k, D, kind, group, stride=1):
    if kind == "blob":
        K = blob_kernel(k, D)[None, None]
    else:
        K = face_kernel(k, D, kind == "face")[:, None]
    conv(name, src, 1, k, stride=stride, pad=k // 2, w=K, b=[0.0])
    relu(name + "r", name)
    GROUPS.setdefault(group, []).append(name + "r")
    LEVEL[name + "r"] = SH[name + "r"][1]


# The SAME blob template on four material maps, so whatever the null response
# level of a centre-surround maximum is, it is shared between them.
for tag in ("cs", "fd", "vg", "wh"):
    tmpl("B%s_48" % tag, tag + "48", 9, 5.0, "blob", "B" + tag)
    tmpl("B%s_24" % tag, tag + "24", 9, 5.0, "blob", "B" + tag)
    tmpl("B%s_12" % tag, tag + "12", 9, 5.0, "blob", "B" + tag)
tmpl("Bcs_48s", "cs48", 5, 3.0, "blob", "Bcs")

tmpl("F_48", "fm48", 9, 5.0, "face", "FACE")
tmpl("F_24", "fm24", 9, 4.5, "face", "FACE")
tmpl("F_24b", "fm24", 11, 6.5, "face", "FACE")
tmpl("F_12", "fm12", 9, 4.5, "face", "FACE")
tmpl("F_12b", "fm12", 11, 6.5, "face", "FACE")
tmpl("H_24", "fm24", 9, 5.0, "head", "FACE")
tmpl("G_24", "gm24", 9, 4.5, "face", "GRAY")
tmpl("G_12", "gm12", 9, 4.5, "face", "GRAY")

# ---------------------------------------------------------------- stage 4
# Remaining scene colours at 24x24 and the global feature vector.
BL = np.array([-1.0, 0.0, 1.0]) * 6.0
RWc, RWb = affine(CR, 0.10, 0.5 - 0.66)
DKc, DKb = affine(-YV, 0.20, 0.25)
rowsS, biasS = [], []
for coef, b0 in [(BL, 0.0), (BL, -1.0), (RWc, RWb), (RWc, RWb - 1.0),
                 (DKc, DKb), (DKc, DKb - 1.0)]:
    rowsS.append(coef); biasS.append(b0)
for coef in [RG, -RG, GB, -GB]:
    rowsS.append(coef); biasS.append(0.0)
rowsS.append(YV); biasS.append(0.0)
nS = len(rowsS)
conv("scA", "rgb24", nS, 1, w=np.array(rowsS)[:, :, None, None], b=np.array(biasS))
relu("rscA", "scA")
SB = np.zeros((6, nS))
for j in range(3):
    SB[2 * j] = np.eye(1, nS, 2 * j)[0] - np.eye(1, nS, 2 * j + 1)[0]
    SB[2 * j + 1] = SB[2 * j]
conv("scB", "rscA", 6, 1, w=SB[:, :, None, None], b=np.array([0., -1., 0., -1., 0., -1.]))
relu("rscB", "scB")
SC = np.zeros((3, 6))
for j in range(3):
    SC[j, 2 * j], SC[j, 2 * j + 1] = 1, -1
conv("scene", "rscB", 3, 1, w=SC[:, :, None, None], b=np.zeros(3))

# ---------------------------------------------------------------- assemble
# Two statistics per template map:
#   raw max  -- how strong the best match is (absolute strength)
#   z max    -- how far it stands out from the rest of THIS image (norm op),
#               which is scale free and cancels the unknown null level.
# Requiring both (a soft AND in the readout) rejects diffuse clutter, which
# has a high raw max but a low z, and faint specks, which have the reverse.
FEATN = []
for g in ("Bcs", "Bfd", "Bvg", "Bwh", "FACE", "GRAY"):
    for t in GROUPS[g]:
        FEATN.append(t)
NT = len(FEATN)
MEANS = ["Bcs_24r", "Bfd_24r"]
FRACS = [("cs24", 1), ("fd24", 1), ("vg24", 1), ("wh24", 1), ("s24", 1),
         ("tex24", 1), ("txn24", 1), ("ex24", 1), ("ey24", 1),
         ("scene", 3), ("rscA", 2)]
C = 2 * NT + len(MEANS) + sum(n for _, n in FRACS)
parts = []


def project(name, src, mapping):
    cin = SH[src][0]
    Wp = np.zeros((C, cin, 1, 1))
    for a, b in mapping.items():
        Wp[b, a, 0, 0] = 1.0
    conv(name, src, C, 1, w=Wp, b=np.zeros(C))
    parts.append(name)


for j, t in enumerate(FEATN):
    h = SH[t][1]
    pool(t + "_mx", t, h, kind="max")
    project(t + "_p", t + "_mx", {0: j})
    layer(t + "_n", "norm", t)
    pool(t + "_mz", t + "_n", h, kind="max")
    project(t + "_q", t + "_mz", {0: NT + j})
for j, t in enumerate(MEANS):
    pool(t + "_av", t, SH[t][1], kind="avg")
    project(t + "_meanp", t + "_av", {0: 2 * NT + j})
off = 2 * NT + len(MEANS)
FIDX = {}
for src, n in FRACS:
    pool(src + "_g", src, SH[src][1], kind="avg")
    if src == "rscA":
        conv("chrsum", src + "_g", 2, 1,
             w=np.stack([np.eye(1, nS, 10)[0], sum(np.eye(1, nS, q)[0] for q in (6, 7, 8, 9))]),
             b=[0.0, 0.0])
        project("frac_chr", "chrsum", {0: off, 1: off + 1})
        FIDX["Y"] = off; FIDX["chroma"] = off + 1
    else:
        project("frac_" + src, src + "_g", {i: off + i for i in range(n)})
        for i in range(n):
            FIDX[src + str(i)] = off + i
    off += n
add("feat", parts)
TIDX = {t: j for j, t in enumerate(FEATN)}
ZIDX = {t: NT + j for j, t in enumerate(FEATN)}
MIDX = {t: 2 * NT + j for j, t in enumerate(MEANS)}

# ---------------------------------------------------------------- stage 5
# Hand-written readout.
#   ramp:  u = clip01((x-t)/s)
#   AND:   a = relu(u_raw + u_z - 1)      (Lukasiewicz t-norm)
#   scale: max across scales, never a sum (a sum saturates on the null level
#          that every scale shares)
#   score: competition between matched positive and negative evidence
RAMPS = [(t, TIDX[t], 0.15, 0.45) for t in FEATN]
ZT = {48: (3.8, 2.0), 24: (3.2, 1.8), 12: (2.0, 1.5)}
RAMPS += [("z_" + t, ZIDX[t], *ZT[LEVEL[t]]) for t in FEATN]
RAMPS += [("A_cs", MIDX["Bcs_24r"], 0.03, 0.15), ("A_fd", MIDX["Bfd_24r"], 0.03, 0.15),
          ("F_cs", FIDX["cs240"], 0.06, 0.25), ("F_fd", FIDX["fd240"], 0.06, 0.25),
          ("F_vg", FIDX["vg240"], 0.25, 0.40), ("F_wh", FIDX["wh240"], 0.25, 0.40),
          ("F_sk", FIDX["s240"], 0.15, 0.30), ("TEX", FIDX["tex240"], 0.05, 0.06),
          ("TXN", FIDX["txn240"], 0.40, 0.40),
          ("EX", FIDX["ex240"], 0.05, 0.06), ("EY", FIDX["ey240"], 0.05, 0.06),
          ("BLU", FIDX["scene0"], 0.30, 0.40), ("RED", FIDX["scene1"], 0.10, 0.25),
          ("DRK", FIDX["scene2"], 0.30, 0.40), ("GRY", FIDX["chroma"], 0.03, -0.025)]
R = len(RAMPS)
RI = {n: i for i, (n, _, _, _) in enumerate(RAMPS)}
D1 = np.zeros((2 * R, C)); B1v = np.zeros(2 * R)
for i, (n, fi, t, s) in enumerate(RAMPS):
    D1[2 * i, fi] = 1.0 / s; B1v[2 * i] = -t / s
    D1[2 * i + 1] = D1[2 * i]; B1v[2 * i + 1] = B1v[2 * i] - 1.0
dense("r1", "feat", 2 * R, D1, B1v)
relu("r1r", "r1")
DU = np.zeros((R, 2 * R))
for i in range(R):
    DU[i, 2 * i], DU[i, 2 * i + 1] = 1.0, -1.0
dense("uv", "r1r", R, DU, np.zeros(R))
relu("uvr", "uv")


def eu(i, n):
    v = np.zeros(n); v[i] = 1.0
    return v


# soft AND of strength and standing-out, one per template map
DA = np.zeros((NT, R)); BA = np.full(NT, -1.0)
for j, t in enumerate(FEATN):
    DA[j, RI[t]] = 1.0
    DA[j, RI["z_" + t]] = 1.0
dense("andv", "uvr", NT, DA, BA)
relu("andr", "andv")


def maxchain(gname, idxs):
    """max of the selected values, built from dense+relu only."""
    base = "mx_" + gname
    k = len(idxs)
    if k == 1:
        dense(base, "andr", 1, eu(idxs[0], NT)[None], [0.0])
        return base
    Wm = np.zeros((k, NT))
    Wm[0, idxs[0]], Wm[0, idxs[1]], Wm[1, idxs[1]] = 1.0, -1.0, 1.0
    for j in range(2, k):
        Wm[j, idxs[j]] = 1.0
    dense(base + "_1", "andr", k, Wm, np.zeros(k))
    relu(base + "_1r", base + "_1")
    cur, L, st = base + "_1r", k, 2
    while L > 2:
        Wn = np.zeros((L - 1, L))
        Wn[0, 0], Wn[0, 1], Wn[0, 2] = 1.0, 1.0, -1.0
        for j in range(1, L - 1):
            Wn[j, j + 1] = 1.0
        dense(base + "_%d" % st, cur, L - 1, Wn, np.zeros(L - 1))
        relu(base + "_%dr" % st, base + "_%d" % st)
        cur, L, st = base + "_%dr" % st, L - 1, st + 1
    dense(base + "_fin", cur, 1, np.ones((1, 2)), [0.0])
    return base + "_fin"


GMAX = {}
for g in ("Bcs", "Bfd", "Bvg", "Bwh", "FACE", "GRAY"):
    GMAX[g] = maxchain(g, [FEATN.index(t) for t in GROUPS[g]])

PASS = ["A_cs", "A_fd", "F_cs", "F_fd", "F_vg", "F_wh", "F_sk", "TEX", "TXN",
        "EX", "EY", "BLU", "RED", "DRK", "GRY"]
C2 = len(GMAX) + len(PASS) + 2
GI = {g: i for i, g in enumerate(GMAX)}
PI = {p: len(GMAX) + i for i, p in enumerate(PASS)}
NFI, GBWI = C2 - 2, C2 - 1
gparts = []
for g, src in GMAX.items():
    dense("gp_" + g, src, C2, eu(GI[g], C2)[:, None], np.zeros(C2))
    gparts.append("gp_" + g)
Wp = np.zeros((C2, R))
for p in PASS:
    Wp[PI[p], RI[p]] = 1.0
dense("gp_pass", "uvr", C2, Wp, np.zeros(C2))
gparts.append("gp_pass")
add("gv", gparts)


def gv(i):
    return eu(i, C2)


# large smooth-skin area with neither a blob nor a face: wood, leather, a wall
nfrow = gv(PI["F_cs"]) - 0.8 * gv(GI["FACE"]) - 0.5 * gv(GI["Bcs"])
gbrow = gv(GI["GRAY"]) + gv(PI["GRY"])
dense("nf", "gv", 4, np.stack([nfrow / 0.5, nfrow / 0.5, gbrow / 0.5, gbrow / 0.5]),
      np.array([-0.5, -1.5, -2.0, -3.0]))
relu("nfr", "nf")
Wnf = np.zeros((C2, 4))
Wnf[NFI, 0], Wnf[NFI, 1] = 1.0, -1.0
Wnf[GBWI, 2], Wnf[GBWI, 3] = 1.0, -1.0
dense("nf_p", "nfr", C2, Wnf, np.zeros(C2))
dense("gv_id", "gv", C2, np.eye(C2), np.zeros(C2))
add("gv2", ["gv_id", "nf_p"])

BIAS = 0.30
SCORE = (3.0 * gv(GI["Bcs"]) + 3.6 * gv(GI["FACE"])
         + 0.25 * gv(GI["GRAY"]) + 1.0 * gv(GBWI)
         - 1.6 * gv(GI["Bfd"]) - 0.7 * gv(GI["Bvg"]) - 0.9 * gv(GI["Bwh"])
         + 0.8 * gv(PI["F_cs"]) - 1.6 * gv(PI["F_fd"])
         - 0.5 * gv(PI["F_vg"]) - 0.7 * gv(PI["F_wh"])
         + 0.4 * gv(PI["A_cs"]) - 0.4 * gv(PI["A_fd"])
         - 0.2 * gv(PI["TEX"]) - 0.3 * gv(PI["RED"]) - 0.15 * gv(PI["BLU"])
         - 0.15 * gv(PI["DRK"]) - 1.4 * gv(NFI))
dense("score", "gv2", 2, np.stack([-0.5 * SCORE, 0.5 * SCORE]),
      np.array([0.5 * BIAS, -0.5 * BIAS]))

# ---------------------------------------------------------------- write
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": LAYERS}, f, indent=1)
np.savez(os.path.join(OUT, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
print("layers", len(LAYERS), "weights", sum(v.size for v in W.values()), "features", C)
