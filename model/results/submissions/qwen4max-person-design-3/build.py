"""Hand-designed person / no-person network for 96x96 COCO photos.

Every number comes from general knowledge about people in photographs and from
closed-form filter formulas (differences of Gaussians, ellipses, bands, Laplacians).
This file reads no files at all.

Architecture
------------
A. Skin-colour likelihood map: a soft AND of seven hand-written linear chromaticity
   rules (hue below ~33 deg, R>G>B, saturation floor and ceiling, luminance window).
   The colour rules are homogeneous in RGB so they survive changes of exposure.
B. Luminance pyramid; at every scale a dark-spot map (pixel darker than its 3x3
   neighbourhood at that scale) and a texture map (|Laplacian| of luminance).
C. Eight scales, cell sizes 1,2,3,4,6,8,12,16 px (1.5x steps), each matching four
   formula templates against those maps:
     blob  : difference of Gaussians -> compact skin region surrounded by non-skin
     head  : blob + "skin below / hair above" asymmetry + vertical elongation
             (people stand upright; animals, floors and food lie horizontally)
     face  : skin ellipse with a non-skin ring, plus two dark eye bands with a
             lighter nose bridge and lighter cheeks between/below them
     tblob : blob minus local texture (fur, food and wood grain are textured)
   Responses are averaged over 2x2 neighbourhoods before the global max, so an
   isolated noise peak cannot win.
D. Global statistics: skin fraction, densest skin windows, top/bottom skin balance.
E. A hand-written rule table (dense/relu x4).  Large skin masses only count when a
   face supports them; that is what separates people from wood, sand, food, animals.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)

layers, weights = [], {}


def layer(name, op, inputs, **st):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(st)
    layers.append(d)
    return name


def conv(name, inp, w, b, stride=1, pad=0):
    w = np.asarray(w, np.float64)
    b = np.asarray(b, np.float64)
    if w.ndim == 2:
        w = w[:, :, None, None]
    k = w.shape[-1]
    layer(name, "conv", [inp], out=w.shape[0], kernel=[k, k], stride=stride, padding=pad)
    weights[name + ".W"], weights[name + ".b"] = w, b
    return name


def dense(name, inp, w, b):
    w, b = np.asarray(w, np.float64), np.asarray(b, np.float64)
    layer(name, "dense", [inp], out=w.shape[0])
    weights[name + ".W"], weights[name + ".b"] = w, b
    return name


def relu(name, inp):
    return layer(name, "relu", [inp])


def pool(name, inp, kind, size, stride=None):
    stride = size if stride is None else stride
    return layer(name, kind, [inp], size=list(size), stride=list(stride))


def add(name, *ins):
    return layer(name, "add", list(ins))


# ---------------------------------------------------------------- A. skin colour
YCOEF = np.array([0.299, 0.587, 0.114])
RULES = [
    # hue < ~33 deg: (G-B) < 0.55*(R-B).  Rejects sand, beige walls, cardboard,
    # yellow food, light wood; keeps pale through very dark skin.
    (np.array([0.55, -1.0, 0.45]), 0.015, 0.042, 2.0),
    # redness R-G: rejects grey / white / black-and-white photos
    (np.array([1.0, -1.0, 0.0]), -0.015, 0.070, 1.0),
    # saturation ceiling B > 0.36R: rejects carrot, red cloth, terracotta, cow brown
    (np.array([-0.36, 0.0, 1.0]), 0.010, 0.050, 1.4),
    # saturation floor R-B: rejects achromatic / washed-out pixels
    (np.array([1.0, 0.0, -1.0]), -0.025, 0.060, 0.8),
    # G > B: rejects pure reds / magentas
    (np.array([0.0, 1.0, -1.0]), -0.005, 0.040, 0.7),
    # G < 1.42 B: skin never has a large green/blue gap.  Rejects dark wood,
    # terracotta courts, brown fur, while keeping pale through very dark skin.
    (np.array([0.0, -1.0, 1.42]), 0.015, 0.045, 1.5),
    (YCOEF, -0.050, 0.080, 0.6),          # not too dark
    (-YCOEF, 0.985, 0.060, 0.4),          # not blown out
]
SUMW = sum(r[3] for r in RULES)
SKIN_T, SKIN_SCALE = SUMW - 1.30, 0.90

n = len(RULES)
Wlin, blin = np.zeros((2 * n, 3)), np.zeros(2 * n)
for i, (a, off, wid, _) in enumerate(RULES):
    Wlin[2 * i] = Wlin[2 * i + 1] = a / wid
    blin[2 * i], blin[2 * i + 1] = off / wid, off / wid - 1.0
conv("c_lin", "image", Wlin, blin)
relu("c_relu", "c_lin")
Wh = np.zeros((n, 2 * n))
for i in range(n):
    Wh[i, 2 * i], Wh[i, 2 * i + 1] = 1.0, -1.0
conv("c_h", "c_relu", Wh, np.zeros(n))
conv("c_skin", "c_h", np.array([[r[3] / SKIN_SCALE for r in RULES]]),
     np.array([-SKIN_T / SKIN_SCALE]))
S = relu("S", "c_skin")
conv("Y", "image", YCOEF[None, :], np.zeros(1))
lap = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
conv("lp", "Y", lap[None, None], np.zeros(1), pad=1)
conv("lps", "lp", np.array([[1.0], [-1.0]]), np.zeros(2))
relu("lpr", "lps")
LA = conv("LA", "lpr", np.array([[1.0, 1.0]]), np.zeros(1))

# ---------------------------------------------------------------- B. pyramid
CELLS = [1, 2, 3, 4, 6, 8, 12, 16]
NL = len(CELLS)
SIZE = [96 // c for c in CELLS]
SP, YP, TP, DP = [S], ["Y"], [LA], []
for lv in range(1, NL):
    c = CELLS[lv]
    SP.append(pool(f"S{lv}", S, "avgpool", [c, c]))
    YP.append(pool(f"Y{lv}", "Y", "avgpool", [c, c]))
    TP.append(pool(f"T{lv}", LA, "avgpool", [c, c]))
k3 = np.ones((3, 3)) / 9.0
k3[1, 1] -= 1.0
for lv in range(NL):
    DP.append(relu(f"D{lv}", conv(f"dk{lv}", YP[lv], 2.5 * k3[None, None], np.zeros(1), pad=1)))

# ---------------------------------------------------------------- C. templates
K = 9
ax = np.arange(K) - K // 2
YY, XX = np.meshgrid(ax, ax, indexing="ij")


def gauss(s, cy=0.0, cx=0.0):
    g = np.exp(-(((YY - cy) ** 2 + (XX - cx) ** 2) / (2 * s * s)))
    return g / g.sum()


def aniso(sy, sx):
    g = np.exp(-((YY / sy) ** 2 + (XX / sx) ** 2) / 2.0)
    return g / g.sum()


def norm_pos(w):
    p, m = np.maximum(w, 0.0), np.minimum(w, 0.0)
    if p.sum() > 0:
        p = p / p.sum()
    if m.sum() < 0:
        m = m / m.sum()
    return p + m


DOG = gauss(1.0) - gauss(2.0)
VERT = aniso(2.0, 0.9) - aniso(0.9, 2.0)
HEAD = DOG + 0.35 * (gauss(1.0, cy=2.2) - gauss(1.0, cy=-2.2)) + 0.30 * VERT
ell = (XX / 1.7) ** 2 + (YY / 2.1) ** 2
FACE_S = norm_pos(np.where(ell <= 1.0, 1.0, 0.0)
                  - np.where((ell > 1.96) & (ell <= 4.4), 1.0, 0.0))
# eyes: two dark patches, referenced against nose bridge and cheeks
FACE_D = np.zeros((K, K))
for (r, c) in [(-1, -1), (-1, 1), (0, -1), (0, 1)]:
    FACE_D[r + K // 2, c + K // 2] = 0.25
for (r, c) in [(-1, 0), (0, 0), (1, -1), (1, 0), (1, 1)]:
    FACE_D[r + K // 2, c + K // 2] = -0.20

K5 = 5
a5 = np.arange(K5) - K5 // 2
Y5, X5 = np.meshgrid(a5, a5, indexing="ij")
g5 = np.exp(-(Y5 ** 2 + X5 ** 2) / 2.0)
g5 /= g5.sum()
Z9, Z5 = np.zeros((K, K)), np.zeros((K5, K5))
TEMPL_S = np.stack([2.0 * DOG, 2.0 * HEAD, 1.2 * FACE_S, 2.0 * DOG])
FACE_D_W = 3.0
FACE_D_CROP = FACE_D[K // 2 - 2:K // 2 + 3, K // 2 - 2:K // 2 + 3]
TEMPL_D = np.stack([Z5, Z5, 0.8 * FACE_D_CROP, Z5])
TEMPL_T = np.stack([Z5, Z5, Z5, -2.0 * g5])

mx, cov = [], []
for lv in range(NL):
    st = 2 if CELLS[lv] == 1 else 1
    a = conv(f"b{lv}", SP[lv], TEMPL_S[:, None], np.zeros(4), stride=st, pad=K // 2)
    b = conv(f"e{lv}", DP[lv], TEMPL_D[:, None], np.zeros(4), stride=st, pad=K5 // 2)
    c = conv(f"t{lv}", TP[lv], TEMPL_T[:, None], np.zeros(4), stride=st, pad=K5 // 2)
    m = add(f"m{lv}", a, b, c)
    h = (SIZE[lv] + 2 * (K // 2) - K) // st + 1
    d = pool(f"dn{lv}", m, "avgpool", [2, 2], [1, 1])
    mx.append(pool(f"mx{lv}", d, "maxpool", [h - 1, h - 1]))
    q = relu(f"cvr{lv}", conv(f"cv{lv}", m, np.array([[1.0, 0, 0, 0]]), np.array([-0.30])))
    cov.append(pool(f"cvp{lv}", q, "avgpool", [h, h]))

# black-and-white photos have R=G=B exactly, so a luminance-only face template
# can recover people in them; it is switched off by any real colourfulness.
BW_LEVELS = [1, 2, 3, 4, 5]
bwm = []
for lv in BW_LEVELS:
    ya = conv(f"gy{lv}", YP[lv], (1.2 * FACE_S)[None, None], np.zeros(1), pad=K // 2)
    yb = conv(f"gd{lv}", DP[lv], (2.0 * FACE_D_CROP)[None, None],
              np.zeros(1), pad=K5 // 2)
    mm = add(f"gm{lv}", ya, yb)
    hh = SIZE[lv]
    dd = pool(f"gn{lv}", mm, "avgpool", [2, 2], [1, 1])
    bwm.append(pool(f"gmx{lv}", dd, "maxpool", [hh - 1, hh - 1]))
conv("cf1", "image", np.array([[1., -1., 0.], [-1., 1., 0.], [0., 1., -1.], [0., -1., 1.]]), np.zeros(4))
relu("cf1r", "cf1")
CF = pool("cf", conv("cf2", "cf1r", np.ones((1, 4)), np.zeros(1)), "avgpool", [96, 96])

# ---------------------------------------------------------------- D. globals
g_frac = pool("g_frac", S, "avgpool", [96, 96])
g_b24 = pool("g_b24", pool("g_b24a", S, "avgpool", [24, 24]), "maxpool", [4, 4])
g_b48 = pool("g_b48", pool("g_b48a", S, "avgpool", [48, 48]), "maxpool", [2, 2])
g_m12 = pool("g_m12", pool("g_m12a", S, "avgpool", [12, 12]), "maxpool", [8, 8])
halves = pool("g_half", S, "avgpool", [48, 96], [48, 96])   # (1,2,1): top, bottom

# ---------------------------------------------------------------- E. feature vector
# 0..31  max of [blob,head,face,tblob] per level (channel-major: c*NL+lv)
# 32..39 blob coverage per level
# 40 skin fraction, 41 max24px density, 42 max48px density, 43 max12px density,
# 44 top-half skin, 45 bottom-half skin
NF = 52
scatters = []


def scatter(name, src, slot, nin, scale=1.0, ch=0):
    w = np.zeros((NF, nin))
    w[slot, ch] = scale
    scatters.append(dense(name, src, w, np.zeros(NF)))


for lv in range(NL):
    for c in range(4):
        scatter(f"sc_{lv}_{c}", mx[lv], c * NL + lv, 4, ch=c)
    scatter(f"sc_cov{lv}", cov[lv], 32 + lv, 1)
scatter("sc_frac", g_frac, 40, 1)
scatter("sc_b24", g_b24, 41, 1)
scatter("sc_b48", g_b48, 42, 1)
scatter("sc_m12", g_m12, 43, 1)
scatter("sc_top", halves, 44, 2, ch=0)
scatter("sc_bot", halves, 45, 2, ch=1)
scatter("sc_cf", CF, 46, 1)
for i, lv in enumerate(BW_LEVELS):
    scatter(f"sc_bw{i}", bwm[i], 47 + i, 1)
FEAT = add("feat", *scatters)

# ---------------------------------------------------------------- F. rule table
BT = [0.40, 0.38, 0.36, 0.36, 0.36, 0.38, 0.40, 0.42]
HT = [0.50, 0.48, 0.46, 0.46, 0.46, 0.48, 0.50, 0.52]
FT = [1.55] * NL          # skin ellipse AND eye bands must both be present
TT = [0.34, 0.32, 0.30, 0.30, 0.30, 0.32, 0.34, 0.36]
FTG = 1.15                # luminance-only face, for black-and-white photos

rows, biases, tags = [], [], []


def unit(coefs, bias, tag):
    r = np.zeros(NF)
    for i, v in coefs.items():
        r[i] = v
    rows.append(r)
    biases.append(bias)
    tags.append(tag)


for lv in range(NL):
    unit({lv: 1.0}, -BT[lv], f"b{lv}")
    unit({NL + lv: 1.0}, -HT[lv], f"h{lv}")
    unit({2 * NL + lv: 1.0}, -FT[lv], f"f{lv}")
    unit({3 * NL + lv: 1.0}, -TT[lv], f"t{lv}")
    unit({32 + lv: 1.0}, 0.0, f"cov{lv}")
for i in range(len(BW_LEVELS)):
    unit({47 + i: 1.0, 46: -50.0}, -FTG, f"bw{i}")
# low-threshold copies used by the "big skin mass" gates
for lv in (4, 5, 6, 7):
    unit({lv: 1.0}, -0.20, f"rb{lv}")
unit({40: 1.0}, -0.20, "rf20")
unit({40: 1.0}, -0.40, "rf40")
unit({41: 1.0}, -0.60, "rb24")
unit({42: 1.0}, -0.60, "rb48")
unit({45: 1.0, 44: -1.0}, -0.25, "rbot")
unit({40: 1.0}, -0.02, "frac02")
unit({40: -1.0}, 0.05, "lowskin")
unit({41: 1.0}, -0.88, "b24hi")
unit({43: 1.0}, -0.55, "m12")

H1 = len(rows)
dense("r1", FEAT, np.stack(rows), np.array(biases))
relu("r1r", "r1")
idx = {t: i for i, t in enumerate(tags)}

rows2 = list(np.eye(H1))
tags2 = list(tags)
r = np.zeros(H1)
for lv in range(NL):
    r[idx[f"f{lv}"]] = 1.0
rows2.append(r)
tags2.append("fsup")
dense("r2", "r1r", np.stack(rows2), np.zeros(H1 + 1))
relu("r2r", "r2")
j = {t: i for i, t in enumerate(tags2)}

rows3, bias3, tags3 = [], [], []


def unit3(coefs, bias, tag):
    r = np.zeros(H1 + 1)
    for t, v in coefs.items():
        r[j[t]] = v
    rows3.append(r)
    bias3.append(bias)
    tags3.append(tag)


for lv in range(NL):
    for p in ("b", "h", "f", "t", "cov"):
        unit3({f"{p}{lv}": 1.0}, 0.0, f"{p}{lv}")
for i in range(len(BW_LEVELS)):
    unit3({f"bw{i}": 1.0}, 0.0, f"bw{i}")
# a large skin-coloured mass is only a person if a face supports it
unit3({"rb6": 1.0, "rb7": 1.0, "fsup": -1.5}, -0.30, "g_coarse")
unit3({"rb5": 1.0, "fsup": -1.2}, -0.10, "g_mid")
unit3({"rb24": 1.0, "fsup": -1.5}, 0.0, "g_b24")
unit3({"rb48": 1.0, "fsup": -1.5}, 0.0, "g_b48")
unit3({"rf40": 1.0, "fsup": -1.2}, 0.0, "g_frac50")
unit3({"rf20": 1.0, "fsup": -0.8}, 0.0, "g_frac30")
unit3({"rbot": 1.0, "fsup": -0.5}, 0.0, "g_bot")
# scattered skin-coloured clutter with no face anywhere is not a person either
_nf = {f"b{lv}": 1.0 for lv in range(NL)}
_nf.update({f"h{lv}": 1.0 for lv in range(NL)})
_nf["fsup"] = -2.0
unit3(_nf, -0.42, "g_noface")
unit3({"frac02": 1.0}, 0.0, "frac02")
unit3({"lowskin": 1.0}, 0.0, "lowskin")
unit3({"b24hi": 1.0}, 0.0, "b24hi")
unit3({"m12": 1.0}, 0.0, "m12")

H3 = len(rows3)
dense("r3", "r2r", np.stack(rows3), np.array(bias3))
relu("r3r", "r3")
k = {t: i for i, t in enumerate(tags3)}

WB = [0.30, 0.50, 0.60, 0.70, 0.60, 0.40, 0.30, 0.20]
WH = [0.30, 0.50, 0.60, 0.70, 0.60, 0.40, 0.30, 0.20]
WF = [1.00, 1.60, 2.20, 2.40, 2.00, 1.60, 1.20, 0.80]
WT = [0.20, 0.30, 0.40, 0.50, 0.40, 0.30, 0.20, 0.15]
WC = [0.15, 0.20, 0.25, 0.25, 0.20, 0.15, 0.10, 0.08]
WG = [1.20, 1.80, 2.00, 1.80, 1.40]
wts = {"frac02": 0.5, "lowskin": -1.0, "b24hi": -1.5, "m12": 0.2,
       "g_coarse": -2.0, "g_mid": -1.8, "g_b24": -2.5, "g_b48": -2.5,
       "g_frac50": -2.5, "g_frac30": -1.5, "g_bot": -1.5, "g_noface": -0.8}
for lv in range(NL):
    wts[f"b{lv}"] = WB[lv]
    wts[f"h{lv}"] = WH[lv]
    wts[f"f{lv}"] = WF[lv]
    wts[f"t{lv}"] = WT[lv]
    wts[f"cov{lv}"] = WC[lv]
for i in range(len(BW_LEVELS)):
    wts[f"bw{i}"] = WG[i]
BIAS = 0.55
v = np.zeros(H3)
for t, w in wts.items():
    v[k[t]] = w
dense("r4", "r3r", np.stack([-v, v]), np.array([0.0, -2.0 * BIAS]))

net = {"layers": layers}
(OUT / "network.json").write_text(json.dumps(net, indent=1))
np.savez(OUT / "weights.npz", **{kk: vv.astype(np.float32) for kk, vv in weights.items()})
print(f"wrote {len(layers)} layers, {sum(x.size for x in weights.values()):,} weights")
