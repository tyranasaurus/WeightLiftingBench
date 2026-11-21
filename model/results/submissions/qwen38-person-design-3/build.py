"""Build weights for a hand-designed person detector on 96x96 COCO photos.

Everything here comes from formulas and general knowledge about people in
everyday photos (faces are skin-coloured discs with a dark eye band and hair
above; heads sit on wider shoulders; bodies are vertically elongated).
No data is read; no fitting is done.

Architecture sketch
-------------------
base maps (96x96): smoothed luminance/RGB, gradient magnitude e, soft skin
mask sk (RGB rule violations), soft dark mask dk (hair/eyes/cloth).
pyramid: avgpool x2 -> levels L1(48), L2(24), L3(12).
At each level a bank of parametric correlation templates (face / head-shoulder
/ skin-blob) over channels [e, sk, dk]; each template kernel is zero-DC per
channel (uniform fields give 0) and normalised so a perfect match scores ~1.
Per template: global max -> scalar -> threshold+gain+clip -> evidence d in [0,1].
Per cue type: max over scales/levels -> D_face, D_hs, D_sk.
Final rule: p = 1.00*D_face + 0.85*D_hs + 0.60*D_sk ; person iff p > 0.80,
emitted as scores (0.80 - p, p - 0.80).
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

layers = []
weights = {}


def L(name, op, inputs, **settings):
    layers.append({"name": name, "op": op, "inputs": inputs, **settings})
    return name


def conv(name, inp, W, b, stride=1, padding=0):
    W = np.asarray(W, np.float64)
    b = np.asarray(b, np.float64)
    assert W.ndim == 4 and b.shape == (W.shape[0],)
    weights[name + ".W"] = W
    weights[name + ".b"] = b
    return L(name, "conv", [inp], out=int(W.shape[0]), kernel=[int(W.shape[2]), int(W.shape[3])],
             stride=int(stride), padding=int(padding))


def dense(name, inp, W, b):
    W = np.asarray(W, np.float64).reshape(len(np.atleast_2d(np.asarray(W, np.float64))), -1)
    b = np.asarray(b, np.float64)
    weights[name + ".W"] = W
    weights[name + ".b"] = b
    return L(name, "dense", [inp], out=int(W.shape[0]))


def gauss_grid(k):
    c = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k].astype(np.float64)
    return x - c, y - c


def blob(k, cx, cy, sig):
    """isotropic gaussian at offset (cx, cy) from kernel centre, in x, y"""
    x, y = gauss_grid(k)
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sig * sig))


def disc(k, cx, cy, r):
    x, y = gauss_grid(k)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.clip(r + 0.5 - d, 0, 1)


def ring(k, cx, cy, r, sig):
    x, y = gauss_grid(k)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.exp(-((d - r) ** 2) / (2.0 * sig * sig))


def hband(k, cy, sy, hx):
    """horizontal band: gaussian in y, soft rectangle in x"""
    x, y = gauss_grid(k)
    return np.exp(-((y - cy) ** 2) / (2.0 * sy * sy)) * np.clip(hx + 0.5 - np.abs(x), 0, 1)


def vband(k, cx, sx, y0, y1):
    x, y = gauss_grid(k)
    return np.exp(-((x - cx) ** 2) / (2.0 * sx * sx)) * np.clip(np.minimum(y - y0, y1 - y) + 0.5, 0, 1)


def annulus(k, cx, cy, r0, r1):
    x, y = gauss_grid(k)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.clip(np.minimum(d - r0, r1 - d) + 0.5, 0, 1)


def finish(kernels):
    """zero-DC per channel (uniform fields score 0), then normalise by the
    positive mass of the raw kernel: the part that can actually be activated
    on a real image, where unmatched background sits at 0."""
    pos = sum(float(np.maximum(c, 0).sum()) for c in kernels)
    for c in kernels:
        c -= c.mean()
        c /= pos
    return kernels


def t_face(r):
    """frontal face: skin disc, dark eye pair + hair cap + mouth, edge ring"""
    k = 2 * int(np.ceil(3.8 * r)) + 1
    sk = 1.40 * disc(k, 0, 0.10 * r, 0.75 * r) - 0.60 * annulus(k, 0, 0, 1.30 * r, 2.00 * r)
    dk = (0.70 * blob(k, -0.40 * r, -0.10 * r, 0.22 * r) + 0.70 * blob(k, 0.40 * r, -0.10 * r, 0.22 * r)
          + 0.70 * blob(k, 0, -0.95 * r, 0.42 * r) + 0.30 * blob(k, 0, 0.45 * r, 0.20 * r)
          - 1.00 * blob(k, 0, 0.55 * r, 0.35 * r)     # cheeks/nose bridge are not dark
          - 0.90 * blob(k, 0, 0.78 * r, 0.22 * r))    # animal muzzle/nose is dark, chin is not
    e = 0.30 * ring(k, 0, 0, 1.05 * r, 0.24 * r) + 0.20 * disc(k, 0, 0.05 * r, 0.55 * r)
    sk_n = -0.50 * blob(k, 0, 2.80 * r, 0.80 * r)     # skin far below = trunk, not neck
    return finish([e, sk + sk_n, dk]), k


def t_hs(r):
    """head above shoulders: edge ring on head, shoulder line at ~+1.3 head
    radii spanning ~+-2.2r, torso/arm sides tapering below, skin face disc,
    dark hair cap"""
    k = 2 * int(np.ceil(4.8 * r)) + 1
    e = (0.80 * ring(k, 0, 0, r, 0.28 * r) + 0.80 * hband(k, 1.30 * r, 0.30 * r, 2.20 * r)
         + 0.45 * vband(k, -1.90 * r, 0.30 * r, 1.50 * r, 4.50 * r)
         + 0.45 * vband(k, 1.90 * r, 0.30 * r, 1.50 * r, 4.50 * r))
    sk = 0.60 * disc(k, 0, 0.05 * r, 0.55 * r)
    dk = 1.00 * blob(k, 0, -0.50 * r, 0.50 * r)
    return finish([e, sk, dk]), k


def t_sk(r):
    """compact skin blob with internal texture / hair above (faces in
    profile, hands, limbs)"""
    k = 2 * int(np.ceil(2.4 * r)) + 1
    sk = 1.00 * disc(k, 0, 0, r) - 0.0  # zero-DC adds the surround suppression
    sk -= 1.00 * annulus(k, 0, 0, 1.40 * r, 2.20 * r)
    e = 0.35 * ring(k, 0, 0, r, 0.25 * r) + 0.30 * disc(k, 0, 0, 0.50 * r)
    dk = 0.40 * blob(k, 0, -1.40 * r, 0.50 * r)
    return finish([e, sk, dk]), k


# ---------------------------------------------------------------- base maps
SIG = 1.1
k5 = 2 * int(np.ceil(3 * SIG)) + 1
g = np.exp(-(gauss_grid(k5)[0] ** 2 + gauss_grid(k5)[1] ** 2) / (2 * SIG * SIG))
g /= g.sum()
LUM = np.array([0.299, 0.587, 0.114])
W = np.zeros((4, 3, k5, k5))
for o, coef in enumerate([LUM, [1, 0, 0], [0, 1, 0], [0, 0, 1]]):
    for i in range(3):
        W[o, i] = g * coef[i]
sm = conv("sm", "image", W, np.zeros(4), stride=1, padding=k5 // 2)

# gradients of luminance (sobel)
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64) / 8.0
SY = SX.T
Wg = np.zeros((2, 4, 3, 3))
Wg[0, 0] = SX
Wg[1, 0] = SY
gxy = conv("gxy", sm, Wg, np.zeros(2), stride=1, padding=1)

W4 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.float64)  # gx, -gx, gy, -gy
g4 = conv("g4", gxy, W4.reshape(4, 2, 1, 1), np.zeros(4))
# |gx| + |gy| : split channels, relu each, sum
g4r_ch = []
for i in range(4):
    Ws = np.zeros((1, 4, 1, 1))
    Ws[0, i] = 1.0
    nm = conv(f"gs{i}", g4, Ws, np.zeros(1))
    g4r_ch.append(L(f"gsr{i}", "relu", [nm]))
eraw = L("eraw", "add", g4r_ch)
esc = conv("esc", eraw, np.array([6.0]).reshape(1, 1, 1, 1), np.array([-0.08]))
escr = L("escr", "relu", [esc])


def clip01(x, tag, nonneg=False):
    """returns layer name of clip(x, 0, 1)"""
    t1 = x if nonneg else L(tag + "_c0", "relu", [x])
    t2 = conv(tag + "_c1", t1, np.ones((1, 1, 1, 1)), np.array([-1.0]))
    t3 = L(tag + "_c2", "relu", [t2])
    t4 = conv(tag + "_c3", t3, -np.ones((1, 1, 1, 1)), np.zeros(1))
    return L(tag + "_c4", "add", [t1, t4])


e_abs = clip01(escr, "eclip", nonneg=True)


def salience(x, tag, slope, bias):
    """per-image contrast gain control: how much x stands out in this image"""
    n = L(tag + "_n", "norm", [x])
    s = conv(tag + "_s", n, np.array([slope]).reshape(1, 1, 1, 1), np.array([bias]))
    sr = L(tag + "_sr", "relu", [s])
    return clip01(sr, tag + "_sc", nonneg=True)


def soft_and(a, b, tag, t=0.9):
    """clip01(a + b - t): fires only where both a and b are high"""
    s = L(tag + "_a", "add", [a, b])
    r = L(tag + "_r", "relu", [conv(tag + "_c", s, np.ones((1, 1, 1, 1)), np.array([-t]))])
    return clip01(r, tag + "_k", nonneg=True)


# edges: absolute strength AND standing out in this image
e0 = soft_and(e_abs, salience(eraw, "esal", 0.30, 0.50), "eand", 0.70)

# skin mask: relu(1 - 2.2 * sum of rule violations)
VW = np.zeros((8, 4, 1, 1))
VB = np.zeros(8)
# channels of sm: 0=L 1=R 2=G 3=B
rules = [
    ([-0, -1, 1, 0], 0.05),   # 0.05 - (R-G)
    ([-0, -1, 0, 1], 0.10),   # 0.10 - (R-B)
    ([0, 0, -1, 1], 0.015),   # 0.015 - (G-B)
    ([-1, 0, 0, 0], 0.33),    # 0.33 - R   (too dark)
    ([0, -1, 0, 0], 0.17),    # 0.17 - G
    ([0, 0, -1, 0], 0.09),    # 0.09 - B
    ([0, 1, 0, -1], -0.60),   # (R-B) - 0.60  (over-saturated red)
    ([0, 1, -1, 0], -0.40),   # (R-G) - 0.40  (over-orange)
]
for j, (coefs, bias) in enumerate(rules):
    # violation v_j = bias - sum(coefs * ch)  where listed coefs multiply ch
    # rules above written as value = bias + sum(c_i * ch_i) with c as given sign
    VW[j, :, 0, 0] = coefs
    VB[j] = bias
skv = conv("skv", sm, VW, VB)
skvr = L("skvr", "relu", [skv])
sk_abs = L("sk0", "relu", [conv("skw", skvr, -5.0 * np.ones((1, 8, 1, 1)), np.array([1.0]))])
# a uniform skin-coloured wall is skin in absolute terms but not salient
sk0 = soft_and(sk_abs, salience(sk_abs, "sksal", 0.50, -0.10), "skand", 0.90)

# dark mask: clip01((0.38 - L) * 3.3)
dk1 = conv("dk1", sm, np.array([[-3.3, 0, 0, 0]]).reshape(1, 4, 1, 1), np.array([0.38 * 3.3]))
dk1r = L("dk1r", "relu", [dk1])
dk_abs = clip01(dk1r, "dkclip", nonneg=True)
dk0 = soft_and(dk_abs, salience(dk_abs, "dksal", 0.50, -0.10), "dkand", 0.90)

# ---------------------------------------------------------------- pyramid
maps = {0: {"e": e0, "sk": sk0, "dk": dk0}}
for lv in (1, 2, 3):
    # edges keep their strength under maxpool; region masks average
    maps[lv] = {"e": L(f"p{lv}e", "maxpool", [maps[lv - 1]["e"]], size=[2, 2], stride=[2, 2])}
    for ch in ("sk", "dk"):
        maps[lv][ch] = L(f"p{lv}{ch}", "avgpool", [maps[lv - 1][ch]], size=[2, 2], stride=[2, 2])

# ---------------------------------------------------------------- templates
SPEC = {
    1: {"stride": 4, "hs": [1.5, 2, 2.5, 3, 4], "face": [2, 2.5, 3, 4], "sk": [2, 3, 4]},
    2: {"stride": 2, "hs": [2, 3, 4], "face": [2, 2.5, 3, 4], "sk": [2, 3, 4]},
    3: {"stride": 2, "hs": [2, 3, 4, 6], "face": [2, 3, 4, 6, 9], "sk": [2, 3, 4, 6]},
}
THETA = {"face": 0.26, "hs": 0.26, "sk": 0.50}
GAIN = 5.0

scalars = {"face": [], "hs": [], "sk": []}
for lv in (1, 2, 3):
    spec = SPEC[lv]
    tlist = []
    for typ in ("hs", "face", "sk"):
        for r in spec[typ]:
            kern, k = {"hs": t_hs, "face": t_face, "sk": t_sk}[typ](r)
            tlist.append((typ, kern, k))
    kmax = max(k for _, _, k in tlist)
    K = len(tlist)
    resp_ch = []
    for ci, ch in enumerate(("e", "sk", "dk")):
        Wt = np.zeros((K, 1, kmax, kmax))
        for ti, (_, kern, k) in enumerate(tlist):
            off = (kmax - k) // 2
            Wt[ti, 0, off:off + k, off:off + k] = kern[ci]
        bt = np.zeros(K)
        resp_ch.append(conv(f"t{lv}{ch}", maps[lv][ch], Wt, bt, stride=spec["stride"], padding=kmax // 2))
    resp = L(f"resp{lv}", "add", resp_ch)
    respr = L(f"respr{lv}", "relu", [resp])
    ho = {1: 12, 2: 12, 3: 6}[lv]
    mp = L(f"mp{lv}", "maxpool", [respr], size=[ho, ho], stride=[1, 1])
    rs = L(f"rs{lv}", "reshape", [mp], shape=[K])
    for ti, (typ, _, _) in enumerate(tlist):
        Wsel = np.zeros((1, K))
        Wsel[0, ti] = GAIN
        a = dense(f"d{lv}_{ti}a", rs, Wsel, np.array([-GAIN * THETA[typ]]))
        a1 = L(f"d{lv}_{ti}b", "relu", [a])
        c1 = dense(f"d{lv}_{ti}c", a1, np.ones((1, 1)), np.array([-1.0]))
        c2 = L(f"d{lv}_{ti}d", "relu", [c1])
        c3 = dense(f"d{lv}_{ti}e", c2, -np.ones((1, 1)), np.zeros(1))
        d = L(f"d{lv}_{ti}f", "add", [a1, c3])
        scalars[typ].append(d)


def maxchain(names, tag):
    m = names[0]
    for i, nm in enumerate(names[1:]):
        ng = dense(f"{tag}m{i}n", m, -np.ones((1, 1)), np.zeros(1))
        sb = L(f"{tag}m{i}s", "add", [nm, ng])
        rb = L(f"{tag}m{i}r", "relu", [sb])
        m = L(f"{tag}m{i}a", "add", [m, rb])
    return m


Df = maxchain(scalars["face"], "mf")
Dh = maxchain(scalars["hs"], "mh")
Ds = maxchain(scalars["sk"], "ms")

WF, WH, WS, THF = 1.00, 0.90, 0.40, 0.85
ff = dense("ff", Df, np.array([[-WF], [WF]]), np.array([THF / 3 * 1.0, -THF / 3]))
fh = dense("fh", Dh, np.array([[-WH], [WH]]), np.array([THF / 3, -THF / 3]))
fs = dense("fs", Ds, np.array([[-WS], [WS]]), np.array([THF / 3, -THF / 3]))
final = L("scores", "add", [ff, fh, fs])

# ---------------------------------------------------------------- write
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "network.json").open("w") as f:
    json.dump({"layers": layers}, f)
np.savez(OUT / "weights.npz", **{k: v.astype(np.float32) for k, v in weights.items()})
total = sum(v.size for v in weights.values())
print(f"wrote {len(layers)} layers, {total} weights to {OUT}")
