"""Hand-designed CIFAR-10 network.  Every weight comes from a closed-form formula
(Gaussians, Gaussian derivatives, Gabor-like rings/sectors, ellipse/ring shape models)
or from a constant I chose by hand from general knowledge of the ten classes.
No training data is read; nothing is fitted.

Usage:  python3 build.py OUT_DIR
"""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ----------------------------------------------------------------------------- net helper
LAYERS = []
WEIGHTS = {}


def layer(name, op, inputs, **settings):
    L = {"name": name, "op": op, "inputs": inputs if isinstance(inputs, list) else [inputs]}
    L.update(settings)
    LAYERS.append(L)
    return name


def conv(name, src, out, k=1, stride=1, pad=0, W=None, b=None):
    layer(name, "conv", src, out=out, kernel=[k, k], stride=stride, padding=pad)
    WEIGHTS[name + ".W"] = np.asarray(W, np.float32)
    WEIGHTS[name + ".b"] = np.asarray(b if b is not None else np.zeros(out), np.float32)
    return name


def dense(name, src, out, W=None, b=None):
    layer(name, "dense", src, out=out)
    WEIGHTS[name + ".W"] = np.asarray(W, np.float32)
    WEIGHTS[name + ".b"] = np.broadcast_to(
        np.asarray(b if b is not None else 0.0, np.float32), (out,)).copy()
    return name


def relu(name, src):
    return layer(name, "relu", src)


def add(name, *srcs):
    return layer(name, "add", list(srcs))


def pool(name, src, size, stride=None, kind="avg"):
    return layer(name, kind + "pool", src, size=[size, size],
                 stride=[stride or size, stride or size])


def reshape(name, src, shape):
    return layer(name, "reshape", src, shape=list(shape))


# ----------------------------------------------------------------------------- formula filters
def grid(k):
    r = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k].astype(np.float64)
    return x - r, y - r


def gauss(k, s):
    x, y = grid(k)
    g = np.exp(-(x * x + y * y) / (2 * s * s))
    return g / g.sum()


def dgauss(k, s, direction=0.0):
    """First derivative of a Gaussian along `direction` (radians, 0 = +x)."""
    x, y = grid(k)
    u = x * np.cos(direction) + y * np.sin(direction)
    g = np.exp(-(x * x + y * y) / (2 * s * s))
    d = -u / (s * s) * g
    d = d - d.mean()
    n = np.abs(d).sum()
    return d / n * 2.0          # unit L1 -> response of a unit step is ~1


def log(k, s):
    x, y = grid(k)
    r2 = x * x + y * y
    g = np.exp(-r2 / (2 * s * s))
    l = (r2 - 2 * s * s) / (s ** 4) * g
    l = l - l.mean()
    return l / np.abs(l).sum() * 2.0


def ring(k, r0, width, sign=1.0):
    """Annulus (a wheel / eye / dark blob outline) minus its surround."""
    x, y = grid(k)
    r = np.sqrt(x * x + y * y)
    a = np.exp(-((r - r0) ** 2) / (2 * width * width))
    b = np.exp(-(r ** 2) / (2 * (r0 * 1.9) ** 2))
    a = a / a.sum()
    b = b / b.sum()
    return sign * (a - b)


LUMA = np.array([0.299, 0.587, 0.114])

# ----------------------------------------------------------------------------- stage 1
# 5x5 bank on the raw photo: smoothed colour, luminance, oriented Gaussian
# derivatives at two scales, Laplacians and colour-opponent edges.
C1 = ["R", "G", "B", "Rs", "Gs", "Bs", "Rb", "Gb", "Bb", "L", "Ls",
      "dxf", "dyf", "d1f", "d2f", "dxc", "dyc", "d1c", "d2c",
      "logf", "logc", "crgx", "crgy", "cbyx", "cbyy", "Lb"]
NC1 = len(C1)
IX = {n: i for i, n in enumerate(C1)}


def build_c1():
    W = np.zeros((NC1, 3, 5, 5))
    for c in range(3):                                   # raw colour
        W[IX["RGB"[c]], c, 2, 2] = 1.0
    for c, nm in enumerate(["Rs", "Gs", "Bs"]):          # fine colour blur
        W[IX[nm], c] = gauss(5, 1.0)
    for c, nm in enumerate(["Rb", "Gb", "Bb"]):          # broad colour blur
        W[IX[nm], c] = gauss(5, 2.0)
    delta = np.zeros((5, 5))
    delta[2, 2] = 1.0
    W[IX["L"]] = LUMA[:, None, None] * delta[None]
    for nm, s in [("Ls", 1.0), ("Lb", 2.0)]:
        W[IX[nm]] = LUMA[:, None, None] * gauss(5, s)[None]
    for nm, s, a in [("dxf", 1.0, 0.0), ("dyf", 1.0, np.pi / 2),
                     ("d1f", 1.0, np.pi / 4), ("d2f", 1.0, 3 * np.pi / 4),
                     ("dxc", 2.0, 0.0), ("dyc", 2.0, np.pi / 2),
                     ("d1c", 2.0, np.pi / 4), ("d2c", 2.0, 3 * np.pi / 4)]:
        W[IX[nm]] = LUMA[:, None, None] * dgauss(5, s, a)[None]
    for nm, s in [("logf", 1.0), ("logc", 2.0)]:
        W[IX[nm]] = LUMA[:, None, None] * log(5, s)[None]
    rg = np.array([1.0, -1.0, 0.0])
    by = np.array([-0.5, -0.5, 1.0])
    for nm, v, a in [("crgx", rg, 0.0), ("crgy", rg, np.pi / 2),
                     ("cbyx", by, 0.0), ("cbyy", by, np.pi / 2)]:
        W[IX[nm]] = v[:, None, None] * dgauss(5, 1.3, a)[None]
    return W


conv("c1", "image", NC1, k=5, pad=2, W=build_c1())


# ----------------------------------------------------------------------------- write
def save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "network.json"), "w") as f:
        json.dump({"layers": LAYERS}, f, indent=1)
    np.savez(os.path.join(OUT, "weights.npz"), **WEIGHTS)
    n = sum(int(v.size) for v in WEIGHTS.values())
    print(f"layers {len(LAYERS)}  weights {n:,}")

# ----------------------------------------------------------------------------- stage 2
# Soft semantic categories.  A category is a conjunction of linear conditions on
# colour / luminance / local smoothness.  Each condition becomes a hinge
# h = clip(k * (form - threshold)) in [0,1]; the conjunction is relu(sum h - (n-1)).
# All thresholds below are round numbers I chose from general knowledge of what
# sky, water, grass, fur, metal, paint and road look like.
HINGES = []          # (name, {channel: weight}, bias)
_HKEY = {}


def H(name, terms, thr, k):
    key = (tuple(sorted((c, round(w * k, 6)) for c, w in terms.items())), round(-thr * k, 6))
    if key in _HKEY:
        return _HKEY[key]
    terms = {c: w * k for c, w in terms.items()}
    HINGES.append((name, terms, -thr * k))
    _HKEY[key] = name
    return name


def _terms(ch):
    return dict(ch) if isinstance(ch, dict) else {ch: 1.0}


def sm(name, ch, thr, k):
    """hinge that the linear form `ch` is SMALLER than thr."""
    return H(name, {c: -w for c, w in _terms(ch).items()}, -thr, k)


def bg(name, ch, thr, k):
    return H(name, _terms(ch), thr, k)


# smoothness: all four fine gradients tiny
SMOOTH = [sm("sm_x+", "dxf", 0.06, 25), bg("sm_x-", "dxf", -0.06, 25),
          sm("sm_y+", "dyf", 0.06, 25), bg("sm_y-", "dyf", -0.06, 25)]

RG = {"Rs": 1.0, "Gs": -1.0}
BY = {"Bs": 1.0, "Rs": -0.5, "Gs": -0.5}
BR = {"Bs": 1.0, "Rs": -1.0}
BG_ = {"Bs": 1.0, "Gs": -1.0}
GR = {"Gs": 1.0, "Rs": -1.0}
GB = {"Gs": 1.0, "Bs": -1.0}
RB = {"Rs": 1.0, "Bs": -1.0}
LM = {"Ls": 1.0}

CATS = []            # (name, [hinge names], [(energy, weight)], threshold_offset)


def cat(name, hinges, extra=(), soft=0.0):
    CATS.append((name, list(hinges), list(extra), soft))


cat("skyblue", [bg("sb_by", BY, 0.035, 25), bg("sb_br", BR, 0.05, 20),
                bg("sb_L", LM, 0.55, 16)], [("efh", -4.0), ("efv", -4.0)])
cat("pale", [bg("pa_L", LM, 0.66, 25), sm("pa_rg+", RG, 0.07, 25), bg("pa_rg-", RG, -0.07, 25),
             sm("pa_by+", BY, 0.09, 20), bg("pa_by-", BY, -0.09, 20)],
    [("efh", -4.0), ("efv", -4.0)])
cat("teal", [bg("te_gr", GR, 0.025, 25), bg("te_br", BR, 0.035, 22),
             sm("te_Lhi", LM, 0.50, 20), bg("te_Llo", LM, 0.08, 10)])
cat("deepblue", [bg("db_br", BR, 0.07, 18), bg("db_bg", BG_, 0.00, 18),
                 sm("db_Lhi", LM, 0.45, 20), bg("db_Llo", LM, 0.06, 10)])
cat("green", [bg("gr_gr", GR, 0.025, 30), bg("gr_gb", GB, 0.035, 25)])
cat("greenbright", [bg("gr_gr", GR, 0.025, 30), bg("gr_gb", GB, 0.035, 25), bg("gb_L", LM, 0.38, 8)])
cat("greendark", [bg("gr_gr", GR, 0.025, 30), bg("gr_gb", GB, 0.035, 25), sm("gd_L", LM, 0.30, 8)])
cat("tan", [bg("ta_rg", RG, 0.015, 30), bg("ta_gb", GB, 0.035, 25),
            bg("ta_Llo", LM, 0.38, 8), sm("ta_Lhi", LM, 0.80, 8)])
cat("brown", [bg("br_rg", RG, 0.05, 22), bg("br_gb", GB, 0.025, 25),
              sm("br_Lhi", LM, 0.50, 8), bg("br_Llo", LM, 0.10, 8)])
cat("red", [bg("re_rg", RG, 0.13, 12), bg("re_rb", RB, 0.13, 12)])
cat("yellow", [bg("ye_rb", RB, 0.16, 12), bg("ye_gb", GB, 0.09, 14), sm("ye_rg", RG, 0.16, 12)])
cat("satblue", [bg("sblue_br", BR, 0.13, 12), bg("sblue_bg", BG_, 0.05, 16)])
cat("gray", [sm("gy_rg+", RG, 0.05, 40), bg("gy_rg-", RG, -0.05, 40),
             sm("gy_by+", BY, 0.055, 36), bg("gy_by-", BY, -0.055, 36),
             bg("gy_Llo", LM, 0.20, 16), sm("gy_Lhi", LM, 0.80, 16)])
cat("white", [bg("wh_L", LM, 0.68, 25), sm("wh_rg+", RG, 0.08, 25), bg("wh_rg-", RG, -0.08, 25),
              sm("wh_by+", BY, 0.10, 20), bg("wh_by-", BY, -0.10, 20)])
cat("dark", [sm("dk_L", LM, 0.18, 14)])
cat("pink", [bg("pi_rg", RG, 0.055, 20), bg("pi_gb", GB, -0.03, 20),
             bg("pi_rb", RB, 0.02, 20), bg("pi_L", LM, 0.45, 8)])
cat("smooth", [], [("efh", -4.0), ("efv", -4.0)], soft=1.0)

# energies: |oriented derivative| built from a positive and a negative hinge
ENER = [("efh", "dxf", 1.0), ("efv", "dyf", 1.0), ("efd1", "d1f", 1.0), ("efd2", "d2f", 1.0),
        ("ech", "dxc", 1.0), ("ecv", "dyc", 1.0), ("elap", "logf", 1.0),
        ("ecol", "crgx", 0.5)]
EPRE = []
for nm, ch, sc in ENER:
    EPRE.append((nm + "+", {ch: sc}, 0.0))
    EPRE.append((nm + "-", {ch: -sc}, 0.0))
# signed blob responses.  log() is (r^2-2s^2)/s^4 * G, i.e. a NEGATIVE centre, so
# +log responds to a dark blob and -log to a bright one.
BLOBS = [("dbf", "logf", 1.0), ("bbf", "logf", -1.0), ("dbc", "logc", 1.0), ("bbc", "logc", -1.0)]
for nm, ch, sc in BLOBS:
    EPRE.append((nm, {ch: sc}, 0.0))

# extra texture categories driven by the pooled fine edge energy
cat("rough", [], [("efh", 4.0), ("efv", 4.0)], soft=-0.80)
cat("vrough", [], [("efh", 4.0), ("efv", 4.0)], soft=-1.60)
cat("hedge", [], [("efv", 4.0)], soft=-0.50)
cat("vedge", [], [("efh", 4.0)], soft=-0.50)

# ---- assemble the hinge / category layers (all 1x1, at 16x16) ----
HNAMES = [h[0] for h in HINGES]
ENAMES = [e[0] for e in EPRE]
PRE = HINGES + EPRE
NP = len(PRE)
PIDX = {n: i for i, n in enumerate([p[0] for p in PRE])}

W2 = np.zeros((NP, NC1, 1, 1))
b2 = np.zeros(NP)
for i, (nm, terms, bias) in enumerate(PRE):
    for c, w in terms.items():
        W2[i, IX[c], 0, 0] = w
    b2[i] = bias
conv("c2", "c1", NP, k=1, stride=2, W=W2, b=b2)
relu("r2", "c2")
conv("c2b", "r2", NP, k=1, W=np.eye(NP)[:, :, None, None], b=-np.ones(NP))
relu("r2b", "c2b")
conv("c2c", "r2b", NP, k=1, W=-np.eye(NP)[:, :, None, None])
add("hc", "r2", "c2c")                      # hinges clipped to [0,1]

FEAT = [c[0] for c in CATS] + [e[0] for e in ENER] + [b[0] for b in BLOBS] + \
       ["dbc2", "bbc2", "dbc3", "bbc3"]
NF = len(FEAT)
FIDX = {n: i for i, n in enumerate(FEAT)}
W3 = np.zeros((NF, NP, 1, 1))
b3 = np.zeros(NF)
for nm, hinges, extra, soft in CATS:
    o = FIDX[nm]
    for h in hinges:
        W3[o, PIDX[h], 0, 0] = 1.0
    for e, a in extra:
        W3[o, PIDX[e + "+"], 0, 0] = a
        W3[o, PIDX[e + "-"], 0, 0] = a
    n = len(hinges)
    b3[o] = -((n - 1) if n else 0) + soft
for nm, ch, sc in ENER:
    o = FIDX[nm]
    W3[o, PIDX[nm + "+"], 0, 0] = 0.5
    W3[o, PIDX[nm + "-"], 0, 0] = 0.5
for nm, ch, sc in BLOBS:
    W3[FIDX[nm], PIDX[nm], 0, 0] = 1.0
conv("c3", "hc", NF, k=1, W=W3, b=b3)

# Proper multi-scale blob detectors: a 5x5 Laplacian of the already broadly blurred
# luminance, sampled at 16x16.  Effective scale sqrt(2^2 + (2s)^2) original pixels,
# i.e. ~3.1 px (wheel sized) and ~4.5 px (body / head sized).
BLOB2 = [("dbc2", 1.2, 1.0), ("bbc2", 1.2, -1.0), ("dbc3", 2.0, 1.0), ("bbc3", 2.0, -1.0)]
Wb = np.zeros((NF, NC1, 5, 5))
for nm, s, sg in BLOB2:
    Wb[FIDX[nm], IX["Lb"]] = sg * log(5, s)
conv("cblob", "c1", NF, k=5, stride=2, pad=2, W=Wb)
add("semraw", "c3", "cblob")
relu("sem", "semraw")                        # (NF, 16, 16) semantic maps

# ----------------------------------------------------------------------------- stage 3
# Coarse shape / layout detectors written from general knowledge of object structure.
pool("sem8", "sem", 2, 2)                     # (NF, 8, 8)
SH = ["wheels", "wheels_w", "horiz_w", "horiz_g", "legs", "popout"]
NS = len(SH)
SIDX = {n: i for i, n in enumerate(SH)}
Ws = np.zeros((NS, NF, 5, 5))
bs = np.zeros(NS)


def put(o, ch, dy, dx, v):
    Ws[SIDX[o], FIDX[ch], 2 + dy, 2 + dx] += v


# two dark round blobs side by side with a gap: a pair of wheels
for dx in (-1, 1):
    put("wheels", "dbc2", 0, dx, 1.0)
    put("wheels", "dark", 0, dx, 0.3)
put("wheels", "dbc2", 0, 0, -1.3)
put("wheels", "gray", 0, 0, 0.25)
for ch in ("gray", "red", "satblue", "white", "brown"):
    put("wheels", ch, -1, 0, 0.30)
bs[SIDX["wheels"]] = -0.35
for dx in (-2, 2):
    put("wheels_w", "dbc3", 0, dx, 1.0)
    put("wheels_w", "dark", 0, dx, 0.3)
put("wheels_w", "dbc3", 0, 0, -1.0)
for ch in ("gray", "red", "satblue", "white", "brown"):
    put("wheels_w", ch, -1, 0, 0.30)
bs[SIDX["wheels_w"]] = -0.35
# a horizon: smooth bright sky above, water below, horizontal edge between
for dx in range(-2, 3):
    for dy in (-2, -1):
        put("horiz_w", "skyblue", dy, dx, 0.16)
        put("horiz_w", "pale", dy, dx, 0.12)
    for dy in (1, 2):
        put("horiz_w", "teal", dy, dx, 0.16)
        put("horiz_w", "deepblue", dy, dx, 0.13)
        put("horiz_w", "gray", dy, dx, 0.05)
    put("horiz_w", "efv", 0, dx, 0.20)
bs[SIDX["horiz_w"]] = -0.55
for dx in range(-2, 3):
    for dy in (-2, -1):
        put("horiz_w", "skyblue", dy, dx, 0.0)
        put("horiz_g", "skyblue", dy, dx, 0.16)
        put("horiz_g", "pale", dy, dx, 0.12)
    for dy in (1, 2):
        put("horiz_g", "green", dy, dx, 0.14)
        put("horiz_g", "tan", dy, dx, 0.12)
        put("horiz_g", "gray", dy, dx, 0.06)
    put("horiz_g", "efv", 0, dx, 0.18)
bs[SIDX["horiz_g"]] = -0.55
# thin dark verticals hanging below a brown / grey body: legs
for dx in range(-2, 3):
    put("legs", "brown", -1, dx, 0.10)
    put("legs", "tan", -1, dx, 0.08)
    put("legs", "gray", -1, dx, 0.05)
put("legs", "dbc2", 1, 0, 0.7)
put("legs", "vedge", 1, 0, 0.5)
put("legs", "dark", 1, 0, 0.3)
bs[SIDX["legs"]] = -0.45
# a textured / colourful object standing out against a smooth surround
put("popout", "rough", 0, 0, 1.0)
put("popout", "satblue", 0, 0, 0.4)
put("popout", "red", 0, 0, 0.4)
for dy in (-2, 2):
    for dx in range(-2, 3):
        put("popout", "smooth", dy, dx, 0.10)
for dx in (-2, 2):
    for dy in (-1, 0, 1):
        put("popout", "smooth", dy, dx, 0.10)
bs[SIDX["popout"]] = -0.60
conv("cshape", "sem8", NS, k=5, pad=2, W=Ws, b=bs)
relu("shape", "cshape")                       # (NS, 8, 8)

# ----------------------------------------------------------------------------- stage 4
# Pool the semantic maps into a 4x4 layout grid plus whole-image summaries.
pool("p4", "sem", 4, 4)                       # (NF,4,4) region fractions
pool("p1", "sem", 16, 16)                     # (NF,1,1) whole-image fractions
pool("m1", "sem", 16, 16, kind="max")         # (NF,1,1) strongest presence anywhere
pool("s4", "shape", 2, 2, kind="max")         # (NS,4,4) strongest shape evidence per cell
pool("s1", "shape", 8, 8, kind="max")         # (NS,1,1) strongest anywhere
pool("gcol", "c1", 32, 32)                    # (NC1,1,1) mean colour / luminance

PATHS = [("p4", "p4", NF, 4), ("p1", "p1", NF, 1), ("m1", "m1", NF, 1),
         ("s4", "s4", NS, 4), ("s1", "s1", NS, 1), ("gc", "gcol", NC1, 1)]
FLAT = {}          # feature name -> (path, index)
for pname, src, nch, g in PATHS:
    reshape(pname + "_f", src, [nch * g * g])
    chans = {"p4": FEAT, "p1": FEAT, "m1": FEAT, "s4": SH, "s1": SH, "gc": C1}[pname]
    for ci, cn in enumerate(chans):
        for r in range(g):
            for c in range(g):
                if pname in ("p4", "s4"):
                    nm = f"{cn}@{r}{c}"
                elif pname == "m1":
                    nm = "mx:" + cn
                elif pname == "gc":
                    nm = "av:" + cn
                else:
                    nm = cn
                FLAT[nm] = (pname, ci * g * g + r * g + c)
NF_TOT = len(FLAT)

CLS = "airplane automobile bird cat deer dog frog horse ship truck".split()
LIN = {k: {} for k in range(10)}
HID = []           # (name, {feature: weight}, bias)


def expand(terms):
    """pattern -> per-cell weights, normalised so a pattern's total contribution is
    w * (mean response over the region it names), i.e. every feature is in [0,1]."""
    d = {}
    for pat, w in terms:
        fs = resolve(pat)
        for f in fs:
            d[f] = d.get(f, 0.0) + w / len(fs)
    return d


def lw(k, pat, w):
    for f, v in expand([(pat, w)]).items():
        LIN[k][f] = LIN[k].get(f, 0.0) + v


def resolve(pat):
    if pat in FLAT:
        return [pat]
    base, _, where = pat.partition("@")
    pre = ""
    if base.startswith("mx:") or base.startswith("av:"):
        pre, base = base.split(":", 1)
        pre += ":"
    out = []
    rows = {"top": [0], "bot": [3], "mid": [1, 2], "all": [0, 1, 2, 3],
            "up": [0, 1], "lo": [2, 3]}.get(where)
    cols = {"left": [0], "right": [3], "cen": [1, 2], "all": [0, 1, 2, 3],
            "mid": [1, 2]}.get(where)
    if where in ("cen", "center"):
        rows, cols = [1, 2], [1, 2]
    if rows is not None and cols is None:
        cols = [0, 1, 2, 3]
    if cols is not None and rows is None:
        rows = [0, 1, 2, 3]
    if rows is None or cols is None:
        if len(where) == 2 and where.isdigit():
            rows, cols = [int(where[0])], [int(where[1])]
        else:
            raise KeyError(pat)
    src = FEAT + SH
    if base not in src:
        raise KeyError(pat)
    for r in rows:
        for c in cols:
            nm = f"{pre}{base}@{r}{c}"
            if nm in FLAT:
                out.append(nm)
    if not out:
        raise KeyError(pat)
    return out

# ----------------------------------------------------------------------------- stage 5
# Hand-written rule table.  Each entry is (feature pattern, weight) chosen from what
# I know about where these objects appear and what they are made of.
RULES = {
0: [  # airplane: bright smooth sky (blue, overcast or sunset), silhouette in the middle
    ("skyblue@top", 3.0), ("skyblue@all", 1.5), ("pale@top", 2.0), ("pale@all", 1.0),
    ("smooth@top", 2.0), ("smooth@all", 1.5), ("horiz_g", 1.2),
    ("gray@cen", 0.8), ("white@cen", 0.7), ("dark@cen", 0.5), ("av:Ls", 0.6),
    ("tan@top", 0.6), ("brown@top", 0.4),
    ("teal@bot", -3.0), ("teal@lo", -2.0), ("deepblue@bot", -2.0),
    ("green@all", -2.0), ("greenbright@bot", -1.5), ("rough@cen", -1.5),
    ("vrough@all", -1.5), ("elap@all", -3.0), ("wheels@all", -2.5), ("wheels_w@all", -2.0),
    ("legs@all", -1.2), ("pink@all", -1.2), ("red@cen", -1.0), ("satblue@cen", -0.8),
    ("horiz_w", -1.5), ("brown@bot", -1.0), ("tan@bot", -0.8)],
8: [  # ship: darker blue / teal water in the lower half, horizon, pale hull amidships
    ("teal@bot", 3.5), ("teal@lo", 2.5), ("deepblue@bot", 2.0), ("deepblue@lo", 1.5),
    ("horiz_w", 3.0), ("skyblue@top", 1.0), ("pale@top", 0.8),
    ("gray@cen", 1.0), ("white@cen", 1.2), ("dark@cen", 0.6),
    ("hedge@bot", 1.0), ("smooth@bot", 0.6), ("red@cen", 0.4),
    ("green@all", -2.0), ("greenbright@all", -1.2), ("brown@cen", -1.0),
    ("wheels@all", -2.5), ("wheels_w@all", -2.0), ("legs@all", -1.2),
    ("rough@cen", -1.0), ("elap@cen", -2.0), ("pink@all", -1.0), ("tan@bot", -0.8)],
1: [  # automobile: a pair of wheels low in the frame, saturated paint, road beneath
    ("wheels@bot", 3.5), ("wheels@lo", 2.5), ("wheels_w@bot", 2.0),
    ("red@cen", 2.2), ("red@all", 1.2), ("satblue@cen", 2.0), ("satblue@all", 1.0),
    ("yellow@cen", 1.2), ("gray@bot", 1.2), ("smooth@bot", 0.7),
    ("dark@bot", 0.8), ("dbc2@bot", 1.0), ("white@cen", 0.5),
    ("teal@bot", -3.0), ("deepblue@bot", -2.0), ("skyblue@top", -0.6),
    ("green@all", -1.0), ("brown@cen", -1.2), ("tan@cen", -1.0), ("pink@all", -1.2),
    ("legs@all", -1.5), ("rough@cen", -1.5), ("vrough@all", -1.2), ("elap@all", -3.0),
    ("white@all", -0.8), ("pale@all", -0.8), ("wheels@top", -1.5)],
9: [  # truck: wide wheelbase and a tall white / painted box above it
    ("wheels_w@bot", 3.0), ("wheels_w@lo", 2.0), ("wheels@bot", 1.5),
    ("white@up", 2.2), ("white@cen", 2.0), ("white@all", 1.2), ("gray@up", 0.8),
    ("red@cen", 1.2), ("satblue@cen", 1.0), ("vedge@all", 1.0), ("efh@all", 1.5),
    ("dark@bot", 0.8), ("pale@up", 0.6),
    ("teal@bot", -3.0), ("deepblue@bot", -2.0), ("green@all", -1.2),
    ("brown@cen", -1.2), ("tan@cen", -1.0), ("pink@all", -1.2), ("legs@all", -1.5),
    ("rough@cen", -1.2), ("elap@all", -2.5), ("skyblue@top", -0.5),
    ("smooth@top", -0.8), ("wheels@top", -1.5)],
2: [  # bird: a small brown / grey / yellow body against blurred vegetation
    ("green@all", 1.8), ("greenbright@all", 1.2), ("greendark@all", 0.8),
    ("brown@cen", 1.6), ("tan@cen", 1.2), ("gray@cen", 0.9),
    ("yellow@cen", 1.6), ("yellow@all", 1.0), ("rough@cen", 1.0), ("elap@cen", 2.5),
    ("dark@cen", 0.5), ("white@cen", 0.4),
    ("skyblue@all", -1.2), ("pale@all", -1.2), ("teal@bot", -2.5), ("deepblue@bot", -1.5),
    ("wheels@all", -2.5), ("wheels_w@all", -2.0), ("legs@all", -1.0), ("red@cen", -0.6),
    ("satblue@cen", -0.8), ("white@all", -0.8), ("smooth@all", -0.8), ("pink@all", -0.6),
    ("horiz_w", -1.5), ("tan@bot", -0.6)],
6: [  # frog: mottled green / brown skin filling the frame, no sky and no metal
    ("green@cen", 2.0), ("green@all", 1.5), ("greenbright@cen", 1.5),
    ("brown@cen", 1.6), ("brown@bot", 1.0), ("tan@cen", 1.0),
    ("rough@all", 1.8), ("vrough@all", 1.8), ("elap@all", 2.5), ("greendark@all", 0.6),
    ("skyblue@all", -3.0), ("pale@all", -2.5), ("white@all", -2.0), ("gray@all", -1.0),
    ("teal@bot", -2.0), ("deepblue@all", -2.0), ("wheels@all", -2.5), ("wheels_w@all", -2.0),
    ("red@cen", -1.0), ("satblue@cen", -1.5), ("pink@all", -1.0), ("smooth@all", -1.8),
    ("horiz_w", -2.0), ("legs@all", -0.8), ("brown@top", -0.8), ("tan@top", -0.8)],
3: [  # cat: close-up low-saturation fur, indoors, grey / tabby / white / black
    ("gray@cen", 1.8), ("gray@all", 1.2), ("rough@cen", 1.5), ("rough@all", 1.0),
    ("elap@all", 2.0), ("efh@all", 1.0), ("efv@all", 1.0), ("vrough@cen", 0.8),
    ("white@cen", 1.2), ("dark@cen", 1.2), ("brown@cen", 0.6), ("tan@cen", 0.4),
    ("pink@cen", 0.3),
    ("green@all", -2.0), ("greenbright@all", -1.5), ("skyblue@all", -2.5),
    ("pale@all", -1.5), ("teal@bot", -3.0), ("deepblue@all", -2.0),
    ("wheels@all", -2.5), ("wheels_w@all", -2.0), ("legs@all", -1.2), ("red@cen", -1.0),
    ("satblue@cen", -1.0), ("yellow@cen", -0.8), ("smooth@all", -1.0),
    ("horiz_w", -2.0), ("horiz_g", -1.0), ("tan@bot", -0.6)],
5: [  # dog: warm brown / tan fur, pink tongue, white coat, grass or indoors
    ("brown@cen", 1.8), ("brown@all", 1.2), ("tan@cen", 1.4), ("tan@all", 0.8),
    ("pink@cen", 2.0), ("pink@all", 1.5), ("white@cen", 1.2),
    ("rough@cen", 1.4), ("rough@all", 1.0), ("elap@all", 2.0), ("green@all", 0.6),
    ("gray@cen", 0.6), ("dark@cen", 0.7),
    ("skyblue@all", -2.5), ("pale@all", -1.5), ("teal@bot", -3.0), ("deepblue@all", -2.0),
    ("wheels@all", -2.5), ("wheels_w@all", -2.0), ("legs@all", -1.0), ("red@cen", -0.8),
    ("satblue@cen", -1.0), ("yellow@cen", -0.6), ("smooth@all", -1.0),
    ("horiz_w", -2.0), ("gray@all", -0.4)],
4: [  # deer: tan / brown body on thin legs, forest shade or grass
    ("green@all", 2.0), ("greendark@all", 1.4), ("greenbright@all", 0.8),
    ("brown@cen", 2.0), ("tan@cen", 1.6), ("legs@lo", 2.2), ("legs@all", 1.2),
    ("horiz_g", 1.0), ("rough@cen", 0.6), ("tan@bot", 0.5),
    ("skyblue@all", -1.2), ("pale@all", -1.5), ("teal@bot", -2.5), ("deepblue@all", -1.5),
    ("wheels@all", -2.5), ("wheels_w@all", -2.0), ("white@cen", -1.0), ("gray@cen", -0.6),
    ("red@cen", -1.0), ("satblue@cen", -1.0), ("pink@all", -0.6), ("smooth@all", -0.6),
    ("vrough@all", -0.8), ("dark@cen", -0.8), ("tan@bot", -0.4)],
7: [  # horse: a big brown / black / white body on legs, sand arena or grass field
    ("brown@cen", 2.0), ("brown@all", 1.0), ("tan@bot", 2.0), ("tan@lo", 1.2),
    ("legs@lo", 2.5), ("legs@all", 1.5), ("dark@cen", 1.2), ("white@cen", 1.0),
    ("green@all", 0.8), ("greenbright@all", 0.6), ("horiz_g", 0.8), ("gray@all", 0.4),
    ("pink@cen", 0.4), ("rough@cen", 0.5),
    ("skyblue@all", -1.0), ("pale@all", -1.2), ("teal@bot", -2.5), ("deepblue@all", -1.5),
    ("wheels@all", -2.5), ("wheels_w@all", -2.0), ("red@cen", -0.8), ("satblue@cen", -0.8),
    ("smooth@all", -0.6), ("greendark@all", -0.8), ("green@cen", -0.4)],
}
# Put every class on the same footing: scale its positive evidence to a common total
# and its negative evidence to a common total, so no class wins by having more terms.
POS_TOTAL, NEG_TOTAL = 14.0, 14.0
for k, rows in RULES.items():
    pos = sum(w for _, w in rows if w > 0)
    neg = -sum(w for _, w in rows if w < 0)
    for pat, w in rows:
        lw(k, pat, w * (POS_TOTAL / pos if w > 0 else NEG_TOTAL / neg))

# ---- nonlinear scene / object concepts: thresholded, then clipped to [0,1] ----
CONCEPTS = [
    ("water",   [("teal@bot", 2.5), ("teal@lo", 1.5), ("deepblue@bot", 2.0),
                 ("horiz_w", 2.0)], -1.0),
    ("sky",     [("skyblue@top", 2.5), ("pale@top", 2.0), ("smooth@top", 1.5),
                 ("skyblue@all", 1.0)], -1.5),
    ("veg",     [("green@all", 2.5), ("greenbright@all", 1.0)], -0.8),
    ("vehicle", [("wheels@bot", 3.0), ("wheels_w@bot", 2.5), ("red@cen", 1.5),
                 ("satblue@cen", 1.5), ("gray@bot", 1.0)], -1.0),
    ("fur",     [("rough@cen", 1.5), ("vrough@all", 1.0), ("gray@cen", 1.2),
                 ("brown@cen", 1.0), ("tan@cen", 0.8), ("white@cen", 0.6), ("dark@cen", 0.6),
                 ("green@all", -2.0), ("skyblue@all", -2.0), ("teal@bot", -2.0),
                 ("pale@all", -1.0)], -1.0),
    ("quad",    [("legs@lo", 2.5), ("brown@cen", 1.5), ("tan@cen", 1.0)], -0.8),
    ("whitebox", [("white@up", 2.5), ("white@cen", 1.5), ("vedge@all", 1.0)], -1.0),
    ("darkobj", [("dark@cen", 2.0), ("dark@all", 1.0)], -0.5),
    ("redobj",  [("red@cen", 2.5)], -0.4),
    ("blueobj", [("satblue@cen", 2.5)], -0.4),
    ("pinkobj", [("pink@cen", 3.0)], -0.3),
    ("smoothbg", [("smooth@all", 2.5)], -0.8),
    ("mottled", [("vrough@all", 2.5), ("elap@all", 2.5)], -0.7),
    ("yellowobj", [("yellow@cen", 2.5)], -0.4),
    ("browncen", [("brown@cen", 2.0), ("tan@cen", 1.5)], -0.6),
    ("greencen", [("green@cen", 2.0), ("greenbright@cen", 1.0)], -0.6),
]
NH = len(CONCEPTS)
for nm, terms, bias in CONCEPTS:
    HID.append((nm, expand(terms), bias))

CRULES = {
    0: [("sky", 2.5), ("smoothbg", 1.2), ("water", -2.5), ("veg", -1.5), ("vehicle", -1.5),
        ("fur", -1.0), ("quad", -1.0)],
    1: [("vehicle", 2.0), ("redobj", 1.5), ("blueobj", 1.5), ("water", -2.0), ("veg", -1.0),
        ("whitebox", -0.8), ("quad", -1.0)],
    2: [("veg", 1.2), ("yellowobj", 1.5), ("browncen", 1.0), ("water", -1.5),
        ("vehicle", -1.5), ("quad", -1.0), ("sky", -0.8)],
    3: [("fur", 1.5), ("browncen", 0.5), ("veg", -1.5), ("water", -1.5), ("vehicle", -1.5),
        ("sky", -1.0), ("quad", -0.8), ("darkobj", 0.5)],
    4: [("quad", 1.5), ("veg", 1.2), ("browncen", 1.0), ("water", -1.5), ("vehicle", -1.5),
        ("sky", -0.8), ("mottled", -0.8)],
    5: [("fur", 1.5), ("pinkobj", 1.5), ("browncen", 0.8), ("water", -1.5),
        ("vehicle", -1.5), ("sky", -1.0), ("quad", -0.5)],
    6: [("mottled", 1.5), ("veg", 1.2), ("greencen", 1.0), ("water", -2.0),
        ("vehicle", -1.5), ("sky", -1.5), ("quad", -1.0), ("whitebox", -1.0)],
    7: [("quad", 1.8), ("veg", 0.8), ("browncen", 1.0), ("darkobj", 0.6),
        ("water", -1.5), ("vehicle", -1.5), ("sky", -0.6)],
    8: [("water", 3.0), ("sky", 0.8), ("veg", -2.0), ("vehicle", -1.5), ("fur", -1.0),
        ("quad", -1.0)],
    9: [("vehicle", 2.0), ("whitebox", 2.0), ("redobj", 0.8), ("blueobj", 0.8),
        ("water", -2.0), ("veg", -1.0), ("quad", -1.0), ("fur", -0.8)],
}
CPOS, CNEG = 6.0, 6.0
for k, rows in CRULES.items():
    pos = sum(w for _, w in rows if w > 0)
    neg = -sum(w for _, w in rows if w < 0)
    CRULES[k] = [(nm, w * (CPOS / pos if w > 0 else CNEG / neg)) for nm, w in rows]


# ---- materialise the dense layers: one per pooled path, summed ----
LINW = {p: np.zeros((10, n * g * g)) for p, _, n, g in PATHS}
LINB = np.zeros(10)
for k in range(10):
    for f, w in LIN[k].items():
        p, i = FLAT[f]
        LINW[p][k, i] += w
HIDW = {p: np.zeros((NH, n * g * g)) for p, _, n, g in PATHS}
HIDB = np.zeros(NH)
for j, (nm, d, bias) in enumerate(HID):
    HIDB[j] = bias
    for f, w in d.items():
        p, i = FLAT[f]
        HIDW[p][j, i] += w

lin_parts, hid_parts = [], []
for p, src, n, g in PATHS:
    lin_parts.append(dense("lin_" + p, p + "_f", 10, W=LINW[p], b=0.0))
    hid_parts.append(dense("hid_" + p, p + "_f", NH, W=HIDW[p], b=0.0))
add("linsum", *lin_parts)
dense("linb", "linsum", 10, W=np.eye(10), b=LINB)
add("hidsum", *hid_parts)
dense("hidb", "hidsum", NH, W=np.eye(NH), b=HIDB)
relu("concepts", "hidb")
CW = np.zeros((10, NH))
for k, rows in CRULES.items():
    for nm, w in rows:
        CW[k, [c[0] for c in CONCEPTS].index(nm)] += w
dense("ccl1", "concepts", NH, W=np.eye(NH), b=-np.ones(NH))
relu("ccl2", "ccl1")
dense("ccl3", "ccl2", NH, W=-np.eye(NH))
add("gate", "concepts", "ccl3")               # concept gates clipped to [0,1]
dense("nl", "gate", 10, W=CW)
add("scores", "linb", "nl")


save()
