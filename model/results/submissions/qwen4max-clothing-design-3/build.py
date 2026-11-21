"""Hand-designed clothing classifier.  Every number here comes from geometry of a
28x28 centred product photo and from general knowledge of garment / shoe / bag
structure.  No training data is read, no statistic of any image is used.

Run:  python3 build.py OUT_DIR   -> writes OUT_DIR/network.json + OUT_DIR/weights.npz
"""
import json
import sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ---------------------------------------------------------------- network plumbing
LAYERS = []
W = {}


def layer(name, op, inputs, **settings):
    L = {"name": name, "op": op, "inputs": inputs}
    L.update(settings)
    LAYERS.append(L)
    return name


def conv(name, src, out, k, stride=1, pad=0, w=None, b=None):
    layer(name, "conv", [src], out=out, kernel=[k, k], stride=stride, padding=pad)
    W[name + ".W"] = np.asarray(w, np.float32)
    W[name + ".b"] = np.asarray(b if b is not None else np.zeros(out), np.float32)
    return name


def dense(name, src, out, w, b):
    layer(name, "dense", [src], out=out)
    W[name + ".W"] = np.asarray(w, np.float32)
    W[name + ".b"] = np.asarray(b, np.float32)
    return name


def ones_conv(name, src, out, bias):
    """1x1 conv, every output = input + bias (used to make threshold ramps)."""
    w = np.ones((out, 1, 1, 1), np.float32)
    return conv(name, src, out, 1, w=w, b=bias)


def place(name, src, n_in, n_out, lo):
    """1x1 conv that copies n_in channels into slots [lo, lo+n_in) of n_out, rest 0.
    Lets separate branches be concatenated with `add`."""
    w = np.zeros((n_out, n_in, 1, 1), np.float32)
    for i in range(n_in):
        w[lo + i, i, 0, 0] = 1.0
    return conv(name, src, n_out, 1, w=w)


# ---------------------------------------------------------------- kernel helpers
def kk():
    """coordinate grids for an 11x11 kernel, centre at (0,0)."""
    ax = np.arange(-5, 6, dtype=np.float64)
    return np.meshgrid(ax, ax, indexing="ij")


DY, DX = kk()
R = np.sqrt(DY ** 2 + DX ** 2)


def norm_pm(pos, neg):
    """weights with total +1 on `pos` mask and -1 on `neg` mask -> (mean pos - mean neg)."""
    w = np.zeros((11, 11), np.float64)
    w[pos] = 1.0 / max(pos.sum(), 1)
    w[neg] = -1.0 / max(neg.sum(), 1)
    return w


def mean_of(mask):
    w = np.zeros((11, 11), np.float64)
    w[mask] = 1.0 / max(mask.sum(), 1)
    return w


def rect(y0, y1, x0, x1):
    return (DY >= y0) & (DY <= y1) & (DX >= x0) & (DX <= x1)


# ---------------------------------------------------------------- front end
# Background of these photos is black, garments are light.  Three soft steps turn
# grey level into a brightness-independent silhouette plus two "how bright" maps.
# softstep(x;t,s) = (relu(x-t) - relu(x-t-s))/s  in [0,1].
TH = [(-0.02, -0.08), (-0.12, -0.28), (-0.35, -0.60)]   # (t, t+s) as negative biases
rmp = ones_conv("rmp", "image", 6, [t for pair in TH for t in pair])
layer("rmpR", "relu", [rmp])
mw = np.zeros((3, 6, 1, 1), np.float64)
for i, (a, b_) in enumerate(TH):
    s = a - b_                      # ramp width, positive
    assert s > 0
    mw[i, 2 * i, 0, 0] = 1.0 / s
    mw[i, 2 * i + 1, 0, 0] = -1.0 / s
masks = conv("masks", "rmpR", 3, 1, w=mw)          # A=silhouette B=mid C=bright

nrm = layer("nrm", "norm", ["image"])
base = layer("base", "add", [
    place("m5", masks, 3, 5, 0),
    place("i5", "image", 1, 5, 3),
    place("n5", nrm, 1, 5, 4),
])                                                 # (5,28,28) A,B,C,gray,normed
SB = conv("SB", base, 3, 1, w=np.array([[[[1]], [[0]], [[0]], [[0]], [[0]]],
                                        [[[0]], [[1]], [[0]], [[0]], [[0]]],
                                        [[[0]], [[0]], [[0]], [[0]], [[1]]]], np.float32))

# ------------------------------------------------- silhouette structure filters
# All of these are formulas over an 11x11 window: arms, rings, bars, differences of
# means.  Nothing is traced from a picture.
A = []          # (kernel(2,11,11), bias, comment)
CH = {}         # name -> index in the merged stage-1 tensor


def addA(name, k0, bias=0.0, k1=None, k2=None):
    w = np.zeros((3, 11, 11), np.float64)
    w[0] = k0
    if k1 is not None:
        w[1] = k1
    if k2 is not None:
        w[2] = k2
    CH[name] = len(A)
    A.append((w, bias))


c3 = rect(-1, 1, -1, 1)                       # 3x3 centre
addA("S", mean_of(rect(0, 0, 0, 0)))
addA("blur3", mean_of(c3))
addA("blur5", mean_of(R <= 2.5))
addA("blur11", mean_of(R <= 5.0))
addA("erA", mean_of(c3), -0.80)               # ramp for interior/erosion
addA("erB", mean_of(c3), -0.95)

# narrow dark vertical stripe (sleeve/body slit, coat opening, placket)
sideN = rect(-4, 4, -2, 2) & ~rect(-4, 4, 0, 0)
addA("vlineN", norm_pm(sideN, rect(-4, 4, 0, 0)))
addA("vlineNn", -norm_pm(sideN, rect(-4, 4, 0, 0)))
# wide dark vertical gap (trouser legs)
addA("vlineW", norm_pm(rect(-5, 5, -4, 4) & ~rect(-5, 5, -1, 1), rect(-5, 5, -1, 1)))
addA("vlineWn", -norm_pm(rect(-5, 5, -4, 4) & ~rect(-5, 5, -1, 1), rect(-5, 5, -1, 1)))
addA("hlineN", norm_pm(rect(-2, 2, -4, 4) & ~rect(0, 0, -4, 4), rect(0, 0, -4, 4)))
addA("hlineNn", -norm_pm(rect(-2, 2, -4, 4) & ~rect(0, 0, -4, 4), rect(0, 0, -4, 4)))
addA("hlineW", norm_pm(rect(-4, 4, -5, 5) & ~rect(-1, 1, -5, 5), rect(-1, 1, -5, 5)))
addA("hlineWn", -norm_pm(rect(-4, 4, -5, 5) & ~rect(-1, 1, -5, 5), rect(-1, 1, -5, 5)))

# four arms around a dark centre: an enclosed hole lights all four, a leg gap does not
addA("holeU", norm_pm(rect(-5, -3, -1, 1), c3))
addA("holeD", norm_pm(rect(3, 5, -1, 1), c3))
addA("holeL", norm_pm(rect(-1, 1, -5, -3), c3))
addA("holeR", norm_pm(rect(-1, 1, 3, 5), c3))
addA("ring", norm_pm((R >= 3) & (R <= 5), R <= 1.5))
addA("ringN", -norm_pm((R >= 3) & (R <= 5), R <= 1.5))

# edge orientation (mean of one side minus the other, over a 3px span)
addA("gxP", 0.5 * (mean_of(rect(-1, 1, 1, 2)) - mean_of(rect(-1, 1, -2, -1))))
addA("gxN", -0.5 * (mean_of(rect(-1, 1, 1, 2)) - mean_of(rect(-1, 1, -2, -1))))
addA("gyP", 0.5 * (mean_of(rect(1, 2, -1, 1)) - mean_of(rect(-2, -1, -1, 1))))
addA("gyN", -0.5 * (mean_of(rect(1, 2, -1, 1)) - mean_of(rect(-2, -1, -1, 1))))
d1 = (DY == DX) & (R > 0) & (R <= 4.3)
d2 = (DY == -DX) & (R > 0) & (R <= 4.3)
addA("dgP", 0.5 * (mean_of(d1) - mean_of(d2)))
addA("dgN", -0.5 * (mean_of(d1) - mean_of(d2)))

# local vertical / horizontal mass asymmetry
addA("updn", norm_pm(rect(-5, -2, -2, 2), rect(2, 5, -2, 2)))
addA("updnN", -norm_pm(rect(-5, -2, -2, 2), rect(2, 5, -2, 2)))
addA("lfrt", norm_pm(rect(-2, 2, -5, -2), rect(-2, 2, 2, 5)))
addA("lfrtN", -norm_pm(rect(-2, 2, -5, -2), rect(-2, 2, 2, 5)))

# plaid / check: dark lines at 0 and +-4 with bright between (period ~4px)
gv = norm_pm(rect(-3, 3, -2, 2) & ~rect(-3, 3, 0, 0), rect(-3, 3, 0, 0))
gv2 = np.zeros((11, 11))
gv2[:, 1] = -1.0 / 14; gv2[:, 9] = -1.0 / 14
gv2[:, 3] = 1.0 / 14; gv2[:, 7] = 1.0 / 14
gv2[0:2, :] = 0; gv2[9:11, :] = 0
addA("gridV", np.zeros((11, 11)), k1=0.5 * (gv + gv2))
addA("gridH", np.zeros((11, 11)), k1=0.5 * (gv.T + gv2.T))
addA("vlineM", np.zeros((11, 11)), k1=norm_pm(sideN, rect(-4, 4, 0, 0)))
addA("hlineM", np.zeros((11, 11)), k1=norm_pm(rect(-2, 2, -4, 4) & ~rect(0, 0, -4, 4), rect(0, 0, -4, 4)))
addA("Smid", np.zeros((11, 11)), k1=mean_of(rect(0, 0, 0, 0)))
# A fold or placket is a shading line that persists over many rows; 9 rows of context
# separates it from a transient texture edge.  0.5 scales unit-spread normed grey.
addA("vlineG11", np.zeros((11, 11)), k2=0.5 * norm_pm(sideN, rect(-4, 4, 0, 0)))
addA("vlineG11n", np.zeros((11, 11)), k2=-0.5 * norm_pm(sideN, rect(-4, 4, 0, 0)))
addA("hlineG11", np.zeros((11, 11)), k2=0.5 * norm_pm(rect(-2, 2, -4, 4) & ~rect(0, 0, -4, 4), rect(0, 0, -4, 4)))

NA = len(A)
wa = np.stack([a[0] for a in A]).astype(np.float32)
ba = np.array([a[1] for a in A], np.float32)
f1a = conv("f1a", SB, NA, 11, pad=5, w=wa, b=ba)
layer("f1aR", "relu", [f1a])

# ------------------------------------------------- brightness / texture filters
Bq = []
BCH = {}


def addB(name, coeffs, bias=0.0):
    """coeffs: dict channel->5x5 kernel.  channels of base: 0=A 1=B 2=C 3=grey 4=normed"""
    w = np.zeros((5, 5, 5), np.float64)
    for c, k in coeffs.items():
        w[c] = k
    BCH[name] = len(Bq)
    Bq.append((w, bias))


ax5 = np.arange(-2, 3, dtype=np.float64)
dy5, dx5 = np.meshgrid(ax5, ax5, indexing="ij")
ctr5 = (dy5 == 0) & (dx5 == 0)
hp = np.full((5, 5), -1.0 / 24.0); hp[ctr5] = 1.0
id5 = np.zeros((5, 5)); id5[ctr5] = 1.0
m3 = np.full((5, 5), 0.0); m3[(np.abs(dy5) <= 1) & (np.abs(dx5) <= 1)] = 1.0 / 9

addB("gray", {3: id5})
addB("nrmP", {4: id5})
addB("nrmN", {4: -id5})
addB("Sbri", {2: id5})
addB("texP", {4: hp})
addB("texN", {4: -hp})
addB("texbP", {1: hp})
addB("texbN", {1: -hp})
addB("texgP", {3: hp})
addB("texgN", {3: -hp})
addB("blur3g", {3: m3})
# shading lines: a coat opening / shirt placket is a fold in the cloth, not a gap in
# the silhouette, so it only shows in the contrast-normalised grey.  5x5 version of
# the narrow dark-line filter, scaled by 0.5 because normed grey has unit spread.
sgN = np.zeros((5, 5))
sgN[:, 2] = -1.0 / 5
for c in (0, 1, 3, 4):
    sgN[:, c] = 1.0 / 20
addB("vlineG", {4: 0.5 * sgN})
addB("vlineGn", {4: -0.5 * sgN})
addB("hlineG", {4: 0.5 * sgN.T})
addB("hlineGn", {4: -0.5 * sgN.T})
addB("vlineB5", {1: sgN})
NB = len(Bq)
wb = np.stack([q[0] for q in Bq]).astype(np.float32)
bb = np.array([q[1] for q in Bq], np.float32)
f1b = conv("f1b", base, NB, 5, pad=2, w=wb, b=bb)
layer("f1bR", "relu", [f1b])

N1 = NA + NB
f1 = layer("f1", "add", [place("pa", f1a + "R", NA, N1, 0),
                         place("pb", f1b + "R", NB, N1, NA)])
for k, v in BCH.items():
    CH[k] = NA + v

# ------------------------------------------------- stage 2: interior / thinness
# erode = softstep(local 3x3 mean of silhouette; 0.80, 0.15): 1 deep inside a solid
# region, 0 on a strap or a 1-2px wide bar.  thin = silhouette - erode.
FCH = {}
F = []          # (row over N1 channels, bias)


def addF(name, terms, bias=0.0):
    r = np.zeros(N1, np.float64)
    for nm, w in terms:
        r[CH[nm]] += w
    FCH[name] = len(F)
    F.append((r, bias))


for nm in list(CH):
    if nm in ("erA", "erB"):
        continue
    addF(nm, [(nm, 1.0)])
addF("erode", [("erA", 1 / 0.15), ("erB", -1 / 0.15)])
addF("thin", [("S", 1.0), ("erA", -1 / 0.15), ("erB", 1 / 0.15)])
addF("edge", [("gxP", 1.0), ("gxN", 1.0), ("gyP", 1.0), ("gyN", 1.0)])
addF("tex", [("texP", 1.0), ("texN", 1.0)])
addF("texb", [("texbP", 1.0), ("texbN", 1.0)])
addF("texg", [("texgP", 1.0), ("texgN", 1.0)])
addF("vlineGa", [("vlineG", 1.0), ("vlineGn", 1.0)])
NC = len(F)
f2 = conv("f2", f1, NC, 1, w=np.stack([q[0] for q in F]).reshape(NC, N1, 1, 1).astype(np.float32),
          b=np.array([q[1] for q in F], np.float32))
layer("f2R", "relu", [f2])

# ------------------------------------------------- pooled grid (mean + max)
gA = layer("gA", "avgpool", ["f2R"], size=[2, 2], stride=[2, 2])
gM = layer("gM", "maxpool", ["f2R"], size=[2, 2], stride=[2, 2])
grid = layer("grid", "add", [place("qa", gA, NC, 2 * NC, 0),
                             place("qm", gM, NC, 2 * NC, NC)])
flat = layer("flat", "reshape", [grid], shape=[2 * NC * 14 * 14])
NIN = 2 * NC * 14 * 14

# ------------------------------------------------- primitive region features
# A primitive is the mean of one feature channel over a rectangle of the photo.
# Rectangles are given in photo pixels on even boundaries; each grid cell is 2x2.
P = []
PN = {}
PB = []


def prim(name, terms, bias=0.0, neg=False):
    """terms: list of (channel, pool, y0, y1, x0, x1, weight).  weight multiplies the
    REGION MEAN, so numbers stay in the channel's own units."""
    r = np.zeros(NIN, np.float64)
    for (ch, pool, y0, y1, x0, x1, w) in terms:
        ci = FCH[ch] + (NC if pool == "max" else 0)
        gy0, gy1 = y0 // 2, (y1 + 1) // 2
        gx0, gx1 = x0 // 2, (x1 + 1) // 2
        n = (gy1 - gy0) * (gx1 - gx0)
        assert n > 0, (name, y0, y1, x0, x1)
        v = w / n * (-1.0 if neg else 1.0)
        for yy in range(gy0, gy1):
            for xx in range(gx0, gx1):
                r[ci * 196 + yy * 14 + xx] += v
    PN[name] = len(P)
    P.append(r)
    PB.append(bias)
    return name


def rg(ch, y0, y1, x0, x1, w=1.0, pool="avg"):
    return (ch, pool, y0, y1, x0, x1, w)


def box(name, ch, y0, y1, x0, x1, w=1.0, pool="avg"):
    return prim(name, [rg(ch, y0, y1, x0, x1, w, pool)])


def low(name, ch, y0, y1, x0, x1, t, w=1.0, pool="avg"):
    """relu(t - region mean): fires when the region is BELOW t."""
    return prim(name, [rg(ch, y0, y1, x0, x1, w, pool)], bias=t, neg=True)

# ---- global amounts
box("fill", "S", 0, 28, 0, 28)
box("fillMid", "Smid", 0, 28, 0, 28)
box("fillBri", "Sbri", 0, 28, 0, 28)
box("grayMean", "gray", 0, 28, 0, 28)
box("thinFrac", "thin", 0, 28, 0, 28)
box("erodeFrac", "erode", 0, 28, 0, 28)
box("edgeFrac", "edge", 0, 28, 0, 28)
box("texAll", "tex", 0, 28, 0, 28)
box("texbAll", "texb", 0, 28, 0, 28)
box("texgAll", "texg", 0, 28, 0, 28)
box("nrmPAll", "nrmP", 0, 28, 0, 28)

# ---- row widths / column heights of the silhouette, in pixels (0..28)
for i in range(14):
    box("R%d" % i, "S", 2 * i, 2 * i + 2, 0, 28, 28.0)
    box("C%d" % i, "S", 0, 28, 2 * i, 2 * i + 2, 28.0)
    low("eR%d" % i, "S", 2 * i, 2 * i + 2, 0, 28, 1.2, 28.0)
    low("eC%d" % i, "S", 0, 28, 2 * i, 2 * i + 2, 1.2, 28.0)

# ---- coarse 4x4 shape blocks
YB = [0, 6, 14, 22, 28]
XB = [0, 6, 14, 22, 28]
for r in range(4):
    for c in range(4):
        box("B%d%d" % (r, c), "S", YB[r], YB[r + 1], XB[c], XB[c + 1])

# ---- targeted structure regions
# neck / collar at the top centre
box("neckS", "S", 0, 6, 10, 18)
box("neckLine", "vlineN", 0, 6, 10, 18, pool="max")
box("neckLineA", "vlineN", 0, 6, 10, 18)
box("neckWide", "vlineW", 0, 6, 8, 20, pool="max")
box("shoulderS", "S", 2, 8, 0, 28, 28.0)
box("topStrip", "S", 0, 4, 0, 28, 28.0)
box("botStrip", "S", 24, 28, 0, 28, 28.0)
# sleeves: upper-outer vs lower-outer mass
box("slUpL", "S", 2, 10, 0, 6)
box("slUpR", "S", 2, 10, 22, 28)
box("slLoL", "S", 14, 26, 0, 8)
box("slLoR", "S", 14, 26, 20, 28)
box("slMidL", "S", 10, 16, 0, 6)
box("slMidR", "S", 10, 16, 22, 28)
# sleeve / body separation slits
box("slitL", "vlineN", 8, 24, 4, 10, pool="max")
box("slitLa", "vlineN", 8, 24, 4, 10)
box("slitR", "vlineN", 8, 24, 18, 24, pool="max")
box("slitRa", "vlineN", 8, 24, 18, 24)
# centre front opening / placket
box("ctrLine", "vlineN", 6, 26, 12, 16, pool="max")
box("ctrLineA", "vlineN", 6, 26, 12, 16)
box("ctrLineLo", "vlineN", 16, 26, 12, 16, pool="max")
box("ctrLineUp", "vlineN", 4, 12, 12, 16, pool="max")
box("ctrLineM", "vlineM", 6, 26, 12, 16, pool="max")
box("ctrWide", "vlineW", 14, 28, 10, 18, pool="max")
box("ctrWideA", "vlineW", 14, 28, 10, 18)
box("ctrWideBot", "vlineW", 22, 28, 10, 18, pool="max")
# trouser legs
box("legL", "S", 16, 28, 6, 12)
box("legR", "S", 16, 28, 16, 22)
box("legGapS", "S", 16, 28, 12, 16)
box("crotch", "S", 8, 14, 10, 18)
# bag handle: enclosed dark loop in the upper part, solid body below it
box("handleRing", "ring", 0, 12, 6, 22, pool="max")
box("handleRingA", "ring", 0, 12, 6, 22)
box("handleD", "holeD", 0, 12, 6, 22, pool="max")
box("handleU", "holeU", 0, 12, 6, 22, pool="max")
box("handleL", "holeL", 0, 12, 6, 22, pool="max")
box("handleR", "holeR", 0, 12, 6, 22, pool="max")
box("handleBody", "S", 12, 26, 4, 24)
box("handleTop", "S", 0, 6, 6, 22)
# enclosed holes anywhere (sandal straps, bag loops)
box("ringAll", "ring", 0, 28, 0, 28, pool="max")
box("ringAllA", "ring", 0, 28, 0, 28)
box("ringLo", "ring", 14, 28, 4, 24, pool="max")
# shoe regions (shoes span the full width and sit in the middle rows)
box("shoeUpL", "S", 4, 12, 0, 10)
box("shoeUpR", "S", 4, 12, 18, 28)
box("shoeUpC", "S", 4, 12, 10, 18)
box("shoeMid", "S", 12, 18, 0, 28)
box("shoeLo", "S", 18, 24, 0, 28)
box("shoeTop", "S", 0, 8, 0, 28)
box("shoeBot", "S", 22, 28, 0, 28)
box("shaftR", "S", 2, 10, 16, 26)
box("shaftL", "S", 2, 10, 2, 12)
box("toeL", "S", 14, 22, 0, 8)
box("heelR", "S", 18, 26, 20, 28)
box("soleEdge", "gyP", 16, 26, 2, 26)
box("topEdge", "gyN", 2, 14, 2, 26)
# texture inside the garment body
box("texBody", "tex", 6, 24, 6, 22)
box("texbBody", "texb", 6, 24, 6, 22)
box("edgeBody", "edge", 6, 24, 6, 22)
box("gridVBody", "gridV", 6, 24, 6, 22)
box("gridHBody", "gridH", 6, 24, 6, 22)
box("texCore", "tex", 8, 22, 8, 20)
box("erodeBody", "erode", 6, 24, 6, 22)
box("thinBody", "thin", 6, 24, 6, 22)
# horizontal structure (ribbed hems, soles, waistbands)
box("hlineLo", "hlineN", 20, 28, 4, 24, pool="max")
box("hlineUp", "hlineN", 0, 8, 4, 24, pool="max")
box("hlineAll", "hlineN", 0, 28, 0, 28, pool="max")
box("hWideAll", "hlineW", 0, 28, 0, 28, pool="max")
# slits near the hem: pullover/coat cuffs reach the hem, dress sleeves usually do not
box("slitLoL", "vlineN", 18, 26, 4, 10, pool="max")
box("slitLoR", "vlineN", 18, 26, 18, 24, pool="max")
box("slitUpL", "vlineN", 6, 14, 4, 10, pool="max")
box("slitUpR", "vlineN", 6, 14, 18, 24, pool="max")
# how straight / vertical the outer edges are (bags) vs sloped (shoes, flares)
box("vertEdgeL", "gxP", 6, 24, 0, 8)
box("vertEdgeR", "gxN", 6, 24, 20, 28)
box("horizEdgeT", "gyN", 2, 12, 4, 24)
box("horizEdgeB", "gyP", 18, 28, 4, 24)
box("diagEdge", "dgP", 0, 28, 0, 28)
box("diagEdgeN", "dgN", 0, 28, 0, 28)
# shading-based centre line / collar / sleeve folds (visible on cloth, not on silhouette)
box("ctrLineG", "vlineG11", 6, 26, 12, 16, pool="max")
box("ctrLineGlo", "vlineG11", 16, 26, 12, 16, pool="max")
box("ctrLineGup", "vlineG11", 4, 12, 12, 16, pool="max")
box("ctrLineGa", "vlineGa", 6, 26, 12, 16, pool="max")
box("ctrLineGavg", "vlineG11", 6, 26, 12, 16)
box("neckG", "vlineG11", 0, 6, 10, 18, pool="max")
box("slitGL", "vlineG", 8, 24, 4, 10, pool="max")
box("slitGR", "vlineG", 8, 24, 18, 24, pool="max")
box("hlineGlo", "hlineG", 20, 28, 4, 24, pool="max")
NP = len(P)

# ------------------------------------------------- primitive dense layer
primL = dense("prim", flat, NP, np.stack(P).astype(np.float32), np.array(PB, np.float32))
primR = layer("primR", "relu", [primL])

# ------------------------------------------------- derived geometry properties
Q = []
QN = {}
QB = []


def prop(name, terms, bias=0.0):
    r = np.zeros(NP, np.float64)
    for nm, w in terms:
        r[PN[nm]] += w
    QN[name] = len(Q)
    Q.append(r)
    QB.append(bias)


# occupied height / width in pixels: a 2-row band counts as occupied once its
# bright width passes ~1 pixel.  28 minus the empty bands.
prop("hgt", [("eR%d" % i, -2 / 1.2) for i in range(14)], 28.0)
prop("wid", [("eC%d" % i, -2 / 1.2) for i in range(14)], 28.0)
prop("hemW", [("R11", 1 / 3), ("R12", 1 / 3), ("R13", 1 / 3)])
prop("waistW", [("R5", 1 / 3), ("R6", 1 / 3), ("R7", 1 / 3)])
prop("shoulderW", [("R1", 1 / 3), ("R2", 1 / 3), ("R3", 1 / 3)])
prop("chestW", [("R3", 1 / 3), ("R4", 1 / 3), ("R5", 1 / 3)])
NQ = len(Q)
propL = dense("prop", primR, NQ, np.stack(Q).astype(np.float32), np.array(QB, np.float32))
propR = layer("propR", "relu", [propL])

# concatenate primitives and properties so the rule layer can read both
NCAT = NP + NQ
cat = layer("cat", "add", [
    place("cp", layer("pr3", "reshape", [primR], shape=[NP, 1, 1]), NP, NCAT, 0),
    place("cq", layer("pp3", "reshape", [propR], shape=[NQ, 1, 1]), NQ, NCAT, NP),
])
catf = layer("catf", "reshape", [cat], shape=[NCAT])
IX = dict(PN)
for k, v in QN.items():
    IX[k] = NP + v

# ------------------------------------------------- cue layer (thresholded rules)
C = []
CN = {}
CB = []


def cue(name, terms, bias=0.0):
    r = np.zeros(NCAT, np.float64)
    for nm, w in terms:
        r[IX[nm]] += w
    CN[name] = len(C)
    C.append(r)
    CB.append(bias)
    return name


# geometry -------------------------------------------------------------------
cue("tall", [("hgt", 1), ("wid", -1)])
cue("wide", [("wid", 1), ("hgt", -1)])
cue("veryTall", [("hgt", 1)], -24.0)
cue("shortItem", [("hgt", -1)], 20.0)
cue("veryShort", [("hgt", -1)], 16.0)
cue("midShort", [("hgt", -1)], 22.0)
cue("narrowItem", [("wid", -1)], 22.0)
cue("veryNarrow", [("wid", -1)], 17.0)
cue("fullWide", [("wid", 1)], -26.0)
cue("notFullWide", [("wid", -1)], 26.0)

# solidity / sparsity --------------------------------------------------------
cue("sparse", [("thinFrac", 30), ("fill", -18)])
cue("verySparse", [("thinFrac", 30), ("fill", -24)])
cue("solid", [("erodeFrac", 10), ("fill", -3)])
cue("fillHi", [("fill", 5)], -2.25)
cue("fillLo", [("fill", -5)], 1.5)
cue("thinHi", [("thinFrac", 5)], -0.6)
cue("erodeHi", [("erodeFrac", 5)], -1.5)

# width profile --------------------------------------------------------------
cue("flare", [("hemW", 1), ("waistW", -1)])
cue("flareSh", [("hemW", 1), ("shoulderW", -1)])
cue("sleeveBump", [("shoulderW", 1), ("waistW", -1)], -3.0)
cue("narrowShoulder", [("shoulderW", -1)], 18.0)
cue("wideShoulder", [("shoulderW", 1)], -20.0)
cue("topHeavy", [("B00", 1), ("B01", 1), ("B02", 1), ("B03", 1),
                 ("B30", -1), ("B31", -1), ("B32", -1), ("B33", -1)])
cue("botHeavy", [("B30", 1), ("B31", 1), ("B32", 1), ("B33", 1),
                 ("B00", -1), ("B01", -1), ("B02", -1), ("B03", -1)])

# trouser --------------------------------------------------------------------
cue("gapWide", [("ctrWide", 2.5)], -1.5)
cue("gapWideBot", [("ctrWideBot", 2.5)], -1.5)
cue("legsBright", [("legL", 1), ("legR", 1)], -0.9)
cue("gapDarkBot", [("legGapS", -3)], 1.05)
cue("crotchBright", [("crotch", 2)], -1.0)
cue("trouserShape", [("ctrWide", 2), ("legL", 0.5), ("legR", 0.5),
                     ("legGapS", -1.2), ("wid", -0.06)], -0.9)
cue("trouserStrict", [("ctrWide", 2), ("ctrWideBot", 1.5), ("legL", 0.5), ("legR", 0.5),
                      ("legGapS", -1.5), ("wid", -0.08), ("hgt", 0.03)], -1.6)

# sleeves / slits ------------------------------------------------------------
cue("longSleeve", [("slLoL", 0.5), ("slLoR", 0.5)])
cue("longSleeveHi", [("slLoL", 0.5), ("slLoR", 0.5)], -0.45)
cue("noLongSleeve", [("slLoL", -0.5), ("slLoR", -0.5)], 0.22)
cue("midSleeve", [("slMidL", 0.5), ("slMidR", 0.5)])
cue("upSleeve", [("slUpL", 0.5), ("slUpR", 0.5)])
cue("slits", [("slitL", 0.5), ("slitR", 0.5)])
cue("slitsHi", [("slitL", 0.5), ("slitR", 0.5)], -0.45)
cue("noSlits", [("slitL", -0.5), ("slitR", -0.5)], 0.35)
cue("slitBoth", [("slitL", 1), ("slitR", 1)], -1.0)

# centre front opening / placket / collar ------------------------------------
cue("ctrLine", [("ctrLine", 1)])
cue("ctrLineHi", [("ctrLine", 2)], -1.2)
cue("ctrLineMid", [("ctrLine", 2)], -0.8)
cue("ctrLineLo", [("ctrLineLo", 2)], -1.0)
cue("ctrLineUp", [("ctrLineUp", 2)], -1.0)
cue("noCtrLine", [("ctrLine", -2)], 0.7)
cue("ctrMidOnly", [("ctrLineM", 2)], -1.0)
cue("neckNotch", [("neckLine", 1)])
cue("neckWide", [("neckWide", 1)])
cue("collarV", [("neckWide", 1.5), ("neckLine", -0.5)], -0.5)

# bag handle -----------------------------------------------------------------
cue("handle4", [("handleL", 1), ("handleR", 1), ("handleD", 1), ("handleU", 1)], -2.2)
cue("handle3", [("handleL", 1), ("handleR", 1), ("handleD", 1)], -1.7)
cue("handleRing", [("handleRing", 2)], -1.1)
cue("handleBody", [("handleBody", 1)])
cue("handleTopThin", [("handleTop", -2)], 0.7)
cue("bagLoop", [("handleRing", 1.5), ("handleD", 1), ("handleBody", 1)], -2.0)
cue("ringAny", [("ringAll", 2)], -1.2)
cue("ringLow", [("ringLo", 2)], -1.2)

# shoes ----------------------------------------------------------------------
cue("shoeTopEmpty", [("shoeTop", -3)], 0.9)
cue("shoeBotEmpty", [("shoeBot", -3)], 0.9)
cue("shaftR", [("shaftR", 1)])
cue("shaftHi", [("shaftR", 2)], -1.0)
cue("shaftLo", [("shaftR", -2)], 0.5)
cue("upAsym", [("shoeUpR", 1), ("shoeUpL", -1)])
cue("upAsymN", [("shoeUpL", 1), ("shoeUpR", -1)])
cue("soleBand", [("shoeLo", 1)])
cue("shoeMidMass", [("shoeMid", 1)])
cue("toeMass", [("toeL", 1)])
cue("heelMass", [("heelR", 1)])
cue("flatBottom", [("soleEdge", 2)], -0.5)
cue("domeTop", [("topEdge", 2)], -0.5)

# texture --------------------------------------------------------------------
cue("texIn", [("texBody", 1), ("edgeBody", -1.2)])
cue("texInHi", [("texBody", 1), ("edgeBody", -1.2)], -0.25)
cue("texbIn", [("texbBody", 1), ("edgeBody", -1.2)])
cue("plaid", [("gridVBody", 1), ("gridHBody", 1)], -0.35)
cue("plaidV", [("gridVBody", 2)], -0.5)
cue("smooth", [("texBody", -1), ("edgeBody", 1.2)], 0.15)
cue("texRatio", [("texBody", 1), ("erodeBody", -0.5)])

# ---- column-height asymmetry: a boot shaft stands up at one end, a bag does not
cue("colAsymR", [("C%d" % i, 1 / 6) for i in range(8, 14)] + [("C%d" % i, -1 / 6) for i in range(0, 6)])
cue("colAsymL", [("C%d" % i, 1 / 6) for i in range(0, 6)] + [("C%d" % i, -1 / 6) for i in range(8, 14)])
cue("hgtOver15", [("hgt", 1)], -15.0)
cue("noHandleTop", [("handleTop", -3)], 0.9)

cue("hUnder18", [("hgt", -1)], 18.0)
cue("hOver18", [("hgt", 1)], -18.0)
cue("hOver20", [("hgt", 1)], -20.0)
cue("hOver22", [("hgt", 1)], -22.0)
cue("widUnder18", [("wid", -1)], 18.0)
cue("wideHem", [("hemW", 1)], -19.0)
cue("notFullTall", [("hgt", -1)], 27.0)
cue("bagProf", [("handleBody", 2), ("handleTop", -3), ("wid", 0.05)], -1.0)
cue("ctrLineGHi", [("ctrLineG", 2)], -1.0)
cue("ctrLineGMid", [("ctrLineG", 2)], -0.5)
cue("ctrLineGAvg", [("ctrLineGavg", 4)], -1.0)
cue("hUnder24", [("hgt", -1)], 24.0)
# trouser: a real background gap between two legs, on a narrow tall silhouette
cue("trouserGap", [("ctrLineLo", 2.5), ("legL", 0.6), ("legR", 0.6), ("legGapS", -1.5),
                   ("wid", -0.06), ("slLoL", -1.0), ("slLoR", -1.0)], -1.0)
cue("trouserNarrow", [("wid", -1.5), ("hgt", 0.06), ("slLoL", -1.0), ("slLoR", -1.0),
                    ("legGapS", -5.0), ("ctrLineLo", 1.5), ("hemW", -0.5), ("waistW", 0.5)], 25.5)

NK = len(C)
cueL = dense("cue", catf, NK, np.stack(C).astype(np.float32), np.array(CB, np.float32))
cueR = layer("cueR", "relu", [cueL])

# concatenate primitives + properties + cues for the shape-evidence layer
NC2 = NP + NQ + NK
cat2 = layer("cat2", "add", [
    place("dp", layer("pr4", "reshape", [primR], shape=[NP, 1, 1]), NP, NC2, 0),
    place("dq", layer("pp4", "reshape", [propR], shape=[NQ, 1, 1]), NQ, NC2, NP),
    place("dc", layer("cc4", "reshape", [cueR], shape=[NK, 1, 1]), NK, NC2, NP + NQ),
])
cat2f = layer("cat2f", "reshape", [cat2], shape=[NC2])
IX2 = dict(IX)
for k, v in CN.items():
    IX2[k] = NP + NQ + v

D = []
DN = {}
DB = []


def det(name, terms, bias=0.0):
    r = np.zeros(NC2, np.float64)
    for nm, w in terms:
        r[IX2[nm]] += w
    DN[name] = len(D)
    D.append(r)
    DB.append(bias)
    return name


# ------------------------------------------------- per-class shape evidence
det("tshirtShape", [("shoulderW", 0.20), ("slLoL", -1.5), ("slLoR", -1.5),
                    ("slitL", -0.4), ("slitR", -0.4), ("ctrLine", -0.6),
                    ("ctrLineGHi", -0.5), ("hgt", 0.05)], -3.6)
det("pulloverShape", [("slitL", 1.2), ("slitR", 1.2), ("slLoL", 0.8), ("slLoR", 0.8),
                      ("wideHem", 0.10), ("ctrLineGHi", -2.5), ("ctrLineGAvg", -2.0),
                      ("ctrLineGMid", -0.5), ("neckWide", -0.5), ("hgt", 0.05), ("wid", 0.04)], -3.6)
det("coatShape", [("ctrLineGHi", 2.0), ("ctrLineGAvg", 2.0), ("slitL", 0.6), ("slitR", 0.6), ("slLoL", 0.4),
                  ("slLoR", 0.4), ("hgt", 0.05), ("wid", 0.03), ("neckWide", 0.5)], -3.0)
det("shirtShape", [("ctrLineGMid", 1.5), ("texIn", 1.6), ("plaid", 2.0), ("neckWide", 0.6),
                   ("hgt", 0.04), ("slitL", 0.3), ("slitR", 0.3), ("slLoL", 0.2),
                   ("slLoR", 0.2), ("notFullTall", 1.0), ("hUnder24", -1.0), ("veryTall", -0.8),
                   ("sparse", -3.0)], -1.8)
det("dressShape", [("flare", 0.15), ("narrowShoulder", 0.12), ("hgt", 0.08),
                   ("slitLoL", -0.5), ("slitLoR", -0.5), ("ctrLineGHi", -1.0),
                   ("slLoL", -0.4), ("slLoR", -0.4), ("wid", -0.03), ("wideHem", -0.5)], -1.6)
det("sandalShape", [("thinFrac", 60), ("fill", -36), ("wid", 0.10), ("hgt", -0.06),
                    ("erodeFrac", -12)], -2.0)
det("sneakerShape", [("hUnder18", 1.5), ("wid", 0.08), ("fill", 2.0), ("erodeFrac", 2.0),
                     ("sparse", -3.0), ("colAsymR", -0.20), ("hOver20", -1.5)], -2.5)
det("bootShape", [("wid", 0.08), ("fill", 2.0), ("erodeFrac", 2.0), ("colAsymR", 0.35),
                  ("veryTall", -2.5), ("sparse", -3.0), ("handle4", -2.5),
                  ("handleRing", -2.0), ("hUnder18", -1.0)], -2.5)
det("bagShape", [("handle4", 2.5), ("handleRing", 1.5), ("handle3", 1.5), ("botHeavy", 1.2),
                 ("bagProf", 1.0), ("erodeFrac", 3.0), ("wid", 0.08),
                 ("colAsymR", -0.35), ("colAsymL", -0.35), ("slitL", -1.0), ("slitR", -1.0),
                 ("ctrLineGHi", -1.0), ("veryTall", -2.0), ("veryShort", -1.5),
                 ("thinFrac", -4.0)], -2.5)
for _nm in CN:
    det(_nm, [(_nm, 1.0)])
ND = len(D)
detL = dense("det", cat2f, ND, np.stack(D).astype(np.float32), np.array(DB, np.float32))
detR = layer("detR", "relu", [detL])

# ------------------------------------------------- class scores
SC = np.zeros((10, ND), np.float64)
SBi = np.zeros(10, np.float64)


def score(k, terms, bias=0.0):
    for nm, w in terms:
        SC[k, DN[nm]] += w
    SBi[k] += bias


score(0, [("tshirtShape", 3.0), ("pulloverShape", -1.0), ("coatShape", -1.2),
          ("dressShape", -0.8), ("shirtShape", -0.8), ("trouserGap", -2.0),
          ("trouserNarrow", -2.0), ("bagShape", -1.0), ("bootShape", -1.0),
          ("sneakerShape", -1.0), ("sparse", -2.0)], 0.2)
score(1, [("trouserGap", 3.0), ("trouserNarrow", 1.0), ("gapWide", 1.0),
          ("dressShape", -1.0), ("bagShape", -1.5), ("pulloverShape", -1.0),
          ("tshirtShape", -1.0), ("coatShape", -1.0)], 0.0)
score(2, [("pulloverShape", 3.0), ("coatShape", -1.3), ("tshirtShape", -0.9),
          ("dressShape", -0.9), ("shirtShape", -0.8), ("bagShape", -1.0),
          ("trouserGap", -2.0), ("sparse", -2.0)], 0.2)
score(3, [("dressShape", 3.2), ("pulloverShape", -0.8), ("coatShape", -0.8),
          ("tshirtShape", -0.8), ("trouserGap", -2.0), ("trouserNarrow", -1.5),
          ("shirtShape", -0.5), ("bagShape", -1.0), ("sparse", -2.0)], 0.1)
score(4, [("coatShape", 3.0), ("pulloverShape", -1.3), ("shirtShape", -0.9),
          ("dressShape", -0.8), ("tshirtShape", -0.7), ("bagShape", -1.0),
          ("trouserGap", -2.0), ("sparse", -2.0)], 0.2)
score(5, [("sandalShape", 6.0), ("sneakerShape", -1.0), ("bootShape", -1.0),
          ("bagShape", -1.0), ("dressShape", -0.5)], 0.1)
score(6, [("shirtShape", 3.0), ("coatShape", -1.0), ("pulloverShape", -0.9),
          ("tshirtShape", -0.9), ("dressShape", -0.6), ("bagShape", -1.0),
          ("trouserGap", -2.0), ("sparse", -2.0)], 0.4)
score(7, [("sneakerShape", 4.0), ("bootShape", -1.2), ("sandalShape", -1.5),
          ("bagShape", -1.2)], 0.1)
score(8, [("bagShape", 3.5), ("bootShape", -1.0), ("sneakerShape", -0.8),
          ("pulloverShape", -0.8), ("coatShape", -0.8), ("tshirtShape", -0.8),
          ("dressShape", -0.8), ("trouserGap", -2.0), ("sparse", -2.0)], 0.1)
score(9, [("bootShape", 3.0), ("sneakerShape", -1.2), ("bagShape", -1.8),
          ("sandalShape", -1.2)], 0.1)

dense("scores", detR, 10, SC.astype(np.float32), SBi.astype(np.float32))

# ------------------------------------------------- write
import os
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": LAYERS}, f, indent=1)
np.savez(os.path.join(OUT, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
print("layers", len(LAYERS), "primitives", NP, "props", NQ, "detectors", ND,
      "weights", sum(v.size for v in W.values()))
