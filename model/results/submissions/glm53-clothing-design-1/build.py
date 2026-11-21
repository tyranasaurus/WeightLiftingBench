"""Hand-designed network that sorts 28x28 greyscale clothing photos into 10 classes.

Nothing here is trained or fitted: every weight comes from a formula (Gaussian
derivatives, bar detectors, a LoG "hole" detector) or from a hand-written table
of regional evidence and rules that I reasoned out from general knowledge of how
garments are shaped (see method.md).

Pipeline:
  image -> conv1 (20 hand filters, 5x5) -> relu -> avgpool 2x2 -> (20,14,14)
        -> dense "concepts" (37 evidence units, each a hand-chosen weighted sum of
           region means / edge energies, with hand thresholds) -> relu
        -> dense "rules" (10 class scores, hand-written linear rule table)

Run as:  python3 build.py NEW_FOLDER   ->  writes network.json and weights.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

K = 5          # conv kernel size
C1 = 20        # conv1 output channels
P = 14         # pooled grid size (28/2)

# ----------------------------------------------------------------------------
# kernel construction helpers (all values from formulas)
# ----------------------------------------------------------------------------
def mk(f):
    """5x5 kernel; f(dy,dx) with dy,dx in -2..2 are offsets (down/right positive).
    Conv here is cross-correlation: out[y,x] = sum W[dy,dx] * in[y+dy, x+dx]."""
    W = np.zeros((K, K))
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            W[dy + 2, dx + 2] = f(dy, dx)
    return W


def g(x, s):
    return float(np.exp(-(x * x) / (2.0 * s * s)))


def norm_step(k, axis):
    """Scale k so that a unit step edge (0 -> 1, bright side downward / rightward
    / along the diagonal) produces response exactly 1."""
    resp = 0.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if axis == "y":
                on = dy >= 1
            elif axis == "x":
                on = dx >= 1
            elif axis == "d1":      # diagonal (down-right)
                on = dy + dx >= 1
            else:                   # diagonal (down-left)
                on = dy - dx >= 1
            if on:
                resp += k[dy + 2, dx + 2]
    return k / resp


def norm_hole(k):
    """Scale a LoG kernel so a dark disk of radius 2 px on a bright field gives 1."""
    resp = 0.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy * dy + dx * dx > 4:   # outside the hole the image is bright (1)
                resp += k[dy + 2, dx + 2]
    return k / resp


# ----------------------------------------------------------------------------
# the 20 conv1 filters
# ----------------------------------------------------------------------------
SG_F, SG_C = 0.9, 1.8      # fine / coarse derivative scales
SB = 1.6                   # blur scale
SL = 1.1                   # LoG scale

k_delta = mk(lambda dy, dx: 1.0 if (dy == 0 and dx == 0) else 0.0)
k_blur = mk(lambda dy, dx: g(dy, SB) * g(dx, SB)); k_blur /= k_blur.sum()

k_hy = norm_step(mk(lambda dy, dx: dy * g(dy, SG_F) * g(dx, SG_F)), "y")       # bright below
k_hyN = -k_hy                                                                 # bright above (dark below)
k_hx = norm_step(mk(lambda dy, dx: dx * g(dy, SG_F) * g(dx, SG_F)), "x")       # bright right
k_hxN = -k_hx                                                                 # bright left
k_d1 = norm_step(mk(lambda dy, dx: (dy + dx) * g(dy, SG_F) * g(dx, SG_F)), "d1")
k_d1N = -k_d1
k_d2 = norm_step(mk(lambda dy, dx: (dy - dx) * g(dy, SG_F) * g(dx, SG_F)), "d2")
k_d2N = -k_d2

k_hy2 = norm_step(mk(lambda dy, dx: dy * g(dy, SG_C) * g(dx, SG_C)), "y")
k_hy2N = -k_hy2
k_hx2 = norm_step(mk(lambda dy, dx: dx * g(dy, SG_C) * g(dx, SG_C)), "x")
k_hx2N = -k_hx2

# vertical bar: bright vertical stripe ~2-3px wide with dark flanks (trouser leg,
# centre placket). Kernel rows are uniform (1/5 each) so an ideal 3px bar of
# brightness 1 gives response exactly 1; a wide bright area gives 0.
VB = [-0.5, 0.25, 0.5, 0.25, -0.5]
k_vbar = mk(lambda dy, dx: VB[dx + 2] / K)
k_vbarD = -k_vbar                        # dark vertical stripe between bright sides
k_hbar = mk(lambda dy, dx: VB[dy + 2] / K)     # bright horizontal stripe (sole line)
k_hbarD = -k_hbar

k_logN = norm_hole(mk(lambda dy, dx: (dy * dy + dx * dx - 2 * SL * SL)
                      * g(np.hypot(dy, dx), SL)))          # dark hole on bright (handle loop)
k_logP = -k_logN                                        # bright blob on dark

FILTERS = [k_delta, k_blur,
           k_hy, k_hyN, k_hx, k_hxN, k_d1, k_d1N, k_d2, k_d2N,
           k_vbar, k_vbarD, k_hbar, k_hbarD,
           k_hy2, k_hy2N, k_hx2, k_hx2N,
           k_logP, k_logN]
CH = dict(delta=0, blur=1, hy=2, hyN=3, hx=4, hxN=5, d1=6, d1N=7, d2=8, d2N=9,
          vbar=10, vbarD=11, hbar=12, hbarD=13, hy2=14, hy2N=15, hx2=16, hx2N=17,
          logP=18, logN=19)

# ----------------------------------------------------------------------------
# dense layer 1: concepts.  Grid is 14x14, cell (r,c) = mean of image rows
# 2r..2r+1 and cols 2c..2c+1.  A "box mean" of the delta channel is literally
# the fraction of that box filled by the garment (values in 0..1), which makes
# hand thresholds meaningful.
# ----------------------------------------------------------------------------
class W1:
    def __init__(self, n_units):
        self.W = np.zeros((n_units, C1 * P * P))
        self.b = np.zeros(n_units)
        self.units = []

    def box(self, u, ch, r0, r1, c0, c1, w):
        """add w * (mean of channel ch over box) to unit u"""
        n = (r1 - r0 + 1) * (c1 - c0 + 1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                self.W[u, ch * P * P + r * P + c] += w / n

    def m2(self, u, ch, r0, r1, c0, c1, w):
        """w * mean of the box [c0..c1] together with its mirror box"""
        self.box(u, ch, r0, r1, c0, c1, w / 2)
        self.box(u, ch, r0, r1, 13 - c1, 13 - c0, w / 2)


def build_concepts():
    w = W1(46)

    def U(name, fn):
        u = len(w.units)
        w.units.append(name)
        fn(u)
        return u

    # ---- fill-fraction means (delta channel) ----
    U("topFill",     lambda u: w.box(u, CH["delta"], 0, 1, 4, 9, 1))          # shoulders / straps
    U("topWide",     lambda u: w.m2(u, CH["delta"], 0, 2, 0, 2, 1))           # sleeves reach top rows
    U("sleeveLong",  lambda u: w.m2(u, CH["delta"], 6, 10, 0, 2, 1))          # sleeves still present below mid-torso
    U("midFill",     lambda u: w.box(u, CH["delta"], 5, 9, 3, 10, 1))
    U("lowFill",     lambda u: w.box(u, CH["delta"], 10, 13, 3, 10, 1))
    U("botFill",     lambda u: w.box(u, CH["delta"], 12, 13, 2, 11, 1))       # hem / bag bottom vs empty shoe rows
    U("centerLow",   lambda u: w.box(u, CH["delta"], 6, 12, 6, 7, 1))         # gap between trouser legs
    U("legSide",     lambda u: (w.m2(u, CH["delta"], 6, 12, 4, 5, 1)))        # the legs themselves
    U("flareLo",     lambda u: w.m2(u, CH["delta"], 9, 13, 0, 1, 1))          # skirt hem width
    U("flareHi",     lambda u: w.m2(u, CH["delta"], 3, 6, 0, 1, 1))           # width at chest level
    U("totalM",      lambda u: w.box(u, CH["delta"], 0, 13, 0, 13, 1))
    U("upSideL",     lambda u: w.box(u, CH["delta"], 1, 4, 2, 5, 1))
    U("upSideR",     lambda u: w.box(u, CH["delta"], 1, 4, 8, 11, 1))
    U("shoeUpper",   lambda u: w.box(u, CH["delta"], 3, 7, 2, 11, 1))         # sneaker/boot band
    U("shoeMid",     lambda u: w.box(u, CH["delta"], 6, 9, 1, 12, 1))         # sneaker main band
    U("handleSide",  lambda u: w.m2(u, CH["delta"], 0, 4, 4, 5, 1))           # handle strokes
    U("handleCtr",   lambda u: w.box(u, CH["delta"], 1, 3, 6, 7, 1))          # inside the handle loop
    U("narrowTopM",  lambda u: w.m2(u, CH["delta"], 0, 3, 0, 2, 1))
    U("tallUpper",   lambda u: w.box(u, CH["delta"], 3, 5, 4, 10, 1))         # boot shaft band

    # ---- edge / stroke energies ----
    U("soleBar",  lambda u: (w.box(u, CH["hbar"], 7, 11, 0, 13, 0.5),
                             w.box(u, CH["hy2N"], 7, 11, 1, 12, 0.5)))        # sole line under the shoe
    U("hemBar",   lambda u: w.box(u, CH["hy2N"], 11, 13, 1, 12, 1))           # hem at the bottom of a garment
    U("placket",  lambda u: (w.box(u, CH["vbar"], 2, 10, 6, 7, 0.5),
                             w.box(u, CH["vbarD"], 2, 10, 6, 7, 0.5)))        # centre vertical line (shirt/coat)
    U("collar",   lambda u: (w.box(u, CH["logN"], 0, 1, 5, 8, 1.0),
                             w.box(u, CH["vbarD"], 0, 2, 6, 7, 1.0)))         # dark collar notch at the very top
    U("hole",     lambda u: w.box(u, CH["logN"], 1, 4, 5, 8, 1))              # big dark hole (bag handle)
    def diagSlv(u):    # diagonal sleeve edges, both sides of the torso
        for ch in ("d1", "d1N", "d2", "d2N"):
            w.box(u, CH[ch], 3, 8, 0, 3, 0.25)
            w.box(u, CH[ch], 3, 8, 10, 13, 0.25)
    U("diagSlv", diagSlv)

    def flareEdge(u):  # outward-facing silhouette edges low down (A-line skirt)
        w.box(u, CH["hx2"], 8, 13, 0, 3, 0.5)     # left edge: bright to the right
        w.box(u, CH["hx2N"], 8, 13, 10, 13, 0.5)  # right edge: bright to the left
    U("flareEdge", flareEdge)
    U("strap",    lambda u: w.box(u, CH["hbar"], 0, 3, 2, 11, 1))             # shoulder strap bar

    def wideMid(u):    # bag: body reaches the outer columns at mid height
        w.m2(u, CH["delta"], 5, 8, 0, 2, 1.0)
    def diagFlare(u):  # dress: diagonal skirt silhouette edges, low on both sides
        for ch in ("d1", "d1N", "d2", "d2N"):
            w.box(u, CH[ch], 6, 13, 0, 3, 0.25)
            w.box(u, CH[ch], 6, 13, 10, 13, 0.25)
    def hemRib(u):     # pullover: dark ribbed band at the hem
        w.box(u, CH["hbarD"], 11, 13, 3, 10, 1.0)
    def brightPul(u):  # pullover: bright solid torso
        w.box(u, CH["delta"], 4, 12, 4, 9, 1.0); w.b[u] = -0.55
    def lapelV(u):     # coat/shirt: lapel or collar V at the very top
        for ch in ("d1", "d1N", "d2", "d2N"):
            w.box(u, CH[ch], 0, 4, 5, 8, 0.25)

    U("wideMid", wideMid); U("diagFlare", diagFlare); U("hemRib", hemRib)
    U("brightPul", brightPul); U("lapelV", lapelV)

    def strapE(u):    # many thin bright/dark strap edges (sandal, not solid sneaker)
        for ch in ("hbarD", "vbarD", "d1", "d1N", "d2", "d2N"):
            w.box(u, CH[ch], 4, 12, 1, 12, 1.0 / 6)
    U("strapE", strapE)

    # ---- derived evidence (region contrasts + hand thresholds, relu keeps >0) ----
    def legGap(u):     # trouser: legs bright beside a dark centre gap
        w.m2(u, CH["delta"], 6, 12, 4, 5, 1.0); w.box(u, CH["delta"], 6, 12, 6, 7, -1.4); w.b[u] = -0.05
    def flare(u):      # dress: much wider at the hem than at the chest
        w.m2(u, CH["delta"], 9, 13, 0, 1, 1.0); w.m2(u, CH["delta"], 3, 6, 0, 1, -1.0); w.b[u] = -0.04
    def botEmpty(u):   # shoes: nothing in the bottom two cell rows
        w.box(u, CH["delta"], 12, 13, 2, 11, -1.0); w.b[u] = 0.30
    def topEmpty(u):   # shoes/bag: nothing in the top two cell rows
        w.box(u, CH["delta"], 0, 1, 4, 9, -1.0); w.b[u] = 0.12
    def sparse(u):     # sandal: very little ink anywhere
        w.box(u, CH["delta"], 0, 13, 0, 13, -1.0); w.b[u] = 0.24
    def shaftL(u):     # ankle boot: shaft hugging one side up top
        w.box(u, CH["delta"], 1, 4, 2, 5, 1.0); w.box(u, CH["delta"], 1, 4, 8, 11, -1.0); w.b[u] = -0.10
    def shaftR(u):
        w.box(u, CH["delta"], 1, 4, 8, 11, 1.0); w.box(u, CH["delta"], 1, 4, 2, 5, -1.0); w.b[u] = -0.10
    def sleeveShort(u):  # t-shirt: sleeves end early
        w.m2(u, CH["delta"], 4, 8, 0, 2, -1.0); w.b[u] = 0.15
    def narrowTop(u):  # trouser/dress/bag: no shoulder mass in the top corners
        w.m2(u, CH["delta"], 0, 3, 0, 2, -1.0); w.b[u] = 0.07
    def holeSur(u):    # bag: bright bar over a deep dark loop with bright below it
        w.box(u, CH["delta"], 0, 1, 5, 8, 1.0)
        w.box(u, CH["delta"], 4, 6, 5, 8, 1.0)
        w.box(u, CH["delta"], 2, 5, 6, 7, -2.5); w.b[u] = -0.10

    def teeSleeve(u):   # t-shirt: shoulders present AND flanks empty below them
        w.m2(u, CH["delta"], 0, 2, 0, 2, 1.0)
        w.m2(u, CH["delta"], 4, 8, 0, 2, -1.0)
    def wideBody(u):    # bag: solid wide body, no shoulders, little top ink
        w.box(u, CH["delta"], 5, 9, 3, 10, 1.0)
        w.box(u, CH["delta"], 0, 1, 4, 9, -1.0)
        w.m2(u, CH["delta"], 0, 2, 0, 2, -2.5)

    U("legGap", legGap); U("flare", flare); U("botEmpty", botEmpty); U("topEmpty", topEmpty)
    U("sparse", sparse); U("shaftL", shaftL); U("shaftR", shaftR)
    U("sleeveShort", sleeveShort); U("narrowTop", narrowTop); U("holeSur", holeSur)
    def shortShoe(u):   # sandal: ink concentrated low, little above
        w.box(u, CH["delta"], 9, 13, 2, 11, 1.0)
        w.box(u, CH["delta"], 3, 7, 2, 11, -1.0); w.b[u] = -0.10

    U("teeSleeve", teeSleeve); U("wideBody", wideBody); U("shortShoe", shortShoe)
    return w


# ----------------------------------------------------------------------------
# dense layer 2: hand rule table.  concept name -> weight, per class score.
# ----------------------------------------------------------------------------
RULES = {
    # garment classes require ink at the top and bottom of the frame, plenty of
    # ink overall, and no sole line - shoes have none of those
    0: dict(topFill=0.8, topWide=2.0, teeSleeve=10.0, diagSlv=0.6, hemBar=0.5,
            sleeveLong=-2.5, narrowTop=-0.8, flare=-0.6, legGap=-0.8, collar=-0.5,
            placket=-0.5, topEmpty=-1.0, botEmpty=-1.0, sparse=-2.0, soleBar=-0.8),
    1: dict(legGap=4.0, topFill=0.6, sleeveShort=1.2, narrowTop=0.8, hemBar=0.4,
            flareEdge=0.5, topWide=-1.0, topEmpty=-1.0, botEmpty=-1.0, sparse=-2.0,
            soleBar=-0.8),
    2: dict(sleeveLong=1.6, topWide=1.6, topFill=0.5, midFill=0.5, hemBar=0.5,
            hemRib=1.5, brightPul=1.0, collar=-1.0, placket=-1.2, flare=-0.5,
            topEmpty=-1.0, botEmpty=-1.0, sparse=-2.0, soleBar=-0.8),
    3: dict(flare=5.5, diagFlare=2.5, wideMid=-2.0, wideBody=-1.5, lowFill=0.6,
            hemBar=0.4, flareEdge=0.6, sleeveShort=0.4, narrowTop=0.5, topFill=0.3,
            topWide=-1.2, sleeveLong=-1.5, holeSur=-0.8, topEmpty=-1.0, botEmpty=-1.0,
            sparse=-2.0, soleBar=-0.8),
    4: dict(sleeveLong=1.5, topWide=0.9, topFill=0.3, midFill=0.3, lowFill=0.5,
            placket=2.2, hemBar=0.5, collar=-1.5, sleeveShort=-0.8,
            legGap=-2.0, topEmpty=-1.0, botEmpty=-1.0, sparse=-2.0, soleBar=-0.8),
    # shoe classes: penalise garment-like solid fill
    5: dict(botEmpty=1.0, topEmpty=1.0, sparse=5.0, soleBar=0.8, sleeveShort=1.0,
            shortShoe=2.0, strapE=3.0, shoeUpper=-2.0, shoeMid=-1.5, tallUpper=-0.8,
            shaftL=-1.5, shaftR=-1.5, topFill=-1.0, botFill=-1.0, midFill=-0.5),
    6: dict(collar=4.0, placket=3.5, lapelV=0.5, sleeveLong=1.5, topWide=0.6,
            topFill=0.5, hole=0.5, diagSlv=0.5, sleeveShort=-1.2, holeSur=-0.8,
            flare=-0.5, legGap=-1.0, topEmpty=-1.0, botEmpty=-1.0, sparse=-2.0,
            soleBar=-0.8),
    7: dict(botEmpty=1.3, topEmpty=1.3, shoeUpper=2.0, shoeMid=1.6, soleBar=1.2,
            shortShoe=-2.0, totalM=0.8, sparse=-3.0, strapE=-1.0, lowFill=-1.2,
            tallUpper=-1.2, shaftL=-2.0, shaftR=-2.0, sleeveShort=-1.2, topFill=-2.0,
            botFill=-1.5, midFill=-1.0),
    8: dict(wideBody=1.0, wideMid=2.0, holeSur=2.2, hole=1.0, strap=1.2, botFill=0.8,
            lowFill=0.6, midFill=0.4, narrowTop=0.6, flareEdge=0.5, topFill=-2.0,
            topWide=-1.2, sleeveLong=-1.5, legGap=-1.5, flare=-3.5, topEmpty=-1.0,
            sparse=-2.0, placket=-1.5, collar=-1.0),
    9: dict(shaftL=2.5, shaftR=2.5, tallUpper=0.8, lowFill=0.6, soleBar=1.5,
            botEmpty=0.8, topEmpty=0.8, shoeUpper=0.6, shoeMid=0.6, sparse=-2.0,
            sleeveShort=-1.2, topFill=-1.5, botFill=-1.0, midFill=-0.8, wideBody=-0.5),
}
BIAS = {4: -0.05}


def build():
    convW = np.zeros((C1, 1, K, K), np.float64)
    for i, f in enumerate(FILTERS):
        convW[i, 0] = f
    convB = np.zeros(C1)

    w1 = build_concepts()
    n1 = w1.W.shape[0]

    w2 = np.zeros((10, n1))
    b2 = np.zeros(10)
    for cls, terms in RULES.items():
        for name, v in terms.items():
            w2[cls, w1.units.index(name)] += v
    for cls, v in BIAS.items():
        b2[cls] += v

    net = {"layers": [
        {"name": "c1", "op": "conv", "inputs": ["image"], "out": C1,
         "kernel": [K, K], "stride": 1, "padding": 2},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "p1", "op": "avgpool", "inputs": ["r1"], "size": [2, 2], "stride": [2, 2]},
        {"name": "fc1", "op": "dense", "inputs": ["p1"], "out": n1},
        {"name": "r2", "op": "relu", "inputs": ["fc1"]},
        {"name": "fc2", "op": "dense", "inputs": ["r2"], "out": 10},
    ]}
    weights = {"c1.W": convW.astype(np.float32), "c1.b": convB.astype(np.float32),
               "fc1.W": w1.W.astype(np.float32), "fc1.b": w1.b.astype(np.float32),
               "fc2.W": w2.astype(np.float32), "fc2.b": b2.astype(np.float32)}
    return net, weights


def main(folder):
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    net, weights = build()
    (out / "network.json").write_text(json.dumps(net))
    np.savez(out / "weights.npz", **weights)
    print(f"wrote {out/'network.json'} and {out/'weights.npz'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
