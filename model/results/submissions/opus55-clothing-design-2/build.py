#!/usr/bin/env python3
"""Writes a hand-designed clothing classifier: python3 build.py OUT_DIR  ->  OUT_DIR/network.json, OUT_DIR/weights.npz

Nothing here is learned or read from data. Every weight is a formula:
  1. a soft silhouette mask (grey level ramp 0.04 -> 0.12),
  2. running sums along rows and columns, clipped by relu, telling for every pixel whether the item has
     started yet when you walk in from the left / right / top / bottom ("open" directions),
  3. a 1x1 conv that turns the four open flags into region codes: notch open to top-left, top-right,
     bottom-left, bottom-right, top only, bottom only, ..., enclosed hole, row-filled span, ...
  4. edge / line maps from small derivative kernels, gated to the inside of the item,
  5. region sums of those maps (= named features in pixel units),
  6. hand-written fuzzy rules (ramps on linear combinations of features) giving scores for sub-classes,
  7. max over the sub-classes of each class.
"""
import json
import sys
from pathlib import Path

import numpy as np

S = 28


# ----------------------------------------------------------------------------------------------------------
# tiny graph builder
# ----------------------------------------------------------------------------------------------------------
class Net:
    def __init__(self):
        self.layers, self.w, self.shape = [], {}, {"image": (1, S, S)}

    def _add(self, L, shape):
        self.layers.append(L)
        self.shape[L["name"]] = tuple(shape)
        return L["name"]

    def conv(self, name, inp, W, b, padding=0, stride=1):
        W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
        o, i, k, _ = W.shape
        c, h, w = self.shape[inp]
        assert i == c, (name, i, c)
        self.w[name + ".W"], self.w[name + ".b"] = W, b
        ho, wo = (h + 2 * padding - k) // stride + 1, (w + 2 * padding - k) // stride + 1
        return self._add(dict(name=name, op="conv", inputs=[inp], out=int(o), kernel=[int(k), int(k)],
                              stride=int(stride), padding=int(padding)), (o, ho, wo))

    def dense(self, name, inp, W, b):
        W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
        n_in = int(np.prod(self.shape[inp]))
        assert W.shape[1] == n_in, (name, W.shape, n_in)
        self.w[name + ".W"], self.w[name + ".b"] = W, b
        return self._add(dict(name=name, op="dense", inputs=[inp], out=int(W.shape[0])), (W.shape[0],))

    def relu(self, name, inp):
        return self._add(dict(name=name, op="relu", inputs=[inp]), self.shape[inp])

    def maxpool(self, name, inp, size, stride):
        c, h, w = self.shape[inp]
        ho, wo = (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1
        return self._add(dict(name=name, op="maxpool", inputs=[inp], size=list(size), stride=list(stride)), (c, ho, wo))

    def reshape(self, name, inp, shape):
        assert int(np.prod(shape)) == int(np.prod(self.shape[inp]))
        return self._add(dict(name=name, op="reshape", inputs=[inp], shape=list(shape)), shape)

    def add(self, name, inps):
        return self._add(dict(name=name, op="add", inputs=list(inps)), self.shape[inps[0]])


# ----------------------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------------------
def region(rows, cols):
    """28x28 0/1 array, rows/cols are inclusive (lo, hi) pairs."""
    r = np.zeros((S, S))
    r[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1] = 1.0
    return r


ALL = ((0, 27), (0, 27))
RB = {"t": (0, 8), "m": (9, 18), "b": (19, 27)}          # row bands
CB = {"l": (0, 8), "c": (9, 18), "r": (19, 27)}          # column bands

# mask ramp: 0 at grey 0.04, 1 at grey 0.12
MASK_A, MASK_T = 12.5, 0.04

# region-code channels made from the open flags NL, NR, NT, NB (1 = item not met yet when walking in from that side)
#            NL  NR  NT  NB   M  bias
CODES = {
    "TL": (1, -1, 1, -1, 0, -1),
    "TR": (-1, 1, 1, -1, 0, -1),
    "BL": (1, -1, -1, 1, 0, -1),
    "BR": (-1, 1, -1, 1, 0, -1),
    "To": (-1, -1, 1, -1, 0, 0),
    "Bo": (-1, -1, -1, 1, 0, 0),
    "Lo": (1, -1, -1, -1, 0, 0),
    "Ro": (-1, 1, -1, -1, 0, 0),
    "Ho": (-1, -1, -1, -1, -3, 1),         # enclosed background; any pixel with mask >= 1/3 is not a hole
    "LR": (1, 1, -1, -1, 0, -1),
    "TB": (-1, -1, 1, 1, 0, -1),
    "Fr": (-1, -1, 0, 0, 0, 1),
    "Fc": (0, 0, -1, -1, 0, 1),
    "M": (0, 0, 0, 0, 1, 0),
}
CODE_NAMES = list(CODES)


class Features:
    """Named linear features: feature = sum over tensors of <weight map, tensor> + bias."""

    def __init__(self, net):
        self.net, self.names, self.parts, self.bias = net, [], [], []

    def add(self, name, parts, bias=0.0):
        """parts: list of (tensor_name, weight array of the tensor's shape)."""
        assert name not in self.names, name
        self.names.append(name); self.parts.append(parts); self.bias.append(float(bias))

    def idx(self, name):
        return self.names.index(name)

    def compile(self, name):
        K = len(self.names)
        tensors = sorted({t for p in self.parts for t, _ in p})
        outs = []
        for t in tensors:
            n_in = int(np.prod(self.net.shape[t]))
            W = np.zeros((K, n_in))
            for k, p in enumerate(self.parts):
                for tt, w in p:
                    if tt == t:
                        W[k] += np.asarray(w, np.float64).reshape(-1)
            b = np.array(self.bias) if not outs else np.zeros(K)
            outs.append(self.net.dense(f"{name}_{t}", t, W, b))
        return self.net.add(name, outs) if len(outs) > 1 else outs[0]


def silhouette(net, tag, t0, t1):
    """Soft mask ramp from grey t0 (0) to t1 (1); open-direction maps N (NL, NR, NT, NB, M) and region codes."""
    a = 1.0 / (t1 - t0)
    m = net.conv("m" + tag, "image", np.full((2, 1, 1, 1), a), [-a * t0, -a * t0 - 1])
    mr = net.relu("m%sr" % tag, m)                 # M = mr[0] - mr[1]
    # running sums from each side, NX = relu(1 - running sum)
    n_in = 2 * S * S
    Wn = np.zeros((5, S, S, 2, S, S))
    for y in range(S):
        for x in range(S):
            for ch, sel in enumerate([
                (slice(y, y + 1), slice(0, x + 1)),       # from left
                (slice(y, y + 1), slice(x, S)),           # from right
                (slice(0, y + 1), slice(x, x + 1)),       # from top
                (slice(y, S), slice(x, x + 1)),           # from bottom
            ]):
                Wn[ch, y, x, 0][sel] = -1.0
                Wn[ch, y, x, 1][sel] = +1.0
            Wn[4, y, x, 0, y, x] = 1.0
            Wn[4, y, x, 1, y, x] = -1.0
    bn = np.zeros((5, S, S)); bn[:4] = 1.0
    nm = net.dense("nmap" + tag, mr, Wn.reshape(5 * S * S, n_in), bn.reshape(-1))
    nmr = net.relu("nmap%sr" % tag, nm)
    N = net.reshape("N" + tag, nmr, [5, S, S])     # channels NL, NR, NT, NB, M
    Wc = np.zeros((len(CODES), 5, 1, 1)); bc = np.zeros(len(CODES))
    for k, nm_ in enumerate(CODE_NAMES):
        v = CODES[nm_]
        Wc[k, :, 0, 0] = v[:5]; bc[k] = v[5]
    codes = net.conv("codes" + tag, N, Wc, bc)
    C = net.relu("C" + tag, codes)
    return N, C


def build():
    net = Net()

    # ---------------- 1. silhouette branches: low threshold (whole item) and high threshold (dark gaps show) ------
    N, C = silhouette(net, "1", MASK_T, MASK_T + 1.0 / MASK_A)
    N2, C2 = silhouette(net, "2", 0.15, 0.25)

    # ---------------- 2. edge branch ----------------
    # 7x7 kernels: mask ramp, small gradients (texture), and tall vertical-line / vertical-edge filters (zip, placket,
    # sleeve seams): the mean over 7 rows of [-.5, 1, -.5] and of [-.5, 0, .5] across columns.
    k7 = np.zeros((12, 1, 7, 7)); b7 = np.zeros(12)
    k7[0, 0, 3, 3] = MASK_A; b7[0] = -MASK_A * MASK_T
    k7[1, 0, 3, 3] = MASK_A; b7[1] = -MASK_A * MASK_T - 1
    k7[10, 0, 3, 3] = 50.0; b7[10] = -1.0                     # hard low-threshold mask: grey 0.02 -> 0.04
    k7[11, 0, 3, 3] = 50.0; b7[11] = -2.0
    sm = np.array([1.0, 2.0, 1.0]) / 4.0
    gx = np.outer(sm, [-0.5, 0.0, 0.5]); gy = gx.T
    vd2 = np.outer(np.ones(7) / 7, [-0.5, 1.0, -0.5])
    vgx = np.outer(np.ones(7) / 7, [-0.5, 0.0, 0.5])
    for c, f in [(2, gx), (4, gy)]:
        k7[c, 0, 2:5, 2:5] = f; k7[c + 1, 0, 2:5, 2:5] = -f
    for c, f in [(6, vd2), (8, vgx)]:
        k7[c, 0, :, 2:5] = f; k7[c + 1, 0, :, 2:5] = -f
    e1 = net.conv("e1", "image", k7, b7, padding=3)
    e1r = net.relu("e1r", e1)
    k = np.zeros((8, 12, 5, 5)); b = np.zeros(8)
    k[0, 0] = 1.0; k[0, 1] = -1.0; b[0] = -24.0               # eroded mask (all 5x5 neighbours inside)
    for c in range(1, 5):                                     # |gx|, |gy|, |vd2|, |vgx|
        k[c, 2 * c, 2, 2] = k[c, 2 * c + 1, 2, 2] = 1.0
    k[5, 0, 1:4, 1:4] = 1.0; k[5, 1, 1:4, 1:4] = -1.0; b[5] = -8.0   # eroded mask (all 3x3 neighbours inside)
    k[6, 10, 1:4, 1:4] = 1.0; k[6, 11, 1:4, 1:4] = -1.0; b[6] = -8.0  # 3x3 erosion of the hard low mask
    k[7, 10, 2, 2] = 1.0; k[7, 11, 2, 2] = -1.0                       # hard low mask itself
    e2 = net.conv("e2", e1r, k, b, padding=2)
    e2r = net.relu("e2r", e2)
    k = np.zeros((8, 8, 1, 1)); b = np.zeros(8)
    for c in range(1, 5):                                     # keep edges only well inside the item
        k[c - 1, c, 0, 0] = 1.0; k[c - 1, 0, 0, 0] = 1.0; b[c - 1] = -1.0
    k[4, 0, 0, 0] = 1.0
    for c in (5, 6, 7):
        k[c, c, 0, 0] = 1.0
    e3 = net.conv("e3", e2r, k, b)
    E = net.relu("E", e3)        # gx, gy, vd2, vgx (inside), eroded mask 5x5, 3x3, eroded hard low mask 3x3, hard low mask
    Emax = net.maxpool("Emax", E, [1, 6], [1, 1])             # row-wise max over 6 columns, (8, 28, 23)
    # per row and window: ramps "is there a tall vertical line" (vd2 0.06 -> 0.12) and "edge" (vgx 0.05 -> 0.10)
    k = np.zeros((4, 8, 1, 1)); b = np.zeros(4)
    k[0, 2, 0, 0] = 1 / 0.06; b[0] = -1.0
    k[2, 3, 0, 0] = 1 / 0.05; b[2] = -1.0
    k[1] = k[0]; b[1] = b[0] - 1.0
    k[3] = k[2]; b[3] = b[2] - 1.0
    Lc = net.conv("lines", Emax, k, b)
    Lr = net.relu("linesr", Lc)

    # ---------------- 3. features ----------------
    F = Features(net)
    shapeC = net.shape[C]

    def cmap(code, rows=ALL[0], cols=ALL[1], weight=None, CC=None):
        w = np.zeros(shapeC)
        w[CODE_NAMES.index(code)] = region(rows, cols) if weight is None else weight
        return (C if CC is None else CC, w)

    def nmapw(ch, w2d):
        w = np.zeros((5, S, S)); w[ch] = w2d
        return (N, w)

    # width / height: column x is occupied iff NT at the bottom row is 0
    w = np.zeros((S, S)); w[S - 1, :] = -1.0
    F.add("W", [nmapw(2, w)], bias=S)
    w = np.zeros((S, S)); w[:, S - 1] = -1.0
    F.add("H", [nmapw(0, w)], bias=S)
    F.add("A", [cmap("M")])
    F.add("AF", [cmap("Fr")])
    for code in ["TL", "TR", "BL", "BR", "To", "Bo", "Lo", "Ro", "Ho", "LR", "TB"]:
        F.add(code, [cmap(code)])
        for rb, rr in RB.items():
            for cb, cc in CB.items():
                F.add(f"{code}_{rb}{cb}", [cmap(code, rr, cc)])
    # row extents e(y) = 28 - left margin - right margin (= -28 for an empty row), row mask counts n(y)
    for y in range(S):
        wl = np.zeros((S, S)); wl[y, :] = -1.0
        F.add(f"e{y}", [nmapw(0, wl), nmapw(1, wl)], bias=S)
        F.add(f"n{y}", [cmap("M", (y, y), (0, 27))])
    # column extents and top margins
    for x in range(S):
        wt = np.zeros((S, S)); wt[:, x] = -1.0
        F.add(f"h{x}", [nmapw(2, wt), nmapw(3, wt)], bias=S)
        F.add(f"tm{x}", [nmapw(2, -wt)])
    # interior edge statistics
    shE, shX = net.shape[E], net.shape[Emax]

    def emap(ch, rows, cols):
        w = np.zeros(shE); w[ch] = region(rows, cols); return (E, w)

    def xmap(ch, rows, j):
        w = np.zeros(shX); w[ch, rows[0]:rows[1] + 1, j] = 1.0; return (Emax, w)

    F.add("AE", [emap(4, ALL[0], ALL[1])])                    # eroded area
    F.add("AE3", [emap(5, ALL[0], ALL[1])])                   # area left after a 3x3 erosion (thin straps vanish)
    F.add("AE3lo", [emap(6, ALL[0], ALL[1])])                 # the same on the hard low-threshold mask (dim items)
    F.add("Alo", [emap(7, ALL[0], ALL[1])])
    F.add("TX", [emap(0, ALL[0], ALL[1]), emap(1, ALL[0], ALL[1])])
    F.add("TXgx", [emap(0, ALL[0], ALL[1])])
    F.add("TXgy", [emap(1, ALL[0], ALL[1])])
    # rows with a vertical line: per row, (max |gx| or |d2x| over 6 columns) - 0.7 (max |gy|), ramp 0.03 -> 0.08
    shL = net.shape[Lr]

    def lmap(ch, rows, j, wgt=1.0):
        w = np.zeros(shL); w[ch, rows[0]:rows[1] + 1, j] = wgt; return (Lr, w)

    F.add("zipRows", [lmap(c, (4, 25), 11, (1.0, -1.0, 1.0, -1.0)[c]) for c in range(4)])
    F.add("zipLow", [lmap(c, (13, 25), 11, (1.0, -1.0, 1.0, -1.0)[c]) for c in range(4)])   # below chest prints
    F.add("sideRows", [lmap(c, (10, 25), j, (0.5, -0.5, 0.5, -0.5)[c]) for c in range(4) for j in (3, 19)])
    # named region features
    F.add("TLlow", [cmap("TL", (9, 27))])
    F.add("TRlow", [cmap("TR", (9, 27))])
    F.add("gap", [cmap("Bo", (13, 22), (9, 18))])                      # between the legs, below the crotch
    F.add("gapAny", [cmap("Bo", (13, 27), (9, 18))])
    F.add("TNtop", [cmap("TL", (0, 8)), cmap("TR", (0, 8))])
    F.add("hemGap", [cmap("Bo", (21, 27), (5, 22)), cmap("Bo", (21, 27), (5, 22), CC=C2)])  # under a hem, between sleeves
    F.add("heelGap", [cmap("Bo", (14, 27), (10, 24))])                 # under the arch of a heeled shoe
    F.add("holes", [cmap("To"), cmap("Ho"), cmap("Lo"), cmap("Ro")])
    F.add("BoSide", [cmap("Bo", (8, 27), (0, 8)), cmap("Bo", (8, 27), (19, 27))])
    F.add("holeTop", [cmap("Ho", (0, 12), (2, 25))])
    F.add("neck", [cmap("To", (0, 9), (8, 19))])
    # the same measures on the high-threshold silhouette
    F.add("A2", [cmap("M", CC=C2)])
    F.add("AF2", [cmap("Fr", CC=C2)])
    F.add("gap2", [cmap("Bo", (13, 22), (9, 18), CC=C2)])
    F.add("BoSide2", [cmap("Bo", (8, 27), (0, 8), CC=C2), cmap("Bo", (8, 27), (19, 27), CC=C2)])
    F.add("Ho2", [cmap("Ho", CC=C2)])
    feat = F.compile("feat")
    return net, F, feat


# ----------------------------------------------------------------------------------------------------------
# fuzzy rules: predicate p = clamp((z - a) / (b - a), 0, 1) with z a linear combination of named features,
# terms = AND of predicates or their negations (relu(sum of literals - (n - 1))), sub-class score = bias + sum w * term
# ----------------------------------------------------------------------------------------------------------
class Rules:
    def __init__(self):
        self.preds, self.pidx, self.terms, self.tidx, self.subs = [], {}, [], {}, {k: [] for k in range(10)}

    def pred(self, name, expr, a, b):
        assert name not in self.pidx and b != a
        self.pidx[name] = len(self.preds); self.preds.append((name, dict(expr), float(a), float(b)))

    def term(self, s):
        if s not in self.tidx:
            lits = []
            for t in s.split("&"):
                neg = t.startswith("!")
                lits.append((self.pidx[t.lstrip("!")], neg))
            self.tidx[s] = len(self.terms); self.terms.append(lits)
        return self.tidx[s]

    def sub(self, k, parts, bias=0.0):
        self.subs[k].append(([(float(w), self.term(t)) for w, t in parts], float(bias)))

    def compile(self, net, F, feat, nsub):
        P, K = len(self.preds), len(F.names)
        W1 = np.zeros((2 * P, K)); b1 = np.zeros(2 * P)
        for i, (name, expr, a, b) in enumerate(self.preds):
            for f, c in expr.items():
                if f == "1":
                    b1[2 * i] += c / (b - a)
                else:
                    W1[2 * i, F.idx(f)] += c / (b - a)
            b1[2 * i] -= a / (b - a)
            W1[2 * i + 1] = W1[2 * i]; b1[2 * i + 1] = b1[2 * i] - 1.0
        h1 = net.relu("predr", net.dense("pred", feat, W1, b1))
        T = len(self.terms)
        W2 = np.zeros((T, 2 * P)); b2 = np.zeros(T)
        for t, lits in enumerate(self.terms):
            b2[t] = -(len(lits) - 1)
            for i, neg in lits:
                sgn = -1.0 if neg else 1.0
                W2[t, 2 * i] += sgn; W2[t, 2 * i + 1] -= sgn
                if neg:
                    b2[t] += 1.0
        h2 = net.relu("termr", net.dense("term", h1, W2, b2))
        W3 = np.zeros((10, nsub, T)); b3 = np.zeros((10, nsub))
        for k in range(10):
            subs = self.subs[k]
            assert 1 <= len(subs) <= nsub, k
            for j in range(nsub):
                parts, bias = subs[min(j, len(subs) - 1)]
                b3[k, j] = bias
                for w, t in parts:
                    W3[k, j, t] += w
        sc = net.dense("subscore", h2, W3.reshape(10 * nsub, T), b3.reshape(-1))
        sc = net.reshape("subgrid", sc, [1, 10, nsub])
        sc = net.maxpool("classmax", sc, [1, nsub], [1, nsub])
        return net.reshape("scores", sc, [10])


def rules():
    R = Rules()
    p = R.pred
    # --- geometry (pixels; the longest side of every item is 28)
    p("wide", {"W": 1}, 24, 26.5)
    p("tall", {"H": 1}, 25, 27)
    p("sq", {"H": 1}, 18.5, 22)                # tops are roughly as tall as wide
    p("lowH", {"H": -1}, -17.5, -15.5)        # H <= 15.5: sneaker height
    p("highH", {"H": 1}, 15.5, 18.5)          # boot height
    p("narrow", {"W": -1}, -16.5, -13.5)      # W <= 13.5
    p("vnarrow", {"W": -1}, -12.5, -10.5)     # W <= 10.5
    p("dressW", {"W": -1}, -21, -17.5)        # W <= 17.5
    p("topW", {"W": 1}, 19.5, 23.5)
    p("wider", {"W": 1, "H": -1}, -1, 2)
    p("vwider", {"W": 1, "H": -1}, 5, 8)      # much wider than tall: not a top
    p("taller", {"H": 1, "W": -1}, 0, 4)
    p("bagH", {"H": -1}, -25, -21.5)          # H <= 21.5
    p("tankW", {"W": 1}, 16, 18.5)
    p("tankW2", {"W": -1}, -23.5, -21.5)      # W <= 21.5
    # --- footwear: toe on the left, so a bigger notch open to the top-left than to the top-right
    p("asym", {"TL": 1, "TR": -1.5}, 5, 25)
    p("asymW", {"TL": 1, "TR": -1}, 0, 15)
    p("sparse", {"AF": 0.85, "Alo": -1}, -15, 15)   # a good part of the row spans is background (straps)
    p("blobby", {"AE3lo": 1, "Alo": -0.45}, -15, 15)   # solid body: much of it survives a 3x3 erosion
    p("sparse2", {"AF2": 0.85, "A2": -1}, -20, 10)
    p("holes", {"holes": 1}, 15, 45)                # background between straps (enclosed or open to one side)
    p("thin", {"H": 12.6, "Alo": -1}, -20, 20)      # mask covers < 45 % of the 28 x H box
    p("bigTL", {"TL": 1}, 80, 120)
    p("toe", dict([("H", 2.5)] + [("h%d" % x, -1.0) for x in range(1, 6)]), -8, 6)   # columns 1-5 under half the height
    p("heel", {"heelGap": 1}, 4, 12)
    # --- trousers: background notch open only to the bottom between the legs
    p("gap", {"gap": 1}, 4, 12)
    p("gapA", {"gapAny": 1}, 6, 16)                 # leg gap anywhere below row 13
    p("gapH", {"gap2": 1}, 6, 16)                   # leg gap seen on the high-threshold silhouette
    p("narrowTop", {"TNtop": 1}, 4, 12)             # top narrower than the hem (dress shoulders / straps)
    # --- dress: notches open to top-left / top-right reaching below row 9 (skirt flares out)
    p("flare", {"TLlow": 1, "TRlow": 1}, 15, 50)
    # --- tops
    p("short", {"BL": 1, "BR": 1}, 40, 90)          # background under short sleeves
    p("longS", {"BL": -1, "BR": -1}, -50, -25)
    p("neck", {"neck": 1}, 8, 20)                   # notch open only to the top, at the top centre
    p("zip", {"zipLow": 1}, 6, 16)                  # tall vertical line down the centre of the lower body
    p("tex", {"TX": 1, "AE": -0.07}, 0, 12)         # busy pattern inside the item
    p("texX", {"TXgx": 1, "AE": -0.04}, 0, 6)       # checks: both vertical and horizontal edges everywhere
    p("texY", {"TXgy": 1, "AE": -0.04}, 0, 6)
    p("sideL", {"sideRows": 1}, 4, 12)              # sleeve seams hanging next to the body
    p("shoulders", {"TL": 1, "TR": 1}, 12, 30)
    p("bigTN", {"TL": 1, "TR": 1}, 80, 120)
    p("sleeveGap", {"BoSide": 1, "BoSide2": 1, "hemGap": 1}, 6, 20)
    # --- bags
    p("handle", {"holeTop": 1}, 5, 15)              # enclosed hole near the top (under a handle)
    p("topN", {"TL": 1, "TR": 1}, 45, 85)
    p("rectN", {"TL": -1, "TR": -1, "BL": -1, "BR": -1}, -50, -20)
    p("rsolid", {"A": 1, "H": -22}, -40, 0)

    FW = "wide&asym"          # shoe seen from the side
    FL = "wide&lowH&asymW"    # low, wide item, toe side lower: shoe
    NOTTOP = [(-6, FW), (-2, "handle"), (-5, "!sq"), (-4, "vwider")]
    LONG = [(3, "longS"), (1, "sideL"), (1, "sleeveGap"), (-3, "short"), (-3, "gap"), (-2, "bigTN")]
    # 0 T-shirt / top
    R.sub(0, [(4, "short&topW"), (2, "topW"), (1, "shoulders")] + NOTTOP)
    R.sub(0, [(3, "neck"), (2, "tankW&tankW2"), (1, "tall"), (1, "!flare"), (-3, "gap")] + NOTTOP)
    # 1 trouser
    R.sub(1, [(6, "gap&tall"), (3, "narrow"), (1, "vnarrow"), (-6, FW)])
    R.sub(1, [(5, "gapA&tall&narrow"), (2, "narrow"), (1, "vnarrow"), (-6, FW)])
    R.sub(1, [(4, "gapH&tall&narrow"), (2, "narrow"), (1, "vnarrow"), (-6, FW)])
    R.sub(1, [(3, "vnarrow&tall"), (1, "narrow"), (-1, "flare"), (-1, "narrowTop"), (-6, FW)])
    # 2 pullover: the default long-sleeved top
    R.sub(2, LONG + [(2, "topW"), (1, "wider"), (1.5, "shoulders"), (-1.5, "zip"), (-1, "tex"), (-1, "neck")] + NOTTOP)
    # 3 dress
    R.sub(3, [(3, "dressW"), (3, "flare&tall"), (1, "tall"), (-4, "gap"), (-3, "gapA&narrow"), (-6, FW)])
    # 4 coat: long sleeves plus a zip / front opening line, often longer than wide
    R.sub(4, LONG + [(1.5, "topW"), (1, "taller"), (2, "zip"), (1, "shoulders"), (-1, "neck"), (-3, "dressW")] + NOTTOP)
    # 5 sandal: thin straps (little survives erosion), background inside the row spans
    R.sub(5, [(6, FW), (3, FW + "&!blobby"), (1.5, FW + "&sparse"), (1.5, FW + "&thin"), (1, FW + "&holes")])
    R.sub(5, [(5, FL), (3, FL + "&!blobby"), (1.5, FL + "&sparse"), (1.5, FL + "&thin"), (1, FL + "&holes")])
    # 6 shirt: needs a busy pattern, preferably checks (vertical and horizontal edges)
    R.sub(6, [(2, "longS"), (2, "topW"), (1, "tex"), (3, "texX&texY"), (1, "zip"), (0.5, "sideL"), (1, "shoulders"),
              (-2, "bigTN"), (-3, "gap")] + NOTTOP)
    R.sub(6, [(3, "short&topW"), (2, "topW"), (1, "tex"), (3, "texX&texY"), (1, "zip")] + NOTTOP)
    # 7 sneaker
    R.sub(7, [(6, FW), (3, FW + "&lowH&blobby"), (1, FW + "&!thin"), (-4, "handle"), (-0.5, "holes"), (-1.5, "!toe")])
    R.sub(7, [(5, FL), (3, FL + "&blobby"), (-3, "rectN"), (-4, "handle"), (-0.5, "holes")])
    # 8 bag
    R.sub(8, [(4, "handle&topN"), (1, "handle"), (1, "topN"), (1, "wide"), (2, "wide&!toe"), (-3, "short"), (-6, FW),
              (-3, "!blobby")])
    R.sub(8, [(3, "rectN&wide"), (2, "bagH"), (2, "rsolid"), (-6, FW), (-2, "sideL"), (-2, "sleeveGap")])
    R.sub(8, [(3, "neck&topN&wide"), (1, "wide"), (2, "bigTN"), (-3, "short"), (-6, FW), (-3, "!blobby")])
    # 9 ankle boot
    R.sub(9, [(6, FW), (3, FW + "&highH&blobby"), (1, FW + "&!thin"), (1, FW + "&bigTL"), (1.5, FW + "&heel"),
              (-4, "handle"), (-0.5, "holes"), (-1.5, "!toe")])
    return R


def build_all():
    net, F, feat = build()
    R = rules()
    out = R.compile(net, F, feat, 4)
    return net, F, R, out


def save(net, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "network.json").write_text(json.dumps({"layers": net.layers}, indent=1))
    np.savez(out_dir / "weights.npz", **{k: v.astype(np.float32) for k, v in net.w.items()})


if __name__ == "__main__":
    net, F, R, last = build_all()
    save(net, sys.argv[1] if len(sys.argv) > 1 else ".")
