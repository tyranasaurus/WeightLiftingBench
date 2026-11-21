"""Hand-designed classifier for 28x28 greyscale clothing photos (10 classes).

Every number comes from formulas (gaussian / derivative / laplacian filters), from
shape models drawn from general knowledge of garments, shoes and bags (rectangles,
trapezoids, arcs), and from hand-picked thresholds.  No data is read and nothing is
fitted.

Pipeline
  image -> analytic filters -> base map B, 6 channels:
            0 M  soft-binarised silhouette (brightness invariant)
            1 I  raw intensity
            2 E  |laplacian|   edge energy
            3 H  |high-freq|   texture energy
            4 Gx |d/dx|        5 Gy |d/dy|
        -> row / column sums and maxima, 4x4 and 7x7 regional means, global means
        -> flat feature vector F (placement denses summed with `add`)
        -> D1  named geometric descriptors (linear)
        -> R1  relu
        -> D2  saturated checklist conditions (each condition = relu(v) and relu(v-1))
        -> R2  relu
        -> D3  per-class additive checklist  -> (10,)
"""
import json
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# analytic filters
# ----------------------------------------------------------------------------


def gauss1d(k, sigma):
    x = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def gauss2d(k, sigma):
    return np.outer(gauss1d(k, sigma), gauss1d(k, sigma))


def dgauss1d(k, sigma):
    g = gauss1d(k, sigma)
    x = np.arange(k) - (k - 1) / 2.0
    d = -g * x / (sigma ** 2)
    return d / np.abs(d).sum()


def lap2d(k, sigma):
    g = gauss2d(k, sigma)
    cx = np.zeros((k, k))
    cx[(k - 1) // 2, (k - 1) // 2] = 1.0
    L = cx - g
    return 2.0 * L / np.abs(L).sum()


K = 5
G1 = gauss2d(K, 1.0)
G2 = gauss2d(K, 2.0)
LAP = lap2d(K, 1.2)
DX = np.outer(gauss1d(K, 1.2), dgauss1d(K, 1.2))
DY = DX.T
DELTA = np.zeros((K, K))
DELTA[2, 2] = 1.0

F1_W = np.zeros((8, 1, K, K))
F1_W[0, 0] = G1
F1_W[1, 0] = LAP
F1_W[2, 0] = -LAP
F1_W[3, 0] = DELTA - G2
F1_W[4, 0] = DX
F1_W[5, 0] = -DX
F1_W[6, 0] = DY
F1_W[7, 0] = -DY
F1_b = np.array([-0.15, 0, 0, 0, 0, 0, 0, 0])

F2_W = np.zeros((2, 1, K, K))
F2_W[0, 0] = DELTA
F2_W[1, 0] = G2
F2_b = np.zeros(2)

# ----------------------------------------------------------------------------
# feature vector layout
# ----------------------------------------------------------------------------
IR_RM, IR_RX, IR_CM, IR_CX, IR_RI = 0, 28, 56, 84, 112      # 28 each
IR_RG0, IR_RG1, IR_RG2, IR_RG3 = 140, 156, 172, 188        # 16 each (4x4)
IR_FG0, IR_FG1, IR_FG2 = 204, 253, 302                     # 49 each (7x7)
IR_G = 351                                                 # 6
IR_T = 357                                                 # templates
NTEMP = 20
NF = IR_T + NTEMP


def fg(base, i, j):
    return base + i * 7 + j


def rg(base, i, j):
    return base + i * 4 + j


# ----------------------------------------------------------------------------
# hand drawn templates (general knowledge of the classes)
# ----------------------------------------------------------------------------
TEMPLATES = []


def _t():
    p = np.zeros((6, 28, 28))
    TEMPLATES.append(p)
    return p


def box(p, ch, r0, r1, c0, c1, v):
    p[ch, r0:r1, c0:c1] += v


p = _t()   # T0 t-shirt: short sleeves, narrow lower body
box(p, 0, 3, 11, 4, 24, 1.0); box(p, 0, 11, 26, 8, 20, 1.0)
box(p, 0, 12, 26, 0, 7, -0.6); box(p, 0, 12, 26, 21, 28, -0.6)
p = _t()   # T1 trouser
box(p, 0, 0, 8, 9, 19, 1.0); box(p, 0, 8, 27, 9, 13, 1.0); box(p, 0, 8, 27, 15, 19, 1.0)
box(p, 0, 12, 27, 13, 15, -1.2); box(p, 0, 0, 27, 0, 8, -0.6); box(p, 0, 0, 27, 20, 28, -0.6)
p = _t()   # T2 pullover: wide top and wide bottom
box(p, 0, 2, 12, 4, 24, 1.0); box(p, 0, 12, 26, 5, 23, 1.0)
p = _t()   # T3 dress: narrow top, flared skirt
box(p, 0, 2, 10, 10, 18, 1.0); box(p, 0, 14, 27, 7, 21, 1.0)
box(p, 0, 2, 12, 0, 8, -0.6); box(p, 0, 2, 12, 20, 28, -0.6)
p = _t()   # T4 coat: widest at the bottom, sleeves full length
box(p, 0, 1, 10, 6, 22, 1.0); box(p, 0, 10, 26, 3, 25, 1.0)
p = _t()   # T5 sandal: thin low band
box(p, 0, 13, 21, 2, 26, 1.0); box(p, 0, 0, 11, 0, 28, -0.5); box(p, 0, 22, 28, 0, 28, -0.5)
p = _t()   # T6 shirt: medium width, straight
box(p, 0, 2, 12, 5, 23, 1.0); box(p, 0, 12, 27, 8, 20, 1.0)
box(p, 0, 12, 27, 0, 6, -0.5); box(p, 0, 12, 27, 22, 28, -0.5)
p = _t()   # T7 sneaker: solid low wedge, flat sole
box(p, 0, 15, 21, 1, 27, 1.0); box(p, 0, 9, 15, 6, 27, 0.7)
box(p, 0, 0, 8, 0, 28, -0.5); box(p, 0, 22, 28, 0, 28, -0.5)
p = _t()   # T8 bag: boxy middle, empty top corners
box(p, 0, 8, 24, 3, 25, 1.0); box(p, 0, 1, 7, 10, 18, 0.5)
box(p, 0, 0, 8, 0, 8, -0.8); box(p, 0, 0, 8, 20, 28, -0.8); box(p, 0, 25, 28, 0, 28, -0.6)
p = _t()   # T9 ankle boot: shaft plus foot
box(p, 0, 4, 15, 11, 22, 1.0); box(p, 0, 14, 23, 4, 22, 1.0)
box(p, 0, 0, 4, 0, 28, -0.5); box(p, 0, 24, 28, 0, 28, -0.5)
p = _t()   # T10 trouser leg gap
box(p, 0, 16, 27, 8, 12, 1.0); box(p, 0, 16, 27, 16, 20, 1.0); box(p, 0, 16, 27, 12, 16, -1.6)
p = _t()   # T11 bag handle arc, empty corners
box(p, 0, 0, 7, 10, 18, 1.0); box(p, 0, 0, 7, 0, 8, -1.0); box(p, 0, 0, 7, 20, 28, -1.0)
p = _t()   # T12 sneaker sole: bright band low on intensity
box(p, 1, 16, 21, 1, 27, 1.0); box(p, 1, 10, 15, 1, 27, -0.7)
p = _t()   # T13 front opening: vertical edge energy centre
box(p, 2, 6, 24, 12, 16, 1.0); box(p, 2, 6, 24, 6, 10, -0.5); box(p, 2, 6, 24, 18, 22, -0.5)
p = _t()   # T14 sleeve/body separation gaps (coat): vertical edges at both sides
box(p, 4, 8, 22, 6, 8, 1.0); box(p, 4, 8, 22, 20, 22, 1.0); box(p, 4, 8, 22, 12, 16, -0.6)
p = _t()   # T15 neckline dip: dark centre top on intensity
box(p, 1, 0, 4, 11, 17, -1.0); box(p, 1, 0, 4, 6, 10, 0.6); box(p, 1, 0, 4, 18, 22, 0.6)
p = _t()   # T16 shirt collar: bright wings beside a dark V
box(p, 1, 1, 6, 10, 13, 1.0); box(p, 1, 1, 6, 15, 18, 1.0); box(p, 1, 1, 7, 13, 15, -1.2)
p = _t()   # T17 coat lapels: bright wide V at the top
box(p, 1, 1, 9, 9, 13, 1.0); box(p, 1, 1, 9, 15, 19, 1.0); box(p, 1, 2, 10, 13, 15, -0.8)
p = _t()   # T18 hem line: horizontal edge energy at the very bottom
box(p, 5, 24, 27, 6, 22, 1.0); box(p, 5, 19, 23, 6, 22, -0.6)
p = _t()   # T19 plaid / knit texture in the body
box(p, 3, 8, 22, 8, 20, 1.0); box(p, 3, 8, 22, 0, 6, -0.5); box(p, 3, 8, 22, 22, 28, -0.5)

for p in TEMPLATES:
    p -= p.mean()          # zero mean: mass outside the pattern penalises
    p /= np.abs(p).sum()

# ----------------------------------------------------------------------------
# graph
# ----------------------------------------------------------------------------
layers = []
W = {}


def L(name, op, inputs, **set):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(set)
    layers.append(d)


def conv1x1(src, n_in, nch, coefs, name, bias=None):
    m = np.zeros((nch, n_in, 1, 1))
    for dst, s, w in coefs:
        m[dst, s, 0, 0] += w
    L(name, "conv", [src], out=nch, kernel=[1, 1], stride=1, padding=0)
    W[name + ".W"] = m
    W[name + ".b"] = np.zeros(nch) if bias is None else np.array(bias, float)


L("f1", "conv", ["image"], out=8, kernel=[K, K], stride=1, padding=2)
W["f1.W"], W["f1.b"] = F1_W, F1_b
L("r1", "relu", ["f1"])
L("f2", "conv", ["image"], out=2, kernel=[K, K], stride=1, padding=2)
W["f2.W"], W["f2.b"] = F2_W, F2_b
# hf- = relu(-(hf+))
conv1x1("r1", 8, 1, [(0, 3, -1.0)], "hn0")
L("hn", "relu", ["hn0"])
# soft binarisation of the silhouette: min(max(20*M-2,0),1)
conv1x1("r1", 8, 1, [(0, 0, 20.0)], "s1", bias=[-2.0])
L("rs1", "relu", ["s1"])
conv1x1("r1", 8, 1, [(0, 0, 20.0)], "s2", bias=[-3.0])
L("rs2", "relu", ["s2"])
conv1x1("rs2", 1, 1, [(0, 0, -1.0)], "s2n")
L("Mb", "add", ["rs1", "s2n"])
conv1x1("Mb", 1, 6, [(0, 0, 1.0)], "q0")
conv1x1("f2", 2, 6, [(1, 0, 1.0)], "q1")
conv1x1("r1", 8, 6, [(2, 1, 1.0), (2, 2, 1.0)], "q2")
conv1x1("r1", 8, 6, [(3, 3, 1.0)], "q3a")
conv1x1("hn", 1, 6, [(3, 0, 1.0)], "q3b")
L("q3", "add", ["q3a", "q3b"])
conv1x1("r1", 8, 6, [(4, 4, 1.0), (4, 5, 1.0)], "q4")
conv1x1("r1", 8, 6, [(5, 6, 1.0), (5, 7, 1.0)], "q5")
L("B", "add", ["q0", "q1", "q2", "q3", "q4", "q5"])

L("Pr", "avgpool", ["B"], size=[1, 28], stride=[1, 1])
L("Prx", "maxpool", ["B"], size=[1, 28], stride=[1, 1])
L("Pc", "avgpool", ["B"], size=[28, 1], stride=[1, 1])
L("Pcx", "maxpool", ["B"], size=[28, 1], stride=[1, 1])
L("Rg", "avgpool", ["B"], size=[7, 7], stride=[7, 7])
L("Fg", "avgpool", ["B"], size=[4, 4], stride=[4, 4])
L("Gl", "avgpool", ["B"], size=[28, 28], stride=[1, 1])
L("Prf", "reshape", ["Pr"], shape=[168])
L("Prxf", "reshape", ["Prx"], shape=[168])
L("Pcf", "reshape", ["Pc"], shape=[168])
L("Pcxf", "reshape", ["Pcx"], shape=[168])
L("Rgf", "reshape", ["Rg"], shape=[96])
L("Fgf", "reshape", ["Fg"], shape=[294])
L("Glf", "reshape", ["Gl"], shape=[6])
L("Bf", "reshape", ["B"], shape=[4704])


def place(name, src_name, src_dim, entries):
    m = np.zeros((NF, src_dim))
    for dst, src_idx, w in entries:
        m[dst, src_idx] += w
    L(name, "dense", [src_name], out=NF)
    W[name + ".W"] = m
    W[name + ".b"] = np.zeros(NF)


place("pPr", "Prf", 168, [(IR_RM + r, r, 28.0) for r in range(28)] +
      [(IR_RI + r, 28 + r, 28.0) for r in range(28)])
place("pPrx", "Prxf", 168, [(IR_RX + r, r, 1.0) for r in range(28)])
place("pPc", "Pcf", 168, [(IR_CM + x, x, 28.0) for x in range(28)])
place("pPcx", "Pcxf", 168, [(IR_CX + x, x, 1.0) for x in range(28)])
place("pRg0", "Rgf", 96, [(IR_RG0 + k, k, 1.0) for k in range(16)])
place("pRg1", "Rgf", 96, [(IR_RG1 + k, 16 + k, 1.0) for k in range(16)])
place("pRg2", "Rgf", 96, [(IR_RG2 + k, 32 + k, 1.0) for k in range(16)])
place("pRg3", "Rgf", 96, [(IR_RG3 + k, 48 + k, 1.0) for k in range(16)])
place("pFg0", "Fgf", 294, [(IR_FG0 + k, k, 1.0) for k in range(49)])
place("pFg1", "Fgf", 294, [(IR_FG1 + k, 49 + k, 1.0) for k in range(49)])
place("pFg2", "Fgf", 294, [(IR_FG2 + k, 98 + k, 1.0) for k in range(49)])
place("pGl", "Glf", 6, [(IR_G + c, c, 1.0) for c in range(6)])
m = np.zeros((NF, 4704))
for t, p in enumerate(TEMPLATES):
    m[IR_T + t, :] = p.reshape(-1)
L("pT", "dense", ["Bf"], out=NF)
W["pT.W"] = m
W["pT.b"] = np.zeros(NF)
L("F", "add", ["pPr", "pPrx", "pPc", "pPcx", "pRg0", "pRg1", "pRg2", "pRg3",
               "pFg0", "pFg1", "pFg2", "pGl", "pT"])

# ----------------------------------------------------------------------------
# D1 descriptors
# ----------------------------------------------------------------------------
D1 = []


def d1(name, terms, bias=0.0):
    D1.append([name, dict(terms), bias])


def rm(r):
    return IR_RM + r


def rx(r):
    return IR_RX + r


def cm(x):
    return IR_CM + x


def cx(x):
    return IR_CX + x


def ri(r):
    return IR_RI + r


def band(fn, a, b):
    return [(fn(i), 1.0 / (b - a)) for i in range(a, b)]


def sumf(fn, a, b):
    return [(fn(i), 1.0) for i in range(a, b)]


d1("w0", band(rm, 0, 3))
d1("wA", band(rm, 3, 8))
d1("wB", band(rm, 8, 13))
d1("wC", band(rm, 13, 18))
d1("wD", band(rm, 18, 23))
d1("wE", band(rm, 23, 27))
d1("drop", band(rm, 8, 13) + [(t, -w) for t, w in band(rm, 18, 23)])
d1("flare", band(rm, 23, 27) + [(t, -w) for t, w in band(rm, 8, 13)])
d1("widen", band(rm, 18, 23) + [(t, -w) for t, w in band(rm, 3, 8)])
d1("A", sumf(rm, 0, 28))
d1("topm", sumf(rm, 0, 7))
d1("botm", sumf(rm, 21, 28))
d1("Hocc", sumf(rx, 0, 28))
d1("htop", sumf(rx, 0, 4))
d1("Wocc", sumf(cx, 0, 28))
d1("asym", sumf(cm, 0, 14) + [(cm(x), -1.0) for x in range(14, 28)])
d1("iTop", sumf(ri, 0, 7))
d1("sole", band(ri, 16, 21) + [(t, -w) for t, w in band(ri, 10, 15)])
# centre-bottom gap (trouser legs): fine grid cells rows 16-27, cols 12-15
d1("cgap", [(fg(IR_FG0, i, j), 1.0 / 6) for i in (4, 5, 6) for j in (3, 4)])
d1("csid", [(fg(IR_FG0, i, j), 1.0 / 12) for i in (4, 5, 6) for j in (2, 5)])
# neck region intensity vs shoulders
d1("neck", [(fg(IR_FG1, 0, 3), 1.0), (fg(IR_FG1, 0, 4), 1.0),
            (fg(IR_FG1, 0, 2), -0.7), (fg(IR_FG1, 0, 5), -0.7)])
# collar wings bright vs centre V (rows 0-7)
d1("collar", [(fg(IR_FG1, 0, 2), 1.0), (fg(IR_FG1, 0, 5), 1.0),
              (fg(IR_FG1, 0, 3), -1.0), (fg(IR_FG1, 0, 4), -1.0)])
# vertical edge energy in mid band at the sides (sleeve separation)
d1("sides", [(fg(IR_FG2, i, j), 1.0 / 8) for i in (2, 3, 4) for j in (1, 6)] +
     [(fg(IR_FG2, i, j), -1.0 / 8) for i in (2, 3, 4) for j in (3, 4)])
for c, nm in enumerate(["meanM", "meanI", "edgeD", "texD", "gxD", "gyD"]):
    d1(nm, [(IR_G + c, 1.0)])
for t in range(NTEMP):
    d1("T%d" % t, [(IR_T + t, 1.0)])

D1N = len(D1)
IDX1 = {nm: i for i, (nm, _, _) in enumerate(D1)}

# ----------------------------------------------------------------------------
# D2 saturated gates:  g = relu(v), g_s = relu(v-1);  D3 uses g - g_s in [0,1]
# ----------------------------------------------------------------------------
D2 = []


def d2raw(name, terms, bias):
    D2.append([name, {IDX1[k]: v for k, v in terms}, bias])


def gate(name, terms, bias):
    d2raw(name, terms, bias)
    d2raw(name + "_s", terms, bias - 1.0)


def P(n):
    return [(n, 1.0)]


def N(n):
    return [(n, -1.0)]


for k in range(16):
    d1("r%d" % k, [(IR_RG0 + k, 1.0)])
for k in range(49):
    d1("q%d" % k, [(IR_FG0 + k, 1.0)])
d1("botasym", [(fg(IR_FG0, 6, j), 1.0) for j in (0, 1, 2)] +
     [(fg(IR_FG0, 6, j), -1.0) for j in (4, 5, 6)])
d1("botfill", [(fg(IR_FG0, 6, j), 1.0 / 7) for j in range(7)])
D1N = len(D1)
IDX1 = {nm: i for i, (nm, _, _) in enumerate(D1)}

gate("g_tall", [("Hocc", 1.0 / 3)], -24.0 / 3)      # fills the frame vertically
gate("g_notTall", [("Hocc", -1.0 / 3)], 24.0 / 3)
gate("g_short", [("Hocc", -1.0 / 3)], 17.0 / 3)     # low profile: sandal/sneaker
gate("g_midH", [("Hocc", -1.0 / 3)], 22.0 / 3)      # boot / bag height band
gate("g_midH2", [("Hocc", 1.0 / 3)], -14.0 / 3)
gate("g_wNarrow", [("Wocc", -1.0 / 3)], 14.0 / 3)   # trouser / dress
gate("g_wMid", [("Wocc", 1.0 / 3)], -14.0 / 3)
gate("g_wMid2", [("Wocc", -1.0 / 3)], 20.0 / 3)
gate("g_wWide", [("Wocc", 1.0 / 4)], -20.0 / 4)     # footwear / bag
gate("g_drop", [("drop", 1.0 / 2)], -1.8 / 2)       # sleeves end high: t-shirt
gate("g_noDrop", [("drop", -1.0 / 2)], 1.0 / 2)
gate("g_flare", [("flare", 1.0 / 1.5)], 0.0)        # skirt flare: dress
gate("g_widen", [("widen", 1.0 / 1.5)], -2.5 / 1.5) # hem wider than shoulders: coat
gate("g_noWiden", [("widen", -1.0 / 2)], 2.0 / 2)
gate("g_bulk", [("wC", 1.0 / 1.5)], -16.5 / 1.5)    # bulky middle: pullover / coat
gate("g_slim", [("wC", -1.0 / 2)], 16.5 / 2)        # slim middle: shirt / t-shirt
gate("g_legs", [("wE", -1.0 / 3)], 9.0 / 3)         # thin ankles: trouser
gate("g_shNar", [("wA", -1.0 / 2)], 12.0 / 2)       # narrow shoulders: dress/trouser
gate("g_shWide", [("wA", 1.0 / 2)], -14.5 / 2)
gate("g_areaS", [("A", -1.0 / 60)], 240.0 / 60)
gate("g_areaM", [("A", 1.0 / 60)], -260.0 / 60)
gate("g_areaB", [("A", 1.0 / 60)], -400.0 / 60)
gate("g_dark", [("meanI", -1.0 / 0.06)], 0.24 / 0.06)
gate("g_bright", [("meanI", 1.0 / 0.06)], -0.31 / 0.06)
gate("g_texHi", [("texD", 1.0 / 0.02)], -0.05 / 0.02)
gate("g_gap", [("cgap", -1.0 / 0.15)], 0.62 / 0.15)   # empty centre bottom: legs
gate("g_nogap", [("cgap", 1.0 / 0.15)], -0.75 / 0.15)
gate("g_topEmpty", [("htop", -1.0 / 1.5)], 2.0 / 1.5) # nothing at the very top
gate("g_topFull", [("htop", 1.0 / 1.0)], -3.0 / 1.0)
gate("g_solidMid", [("wC", 1.0 / 3)], -16.0 / 3)    # solid mid band: sneaker/boot
gate("g_sparseMid", [("wC", -1.0 / 3)], 14.0 / 3)   # strappy mid band: sandal
gate("g_footLow", [("wD", -1.0 / 3)], 12.0 / 3)     # little below mid: sneaker/sandal
gate("g_bootLow", [("wD", 1.0 / 4)], -15.0 / 4)     # full foot block: boot
gate("g_aHsparse", [("A", -1.0 / 25), ("Hocc", 12.5 / 25)], 0.0)  # low fill
gate("g_collarN", [("collar", -1.0 / 0.1)], 0.12 / 0.1)  # dark centre V: shirt/coat
gate("g_collarP", [("collar", 1.0 / 0.1)], 0.14 / 0.1)  # round neck: tshirt/pullover
gate("g_widenBig", [("widen", 1.0 / 4)], -8.0 / 4)  # bag / boot: hem much wider
gate("g_hnd", [("r9", 1.0), ("r10", 1.0), ("r0", -1.0), ("r3", -1.0),
               ("r1", -1.0), ("r2", -1.0)], -1.0 / 0.4)
gate("g_bagH", [("Hocc", -1.0 / 3)], 26.0 / 3)
gate("g_bagH2", [("Hocc", 1.0 / 3)], -17.0 / 3)
gate("g_heel", [("botasym", 1.0 / 0.05)], -0.06 / 0.05)
gate("g_boxy", [("r9", 1.0), ("r10", 1.0)], -1.3 / 0.3)
gate("g_hndP", [("r1", 1.0), ("r2", 1.0)], -0.25 / 0.15)
gate("g_hndS", [("r1", -1.0), ("r2", -1.0)], 0.9 / 0.3)
gate("g_botfill", [("botfill", 1.0 / 0.15)], -0.5 / 0.15)
gate("g_botSparse", [("botfill", -1.0 / 0.15)], 0.5 / 0.15)
gate("g_midSparse", [("r9", -1.0), ("r10", -1.0)], 1.2 / 0.3)
gate("g_botWide", [("wE", 1.0 / 2)], -10.0 / 2)
gate("g_hemNar", [("wE", -1.0 / 1.5)], 13.5 / 1.5)
for t in range(NTEMP):
    gate("g_tT%d" % t, [("T%d" % t, 1.0 / 0.15)], -0.05 / 0.15)
    gate("g_nT%d" % t, [("T%d" % t, -1.0 / 0.15)], -0.05 / 0.15)

D2N = len(D2)
IDX2 = {nm: i for i, (nm, _, _) in enumerate(D2)}

# soft ANDs over gate values: min(a,b) = a + b - relu(a-b) - relu(b-a)
ANDS = [("g_widen", "g_tall"), ("g_bagH", "g_bagH2"), ("g_hnd", "g_bagH"),
        ("g_noDrop", "g_wMid"), ("g_heel", "g_midH2"), ("g_boxy", "g_widenBig"), ("g_widenBig", "g_bagH"),
        ("g_collarN", "g_wMid"), ("g_shNar", "g_tall"),
        ("g_hndP", "g_hndS"), ("g_hndP", "g_botfill"), ("g_hndP", "g_widenBig"),
        ("g_midSparse", "g_tall"), ("g_topEmpty", "g_midH2"), ("g_shNar", "g_botWide"), ("g_drop", "g_hemNar"),
        ("g_botSparse", "g_bootLow"), ("g_midH2", "g_bootLow"), ("g_midH", "g_midH2"), ("g_wMid", "g_wMid2"),
        ("g_tall", "g_wNarrow"), ("g_short", "g_wWide"), ("g_midH", "g_wWide"),
        ("g_tall", "g_slim"), ("g_tall", "g_bulk"), ("g_drop", "g_tall"),
        ("g_noDrop", "g_bulk"), ("g_noDrop", "g_slim"), ("g_widen", "g_bulk"),
        ("g_flare", "g_shNar"), ("g_gap", "g_tall"), ("g_short", "g_solidMid"), ("g_short", "g_sparseMid"),
        ("g_midH", "g_solidMid"), ("g_midH2", "g_bootLow"),
        ("g_widenBig", "g_midH"), ("g_drop", "g_shWide"), ("g_noDrop", "g_wMid2"),
        ("g_noWiden", "g_slim"), ("g_widen", "g_wMid"), ("g_topEmpty", "g_wWide"),
        ("g_aHsparse", "g_short"), ("g_collarN", "g_noDrop"),
        ("g_collarP", "g_drop"), ("g_gap", "g_wNarrow"), ("g_flare", "g_tall"),
        ("g_bulk", "g_wMid"), ("g_slim", "g_wMid")]
D2B = []
for a, b in ANDS:
    D2B.append(["ab_%s_%s" % (a, b), {IDX2[a]: 1.0, IDX2[b]: -1.0}, 0.0])
    D2B.append(["ab_%s_%s_r" % (a, b), {IDX2[a]: -1.0, IDX2[b]: 1.0}, 0.0])

m = np.zeros((D1N, NF))
b = np.zeros(D1N)
for i, (nm, terms, bi) in enumerate(D1):
    for k, v in terms.items():
        m[i, k] += v
    b[i] = bi
L("D1", "dense", ["F"], out=D1N)
W["D1.W"], W["D1.b"] = m, b
L("R1", "relu", ["D1"])

m = np.zeros((D2N, D1N))
b = np.zeros(D2N)
for i, (nm, terms, bi) in enumerate(D2):
    for k, v in terms.items():
        m[i, k] += v
    b[i] = bi
L("D2", "dense", ["R1"], out=D2N)
W["D2.W"], W["D2.b"] = m, b
L("R2", "relu", ["D2"])

# D2b: identity of every gate, plus a clipped copy clip(v,0,1) = relu(v) - relu(v-1)
n2b = 2 * D2N
IDXC = {D2[i][0]: i for i in range(D2N)}
m = np.zeros((n2b, D2N))
b = np.zeros(n2b)
for i in range(D2N):
    m[i, i] = 1.0
for i in range(D2N):
    nm = D2[i][0]
    if nm.endswith("_s"):
        base = nm[:-2]
        m[D2N + i, IDX2[base]] = 1.0
        m[D2N + i, IDX2[nm]] = -1.0
        IDXC[base + "_c"] = D2N + i
L("D2b", "dense", ["R2"], out=n2b)
W["D2b.W"], W["D2b.b"] = m, b
L("R2b", "relu", ["D2b"])

# D2c: identity of R2b plus soft-AND helpers min(a,b) over clipped gates
D2C = []
for a, b_ in ANDS:
    D2C.append(["mn_%s_%s" % (a, b_), {IDXC[a + "_c"]: 1.0, IDXC[b_ + "_c"]: -1.0}, 0.0])
    D2C.append(["mn_%s_%s_r" % (a, b_), {IDXC[a + "_c"]: -1.0, IDXC[b_ + "_c"]: 1.0}, 0.0])
n2c = n2b + len(D2C)
m = np.zeros((n2c, n2b))
b = np.zeros(n2c)
for i in range(n2b):
    m[i, i] = 1.0
for r, (nm, terms, bi) in enumerate(D2C):
    for k, v in terms.items():
        m[n2b + r, k] += v
L("D2c", "dense", ["R2b"], out=n2c)
W["D2c.W"], W["D2c.b"] = m, b
L("R2c", "relu", ["D2c"])
IDX2B = {nm: i for i, (nm, _, _) in enumerate(D2)}
for k, v in IDXC.items():
    IDX2B[k] = v
for r, (nm, _, _) in enumerate(D2C):
    IDX2B[nm] = n2b + r

# ----------------------------------------------------------------------------
# D3 per-class checklist (hand written scoring rules)
# ----------------------------------------------------------------------------
CLS = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt",
       "sneaker", "bag", "boot"]
D3 = {c: {} for c in CLS}


def give(cls, unit, w):
    D3[cls][unit] = D3[cls].get(unit, 0.0) + w


def gsat(cls, g, w):
    """w * clip(v, 0, 1) for a gate"""
    give(cls, g + "_c", w)


def gand(cls, a, b, w):
    """w * min(clip(a), clip(b)) for two gates"""
    gsat(cls, a, w)
    gsat(cls, b, w)
    give(cls, "mn_%s_%s" % (a, b), -w)
    give(cls, "mn_%s_%s_r" % (a, b), -w)


def gtemp(cls, t, w):
    gsat(cls, "g_tT%d" % t, w)
    gsat(cls, "g_nT%d" % t, -w)


# --- t-shirt: short sleeves, slim body, round neck --------------------------
gand("tshirt", "g_drop", "g_tall", 4.0)
gand("tshirt", "g_drop", "g_hemNar", 3.0)
gsat("tshirt", "g_drop", 3.0)
gand("tshirt", "g_slim", "g_wMid", 2.0)
gsat("tshirt", "g_tall", 1.5)
gsat("tshirt", "g_collarP", 1.0)
gsat("tshirt", "g_noDrop", -4.0)
gsat("tshirt", "g_bulk", -3.0)
gsat("tshirt", "g_wNarrow", -3.0)
gsat("tshirt", "g_flare", -2.0)
gsat("tshirt", "g_short", -6.0)
gsat("tshirt", "g_midH", -3.0)
gtemp("tshirt", 0, 1.0)
# --- trouser: narrow, tall, open between the legs ---------------------------
gand("trouser", "g_gap", "g_tall", 4.0)
gsat("trouser", "g_wNarrow", 4.0)
gsat("trouser", "g_tall", 2.0)
gsat("trouser", "g_gap", 3.0)
gsat("trouser", "g_legs", 3.0)
gsat("trouser", "g_shNar", 1.0)
gand("trouser", "g_midSparse", "g_tall", 4.0)
gsat("trouser", "g_wWide", -5.0)
gsat("trouser", "g_bulk", -4.0)
gsat("trouser", "g_topEmpty", -4.0)
gsat("trouser", "g_nogap", -3.0)
gtemp("trouser", 1, 2.5)
gtemp("trouser", 10, 1.5)
# --- pullover: long sleeves, bulky knit body, round neck -------------------
gand("pullover", "g_noDrop", "g_bulk", 5.0)
gand("pullover", "g_noDrop", "g_wMid", 2.0)
gsat("pullover", "g_noDrop", 2.0)
gsat("pullover", "g_bulk", 2.5)
gsat("pullover", "g_tall", 1.5)
gsat("pullover", "g_collarP", 1.5)
gsat("pullover", "g_noWiden", 1.0)
gsat("pullover", "g_drop", -4.0)
gsat("pullover", "g_slim", -3.0)
gsat("pullover", "g_wNarrow", -3.0)
gsat("pullover", "g_widen", -1.0)
gsat("pullover", "g_short", -6.0)
gsat("pullover", "g_midH", -3.0)
gtemp("pullover", 2, 1.0)
# --- dress: narrow shoulders, flared skirt ---------------------------------
gand("dress", "g_flare", "g_shNar", 4.0)
gand("dress", "g_shNar", "g_tall", 2.0)
gand("dress", "g_shNar", "g_botWide", 4.0)
gsat("dress", "g_shNar", 3.0)
gsat("dress", "g_wNarrow", 2.5)
gsat("dress", "g_flare", 2.0)
gsat("dress", "g_tall", 1.5)
gsat("dress", "g_shWide", -4.0)
gsat("dress", "g_wWide", -3.0)
gsat("dress", "g_gap", -3.0)
gsat("dress", "g_short", -6.0)
gsat("dress", "g_midH", -3.0)
gtemp("dress", 3, 1.0)
# --- coat: widest at the hem, bulky, open front ----------------------------
gand("coat", "g_widen", "g_tall", 5.0)
gand("coat", "g_widen", "g_bulk", 3.0)
gsat("coat", "g_widen", 1.0)
gsat("coat", "g_bulk", 1.0)
gsat("coat", "g_noDrop", 1.0)
gsat("coat", "g_tall", 1.0)
gsat("coat", "g_collarN", 0.5)
gsat("coat", "g_hnd", -3.0)
gsat("coat", "g_drop", -4.0)
gsat("coat", "g_slim", -3.0)
gsat("coat", "g_wNarrow", -3.0)
gsat("coat", "g_short", -6.0)
gsat("coat", "g_midH", -3.0)
gtemp("coat", 4, 1.0)
gtemp("coat", 17, 0.5)
gtemp("coat", 14, 0.5)
# --- sandal: low, wide, strappy (sparse) -----------------------------------
gand("sandal", "g_short", "g_sparseMid", 4.0)
gand("sandal", "g_short", "g_wWide", 3.0)
gsat("sandal", "g_sparseMid", 2.0)
gand("sandal", "g_aHsparse", "g_short", 3.0)
gsat("sandal", "g_short", 2.0)
gsat("sandal", "g_wWide", 2.0)
gsat("sandal", "g_topEmpty", 2.0)
gsat("sandal", "g_dark", 1.0)
gsat("sandal", "g_tall", -6.0)
gsat("sandal", "g_solidMid", -4.0)
gsat("sandal", "g_areaB", -4.0)
gsat("sandal", "g_midH2", -2.0)
gtemp("sandal", 5, 1.0)
# --- shirt: long sleeves, slim straight body, collar -----------------------
gand("shirt", "g_noDrop", "g_slim", 4.0)
gand("shirt", "g_collarN", "g_noDrop", 1.5)
gand("shirt", "g_noDrop", "g_wMid", 2.0)
gand("shirt", "g_slim", "g_wMid", 2.0)
gand("shirt", "g_collarN", "g_wMid", 1.5)
gsat("shirt", "g_collarN", 1.5)
gsat("shirt", "g_noDrop", 1.0)
gsat("shirt", "g_slim", 1.0)
gsat("shirt", "g_flare", -2.0)
gsat("shirt", "g_shNar", -2.0)
gsat("shirt", "g_tall", 1.0)
gsat("shirt", "g_noWiden", 1.0)
gsat("shirt", "g_texHi", 1.0)
gsat("shirt", "g_drop", -4.0)
gsat("shirt", "g_bulk", -3.0)
gsat("shirt", "g_wNarrow", -3.0)
gsat("shirt", "g_widen", -1.0)
gsat("shirt", "g_short", -6.0)
gsat("shirt", "g_midH", -3.0)
gtemp("shirt", 6, 1.0)
gtemp("shirt", 16, 1.0)
# --- sneaker: low, wide, solid, flat sole ----------------------------------
gand("sneaker", "g_short", "g_solidMid", 4.0)
gsat("sneaker", "g_short", 2.0)
gsat("sneaker", "g_wWide", 2.0)
gsat("sneaker", "g_footLow", 2.0)
gsat("sneaker", "g_topEmpty", 2.0)
gsat("sneaker", "g_tall", -6.0)
gsat("sneaker", "g_sparseMid", -4.0)
gsat("sneaker", "g_midH2", -3.0)
gsat("sneaker", "g_dark", -1.0)

gtemp("sneaker", 7, 1.0)
gtemp("sneaker", 12, 0.5)
# --- bag: boxy body under a handle, empty top corners ----------------------
gand("bag", "g_hndP", "g_hndS", 8.0)
gand("bag", "g_hndP", "g_botfill", 2.0)
gand("bag", "g_hndP", "g_widenBig", 2.0)
gand("bag", "g_bagH", "g_bagH2", 3.0)
gsat("bag", "g_heel", -2.0)
gand("bag", "g_widenBig", "g_midH", 3.0)
gsat("bag", "g_wWide", 2.0)
gand("bag", "g_midH", "g_midH2", 2.0)
gsat("bag", "g_topEmpty", 2.0)
gsat("bag", "g_tall", -4.0)
gsat("bag", "g_wNarrow", -4.0)
gsat("bag", "g_short", -4.0)
gsat("bag", "g_gap", -3.0)
gsat("bag", "g_bootLow", -3.0)
gtemp("bag", 8, 1.0)
gtemp("bag", 11, 2.0)
# --- ankle boot: mid height, wide, solid foot block ------------------------
gand("boot", "g_midH", "g_midH2", 3.0)
gand("boot", "g_midH2", "g_bootLow", 3.0)
gsat("boot", "g_wWide", 2.0)
gsat("boot", "g_solidMid", 2.0)
gsat("boot", "g_boxy", -3.0)
gand("boot", "g_midH", "g_midH2", 4.0)
gand("boot", "g_topEmpty", "g_midH2", 2.0)
gsat("boot", "g_midH", 2.0)
gsat("boot", "g_heel", 2.0)
gsat("boot", "g_hnd", -3.0)
gsat("boot", "g_topEmpty", 1.0)
gsat("boot", "g_short", -3.0)
gsat("boot", "g_tall", -4.0)
gsat("boot", "g_wNarrow", -2.0)
gsat("boot", "g_sparseMid", -3.0)
gtemp("boot", 9, 1.0)


m = np.zeros((10, n2c))
b = np.zeros(10)
for c, cls in enumerate(CLS):
    for unit, w in D3[cls].items():
        m[c, IDX2B[unit]] += w
L("D3", "dense", ["R2c"], out=10)
W["D3.W"], W["D3.b"] = m, b


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f)
    np.savez(os.path.join(outdir, "weights.npz"),
             **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
