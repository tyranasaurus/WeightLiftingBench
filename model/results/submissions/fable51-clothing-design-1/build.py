#!/usr/bin/env python3
"""Hand-wired Fashion-MNIST classifier: writes network.json and weights.npz into the folder given as argv[1].

Every number below is a design constant chosen from general knowledge of how clothing, shoes and bags are
shaped (product photos, item scaled so its longest side is 28 px and centred, light on black).  Nothing is
fitted, averaged, copied or measured from training images.

Computation performed by the network
------------------------------------
 1. B       soft silhouette: B = clip(10*x - 0.5, 0, 1)   (grey 0.05 -> 0, grey 0.15 -> 1)
 2. pooled geometry of B
      rowmax (28)    is row y occupied?            -> H = sum = object height in px
      colmax (28)    is column x occupied?         -> W = sum = object width in px
      halves (2,28)  column fill of the upper / lower image half
      gap            two-sided "bright - dark - bright" detector on the lower-half column fill
                     (gap between trouser legs, open coat front at the centre; hanging sleeves at the sides)
      rowwidth/jump  row-width profile and its sudden increases going down (thin handle above a bag body)
      closing        7x7 and 3x3 morphological closings of B: their difference is the area of 2-6 px gaps
      erosion        5x5 erosion of B: the "solid core" that survives removing everything thinner than 5 px
      edges          horizontal and vertical transitions of B (horizontal sleeve hems at the sides)
      vline          vertical line detail of the grey image (plackets, zips, button rows)
 3. predicates: soft thresholds  p = clip((feature - t) / w, 0, 1)  of linear functionals of the above
 4. gates:      soft ANDs  clip(sum p_i - (n - 1), 0, 1)   (a negated term uses 1 - p)
 5. scores:     class k = sum of gate weights

Each clip(z, 0, 1) is realised as relu(z) - relu(z - 1): a pair of relu units whose difference is taken by the
next dense layer.  min(a, b) of two non-negative numbers is realised as (a + b - relu(a - b) - relu(b - a)) / 2.
"""
import json
import sys
from pathlib import Path

import numpy as np

S = 28


def region(r0, r1, c0, c1, value=1.0):
    """784-vector over the silhouette B with `value` on rows r0..r1-1, columns c0..c1-1."""
    m = np.zeros((S, S))
    m[r0:r1, c0:c1] = value
    return m.reshape(-1)


def band_mean_width(r0, r1):
    """Mean row width (in pixels) over rows r0..r1-1, as a functional of B."""
    return region(r0, r1, 0, S) / (r1 - r0)


layers = []
weights = {}


def add_layer(name, op, inputs, **settings):
    layers.append({"name": name, "op": op, "inputs": inputs, **settings})


# ==============================================================================================================
# 1. soft binarisation
# ==============================================================================================================
BIN_GAIN, BIN_T = 10.0, 0.05                    # ramp from grey 0.05 to grey 0.15
add_layer("bin_pre", "conv", ["image"], out=2, kernel=[1, 1], stride=1, padding=0)
weights["bin_pre.W"] = np.array([[[[BIN_GAIN]]], [[[BIN_GAIN]]]])
weights["bin_pre.b"] = np.array([-BIN_GAIN * BIN_T, -BIN_GAIN * BIN_T - 1.0])
add_layer("bin_relu", "relu", ["bin_pre"])
add_layer("B", "conv", ["bin_relu"], out=1, kernel=[1, 1], stride=1, padding=0)
weights["B.W"] = np.array([[[[1.0]], [[-1.0]]]])
weights["B.b"] = np.zeros(1)

# ==============================================================================================================
# 2. pooled geometry
# ==============================================================================================================
add_layer("rowmax", "maxpool", ["B"], size=[1, S], stride=[1, S])        # (1, 28, 1)
add_layer("colmax", "maxpool", ["B"], size=[S, 1], stride=[S, 1])        # (1, 1, 28)
add_layer("halves", "avgpool", ["B"], size=[14, 1], stride=[14, 1])      # (1, 2, 28)

# --- two-sided gap detector on the half-column fill h(x):  g_s(x) = min(h[x-s] - h[x], h[x+s] - h[x])+ ----------
GAP_SCALES = [2, 3, 5]
K = 11
NS = len(GAP_SCALES)
# stage a: per scale the four quantities  a = h[x-s]-h[x],  b = h[x+s]-h[x],  a-b,  b-a   (then relu)
add_layer("gap_a", "conv", ["halves"], out=4 * NS, kernel=[K, K], stride=1, padding=K // 2)
w = np.zeros((4 * NS, 1, K, K))
c = K // 2
for i, s in enumerate(GAP_SCALES):
    w[4 * i + 0, 0, c, c - s], w[4 * i + 0, 0, c, c] = 1.0, -1.0           # a
    w[4 * i + 1, 0, c, c + s], w[4 * i + 1, 0, c, c] = 1.0, -1.0           # b
    w[4 * i + 2, 0, c, c - s], w[4 * i + 2, 0, c, c + s] = 1.0, -1.0       # a - b
    w[4 * i + 3, 0, c, c + s], w[4 * i + 3, 0, c, c - s] = 1.0, -1.0       # b - a
weights["gap_a.W"] = w
weights["gap_a.b"] = np.zeros(4 * NS)
add_layer("gap_a_relu", "relu", ["gap_a"])
# stage b: min(a+, b+) = (a+ + b+ - (a-b)+ - (b-a)+) / 2, then relu
add_layer("gap_b", "conv", ["gap_a_relu"], out=NS, kernel=[1, 1], stride=1, padding=0)
w = np.zeros((NS, 4 * NS, 1, 1))
for i in range(NS):
    w[i, 4 * i + 0], w[i, 4 * i + 1], w[i, 4 * i + 2], w[i, 4 * i + 3] = 0.5, 0.5, -0.5, -0.5
weights["gap_b.W"] = w
weights["gap_b.b"] = np.zeros(NS)
add_layer("gap_relu", "relu", ["gap_b"])                                  # (3, 2, 28)
# windows of 10 columns with stride 4: starts 0,4,8,12,16,20 -> window 2 = cols 8..17 (centre),
# window 0 = cols 0..9 (left side), window 4 = cols 16..25 (right side)
add_layer("gap", "maxpool", ["gap_relu"], size=[1, 10], stride=[1, 4])   # (3, 2, 5)
GAP_WIN = 5

# --- row-width profile and sudden width increases going down the image -----------------------------------------
# rw(y) = width of row y / 28.  jump2 = rw[y] - rw[y-2], jump3 = rw[y] - rw[y-3], maximised over rows 3..24
# (a thin handle above a wide bag body)
add_layer("rowwidth", "avgpool", ["B"], size=[1, S], stride=[1, S])      # (1, 28, 1)
add_layer("rw", "reshape", ["rowwidth"], shape=[1, 1, S])                 # (1, 1, 28): rows laid out along x
# channels: jump2, jump3, and for the "handle" test a = jump2 - 0.43, b = rw[y-2] - 0.04, a-b, b-a
add_layer("jump_pre", "conv", ["rw"], out=6, kernel=[7, 7], stride=1, padding=3)
w = np.zeros((6, 1, 7, 7))
b = np.zeros(6)
w[0, 0, 3, 3], w[0, 0, 3, 3 - 2] = 1.0, -1.0
w[1, 0, 3, 3], w[1, 0, 3, 3 - 3] = 1.0, -1.0
w[2, 0, 3, 3], w[2, 0, 3, 3 - 2], b[2] = 1.0, -1.0, -0.57           # a: width grows by >= 16 px in 2 rows
w[3, 0, 3, 3 - 2], b[3] = 1.0, -0.04                                 # b: the row 2 above is not empty
w[4] = w[2] - w[3]; b[4] = b[2] - b[3]                               # a - b
w[5] = w[3] - w[2]; b[5] = b[3] - b[2]                               # b - a
weights["jump_pre.W"] = w
weights["jump_pre.b"] = b
add_layer("jump_relu", "relu", ["jump_pre"])                              # (6, 1, 28)
add_layer("jump_mix", "conv", ["jump_relu"], out=3, kernel=[1, 1], stride=1, padding=0)
w = np.zeros((3, 6, 1, 1))
w[0, 0], w[1, 1] = 1.0, 1.0
w[2, 2], w[2, 3], w[2, 4], w[2, 5] = 0.5, 0.5, -0.5, -0.5            # min(a+, b+): thin row above a wide body
weights["jump_mix.W"] = w
weights["jump_mix.b"] = np.zeros(3)
add_layer("jump_mix_relu", "relu", ["jump_mix"])                          # (3, 1, 28)
add_layer("jump", "maxpool", ["jump_mix_relu"], size=[1, 22], stride=[1, 3])  # (3, 1, 3); [:, 0, 1] = rows 3..24

# --- morphological closing of B (fills gaps up to 6 px wide): closing - B = area of thin gaps -------------------
# dilation = 7x7 max of B; erosion of the dilation = 1 - 7x7 max of (1 - dilation).  Zero padding is added with
# 1x1 convolutions since pooling has none.
CL = 7
add_layer("B_pad", "conv", ["B"], out=1, kernel=[1, 1], stride=1, padding=CL // 2)      # (1, 34, 34)
weights["B_pad.W"] = np.ones((1, 1, 1, 1))
weights["B_pad.b"] = np.zeros(1)
add_layer("dil", "maxpool", ["B_pad"], size=[CL, CL], stride=[1, 1])                     # (1, 28, 28)
add_layer("dil_inv_pad", "conv", ["dil"], out=1, kernel=[1, 1], stride=1, padding=CL // 2)   # 1 - dil, padded
weights["dil_inv_pad.W"] = -np.ones((1, 1, 1, 1))
weights["dil_inv_pad.b"] = np.ones(1)
add_layer("ero_inv", "maxpool", ["dil_inv_pad"], size=[CL, CL], stride=[1, 1])         # (1, 28, 28) = 1 - closing
# the same with a 3x3 window: closing7 - closing3 = area of gaps between 2 and 6 px wide (ignores speckle)
add_layer("B_pad1", "conv", ["B"], out=1, kernel=[1, 1], stride=1, padding=1)           # (1, 30, 30)
weights["B_pad1.W"] = np.ones((1, 1, 1, 1))
weights["B_pad1.b"] = np.zeros(1)
add_layer("dil3", "maxpool", ["B_pad1"], size=[3, 3], stride=[1, 1])                     # (1, 28, 28)
add_layer("dil3_inv_pad", "conv", ["dil3"], out=1, kernel=[1, 1], stride=1, padding=1)
weights["dil3_inv_pad.W"] = -np.ones((1, 1, 1, 1))
weights["dil3_inv_pad.b"] = np.ones(1)
add_layer("ero_inv3", "maxpool", ["dil3_inv_pad"], size=[3, 3], stride=[1, 1])         # (1, 28, 28) = 1 - closing3
# 5x5 erosion of B: 1 - 5x5 max of (1 - B).  Its mass is the "solid core": structures thinner than 5 px vanish.
add_layer("B_inv_pad2", "conv", ["B"], out=1, kernel=[1, 1], stride=1, padding=2)       # (1, 32, 32) = 1 - B
weights["B_inv_pad2.W"] = -np.ones((1, 1, 1, 1))
weights["B_inv_pad2.b"] = np.ones(1)
add_layer("ero5_inv", "maxpool", ["B_inv_pad2"], size=[5, 5], stride=[1, 1])            # (1, 28, 28) = 1 - erosion5

# --- transitions of the silhouette: dark->bright / bright->dark, horizontally and vertically -------------------
add_layer("edge_pre", "conv", ["B"], out=4, kernel=[3, 3], stride=1, padding=1)
w = np.zeros((4, 1, 3, 3))
w[0, 0, 1, 1], w[0, 0, 1, 0] = 1.0, -1.0       # dark -> bright going right
w[1, 0, 1, 1], w[1, 0, 1, 2] = 1.0, -1.0       # bright -> dark going right
w[2, 0, 1, 1], w[2, 0, 0, 1] = 1.0, -1.0       # dark -> bright going down
w[3, 0, 1, 1], w[3, 0, 2, 1] = 1.0, -1.0       # bright -> dark going down
weights["edge_pre.W"] = w
weights["edge_pre.b"] = np.zeros(4)
add_layer("edges", "relu", ["edge_pre"])                                  # (4, 28, 28)

# --- vertical line detail of the grey image ----------------------------------------------------------------------
# |second horizontal derivative| above 0.1, summed over a vertical run of 5 pixels, minus 0.6: responds to
# vertical lines (placket, zip, button row, open front) and not to isolated speckle or smooth shading
DETAIL_T = 0.10
add_layer("detail_pre", "conv", ["image"], out=2, kernel=[3, 3], stride=1, padding=1)
w = np.zeros((2, 1, 3, 3))
w[0, 0, 1, :] = [1.0, -2.0, 1.0]
w[1, 0, 1, :] = [-1.0, 2.0, -1.0]
weights["detail_pre.W"] = w
weights["detail_pre.b"] = np.array([-DETAIL_T, -DETAIL_T])
add_layer("detail", "relu", ["detail_pre"])                               # (2, 28, 28)
add_layer("vline_pre", "conv", ["detail"], out=1, kernel=[5, 5], stride=1, padding=2)
w = np.zeros((1, 2, 5, 5))
w[0, :, :, 2] = 1.0
weights["vline_pre.W"] = w
weights["vline_pre.b"] = np.array([-0.6])
add_layer("vline", "relu", ["vline_pre"])                                 # (1, 28, 28)

# ==============================================================================================================
# 3. linear features (functionals of the pooled tensors)
# ==============================================================================================================
SRC = {"B": S * S, "rowmax": S, "colmax": S, "gap": NS * 2 * GAP_WIN, "jump": 3 * 3, "edges": 4 * S * S,
       "vline": S * S, "ero_inv": S * S, "ero_inv3": S * S, "ero5_inv": S * S}


def feat(const=0.0, **parts):
    f = {k: np.zeros(n) for k, n in SRC.items()}
    for k, v in parts.items():
        f[k] = f[k] + v
    f["const"] = float(const)
    return f


def fsum(*terms):
    out = {k: np.zeros(n) for k, n in SRC.items()}
    out["const"] = 0.0
    for coef, f in terms:
        for k in out:
            out[k] = out[k] + coef * f[k]
    return out


ones = np.ones
F_H = feat(rowmax=ones(S))                                   # height in px
F_W = feat(colmax=ones(S))                                   # width in px
F_M = feat(B=region(0, S, 0, S))                             # silhouette mass in px
F_UL = feat(B=region(0, 14, 0, 14))                          # upper-left quadrant mass
F_UR = feat(B=region(0, 14, 14, S))                          # upper-right quadrant mass
F_L = feat(B=region(0, S, 0, 14))                            # left half mass
F_R = feat(B=region(0, S, 14, S))                            # right half mass
F_TOP5 = feat(B=region(0, 5, 0, S))                          # mass of the top five rows
j = np.zeros((3, 1, 3)); j[0, 0, 1] = 1.0
F_JUMP2 = feat(jump=j.reshape(-1))                           # largest width increase over 2 rows (fraction of 28)
j = np.zeros((3, 1, 3)); j[2, 0, 1] = 1.0
F_HANDLE = feat(jump=j.reshape(-1))                          # > 0 where a non-empty thin row sits 2 rows above a wide one
e = np.zeros((4, S, S)); e[3, 7:18, 0:8] = 1.0; e[3, 7:18, 20:S] = 1.0
F_SLEEVEEDGE = feat(edges=e.reshape(-1))                     # bright->dark going down at the sides: sleeve hems
F_WSH = feat(B=band_mean_width(3, 9))                        # mean width, shoulder / sleeve band
F_WMID = feat(B=band_mean_width(14, 22))                     # mean width, mid body band
F_WWAIST = feat(B=band_mean_width(10, 16))                   # mean width, waist band
F_WBOT = feat(B=band_mean_width(22, S))                      # mean width, hem band
F_WBODY = feat(B=band_mean_width(8, S))                      # mean width below the top rows

g = np.zeros((NS, 2, GAP_WIN)); g[:, 1, 2] = 1.0
F_CGAP = feat(gap=g.reshape(-1))                             # centre gap, lower half (sum over scales)
g = np.zeros((NS, 2, GAP_WIN)); g[:, 1, 0] = 1.0; g[:, 1, 4] = 1.0
F_SGAP = feat(gap=g.reshape(-1))                             # side gaps, lower half (both sides, all scales)

v = np.zeros((S, S)); v[4:14, 10:18] = 1.0
F_VUP = feat(vline=v.reshape(-1))                            # vertical line detail, upper centre of the body
v = np.zeros((S, S)); v[15:27, 10:18] = 1.0
F_VLOW = feat(vline=v.reshape(-1))                           # vertical line detail, lower centre of the body

# area of gaps between 2 and 6 px wide = sum(closing7 - closing3) = sum(ero_inv3) - sum(ero_inv7)
F_HOLES = feat(ero_inv3=ones(S * S), ero_inv=-ones(S * S))
# solid core = sum(erosion5(B)) = 784 - sum(ero5_inv)
F_CORE = feat(const=S * S, ero5_inv=-ones(S * S))

# ==============================================================================================================
# 4. predicates  p = clip((feature - t) / w, 0, 1)
# ==============================================================================================================
predicates = {}


def pred(name, f, t, w):
    predicates[name] = (f, float(t), float(w))


# shoes point left: heel / ankle / shaft in the upper-right quadrant, the low toe on the left; right half heavier
pred("asym", fsum((0.7, F_UR), (-1.3, F_UL), (0.9, F_R), (-1.1, F_L)), t=5, w=15)
pred("low", fsum((-1.0, F_H)), t=-18, w=3)                 # H <= 15 -> 1, H >= 18 -> 0  (sneaker height)
pred("bootH", F_H, t=15, w=3)                               # H >= 18 -> 1               (ankle boot height)
pred("tall", F_H, t=24, w=3)                                # H >= 27 -> 1 (garment or tall bag)
pred("shaft", fsum((1.0, F_R), (-1.7, F_L)), t=0, w=40)     # right half more than 1.7x the left: boot shaft
pred("sparse40", fsum((11.2, F_H), (-1.0, F_M)), t=-15, w=30)   # fill of the 28 x H box below ~40 %
pred("sparse45", fsum((12.6, F_H), (-1.0, F_M)), t=-15, w=30)   # fill below ~45 %
pred("sparse58", fsum((16.2, F_H), (-1.0, F_M)), t=-15, w=30)   # fill below ~58 %
pred("holey", fsum((1.0, F_HOLES), (-0.15, F_M)), t=0, w=15)  # 2-6 px gaps cover > 15 % of the mass: straps
pred("solid", F_CORE, t=15, w=25)                           # >= 40 px survive a 5x5 erosion: a solid body
pred("narrow", fsum((-1.0, F_W)), t=-18, w=3)               # W <= 15 -> 1, W >= 18 -> 0
pred("wide", F_W, t=18, w=3)                                # W >= 21 -> 1, W <= 18 -> 0
pred("shoulders", F_WSH, t=13, w=3)                         # shoulder band at least 16 px wide
pred("cgap", F_CGAP, t=0.5, w=0.5)                          # vertical gap at the centre of the lower half
pred("sgap", F_SGAP, t=0.5, w=0.5)                          # gaps at the sides of the lower half (hanging sleeves)
pred("short", fsum((1.0, F_WSH), (-1.0, F_WMID)), t=3, w=3)  # sleeve band 6 px wider than mid body: short sleeves
pred("flare", fsum((1.0, F_WBOT), (-1.0, F_WWAIST)), t=1, w=3)  # hem wider than the waist
pred("topnarrow", fsum((-1.0, F_TOP5)), t=-45, w=15)        # thin handle / straps at the top
pred("jump", F_JUMP2, t=0.4, w=0.12)                        # width grows by >= 15 px within 2 rows: handle -> body
pred("handle", F_HANDLE, t=0.0, w=0.03)                     # a thin non-empty row sits just above a sudden wide one
pred("widebody", F_WBODY, t=16, w=3)                        # boxy wide body below the top rows (>= 19 px)
pred("sleeveedge", F_SLEEVEEDGE, t=6, w=4)                  # horizontal lower sleeve edges at the sides (short sleeves)
pred("vup", F_VUP, t=1.5, w=1.5)                            # vertical line detail in the upper body centre
pred("vlow", F_VLOW, t=1.5, w=1.5)                          # ... and in the lower body centre

# ==============================================================================================================
# 5. gates (soft ANDs of predicates or their negations) and class evidence
# ==============================================================================================================
CLASSES = {"tshirt": 0, "trouser": 1, "pullover": 2, "dress": 3, "coat": 4, "sandal": 5, "shirt": 6,
           "sneaker": 7, "bag": 8, "boot": 9}
gates = []   # (list of (predicate, sign), {class: weight})


def gate(terms, **evidence):
    gates.append(([(t[1:], -1.0) if t.startswith("!") else (t, 1.0) for t in terms], evidence))


# --- shoes --------------------------------------------------------------------------------------------------
gate(["asym", "low", "solid", "!sparse45"], sneaker=1.0)
gate(["!asym", "low", "!tall", "solid", "!sparse45", "!handle"], sneaker=0.7, bag=0.3)   # flat-topped sneaker
gate(["asym", "bootH", "solid", "!sparse40"], boot=1.0)
gate(["asym", "shaft", "solid", "!sparse40"], boot=0.5)
gate(["asym", "!tall", "!solid"], sandal=1.0)                              # only thin parts: straps, thin sole
gate(["asym", "sparse40"], sandal=0.8)
gate(["asym", "holey", "!tall"], sandal=0.6)
# flat sandals are nearly symmetric: low, thin, with holes
gate(["!asym", "low", "!tall", "!solid"], sandal=0.8)
gate(["low", "sparse40", "!tall"], sandal=0.5)
gate(["low", "holey", "!tall"], sandal=0.5)
# --- bags (symmetric) -------------------------------------------------------------------------------------------
gate(["!asym", "!tall", "!low", "solid", "!sparse40", "!holey"], bag=1.0)
gate(["!asym", "!tall", "!low", "solid", "!sparse40", "holey"], bag=0.5, sandal=0.6)   # patterned bag or strappy sandal
gate(["!asym", "!tall", "handle", "solid", "!sparse58"], bag=1.0)         # low bag with a handle on top
gate(["!asym", "tall", "topnarrow", "widebody"], bag=1.0)                 # thin handle above a wide body
gate(["!asym", "tall", "topnarrow", "jump"], bag=1.0)                     # thin handle, sudden widening to the body
# --- trousers and dresses ---------------------------------------------------------------------------------------
gate(["tall", "!asym", "!wide", "!shoulders", "cgap"], trouser=1.0)
gate(["tall", "!asym", "narrow", "shoulders", "cgap"], trouser=0.8)
gate(["tall", "!asym", "!wide", "!narrow", "shoulders", "cgap"], trouser=0.6, coat=0.4)
gate(["tall", "!asym", "narrow", "!flare", "!topnarrow", "!short", "!shoulders"], trouser=0.4)
gate(["tall", "!asym", "narrow", "!cgap"], dress=0.7)
gate(["tall", "!asym", "narrow", "!cgap", "short", "shoulders"], tshirt=0.8, dress=-0.3)   # fitted tee
gate(["tall", "!asym", "!wide", "!cgap", "!shoulders", "flare"], dress=0.7)
gate(["tall", "!asym", "!cgap", "topnarrow", "!widebody", "!jump"], dress=0.7)
gate(["tall", "!asym", "!wide", "!narrow", "!cgap", "!sgap", "!shoulders"], dress=0.6)
# --- tops ---------------------------------------------------------------------------------------------------
gate(["tall", "!asym", "!narrow", "!cgap", "shoulders", "short"], tshirt=1.0)
gate(["tall", "!asym", "!narrow", "!cgap", "shoulders", "!short", "sleeveedge"], tshirt=0.9)   # boxy tee
gate(["tall", "!asym", "!narrow", "!cgap", "shoulders", "short", "vup", "vlow"], shirt=0.6, tshirt=-0.2)
LONG = ["tall", "!asym", "!narrow", "!topnarrow", "!short", "!cgap"]
gate(LONG + ["wide"], pullover=0.8, coat=0.3, shirt=0.3)
gate(LONG + ["!wide", "sgap"], pullover=0.7, coat=0.3, shirt=0.3)
gate(LONG + ["!wide", "!sgap", "shoulders", "!flare"], pullover=0.4, shirt=0.3, dress=0.2, coat=0.2)
gate(LONG + ["!wide", "!sgap", "shoulders", "flare"], dress=0.5, coat=0.4)
gate(LONG + ["wide", "vup", "vlow"], coat=1.0, shirt=0.9, pullover=-0.5)
gate(LONG + ["!wide", "sgap", "vup", "vlow"], shirt=1.0, coat=0.8, pullover=-0.5)
gate(["tall", "!asym", "!narrow", "!topnarrow", "!short", "cgap", "wide"], coat=1.0)
gate(["tall", "!asym", "!narrow", "!topnarrow", "!short", "cgap", "!wide", "shoulders"], coat=1.0)

CLASS_BIAS = np.zeros(10)

# ==============================================================================================================
# 6. compile: predicates -> dense layers (one per source tensor, summed), relu
# ==============================================================================================================
pred_names = list(predicates)
NP = len(pred_names)
pred_layers = []
for src, n in SRC.items():
    Wd = np.zeros((2 * NP, n))
    bd = np.zeros(2 * NP)
    for j, name in enumerate(pred_names):
        f, t, w = predicates[name]
        Wd[2 * j] = f[src] / w
        Wd[2 * j + 1] = f[src] / w
        if src == "B":                        # the bias sits on one branch only
            bd[2 * j] = (f["const"] - t) / w
            bd[2 * j + 1] = (f["const"] - t) / w - 1.0
    lname = f"pred_{src}"
    add_layer(lname, "dense", [src], out=2 * NP)
    weights[lname + ".W"] = Wd
    weights[lname + ".b"] = bd
    pred_layers.append(lname)
add_layer("pred_sum", "add", pred_layers)
add_layer("pred", "relu", ["pred_sum"])                                   # (2*NP,)

# --- gates: dense over the predicate pairs, relu -------------------------------------------------------------------
NG = len(gates)
Wg = np.zeros((2 * NG, 2 * NP))
bg = np.zeros(2 * NG)
for gi, (terms, _) in enumerate(gates):
    bias = -(len(terms) - 1)
    for name, sign in terms:
        j = pred_names.index(name)
        if sign > 0:
            Wg[2 * gi, 2 * j], Wg[2 * gi, 2 * j + 1] = 1.0, -1.0           # + p
        else:
            Wg[2 * gi, 2 * j], Wg[2 * gi, 2 * j + 1] = -1.0, 1.0           # + (1 - p)
            bias += 1.0
    Wg[2 * gi + 1] = Wg[2 * gi]
    bg[2 * gi] = bias
    bg[2 * gi + 1] = bias - 1.0
add_layer("gate_pre", "dense", ["pred"], out=2 * NG)
weights["gate_pre.W"] = Wg
weights["gate_pre.b"] = bg
add_layer("gate", "relu", ["gate_pre"])                                   # (2*NG,)

# --- class scores -----------------------------------------------------------------------------------------------
Wo = np.zeros((10, 2 * NG))
for gi, (_, evidence) in enumerate(gates):
    for cls, wgt in evidence.items():
        k = CLASSES[cls]
        Wo[k, 2 * gi] += wgt
        Wo[k, 2 * gi + 1] -= wgt
add_layer("scores", "dense", ["gate"], out=10)
weights["scores.W"] = Wo
weights["scores.b"] = CLASS_BIAS

# ==============================================================================================================
# write
# ==============================================================================================================
if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    (out / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
    np.savez(out / "weights.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in weights.items()})
    total = sum(int(np.prod(v.shape)) for v in weights.values())
    print(f"wrote {len(layers)} layers, {total:,} weights to {out}")
