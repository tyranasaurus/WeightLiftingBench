"""Hand-designed classifier for 32x32 colour photos.  python3 build.py OUT_DIR  writes network.json and weights.npz.

Every number here is typed by hand from general knowledge of colours, image geometry and the ten classes.
Nothing is read from files and nothing is fitted to data.

Structure
  1. colour branch  : three 1x1 conv+relu layers turn each pixel's RGB into soft colour categories
                      (blue, light/sky blue, white, vegetation green, warm/brown, vivid red, vivid yellow, dark,
                      neutral grey, saturation, luminance) using colour-cone thresholds chosen by hand; dense layers
                      take the fraction of each category in fixed regions (top band, bottom band, frame, centre).
  2. smoothness     : Sobel gradient energy; fraction of "smooth" pixels (sky, walls, blurred backgrounds) in the
                      frame, the centre, the top and the bottom band.
  3. structure      : on contrast-normalised luminance, central differences and the same differences smeared over
                      7 px along the edge (long straight edges keep their strength, texture loses it), and the
                      balance of horizontal against vertical edges.
  4. horizon        : strongest full-width horizontal luminance step.
  5. cells / mask   : 4x4-pixel cell colours: frame uniformity, centre-vs-frame and top-vs-bottom colour differences;
                      an object mask (pixels far from the frame's mean colour) giving object size and its spread
                      along x versus along y.
  6. readout        : each scalar cue goes through hand-placed knots (piecewise linear) and each class adds
                      w * log P(cue | class) from hand-typed belief tables (naive Bayes with hand-set tables).
"""
import json, sys, os
import numpy as np

CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
H = W = 32


class Net:
    def __init__(self):
        self.layers, self.w = [], {}

    def add(self, name, op, inputs, **kw):
        self.layers.append(dict(name=name, op=op, inputs=list(inputs), **kw))
        return name

    def conv(self, name, inp, Wt, b, pad, stride=1):
        Wt = np.asarray(Wt, np.float64); k = Wt.shape[2]
        self.w[name + ".W"], self.w[name + ".b"] = Wt, np.asarray(b, np.float64)
        return self.add(name, "conv", [inp], out=int(Wt.shape[0]), kernel=[k, k], stride=stride, padding=pad)

    def dense(self, name, inp, Wt, b):
        Wt = np.asarray(Wt, np.float64)
        self.w[name + ".W"], self.w[name + ".b"] = Wt, np.asarray(b, np.float64)
        return self.add(name, "dense", [inp], out=int(Wt.shape[0]))

    def relu(self, name, inp):
        return self.add(name, "relu", [inp])


# ----------------------------------------------------------------------------------------------------------------
# 1x1 "unit" layers written symbolically: each unit = relu(sum coef * previous unit + bias)
class UnitLayer:
    def __init__(self, prev):
        self.prev = list(prev); self.names = []; self.rows = []; self.bias = []; self.wd = {}

    def unit(self, name, lin, bias=0.0):
        row = np.zeros(len(self.prev))
        for k, v in lin.items():
            row[self.prev.index(k)] += v
        self.names.append(name); self.rows.append(row); self.bias.append(bias)
        return name

    def ramp(self, tag, lin, t0, w):
        """two units relu(e - t0), relu(e - t0 - w); next layer reads clamp((e - t0) / w) = (a - b) / w"""
        self.unit(tag + "_a", lin, -t0); self.unit(tag + "_b", lin, -t0 - w)
        self.wd[tag] = w

    def matrix(self):
        return np.array(self.rows), np.array(self.bias)


def clamp_of(layer, tag):
    """linear expression (over layer's units) equal to clamp((e - t0) / w, 0, 1)"""
    w = layer.wd[tag]
    return {tag + "_a": 1.0 / w, tag + "_b": -1.0 / w}


def lsum(*ds):
    out = {}
    for sgn, d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + sgn * v
    return out


def colour_branch(net):
    R, G, B = {"R": 1.0}, {"G": 1.0}, {"B": 1.0}
    L = {"R": 1 / 3, "G": 1 / 3, "B": 1 / 3}

    def d(a, b):
        return {a: 1.0, b: -1.0}

    l1 = UnitLayer(["R", "G", "B"])
    # sky / water blue: B clearly above R, not far below G, not black
    l1.ramp("blue1", d("B", "R"), 0.03, 0.08)
    l1.ramp("blue2", d("B", "G"), -0.06, 0.06)
    l1.ramp("lumlo", L, 0.10, 0.10)
    # light enough to be sky rather than water
    l1.ramp("lumbl", L, 0.42, 0.13)
    # bright (for white)
    l1.ramp("lumhi", L, 0.60, 0.15)
    # vegetation green: G above R and clearly above B
    l1.ramp("green1", d("G", "R"), 0.0, 0.04)
    l1.ramp("green2", d("G", "B"), 0.02, 0.06)
    # warm (brown, tan, orange, wood, fur, skin): R above B, R not below G, but not vivid red
    l1.ramp("warm1", d("R", "B"), 0.04, 0.10)
    l1.ramp("warm2", d("R", "G"), -0.03, 0.04)
    l1.ramp("warm3", d("R", "G"), 0.30, 0.12)
    # vivid red paint
    l1.ramp("red1", d("R", "G"), 0.28, 0.12)
    l1.ramp("red2", d("R", "B"), 0.25, 0.12)
    # vivid yellow paint
    l1.ramp("yel1", d("G", "B"), 0.30, 0.12)
    l1.ramp("yel2", d("R", "B"), 0.30, 0.12)
    l1.ramp("yel3", d("R", "G"), 0.12, 0.10)
    l1.ramp("yel4", d("G", "R"), 0.12, 0.10)
    # dark: L below 0.08 -> 1, above 0.18 -> 0
    l1.ramp("dark", {"R": -1 / 3, "G": -1 / 3, "B": -1 / 3}, -0.18, 0.10)
    # luminance window for neutral grey
    l1.ramp("gl1", L, 0.10, 0.08)
    l1.ramp("gl2", L, 0.65, 0.10)
    # absolute channel differences (for saturation = max - min = half their sum)
    for a, b in (("R", "G"), ("G", "B"), ("B", "R")):
        l1.unit(f"ad_{a}{b}", d(a, b)); l1.unit(f"ad_{b}{a}", d(b, a))
    l1.unit("lum", L)

    l2 = UnitLayer(l1.names)
    c = lambda tag: clamp_of(l1, tag)
    l2.unit("BLUE", lsum((1, c("blue1")), (1, c("blue2")), (1, c("lumlo"))), -2.0)
    l2.unit("GREEN", lsum((1, c("green1")), (1, c("green2"))), -1.0)
    l2.unit("WARM", lsum((1, c("warm1")), (1, c("warm2")), (-1, c("warm3"))), -1.0)
    l2.unit("RED", lsum((1, c("red1")), (1, c("red2"))), -1.0)
    l2.unit("YELLOW", lsum((1, c("yel1")), (1, c("yel2")), (-1, c("yel3")), (-1, c("yel4"))), -1.0)
    l2.unit("DARK", c("dark"))
    sat = {n: 0.5 for n in l1.names if n.startswith("ad_")}
    l2.unit("SAT", sat)
    l2.ramp("satw", sat, 0.07, 0.08)
    l2.ramp("satg", sat, 0.05, 0.06)
    l2.unit("lumhi_c", c("lumhi")); l2.unit("gl1_c", c("gl1")); l2.unit("gl2_c", c("gl2")); l2.unit("lumbl_c", c("lumbl"))
    l2.unit("L", {"lum": 1.0})

    l3 = UnitLayer(l2.names)
    c2 = lambda tag: clamp_of(l2, tag)
    out_names = ["BLUE", "WHITE", "GREEN", "WARM", "RED", "YELLOW", "DARK", "GRAY", "SAT", "L", "LBLUE"]
    for n in out_names:
        if n == "WHITE":
            l3.unit("WHITE", lsum((1, {"lumhi_c": 1.0}), (-1, c2("satw"))))
        elif n == "LBLUE":
            l3.unit("LBLUE", {"BLUE": 1.0, "lumbl_c": 1.0}, -1.0)
        elif n == "GRAY":
            l3.unit("GRAY", lsum((1, {"gl1_c": 1.0}), (-1, {"gl2_c": 1.0}), (-1, c2("satg"))))
        else:
            l3.unit(n, {n: 1.0})

    x = "image"
    for i, lay in enumerate((l1, l2, l3), 1):
        Wm, bm = lay.matrix()
        x = net.conv(f"col{i}", x, Wm[:, :, None, None], bm, pad=0)
        x = net.relu(f"col{i}r", x)
    return x, out_names, (l1, l2, l3)


# ----------------------------------------------------------------------------------------------------------------
def region(y0, y1, x0, x1):
    m = np.zeros((H, W)); m[y0:y1 + 1, x0:x1 + 1] = 1.0
    return m


ALL = region(0, 31, 0, 31)
TOP = region(0, 7, 0, 31)
BOT = region(24, 31, 0, 31)
CENTER = region(8, 23, 8, 23)
MIDDLE = region(4, 27, 4, 27)
FRAME = ALL - region(4, 27, 4, 27)
TAREA = region(4, 27, 4, 27)          # 9x9-filter maps: away from zero-padding artefacts
SFRAME = region(1, 30, 1, 30) - region(6, 25, 6, 25)   # 3x3-filter maps: skip only the outermost pixel
STOP = region(1, 7, 1, 30)
SBOT = region(24, 30, 1, 30)
DKLOW = region(20, 29, 0, 31)
DKHIGH = region(2, 11, 0, 31)


def mean_mask(m):
    return m / m.sum()


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0


def kernels_9():
    """unsmoothed central differences and their 7-pixel smear along the edge direction.
    For white noise the smeared response is 1/sqrt(7) = 0.38 of the plain one; for a long straight edge it is 1."""
    gx = np.zeros((9, 9)); gx[4, 3], gx[4, 5] = -0.5, 0.5
    gy = gx.T.copy()
    hl = np.zeros((9, 9))
    for s in range(-3, 4):
        hl[3, 4 + s] -= 0.5 / 7; hl[5, 4 + s] += 0.5 / 7
    vl = hl.T.copy()
    return gx, gy, hl, vl


FEATURES = ["bt", "bbl", "bbd", "wt", "wb", "gr", "gd", "wc", "wf", "rd", "yl", "gy", "dkb", "tb", "tbc",
            "sf", "sc", "stb", "Sn", "En", "GYn", "GXn", "hz", "fu", "cs", "ms", "MX", "MY"]
FI = {n: i for i, n in enumerate(FEATURES)}
NA = len(FEATURES)


def build_network():
    net = Net()
    # ---- colour branch -> fractions of pixels of each colour category in each region
    col, cn, _ = colour_branch(net)
    Wc = np.zeros((NA, len(cn), H, W))
    ch = {n: i for i, n in enumerate(cn)}

    def put(f, chan, mask, s=1.0):
        Wc[FI[f], ch[chan]] += s * mean_mask(mask)

    put("bt", "BLUE", TOP); put("wt", "WHITE", TOP); put("wb", "WHITE", BOT)
    put("bbl", "LBLUE", BOT); put("bbd", "BLUE", BOT); put("bbd", "LBLUE", BOT, -1)
    put("gr", "GREEN", ALL); put("gd", "GREEN", FRAME); put("gd", "GREEN", CENTER, -1)
    put("wc", "WARM", CENTER); put("wf", "WARM", FRAME)
    put("rd", "RED", MIDDLE); put("yl", "YELLOW", MIDDLE); put("gy", "GRAY", ALL)
    put("dkb", "DARK", DKLOW); put("dkb", "DARK", DKHIGH, -1)
    put("tb", "L", TOP); put("tb", "L", BOT, -1)
    colA = net.dense("colA", col, Wc.reshape(NA, -1), np.zeros(NA))

    # ---- luminance, raw and contrast-normalised
    lum = net.conv("lum", "image", np.full((1, 3, 1, 1), 1 / 3), [0.0], pad=0)
    lumn = net.add("lumn", "norm", [lum])

    # ---- smooth pixels: Sobel gradient energy |gx|+|gy| below 0.015 -> 1, above 0.035 -> 0 (sky, walls, blur)
    Ka = np.stack([SOBEL_X, -SOBEL_X, SOBEL_X.T, -SOBEL_X.T])[:, None]
    ta = net.relu("texar", net.conv("texa", lum, Ka, np.zeros(4), pad=1))
    sm = net.relu("smr", net.conv("sm", ta, -np.ones((2, 4, 1, 1)), [0.035, 0.015], pad=0))
    Ws = np.zeros((NA, 2, H, W))
    for f, mk, s in (("sf", SFRAME, 1), ("sc", CENTER, 1), ("stb", STOP, 1), ("stb", SBOT, -1)):
        Ws[FI[f], 0] += s * mean_mask(mk) / 0.02; Ws[FI[f], 1] -= s * mean_mask(mk) / 0.02
    smA = net.dense("smA", sm, Ws.reshape(NA, -1), np.zeros(NA))

    # ---- normalised texture: |gx|, |gy|, |long horizontal|, |long vertical|
    gx, gy, hl, vl = kernels_9()
    Kn = np.stack([gx, -gx, gy, -gy, hl, -hl, vl, -vl])[:, None]
    tn = net.relu("texnr", net.conv("texn", lumn, Kn, np.zeros(8), pad=4))
    Wn = np.zeros((NA, 8, H, W)); m = mean_mask(TAREA)
    for k in (0, 1, 2, 3):
        Wn[FI["En"], k] = m
    for k in (4, 5, 6, 7):
        Wn[FI["Sn"], k] = m
    Wn[FI["GXn"], 0] = Wn[FI["GXn"], 1] = m
    Wn[FI["GYn"], 2] = Wn[FI["GYn"], 3] = m
    texnA = net.dense("texnA", tn, Wn.reshape(NA, -1), np.zeros(NA))

    # ---- horizon: max over rows of |mean over the row of (2 rows below - 2 rows above)|
    rows = list(range(3, 29))
    Wh = np.zeros((2 * len(rows), H, W))
    for j, y in enumerate(rows):
        for dy, s in ((1, .5), (2, .5), (-1, -.5), (-2, -.5)):
            Wh[j, y + dy, 2:30] += s / 28.0
    Wh[len(rows):] = -Wh[:len(rows)]
    hz = net.relu("hzr", net.dense("hz", lumn, Wh.reshape(2 * len(rows), -1), np.zeros(2 * len(rows))))
    hz = net.add("hzs", "reshape", [hz], shape=[1, 2 * len(rows), 1])
    hz = net.add("hzm", "maxpool", [hz], size=[2 * len(rows), 1], stride=[1, 1])
    Whz = np.zeros((NA, 1)); Whz[FI["hz"], 0] = 1.0
    hzA = net.dense("hzA", hz, Whz, np.zeros(NA))

    # ---- 4x4-pixel cells: frame uniformity, centre/frame and top/bottom colour differences
    cells = net.add("cells", "avgpool", ["image"], size=[4, 4], stride=[4, 4])   # (3, 8, 8)
    fr = np.zeros((8, 8), bool); fr[0, :] = fr[7, :] = fr[:, 0] = fr[:, 7] = True
    ce = np.zeros((8, 8), bool); ce[2:6, 2:6] = True
    tp = np.zeros((8, 8), bool); tp[0:2, :] = True
    bt = np.zeros((8, 8), bool); bt[6:8, :] = True
    fidx = [(y, x) for y in range(8) for x in range(8) if fr[y, x]]
    nf = len(fidx)
    rowsF, tags = [], []
    for c in range(3):
        fm = np.zeros((3, 8, 8)); fm[c][fr] = 1.0 / nf
        for (y, x) in fidx:
            r = -fm.copy(); r[c, y, x] += 1.0
            rowsF += [r.ravel(), -r.ravel()]; tags += ["fu", "fu"]
    for c in range(3):
        r = np.zeros((3, 8, 8)); r[c][ce] = 1.0 / ce.sum(); r[c][fr] -= 1.0 / nf
        rowsF += [r.ravel(), -r.ravel()]; tags += ["cs", "cs"]
        r = np.zeros((3, 8, 8)); r[c][tp] = 1.0 / tp.sum(); r[c][bt] -= 1.0 / bt.sum()
        rowsF += [r.ravel(), -r.ravel()]; tags += ["tbc", "tbc"]
    WF = np.array(rowsF)
    fs = net.relu("fsr", net.dense("fs", cells, WF, np.zeros(len(WF))))
    WfA = np.zeros((NA, len(WF)))
    for j, t in enumerate(tags):
        WfA[FI[t], j] = 1.0 / nf if t == "fu" else 1.0
    fsA = net.dense("fsA", fs, WfA, np.zeros(NA))

    # ---- object mask: pixels whose colour differs from the frame's mean colour (sum over channels of |diff|
    #      above 0.24 -> 1, below 0.12 -> 0); its size and its spread along x and along y about the image centre
    Wb = np.zeros((3, H, W, 3, 8, 8))
    for c in range(3):
        Wb[c, :, :, c][:, :, fr] = -1.0 / nf
    fmb = net.dense("fmb", cells, Wb.reshape(3 * H * W, -1), np.zeros(3 * H * W))
    fmr = net.add("fmr", "reshape", [fmb], shape=[3, H, W])
    dev = net.add("dev", "add", ["image", fmr])
    Kd = np.concatenate([np.eye(3), -np.eye(3)])[:, :, None, None]
    devr = net.relu("devr", net.conv("devc", dev, Kd, np.zeros(6), pad=0))
    msk = net.relu("mskr", net.conv("msk", devr, np.ones((2, 6, 1, 1)), [-0.12, -0.24], pad=0))
    yy, xx = np.mgrid[0:H, 0:W]
    Wm = np.zeros((NA, 2, H, W))
    for f, g in (("ms", np.ones((H, W))), ("MX", (xx - 15.5) ** 2 / 100), ("MY", (yy - 15.5) ** 2 / 100)):
        Wm[FI[f], 0] = g / (H * W) / 0.12; Wm[FI[f], 1] = -g / (H * W) / 0.12
    mskA = net.dense("mskA", msk, Wm.reshape(NA, -1), np.zeros(NA))

    A = net.add("A", "add", [colA, smA, texnA, hzA, fsA, mskA])
    return net, A


# ----------------------------------------------------------------------------------------------------------------
# Belief tables.  For each cue: knots (or ratio steps), a weight, and for every class the relative chance of the cue
# falling at each knot (rows normalised per class, so they act as P(cue bin | class)).  Typed from general knowledge.
#            plane  car   bird   cat   deer   dog   frog  horse  ship  truck
TABLES = """
bt  knots 0 0.15 0.45 0.8   w 1.0
 plane .22 .14 .26 .38 | car .65 .16 .12 .07 | bird .50 .16 .17 .17 | cat .76 .12 .07 .05 | deer .72 .14 .09 .05
 dog .76 .12 .07 .05 | frog .88 .08 .03 .01 | horse .58 .17 .15 .10 | ship .30 .18 .26 .26 | truck .52 .17 .17 .14
bbl knots 0 0.15 0.45 0.8   w 1.0
 plane .45 .13 .17 .25 | car .88 .07 .03 .02 | bird .70 .12 .10 .08 | cat .88 .07 .03 .02 | deer .92 .05 .02 .01
 dog .88 .07 .03 .02 | frog .92 .05 .02 .01 | horse .90 .06 .03 .01 | ship .55 .18 .16 .11 | truck .88 .07 .03 .02
bbd knots 0 0.15 0.45 0.8   w 1.0
 plane .80 .10 .06 .04 | car .87 .08 .03 .02 | bird .80 .10 .06 .04 | cat .87 .08 .03 .02 | deer .90 .06 .03 .01
 dog .87 .08 .03 .02 | frog .90 .06 .03 .01 | horse .90 .06 .03 .01 | ship .35 .17 .23 .25 | truck .87 .08 .03 .02
wb  knots 0 0.2 0.5 0.85   w 0.8
 plane .45 .15 .17 .23 | car .64 .16 .12 .08 | bird .70 .14 .09 .07 | cat .70 .15 .09 .06 | deer .88 .07 .03 .02
 dog .70 .15 .09 .06 | frog .85 .09 .04 .02 | horse .82 .10 .05 .03 | ship .62 .20 .11 .07 | truck .66 .16 .11 .07
wt  knots 0 0.2 0.5 0.85   w 1.0
 plane .42 .16 .17 .25 | car .62 .18 .12 .08 | bird .60 .15 .12 .13 | cat .62 .18 .12 .08 | deer .75 .12 .08 .05
 dog .65 .17 .10 .08 | frog .80 .10 .06 .04 | horse .65 .17 .10 .08 | ship .50 .18 .15 .17 | truck .52 .20 .14 .14
tbc knots 0.03 0.1 0.25 0.45   w 0.7
 plane .25 .28 .27 .20 | car .12 .30 .35 .23 | bird .18 .32 .30 .20 | cat .18 .35 .30 .17 | deer .12 .30 .33 .25
 dog .16 .33 .31 .20 | frog .22 .38 .27 .13 | horse .10 .25 .33 .32 | ship .14 .27 .32 .27 | truck .10 .27 .35 .28
gr  knots 0 0.1 0.3 0.6   w 1.0
 plane .66 .17 .11 .06 | car .58 .22 .14 .06 | bird .40 .20 .21 .19 | cat .62 .20 .12 .06 | deer .24 .20 .28 .28
 dog .45 .20 .19 .16 | frog .42 .22 .20 .16 | horse .36 .20 .25 .19 | ship .68 .18 .09 .05 | truck .55 .22 .15 .08
gd  knots -0.2 0 0.15 0.4   w 0.3
 plane .05 .78 .11 .06 | car .06 .72 .14 .08 | bird .10 .50 .22 .18 | cat .05 .76 .12 .07 | deer .08 .34 .28 .30
 dog .06 .55 .21 .18 | frog .20 .52 .16 .12 | horse .06 .38 .28 .28 | ship .04 .80 .10 .06 | truck .05 .72 .15 .08
wc  knots 0 0.15 0.4 0.7   w 1.0
 plane .58 .20 .13 .09 | car .42 .25 .19 .14 | bird .22 .28 .30 .20 | cat .22 .26 .30 .22 | deer .12 .20 .33 .35
 dog .12 .18 .30 .40 | frog .14 .20 .30 .36 | horse .13 .18 .32 .37 | ship .52 .22 .15 .11 | truck .40 .25 .20 .15
wf  knots 0 0.15 0.4 0.7   w 0.8
 plane .60 .20 .12 .08 | car .42 .25 .19 .14 | bird .30 .25 .25 .20 | cat .24 .25 .27 .24 | deer .22 .25 .28 .25
 dog .26 .25 .26 .23 | frog .14 .20 .30 .36 | horse .25 .25 .27 .23 | ship .60 .20 .12 .08 | truck .40 .25 .20 .15
rd  knots 0 0.05 0.15 0.4   w 1.0
 plane .76 .12 .08 .04 | car .52 .14 .14 .20 | bird .78 .13 .06 .03 | cat .78 .13 .06 .03 | deer .78 .13 .06 .03
 dog .74 .14 .08 .04 | frog .82 .11 .05 .02 | horse .62 .18 .13 .07 | ship .72 .14 .09 .05 | truck .46 .15 .17 .22
yl  knots 0 0.05 0.2   w 1.0
 plane .86 .10 .04 | car .83 .10 .07 | bird .86 .10 .04 | cat .90 .08 .02 | deer .91 .07 .02
 dog .89 .08 .03 | frog .86 .10 .04 | horse .90 .08 .02 | ship .88 .09 .03 | truck .75 .13 .12
gy  knots 0 0.1 0.3 0.6   w 1.0
 plane .15 .20 .33 .32 | car .15 .25 .35 .25 | bird .28 .30 .26 .16 | cat .18 .28 .32 .22 | deer .38 .32 .21 .09
 dog .30 .32 .24 .14 | frog .35 .31 .23 .11 | horse .32 .32 .24 .12 | ship .16 .25 .33 .26 | truck .16 .25 .34 .25
dkb knots -0.1 0 0.08 0.2   w 0.8
 plane .12 .58 .20 .10 | car .08 .30 .34 .28 | bird .15 .48 .25 .12 | cat .15 .45 .25 .15 | deer .12 .48 .27 .13
 dog .15 .45 .25 .15 | frog .15 .45 .25 .15 | horse .12 .45 .28 .15 | ship .14 .52 .24 .10 | truck .08 .32 .33 .27
tb  knots -0.15 0 0.1 0.25   w 0.6
 plane .14 .32 .30 .24 | car .10 .27 .35 .28 | bird .15 .35 .30 .20 | cat .22 .36 .27 .15 | deer .15 .35 .30 .20
 dog .20 .36 .28 .16 | frog .20 .40 .27 .13 | horse .12 .30 .33 .25 | ship .10 .25 .35 .30 | truck .12 .28 .33 .27
sf  knots 0 0.2 0.5 0.8   w 1.0
 plane .08 .17 .30 .45 | car .20 .30 .30 .20 | bird .08 .17 .32 .43 | cat .22 .30 .30 .18 | deer .40 .32 .19 .09
 dog .25 .30 .28 .17 | frog .42 .32 .18 .08 | horse .30 .33 .24 .13 | ship .12 .25 .33 .30 | truck .22 .30 .30 .18
sc  knots 0 0.2 0.5 0.8   w 0.6
 plane .15 .25 .32 .28 | car .25 .38 .27 .10 | bird .22 .33 .30 .15 | cat .22 .33 .30 .15 | deer .22 .36 .28 .14
 dog .22 .33 .30 .15 | frog .45 .33 .16 .06 | horse .20 .33 .30 .17 | ship .22 .35 .28 .15 | truck .28 .38 .25 .09
stb knots -0.3 0 0.2 0.5   w 0.5
 plane .12 .40 .28 .20 | car .22 .38 .25 .15 | bird .15 .40 .28 .17 | cat .15 .45 .26 .14 | deer .12 .38 .30 .20
 dog .15 .43 .27 .15 | frog .12 .50 .26 .12 | horse .10 .33 .32 .25 | ship .10 .32 .31 .27 | truck .15 .38 .28 .19
st  ratio Sn En 0.58 0.66 0.74 0.82 delta 0.03   w 0.8
 plane .08 .18 .30 .27 .17 | car .06 .15 .28 .31 .20 | bird .15 .28 .30 .19 .08 | cat .12 .26 .31 .21 .10
 deer .17 .30 .29 .17 .07 | dog .12 .26 .31 .21 .10 | frog .20 .32 .28 .14 .06 | horse .13 .27 .30 .20 .10
 ship .07 .17 .29 .29 .18 | truck .05 .14 .28 .31 .22
hv  ratio GYn GXn 0.85 1.0 1.2 1.5 delta 0.03   w 0.6
 plane .14 .21 .28 .22 .15 | car .10 .18 .28 .26 .18 | bird .16 .24 .29 .20 .11 | cat .16 .25 .29 .19 .11
 deer .22 .27 .27 .16 .08 | dog .16 .25 .29 .19 .11 | frog .15 .25 .30 .19 .11 | horse .15 .24 .28 .20 .13
 ship .10 .18 .27 .26 .19 | truck .17 .24 .28 .19 .12
hz  knots 0.25 0.6 1.0 1.5   w 0.6
 plane .25 .32 .26 .17 | car .20 .32 .29 .19 | bird .28 .33 .25 .14 | cat .30 .34 .24 .12 | deer .22 .35 .28 .15
 dog .29 .34 .25 .12 | frog .35 .35 .20 .10 | horse .20 .33 .30 .17 | ship .14 .28 .32 .26 | truck .20 .32 .29 .19
fu  knots 0.04 0.1 0.2 0.32   w 0.5
 plane .30 .30 .25 .15 | car .13 .26 .33 .28 | bird .18 .30 .31 .21 | cat .14 .29 .32 .25 | deer .09 .25 .35 .31
 dog .14 .29 .32 .25 | frog .12 .29 .34 .25 | horse .09 .25 .35 .31 | ship .12 .27 .34 .27 | truck .11 .24 .33 .32
cs  knots 0.03 0.1 0.2 0.35   w 0.6
 plane .20 .30 .30 .20 | car .14 .28 .32 .26 | bird .17 .30 .32 .21 | cat .24 .34 .27 .15 | deer .30 .37 .23 .10
 dog .18 .30 .30 .22 | frog .35 .37 .20 .08 | horse .25 .35 .27 .13 | ship .20 .33 .30 .17 | truck .18 .30 .30 .22
ms  knots 0.05 0.2 0.45 0.7   w 0.6
 plane .25 .35 .25 .15 | car .08 .25 .37 .30 | bird .30 .36 .22 .12 | cat .10 .25 .35 .30 | deer .12 .28 .33 .27
 dog .10 .25 .35 .30 | frog .12 .26 .34 .28 | horse .10 .25 .35 .30 | ship .16 .32 .30 .22 | truck .08 .24 .36 .32
el  ratio MX MY 0.7 1.0 1.5 2.3 delta 0.005   w 0.7
 plane .06 .14 .25 .30 .25 | car .05 .15 .30 .30 .20 | bird .16 .30 .32 .15 .07 | cat .16 .30 .32 .15 .07
 deer .14 .28 .32 .17 .09 | dog .16 .30 .32 .15 .07 | frog .10 .24 .34 .22 .10 | horse .06 .16 .30 .28 .20
 ship .05 .12 .26 .30 .27 | truck .08 .20 .32 .25 .15
"""


def parse_tables(text):
    cues, cur = [], None
    for line in text.strip().splitlines():
        if not line.startswith(" "):
            t = line.split(); name = t[0]
            wi = t.index("w"); weight = float(t[wi + 1])
            if t[1] == "knots":
                cur = dict(name=name, kind="knots", knots=[float(v) for v in t[2:wi]], w=weight, p={})
            else:
                di = t.index("delta")
                cur = dict(name=name, kind="ratio", num=t[2], den=t[3], steps=[float(v) for v in t[4:di]],
                           delta=float(t[di + 1]), w=weight, p={})
            cues.append(cur)
        else:
            for part in line.split("|"):
                t = part.split()
                if t:
                    cur["p"][t[0]] = np.array([float(v) for v in t[1:]])
    for cu in cues:
        nb = len(cu["knots"]) if cu["kind"] == "knots" else len(cu["steps"]) + 1
        assert set(cu["p"]) == set(CLASSES), cu["name"]
        for k in CLASSES:
            assert len(cu["p"][k]) == nb, (cu["name"], k)
            cu["p"][k] = cu["p"][k] / cu["p"][k].sum()
    return cues


def readout(net, A, cues):
    """For every segment i of every cue: z_i linear in A; units relu(z_i), relu(z_i - 1); their difference is the
    clamped position inside the segment.  Class score = sum_cues w * (log p_0 + sum_i (log p_{i+1} - log p_i) clamp_i)."""
    zrows, zb, segs = [], [], []        # segs: (cue, segment index)
    for cu in cues:
        if cu["kind"] == "knots":
            t = cu["knots"]; f = FI[cu["name"]]
            for i in range(len(t) - 1):
                r = np.zeros(NA); r[f] = 1.0 / (t[i + 1] - t[i])
                zrows.append(r); zb.append(-t[i] / (t[i + 1] - t[i])); segs.append((cu, i))
        else:
            for i, a in enumerate(cu["steps"]):
                r = np.zeros(NA); r[FI[cu["num"]]] = 1.0 / cu["delta"]; r[FI[cu["den"]]] = -a / cu["delta"]
                zrows.append(r); zb.append(0.5); segs.append((cu, i))
    Z = np.array(zrows); zb = np.array(zb)
    WB = np.concatenate([Z, Z]); bB = np.concatenate([zb, zb - 1.0])
    Bz = net.relu("Br", net.dense("B", A, WB, bB))
    n = len(segs)
    Ws = np.zeros((10, 2 * n)); bs = np.zeros(10)
    for k, cname in enumerate(CLASSES):
        for cu in cues:
            bs[k] += cu["w"] * np.log(cu["p"][cname][0])
        for j, (cu, i) in enumerate(segs):
            lp = np.log(cu["p"][cname])
            dv = cu["w"] * (lp[i + 1] - lp[i])
            Ws[k, j] += dv; Ws[k, n + j] -= dv
    net.dense("scores", Bz, Ws, bs)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    net, A = build_network()
    readout(net, A, parse_tables(TABLES))
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": net.layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in net.w.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
