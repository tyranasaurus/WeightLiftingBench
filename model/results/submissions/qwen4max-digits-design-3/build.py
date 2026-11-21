"""Hand-designed digit classifier.  Every number here comes from formulas, from
geometry I reason about, or from pen paths I type from general knowledge of how
the ten digits are written.  No training data is read (the script reads no files
at all).

Architecture overview
---------------------
1. binarise the ink at a hand-picked grey level.
2. flood-fill the background inward from the frame border with 4-connectivity,
   implemented exactly with conv+relu+add arithmetic.  Whatever background the
   flood cannot reach is a hole (a loop interior).  Counting holes and locating
   them is the single most style-invariant cue for 0/4/6/8/9.
3. stroke geometry: run counts per row/column band, long horizontal / vertical /
   diagonal segment detectors, coarse ink grid.
4. shape templates: digits I draw myself along typed pen paths, each rendered,
   normalised by itself alone, blurred and pooled; matched by correlation.
5. a hand-written rule table combines the three evidence sources.

Binary logic with the allowed ops:  AND(a,b) = relu(a+b-1) for a,b in {0,1};
clamp(x) = relu(x) - relu(x-1) saturates a non-negative integer at 1.
"""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."


class Net:
    def __init__(self):
        self.layers = []
        self.w = {}
        self.n = 0

    def name(self, tag):
        self.n += 1
        return f"{tag}{self.n}"

    def layer(self, tag, op, inputs, **settings):
        nm = self.name(tag)
        self.layers.append(dict(name=nm, op=op, inputs=list(inputs), **settings))
        return nm

    def conv(self, tag, inp, W, b, stride=1, padding=0):
        W = np.asarray(W, np.float64)
        b = np.asarray(b, np.float64)
        nm = self.name(tag)
        self.layers.append(dict(name=nm, op="conv", inputs=[inp], out=int(W.shape[0]),
                                kernel=[int(W.shape[2]), int(W.shape[3])],
                                stride=int(stride), padding=int(padding)))
        self.w[nm + ".W"] = W.astype(np.float32)
        self.w[nm + ".b"] = b.astype(np.float32)
        return nm

    def dense(self, tag, inp, W, b):
        W = np.asarray(W, np.float64)
        b = np.asarray(b, np.float64)
        nm = self.name(tag)
        self.layers.append(dict(name=nm, op="dense", inputs=[inp], out=int(W.shape[0])))
        self.w[nm + ".W"] = W.astype(np.float32)
        self.w[nm + ".b"] = b.astype(np.float32)
        return nm

    def relu(self, tag, inp):
        return self.layer(tag, "relu", [inp])

    def add(self, tag, *ins):
        return self.layer(tag, "add", list(ins))

    def reshape(self, tag, inp, shape):
        return self.layer(tag, "reshape", [inp], shape=list(shape))

    def pool(self, tag, inp, size, stride, kind="avg"):
        return self.layer(tag, kind + "pool", [inp], size=list(size), stride=list(stride))

    def norm(self, tag, inp):
        return self.layer(tag, "norm", [inp])

    # ---- small helpers -------------------------------------------------
    def c1x1(self, tag, inp, gain, bias, cin=1, cout=1):
        """per-channel 1x1 conv, gain/bias scalars or per-output-channel lists"""
        g = np.broadcast_to(np.asarray(gain, np.float64), (cout,)).copy()
        bb = np.broadcast_to(np.asarray(bias, np.float64), (cout,)).copy()
        W = np.zeros((cout, cin, 1, 1))
        for o in range(cout):
            W[o, min(o, cin - 1), 0, 0] = g[o]
        return self.conv(tag, inp, W, bb)

    def const(self, tag, value, shape_from="image", cout=1):
        """constant map, made with a zero-weight conv"""
        return self.c1x1(tag, shape_from, 0.0, value, cin=1, cout=cout)

    def clamp(self, tag, inp, cin=1):
        """relu(x) - relu(x-1): saturates a non-negative integer at 1"""
        p = self.relu(tag + "p", inp)
        q = self.c1x1(tag + "q", p, 1.0, -1.0, cin=cin, cout=cin)
        r = self.relu(tag + "r", q)
        s = self.c1x1(tag + "s", r, -1.0, 0.0, cin=cin, cout=cin)
        return self.add(tag + "o", p, s)

    def step(self, tag, inp, t, K=8.0, cin=1):
        """~1 where inp > t, ~0 below (sharp hand-set threshold)"""
        g = self.c1x1(tag + "g", inp, K, -K * t, cin=cin, cout=cin)
        return self.clamp(tag + "c", g, cin=cin)

    def neg(self, tag, inp, cin=1):
        return self.c1x1(tag + "n", inp, -1.0, 0.0, cin=cin, cout=cin)

    def total(self, tag, inp, hw=28):
        """sum of all entries -> (1,1,1)"""
        return self.conv(tag, inp, np.ones((1, 1, hw, hw)), np.zeros(1))


PLUS = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
ONES3 = np.ones((3, 3))


def binarise(net, t=0.20, K=2000.0):
    g = net.c1x1("inkg", "image", K, -K * t)
    return net.clamp("inkc", g)


def flood_holes(net, M, iters=36):
    """Returns (hole mask O, raw hole mask H).  Exact 4-connected background
    flood from the border; unreachable background = loop interiors."""
    Bg = net.c1x1("bg", M, -1.0, 1.0)
    # border indicator: a 3x3 box sum of a constant-1 map is 9 inside, <9 at the rim
    ones = net.const("ones", 1.0)
    box = net.conv("box", ones, ONES3[None, None], np.zeros(1), padding=1)
    rim = net.c1x1("rim", box, -1.0, 9.0)
    border = net.step("bord", rim, 0.5, K=1.0)
    m1 = net.const("m1", -1.0)
    R0 = net.relu("seed", net.add("seedsum", Bg, border, m1))
    # state = (P, N, Bg) with reachable R = P - N
    a = net.c1x1("stA", R0, [1.0, 0.0, 0.0], 0.0, cin=1, cout=3)
    b = net.c1x1("stB", Bg, [0.0, 0.0, 1.0], 0.0, cin=1, cout=3)
    st = net.add("st0", a, b)
    # One step:  s = (#reachable 4-neighbours) + 6*Bg - 6.
    # Bg=1 -> s = count in 0..5, so clamp(s) = OR of neighbours, masked to background.
    # Bg=0 -> s <= 4-6 < 0, so clamp(s) = 0: the flood can never enter ink.
    # clamp is carried as the pair (relu(s), relu(s-1)); their difference is the bit,
    # and the next conv reads that difference linearly, so 2 layers per step suffice.
    W = np.zeros((3, 3, 3, 3))
    W[0, 0] = PLUS
    W[0, 1] = -PLUS
    W[0, 2, 1, 1] = 6.0
    W[1, 0] = PLUS
    W[1, 1] = -PLUS
    W[1, 2, 1, 1] = 6.0
    W[2, 2, 1, 1] = 1.0
    for i in range(iters):
        st = net.conv(f"fl{i}", st, W, np.array([-6.0, -7.0, 0.0]), padding=1)
        st = net.relu(f"fr{i}", st)
    H = net.conv("holes", st, np.array([[-1.0, 1.0, 1.0]])[:, :, None, None], np.zeros(1))
    cnt = net.conv("erc", H, ONES3[None, None], np.array([-2.0]), padding=1)
    O = net.clamp("opn", cnt)          # 1 where >=3 of the 9 neighbours are hole
    return O, H


def euler_count(net, mask, tag="eu", hw=27):
    """Exact number of 8-connected components of a binary mask that has no
    holes of its own, via the 2x2 pattern formula
        chi = (n1 - n3 - 2*ndiag) / 4
    where n1/n3 count 2x2 windows with exactly one / three set pixels and
    ndiag counts the two diagonal-only windows.  For integer s in 0..4,
    1[s==k] = relu(s-k+1) - 2*relu(s-k) + relu(s-k-1)."""
    s = net.conv(tag + "s", mask, np.ones((1, 1, 2, 2)), np.zeros(1))
    six = net.conv(tag + "6", s, np.array([1., 1., 1., 1., 1., 1.])[:, None, None, None],
                   np.array([0., -1., -2., -2., -3., -4.]))
    r = net.relu(tag + "r", six)
    Wc = np.zeros((2, 6, 1, 1))
    Wc[0, :3, 0, 0] = [1., -2., 1.]
    Wc[1, 3:, 0, 0] = [1., -2., 1.]
    i13 = net.conv(tag + "i", r, Wc, np.zeros(2))
    Kd = np.zeros((2, 1, 2, 2))
    Kd[0, 0] = [[1., -1.], [-1., 1.]]
    Kd[1, 0] = [[-1., 1.], [1., -1.]]
    nd = net.relu(tag + "d", net.conv(tag + "dc", mask, Kd, np.array([-1., -1.])))
    big = np.ones((1, 1, hw, hw))
    sa = net.conv(tag + "sa", i13, np.concatenate([big, -big], axis=1), np.zeros(1))
    sb = net.conv(tag + "sb", nd, np.concatenate([big, big], axis=1), np.zeros(1))
    a = net.c1x1(tag + "ga", sa, 0.25, 0.0)
    b = net.c1x1(tag + "gb", sb, -0.5, 0.0)
    return net.add(tag + "n", a, b)


def hole_features(net, O):
    n = euler_count(net, O, "eu")
    q = net.pool("hq", O, [7, 28], [7, 28])          # (1,4,1) quarter-row means
    qa = net.c1x1("hqa", q, 196.0, 0.0)               # -> areas
    h = net.pool("hh", O, [28, 14], [28, 14])         # (1,1,2) left/right means
    ha = net.c1x1("hha", h, 392.0, 0.0)
    area = net.total("harea", O)
    return dict(n=n, quarters=qa, halves=ha, area=area)




def seg_map(net, M, L, kind):
    """1 where a straight ink run of length L is centred, in the given direction.
    Square kernel with only the relevant line filled; padding keeps 28x28."""
    K = np.zeros((1, 1, L, L))
    c = L // 2
    if kind == "h":
        K[0, 0, c, :] = 1.0
    elif kind == "v":
        K[0, 0, :, c] = 1.0
    elif kind == "d":          # top-left to bottom-right  "\"
        for i in range(L):
            K[0, 0, i, i] = 1.0
    elif kind == "b":          # straight bar: rows c and c-3 both full
        K[0, 0, c, :] = 1.0
        K[0, 0, c - 3, :] = 1.0
    elif kind == "a":          # top-right to bottom-left  "/"
        for i in range(L):
            K[0, 0, i, L - 1 - i] = 1.0
    nline = 2.0 if kind == "b" else 1.0
    s = net.conv(f"sg{kind}{L}", M, K, np.array([-nline * L + 1.0]), padding=c)
    return net.relu(f"sr{kind}{L}", s)


def stroke_features(net, M):
    """Returns an ordered list of (label, tensor, flat size)."""
    out = []
    # handwriting is wavy and slanted, so a strict straight-run detector on the
    # raw mask misses most real bars.  A one-pixel plus dilation makes a nearly
    # straight stroke continuous without bridging the gaps between strokes.
    out.append(("grid", net.pool("grid", M, [7, 7], [7, 7]), 16))       # ink fraction, 4x4
    K = np.zeros((1, 1, 3, 3)); K[0, 0, 1, 1] = 1.0; K[0, 0, 1, 0] = -1.0
    rs = net.relu("rs", net.conv("rsc", M, K, np.zeros(1), padding=1))
    rb = net.c1x1("rbs", net.pool("rb", rs, [4, 28], [4, 28]), 28.0, 0.0)
    out.append(("rowruns", rb, 7))                                      # runs per row, 7 bands
    K = np.zeros((1, 1, 3, 3)); K[0, 0, 1, 1] = 1.0; K[0, 0, 0, 1] = -1.0
    cs = net.relu("cs", net.conv("csc", M, K, np.zeros(1), padding=1))
    cb = net.c1x1("cbs", net.pool("cb", cs, [28, 4], [28, 4]), 28.0, 0.0)
    out.append(("colruns", cb, 7))
    Md = net.clamp("mdil", net.conv("mdilc", M, PLUS[None, None], np.zeros(1), padding=1))
    for L, kind in ((13, "h"), (11, "b")):
        h = seg_map(net, Md, L, kind)
        hb = net.pool(f"hband{L}", net.pool(f"hrow{L}", h, [1, 28], [1, 28]), [7, 1], [7, 1], "max")
        out.append((f"h{kind}{L}", hb, 4))                                    # bar present, 4 row bands
    v = seg_map(net, Md, 11, "v")
    vr = net.pool("vband", net.pool("vrow", v, [1, 28], [1, 28]), [7, 1], [7, 1], "max")
    out.append(("v11r", vr, 4))
    vc = net.pool("vcolband", net.pool("vcol", v, [28, 1], [28, 1]), [1, 7], [1, 7], "max")
    out.append(("v11c", vc, 4))
    v15 = seg_map(net, Md, 15, "v")
    vr15 = net.pool("v15band", net.pool("v15row", v15, [1, 28], [1, 28]), [7, 1], [7, 1], "max")
    out.append(("v15r", vr15, 4))
    for kind in ("d", "a"):
        d = seg_map(net, Md, 9, kind)
        out.append((f"diag{kind}", net.pool(f"dq{kind}", d, [14, 14], [14, 14], "max"), 4))
    return out


# --------------------------------------------------------------------------
# Shape templates.  Every path below is typed from general knowledge of how
# the digit is written (unit box: x right, y down).  Each rendering is then
# normalised by itself alone -- fit its own bounding box to 20 px keeping its
# aspect, put its own centre of mass at the frame centre -- exactly the
# documented preprocessing.  No statistic is taken across renderings.
# --------------------------------------------------------------------------

def ellipse(cx, cy, rx, ry, n=64, a0=0.0):
    t = a0 + np.arange(n) * (2 * np.pi / n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def catmull(pts, per=14, closed=False):
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t = np.linspace(0, 1, per, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2][None])
    return np.vstack(out)


NSTYLE = [4, 6, 6, 6, 4, 4, 4, 6, 4, 4]


def digit_paths(d, rng, style):
    """Polylines in the unit box.  Besides slant / width / pen, the structural
    proportions (waist height, bowl and loop size, bar position, stem angle) are
    drawn per template, because handwriting varies in structure and not just in
    style.  All ranges are general knowledge of how the digit is written."""
    U = lambda lo, hi: float(rng.uniform(lo, hi))
    j = lambda a: U(-a, a)
    if d == 0:
        rx, ry = U(0.22, 0.36), U(0.42, 0.50)
        if style == 1:
            rx *= 0.78
        pts = ellipse(0.5 + j(0.03), 0.5, rx, ry)
        if style == 2:                      # egg-shaped, wider at the bottom
            pts[:, 0] = 0.5 + (pts[:, 0] - 0.5) * (1.0 + 0.25 * (pts[:, 1] - 0.5))
        return [pts]
    if d == 1:
        x = 0.5 + j(0.05)
        main = [[x + j(0.05), 0.03], [x + j(0.03), 0.5], [x + j(0.05), 0.97]]
        strokes = [catmull(main, 10)]
        if style in (1, 3):
            strokes.append(catmull([[x, 0.05], [x - U(0.08, 0.16), 0.14],
                                    [x - U(0.16, 0.28), U(0.18, 0.26)]], 8))
        if style in (2, 3):
            w = U(0.16, 0.30)
            strokes.append(catmull([[x - w, 0.97], [x, 0.975], [x + w, 0.97]], 8))
        return strokes
    if d == 2:
        sy, ay, rx = U(0.18, 0.30), U(0.02, 0.08), U(0.72, 0.86)
        ex, by, bl = U(0.10, 0.22), U(0.90, 0.97), U(0.70, 0.90)
        top = [[0.14, sy], [0.24, ay + 0.04], [0.45, ay], [0.66, ay + 0.03],
               [rx, 0.20], [rx - 0.06, 0.38], [0.52, 0.58], [0.26, 0.80], [ex, by]]
        if style == 1:                      # looped head
            strokes = [catmull(top[:6] + [[0.60, 0.44], [0.40, 0.40], [0.34, 0.26],
                                          [0.45, 0.14], [0.62, 0.16]], 12),
                       catmull([[0.52, 0.58], [0.26, 0.80], [ex, by]], 12)]
        else:
            strokes = [catmull(top, 12)]
        strokes.append(catmull([[ex, by + 0.005], [0.45, by + 0.01], [bl, by]], 10))
        return strokes
    if d == 3:
        wy, wx = U(0.40, 0.55), U(0.38, 0.55)
        tr, br = U(0.70, 0.85), U(0.72, 0.88)
        ty, ey = U(0.08, 0.20), U(0.82, 0.95)
        if style == 2:                      # flat-topped three
            pts = [[0.16, 0.05], [U(0.62, 0.80), 0.05], [wx, wy - 0.06],
                   [wx + 0.20, wy + 0.02], [br, 0.66], [br - 0.10, ey - 0.06],
                   [0.46, ey], [0.18, ey - 0.09]]
        else:
            pts = [[0.15, ty], [0.34, 0.04], [0.60, 0.05], [tr, 0.20],
                   [tr - 0.12, wy - 0.07], [wx, wy], [wx + 0.20, wy + 0.06],
                   [br, 0.70], [br - 0.13, ey - 0.07], [0.44, ey], [0.16, ey - 0.10]]
        return [catmull(pts, 12)]
    if d == 4:
        by, sx, lx = U(0.55, 0.72), U(0.60, 0.76), U(0.08, 0.20)
        if style in (0, 1, 2):              # open top
            a = [[U(0.24, 0.44), 0.04], [lx + 0.09, 0.34], [lx, by],
                 [0.50, by], [U(0.82, 0.92), by - 0.01]]
            b = [[sx + j(0.03), 0.05], [sx, 0.5], [sx - 0.01, 0.97]]
        else:                               # closed triangle
            a = [[sx, 0.04], [lx + 0.20, 0.34], [lx, by], [0.50, by], [0.88, by]]
            b = [[sx, 0.04], [sx, 0.5], [sx, 0.97]]
        return [catmull(a, 12), catmull(b, 10)]
    if d == 5:
        tr, lx = U(0.68, 0.88), U(0.20, 0.30)
        vy, br, ey = U(0.36, 0.48), U(0.74, 0.88), U(0.86, 0.97)
        top = [[tr, 0.045], [0.50, 0.04], [lx, 0.05], [lx - 0.03, 0.24], [lx - 0.04, vy]]
        bowl = [[lx - 0.04, vy], [0.45, vy - 0.06], [br - 0.10, vy + 0.04],
                [br, 0.66], [br - 0.14, ey - 0.08], [0.42, ey], [0.18, ey - 0.09]]
        if style == 2:
            bowl = [[lx - 0.04, vy], [0.50, vy - 0.04], [br, vy + 0.10],
                    [br - 0.02, 0.74], [0.58, ey], [0.28, ey], [0.14, ey - 0.12]]
        return [catmull(top, 12), catmull(bowl, 12)]
    if d == 6:
        cy, rx, ry = U(0.66, 0.78), U(0.24, 0.32), U(0.20, 0.28)
        tx = U(0.62, 0.84)
        stem = [[tx, 0.04], [0.52, 0.20], [0.32, 0.44], [0.5 - rx, cy - 0.06]]
        if style == 2:
            stem = [[tx, 0.04], [0.44, 0.26], [0.28, 0.52], [0.5 - rx, cy - 0.04]]
        return [catmull(stem, 12), ellipse(0.5, cy, rx, ry, 56, a0=np.pi * 0.85)]
    if d == 7:
        by, ex, ey = U(0.03, 0.10), U(0.30, 0.55), U(0.92, 0.98)
        lx = U(0.10, 0.20)
        bar = [[lx, by + j(0.03)], [0.45, by - 0.01], [U(0.80, 0.90), by]]
        leg = [[0.86, by], [0.66, 0.42], [ex, ey]]
        strokes = [catmull(bar, 10), catmull(leg, 12)]
        if style == 2:
            cy = U(0.46, 0.58)
            strokes.append(catmull([[0.32, cy], [0.50, cy - 0.005], [0.68, cy]], 8))
        if style == 3:
            strokes[0] = catmull([[lx + 0.03, by + 0.14], [lx, by + 0.02],
                                  [0.45, by - 0.01], [0.86, by]], 10)
        return strokes
    if d == 8:
        uy, urx, ury = U(0.22, 0.30), U(0.18, 0.28), U(0.18, 0.26)
        ly, lrx, lry = U(0.68, 0.76), U(0.24, 0.34), U(0.22, 0.30)
        if style == 2:
            urx, ury, uy = urx * 0.78, ury * 0.82, 0.235
            lrx, lry, ly = lrx * 1.05, lry * 1.08, 0.70
        up = ellipse(0.5 + j(0.03), uy, urx, ury, 56)
        lo = ellipse(0.5 + j(0.03), ly, lrx, lry, 56)
        return [up, lo]
    if d == 9:
        cy, rx, ry = U(0.22, 0.34), U(0.22, 0.32), U(0.20, 0.28)
        loop = ellipse(0.5 + j(0.03), cy, rx, ry, 56)
        sx = 0.5 + rx
        if style == 2:
            stem = [[sx, cy], [sx - 0.03, 0.60], [0.62, 0.86], [U(0.36, 0.50), 0.96]]
        elif style == 3:
            stem = [[sx, cy], [sx + 0.01, 0.60], [sx - 0.01, 0.97]]
        else:
            stem = [[sx, cy], [sx - 0.05, 0.62], [U(0.46, 0.72), 0.97]]
        return [loop, catmull(stem, 12)]
    raise ValueError(d)


def _resample(pts, step=0.7):
    d = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(1))
    keep = [0]
    acc = 0.0
    for i, v in enumerate(d):
        acc += v
        if acc >= step:
            keep.append(i + 1); acc = 0.0
    keep.append(len(pts) - 1)
    return pts[sorted(set(keep))]


def _stamp(dist, pts, r):
    H, W = dist.shape
    R = int(np.ceil(r + 1))
    for x, y in pts:
        xi, yi = int(np.floor(x)), int(np.floor(y))
        x0, x1 = max(0, xi - R), min(W, xi + R + 1)
        y0, y1 = max(0, yi - R), min(H, yi + R + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dd = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        sub = dist[y0:y1, x0:x1]
        np.minimum(sub, dd, out=sub)


def render_digit(strokes, pen, shear, xscale, yscale):
    """Render, then normalise this one drawing by itself alone: own bounding box
    to 20 px (aspect kept), own centre of mass to the frame centre."""
    S, off = 40.0, 48.0
    pts = []
    for st in strokes:
        u = st[:, 0] - 0.5
        v = st[:, 1] - 0.5
        X = off + (u * xscale + shear * (-v)) * S
        Y = off + v * yscale * S
        pts.append(_resample(np.stack([X, Y], 1)))
    dist = np.full((96, 96), 1e6)
    for p in pts:
        _stamp(dist, p, pen)
    ink = np.clip(pen + 0.5 - dist, 0, 1)
    m = ink > 0.05
    ys, xs = np.where(m)
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    k = 40.0 / max(bh, bw)
    tot = ink.sum()
    cy = float((ink * np.mgrid[0:96, 0:96][0]).sum() / tot)
    cx = float((ink * np.mgrid[0:96, 0:96][1]).sum() / tot)
    d2 = np.full((56, 56), 1e6)
    for p in pts:
        _stamp(d2, (p - [cx, cy]) * k + 27.0, pen * k)
    ink2 = np.clip(pen * k + 0.5 - d2, 0, 1)
    return ink2.reshape(28, 2, 28, 2).mean(axis=(1, 3))


def gauss_kernel(sigma=1.1, rad=3):
    a = np.arange(-rad, rad + 1, dtype=float)
    g = np.exp(-0.5 * (a / sigma) ** 2)
    g /= g.sum()
    return np.outer(g, g)


def blur28(img28, G):
    p = np.pad(img28, 3, mode="constant")
    b = np.zeros((28, 28))
    for dy in range(7):
        for dx in range(7):
            b += G[dy, dx] * p[dy:dy + 28, dx:dx + 28]
    return b


def unit_var(v):
    v = np.asarray(v, np.float64).ravel()
    v = v - v.mean()
    return v / np.sqrt((v ** 2).mean() + 1e-5)


def normalise_vec(img28, G):
    """coarse view: blur, 2x2 average, zero-mean/unit-variance over 196 slots"""
    return unit_var(blur28(img28, G).reshape(14, 2, 14, 2).mean(axis=(1, 3)))


def normalise_vec_fine(img28, G):
    """fine view: blur only, zero-mean/unit-variance over the 784 slots"""
    return unit_var(blur28(img28, G))


def build_templates(per_class=128, seed=20261119):
    """Draw each template, normalise it by itself alone, and store both views.
    Nothing is averaged or measured across drawings."""
    rng = np.random.default_rng(seed)
    G = gauss_kernel()
    coarse, fine = [], []
    for d in range(10):
        for t in range(per_class):
            style = t % NSTYLE[d]
            strokes = digit_paths(d, rng, style)
            pen = float(rng.uniform(2.4, 5.6))
            shear = float(rng.uniform(-0.40, 0.40))
            xscale = float(rng.uniform(0.72, 1.28))
            yscale = float(rng.uniform(0.92, 1.06))
            img = render_digit(strokes, pen, shear, xscale, yscale)
            coarse.append(normalise_vec(img, G))
            fine.append(normalise_vec_fine(img, G))
    return np.asarray(coarse), np.asarray(fine)


def rule_bank(net, groups, conds, k=4.0):
    """groups: list of (tensor, flat size).  conds: list of (coef dict, threshold)
    where the coef dict maps (group index, flat index) -> coefficient.
    Returns a binary condition vector built with dense+relu only."""
    R = len(conds)
    # the threshold bias belongs to the condition, not to each group that
    # happens to feed it -- charge it to the first contributing group only
    owner = []
    for co, thr in conds:
        gs = sorted({g for (g, i) in co})
        owner.append(gs[0] if gs else 0)
    pre = []
    for gi, (t, sz) in enumerate(groups):
        W = np.zeros((R, sz)); b = np.zeros(R)
        for ri, (co, thr) in enumerate(conds):
            if owner[ri] == gi:
                b[ri] = -k * thr
            for (g, i), v in co.items():
                if g == gi:
                    W[ri, i] = k * v
        if np.any(W) or np.any(b):
            pre.append(net.dense(f"cnd{gi}", t, W, b))
    p = pre[0] if len(pre) == 1 else net.add("cndsum", *pre)
    h = net.relu("cndh", p)
    h2 = net.relu("cndh2", net.dense("cndsh", h, np.eye(R), -np.ones(R)))
    return net.add("cndbin", h, net.dense("cndneg", h2, -np.eye(R), np.zeros(R)))


def topo_branch(net, hf):
    groups = [(hf["n"], 1), (hf["quarters"], 4), (hf["halves"], 2), (hf["area"], 1)]
    N, Q, A = 0, 1, 3
    conds = [
        ({(N, 0): 1.0}, 0.5),                                  # c0 has a hole
        ({(N, 0): 1.0}, 1.5),                                  # c1 two holes
        ({(Q, 1): 1.0}, 10.0),                                 # c2 hole in rows 7-13
        ({(Q, 2): 1.0}, 10.0),                                 # c3 hole in rows 14-20
        ({(A, 0): 1.0}, 30.0),                                 # c4 big hole
        ({(Q, 0): 1.0, (Q, 1): 1.0, (Q, 2): -2.0, (Q, 3): -2.0}, 4.0),   # c5 top heavy
        ({(Q, 0): -2.0, (Q, 1): -2.0, (Q, 2): 1.0, (Q, 3): 1.0}, 4.0),   # c6 bottom heavy
        ({(A, 0): -1.0}, -22.0),                               # c7 small hole
        ({(Q, 3): 1.0}, 5.0),                                  # c8 hole reaches bottom
    ]
    c = rule_bank(net, groups, conds)
    R = len(conds)
    # stage 2: conjunctions.  weight -1 on a literal means "not it".
    W2 = np.zeros((5, R)); b2 = np.zeros(5)
    # AND of k literals with m negated: relu(sum - (k-1)), and each negated
    # literal adds a constant 1, so the bias is m - (k-1).
    W2[0, [0, 1, 2, 3, 5, 6]] = [1, -1, 1, 1, -1, -1]; b2[0] = -2   # ZERO (k=6,m=3)
    W2[1, [0, 1, 5]] = [1, -1, 1]; b2[1] = -1                  # NINE0 (k=3,m=1)
    W2[2, [0, 1, 6]] = [1, -1, 1]; b2[2] = -1                  # SIX0  (k=3,m=1)
    W2[3, [0, 1, 7]] = [1, -1, 1]; b2[3] = -1                  # FOURH0 (k=3,m=1)
    W2[4, 0] = -1; b2[4] = 1                                   # NOHOLE
    r2 = net.relu("topor2", net.dense("topoand", c, W2, b2))
    # stage 3: remove the zero case from the nine / six cases
    W3 = np.zeros((3, 5)); b3 = np.zeros(3)
    W3[0, 1] = 1; W3[0, 0] = -1; b3[0] = 0                     # k=2, m=1
    W3[1, 2] = 1; W3[1, 0] = -1; b3[1] = 0
    W3[2, 3] = 1; W3[2, 0] = -1; b3[2] = 0
    r3 = net.relu("topor3", net.dense("topoand2", r2, W3, b3))
    # final evidence vector: c0, c1, NOHOLE, ZERO, NINE, SIX, FOURH
    Wsel = np.zeros((7, 5)); bsel = np.zeros(7)
    Wsel[2, 4] = 1; Wsel[3, 0] = 1
    Wsel3 = np.zeros((7, 3)); Wsel3[4, 0] = 1; Wsel3[5, 1] = 1; Wsel3[6, 2] = 1
    ev = net.add("topoev", net.dense("toposel", r2, Wsel, bsel),
                 net.dense("toposel3", r3, Wsel3, np.zeros(7)))
    # c0 and c1 come straight from the condition vector
    Wc = np.zeros((7, R)); Wc[0, 0] = 1; Wc[1, 1] = 1
    ev = net.add("topoev2", ev, net.dense("topoc", c, Wc, np.zeros(7)))
    # hand-written bonus table, rows = evidence, cols = classes 0..9
    T = np.array([
        # 0     1     2     3     4     5     6     7     8     9
        [ .05, -.12, -.10, -.10,  .00, -.10,  .05, -.12,  .05,  .05],   # has a hole
        [-.10, -.05, -.05, -.05, -.05, -.05, -.10, -.05,  .45, -.10],   # two holes -> 8
        [-.32,  .05,  .04,  .04,  .02,  .04, -.16,  .05, -.18, -.16],   # no hole
        [ .40, -.05, -.05, -.05, -.05, -.05, -.12, -.05, -.12, -.12],   # central big hole -> 0
        [-.12, -.08, -.06, -.06,  .10, -.06, -.18, -.08, -.08,  .32],   # top hole -> 9 (or 4)
        [-.10, -.08, -.06, -.06, -.06, -.06,  .34, -.08, -.08, -.15],   # bottom hole -> 6
        [-.04, -.04, -.04, -.04,  .16, -.04,  .04, -.04,  .10,  .06],   # small hole -> 4 / 8
    ])
    return net.dense("topoout", ev, T.T, np.zeros(10))


def hinge_bank(net, groups, specs):
    """specs: (coef dict {(group, flat index): v}, threshold, direction).
    Unit = relu(direction * (coef . feature - threshold)).  One-sided: with no
    evidence the unit is exactly 0, so no class collects a constant bonus."""
    R = len(specs)
    owner = []
    for co, thr, dr in specs:
        gs = sorted({g for (g, i) in co})
        owner.append(gs[0] if gs else 0)
    pre = []
    for gi, (t, sz) in enumerate(groups):
        W = np.zeros((R, sz)); b = np.zeros(R)
        for ri, (co, thr, dr) in enumerate(specs):
            if owner[ri] == gi:
                b[ri] = -dr * thr
            for (g, i), v in co.items():
                if g == gi:
                    W[ri, i] = dr * v
        if np.any(W) or np.any(b):
            pre.append(net.dense(f"hng{gi}", t, W, b))
    p = pre[0] if len(pre) == 1 else net.add("hngsum", *pre)
    return net.relu("hngout", p)


# Hand-written stroke table.  Each entry rewards or penalises a class only when
# the corresponding geometric evidence is actually present.  Thresholds come
# from where the strokes of each digit sit inside the 20 px normalised box.
def stroke_table(net, groups):
    idx, off = {}, 0
    for nm, t, sz in groups:
        idx[nm] = off; off += sz
    GI = {nm: gi for gi, (nm, t, sz) in enumerate(groups)}

    def F(nm, k): return (GI[nm], k)   # (group, index within that group)
    def G(r, c): return F("grid", 4 * r + c)
    def RR(b): return F("rowruns", b)
    def CR(b): return F("colruns", b)
    def H7(b): return F("hb11", b)
    def H11(b): return F("hh13", b)
    def V11C(b): return F("v11c", b)
    def DA(q): return F("diaga", q)
    def V15(b): return F("v15r", b)
    def V11R(b): return F("v11r", b)

    specs, rows = [], []
    def unit(f, thr=0.0, dr=1.0):
        specs.append(({f: 1.0}, thr, dr)); return len(specs) - 1
    def many(f, thr): return unit(f, thr, 1.0)
    def few(f, thr): return unit(f, thr, -1.0)
    def rule(u, w):
        rows.append((u, w))

    # long horizontal bar (>=13 px), by row quarter (0: rows 0-6 ... 3: rows 21-27)
    rule(unit(H11(0)), {7: .20, 5: .16, 2: .04, 1: -.10, 0: -.06, 3: -.04, 6: -.04, 9: -.04, 8: -.04})
    rule(unit(H11(1)), {4: .05, 7: .04})
    rule(unit(H11(2)), {4: .28, 7: .06, 2: -.06, 3: -.06, 5: -.04, 0: -.05, 8: -.05, 9: -.05, 6: -.04, 1: -.04})
    rule(unit(H11(3)), {2: .28, 3: -.10, 5: -.06, 7: -.08, 0: -.05, 8: -.05, 9: -.05, 6: -.04, 4: -.04})
    rule(unit(H7(0)), {7: .05, 5: .05, 2: .02})
    rule(unit(H7(2)), {4: .06})
    rule(unit(H7(3)), {2: .07})
    # long vertical stem (>=15 px), by row quarter of its centre
    for b in (1, 2):
        rule(unit(V15(b)), {1: .20, 4: .07, 7: -.08, 2: -.05, 3: -.05, 5: -.03, 0: -.06, 8: -.06, 6: -.03, 9: -.03})
        rule(unit(V11C(b)), {1: .05, 4: .04, 9: .03, 7: -.05})
        rule(unit(V11R(b)), {1: .10, 4: .04, 7: -.06, 3: -.04, 2: -.04, 5: -.03})
    # several separate strokes crossing a row band, versus a single stroke
    rule(many(RR(2), 1.5), {9: .10, 4: .10, 0: .08, 8: .08, 1: -.10, 7: -.08, 3: -.05, 2: -.05, 5: -.03, 6: -.03})
    rule(many(RR(3), 1.5), {0: .12, 8: .12, 9: .06, 4: .06, 1: -.14, 7: -.10, 3: -.06, 2: -.04})
    rule(many(RR(5), 1.4), {0: .08, 8: .08, 6: .05, 1: -.10, 7: -.08, 3: -.04, 2: -.04})
    # ink reaching the far left / right columns: wide digits, not a bare 1
    for b in (0, 6):
        rule(many(CR(b), 0.5), {0: .06, 8: .04, 2: .04, 7: .04, 4: .03})
    # "/" diagonals: the leg of a 7, the back of a 2, the slope of a 4
    rule(unit(DA(1)), {7: .08, 2: .05, 4: .05, 1: .02, 0: -.03, 8: -.03, 3: -.02})
    rule(unit(DA(2)), {2: .06, 7: .03, 4: .04})
    # ink far to the left in the lower middle: the crossbar of a 4, the foot of a 2
    rule(many(G(2, 0), 0.05), {4: .55, 2: .15})
    rule(many(G(3, 0), 0.05), {2: .45, 4: .10})

    W = np.zeros((10, len(specs)))
    for u, w in rows:
        for c, v in w.items():
            W[c, u] += v
    # these stroke heuristics are weaker evidence than the topological facts,
    # so they carry half the weight of the topology table
    W *= 0.5
    return net.dense("stout", hinge_bank(net, [(t, sz) for _, t, sz in groups], specs), W, np.zeros(10))


def build_network(per_class=128):
    net = Net()
    M = binarise(net)
    O, H = flood_holes(net, M)
    hf = hole_features(net, O)
    sf = stroke_features(net, M)

    # template branch: same blur / pool / normalisation as the templates get
    G = gauss_kernel()
    blur = net.conv("tblur", "image", G[None, None], np.zeros(1), padding=3)
    pooled = net.pool("tpool", blur, [2, 2], [2, 2])
    nrm = net.norm("tnorm", pooled)
    fine = net.norm("tnormf", blur)
    T, Tf = build_templates(per_class=per_class)
    parts = []
    for tag, feat, mat, npos, wt in (("c", nrm, T, 196, 0.5), ("f", fine, Tf, 784, 0.5)):
        sc = net.dense("td" + tag, feat, mat, np.zeros(mat.shape[0]))
        mx = net.reshape("ts" + tag, sc, [1, 10, per_class])
        bp = net.pool("tm" + tag, mx, [1, per_class], [1, per_class], "max")
        # norm gives unit variance over the npos positions, so the dot product
        # is npos * correlation; divide to get a correlation in [-1, 1], the
        # same units as the hand-written bonus tables below.
        parts.append(net.c1x1("tsc" + tag, bp, wt / npos, 0.0))
    best = net.add("tscale", *parts)

    topo = net.reshape("tpr", topo_branch(net, hf), [1, 10, 1])
    strk = net.reshape("str", stroke_table(net, sf), [1, 10, 1])
    tot = net.add("final", best, topo, strk)
    net.reshape("scores", tot, [10])
    return net


if __name__ == "__main__":
    net = build_network()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "network.json"), "w") as fh:
        json.dump({"layers": net.layers}, fh)
    np.savez(os.path.join(OUT, "weights.npz"), **net.w)
    print("layers", len(net.layers), "weights", sum(v.size for v in net.w.values()))
