"""Writes a hand-designed clothing classifier: python3 build.py OUT_DIR  ->  OUT_DIR/network.json, OUT_DIR/weights.npz

Every number below is typed by hand or computed from formulas (shape models, filter kernels, region masks).
No data is read.
"""
import json, math, sys
from pathlib import Path
import numpy as np

N = 28
SS = 4  # supersampling used when drawing shape models

# =====================================================================================================
# 1. Hand-written shape models (silhouettes), drawn as polygons from general knowledge of the garments.
# =====================================================================================================

def poly_mask(poly, n=N * SS):
    p = np.asarray(poly, float) * SS
    ys, xs = np.mgrid[0:n, 0:n] + 0.5
    inside = np.zeros((n, n), bool)
    x0, y0 = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for a, b, c, d in zip(x0, y0, x1, y1):
        if b == d:
            continue
        cond = ((b <= ys) & (ys < d)) | ((d <= ys) & (ys < b))
        xint = a + (ys - b) * (c - a) / (d - b)
        inside ^= cond & (xs < xint)
    return inside


def fit(polys):
    """Scale so the longest side of the joint bounding box is 28 pixels and centre it (as the shop photos were)."""
    allp = np.concatenate([np.asarray(p, float) for p in polys])
    lo, hi = allp.min(0), allp.max(0)
    s = 28.0 / max(hi - lo)
    c = (lo + hi) / 2
    return [(np.asarray(p, float) - c) * s + 14.0 for p in polys]


def render(polys):
    m = np.zeros((N * SS, N * SS), bool)
    for p in fit(polys):
        m |= poly_mask(p)
    return m.reshape(N, SS, N, SS).mean((1, 3))


def mirror(right):
    left = [(-x, y) for (x, y) in reversed(right)]
    return list(right) + left


def arc(cx, cy, rx, ry, a0, a1, n=12):
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in np.linspace(a0, a1, n)]


def tees():
    """Short-sleeved tops: body rectangle, neck dip, short sleeves angled down."""
    out = []
    for hb in (0.27, 0.32, 0.37):
        for nh, nd in ((0.11, 0.03), (0.15, 0.07)):
            for sl in (0.12, 0.2, 0.28):
                for th in (25, 42):
                    for ay in (0.24, 0.32):
                        t = math.radians(th)
                        S = (hb + 0.02, 0.05)
                        A = (S[0] + sl * math.cos(t), S[1] + sl * math.sin(t))
                        so = 0.19
                        B = (A[0] - so * math.sin(t), A[1] + so * math.cos(t))
                        P = (hb, max(ay, B[1] + 0.02))
                        right = [(0, nd), (nh, 0), S, A, B, P, (hb + 0.02, 1.0), (0, 1.0)]
                        out.append([mirror(right)])
    return out


def tanks():
    """Sleeveless tops: straps, deep neck, straight body."""
    out = []
    for th in (0.13, 0.19):
        for nd in (0.14, 0.24):
            for hb in (0.24, 0.3, 0.36):
                for ay in (0.2, 0.3):
                    nh = th - 0.055
                    right = [(0, nd), (nh, 0), (th, 0), (th + 0.02, ay * 0.5), (hb - 0.02, ay * 0.85), (hb, ay),
                             (hb + 0.03, 1.0), (0, 1.0)]
                    out.append([mirror(right)])
    return out


def longsleeves(kind="pullover"):
    """Long-sleeved tops seen flat from the front: rounded shoulders that reach nearly the full width, sleeves
    hanging almost straight down beside the body to the cuffs, optional gap between sleeve and body.
    kind: 'pullover' (round neck dip), 'coat' (narrower, collar or hood standing up), 'shirt' (collar, small dip)."""
    out = []
    hbs = (0.24, 0.28) if kind == "coat" else (0.28, 0.33)
    for hb in hbs:
        for sw in (0.11, 0.15):
            for sp in (0.0, 0.03):
                for flare in (0.0, 0.05):
                    for r in (0.06, 0.13):
                        for cy in (0.9, 1.0):
                            Wc = hb + sp + sw
                            Ws = Wc - flare
                            if kind == "pullover":
                                top = [(0, 0.05), (0.07, 0.035), (0.12, 0.0)]
                            elif kind == "shirt":
                                top = [(0, 0.03), (0.04, 0.0), (0.12, 0.0)]
                            else:
                                top = [(0, -0.04), (0.1, -0.04), (0.13, 0.0)] if r < 0.1 else \
                                      [(0, -0.09), (0.08, -0.08), (0.13, -0.02), (0.15, 0.0)]
                            right = top + [(Ws - r, 0.01), (Ws - 0.3 * r, 0.01 + 0.3 * r), (Ws, 0.01 + r),
                                           (Wc, cy), (hb + sp, cy), (hb, 0.35), (hb, 1.0), (0, 1.0)]
                            out.append([mirror(right)])
    return out


def trousers():
    """Waistband on top, two legs with a gap that opens downward from the crotch."""
    out = []
    for wh in (0.17, 0.21, 0.25, 0.29):
        for cy in (0.3, 0.4):
            for lw in (0.13, 0.17, 0.21):
                for gw in (0.02, 0.06, 0.11):
                    poly = [(-wh, 0), (wh, 0), (wh + 0.01, cy), (gw / 2 + lw, 1.0), (gw / 2, 1.0), (0.0, cy + 0.02),
                            (-gw / 2, 1.0), (-gw / 2 - lw, 1.0), (-wh - 0.01, cy)]
                    out.append([poly])
    return out


def dresses():
    """Bodice or straps on top (narrower than the skirt), waist, skirt falling straight or widening to the hem;
    sleeveless, cap sleeves or narrow long sleeves."""
    out = []
    for th in (0.12, 0.16, 0.2):
        for nd in (0.04, 0.11):
            for wr in (0.8, 1.0):
                for bh in (0.2, 0.27, 0.34, 0.42):
                    for sleeve in ("none", "cap", "long"):
                        if sleeve == "long" and nd > 0.1:
                            continue
                        nh = th - 0.06
                        wh = th * wr
                        right = [(0, nd), (nh, 0), (th, 0)]
                        if sleeve == "cap":
                            right += [(th + 0.06, 0.05), (th + 0.05, 0.15)]
                        elif sleeve == "long":
                            right += [(th + 0.06, 0.04), (th + 0.1, 0.45), (th + 0.03, 0.47), (th, 0.2)]
                        right += [(wh, 0.35), (bh, 1.0), (0, 1.0)]
                        out.append([mirror(right)])
    return out


def bags():
    """Rectangular or trapezoid body, with or without a handle arch on top (the arch is filled in the silhouette).
    Bags without a handle are drawn wider than tall (clutches, shoulder bags)."""
    out = []
    for bh in (0.4, 0.55, 0.7):
        for tr in (0.85, 1.0):
            out.append([[(-0.5 * tr, 0), (0.5 * tr, 0), (0.5, bh), (-0.5, bh)]])
    for bh in (0.55, 0.75, 1.0, 1.25):
        for tr in (0.75, 0.9, 1.0):
            body = [(-0.5 * tr, 0), (0.5 * tr, 0), (0.5, bh), (-0.5, bh)]
            for hs in (0.2, 0.32):
                for hh in (0.2, 0.35, 0.5):
                    handle = arc(0, 0.01, hs, hh, math.pi, 2 * math.pi, 14)
                    out.append([body, handle])
    return out


def flip(poly):
    return [(x, -y) for (x, y) in poly]


def sneakers():
    """Low shoe seen from the side, toe on the left: flat sole, upper rising from the toe to the heel collar."""
    out = []
    for h in (0.3, 0.36, 0.42, 0.5, 0.58):
        for tr in (0.35, 0.5):
            for xc in (0.75, 0.88):
                for bend in (0.25, 0.45):
                    th = h * tr
                    poly = [(0.03, 0), (0, th * 0.45), (0.05, th), (0.3, th + (h - th) * bend),
                            (0.58, th + (h - th) * (0.35 + bend)), (xc, h), (0.97, h * 0.93), (1.0, h * 0.6), (1.0, 0)]
                    out.append([flip(poly)])
    return out


def boots():
    """Ankle boot: toe on the left, tall shaft on the right, instep curving up between them, flat sole or a heel
    with an arch gap."""
    out = []
    for h in (0.65, 0.75, 0.87, 1.0, 1.15):
        for th in (0.2, 0.28, 0.36):
            for sf in (0.4, 0.5, 0.6):
                for heel in (None, 0.15):
                    top = [(0.0, th * 0.5), (0.04, th), (0.18, th + 0.04), (sf - 0.08, th + 0.45 * (h - th)),
                           (sf, h), (1.0, h)]
                    if heel is None:
                        bottom = [(1.0, 0), (0.03, 0)]
                    else:
                        bottom = [(1.0, 0), (0.8, 0), (0.78, heel), (0.6, heel * 0.8), (0.42, 0.02), (0.03, 0)]
                    out.append([flip(top + bottom)])
    return out


def sandals():
    """Sandals as their filled outline: flat sole with strap dome, heeled sandal (toe strap, arch, heel post,
    ankle strap), and wedge with straps."""
    out = []
    for hs in (0.1, 0.17, 0.25):
        for a, b in ((0.1, 0.7), (0.2, 0.88)):
            for back in (None, 0.28):
                sole = [(0, 0), (0, 0.05), (1, 0.06), (1, 0)]
                dome = [(a, 0.04)] + arc((a + b) / 2, 0.04, (b - a) / 2, hs, 0, math.pi, 12)[::-1] + [(b, 0.04)]
                polys = [flip(sole), flip(dome)]
                if back:
                    polys.append(flip([(0.86, 0.04), (0.9, back), (1.0, back), (1.0, 0.04)]))
                out.append(polys)
    for ah in (0.3, 0.45, 0.62):
        for hw in (0.05, 0.12):
            for st in (0.18, 0.32):
                for ts in (0.14, 0.24):
                    fore = [(0, 0), (0, 0.06), (0.36, 0.07), (0.36, 0)]
                    toe = [(0.02, 0.05)] + arc(0.19, 0.05, 0.17, ts, 0, math.pi, 10)[::-1] + [(0.36, 0.05)]
                    archl = [(0.34, 0.0), (0.34, 0.07), (0.83, ah + 0.04), (0.83, ah - 0.03)]
                    post = [(0.95 - hw, 0), (0.95, 0), (0.97, ah), (0.8, ah)]
                    ankle = [(0.74, ah), (0.76, ah + st), (1.0, ah + st), (1.0, ah - 0.05)]
                    out.append([flip(p) for p in (fore, toe, archl, post, ankle)])
    # heeled sandal as its filled outline: straps span the space between toe strap, sole line and ankle strap,
    # only the gap under the arch stays open
    for ah in (0.3, 0.45, 0.62):
        for st in (0.18, 0.32):
            for ts in (0.14, 0.24):
                for xa in (0.62, 0.74):
                    out.append([flip([(0, 0.0), (0.36, 0.0), (0.8, ah - 0.04), (0.84, 0.0), (0.95, 0.0),
                                      (1.0, ah + st), (xa, ah + st), (0.3, ts + 0.04), (0.06, ts), (0.0, 0.05)])])
    for wh in (0.22, 0.34):
        for hs in (0.15, 0.25):
            for back in (0.3, 0.45):
                wedge = [(0, 0), (0, 0.08), (0.45, 0.1), (1.0, wh), (1.0, 0)]
                dome = [(0.08, 0.07)] + arc(0.4, 0.08, 0.32, hs, 0, math.pi, 12)[::-1] + [(0.72, 0.1)]
                ank = [(0.78, wh), (0.82, wh + back * 0.6), (1.0, wh + back * 0.6), (1.0, wh)]
                out.append([flip(p) for p in (wedge, dome, ank)])
    return out


def family(fn, offset=0.0):
    return [(render(p), offset) for p in fn()]


def class_templates():
    """Per class: list of (silhouette, offset). Offsets (in pixels of agreement) make a class that borrows another
    class's shapes pay a little for them."""
    tee, tank = family(tees), family(tanks)
    # pullovers, coats and shirts share one outline family (round neck, collar or hood; wide or narrow); what tells
    # them apart is inside the outline (zip line, pattern), handled by the evidence rules
    lng = sum((family(lambda k=k: longsleeves(k)) for k in ("pullover", "coat", "shirt")), [])
    off = lambda fam, o: [(t, o) for t, _ in fam]
    return {
        0: tee + tank,                                  # T-shirt / top: short sleeves or sleeveless
        1: family(trousers),
        2: lng,                                         # pullover
        3: family(dresses),
        4: lng,                                         # coat
        5: family(sandals) + off(family(sneakers), -10) + off(family(boots), -10),   # straps fill a shoe outline
        6: lng + off(tee, -8),                          # shirt: long or short sleeves
        7: family(sneakers),
        8: family(bags),
        9: family(boots),
    }


# =====================================================================================================
# 2. Network construction helpers
# =====================================================================================================

class Net:
    def __init__(self):
        self.layers, self.w, self.shape = [], {}, {"image": (1, N, N)}

    def _add(self, L, shape):
        self.layers.append(L)
        self.shape[L["name"]] = shape
        return L["name"]

    def conv(self, name, inp, W, b, pad):
        W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
        c, h, w = self.shape[inp]
        o, ci, k, _ = W.shape
        assert ci == c, (name, ci, c)
        self.w[name + ".W"], self.w[name + ".b"] = W, b
        ho, wo = h + 2 * pad - k + 1, w + 2 * pad - k + 1
        return self._add({"name": name, "op": "conv", "inputs": [inp], "out": int(o), "kernel": [k, k], "stride": 1,
                          "padding": pad}, (o, ho, wo))

    def dense(self, name, inp, W, b):
        W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
        assert W.shape[1] == int(np.prod(self.shape[inp])), (name, W.shape, self.shape[inp])
        self.w[name + ".W"], self.w[name + ".b"] = W, b
        return self._add({"name": name, "op": "dense", "inputs": [inp], "out": int(W.shape[0])}, (W.shape[0],))

    def relu(self, name, inp):
        return self._add({"name": name, "op": "relu", "inputs": [inp]}, self.shape[inp])

    def reshape(self, name, inp, shape):
        assert int(np.prod(shape)) == int(np.prod(self.shape[inp]))
        return self._add({"name": name, "op": "reshape", "inputs": [inp], "shape": list(shape)}, tuple(shape))

    def avgpool(self, name, inp, size, stride):
        c, h, w = self.shape[inp]
        return self._add({"name": name, "op": "avgpool", "inputs": [inp], "size": list(size), "stride": list(stride)},
                         (c, (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1))

    def maxpool(self, name, inp, size, stride):
        c, h, w = self.shape[inp]
        return self._add({"name": name, "op": "maxpool", "inputs": [inp], "size": list(size), "stride": list(stride)},
                         (c, (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1))

    def add(self, name, inps):
        return self._add({"name": name, "op": "add", "inputs": list(inps)}, self.shape[inps[0]])


def onehot_1x1(rows, cin):
    """rows: list of dicts {in_channel: coef}; returns 1x1 conv weight (out, cin, 1, 1)."""
    W = np.zeros((len(rows), cin, 1, 1))
    for o, r in enumerate(rows):
        for i, v in r.items():
            W[o, i, 0, 0] = v
    return W


def region(y0, y1, x0, x1):
    """Indicator of rows y0..y1, cols x0..x1 (inclusive) on the 28x28 grid."""
    r = np.zeros((N, N))
    r[y0:y1 + 1, x0:x1 + 1] = 1
    return r


# =====================================================================================================
# 3. The network
# =====================================================================================================

def build():
    net = Net()

    # ---- stage A: pixel filters on the image (5x5 kernels, padding 2) ----
    # channel: (kind, list of (dy, dx, coef), bias)
    def tap(*pairs):
        return list(pairs)
    on_lo, on_hi = 0.02, 0.06                     # grey level where an item pixel starts / is surely on
    g = 1.0 / (on_hi - on_lo)
    on2_lo, on2_hi = 0.08, 0.15                   # stricter: ignores the faint halo that sharpening leaves
    g2 = 1.0 / (on2_hi - on2_lo)
    A = [
        (tap((0, 0, g)), -on_lo * g),             # 0  ramp start  -> mask = relu(0) - relu(1)
        (tap((0, 0, g)), -on_lo * g - 1.0),       # 1  ramp end
        (tap((0, 0, 1.0)), 0.0),                  # 2  grey level
        (tap((0, -1, 1.0), (0, 0, -1.0)), 0.0),   # 3  left neighbour brighter than centre
        (tap((0, 1, 1.0), (0, 0, -1.0)), 0.0),    # 4  right neighbour brighter
        (tap((0, -2, 1.0), (0, 0, -1.0)), 0.0),   # 5  2nd left brighter
        (tap((0, 2, 1.0), (0, 0, -1.0)), 0.0),    # 6  2nd right brighter
        (tap((0, 0, 1.0), (0, -1, -1.0)), 0.0),   # 7  centre brighter than left
        (tap((0, 0, 1.0), (0, 1, -1.0)), 0.0),    # 8  centre brighter than right
        (tap((0, 1, 1.0), (0, -1, -1.0)), 0.0),   # 9  horizontal gradient +
        (tap((0, 1, -1.0), (0, -1, 1.0)), 0.0),   # 10 horizontal gradient -
        (tap((1, 0, 1.0), (-1, 0, -1.0)), 0.0),   # 11 vertical gradient +
        (tap((1, 0, -1.0), (-1, 0, 1.0)), 0.0),   # 12 vertical gradient -
        (tap((0, 0, 1.0), (0, -2, -1.0)), 0.0),   # 13 centre brighter than 2nd left
        (tap((0, 0, 1.0), (0, 2, -1.0)), 0.0),    # 14 centre brighter than 2nd right
        (tap((0, 0, g2)), -on2_lo * g2),          # 15 strict mask ramp start
        (tap((0, 0, g2)), -on2_lo * g2 - 1.0),    # 16 strict mask ramp end
    ]
    W = np.zeros((len(A), 1, 5, 5)); b = np.zeros(len(A))
    for o, (taps, bias) in enumerate(A):
        for dy, dx, v in taps:
            W[o, 0, dy + 2, dx + 2] += v
        b[o] = bias
    net.conv("px1", "image", W, b, pad=2)
    net.relu("px1r", "px1")
    rows = [
        {0: 1, 1: -1},          # 0 mask m in [0,1]
        {2: 1},                 # 1 grey level
        {3: 1},                 # 2 p1
        {3: 1, 4: -1},          # 3 p1 - q1      (min(p1,q1) = p1 - relu(p1-q1))
        {5: 1},                 # 4 p2
        {5: 1, 6: -1},          # 5 p2 - q2
        {7: 1},                 # 6 s1
        {7: 1, 8: -1},          # 7 s1 - t1
        {9: 1, 10: 1},          # 8 |gx|
        {11: 1, 12: 1},         # 9 |gy|
        {13: 1},                # 10 s2
        {13: 1, 14: -1},        # 11 s2 - t2
        {15: 1, 16: -1},        # 12 strict mask
    ]
    net.conv("px2", "px1r", onehot_1x1(rows, len(A)), np.zeros(len(rows)), pad=0)
    net.relu("px2r", "px2")
    dv = 0.06                                     # minimum contrast of a thin dark / bright line
    rows = [
        ({0: 1}, 0.0),                  # 0 m
        ({1: 1}, 0.0),                  # 1 grey
        ({2: 1, 3: -1}, -dv),           # 2 dark vertical line, width 1
        ({4: 1, 5: -1}, -dv),           # 3 dark vertical line, width up to 3
        ({6: 1, 7: -1}, -dv),           # 4 bright vertical line, width 1
        ({8: 1}, 0.0),                  # 5 |gx|
        ({9: 1}, 0.0),                  # 6 |gy|
        ({10: 1, 11: -1}, -dv),         # 7 bright vertical line, width up to 3
        ({12: 1}, 0.0),                 # 8 strict mask
    ]
    net.conv("px3", "px2r", onehot_1x1([r for r, _ in rows], 13), [c for _, c in rows], pad=0)
    P = net.relu("pix", "px3")                    # (9, 28, 28)
    NP = len(rows)

    # ---- stage B: where is the item to the left / right / above / below each pixel ----
    net.conv("msel", P, onehot_1x1([{0: 1}], NP), [0.0], pad=0)
    idx = np.arange(N * N).reshape(N, N)
    Wc = np.zeros((9, N, N, N * N)); bc = np.zeros((9, N, N))
    for y in range(N):
        for x in range(N):
            Wc[0, y, x, idx[y, :x + 1]] = 1      # SL: on-pixels at or left of x
            Wc[1, y, x, idx[y, x:]] = 1          # SR
            Wc[2, y, x, idx[:y + 1, x]] = 1      # SU
            Wc[3, y, x, idx[y:, x]] = 1          # SD
            Wc[8, y, x, idx[y, x]] = 1           # copy of the mask
    Wc[4:8] = Wc[0:4]; bc[4:8] = -1.0            # for clamping: min(S,1) = relu(S) - relu(S-1)
    net.dense("cum", "msel", Wc.reshape(9 * N * N, N * N), bc.reshape(-1))
    net.reshape("cumr", "cum", (9, N, N))
    net.relu("cumrr", "cumr")
    # L,R,U,D,m as combinations of the 9 relu channels
    Lc, Rc, Uc, Dc, Mc = {0: 1, 4: -1}, {1: 1, 5: -1}, {2: 1, 6: -1}, {3: 1, 7: -1}, {8: 1}

    def comb(*terms):
        out = {}
        for coef, d in terms:
            for k, v in d.items():
                out[k] = out.get(k, 0) + coef * v
        return out
    Q_rows = [
        (comb((1, Lc)), 0.0),                                               # 0 L
        (comb((1, Rc)), 0.0),                                               # 1 R
        (comb((1, Uc)), 0.0),                                               # 2 U
        (comb((1, Dc)), 0.0),                                               # 3 D
        (comb((1, Mc)), 0.0),                                               # 4 m
        (comb((1, Lc), (1, Rc), (1, Uc), (1, Dc)), -3.0),                   # 5 hull (inside all four spans)
        (comb((1, Rc), (1, Dc), (-1, Lc), (-1, Uc)), -1.0),                 # 6 outside top-left
        (comb((1, Lc), (1, Dc), (-1, Rc), (-1, Uc)), -1.0),                 # 7 outside top-right
        (comb((1, Rc), (1, Uc), (-1, Lc), (-1, Dc)), -1.0),                 # 8 outside bottom-left
        (comb((1, Lc), (1, Uc), (-1, Rc), (-1, Dc)), -1.0),                 # 9 outside bottom-right
        (comb((1, Lc), (1, Rc), (1, Uc), (-1, Dc), (-1, Mc)), -2.0),        # 10 gap open downward
        (comb((1, Lc), (1, Rc), (1, Dc), (-1, Uc), (-1, Mc)), -2.0),        # 11 notch open upward
        (comb((1, Lc), (1, Rc), (1, Uc), (1, Dc), (-1, Mc)), -3.0),         # 12 enclosed hole
        (comb((1, Lc), (1, Rc)), -1.0),                                     # 13 inside row span
        (comb((1, Rc), (1, Uc), (1, Dc), (-1, Lc), (-1, Mc)), -2.0),        # 14 bay open to the left
        (comb((1, Lc), (1, Uc), (1, Dc), (-1, Rc), (-1, Mc)), -2.0),        # 15 bay open to the right
    ]
    net.conv("qc", "cumrr", onehot_1x1([r for r, _ in Q_rows], 9), [c for _, c in Q_rows], pad=0)
    Q = net.relu("q", "qc")
    NQ = len(Q_rows)

    # ---- stage C: silhouette template matching ----
    T = class_templates()
    V = max(len(v) for v in T.values())
    Wt = np.zeros((10, V, N * N)); bt = np.zeros((10, V))
    for k in range(10):
        ts = T[k]
        for v in range(V):
            t, o = ts[v % len(ts)]
            t = t.reshape(-1)
            Wt[k, v] = 2 * t - 1                 # agreement = sum hull*(2t-1) + sum(1-t)
            bt[k, v] = np.sum(1 - t) + o
    net.conv("hsel", Q, onehot_1x1([{5: 1}], NQ), [0.0], pad=0)
    net.dense("tmatch", "hsel", Wt.reshape(10 * V, N * N), bt.reshape(-1))
    net.reshape("tm3", "tmatch", (10, V, 1))
    net.maxpool("tmax", "tm3", (V, 1), (V, 1))
    tscore = net.reshape("tscore", "tmax", (10,))

    # ---- stage C2: long straight vertical lines (zip, front edge, placket) ----
    # averaging 8 rows first keeps a faint but straight line while noise and small prints average out
    net.avgpool("vavg", "image", (8, 1), (1, 1))                              # (1, 21, 28)
    Z = []
    for s in (1, 2):
        Z += [[(0, 1.0), (-s, -1.0)], [(0, 1.0), (s, -1.0)], [(0, -1.0), (-s, 1.0)], [(0, -1.0), (s, 1.0)]]
    # order: bright vs left, bright vs right, dark vs left, dark vs right (s=1), same for s=2
    Wz = np.zeros((8, 1, 5, 5))
    for o, taps in enumerate(Z):
        for dx, v in taps:
            Wz[o, 0, 2, dx + 2] += v
    net.conv("z1", "vavg", Wz, np.zeros(8), pad=2)
    net.relu("z1r", "z1")
    rowsz = []
    for p in range(4):
        rowsz += [{2 * p: 1}, {2 * p: 1, 2 * p + 1: -1}]
    net.conv("z2", "z1r", onehot_1x1(rowsz, 8), np.zeros(8), pad=0)
    net.relu("z2r", "z2")
    # sum of min(a, b) = a - relu(a - b) over the four line types
    net.conv("z3", "z2r", onehot_1x1([{k: (1 if k % 2 == 0 else -1) for k in range(8)}], 8), [0.0], pad=0)
    ZL = net.relu("zl", "z3")                                                  # (1, 21, 28)

    # ---- stage D: region features (sums of the maps over hand-chosen regions) ----
    full = np.ones((N, N))
    rowsel = np.zeros((N, N)); rowsel[:, 0] = 1          # R at x=0: the row holds part of the item
    colsel = np.zeros((N, N)); colsel[0, :] = 1          # D at y=0: the column holds part of the item
    torso = region(8, 24, 9, 18)
    feats = []                                            # (name, branch, {channel: 28x28 weights})
    feats += [
        ("H", "Q", {1: rowsel}),                          # item height in rows
        ("W", "Q", {3: colsel}),                          # item width in columns
        ("hull", "Q", {5: full}),                         # filled outline area
        ("hole", "Q", {12: full}),                        # enclosed background inside the outline
        ("holeTop", "Q", {12: region(0, 11, 0, 27)}),     # ... in the top part (under a handle)
        ("notchAll", "Q", {11: full}),                    # background open only upward
        ("gapAll", "Q", {10: full}),                      # background open only downward
        ("gapC", "Q", {10: region(18, 27, 10, 17)}),      # opening between legs
        ("gapS", "Q", {10: region(12, 27, 2, 9) + region(12, 27, 18, 25)}),   # sleeve / body gaps
        ("notch", "Q", {11: region(0, 9, 7, 20)}),        # neckline dip
        ("texx", "P", {5: torso}), ("texy", "P", {6: torso}), ("bright", "P", {1: torso}),
        ("sideAll", "Q", {14: full, 15: full}),           # background open only sideways
        ("on", "Q", {4: full}),                           # lit area
        ("wTop", "Q", {13: region(2, 6, 0, 27)}),         # outline width near the top (5 rows)
        ("wMid", "Q", {13: region(12, 16, 0, 27)}),       # ... at mid height
        ("wBot", "Q", {13: region(22, 26, 0, 27)}),       # ... near the hem
        ("on2", "P", {8: full}),                          # lit area with the strict mask
        ("topC", "Q", {4: region(0, 2, 11, 16)}),         # collar / hood standing up in the middle
        ("topS", "Q", {4: region(0, 2, 3, 8) + region(0, 2, 19, 24)}),   # shoulders reaching the top
    ]
    for c in range(N):    # per-column sums of thin dark vertical lines (low part, and from mid height down)
        feats.append((f"vlow{c}", "P", {2: region(16, 27, c, c), 3: region(16, 27, c, c)}))
        feats.append((f"vside{c}", "P", {2: region(10, 27, c, c), 3: region(10, 27, c, c)}))
    for c in range(10, 18):   # thin dark or bright vertical line down the middle (zip, button placket)
        feats.append((f"zip{c}", "P", {k: region(12, 26, c, c) for k in (2, 3, 4, 7)}))
    # line strength per column in the upper torso (8-row windows starting at rows 9..12, i.e. rows 9-19) and in
    # the lower torso (windows starting at rows 16..19, i.e. rows 16-26)
    for c in range(8, 20):
        up_ = np.zeros((21, N)); up_[9:13, c] = 1
        lo_ = np.zeros((21, N)); lo_[16:20, c] = 1
        feats.append((f"zu{c}", "Z", {0: up_}))
        feats.append((f"zd{c}", "Z", {0: lo_}))
    names = [n for n, _, _ in feats]
    F = len(names)
    fi = {n: i for i, n in enumerate(names)}
    WQ = np.zeros((F, NQ, N, N)); WP = np.zeros((F, NP, N, N)); WZ = np.zeros((F, 1, 21, N))
    for i, (n, br, d) in enumerate(feats):
        for c, mp in d.items():
            {"Q": WQ, "P": WP, "Z": WZ}[br][i, c] = mp
    net.dense("fq", Q, WQ.reshape(F, -1), np.zeros(F))
    net.dense("fp", P, WP.reshape(F, -1), np.zeros(F))
    net.dense("fz", ZL, WZ.reshape(F, -1), np.zeros(F))
    f = net.add("feat", ["fq", "fp", "fz"])

    # ---- stage E: hand-set evidence rules ----
    # E1: relu units over the features: pass-throughs and "a line concentrated in one or two columns"
    gu = []                                               # (name, {feature: coef}, bias)
    for n in ("H", "W", "hull", "hole", "holeTop", "notchAll", "gapAll", "gapC", "gapS", "notch", "texx", "texy",
              "bright", "sideAll", "on", "wTop", "wMid", "wBot", "on2", "topC", "topS"):
        gu.append((n, {n: 1.0}, 0.0))
    for c in range(9, 18):
        gu.append((f"tcol{c}", {f"vlow{c}": 1.0, f"vlow{c + 1}": 1.0}, -2.0))
    for c in list(range(2, 9)) + list(range(18, 25)):
        gu.append((f"scol{c}", {f"vside{c}": 1.0, f"vside{c + 1}": 1.0}, -2.5))
    for c in range(11, 17):
        gu.append((f"zcol{c}", {f"zip{c}": 1.0}, -1.5))
    for c in range(8, 20):           # min(upper, lower) = upper - relu(upper - lower): the line must run through both
        gu.append((f"zu{c}", {f"zu{c}": 1.0}, 0.0))
        gu.append((f"zdiff{c}", {f"zu{c}": 1.0, f"zd{c}": -1.0}, 0.0))
    G = len(gu); gi = {n: i for i, (n, _, _) in enumerate(gu)}
    Wg = np.zeros((G, F)); bg = np.zeros(G)
    for i, (n, d, c) in enumerate(gu):
        for k, v in d.items():
            Wg[i, fi[k]] = v
        bg[i] = c
    net.dense("g1", f, Wg, bg)
    net.relu("g1r", "g1")

    # E2: soft indicators phi = clamp((sum a_j g_j + c) / s) = relu(z) - relu(z - 1)
    tcols = {f"tcol{c}": 1.0 for c in range(9, 18)}
    scols = {f"scol{c}": 1.0 for c in list(range(2, 9)) + list(range(18, 25))}
    # continuous line in the middle columns, compared with the same measure in the columns beside them (knit ribs,
    # folds and shading show up in both and cancel)
    zcols = {}
    for c in range(8, 20):
        sgn = 1.0 if 11 <= c <= 16 else -1.0
        zcols[f"zu{c}"] = sgn
        zcols[f"zdiff{c}"] = -sgn
    phis = [
        ("gap",    dict(gapC=1.0, **tcols), -3.0, 8.0),                     # opening between two legs
        ("narrow", dict(W=-1.0), 21.0, 4.0),                                # item much narrower than tall
        ("wide",   dict(W=1.0), -24.0, 3.0),                                # item spans (nearly) the full width
        ("open",   dict(hole=1.0, notchAll=1.0, gapAll=1.0, sideAll=1.0, hull=-0.15), -5.0, 25.0),  # gaps between straps
        ("sparse", dict(H=14.0, on2=-1.0), -10.0, 40.0),                    # footwear (full width) mostly empty
        ("collar", dict(topC=1.0, topS=-0.5), -6.0, 8.0),                   # collar or hood above the shoulders
        ("flare",  dict(wBot=1.0, wMid=-1.0), -5.0, 15.0),                  # skirt widening to the hem
        ("topwide", dict(wTop=1.0, wBot=-1.0), -10.0, 20.0),                # sleeves wider than the hem
        ("tall",   dict(H=1.0), -16.0, 5.0),                                # tall footwear (boot shaft)
        ("zip",    zcols, -0.3, 0.6),                                       # line down the middle
        ("neck",   dict(notch=1.0), -4.0, 8.0),                             # round neckline dip
        ("texx",   dict(texx=1.0, bright=-0.2), -3.0, 10.0),                # many vertical edges on the torso
        ("texy",   dict(texy=1.0, bright=-0.2), -3.0, 10.0),                # many horizontal edges on the torso
        ("handle", dict(holeTop=1.0), -6.0, 12.0),                          # hole under a handle
        ("sleeve", dict(gapS=1.0, **scols), -4.0, 10.0),                    # long sleeve beside the body
    ]
    P_n = len(phis); pi_ = {n: i for i, (n, _, _, _) in enumerate(phis)}
    W1 = np.zeros((2 * P_n, G)); b1 = np.zeros(2 * P_n)
    for i, (n, coefs, c, s) in enumerate(phis):
        for k, v in coefs.items():
            W1[2 * i, gi[k]] = v / s
            W1[2 * i + 1, gi[k]] = v / s
        b1[2 * i] = c / s
        b1[2 * i + 1] = c / s - 1.0
    net.dense("ev1", "g1r", W1, b1)
    net.relu("ev1r", "ev1")

    # E3: indicators and their conjunctions (AND of [0,1] values a, b = relu(a + b - 1))
    units = [(n, [n]) for n, _, _, _ in phis] + [("trouser", ["gap", "narrow"]), ("plaid", ["texx", "texy"]),
                                                 ("sparseshoe", ["sparse", "wide"]), ("teezip", ["topwide", "zip"])]
    U = len(units); ui = {n: i for i, (n, _) in enumerate(units)}
    W3 = np.zeros((U, 2 * P_n)); b3 = np.zeros(U)
    for i, (n, parts) in enumerate(units):
        for p in parts:
            W3[i, 2 * pi_[p]] += 1.0
            W3[i, 2 * pi_[p] + 1] -= 1.0
        b3[i] = -(len(parts) - 1.0)
    net.dense("ev3", "ev1r", W3, b3)
    net.relu("ev3r", "ev3")

    # E4: evidence table, in pixels of silhouette agreement
    table = {
        "trouser": {1: 45},
        "gap":     {3: -15},
        "open":    {5: 35, 7: -15, 9: -15},
        "sparseshoe": {5: 35, 7: -15, 9: -15},
        "teezip":  {6: 20, 0: -10},
        "flare":   {3: 15, 0: -10},
        "topwide": {0: 10, 3: -15},
        "tall":    {9: 12, 7: -20},
        "zip":     {4: 25, 6: 10, 2: -10},
        "collar":  {4: 8, 2: -4},
        "neck":    {2: 12, 0: 8, 4: -12, 6: -8, 8: -10},
        "plaid":   {6: 35, 2: -10, 4: -10, 0: -5},
        "handle":  {8: 25},
        "sleeve":  {2: 15, 4: 15, 6: 12, 8: -20, 0: -5},
    }
    prior = np.zeros(10)
    prior[1] = -20.0     # trousers must show the opening between the legs
    prior[4] = -8.0      # a coat should show its front line
    prior[6] = -10.0     # a shirt should show its pattern or placket
    W4 = np.zeros((10, U))
    for n, d in table.items():
        for k, v in d.items():
            W4[k, ui[n]] += v
    net.dense("ev4", "ev3r", W4, prior)
    net.add("scores", [tscore, "ev4"])
    return net


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = build()
    (out / "network.json").write_text(json.dumps({"layers": net.layers}, indent=1))
    np.savez(out / "weights.npz", **{k: v.astype(np.float32) for k, v in net.w.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
