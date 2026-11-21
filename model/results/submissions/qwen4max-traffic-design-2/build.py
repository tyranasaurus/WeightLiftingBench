"""Hand-designed (no training, no data) traffic-sign classifier.

Everything here comes from general knowledge of German road signs (shape, colour,
pictogram geometry) and from formulas (Gaussians, capsules, polygons, arcs).
No training image, label statistic or fitted number is used anywhere.

Architecture
  1x1 convs -> signed, illumination-covariant colour / contrast channels
  per-channel `norm` (removes gain and ambient offset of the lighting)
  one big conv whose kernels are analytic full-sign templates (43 classes x 9
  scale/aspect variants).  Each template channel is  +1/area(pos) on the region
  where that feature is expected and -1/area(neg) where it is not, so a kernel
  response is a difference of regional means: a contrast score, independent of
  region size and of image brightness.
  max over variants and over a 5x5 grid of positions -> 43 class scores.
"""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
K = 28                      # template kernel size (5x5 position grid on 32x32)
CEN = K / 2.0
NCH = 6                     # red, blue, yellow, green, achromatic, dark
VARIANTS = [(21, 21), (25, 25), (29, 29),
            (25, 21), (21, 25), (29, 25), (25, 29), (29, 21), (21, 29),
]
NV = len(VARIANTS)
NCLASS = 43

layers = []
weights = {}


def conv(name, inp, out, k, stride=1, pad=0, W=None, b=None):
    layers.append({"name": name, "op": "conv", "inputs": [inp], "out": out,
                   "kernel": [k, k], "stride": stride, "padding": pad})
    weights[name + ".W"] = np.asarray(W, np.float32)
    weights[name + ".b"] = np.asarray(b if b is not None else np.zeros(out), np.float32)


def simple(name, op, inp, **kw):
    d = {"name": name, "op": op, "inputs": [inp]}
    d.update(kw)
    layers.append(d)


def add(name, ins):
    layers.append({"name": name, "op": "add", "inputs": list(ins)})


# ---------------------------------------------------------------- feature stack
# signed linear colour forms: homogeneous degree 1 in the pixel values, so a
# multiplicative lighting change only scales them and an additive ambient light
# only shifts them; the per-channel norm removes both.
LIN = [
    (1, 1, 1),          # 0 S
    (1, -1, 0),         # 1 r-g
    (-1, 1, 0),         # 2 g-r
    (0, 1, -1),         # 3 g-b
    (0, -1, 1),         # 4 b-g
    (-1, 0, 1),         # 5 b-r
    (1, 0, -1),         # 6 r-b
    (1 / 3, 1 / 3, 1 / 3),   # 7 I
]
S = (1, 1, 1)


def lin(form, extra=0.0):
    return tuple(form[i] + extra * S[i] for i in range(3))


rows = list(LIN)
rows.append(lin((1, -1, 0), -0.12))   # 8 red
rows.append(lin((-1, 0, 1), -0.18))   # 9 blue
rows.append(lin((0, 1, -1), -0.18))   # 10 yellow base
rows.append(lin((0, 1, -1), -0.06))   # 11 green base  (g-r)-.06S  -> fixed below
rows[11] = lin((-1, 1, 0), -0.06)
rows.append(lin((0, 0, 0), 0.14))     # 12 ach base
rows.append(lin((-1, 1, 0), -0.02))   # 13 vegetation (g>r)
W = np.zeros((len(rows), 3, 1, 1), np.float32)
for i, r in enumerate(rows):
    W[i, :, 0, 0] = r
conv("lin", "image", len(rows), 1, W=W)
simple("linr", "relu", "lin")

M = np.zeros((NCH, len(rows), 1, 1), np.float32)
M[0, 8] = 1                                   # red
M[1, 9] = 1                                   # blue
M[2, 10] = 1; M[2, 13] = -0.6                 # yellow, not vegetation
M[3, 11] = 1; M[3, 4] = -0.6                  # green, not blue
M[4, 12] = 1                                  # achromatic
for j in (1, 2, 3, 4, 5, 6):
    M[4, j] = -1
M[5, 7] = 1                                   # intensity
conv("mix", "linr", NCH, 1, W=M)
Wi = np.zeros((1, NCH, 1, 1), np.float32); Wi[0, 5] = 1
conv("isel", "mix", 1, 1, W=Wi)


def gauss(k, sig, sign=1.0):
    ax = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-0.5 * (ax / sig) ** 2)
    g = np.outer(g, g)
    g /= g.sum()
    return (sign * g)[None, None, :, :].astype(np.float32)


conv("g1", "isel", 1, 5, pad=2, W=gauss(5, 0.9))
conv("g2", "isel", 1, 9, pad=4, W=gauss(9, 2.2, -1.0))
add("dog", ["g1", "g2"])                      # centre-surround contrast
Wd = np.zeros((NCH, 1, 1, 1), np.float32); Wd[5, 0] = -1
conv("darkc", "dog", NCH, 1, W=Wd)
Ws = np.zeros((NCH, NCH, 1, 1), np.float32)
for j in range(5):
    Ws[j, j] = 1
conv("stackc", "mix", NCH, 1, W=Ws)
add("feat", ["stackc", "darkc"])
simple("fn", "norm", "feat")

# ---------------------------------------------------------------- draw helpers
ax = np.arange(K) + 0.5 - CEN
XX, YY = np.meshgrid(ax, ax)


def gblur(a, sig=0.6):
    r = int(np.ceil(3 * sig))
    t = np.arange(-r, r + 1)
    g = np.exp(-0.5 * (t / sig) ** 2); g /= g.sum()
    p = np.pad(a, r, mode="edge")
    out = np.zeros_like(a)
    for i, w in enumerate(g):
        out += w * p[r:r + K, i:i + K]
    p = np.pad(out, r, mode="edge")
    out2 = np.zeros_like(a)
    for i, w in enumerate(g):
        out2 += w * p[i:i + K, r:r + K]
    return out2


class Sign:
    """unit sign coordinates: u right, v down, sign spans [-1,1] in both."""

    def __init__(self, dx, dy):
        self.u = XX / (dx / 2.0)
        self.v = YY / (dy / 2.0)
        self.T = np.zeros((NCH, K, K), np.float32)

    def reg(self, ch, pos, neg, w):
        if pos is not None:
            P = gblur(np.asarray(pos, np.float64))
            sp = P.sum()
            if sp > 1e-6:
                self.T[ch] += w * P / sp
        N = gblur(np.asarray(neg, np.float64)) if neg is not None else None
        if N is not None:
            sn = N.sum()
            if sn > 1e-6:
                self.T[ch] -= w * N / sn

    def coarse(self, m, ur=0.85, vr=0.65, nc=8, nr=6, hi=0.22, lo=0.06):
        """coarse full-pattern bitmap: cells that are mostly stroke become
        positive, cells that are mostly background become negative.  Coarse
        cells survive blur, blockiness and small misalignment, and specifying
        the white cells too penalises partial / wrong-symbol matches."""
        m = np.asarray(m, float)
        ue = np.linspace(-ur, ur, nc + 1)
        ve = np.linspace(-vr, vr, nr + 1)
        pos = np.zeros(m.shape, bool)
        neg = np.zeros(m.shape, bool)
        for i in range(nr):
            for j in range(nc):
                cell = (self.u >= ue[j]) & (self.u < ue[j + 1]) & (self.v >= ve[i]) & (self.v < ve[i + 1])
                if cell.sum() == 0:
                    continue
                f = m[cell].mean()
                if f > hi:
                    pos |= cell
                elif f < lo:
                    neg |= cell
        return pos, neg

    # --- primitives (unit coords) ---
    def disc(self, r=1.0, cu=0.0, cv=0.0):
        return ((self.u - cu) ** 2 + (self.v - cv) ** 2) <= r * r

    def ann(self, r0, r1):
        d = np.sqrt(self.u ** 2 + self.v ** 2)
        return (d >= r0) & (d <= r1)

    def bbox(self):
        return (np.abs(self.u) <= 1.25) & (np.abs(self.v) <= 1.25)

    def poly(self, pts):
        pts = np.asarray(pts, float)
        c = pts.mean(0)
        d = np.full(self.u.shape, 1e9)
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            n = np.array([b[1] - a[1], -(b[0] - a[0])])
            n /= np.linalg.norm(n)
            if n @ (c - a) < 0:
                n = -n
            d = np.minimum(d, n[0] * (self.u - a[0]) + n[1] * (self.v - a[1]))
        return d

    def tri(self, inv=False, t=None):
        s = -1.0 if inv else 1.0
        pts = [(0, -0.87 * s), (1, 0.87 * s), (-1, 0.87 * s)]
        d = self.poly(pts)
        if t is None:
            return d > 0
        return (d > 0) & (d < t)

    def octa(self, t=None):
        a = np.cos(np.pi / 8) * 1.08
        m = np.full(self.u.shape, -1e9)
        for k in range(8):
            th = np.pi / 8 + k * np.pi / 4
            m = np.maximum(m, np.cos(th) * self.u + np.sin(th) * self.v)
        if t is None:
            return m < a
        return (m < a) & (m > a - t)

    def diam(self, t=None):
        d = np.abs(self.u) + np.abs(self.v)
        if t is None:
            return d < 1.0
        return (d < 1.0) & (d > 1.0 - t)

    def seg(self, p, q, w):
        p = np.asarray(p, float); q = np.asarray(q, float)
        d = q - p
        L2 = d @ d
        t = np.clip(((self.u - p[0]) * d[0] + (self.v - p[1]) * d[1]) / max(L2, 1e-9), 0, 1)
        dist = np.sqrt((self.u - p[0] - t * d[0]) ** 2 + (self.v - p[1] - t * d[1]) ** 2)
        return dist <= w / 2

    def stroke(self, pts, w):
        m = np.zeros(self.u.shape, bool)
        for i in range(len(pts) - 1):
            m |= self.seg(pts[i], pts[i + 1], w)
        return m

    def dot(self, cu, cv, r):
        return self.disc(r, cu, cv)

    def ring(self, cu, cv, r, w):
        d = np.sqrt((self.u - cu) ** 2 + (self.v - cv) ** 2)
        return np.abs(d - r) <= w / 2

    def arc(self, cu, cv, r, a0, a1, n=14):
        th = np.linspace(a0, a1, n) * np.pi / 180
        return [(cu + r * np.cos(t), cv + r * np.sin(t)) for t in th]

    def head(self, cu, cv, ang, size):
        th = ang * np.pi / 180
        d = np.array([np.cos(th), np.sin(th)])
        p = np.array([-d[1], d[0]])
        tip = np.array([cu, cv])
        return self.poly([tip, tip - size * d + 0.55 * size * p, tip - size * d - 0.55 * size * p]) > 0

    def box(self, cu, cv, w, h, rot=0.0):
        th = rot * np.pi / 180
        u = np.cos(th) * (self.u - cu) + np.sin(th) * (self.v - cv)
        v = -np.sin(th) * (self.u - cu) + np.cos(th) * (self.v - cv)
        return (np.abs(u) <= w / 2) & (np.abs(v) <= h / 2)


# ------------------------------------------------------------------ digit font
def digit(sg, ch, u0, v0, h, w, sw):
    """DIN-1451-like stroke digit in a local [0,1]x[0,1] box."""
    W = sw * h

    def P(uu, vv):
        return (u0 + uu * w, v0 + vv * h)

    def seg(a, b, ww=None):
        return sg.seg(P(*a), P(*b), ww if ww else W)

    def arc(cu, cv, r, a0, a1, ww=None):
        pts = [P(*q) for q in sg.arc(cu, cv, r, a0, a1)]
        return sg.stroke(pts, ww if ww else W)

    m = np.zeros(sg.u.shape, bool)
    if ch == "0":
        # ellipse ring
        uu = (sg.u - u0) / w; vv = (sg.v - v0) / h
        e = ((uu - 0.5) / 0.5) ** 2 + ((vv - 0.5) / 0.5) ** 2
        m |= (e <= 1.0) & (e >= (1 - sw / 0.5) ** 2)
    elif ch == "1":
        m |= seg((0.62, 0.04), (0.62, 1.0))
        m |= seg((0.26, 0.30), (0.62, 0.04))
    elif ch == "2":
        m |= arc(0.5, 0.42, 0.40, 180, 350)
        m |= seg((0.88, 0.50), (0.10, 0.93))
        m |= seg((0.10, 0.95), (0.90, 0.95))
    elif ch == "3":
        m |= arc(0.46, 0.28, 0.30, 200, 450)
        m |= arc(0.46, 0.72, 0.30, -90, 200)
    elif ch == "5":
        m |= seg((0.12, 0.05), (0.88, 0.05))
        m |= seg((0.14, 0.05), (0.14, 0.46))
        m |= seg((0.14, 0.46), (0.60, 0.46))
        m |= arc(0.48, 0.68, 0.32, -80, 200)
    elif ch == "6":
        m |= arc(0.52, 0.40, 0.38, -60, -240)
        m |= sg.ring(*P(0.50, 0.72), 0.30 * h - W / 2, W)
    elif ch == "7":
        m |= seg((0.10, 0.05), (0.90, 0.05))
        m |= seg((0.88, 0.05), (0.35, 0.98))
    elif ch == "8":
        m |= sg.ring(*P(0.50, 0.26), 0.25 * h - W / 2, W)
        m |= sg.ring(*P(0.50, 0.72), 0.30 * h - W / 2, W)
    return m


def numstr(sg, text, h, w, gap, sw):
    n = len(text)
    total = n * w + (n - 1) * gap
    u0 = -total / 2
    m = np.zeros(sg.u.shape, bool)
    for i, ch in enumerate(text):
        m |= digit(sg, ch, u0 + i * (w + gap), -h / 2, h, w, sw)
    return m


# ------------------------------------------------------------------ pictograms
def person(sg, cu, cv, s, run=0.0):
    m = sg.dot(cu + 0.05 * s, cv - 0.70 * s, 0.17 * s)
    m |= sg.seg((cu + 0.05 * s, cv - 0.50 * s), (cu, cv + 0.05 * s), 0.26 * s)
    m |= sg.seg((cu, cv + 0.05 * s), (cu - (0.35 + run) * s, cv + 0.75 * s), 0.22 * s)
    m |= sg.seg((cu, cv + 0.05 * s), (cu + (0.30 + run) * s, cv + 0.70 * s), 0.22 * s)
    m |= sg.seg((cu, cv - 0.35 * s), (cu - 0.35 * s, cv - 0.05 * s), 0.18 * s)
    m |= sg.seg((cu, cv - 0.35 * s), (cu + 0.35 * s, cv - 0.15 * s), 0.18 * s)
    return m


def car(sg, cu, cv, s):
    m = sg.box(cu, cv + 0.05 * s, 0.55 * s, 0.18 * s)
    m |= sg.box(cu - 0.02 * s, cv - 0.10 * s, 0.30 * s, 0.16 * s)
    m |= sg.dot(cu - 0.18 * s, cv + 0.14 * s, 0.06 * s)
    m |= sg.dot(cu + 0.18 * s, cv + 0.14 * s, 0.06 * s)
    return m


def truck(sg, cu, cv, s):
    m = sg.box(cu - 0.08 * s, cv - 0.06 * s, 0.34 * s, 0.28 * s)
    m |= sg.box(cu + 0.20 * s, cv + 0.03 * s, 0.16 * s, 0.18 * s)
    m |= sg.dot(cu - 0.18 * s, cv + 0.14 * s, 0.06 * s)
    m |= sg.dot(cu + 0.16 * s, cv + 0.14 * s, 0.06 * s)
    return m


def picto(sg, kind):
    """black pictogram in local [-1,1] coords, mapped into the triangle."""
    s = 0.55
    cu, cv = 0.0, 0.18

    def P(u, v):
        return (cu + s * u, cv + s * v)

    def seg(a, b, w):
        return sg.seg(P(*a), P(*b), w * s)

    def stroke(pts, w):
        return sg.stroke([P(*q) for q in pts], w * s)

    def dot(u, v, r):
        return sg.dot(*P(u, v), r * s)

    def head(u, v, ang, size):
        return sg.head(*P(u, v), ang, size * s)

    m = np.zeros(sg.u.shape, bool)
    extra = {}
    if kind == "excl":
        m |= seg((0, -0.8), (0, 0.25), 0.30)
        m |= dot(0, 0.72, 0.20)
    elif kind == "cross":
        m |= seg((0, -0.8), (0, 0.8), 0.34)
        m |= seg((-0.6, -0.1), (0.6, -0.1), 0.26)
    elif kind == "bendL":
        m |= stroke([(0.25, 0.8), (0.25, 0.0), (-0.1, -0.35), (-0.45, -0.45)], 0.26)
        m |= head(-0.5, -0.45, 180, 0.5)
    elif kind == "bendR":
        m |= stroke([(-0.25, 0.8), (-0.25, 0.0), (0.1, -0.35), (0.45, -0.45)], 0.26)
        m |= head(0.5, -0.45, 0, 0.5)
    elif kind == "double":
        m |= stroke([(0.3, 0.8), (0.3, 0.3), (-0.25, -0.1), (-0.25, -0.5), (0.15, -0.75)], 0.26)
        m |= head(0.2, -0.8, -60, 0.45)
    elif kind == "bumps":
        m |= (dot(-0.4, 0.75, 0.45) | dot(0.4, 0.75, 0.45)) & (sg.v < cv + 0.55 * s)
    elif kind == "slip":
        m |= sg.box(*P(0, -0.3), 0.9 * s, 0.45 * s, rot=-15)
        m |= stroke([(-0.5, 0.2), (-0.2, 0.4), (-0.5, 0.6), (-0.2, 0.85)], 0.18)
        m |= stroke([(0.1, 0.2), (0.4, 0.4), (0.1, 0.6), (0.4, 0.85)], 0.18)
    elif kind == "narrow":
        m |= seg((-0.45, -0.8), (-0.45, 0.8), 0.22)
        m |= stroke([(0.45, 0.8), (0.45, 0.1), (0.15, -0.4), (0.15, -0.8)], 0.22)
    elif kind == "works":
        m |= dot(-0.15, -0.65, 0.18)
        m |= seg((-0.15, -0.45), (0.1, 0.1), 0.26)
        m |= seg((0.1, 0.1), (-0.15, 0.7), 0.22)
        m |= seg((0.1, 0.1), (0.35, 0.7), 0.22)
        m |= seg((-0.1, -0.3), (0.45, -0.1), 0.20)
        m |= seg((0.45, -0.5), (0.6, 0.5), 0.14)
        m |= dot(0.6, 0.6, 0.20)
        m |= sg.poly([P(0.25, 0.85), P(0.6, 0.45), P(0.95, 0.85)]) > 0
    elif kind == "lights":
        m |= sg.box(*P(0, 0), 0.5 * s, 1.5 * s) & ~sg.box(*P(0, 0), 0.26 * s, 1.26 * s)
        extra["red"] = dot(0, -0.45, 0.16)
        extra["yellow"] = dot(0, 0.0, 0.16)
        extra["green"] = dot(0, 0.45, 0.16)
    elif kind == "ped":
        m |= person(sg, cu, cv, s)
    elif kind == "children":
        m |= person(sg, cu - 0.35 * s, cv, 0.8 * s, run=0.15)
        m |= person(sg, cu + 0.38 * s, cv + 0.1 * s, 0.7 * s, run=0.15)
    elif kind == "cycle":
        m |= sg.ring(*P(-0.45, 0.45), 0.32 * s, 0.14 * s)
        m |= sg.ring(*P(0.45, 0.45), 0.32 * s, 0.14 * s)
        m |= stroke([(-0.45, 0.45), (-0.1, -0.1), (0.3, -0.1), (0.45, 0.45), (-0.05, 0.45), (-0.1, -0.1)], 0.12)
        m |= seg((0.3, -0.1), (0.42, -0.35), 0.12)
        m |= seg((-0.05, -0.15), (0.1, -0.6), 0.20)
        m |= dot(0.12, -0.75, 0.15)
    elif kind == "snow":
        for k in range(6):
            a = k * 60 * np.pi / 180
            d = np.array([np.cos(a), np.sin(a)])
            m |= seg((0, 0), (0.8 * d[0], 0.8 * d[1]), 0.16)
            p = np.array([-d[1], d[0]])
            b = 0.5 * d
            m |= seg(tuple(b), tuple(b + 0.25 * (d + p) / np.linalg.norm(d + p)), 0.12)
            m |= seg(tuple(b), tuple(b + 0.25 * (d - p) / np.linalg.norm(d - p)), 0.12)
    elif kind == "deer":
        m |= seg((-0.3, 0.0), (0.35, 0.1), 0.35)
        m |= seg((-0.3, 0.0), (-0.6, -0.5), 0.22)
        m |= dot(-0.65, -0.6, 0.16)
        m |= seg((-0.65, -0.7), (-0.5, -0.95), 0.10)
        m |= seg((-0.65, -0.7), (-0.85, -0.9), 0.10)
        m |= seg((0.25, 0.1), (0.6, 0.6), 0.14)
        m |= seg((0.1, 0.1), (0.3, 0.7), 0.14)
        m |= seg((-0.2, 0.1), (-0.5, 0.6), 0.14)
        m |= seg((-0.35, 0.0), (-0.75, 0.4), 0.14)
    return m, extra


def stripes(sg):
    m = np.zeros(sg.u.shape, bool)
    for off in (-0.32, -0.16, 0.0, 0.16, 0.32):
        # lines v = -u + off  (bottom-left to top-right)
        m |= sg.seg((-1.2, 1.2 + off), (1.2, -1.2 + off), 0.05)
    return m & sg.disc(0.98)


# ------------------------------------------------------------------ per class
def build_template(c, dx, dy):
    sg = Sign(dx, dy)
    disc = sg.disc(1.0)
    inner = sg.disc(0.76)
    ring = sg.ann(0.76, 1.0)
    out = sg.bbox() & ~disc
    tri_b = sg.tri(t=0.18)
    tri_i = sg.tri() & ~sg.tri(t=0.30)
    itri_b = sg.tri(inv=True, t=0.18)
    itri_i = sg.tri(inv=True) & ~sg.tri(inv=True, t=0.30)

    def halo(m, s=1.0):
        """matched-filter negative: a ring hugging the positive strokes, so a
        blob without the right shape cannot score by covering the pos region."""
        return (gblur(np.asarray(m, float), s) > 0.25) & ~np.asarray(m, bool)

    def dil(m, s=1.2):
        return gblur(np.asarray(m, float), s) > 0.2

    def symreg(ch, m, area, w, coarse=True):
        if coarse:
            pos, neg = sg.coarse(m, 0.85, 0.65, 10, 6, 0.22, 0.06)
            sg.reg(ch, pos, neg, w)
        else:
            sg.reg(ch, m, halo(m, 1.2), w)

    def circle(symbol=None, symch=5, w=1.6, red_extra=None, coarse=True):
        sg.reg(0, ring, inner, 2.5)
        sg.reg(4, inner, ring, 1.75)
        if symbol is None:
            sg.reg(5, None, inner, w)
        else:
            symreg(symch, symbol, inner, w, coarse)
        if red_extra is not None:
            rp, rn = sg.coarse(red_extra, 0.85, 0.5, 10, 5, 0.22, 0.06)
            sg.reg(0, rp, rn, 0.9)
        sg.reg(1, None, disc, 1.25)
        sg.reg(2, None, disc, 1.25)
        sg.reg(3, None, disc, 0.75)

    def triangle(symbol=None, inv=False, w=1.6, extra=None):
        b, i = (itri_b, itri_i) if inv else (tri_b, tri_i)
        sg.reg(0, b, i, 2.5)
        sg.reg(4, i, b, 1.75)
        if symbol is None:
            sg.reg(5, None, i, w)
        else:
            symreg(5, symbol, i, w)
        sg.reg(1, None, i, 1.0)
        sg.reg(2, None, i, 1.0)
        if extra:
            for ch, m in extra.items():
                idx = {"red": 0, "yellow": 2, "green": 3}[ch]
                cp, cn = sg.coarse(m, 0.7, 0.6, 6, 6, 0.10, 0.02)
                sg.reg(idx, cp, cn, 0.8)
        else:
            sg.reg(3, None, i, 0.75)

    def bluedisc(arrow):
        sg.reg(1, disc & ~arrow, out, 3.0)
        ap, an = sg.coarse(arrow, 0.8, 0.8, 8, 8, 0.25, 0.06)
        sg.reg(4, ap, an, 1.2)
        sg.reg(5, None, disc, 1.25)
        sg.reg(0, None, disc, 1.25)
        sg.reg(2, None, disc, 1.0)

    def grey(ghost=None, gw=0.6, ghost_neg=False):
        st = stripes(sg)
        sg.reg(4, disc, out, 2.0)
        sg.reg(0, None, disc, 2.5)
        sg.reg(1, None, disc, 1.5)
        sg.reg(2, None, disc, 1.0)
        sg.reg(5, st, disc & ~dil(st, 1.0), 1.2)
        if ghost is not None:
            gp, gn = sg.coarse(ghost)
            sg.reg(5, gp, gn, gw)
        if ghost_neg:
            sg.reg(5, None, ghost, 0.5)

    if c in (0, 1, 2, 3, 4, 5, 7, 8):
        text = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}[c]
        if len(text) == 2:
            sym = numstr(sg, text, 1.0, 0.55, 0.12, 0.15)
        else:
            sym = numstr(sg, text, 0.88, 0.40, 0.07, 0.15)
        circle(sym, coarse=False)
    elif c == 6:
        grey(numstr(sg, "80", 0.95, 0.52, 0.12, 0.15), gw=0.7)
    elif c == 9:
        circle(car(sg, 0.42, 0.0, 1.0), red_extra=car(sg, -0.42, 0.0, 1.0))
    elif c == 10:
        circle(car(sg, 0.42, 0.0, 1.0), red_extra=truck(sg, -0.42, 0.0, 1.0))
    elif c == 11:
        triangle(picto(sg, "cross")[0])
    elif c == 12:
        sg.reg(2, sg.diam() & ~sg.diam(t=0.16), sg.bbox() & ~sg.diam(), 3.0)
        sg.reg(4, sg.diam(t=0.16), halo(sg.diam(t=0.16)), 0.6)
        sg.reg(0, None, sg.diam(), 1.25)
        sg.reg(1, None, sg.diam(), 1.25)
        sg.reg(5, None, sg.diam(), 1.5)
    elif c == 13:
        triangle(None, inv=True)
    elif c == 14:
        octa = sg.octa()
        txt = np.zeros(sg.u.shape, bool)
        # STOP: four block letters across the middle
        lw, lh, gap = 0.34, 0.42, 0.10
        total = 4 * lw + 3 * gap
        u0 = -total / 2
        for i, ch in enumerate("STOP"):
            uu = u0 + i * (lw + gap)
            if ch == "S":
                txt |= sg.stroke([(uu + lw, -0.18), (uu, -0.18), (uu, 0.0), (uu + lw, 0.0), (uu + lw, 0.18), (uu, 0.18)], 0.13)
            elif ch == "T":
                txt |= sg.seg((uu, -0.18), (uu + lw, -0.18), 0.13)
                txt |= sg.seg((uu + lw / 2, -0.18), (uu + lw / 2, 0.18), 0.13)
            elif ch == "O":
                txt |= sg.ring(uu + lw / 2, 0.0, 0.16, 0.13)
            elif ch == "P":
                txt |= sg.seg((uu, -0.18), (uu, 0.18), 0.13)
                txt |= sg.ring(uu + lw / 2 - 0.02, -0.09, 0.10, 0.13)
        sg.reg(0, octa & ~txt, sg.bbox() & ~octa, 3.0)
        sg.reg(4, txt, halo(txt, 1.2), 1.0)
        sg.reg(5, None, octa, 1.5)
        sg.reg(1, None, octa, 1.0)
        sg.reg(2, None, octa, 1.0)
    elif c == 15:
        circle(None)
    elif c == 16:
        circle(truck(sg, 0.0, 0.0, 1.3))
    elif c == 17:
        bar = sg.box(0, 0, 1.24, 0.26)
        sg.reg(0, disc & ~bar, out, 3.0)
        bp, bn = sg.coarse(bar, 0.8, 0.8, 8, 8, 0.25, 0.06)
        sg.reg(4, bp, bn, 1.0)
        sg.reg(5, None, disc, 1.5)
        sg.reg(1, None, disc, 1.0)
    elif c == 18:
        triangle(picto(sg, "excl")[0])
    elif c == 19:
        triangle(picto(sg, "bendL")[0])
    elif c == 20:
        triangle(picto(sg, "bendR")[0])
    elif c == 21:
        triangle(picto(sg, "double")[0])
    elif c == 22:
        triangle(picto(sg, "bumps")[0])
    elif c == 23:
        triangle(picto(sg, "slip")[0])
    elif c == 24:
        triangle(picto(sg, "narrow")[0])
    elif c == 25:
        triangle(picto(sg, "works")[0])
    elif c == 26:
        m, ex = picto(sg, "lights")
        triangle(m, extra=ex)
    elif c == 27:
        triangle(picto(sg, "ped")[0])
    elif c == 28:
        triangle(picto(sg, "children")[0])
    elif c == 29:
        triangle(picto(sg, "cycle")[0])
    elif c == 30:
        triangle(picto(sg, "snow")[0])
    elif c == 31:
        triangle(picto(sg, "deer")[0])
    elif c == 32:
        grey(None)
        sg.reg(5, None, numstr(sg, "80", 0.95, 0.52, 0.12, 0.15), 0.4)
    elif c in (33, 34, 35, 36, 37, 38, 39, 40):
        a = np.zeros(sg.u.shape, bool)
        w = 0.20
        if c == 35:
            a |= sg.seg((0, 0.60), (0, -0.30), w)
            a |= sg.head(0, -0.55, -90, 0.45)
        elif c in (33, 34):
            s = 1 if c == 33 else -1
            a |= sg.stroke([(-0.12 * s, 0.60), (-0.12 * s, -0.10), (0.05 * s, -0.30), (0.30 * s, -0.32)], w)
            a |= sg.head(0.45 * s, -0.32, 0 if s > 0 else 180, 0.42)
        elif c in (36, 37):
            s = 1 if c == 36 else -1
            a |= sg.seg((-0.10 * s, 0.60), (-0.10 * s, -0.30), w)
            a |= sg.head(-0.10 * s, -0.55, -90, 0.42)
            a |= sg.stroke([(-0.10 * s, 0.10), (0.25 * s, 0.10)], w)
            a |= sg.head(0.42 * s, 0.10, 0 if s > 0 else 180, 0.40)
        elif c in (38, 39):
            s = 1 if c == 38 else -1
            a |= sg.seg((-0.30 * s, -0.35), (0.25 * s, 0.25), w)
            a |= sg.head(0.40 * s, 0.40, 45 if s > 0 else 135, 0.45)
        elif c == 40:
            for k in range(3):
                a0 = k * 120 + 20
                pts = sg.arc(0, 0, 0.45, a0, a0 - 80)
                a |= sg.stroke(pts, 0.16)
                th = (a0 - 80) * np.pi / 180
                tip = (0.45 * np.cos(th), 0.45 * np.sin(th))
                a |= sg.head(tip[0], tip[1], a0 - 80 - 90, 0.30)
        bluedisc(a)
    elif c == 41:
        g = car(sg, -0.42, 0.0, 1.0) | car(sg, 0.42, 0.0, 1.0)
        grey(g, gw=0.6)
    elif c == 42:
        g = truck(sg, -0.42, 0.0, 1.0) | car(sg, 0.42, 0.0, 1.0)
        grey(g, gw=0.6)
    return sg.T


TM = np.zeros((NCLASS * NV, NCH, K, K), np.float32)
for c in range(NCLASS):
    for vi, (dx, dy) in enumerate(VARIANTS):
        TM[c * NV + vi] = build_template(c, dx, dy)
conv("tm", "fn", NCLASS * NV, K, stride=1, pad=0, W=TM)
simple("rs", "reshape", "tm", shape=[NCLASS, NV, 25])
simple("p1", "maxpool", "rs", size=[NV, 1], stride=[1, 1])
simple("p2", "maxpool", "p1", size=[1, 25], stride=[1, 1])
simple("scores", "reshape", "p2", shape=[NCLASS])

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": layers}, f)
np.savez(os.path.join(OUT, "weights.npz"), **weights)
print("wrote", len(layers), "layers,", sum(v.size for v in weights.values()), "weights")
