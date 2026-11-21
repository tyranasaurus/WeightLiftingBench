"""Hand-designed MNIST digit classifier: typed stroke models -> oriented-gradient + ink
template bank, matched by correlation over a shift grid.

No data is read. Every number comes from typed knowledge of how digits are written:
each digit style is generated from pen-path waypoints with hand-typed style parameters
(crossbar height, waist height, loop ratio, tail type, base type, slant, ...).
Templates are rendered from those paths, passed through the same oriented
Gaussian-derivative + ink filter bank the network applies to the image (at half
resolution, standardised per image), and stored as convolution kernels. The network
correlates the image feature map with every template over a small shift grid, keeps
the best shift per template, then the best template per class.
"""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
S = 28  # image size
H = S // 2  # features are matched at half resolution for shift tolerance

# ---------------------------------------------------------------- filters
def gauss(sigma, size):
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return g / g.sum()


def dgauss(sigma, size, theta):
    """first derivative of a Gaussian along direction theta (edge normal)."""
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return -(x * np.cos(theta) + y * np.sin(theta)) / (sigma * sigma) * g


def conv_same(x, k):
    h = k.shape[0] // 2
    xp = np.pad(x, h)
    out = np.zeros_like(x)
    for dy in range(k.shape[0]):
        for dx in range(k.shape[1]):
            out += k[dy, dx] * xp[dy:dy + x.shape[0], dx:dx + x.shape[1]]
    return out


ANGLES = [np.pi * t / 6.0 for t in range(6)]          # 0,30,...,150 deg edge normals
DK = [dgauss(1.0, 5, a) for a in ANGLES]
BK = gauss(1.1, 5)                                     # ink-presence (blur) channel
NCH = 2 * len(ANGLES) + 1

# ---------------------------------------------------------------- stroke renderer
def subdivide(pts, step=0.6):
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = np.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(np.ceil(d / step)))
        for t in np.linspace(0.0, 1.0, n + 1)[:-1]:
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    out.append(tuple(pts[-1]))
    return out


def shear_pts(strokes, s):
    return [[(x + s * (14.0 - y), y) for (x, y) in st] for st in strokes]


# hand-typed per-class aspect: real digits fill the 20x20 box wider and a bit
# shorter than a naive slim drawing (widths/heights of the class in general)
ASPECT = {0: (1.22, 0.94), 1: (1.25, 0.97), 2: (1.32, 0.94), 3: (1.22, 0.95),
          4: (1.25, 0.95), 5: (1.32, 0.94), 6: (1.10, 0.95), 7: (1.18, 0.95),
          8: (1.15, 0.94), 9: (1.06, 0.94)}


def scale_pts(strokes, sx, sy):
    return [[(14.0 + sx * (x - 14.0), 14.0 + sy * (y - 14.0)) for (x, y) in st] for st in strokes]


def render(strokes, width=3.0, smooth=0.55):
    canvas = np.zeros((S, S))
    r = width / 2.0
    Y, X = np.mgrid[0:S, 0:S].astype(np.float64)
    for st in strokes:
        for (px, py) in subdivide(st):
            canvas += ((X - px) ** 2 + (Y - py) ** 2 <= r * r)
    canvas = np.clip(canvas, 0, 1)
    return np.clip(conv_same(canvas, gauss(smooth, 5)), 0, 1)


def render_centered(strokes, width=3.0):
    """render, then re-centre so the centre of mass sits at the frame centre,
    exactly like the dataset's normalisation"""
    T = render(strokes, width)
    m = T.sum()
    ys, xs = np.mgrid[0:S, 0:S]
    cx, cy = (T * xs).sum() / m, (T * ys).sum() / m
    shifted = [[(x + 14.0 - cx, y + 14.0 - cy) for (x, y) in st] for st in strokes]
    return render(shifted, width)


def ellipse(cx, cy, a, b, n=28):
    t = np.linspace(0, 2 * np.pi, n + 1)
    return [(cx + a * np.cos(u), cy + b * np.sin(u)) for u in t]


# ---------------------------------------------------------------- typed style generators
# each generator draws one handwriting style of a digit from typed parameters
V = []


def add(cls, strokes, shear=0.0):
    V.append((cls, shear_pts(strokes, shear) if shear else strokes))


def g0(a, b):
    return [ellipse(14, 14, a, b)]


for p in [(6.0, 9.5), (5.4, 9.7), (6.6, 9.2), (5.8, 9.5)]:
    add(0, g0(*p))
add(0, g0(6.2, 9.6), 0.16)
add(0, g0(6.4, 9.3), -0.14)


def g1(flag, serif, slantbar):
    bar = [(16.3, 5.0), (13.0, 23.2)] if slantbar else [(14.3, 5.0), (14.3, 23.2)]
    top = bar[0]
    st = [bar]
    if flag:
        st.append([(10.2, 8.8), top])
    if serif:
        st.append([(10.3, 23.0), (18.6, 23.0)])
    return st


for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]:
    add(1, g1(*p))
add(1, g1(1, 0, 0), 0.2)
add(1, g1(1, 1, 0), 0.15)


def g2(base, loopw):
    top = [(9.2, 9.2), (10.2, 6.2), (12.2, 4.7), (14.6, 4.7), (17.2, 4.9), (19.0, 6.6), (19.2, 9.0)]
    top = [(14 + loopw * (x - 14), y) for (x, y) in top]
    mid = [(18.6, 11.6), (16.4, 14.6), (13.4, 17.6), (10.4, 20.6)]
    bot = ([(9.2, 22.6), (12.0, 22.8), (15.5, 22.8), (19.4, 22.6)] if base == "flat" else
           [(9.6, 22.2), (11.0, 23.4), (14.5, 23.6), (17.6, 23.2), (19.2, 22.2)])
    return [top + mid + bot]


for p in [("flat", 1.0), ("curve", 1.0), ("flat", 0.9), ("curve", 1.1), ("flat", 1.1)]:
    add(2, g2(*p))
add(2, g2("curve", 0.95), 0.15)
add(2, g2("flat", 1.0), -0.1)


def g3(top, wide_bottom):
    if top == "round":
        st = [(8.8, 8.0), (9.6, 5.6), (11.8, 4.5), (14.3, 4.7), (16.9, 5.1), (18.8, 6.8), (18.8, 9.2),
              (17.8, 11.4), (15.6, 12.9), (13.6, 13.5), (16.4, 14.0), (18.6, 15.6), (19.3, 18.2),
              (18.8, 20.9), (16.8, 23.2), (13.8, 23.9), (11.0, 23.4), (9.0, 21.8), (8.4, 20.2)]
    else:
        st = [(8.6, 6.0), (12.0, 5.4), (17.0, 5.2), (14.6, 9.0), (12.9, 13.2), (16.0, 13.8),
              (18.6, 15.6), (19.2, 18.4), (18.6, 21.0), (16.4, 23.2), (13.4, 23.8), (10.8, 23.2),
              (8.9, 21.6), (8.4, 20.0)]
    if wide_bottom:
        st = [(14 + 1.12 * (x - 14), y) if y > 13.5 else (x, y) for (x, y) in st]
    return [st]


for p in [("round", 0), ("flat", 0), ("round", 1), ("flat", 1)]:
    add(3, g3(*p))
add(3, g3("round", 0), 0.15)
add(3, g3("flat", 0), 0.12)
add(3, g3("round", 1), -0.12)


def g4(cy, closed, sx):
    stem = [(sx, 4.8), (sx, 23.8)]
    bar = [(sx - 7.8, cy), (20.0, cy)]
    if closed:
        diag = [(sx - 0.2, 5.2), (sx - 5.0, 9.0), (sx - 6.6, cy - 1.0), (sx - 7.0, cy)]
    else:
        diag = [(sx - 0.4, 5.6), (sx - 4.0, (5.6 + cy) / 2), (sx - 7.8, cy)]
    return [stem, diag, bar]


for p in [(17.5, 0, 16.6), (15.8, 0, 16.6), (19.0, 0, 16.6), (17.5, 1, 16.6), (17.0, 0, 15.6)]:
    add(4, g4(*p))
add(4, g4(16.5, 0, 16.6), 0.14)
add(4, g4(18.0, 0, 16.8), -0.1)


def g5(bowl, barlen):
    bar = [(18.8, 5.2), (14.6, 5.0), (14 - 4.0 * barlen, 5.4)]
    drop = [(14 - 4.0 * barlen + 0.2, 5.4), (14 - 4.2 * barlen, 12.4)]
    if bowl == "round":
        b = [(13.0, 11.6), (16.2, 12.0), (18.4, 14.0), (19.4, 16.8), (18.8, 19.8), (16.6, 22.4),
             (13.6, 23.8), (10.8, 23.4), (8.8, 21.8)]
    elif bowl == "angular":
        b = [(15.8, 12.2), (18.8, 14.4), (19.2, 18.0), (17.4, 21.4), (14.2, 23.4), (11.0, 23.2),
             (8.9, 21.6)]
    else:
        b = [(14.0, 11.8), (17.4, 12.8), (19.0, 15.6), (18.8, 19.0), (17.0, 22.0), (13.0, 23.2),
             (9.6, 23.0)]
    return [bar + drop + b]


for p in [("round", 1.0), ("angular", 1.0), ("flat", 1.0), ("round", 0.9), ("round", 1.1)]:
    add(5, g5(*p))
add(5, g5("angular", 1.0), -0.1)
add(5, g5("round", 1.0), 0.14)


def g6(loop, hook):
    hk = ([(18.4, 6.4), (16.2, 4.6), (12.6, 4.6), (10.2, 6.8)] if hook else
          [(19.2, 8.0), (16.0, 5.6), (12.0, 5.4), (10.2, 6.8)])
    left = [(8.4, 10.0), (7.6, 13.6), (7.8, 17.2)]
    lp = [(8.8, 20.6), (10.8, 23.2), (13.8, 23.9), (16.6, 23.2), (18.4, 21.0), (18.8, 18.2),
          (17.8, 15.8), (15.4, 14.6), (12.6, 14.9), (10.2, 16.4), (8.6, 18.6)]
    lp = [(13 + loop * (x - 13), 19 + loop * (y - 19)) for (x, y) in lp]
    return [hk + left + lp]


for p in [(1.0, 1), (0.85, 1), (1.1, 1), (1.0, 0)]:
    add(6, g6(*p))
add(6, g6(1.0, 1), 0.15)
add(6, g6(0.9, 1), -0.12)


def g7(cross, endx):
    st = [[(8.6, 5.6), (14.0, 5.4), (19.6, 5.6), (16.4 + 0.4 * (endx - 12.6), 14.0), (endx, 23.8)]]
    if cross:
        st.append([(12.8, 15.4), (17.4, 15.2)])
    return st


for p in [(0, 12.6), (1, 12.6), (0, 15.4), (0, 17.0), (1, 15.4)]:
    add(7, g7(*p))
add(7, g7(0, 12.6), 0.16)
add(7, g7(0, 14.0), -0.08)


def g8(wy, ratio):
    ty, by = (5.0 + wy) / 2, (wy + 24.0) / 2
    tb, bb = (wy - 4.6) / 2, (24.0 - wy) / 2
    return [ellipse(14, ty, 4.6 * ratio, tb), ellipse(14, by, 5.4 / ratio, bb)]


for p in [(13.5, 1.0), (12.0, 0.9), (15.0, 1.1), (14.0, 0.95)]:
    add(8, g8(*p))
add(8, g8(13.5, 1.0), 0.15)
add(8, g8(12.5, 1.05), -0.12)


def g9(tail, loop):
    lp = ellipse(13.8, 9.8, 4.9 * loop, 5.0 * loop)
    if tail == "curve":
        t = [(18.7, 9.8), (19.6, 15.0), (18.6, 20.0), (16.0, 23.2), (12.5, 24.0), (10.0, 22.8)]
    elif tail == "straight":
        t = [(18.6, 10.0), (17.8, 20.0), (16.8, 23.6)]
    else:
        t = [(18.6, 10.0), (18.2, 18.0), (16.5, 22.5), (13.5, 23.8), (11.0, 23.0)]
    return [lp, t]


for p in [("curve", 1.0), ("straight", 1.0), ("hook", 0.95), ("straight", 1.05)]:
    add(9, g9(*p))
add(9, g9("curve", 0.9), 0.15)
add(9, g9("hook", 1.0), 0.12)
add(9, g9("curve", 1.0), -0.1)

# ---------------------------------------------------------------- templates -> match kernels
GAMMA = 2.4      # weight of the ink-presence (blank-expectation) channel
BOX_W = 0.55     # how strongly ink outside the expected box is penalised
WIDTHS = (3.0, 2.4)          # thick and thin handwriting
SHEARS = (0.0, 0.12)         # extra slant tolerance around each style's own
JITTERS = (0.0, 0.9)         # second copy of every style with hand tremor


def jitter(strokes, s, rng):
    if s == 0.0:
        return strokes
    return [[(x + rng.normal(0.0, s), y + rng.normal(0.0, s)) for (x, y) in st] for st in strokes]


_rng = np.random.default_rng(5)
TEMPL = [(cls, render_centered(shear_pts(scale_pts(jitter(st, jt, _rng), *ASPECT[cls]), sh), wd))
         for (cls, st) in V for wd in WIDTHS for sh in SHEARS for jt in JITTERS]
NV = len(TEMPL)
W = np.zeros((NV, NCH, H, H))
for v, (cls, T) in enumerate(TEMPL):
    chans = []
    for k in DK:
        p = conv_same(T, k)
        chans.append(np.maximum(p, 0))
        chans.append(np.maximum(-p, 0))
    B = conv_same(T, BK)
    mask = T > 0.25
    ys, xs = np.where(mask)
    box = np.zeros((S, S))
    box[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3] = 1.0
    boxb = conv_same(box, BK)
    # the network averages 2x2 then standardises per image (avgpool + norm ops),
    # so the template channels are built the same way at half resolution
    pool2 = lambda a: a.reshape(H, 2, H, 2).mean((1, 3))
    std = lambda a: (a - a.mean()) / np.sqrt(a.var() + 1e-5)
    chans = [std(pool2(c)) for c in chans]
    kb = GAMMA * (std(pool2(B)) - BOX_W * std(pool2(boxb)))
    chans.append(kb)
    # soft template weights: spread each channel with a Gaussian so edges that
    # land a pixel or two away from the modelled stroke still score partial
    # credit (real handwriting varies in curvature and proportion)
    soft = gauss(1.0, 5)
    chans = [conv_same(c, soft) for c in chans]
    chans = np.stack(chans)
    # unit L2 norm over all channels: with equal template norms, correlation
    # ranking equals nearest-neighbour ranking in feature space; after the norm
    # op the image feature has constant norm sqrt(NCH*H*H), so dividing by it
    # makes every match a cosine in [-1, 1]
    W[v] = chans / np.sqrt((chans ** 2).sum()) / np.sqrt(NCH * H * H)

# ---------------------------------------------------------------- network
MNW = 0.15       # weight of the mean-over-variants tie-breaker beside the max

c1W = np.zeros((NCH, 1, 5, 5))
for i, k in enumerate(DK):
    c1W[2 * i, 0] = k
    c1W[2 * i + 1, 0] = -k
c1W[NCH - 1, 0] = BK

cls_of = np.array([c for c, _ in TEMPL])
ind = np.zeros((10, NV))
ind[cls_of, np.arange(NV)] = 1.0
n_c = ind.sum(1)

# per-class max over its variants: lay variant scores out as (10, NMAX, 1) and
# max-pool along the variant axis (unused slots get a very negative bias)
NMAX = int(n_c.max())
mxW = np.zeros((10 * NMAX, NV))
mxb = np.full(10 * NMAX, -100.0)
for c in range(10):
    vs = np.where(cls_of == c)[0]
    for j, v in enumerate(vs):
        mxW[c * NMAX + j, v] = 1.0
        mxb[c * NMAX + j] = 0.0

weights = {
    "c1.W": c1W.astype(np.float32), "c1.b": np.zeros(NCH, np.float32),
    "m.W": W.astype(np.float32), "m.b": np.zeros(NV, np.float32),
    "mx.W": mxW.astype(np.float32), "mx.b": mxb.astype(np.float32),
    "mn.W": (MNW * ind / n_c[:, None]).astype(np.float32), "mn.b": np.zeros(10, np.float32),
}
net = {"layers": [
    {"name": "c1", "op": "conv", "inputs": ["image"], "out": NCH, "kernel": [5, 5], "stride": 1, "padding": 2},
    {"name": "r1", "op": "relu", "inputs": ["c1"]},
    {"name": "p1", "op": "avgpool", "inputs": ["r1"], "size": [2, 2], "stride": [2, 2]},
    {"name": "n1", "op": "norm", "inputs": ["p1"]},
    {"name": "m", "op": "conv", "inputs": ["n1"], "out": NV, "kernel": [H, H], "stride": 1, "padding": 3},
    {"name": "mp", "op": "maxpool", "inputs": ["m"], "size": [7, 7], "stride": [7, 7]},
    {"name": "fl", "op": "reshape", "inputs": ["mp"], "shape": [NV]},
    {"name": "mx", "op": "dense", "inputs": ["fl"], "out": 10 * NMAX},
    {"name": "mr", "op": "reshape", "inputs": ["mx"], "shape": [10, NMAX, 1]},
    {"name": "mpv", "op": "maxpool", "inputs": ["mr"], "size": [NMAX, 1], "stride": [NMAX, 1]},
    {"name": "mv", "op": "reshape", "inputs": ["mpv"], "shape": [10]},
    {"name": "mn", "op": "dense", "inputs": ["fl"], "out": 10},
    {"name": "out", "op": "add", "inputs": ["mv", "mn"]},
]}

os.makedirs(OUT, exist_ok=True)
np.savez(os.path.join(OUT, "weights.npz"), **weights)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump(net, f)
print(f"wrote {NV} templates, {sum(w.size for w in weights.values())} weights")
