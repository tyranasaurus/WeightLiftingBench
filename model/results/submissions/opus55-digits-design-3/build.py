"""Hand-designed digit classifier: typed stroke models of the ten digits are rendered into templates;
the network compares an image with every template using oriented-edge and enclosed-hole features
and outputs, for each class, the best match among that class's templates.
No data is read: every number below comes from this code."""
import json, os, sys
import numpy as np

# ================================================================ geometry helpers
def cr(P, n=20, closed=False):
    """Catmull-Rom spline through control points P (k, 2)."""
    P = np.asarray(P, float)
    if closed:
        ext = np.vstack([P[-1], P, P[0], P[1]]); segs = len(P)
    else:
        ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]]); segs = len(P) - 1
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    out = []
    for i in range(segs):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(ext[segs + 1:segs + 2] if closed else P[-1:])
    return np.vstack(out)


def poly(P):
    return np.asarray(P, float)


def arc(cx, cy, rx, ry, a0, a1):
    """Ellipse arc; angles in degrees, 90 = top (y grows downwards)."""
    n = max(8, int(abs(a1 - a0) / 4))
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy - ry * np.sin(a)], 1)


class Jit:
    """Small independent jitter of control points (handwriting wobble)."""
    def __init__(self, r, amt):
        self.r, self.amt = r, amt

    def __call__(self, P):
        P = np.asarray(P, float)
        return P + self.r.uniform(-self.amt, self.amt, P.shape)


# ================================================================ digit models
# x to the right, y downwards; each digit spans roughly y in [0, 1]. Each model draws its own style
# parameters from the generator r; the numbers are typed from general knowledge of how digits are written.

def d0(r, J):
    rx = 0.5 * r.uniform(0.4, 0.95)
    a0 = r.uniform(30, 150)
    over = r.uniform(-25, 45)                      # gap (<0) or overlap (>0) where the pen closes the loop
    a = np.radians(np.linspace(a0, a0 + 360 + over, 100))
    ph = r.uniform(0, 2 * np.pi)
    m = (1 + r.uniform(-0.07, 0.07) * np.cos(2 * a + ph)) * (1 - r.uniform(0.0, 0.1) * np.linspace(0, 1, len(a)))
    x, y = rx * np.cos(a) * m, -0.5 * np.sin(a) * m
    x = x * (1 + r.uniform(-0.3, 0.3) * y)         # egg shape
    P = np.stack([x, y], 1)
    if r.uniform() < 0.25:                         # the pen runs on a little past the closing point
        d = P[-1] - P[-4]
        P = np.vstack([P, P[-1] + d / np.linalg.norm(d) * r.uniform(0.04, 0.12)])
    return [P]


def d1(r, J):
    style = r.choice(3, p=[0.75, 0.18, 0.07])     # plain stroke, with flag, with flag and base
    b = r.uniform(-0.07, 0.07)
    S = [cr(J([(0, 0), (b, 0.5), (0, 1)]))]
    if style >= 1:
        S.append(poly(J([(r.uniform(-0.3, -0.1), r.uniform(0.08, 0.3)), (0, 0)])))
    if style == 2:
        w = r.uniform(0.15, 0.3)
        S.append(poly(J([(-w, 1), (w, 1)])))
    return S


def d2(r, J):
    style = r.choice(4, p=[0.6, 0.2, 0.12, 0.08])   # plain, loop (pen turns right), loop (pen turns left), Z-like
    end = (r.uniform(0.0, 0.15), 1.0)
    base = (r.uniform(0.7, 1.05), 1.0 + r.uniform(-0.1, 0.04))
    if style <= 2:
        sx, sy = r.uniform(0.0, 0.2), r.uniform(0.1, 0.32)
        tr = r.uniform(0.65, 0.9)
        top = [(sx, sy), (0.25, 0.03), (0.5, 0.0), (tr, 0.1), (tr + 0.02, r.uniform(0.25, 0.4))]
        if style == 0 and r.uniform() < 0.25:      # small curl where the pen starts
            top = [(sx + 0.1, sy + 0.1), (sx - 0.03, sy + 0.08)] + top
        mid = (r.uniform(0.35, 0.62), r.uniform(0.5, 0.68))
        if style == 0:
            dip = r.uniform(-0.07, 0.05)
            return [cr(J(top + [mid, end])), cr(J([end, ((end[0] + base[0]) / 2, 1.0 + dip), base]))]
        L = r.uniform(0.08, 0.17)
        base = (max(base[0], 0.8), base[1])
        if style == 1:
            cx, cy = r.uniform(0.15, 0.25), 1.0 - 1.2 * L
            e = (cx + 0.7 * L, cy + 0.7 * L)
            ex = (cx + L, cy)
            return [cr(J(top + [mid, e])), arc(cx, cy, L, L, 315, -5), cr(J([ex, (ex[0] + 0.1, 0.99), (0.6, 1.0), base]))]
        cx = r.uniform(0.08, 0.15) + L
        lp = arc(cx, 1.0 - L, L * r.uniform(0.8, 1.1), L, 180, 630)
        return [cr(J(top + [mid, (cx - L, 1.0 - L)])), lp, poly(J([(cx, 1.0), base]))]
    tl = (r.uniform(0.0, 0.15), r.uniform(0.0, 0.1))
    trp = (r.uniform(0.75, 0.95), 0.0)
    return [poly(J([tl, trp])), cr(J([trp, (trp[0] - 0.02, 0.15), (0.45, 0.55), end])), poly(J([end, base]))]


def d3(r, J):
    style = r.choice(2, p=[0.8, 0.2])             # round top, flat top
    my = r.uniform(0.4, 0.52)
    mx = r.uniform(0.28, 0.5)
    ux = r.uniform(0.6, 0.85)
    lx = r.uniform(0.75, 1.0)
    S = []
    if style == 0:
        up = [(r.uniform(0.0, 0.15), r.uniform(0.06, 0.22)), (0.35, 0.0), (ux - 0.05, 0.05), (ux, my * 0.5), (mx, my)]
        S.append(cr(J(up)))
    else:
        tl, trp = (r.uniform(0.0, 0.15), 0.0), (ux, r.uniform(-0.03, 0.03))
        S += [poly(J([tl, trp])), poly(J([trp, (mx, my)]))]
    lo = [(mx, my), (lx - 0.12, my + 0.08), (lx, 0.78), (0.55, 1.0), (r.uniform(0.0, 0.15), r.uniform(0.8, 0.95))]
    S.append(cr(J(lo)))
    return S


def d4(r, J):
    style = r.choice(3, p=[0.62, 0.22, 0.16])     # open top, closed triangle, rounded (u-like) left stroke
    vx = r.uniform(0.5, 0.8)
    hy = r.uniform(0.45, 0.72)
    hr = vx + (r.uniform(0.05, 0.3) if style != 1 else r.uniform(0.12, 0.3))
    vtop = r.uniform(-0.05, 0.25)
    vb = (vx + r.uniform(-0.12, 0.04), 1.0)
    if style == 0:
        S = [poly(J([(r.uniform(0.0, 0.35), 0.0), (r.uniform(-0.05, 0.1), hy), (hr, hy + r.uniform(-0.06, 0.03))]))]
    elif style == 1:
        vtop = 0.0
        S = [poly(J([(vx - r.uniform(0.0, 0.12), 0.0), (0.0, hy), (hr, hy)]))]
    else:
        S = [cr(J([(0.05, 0.0), (0.04, hy - 0.2), (0.25, hy), (vx, hy - r.uniform(0.0, 0.1))]))]
    S.append(poly(J([(vx, vtop), vb])))
    return S


def d5(r, J):
    style = r.choice(3, p=[0.5, 0.3, 0.2])        # sharp corner, rounded corner, flag lifted off the stem
    fx0, fx1 = r.uniform(0.12, 0.3), r.uniform(0.65, 1.0)
    ky = r.uniform(0.33, 0.5)
    kx = fx0 - r.uniform(-0.02, 0.1)
    bx = r.uniform(0.75, 1.0)
    tail = (r.uniform(-0.02, 0.15), r.uniform(0.8, 0.95))
    bowl = [(kx, ky), (0.45, ky - r.uniform(0.0, 0.1)), (bx, ky + 0.17), (bx, 0.78), (0.5, 1.0), tail]
    flag = (fx1, r.uniform(-0.06, 0.06))
    if style == 0:
        return [poly(J([flag, (fx0, 0.0)])), poly(J([(fx0, 0.0), (kx, ky)])), cr(J(bowl))]
    if style == 1:
        return [poly(J([flag, (fx0, 0.0)])), cr(J([(fx0, 0.0)] + bowl))]
    g = r.uniform(0.05, 0.15)
    return [poly(J([flag, (fx0 + g, 0.0)])), poly(J([(fx0, 0.02), (kx, ky)])), cr(J(bowl))]


def d6(r, J):
    lt = r.uniform(0.38, 0.62)                    # top of the loop (the stem keeps at least ~40% of the height)
    rx = r.uniform(0.28, 0.45)
    cy, ry, cx = (1 + lt) / 2, (1 - lt) / 2, r.uniform(0.4, 0.55)
    sx = r.uniform(0.15, 0.9)                     # from b-like (stem straight above the loop) to a long slanted stem
    bend = r.uniform(-0.1, 0.12)
    stem = cr(J([(sx, 0.0), ((sx + cx - rx) / 2 - bend, cy * 0.45), (cx - rx, cy)]))
    loop = arc(cx, cy, rx, ry, 180, 540 + r.uniform(-35, 15))
    return [np.vstack([stem, loop[1:]])]


def d7(r, J):
    style = r.choice(5, p=[0.45, 0.1, 0.1, 0.2, 0.15])   # plain, tick at top left, crossbar, curved stem, rounded corner
    tl = (0.0, r.uniform(0.0, 0.12))
    trp = (r.uniform(0.45, 0.9), r.uniform(-0.04, 0.04))
    bot = (trp[0] - r.uniform(0.1, 0.6), 1.0)
    S = [poly(J([tl, trp]))]
    if style == 3:
        b = r.uniform(0.03, 0.12)
        S.append(cr(J([trp, ((trp[0] + bot[0]) / 2 + b, 0.5), bot])))
    elif style == 4:
        S = [cr(J([tl, (trp[0] - 0.12, trp[1]), (trp[0], trp[1] + 0.1), ((trp[0] + bot[0]) / 2 + 0.03, 0.55), bot]))]
    else:
        S.append(poly(J([trp, bot])))
    if style == 1:
        S.append(poly(J([(tl[0] + r.uniform(-0.05, 0.03), tl[1] + r.uniform(0.12, 0.25)), tl])))
    if style == 2:
        yc = r.uniform(0.42, 0.58)
        xc = trp[0] + (bot[0] - trp[0]) * yc
        w = r.uniform(0.12, 0.25)
        S.append(poly(J([(xc - w, yc), (xc + w, yc - 0.02)])))
    return S


def d8(r, J):
    style = r.choice(3, p=[0.5, 0.25, 0.25])      # one closed stroke, two stacked loops, S plus open return stroke
    cy = r.uniform(0.35, 0.58)                    # where the strokes cross
    ux, lx = 0.5 + r.uniform(-0.05, 0.15), 0.5 + r.uniform(-0.08, 0.05)
    uw, lw = r.uniform(0.14, 0.3), r.uniform(0.16, 0.36)
    if r.uniform() < 0.2:                         # wide upper loop over a collapsed, narrow lower loop
        uw, lw = r.uniform(0.2, 0.3), r.uniform(0.08, 0.15)
    cxp = (ux + lx) / 2
    if style == 2:
        uw, lw = max(uw, 0.2), max(lw, 0.22)      # an open upper loop needs some width to stay an 8
        cxp = (ux + lx) / 2
        P = [(ux + uw * 0.9, 0.12), (ux, 0.0), (ux - uw, 0.14), (ux - uw * 0.6, cy * 0.8), (cxp, cy),
             (lx + lw, cy + (1 - cy) * 0.4), (lx + lw * 0.8, 0.88), (lx, 1.0), (lx - lw, 0.86),
             (lx - lw * 0.85, cy + (1 - cy) * 0.3), (cxp, cy), (ux + uw * 0.8, cy * 0.55),
             (ux + uw * r.uniform(0.9, 1.2), r.uniform(0.12, 0.28))]
        return [cr(J(P))]
    if style == 0:
        P = [(ux + uw * 0.8, 0.1), (ux, 0.0), (ux - uw, 0.14), (ux - uw * 0.6, cy * 0.8), (cxp, cy),
             (lx + lw, cy + (1 - cy) * 0.4), (lx + lw * 0.8, 0.88), (lx, 1.0), (lx - lw, 0.86),
             (lx - lw * 0.85, cy + (1 - cy) * 0.3), (cxp, cy), (ux + uw * 0.8, cy * 0.6)]
        return [cr(J(P), closed=True)]
    return [arc(ux, cy / 2, uw, cy / 2, 0, 360), arc(lx, (1 + cy) / 2, lw, (1 - cy) / 2, 0, 360)]


def d9(r, J):
    lb = r.uniform(0.35, 0.6)                     # bottom of the loop
    rx = r.uniform(0.22, 0.42)
    cx, cy, ry = 0.45, lb / 2, lb / 2
    a0 = r.uniform(-20, 30)
    loop = arc(cx, cy, rx, ry, a0, a0 + 360 + r.uniform(-35, 20))
    sp = (cx + rx * np.cos(np.radians(a0)) * 0.98, cy - ry * np.sin(np.radians(a0)))
    style = r.choice(3, p=[0.55, 0.2, 0.25])      # straight stem, hooked stem, slanted stem
    if style == 0:
        stem = poly(J([sp, (sp[0] + r.uniform(-0.12, 0.04), 1.0)]))
    elif style == 1:
        stem = cr(J([sp, (sp[0] - r.uniform(0.0, 0.1), 0.72), (sp[0] - r.uniform(0.2, 0.35), 1.0)]))
    else:
        stem = poly(J([sp, (sp[0] - r.uniform(0.15, 0.3), 1.0)]))
    return [loop, stem]


MODELS = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]
SLANT = {1: (-0.4, 0.8), 8: (-0.1, 0.85)}         # shear of the top to the right, per unit height
SLANT_DEFAULT = (-0.2, 0.7)

# ================================================================ rendering (MNIST-style normalisation)
_YY, _XX = np.mgrid[0:28, 0:28].astype(float) + 0.5
_PIX = np.stack([_XX.ravel(), _YY.ravel()], 1)


def _draw(strokes, rad):
    d = np.full(784, np.inf)
    for S in strokes:
        A, B = S[:-1], S[1:]
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-12)
        PA = _PIX[:, None, :] - A[None]
        t = np.clip((PA * AB[None]).sum(2) / L2[None], 0, 1)
        q = PA - t[..., None] * AB[None]
        d = np.minimum(d, np.sqrt((q ** 2).sum(2)).min(1))
    return np.clip(rad + 0.5 - d, 0, 1).reshape(28, 28)    # pen of radius rad with a one-pixel soft edge


def transform(strokes, slant, rot, aspect):
    pts = np.vstack(strokes)
    c = (pts.min(0) + pts.max(0)) / 2
    ca, sa = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    out = []
    for S in strokes:
        q = S - c
        x, y = q[:, 0] * aspect, q[:, 1]
        x = x - slant * y
        out.append(np.stack([ca * x - sa * y, sa * x + ca * y], 1))
    return out


def render(strokes, rad):
    """Scale the ink to fit a 20 x 20 box (keeping proportions), then put its centre of mass at the frame centre."""
    pts = np.vstack(strokes)
    lo, hi = pts.min(0), pts.max(0)
    k = (20 - 2 * rad) / max((hi - lo).max(), 1e-6)
    c = (lo + hi) / 2
    S = [(s - c) * k + 14 for s in strokes]
    img = _draw(S, rad)
    m = img.sum()
    shift = np.array([14 - (img * _XX).sum() / m, 14 - (img * _YY).sum() / m])
    return _draw([s + shift for s in S], rad)


def sample(digit, r):
    S = MODELS[digit](r, Jit(r, 0.03))
    lo, hi = SLANT.get(digit, SLANT_DEFAULT)
    S = transform(S, r.uniform(lo, hi), r.uniform(-10, 10), r.uniform(0.75 if digit == 8 else 0.65, 1.2))
    rad = r.uniform(0.6, 1.4) if r.uniform() < 0.7 else r.uniform(1.4, 2.3)
    if digit == 8:
        rad = min(rad, 1.8)                        # a bolder pen fills both loops of a narrow 8
    return render(S, rad)


# ================================================================ features (the same arithmetic as the network layers)
K = 8              # edge directions, 45 degrees apart
SIG = 1.0          # Gaussian-derivative scale in pixels
SHARP = 0.5        # orientation sharpening: s_k = relu(e_k - SHARP * (e_{k+2} + e_{k-2}))
GAP_CLOSE = 0.7    # the flood-fill barrier is max(ink, GAP_CLOSE * 3x3-dilated ink): closes 1-2 pixel gaps softly
FLOOD_STEPS = 28   # flood-fill steps from the frame border (one pixel per step)
HOLE_W = 1.2       # weight of the hole channel next to the edge channels
C = K + 1          # feature channels


def edge_kernels():
    o = np.arange(-3, 4, dtype=float)
    g = np.exp(-o ** 2 / (2 * SIG ** 2)); g /= g.sum()
    dg = o * g; dg /= dg[o > 0].sum()          # unit response to a unit step edge
    Kx, Ky = np.outer(g, dg), np.outer(dg, g)  # indexed [dy, dx]
    th = np.arange(K) * 2 * np.pi / K
    return np.stack([np.cos(t) * Kx + np.sin(t) * Ky for t in th])[:, None]   # (K, 1, 7, 7)


def sharpen_matrix():
    M = np.eye(K)
    for k in range(K):
        M[k, (k + 2) % K] -= SHARP
        M[k, (k - 2) % K] -= SHARP
    return M


def conv_np(x, W, pad):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = h + 2 * pad - kh + 1, w + 2 * pad - kw + 1
    out = np.zeros((n, o, ho, wo))
    for dy in range(kh):
        for dx in range(kw):
            out += np.einsum('nchw,oc->nohw', xp[:, :, dy:dy + ho, dx:dx + wo], W[:, :, dy, dx])
    return out


def maxpool_np(x, s, st):
    n, c, h, w = x.shape
    ho, wo = (h - s) // st + 1, (w - s) // st + 1
    return np.max(np.stack([x[:, :, dy:dy + st * ho:st, dx:dx + st * wo:st] for dy in range(s) for dx in range(s)]), 0)


def avgpool_np(x, s, st):
    n, c, h, w = x.shape
    ho, wo = (h - s) // st + 1, (w - s) // st + 1
    return np.mean(np.stack([x[:, :, dy:dy + st * ho:st, dx:dx + st * wo:st] for dy in range(s) for dx in range(s)]), 0)


def pad1(x):
    return np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))


def holes_np(ink):
    """ink (n, 1, 28, 28) -> enclosed background (n, 1, 28, 28): background the border flood cannot reach."""
    barrier = ink + np.maximum(GAP_CLOSE * maxpool_np(pad1(ink), 3, 1) - ink, 0)
    v = np.zeros_like(ink) - 1.0                # v = outside - 1; the zero padding is the outside world
    for _ in range(FLOOD_STEPS):
        M = maxpool_np(pad1(v), 3, 1)
        v = M + -np.maximum(M + barrier, 0)     # min(dilated v, -barrier)
    return np.maximum(-(v + barrier), 0)        # enclosure above the barrier itself (no halo around strokes)


def features(imgs):
    """imgs (n, 28, 28) -> F (n, C, 14, 14) and its 3x3 dilation D."""
    x = imgs[:, None]
    e = np.maximum(conv_np(x, edge_kernels(), 3), 0)
    s = np.maximum(np.einsum('nchw,oc->nohw', e, sharpen_matrix()), 0)
    E = maxpool_np(s, 2, 2)
    H = maxpool_np(holes_np(x), 2, 2)
    F = np.concatenate([E, HOLE_W * H], 1)
    D = maxpool_np(pad1(F), 3, 1)
    return F, D


def unit_rows(X):
    X = X.reshape(len(X), -1)
    X = X - X.mean(1, keepdims=True)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ================================================================ network
COUNTS = [3500, 1800, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500]   # templates per class (a plain stroke needs fewer)
BLOCK = 100        # templates are maxed in blocks of this size, then per class over its blocks


def conv1x1(name, inp, W, b=None, pad=0):
    W = np.asarray(W, float)
    o = W.shape[0]
    return ({"name": name, "op": "conv", "inputs": [inp], "out": o, "kernel": [1, 1], "stride": 1, "padding": pad},
            {name + ".W": W.reshape(o, -1, 1, 1), name + ".b": np.zeros(o) if b is None else np.asarray(b, float)})


def build(out_dir):
    tA, tB = [], []
    for digit in range(10):
        r = np.random.RandomState(1000 + digit)
        imgs = np.stack([sample(digit, r) for _ in range(COUNTS[digit])])
        for i in range(0, COUNTS[digit], 400):
            F, D = features(imgs[i:i + 400])
            tA.append(unit_rows(F).astype(np.float32))
            tB.append(unit_rows(D).astype(np.float32))
    tA, tB = np.concatenate(tA), np.concatenate(tB)
    N = sum(COUNTS)
    layers, W = [], {}

    def add_layer(L, w=None):
        layers.append(L)
        if w: W.update(w)

    # --- oriented edges
    add_layer({"name": "edge", "op": "conv", "inputs": ["image"], "out": K, "kernel": [7, 7], "stride": 1, "padding": 3},
              {"edge.W": edge_kernels(), "edge.b": np.zeros(K)})
    add_layer({"name": "edge_r", "op": "relu", "inputs": ["edge"]})
    add_layer(*conv1x1("orient", "edge_r", sharpen_matrix()))
    add_layer({"name": "orient_r", "op": "relu", "inputs": ["orient"]})
    add_layer({"name": "E", "op": "maxpool", "inputs": ["orient_r"], "size": [2, 2], "stride": [2, 2]})
    # --- holes: flood fill from the border
    add_layer(*conv1x1("ink_pad", "image", [[1.0]], pad=1))
    add_layer({"name": "ink_dil", "op": "maxpool", "inputs": ["ink_pad"], "size": [3, 3], "stride": [1, 1]})
    add_layer(*conv1x1("ink_dil_s", "ink_dil", [[GAP_CLOSE]]))
    add_layer(*conv1x1("ink_neg", "image", [[-1.0]]))
    add_layer({"name": "gap", "op": "add", "inputs": ["ink_dil_s", "ink_neg"]})
    add_layer({"name": "gap_r", "op": "relu", "inputs": ["gap"]})
    add_layer({"name": "barrier", "op": "add", "inputs": ["image", "gap_r"]})
    add_layer(*conv1x1("v0", "image", [[0.0]], b=[-1.0]))
    v = "v0"
    for t in range(FLOOD_STEPS):
        add_layer(*conv1x1(f"fl{t}_pad", v, [[1.0]], pad=1))
        add_layer({"name": f"fl{t}_max", "op": "maxpool", "inputs": [f"fl{t}_pad"], "size": [3, 3], "stride": [1, 1]})
        add_layer({"name": f"fl{t}_z", "op": "add", "inputs": [f"fl{t}_max", "barrier"]})
        add_layer({"name": f"fl{t}_zr", "op": "relu", "inputs": [f"fl{t}_z"]})
        add_layer(*conv1x1(f"fl{t}_zn", f"fl{t}_zr", [[-1.0]]))
        add_layer({"name": f"fl{t}_v", "op": "add", "inputs": [f"fl{t}_max", f"fl{t}_zn"]})
        v = f"fl{t}_v"
    add_layer({"name": "hole_s", "op": "add", "inputs": [v, "barrier"]})
    add_layer(*conv1x1("hole_n", "hole_s", [[-1.0]]))
    add_layer({"name": "hole", "op": "relu", "inputs": ["hole_n"]})
    add_layer({"name": "H", "op": "maxpool", "inputs": ["hole"], "size": [2, 2], "stride": [2, 2]})
    # --- stack edges and holes into C channels
    PE = np.zeros((C, K)); PE[:K, :K] = np.eye(K)
    PH = np.zeros((C, 1)); PH[K, 0] = HOLE_W
    add_layer(*conv1x1("F_e", "E", PE))
    add_layer(*conv1x1("F_h", "H", PH))
    add_layer({"name": "F", "op": "add", "inputs": ["F_e", "F_h"]})
    add_layer(*conv1x1("F_pad", "F", np.eye(C), pad=1))
    add_layer({"name": "D", "op": "maxpool", "inputs": ["F_pad"], "size": [3, 3], "stride": [1, 1]})
    # --- correlation with every template (both directions), best template per class
    add_layer({"name": "F_flat", "op": "reshape", "inputs": ["F"], "shape": [1, C * 14, 14]})
    add_layer({"name": "F_n", "op": "norm", "inputs": ["F_flat"]})
    add_layer({"name": "D_flat", "op": "reshape", "inputs": ["D"], "shape": [1, C * 14, 14]})
    add_layer({"name": "D_n", "op": "norm", "inputs": ["D_flat"]})
    add_layer({"name": "match_A", "op": "dense", "inputs": ["D_n"], "out": N}, {"match_A.W": tA, "match_A.b": np.zeros(N)})
    add_layer({"name": "match_B", "op": "dense", "inputs": ["F_n"], "out": N}, {"match_B.W": tB, "match_B.b": np.zeros(N)})
    add_layer({"name": "match", "op": "add", "inputs": ["match_A", "match_B"]})
    nb = [c // BLOCK for c in COUNTS]
    NB, BM = sum(nb), max(nb)
    sel = np.zeros((10 * BM, NB))                 # copies each class's block maxima into BM slots (repeating the last)
    first = np.cumsum([0] + nb)
    for c in range(10):
        for j in range(BM):
            sel[c * BM + j, first[c] + min(j, nb[c] - 1)] = 1.0
    add_layer({"name": "match_blocks", "op": "reshape", "inputs": ["match"], "shape": [1, NB, BLOCK]})
    add_layer({"name": "block_best", "op": "maxpool", "inputs": ["match_blocks"], "size": [1, BLOCK], "stride": [1, BLOCK]})
    add_layer({"name": "class_blocks", "op": "dense", "inputs": ["block_best"], "out": 10 * BM},
              {"class_blocks.W": sel, "class_blocks.b": np.zeros(10 * BM)})
    add_layer({"name": "class_3d", "op": "reshape", "inputs": ["class_blocks"], "shape": [10, BM, 1]})
    add_layer({"name": "best", "op": "maxpool", "inputs": ["class_3d"], "size": [BM, 1], "stride": [BM, 1]})
    add_layer({"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]})

    W = {k: np.asarray(v, np.float32) for k, v in W.items()}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **W)


if __name__ == "__main__":
    build(sys.argv[1])
