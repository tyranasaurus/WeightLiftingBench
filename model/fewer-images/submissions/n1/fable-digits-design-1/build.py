"""Builds the digit network: weights come only from formulas and from pen-path drawings typed below.
Usage: python3 build.py OUT_DIR   (reads no files; writes OUT_DIR/network.json and OUT_DIR/weights.npz)

Network
  image -> 8 gradient-direction maps (Gaussian-derivative filters, direction decomposition with relus)
        -> Gaussian blur + subsampling to 9 x 9 cells -> piecewise-linear square root -> norm of the whole feature vector
        -> four overlapping 6 x 6-cell regions; in each, a bank of templates (one per drawn digit, each normalised by itself)
           scored by negative squared distance; max over each class's templates (in 4 groups, averaged) -> sum over regions.
"""
import json, sys
import numpy as np


P = 128          # path points per glyph


# ----------------------------------------------------------------------------- splines
def crom(K, n=10):
    """Centripetal Catmull-Rom spline through knots K (k, 2); returns a dense polyline."""
    K = np.asarray(K, dtype=np.float64)
    if len(K) == 2:
        t = np.linspace(0, 1, n + 1)[:, None]
        return K[0] * (1 - t) + K[1] * t
    Q = np.vstack([2 * K[0] - K[1], K, 2 * K[-1] - K[-2]])
    p0, p1, p2, p3 = Q[:-3, None], Q[1:-2, None], Q[2:-1, None], Q[3:, None]

    def dt(a, b):
        return np.maximum(np.sqrt(np.sqrt(((a - b) ** 2).sum(-1, keepdims=True))), 1e-4)

    t0 = 0.0
    t1 = t0 + dt(p0, p1)
    t2 = t1 + dt(p1, p2)
    t3 = t2 + dt(p2, p3)
    u = np.linspace(0, 1, n, endpoint=False)[None, :, None]
    t = t1 + (t2 - t1) * u
    A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
    C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
    return np.vstack([C.reshape(-1, 2), K[-1:]])


def stroke(*segs):
    """One pen stroke made of smooth pieces; consecutive pieces meet at a sharp corner."""
    return np.vstack([crom(s) for s in segs])


def resample(strokes, npts=P):
    """Spread npts points evenly (by arc length) over all strokes of a glyph. Returns (npts, 2) and stroke index (npts,)."""
    lens = []
    for s in strokes:
        d = np.sqrt((np.diff(s, axis=0) ** 2).sum(1))
        lens.append(d.sum())
    lens = np.array(lens)
    cnt = np.maximum(2, np.floor(npts * lens / lens.sum()).astype(int))
    while cnt.sum() > npts:
        cnt[np.argmax(cnt)] -= 1
    while cnt.sum() < npts:
        cnt[np.argmax(lens / cnt)] += 1
    out, pos = [], []
    for s, c in zip(strokes, cnt):
        d = np.sqrt((np.diff(s, axis=0) ** 2).sum(1))
        a = np.concatenate([[0], np.cumsum(d)])
        u = np.linspace(0, a[-1], c)
        out.append(np.stack([np.interp(u, a, s[:, 0]), np.interp(u, a, s[:, 1])], 1))
        pos.append(np.linspace(0, 1, c))
    return np.vstack(out), np.concatenate(pos)


# ----------------------------------------------------------------------------- digit models
# Nominal frame: x in [0, 1] left to right, y in [0, 1] top to bottom. Each model takes the random generator r and a
# looseness factor s (how shaky the hand is) and returns (strokes, aspect): pen strokes and the width / height ratio by
# which x is then scaled. All shapes and ranges are typed from general knowledge of how people write digits.
def _j(r, K, s=0.02):
    K = np.asarray(K, dtype=np.float64)
    return K + r.normal(0, s, K.shape)


def _ell(cx, cy, rx, ry, degs):
    t = np.deg2rad(np.asarray(degs, dtype=np.float64))
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def d0(r, s):
    n = 13
    th0 = -90 + r.normal(15, 20)
    sweep = 360 + r.uniform(-30, 55)
    f = np.linspace(0, 1, n)
    t = np.deg2rad(th0 - f * sweep)
    ex = r.uniform(1.55, 2.6)
    c, sn = np.cos(t), np.sin(t)
    x = np.sign(c) * np.abs(c) ** (2 / ex)
    y = np.sign(sn) * np.abs(sn) ** (2 / ex)
    drift = 1 + f * r.uniform(-0.14, 0.12)
    x = x * (1 + r.uniform(-0.25, 0.25) * y) * drift
    y = y * drift
    K = _j(r, np.stack([0.5 + 0.5 * x, 0.5 + 0.5 * y], 1), 0.015 * s)
    if r.random() < 0.15:  # lead-in from the upper right
        K = np.vstack([K[0] + [r.uniform(0.05, 0.2), -r.uniform(0.0, 0.12)], K])
    return [stroke(K)], r.uniform(0.45, 1.0)


def d1(r, s):
    style = r.choice(4, p=[0.62, 0.22, 0.12, 0.04])
    main = np.array([[0.5, 0], [0.5 + r.normal(0, 0.025 * s), 0.5], [0.5, 1.0]])
    S = []
    if style in (1, 2):   # flag: a short up-stroke into the top
        fl, phi = r.uniform(0.15, 0.42), np.deg2rad(r.uniform(25, 65))
        S.append(stroke([[0.5 - fl * np.cos(phi), fl * np.sin(phi)], [0.5, 0]], main))
    else:
        S.append(stroke(main))
    if style in (2, 3):   # foot
        S.append(stroke([[0.5 - r.uniform(0.15, 0.4), 1 + r.normal(0, 0.02)], [0.5 + r.uniform(0.15, 0.4), 1 + r.normal(0, 0.02)]]))
    return S, 1.0


def d2(r, s):
    style = r.choice(4, p=[0.4, 0.28, 0.24, 0.08])
    xs, ys = r.uniform(0.05, 0.25), r.uniform(0.17, 0.35)
    xr, yr = r.uniform(0.75, 0.92), r.uniform(0.2, 0.34)
    top = [[xs, ys], [0.28, 0.06], [0.55, 0.0], [xr - 0.06, yr - 0.18], [xr, yr], [0.7, 0.52]]
    if r.random() < 0.3:  # hooked start
        top = [[xs + r.uniform(0.05, 0.12), ys + r.uniform(0.06, 0.14)]] + top
    th = r.uniform(0.75, 1.25)   # small or large top arc
    top = [[x, y * th] for x, y in top[:-1]] + [top[-1]]
    ye = r.uniform(-0.12, 0.04)
    if style == 0:    # sharp corner at the lower left, then the base
        a = _j(r, top + [[0.42, 0.75], [0.12, 0.97]], 0.02 * s)
        b = np.vstack([a[-1], _j(r, [[0.5, 0.95 + r.normal(0, 0.03)], [0.95, 0.97 + ye]], 0.02 * s)])
        S = [stroke(a, b)]
    elif style == 1:  # small loop at the lower left
        ls = r.uniform(0.5, 2.0)
        K = top + [[0.47, 0.72], [0.27, 0.92], [0.13, 1.0], [0.13 - 0.1 * ls, 0.94 - 0.03 * ls], [0.13 - 0.03 * ls, 0.94 - 0.13 * ls],
                   [0.32, 0.85], [0.62, 0.93], [0.97, 0.96 + ye]]
        S = [stroke(_j(r, K, 0.015 * s))]
    elif style == 2:  # rounded bottom running into a rising base
        K = top + [[0.45, 0.72], [0.3, 0.88], [0.4, 1.0], [0.66, 0.97], [0.95, 0.85 + ye]]
        S = [stroke(_j(r, K, 0.02 * s))]
    else:             # angular top, like a Z with a bowed roof
        a = _j(r, [[xs, 0.1], [0.5, 0.0], [xr, 0.08]], 0.02 * s)
        b = np.vstack([a[-1], _j(r, [[0.55, 0.55], [0.12, 0.97]], 0.02 * s)])
        c = np.vstack([b[-1], _j(r, [[0.5, 0.95 + r.normal(0, 0.03)], [0.95, 0.97 + ye]], 0.02 * s)])
        S = [stroke(a, b, c)]
    return S, r.uniform(0.55, 1.1)


def d3(r, s):
    cx = r.uniform(0.3, 0.55)
    wl = r.uniform(0.9, 1.12)
    low = [[cx + 0.25, 0.51], [0.87 * wl, 0.68], [0.8 * wl, 0.89], [0.52, 1.0], [0.27, 0.97], [0.08, 0.86]]
    if r.random() < 0.25:
        low = low + [[0.07, 0.78]]
    style = r.choice(3, p=[0.5, 0.25, 0.25])
    if style == 0:    # round top, sharp waist
        a = _j(r, [[0.15, 0.15], [0.38, 0.02], [0.62, 0.02], [0.82, 0.14], [0.78, 0.32], [cx, 0.46]], 0.02 * s)
        S = [stroke(a, np.vstack([a[-1], _j(r, low, 0.02 * s)]))]
    elif style == 1:  # round top, rounded waist
        K = [[0.15, 0.15], [0.38, 0.02], [0.62, 0.02], [0.82, 0.14], [0.78, 0.32], [cx + 0.08, 0.44], [cx, 0.47]] + low
        S = [stroke(_j(r, K, 0.015 * s))]
    else:             # flat top, diagonal to the waist
        a = _j(r, [[0.15, 0.03], [0.5, 0.02], [0.84, 0.02]], 0.02 * s)
        b = np.vstack([a[-1], _j(r, [[cx, 0.43]], 0.02 * s)])
        S = [stroke(a, b, np.vstack([b[-1], _j(r, low, 0.02 * s)]))]
    return S, r.uniform(0.5, 0.95)


def d4(r, s):
    xv, yb = r.uniform(0.58, 0.8), r.uniform(0.5, 0.7)
    xr = r.uniform(0.86, 1.0) if r.random() < 0.75 else xv + r.uniform(-0.02, 0.06)   # bar may stop at the vertical
    lean = r.normal(0, 0.04)
    style = r.choice(3, p=[0.5, 0.25, 0.25])
    yr = yb + r.normal(0, 0.05)
    if style == 0:    # open top: down, corner, across; then the vertical
        xb = r.uniform(0.0, 0.2)
        a = _j(r, [[xb + r.uniform(0.0, 0.25), r.uniform(0.0, 0.15)], [xb, yb]], 0.01 * s)
        S = [stroke(a, [a[-1], [xr, yr]]), stroke([[xv + lean, r.uniform(-0.03, 0.2)], [xv - lean, 1.0]])]
    elif style == 1:  # closed top: diagonal from the top of the vertical
        xb = r.uniform(0.0, 0.12)
        xr = max(xr, xv + 0.08)
        a = [[xv + lean + r.normal(0, 0.03), r.uniform(0.0, 0.06)], [xb, yb]]
        S = [stroke(a, [[xb, yb], [xr, yr]]), stroke([[xv + lean, 0.0], [xv - lean, 1.0]])]
    else:             # open top with a rounded corner
        xb = r.uniform(0.02, 0.2)
        K = _j(r, [[xb + r.uniform(0.0, 0.15), 0.03], [xb, 0.33], [xb + 0.1, yb - 0.04], [0.5, yb], [xr, yr - 0.02]], 0.015 * s)
        S = [stroke(K), stroke([[xv + lean, r.uniform(-0.03, 0.2)], [xv - lean, 1.0]])]
    return S, r.uniform(0.6, 1.1)


def d5(r, s):
    yc = r.uniform(0.3, 0.48)
    xs = r.uniform(0.2, 0.4)
    xc = xs - r.uniform(0.0, 0.12)
    belly = [[0.5, yc - r.uniform(0.02, 0.09)], [0.8, yc + 0.1], [0.88, 0.72], [0.66, 0.96], [0.36, 1.0], [0.1, 0.9]]
    x0, bl = xs + r.uniform(-0.06, 0.14), r.uniform(0.35, 0.65)
    y0, y1 = r.uniform(-0.06, 0.05), r.uniform(-0.1, 0.05)
    bar = stroke([[x0, y0], [x0 + bl / 2, (y0 + y1) / 2 + r.normal(0, 0.02)], [x0 + bl, y1]])
    e = r.random()
    if e < 0.55:    # two strokes, sharp corner between the down-stroke and the belly
        a = np.array([[xs, 0.0], [xc, yc]]) + r.normal(0, 0.012 * s, (2, 2))
        S = [stroke(a, np.vstack([a[-1], _j(r, belly, 0.02 * s)])), bar]
    elif e < 0.8:   # two strokes, down-stroke flowing into the belly
        S = [stroke(_j(r, [[xs, 0.0], [xc, yc * 0.55], [xc + 0.03, yc]] + belly, 0.015 * s)), bar]
    else:           # one stroke: bar from its right end, rounded turn, down, belly
        a = _j(r, [[xs + bl, y1], [xs + 0.16, 0.0], [xs, 0.07], [xc, yc]], 0.012 * s)
        S = [stroke(a, np.vstack([a[-1], _j(r, belly, 0.02 * s)]))]
    return S, r.uniform(0.5, 1.0)


def d6(r, s):
    lh = r.uniform(0.35, 0.65)
    xt = r.uniform(0.25, 0.95)
    bow = r.uniform(0.0, 0.14)
    yt = 1 - lh
    q = lh / 0.5
    K = [[xt, 0.0], [0.2 + (xt - 0.2) * 0.5 - bow, 0.2], [0.2, 0.45], [0.13, 0.72], [0.27, 0.93], [0.5, 1.0], [0.75, 0.92],
         [0.86, 1 - lh * 0.45], [0.72, yt + 0.04], [0.48, yt], [0.26, yt + 0.1 * q]]
    e = r.random()
    if e < 0.5:      # loop closes onto the down-stroke
        K.append([0.17, yt + 0.25 * q])
    elif e < 0.75:   # loop end curls inside
        K += [[0.2, yt + 0.25 * q], [0.3, yt + 0.33 * q]]
    if r.random() < 0.15:  # hooked start
        K = [[xt + r.uniform(0.04, 0.12), r.uniform(0.05, 0.14)]] + K
    return [stroke(_j(r, K, 0.015 * s))], r.uniform(0.45, 0.9)


def d7(r, s):
    xb = r.uniform(0.25, 0.65)
    xm = (0.9 + xb) / 2 + r.normal(0, 0.06)
    bar = _j(r, [[r.uniform(0.0, 0.2), 0.05 + r.uniform(-0.04, 0.1)], [0.5, 0.03 + r.uniform(-0.05, 0.06)], [0.9, 0.02]], 0.02 * s)
    diag = _j(r, [[xm, 0.5], [xb, 1.0]], 0.01 * s)
    if r.random() < 0.8:
        segs = [bar, np.vstack([bar[-1], diag])]
    else:
        segs = [np.vstack([bar, diag])]
    if r.random() < 0.25:  # tick at the start of the bar
        segs = [np.array([bar[0] + [r.normal(0, 0.02), r.uniform(0.1, 0.25)], bar[0]])] + segs
    S = [stroke(*segs)]
    if r.random() < 0.15:  # crossbar
        yc = r.uniform(0.45, 0.62)
        xc = 0.9 + (xb - 0.9) * yc + (xm - (0.9 + xb) / 2) * 4 * yc * (1 - yc)
        S.append(stroke([[xc - r.uniform(0.15, 0.3), yc + r.uniform(0, 0.05)], [xc + r.uniform(0.15, 0.3), yc - r.uniform(0, 0.05)]]))
    return S, r.uniform(0.5, 1.05)


def d8(r, s):
    cy = r.uniform(0.4, 0.56)
    cx = 0.5 + r.normal(0, 0.03)
    ht, hb = r.uniform(0.2, 0.34), r.uniform(0.22, 0.38)
    xt, xb = cx + r.normal(0, 0.03), cx + r.normal(0, 0.03)
    end = -45 + r.uniform(-45, 40)
    T1 = _ell(xt, cy / 2, ht, cy / 2, [-50, -90, -135, -180, -228])
    B = _ell(xb, (1 + cy) / 2, hb, (1 - cy) / 2, [-42, 0, 45, 90, 135, 180, 222])
    T2 = _ell(xt, cy / 2, ht, cy / 2, [48, 0, end])
    K = _j(r, np.vstack([T1, [[cx, cy]], B, [[cx, cy]], T2]), 0.015 * s)
    if r.random() < 0.15:  # the last stroke runs on past the start
        K = np.vstack([K, K[-1] + (K[-1] - K[-2]) * r.uniform(0.3, 0.8)])
    return [stroke(K)], r.uniform(0.4, 0.8)


def d9(r, s):
    lh = r.uniform(0.4, 0.62)
    lx, hw = 0.42, 0.34
    xs = lx + hw * r.uniform(0.85, 1.0)
    gap = r.uniform(0.0, 0.16) if r.random() < 0.4 else 0.0
    ys = lh * r.uniform(0.1, 0.3)
    yj = lh * r.uniform(0.55, 0.8)
    loop = [[xs - gap, ys], [lx + 0.05, 0.0], [lx - hw * 0.8, lh * 0.2], [lx - hw, lh * 0.55], [lx - hw * 0.5, lh * 0.95], [lx + hw * 0.4, lh * 0.9],
            [xs, yj - 0.12 * lh]]
    top = [xs + r.normal(0, 0.015), ys - r.uniform(0.0, 0.1) * (gap == 0)]
    if r.random() < 0.6:   # straight tail
        xb = xs + r.normal(-0.06, 0.1)
        stem = [top, [xs + (xb - xs) * 0.5 + r.normal(0, 0.02), 0.6], [xb, 1.0]]
    else:                  # tail hooked to the left
        hk = r.uniform(0.3, 0.6)
        stem = [top, [xs, 0.55], [xs - 0.08, 0.84], [xs - hk * 0.6, 0.99], [xs - hk, 0.94]]
    return [stroke(_j(r, loop, 0.015 * s)), stroke(_j(r, stem, 0.012 * s))], r.uniform(0.4, 0.85)


MODELS = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]
# slant (shear) mean and spread per class: handwriting leans to the right on average; a bare stroke (1) leans the most
SLANT = [(0.13, 0.17), (0.2, 0.28), (0.12, 0.17), (0.13, 0.17), (0.13, 0.17), (0.13, 0.17), (0.13, 0.17), (0.13, 0.17), (0.13, 0.17), (0.14, 0.17)]


def glyph(c, r):
    """One random glyph of class c: (P, 2) points in unit coordinates, and position along its stroke for each point."""
    S, asp = MODELS[c](r, r.uniform(0.5, 2.0))
    pts, pos = resample(S)
    x, y = pts[:, 0], pts[:, 1]
    # proportions: upper against lower part, left against right part, top against bottom width
    y = y + r.uniform(-0.07, 0.07) * np.sin(np.pi * np.clip(y, 0, 1))
    if c != 1:
        x = x + r.uniform(-0.05, 0.05) * np.sin(np.pi * np.clip(x, 0, 1))
    x = 0.5 + (x - 0.5) * (1 + r.uniform(-0.25, 0.25) * (y - 0.5))
    x = x * asp
    # slow wobble of the hand
    a1, a2 = r.normal(0, 0.012, 2)
    f1, f2 = r.uniform(0.6, 1.6, 2)
    p1, p2 = r.uniform(0, 2 * np.pi, 2)
    x = x + a1 * np.sin(2 * np.pi * f1 * y + p1)
    y = y + a2 * np.sin(2 * np.pi * f2 * x / max(asp, 0.3) + p2)
    k = np.clip(r.normal(*SLANT[c]), -0.45, 0.9)
    x = x - k * (y - 0.5)
    rot = np.deg2rad(r.normal(0, 5))
    cr, sr = np.cos(rot), np.sin(rot)
    return np.stack([cr * x - sr * y, sr * x + cr * y], 1), pos


# ----------------------------------------------------------------------------- rendering
_AX = np.arange(28, dtype=np.float32)


def _raster(pts, rad, edge):
    """pts (B, n, 2) pixel coordinates (x, y); rad (B, n) half-width; edge (B,) width of the grey ramp."""
    dx2 = (pts[:, :, 0, None] - _AX) ** 2
    dy2 = (pts[:, :, 1, None] - _AX) ** 2
    inv = (1.0 / rad ** 2).astype(np.float32)
    q = (dy2[:, :, :, None] + dx2[:, :, None, :]) * inv[:, :, None, None]
    dn = np.sqrt(q.min(1))
    r0 = rad.mean(1)
    return np.clip(0.5 + (1 - dn) * (r0 / edge)[:, None, None], 0, 1).astype(np.float32)


def render(c, n, r):
    """n random images (n, 28, 28) of class c."""
    pts = np.empty((n, P, 2), np.float32)
    rad = np.empty((n, P), np.float32)
    for i in range(n):
        g, pos = glyph(c, r)
        w = float(np.clip(np.exp(r.normal(np.log(2.5), 0.33)), 1.3, 5.6))      # stroke width in pixels
        mod = 1 + r.uniform(0, 0.15) * np.sin(2 * np.pi * (r.uniform(0.5, 2.0) * np.arange(P) / P + r.random()))
        if r.random() < 0.3:
            mod = mod * (1 - 0.3 * np.clip((pos - 0.8) / 0.2, 0, 1))             # pen lifting at the end of a stroke
        rr = 0.5 * w * mod
        lo, hi = g.min(0), g.max(0)
        s = (20.0 - w) / max(hi[0] - lo[0], hi[1] - lo[1], 1e-3)
        pts[i] = (g - (lo + hi) / 2) * s
        rad[i] = rr
    edge = r.uniform(0.9, 1.6, n).astype(np.float32)
    ctr = (13.5 + r.uniform(-1.0, 1.0, (n, 2))).astype(np.float32)   # frame centre, give or take a pixel of rounding
    # first pass on a quarter of the points to find the centre of mass, second pass centred on it
    rough = _raster(pts[:, ::4] + 13.5, rad[:, ::4], edge)
    m = rough.sum((1, 2))
    com = np.stack([(rough.sum(1) * _AX).sum(1) / m, (rough.sum(2) * _AX).sum(1) / m], 1)
    return _raster(pts + (13.5 - com + ctr)[:, None, :], rad, edge)


# ----------------------------------------------------------------------------- the network
NDIR = 8            # gradient directions, 45 degrees apart
SIG_G = 0.8         # Gaussian-derivative scale of the gradient filters (pixels)
KG = 5              # their kernel size
SIG_B = 1.6         # blur before subsampling (pixels)
KB, SB, PB = 12, 3, 4   # blur kernel, stride, padding: 28 -> 9 cells, centres at 1.5 + 3 i
NC = 9              # cells per side
RS = 6              # region side in cells
R0 = (0, 3)         # region offsets -> 2 x 2 overlapping regions
SQ_H, SQ_K = 0.05, 12   # square-root knots at (SQ_H k)^2, k = 0 .. SQ_K - 1
NCLS, GROUPS, PER_GROUP = 10, 4, 2590
T = NCLS * GROUPS * PER_GROUP
D = NDIR * NC * NC
SEED = 20240


def grad_filters():
    a = np.arange(KG) - KG // 2
    g = np.exp(-(a[:, None] ** 2 + a[None, :] ** 2) / (2 * SIG_G ** 2))
    kx = a[None, :] * g
    kx = kx / (kx * a[None, :]).sum()          # a ramp of slope 1 along x gives 1
    ky = kx.T
    W = np.zeros((NDIR, 1, KG, KG))
    for d in range(NDIR):
        th = 2 * np.pi * d / NDIR
        W[d, 0] = np.cos(th) * kx + np.sin(th) * ky
    return W


def dir_mix():
    """relu(r_d - r_(d+2) - r_(d-2)) = |g| relu(cos - |sin|): splits the gradient between its two nearest directions."""
    M = np.zeros((NDIR, NDIR, 1, 1))
    for d in range(NDIR):
        M[d, d] = 1
        M[d, (d + 2) % NDIR] = -1
        M[d, (d - 2) % NDIR] = -1
    return M


def blur_taps():
    a = np.arange(KB) - (KB - 1) / 2
    g = np.exp(-a ** 2 / (2 * SIG_B ** 2))
    return g / g.sum()


def sqrt_units():
    """Piecewise-linear square root: sum_k c_k relu(x - t_k) equals SQ_H k at x = t_k = (SQ_H k)^2."""
    k = np.arange(SQ_K)
    t = (SQ_H * k) ** 2
    slope = 1.0 / (SQ_H * (2 * k + 1))
    c = np.concatenate([[slope[0]], np.diff(slope)])
    return t, c


def features(img):
    """The feature part of the network, in numpy. img (N, 28, 28) -> (N, 8, 9, 9), normalised per image."""
    n = len(img)
    Wg = grad_filters().reshape(NDIR, KG * KG).astype(np.float32)
    p = KG // 2
    xp = np.pad(img.astype(np.float32), ((0, 0), (p, p), (p, p)))
    cols = np.stack([xp[:, dy:dy + 28, dx:dx + 28] for dy in range(KG) for dx in range(KG)], -1)
    r = np.maximum(cols @ Wg.T, 0)                                     # (N, 28, 28, 8)
    q = np.maximum(r @ dir_mix()[:, :, 0, 0].T.astype(np.float32), 0)
    taps = blur_taps()
    B = np.zeros((NC, 28 + 2 * PB))
    for i in range(NC):
        B[i, i * SB:i * SB + KB] = taps
    B = B[:, PB:PB + 28].astype(np.float32)                            # (9, 28): zero padding drops out
    f = np.einsum('ay,nyxc,bx->ncab', B, q, B, optimize=True)          # (N, 8, 9, 9)
    t, c = sqrt_units()
    s = np.zeros_like(f)
    for tk, ck in zip(t, c):
        s += np.float32(ck) * np.maximum(f - np.float32(tk), 0)
    s = s.reshape(n, -1)
    mu = s.mean(1, keepdims=True)
    s = (s - mu) / np.sqrt(((s - mu) ** 2).mean(1, keepdims=True) + 1e-5)
    return s.reshape(n, NDIR, NC, NC)


def regions():
    return [(y, x) for y in R0 for x in R0]


def layers():
    L = [dict(name="g", op="conv", inputs=["image"], out=NDIR, kernel=[KG, KG], stride=1, padding=KG // 2),
         dict(name="g_r", op="relu", inputs=["g"]),
         dict(name="dir", op="conv", inputs=["g_r"], out=NDIR, kernel=[1, 1], stride=1, padding=0),
         dict(name="dir_r", op="relu", inputs=["dir"]),
         dict(name="blur", op="conv", inputs=["dir_r"], out=NDIR, kernel=[KB, KB], stride=SB, padding=PB),
         dict(name="sq1", op="conv", inputs=["blur"], out=NDIR * SQ_K, kernel=[1, 1], stride=1, padding=0),
         dict(name="sq1_r", op="relu", inputs=["sq1"]),
         dict(name="sq2", op="conv", inputs=["sq1_r"], out=NDIR, kernel=[1, 1], stride=1, padding=0),
         dict(name="flat", op="reshape", inputs=["sq2"], shape=[1, NDIR, NC * NC]),
         dict(name="nrm", op="norm", inputs=["flat"]),
         dict(name="fmap", op="reshape", inputs=["nrm"], shape=[NDIR, NC, NC])]
    outs = []
    for i, _ in enumerate(regions()):
        L += [dict(name=f"crop{i}", op="conv", inputs=["fmap"], out=NDIR, kernel=[NC - RS + 1, NC - RS + 1], stride=1, padding=0),
              dict(name=f"tmpl{i}", op="dense", inputs=[f"crop{i}"], out=T),
              dict(name=f"grp{i}", op="reshape", inputs=[f"tmpl{i}"], shape=[1, NCLS * GROUPS, PER_GROUP]),
              dict(name=f"best{i}", op="maxpool", inputs=[f"grp{i}"], size=[1, PER_GROUP], stride=[1, PER_GROUP]),
              dict(name=f"cls{i}", op="reshape", inputs=[f"best{i}"], shape=[1, NCLS, GROUPS]),
              dict(name=f"avg{i}", op="avgpool", inputs=[f"cls{i}"], size=[1, GROUPS], stride=[1, GROUPS])]
        outs.append(f"avg{i}")
    L += [dict(name="sum", op="add", inputs=outs), dict(name="scores", op="reshape", inputs=["sum"], shape=[NCLS])]
    return L


def fixed_weights():
    w = {}
    w["g.W"], w["g.b"] = grad_filters(), np.zeros(NDIR)
    w["dir.W"], w["dir.b"] = dir_mix(), np.zeros(NDIR)
    taps = blur_taps()
    Wb = np.zeros((NDIR, NDIR, KB, KB))
    for d in range(NDIR):
        Wb[d, d] = np.outer(taps, taps)
    w["blur.W"], w["blur.b"] = Wb, np.zeros(NDIR)
    t, c = sqrt_units()
    W1, b1, W2 = np.zeros((NDIR * SQ_K, NDIR, 1, 1)), np.zeros(NDIR * SQ_K), np.zeros((NDIR, NDIR * SQ_K, 1, 1))
    for d in range(NDIR):
        for k in range(SQ_K):
            W1[d * SQ_K + k, d] = 1
            b1[d * SQ_K + k] = -t[k]
            W2[d, d * SQ_K + k] = c[k]
    w["sq1.W"], w["sq1.b"], w["sq2.W"], w["sq2.b"] = W1, b1, W2, np.zeros(NDIR)
    kc = NC - RS + 1
    for i, (y, x) in enumerate(regions()):
        Wc = np.zeros((NDIR, NDIR, kc, kc))
        for d in range(NDIR):
            Wc[d, d, y, x] = 1
        w[f"crop{i}.W"], w[f"crop{i}.b"] = Wc, np.zeros(NDIR)
    return w


def templates(per_class, batch=250, log=None):
    """per_class drawn digits of every class -> their feature maps (10 * per_class, 8, 9, 9), class by class."""
    F = np.empty((NCLS * per_class, NDIR, NC, NC), np.float32)
    for c in range(NCLS):
        r = np.random.default_rng([SEED, c])
        for i in range(0, per_class, batch):
            m = min(batch, per_class - i)
            F[c * per_class + i:c * per_class + i + m] = features(render(c, m, r))
        if log:
            log(f"class {c} drawn")
    return F


def main(out_dir):
    w = fixed_weights()
    F = templates(GROUPS * PER_GROUP, log=lambda s: print(s, flush=True))
    for i, (y, x) in enumerate(regions()):
        Tr = F[:, :, y:y + RS, x:x + RS].reshape(T, -1)
        w[f"tmpl{i}.W"] = Tr / np.float32(D)
        w[f"tmpl{i}.b"] = -0.5 * (Tr.astype(np.float64) ** 2).sum(1) / D     # score = -(squared distance)/2D + const
    with open(f"{out_dir}/network.json", "w") as f:
        json.dump({"layers": layers()}, f)
    np.savez(f"{out_dir}/weights.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in w.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
