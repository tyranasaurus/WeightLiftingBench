"""build.py -- writes the designed network's weights from scratch.

The network classifies 28x28 handwritten   All weights come from this
source code alone: a bank of oriented bar filters built from Gaussian formulas,
plus a dense layer whose rows are feature maps of hand-drawn template digits
(pen-path stroke models typed from general knowledge of how each digit is
written, rendered with a deterministic rasterizer).  No training data is read.

Usage: python3 build.py OUTPUT_DIR   (writes network.json and weights.npz)
"""
import json
import sys
import numpy as np

# ---------------------------------------------------------------- spline


def catmull_rom(points, closed=False, seg=10):
    """Dense polyline through points (Catmull-Rom, centripetal-ish uniform)."""
    P = np.asarray(points, float)
    n = len(P)
    if n < 3:
        t = np.linspace(0, 1, seg + 1)[:, None]
        return (1 - t) * P[0] + t * P[-1]
    if closed:
        idx = [(i, (i + 1) % n, (i - 1) % n, (i + 2) % n) for i in range(n)]
    else:
        idx = [(0, 1, 0, 2)] + [(i, i + 1, i - 1, i + 2) for i in range(1, n - 2)] + [(n - 2, n - 1, n - 3, n - 1)]
    out = []
    for i0, i1, im1, ip1 in idx:
        p0, p1, p2, p3 = P[im1], P[i0], P[i1], P[ip1]
        t0, t1 = (p1 - p0) / 2.0, (p3 - p2) / 2.0
        for t in np.linspace(0, 1, seg, endpoint=False)[:, None]:
            t2, t3 = t * t, t * t * t
            out.append((2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * t0 + (-2 * t3 + 3 * t2) * p2 + (t3 - t2) * t1)
    if not closed:
        out.append(P[-1])
    return np.asarray(out)


def ellipse_pts(cx, cy, rx, ry, a0, a1, n=40):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def resample(pts, step=0.35):
    """Uniform arc-length resampling of a polyline."""
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(1))
    if d.sum() < 1e-9:
        return pts[:1]
    t = np.concatenate([[0.0], np.cumsum(d)])
    u = np.arange(0.0, t[-1], step)
    x = np.interp(u, t, pts[:, 0])
    y = np.interp(u, t, pts[:, 1])
    return np.stack([x, y], 1)


# ---------------------------------------------------------------- base shapes
# Each family: list of strokes; a stroke is (points, closed) where points is
# an (n,2) array or a callable params->points.  We use plain lists of tuples.

def _f0(p):
    rx, ry = p.get("rx", 7.5), p.get("ry", 8.7)
    rot = p.get("rot0", 0.0)
    a = np.linspace(0, 2 * np.pi, 28)
    pts = np.stack([rx * np.cos(a), ry * np.sin(a)], 1)
    if rot:
        c, s = np.cos(rot), np.sin(rot)
        pts = pts @ np.array([[c, s], [-s, c]]).T
    return [(pts, True)]


def _f1(p):
    v = [(0.4, -8.7), (0.5, -4.0), (0.6, 0.0), (0.5, 4.5), (0.3, 8.6)]
    out = [(np.asarray(v), False)]
    if p.get("flag", 1):
        out.append((np.asarray([(-4.4, -6.0), (-1.8, -7.9), (0.4, -8.7)]), False))
    if p.get("base", 0):
        out.append((np.asarray([(-3.8, 8.5), (0.3, 8.6), (4.0, 8.2)]), False))
    return out


def _f2(p):
    topw = p.get("topw", 1.0)     # top loop width scale
    dy = p.get("diag_end", 6.3)   # how low the diagonal reaches before bottom
    dx = p.get("diag_x", -6.6)
    loop = p.get("loop", 0)       # small loop at bottom-left
    up = p.get("upcurl", 0)
    head = [(-6.0 * topw, -5.6), (-2.8 * topw, -8.3), (2.0 * topw, -8.5),
            (5.8 * topw, -6.4), (6.4 * topw, -3.4), (3.6, -1.2), (0.6, 0.9),
            (-2.8, 3.0)]
    if loop:
        S1 = head + [(-3.0, 3.6), (-2.6, 4.5)]
        S2 = ellipse_pts(-4.3, 5.7, 2.2, 2.2, -0.25 * np.pi, -1.75 * np.pi, 26)
        S3 = [(-1.8, 6.1), (1.8, 7.3), (5.4, 7.3), (7.7, 6.4)]
        return [(np.asarray(S1), False), (S2, False), (np.asarray(S3), False)]
    pts = head + [(dx, dy), (-3.8, 7.9), (1.0, 7.9), (5.2, 7.6), (7.9, 7.2 - 0.5 * up)]
    if up:
        pts += [(8.5, 5.9)]
    return [(np.asarray(pts), False)]


def _f3(p):
    waist = p.get("waist", -1.1)
    flat = p.get("flat", 0)
    pts = [(-6.0, -7.6), (-1.6, -8.6), (3.2, -8.1), (5.8, -6.4),
           (7.0, -3.2), (3.0, waist - 0.6), (-1.4, waist),
           (2.6, waist + 0.6), (6.4, waist + 2.3), (7.1, waist + 4.8),
           (5.2, waist + 7.0), (1.6, waist + 8.5), (-3.2, waist + 8.0),
           (-5.8, waist + 6.6)]
    if flat:  # flat/straight top stroke style
        pts[0] = (-5.0, -7.2)
        pts[1] = (-1.6, -8.0)
        pts[2] = (2.2, -8.0)
    return [(np.asarray(pts), False)]


def _f4(p):
    dx0 = p.get("dx0", -3.0)   # left stroke: top x
    dx1 = p.get("dx1", -6.4)   # left stroke: bottom x (crossbar level)
    cb = p.get("cb", 0.6)      # crossbar height
    vx = p.get("vx", 2.2)      # vertical x at top
    vxb = p.get("vxb", 0.8)    # vertical x at bottom
    open4 = p.get("open", 0)   # crossbar extends right past the vertical
    A = [(dx0, -8.6), (dx0 - 0.8, -4.4), (dx1 + 0.5, -2.4), (dx1, cb)]
    right = 6.2 if open4 else vx + 0.4
    B = [(dx1 - 0.5, cb - 0.1), (-2.4, cb + 0.15), (right, cb)]
    vxm = (vx + vxb) / 2.0
    C = [(vx, -8.6), (vx - 0.1, -4.2), (vxm - 0.1, cb), (vxb - 0.2, 8.6)]
    return [(np.asarray(A), False), (np.asarray(B), False), (np.asarray(C), False)]


def _f5(p):
    tl = p.get("tlen", 7.0)   # top bar right end
    bl = p.get("bulge", 1.0)  # bowl size
    S1 = [(-6.6, -7.4), (-1.8, -8.0), (tl - 2.2, -7.8), (tl, -7.2)]
    S2 = [(-5.9, -7.2), (-5.6, -4.0), (-5.3 * bl, -1.0), (-4.1, -0.2),
          (-1.2, 0.3), (2.8 * bl, 1.2), (5.7 * bl, 3.1), (5.9 * bl, 5.3),
          (4.0 * bl, 7.3), (0.8, 8.2), (-2.8, 8.1), (-5.2 * bl, 6.6), (-5.9 * bl, 4.6)]
    return [(np.asarray(S1), False), (np.asarray(S2), False)]


def _f6(p):
    rx, ry = p.get("rx", 4.6), p.get("ry", 4.2)
    lcy = p.get("lcy", 4.3)     # loop centre y
    drop = p.get("drop", 4.3)   # where the tail meets the loop
    S1 = [(5.0, -8.4), (1.8, -8.0), (-1.6, -6.8), (-4.4, -3.9),
          (-5.6, -0.8), (-5.3, 1.8), (-4.8, drop)]
    loop = ellipse_pts(0.2, lcy, rx, ry, np.pi * 0.95, np.pi * 0.95 - 2 * np.pi, 48)
    return [(np.asarray(S1), False), (loop, True)]


def _f7(p):
    cross = p.get("cross", 0)
    curve = p.get("curve", 0)
    ex = p.get("endx", -0.4)
    S1 = [(-6.8, -7.0), (-2.8, -8.3), (1.8, -8.4), (5.4, -6.9), (6.7, -4.9)] if curve \
        else [(-6.8, -7.6), (-1.8, -7.9), (4.0, -7.8), (7.0, -7.3)]
    S2 = [(6.6, -5.0), (4.8, -3.4), (2.6, 0.4), (0.8, 4.4), (ex, 8.3)]
    out = [(np.asarray(S1), False), (np.asarray(S2), False)]
    if cross:
        out.append((np.asarray([(-4.6, -0.2), (0.4, -1.7), (4.0, -3.1)]), False))
    return out


def _f8(p):
    trx, tr_y = p.get("trx", 4.9), p.get("tr_y", 3.9)
    brx, bry = p.get("brx", 5.9), p.get("bry", 4.7)
    wy = p.get("wy", -0.5)  # waist y
    top = ellipse_pts(0.0, wy - tr_y, trx, tr_y, 0, 2 * np.pi, 44)
    bot = ellipse_pts(0.0, wy + bry, brx, bry, 0, 2 * np.pi, 44)
    return [(top, True), (bot, True)]


def _f9(p):
    rx, ry = p.get("rx", 4.4), p.get("ry", 4.0)
    lcy = p.get("lcy", -4.5)
    tail = p.get("tail", "curve")
    loop = ellipse_pts(0.5, lcy, rx, ry, 0, 2 * np.pi, 44)
    if tail == "curve":
        T = [(4.9, lcy - 0.4), (4.5, -0.6), (3.8, 3.2), (2.9, 6.3), (2.4, 8.3)]
    elif tail == "straight":
        T = [(4.8, lcy - 0.3), (4.3, 0.0), (3.8, 4.0), (3.4, 8.2)]
    else:  # hooked: swings left at the very bottom
        T = [(4.9, lcy - 0.4), (4.5, 0.0), (3.4, 3.8), (1.9, 6.8), (1.6, 7.8), (-0.3, 8.3)]
    return [(loop, True), (np.asarray(T), False)]


FAMILIES = {
    0: [dict(), dict(rx=6.8, ry=8.2), dict(rx=8.0, ry=9.0)],
    1: [dict(flag=1), dict(flag=0), dict(flag=1, base=1)],
    2: [dict(), dict(diag_x=-4.4, diag_end=5.4, upcurl=1), dict(loop=1), dict(topw=0.85, diag_end=6.8)],
    3: [dict(), dict(waist=-0.2), dict(flat=1, waist=-1.4)],
    4: [dict(), dict(open=1), dict(dx0=-4.4, dx1=-5.3, cb=1.4),
        dict(open=1, vx=1.0, vxb=0.2, dx0=-2.4, dx1=-5.8, cb=-0.6),
        dict(cb=2.6, dx0=-3.4, dx1=-7.2)],
    5: [dict(), dict(tlen=5.6, bulge=0.88), dict(bulge=1.12), dict(tlen=7.8)],
    6: [dict(), dict(rx=4.2, ry=3.9, lcy=3.8), dict(rx=5.1, ry=4.6, lcy=4.6, drop=4.9)],
    7: [dict(), dict(cross=1), dict(curve=1), dict(endx=1.4)],
    8: [dict(), dict(trx=4.2, tr_y=3.4, brx=6.3, bry=5.0, wy=-1.2), dict(trx=5.2, tr_y=4.2, brx=5.5, bry=4.5),
        dict(trx=3.8, tr_y=3.2, brx=6.0, bry=4.9, wy=0.2)],
    9: [dict(), dict(tail="straight"), dict(tail="hooked"), dict(rx=4.0, ry=3.8, lcy=-4.7)],
}

MAKERS = {0: _f0, 1: _f1, 2: _f2, 3: _f3, 4: _f4, 5: _f5, 6: _f6, 7: _f7, 8: _f8, 9: _f9}


# ---------------------------------------------------------------- renderer


_GK = {}


def _gauss_kernel1d(sigma):
    if sigma not in _GK:
        r = int(np.ceil(3.5 * sigma))
        x = np.arange(-r, r + 1)
        g = np.exp(-x ** 2 / (2 * sigma ** 2))
        _GK[sigma] = (g / g.sum(), r)
    return _GK[sigma]


def _blur_sep(m, sigma):
    g, r = _gauss_kernel1d(sigma)
    C, H, W = m.shape
    out = np.zeros((C, H, W), m.dtype)
    p = np.pad(m, ((0, 0), (0, 0), (r, r)))
    for i, gi in enumerate(g):
        out += gi * p[:, :, i:i + W]
    p = np.pad(out, ((0, 0), (r, r), (0, 0)))
    out2 = np.zeros((C, H, W), m.dtype)
    for i, gi in enumerate(g):
        out2 += gi * p[:, i:i + H, :]
    return out2


def _stamp_fast(out, scale, dx, dy, big, cen, brush, sample=0.15, amp=2.6):
    """Rasterize pen mass then convolve with a Gaussian brush.

    `brush` is in base (glyph) units; the canvas sigma is brush*scale, and the
    amplitude factor below makes the cross-section amp*exp(-v^2/2 brush^2) in
    base units regardless of scale (scale-invariant stroke profile)."""
    M = np.zeros((big, big))
    segs = []
    for dense in out:
        pts = (dense + np.array([dx, dy])) * scale + cen
        segs.append(pts)
    allpts = np.concatenate(segs)
    np.add.at(M, (np.clip(allpts[:, 1].astype(int), 0, big - 1),
                  np.clip(allpts[:, 0].astype(int), 0, big - 1)), 1.0)
    sig = brush * scale
    img = _blur_sep(M[None], sig)[0] * sample * scale * np.sqrt(2 * np.pi) * sig * amp
    return img


def render(digit, family=0, brush=0.85, shear=0.0, rot=0.0, sx=1.0, sy=1.0,
           jitter=None, size=28, fit=19.2, fit_jit=None, sample=0.15, amp=2.6, pover=None):
    """Render digit with given style; returns float32 (size,size) in [0,1].

    Mimics the dataset's normalisation: strokes are drawn, scaled uniformly
    so the ink bounding box fits inside a `fit` x `fit` box, and the ink
    centre of mass is placed at the frame centre (13.5, 13.5).
    """
    params = dict(FAMILIES[digit][family if family >= 0 else 0])
    if pover:
        params.update(pover)
    strokes = MAKERS[digit](params)
    c, s = np.cos(rot), np.sin(rot)
    out = []
    for pts, closed in strokes:
        pts = pts.astype(float).copy()
        if jitter is not None:
            rng, amp, freq = jitter
            n = len(pts)
            base = rng.randn(max(n, 2), 2)
            k = np.exp(-0.5 * (np.arange(max(n, 2))[:, None] - np.arange(max(n, 2))[None, :]) ** 2 / freq**2)
            off = (k @ base) / np.sqrt(k.sum(1, keepdims=True)).clip(1e-9)
            pts = pts + amp * off[:n]
        pts = np.stack([pts[:, 0] + shear * pts[:, 1], pts[:, 1]], 1)
        pts = pts @ np.array([[c, s], [-s, c]]).T
        pts = pts * np.array([sx, sy])
        dense = catmull_rom(pts, closed=closed, seg=12) if len(pts) > 2 else pts
        out.append(resample(dense, sample))

    big = 80 if size <= 40 else size * 3
    cen = big / 2.0

    def stamp(scale, dx, dy):
        return np.clip(_stamp_fast(out, scale, dx, dy, big, cen, brush, sample, amp), 0, 1)

    # pass 1: measure the ink bounding box (threshold ~ visible ink)
    img1 = stamp(1.0, 0, 0)
    m = img1 > 0.2
    if not m.any():
        m = img1 > 0
    ys, xs = np.where(m)
    hh, ww = (ys.max() - ys.min() + 1), (xs.max() - xs.min() + 1)
    target = fit if fit_jit is None else fit * fit_jit
    scale = target / max(hh, ww)
    ymid, xmid = (ys.max() + ys.min()) / 2.0, (xs.max() + xs.min()) / 2.0
    dx, dy = (cen - xmid) / scale, (cen - ymid) / scale
    # pass 2: stamp at final scale, find grey centre of mass, shift it to centre
    img2 = stamp(scale, dx, dy)
    tot = img2.sum()
    cy = (img2.sum(1) @ np.arange(big)) / tot
    cx = (img2.sum(0) @ np.arange(big)) / tot
    dx += (cen - cx) / scale
    dy += (cen - cy) / scale
    img3 = stamp(scale, dx, dy)
    o = int(big / 2 - size / 2)
    return img3[o:o + size, o:o + size].astype(np.float32)


def random_style(rng, d=None, families=None):
    if d is None:
        d = int(rng.randint(10))
    fams = families if families is not None else list(range(len(FAMILIES[d])))
    fam = int(rng.choice(fams))
    shear = float(np.clip(rng.normal(0, 0.10), -0.28, 0.28))
    rot = float(rng.normal(0, 0.035))
    brush = float(rng.uniform(0.60, 0.95))
    fit_jit = float(rng.uniform(0.96, 1.07))
    sx, sy = float(rng.uniform(0.97, 1.03)), float(rng.uniform(0.97, 1.03))
    jit = (rng, 0.45, 2.8) if rng.rand() < 0.8 else None
    return dict(digit=d, family=fam, shear=shear, rot=rot, brush=brush,
                fit_jit=fit_jit, sx=sx, sy=sy, jitter=jit)



def template_grid():
    """Deterministic template bank: (digit, render-kwargs) list."""
    shears = [-0.22, -0.11, 0.0, 0.11, 0.22]
    brushes = [0.65, 0.8, 0.95]
    out = []
    for d in range(10):
        fams = list(range(len(FAMILIES[d])))
        while len(fams) < 4:  # pad with param-perturbed copies
            fams.append(-1 - (len(fams) - len(FAMILIES[d])))
        for fi, fam in enumerate(fams[:4]):
            for sh in shears:
                for br in brushes[:2 if fi % 2 else 3]:
                    rot = 0.0
                    if d in (0, 8) and fi % 2:
                        rot = 0.07 if sh > 0 else (-0.07 if sh < 0 else 0.0)
                    kw = dict(digit=d, family=fam if fam >= 0 else 0, shear=sh,
                              brush=br, rot=rot,
                              fit_jit=1.0 if (fi + len(shears)) % 2 else 0.95)
                    out.append((d, kw))
    return out



def template_variants(rand_per_class=60, seed=123):
    variants = list(template_grid())
    rng = np.random.RandomState(seed)
    for d in range(10):
        for _ in range(rand_per_class):
            kw = random_style(rng, d)
            kw['jitter'] = (rng, 0.4, 2.8)
            variants.append((d, kw))
    return variants


# ---------------------------------------------------------------- ops
# These mirror /app/blocks.py exactly (same padding, stride, window rules).


def conv(x, W, b, stride=1, padding=0):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
                     for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def avgpool(x, size, stride):
    n, c, h, w = x.shape
    sh = sw = size
    th = tw = stride
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw]
                    for dy in range(sh) for dx in range(sw)], axis=0)
    return win.mean(0)


def relu(x):
    return np.maximum(x, 0)


def dense(x, W, b):
    return x.reshape(len(x), -1) @ W.T + b


# ---------------------------------------------------------------- filter bank


def ridge_filter(k, theta, sigma, ksize=7, taper=True, dog=True):
    """Ridge (bar) filter. dog=True: Gaussian minus wider Gaussian (zero-sum,
    width-selective). dog=False: plain Gaussian ridge (width-robust)."""
    r = ksize // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    u = xx * np.cos(theta) + yy * np.sin(theta)   # along-stroke
    v = -xx * np.sin(theta) + yy * np.cos(theta)  # across-stroke
    if dog:
        s2 = 2.2 * sigma
        W = np.exp(-v ** 2 / (2 * sigma ** 2)) - (sigma / s2) * np.exp(-v ** 2 / (2 * s2 ** 2))
    else:
        W = np.exp(-v ** 2 / (2 * sigma ** 2))
    if taper:
        W = W * np.cos(np.clip(u / (r + 0.5), -1, 1) * np.pi / 2) ** 2
    W = W / np.sqrt((W ** 2).sum())
    return W.astype(np.float32)


def make_conv1(norient=8):
    """Filter bank: per orientation, ridges at three LENGTHS (short 5x5,
    medium 7x7, long 11x11) at width sigma 0.85, plus one thick 7x7.
    Long filters respond only to long straight strokes -> straightness cue."""
    fils = []
    for o in range(norient):
        th = o * np.pi / norient
        for k, sig in ((5, 0.85), (7, 0.85), (11, 0.85), (7, 1.4)):
            f = ridge_filter(k, th, sig, k)
            pad = (11 - k) // 2
            if pad:
                f = np.pad(f, pad)
            fils.append(f)
    W = np.stack(fils)[:, None, :, :]     # (O, 1, 11, 11), short kernels zero-padded
    b = np.zeros(len(fils), np.float32)
    return W, b


# ---------------------------------------------------------------- templates


def gauss_blur_sep(maps, sigma):
    """Separable Gaussian blur over the last two axes of (C, H, W)."""
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    C, H, W = maps.shape
    p = np.pad(maps, ((0, 0), (0, 0), (r, r)))
    out = np.zeros_like(maps)
    for i, gi in enumerate(g):
        out += gi * p[:, :, i:i + W]
    p = np.pad(out, ((0, 0), (r, r), (0, 0)))
    out2 = np.zeros_like(maps)
    for i, gi in enumerate(g):
        out2 += gi * p[:, i:i + H, :]
    return out2


class Net:
    def __init__(self, conv1W, conv1b, blur_sigma=0.9, use_norm=False):
        self.W1, self.b1 = conv1W, conv1b
        self.blur = blur_sigma
        self.use_norm = use_norm

    @staticmethod
    def _norm(x):
        # mirrors blocks.py: per-channel standardisation over positions
        mu = x.mean(axis=(2, 3), keepdims=True)
        return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)

    def features(self, images):
        """images: (N,1,28,28) -> pooled (and optionally normalised)
        relu feature maps (N,C,14,14)."""
        pad = (self.W1.shape[-1] - 1) // 2
        x = conv(images, self.W1, self.b1, 1, pad)
        x = relu(x)
        x = avgpool(x, 2, 2)
        if self.use_norm:
            x = self._norm(x)
        return x

    def template_rows(self, images):
        """For template images: blurred features (for W) and raw norm (for b)."""
        f = self.features(images)                     # (N,C,14,14)
        n = len(f)
        fb = np.stack([gauss_blur_sep(f[i], self.blur) for i in range(n)])
        norms = (f.reshape(n, -1) ** 2).sum(1)
        return fb.reshape(n, -1), norms


def build_template_bank(net, variants, extra_parts=()):
    """variants: list of (digit, kwargs) for render.
    Returns dense1 W, b (template rows then extra part rows), and metadata."""
    imgs = np.stack([render(**dict(kw, digit=d))[None] for d, kw in variants])
    fb, norms = net.template_rows(imgs)
    W = 2.0 * fb
    b = -norms
    return W.astype(np.float32), b.astype(np.float32), imgs


# ---------------------------------------------------------------- profiles

PROFILES = [("aa0.5", 0.5, 0.0), ("aab.5", 0.5, 0.45), ("aab.3", 0.3, 0.45)]
GCHAN = [0.5, 1.0, 3.0, 0.5]
BLUR = 0.9
RPC = 60
SEED = 123


def realify(kw, thresh, blur=0.0, scale=4):
    """Render at 4x, binarize, area-average down (MNIST-like coverage
    antialiasing), optional small blur for softer edges."""
    kw4 = dict(kw)
    kw4['fit'] = kw.get('fit', 19.2) * scale
    kw4['fit_jit'] = kw.get('fit_jit', 1.0)
    kw4['size'] = 28 * scale
    kw4['sample'] = 0.15 * scale
    img = render(**kw4)
    b = (img > thresh).astype(np.float32)
    small = b.reshape(28, scale, 28, scale).mean((1, 3))
    if blur:
        small = gauss_blur_sep(small[None], blur)[0]
    return np.clip(small, 0, 1)


def features_chunked(net, imgs, chunk=200):
    return np.concatenate([net.features(imgs[i:i + chunk].astype(np.float32))
                           for i in range(0, len(imgs), chunk)])


def build(out_dir):
    W1, b1 = make_conv1()
    net = Net(W1, b1, blur_sigma=BLUR, use_norm=True)
    variants = template_variants(RPC, SEED)

    Wd_rows, bd_rows, cls_all = [], [], []
    for pname, th, bl in PROFILES:
        imgs = np.stack([realify(dict(kw, digit=dd), th, bl)[None] for dd, kw in variants])
        f = features_chunked(net, imgs)
        n, C, H, W = f.shape
        fb = np.stack([gauss_blur_sep(f[i], BLUR) for i in range(n)])
        gg = np.tile(np.array(GCHAN * 8, np.float32)[:, None, None], (1, H, W)).reshape(-1)
        Wd_rows.append((2.0 * (fb.reshape(n, -1) * gg)).astype(np.float32))
        bd_rows.append((-(f.reshape(n, -1) ** 2 * gg).sum(1)).astype(np.float32))
        cls_all.append(np.array([dd for dd, kw in variants]))

    Wd = np.concatenate(Wd_rows)
    bd = np.concatenate(bd_rows)
    cls = np.concatenate(cls_all)
    n = len(cls)
    W2 = np.zeros((10, n), np.float32)
    for k in range(10):
        W2[k, cls == k] = 1.0 / (cls == k).sum()
    b2 = np.zeros(10, np.float32)

    layers = [
        {"name": "c1", "op": "conv", "inputs": ["image"], "out": 32, "kernel": [11, 11], "stride": 1, "padding": 5},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "p1", "op": "avgpool", "inputs": ["r1"], "size": [2, 2], "stride": [2, 2]},
        {"name": "n1", "op": "norm", "inputs": ["p1"]},
        {"name": "d1", "op": "dense", "inputs": ["n1"], "out": n},
        {"name": "d2", "op": "dense", "inputs": ["d1"], "out": 10},
    ]
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as fh:
        json.dump({"layers": layers}, fh)
    np.savez(os.path.join(out_dir, "weights.npz"), **{
        "c1.W": W1, "c1.b": b1, "d1.W": Wd, "d1.b": bd, "d2.W": W2, "d2.b": b2})
    total = W1.size + b1.size + Wd.size + bd.size + W2.size + b2.size
    print(f"built: {n} templates, total weights {total:,}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
