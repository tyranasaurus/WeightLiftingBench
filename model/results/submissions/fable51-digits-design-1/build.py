#!/usr/bin/env python3
"""
build.py  --  writes network.json and weights.npz for a hand-designed MNIST classifier.

No training data is used.  Every number comes from:
  * closed-form filters (Gaussian second-derivative "ridge" filters at 8 orientations, Gaussian blurs), and
  * synthetic digit templates: each digit is drawn as pen strokes along typed pen paths (ellipse arcs, lines and
    splines through hand-typed control points), with hand-chosen ranges of slant, width, stroke thickness and a
    smooth random warp (fixed seed).  Each drawing is normalised like MNIST (20 px box, centre of mass at the centre),
    passed through the fixed front-end, standardised by ITS OWN mean / std, and stored as one row of a dense layer.
    No statistic across several drawings is used anywhere.

Network:
  image -> conv c1 (8 ridge filters + 1 ink blur) -> relu -> maxpool 2x2 -> p1 (9,14,14)
  coarse branch: p1 -> conv c2 (Gaussian, stride 2) -> (9,7,7)   -> reshape (1,1,441)  -> norm -> dense dA (T)
  fine branch:   p1 -> conv c3 (Gaussian, stride 1) -> (9,14,14) -> reshape (1,1,1764) -> norm -> dense dB (T)
  add -> (T,) : for each template, Pearson correlation (coarse) + Pearson correlation (fine) with the image
  reshape (1,10,T/10) -> maxpool over the templates of each class -> reshape (10,)
  i.e. the class score is the best correlation between the image and any synthetic drawing of that digit.
"""
import sys, os, json
import numpy as np

SEED = 12345
N_PER_CLASS = 2500          # templates per class
N_ORIENT = 8                # ridge-filter orientations
K1 = 11                     # first conv kernel size
SIG_U, SIG_V = 1.6, 2.5     # across / along stroke sigma of the ridge filters (pixels); the across sigma is chosen so
                            # that strokes from 2 to 5 px wide give a similar response (thin and thick writing alike)
SIG_INK = 1.0               # sigma of the ink-blur channel
GAIN_INK = 1.5              # weight of the ink channel relative to the orientation channels
GAIN_RIDGE = 1.0
POOL, POOL_STRIDE = 2, 2    # max-pool window / stride after the first conv: 28 -> 14
K2, SIG_POOL = 5, 1.0       # coarse branch: stride-2 Gaussian pooling conv (in 14x14 units) -> 7x7
K3, SIG_FINE = 3, 0.7       # fine branch: light Gaussian smoothing at 14x14


# ----------------------------------------------------------------------------------------------------------------
# geometry helpers: all shapes live in a unit box, x to the right, y DOWN, before transformation
# ----------------------------------------------------------------------------------------------------------------
def U(rng, a, b):
    return a + (b - a) * rng.random()


def ell(cx, cy, rx, ry, a0, a1, step=9.0):
    """points along an ellipse arc; angles in degrees, 0 = right, 90 = down, 180 = left, 270 = up."""
    n = max(6, int(abs(a1 - a0) / step) + 1)
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def seg(p0, p1, n=4):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(p0, float)[None] * (1 - t) + np.asarray(p1, float)[None] * t


def catrom(pts, n_per=6):
    """Catmull-Rom spline through the typed control points."""
    pts = np.asarray(pts, float)
    P = np.concatenate([pts[:1], pts, pts[-1:]], 0)
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t = np.linspace(0, 1, n_per, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1:])
    return np.concatenate(out, 0)


def join(*parts):
    return np.concatenate([np.asarray(p, float) for p in parts], 0)


# ----------------------------------------------------------------------------------------------------------------
# digit pen paths (typed from general knowledge of how each digit is written)
# ----------------------------------------------------------------------------------------------------------------
def gen0(rng):
    rx = U(rng, 0.2, 0.42)
    g = U(rng, -25, 8)                       # <0: the loop overlaps itself a little, >0: small gap at the top
    taper = U(rng, -0.15, 0.3)               # >0: narrower at the top (egg shape)
    pts = ell(0.5, 0.5, rx, 0.5, -90 + g, 270 - g, step=6)
    pts[:, 0] = 0.5 + (pts[:, 0] - 0.5) * (1 + taper * (pts[:, 1] - 0.5) * 2)
    if rng.random() < 0.25:                  # the stroke starts with a small hook inside the loop, near the top
        return [join(seg((0.5 + rx * U(rng, 0.2, 0.5), U(rng, 0.15, 0.3)), pts[0], 3), pts[1:])]
    return [pts]


def gen1(rng):
    style = rng.random()
    bow = U(rng, -0.06, 0.06)
    t = np.linspace(0, 1, 12)[:, None]
    bar = np.concatenate([0.5 + bow * np.sin(np.pi * t), t], 1)
    strokes = [bar]
    if style > 0.45:                         # small flag at the top, going down-left
        fl = U(rng, 0.1, 0.35)
        fx = U(rng, 0.4, 1.0)
        strokes.append(seg((0.5 - fl * fx, fl), (0.5, 0.0)))
    if style > 0.85:                         # base
        bw = U(rng, 0.1, 0.25)
        strokes.append(seg((0.5 - bw, 1.0), (0.5 + bw, 1.0)))
    return strokes


def gen2(rng):
    style = rng.random()
    hx = U(rng, 0.16, 0.38)                  # head half-width
    hy = U(rng, 0.15, 0.3)                   # head half-height
    hcy = U(rng, 0.18, 0.33)
    base_y = U(rng, 0.92, 1.0)
    a0 = U(rng, 180, 215)
    a1 = U(rng, 360, 410)
    bl = U(rng, 0.03, 0.15)                  # x of the bottom-left corner
    xr = U(rng, 0.75, 0.98)
    rise = U(rng, 0.0, 0.12)                 # the base may rise towards its right end
    head = ell(0.5, hcy, hx, hy, a0, a1)
    end = head[-1]
    if style < 0.45:                         # ordinary 2: curved diagonal, base
        c = U(rng, -0.1, 0.12)
        mid = ((end[0] + bl) / 2 + c, (end[1] + base_y) / 2 + c * 0.3)
        diag = catrom([end, mid, (bl, base_y)])
        base = catrom([(bl, base_y), ((bl + xr) / 2, base_y + U(rng, -0.03, 0.02)), (xr, base_y - rise)], 5)
        body = join(head, diag[1:], base[1:])
    elif style < 0.7:                        # 2 with a small loop at the bottom left
        lr = U(rng, 0.05, 0.11)
        lcx, lcy = bl + lr + 0.03, base_y - lr
        entry = (lcx + lr * 0.707, lcy - lr * 0.707)
        diag = catrom([end, ((end[0] + entry[0]) / 2 + 0.04, (end[1] + entry[1]) / 2), entry])
        loop = ell(lcx, lcy, lr, lr, -45, -45 - 300, step=15)
        body = join(head, diag[1:], loop[1:], seg(loop[-1], (xr, base_y - rise * 0.5)))
    else:                                    # Z-like 2: flatter head, straight diagonal, straight base
        head = ell(0.5, hcy * 0.85, hx, hy * 0.75, a0, a1 - 10)
        body = join(head, seg(head[-1], (bl, base_y)), seg((bl, base_y), (xr, base_y - rise * 0.5)))
    return [body]


def gen3(rng):
    style = rng.random()
    rx1 = U(rng, 0.2, 0.34)
    ry1 = U(rng, 0.18, 0.27)
    rx2 = U(rng, 0.25, 0.4)
    ry2 = U(rng, 0.24, 0.32)
    cy1 = ry1 + 0.02
    cy2 = 1.0 - ry2
    e1 = U(rng, 0, 40)
    e2 = U(rng, 0, 40)
    bot = ell(0.5, cy2, rx2, ry2, 270 - e2, 520 + U(rng, -10, 20))     # over the right side, round to bottom-left
    if style < 0.8:                          # round top: over the top, round the right, in to the middle
        top = ell(0.5, cy1, rx1, ry1, U(rng, 190, 215), 450 + e1)
    else:                                    # flat top: bar, then diagonal down to the middle
        top = seg((U(rng, 0.1, 0.2), U(rng, 0.02, 0.06)), (U(rng, 0.78, 0.9), 0.03))
    return [join(top, seg(top[-1], bot[0], 3)[1:], bot[1:])]


def gen4(rng):
    style = rng.random()
    yc = U(rng, 0.45, 0.66)                  # crossbar height
    xv = U(rng, 0.55, 0.76)                  # x of the vertical
    sl = U(rng, -0.05, 0.15)                 # lean of the vertical
    xl = U(rng, 0.02, 0.2)                   # left end of the crossbar
    xr = U(rng, xv + 0.05, min(xv + 0.32, 1.0))   # the crossbar overhangs the vertical by a little
    vtop = 0.0 if rng.random() < 0.85 else U(rng, 0.0, 0.15)   # the vertical nearly always starts at the top
    vert = seg((xv + sl * (0.5 - vtop), vtop), (xv - sl * 0.5, 1.0))
    if style < 0.3:                          # open 4, straight slanted left stroke
        xt = U(rng, 0.3, 0.6)
        left = catrom([(xt, 0.0), ((xt + xl) / 2 - U(rng, 0, 0.05), yc / 2), (xl, yc)], 5)
        a = join(left, seg((xl, yc), (xr, yc))[1:])
    elif style < 0.65:                       # open 4, left stroke goes down then turns into the crossbar
        xt = xl + U(rng, 0.0, 0.15)
        left = catrom([(xt, 0.0), (xl + 0.01, yc * 0.7), (xl, yc)], 5)
        a = join(left, seg((xl, yc), (xr, yc))[1:])
    else:                                    # closed 4 (triangle)
        a = join(seg((xv + sl * 0.5 - 0.02, 0.02), (xl, yc)), seg((xl, yc), (xr, yc))[1:])
    return [a, vert]


def gen5(rng):
    style = rng.random()
    bx0 = U(rng, 0.6, 0.92)                  # top bar from bx0 to bx1
    bx1 = U(rng, 0.15, 0.32)
    by = U(rng, 0.0, 0.1)
    yv = U(rng, 0.38, 0.55)                  # bottom of the "vertical"
    vslant = U(rng, -0.02, 0.18)             # the vertical leans down-left
    rx = U(rng, 0.26, 0.38)
    ry = U(rng, 0.23, 0.32)
    cy = 1 - ry - 0.01
    bcx = min(U(rng, 0.47, 0.55), 0.93 - rx)      # the belly stays within the digit's width
    belly = ell(bcx, cy, rx, ry, U(rng, 230, 262), U(rng, 495, 540))  # over the right side, round to bottom-left
    vbot = (bx1 - vslant, yv)
    if style < 0.55:                         # bar + (vertical, belly) as two strokes
        bar = seg((bx0, by + U(rng, -0.03, 0.03)), (bx1, by))
        vert = catrom([(bx1 + U(rng, -0.02, 0.02), by), vbot, belly[0]], 5)
        return [bar, join(vert, belly[1:])]
    if style < 0.8:                          # one continuous stroke with a rounded top-left corner
        path = catrom([(bx0, by + 0.02), (bx1 + 0.04, by), vbot, belly[0]], 5)
        return [join(path, belly[1:])]
    path = catrom([(bx0, by), (bx1 + 0.05, by + 0.02), (bx1 - vslant - 0.04, yv * 0.7), vbot, belly[0]], 5)   # S-like
    return [join(path, belly[1:])]


def gen6(rng):
    style = rng.random()
    rx = U(rng, 0.22, 0.34)
    ry = U(rng, 0.2, 0.3)
    cx = U(rng, 0.42, 0.54)
    cy = 1 - ry - 0.01
    loop = ell(cx, cy, rx, ry, 180, -180 + U(rng, -40, 10))    # from the left: down, right, up, back to the left
    top = (U(rng, 0.55, 0.85), 0.0)
    if style < 0.65:                         # curved stem
        stem = catrom([top, (U(rng, 0.35, 0.5), U(rng, 0.12, 0.22)),
                       (cx - rx + 0.03 - U(rng, 0.0, 0.06), U(rng, 0.4, 0.5)), (cx - rx, cy)], 5)
    else:                                    # straighter stem
        stem = catrom([top, ((top[0] + cx - rx) / 2 - U(rng, 0, 0.04), cy / 2), (cx - rx, cy)], 5)
    return [join(stem, loop[1:])]


def gen7(rng):
    style = rng.random()
    x0 = U(rng, 0.02, 0.18)
    x1 = U(rng, 0.8, 1.0)
    y0 = U(rng, 0.0, 0.12)
    y1 = y0 + U(rng, -0.08, 0.05)
    xe = U(rng, 0.15, 0.5)
    c = U(rng, -0.1, 0.08)
    bar = catrom([(x0, y0), ((x0 + x1) / 2, (y0 + y1) / 2 + U(rng, -0.04, 0.04)), (x1, y1)], 4)
    mid = ((x1 + xe) / 2 + c, 0.5 + U(rng, -0.05, 0.05))
    diag = catrom([(x1, y1), mid, (xe, 1.0)])
    strokes = [join(bar, diag[1:])]
    if style < 0.12:                         # crossbar
        yc = U(rng, 0.42, 0.6)
        xc = mid[0] + (xe - mid[0]) * (yc - 0.5) / 0.5 if yc > 0.5 else mid[0] + (x1 - mid[0]) * (0.5 - yc) / 0.5
        strokes.append(seg((xc - U(rng, 0.12, 0.2), yc), (xc + U(rng, 0.12, 0.22), yc)))
    elif style < 0.4:                        # small tick at the top left
        strokes.append(seg((x0, y0), (x0 + U(rng, -0.04, 0.03), y0 + U(rng, 0.08, 0.22)), 3))
    return strokes


def gen8(rng):
    style = rng.random()
    if style < 0.45:                         # two touching loops, the upper one smaller
        rt = U(rng, 0.14, 0.26)
        ryt = U(rng, 0.17, 0.25)
        rb = U(rng, 0.18, 0.32)
        ryb = U(rng, 0.22, 0.3)
        ox = U(rng, -0.04, 0.1)
        top = ell(0.5 + ox, 0.01 + ryt, rt, ryt, 90, 450)
        bot = ell(0.5, 2 * ryt + ryb, rb, ryb, 270, -90)
        return [join(top, bot)]
    a_t = U(rng, 0.13, 0.25)                 # crossing figure-eight (lemniscate), upper half narrower
    a_b = U(rng, 0.16, 0.3)
    bt = U(rng, 0.4, 0.5)
    bb = U(rng, 0.46, 0.54)
    ox = U(rng, -0.03, 0.1)
    t = np.linspace(0, 2 * np.pi, 48)
    up = np.cos(t) < 0
    a = np.where(up, a_t, a_b)
    b = np.where(up, bt, bb)
    x = 0.5 + a * np.sin(2 * t) + np.where(up, ox, 0.0) * (-np.cos(t))
    y = 0.5 + b * np.cos(t)
    return [np.stack([x, y], 1)]


def gen9(rng):
    style = rng.random()
    if style < 0.4:                          # "P-like" 9: straight stem from the very top, loop hung on its upper left
        xs_top = U(rng, 0.62, 0.8)
        xs_bot = xs_top - U(rng, 0.05, 0.35)
        rx = U(rng, 0.22, 0.36)
        ry = U(rng, 0.17, 0.28)
        e = U(rng, 0.15, 0.4) * rx            # how far the loop's ellipse reaches past the stem (cut off by it)
        cx = xs_top - rx + e
        a = np.degrees(np.arccos((rx - e) / rx))
        loop = ell(cx, ry, rx, ry, -a, a - 360)          # from the stem top: over the top, down the left, back to the stem
        stem = seg((xs_top, 0.0), (xs_bot, 1.0))
        return [loop, stem]
    rx = U(rng, 0.17, 0.33)                  # round loop with the stem leaving its right side
    ry = U(rng, 0.16, 0.28)
    cx = U(rng, 0.42, 0.52)
    cy = ry + U(rng, 0.0, 0.05)
    loop = ell(cx, cy, rx, ry, U(rng, -20, 10), 360 + U(rng, -25, 30))
    start = (cx + rx, cy)
    if style < 0.7:                          # straight stem
        stem = seg(start, (cx + rx - U(rng, 0.05, 0.4), 1.0))
    elif style < 0.9:                        # stem curving to the left at the bottom
        stem = catrom([start, (cx + rx - U(rng, 0.0, 0.06), 0.62), (cx + rx - U(rng, 0.12, 0.25), 0.95),
                       (cx + rx - U(rng, 0.3, 0.45), 0.98)], 5)
    else:                                    # slightly bent stem
        stem = catrom([start, (cx + rx - U(rng, 0.05, 0.2), 0.75), (cx + rx - U(rng, 0.15, 0.35), 1.0)], 5)
    return [join(loop, seg(loop[-1], start, 2)[1:], stem[1:])]


GEN = [gen0, gen1, gen2, gen3, gen4, gen5, gen6, gen7, gen8, gen9]
SHEAR = {k: (-0.3, 0.55) for k in range(10)}
SHEAR[1] = (-0.45, 0.85)
SHEAR[7] = (-0.3, 0.6)
SHEAR[8] = (-0.3, 0.7)
XSCALE = {k: (0.7, 1.2) for k in range(10)}
for k in (4, 7, 9):
    XSCALE[k] = (0.6, 1.15)


def transform(strokes, rng, cls):
    """hand-chosen ranges: smooth warp, width scaling, shear (slant), rotation."""
    sh = U(rng, *SHEAR[cls])
    rot = np.radians(U(rng, -4, 4))          # slant is modelled by shear; whole-digit rotation is small
    xs = U(rng, *XSCALE[cls])
    amp = U(rng, 0, 0.03)
    f1, f2 = U(rng, 0.5, 1.5), U(rng, 0.5, 1.5)
    ph1, ph2 = U(rng, 0, 2 * np.pi), U(rng, 0, 2 * np.pi)
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]])
    out = []
    for p in strokes:
        q = p.copy()
        q[:, 0] += amp * np.sin(2 * np.pi * f1 * q[:, 1] + ph1)
        q[:, 1] += amp * np.sin(2 * np.pi * f2 * q[:, 0] + ph2)
        q[:, 0] = 0.5 + (q[:, 0] - 0.5) * xs
        q[:, 0] += sh * (0.5 - q[:, 1])
        q = (q - 0.5) @ R.T + 0.5
        out.append(q)
    return out


# ----------------------------------------------------------------------------------------------------------------
# rasteriser with MNIST-style normalisation
# ----------------------------------------------------------------------------------------------------------------
def raster(strokes_px, r, size=28, ss=2):
    """strokes in pixel coordinates (pixel i has centre i); round brush of radius r; ss x ss supersampling."""
    S = size * ss
    g = (np.arange(S) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    A = np.concatenate([p[:-1] for p in strokes_px if len(p) > 1], 0)            # segment starts (m, 2)
    D = np.concatenate([p[1:] - p[:-1] for p in strokes_px if len(p) > 1], 0)    # segment vectors (m, 2)
    PA, PD = P @ A.T, P @ D.T
    AD, AA, DD, PP = (A * D).sum(1), (A * A).sum(1), (D * D).sum(1) + 1e-12, (P * P).sum(1)
    t = np.clip((PD - AD) / DD, 0, 1)
    d2 = PP[:, None] + AA[None] - 2 * PA - 2 * t * (PD - AD) + t * t * DD[None]   # |P - a - t d|^2
    d = np.sqrt(np.maximum(d2.min(1), 0))
    cov = np.clip((r - d) * ss + 0.5, 0, 1).reshape(size, ss, size, ss).mean((1, 3))
    return cov


def render(strokes, r, size=28, box=20, ss=2):
    """scale the drawing so its ink fits a box x box square (aspect kept), centre of mass at the centre."""
    allp = np.concatenate(strokes, 0)
    mn, mx = allp.min(0), allp.max(0)
    ext = max(float((mx - mn).max()), 1e-6)
    s = (box - 2 * r) / ext
    strokes_px = [(p - mn) * s + r + 2.0 for p in strokes]
    img = raster(strokes_px, r, size, 1)             # cheap pass, only to find the centre of mass
    ys, xs = np.mgrid[0:size, 0:size]
    m = img.sum() + 1e-9
    cy, cx = (img * ys).sum() / m, (img * xs).sum() / m
    shift = np.array([(size - 1) / 2 - cx, (size - 1) / 2 - cy])
    return raster([p + shift for p in strokes_px], r, size, ss)


def make_templates(n_per_class, seed):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((10 * n_per_class, 1, 28, 28), np.float32)
    labels = np.repeat(np.arange(10), n_per_class)
    i = 0
    for cls in range(10):
        for _ in range(n_per_class):
            strokes = transform(GEN[cls](rng), rng, cls)
            r = U(rng, 0.9, 2.6)                      # stroke radius in pixels (width 1.8 .. 5.2 px)
            imgs[i, 0] = render(strokes, r)
            i += 1
    return imgs, labels


# ----------------------------------------------------------------------------------------------------------------
# fixed front-end (closed-form filters) and a numpy forward pass with the scorer's semantics
# ----------------------------------------------------------------------------------------------------------------
def gauss2d(k, sigma):
    c = k // 2
    y, x = np.mgrid[-c:c + 1, -c:c + 1].astype(float)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ridge_filters(n_orient, k, su, sv):
    c = k // 2
    y, x = np.mgrid[-c:c + 1, -c:c + 1].astype(float)
    Ws = []
    for i in range(n_orient):
        th = np.pi * i / n_orient
        v = x * np.cos(th) + y * np.sin(th)          # along the stroke
        u = -x * np.sin(th) + y * np.cos(th)         # across the stroke
        f = (1 - u ** 2 / su ** 2) * np.exp(-u ** 2 / (2 * su ** 2)) * np.exp(-v ** 2 / (2 * sv ** 2))
        f = f - f.mean()
        f = f / f[f > 0].sum()
        Ws.append(f)
    return np.stack(Ws)


def depthwise(C, g):
    W = np.zeros((C, C, g.shape[0], g.shape[1]))
    for c in range(C):
        W[c, c] = g
    return W


def front_end_weights():
    C = N_ORIENT + 1
    W1 = np.zeros((C, 1, K1, K1))
    W1[:N_ORIENT, 0] = GAIN_RIDGE * ridge_filters(N_ORIENT, K1, SIG_U, SIG_V)
    W1[N_ORIENT, 0] = GAIN_INK * gauss2d(K1, SIG_INK)
    W2 = depthwise(C, gauss2d(K2, SIG_POOL))
    W3 = depthwise(C, gauss2d(K3, SIG_FINE))
    return W1, np.zeros(C), W2, np.zeros(C), W3, np.zeros(C)


def conv2d(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
                     for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def maxpool(x, size, stride):
    n, c, h, w = x.shape
    ho, wo = (h - size) // stride + 1, (w - size) // stride + 1
    win = np.stack([x[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
                    for dy in range(size) for dx in range(size)], axis=0)
    return win.max(0)


def features(imgs, W1, b1, W2, b2, W3, b3):
    FA, FB = [], []
    for i in range(0, len(imgs), 250):
        x = imgs[i:i + 250].astype(np.float64)
        x = np.maximum(conv2d(x, W1, b1, 1, K1 // 2), 0)
        x = maxpool(x, POOL, POOL_STRIDE)
        FA.append(conv2d(x, W2, b2, 2, K2 // 2).reshape(len(x), -1))
        FB.append(conv2d(x, W3, b3, 1, K3 // 2).reshape(len(x), -1))
    return np.concatenate(FA, 0), np.concatenate(FB, 0)


def standardise_rows(F):
    """each template standardised by its own mean and std (as the network's norm layer does to the image), then
    divided by the dimension so that the dense output is the Pearson correlation.  Each row uses its own drawing only."""
    mu = F.mean(1, keepdims=True)
    sd = np.sqrt(((F - mu) ** 2).mean(1, keepdims=True) + 1e-5)
    return ((F - mu) / sd) / F.shape[1]


# ----------------------------------------------------------------------------------------------------------------
def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    W1, b1, W2, b2, W3, b3 = front_end_weights()
    imgs, labels = make_templates(N_PER_CLASS, SEED)
    FA, FB = features(imgs, W1, b1, W2, b2, W3, b3)
    T, DA = FA.shape
    DB = FB.shape[1]
    WA, WB = standardise_rows(FA), standardise_rows(FB)
    C = N_ORIENT + 1
    layers = [
        {"name": "c1", "op": "conv", "inputs": ["image"], "out": C, "kernel": [K1, K1], "stride": 1, "padding": K1 // 2},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "p1", "op": "maxpool", "inputs": ["r1"], "size": [POOL, POOL], "stride": [POOL_STRIDE, POOL_STRIDE]},
        {"name": "c2", "op": "conv", "inputs": ["p1"], "out": C, "kernel": [K2, K2], "stride": 2, "padding": K2 // 2},
        {"name": "flatA", "op": "reshape", "inputs": ["c2"], "shape": [1, 1, DA]},
        {"name": "nrmA", "op": "norm", "inputs": ["flatA"]},
        {"name": "dA", "op": "dense", "inputs": ["nrmA"], "out": T},
        {"name": "c3", "op": "conv", "inputs": ["p1"], "out": C, "kernel": [K3, K3], "stride": 1, "padding": K3 // 2},
        {"name": "flatB", "op": "reshape", "inputs": ["c3"], "shape": [1, 1, DB]},
        {"name": "nrmB", "op": "norm", "inputs": ["flatB"]},
        {"name": "dB", "op": "dense", "inputs": ["nrmB"], "out": T},
        {"name": "sum", "op": "add", "inputs": ["dA", "dB"]},
        {"name": "grp", "op": "reshape", "inputs": ["sum"], "shape": [1, 10, N_PER_CLASS]},
        {"name": "cls", "op": "maxpool", "inputs": ["grp"], "size": [1, N_PER_CLASS], "stride": [1, N_PER_CLASS]},
        {"name": "out", "op": "reshape", "inputs": ["cls"], "shape": [10]},
    ]
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    np.savez(os.path.join(out_dir, "weights.npz"),
             **{"c1.W": f32(W1), "c1.b": f32(b1), "c2.W": f32(W2), "c2.b": f32(b2), "c3.W": f32(W3), "c3.b": f32(b3),
                "dA.W": f32(WA), "dA.b": f32(np.zeros(T)), "dB.W": f32(WB), "dB.b": f32(np.zeros(T))})
    return imgs, labels


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
