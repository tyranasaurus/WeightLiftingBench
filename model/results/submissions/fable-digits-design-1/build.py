"""build.py - writes network.json and weights.npz for the handwritten-digit network.   usage: python3 build.py OUT_DIR

Nothing is read from disk.  Every weight comes from the code below:
  * the front-end normalises the digit's position, slant, width and height.  Each stage holds transformed copies of the image
    (linear-interpolation matrices from a formula) and keeps the copy whose bin contains the measured value, a ratio of
    image moments tested with ReLUs;
  * the convolution filters are formulas (derivatives of a Gaussian in 8 directions, a 1x1 "sharpening" of the direction
    tuning, a Gaussian pooling filter);
  * the matching layers hold templates.  Each template is ONE digit image drawn by this program from pen paths typed
    from general knowledge of how the ten digits are written (with a seeded random wobble, slant, width and pen
    thickness), passed through the same normalisation and filters and normalised by itself alone (its own mean and its
    own standard deviation).  No statistic is taken across generated images, and no training data is used anywhere.
    A template is cut into 3 x 3 parts; each part may shift by one cell (2 pixels) to follow the test digit.
The network reports, for every class, the score of the best-matching template of that class (classes with longer,
more variable pen paths get more templates).
"""
import sys, json
import numpy as np

SEED = 20261119       # fixed seed for the pen wobble

# ====================================================================================================
# gen
# ====================================================================================================
# ---------------------------------------------------------------- splines
def _lerp(a, b, ta, tb, t):
    w = ((t - ta) / (tb - ta))[..., None]
    return a * (1 - w) + b * w

def spline(P, n=6, closed=False):
    """Centripetal Catmull-Rom curve through control points P (B, K, 2) -> polyline (B, M, 2)."""
    B, K, _ = P.shape
    if K == 2 and not closed:
        t = np.linspace(0, 1, n + 1)[None, :, None]
        return P[:, :1] * (1 - t) + P[:, 1:] * t
    if closed:
        Pp = np.concatenate([P[:, -1:], P, P[:, :2]], 1)
    else:
        Pp = np.concatenate([2 * P[:, :1] - P[:, 1:2], P, 2 * P[:, -1:] - P[:, -2:-1]], 1)
    d = np.sqrt(np.sqrt(((Pp[:, 1:] - Pp[:, :-1]) ** 2).sum(-1)) + 1e-6)          # |dP|^0.5
    p0, p1, p2, p3 = (Pp[:, i:Pp.shape[1] - 3 + i][:, :, None] for i in range(4))  # (B, S, 1, 2)
    d01, d12, d23 = d[:, :-2, None], d[:, 1:-1, None], d[:, 2:, None]              # (B, S, 1)
    u = np.arange(n) / n
    t0 = np.zeros_like(d01); t1 = d01; t2 = t1 + d12; t3 = t2 + d23
    t = t1 + u * d12                                                               # (B, S, n)
    A1 = _lerp(p0, p1, t0, t1, t); A2 = _lerp(p1, p2, t1, t2, t); A3 = _lerp(p2, p3, t2, t3, t)
    B1 = _lerp(A1, A2, t0, t2, t); B2 = _lerp(A2, A3, t1, t3, t)
    C = _lerp(B1, B2, t1, t2, t)                                                   # (B, S, n, 2)
    pts = C.reshape(B, -1, 2)
    last = P[:, :1] if closed else P[:, -1:]
    return np.concatenate([pts, last], 1)

# ---------------------------------------------------------------- rendering
_YY, _XX = np.meshgrid(np.arange(28.0), np.arange(28.0), indexing="ij")
_PX = _XX.reshape(-1).astype(np.float32); _PY = _YY.reshape(-1).astype(np.float32)
SOFT = 1.2          # width in pixels of the grey ramp at a stroke edge (anti-aliasing)

def render(A, Bp, th):
    """A, Bp: (B, S, 2) segment end points in pixels (x, y); th (B, S) stroke thickness. -> (B, 784) in [0, 1]."""
    ax = A[:, :, 0, None].astype(np.float32); ay = A[:, :, 1, None].astype(np.float32)
    dx = Bp[:, :, 0, None].astype(np.float32) - ax; dy = Bp[:, :, 1, None].astype(np.float32) - ay
    qx = _PX - ax; qy = _PY - ay
    t = np.clip((qx * dx + qy * dy) / (dx * dx + dy * dy + 1e-9), 0, 1)
    ex = qx - t * dx; ey = qy - t * dy
    m = (th[:, :, None].astype(np.float32) / 2 - np.sqrt(ex * ex + ey * ey)).max(1)   # (B, 784) signed depth inside the stroke
    return np.clip(0.5 + m / SOFT, 0, 1)

def rasterise(strokes, th, rng, jitter=0.5, centre=13.5, box=20.0, thvar=0.0):
    """strokes: list of polylines (B, M_i, 2) in any common unit.  Mimics the dataset's preparation: the ink is
    scaled to fit a 20 x 20 box (keeping proportions) and placed with its centre of mass at the frame centre."""
    B = strokes[0].shape[0]
    allp = np.concatenate(strokes, 1)
    mn, mx = allp.min(1), allp.max(1)
    scale = (box - th) / (mx - mn).max(1)                                  # path box + thickness = 20 pixels
    mid = (mn + mx) / 2
    A = np.concatenate([(s[:, :-1] - mid[:, None]) * scale[:, None, None] for s in strokes], 1)
    Bp = np.concatenate([(s[:, 1:] - mid[:, None]) * scale[:, None, None] for s in strokes], 1)
    S = A.shape[1]
    ths = th[:, None] * np.ones((1, S))
    if thvar > 0:    # slow variation of pen width along the path
        ph = rng.uniform(0, 2 * np.pi, (B, 1)); fr = rng.uniform(0.5, 1.5, (B, 1)); am = rng.uniform(0, thvar, (B, 1))
        ths = ths * (1 + am * np.sin(ph + fr * 2 * np.pi * np.arange(S)[None] / S))
    img = render(A + centre, Bp + centre, ths)
    w = img.sum(1) + 1e-9
    com = np.stack([(img * _PX).sum(1) / w, (img * _PY).sum(1) / w], 1)
    shift = centre - com + rng.uniform(-jitter, jitter, (B, 2))
    return render(A + centre + shift[:, None], Bp + centre + shift[:, None], ths)

# ---------------------------------------------------------------- global shape variation
def warp(strokes, rng, aspect, shear=(0.25, 0.3, -0.35, 0.9), rot=6.0, taper=0.2, bend=0.05, vshift=0.25, hshift=0.12, wave=0.03):
    """Whole-digit variation: width, slant (shear), small rotation, taper, bend, moving the middle up or down,
    and a gentle smooth wave that shifts parts of the digit against each other."""
    B = strokes[0].shape[0]
    a = rng.uniform(aspect[0], aspect[1], (B, 1))
    s = np.clip(rng.normal(shear[0], shear[1], (B, 1)), shear[2], shear[3])
    r = np.deg2rad(rng.normal(0, rot, (B, 1)))
    k = rng.normal(0, taper, (B, 1)); b = rng.normal(0, bend, (B, 1)); c = rng.normal(0, vshift, (B, 1)); e = rng.normal(0, hshift, (B, 1))
    wa = rng.normal(0, wave, (B, 4)); wf = rng.uniform(-1.2, 1.2, (B, 4, 2)); wp = rng.uniform(0, 2 * np.pi, (B, 4))
    out = []
    for P in strokes:
        u = P[..., 0] - 0.5; v = P[..., 1] - 0.5
        ph = [2 * np.pi * (wf[:, j, :1] * u + wf[:, j, 1:] * v) + wp[:, j, None] for j in range(4)]
        u = u + wa[:, 0, None] * np.sin(ph[0]) + wa[:, 1, None] * np.sin(ph[1])
        v = v + wa[:, 2, None] * np.sin(ph[2]) + wa[:, 3, None] * np.sin(ph[3])
        v2 = v + c * (0.25 - v * v)
        u = u + e * (0.25 - u * u)
        x = a * u * (1 + k * v) + b * np.cos(np.pi * v) - s * v2
        y = v2
        out.append(np.stack([x * np.cos(r) - y * np.sin(r), x * np.sin(r) + y * np.cos(r)], -1))
    return out

def thickness(rng, B, lo=1.3, hi=5.0, mode=2.2):
    """Pen width in pixels after the 20-pixel normalisation: mostly 2 to 3.5, sometimes hairline or bold."""
    return rng.triangular(lo, mode, hi, B)

# ====================================================================================================
# styles
# ====================================================================================================
def U(rng, B, lo, hi):
    return rng.uniform(lo, hi, B)

def pts(rng, B, sig, *xy):
    """Control points from typed positions.  Each position is (x, y); x and y may be numbers or per-sample arrays.
    `sig` is the independent pen wobble added to every point."""
    Q = np.zeros((B, len(xy), 2))
    for i, (x, y) in enumerate(xy):
        Q[:, i, 0] = x; Q[:, i, 1] = y
    return Q + rng.normal(0, sig, Q.shape)

# ------------------------------------------------------------------ 0
def zero(rng, B, v):
    n = 40
    u = np.linspace(0, 1, n)[None]
    start = U(rng, B, -0.5, 0.9)[:, None]                  # where on the top the pen starts (radians from the top)
    over = np.clip(rng.normal(0.15, 0.25, (B, 1)), -0.3, 0.8)   # overlap (>0) or gap (<0) where the loop closes
    spiral = rng.normal(0, 0.08, (B, 1))
    if v == "curl":                                         # the end curls into the inside of the loop
        over = U(rng, B, 0.3, 0.9)[:, None]; spiral = -U(rng, B, 0.12, 0.3)[:, None]
    if v == "tail":                                         # the end crosses the start and sticks out
        over = U(rng, B, 0.3, 0.9)[:, None]; spiral = U(rng, B, 0.15, 0.45)[:, None]
    if v == "gap":                                          # left clearly open at the top
        over = -U(rng, B, 0.3, 0.9)[:, None]
    phi = -np.pi / 2 + start - u * (2 * np.pi + over)       # counter-clockwise on the page
    c2 = rng.normal(0, 0.05, (B, 1)); p2 = U(rng, B, 0, 2 * np.pi)[:, None]
    c3 = rng.normal(0, 0.035, (B, 1)); p3 = U(rng, B, 0, 2 * np.pi)[:, None]
    r = (1 + c2 * np.cos(2 * phi + p2) + c3 * np.cos(3 * phi + p3)) * (1 + spiral * (u - 0.5) * 2 * (u > 0.5))
    return [np.stack([0.5 + 0.5 * r * np.cos(phi), 0.5 + 0.5 * r * np.sin(phi)], -1)]

# ------------------------------------------------------------------ 1
def one(rng, B, v):
    bow = rng.normal(0, 0.03, B)
    Q = pts(rng, B, 0.0, (0.5, 0.0), (0.5 + bow, 0.33), (0.5 + bow + rng.normal(0, 0.015, B), 0.67), (0.5, 1.0))
    if v == "hook":                                         # slight hooks at the ends of the stroke
        a = pts(rng, B, 0.0, (0.5 - U(rng, B, 0.0, 0.07), U(rng, B, 0.0, 0.06)))
        b = pts(rng, B, 0.0, (0.5 + U(rng, B, -0.03, 0.08), 1.0 - U(rng, B, 0.0, 0.04)))
        Q = np.concatenate([a, Q, b], 1)
    out = [spline(Q)]
    if v in ("flag", "flagbase"):                           # short straight stroke up to the top from the lower left
        ln = U(rng, B, 0.15, 0.3); an = np.deg2rad(U(rng, B, 30, 60))
        out.append(spline(pts(rng, B, 0.0, (0.5 - ln * np.cos(an), ln * np.sin(an)), (0.5, 0.0))))
    if v == "flagbase":                                     # foot bar, reaching out on both sides of the stroke
        out.append(spline(pts(rng, B, 0.0, (0.5 - U(rng, B, 0.22, 0.4), 1.0 + rng.normal(0, 0.015, B)), (0.5 + U(rng, B, 0.22, 0.4), 1.0 + rng.normal(0, 0.015, B)))))
    return out

# ------------------------------------------------------------------ 2
def two(rng, B, v):
    xc = U(rng, B, 0.0, 0.25); yc = U(rng, B, 0.88, 1.0)    # bottom-left corner
    xe = U(rng, B, 0.8, 1.2); ye = yc + U(rng, B, -0.18, 0.06)   # end of the base
    if v.endswith("-short"):                                # the base is only a short flick
        xe = xc + U(rng, B, 0.4, 0.6); ye = yc + U(rng, B, -0.12, 0.04); v = v[:-6] + "-sharp"
    sag = U(rng, B, -0.06, 0.06); wav = U(rng, B, -0.05, 0.07)   # the base may sag and wave
    def base(x_from):
        return [(x_from + (xe - x_from) * 0.35, yc + (ye - yc) * 0.35 + sag + wav), (x_from + (xe - x_from) * 0.7, yc + (ye - yc) * 0.7 + sag - wav), (xe, ye)]
    if v == "z":                                            # angular, Z-like
        x0 = U(rng, B, 0.0, 0.2); xr = U(rng, B, 0.75, 0.95); y0 = U(rng, B, 0.02, 0.2)
        Q = pts(rng, B, 0.025, (x0, y0), ((x0 + xr) / 2, U(rng, B, -0.02, 0.08)), (xr, 0.05), ((xr + xc) / 2 + U(rng, B, -0.04, 0.08), 0.5), (xc, yc), *base(xc))
        return [spline(Q[:, :3]), spline(Q[:, 2:5]), spline(Q[:, 4:])]
    head_kind, bottom = v.split("-")
    x0 = U(rng, B, 0.08, 0.3); y0 = U(rng, B, 0.12, 0.32)   # left end of the top arc
    xa = U(rng, B, 0.4, 0.62)                               # top of the arc
    xr = U(rng, B, 0.72, 0.95); yr = U(rng, B, 0.15, 0.3)   # right-most point of the arc
    blg = U(rng, B, -0.08, 0.1)                             # the neck may bow either way
    head = [(x0, y0), (x0 + (xa - x0) * 0.4, y0 * 0.25), (xa, 0.0), (xr, yr), (xr - 0.07, yr + 0.2), (0.5 * xr + 0.5 * xc + blg, 0.5 * (yr + 0.2) + 0.5 * yc - 0.03)]
    if head_kind == "hook":                                 # the pen starts lower down: a short leg on the left of the arc
        head = [(x0 + U(rng, B, -0.02, 0.08), y0 + U(rng, B, 0.1, 0.22))] + head
    if head_kind == "curl":                                 # the pen starts with a small curl inside the arc
        head = [(x0 + U(rng, B, 0.06, 0.16), y0 + U(rng, B, 0.0, 0.1)), (x0 + U(rng, B, -0.02, 0.05), y0 + U(rng, B, 0.08, 0.16))] + head
    nh = len(head)
    if bottom == "loop":                                    # down the neck, round a loop at the bottom left, out along the base
        g = U(rng, B, 0.5, 1.8)                             # loop size
        lx = xc + 0.1; ly = yc - 0.05
        Q = pts(rng, B, 0.022, *head, (lx - 0.02 * g, ly + 0.02 * g), (lx - 0.14 * g, ly - 0.03 * g), (lx - 0.06 * g, ly - 0.15 * g), (lx + 0.14 * g, ly - 0.06 * g), *base(lx + 0.14 * g)[1:])
        return [spline(Q)]
    Q = pts(rng, B, 0.025, *head, (xc, yc), *base(xc))
    if bottom == "smooth":
        return [spline(Q)]
    return [spline(Q[:, :nh + 1]), spline(Q[:, nh:])]       # sharp corner at the bottom left

# ------------------------------------------------------------------ 3
def three(rng, B, v):
    xm = U(rng, B, 0.28, 0.6); ym = U(rng, B, 0.38, 0.55)   # the middle tip
    x1 = U(rng, B, 0.68, 0.9); x2 = U(rng, B, 0.76, 0.95)   # right extent of the upper and lower bowls
    xt = U(rng, B, 0.0, 0.25); yt = U(rng, B, 0.78, 0.95)   # end of the tail at the bottom left
    low = [(x2 - 0.1, ym + 0.1 * (1 - ym) + 0.03), (x2, ym + 0.5 * (1 - ym)), (0.5 * x2 + 0.5 * xt + 0.08, 1.0), (xt, yt)]
    if v == "flat":                                         # flat top bar, then a diagonal into the middle
        x0 = U(rng, B, 0.05, 0.25)
        Q = pts(rng, B, 0.025, (x0, U(rng, B, 0.0, 0.08)), (x1 + 0.02, U(rng, B, 0.0, 0.06)), (xm, ym), *low)
        return [spline(Q[:, :2]), spline(Q[:, 1:3]), spline(Q[:, 2:])]
    x0 = U(rng, B, 0.05, 0.3); y0 = U(rng, B, 0.06, 0.25)
    up = [(x0, y0), (0.5 * x0 + 0.5 * x1 - 0.05, 0.0), (x1 - 0.03, 0.2 * ym), (x1, 0.6 * ym)]
    if v == "loopmid":                                      # a tiny loop where the two bowls meet
        Q = pts(rng, B, 0.02, *up, (xm + 0.04, ym + 0.02), (xm - 0.04, ym - 0.03), (xm + 0.06, ym - 0.06), *low)
        return [spline(Q)]
    Q = pts(rng, B, 0.025, *up, (xm, ym), *low)
    if v == "smooth":
        return [spline(Q)]
    return [spline(Q[:, :5]), spline(Q[:, 4:])]             # cusp in the middle

# ------------------------------------------------------------------ 4
def four(rng, B, v):
    sx = U(rng, B, 0.55, 0.82)                              # stem position
    by = U(rng, B, 0.45, 0.72)                              # bar height
    st = U(rng, B, -0.02, 0.3)                              # top of the stem
    bar_end = np.maximum(sx + U(rng, B, 0.02, 0.3), 0.8)
    def stem(top):
        return pts(rng, B, 0.015, (sx + rng.normal(0, 0.04, B), top), (sx, 0.55), (sx + rng.normal(0, 0.05, B), 1.0))
    if v == "open":
        L = pts(rng, B, 0.02, (U(rng, B, 0.05, 0.42), U(rng, B, 0.0, 0.15)), (U(rng, B, 0.0, 0.2), by), (bar_end, by + U(rng, B, -0.12, 0.04)))
        return [spline(L[:, :2]), spline(L[:, 1:]), spline(stem(st))]
    if v == "round":                                        # the left stroke and bar are one rounded U-like curve
        x0 = np.minimum(U(rng, B, 0.03, 0.32), sx - 0.42)
        L = pts(rng, B, 0.02, (x0, U(rng, B, 0.0, 0.1)), (x0 - 0.04, by * 0.6), (x0 + 0.1, by + 0.01), (sx - 0.1, by), (bar_end, by + U(rng, B, -0.1, 0.03)))
        return [spline(L), spline(stem(st))]
    # closed top: a diagonal from the top of the stem to the left end of the bar
    top = U(rng, B, -0.02, 0.05); by2 = U(rng, B, 0.55, 0.75)
    S = stem(top)
    L = pts(rng, B, 0.02, (S[:, 0, 0] + rng.normal(0, 0.03, B), S[:, 0, 1] + U(rng, B, -0.02, 0.08)), (U(rng, B, 0.0, 0.15), by2), (bar_end, by2 + U(rng, B, -0.06, 0.03)))
    return [spline(L[:, :2]), spline(L[:, 1:]), spline(S)]

# ------------------------------------------------------------------ 5
def five(rng, B, v):
    xs = U(rng, B, 0.15, 0.35)                              # down stroke position
    yk = U(rng, B, 0.3, 0.5)                                # where the down stroke turns into the bowl
    xk = xs + U(rng, B, -0.1, 0.06)
    xr = U(rng, B, 0.7, 0.95)                               # right extent of the bowl
    xt = U(rng, B, 0.0, 0.25); yt = U(rng, B, 0.76, 0.95)   # end of the tail
    bowl = [(0.5 * xk + 0.5 * xr, yk - 0.05), (xr - 0.03, yk + 0.25 * (1 - yk)), (xr, yk + 0.6 * (1 - yk)), (0.55 * xr + 0.45 * xt, 0.98), (xt, yt)]
    if v == "s":                                            # written in one go, like an S
        xb = U(rng, B, 0.7, 0.98); yb = U(rng, B, -0.08, 0.08)
        Q = pts(rng, B, 0.025, (xb, yb), (0.5 * xb + 0.5 * xs, 0.02), (xs, 0.12), (xk, yk - 0.06), *bowl)
        return [spline(Q)]
    Q = pts(rng, B, 0.025, (xs, 0.04), (xk, yk), *bowl)
    ln = U(rng, B, 0.18, 0.38) if v == "shortbar" else U(rng, B, 0.35, 0.85)
    rise = U(rng, B, -0.03, 0.22)
    ox = U(rng, B, 0.0, 0.18) if v == "apart" else 0.0      # the top bar may be written apart from the down stroke
    oy = U(rng, B, -0.1, 0.04) if v == "apart" else 0.0
    bx = Q[:, 0, 0] + ox; byy = Q[:, 0, 1] + oy
    T = pts(rng, B, 0.012, (bx, byy), (bx + ln / 2, byy - rise / 2), (bx + ln, byy - rise))
    if v == "roundc":                                       # rounded turn from the down stroke into the bowl
        return [spline(Q), spline(T)]
    return [spline(Q[:, :2]), spline(Q[:, 1:]), spline(T)]

# ------------------------------------------------------------------ 6
def six(rng, B, v):
    x0 = U(rng, B, 0.4, 1.0)                                # where the pen starts at the top
    lt = U(rng, B, 0.38, 0.65)                              # top of the loop
    if v == "bigloop": lt = U(rng, B, 0.3, 0.42)
    xl = U(rng, B, 0.1, 0.25); xr = U(rng, B, 0.7, 0.92)    # left and right of the loop
    hm = (lt + 1) / 2
    if v == "straight":                                     # straight stem, like a b
        x0 = U(rng, B, 0.15, 0.6); mid = (0.6 * x0 + 0.4 * xl, 0.4 * lt)
    else:
        mid = (0.35 * x0 + 0.65 * xl - 0.05, 0.45 * lt)
    Q = pts(rng, B, 0.025, (x0, U(rng, B, 0.0, 0.06)), mid, (xl, hm - 0.1), (xl + 0.03, hm + 0.14), (0.5 * xl + 0.5 * xr, 1.0), (xr, hm + 0.1),
            (xr - 0.06, lt + 0.06), (0.5 * xl + 0.5 * xr, lt), (xl + 0.03, lt + 0.14))
    if v == "open":                                         # the loop does not quite close
        Q = Q[:, :8]; Q[:, 7, 0] += U(rng, B, 0.0, 0.15); Q[:, 7, 1] += U(rng, B, -0.03, 0.08)
    if v == "spiral":                                       # the loop closes and carries on inside
        Q = np.concatenate([Q, pts(rng, B, 0.025, (0.4 * xl + 0.6 * xr - 0.15, hm + 0.12))], 1)
    return [spline(Q)]

# ------------------------------------------------------------------ 7
def seven(rng, B, v):
    xa = U(rng, B, 0.0, 0.2); ya = U(rng, B, 0.0, 0.14)     # left end of the top bar
    xb = U(rng, B, 0.8, 1.0); yb = U(rng, B, 0.0, 0.08)     # the corner
    xf = U(rng, B, 0.2, 0.8)                                # the foot
    blg = U(rng, B, -0.04, 0.12)
    Q = pts(rng, B, 0.022, (xa, ya), ((xa + xb) / 2, (ya + yb) / 2 + U(rng, B, -0.05, 0.04)), (xb, yb), ((xb + xf) / 2 + blg, 0.5), (xf, 1.0))
    if "round" in v:
        Q[:, 2, 0] -= 0.04; Q[:, 2, 1] += 0.02; out = [spline(Q)]
    else:
        out = [spline(Q[:, :3]), spline(Q[:, 2:])]
    if "tick" in v:                                         # short down-stroke at the left end of the top bar
        out.append(spline(pts(rng, B, 0.0, (Q[:, 0, 0] + U(rng, B, -0.06, 0.06), Q[:, 0, 1] + U(rng, B, 0.1, 0.3)), (Q[:, 0, 0], Q[:, 0, 1]))))
    if "cross" in v:                                        # bar across the middle of the down stroke
        cy = U(rng, B, 0.42, 0.62); cx = xb + (xf - xb) * cy + blg * 0.8
        out.append(spline(pts(rng, B, 0.0, (cx - U(rng, B, 0.18, 0.36), cy + U(rng, B, -0.03, 0.06)), (cx + U(rng, B, 0.12, 0.3), cy + U(rng, B, -0.08, 0.03)))))
    return out

# ------------------------------------------------------------------ 8
def eight(rng, B, v):
    cy = U(rng, B, 0.36, 0.58)                              # height of the crossing
    tw = U(rng, B, 0.5, 1.05); bw = U(rng, B, 0.45, 1.1)    # widths of the top and bottom loops
    if v == "thinlow": bw = U(rng, B, 0.33, 0.55)            # lower loop squeezed almost shut
    if v == "thintop": tw = U(rng, B, 0.33, 0.5)
    cx = 0.5 + U(rng, B, -0.06, 0.06)
    def T(x, y): return (0.5 + (x - 0.5) * tw, y * cy / 0.46)                     # a point of the top loop
    def L(x, y): return (0.5 + (x - 0.5) * bw, 1 - (1 - y) * (1 - cy) / 0.54)     # a point of the bottom loop
    if v == "closed":
        Q = pts(rng, B, 0.022, (cx, cy), T(0.78, 0.24), T(0.5, 0.0), T(0.2, 0.22), (cx, cy), L(0.84, 0.74), L(0.5, 1.0), L(0.16, 0.74))
        return [spline(Q, closed=True)]
    # written as an S and back: starts at the top right, ends by coming back up through the middle
    if v == "tails":                                        # the two ends stick out at the top right
        a = T(0.82 + U(rng, B, 0.0, 0.15), 0.14 + U(rng, B, -0.1, 0.05)); b = T(0.8 + U(rng, B, -0.05, 0.25), 0.12 + U(rng, B, -0.15, 0.05))
    elif v == "opentop":                                    # the stroke coming back up stops short: the top loop stays open
        a = T(0.82 + U(rng, B, -0.05, 0.1), 0.14 + U(rng, B, -0.06, 0.06)); b = T(0.7 + U(rng, B, -0.06, 0.1), 0.25 + U(rng, B, -0.06, 0.06))
    else:
        a = T(0.82, 0.14); b = T(0.8 + U(rng, B, -0.1, 0.05), 0.12 + U(rng, B, -0.02, 0.1))
    Q = pts(rng, B, 0.022, a, T(0.52, 0.0), T(0.22, 0.2), (cx, cy), L(0.82, 0.74), L(0.5, 1.0), L(0.18, 0.76), (cx, cy), b)
    return [spline(Q)]

# ------------------------------------------------------------------ 9
def nine(rng, B, v):
    lh = U(rng, B, 0.36, 0.62)                              # height of the loop
    if v == "small": lh = U(rng, B, 0.3, 0.4)
    lw = U(rng, B, 0.7, 1.1)                                # width of the loop
    def Lp(x, y): return (0.82 + (x - 0.82) * lw, y * lh / 0.5)
    gap = U(rng, B, 0.05, 0.35) if v == "gap" else 0.0      # the loop may be left open at the top right
    ax, ay = Lp(0.8 - gap, 0.16 - gap * 0.3)
    jx = 0.82 + U(rng, B, 0.0, 0.05); jy = (0.16 + U(rng, B, -0.1, 0.05)) * lh / 0.5       # where the stem leaves the loop
    Q = pts(rng, B, 0.022, (ax, ay), Lp(0.52, 0.0), Lp(0.2, 0.24), Lp(0.46, 0.5), Lp(0.8, 0.3), (jx, jy))
    if v == "hook":                                         # stem curls to the left at the bottom
        S = pts(rng, B, 0.025, (jx, jy), (jx + rng.normal(0, 0.03, B), 0.62), (0.62, 0.95), (0.25 + U(rng, B, -0.1, 0.15), 0.92 + U(rng, B, -0.1, 0.05)))
    else:
        fx = jx + U(rng, B, -0.3, 0.06)
        S = pts(rng, B, 0.018, (jx, jy), ((jx + fx) / 2 + rng.normal(0, 0.03, B), 0.56), (fx, 1.0))
    S[:, 0] = Q[:, -1]
    return [spline(Q), spline(S)]

# ------------------------------------------------------------------ the bank
# Templates per class: more for digits with longer, more variable pen paths (multiples of GROUP).
PER_CLASS = (1600, 1200, 4800, 2600, 1800, 2600, 2400, 2200, 4200, 3000)
# (class, style, variant, share of the class's templates, width range)
BANK = [
    (0, zero, "plain", 0.68, (0.5, 1.0)), (0, zero, "curl", 0.12, (0.5, 1.0)), (0, zero, "tail", 0.12, (0.5, 1.0)), (0, zero, "gap", 0.08, (0.5, 1.0)),
    (1, one, "plain", 0.80, (1, 1)), (1, one, "hook", 0.08, (1, 1)), (1, one, "flag", 0.09, (1, 1)), (1, one, "flagbase", 0.03, (1, 1)),
    (2, two, "arc-sharp", 0.19, (0.6, 1.1)), (2, two, "arc-loop", 0.19, (0.6, 1.1)), (2, two, "arc-smooth", 0.07, (0.6, 1.1)), (2, two, "arc-short", 0.04, (0.6, 1.1)),
    (2, two, "hook-sharp", 0.14, (0.6, 1.1)), (2, two, "hook-loop", 0.13, (0.6, 1.1)), (2, two, "hook-smooth", 0.04, (0.6, 1.1)), (2, two, "hook-short", 0.03, (0.6, 1.1)),
    (2, two, "curl-sharp", 0.04, (0.6, 1.1)), (2, two, "curl-loop", 0.04, (0.6, 1.1)), (2, two, "z", 0.09, (0.6, 1.1)),
    (3, three, "cusp", 0.35, (0.5, 0.9)), (3, three, "smooth", 0.30, (0.5, 0.9)), (3, three, "flat", 0.25, (0.5, 0.9)), (3, three, "loopmid", 0.10, (0.5, 0.9)),
    (4, four, "open", 0.72, (0.6, 1.0)), (4, four, "round", 0.21, (0.7, 1.05)), (4, four, "closed", 0.07, (0.6, 1.0)),
    (5, five, "corner", 0.32, (0.55, 1.0)), (5, five, "roundc", 0.22, (0.55, 1.0)), (5, five, "apart", 0.15, (0.55, 1.0)), (5, five, "s", 0.23, (0.55, 1.0)), (5, five, "shortbar", 0.08, (0.55, 1.0)),
    (6, six, "curved", 0.47, (0.5, 0.85)), (6, six, "straight", 0.27, (0.5, 0.85)), (6, six, "open", 0.08, (0.5, 0.85)), (6, six, "spiral", 0.15, (0.5, 0.85)), (6, six, "bigloop", 0.03, (0.5, 0.85)),
    (7, seven, "plain", 0.40, (0.55, 1.0)), (7, seven, "round", 0.22, (0.55, 1.0)), (7, seven, "tick", 0.14, (0.55, 1.0)), (7, seven, "cross", 0.12, (0.55, 1.0)), (7, seven, "tickcross", 0.05, (0.55, 1.0)), (7, seven, "roundcross", 0.07, (0.55, 1.0)),
    (8, eight, "closed", 0.26, (0.55, 0.95)), (8, eight, "s", 0.22, (0.55, 0.95)), (8, eight, "tails", 0.18, (0.55, 0.95)), (8, eight, "opentop", 0.10, (0.55, 0.95)), (8, eight, "thinlow", 0.14, (0.55, 0.95)), (8, eight, "thintop", 0.10, (0.55, 0.95)),
    (9, nine, "straight", 0.44, (0.55, 0.95)), (9, nine, "hook", 0.26, (0.55, 0.95)), (9, nine, "gap", 0.20, (0.55, 0.95)), (9, nine, "small", 0.10, (0.55, 0.95)),
]

# ====================================================================================================
# net
# ====================================================================================================
# ---- normalising front-end.  Each stage holds transformed copies of its input (one linear-interpolation matrix per
#      candidate value) and keeps the copy whose bin contains the measured value.  The measurements are ratios of
#      image moments about the frame centre c (the ink's centre of mass is already there):
#        offset d  = sum I (x-c) / sum I                    (centre of mass relative to the frame centre; likewise for y)
#        slant  a  = sum I (x-c)(y-c) / sum I (y-c)^2      (regression slope of x on y)
#        width  sx = sqrt(sum I (x-c)^2 / sum I)            height sy likewise
#      "ratio > bound" is linear in the image, so a pair of ReLUs can veto every copy outside its bin.
CEN = 13.5
CENTRE = (9, -1.0, 0.25)                      # re-centring: copies shifted by -1 ... +1 pixel in steps of a quarter pixel (once along x, once along y)
VETO = 50.0                                   # a copy outside its bin is pushed below zero by VETO * (moment excess)
SLANT = ((6, -1.0, 0.3), (7, -0.15, 0.05))    # two shear stages (copies, first candidate slant, step); right-leaning is negative (y points down)
SL_LO, SL_HI = -1.15, 0.65                    # total range of slant the two stages can remove
# Targets from the geometry of an outline with centre-line half-axes (a, b): its ink spread is about 0.65 a (between
# a filled shape, a / sqrt(3), and a ring, a / sqrt(2)).  Height: the dataset's 20-pixel box minus the pen width gives
# b ~ 8.5 to 9, spread ~ 5.8.  Width: a canonical half-width of 6.5 pixels (digit about 17 pixels wide), spread 4.2;
# narrow digits (half-width 3.5 to 6) then need a stretch of 1.1 to 1.9, a bare stroke stops at the largest stretch.
SX_T, SX_N, SX_LO, SX_HI = 4.2, 12, 0.9, 2.0   # width: target spread (pixels), copies, smallest and largest stretch
SY_T, SY_N, SY_LO, SY_HI = 5.8, 8, 0.8, 1.3   # height: the same
# ---- features
NDIR = 8            # edge directions, 45 degrees apart (signed: which side the ink is on)
SIG_D = 0.8         # scale of the derivative-of-Gaussian edge filter, pixels
SIG_P = (1.0, 2.5)  # the pooling filter is the mean of a fine and a coarse Gaussian (pixels)
GAIN = 20.0         # brings pooled features to order 1 so the 1e-5 in `norm` is negligible
KD, KP, GRID = 7, 14, 14
# ---- matching in parts: the central 12 x 12 cells of a template are cut into 3 x 3 parts of 4 x 4 cells (8 x 8 pixels);
#      each part is tried at its home position and one cell (2 pixels) left, right, up and down.
PART, NPART = 4, 3
SHIFTS = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))
TDIM = NDIR * (PART * NPART) ** 2
GROUP = 100         # templates are read out in groups of this size, so classes may have different numbers of templates

def grids():
    y, x = np.meshgrid(np.arange(28.0) - CEN, np.arange(28.0) - CEN, indexing="ij")
    return x.reshape(-1), y.reshape(-1)

def interp_matrix(src_x, src_y):
    """Linear map (784 x 784): out[y, x] = in[src_y(y, x), src_x(y, x)]; one of the two is the identity, the other is
    interpolated linearly."""
    M = np.zeros((784, 784))
    for y in range(28):
        for x in range(28):
            sx, sy = src_x(y, x), src_y(y, x)
            if sx != x:
                x0 = int(np.floor(sx)); w = sx - x0
                for xx, ww in ((x0, 1 - w), (x0 + 1, w)):
                    if 0 <= xx < 28: M[y * 28 + x, y * 28 + xx] += ww
            elif sy != y:
                y0 = int(np.floor(sy)); w = sy - y0
                for yy, ww in ((y0, 1 - w), (y0 + 1, w)):
                    if 0 <= yy < 28: M[y * 28 + x, yy * 28 + x] += ww
            else:
                M[y * 28 + x, y * 28 + x] = 1
    return M

def scale_steps(n, lo, hi):
    g = (hi / lo) ** (1.0 / n)                       # n bins of equal ratio between lo and hi
    return lo * g ** (np.arange(n) + 0.5), g         # bin centres, ratio

def stages():
    """List of (name, tests (2n, 784), matrices (n*784, 784)).  tests[2j] > 0: the value is above bin j; tests[2j+1] > 0: below."""
    gx, gy = grids(); one = np.ones(784)
    out = []
    n, d0, dd = CENTRE
    for name, g1 in (("centre_x", gx), ("centre_y", gy)):
        T = np.zeros((2 * n, 784))
        for j in range(n):
            d = d0 + j * dd
            if j < n - 1: T[2 * j] = g1 - (d + dd / 2) * one
            if j > 0: T[2 * j + 1] = (d - dd / 2) * one - g1
        if name == "centre_x":
            M = np.concatenate([interp_matrix(lambda y, x, d=d0 + j * dd: x + d, lambda y, x: y) for j in range(n)])
        else:
            M = np.concatenate([interp_matrix(lambda y, x: x, lambda y, x, d=d0 + j * dd: y + d) for j in range(n)])
        out.append((name, T, M))
    for i, (n, a0, da) in enumerate(SLANT):
        T = np.zeros((2 * n, 784))
        for j in range(n):
            a = a0 + j * da
            if j < n - 1: T[2 * j] = gx * gy - (a + da / 2) * gy * gy
            if j > 0: T[2 * j + 1] = (a - da / 2) * gy * gy - gx * gy
        M = np.concatenate([interp_matrix(lambda y, x, a=a0 + j * da: x + a * (y - CEN), lambda y, x: y) for j in range(n)])
        out.append(("slant%d" % (i + 1), T, M))
    for name, (tgt, n, lo, hi), g2 in (("width", (SX_T, SX_N, SX_LO, SX_HI), gx * gx), ("height", (SY_T, SY_N, SY_LO, SY_HI), gy * gy)):
        s, g = scale_steps(n, lo, hi)
        T = np.zeros((2 * n, 784))
        for j in range(n):                            # copy j stretches by s[j]; it is right when the spread is tgt / s[j]
            if j > 0: T[2 * j] = g2 - (tgt * np.sqrt(g) / s[j]) ** 2 * one          # spread too large for this copy
            if j < n - 1: T[2 * j + 1] = (tgt / (s[j] * np.sqrt(g))) ** 2 * one - g2  # spread too small for this copy
        if name == "width":
            M = np.concatenate([interp_matrix(lambda y, x, q=q: CEN + (x - CEN) / q, lambda y, x: y) for q in s])
        else:
            M = np.concatenate([interp_matrix(lambda y, x: x, lambda y, x, q=q: CEN + (y - CEN) / q) for q in s])
        out.append((name, T, M))
    return out

def resample(img, src, axis):
    """img (n, 28, 28); src (n, 28, 28) source coordinate along `axis` (2: along rows, 1: along columns), linear interpolation, blank outside."""
    x0 = np.floor(src); w = src - x0; x0 = x0.astype(int)
    padw = [(0, 0), (0, 0), (0, 0)]; padw[axis] = (1, 1)
    pad = np.pad(img, padw)
    g = lambda ix: np.take_along_axis(pad, np.clip(ix + 1, 0, 29), axis)
    return (1 - w) * g(x0) + w * g(x0 + 1)

def normalise(img, split):
    """What the front-end does, without the binning: centre the ink, remove the slant, then bring width and height to the target spreads.
    Like the network it shears twice (`split` is the part of the slant left to the second shear), so a template is
    interpolated as often as a test image.  Continuous in the image, so a template depends smoothly on its own drawing."""
    gx, gy = grids()
    img = img.astype(np.float64); n = len(img)
    xs = np.arange(28.0)[None, None, :] + np.zeros((n, 28, 1)); ys = np.arange(28.0)[None, :, None] + np.zeros((n, 1, 28))
    lim = -CENTRE[1]
    f = img.reshape(n, -1)
    img = resample(img, xs + np.clip((f @ gx) / (f.sum(1) + 1e-9), -lim, lim)[:, None, None], 2)
    f = img.reshape(n, -1)
    img = resample(img, ys + np.clip((f @ gy) / (f.sum(1) + 1e-9), -lim, lim)[:, None, None], 1)
    f = img.reshape(n, -1)
    a = np.clip((f @ (gx * gy)) / (f @ (gy * gy) + 1e-9), SL_LO, SL_HI)
    img = resample(img, xs + (a - split)[:, None, None] * (ys - CEN), 2)
    img = resample(img, xs + split[:, None, None] * (ys - CEN), 2)
    sx_c, gxr = scale_steps(SX_N, SX_LO, SX_HI); sy_c, gyr = scale_steps(SY_N, SY_LO, SY_HI)
    f = img.reshape(n, -1)
    s = np.clip(SX_T / np.sqrt((f @ (gx * gx)) / (f.sum(1) + 1e-9) + 1e-9), sx_c[0], sx_c[-1])
    img = resample(img, CEN + (xs - CEN) / s[:, None, None], 2)
    f = img.reshape(n, -1)
    s = np.clip(SY_T / np.sqrt((f @ (gy * gy)) / (f.sum(1) + 1e-9) + 1e-9), sy_c[0], sy_c[-1])
    return resample(img, CEN + (ys - CEN) / s[:, None, None], 1)

def taps():
    o = np.arange(KD) - KD // 2
    g = np.exp(-o ** 2 / (2 * SIG_D ** 2)); g /= g.sum()
    d = o * g / SIG_D ** 2
    q = np.arange(KP) - (KP - 1) / 2
    p = sum(np.exp(-q ** 2 / (2 * s ** 2)) / np.exp(-q ** 2 / (2 * s ** 2)).sum() for s in SIG_P) / len(SIG_P)
    return g, d, p

def conv_weights():
    g, d, p = taps()
    gx = np.outer(g, d); gy = np.outer(d, g)                 # d/dx and d/dy of the smoothed image
    th = np.arange(NDIR) * 2 * np.pi / NDIR
    w1 = (np.cos(th)[:, None, None] * gx + np.sin(th)[:, None, None] * gy)[:, None]      # (8, 1, 7, 7)
    w2 = np.zeros((NDIR, NDIR, 1, 1))                         # keep a direction only where it beats its two perpendicular ones
    for k in range(NDIR):
        w2[k, k] = 1; w2[k, (k + 2) % NDIR] = -1; w2[k, (k - 2) % NDIR] = -1
    w3 = np.zeros((NDIR, NDIR, KP, KP))
    for k in range(NDIR): w3[k, k] = GAIN * np.outer(p, p)
    return w1, w2, w3

def pooled_map(img):
    """Normalised image (n, 28, 28) -> pooled edge maps (n, 8, 14, 14); the same arithmetic as the network's layers."""
    g, d, p = taps()
    img = img.astype(np.float64)
    S = np.zeros((28, 28)); D = np.zeros((28, 28)); A = np.zeros((GRID, 28))
    for i in range(28):
        for j in range(max(0, i - KD // 2), min(28, i + KD // 2 + 1)):
            S[i, j] = g[j - i + KD // 2]; D[i, j] = d[j - i + KD // 2]
    for i in range(GRID):
        for t in range(KP):
            j = 2 * i - (KP - 2) // 2 + t
            if 0 <= j < 28: A[i, j] = p[t]
    ix = S @ img @ D.T; iy = D @ img @ S.T
    th = np.arange(NDIR) * 2 * np.pi / NDIR
    r = np.maximum(np.cos(th)[None, :, None, None] * ix[:, None] + np.sin(th)[None, :, None, None] * iy[:, None], 0)
    s = np.maximum(r - np.roll(r, 2, 1) - np.roll(r, -2, 1), 0)
    return GAIN * (A @ s @ A.T)

def zscore(f):
    f = f.reshape(len(f), -1); mu = f.mean(1, keepdims=True)
    return (f - mu) / np.sqrt(((f - mu) ** 2).mean(1, keepdims=True) + 1e-5)

def template(img, split):
    """One generated digit -> its template: the central 12 x 12 cells of its own pooled edge maps, normalised by itself alone."""
    m = (GRID - PART * NPART) // 2
    f = pooled_map(normalise(img, split))[:, :, m:GRID - m, m:GRID - m]
    return zscore(f).reshape(len(img), NDIR, PART * NPART, PART * NPART)

def crop_weights(a, b):
    """Selection matrix: from the normalised map (8 * 14 * 14) to part (a, b)'s windows, one per shift, laid side by side: (8, 4, 4 * shifts)."""
    m = (GRID - PART * NPART) // 2
    W = np.zeros((NDIR, PART, PART * len(SHIFTS), NDIR, GRID, GRID))
    for c in range(NDIR):
        for s, (dy, dx) in enumerate(SHIFTS):
            for i in range(PART):
                for j in range(PART):
                    W[c, i, s * PART + j, c, m + a * PART + i + dy, m + b * PART + j + dx] = 1
    return W.reshape(NDIR * PART * PART * len(SHIFTS), NDIR * GRID * GRID)

def layers(per_class):
    N = int(np.sum(per_class)); R = N // GROUP; rmax = max(per_class) // GROUP
    L, src = [], "image"
    for name, T, M in stages():
        n = len(T) // 2
        L += [
            {"name": name + "_test", "op": "dense", "inputs": [src], "out": 2 * n},
            {"name": name + "_out", "op": "relu", "inputs": [name + "_test"]},
            {"name": name + "_veto", "op": "dense", "inputs": [name + "_out"], "out": n * 784},
            {"name": name + "_copies", "op": "dense", "inputs": [src], "out": n * 784},
            {"name": name + "_sum", "op": "add", "inputs": [name + "_copies", name + "_veto"]},
            {"name": name + "_kept", "op": "relu", "inputs": [name + "_sum"]},
            {"name": name + "_stack", "op": "reshape", "inputs": [name + "_kept"], "shape": [n, 28, 28]},
            {"name": name + "_done", "op": "conv", "inputs": [name + "_stack"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0}]
        src = name + "_done"
    L += [
        {"name": "edges", "op": "conv", "inputs": [src], "out": NDIR, "kernel": [KD, KD], "stride": 1, "padding": KD // 2},
        {"name": "edges_pos", "op": "relu", "inputs": ["edges"]},
        {"name": "tuned", "op": "conv", "inputs": ["edges_pos"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "tuned_pos", "op": "relu", "inputs": ["tuned"]},
        {"name": "pooled", "op": "conv", "inputs": ["tuned_pos"], "out": NDIR, "kernel": [KP, KP], "stride": 2, "padding": (KP - 2) // 2},
        {"name": "flat", "op": "reshape", "inputs": ["pooled"], "shape": [1, NDIR * GRID, GRID]},
        {"name": "normed", "op": "norm", "inputs": ["flat"]}]
    parts = []
    for a in range(NPART):
        for b in range(NPART):
            p = "p%d%d" % (a, b)
            L += [
                {"name": p + "_win", "op": "dense", "inputs": ["normed"], "out": NDIR * PART * PART * len(SHIFTS)},
                {"name": p + "_img", "op": "reshape", "inputs": [p + "_win"], "shape": [NDIR, PART, PART * len(SHIFTS)]},
                {"name": p + "_match", "op": "conv", "inputs": [p + "_img"], "out": N, "kernel": [PART, PART], "stride": PART, "padding": 0},
                {"name": p + "_best", "op": "maxpool", "inputs": [p + "_match"], "size": [1, len(SHIFTS)], "stride": [1, len(SHIFTS)]}]
            parts.append(p + "_best")
    L += [
        {"name": "match", "op": "add", "inputs": parts},
        {"name": "groups", "op": "reshape", "inputs": ["match"], "shape": [1, R, GROUP]},
        {"name": "group_best", "op": "maxpool", "inputs": ["groups"], "size": [1, GROUP], "stride": [1, GROUP]},
        {"name": "sorted", "op": "dense", "inputs": ["group_best"], "out": 10 * rmax},
        {"name": "byclass", "op": "reshape", "inputs": ["sorted"], "shape": [1, 10, rmax]},
        {"name": "best", "op": "maxpool", "inputs": ["byclass"], "size": [1, rmax], "stride": [1, rmax]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]}]
    return {"layers": L}

def write(out_dir, T, per_class):
    """T (N, 8, 12, 12): templates, class by class; per_class[c] of them belong to class c."""
    w1, w2, w3 = conv_weights()
    W = {"edges.W": w1, "edges.b": np.zeros(NDIR), "tuned.W": w2, "tuned.b": np.zeros(NDIR), "pooled.W": w3, "pooled.b": np.zeros(NDIR)}
    for name, Tst, M in stages():
        n = len(Tst) // 2
        veto = np.zeros((n * 784, 2 * n))
        for j in range(n): veto[j * 784:(j + 1) * 784, 2 * j:2 * j + 2] = -VETO
        W.update({name + "_test.W": Tst, name + "_test.b": np.zeros(2 * n), name + "_veto.W": veto, name + "_veto.b": np.zeros(n * 784),
                  name + "_copies.W": M, name + "_copies.b": np.zeros(n * 784), name + "_done.W": np.ones((1, n, 1, 1)), name + "_done.b": np.zeros(1)})
    for a in range(NPART):
        for b in range(NPART):
            p = "p%d%d" % (a, b)
            W[p + "_win.W"] = crop_weights(a, b); W[p + "_win.b"] = np.zeros(NDIR * PART * PART * len(SHIFTS))
            W[p + "_match.W"] = T[:, :, a * PART:(a + 1) * PART, b * PART:(b + 1) * PART] / TDIM; W[p + "_match.b"] = np.zeros(len(T))
    # read-out: the best of each group of templates goes to its class's row; unused places in a row get a large negative bias
    R = len(T) // GROUP; rmax = max(per_class) // GROUP
    sw = np.zeros((10 * rmax, R)); sb = np.full(10 * rmax, -100.0); r = 0
    for c in range(10):
        for i in range(per_class[c] // GROUP):
            sw[c * rmax + i, r] = 1; sb[c * rmax + i] = 0; r += 1
    W["sorted.W"] = sw; W["sorted.b"] = sb
    json.dump(layers(per_class), open(f"{out_dir}/network.json", "w"), indent=1)
    np.savez(f"{out_dir}/weights.npz", **{k: np.asarray(v, np.float32) for k, v in W.items()})

# ====================================================================================================
# bank
# ====================================================================================================
def make_images(per_class, seed, chunk=250):
    """per_class[c] generated digits of class c, class by class.  Returns images (N, 28, 28) float32 and, for each
    image, the share of its slant left to the second shear (see normalise)."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(10):
        rows = [b for b in BANK if b[0] == c]
        share = np.array([b[3] for b in rows]); cnt = np.floor(share / share.sum() * per_class[c]).astype(int); cnt[0] += per_class[c] - cnt.sum()
        for (cls, fn, v, sh, asp), m in zip(rows, cnt):
            done = 0
            while done < m:
                B = min(chunk, m - done)
                st = warp(fn(rng, B, v), rng, asp)
                out.append(rasterise(st, (thickness(rng, B, hi=7.0, mode=3.0) if cls == 1 else thickness(rng, B)), rng, thvar=0.15).reshape(B, 28, 28).astype(np.float32)); done += B
    return np.concatenate(out), rng.uniform(SLANT[1][1], -SLANT[1][1], int(np.sum(per_class)))

# ====================================================================================================
# main
# ====================================================================================================
if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    images, split = make_images(PER_CLASS, SEED)             # (N, 28, 28) generated digits, class by class
    T = np.concatenate([template(images[i:i + 2000], split[i:i + 2000]) for i in range(0, len(images), 2000)]).astype(np.float32)
    write(out_dir, T, PER_CLASS)
    print("wrote", out_dir, "templates", T.shape)
