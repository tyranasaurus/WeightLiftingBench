"""Builds the digit network: python3 build.py OUT_DIR

Nothing is read from disk.  Every weight comes from this file:
  * the front end is made of formula filters.
      - edge branch: Gaussian-derivative edge filters in 8 directions, a direction-sharpening 1x1 layer,
        Gaussian pooling on a 7 x 7 grid and a piecewise-linear square-root curve;
      - bay branch: for every background pixel, is there ink above / below / left / right of it (sums of ink
        along the four rays), combined by hand-written rules into nine maps (enclosed hole; bay open to one
        side, four maps; corner blocked on two adjacent sides, four maps), Gaussian pooling on a 5 x 5 grid.
  * the classifier is a bank of templates.  Each template is one drawing of a digit, made here from pen
    paths typed from general knowledge of how digits are written (loops, bars, stems, hooks), rendered as a
    stroke of some thickness, slant and proportion, pushed through the front end and divided by its own
    length.  A template is never combined with any other template.
The class score is the largest template match within the class.
"""
import sys, os, json
import numpy as np

SEED = 20240611
POOL_A, POOL_B = 100, 190          # the best match is taken in two steps
TC = POOL_A * POOL_B               # templates per class
GRID = 28
NDIR = 8
SIG_G = 1.0           # edge filter scale (pixels)
KG = 7                # edge filter size
SIG_P = 1.8           # pooling scale (pixels) = 0.45 x the spacing of the sample points
KP = 12               # pooling filter size (even: sample points fall between pixels)
STRIDE = 4
PADP = 4
NPOOL = 7             # 7 x 7 sample points at 1.5, 5.5, ..., 25.5
# Largest pooled value of one straight full-contrast edge through a sample point: the edge plane integrates to 1
# across the edge, so the pooled value is the peak of a 1-D Gaussian of scale SIG_P.
MPEAK = 1.0 / (np.sqrt(2 * np.pi) * SIG_P)
# bay branch (works on the image at half resolution, 14 x 14)
INK_T = 0.3           # grey level above which a pixel counts as ink
HALF = 14
TAU = 0.35            # ray sum that counts as fully blocked (a solid stroke one pixel wide)
INK_VETO = 6.0        # ink under the pixel itself switches the bay maps off
NBAY = 9
KC, SC, PC = 8, 3, 3  # pooling of the bay maps: 5 x 5 sample points, 6 pixels apart
SIG_C = 0.45 * SC     # in half-resolution pixels
NC = 5
BAY_GAIN = 2.0        # weight of the bay maps against the edge maps (hand-chosen)
CORNER_GAIN = 0.5     # the four corner maps are less specific than holes and bays: half weight


# ----------------------------------------------------------------------------------------------------------
# front end (formula filters)
# ----------------------------------------------------------------------------------------------------------
def gauss(k, sig):
    a = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-a ** 2 / (2 * sig ** 2))
    return g / g.sum()


# rule table of the bay maps over (open above, open below, open left, open right), then bias
BAY_RULES = [
    ((-1, -1, -1, -1), 1.0),      # hole: blocked on all four sides
    ((+1, -1, -1, -1), 0.0),      # bay open upwards only
    ((-1, +1, -1, -1), 0.0),      # open downwards only
    ((-1, -1, +1, -1), 0.0),      # open to the left only
    ((-1, -1, -1, +1), 0.0),      # open to the right only
    ((-1, +1, +1, -1), -1.0),     # corner: blocked above and right, open below and left
    ((-1, +1, -1, +1), -1.0),     # blocked above and left
    ((+1, -1, +1, -1), -1.0),     # blocked below and right
    ((+1, -1, -1, +1), -1.0),     # blocked below and left
]


def frontend_weights():
    w = {}
    ax = np.arange(KG) - (KG - 1) / 2.0
    X, Y = np.meshgrid(ax, ax)                       # X: column offset, Y: row offset
    G = np.exp(-(X ** 2 + Y ** 2) / (2 * SIG_G ** 2))
    Gx = X * G; Gx = Gx / np.sum(X * Gx)             # response 1 to a unit ramp rising to the right
    Gy = Y * G; Gy = Gy / np.sum(Y * Gy)
    W1 = np.zeros((NDIR, 1, KG, KG))
    for k in range(NDIR):
        th = 2 * np.pi * k / NDIR
        W1[k, 0] = np.cos(th) * Gx + np.sin(th) * Gy
    w["edge.W"] = W1; w["edge.b"] = np.zeros(NDIR)
    # direction sharpening: keep what points along direction k beyond what points sideways (k +- 90 degrees).
    # This splits every gradient between its two neighbouring directions (parallelogram rule).
    W2 = np.zeros((NDIR, NDIR, 1, 1))
    for k in range(NDIR):
        W2[k, k, 0, 0] = 1.0
        W2[k, (k + 2) % NDIR, 0, 0] = -1.0
        W2[k, (k - 2) % NDIR, 0, 0] = -1.0
    w["dir.W"] = W2; w["dir.b"] = np.zeros(NDIR)
    # Gaussian pooling, stride 4
    g = gauss(KP, SIG_P)
    W3 = np.zeros((NDIR, NDIR, KP, KP))
    for k in range(NDIR):
        W3[k, k] = np.outer(g, g)
    w["pool.W"] = W3; w["pool.b"] = np.zeros(NDIR)
    # piecewise-linear square root with knots at MPEAK/16, MPEAK/4, MPEAK (chords of sqrt(x / MPEAK))
    knots = np.array([0.0, MPEAK / 16, MPEAK / 4, MPEAK])
    vals = np.array([0.0, 0.25, 0.5, 1.0])
    slopes = list((vals[1:] - vals[:-1]) / (knots[1:] - knots[:-1])) + [1.0 / (3 * MPEAK)]
    W4 = np.zeros((4 * NDIR, NDIR, 1, 1)); b4 = np.zeros(4 * NDIR)
    W5 = np.zeros((NDIR, 4 * NDIR, 1, 1))
    for k in range(NDIR):
        for j in range(4):
            W4[4 * k + j, k, 0, 0] = 1.0
            b4[4 * k + j] = -knots[j]
            W5[k, 4 * k + j, 0, 0] = slopes[j] - (slopes[j - 1] if j > 0 else 0.0)
    w["bend.W"] = W4; w["bend.b"] = b4
    w["root.W"] = W5; w["root.b"] = np.zeros(NDIR)
    # ---- bay branch
    w["solid.W"] = np.ones((1, 1, 1, 1)); w["solid.b"] = np.array([-INK_T])
    KR = 2 * HALF - 1; c = HALF - 1                   # rays reach the border of the 14 x 14 map
    Wr = np.zeros((5, 1, KR, KR)); br = np.zeros(5)
    Wr[0, 0, :c, c] = -1.0 / TAU                      # ink above    -> "open above" = relu(1 - sum / TAU)
    Wr[1, 0, c + 1:, c] = -1.0 / TAU                  # ink below
    Wr[2, 0, c, :c] = -1.0 / TAU                      # ink to the left
    Wr[3, 0, c, c + 1:] = -1.0 / TAU                  # ink to the right
    br[:4] = 1.0
    Wr[4, 0, c, c] = 1.0                              # ink under the pixel itself
    w["rays.W"] = Wr; w["rays.b"] = br
    Wb = np.zeros((NBAY, 5, 1, 1)); bb = np.zeros(NBAY)
    for i, (rule, bias) in enumerate(BAY_RULES):
        Wb[i, :4, 0, 0] = rule; Wb[i, 4, 0, 0] = -INK_VETO; bb[i] = bias
    w["bays.W"] = Wb; w["bays.b"] = bb
    gc = gauss(KC, SIG_C)
    Wc = np.zeros((NBAY, NBAY, KC, KC))
    for i in range(NBAY):
        Wc[i, i] = np.outer(gc, gc) * BAY_GAIN * (CORNER_GAIN if i >= 5 else 1.0)
    w["baypool.W"] = Wc; w["baypool.b"] = np.zeros(NBAY)
    return w


def conv(x, W, b, stride, padding):
    n, c, h, ww = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (ww + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def pool_matrix(nout, k, stride, pad, size, g):
    A = np.zeros((nout, size + 2 * pad))
    for i in range(nout):
        A[i, i * stride:i * stride + k] = g
    return A[:, pad:pad + size]


def edge_features(img, w):
    """img (n, 28, 28) -> (n, 392): the same arithmetic as the network's edge branch."""
    x = img[:, None]
    x = np.maximum(conv(x, w["edge.W"], w["edge.b"], 1, (KG - 1) // 2), 0)
    x = np.maximum(conv(x, w["dir.W"], w["dir.b"], 1, 0), 0)
    # pooling: the filter is separable and the same for every direction, so do it as two small products
    A = pool_matrix(NPOOL, KP, STRIDE, PADP, GRID, gauss(KP, SIG_P))
    x = np.einsum("iy,ncyx->ncix", A, x, optimize=True)
    x = np.einsum("jx,ncix->ncij", A, x, optimize=True)
    x = np.maximum(conv(x, w["bend.W"], w["bend.b"], 1, 0), 0)
    x = conv(x, w["root.W"], w["root.b"], 1, 0)
    return x.reshape(len(x), -1)


def bay_features(img, w):
    """img (n, 28, 28) -> (n, 225): the same arithmetic as the network's bay branch."""
    n = len(img)
    s = np.maximum(img - INK_T, 0)
    m = s.reshape(n, HALF, 2, HALF, 2).mean(axis=(2, 4))
    cs = np.cumsum(m, axis=1)
    up = cs - m; down = cs[:, -1:, :] - cs
    cs = np.cumsum(m, axis=2)
    left = cs - m; right = cs[:, :, -1:] - cs
    q = np.stack([np.maximum(1 - r / TAU, 0) for r in (up, down, left, right)] + [m], axis=1)      # (n, 5, 14, 14)
    maps = np.maximum(np.einsum("oc,ncyx->noyx", w["bays.W"][:, :, 0, 0], q, optimize=True) + w["bays.b"][None, :, None, None], 0)
    A = pool_matrix(NC, KC, SC, PC, HALF, gauss(KC, SIG_C))
    x = np.einsum("iy,ncyx->ncix", A, maps, optimize=True)
    x = np.einsum("jx,ncix->ncij", A, x, optimize=True)
    gain = np.array([BAY_GAIN * (CORNER_GAIN if i >= 5 else 1.0) for i in range(NBAY)])
    x = x * gain[None, :, None, None]
    return x.reshape(n, -1)


# ----------------------------------------------------------------------------------------------------------
# drawing: pen paths -> stroke images
# ----------------------------------------------------------------------------------------------------------
class Rand:
    """Random numbers for a batch of drawings (fixed seed)."""
    def __init__(self, rng, n):
        self.rng, self.n = rng, n

    def u(self, a, b):
        return a + (b - a) * self.rng.random(self.n)

    def n_(self, m, s, lo=None, hi=None):
        v = m + s * self.rng.standard_normal(self.n)
        return v if lo is None else np.clip(v, lo, hi)

    def coin(self, p):
        return (self.rng.random(self.n) < p).astype(float)


def P(R, *pts):
    """Stack control points [(x, y), ...] (numbers or per-drawing arrays) into (n, K, 2)."""
    out = np.zeros((R.n, len(pts), 2))
    for i, (x, y) in enumerate(pts):
        out[:, i, 0] = x; out[:, i, 1] = y
    return out


def spline(Q, nsub, closed=False):
    """Smooth curve through control points Q (n, K, 2): centripetal Catmull-Rom.  Returns (n, M, 2)."""
    if closed:
        E = np.concatenate([Q[:, -1:], Q, Q[:, :2]], axis=1)
    else:
        E = np.concatenate([2 * Q[:, :1] - Q[:, 1:2], Q, 2 * Q[:, -1:] - Q[:, -2:-1]], axis=1)
    D = E[:, 1:] - E[:, :-1]
    d = np.sqrt(np.sqrt((D ** 2).sum(-1)) + 1e-9)[..., None]          # chord length ^ 0.5
    m = (D[:, 1:] / d[:, 1:] * d[:, :-1] + D[:, :-1] / d[:, :-1] * d[:, 1:]) / (d[:, :-1] + d[:, 1:])   # tangent at E[1..-2]
    P0, P1 = E[:, 1:-2], E[:, 2:-1]
    h = d[:, 1:-1]
    m0, m1 = m[:, :-1] * h, m[:, 1:] * h
    s = (np.arange(nsub) / nsub)[None, None, :, None]
    P0, P1, m0, m1 = P0[:, :, None], P1[:, :, None], m0[:, :, None], m1[:, :, None]
    C = (2 * s ** 3 - 3 * s ** 2 + 1) * P0 + (s ** 3 - 2 * s ** 2 + s) * m0 + (-2 * s ** 3 + 3 * s ** 2) * P1 + (s ** 3 - s ** 2) * m1
    C = C.reshape(Q.shape[0], -1, 2)
    last = E[:, 2:3] if closed else Q[:, -1:]
    return np.concatenate([C, last], axis=1)


_YY, _XX = np.meshgrid(np.arange(GRID, dtype=float), np.arange(GRID, dtype=float), indexing="ij")
_PX = _XX.reshape(1, 1, -1); _PY = _YY.reshape(1, 1, -1)
_c = np.arange(0.5, GRID, 2.0)                                              # coarse 14 x 14 grid: 0.5, 2.5, ..., 26.5
_QY, _QX = np.meshgrid(_c, _c, indexing="ij")
_QX = _QX.reshape(1, 1, -1); _QY = _QY.reshape(1, 1, -1)


def ink(seg, r, px, py, ramp):
    """seg (n, S, 4) pixel coordinates, r (n,) or (n, S) pen radius -> ink at the sample points (n, npts)."""
    x0, y0 = seg[:, :, 0:1], seg[:, :, 1:2]
    dx, dy = seg[:, :, 2:3] - x0, seg[:, :, 3:4] - y0
    L2 = dx * dx + dy * dy + 1e-12
    qx, qy = px - x0, py - y0
    t = np.clip((qx * dx + qy * dy) / L2, 0.0, 1.0)
    ex, ey = qx - t * dx, qy - t * dy
    d2 = ex * ex + ey * ey
    if r.ndim == 2:
        d = np.sqrt(d2) - r[:, :, None]
        dmin = d.min(axis=1)
        return np.clip(0.5 - dmin / ramp, 0.0, 1.0)
    d = np.sqrt(d2.min(axis=1))
    return np.clip(0.5 + (r[:, None] - d) / ramp, 0.0, 1.0)


def draw(R, pts, pieces, wid, shear, rmax=None, jit=0.022, thick=None, nsub=4):
    """pts: control points (n, K, 2); pieces: list of (list of point numbers, closed flag).  Returns (n, 28, 28).

    wid: width factor applied to x.  shear: slant (x shift per unit height, positive leans right).
    jit: size of the wobble of the control points (fraction of the digit height)."""
    n = R.n
    rng = R.rng
    pts = pts + jit * rng.standard_normal(pts.shape)
    polys = [spline(pts[:, idx], nsub, closed) for idx, closed in pieces]
    # segments (pairs of consecutive points inside each piece)
    A = np.concatenate([p[:, :-1] for p in polys], axis=1)
    Bp = np.concatenate([p[:, 1:] for p in polys], axis=1)
    S = A.shape[1]
    x = np.concatenate([A[:, :, 0], Bp[:, :, 0]], axis=1); y = np.concatenate([A[:, :, 1], Bp[:, :, 1]], axis=1)
    # centre and measure
    cx = 0.5 * (x.max(1) + x.min(1))[:, None]; cy = 0.5 * (y.max(1) + y.min(1))[:, None]
    hh = (y.max(1) - y.min(1))[:, None] + 1e-6; ww = (x.max(1) - x.min(1))[:, None] + 1e-6
    u = (x - cx) / ww; v = (y - cy) / hh                                   # in [-0.5, 0.5]
    # smooth changes of proportion (a hand never draws the same proportions twice)
    c1 = R.n_(0, 0.10, -0.25, 0.25)[:, None]    # top wider / bottom wider
    c2 = R.n_(0, 0.04, -0.10, 0.10)[:, None]    # bowed spine
    c3 = R.n_(0, 0.05, -0.12, 0.12)[:, None]    # middle pushed up / down
    c4 = R.n_(0, 0.05, -0.12, 0.12)[:, None]    # left taller / right taller
    c5 = R.n_(0, 0.04, -0.10, 0.10)[:, None]    # left half squeezed / right half squeezed
    x, y = (x - cx) * (1 + 2 * c1 * v) + ww * (c2 * (1 - 4 * v * v) + c5 * (1 - 4 * u * u)), (y - cy) * (1 + 2 * c4 * u) + hh * c3 * (1 - 4 * v * v)
    x = x * wid[:, None]
    x = x - shear[:, None] * y
    y = y - R.n_(0.06, 0.12, -0.20, 0.40)[:, None] * x                     # level strokes rise (or fall) to the right
    ang = R.n_(0, 0.07, -0.2, 0.2)[:, None]                                # small rotation
    ca, sa = np.cos(ang), np.sin(ang)
    x, y = ca * x - sa * y, sa * x + ca * y
    # pen radius in pixels: most strokes are 2-3 pixels wide, some hair-thin, some very bold
    if thick is None:
        thick = np.exp(R.n_(np.log(1.25), 0.33, np.log(0.65), np.log(3.2)))
    r = thick if rmax is None else np.minimum(thick, rmax)
    # fit the ink into a 20 x 20 box, keeping proportions
    bw = (x.max(1) - x.min(1)); bh = (y.max(1) - y.min(1))
    sc = (20.0 - 2 * r) / np.maximum(np.maximum(bw, bh), 1e-6)
    mx = 0.5 * (x.max(1) + x.min(1)); my = 0.5 * (y.max(1) + y.min(1))
    x = (x - mx[:, None]) * sc[:, None] + 13.5; y = (y - my[:, None]) * sc[:, None] + 13.5
    seg = np.stack([x[:, :S], y[:, :S], x[:, S:], y[:, S:]], axis=2)
    # centre of mass from a coarse drawing, then the final drawing with the centre of mass at the centre
    # of the frame (to within the half pixel that whole-pixel placement leaves)
    c = ink(seg, r, _QX, _QY, 2.0)
    tot = c.sum(1) + 1e-9
    mx = (c * _QX[0]).sum(1) / tot; my = (c * _QY[0]).sum(1) / tot
    ox = 13.5 - mx + R.u(-0.5, 0.5); oy = 13.5 - my + R.u(-0.5, 0.5)
    seg = seg + np.stack([ox, oy, ox, oy], axis=1)[:, None, :]
    img = ink(seg, r, _PX, _PY, 1.0)
    return img.reshape(n, GRID, GRID)


# ----------------------------------------------------------------------------------------------------------
# pen paths.  x runs to the right, y runs down, the digit is about 1 unit tall.  Every number below is typed
# from general knowledge of how each digit is written; the random ranges describe how hands differ.
# Each style returns: control points, pieces (point numbers, closed?), width factor, slant.
# ----------------------------------------------------------------------------------------------------------
def slant(R, m=0.18, s=0.27, lo=-0.45, hi=0.95):
    return R.n_(m, s, lo, hi)


def width(R, m=1.0, s=0.14):
    return m * np.exp(R.n_(0, s, -2.2 * s, 2.2 * s))


def ring(R, cx, cy, rx, ry, a0, sweep, K, egg=None, pe=None, spir=None):
    """K points on an oval, from angle a0 turning by -sweep (anticlockwise on the page, as most loops are written)."""
    k = (np.arange(K) / (K - 1.0))[None]
    th = a0[:, None] - sweep[:, None] * k
    rad = np.ones_like(th)
    if egg is not None:
        rad = rad * (1 + egg[:, None] * np.cos(th - pe[:, None]))
    if spir is not None:
        rad = rad * (1 + spir[:, None] * (2 * k - 1))
    z = np.zeros(R.n)
    x = (cx + z)[:, None] + (rx + z)[:, None] * rad * np.cos(th)
    y = (cy + z)[:, None] + (ry + z)[:, None] * rad * np.sin(th)
    return np.stack([x, y], axis=-1)


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def zero(R, kind=0):
    K = 12
    rx = R.u(0.19, 0.46)
    a0 = -np.pi / 2 + R.n_(0, 0.35, -0.8, 0.8)            # the pen starts near the top
    if kind == 2:                                         # the stroke runs on and ends in a curl inside the oval
        sweep = 2 * np.pi + R.u(0.5, 1.3); spir = -R.u(0.08, 0.22)
    else:
        sweep = 2 * np.pi + R.n_(0.05, 0.28, -0.45, 0.85)  # a gap, or an overlap, where the stroke ends
        spir = R.n_(0, 0.045, -0.11, 0.11)
    pts = ring(R, 0, 0, rx, 0.5, a0, sweep, K, egg=R.u(0, 0.13), pe=R.u(0, 2 * np.pi), spir=spir)
    if kind == 1:                                         # a tail: the pen lands outside, up and to the right, and runs into the oval
        phi = -R.u(0.2, 1.4); t = R.u(0.10, 0.32)
        pre = pts[:, 0] + np.stack([t * np.cos(phi), t * np.sin(phi)], axis=-1)
        pts = np.concatenate([pre[:, None], pts], axis=1)
    return pts, [(list(range(pts.shape[1])), False)], width(R, 1.0, 0.08), slant(R)


def one_plain(R):
    bow = R.n_(0, 0.03, -0.08, 0.08); b2 = R.n_(0, 0.012, -0.03, 0.03)
    pts = P(R, (0, 0), (bow + b2, 0.33), (bow - b2, 0.67), (0, 1))
    return pts, [([0, 1, 2, 3], False)], width(R), R.n_(0.28, 0.36, -0.55, 1.25)


def one_flag(R):
    fx = R.u(0.08, 0.28); fy = np.minimum(fx * R.u(0.8, 1.7), 0.40)      # a short flag pointing down-left
    bow = R.n_(0, 0.02, -0.05, 0.05)
    pts = P(R, (-fx, fy), (0, 0), (bow, 0.5), (0, 1))
    return pts, [([0, 1], False), ([1, 2, 3], False)], width(R), R.n_(0.12, 0.20, -0.3, 0.55)


def one_base(R, flag=True):
    fx = R.u(0.08, 0.28); fy = np.minimum(fx * R.u(0.8, 1.7), 0.40)
    bl = R.u(0.18, 0.42); br = bl * R.u(0.7, 1.4)                         # the base reaches out on both sides of the stem
    bow = R.n_(0, 0.02, -0.05, 0.05)
    pts = P(R, (-fx, fy), (0, 0), (bow, 0.5), (0, 1), (-bl, 1.0), (br, 1.0))
    pieces = [([1, 2, 3], False), ([4, 3, 5], False)]
    if flag:
        pieces.append(([0, 1], False))
    return pts, pieces, width(R), R.n_(0.05, 0.16, -0.3, 0.5)


def two(R, loop=False, top=0):
    by = R.u(0.84, 1.0)                                   # height of the lower-left corner (or loop)
    tw = R.u(0.50, 0.90); ty = R.u(0.18, 0.38)            # right-most point of the top arc (small: a narrow 2)
    f = tw / 0.78
    hx = R.u(0.04, 0.20) * f; hy = R.u(0.10, 0.45)        # where the hook starts
    ex = R.u(0.38, 1.10); ey = np.minimum(by + R.u(-0.16, 0.16), 1.02)   # end of the base stroke
    db = R.n_(0, 0.045, -0.10, 0.10)                      # bow of the diagonal
    A = (tw, ty)
    C = (0.12 if loop else 0.06, by)
    d1 = lerp(A, C, 0.38); d2 = lerp(A, C, 0.72)
    d1 = (d1[0] + db + 0.07 * f, d1[1]); d2 = (d2[0] + db + 0.05 * f, d2[1])
    if top == 1:      # Z-like: a straight top bar and a sharp corner
        A = (tw, R.u(0.0, 0.08))
        d1 = lerp(A, C, 0.38); d2 = lerp(A, C, 0.72)
        d1 = (d1[0] + db, d1[1]); d2 = (d2[0] + db, d2[1])
        head = [(R.u(0.0, 0.18) * f, R.u(0.0, 0.12)), A, d1, d2]
        head_pieces = [[0, 1], [1, 2, 3]]
    elif top == 2:    # the stroke starts with a small curl inside the arc
        head = [(hx + R.u(0.10, 0.20) * f, hy + R.u(0.0, 0.12)), (hx, hy), (0.26 * f, 0.06), (0.50 * f, 0.00), (tw - 0.07 * f, 0.08), A, d1, d2]
        head_pieces = [[0, 1, 2, 3, 4, 5, 6, 7]]
    else:
        head = [(hx, hy), (0.26 * f, 0.06), (0.50 * f, 0.00), (tw - 0.07 * f, 0.08), A, d1, d2]
        head_pieces = [[0, 1, 2, 3, 4, 5, 6]]
    n0 = len(head)
    if not loop:
        mid = (0.5 * (C[0] + ex), 0.5 * (by + ey) + R.u(-0.07, 0.07))
        pts = P(R, *head, C, mid, (ex, ey))
        head_pieces[-1] = head_pieces[-1] + [n0]
        pieces = [(p, False) for p in head_pieces] + [([n0, n0 + 1, n0 + 2], False)]
        return pts, pieces, width(R, 0.88, 0.22), slant(R)
    ls = R.u(0.4, 2.2)                                    # loop size
    mid = (0.5 * (C[0] + 0.18 * ls + ex), 0.5 * (by + ey) + R.u(-0.07, 0.07))
    pts = P(R, *head, C, (C[0] - 0.10 * ls, by - 0.08 * ls), (C[0] + 0.01 * ls, by - 0.19 * ls), (C[0] + 0.18 * ls, by - 0.09 * ls), mid, (ex, ey))
    head_pieces[-1] = head_pieces[-1] + [n0, n0 + 1, n0 + 2, n0 + 3, n0 + 4, n0 + 5]
    return pts, [(p, False) for p in head_pieces], width(R, 0.88, 0.22), slant(R)


def three(R, flat=False, smooth=False):
    wx = R.u(0.15, 0.50); wy = R.u(0.30, 0.56)            # the waist: how far left, how high
    uw = R.u(0.60, 0.82); lw = uw + R.u(-0.22, 0.16)      # right edges of the upper and lower bowls
    tx = R.u(-0.12, 0.22); ty = R.u(0.70, 0.97)           # tail end
    low = [(wx + 0.22, wy + 0.04), (lw, 0.5 * (wy + 1.0) - 0.03), (lw - 0.08, 0.90), (0.42, 1.0), (tx, ty)]
    wid = width(R, 0.95, 0.18); sl = slant(R, 0.15, 0.23)
    if flat:
        pts = P(R, (R.u(-0.05, 0.12), 0.02), (0.45, 0.01 + R.n_(0, 0.02, -0.04, 0.04)), (uw + 0.03, 0.02), (0.5 * (uw + wx) + 0.06, 0.5 * wy), (wx, wy), *low)
        return pts, [([0, 1, 2], False), ([2, 3, 4], False), ([4, 5, 6, 7, 8, 9], False)], wid, sl
    pts = P(R, (R.u(-0.05, 0.20), R.u(0.04, 0.25)), (0.36, 0.01), (uw - 0.08, 0.06), (uw, 0.5 * wy), (wx + 0.20, wy - 0.07), (wx, wy), *low)
    if smooth:
        return pts, [(list(range(11)), False)], wid, sl
    return pts, [([0, 1, 2, 3, 4, 5], False), ([5, 6, 7, 8, 9, 10], False)], wid, sl


def four_open(R, rounded=False):
    sx = R.u(0.58, 0.80)                                  # stem
    lx = np.maximum(sx - R.u(0.15, 0.62), 0.02); ly = R.u(0.0, 0.12)   # top of the left stroke: the top stays open
    by = R.u(0.50, 0.68); bx = np.minimum(R.u(0.02, 0.12), lx)         # the corner where the left stroke meets the bar
    be = sx + R.u(0.10, 0.34); bey = by + R.u(-0.12, 0.04)  # right end of the bar: it always sticks out past the stem
    st = R.u(-0.08, 0.30); sb = sx + R.n_(0, 0.04, -0.09, 0.09)
    mid = (0.5 * (bx + be), 0.5 * (by + bey) + R.n_(0, 0.015, -0.03, 0.03))
    stem = [(sx, st), (0.5 * (sx + sb), 0.6), (sb, 1.0)]
    if rounded:
        pts = P(R, (lx, ly), (bx + 0.03 + 0.25 * (lx - bx), by - 0.17), (bx + 0.11, by - 0.01), mid, (be, bey), *stem)
        return pts, [([0, 1, 2, 3, 4], False), ([5, 6, 7], False)], width(R, 0.9), slant(R)
    pts = P(R, (lx, ly), (bx, by), mid, (be, bey), *stem)
    return pts, [([0, 1], False), ([1, 2, 3], False), ([4, 5, 6], False)], width(R, 0.9), slant(R)


def four_closed(R):
    sx = R.u(0.60, 0.78); by = R.u(0.55, 0.72); bx = R.u(0.0, 0.10)
    be = sx + R.u(0.12, 0.32)
    ax = sx + R.u(-0.12, 0.0); ay = R.u(0.0, 0.06)
    st = R.u(0.0, 0.10); sb = sx + R.n_(0, 0.04, -0.09, 0.09)
    pts = P(R, (ax, ay), (bx, by), (be, by + R.u(-0.04, 0.03)), (sx, st), (sb, 1.0))
    return pts, [([0, 1], False), ([1, 2], False), ([3, 4], False)], width(R, 0.9), slant(R)


def five(R, kind=0):
    tx = R.u(0.18, 0.32)
    bx = tx + R.u(0.32, 0.90); bym = R.u(-0.22, 0.06)     # right end of the top bar (it may rise)
    vx = tx + R.u(-0.20, 0.06); vy = R.u(0.26, 0.50)      # foot of the short stem
    # the bowl: an arc from just beside the foot of the stem, round the right side and the bottom, back to the left
    ry = 0.5 * (1.0 - vy) + R.u(0.0, 0.04)
    a0 = -R.u(1.75, 2.3)                                  # start: upper left of the bowl
    a1 = R.u(2.0, 3.2)                                    # end: lower left; larger = curling up towards the stem (never closing)
    bowl = ring(R, vx + R.u(0.10, 0.26), 1.0 - ry, R.u(0.24, 0.40), ry, a0, -(a1 - a0), 7, egg=R.u(0, 0.12), pe=R.u(0, 2 * np.pi))
    pts = np.concatenate([P(R, (bx, 0.02 + bym), (tx, 0.03), (vx, vy)), bowl,
                          P(R, (tx + R.u(-0.10, 0.10), 0.03 + R.u(-0.04, 0.06)))], axis=1)
    b = list(range(3, 10))
    if kind == 0:      # bar, stem and bowl meet at corners
        pieces = [([0, 1], False), ([1, 2], False), ([2] + b, False)]
    elif kind == 1:    # the stem flows into the bowl
        pieces = [([0, 1], False), ([1, 2] + b, False)]
    elif kind == 2:    # written in one smooth movement, like an S
        pieces = [([0, 1, 2] + b, False)]
    else:              # bar added afterwards as its own stroke
        pieces = [([0, 10], False), ([1, 2], False), ([2] + b, False)]
    return pts, pieces, width(R, 0.95, 0.20), slant(R, 0.24, 0.30)


def six(R, curl=False):
    tx = R.u(0.22, 0.85); ty = R.u(0.0, 0.10)             # where the pen starts (far right: a curved top)
    lw = R.u(0.56, 0.90); lt = R.u(0.46, 0.68)            # the loop: right edge, top
    yj = 0.5 * (lt + 1.0) - 0.10 + R.u(-0.04, 0.12)
    body = [(tx, ty), (0.22 + 0.45 * (tx - 0.25), 0.16), (0.16 + 0.12 * (tx - 0.25), 0.40), (0.14, 0.68), (0.24, 0.92), (0.46, 1.0),
            (lw - 0.06, 0.92), (lw, 0.5 * (lt + 1.0) - 0.06), (lw - 0.14, lt + 0.02), (0.38, lt + 0.03)]
    if curl:           # the stroke runs on into the loop
        pts = P(R, *body, (0.24, lt + 0.13), (0.28, lt + 0.27), (0.42, lt + 0.26))
    else:              # the loop closes on the stem
        pts = P(R, *body, (0.15 + R.u(0.0, 0.07), yj))
    return pts, [(list(range(pts.shape[1])), False)], width(R, 0.85), slant(R)


def seven(R, serif=False, cross=False, rounded=False):
    bl = R.u(0.38, 0.92)                                  # length of the bar (the corner is at x = 0)
    b0 = R.u(-0.03, 0.17)                                 # left end of the bar (the bar may rise towards the corner)
    fo = R.u(-0.05, 0.62)                                 # how far left of the corner the foot lands
    bend = R.n_(0.02, 0.05, -0.08, 0.12)
    xm = -0.5 * fo + bend
    pts = P(R, (-bl, b0), (-0.5 * bl, 0.4 * b0 + R.n_(-0.02, 0.035, -0.09, 0.05)), (0, 0.0), (xm, 0.5), (-fo, 1.0),
            (-bl + R.n_(0, 0.02, -0.04, 0.04), b0 + R.u(0.12, 0.28)),
            (xm - R.u(0.16, 0.30), 0.52 + R.n_(0, 0.03, -0.06, 0.06)), (xm + R.u(0.14, 0.28), 0.50 + R.n_(0, 0.03, -0.06, 0.06)))
    pieces = [([0, 1, 2, 3, 4], False)] if rounded else [([0, 1, 2], False), ([2, 3, 4], False)]
    if serif:
        pieces.append(([5, 0], False))
    if cross:
        pieces.append(([6, 7], False))
    return pts, pieces, width(R, 1.0), slant(R)


def eight(R):
    K = 15
    t0 = R.u(-0.25, 0.45); t1 = R.u(-0.25, 0.45)          # where the stroke starts and stops near the top
    wt = R.u(0.10, 0.32); wb = R.u(0.06, 0.34)            # half-widths of the top and bottom loops (one loop may nearly close up,
    wt = np.where(np.maximum(wt, wb) < 0.20, wt + 0.12, wt)  # never both: that would be a stick, not an 8)
    yc = R.u(0.34, 0.62)                                  # height of the crossing
    k = (np.arange(K) / (K - 1.0))[None]
    t = t0[:, None] + (2 * np.pi - t1 - t0)[:, None] * k
    y0 = 0.5 * (1 - np.cos(t))
    y = np.where(y0 < 0.5, y0 * yc[:, None] / 0.5, yc[:, None] + (y0 - 0.5) * (1 - yc[:, None]) / 0.5)
    x = (wt[:, None] + (wb - wt)[:, None] * y0) * np.sin(2 * t)
    pts = np.stack([x, y], axis=-1)
    wid = width(R, 1.05, 0.20)
    return pts, [(list(range(K)), False)], wid, slant(R), 0.4 + 6.0 * np.maximum(wt, wb) * wid   # pen thin enough to leave a loop open


def eight_s(R):
    ut = R.u(0.5, 1.4); lb = R.u(0.3, 1.4)                # widths of the top and bottom loops
    ut = np.where(np.maximum(ut, lb) < 0.8, ut + 0.5, ut)
    dy = R.u(-0.07, 0.08)                                 # crossing higher or lower
    sx = 0.74 + R.u(-0.10, 0.30); sy = 0.10 + R.u(-0.08, 0.10)    # start, upper right
    ex = 0.78 + R.u(-0.12, 0.35); ey = 0.04 + R.u(-0.15, 0.12)    # end of the upstroke (may overshoot as a tail)
    def X(x, f): return 0.5 + (x - 0.5) * f
    pts = P(R, (X(sx, ut), sy), (X(0.50, ut), 0.0), (X(0.27, ut), 0.13), (X(0.34, ut), 0.33 + dy), (0.52, 0.47 + dy),
            (X(0.70, lb), 0.66), (X(0.66, lb), 0.88), (X(0.46, lb), 1.0), (X(0.25, lb), 0.88), (X(0.28, lb), 0.68),
            (0.50, 0.45 + dy), (X(0.68, ut), 0.22), (X(ex, ut), ey))
    wid = width(R, 1.0, 0.16)
    return pts, [(list(range(13)), False)], wid, slant(R), 0.4 + 6.0 * 0.21 * np.maximum(ut, lb) * wid


def eight_rings(R):
    """Two ovals, one on top of the other."""
    yc = R.u(0.40, 0.56)
    rt = R.u(0.16, 0.30); rb = R.u(0.20, 0.34)
    K = 8
    top = ring(R, R.n_(0, 0.02, -0.05, 0.05), 0.5 * yc, rt, 0.5 * yc, R.u(0, 2 * np.pi), 2 * np.pi * (1 - 1.0 / K) + 0 * yc, K)
    bot = ring(R, R.n_(0, 0.02, -0.05, 0.05), 0.5 * (1 + yc), rb, 0.5 * (1 - yc), R.u(0, 2 * np.pi), 2 * np.pi * (1 - 1.0 / K) + 0 * yc, K)
    pts = np.concatenate([top, bot], axis=1)
    return pts, [(list(range(K)), True), (list(range(K, 2 * K)), True)], width(R, 1.1, 0.22), slant(R)


def nine(R, hook=False, open_top=False):
    lx = R.u(0.15, 0.36); ly = R.u(0.19, 0.30)            # the loop (round, narrow or egg-shaped)
    cx = 0.40
    K = 9
    if open_top:      # the loop starts at its top and only meets the stem lower down: a gap at the upper right
        a0 = -R.u(0.9, 1.5)                               # (the top of the loop still curls over towards the stem)
        sweep = 2 * np.pi + a0 - R.u(-0.3, 0.3)
    else:             # the loop starts at its upper right
        a0 = -R.u(0.2, 1.1)
        sweep = 2 * np.pi - R.u(-0.25, 0.70)
    loop = ring(R, cx, ly, lx, ly, a0, sweep, K, egg=R.u(0, 0.18), pe=R.u(0, 2 * np.pi))
    sx = cx + lx + R.u(-0.04, 0.05); sy = ly - R.u(-0.05, 0.27)   # top of the stem, on the right of the loop
    if hook:
        tail = P(R, (sx, sy), (sx - 0.01, 0.66), (sx - 0.14, 0.92), (sx - 0.38, 1.0), (sx - 0.58, 0.90))
    else:
        fx = sx + R.u(-0.45, 0.08)
        tail = P(R, (sx, sy), (0.5 * (sx + fx) + R.n_(0.02, 0.03, -0.05, 0.09), 0.5 * (sy + 1.0)), (fx, 1.0))
    pts = np.concatenate([loop, tail], axis=1)
    wid = width(R, 0.95)
    return pts, [(list(range(K)), False), (list(range(K, K + tail.shape[1])), False)], wid, slant(R), 0.4 + 6.0 * lx * wid


# share of each style inside its class (styles with more freedom get more drawings)
STYLES = {
    0: [(0.62, zero, {}), (0.23, zero, {"kind": 1}), (0.15, zero, {"kind": 2})],
    1: [(0.68, one_plain, {}), (0.20, one_flag, {}), (0.10, one_base, {"flag": True}), (0.02, one_base, {"flag": False})],
    2: [(0.30, two, {}), (0.30, two, {"loop": True}), (0.10, two, {"top": 1}), (0.08, two, {"top": 1, "loop": True}),
        (0.10, two, {"top": 2}), (0.12, two, {"top": 2, "loop": True})],
    3: [(0.45, three, {}), (0.30, three, {"smooth": True}), (0.25, three, {"flat": True})],
    4: [(0.58, four_open, {}), (0.24, four_open, {"rounded": True}), (0.18, four_closed, {})],
    5: [(0.30, five, {"kind": 0}), (0.25, five, {"kind": 1}), (0.20, five, {"kind": 2}), (0.25, five, {"kind": 3})],
    6: [(0.80, six, {}), (0.20, six, {"curl": True})],
    7: [(0.34, seven, {}), (0.22, seven, {"rounded": True}), (0.17, seven, {"serif": True}), (0.21, seven, {"cross": True}),
        (0.06, seven, {"serif": True, "cross": True})],
    8: [(0.45, eight, {}), (0.42, eight_s, {}), (0.13, eight_rings, {})],
    9: [(0.58, nine, {}), (0.20, nine, {"hook": True}), (0.17, nine, {"open_top": True}), (0.05, nine, {"hook": True, "open_top": True})],
}


def class_drawings(k, count, rng, batch=250):
    """count drawings of digit k -> images (count, 28, 28)."""
    out = []
    shares = np.array([s[0] for s in STYLES[k]]); shares = shares / shares.sum()
    counts = np.floor(shares * count).astype(int); counts[0] += count - counts.sum()
    for (_, fn, kw), c in zip(STYLES[k], counts):
        done = 0
        while done < c:
            b = min(batch, c - done)
            R = Rand(rng, b)
            out.append(draw(R, *fn(R, **kw)))
            done += b
    return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------------------------------------------
# the network
# ----------------------------------------------------------------------------------------------------------
def network():
    KR = 2 * HALF - 1
    return {"layers": [
        {"name": "edge", "op": "conv", "inputs": ["image"], "out": NDIR, "kernel": [KG, KG], "stride": 1, "padding": (KG - 1) // 2},
        {"name": "edge_r", "op": "relu", "inputs": ["edge"]},
        {"name": "dir", "op": "conv", "inputs": ["edge_r"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "dir_r", "op": "relu", "inputs": ["dir"]},
        {"name": "pool", "op": "conv", "inputs": ["dir_r"], "out": NDIR, "kernel": [KP, KP], "stride": STRIDE, "padding": PADP},
        {"name": "bend", "op": "conv", "inputs": ["pool"], "out": 4 * NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "bend_r", "op": "relu", "inputs": ["bend"]},
        {"name": "root", "op": "conv", "inputs": ["bend_r"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "match_edge", "op": "dense", "inputs": ["root"], "out": 10 * TC},
        {"name": "solid", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "solid_r", "op": "relu", "inputs": ["solid"]},
        {"name": "half", "op": "avgpool", "inputs": ["solid_r"], "size": [2, 2], "stride": [2, 2]},
        {"name": "rays", "op": "conv", "inputs": ["half"], "out": 5, "kernel": [KR, KR], "stride": 1, "padding": HALF - 1},
        {"name": "rays_r", "op": "relu", "inputs": ["rays"]},
        {"name": "bays", "op": "conv", "inputs": ["rays_r"], "out": NBAY, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "bays_r", "op": "relu", "inputs": ["bays"]},
        {"name": "baypool", "op": "conv", "inputs": ["bays_r"], "out": NBAY, "kernel": [KC, KC], "stride": SC, "padding": PC},
        {"name": "match_bay", "op": "dense", "inputs": ["baypool"], "out": 10 * TC},
        {"name": "match", "op": "add", "inputs": ["match_edge", "match_bay"]},
        {"name": "bank", "op": "reshape", "inputs": ["match"], "shape": [10, POOL_A, POOL_B]},
        {"name": "best_a", "op": "maxpool", "inputs": ["bank"], "size": [1, POOL_B], "stride": [1, POOL_B]},
        {"name": "best_b", "op": "maxpool", "inputs": ["best_a"], "size": [POOL_A, 1], "stride": [POOL_A, 1]},
        {"name": "scores", "op": "reshape", "inputs": ["best_b"], "shape": [10]},
    ]}


def build(out_dir, verbose=True):
    w = frontend_weights()
    rng = np.random.default_rng(SEED)
    De = NDIR * NPOOL * NPOOL; Db = NBAY * NC * NC
    We = np.zeros((10 * TC, De), dtype=np.float32)
    Wb = np.zeros((10 * TC, Db), dtype=np.float32)
    for k in range(10):
        done = 0
        while done < TC:
            b = min(2000, TC - done)
            imgs = class_drawings(k, b, rng)
            fe = edge_features(imgs, w); fb = bay_features(imgs, w)
            norm = np.sqrt((fe * fe).sum(1, keepdims=True) + (fb * fb).sum(1, keepdims=True))   # each template divided by its own length
            We[k * TC + done:k * TC + done + b] = fe / norm
            Wb[k * TC + done:k * TC + done + b] = fb / norm
            done += b
        if verbose:
            print("class", k, "done", flush=True)
    w["match_edge.W"] = We; w["match_edge.b"] = np.zeros(10 * TC)
    w["match_bay.W"] = Wb; w["match_bay.b"] = np.zeros(10 * TC)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as fh:
        json.dump(network(), fh, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in w.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
