#!/usr/bin/env python3
"""Builds a hand-designed digit classifier: python3 build.py OUT_DIR

No data is read.  Every weight comes from formulas and from pen-path models of the ten digits typed below.

Network
  image -> 8 directional derivative-of-Gaussian filters -> relu
        -> 1x1 conv + relu: split the gradient onto its two nearest of 8 directions
        -> Gaussian blur, stride 3  (8 x 8 x 8 map of "how much edge of each direction is near here")
        -> 1x1 conv + relu + 1x1 conv: piecewise-linear square root (variance-stabilising compression)
        -> dense layer: one row per template  (cosine similarity with that template)
        -> maxpool over the templates of each class -> 10 class scores

Templates
  Each digit is drawn as pen strokes along typed control points (several ways of writing each digit), with randomly
  drawn (fixed seed) style parameters, slant, proportions and pen width, rendered like the data (scaled to a 20 x 20
  box, centre of mass moved to the centre of the 28 x 28 frame).  Each rendered drawing is passed through the same
  fixed front-end and becomes one row of the dense layer, divided by its own length.  Nothing is averaged or
  measured across drawings.
"""
import sys, json
import numpy as np

SEED = 20261120
N_PER_CLASS = 23000         # templates per class
SIGMA_G = 1.0               # derivative-of-Gaussian scale (pixels)
SIGMA_B = 2.0               # blur scale (pixels)
KB, STRIDE, PAD = 13, 3, 3  # blur kernel size, stride, padding -> 8 x 8 samples centred at 3, 6, ..., 24
NDIR = 8
GRID = 8
# knots of the piecewise-linear square root: v = 2 (k/10)^2
KNOTS = [2.0 * (k / 10.0) ** 2 for k in range(0, 10)]


# ----------------------------------------------------------------------------------------------------------------------
# fixed front-end
# ----------------------------------------------------------------------------------------------------------------------
def frontend():
    """Returns (layers, weights) of the feature extractor: image (1,28,28) -> (8,8,8)."""
    W = {}
    # 1. directional derivatives of a Gaussian: response = (gradient) . (unit vector of direction k)
    ax = np.arange(5) - 2.0
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    G = np.exp(-(xx ** 2 + yy ** 2) / (2 * SIGMA_G ** 2))
    hx = xx * G
    hx = hx / (hx * xx).sum()          # a ramp of slope 1 gives response 1
    hy = hx.T
    g = np.zeros((NDIR, 1, 5, 5))
    for k in range(NDIR):
        th = 2 * np.pi * k / NDIR
        g[k, 0] = np.cos(th) * hx + np.sin(th) * hy
    W["g.W"], W["g.b"] = g, np.zeros(NDIR)
    # 2. relu(r_k - r_{k+90deg} - r_{k-90deg}) = |grad| (cos phi - |sin phi|)+ : the parallelogram split of the
    #    gradient vector onto the two nearest of the 8 directions
    d = np.zeros((NDIR, NDIR, 1, 1))
    for k in range(NDIR):
        d[k, k, 0, 0] = 1.0
        d[k, (k + 2) % NDIR, 0, 0] = -1.0
        d[k, (k - 2) % NDIR, 0, 0] = -1.0
    W["d.W"], W["d.b"] = d, np.zeros(NDIR)
    # 3. Gaussian blur and subsampling, each direction plane separately; scaled so that one long straight
    #    full-contrast edge through the sample point gives 1
    ax = np.arange(KB) - (KB - 1) / 2.0
    g1 = np.exp(-ax ** 2 / (2 * SIGMA_B ** 2))
    g2 = np.outer(g1, g1)
    g2 = g2 / g2.sum() * (np.sqrt(2 * np.pi) * SIGMA_B)
    bl = np.zeros((NDIR, NDIR, KB, KB))
    for k in range(NDIR):
        bl[k, k] = g2
    W["blur.W"], W["blur.b"] = bl, np.zeros(NDIR)
    # 4. piecewise-linear square root through the knots (c_j, sqrt(c_j)), linear beyond the last knot:
    #    f(v) = sum_j (m_j - m_{j-1}) relu(v - c_j)
    nk = len(KNOTS)
    c = np.array(KNOTS + [2.0])
    m = np.diff(np.sqrt(c)) / np.diff(c)            # slopes of the nk segments
    inc = np.diff(np.concatenate([[0.0], m]))       # slope increments at the knots
    e = np.zeros((NDIR * nk, NDIR, 1, 1)); eb = np.zeros(NDIR * nk)
    f = np.zeros((NDIR, NDIR * nk, 1, 1))
    for k in range(NDIR):
        for j in range(nk):
            e[k * nk + j, k, 0, 0] = 1.0
            eb[k * nk + j] = -KNOTS[j]
            f[k, k * nk + j, 0, 0] = inc[j]
    W["e.W"], W["e.b"] = e, eb
    W["f.W"], W["f.b"] = f, np.zeros(NDIR)
    layers = [
        {"name": "g", "op": "conv", "inputs": ["image"], "out": NDIR, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "g_r", "op": "relu", "inputs": ["g"]},
        {"name": "d", "op": "conv", "inputs": ["g_r"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "d_r", "op": "relu", "inputs": ["d"]},
        {"name": "blur", "op": "conv", "inputs": ["d_r"], "out": NDIR, "kernel": [KB, KB], "stride": STRIDE, "padding": PAD},
        {"name": "e", "op": "conv", "inputs": ["blur"], "out": NDIR * nk, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "e_r", "op": "relu", "inputs": ["e"]},
        {"name": "f", "op": "conv", "inputs": ["e_r"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
    ]
    return layers, W


def conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def features(layers, W, images):
    """Runs the front-end layers on images (N,1,28,28); returns (N, D)."""
    vals = {"image": images}
    for L in layers:
        x = vals[L["inputs"][0]]
        if L["op"] == "conv":
            vals[L["name"]] = conv(x, W[L["name"] + ".W"], W[L["name"] + ".b"], L["stride"], L["padding"])
        elif L["op"] == "relu":
            vals[L["name"]] = np.maximum(x, 0)
    out = vals[layers[-1]["name"]]
    return out.reshape(len(out), -1)


# ----------------------------------------------------------------------------------------------------------------------
# drawing: splines, pen, normalisation like the data
# ----------------------------------------------------------------------------------------------------------------------
NSPAN = 26  # samples per spline span


def spline(P, closed=False, n=NSPAN):
    """Centripetal Catmull-Rom spline through control points P (B, K, 2) -> (B, S, 2)."""
    B, K, _ = P.shape
    if closed:
        Q = np.concatenate([P[:, -1:], P, P[:, :2]], 1)
        nspan = K
    else:
        Q = np.concatenate([2 * P[:, :1] - P[:, 1:2], P, 2 * P[:, -1:] - P[:, -2:-1]], 1)
        nspan = K - 1
    dq = np.sqrt(np.sqrt(((Q[:, 1:] - Q[:, :-1]) ** 2).sum(-1)) + 1e-6)
    u = (np.arange(n) / n)[None, :, None]
    out = []
    for i in range(nspan):
        P0, P1, P2, P3 = (Q[:, i + j][:, None, :] for j in range(4))
        d01, d12, d23 = (dq[:, i + j][:, None, None] for j in range(3))
        t1 = d01; t2 = t1 + d12; t3 = t2 + d23
        t = t1 + u * d12
        A1 = ((t1 - t) * P0 + t * P1) / t1
        A2 = ((t2 - t) * P1 + (t - t1) * P2) / d12
        A3 = ((t3 - t) * P2 + (t - t2) * P3) / d23
        B1 = ((t2 - t) * A1 + t * A2) / t2
        B2 = ((t3 - t) * A2 + (t - t1) * A3) / (t3 - t1)
        out.append(((t2 - t) * B1 + (t - t1) * B2) / d12)
    if not closed:
        out.append(P[:, -1:, :])
    return np.concatenate(out, 1)


class Gen:
    """Random numbers for a batch of B drawings."""
    def __init__(self, rng, B):
        self.rng, self.B = rng, B

    def U(self, lo, hi):
        return self.rng.uniform(lo, hi, self.B)

    def N(self, mu, sd, lo=None, hi=None):
        v = mu + sd * self.rng.standard_normal(self.B)
        return v if lo is None else np.clip(v, lo, hi)


def S(pts, corners=(), closed=False):
    """A stroke: control points [(x, y), ...] (numbers or arrays of length B), indices of sharp corners."""
    return {"pts": pts, "corners": tuple(corners), "closed": closed}


def trace(strokes, B, rng, jitter):
    """Strokes -> sample points (B, M, 2): jitter the control points, then spline between the corners."""
    out = []
    for s in strokes:
        P = np.stack([np.stack([np.broadcast_to(np.asarray(x, float), (B,)), np.broadcast_to(np.asarray(y, float), (B,))], -1) for x, y in s["pts"]], 1)
        P = P + jitter[:, None, None] * rng.standard_normal(P.shape)
        if s["closed"]:
            out.append(spline(P, closed=True))
            continue
        cuts = [0] + sorted(s["corners"]) + [P.shape[1] - 1]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                out.append(spline(P[:, a:b + 1]))
    return np.concatenate(out, 1)


def render(pts, r, aa):
    """pts (B, M, 2) pixel coordinates (x, y); pen radius r (B,), edge ramp aa (B,) -> images (B, 28, 28)."""
    gx = np.arange(28.0)
    dx2 = (gx[None, :, None] - pts[:, None, :, 0]) ** 2
    dy2 = (gx[None, :, None] - pts[:, None, :, 1]) ** 2
    d = np.sqrt((dy2[:, :, None, :] + dx2[:, None, :, :]).min(-1))
    return np.clip(0.5 + (r[:, None, None] - d) / aa[:, None, None], 0.0, 1.0)


def draw(pts, r, aa):
    """Like the data: scale the drawing to a 20 x 20 box, then move its centre of mass to the centre of the frame
    by a whole number of pixels."""
    mn, mx = pts.min(1), pts.max(1)
    ext = (mx - mn).max(1)
    s = (20.0 - 2 * r) / ext
    p = (pts - ((mn + mx) / 2)[:, None, :]) * s[:, None, None] + 13.5
    img = render(p, r, aa)
    m = img.sum((1, 2))
    idx = np.arange(28.0)
    cx = (img.sum(1) * idx).sum(1) / m
    cy = (img.sum(2) * idx).sum(1) / m
    sx = np.rint(13.5 - cx).astype(int)
    sy = np.rint(13.5 - cy).astype(int)
    out = np.zeros_like(img)
    for i in range(len(img)):
        a = np.roll(img[i], (sy[i], sx[i]), (0, 1))
        out[i] = a
    return out


def distort(pts, g, slant=(0.2, 0.25, -0.35, 0.9), asp_sd=0.2, rot_sd=0.1):
    """Whole-digit variation: gentle smooth bends, proportions, slant, rotation."""
    x, y = pts[..., 0], pts[..., 1]
    c = lambda v: v[:, None]
    u = x - (x.min(1) + x.max(1))[:, None] / 2
    v = y - (y.min(1) + y.max(1))[:, None] / 2
    q = lambda: c(g.N(0, 0.15, -0.3, 0.3))
    u, v = u + q() * u * u + q() * u * v + q() * v * v, v + q() * u * u + q() * u * v + q() * v * v   # smooth bends
    u = u * c(np.exp(g.N(0, asp_sd, -2 * asp_sd, 2 * asp_sd)))        # narrow / wide
    u = u - c(g.N(*slant)) * v                                        # slant
    a = c(g.N(0, rot_sd, -2.5 * rot_sd, 2.5 * rot_sd))                # rotation (radians)
    u, v = np.cos(a) * u - np.sin(a) * v, np.sin(a) * u + np.cos(a) * v
    return np.stack([u, v], -1)


# ----------------------------------------------------------------------------------------------------------------------
# pen paths of the ten digits.  Coordinates: x to the right, y downwards, the digit is about 1 tall.
# Every number below is typed from how the digit is usually written; each model returns its strokes.
# ----------------------------------------------------------------------------------------------------------------------
def zero(g):
    a = g.U(0.2, 0.5)
    t0 = -np.pi / 2 + g.N(0, 0.3, -0.7, 0.7)
    ov = g.U(-0.3, 0.7)
    drift = g.N(0, 0.07, -0.15, 0.15)
    egg = g.N(0, 0.1, -0.2, 0.2)
    K = 12
    pts = []
    for j in range(K):
        f = j / (K - 1.0)
        t = t0 - (2 * np.pi + ov) * f
        rho = 1 + drift * (2 * f - 1)
        pts.append((a * rho * np.cos(t) * (1 + egg * np.sin(t)), 0.5 + 0.5 * rho * np.sin(t)))
    return [S(pts)]


def zero_tail(g):
    """closed oval whose end runs on past the start as a short tail, outwards or inwards"""
    a = g.U(0.2, 0.46)
    t0 = -np.pi / 2 + g.N(0.2, 0.3, -0.5, 0.9)
    egg = g.N(0, 0.1, -0.2, 0.2)
    K = 11
    pts = []
    for j in range(K):
        t = t0 - 2 * np.pi * j / (K - 1.0)
        pts.append((a * np.cos(t) * (1 + egg * np.sin(t)), 0.5 + 0.5 * np.sin(t)))
    tl = g.U(0.12, 0.3) * np.sign(g.U(-1, 1.5))      # tail length; positive = outwards
    ta = t0 - g.U(0.2, 0.7)
    pts.append((a * (1 + tl) * np.cos(ta), 0.5 + 0.5 * (1 + tl) * np.sin(ta)))
    return [S(pts)]


def _stem(g):
    return [(0, 0), (g.N(0, 0.03), 0.5), (0, 1)]


def one_plain(g):
    return [S(_stem(g))]


def _flag(g):
    fl = g.U(0.1, 0.3)
    return (-fl, fl * g.U(0.7, 1.4))


def one_flag(g):
    return [S([_flag(g)] + _stem(g), corners=[1])]


def _base(g):
    return S([(-g.U(0.15, 0.4), 1 + g.N(0, 0.02)), (g.U(0.15, 0.4), 1 + g.N(0, 0.02))])


def one_flag_base(g):
    return [S([_flag(g)] + _stem(g), corners=[1]), _base(g)]


def one_base(g):
    return [S(_stem(g)), _base(g)]


def _two_top(g):
    """the arc of a 2: from its left end over the top to the right side, then the start of the diagonal"""
    sx, sy = g.U(0.0, 0.25), g.U(0.12, 0.38)
    px = sx + g.U(0.2, 0.35)
    rx, ry = px + g.U(0.2, 0.35), g.U(0.15, 0.32)
    return sx, sy, rx, ry, [(sx, sy), ((sx + px) / 2 - 0.03, 0.06), (px, 0.0), ((px + rx) / 2 + 0.03, 0.06), (rx, ry)]


def _two_diag(g, rx, ry, cx, cy):
    return ((rx + cx) / 2 + g.N(0.04, 0.05), (ry + cy) / 2 + g.N(0.0, 0.04))


def _two_base(g, cx, cy):
    ex, ey = cx + g.U(0.6, 1.05), cy + g.N(0, 0.07)
    return [((cx + ex) / 2, (cy + ey) / 2 + g.N(0, 0.05)), (ex, ey)]


def two_corner(g):
    sx, sy, rx, ry, top = _two_top(g)
    cx, cy = g.U(0.0, 0.15), g.U(0.88, 1.0)
    return [S(top + [_two_diag(g, rx, ry, cx, cy), (cx, cy)] + _two_base(g, cx, cy), corners=[6])]


def two_hook(g):
    """starts with a small inward curl"""
    sx, sy, rx, ry, top = _two_top(g)
    cx, cy = g.U(0.0, 0.15), g.U(0.88, 1.0)
    return [S([(sx + g.U(0.06, 0.18), sy + g.U(0.04, 0.16))] + top + [_two_diag(g, rx, ry, cx, cy), (cx, cy)] + _two_base(g, cx, cy), corners=[7])]


def two_wave(g):
    """base drawn as a wave that dips and then rises"""
    sx, sy, rx, ry, top = _two_top(g)
    cx, cy = g.U(0.0, 0.15), g.U(0.85, 0.97)
    w = g.U(0.6, 1.05)
    return [S(top + [_two_diag(g, rx, ry, cx, cy), (cx, cy), (cx + 0.35 * w, cy + g.U(0.0, 0.08)), (cx + 0.7 * w, cy - g.U(0.0, 0.06)),
                     (cx + w, cy - g.U(-0.04, 0.16))], corners=[6])]


def two_loop(g):
    """the diagonal turns back in a loop at the bottom left before the base"""
    sx, sy, rx, ry, top = _two_top(g)
    L = g.U(0.1, 0.36)                 # loop width
    H = L * g.U(0.5, 1.0)              # loop height
    cx, cy = g.U(0.0, 0.1), g.U(0.88, 1.0)
    ex, ey = cx + g.U(0.65, 1.1), cy + g.N(0, 0.08)
    x1 = cx + 1.4 * L
    return [S(top + [_two_diag(g, rx, ry, cx + 0.6 * L, cy), (cx + 0.6 * L, cy), (cx, cy - 0.45 * H), (cx + 0.5 * L, cy - H),
                     (x1, cy - 0.45 * H), ((x1 + ex) / 2, (cy + ey) / 2 + g.N(0.0, 0.05)), (ex, ey)])]


def two_z(g):
    """angular, like a Z with a bowed top"""
    sx, sy = g.U(0.0, 0.15), g.U(0.03, 0.18)
    rx, ry = g.U(0.55, 0.85), g.U(0.0, 0.12)
    cx, cy = g.U(0.0, 0.15), g.U(0.88, 1.0)
    return [S([(sx, sy), ((sx + rx) / 2, (sy + ry) / 2 - g.U(0.0, 0.1)), (rx, ry), _two_diag(g, rx, ry, cx, cy), (cx, cy)] + _two_base(g, cx, cy),
              corners=[2, 4])]


def _three_low(g, mx, my):
    """lower bowl of a 3 from the middle point, ending at the lower left"""
    lx, ly = g.U(0.55, 0.9), g.U(0.66, 0.78)
    bx = g.U(0.25, 0.5)
    ex, ey = g.U(-0.2, 0.15), g.U(0.78, 0.98)
    return [((mx + lx) / 2 + 0.05, my + 0.05), (lx, ly), ((bx + lx) / 2 + 0.05, 0.95), (bx, 1.0), (ex, ey)]


def three_round(g):
    sx, sy = g.U(-0.05, 0.2), g.U(0.05, 0.25)
    ax = g.U(0.25, 0.45)
    ux, uy = g.U(0.5, 0.8), g.U(0.14, 0.24)
    mx, my = g.U(0.15, 0.45), g.U(0.38, 0.52)
    return [S([(sx, sy), (ax, 0.0), (ux, uy), ((ux + mx) / 2 + 0.06, (uy + my) / 2 + 0.06), (mx, my)] + _three_low(g, mx, my), corners=[4])]


def three_flat(g):
    """flat top bar, then a diagonal back to the middle"""
    sx = g.U(-0.05, 0.15)
    ux = g.U(0.5, 0.85)
    mx, my = g.U(0.15, 0.45), g.U(0.34, 0.5)
    return [S([(sx, g.N(0.03, 0.03)), (ux, g.N(0.02, 0.03)), (mx, my)] + _three_low(g, mx, my), corners=[1, 2])]


def _four_stem(g, vx, top):
    return S([(vx + g.N(0.03, 0.04), top), (vx, 0.55), (vx + g.N(-0.02, 0.05), 1.0)])


def four_open(g):
    ax, ay = g.U(0.0, 0.35), g.U(0.0, 0.15)
    bx, by = g.U(0.0, 0.12), g.U(0.45, 0.7)
    vx = g.U(0.45, 0.75)
    ex, ey = vx + g.U(0.0, 0.35), by + g.N(-0.02, 0.05)
    return [S([(ax, ay), ((ax + bx) / 2 + g.N(0, 0.04), (ay + by) / 2), (bx, by), (ex, ey)], corners=[2]), _four_stem(g, vx, g.U(-0.1, 0.3))]


def four_u(g):
    """left stroke curving round into the bar"""
    ax, ay = g.U(0.0, 0.25), g.U(0.0, 0.15)
    by = g.U(0.42, 0.66)
    vx = g.U(0.45, 0.75)
    return [S([(ax, ay), (ax - 0.02 + g.N(0, 0.03), by - 0.22), (ax + 0.1, by), (vx + g.U(0.0, 0.2), by - g.U(0.0, 0.1))]), _four_stem(g, vx, g.U(-0.1, 0.25))]


def four_closed(g):
    tx = g.U(0.4, 0.68)
    bx, by = g.U(0.0, 0.1), g.U(0.5, 0.74)
    ex, ey = tx + g.U(0.08, 0.35), by + g.N(-0.01, 0.04)
    return [S([(tx, 0.0), (bx, by), (ex, ey)], corners=[1]),
            S([(tx + g.N(0.0, 0.02), g.N(0.0, 0.02)), (tx + g.N(0, 0.02), 0.5), (tx + g.N(0, 0.05), 1.0)])]


def _five_bowl(g, vx, vy):
    """bowl of a 5 from the foot of its vertical stroke round to the lower left"""
    bx = g.U(0.58, 0.9)
    ex, ey = g.U(-0.1, 0.15), g.U(0.75, 0.95)
    mx = g.U(0.3, 0.5)
    return [(vx + 0.2, vy - g.U(0.02, 0.1)), (bx - 0.08, vy + 0.07), (bx, 0.5 * (vy + 1.0) + g.N(0.02, 0.03)), ((bx + mx) / 2 + 0.04, 0.95), (mx, 1.0), (ex, ey)]


def _five_start(g):
    tx = g.U(0.1, 0.3)
    vx, vy = tx + g.N(-0.04, 0.06), g.U(0.3, 0.52)
    rx, ry = tx + g.U(0.3, 0.8), g.N(-0.03, 0.06)
    return tx, vx, vy, rx, ry


def five_std(g):
    tx, vx, vy, rx, ry = _five_start(g)
    return [S([(rx, ry), (tx, 0.04), (vx, vy)] + _five_bowl(g, vx, vy), corners=[1, 2])]


def five_two_strokes(g):
    """top bar drawn separately, not quite meeting the vertical"""
    tx, vx, vy, rx, ry = _five_start(g)
    return [S([(tx, 0.05), (vx, vy)] + _five_bowl(g, vx, vy), corners=[1]),
            S([(tx + g.N(0.03, 0.05), g.N(0.02, 0.04)), (rx, ry)])]


def five_s(g):
    """rounded, like an S with a flat top"""
    tx, vx, vy, rx, ry = _five_start(g)
    b = _five_bowl(g, vx, vy)
    return [S([(rx, ry), (tx + 0.14, 0.02), (tx, 0.15), (vx + 0.02, vy - 0.06)] + b[1:])]


def five_curl(g):
    """bowl whose end curls back up towards the vertical"""
    tx, vx, vy, rx, ry = _five_start(g)
    b = _five_bowl(g, vx, vy)
    ex, ey = b[-1]
    return [S([(rx, ry), (tx, 0.04), (vx, vy)] + b + [(ex + g.U(0.0, 0.08), ey - g.U(0.05, 0.14))], corners=[1, 2])]


def _six_loop(g, rx, ry):
    """loop of a 6, from its left side round the bottom and back to the stem"""
    cx, cy = rx + 0.02, 1.0 - ry
    return cx, cy, [(cx - rx, cy + 0.1 * ry), (cx - 0.25 * rx, 1.0), (cx + 0.8 * rx, cy + 0.6 * ry), (cx + rx, cy - 0.15 * ry), (cx + 0.2 * rx, cy - ry),
                    (cx - rx * g.U(0.5, 1.0), cy - ry * g.U(-0.1, 0.6))]


def six_curved(g):
    rx, ry = g.U(0.2, 0.36), g.U(0.18, 0.3)
    sx = g.U(0.3, 0.8)
    cx, cy, loop = _six_loop(g, rx, ry)
    return [S([(sx, 0.0), (0.45 * sx + 0.08 + g.N(0, 0.03), 0.22), (0.06 + g.N(0, 0.02), cy - 0.9 * ry)] + loop)]


def six_straight(g):
    rx, ry = g.U(0.18, 0.32), g.U(0.17, 0.28)
    sx = g.U(0.05, 0.4)
    cx, cy, loop = _six_loop(g, rx, ry)
    return [S([(sx, 0.0), (0.5 * sx + 0.03 + g.N(0, 0.03), 0.35)] + loop)]


def _seven_bar(g):
    lx, ly = g.U(0.0, 0.12), g.N(0.06, 0.04)
    kx, ky = lx + g.U(0.45, 0.85), g.N(0.03, 0.04)
    return lx, ly, kx, ky, [(lx, ly), ((lx + kx) / 2, (ly + ky) / 2 - g.U(-0.02, 0.07)), (kx, ky)]


def _seven_stem(g, kx, ky):
    bx = g.U(0.1, 0.6)
    return [((kx + bx) / 2 + g.N(0.03, 0.06), 0.52), (bx, 1.0)]


def seven_plain(g):
    lx, ly, kx, ky, bar = _seven_bar(g)
    return [S(bar + _seven_stem(g, kx, ky), corners=[2])]


def seven_serif(g):
    """bar starting with a short downward tick"""
    lx, ly, kx, ky, bar = _seven_bar(g)
    return [S([(lx + g.N(0, 0.03), ly + g.U(0.08, 0.25))] + bar + _seven_stem(g, kx, ky), corners=[1, 3])]


def seven_bar(g):
    """with a cross-stroke through the stem"""
    lx, ly, kx, ky, bar = _seven_bar(g)
    stem = _seven_stem(g, kx, ky)
    mx = stem[0][0]
    cy = g.U(0.42, 0.62)
    return [S(bar + stem, corners=[2]),
            S([(mx - g.U(0.15, 0.32), cy + g.N(0, 0.03)), (mx + g.U(0.12, 0.3), cy + g.N(0, 0.03))])]


def seven_round(g):
    """bar turning into the stem in a curve"""
    kx = g.U(0.5, 0.85)
    bx = g.U(0.15, 0.6)
    return [S([(g.U(0.0, 0.1), g.U(0.04, 0.18)), (0.3, g.N(0.02, 0.02)), (kx, 0.06), (kx - 0.02, 0.25), ((kx + bx) / 2 + g.N(0, 0.04), 0.6), (bx, 1.0)])]


def eight_s(g):
    """one stroke: over the top, down through the middle, round the bottom and back up to the start"""
    h = g.U(0.36, 0.6)                          # height of the crossing
    wt, wb = g.U(0.12, 0.28), g.U(0.13, 0.32)   # half widths of the top and bottom loops
    cx = 0.35
    ox = g.N(0.02, 0.05)
    sx, sy = cx + ox + wt * g.U(0.3, 1.0), g.U(0.04, 0.2) * h / 0.5
    return [S([(sx, sy), (cx + ox, 0.0), (cx + ox - wt, 0.45 * h), (cx, h), (cx + wb, h + 0.55 * (1 - h)), (cx + 0.1 * wb, 1.0),
               (cx - wb, h + 0.55 * (1 - h)), (cx, h), (cx + ox + wt, 0.45 * h), (sx + g.N(0.02, 0.07), sy + g.N(-0.04, 0.08))])]


def eight_rings(g):
    """two rings, one on top of the other"""
    h = g.U(0.38, 0.56)
    wt, wb = g.U(0.15, 0.28), g.U(0.17, 0.32)
    cx = 0.35 + g.N(0, 0.04)
    top = [(cx + wt * np.cos(t), 0.5 * h + 0.5 * h * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 9)[:-1] + 0.3]
    bot = [(0.35 + wb * np.cos(t), h + 0.5 * (1 - h) + 0.5 * (1 - h) * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 9)[:-1] + 0.3]
    return [S(top, closed=True), S(bot, closed=True)]


def _nine_loop(g):
    """loop of a 9 drawn from its upper right round to the top of the tail; may be left a little open"""
    r = g.U(0.2, 0.34)             # loop half width
    h = g.U(0.42, 0.62)            # loop height
    jx = 0.05 + 2 * r
    gap = g.U(0.0, 0.6)
    return r, h, jx, [(jx - gap * r, g.U(0.02, 0.25) * h), (0.05 + r, 0.0), (0.05, 0.45 * h), (0.05 + 0.8 * r, h), (jx - 0.04, 0.62 * h), (jx, g.U(-0.05, 0.2) * h)]


def _nine_tail(g, jx, h):
    bx = g.U(0.1, jx + 0.1)
    return [((jx + bx) / 2 + g.N(0.04, 0.04), 0.5 * (1 + 0.3 * h)), (bx, 1.0)]


def nine_straight(g):
    r, h, jx, loop = _nine_loop(g)
    return [S(loop + _nine_tail(g, jx, h), corners=[5])]


def nine_hook(g):
    """tail curling to the left at the bottom"""
    r, h, jx, loop = _nine_loop(g)
    return [S(loop + [(jx + g.N(0, 0.03), 0.62), (jx - 0.5 * r, 0.93), (0.05 + 0.6 * r, 1.0), (g.U(0.0, 0.15), g.U(0.82, 0.96))], corners=[5])]


def nine_ring(g):
    """a closed ring with a separate stem down its right side"""
    r = g.U(0.2, 0.34)
    h = g.U(0.42, 0.62)
    cx = 0.05 + r
    ring = [(cx + r * np.cos(t), 0.5 * h + 0.5 * h * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 9)[:-1] + 0.4]
    jx = cx + r + g.N(0, 0.02)
    return [S(ring, closed=True), S([(jx + g.N(0.01, 0.02), g.U(-0.1, 0.2) * h)] + _nine_tail(g, jx, h))]


ONE_SLANT = (0.3, 0.3, -0.3, 0.9)
# (class, pen-path model, share of the class's templates, keyword settings for distort())
MODELS = [
    (0, zero, 0.8, {}), (0, zero_tail, 0.2, {}),
    (1, one_plain, 0.7, {"slant": ONE_SLANT, "rot_sd": 0.05}), (1, one_flag, 0.15, {}), (1, one_flag_base, 0.12, {}), (1, one_base, 0.03, {}),
    (2, two_corner, 0.25, {}), (2, two_loop, 0.35, {}), (2, two_hook, 0.15, {}), (2, two_wave, 0.15, {}), (2, two_z, 0.1, {}),
    (3, three_round, 0.7, {}), (3, three_flat, 0.3, {}),
    (4, four_open, 0.55, {}), (4, four_u, 0.3, {}), (4, four_closed, 0.15, {}),
    (5, five_std, 0.4, {}), (5, five_two_strokes, 0.25, {}), (5, five_s, 0.2, {}), (5, five_curl, 0.15, {}),
    (6, six_curved, 0.75, {}), (6, six_straight, 0.25, {}),
    (7, seven_plain, 0.4, {}), (7, seven_serif, 0.2, {}), (7, seven_bar, 0.2, {}), (7, seven_round, 0.2, {}),
    (8, eight_s, 0.85, {}), (8, eight_rings, 0.15, {}),
    (9, nine_straight, 0.7, {}), (9, nine_hook, 0.1, {}), (9, nine_ring, 0.2, {}),
]


def make_images(model, n, rng, kw=None, chunk=64):
    """n drawings of one pen-path model -> (n, 28, 28)."""
    out = []
    for s in range(0, n, chunk):
        B = min(chunk, n - s)
        g = Gen(rng, B)
        strokes = model(g)
        pts = trace(strokes, B, rng, g.U(0.01, 0.04))
        pts = distort(pts, g, **(kw or {}))
        r = np.exp(g.N(np.log(1.45), 0.25, np.log(0.8), np.log(2.6)))
        out.append(draw(pts, r, g.U(1.0, 1.6)))
    return np.concatenate(out) if out else np.zeros((0, 28, 28))


def class_images(c, n, rng):
    ms = [m for m in MODELS if m[0] == c]
    counts = [int(round(m[2] * n)) for m in ms]
    counts[0] += n - sum(counts)
    return np.concatenate([make_images(m[1], k, rng, m[3]) for m, k in zip(ms, counts)])


def main(out_dir):
    layers, W = frontend()
    rows = []
    for c in range(10):
        rng = np.random.default_rng([SEED, c])
        imgs = class_images(c, N_PER_CLASS, rng)
        F = np.concatenate([features(layers, W, imgs[i:i + 250, None]) for i in range(0, len(imgs), 250)])
        F = F / np.sqrt((F ** 2).sum(1, keepdims=True))      # each template divided by its own length
        rows.append(F.astype(np.float32))
    T = 10 * N_PER_CLASS
    D = NDIR * GRID * GRID
    W["t.W"] = np.concatenate(rows)
    W["t.b"] = np.zeros(T)
    layers = layers + [
        {"name": "t", "op": "dense", "inputs": ["f"], "out": T},
        {"name": "t_m", "op": "reshape", "inputs": ["t"], "shape": [1, 10, N_PER_CLASS]},
        {"name": "best", "op": "maxpool", "inputs": ["t_m"], "size": [1, N_PER_CLASS], "stride": [1, N_PER_CLASS]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]},
    ]
    assert W["t.W"].shape == (T, D)
    with open(out_dir + "/network.json", "w") as fh:
        json.dump({"layers": layers}, fh, indent=1)
    np.savez(out_dir + "/weights.npz", **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
