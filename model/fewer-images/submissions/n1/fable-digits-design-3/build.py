"""Builds a hand-designed digit classifier: python3 build.py OUT_DIR  ->  OUT_DIR/network.json, OUT_DIR/weights.npz

No data is read.  Every weight comes from formulas and from pen paths typed below.

  1. Fixed front end (formulas only):
       a. oriented edge energy: 16 Gaussian-derivative filters -> relu -> 1x1 mixing -> relu gives 8 unsigned,
          sharpened orientation channels; a Gaussian pooling conv (stride 3) reduces each to 8 x 8.
       b. concavity maps: a dense layer integrates the ink along the eight compass rays leaving every pixel, the
          sums are clipped to [0, 1] ("there is ink in that direction"), and five channels mark background pixels
          that are closed in all directions (a hole) or closed everywhere except upward / downward / to the left /
          to the right (an open cup); same pooling.
  2. Templates: digits are drawn as pen strokes along typed paths (several writing styles per digit, random
     slant, proportions, pen width from a fixed seed).  Each drawing, passed through the front end and divided
     by its own norm, is one row of the dense template layer.  The score of a class is the largest
     template response of that class (maxpool).
"""
import sys, json
from pathlib import Path
import numpy as np

F = np.float32
SEED = 20261120
PER_CLASS = 13700          # templates per digit
GROUP = 137                # first max-pooling stage (PER_CLASS = GROUP * 100)

# ------------------------------------------------------------------------------------------------------------------
# front-end constants (chosen by hand)
# ------------------------------------------------------------------------------------------------------------------
SIG_D = 1.0                # scale of the derivative filters (pixels)
N_ORI = 8                  # unsigned orientations, 22.5 degrees apart
ORI_ZERO = 45.0            # orientation tuning |cos| - |sin| falls to zero 45 degrees away from a channel's orientation;
                           # with channels 22.5 degrees apart the summed response is then nearly the same for every
                           # stroke direction (2.08 on a channel, 2.12 half-way between two)
SIG_P = 2.0                # pooling scale (pixels)
POOL_K, POOL_S, POOL_PAD = 13, 3, 3      # pooling conv: 28 -> 8 cells centred on pixels 3, 6, ..., 24
RAY_GAIN = 1.5             # a ray is "closed" when the ink integrated along it reaches 1 / RAY_GAIN pixels
INK_SUP = 2.0              # concavity maps are suppressed on ink
N_CC = 5
# Weight of the concavity channels relative to the orientation channels, from a calculation on ideal strokes:
# an edge of a stroke of half-width r seen through the derivative filter carries mass m = erf(r / (sqrt(2) * 1.07)),
# about 0.8 for a typical pen; pooled (sigma about 2.3 across the edge, two edges 2r apart, stride 3, about 1.5 from
# the overlapping orientation channels) a stroke of length l gives a squared feature norm of about 0.07 * m^2 * l,
# i.e. roughly 2 for 45 pixels of stroke.  A concavity region of area A gives about 0.8 * A / 9, i.e. roughly 3 for
# a 30 to 40 pixel region.  The concavity part should count for a third to a half of the orientation part:
# CC_GAIN^2 * 3 = 0.75, so CC_GAIN = 0.5.
CC_GAIN = 0.5
RAY_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]   # (dy, dx): up, up-right, right, ... clockwise


def dog_filters():
    """16 first-derivative-of-Gaussian filters, directions 22.5 degrees apart: response = gradient . direction."""
    r = np.arange(-3, 4)
    dy, dx = np.meshgrid(r, r, indexing="ij")
    gauss = np.exp(-(dx ** 2 + dy ** 2) / (2 * SIG_D ** 2))
    gauss /= gauss.sum()
    W = np.zeros((2 * N_ORI, 1, 7, 7))
    for k in range(2 * N_ORI):
        th = np.pi * k / N_ORI
        W[k, 0] = (dx * np.cos(th) + dy * np.sin(th)) * gauss / SIG_D ** 2
    return W


def mix_weights():
    """|g.u| - beta |g.u_perp| per orientation: unsigned, with tuning that vanishes ORI_ZERO degrees away."""
    beta = 1.0 / np.tan(np.deg2rad(ORI_ZERO))
    W = np.zeros((N_ORI, 2 * N_ORI, 1, 1))
    for j in range(N_ORI):
        W[j, j, 0, 0] = W[j, j + N_ORI, 0, 0] = 1.0
        W[j, (j + N_ORI // 2) % (2 * N_ORI), 0, 0] = -beta
        W[j, (j + N_ORI // 2 + N_ORI) % (2 * N_ORI), 0, 0] = -beta
    return W


def pool_weights(C):
    r = np.arange(POOL_K) - (POOL_K - 1) / 2
    g = np.exp(-r ** 2 / (2 * SIG_P ** 2))
    g /= g.sum()
    W = np.zeros((C, C, POOL_K, POOL_K))
    for c in range(C):
        W[c, c] = np.outer(g, g)
    return W


def ray_weights():
    """(8 * 784, 784): ink integrated along the ray leaving each pixel in each of the 8 compass directions."""
    W = np.zeros((8, 28, 28, 28, 28))
    for d, (dy, dx) in enumerate(RAY_DIRS):
        step = np.hypot(dy, dx)                       # diagonal samples are sqrt(2) apart
        for y in range(28):
            for x in range(28):
                yy, xx = y + dy, x + dx
                while 0 <= yy < 28 and 0 <= xx < 28:
                    W[d, y, x, yy, xx] = RAY_GAIN * step
                    yy, xx = yy + dy, xx + dx
    return W.reshape(8 * 784, 784)


def cc_weights():
    """From the clipped rays R_d = relu(S_d) - relu(S_d - 1), d = 0..7 clockwise from up:
    channel 0 (hole): closed in all 8 directions;
    channels 1..4 (cups): closed in the 5 directions that are not within 45 degrees of up / down / left / right."""
    W = np.zeros((N_CC, 16, 1, 1))
    b = np.zeros(N_CC)
    use = np.zeros((N_CC, 8))
    use[0] = 1.0
    for c, centre in zip(range(1, N_CC), (0, 4, 6, 2)):          # up, down, left, right
        for d in range(8):
            if min((d - centre) % 8, (centre - d) % 8) > 1:
                use[c, d] = 1.0
    for c in range(N_CC):
        W[c, :8, 0, 0] = use[c]
        W[c, 8:, 0, 0] = -use[c]
        b[c] = -(use[c].sum() - 1.0)
    return W, b


def pool_c_weights():
    """Gaussian pooling of: hole, and each cup minus the hole (a cup that is open on its own side)."""
    P = pool_weights(1)[0, 0]
    W = np.zeros((N_CC, N_CC, POOL_K, POOL_K))
    W[0, 0] = P
    for c in range(1, N_CC):
        W[c, c] = P
        W[c, 0] = -P
    return W


def front_end_weights():
    w = {}
    w["g.W"], w["g.b"] = dog_filters(), np.zeros(2 * N_ORI)
    w["mix.W"], w["mix.b"] = mix_weights(), np.zeros(N_ORI)
    w["pool_o.W"], w["pool_o.b"] = pool_weights(N_ORI), np.zeros(N_ORI)
    w["rays.W"], w["rays.b"] = ray_weights(), np.zeros(8 * 784)
    rc = np.zeros((16, 8, 1, 1))
    for d in range(8):
        rc[d, d, 0, 0] = rc[8 + d, d, 0, 0] = 1.0
    w["rc.W"], w["rc.b"] = rc, np.concatenate([np.zeros(8), -np.ones(8)])
    w["cc_pre.W"], w["cc_pre.b"] = cc_weights()
    w["ink.W"], w["ink.b"] = np.full((N_CC, 1, 1, 1), -INK_SUP), np.zeros(N_CC)
    w["pool_c.W"], w["pool_c.b"] = pool_c_weights(), np.zeros(N_CC)
    return {k: v.astype(F) for k, v in w.items()}


def network(n_templates, per_class, group):
    L = [
        {"name": "g", "op": "conv", "inputs": ["image"], "out": 2 * N_ORI, "kernel": [7, 7], "stride": 1, "padding": 3},
        {"name": "g_r", "op": "relu", "inputs": ["g"]},
        {"name": "mix", "op": "conv", "inputs": ["g_r"], "out": N_ORI, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "mix_r", "op": "relu", "inputs": ["mix"]},
        {"name": "pool_o", "op": "conv", "inputs": ["mix_r"], "out": N_ORI, "kernel": [POOL_K, POOL_K], "stride": POOL_S, "padding": POOL_PAD},
        {"name": "t_or", "op": "dense", "inputs": ["pool_o"], "out": n_templates},
        {"name": "rays", "op": "dense", "inputs": ["image"], "out": 8 * 784},
        {"name": "rays_m", "op": "reshape", "inputs": ["rays"], "shape": [8, 28, 28]},
        {"name": "rc", "op": "conv", "inputs": ["rays_m"], "out": 16, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "rc_r", "op": "relu", "inputs": ["rc"]},
        {"name": "cc_pre", "op": "conv", "inputs": ["rc_r"], "out": N_CC, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "ink", "op": "conv", "inputs": ["image"], "out": N_CC, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "cc_sum", "op": "add", "inputs": ["cc_pre", "ink"]},
        {"name": "cc", "op": "relu", "inputs": ["cc_sum"]},
        {"name": "pool_c", "op": "conv", "inputs": ["cc"], "out": N_CC, "kernel": [POOL_K, POOL_K], "stride": POOL_S, "padding": POOL_PAD},
        {"name": "t_cc", "op": "dense", "inputs": ["pool_c"], "out": n_templates},
        {"name": "sim", "op": "add", "inputs": ["t_or", "t_cc"]},
        {"name": "sim_m", "op": "reshape", "inputs": ["sim"], "shape": [10, 1, per_class]},
        {"name": "mp1", "op": "maxpool", "inputs": ["sim_m"], "size": [1, group], "stride": [1, group]},
        {"name": "mp2", "op": "maxpool", "inputs": ["mp1"], "size": [1, per_class // group], "stride": [1, per_class // group]},
        {"name": "scores", "op": "reshape", "inputs": ["mp2"], "shape": [10]},
    ]
    return {"layers": L}


# ------------------------------------------------------------------------------------------------------------------
# the front end in numpy (same arithmetic as the network), used to turn each drawing into its template
# ------------------------------------------------------------------------------------------------------------------
def conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def front_end(img, w):
    """img (B, 28, 28) float32 -> orientation features (B, 512), concavity features (B, 320)."""
    x = img[:, None].astype(F)
    n = len(x)
    g = np.maximum(conv(x, w["g.W"], w["g.b"], 1, 3), 0)
    o = np.maximum(conv(g, w["mix.W"], w["mix.b"], 1, 0), 0)
    fo = conv(o, w["pool_o.W"], w["pool_o.b"], POOL_S, POOL_PAD).reshape(n, -1)
    rays = (x.reshape(n, -1) @ w["rays.W"].T + w["rays.b"]).reshape(n, 8, 28, 28)
    rc = np.maximum(conv(rays, w["rc.W"], w["rc.b"], 1, 0), 0)
    cc = np.maximum(conv(rc, w["cc_pre.W"], w["cc_pre.b"], 1, 0) + conv(x, w["ink.W"], w["ink.b"], 1, 0), 0)
    fc = conv(cc, w["pool_c.W"], w["pool_c.b"], POOL_S, POOL_PAD).reshape(n, -1)
    return fo, fc


# ------------------------------------------------------------------------------------------------------------------
# pen-stroke models of the digits
# ------------------------------------------------------------------------------------------------------------------
class G:
    """Random draws of shape (B,) and knot lists for one batch of drawings."""

    def __init__(self, rng, B):
        self.rng, self.B = rng, B
        self.sloppy = rng.uniform(0.6, 2.2, (B, 1, 1))      # some writers are neat, some are not

    def U(self, a, b):
        return self.rng.uniform(a, b, self.B)

    def N(self, m, s):
        return m + s * self.rng.standard_normal(self.B)

    def pts(self, lst):
        out = np.zeros((self.B, len(lst), 2))
        for k, (x, y) in enumerate(lst):
            out[:, k, 0] = x
            out[:, k, 1] = y
        return out


def crspline(P, n):
    """Centripetal Catmull-Rom curve through the knots P (B, K, 2); n samples per segment, plus the last knot."""
    B, K, _ = P.shape
    Q = np.concatenate([2 * P[:, :1] - P[:, 1:2], P, 2 * P[:, -1:] - P[:, -2:-1]], axis=1)
    d = np.sqrt(np.sqrt(((Q[:, 1:] - Q[:, :-1]) ** 2).sum(-1)) + 1e-9)
    p0, p1, p2, p3 = Q[:, :-3], Q[:, 1:-2], Q[:, 2:-1], Q[:, 3:]
    t1 = d[:, :-2]
    t2 = t1 + d[:, 1:-1]
    t3 = t2 + d[:, 2:]
    u = np.linspace(0, 1, n, endpoint=False)
    t = t1[:, :, None] + (t2 - t1)[:, :, None] * u[None, None, :]
    e = lambda a: a[:, :, None, None]
    tt = t[..., None]
    q0, q1, q2, q3 = p0[:, :, None, :], p1[:, :, None, :], p2[:, :, None, :], p3[:, :, None, :]
    A1 = (e(t1) - tt) / e(t1) * q0 + tt / e(t1) * q1
    A2 = (e(t2) - tt) / e(t2 - t1) * q1 + (tt - e(t1)) / e(t2 - t1) * q2
    A3 = (e(t3) - tt) / e(t3 - t2) * q2 + (tt - e(t2)) / e(t3 - t2) * q3
    B1 = (e(t2) - tt) / e(t2) * A1 + tt / e(t2) * A2
    B2 = (e(t3) - tt) / e(t3 - t1) * A2 + (tt - e(t1)) / e(t3 - t1) * A3
    C = (e(t2) - tt) / e(t2 - t1) * B1 + (tt - e(t1)) / e(t2 - t1) * B2
    return np.concatenate([C.reshape(B, (K - 1) * n, 2), P[:, -1:]], axis=1)


DENS = 45.0   # samples per unit of chord length (the digit height is about 1 unit, about 17 pixels)


def stroke(g, lst, corners=(), jit=0.02):
    """One pen stroke through the knots lst [(x, y), ...]; corners: knots where the pen turns sharply."""
    knots = g.pts(lst)
    K = knots.shape[1]
    kn = knots + jit * g.sloppy * g.rng.standard_normal(knots.shape)
    cuts = [0] + sorted(c for c in corners if 0 < c < K - 1) + [K - 1]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        sub = kn[:, a:b + 1]
        chord = np.sqrt(((sub[:, 1:] - sub[:, :-1]) ** 2).sum(-1)).max()
        out.append(crspline(sub, int(np.ceil(chord * DENS)) + 2))
    return np.concatenate(out, axis=1)


def ell(cx, cy, rx, ry, deg):
    """Point of an ellipse; deg is measured clockwise from the top, as on a clock face (y points down)."""
    a = np.deg2rad(deg)
    return (cx + rx * np.sin(a), cy - ry * np.cos(a))


# Coordinates: x to the right, y down; a digit is about 1 unit tall (y from 0 to 1) and w units wide.

def d0(g, style):
    w = g.U(0.45, 1.05)
    cx, cy, rx, ry = 0.5 * w, 0.5, 0.5 * w, 0.5
    egg = g.N(0, 0.10)                      # wider at the bottom (> 0) or at the top (< 0)
    sq = g.U(0.80, 1.10)                    # < 1 squarer, > 1 more pointed
    a0 = g.N(15, 25)                        # where the pen starts
    over = g.U(-35, 45)                     # overlap past the start (negative: a gap)
    rend = g.N(1.0, 0.10)
    rbeg = 1.0
    if style == 1:                          # the end curls inside
        rend = g.U(0.55, 0.85)
    elif style == 2:                        # the end overshoots outward past the start
        rend, over = g.U(1.15, 1.45), g.U(15, 60)
    elif style == 3:                        # the pen comes in from outside, upper right
        rbeg, a0, over = g.U(1.1, 1.3), g.N(40, 15), g.U(0, 45)
    K = 13
    lst = []
    for k in range(K):
        f = k / (K - 1)
        a = np.deg2rad(a0 - f * (360 + over))
        s, c = np.sin(a), -np.cos(a)
        rr = 1 + (rend - 1) * max(0.0, (f - 0.75) / 0.25) + (rbeg - 1) * (1.0 if k == 0 else 0.0)
        lst.append((cx + rx * np.sign(s) * np.abs(s) ** sq * (1 + egg * c) * rr, cy + ry * np.sign(c) * np.abs(c) ** sq * rr))
    return [stroke(g, lst, jit=0.018)]


def d1(g, style):
    bend = g.N(0, 0.035)
    if style == 0:                          # a bare stroke
        return [stroke(g, [(0, 0), (bend, 0.5), (0, 1)], jit=0.012)]
    fl = g.U(0.12, 0.38)                    # flag
    fa = np.deg2rad(g.U(25, 60))
    out = [stroke(g, [(-fl * np.sin(fa), fl * np.cos(fa)), (0, 0), (bend, 0.5), (0, 1)], corners=[1], jit=0.012)]
    if style == 2:                          # flag and foot
        bl, br = g.U(0.15, 0.4), g.U(0.15, 0.4)
        out.append(stroke(g, [(-bl, 1.0 + g.N(0, 0.02)), (br, 1.0 + g.N(0, 0.02))], jit=0.01))
    return out


def d2(g, style):
    w = g.U(0.55, 1.0)
    top = [(0.06 * w, g.U(0.2, 0.34)), (0.24 * w, 0.05), (0.55 * w, 0.0), (0.86 * w, 0.11), (0.92 * w, g.U(0.26, 0.36))]
    if style == 3:                          # starts with a small inward curl
        top = [(0.22 * w, g.U(0.3, 0.4))] + top
    by = g.U(0.5, 0.62)
    if style in (0, 3):                     # sharp corner, straight base
        rise = g.N(0.0, 0.04)
        lst = top + [(0.66 * w, by), (0.3 * w, 0.82), (0.02 * w, 1.0), (0.5 * w, 1.0 + g.N(0, 0.03) - rise * 0.3), (1.0 * w, 1.0 - rise)]
        return [stroke(g, lst, corners=[len(top) + 2])]
    if style == 1:                          # loop at the lower left
        lw = g.U(0.12, 0.3)
        lh = g.U(0.1, 0.22)
        lst = top + [(0.66 * w, by), (0.34 * w, 0.8), (0.12 * w, 0.95), (0.0 * w, 0.97 - 0.5 * lh), (0.1 * w, 0.97 - lh),
                     ((0.1 + lw) * w, 0.97 - 0.6 * lh), (0.55 * w, 0.95), (0.8 * w, 1.0), (1.0 * w, g.U(0.84, 0.98))]
        return [stroke(g, lst)]
    lst = top + [(0.66 * w, by), (0.3 * w, 0.83), (0.04 * w, 0.99), (0.3 * w, 0.93), (0.65 * w, 1.0), (1.0 * w, g.U(0.86, 0.98))]
    return [stroke(g, lst, corners=[len(top) + 2])]     # wavy base


def d3(g, style):
    w = g.U(0.5, 0.82)
    wy = g.U(0.4, 0.5)                      # waist
    wx = g.U(0.25, 0.5) * w
    ur = g.U(0.82, 1.0) * w                 # reach of the upper bump
    low = [(0.72 * w, wy + 0.04), (1.0 * w, 0.5 * (wy + 1) - 0.02), (0.85 * w, 0.93), (0.48 * w, 1.0), (0.12 * w, 0.95), (0.0 * w, g.U(0.8, 0.9))]
    if style == 0:
        up = [(0.05 * w, g.U(0.1, 0.2)), (0.3 * w, 0.02), (0.6 * w, 0.0), (ur, 0.12), (ur, 0.27), (0.7 * w, wy - 0.05), (wx, wy)]
        return [stroke(g, up + low, corners=[len(up) - 1])]
    up = [(0.05 * w, g.N(0.02, 0.02)), (ur, 0.0), (0.62 * w, 0.5 * wy), (wx, wy)]     # flat top
    return [stroke(g, up + low, corners=[1, len(up) - 1])]


def d4(g, style):
    w = g.U(0.6, 0.95)
    by = g.U(0.52, 0.72)                    # height of the bar
    xv = g.U(0.58, 0.8) * w                 # the vertical
    tilt = g.N(0.0, 0.035)
    lean = g.N(0.0, 0.04) * w
    ext = g.U(0.9, 1.1)
    if style in (0, 2):                     # open top
        xt = g.U(0.05, 0.32) * w            # keeps the top clearly open
        yt = g.U(0.0, 0.12)
        xe = xv + g.U(0.0, 0.42) * w        # the bar may stop at the vertical or run well past it
        if style == 0:
            s1 = stroke(g, [(xt, yt), (0.5 * (xt + 0.02 * w), 0.5 * (yt + by)), (0.02 * w, by), (0.5 * xe, by - 0.5 * tilt), (xe, by - tilt)], corners=[2])
        else:                               # rounded turn into the bar
            s1 = stroke(g, [(xt, yt), (0.5 * (xt + 0.03 * w), 0.5 * (yt + by) - 0.03), (0.06 * w, by - 0.06), (0.22 * w, by), (0.55 * xe, by - 0.5 * tilt), (xe, by - tilt)])
        s2 = stroke(g, [(xv + lean, g.U(-0.05, 0.3)), (xv + 0.5 * lean + g.N(0, 0.015), 0.6), (xv, 1.0)], jit=0.012)
        return [s1, s2]
    s1 = stroke(g, [(xv + lean + g.N(0, 0.03), g.U(0.0, 0.06)), (0.5 * (xv + lean), 0.5 * by - g.U(0, 0.05)), (0.0 * w, by), (0.5 * w, by - 0.5 * tilt), (ext * w, by - tilt)], corners=[2])
    s2 = stroke(g, [(xv + lean, 0.0), (xv + 0.5 * lean + g.N(0, 0.015), 0.55), (xv, 1.0)], jit=0.012)
    return [s1, s2]                         # closed top


def d5(g, style):
    w = g.U(0.55, 0.9)
    vy = g.U(0.33, 0.5)                     # where the vertical ends and the belly starts
    xt = g.U(0.12, 0.35) * w
    xb = xt - g.U(0.0, 0.15) * w
    br = g.U(0.75, 1.05) * w                # reach of the top bar
    bt = g.N(0.0, 0.035)
    mid = 0.5 * (vy + 1.0)
    belly = [(xb + 0.3 * w, vy - g.U(0.0, 0.07)), (0.78 * w, vy + 0.06), (0.97 * w, mid + 0.02), (0.82 * w, 0.93), (0.48 * w, 1.0), (0.14 * w, 0.95), (0.0 * w, g.U(0.8, 0.9))]
    if style in (0, 3):                     # body first, bar added afterwards
        body = stroke(g, [(xt, 0.0), (0.5 * (xt + xb), 0.5 * vy), (xb, vy)] + belly, corners=[2])
        if style == 0:
            bar = stroke(g, [(xt + g.N(0, 0.03), g.N(0.0, 0.02)), (br, -bt)], jit=0.012)
        else:                               # the bar does not touch the body
            bar = stroke(g, [(xt + g.U(0.08, 0.25) * w, g.U(-0.07, 0.03)), (br + g.U(0.0, 0.15) * w, -bt + g.U(-0.07, 0.03))], jit=0.012)
        return [body, bar]
    if style == 1:                          # one stroke starting at the right end of the bar
        return [stroke(g, [(br, -bt), (xt, 0.0), (0.5 * (xt + xb), 0.5 * vy), (xb, vy)] + belly, corners=[1, 3])]
    lst = [(br, -bt), (xt, 0.0), (0.5 * (xt + xb) - 0.02 * w, 0.5 * vy), (xb + 0.04 * w, vy - 0.02)] + belly[1:]
    return [stroke(g, lst, corners=[1])]    # rounded, S-like


def d6(g, style):
    w = g.U(0.5, 0.85)
    lt = g.U(0.36, 0.66)                    # top of the loop
    xs = g.U(0.45, 1.0) * w                 # where the pen starts
    cy, ry = 0.5 * (lt + 1.0), 0.5 * (1.0 - lt)
    cx, rx = 0.52 * w, 0.46 * w
    if style == 0:                          # curved stem
        stem = [(xs, 0.0), (0.5 * xs + 0.04 * w, g.U(0.12, 0.22)), (0.1 * w, 0.48)]
    else:                                   # straighter stem
        stem = [(xs, 0.0), (0.5 * xs + 0.12 * w, 0.26), (0.1 * w, 0.52)]
    loop = [(0.03 * w, cy)] + [ell(cx, cy, rx, ry, a) for a in (225, 180, 135, 90, 45, 0, -45)] + [(g.U(0.03, 0.2) * w, cy + g.U(-0.05, 0.12))]
    return [stroke(g, stem + loop)]


def d7(g, style):
    w = g.U(0.55, 0.95)
    fx = g.U(0.1, 0.65) * w                 # foot
    bow = g.N(0.0, 0.04)
    sag = g.N(0.0, 0.03)
    yl = g.U(0.0, 0.1)
    lst = [(0.0, yl), (0.5 * w, 0.5 * yl + sag), (1.0 * w, 0.0), (0.5 * (w + fx) + bow * w, 0.5), (fx, 1.0)]
    corners = [2]
    if style in (1, 3):                     # small tick at the start of the bar
        lst = [(g.N(0.0, 0.02), yl + g.U(0.1, 0.22))] + lst
        corners = [1, 3]
    out = [stroke(g, lst, corners=corners)]
    if style in (2, 3):                     # crossbar
        cyb = g.U(0.45, 0.62)
        xm = w + (fx - w) * cyb + bow * w * 0.8
        hl = g.U(0.2, 0.36) * w
        ct = g.N(0.0, 0.03)
        out.append(stroke(g, [(xm - hl, cyb + ct), (xm + hl, cyb - ct)], jit=0.012))
    return out


def d8(g, style):
    w = g.U(0.42, 0.75)
    yc = g.U(0.4, 0.56)                     # height of the crossing
    tw = g.U(0.7, 1.05)                     # width of the top loop relative to the bottom loop
    cx = 0.5 * w
    off = g.N(0.0, 0.05) * w
    ty, tr = 0.5 * yc, 0.5 * yc
    by, br = 0.5 * (yc + 1), 0.5 * (1 - yc)
    rxt, rxb = 0.5 * w * tw, 0.5 * w
    T = lambda d: ell(cx + off, ty, rxt, tr, d)
    Bm = lambda d: ell(cx, by, rxb, br, d)
    X = (cx + 0.3 * off, yc)
    a0 = g.U(20, 70)
    a1 = a0 + g.U(-25, 25)
    if style == 2:                          # the top loop is left open
        a0, a1 = g.U(50, 100), g.U(-20, 30)
    if style in (0, 2):                     # an S and its return stroke, crossing in the middle
        lst = [T(a0), T(0), T(-60), T(-105), X, Bm(75), Bm(120), Bm(180), Bm(240), Bm(285), X, T(105), T(60), T(a1)]
    else:                                   # two round loops
        lst = [T(a0), T(0), T(-50), T(-95), T(-135), X, Bm(50), Bm(95), Bm(140), Bm(180), Bm(220), Bm(265), Bm(310), X, T(135), T(95), T(a1)]
    return [stroke(g, lst, jit=0.018)]


def d9(g, style):
    w = g.U(0.5, 0.85)
    lb = g.U(0.45, 0.62)                    # bottom of the loop
    cx, cy, rx, ry = 0.46 * w, 0.5 * lb, 0.46 * w, 0.5 * lb
    xr = g.U(0.86, 1.0) * w
    a0 = g.U(5, 75)                         # small values leave the loop open at the top
    loop = [ell(cx, cy, rx, ry, a) for a in (a0, np.minimum(a0 - 25, 0), -50, -100, -150, -200, -245)] + [(xr, g.U(0.03, 0.15))]
    if style == 0:                          # straight tail
        fx = xr + g.N(-0.08, 0.1) * w
        lst = loop + [(0.5 * (xr + fx) + g.N(0, 0.035) * w, 0.58), (fx, 1.0)]
    else:                                   # hooked tail
        hx = g.U(0.05, 0.4) * w
        lst = loop + [(xr - 0.02 * w, 0.55), (0.8 * w, 0.85), (0.5 * (0.8 * w + hx), 0.99), (hx, g.U(0.88, 0.97))]
    return [stroke(g, lst, corners=[len(loop) - 1])]


# digit: (model, [(style, share of the templates), ...], mean slant, spread of slant)
MODELS = {
    0: (d0, [(0, 0.66), (1, 0.14), (2, 0.12), (3, 0.08)], 0.12, 0.2),
    1: (d1, [(0, 0.66), (1, 0.24), (2, 0.10)], 0.22, 0.3),
    2: (d2, [(0, 0.3), (1, 0.3), (2, 0.25), (3, 0.15)], 0.12, 0.2),
    3: (d3, [(0, 0.75), (1, 0.25)], 0.12, 0.2),
    4: (d4, [(0, 0.5), (2, 0.2), (1, 0.3)], 0.12, 0.2),
    5: (d5, [(0, 0.34), (1, 0.32), (2, 0.22), (3, 0.12)], 0.12, 0.2),
    6: (d6, [(0, 0.6), (1, 0.4)], 0.12, 0.2),
    7: (d7, [(0, 0.5), (1, 0.22), (2, 0.2), (3, 0.08)], 0.12, 0.2),
    8: (d8, [(0, 0.6), (1, 0.28), (2, 0.12)], 0.12, 0.2),
    9: (d9, [(0, 0.65), (1, 0.35)], 0.12, 0.2),
}


def render(pts, r, a, size=28):
    """Ink of a round pen of radius r dragged through pts (B, P, 2) (pixel units), edge ramp a pixels wide."""
    B = pts.shape[0]
    grid = np.arange(size, dtype=F)
    ax = (grid[None, :, None] - pts[:, None, :, 0].astype(F)) ** 2
    ay = (grid[None, :, None] - pts[:, None, :, 1].astype(F)) ** 2
    d2 = np.empty((B, size, size), F)
    for j in range(size):
        d2[:, j, :] = (ax + ay[:, j, None, :]).min(-1)
    return np.clip((r[:, None, None].astype(F) - np.sqrt(d2)) / a[:, None, None].astype(F) + F(0.5), 0, 1)


def draw(rng, digit, style, B):
    """B drawings (B, 28, 28) of one digit style, laid out the way the task describes its images:
    ink box scaled to 20 pixels, centre of mass at the centre of the frame."""
    model, _, slant_mu, slant_sd = MODELS[digit]
    g = G(rng, B)
    P = np.concatenate(model(g, style), axis=1)
    x, y = P[..., 0], P[..., 1]
    x = x - 0.5 * (x.min(1, keepdims=True) + x.max(1, keepdims=True))
    y = y - 0.5
    # smooth distortions of about a pixel: bowing, widening towards the top or bottom, sagging, fanning,
    # and a middle that sits higher or lower
    k = 0.2 * rng.standard_normal((5, B, 1))
    x, y = x + k[0] * (y * y - 0.08) + k[1] * x * y, y + k[2] * (x * x - 0.04) + k[3] * x * y + k[4] * (0.25 - y * y)
    s = np.clip(slant_mu + slant_sd * rng.standard_normal((B, 1)), slant_mu - 2.2 * slant_sd, slant_mu + 2.6 * slant_sd)
    x = x - s * y                                                  # slant: top to the right
    th = np.deg2rad(-2.0 + 6.0 * rng.standard_normal((B, 1)))       # horizontals tend to rise to the right
    x, y = np.cos(th) * x - np.sin(th) * y, np.sin(th) * x + np.cos(th) * y
    x = x * np.exp(0.10 * rng.standard_normal((B, 1)))
    r = np.clip(1.5 * np.exp(0.3 * rng.standard_normal(B)), 0.8, 3.0)      # pen radius in pixels
    a = rng.uniform(1.0, 1.6, B)                                           # softness of the edge
    sc = (20.0 - 2 * r) / np.maximum(x.max(1) - x.min(1), y.max(1) - y.min(1))
    x = (x - 0.5 * (x.max(1, keepdims=True) + x.min(1, keepdims=True))) * sc[:, None] + 13.5
    y = (y - 0.5 * (y.max(1, keepdims=True) + y.min(1, keepdims=True))) * sc[:, None] + 13.5
    pts = np.stack([x, y], -1)
    img = render(pts, r, a)
    m = img.sum((1, 2)).astype(np.float64)
    grid = np.arange(28)
    cxm = (img.sum(1) * grid).sum(1) / m
    cym = (img.sum(2) * grid).sum(1) / m
    # centre of mass at the centre of the frame; a whole-pixel placement leaves it anywhere within half a pixel,
    # and the centre of a 28-pixel frame may be counted as 13.5 or 14, so the target is spread over 13 .. 14.5
    pts[..., 0] += (rng.uniform(13.0, 14.5, B) - cxm)[:, None]
    pts[..., 1] += (rng.uniform(13.0, 14.5, B) - cym)[:, None]
    return render(pts, r, a)


def build_templates(w, per_class, batch=500, log=None):
    rng = np.random.default_rng(SEED)
    T_or = np.zeros((10 * per_class, N_ORI * 64), F)
    T_cc = np.zeros((10 * per_class, N_CC * 64), F)
    for digit in range(10):
        styles = MODELS[digit][1]
        counts = [int(round(per_class * share)) for _, share in styles]
        counts[0] += per_class - sum(counts)
        row = digit * per_class
        for (style, _), cnt in zip(styles, counts):
            done = 0
            while done < cnt:
                b = min(batch, cnt - done)
                img = draw(rng, digit, style, b)
                fo, fc = front_end(img, w)
                fo = fo.astype(np.float64)
                fc = fc.astype(np.float64) * CC_GAIN
                nrm = np.sqrt((fo ** 2).sum(1) + (fc ** 2).sum(1))[:, None]      # each drawing: its own norm
                T_or[row:row + b] = fo / nrm
                T_cc[row:row + b] = fc * CC_GAIN / nrm
                row += b
                done += b
        if log:
            log(f"digit {digit} done")
    return T_or, T_cc


def main(out_dir, per_class=PER_CLASS, group=GROUP):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w = front_end_weights()
    T_or, T_cc = build_templates(w, per_class, log=lambda s: print(s, flush=True))
    n = 10 * per_class
    w["t_or.W"], w["t_or.b"] = T_or, np.zeros(n, F)
    w["t_cc.W"], w["t_cc.b"] = T_cc, np.zeros(n, F)
    (out_dir / "network.json").write_text(json.dumps(network(n, per_class, group), indent=1))
    np.savez(out_dir / "weights.npz", **w)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
