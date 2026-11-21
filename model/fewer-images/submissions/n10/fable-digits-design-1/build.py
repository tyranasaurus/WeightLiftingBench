"""Builds a hand-designed digit classifier: python3 build.py OUT_DIR

Nothing is read from disk and nothing is fitted.  Every number comes from this file:

  * a fixed front end of formula filters (Sobel directional derivatives split into 8 gradient
    directions, a Gaussian blur sampled on an 8 x 8 grid, a piecewise-linear square root);
  * a bank of templates.  Each template is one digit drawn by this program from a pen path typed
    from general knowledge of how the ten digits are written (loops, bars, hooks, stems ...),
    with the writing style varied by a fixed-seed random number generator (slant, proportions,
    pen width, small wobble of the pen path).  Each drawing is pushed through the front end and
    becomes its own template, normalised by its own length only.  No statistic is ever taken
    across drawings.
  * the class score is the best match (maximum cosine similarity) among that class's templates.
"""
import sys, json
import numpy as np

SEED = 20261120
PER_CLASS = 23000          # templates per class
GRID = 8                   # 8 x 8 sampling grid of the blurred direction maps (centres at pixels 3, 6, ..., 24)
NDIR = 8                   # gradient directions
BLUR_K, BLUR_STRIDE, BLUR_PAD, BLUR_SIGMA = 11, 3, 2, 1.6
SQRT_STEP, SQRT_KNOTS = 0.07, 8
M_SUB = 5                  # line pieces per spline span
BOX = 20.0                 # digits are scaled to fit a 20 x 20 box ...
FRAME = 28                 # ... and placed by centre of mass in a 28 x 28 frame


# ----------------------------------------------------------------------------------------------
# the fixed front end (formula filters) and a numpy copy of the network operations
# ----------------------------------------------------------------------------------------------
def front_end():
    """Returns (layers, weights) of the feature part of the network."""
    W = {}
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0   # d/dx, unit gain on a unit ramp
    sy = sx.T
    ang = np.arange(NDIR) * (2 * np.pi / NDIR)
    W["c1.W"] = np.stack([np.cos(a) * sx + np.sin(a) * sy for a in ang])[:, None]  # (8, 1, 3, 3)
    W["c1.b"] = np.zeros(NDIR)
    # split the gradient between its two neighbouring directions (parallelogram rule):
    # |g| (cos d - |sin d|) = r[d] - r[d + 90 deg] - r[d - 90 deg] for half-rectified r
    w2 = np.zeros((NDIR, NDIR, 1, 1))
    for d in range(NDIR):
        w2[d, d, 0, 0] = 1.0
        w2[d, (d + 2) % NDIR, 0, 0] = -1.0
        w2[d, (d - 2) % NDIR, 0, 0] = -1.0
    W["c2.W"], W["c2.b"] = w2, np.zeros(NDIR)
    # Gaussian blur of each direction map, sampled every 3 pixels
    x = np.arange(BLUR_K) - (BLUR_K - 1) / 2.0
    g = np.exp(-x * x / (2 * BLUR_SIGMA ** 2)); g /= g.sum()
    w3 = np.zeros((NDIR, NDIR, BLUR_K, BLUR_K))
    for d in range(NDIR):
        w3[d, d] = np.outer(g, g)
    W["c3.W"], W["c3.b"] = w3, np.zeros(NDIR)
    # piecewise-linear square root: knots at (k s)^2, value k s there
    s = SQRT_STEP
    knots = (np.arange(SQRT_KNOTS) * s) ** 2
    slopes = 1.0 / (s * (2 * np.arange(SQRT_KNOTS) + 1))
    coef = np.diff(np.concatenate([[0.0], slopes]))
    w4 = np.zeros((NDIR * SQRT_KNOTS, NDIR, 1, 1)); b4 = np.zeros(NDIR * SQRT_KNOTS)
    w5 = np.zeros((NDIR, NDIR * SQRT_KNOTS, 1, 1))
    for d in range(NDIR):
        for k in range(SQRT_KNOTS):
            w4[d * SQRT_KNOTS + k, d, 0, 0] = 1.0
            b4[d * SQRT_KNOTS + k] = -knots[k]
            w5[d, d * SQRT_KNOTS + k, 0, 0] = coef[k]
    W["c4.W"], W["c4.b"] = w4, b4
    W["c5.W"], W["c5.b"] = w5, np.zeros(NDIR)
    layers = [
        {"name": "c1", "op": "conv", "inputs": ["image"], "out": NDIR, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "c2", "op": "conv", "inputs": ["r1"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "r2", "op": "relu", "inputs": ["c2"]},
        {"name": "c3", "op": "conv", "inputs": ["r2"], "out": NDIR, "kernel": [BLUR_K, BLUR_K], "stride": BLUR_STRIDE, "padding": BLUR_PAD},
        {"name": "c4", "op": "conv", "inputs": ["c3"], "out": NDIR * SQRT_KNOTS, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "r4", "op": "relu", "inputs": ["c4"]},
        {"name": "c5", "op": "conv", "inputs": ["r4"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
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


def features(imgs, layers, W):
    """imgs (n, 28, 28) -> (n, 512): the front end applied to drawings, in float64."""
    x = imgs[:, None]
    for L in layers:
        if L["op"] == "conv":
            x = conv(x, W[L["name"] + ".W"], W[L["name"] + ".b"], L["stride"], L["padding"])
        else:
            x = np.maximum(x, 0)
    return x.reshape(len(x), -1)


# ----------------------------------------------------------------------------------------------
# drawing: pen paths -> 28 x 28 images made the way the task describes
# ----------------------------------------------------------------------------------------------
class Pen:
    """Random style source for a batch of drawings of one way of writing a digit."""

    def __init__(self, rs, n):
        self.rs, self.n = rs, n
        self.wob = 0.008 + 0.03 * rs.random_sample(n)       # how unsteady this writer's hand is
        self.hole = None                                    # (width, height) of the narrowest loop or opening, drawing units

    def U(self, lo, hi):
        return lo + (hi - lo) * self.rs.random_sample(self.n)

    def N(self):                                             # bell-shaped, in [-3, 3], unit spread
        return (self.rs.random_sample((3, self.n)).sum(0) - 1.5) * 2.0

    def pt(self, x, y, wob=1.0):
        j = (self.rs.random_sample((self.n, 2)) - 0.5) * 2.0 * (self.wob * wob)[:, None]
        return np.stack([np.broadcast_to(np.asarray(x, dtype=np.float64), (self.n,)),
                         np.broadcast_to(np.asarray(y, dtype=np.float64), (self.n,))], axis=1) + j


def ell(g, cx, cy, rx, ry, deg, r=1.0, wob=1.0):
    """Point on an ellipse; deg measured with y pointing down (0 right, 90 bottom, 180 left, -90 top)."""
    a = np.deg2rad(deg)
    return g.pt(cx + r * rx * np.cos(a), cy + r * ry * np.sin(a), wob)


def spline(ctrl, m=M_SUB):
    """ctrl: list of (n, 2) control points -> (n, (K-1) m + 1, 2) points on a Catmull-Rom spline through them."""
    C = np.stack(ctrl, axis=1)
    n, K, _ = C.shape
    E = np.concatenate([2 * C[:, :1] - C[:, 1:2], C, 2 * C[:, -1:] - C[:, -2:-1]], axis=1)
    P0, P1, P2, P3 = E[:, :-3, None], E[:, 1:-2, None], E[:, 2:-1, None], E[:, 3:, None]
    t = (np.arange(m) / m)[None, None, :, None]
    pts = 0.5 * (2 * P1 + (P2 - P0) * t + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t ** 2 + (3 * P1 - P0 - 3 * P2 + P3) * t ** 3)
    return np.concatenate([pts.reshape(n, (K - 1) * m, 2), C[:, -1:]], axis=1)


# ---- the ten digits.  x runs 0 (left) .. 1 (right), y runs 0 (top) .. 1 (bottom). --------------

def zero(g, tail=False, lead=False):
    """An oval drawn counter-clockwise from the top; the ends may leave a gap or overlap."""
    a0 = g.U(-35, 35)                       # where the pen starts, degrees from the top
    over = g.U(-35, 45)                     # < 0: gap, > 0: overlap
    drift = g.U(-0.14, 0.14)                # radius drift so that the two ends do not meet exactly
    sq = g.U(0.82, 1.18)                    # squarish .. pointed oval
    K = 13
    pts = []
    g.hole = (1.0, 1.0)
    for i in range(K):
        f = i / (K - 1.0)
        a = np.deg2rad(a0 + f * (360.0 + over))
        r = 1.0 + drift * (f - 0.5)
        s, c = np.sin(a), np.cos(a)
        pts.append(g.pt(0.5 - 0.5 * r * np.sign(s) * np.abs(s) ** sq, 0.5 - 0.5 * r * np.sign(c) * np.abs(c) ** sq))
    if tail:                                # the pen runs on past the closing point
        a = np.deg2rad(a0 + 360.0 + over + g.U(15, 40))
        r = 1.0 + drift * 0.5 + g.U(-0.45, 0.3)
        pts.append(g.pt(0.5 - 0.5 * r * np.sin(a), 0.5 - 0.5 * r * np.cos(a)))
    if lead:                                # the pen comes in from outside, above and to the right of the start
        a = np.deg2rad(a0 - g.U(8, 30))
        r = 1.0 - 0.5 * drift + g.U(0.15, 0.4)
        pts.insert(0, g.pt(0.5 - 0.5 * r * np.sin(a), 0.5 - 0.5 * r * np.cos(a)))
    return [pts]


def one(g, flag=False, base=False):
    """A stem; some writers add a flag at the top and a few add a foot."""
    bow = g.U(-0.05, 0.05)
    top, bot = g.pt(0.5, 0.0), g.pt(0.5, 1.0)
    out = [[top, g.pt(0.5 + bow, 0.5, 0.5), bot]]
    if flag:
        fl, fa = (g.U(0.14, 0.35) if base else g.U(0.14, 0.45)), np.deg2rad(g.U(32, 65))
        out.append([g.pt(0.5 - fl * np.cos(fa), fl * np.sin(fa)), top])
        g.hole = (fl * np.cos(fa), 9.0)
    if base:
        out.append([g.pt(0.5 - g.U(0.2, 0.45), 1.0), g.pt(0.5 + g.U(0.2, 0.45), 1.0)])
    return out


def two(g, loop=False, flat_top=False):
    """Hook over the top, a sweep down to the bottom left, then a foot to the right."""
    if flat_top:                            # angular, Z-like top
        s = g.pt(g.U(0.03, 0.2), g.U(0.05, 0.2))
        head = [s, g.pt(0.45, g.U(-0.02, 0.05)), g.pt(g.U(0.8, 0.95), g.U(0.02, 0.12))]
        shoulder = g.pt(g.U(0.84, 0.96), g.U(0.16, 0.3))
    else:
        hx, hy = g.U(0.04, 0.26), g.U(0.16, 0.4)
        head = [g.pt(hx, hy), g.pt(hx + 0.14, 0.05), g.pt(0.58, 0.0), g.pt(0.88, 0.14)]
        if g.rs.random_sample() < 0.3:      # curled-in start of the hook
            head.insert(0, g.pt(hx + g.U(0.08, 0.2), hy + g.U(0.03, 0.14)))
        shoulder = g.pt(g.U(0.82, 0.95), g.U(0.26, 0.42))
    cx = g.U(0.0, 0.16)
    by = g.U(0.8, 1.0)                      # height of the bottom-left turn: the foot may run level, rise, or drop away from it
    nb = g.U(-0.1, 0.1)                     # neck: bowed in or out
    ey = g.U(0.74, 1.03)
    ex = g.U(0.82, 1.3)
    sag = g.U(-0.04, 0.08)                  # foot sags or humps in the middle
    sh = shoulder
    if loop:
        lw, lh = g.U(0.1, 0.42), g.U(0.07, 0.3)     # loop width and height: a small eye .. a large round loop
        bx = cx + 0.5 * lw                           # bottom of the loop, where the sweep arrives
        d1 = g.pt(bx + 0.62 * (0.87 - bx) + nb, 0.33 + 0.38 * (by - 0.33)); d2 = g.pt(bx + 0.25 * (0.87 - bx) + nb * 0.5, 0.33 + 0.75 * (by - 0.33))
        la = g.pt(bx, by); lb = g.pt(cx, by - 0.5 * lh); lc = g.pt(bx, by - lh)
        ld = g.pt(cx + lw + 0.04, by - 0.55 * lh + g.U(-0.02, 0.03)); le = g.pt(0.5 * (cx + lw + 0.04 + ex), 0.5 * (by - 0.3 * lh + ey) + sag)
        return [head + [sh, d1, d2, la, lb, lc, ld, le, g.pt(ex, ey)]]
    c = g.pt(cx, by)
    d1 = g.pt(0.87 - 0.35 * (0.87 - cx) + nb, 0.33 + 0.35 * (by - 0.33)); d2 = g.pt(0.87 - 0.72 * (0.87 - cx) + nb * 0.6, 0.33 + 0.72 * (by - 0.33))
    foot = [c, g.pt(0.5 * (cx + ex), 0.5 * (by + ey) + sag), g.pt(ex, ey)]
    if g.rs.random_sample() < 0.35:         # rounded corner instead of a sharp one
        return [head + [sh, d1, d2, g.pt(cx + 0.03, by - 0.03)] + foot[1:]]
    return [head + [sh, d1, d2, c], foot]


def three(g, flat_top=False, loop=False):
    """Two bowls open to the left, meeting at a waist."""
    mx, my = g.U(0.3, 0.6), g.U(0.34, 0.54)
    m = g.pt(mx, my)
    if flat_top:                            # a bar, then a diagonal back to the waist
        a = g.pt(g.U(-0.1, 0.25), g.U(0.0, 0.08)); b = g.pt(g.U(0.78, 1.0), g.U(-0.03, 0.05))
        upper = [[a, b], [b, g.pt(0.5 * (0.86 + mx) + g.U(0.0, 0.1), 0.5 * my), m]]
        top_open = my - 0.05
        lr = g.U(0.62, 1.0)
    else:
        ur = g.U(0.7, 0.95)                 # how far right the upper bowl reaches
        sx, sy = g.U(0.0, 0.3), g.U(0.06, 0.24)
        up = [g.pt(sx, sy), g.pt(0.5 * (sx + 0.8 * ur), 0.01), g.pt(0.8 * ur, 0.03), g.pt(ur, 0.45 * my), g.pt(ur - 0.14, 0.82 * my), m]
        top_open = my - sy
        if g.rs.random_sample() < 0.3:      # curled-in start
            cd = g.U(0.06, 0.16)
            up.insert(0, g.pt(sx + g.U(0.04, 0.14), sy + cd))
            top_open = top_open - cd
        upper = [up]
        lr = g.U(0.8, 1.0)
    lr = np.maximum(lr, mx + 0.2)           # the lower bowl always bulges right of the waist
    ex, ey = g.U(-0.2, 0.2), g.U(0.76, 1.0)  # the tail may sweep far to the left along the bottom
    bxm = np.minimum(g.U(0.3, 0.55), lr - 0.25)
    lower = [g.pt(0.5 * (mx + lr) + 0.08, my + 0.04), g.pt(lr, 0.5 * (my + 1.0) + 0.02), g.pt(lr - 0.16, 0.94), g.pt(bxm, 1.0),
             g.pt(0.4 * bxm + 0.6 * ex, np.minimum(0.35 + 0.65 * ey + 0.03, 1.0)), g.pt(ex, ey)]
    low_open = ey - my
    if g.rs.random_sample() < 0.3:
        cd = g.U(0.06, 0.14)
        lower.append(g.pt(ex + g.U(0.0, 0.08), ey - cd))                    # curl at the end
        low_open = low_open - cd
    g.hole = (9.0, np.minimum(top_open, low_open))
    if loop:                                # the pen turns through a small loop at the waist
        k = g.pt(mx - g.U(0.04, 0.12), my + g.U(-0.02, 0.06))
        return upper[:-1] + [upper[-1] + [k] + lower]
    return upper + [[m] + lower]


def four(g, closed=False, round_corner=False):
    """Left stroke down to a bar, bar to the right, and a stem through the bar."""
    hb = g.U(0.45, 0.74)                    # height of the bar
    vx = g.U(0.55, 0.82)                    # where the stem crosses the bar
    bx = g.U(0.0, 0.16)
    tilt = g.U(-0.1, 0.06)
    lean = g.U(-0.06, 0.1)                  # stem top is right of its foot
    corner = g.pt(bx, hb)
    bar_end = g.pt(np.minimum(vx + g.U(0.0, 0.32), 1.08), hb + tilt)
    foot = g.pt(vx - lean * 0.6, 1.0)
    if closed:
        top = g.pt(vx + lean * 0.6, 0.0)
        g.hole = (0.5 * (vx - bx), 0.5 * hb)
        return [[top, corner], [corner, bar_end], [top, g.pt(vx, hb + tilt * 0.6, 0.5), foot]]
    ax = np.minimum(bx + g.U(0.02, 0.34), vx - 0.25)
    a = g.pt(ax, g.U(0.0, 0.12))
    g.hole = (vx - ax, 9.0)
    top = g.pt(vx + lean * 0.6, g.U(-0.02, 0.36))
    stem = [top, g.pt(vx, hb + tilt * 0.6, 0.5), foot]
    if round_corner:
        return [[a, g.pt(bx + 0.25 * (ax - bx), hb - 0.2), g.pt(bx + 0.04, hb - 0.03), g.pt(bx + 0.2, hb + tilt * 0.2), bar_end], stem]
    return [[a, corner], [corner, bar_end], stem]


def five(g, style=0):
    """Top bar, short down stroke, and a bowl open to the left.  style 0: corners, 1: loose bar, 2: rounded."""
    tx = g.U(0.14, 0.38)
    bl = g.U(0.35, 1.0)
    rise = g.U(-0.25, 0.05)
    vy = g.U(0.3, 0.58)
    vx = tx - g.U(-0.03, 0.22)
    t0 = g.pt(tx, 0.03); t1 = g.pt(tx + bl, 0.03 + rise)
    v = g.pt(vx, vy)
    br = g.U(0.65, 1.0)                     # how far right the bowl reaches
    ex, ey = g.U(0.0, 0.16), g.U(0.78, 0.94)
    bowl = [g.pt(0.5 * (vx + br), vy - g.U(0.03, 0.1)), g.pt(br - 0.07, vy + 0.1), g.pt(br, 0.5 * (vy + 1.0) + 0.04),
            g.pt(br - 0.18, 0.94), g.pt(0.42, 1.0), g.pt(ex + 0.07, ey + 0.05), g.pt(ex, ey)]
    low_open = ey - vy
    if g.rs.random_sample() < 0.25:
        cd = g.U(0.06, 0.16)
        bowl.append(g.pt(ex + g.U(0.0, 0.1), ey - cd))                      # bowl curls back up
        low_open = low_open - cd
    g.hole = (9.0, np.minimum(low_open, vy - 0.03))
    if style == 2:
        return [[t1, g.pt(tx + 0.5 * bl, 0.03 + 0.5 * rise), t0, g.pt(0.5 * (tx + vx) - 0.02, 0.5 * vy), v] + bowl]
    if style == 1:                          # bar written separately: it may miss the down stroke and fly off to the right
        b0 = g.pt(tx + g.U(-0.04, 0.2), 0.03 + g.U(-0.05, 0.08))
        b1 = g.pt(tx + bl + g.U(0.0, 0.3), 0.03 + rise + g.U(-0.1, 0.02))
        return [[b0, b1], [t0, v], [v] + bowl]
    return [[t1, t0], [t0, v], [v] + bowl]


def six(g, curl=False, open_loop=False):
    """A stem from the top right curving down into a closed loop at the bottom."""
    lh = g.U(0.36, 0.64)                    # loop height
    lw = g.U(0.6, 1.0)                      # loop width
    sx = g.U(0.2, 0.8)                      # where the pen starts at the top
    g.hole = (lw, lh)
    bow = g.U(0.0, 0.16)
    cx, cy, rx, ry = 0.02 + lw / 2, 1.0 - lh / 2, lw / 2, lh / 2
    pts = []
    if curl:
        pts.append(g.pt(sx + g.U(0.06, 0.16), g.U(0.08, 0.2)))
    pts += [g.pt(sx, 0.0), g.pt(0.5 * (sx + 0.02) - bow, 0.5 * cy)]
    for a in (180, 135, 90, 45, 0, -45, -90):
        pts.append(ell(g, cx, cy, rx, ry, a))
    if open_loop:                           # the pen stops before it gets back to the stem
        pts.append(ell(g, cx, cy, rx, ry, g.U(-140, -105), g.U(0.9, 1.05)))
        return [pts]
    pts.append(ell(g, cx, cy, rx, ry, -135))
    endr = g.U(0.55, 1.05)                  # the loop closes on the stem or spirals inside it
    pts.append(ell(g, cx, cy, rx, ry, g.U(-185, -150), endr))
    return [pts]


def seven(g, serif=False, cross=False, round_corner=False):
    """A bar along the top and a stem down from its right end."""
    lx, ly = g.U(0.0, 0.14), g.U(-0.03, 0.18)
    rx = g.U(0.84, 1.0)
    fx = g.U(0.15, 0.85)
    bow = g.U(-0.04, 0.12)
    l, r, f = g.pt(lx, ly), g.pt(rx, 0.02), g.pt(fx, 1.0)
    mid = g.pt(0.5 * (rx + fx) - bow, 0.5)
    barmid = g.pt(0.5 * (lx + rx), 0.5 * ly + g.U(-0.04, 0.04))
    if round_corner:
        out = [[l, barmid, r, g.pt(rx - 0.25 * (rx - fx) - 0.5 * bow + 0.03, 0.24), mid, f]]
    else:
        out = [[l, barmid, r], [r, mid, f]]
    if serif:
        out.append([g.pt(lx + g.U(-0.03, 0.03), ly + g.U(0.12, 0.28)), l])
    if cross:
        mx, cyy = 0.5 * (rx + fx) - bow, g.U(0.42, 0.6)
        cl = g.U(0.18, 0.36)
        out.append([g.pt(mx - cl, cyy + g.U(-0.03, 0.05)), g.pt(mx + g.U(0.14, 0.32), cyy + g.U(-0.06, 0.03))])
        g.hole = (2.0 * cl, cyy - 0.05)
    return out


def eight(g, circles=False, tail=False, mirror=False, pointed=False):
    """Two loops, one above the other, usually drawn as one S-shaped stroke that returns to its start."""
    th = g.U(0.38, 0.54)                    # where the two loops meet
    tw, bw = g.U(0.5, 0.86), g.U(0.62, 1.0)
    tcx, bcx = 0.5 + g.U(-0.06, 0.1), 0.5 + g.U(-0.08, 0.06)
    T = (tcx, th / 2, tw / 2, th / 2)
    B = (bcx, (1 + th) / 2, bw / 2, (1 - th) / 2)
    g.hole = (np.minimum(tw, bw), np.minimum(th, 1 - th))
    if circles:                             # two separate rings
        return [[ell(g, *T, a) for a in (-90, -135, -180, -225, -270, -315, -360, -405, -450)],
                [ell(g, *B, a) for a in (-90, -135, -180, -225, -270, -315, -360, -405, -450)]]
    a0 = g.U(-75, -25)                      # start on the upper right of the top loop
    if pointed:                             # straight diagonals through the crossing: teardrop loops
        pts = [ell(g, *T, a0)] + [ell(g, *T, a) for a in (-100, -140, -180)]
        pts += [ell(g, *B, a) for a in (0, 45, 90, 135, 180)]
        pts += [ell(g, *T, 0), ell(g, *T, a0 + g.U(-25, 45), g.U(0.85, 1.1))]
        return [pts]
    pts = [ell(g, *T, a0)] + [ell(g, *T, a) for a in (-100, -140, -180, -222)]
    pts += [ell(g, *B, a) for a in (-48, 0, 45, 90, 135, 180, 228)]
    if tail:                                # the return stroke does not curl over to close the top loop: it runs on, up and to the right
        pts.append(g.pt(tcx + tw * g.U(0.1, 0.25), th * g.U(0.62, 0.8)))
        pts.append(g.pt(tcx + tw / 2 * g.U(0.7, 1.4), th * g.U(-0.05, 0.3)))
    else:
        pts += [ell(g, *T, a) for a in (42, 0)]
        pts.append(ell(g, *T, a0 + g.U(-25, 45), g.U(0.85, 1.1)))            # closes, overlaps, or stops short
    if mirror:                              # started towards the right instead: the same figure, left and right swapped
        pts = [np.stack([1.0 - p[:, 0], p[:, 1]], axis=1) for p in pts]
    return [pts]


def nine(g, hook=False, open_top=False):
    """A loop at the top with a stem down its right side; the stem may end in a hook."""
    lh = g.U(0.34, 0.64)
    lw = g.U(0.45, 1.0)
    xr = g.U(0.78, 0.98)
    cx, cy, rx, ry = xr - lw / 2, lh / 2, lw / 2, lh / 2
    g.hole = (lw, lh)
    a0 = g.U(-75, -35) if open_top else g.U(-45, -5)
    rest = (-105, -140, -180, -225, -270, -315) if open_top else (-90, -135, -180, -225, -270, -315)
    loop = [ell(g, cx, cy, rx, ry, a0)] + [ell(g, cx, cy, rx, ry, a) for a in rest]
    fx = xr - g.U(-0.04, 0.5)               # foot of the stem, usually left of the loop's right side
    if open_top:
        top = g.pt(xr + g.U(-0.02, 0.08), g.U(-0.02, 0.12))
        loop.append(ell(g, cx, cy, rx, ry, g.U(-350, -330)))
    else:
        top = loop[0]
        loop.append(ell(g, cx, cy, rx, ry, g.U(-360, -335)))
    j = g.pt(xr - 0.02, cy + 0.08, 0.5)
    if hook:
        hx = g.U(0.3, 0.75)
        stem = [top, j, g.pt(xr - 0.06, 0.74), g.pt(xr - 0.5 * hx, 0.97), g.pt(xr - hx, 0.98 - g.U(0.0, 0.12))]
    else:
        stem = [top, j, g.pt(0.5 * (xr + fx) + g.U(-0.02, 0.05), 0.5 * (cy + 1.0) + 0.04), g.pt(fx, 1.0)]
    return [loop, stem]


# (function, keyword arguments, share of the class's templates)
WAYS = {
    0: [(zero, {}, 0.6), (zero, {"tail": True}, 0.2), (zero, {"lead": True}, 0.2)],
    1: [(one, {}, 0.68), (one, {"flag": True}, 0.26), (one, {"flag": True, "base": True}, 0.06)],
    2: [(two, {}, 0.45), (two, {"loop": True}, 0.35), (two, {"flat_top": True}, 0.12), (two, {"flat_top": True, "loop": True}, 0.08)],
    3: [(three, {}, 0.55), (three, {"flat_top": True}, 0.25), (three, {"loop": True}, 0.12), (three, {"flat_top": True, "loop": True}, 0.08)],
    4: [(four, {}, 0.5), (four, {"round_corner": True}, 0.25), (four, {"closed": True}, 0.25)],
    5: [(five, {"style": 0}, 0.4), (five, {"style": 1}, 0.35), (five, {"style": 2}, 0.25)],
    6: [(six, {}, 0.7), (six, {"curl": True}, 0.2), (six, {"open_loop": True}, 0.1)],
    7: [(seven, {}, 0.4), (seven, {"serif": True}, 0.15), (seven, {"cross": True}, 0.2), (seven, {"round_corner": True}, 0.15),
        (seven, {"serif": True, "cross": True}, 0.1)],
    8: [(eight, {}, 0.38), (eight, {"tail": True}, 0.22), (eight, {"pointed": True}, 0.2), (eight, {"circles": True}, 0.08), (eight, {"mirror": True}, 0.12)],
    9: [(nine, {}, 0.55), (nine, {"hook": True}, 0.25), (nine, {"open_top": True}, 0.2)],
}
# width of the unit drawing square relative to its height: (narrowest, widest)
ASPECT = {0: (0.42, 1.1), 1: (0.5, 1.0), 2: (0.55, 1.3), 3: (0.5, 1.15), 4: (0.55, 1.15),
          5: (0.5, 1.4), 6: (0.45, 0.95), 7: (0.5, 1.2), 8: (0.42, 0.85), 9: (0.42, 0.9)}
# slant: (centre, spread) of the shear, positive leans the top to the right
SLANT = {d: (0.15, 0.24) for d in range(10)}
SLANT[1] = (0.25, 0.33)
WAY_SLANT = {(1, 2): (0.04, 0.07)}         # the flag-and-foot 1 stands upright


def draw(digit, way, n, rs):
    """n drawings of one way of writing `digit` -> (n, 28, 28) images in [0, 1]."""
    fn, kw, _ = WAYS[digit][way]
    g = Pen(rs, n)
    strokes = [spline(s) for s in fn(g, **kw)]
    lo, hi = ASPECT[digit]
    asp = g.U(lo, hi)
    wq = [0.18 * g.N()[:, None] for _ in range(6)]   # smooth uneven proportions of the whole figure (second-order warp)
    sl_c, sl_s = WAY_SLANT.get((digit, way), SLANT[digit])
    sl = sl_c + sl_s * g.N()
    rot = np.deg2rad(4.0) * g.N()
    width = 1.3 + 4.0 * g.U(0, 1) ** 2      # pen width in pixels of the final image
    if g.hole is not None:                  # writers keep loops and openings open (a drawing unit is about 17 pixels)
        gap = 17.0 * np.minimum(g.hole[0] * asp, g.hole[1])
        width = np.minimum(width, np.minimum(1.0 + 0.5 * gap, np.maximum(gap - 1.0, 1.3)))
    ramp = g.U(0.9, 1.6)                    # softness of the ink edge
    centre = np.where(g.U(0, 1) < 0.5, 13.5, 14.0)   # frame centre: between the middle pixels, or the 15th pixel
    A, Bp = [], []
    lo_xy = np.full((n, 2), np.inf); hi_xy = np.full((n, 2), -np.inf)
    out = []
    for p in strokes:
        x, y = p[..., 0], p[..., 1]
        u, v = x - 0.5, y - 0.5
        xx = x + wq[0] * u * v + wq[1] * (0.25 - v * v) + wq[2] * (u * u - 0.25)   # wider top or bottom, bent spine, uneven halves
        yy = y + wq[3] * (0.25 - v * v) + wq[4] * u * v + wq[5] * (0.25 - u * u)   # larger upper or lower part, fanned, arched
        xx = xx * asp[:, None]
        xx = xx + sl[:, None] * (0.5 - yy)                                   # slant
        c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
        q = np.stack([c * xx - s * yy, s * xx + c * yy], axis=-1)
        out.append(q)
        lo_xy = np.minimum(lo_xy, q.min(1)); hi_xy = np.maximum(hi_xy, q.max(1))
    size = hi_xy - lo_xy                                                     # pen-path extent
    scale = (BOX - width) / np.maximum(size.max(1), 1e-6)                    # ink (path + pen width) fits the 20 x 20 box
    ink = size * scale[:, None] + width[:, None]
    off = (BOX - ink) / 2.0 + width[:, None] / 2.0                           # centred in the box along the shorter side
    for q in out:
        q = (q - lo_xy[:, None]) * scale[:, None, None] + off[:, None]
        A.append(q[:, :-1]); Bp.append(q[:, 1:])
    A, Bp = np.concatenate(A, 1), np.concatenate(Bp, 1)                      # (n, S, 2) line pieces, box coordinates
    # distance from every pixel centre of the box (plus a one pixel rim) to the nearest line piece
    c1 = np.arange(-1, 21) + 0.5
    ex, ey = Bp[..., 0] - A[..., 0], Bp[..., 1] - A[..., 1]
    inv = 1.0 / np.maximum(ex * ex + ey * ey, 1e-12)
    dx = c1[None, None, :] - A[..., 0, None]                                 # (n, S, 22)
    dy = c1[None, None, :] - A[..., 1, None]
    t = (dx * ex[..., None])[:, :, None, :] + (dy * ey[..., None])[:, :, :, None]   # (n, S, 22 rows, 22 cols)
    t *= inv[:, :, None, None]
    np.clip(t, 0.0, 1.0, out=t)
    rx_ = dx[:, :, None, :] - t * ex[:, :, None, None]
    ry_ = dy[:, :, :, None] - t * ey[:, :, None, None]
    d2 = (rx_ * rx_ + ry_ * ry_).min(1)                                      # (n, 22, 22)
    box = np.clip((width[:, None, None] / 2.0 - np.sqrt(d2)) / ramp[:, None, None] + 0.5, 0.0, 1.0)
    # place in the frame: whole-pixel shift that brings the centre of mass nearest to the frame centre
    mass = box.sum((1, 2))
    com_y = (box.sum(2) * np.arange(22)).sum(1) / mass
    com_x = (box.sum(1) * np.arange(22)).sum(1) / mass
    oy = np.floor(centre - com_y + 0.5).astype(int); ox = np.floor(centre - com_x + 0.5).astype(int)
    imgs = np.zeros((n, FRAME + 44, FRAME + 44))
    for i in range(n):
        imgs[i, 22 + oy[i]:44 + oy[i], 22 + ox[i]:44 + ox[i]] = box[i]
    return imgs[:, 22:22 + FRAME, 22:22 + FRAME]


def class_drawings(digit, total, chunk=100):
    """Yields batches of drawings of `digit`, split between the ways of writing it."""
    ways = WAYS[digit]
    counts = [int(round(total * w[2])) for w in ways]
    counts[0] += total - sum(counts)
    for wi, cnt in enumerate(counts):
        rs = np.random.RandomState(SEED + 1000 * digit + wi)
        done = 0
        while done < cnt:
            n = min(chunk, cnt - done)
            yield draw(digit, wi, n, rs)
            done += n


def build(out_dir, per_class=PER_CLASS):
    layers, W = front_end()
    bank = np.zeros((10 * per_class, NDIR * GRID * GRID), dtype=np.float32)
    row = 0
    for digit in range(10):
        for imgs in class_drawings(digit, per_class):
            f = features(imgs, layers, W)
            f = f / np.sqrt((f * f).sum(1, keepdims=True))                   # each template normalised by itself alone
            bank[row:row + len(f)] = f
            row += len(f)
    assert row == len(bank)
    weights = {k: v.astype(np.float32) for k, v in W.items()}
    weights["fc.W"] = bank
    weights["fc.b"] = np.zeros(len(bank), dtype=np.float32)
    layers = layers + [
        {"name": "fc", "op": "dense", "inputs": ["c5"], "out": 10 * per_class},
        {"name": "byclass", "op": "reshape", "inputs": ["fc"], "shape": [1, 10, per_class]},
        {"name": "best", "op": "maxpool", "inputs": ["byclass"], "size": [1, per_class], "stride": [1, per_class]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]},
    ]
    with open(out_dir + "/network.json", "w") as fh:
        json.dump({"layers": layers}, fh, indent=1)
    np.savez(out_dir + "/weights.npz", **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
