"""Builds the digit network: python3 build.py OUT_DIR  ->  OUT_DIR/network.json, OUT_DIR/weights.npz

Nothing here is trained or fitted, and no file is read.  The weights come from two sources only:

1. Formulas.  The front end is a fixed bank of Gaussian-derivative edge filters at 8 directions, a
   direction-sharpening step, a clip, Gaussian pooling to a 7 x 7 grid and a piecewise-linear square root.

2. Typed pen paths.  Each digit is described below as strokes through control points that I typed from
   general knowledge of how the ten digits are written (several writing styles per digit).  A seeded random
   number generator varies the shape parameters, slant, proportions and pen width; every drawing is rendered
   to a 28 x 28 image, passed through the same formula front end, divided by its own length, and stored as
   one row of the dense layer: one drawing = one template, normalised by itself alone.

The network scores a class by how well its best templates match (max over groups of templates, then mean).
"""
import sys, json
import numpy as np

SEED = 7
PER_CLASS = 30000          # templates per class
GROUPS = 4                 # per class: max inside each group of templates, then mean over the groups
NDIR = 8                   # edge directions
SIG_D = 1.0                # sigma of the Gaussian-derivative edge filter (pixels)
KD = 7                     # its kernel size
BETA = 1.0                 # direction sharpening: response_k - BETA * (responses at +-90 degrees)
CLIP = 0.35                # edge responses are clipped at this fraction of a full-contrast step edge
SIG_P = 1.8                # sigma of the Gaussian pooling window (pixels)
KP, SP, PP = 12, 4, 4      # pooling kernel, stride, padding  -> 7 x 7 cells
KNOTS = (1 / 64, 1 / 16, 1 / 4)   # knots of the piecewise-linear square root (exact at 0, the knots and 1)
GRID = 7
FEAT = NDIR * GRID * GRID


# ----------------------------------------------------------------------------------------------------------
# random numbers: uniform draws only (bit-exact everywhere); the bell-shaped draw is a sum of four uniforms
# ----------------------------------------------------------------------------------------------------------
class Rand:
    def __init__(self, seed):
        self.g = np.random.default_rng(seed)

    def u(self, n, a, b):
        return a + (b - a) * self.g.random(n)

    def n(self, n, mu, sd):
        """Bell-shaped, mean mu, standard deviation sd, never further than 3.46 sd from mu."""
        return mu + sd * (self.g.random((4, n)).sum(0) - 2.0) * np.sqrt(3.0)

    def b(self, n, p):
        return self.g.random(n) < p


# ----------------------------------------------------------------------------------------------------------
# curve helpers.  Points are arrays (B, 2) = (x, y), y pointing down, digit height about 1.
# ----------------------------------------------------------------------------------------------------------
def P(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    return np.stack([x, y], -1)


def lerp(p, q, t):
    t = np.asarray(t, float)
    return p + (q - p) * (t[..., None] if t.ndim else t)


def line(p, q, n=12):
    t = np.linspace(0.0, 1.0, n + 1)[None, :, None]
    return p[:, None, :] + (q - p)[:, None, :] * t


def spline(pts, n=10):
    """Catmull-Rom curve through the points (list of (B, 2)); n samples per span."""
    c = np.stack(pts, 1)                                    # (B, m, 2)
    first, last = 2 * c[:, :1] - c[:, 1:2], 2 * c[:, -1:] - c[:, -2:-1]
    c = np.concatenate([first, c, last], 1)
    t = (np.arange(n) / n)[None, None, :, None]
    p0, p1, p2, p3 = c[:, :-3, None], c[:, 1:-2, None], c[:, 2:-1, None], c[:, 3:, None]
    s = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
    s = s.reshape(s.shape[0], -1, 2)
    return np.concatenate([s, c[:, -2:-1]], 1)


def bow(p, q, amount, t=0.5):
    """A point between p and q pushed sideways by `amount` (positive: to the right hand of someone walking from p to q on the screen)."""
    d = q - p
    nrm = np.stack([d[:, 1], -d[:, 0]], -1)
    return lerp(p, q, t) + nrm * np.asarray(amount, float)[:, None]


def keep(flag, pts, anchor):
    """Optional stroke: where flag is False all of its points collapse onto `anchor` (a point already inked)."""
    return np.where(flag[:, None, None], pts, anchor[:, None, :])


def pick(flag, a, b):
    """Path a where flag else path b; the shorter one is padded by repeating its last point."""
    k = max(a.shape[1], b.shape[1])
    a, b = [np.concatenate([o, np.repeat(o[:, -1:], k - o.shape[1], 1)], 1) for o in (a, b)]
    return np.where(flag[:, None, None], a, b)


def jit(R, p, s=0.018):
    return p + np.stack([R.n(len(p), 0, s), R.n(len(p), 0, s)], -1)


# ----------------------------------------------------------------------------------------------------------
# the ten digits as pen paths.  Each function returns a list of strokes, each (B, k, 2).
# ----------------------------------------------------------------------------------------------------------
def digit0(R, B):
    a = R.u(B, 0.25, 0.47)                       # half width (height is 1)
    egg = R.n(B, 0.0, 0.10)                      # top narrower (+) or wider (-) than the bottom
    pw = R.u(B, 0.80, 1.12)                      # < 1 boxy, > 1 pointed
    th0 = R.n(B, 0.25, 0.35)                     # where the pen starts, measured from the top
    sweep = 2 * np.pi + R.u(B, -0.45, 0.9)       # a gap (< full turn) or an overlap (> full turn)
    drift = R.n(B, 0.0, 0.07)                    # the end lands inside or outside the start
    t = np.linspace(0.0, 1.0, 56)[None, :]
    th = th0[:, None] + sweep[:, None] * t
    rad = 1.0 + drift[:, None] * (2 * t - 1.0)
    sx, sy = np.sin(th), np.cos(th)
    x = -a[:, None] * np.sign(sx) * np.abs(sx) ** pw[:, None] * rad * (1.0 - egg[:, None] * sy)
    y = 0.5 - 0.5 * np.sign(sy) * np.abs(sy) ** pw[:, None] * rad
    return [np.stack([x, y], -1)]


def digit1(R, B):
    bend, wig = R.n(B, 0.0, 0.035), R.n(B, 0.0, 0.012)
    t = np.linspace(0.0, 1.0, 29)[None, :]
    stem = np.stack([bend[:, None] * np.sin(np.pi * t) + wig[:, None] * np.sin(2 * np.pi * t), t + 0 * bend[:, None]], -1)
    top, bot = stem[:, 0], stem[:, -1]
    flag = R.b(B, 0.24)
    fl, fa = R.u(B, 0.12, 0.38), R.u(B, 0.45, 1.05)
    fstart = top + P(-fl * np.sin(fa), fl * np.cos(fa))
    fstroke = keep(flag, line(fstart, top, 10), top)
    base = R.b(B, 0.03) | (flag & R.b(B, 0.42))
    bl, br = R.u(B, 0.20, 0.42), R.u(B, 0.15, 0.42)
    bstroke = keep(base, line(bot + P(-bl, R.n(B, 0, 0.015)), bot + P(br, R.n(B, 0, 0.015)), 14), bot)
    return [stem, fstroke, bstroke]


def digit2(R, B):
    # a head (hook over the top), a long diagonal to the bottom left, a foot (sharp corner or a loop), a base to the right
    rx, ry, cxh = R.u(B, 0.19, 0.33), R.u(B, 0.14, 0.26), R.u(B, 0.30, 0.42)   # the head is part of an ellipse
    loop = R.b(B, 0.45)                                            # the foot is a loop rather than a sharp corner
    a0 = np.where(loop, R.u(B, 2.6, 3.5), R.u(B, 2.6, 4.0))        # the head starts at the left, or curled under
    a1 = R.u(B, -0.9, -0.1)                                        # ... and runs over the top to the right
    hd = ring(cxh, ry, rx, ry, a0, a1, 6)
    p2 = hd[-1]
    p4 = P(R.u(B, 0.0, 0.12), R.u(B, 0.88, 1.0))                   # bottom left
    p3 = bow(p2, p4, R.n(B, 0.03, 0.07), R.u(B, 0.4, 0.6))         # the neck: straight, or bulging to the right
    head = spline([jit(R, q, 0.012) for q in hd], 6)
    zed = R.b(B, 0.10)                                             # a flat top stroke with a sharp corner, like a Z
    z0, z1 = P(R.u(B, 0.0, 0.12), R.u(B, 0.0, 0.10)), P(R.u(B, 0.55, 0.78), R.u(B, 0.0, 0.08))
    zhead = spline([z0, bow(z0, z1, R.n(B, 0.01, 0.03)), z1], 13)
    q2 = np.where(zed[:, None], z1, p2)
    q3 = np.where(zed[:, None], bow(z1, p4, R.n(B, 0.0, 0.025)), p3)
    head = pick(zed, zhead, head)
    ex = R.u(B, 0.60, 1.25)                                        # how far the base runs to the right
    # sharp corner foot
    e = P(ex, p4[:, 1] + R.u(B, -0.22, 0.06))
    m = bow(p4, e, R.n(B, -0.02, 0.05), 0.5)
    sharp = np.concatenate([spline([q2, jit(R, q3), p4], 12)[:, 1:], spline([p4, jit(R, m), jit(R, e)], 15)], 1)
    # loop foot: the diagonal runs round the bottom left, up, back over itself, and away to the right as the base
    rw = R.u(B, 0.11, 0.30)
    rh = rw * R.u(B, 0.50, 1.0)
    lx, ly = p4[:, 0] + rw, 1.0 - rh - R.u(B, 0.0, 0.06)
    lp = ring(lx, ly, rw, rh, -0.1 * np.pi, -1.5 * np.pi, 5)       # right, bottom, left, top of the loop
    xo = P(lx + rw * R.u(B, 0.9, 1.4), ly + rh * R.u(B, -0.4, 0.5))   # where it crosses back over the diagonal
    le = P(np.maximum(ex, lx + rw + 0.25), ly + rh * R.u(B, -0.3, 1.2) + R.u(B, 0.0, 0.06))
    lm = bow(xo, le, R.n(B, -0.02, 0.05), 0.5)
    looped = spline([q2, jit(R, q3)] + [jit(R, q, 0.012) for q in lp] + [jit(R, xo, 0.012), jit(R, lm), jit(R, le)], 8)[:, 1:]
    return [head, pick(loop, looped, sharp)]


def bowl(R, B, cx, top, rx, a0, tail):
    """The lower bowl of a 3 or a 5: an ellipse arc from its upper left, round the right and the bottom, out to the left."""
    ry = 0.5 * (1.0 - top)
    pts = ring(cx, 1.0 - ry, rx, ry, a0, -tail, 7)
    return pts


def digit3(R, B):
    flat = R.b(B, 0.28)                                            # flat top bar with a corner, or a round top
    yw = R.u(B, 0.38, 0.54)                                        # height of the waist
    cxu, rxu = R.u(B, 0.36, 0.46), R.u(B, 0.17, 0.30)
    up = ring(cxu, 0.5 * yw, rxu, 0.5 * yw, R.u(B, 1.9, 3.1), -R.u(B, 1.75, 2.3), 6)
    w = up[-1]                                                     # the waist: the end of the upper bowl
    top_r = spline([jit(R, q, 0.012) for q in up[:-1]] + [w], 8)
    f0 = P(R.u(B, 0.0, 0.14), R.u(B, 0.0, 0.07))
    f1 = P(R.u(B, 0.56, 0.76), R.u(B, 0.0, 0.06))
    top_f = np.concatenate([spline([jit(R, f0), bow(f0, f1, R.n(B, 0, 0.03)), f1], 10),
                            spline([f1, bow(f1, w, R.n(B, 0.0, 0.03)), w], 10)], 1)
    lo = bowl(R, B, cxu + R.u(B, -0.06, 0.08), yw - R.u(B, 0.0, 0.04), R.u(B, 0.21, 0.37), R.u(B, 1.75, 2.3), R.u(B, 2.45, 3.25))
    low = spline([w] + [jit(R, q, 0.012) for q in lo[1:]], 8)
    return [pick(flat, top_f, top_r), low]


def digit4(R, B):
    xs = R.u(B, 0.42, 0.66)                        # where the stem crosses the bar
    closed = R.b(B, 0.10)                          # closed (printed) top or open top
    yb = R.u(B, 0.50, 0.70)                        # height of the bar
    c = P(R.u(B, 0.0, 0.14), yb)                   # left end of the bar
    q0 = P(xs + R.u(B, -0.03, 0.10), R.u(B, -0.02, 0.30))
    q0 = np.where(closed[:, None], P(xs + R.u(B, -0.02, 0.08), 0.0), q0)
    q1 = P(xs + R.u(B, -0.10, 0.04), 1.0)
    stem = spline([q0, bow(q0, q1, R.n(B, 0, 0.02)), q1], 14)
    p0 = P(R.u(B, 0.02, 0.40), R.u(B, 0.0, 0.12))
    p0 = np.where(closed[:, None], q0 + P(-R.u(B, 0.0, 0.08), R.u(B, 0.0, 0.06)), p0)
    short = R.b(B, 0.15) & ~closed                 # the bar stops at the stem
    e = P(xs + np.where(short, R.u(B, -0.02, 0.03), R.u(B, 0.08, 0.36)), yb + R.u(B, -0.10, 0.06))
    rnd = R.b(B, 0.35) & ~closed                   # rounded corner (like a u) or a sharp one
    pm = bow(p0, c, R.n(B, 0.0, 0.03))
    sharp = np.concatenate([spline([jit(R, p0), pm, c], 10), spline([c, bow(c, e, R.n(B, 0, 0.02)), jit(R, e)], 10)], 1)
    c2 = c + P(R.u(B, 0.04, 0.12), R.u(B, 0.0, 0.05))
    smooth = spline([jit(R, p0), pm, c + P(0.0, -R.u(B, 0.05, 0.10)), c2, lerp(c2, e, 0.5), jit(R, e)], 8)
    return [pick(rnd, smooth, sharp), stem]


def digit5(R, B):
    yv = R.u(B, 0.30, 0.52)                                        # where the down stroke turns into the bowl
    lo = bowl(R, B, R.u(B, 0.36, 0.46), yv - R.u(B, 0.0, 0.05), R.u(B, 0.22, 0.38), R.u(B, 1.8, 2.4), R.u(B, 2.2, 3.05))
    v1 = lo[0]
    v0 = P(v1[:, 0] + R.u(B, -0.05, 0.17), R.u(B, 0.0, 0.06))      # top of the down stroke
    e = P(v0[:, 0] + R.u(B, 0.28, 0.78), v0[:, 1] + R.u(B, -0.15, 0.04))   # right end of the top bar
    s_like = R.b(B, 0.2)                                           # one smooth stroke, like an S
    down = spline([v0, bow(v0, v1, R.n(B, 0, 0.02)), v1], 10)
    two = np.concatenate([down, spline([v1] + [jit(R, q, 0.012) for q in lo[1:]], 8)], 1)
    one = spline([e, lerp(v0, e, 0.5), v0 + P(0.02, 0.03), lerp(v0, v1, 0.6)] + [jit(R, q, 0.012) for q in lo[1:]], 8)
    body = pick(s_like, one, two)
    off = R.b(B, 0.3)                                              # the bar is drawn apart from the down stroke
    bs = v0 + np.where(off[:, None], P(R.u(B, -0.06, 0.10), R.u(B, -0.06, 0.03)), P(0.0, 0.0))
    bar = spline([bs, bow(bs, e, R.n(B, 0, 0.03)), jit(R, e)], 13)
    return [body, bar]


def ring(cx, cy, rx, ry, a0, a1, k):
    """k points on an ellipse from angle a0 to a1 (radians; 0 = right, pi/2 = top of the screen, turning anticlockwise)."""
    out = []
    for i in range(k):
        a = a0 + (a1 - a0) * (i / (k - 1.0))
        out.append(P(cx + rx * np.cos(a), cy - ry * np.sin(a)))
    return out


def digit6(R, B):
    hl, wl = R.u(B, 0.32, 0.58), R.u(B, 0.14, 0.28)                # height and half width of the loop
    cx, cy = 0.36, 1.0 - 0.5 * hl
    p0 = P(cx + R.u(B, -0.14, 0.26), 0.0)                          # top of the stem
    l = P(cx - wl, cy - R.u(B, 0.0, 0.08))                         # the stem runs into the left side of the loop
    p1 = bow(p0, l, R.u(B, -0.16, 0.01), R.u(B, 0.4, 0.6))         # straight or curved like a C
    hook = R.b(B, 0.15)                                            # the start bends over to the right
    ph = np.where(hook[:, None], p0 + P(R.u(B, 0.05, 0.14), R.u(B, 0.03, 0.10)), lerp(p0, p1, -0.02))
    lp = ring(cx, cy, wl, 0.5 * hl, np.pi * 1.25, np.pi * 2.5, 6)  # lower left, bottom, right, top of the loop
    close = P(cx - wl * R.u(B, 0.55, 1.0), cy - 0.5 * hl * R.u(B, 0.0, 0.75))   # the loop closes onto the stem
    opened = R.b(B, 0.18)                                          # ... or stops short and stays open
    close = np.where(opened[:, None], lerp(lp[-1], close, R.u(B, 0.05, 0.5)), close)
    inward = R.b(B, 0.3) & ~opened                                 # ... and sometimes runs on inside
    c2 = np.where(inward[:, None], close + P(R.u(B, 0.0, 0.08), R.u(B, 0.04, 0.12)), lerp(lp[-1], close, 1.03))
    return [spline([ph, jit(R, p0), jit(R, p1), l] + [jit(R, q, 0.012) for q in lp] + [close, c2], 8)]


def digit7(R, B):
    p0 = P(R.u(B, 0.0, 0.12), R.u(B, 0.0, 0.18))
    p1 = P(R.u(B, 0.52, 0.80), R.u(B, 0.0, 0.08))
    p2 = P(R.u(B, 0.16, 0.58), 1.0)
    pm = bow(p0, p1, R.n(B, 0.025, 0.04))
    dm = bow(p1, p2, R.n(B, -0.01, 0.045))
    rnd = R.b(B, 0.4)
    sharp = np.concatenate([spline([p0, pm, p1], 11), spline([p1, dm, p2], 14)], 1)
    smooth = spline([p0, pm, lerp(pm, p1, 0.8), p1 + P(-0.01, 0.06), dm, p2], 10)
    body = pick(rnd, smooth, sharp)
    serif = R.b(B, 0.22)
    s0 = p0 + P(R.u(B, -0.03, 0.03), R.u(B, 0.10, 0.24))
    tick = keep(serif, line(s0, p0, 8), p0)
    cross = R.b(B, 0.2)
    cc = lerp(p1, p2, R.u(B, 0.40, 0.60))
    c0 = cc + P(-R.u(B, 0.14, 0.30), R.u(B, -0.04, 0.06))
    c1 = cc + P(R.u(B, 0.12, 0.26), R.u(B, -0.06, 0.04))
    bar = keep(cross, line(c0, c1, 10), cc)
    return [body, tick, bar]


def digit8_rings(R, B):
    """An 8 made of two round loops, one on top of the other."""
    yw = R.u(B, 0.40, 0.55)                                        # height of the crossing
    wt, wb = R.u(B, 0.15, 0.27), R.u(B, 0.16, 0.31)                # half widths of the two loops
    cxt, cxb = 0.35 + R.n(B, 0.0, 0.03), 0.35 + R.n(B, 0.0, 0.03)
    w = P(0.5 * (cxt + cxb) + R.n(B, 0.0, 0.02), yw)
    yt, ht, yb, hb = 0.5 * yw, 0.5 * yw, 0.5 * (1 + yw), 0.5 * (1 - yw)
    a0 = R.u(B, 0.4, 1.7)                                          # where the pen starts on the upper loop
    up1 = ring(cxt, yt, wt, ht, a0, np.pi * 1.25, 4)               # over the top and down the left side
    low = ring(cxb, yb, wb, hb, np.pi * 0.25, -np.pi * 1.25, 7)    # right, bottom, left of the lower loop
    a1 = a0 + R.u(B, -0.9, 0.3)                                    # the end stops short of the start or overshoots it
    up2 = ring(cxt, yt, wt, ht, -np.pi * 0.25, np.maximum(a1, 0.15), 4)
    pts = [jit(R, q, 0.012) for q in up1] + [w] + [jit(R, q, 0.012) for q in low]
    pts += [w + P(R.n(B, 0, 0.012), R.n(B, 0, 0.012))] + [jit(R, q, 0.012) for q in up2]
    return spline(pts, 7)


def digit8(R, B):
    # one stroke: an S from the top (left round the upper loop, across, right round the lower loop), then a return
    # stroke back up through the crossing.  The return is anything from a mirror image of the S to a straight line.
    yw = R.u(B, 0.38, 0.60)                                        # height of the crossing
    kt, kb = R.u(B, 0.0, 1.0), R.u(B, 0.0, 1.0)                    # how far the return stroke bulges (upper, lower loop)
    wt = R.u(B, 0.26, 0.60) / (1 + kt)                             # the loops' widths are shared between S and return
    wb = R.u(B, 0.12, 0.54) / (1 + kb)
    fat = R.u(B, 0.70, 1.10)                                       # < 1 rounder loops, > 1 slimmer, pointed at the crossing
    f0 = R.u(B, -1.1, 0.4)                                         # where the pen starts (0 = top; negative = to the right)
    f1 = f0 + 2 * np.pi + R.u(B, -0.55, 0.45)                      # where it stops: short of the start or past it
    fly = R.b(B, 0.38)                                             # the last part runs straight on, up and to the right
    ff = 2 * np.pi - R.u(B, 0.45, 1.1)
    f1 = np.where(fly, np.maximum(f1, ff + R.u(B, 0.4, 1.4)), f1)
    lead = R.b(B, 0.2)                                             # the first part comes in straight from the right
    lf = -R.u(B, 0.3, 0.8)
    lean = (wt * (1 - kt) + wb * (1 - kb)) * R.u(B, 0.7, 1.2)      # a straight return stroke runs from lower left to upper right
    t = np.linspace(0.0, 1.0, 120)[None, :]
    ph = f0[:, None] + (f1 - f0)[:, None] * t

    def curve(p):
        c, s2 = np.cos(p), np.sin(2 * p)
        ret = (p > np.pi) | (p < 0)                                # on the return stroke (or the lead-in beside it)
        w = (wt[:, None] * np.where(ret, kt[:, None], 1.0)) * 0.5 * (1 + c) + (wb[:, None] * np.where(ret, kb[:, None], 1.0)) * 0.5 * (1 - c)
        y = 0.5 * (1 - c) + (yw[:, None] - 0.5) * (1 - c * c)
        x = 0.35 - w * np.sign(s2) * np.abs(s2) ** fat[:, None] + lean[:, None] * (0.5 - y)
        return x, y

    def straight(p, at):                                           # the tangent line of the curve at `at`
        h = 1e-3
        xa, ya = curve(at[:, None] + 0 * p)
        xb, yb = curve(at[:, None] + h + 0 * p)
        return xa + (xb - xa) / h * (p - at[:, None]), ya + (yb - ya) / h * (p - at[:, None])

    x, y = curve(ph)
    on_f, on_l = (ph > ff[:, None]) & fly[:, None], (ph < lf[:, None]) & lead[:, None]
    xf, yf = straight(ph, ff)
    xl, yl = straight(ph, lf)
    x = np.where(on_f, xf, np.where(on_l, xl, x))
    y = np.where(on_f, yf, np.where(on_l, yl, y))
    one = np.stack([x, y], -1)
    return [pick(R.b(B, 0.12), digit8_rings(R, B), one)]


def digit9(R, B):
    hl, wl = R.u(B, 0.32, 0.64), R.u(B, 0.13, 0.35)                # height and half width of the loop
    cx, cy = R.u(B, 0.28, 0.36), 0.5 * hl
    a0 = R.u(B, 0.1, 1.3)                                          # where the pen starts, from the right going up
    gap = R.u(B, 0.6, 1.05)                                        # < 1: the loop is left open at the top right
    unclosed = R.b(B, 0.06)                                        # the loop stops at the bottom and never closes
    a1 = np.where(unclosed, R.u(B, 1.3, 1.6), 1.75) * np.pi
    lp = ring(cx, cy, wl, 0.5 * hl, a0, a1, 6)                     # over the top, down the left, along the bottom
    lp[0] = P(cx + (lp[0][:, 0] - cx) * gap, lp[0][:, 1])
    j = P(cx + wl, cy + R.u(B, -0.12, 0.04))                       # the loop closes onto the tail here
    last = np.where(unclosed[:, None], lerp(lp[-2], lp[-1], 1.05), j)
    loop = spline([jit(R, q, 0.012) for q in lp] + [last], 8)
    tt = P(cx + wl + R.u(B, -0.02, 0.05), R.u(B, 0.04, 0.22))      # top of the tail
    tt = np.where(unclosed[:, None], lp[0] + P(R.u(B, 0.0, 0.05), R.u(B, 0.0, 0.05)), tt)
    e = P(R.u(B, 0.12, 0.66), 1.0)
    tm = bow(j, e, R.n(B, 0.0, 0.05))
    hook = R.b(B, 0.15)                                            # the tail curls to the left at the bottom
    kick = R.b(B, 0.14) & ~hook & (e[:, 0] > 0.4)                  # ... or kicks out a little to the right
    h1 = e + P(-R.u(B, 0.08, 0.16), R.u(B, -0.02, 0.03))
    h2 = h1 + P(-R.u(B, 0.10, 0.22), -R.u(B, 0.02, 0.10))
    k1 = e + P(R.u(B, 0.04, 0.09), R.u(B, -0.01, 0.03))
    k2 = k1 + P(R.u(B, 0.03, 0.10), -R.u(B, 0.0, 0.06))
    e1 = np.where(hook[:, None], h1, np.where(kick[:, None], k1, lerp(tm, e, 1.01)))
    e2 = np.where(hook[:, None], h2, np.where(kick[:, None], k2, lerp(tm, e, 1.02)))
    eh = np.where((hook | kick)[:, None], e + P(0.0, -0.06), e)
    tail = spline([tt, j, tm, eh, e1, e2], 9)
    return [loop, tail]


DIGITS = (digit0, digit1, digit2, digit3, digit4, digit5, digit6, digit7, digit8, digit9)
SLANT = {1: (0.25, 0.30), 8: (0.22, 0.20)}   # mean and spread of the shear; 1 leans more freely, 8 leans more to the right


# ----------------------------------------------------------------------------------------------------------
# drawing: vary the pose, fit the 20 x 20 box, put the centre of mass at the centre, render the pen stroke
# ----------------------------------------------------------------------------------------------------------
_YY, _XX = np.meshgrid(np.arange(28.0), np.arange(28.0), indexing="ij")
_PX, _PY = _XX.ravel()[None, :, None], _YY.ravel()[None, :, None]


def render(pts, rad, ramp):
    """pts (B, K, 2) in pixels, pen radius (B,), edge ramp width (B,) -> images (B, 784) in 0..1."""
    d2 = (_PX - pts[:, None, :, 0]) ** 2 + (_PY - pts[:, None, :, 1]) ** 2
    d = np.sqrt(d2.min(2))
    return np.clip((rad[:, None] - d) / ramp[:, None] + 0.5, 0.0, 1.0)


def pose(R, cls, pts):
    """Wobble, proportions, slant and rotation of the pen path (B, K, 2)."""
    B = len(pts)
    x, y = pts[..., 0], pts[..., 1]
    for _ in range(2):                                             # slow wobble of the whole drawing
        ang, freq, ph, amp = R.u(B, 0, 2 * np.pi), R.u(B, 0.5, 1.3), R.u(B, 0, 2 * np.pi), R.u(B, 0.0, 0.035)
        ang2 = R.u(B, 0, 2 * np.pi)
        w = amp[:, None] * np.sin(2 * np.pi * freq[:, None] * (np.cos(ang)[:, None] * x + np.sin(ang)[:, None] * y) + ph[:, None])
        x, y = x + w * np.cos(ang2)[:, None], y + w * np.sin(ang2)[:, None]
    x = x * np.exp(R.n(B, 0.0, 0.13))[:, None]                     # wider or narrower
    mu, sd = SLANT.get(cls, (0.19, 0.22))
    x = x + R.n(B, mu, sd)[:, None] * (0.5 - y)                    # slant: positive leans to the right
    rot = R.n(B, 0.0, 0.07)
    c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
    return np.stack([c * x - s * y, s * x + c * y], -1)


def draw(R, cls, B):
    """B drawings of digit `cls` as images (B, 784)."""
    pts = pose(R, cls, np.concatenate(DIGITS[cls](R, B), 1))
    width = np.clip(np.exp(R.n(B, np.log(2.8), 0.34)), 1.3, 6.0)   # pen width in pixels
    rad, ramp = 0.5 * width, R.u(B, 0.9, 1.5)
    lo, hi = pts.min(1), pts.max(1)
    scale = (20.0 - 2 * rad) / np.maximum((hi - lo).max(1), 1e-6)  # the inked box is 20 pixels on its long side
    pts = (pts - 0.5 * (lo + hi)[:, None, :]) * scale[:, None, None]
    target = np.stack([R.u(B, 13.0, 14.5), R.u(B, 13.0, 14.5)], -1)
    img = render(pts[:, ::2] + 13.5, rad, ramp)                    # first pass: find the centre of mass
    m = img.sum(1)
    com = np.stack([(img * _PX[:, :, 0]).sum(1) / m, (img * _PY[:, :, 0]).sum(1) / m], -1)
    return render(pts + 13.5 + (target - com)[:, None, :], rad, ramp)


# ----------------------------------------------------------------------------------------------------------
# the formula front end: the same numbers are used for the network's layers and for turning drawings into templates
# ----------------------------------------------------------------------------------------------------------
def edge_filters():
    """(8, 7, 7): Gaussian-derivative filters; a full-contrast step edge facing direction k gives about 1."""
    r = np.arange(KD) - KD // 2
    g = np.exp(-r ** 2 / (2 * SIG_D ** 2))
    step = (r[r > 0] * g[r > 0]).sum() * g.sum()                   # response of the x filter to a unit step
    gx = (r[None, :] * g[None, :] * g[:, None]) / step             # W[dy, dx]
    gy = gx.T
    th = np.arange(NDIR) * 2 * np.pi / NDIR
    return np.cos(th)[:, None, None] * gx[None] + np.sin(th)[:, None, None] * gy[None]


def pool_window():
    r = np.arange(KP) - (KP - 1) / 2.0
    g = np.exp(-r ** 2 / (2 * SIG_P ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def sqrt_pieces():
    """sqrt(v) on 0..1 as slope0 * v + sum_j coef_j * relu(v - knot_j)."""
    xs = np.array((0.0,) + KNOTS + (1.0,))
    slopes = np.diff(np.sqrt(xs)) / np.diff(xs)
    return slopes[0], np.diff(slopes)


def corr2d(x, w, stride=1, pad=0):
    """x (B, H, W), w (kh, kw) -> correlation exactly as the network's conv computes it."""
    kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    ho, wo = (xp.shape[1] - kh) // stride + 1, (xp.shape[2] - kw) // stride + 1
    out = np.zeros((x.shape[0], ho, wo))
    for dy in range(kh):
        for dx in range(kw):
            out += w[dy, dx] * xp[:, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
    return out


def features(img):
    """img (B, 784) -> (B, 392): what the network's front end computes, in float64."""
    B = len(img)
    x = img.reshape(B, 28, 28)
    r = np.arange(KD) - KD // 2
    g = np.exp(-r ** 2 / (2 * SIG_D ** 2))
    step = (r[r > 0] * g[r > 0]).sum() * g.sum()
    gx = corr2d(x, (r[None, :] * g[None, :] * g[:, None]) / step, 1, KD // 2)
    gy = corr2d(x, (r[:, None] * g[:, None] * g[None, :]) / step, 1, KD // 2)
    th = np.arange(NDIR) * 2 * np.pi / NDIR
    d = np.maximum(np.cos(th)[None, :, None, None] * gx[:, None] + np.sin(th)[None, :, None, None] * gy[:, None], 0.0)
    q = NDIR // 4
    s = np.maximum(d - BETA * (np.roll(d, q, 1) + np.roll(d, -q, 1)), 0.0)
    u = np.minimum(s, CLIP) / CLIP
    pw = pool_window()
    v = np.stack([corr2d(u[:, k], pw, SP, PP) for k in range(NDIR)], 1)        # (B, 8, 7, 7)
    m0, coef = sqrt_pieces()
    f = m0 * v
    for kn, cf in zip(KNOTS, coef):
        f = f + cf * np.maximum(v - kn, 0.0)
    return f.reshape(B, FEAT)


def templates(seed=SEED, per_class=PER_CLASS, batch=100):
    """All templates, (10 * per_class, 392) float32, class by class; each row is one drawing divided by its own length."""
    out = np.zeros((10 * per_class, FEAT), np.float32)
    for cls in range(10):
        R = Rand([seed, cls])
        for i in range(0, per_class, batch):
            f = features(draw(R, cls, batch))
            f = f / np.sqrt((f ** 2).sum(1, keepdims=True))
            out[cls * per_class + i: cls * per_class + i + batch] = f
    return out


def network(per_class=PER_CLASS):
    q = NDIR // 4
    m0, coef = sqrt_pieces()
    nk = len(KNOTS) + 1
    W = {}
    W["edge.W"] = edge_filters()[:, None]
    W["edge.b"] = np.zeros(NDIR)
    sh = np.zeros((2 * NDIR, NDIR, 1, 1)); shb = np.zeros(2 * NDIR)
    for k in range(NDIR):
        for o in (k, NDIR + k):                    # channel k: sharpened response; channel 8 + k: the same minus CLIP
            sh[o, k, 0, 0] = 1.0
            sh[o, (k + q) % NDIR, 0, 0] -= BETA
            sh[o, (k - q) % NDIR, 0, 0] -= BETA
        shb[NDIR + k] = -CLIP
    W["sharp.W"], W["sharp.b"] = sh, shb
    pw = pool_window()
    po = np.zeros((NDIR, 2 * NDIR, KP, KP))
    for k in range(NDIR):                          # min(s, CLIP) / CLIP = (relu(s) - relu(s - CLIP)) / CLIP, pooled
        po[k, k], po[k, NDIR + k] = pw / CLIP, -pw / CLIP
    W["pool.W"], W["pool.b"] = po, np.zeros(NDIR)
    ex = np.zeros((nk * NDIR, NDIR, 1, 1)); exb = np.zeros(nk * NDIR)
    co = np.zeros((NDIR, nk * NDIR, 1, 1))
    for k in range(NDIR):
        for j in range(nk):
            ex[k * nk + j, k, 0, 0] = 1.0
            exb[k * nk + j] = 0.0 if j == 0 else -KNOTS[j - 1]
            co[k, k * nk + j, 0, 0] = m0 if j == 0 else coef[j - 1]
    W["split.W"], W["split.b"], W["root.W"], W["root.b"] = ex, exb, co, np.zeros(NDIR)
    gsz = per_class // GROUPS
    layers = [
        {"name": "edge", "op": "conv", "inputs": ["image"], "out": NDIR, "kernel": [KD, KD], "stride": 1, "padding": KD // 2},
        {"name": "edge_r", "op": "relu", "inputs": ["edge"]},
        {"name": "sharp", "op": "conv", "inputs": ["edge_r"], "out": 2 * NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "sharp_r", "op": "relu", "inputs": ["sharp"]},
        {"name": "pool", "op": "conv", "inputs": ["sharp_r"], "out": NDIR, "kernel": [KP, KP], "stride": SP, "padding": PP},
        {"name": "split", "op": "conv", "inputs": ["pool"], "out": nk * NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "split_r", "op": "relu", "inputs": ["split"]},
        {"name": "root", "op": "conv", "inputs": ["split_r"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "match", "op": "dense", "inputs": ["root"], "out": 10 * per_class},
        {"name": "bank", "op": "reshape", "inputs": ["match"], "shape": [1, 10 * GROUPS, gsz]},
        {"name": "best", "op": "maxpool", "inputs": ["bank"], "size": [1, gsz], "stride": [1, gsz]},
        {"name": "byclass", "op": "reshape", "inputs": ["best"], "shape": [1, 10, GROUPS]},
        {"name": "vote", "op": "avgpool", "inputs": ["byclass"], "size": [1, GROUPS], "stride": [1, GROUPS]},
        {"name": "scores", "op": "reshape", "inputs": ["vote"], "shape": [10]},
    ]
    return {"layers": layers}, W


def main(out_dir):
    net, W = network()
    W["match.W"] = templates()
    W["match.b"] = np.zeros(len(W["match.W"]))
    with open(out_dir + "/network.json", "w") as f:
        json.dump(net, f, indent=1)
    np.savez(out_dir + "/weights.npz", **{k: np.asarray(v, np.float32) for k, v in W.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
