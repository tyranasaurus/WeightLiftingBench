"""Build a hand-designed digit classifier: python3 build.py OUT_DIR

Nothing here reads any file.  Every weight comes from formulas and from digit shape models typed below:
  * features: 16 oriented derivative-of-Gaussian edge filters + relu, folded into 8 unsigned orientations plus a
    blurred ink channel, Gaussian-pooled to 14 x 14; plus an "enclosed background" (hole) channel computed inside
    the network by flood-filling the background from the frame border (36 steps of erode-with-zero-padding,
    blocked by a soft ink mask), also Gaussian-pooled to 14 x 14,
  * the template bank is made of synthetic digits drawn from hand-typed pen paths (stroke models of each digit's
    common writing styles, with random slant / width / thickness / wobble from a fixed seed), rendered the way the
    data is described (ink box scaled to 20 px, centre of mass at the frame centre), passed through the same
    fixed feature computation and normalised,
  * the image's normalised feature vector is correlated with every template; each class then scores
    mean over its templates of relu(correlation - best overall correlation + DELTA): a soft vote among the
    templates (of any class) that come within DELTA of the best match.
"""
import json, sys
from pathlib import Path
import numpy as np

SEED = 12345
EDGE = 1.25          # width of the soft (anti-aliased) stroke edge, pixels
JITTER, WARP = 0.018, 0.03
# templates per class: more for classes whose shapes vary more
N_CLASS = [3400, 1900, 7600, 6700, 6700, 7100, 5700, 5700, 6700, 5700]
DELTA = 0.03         # correlation margin for the vote among near-best templates

# ----------------------------------------------------------------------------------------------------------
# curve helpers (unit box: x to the right, y downward, digit height about 1)
# ----------------------------------------------------------------------------------------------------------

def catmull(P, step=0.01):
    """Smooth curve through control points P (n, 2) (uniform Catmull-Rom); returns dense points."""
    P = np.asarray(P, float)
    if len(P) == 1:
        return P.copy()
    if len(P) == 2:
        L = np.linalg.norm(P[1] - P[0]); m = max(2, int(L / step) + 1)
        t = np.linspace(0, 1, m)[:, None]
        return P[0] * (1 - t) + P[1] * t
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        L = np.linalg.norm(p2 - p1); m = max(2, int(L / step) + 1)
        t = np.linspace(0, 1, m, endpoint=(i == len(Q) - 3))[:, None]
        t2, t3 = t * t, t * t * t
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.vstack(out)


def ell(cx, cy, rx, ry, a0, a1, n=None, tilt=0.0, shrink=0.0):
    """Control points on an ellipse from angle a0 to a1 (degrees, 0 = right, 90 = top, increasing anticlockwise)."""
    if n is None:
        n = max(3, int(abs(a1 - a0) / 25) + 1)
    a = np.radians(np.linspace(a0, a1, n))
    s = 1 - shrink * np.linspace(0, 1, n)
    x, y = rx * np.cos(a) * s, -ry * np.sin(a) * s
    c, sn = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
    return np.stack([cx + c * x - sn * y, cy + sn * x + c * y], 1)


def U(rng, a, b):
    return rng.uniform(a, b)


def pick(rng, weights):
    w = np.asarray(weights, float)
    return int(rng.choice(len(w), p=w / w.sum()))

# ----------------------------------------------------------------------------------------------------------
# digit shape models: each returns a list of strokes, each stroke a list of control points
# ----------------------------------------------------------------------------------------------------------

def d0(r):
    rx, ry = U(r, 0.3, 0.56), 0.5
    a0 = U(r, 60, 130)
    sweep = U(r, 330, 390)
    pts = ell(0.5, 0.5, rx, ry, a0, a0 + sweep, n=16, tilt=U(r, -15, 15), shrink=U(r, 0.0, 0.12))
    # egg / pear shapes: squeeze one end
    k = U(r, -0.12, 0.12)
    pts[:, 0] = 0.5 + (pts[:, 0] - 0.5) * (1 + k * (pts[:, 1] - 0.5) * 2)
    return [pts]


def d1(r):
    s = pick(r, [6, 3, 1])
    bend = U(r, -0.06, 0.06)
    stem = [(0.5, 0.0), (0.5 + bend, 0.5), (0.5, 1.0)]
    out = [stem]
    if s >= 1:
        fx, fy = U(r, 0.12, 0.28), U(r, 0.08, 0.25)
        out = [[(0.5 - fx, fy), (0.5, 0.0)], stem]
    if s == 2:
        b = U(r, 0.12, 0.25)
        out.append([(0.5 - b, 1.0), (0.5 + b * U(r, 0.8, 1.2), 1.0)])
    return out


def d2(r):
    s = pick(r, [5, 4, 1])
    hx, hy = U(r, 0.04, 0.25), U(r, 0.12, 0.4)       # start of the top hook
    tx = U(r, 0.38, 0.6)                              # top of the arch
    rx, ry = U(r, 0.78, 0.95), U(r, 0.15, 0.35)       # right side of the arch
    nx, ny = U(r, 0.38, 0.62), U(r, 0.5, 0.68)        # middle of the neck
    bx, by = U(r, 0.02, 0.2), U(r, 0.93, 1.0)         # bottom-left corner
    ex, ey = U(r, 0.7, 1.05), U(r, 0.86, 1.02)        # end of the base
    if s == 0:   # classic: hook over the top, neck down to the left, flat base
        curve = [(hx, hy), (0.5 * (hx + tx) - 0.05, 0.05), (tx, 0.0), (rx, ry), (nx, ny), (bx, by)]
        return [curve, [(bx, by), ((bx + ex) / 2, 0.5 * (by + ey) + U(r, -0.03, 0.03)), (ex, ey)]]
    if s == 1:   # small loop at the bottom left, then the base
        L = U(r, 0.08, 0.22)
        lx, ly = bx + 0.6 * L, by - 0.6 * L
        curve = [(hx, hy), (0.5 * (hx + tx) - 0.05, 0.05), (tx, 0.0), (rx, ry), (nx, ny), (lx + 0.3 * L, ly + 0.3 * L),
                 (lx - 0.6 * L, ly + 0.5 * L), (lx - 0.8 * L, ly - 0.3 * L), (lx - 0.1 * L, ly - 0.9 * L),
                 (lx + 0.6 * L, ly - 0.1 * L), (lx + 1.2 * L, by), ((lx + ex) / 2 + 0.1, 0.5 * (by + ey)), (ex, ey)]
        return [curve]
    # flatter, Z-like top
    curve = [(hx, 0.08), (0.45, 0.0), (rx - 0.05, 0.06), (rx - 0.05, 0.25), (nx, ny), (bx, by)]
    return [curve, [(bx, by), (ex, ey)]]


def d3(r):
    s = pick(r, [7, 3])
    mx, my = U(r, 0.3, 0.55), U(r, 0.4, 0.5)
    lower = [(mx, my), (0.75, my + 0.1), (0.85, 0.75), (0.65, 0.96), (0.4, 1.0), (0.12, U(r, 0.8, 0.92))]
    if s == 0:
        ur = U(r, 0.62, 0.8)
        upper = [(0.15, U(r, 0.08, 0.25)), (0.42, 0.0), (ur, 0.08), (ur, 0.3), (mx, my)]
    else:
        upper = [(0.12, 0.02), (0.8, 0.0), (mx, my)]
        upper = [upper[0], upper[1]], [upper[1], upper[2]]
        return [upper[0], upper[1], lower]
    if r.uniform() < 0.3:   # rounded middle instead of a sharp point
        return [upper[:-1] + [(mx - 0.05, my), (mx, my + 0.02)] + lower[1:]]
    return [upper, lower]


def d4(r):
    s = pick(r, [6, 2, 2])
    cy = U(r, 0.52, 0.72)            # crossbar height
    sx = U(r, 0.58, 0.8)             # stem x
    ext = U(r, 0.04, 0.3)            # crossbar extends past the stem
    ax = U(r, 0.0, 0.15)             # left end of the crossbar
    tilt = U(r, -0.02, 0.08)         # crossbar rises to the right
    if s == 0:   # open top: left arm down, crossbar, separate stem
        lx0 = U(r, 0.08, 0.45)
        stem = [(sx, U(r, -0.05, 0.3)), (sx + U(r, -0.05, 0.05), 1.0)]
        if r.uniform() < 0.3:        # rounded corner between arm and crossbar
            return [[(lx0, 0.0), (ax + 0.03, cy - 0.12), (ax + 0.1, cy), (sx + ext, cy - tilt)], stem]
        return [[(lx0, 0.0), (ax, cy)], [(ax, cy), (sx + ext, cy - tilt)], stem]
    if s == 1:   # closed triangle top
        top = (sx, 0.0)
        return [[top, (ax, cy)], [(ax, cy), (sx + ext, cy - tilt)], [top, (sx + U(r, -0.05, 0.05), 1.0)]]
    # rounded, single stroke: left arm curving into the crossbar and up to the stem top
    lx0 = U(r, 0.05, 0.25)
    curve = [(lx0, 0.0), (lx0 * 0.5 + 0.05, cy * 0.7), (0.3, cy), (sx - 0.05, cy - 0.02), (sx, cy * 0.5), (sx, 0.0)]
    return [curve, [(sx, 0.0), (sx + U(r, -0.05, 0.05), 1.0)]]


def d5(r):
    s = pick(r, [6, 3, 2])
    tx = U(r, 0.1, 0.3)                       # top of the stem (left end of the bar)
    sy = U(r, 0.32, 0.5)                      # bottom of the stem
    bx = tx + U(r, -0.08, 0.02)
    bar = [(tx, 0.0), (U(r, 0.65, 0.95), -U(r, -0.02, 0.08))]
    wr = U(r, 0.72, 0.92)                     # right edge of the belly
    ex, ey = U(r, 0.02, 0.3), U(r, 0.75, 0.95)    # end of the belly hook
    belly = [(bx, sy), (0.42, sy - U(r, 0.02, 0.1)), (wr - 0.08, sy + 0.04), (wr, 0.7), (wr - 0.15, 0.93), (0.42, 1.0), (ex, ey)]
    if s == 0:   # sharp corner between stem and belly
        return [bar, [(tx, 0.0), (bx, sy)], belly]
    if s == 1:   # stem flows into the belly
        return [bar, [(tx, 0.0), (bx, sy * 0.7)] + belly[1:]]
    # rounder belly that curls back up towards the stem
    belly = belly[:-1] + [(0.2, 0.96), (U(r, 0.12, 0.25), U(r, 0.72, 0.85))]
    return [bar, [(tx, 0.0), (bx, sy)], belly]


def d6(r):
    s = pick(r, [6, 4])
    cy = U(r, 0.68, 0.8)
    ry = 1.0 - cy
    rx = U(r, 0.2, 0.32)
    cx = 0.5
    top = (U(r, 0.55, 0.85), 0.0)
    a_end = U(r, 540, 580) if r.uniform() < 0.85 else U(r, 470, 520)   # a few loops left open
    loop = ell(cx, cy, rx, ry, 180, a_end, n=14)
    if s == 0:   # curved stem
        mid = (U(r, 0.22, 0.4), U(r, 0.2, 0.75 - ry))
        return [[top, mid] + [tuple(p) for p in loop]]
    # nearly straight stem from the top right into the left side of the loop
    return [[top, (0.5 * (top[0] + cx - rx) - 0.03, 0.5 * cy)] + [tuple(p) for p in loop]]


def d7(r):
    s = pick(r, [7, 2, 1])
    tx0 = U(r, 0.0, 0.45)            # left end of the top bar
    tx1 = 0.9
    y0, y1 = U(r, -0.02, 0.1), U(r, -0.05, 0.05)
    bx = U(r, 0.3, 0.8)              # foot of the stem
    bend = U(r, -0.06, 0.1)          # stem bulges to the right when positive
    top = [(tx0, y0), (tx1, y1)]
    stem = [(tx1, y1), ((tx1 + bx) / 2 + bend, 0.5), (bx, 1.0)]
    out = [top, stem]
    if s == 1:   # crossbar
        my = U(r, 0.45, 0.6)
        xm = tx1 + (bx - tx1) * my + bend
        out.append([(xm - U(r, 0.15, 0.3), my), (xm + U(r, 0.1, 0.25), my - U(r, 0, 0.04))])
    if s == 2:   # small hook down at the start of the top bar
        out.append([(tx0 + U(r, -0.03, 0.03), y0 + U(r, 0.1, 0.25)), (tx0, y0)])
    return out


def d8(r):
    s = pick(r, [8, 2])
    xc = U(r, 0.44, 0.56)
    yc = U(r, 0.4, 0.55)
    wt, wb = U(r, 0.12, 0.27), U(r, 0.15, 0.31)       # half widths of the top and bottom loops
    if s == 0:   # one stroke: start top right, over the top, diagonally through the middle, around the bottom
        b = 1 - yc
        P = [(xc + 0.9 * wt, 0.12 + U(r, -0.04, 0.06)), (xc + 0.1 * wt, 0.0), (xc - wt, 0.35 * yc), (xc - 0.4 * wt, 0.8 * yc),
             (xc, yc), (xc + wb, yc + 0.35 * b), (xc + 0.8 * wb, yc + 0.8 * b), (xc, 1.0), (xc - wb, yc + 0.65 * b),
             (xc - 0.5 * wb, yc + 0.2 * b), (xc, yc), (xc + 0.8 * wt, 0.6 * yc), (xc + 0.9 * wt, 0.15 + U(r, -0.04, 0.06))]
        return [P]
    top = ell(xc, yc / 2, wt, yc / 2, 270, 270 + 360, n=16)
    bot = ell(xc, (1 + yc) / 2, wb, (1 - yc) / 2, 90, 90 + 360, n=16)
    return [top, bot]


def d9(r):
    s = pick(r, [6, 2, 2])
    cy = U(r, 0.16, 0.3)
    ry = cy
    rx = U(r, 0.2, 0.36)
    cx = 0.5
    tilt = U(r, -20, 20)
    a0 = U(r, -10, 30)
    gap = 0.0 if r.uniform() < 0.8 else U(r, 15, 50)   # loop sometimes left open where it meets the stem
    loop = ell(cx, cy, rx, ry, a0, a0 + U(r, 335, 365) - gap, n=15, tilt=tilt)
    sx, sy = loop[0]
    if s == 0:   # straight stem
        stem = [(sx, sy), (sx + U(r, -0.12, 0.06), 1.0)]
    elif s == 1: # stem curving to the left at the bottom
        stem = [(sx, sy), (sx, 0.7), (sx - 0.1, 0.95), (sx - U(r, 0.2, 0.35), 1.0)]
    else:        # stem slightly bowed
        stem = [(sx, sy), (sx + U(r, 0.0, 0.08), 0.6), (sx - U(r, 0.0, 0.1), 1.0)]
    return [loop, stem]


PROTOS = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]

# ----------------------------------------------------------------------------------------------------------
# random deformation and rendering in the data's format
# ----------------------------------------------------------------------------------------------------------

def deform(strokes, r):
    """Jitter control points, smooth, warp, slant, rotate, stretch.  Returns list of dense point arrays."""
    dense = []
    for S in strokes:
        P = np.asarray(S, float)
        P = P + r.normal(0, JITTER, P.shape)
        dense.append(catmull(P))
    # smooth random warp
    pts = np.vstack(dense)
    W = np.zeros_like(pts)
    for _ in range(3):
        ang = U(r, 0, 2 * np.pi); f = U(r, 0.4, 1.2)
        ph = U(r, 0, 2 * np.pi)
        A = r.normal(0, WARP, 2)
        W += A[None, :] * np.sin(2 * np.pi * f * (pts[:, 0] * np.cos(ang) + pts[:, 1] * np.sin(ang)) + ph)[:, None]
    pts = pts + W
    shear = np.clip(r.normal(0.15, 0.25), -0.5, 0.8)
    rot = np.radians(r.normal(0, 6))
    sx = np.exp(r.normal(0, 0.18))
    x, y = pts[:, 0] - 0.5, pts[:, 1] - 0.5
    x = x * sx - shear * y
    c, s = np.cos(rot), np.sin(rot)
    pts = np.stack([c * x - s * y, s * x + c * y], 1)
    out, k = [], 0
    for D in dense:
        out.append(pts[k:k + len(D)]); k += len(D)
    return out


def render_points(P, rad, H=28):
    """P (n, 2) points in pixel coordinates (x = column, y = row), rad (n,) pen radius at each point.
    Ink = clip(0.5 - min over points of (distance - radius) / EDGE, 0, 1): a soft edge EDGE pixels wide."""
    g = np.arange(H, dtype=float)
    dx = g[None, :] - P[:, 0:1]          # (n, W)
    dy = g[None, :] - P[:, 1:2]          # (n, H)
    d = np.sqrt(dy[:, :, None] ** 2 + dx[:, None, :] ** 2) - rad[:, None, None]
    return np.clip(0.5 - d.min(0) / EDGE, 0, 1)


def resample(P, spacing):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    out = [P[:1]]
    for i in range(len(seg)):
        m = int(np.ceil(seg[i] / spacing))
        if m > 1:
            t = np.arange(1, m + 1)[:, None] / m
            out.append(P[i] * (1 - t) + P[i + 1] * t)
        else:
            out.append(P[i + 1:i + 2])
    return np.vstack(out)


def make_digit(cls, r):
    strokes = PROTOS[cls](r)
    dense = deform(strokes, r)
    rad = U(r, 0.65, 2.1)
    allp = np.vstack(dense)
    lo, hi = allp.min(0), allp.max(0)
    ext = max(hi[0] - lo[0], hi[1] - lo[1])
    scale = (20.0 - 2 * rad - 0.5) / ext
    mid = (lo + hi) / 2
    pix = [(D - mid) * scale + 14.0 for D in dense]
    P = np.vstack([resample(D, 0.35) for D in pix])
    # pen pressure: smooth variation of the radius along the path
    t = np.arange(len(P)) * 0.35
    amp, f, ph = U(r, 0.0, 0.3), U(r, 0.01, 0.04), U(r, 0, 2 * np.pi)
    R = rad * (1 + amp * np.sin(2 * np.pi * f * t + ph))
    img = render_points(P, R)
    m = img.sum()
    g = np.arange(28)
    com = np.array([(img.sum(0) * g).sum() / m, (img.sum(1) * g).sum() / m])   # (x, y)
    P = P + (14.0 - com)[None, :]
    return render_points(P, R)


def synth_class(c, n, seed=SEED):
    r = np.random.default_rng([seed, c])
    return np.array([make_digit(c, r) for _ in range(n)])


def synth_set(n_per_class, seed=SEED):
    return np.array([synth_class(c, n_per_class, seed) for c in range(10)])


# ----------------------------------------------------------------------------------------------------------
# fixed feature filters (formulas only)
# ----------------------------------------------------------------------------------------------------------
N_DIR = 16          # signed edge directions, 22.5 degrees apart
N_ORI = N_DIR // 2  # unsigned orientations
SIG_ACROSS, SIG_ALONG = 1.0, 1.8
BLUR_W = 1.0        # weight of the plain ink channel
POOL_SIG = 1.1
HOLE_W, HOLE_SIG = 1.5, 1.6   # weight and blur of the enclosed-background channel
HOLE_ITERS = 36               # flood-fill steps (one pixel each) from the frame border
INK_GAIN, INK_LO = 4.0, 0.2   # soft ink mask: clip(INK_GAIN * (x - INK_LO), 0, 1)


def edge_filters(k=7):
    h = k // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(float)
    F = []
    for d in range(N_DIR):
        t = 2 * np.pi * d / N_DIR
        p = xx * np.cos(t) + yy * np.sin(t)
        q = -xx * np.sin(t) + yy * np.cos(t)
        g = np.exp(-p ** 2 / (2 * SIG_ACROSS ** 2) - q ** 2 / (2 * SIG_ALONG ** 2))
        f = p * g
        # response to a unit step edge across the filter = sum over the positive half
        F.append(f / f[f > 0].sum())
    g = np.exp(-(xx ** 2 + yy ** 2) / 2.0)
    F.append(g / g.sum())
    return np.array(F)[:, None]          # (17, 1, k, k)


def pool_filter(k=5):
    h = k // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(float)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * POOL_SIG ** 2))
    return g / g.sum()


def mix_matrix():
    M = np.zeros((N_ORI + 1, N_DIR + 1))
    for o in range(N_ORI):
        M[o, o] = M[o, o + N_ORI] = 1.0
    M[N_ORI, N_DIR] = BLUR_W
    return M


def conv_np(x, W, b, stride, pad):
    """Same arithmetic as the conv layer: x (n, c, h, w), W (o, c, k, k)."""
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def gauss(k, sig):
    h = k // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(float)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
    return g / g.sum()


def feature_layers():
    W1 = edge_filters()
    b1 = np.zeros(len(W1))
    M = mix_matrix()                                   # (9, 17)
    C = M.shape[0] + 1                                 # + the hole channel
    W2 = np.zeros((C, W1.shape[0], 5, 5))
    W2[:-1] = M[:, :, None, None] * pool_filter()[None, None]
    b2 = np.zeros(C)
    # hole channel from the 4-channel flood-fill state z = [B + 1, relu(B + m), 1, m]: hole = -z0 + z1 + z2 - z3
    WH = np.zeros((C, 4, 7, 7))
    WH[-1] = HOLE_W * np.array([-1.0, 1.0, 1.0, -1.0])[:, None, None] * gauss(7, HOLE_SIG)[None]
    bH = np.zeros(C)
    return W1, b1, W2, b2, WH, bH


def ink_mask(X):
    return np.clip(INK_GAIN * (X - INK_LO), 0, 1)


def hole_state(X):
    """Flood fill of the background from outside the frame, as the network does it.  X (n, 28, 28).
    N = 1 - reachable; start N = 1, repeat N = max(ink, erode(N)) with zero padding (outside reachable).
    Kept as Q = -N:  B = maxpool3(pad0(Q)),  Q = B - relu(B + m).  Returns z (n, 4, 28, 28)."""
    m = ink_mask(X)
    Q = -np.ones_like(X)
    for _ in range(HOLE_ITERS):
        P = np.pad(Q, ((0, 0), (1, 1), (1, 1)))
        B = np.max([P[:, dy:dy + 28, dx:dx + 28] for dy in range(3) for dx in range(3)], 0)
        R = np.maximum(B + m, 0)
        Q = B - R
    return np.stack([B + 1, R, np.ones_like(X), m], 1)


def features(X, layers):
    """X (n, 28, 28) -> feature vectors (n, D), as computed by the network before the norm layer."""
    W1, b1, W2, b2, WH, bH = layers
    out = []
    for i in range(0, len(X), 500):
        x = X[i:i + 500]
        h = np.maximum(conv_np(x[:, None], W1, b1, 1, 3), 0)
        f = conv_np(h, W2, b2, 2, 2) + conv_np(hole_state(x), WH, bH, 2, 3)
        out.append(f.reshape(len(f), -1))
    return np.vstack(out)


def normalise(F):
    F = F - F.mean(1, keepdims=True)
    return F / np.sqrt((F ** 2).mean(1, keepdims=True) + 1e-5)


def build(out_dir, counts=N_CLASS):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layers = feature_layers()
    W1, b1, W2, b2, WH, bH = layers
    C = W2.shape[0]
    net = [
        {"name": "edges", "op": "conv", "inputs": ["image"], "out": int(W1.shape[0]), "kernel": [7, 7], "stride": 1, "padding": 3},
        {"name": "edges_r", "op": "relu", "inputs": ["edges"]},
        {"name": "pooled_e", "op": "conv", "inputs": ["edges_r"], "out": int(C), "kernel": [5, 5], "stride": 2, "padding": 2},
        # soft ink mask m and the constant maps used by the flood fill
        {"name": "ink_a", "op": "conv", "inputs": ["image"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "ink_r", "op": "relu", "inputs": ["ink_a"]},
        {"name": "inkmap", "op": "conv", "inputs": ["ink_r"], "out": 4, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "z_init", "op": "conv", "inputs": ["image"], "out": 4, "kernel": [1, 1], "stride": 1, "padding": 0},
    ]
    weights = {"edges.W": W1, "edges.b": b1, "pooled_e.W": W2, "pooled_e.b": b2}
    weights["ink_a.W"] = np.full((2, 1, 1, 1), INK_GAIN)
    weights["ink_a.b"] = np.array([-INK_GAIN * INK_LO, -INK_GAIN * INK_LO - 1.0])
    Wi = np.zeros((4, 2, 1, 1)); Wi[1, :, 0, 0] = Wi[3, :, 0, 0] = [1.0, -1.0]
    weights["inkmap.W"], weights["inkmap.b"] = Wi, np.array([1.0, 0.0, 1.0, 0.0])     # [1, m, 1, m]
    weights["z_init.W"], weights["z_init.b"] = np.zeros((4, 1, 1, 1)), np.array([0.0, 0.0, 1.0, 0.0])
    Wp = np.zeros((4, 4, 1, 1)); Wp[0, :3, 0, 0] = Wp[1, :3, 0, 0] = [1.0, -1.0, -1.0]   # Q = z0 - z1 - z2, zero outside
    prev = "z_init"
    for k in range(HOLE_ITERS):
        net += [{"name": f"fill{k}_pad", "op": "conv", "inputs": [prev], "out": 4, "kernel": [1, 1], "stride": 1, "padding": 1},
                {"name": f"fill{k}_max", "op": "maxpool", "inputs": [f"fill{k}_pad"], "size": [3, 3], "stride": [1, 1]},
                {"name": f"fill{k}_add", "op": "add", "inputs": [f"fill{k}_max", "inkmap"]},
                {"name": f"fill{k}", "op": "relu", "inputs": [f"fill{k}_add"]}]
        weights[f"fill{k}_pad.W"], weights[f"fill{k}_pad.b"] = Wp, np.zeros(4)
        prev = f"fill{k}"
    net += [{"name": "pooled_h", "op": "conv", "inputs": [prev], "out": int(C), "kernel": [7, 7], "stride": 2, "padding": 3},
            {"name": "pooled", "op": "add", "inputs": ["pooled_e", "pooled_h"]}]
    weights["pooled_h.W"], weights["pooled_h.b"] = WH, bH
    D = C * 14 * 14
    net += [{"name": "flat", "op": "reshape", "inputs": ["pooled"], "shape": [1, 1, D]},
            {"name": "unit", "op": "norm", "inputs": ["flat"]}]
    # stage 1: correlation with every template of each class, and each class's best match
    for c in range(10):
        n = int(counts[c])
        F = normalise(features(synth_class(c, n), layers))
        assert F.shape[1] == D
        net += [{"name": f"match{c}", "op": "dense", "inputs": ["unit"], "out": n},
                {"name": f"match{c}_3", "op": "reshape", "inputs": [f"match{c}"], "shape": [1, 1, n]},
                {"name": f"best{c}", "op": "maxpool", "inputs": [f"match{c}_3"], "size": [1, n], "stride": [1, n]},
                {"name": f"place{c}", "op": "dense", "inputs": [f"best{c}"], "out": 10}]
        weights[f"match{c}.W"] = F / D            # dot product with the normalised input = correlation
        weights[f"match{c}.b"] = np.zeros(n)
        P = np.zeros((10, 1)); P[c, 0] = 1.0
        weights[f"place{c}.W"], weights[f"place{c}.b"] = P, np.zeros(10)
    # stage 2: overall best correlation M; each class scores the share of its templates within DELTA of M,
    # weighted by how close they are: mean over templates of relu(corr - M + DELTA)
    net += [{"name": "bestvec", "op": "add", "inputs": [f"place{c}" for c in range(10)]},
            {"name": "bestvec3", "op": "reshape", "inputs": ["bestvec"], "shape": [1, 1, 10]},
            {"name": "top", "op": "maxpool", "inputs": ["bestvec3"], "size": [1, 10], "stride": [1, 10]}]
    for c in range(10):
        n = int(counts[c])
        net += [{"name": f"thr{c}", "op": "dense", "inputs": ["top"], "out": n},
                {"name": f"near{c}", "op": "add", "inputs": [f"match{c}", f"thr{c}"]},
                {"name": f"near{c}_r", "op": "relu", "inputs": [f"near{c}"]},
                {"name": f"near{c}_3", "op": "reshape", "inputs": [f"near{c}_r"], "shape": [1, 1, n]},
                {"name": f"share{c}", "op": "avgpool", "inputs": [f"near{c}_3"], "size": [1, n], "stride": [1, n]},
                {"name": f"vote{c}", "op": "dense", "inputs": [f"share{c}"], "out": 10}]
        weights[f"thr{c}.W"], weights[f"thr{c}.b"] = -np.ones((n, 1)), np.full(n, DELTA)
        P = np.zeros((10, 1)); P[c, 0] = 1.0 / DELTA
        weights[f"vote{c}.W"], weights[f"vote{c}.b"] = P, np.zeros(10)
    net.append({"name": "scores", "op": "add", "inputs": [f"vote{c}" for c in range(10)]})
    (out_dir / "network.json").write_text(json.dumps({"layers": net}, indent=1))
    np.savez(out_dir / "weights.npz", **{k: v.astype(np.float32) for k, v in weights.items()})


if __name__ == "__main__":
    build(sys.argv[1])
