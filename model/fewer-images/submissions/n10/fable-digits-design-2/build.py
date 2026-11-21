"""build.py -- writes network.json and weights.npz for a hand-designed digit classifier.

Nothing here is trained or fitted and no file is read.  The network is

  image -> soft binarisation -> "ray" sums (is there ink to the left / right / above / below?)
        -> 8 gradient-direction maps (Gaussian-derivative filters, formula)  +  11 concavity-code maps
        -> Gaussian pooling to a 6x6 grid
        -> one dense layer whose rows are templates -> max over the templates of each class.

Every template row is the feature map of ONE digit image drawn by the stroke model below (pen paths typed from
general knowledge of how digits are written, varied with a fixed-seed random generator), normalised by itself
alone (divided by its own length).  No statistic over several generated images is used anywhere.
"""
import sys, json, os
import numpy as np

SEED = 20261120
K_PER_CLASS = 15000         # templates per class
P = 100                     # points sampled along the pen path of one template
CENTRE = 13.75              # centre of mass target (frame centre, between 13.5 and 14.0 conventions)

# ----------------------------------------------------------------------------------------------------------
# pen-path helpers
# ----------------------------------------------------------------------------------------------------------

def cr(pts, m=8):
    """Catmull-Rom spline through the given points (a smooth pen path)."""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 3:
        t = np.linspace(0.0, 1.0, 2 * m)[:, None]
        return p[0] + (p[-1] - p[0]) * t
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = np.linspace(0.0, 1.0, m, endpoint=False)[None, :, None]
    p0, p1, p2, p3 = q[:-3, None, :], q[1:-2, None, :], q[2:-1, None, :], q[3:, None, :]
    s = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
    return np.vstack([s.reshape(-1, 2), p[-1:]])


def seg(a, b, bow=0.0, m=14):
    """Straight pen stroke from a to b, optionally bowed sideways (bow is a fraction of its length)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    t = np.linspace(0.0, 1.0, m)[:, None]
    d = b - a
    return a + d * t + np.array([-d[1], d[0]]) * bow * np.sin(np.pi * t)


def resample(strokes, n):
    """n points spread evenly by arc length over all strokes of a glyph."""
    lens = [np.sqrt((np.diff(s, axis=0) ** 2).sum(1)) for s in strokes]
    tot = np.array([l.sum() for l in lens])
    cnt = np.maximum(2, np.floor(n * tot / tot.sum()).astype(int))
    while cnt.sum() > n: cnt[np.argmax(cnt)] -= 1
    while cnt.sum() < n: cnt[np.argmax(tot / cnt)] += 1
    out = []
    for s, l, c in zip(strokes, lens, cnt):
        cum = np.concatenate([[0.0], np.cumsum(l)])
        u = np.linspace(0.0, cum[-1], c)
        out.append(np.stack([np.interp(u, cum, s[:, 0]), np.interp(u, cum, s[:, 1])], 1))
    return np.vstack(out)

# ----------------------------------------------------------------------------------------------------------
# the ten digits as pen paths.  Frame: x to the right, y downwards, the digit roughly inside [0,1] x [0,1].
# Each function returns (list of strokes, width/height ratio, extra slant).
# All numbers are typed by hand from how the digits are normally written.
# ----------------------------------------------------------------------------------------------------------

def d0(R):
    U = R.uniform
    th0 = -U(-0.5, 0.9)                         # where the pen starts, measured from the top
    sweep = 2 * np.pi + U(-0.25, 0.7)           # a gap (negative) or an overlap where the loop closes
    th = th0 + np.linspace(0.0, sweep, 72)
    rho = 1 + R.normal(0, 0.035) * np.cos(2 * th + U(0, 6.3)) + R.normal(0, 0.025) * np.cos(3 * th + U(0, 6.3))
    rho = rho * (1 + U(-0.06, 0.12) * np.exp(-(th - th0) / 0.35))             # short lead-in from outside / inside
    rho = rho * (1 + U(-0.35, 0.15) * np.exp(-(th0 + sweep - th) / 0.35))     # the end often cuts inside
    y = 0.5 - 0.5 * np.cos(th) * rho
    egg = R.normal(0, 0.14)
    x = 0.5 - 0.5 * np.sin(th) * rho * (1 + egg * (y - 0.5))
    return [np.stack([x, y], 1)], U(0.45, 0.98), 0.0


def d1(R):
    U = R.uniform
    style = R.choice(4, p=[0.66, 0.25, 0.06, 0.03])
    top = np.array([0.5, 0.0]); bot = np.array([0.5 + R.normal(0, 0.02), 1.0])
    S = [seg(top, bot, bow=R.normal(0, 0.025), m=24)]
    if style in (1, 2):                          # a flag: short upstroke into the top
        fx = U(0.1, 0.26)
        S.insert(0, seg(top + [-fx, min(0.3, fx * U(0.8, 1.4))], top, bow=R.normal(0, 0.05)))
    if style in (2, 3):                          # a foot
        S.append(seg(bot + [-U(0.15, 0.4), U(-0.02, 0.02)], bot + [U(0.15, 0.4), U(-0.02, 0.02)]))
    return S, 1.0, (R.normal(0.15, 0.3) if style == 0 else R.normal(0.06, 0.13))


def d2(R):
    U = R.uniform
    sx, sy = U(0.04, 0.25), U(0.16, 0.38)        # pen starts at the upper left, below the top
    tx = U(0.4, 0.6)
    rx, ry = U(0.78, 0.96), U(0.17, 0.36)        # right-most point of the top arc
    bx, by = U(0.02, 0.2), U(0.9, 1.0)           # lower-left corner
    mx, my = (rx + bx) / 2 + U(-0.06, 0.16), (ry + by) / 2 + U(-0.06, 0.08)
    ex, ey = U(0.86, 1.06), by + U(-0.14, 0.06)  # end of the base
    top = [(sx, sy), (sx + 0.3 * (tx - sx), 0.3 * sy), (tx, 0.0), (rx - 0.1, 0.07), (rx, ry), (mx, my)]
    if U() < 0.3: top.insert(0, (sx + U(0.05, 0.12), sy + U(0.04, 0.12)))    # small curl at the start
    style = R.choice(4, p=[0.36, 0.31, 0.23, 0.10])
    if style == 0:                               # sharp corner, straight or swung base
        S = [cr(top + [(bx, by)]), seg((bx, by), (ex, ey), bow=R.normal(0.0, 0.07))]
    elif style == 3:                             # angular, Z-like: bar, corner, diagonal, corner, base
        zy = U(0.0, 0.08); zc = (U(0.8, 0.98), zy + U(-0.03, 0.05))
        S = [seg((sx, zy + U(0.0, 0.08)), zc, bow=R.normal(-0.04, 0.04)), seg(zc, (bx, by), bow=R.normal(0, 0.04)),
             seg((bx, by), (ex, ey), bow=R.normal(0.0, 0.05))]
    elif style == 1:                             # a loop at the lower left, base leaves the loop to the right
        lw, lh = U(0.1, 0.42), U(0.1, 0.36)
        S = [cr(top + [(bx + 0.02, by), (bx - 0.02, by - 0.55 * lh), (bx + 0.5 * lw, by - lh),
                       (bx + lw + 0.1, by - 0.2 * lh), (0.5 * (bx + lw + 0.1 + ex), ey + U(-0.02, 0.07)), (ex, ey)])]
    else:                                        # rounded turn into the base
        S = [cr(top + [(bx, by - 0.02), (0.5 * (bx + ex), by + U(-0.04, 0.04)), (ex, ey)])]
    return S, U(0.55, 1.05), 0.0


def d3(R):
    U = R.uniform
    sx, sy = U(0.05, 0.3), U(0.06, 0.2)
    wt = U(0.56, 0.92)                           # right edge of the upper bowl
    cx, cy = U(0.22, 0.56), U(0.36, 0.56)        # the waist, where the two bowls meet
    wb = U(0.78, 1.02)                           # right edge of the lower bowl
    ex, ey = U(-0.2, 0.16), U(0.74, 0.92)        # the tail at the lower left, often sweeping far to the left
    top = R.choice(3, p=[0.62, 0.28, 0.10])
    if top == 1:                                 # flat top: bar, corner, diagonal to the waist
        tyl, tyr = U(0.0, 0.07), U(0.0, 0.07)
        A = [seg((sx, tyl), (wt, tyr), bow=R.normal(0, 0.03)), seg((wt, tyr), (cx, cy), bow=R.normal(0.03, 0.05))]
        if U() < 0.25: A.insert(0, seg((sx, tyl + U(0.06, 0.16)), (sx, tyl)))     # tick at the start of the bar
    elif top == 0:
        A = [cr([(sx, sy), (0.5 * (sx + wt) - 0.05, 0.0), (wt - 0.06, 0.07), (wt, 0.45 * cy),
                 (wt - 0.13, 0.86 * cy), (cx, cy)])]
    else:                                        # top begins with a small curl
        A = [cr([(sx + U(0.03, 0.1), sy + U(0.05, 0.12)), (sx, sy), (0.5 * (sx + wt) - 0.05, 0.0), (wt - 0.06, 0.07),
                 (wt, 0.45 * cy), (wt - 0.13, 0.86 * cy), (cx, cy)])]
    low = [(wb - 0.16, cy + 0.05), (wb, cy + 0.45 * (1 - cy)), (wb - 0.14, 0.94), (0.45, 1.0),
           (ex + 0.12, 0.97), (ex, ey)]
    if U() < 0.3: low.append((ex + U(0.0, 0.08), ey - U(0.04, 0.14)))         # tail curls up
    if top != 1 and U() < 0.15:                  # the pen makes a small loop at the waist instead of a cusp
        lw = U(0.06, 0.14)
        B = cr(list(A.pop()[::6]) + [(cx - lw, cy + 0.03), (cx - 0.5 * lw, cy + 0.09), (cx + 0.06, cy + 0.03)] + low)
    else:
        B = cr([(cx, cy)] + low)
    return A + [B], U(0.55, 1.05), 0.0


def d4(R):
    U = R.uniform
    style = R.choice(3, p=[0.55, 0.2, 0.25])
    stx = U(0.56, 0.8)                           # the stem
    lean = U(-0.04, 0.1)
    st_top = (stx + lean, U(0.0, 0.2)); st_bot = (stx - lean, 1.0)
    bar_y = U(0.5, 0.72)
    lbx = U(0.0, 0.16)                           # left end of the bar
    bar_e = (U(0.86, 1.06), bar_y + U(-0.08, 0.05))
    if style != 1 and U() < 0.1: bar_e = (stx + U(-0.02, 0.05), bar_e[1])     # the bar stops at the stem
    stem = seg(st_top, st_bot, bow=R.normal(0, 0.03), m=22)
    if style == 0:                               # open top: left stroke, corner, bar
        lt = (lbx + U(0.0, 0.3), U(0.0, 0.16))
        S = [seg(lt, (lbx, bar_y), bow=R.normal(0, 0.04)), seg((lbx, bar_y), bar_e, bow=R.normal(0, 0.03)), stem]
    elif style == 1:                             # closed top: the left stroke starts at the top of the stem
        S = [seg((st_top[0] + U(-0.06, 0.02), st_top[1] + U(0.0, 0.06)), (lbx, bar_y), bow=R.normal(0, 0.04)),
             seg((lbx, bar_y), bar_e, bow=R.normal(0, 0.03)), stem]
    else:                                        # rounded, like a u with a stem
        lt = (lbx + U(0.0, 0.22), U(0.0, 0.16))
        S = [cr([lt, (lbx + 0.02, bar_y - 0.22), (lbx + 0.07, bar_y - 0.03), (0.42, bar_y + U(-0.03, 0.03)), bar_e]), stem]
    return S, U(0.6, 1.0), 0.0


def d5(R):
    U = R.uniform
    vt = (U(0.2, 0.42), U(0.0, 0.06))            # top of the down-stroke
    vb = (vt[0] + U(-0.16, 0.05), U(0.34, 0.5))  # where the bowl starts
    bx = U(0.74, 0.96)                           # right edge of the bowl
    ex, ey = U(0.0, 0.2), U(0.78, 0.95)          # the tail
    bowl = [vb, (vb[0] + 0.26, vb[1] - U(0.02, 0.08)), (bx, vb[1] + 0.42 * (1 - vb[1])), (bx - 0.13, 0.93),
            (0.45, 1.0), (ex + 0.1, 0.97), (ex, ey)]
    if U() < 0.65: body = [seg(vt, vb, bow=R.normal(0, 0.04)), cr(bowl)]      # sharp junction
    else: body = [cr([vt, (0.5 * (vt[0] + vb[0]), 0.5 * (vt[1] + vb[1]))] + bowl)]   # one flowing S-like stroke
    bs = (vt[0] + (U(0.0, 0.12) if U() < 0.3 else 0.0), vt[1] + U(-0.03, 0.03))  # bar, sometimes detached
    bar = seg(bs, (U(0.76, 1.05), vt[1] + U(-0.18, 0.06)), bow=R.normal(0, 0.05))
    return body + [bar], U(0.55, 1.1), 0.0


def d6(R):
    U = R.uniform
    sx = U(0.32, 0.95)                           # pen starts at the top, usually right of the middle
    lx = U(0.06, 0.22)                           # left-most point
    lt = U(0.38, 0.64)                           # top of the loop: the stem above it stays clearly visible
    rx = U(0.7, 0.96)                            # right edge of the loop
    c = U(0.0, 0.18)                             # how much the down-stroke is curved
    pts = [(sx, 0.0), (0.5 * (sx + lx) + 0.08 - c, 0.22), (lx + 0.06, 0.46), (lx, 0.7), (lx + 0.12, 0.93),
           (0.5 * (lx + rx), 1.0), (rx - 0.06, 0.92), (rx, 0.5 * (lt + 1.0) + 0.02), (rx - 0.1, lt + 0.04),
           (0.5 * (lx + rx), lt), (lx + U(0.04, 0.16), lt + U(0.06, 0.2))]
    if U() < 0.3: pts.insert(0, (sx + U(0.04, 0.12), U(0.02, 0.1)))           # small hook at the start
    if U() < 0.3: pts.append((pts[-1][0] + U(0.0, 0.1), pts[-1][1] + U(0.05, 0.15)))   # end curls into the loop
    return [cr(pts)], U(0.5, 0.9), 0.0


def d7(R):
    U = R.uniform
    lx, ly = U(0.0, 0.16), U(0.02, 0.12)
    cx, cy = U(0.8, 1.0), U(0.0, 0.1)            # the corner at the top right
    bx = U(0.22, 0.62)                           # foot of the stem
    bow = U(-0.07, 0.1)
    if U() < 0.7:
        S = [seg((lx, ly), (cx, cy), bow=R.normal(0, 0.04)), seg((cx, cy), (bx, 1.0), bow=bow, m=22)]
    else:                                        # rounded corner
        S = [cr([(lx, ly), (0.5 * (lx + cx), 0.5 * (ly + cy) - U(0.0, 0.05)), (cx - 0.04, cy + 0.01),
                 (cx - 0.08, cy + 0.2), (0.5 * (cx + bx) + bow * 0.5, 0.55), (bx, 1.0)])]
    cross = U() < 0.22
    if not cross and U() < 0.18: S.insert(0, seg((lx + U(-0.02, 0.03), ly + U(0.07, 0.17)), (lx, ly)))   # tick at the start of the bar
    if cross:                                    # crossbar through the stem
        t = U(0.42, 0.62); px, py = cx + t * (bx - cx), cy + t * (1.0 - cy)
        S.append(seg((px - U(0.18, 0.32), py + R.normal(0, 0.03)), (px + U(0.14, 0.3), py + R.normal(0, 0.03))))
    return S, U(0.55, 0.95), -0.12


def d8(R):
    U = R.uniform
    at, ab = U(0.2, 0.4), U(0.24, 0.44)          # half widths of the upper and lower loops
    yc = U(0.4, 0.56)                            # height of the crossing
    t0 = U(-0.2, 0.9); sweep = 2 * np.pi + U(-0.6, 0.3)
    t = t0 - np.linspace(0.0, sweep, 84)         # start near the top (right of it), go left first
    x0, y0 = np.sin(2 * t), -np.cos(t)
    y = yc + np.where(y0 < 0, y0 * yc, y0 * (1 - yc))
    x = 0.5 + np.where(y0 < 0, at, ab) * x0 + np.where(y0 < 0, U(-0.05, 0.08), U(-0.08, 0.05)) * np.abs(y0)
    pts = np.stack([x, y], 1)
    if U() < 0.25:                               # the closing upstroke overshoots as a short tail
        d = pts[-1] - pts[-4]; d = d / (np.sqrt((d ** 2).sum()) + 1e-9)
        pts = np.vstack([pts, pts[-1] + d * U(0.05, 0.22)])
    return [pts], U(0.5, 0.88), 0.0


def d9(R):
    U = R.uniform
    lrx, lry = U(0.34, 0.5), U(0.15, 0.3)        # the loop spans the width of the digit
    lcx, lcy = lrx + U(0.0, 0.06), lry
    a0 = U(-0.2, 0.9)                            # start on the right side of the loop (angle above 3 o'clock)
    sweep = 2 * np.pi + U(-0.55, 0.15)           # the loop may stay a little open at the upper right
    a = a0 + np.linspace(0.0, sweep, 48)
    rho = 1 + R.normal(0, 0.04) * np.cos(2 * a + U(0, 6.3))
    loop = np.stack([lcx + lrx * rho * np.cos(a), lcy - lry * rho * np.sin(a)], 1)
    jx = lcx + lrx * U(0.85, 1.05)               # the stem runs down the right side of the loop
    jy = lcy + lry * U(-0.9, 0.1)
    kind = R.choice(3, p=[0.5, 0.3, 0.2])
    if kind == 0:                                # straight stem
        stem = seg((jx, jy), (jx + U(-0.5, 0.04), 1.0), bow=R.normal(0, 0.03), m=22)
    elif kind == 1:                              # stem ends in a hook to the left
        hx = U(0.1, 0.45)
        stem = cr([(jx, jy), (jx + U(-0.04, 0.02), 0.55), (jx - 0.08, 0.84), (0.5 * (jx + hx), 0.98), (hx, U(0.86, 0.97))])
    else:                                        # gently curved stem
        stem = cr([(jx, jy), (jx + U(-0.05, 0.03), 0.5), (jx - U(0.05, 0.2), 0.8), (jx - U(0.15, 0.55), 1.0)])
    return [loop, stem], U(0.42, 0.8), 0.0


DIGITS = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]


def glyph(c, R):
    """One random handwritten digit of class c: P points in pixel units (before centring) and the pen half-width."""
    strokes, aspect, slant0 = DIGITS[c](R)
    U = R.uniform
    pts = resample(strokes, P)
    # proportions: the middle of the glyph sits higher or lower, and the glyph tapers upwards or downwards
    gam = np.exp(R.normal(0, 0.14)); q = R.normal(0, 0.12)
    yy = np.sign(pts[:, 1]) * np.abs(pts[:, 1]) ** gam
    xx = np.sign(pts[:, 0]) * np.abs(pts[:, 0]) ** np.exp(R.normal(0, 0.1))
    pts = np.stack([0.5 + (xx - 0.5) * (1 + q * (yy - 0.5)), yy], 1)
    # smooth, low-frequency bending of the whole glyph (the wobble of the hand)
    jit = U(0.01, 0.045)
    for _ in range(3):
        f = np.clip(R.normal(0, 0.6, 2), -1.0, 1.0); ph = U(0, 6.3); amp = np.clip(R.normal(0, jit, 2), -2 * jit, 2 * jit)
        pts = pts + amp[None, :] * np.sin(2 * np.pi * (pts @ f) + ph)[:, None]
    x = (pts[:, 0] - 0.5) * aspect; y = pts[:, 1] - 0.5
    k = slant0 + (R.normal(0.12, 0.2) if c != 1 else 0.0)       # slant: handwriting usually leans to the right
    k = float(np.clip(k, -0.5, 1.0))
    x = x - k * y
    y = y - R.normal(0.04, 0.11) * x             # tilt of the writing line: horizontal strokes often rise to the right
    r = float(np.clip(np.exp(R.normal(np.log(1.45), 0.35)), 0.7, 2.5))   # pen half-width in pixels, thin to bold
    ext = max(x.max() - x.min(), y.max() - y.min())
    s = (20.0 - 2 * r) / ext                     # the inked box is 20 pixels on its longer side
    x = (x - 0.5 * (x.max() + x.min())) * s; y = (y - 0.5 * (y.max() + y.min())) * s
    return np.stack([x, y], 1), r


def render(pts, r, soft=1.2):
    """pts (B,P,2) pixel coordinates (x,y), r (B,) half-widths -> images (B,28,28) with soft stroke edges."""
    g = np.arange(28, dtype=np.float32)
    dx2 = (g[None, :, None] - pts[:, None, :, 0].astype(np.float32)) ** 2      # (B,28,P)
    dy2 = (g[None, :, None] - pts[:, None, :, 1].astype(np.float32)) ** 2
    d = np.sqrt((dy2[:, :, None, :] + dx2[:, None, :, :]).min(-1))             # (B,28,28) distance to the path
    return np.clip((r[:, None, None].astype(np.float32) - d) / soft + 0.5, 0.0, 1.0)


def draw(pts, r, batch=250):
    """Render glyphs with the centre of mass at the frame centre (render, measure, shift, render again)."""
    out = np.empty((len(pts), 28, 28), dtype=np.float32)
    g = np.arange(28, dtype=np.float64)
    for i in range(0, len(pts), batch):
        p = pts[i:i + batch] + 13.5; rr = r[i:i + batch]
        for _ in range(2):
            im = render(p, rr).astype(np.float64)
            m = im.sum((1, 2))
            cx = (im.sum(1) * g).sum(1) / m; cy = (im.sum(2) * g).sum(1) / m
            p = p + np.stack([CENTRE - cx, CENTRE - cy], 1)[:, None, :]
        out[i:i + batch] = render(p, rr)
    return out


def make_glyphs(k_per_class, seed=SEED):
    R = np.random.default_rng(seed)
    pts = np.empty((10 * k_per_class, P, 2)); r = np.empty(10 * k_per_class)
    for c in range(10):
        for j in range(k_per_class):
            pts[c * k_per_class + j], r[c * k_per_class + j] = glyph(c, R)
    return pts, r

# ----------------------------------------------------------------------------------------------------------
# the network: fixed filters from formulas
# ----------------------------------------------------------------------------------------------------------
BIN_GAIN, BIN_OFF = 2.0, 0.2        # soft binarisation: s = clip(2*x - 0.2, 0, 1)
SIG_D = 1.0                         # Gaussian-derivative filter (5x5)
SIG_G, SIG_C = 2.0, 2.5             # pooling Gaussians for gradient maps and concavity maps (12x12, stride 4)
GAIN_INK = 0.2                      # weight of the plain ink map
GAIN_C3, GAIN_C2 = 0.32, 0.22       # weight of concavity codes closed on >=3 sides / on exactly 2 sides
CODES = [s for s in range(16) if bin(s).count("1") >= 2]     # subsets of {left,right,up,down} with ink in sight


def build_front():
    """Layers and weights of the fixed front end.  Returns (layers, weights)."""
    L, W = [], {}
    # 1. soft binarisation needs two relus: s = relu(a x - b) - relu(a x - b - 1)
    L.append({"name": "c0", "op": "conv", "inputs": ["image"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0})
    W["c0.W"] = np.full((2, 1, 1, 1), BIN_GAIN); W["c0.b"] = np.array([-BIN_OFF, -BIN_OFF - 1.0])
    L.append({"name": "r0", "op": "relu", "inputs": ["c0"]})
    # 2. rays: channel 0 = s; channels 1..4 = amount of ink strictly to the left / right / above / below
    eye = np.eye(28); lo = np.tril(np.ones((28, 28)), -1)          # lo[x, x'] = 1 if x' < x
    maps = [np.kron(eye, eye), np.kron(eye, lo), np.kron(eye, lo.T), np.kron(lo, eye), np.kron(lo.T, eye)]
    M = np.vstack(maps)                                             # (5*784, 784) acting on s
    W["rays.W"] = np.hstack([M, -M]); W["rays.b"] = np.zeros(5 * 784)
    L.append({"name": "rays", "op": "dense", "inputs": ["r0"], "out": 5 * 784})
    L.append({"name": "rs", "op": "reshape", "inputs": ["rays"], "shape": [5, 28, 28]})
    # 3. f1: 8 directional derivatives of s, s itself, and the two relu halves of clip(ray, 0, 1)
    ax = np.arange(5) - 2.0
    X, Y = np.meshgrid(ax, ax)                                      # X along columns, Y along rows
    G = np.exp(-(X ** 2 + Y ** 2) / (2 * SIG_D ** 2))
    gx = X * G / (X * X * G).sum(); gy = Y * G / (Y * Y * G).sum()  # unit response to a unit ramp
    f1 = np.zeros((17, 5, 5, 5)); b1 = np.zeros(17)
    for d in range(8):
        th = 2 * np.pi * d / 8
        f1[d, 0] = np.cos(th) * gx + np.sin(th) * gy
    f1[8, 0, 2, 2] = 1.0
    for i in range(4):
        f1[9 + i, 1 + i, 2, 2] = 1.0
        f1[13 + i, 1 + i, 2, 2] = 1.0; b1[13 + i] = -1.0
    W["f1.W"], W["f1.b"] = f1, b1
    L.append({"name": "f1", "op": "conv", "inputs": ["rs"], "out": 17, "kernel": [5, 5], "stride": 1, "padding": 2})
    L.append({"name": "r1", "op": "relu", "inputs": ["f1"]})
    # 4. f2: sharpen the direction tuning (gradient split between its two neighbouring directions),
    #        and concavity codes: background pixel that sees ink exactly in the directions of the subset
    f2 = np.zeros((9 + len(CODES), 17, 1, 1)); b2 = np.zeros(9 + len(CODES))
    for d in range(8):
        f2[d, d, 0, 0] = 1.0; f2[d, (d + 2) % 8, 0, 0] = -1.0; f2[d, (d - 2) % 8, 0, 0] = -1.0
    f2[8, 8, 0, 0] = 1.0
    for j, sub in enumerate(CODES):
        o = 9 + j
        for i in range(4):
            sign = 1.0 if (sub >> i) & 1 else -1.0
            f2[o, 9 + i, 0, 0] = sign; f2[o, 13 + i, 0, 0] = -sign      # clip(ray) = relu(ray) - relu(ray - 1)
        f2[o, 8, 0, 0] = -1.0                                            # not on ink
        b2[o] = -(bin(sub).count("1") - 1.0)
    W["f2.W"], W["f2.b"] = f2, b2
    L.append({"name": "f2", "op": "conv", "inputs": ["r1"], "out": 9 + len(CODES), "kernel": [1, 1], "stride": 1, "padding": 0})
    L.append({"name": "r2", "op": "relu", "inputs": ["f2"]})
    # 5. Gaussian pooling to a 6x6 grid (window centres 3.5, 7.5, ..., 23.5: the digit box and its margin)
    a12 = np.arange(12) - 5.5
    def g2(sig):
        g = np.exp(-a12 ** 2 / (2 * sig ** 2)); g = np.outer(g, g); return g / g.sum()
    nc = 9 + len(CODES)
    pg = np.zeros((9, nc, 12, 12))
    for d in range(8): pg[d, d] = g2(SIG_G)
    pg[8, 8] = GAIN_INK * g2(SIG_G)
    W["pg.W"], W["pg.b"] = pg, np.zeros(9)
    L.append({"name": "pg", "op": "conv", "inputs": ["r2"], "out": 9, "kernel": [12, 12], "stride": 4, "padding": 2})
    pc = np.zeros((len(CODES), nc, 12, 12))
    for j, sub in enumerate(CODES):
        pc[j, 9 + j] = (GAIN_C3 if bin(sub).count("1") >= 3 else GAIN_C2) * g2(SIG_C)
    W["pc.W"], W["pc.b"] = pc, np.zeros(len(CODES))
    L.append({"name": "mp", "op": "maxpool", "inputs": ["r2"], "size": [3, 3], "stride": [1, 1]})
    L.append({"name": "pc", "op": "conv", "inputs": ["mp"], "out": len(CODES), "kernel": [12, 12], "stride": 4, "padding": 3})
    return L, {k: v.astype(np.float32) for k, v in W.items()}


# a small interpreter with the same arithmetic as the task's building blocks (used to turn each drawn digit
# into its own template, and for self-checks)
def conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def pool(x, size, stride, reduce):
    (sh, sw), (th, tw) = size, stride
    ho, wo = (x.shape[2] - sh) // th + 1, (x.shape[3] - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if reduce == "max" else win.mean(0)


def run(layers, weights, images, want=None):
    vals = {"image": images}
    for L in layers:
        ins = [vals[i] for i in L["inputs"]]; op, name = L["op"], L["name"]
        if op == "conv": vals[name] = conv(ins[0], weights[name + ".W"], weights[name + ".b"], L.get("stride", 1), L.get("padding", 0))
        elif op == "dense": vals[name] = ins[0].reshape(len(ins[0]), -1) @ weights[name + ".W"].T + weights[name + ".b"]
        elif op == "relu": vals[name] = np.maximum(ins[0], 0)
        elif op in ("maxpool", "avgpool"): vals[name] = pool(ins[0], L["size"], L["stride"], op[:3])
        elif op == "add": vals[name] = sum(ins)
        elif op == "reshape": vals[name] = ins[0].reshape((len(ins[0]),) + tuple(L["shape"]))
        elif op == "norm":
            mu = ins[0].mean(axis=(2, 3), keepdims=True)
            vals[name] = (ins[0] - mu) / np.sqrt(((ins[0] - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
    return vals if want is None else [vals[w] for w in want]


def features(layers, weights, ims, batch=500):
    """Front-end features of images (N,28,28): (N, 9*36) gradient part and (N, 11*36) concavity part."""
    fg, fc = [], []
    for i in range(0, len(ims), batch):
        a, b = run(layers, weights, ims[i:i + batch, None].astype(np.float32), want=["pg", "pc"])
        fg.append(a.reshape(len(a), -1)); fc.append(b.reshape(len(b), -1))
    return np.concatenate(fg), np.concatenate(fc)


def build(k_per_class=K_PER_CLASS, seed=SEED, verbose=True):
    layers, W = build_front()
    pts, r = make_glyphs(k_per_class, seed)
    if verbose: print("glyphs drawn", flush=True)
    ims = draw(pts, r)
    if verbose: print("glyphs rendered", flush=True)
    fg, fc = features(layers, W, ims)
    # each drawn digit is its own template, divided by its own length (nothing is shared between templates)
    nrm = np.sqrt((fg.astype(np.float64) ** 2).sum(1) + (fc.astype(np.float64) ** 2).sum(1))[:, None]
    T = 10 * k_per_class
    W["dg.W"] = (fg / nrm).astype(np.float32); W["dg.b"] = np.zeros(T, np.float32)
    W["dc.W"] = (fc / nrm).astype(np.float32); W["dc.b"] = np.zeros(T, np.float32)
    layers.append({"name": "dg", "op": "dense", "inputs": ["pg"], "out": T})
    layers.append({"name": "dc", "op": "dense", "inputs": ["pc"], "out": T})
    layers.append({"name": "sim", "op": "add", "inputs": ["dg", "dc"]})
    layers.append({"name": "byclass", "op": "reshape", "inputs": ["sim"], "shape": [10, 1, k_per_class]})
    layers.append({"name": "best", "op": "maxpool", "inputs": ["byclass"], "size": [1, k_per_class], "stride": [1, k_per_class]})
    layers.append({"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]})
    return layers, W


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    layers, W = build()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **W)
    print("wrote", sum(v.size for v in W.values()), "weights")
