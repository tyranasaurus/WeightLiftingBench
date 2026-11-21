#!/usr/bin/env python3
"""Builds a hand-designed sign-language classifier.  python3 build.py OUT_DIR

Nothing here is trained or fitted and no file is read.  Every weight comes from
  (a) formulas: Gaussian-derivative edge filters, an orientation-sharpening rule, a fixed
      compressive curve, a Gaussian blur;
  (b) templates: drawings of a schematic hand (palm box, finger capsules, thumb, forearm) posed by
      hand for each of the 24 letters from general knowledge of the manual alphabet.  Each drawing
      is passed through the same front-end as the network uses and becomes one template of the
      matching layer, normalised by itself alone.
"""
import sys, json, math
import numpy as np

NAMES = "ABCDEFGHIKLMNOPQRSTUVWXY"
SIZE = 28
SS = 4                      # supersampling of the drawings

# ----------------------------------------------------------------------------------------------
# front-end constants (chosen by hand)
# ----------------------------------------------------------------------------------------------
NDIR = 12                   # signed gradient directions, 30 degrees apart
SIG_G = 1.0                 # Gaussian-derivative scale (pixels)
KG = 5                      # derivative kernel size
SHARP = math.sqrt(3.0)      # orientation sharpening: response falls to 0 at 30 degrees off
T0 = 0.06                   # noise floor on the (z-scored) edge strength
CA, CB = 0.4, 1.2           # knees of the compressive curve
SA, SB = 0.6, 0.4           # slope removed at each knee (slope 1 -> 0.4 -> 0)
LAM = 0.4                   # share of the opposite-polarity edge mixed in
SIG_B = 1.3                 # blur before 2x subsampling (pixels): narrow part
SIG_B2 = 2.6                # ... and wide part
MIX_B2 = 0.4                # share of the wide part
KB = 12                     # blur kernel size
FM = 12                     # feature map side after subsampling
OPP = 0.0                   # how much of the mean over orientations is removed at each place
R_IN, R_OUT = 0.3, 0.3      # template weight deep inside the hand and far out in the background
TS, CS = 0.2, 1.0           # figure map: dead zone and ceiling, in image standard deviations
SIG_S = 1.6                 # figure map blur (pixels)
BROWS, BCOLS, BDEPTH = 2, 2, 18   # wall sample: top rows, and side columns down to this row
BSHRINK = 40.0              # damping of the fitted wall slopes
W_EDGE, W_FIG = 1.0, 1.0    # weights of the two correlations in the score
PADT = 2                    # template shifts: +-2 cells (= +-4 pixels)
NVAR = 400                  # drawings per letter


# ----------------------------------------------------------------------------------------------
# the eight building blocks, as the scorer defines them (used to turn drawings into templates)
# ----------------------------------------------------------------------------------------------
def conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def norm(x):
    mu = x.mean(axis=(2, 3), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)


# ----------------------------------------------------------------------------------------------
# front-end weights from formulas
# ----------------------------------------------------------------------------------------------
def front_weights():
    w = {}
    r = KG // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * SIG_G ** 2))
    z = (xx * g)[xx > 0].sum()                       # a unit step edge answers 1
    W = np.zeros((NDIR, 1, KG, KG))
    for k in range(NDIR):
        th = 2 * math.pi * k / NDIR
        W[k, 0] = (math.cos(th) * xx + math.sin(th) * yy) * g / z
    w["grad.W"], w["grad.b"] = W, np.zeros(NDIR)
    # sharpen: s_k = r_k - c (r_{k+90deg} + r_{k-90deg}) - t0
    q = NDIR // 4
    W = np.zeros((NDIR, NDIR, 1, 1))
    for k in range(NDIR):
        W[k, k, 0, 0] = 1.0
        W[k, (k + q) % NDIR, 0, 0] -= SHARP
        W[k, (k - q) % NDIR, 0, 0] -= SHARP
    w["sharp.W"], w["sharp.b"] = W, np.full(NDIR, -T0)
    # knees: channels [u, u - a, u - b]
    W = np.zeros((3 * NDIR, NDIR, 1, 1)); b = np.zeros(3 * NDIR)
    for k in range(NDIR):
        W[k, k, 0, 0] = 1.0
        W[NDIR + k, k, 0, 0] = 1.0; b[NDIR + k] = -CA
        W[2 * NDIR + k, k, 0, 0] = 1.0; b[2 * NDIR + k] = -CB
    w["knee.W"], w["knee.b"] = W, b
    # compress + polarity mix + blur + subsample
    c = (KB - 1) / 2.0
    yy, xx = np.mgrid[0:KB, 0:KB].astype(np.float64)
    g1 = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * SIG_B ** 2)); g1 /= g1.sum()
    g2 = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * SIG_B2 ** 2)); g2 /= g2.sum()
    gb = (1 - MIX_B2) * g1 + MIX_B2 * g2
    W = np.zeros((NDIR, 3 * NDIR, KB, KB))
    for k in range(NDIR):
        for kk, a in ((k, 1.0), ((k + NDIR // 2) % NDIR, LAM)):
            W[k, kk] += a * gb
            W[k, NDIR + kk] -= a * SA * gb
            W[k, 2 * NDIR + kk] -= a * SB * gb
    w["feat.W"], w["feat.b"] = W, np.zeros(NDIR)
    # orientation opponency: at every place subtract the mean over the orientations
    w["opp.W"] = (np.eye(NDIR) - OPP / NDIR).reshape(NDIR, NDIR, 1, 1)
    w["opp.b"] = np.zeros(NDIR)
    # wall removal: subtract the plane through the picture's upper border (closed form in pixel coordinates)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    inb = (yy < BROWS) | (((xx < BCOLS) | (xx >= SIZE - BCOLS)) & (yy < BDEPTH))
    A = np.stack([np.ones(SIZE * SIZE), (xx.ravel() - 13.5) / 13.5, (yy.ravel() - 13.5) / 13.5], axis=1)
    AB = A * inb.ravel()[:, None]
    G = AB.T @ AB + np.diag([0.0, BSHRINK, BSHRINK])
    w["wall.W"] = np.eye(SIZE * SIZE) - A @ np.linalg.solve(G, AB.T)
    w["wall.b"] = np.zeros(SIZE * SIZE)
    # figure map: min(relu(|r| - TS), CS), blurred and subsampled on the same grid as the edge map
    w["fig1.W"] = np.array([1.0, -1.0]).reshape(2, 1, 1, 1); w["fig1.b"] = np.full(2, -TS)
    w["fig2.W"] = np.ones((2, 2, 1, 1)); w["fig2.b"] = np.array([0.0, -CS])
    c = (KB - 1) / 2.0
    yy, xx = np.mgrid[0:KB, 0:KB].astype(np.float64)
    gs = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * SIG_S ** 2)); gs /= gs.sum()
    w["fig3.W"] = np.stack([gs, -gs])[None]; w["fig3.b"] = np.zeros(1)
    # put the two maps side by side as 13 channels
    W = np.zeros((NDIR + 1, NDIR, 1, 1)); W[:NDIR, :, 0, 0] = np.eye(NDIR)
    w["cat_e.W"], w["cat_e.b"] = W, np.zeros(NDIR + 1)
    W = np.zeros((NDIR + 1, 1, 1, 1)); W[NDIR, 0, 0, 0] = 1.0
    w["cat_f.W"], w["cat_f.b"] = W, np.zeros(NDIR + 1)
    return w


def front(images, w):
    """images (N,1,28,28) in 0..1 -> (N, NDIR+1, FM, FM): edge map and figure map; the same operations as the network."""
    n = len(images)
    z = norm(images)
    x = np.maximum(conv(z, w["grad.W"], w["grad.b"], 1, 0), 0)
    x = np.maximum(conv(x, w["sharp.W"], w["sharp.b"], 1, 0), 0)
    x = np.maximum(conv(x, w["knee.W"], w["knee.b"], 1, 0), 0)
    x = conv(x, w["feat.W"], w["feat.b"], 2, (KB - 2) // 2)
    x = conv(x, w["opp.W"], w["opp.b"], 1, 0)
    e = norm(x.reshape(n, 1, FM * FM, NDIR)).reshape(n, NDIR, FM, FM)
    r = (z.reshape(n, -1) @ w["wall.W"].T + w["wall.b"]).reshape(n, 1, SIZE, SIZE)
    f = np.maximum(conv(r, w["fig1.W"], w["fig1.b"], 1, 0), 0)
    f = np.maximum(conv(f, w["fig2.W"], w["fig2.b"], 1, 0), 0)
    f = norm(conv(f, w["fig3.W"], w["fig3.b"], 2, (KB - 6) // 2))
    return conv(e, w["cat_e.W"], w["cat_e.b"], 1, 0) + conv(f, w["cat_f.W"], w["cat_f.b"], 1, 0)


# ----------------------------------------------------------------------------------------------
# schematic hand: primitives in "palm widths", x to the right, y up, palm centre at the origin.
# The hand is a right hand seen from the front, so the thumb side is on the right of the picture.
# ----------------------------------------------------------------------------------------------
def cap(x0, y0, x1, y1, r0, r1=None):
    return ("cap", x0, y0, x1, y1, r0, r0 if r1 is None else r1)


def ell(cx, cy, rx, ry, ang=0.0):
    return ("ell", cx, cy, rx, ry, ang)


def box(cx, cy, hx, hy, r, ang=0.0):
    return ("box", cx, cy, hx, hy, r, ang)


def chain(pts, r0, r1):
    out = []
    n = len(pts) - 1
    for i in range(n):
        a = r0 + (r1 - r0) * i / n
        b = r0 + (r1 - r0) * (i + 1) / n
        out.append(cap(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], a, b))
    return out


def dirv(a):
    a = math.radians(a)
    return math.sin(a), math.cos(a)


BASE = {"I": (0.36, 0.50), "M": (0.12, 0.56), "R": (-0.12, 0.53), "P": (-0.36, 0.44)}
FLEN = {"I": 0.75, "M": 0.84, "R": 0.78, "P": 0.62}
FRAD = {"I": 0.105, "M": 0.108, "R": 0.103, "P": 0.09}


class Jit:
    """Fixed-seed jitter so that each letter is drawn in many slightly different ways."""
    def __init__(self, rng, amount=1.0):
        self.rng, self.amount = rng, amount

    def n(self, s):
        return float(self.rng.normal()) * s * self.amount

    def u(self, lo, hi):
        return float(self.rng.uniform(lo, hi))

    def s(self, s):
        return math.exp(self.n(s))


class Hand:
    def __init__(self, j):
        self.j = j
        self.sil = []        # silhouette parts (one merged shape)
        self.over = []       # parts painted on top, each with its own dark rim: (prims, tone)
        self.arm = None
        self.lean = 0.0      # clockwise tilt of the whole drawing, degrees
        self.extra_bbox = []
        self.fw = j.s(0.07)  # finger width factor
        self.fl = j.s(0.05)  # finger length factor

    # ---- pieces ----
    def palm(self, hx=0.52, hy=0.58, cy=-0.02, cx=0.0):
        self.sil.append(box(cx, cy, hx, hy, 0.2))
        self.sil.append(ell(0.40, -0.22, 0.24, 0.36, 0.0))      # ball of the thumb

    def forearm(self, x=0.0, y=-0.45, ang=180.0, r=0.43, length=3.5, stub=0.0):
        ang = ang + self.j.n(6)
        r = r * self.j.s(0.06)
        dx, dy = dirv(ang)
        self.arm = cap(x, y, x + dx * length, y + dy * length, r, r * 1.15)
        self.extra_bbox.append((x + dx * stub, y + dy * stub, r))

    def ext(self, f, ang, ln=1.0, over=False, start=0.12, wid=1.0):
        """straight finger from its knuckle, ang degrees clockwise from straight up"""
        bx, by = BASE[f]
        ang = ang + self.j.n(4)
        dx, dy = dirv(ang)
        L = FLEN[f] * ln * self.fl * self.j.s(0.04)
        r = FRAD[f] * self.fw * wid
        c = cap(bx - dx * start, by - dy * start, bx + dx * L, by + dy * L, r, r * 0.88)
        self.sil.append(c)
        if over:
            self.over.append(([c], 1.0))
        return bx + dx * L, by + dy * L

    def curl(self, f, top=0.10, drop=0.40, ang=0.0, dx0=0.0, wid=1.1):
        """finger folded into the palm: seen from the front it is a short lump under its knuckle"""
        bx, by = BASE[f]
        bx += dx0
        r = FRAD[f] * self.fw * wid
        ux, uy = dirv(ang)
        ty = by + top + self.j.n(0.02)
        x0, y0 = bx - ux * 0.0, ty - r
        c = cap(x0, y0, x0 - ux * drop, y0 - uy * drop, r, r)
        self.sil.append(c)
        self.over.append(([c], 1.0))

    def thumb(self, pts, r0=0.13, r1=0.105, over=True, sil=True):
        cs = chain(pts, r0 * self.fw, r1 * self.fw)
        if sil:
            self.sil += cs
        if over:
            self.over.append((cs, 1.0))

    def part(self, prims, over=True, sil=True, tone=1.0):
        if sil:
            self.sil += prims
        if over:
            self.over.append((prims, tone))

    def hole(self, prims):
        self.over.append((prims, "bg"))


# ---- the 24 letters -----------------------------------------------------------------------------
def fist_base(h, tops=(0.10, 0.12, 0.10, 0.06), drop=0.40, ang=0.0):
    h.palm()
    for f, t in zip("PRMI", tops[::-1]):
        h.curl(f, top=t, drop=drop, ang=ang)


def letter_A(h):
    # fist, fingers folded straight down, thumb upright against the side of the index finger
    j = h.j
    fist_base(h, tops=(0.10, 0.13, 0.11, 0.05))
    tx = 0.60 + j.n(0.03)
    h.thumb([(0.50, -0.25), (tx, 0.15), (tx + 0.02 + j.n(0.03), 0.55 + j.n(0.05))], 0.14, 0.105)
    h.forearm()
    h.lean = j.n(7)


def letter_S(h):
    # fist with the thumb wrapped across the front of the fingers
    j = h.j
    a = 14 + j.n(6)
    fist_base(h, tops=(0.16, 0.16, 0.12, 0.06), ang=a)
    h.thumb([(0.50, -0.20), (0.30, 0.02), (-0.05 + j.n(0.05), 0.10 + j.n(0.04))], 0.14, 0.11)
    h.forearm()
    h.lean = 4 + j.n(8)


def letter_E(h):
    # fingers bent so the tips come down to the thumb, which lies across the palm: a tall, even block
    j = h.j
    h.palm()
    for f, t in zip("PRMI", (0.20, 0.30, 0.32, 0.26)):
        h.curl(f, top=t, drop=0.36)
    h.thumb([(0.48, -0.25), (0.25, -0.02 + j.n(0.03)), (-0.22 + j.n(0.05), 0.02 + j.n(0.03))], 0.13, 0.11)
    h.forearm()
    h.lean = j.n(6)


def letter_M(h):
    # thumb tucked under the first three fingers, which drape forward over it
    j = h.j
    a = 10 + j.n(6)
    h.palm()
    h.curl("P", top=0.02, drop=0.36, ang=a)
    for f, t in zip("RMI", (0.16, 0.20, 0.16)):
        h.curl(f, top=t, drop=0.52, ang=a)
    h.thumb([(0.46, -0.30), (0.10, -0.22), (-0.20 + j.n(0.04), -0.10 + j.n(0.04))], 0.13, 0.11)
    h.forearm()
    h.lean = 6 + j.n(8)


def letter_N(h):
    # thumb tucked under the first two fingers; ring and little finger folded lower
    j = h.j
    a = 10 + j.n(6)
    h.palm()
    h.curl("P", top=-0.02, drop=0.34, ang=a)
    h.curl("R", top=0.03, drop=0.36, ang=a)
    for f, t in zip("MI", (0.22, 0.20)):
        h.curl(f, top=t, drop=0.52, ang=a)
    h.thumb([(0.46, -0.30), (0.15, -0.20), (0.00 + j.n(0.04), -0.05 + j.n(0.04))], 0.13, 0.11)
    h.forearm()
    h.lean = 6 + j.n(8)


def letter_B(h):
    # flat hand, four fingers straight up side by side, thumb folded across the palm
    j = h.j
    h.palm()
    for f, a in zip("PRMI", (-3, -1, 0, 2)):
        h.ext(f, a + j.n(1.5), over=True)
    h.thumb([(0.46, -0.30), (0.25, -0.05), (-0.02 + j.n(0.05), 0.12 + j.n(0.05))], 0.13, 0.11, sil=False)
    h.forearm()
    h.lean = j.n(6)


def letter_D(h):
    # index straight up; the other fingers curl round to meet the thumb
    j = h.j
    h.palm(hy=0.56)
    for f, t in zip("PRM", (0.10, 0.18, 0.20)):
        h.curl(f, top=t, drop=0.36, dx0=-0.03)
    bx, by = 0.24 + j.n(0.05), 0.50
    a = 0 + j.n(5)
    dx, dy = dirv(a)
    L = 0.80 * h.fl * j.s(0.05)
    r = 0.105 * h.fw
    h.sil.append(cap(bx - dx * 0.12, by - dy * 0.12, bx + dx * L, by + dy * L, r, r * 0.88))
    h.thumb([(0.46, -0.25), (0.32, 0.05), (0.10 + j.n(0.04), 0.20 + j.n(0.04))], 0.13, 0.11)
    h.forearm()
    h.lean = j.n(7)


def letter_F(h):
    # index and thumb touch in a ring; the other three fingers stand up, spread
    j = h.j
    h.palm()
    h.ext("P", -20 + j.n(5), ln=1.1)
    h.ext("R", -2 + j.n(4), ln=1.08)
    h.ext("M", 20 + j.n(5), ln=1.05)
    mx, my = 0.86 + j.n(0.05), 0.30 + j.n(0.05)
    idx = chain([(0.30, 0.42), (0.62, 0.62), (mx, my)], 0.105 * h.fw, 0.095 * h.fw)
    thb = chain([(0.42, -0.28), (0.70, -0.12), (mx, my)], 0.13 * h.fw, 0.10 * h.fw)
    h.part(idx, over=False)
    h.part(thb, over=False)
    h.forearm()
    h.lean = j.n(7)


def side_block(h, cx=0.0, cy=-0.05, hx=0.40, hy=0.55):
    # hand seen from the thumb side: the back of the hand is on the left, the palm faces right
    h.sil.append(box(cx, cy, hx, hy, 0.2))


def letter_G(h):
    # hand turned sideways, index pointing across, thumb parallel under it, other fingers folded
    j = h.j
    side_block(h)
    y = 0.36 + j.n(0.03)
    a = 90 - 4 + j.n(6)
    dx, dy = dirv(a)
    L = 0.85 * h.fl
    h.part([cap(0.10, y, 0.30 + dx * L, y + dy * L, 0.11 * h.fw, 0.095 * h.fw)], over=True)
    h.thumb([(0.15, 0.08), (0.45, 0.10), (0.72 + j.n(0.06), 0.14 + j.n(0.04))], 0.12, 0.10)
    for i, yy in enumerate((-0.10, -0.30)):
        h.part([cap(0.25, yy, 0.42 - 0.05 * i, yy, 0.115 * h.fw)], over=True)
    h.forearm(x=-0.08, y=-0.40, ang=195 + j.n(8))
    h.lean = j.n(7)


def letter_H(h):
    # as G but index and middle finger together point across
    j = h.j
    side_block(h)
    y = 0.38 + j.n(0.03)
    a = 90 - 4 + j.n(6)
    dx, dy = dirv(a)
    L = 0.88 * h.fl
    w = 0.108 * h.fw
    h.part([cap(0.10, y, 0.30 + dx * L, y + dy * L, w, w * 0.9)], over=True)
    h.part([cap(0.10, y - 2 * w, 0.32 + dx * L * 1.04, y - 2 * w + dy * L, w, w * 0.9)], over=True)
    h.thumb([(0.15, -0.06), (0.40, -0.04), (0.55 + j.n(0.05), -0.02 + j.n(0.04))], 0.12, 0.10)
    h.part([cap(0.25, -0.30, 0.38, -0.30, 0.115 * h.fw)], over=True)
    h.forearm(x=-0.08, y=-0.40, ang=195 + j.n(8))
    h.lean = j.n(7)


def letter_T(h):
    # fist seen from the thumb side: the index is folded forward over the thumb, whose tip stands up
    # between index and middle finger
    j = h.j
    side_block(h, hy=0.52)
    y = 0.30 + j.n(0.03)
    L = (0.62 + j.n(0.06)) * h.fl
    a = 90 - 2 + j.n(6)
    dx, dy = dirv(a)
    h.part([cap(0.10, y, 0.30 + dx * L, y + dy * L, 0.12 * h.fw, 0.10 * h.fw)], over=True)
    tx = 0.12 + j.n(0.05)
    h.thumb([(tx - 0.02, 0.30), (tx + j.n(0.02), 0.50 + 0.36 * h.fl + j.n(0.04))], 0.11, 0.10)
    for i, yy in enumerate((0.04, -0.18, -0.38)):
        h.part([cap(0.22, yy, 0.42 - 0.04 * i, yy, 0.115 * h.fw)], over=True)
    h.forearm(x=-0.02, y=-0.40, ang=183 + j.n(6))
    h.lean = j.n(7)


def letter_X(h):
    # fist seen from the thumb side with the index raised and bent into a hook
    j = h.j
    side_block(h, hy=0.52)
    x0 = 0.08 + j.n(0.04)
    top = 0.50 + 0.36 * h.fl + j.n(0.04)
    ex = x0 + 0.42 * h.fl + j.n(0.04)
    pts = [(x0, 0.30), (x0 + 0.02, top), (ex, top + 0.04 + j.n(0.04)), (ex + 0.08, top - 0.16 + j.n(0.03))]
    h.part(chain(pts, 0.115 * h.fw, 0.095 * h.fw), over=False)
    h.thumb([(0.15, 0.00), (0.40, 0.14 + j.n(0.03))], 0.13, 0.11)
    for i, yy in enumerate((0.20, -0.02, -0.24)):
        h.part([cap(0.22, yy, 0.44 - 0.04 * i, yy, 0.115 * h.fw)], over=True)
    h.forearm(x=-0.02, y=-0.40, ang=186 + j.n(6))
    h.lean = j.n(7)


def letter_C(h):
    # curved hand seen from the thumb side: fingers arch over the top, thumb below, open to the right
    j = h.j
    h.sil += [cap(-0.30, -0.50, -0.20, 0.28, 0.32, 0.30)]
    tipx, tipy = 0.90 + j.n(0.05), 0.44 + j.n(0.05)
    h.part(chain([(-0.15, 0.36), (0.20, 0.72 + j.n(0.03)), (0.62, 0.70), (tipx, tipy)], 0.21 * h.fw, 0.12 * h.fw), over=False)
    tx, ty = 0.90 + j.n(0.06), -0.12 + j.n(0.06)
    h.part(chain([(-0.10, -0.42), (0.45, -0.40), (tx, ty)], 0.19 * h.fw, 0.11 * h.fw), over=False)
    h.forearm(x=-0.28, y=-0.45, ang=198 + j.n(8), r=0.37)
    h.lean = j.n(7)


def letter_O(h):
    # as C but the fingertips come round to meet the thumb, leaving a hole
    j = h.j
    h.sil += [cap(-0.25, -0.50, -0.15, 0.25, 0.33, 0.31)]
    mx, my = 0.66 + j.n(0.04), 0.12 + j.n(0.04)
    h.part(chain([(-0.10, 0.34), (0.20, 0.62 + j.n(0.03)), (0.55, 0.50), (mx, my)], 0.22 * h.fw, 0.14 * h.fw), over=False)
    h.part(chain([(-0.10, -0.38), (0.40, -0.30), (mx, my)], 0.19 * h.fw, 0.13 * h.fw), over=False)
    h.forearm(x=-0.24, y=-0.45, ang=194 + j.n(8), r=0.37)
    h.lean = j.n(7)


def letter_I(h):
    # fist with the little finger straight up
    j = h.j
    h.palm()
    for f, t in zip("RMI", (0.10, 0.12, 0.08)):
        h.curl(f, top=t)
    h.ext("P", -6 + j.n(4), ln=1.05)
    h.thumb([(0.50, -0.20), (0.30, 0.02), (0.00 + j.n(0.05), 0.10 + j.n(0.04))], 0.13, 0.11)
    h.forearm()
    h.lean = j.n(7)


def letter_K(h):
    # index straight up, middle finger angled forward, thumb between them, other two folded
    j = h.j
    h.palm()
    for f, t in zip("PR", (0.06, 0.10)):
        h.curl(f, top=t)
    h.ext("I", 3 + j.n(3))
    bx, by = BASE["M"]
    a = -38 + j.n(7)
    dx, dy = dirv(a)
    L = 0.55 * h.fl
    h.part([cap(bx, by - 0.1, bx + dx * L, by + dy * L, 0.11 * h.fw, 0.10 * h.fw)], over=True)
    h.thumb([(0.46, -0.25), (0.34, 0.15), (0.16 + j.n(0.04), 0.50 + j.n(0.04))], 0.13, 0.10)
    h.forearm()
    h.lean = j.n(7)


def letter_L(h):
    # index straight up, thumb straight out to the side, other fingers folded
    j = h.j
    h.palm()
    for f, t in zip("PRM", (0.06, 0.10, 0.10)):
        h.curl(f, top=t)
    h.ext("I", 0 + j.n(3))
    a = 72 + j.n(8)
    dx, dy = dirv(a)
    L = 0.85 * h.fl
    h.thumb([(0.40, -0.30), (0.52, -0.18), (0.52 + dx * L, -0.18 + dy * L)], 0.15, 0.10, over=False)
    h.forearm()
    h.lean = j.n(7)


def letter_P(h):
    # the K hand tipped over: index pointing across to the left, middle finger hanging down,
    # thumb between; the wrist is on the right and the arm leaves the picture there
    j = h.j
    h.sil.append(ell(0.25, -0.05, 0.55, 0.50, 0.0))
    a = -75 + j.n(9)
    dx, dy = dirv(a)
    L = 0.75 * h.fl
    y = 0.28 + j.n(0.03)
    h.part([cap(0.05, y - 0.05, -0.10 + dx * L, y + dy * L, 0.12 * h.fw, 0.095 * h.fw)], over=True)
    a2 = -165 + j.n(9)
    dx, dy = dirv(a2)
    L2 = 0.50 * h.fl
    h.part([cap(0.00, -0.10, -0.05 + dx * L2, -0.22 + dy * L2, 0.115 * h.fw, 0.10 * h.fw)], over=True)
    h.thumb([(0.30, -0.25), (0.05, -0.10), (-0.22 + j.n(0.04), 0.02 + j.n(0.04))], 0.12, 0.10)
    h.part([cap(0.42, -0.32, 0.22, -0.38, 0.115 * h.fw)], over=True)
    h.forearm(x=0.48, y=0.00, ang=132 + j.n(10), r=0.42, stub=0.0)
    h.lean = j.n(8)


def letter_Q(h):
    # the G hand turned to point down: the back of the hand is a broad dome on top, index and thumb
    # hang down from it, the wrist is on the right and the arm leaves the picture there
    j = h.j
    h.sil.append(ell(0.05, 0.25, 0.62, 0.58, 0.0))
    a = -150 + j.n(10)
    dx, dy = dirv(a)
    L = 0.70 * h.fl
    h.part([cap(-0.30, 0.05, -0.40 + dx * L, -0.15 + dy * L, 0.125 * h.fw, 0.10 * h.fw)], over=True)
    a2 = -178 + j.n(8)
    dx, dy = dirv(a2)
    L2 = 0.42 * h.fl
    h.thumb([(0.12, 0.00), (0.12 + dx * L2, -0.25 + dy * L2)], 0.15, 0.13)
    for i, xx in enumerate((-0.42, -0.20, 0.02)):          # folded fingers seen on the back of the hand
        h.part([cap(xx + 0.06, 0.52 - 0.02 * i, xx - 0.04, 0.26, 0.10 * h.fw)], over=True)
    h.forearm(x=0.42, y=0.22, ang=125 + j.n(10), r=0.42, stub=0.0)
    h.lean = j.n(8)


def letter_R(h):
    # index and middle finger crossed, pointing up
    j = h.j
    h.palm()
    for f, t in zip("PR", (0.06, 0.10)):
        h.curl(f, top=t)
    bx, by = 0.24, 0.52
    a = 2 + j.n(3)
    dx, dy = dirv(a)
    L = 0.88 * h.fl
    w = 0.105 * h.fw
    c1 = cap(bx + 0.07, by - 0.1, bx - 0.03 + dx * L, by + dy * L, w, w * 0.9)
    c2 = cap(bx - 0.08, by - 0.1, bx + 0.04 + dx * L * 0.93, by + dy * L * 0.93, w, w * 0.9)
    h.part([c1], over=True)
    h.part([c2], over=True)
    h.thumb([(0.50, -0.20), (0.30, 0.02), (0.00 + j.n(0.05), 0.12 + j.n(0.04))], 0.13, 0.11)
    h.forearm()
    h.lean = j.n(7)


def letter_U(h):
    # index and middle finger straight up, held together
    j = h.j
    h.palm()
    for f, t in zip("PR", (0.06, 0.10)):
        h.curl(f, top=t)
    a = 1 + j.n(3)
    h.ext("I", a + 1.5, over=True)
    h.ext("M", a - 1.5, over=True)
    h.thumb([(0.50, -0.20), (0.30, 0.02), (0.00 + j.n(0.05), 0.12 + j.n(0.04))], 0.13, 0.11)
    h.forearm()
    h.lean = j.n(7)


def letter_V(h):
    # index and middle finger up and spread apart
    j = h.j
    h.palm()
    for f, t in zip("PR", (0.06, 0.10)):
        h.curl(f, top=t)
    s = 14 + j.n(5)
    h.ext("I", s + 2)
    h.ext("M", -s + 2)
    h.thumb([(0.50, -0.20), (0.30, 0.02), (0.00 + j.n(0.05), 0.12 + j.n(0.04))], 0.13, 0.11)
    h.forearm()
    h.lean = j.n(7)


def letter_W(h):
    # index, middle and ring finger up and spread; thumb holds the little finger down
    j = h.j
    h.palm()
    h.curl("P", top=0.02)
    s = 16 + j.n(5)
    h.ext("I", s + 1)
    h.ext("M", 0 + j.n(2))
    h.ext("R", -s)
    h.thumb([(0.48, -0.25), (0.22, -0.05), (-0.20 + j.n(0.05), 0.10 + j.n(0.04))], 0.13, 0.11)
    h.forearm()
    h.lean = j.n(7)


def letter_Y(h):
    # thumb and little finger stretched out to either side, middle three fingers folded
    j = h.j
    h.palm()
    for f, t in zip("RMI", (0.10, 0.12, 0.08)):
        h.curl(f, top=t)
    h.ext("P", -22 + j.n(6), ln=1.05)
    a = 52 + j.n(8)
    dx, dy = dirv(a)
    L = 0.85 * h.fl
    h.thumb([(0.40, -0.30), (0.50, -0.12), (0.50 + dx * L, -0.12 + dy * L)], 0.15, 0.10, over=False)
    h.forearm()
    h.lean = j.n(7)


LETTERS = {"A": letter_A, "B": letter_B, "C": letter_C, "D": letter_D, "E": letter_E, "F": letter_F,
           "G": letter_G, "H": letter_H, "I": letter_I, "K": letter_K, "L": letter_L, "M": letter_M,
           "N": letter_N, "O": letter_O, "P": letter_P, "Q": letter_Q, "R": letter_R, "S": letter_S,
           "T": letter_T, "U": letter_U, "V": letter_V, "W": letter_W, "X": letter_X, "Y": letter_Y}


# ----------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------
def sdf(p, X, Y):
    k = p[0]
    if k == "cap":
        _, x0, y0, x1, y1, r0, r1 = p
        vx, vy = x1 - x0, y1 - y0
        L2 = vx * vx + vy * vy + 1e-12
        t = np.clip(((X - x0) * vx + (Y - y0) * vy) / L2, 0, 1)
        return np.hypot(X - x0 - t * vx, Y - y0 - t * vy) - (r0 + t * (r1 - r0))
    if k == "ell":
        _, cx, cy, rx, ry, ang = p
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        u = (X - cx) * c + (Y - cy) * s
        v = -(X - cx) * s + (Y - cy) * c
        return (np.sqrt((u / rx) ** 2 + (v / ry) ** 2) - 1.0) * min(rx, ry)
    if k == "box":
        _, cx, cy, hx, hy, r, ang = p
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        u = np.abs((X - cx) * c + (Y - cy) * s) - (hx - r)
        v = np.abs(-(X - cx) * s + (Y - cy) * c) - (hy - r)
        return np.hypot(np.maximum(u, 0), np.maximum(v, 0)) + np.minimum(np.maximum(u, v), 0) - r
    raise ValueError(k)


def extent(p):
    k = p[0]
    if k == "cap":
        _, x0, y0, x1, y1, r0, r1 = p
        return [(x0, y0, r0), (x1, y1, r1)]
    if k == "ell":
        _, cx, cy, rx, ry, ang = p
        return [(cx - rx, cy, 0), (cx + rx, cy, 0), (cx, cy - ry, 0), (cx, cy + ry, 0)]
    _, cx, cy, hx, hy, r, ang = p
    return [(cx - hx + r, cy - hy + r, r), (cx + hx - r, cy - hy + r, r), (cx - hx + r, cy + hy - r, r), (cx + hx - r, cy + hy - r, r)]


def draw(letter, rng, amount=1.0):
    """One drawing of a letter: (28,28) grey image in 0..1, and the signed distance to its outline in pixels (inside < 0)."""
    j = Jit(rng, amount)
    h = Hand(j)
    LETTERS[letter](h)
    lean = math.radians(h.lean)
    cl, sl = math.cos(lean), math.sin(lean)
    asp = j.s(0.05)

    def fwd(x, y):                       # hand coordinates -> tilted, stretched coordinates (y still up)
        return (x * cl + y * sl) * asp, (-x * sl + y * cl)

    pts = []
    for p in h.sil + [q for o in h.over if o[1] != "bg" for q in o[0]]:
        pts += extent(p)
    pts += h.extra_bbox
    xs0, xs1, ys0, ys1 = [], [], [], []
    for (x, y, r) in pts:
        u, v = fwd(x, y)
        xs0.append(u - r); xs1.append(u + r); ys0.append(v - r); ys1.append(v + r)
    bx0, bx1, by0, by1 = min(xs0), max(xs1), min(ys0), max(ys1)
    fill = j.u(0.86, 1.06)
    scale = fill * SIZE / max(bx1 - bx0, by1 - by0)          # pixels per palm width
    ox = SIZE / 2 + j.u(-1.5, 1.5)
    oy = SIZE / 2 + j.u(-1.5, 1.5)
    mx, my = (bx0 + bx1) / 2, (by0 + by1) / 2
    n = SIZE * SS
    gy, gx = np.mgrid[0:n, 0:n].astype(np.float64)
    px = (gx + 0.5) / SS
    py = (gy + 0.5) / SS
    u = (px - ox) / scale + mx           # tilted coordinates of every sample
    v = -(py - oy) / scale + my
    u = u / asp
    X = u * cl - v * sl                  # back to hand coordinates
    Y = u * sl + v * cl

    style = int(rng.randint(0, 2))
    if style == 0:                       # hand in shade against a light wall
        bg, core, rim = j.u(0.66, 0.80), j.u(0.52, 0.66), j.u(0.45, 0.60)
    else:                                # lit hand against a darker wall
        bg, core, rim = j.u(0.35, 0.50), j.u(0.70, 0.85), j.u(0.30, 0.45)
    rw = 0.055                           # rim width in palm widths

    def shade(d):
        return core * (1.0 - rim * np.exp(np.minimum(d, 0) / rw))

    img = np.full((n, n), bg)
    dsil = sdf(h.arm, X, Y) if h.arm is not None else np.full((n, n), 9.0)
    for p in h.sil:
        dsil = np.minimum(dsil, sdf(p, X, Y))
    img = np.where(dsil < 0, shade(dsil), img)
    dark = core * j.u(0.40, 0.60)        # creases and contact shadows between touching parts
    hw = 0.05                            # their width in palm widths
    for prims, tone in h.over:
        d = np.full((n, n), 9.0)
        for p in prims:
            d = np.minimum(d, sdf(p, X, Y))
        if tone == "bg":
            img = np.where(d < 0, bg, img)
        elif tone == "shadow":
            img = np.where((d < 0) & (dsil < 0), dark, img)
        else:
            img = np.where((d < hw) & (dsil < 0), dark, img)
            img = np.where(d < 0, shade(d) * tone, img)
    return img.reshape(SIZE, SS, SIZE, SS).mean(axis=(1, 3)), (dsil * scale).reshape(SIZE, SS, SIZE, SS).mean(axis=(1, 3))


def soften(imgs):
    """Slight blur, as a photograph shrunk to 28 x 28 has: 3x3 binomial."""
    k = np.array([1.0, 2.0, 1.0]) / 4.0
    p = np.pad(imgs, ((0, 0), (1, 1), (1, 1)), mode="edge")
    t = k[0] * p[:, :-2, :] + k[1] * p[:, 1:-1, :] + k[2] * p[:, 2:, :]
    return k[0] * t[:, :, :-2] + k[1] * t[:, :, 1:-1] + k[2] * t[:, :, 2:]


def drawings(letter, count):
    rng = np.random.RandomState(1000 + NAMES.index(letter))
    pairs = [draw(letter, rng) for _ in range(count)]
    return soften(np.stack([p[0] for p in pairs])), np.stack([p[1] for p in pairs])


# ----------------------------------------------------------------------------------------------
# templates and network
# ----------------------------------------------------------------------------------------------
def window():
    """Spatial weight on the feature map: the lower rows (wrist, sleeve, table) count less."""
    rows = np.ones(FM)
    rows[-3:] = [0.75, 0.55, 0.40]
    cols = np.ones(FM)
    cols[0] = cols[-1] = 0.7
    return rows[:, None] * cols[None, :]


def ramp(d, d0, v0, d1, v1):
    return v0 + (v1 - v0) * np.clip((d - d0) / (d1 - d0), 0, 1)


def regions(dist):
    """dist (N,28,28): signed distance to the outline in pixels -> weights on the feature grid for the two maps."""
    d = dist[:, 2:2 + 2 * FM, 2:2 + 2 * FM].reshape(len(dist), FM, 2, FM, 2).mean(axis=(2, 4))
    # edge map: full weight from 2 px inside the outline to 5 px outside it
    re = np.minimum(ramp(d, -4.5, R_IN, -2.0, 1.0), ramp(d, 5.0, 1.0, 9.0, R_OUT))
    # figure map: full weight within 5 px of the outline on either side
    rf = np.minimum(ramp(d, -8.0, R_IN, -5.0, 1.0), ramp(d, 5.0, 1.0, 9.0, R_OUT))
    return re[:, None], rf[:, None]


def templates(fw):
    win = window()[None, None]
    out = []
    for letter in NAMES:
        imgs, dist = drawings(letter, NVAR)
        f = front(imgs[:, None], fw)                                   # (NVAR, NDIR+1, FM, FM)
        re, rf = regions(dist)
        parts = []
        for part, reg, wt in ((f[:, :NDIR], re, W_EDGE), (f[:, NDIR:], rf, W_FIG)):
            wgt = reg * win                                            # (NVAR, 1, FM, FM)
            cnt = wgt.sum(axis=(1, 2, 3), keepdims=True) * part.shape[1]
            mu = (part * wgt).sum(axis=(1, 2, 3), keepdims=True) / cnt
            t = (part - mu) * wgt
            t /= np.sqrt(((part - mu) ** 2 * wgt).sum(axis=(1, 2, 3), keepdims=True))   # each drawing by itself
            parts.append(t * wt / np.sqrt(cnt))        # the picture's map has unit variance, so this is about wt * correlation
        out.append(np.concatenate(parts, axis=1))
    return np.concatenate(out)                                          # (24*NVAR, NDIR+1, FM, FM)


def network():
    npos = 2 * PADT + 1
    L = []

    def add(name, op, inputs, **kw):
        L.append(dict(name=name, op=op, inputs=inputs, **kw))

    def c(name, inp, out, k, stride=1, padding=0):
        add(name, "conv", [inp], out=out, kernel=[k, k], stride=stride, padding=padding)

    add("z", "norm", ["image"])
    # edge map
    c("grad", "z", NDIR, KG); add("grad_r", "relu", ["grad"])
    c("sharp", "grad_r", NDIR, 1); add("sharp_r", "relu", ["sharp"])
    c("knee", "sharp_r", 3 * NDIR, 1); add("knee_r", "relu", ["knee"])
    c("feat", "knee_r", NDIR, KB, 2, (KB - 2) // 2)
    c("opp", "feat", NDIR, 1)
    add("opp_flat", "reshape", ["opp"], shape=[1, FM * FM, NDIR])
    add("opp_norm", "norm", ["opp_flat"])
    add("edge", "reshape", ["opp_norm"], shape=[NDIR, FM, FM])
    # figure map
    add("wall", "dense", ["z"], out=SIZE * SIZE)
    add("resid", "reshape", ["wall"], shape=[1, SIZE, SIZE])
    c("fig1", "resid", 2, 1); add("fig1_r", "relu", ["fig1"])
    c("fig2", "fig1_r", 2, 1); add("fig2_r", "relu", ["fig2"])
    c("fig3", "fig2_r", 1, KB, 2, (KB - 6) // 2)
    add("fig", "norm", ["fig3"])
    # both maps as 13 channels, template matching over shifts, best drawing per letter
    c("cat_e", "edge", NDIR + 1, 1)
    c("cat_f", "fig", NDIR + 1, 1)
    add("maps", "add", ["cat_e", "cat_f"])
    c("match", "maps", 24 * NVAR, FM, 1, PADT)
    add("best_shift", "maxpool", ["match"], size=[npos, npos], stride=[npos, npos])
    add("by_class", "reshape", ["best_shift"], shape=[1, 24, NVAR])
    add("best_var", "maxpool", ["by_class"], size=[1, NVAR], stride=[1, NVAR])
    add("scores", "reshape", ["best_var"], shape=[24])
    return {"layers": L}


def build():
    w = front_weights()
    w["match.W"] = templates(w)
    w["match.b"] = np.zeros(len(w["match.W"]))
    return network(), {k: v.astype(np.float32) for k, v in w.items()}


def main(out_dir):
    net, w = build()
    with open(out_dir + "/network.json", "w") as f:
        json.dump(net, f, indent=1)
    np.savez(out_dir + "/weights.npz", **w)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
