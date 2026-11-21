#!/usr/bin/env python3
"""build.py: writes network.json and weights.npz for a hand-designed ASL fingerspelling classifier.

No training data is read. Every number comes from this file:
  * fixed image filters from formulas (Gaussian blur, Gaussian derivatives, bar / gap / hole detectors)
  * class templates rendered from a parametric hand model (palm / fist, fingers, thumb, forearm)
    typed from general knowledge of the ASL manual alphabet, passed through the same filters.
Usage: python3 build.py OUT_DIR
"""
import sys, json, math, os
import numpy as np

N = 28            # image side
SS = 4            # supersampling for rendering
EPS = 1e-5

# ----------------------------------------------------------------------------------------------
# shape primitives: signed distance functions on a (SS*N, SS*N) grid in pixel coordinates
# ----------------------------------------------------------------------------------------------
_g = (np.arange(N * SS) + 0.5) / SS
GX, GY = np.meshgrid(_g, _g)


def capsule(p0, p1, w):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = ((GX - x0) * dx + (GY - y0) * dy) / (L2 if L2 > 1e-9 else 1e-9)
    t = np.clip(t, 0, 1)
    px, py = x0 + t * dx, y0 + t * dy
    return np.hypot(GX - px, GY - py) - w / 2


def ellipse(c, rx, ry, ang=0.0):
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    x, y = GX - c[0], GY - c[1]
    xr, yr = ca * x + sa * y, -sa * x + ca * y
    return (np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2) - 1.0) * min(rx, ry)


def roundrect(c, hx, hy, r, ang=0.0):
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    x, y = GX - c[0], GY - c[1]
    xr, yr = ca * x + sa * y, -sa * x + ca * y
    qx, qy = np.abs(xr) - (hx - r), np.abs(yr) - (hy - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def arc(c, R, a0, a1, w, n=12):
    """thick arc (polyline of capsules) around centre c, radius R, from angle a0 to a1 (degrees, y down)."""
    ds = []
    angs = np.linspace(a0, a1, n + 1)
    pts = [(c[0] + R * math.cos(math.radians(a)), c[1] + R * math.sin(math.radians(a))) for a in angs]
    for i in range(n):
        ds.append(capsule(pts[i], pts[i + 1], w))
    return np.minimum.reduce(ds)


DIM = 0.9       # brightness of the forearm / wrist relative to the hand
LINE = 0.6      # brightness of the crease between touching fingers
TEXTURE = 0.0   # amplitude of the smooth texture inside the hand (skin shading / creases; the wall is smooth)


def _texture():
    rng = np.random.RandomState(12345)
    noise = rng.standard_normal((N, N))
    g = np.exp(-(np.arange(-3, 4) ** 2) / (2 * 1.2 ** 2)); g /= g.sum()
    sm = np.apply_along_axis(lambda r: np.convolve(r, g, 'same'), 0, noise)
    sm = np.apply_along_axis(lambda r: np.convolve(r, g, 'same'), 1, sm)
    sm = sm / sm.std()
    return np.kron(sm, np.ones((SS, SS)))      # back to the supersampled grid


TEX = _texture()


def _down(a):
    return a.reshape(N, SS, N, SS).mean(axis=(1, 3))


def render(shapes, holes=(), lines=(), dim=()):
    """shapes/holes/lines/dim: lists of distance fields.  Returns (image, hand mask, forearm mask), each (N, N)
    in [0, 1] (hand = 1 in the image)."""
    d = np.minimum.reduce(shapes)
    hand = (d < 0).astype(np.float64)
    if holes:
        hand = hand * (np.minimum.reduce(holes) >= 0)
    img = hand.copy()
    arm = np.zeros_like(hand)
    if dim:
        arm = (np.minimum.reduce(dim) < 0) * (1.0 - hand)
        img = np.maximum(img, DIM * arm)
    if lines:
        img = img * np.where(np.minimum.reduce(lines) < 0, LINE, 1.0)
    img = img * (1.0 + TEXTURE * TEX)
    return _down(img), _down(hand), _down(arm)


# ----------------------------------------------------------------------------------------------
# affine helper for variants: points are transformed, widths scaled
# ----------------------------------------------------------------------------------------------
class T:
    def __init__(self, scale=1.0, rot=0.0, centre=(14.0, 15.0), shift=(0.0, 0.0)):
        self.s, self.r, self.c, self.sh = scale, rot, centre, shift

    def p(self, pt):
        x, y = pt[0] - self.c[0], pt[1] - self.c[1]
        ca, sa = math.cos(math.radians(self.r)), math.sin(math.radians(self.r))
        xr, yr = ca * x - sa * y, sa * x + ca * y
        return (self.c[0] + self.s * xr + self.sh[0], self.c[1] + self.s * yr + self.sh[1])

    def w(self, v):
        return self.s * v

    def cap(self, p0, p1, w):
        return capsule(self.p(p0), self.p(p1), self.w(w))

    def ell(self, c, rx, ry, ang=0.0):
        return ellipse(self.p(c), self.w(rx), self.w(ry), ang + self.r)

    def rr(self, c, hx, hy, r, ang=0.0):
        return roundrect(self.p(c), self.w(hx), self.w(hy), self.w(r), ang + self.r)

    def arc(self, c, R, a0, a1, w):
        return arc(self.p(c), self.w(R), a0 + self.r, a1 + self.r, self.w(w))


# ----------------------------------------------------------------------------------------------
# hand poses for the 24 letters (viewer's coordinates: x right, y down; right hand, palm to viewer,
# so the thumb is on the viewer's right and the little finger on the viewer's left).
# Proportions: fist ~15 px wide, finger ~3.7 px wide, finger ~11 px long, hand fills the frame.
# ----------------------------------------------------------------------------------------------
FW = 3.7      # finger width
FOREARM = 11.0
LW = 0.9      # crease line width
FTOP = 1.5    # y of the fingertips of a straight extended finger
FBASE = 13.0  # y where the fingers leave the fist


class Pose:
    def __init__(self):
        self.S, self.H, self.L, self.D = [], [], [], []


def forearm(P, t, top, bottom=None, w=FOREARM):
    if bottom is None:
        bottom = (top[0], top[1] + 14)
    P.D.append(t.cap(top, bottom, w))


def base_upright(P, t):
    """shared fist / palm + forearm of every upright letter, so that templates differ only in the fingers"""
    P.S.append(t.rr((14.0, 17.5), 7.5, 6.5, 3.5))
    forearm(P, t, (14.0, 23.0))


def base_side(P, t):
    """shared fist + forearm of the sideways letters G and H (fist on the left, forearm from the lower left)"""
    P.S.append(t.rr((8.5, 15.0), 7.0, 6.0, 3.5))
    forearm(P, t, (3.0, 17.0), (-8.0, 21.0))


def base_down(P, t):
    """shared palm + forearm of the hanging letters P and Q (palm at the top, forearm from the upper left)"""
    P.S.append(t.ell((12.5, 8.5), 7.5, 5.5))
    forearm(P, t, (6.5, 6.0), (-5.0, 0.0))


def finger(t, x0, x1, w=FW, y0=FBASE, y1=FTOP):
    return t.cap((x0, y0), (x1, y1), w)


def pose(k, v, t):
    """Returns a Pose (shapes S, holes H, crease lines L, dim parts D) for class k, variant v, transform t."""
    P = Pose()
    S, H, L = P.S, P.H, P.L
    d = [0.0, 1.0, -1.0][v]          # generic small pose variation
    sp = [1.0, 1.3, 0.7][v]          # finger spread factor
    if k == 0:    # A: fist, thumb up alongside on the right
        base_upright(P, t)
        S.append(t.cap((20.0, 18.0), (20.5, 10.0 - d), FW + 0.3))
        L.append(t.cap((18.4, 19.0), (18.8, 11.5), LW))
    elif k == 1:  # B: flat hand, four fingers up together (a block), thumb across the palm
        base_upright(P, t)
        xs = [8.9, 12.3, 15.7, 19.1]
        tops = [5.0, 2.0, 1.2, 2.5]
        for i, (x, yt) in enumerate(zip(xs, tops)):
            S.append(t.cap((x, FBASE), (x + 0.3 * d * (i - 1.5), yt), FW - 0.3))
        for i in range(3):
            xm = (xs[i] + xs[i + 1]) / 2
            L.append(t.cap((xm, 12.0), (xm + 0.3 * d * (i - 1.0), max(tops[i], tops[i + 1]) + 0.5), LW))
    elif k == 2:  # C: curved hand, opening to the right
        R = [9.0, 10.0, 8.3][v]
        a = [0, 10, -10][v]
        S.append(t.arc((13.5, 13.0), R, 55 + a, 305 - a, 6.5))
        forearm(P, t, (9.0, 21.0), (3.0, 30.0))
    elif k == 3:  # D: index up, other fingers curled to the thumb (small loop)
        base_upright(P, t)
        S.append(finger(t, 16.0, 16.5 + d))
        H.append(t.ell((11.5, 17.0), 1.5, 1.8))
    elif k == 4:  # E: fist, fingers curled with the tips on the palm: bumpy flat top
        base_upright(P, t)
        for i, x in enumerate([9.0, 12.3, 15.7, 19.0]):
            S.append(t.ell((x + 0.3 * d * (i - 1.5), 11.5), 1.9, 2.4))
        for x in [10.6, 14.0, 17.4]:
            L.append(t.cap((x, 9.5), (x, 13.5), LW))
    elif k == 5:  # F: thumb+index circle at the lower right, three fingers fanned up-left
        base_upright(P, t)
        H.append(t.ell((17.0, 17.5), 2.0, 2.0))
        S.append(t.cap((11.0, 14.0), (11.0 - 6.0 * sp, 4.0), FW))
        S.append(t.cap((13.0, 13.0), (12.0 - 2.0 * sp, 1.2), FW))
        S.append(t.cap((15.5, 12.5), (16.5 + 1.5 * sp, 1.5), FW))
    elif k == 6:  # G: hand sideways, index pointing right, thumb above it
        base_side(P, t)
        S.append(t.cap((13.0, 14.5), (27.0, 13.0 + 1.5 * d), FW))
        S.append(t.cap((10.0, 10.0), (19.0, 8.5 + 1.5 * d), FW))
    elif k == 7:  # H: sideways, index+middle together pointing right (thick bar with a crease)
        base_side(P, t)
        S.append(t.cap((13.0, 14.0), (27.0, 12.5 + 2.0 * d), 2 * FW - 0.8))
        L.append(t.cap((14.0, 14.0), (27.0, 12.5 + 2.0 * d), LW))
        S.append(t.cap((9.0, 9.5), (14.0, 9.5), FW - 0.5))
    elif k == 8:  # I: fist, little finger up on the left
        base_upright(P, t)
        S.append(finger(t, 9.0, 8.5 - d, FW - 0.5, y1=3.5))
    elif k == 9:  # K: index up, middle angled up-left, thumb between
        base_upright(P, t)
        S.append(finger(t, 16.5, 17.0))
        S.append(t.cap((13.0, FBASE), (12.5 - 6.5 * sp, 4.0), FW))
        S.append(t.cap((15.0, FBASE), (13.5 - 2.0 * sp, 7.0), FW - 0.4))
    elif k == 10:  # L: index up (left of centre), thumb out to the right
        base_upright(P, t)
        S.append(finger(t, 10.5, 10.5))
        ang = [10.0, 25.0, 0.0][v]
        Lt = 11.0
        S.append(t.cap((17.0, 16.0), (17.0 + Lt * math.cos(math.radians(ang)), 16.0 - Lt * math.sin(math.radians(ang))), FW))
    elif k == 11:  # M: fist, three fingers folded down over the thumb (knuckle ridge on top, creases in front)
        base_upright(P, t)
        S.append(t.rr((13.0, 11.5), 6.0, 2.2, 1.5, [-8, 0, -15][v]))
        for x in [10.5, 14.0, 17.5]:
            L.append(t.cap((x, 15.0), (x + 0.5, 22.0), LW))
    elif k == 12:  # N: fist, two fingers folded down over the thumb
        base_upright(P, t)
        S.append(t.rr((13.5, 11.5), 5.0, 2.2, 1.5, [-8, 0, -15][v]))
        for x in [11.5, 15.5]:
            L.append(t.cap((x, 15.0), (x + 0.5, 22.0), LW))
    elif k == 13:  # O: fingers curled to meet the thumb, round hand with a hole
        S.append(t.ell((14.0, 14.5), 8.0 + 0.6 * d, 9.5))
        H.append(t.ell((11.0, 14.0), 2.3, 3.0))
        forearm(P, t, (13.0, 22.0), (10.0, 32.0))
    elif k == 14:  # P: like K pointing down: palm at top, index right, middle down-right
        base_down(P, t)
        S.append(t.cap((15.0, 9.5), (27.0, 11.0 + 2.0 * d), FW))
        S.append(t.cap((13.0, 12.0), (15.0, 23.0), FW))
        S.append(t.cap((9.0, 12.0), (8.0, 17.0), FW - 0.4))
    elif k == 15:  # Q: like G pointing down: palm at top, index and thumb pointing down
        base_down(P, t)
        S.append(t.cap((16.0, 11.0), (17.0 + d, 23.0), FW))
        S.append(t.cap((9.0, 11.0), (8.0 - d, 21.0), FW))
    elif k == 16:  # R: index and middle crossed, pointing up (one thick finger with a crease)
        base_upright(P, t)
        S.append(finger(t, 14.5, 15.0 + d, FW + 1.4))
        L.append(t.cap((13.5, 12.0), (16.0 + d, 2.0), LW))
    elif k == 17:  # S: fist, thumb across the front (horizontal crease)
        base_upright(P, t)
        L.append(t.cap((8.0, 17.5 + d), (21.0, 16.5 + d), LW))
    elif k == 18:  # T: fist, thumb tip between index and middle (bump at top right)
        base_upright(P, t)
        S.append(t.cap((17.0, 15.0), (17.5, 8.5 - d), FW))
    elif k == 19:  # U: index and middle together, pointing up (crease between them)
        base_upright(P, t)
        S.append(finger(t, 14.5, 15.0 + d, 2 * FW - 0.6))
        L.append(t.cap((14.5, 12.0), (15.0 + d, 2.0), LW))
    elif k == 20:  # V: index and middle spread, pointing up
        base_upright(P, t)
        S.append(finger(t, 12.5, 12.5 - 3.5 * sp))
        S.append(finger(t, 16.5, 16.5 + 3.5 * sp))
    elif k == 21:  # W: index, middle, ring spread, pointing up
        base_upright(P, t)
        S.append(finger(t, 10.5, 10.5 - 4.5 * sp, y1=2.5))
        S.append(finger(t, 14.5, 14.5, y1=1.0))
        S.append(finger(t, 18.5, 18.5 + 4.5 * sp, y1=2.5))
    elif k == 22:  # X: index bent into a hook, pointing up
        base_upright(P, t)
        S.append(t.cap((16.5, FBASE), (17.0, 5.5), FW))
        S.append(t.cap((17.0, 5.5), (13.5 - d, 4.0), FW - 0.4))
    elif k == 23:  # Y: thumb out to the upper right, little finger up on the left
        base_upright(P, t)
        S.append(finger(t, 8.5, 6.0 - d, FW - 0.4, y1=3.5))
        S.append(t.cap((19.5, 15.5), (26.5, 8.5 + d), FW))
    return P




TARGET = 22.0     # the photos are cropped to a square around the hand: the hand (without the forearm) spans
                  # most of the frame, so each pose is scaled so its larger extent is TARGET px and centred
_FIT = {}


def fit_params(k, v):
    """scale and bounding-box centre that put the hand (without the forearm) of pose (k, v) into the frame."""
    if (k, v) not in _FIT:
        P = pose(k, v, T())
        _, hand, _ = render(P.S, P.H, P.L, P.D)
        ys, xs = np.where(hand > 0.5)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        s = TARGET / max(x1 - x0, y1 - y0)
        _FIT[(k, v)] = (s, ((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    return _FIT[(k, v)]


def render_class(k, v=0, scale=1.0, rot=0.0, shift=(0.0, 0.0), masks=False):
    s, c = fit_params(k, v)
    t = T(scale * s, rot, c, (N / 2.0 - c[0] + shift[0], N / 2.0 - c[1] + shift[1]))
    P = pose(k, v, t)
    img, hand, arm = render(P.S, P.H, P.L, P.D)
    return (img, hand, arm) if masks else img


# ----------------------------------------------------------------------------------------------
# fixed filters (all from formulas)
# ----------------------------------------------------------------------------------------------
K1 = 7           # first conv kernel (edges, blur)
K2 = 13          # detector kernel (ridges, holes, energy)
EDGE_SIGMA = 1.2
EDGE_T, EDGE_C = 0.20, 0.5        # unsigned edge feature = clip(|e| - T, 0, C): texture suppressed, outline saturated
RIDGE_T, RIDGE_C = 0.12, 0.6      # unsigned ridge feature (finger or crease, either polarity)
HOLE_T, HOLE_C = 0.10, 0.5        # unsigned hole feature (spot differing from its surround)
ENERGY_SIGMA = 2.0
NBAR_ORI = 4
BAR_WIDTHS = (3.5, 7.0)
HOLE_R = (1.8, 3.5)
NRIDGE = NBAR_ORI * len(BAR_WIDTHS)     # 8
NHOLE = len(HOLE_R)                     # 2
# feature channels after c3: 0-3 unsigned edges, 4 edge energy (coarse layout), 5-12 ridges, 13-14 holes
NCH = 4 + 1 + NRIDGE + NHOLE            # 15 (+ 1 constant channel)


def gaussian(size, sigma):
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    g = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
    return g / g.sum()


def edge_kernels(size=K1, sigma=EDGE_SIGMA):
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    gd = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
    ks = []
    for ang in (0, 45, 90, 135):
        a = math.radians(ang)
        d = (math.cos(a) * x + math.sin(a) * y) * gd
        d -= d.mean()
        d /= np.abs(d).sum() / 2            # unit response to a unit step
        ks.append(d)
    return ks                               # 4 orientations (signed)


def bar_kernel(theta_deg, width, size=K2, length=9.0):
    """bar of given width along direction theta against flanks; response in [-1, 1] for inputs in [0, 1]."""
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    th = math.radians(theta_deg)
    u = -math.sin(th) * x + math.cos(th) * y        # across the bar
    v = math.cos(th) * x + math.sin(th) * y         # along the bar
    along = np.clip(length / 2 + 0.5 - np.abs(v), 0, 1)
    pos = np.clip(width / 2 + 0.5 - np.abs(u), 0, 1) * along
    neg = np.clip(np.abs(u) - (width / 2 + 0.5), 0, 1) * np.clip(width / 2 + 3.0 - np.abs(u), 0, 1) * along
    return pos / pos.sum() - neg / neg.sum()


def hole_kernel(rc, size=K2):
    """centre of radius rc against a surrounding ring; response in [-1, 1] for inputs in [0, 1]."""
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    d = np.hypot(x, y)
    cen = np.clip(rc + 0.5 - d, 0, 1)
    ring = np.clip(d - (rc + 0.5), 0, 1) * np.clip(rc + 3.5 - d, 0, 1)
    return ring / ring.sum() - cen / cen.sum()


def first_layer():
    """c1 (18 channels from z): 0-3: e_i - T, 4-7: -e_i - T, 8-11: e_i - T - C, 12-15: -e_i - T - C
    (so that relu differences give the unsigned, thresholded, capped edges), 16: bz, 17: -bz."""
    E = edge_kernels()
    g = gaussian(K1, 1.0)
    W = np.zeros((18, 1, K1, K1)); b = np.zeros(18)
    for i in range(4):
        W[i, 0] = E[i]; b[i] = -EDGE_T
        W[4 + i, 0] = -E[i]; b[4 + i] = -EDGE_T
        W[8 + i, 0] = E[i]; b[8 + i] = -EDGE_T - EDGE_C
        W[12 + i, 0] = -E[i]; b[12 + i] = -EDGE_T - EDGE_C
    W[16, 0] = g
    W[17, 0] = -g
    return W, b


N2 = 5 + 4 * NRIDGE + 4 * NHOLE     # 45 channels out of c2


def second_layer():
    """c2 on the 18 c1 channels: 0-3 unsigned edges, 4 coarse edge energy, then for each ridge / hole detector
    four channels (+r - T, +r - T - C, -r - T, -r - T - C) whose relu combination is the unsigned capped response."""
    W = np.zeros((N2, 18, K2, K2)); b = np.zeros(N2)
    c = K2 // 2
    ident = np.zeros((K2, K2)); ident[c, c] = 1.0
    for i in range(4):
        W[i, i] = ident; W[i, 4 + i] = ident; W[i, 8 + i] = -ident; W[i, 12 + i] = -ident
    g = gaussian(K2, ENERGY_SIGMA)
    for i in range(4):
        W[4, i] = g; W[4, 4 + i] = g; W[4, 8 + i] = -g; W[4, 12 + i] = -g
    dets = [bar_kernel(90 + 180 * o / NBAR_ORI, wdt) for wdt in BAR_WIDTHS for o in range(NBAR_ORI)]
    caps = [(RIDGE_T, RIDGE_C)] * NRIDGE
    dets += [hole_kernel(rc) for rc in HOLE_R]
    caps += [(HOLE_T, HOLE_C)] * NHOLE
    j = 5
    for k, (t, cc) in zip(dets, caps):
        for sgn, off in ((1, t), (1, t + cc), (-1, t), (-1, t + cc)):
            W[j, 16] = sgn * k; W[j, 17] = -sgn * k; b[j] = -off; j += 1
    assert j == N2
    return W, b


def third_layer():
    """c3: 1x1 conv to the NCH feature channels: edges and energy pass through, ridges / holes are combined to
    clip(|r| - T, 0, C)."""
    W = np.zeros((NCH, N2, 1, 1)); b = np.zeros(NCH)
    for i in range(5):
        W[i, i, 0, 0] = 1.0
    for d in range(NRIDGE + NHOLE):
        j = 5 + 4 * d
        W[5 + d, j, 0, 0] = 1.0; W[5 + d, j + 1, 0, 0] = -1.0; W[5 + d, j + 2, 0, 0] = 1.0; W[5 + d, j + 3, 0, 0] = -1.0
    return W, b


# numpy reference implementation of the network ops (same maths as the scorer)
def conv2d(x, W, b, stride=1, pad=0):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
                     for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def normalise(x):
    mu = x.mean(axis=(2, 3), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + EPS)


def avgpool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))


def features(imgs, L, batch=24):
    """imgs (n, 1, 28, 28) -> (n, NCH, 14, 14): pooled feature maps, each channel standardised over positions,
    exactly as the network computes them (processed in small batches to keep memory low)."""
    (W1, b1), (W2, b2), (W3, b3) = L[:3]
    out = []
    for i in range(0, len(imgs), batch):
        z = normalise(imgs[i:i + batch])
        f = np.maximum(conv2d(z, W1, b1, 1, K1 // 2), 0)
        f = np.maximum(conv2d(f, W2, b2, 1, K2 // 2), 0)
        f = conv2d(f, W3, b3, 1, 0)
        out.append(normalise(avgpool2(f)))
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------
# template bank
# ----------------------------------------------------------------------------------------------
SCALES = (0.85, 1.0, 1.15)
ROTS = (-8.0, 0.0, 8.0)
POSES = (0, 1, 2)
TK = 12                    # template kernel size on the 14x14 feature map
GROUPS = {"edges": [0, 1, 2, 3], "energy": [4], "ridges": list(range(5, 5 + NRIDGE)),
          "holes": list(range(5 + NRIDGE, NCH))}
GROUP_W = {"edges": 1.0, "energy": 1.0, "ridges": 0.6, "holes": 0.3}
ARM_W = 0.6                # the forearm counts less than the hand in the textured-region template


def template_bank(L):
    """Returns (WA, NV): WA (24*NV, NCH, TK, TK) template weights = standardised feature maps of the rendered
    hands (central TK x TK crop), each feature group scaled to unit norm times its group weight, so that the
    template conv computes a weighted sum of per-channel normalised correlations with each template."""
    variants = [(k, v, s, r) for k in range(24) for v in POSES for s in SCALES for r in ROTS]
    nv = len(POSES) * len(SCALES) * len(ROTS)
    rend = [render_class(k, v, s, r, masks=True) for (k, v, s, r) in variants]
    imgs = np.stack([r[0] for r in rend])[:, None]
    region = np.stack([r[1] + ARM_W * r[2] for r in rend])[:, None]
    off = (14 - TK) // 2
    # a real hand is textured all over (creases, knuckles, shading) while the wall is smooth, so the textured-
    # region map E of a photo is the filled hand region: for the templates E is the blurred filled shape model
    F = features(imgs, L)                                                # (24*nv, NCH, 14, 14)
    g = gaussian(K2, ENERGY_SIGMA)[None, None]
    filled = normalise(avgpool2(conv2d(region, g, np.zeros(1), 1, K2 // 2)))
    F[:, 4:5] = normalise(0.5 * F[:, 4:5] + 0.5 * filled)
    F = F[:, :, off:off + TK, off:off + TK]                              # (24*nv, NCH, TK, TK)
    WA = np.zeros_like(F)
    for g, chs in GROUPS.items():
        part = F[:, chs]
        nrm = np.sqrt((part ** 2).sum(axis=(1, 2, 3), keepdims=True))
        WA[:, chs] = GROUP_W[g] * part / np.maximum(nrm, 1e-6)
    return WA, nv


def build():
    L = [first_layer(), second_layer(), third_layer()]
    WA, NV = template_bank(L)
    layers = [
        {"name": "z", "op": "norm", "inputs": ["image"]},
        {"name": "c1", "op": "conv", "inputs": ["z"], "out": 18, "kernel": [K1, K1], "stride": 1, "padding": K1 // 2},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "c2", "op": "conv", "inputs": ["r1"], "out": N2, "kernel": [K2, K2], "stride": 1, "padding": K2 // 2},
        {"name": "r2", "op": "relu", "inputs": ["c2"]},
        {"name": "c3", "op": "conv", "inputs": ["r2"], "out": NCH, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "p1", "op": "avgpool", "inputs": ["c3"], "size": [2, 2], "stride": [2, 2]},
        {"name": "n1", "op": "norm", "inputs": ["p1"]},
        {"name": "tm", "op": "conv", "inputs": ["n1"], "out": 24 * NV, "kernel": [TK, TK], "stride": 1, "padding": 1},
        {"name": "mp", "op": "maxpool", "inputs": ["tm"], "size": [5, 5], "stride": [1, 1]},
        {"name": "rs", "op": "reshape", "inputs": ["mp"], "shape": [1, 24, NV]},
        {"name": "mv", "op": "maxpool", "inputs": ["rs"], "size": [1, NV], "stride": [1, 1]},
        {"name": "out", "op": "reshape", "inputs": ["mv"], "shape": [24]},
    ]
    weights = {}
    for name, (W, b) in zip(("c1", "c2", "c3"), L):
        weights[name + ".W"] = W.astype(np.float32); weights[name + ".b"] = b.astype(np.float32)
    weights["tm.W"] = WA.astype(np.float32); weights["tm.b"] = np.zeros(len(WA), np.float32)
    return {"layers": layers}, weights


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    net, weights = build()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(net, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **weights)
    print("wrote", out, {k: v.shape for k, v in weights.items()})
