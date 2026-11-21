"""Hand-designed ASL letter classifier (no data used). Renders 2D hand models (palm, finger capsules, thumb, wrist)
for each letter from general ASL knowledge, turns them into templates on polarity-free oriented-edge and |intensity|
channels, and builds a network: norm -> fixed Gaussian / derivative-of-Gaussian convs -> template convs with a +-3 px
shift search -> max over shifts and 48 pose variants per letter. Usage: python3 build.py OUT_DIR"""
"""Parametric 2D hand renderer: palm, finger capsules, thumb, wrist. Hand units: palm width = 1,
x to the viewer's right, y down, knuckle line at y = 0, wrist crease at y = 1. Right hand."""
import numpy as np

import os
SS = 4          # supersampling
DRAW_CREASES = False
N = 28


def D(a):
    """unit vector for an angle in degrees measured from straight up, clockwise (toward +x)."""
    r = np.deg2rad(a)
    return np.array([np.sin(r), -np.cos(r)])


# ---------------- primitives (signed distance, positive inside) ----------------
def sd_capsule(U, V, p0, p1, r0, r1):
    d = p1 - p0
    L2 = max(float(d @ d), 1e-9)
    t = np.clip(((U - p0[0]) * d[0] + (V - p0[1]) * d[1]) / L2, 0, 1)
    px, py = p0[0] + t * d[0], p0[1] + t * d[1]
    return r0 + (r1 - r0) * t - np.hypot(U - px, V - py)


def sd_superellipse(U, V, c, a, b, ang=0.0, n=3.0):
    r = np.deg2rad(ang)
    du, dv = U - c[0], V - c[1]
    u = np.cos(r) * du + np.sin(r) * dv
    v = -np.sin(r) * du + np.cos(r) * dv
    q = (np.abs(u / a) ** n + np.abs(v / b) ** n) ** (1.0 / n)
    return (1.0 - q) * min(a, b)


class P:
    def __init__(self, kind, args, shade=1.0, rim=0.35, part="hand", imp=1.0):
        self.kind, self.args, self.shade, self.rim, self.part, self.imp = kind, args, shade, rim, part, imp

    def sd(self, U, V):
        if self.kind == "cap":
            return sd_capsule(U, V, *self.args)
        return sd_superellipse(U, V, *self.args)


def cap(p0, p1, r0, r1=None, **kw):
    return P("cap", (np.asarray(p0, float), np.asarray(p1, float), r0, r0 if r1 is None else r1), **kw)


def blob(c, a, b, ang=0.0, n=3.0, **kw):
    return P("se", (np.asarray(c, float), a, b, ang, n), **kw)


def chain(base, angles, lens, w0, w1, **kw):
    """finger as joined capsules; angles absolute (deg from up), widths are full widths."""
    pts = [np.asarray(base, float)]
    for a, L in zip(angles, lens):
        pts.append(pts[-1] + L * D(a))
    tot = sum(lens)
    out, acc = [], 0.0
    for i in range(len(lens)):
        ra = (w0 + (w1 - w0) * acc / tot) / 2
        acc += lens[i]
        rb = (w0 + (w1 - w0) * acc / tot) / 2
        out.append(cap(pts[i], pts[i + 1], ra, rb, **kw))
    return out


# ---------------- rendering ----------------
def render(prims, scale, theta, center, rimw=0.07):
    """prims in hand units -> (img 28x28 in [0,1] with background 0, handmask 28x28).
    scale: pixels per hand unit; theta: rotation (deg, clockwise on screen); center: image position of hand-unit origin."""
    g = (np.arange(N * SS) + 0.5) / SS
    X, Y = np.meshgrid(g, g)
    r = np.deg2rad(theta)
    dx, dy = X - center[0], Y - center[1]
    U = (np.cos(r) * dx + np.sin(r) * dy) / scale
    V = (-np.sin(r) * dx + np.cos(r) * dy) / scale
    aa = 1.0 / (SS * scale)
    img = np.zeros_like(U)
    mask = np.zeros_like(U)
    imp = np.ones_like(U)
    dil = 1.2 / scale
    for p in prims:
        if p.part == "crease" and not DRAW_CREASES:
            continue
        s = p.sd(U, V)
        m = np.clip(s / aa + 0.5, 0, 1)
        rim = np.clip(1 - s / rimw, 0, 1) if p.rim > 0 else 0.0
        val = p.shade * (1 - p.rim * rim)
        img = img * (1 - m) + m * val
        if p.part == "crease":
            continue
        if p.shade > 0.5:
            mask = np.maximum(mask, m)
        else:
            mask = mask * (1 - m)
        if p.imp != 1.0:
            imp = np.maximum(imp, 1 + (p.imp - 1) * np.clip((s + dil) / aa + 0.5, 0, 1))
    ds = lambda a: a.reshape(N, SS, N, SS).mean((1, 3))
    return ds(img), ds(mask), ds(imp)


def extent(prims, theta):
    """bounding box (after rotation by theta) of the non-forearm parts, in hand units."""
    g = np.arange(-3, 3, 0.02)
    U, V = np.meshgrid(g, g)
    m = np.zeros_like(U, bool)
    for p in prims:
        if p.part == "hand" and p.shade > 0.5 and p.imp >= 0:
            m |= p.sd(U, V) > 0
    u, v = U[m], V[m]
    r = np.deg2rad(theta)
    x = np.cos(r) * u - np.sin(r) * v
    y = np.sin(r) * u + np.cos(r) * v
    return x.min(), x.max(), y.min(), y.max()


def place(prims, theta, fill, jx, jy):
    """choose scale/center so the hand's bounding box spans `fill` of the frame and is roughly centred."""
    x0, x1, y0, y1 = extent(prims, theta)
    s = fill * N / max(x1 - x0, y1 - y0)
    cxh, cyh = (x0 + x1) / 2, (y0 + y1) / 2   # bbox centre (rotated hand units)
    cx = N / 2 + jx - s * cxh
    cy = N / 2 + jy - s * cyh
    return s, (cx, cy)


# ======================= letter models =======================
"""Per-letter 2D hand layouts (right hand, seen by a person facing the signer), from general ASL knowledge.
Each function takes a numpy Generator for pose jitter and returns primitives in hand units
(palm width = 1, x to the viewer's right, y down, knuckle line at y = 0, wrist crease near y = 1)."""
import numpy as np

BASE = dict(P=(-0.37, 0.12), R=(-0.12, 0.03), M=(0.12, 0.0), I=(0.36, 0.04))   # finger bases: little, ring, middle, index
LEN = dict(I=0.88, M=0.98, R=0.90, P=0.70)
WID = dict(I=0.23, M=0.24, R=0.22, P=0.19)
SKIN = 0.95
WRIST = [0.35]   # how much wrist (hand units past the palm) the square crop takes in; set per variant


def jit(rng, s):
    return float(rng.normal(0, s))


def forearm(x=0.0, y=0.85, ang=0.0, w=0.74):
    """wrist/forearm leaving the palm; ang = direction (deg from straight down, clockwise toward -x).
    The first WRIST[0] units count as part of the hand for framing, the rest runs off the frame."""
    d = np.array([-np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang))])
    p0 = np.array([x, y])
    return [cap(p0, p0 + 3.0 * d, w / 2, w / 2 * 1.1, shade=0.85, rim=0.0, part="arm"),
            blob(p0 + (0.15 + WRIST[0]) / 2 * d, w / 2 * 0.98, (0.15 + WRIST[0]) / 2, ang, n=6.0, shade=0.85, rim=0.0, part="hand")]


def palm(rng, w=1.0, h=1.0, c=(0.0, 0.5)):
    """palm: rounded square, narrowing toward the wrist."""
    a, b = 0.5 * w * (1 + jit(rng, 0.04)), 0.5 * h * (1 + jit(rng, 0.04))
    return [blob((c[0], c[1] - 0.1 * b), a, 0.75 * b, n=3.0, shade=SKIN, rim=0.25),
            blob((c[0], c[1] + 0.25 * b), 0.85 * a, 0.75 * b, n=2.2, shade=SKIN, rim=0.0)]


def ext(f, a, rng, bend=0.0, lscale=1.0):
    """extended finger f at angle a (deg from straight up, clockwise)."""
    L = LEN[f] * lscale * (1 + jit(rng, 0.05))
    a = a + jit(rng, 3.0)
    return chain(BASE[f], [a, a + bend, a + 2 * bend], [0.45 * L, 0.30 * L, 0.25 * L], WID[f], WID[f] * 0.82, shade=SKIN)


def curl(f, rng, top=-0.10, bot=0.38):
    """finger folded into the palm, seen from the front: a short stub over the upper palm (its outline = creases)."""
    x, y = BASE[f]
    w = WID[f] * 1.12
    t = top + 0.5 * y + jit(rng, 0.03)
    return [cap((x, t), (x, bot + jit(rng, 0.03)), w / 2, w / 2 * 0.95, shade=SKIN, rim=0.5)]


def thumb_across(rng, y0=0.78, y1=0.45, xend=-0.25):
    yy = y1 + jit(rng, 0.04)
    return [cap((0.45, y0), (0.05, yy), 0.14, 0.12, shade=SKIN, rim=0.45),
            cap((0.05, yy), (xend, yy - 0.04), 0.12, 0.10, shade=SKIN, rim=0.45)]


def thumb_out(rng, a, base=(0.45, 0.68)):
    a = a + jit(rng, 5.0)
    return chain(base, [a, a + 5], [0.45, 0.38], 0.27, 0.20, shade=SKIN)


def hole(c, a, b, ang=0.0):
    """dark gap / shadow where background shows through or the hand is deeply shaded."""
    return blob(c, a, b, ang, n=2.0, shade=0.25, rim=0.0)


def side_fist(r):
    """fist turned so the palm faces the signer's left: curled fingers seen as stacked horizontal bars."""
    return [blob((0.0, 0.45), 0.5, 0.5, n=3.0, shade=SKIN, rim=0.25)] + \
        [cap((-0.38, y + jit(r, 0.02)), (0.35, y), 0.11, 0.11, shade=SKIN, rim=0.5) for y in (0.22, 0.44, 0.66)]


# ---------------------------------------------------------------- letters
def A(r):   # fist, thumb straight up against the side of the index finger
    return (forearm() + palm(r) + [p for f in "PRMI" for p in curl(f, r, top=-0.12, bot=0.40)] +
            [cap((0.50, 0.85), (0.60, 0.25), 0.14, 0.13, shade=SKIN, rim=0.45),
             cap((0.60, 0.25), (0.59, -0.02 + jit(r, 0.05)), 0.13, 0.11, shade=SKIN, rim=0.45)])


def B(r):   # flat hand, four fingers up together, thumb folded across the palm
    return (forearm() + palm(r) + ext("P", -4, r) + ext("R", -1.5, r) + ext("M", 0, r) + ext("I", 2, r) +
            thumb_across(r, y0=0.82, y1=0.55, xend=-0.2))


def C(r):   # curved hand seen from the thumb side, opening to the viewer's right
    k = jit(r, 6)
    return (forearm(-0.3, 0.85, 15, 0.7) + [blob((-0.3, 0.45), 0.30, 0.55, n=2.5, shade=SKIN, rim=0.25)] +
            chain((-0.35, 0.0), [60 + k, 95 + k, 135 + k], [0.45, 0.40, 0.30], 0.42, 0.32, shade=SKIN) +
            chain((-0.2, 0.85), [65 + k / 2, 40 + k / 2], [0.45, 0.35], 0.28, 0.22, shade=SKIN))


def D_(r):  # index up, other fingers curled to meet the thumb in a ring
    return (forearm() + palm(r) + [p for f in "PRM" for p in curl(f, r, top=-0.05, bot=0.35)] +
            [cap((0.45, 0.80), (0.15, 0.45), 0.14, 0.12, shade=SKIN, rim=0.45), hole((0.22, 0.30), 0.08, 0.10)] +
            ext("I", 1, r))


def E(r):   # fingers bent at the middle joints, tips resting on the thumb across the palm
    out = forearm() + palm(r) + thumb_across(r, y0=0.75, y1=0.50, xend=-0.32)
    for f in "PRMI":
        x, y = BASE[f]
        pip = y - 0.34 * LEN[f] / 0.98 + jit(r, 0.03)
        w = WID[f] * 1.1
        out += [cap((x, y + 0.1), (x, pip), w / 2, w / 2, shade=SKIN, rim=0.3),
                cap((x, pip), (x, 0.40), w / 2 * 1.05, w / 2, shade=SKIN, rim=0.55)]
    return out


def F(r):   # index and thumb touch in a ring, other three fingers up and spread
    s = jit(r, 4)
    return (forearm() + palm(r) + ext("P", -26 + s, r) + ext("R", -13 + s / 2, r) + ext("M", -2, r) +
            chain(BASE["I"], [30, 120, 200], [0.36, 0.30, 0.22], 0.23, 0.19, shade=SKIN) +
            chain((0.45, 0.75), [40, 10], [0.36, 0.32], 0.26, 0.20, shade=SKIN))


def G(r):   # index pointing to the viewer's right, thumb parallel above it
    a = jit(r, 4)
    return (forearm(-0.1, 0.8, 25) + side_fist(r) +
            [cap((0.1, -0.17), (0.8, -0.24 + jit(r, 0.04)), 0.12, 0.10, shade=SKIN, rim=0.45),
             cap((0.0, 0.02), (0.0, 0.02) + 1.3 * D(90 + a), 0.12, 0.10, shade=SKIN, rim=0.45)])


def H(r):   # index and middle together pointing to the viewer's right
    a = jit(r, 4)
    return (forearm(-0.1, 0.8, 25) + side_fist(r) +
            [cap((0.0, 0.20), (0.0, 0.20) + 1.32 * D(90 + a), 0.12, 0.10, shade=SKIN, rim=0.45),
             cap((0.0, -0.03), (0.0, -0.03) + 1.25 * D(90 + a), 0.12, 0.10, shade=SKIN, rim=0.45)])


def I(r):   # fist with the little finger up
    return (forearm() + palm(r) + [p for f in "RMI" for p in curl(f, r)] + ext("P", -4, r) + thumb_across(r, 0.7, 0.36, -0.2))


def K(r):   # index up, middle angled out toward the viewer (foreshortened), thumb at their base
    return (forearm() + palm(r) + [p for f in "PR" for p in curl(f, r)] +
            chain(BASE["M"], [-28 + jit(r, 5)] * 2, [0.42, 0.34], 0.24, 0.2, shade=SKIN) +
            [cap((0.45, 0.75), (0.24, 0.02), 0.14, 0.11, shade=SKIN, rim=0.45)] + ext("I", 3, r))


def L(r):   # index up, thumb straight out to the viewer's right
    return forearm() + palm(r) + [p for f in "PRM" for p in curl(f, r)] + ext("I", 0, r) + thumb_out(r, 82)


def M(r):   # thumb tucked under three fingers that drape over it
    return (forearm() + palm(r) + curl("P", r) + [hole((-0.25, 0.66), 0.06, 0.06)] +
            [cap((-0.25, 0.66), (-0.2, 0.75), 0.09, 0.09, shade=SKIN, rim=0.4)] +
            [p for f in "RMI" for p in curl(f, r, top=-0.12, bot=0.62)])


def N_(r):  # thumb tucked under two fingers
    return (forearm() + palm(r) + curl("P", r) + curl("R", r) +
            [cap((0.0, 0.66), (0.05, 0.75), 0.09, 0.09, shade=SKIN, rim=0.4)] +
            [p for f in "MI" for p in curl(f, r, top=-0.12, bot=0.62)])


def O(r):   # all fingertips meet the thumb in a ring, seen from the thumb side
    k = jit(r, 5)
    return (forearm(-0.3, 0.85, 15, 0.7) + [blob((-0.3, 0.45), 0.30, 0.55, n=2.5, shade=SKIN, rim=0.25)] +
            chain((-0.35, 0.0), [60 + k, 110 + k, 165 + k], [0.42, 0.36, 0.32], 0.40, 0.30, shade=SKIN) +
            chain((-0.2, 0.85), [55, 25], [0.40, 0.32], 0.28, 0.22, shade=SKIN))


def P_(r):  # K pointed downward: index out to the upper left, middle down, hand body upper right
    a = jit(r, 8)
    return (forearm(0.4, 0.4, -30, 0.75) + [blob((0.25, 0.2), 0.5, 0.42, -30, n=2.5, shade=SKIN, rim=0.25)] +
            [hole((-0.02, 0.32), 0.1, 0.12),
             cap((0.0, 0.25), (0.0, 0.25) + 0.75 * D(200 + a / 2), 0.12, 0.10, shade=SKIN, rim=0.45),
             cap((0.2, 0.1), (-0.15, 0.25), 0.12, 0.11, shade=SKIN, rim=0.45),
             cap((0.05, 0.0), (0.05, 0.0) + 1.0 * D(-62 + a), 0.12, 0.10, shade=SKIN, rim=0.45)])


def Q(r):   # G pointed downward: index and thumb hang down, back of the hand on top
    a = jit(r, 5)
    return (forearm(0.5, 0.2, -25, 0.75) + [blob((0.15, 0.0), 0.72, 0.42, 8, n=2.5, shade=SKIN, rim=0.25)] +
            [hole((-0.12, 0.42), 0.12, 0.16),
             cap((0.3, 0.25), (0.3, 0.25) + 0.62 * D(182 + a), 0.14, 0.12, shade=SKIN, rim=0.45),
             cap((-0.4, 0.1), (-0.4, 0.1) + 0.95 * D(192 + a), 0.13, 0.10, shade=SKIN, rim=0.45)])


def R(r):   # index and middle crossed, pointing up
    return (forearm() + palm(r) + [p for f in "PR" for p in curl(f, r)] + ext("I", -7, r) + ext("M", 7, r) +
            thumb_across(r, 0.75, 0.42, -0.3))


def S(r):   # fist with the thumb across the front of the fingers
    return (forearm() + palm(r) + [p for f in "PRMI" for p in curl(f, r, top=-0.1, bot=0.36)] +
            [cap((0.55, 0.65), (0.1, 0.36 + jit(r, 0.04)), 0.14, 0.13, shade=SKIN, rim=0.45),
             cap((0.1, 0.36), (-0.2, 0.31), 0.13, 0.11, shade=SKIN, rim=0.45)])


def T(r):   # fist turned sideways, thumb tip pokes up between index and middle, index bent across
    a = jit(r, 4)
    return (forearm(0.0, 0.85, 10) + side_fist(r) +
            [cap((0.05, 0.1), (0.05 + jit(r, 0.05), -0.45), 0.12, 0.11, shade=SKIN, rim=0.45),
             cap((-0.15, 0.04), (-0.15, 0.04) + 1.1 * D(95 + a), 0.12, 0.10, shade=SKIN, rim=0.45)])


def U(r):   # index and middle up together
    return (forearm() + palm(r) + [p for f in "PR" for p in curl(f, r)] + ext("M", -1, r) + ext("I", 1.5, r) +
            thumb_across(r, 0.75, 0.42, -0.3))


def V(r):   # index and middle up, spread
    s = jit(r, 4)
    return (forearm() + palm(r) + [p for f in "PR" for p in curl(f, r)] + ext("M", -12 - s, r) + ext("I", 13 + s, r) +
            thumb_across(r, 0.75, 0.42, -0.3))


def W(r):   # index, middle and ring up, spread
    s = jit(r, 4)
    return (forearm() + palm(r) + curl("P", r) + ext("R", -15 - s, r) + ext("M", 0, r) + ext("I", 15 + s, r) +
            thumb_across(r, 0.75, 0.45, -0.38))


def X(r):   # index raised and hooked
    return (forearm() + palm(r) + [p for f in "PRM" for p in curl(f, r)] +
            chain(BASE["I"], [jit(r, 5), 75 + jit(r, 8), 150 + jit(r, 10)], [0.36, 0.26, 0.2], 0.24, 0.2, shade=SKIN) +
            thumb_across(r, 0.72, 0.38, -0.15))


def Y(r):   # little finger up and out, thumb out
    return forearm() + palm(r) + [p for f in "RMI" for p in curl(f, r)] + ext("P", -22, r) + thumb_out(r, 58)


LETTERS = [A, B, C, D_, E, F, G, H, I, K, L, M, N_, O, P_, Q, R, S, T, U, V, W, X, Y]
NAMES = "ABCDEFGHIKLMNOPQRSTUVWXY"

# ======================= network: features from formulas, templates from the hand models =======================
import json, sys
V = 48                                # pose variants per letter
PAD = 3                               # template shift search: +-PAD pixels
SIG_I, SIG_D, SIG_B = 0.8, 1.0, 1.0   # intensity blur, derivative-of-Gaussian scale, edge-energy blur
W_EDGE, W_INT, INTERIOR = 1.0, 0.5, 0.25


def conv(x, W, b, pad):
    c, h, w = x.shape
    o, _, k, _ = W.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    ho, wo = h + 2 * pad - k + 1, w + 2 * pad - k + 1
    out = np.zeros((o, ho, wo))
    for dy in range(k):
        for dx in range(k):
            out += np.einsum('oi,ihw->ohw', W[:, :, dy, dx], xp[:, dy:dy + ho, dx:dx + wo])
    return out + b[:, None, None]


def norm(x):
    mu = x.mean((1, 2), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean((1, 2), keepdims=True) + 1e-5)


def gauss(sig, r):
    g = np.arange(-r, r + 1)
    X, Y = np.meshgrid(g, g)
    k = np.exp(-(X ** 2 + Y ** 2) / (2 * sig ** 2))
    return k / k.sum(), X, Y


def filters():
    gI, _, _ = gauss(SIG_I, 2)
    WI = gI[None, None]
    g, X, Y = gauss(SIG_D, 3)
    WE = np.zeros((8, 1, 7, 7))
    for j, th in enumerate(np.deg2rad([0, 45, 90, 135])):
        d = (np.cos(th) * X + np.sin(th) * Y) * g        # derivative of Gaussian along direction th, both signs
        d = d / np.abs(d).sum() * 2
        WE[2 * j, 0], WE[2 * j + 1, 0] = d, -d
    gB, _, _ = gauss(SIG_B, 2)
    WB = np.zeros((4, 8, 5, 5))                           # sum the two signs (polarity-free edge energy), blur
    for j in range(4):
        WB[j, 2 * j], WB[j, 2 * j + 1] = gB, gB
    return WI, WE, WB


WI, WE, WB = filters()
Z1, Z8, Z4 = np.zeros(1), np.zeros(8), np.zeros(4)


def features(img):
    n0 = norm(img[None])
    fI = conv(n0, WI, Z1, 2)
    eE = conv(np.maximum(conv(n0, WE, Z8, 3), 0), WB, Z4, 2)
    return fI, eE


def std(a):
    a = a - a.mean()
    return a / np.sqrt((a ** 2).sum())


def make_templates(seed=0):
    rng = np.random.default_rng(seed)
    TE = np.zeros((24 * V, 4, N, N)); TI = np.zeros((24 * V, 1, N, N))
    for c in range(24):
        for v in range(V):
            WRIST[0] = rng.uniform(0.15, 0.5)
            prims = LETTERS[c](rng)
            th = rng.uniform(-10, 10)
            fill = rng.uniform(0.84, 1.0)
            s, ctr = place(prims, th, fill, rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
            img, mask, _ = render(prims, s, th, ctr)
            fI, eE = features(img)
            hb = conv(mask[None], WI, Z1, 2)
            e = eE / eE.max() + INTERIOR * hb
            t = c * V + v
            TE[t] = W_EDGE * std(e - e.mean((1, 2), keepdims=True))
            TI[t] = W_INT * std(fI)
    return TE, TI


def write(out_dir):
    TE, TI = make_templates()
    T = 24 * V
    k = 2 * PAD + 1
    L = []
    def add(name, op, inputs, **kw):
        L.append(dict(name=name, op=op, inputs=inputs, **kw))
    add('n0', 'norm', ['image'])
    add('cI', 'conv', ['n0'], out=1, kernel=[5, 5], stride=1, padding=2)
    add('nI', 'norm', ['cI'])
    add('cE', 'conv', ['n0'], out=8, kernel=[7, 7], stride=1, padding=3)
    add('rE', 'relu', ['cE'])
    add('bE', 'conv', ['rE'], out=4, kernel=[5, 5], stride=1, padding=2)
    add('nE', 'norm', ['bE'])
    add('tE', 'conv', ['nE'], out=T, kernel=[N, N], stride=1, padding=PAD)
    add('tIp', 'conv', ['nI'], out=T, kernel=[N, N], stride=1, padding=PAD)
    add('tIn', 'conv', ['nI'], out=T, kernel=[N, N], stride=1, padding=PAD)
    add('rIp', 'relu', ['tIp'])
    add('rIn', 'relu', ['tIn'])
    add('s', 'add', ['tE', 'rIp', 'rIn'])
    add('m1', 'maxpool', ['s'], size=[k, k], stride=[k, k])
    add('r1', 'reshape', ['m1'], shape=[1, 24, V])
    add('m2', 'maxpool', ['r1'], size=[1, V], stride=[1, V])
    add('scores', 'reshape', ['m2'], shape=[24])
    w = {'cI.W': WI, 'cI.b': Z1, 'cE.W': WE, 'cE.b': Z8, 'bE.W': WB, 'bE.b': Z4,
         'tE.W': TE, 'tE.b': np.zeros(T), 'tIp.W': TI, 'tIp.b': np.zeros(T), 'tIn.W': -TI, 'tIn.b': np.zeros(T)}
    os.makedirs(out_dir, exist_ok=True)
    json.dump({'layers': L}, open(os.path.join(out_dir, 'network.json'), 'w'), indent=1)
    np.savez(os.path.join(out_dir, 'weights.npz'), **{k: v.astype(np.float32) for k, v in w.items()})


if __name__ == '__main__':
    write(sys.argv[1])
