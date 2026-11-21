"""Builds a hand-designed digit classifier: python3 build.py OUT_DIR  ->  OUT_DIR/network.json, OUT_DIR/weights.npz

No data is read.  Every weight comes from formulas and from digit shapes typed here as pen paths.

Front end (formulas): eight Gaussian-derivative direction filters -> relu -> 8-direction "parallelogram" split
(1x1 conv + relu) -> Gaussian pooling at every pixel -> one global normalisation of the whole feature map.
Templates read that map at 14 x 14 taps two pixels apart.

Templates: each digit is drawn by code as pen strokes along typed control points (several writing styles per
digit, random slant / proportions / thickness from a fixed seed), put through the same size and centre-of-mass
framing the task describes, and passed through the same front end.  Every drawing becomes one template, normalised
by itself alone.  A template is cut into four quadrants; each quadrant may slide by one pixel in any direction
(a "gather" conv lays the 3x3 shifted samplings out as its 3x3 output positions, a 1x1 conv holds the templates,
then maxpool).  A quadrant match is 2<x,t> - |t|^2, i.e. minus the squared distance up to the image's own energy.
Class score = (best whole template: its four quadrant matches plus a coarse whole-template match at twice the
blur and half the weight) + (image-to-class: each quadrant matched by the best of any template of the class).
"""
import json
import sys
import numpy as np

SEED = 20240611
COUNTS = [1000, 1000, 3400, 2600, 2200, 2400, 2400, 2200, 3600, 3200]   # templates per class (more for the more varied digits)
NDIR = 8                    # gradient directions
SIG_G = 1.0                 # scale of the Gaussian-derivative filters (pixels)
SIG_P = 1.8                 # scale of the Gaussian pooling (pixels)
SIG_C = 3.6                 # scale of the coarse map (pixels), sampled every 4 pixels
COARSE_WEIGHT = 0.5         # the coarse match counts half as much as the fine one (as in a spatial pyramid)
FULL = 29                   # pooled map: FULL x FULL positions one pixel apart, position i centred at pixel i - 0.5
QUAD = 7                    # each quadrant is 7 x 7 taps, 2 pixels apart (taps at pixels 0.5, 2.5, ..., 26.5)


# ----------------------------------------------------------------------------------------------------------------
# Pen-path digit drawings
# ----------------------------------------------------------------------------------------------------------------
class Pen:
    """Random helpers for a batch of B drawings (coordinates: x to the right, y downward, digit height about 1)."""

    def __init__(self, rng, B):
        self.rng, self.B = rng, B

    def U(self, lo, hi):
        return self.rng.uniform(lo, hi, self.B)

    def N(self, mu, sd):
        return self.rng.normal(mu, sd, self.B)

    def coin(self, p):
        return (self.rng.uniform(0, 1, self.B) < p).astype(np.float64)

    def pt(self, x, y, j=0.025):
        out = np.empty((self.B, 2))
        out[:, 0] = x
        out[:, 1] = y
        return out + self.rng.normal(0, j, (self.B, 2))


def curve(*pts):
    return np.stack(pts, axis=1)            # (B, m, 2): one smooth curve through m points


def lerp(a, b, t):
    t = np.asarray(t, dtype=np.float64)
    if t.ndim == 1:
        t = t[:, None]
    return a + (b - a) * t


def spline(P, per_unit=60.0):
    """Centripetal Catmull-Rom curve through the points P (B, m, 2); returns sample points (B, S, 2)."""
    B, m, _ = P.shape
    ext = np.concatenate([2 * P[:, :1] - P[:, 1:2], P, 2 * P[:, -1:] - P[:, -2:-1]], axis=1)
    out = []
    for i in range(m - 1):
        p0, p1, p2, p3 = ext[:, i], ext[:, i + 1], ext[:, i + 2], ext[:, i + 3]
        d01 = np.sqrt(np.sqrt(((p1 - p0) ** 2).sum(1))) + 1e-4
        d12 = np.sqrt(np.sqrt(((p2 - p1) ** 2).sum(1))) + 1e-4
        d23 = np.sqrt(np.sqrt(((p3 - p2) ** 2).sum(1))) + 1e-4
        n = int(np.ceil(np.sqrt(((p2 - p1) ** 2).sum(1)).max() * per_unit)) + 1
        u = (np.arange(n) / n)[None, :, None]                          # (1, n, 1)
        t0 = np.zeros(B)[:, None, None]
        t1 = d01[:, None, None]
        t2 = t1 + d12[:, None, None]
        t3 = t2 + d23[:, None, None]
        t = t1 + u * (t2 - t1)
        q0, q1, q2, q3 = p0[:, None], p1[:, None], p2[:, None], p3[:, None]
        a1 = (t1 - t) / (t1 - t0) * q0 + (t - t0) / (t1 - t0) * q1
        a2 = (t2 - t) / (t2 - t1) * q1 + (t - t1) / (t2 - t1) * q2
        a3 = (t3 - t) / (t3 - t2) * q2 + (t - t2) / (t3 - t2) * q3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(P[:, -1:])
    return np.concatenate(out, axis=1)


# Each style returns a list of curves.  Corners are where two curves share an end point.

def digit0(g):
    a = g.U(0.22, 0.42)                               # half width (half height is 0.5)
    th0 = np.deg2rad(g.U(45, 115))                    # start near the top
    sweep = np.deg2rad(g.U(335, 400))                 # anticlockwise, a little short of or past closing
    rho_end = g.U(0.80, 1.08)                         # the end lands inside or outside the start
    egg = g.N(0, 0.12)                                # wider at the top or at the bottom
    pinch = g.coin(0.3) * g.U(0.05, 0.25)             # pointed join at the top
    m = 14
    pts = []
    for i in range(m):
        f = i / (m - 1)
        th = th0 + sweep * f
        rho = 1 + (rho_end - 1) * f ** 2
        if i == 0 or i == m - 1:
            rho = rho * (1 + pinch)
        x = a * rho * np.cos(th) * (1 + egg * np.sin(th))
        y = 0.5 - 0.5 * rho * np.sin(th)
        pts.append(g.pt(x, y, 0.012))
    return [curve(*pts)]


def digit1(g, flag, base):
    top = g.pt(0, 0, 0.0)
    bot = g.pt(0, 1, 0.0)
    mid = g.pt(g.N(0, 0.035), g.U(0.35, 0.65), 0.0)
    cs = [curve(top, mid, bot)]
    if flag:                                                  # short stroke rising to the top from the lower left
        f = g.U(0.15, 0.36)
        ang = np.deg2rad(g.U(25, 55))                         # measured from the vertical
        cs.append(curve(g.pt(-f * np.sin(ang), f * np.cos(ang), 0.0), top))
    if base:
        yb = g.N(1.0, 0.015)
        cs.append(curve(g.pt(-g.U(0.15, 0.38), yb, 0.0), g.pt(g.U(0.15, 0.38), yb + g.N(0, 0.02), 0.0)))
    return cs


def digit2(g, loop):
    S = g.pt(g.U(-0.02, 0.20), g.U(0.06, 0.42))               # start, left and below the top
    T = g.pt(g.U(0.20, 0.45), 0.0)
    R = g.pt(g.U(0.48, 0.70), g.U(0.16, 0.34))
    M = g.pt(g.U(0.20, 0.46), g.U(0.52, 0.72))                # on the diagonal
    if not loop:
        yb = g.U(0.93, 1.0)
        C = g.pt(g.U(-0.24, 0.10), yb)                        # bottom-left corner
        rise = g.N(-0.03, 0.08)                               # base runs level, up or down
        E = g.pt(g.U(0.60, 1.00), yb + rise)
        Bm = lerp(C, E, 0.5) + g.pt(0, g.N(0.0, 0.05), 0.0)
        return [curve(S, T, R, M, C), curve(C, Bm, E)]
    ls = g.U(0.08, 0.34)                                      # loop size
    x0 = g.U(-0.20, 0.06)
    yb = g.U(0.90, 1.0)
    L1 = g.pt(x0 + 0.7 * ls, yb, 0.012)                       # bottom of the loop
    L2 = g.pt(x0, yb - 0.55 * ls, 0.012)                      # left of the loop
    L3 = g.pt(x0 + 0.6 * ls, yb - 1.05 * ls, 0.012)           # top of the loop
    L4 = g.pt(x0 + 1.5 * ls + 0.10, yb - 0.35 * ls + g.N(0, 0.04), 0.012)
    E = g.pt(g.U(0.62, 1.00), yb + g.N(-0.04, 0.08))
    return [curve(S, T, R, M, L1, L2, L3, L4, E)]


def digit2z(g, tick, rounded):
    """A two written like a Z: flat top bar, straight diagonal, long base."""
    S = g.pt(g.U(-0.05, 0.15), g.U(0.0, 0.10))                # left end of the top bar
    R = g.pt(g.U(0.50, 0.72), g.U(0.0, 0.12))                 # top-right corner
    Tm = lerp(S, R, 0.5) + g.pt(0, -g.U(0.0, 0.06), 0.0)      # the bar arches up a little
    yb = g.U(0.93, 1.0)
    C = g.pt(g.U(-0.24, 0.10), yb)                            # bottom-left corner
    M = lerp(R, C, g.U(0.4, 0.6)) + g.pt(g.N(0.0, 0.04), 0, 0.0)
    E = g.pt(g.U(0.60, 1.00), yb + g.N(-0.03, 0.08))
    Bm = lerp(C, E, 0.5) + g.pt(0, g.N(0.0, 0.05), 0.0)
    if rounded:
        cs = [curve(S, Tm, R, M, C), curve(C, Bm, E)]
    else:
        cs = [curve(S, Tm, R), curve(R, M, C), curve(C, Bm, E)]
    if tick:
        cs.append(curve(g.pt(S[:, 0] + g.N(0, 0.02), S[:, 1] + g.U(0.06, 0.18), 0.0), S))
    return cs


def digit3(g, flat):
    W = g.pt(g.U(0.10, 0.32) if flat else g.U(0.0, 0.32), g.U(0.36, 0.52))   # waist (left end of the middle bar)
    Wb = W + g.pt(g.U(0.12, 0.2), g.U(0.0, 0.05), 0.01)
    R2 = g.pt(g.U(0.50, 0.70), g.U(0.64, 0.78))
    Bt = g.pt(g.U(0.15, 0.40), 1.0)
    E = g.pt(g.U(-0.25, 0.10), g.U(0.76, 0.95))               # the tail may run far to the left
    low = curve(W, Wb, R2, Bt, E)
    if flat:
        S = g.pt(g.U(-0.12, 0.10), g.U(0.0, 0.06))
        TR = g.pt(g.U(0.44, 0.66), g.U(-0.02, 0.06))
        return [curve(S, TR), curve(TR, W), low]
    S = g.pt(g.U(-0.15, 0.12), g.U(0.04, 0.30))
    T = g.pt(g.U(0.18, 0.38), 0.0)
    R1 = g.pt(g.U(0.42, 0.64), g.U(0.12, 0.28))
    Wa = W + g.pt(g.U(0.12, 0.2), -g.U(0.0, 0.06), 0.01)
    return [curve(S, T, R1, Wa, W), low]


def digit4(g, closed, rounded=False):
    xv = g.U(0.36, 0.56)                                      # x of the long vertical stroke
    yb = g.U(0.48, 0.70)                                      # height of the cross bar
    P1 = g.pt(g.U(-0.04, 0.08), yb)                           # left end of the bar
    P2 = g.pt(xv + (g.U(0.10, 0.32) if closed else g.U(0.02, 0.32)), yb + g.N(-0.02, 0.04))    # right end of the bar
    Q1 = g.pt(xv + g.N(0, 0.03), 1.0)
    if closed:
        Q0 = g.pt(xv + g.N(0, 0.02), g.U(0.0, 0.06))
        P0 = Q0 + g.pt(g.N(0, 0.02), g.N(0.01, 0.02), 0.0)
        return [curve(P0, P1), curve(P1, P2), curve(Q0, Q1)]
    P0 = g.pt(np.minimum(g.U(-0.02, 0.26), xv - 0.2), g.U(0.0, 0.16))   # top of the left stroke, well left of the stem
    Q0 = g.pt(xv + g.N(0.01, 0.03), g.U(-0.04, 0.22))
    Pm = lerp(P0, P1, 0.5) + g.pt(g.N(0.0, 0.03), 0, 0.0)
    if rounded:                                               # left stroke runs into the bar in one sweep
        return [curve(P0, Pm, P1, lerp(P1, P2, 0.5), P2), curve(Q0, Q1)]
    return [curve(P0, Pm, P1), curve(P1, P2), curve(Q0, Q1)]


def digit5(g, smooth):
    TL = g.pt(g.U(0.06, 0.22), g.U(0.0, 0.06))
    TR = g.pt(g.U(0.45, 0.88), g.U(-0.10, 0.06))
    D = g.pt(TL[:, 0] + g.N(-0.03, 0.04), g.U(0.28, 0.52))    # where the down stroke meets the bowl
    B2 = g.pt(g.U(0.50, 0.72), g.U(0.56, 0.74))
    B3 = g.pt(g.U(0.22, 0.42), g.U(0.96, 1.0))
    E = g.pt(g.U(-0.10, 0.14), g.U(0.72, 0.95))               # the bowl may curl well round
    bar0 = TL + g.pt(g.N(0.0, 0.03), g.N(0.0, 0.02), 0.0)     # the bar may sit a little off the down stroke
    if smooth:
        B1 = g.pt(g.U(0.26, 0.40), D[:, 1] + g.U(0.0, 0.08))
        return [curve(TR, bar0), curve(TL, D, B1, B2, B3, E)]
    B1 = g.pt(g.U(0.28, 0.44), D[:, 1] - g.U(0.0, 0.09))
    return [curve(TR, bar0), curve(TL, D), curve(D, B1, B2, B3, E)]


def digit6(g):
    S = g.pt(g.U(0.10, 0.70), 0.0)                            # top of the stem
    A = g.pt(g.U(0.02, 0.26), g.U(0.26, 0.44))
    Bl = g.pt(g.U(-0.02, 0.06), g.U(0.58, 0.78))              # leftmost point
    Bt = g.pt(g.U(0.22, 0.42), 1.0)
    top = g.U(0.32, 0.64)                                     # top of the loop
    Rr = g.pt(g.U(0.45, 0.80), (top + 1) / 2 + g.N(0, 0.03))
    Tp = g.pt(g.U(0.26, 0.46), top)
    Cl = g.pt(g.U(0.0, 0.22), top + g.U(0.05, 0.25))          # the loop closes on or near the stem
    return [curve(S, A, Bl, Bt, Rr, Tp, Cl)]


def digit7(g, tick, bar, rounded=False):
    TL = g.pt(g.U(-0.02, 0.06), g.U(0.0, 0.08))
    TR = g.pt(g.U(0.50, 0.70), g.U(-0.02, 0.08))
    Bt = g.pt(g.U(0.10, 0.44), 1.0)
    Mid = lerp(TR, Bt, g.U(0.4, 0.6)) + g.pt(g.N(0.015, 0.04), 0, 0.0)
    Tm = lerp(TL, TR, 0.5) + g.pt(0, g.N(-0.01, 0.025), 0.0)
    if rounded:
        cs = [curve(TL, Tm, TR, Mid, Bt)]
    else:
        cs = [curve(TL, Tm, TR), curve(TR, Mid, Bt)]
    if tick:
        cs.append(curve(g.pt(TL[:, 0] + g.N(0, 0.02), TL[:, 1] + g.U(0.10, 0.24), 0.0), TL))
    if bar:
        yc = g.U(0.46, 0.60)
        xm = lerp(TR, Bt, yc)[:, 0]
        cs.append(curve(g.pt(xm - g.U(0.14, 0.28), yc + g.N(0, 0.02), 0.0), g.pt(xm + g.U(0.12, 0.26), yc + g.N(0, 0.02), 0.0)))
    return cs


def digit8(g, straight=False):
    cx = 0.3
    yc = g.U(0.36, 0.60)                                      # height of the crossing
    big = g.U(0.16, 0.32)                                     # half width of the wider loop
    small = np.maximum(big * g.U(0.45, 1.0), 0.10)            # the other loop may be narrower, but stays a loop
    which = g.coin(0.5)
    wt = which * big + (1 - which) * small                    # half width of the top loop
    wb = which * small + (1 - which) * big                    # half width of the bottom loop
    S = g.pt(cx + wt * g.U(0.3, 1.6), g.U(-0.04, 0.20))       # start, upper right
    T = g.pt(cx + g.N(0, 0.04), 0.0)
    TL = g.pt(cx - wt, yc * g.U(0.35, 0.55))
    X1 = g.pt(cx + g.N(0, 0.03), yc, 0.012)
    BR = g.pt(cx + wb, (1 + yc) / 2 + g.N(0.01, 0.04))
    Bt = g.pt(cx + g.N(0, 0.04), 1.0)
    BL = g.pt(cx - wb, (1 + yc) / 2 + g.N(0.0, 0.04))
    X2 = X1 + g.pt(0, 0, 0.012)
    if straight:                                              # the stroke comes back up in a straight sweep
        E = g.pt(cx + wt * g.U(0.8, 2.0), g.U(-0.06, 0.20))
        return [curve(S, T, TL, X1, BR, Bt, BL, X2, E)]
    TR = g.pt(cx + wt, yc * g.U(0.4, 0.6))
    E = g.pt(cx + wt * g.U(-0.2, 1.4), g.U(-0.06, 0.14))
    return [curve(S, T, TL, X1, BR, Bt, BL, X2, TR, E)]


def digit9(g, hook, open_top=False, big_loop=False):
    xr = g.U(0.44, 0.64)                                      # x of the stem
    Sj = g.pt(xr + g.N(0.01, 0.025), g.U(0.0, 0.16))          # top of the stem
    if big_loop:                                              # like a written "a": tall loop, short stem
        L = g.pt(g.U(-0.04, 0.10), g.U(0.25, 0.45))
        Bm = g.pt(g.U(0.18, 0.38), g.U(0.55, 0.80))
        J = g.pt(xr, g.U(0.10, 0.40), 0.012)
    else:
        L = g.pt(g.U(-0.04, 0.10), g.U(0.16, 0.32))
        Bm = g.pt(g.U(0.18, 0.38), g.U(0.36, 0.60))
        J = g.pt(xr, g.U(0.10, 0.34), 0.012)                  # the loop comes back to the stem here
    if open_top:                                              # the loop starts a little short of the stem: a small gap
        S0 = Sj + g.pt(-g.U(0.08, 0.22), g.U(-0.02, 0.08), 0.0)
        T = g.pt(np.minimum(S0[:, 0] - 0.08, g.U(0.20, 0.34)), 0.0)
        loop = curve(S0, T, L, Bm, J)
    else:
        S0 = Sj + g.pt(g.N(-0.02, 0.03), g.N(0.0, 0.03), 0.0)
        T = g.pt(g.U(0.20, 0.38), 0.0)
        loop = curve(S0, T, L, Bm, J)
    if hook:
        H1 = g.pt(xr + g.N(0, 0.03), g.U(0.55, 0.7))
        H2 = g.pt(g.U(0.26, 0.44), g.U(0.94, 1.0))
        H3 = g.pt(g.U(0.0, 0.18), g.U(0.82, 0.96))
        return [loop, curve(Sj, J, H1, H2, H3)]
    End = g.pt(xr + g.U(-0.08, 0.12), 1.0) if big_loop else g.pt(g.U(0.10, 0.64), 1.0)
    Mid = lerp(J, End, 0.5) + g.pt(g.N(0.01, 0.03), 0, 0.0)
    return [loop, curve(Sj, J, Mid, End)]


# (class, share of the class's templates, style function, mean slant, sd of slant)
STYLES = [
    (0, 1.00, digit0, 0.20, 0.25),
    (1, 0.66, lambda g: digit1(g, False, False), 0.28, 0.32),
    (1, 0.17, lambda g: digit1(g, True, False), 0.15, 0.20),
    (1, 0.17, lambda g: digit1(g, True, True), 0.08, 0.15),
    (2, 0.44, lambda g: digit2(g, False), 0.12, 0.22),
    (2, 0.34, lambda g: digit2(g, True), 0.12, 0.22),
    (2, 0.07, lambda g: digit2z(g, False, False), 0.10, 0.20),
    (2, 0.05, lambda g: digit2z(g, True, False), 0.10, 0.20),
    (2, 0.06, lambda g: digit2z(g, False, True), 0.10, 0.20),
    (2, 0.04, lambda g: digit2z(g, True, True), 0.10, 0.20),
    (3, 0.75, lambda g: digit3(g, False), 0.15, 0.22),
    (3, 0.25, lambda g: digit3(g, True), 0.15, 0.22),
    (4, 0.50, lambda g: digit4(g, False), 0.18, 0.22),
    (4, 0.34, lambda g: digit4(g, False, True), 0.18, 0.22),
    (4, 0.16, lambda g: digit4(g, True), 0.15, 0.20),
    (5, 0.65, lambda g: digit5(g, False), 0.18, 0.22),
    (5, 0.35, lambda g: digit5(g, True), 0.18, 0.22),
    (6, 1.00, digit6, 0.15, 0.22),
    (7, 0.40, lambda g: digit7(g, False, False), 0.10, 0.20),
    (7, 0.15, lambda g: digit7(g, False, False, True), 0.10, 0.20),
    (7, 0.20, lambda g: digit7(g, True, False), 0.10, 0.20),
    (7, 0.18, lambda g: digit7(g, False, True), 0.10, 0.20),
    (7, 0.07, lambda g: digit7(g, True, True), 0.10, 0.20),
    (8, 0.50, lambda g: digit8(g, False), 0.20, 0.25),
    (8, 0.50, lambda g: digit8(g, True), 0.20, 0.25),
    (9, 0.64, lambda g: digit9(g, False), 0.15, 0.22),
    (9, 0.14, lambda g: digit9(g, False, True), 0.15, 0.22),
    (9, 0.14, lambda g: digit9(g, False, False, True), 0.15, 0.22),
    (9, 0.08, lambda g: digit9(g, True), 0.15, 0.22),
]


def render(curves, rng, slant_mu, slant_sd):
    """Curves of B drawings -> (B, 28, 28) images framed as the task describes (20-pixel box, centre of mass centred)."""
    pts = np.concatenate([spline(c) for c in curves], axis=1)          # (B, S, 2)
    B = pts.shape[0]
    x = pts[:, :, 0] - pts[:, :, 0].mean(1, keepdims=True)
    y = pts[:, :, 1] - 0.5
    # gentle bending of the whole drawing
    q = rng.normal(0, 1.0, (B, 5)) * np.array([0.20, 0.25, 0.30, 0.20, 0.25])
    x, y = (x + q[:, 0:1] * y * y + q[:, 1:2] * x * y,                       # bow, taper
            y + q[:, 2:3] * x * x + q[:, 3:4] * x * y + q[:, 4:5] * (0.25 - y * y))   # arch, tilt, middle up or down
    # width, slant (top leans right for positive values) and a small rotation
    x = x * np.exp(rng.normal(0, 0.20, (B, 1)))
    x = x - np.clip(rng.normal(slant_mu, slant_sd, (B, 1)), -0.5, 1.1) * y
    rot = np.deg2rad(rng.normal(0, 5.0, (B, 1)))
    x, y = x * np.cos(rot) - y * np.sin(rot), x * np.sin(rot) + y * np.cos(rot)
    # pen radius in pixels of the final frame
    r = 0.5 * np.clip(2.9 * np.exp(rng.normal(0, 0.30, (B, 1))), 1.3, 5.6)
    w = x.max(1, keepdims=True) - x.min(1, keepdims=True)
    h = y.max(1, keepdims=True) - y.min(1, keepdims=True)
    s = (20.0 - 2 * r) / np.maximum(w, h)
    x = (x - 0.5 * (x.max(1, keepdims=True) + x.min(1, keepdims=True))) * s + 13.5
    y = (y - 0.5 * (y.max(1, keepdims=True) + y.min(1, keepdims=True))) * s + 13.5
    grid = np.arange(28.0)
    out = np.empty((B, 28, 28))
    for i in range(0, B, 100):
        dx2 = (grid[None, :, None] - x[i:i + 100, None, :]) ** 2       # (b, 28 cols, S)
        dy2 = (grid[None, :, None] - y[i:i + 100, None, :]) ** 2       # (b, 28 rows, S)
        d2 = (dy2[:, :, None, :] + dx2[:, None, :, :]).min(-1)         # (b, 28, 28)
        out[i:i + 100] = np.clip(r[i:i + 100, :, None] + 0.5 - np.sqrt(d2), 0.0, 1.0)
    # move the centre of mass to the centre of the frame, by whole pixels
    m = out.sum((1, 2))
    cy = (out.sum(2) * grid).sum(1) / m
    cx = (out.sum(1) * grid).sum(1) / m
    sy = np.clip(np.rint(13.5 - cy), -4, 4).astype(int)
    sx = np.clip(np.rint(13.5 - cx), -4, 4).astype(int)
    for i in range(B):
        out[i] = np.roll(out[i], (sy[i], sx[i]), axis=(0, 1))
    return out.astype(np.float32)


def draw_templates(counts=None, seed=SEED):
    """Returns the drawings (sum(counts), 28, 28), ordered by class; counts[c] drawings of digit c."""
    counts = COUNTS if counts is None else counts
    rng = np.random.default_rng(seed)
    out = []
    for cls in range(10):
        styles = [st for st in STYLES if st[0] == cls]
        ns = [int(round(st[1] * counts[cls])) for st in styles]
        ns[0] += counts[cls] - sum(ns)
        for (_, _, fn, mu, sd), n in zip(styles, ns):
            out.append(render(fn(Pen(rng, n)), rng, mu, sd))
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------------------------
# Front end weights (formulas only)
# ----------------------------------------------------------------------------------------------------------------
def gradient_kernels():
    k = 7
    ax = np.arange(k) - (k - 1) / 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    gauss = np.exp(-(xx ** 2 + yy ** 2) / (2 * SIG_G ** 2))
    z = (xx ** 2 * gauss).sum()
    return xx * gauss / z, yy * gauss / z            # derivative along x, along y


def pooling_kernel_1d():
    k = 10
    ax = np.arange(k) - (k - 1) / 2
    g1 = np.exp(-ax ** 2 / (2 * SIG_P ** 2))
    return g1 / g1.sum()


POOL_GAIN = 20.0


def front_end_weights():
    w = {}
    # c1: derivative of a Gaussian along each of 8 directions
    fx, fy = gradient_kernels()
    W1 = np.zeros((NDIR, 1, 7, 7))
    for d in range(NDIR):
        th = 2 * np.pi * d / NDIR
        W1[d, 0] = np.cos(th) * fx + np.sin(th) * fy
    w["c1.W"], w["c1.b"] = W1, np.zeros(NDIR)
    # c2: keep the part of the gradient along a direction that exceeds its part across it
    W2 = np.zeros((NDIR, NDIR, 1, 1))
    for d in range(NDIR):
        W2[d, d] = 1.0
        W2[d, (d + NDIR // 4) % NDIR] = -1.0
        W2[d, (d - NDIR // 4) % NDIR] = -1.0
    w["c2.W"], w["c2.b"] = W2, np.zeros(NDIR)
    # c3: Gaussian pooling at every pixel
    g1 = pooling_kernel_1d()
    W3 = np.zeros((NDIR, NDIR, 10, 10))
    for d in range(NDIR):
        W3[d, d] = POOL_GAIN * np.outer(g1, g1)
    w["c3.W"], w["c3.b"] = W3, np.zeros(NDIR)
    return {a: b.astype(np.float32) for a, b in w.items()}


def corr2(x, k):
    """x (N, H, W) already padded, k (kh, kw): plain sliding-window sum, 'valid' size."""
    kh, kw = k.shape
    ho, wo = x.shape[1] - kh + 1, x.shape[2] - kw + 1
    out = np.zeros((x.shape[0], ho, wo))
    for dy in range(kh):
        for dx in range(kw):
            if k[dy, dx] != 0.0:
                out += k[dy, dx] * x[:, dy:dy + ho, dx:dx + wo]
    return out


def coarse_kernel_1d():
    """Extra blur that takes the pooled map from scale SIG_P to scale SIG_C."""
    k = 13
    ax = np.arange(k) - (k - 1) / 2
    g1 = np.exp(-ax ** 2 / (2 * (SIG_C ** 2 - SIG_P ** 2)))
    return g1 / g1.sum()


def unit(v):
    """Zero mean, unit variance over each row's own entries (what the norm layer does, eps included)."""
    mu = v.mean(1, keepdims=True)
    return (v - mu) / np.sqrt(((v - mu) ** 2).mean(1, keepdims=True) + 1e-5)


def front_end(images):
    """images (N, 28, 28) -> (fine maps (N, NDIR, FULL, FULL), coarse maps (N, NDIR, 7, 7)), both normalised:
    the same arithmetic as layers c1 .. feat and cc .. cfeat of the network."""
    x = np.pad(images.astype(np.float64), ((0, 0), (3, 3), (3, 3)))
    fx, fy = gradient_kernels()
    gx, gy = corr2(x, fx), corr2(x, fy)                                        # (N, 28, 28)
    d = [np.maximum(np.cos(2 * np.pi * i / NDIR) * gx + np.sin(2 * np.pi * i / NDIR) * gy, 0) for i in range(NDIR)]
    q = NDIR // 4
    s = np.stack([np.maximum(d[i] - d[(i + q) % NDIR] - d[(i - q) % NDIR], 0) for i in range(NDIR)], axis=1)   # (N, 8, 28, 28)
    g1 = pooling_kernel_1d()
    sp = np.pad(s, ((0, 0), (0, 0), (5, 5), (5, 5)))                           # (N, 8, 38, 38)
    rows = sum(g1[i] * sp[:, :, i:i + FULL, :] for i in range(10))             # (N, 8, 29, 38)
    f = POOL_GAIN * sum(g1[i] * rows[:, :, :, i:i + FULL] for i in range(10))  # (N, 8, 29, 29)
    n = len(f)
    fine = unit(f.reshape(n, -1)).reshape(f.shape)
    h1 = coarse_kernel_1d()
    fp = np.pad(f, ((0, 0), (0, 0), (4, 4), (4, 4)))                           # (N, 8, 37, 37)
    rows = sum(h1[i] * fp[:, :, i:i + 25:4, :] for i in range(13))             # (N, 8, 7, 37)
    c = sum(h1[i] * rows[:, :, :, i:i + 25:4] for i in range(13))              # (N, 8, 7, 7)
    coarse = unit(c.reshape(n, -1)).reshape(c.shape)
    return fine, coarse


# ----------------------------------------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------------------------------------
def build(counts=None, seed=SEED):
    counts = COUNTS if counts is None else counts
    w = front_end_weights()
    T = sum(counts)
    L = [
        {"name": "c1", "op": "conv", "inputs": ["image"], "out": NDIR, "kernel": [7, 7], "stride": 1, "padding": 3},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "c2", "op": "conv", "inputs": ["r1"], "out": NDIR, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "r2", "op": "relu", "inputs": ["c2"]},
        {"name": "c3", "op": "conv", "inputs": ["r2"], "out": NDIR, "kernel": [10, 10], "stride": 1, "padding": 5},
        # fine map, normalised as a whole
        {"name": "flat", "op": "reshape", "inputs": ["c3"], "shape": [1, NDIR * FULL, FULL]},
        {"name": "nrm", "op": "norm", "inputs": ["flat"]},
        {"name": "feat", "op": "reshape", "inputs": ["nrm"], "shape": [NDIR, FULL, FULL]},
        # coarse map: more blur, one sample every 4 pixels, normalised as a whole
        {"name": "cc", "op": "conv", "inputs": ["c3"], "out": NDIR, "kernel": [13, 13], "stride": 4, "padding": 4},
        {"name": "cflat", "op": "reshape", "inputs": ["cc"], "shape": [1, NDIR * 7, 7]},
        {"name": "cfeat", "op": "norm", "inputs": ["cflat"]},
    ]
    h1 = coarse_kernel_1d()
    Wc = np.zeros((NDIR, NDIR, 13, 13))
    for d in range(NDIR):
        Wc[d, d] = np.outer(h1, h1)
    w["cc.W"], w["cc.b"] = Wc.astype(np.float32), np.zeros(NDIR, dtype=np.float32)

    # gather: output channel (c, i, j) at output position (u, v) = feat[c, 2 (7 qi + i) + u, 2 (7 qj + j) + v],
    # so the 3 x 3 output positions are the quadrant's taps read one pixel up/down/left/right of their home place
    nfeat = NDIR * QUAD * QUAD
    kg = FULL - 2
    for qi in range(2):
        for qj in range(2):
            q = 2 * qi + qj
            Wg = np.zeros((nfeat, NDIR, kg, kg), dtype=np.float32)
            for c in range(NDIR):
                for i in range(QUAD):
                    for j in range(QUAD):
                        Wg[c * QUAD * QUAD + i * QUAD + j, c, 2 * (QUAD * qi + i), 2 * (QUAD * qj + j)] = 1.0
            w[f"gather{q}.W"], w[f"gather{q}.b"] = Wg, np.zeros(nfeat, dtype=np.float32)
            L.append({"name": f"gather{q}", "op": "conv", "inputs": ["feat"], "out": nfeat, "kernel": [kg, kg], "stride": 1, "padding": 0})

    # templates: every drawing through the same front end, each normalised by itself alone (its own norm layer)
    images = draw_templates(counts, seed)
    fine, coarse = [], []
    for i in range(0, T, 500):
        f, c = front_end(images[i:i + 500])
        fine.append(f[:, :, 1:28:2, 1:28:2].astype(np.float32))      # the 14 x 14 taps
        coarse.append(c.astype(np.float32))
    fine = np.concatenate(fine).astype(np.float64)                   # (T, 8, 14, 14)
    coarse = np.concatenate(coarse).astype(np.float64).reshape(T, -1)   # (T, 392)
    ntap = fine[0].size
    ncoarse = coarse.shape[1]

    # A quadrant's match is 2 <x, t> - |t|^2 (per tap): the squared distance |x - t|^2 up to the image's own energy.
    #   whole-template score of a class: max over templates of (sum of its four quadrant matches + coarse match)
    #   image-to-class score of a class: sum over quadrants of the best match by any of the class's templates
    # The class score is the sum of the two.
    start = 0
    finals = []
    for cls in range(10):
        k = counts[cls]
        tf, tc = fine[start:start + k], coarse[start:start + k]
        start += k
        terms, tops = [], []
        for qi in range(2):
            for qj in range(2):
                q = 2 * qi + qj
                part = tf[:, :, QUAD * qi:QUAD * (qi + 1), QUAD * qj:QUAD * (qj + 1)].reshape(k, nfeat)
                w[f"part{cls}_{q}.W"] = (2 * part / ntap).reshape(k, nfeat, 1, 1).astype(np.float32)
                w[f"part{cls}_{q}.b"] = (-(part ** 2).sum(1) / ntap).astype(np.float32)
                L.append({"name": f"part{cls}_{q}", "op": "conv", "inputs": [f"gather{q}"], "out": k, "kernel": [1, 1], "stride": 1, "padding": 0})
                L.append({"name": f"best{cls}_{q}", "op": "maxpool", "inputs": [f"part{cls}_{q}"], "size": [3, 3], "stride": [3, 3]})
                terms.append(f"best{cls}_{q}")
                L.append({"name": f"qrow{cls}_{q}", "op": "reshape", "inputs": [f"best{cls}_{q}"], "shape": [1, 1, k]})
                L.append({"name": f"qtop{cls}_{q}", "op": "maxpool", "inputs": [f"qrow{cls}_{q}"], "size": [1, k], "stride": [1, k]})
                tops.append(f"qtop{cls}_{q}")
        w[f"coarse{cls}.W"] = (COARSE_WEIGHT * 2 * tc / ncoarse).astype(np.float32)
        w[f"coarse{cls}.b"] = (-COARSE_WEIGHT * (tc ** 2).sum(1) / ncoarse).astype(np.float32)
        L.append({"name": f"coarse{cls}", "op": "dense", "inputs": ["cfeat"], "out": k})
        L.append({"name": f"coarse{cls}r", "op": "reshape", "inputs": [f"coarse{cls}"], "shape": [k, 1, 1]})
        terms.append(f"coarse{cls}r")
        L.append({"name": f"match{cls}", "op": "add", "inputs": terms})
        L.append({"name": f"row{cls}", "op": "reshape", "inputs": [f"match{cls}"], "shape": [1, 1, k]})
        L.append({"name": f"top{cls}", "op": "maxpool", "inputs": [f"row{cls}"], "size": [1, k], "stride": [1, k]})
        L.append({"name": f"both{cls}", "op": "add", "inputs": [f"top{cls}"] + tops})
        onehot = np.zeros((10, 1), dtype=np.float32)
        onehot[cls, 0] = 1.0
        w[f"put{cls}.W"], w[f"put{cls}.b"] = onehot, np.zeros(10, dtype=np.float32)
        L.append({"name": f"put{cls}", "op": "dense", "inputs": [f"both{cls}"], "out": 10})
        finals.append(f"put{cls}")
    L.append({"name": "scores", "op": "add", "inputs": finals})
    return {"layers": L}, w


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    net, weights = build()
    with open(out_dir + "/network.json", "w") as f:
        json.dump(net, f, indent=1)
    np.savez(out_dir + "/weights.npz", **weights)
