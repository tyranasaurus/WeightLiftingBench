#!/usr/bin/env python3
"""build.py -- writes network.json and weights.npz for a hand-designed 32x32 photo classifier.

Usage: python3 build.py OUT_DIR          (reads no files; every number comes from this source)

How the weights are produced (no training, no data):
  1. A fixed front end made from formulas.  Six feature blocks:
       A   opponent colours (brightness, red-green, blue-yellow) on an 8x8 grid, absolute
       B   brightness layout with the image's own mean and contrast normalised away
       C   chroma layout divided by the image's own chroma strength (with a floor)
       S   figure against ground: how far each cell's colour is from the colour of the image border
       E1  oriented Gaussian-derivative edge energy (six orientations), fine scale, 7x7 cells;
       E2  the same at half resolution, 4x4 cells; both normalised per cell across orientations
  2. A scene renderer written from general knowledge of the ten classes: backgrounds (sky, sea, grass, road,
     rooms, foliage ...) and objects drawn from ellipses, capsules and polygons (an aeroplane as a fuselage with
     swept wings and a tail fin, a car as a body with a cabin and two dark wheels, a horse as a heavy body with
     neck, long head and four legs ...), with hand-typed colours, sizes and proportions and a fixed random seed.
  3. Every rendered scene is passed through the front end and becomes ONE template: one row of the dense layers,
     with its own bias -1/2 |t|^2, so that row . x + bias ranks templates by distance to the image.  No statistic
     is taken across scenes.  The score of a class is the similarity of its best-matching templates: the maximum
     within each of 24 groups of 500 templates, then the mean of the 24 maxima.
"""
import json, os, sys
import numpy as np

S = 32
SEED = 20261120
RNG = np.random.default_rng(SEED)
_g = np.arange(S, dtype=np.float64) + 0.5
YY = _g[None, :, None]
XX = _g[None, None, :]
ONES = np.ones((1, S, S))


# ----------------------------------------------------------------------------------------------------------
# small helpers (everything is batched over n scenes)
# ----------------------------------------------------------------------------------------------------------
def A(x):
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1, 1) if x.ndim == 1 else x


def A4(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 1, 1, 1)


def U(n, a, b):
    return RNG.uniform(a, b, n)


def Nn(n, s=1.0):
    return RNG.standard_normal(n) * s


def B(n, p):
    return (RNG.random(n) < p).astype(np.float64)


def SG(n):
    return np.where(RNG.random(n) < 0.5, -1.0, 1.0)


def _upmat(k):
    pos = (np.arange(S) + 0.5) * k / S - 0.5
    i0 = np.floor(pos).astype(int)
    f = pos - i0
    M = np.zeros((S, k))
    for r in range(S):
        M[r, min(max(i0[r], 0), k - 1)] += 1 - f[r]
        M[r, min(max(i0[r] + 1, 0), k - 1)] += f[r]
    return M


UP = {k: _upmat(k) for k in (1, 2, 3, 4, 6, 8, 12, 16, 32)}


def noise(n, ky, kx=None):
    """smooth random field (n, S, S): white noise on a ky x kx grid, bilinearly enlarged"""
    kx = ky if kx is None else kx
    return UP[ky] @ RNG.standard_normal((n, ky, kx)) @ UP[kx].T


def noise1(n, k):
    """smooth random profile along x: (n, 1, S)"""
    return RNG.standard_normal((n, 1, k)) @ UP[k].T


def fbm(n, ks=(4, 8, 16, 32), ws=(1.0, 0.7, 0.5, 0.35)):
    t = 0.0
    for k, w in zip(ks, ws):
        t = t + w * noise(n, k)
    return t / np.sqrt(sum(w * w for w in ws))


def C(n, base, v=0.10, c=0.03):
    """a colour with brightness jitter v and per-channel jitter c: (n, 3)"""
    base = np.asarray(base, dtype=np.float64)
    return np.clip(base[None] * (1 + v * RNG.standard_normal((n, 1))) + c * RNG.standard_normal((n, 3)), 0, 1)


def PAL(n, table, v=0.10, c=0.03):
    """pick one colour per scene from a weighted palette, with jitter"""
    w = np.array([t[0] for t in table], dtype=np.float64)
    idx = RNG.choice(len(table), size=n, p=w / w.sum())
    base = np.array([t[1] for t in table], dtype=np.float64)[idx]
    return np.clip(base * (1 + v * RNG.standard_normal((n, 1))) + c * RNG.standard_normal((n, 3)), 0, 1)


def shade(col, f):
    f = np.asarray(f, dtype=np.float64)
    return np.clip(col * (f[:, None] if f.ndim == 1 else f), 0, 1)


def lerp(c1, c2, t):
    t = np.asarray(t, dtype=np.float64)
    return c1 + (c2 - c1) * (t[:, None] if t.ndim == 1 else t)


def mix(n, table):
    """split the batch among several generator functions fn(m) -> array or tuple of arrays"""
    w = np.array([t[0] for t in table], dtype=np.float64)
    idx = RNG.choice(len(table), size=n, p=w / w.sum())
    out = None
    for i, t in enumerate(table):
        sel = np.where(idx == i)[0]
        if len(sel) == 0:
            continue
        r = t[1](len(sel))
        r = r if isinstance(r, tuple) else (r,)
        if out is None:
            out = [np.zeros((n,) + a.shape[1:]) for a in r]
        for o, a in zip(out, r):
            o[sel] = a
    return tuple(out) if len(out) > 1 else out[0]


class Frame:
    """object coordinates: pixel = centre + Rot(th) @ (flip * s * u, s * sq * v)"""

    def __init__(self, cx, cy, s=1.0, th=0.0, flip=1.0, sq=1.0):
        cx, cy, s, th, flip, sq = (A(v) for v in (cx, cy, s, th, flip, sq))
        dx = XX - cx
        dy = YY - cy
        c, sn = np.cos(th), np.sin(th)
        self.U = flip * (c * dx + sn * dy) / s
        self.V = (-sn * dx + c * dy) / (s * sq)
        self.px = s * np.sqrt(sq)


PIX = Frame(0.0, 0.0)


def _aa(F, sdf, soft=1.0):
    return np.clip(0.5 - sdf * (F.px / soft), 0.0, 1.0)


def _rot(du, dv, ang):
    ang = A(ang)
    if np.any(ang != 0):
        c, s = np.cos(ang), np.sin(ang)
        return c * du + s * dv, -s * du + c * dv
    return du, dv


def ell(F, x, y, rx, ry, ang=0.0, soft=1.0):
    x, y, rx, ry = (A(v) for v in (x, y, rx, ry))
    du, dv = _rot(F.U - x, F.V - y, ang)
    q = np.sqrt((du / rx) ** 2 + (dv / ry) ** 2)
    return _aa(F, (q - 1.0) * np.minimum(rx, ry), soft)


def cap(F, x0, y0, x1, y1, r, r1=None, soft=1.0):
    x0, y0, x1, y1, r = (A(v) for v in (x0, y0, x1, y1, r))
    ex, ey = x1 - x0, y1 - y0
    du, dv = F.U - x0, F.V - y0
    t = np.clip((du * ex + dv * ey) / (ex * ex + ey * ey + 1e-6), 0, 1)
    d = np.sqrt((du - t * ex) ** 2 + (dv - t * ey) ** 2)
    rr = r if r1 is None else r + t * (A(r1) - r)
    return _aa(F, d - rr, soft)


def poly(F, pts, soft=1.0):
    """convex polygon, vertices in either order"""
    P = [(A(x), A(y)) for x, y in pts]
    k = len(P)
    area = 0.0
    for i in range(k):
        (x0, y0), (x1, y1) = P[i], P[(i + 1) % k]
        area = area + (x0 * y1 - x1 * y0)
    sgn = np.where(area >= 0, 1.0, -1.0)
    sdf = None
    for i in range(k):
        (x0, y0), (x1, y1) = P[i], P[(i + 1) % k]
        ex, ey = x1 - x0, y1 - y0
        L = np.sqrt(ex * ex + ey * ey) + 1e-6
        d = sgn * ((F.U - x0) * ey - (F.V - y0) * ex) / L
        sdf = d if sdf is None else np.maximum(sdf, d)
    return _aa(F, sdf, soft)


def rbox(F, x, y, hw, hh, r=0.0, ang=0.0, soft=1.0):
    x, y, hw, hh, r = (A(v) for v in (x, y, hw, hh, r))
    du, dv = _rot(F.U - x, F.V - y, ang)
    qx = np.abs(du) - hw + r
    qy = np.abs(dv) - hh + r
    sdf = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r
    return _aa(F, sdf, soft)


def paint(img, m, col, a=None, tex=None):
    """alpha-blend colour col (n,3) into img (n,3,S,S) through mask m (n,S,S); a: per-scene opacity; tex: shading"""
    col = np.asarray(col, dtype=np.float64)
    c = col[:, :, None, None] if col.ndim == 2 else col
    if tex is not None:
        c = np.clip(c * (1 + tex[:, None]), 0, 1)
    if a is not None:
        m = m * A(a)
    img += m[:, None] * (c - img)


def blend(img, other, m):
    return img * (1 - m[:, None]) + other * m[:, None]


def furtex(n, lo=0.05, hi=0.2):
    return A(U(n, lo, hi)) * (0.6 * noise(n, 8) + 0.6 * noise(n, 16) + 0.5 * noise(n, 32))


def darkc(n):
    return C(n, (0.07, 0.07, 0.08), 0.3, 0.01)


def whitec(n):
    return C(n, (0.93, 0.92, 0.90), 0.05, 0.015)


def glassc(n):
    return PAL(n, [(3, (0.12, 0.15, 0.18)), (1.5, (0.30, 0.36, 0.42)), (1, (0.55, 0.65, 0.72)), (1, (0.05, 0.05, 0.06))], 0.15, 0.02)


def blur3(img):
    p = np.pad(img, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    v = 0.25 * p[:, :, :-2] + 0.5 * p[:, :, 1:-1] + 0.25 * p[:, :, 2:]
    return 0.25 * v[:, :, :, :-2] + 0.5 * v[:, :, :, 1:-1] + 0.25 * v[:, :, :, 2:]


def finish(img):
    """camera effects, each image on its own: soft focus, exposure, contrast, saturation, colour cast, noise"""
    n = len(img)
    img = img + A4(U(n, 0.0, 0.9)) * (blur3(img) - img)
    m = img.mean(axis=(1, 2, 3), keepdims=True)
    img = m + (img - m) * A4(np.exp(Nn(n, 0.15)))
    g = img.mean(axis=1, keepdims=True)
    img = g + (img - g) * A4(U(n, 0.4, 1.05))
    img = np.clip(img, 0, 1) ** A4(np.exp(Nn(n, 0.15)))
    img = img * A4(np.exp(Nn(n, 0.20))) * np.exp(RNG.standard_normal((n, 3, 1, 1)) * 0.05)
    img = img + A4(U(n, 0.0, 0.02)) * RNG.standard_normal(img.shape)
    return np.clip(img, 0, 1)


# ----------------------------------------------------------------------------------------------------------
# backgrounds: every function returns (image batch (n,3,S,S), a nominal horizon / ground row (n,))
# ----------------------------------------------------------------------------------------------------------
SKY = [(30, (0.30, 0.50, 0.85)), (20, (0.45, 0.62, 0.88)), (15, (0.62, 0.74, 0.90)), (15, (0.75, 0.78, 0.82)),
       (8, (0.55, 0.58, 0.63)), (7, (0.93, 0.94, 0.96)), (3, (0.15, 0.25, 0.55)), (2, (0.85, 0.60, 0.40))]
GROUND = {
    "grass": [(4, (0.33, 0.48, 0.17)), (3, (0.42, 0.55, 0.22)), (2, (0.22, 0.36, 0.12)), (2, (0.50, 0.58, 0.28)),
              (3, (0.38, 0.42, 0.22)), (2, (0.30, 0.34, 0.20))],
    "dry": [(1, (0.68, 0.62, 0.38)), (1, (0.58, 0.52, 0.30)), (1, (0.75, 0.70, 0.50))],
    "dirt": [(1, (0.55, 0.44, 0.32)), (1, (0.45, 0.36, 0.26)), (1, (0.68, 0.58, 0.45))],
    "road": [(3, (0.42, 0.42, 0.44)), (2, (0.55, 0.55, 0.55)), (2, (0.30, 0.30, 0.32)), (1, (0.66, 0.64, 0.60))],
    "snow": [(1, (0.90, 0.92, 0.95)), (1, (0.80, 0.84, 0.90))],
    "water": [(4, (0.22, 0.38, 0.58)), (3, (0.40, 0.50, 0.58)), (2, (0.25, 0.42, 0.42)), (2, (0.12, 0.20, 0.32)),
              (2, (0.50, 0.65, 0.78)), (1, (0.55, 0.58, 0.60))],
    "sand": [(1, (0.80, 0.72, 0.56)), (1, (0.70, 0.62, 0.48))],
}
TREES = [(4, (0.16, 0.28, 0.10)), (3, (0.25, 0.38, 0.15)), (2, (0.10, 0.16, 0.08)), (1.5, (0.35, 0.42, 0.20)),
         (1, (0.40, 0.32, 0.18)), (1.5, (0.35, 0.45, 0.50)), (1, (0.45, 0.40, 0.35)), (2.5, (0.22, 0.26, 0.16)),
         (1.5, (0.30, 0.30, 0.22))]
FOLI = [(4, (0.30, 0.45, 0.18)), (3, (0.45, 0.55, 0.25)), (2, (0.18, 0.28, 0.12)), (2, (0.55, 0.50, 0.30)),
        (2, (0.40, 0.33, 0.22)), (1.5, (0.60, 0.65, 0.45)), (1, (0.50, 0.50, 0.48)), (1, (0.70, 0.78, 0.85)),
        (1, (0.25, 0.22, 0.18)), (3, (0.35, 0.38, 0.24)), (2, (0.28, 0.28, 0.20))]
TERR = [(3, (0.30, 0.40, 0.20)), (2, (0.45, 0.42, 0.30)), (2, (0.20, 0.30, 0.15)), (1, (0.55, 0.50, 0.40)),
        (1, (0.20, 0.30, 0.45)), (1, (0.40, 0.40, 0.40))]
URBAN = [(3, (0.60, 0.58, 0.55)), (2, (0.75, 0.72, 0.66)), (2, (0.45, 0.44, 0.45)), (1.5, (0.55, 0.35, 0.28)),
         (1.5, (0.85, 0.85, 0.85)), (1, (0.25, 0.27, 0.30)), (2, (0.25, 0.36, 0.16)), (1, (0.35, 0.45, 0.20)),
         (1, (0.55, 0.68, 0.85))]
INDOOR = [(3, (0.85, 0.82, 0.75)), (2, (0.70, 0.62, 0.50)), (2, (0.50, 0.38, 0.26)), (2, (0.60, 0.60, 0.62)),
          (1.5, (0.35, 0.33, 0.33)), (1, (0.15, 0.13, 0.12)), (1, (0.55, 0.25, 0.20)), (1, (0.30, 0.40, 0.55)),
          (1, (0.40, 0.50, 0.38)), (1, (0.93, 0.93, 0.93)), (1, (0.75, 0.55, 0.35)), (0.7, (0.80, 0.70, 0.75)),
          (0.7, (0.25, 0.22, 0.30))]
CLOSE = {
    "soil": [(2, (0.42, 0.33, 0.24)), (2, (0.30, 0.24, 0.18)), (1, (0.55, 0.45, 0.33)), (1, (0.38, 0.36, 0.30))],
    "leaf": [(3, (0.25, 0.42, 0.14)), (2, (0.38, 0.55, 0.18)), (1.5, (0.15, 0.28, 0.10)), (1, (0.50, 0.60, 0.25))],
    "rock": [(2, (0.50, 0.50, 0.48)), (1, (0.36, 0.36, 0.36)), (1, (0.62, 0.58, 0.52))],
    "litter": [(2, (0.40, 0.30, 0.18)), (1, (0.50, 0.36, 0.20)), (1, (0.28, 0.22, 0.15)), (1, (0.35, 0.38, 0.22))],
    "pond": [(2, (0.12, 0.18, 0.12)), (1, (0.20, 0.25, 0.20)), (1, (0.10, 0.12, 0.14)), (1, (0.25, 0.30, 0.22))],
    "grass": GROUND["grass"],
    "sand": GROUND["sand"] + GROUND["dirt"],
}


def vgrad(top, bot):
    t = ((YY - 0.5) / (S - 1.0))[:, None] * np.ones((1, 1, 1, S))
    return top[:, :, None, None] + (bot - top)[:, :, None, None] * t


def tex_fill(n, col, lo=0.03, hi=0.10, ks=(4, 8, 16, 32), ws=(1.0, 0.7, 0.5, 0.35)):
    return np.clip(col[:, :, None, None] * (1 + A4(U(n, lo, hi)) * fbm(n, ks, ws)[:, None]), 0, 1)


def bg_sky(n, clouds=0.4):
    top = PAL(n, SKY, 0.08, 0.03)
    bot = lerp(top, np.array([0.85, 0.88, 0.92]), U(n, 0.1, 0.7))
    img = vgrad(top, bot)
    cm = np.clip(noise(n, 4) * 0.8 + noise(n, 8) * 0.4 - A(U(n, 0.2, 1.0)), 0, 1) * A(B(n, clouds))
    paint(img, np.clip(cm * 1.5, 0, 1), C(n, (0.95, 0.95, 0.96), 0.04, 0.01))
    img = img * (1 + 0.015 * noise(n, 16)[:, None])
    return np.clip(img, 0, 1), np.zeros(n)


def ground_tex(n, kind):
    col = PAL(n, GROUND[kind], 0.10, 0.03)[:, :, None, None]
    if kind == "water":
        t = (noise(n, 16, 4) + 0.7 * noise(n, 32, 8) + 0.5 * noise(n, 4)) / 1.3
        img = col * (1 + A4(U(n, 0.03, 0.14)) * t[:, None])
    elif kind in ("grass", "dry"):
        col2 = PAL(n, GROUND[kind], 0.12, 0.04)[:, :, None, None]
        a = np.clip(0.5 + noise(n, 4) * A(U(n, 0.0, 1.0)), 0, 1)[:, None]
        img = (col * (1 - a) + col2 * a) * (1 + A4(U(n, 0.08, 0.25)) * fbm(n, (4, 8, 16, 32), (0.8, 0.8, 0.8, 0.8))[:, None])
    elif kind == "road":
        img = col * (1 + A4(U(n, 0.02, 0.07)) * fbm(n, (4, 16, 32), (1.0, 0.6, 0.6))[:, None])
    elif kind == "snow":
        img = col * (1 + A4(U(n, 0.01, 0.05)) * fbm(n, (4, 8, 16), (1.0, 0.7, 0.5))[:, None])
    else:
        img = col * (1 + A4(U(n, 0.04, 0.14)) * fbm(n)[:, None])
    img = img * (1 + A4(Nn(n, 0.10)) * ((YY - 16) / 16)[:, None])
    return np.clip(img, 0, 1)


def gmix(kinds):
    return lambda n: mix(n, [(w, (lambda m, k=k: ground_tex(m, k))) for w, k in kinds])


def bg_land(n, kinds, hlo, hhi, tree_p=0.5):
    """sky above a horizon, optional band of trees / hills / far shore on the horizon, ground below"""
    hy = U(n, hlo, hhi)
    img, _ = bg_sky(n)
    th = np.where(RNG.random(n) < 0.3, U(n, 6, 30), U(n, 1.5, 9)) * B(n, tree_p)
    rough = noise1(n, 8) * A(U(n, 0.3, 2.0)) + noise1(n, 3) * A(U(n, 0.0, 2.5))
    m = np.clip(YY - (A(hy) - A(th) + rough) + 0.5, 0, 1) * (A(th) > 0)
    trees = np.clip(PAL(n, TREES, 0.15, 0.03)[:, :, None, None] * (1 + A4(U(n, 0.1, 0.35)) * fbm(n, (6, 12, 32), (1.0, 1.0, 0.8))[:, None]), 0, 1)
    img = blend(img, trees, m)
    haze = np.where(RNG.random(n) < 0.4, U(n, 1.5, 5.0), 1.0)
    img = blend(img, gmix(kinds)(n), np.clip((YY - A(hy)) / A(haze) + 0.5, 0, 1) * ONES)
    return img, hy


BRUSH = [(3, (0.30, 0.36, 0.18)), (3, (0.42, 0.40, 0.25)), (2, (0.20, 0.24, 0.12)), (2, (0.55, 0.50, 0.32)),
         (2, (0.35, 0.28, 0.18)), (1, (0.14, 0.14, 0.10)), (1, (0.62, 0.60, 0.48))]


def clutter(img, n, ylo, yhi, k, pal, p):
    """random rectangles: buildings, windows, doors, signs, furniture, things lying about; p: chance of each, per scene"""
    for _ in range(k):
        paint(img, rbox(PIX, U(n, 0, 32), ylo + U(n, 0, 1) * (yhi - ylo), U(n, 1.5, 7), U(n, 1.5, 8), 0.3, Nn(n, 0.05)),
              PAL(n, pal, 0.15, 0.04), B(n, p))


def bg_brush(n):
    """undergrowth, tall grass, scrub: blotchy greens, tans and browns with stems"""
    c1, c2, c3 = (PAL(n, BRUSH, 0.15, 0.04)[:, :, None, None] for _ in range(3))
    a = np.clip(0.5 + noise(n, 6) * A(U(n, 0.5, 1.5)), 0, 1)[:, None]
    b = np.clip(noise(n, 12) * A(U(n, 0.5, 1.5)), 0, 1)[:, None]
    img = c1 * (1 - a) + c2 * a
    img = img * (1 - b) + c3 * b
    img = img * (1 + A4(U(n, 0.10, 0.30)) * fbm(n, (8, 16, 32), (0.8, 0.9, 0.9))[:, None] + A4(U(n, 0.0, 0.2)) * noise(n, 4, 16)[:, None])
    return np.clip(img, 0, 1), U(n, 20, 28)


def bg_bokeh(n, pal=FOLI, lo=0.02, hi=0.07):
    """out-of-focus backdrop: two or three colours merging smoothly"""
    c1, c2, c3 = (PAL(n, pal, 0.12, 0.04)[:, :, None, None] for _ in range(3))
    a = np.clip(0.5 + noise(n, 3) * A(U(n, 0.3, 1.2)), 0, 1)[:, None]
    b = np.clip(noise(n, 4) * A(U(n, 0.2, 1.0)), 0, 1)[:, None]
    img = c1 * (1 - a) + c2 * a
    img = img * (1 - b) + c3 * b
    img = img * (1 + A4(U(n, lo, hi)) * fbm(n, (8, 16, 32), (1.0, 0.7, 0.5))[:, None])
    return np.clip(img, 0, 1), U(n, 20, 28)


def bg_close(n, kind, lo=0.08, hi=0.25):
    """close-up ground: soil, leaves, rock, leaf litter, pond"""
    c1 = PAL(n, CLOSE[kind], 0.12, 0.04)[:, :, None, None]
    c2 = PAL(n, CLOSE[kind], 0.20, 0.05)[:, :, None, None]
    a = np.clip(0.5 + noise(n, 4) * A(U(n, 0.2, 1.0)), 0, 1)[:, None]
    img = (c1 * (1 - a) + c2 * a) * (1 + A4(U(n, lo, hi)) * fbm(n, (6, 12, 16, 32), (0.8, 0.8, 0.9, 0.9))[:, None])
    return np.clip(img, 0, 1), U(n, 20, 28)


def bg_forest(n):
    img, _ = bg_bokeh(n, [(4, (0.14, 0.24, 0.10)), (3, (0.22, 0.32, 0.14)), (2, (0.08, 0.12, 0.07)), (2, (0.30, 0.26, 0.16)),
                          (1, (0.40, 0.48, 0.25)), (1, (0.55, 0.60, 0.50))], 0.08, 0.20)
    for _ in range(3):
        x = U(n, 0, 32)
        paint(img, cap(PIX, x, -2, x + Nn(n, 2), U(n, 14, 34), U(n, 0.6, 2.2)),
              PAL(n, [(2, (0.22, 0.17, 0.12)), (1, (0.35, 0.30, 0.25)), (1, (0.12, 0.10, 0.08))]), B(n, 0.6))
    hy = U(n, 16, 26)
    g = gmix([(3, "grass"), (2, "dry"), (2, "dirt"), (0.7, "snow")])(n)
    img = blend(img, g, np.clip((YY - A(hy) + noise1(n, 6) * 1.5) / 3 + 0.5, 0, 1) * ONES)
    return img, hy


def bg_indoor(n):
    """a room: wall above, floor / sofa / bedding below (often patterned), furniture and things lying about"""
    c1, c2 = (PAL(n, INDOOR, 0.12, 0.04) for _ in range(2))
    img = tex_fill(n, c1, 0.02, 0.08, (4, 8, 16), (1.0, 0.7, 0.5))
    y0 = U(n, 6, 28)
    m = np.clip(YY - (A(y0) + A(Nn(n, 0.25)) * (XX - 16)) + 0.5, 0, 1)
    pat = np.where(RNG.random(n) < 0.4, U(n, 0.12, 0.30), U(n, 0.03, 0.12))
    lower = np.clip(c2[:, :, None, None] * (1 + A4(pat) * fbm(n, (6, 12, 16), (1.0, 0.8, 0.6))[:, None]), 0, 1)
    img = blend(img, lower, m)
    lev = U(n, 0.15, 0.7)                                   # from a tidy room to a cluttered one
    for _ in range(3):
        F0 = Frame(U(n, 0, 32), U(n, 0, 32), 1.0, Nn(n, 0.3))
        paint(img, rbox(F0, 0, 0, U(n, 2, 10), U(n, 2, 10), 1.0), PAL(n, INDOOR, 0.15, 0.04), B(n, lev))
    img = img * (1 + 0.15 * noise(n, 3)[:, None])
    return np.clip(img, 0, 1), y0


def bg_street(n, sky=1.2):
    hy = U(n, 8, 20)
    up, _ = mix(n, [(sky, bg_sky), (3.5, lambda m: (tex_fill(m, PAL(m, URBAN, 0.12, 0.03), 0.03, 0.12), np.zeros(m))), (2.5, bg_bokeh)])
    clutter(up, n, 0.0, hy, 7, URBAN, U(n, 0.15, 0.8))      # from a nearly bare wall to a busy street
    for _ in range(2):          # other vehicles parked along the far side of the road
        px, pw, ph = U(n, 0, 32), U(n, 3, 6), U(n, 1.3, 2.5)
        pa = B(n, 0.3)
        paint(up, rbox(PIX, px, hy - ph, pw, ph, 0.8), PAL(n, URBAN + [(4, (0.7, 0.1, 0.1)), (4, (0.9, 0.9, 0.9)), (3, (0.1, 0.1, 0.12)), (3, (0.15, 0.25, 0.55))], 0.12, 0.03), pa)
        paint(up, rbox(PIX, px, hy - 0.5, pw, 0.6), darkc(n), pa)
    road = gmix([(6, "road"), (1, "dirt"), (1, "grass"), (0.5, "snow")])(n)
    img = blend(up, road, np.clip(YY - (A(hy) + A(Nn(n, 0.08)) * (XX - 16)) + 0.5, 0, 1))
    ly = hy + U(n, 0.5, 3)
    paint(img, rbox(PIX, 16, ly, 20, U(n, 0.3, 0.8)), C(n, (0.8, 0.8, 0.78), 0.1, 0.02), B(n, 0.3))
    return img, hy


def bg_plain(n):
    c = PAL(n, [(6, (0.97, 0.97, 0.97)), (1.5, (0.04, 0.04, 0.04)), (1.5, (0.80, 0.80, 0.82)), (1, (0.55, 0.55, 0.58)),
                (0.5, (0.90, 0.88, 0.80))], 0.03, 0.01)
    img = np.ones((n, 3, S, S)) * c[:, :, None, None]
    img = img * (1 + A4(Nn(n, 0.06)) * ((YY - 16) / 16)[:, None])
    return np.clip(img, 0, 1), U(n, 22, 28)


def bg_water(n):
    return ground_tex(n, "water"), U(n, 0, 6)


def bg_sea(n, hlo=7, hhi=19):
    return bg_land(n, [(1, "water")], hlo, hhi, 0.35)


def bg_arena(n):
    hy = U(n, 10, 21)
    up, _ = mix(n, [(3, bg_bokeh), (2, lambda m: (tex_fill(m, PAL(m, URBAN, 0.12, 0.03), 0.03, 0.12), np.zeros(m))), (1.5, bg_sky),
                    (1, lambda m: (tex_fill(m, PAL(m, [(1, (0.15, 0.12, 0.10)), (1, (0.30, 0.25, 0.20))]), 0.05, 0.15), np.zeros(m)))])
    fy = hy - U(n, 0.5, 6)
    fc = PAL(n, [(3, (0.90, 0.90, 0.88)), (2, (0.45, 0.33, 0.22)), (1, (0.30, 0.30, 0.30))])
    f = B(n, 0.6)
    paint(up, rbox(PIX, 16, fy, 20, U(n, 0.4, 0.9)), fc, f)
    paint(up, rbox(PIX, 16, fy - U(n, 2, 4), 20, U(n, 0.4, 0.8)), fc, f * B(n, 0.6))
    g = gmix([(4, "dirt"), (2, "sand"), (2, "dry"), (2, "grass")])(n)
    return blend(up, g, np.clip(YY - A(hy) + 0.5, 0, 1) * ONES), hy


def bg_dark(n):
    c = PAL(n, [(2, (0.06, 0.05, 0.05)), (1, (0.15, 0.12, 0.10)), (1, (0.10, 0.12, 0.16)), (1, (0.20, 0.20, 0.20))], 0.2, 0.02)
    return tex_fill(n, c, 0.05, 0.30, (3, 6, 16), (1.0, 0.7, 0.5)), U(n, 22, 28)


def bgc(kind, lo=0.08, hi=0.25):
    return lambda n: bg_close(n, kind, lo, hi)


def bgl(kinds, hlo, hhi, tree_p=0.5):
    return lambda n: bg_land(n, kinds, hlo, hhi, tree_p)


# ----------------------------------------------------------------------------------------------------------
# objects.  Local units are roughly pixels at scale 1; +x is the direction the object faces; +y is down.
# ----------------------------------------------------------------------------------------------------------
class Coat:
    """an animal's coat: base colour, fur texture, optional patches of a second colour"""

    def __init__(self, n, pal, patch=((1, (0.93, 0.92, 0.90)),), patch_p=0.3, lo=0.05, hi=0.2, stripe_p=0.0):
        self.col = PAL(n, pal, 0.10, 0.03)
        self.tex = furtex(n, lo, hi)
        if stripe_p > 0:
            self.tex = self.tex + A(B(n, stripe_p) * U(n, 0.1, 0.3)) * noise(n, 16, 4)
        self.pc = PAL(n, list(patch), 0.08, 0.02)
        self.pm = np.clip(noise(n, 4) * 1.5 + A(U(n, -0.5, 0.5)), 0, 1) * A(B(n, patch_p))

    def put(self, img, m, f=None, a=None):
        paint(img, m, self.col if f is None else shade(self.col, f), a, tex=self.tex)
        paint(img, m * self.pm, self.pc, a)


def wheel(img, F, n, x, y, rx, ry, a=None):
    paint(img, ell(F, x, y, rx, ry), darkc(n), a)
    paint(img, ell(F, x, y, np.asarray(rx) * 0.48, np.asarray(ry) * 0.48), C(n, (0.60, 0.60, 0.62), 0.25, 0.02), a)


def toplit(F, n, ym, hh, lo=0.05, hi=0.25):
    return A(U(n, lo, hi)) * np.clip(-(F.V - A(ym)) / A(hh), -1, 1) * ONES


# ---- class 0: airplane -----------------------------------------------------------------------------------
PLANEC = [(5, (0.90, 0.90, 0.92)), (3, (0.70, 0.72, 0.75)), (2, (0.45, 0.47, 0.50)), (2, (0.22, 0.24, 0.27)),
          (1, (0.10, 0.10, 0.12)), (0.5, (0.75, 0.20, 0.15)), (0.5, (0.85, 0.70, 0.20)), (0.5, (0.20, 0.30, 0.60)),
          (0.5, (0.35, 0.40, 0.30))]
LIVERY = [(2, (0.15, 0.25, 0.60)), (2, (0.75, 0.12, 0.12)), (1, (0.10, 0.10, 0.12)), (1, (0.85, 0.65, 0.10)), (1, (0.10, 0.40, 0.30))]


def plane_planform(img, F, n, body, wingc):
    sweep, span, chord, xr = U(n, -0.5, 7), U(n, 8, 13.5), U(n, 2.5, 5.0), U(n, 0, 3)
    tip = chord * U(n, 0.2, 0.7)
    eng = B(n, 0.5)
    for sg in (-1.0, 1.0):
        paint(img, poly(F, [(xr, 0), (xr - chord, 0), (xr - sweep - tip, sg * span), (xr - sweep, sg * span)]), wingc)
        paint(img, poly(F, [(-8.3, 0), (-10.8, 0), (-11.8, sg * 4.2), (-10.6, sg * 4.2)]), wingc)
        paint(img, ell(F, xr - sweep * 0.38 - 0.2, sg * span * 0.38, 1.7, 0.7), shade(wingc, 0.6), eng)
    paint(img, ell(F, 0, 0, 11.5, U(n, 1.2, 2.0)), body)


def plane_side(img, F, n, body, gear):
    r, fh = U(n, 1.4, 3.0), U(n, 5.0, 7.5)
    tailc = np.where(B(n, 0.45)[:, None] > 0, PAL(n, LIVERY), body)
    dk = darkc(n)
    paint(img, poly(F, [(-7.8, -r * 0.5), (-11.6, -r * 0.5), (-12.8, -fh), (-10.8, -fh)]), tailc)
    paint(img, cap(F, -9.0, -0.3, -12.3, -0.9, 0.55), shade(body, 0.8))
    for gx in (7.0, -1.8):
        paint(img, cap(F, gx, r * 0.7, gx, 3.9, 0.35), dk, gear)
        paint(img, ell(F, gx, 3.9, 0.9, 0.8), dk, gear)
    paint(img, ell(F, 0, 0, 12, r), body)
    paint(img, ell(F, -0.3, r * 0.5, 11.3, r * 0.5), np.where(B(n, 0.5)[:, None] > 0, PAL(n, LIVERY), shade(body, 0.7)), B(n, 0.5))
    paint(img, poly(F, [(2.5, 0.2), (-2.0, 0.2), (-5.5, 2.4), (-4.0, 2.4)]), shade(body, 0.65))
    paint(img, ell(F, 0.4, r + 0.8, 1.9, 0.9), shade(body, 0.55), B(n, 0.6))
    paint(img, ell(F, 9.4, -r * 0.3, 1.1, 0.45), dk)


def plane_front(img, F, n, body, gear):
    r, span, dih, finh = U(n, 1.8, 3.0), U(n, 10, 14), U(n, -1.5, 2.0), U(n, 3.5, 6.5)
    dk, wc = darkc(n), shade(body, U(n, 0.6, 1.0))
    paint(img, cap(F, 0, -r, 0, -r - finh, 0.5), wc)
    paint(img, cap(F, -4.2, -r * 0.7, 4.2, -r * 0.7, 0.35), wc)
    eng = B(n, 0.7)
    for sg in (-1.0, 1.0):
        paint(img, poly(F, [(0, -0.5), (0, 1.0), (sg * span, -dih + 0.25), (sg * span, -dih - 0.25)]), wc)
        paint(img, ell(F, sg * span * 0.42, 1.3 - dih * 0.42, 1.25, 1.25), shade(body, 0.5), eng)
        paint(img, cap(F, sg * 2.6, r * 0.5, sg * 2.6, r + 2.6, 0.4), dk, gear)
    paint(img, cap(F, 0, r * 0.5, 0, r + 2.6, 0.4), dk, gear)
    paint(img, ell(F, 0, 0, r, r), body)
    paint(img, ell(F, 0, -r * 0.4, r * 0.6, r * 0.28), dk)


def sc_plane_top(n):
    img, _ = mix(n, [(8, bg_sky), (0.7, bg_plain), (1.5, lambda m: bg_bokeh(m, TERR, 0.03, 0.12)),
                     (0.8, bgl([(1, "dry"), (1, "grass"), (1, "water"), (1, "dirt")], 22, 30, 0.5))])
    F = Frame(U(n, 10, 22), U(n, 10, 22), U(n, 0.6, 1.3), U(n, 0, 2 * np.pi), 1.0, U(n, 0.35, 1.0))
    body = PAL(n, PLANEC, 0.08, 0.02)
    plane_planform(img, F, n, body, shade(body, U(n, 0.65, 1.05)))
    return finish(img)


def sc_plane_side(n):
    img, _ = mix(n, [(8, bg_sky), (0.7, bg_plain),
                     (1.3, bgl([(1, "dry"), (1, "grass"), (1, "water"), (1, "dirt"), (1, "road")], 23, 31, 0.6))])
    F = Frame(U(n, 12, 20), U(n, 10, 22), U(n, 0.6, 1.25), Nn(n, 0.22), SG(n))
    plane_side(img, F, n, PAL(n, PLANEC, 0.08, 0.02), B(n, 0.15))
    return finish(img)


def sc_plane_ground(n):
    img, hy = bg_land(n, [(4, "road"), (2, "grass"), (1, "dry"), (0.5, "snow")], 16, 25, 0.5)
    s = U(n, 0.8, 1.45)
    gy = np.minimum(hy + U(n, 0.5, 6), 30.5)
    F = Frame(U(n, 10, 22), gy - 4.7 * s, s, Nn(n, 0.04), SG(n))
    paint(img, ell(F, 0, 4.8, 11, 0.9), darkc(n), U(n, 0.2, 0.6))
    plane_side(img, F, n, PAL(n, PLANEC[:4], 0.08, 0.02), np.ones(n))
    return finish(img)


def sc_plane_34(n):
    """three-quarter front view: the fuselage comes toward the camera and thickens; long near wing, short far wing"""
    img, hy = mix(n, [(5, bgl([(4, "road"), (2, "grass"), (1, "dry")], 15, 24, 0.5)), (5, bg_sky)])
    s = U(n, 0.8, 1.4)
    onground = (hy > 0).astype(np.float64)
    cy = np.where(hy > 0, np.minimum(hy + U(n, 1, 6), 30.5) - 7.5 * s, U(n, 10, 22))
    F = Frame(U(n, 12, 20), cy, s, Nn(n, 0.05) + (1 - onground) * Nn(n, 0.3), SG(n))
    body = PAL(n, PLANEC, 0.08, 0.02)
    r1 = U(n, 2.4, 3.6)
    r0 = r1 * U(n, 0.35, 0.55)
    tailc = np.where(B(n, 0.45)[:, None] > 0, PAL(n, LIVERY), body)
    dk = darkc(n)
    fw, nw = U(n, 4, 7), U(n, 0, 2.5)
    paint(img, ell(F, -1, 7.6, 11, 1.0), dk, onground * U(n, 0.2, 0.6))
    paint(img, poly(F, [(0.5, -1.0), (-3.0, -1.0), (-6.5, -fw), (-5.0, -fw)]), shade(body, 0.75))
    paint(img, poly(F, [(-8.5, -3.2), (-11.0, -3.6), (-12.5, -9.0), (-10.8, -9.0)]), tailc)
    paint(img, cap(F, -9.5, -3.0, -13.0, -3.8, 0.5), shade(body, 0.8))
    for gx, gy0 in ((5.5, 3.5), (-3.0, 3.0)):
        paint(img, cap(F, gx, gy0, gx, 6.5, 0.4), dk, onground)
        paint(img, ell(F, gx, 6.8, 1.0, 0.9), dk, onground)
    paint(img, cap(F, -10, -3, 8, 1.5, r0, r1), body, tex=toplit(F, n, 0.0, 3.0, 0.0, 0.2))
    paint(img, poly(F, [(3.5, 2.0), (-2.5, 0.8), (-11.5, 6.5 + nw), (-8.5, 7.5 + nw)]), shade(body, U(n, 0.6, 0.9)))
    paint(img, ell(F, -2.5, 4.6, 2.3, 1.4, 0.25), shade(body, 0.55), B(n, 0.7))
    paint(img, ell(F, 8.2, 0.2, 1.3, 0.6, 0.25), dk)
    return finish(img)


def sc_plane_front(n):
    img, hy = mix(n, [(5, bgl([(4, "road"), (2, "grass"), (1, "dry")], 17, 26, 0.5)), (5, bg_sky)])
    s = U(n, 0.8, 1.25)
    onground = (hy > 0).astype(np.float64)
    cy = np.where(hy > 0, np.minimum(hy + U(n, 0, 5), 30) - 5.2 * s, U(n, 10, 22))
    F = Frame(U(n, 12, 20), cy, s, Nn(n, 0.05) + (1 - onground) * Nn(n, 0.3), 1.0)
    plane_front(img, F, n, PAL(n, PLANEC, 0.08, 0.02), onground)
    return finish(img)


# ---- class 1: automobile ---------------------------------------------------------------------------------
CARC = [(3, (0.72, 0.10, 0.10)), (3, (0.75, 0.76, 0.78)), (2.5, (0.92, 0.92, 0.92)), (2, (0.10, 0.10, 0.12)),
        (2, (0.15, 0.25, 0.55)), (1, (0.45, 0.50, 0.60)), (1, (0.88, 0.72, 0.12)), (1, (0.12, 0.32, 0.20)),
        (1, (0.70, 0.62, 0.48)), (0.7, (0.40, 0.08, 0.10)), (0.7, (0.85, 0.45, 0.10)), (0.8, (0.35, 0.36, 0.38))]


def car_bg(n):
    return mix(n, [(6.5, bg_street), (1.0, bg_plain), (0.6, bg_dark), (1.0, bgl([(2, "road"), (1, "grass"), (1, "dirt")], 8, 18, 0.8)),
                   (0.7, lambda m: (ground_tex(m, "road"), np.zeros(m)))])


def lightc(n, front):
    return np.where(np.asarray(front)[:, None] > 0, C(n, (0.95, 0.93, 0.80), 0.05, 0.02), C(n, (0.75, 0.10, 0.08), 0.1, 0.02))


def car_side(img, F, n, col):
    hb, hc = U(n, 4.0, 5.6), U(n, 3.0, 4.6)
    ys = -1.5
    yb = ys - hb
    cf, cr, sf, sr = U(n, 2.5, 7.5), U(n, -10.5, -5.0), U(n, 1.5, 4.0), U(n, 0.3, 3.0)
    paint(img, ell(F, 0, 0.2, 12.5, 1.3), darkc(n), U(n, 0.4, 0.9))
    paint(img, poly(F, [(cr, yb + 0.5), (cf, yb + 0.5), (cf - sf, yb - hc), (cr + sr, yb - hc)]), col)
    paint(img, poly(F, [(cr + 0.8, yb + 0.1), (cf - 0.9, yb + 0.1), (cf - sf - 0.1, yb - hc + 0.7), (cr + sr + 0.5, yb - hc + 0.7)]), glassc(n))
    pm = (cr + cf) / 2 - 0.5
    paint(img, cap(F, pm, yb, pm, yb - hc + 0.5, 0.35), col)
    paint(img, rbox(F, 0, (ys + yb) / 2, 12, hb / 2, 1.6), col, tex=toplit(F, n, (ys + yb) / 2, hb / 2))
    wr, wx = U(n, 2.2, 2.9), U(n, 6.6, 8.0)
    for sg in (-1.0, 1.0):
        wheel(img, F, n, sg * wx, -wr + 0.2, wr, wr)
    paint(img, ell(F, 11.3, yb + 1.4, 0.8, 0.7), C(n, (0.95, 0.93, 0.80), 0.05, 0.02))
    paint(img, ell(F, -11.5, yb + 1.4, 0.7, 0.7), C(n, (0.75, 0.10, 0.08), 0.1, 0.02))


def car_34(img, F, n, col):
    hb, hc = U(n, 4.5, 6.0), U(n, 3.2, 4.6)
    ys = -1.3
    yb = ys - hb
    fx, rise, hd = U(n, 1.0, 4.5), U(n, 0.3, 1.5), U(n, 0.8, 2.2)
    fw = 11.5 - fx
    front = B(n, 0.65)
    g = glassc(n)
    paint(img, ell(F, 0, 0.3, 12.5, 1.6), darkc(n), U(n, 0.4, 0.9))
    paint(img, poly(F, [(-12, yb - rise + 1.0), (fx, yb), (fx, ys), (-12, ys - rise)]), shade(col, U(n, 0.7, 0.95)))
    cy0 = yb - hd
    paint(img, poly(F, [(-10.5, yb - rise * 0.85 + 0.6), (10.3, cy0), (8.8, cy0 - hc), (-7.5, cy0 - hc - rise * 0.6)]), col)
    paint(img, rbox(F, fx + fw / 2, (ys + yb) / 2, fw / 2, hb / 2, 1.2), col, tex=toplit(F, n, (ys + yb) / 2, hb / 2))
    paint(img, poly(F, [(fx, yb + 0.3), (11.2, yb + 0.4), (10.3, cy0), (fx - 1.5, cy0 - rise * 0.2)]), shade(col, U(n, 1.0, 1.25)))
    paint(img, poly(F, [(fx - 0.8, cy0 - 0.3), (9.6, cy0 - 0.3), (8.4, cy0 - hc + 0.6), (fx - 1.4, cy0 - hc + 0.5)]), g)
    paint(img, poly(F, [(-9.3, yb - rise * 0.8 + 0.2), (fx - 2.2, cy0 - 0.2), (fx - 2.6, cy0 - hc + 0.6), (-7.2, cy0 - hc - rise * 0.5 + 0.7)]), shade(g, 0.85))
    lc = lightc(n, front)
    paint(img, ell(F, fx + 1.4, yb + 1.6, 1.2, 0.8), lc)
    paint(img, ell(F, 10.3, yb + 1.6, 1.0, 0.8), lc)
    paint(img, rbox(F, fx + fw / 2, yb + 1.9, fw / 2 - 2.6, 0.8), darkc(n), front)
    paint(img, rbox(F, fx + fw / 2, ys - 0.9, fw / 2 + 0.2, 0.8, 0.4), shade(col, 0.55))
    wheel(img, F, n, fx - 1.8, -2.2, 1.9, 2.5)
    wheel(img, F, n, -9.5, -2.2 - rise, 1.5, 2.1)
    paint(img, ell(F, 10.6, -1.9, 1.0, 2.0), darkc(n), 0.8 * np.ones(n))


def car_front(img, F, n, col):
    w, hb, hc = U(n, 8.0, 10.5), U(n, 5.0, 6.5), U(n, 3.5, 5.0)
    ys = -1.5
    yb = ys - hb
    front = B(n, 0.6)
    dk = darkc(n)
    paint(img, ell(F, 0, 0.2, w + 1, 1.3), dk, U(n, 0.4, 0.9))
    for sg in (-1.0, 1.0):
        paint(img, rbox(F, sg * (w - 1.2), -1.6, 1.3, 1.9, 0.5), dk)
    paint(img, poly(F, [(-w + 0.8, yb + 0.5), (w - 0.8, yb + 0.5), (w - 2.6, yb - hc), (-w + 2.6, yb - hc)]), col)
    paint(img, poly(F, [(-w + 1.7, yb), (w - 1.7, yb), (w - 3.1, yb - hc + 0.7), (-w + 3.1, yb - hc + 0.7)]), glassc(n))
    paint(img, rbox(F, 0, (ys + yb) / 2, w, hb / 2, 1.5), col, tex=toplit(F, n, (ys + yb) / 2, hb / 2))
    lc = lightc(n, front)
    for sg in (-1.0, 1.0):
        paint(img, ell(F, sg * (w - 2.0), yb + 2.0, 1.5, 0.9), lc)
        paint(img, ell(F, sg * (w + 0.6), yb - 0.6, 0.8, 0.6), col)
    paint(img, rbox(F, 0, yb + 2.2, w - 4.2, 0.9), dk, front)
    paint(img, rbox(F, 0, ys - 1.7, 1.8, 0.7), whitec(n), 1 - front)
    paint(img, rbox(F, 0, ys - 0.6, w, 0.8, 0.3), shade(col, 0.55))


def sc_car_side(n):
    img, _ = car_bg(n)
    F = Frame(U(n, 13, 19), U(n, 21, 29), U(n, 0.8, 1.22), Nn(n, 0.04), SG(n))
    car_side(img, F, n, PAL(n, CARC, 0.10, 0.03))
    return finish(img)


def sc_car_34(n):
    img, _ = car_bg(n)
    F = Frame(U(n, 13, 19), U(n, 22, 30), U(n, 0.85, 1.4), Nn(n, 0.06), SG(n))
    car_34(img, F, n, PAL(n, CARC, 0.10, 0.03))
    return finish(img)


def sc_car_front(n):
    img, _ = car_bg(n)
    F = Frame(U(n, 12, 20), U(n, 22, 30), U(n, 0.9, 1.5), Nn(n, 0.05), 1.0)
    car_front(img, F, n, PAL(n, CARC, 0.10, 0.03))
    return finish(img)


# ---- class 9: truck --------------------------------------------------------------------------------------
TRUCKC = [(3, (0.75, 0.12, 0.10)), (3, (0.92, 0.92, 0.92)), (1.5, (0.15, 0.25, 0.55)), (1.5, (0.90, 0.72, 0.10)),
          (1, (0.85, 0.45, 0.10)), (1, (0.12, 0.35, 0.20)), (1, (0.55, 0.57, 0.60)), (1, (0.10, 0.10, 0.12)),
          (0.7, (0.35, 0.55, 0.75)), (0.5, (0.45, 0.10, 0.10))]
CARGOC = [(4, (0.92, 0.92, 0.92)), (2, (0.72, 0.74, 0.76)), (1, (0.75, 0.12, 0.10)), (1, (0.15, 0.25, 0.55)),
          (1, (0.90, 0.72, 0.10)), (1, (0.30, 0.32, 0.34)), (0.7, (0.12, 0.35, 0.20)), (0.7, (0.50, 0.40, 0.30)),
          (0.7, (0.85, 0.45, 0.10))]
CHROME = [(2, (0.78, 0.80, 0.82)), (1, (0.50, 0.52, 0.55)), (1.5, (0.12, 0.12, 0.13))]


def truck_bg(n):
    return mix(n, [(5, lambda m: bg_street(m, 3.0)), (2.5, bgl([(4, "road"), (1.5, "dirt"), (1, "grass"), (0.5, "snow")], 12, 24, 0.6)),
                   (1.5, bg_plain), (0.4, bg_dark)])


def truck_side(img, F, n, col, cargo):
    L, yc = 13.0, -3.6
    hood = B(n, 0.5)
    hl, cw, ch, hh = hood * U(n, 2.5, 4.0), U(n, 4.2, 5.5), U(n, 8.5, 11.5), U(n, 4.0, 5.5)
    cx0 = 13.0 - hl - cw
    typ = RNG.choice(4, size=n, p=[0.45, 0.2, 0.2, 0.15])
    bh, dh = U(n, 9.5, 14.0), U(n, 6.0, 9.0)
    xm, xw = (cx0 - 0.6 - L) / 2, (cx0 - 0.6 + L) / 2
    paint(img, ell(F, 0, 0.3, 13.5, 1.4), darkc(n), U(n, 0.4, 0.9))
    ct = toplit(F, n, yc - bh / 2, bh / 2, 0.0, 0.15)
    paint(img, rbox(F, xm, yc - bh / 2, xw, bh / 2, 0.4), cargo, typ == 0, tex=ct)
    paint(img, rbox(F, xm, yc - 4.2, xw, 3.9, 3.5), cargo, typ == 1, tex=toplit(F, n, yc - 4.2, 3.9, 0.1, 0.35))
    paint(img, poly(F, [(-L, yc - dh), (cx0 + 1.5, yc - dh - 1.2), (cx0 - 1.0, yc), (-L + 2.0, yc)]), cargo, typ == 2)
    paint(img, rbox(F, xm, yc - 2.0, xw, 2.0, 0.3), cargo, typ == 3)
    paint(img, rbox(F, 0, yc + 0.8, L, 0.9), darkc(n))
    paint(img, cap(F, cx0 - 0.4, yc - ch - 1.5, cx0 - 0.4, yc, 0.35), PAL(n, CHROME), B(n, 0.4))
    paint(img, rbox(F, cx0 + cw / 2, yc - ch / 2 + 0.3, cw / 2, ch / 2 + 0.3, 0.8), col)
    paint(img, rbox(F, 13.0 - hl / 2 - 0.2, yc - hh / 2 + 0.3, hl / 2 + 0.2, hh / 2 + 0.3, 0.6), col, hood)
    paint(img, rbox(F, cx0 + cw - 1.7, yc - ch + 2.5, 1.4, 1.6, 0.3), glassc(n))
    paint(img, rbox(F, 13.3, -3.3, 0.6, 1.0, 0.2), PAL(n, CHROME))
    wr = U(n, 2.4, 3.0)
    wheel(img, F, n, 13.0 - 2.8 - hl * 0.3, -wr + 0.2, wr, wr)
    wheel(img, F, n, -L + 3.3, -wr + 0.2, wr, wr)
    wheel(img, F, n, -L + 3.3 + 2 * wr + 0.4, -wr + 0.2, wr, wr, B(n, 0.6))


def truck_face(img, F, n, col, x0, fw, yb, fh, hood):
    """the front of a truck cab: windscreen band on top, grille, bumper, headlights"""
    xm = x0 + fw / 2
    paint(img, rbox(F, xm, yb - fh / 2, fw / 2, fh / 2, 0.8), col, tex=toplit(F, n, yb - fh / 2, fh / 2, 0.0, 0.15))
    wh = fh * U(n, 0.25, 0.36)
    wy = yb - fh + 1.2 + wh / 2
    paint(img, rbox(F, xm, wy, fw / 2 - 0.8, wh / 2, 0.4), glassc(n))
    paint(img, cap(F, xm, wy - wh / 2, xm, wy + wh / 2, 0.3), col, B(n, 0.5))
    paint(img, rbox(F, xm, wy + wh / 2 + 0.9, fw / 2 - 0.3, 0.8), shade(col, 1.2), hood)
    gh = fh * U(n, 0.25, 0.40)
    paint(img, rbox(F, xm, yb - 2.2 - gh / 2, fw * U(n, 0.22, 0.36), gh / 2, 0.3), PAL(n, CHROME))
    paint(img, rbox(F, xm, yb - 0.6, fw / 2 + 0.6, 1.0, 0.3), PAL(n, CHROME + [(2, (0.6, 0.15, 0.1))]))
    lc = C(n, (0.95, 0.93, 0.80), 0.05, 0.02)
    paint(img, ell(F, x0 + 1.3, yb - 2.4, 1.0, 0.8), lc)
    paint(img, ell(F, x0 + fw - 1.3, yb - 2.4, 1.0, 0.8), lc)


def truck_34(img, F, n, col, cargo):
    fw, fh, rise, sd = U(n, 8.5, 11.0), U(n, 11, 15), U(n, 0.5, 2.0), U(n, 2.5, 4.5)
    fx, yb = 12.0 - fw, -2.6
    hood, hasbox, tandem = B(n, 0.45), B(n, 0.8), B(n, 0.5)
    bh = fh + U(n, -1, 4)
    dk = darkc(n)
    paint(img, ell(F, 0, 0.3, 13.5, 1.7), dk, U(n, 0.4, 0.9))
    paint(img, poly(F, [(-13, yb - rise - bh * 0.8), (fx + 1, yb - bh - 0.5), (fx + 1, yb + 0.5), (-13, yb - rise * 1.2)]),
          shade(cargo, U(n, 0.75, 1.0)), hasbox)
    paint(img, poly(F, [(-13, yb - rise * 1.2 - 1.2), (fx + 1, yb - 0.6), (fx + 1, yb + 0.8), (-13, yb - rise * 1.2 + 0.2)]), dk)
    wheel(img, F, n, -7.5, -2.6 - rise * 0.8, 1.8, 2.3)
    wheel(img, F, n, -11.0, -2.8 - rise, 1.6, 2.1, tandem)
    paint(img, poly(F, [(fx - sd, yb - rise * 0.4 - fh * 0.96), (fx, yb - fh), (fx, yb), (fx - sd, yb - rise * 0.4)]), shade(col, U(n, 0.6, 0.9)))
    paint(img, poly(F, [(fx - sd + 0.8, yb - rise * 0.4 - fh * 0.88), (fx - 0.6, yb - fh + 1.3), (fx - 0.6, yb - fh * 0.62), (fx - sd + 0.8, yb - rise * 0.4 - fh * 0.60)]), shade(glassc(n), 0.85))
    wheel(img, F, n, fx + 1.2, -2.3, 1.5, 2.6)
    paint(img, ell(F, fx + fw - 1.0, -2.2, 1.3, 2.5), dk)
    truck_face(img, F, n, col, fx, fw, yb, fh, hood)


def truck_front(img, F, n, col, cargo):
    w, fh, yb = U(n, 7.5, 10), U(n, 12, 17), -2.6
    dk = darkc(n)
    paint(img, ell(F, 0, 0.2, w + 1.5, 1.4), dk, U(n, 0.4, 0.9))
    paint(img, rbox(F, 0, yb - fh / 2 - 2, w + 1.2, fh / 2 + 2, 0.3), cargo, B(n, 0.5))
    for sg in (-1.0, 1.0):
        paint(img, rbox(F, sg * (w - 1.3), -2.0, 1.4, 2.2, 0.5), dk)
        paint(img, cap(F, sg * (w + 1.2), yb - fh + 2, sg * (w + 1.2), yb - fh + 5, 0.5), dk)
    truck_face(img, F, n, col, -w, 2 * w, yb, fh, B(n, 0.45))


def sc_truck_side(n):
    img, _ = truck_bg(n)
    F = Frame(U(n, 13, 19), U(n, 22, 30), U(n, 0.8, 1.15), Nn(n, 0.04), SG(n))
    truck_side(img, F, n, PAL(n, TRUCKC, 0.10, 0.03), PAL(n, CARGOC, 0.10, 0.03))
    return finish(img)


def sc_truck_34(n):
    img, _ = truck_bg(n)
    F = Frame(U(n, 13, 19), U(n, 23, 31), U(n, 0.85, 1.25), Nn(n, 0.05), SG(n))
    truck_34(img, F, n, PAL(n, TRUCKC, 0.10, 0.03), PAL(n, CARGOC, 0.10, 0.03))
    return finish(img)


def sc_truck_front(n):
    img, _ = truck_bg(n)
    F = Frame(U(n, 12, 20), U(n, 24, 31), U(n, 0.85, 1.3), Nn(n, 0.04), 1.0)
    truck_front(img, F, n, PAL(n, TRUCKC, 0.10, 0.03), PAL(n, CARGOC, 0.10, 0.03))
    return finish(img)


# ---- class 8: ship ---------------------------------------------------------------------------------------
HULLC = [(3, (0.10, 0.10, 0.12)), (2, (0.92, 0.92, 0.92)), (2, (0.50, 0.10, 0.08)), (1.5, (0.12, 0.20, 0.40)),
         (2, (0.45, 0.47, 0.50)), (1, (0.75, 0.15, 0.12)), (0.7, (0.10, 0.30, 0.20)), (0.7, (0.15, 0.40, 0.70))]
SUPERC = [(5, (0.93, 0.93, 0.92)), (1.5, (0.60, 0.62, 0.65)), (1, (0.80, 0.72, 0.55)), (0.7, (0.85, 0.55, 0.15))]
FUNNELC = [(2, (0.75, 0.15, 0.10)), (2, (0.10, 0.10, 0.12)), (1, (0.90, 0.75, 0.15)), (1, (0.15, 0.25, 0.55)), (1, (0.90, 0.90, 0.90))]


def ship_side(img, F, n):
    """side view: waterline at y = 0, bow toward +x; cargo ship / liner or ferry / warship / small working boat"""
    typ = RNG.choice(4, size=n, p=[0.3, 0.25, 0.15, 0.3])
    L = np.where(typ == 3, U(n, 7.5, 11), U(n, 11.5, 13.5))
    hh, bow, stern = U(n, 2.2, 4.5), U(n, 1.0, 3.5), U(n, 0.0, 1.5)
    hull, sup, fun, dk = PAL(n, HULLC), PAL(n, SUPERC, 0.06, 0.02), PAL(n, FUNNELC), darkc(n)
    hull = np.where((typ == 2)[:, None], C(n, (0.50, 0.52, 0.55), 0.12, 0.02), hull)
    sup = np.where((typ == 2)[:, None], shade(hull, 1.1), sup)
    paint(img, ell(F, 0, 1.0, L, 1.2), dk, U(n, 0.15, 0.5))
    t0, t1, t2, t3 = (typ == 0), (typ == 1), (typ == 2), (typ == 3)
    # cargo ship: bridge at the stern, funnel, deck load, masts
    paint(img, rbox(F, -L + 4.2, -hh - 2.6, 2.4, 2.8, 0.2), sup, t0)
    paint(img, rbox(F, -L + 3.4, -hh - 6.3, 0.9, 1.3), fun, t0)
    for i in range(4):
        paint(img, rbox(F, -L + 9.0 + 3.6 * i, -hh - U(n, 0.8, 1.5), 1.7, 1.4), PAL(n, CARGOC + HULLC), t0 * B(n, 0.6))
    paint(img, cap(F, L - 8, -hh, L - 8, -hh - U(n, 4, 7), 0.3), dk, t0 * B(n, 0.6))
    # liner / ferry: long white superstructure in two tiers, funnel, dark window band
    paint(img, rbox(F, -0.5, -hh - 2.2, L - 2.5, 2.3, 0.5), sup, t1)
    paint(img, rbox(F, -1.0, -hh - 5.4, L - 5.5, 1.1, 0.4), sup, t1)
    paint(img, rbox(F, -2.5, -hh - 7.6, 1.3, 1.3, 0.2), fun, t1)
    paint(img, rbox(F, -0.5, -hh - 2.7, L - 3.5, 0.3), glassc(n), t1 * 0.7)
    # warship: stepped grey superstructure, mast, funnel, gun
    paint(img, rbox(F, -1.0, -hh - 1.4, 6.5, 1.5, 0.2), sup, t2)
    paint(img, rbox(F, 0.0, -hh - 3.8, 3.0, 1.3, 0.2), sup, t2)
    paint(img, cap(F, 0.5, -hh - 4, 0.5, -hh - 9.5, 0.35), shade(sup, 0.8), t2)
    paint(img, rbox(F, -3.5, -hh - 3.6, 1.0, 1.4), shade(sup, 0.8), t2)
    paint(img, rbox(F, 7.5, -hh - 0.7, 1.2, 0.7), shade(sup, 0.9), t2)
    # fishing boat / tug: cabin, mast and boom
    cxb = U(n, -3, 3)
    paint(img, rbox(F, cxb, -hh - 2.4, 2.8, 2.5, 0.3), sup, t3)
    paint(img, rbox(F, cxb + 0.6, -hh - 3.2, 1.8, 0.8, 0.1), glassc(n), t3)
    paint(img, cap(F, cxb - 2.2, -hh - 4.5, cxb - 2.2, -hh - 10, 0.3), dk, t3 * B(n, 0.7))
    paint(img, cap(F, cxb - 2.2, -hh - 9, cxb - 7, -hh - 4, 0.25), dk, t3 * B(n, 0.4))
    # hull
    paint(img, poly(F, [(-L, -hh), (L + bow * 0.3, -hh - bow * 0.3), (L - bow, 0.4), (-L + stern, 0.4)]), hull,
          tex=toplit(F, n, -hh / 2, hh / 2, 0.0, 0.2))
    paint(img, poly(F, [(-L + stern * 0.8, -0.9), (L - bow * 0.8, -0.9), (L - bow, 0.4), (-L + stern, 0.4)]),
          C(n, (0.55, 0.12, 0.10), 0.15, 0.02), B(n, 0.3))
    paint(img, ell(F, L - 1.5, 0.4, 3.0, 0.6), whitec(n), B(n, 0.4) * 0.7)
    return L


def sc_ship_side(n):
    img, hy = mix(n, [(8, bg_sea), (1, bg_plain), (1, lambda m: bg_land(m, [(1, "water")], 2, 8, 0.8))])
    s = U(n, 0.75, 1.5)
    wy = np.clip(hy + U(n, -1, 10), 14, 29)
    F = Frame(U(n, 10, 22), wy, s, Nn(n, 0.04), SG(n), U(n, 1.0, 1.7))
    ship_side(img, F, n)
    return finish(img)


def sc_ship_34(n):
    """three-quarter view: tall bow near the camera on the +x side, hull receding and rising toward the horizon"""
    img, hy = mix(n, [(7, bg_sea), (2, lambda m: bg_land(m, [(1, "water")], 2, 9, 0.8)), (1, bg_water)])
    s = U(n, 0.8, 1.4)
    F = Frame(U(n, 11, 21), np.clip(hy + U(n, 2, 14), 17, 30), s, Nn(n, 0.05), SG(n), U(n, 1.0, 1.4))
    hull, sup, fun, dk = PAL(n, HULLC), PAL(n, SUPERC, 0.06, 0.02), PAL(n, FUNNELC), darkc(n)
    r, hb, hs, sh = U(n, 0.5, 3.5), U(n, 5.5, 8.5), U(n, 2.2, 3.5), U(n, 3, 6)
    paint(img, ell(F, 0, 0.8, 13, 1.3), dk, U(n, 0.15, 0.5))
    paint(img, poly(F, [(-11, -hs - r - sh * 0.7), (4, -hb * 0.75 - sh), (4, -hb * 0.75 + 0.5), (-11, -hs - r + 0.5)]), sup)
    paint(img, poly(F, [(4, -hb * 0.75 - sh), (6.5, -hb * 0.75 - sh * 0.9), (6.5, -hb * 0.75 + 0.5), (4, -hb * 0.75 + 0.5)]), shade(sup, 0.8))
    paint(img, rbox(F, -3, -hb * 0.7 - sh - 1.2, 1.2, 1.5), fun, B(n, 0.6))
    paint(img, cap(F, 1, -hb * 0.75 - sh, 1, -hb * 0.75 - sh - U(n, 2, 5), 0.3), dk, B(n, 0.5))
    paint(img, poly(F, [(-13, -hs - r), (9.5, -hb), (9.0, 0.5), (-13, 0.2 - r)]), hull, tex=toplit(F, n, -hb / 2, hb / 2, 0.0, 0.2))
    paint(img, poly(F, [(9.5, -hb), (12.5, -hb * 0.8), (9.0, 0.5)]), shade(hull, U(n, 0.6, 1.3)))
    paint(img, poly(F, [(-13, -0.9 - r), (9.1, -0.8), (9.0, 0.5), (-13, 0.2 - r)]), C(n, (0.55, 0.12, 0.10), 0.15, 0.02), B(n, 0.3))
    paint(img, ell(F, 9.5, 0.7, 4.0, 0.9), whitec(n), B(n, 0.6) * 0.8)
    return finish(img)


def sc_ship_sail(n):
    img, hy = bg_sea(n, 10, 22)
    s = U(n, 0.65, 1.3)
    F = Frame(U(n, 10, 22), np.clip(hy + U(n, 0, 9), 16, 30), s, Nn(n, 0.06), SG(n))
    top = U(n, 14, 20)
    sail = PAL(n, [(5, (0.95, 0.95, 0.93)), (1, (0.85, 0.80, 0.65)), (1, (0.75, 0.20, 0.15)), (0.7, (0.2, 0.3, 0.6))], 0.05, 0.02)
    paint(img, ell(F, 0, 1.0, 6, 1.0), darkc(n), U(n, 0.1, 0.4))
    paint(img, poly(F, [(-0.3, -top + 0.5), (-0.3, -3.0), (-U(n, 5, 8), -3.0)]), sail)
    paint(img, poly(F, [(0.9, -top * 0.92), (0.9, -2.6), (U(n, 4.5, 7), -2.4)]), shade(sail, U(n, 0.85, 1.0)), B(n, 0.8))
    paint(img, cap(F, 0.3, -1.5, 0.3, -top, 0.3), darkc(n))
    paint(img, poly(F, [(-6.5, -1.6), (7.5, -1.9), (5.5, 0.3), (-5.5, 0.3)]), PAL(n, HULLC + [(3, (0.93, 0.93, 0.92))]))
    return finish(img)


def sc_ship_speed(n):
    img, _ = mix(n, [(7, bg_water), (3, lambda m: bg_land(m, [(1, "water")], 2, 9, 0.8))])
    F = Frame(U(n, 11, 21), U(n, 12, 22), U(n, 0.75, 1.3), Nn(n, 0.45), SG(n), U(n, 0.55, 1.0))
    hull = PAL(n, [(6, (0.94, 0.94, 0.93)), (1, (0.75, 0.15, 0.12)), (1, (0.15, 0.25, 0.55)), (1, (0.90, 0.75, 0.15)), (1, (0.1, 0.1, 0.12))], 0.05, 0.02)
    foam = np.clip(0.6 + 0.5 * noise(n, 8), 0, 1) * A(U(n, 0.3, 0.9))
    paint(img, poly(F, [(-8, -2.8), (-8, 2.8), (-24, 8.5), (-24, -8.5)]) * foam, whitec(n))
    paint(img, ell(F, 6, 2.5, 6, 1.5), whitec(n), U(n, 0.0, 0.7))
    paint(img, poly(F, [(-9, -3.2), (3, -3.6), (11.5, 0), (3, 3.6), (-9, 3.2)]), hull)
    paint(img, poly(F, [(-9, 1.2), (3, 1.4), (11.5, 0), (3, 3.6), (-9, 3.2)]), shade(hull, U(n, 0.55, 0.9)))
    paint(img, rbox(F, -3.5, -0.6, 4.0, 1.7, 0.5), PAL(n, [(2, (0.2, 0.25, 0.35)), (1, (0.5, 0.5, 0.5)), (1, (0.75, 0.7, 0.6))]))
    paint(img, rbox(F, 1.2, -0.6, 0.7, 2.0, 0.2), glassc(n))
    return finish(img)


# ---- class 2: bird ---------------------------------------------------------------------------------------
BIRDC = [(3, (0.42, 0.32, 0.22)), (2, (0.50, 0.50, 0.50)), (2, (0.12, 0.12, 0.13)), (1.5, (0.90, 0.90, 0.88)),
         (1, (0.30, 0.24, 0.18)), (1, (0.65, 0.55, 0.40)), (0.5, (0.80, 0.25, 0.12)), (0.5, (0.20, 0.35, 0.70)),
         (0.5, (0.85, 0.75, 0.20)), (0.5, (0.25, 0.40, 0.30))]
BELLYC = [(3, (0.90, 0.88, 0.82)), (2, (0.75, 0.65, 0.50)), (1, (0.85, 0.55, 0.25)), (1, (0.60, 0.60, 0.60)), (0.5, (0.90, 0.80, 0.25))]
BEAKC = [(2, (0.20, 0.18, 0.15)), (2, (0.85, 0.65, 0.20)), (1, (0.60, 0.55, 0.50)), (0.5, (0.85, 0.35, 0.15))]


def bird_body(img, F, n, col, tex, legs=None, legc=None):
    """small bird in profile, body tilted head-up, facing +x; origin at the body centre"""
    tilt = -U(n, 0.3, 0.9)
    tl = U(n, -1.5, 2.0)
    headc = np.where(B(n, 0.3)[:, None] > 0, PAL(n, BIRDC), col)
    if legs is not None:
        for lx in (0.4, -1.1):
            paint(img, cap(F, lx, 2.8, lx + 0.3, 6.4, 0.38), legc, legs)
    paint(img, poly(F, [(-3.2, 0.8), (-4.2, 2.6), (-10.5, 6.0 + tl), (-9.5, 4.3 + tl)]), shade(col, 0.75))
    paint(img, ell(F, 0, 0, 5.5, 3.5, tilt), col, tex=tex)
    paint(img, ell(F, 0.9, 1.0, 4.2, 2.3, tilt), PAL(n, BELLYC), U(n, 0.2, 1.0))
    paint(img, ell(F, -1.2, -0.3, 4.0, 2.0, tilt + 0.15), shade(col, 0.7), U(n, 0.3, 1.0), tex=tex)
    paint(img, ell(F, 4.2, -3.8, 2.4, 2.2), headc)
    paint(img, poly(F, [(6.0, -4.6), (9.0, -3.6), (6.0, -2.9)]), PAL(n, BEAKC))
    paint(img, ell(F, 5.0, -4.3, 0.55, 0.55), darkc(n))


def sc_bird_perch(n):
    img, _ = mix(n, [(5, bg_bokeh), (2.5, bg_sky), (0.7, bg_plain), (1, bg_forest), (0.5, bgc("grass", 0.05, 0.15)), (1, bg_brush)])
    s = U(n, 0.8, 2.0)
    F = Frame(U(n, 11, 21), U(n, 11, 19), s, Nn(n, 0.2), SG(n))
    wood = PAL(n, [(3, (0.38, 0.28, 0.20)), (2, (0.50, 0.48, 0.45)), (1, (0.20, 0.16, 0.13)), (1, (0.60, 0.50, 0.38))])
    kind = RNG.choice(3, size=n, p=[0.6, 0.2, 0.2])
    sl = Nn(n, 3)
    paint(img, cap(F, -30, 7.6 + sl, 30, 7.6 - sl, U(n, 0.6, 1.6)), wood, kind == 0)
    paint(img, rbox(F, 0, 6.6 + 14, U(n, 1.3, 2.6), 14, 0.3), wood, kind == 1)
    paint(img, ell(F, 0, 6.4 + 6, U(n, 6, 14), 6), wood, kind == 2, tex=furtex(n, 0.05, 0.2))
    bird_body(img, F, n, PAL(n, BIRDC), furtex(n, 0.03, 0.15), np.ones(n), PAL(n, BEAKC))
    return finish(img)


def sc_bird_ground(n):
    img, _ = mix(n, [(2, bgc("soil")), (2, bgc("grass", 0.06, 0.2)), (1, bgc("sand", 0.04, 0.15)), (1, bgc("rock")), (1, bgc("litter")),
                     (1.5, bg_brush), (2, bgl([(3, "grass"), (2, "dirt"), (1, "sand"), (1, "snow")], 3, 14, 0.6))])
    s = U(n, 0.8, 1.8)
    F = Frame(U(n, 11, 21), U(n, 12, 20), s, Nn(n, 0.12), SG(n))
    paint(img, ell(F, -1, 6.6, 7, 1.2), darkc(n), U(n, 0.1, 0.5))
    bird_body(img, F, n, PAL(n, BIRDC), furtex(n, 0.03, 0.15), np.ones(n), PAL(n, BEAKC))
    return finish(img)


def sc_bird_fly(n):
    """bird on the wing: short body, wings bent at the wrist and tapering to the tips, fanned tail"""
    img, _ = mix(n, [(8, bg_sky), (1, bg_bokeh), (1, lambda m: bg_sea(m, 20, 30))])
    F = Frame(U(n, 9, 23), U(n, 9, 23), U(n, 0.7, 1.6), Nn(n, 0.5), SG(n), U(n, 0.45, 1.0))
    col = PAL(n, BIRDC + [(4, (0.10, 0.10, 0.10)), (2, (0.92, 0.92, 0.92))])
    wl, fw, bk = U(n, 7, 12), U(n, 0.5, 3.0), U(n, 1.0, 5.0)
    wc = shade(col, U(n, 0.7, 1.0))
    for sg in (-1.0, 1.0):
        paint(img, cap(F, 0.5, sg * 0.8, 0.5 + fw, sg * wl * 0.5, 1.5, 1.2), wc)
        paint(img, cap(F, 0.5 + fw, sg * wl * 0.5, 0.5 + fw - bk, sg * wl, 1.2, 0.4), wc)
    paint(img, poly(F, [(-3, 0.7), (-3, -0.7), (-7, -2.0), (-7, 2.0)]), col)
    paint(img, ell(F, 0, 0, 4.2, 1.6), col)
    paint(img, ell(F, 4.2, 0, 1.5, 1.3), col)
    paint(img, poly(F, [(5.3, -0.5), (7.3, 0), (5.3, 0.5)]), PAL(n, BEAKC))
    return finish(img)


def sc_bird_water(n):
    img, _ = mix(n, [(6, bg_water), (4, lambda m: bg_land(m, [(1, "water")], 2, 12, 0.8))])
    F = Frame(U(n, 11, 21), U(n, 14, 22), U(n, 0.8, 1.7), Nn(n, 0.05), SG(n))
    col = PAL(n, BIRDC + [(4, (0.93, 0.93, 0.92))])
    nk = U(n, 3, 8.5)
    paint(img, ell(F, 0, 3.6, 7, 1.8), darkc(n), U(n, 0.15, 0.5))
    paint(img, poly(F, [(-6, -1.5), (-9.5, -3.0), (-6, 0.8)]), col)
    paint(img, ell(F, 0, 0, 7, 3.2), col, tex=furtex(n, 0.03, 0.12))
    paint(img, ell(F, -1, -0.5, 4.5, 1.8), shade(col, 0.8), U(n, 0.2, 1.0))
    paint(img, cap(F, 4.8, -1.5, 6.0, -nk, 1.0), col)
    paint(img, ell(F, 6.8, -nk - 0.8, 1.8, 1.5), np.where(B(n, 0.3)[:, None] > 0, PAL(n, BIRDC), col))
    paint(img, poly(F, [(8.2, -nk - 1.4), (11.0, -nk - 0.4), (8.2, -nk)]), PAL(n, BEAKC))
    return finish(img)


def sc_bird_big(n):
    """ostrich / emu / heron: round body on two long legs, long neck, small head"""
    img, hy = mix(n, [(6, bgl([(3, "grass"), (2, "dry"), (2, "dirt"), (1, "sand")], 5, 18, 0.6)), (2, bg_bokeh), (1.5, bgc("grass", 0.06, 0.2)), (1, bgc("sand", 0.04, 0.15))])
    s = U(n, 0.7, 1.2)
    F = Frame(U(n, 11, 21), U(n, 22, 30) - 14 * s, s, Nn(n, 0.05), SG(n))
    col = PAL(n, [(3, (0.25, 0.22, 0.20)), (2, (0.45, 0.40, 0.35)), (2, (0.10, 0.10, 0.10)), (1, (0.60, 0.58, 0.55)), (1, (0.90, 0.90, 0.88))])
    neckc = PAL(n, [(2, (0.70, 0.60, 0.55)), (2, (0.50, 0.48, 0.46)), (1, (0.30, 0.45, 0.70)), (1, (0.85, 0.82, 0.78))])
    legc = PAL(n, [(2, (0.65, 0.58, 0.52)), (1, (0.35, 0.32, 0.30)), (1, (0.80, 0.60, 0.50))])
    nl = U(n, 7, 14)
    paint(img, ell(F, 0, 14.3, 6, 1.0), darkc(n), U(n, 0.1, 0.5))
    paint(img, cap(F, -0.5, 4, -1.0 + Nn(n, 1), 14, 0.7), legc)
    paint(img, cap(F, 1.5, 4, 2.0 + Nn(n, 1), 14, 0.7), legc)
    paint(img, cap(F, 5.0, -3.0, 7.5, -nl, 1.2, 0.9), neckc)
    paint(img, ell(F, 0, 0, 7, 5, 0.0, 1.5), col, tex=furtex(n, 0.08, 0.25))
    paint(img, ell(F, 8.3, -nl - 0.8, 2.0, 1.4), neckc)
    paint(img, poly(F, [(9.8, -nl - 1.3), (12.5, -nl - 0.5), (9.8, -nl - 0.1)]), PAL(n, BEAKC))
    return finish(img)


def sc_bird_head(n):
    """close-up of a long-necked bird: thick neck rising to a head with a beak"""
    img, _ = mix(n, [(6, bg_bokeh), (1, bg_plain), (2, bg_sky), (1, bg_dark)])
    F = Frame(U(n, 10, 20), U(n, 9, 17), U(n, 0.8, 1.4), Nn(n, 0.15), SG(n))
    col = PAL(n, BIRDC + [(3, (0.60, 0.55, 0.50)), (2, (0.80, 0.75, 0.70))])
    tex = furtex(n, 0.05, 0.2)
    paint(img, cap(F, U(n, -6, 4), 26, 0, 1, U(n, 3.5, 6), 3.5, 1.5), shade(col, U(n, 0.8, 1.1)), tex=tex)
    paint(img, ell(F, 1, -3, U(n, 5.5, 7.5), U(n, 4.5, 6), 0.0, 1.5), col, tex=tex)
    bl = U(n, 5, 10)
    paint(img, poly(F, [(6, -4.5), (6 + bl, -1.5), (6, 0.8)]), PAL(n, BEAKC))
    paint(img, ell(F, 3.0, -4.5, 1.6, 1.6), whitec(n), B(n, 0.4))
    paint(img, ell(F, 3.0, -4.5, 1.1, 1.1), darkc(n))
    return finish(img)


# ---- classes 3 and 5: cat and dog ------------------------------------------------------------------------
CATC = [(3, (0.50, 0.47, 0.44)), (2.5, (0.78, 0.52, 0.25)), (2.5, (0.10, 0.10, 0.10)), (2, (0.92, 0.90, 0.86)),
        (1.5, (0.45, 0.33, 0.22)), (1.5, (0.82, 0.74, 0.60)), (1, (0.30, 0.30, 0.32))]
DOGC = [(3, (0.92, 0.90, 0.85)), (2.5, (0.78, 0.62, 0.40)), (2.5, (0.45, 0.30, 0.18)), (2.5, (0.10, 0.09, 0.09)),
        (1, (0.60, 0.58, 0.55)), (1.5, (0.62, 0.42, 0.25)), (1, (0.85, 0.78, 0.65))]
EYEC = [(2, (0.70, 0.65, 0.20)), (2, (0.45, 0.60, 0.25)), (1, (0.80, 0.55, 0.15)), (1, (0.35, 0.55, 0.70))]
PINK = (0.85, 0.55, 0.55)


def cat_bg(n):
    return mix(n, [(7.5, bg_indoor), (0.8, bgc("grass", 0.06, 0.2)), (0.5, bg_bokeh), (0.7, bg_plain), (0.5, bg_dark)])


def dog_bg(n):
    return mix(n, [(5, bg_indoor), (2, bgc("grass", 0.06, 0.2)), (1, bgl([(3, "grass"), (1, "dirt"), (1, "snow"), (1, "road")], 4, 14, 0.7)),
                   (0.8, bg_bokeh), (0.7, bg_plain), (0.5, bg_dark)])


def cat_head(img, F, n, x, y, rw, rh, coat, soft=1.3):
    """round head, two pointed ears on top, slanted eyes, small nose"""
    earin = C(n, PINK, 0.1, 0.03)
    for sg in (-1.0, 1.0):
        coat.put(img, poly(F, [(x + sg * rw * 0.95, y - rh * 0.30), (x + sg * rw * 0.20, y - rh * 0.88), (x + sg * rw * 0.85, y - rh * 1.65)]))
        paint(img, poly(F, [(x + sg * rw * 0.82, y - rh * 0.60), (x + sg * rw * 0.45, y - rh * 0.90), (x + sg * rw * 0.78, y - rh * 1.38)]), earin, 0.6 * np.ones(n))
    coat.put(img, ell(F, x, y, rw, rh, 0.0, soft))
    paint(img, ell(F, x, y + rh * 0.45, rw * 0.45, rh * 0.40), whitec(n), B(n, 0.4))
    eyec, dk = PAL(n, EYEC), darkc(n)
    for sg in (-1.0, 1.0):
        paint(img, ell(F, x + sg * rw * 0.42, y - rh * 0.12, rw * 0.20, rh * 0.17), eyec)
        paint(img, ell(F, x + sg * rw * 0.42, y - rh * 0.12, rw * 0.07, rh * 0.15), dk)
    paint(img, poly(F, [(x - rw * 0.10, y + rh * 0.25), (x + rw * 0.10, y + rh * 0.25), (x, y + rh * 0.42)]), C(n, PINK, 0.15, 0.03))


def dog_head(img, F, n, x, y, rw, rh, coat, soft=1.3):
    """head with hanging or pricked ears, lighter muzzle, big dark nose, dark eyes"""
    floppy = B(n, 0.65)
    earf = np.where(B(n, 0.6) > 0, U(n, 0.45, 0.8), 1.0)
    snout = np.where(B(n, 0.5)[:, None] > 0, whitec(n), shade(coat.col, U(n, 0.9, 1.3)))
    dk = darkc(n)
    for sg in (-1.0, 1.0):
        coat.put(img, ell(F, x + sg * rw * 0.95, y + rh * 0.10, rw * 0.38, rh * 0.75, -0.2 * sg, soft), earf, floppy)
        coat.put(img, poly(F, [(x + sg * rw * 1.0, y - rh * 0.30), (x + sg * rw * 0.30, y - rh * 0.85), (x + sg * rw * 0.95, y - rh * 1.70)]), earf, 1 - floppy)
    coat.put(img, ell(F, x, y, rw, rh, 0.0, soft))
    paint(img, ell(F, x, y - rh * 0.25, rw * 0.16, rh * 0.65), whitec(n), B(n, 0.25))
    paint(img, ell(F, x, y + rh * 0.45, rw * 0.50, rh * 0.48), snout)
    paint(img, ell(F, x, y + rh * 0.85, rw * 0.18, rh * 0.25), C(n, PINK, 0.1, 0.03), B(n, 0.3))
    paint(img, ell(F, x, y + rh * 0.28, rw * 0.22, rh * 0.17), dk)
    for sg in (-1.0, 1.0):
        paint(img, ell(F, x + sg * rw * 0.42, y - rh * 0.20, rw * 0.13, rh * 0.13), dk)


def pet_face(img, F, n, coat, headfn):
    """close-up: the head fills much of the frame, shoulders below"""
    rw = U(n, 7.5, 11)
    rh = rw * U(n, 0.80, 0.98)
    coat.put(img, ell(F, U(n, -3, 3), rh + 8, U(n, 11, 16), 10, 0.0, 1.5), U(n, 0.85, 1.05))
    headfn(img, F, n, 0.0, 0.0, rw, rh, coat)


def pet_sit(img, F, n, coat, headfn, hr):
    coat.put(img, cap(F, -4, 8.5, -11, 7 + Nn(n, 2), 1.2, None, 1.5))
    coat.put(img, ell(F, -4.5, 8.5, 4, 4, 0.0, 1.3), 0.92 * np.ones(n))
    coat.put(img, ell(F, 0, 3.5, 6.2, 8.5, 0.0, 1.3))
    paint(img, ell(F, 0.5, 2.5, 3.0, 5.5, 0.0, 1.5), whitec(n), B(n, 0.35))
    for lx in (-1.8, 2.2):
        coat.put(img, cap(F, lx, 5, lx, 12, 1.3), 0.95 * np.ones(n))
    headfn(img, F, n, U(n, -1.5, 2.0), -7.5, hr, hr * U(n, 0.82, 0.95), coat)


def pet_lie(img, F, n, coat, headfn, hr):
    coat.put(img, cap(F, -10, 3.5, -13.5, 6.5 - U(n, 0, 6), 1.1, None, 1.5))
    coat.put(img, ell(F, 0, 2, 11, U(n, 4.2, 6.0), 0.0, 1.3))
    wc = np.where(B(n, 0.4)[:, None] > 0, whitec(n), coat.col)
    paint(img, ell(F, 9.5, 6.0, 3.0, 1.3), wc)
    paint(img, ell(F, 5.0, 6.5, 3.0, 1.2), wc)
    headfn(img, F, n, U(n, 6.5, 9.5), -2.5, hr, hr * U(n, 0.82, 0.95), coat)


def pet_stand(img, F, n, coat, headfn, hr, leg, legr, snout=0.0):
    """side view facing +x, head turned to the camera"""
    for i, lx in enumerate((-7.0, -5.2, 5.0, 6.6)):
        coat.put(img, cap(F, lx, 2, lx + Nn(n, 0.8), 4 + leg, legr, legr * 0.8), (0.8 if i in (0, 3) else 1.0) * np.ones(n))
    coat.put(img, cap(F, -8.5, -1.5, -12.5 - U(n, 0, 2), -1.5 + Nn(n, 4), 0.9, None, 1.3))
    coat.put(img, ell(F, 0, 0, 9, U(n, 3.8, 4.8), 0.0, 1.3))
    coat.put(img, cap(F, 7, -1.5, 9.5, -4.5, 2.6))
    if snout > 0:
        coat.put(img, ell(F, 10.5 + hr * 0.8, -5.0, hr * 0.75 * snout, hr * 0.5), 1.1 * np.ones(n))
    headfn(img, F, n, 10.5, -5.5, hr, hr * 0.92, coat)


def catcoat(n):
    return Coat(n, CATC, ((3, (0.93, 0.92, 0.90)), (1, (0.10, 0.10, 0.10)), (1, (0.78, 0.52, 0.25))), 0.35, 0.05, 0.2, 0.5)


def dogcoat(n):
    return Coat(n, DOGC, ((3, (0.93, 0.92, 0.90)), (1.5, (0.10, 0.09, 0.09)), (1.5, (0.62, 0.42, 0.25))), 0.4, 0.05, 0.22, 0.0)


def sc_cat_face(n):
    img, _ = cat_bg(n)
    F = Frame(U(n, 12, 20), U(n, 11, 18), U(n, 0.8, 1.5), Nn(n, 0.2), SG(n))
    pet_face(img, F, n, catcoat(n), cat_head)
    return finish(img)


def sc_cat_sit(n):
    img, _ = cat_bg(n)
    F = Frame(U(n, 12, 20), U(n, 14, 19), U(n, 0.85, 1.35), Nn(n, 0.12), SG(n))
    pet_sit(img, F, n, catcoat(n), cat_head, U(n, 4.3, 5.5))
    return finish(img)


def sc_cat_lie(n):
    img, _ = cat_bg(n)
    F = Frame(U(n, 12, 20), U(n, 13, 21), U(n, 0.8, 1.3), Nn(n, 0.25), SG(n))
    pet_lie(img, F, n, catcoat(n), cat_head, U(n, 3.8, 5.0))
    return finish(img)


def sc_cat_stand(n):
    img, _ = cat_bg(n)
    s = U(n, 0.8, 1.3)
    F = Frame(U(n, 12, 19), U(n, 12, 19), s, Nn(n, 0.1), SG(n))
    pet_stand(img, F, n, catcoat(n), cat_head, U(n, 3.4, 4.2), U(n, 4.5, 6.5), 1.1)
    return finish(img)


def sc_dog_face(n):
    img, _ = dog_bg(n)
    F = Frame(U(n, 12, 20), U(n, 11, 18), U(n, 0.8, 1.5), Nn(n, 0.2), SG(n))
    pet_face(img, F, n, dogcoat(n), dog_head)
    return finish(img)


def sc_dog_sit(n):
    img, _ = dog_bg(n)
    F = Frame(U(n, 12, 20), U(n, 14, 19), U(n, 0.85, 1.35), Nn(n, 0.1), SG(n))
    pet_sit(img, F, n, dogcoat(n), dog_head, U(n, 4.6, 6.0))
    return finish(img)


def sc_dog_lie(n):
    img, _ = dog_bg(n)
    F = Frame(U(n, 12, 20), U(n, 13, 21), U(n, 0.8, 1.3), Nn(n, 0.2), SG(n))
    pet_lie(img, F, n, dogcoat(n), dog_head, U(n, 4.2, 5.6))
    return finish(img)


def sc_dog_stand(n):
    img, _ = dog_bg(n)
    s = U(n, 0.8, 1.25)
    F = Frame(U(n, 12, 19), U(n, 10, 17), s, Nn(n, 0.08), SG(n))
    pet_stand(img, F, n, dogcoat(n), dog_head, U(n, 3.5, 4.4), U(n, 6.5, 9.5), 1.3, 1.0)
    return finish(img)


# ---- classes 4 and 7: deer and horse ---------------------------------------------------------------------
DEERC = [(4, (0.55, 0.40, 0.25)), (3, (0.45, 0.32, 0.20)), (2, (0.65, 0.48, 0.30)), (1.5, (0.35, 0.26, 0.18)),
         (1, (0.50, 0.45, 0.38)), (1, (0.62, 0.35, 0.18))]
HORSEC = [(4, (0.45, 0.25, 0.12)), (3, (0.32, 0.18, 0.10)), (2, (0.10, 0.09, 0.09)), (2, (0.90, 0.89, 0.86)),
          (1.5, (0.60, 0.58, 0.56)), (1.5, (0.72, 0.55, 0.32)), (1, (0.58, 0.32, 0.15))]
ANTLERC = [(2, (0.70, 0.62, 0.48)), (1, (0.45, 0.36, 0.26)), (1, (0.85, 0.80, 0.70))]


def deer_bg(n):
    return mix(n, [(4, bgl([(4, "grass"), (2.5, "dry"), (1, "dirt"), (1, "snow")], 4, 17, 0.85)), (3, bg_forest), (2, bg_bokeh),
                   (2.5, bg_brush), (1, bgc("grass", 0.06, 0.2)), (0.4, bg_plain)])


def horse_bg(n):
    return mix(n, [(5, bgl([(5, "grass"), (1.5, "dry"), (1.5, "dirt"), (0.7, "snow")], 8, 20, 0.75)), (2.5, bg_arena), (0.8, bg_dark),
                   (1.5, bg_bokeh), (0.5, bg_plain), (0.6, bg_forest), (1.0, bgc("grass", 0.06, 0.2)), (0.5, bg_brush)])


def antlers(img, F, n, x, y, k, col, a):
    for dx in (0.0, 1.6):
        paint(img, cap(F, x + dx, y, x + dx - 1.5 * k, y - 7 * k, 0.45), col, a)
        paint(img, cap(F, x + dx - 0.8 * k, y - 4 * k, x + dx + 2.5 * k, y - 7.5 * k, 0.4), col, a)
        paint(img, cap(F, x + dx - 0.4 * k, y - 2 * k, x + dx - 3.5 * k, y - 4.5 * k, 0.4), col, a)


def deer_side(img, F, n, coat):
    """slim body on four thin legs, upright neck, small head, big ears, optional antlers; feet at y = 12.5"""
    graze = B(n, 0.15)
    nx = np.where(graze > 0, 10.5, 9.3 + Nn(n, 0.6))
    ny = np.where(graze > 0, 5.5, -8.5 + Nn(n, 0.8))
    hx, hy = nx + 1.5, ny - 1.3 + 3.0 * graze
    for i, lx in enumerate((-6.4, -5.0, 5.0, 6.3)):
        coat.put(img, cap(F, lx, 2.5, lx + Nn(n, 1.0), 12.5, 0.75, 0.5), (0.75 if i in (0, 3) else 1.0) * np.ones(n))
    coat.put(img, ell(F, 0, 0, 8, U(n, 3.4, 4.3), 0.0, 1.2))
    paint(img, ell(F, 0, 2.3, 6.5, 1.5), shade(coat.col, 1.35), U(n, 0.0, 0.7))
    paint(img, ell(F, -8.0, -1.2, 1.3, 1.6), whitec(n), B(n, 0.6))
    coat.put(img, cap(F, 6.3, -1.5, nx, ny, 1.9, 1.4))
    coat.put(img, ell(F, hx, hy, 2.9, 1.8, 0.35 + 0.9 * graze))
    coat.put(img, ell(F, hx - 2.1, hy - 2.0, 0.8, 2.1, -0.5))
    paint(img, ell(F, hx + 2.3, hy + 0.8 + graze, 0.7, 0.6), darkc(n))
    antlers(img, F, n, hx - 1.3, hy - 1.8, U(n, 0.7, 1.2), PAL(n, ANTLERC), B(n, 0.45) * (1 - graze))


def deer_front(img, F, n, coat):
    """facing the camera: narrow body, legs below, head on top with ears out sideways"""
    for lx, f in ((-1.0, 0.7), (1.0, 0.7), (-2.3, 1.0), (2.3, 1.0)):
        coat.put(img, cap(F, lx, 6, lx, 15, 0.75, 0.5), f * np.ones(n))
    coat.put(img, ell(F, 0, 2, U(n, 3.6, 4.8), 6, 0.0, 1.2))
    paint(img, ell(F, 0, 0, 2.2, 3.0), shade(coat.col, 1.3), U(n, 0.0, 0.7))
    coat.put(img, cap(F, 0, -3, 0, -8, 2.0))
    for sg in (-1.0, 1.0):
        coat.put(img, ell(F, sg * 4.0, -12.0, 2.4, 1.1, sg * 0.45))
    coat.put(img, ell(F, 0, -10.5, 2.4, 3.2))
    paint(img, ell(F, 0, -8.4, 0.9, 0.7), darkc(n))
    a, k, ac = B(n, 0.45), U(n, 0.7, 1.2), PAL(n, ANTLERC)
    for sg in (-1.0, 1.0):
        paint(img, cap(F, sg * 1.0, -13, sg * 5 * k, -13 - 6 * k, 0.45), ac, a)
        paint(img, cap(F, sg * 3 * k, -13 - 3 * k, sg * 2.5 * k, -13 - 7.5 * k, 0.4), ac, a)
        paint(img, cap(F, sg * 4.2 * k, -13 - 5 * k, sg * 7 * k, -13 - 7 * k, 0.4), ac, a)


def deer_head(img, F, n, coat):
    coat.put(img, ell(F, U(n, -4, 4), 19, U(n, 8, 12), 11, 0.0, 1.4), U(n, 0.85, 1.05))
    coat.put(img, cap(F, 0, 12, 0, 2, 4.0, 3.2))
    for sg in (-1.0, 1.0):
        coat.put(img, ell(F, sg * 8.5, -5.0, 5.0, 2.3, sg * 0.45))
        paint(img, ell(F, sg * 8.5, -5.0, 3.6, 1.2, sg * 0.45), shade(coat.col, 1.4), 0.6 * np.ones(n))
    coat.put(img, ell(F, 0, 0, 5.3, 7.5))
    coat.put(img, ell(F, 0, 5.5, 3.3, 4.2), 1.1 * np.ones(n))
    dk = darkc(n)
    for sg in (-1.0, 1.0):
        paint(img, ell(F, sg * 3.4, -2.0, 1.1, 1.0), dk)
    paint(img, ell(F, 0, 8.3, 2.0, 1.4), dk)
    a, k, ac = B(n, 0.4), U(n, 0.8, 1.3), PAL(n, ANTLERC)
    for sg in (-1.0, 1.0):
        paint(img, cap(F, sg * 2.0, -7, sg * 7 * k, -7 - 8 * k, 0.7), ac, a)
        paint(img, cap(F, sg * 4.5 * k, -7 - 4 * k, sg * 3.5 * k, -7 - 10 * k, 0.6), ac, a)


def deercoat(n):
    return Coat(n, DEERC, ((1, (0.93, 0.92, 0.90)),), 0.08, 0.04, 0.15)


def sc_deer_side(n):
    img, _ = deer_bg(n)
    s = U(n, 0.5, 1.1)
    F = Frame(U(n, 10, 20), U(n, 21, 30.5) - 12.5 * s, s, Nn(n, 0.05), SG(n))
    paint(img, ell(F, 0, 12.8, 8, 1.0), darkc(n), U(n, 0.0, 0.4))
    deer_side(img, F, n, deercoat(n))
    return finish(img)


def sc_deer_front(n):
    img, _ = deer_bg(n)
    s = U(n, 0.7, 1.25)
    F = Frame(U(n, 11, 21), U(n, 23, 31) - 15 * s, s, Nn(n, 0.06), SG(n))
    deer_front(img, F, n, deercoat(n))
    return finish(img)


def sc_deer_head(n):
    img, _ = deer_bg(n)
    F = Frame(U(n, 11, 21), U(n, 11, 18), U(n, 0.75, 1.4), Nn(n, 0.15), SG(n))
    deer_head(img, F, n, deercoat(n))
    return finish(img)


def horse_side(img, F, n, coat, rider):
    """heavy body on four sturdy legs, thick sloping neck, long head, mane and hanging tail; feet at y = 14"""
    dark = np.where(B(n, 0.55)[:, None] > 0, darkc(n), shade(coat.col, 0.7))
    sock = B(n, 0.3)
    for i, lx in enumerate((-7.6, -6.0, 6.0, 7.6)):
        ex = lx + Nn(n, 1.3)
        coat.put(img, cap(F, lx, 3, ex, 14, 1.3, 0.9), (0.75 if i in (0, 3) else 1.0) * np.ones(n))
        paint(img, cap(F, lx + (ex - lx) * 0.7, 10.7, ex, 14, 0.95), whitec(n), sock)
    paint(img, cap(F, -10.8, -2.5, -12.8 - U(n, 0, 2), 6 + Nn(n, 2), 1.6, 0.8, 1.4), dark)
    coat.put(img, ell(F, 0, 0, 9.5, 5.0, 0.0, 1.2))
    coat.put(img, ell(F, 6.5, -0.3, 4.5, 5.2, 0.0, 1.2))
    coat.put(img, ell(F, -6.5, -0.5, 4.8, 5.2, 0.0, 1.2))
    nx, ny = 11.5 + Nn(n, 0.8), -9.5 + Nn(n, 1.2)
    coat.put(img, cap(F, 7.5, -2.5, nx, ny, 3.2, 2.2))
    paint(img, cap(F, 6.8, -5.5, nx - 0.5, ny - 2.0, 1.0, None, 1.3), dark)
    coat.put(img, ell(F, nx + 1.9, ny + 0.7, 4.0, 2.0, 0.95))
    coat.put(img, poly(F, [(nx - 0.6, ny - 1.5), (nx + 0.6, ny - 1.8), (nx - 0.2, ny - 3.6)]))
    # rider: torso, head, leg, saddle
    shirt = PAL(n, [(2, (0.90, 0.90, 0.90)), (2, (0.75, 0.12, 0.12)), (2, (0.15, 0.25, 0.60)), (2, (0.10, 0.10, 0.12)), (1, (0.9, 0.8, 0.2))])
    paint(img, rbox(F, 0.5, -5.0, 2.6, 1.0, 0.5), darkc(n), rider)
    paint(img, cap(F, 0.8, -5.5, 2.3, 0.5, 1.0), PAL(n, [(2, (0.12, 0.12, 0.15)), (1, (0.85, 0.82, 0.75)), (1, (0.25, 0.3, 0.5))]), rider)
    paint(img, rbox(F, 0.5, -9.3, 1.9, 3.3, 1.0), shirt, rider)
    paint(img, ell(F, 0.9, -14.2, 1.6, 1.6), np.where(B(n, 0.6)[:, None] > 0, darkc(n), C(n, (0.85, 0.68, 0.55), 0.08, 0.02)), rider)


def horse_front(img, F, n, coat):
    dark = np.where(B(n, 0.55)[:, None] > 0, darkc(n), shade(coat.col, 0.7))
    for lx, f in ((-1.2, 0.7), (1.2, 0.7), (-2.6, 1.0), (2.6, 1.0)):
        coat.put(img, cap(F, lx, 5, lx, 15.5, 1.3, 1.0), f * np.ones(n))
    coat.put(img, ell(F, 0, 1, U(n, 4.5, 5.8), 6.5, 0.0, 1.2))
    coat.put(img, cap(F, 0.5, -4, 1.5, -10, 2.9, 2.2))
    paint(img, cap(F, -1.5, -4, -0.5, -11, 1.0, None, 1.3), dark, B(n, 0.6))
    for sg in (-1.0, 1.0):
        coat.put(img, poly(F, [(2 + sg * 1.6, -14.5), (2 + sg * 0.6, -15), (2 + sg * 1.5, -17)]))
    coat.put(img, ell(F, 2, -11.5, 2.3, 4.3, -0.1))
    paint(img, ell(F, 2.1, -11.5, 0.7, 3.4, -0.1), whitec(n), B(n, 0.4))


def horse_head(img, F, n, coat):
    """close-up: thick neck from a lower corner, long head pointing down and forward, ears, mane"""
    dark = np.where(B(n, 0.55)[:, None] > 0, darkc(n), shade(coat.col, 0.7))
    coat.put(img, cap(F, -13, 16, 0, -2, U(n, 6.5, 9), 5.0, 1.3))
    paint(img, cap(F, -15, 6, -1.5, -7.5, 2.0, 1.2, 1.5), dark, B(n, 0.8))
    for dx in (-1.6, 0.6):
        coat.put(img, poly(F, [(dx - 0.6, -6.5), (dx + 1.4, -7.2), (dx + 0.2, -11.5)]))
    coat.put(img, ell(F, 5, 2, 4.6, 9.5, -0.62))
    coat.put(img, ell(F, 9.8, 9.5, 3.4, 3.0), np.where(B(n, 0.5) > 0, 0.7, 1.15))
    paint(img, ell(F, 5.2, 2.0, 1.2, 7.0, -0.62), whitec(n), B(n, 0.4))
    paint(img, ell(F, 2.6, -2.6, 1.0, 0.9), darkc(n))
    paint(img, ell(F, 11.0, 10.0, 0.7, 0.6), darkc(n))
    br = B(n, 0.35)
    bc = PAL(n, [(2, (0.12, 0.10, 0.08)), (1, (0.45, 0.28, 0.15)), (1, (0.7, 0.1, 0.1)), (1, (0.15, 0.2, 0.5))])
    paint(img, cap(F, 5.5, 5.5, 10.5, 3.0, 0.45), bc, br)
    paint(img, cap(F, 1.0, -5.0, 8.0, 6.5, 0.4), bc, br)


def horsecoat(n):
    return Coat(n, HORSEC, ((3, (0.93, 0.92, 0.90)), (1, (0.35, 0.20, 0.10))), 0.12, 0.03, 0.12)


def sc_horse_side(n):
    img, _ = horse_bg(n)
    s = U(n, 0.7, 1.25)
    F = Frame(U(n, 11, 21), U(n, 24, 32) - 14 * s, s, Nn(n, 0.05), SG(n))
    paint(img, ell(F, 0, 14.4, 10, 1.1), darkc(n), U(n, 0.0, 0.5))
    horse_side(img, F, n, horsecoat(n), B(n, 0.33))
    return finish(img)


def sc_horse_front(n):
    img, _ = horse_bg(n)
    s = U(n, 0.75, 1.25)
    F = Frame(U(n, 11, 21), U(n, 24, 31) - 15.5 * s, s, Nn(n, 0.05), SG(n))
    horse_front(img, F, n, horsecoat(n))
    return finish(img)


def sc_horse_head(n):
    img, _ = mix(n, [(3, bg_sky), (3, bg_bokeh), (2, bg_dark), (2, bgl([(3, "grass"), (1, "dirt")], 12, 26, 0.8)), (1, bg_plain), (1.5, bg_arena)])
    F = Frame(U(n, 9, 19), U(n, 10, 18), U(n, 0.75, 1.35), Nn(n, 0.15), SG(n))
    horse_head(img, F, n, horsecoat(n))
    return finish(img)


# ---- class 6: frog ---------------------------------------------------------------------------------------
FROGC = [(4, (0.35, 0.58, 0.15)), (3, (0.45, 0.50, 0.20)), (3, (0.42, 0.33, 0.20)), (2, (0.55, 0.48, 0.32)),
         (1.5, (0.28, 0.30, 0.16)), (1, (0.50, 0.50, 0.46)), (0.6, (0.80, 0.45, 0.12)), (0.4, (0.75, 0.15, 0.10)),
         (0.3, (0.20, 0.35, 0.75)), (0.6, (0.75, 0.70, 0.25))]


def frog_bg(n):
    return mix(n, [(2.5, bgc("leaf", 0.05, 0.2)), (2, bgc("soil")), (1.5, bgc("litter")), (1, bgc("rock")), (1.2, bgc("pond", 0.1, 0.3)),
                   (1.2, bg_bokeh), (1, bg_plain), (0.6, bg_dark), (0.6, bgc("grass", 0.08, 0.25)), (0.8, bg_brush),
                   (0.4, lambda m: (tex_fill(m, C(m, (0.82, 0.62, 0.50), 0.1, 0.03), 0.03, 0.10), np.zeros(m)))])


def frogcoat(n):
    c = Coat(n, FROGC, ((2, (0.15, 0.14, 0.10)), (1, (0.25, 0.35, 0.12)), (1, (0.60, 0.55, 0.35))), 0.45, 0.05, 0.25)
    c.pm = np.clip(noise(n, 8) * 2.0 + A(U(n, -1.2, 0.2)), 0, 1) * A(B(n, 0.45))     # blotches rather than large patches
    return c


def frog_eye(img, F, n, coat, x, y, r):
    coat.put(img, ell(F, x, y, r, r), 0.85 * np.ones(n))
    paint(img, ell(F, x, y, r * 0.6, r * 0.6), PAL(n, [(2, (0.75, 0.60, 0.20)), (1, (0.60, 0.25, 0.10)), (2, (0.12, 0.10, 0.08))]))
    paint(img, ell(F, x, y, r * 0.5, r * 0.3), darkc(n))


def frog_top(img, F, n, coat):
    """seen from above / behind: wide body, head toward -y, bulging eyes, folded hind legs at the sides"""
    for sg in (-1.0, 1.0):
        coat.put(img, ell(F, sg * 8.5, 4.0, 5.0, 2.5, sg * 0.9), 0.9 * np.ones(n))
        coat.put(img, ell(F, sg * 10.5, 3.0, 4.5, 1.6, -sg * 0.7), 0.85 * np.ones(n))
        coat.put(img, cap(F, sg * 9.5, 7.5, sg * 13.5, 9.5, 0.8), 0.85 * np.ones(n))
        coat.put(img, cap(F, sg * 5.5, -3.0, sg * 9.5, -7.0, 1.2), 0.9 * np.ones(n))
    coat.put(img, ell(F, 0, 0, U(n, 6.5, 8.5), U(n, 6.0, 7.5), 0.0, 1.2))
    coat.put(img, ell(F, 0, -5, 6, 4.5, 0.0, 1.2))
    paint(img, cap(F, -2.5, -4, -3.5, 5, 0.5, None, 1.5), shade(coat.col, 1.35), B(n, 0.3))
    paint(img, cap(F, 2.5, -4, 3.5, 5, 0.5, None, 1.5), shade(coat.col, 1.35), B(n, 0.3))
    for sg in (-1.0, 1.0):
        frog_eye(img, F, n, coat, sg * 4.0, -6.5, 1.9)


def frog_side(img, F, n, coat):
    """sitting in profile facing +x: sloping body, head high, eye on top, pale throat, folded hind leg"""
    belly = PAL(n, [(3, (0.90, 0.88, 0.80)), (2, (0.85, 0.80, 0.45)), (1, (0.70, 0.65, 0.55))])
    paint(img, ell(F, 0, 6.2, 10, 1.3), darkc(n), U(n, 0.1, 0.5))
    coat.put(img, cap(F, 5.0, 2.0, 6.0, 7.5, 1.3), 0.9 * np.ones(n))
    coat.put(img, cap(F, 6.0, 7.3, 8.5, 7.6, 0.6), 0.9 * np.ones(n))
    coat.put(img, ell(F, 0, 0, 9, 5, -0.35, 1.2))
    coat.put(img, ell(F, 6.5, -3.5, 4.5, 3.5, 0.0, 1.2))
    paint(img, ell(F, 5.0, 0.8, 5.5, 2.6, -0.2, 1.3), belly, U(n, 0.3, 1.0))
    coat.put(img, ell(F, -5.0, 2.5, 5.5, 2.8, 0.5), 0.85 * np.ones(n))
    coat.put(img, ell(F, -3.5, 5.6, 5.5, 1.4, -0.2), 0.8 * np.ones(n))
    paint(img, cap(F, 5.0, -2.2, 10.5, -2.8, 0.3), darkc(n), 0.6 * np.ones(n))
    frog_eye(img, F, n, coat, 6.3, -6.0, 2.0)


def frog_face(img, F, n, coat):
    """facing the camera: broad low body, two eyes on top, wide mouth line, pale throat"""
    belly = PAL(n, [(3, (0.90, 0.88, 0.80)), (2, (0.85, 0.80, 0.45)), (1, (0.70, 0.65, 0.55))])
    for sg in (-1.0, 1.0):
        coat.put(img, ell(F, sg * 11, 6, 3, 4), 0.85 * np.ones(n))
        coat.put(img, cap(F, sg * 6.5, 5, sg * 8, 10.5, 1.5), 0.9 * np.ones(n))
    coat.put(img, ell(F, 0, 2, 9.5, 7, 0.0, 1.2))
    paint(img, ell(F, 0, 5.2, 7.5, 3.8, 0.0, 1.3), belly, U(n, 0.3, 1.0))
    paint(img, cap(F, -7, 1, 7, 1, 0.4), darkc(n), 0.7 * np.ones(n))
    for sg in (-1.0, 1.0):
        frog_eye(img, F, n, coat, sg * 5.0, -5.0, 2.6)


def sc_frog_top(n):
    img, _ = frog_bg(n)
    F = Frame(U(n, 11, 21), U(n, 12, 20), U(n, 0.75, 1.35), Nn(n, 0.9), SG(n), U(n, 0.7, 1.0))
    paint(img, ell(F, 1, 2, 10, 8, 0.0, 3.0), darkc(n), U(n, 0.0, 0.4))
    frog_top(img, F, n, frogcoat(n))
    return finish(img)


def sc_frog_side(n):
    img, _ = frog_bg(n)
    F = Frame(U(n, 11, 20), U(n, 13, 20), U(n, 0.85, 1.5), Nn(n, 0.15), SG(n))
    frog_side(img, F, n, frogcoat(n))
    return finish(img)


def sc_frog_face(n):
    img, _ = frog_bg(n)
    F = Frame(U(n, 12, 20), U(n, 12, 19), U(n, 0.8, 1.4), Nn(n, 0.12), SG(n))
    frog_face(img, F, n, frogcoat(n))
    return finish(img)


# ----------------------------------------------------------------------------------------------------------
# the ten classes as mixtures of scenes
# ----------------------------------------------------------------------------------------------------------
SCENES = [
    [(0.28, sc_plane_top), (0.25, sc_plane_side), (0.22, sc_plane_ground), (0.17, sc_plane_34), (0.08, sc_plane_front)],                 # 0 airplane
    [(0.30, sc_car_side), (0.45, sc_car_34), (0.25, sc_car_front)],                                                 # 1 automobile
    [(0.33, sc_bird_perch), (0.12, sc_bird_ground), (0.13, sc_bird_fly), (0.12, sc_bird_water), (0.17, sc_bird_big),
     (0.13, sc_bird_head)],                                                                                         # 2 bird
    [(0.35, sc_cat_face), (0.25, sc_cat_sit), (0.25, sc_cat_lie), (0.15, sc_cat_stand)],                            # 3 cat
    [(0.60, sc_deer_side), (0.22, sc_deer_front), (0.18, sc_deer_head)],                                            # 4 deer
    [(0.35, sc_dog_face), (0.20, sc_dog_sit), (0.20, sc_dog_lie), (0.25, sc_dog_stand)],                            # 5 dog
    [(0.45, sc_frog_top), (0.40, sc_frog_side), (0.15, sc_frog_face)],                                              # 6 frog
    [(0.60, sc_horse_side), (0.15, sc_horse_front), (0.25, sc_horse_head)],                                         # 7 horse
    [(0.45, sc_ship_side), (0.20, sc_ship_34), (0.12, sc_ship_sail), (0.23, sc_ship_speed)],                                            # 8 ship
    [(0.30, sc_truck_side), (0.50, sc_truck_34), (0.20, sc_truck_front)],                                           # 9 truck
]


def render_class(c, n):
    """n scenes of class c, in a fixed random order"""
    table = SCENES[c]
    counts = [int(round(w * n)) for w, _ in table]
    counts[0] += n - sum(counts)
    imgs = np.concatenate([fn(k) for k, (_, fn) in zip(counts, table) if k > 0])
    return imgs[RNG.permutation(n)]


# ----------------------------------------------------------------------------------------------------------
# the network: fixed front end from formulas, then one row of weights per rendered scene
# ----------------------------------------------------------------------------------------------------------
N_ORI = 6            # edge orientations
EDGE_CLIP = 2.0      # edge energy is capped at this many standard deviations above the image's mean energy
CHROMA_FLOOR = 0.707 # semi-saturation constant of the chroma normalisation
LOCAL_FLOOR = 1.0    # floor of the per-cell orientation normalisation
GROUPS, PER_GROUP = 24, 500         # templates per class = GROUPS * PER_GROUP
# hand-chosen weights of the five feature blocks in the distance between an image and a template
BLOCK_W = {"A": 1.0, "B": 0.5, "C": 0.5, "S": 0.5, "E1": 0.5, "E2": 0.5}


def edge_filters():
    """Gaussian-derivative edge filters, 5x5, six orientations: narrow across the edge, longer along it"""
    r = np.arange(-2, 3, dtype=np.float64)
    y, x = np.meshgrid(r, r, indexing="ij")
    out = []
    for k in range(N_ORI):
        th = np.pi * k / N_ORI
        u = x * np.cos(th) + y * np.sin(th)
        v = -x * np.sin(th) + y * np.cos(th)
        g = -u / 0.9 ** 2 * np.exp(-u ** 2 / (2 * 0.9 ** 2) - v ** 2 / (2 * 1.6 ** 2))
        g = g - g.mean()
        out.append(g / np.abs(g).sum())
    return out


def front_end():
    """layers and weights of the fixed part of the network"""
    L, W = [], {}

    def add(name, op, inputs, **kw):
        L.append(dict(name=name, op=op, inputs=inputs, **kw))

    # --- colour at 8x8 -------------------------------------------------------------------------------------
    add("p8", "avgpool", ["image"], size=[4, 4], stride=[4, 4])
    # block A: opponent colours, absolute.  rows: brightness, red-green, blue-yellow, each divided by a typical spread
    add("fA", "conv", ["p8"], out=3, kernel=[1, 1], stride=1, padding=0)
    W["fA.W"] = np.array([[1 / 3, 1 / 3, 1 / 3], [1.0, -1.0, 0.0], [-0.5, -0.5, 1.0]]).reshape(3, 3, 1, 1) / np.array([0.30, 0.10, 0.15]).reshape(3, 1, 1, 1)
    W["fA.b"] = np.array([-0.5 / 0.30, 0.0, 0.0])
    # block B: brightness layout with the image's own mean removed and contrast normalised
    add("lum", "conv", ["p8"], out=1, kernel=[1, 1], stride=1, padding=0)
    W["lum.W"] = np.full((1, 3, 1, 1), 1 / 3)
    W["lum.b"] = np.zeros(1)
    add("fB", "norm", ["lum"])
    # block C: chroma layout divided by the image's own chroma strength (with a floor): x and -x and +-const
    # share one normalisation group, so the mean is zero by construction and only the scale is normalised
    add("chr", "conv", ["p8"], out=6, kernel=[1, 1], stride=1, padding=0)
    cw = np.zeros((6, 3, 1, 1))
    cw[0, :, 0, 0] = np.array([1.0, -1.0, 0.0]) / 0.10
    cw[1, :, 0, 0] = np.array([-0.5, -0.5, 1.0]) / 0.15
    cw[2], cw[3] = -cw[0], -cw[1]
    W["chr.W"] = cw
    W["chr.b"] = np.array([0, 0, 0, 0, CHROMA_FLOOR, -CHROMA_FLOOR])
    add("chr_r", "reshape", ["chr"], shape=[1, 48, 8])
    add("chr_n", "norm", ["chr_r"])
    add("chr_s", "reshape", ["chr_n"], shape=[6, 8, 8])
    add("fC", "conv", ["chr_s"], out=2, kernel=[1, 1], stride=1, padding=0)
    sw = np.zeros((2, 6, 1, 1))
    sw[0, 0] = sw[1, 1] = 1.0
    W["fC.W"] = sw
    W["fC.b"] = np.zeros(2)

    # block S: figure against ground.  |colour of the cell - mean colour of the border ring|, summed over the
    # three opponent channels, then normalised over the image: the layout of whatever stands out from the surround
    ring = np.zeros((8, 8))
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = 1.0
    D = np.eye(64) - np.ones((64, 1)) * (ring.reshape(1, 64) / ring.sum())
    dw = np.zeros((6 * 64, 3 * 64))
    for ch in range(3):
        for pol in range(2):
            dw[(ch * 2 + pol) * 64:(ch * 2 + pol + 1) * 64, ch * 64:(ch + 1) * 64] = D * (1 - 2 * pol)
    add("sal_d", "dense", ["fA"], out=6 * 64)
    W["sal_d.W"], W["sal_d.b"] = dw, np.zeros(6 * 64)
    add("sal_s", "reshape", ["sal_d"], shape=[6, 8, 8])
    add("sal_r", "relu", ["sal_s"])
    add("sal_m", "conv", ["sal_r"], out=1, kernel=[1, 1], stride=1, padding=0)
    W["sal_m.W"], W["sal_m.b"] = np.ones((1, 6, 1, 1)), np.zeros(1)
    add("fS", "norm", ["sal_m"])

    # --- oriented edge energy at two scales ----------------------------------------------------------------
    G = edge_filters()
    gw = np.zeros((N_ORI * 6, 3, 5, 5))
    mw = np.zeros((N_ORI, N_ORI * 6, 1, 1))
    for o in range(N_ORI):
        for ch in range(3):
            for pol in range(2):
                idx = (o * 3 + ch) * 2 + pol
                gw[idx, ch] = G[o] * (1 - 2 * pol)
                mw[o, idx] = 1.0
    for tag, src, size, pool, cells in (("1", "image", 28, 4, 7), ("2", "half", 12, 3, 4)):
        if tag == "2":
            add("half", "avgpool", ["image"], size=[2, 2], stride=[2, 2])
        add("g" + tag, "conv", [src], out=N_ORI * 6, kernel=[5, 5], stride=1, padding=0)
        W["g" + tag + ".W"], W["g" + tag + ".b"] = gw, np.zeros(N_ORI * 6)
        add("g" + tag + "r", "relu", ["g" + tag])
        add("e" + tag, "conv", ["g" + tag + "r"], out=N_ORI, kernel=[1, 1], stride=1, padding=0)       # |response| summed over colours
        W["e" + tag + ".W"], W["e" + tag + ".b"] = mw, np.zeros(N_ORI)
        add("e" + tag + "a", "reshape", ["e" + tag], shape=[1, N_ORI * size, size])
        add("e" + tag + "b", "norm", ["e" + tag + "a"])                                              # energy in units of this image's spread
        add("e" + tag + "c", "reshape", ["e" + tag + "b"], shape=[N_ORI, size, size])
        add("e" + tag + "d", "conv", ["e" + tag + "c"], out=N_ORI, kernel=[1, 1], stride=1, padding=0)  # EDGE_CLIP - x
        W["e" + tag + "d.W"] = -np.eye(N_ORI).reshape(N_ORI, N_ORI, 1, 1)
        W["e" + tag + "d.b"] = np.full(N_ORI, EDGE_CLIP)
        add("e" + tag + "e", "relu", ["e" + tag + "d"])                                              # = EDGE_CLIP - min(x, EDGE_CLIP)
        add("e" + tag + "p", "avgpool", ["e" + tag + "e"], size=[pool, pool], stride=[pool, pool])
        # local normalisation: within each cell, the six orientation values minus their mean, divided by their own
        # spread (two constant entries +-LOCAL_FLOOR join each cell's group and act as a floor).  What remains is
        # which orientation dominates where, not how strong or how textured the picture is.
        nc = cells * cells
        tw, tb = np.zeros((nc * (N_ORI + 2), N_ORI * nc)), np.zeros(nc * (N_ORI + 2))
        bw = np.zeros((N_ORI * nc, nc * (N_ORI + 2)))
        for cell in range(nc):
            for j in range(N_ORI):
                for o in range(N_ORI):
                    tw[cell * (N_ORI + 2) + j, o * nc + cell] = (1.0 if o == j else 0.0) - 1.0 / N_ORI
                bw[j * nc + cell, cell * (N_ORI + 2) + j] = 1.0
            tb[cell * (N_ORI + 2) + N_ORI], tb[cell * (N_ORI + 2) + N_ORI + 1] = LOCAL_FLOOR, -LOCAL_FLOOR
        add("e" + tag + "t", "dense", ["e" + tag + "p"], out=nc * (N_ORI + 2))
        W["e" + tag + "t.W"], W["e" + tag + "t.b"] = tw, tb
        add("e" + tag + "u", "reshape", ["e" + tag + "t"], shape=[nc, N_ORI + 2, 1])
        add("e" + tag + "v", "norm", ["e" + tag + "u"])
        add("e" + tag + "w", "dense", ["e" + tag + "v"], out=N_ORI * nc)
        W["e" + tag + "w.W"], W["e" + tag + "w.b"] = bw, np.zeros(N_ORI * nc)
        add("e" + tag + "q", "reshape", ["e" + tag + "w"], shape=[1, N_ORI * cells, cells])
        add("fE" + tag, "norm", ["e" + tag + "q"])
    return L, W


# ---- a small interpreter for the front end (same arithmetic as the scorer's), used to turn scenes into templates
def _conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def _pool(x, size, stride, reduce):
    n, c, h, w = x.shape
    (sh, sw), (th, tw) = size, stride
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if reduce == "max" else win.mean(0)


def run_layers(layers, weights, images, want):
    vals = {"image": images}
    for L in layers:
        x = vals[L["inputs"][0]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = _conv(x, weights[name + ".W"], weights[name + ".b"], L.get("stride", 1), L.get("padding", 0))
        elif op == "relu":
            vals[name] = np.maximum(x, 0)
        elif op == "dense":
            vals[name] = x.reshape(len(x), -1) @ weights[name + ".W"].T + weights[name + ".b"]
        elif op in ("maxpool", "avgpool"):
            vals[name] = _pool(x, L["size"], L["stride"], op[:3])
        elif op == "reshape":
            vals[name] = x.reshape((len(x),) + tuple(L["shape"]))
        elif op == "norm":
            mu = x.mean(axis=(2, 3), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        else:
            raise ValueError(op)
    return {k: vals[k].reshape(len(images), -1) for k in want}


# ---- the distance between an image and a template ---------------------------------------------------------
def pool_matrix(cells, win, stride):
    """(k*k, cells*cells) matrix averaging win x win windows of a cells x cells grid"""
    k = (cells - win) // stride + 1
    P = np.zeros((k * k, cells * cells))
    for i in range(k):
        for j in range(k):
            for dy in range(win):
                for dx in range(win):
                    P[i * k + j, (i * stride + dy) * cells + (j * stride + dx)] = 1.0 / (win * win)
    return P


def pyramid_metric(cells, levels):
    """M = mean over pyramid levels of P'P / (cells at that level): squared differences are compared at every
    level of a coarse-to-fine pyramid, so that a match in coarse layout counts even when details are displaced"""
    M = np.zeros((cells * cells, cells * cells))
    for win, stride in levels:
        P = pool_matrix(cells, win, stride)
        M += P.T @ P / len(P)
    return M / len(levels)


# block name -> (feature layer, channels, cells per side, pyramid levels as (window, stride))
BLOCKS = {
    "A": ("fA", 3, 8, [(1, 1), (2, 2), (4, 4), (8, 8)]),
    "B": ("fB", 1, 8, [(1, 1), (2, 2), (4, 4)]),
    "C": ("fC", 2, 8, [(1, 1), (2, 2), (4, 4), (8, 8)]),
    "S": ("fS", 1, 8, [(1, 1), (2, 2), (4, 4)]),
    "E1": ("fE1", N_ORI, 7, [(1, 1), (3, 2), (7, 7)]),
    "E2": ("fE2", N_ORI, 4, [(1, 1), (2, 2), (4, 4)]),
}


def build(out_dir, groups=GROUPS, per_group=PER_GROUP, verbose=True):
    layers, weights = front_end()
    per_class = groups * per_group
    T = 10 * per_class
    feats = {b: np.zeros((T, BLOCKS[b][1] * BLOCKS[b][2] ** 2)) for b in BLOCKS}
    want = [BLOCKS[b][0] for b in BLOCKS]
    for c in range(10):
        imgs = render_class(c, per_class)          # already in random order, so every group mixes all scene types
        for i in range(0, per_class, 500):
            f = run_layers(layers, weights, imgs[i:i + 500], want)
            for b in BLOCKS:
                feats[b][c * per_class + i: c * per_class + i + len(f[BLOCKS[b][0]])] = f[BLOCKS[b][0]]
        if verbose:
            print("rendered class", c, flush=True)
    # each template alone: weight row = M t (scaled), bias = -1/2 t'M t, so that
    # row . x + bias = -1/2 (x - t)' M (x - t) + (a term that depends on the image only)
    names = []
    for b, (layer, ch, cells, levels) in list(BLOCKS.items()):
        M = pyramid_metric(cells, levels)
        t = feats.pop(b).reshape(T, ch, cells * cells)
        tm = t @ M
        scale = BLOCK_W[b] / ch
        layers.append(dict(name="d" + b, op="dense", inputs=[layer], out=T))
        weights["d" + b + ".b"] = -0.5 * scale * (t * tm).sum(axis=(1, 2))
        weights["d" + b + ".W"] = (scale * tm).reshape(T, -1).astype(np.float32)
        del t, tm
        names.append("d" + b)
    layers.append(dict(name="sim", op="add", inputs=names))
    layers.append(dict(name="sim_r", op="reshape", inputs=["sim"], shape=[1, 10 * groups, per_group]))
    layers.append(dict(name="best", op="maxpool", inputs=["sim_r"], size=[1, per_group], stride=[1, per_group]))
    layers.append(dict(name="best_r", op="reshape", inputs=["best"], shape=[1, 10, groups]))
    layers.append(dict(name="cls", op="avgpool", inputs=["best_r"], size=[1, groups], stride=[1, groups]))
    layers.append(dict(name="out", op="reshape", inputs=["cls"], shape=[10]))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in weights.items()})
    if verbose:
        print("weights:", sum(int(v.size) for v in weights.values()))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
