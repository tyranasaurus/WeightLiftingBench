"""build.py - writes network.json and weights.npz for a hand-built 32x32 photo classifier.

Nothing here is trained or fitted and no file is read.  The network has two hand-made parts:

1. A fixed feature extractor built from formulas: soft hue bins (chroma-plane tents), lightness tents,
   and Gaussian-derivative orientation energies, each average-pooled on 4x4, 2x2 and 1x1 grids.
2. A bank of templates.  Each template is the feature vector of ONE picture drawn by the code below
   (sky / water / grass / road / room backgrounds plus an object assembled from ellipses, polygons and
   lines, all typed from general knowledge of what the ten classes look like).  Every drawing is used
   by itself as one template (weights = its own features, bias = -|its own features|^2 / 2); no
   statistic is ever taken across drawings.  The network measures how close the photo is to every
   template and lets the closest ones vote for their class.

Usage: python3 build.py OUT_DIR
"""
import json
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------------------------
# drawing primitives (pictures are drawn at 64x64 and shrunk to 32x32)
# ----------------------------------------------------------------------------------------------
S = 64
_yy, _xx = np.mgrid[0:S, 0:S]
X = (_xx + 0.5) / S
Y = (_yy + 0.5) / S
PI = np.pi


def rgb(*c):
    return np.array(c, dtype=np.float64)


def U(r, a, b):
    return a + (b - a) * r.random()


def choose(r, *a):
    items = a[0::2]
    w = np.cumsum(np.array(a[1::2], dtype=np.float64))
    return items[min(int(np.searchsorted(w / w[-1], r.random())), len(items) - 1)]


def snoise(r, nx, ny=None):
    """smooth random field with about nx by ny blobs across the picture, roughly unit variance"""
    ny = nx if ny is None else ny
    nx, ny = max(1, int(nx)), max(1, int(ny))
    g = r.standard_normal((ny + 3, nx + 3))
    fx = X[0] * nx + r.random()
    fy = Y[:, 0] * ny + r.random()
    ix, iy = fx.astype(int), fy.astype(int)
    tx, ty = fx - ix, fy - iy
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    gy0, gy1 = g[iy], g[iy + 1]
    top = gy0[:, ix] + (gy0[:, ix + 1] - gy0[:, ix]) * tx[None, :]
    bot = gy1[:, ix] + (gy1[:, ix + 1] - gy1[:, ix]) * tx[None, :]
    return 1.35 * (top + (bot - top) * ty[:, None])


def tex(r, fine=0.0, mid=0.0, coarse=0.0, nmid=12, ncoarse=4, aniso=1.0):
    """multiplicative texture field (mean 0)"""
    t = np.zeros((S, S))
    if fine:
        t = t + fine * r.standard_normal((S, S))
    if mid:
        t = t + mid * snoise(r, nmid, nmid * aniso)
    if coarse:
        t = t + coarse * snoise(r, ncoarse, ncoarse * aniso)
    return t


def cf(col, t):
    """colour field: colour times (1 + texture)"""
    return np.clip(np.asarray(col)[None, None, :] * (1 + t[..., None]), 0, 1)


def shd(r, col, cy=0.5, h=0.4, k=0.22, fine=0.05, mid=0.06, nmid=9):
    """object colour with light from above (lighter above cy, darker below) and some texture"""
    t = k * np.clip((cy - Y) / h, -1, 1) + tex(r, fine=fine, mid=mid, nmid=nmid)
    return cf(col, t)


def jc(r, col, j=0.04, lj=0.12):
    c = np.asarray(col, dtype=np.float64) * (1 + lj * (2 * r.random() - 1)) + j * (2 * r.random(3) - 1)
    return np.clip(c, 0, 1)


def ell(cx, cy, rx, ry, ang=0.0, soft=1.0):
    c, s = np.cos(ang), np.sin(ang)
    dx, dy = X - cx, Y - cy
    u = (dx * c + dy * s) / rx
    v = (-dx * s + dy * c) / ry
    q = np.sqrt(u * u + v * v)
    return np.clip((1 - q) * min(rx, ry) * S / soft + 0.5, 0, 1)


def seg(x0, y0, x1, y1, w0, w1=None):
    """thick line from (x0,y0) to (x1,y1); width w0 at the start and w1 at the end"""
    w1 = w0 if w1 is None else w1
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / (dx * dx + dy * dy + 1e-12), 0, 1)
    d = np.hypot(X - x0 - t * dx, Y - y0 - t * dy)
    return np.clip(((w0 + (w1 - w0) * t) / 2 - d) * S + 0.5, 0, 1)


def poly(pts):
    inside = np.zeros((S, S), dtype=bool)
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if y0 == y1:
            continue
        cond = (y0 > Y) != (y1 > Y)
        xint = (x1 - x0) * (Y - y0) / (y1 - y0) + x0
        inside ^= cond & (X < xint)
    return inside.astype(np.float64)


def rect(x0, y0, x1, y1):
    return ((X >= min(x0, x1)) & (X < max(x0, x1)) & (Y >= min(y0, y1)) & (Y < max(y0, y1))).astype(np.float64)


def put(im, m, col, a=1.0):
    im += (a * m)[..., None] * (np.asarray(col) - im)


def shade(im, m, f):
    im *= (1 - m * (1 - f))[..., None]


def boxblur(a, k):
    k = int(k)
    if k < 1:
        return a
    for ax in (0, 1):
        n = a.shape[ax]
        p = np.concatenate([np.repeat(a.take([0], ax), k + 1, ax), a, np.repeat(a.take([n - 1], ax), k, ax)], ax)
        c = np.cumsum(p, ax)
        a = (c.take(np.arange(2 * k + 1, 2 * k + 1 + n), ax) - c.take(np.arange(0, n), ax)) / (2 * k + 1)
    return a


class Xf:
    """local object coordinates -> picture coordinates (uniform scale, optional mirror, rotation, squash)"""

    def __init__(self, cx, cy, sc, flip=1, ang=0.0, sq=1.0):
        self.cx, self.cy, self.sc, self.f, self.ang, self.sq = cx, cy, sc, flip, ang, sq
        self.c, self.s = np.cos(ang), np.sin(ang)

    def pt(self, x, y):
        x, y = x * self.f, y * self.sq
        return self.cx + self.sc * (x * self.c - y * self.s), self.cy + self.sc * (x * self.s + y * self.c)

    def ell(self, x, y, rx, ry, a=0.0, soft=1.0):
        px, py = self.pt(x, y)
        return ell(px, py, rx * self.sc, ry * self.sc * self.sq, self.f * a + self.ang, soft)

    def poly(self, pts):
        return poly([self.pt(x, y) for x, y in pts])

    def rect(self, x0, y0, x1, y1):
        return self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def seg(self, x0, y0, x1, y1, w0, w1=None):
        p0, p1 = self.pt(x0, y0), self.pt(x1, y1)
        k = self.sc * (0.5 + 0.5 * self.sq)
        return seg(p0[0], p0[1], p1[0], p1[1], w0 * k, None if w1 is None else w1 * k)


# ----------------------------------------------------------------------------------------------
# backgrounds
# ----------------------------------------------------------------------------------------------
def sky(r, kind=None):
    kind = kind or choose(r, 'clear', 28, 'pale', 22, 'overcast', 25, 'clouds', 15, 'sunset', 4, 'dark', 6)
    if kind == 'clear':
        top = rgb(U(r, .15, .42), U(r, .38, .62), U(r, .72, .96))
        bot = top + (1 - top) * U(r, .25, .65)
    elif kind == 'pale':
        top = rgb(U(r, .5, .72), U(r, .66, .82), U(r, .82, .97))
        bot = top + (1 - top) * U(r, .2, .7)
    elif kind == 'overcast':
        v = U(r, .62, .97)
        top = rgb(v - U(r, 0, .04), v, min(1.0, v + U(r, 0, .05)))
        bot = top + (1 - top) * U(r, 0, .5)
    elif kind == 'clouds':
        top = rgb(U(r, .2, .45), U(r, .42, .65), U(r, .75, .95))
        bot = top + (1 - top) * U(r, .3, .7)
    elif kind == 'sunset':
        top = rgb(U(r, .25, .5), U(r, .25, .45), U(r, .4, .6))
        bot = rgb(U(r, .85, 1), U(r, .5, .75), U(r, .25, .5))
    else:
        v = U(r, .3, .55)
        top = rgb(v - .03, v, v + U(r, 0, .08))
        bot = top * U(r, 1.0, 1.3)
    im = top + (bot - top) * (Y ** U(r, .7, 1.5))[..., None]
    if kind == 'clouds' or (kind in ('clear', 'pale') and r.random() < .25):
        m = np.clip(snoise(r, U(r, 2, 4.9), U(r, 3, 7.9)) * U(r, .6, 1.2) + U(r, -.4, .3), 0, 1)
        put(im, m, rgb(1, 1, 1) * U(r, .85, 1.0))
    im = im * (1 + 0.02 * snoise(r, 3)[..., None])
    return np.clip(im, 0, 1)


def water(r, kind=None):
    kind = kind or choose(r, 'blue', 40, 'gray', 25, 'dark', 14, 'green', 13, 'bright', 8)
    if kind == 'blue':
        c = rgb(U(r, .08, .3), U(r, .25, .5), U(r, .45, .75))
    elif kind == 'gray':
        v = U(r, .35, .6)
        c = rgb(v - .05, v, v + U(r, .02, .1))
    elif kind == 'dark':
        c = rgb(U(r, .05, .15), U(r, .1, .22), U(r, .15, .32))
    elif kind == 'green':
        c = rgb(U(r, .15, .3), U(r, .3, .45), U(r, .25, .42))
    else:
        c = rgb(U(r, .6, .8), U(r, .7, .85), U(r, .8, .95))
    t = U(r, .04, .13) * snoise(r, U(r, 3, 8), U(r, 16, 40)) + U(r, .02, .07) * r.standard_normal((S, S)) \
        + U(r, .02, .08) * snoise(r, 3, 6)
    return cf(c, t + U(r, -.25, .25) * (Y - .6))


def ground(r, kind):
    if kind == 'water':
        return water(r)
    if kind == 'grass':
        c = rgb(U(r, .22, .48), U(r, .4, .62), U(r, .12, .3))
        t = tex(r, fine=U(r, .06, .14), mid=U(r, .05, .12), coarse=U(r, .03, .1), nmid=14)
    elif kind == 'drygrass':
        c = rgb(U(r, .5, .72), U(r, .46, .62), U(r, .25, .4))
        t = tex(r, fine=U(r, .06, .12), mid=U(r, .04, .1), coarse=U(r, .03, .08), nmid=14)
    elif kind == 'dirt':
        v = U(r, .38, .64)
        c = rgb(v, v * U(r, .68, .82), v * U(r, .42, .62))
        t = tex(r, fine=U(r, .05, .1), mid=U(r, .04, .1), coarse=U(r, .03, .08))
    elif kind == 'sand':
        c = rgb(U(r, .72, .9), U(r, .65, .8), U(r, .48, .65))
        t = tex(r, fine=U(r, .03, .06), mid=.03, coarse=.04)
    elif kind == 'road':
        v = U(r, .28, .56)
        c = rgb(v + U(r, -.02, .02), v, v + U(r, -.02, .04))
        t = tex(r, fine=U(r, .03, .07), mid=.03, coarse=U(r, .02, .06))
    elif kind == 'concrete':
        v = U(r, .55, .82)
        c = rgb(v + U(r, 0, .03), v, v - U(r, 0, .04))
        t = tex(r, fine=U(r, .02, .05), mid=.02, coarse=.04)
    else:  # snow
        v = U(r, .82, .98)
        c = rgb(v - .02, v, min(1.0, v + U(r, 0, .03)))
        t = tex(r, fine=.015, mid=.02, coarse=.04)
    return cf(c, t - U(r, 0, .25) * (Y - .7))


def treeline(r, im, hor, h=None, col=None):
    h = U(r, .06, .32) if h is None else h
    b = snoise(r, U(r, 3, 9.9), 1)[0]
    top = hor - h * np.clip(0.75 + 0.35 * b, 0.2, 1.4)
    m = ((Y > top[None, :]) & (Y <= hor)).astype(np.float64)
    if col is None:
        k = choose(r, 'green', 70, 'autumn', 12, 'far', 18)
        col = {'green': rgb(U(r, .1, .28), U(r, .2, .4), U(r, .06, .2)),
               'autumn': rgb(U(r, .35, .5), U(r, .25, .35), U(r, .1, .2)),
               'far': rgb(U(r, .25, .4), U(r, .35, .48), U(r, .35, .5))}[k]
    put(im, m, cf(col, tex(r, fine=.12, mid=.22, coarse=.1, nmid=10)))


def buildings(r, im, hor):
    x = U(r, -.25, .05)
    while x < 1:
        w, h, v = U(r, .15, .45), U(r, .1, .5), U(r, .35, .88)
        c = choose(r, rgb(v + U(r, 0, .08), v, v - U(r, 0, .08)), 60, rgb(v, v, v), 25, rgb(.55, .3, .24), 15)
        m = rect(x, hor - h, x + w, hor)
        put(im, m, cf(c, tex(r, fine=.03, coarse=.04)))
        if r.random() < .65:
            nc, nr = int(U(r, 2, 6)), int(U(r, 2, 6))
            wm = m * (((X - x) / w * nc) % 1 > .3) * (((X - x) / w * nc) % 1 < .75) \
                * (((Y - hor + h) / h * nr) % 1 > .3) * (((Y - hor + h) / h * nr) % 1 < .7)
            put(im, wm, c * U(r, .25, .6))
        x += w + U(r, 0, .06)


def hills(r, im, hor):
    b = snoise(r, U(r, 1, 3.9), 1)[0]
    h = U(r, .05, .3)
    top = hor - h * np.clip(0.6 + 0.5 * b, 0.05, 1.5)
    m = ((Y > top[None, :]) & (Y <= hor)).astype(np.float64)
    c = choose(r, rgb(U(r, .3, .45), U(r, .4, .5), U(r, .45, .6)), 40, rgb(U(r, .2, .4), U(r, .35, .5), U(r, .15, .3)), 40,
               rgb(U(r, .45, .6), U(r, .4, .5), U(r, .3, .4)), 20)
    put(im, m, cf(c, tex(r, fine=.04, mid=.08, coarse=.08)))


def outdoor(r, skyk=None, gk='grass', hor=None, back=None):
    """sky above the horizon, optional backdrop standing on the horizon, ground below"""
    im = sky(r, skyk)
    hor = U(r, .35, .65) if hor is None else hor
    if back == 'trees':
        treeline(r, im, hor)
    elif back == 'buildings':
        buildings(r, im, hor)
    elif back == 'hills':
        hills(r, im, hor)
    elif back == 'forest':
        treeline(r, im, hor, h=U(r, .5, 1.2))
    g = ground(r, gk)
    put(im, (Y > hor).astype(np.float64), g)
    return im, hor


def foliage(r, sharp=None):
    """out-of-focus or close-up vegetation: a few greens / browns mixed by smooth blobs"""
    def c(k):
        return {'dg': rgb(U(r, .08, .25), U(r, .18, .38), U(r, .04, .18)),
                'mg': rgb(U(r, .25, .45), U(r, .42, .62), U(r, .1, .3)),
                'yg': rgb(U(r, .5, .68), U(r, .58, .72), U(r, .18, .36)),
                'br': rgb(U(r, .32, .52), U(r, .22, .36), U(r, .1, .24)),
                'dk': rgb(U(r, .03, .12), U(r, .05, .14), U(r, .02, .1)),
                'sk': rgb(U(r, .6, .9), U(r, .75, .95), U(r, .85, 1.0)),
                'tn': rgb(U(r, .6, .78), U(r, .52, .65), U(r, .35, .48))}[k]
    im = np.zeros((S, S, 3)) + c(choose(r, 'dg', 35, 'mg', 40, 'yg', 10, 'br', 10, 'dk', 5))
    for _ in range(int(U(r, 1, 4.9))):
        k = choose(r, 'dg', 25, 'mg', 28, 'yg', 15, 'br', 14, 'dk', 8, 'sk', 6, 'tn', 4)
        m = np.clip(snoise(r, U(r, 1.5, 6)) * U(r, .8, 3) + U(r, -.8, .3), 0, 1)
        put(im, m, c(k))
    sharp = r.random() if sharp is None else sharp
    t = tex(r, fine=.1 * sharp, mid=.2 * sharp, nmid=U(r, 8, 16)) + .05 * snoise(r, 4)
    return np.clip(im * (1 + t[..., None]), 0, 1)


def earth(r):
    """soil, leaf litter, bark, rock, sand"""
    k = choose(r, 'brown', 40, 'tan', 18, 'dark', 12, 'gray', 20, 'red', 10)
    c = {'brown': rgb(U(r, .32, .55), U(r, .23, .4), U(r, .12, .28)), 'tan': rgb(U(r, .58, .78), U(r, .48, .64), U(r, .32, .48)),
         'dark': rgb(U(r, .12, .25), U(r, .09, .19), U(r, .05, .14)), 'gray': rgb(*([U(r, .35, .65)] * 3)) * rgb(1.03, 1, .95),
         'red': rgb(U(r, .5, .65), U(r, .28, .38), U(r, .18, .26))}[k]
    im = np.zeros((S, S, 3)) + c
    for _ in range(int(U(r, 0, 2.9))):
        m = np.clip(snoise(r, U(r, 2, 7)) * U(r, .8, 3) + U(r, -.8, .2), 0, 1)
        put(im, m, np.clip(c * U(r, .45, 1.5) + .06 * (2 * r.random(3) - 1), 0, 1))
    t = tex(r, fine=U(r, .04, .14), mid=U(r, .05, .22), nmid=U(r, 8, 18), coarse=.06)
    return np.clip(im * (1 + t[..., None]), 0, 1)


def roomcol(r, what):
    if what == 'wall':
        k = choose(r, 'white', 28, 'beige', 20, 'gray', 14, 'pastel', 12, 'wood', 10, 'dark', 10, 'strong', 6)
    else:
        k = choose(r, 'wood', 28, 'beige', 20, 'gray', 14, 'dark', 10, 'white', 8, 'strong', 12, 'pastel', 8)
    if k == 'white':
        v = U(r, .8, .97)
        return rgb(v, v - U(r, 0, .04), v - U(r, 0, .1))
    if k == 'beige':
        return rgb(U(r, .65, .85), U(r, .56, .72), U(r, .4, .58))
    if k == 'gray':
        v = U(r, .4, .72)
        return rgb(v + U(r, -.02, .03), v, v + U(r, -.04, .03))
    if k == 'pastel':
        return np.clip(rgb(.72, .72, .72) + .16 * (2 * r.random(3) - 1), 0, 1)
    if k == 'wood':
        return rgb(U(r, .4, .68), U(r, .26, .46), U(r, .12, .3))
    if k == 'dark':
        v = U(r, .08, .3)
        return rgb(v + U(r, 0, .05), v, v - U(r, 0, .04))
    return choose(r, rgb(U(r, .5, .7), U(r, .1, .22), U(r, .1, .2)), 35, rgb(U(r, .12, .28), U(r, .22, .4), U(r, .45, .65)), 30,
                  rgb(U(r, .2, .35), U(r, .4, .5), U(r, .2, .35)), 15, rgb(U(r, .75, .9), U(r, .6, .75), U(r, .15, .3)), 10,
                  rgb(U(r, .4, .55), U(r, .2, .3), U(r, .4, .6)), 10)


def indoor(r):
    """room: wall over floor, or one big piece of fabric / furniture, with a few boxes of other colours"""
    wall, floor = roomcol(r, 'wall'), roomcol(r, 'floor')
    im = cf(wall, tex(r, fine=.02, coarse=U(r, .02, .08)))
    if r.random() < .68:
        yb = U(r, .3, .85)
        fl = cf(floor, tex(r, fine=U(r, .02, .08), mid=U(r, 0, .08), nmid=U(r, 3, 8), aniso=U(r, 1, 6), coarse=.05))
        put(im, (Y > yb + U(r, -.1, .1) * (X - .5)).astype(np.float64), fl)
    else:
        put(im, np.clip(snoise(r, U(r, 1, 3)) * 2, 0, 1), np.clip(wall * U(r, .6, 1.2), 0, 1))
        im = np.clip(im * (1 + tex(r, fine=U(r, .02, .08), mid=U(r, .02, .1))[..., None]), 0, 1)
    for _ in range(int(U(r, 0, 3.6))):
        x0, y0 = U(r, -.2, .9), U(r, -.2, .8)
        m = rect(x0, y0, x0 + U(r, .12, .6), y0 + U(r, .12, .7))
        c = choose(r, roomcol(r, 'wall'), 50, roomcol(r, 'floor'), 40, rgb(.95, .95, .92), 10)
        put(im, m, cf(c, tex(r, fine=.03, coarse=.05)))
    im *= (1 + U(r, -.25, .25) * (X - .5) + U(r, -.2, .2) * (Y - .5))[..., None]
    return np.clip(im, 0, 1)


def plain(r, kind=None):
    """studio / cut-out background"""
    kind = kind or choose(r, 'white', 55, 'gray', 20, 'black', 10, 'color', 15)
    if kind == 'white':
        v = U(r, .86, 1.0)
        c = rgb(v, v, v)
    elif kind == 'gray':
        v = U(r, .4, .82)
        c = rgb(v + U(r, -.02, .03), v, v + U(r, -.03, .03))
    elif kind == 'black':
        v = U(r, .01, .14)
        c = rgb(v, v, v)
    else:
        c = np.clip(rgb(.55, .55, .55) + .35 * (2 * r.random(3) - 1), 0, 1)
    im = np.zeros((S, S, 3)) + c
    im *= (1 + U(r, -.12, .12) * (Y - .5) + U(r, -.08, .08) * (X - .5) + .02 * snoise(r, 2))[..., None]
    return np.clip(im, 0, 1)


def finish(r, im):
    """shrink to 32x32 and vary focus, exposure, contrast and colour cast like a photo"""
    k = choose(r, 0, 62, 1, 33, 2, 5)
    im = boxblur(im, k)
    im = im.reshape(32, 2, 32, 2, 3).mean(axis=(1, 3))
    im = im * (1 + U(r, -.12, .12) * (X[::2, ::2] - .5) + U(r, -.12, .12) * (Y[::2, ::2] - .5))[..., None]
    m = im.mean()
    im = m + (im - m) * U(r, .8, 1.2)
    im = im * U(r, .82, 1.15)
    cast = U(r, -.07, .07)
    im = im * rgb(1 + cast, 1 + U(r, -.03, .03), 1 - cast)
    sat = choose(r, 1.0, 50, U(r, .4, 1.0), 35, U(r, .1, .4), 12, 0.0, 3)      # photos are often dull, washed out or grey
    if sat < 1:
        g = (im @ np.array(LUMA))[..., None]
        im = g + (im - g) * sat
    im = np.clip(im, 0, 1) ** U(r, .8, 1.25)
    im = im + U(r, .005, .02) * r.standard_normal(im.shape)
    return np.clip(im, 0, 1)


# ----------------------------------------------------------------------------------------------
# colours
# ----------------------------------------------------------------------------------------------
PAINT = {'red': (.72, .1, .1), 'black': (.08, .08, .09), 'white': (.92, .92, .92), 'silver': (.66, .68, .7),
         'blue': (.15, .3, .7), 'yellow': (.92, .78, .12), 'green': (.12, .4, .2), 'darkblue': (.08, .12, .32),
         'orange': (.9, .45, .1), 'beige': (.78, .7, .55), 'gray': (.45, .45, .47), 'brown': (.4, .26, .15)}
TIRE = rgb(.08, .08, .09)


def paint(r, k):
    return jc(r, PAINT[k], .03, .12)


def glass(r):
    v = U(r, .1, .34)
    return rgb(v, v + .03, v + U(r, .04, .1))


def blotch(r, n=5, thr=0.3, gain=4.0):
    """soft random patches (for coat markings, spots, stripes)"""
    return np.clip((snoise(r, n) - thr) * gain, 0, 1)


# ----------------------------------------------------------------------------------------------
# airplanes
# ----------------------------------------------------------------------------------------------
def plane_side(im, r, T, c, kind, gear):
    b, u, a, d = c
    if kind == 'prop':
        put(im, T.seg(-.46, -.02, .1, 0, .05, .13), b)
        put(im, T.seg(.1, 0, .36, .01, .14, .1), b)
        put(im, T.poly([(-.36, -.03), (-.47, -.2), (-.5, -.2), (-.47, -.01)]), a)
        put(im, T.seg(-.5, -.03, -.36, -.03, .02), u)
        put(im, T.seg(-.08, -.085, .3, -.085, .035), u)
        put(im, T.poly([(.08, -.06), (.12, -.01), (.27, -.01), (.24, -.06)]), d)
        put(im, T.seg(.38, -.1, .38, .12, .012), d)
        put(im, T.seg(-.4, 0, .3, .02, .02), a)
        if gear:
            put(im, T.seg(.15, .05, .12, .14, .015), d)
            put(im, T.ell(.12, .15, .035, .035), d)
            put(im, T.seg(.3, .05, .32, .14, .015), d)
            put(im, T.ell(.32, .15, .03, .03), d)
        return
    if kind == 'fighter':
        put(im, T.poly([(-.5, -.03), (-.42, -.05), (.2, -.05), (.5, 0), (.2, .035), (-.48, .04)]), b)
        put(im, T.poly([(-.3, -.05), (-.42, -.22), (-.5, -.22), (-.46, -.04)]), a)
        put(im, T.poly([(.1, .0), (-.25, .09), (-.33, .09), (-.2, .0)]), u)
        put(im, T.poly([(-.36, .0), (-.5, .05), (-.5, .01)]), u)
        put(im, T.ell(.22, -.05, .09, .03), d)
        if gear:
            put(im, T.seg(.25, .03, .25, .1, .012), d)
            put(im, T.seg(-.1, .03, -.1, .1, .015), d)
        return
    put(im, T.poly([(.06, -.03), (-.1, -.12), (-.17, -.12), (-.07, -.03)]), u)
    put(im, T.seg(-.44, -.012, .43, 0, .075, .1), b)
    put(im, T.ell(.43, .0, .07, .048), b)
    put(im, T.seg(-.42, .012, .42, .028, .03), u)
    put(im, T.poly([(-.32, -.04), (-.43, -.22), (-.5, -.22), (-.47, -.02)]), a)
    put(im, T.poly([(-.36, -.01), (-.5, -.045), (-.5, -.015), (-.4, .015)]), u)
    put(im, T.poly([(.1, .015), (-.14, .15), (-.22, .15), (-.08, .015)]), u)
    put(im, T.ell(.0, .075, .055, .024), d)
    put(im, T.seg(-.3, -.018, .34, -.012, .012), d, .7)
    put(im, T.ell(.42, -.02, .022, .012), d)
    if gear:
        put(im, T.seg(.3, .04, .3, .1, .012), d)
        put(im, T.ell(.3, .1, .02, .02), d)
        put(im, T.seg(-.05, .05, -.05, .1, .015), d)
        put(im, T.ell(-.05, .1, .025, .025), d)


def plane_top(im, r, T, c, kind):
    b, u, a, d = c
    if kind == 'fighter':
        for s in (-1, 1):
            put(im, T.poly([(.12, 0), (-.3, s * .34), (-.4, s * .34), (-.36, 0)]), u)
            put(im, T.poly([(-.36, 0), (-.47, s * .16), (-.5, s * .16), (-.48, 0)]), u)
        put(im, T.poly([(-.5, -.045), (.2, -.045), (.5, 0), (.2, .045), (-.5, .045)]), b)
        put(im, T.ell(.22, 0, .08, .025), d)
        return
    sw = 0.0 if kind == 'prop' else U(r, .12, .3)
    span = U(r, .42, .55)
    ch = .12 if kind == 'prop' else .16
    x0 = .12 if kind == 'prop' else .06
    for s in (-1, 1):
        put(im, T.poly([(x0, 0), (x0 - sw, s * span), (x0 - sw - ch * .45, s * span), (x0 - ch, 0)]), u)
        put(im, T.poly([(-.37, 0), (-.44 - sw * .3, s * .18), (-.49 - sw * .3, s * .18), (-.46, 0)]), u)
        if kind == 'jet':
            put(im, T.ell(x0 - sw * .4, s * span * .4, .05, .022), d)
    put(im, T.seg(-.46, 0, .44, 0, .06 if kind == 'prop' else .075, .1), b)
    put(im, T.ell(.43, 0, .07, .048), b)
    if kind == 'prop':
        put(im, T.seg(.5, -.1, .5, .1, .012), d)
    put(im, T.ell(.4, 0, .03, .03), d)
    put(im, T.seg(-.47, 0, -.36, 0, .02), a)


def plane_front(im, r, T, c, gear):
    b, u, a, d = c
    put(im, T.poly([(-.5, -.035), (0, .0), (.5, -.035), (.5, -.02), (0, .04), (-.5, -.02)]), u)
    put(im, T.seg(0, -.06, 0, -.25, .03, .02), a)
    put(im, T.seg(-.15, -.1, .15, -.1, .018), u)
    for sg in (-1, 1):
        put(im, T.ell(sg * .2, .045, .04, .04), d)
    put(im, T.ell(0, 0, .075, .08), b)
    put(im, T.seg(-.04, -.03, .04, -.03, .02), d)
    if gear:
        for x in (-.07, 0, .07):
            put(im, T.seg(x, .06, x, .14, .012), d)
            put(im, T.ell(x, .15, .02, .025), d)


def scene_airplane(r):
    onground = r.random() > 0.72
    if not onground:
        im = sky(r)
        if r.random() < 0.1:
            hor = U(r, .78, .95)
            g = ground(r, choose(r, 'grass', 1, 'drygrass', 1, 'water', 1, 'dirt', 1))
            put(im, (Y > hor).astype(np.float64), g, .85)
    else:
        gk = choose(r, 'road', 55, 'concrete', 20, 'grass', 25)
        im, hor = outdoor(r, gk=gk, hor=U(r, .55, .78), back=choose(r, None, 40, 'trees', 35, 'buildings', 15, 'hills', 10))
    kind = choose(r, 'jet', 55, 'fighter', 20, 'prop', 25)
    view = choose(r, 'side', 66, 'front', 24, 'top', 10) if onground else choose(r, 'side', 42, 'top', 45, 'front', 13)
    cs = choose(r, 'white', 50, 'gray', 20, 'dark', 15, 'color', 15)
    if kind == 'fighter' and cs == 'white':
        cs = 'gray'
    if cs == 'white':
        v = U(r, .82, .98)
        b = rgb(v, v, v)
    elif cs == 'gray':
        v = U(r, .42, .72)
        b = rgb(v, v + .01, v + U(r, 0, .04))
    elif cs == 'dark':
        v = U(r, .04, .25)
        b = rgb(v, v, v + .02)
    else:
        b = paint(r, choose(r, 'red', 3, 'yellow', 3, 'blue', 2, 'green', 2, 'orange', 1, 'silver', 2))
    u = b * U(r, .6, .9)
    a = b * .9 if (cs == 'dark' or r.random() < .35) else paint(r, choose(r, 'red', 3, 'blue', 3, 'darkblue', 3, 'orange', 1, 'green', 1, 'yellow', 1))
    d = b * .5 if cs == 'dark' else rgb(.13, .14, .18)
    c = (b, u, a, d)
    L, cx, cy, flip = U(r, .62, 1.04), U(r, .42, .58), U(r, .36, .62), choose(r, -1, 1, 1, 1)
    if view == 'side':
        if onground:
            cy = min(hor + U(r, .03, .16), .9) - .1 * L
            T = Xf(cx, cy, L, flip, U(r, -.04, .04))
            shade(im, ell(cx, cy + .11 * L, .45 * L, .035 * L), U(r, .45, .75))
        else:
            T = Xf(cx, cy, L, flip, U(r, -.35, .35))
        plane_side(im, r, T, c, kind, onground or r.random() < .15)
    elif view == 'front':
        if onground:
            cy = min(hor + U(r, .03, .16), .9) - .16 * L
            shade(im, ell(cx, cy + .17 * L, .3 * L, .03 * L), U(r, .45, .75))
        plane_front(im, r, Xf(cx, cy, L, flip, U(r, -.05, .05) if onground else U(r, -.4, .4)), c, onground or r.random() < .3)
    else:
        T = Xf(cx, cy, L, flip, U(r, 0, 2 * PI), U(r, .3, 1.0))
        plane_top(im, r, T, c, kind)
    return im


# ----------------------------------------------------------------------------------------------
# cars and trucks
# ----------------------------------------------------------------------------------------------
def wheel(im, r, T, x, y, rx, ry, hub):
    put(im, T.ell(x, y, rx, ry), TIRE)
    put(im, T.ell(x, y, rx * .52, ry * .52), hub)


def car_side(im, r, T, b, style):
    wr = U(r, .065, .08)
    H = {'sedan': U(r, .27, .31), 'suv': U(r, .33, .4), 'sport': U(r, .22, .26), 'hatch': U(r, .28, .32), 'van': U(r, .38, .46)}[style]
    belt = H * U(r, .5, .58)
    g, hub = glass(r), rgb(.6, .6, .62) * U(r, .7, 1.3)
    shade(im, T.ell(0, .0, .52, .04), U(r, .35, .65))
    put(im, T.poly([(-.5, -.05), (-.5, -belt * .9), (-.47, -belt), (.46, -belt), (.5, -belt * .72), (.5, -.05)]), b)
    if style in ('sedan', 'sport'):
        cab = [(-.36, -belt), (-.22, -H), (.08, -H), (.26, -belt)]
    elif style == 'hatch':
        cab = [(-.46, -belt), (-.36, -H), (.08, -H), (.25, -belt)]
    else:
        cab = [(-.48, -belt), (-.45, -H), (.14, -H), (.3, -belt)]
    put(im, T.poly(cab), b)
    k = .78
    win = [(cab[0][0] + .05, -belt - .01), (cab[1][0] + .03, -belt - (H - belt) * k), (cab[2][0] - .02, -belt - (H - belt) * k),
           (cab[3][0] - .04, -belt - .01)]
    put(im, T.poly(win), g)
    put(im, T.seg(-.06, -belt, -.07, -H, .025), b)
    put(im, T.rect(-.47, -belt, .46, -belt + .025), np.clip(b * 1.3 + .08, 0, 1), .7)
    put(im, T.rect(-.5, -.1, .5, -.05), b * .6)
    for wx in (-.3 + U(r, -.02, .02), .31 + U(r, -.02, .02)):
        put(im, T.ell(wx, -wr, wr * 1.2, wr * 1.2), b * .25)
        wheel(im, r, T, wx, -wr, wr, wr, hub)
    put(im, T.ell(.485, -belt * .8, .02, .025), rgb(.95, .95, .8))
    put(im, T.ell(-.49, -belt * .8, .015, .03), rgb(.75, .1, .08))


def car_q(im, r, T, b, style):
    xf = U(r, .0, .2)
    H = U(r, .3, .38) if style in ('sedan', 'sport', 'hatch') else U(r, .4, .5)
    belt = H * .56
    g, hub = glass(r), rgb(.6, .6, .62) * U(r, .7, 1.3)
    shade(im, T.ell(0, .0, .55, .05), U(r, .35, .65))
    put(im, T.poly([(-.5, -.07), (-.5, -belt * .95), (-.42, -H * .97), (xf - .2, -H), (xf - .02, -belt), (xf, -.05)]), b * .82)
    put(im, T.poly([(-.44, -belt), (-.4, -H * .92), (xf - .22, -H * .94), (xf - .07, -belt)]), g)
    put(im, T.seg(-.22, -belt, -.23, -H * .94, .022), b * .82)
    put(im, T.poly([(xf, -.05), (xf, -belt * .95), (.5, -belt * .88), (.5, -.04)]), b)
    put(im, T.poly([(xf - .02, -belt), (xf + .08, -belt * 1.12), (.44, -belt * 1.05), (.5, -belt * .88), (xf, -belt * .95)]),
        np.clip(b * 1.15 + .05, 0, 1))
    put(im, T.poly([(xf - .2, -H), (xf + .08, -belt * 1.12), (.44, -belt * 1.05), (.3, -H * .96)]), g)
    put(im, T.poly([(xf + .1, -belt * .8), (.42, -belt * .75), (.42, -belt * .48), (xf + .1, -belt * .5)]), rgb(.1, .1, .11))
    put(im, T.ell(xf + .05, -belt * .72, .035, .03), rgb(.95, .95, .85))
    put(im, T.ell(.46, -belt * .68, .03, .03), rgb(.95, .95, .85))
    put(im, T.poly([(xf, -belt * .4), (.5, -belt * .36), (.5, -.04), (xf, -.05)]), b * U(r, .45, .95))
    wheel(im, r, T, -.33, -.075, .065, .078, hub)
    wheel(im, r, T, xf + .01, -.07, .05, .078, hub)


def car_front(im, r, T, b, style, rear):
    H = U(r, .62, .8) if style in ('sedan', 'sport', 'hatch') else U(r, .8, 1.0)
    belt = H * .52
    g = glass(r)
    shade(im, T.ell(0, .0, .56, .06), U(r, .35, .65))
    put(im, T.rect(-.47, -.14, -.3, 0), TIRE)
    put(im, T.rect(.3, -.14, .47, 0), TIRE)
    put(im, T.poly([(-.5, -.1), (-.5, -belt * .9), (-.46, -belt), (.46, -belt), (.5, -belt * .9), (.5, -.1)]), b)
    put(im, T.poly([(-.43, -belt), (-.33, -H), (.33, -H), (.43, -belt)]), b * .92)
    put(im, T.poly([(-.38, -belt - .01), (-.3, -H + .04), (.3, -H + .04), (.38, -belt - .01)]), g)
    if rear:
        put(im, T.ell(-.38, -belt * .7, .08, .045), rgb(.75, .1, .08))
        put(im, T.ell(.38, -belt * .7, .08, .045), rgb(.75, .1, .08))
    else:
        put(im, T.rect(-.22, -belt * .72, .22, -belt * .45), rgb(.1, .1, .11))
        put(im, T.ell(-.37, -belt * .62, .08, .05), rgb(.95, .95, .85))
        put(im, T.ell(.37, -belt * .62, .08, .05), rgb(.95, .95, .85))
    put(im, T.rect(-.5, -belt * .36, .5, -.1), b * U(r, .45, .95))
    put(im, T.rect(-.1, -belt * .32, .1, -belt * .2), rgb(.9, .9, .88))
    put(im, T.ell(-.5, -belt - .03, .035, .025), b * .7)
    put(im, T.ell(.5, -belt - .03, .035, .025), b * .7)


def street(r, kinds):
    """background for vehicles; returns picture and the height of the line where wheels touch the ground"""
    bgk = choose(r, *kinds)
    if bgk == 'plain':
        im = plain(r, choose(r, 'white', 60, 'gray', 25, 'black', 8, 'color', 7))
        gy = U(r, .7, .88)
        if r.random() < .5:
            shade(im, (Y > gy - U(r, .05, .2)).astype(np.float64), U(r, .8, .97))
        return im, gy
    if bgk == 'dark':
        im = plain(r, 'black')
        gy = U(r, .7, .88)
        put(im, (Y > gy - U(r, .1, .25)).astype(np.float64), ground(r, 'road'), U(r, .3, .9))
        return im, gy
    if bgk == 'green':
        im = foliage(r, U(r, .2, .9))
        gy = U(r, .7, .88)
        gk = choose(r, 'road', 4, 'grass', 3, 'dirt', 2, 'concrete', 1)
        put(im, (Y > gy - U(r, .08, .3)).astype(np.float64), ground(r, gk))
        return im, gy
    gk = {'road': 'road', 'dirt': 'dirt', 'grass': 'grass', 'lot': 'concrete', 'snow': 'snow'}[bgk]
    hor = U(r, .32, .62)
    back = choose(r, None, 20, 'trees', 38, 'buildings', 32, 'hills', 5, 'forest', 5)
    im, hor = outdoor(r, gk=gk, hor=hor, back=back)
    if bgk in ('road', 'lot') and r.random() < .3:
        put(im, rect(0, hor, 1, hor + U(r, .02, .08)), ground(r, choose(r, 'grass', 2, 'concrete', 1)))
    return im, U(r, max(hor + .12, .68), .9)


def scene_automobile(r):
    im, gy = street(r, ('road', 46, 'lot', 14, 'plain', 14, 'grass', 8, 'dark', 6, 'green', 8, 'dirt', 3, 'snow', 1))
    b = paint(r, choose(r, 'red', 18, 'black', 13, 'white', 15, 'silver', 16, 'blue', 12, 'yellow', 6, 'green', 6,
                        'darkblue', 5, 'orange', 3, 'beige', 6))
    style = choose(r, 'sedan', 40, 'suv', 18, 'sport', 16, 'hatch', 16, 'van', 10)
    view = choose(r, 'side', 34, 'q', 42, 'front', 24)
    flip, cx = choose(r, -1, 1, 1, 1), U(r, .44, .56)
    bf = shd(r, b, gy - .2, .25, k=.3, fine=.02, mid=.05)
    if view == 'side':
        car_side(im, r, Xf(cx, gy, U(r, .78, 1.05), flip, U(r, -.03, .03)), bf, style)
    elif view == 'q':
        car_q(im, r, Xf(cx, gy, U(r, .78, 1.05), flip, U(r, -.04, .04)), bf, style)
    else:
        car_front(im, r, Xf(cx, gy, U(r, .6, .9), flip, U(r, -.03, .03)), bf, style, r.random() < .35)
    return im


def truck_side(im, r, T, b, cargo_c, kind):
    wr = U(r, .065, .09)
    ch = wr * 1.25
    cabw, cabh = U(r, .2, .28), U(r, .3, .42)
    hood = choose(r, 0.0, 1, U(r, .08, .15), 1)
    x1 = .5 - hood
    x0 = x1 - cabw
    g, hub = glass(r), rgb(.6, .6, .62) * U(r, .6, 1.4)
    shade(im, T.ell(0, .0, .53, .045), U(r, .35, .65))
    put(im, T.rect(-.5, -ch - .03, x1, -ch + .04), rgb(.12, .12, .13))
    if kind == 'box':
        put(im, T.rect(-.5, -ch - U(r, .36, .52), x0 - .02, -ch), cargo_c)
    elif kind == 'tank':
        put(im, T.seg(-.36, -ch - .15, x0 - .16, -ch - .15, .29), cargo_c)
    elif kind == 'dump':
        put(im, T.poly([(-.5, -ch), (-.47, -ch - .3), (x0 - .02, -ch - .34), (x0 - .05, -ch)]), cargo_c)
    elif kind == 'fire':
        put(im, T.rect(-.5, -ch - U(r, .22, .3), x0 - .02, -ch), cargo_c)
        put(im, T.seg(-.48, -ch - .33, x0 + .1, -ch - .36, .035), rgb(.75, .75, .75))
        put(im, T.rect(-.5, -ch - .12, x1, -ch - .08), rgb(.92, .92, .9))
    else:  # flatbed with a load
        put(im, T.rect(-.5, -ch - .05, x0 - .02, -ch), cargo_c)
        if r.random() < .6:
            put(im, T.rect(-.45, -ch - U(r, .12, .3), x0 - .08, -ch - .05), jc(r, rgb(.5, .4, .3), .15, .3))
    put(im, T.poly([(x0, -ch), (x0, -ch - cabh), (x1 - .04, -ch - cabh), (x1, -ch - cabh * .55), (x1, -ch)]), b)
    if hood:
        put(im, T.rect(x1, -ch - cabh * .5, .5, -ch), b)
    put(im, T.poly([(x0 + .04, -ch - cabh * .55), (x0 + .04, -ch - cabh * .9), (x1 - .06, -ch - cabh * .9), (x1 - .02, -ch - cabh * .55)]), g)
    put(im, T.rect(.44, -ch - .02, .52, -ch + .05), rgb(.5, .5, .52))
    xs = [x1 - .07 if not hood else x1 + hood * .45, -.32]
    if r.random() < .6:
        xs.append(-.32 + 2.25 * wr)
    for wx in xs:
        wheel(im, r, T, wx, -wr, wr, wr, hub)


def truck_q(im, r, T, b, cargo_c, kind):
    xf = U(r, .0, .2)
    H = U(r, .5, .68)
    BH = H * U(r, .95, 1.3) if kind != 'none' else H
    far = BH * U(r, .7, .92)
    g, hub = glass(r), rgb(.6, .6, .62) * U(r, .6, 1.4)
    shade(im, T.ell(0, .0, .56, .06), U(r, .35, .65))
    if kind != 'none':
        put(im, T.poly([(-.5, -.12), (-.5, -far), (xf - .12, -BH), (xf - .12, -.08)]), cargo_c * .85)
    put(im, T.poly([(-.5, -.14), (xf - .1, -.1), (xf - .1, -.06), (-.5, -.1)]), rgb(.12, .12, .13))
    put(im, T.poly([(xf - .14, -.07), (xf - .14, -H * .98), (xf, -H), (xf, -.04)]), b * .78)
    put(im, T.poly([(xf - .12, -H * .56), (xf - .12, -H * .9), (xf - .02, -H * .91), (xf - .02, -H * .57)]), g)
    put(im, T.poly([(xf, -.04), (xf, -H), (.5, -H * .96), (.5, -.03)]), b)
    put(im, T.poly([(xf + .03, -H * .92), (.47, -H * .89), (.47, -H * .58), (xf + .03, -H * .58)]), g)
    put(im, T.poly([(xf + .08, -H * .5), (.42, -H * .49), (.42, -H * .22), (xf + .08, -H * .22)]),
        choose(r, rgb(.1, .1, .11), 2, rgb(.7, .7, .72), 1))
    put(im, T.ell(xf + .045, -H * .25, .028, .035), rgb(.95, .95, .85))
    put(im, T.ell(.46, -H * .24, .025, .035), rgb(.95, .95, .85))
    put(im, T.poly([(xf, -.14), (.5, -.13), (.5, -.03), (xf, -.04)]), choose(r, rgb(.5, .5, .52), 1, b * .6, 1, rgb(.15, .15, .16), 1))
    wheel(im, r, T, xf + .03, -.075, .05, .085, hub)
    put(im, T.ell(.47, -.065, .035, .075), TIRE)
    wheel(im, r, T, -.3, -.105, .065, .085, hub)
    if r.random() < .5:
        wheel(im, r, T, -.44, -.115, .06, .08, hub)


def truck_front(im, r, T, b, cargo_c, kind):
    H = U(r, 1.0, 1.35)
    g = glass(r)
    shade(im, T.ell(0, .0, .58, .07), U(r, .35, .65))
    if kind != 'none':
        put(im, T.rect(-.54, -H * U(r, 1.05, 1.25), .54, -H * .8), cargo_c * .9)
    put(im, T.rect(-.5, -.16, -.3, 0), TIRE)
    put(im, T.rect(.3, -.16, .5, 0), TIRE)
    put(im, T.rect(-.5, -H, .5, -.1), b)
    put(im, T.rect(-.43, -H * .92, .43, -H * .58), g)
    if r.random() < .5:
        put(im, T.rect(-.015, -H * .92, .015, -H * .58), b)
    put(im, T.rect(-.28, -H * .5, .28, -H * .22), choose(r, rgb(.1, .1, .11), 2, rgb(.7, .7, .72), 1))
    put(im, T.ell(-.39, -H * .27, .06, .05), rgb(.95, .95, .85))
    put(im, T.ell(.39, -H * .27, .06, .05), rgb(.95, .95, .85))
    put(im, T.rect(-.53, -.22, .53, -.09), choose(r, rgb(.5, .5, .52), 1, b * .6, 1, rgb(.15, .15, .16), 1))
    put(im, T.rect(-.6, -H * .8, -.53, -H * .62), rgb(.15, .15, .16))
    put(im, T.rect(.53, -H * .8, .6, -H * .62), rgb(.15, .15, .16))


def scene_truck(r):
    im, gy = street(r, ('road', 54, 'lot', 12, 'plain', 15, 'dirt', 9, 'grass', 5, 'green', 3, 'snow', 2))
    kind = choose(r, 'box', 36, 'tank', 8, 'dump', 16, 'fire', 14, 'flat', 12, 'none', 14)
    bk = choose(r, 'white', 25, 'red', 22, 'blue', 10, 'yellow', 8, 'orange', 6, 'green', 8, 'black', 6, 'silver', 7, 'gray', 4, 'darkblue', 4)
    if kind == 'fire':
        bk = choose(r, 'red', 9, 'yellow', 1)
    if kind == 'dump':
        bk = choose(r, 'yellow', 4, 'orange', 3, 'red', 2, 'white', 2, 'green', 1, 'blue', 1)
    b = paint(r, bk)
    if kind in ('fire', 'dump') or r.random() < .3:
        cc = b
    else:
        cc = paint(r, choose(r, 'white', 50, 'silver', 18, 'gray', 8, 'red', 5, 'blue', 6, 'yellow', 4, 'green', 4, 'beige', 5))
    view = choose(r, 'side', 32, 'q', 48, 'front', 20)
    flip, cx = choose(r, -1, 1, 1, 1), U(r, .44, .56)
    bf = shd(r, b, gy - .35, .4, k=.25, fine=.02, mid=.04)
    cf_ = shd(r, cc, gy - .4, .4, k=.2, fine=.02, mid=.04)
    if view == 'side':
        if kind == 'none':
            kind = 'box'
        truck_side(im, r, Xf(cx, gy, U(r, .85, 1.08), flip, U(r, -.03, .03)), bf, cf_, kind)
    elif view == 'q':
        truck_q(im, r, Xf(cx, gy, U(r, .85, 1.1), flip, U(r, -.04, .04)), bf, cf_, kind)
    else:
        truck_front(im, r, Xf(cx, gy, U(r, .52, .72), flip, U(r, -.03, .03)), bf, cf_, kind)
    return im


# ----------------------------------------------------------------------------------------------
# ships
# ----------------------------------------------------------------------------------------------
def ship_draw(im, r, T, kind):
    wh = rgb(.93, .93, .92) * U(r, .85, 1.03)
    dk = rgb(.1, .1, .12)
    if kind == 'cargo':
        hc = choose(r, rgb(.12, .12, .14), 30, rgb(.1, .15, .32), 20, rgb(.5, .14, .1), 20, rgb(.45, .47, .5), 15, rgb(.1, .3, .2), 8, rgb(.2, .35, .6), 7)
        hh = U(r, .07, .14)
        put(im, T.poly([(-.5, 0), (-.5, -hh), (.4, -hh), (.5, -hh * 1.5), (.44, 0)]), jc(r, hc))
        if r.random() < .5:
            put(im, T.poly([(-.5, 0), (-.5, -hh * .28), (.455, -hh * .28), (.44, 0)]), rgb(.55, .12, .1))
        sx = choose(r, -.45, 2, .18, 1)
        sw, sh = U(r, .12, .2), U(r, .1, .2)
        put(im, T.rect(sx, -hh - sh, sx + sw, -hh), wh)
        put(im, T.rect(sx + .01, -hh - sh * .8, sx + sw - .01, -hh - sh * .65), dk, .7)
        put(im, T.rect(sx + sw * .3, -hh - sh - .08, sx + sw * .6, -hh - sh), jc(r, choose(r, rgb(.7, .15, .1), 1, rgb(.15, .2, .45), 1, rgb(.15, .15, .15), 1, rgb(.9, .7, .2), 1)))
        if r.random() < .6:
            x = -.25 if sx < 0 else -.45
            xe = .38 if sx < 0 else .15
            while x < xe:
                w = U(r, .05, .12)
                put(im, T.rect(x, -hh - U(r, .03, .1), x + w, -hh), jc(r, choose(r, rgb(.6, .2, .15), 1, rgb(.2, .3, .55), 1, rgb(.75, .6, .2), 1, rgb(.25, .45, .3), 1, rgb(.6, .6, .6), 1, rgb(.5, .3, .2), 1), .08))
                x += w + .005
        else:
            for x in (-.1, .15):
                put(im, T.seg(x, -hh, x, -hh - U(r, .12, .25), .012), dk)
    elif kind == 'cruise':
        hh = U(r, .09, .16)
        put(im, T.poly([(-.5, 0), (-.5, -hh), (.42, -hh), (.5, -hh * 1.3), (.45, 0)]), wh)
        if r.random() < .6:
            put(im, T.poly([(-.5, 0), (-.5, -hh * .3), (.46, -hh * .3), (.45, 0)]), jc(r, choose(r, rgb(.1, .15, .35), 2, rgb(.6, .12, .1), 1, rgb(.1, .1, .12), 1)))
        n = int(U(r, 1, 4.5))
        x0, x1, y = -.44, .34, -hh
        for i in range(n):
            h = U(r, .04, .07)
            put(im, T.rect(x0, y - h, x1, y), wh * (1 - .04 * i))
            put(im, T.seg(x0 + .02, y - h * .5, x1 - .03, y - h * .5, .012), dk, .6)
            y -= h
            x0 += U(r, .01, .08)
            x1 -= U(r, .03, .12)
        fx = U(r, -.2, .05)
        put(im, T.poly([(fx, y), (fx - .01, y - .09), (fx + .07, y - .09), (fx + .1, y)]), jc(r, choose(r, rgb(.75, .15, .1), 1, rgb(.12, .2, .5), 1, rgb(.9, .75, .15), 1, rgb(.12, .12, .12), 1)))
    elif kind == 'sail':
        hl = U(r, .28, .4)
        hc = choose(r, wh, 55, rgb(.12, .15, .3), 20, rgb(.55, .15, .1), 10, rgb(.4, .25, .15), 15)
        put(im, T.poly([(-hl, 0), (-hl - .03, -.07), (hl + .08, -.08), (hl - .05, 0)]), jc(r, hc))
        mh = U(r, .7, 1.15)
        put(im, T.seg(0, -.07, 0, -mh, .014), dk)
        sc = choose(r, wh, 70, rgb(.9, .85, .7), 12, jc(r, rgb(.75, .2, .15), .1), 9, jc(r, rgb(.2, .3, .6), .1), 9)
        if r.random() < .85:
            put(im, T.poly([(-.02, -.12), (-.02, -mh * .97), (-hl * U(r, .8, 1.3), -.13)]), sc)
        if r.random() < .7:
            put(im, T.poly([(.02, -mh * .9), (hl + .06, -.1), (.03, -.12)]), sc * .93)
    elif kind == 'motor':
        put(im, T.poly([(-.42, 0), (-.47, -.08), (.2, -.1), (.5, -.15), (.32, 0)]), wh)
        if r.random() < .5:
            put(im, T.poly([(-.44, -.03), (-.455, -.06), (.36, -.08), (.42, -.07), (.34, -.03)]), jc(r, choose(r, rgb(.1, .15, .4), 2, rgb(.7, .12, .1), 2, rgb(.1, .1, .1), 1)))
        if r.random() < .75:
            put(im, T.poly([(-.25, -.085), (-.2, -.19), (.02, -.19), (.14, -.1)]), wh * .97)
            put(im, T.poly([(-.19, -.11), (-.16, -.17), (.0, -.17), (.08, -.11)]), glass(r))
        put(im, T.poly([(-.38, .0), (-1.0, .05), (-1.0, -.03), (-.45, -.04)]), rgb(.95, .96, .97), U(r, .3, .9))
        put(im, T.seg(-.5, .02, .4, .02, .035, .01), rgb(.95, .96, .97), U(r, .3, .9))
    elif kind == 'war':
        hc = jc(r, rgb(.5, .52, .55), .02, .25)
        put(im, T.poly([(-.5, 0), (-.5, -.07), (.38, -.07), (.5, -.11), (.42, 0)]), hc)
        put(im, T.rect(-.3, -.13, .22, -.07), hc * .95)
        put(im, T.rect(-.18, -.2, .1, -.13), hc * 1.05)
        put(im, T.rect(-.08, -.27, .03, -.2), hc)
        put(im, T.seg(-.03, -.27, -.03, -.42, .014), hc * .6)
        put(im, T.seg(-.08, -.36, .02, -.36, .01), hc * .6)
        put(im, T.rect(-.17, -.26, -.12, -.2), hc * .8)
        put(im, T.seg(.28, -.085, .4, -.1, .012), hc * .6)
    else:  # tug / fishing boat / small working boat
        hl = U(r, .32, .46)
        hc = jc(r, choose(r, rgb(.12, .12, .14), 2, rgb(.6, .15, .1), 2, rgb(.12, .25, .5), 2, rgb(.15, .4, .25), 1, rgb(.9, .9, .88), 3, rgb(.45, .3, .18), 2), .05)
        hh = U(r, .08, .14)
        put(im, T.poly([(-hl, 0), (-hl - .02, -hh), (hl * .7, -hh), (hl + .06, -hh * 1.5), (hl - .04, 0)]), hc)
        cw = U(r, .15, .3)
        cx0 = U(r, -hl + .05, hl * .5 - cw)
        ch = U(r, .08, .16)
        put(im, T.rect(cx0, -hh - ch, cx0 + cw, -hh), wh)
        put(im, T.rect(cx0 + .02, -hh - ch * .8, cx0 + cw - .02, -hh - ch * .45), dk, .75)
        put(im, T.seg(cx0 + cw * .5, -hh - ch, cx0 + cw * .5, -hh - ch - U(r, .1, .3), .014), dk)
        if r.random() < .5:
            put(im, T.rect(cx0 + cw * .6, -hh - ch - .07, cx0 + cw * .85, -hh - ch), jc(r, rgb(.6, .2, .12), .1))


def scene_ship(r):
    hor = U(r, .28, .64)
    aerial = r.random() < .06
    sk = choose(r, 'clear', 28, 'pale', 22, 'overcast', 25, 'clouds', 15, 'sunset', 4, 'dark', 6)
    im = sky(r, sk)
    if sk in ('overcast', 'dark'):         # water mirrors the sky: grey under grey skies, blue under blue ones
        w = water(r, choose(r, 'gray', 55, 'dark', 25, 'green', 12, 'blue', 8))
    else:
        w = water(r, choose(r, 'blue', 60, 'green', 12, 'dark', 10, 'gray', 10, 'bright', 8))
    if aerial:
        im = w
        hor = 0.0
    else:
        b = choose(r, None, 52, 'trees', 16, 'hills', 16, 'buildings', 11, 'land', 5)
        if b == 'trees':
            treeline(r, im, hor, h=U(r, .04, .25))
        elif b == 'hills':
            hills(r, im, hor)
        elif b == 'buildings':
            buildings(r, im, hor)
        elif b == 'land':
            put(im, rect(0, hor - U(r, .02, .08), 1, hor), jc(r, rgb(.4, .4, .3), .1))
        hz = np.clip(1 - (Y - hor) / U(r, .1, .5), 0, 1) * U(r, 0, .5)
        skyc = im[max(0, int(hor * S) - 2)].mean(0)
        w = w + hz[..., None] * (skyc - w)
        put(im, (Y > hor).astype(np.float64), w)
    kind = choose(r, 'cargo', 22, 'cruise', 16, 'sail', 20, 'motor', 18, 'war', 9, 'tug', 15)
    L, cx, flip = U(r, .58, 1.02), U(r, .4, .6), choose(r, -1, 1, 1, 1)
    wy = float(np.clip(hor + U(r, .03, .3), .42, .88))
    T = Xf(cx, wy, L, flip, U(r, -.03, .03), 1.0 if kind == 'sail' else U(r, 1.0, 1.7))
    if kind != 'motor':
        shade(im, T.poly([(-.5, 0), (.46, 0), (.4, .09), (-.46, .09)]) * (Y > hor), U(r, .6, .95))
    ship_draw(im, r, T, kind)
    if r.random() < .5:
        put(im, T.seg(-.52, .012, .5, .012, .03, .012), rgb(.93, .95, .96), U(r, .2, .8))
    return im


# ----------------------------------------------------------------------------------------------
# animals
# ----------------------------------------------------------------------------------------------
def coat(r, base, kind='plain', second=None, cy=.5, h=.4):
    f = shd(r, base, cy, h, k=.2, fine=.07, mid=.07, nmid=10)
    if kind == 'tabby':
        st = np.clip(snoise(r, U(r, 9, 18), U(r, 3, 6)) * 2, 0, 1)
        f = f * (1 - U(r, .25, .5) * st)[..., None]
    elif kind == 'patch':
        put(f, blotch(r, U(r, 2.5, 6), U(r, -.2, .4)), shd(r, second, cy, h, k=.2, fine=.07, mid=.07))
    elif kind == 'spots':
        put(f, blotch(r, U(r, 7, 14), U(r, .2, .7), 5), np.clip(base * U(r, .25, .6), 0, 1))
    return f


def leg2(im, T, x, y0, y1, w0, w1, col, kx=0.0, fx=0.0, col2=None):
    ym = y0 + (y1 - y0) * .52
    put(im, T.seg(x, y0, x + kx, ym, w0, (w0 + w1) / 2), col)
    put(im, T.seg(x + kx, ym, x + fx, y1, (w0 + w1) / 2, w1), col if col2 is None else col2)


def horse_side(im, r, T, b, dk, low, pose, rider):
    gy = .46
    sp = U(r, -.03, .1)
    far = b * .72
    leg2(im, T, -.2, .05, gy, .075, .035, far, -.03, sp, low * .8)
    leg2(im, T, .21, .05, gy, .06, .033, far, .02, -sp, low * .8)
    put(im, T.seg(-.3, -.07, -.37 - U(r, 0, .06), .22, .06, .03), dk)
    put(im, T.ell(0, 0, .3, .14), b)
    put(im, T.ell(-.19, -.01, .14, .15), b)
    put(im, T.ell(.2, .0, .12, .145), b)
    if pose == 'up':
        hx, hy, ha = .38, -.33, .9
    elif pose == 'graze':
        hx, hy, ha = .46, .22, 1.3
    else:
        hx, hy, ha = .47, -.14, .6
    put(im, T.seg(.2, -.03, hx, hy, .19, .09), b)
    put(im, T.seg(hx, hy, hx + .17 * np.cos(ha), hy + .17 * np.sin(ha), .095, .055), b)
    put(im, T.seg(hx - .01, hy - .02, hx - .03, hy - .09, .03, .015), b)
    put(im, T.seg(.2, -.13, hx - .04, hy - .02, .045), dk)
    leg2(im, T, -.25, .05, gy, .085, .038, b, -.02, -sp, low)
    leg2(im, T, .16, .05, gy, .065, .035, b, .0, sp, low)
    if rider:
        rc = jc(r, choose(r, rgb(.75, .12, .1), 2, rgb(.15, .2, .5), 2, rgb(.9, .9, .9), 2, rgb(.1, .1, .1), 2, rgb(.2, .45, .25), 1, rgb(.8, .7, .2), 1), .05)
        put(im, T.seg(.02, -.12, -.01, -.36, .17, .15), rc)
        put(im, T.seg(.0, -.3, .12, -.14, .05, .04), rc)
        put(im, T.ell(.0, -.46, .055, .06), choose(r, rgb(.1, .1, .1), 1, rgb(.8, .62, .5), 1, rgb(.9, .9, .9), 1))
        put(im, T.seg(.02, -.14, .07, .1, .06, .04), jc(r, choose(r, rgb(.15, .15, .2), 1, rgb(.8, .78, .7), 1, rgb(.3, .35, .5), 1), .04))


def horse_head(im, r, T, b, dk):
    put(im, T.poly([(-.7, .7), (-.52, -.1), (-.15, -.36), (.06, -.15), (-.02, .7)]), b)
    put(im, T.poly([(-.54, -.08), (-.2, -.4), (-.3, -.05), (-.6, .35)]), dk)
    put(im, T.seg(-.12, -.25, .28, .25, .3, .17), b)
    put(im, T.ell(.3, .27, .1, .085), b * .8)
    put(im, T.seg(-.2, -.3, -.25, -.5, .08, .03), b)
    put(im, T.seg(-.08, -.33, -.08, -.52, .08, .03), b)
    if r.random() < .35:
        put(im, T.seg(-.06, -.2, .26, .22, .06, .04), rgb(.92, .9, .86))
    put(im, T.ell(.0, -.1, .035, .03), rgb(.05, .04, .04))
    put(im, T.ell(.34, .26, .022, .016), rgb(.08, .06, .06))
    if r.random() < .45:
        hc = choose(r, rgb(.1, .08, .07), 2, rgb(.35, .2, .1), 1, rgb(.7, .1, .1), 1, rgb(.15, .2, .5), 1)
        put(im, T.seg(.12, .2, .28, .06, .022), hc)
        put(im, T.seg(.2, .13, -.12, -.2, .022), hc)


def antlers(im, T, x, y, s, col, front):
    for sg in (-1, 1):
        if front:
            p = [(x + sg * .04, y), (x + sg * .2 * s, y - .16 * s), (x + sg * .16 * s, y - .36 * s)]
            tn = [((p[1][0] + p[0][0]) / 2, (p[1][1] + p[0][1]) / 2, sg * .01, -.14 * s), (p[1][0], p[1][1], -sg * .05 * s, -.13 * s)]
        else:
            ox = sg * .025
            p = [(x + ox, y), (x + ox - .07 * s, y - .14 * s), (x + ox + .05 * s, y - .3 * s)]
            tn = [(p[1][0], p[1][1], .09 * s, -.05 * s), ((p[1][0] + p[2][0]) / 2, (p[1][1] + p[2][1]) / 2, .08 * s, -.06 * s)]
        put(im, T.seg(p[0][0], p[0][1], p[1][0], p[1][1], .022), col)
        put(im, T.seg(p[1][0], p[1][1], p[2][0], p[2][1], .02, .012), col)
        for tx, ty, dx, dy in tn:
            put(im, T.seg(tx, ty, tx + dx, ty + dy, .016, .01), col)


def deer_side(im, r, T, b, lt, ant, pose):
    gy = .45
    sp = U(r, -.03, .08)
    far = b * .72
    leg2(im, T, -.19, .05, gy, .05, .022, far, -.03, sp)
    leg2(im, T, .2, .05, gy, .04, .02, far, .015, -sp)
    put(im, T.ell(0, 0, .28, .12), b)
    put(im, T.ell(-.18, -.01, .12, .125), b)
    put(im, T.ell(.02, .07, .2, .045), lt, .6)
    put(im, T.ell(-.28, -.03, .04, .07), lt)
    if pose == 'graze':
        hx, hy = .42, .2
        put(im, T.seg(.2, -.03, hx, hy, .11, .06), b)
        put(im, T.seg(hx, hy, hx + .06, hy + .1, .07, .04), b)
    else:
        hx, hy = .32 + U(r, -.03, .04), -.32
        put(im, T.seg(.2, -.03, hx, hy, .12, .06), b)
        put(im, T.seg(hx - .01, hy - .01, hx + .12, hy + .04, .08, .045), b)
        put(im, T.ell(hx + .13, hy + .045, .018, .016), rgb(.08, .06, .05))
        put(im, T.ell(hx - .05, hy - .07, .055, .022, -1.0), b)
        put(im, T.ell(hx + .0, hy - .08, .05, .02, -1.5), b * .9)
        if ant is not None:
            antlers(im, T, hx - .01, hy - .05, U(r, .7, 1.2), ant, False)
    leg2(im, T, -.24, .05, gy, .055, .024, b, -.02, -sp)
    leg2(im, T, .16, .05, gy, .042, .022, b, .0, sp)


def deer_front(im, r, T, b, lt, ant):
    put(im, T.ell(0, .34, .17, .24), b)
    for x in (-.08, .08):
        put(im, T.seg(x, .5, x * 1.1, .85, .045, .03), b * .9)
    put(im, T.seg(0, .2, 0, -.1, .15, .1), b)
    put(im, T.ell(0, .0, .045, .07), lt, .7)
    for sg in (-1, 1):
        put(im, T.ell(sg * .135, -.26, .075, .035, -sg * .5), b)
        put(im, T.ell(sg * .135, -.26, .045, .018, -sg * .5), lt, .6)
    put(im, T.ell(0, -.2, .085, .1), b)
    put(im, T.ell(0, -.13, .05, .05), b * .85)
    put(im, T.ell(0, -.11, .022, .016), rgb(.07, .05, .05))
    for sg in (-1, 1):
        put(im, T.ell(sg * .045, -.22, .014, .014), rgb(.06, .05, .05))
    if ant is not None:
        antlers(im, T, 0, -.29, U(r, .7, 1.3), ant, True)


def cat_ears(im, T, x, y, w, h, b, inner):
    for sg in (-1, 1):
        put(im, T.poly([(x + sg * w, y + .12 * h), (x + sg * w * .92, y - h), (x + sg * w * .25, y - .2 * h)]), b)
        put(im, T.poly([(x + sg * w * .85, y), (x + sg * w * .82, y - h * .7), (x + sg * w * .45, y - .15 * h)]), inner, .7)


def cat_face(im, r, T, b, lt, eyec):
    put(im, T.ell(0, .62, .5, .4), b)
    cat_ears(im, T, 0, -.2, .31, .28, b, rgb(.85, .6, .58))
    put(im, T.ell(0, 0, .36, .31), b)
    put(im, T.ell(0, .13, .15, .11), lt)
    for sg in (-1, 1):
        put(im, T.ell(sg * .15, -.04, .068, .052, sg * .2), eyec)
        put(im, T.ell(sg * .15, -.04, U(r, .014, .04), .045), rgb(.04, .04, .04))
    put(im, T.poly([(-.035, .06), (.035, .06), (0, .1)]), rgb(.8, .5, .5))
    put(im, T.seg(0, .1, 0, .16, .012), rgb(.2, .12, .1), .7)


def cat_sit(im, r, T, b, lt, eyec):
    put(im, T.seg(.12, .42, .34, .4 + U(r, -.08, .04), .055, .035), b)
    put(im, T.ell(0, .2, .2, .27), b)
    put(im, T.ell(-.1, .33, .14, .13), b)
    put(im, T.ell(.02, .12, .09, .14), lt, .8)
    put(im, T.seg(-.04, .2, -.04, .46, .055), lt)
    put(im, T.seg(.07, .2, .07, .46, .055), lt)
    cat_ears(im, T, 0, -.27, .125, .13, b, rgb(.85, .6, .58))
    put(im, T.ell(0, -.2, .15, .13), b)
    put(im, T.ell(0, -.15, .06, .045), lt, .8)
    for sg in (-1, 1):
        put(im, T.ell(sg * .06, -.22, .028, .022), eyec)
        put(im, T.ell(sg * .06, -.22, .01, .02), rgb(.04, .04, .04))
    put(im, T.ell(0, -.17, .014, .01), rgb(.8, .5, .5))


def cat_lie(im, r, T, b, lt, eyec):
    put(im, T.seg(-.3, .06, -.5, .0 + U(r, -.1, .12), .06, .035), b)
    put(im, T.ell(0, .02, .36, .16), b)
    put(im, T.ell(-.2, .02, .17, .17), b)
    put(im, T.ell(.1, .1, .2, .06), lt, .7)
    put(im, T.seg(.18, .13, .4, .15, .06, .05), lt)
    hx, hy = .3, -.1
    cat_ears(im, T, hx, hy - .07, .105, .11, b, rgb(.85, .6, .58))
    put(im, T.ell(hx, hy, .13, .115), b)
    put(im, T.ell(hx, hy + .045, .055, .04), lt, .8)
    for sg in (-1, 1):
        put(im, T.ell(hx + sg * .055, hy - .015, .025, .02), eyec)
        put(im, T.ell(hx + sg * .055, hy - .015, .009, .018), rgb(.04, .04, .04))


def cat_side(im, r, T, b, lt, eyec):
    gy = .3
    for x, f in ((-.2, .72), (.2, .72), (-.24, 1), (.16, 1)):
        leg2(im, T, x, .03, gy, .055, .035, b * f, 0, U(r, -.05, .05), (lt if f == 1 else b * f))
    put(im, T.seg(-.27, -.03, -.4, -.12, .045, .04), b)
    put(im, T.seg(-.4, -.12, -.43, -.33, .04, .03), b)
    put(im, T.ell(0, 0, .29, .115), b)
    hx, hy = .34, -.1
    cat_ears(im, T, hx, hy - .06, .08, .1, b, rgb(.85, .6, .58))
    put(im, T.ell(hx, hy, .1, .09), b)
    put(im, T.ell(hx + .05, hy + .03, .04, .03), lt, .8)
    put(im, T.ell(hx + .04, hy - .02, .018, .016), eyec)


def dog_ears(im, r, T, x, y, s, ec, erect):
    for sg in (-1, 1):
        if erect:
            put(im, T.poly([(x + sg * .28 * s, y - .1 * s), (x + sg * .22 * s, y - .5 * s), (x + sg * .06 * s, y - .27 * s)]), ec)
        else:
            put(im, T.ell(x + sg * .3 * s, y + .04 * s, .095 * s, .21 * s, -sg * .22), ec)


def dog_face(im, r, T, b, lt, ec, erect):
    put(im, T.ell(0, .64, .46, .4), b)
    if erect:
        dog_ears(im, r, T, 0, 0, 1, ec, True)
    put(im, T.ell(0, 0, .3, .31), b)
    if not erect:
        dog_ears(im, r, T, 0, 0, 1, ec, False)
    put(im, T.ell(0, .14, .15, .135), lt)
    put(im, T.ell(0, .08, .058, .042), rgb(.06, .05, .05))
    for sg in (-1, 1):
        put(im, T.ell(sg * .125, -.08, .036, .036), rgb(.08, .06, .05))
    put(im, T.seg(0, .12, 0, .19, .014), rgb(.15, .1, .1), .8)
    if r.random() < .35:
        put(im, T.ell(.01, .25, .05, .07), rgb(.85, .45, .5))


def dog_sit(im, r, T, b, lt, ec, erect):
    put(im, T.poly([(-.16, -.05), (.16, -.05), (.26, .45), (-.26, .45)]), b)
    put(im, T.ell(0, .3, .27, .2), b)
    put(im, T.ell(0, .12, .1, .16), lt, .8)
    put(im, T.seg(-.08, .12, -.09, .47, .07, .06), b)
    put(im, T.seg(.08, .12, .09, .47, .07, .06), b)
    put(im, T.ell(-.09, .47, .05, .03), lt)
    put(im, T.ell(.09, .47, .05, .03), lt)
    hy, s = -.2, .58
    if erect:
        dog_ears(im, r, T, 0, hy, s, ec, True)
    put(im, T.ell(0, hy, .3 * s, .31 * s), b)
    if not erect:
        dog_ears(im, r, T, 0, hy, s, ec, False)
    put(im, T.ell(0, hy + .14 * s, .15 * s, .135 * s), lt)
    put(im, T.ell(0, hy + .08 * s, .06 * s, .045 * s), rgb(.06, .05, .05))
    for sg in (-1, 1):
        put(im, T.ell(sg * .125 * s, hy - .08 * s, .04 * s, .04 * s), rgb(.08, .06, .05))


def dog_side(im, r, T, b, lt, ec, erect):
    gy = .34
    sp = U(r, -.04, .08)
    leg2(im, T, -.2, .04, gy, .07, .045, b * .72, -.02, sp)
    leg2(im, T, .21, .04, gy, .06, .045, b * .72, .01, -sp)
    put(im, T.seg(-.28, -.05, -.4 - U(r, 0, .05), -.2 + U(r, -.05, .25), .045, .02), b)
    put(im, T.ell(0, 0, .3, .13), b)
    put(im, T.ell(.18, .0, .13, .145), b)
    put(im, T.ell(.05, .08, .2, .04), lt, .6)
    put(im, T.seg(.22, -.04, .33, -.17, .15, .12), b)
    hx, hy = .37, -.2
    put(im, T.ell(hx, hy, .1, .09), b)
    put(im, T.seg(hx + .03, hy + .03, hx + .15, hy + .06, .075, .055), lt)
    put(im, T.ell(hx + .16, hy + .05, .022, .02), rgb(.06, .05, .05))
    if erect:
        put(im, T.poly([(hx - .07, hy - .04), (hx - .05, hy - .17), (hx + .02, hy - .07)]), ec)
    else:
        put(im, T.ell(hx - .04, hy + .02, .045, .09, .2), ec)
    put(im, T.ell(hx + .04, hy - .02, .014, .014), rgb(.06, .05, .05))
    leg2(im, T, -.24, .04, gy, .08, .05, b, -.02, -sp, lt if r.random() < .3 else None)
    leg2(im, T, .16, .04, gy, .065, .05, b, 0, sp, lt if r.random() < .3 else None)


def dog_lie(im, r, T, b, lt, ec, erect):
    put(im, T.ell(-.05, .05, .36, .17), b)
    put(im, T.ell(-.25, .05, .17, .18), b)
    put(im, T.seg(.1, .17, .42, .2, .07, .06), lt)
    put(im, T.seg(.0, .2, .3, .24, .07, .06), b)
    hx, hy, s = .27, -.1, .55
    if erect:
        dog_ears(im, r, T, hx, hy, s, ec, True)
    put(im, T.ell(hx, hy, .3 * s, .3 * s), b)
    if not erect:
        dog_ears(im, r, T, hx, hy, s, ec, False)
    put(im, T.ell(hx + .02, hy + .14 * s, .15 * s, .13 * s), lt)
    put(im, T.ell(hx + .02, hy + .08 * s, .06 * s, .045 * s), rgb(.06, .05, .05))
    for sg in (-1, 1):
        put(im, T.ell(hx + sg * .125 * s, hy - .08 * s, .04 * s, .04 * s), rgb(.08, .06, .05))


def frog_side(im, r, T, b, lt, eyec):
    put(im, T.ell(-.17, .09, .18, .08, .45), b * .85)
    put(im, T.seg(-.3, .14, -.06, .2, .055, .045), b * .85)
    put(im, T.seg(-.06, .21, -.3, .23, .035, .025), b * .8)
    put(im, T.ell(0, .02, .3, .16, -.3), b)
    put(im, T.ell(.24, -.08, .13, .095, -.3), b)
    put(im, T.ell(.2, .04, .12, .06, -.2), lt)
    put(im, T.seg(.14, .06, .2, .2, .05, .04), b * .9)
    put(im, T.seg(.2, .21, .3, .22, .035, .02), b * .85)
    put(im, T.ell(.23, -.165, .055, .05), b * 1.1)
    put(im, T.ell(.235, -.165, .036, .03), eyec)
    put(im, T.ell(.24, -.165, .024, .014), rgb(.04, .04, .04))
    put(im, T.seg(.2, -.04, .37, -.07, .012), b * .5, .7)


def frog_front(im, r, T, b, lt, eyec):
    for sg in (-1, 1):
        put(im, T.ell(sg * .33, .16, .1, .14, sg * .3), b * .85)
        put(im, T.seg(sg * .36, .26, sg * .46, .33, .04, .025), b * .8)
    put(im, T.ell(0, .1, .3, .2), b)
    put(im, T.ell(0, -.05, .26, .15), b)
    put(im, T.ell(0, .1, .18, .1), lt)
    put(im, T.seg(-.21, .0, .21, .0, .014), b * .45, .8)
    for sg in (-1, 1):
        put(im, T.seg(sg * .19, .12, sg * .24, .3, .055, .04), b * .92)
        put(im, T.seg(sg * .24, .31, sg * .33, .33, .035, .02), b * .85)
        put(im, T.ell(sg * .15, -.17, .07, .065), b * 1.08)
        put(im, T.ell(sg * .15, -.17, .048, .042), eyec)
        put(im, T.ell(sg * .15, -.17, .034, .016), rgb(.04, .04, .04))


def frog_top(im, r, T, b, lt, eyec):
    for sg in (-1, 1):
        put(im, T.seg(-.2, sg * .1, -.05, sg * .26, .08, .06), b * .88)
        put(im, T.seg(-.05, sg * .26, -.3, sg * .31, .055, .04), b * .85)
        put(im, T.seg(-.3, sg * .31, -.13, sg * .4, .035, .02), b * .8)
        put(im, T.seg(.12, sg * .1, .22, sg * .26, .045, .035), b * .88)
        put(im, T.seg(.22, sg * .26, .32, sg * .27, .03, .02), b * .85)
    put(im, T.ell(0, 0, .26, .165), b)
    put(im, T.ell(.2, 0, .13, .135), b)
    for sg in (-1, 1):
        put(im, T.ell(.24, sg * .095, .042, .042), eyec)
        put(im, T.ell(.245, sg * .095, .024, .024), rgb(.04, .04, .04))


def bird_perch(im, r, T, b, belly, wing, beakc, py):
    tilt = U(r, .25, 1.0)
    ca, sa = np.cos(tilt), np.sin(tilt)

    def A(u, v):
        return u * ca + v * sa, -u * sa + v * ca
    p = [A(-.2, -.03), A(-.6, -.03), A(-.57, .07), A(-.18, .08)]
    put(im, T.poly(p), wing)
    for o in (-.03, .04):
        lx, ly = A(o, .14)
        put(im, T.seg(lx, ly, lx + .01, py, .018), rgb(.2, .15, .12))
    put(im, T.ell(0, 0, .27, .17, -tilt), b)
    bx, by = A(.06, .07)
    put(im, T.ell(bx, by, .2, .085, -tilt), belly)
    wx, wy = A(-.09, -.045)
    put(im, T.ell(wx, wy, .21, .09, -tilt - .08), wing)
    hx, hy = A(.27, .0)
    put(im, T.ell(hx, hy, .105, .1), b)
    bl = U(r, .08, .2)
    put(im, T.poly([(hx + .08, hy - .03), (hx + .08 + bl, hy + .015), (hx + .08, hy + .045)]), beakc)
    put(im, T.ell(hx + .035, hy - .02, .02, .02), rgb(.04, .04, .04))


def bird_fly(im, r, T, b, wing, beakc, style):
    if style == 'soar':
        for sg in (-1, 1):
            put(im, T.poly([(.1, 0), (.06, sg * .5), (-.02, sg * .52), (-.1, sg * .3), (-.1, 0)]), wing)
        put(im, T.poly([(-.16, -.03), (-.36, -.1), (-.38, .1), (-.16, .03)]), wing)
        put(im, T.ell(0, 0, .2, .065), b)
        put(im, T.ell(.2, 0, .055, .05), b)
        put(im, T.poly([(.24, -.02), (.32, 0), (.24, .02)]), beakc)
        return
    d = -1 if style == 'up' else 1
    put(im, T.poly([(.1, 0), (.14, d * .42), (.04, d * .4), (-.06, 0)]), wing * .8)
    put(im, T.poly([(-.16, -.03), (-.38, -.07), (-.38, .05), (-.16, .03)]), wing)
    put(im, T.ell(0, 0, .2, .07), b)
    put(im, T.ell(.2, -.015, .06, .055), b)
    put(im, T.poly([(.25, -.035), (.35, -.01), (.25, .015)]), beakc)
    put(im, T.poly([(.08, -.02), (-.02, d * .5), (-.14, d * .42), (-.1, -.01)]), wing)
    put(im, T.ell(.215, -.025, .014, .014), rgb(.04, .04, .04))


def bird_tall(im, r, T, b, neckc, legc, beakc):
    gy = .5
    put(im, T.seg(-.03, .05, -.05, gy, .024), legc)
    put(im, T.seg(.04, .05, .07, gy, .024), legc)
    put(im, T.ell(-.2, -.02, .1, .1), b)
    put(im, T.ell(0, 0, .23, .155, -.15), b)
    put(im, T.seg(.15, -.05, .2, -.3, .065, .04), neckc)
    put(im, T.seg(.2, -.3, .23, -.5, .04, .035), neckc)
    put(im, T.ell(.25, -.53, .052, .042), neckc)
    put(im, T.poly([(.29, -.555), (.29 + U(r, .08, .22), -.52), (.29, -.5)]), beakc)
    put(im, T.ell(.26, -.54, .012, .012), rgb(.04, .04, .04))


def bird_swim(im, r, T, b, headc, beakc, wl):
    m = (Y < wl).astype(np.float64)
    put(im, T.poly([(-.26, -.06), (-.42, -.16), (-.3, .0)]) * m, b)
    put(im, T.ell(0, -.03, .31, .14) * m, b)
    put(im, T.seg(.2, -.08, .24, -.28, .075, .05), headc)
    put(im, T.ell(.27, -.32, .07, .06), headc)
    put(im, T.poly([(.33, -.34), (.45, -.31), (.33, -.28)]), beakc)
    put(im, T.ell(.29, -.335, .013, .013), rgb(.04, .04, .04))
    put(im, T.seg(-.4, .012, .4, .012, .02), rgb(.9, .93, .95), .4)


def bird_head(im, r, T, b, belly, beakc):
    put(im, T.poly([(-.7, .8), (-.35, .0), (.1, .0), (.3, .8)]), b)
    put(im, T.ell(-.05, -.08, .3, .28), belly)
    put(im, T.poly([(.2, -.17), (.2 + U(r, .2, .42), -.02), (.2, .06)]), beakc)
    put(im, T.ell(.07, -.14, .05, .05), rgb(.85, .7, .2))
    put(im, T.ell(.075, -.14, .028, .028), rgb(.04, .04, .04))


def soften(r, im, p=.5):
    """out-of-focus background"""
    if r.random() < p:
        return boxblur(im, choose(r, 1, 3, 2, 3, 3, 2, 5, 1))
    return im


def fence(r, im, hor):
    c = choose(r, rgb(.9, .9, .88), 1, rgb(.42, .3, .2), 1, rgb(.25, .2, .15), 1)
    y0 = hor + U(r, -.12, .05)
    for i in range(int(U(r, 1, 3.9))):
        put(im, rect(0, y0 + i * .07, 1, y0 + i * .07 + .018), c)
    x = U(r, 0, .2)
    while x < 1:
        put(im, rect(x, y0 - .03, x + .025, y0 + .2), c)
        x += U(r, .2, .4)


def scene_horse(r):
    bgk = choose(r, 'field', 45, 'arena', 18, 'allgrass', 10, 'stable', 8, 'other', 6, 'skyline', 8, 'plain', 5)
    hor = 0.0
    if bgk == 'field':
        im, hor = outdoor(r, gk=choose(r, 'grass', 5, 'drygrass', 1), hor=U(r, .25, .6), back=choose(r, 'trees', 50, 'forest', 20, 'hills', 12, None, 18))
        if r.random() < .3:
            fence(r, im, hor)
    elif bgk == 'arena':
        im, hor = outdoor(r, gk=choose(r, 'dirt', 3, 'sand', 1), hor=U(r, .28, .55), back=choose(r, 'trees', 45, 'buildings', 25, 'forest', 15, None, 15))
        if r.random() < .5:
            fence(r, im, hor)
    elif bgk == 'allgrass':
        im = ground(r, choose(r, 'grass', 5, 'drygrass', 1))
    elif bgk == 'stable':
        im = cf(roomcol(r, choose(r, 'wall', 1, 'floor', 1)) * U(r, .4, 1.0), tex(r, fine=.04, mid=.08, nmid=4, aniso=6, coarse=.08))
        hor = U(r, .55, .8)
        put(im, (Y > hor).astype(np.float64), ground(r, choose(r, 'dirt', 2, 'drygrass', 1, 'concrete', 1)))
    elif bgk == 'other':
        im, hor = outdoor(r, gk=choose(r, 'snow', 1, 'sand', 1), hor=U(r, .3, .55), back=choose(r, 'trees', 2, 'hills', 1, None, 1))
    elif bgk == 'skyline':
        im, hor = outdoor(r, gk=choose(r, 'grass', 3, 'dirt', 1, 'drygrass', 1), hor=U(r, .62, .82), back=choose(r, None, 3, 'hills', 1, 'trees', 1))
    else:
        im = plain(r)
    k = choose(r, 'bay', 34, 'chestnut', 22, 'dark', 12, 'black', 8, 'gray', 14, 'palomino', 6, 'pinto', 4)
    base = jc(r, {'bay': (.42, .24, .13), 'chestnut': (.58, .32, .16), 'dark': (.26, .16, .1), 'black': (.1, .09, .08),
                  'gray': (.82, .8, .76), 'palomino': (.8, .64, .4), 'pinto': (.45, .27, .15)}[k], .03, .15)
    if k in ('bay', 'dark', 'black', 'pinto'):
        dk = rgb(.07, .06, .05)
    elif k == 'chestnut':
        dk = base * U(r, .6, 1.0)
    elif k == 'gray':
        dk = rgb(.6, .6, .58) * U(r, .5, 1.5)
    else:
        dk = rgb(.92, .88, .8)
    low = dk if k == 'bay' else base
    if r.random() < .25:
        low = rgb(.9, .88, .85)
    view = choose(r, 'side', 68, 'head', 26, 'front', 6)
    flip = choose(r, -1, 1, 1, 1)
    if view == 'side':
        s = U(r, .66, .98)
        feet = U(r, max(.76, hor + .1), .96)
        cy = feet - .46 * s
        b = coat(r, base, 'patch' if k == 'pinto' else 'plain', rgb(.92, .9, .86), cy, .3 * s)
        shade(im, ell(U(r, .45, .55), feet, .36 * s, .04 * s), U(r, .5, .8))
        T = Xf(U(r, .42, .58) - .06 * s * flip, cy, s, flip, U(r, -.05, .05))
        horse_side(im, r, T, b, dk, low, choose(r, 'up', 62, 'graze', 14, 'level', 24), r.random() < .24)
    elif view == 'head':
        s = U(r, .7, 1.1)
        b = coat(r, base, 'plain', None, .5, .4)
        horse_head(im, r, Xf(U(r, .42, .6), U(r, .42, .58), s, flip, U(r, -.25, .25)), b, dk)
    else:
        s = U(r, .7, 1.0)
        cy = U(r, .42, .55)
        b = coat(r, base, 'plain', None, cy, .4)
        T = Xf(U(r, .44, .56), cy, s, flip)
        for x in (-.08, .08):
            leg2(im, T, x, .25, .62, .07, .04, b * .9, 0, 0, low)
        put(im, T.ell(0, .12, .17, .23), b)
        put(im, T.seg(0, .0, 0, -.25, .17, .13), b)
        put(im, T.seg(-.035, -.36, -.05, -.46, .035, .015), b)
        put(im, T.seg(.035, -.36, .05, -.46, .035, .015), b)
        put(im, T.seg(0, -.36, 0, -.06, .14, .085), b)
        put(im, T.seg(0, -.4, 0, -.3, .05), dk)
    return im


def scene_deer(r):
    bgk = choose(r, 'forest', 25, 'meadow', 25, 'dry', 21, 'earth', 15, 'snow', 7, 'plain', 2, 'allgrass', 5)
    hor = 0.0
    if bgk == 'forest':
        im = foliage(r, U(r, .3, 1.0))
        im = im * U(r, .6, 1.0)
        for _ in range(int(U(r, 0, 4.9))):
            x = U(r, 0, 1)
            put(im, rect(x, 0, x + U(r, .03, .1), U(r, .5, 1.0)), jc(r, rgb(.25, .18, .12), .04, .4))
        if r.random() < .6:
            hor = U(r, .5, .8)
            put(im, (Y > hor).astype(np.float64), choose(r, ground(r, 'grass'), 2, earth(r), 2, ground(r, 'drygrass'), 1), U(r, .6, 1))
    elif bgk == 'meadow':
        im, hor = outdoor(r, gk='grass', hor=U(r, .15, .52), back=choose(r, 'forest', 50, 'trees', 35, 'hills', 5, None, 10))
    elif bgk == 'dry':
        im, hor = outdoor(r, gk=choose(r, 'drygrass', 3, 'dirt', 1), hor=U(r, .15, .52), back=choose(r, 'forest', 45, 'trees', 30, 'hills', 10, None, 15))
    elif bgk == 'earth':
        im = earth(r)
    elif bgk == 'snow':
        im, hor = outdoor(r, gk='snow', hor=U(r, .15, .5), back=choose(r, 'forest', 55, 'trees', 30, None, 15))
    elif bgk == 'plain':
        im = plain(r)
    else:
        im = ground(r, choose(r, 'grass', 3, 'drygrass', 1))
    im = soften(r, im, .35)
    base = jc(r, choose(r, (.62, .47, .3), 35, (.58, .36, .2), 30, (.48, .4, .32), 25, (.36, .26, .18), 10), .03, .15)
    lt = rgb(.9, .86, .78) * U(r, .8, 1.05)
    ant = None
    if r.random() < .42:
        ant = choose(r, rgb(.75, .68, .55), 1, rgb(.35, .25, .15), 1, rgb(.55, .45, .32), 1)
    view = choose(r, 'side', 56, 'front', 30, 'head', 14)
    flip = choose(r, -1, 1, 1, 1)
    if view == 'side':
        s = U(r, .6, .95)
        feet = U(r, max(.74, hor + .1), .97)
        cy = feet - .45 * s
        b = coat(r, base, 'plain', None, cy, .3 * s)
        T = Xf(U(r, .42, .58) - .05 * s * flip, cy, s, flip, U(r, -.05, .05))
        deer_side(im, r, T, b, lt, ant, choose(r, 'up', 85, 'graze', 15))
    elif view == 'front':
        s = U(r, .7, 1.0)
        cy = U(r, .4, .55)
        deer_front(im, r, Xf(U(r, .42, .58), cy, s, flip, U(r, -.08, .08)), coat(r, base, 'plain', None, cy + .2, .4), lt, ant)
    else:
        s = U(r, 1.2, 1.8)
        cy = U(r, .55, .75)
        if ant is None and r.random() < .6:
            ant = rgb(.6, .5, .38)
        deer_front(im, r, Xf(U(r, .4, .6), cy, s, flip, U(r, -.12, .12)), coat(r, base, 'plain', None, cy + .2, .5), lt, ant)
    return im


def pet_bg(r, kinds):
    bgk = choose(r, *kinds)
    if bgk == 'indoor':
        im = indoor(r)
    elif bgk == 'grass':
        if r.random() < .6:
            im = ground(r, 'grass')
        else:
            im, _ = outdoor(r, gk='grass', hor=U(r, .1, .45), back=choose(r, 'trees', 3, 'forest', 3, 'buildings', 1, None, 1))
    elif bgk == 'pave':
        k = choose(r, 'road', 2, 'concrete', 2, 'dirt', 1, 'sand', 1, 'snow', 1)
        if r.random() < .55:
            im = ground(r, k)
        else:
            im, _ = outdoor(r, gk=k, hor=U(r, .1, .45), back=choose(r, 'trees', 2, 'buildings', 2, 'hills', 1, None, 2))
    elif bgk == 'plain':
        im = plain(r)
    elif bgk == 'foliage':
        im = foliage(r)
    else:
        im = earth(r)
    return soften(r, im, .45)


def scene_cat(r):
    im = pet_bg(r, ('indoor', 64, 'grass', 9, 'foliage', 7, 'pave', 6, 'earth', 4, 'plain', 10))
    k = choose(r, 'tabby', 20, 'btabby', 14, 'orange', 14, 'black', 12, 'white', 10, 'gray', 8, 'cream', 5, 'siamese', 5, 'tuxedo', 6, 'calico', 6)
    base = jc(r, {'tabby': (.5, .47, .42), 'btabby': (.5, .4, .3), 'orange': (.82, .52, .22), 'black': (.08, .08, .08), 'white': (.92, .9, .88),
                  'gray': (.5, .52, .55), 'cream': (.88, .78, .62), 'siamese': (.86, .78, .66), 'tuxedo': (.08, .08, .08), 'calico': (.92, .9, .86)}[k], .02, .12)
    lt = base
    if k in ('tabby', 'btabby'):
        lt = choose(r, np.clip(base * 1.5, 0, 1), 2, rgb(.92, .9, .86), 1)
    elif k == 'orange':
        lt = choose(r, rgb(.95, .85, .7), 1, rgb(.93, .92, .9), 1)
    elif k in ('tuxedo', 'calico'):
        lt = rgb(.93, .92, .9)
    elif k == 'siamese':
        lt = rgb(.3, .22, .18)
    eyec = jc(r, choose(r, rgb(.8, .7, .2), 3, rgb(.5, .7, .3), 3, rgb(.4, .6, .85), 1, rgb(.8, .5, .15), 2), .05)
    kind = 'tabby' if k in ('tabby', 'btabby') or (k == 'orange' and r.random() < .6) else ('patch' if k == 'calico' else 'plain')
    view = choose(r, 'face', 36, 'sit', 24, 'lie', 26, 'side', 14)
    flip = choose(r, -1, 1, 1, 1)
    cx, cy = U(r, .4, .6), U(r, .4, .6)
    b = coat(r, base, kind, jc(r, choose(r, rgb(.8, .5, .2), 1, rgb(.1, .1, .1), 1), .03), cy, .35)
    if view == 'face':
        cat_face(im, r, Xf(cx, U(r, .36, .56), U(r, .75, 1.3), flip, U(r, -.3, .3)), b, lt, eyec)
    elif view == 'sit':
        cat_sit(im, r, Xf(cx, U(r, .44, .56), U(r, .8, 1.1), flip, U(r, -.1, .1)), b, lt, eyec)
    elif view == 'lie':
        cat_lie(im, r, Xf(cx, U(r, .45, .65), U(r, .8, 1.15), flip, U(r, -.15, .15)), b, lt, eyec)
    else:
        cat_side(im, r, Xf(cx, U(r, .45, .62), U(r, .8, 1.1), flip, U(r, -.06, .06)), b, lt, eyec)
    return im


def scene_dog(r):
    im = pet_bg(r, ('indoor', 40, 'grass', 25, 'pave', 14, 'plain', 8, 'foliage', 7, 'earth', 6))
    k = choose(r, 'white', 18, 'golden', 20, 'tan', 12, 'brown', 12, 'black', 14, 'gray', 5, 'cream', 7, 'blacktan', 6, 'patched', 6)
    base = jc(r, {'white': (.9, .88, .84), 'golden': (.82, .62, .35), 'tan': (.7, .5, .3), 'brown': (.42, .27, .16), 'black': (.1, .09, .09),
                  'gray': (.55, .53, .5), 'cream': (.9, .82, .68), 'blacktan': (.1, .09, .09), 'patched': (.92, .9, .86)}[k], .02, .12)
    lt = np.clip(base * U(r, 1.0, 1.3), 0, 1)
    if k == 'blacktan':
        lt = rgb(.7, .48, .28)
    elif k in ('brown', 'black', 'tan') and r.random() < .3:
        lt = rgb(.92, .9, .86)
    ec = base * U(r, .6, 1.0)
    second = jc(r, choose(r, rgb(.42, .27, .16), 2, rgb(.1, .09, .09), 2, rgb(.7, .5, .3), 1), .03)
    if k in ('white', 'patched') and r.random() < .6:
        ec = second
    erect = r.random() < .3
    view = choose(r, 'face', 38, 'sit', 22, 'side', 25, 'lie', 15)
    flip = choose(r, -1, 1, 1, 1)
    cx, cy = U(r, .4, .6), U(r, .4, .6)
    b = coat(r, base, 'patch' if k == 'patched' else 'plain', second, cy, .35)
    if view == 'face':
        dog_face(im, r, Xf(cx, U(r, .36, .56), U(r, .85, 1.4), flip, U(r, -.3, .3)), b, lt, ec, erect)
    elif view == 'sit':
        dog_sit(im, r, Xf(cx, U(r, .44, .56), U(r, .85, 1.15), flip, U(r, -.1, .1)), b, lt, ec, erect)
    elif view == 'side':
        s = U(r, .7, 1.0)
        feet = U(r, .72, .95)
        dog_side(im, r, Xf(cx - .05 * s * flip, feet - .34 * s, s, flip, U(r, -.06, .06)), b, lt, ec, erect)
    else:
        dog_lie(im, r, Xf(cx, U(r, .45, .65), U(r, .8, 1.15), flip, U(r, -.12, .12)), b, lt, ec, erect)
    return im


def scene_frog(r):
    bgk = choose(r, 'foliage', 30, 'earth', 26, 'pond', 8, 'white', 10, 'stone', 6, 'dark', 6, 'skin', 4, 'mixed', 10)
    if bgk == 'foliage':
        im = foliage(r, U(r, .1, .8))
    elif bgk == 'earth':
        im = earth(r)
    elif bgk == 'pond':
        im = water(r, choose(r, 'green', 3, 'dark', 2, 'gray', 1))
    elif bgk == 'white':
        im = plain(r, 'white')
    elif bgk == 'stone':
        im = cf(rgb(*([U(r, .35, .7)] * 3)) * rgb(1.03, 1, .96), tex(r, fine=U(r, .04, .1), mid=U(r, .05, .15), coarse=.08))
    elif bgk == 'dark':
        im = plain(r, 'black')
    elif bgk == 'skin':
        im = cf(jc(r, rgb(.85, .65, .55), .04, .15), tex(r, fine=.02, mid=.03, coarse=.06))
    else:
        im = earth(r)
        put(im, np.clip(snoise(r, U(r, 1.5, 4)) * 3, 0, 1), foliage(r, U(r, .2, .8)))
    im = soften(r, im, .4)
    if r.random() < .22:
        put(im, ell(U(r, .35, .65), U(r, .5, .75), U(r, .35, .6), U(r, .2, .4), U(r, -.5, .5)),
            cf(rgb(U(r, .2, .45), U(r, .45, .65), U(r, .1, .28)), tex(r, fine=.04, mid=.05, coarse=.06)))
    k = choose(r, 'green', 32, 'bright', 14, 'olive', 16, 'brown', 16, 'tan', 6, 'dark', 5, 'yellow', 4, 'red', 4, 'blue', 3)
    base = jc(r, {'green': (.35, .6, .15), 'bright': (.45, .75, .2), 'olive': (.42, .45, .2), 'brown': (.45, .33, .2), 'tan': (.65, .55, .35),
                  'dark': (.2, .22, .12), 'yellow': (.85, .75, .2), 'red': (.85, .3, .1), 'blue': (.15, .35, .8)}[k], .04, .15)
    lt = jc(r, choose(r, rgb(.9, .88, .7), 3, rgb(.93, .93, .9), 2, rgb(.85, .8, .4), 2, base * 1.2, 2), .02)
    eyec = choose(r, rgb(.85, .65, .15), 4, rgb(.8, .15, .1), 1, rgb(.9, .45, .1), 2, rgb(.15, .12, .1), 2)
    cx, cy = U(r, .4, .6), U(r, .42, .62)
    b = coat(r, base, choose(r, 'spots', 1, 'plain', 1), None, cy, .25)
    shade(im, ell(cx, cy + .12, .33, .12), U(r, .55, .9))
    view = choose(r, 'side', 45, 'front', 30, 'top', 25)
    flip = choose(r, -1, 1, 1, 1)
    if view == 'side':
        frog_side(im, r, Xf(cx, cy, U(r, .85, 1.35), flip, U(r, -.3, .3)), b, lt, eyec)
    elif view == 'front':
        frog_front(im, r, Xf(cx, cy, U(r, .8, 1.3), flip, U(r, -.2, .2)), b, lt, eyec)
    else:
        frog_top(im, r, Xf(cx, cy, U(r, .85, 1.35), flip, U(r, 0, 2 * PI)), b, lt, eyec)
    return im


def scene_bird(r):
    pose = choose(r, 'perch', 46, 'fly', 16, 'tall', 13, 'swim', 9, 'head', 6, 'stand', 10)
    k = choose(r, 'brown', 26, 'gray', 16, 'black', 12, 'white', 12, 'blue', 8, 'red', 6, 'yellow', 8, 'green', 6, 'orange', 3, 'darkblue', 3)
    base = jc(r, {'brown': (.45, .33, .22), 'gray': (.5, .5, .52), 'black': (.1, .1, .11), 'white': (.92, .92, .9), 'blue': (.2, .4, .8),
                  'red': (.8, .15, .12), 'yellow': (.9, .8, .2), 'green': (.3, .6, .25), 'orange': (.9, .5, .15), 'darkblue': (.1, .15, .4)}[k], .04, .15)
    belly = choose(r, np.clip(base * 1.3, 0, 1), 30, rgb(.93, .92, .9), 25, rgb(.9, .84, .7), 15, rgb(.9, .8, .3), 10, rgb(.85, .45, .2), 10, base, 10)
    wing = choose(r, base * U(r, .55, .9), 3, rgb(.12, .11, .1), 1, rgb(.35, .3, .26), 1)
    beakc = choose(r, rgb(.9, .75, .2), 35, rgb(.9, .5, .15), 20, rgb(.15, .13, .12), 35, rgb(.8, .2, .15), 5, rgb(.6, .6, .6), 5)
    flip = choose(r, -1, 1, 1, 1)
    cx, cy = U(r, .42, .58), U(r, .42, .58)
    if pose in ('perch', 'head'):
        bgk = choose(r, 'foliage', 45, 'sky', 22, 'earth', 10, 'plain', 7, 'outdoor', 8, 'water', 8)
        if bgk == 'foliage':
            im = foliage(r, U(r, 0, .6))
        elif bgk == 'sky':
            im = sky(r)
        elif bgk == 'earth':
            im = earth(r)
        elif bgk == 'plain':
            im = plain(r)
        elif bgk == 'water':
            im = water(r)
        else:
            im, _ = outdoor(r, gk=choose(r, 'grass', 2, 'drygrass', 1, 'water', 1), hor=U(r, .4, .8), back=choose(r, 'trees', 2, 'forest', 1, None, 2))
        im = soften(r, im, .6)
        if pose == 'head':
            bird_head(im, r, Xf(cx, U(r, .45, .6), U(r, .8, 1.2), flip, U(r, -.2, .2)), shd(r, base, .7, .3), shd(r, belly, .4, .3), beakc)
            return im
        s = U(r, .55, .98)
        py = cy + U(r, .2, .26) * s
        pk = choose(r, 'branch', 60, 'post', 15, 'wire', 10, 'rock', 15)
        bc = jc(r, choose(r, rgb(.36, .26, .17), 3, rgb(.2, .15, .1), 1, rgb(.5, .45, .4), 1), .03, .2)
        if pk == 'branch':
            a = U(r, -.25, .25)
            put(im, seg(-.1, py + .02 + a * .6, 1.1, py + .02 - a * .6, U(r, .035, .09)), cf(bc, tex(r, fine=.08, mid=.1)))
            if r.random() < .5:
                x = U(r, .1, .9)
                put(im, seg(x, py + .02, x + U(r, -.3, .3), py - U(r, .3, .8), U(r, .015, .035)), bc)
        elif pk == 'post':
            w = U(r, .08, .2)
            put(im, rect(cx - w / 2, py, cx + w / 2, 1.1), cf(bc, tex(r, fine=.06, mid=.08, nmid=3, aniso=5)))
        elif pk == 'wire':
            put(im, seg(-.1, py + .005, 1.1, py + .005 + U(r, -.05, .05), .014), rgb(.12, .12, .12))
        else:
            put(im, ell(cx + U(r, -.1, .1), py + .25, U(r, .3, .6), .26), cf(jc(r, rgb(.5, .48, .45), .04, .3), tex(r, fine=.06, mid=.1)))
        bird_perch(im, r, Xf(cx + .04 * s * flip, cy, s, flip), shd(r, base, cy, .2 * s), shd(r, belly, cy, .2 * s), wing, beakc, (py - cy) / s)
    elif pose == 'stand':
        gk = choose(r, 'grass', 40, 'dirt', 12, 'snow', 8, 'concrete', 10, 'sand', 15, 'drygrass', 15)
        lay = choose(r, 'ground', 35, 'outdoor', 25, 'ledge', 40)
        s = U(r, .55, .95)
        if lay == 'ground':
            im = ground(r, gk)
        elif lay == 'outdoor':
            im, _ = outdoor(r, gk=gk, hor=U(r, .15, .5), back=choose(r, 'trees', 2, 'forest', 2, None, 2, 'hills', 1))
        else:
            # standing on a ledge, wall, railing or patch of ground with something else behind
            bk = choose(r, 'foliage', 30, 'sky', 20, 'earth', 15, 'wall', 20, 'water', 8, 'plain', 7)
            if bk == 'foliage':
                im = foliage(r, U(r, 0, .6))
            elif bk == 'sky':
                im = sky(r)
            elif bk == 'earth':
                im = earth(r)
            elif bk == 'wall':
                im = cf(roomcol(r, 'wall'), tex(r, fine=U(r, .02, .08), mid=U(r, .02, .1), coarse=.05))
            elif bk == 'water':
                im = water(r)
            else:
                im = plain(r)
            im = soften(r, im, .5)
            sf = choose(r, ground(r, gk), 40, earth(r), 30, cf(roomcol(r, 'floor'), tex(r, fine=.05, mid=.06, nmid=4, aniso=5)), 30)
            put(im, (Y > cy + U(r, .16, .3) * s + U(r, -.1, .1) * (X - .5)).astype(np.float64), sf)
        im = soften(r, im, .4)
        bird_perch(im, r, Xf(cx + .04 * s * flip, cy, s, flip), shd(r, base, cy, .2 * s), shd(r, belly, cy, .2 * s), wing, beakc, U(r, .2, .26))
    elif pose == 'fly':
        bgk = choose(r, 'sky', 74, 'foliage', 10, 'water', 8, 'outdoor', 8)
        if bgk == 'sky':
            im = sky(r)
        elif bgk == 'foliage':
            im = foliage(r, U(r, 0, .4))
        elif bgk == 'water':
            im = water(r)
        else:
            im, _ = outdoor(r, gk=choose(r, 'grass', 1, 'water', 2), hor=U(r, .6, .9), back=choose(r, 'trees', 1, 'hills', 1, None, 2))
        style = choose(r, 'soar', 40, 'up', 35, 'down', 25)
        if k == 'white' or r.random() < .3:
            wing = base * .85
        ang = U(r, 0, 2 * PI) if style == 'soar' else U(r, -.4, .4)
        bird_fly(im, r, Xf(cx, cy, U(r, .6, 1.05), flip, ang, U(r, .5, 1) if style == 'soar' else 1.0), base, wing, beakc, style)
    elif pose == 'tall':
        bgk = choose(r, 'outdoor', 60, 'allgrass', 18, 'foliage', 12, 'earth', 10)
        hor = 0.0
        if bgk == 'outdoor':
            im, hor = outdoor(r, gk=choose(r, 'grass', 3, 'drygrass', 2, 'dirt', 1, 'sand', 1, 'water', 2), hor=U(r, .25, .65),
                              back=choose(r, 'trees', 3, 'forest', 2, 'hills', 1, None, 3))
        elif bgk == 'allgrass':
            im = ground(r, choose(r, 'grass', 2, 'drygrass', 1))
        elif bgk == 'foliage':
            im = foliage(r)
        else:
            im = earth(r)
        im = soften(r, im, .4)
        s = U(r, .7, .98)
        feet = U(r, max(.8, hor + .1), .98)
        neckc = choose(r, base, 2, rgb(.8, .65, .6), 1, rgb(.92, .92, .9), 1)
        legc = choose(r, rgb(.2, .17, .15), 2, rgb(.8, .6, .55), 1, rgb(.85, .7, .2), 1)
        bird_tall(im, r, Xf(cx - .05 * s * flip, feet - .5 * s, s, flip, U(r, -.05, .05)), shd(r, base, feet - .5 * s, .2), neckc, legc, beakc)
    else:
        im = water(r)
        if r.random() < .4:
            hb = U(r, .1, .35)
            put(im, (Y < hb).astype(np.float64), choose(r, foliage(r, .3), 2, earth(r), 1, sky(r), 1))
        im = soften(r, im, .3)
        s = U(r, .6, 1.0)
        wl = U(r, .5, .75)
        headc = choose(r, base, 3, rgb(.1, .3, .2), 1, rgb(.93, .93, .9), 1, rgb(.12, .1, .1), 1)
        bird_swim(im, r, Xf(cx, wl, s, flip), shd(r, base, wl - .1, .2), headc, beakc, wl)
    return im


SCENES = [scene_airplane, scene_automobile, scene_bird, scene_cat, scene_deer, scene_dog, scene_frog, scene_horse, scene_ship, scene_truck]


def draw(cls, idx):
    """drawing number idx of class cls: a (3, 32, 32) picture with values 0..1"""
    r = np.random.default_rng([20240917, cls, idx])
    return finish(r, SCENES[cls](r)).transpose(2, 0, 1)


# ----------------------------------------------------------------------------------------------
# the network's operations (same definitions as the task statement), used here only to turn each
# drawing into its template
# ----------------------------------------------------------------------------------------------
NORM_EPS = 1e-5


def op_conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def op_pool(x, size, stride, mode):
    n, c, h, w = x.shape
    (sh, sw), (th, tw) = size, stride
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if mode == 'max' else win.mean(0)


def run_layers(layers, weights, images):
    vals = {'image': images}
    for L in layers:
        ins = [vals[i] for i in L['inputs']]
        op, name = L['op'], L['name']
        if op == 'conv':
            vals[name] = op_conv(ins[0], weights[name + '.W'], weights[name + '.b'], L.get('stride', 1), L.get('padding', 0))
        elif op == 'dense':
            vals[name] = ins[0].reshape(len(ins[0]), -1) @ weights[name + '.W'].T + weights[name + '.b']
        elif op == 'relu':
            vals[name] = np.maximum(ins[0], 0)
        elif op in ('maxpool', 'avgpool'):
            vals[name] = op_pool(ins[0], L['size'], L['stride'], op[:3])
        elif op == 'add':
            vals[name] = sum(ins)
        elif op == 'reshape':
            vals[name] = ins[0].reshape((len(ins[0]),) + tuple(L['shape']))
        elif op == 'norm':
            x = ins[0]
            mu = x.mean(axis=(2, 3), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + NORM_EPS)
    return vals


# ----------------------------------------------------------------------------------------------
# part 1 of the network: fixed features from formulas
# ----------------------------------------------------------------------------------------------
N_HUE = 12            # hue bins, 30 degrees apart
HUE_WIDTH = 45.0      # a hue bin answers to hues up to this many degrees away: forgiving of the +-15..20 degree
                      # spread of one material between photos, still separating materials 35+ degrees apart
LIGHT_WIDTH = 0.5     # a lightness bin (centres 0, .25, .5, .75, 1) answers to lightness up to this far away
HUE_GAIN = 5.0        # chroma 0.2 already counts as fully "tinted"
VIVID_AT = 0.3        # chroma above this counts as "vivid"
EDGE_CLIP = 1.0       # edge strength (in units of the picture's own contrast) at which an edge counts as full
EDGE_FLOOR = 0.03     # spread of pooled edge strength below which a picture counts as having no edge layout
LUMA = (0.299, 0.587, 0.114)


def feature_layers():
    layers, W = [], {}

    def add(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=inputs, **kw))

    def conv(name, inp, w, b, pad=0):
        w = np.asarray(w, dtype=np.float64)
        add(name, 'conv', [inp], out=int(w.shape[0]), kernel=[int(w.shape[2]), int(w.shape[3])], stride=1, padding=pad)
        W[name + '.W'], W[name + '.b'] = w, np.asarray(b, dtype=np.float64)

    # --- colour: chroma plane cx = R - (G+B)/2, cy = sqrt(3)/2 (G - B); hue angle 0 = red, 120 = green, 240 = blue
    w1, b1 = np.zeros((N_HUE + 8, 3)), np.zeros(N_HUE + 8)
    for k in range(N_HUE):
        t = 2 * PI * k / N_HUE
        w1[k] = [np.cos(t), -np.cos(t) / 2 + np.sin(t) * np.sqrt(3) / 2, -np.cos(t) / 2 - np.sin(t) * np.sqrt(3) / 2]
    lum = np.array(LUMA)
    w1[N_HUE + 0], b1[N_HUE + 0] = -lum / LIGHT_WIDTH, 1.0                    # lightness tent centred on 0 (dark)
    w1[N_HUE + 1], b1[N_HUE + 1] = lum / LIGHT_WIDTH, 1.0 - 1.0 / LIGHT_WIDTH   # lightness tent centred on 1 (bright)
    for j, c in enumerate((.25, .5, .75)):                # |Y - c| in two halves
        w1[N_HUE + 2 + 2 * j], b1[N_HUE + 2 + 2 * j] = lum, -c
        w1[N_HUE + 3 + 2 * j], b1[N_HUE + 3 + 2 * j] = -lum, c
    conv('c1', 'image', w1[:, :, None, None], b1)
    add('c1r', 'relu', ['c1'])
    # hue tents: chroma along the bin direction minus cot(HUE_WIDTH) times chroma across it, which is zero
    # HUE_WIDTH degrees away from the bin; neighbouring bins overlap so that nearby hues count as similar
    w2, b2 = np.zeros((N_HUE + 5, N_HUE + 8)), np.zeros(N_HUE + 5)
    q = N_HUE // 4
    cot = 1.0 / np.tan(np.deg2rad(HUE_WIDTH))
    for h in range(N_HUE):
        w2[h, h] += 1
        w2[h, (h + 2 * q) % N_HUE] -= 1
        w2[h, (h + q) % N_HUE] -= cot
        w2[h, (h + 3 * q) % N_HUE] -= cot
    w2[N_HUE + 0, N_HUE + 0] = 1                          # dark
    for j in range(3):                                    # tents at .25, .5, .75
        w2[N_HUE + 1 + j, N_HUE + 2 + 2 * j] = -1 / LIGHT_WIDTH
        w2[N_HUE + 1 + j, N_HUE + 3 + 2 * j] = -1 / LIGHT_WIDTH
        b2[N_HUE + 1 + j] = 1
    w2[N_HUE + 4, N_HUE + 1] = 1                          # bright
    conv('c2', 'c1r', w2[:, :, None, None], b2)
    add('c2r', 'relu', ['c2'])
    # hue features: (1 - clipped tint) and vivid part, per hue bin
    w3, b3 = np.zeros((2 * N_HUE, N_HUE + 5)), np.zeros(2 * N_HUE)
    for h in range(N_HUE):
        w3[h, h], b3[h] = -HUE_GAIN, 1.0
        w3[N_HUE + h, h], b3[N_HUE + h] = 2.0, -2.0 * VIVID_AT
    conv('hue', 'c2r', w3[:, :, None, None], b3)
    add('huer', 'relu', ['hue'])
    add('hp', 'avgpool', ['huer'], size=[8, 8], stride=[8, 8])
    w4 = np.zeros((5, N_HUE + 5))
    for j in range(5):
        w4[j, N_HUE + j] = 1
    conv('lig', 'c2r', w4[:, :, None, None], np.zeros(5))
    add('ligr', 'relu', ['lig'])
    add('lp', 'avgpool', ['ligr'], size=[8, 8], stride=[8, 8])

    # --- edges: luminance, standardised over the picture, Gaussian-derivative filters in 4 directions
    conv('lum', 'image', np.array(LUMA)[None, :, None, None], np.zeros(1))
    add('lumn', 'norm', ['lum'])
    add('lumh', 'avgpool', ['lumn'], size=[2, 2], stride=[2, 2])
    yy, xx = np.mgrid[-2:3, -2:3].astype(np.float64)
    g = np.exp(-(xx * xx + yy * yy) / 2.0)
    bank = []
    for a in (0, 45, 90, 135):
        t = np.deg2rad(a)
        f = (np.cos(t) * xx + np.sin(t) * yy) * g
        f = 2 * f / np.abs(f).sum()                       # a unit step edge gives response 1
        bank += [f, -f]
    bank = np.array(bank)[:, None]
    we = np.zeros((4, 8))
    for o in range(4):                                    # |gradient along o| - |gradient across o|
        we[o, 2 * o] = we[o, 2 * o + 1] = 1
        we[o, 2 * ((o + 2) % 4)] = we[o, 2 * ((o + 2) % 4) + 1] = -1
    for tag, src, cell in (('e', 'lumn', 7), ('f', 'lumh', 3)):
        conv(tag + '1', src, bank, np.zeros(8))
        add(tag + '1r', 'relu', [tag + '1'])
        conv(tag + '2', tag + '1r', we[:, :, None, None], np.zeros(4))
        add(tag + '2r', 'relu', [tag + '2'])
        conv(tag + '3', tag + '2r', -np.eye(4)[:, :, None, None] / EDGE_CLIP, np.ones(4))   # 1 - clipped edge strength
        add(tag + '3r', 'relu', [tag + '3'])
        add(tag + 'p', 'avgpool', [tag + '3r'], size=[cell, cell], stride=[cell, cell])
    # fine and coarse edge maps side by side, each standardised over its own 64 numbers, so that only the
    # layout of edges counts and not how much texture the picture has (EDGE_FLOOR keeps flat pictures quiet)
    a = np.sqrt(NORM_EPS) / EDGE_FLOOR
    sel = np.zeros((2, 8, 4))
    sel[0, :4], sel[1, 4:] = a * np.eye(4), a * np.eye(4)
    conv('ea', 'ep', sel[0][:, :, None, None], np.zeros(8))
    conv('eb', 'fp', sel[1][:, :, None, None], np.zeros(8))
    add('ec', 'add', ['ea', 'eb'])
    add('ed', 'reshape', ['ec'], shape=[2, 64, 1])
    add('en', 'norm', ['ed'])
    return layers, W


FEATS = ('hp', 'lp', 'en')                # hue (24,4,4), lightness (5,4,4), edges (2,64,1) = (8,4,4)
GROUP = {'hp': 'sh', 'lp': 'sl', 'en': 'se'}
KAPPA = 50.0


def template_of(f):
    """one drawing's feature block (C,4,4) -> weights and bias that score closeness to it.
    Closeness is minus half the squared distance, taken on the 4x4 grid, on its 2x2 block means and on its
    overall mean (each level averaged over its cells), up to a term that does not depend on the template."""
    f = f.reshape(-1, 4, 4)
    c = f.shape[0]
    f2 = f.reshape(c, 2, 2, 2, 2).mean(axis=(2, 4))
    f1 = f.mean(axis=(1, 2))
    w = (f + np.repeat(np.repeat(f2, 2, 1), 2, 2) + f1[:, None, None]) * (KAPPA / 16)
    b = -KAPPA / 2 * ((f * f).sum() / 16 + (f2 * f2).sum() / 4 + (f1 * f1).sum())
    return w.reshape(-1), b


# ----------------------------------------------------------------------------------------------
# part 2: one template per drawing, then the vote
# ----------------------------------------------------------------------------------------------
PER_CLASS = 3000
MIX = {'zh': 5, 'zl': 2, 'ze': 2}        # relative say of hue layout, lightness layout, edge layout
KERNELS = ((0.5, 2.0), (1.5, 1.0))       # (reach below the best match in standard deviations, weight)


def build(out_dir):
    layers, W = feature_layers()
    T = 10 * PER_CLASS
    tw = {k: [] for k in FEATS}
    tb = {k: [] for k in FEATS}
    B = 250
    for cls in range(10):
        for s in range(0, PER_CLASS, B):
            ims = np.stack([draw(cls, i) for i in range(s, min(PER_CLASS, s + B))])
            vals = run_layers(layers, W, ims)
            for k in FEATS:
                for f in vals[k]:
                    w, b = template_of(f)
                    tw[k].append(w)
                    tb[k].append(b)

    def add(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=inputs, **kw))

    for k in FEATS:
        add(GROUP[k], 'dense', [k], out=T)
        W[GROUP[k] + '.W'], W[GROUP[k] + '.b'] = np.array(tw[k]), np.array(tb[k])
    for g in ('h', 'l', 'e'):
        add('r' + g, 'reshape', ['s' + g], shape=[1, T, 1])
        add('z' + g, 'norm', ['r' + g])                  # standardise the closeness scores across templates
    add('zs', 'add', [k for k, n in MIX.items() for _ in range(n)])
    add('z', 'norm', ['zs'])
    add('zmax', 'maxpool', ['z'], size=[T, 1], stride=[T, 1])
    cls_of = np.repeat(np.arange(10), PER_CLASS)
    outs = []
    for i, (reach, weight) in enumerate(KERNELS):
        add('k%d' % i, 'dense', ['zmax'], out=T)         # reach - best score, copied to every template
        W['k%d.W' % i], W['k%d.b' % i] = -np.ones((T, 1)), np.full(T, reach)
        add('kr%d' % i, 'reshape', ['k%d' % i], shape=[1, T, 1])
        add('a%d' % i, 'add', ['z', 'kr%d' % i])
        add('v%d' % i, 'relu', ['a%d' % i])               # templates within reach of the best one vote
        add('o%d' % i, 'dense', ['v%d' % i], out=10)
        W['o%d.W' % i] = weight * (cls_of[None, :] == np.arange(10)[:, None]).astype(np.float64)
        W['o%d.b' % i] = np.zeros(10)
        outs.append('o%d' % i)
    add('out', 'add', outs)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'network.json'), 'w') as fh:
        json.dump({'layers': layers}, fh, indent=1)
    np.savez(os.path.join(out_dir, 'weights.npz'), **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
