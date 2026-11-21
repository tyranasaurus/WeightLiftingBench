"""build.py - writes network.json and weights.npz for the 32x32 photo classifier.

Usage: python3 build.py OUT_FOLDER        (reads no files, no data; numpy only)

How the weights are made (no training, no fitting, no statistics of any images):
  * Front end: fixed formulas - a lattice of soft colour categories in lightness / red-green / yellow-blue space,
    oriented derivative-of-Gaussian edge filters at three scales, pooling grids, per-picture normalisation.
  * Templates: this program DRAWS scenes of the ten classes from shape models typed from general knowledge
    (fuselage + swept wings, hull + superstructure, car body + dark wheels, animal bodies from ellipses and legs ...)
    in typed colour ranges and typed scene layouts, with a fixed random seed. Each drawing becomes its own template:
    one row of the dense layer "sim" is the front-end descriptor of one drawing, normalised by itself alone (by the
    very norm layers of the network). No number is computed across drawings.
  * Read-out: a soft nearest-template vote per class with hand-chosen thresholds.
"""
import json
import sys
import numpy as np

N_PER_CLASS = 5000
SEED = 20261120
# ---------------------------------------------------------------------------------------------
# PART 1. Drawing toolkit: soft-edged shapes, procedural textures, backgrounds, photo effects.
# Every picture is drawn from typed shape models and typed colour ranges (general knowledge of
# what the classes look like); nothing here reads any data.
# ---------------------------------------------------------------------------------------------

S = 32
_ax = (np.arange(S) + 0.5) / S
XX, YY = np.meshgrid(_ax, _ax)          # XX[y, x] = x in [0,1], YY[y, x] = y in [0,1] (y grows downward)
_IM = {}


def _interp(g):
    """(S, g) smooth interpolation matrix from g grid nodes to S pixels."""
    if g not in _IM:
        if g >= S:
            _IM[g] = np.eye(S)
        elif g == 1:
            _IM[g] = np.ones((S, 1))
        else:
            pos = _ax * (g - 1)
            i0 = np.clip(np.floor(pos).astype(int), 0, g - 2)
            f = pos - i0
            f = f * f * (3 - 2 * f)
            A = np.zeros((S, g))
            A[np.arange(S), i0] = 1 - f
            A[np.arange(S), i0 + 1] = f
            _IM[g] = A
    return _IM[g]


def noise(rs, gy, gx=None):
    """Smooth random field in [-1, 1]: a gy x gx grid of random values interpolated to S x S."""
    gx = gy if gx is None else gx
    gy, gx = min(gy, S), min(gx, S)
    return _interp(gy) @ rs.uniform(-1, 1, (gy, gx)) @ _interp(gx).T


def fbm(rs, g, octaves=3, pers=0.5):
    f = np.zeros((S, S)); a = 1.0; tot = 0.0
    for _ in range(octaves):
        f += a * noise(rs, g); tot += a; a *= pers; g *= 2
    return f / tot


def pink(rs, g0=3, octaves=3, gx_mul=1.0):
    """Texture with equal contrast in every octave, as surfaces in photographs have (about 1/f amplitude spectrum).
    gx_mul stretches it: > 1 gives vertical streaks (grass blades, bark), < 1 horizontal streaks (ripples)."""
    f = np.zeros((S, S)); g = g0
    for _ in range(octaves):
        f += noise(rs, g, max(2, int(round(g * gx_mul)))); g *= 2
    return f / np.sqrt(octaves)


def nat(rs, amp, g0=3, chroma=0.25, gx_mul=1.0, octaves=3):
    """(S, S, 3) multiplicative surface texture: brightness variation plus a smaller variation of colour."""
    t = amp * pink(rs, g0, octaves, gx_mul)[..., None]
    if chroma > 0:
        t = t + (amp * chroma) * np.stack([pink(rs, g0, 2, gx_mul) for _ in range(3)], axis=2)
    return t


def C(*v):
    return np.array(v, float)


def lerp(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def U(rs, a, b):
    return rs.uniform(a, b)


def jit(rs, c, s=0.04, v=0.12):
    """Colour jitter: brightness factor 1 +- v and per-channel shift +- s."""
    return np.clip(np.asarray(c, float) * (1 + rs.uniform(-v, v)) + rs.uniform(-s, s, 3), 0, 1)


def pick(rs, items, p=None):
    if p is not None:
        p = np.asarray(p, float); p = p / p.sum()
    return items[rs.choice(len(items), p=p)]


def chance(rs, p):
    return rs.uniform() < p


class Cv:
    """Canvas with a local drawing frame. Shapes are given in local units and come back as soft masks."""

    def __init__(self, rs):
        self.rs = rs
        self.im = np.zeros((S, S, 3))
        self.frame(0.5, 0.5, 1.0)

    def frame(self, cx, cy, sc, ang=0.0, flip=1, sy=1.0, sx=1.0):
        """Local origin at image point (cx, cy); sc image-widths per local unit; rotation ang; flip=-1 mirrors x;
        sy / sx squeeze the local y / x axis (foreshortening)."""
        dx = (XX - cx) / sc; dy = (YY - cy) / sc
        c, n = np.cos(ang), np.sin(ang)
        self.lx = (c * dx + n * dy) * flip / sx
        self.ly = (-n * dx + c * dy) / sy
        self.px = sc * S * min(1.0, sy, sx)
        self.fr = (cx, cy, sc, ang, flip, sy)

    def to_img(self, x, y):
        cx, cy, sc, ang, flip, sy = self.fr
        c, n = np.cos(ang), np.sin(ang)
        x = x * flip; y = y * sy
        return cx + sc * (c * x - n * y), cy + sc * (n * x + c * y)

    def aa(self, d, soft=1.0):
        return np.clip(0.5 - d * self.px / soft, 0.0, 1.0)

    def ell(self, x, y, rx, ry, a=0.0, soft=1.0):
        dx = self.lx - x; dy = self.ly - y
        if a != 0.0:
            c, n = np.cos(a), np.sin(a); dx, dy = c * dx + n * dy, -n * dx + c * dy
        k = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
        return self.aa((k - 1.0) * min(rx, ry), soft)

    def cap(self, x0, y0, x1, y1, r0, r1=None, soft=1.0):
        r1 = r0 if r1 is None else r1
        vx, vy = x1 - x0, y1 - y0
        t = np.clip(((self.lx - x0) * vx + (self.ly - y0) * vy) / (vx * vx + vy * vy + 1e-12), 0, 1)
        d = np.sqrt((self.lx - x0 - t * vx) ** 2 + (self.ly - y0 - t * vy) ** 2) - (r0 + (r1 - r0) * t)
        return self.aa(d, soft)

    def poly(self, pts, soft=1.0):
        P = np.asarray(pts, float); n = len(P)
        area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        sg = 1.0 if area > 0 else -1.0
        d = np.full((S, S), -1e9)
        for i in range(n):
            x0, y0 = P[i]; x1, y1 = P[(i + 1) % n]
            ex, ey = x1 - x0, y1 - y0
            d = np.maximum(d, sg * ((self.lx - x0) * ey - (self.ly - y0) * ex) / (np.hypot(ex, ey) + 1e-12))
        return self.aa(d, soft)

    def box(self, x, y, w, h, a=0.0, r=0.0, soft=1.0):
        dx = self.lx - x; dy = self.ly - y
        if a != 0.0:
            c, n = np.cos(a), np.sin(a); dx, dy = c * dx + n * dy, -n * dx + c * dy
        qx = np.abs(dx) - (w / 2 - r); qy = np.abs(dy) - (h / 2 - r)
        d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
        return self.aa(d, soft)

    def paint(self, al, col, tex=None, op=1.0):
        c = np.asarray(col, float)
        if tex is not None:
            c = c * (1.0 + (tex if np.ndim(tex) == 3 else tex[..., None]))
        self.im = self.im + (al * op)[..., None] * (c - self.im)

    def ramp(self, y0, y1):
        """0 at local y0 rising to 1 at local y1."""
        return np.clip((self.ly - y0) / (y1 - y0), 0.0, 1.0)

    def rampx(self, x0, x1):
        return np.clip((self.lx - x0) / (x1 - x0), 0.0, 1.0)

    def fur(self, fine=0.06, mid=0.08, g=7):
        """Multiplicative texture field for fur, feathers, skin: tufts, sheen and shadow at every scale."""
        rs = self.rs
        return (2.0 * (fine + mid)) * pink(rs, 4, 3)


# ------------------------------------------------------------------ backgrounds
def below(yh, slope=0.0, soft=1.0):
    """Soft mask of the image region below the line y = yh + slope * (x - 0.5)."""
    return np.clip((YY - (yh + slope * (XX - 0.5))) * S / soft + 0.5, 0.0, 1.0)


def vgrad(top, bot, y0=0.0, y1=1.0):
    t = np.clip((YY - y0) / (y1 - y0 + 1e-9), 0, 1)[..., None]
    return np.asarray(top, float) * (1 - t) + np.asarray(bot, float) * t


def sky(cv, kind=None, clouds=None):
    """Sky over the whole canvas: darker and bluer overhead, paler toward the horizon."""
    rs = cv.rs
    if kind is None:
        kind = pick(rs, ['blue', 'pale', 'over', 'white', 'deep', 'dusk', 'storm'], [34, 20, 20, 10, 8, 3, 5])
    if kind == 'blue':
        top = C(U(rs, .2, .45), U(rs, .45, .66), U(rs, .8, .97)); bot = lerp(top, C(.85, .9, .97), U(rs, .3, .8))
    elif kind == 'pale':
        top = C(U(rs, .55, .75), U(rs, .7, .85), U(rs, .88, 1.0)); bot = lerp(top, C(.92, .95, .98), U(rs, .3, .8))
    elif kind == 'over':
        g = U(rs, .62, .88); top = C(g, g + U(rs, 0, .02), g + U(rs, 0, .06)); bot = lerp(top, C(.93, .93, .93), U(rs, .2, .7))
    elif kind == 'white':
        g = U(rs, .88, 1.0); top = C(g, g, g); bot = top
    elif kind == 'deep':
        top = C(U(rs, .05, .22), U(rs, .2, .42), U(rs, .55, .85)); bot = lerp(top, C(.6, .75, .95), U(rs, .3, .7))
    elif kind == 'dusk':
        top = C(U(rs, .3, .5), U(rs, .35, .5), U(rs, .55, .7)); bot = C(U(rs, .85, 1), U(rs, .55, .75), U(rs, .3, .5))
    else:  # storm
        g = U(rs, .35, .58); top = C(g, g + .02, g + U(rs, .02, .08)); bot = lerp(top, C(.7, .72, .75), U(rs, .2, .6))
    cv.im = vgrad(top, bot, 0.0, U(rs, 0.7, 1.2))
    if clouds is None:
        clouds = chance(rs, {'blue': .4, 'pale': .35, 'over': .3, 'white': 0, 'deep': .35, 'dusk': .4, 'storm': .5}[kind])
    if clouds:
        f = fbm(rs, int(pick(rs, [2, 3, 4])), 3)
        m = np.clip((f - U(rs, -.1, .25)) * U(rs, 2, 6), 0, 1)
        cc = C(1, 1, 1) * U(rs, .82, 1.0) if kind != 'storm' else C(1, 1, 1) * U(rs, .55, .8)
        cv.paint(m, cc, tex=0.06 * noise(rs, 8), op=U(rs, .5, 1.0))
    return kind


GROUNDS = {
    # name: (colour low, colour high, texture contrast low, texture contrast high)
    'grass': (C(.3, .46, .13), C(.6, .76, .4), .12, .3),
    'dgrass': (C(.14, .28, .06), C(.36, .5, .2), .15, .35),
    'dry': (C(.62, .54, .3), C(.9, .8, .54), .1, .25),
    'dirt': (C(.52, .42, .28), C(.84, .7, .52), .07, .2),
    'soil': (C(.26, .18, .1), C(.48, .36, .24), .12, .3),
    'road': (C(.3, .3, .31), C(.68, .68, .7), .03, .1),
    'concrete': (C(.62, .62, .6), C(.9, .89, .86), .03, .08),
    'snow': (C(.82, .85, .9), C(.98, .98, 1.0), .02, .07),
    'sand': (C(.72, .64, .46), C(.92, .84, .66), .04, .12),
}


def ground(cv, kind, yh, slope=0.0, shade=None):
    """Ground plane below the horizon line, slightly hazier far away and darker close to the camera."""
    rs = cv.rs
    lo, hi, a0, a1 = GROUNDS[kind]
    grey = kind in ('road', 'concrete', 'snow')
    tint = .03 if grey else .18
    base = lerp(lo, hi, np.clip(U(rs, 0, 1) + rs.uniform(-tint, tint, 3), 0, 1))
    amp = U(rs, a0, a1)
    if kind in ('grass', 'dgrass', 'dry'):
        tex = nat(rs, amp, 3, .3, gx_mul=pick(rs, [1.0, 2.0, 3.0]))
    else:
        tex = nat(rs, amp, 3, .05 if grey else .2)
    sh = U(rs, -.12, .22) if shade is None else shade
    tex = tex + (sh * (0.5 - np.clip((YY - yh) / (1.0 - yh + 1e-6), 0, 1)))[..., None]
    cv.paint(below(yh, slope), base, tex=tex)
    if chance(rs, .25):                                 # a shadow or a worn / wet patch across the ground
        m = np.clip((fbm(rs, 3, 2) - U(rs, 0, .3)) * 4, 0, 1) * below(yh, slope)
        cv.paint(m, base * U(rs, .45, .8))
    return base


def water(cv, yh, kind=None, slope=0.0):
    rs = cv.rs
    if kind is None:
        kind = pick(rs, ['blue', 'grey', 'teal', 'dark', 'deepblue', 'steel', 'murky'], [26, 18, 12, 10, 12, 14, 8])
    if kind == 'steel':
        g = U(rs, .3, .7); base = C(g, g + U(rs, 0, .03), g + U(rs, 0, .06))
    elif kind == 'murky':
        base = C(U(rs, .25, .45), U(rs, .32, .48), U(rs, .25, .4))
    else:
        base = {'blue': C(U(rs, .15, .4), U(rs, .38, .6), U(rs, .58, .82)),
                'grey': C(U(rs, .35, .58), U(rs, .43, .62), U(rs, .48, .68)),
                'teal': C(U(rs, .1, .3), U(rs, .38, .58), U(rs, .38, .58)),
                'dark': C(U(rs, .04, .18), U(rs, .12, .3), U(rs, .22, .42)),
                'deepblue': C(U(rs, .05, .18), U(rs, .15, .35), U(rs, .45, .75))}[kind]
    tex = nat(rs, U(rs, .06, .22), 4, .1, gx_mul=pick(rs, [.2, .3, .5]))
    tex = tex + (U(rs, .0, .35) * (0.5 - np.clip((YY - yh) / (1.0 - yh + 1e-6), 0, 1)))[..., None]   # paler near the horizon
    cv.paint(below(yh, slope), base, tex=tex)
    if chance(rs, .3):   # glints or whitecaps
        g = np.clip((noise(rs, 32, 6) - U(rs, .55, .8)) * 6, 0, 1) * below(yh, slope)
        cv.paint(g, C(.95, .97, 1.0), op=U(rs, .3, .8))
    return base


def band_mask(ytop_fn, ybot):
    return np.clip((YY - ytop_fn) * S + 0.5, 0, 1) * np.clip((ybot - YY) * S + 0.5, 0, 1)


def edge1d(rs, g, amp):
    """Irregular skyline: per-column vertical offset."""
    return amp * (_interp(g) @ rs.uniform(-1, 1, g))[None, :]


def treeline(cv, ytop, ybot, col=None):
    rs = cv.rs
    if col is None:
        col = pick(rs, [C(.1, .22, .08), C(.18, .32, .12), C(.25, .4, .15), C(.12, .18, .1), C(.3, .33, .15), C(.35, .25, .12)],
                   [25, 25, 18, 14, 10, 8])
    col = jit(rs, col, .03, .2)
    top = ytop + edge1d(rs, int(pick(rs, [5, 8, 12])), U(rs, .02, .07)) + edge1d(rs, 20, .015)
    tex = nat(rs, U(rs, .2, .45), 4, .25) + (0.2 * (0.5 - np.clip((YY - ytop) / (ybot - ytop + 1e-6), 0, 1)))[..., None]
    cv.paint(band_mask(top, ybot), col, tex=tex)


def hills(cv, ytop, ybot):
    rs = cv.rs
    col = jit(rs, pick(rs, [C(.45, .55, .65), C(.35, .45, .4), C(.3, .42, .25), C(.55, .5, .42), C(.6, .68, .78)]), .03, .15)
    top = ytop + edge1d(rs, int(pick(rs, [3, 4, 6])), U(rs, .03, .1))
    cv.paint(band_mask(top, ybot), col, tex=0.06 * noise(rs, 6))


def buildings(cv, ytop, ybot):
    rs = cv.rs
    x = U(rs, -.2, 0.0)
    while x < 1.0:
        w = U(rs, .15, .5); h = (ybot - ytop) * U(rs, .5, 1.25)
        col = jit(rs, pick(rs, [C(.78, .72, .6), C(.6, .6, .6), C(.9, .9, .88), C(.55, .3, .22), C(.85, .8, .7), C(.4, .42, .45), C(.7, .75, .8), C(.5, .4, .3)]), .03, .15)
        m = np.clip((XX - x) * S + .5, 0, 1) * np.clip((x + w - XX) * S + .5, 0, 1) * np.clip((YY - (ybot - h)) * S + .5, 0, 1) * np.clip((ybot - YY) * S + .5, 0, 1)
        cv.paint(m, col, tex=0.04 * noise(rs, 6))
        if chance(rs, .6):   # rows of windows or a door: darker patches
            g = int(pick(rs, [3, 4, 5]))
            wm = (np.sin(XX * np.pi * 2 * g / max(w, .2) + U(rs, 0, 6)) > U(rs, 0, .6)) * (np.sin(YY * np.pi * 2 * g / max(h, .2) + U(rs, 0, 6)) > U(rs, 0, .6))
            cv.paint(m * wm, col * U(rs, .3, .7), op=U(rs, .5, 1))
        x += w + (U(rs, 0, .1) if chance(rs, .3) else 0)


def fence(cv, y, col=None, rails=None):
    rs = cv.rs
    col = jit(rs, pick(rs, [C(.92, .92, .9), C(.45, .32, .2), C(.6, .58, .55), C(.3, .25, .2)]) if col is None else col, .02, .1)
    rails = int(pick(rs, [2, 3, 4])) if rails is None else rails
    gap = U(rs, .035, .07); th = U(rs, .008, .016)
    m = np.zeros((S, S))
    for i in range(rails):
        m = np.maximum(m, np.clip((th - np.abs(YY - (y + i * gap))) * S + .5, 0, 1))
    sp = U(rs, .18, .4); x0 = U(rs, 0, sp)
    post = np.clip((th * 1.3 - np.abs(((XX - x0) % sp) - sp / 2)) * S + .5, 0, 1) * np.clip((YY - (y - .02)) * S + .5, 0, 1) * np.clip((y + rails * gap + .01 - YY) * S + .5, 0, 1)
    cv.paint(np.maximum(m, post), col)


def bokeh(cv, cols, g=None, sharp=None):
    """Out-of-focus or leafy backdrop: a few colours blended by smooth random fields."""
    rs = cv.rs
    g = int(pick(rs, [2, 3, 4, 5, 7])) if g is None else g
    cv.im = np.ones((S, S, 3)) * jit(rs, cols[0], .03, .15)
    for c in cols[1:]:
        f = fbm(rs, g, 2)
        k = U(rs, 1.5, 5) if sharp is None else sharp
        cv.paint(np.clip((f - U(rs, -.15, .3)) * k, 0, 1), jit(rs, c, .03, .15))


FOLIAGE = [C(.12, .25, .08), C(.25, .42, .15), C(.4, .58, .2), C(.55, .68, .3), C(.2, .3, .12), C(.35, .3, .15),
           C(.5, .45, .25), C(.7, .78, .5), C(.08, .12, .06), C(.3, .45, .3), C(.6, .62, .35)]
EARTH = [C(.35, .25, .15), C(.5, .38, .25), C(.25, .18, .12), C(.6, .5, .38), C(.42, .36, .3), C(.15, .1, .07), C(.55, .45, .3)]
GREYS = [C(.5, .5, .5), C(.65, .65, .63), C(.35, .36, .38), C(.75, .74, .7), C(.45, .43, .4)]


def foliage(cv, n=None, g=None, dark=None, leafy=None):
    rs = cv.rs
    n = int(pick(rs, [2, 3, 4])) if n is None else n
    idx = rs.choice(len(FOLIAGE), n, replace=False)
    cols = [FOLIAGE[i] for i in idx]
    if chance(rs, .2):
        cols.append(pick(rs, EARTH))
    if chance(rs, .15):
        cols.append(C(.8, .88, .95))   # sky showing through
    bokeh(cv, cols, g)
    if dark is None:
        dark = U(rs, .75, 1.25)
    cv.im = np.clip(cv.im * dark, 0, 1)
    if chance(rs, .6) if leafy is None else leafy:      # in focus: leaves, twigs, light and shade at every scale
        cv.im = np.clip(cv.im * (1 + nat(rs, U(rs, .2, .5), 3, .25)), 0, 1)


def litter(cv, cols=None):
    """Mottled ground seen from close above: soil, dead leaves, pebbles, moss."""
    rs = cv.rs
    if cols is None:
        kind = pick(rs, ['soil', 'leaf', 'rock', 'moss', 'sand'], [30, 25, 15, 18, 12])
        cols = {'soil': [EARTH[0], EARTH[2], EARTH[1], EARTH[5]], 'leaf': [EARTH[1], C(.6, .45, .2), EARTH[0], C(.7, .6, .35), FOLIAGE[4]],
                'rock': [GREYS[0], GREYS[2], GREYS[1], GREYS[4]], 'moss': [FOLIAGE[1], FOLIAGE[4], EARTH[0], FOLIAGE[2]],
                'sand': [C(.75, .66, .5), C(.62, .55, .42), C(.85, .78, .62)]}[kind]
    cv.im = np.ones((S, S, 3)) * jit(rs, cols[0], .03, .15)
    for c in cols[1:]:
        f = fbm(rs, int(pick(rs, [5, 8, 11])), 2)
        cv.paint(np.clip((f - U(rs, -.05, .3)) * U(rs, 3, 8), 0, 1), jit(rs, c, .03, .15))
    cv.im = np.clip(cv.im * (1 + nat(rs, U(rs, .12, .3), 3, .2)), 0, 1)


def strokes(cv, n, cols, r0=.006, r1=.02, vertical=False, lmin=.15, lmax=.9):
    """Twigs, stalks, grass blades, branches: thin straight strokes scattered over the picture."""
    rs = cv.rs
    cv.frame(.5, .5, 1.0)
    for _ in range(n):
        x, y = U(rs, -.6, .6), U(rs, -.6, .6)
        a = U(rs, -.35, .35) + np.pi / 2 if vertical else U(rs, 0, np.pi)
        l = U(rs, lmin, lmax)
        cv.paint(cv.cap(x, y, x + l * np.cos(a), y + l * np.sin(a), U(rs, r0, r1)), jit(rs, pick(rs, cols), .03, .2), op=U(rs, .5, 1))


def dapple(cv, amp=None):
    """Patchy light and shade falling over the whole scene (sun through trees, uneven lighting)."""
    rs = cv.rs
    amp = U(rs, .1, .35) if amp is None else amp
    cv.im = np.clip(cv.im * (1 + amp * pink(rs, 2, 3))[..., None], 0, 1)


def plain(cv, col=None, grad=None):
    rs = cv.rs
    if col is None:
        col = pick(rs, [C(.97, .97, .97), C(.04, .04, .04), C(.5, .5, .5), C(.78, .78, .78), C(.2, .2, .22), None], [44, 14, 8, 12, 5, 17])
        if col is None:
            col = rs.uniform(.25, .95, 3) * .5 + U(rs, .1, .45)
    col = jit(rs, col, .02, .05)
    g = U(rs, -.25, .25) if grad is None else grad
    cv.im = vgrad(np.clip(col * (1 + g), 0, 1), np.clip(col * (1 - g), 0, 1))
    cv.im = cv.im * (1 + 0.05 * noise(rs, 2))[..., None]
    return col


WALLS = [C(.92, .9, .85), C(.85, .8, .7), C(.75, .75, .75), C(.95, .95, .95), C(.7, .6, .45), C(.55, .65, .75), C(.6, .7, .6),
         C(.8, .7, .65), C(.4, .3, .22), C(.3, .3, .32), C(.85, .82, .6), C(.6, .35, .3)]
FLOORS = [C(.55, .4, .25), C(.7, .55, .38), C(.4, .28, .18), C(.6, .6, .58), C(.75, .7, .62), C(.3, .3, .3), C(.5, .2, .15),
          C(.35, .4, .5), C(.8, .78, .75), C(.45, .5, .35)]
FABRIC = [C(.6, .2, .2), C(.25, .35, .6), C(.3, .5, .35), C(.8, .75, .65), C(.5, .4, .3), C(.85, .85, .88), C(.2, .2, .25),
          C(.75, .6, .3), C(.6, .45, .6), C(.9, .7, .7), C(.45, .6, .7), C(.35, .25, .2), C(.7, .7, .72), C(.85, .5, .2)]


def indoor(cv):
    """Room: wall and floor, plus big soft furniture or cloth shapes and a bright or dark patch."""
    rs = cv.rs
    kind = pick(rs, ['room', 'cloth', 'floor'], [45, 35, 20])
    if kind == 'room':
        cv.im = np.ones((S, S, 3)) * jit(rs, pick(rs, WALLS), .03, .15)
        cv.im = cv.im * (1 + 0.1 * noise(rs, 2))[..., None]
        yh = U(rs, .35, .8)
        cv.paint(below(yh, U(rs, -.25, .25)), jit(rs, pick(rs, FLOORS), .03, .15), tex=0.04 * noise(rs, 12, 3) + 0.06 * noise(rs, 3))
    elif kind == 'cloth':
        cv.im = np.ones((S, S, 3)) * jit(rs, pick(rs, FABRIC), .04, .2)
        cv.im = cv.im * (1 + 0.14 * noise(rs, int(pick(rs, [3, 4, 6]))) + 0.05 * noise(rs, 12))[..., None]
        if chance(rs, .3):                               # stripes or checks
            f = U(rs, 20, 60); a_ = U(rs, 0, np.pi)
            p = np.sin((XX * np.cos(a_) + YY * np.sin(a_)) * f)
            if chance(rs, .4):
                p = p * np.sin((-XX * np.sin(a_) + YY * np.cos(a_)) * f)
            cv.im = np.clip(cv.im * (1 + U(rs, .08, .25) * p)[..., None], 0, 1)
    else:
        cv.im = np.ones((S, S, 3)) * jit(rs, pick(rs, FLOORS), .03, .15)
        cv.im = cv.im * (1 + 0.05 * noise(rs, 12, 3) + 0.08 * noise(rs, 3))[..., None]
    cv.frame(0.5, 0.5, 1.0)
    for _ in range(int(pick(rs, [0, 1, 2, 3], [20, 35, 30, 15]))):
        col = jit(rs, pick(rs, FABRIC + WALLS + [C(.05, .05, .05), C(.98, .98, .95)]), .04, .2)
        x, y = U(rs, -.5, .5), U(rs, -.5, .5)
        if chance(rs, .5):
            m = cv.box(x, y, U(rs, .3, 1.0), U(rs, .2, .8), a=U(rs, -.4, .4), r=U(rs, 0, .08), soft=U(rs, 1, 3))
        else:
            m = cv.ell(x, y, U(rs, .2, .6), U(rs, .15, .5), a=U(rs, -1, 1), soft=U(rs, 1, 4))
        cv.paint(m, col, tex=0.08 * noise(rs, 4))


def photo(im, rs, blur=None):
    """Camera and web-photo effects: blur, exposure, contrast, saturation, colour cast, noise."""
    b = (U(rs, 0, .7) if chance(rs, .5) else 0.0) if blur is None else blur
    if b > 0:
        p = np.pad(im, ((1, 1), (1, 1), (0, 0)), mode='edge')
        k = (p[:-2] + 2 * p[1:-1] + p[2:]) / 4
        k = (k[:, :-2] + 2 * k[:, 1:-1] + k[:, 2:]) / 4
        im = im + b * (k - im)
    im = np.clip(im, 0, 1)
    im = im * U(rs, .88, 1.22)
    im = (im - .5) * U(rs, .95, 1.4) + .5
    g = im.mean(2, keepdims=True)
    sat = 0.0 if chance(rs, .03) else U(rs, .65, 1.15)
    im = g + (im - g) * sat
    im = im * (1 + rs.uniform(-.05, .05, 3))
    if chance(rs, .15):   # haze or flare: lifted blacks
        im = im + U(rs, 0, .1) * (1 - im)
    im = np.clip(im, 0, 1) ** U(rs, .75, 1.15)
    a = U(rs, .0, .012)                                  # faint blotchy colour error of compressed photos
    im = im + a * np.stack([noise(rs, 16) for _ in range(3)], axis=2)
    im = im + rs.normal(0, U(rs, .002, .008), im.shape)
    return np.clip(im, 0, 1)

# ---------------------------------------------------------------------------------------------
# PART 2. Shape models of the four vehicle classes, typed from general knowledge:
# airplane (fuselage, swept wings, fin), ship (hull, superstructure, sails, wake),
# automobile (low body, cabin with dark glass, two dark wheels), truck (tall cab, big box, many wheels).
# ---------------------------------------------------------------------------------------------

GLASS = C(.13, .16, .2)
TYRE = C(.07, .07, .07)
CARCOL = [C(.75, .08, .08), C(.72, .74, .76), C(.93, .93, .93), C(.07, .07, .08), C(.12, .25, .65), C(.92, .76, .1),
          C(.42, .44, .46), C(.1, .35, .18), C(.75, .65, .45), C(.45, .07, .1), C(.1, .13, .3), C(.9, .42, .08),
          C(.45, .6, .8), C(.35, .22, .15), C(.5, .55, .45)]
CARP = [20, 14, 11, 10, 11, 6, 8, 3, 5, 3, 3, 2, 2, 1, 1]
TRUCKCOL = [C(.93, .93, .93), C(.78, .1, .08), C(.92, .76, .1), C(.12, .28, .65), C(.1, .38, .2), C(.68, .7, .72),
            C(.1, .1, .1), C(.9, .45, .08), C(.4, .42, .45), C(.5, .08, .1), C(.3, .5, .75), C(.1, .45, .45)]
TRUCKP = [22, 22, 9, 12, 5, 8, 4, 5, 4, 3, 4, 2]


def glass_col(rs):
    return jit(rs, GLASS, .02, .3) if chance(rs, .75) else jit(rs, C(.5, .62, .72), .04, .2)


def wheel(cv, x, y, r, hub=None, rx=None):
    rs = cv.rs
    rx = r if rx is None else rx
    cv.paint(cv.ell(x, y, rx, r), TYRE)
    hub = jit(rs, C(.7, .7, .72), .02, .15) if hub is None else hub
    cv.paint(cv.ell(x, y, rx * .55, r * .55), hub)


# ------------------------------------------------------------------ airplane
def plane_cols(rs):
    k = pick(rs, ['white', 'lgrey', 'dgrey', 'sil', 'col', 'camo'], [50, 14, 12, 9, 10, 5])
    if k == 'white':
        b = jit(rs, C(.93, .93, .94), .01, .05)
    elif k == 'lgrey':
        b = jit(rs, C(.7, .72, .75), .02, .1)
    elif k == 'dgrey':
        b = jit(rs, C(.32, .34, .37), .03, .2)
    elif k == 'sil':
        b = jit(rs, C(.1, .1, .12), .02, .3)
    elif k == 'col':
        b = jit(rs, pick(rs, [C(.9, .78, .1), C(.8, .12, .1), C(.15, .3, .7), C(.9, .5, .1), C(.2, .5, .25)]), .03, .1)
    else:
        b = jit(rs, pick(rs, [C(.35, .4, .25), C(.45, .42, .3), C(.4, .45, .5)]), .03, .15)
    acc = jit(rs, pick(rs, [C(.1, .2, .6), C(.8, .1, .1), C(.1, .1, .15), C(.9, .5, .1), C(.1, .45, .3), C(.5, .5, .55), C(.95, .8, .1)]), .03, .1)
    return k, b, acc


def airplane_side(cv, gear=False):
    rs = cv.rs
    k, b, acc = plane_cols(rs)
    fr = U(rs, .06, .1)
    under = b * U(rs, .55, .85)
    sh = -0.3 * cv.ramp(-fr, fr)                       # lit from above
    tipy = fr + U(rs, .05, .2) if chance(rs, .7) else -fr - U(rs, .04, .16)
    tx = -.28 + U(rs, -.06, .1)
    if chance(rs, .5):                                  # far wing, seen behind the body
        cv.paint(cv.poly([(.12, -fr * .3), (-.06, -fr * .3), (tx + .08, -tipy * .7), (tx + .16, -tipy * .7)]), under)
    fin_c = acc if (k in ('white', 'lgrey') and chance(rs, .6)) else b
    fx = U(rs, 0, .05); fh = U(rs, .16, .27)
    cv.paint(cv.poly([(-.52, -fr * .4), (-.33, -fr * .8), (-.46 - fx, -fr - fh), (-.56 - fx, -fr - fh)]), fin_c)
    cv.paint(cv.cap(-.3, 0, -.52, -fr * .5, fr, fr * .35), b, tex=sh)
    cv.paint(cv.cap(-.34, 0, .4, 0, fr), b, tex=sh)
    cv.paint(cv.ell(.4, 0, .13, fr), b, tex=sh)
    if k in ('white', 'lgrey') and chance(rs, .4):      # painted belly
        cv.paint(cv.cap(-.34, fr * .55, .4, fr * .55, fr * .5), acc if chance(rs, .5) else under)
    if chance(rs, .4):                                  # cheat line / window row
        cv.paint(cv.cap(-.36, -fr * .1, .4, -fr * .1, .012), acc if chance(rs, .6) else C(.15, .15, .2))
    cv.paint(cv.poly([(-.5, -fr * .35), (-.37, -fr * .25), (-.58, -fr * .1 + U(rs, 0, .05))]), under)
    cv.paint(cv.poly([(.1, fr * .3), (-.12, fr * .3), (tx, tipy), (tx + .1, tipy)]), under if tipy > 0 else lerp(b, under, .4))
    if chance(rs, .7):
        cv.paint(cv.ell(.02 + U(rs, -.05, .05), fr + .045, .075, .032), under * U(rs, .5, 1.0))
    if chance(rs, .5):
        cv.paint(cv.ell(.4, -fr * .35, .035, fr * .3), C(.12, .14, .2))
    if gear:
        for x in (.3, -.06, -.1):
            cv.paint(cv.cap(x, fr * .8, x, fr + .07, .012), C(.2, .2, .2))
            cv.paint(cv.ell(x, fr + .08, .025, .025), TYRE)
    return fr + (.105 if gear else 0)


def airplane_plan(cv):
    rs = cv.rs
    k, b, acc = plane_cols(rs)
    if chance(rs, .35):
        b = b * U(rs, .25, .7)                          # underside in shadow against a bright sky
    span = U(rs, .38, .56); sw = U(rs, -.12, .12); tc = U(rs, .05, .1)
    for sg in (-1, 1):
        cv.paint(cv.poly([(.13, sg * .02), (-.08, sg * .02), (-.3 + sw, sg * span), (-.3 + sw + tc, sg * span)]), b * U(rs, .85, 1.0))
        cv.paint(cv.poly([(-.36, sg * .02), (-.48, sg * .02), (-.57, sg * U(rs, .15, .2)), (-.5, sg * .18)]), b * .95)
        if chance(rs, .5):
            cv.paint(cv.ell(-.0 + sw * .3, sg * U(rs, .17, .25), .065, .025), b * U(rs, .5, .9))
    cv.paint(cv.cap(-.5, 0, .42, 0, U(rs, .04, .06)), b)
    cv.paint(cv.ell(.42, 0, .1, .05), b)
    if chance(rs, .3):
        cv.paint(cv.cap(-.45, 0, .4, 0, .012), acc)


def airplane_fighter(cv):
    rs = cv.rs
    b = jit(rs, pick(rs, [C(.45, .47, .5), C(.3, .32, .35), C(.6, .62, .65), C(.1, .1, .12), C(.35, .4, .28), C(.4, .45, .55), C(.8, .8, .82)],
                     [22, 18, 16, 16, 10, 10, 8]), .03, .15)
    if chance(rs, .3):                                  # flying wing
        for sg in (-1, 1):
            cv.paint(cv.poly([(.25, 0), (-.25, sg * .58), (-.36, sg * .52), (-.12, 0)]), b * U(rs, .85, 1.0))
        cv.paint(cv.ell(.05, 0, .16, .06), b * .9)
    else:
        cv.paint(cv.poly([(.18, 0), (-.38, -.36), (-.46, -.36), (-.46, .36), (-.38, .36)]), b)
        cv.paint(cv.cap(-.48, 0, .3, 0, .055), b * .92)
        cv.paint(cv.poly([(.3, -.05), (.3, .05), (.58, 0)]), b * .92)
        for sg in (-1, 1):
            cv.paint(cv.poly([(-.3, sg * .06), (-.5, sg * .06), (-.56, sg * .2), (-.46, sg * .2)]), b * .75)
        cv.paint(cv.ell(.24, 0, .08, .03), C(.12, .15, .2) if chance(rs, .7) else C(.6, .7, .8))


def airplane_prop(cv, gear=True):
    rs = cv.rs
    b = jit(rs, pick(rs, [C(.92, .78, .12), C(.8, .12, .1), C(.93, .93, .93), C(.15, .3, .7), C(.7, .72, .75), C(.2, .45, .25), C(.9, .5, .1), C(.35, .38, .3)],
                     [20, 18, 22, 10, 10, 6, 6, 8]), .03, .1)
    acc = jit(rs, pick(rs, [C(.8, .1, .1), C(.1, .2, .6), C(.1, .1, .1), C(.95, .95, .95), C(.95, .8, .1)]), .03, .1)
    sh = -0.25 * cv.ramp(-.09, .09)
    cv.paint(cv.poly([(-.5, -.02), (-.36, -.05), (-.42, -.28), (-.53, -.28)]), acc if chance(rs, .4) else b)
    cv.paint(cv.cap(-.1, 0, -.48, -.03, .09, .035), b, tex=sh)
    cv.paint(cv.cap(-.12, 0, .3, 0, .09), b, tex=sh)
    cv.paint(cv.ell(.33, 0, .1, .085), acc if chance(rs, .4) else b, tex=sh)
    cv.paint(cv.cap(.445, -.2, .445, .2, .012), C(.25, .25, .25), op=.8)
    cv.paint(cv.ell(.08, -.075, .08, .04), C(.15, .18, .25) if chance(rs, .7) else C(.6, .7, .8))
    cv.paint(cv.poly([(-.5, -.03), (-.36, -.03), (-.58, .03)]), b * .8)
    if chance(rs, .3):                                  # biplane
        cv.paint(cv.cap(-.06, -.22, .26, -.22, .022), b)
        cv.paint(cv.cap(-.04, .07, .24, .07, .022), b * .85)
        for x in (.0, .2):
            cv.paint(cv.cap(x, -.2, x, .06, .01), C(.3, .3, .3))
    elif chance(rs, .5):                                # high wing seen edge-on
        cv.paint(cv.cap(-.04, -.115, .24, -.115, .026), b * .9)
    else:                                               # low wing toward the camera
        wy = U(rs, .1, .22)
        cv.paint(cv.poly([(.16, .04), (-.06, .04), (-.2, wy), (-.02, wy)]), b * .8)
    if chance(rs, .4):
        cv.paint(cv.cap(-.3, 0, .3, 0, .014), acc)
    if gear:
        cv.paint(cv.cap(.14, .07, .2, .21, .012), C(.25, .25, .25))
        cv.paint(cv.ell(.2, .23, .045, .045), TYRE)
        cv.paint(cv.ell(-.42, .06, .02, .02), TYRE)
    return .275


# ------------------------------------------------------------------ ship
HULLCOL = [C(.08, .08, .1), C(.1, .15, .35), C(.5, .1, .08), C(.92, .92, .92), C(.45, .48, .52), C(.1, .3, .2), C(.15, .35, .7),
           C(.75, .15, .1), C(.3, .2, .15)]


def ship_side(cv, kind=None):
    """Hull sits on the waterline y = 0 and extends up to y = -hh; unit = hull length."""
    rs = cv.rs
    if kind is None:
        kind = pick(rs, ['cargo', 'cruise', 'fish', 'war', 'sail', 'speed'], [26, 16, 20, 9, 15, 14])
    white = jit(rs, C(.93, .93, .92), .01, .06)
    if kind == 'sail':
        hc = jit(rs, pick(rs, [C(.93, .93, .93), C(.1, .15, .35), C(.6, .1, .1), C(.1, .1, .1), C(.4, .25, .15)], [45, 18, 12, 12, 13]), .03, .1)
        sc_ = jit(rs, pick(rs, [C(.95, .95, .93), C(.9, .85, .75), C(.8, .2, .15), C(.2, .3, .7), C(.75, .75, .78)], [55, 20, 8, 7, 10]), .02, .06)
        mh = U(rs, .8, 1.15); mx = U(rs, -.05, .08)
        cv.paint(cv.cap(mx, 0, mx, -mh, .012), C(.3, .3, .3))
        cv.paint(cv.poly([(mx - .015, -mh * .97), (mx - .015, -.1), (mx - U(rs, .3, .45), -.1)]), sc_, tex=-.15 * cv.rampx(-.4, 0))
        if chance(rs, .75):
            cv.paint(cv.poly([(mx + .02, -mh * U(rs, .7, .95)), (mx + .02, -.09), (mx + U(rs, .25, .4), -.09)]), sc_ * U(rs, .85, 1.0))
        cv.paint(cv.poly([(-.34, -.08), (.4, -.09), (.3, 0), (-.3, 0)]), hc)
        return mh
    if kind == 'speed':
        hc = white if chance(rs, .7) else jit(rs, pick(rs, HULLCOL), .03, .1)
        cv.paint(cv.poly([(-.5, -.09), (.56, -.15), (.36, 0), (-.48, 0)]), hc, tex=-.2 * cv.ramp(-.12, 0))
        if chance(rs, .6):
            cv.paint(cv.cap(-.45, -.045, .4, -.06, .012), jit(rs, pick(rs, [C(.1, .2, .6), C(.8, .1, .1), C(.1, .1, .1), C(.1, .5, .5)]), .03, .1))
        if chance(rs, .6):                              # cabin
            cv.paint(cv.poly([(-.3, -.1), (.12, -.11), (.02, -.21), (-.26, -.21)]), white)
            cv.paint(cv.poly([(-.24, -.12), (.07, -.125), (.0, -.185), (-.22, -.185)]), GLASS)
        else:                                           # windscreen
            cv.paint(cv.poly([(-.05, -.1), (.12, -.11), (.02, -.17)]), GLASS)
        f = np.clip(noise(rs, 9) + .4, 0, 1)            # spray behind and along the hull
        cv.paint(cv.ell(-.5, -.02, U(rs, .15, .35), U(rs, .04, .1), soft=3) * f, C(.97, .98, 1))
        cv.paint(cv.cap(-.6, .01, .4, .01, .02, soft=2) * f, C(.97, .98, 1), op=.8)
        return .22
    hh = {'cargo': U(rs, .09, .17), 'cruise': U(rs, .08, .14), 'fish': U(rs, .12, .22), 'war': U(rs, .06, .1)}[kind]
    if kind == 'war':
        hc = jit(rs, C(.48, .52, .56), .02, .15)
    elif kind == 'cruise':
        hc = white if chance(rs, .65) else jit(rs, pick(rs, HULLCOL[:2]), .03, .1)
    else:
        hc = jit(rs, pick(rs, HULLCOL, [22, 14, 12, 12, 8, 6, 9, 9, 8]), .03, .12)
    top = -hh
    if kind == 'cargo':
        sx = pick(rs, [-.34, -.3, 0.0, .3], [50, 25, 12, 13]) + U(rs, -.04, .04); sw = U(rs, .12, .22); sh_ = U(rs, .12, .26)
        if chance(rs, .5):                              # deck cargo: containers / hatches
            x = -.42
            while x < .4:
                w = U(rs, .06, .16)
                if abs(x + w / 2 - sx) > sw / 2 + w / 2:
                    h = U(rs, .03, .09)
                    cv.paint(cv.box(x + w / 2, top - h / 2, w, h), jit(rs, pick(rs, [C(.55, .25, .15), C(.2, .3, .55), C(.3, .45, .3), C(.6, .6, .6), C(.85, .85, .85), C(.75, .6, .2)]), .03, .15))
                x += w + .01
        else:                                           # masts and derricks
            for x in rs.uniform(-.2, .42, int(pick(rs, [1, 2, 3]))):
                h = U(rs, .12, .28)
                cv.paint(cv.cap(x, top, x, top - h, .01), jit(rs, C(.5, .45, .35), .03, .3))
                cv.paint(cv.cap(x, top - h, x + U(rs, -.15, .15), top - h * U(rs, .3, .8), .008), C(.3, .3, .3))
        cv.paint(cv.box(sx, top - sh_ / 2, sw, sh_), white)
        cv.paint(cv.cap(sx - sw * .4, top - sh_ * .75, sx + sw * .4, top - sh_ * .75, .01), GLASS)
        cv.paint(cv.box(sx + U(rs, -.03, .03), top - sh_ - .035, .05, .07), jit(rs, pick(rs, [C(.75, .15, .1), C(.1, .1, .1), C(.15, .3, .65), C(.9, .75, .1), C(.9, .9, .9)]), .03, .1))
    elif kind == 'cruise':
        d1 = U(rs, .1, .2)
        cv.paint(cv.box(-.04, top - d1 / 2, U(rs, .68, .8), d1), white)
        cv.paint(cv.box(-.06, top - d1 - .025, U(rs, .4, .55), .05), white * .97)
        for r_ in range(int(pick(rs, [1, 2, 3]))):
            cv.paint(cv.cap(-.36, top - .03 - r_ * .04, .3, top - .03 - r_ * .04, .009), GLASS, op=.8)
        cv.paint(cv.box(-.12 + U(rs, -.1, .1), top - d1 - .08, .07, .08), jit(rs, pick(rs, [C(.8, .15, .1), C(.1, .2, .6), C(.9, .7, .1), C(.1, .1, .1), C(.93, .93, .93)]), .03, .1))
    elif kind == 'fish':
        cx_ = U(rs, -.2, .15); cw = U(rs, .2, .34); ch = U(rs, .14, .26)
        cv.paint(cv.box(cx_, top - ch / 2, cw, ch), white if chance(rs, .75) else jit(rs, pick(rs, HULLCOL), .03, .1))
        cv.paint(cv.cap(cx_ - cw * .35, top - ch * .7, cx_ + cw * .35, top - ch * .7, .014), GLASS)
        mxp = cx_ + U(rs, -.1, .3); mh = U(rs, .3, .55)
        cv.paint(cv.cap(mxp, top, mxp, top - mh, .011), C(.3, .28, .25))
        if chance(rs, .6):
            cv.paint(cv.cap(mxp, top - mh * .8, mxp + U(rs, -.3, .3), top - .05, .008), C(.3, .28, .25))
    else:                                               # warship: stepped grey superstructure
        cv.paint(cv.box(-.02, top - .04, U(rs, .35, .5), .08), hc * 1.05)
        cv.paint(cv.box(-.02, top - .11, U(rs, .15, .25), .07), hc * 1.08)
        cv.paint(cv.cap(0, top - .12, 0, top - .12 - U(rs, .1, .22), .01), hc * .8)
        cv.paint(cv.box(.28, top - .02, .07, .04), hc * .95)
    bow = U(rs, .03, .08)
    cv.paint(cv.poly([(-.5, top), (.5 + bow, top * 1.3), (.44, 0), (-.47, 0)]), hc, tex=-.15 * cv.ramp(top, 0))
    if chance(rs, .4):
        cv.paint(cv.poly([(-.475, -.022), (.45, -.022), (.44, 0), (-.47, 0)]), jit(rs, C(.6, .14, .1), .03, .1))
    if kind in ('cargo', 'fish') and chance(rs, .3):
        cv.paint(cv.cap(-.45, top + .015, .46, top + .015, .008), white)
    return hh + .25


def ship_wake(cv, y=0.0):
    rs = cv.rs
    if chance(rs, .55):
        f = np.clip(noise(rs, 32, 10) + .5, 0, 1)
        cv.paint(cv.cap(-.55, y + .005, .52, y + .005, U(rs, .008, .02), soft=1.5) * f, C(.95, .97, 1), op=U(rs, .4, .9))


def boat_above(cv):
    rs = cv.rs
    hc = jit(rs, C(.94, .94, .94), .01, .05) if chance(rs, .65) else jit(rs, pick(rs, [C(.8, .12, .1), C(.15, .3, .7), C(.95, .8, .1), C(.1, .1, .12), C(.9, .5, .1)]), .03, .1)
    f = np.clip(fbm(rs, 6, 2) * 1.5 + .6, 0, 1)
    cv.paint(cv.poly([(-.4, -.1), (-.4, .1), (-1.6, U(rs, .25, .5)), (-1.6, -U(rs, .25, .5))], soft=3) * f, C(.95, .97, 1), op=U(rs, .5, .95))
    cv.paint(cv.poly([(.56, 0), (.22, -.14), (-.46, -.13), (-.46, .13), (.22, .14)]), hc)
    cv.paint(cv.box(-.14, 0, U(rs, .3, .45), .17, r=.03), jit(rs, pick(rs, [C(.15, .2, .35), C(.4, .4, .42), C(.7, .6, .45), C(.5, .3, .2), C(.85, .85, .85), C(.6, .1, .1)]), .03, .15))
    cv.paint(cv.poly([(.1, -.1), (.2, 0), (.1, .1), (.06, .1), (.06, -.1)]), GLASS)


# ------------------------------------------------------------------ automobile
def car_side(cv):
    """Side view; unit = car length; wheel bottoms at y = 0.16."""
    rs = cv.rs
    b = jit(rs, pick(rs, CARCOL, CARP), .03, .12)
    gy = .16; wr = U(rs, .082, .1); wx = U(rs, .28, .33)
    topb = -U(rs, .04, .08); botb = gy - wr * U(rs, .5, .8)
    style = pick(rs, ['sedan', 'hatch', 'sport', 'van'], [45, 28, 15, 12])
    roof = topb - {'sedan': U(rs, .11, .15), 'hatch': U(rs, .12, .16), 'sport': U(rs, .08, .11), 'van': U(rs, .17, .24)}[style]
    xr0 = {'sedan': -.34, 'hatch': -.47, 'sport': -.3, 'van': -.49}[style] + U(rs, -.02, .02)
    xr1 = {'sedan': -.2, 'hatch': -.4, 'sport': -.12, 'van': -.46}[style] + U(rs, -.02, .02)
    xf0 = {'sedan': .2, 'hatch': .22, 'sport': .16, 'van': .36}[style] + U(rs, -.03, .03)
    xf1 = {'sedan': .07, 'hatch': .08, 'sport': .03, 'van': .26}[style] + U(rs, -.03, .03)
    cv.paint(cv.ell(0, gy + .005, .53, .035, soft=2), C(0, 0, 0), op=U(rs, .3, .7))
    sh = .1 - 0.35 * cv.ramp(topb, botb) + 0.04 * noise(rs, 5)
    cv.paint(cv.poly([(xr0, topb + .012), (xf0, topb + .012), (xf1, roof), (xr1, roof)]), b, tex=sh)
    cv.paint(cv.box(0, (topb + botb) / 2, 1.0, botb - topb, r=.045), b, tex=sh)
    g = glass_col(rs); d = .022
    cv.paint(cv.poly([(xr0 + d * 1.6, topb), (xf0 - d * 1.6, topb), (xf1 - d * .3, roof + d), (xr1 + d * .6, roof + d)]), g)
    px_ = (xr1 + xf1) / 2 + U(rs, -.04, .04)
    cv.paint(cv.cap(px_, topb, px_, roof + d, .012), b)
    cv.paint(cv.box(0, topb + .028, .92, .022), np.clip(b * 1.25 + .1, 0, 1), op=U(rs, .2, .7))
    cv.paint(cv.box(0, botb - .015, .96, .03), b * U(rs, .35, .7))
    hub = jit(rs, C(.72, .72, .74), .02, .15) if chance(rs, .8) else b
    for x in (-wx, wx):
        cv.paint(cv.ell(x, gy - wr, wr * 1.18, wr * 1.18), b * .3)
        wheel(cv, x, gy - wr, wr, hub)
    cv.paint(cv.ell(.475, topb + .05, .028, .02), C(.98, .96, .85))
    cv.paint(cv.ell(-.48, topb + .045, .022, .02), C(.8, .1, .08))
    return gy


def car_34(cv):
    """Three-quarter view, front (or rear) face toward +x; unit = apparent length."""
    rs = cv.rs
    b = jit(rs, pick(rs, CARCOL, CARP), .03, .12)
    rear = chance(rs, .3)
    xc = U(rs, .1, .3)                                  # where the nearest vertical edge is
    r1 = U(rs, -.26, -.19); r2 = r1 + U(rs, -.02, .05)
    so = U(rs, .3, .7)
    cv.paint(np.maximum(cv.poly([(-.53, .09), (xc, .2), (xc, .27), (-.5, .15)], soft=2), cv.poly([(xc, .2), (.53, .15), (.5, .21), (xc, .27)], soft=2)), C(0, 0, 0), op=so)
    side = b * U(rs, .7, .95); front = b * U(rs, .8, 1.0)
    g = glass_col(rs)
    cv.paint(np.maximum(cv.poly([(-.44, -.03), (-.32, r1), (xc - .1, r1 - .03), (xc, -.07)]), cv.poly([(xc - .1, r1 - .03), (xc + .14, r2), (xc + .22, -.06), (xc, -.07)])), b)
    cv.paint(cv.poly([(-.5, -.03), (xc, -.07), (xc, .17), (-.5, .1)]), side, tex=-.2 * cv.ramp(-.07, .17))
    cv.paint(cv.poly([(xc, -.07), (.5, -.03), (.5, .13), (xc, .17)]), front, tex=-.2 * cv.ramp(-.07, .17))
    cv.paint(cv.poly([(xc, -.07), (xc + .22, -.06), (.5, -.03), (xc + .05, -.035)]), np.clip(b * 1.2 + .08, 0, 1))   # bonnet / boot lid
    cv.paint(cv.poly([(-.37, -.055), (-.29, r1 + .025), (xc - .11, r1 - .005), (xc - .04, -.085)]), g)
    cv.paint(cv.poly([(xc - .07, r1 - .005), (xc + .13, r2 + .02), (xc + .2, -.07), (xc + .01, -.085)]), g * U(rs, .9, 1.6))
    px_ = (-.33 + xc - .08) / 2
    cv.paint(cv.cap(px_, -.07, px_ + .02, r1 + .01, .012), b)
    fx = (xc + .5) / 2
    if rear:
        cv.paint(cv.ell(xc + .04, .0, .03, .022), C(.8, .1, .08)); cv.paint(cv.ell(.47, .01, .025, .02), C(.8, .1, .08))
        cv.paint(cv.box(fx, .075, .1, .035, a=-.12), C(.92, .92, .88))
    else:
        cv.paint(cv.ell(xc + .045, .005, .035, .022), C(.98, .97, .88)); cv.paint(cv.ell(.465, .015, .028, .02), C(.98, .97, .88))
        cv.paint(cv.box(fx, .04, (.5 - xc) * .55, .04, a=-.12), C(.1, .1, .1))
    cv.paint(cv.poly([(xc, .115), (.5, .085), (.5, .13), (xc, .17)]), b * .55 if chance(rs, .6) else C(.15, .15, .15))
    hub = jit(rs, C(.72, .72, .74), .02, .15)
    wheel(cv, xc - .1, .135, .085, hub, rx=.06)
    wheel(cv, -.35, .085, .07, hub, rx=.052)
    cv.paint(cv.ell(.47, .13, .03, .06), TYRE)
    return .2


def car_front(cv):
    """Front or rear view; body about 0.8 units wide."""
    rs = cv.rs
    b = jit(rs, pick(rs, CARCOL, CARP), .03, .12)
    rear = chance(rs, .4)
    w = U(rs, .76, .86); h = U(rs, .26, .34); rt = -U(rs, .22, .3); rw = w * U(rs, .62, .72)
    cv.paint(cv.ell(0, .1 + h / 2, w * .58, .045, soft=2), C(0, 0, 0), op=U(rs, .3, .7))
    for sg in (-1, 1):
        cv.paint(cv.box(sg * w * .39, .1 + h / 2 - .02, .12, .11, r=.02), TYRE)
    cv.paint(cv.poly([(-w * .46, -.03), (-rw / 2, rt), (rw / 2, rt), (w * .46, -.03)]), b)
    cv.paint(cv.box(0, .1, w, h, r=.06), b, tex=-.2 * cv.ramp(-.05, .25))
    cv.paint(cv.poly([(-w * .41, -.05), (-rw / 2 + .03, rt + .03), (rw / 2 - .03, rt + .03), (w * .41, -.05)]), glass_col(rs))
    if not rear:
        cv.paint(cv.poly([(-w * .46, -.04), (w * .46, -.04), (w * .5, .01), (-w * .5, .01)]), np.clip(b * 1.2 + .08, 0, 1))
        cv.paint(cv.box(0, .07, w * U(rs, .3, .5), U(rs, .05, .09), r=.01), C(.1, .1, .1))
    lc = C(.8, .1, .08) if rear else C(.98, .97, .88)
    for sg in (-1, 1):
        cv.paint(cv.ell(sg * w * .36, .05, .065, .035), lc)
        if chance(rs, .5):
            cv.paint(cv.box(sg * (w / 2 + .03), -.05, .05, .035), b * .7)
    cv.paint(cv.box(0, .1 + h / 2 - .06, w * 1.0, .06, r=.02), b * .6 if chance(rs, .6) else C(.2, .2, .2))
    cv.paint(cv.box(0, .1 + h / 2 - .07, .15, .04), C(.93, .93, .88) if chance(rs, .7) else C(.9, .8, .2))
    return .1 + h / 2 + .03


# ------------------------------------------------------------------ truck
def truck_side(cv):
    """Side view; unit = truck length; wheel bottoms at y = 0.30."""
    rs = cv.rs
    gy = .3; wr = U(rs, .062, .085)
    kind = pick(rs, ['box', 'semi', 'dump', 'tank', 'fire', 'flat'], [30, 24, 16, 8, 12, 10])
    cab = jit(rs, pick(rs, TRUCKCOL, TRUCKP), .03, .1)
    if kind == 'fire':
        cab = jit(rs, C(.78, .08, .06), .03, .1)
    if kind == 'dump' and chance(rs, .6):
        cab = jit(rs, pick(rs, [C(.92, .76, .1), C(.9, .45, .08), C(.78, .1, .08)]), .03, .1)
    white = jit(rs, C(.93, .93, .93), .01, .06)
    cargo = white if chance(rs, .45) else (jit(rs, C(.7, .72, .74), .02, .1) if chance(rs, .35) else (cab if chance(rs, .5) else jit(rs, pick(rs, TRUCKCOL), .03, .1)))
    conv = chance(rs, .5)
    cx0 = U(rs, .2, .27) if kind != 'semi' else U(rs, .26, .32)
    ctop = -U(rs, .16, .26)
    cv.paint(cv.ell(0, gy + .005, .54, .035, soft=2), C(0, 0, 0), op=U(rs, .3, .7))
    cv.paint(cv.box(-.02, .175, .94, .09), C(.1, .1, .1))                       # chassis
    sh = .08 - 0.28 * cv.ramp(-.3, .2)
    btop = -U(rs, .28, .4)
    if kind in ('box', 'semi'):
        x0 = -.5; x1 = cx0 - .03
        cv.paint(cv.box((x0 + x1) / 2, (btop + .14) / 2, x1 - x0, .14 - btop, r=.01), cargo, tex=sh + .03 * noise(rs, 8, 3))
        if chance(rs, .35):
            cv.paint(cv.box((x0 + x1) / 2 + U(rs, -.1, .1), (btop + .14) / 2 + U(rs, -.08, .08), U(rs, .15, .5), U(rs, .05, .2)), jit(rs, pick(rs, TRUCKCOL[1:]), .03, .1))
        if chance(rs, .25):
            cv.paint(cv.box((x0 + x1) / 2, .12, x1 - x0, .03), cargo * .6)
    elif kind == 'dump':
        dc = cab if chance(rs, .6) else jit(rs, pick(rs, [C(.4, .42, .45), C(.92, .76, .1), C(.2, .3, .5), C(.5, .5, .5)]), .03, .1)
        tilt = U(rs, .25, .6) if chance(rs, .2) else 0.0
        cv.paint(cv.poly([(-.5, btop * .7 - tilt * .7), (cx0 - .04, btop * .85), (cx0 - .02, .12), (-.44, .13 - tilt * .2)]), dc, tex=sh)
        cv.paint(cv.poly([(-.5, btop * .7 - tilt * .7), (cx0 - .04, btop * .85), (cx0 - .04, btop * .85 + .04), (-.5, btop * .7 - tilt * .7 + .04)]), dc * .7)
    elif kind == 'tank':
        tc = jit(rs, pick(rs, [C(.75, .77, .8), C(.93, .93, .93), C(.8, .1, .1), C(.9, .75, .1), C(.2, .3, .6)], [40, 30, 10, 10, 10]), .02, .08)
        cv.paint(cv.box((-.5 + cx0 - .03) / 2, -.07, cx0 + .47, .38, r=.16), tc, tex=-.3 * cv.ramp(-.26, .12))
        cv.paint(cv.cap(-.4, -.17, cx0 - .12, -.17, .02, soft=2), C(1, 1, 1), op=.5)
    elif kind == 'fire':
        cv.paint(cv.box((-.5 + cx0) / 2, -.03, cx0 + .5, .36, r=.01), cab, tex=sh)
        for x in rs.uniform(-.45, cx0 - .1, 3):
            cv.paint(cv.box(x, U(rs, -.02, .08), U(rs, .08, .16), U(rs, .08, .14)), jit(rs, C(.65, .66, .68), .02, .15))
        cv.paint(cv.cap(-.48, -.1, .48, -.1, .014), C(.95, .95, .9), op=.9)
        if chance(rs, .7):
            cv.paint(cv.box(-.1, -.25, .75, .045, a=U(rs, -.06, .02)), jit(rs, C(.7, .7, .7), .02, .15))
    else:                                               # flat bed with load
        cv.paint(cv.box((-.5 + cx0) / 2, .1, cx0 + .48, .05), cab * .7)
        for x in rs.uniform(-.42, cx0 - .12, int(pick(rs, [1, 2, 3]))):
            hh = U(rs, .08, .28)
            cv.paint(cv.box(x, .08 - hh / 2, U(rs, .12, .3), hh, r=.01), jit(rs, pick(rs, [C(.5, .35, .2), C(.6, .6, .6), C(.3, .4, .3), C(.8, .7, .5), C(.4, .3, .25)]), .03, .15))
    # cab
    if conv and kind != 'fire':
        cv.paint(cv.box((cx0 + .37) / 2, (ctop + .2) / 2, .37 - cx0, .2 - ctop, r=.02), cab, tex=sh)
        cv.paint(cv.box(.43, .1, .15, .2, r=.02), cab, tex=sh)                  # bonnet
        cv.paint(cv.box((cx0 + .37) / 2 + .02, ctop + .08, (.37 - cx0) * .6, .1), glass_col(rs))
    else:
        cv.paint(cv.box((cx0 + .5) / 2, (ctop + .2) / 2, .5 - cx0, .2 - ctop, r=.025), cab, tex=sh)
        cv.paint(cv.box((cx0 + .5) / 2 + .035, ctop + .09, (.5 - cx0) * .62, .12), glass_col(rs))
    cv.paint(cv.box(.485, .17, .05, .05), C(.6, .6, .62) if chance(rs, .6) else C(.15, .15, .15))  # bumper
    cv.paint(cv.ell(.485, .1, .015, .02), C(.98, .96, .85))
    if kind == 'semi' and chance(rs, .6):
        cv.paint(cv.cap(cx0 - .012, ctop - U(rs, .08, .2), cx0 - .012, .1, .012), jit(rs, C(.6, .6, .62), .02, .2))
    hub = jit(rs, pick(rs, [C(.72, .72, .74), C(.9, .9, .9), C(.75, .15, .1), C(.9, .75, .15), C(.3, .3, .3)], [45, 20, 12, 10, 13]), .02, .1)
    xs = [.37, -.32]
    if chance(rs, .7):
        xs.append(-.32 + 2.1 * wr)
    if kind == 'semi':
        xs += [.1, -.42 + U(rs, 0, .04)]
    for x in xs:
        wheel(cv, x, gy - wr, wr, hub)
    return gy


def truck_34(cv):
    """Three-quarter view with the cab front toward +x and the body receding to -x."""
    rs = cv.rs
    cab = jit(rs, pick(rs, TRUCKCOL, TRUCKP), .03, .1)
    white = jit(rs, C(.93, .93, .93), .01, .06)
    cargo = white if chance(rs, .5) else (cab if chance(rs, .5) else jit(rs, pick(rs, TRUCKCOL), .03, .1))
    xe = U(rs, .08, .22)                                # nearest vertical edge of the cab
    ct = -U(rs, .24, .34)
    hasbox = chance(rs, .75)
    bt = ct - U(rs, .04, .14) if hasbox else ct
    conv = chance(rs, .45)
    so = U(rs, .3, .7)
    cv.paint(np.maximum(cv.poly([(-.55, .1), (xe, .3), (xe, .38), (-.52, .18)], soft=2), cv.poly([(xe, .3), (.54, .26), (.52, .32), (xe, .38)], soft=2)), C(0, 0, 0), op=so)
    sh = .08 - 0.28 * cv.ramp(-.35, .25)
    if hasbox:
        kind = pick(rs, ['box', 'dump', 'tank'], [65, 22, 13])
        if kind == 'box':
            cv.paint(cv.box((xe - .04 + .5) / 2, (bt + ct) / 2 + .02, .5 - xe + .04, ct - bt + .06), cargo * U(rs, .9, 1.0))
            cv.paint(cv.poly([(xe - .05, bt), (xe - .05, .17), (-.5, .08), (-.5, bt * U(rs, .6, .8))]), cargo * U(rs, .7, .92), tex=sh)
        elif kind == 'dump':
            dc = cab if chance(rs, .6) else jit(rs, pick(rs, [C(.4, .42, .45), C(.92, .76, .1), C(.5, .5, .5)]), .03, .1)
            cv.paint(cv.poly([(xe - .03, bt), (xe - .05, .1), (-.5, .05), (-.52, bt * .75)]), dc * .85, tex=sh)
            cv.paint(cv.poly([(xe - .03, bt), (.4, bt + .02), (.4, bt + .07), (xe - .03, bt + .05)]), dc)
        else:
            tc = jit(rs, pick(rs, [C(.75, .77, .8), C(.93, .93, .93), C(.8, .1, .1)]), .02, .08)
            cv.paint(cv.ell(-.2, -.06, .36, .2, a=.1), tc, tex=-.3 * cv.ramp(-.25, .12))
    else:
        cv.paint(cv.poly([(xe - .05, -.02), (xe - .05, .17), (-.5, .08), (-.5, -.04)]), cab * .6)
    cv.paint(cv.poly([(-.5, .06), (xe, .18), (xe, .25), (-.5, .11)]), C(.1, .1, .1))            # chassis side
    cv.paint(cv.poly([(xe, ct), (xe, .24), (xe - .13, .2), (xe - .13, ct + .03)]), cab * U(rs, .6, .85), tex=sh)   # cab side
    cv.paint(cv.poly([(xe - .02, ct + .05), (xe - .02, ct + .2), (xe - .11, ct + .2), (xe - .11, ct + .07)]), glass_col(rs))
    cv.paint(cv.box((xe + .5) / 2, (ct + .24) / 2, .5 - xe, .24 - ct, r=.02), cab, tex=sh)       # cab front
    g = glass_col(rs)
    fw = .5 - xe; fx = (xe + .5) / 2
    if conv:
        cv.paint(cv.box(fx, ct + .09, fw * .8, .13), g)
        cv.paint(cv.box(fx, .06, fw * .6, .2, r=.01), C(.62, .64, .67) if chance(rs, .6) else C(.12, .12, .12))
        cv.paint(cv.box(fx, -.045, fw * .92, .03), np.clip(cab * 1.2 + .05, 0, 1))
    else:
        cv.paint(cv.box(fx, ct + .12, fw * .86, .18), g)
        cv.paint(cv.box(fx, .07, fw * .55, .1, r=.01), C(.12, .12, .12) if chance(rs, .6) else C(.62, .64, .67))
    if chance(rs, .4):
        cv.paint(cv.cap(fx, ct + .03, fx, ct + .2, .008), cab)
    for x in (xe + .05, .455):
        cv.paint(cv.ell(x, .15, .03, .025), C(.98, .97, .88))
    cv.paint(cv.box(fx, .215, fw + .04, .06, r=.01), C(.65, .66, .68) if chance(rs, .5) else (C(.12, .12, .12) if chance(rs, .5) else cab * .8))
    for x in (xe + .06, .45):
        cv.paint(cv.box(x, .275, .08, .09, r=.02), TYRE)
    hub = jit(rs, C(.72, .72, .74), .02, .15)
    wheel(cv, xe - .08, .245, .085, hub, rx=.05)
    wheel(cv, -.3, .15, .065, hub, rx=.04)
    if chance(rs, .6):
        wheel(cv, -.19, .17, .07, hub, rx=.043)
    return .32


def truck_front(cv):
    """Head-on view; cab about 0.66 units wide."""
    rs = cv.rs
    cab = jit(rs, pick(rs, TRUCKCOL, TRUCKP), .03, .1)
    white = jit(rs, C(.93, .93, .93), .01, .06)
    w = U(rs, .6, .72); top = -U(rs, .36, .46); bot = .32
    cv.paint(cv.ell(0, bot + .08, w * .62, .05, soft=2), C(0, 0, 0), op=U(rs, .3, .7))
    if chance(rs, .55):
        cargo = white if chance(rs, .6) else jit(rs, pick(rs, TRUCKCOL), .03, .1)
        cv.paint(cv.box(0, top + .1, w + U(rs, .04, .14), .6, r=.01), cargo)
    for sg in (-1, 1):
        cv.paint(cv.box(sg * w * .38, bot + .05, .13, .13, r=.02), TYRE)
        if chance(rs, .35):
            cv.paint(cv.cap(sg * (w / 2 + .03), -.1, sg * (w / 2 + .03), top - U(rs, .02, .12), .014), jit(rs, C(.65, .65, .67), .02, .2))
        cv.paint(cv.box(sg * (w / 2 + .045), -.16, .04, .1), C(.15, .15, .15))
    ct = top + U(rs, .04, .1)
    cv.paint(cv.box(0, (ct + bot) / 2, w, bot - ct, r=.04), cab, tex=-.15 * cv.ramp(ct, bot))
    wh = U(rs, .17, .24)
    cv.paint(cv.box(0, ct + .05 + wh / 2, w * .86, wh, r=.02), glass_col(rs))
    if chance(rs, .4):
        cv.paint(cv.cap(0, ct + .05, 0, ct + .05 + wh, .01), cab)
    gh = U(rs, .12, .24)
    cv.paint(cv.box(0, bot - .1 - gh / 2, w * U(rs, .4, .6), gh, r=.01), C(.62, .64, .67) if chance(rs, .5) else C(.12, .12, .12))
    for sg in (-1, 1):
        cv.paint(cv.ell(sg * w * .37, bot - .12, .05, .035), C(.98, .97, .88))
    cv.paint(cv.box(0, bot - .03, w + .06, .075, r=.01), C(.65, .66, .68) if chance(rs, .5) else (C(.12, .12, .12) if chance(rs, .5) else cab * .8))
    return bot + .11

# ---------------------------------------------------------------------------------------------
# PART 3. Shape models of the six animal classes, typed from general knowledge:
# ellipses for bodies and heads, tapered segments for necks, legs and tails, small parts for ears,
# beaks, eyes and antlers.
# ---------------------------------------------------------------------------------------------

BLACK = C(.07, .065, .06)
WHITE = C(.93, .92, .9)


def leg(cv, x0, y0, x1, y1, r, cu, cl, knee=0.0, tex=None):
    xm, ym = (x0 + x1) / 2 + knee, (y0 + y1) / 2
    cv.paint(cv.cap(x0, y0, xm, ym, r * 1.3, r * .85), cu, tex=tex)
    cv.paint(cv.cap(xm, ym, x1, y1, r * .85, r * .75), cl, tex=tex)


def tri_ear(cv, x, y, w, h, lean, col, inner=None):
    cv.paint(cv.poly([(x - w / 2, y), (x + w / 2, y), (x + lean, y - h)]), col)
    if inner is not None:
        cv.paint(cv.poly([(x - w / 4, y), (x + w / 4, y), (x + lean * .9, y - h * .7)]), inner)


def patches(cv, mask, col, n, r0=.08, r1=.25, lim=.5):
    rs = cv.rs
    for _ in range(n):
        m = cv.ell(U(rs, -lim, lim), U(rs, -lim * .6, lim * .6), U(rs, r0, r1), U(rs, r0, r1), a=U(rs, -1, 1), soft=1.5)
        cv.paint(m * mask, col)


# ------------------------------------------------------------------ horse
def horse_cols(rs):
    kind = pick(rs, ['bay', 'chestnut', 'black', 'grey', 'white', 'palomino', 'darkbay', 'pinto'], [22, 18, 11, 9, 13, 8, 13, 6])
    bc = {'bay': C(.45, .27, .14), 'chestnut': C(.6, .33, .16), 'black': C(.1, .09, .08), 'grey': C(.6, .6, .6), 'white': C(.9, .9, .87),
          'palomino': C(.82, .65, .38), 'darkbay': C(.28, .17, .1), 'pinto': C(.5, .3, .16)}[kind]
    bc = jit(rs, bc, .03, .15)
    if kind in ('bay', 'darkbay', 'black'):
        pc = BLACK
    elif kind == 'palomino':
        pc = C(.92, .88, .76)
    elif kind in ('grey', 'white'):
        pc = bc * U(rs, .5, 1.0)
    else:
        pc = bc * U(rs, .55, .95)
    lc = pc if kind in ('bay', 'darkbay') else (WHITE if chance(rs, .25) else bc)
    return kind, bc, pc, lc


def rider(cv, ry):
    rs = cv.rs
    shirt = jit(rs, pick(rs, [C(.92, .92, .92), C(.75, .1, .1), C(.15, .25, .65), C(.08, .08, .1), C(.85, .5, .6), C(.2, .5, .25), C(.9, .8, .2), C(.5, .5, .55)]), .03, .1)
    pants = jit(rs, pick(rs, [C(.85, .82, .75), C(.15, .15, .2), C(.25, .3, .5), C(.93, .93, .93)]), .03, .1)
    lean = U(rs, -.03, .1)
    cv.paint(cv.cap(.03, -ry + .05, .13, ry * .7, .055, .045), pants)
    cv.paint(cv.cap(.13, ry * .7, .16, ry * .7 + .1, .04), BLACK)
    cv.paint(cv.cap(.02, -ry, .04 + lean, -ry - .38, .1, .085), shirt)
    cv.paint(cv.cap(.06 + lean, -ry - .3, .2 + lean, -ry - .08, .035), shirt * .9)
    cv.paint(cv.ell(.06 + lean, -ry - .52, .07, .075), jit(rs, C(.85, .68, .55), .03, .1))
    cv.paint(cv.ell(.055 + lean, -ry - .56, .078, .05), BLACK if chance(rs, .7) else jit(rs, C(.5, .35, .2), .05, .3))
    return ry + .62


def horse_side(cv, graze=False, with_rider=False):
    """Side view facing +x; unit = body length; hooves at about y = 0.9."""
    rs = cv.rs
    kind, bc, pc, lc = horse_cols(rs)
    ry = U(rs, .22, .29); gy = U(rs, .82, .95); r = U(rs, .04, .055)
    tex = cv.fur(.025, .04)
    sw = rs.uniform(-.1, .1, 4) if chance(rs, .6) else rs.uniform(-.3, .3, 4)
    xf = .34 + U(rs, -.03, .03); xb = -.37 + U(rs, -.03, .03)
    leg(cv, xf + .06, ry * .4, xf + .06 + sw[0], gy - .03, r * .9, bc * .6, lc * .6, knee=sw[0] * .3)
    leg(cv, xb + .07, ry * .4, xb + .07 + sw[1], gy - .03, r * .9, bc * .6, lc * .6, knee=-abs(sw[1]) * .3)
    cv.paint(cv.cap(-.5, -.1, -.6 + U(rs, -.1, .03), U(rs, .25, .55), .04, .075, soft=1.5), pc)
    sh = tex + .15 - 0.5 * cv.ramp(-ry * .6, ry)
    cv.paint(cv.ell(0, 0, .5, ry), bc, tex=sh)
    cv.paint(cv.ell(-.33, -.02, .2, ry * 1.05), bc, tex=sh)
    cv.paint(cv.ell(.34, 0, .18, ry), bc, tex=sh)
    if kind == 'pinto':
        patches(cv, cv.ell(0, 0, .5, ry), WHITE, 3, .1, .22, .4)
    leg(cv, xf, ry * .4, xf + sw[2], gy, r, bc, lc, knee=sw[2] * .3, tex=tex)
    leg(cv, xb, ry * .4, xb + sw[3], gy, r, bc, lc, knee=-abs(sw[3]) * .3, tex=tex)
    if graze:
        nx, ny = .7 + U(rs, -.05, .05), .45 + U(rs, -.1, .1)
        hx, hy = nx + U(rs, .05, .15), min(ny + U(rs, .22, .32), gy - .04)
        cv.paint(cv.cap(.38, -.06, nx, ny, .17, .1), bc, tex=sh)
        top = -ry
    else:
        nx, ny = .62 + U(rs, -.07, .07), -.5 + U(rs, -.12, .1)
        hx, hy = nx + U(rs, .15, .25), ny + U(rs, .14, .3)
        cv.paint(cv.cap(.36, -.1, nx, ny, .18, .105), bc, tex=sh)
        top = -ny + .12
    cv.paint(cv.cap(.3, -ry * .75, nx - .09, ny - .05, .035, .04, soft=1.5), pc)          # mane
    cv.paint(cv.cap(nx, ny - .01, hx, hy, .095, .055), bc, tex=tex)
    if chance(rs, .3):
        cv.paint(cv.cap(nx + .04, ny + .0, hx, hy - .01, .02), WHITE)                       # blaze
    tri_ear(cv, nx - .03, ny - .08, .06, .09, .01, bc)
    cv.paint(cv.ell(nx + .05, ny, .018, .018), BLACK)
    if with_rider:
        top = max(top, rider(cv, ry))
    return gy, top


def horse_head(cv):
    """Portrait: neck rising from the lower-left corner, long head pointing right and down."""
    rs = cv.rs
    kind, bc, pc, lc = horse_cols(rs)
    tex = cv.fur(.025, .04)
    cv.paint(cv.cap(-.55, .95, -.02, -.1, .4, .2), bc, tex=tex - .2 * cv.rampx(-.5, .2))
    cv.paint(cv.cap(-.75, .6, -.17, -.25, .1, .06, soft=2), pc)                          # mane down the neck
    hx, hy = U(rs, .32, .48), U(rs, .2, .42)
    cv.paint(cv.cap(-.02, -.15, hx, hy, .19, .1), bc, tex=tex)
    tri_ear(cv, -.1, -.3, .09, .15, -.02, bc); tri_ear(cv, .03, -.31, .09, .15, .03, bc * .85)
    if chance(rs, .4):
        cv.paint(cv.cap(.05, -.2, hx, hy - .03, .035, .03), WHITE)
    cv.paint(cv.ell(.07, -.08, .03, .028), BLACK)
    cv.paint(cv.ell(hx + .02, hy + .01, .025, .02), BLACK)
    if chance(rs, .35):                                  # halter / bridle straps
        sc_ = jit(rs, pick(rs, [C(.1, .08, .06), C(.4, .25, .12), C(.6, .1, .1), C(.15, .2, .5)]), .03, .1)
        cv.paint(cv.cap(.25, .0, .16, .3, .015), sc_); cv.paint(cv.cap(-.05, -.25, .2, .25, .012), sc_)


def horse_front(cv):
    """Front or three-quarter view: narrow body, long face, four straight legs."""
    rs = cv.rs
    kind, bc, pc, lc = horse_cols(rs)
    tex = cv.fur(.025, .04)
    rx = U(rs, .2, .3); gy = U(rs, .85, .97); r = U(rs, .045, .06)
    hxo = U(rs, -.15, .15)
    for sg in (-1, 1):
        leg(cv, sg * rx * .75 + .05, .2, sg * rx * .7 + .05, gy - .04, r * .9, bc * .6, lc * .6)
    cv.paint(cv.ell(.08 * np.sign(hxo + 1e-9) * -1, .0, rx * 1.1, .3), bc * .8, tex=tex)   # barrel behind
    cv.paint(cv.ell(0, .05, rx, .3), bc, tex=tex - .2 * cv.ramp(-.2, .35))
    for sg in (-1, 1):
        leg(cv, sg * rx * .45, .2, sg * rx * .45 + U(rs, -.03, .03), gy, r, bc, lc, tex=tex)
    cv.paint(cv.cap(0, -.05, hxo * .6, -.42, .2, .13), bc, tex=tex)
    cv.paint(cv.cap(hxo * .6, -.48, hxo, -.2 + U(rs, -.1, .1), .11, .075), bc, tex=tex)
    tri_ear(cv, hxo * .6 - .07, -.56, .06, .1, -.01, bc); tri_ear(cv, hxo * .6 + .07, -.56, .06, .1, .01, bc)
    cv.paint(cv.cap(hxo * .6, -.6, hxo * .6, -.48, .04, soft=1.5), pc)
    if chance(rs, .4):
        cv.paint(cv.cap(hxo * .6, -.5, hxo, -.25, .025), WHITE)
    return gy, .68


# ------------------------------------------------------------------ deer
def deer_cols(rs):
    bc = jit(rs, pick(rs, [C(.68, .52, .34), C(.58, .37, .2), C(.46, .38, .3), C(.3, .2, .13), C(.78, .66, .48), C(.5, .32, .18), C(.16, .12, .09)],
                      [24, 22, 16, 10, 10, 12, 6]), .03, .15)
    return bc


def antlers(cv, x, y, s, col, side=True):
    """Two branching beams; s = size. side=True: both sweep up and slightly back; else spread left/right."""
    rs = cv.rs
    r = .016 * max(s, .7)
    for k, sg in enumerate((-1, 1)):
        if side:
            x0 = x - .03 * k; dx1, dx2 = -.1 - .05 * k + U(rs, -.03, .03), .1 + U(rs, -.04, .06)
        else:
            x0 = x + sg * .04; dx1, dx2 = sg * U(rs, .12, .2), -sg * U(rs, .0, .1)
        p1 = (x0 + dx1 * s, y - U(rs, .16, .24) * s)
        p2 = (p1[0] + dx2 * s, p1[1] - U(rs, .14, .22) * s)
        cv.paint(cv.cap(x0, y, p1[0], p1[1], r), col)
        cv.paint(cv.cap(p1[0], p1[1], p2[0], p2[1], r * .9), col)
        fw = 1 if side else -sg
        for t in (.45, 1.0):
            bx, by = x0 + (p1[0] - x0) * t, y + (p1[1] - y) * t
            cv.paint(cv.cap(bx, by, bx + fw * U(rs, .06, .12) * s, by - U(rs, .07, .13) * s, r * .8), col)
        if chance(rs, .5):
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            cv.paint(cv.cap(mx, my, mx - fw * U(rs, .04, .1) * s, my - U(rs, .06, .12) * s, r * .8), col)


def deer_side(cv, pose='stand'):
    """Side view facing +x: slim body, thin legs, upright neck, small head, antlers on about half."""
    rs = cv.rs
    bc = deer_cols(rs)
    big = chance(rs, .12)                                # moose / elk: bulkier and darker
    ry = U(rs, .19, .25) * (1.15 if big else 1.0); gy = U(rs, .8, .95); r = U(rs, .022, .033) * (1.3 if big else 1.0)
    tex = cv.fur(.03, .04)
    belly = np.clip(bc * 1.25 + .1, 0, 1) if chance(rs, .6) else bc
    acol = jit(rs, pick(rs, [C(.75, .65, .5), C(.5, .4, .3), C(.3, .22, .15), C(.88, .82, .7)]), .03, .1)
    if pose == 'lie':
        gy = ry + .08
        cv.paint(cv.ell(.25, gy - .04, .22, .06), bc * .7); cv.paint(cv.ell(-.3, gy - .05, .2, .08), bc * .75)
    else:
        sw = rs.uniform(-.08, .08, 4) if chance(rs, .65) else rs.uniform(-.28, .28, 4)
        xf = .36 + U(rs, -.03, .03); xb = -.38 + U(rs, -.03, .03)
        leg(cv, xf + .05, ry * .4, xf + .05 + sw[0], gy - .03, r * .9, bc * .6, bc * .5, knee=sw[0] * .3)
        leg(cv, xb + .06, ry * .4, xb + .06 + sw[1], gy - .03, r * .9, bc * .6, bc * .5, knee=-abs(sw[1]) * .4)
    sh = tex + .12 - 0.3 * cv.ramp(-ry, ry * .4)
    cv.paint(cv.ell(0, 0, .5, ry), bc, tex=sh)
    cv.paint(cv.ell(-.34, -.02, .18, ry * 1.05), bc, tex=sh)
    cv.paint(cv.ell(.02, ry * .55, .4, ry * .5, soft=2), belly, op=.8)
    if chance(rs, .4):
        cv.paint(cv.ell(-.47, -.02, .07, ry * .6, soft=1.5), WHITE)                         # rump patch
    cv.paint(cv.cap(-.5, -ry * .4, -.56, -ry * .4 + U(rs, .04, .12), .035, .025), WHITE if chance(rs, .5) else bc * .6)
    if chance(rs, .08):
        patches(cv, cv.ell(0, 0, .5, ry), WHITE, 9, .015, .03, .4)                          # fawn spots
    if pose != 'lie':
        leg(cv, xf, ry * .4, xf + sw[2], gy, r, bc, bc * .75, knee=sw[2] * .3, tex=tex)
        leg(cv, xb, ry * .4, xb + sw[3], gy, r, bc, bc * .75, knee=-abs(sw[3]) * .4, tex=tex)
    if pose == 'graze':
        nx, ny = .68 + U(rs, -.05, .05), .4 + U(rs, -.1, .1)
        hx, hy = nx + U(rs, .08, .16), min(ny + U(rs, .15, .25), gy - .03)
        cv.paint(cv.cap(.4, -.05, nx, ny, .11, .065), bc, tex=sh)
        top = ry
    else:
        nx, ny = .5 + U(rs, -.05, .1), -.5 + U(rs, -.12, .1)
        hx, hy = nx + U(rs, .15, .22), ny + U(rs, -.02, .1)
        cv.paint(cv.cap(.39, -.07, nx, ny, .115 * (1.2 if big else 1), .065), bc, tex=sh)
        top = -ny + .1
    cv.paint(cv.cap(nx, ny - .02, hx, hy, .078, .042), bc, tex=tex)
    cv.paint(cv.ell(hx + .01, hy, .022, .022), BLACK)
    cv.paint(cv.ell(nx - .07, ny - .1, .035, .09, a=-.5), bc * .9)
    cv.paint(cv.ell(nx + .05, ny - .03, .016, .016), BLACK)
    if chance(rs, .55) and pose != 'graze':
        s = U(rs, .7, 1.3)
        antlers(cv, nx - .01, ny - .09, s, acol, side=True)
        top += .38 * s
    return gy, top


def deer_front(cv):
    """Facing the camera: narrow body, neck and head on top, ears out to the sides, antlers spread."""
    rs = cv.rs
    bc = deer_cols(rs)
    tex = cv.fur(.03, .04)
    rx = U(rs, .16, .23); gy = U(rs, .85, .97); r = U(rs, .024, .034)
    acol = jit(rs, pick(rs, [C(.75, .65, .5), C(.5, .4, .3), C(.3, .22, .15), C(.88, .82, .7)]), .03, .1)
    for sg in (-1, 1):
        leg(cv, sg * rx * .8 + .04, .25, sg * rx * .8 + .04, gy - .04, r * .9, bc * .6, bc * .5)
    off = U(rs, -.2, .2)
    cv.paint(cv.ell(off, .08, rx * 1.2, .26), bc * .8, tex=tex)
    cv.paint(cv.ell(0, .1, rx, .3), bc, tex=tex - .15 * cv.ramp(-.2, .4))
    if chance(rs, .5):
        cv.paint(cv.ell(0, .0, rx * .5, .16, soft=2), np.clip(bc * 1.3 + .1, 0, 1), op=.8)
    for sg in (-1, 1):
        leg(cv, sg * rx * .4, .3, sg * rx * .4 + U(rs, -.02, .02), gy, r, bc, bc * .75, tex=tex)
    hx = U(rs, -.05, .05)
    cv.paint(cv.cap(0, -.1, hx, -.42, .1, .075), bc, tex=tex)
    for sg in (-1, 1):
        cv.paint(cv.ell(hx + sg * .13, -.58, .085, .035, a=sg * -.4), bc * .9)
    cv.paint(cv.ell(hx, -.5, .078, .11), bc, tex=tex)
    cv.paint(cv.ell(hx, -.42, .045, .04), bc * .5); cv.paint(cv.ell(hx, -.41, .02, .015), BLACK)
    for sg in (-1, 1):
        cv.paint(cv.ell(hx + sg * .04, -.53, .014, .014), BLACK)
    top = .62
    if chance(rs, .6):
        s = U(rs, .7, 1.3); antlers(cv, hx, -.59, s, acol, side=False); top += .36 * s
    return gy, top


def deer_bust(cv):
    """Head and neck close-up, facing the camera: big ears, dark muzzle, antlers."""
    rs = cv.rs
    bc = deer_cols(rs)
    tex = cv.fur(.03, .04)
    acol = jit(rs, pick(rs, [C(.75, .65, .5), C(.5, .4, .3), C(.3, .22, .15)]), .03, .1)
    cv.paint(cv.cap(U(rs, -.2, .2), 1.0, 0, .1, .28, .17), bc, tex=tex)
    for sg in (-1, 1):
        cv.paint(cv.ell(sg * .3, -.24, .17, .075, a=sg * -.45), bc * .9)
        cv.paint(cv.ell(sg * .3, -.24, .11, .035, a=sg * -.45), np.clip(bc * 1.3 + .1, 0, 1))
    cv.paint(cv.ell(0, -.1, .17, .23), bc, tex=tex)
    cv.paint(cv.ell(0, .08, .105, .1), np.clip(bc * 1.2 + .05, 0, 1))
    cv.paint(cv.ell(0, .12, .055, .04), BLACK)
    for sg in (-1, 1):
        cv.paint(cv.ell(sg * .09, -.14, .028, .028), BLACK)
    if chance(rs, .6):
        antlers(cv, 0, -.3, U(rs, 1.4, 2.2), acol, side=False)


# ------------------------------------------------------------------ dog
def dog_cols(rs):
    """Returns main colour, second colour (patches, ears, muzzle) and whether two-toned."""
    k = pick(rs, ['white', 'black', 'brown', 'gold', 'tan', 'grey', 'bw', 'brw', 'bt', 'cream'], [15, 12, 12, 13, 8, 5, 10, 10, 8, 7])
    m = {'white': WHITE, 'black': BLACK, 'brown': C(.4, .26, .15), 'gold': C(.8, .6, .32), 'tan': C(.68, .5, .32), 'grey': C(.45, .45, .46),
         'bw': BLACK, 'brw': C(.5, .3, .16), 'bt': C(.1, .09, .08), 'cream': C(.9, .84, .7)}[k]
    s2 = {'bw': WHITE, 'brw': WHITE, 'bt': C(.65, .42, .22)}.get(k, None)
    m = jit(rs, m, .03, .12)
    two = s2 is not None
    if s2 is None:
        s2 = np.clip(m * U(rs, .55, 1.3) + U(rs, 0, .08), 0, 1)
    return m, s2, two


def dog_face(cv):
    """Head close-up: round head, hanging or pricked ears, pale muzzle with a big dark nose, dark eyes."""
    rs = cv.rs
    m, s2, two = dog_cols(rs)
    fluffy = chance(rs, .3)
    sf = 2.5 if fluffy else 1.0
    tex = cv.fur(.04, .06) if fluffy else cv.fur(.03, .04)
    rx = U(rs, .3, .4); ry = U(rs, .3, .38)
    cv.paint(cv.ell(U(rs, -.15, .15), .75, U(rs, .4, .6), .5, soft=sf), m if chance(rs, .7) else s2, tex=tex)   # chest / body below
    if chance(rs, .3):
        cv.paint(cv.cap(-.25, .36, .25, .36, .035), jit(rs, pick(rs, [C(.7, .1, .1), C(.15, .25, .6), C(.1, .1, .1), C(.6, .4, .2)]), .03, .1))
    ec = s2 if (two or chance(rs, .5)) else m * U(rs, .6, .9)
    if chance(rs, .6):                                   # hanging ears
        for sg in (-1, 1):
            cv.paint(cv.ell(sg * (rx + .0), U(rs, -.05, .1), U(rs, .09, .14), U(rs, .2, .3), a=sg * U(rs, -.1, .3), soft=sf), ec, tex=tex)
        cv.paint(cv.ell(0, 0, rx, ry, soft=sf), m, tex=tex)
    else:                                                # pricked ears
        for sg in (-1, 1):
            tri_ear(cv, sg * rx * .7, -ry * .6, U(rs, .16, .24), U(rs, .2, .32), sg * U(rs, .0, .1), ec, inner=jit(rs, C(.8, .6, .55), .03, .1) if chance(rs, .5) else None)
        cv.paint(cv.ell(0, 0, rx, ry, soft=sf), m, tex=tex)
    if two and chance(rs, .7):                           # blaze up the middle of the face
        cv.paint(cv.cap(0, -ry, 0, .1, U(rs, .04, .1), U(rs, .08, .14), soft=1.5), s2)
    mc = s2 if (two or chance(rs, .35)) else np.clip(m * U(rs, .9, 1.25) + U(rs, 0, .06), 0, 1)
    my = U(rs, .1, .18)
    cv.paint(cv.ell(0, my, U(rs, .15, .21), U(rs, .13, .18), soft=1.5 * sf), mc, tex=tex)
    cv.paint(cv.ell(0, my - U(rs, .03, .07), U(rs, .06, .085), U(rs, .045, .06)), BLACK)
    if chance(rs, .3):
        cv.paint(cv.ell(0, my + .13, .05, .07), jit(rs, C(.85, .4, .45), .03, .1))          # tongue
    else:
        cv.paint(cv.cap(-.06, my + .08, .06, my + .08, .012), BLACK, op=.7)
    ey = -U(rs, .06, .12); ex = U(rs, .12, .17)
    for sg in (-1, 1):
        cv.paint(cv.ell(sg * ex, ey, .038, .038), BLACK)


def dog_body(cv, pose='stand'):
    """Whole dog, facing +x: stocky body, short thick legs, big head with muzzle, tail."""
    rs = cv.rs
    m, s2, two = dog_cols(rs)
    fluffy = chance(rs, .3); sf = 2.2 if fluffy else 1.0
    tex = cv.fur(.04, .06) if fluffy else cv.fur(.03, .04)
    r = U(rs, .05, .075); hr = U(rs, .17, .23)
    ec = s2 if (two or chance(rs, .5)) else m * U(rs, .6, .9)
    if pose == 'stand':
        ry = U(rs, .22, .3); gy = U(rs, .58, .74)
        sw = rs.uniform(-.1, .1, 4)
        leg(cv, .36, ry * .4, .36 + sw[0], gy - .03, r * .9, m * .6, m * .6); leg(cv, -.32, ry * .4, -.32 + sw[1], gy - .03, r * .9, m * .6, m * .6)
        ty = U(rs, -.45, .2)
        cv.paint(cv.cap(-.48, -.08, -.68 + U(rs, -.05, .1), ty, .05, .035, soft=sf), m, tex=tex)
        body = cv.ell(0, 0, .5, ry, soft=sf)
        cv.paint(body, m, tex=tex + .1 - .4 * cv.ramp(-ry * .5, ry))
        if two:
            patches(cv, body, s2, int(pick(rs, [1, 2, 3])), .1, .25, .45)
            cv.paint(cv.ell(.1, ry * .7, .35, ry * .4, soft=2) * body, s2)
        fl = s2 if two and chance(rs, .6) else m
        leg(cv, .3, ry * .4, .3 + sw[2], gy, r, m, fl, tex=tex); leg(cv, -.38, ry * .4, -.38 + sw[3], gy, r, m, fl, knee=-.04, tex=tex)
        hx, hy = .55 + U(rs, -.05, .05), -.3 + U(rs, -.12, .1)
        cv.paint(cv.cap(.36, -.08, hx, hy, .19, .16, soft=sf), m, tex=tex)
        top = -hy + hr
    elif pose == 'sit':
        gy = U(rs, .68, .8)
        cv.paint(cv.cap(-.3, gy - .05, -.6, gy - U(rs, .02, .15), .045, .03, soft=sf), m)
        cv.paint(cv.ell(-.14, gy - .22, .28, .23, soft=sf), m, tex=tex)                       # haunch
        body = cv.ell(-.02, .12, .25, .4, a=-.3, soft=sf)
        cv.paint(body, m, tex=tex)
        if two:
            cv.paint(cv.ell(.1, .15, .12, .3, a=-.3, soft=2) * body, s2)                    # chest
        fl = s2 if two and chance(rs, .6) else m
        leg(cv, .13, .2, .15, gy, r, m, fl, tex=tex); leg(cv, .02, .25, .04, gy, r * .95, m * .8, fl * .8)
        hx, hy = .12 + U(rs, -.04, .06), -.36 + U(rs, -.06, .06)
        top = -hy + hr
    else:                                                # lying
        gy = U(rs, .38, .5)
        body = cv.ell(-.05, gy - .22, .5, .22, soft=sf)
        cv.paint(cv.cap(-.5, gy - .1, -.7, gy - .04, .045, .03, soft=sf), m)
        cv.paint(body, m, tex=tex - .2 * cv.ramp(gy - .44, gy))
        if two:
            patches(cv, body, s2, int(pick(rs, [1, 2])), .1, .25, .45)
        fl = s2 if two and chance(rs, .6) else m
        cv.paint(cv.cap(.3, gy - .06, .68, gy - .04, .06, .05), fl, tex=tex)
        hx, hy = .42 + U(rs, -.05, .05), gy - .36 + U(rs, -.08, .08)
        top = -hy + hr
    hang = chance(rs, .6)
    if not hang:
        tri_ear(cv, hx - .08, hy - hr * .6, .13, .2, -.02, ec)
    cv.paint(cv.ell(hx, hy, hr, hr * U(rs, .9, 1.0), soft=sf), m, tex=tex)
    mc = s2 if (two or chance(rs, .3)) else m
    mx, my = hx + hr * U(rs, .9, 1.25), hy + U(rs, .03, .1)
    cv.paint(cv.cap(hx + .04, hy + .04, mx, my, hr * .55, hr * .4, soft=sf), mc, tex=tex)
    cv.paint(cv.ell(mx + .03, my - .02, .04, .035), BLACK)
    if hang:
        cv.paint(cv.ell(hx - hr * .5, hy + .04, hr * .42, hr * .85, a=U(rs, -.2, .3), soft=sf), ec, tex=tex)
    cv.paint(cv.ell(hx + hr * .3, hy - hr * .15, .028, .028), BLACK)
    if chance(rs, .25):
        cv.paint(cv.cap(hx - hr * .2, hy + hr * .85, hx + hr * .3, hy + hr * .75, .03), jit(rs, pick(rs, [C(.7, .1, .1), C(.15, .25, .6), C(.1, .1, .1)]), .03, .1))
    return gy, top


# ------------------------------------------------------------------ cat
def cat_cols(rs):
    """Returns coat colour, stripe colour (or None), white-marked flag and eye colour."""
    k = pick(rs, ['orange', 'gtabby', 'btabby', 'black', 'white', 'grey', 'tux', 'calico', 'siam', 'cream'], [15, 18, 13, 11, 9, 10, 9, 5, 5, 5])
    m = {'orange': C(.8, .5, .22), 'gtabby': C(.5, .48, .44), 'btabby': C(.48, .37, .26), 'black': C(.08, .075, .07), 'white': WHITE,
         'grey': C(.42, .43, .46), 'tux': C(.09, .085, .08), 'calico': C(.85, .82, .76), 'siam': C(.85, .78, .65), 'cream': C(.88, .76, .58)}[k]
    m = jit(rs, m, .03, .12)
    stripe = {'orange': m * .7, 'gtabby': m * .4, 'btabby': m * .4, 'cream': m * .85}.get(k, None)
    eye = jit(rs, pick(rs, [C(.8, .7, .2), C(.45, .65, .3), C(.4, .6, .8), C(.75, .55, .15)], [40, 30, 12, 18]), .03, .1)
    return k, m, stripe, eye


def cat_head(cv, hx, hy, r, k, m, stripe, eye, tex, detail=True):
    """Round wide head with two pointed ears, light eyes with dark pupils, small nose."""
    rs = cv.rs
    rx = r * U(rs, 1.08, 1.22); ry = r * U(rs, .88, 1.0)
    inner = jit(rs, C(.85, .62, .6), .03, .1)
    for sg in (-1, 1):
        cv.paint(cv.poly([(hx + sg * rx * .28, hy - ry * .6), (hx + sg * rx * .98, hy - ry * .35), (hx + sg * rx * U(rs, .78, .95), hy - ry * U(rs, 1.45, 1.75))]), m)
        if detail:
            cv.paint(cv.poly([(hx + sg * rx * .45, hy - ry * .65), (hx + sg * rx * .85, hy - ry * .5), (hx + sg * rx * .78, hy - ry * 1.3)]), inner if k != 'black' else m * 1.5)
    cv.paint(cv.ell(hx, hy, rx, ry), m, tex=tex)
    if k == 'siam':
        cv.paint(cv.ell(hx, hy + ry * .15, rx * .6, ry * .65, soft=2.5), C(.25, .18, .14))
    if k in ('tux', 'calico') or (k not in ('black', 'siam', 'white') and chance(rs, .35)):
        cv.paint(cv.ell(hx, hy + ry * .5, rx * .5, ry * .45, soft=1.5), WHITE)                # white muzzle and chin
        if chance(rs, .5):
            cv.paint(cv.cap(hx, hy + ry * .3, hx, hy - ry * .5, rx * .12, rx * .05), WHITE)
    if k == 'calico':
        patches(cv, cv.ell(hx, hy, rx, ry), C(.8, .5, .22), 1, r * .3, r * .6, r)
        patches(cv, cv.ell(hx, hy, rx, ry), C(.1, .09, .08), 1, r * .3, r * .6, r)
    if stripe is not None and detail:
        for sg in (-1, 0, 1):
            cv.paint(cv.cap(hx + sg * rx * .2, hy - ry * .9, hx + sg * rx * .15, hy - ry * .45, r * .045), stripe)
        for sg in (-1, 1):
            cv.paint(cv.cap(hx + sg * rx * .95, hy + ry * .1, hx + sg * rx * .6, hy + ry * .2, r * .04), stripe)
    ex = rx * U(rs, .38, .46); ey = hy - ry * U(rs, .05, .18)
    for sg in (-1, 1):
        cv.paint(cv.ell(hx + sg * ex, ey, r * .2, r * .15), eye)
        cv.paint(cv.ell(hx + sg * ex, ey, r * U(rs, .05, .12), r * .14), BLACK)
    cv.paint(cv.poly([(hx - r * .09, hy + ry * .25), (hx + r * .09, hy + ry * .25), (hx, hy + ry * .4)]), jit(rs, C(.8, .5, .5), .03, .1) if chance(rs, .7) else BLACK)


def body_stripes(cv, mask, stripe, x0, x1, y0, y1, n, r=.02, lean=0.0):
    rs = cv.rs
    for x in np.linspace(x0, x1, n):
        x = x + U(rs, -.02, .02)
        cv.paint(cv.cap(x, y0, x + lean + U(rs, -.04, .04), y1, r) * mask, stripe)


def cat_face(cv):
    rs = cv.rs
    k, m, stripe, eye = cat_cols(rs)
    tex = cv.fur(.035, .05)
    by = U(rs, .6, .85)
    body = cv.ell(U(rs, -.2, .2), by, U(rs, .4, .65), .5)
    cv.paint(body, m, tex=tex)
    if stripe is not None:
        body_stripes(cv, body, stripe, -.5, .5, by - .4, by + .3, 7, .025, U(rs, -.1, .1))
    if k in ('tux', 'calico') or chance(rs, .25):
        cv.paint(cv.ell(0, by - .1, .2, .3, soft=2) * body, WHITE)
    cat_head(cv, 0, 0, U(rs, .3, .37), k, m, stripe, eye, tex)


def cat_body(cv, pose='lie'):
    """Whole cat facing +x (head turned to the camera): loaf, sitting, or standing with tail up."""
    rs = cv.rs
    k, m, stripe, eye = cat_cols(rs)
    tex = cv.fur(.035, .05)
    hr = U(rs, .17, .22)
    if pose == 'lie':
        ry = U(rs, .2, .3); gy = ry + U(rs, .0, .05)
        cv.paint(cv.cap(-.45, gy - .08, -.15 + U(rs, -.3, .2), gy + U(rs, -.02, .05), .05, .035), m if stripe is None else stripe)
        body = cv.ell(0, 0, .5, ry)
        cv.paint(body, m, tex=tex + .1 - .4 * cv.ramp(-ry * .5, ry))
        if stripe is not None:
            body_stripes(cv, body, stripe, -.4, .3, -ry, ry * .5, 7, .022, U(rs, -.08, .08))
        if k in ('tux', 'calico') or chance(rs, .2):
            cv.paint(cv.ell(.3, ry * .6, .25, ry * .5, soft=2) * body, WHITE)
        if k == 'calico':
            patches(cv, body, C(.8, .5, .22), 2, .1, .2, .4); patches(cv, body, BLACK, 2, .1, .2, .4)
        cv.paint(cv.ell(.32, gy - .04, .1, .045), WHITE if k in ('tux', 'calico') else m)
        hx, hy = .36 + U(rs, -.08, .08), -ry * .4 + U(rs, -.08, .1)
        top = -hy + hr * 1.6
    elif pose == 'sit':
        gy = U(rs, .58, .68)
        cv.paint(cv.cap(.15, gy - .04, .42 + U(rs, -.1, .1), gy - U(rs, 0, .1), .05, .035), m if stripe is None else stripe)
        body = cv.ell(0, gy - .4, U(rs, .25, .32), .42)
        cv.paint(body, m, tex=tex)
        if stripe is not None:
            body_stripes(cv, body, stripe, -.3, .3, gy - .6, gy - .1, 5, .022, .1)
        if k in ('tux', 'calico') or chance(rs, .25):
            cv.paint(cv.ell(.02, gy - .4, .12, .28, soft=2) * body, WHITE)
        for sg in (-1, 1):
            cv.paint(cv.cap(sg * .07, gy - .35, sg * .08, gy - .02, .05, .045), WHITE if k in ('tux', 'calico') else m * .95)
        hx, hy = U(rs, -.05, .05), gy - .88 + U(rs, -.04, .04)
        top = -hy + hr * 1.6
    else:                                                # standing / walking
        ry = U(rs, .17, .22); gy = U(rs, .5, .6); r = U(rs, .035, .045)
        sw = rs.uniform(-.1, .1, 4)
        leg(cv, .34, ry * .4, .34 + sw[0], gy - .03, r * .9, m * .6, m * .6); leg(cv, -.34, ry * .4, -.34 + sw[1], gy - .03, r * .9, m * .6, m * .6)
        tx, ty = -.62 + U(rs, -.1, .08), -U(rs, .2, .5)
        cv.paint(cv.cap(-.48, -.05, tx, ty, .04, .035), m if stripe is None else stripe)
        cv.paint(cv.cap(tx, ty, tx + U(rs, -.05, .15), ty - U(rs, .05, .2), .035, .03), m if stripe is None else stripe)
        body = cv.ell(0, 0, .5, ry)
        cv.paint(body, m, tex=tex - .2 * cv.ramp(-ry * .3, ry))
        if stripe is not None:
            body_stripes(cv, body, stripe, -.4, .35, -ry, ry * .6, 8, .02, U(rs, -.06, .06))
        fl = WHITE if k in ('tux', 'calico') else m
        leg(cv, .28, ry * .4, .28 + sw[2], gy, r, m, fl, tex=tex); leg(cv, -.4, ry * .4, -.4 + sw[3], gy, r, m, fl, knee=-.03, tex=tex)
        hx, hy = .5 + U(rs, -.05, .05), -.1 + U(rs, -.1, .08)
        top = -hy + hr * 1.6
    cat_head(cv, hx, hy, hr, k, m, stripe, eye, tex, detail=hr > .19)
    return gy, top


# ------------------------------------------------------------------ frog
FROGCOL = [C(.3, .6, .14), C(.42, .46, .2), C(.2, .36, .12), C(.45, .33, .2), C(.46, .42, .34), C(.68, .56, .38), C(.85, .5, .14),
           C(.6, .66, .25), C(.58, .3, .2), C(.5, .5, .47), C(.3, .24, .17)]
FROGP = [12, 16, 8, 17, 12, 11, 4, 6, 4, 5, 5]


def frog_cols(rs):
    b = jit(rs, pick(rs, FROGCOL, FROGP), .04, .15)
    belly = jit(rs, pick(rs, [C(.9, .88, .78), C(.85, .8, .5), C(.75, .72, .6), C(.9, .9, .88)]), .03, .08) if chance(rs, .75) else b
    spot = b * U(rs, .3, .6) if chance(rs, .8) else np.clip(b * 1.5 + .1, 0, 1)
    eyec = jit(rs, pick(rs, [C(.8, .65, .2), C(.75, .3, .1), C(.1, .1, .08), C(.6, .55, .3)]), .03, .1)
    return b, belly, spot, eyec


def frog_spots(cv, mask, spot, n, lim=.4):
    rs = cv.rs
    for _ in range(n):
        cv.paint(cv.ell(U(rs, -lim, lim), U(rs, -lim * .7, lim * .7), U(rs, .02, .07), U(rs, .02, .06), a=U(rs, -1, 1)) * mask, spot, op=U(rs, .5, 1))


def frog_side(cv):
    """Sitting frog facing +x: body tilted head-up, bulging eye on top, pale throat, folded hind leg."""
    rs = cv.rs
    b, belly, spot, eyec = frog_cols(rs)
    tex = cv.fur(.06, .1, 9)
    a = -U(rs, .1, .45)
    sh = tex + .15 - .5 * cv.ramp(-.25, .3)
    cv.paint(cv.cap(.2, .08, .3 + U(rs, -.04, .06), .34, .05, .038), b * .8, tex=tex)
    cv.paint(cv.cap(.3, .34, .42, .36, .03), b * .8)
    body = cv.ell(-.05, .05, .42, U(rs, .2, .27), a=a)
    cv.paint(body, b, tex=sh)
    hx, hy = .28 + U(rs, -.03, .04), -.1 + a * .25
    head = cv.ell(hx, hy, .23, .17, a=a * .6)
    cv.paint(head, b, tex=sh)
    if chance(rs, .6):
        cv.paint(cv.ell(hx - .06, hy + .17, .27, .12, a=a * .5, soft=2.5), lerp(b, belly, U(rs, .3, .9)))
    if chance(rs, .75):
        frog_spots(cv, np.maximum(body, head), spot, int(pick(rs, [3, 6, 10, 16])), .42)
    if chance(rs, .35):
        cv.paint(cv.cap(-.4, .0 - a * .2, hx + .1, hy - .05, .02) * np.maximum(body, head), spot)   # stripe along the flank
    cv.paint(cv.ell(-.27, .16, .25, .115, a=U(rs, .3, .7)), b, tex=tex - .1)
    cv.paint(cv.cap(-.12, .3, -.44, .33 + U(rs, -.02, .03), .05, .04), b * .9, tex=tex)
    cv.paint(cv.cap(-.44, .34, -.2 + U(rs, -.1, .1), .38, .03), b * .85)
    ex, ey = hx + U(rs, -.02, .05), hy - .14
    cv.paint(cv.ell(ex, ey, .085, .08), b, tex=tex)
    cv.paint(cv.ell(ex + .01, ey, .058, .055), eyec)
    cv.paint(cv.ell(ex + .012, ey, .04, .035), BLACK)
    cv.paint(cv.cap(hx + .2, hy + .02, hx - .05, hy + .06, .01), b * .5, op=.8)             # mouth line
    if chance(rs, .5):
        cv.paint(cv.ell(U(rs, -.15, .15), -.08 + a * .1, U(rs, .08, .2), .035, a=a, soft=2.5), C(1, 1, 1), op=U(rs, .2, .6))
    return .4


def frog_top(cv):
    """Seen from above: oval body, eyes bulging at the sides of the head, legs folded in a Z."""
    rs = cv.rs
    b, belly, spot, eyec = frog_cols(rs)
    tex = cv.fur(.06, .1, 9)
    k = U(rs, .7, 1.2)
    for sg in (-1, 1):
        cv.paint(cv.cap(sg * .12, .2, sg * .38 * k, .1 + U(rs, -.08, .08), .075, .06), b * .9, tex=tex)
        cv.paint(cv.cap(sg * .38 * k, .1, sg * .22, .42 * k, .05, .04), b * .85, tex=tex)
        cv.paint(cv.cap(sg * .22, .42 * k, sg * .42, .5 * k + .08, .03, .025), b * .8)
        cv.paint(cv.cap(sg * .14, -.12, sg * .34, -.2 + U(rs, -.05, .1), .045, .04), b * .9, tex=tex)
        cv.paint(cv.cap(sg * .34, -.2, sg * .38, -.36, .03), b * .8)
    body = cv.ell(0, .02, U(rs, .18, .23), .3)
    cv.paint(body, b, tex=tex)
    head = cv.ell(0, -.24, .17, .15)
    cv.paint(head, b, tex=tex)
    if chance(rs, .8):
        frog_spots(cv, np.maximum(body, head), spot, int(pick(rs, [4, 8, 14])), .3)
    if chance(rs, .3):
        for sg in (-1, 1):
            cv.paint(cv.cap(sg * .09, -.3, sg * .13, .25, .018) * np.maximum(body, head), spot)
    for sg in (-1, 1):
        cv.paint(cv.ell(sg * .12, -.27, .058, .058), b * 1.1)
        cv.paint(cv.ell(sg * .125, -.275, .035, .035), BLACK)


def frog_front(cv):
    """Facing the camera: wide low body, two eye bumps on top, wide mouth, pale belly, front legs."""
    rs = cv.rs
    b, belly, spot, eyec = frog_cols(rs)
    tex = cv.fur(.06, .1, 9)
    rx = U(rs, .38, .46); ry = U(rs, .24, .32)
    for sg in (-1, 1):
        cv.paint(cv.ell(sg * rx * .95, .18, .17, .12, a=sg * .5), b * .85, tex=tex)        # hind legs showing at the sides
    body = cv.ell(0, .02, rx, ry)
    cv.paint(body, b, tex=tex - .2 * cv.ramp(-ry, ry))
    if chance(rs, .6):
        cv.paint(cv.ell(0, .18, rx * U(rs, .45, .7), ry * U(rs, .35, .55), soft=3) * body, lerp(b, belly, U(rs, .3, .9)))
    if chance(rs, .7):
        frog_spots(cv, body * (cv.ly < .05), spot, int(pick(rs, [3, 6, 10])), .4)
    cv.paint(cv.cap(-rx * .75, U(rs, -.04, .04), rx * .75, U(rs, -.04, .04), .012), b * .4)
    for sg in (-1, 1):
        cv.paint(cv.cap(sg * .26, .12, sg * .34, .38, .05, .04), b * .9, tex=tex)
        cv.paint(cv.cap(sg * .34, .38, sg * .45, .42, .03), b * .85)
        ex, ey = sg * U(rs, .17, .23), -ry * .95
        cv.paint(cv.ell(ex, ey, .095, .09), b, tex=tex)
        cv.paint(cv.ell(ex, ey - .01, .06, .055), eyec)
        cv.paint(cv.ell(ex, ey - .01, .045, .03), BLACK)
    return .44


# ------------------------------------------------------------------ bird
BIRDCOL = [C(.45, .35, .23), C(.5, .5, .5), C(.14, .13, .12), C(.46, .46, .26), C(.2, .35, .7), C(.9, .9, .88), C(.85, .75, .2),
           C(.75, .2, .12), C(.6, .5, .4), C(.3, .3, .34), C(.35, .5, .3)]
BIRDP = [22, 14, 12, 9, 6, 8, 7, 4, 9, 6, 3]
BREAST = [C(.9, .88, .82), C(.85, .5, .2), C(.88, .8, .3), C(.75, .22, .15), C(.7, .68, .64), C(.8, .7, .55)]
BEAK = [C(.15, .14, .13), C(.9, .7, .15), C(.9, .5, .12), C(.5, .45, .4), C(.8, .75, .65)]


def bird_perch(cv, perch='branch'):
    """Small perching bird facing +x: tilted oval body, round head, short beak, long tail, thin legs."""
    rs = cv.rs
    b = jit(rs, pick(rs, BIRDCOL, BIRDP), .03, .12)
    wc = b * U(rs, .5, .9) if chance(rs, .8) else jit(rs, pick(rs, BIRDCOL, BIRDP), .03, .12)
    br = jit(rs, pick(rs, BREAST), .03, .08) if chance(rs, .65) else b
    hc = b if chance(rs, .6) else jit(rs, pick(rs, BIRDCOL + [C(.1, .1, .1), C(.8, .2, .1)]), .03, .1)
    tex = cv.fur(.03, .05, 9)
    a = -U(rs, .35, 1.1)
    ca, sa = np.cos(a), np.sin(a)
    yp = U(rs, .3, .4)
    pc = jit(rs, pick(rs, [C(.4, .3, .2), C(.3, .24, .18), C(.55, .5, .45), C(.25, .2, .15), C(.5, .42, .3)]), .03, .15)
    if perch == 'branch':
        y0, y1 = yp + U(rs, -.25, .25), yp + U(rs, -.25, .25)
        r = U(rs, .025, .06)
        cv.paint(cv.cap(-2, 2 * y0 - (y0 + y1) / 2 + r, 2, 2 * y1 - (y0 + y1) / 2 + r, r), pc, tex=.12 * noise(rs, 10, 5))
        if chance(rs, .4):
            cv.paint(cv.cap(U(rs, -.8, .8), yp, U(rs, -1, 1), yp - U(rs, .3, 1), r * .5), pc)
    elif perch == 'post':
        w = U(rs, .1, .25)
        cv.paint(cv.box(.03, yp + 1.0, w, 2.0, r=.01), pc, tex=.1 * noise(rs, 5, 14))
    elif perch == 'rock':
        cv.paint(cv.ell(U(rs, -.2, .2), yp + .42, U(rs, .5, .9), .42), jit(rs, pick(rs, GREYS + EARTH), .03, .15), tex=.12 * noise(rs, 10) + .06 * noise(rs, 16))
    elif perch == 'wire':
        cv.paint(cv.cap(-2, yp + .01, 2, yp + U(rs, -.05, .05), .012), C(.2, .2, .2))
    rx, ry = .3, U(rs, .16, .21)
    tx, ty = -(rx - .04) * ca, -(rx - .04) * sa             # rear end of the body
    at = a + U(rs, -.5, .25)
    dx, dy = -np.cos(at), -np.sin(at); nx, ny = -dy, dx
    tl = U(rs, .25, .45)
    cv.paint(cv.poly([(tx + nx * .045, ty + ny * .045), (tx - nx * .045, ty - ny * .045), (tx + dx * tl - nx * .03, ty + dy * tl - ny * .03), (tx + dx * tl + nx * .03, ty + dy * tl + ny * .03)]), wc)
    for o in (-.03, .05):
        cv.paint(cv.cap(o, ry * .6, o + U(rs, -.02, .03), yp, .012), jit(rs, pick(rs, [C(.3, .25, .2), C(.7, .5, .4), C(.15, .15, .15), C(.8, .6, .2)]), .03, .1))
    body = cv.ell(0, 0, rx, ry, a=a)
    cv.paint(body, b, tex=tex)
    cv.paint(cv.ell(-sa * .07, ca * .07, rx * .85, ry * .62, a=a, soft=1.5) * body, br, tex=tex)       # breast and belly
    cv.paint(cv.ell(sa * .05 - ca * .05, -ca * .05 - sa * .05, rx * .72, ry * .55, a=a + .12), wc, tex=tex)   # folded wing
    if chance(rs, .3):
        cv.paint(cv.cap(sa * .05 - ca * .2, -ca * .05 - sa * .2, sa * .05 + ca * .05, -ca * .05 + sa * .05, .015), np.clip(wc * 1.6 + .2, 0, 1))  # wing bar
    hr = U(rs, .11, .14)
    hx, hy = (rx - .03) * ca + .02, (rx - .03) * sa - .05
    cv.paint(cv.ell(hx, hy, hr, hr), hc, tex=tex)
    if chance(rs, .3):
        cv.paint(cv.ell(hx + .03, hy + hr * .5, hr * .7, hr * .5) * cv.ell(hx, hy, hr, hr), br)         # pale throat or cheek
    bl = U(rs, .08, .2); bk = jit(rs, pick(rs, BEAK), .03, .1); by = hy + U(rs, -.02, .02)
    cv.paint(cv.poly([(hx + hr * .8, by - .035), (hx + hr * .8, by + .035), (hx + hr * .8 + bl, by + U(rs, -.01, .04))]), bk)
    cv.paint(cv.ell(hx + hr * .35, hy - hr * .2, .024, .024), BLACK)
    return yp, -hy + hr


def bird_head(cv):
    """Ostrich or emu portrait: long neck from below, small flat head, big eye, broad flat beak."""
    rs = cv.rs
    b = jit(rs, pick(rs, [C(.6, .52, .46), C(.5, .46, .42), C(.75, .62, .58), C(.72, .7, .68), C(.26, .23, .2), C(.4, .36, .3), C(.2, .4, .75)],
                     [22, 20, 12, 12, 14, 14, 6]), .03, .12)
    tex = cv.fur(.06, .07, 10)
    nr = U(rs, .09, .15)
    cv.paint(cv.cap(U(rs, -.35, .25), 1.3, -.02, .0, nr * 1.3, nr, soft=1.8), b, tex=tex)
    a = U(rs, -.25, .2)
    bk = jit(rs, pick(rs, [C(.75, .6, .45), C(.85, .6, .5), C(.3, .27, .25), C(.85, .7, .3), C(.55, .5, .45)]), .03, .1)
    bx, by = .2 * np.cos(a), -.1 + .2 * np.sin(a)
    cv.paint(cv.poly([(bx - .02, by - .075), (bx - .02, by + .06), (bx + U(rs, .28, .4), by + .04 + U(rs, -.04, .06)), (bx + .3, by - .03)]), bk)
    cv.paint(cv.cap(bx, by + .01, bx + .3, by + .02, .008), bk * .5)
    hc = b if chance(rs, .75) else b * U(rs, .5, .8)
    cv.paint(cv.ell(.05, -.1, .22, .15, a=a, soft=1.6), hc, tex=tex)
    cv.paint(cv.ell(.1, -.13, .062, .062), np.clip(b * 1.3 + .1, 0, 1))
    cv.paint(cv.ell(.1, -.13, .045, .045), BLACK)


def bird_big(cv):
    """Large standing bird (ostrich, emu, crane, heron): bulky feathered body, long thin neck, long bare legs."""
    rs = cv.rs
    b = jit(rs, pick(rs, [C(.14, .13, .12), C(.35, .3, .25), C(.5, .48, .46), C(.9, .9, .88), C(.5, .4, .3), C(.9, .6, .6), C(.3, .3, .34)],
                     [24, 22, 16, 12, 14, 4, 8]), .03, .12)
    nc = b if chance(rs, .5) else jit(rs, pick(rs, [C(.75, .6, .58), C(.6, .55, .5), C(.9, .9, .88), C(.2, .2, .2)]), .03, .1)
    lc = jit(rs, pick(rs, [C(.7, .58, .55), C(.45, .4, .36), C(.2, .2, .2), C(.85, .65, .25)]), .03, .1)
    tex = cv.fur(.07, .09, 9)
    gy = U(rs, .75, .92); r = U(rs, .022, .032)
    sw = rs.uniform(-.12, .12, 2)
    leg(cv, .02, .15, .02 + sw[0], gy - .02, r, lc * .7, lc * .7, knee=-.03)
    leg(cv, -.05, .15, -.05 + sw[1], gy, r, lc, lc, knee=-.03)
    cv.paint(cv.ell(-.3, .03, .18, .14, a=.5, soft=2.2), b * .9 if chance(rs, .7) else np.clip(b * 1.5 + .3, 0, 1), tex=tex)     # tail plumes
    cv.paint(cv.ell(0, 0, U(rs, .3, .36), U(rs, .2, .26), a=U(rs, -.25, .1), soft=2.0), b, tex=tex)
    n1x, n1y = .3 + U(rs, -.04, .1), -.42 + U(rs, -.05, .1)
    hx, hy = n1x + U(rs, -.08, .12), -.82 + U(rs, 0, .2)
    if chance(rs, .2):                                   # head down, feeding
        n1x, n1y, hx, hy = .5, .25, .55, .6
    cv.paint(cv.cap(.24, -.08, n1x, n1y, .05, .035), nc); cv.paint(cv.cap(n1x, n1y, hx, hy, .035, .028), nc)
    cv.paint(cv.ell(hx + .02, hy, .07, .05), nc)
    cv.paint(cv.poly([(hx + .07, hy - .025), (hx + .07, hy + .03), (hx + U(rs, .16, .3), hy + .02)]), jit(rs, pick(rs, BEAK), .03, .1))
    cv.paint(cv.ell(hx + .03, hy - .01, .016, .016), BLACK)
    return gy, max(-hy + .06, .3)


def bird_fly(cv):
    """Flying bird seen from below or above: slim body, two long pointed wings, fanned tail."""
    rs = cv.rs
    b = jit(rs, pick(rs, [C(.1, .1, .1), C(.3, .26, .22), C(.9, .9, .9), C(.55, .55, .56), C(.45, .35, .25)], [36, 22, 18, 12, 12]), .03, .12)
    span = U(rs, .5, .75); sw = U(rs, -.2, .1); up = U(rs, -.25, .25)
    for sg in (-1, 1):
        mx, my = -.02 + sw * .4 + sg * up * .0, sg * span * .5
        cv.paint(cv.poly([(.12, sg * .03), (-.14, sg * .03), (mx - .1, my), (mx + .1, my)]), b * U(rs, .85, 1.0))
        cv.paint(cv.poly([(mx - .1, my), (mx + .1, my), (sw + U(rs, -.1, .05), sg * span)]), b * U(rs, .7, 1.0))
    cv.paint(cv.poly([(-.25, -.03), (-.25, .03), (-.48, U(rs, .06, .12)), (-.48, -U(rs, .06, .12))]), b * .9)
    cv.paint(cv.cap(-.28, 0, .26, 0, .06, .05), b)
    cv.paint(cv.ell(.3, 0, .065, .055), b if chance(rs, .7) else np.clip(b * 1.5 + .3, 0, 1))
    cv.paint(cv.poly([(.35, -.02), (.35, .02), (.46, 0)]), jit(rs, pick(rs, BEAK), .03, .1))


def bird_swim(cv):
    """Duck, goose or swan on water: boat-shaped body on the waterline y = 0, neck up, small head, flat bill."""
    rs = cv.rs
    b = jit(rs, pick(rs, [C(.92, .92, .9), C(.45, .36, .26), C(.5, .5, .5), C(.15, .14, .13), C(.6, .5, .4)], [28, 30, 14, 12, 16]), .03, .1)
    hc = b if chance(rs, .65) else jit(rs, pick(rs, [C(.1, .35, .2), C(.1, .1, .1), C(.35, .2, .12), C(.9, .9, .9)]), .03, .1)
    tex = cv.fur(.04, .06, 9)
    cv.paint(cv.ell(0, .1, .42, .07, soft=3), C(0, 0, 0), op=U(rs, .15, .4))                 # reflection
    body = cv.ell(0, -.1, .42, U(rs, .15, .2)) * np.clip((-cv.ly) * cv.px + .5, 0, 1)
    cv.paint(body, b, tex=tex)
    cv.paint(cv.poly([(-.3, -.15), (-.42, -.05), (-.56, -.26)]), b * .9)
    if chance(rs, .5):
        cv.paint(cv.ell(-.05, -.14, .26, .09, a=.1) * body, b * U(rs, .6, .9), tex=tex)      # wing
    nh = U(rs, .3, .6)
    nx = .3 + U(rs, -.03, .08)
    cv.paint(cv.cap(.27, -.14, nx, -nh, .065, .05), hc if chance(rs, .5) else b)
    cv.paint(cv.ell(nx + .03, -nh - .02, .085, .07), hc)
    cv.paint(cv.cap(nx + .1, -nh - .01, nx + .24, -nh + .02, .03, .02), jit(rs, pick(rs, BEAK[1:3] + [BEAK[0]]), .03, .1))
    cv.paint(cv.ell(nx + .05, -nh - .04, .015, .015), BLACK)
    return nh + .1

# ---------------------------------------------------------------------------------------------
# PART 4. Scenes: where each class is usually photographed, how big it is in the frame, and where
# the horizon falls (the horizon is at camera height: above a car roof, half-way up a truck,
# through a horse's back, far above a pet or a frog).
# ---------------------------------------------------------------------------------------------

TWO_PI = 2 * np.pi


def land(cv, gkind, yh, mid=None, midh=0.1, sky_kind=None, slope=None, wat=False):
    """Sky, an optional band of far things standing on the horizon, then ground (or water) below it."""
    rs = cv.rs
    sky(cv, sky_kind)
    slope = U(rs, -.05, .05) if slope is None else slope
    if mid == 'trees':
        treeline(cv, yh - midh, yh + .04)
    elif mid == 'hills':
        hills(cv, yh - midh, yh + .04)
    elif mid == 'build':
        buildings(cv, yh - midh, yh + .04)
    elif mid == 'wall':
        col = jit(rs, pick(rs, WALLS + [C(.5, .3, .2), C(.3, .4, .3), C(.9, .9, .9)]), .03, .15)
        cv.paint(band_mask(yh - midh, yh + .04), col, tex=.04 * noise(rs, 10, 3) + .05 * noise(rs, 3))
    elif mid == 'bokeh':
        keep = cv.im.copy()
        foliage(cv)
        m = band_mask(yh - midh + edge1d(rs, 5, .05), yh + .04)[..., None]
        cv.im = keep * (1 - m) + cv.im * m
    if wat:
        return water(cv, yh, slope=slope * .3)
    return ground(cv, gkind, yh, slope)


def gshadow(cv, x, y, rx, ry=.035, op=None):
    rs = cv.rs
    cv.paint(cv.ell(x, y, rx, ry, soft=2.5), C(0, 0, 0), op=U(rs, .2, .55) if op is None else op)


def flipsign(rs):
    return 1 if chance(rs, .5) else -1


def person(cv, x, yfoot, h):
    """A standing person: legs, coloured torso, head. (x, yfoot) in image units, h = height in image units."""
    rs = cv.rs
    cv.frame(x, yfoot, h)
    shirt = jit(rs, pick(rs, [C(.92, .92, .92), C(.75, .1, .1), C(.15, .25, .65), C(.08, .08, .1), C(.2, .5, .25), C(.9, .8, .2), C(.5, .5, .55), C(.85, .5, .6)]), .03, .1)
    pants = jit(rs, pick(rs, [C(.2, .25, .45), C(.12, .12, .15), C(.75, .7, .6), C(.4, .3, .2)]), .03, .1)
    cv.paint(cv.cap(-.04, -.5, -.05, 0, .06, .05), pants); cv.paint(cv.cap(.04, -.5, .06, 0, .06, .05), pants)
    cv.paint(cv.cap(0, -.52, 0, -.8, .11, .12), shirt)
    cv.paint(cv.ell(0, -.93, .065, .075), jit(rs, C(.85, .68, .55), .03, .1))
    if chance(rs, .5):
        cv.paint(cv.ell(0, -.97, .068, .045), jit(rs, pick(rs, [C(.1, .08, .06), C(.4, .28, .15), C(.8, .7, .4)]), .02, .2))


def street_things(cv, yh):
    """Things that stand behind a vehicle in a street: parked cars, poles, signs."""
    rs = cv.rs
    cv.frame(.5, .5, 1.0)
    if chance(rs, .3):
        for _ in range(int(pick(rs, [1, 2]))):
            x = U(rs, -.5, .5); w = U(rs, .2, .4); h = w * U(rs, .3, .45)
            col = jit(rs, pick(rs, CARCOL, CARP), .03, .12)
            y = yh - .5 + U(rs, .0, .04)
            cv.paint(cv.box(x, y - h / 2, w, h, r=.02), col)
            cv.paint(cv.box(x, y - h * .75, w * .6, h * .3), GLASS)
    if chance(rs, .25):
        for _ in range(int(pick(rs, [1, 2]))):
            x = U(rs, -.5, .5)
            cv.paint(cv.cap(x, yh - .5 + .02, x, -.6, U(rs, .006, .015)), jit(rs, pick(rs, [C(.3, .3, .3), C(.5, .4, .3), C(.7, .7, .7)]), .02, .2))
            if chance(rs, .4):
                cv.paint(cv.box(x, yh - .5 - U(rs, .15, .4), U(rs, .05, .12), U(rs, .04, .1)), jit(rs, pick(rs, FABRIC), .03, .1))


# ------------------------------------------------------------------ 0 airplane
def gen_airplane(rs):
    cv = Cv(rs)
    scene = pick(rs, ['air', 'ground', 'aerial'], [58, 32, 10])
    fl = flipsign(rs)
    if scene == 'air':
        sky(cv)
        v = pick(rs, ['side', 'plan', 'fighter', 'prop'], [42, 28, 18, 12])
        cx, cy = .5 + U(rs, -.08, .08), .5 + U(rs, -.1, .1)
        if v == 'side':
            sx = U(rs, .6, 1.0)
            cv.frame(cx, cy, U(rs, .78, 1.1) / sx ** .7, rs.normal(0, .2), fl, sx=sx); airplane_side(cv, gear=chance(rs, .2))
        elif v == 'prop':
            sx = U(rs, .7, 1.0)
            cv.frame(cx, cy, U(rs, .72, 1.05) / sx ** .7, rs.normal(0, .25), fl, sx=sx); airplane_prop(cv, gear=chance(rs, .6))
        elif v == 'plan':
            cv.frame(cx, cy, U(rs, .75, 1.1), U(rs, 0, TWO_PI), fl, sy=U(rs, .4, 1.0)); airplane_plan(cv)
        else:
            cv.frame(cx, cy, U(rs, .75, 1.1), U(rs, 0, TWO_PI), fl, sy=U(rs, .4, 1.0)); airplane_fighter(cv)
    elif scene == 'ground':
        yh = U(rs, .58, .86)
        land(cv, pick(rs, ['road', 'grass', 'concrete', 'dirt', 'dry', 'snow'], [34, 30, 14, 8, 9, 5]), yh,
             mid=pick(rs, [None, 'trees', 'hills', 'build'], [45, 25, 15, 15]), midh=U(rs, .04, .16))
        v = pick(rs, ['side', 'prop', 'fighter'], [66, 24, 10])
        sx = U(rs, .6, 1.0)
        sc = U(rs, .8, 1.08) / sx ** .7
        flying = chance(rs, .3)
        yg = min(yh + U(rs, .02, .12), .97)
        if v == 'side':
            cy = yh - U(rs, .12, .4) if flying else yg - .17 * sc
            if not flying:
                cv.frame(.5, yg, sc, sx=sx); gshadow(cv, 0, 0, .45, .02)
            cv.frame(.5 + U(rs, -.06, .06), cy, sc, U(rs, -.3, .08) if flying else U(rs, -.04, .04), fl, sx=sx); airplane_side(cv, gear=True)
        elif v == 'prop':
            cy = yh - U(rs, .12, .4) if flying else yg - .275 * sc
            if not flying:
                cv.frame(.5, yg, sc); gshadow(cv, 0, 0, .4, .02)
            cv.frame(.5 + U(rs, -.06, .06), cy, sc * .9, U(rs, -.3, .08) if flying else U(rs, -.12, .02), fl); airplane_prop(cv, gear=True)
        else:
            cv.frame(.5, yg, sc); gshadow(cv, 0, 0, .4, .02)
            cv.frame(.5 + U(rs, -.06, .06), yg - .08 * sc, sc, U(rs, -.05, .05), fl, sy=U(rs, .25, .45)); airplane_fighter(cv)
    else:                                                # seen from above, over land or sea
        k = pick(rs, ['fields', 'sea', 'hills'], [50, 25, 25])
        if k == 'fields':
            bokeh(cv, [pick(rs, FOLIAGE), pick(rs, FOLIAGE), pick(rs, EARTH), pick(rs, FOLIAGE)], g=int(pick(rs, [3, 5, 7])))
        elif k == 'sea':
            water(cv, -.3)
        else:
            land(cv, 'grass', U(rs, .5, .8), mid='hills', midh=U(rs, .2, .5))
        v = pick(rs, ['side', 'plan', 'fighter', 'prop'], [30, 35, 20, 15])
        cx, cy = .5 + U(rs, -.08, .08), .5 + U(rs, -.1, .1)
        if v == 'side':
            cv.frame(cx, cy, U(rs, .6, 1.0), rs.normal(0, .2), fl); airplane_side(cv)
        elif v == 'prop':
            cv.frame(cx, cy, U(rs, .55, .95), rs.normal(0, .25), fl); airplane_prop(cv, gear=False)
        elif v == 'plan':
            cv.frame(cx, cy, U(rs, .6, 1.0), U(rs, 0, TWO_PI), fl, sy=U(rs, .5, 1.0)); airplane_plan(cv)
        else:
            cv.frame(cx, cy, U(rs, .55, .95), U(rs, 0, TWO_PI), fl, sy=U(rs, .5, 1.0)); airplane_fighter(cv)
    return photo(cv.im, rs)


# ------------------------------------------------------------------ 8 ship
def gen_ship(rs):
    cv = Cv(rs)
    scene = pick(rs, ['sea', 'above', 'harbour'], [72, 15, 13])
    fl = flipsign(rs)
    if scene == 'above':
        water(cv, -U(rs, .1, .5))
        if chance(rs, .6):
            cv.frame(.5 + U(rs, -.08, .08), .5 + U(rs, -.08, .08), U(rs, .65, 1.0), U(rs, -.7, .7), fl, sy=U(rs, .55, .9)); boat_above(cv)
        else:
            sx = U(rs, .5, .9); sc = U(rs, .8, 1.08) / sx
            cv.frame(.5 + U(rs, -.06, .06), .6 + U(rs, -.05, .15), sc, U(rs, -.12, .12), fl, sx=sx)
            ship_side(cv, pick(rs, ['cargo', 'cruise', 'fish', 'war', 'speed'], [25, 15, 25, 10, 25])); ship_wake(cv)
    else:
        yw = U(rs, .62, .92)                                 # where the hull meets the water
        yh = min(U(rs, .2, .62), yw - .06)
        if scene == 'harbour':
            mid, midh = pick(rs, ['build', 'trees', 'hills'], [60, 20, 20]), U(rs, .08, .25)
        else:
            mid, midh = pick(rs, [None, 'hills', 'trees', 'build'], [58, 20, 14, 8]), U(rs, .025, .12)
        wc = land(cv, None, yh, mid=mid, midh=midh, wat=True, slope=U(rs, -.03, .03))
        kind = pick(rs, ['cargo', 'cruise', 'fish', 'war', 'sail', 'speed'], [26, 15, 21, 8, 16, 14])
        sx = U(rs, .5, .9) if kind != 'sail' else 1.0                       # hull seen partly end-on looks shorter and taller
        sc = U(rs, .5, .78) if kind == 'sail' else U(rs, .8, 1.08) / sx
        if kind == 'sail':
            yw = max(yw, 1.0 * sc + .02)
        cv.frame(.5 + U(rs, -.1, .1), yw, sc, U(rs, -.04, .04), fl, sx=sx)
        cv.paint(cv.box(0, .06, .9, .1, soft=4), wc * .5, op=U(rs, .1, .35))       # reflection of the hull
        ship_side(cv, kind); ship_wake(cv)
        if scene == 'harbour' and chance(rs, .5):
            ground(cv, pick(rs, ['concrete', 'road', 'dirt']), U(rs, .86, .95), U(rs, -.1, .1))
    return photo(cv.im, rs)


# ------------------------------------------------------------------ 1 automobile
def gen_car(rs):
    cv = Cv(rs)
    view = pick(rs, ['side', 'q', 'front'], [30, 42, 28])
    scene = pick(rs, ['street', 'plain', 'nature'], [52, 30, 18])
    sx = U(rs, .6, .85) if view == 'q' else 1.0
    sc = {'side': U(rs, .9, 1.15), 'q': min(U(rs, .9, 1.1) / sx, 1.55), 'front': U(rs, .95, 1.3)}[view]
    bot = {'side': .16, 'q': .2, 'front': .28}[view]
    yg = U(rs, .72, .93)
    cy = yg - bot * sc
    yh = float(np.clip(cy - U(rs, -.05, .3) * sc, .12, .62))
    if scene == 'street':
        land(cv, pick(rs, ['road', 'concrete', 'dirt', 'grass', 'snow'], [62, 12, 9, 12, 5]), yh,
             mid=pick(rs, ['build', 'trees', 'wall', None, 'hills'], [35, 35, 10, 12, 8]), midh=U(rs, .15, .6))
        street_things(cv, yh)
        if chance(rs, .15):
            cv.frame(.5, yg + U(rs, .03, .1), 1.0); cv.paint(cv.box(0, 0, 2, U(rs, .012, .025), a=U(rs, -.1, .1)), C(.9, .9, .85), op=.8)
    elif scene == 'plain':
        col = plain(cv)
        if chance(rs, .55):
            cv.paint(below(yh + U(rs, 0, .15)), np.clip(col * U(rs, .6, 1.15), 0, 1), tex=.03 * noise(rs, 3))
    else:
        foliage(cv)
        ground(cv, pick(rs, ['grass', 'dirt', 'road', 'dry', 'dgrass'], [35, 20, 25, 10, 10]), yh)
    cv.frame(.5 + U(rs, -.07, .07), cy, sc, U(rs, -.06, .06), flipsign(rs), sx=sx)
    {'side': car_side, 'q': car_34, 'front': car_front}[view](cv)
    return photo(cv.im, rs)


# ------------------------------------------------------------------ 9 truck
def gen_truck(rs):
    cv = Cv(rs)
    view = pick(rs, ['side', 'q', 'front'], [34, 42, 24])
    scene = pick(rs, ['road', 'plain', 'site'], [52, 28, 20])
    sc = {'side': U(rs, .9, 1.15), 'q': U(rs, .92, 1.2), 'front': U(rs, .85, 1.15)}[view]
    bot = {'side': .3, 'q': .32, 'front': .43}[view]
    yg = U(rs, .8, .96)
    cy = yg - bot * sc
    yh = float(np.clip(yg - U(rs, .08, .32), .38, .8))
    if scene == 'road':
        land(cv, pick(rs, ['road', 'concrete', 'grass', 'snow'], [70, 14, 11, 5]), yh,
             mid=pick(rs, ['trees', 'build', None, 'hills', 'wall'], [32, 25, 25, 10, 8]), midh=U(rs, .08, .35))
        street_things(cv, yh)
    elif scene == 'plain':
        col = plain(cv)
        if chance(rs, .5):
            cv.paint(below(yh + U(rs, 0, .1)), np.clip(col * U(rs, .6, 1.15), 0, 1), tex=.03 * noise(rs, 3))
    else:                                                # yard, quarry, building site
        land(cv, pick(rs, ['dirt', 'sand', 'soil', 'dry'], [45, 20, 20, 15]), yh,
             mid=pick(rs, ['trees', 'hills', None, 'build'], [35, 25, 25, 15]), midh=U(rs, .08, .3))
    cv.frame(.5 + U(rs, -.06, .06), cy, sc, U(rs, -.05, .05), flipsign(rs))
    {'side': truck_side, 'q': truck_34, 'front': truck_front}[view](cv)
    return photo(cv.im, rs)


# ------------------------------------------------------------------ 2 bird
def soft_backdrop(cv, probs):
    """Blurred backdrop behind a small close subject."""
    rs = cv.rs
    k = pick(rs, ['foliage', 'sky', 'muted', 'earth', 'plain'], probs)
    if k == 'foliage':
        foliage(cv)
    elif k == 'sky':
        sky(cv)
    elif k == 'muted':
        bokeh(cv, [pick(rs, GREYS + EARTH + FOLIAGE), pick(rs, GREYS + EARTH + FOLIAGE), pick(rs, GREYS + FOLIAGE)], g=int(pick(rs, [2, 3])), sharp=1.5)
    elif k == 'earth':
        bokeh(cv, [pick(rs, EARTH), pick(rs, EARTH), pick(rs, FOLIAGE)], g=int(pick(rs, [2, 3, 4])))
    else:
        plain(cv)
    return k


def gen_bird(rs):
    cv = Cv(rs)
    v = pick(rs, ['perch', 'head', 'big', 'fly', 'swim', 'ground'], [38, 12, 16, 9, 9, 16])
    fl = flipsign(rs)
    if v == 'perch':
        k = soft_backdrop(cv, [44, 22, 18, 8, 8])
        if k in ('foliage', 'earth', 'muted') and chance(rs, .5):
            strokes(cv, int(pick(rs, [1, 2, 4, 7])), [C(.3, .22, .15), C(.2, .15, .1), C(.5, .42, .32), C(.25, .35, .15)], .006, .025)
        sc = U(rs, .62, 1.05)
        cv.frame(.5 + U(rs, -.1, .1) - .03 * fl, .47 + U(rs, -.1, .08), sc, U(rs, -.15, .15), fl)
        bird_perch(cv, pick(rs, ['branch', 'post', 'rock', 'wire', 'none'], [46, 16, 16, 7, 15]))
    elif v == 'ground':
        k = pick(rs, ['litter', 'grass', 'other', 'far'], [30, 30, 20, 20])
        sc = U(rs, .5, .92); cy = .5 + U(rs, -.05, .12)
        if k == 'litter':
            litter(cv)
        elif k == 'grass':
            cv.im[:] = .3; ground(cv, pick(rs, ['grass', 'dgrass', 'dry']), -.3)
        elif k == 'other':
            cv.im[:] = .5; ground(cv, pick(rs, ['snow', 'sand', 'dirt', 'concrete', 'road']), -.3)
        else:
            foliage(cv); ground(cv, pick(rs, ['grass', 'dirt', 'dry', 'snow', 'sand']), U(rs, .15, .5))
        cv.frame(.5 + U(rs, -.1, .1), cy, sc, U(rs, -.1, .1), fl)
        gshadow(cv, 0, .37, .3, .05)
        bird_perch(cv, 'none')
    elif v == 'head':
        soft_backdrop(cv, [45, 15, 22, 12, 6])
        cv.frame(.45 + U(rs, -.1, .1) + .05 * (1 - fl) / 2, .45 + U(rs, -.1, .12), U(rs, .85, 1.3), U(rs, -.2, .2), fl)
        bird_head(cv)
    elif v == 'big':
        sc = U(rs, .58, .85); yg = U(rs, .84, .97)
        yh = float(np.clip(yg - U(rs, .2, .7), .1, .7))
        land(cv, pick(rs, ['grass', 'dirt', 'dry', 'sand', 'dgrass', 'snow'], [34, 20, 18, 12, 10, 6]), yh,
             mid=pick(rs, ['bokeh', 'trees', None, 'hills'], [45, 25, 20, 10]), midh=U(rs, .15, .8))
        cv.frame(.5 + U(rs, -.1, .1) - .06 * fl, yg, sc); gshadow(cv, 0, 0, .3, .03)
        cv.frame(.5 + U(rs, -.1, .1) - .06 * fl, yg - .85 * sc, sc, U(rs, -.05, .05), fl)
        bird_big(cv)
    elif v == 'fly':
        if chance(rs, .8):
            sky(cv)
        else:
            soft_backdrop(cv, [50, 0, 30, 10, 10])
        cv.frame(.5 + U(rs, -.1, .1), .5 + U(rs, -.1, .1), U(rs, .55, 1.05), U(rs, 0, TWO_PI), fl, sy=U(rs, .4, 1.0))
        bird_fly(cv)
    else:
        yh = U(rs, -.4, .3)
        if yh > 0:
            land(cv, None, yh, mid=pick(rs, ['bokeh', 'trees', None]), midh=U(rs, .1, .4), wat=True)
        else:
            water(cv, yh, pick(rs, ['blue', 'grey', 'teal', 'dark', 'deepblue'], [22, 28, 20, 20, 10]))
        cv.frame(.5 + U(rs, -.1, .1), U(rs, .58, .8), U(rs, .6, 1.0), U(rs, -.04, .04), fl)
        bird_swim(cv)
    return photo(cv.im, rs)


# ------------------------------------------------------------------ 3 cat, 5 dog
def pet_bg(cv, probs):
    rs = cv.rs
    k = pick(rs, ['indoor', 'plain', 'grass', 'yard', 'garden', 'dark'], probs)
    if k == 'indoor':
        indoor(cv)
    elif k == 'plain':
        plain(cv)
    elif k == 'grass':
        if chance(rs, .6):
            cv.im[:] = .3; ground(cv, pick(rs, ['grass', 'dgrass', 'grass', 'dry']), -.3)
        else:
            foliage(cv); ground(cv, pick(rs, ['grass', 'dgrass']), U(rs, .1, .45))
    elif k == 'yard':
        if chance(rs, .5):
            cv.im[:] = .5; ground(cv, pick(rs, ['concrete', 'road', 'snow', 'sand', 'dirt']), -.3)
        else:
            soft_backdrop(cv, [40, 5, 35, 10, 10]); ground(cv, pick(rs, ['concrete', 'road', 'snow', 'sand', 'dirt']), U(rs, .15, .5))
    elif k == 'garden':
        foliage(cv)
    else:
        plain(cv, C(.06, .055, .05) + cv.rs.uniform(0, .1))
    return k


def gen_cat(rs):
    cv = Cv(rs)
    v = pick(rs, ['face', 'lie', 'sit', 'stand'], [40, 26, 20, 14])
    pet_bg(cv, [62, 12, 10, 4, 6, 6])
    fl = flipsign(rs)
    if v == 'face':
        cv.frame(.5 + U(rs, -.1, .1), .45 + U(rs, -.1, .1), U(rs, .9, 1.45), U(rs, -.35, .35), fl); cat_face(cv)
    elif v == 'lie':
        sc = U(rs, .78, 1.12)
        cv.frame(.5 + U(rs, -.08, .08), .52 + U(rs, -.06, .12), sc, U(rs, -.2, .2), fl)
        gshadow(cv, 0, .28, .5, .06, op=U(rs, .1, .4)); cat_body(cv, 'lie')
    elif v == 'sit':
        sc = U(rs, .66, .95)
        cv.frame(.5 + U(rs, -.1, .1), .5 + U(rs, -.05, .08), sc, U(rs, -.1, .1), fl)
        gshadow(cv, 0, .64, .35, .05, op=U(rs, .1, .4)); cat_body(cv, 'sit')
    else:
        sc = U(rs, .62, .9)
        cv.frame(.5 + U(rs, -.08, .08) - .04 * fl, .5 + U(rs, -.02, .12), sc, U(rs, -.08, .08), fl)
        gshadow(cv, 0, .56, .5, .04, op=U(rs, .1, .4)); cat_body(cv, 'stand')
    return photo(cv.im, rs)


def gen_dog(rs):
    cv = Cv(rs)
    v = pick(rs, ['face', 'stand', 'sit', 'lie'], [40, 22, 24, 14])
    pet_bg(cv, [36, 12, 30, 12, 6, 4])
    fl = flipsign(rs)
    if v == 'face':
        cv.frame(.5 + U(rs, -.1, .1), .43 + U(rs, -.1, .1), U(rs, .9, 1.4), U(rs, -.3, .3), fl); dog_face(cv)
    elif v == 'stand':
        sc = U(rs, .62, .9)
        cv.frame(.5 + U(rs, -.08, .08) - .06 * fl, .47 + U(rs, -.04, .1), sc, U(rs, -.08, .08), fl)
        gshadow(cv, 0, .68, .5, .04, op=U(rs, .1, .4)); dog_body(cv, 'stand')
    elif v == 'sit':
        sc = U(rs, .68, .98)
        cv.frame(.5 + U(rs, -.08, .08), .42 + U(rs, -.04, .08), sc, U(rs, -.08, .08), fl)
        gshadow(cv, 0, .76, .4, .04, op=U(rs, .1, .4)); dog_body(cv, 'sit')
    else:
        sc = U(rs, .7, 1.0)
        cv.frame(.5 + U(rs, -.08, .08), .5 + U(rs, -.04, .12), sc, U(rs, -.1, .1), fl)
        gshadow(cv, 0, .45, .55, .05, op=U(rs, .1, .4)); dog_body(cv, 'lie')
    return photo(cv.im, rs)


# ------------------------------------------------------------------ 4 deer
def trunks(cv, n):
    rs = cv.rs
    cv.frame(.5, .5, 1.0)
    for _ in range(n):
        x = U(rs, -.5, .5)
        cv.paint(cv.box(x, U(rs, -.3, 0), U(rs, .04, .12), 1.2, a=U(rs, -.06, .06)), jit(rs, pick(rs, [C(.2, .15, .1), C(.3, .25, .2), C(.12, .1, .08), C(.4, .35, .3)]), .03, .2), tex=.12 * noise(rs, 5, 14))


def wild_bg(cv, yg, probs):
    """Woodland, meadow with trees behind, dry field, or snow."""
    rs = cv.rs
    bg = pick(rs, ['forest', 'meadow', 'dry', 'snow'], probs)
    yh = float(np.clip(yg - U(rs, .08, .45), .15, .85))
    if bg == 'forest':
        foliage(cv, dark=U(rs, .65, 1.2))
        if chance(rs, .4):
            trunks(cv, int(pick(rs, [1, 2, 3])))
        if chance(rs, .6):
            strokes(cv, int(pick(rs, [2, 4, 8, 14])), [C(.25, .18, .12), C(.12, .1, .08), C(.45, .38, .3), C(.6, .55, .45), C(.2, .3, .12)], .006, .02)
        ground(cv, pick(rs, ['dgrass', 'grass', 'soil', 'dry'], [35, 30, 20, 15]), float(np.clip(yg - U(rs, .03, .3), .3, .9)))
    elif bg == 'meadow':
        land(cv, pick(rs, ['grass', 'dgrass', 'dry'], [55, 25, 20]), yh, mid=pick(rs, ['trees', 'bokeh', 'hills', None], [50, 30, 10, 10]), midh=U(rs, .15, .9))
    elif bg == 'dry':
        bokeh(cv, [pick(rs, EARTH), C(.6, .5, .3), pick(rs, FOLIAGE), pick(rs, EARTH)], g=int(pick(rs, [3, 4, 6])))
        ground(cv, pick(rs, ['dry', 'dirt', 'soil']), yh)
    else:
        if chance(rs, .5):
            foliage(cv, dark=U(rs, .5, .9))
            if chance(rs, .5):
                trunks(cv, int(pick(rs, [1, 2, 3])))
        else:
            land(cv, 'snow', yh, mid=pick(rs, ['trees', 'hills']), midh=U(rs, .15, .6), sky_kind=pick(rs, ['over', 'white', 'pale']))
        ground(cv, 'snow', yh)
    return bg


def gen_deer(rs):
    cv = Cv(rs)
    v = pick(rs, ['side', 'front', 'bust', 'graze', 'lie'], [50, 19, 9, 12, 10])
    fl = flipsign(rs)
    if v == 'bust':
        soft_backdrop(cv, [62, 6, 14, 14, 4])
        cv.frame(.5 + U(rs, -.1, .1), .55 + U(rs, -.08, .1), U(rs, .8, 1.2), U(rs, -.15, .15), fl); deer_bust(cv)
        return photo(cv.im, rs)
    sc = U(rs, .48, .74) if v != 'front' else U(rs, .52, .8)
    yg = U(rs, .8, .96)
    bg = wild_bg(cv, yg, [34, 38, 20, 8])
    cx = .5 + U(rs, -.1, .1) - (.06 * fl if v != 'front' else 0)
    if chance(rs, .15):                                  # another of the herd, farther away
        s2 = sc * U(rs, .5, .8); x2 = cx + U(rs, .25, .45) * (1 if chance(rs, .5) else -1); y2 = yg - U(rs, .03, .15)
        cv.frame(x2, y2 - .88 * s2, s2, 0.0, flipsign(rs)); deer_side(cv, pick(rs, ['stand', 'graze']))
    if v == 'lie':
        yg = U(rs, .6, .85)
        cv.frame(cx, yg - .3 * sc, sc, U(rs, -.04, .04), fl); deer_side(cv, 'lie')
    else:
        cv.frame(cx, yg, sc); gshadow(cv, 0, 0, .4 if v != 'front' else .2, .03)
        cv.frame(cx, yg - .88 * sc, sc, U(rs, -.05, .05), fl)
        if v == 'front':
            deer_front(cv)
        else:
            deer_side(cv, 'graze' if v == 'graze' else 'stand')
    if bg != 'snow' and chance(rs, .35):                 # grass and stalks in front of the legs
        n = int(pick(rs, [4, 8, 14]))
        cols = [C(.3, .45, .15), C(.5, .6, .25), C(.6, .55, .3), C(.2, .3, .1), C(.7, .65, .4)]
        cv.frame(.5, .5, 1.0)
        for _ in range(n):
            x = U(rs, -.5, .5); h = U(rs, .08, .3)
            cv.paint(cv.cap(x, .5, x + U(rs, -.06, .06), yg - .5 - h + .1, U(rs, .006, .014)), jit(rs, pick(rs, cols), .03, .2), op=U(rs, .5, 1))
    if bg in ('forest', 'meadow') and chance(rs, .5):
        dapple(cv)
    return photo(cv.im, rs)


# ------------------------------------------------------------------ 7 horse
def horse_bg(cv, yg, sc):
    rs = cv.rs
    bg = pick(rs, ['pasture', 'arena', 'plain', 'woods'], [52, 26, 7, 15])
    yh = float(np.clip(yg - U(rs, .15, .8) * sc, .25, .78))
    if bg == 'pasture':
        land(cv, pick(rs, ['grass', 'dgrass', 'dry'], [66, 17, 17]), yh, mid=pick(rs, ['trees', 'hills', None, 'bokeh'], [50, 10, 28, 12]), midh=U(rs, .08, .45))
        if chance(rs, .35):
            fence(cv, yh - U(rs, .0, .1))
    elif bg == 'arena':
        land(cv, pick(rs, ['dirt', 'sand', 'soil', 'dry'], [45, 30, 10, 15]), yh, mid=pick(rs, ['wall', 'trees', 'build', 'bokeh', None], [25, 30, 15, 15, 15]), midh=U(rs, .1, .5))
        if chance(rs, .55):
            fence(cv, yh - U(rs, .0, .12))
    elif bg == 'plain':
        plain(cv, None if chance(rs, .5) else C(.07, .06, .05) + rs.uniform(0, .1))
        if chance(rs, .5):
            ground(cv, pick(rs, ['dirt', 'sand', 'concrete', 'dry']), yh)
    else:
        foliage(cv)
        ground(cv, pick(rs, ['grass', 'dgrass', 'dirt', 'dry'], [40, 25, 20, 15]), yh)


def gen_horse(rs):
    cv = Cv(rs)
    v = pick(rs, ['side', 'head', 'front', 'graze'], [56, 17, 15, 12])
    fl = flipsign(rs)
    if v == 'head':
        k = pick(rs, ['sky', 'soft', 'field', 'dark'], [22, 40, 26, 12])
        if k == 'sky':
            sky(cv)
        elif k == 'soft':
            soft_backdrop(cv, [55, 0, 20, 15, 10])
        elif k == 'field':
            land(cv, pick(rs, ['grass', 'dry', 'dirt']), U(rs, .5, .85), mid=pick(rs, ['trees', None, 'bokeh']), midh=U(rs, .1, .4))
        else:
            plain(cv, C(.07, .06, .05) + rs.uniform(0, .12))
        cv.frame(.48 + U(rs, -.1, .1) - .03 * fl, .5 + U(rs, -.1, .1), U(rs, .85, 1.25), U(rs, -.2, .2), fl); horse_head(cv)
        return photo(cv.im, rs)
    rid = chance(rs, .28)
    sc = U(rs, .54, .78) if v != 'front' else U(rs, .56, .8)
    if rid:
        sc *= .85
    yg = U(rs, .84, .97)
    horse_bg(cv, yg, sc)
    cx = .5 + U(rs, -.06, .06) - (.12 * sc * fl if v != 'front' else 0)
    if chance(rs, .14):                                  # handler or onlooker standing by
        person(cv, cx + U(rs, .3, .48) * (1 if chance(rs, .5) else -1), yg - U(rs, 0, .06), sc * U(rs, 1.0, 1.25))
    cv.frame(cx, yg, sc); gshadow(cv, 0, 0, .45 if v != 'front' else .25, .035)
    cv.frame(cx, yg - .89 * sc, sc, U(rs, -.05, .05), fl)
    if v == 'front':
        horse_front(cv)
    else:
        horse_side(cv, graze=(v == 'graze'), with_rider=rid and v == 'side')
    if chance(rs, .25):
        dapple(cv, U(rs, .08, .25))
    return photo(cv.im, rs)


# ------------------------------------------------------------------ 6 frog
def gen_frog(rs):
    cv = Cv(rs)
    v = pick(rs, ['side', 'top', 'front'], [46, 24, 30])
    bg = pick(rs, ['leaf', 'litter', 'rock', 'pond', 'white', 'dark', 'hand', 'wood'], [22, 27, 10, 9, 10, 10, 4, 8])
    if bg == 'leaf':
        if chance(rs, .5):
            bokeh(cv, [pick(rs, FOLIAGE[:5]), pick(rs, FOLIAGE[:5]), pick(rs, FOLIAGE)], g=int(pick(rs, [2, 3, 4])))
        else:
            foliage(cv)
        if chance(rs, .4):                               # one big leaf under the frog
            cv.frame(.5 + U(rs, -.2, .2), .6 + U(rs, -.1, .2), 1.0, U(rs, -.6, .6))
            lc = jit(rs, pick(rs, FOLIAGE[1:4]), .03, .15)
            cv.paint(cv.ell(0, 0, U(rs, .5, .9), U(rs, .25, .45)), lc, tex=.06 * noise(rs, 5))
            cv.paint(cv.cap(-.9, 0, .9, 0, .01), lc * 1.3 + .05, op=.7)
    elif bg == 'litter':
        litter(cv)
    elif bg == 'rock':
        litter(cv, [GREYS[0], GREYS[2], GREYS[1], GREYS[4], EARTH[4]])
    elif bg == 'pond':
        cv.im = np.ones((S, S, 3)) * jit(rs, pick(rs, [C(.12, .2, .12), C(.2, .22, .15), C(.1, .15, .2), C(.3, .32, .25), C(.05, .08, .06)]), .03, .2)
        cv.im = cv.im * (1 + .12 * noise(rs, 14, 3) + .1 * noise(rs, 4))[..., None]
        cv.frame(.5, .5, 1.0)
        for _ in range(int(pick(rs, [0, 1, 2, 3]))):
            cv.paint(cv.ell(U(rs, -.5, .5), U(rs, -.5, .5), U(rs, .1, .3), U(rs, .06, .18), a=U(rs, -.3, .3)), jit(rs, pick(rs, FOLIAGE[1:4]), .03, .15))
    elif bg == 'white':
        plain(cv, C(.95, .95, .95))
    elif bg == 'dark':
        plain(cv, C(.05, .05, .05) + rs.uniform(0, .08))
    elif bg == 'hand':
        soft_backdrop(cv, [40, 5, 30, 15, 10])
        cv.frame(.5, .5, 1.0)
        sk = jit(rs, pick(rs, [C(.85, .65, .55), C(.75, .55, .45), C(.6, .42, .32)]), .03, .1)
        cv.paint(cv.ell(U(rs, -.1, .1), U(rs, .1, .3), U(rs, .4, .6), U(rs, .25, .4), a=U(rs, -.4, .4), soft=2), sk, tex=.05 * noise(rs, 5))
    else:
        cv.im = np.ones((S, S, 3)) * jit(rs, pick(rs, EARTH), .03, .2)
        cv.im = cv.im * (1 + .14 * noise(rs, 5, 14) + .08 * noise(rs, 3))[..., None]
    if bg in ('leaf', 'litter', 'pond', 'wood') and chance(rs, .4):
        strokes(cv, int(pick(rs, [2, 4, 8])), [C(.3, .22, .15), C(.5, .42, .3), C(.25, .4, .15), C(.6, .6, .35), C(.15, .12, .1)], .006, .022)
    sc = U(rs, .72, 1.2)
    cx, cy = .5 + U(rs, -.08, .08), .5 + U(rs, -.1, .1)
    fl = flipsign(rs)
    if v == 'side':
        cv.frame(cx, cy, sc, U(rs, -.2, .2), fl); gshadow(cv, -.05, .36, .5, .07); frog_side(cv)
    elif v == 'top':
        ang = U(rs, 0, TWO_PI)
        cv.frame(cx + .015, cy + .02, sc * .95, ang, fl); gshadow(cv, 0, .05, .3, .36)
        cv.frame(cx, cy, sc * .95, ang, fl); frog_top(cv)
    else:
        cv.frame(cx, cy - .03, sc, U(rs, -.15, .15), fl); gshadow(cv, 0, .38, .5, .07); frog_front(cv)
    if bg not in ('white', 'dark') and chance(rs, .25):
        dapple(cv, U(rs, .08, .3))
    return photo(cv.im, rs)


GENS = [gen_airplane, gen_car, gen_bird, gen_cat, gen_deer, gen_dog, gen_frog, gen_horse, gen_ship, gen_truck]

# ---------------------------------------------------------------------------------------------
# PART 5. The network. Front end: weights from formulas (colour lattice, oriented edge filters).
# Back end: one dense layer whose rows are the front-end descriptors of the drawn scenes (each
# drawing is its own template, normalised by itself alone by the same norm layers), followed by a
# soft nearest-template vote per class.
# ---------------------------------------------------------------------------------------------

NORM_EPS = 1e-5


# ---- interpreter with the same semantics as the scorer's blocks, used to describe the drawings
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
    """Runs the layers and returns {name: value} for the names in want."""
    vals = {"image": images}
    for L in layers:
        ins = [vals[i] for i in L["inputs"]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = _conv(ins[0], weights[name + ".W"], weights[name + ".b"], L.get("stride", 1), L.get("padding", 0))
        elif op == "dense":
            vals[name] = ins[0].reshape(len(ins[0]), -1) @ weights[name + ".W"].T + weights[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op in ("maxpool", "avgpool"):
            vals[name] = _pool(ins[0], L["size"], L["stride"], op[:3])
        elif op == "add":
            vals[name] = sum(ins)
        elif op == "reshape":
            vals[name] = ins[0].reshape((len(ins[0]),) + tuple(L["shape"]))
        elif op == "norm":
            x = ins[0]
            mu = x.mean(axis=(2, 3), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + NORM_EPS)
    return {k: vals[k] for k in want}


# ---- colour lattice: soft colour categories = cones around nodes in (lightness, red-green, yellow-blue)
L_LEVELS = [0.12, 0.45, 0.82]          # dark, middle, bright
R_L = 0.45                             # lightness tolerance (wide: exposure varies a lot between photos)
HUES = [30, 60, 90, 125, 165, 215, 250, 310]   # red, orange/brown, yellow/olive, grass green, green, cyan/teal, blue, purple/pink
RINGS = [(0.14, 0.14), (0.38, 0.38)]   # (chroma radius of the ring, tolerance): muted colours and vivid colours
R_GREY = 0.10
M_OPP = np.array([[1 / 3, 1 / 3, 1 / 3],
                  [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                  [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)]])


def colour_nodes():
    ch = [(0.0, 0.0, R_GREY)]
    for rad, tol in RINGS:
        for h in HUES:
            ch.append((rad * np.cos(np.deg2rad(h)), rad * np.sin(np.deg2rad(h)), tol))
    return ch


# concave squashing of pooled amounts (so that a small amount of a colour or an edge still counts):
# f(x) = s0*x - (s0-s1)*relu(x-t1) - (s1-s2)*relu(x-t2)
SQ_COL = (4.0, 1.5, 0.6, 0.05, 0.25)
SQ_EDGE = (3.0, 1.5, 0.6, 0.10, 0.40)

ORIENT = 6


def edge_bank(ksize, su, sv):
    """Oriented first-derivative-of-Gaussian filters (edge detectors), both polarities; unit step edge -> response 1."""
    r = ksize // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    bank = []
    for k in range(ORIENT):
        th = np.pi * k / ORIENT
        u = xx * np.cos(th) + yy * np.sin(th)
        v = -xx * np.sin(th) + yy * np.cos(th)
        g = -u * np.exp(-u * u / (2 * su * su) - v * v / (2 * sv * sv))
        g = g - g.mean()
        g = g / np.abs(g).sum() * 2.0
        bank.append(g); bank.append(-g)
    return np.array(bank)[:, None, :, :]


GROUPS = [  # name, weight in the similarity. Colour 2.0 in all; shape 2.0 in all, coarse scales weighted more than
    # fine ones because fine texture is where drawings differ most from photographs.
    ('c4', 1.0), ('c2', 0.6), ('c1', 0.4), ('e1', 0.3), ('e2', 0.6), ('e3', 0.6), ('g8', 0.5)]


def build_frontend():
    layers, W = [], {}

    def add(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=inputs, **kw)); return name

    def squash(prefix, src, ch, sq):
        s0, s1, s2, t1, t2 = sq
        a = np.zeros((3 * ch, ch, 1, 1)); b = np.zeros(3 * ch); c = np.zeros((ch, 3 * ch, 1, 1))
        for i in range(ch):
            a[3 * i:3 * i + 3, i, 0, 0] = 1.0
            b[3 * i + 1], b[3 * i + 2] = -t1, -t2
            c[i, 3 * i:3 * i + 3, 0, 0] = [s0, -(s0 - s1), -(s1 - s2)]
        add(prefix + 'a', 'conv', [src], out=3 * ch, kernel=[1, 1], stride=1, padding=0); W[prefix + 'a.W'], W[prefix + 'a.b'] = a, b
        add(prefix + 'r', 'relu', [prefix + 'a'])
        add(prefix + 's', 'conv', [prefix + 'r'], out=ch, kernel=[1, 1], stride=1, padding=0); W[prefix + 's.W'], W[prefix + 's.b'] = c, np.zeros(ch)
        return prefix + 's'

    # ----- colour path (on the 2x2-averaged image)
    add('cp', 'avgpool', ['image'], size=[2, 2], stride=[2, 2])
    nodes = colour_nodes()
    nl, nc = len(L_LEVELS), len(nodes)
    m = 2 * (nl + 2 * nc)
    A = np.zeros((m, 3, 1, 1)); bA = np.zeros(m)
    for i, lv in enumerate(L_LEVELS):                       # |L - level| as two rectified halves
        A[2 * i, :, 0, 0] = M_OPP[0]; bA[2 * i] = -lv
        A[2 * i + 1, :, 0, 0] = -M_OPP[0]; bA[2 * i + 1] = lv
    for j, (a1, a2, tol) in enumerate(nodes):               # |c1 - a1| and |c2 - a2|
        o = 2 * nl + 4 * j
        A[o, :, 0, 0] = M_OPP[1]; bA[o] = -a1
        A[o + 1, :, 0, 0] = -M_OPP[1]; bA[o + 1] = a1
        A[o + 2, :, 0, 0] = M_OPP[2]; bA[o + 2] = -a2
        A[o + 3, :, 0, 0] = -M_OPP[2]; bA[o + 3] = a2
    add('ca', 'conv', ['cp'], out=m, kernel=[1, 1], stride=1, padding=0); W['ca.W'], W['ca.b'] = A, bA
    add('car', 'relu', ['ca'])
    K = nl * nc
    B = np.zeros((K, m, 1, 1)); bB = np.ones(K)
    for i in range(nl):
        for j, (a1, a2, tol) in enumerate(nodes):
            k = i * nc + j
            B[k, 2 * i:2 * i + 2, 0, 0] = -1.0 / R_L
            B[k, 2 * nl + 4 * j:2 * nl + 4 * j + 4, 0, 0] = -1.0 / tol
    add('cb', 'conv', ['car'], out=K, kernel=[1, 1], stride=1, padding=0); W['cb.W'], W['cb.b'] = B, bB
    add('cbr', 'relu', ['cb'])
    feats = []
    for name, size, cells in (('c4', 4, 16), ('c2', 8, 4), ('c1', 16, 1)):
        add(name + 'p', 'avgpool', ['cbr'], size=[size, size], stride=[size, size])
        s = squash(name + 'q', name + 'p', K, SQ_COL)
        add(name + 'v', 'reshape', [s], shape=[1, K * cells, 1])
        add(name + 'n', 'norm', [name + 'v'])
        feats.append((name, K * cells))

    # ----- edge path (on brightness, normalised over the whole picture)
    add('el', 'conv', ['image'], out=1, kernel=[1, 1], stride=1, padding=0)
    W['el.W'] = np.array([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1); W['el.b'] = np.zeros(1)
    add('en', 'norm', ['el'])
    pair = np.zeros((ORIENT, 2 * ORIENT, 1, 1))
    for k in range(ORIENT):
        pair[k, 2 * k:2 * k + 2, 0, 0] = 1.0
    for name, down, ks, su, sv, pool in (('e1', 1, 9, 1.0, 2.0, 8), ('e2', 2, 9, 1.0, 2.0, 4), ('e3', 4, 5, 0.8, 1.2, 2)):
        src = 'en'
        if down > 1:
            src = add(name + 'd', 'avgpool', ['en'], size=[down, down], stride=[down, down])
        add(name + 'c', 'conv', [src], out=2 * ORIENT, kernel=[ks, ks], stride=1, padding=ks // 2)
        W[name + 'c.W'] = edge_bank(ks, su, sv); W[name + 'c.b'] = np.zeros(2 * ORIENT)
        add(name + 'cr', 'relu', [name + 'c'])
        add(name + 'm', 'conv', [name + 'cr'], out=ORIENT, kernel=[1, 1], stride=1, padding=0); W[name + 'm.W'], W[name + 'm.b'] = pair, np.zeros(ORIENT)
        add(name + 'p', 'avgpool', [name + 'm'], size=[pool, pool], stride=[pool, pool])
        s = squash(name + 'q', name + 'p', ORIENT, SQ_EDGE)
        add(name + 'v', 'reshape', [s], shape=[1, ORIENT * 16, 1])
        add(name + 'n', 'norm', [name + 'v'])
        feats.append((name, ORIENT * 16))

    # ----- coarse layout of light and dark: the normalised brightness picture shrunk to 8 x 8
    add('g8p', 'avgpool', ['en'], size=[4, 4], stride=[4, 4])
    add('g8v', 'reshape', ['g8p'], shape=[1, 64, 1])
    add('g8n', 'norm', ['g8v'])
    feats.append(('g8', 64))

    # ----- put the groups side by side in one descriptor. Each group is scaled by sqrt(weight / size), so the dot
    # product of two descriptors is the weighted sum of the groups' correlations.
    D = sum(d for _, d in feats)
    wts = dict(GROUPS)
    off = 0
    for name, d in feats:
        E = np.zeros((D, d)); E[off:off + d, :] = np.eye(d) * np.sqrt(wts[name] / d)
        add(name + 'e', 'dense', [name + 'n'], out=D); W[name + 'e.W'], W[name + 'e.b'] = E, np.zeros(D)
        off += d
    add('desc', 'add', [name + 'e' for name, _ in feats])
    return layers, W, D


# Soft nearest-template vote. Similarities of one picture to all templates are standardised (norm layer); a template
# votes for its class with weight relu(z - theta). For a standard normal z the mean of relu(z - theta) is 0.0833, 0.0085,
# 0.00038 at theta = 1, 2, 3; weighting the three thresholds by 1, 10, 200 makes the broad, middle and nearest
# neighbourhoods count about equally.
THETAS = [(1.0, 1.0), (2.0, 10.0), (3.0, 200.0)]


def build_backend(layers, W, templates, n_per_class, classes=10):
    """templates: (N, D) descriptors of the drawings, ordered by class."""
    N = n_per_class * classes
    assert templates.shape[0] == N
    nt = len(THETAS)

    def add(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=inputs, **kw)); return name
    add('sim', 'dense', ['desc'], out=N); W['sim.W'], W['sim.b'] = templates, np.zeros(N)
    add('simv', 'reshape', ['sim'], shape=[1, N, 1])
    add('z', 'norm', ['simv'])
    add('th', 'conv', ['z'], out=nt, kernel=[1, 1], stride=1, padding=0)
    W['th.W'] = np.ones((nt, 1, 1, 1)); W['th.b'] = -np.array([t for t, _ in THETAS], float)
    add('thr', 'relu', ['th'])
    add('grp', 'reshape', ['thr'], shape=[nt * classes, n_per_class, 1])
    add('vote', 'avgpool', ['grp'], size=[n_per_class, 1], stride=[n_per_class, 1])
    O = np.zeros((classes, nt * classes))
    for b, (_, wt) in enumerate(THETAS):
        for c in range(classes):
            O[c, b * classes + c] = wt
    add('out', 'dense', ['vote'], out=classes); W['out.W'], W['out.b'] = O, np.zeros(classes)
    return layers, W


# ---------------------------------------------------------------------------------------------
# PART 6. Build: draw the templates, describe each with the front end, write the files.
# ---------------------------------------------------------------------------------------------
def draw_templates(n_per_class, seed=SEED):
    ims = np.zeros((10 * n_per_class, 3, S, S), np.float32)
    for c in range(10):
        rs = np.random.RandomState(seed + c)
        for i in range(n_per_class):
            ims[c * n_per_class + i] = GENS[c](rs).transpose(2, 0, 1)
    return ims


def describe(layers, W, ims, bs=500):
    # double precision, so that the rebuilt numbers agree to far better than the 1e-5 tolerance on any machine
    return np.concatenate([run_layers(layers, W, ims[i:i + bs].astype(np.float64), ['desc'])['desc'].astype(np.float32)
                           for i in range(0, len(ims), bs)])


def main(out):
    layers, W, D = build_frontend()
    ims = draw_templates(N_PER_CLASS)
    T = describe(layers, W, ims)          # each row: the descriptor of one drawing, normalised by itself alone
    layers, W = build_backend(layers, W, T, N_PER_CLASS)
    with open(out + '/network.json', 'w') as f:
        json.dump({"layers": layers}, f)
    np.savez(out + '/weights.npz', **{k: np.asarray(v, np.float32) for k, v in W.items()})


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
