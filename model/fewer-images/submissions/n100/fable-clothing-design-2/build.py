"""build.py - writes network.json and weights.npz for a hand-designed clothing classifier.

Usage: python3 build.py OUT_DIR        (reads no files; every number comes from this source)

How the weights are made (no training, no fitting, no data):
  1. SHAPE MODELS.  For each of the 10 classes a small drawing program, typed from general knowledge of how
     these items are cut and photographed (a tee is a torso with two short sleeves hanging at the sides, a
     pullover has a wide flat neckline and cuffs that hang below its ribbed hem, a coat has a narrow stand-up
     collar, trousers are two legs joined above a crotch, a boot is a toe box plus a shaft over the heel, a
     sandal is a sole with straps, a bag is a body with a handle arch, ...).  Random numbers with a fixed seed
     pick the proportions inside hand-chosen ranges.  Each drawing is trimmed to its own bounding box, scaled
     so its longest side is 28 px and centred - the framing the task describes.
  2. TEMPLATES.  Each drawing becomes one template on its own: its silhouette is blurred with a fixed Gaussian
     and stored as one row of the dense layer `match` (weights 2*w*t, bias -sum(w*t^2), so the unit's output is
     minus the weighted squared distance to that drawing, up to a term common to all units; w is a fixed row
     weight that makes the neckline rows count more).  No statistic is taken across drawings.
  3. NETWORK.  image -> ink mask (1x1 convs + relu: min(1, k*(x - t0))) -> Gaussian blur conv -> `match`
     -> max over the templates of each class (reshape + maxpool) = class score.
     Two small hand-wired side branches add evidence a silhouette cannot carry:
       * front line: a thin vertical line down the whole middle of the front (zip / button placket) adds a
         bonus to coat and shirt (min over four row segments, max over column placement and polarity);
       * openwork: background trapped in narrow horizontal gaps (morphological closing built from max-pooling)
         on a tall item adds a bonus to sandal (strappy heeled sandal versus boot).
     All thresholds and bonuses are constants chosen by hand (see the block of constants further down).
"""
import json, os, sys
import numpy as np

# ----------------------------------------------------------------- drawing toolkit

SS = 4                      # supersampling
G = 28 * SS
_c = (np.arange(G) + 0.5) / SS
PX, PY = np.meshgrid(_c, _c)
DEG = np.pi / 180.0


def fill_poly(mask, pts, val=True):
    inside = np.zeros((G, G), bool)
    n = len(pts)
    for i in range(n):
        a, b = pts[i]
        c, d = pts[(i + 1) % n]
        if b == d:
            continue
        lo, hi = (b, d) if b < d else (d, b)
        r0 = max(int(np.ceil(lo * SS - 0.5)), 0)
        r1 = min(int(np.ceil(hi * SS - 0.5)), G)
        if r1 <= r0:
            continue
        yy = _c[r0:r1]
        xint = a + (yy - b) * (c - a) / (d - b)
        inside[r0:r1] ^= _c[None, :] < xint[:, None]
    if val:
        mask |= inside
    else:
        mask &= ~inside


def stroke(mask, pts, w, val=True):
    r = w / 2.0
    for (a, b), (c, d) in zip(pts[:-1], pts[1:]):
        x0 = max(int(np.floor((min(a, c) - r) * SS)), 0)
        x1 = min(int(np.ceil((max(a, c) + r) * SS)) + 1, G)
        y0 = max(int(np.floor((min(b, d) - r) * SS)), 0)
        y1 = min(int(np.ceil((max(b, d) + r) * SS)) + 1, G)
        if x1 <= x0 or y1 <= y0:
            continue
        X = PX[y0:y1, x0:x1]
        Y = PY[y0:y1, x0:x1]
        dx, dy = c - a, d - b
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            t = np.zeros_like(X)
        else:
            t = np.clip(((X - a) * dx + (Y - b) * dy) / L2, 0, 1)
        hit = (X - a - t * dx) ** 2 + (Y - b - t * dy) ** 2 <= r * r
        if val:
            mask[y0:y1, x0:x1] |= hit
        else:
            mask[y0:y1, x0:x1] &= ~hit


class Shape:
    """A drawing: list of (kind, points, width, add?) in the item's own units (x right, y down)."""

    def __init__(self):
        self.prims = []

    def poly(self, pts, add=True):
        self.prims.append(("p", np.asarray(pts, float), 0.0, add))
        return self

    def line(self, pts, w, add=True):
        self.prims.append(("s", np.asarray(pts, float), float(w), add))
        return self

    def ellipse(self, cx, cy, rx, ry, add=True, n=28):
        t = np.arange(n) * 2 * np.pi / n
        return self.poly(np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1), add)

    def mirror(self):
        """add the left-right mirror image of everything drawn so far (about x = 0)"""
        for k, p, w, add in list(self.prims):
            q = p.copy()
            q[:, 0] = -q[:, 0]
            self.prims.append((k, q[::-1].copy(), w, add))
        return self

    def render(self, min_w=0.0):
        lo = np.array([np.inf, np.inf])
        hi = -lo
        for k, p, w, add in self.prims:
            if add:
                lo = np.minimum(lo, p.min(0) - w / 2)
                hi = np.maximum(hi, p.max(0) + w / 2)
        size = hi - lo
        s = 28.0 / size.max()
        off = (28.0 - size * s) / 2.0 - lo * s
        mask = np.zeros((G, G), bool)
        for k, p, w, add in self.prims:
            q = p * s + off
            if k == "p":
                fill_poly(mask, q, add)
            else:
                stroke(mask, q, max(w * s, min_w), add)
        return mask.reshape(28, SS, 28, SS).mean((1, 3))


def arc(cx, cy, rx, ry, a0, a1, n=14):
    """points on an ellipse arc; angles in degrees, 0 = +x, 90 = +y (down)"""
    t = np.linspace(a0, a1, n) * DEG
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def bez(p0, p1, p2, n=8):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t * t * p2


# ----------------------------------------------------------------- class shape models


def U(rng, a, b):
    return a + (b - a) * rng.random()


# ---------------------------------------------------------------- upper-body garments
def garment(bw, nw, nd, sd, sw, ah, hw, hc=0.0, ww=None, wy=0.6, sleeve=None, collar=None, hood=None,
            armhole=0.0, length=1.0, vneck=None):
    """Front view, y down, centre x = 0, neck/shoulder point at y = 0, hem at y = length.
    bw chest half-width (armpit at height ah), nw neck half-width, nd neck dip, sd shoulder drop,
    sw shoulder half-width, hw hem half-width, hc extra drop of a curved hem at the centre,
    sleeve = (angle from vertical in degrees, length along outer edge, cuff width),
    collar = (base half-width, top half-width, height), vneck = (half-width, depth) of an open V at the throat."""
    sh = Shape()
    hy = length - hc
    right = [(nw, 0.0), (sw, sd)]
    if armhole > 0:                                   # cut-in armhole of a sleeveless top
        right.append((sw - armhole, sd + 0.55 * (ah - sd)))
    right.append((bw, ah))
    if ww is not None:
        right.append((ww, wy))
    right.append((hw, hy))
    if hc > 0:
        right += [(hw * 0.7, hy + hc * 0.6), (hw * 0.35, hy + hc * 0.93)]
    left = [(-x, y) for x, y in right[::-1]]
    sh.poly([(0.0, nd), (nw * 0.55, nd * 0.85)] + right + [(0.0, hy + hc)] + left + [(-nw * 0.55, nd * 0.85)])
    if sleeve is not None:
        th, L, c = sleeve
        th *= DEG
        S = np.array([sw, sd]); A = np.array([bw, ah])
        d = np.array([np.sin(th), np.cos(th)]); n = np.array([-np.cos(th), np.sin(th)])
        P1 = S + L * d
        P2 = P1 + c * n
        for sgn in (1, -1):
            sh.poly(np.array([S, P1, P2, A]) * [sgn, 1])
    top = 0.0
    if collar is not None:
        cw, ct, ch = collar
        sh.poly([(-cw, 0.02), (-ct, -ch), (ct, -ch), (cw, 0.02)])
        top = -ch
    if hood is not None:
        hr, hh = hood
        sh.poly(np.concatenate([[(hr, 0.04)], arc(0.0, 0.0, hr, hh, 0, -180, 14), [(-hr, 0.04)]]))
        top = -hh
    if vneck is not None:
        vw, vd = vneck
        sh.poly([(-vw, top - 0.01), (vw, top - 0.01), (0.0, vd)], add=False)
    return sh


def tshirt(rng):
    r = rng.random()
    vneck = (U(rng, 0.05, 0.1), U(rng, 0.06, 0.16)) if rng.random() < 0.15 else None
    if r < 0.70:                                       # classic tee: short sleeves hanging down at the sides
        bw = U(rng, 0.23, 0.32)
        return garment(bw=bw, nw=U(rng, 0.11, 0.18), nd=U(rng, 0.0, 0.05), sd=U(rng, 0.03, 0.12),
                       sw=bw + U(rng, -0.01, 0.04), ah=U(rng, 0.26, 0.38), hw=bw * U(rng, 0.92, 1.08),
                       hc=U(rng, 0.0, 0.03), vneck=vneck,
                       sleeve=(U(rng, 15, 55), U(rng, 0.14, 0.36), U(rng, 0.12, 0.2)))
    if r < 0.84:                                       # fitted top with cap sleeves
        bw = U(rng, 0.2, 0.29)
        return garment(bw=bw, nw=U(rng, 0.10, 0.17), nd=U(rng, 0.02, 0.12), sd=U(rng, 0.02, 0.07),
                       sw=bw + U(rng, -0.02, 0.03), ah=U(rng, 0.24, 0.34), hw=bw * U(rng, 0.95, 1.2),
                       ww=bw * U(rng, 0.85, 1.0), wy=U(rng, 0.5, 0.65), vneck=vneck,
                       sleeve=(U(rng, 30, 75), U(rng, 0.06, 0.16), U(rng, 0.1, 0.18)))
    # sleeveless top / vest: from thin straps over a deep scoop to a plain high-necked column
    bw = U(rng, 0.17, 0.31)
    nw = U(rng, 0.08, 0.17)
    return garment(bw=bw, nw=nw, nd=U(rng, 0.0, 0.3), sd=U(rng, 0.0, 0.05), sw=U(rng, nw + 0.04, max(nw + 0.05, bw)),
                   ah=U(rng, 0.25, 0.42), hw=bw * U(rng, 0.9, 1.15), armhole=U(rng, 0.0, 0.03))


def bulky_long_sleeved(rng, long=False):
    """Proportions shared by pullovers and coats (the same torso and arms fill both): chest half-width, sloping
    shoulders that run into sleeves hanging beside the body, sleeve width and angle, hem width.
    The two classes then differ only in what really tells them apart: neckline/collar and where the hem sits."""
    bw = U(rng, 0.17, 0.26) if long else U(rng, 0.21, 0.32)
    return dict(bw=bw, sd=U(rng, 0.05, 0.18), sw=bw + U(rng, 0.04, 0.13), ah=U(rng, 0.28, 0.42),
                hw=bw * U(rng, 0.85, 1.15)), U(rng, 0, 9), U(rng, 0.085, 0.155)


def pullover(rng):
    p, th, c = bulky_long_sleeved(rng)
    drop = U(rng, -0.06, 0.2)                          # cuffs usually hang below the ribbed hem
    L = (1.0 + drop - p["sd"]) / np.cos(th * DEG)
    hood = (U(rng, 0.12, 0.18), U(rng, 0.09, 0.17)) if rng.random() < 0.1 else None
    vneck = (U(rng, 0.05, 0.09), U(rng, 0.05, 0.14)) if rng.random() < 0.1 else None
    return garment(nw=U(rng, 0.13, 0.20), nd=U(rng, 0.0, 0.04), sleeve=(th, L, c),      # wide, flat neckline
                   hood=hood, vneck=None if hood else vneck, **p)


def coat(rng):
    long = rng.random() < 0.25
    p, th, c = bulky_long_sleeved(rng, long)
    rise = U(rng, 0.18, 0.42) if long else U(rng, -0.16, 0.22)   # cuffs end above the hem, or a little below on short jackets
    L = (1.0 - rise - p["sd"]) / np.cos(th * DEG)
    cw = U(rng, 0.07, 0.12)                            # stand-up collar: a narrow peak above the shoulder line
    collar = (cw, cw * U(rng, 0.55, 0.9), U(rng, 0.04, 0.11))
    hood = None
    vneck = (U(rng, 0.03, 0.06), U(rng, 0.0, 0.1)) if rng.random() < 0.35 else None     # open lapels
    if rng.random() < 0.2:
        hood, collar, vneck = (U(rng, 0.12, 0.19), U(rng, 0.1, 0.18)), None, None
    return garment(nw=cw, nd=0.0, sleeve=(th, L, c), collar=collar, hood=hood, vneck=vneck, **p)


def shirt(rng):
    r = rng.random()
    bw = U(rng, 0.20, 0.31)
    cw = U(rng, 0.07, 0.11)
    collar = (cw, cw * U(rng, 0.85, 1.0), U(rng, 0.01, 0.04))                           # small flat collar
    vneck = (U(rng, 0.04, 0.08), U(rng, 0.05, 0.18)) if rng.random() < 0.7 else None    # worn open at the throat
    sd = U(rng, 0.04, 0.16)
    if r < 0.75:                                       # thin long sleeves, shirt-tail hem at or below the cuffs
        th = U(rng, 1, 16)
        L = (1.0 - U(rng, -0.04, 0.25) - sd) / np.cos(th * DEG)
        return garment(bw=bw, nw=cw, nd=0.0, sd=sd, sw=bw + U(rng, 0.02, 0.11), ah=U(rng, 0.28, 0.40),
                       hw=bw * U(rng, 0.9, 1.2), hc=U(rng, 0.0, 0.08),
                       sleeve=(th, L, U(rng, 0.065, 0.12)), collar=collar, vneck=vneck)
    # blouse with three-quarter sleeves ending around hip level, pointing outwards
    return garment(bw=bw, nw=cw, nd=0.0, sd=sd, sw=bw + U(rng, 0.02, 0.07), ah=U(rng, 0.28, 0.38),
                   hw=bw * U(rng, 0.95, 1.15), hc=U(rng, 0.0, 0.06),
                   sleeve=(U(rng, 8, 28), U(rng, 0.5, 0.75), U(rng, 0.09, 0.14)), collar=collar, vneck=vneck)


def dress(rng):
    b = U(rng, 0.12, 0.20)                              # bust half-width
    by = U(rng, 0.2, 0.28)
    w = b * U(rng, 0.75, 0.98)
    wy = U(rng, 0.36, 0.46)
    h = b + U(rng, 0.0, 0.2)
    r = rng.random()
    if r < 0.25:                                       # thin straps over a straight bodice top
        sh = Shape()
        ty = U(rng, 0.1, 0.2)
        right = [(b * 0.9, ty), (b, by), (w, wy), (h, 1.0)]
        sh.poly([(0, ty + U(rng, 0, 0.05))] + right + [(-x, y) for x, y in right[::-1]])
        sx = U(rng, 0.06, min(0.13, b * 0.85)); tx = sx + U(rng, -0.03, 0.02)
        wd = U(rng, 0.02, 0.035)
        sh.line([(sx, ty + 0.02), (tx, wd / 2)], wd).line([(-sx, ty + 0.02), (-tx, wd / 2)], wd)
        return sh
    nw = U(rng, 0.07, 0.14)
    if r < 0.6:                                        # sleeveless with shoulders
        return garment(bw=b, nw=nw, nd=U(rng, 0.03, 0.16), sd=U(rng, 0.0, 0.03), sw=min(nw + U(rng, 0.04, 0.08), b + 0.02), ah=by,
                       hw=h, ww=w, wy=wy, armhole=U(rng, 0.0, 0.02))
    if r < 0.8:                                        # short sleeves
        return garment(bw=b, nw=nw, nd=U(rng, 0.02, 0.1), sd=U(rng, 0.02, 0.05), sw=b + U(rng, -0.01, 0.03), ah=by,
                       hw=h, ww=w, wy=wy, sleeve=(U(rng, 20, 60), U(rng, 0.07, 0.2), U(rng, 0.08, 0.13)))
    return garment(bw=b, nw=nw, nd=U(rng, 0.02, 0.1), sd=U(rng, 0.02, 0.05), sw=b + U(rng, 0.03, 0.07), ah=by,
                   hw=h, ww=w, wy=wy, sleeve=(U(rng, 2, 12), U(rng, 0.42, 0.68), U(rng, 0.05, 0.085)))


# ---------------------------------------------------------------- trousers
def trouser(rng):
    sh = Shape()
    if rng.random() < 0.08:                            # shorts: wide, short legs, small notch
        w0 = U(rng, 0.38, 0.5); w1 = w0 + U(rng, 0.0, 0.08)
        cy = U(rng, 0.6, 0.8); gap = U(rng, 0.03, 0.1)
        sh.poly([(-w0, 0), (w0, 0), (w1, 1), (gap, 1), (0, cy), (-gap, 1), (-w1, 1)])
        return sh
    w0 = U(rng, 0.17, 0.24)                            # waist half-width
    w1 = w0 + U(rng, 0.0, 0.035)                       # hip
    cy = U(rng, 0.24, 0.36)                            # crotch height
    leg = U(rng, 0.11, 0.19)                           # leg width at the ankle
    xo = U(rng, max(leg + 0.005, w0 - 0.03), w1 + 0.06)  # outer ankle
    xi = xo - leg
    kn = U(rng, -0.015, 0.015)
    sk = U(rng, -0.03, 0.03) if rng.random() < 0.3 else 0.0   # slight lean of the legs
    op = U(rng, 0.3, 0.9) * xi                         # the gap opens quickly below the crotch
    pts = [(-w0, 0), (w0, 0), (w1, 0.16), ((w1 + xo) / 2 + kn, 0.6), (xo + sk, 1), (xi + sk, 1), (op, cy + 0.14), (0, cy),
           (-op, cy + 0.14), (-xi + sk, 1), (-xo + sk, 1), (-(w1 + xo) / 2 - kn, 0.6), (-w1, 0.16)]
    sh.poly(pts)
    return sh


# ---------------------------------------------------------------- shoes: side view, toe to the left, y up (flipped at the end)
def _flip(sh):
    for i, (k, p, w, add) in enumerate(sh.prims):
        q = p.copy(); q[:, 1] = -q[:, 1]
        sh.prims[i] = (k, q, w, add)
    return sh


def sneaker(rng):
    sh = Shape()
    high = rng.random() < 0.08
    toe = U(rng, 0.11, 0.18)
    ts = U(rng, 0.0, 0.05)
    if high:
        ht = U(rng, 0.5, 0.72); xt = U(rng, 0.6, 0.75)
        hb = ht - U(rng, 0.0, 0.1)
        top = [(xt, ht), (0.96, hb), (1.0, hb - 0.03)]
    else:
        ht = U(rng, 0.29, 0.46); xt = U(rng, 0.55, 0.7)
        hc = ht - U(rng, 0.03, 0.12); hh = hc + U(rng, 0.0, 0.09)
        top = [(xt, ht), (xt + 0.05, ht - 0.02), (U(rng, 0.78, 0.86), hc), (0.95, hh), (1.0, hh - 0.02)]
    hv = toe + (ht - toe) * U(rng, 0.25, 0.5)
    pts = [(0.0, ts + 0.04), (0.03, toe * 0.8 + ts * 0.5), (0.1, toe), (0.3, hv), (xt - 0.04, ht - 0.015)] + top + \
          [(0.99, 0.1), (0.97, 0.0), (0.14, 0.0), (0.04, ts * 0.6)]
    sh.poly(pts)
    return _flip(sh)


def boot(rng):
    sh = Shape()
    H = U(rng, 0.48, 1.08)
    xs = U(rng, 0.38, 0.62)
    xs0 = xs - U(rng, -0.03, 0.12)
    ya = U(rng, 0.38, 0.9) * min(H, 0.9)
    toe = U(rng, 0.14, 0.23)
    ts = U(rng, 0.0, 0.04)
    xb = U(rng, 0.93, 1.0)
    Hb = H + U(rng, -0.06, 0.05)
    hh = U(rng, 0.05, 0.32) if rng.random() < 0.6 else 0.0
    upper = [(0.0, ts + 0.05), (0.03, toe * 0.8 + ts * 0.4), (0.1, toe), (0.28, toe + (ya - toe) * U(rng, 0.3, 0.55)),
             (xs0, ya), (xs, H), (xb, Hb), (min(1.0, xb + 0.03), 0.3), (0.985, 0.0)]
    if hh > 0:
        hwd = U(rng, 0.1, 0.28)
        under = [(0.985 - hwd, 0.0), (0.985 - hwd - 0.02, hh * 0.9), (0.5, hh * U(rng, 0.25, 0.5)), (0.33, 0.0)]
    else:
        under = []
    sh.poly(upper + under + [(0.12, 0.0), (0.03, ts * 0.6)])
    return _flip(sh)


def sandal(rng):
    sh = Shape()
    r = rng.random()
    sw = U(rng, 0.03, 0.065)                           # strap width
    if r < 0.42:                                       # flat sandal: thin sole and a few straps
        tk = U(rng, 0.03, 0.1)
        ts = U(rng, 0.0, 0.04)
        sh.poly([(0.0, ts), (0.0, ts + tk), (0.15, tk), (1.0, tk + U(rng, 0, 0.03)), (1.0, 0.0), (0.12, 0.0)])
        ax, ay = U(rng, 0.68, 0.86), U(rng, 0.2, 0.42)  # ankle point
        if rng.random() < 0.75:                        # toe strap
            x0 = U(rng, 0.06, 0.2); x1 = x0 + U(rng, 0.15, 0.3)
            sh.line(arc((x0 + x1) / 2, tk, (x1 - x0) / 2, U(rng, 0.06, 0.16), 180, 0), sw)
        if rng.random() < 0.5:                         # thong / instep strap to the ankle
            sh.line([(U(rng, 0.1, 0.3), tk), (ax, ay)], sw)
        if rng.random() < 0.7:                         # strap from mid-sole to ankle
            sh.line([(U(rng, 0.4, 0.62), tk), (ax, ay)], sw)
        if rng.random() < 0.75:                        # heel strap
            sh.line([(ax, ay), (0.97, U(rng, 0.12, 0.3)), (0.97, tk)], sw)
        if rng.random() < 0.4:                         # ankle band
            sh.line([(ax - U(rng, 0.0, 0.1), ay), (1.0 - sw / 2, ay + U(rng, -0.03, 0.05))], sw)
        if rng.random() < 0.3:                         # closed heel cup
            sh.poly([(0.78, tk), (0.99, tk), (0.99, U(rng, 0.15, 0.3)), (0.84, U(rng, 0.12, 0.25))])
        if rng.random() < 0.3:                         # broad band over the toes
            x0 = U(rng, 0.05, 0.15); x1 = x0 + U(rng, 0.2, 0.35); hb = U(rng, 0.08, 0.16)
            sh.poly([(x0, tk), (x0 + 0.06, tk + hb), (x1 - 0.03, tk + hb * 1.2), (x1, tk)])
        return _flip(sh)
    if r < 0.84:                                       # heeled sandal: rising sole, heel, ankle strap
        xb = U(rng, 0.28, 0.45)
        xh, yh = U(rng, 0.84, 0.95), U(rng, 0.28, 0.68)
        pf = U(rng, 0.0, 0.12) if rng.random() < 0.4 else 0.0       # platform under the forefoot
        sole = [(0.0, pf + 0.03), (xb, pf + 0.03)] + list(bez((xb, pf + 0.03), ((xb + xh) / 2 + 0.08, pf + 0.03 + (yh - pf) * 0.3), (xh, yh), 7))[1:] + [(1.0 - sw / 2, yh + 0.01)]
        sh.line(sole, U(rng, 0.04, 0.075))
        if pf > 0:
            sh.poly([(0.0, 0.0), (0.0, pf + 0.03), (xb, pf + 0.03), (xb + 0.05, 0.0)])
        hwd = U(rng, 0.035, 0.07) if rng.random() < 0.5 else U(rng, 0.1, 0.24)   # stiletto or block heel
        sh.poly([(1.0 - hwd * 1.4 - 0.02, yh), (1.0, yh), (1.0 - 0.01, 0.0), (1.0 - hwd - 0.01, 0.0)])
        if rng.random() < 0.8:                         # toe strap / band
            x0 = U(rng, 0.04, 0.16); x1 = x0 + U(rng, 0.14, 0.3)
            if rng.random() < 0.5:
                sh.line(arc((x0 + x1) / 2, pf + 0.04, (x1 - x0) / 2, U(rng, 0.07, 0.17), 180, 0), sw)
            else:
                hb = U(rng, 0.08, 0.18)
                sh.poly([(x0, pf + 0.03), (x0 + 0.07, pf + 0.03 + hb), (x1, pf + 0.03 + hb * 1.25), (x1 + 0.03, pf + 0.04)])
        top = yh + U(rng, 0.2, 0.45)
        if rng.random() < 0.85:                        # strap up the back of the heel and round the ankle
            bx = U(rng, 0.9, 0.98)
            sh.line([(bx, yh), (bx + U(rng, -0.04, 0.02), top)], sw)
            lx = U(rng, 0.5, 0.75)
            cuff = sw * U(rng, 1.8, 3.5) if rng.random() < 0.3 else sw        # thin strap or a broad cuff
            sh.line([(lx, top + U(rng, -0.06, 0.06) - cuff / 2), (1.0 - cuff / 2, top - cuff / 2)], cuff)
            if rng.random() < 0.25:                    # strap running down to the toes
                sh.line([(lx, top), (U(rng, 0.15, 0.35), pf + 0.1)], sw)
        if rng.random() < 0.35:                        # closed heel cup
            sh.poly([(0.78, yh - 0.12), (1.0, yh), (0.99, yh + U(rng, 0.15, 0.3)), (0.86, yh + U(rng, 0.1, 0.2))])
        return _flip(sh)
    # wedge / chunky sandal: thick sole block, broad bands over toes, instep and heel with openings between them
    yh = U(rng, 0.1, 0.5); tk = U(rng, 0.06, 0.18)
    sh.poly([(0.0, 0.02), (0.0, tk), (0.35, tk), (1.0, max(yh, tk)), (1.0, 0.0), (0.08, 0.0)])
    def top_at(x):                                     # height of the foot bed at x
        return tk if x < 0.35 else tk + (max(yh, tk) - tk) * (x - 0.35) / 0.65
    x0 = U(rng, 0.04, 0.12); x1 = x0 + U(rng, 0.2, 0.36); hb = U(rng, 0.1, 0.2)
    sh.poly([(x0, tk), (x0 + 0.08, tk + hb), (x1, tk + hb * 1.3), (x1 + 0.04, top_at(x1 + 0.04))])
    top = max(yh, tk) + U(rng, 0.18, 0.4)
    if rng.random() < 0.5:                             # band over the instep
        xa = U(rng, 0.42, 0.52); xb2 = xa + U(rng, 0.1, 0.2)
        sh.poly([(xa, top_at(xa)), (xa + 0.1, top - U(rng, 0.0, 0.1)), (xb2 + 0.1, top), (xb2, top_at(xb2))])
    if rng.random() < 0.6:                             # closed heel
        sh.poly([(0.74, top_at(0.74)), (1.0, max(yh, tk)), (1.0, top), (0.82, top)])
    else:
        sh.line([(0.95, max(yh, tk)), (0.94, top)], sw * 1.3)
    sh.line([(U(rng, 0.5, 0.7), top - U(rng, 0.0, 0.08)), (1.0 - sw, top)], sw * 1.3)
    return _flip(sh)


# ---------------------------------------------------------------- bags
def bag(rng):
    sh = Shape()
    hb = U(rng, 0.48, 1.0)                              # body height for width 1
    kind = rng.random()
    if kind < 0.45:                                    # rectangle with rounded corners
        rc = U(rng, 0.02, 0.12)
        body = np.concatenate([arc(0.5 - rc, rc, rc, rc, 270, 360, 4), arc(0.5 - rc, hb - rc, rc, rc, 0, 90, 4),
                               arc(-0.5 + rc, hb - rc, rc, rc, 90, 180, 4), arc(-0.5 + rc, rc, rc, rc, 180, 270, 4)])
        topw = 0.5
    elif kind < 0.8:                                   # trapezoid: narrower or wider at the top
        topw = 0.5 * U(rng, 0.6, 1.0); botw = 0.5
        if rng.random() < 0.3:
            topw, botw = botw, topw * 1.15
        body = np.array([(-topw, 0), (topw, 0), (botw, hb * 0.9), (botw * 0.92, hb), (-botw * 0.92, hb), (-botw, hb * 0.9)])
    else:                                              # rounded sack
        topw = 0.5 * U(rng, 0.55, 0.9)
        body = np.concatenate([[(-topw, 0), (topw, 0)], arc(0, hb * 0.45, 0.5, hb * 0.55, 0, 180, 12)])
    sh.poly(body)
    r = rng.random()
    wd = U(rng, 0.03, 0.06)
    if r < 0.5:                                        # carry handle: an arch over the top
        hx = topw * U(rng, 0.4, 0.95)
        hh = U(rng, 0.18, 0.7)
        if rng.random() < 0.75:
            sh.line(arc(0, 0.02, hx, hh, 180, 360, 16), wd)
        else:                                          # strap pulled up to a point
            sh.line([(-hx, 0.02), (U(rng, -0.05, 0.05), -hh), (hx, 0.02)], wd)
    elif r < 0.65:                                     # shoulder strap looping out to the sides
        hh = U(rng, 0.05, 0.3)
        sh.line(arc(0, hb * 0.3, 0.5 + U(rng, 0.0, 0.05), hb * 0.3 + hh, 180, 360, 16), wd)
    return sh


CLASS_MODELS = [tshirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, boot]


# ----------------------------------------------------------------- network assembly

K = 3000                      # templates per class
SEED = 20261120
T0, SAT = 0.02, 0.08          # ink threshold and the grey level at which a pixel counts as fully "item"
EDGE_GAIN = 4.0               # a template pixel a quarter covered by the drawing counts as item (same saturation as above)
SIGMA, RAD = 1.4, 4           # Gaussian blur of the silhouette before matching
TWINS = 600                   # short-sleeved drawings repeated in the shirt class ...
TWIN_HANDICAP = 7.0           # ... which only win when a button line is seen
LINE_W = 12.0                 # score bonus (coat and shirt) for a clear line all the way down the middle of the front
LINE_T, LINE_S = 0.03, 0.07   # contrast at which the line starts to count, and range over which it saturates
GAP_K = 9                     # openwork: background trapped in horizontal gaps narrower than GAP_K px between ink
GAP_T, GAP_S = 15.0, 24.0     # trapped area (px) at which the sandal bonus starts (a sneaker's ankle opening traps a few px), and range to saturation
GAP_W = 25.0                  # sandal bonus for clearly open strap work on a tall shoe (boots rarely have such gaps)
TALL_ROWS, TALL_T, TALL_S = 6, 1.0, 3.0   # "tall item": ink in the top TALL_ROWS rows (more than TALL_T px, saturating TALL_S px later)
TOP_GAIN, TOP_ROWS = 1.5, 3.0  # neckline / collar rows count more in the match: weight 1 + TOP_GAIN * exp(-row / TOP_ROWS)


def gauss1d():
    x = np.arange(-RAD, RAD + 1)
    g = np.exp(-x * x / (2 * SIGMA * SIGMA))
    return g / g.sum()


def blur_matrix():
    g = gauss1d()
    B = np.zeros((28, 28))
    for i in range(28):
        for k in range(-RAD, RAD + 1):
            if 0 <= i + k < 28:
                B[i, i + k] = g[k + RAD]
    return B


def draw_class(c, n, seed):
    rng = np.random.default_rng(seed)
    out = np.empty((n, 28, 28))
    for i in range(n):
        out[i] = np.clip(CLASS_MODELS[c](rng).render() * EDGE_GAIN, 0, 1)
    return out


def templates():
    T = np.empty((10, K, 28, 28))
    handicap = np.zeros((10, K))
    for c in range(10):
        T[c] = draw_class(c, K, SEED + c)
    # shirt class: the last TWINS slots repeat short-sleeved drawings of class 0 (same seed, so identical silhouettes)
    T[6, K - TWINS:] = T[0, :TWINS]
    handicap[6, K - TWINS:] = TWIN_HANDICAP
    return T, handicap


def line_branch():
    """Detector for a thin vertical line (zip, button placket) running all the way down the middle of the front.
    Four stacked row segments must each show the line (same column, same polarity): min over segments, max over
    3 column placements x 2 polarities.  Prints on the chest or the gap between trouser legs do not qualify."""
    opts = [({13: 1.0}, (11, 15)), ({14: 1.0}, (12, 16)), ({13: 0.5, 14: 0.5}, (11, 16))]
    segs = [range(3, 8), range(8, 13), range(13, 18), range(18, 23)]
    W1 = np.zeros((24, 28, 28)); b1 = np.full(24, -LINE_T)
    u = 0
    for centre, sides in opts:
        for sgn in (1.0, -1.0):
            for h, rows in enumerate(segs):
                for r in rows:
                    for col, w in centre.items():
                        W1[u * 4 + h, r, col] += sgn * w / len(rows)
                    for col in sides:
                        W1[u * 4 + h, r, col] -= sgn * 0.5 / len(rows)
            u += 1
    # min(a, b) = a - relu(a - b) for a, b >= 0, applied twice
    W2 = np.zeros((24, 24)); W3 = np.zeros((12, 24)); W4 = np.zeros((12, 12)); W5 = np.zeros((6, 12))
    for u in range(6):
        for p in range(2):                               # pair p of segments (0,1) and (2,3)
            a, b = 4 * u + 2 * p, 4 * u + 2 * p + 1
            W2[a, a] = 1.0                               # a
            W2[b, a] = 1.0; W2[b, b] = -1.0              # a - b
            W3[2 * u + p, a] = 1.0; W3[2 * u + p, b] = -1.0     # min(a, b)
        W4[2 * u, 2 * u] = 1.0
        W4[2 * u + 1, 2 * u] = 1.0; W4[2 * u + 1, 2 * u + 1] = -1.0
        W5[u, 2 * u] = 1.0; W5[u, 2 * u + 1] = -1.0
    bonus = np.zeros(10); bonus[4] = LINE_W; bonus[6] = LINE_W
    return {"ln1.W": W1.reshape(24, 784), "ln1.b": b1, "ln2.W": W2, "ln2.b": np.zeros(24), "ln3.W": W3, "ln3.b": np.zeros(12),
            "ln4.W": W4, "ln4.b": np.zeros(12), "ln5.W": W5, "ln5.b": np.zeros(6),
            "ln6.W": np.array([[-1.0 / LINE_S]]), "ln6.b": np.array([1.0]), "ln7.W": -bonus[:, None], "ln7.b": bonus}


def gap_branch():
    """Openwork detector, used to tell strappy heeled sandals from boots.
    Horizontal closing of the ink mask (dilate then erode along rows, done with max-pooling) fills gaps narrower than
    GAP_K between pieces of ink on the same row; the filled area is the trapped background.
    Layers: mask -> zero-pad -> maxpool [1,K] (dilate) -> 1 - x -> maxpool [1,K] (this is 1 - closed) -> sum.
    The pad is wide enough that everything outside the picture behaves as empty background.
    The bonus is gated by "tall item" (ink in the top rows): among low shoes, dark sneakers with white stripes, laces
    and logos (which photograph as background) trap background too, so there the cue says nothing.
    g = clip((trapped - GAP_T) / GAP_S, 0, 1), t = clip((top ink - TALL_T) / TALL_S, 0, 1), bonus = GAP_W * min(g, t)."""
    p = GAP_K - 1                                        # pad on every side (rows are padded too, and ignored)
    Wc = np.zeros((1, 28 + 2 * p, 28)); Wc[0, p:p + 28, :] = -1.0        # -(1 - closed) on the real rows
    Wt = np.zeros((1, 28, 28)); Wt[0, :TALL_ROWS, :] = 1.0 / TALL_S
    bonus = np.zeros(10); bonus[5] = GAP_W
    return {"gp1.W": np.ones((1, 1, 1, 1)), "gp1.b": np.zeros(1),
            "gp2.W": -np.ones((1, 1, 1, 1)), "gp2.b": np.ones(1),
            # trapped area = sum(closed) - sum(mask) = 784 - sum(1 - closed) - sum(mask); then (area - GAP_T) / GAP_S
            "gp4.W": Wc.reshape(1, -1) / GAP_S, "gp4.b": np.array([(784.0 - GAP_T) / GAP_S]),
            "gp5.W": -np.ones((1, 784)) / GAP_S, "gp5.b": np.zeros(1),
            "gp6.W": -np.ones((1, 1)), "gp6.b": np.ones(1),              # ug = relu(1 - relu(v)) = 1 - g
            "tl1.W": Wt.reshape(1, 784), "tl1.b": np.array([-TALL_T / TALL_S]),
            "tl2.W": -np.ones((1, 1)), "tl2.b": np.ones(1),              # ut = relu(1 - relu(tv)) = 1 - t
            # min(g, t) = 1 - max(ug, ut) = 1 - ug - relu(ut - ug)
            "gpa.W": np.array([[1.0], [-1.0]]), "gpa.b": np.zeros(2),    # [ug, -ug]
            "gpb.W": np.array([[0.0], [1.0]]), "gpb.b": np.zeros(2),     # [0, ut]
            "gp7.W": -bonus[:, None] * np.ones((1, 2)), "gp7.b": bonus}


def build(out_dir):
    T, handicap = templates()
    B = blur_matrix()
    BT = np.einsum('ij,ckjl,ml->ckim', B, T, B).reshape(10 * K, 784)     # blurred templates, one row per drawing
    roww = 1.0 + TOP_GAIN * np.exp(-np.arange(28) / TOP_ROWS)
    pw = np.repeat(roww, 28)                                             # weight of each pixel in the match
    g = gauss1d()
    k = 1.0 / (SAT - T0)
    w = {
        "ink.W": np.full((1, 1, 1, 1), k), "ink.b": np.array([-k * T0]),
        "inv.W": np.full((1, 1, 1, 1), -1.0), "inv.b": np.array([1.0]),
        "mask.W": np.full((1, 1, 1, 1), -1.0), "mask.b": np.array([1.0]),
        "blur.W": np.outer(g, g)[None, None], "blur.b": np.zeros(1),
        # unit j: -(weighted squared distance between the blurred mask and blurred drawing j) + a term common to all units
        "match.W": 2.0 * BT * pw, "match.b": -(BT * BT * pw).sum(1) - handicap.reshape(-1),
    }
    w.update(line_branch())
    w.update(gap_branch())
    layers = [
        {"name": "ink", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "ink_r", "op": "relu", "inputs": ["ink"]},
        {"name": "inv", "op": "conv", "inputs": ["ink_r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "inv_r", "op": "relu", "inputs": ["inv"]},
        {"name": "mask", "op": "conv", "inputs": ["inv_r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "blur", "op": "conv", "inputs": ["mask"], "out": 1, "kernel": [2 * RAD + 1, 2 * RAD + 1], "stride": 1, "padding": RAD},
        {"name": "match", "op": "dense", "inputs": ["blur"], "out": 10 * K},
        {"name": "bank", "op": "reshape", "inputs": ["match"], "shape": [10, 1, K]},
        {"name": "best", "op": "maxpool", "inputs": ["bank"], "size": [1, K], "stride": [1, K]},
        {"name": "shape_score", "op": "reshape", "inputs": ["best"], "shape": [10]},
        {"name": "ln1", "op": "dense", "inputs": ["image"], "out": 24},
        {"name": "ln1_r", "op": "relu", "inputs": ["ln1"]},
        {"name": "ln2", "op": "dense", "inputs": ["ln1_r"], "out": 24},
        {"name": "ln2_r", "op": "relu", "inputs": ["ln2"]},
        {"name": "ln3", "op": "dense", "inputs": ["ln2_r"], "out": 12},
        {"name": "ln4", "op": "dense", "inputs": ["ln3"], "out": 12},
        {"name": "ln4_r", "op": "relu", "inputs": ["ln4"]},
        {"name": "ln5", "op": "dense", "inputs": ["ln4_r"], "out": 6},
        {"name": "ln5_m", "op": "reshape", "inputs": ["ln5"], "shape": [1, 1, 6]},
        {"name": "ln_max", "op": "maxpool", "inputs": ["ln5_m"], "size": [1, 6], "stride": [1, 6]},
        {"name": "ln6", "op": "dense", "inputs": ["ln_max"], "out": 1},
        {"name": "ln6_r", "op": "relu", "inputs": ["ln6"]},
        {"name": "ln7", "op": "dense", "inputs": ["ln6_r"], "out": 10},
        {"name": "gp1", "op": "conv", "inputs": ["mask"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": GAP_K - 1},
        {"name": "gp_dil", "op": "maxpool", "inputs": ["gp1"], "size": [1, GAP_K], "stride": [1, 1]},
        {"name": "gp2", "op": "conv", "inputs": ["gp_dil"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "gp_ero", "op": "maxpool", "inputs": ["gp2"], "size": [1, GAP_K], "stride": [1, 1]},
        {"name": "gp4", "op": "dense", "inputs": ["gp_ero"], "out": 1},
        {"name": "gp5", "op": "dense", "inputs": ["mask"], "out": 1},
        {"name": "gp_v", "op": "add", "inputs": ["gp4", "gp5"]},
        {"name": "gp_v_r", "op": "relu", "inputs": ["gp_v"]},
        {"name": "gp6", "op": "dense", "inputs": ["gp_v_r"], "out": 1},
        {"name": "gp6_r", "op": "relu", "inputs": ["gp6"]},
        {"name": "tl1", "op": "dense", "inputs": ["mask"], "out": 1},
        {"name": "tl1_r", "op": "relu", "inputs": ["tl1"]},
        {"name": "tl2", "op": "dense", "inputs": ["tl1_r"], "out": 1},
        {"name": "tl2_r", "op": "relu", "inputs": ["tl2"]},
        {"name": "gpa", "op": "dense", "inputs": ["gp6_r"], "out": 2},
        {"name": "gpb", "op": "dense", "inputs": ["tl2_r"], "out": 2},
        {"name": "gp_ab", "op": "add", "inputs": ["gpa", "gpb"]},
        {"name": "gp_ab_r", "op": "relu", "inputs": ["gp_ab"]},
        {"name": "gp7", "op": "dense", "inputs": ["gp_ab_r"], "out": 10},
        {"name": "scores", "op": "add", "inputs": ["shape_score", "ln7", "gp7"]},
    ]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k_: v.astype(np.float32) for k_, v in w.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")


