"""build.py -- writes network.json and weights.npz for the clothing classifier.   Usage: python3 build.py OUT_DIR

Reads no files. Every weight comes from the code below:
  1. geom.py part   : a small renderer (signed-distance polygons / strokes) that draws a shape, fits its longest side to
                      28 pixels and centres it, the way the dataset trims and centres each item.
  2. models.py part : hand-typed shape models of the ten classes (proportions from general knowledge of the garments),
                      and hand-written tables saying how typical each attribute is for pullover / coat / shirt / T-shirt.
  3. net.py part    : renders a bank of prototypes from those models with a fixed seed and wires the network:
                      silhouette -> row/column-filled silhouettes -> distance to every prototype -> best prototype per
                      shape group -> class scores from the tables, plus a centre-line (zip / placket) cue.
No training, fitting or data statistics are involved.
"""
# ============================== geom.py ==============================
"""Shape renderer: signed-distance primitives, auto-fit to the 28x28 frame the way the dataset trims and centres items."""
import numpy as np

G = 28
_c = np.arange(G) + 0.5
PX, PY = np.meshgrid(_c, _c)          # pixel-centre coordinates, PX[y, x]


def sd_poly(V):
    """Signed distance (negative inside) to a simple polygon with vertices V (n, 2) given in pixel coordinates."""
    V = np.asarray(V, float)
    n = len(V)
    d = np.full(PX.shape, 1e18)
    s = np.ones(PX.shape)
    for i in range(n):
        a = V[i]; b = V[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        wx, wy = PX - a[0], PY - a[1]
        den = ex * ex + ey * ey
        t = np.clip((wx * ex + wy * ey) / max(den, 1e-12), 0.0, 1.0)
        dx, dy = wx - ex * t, wy - ey * t
        d = np.minimum(d, dx * dx + dy * dy)
        c1 = PY >= a[1]; c2 = PY < b[1]; c3 = ex * wy > ey * wx
        flip = (c1 & c2 & c3) | ((~c1) & (~c2) & (~c3))
        s = np.where(flip, -s, s)
    return s * np.sqrt(d)


def sd_capsule(a, b, r):
    ex, ey = b[0] - a[0], b[1] - a[1]
    wx, wy = PX - a[0], PY - a[1]
    den = ex * ex + ey * ey
    t = np.clip((wx * ex + wy * ey) / max(den, 1e-12), 0.0, 1.0)
    dx, dy = wx - ex * t, wy - ey * t
    return np.sqrt(dx * dx + dy * dy) - r


_EA = (np.arange(28) + 0.5) * (2 * np.pi / 28)
_ECOS, _ESIN = np.cos(_EA), np.sin(_EA)


class Scene:
    """A shape in its own units: filled polygons / ellipses / strokes that are added or cut away."""

    def __init__(self):
        self.items = []          # (mode, kind, data, layer)
        self.layer = 0           # 0 = solid part, 1 = soft zone (material may or may not be there)
        self.level = 0.0         # silhouette value given to the soft zone

    def poly(self, V, mode="add"):
        self.items.append((mode, "poly", np.asarray(V, float), self.layer)); return self

    def ell(self, c, rx, ry, mode="add"):
        self.items.append((mode, "ell", (float(c[0]), float(c[1]), float(rx), float(ry)), self.layer)); return self

    def stroke(self, pts, r_px, mode="add"):
        """polyline with half-thickness r_px given in output pixels"""
        self.items.append((mode, "stroke", (np.asarray(pts, float), float(r_px)), self.layer)); return self

    def sym(self, V, mode="add"):
        """polygon plus its mirror image about x = 0"""
        V = np.asarray(V, float)
        self.poly(V, mode); self.poly(V * np.array([-1.0, 1.0]), mode); return self

    def bounds(self, scale):
        lo = np.array([1e18, 1e18]); hi = -lo
        for mode, kind, d, _ in self.items:
            if mode != "add":
                continue
            if kind == "poly":
                lo = np.minimum(lo, d.min(0)); hi = np.maximum(hi, d.max(0))
            elif kind == "ell":
                lo = np.minimum(lo, [d[0] - d[2], d[1] - d[3]]); hi = np.maximum(hi, [d[0] + d[2], d[1] + d[3]])
            else:
                r = d[1] / scale
                lo = np.minimum(lo, d[0].min(0) - r); hi = np.maximum(hi, d[0].max(0) + r)
        return lo, hi

    def render(self, fat=0.42, ramp=0.2, jit=None):
        """Fit the longest side to 28 pixels, centre the other. Returns (solid, zone): soft silhouettes (28, 28) in [0, 1]
        of the solid part and of the soft zone. jit = three numbers in [0, 1): the item may be a little smaller than the
        frame and off-centre, because faint parts (shadow, hanger) that were trimmed with it do not show."""
        scale = 1.0
        for _ in range(3):
            lo, hi = self.bounds(scale)
            scale = G / max(hi[0] - lo[0], hi[1] - lo[1])
        j = np.zeros(2)
        if jit is not None:
            wide = (hi[0] - lo[0]) >= (hi[1] - lo[1])
            shrink = 1.0 - (0.07 if wide else 0.05) * jit[0]
            room = 0.5 * G * (1.0 - shrink)
            a, b = 2.0 * jit[1] - 1.0, 2.0 * jit[2] - 1.0
            j = np.array([room * a, 3.0 * jit[2] - 2.0]) if wide else np.array([0.7 * a, room * b])
            scale = scale * shrink
        off = (G - (hi - lo) * scale) / 2.0 - lo * scale + j
        sd = [np.full(PX.shape, 1e18), np.full(PX.shape, 1e18)]
        for mode, kind, d, layer in self.items:
            if kind == "poly":
                v = sd_poly(d * scale + off)
            elif kind == "ell":
                v = sd_poly(np.stack([d[0] + d[2] * _ECOS, d[1] + d[3] * _ESIN], 1) * scale + off)
            else:
                P = d[0] * scale + off
                v = np.full(PX.shape, 1e18)
                for i in range(len(P) - 1):
                    v = np.minimum(v, sd_capsule(P[i], P[i + 1], d[1]))
            sd[layer] = np.minimum(sd[layer], v) if mode == "add" else np.maximum(sd[layer], -v)
        return np.clip((fat - sd[0]) / ramp, 0.0, 1.0), np.clip((fat - sd[1]) / ramp, 0.0, 1.0)


# ============================== models.py ==============================
"""Hand-typed shape models of the ten classes. Every number is a proportion chosen from general knowledge of the garment
(unit = garment length for upper-body items, trousers and dresses; unit = shoe length for footwear; unit = bag width)."""
import numpy as np


D2R = np.pi / 180.0


def U(r, a, b):
    return a + (b - a) * r.random()


# ------------------------------------------------------------------ upper-body garments
def top_scene(nw, sw, sdrop, bw, ya, hw, hem=1.0, sleeve=None, neck=0.0, collar=None, tail=0.0, rnd=0.0):
    """Body = neck point, shoulder tip, armpit, hem; optional sleeves (length, angle from vertical, cuff width);
    neck > 0 cuts a scooped neckline, collar = (height, half-width) adds a standing collar / hood above the shoulders;
    tail > 0 rounds the hem downwards (shirt tail); rnd > 0 bulges the shoulder line (rounded shoulders)."""
    s = Scene()
    mx, my = 0.5 * (nw + sw) + rnd * sdrop, 0.5 * sdrop - rnd * (sw - nw) * 0.5
    s.poly([(-nw, 0), (nw, 0), (mx, my), (sw, sdrop), (bw, ya), (hw, hem), (-hw, hem), (-bw, ya), (-sw, sdrop), (-mx, my)])
    if tail > 0.012:
        s.ell((0, hem), hw * 0.98, tail)
    if sleeve is not None:
        L, al, wc = sleeve
        dx, dy = np.sin(al), np.cos(al)
        S = np.array([sw, sdrop]); O = S + L * np.array([dx, dy]); I = O + wc * np.array([-dy, dx])
        s.sym([(mx, my), S, O, I, (bw, ya)])
    if neck > 0:
        s.ell((0, 0), nw * 0.9, neck, "sub")
    if collar is not None:
        ch, cw = collar
        s.poly([(-nw, 0.01), (-cw, -ch), (cw, -ch), (nw, 0.01)])
    return s


def long_top(r, nw, sw, sdrop, bw, wc, cuff_y, hem, al_max, collar=None, neck=0.0, tail=0.0, flare=(0.92, 1.06), gap=None):
    """Long-sleeved garment hanging flat: sloping rounded shoulders, sleeves down the sides ending at height cuff_y.
    gap: if given, the body width at the hem is set so that this much space shows between sleeve and body there."""
    al = U(r, 0, al_max) * D2R
    L = (cuff_y - sdrop) / np.cos(al)
    hw = bw * U(r, *flare)
    if gap is not None:
        hw = max(sw + np.tan(al) * (cuff_y - sdrop) - wc / np.cos(al) - gap, 0.2)
        bw = hw / U(r, *flare)
    return top_scene(nw, sw, sdrop, bw, U(r, 0.36, 0.5), hw, hem=hem, sleeve=(L, al, wc), neck=neck,
                     collar=collar, tail=tail, rnd=U(r, 0.1, 0.45))


def dome(r):
    """Long-sleeved garment (pullover, coat/jacket or long-sleeved shirt): the three share this shape family.
    Returns the shape and its attributes; which class a given shape suggests is set by the tables in TOP_TABLES."""
    top = "flat" if r.random() < 0.5 else "collar"
    v = r.random()
    bottom = "notch" if v < 0.4 else ("level" if v < 0.7 else "long")
    sw = U(r, 0.27, 0.46) if bottom != "long" else U(r, 0.25, 0.42)
    wc = U(r, 0.07, 0.17); gap = U(r, -0.04, 0.05); sdrop = U(r, 0.15, 0.30)
    if top == "flat":
        nw, neck, col = U(r, 0.13, 0.21), U(r, 0.0, 0.05), None
    else:
        nw = U(r, 0.08, 0.16); neck = 0.0; col = (U(r, 0.02, 0.14), nw * U(r, 0.5, 1.0))   # standing collar ... hood
    if bottom == "notch":
        cuff, hem = 1.0, U(r, 0.86, 0.96)         # sleeves longer than the body: gap at the bottom between the cuffs
    elif bottom == "level":
        cuff, hem = U(r, 0.97, 1.0), 1.0
    else:
        cuff, hem = U(r, 0.70, 0.94), 1.0         # body clearly longer than the sleeves
    al = U(r, 0, 6) * D2R
    L = (cuff - sdrop) / np.cos(al)
    hw = max(sw + np.tan(al) * (cuff - sdrop) - wc / np.cos(al) - gap, 0.2)
    bw = hw / U(r, 0.92, 1.08)
    s = top_scene(nw, sw, sdrop, bw, U(r, 0.36, 0.5), hw, hem=hem, sleeve=(L, al, wc), neck=neck, collar=col,
                  tail=U(r, 0.0, 0.05) if bottom != "notch" else 0.0, rnd=U(r, 0.1, 0.45))
    wtot = sw + np.tan(al) * (cuff - sdrop)
    s.attrs = dict(top=top, bottom=bottom, width="narrow" if wtot < 0.36 else ("medium" if wtot < 0.43 else "wide"),
                   sleeves="merged" if gap <= 0.012 else ("apart_thin" if wc < 0.105 else "apart_thick"))
    return s


def short_top(r):
    """Short-sleeved garment (T-shirt or short-sleeved shirt)."""
    sw = U(r, 0.25, 0.36); bw = sw * U(r, 0.88, 1.06)
    L = U(r, 0.12, 0.40)
    col = (U(r, 0.02, 0.06), U(r, 0.05, 0.10)) if r.random() < 0.35 else None
    s = top_scene(U(r, 0.09, 0.16), sw, U(r, 0.04, 0.11), bw, U(r, 0.27, 0.42), bw * U(r, 0.92, 1.12),
                  sleeve=(L, U(r, 25, 65) * D2R, U(r, 0.15, 0.26)), collar=col,
                  neck=0.0 if col else U(r, 0.02, 0.09), tail=U(r, 0.0, 0.05), rnd=U(r, 0, 0.3))
    s.attrs = dict(sleeve="short" if L < 0.24 else "elbow", body="wide" if bw > 0.31 else "regular")
    return s


def sleeveless_top(r):
    """Sleeveless top (tank top / vest, or sleeveless blouse)."""
    so = U(r, 0.15, 0.27); bw = max(U(r, 0.225, 0.33), so); ya = U(r, 0.25, 0.4); fl = U(r, 0.95, 1.2)
    nd = U(r, 0.02, 0.30)
    s = Scene()
    s.poly([(-so, 0), (so, 0), (bw, ya), (bw * fl, 1), (-bw * fl, 1), (-bw, ya)])
    s.ell((0, 0), so - U(r, 0.03, 0.07), nd, "sub")
    s.attrs = dict(scoop="deep" if nd > 0.12 else "shallow", hem="flared" if fl > 1.08 else "straight", body="wide" if bw > 0.275 else "regular")
    return s


# How typical each attribute is for each class (rough shares from general knowledge of these garments).
# class ids: 0 T-shirt/top, 2 pullover, 4 coat, 6 shirt
TOP_TABLES = {
    "dome": {
        "share": {2: 0.95, 4: 0.95, 6: 0.50},     # how much of the class is long-sleeved at all
        # crew neck (flat top) is the pullover's; a standing collar or hood is the coat's
        "top": {"flat": {2: 0.70, 4: 0.30, 6: 0.45}, "collar": {2: 0.30, 4: 0.70, 6: 0.55}},
        # ribbed pullover hems sit above the cuffs; coats and shirts are longer in the body
        "bottom": {"notch": {2: 0.65, 4: 0.40, 6: 0.30}, "level": {2: 0.25, 4: 0.30, 6: 0.40}, "long": {2: 0.10, 4: 0.30, 6: 0.30}},
        # pullovers are boxy; coats and shirts are longer, so narrower once scaled to the frame
        "width": {"narrow": {2: 0.10, 4: 0.35, 6: 0.35}, "medium": {2: 0.50, 4: 0.50, 6: 0.50}, "wide": {2: 0.40, 4: 0.15, 6: 0.15}},
        # sleeves hanging clear of the body: light garments; thin sleeves: shirts
        "sleeves": {"merged": {2: 0.50, 4: 0.80, 6: 0.60}, "apart_thin": {2: 0.125, 4: 0.02, 6: 0.22}, "apart_thick": {2: 0.375, 4: 0.18, 6: 0.18}},
    },
    "short": {
        "share": {0: 0.78, 6: 0.30},
        "sleeve": {"short": {0: 0.80, 6: 0.45}, "elbow": {0: 0.20, 6: 0.55}},
        "body": {"wide": {0: 0.30, 6: 0.50}, "regular": {0: 0.70, 6: 0.50}},
    },
    "sleeveless": {
        "share": {0: 0.15, 6: 0.20},
        "scoop": {"deep": {0: 0.60, 6: 0.30}, "shallow": {0: 0.40, 6: 0.70}},
        "hem": {"flared": {0: 0.25, 6: 0.50}, "straight": {0: 0.75, 6: 0.50}},
        "body": {"wide": {0: 0.35, 6: 0.50}, "regular": {0: 0.65, 6: 0.50}},
    },
}


# Corrections to the tables (in log units) made by hand after the checks: a shape that merely could be a shirt is
# usually something else, and coats were being handed to the pullover too readily.
CLASS_ADJ = {6: -2.5, 4: 1.0}


def table_scores(family, attrs):
    """class -> log of (share of the class that looks like this shape), from the tables above"""
    t = TOP_TABLES[family]
    out = {}
    for k, sh in t["share"].items():
        v = np.log(sh) + CLASS_ADJ.get(k, 0.0)
        for a, val in attrs.items():
            v += np.log(t[a][val][k])
        out[k] = v
    return out


# ------------------------------------------------------------------ trousers
def trouser(r):
    ww = U(r, 0.15, 0.21); hp = ww + U(r, 0.0, 0.03); yh = U(r, 0.12, 0.25); yc = U(r, 0.30, 0.48)
    xo = hp + U(r, -0.05, 0.03)
    g0 = U(r, 0.015, 0.03); g1 = g0 + U(r, 0.012, 0.06)      # half-gap between the legs at the crotch and at the hem
    s = Scene()
    s.poly([(-ww, 0), (ww, 0), (hp, yh), (xo, 1), (g1, 1), (g0, yc), (-g0, yc), (-g1, 1), (-xo, 1), (-hp, yh)])
    return s


def trouser_folded(r):
    ww = U(r, 0.12, 0.2); hw = ww * U(r, 0.75, 1.05); e = U(r, 0, 0.02)
    s = Scene()
    s.poly([(-ww, 0), (ww, 0), (ww + e, 0.2), (hw, 1), (-hw, 1), (-ww - e, 0.2)])
    return s


def shorts(r):
    ww = U(r, 0.34, 0.5); xo = ww + U(r, 0.0, 0.12); yc = U(r, 0.45, 0.7); xi = U(r, 0.03, 0.14)
    s = Scene()
    s.poly([(-ww, 0), (ww, 0), (xo, 1), (xi, 1), (0, yc), (-xi, 1), (-xo, 1)])
    return s


# ------------------------------------------------------------------ dresses
def dress(r, kind):
    nw = U(r, 0.07, 0.12); sw = U(r, 0.15, 0.22); sdrop = U(r, 0.02, 0.06); bw = U(r, 0.15, 0.21); ya = U(r, 0.16, 0.24)
    ww = bw * U(r, 0.85, 1.0); yw = U(r, 0.3, 0.42); hw = U(r, 0.17, 0.30)
    if kind == "strap":
        sw = nw + U(r, 0.01, 0.03)
    if kind == "sheath":                          # straight column
        sw = U(r, 0.13, 0.2); bw = sw * U(r, 0.95, 1.1); ww = bw * U(r, 0.92, 1.0); hw = bw * U(r, 0.9, 1.15)
    if kind == "long":
        return long_top(r, nw, U(r, 0.2, 0.28), U(r, 0.08, 0.16), bw, U(r, 0.06, 0.09), U(r, 0.5, 0.72), 1.0, 10,
                        neck=U(r, 0.01, 0.06), flare=(1.0, 1.6))
    s = Scene()
    s.poly([(-nw, 0), (nw, 0), (sw, sdrop), (bw, ya), (ww, yw), (hw, 1), (-hw, 1), (-ww, yw), (-bw, ya), (-sw, sdrop)])
    if kind == "short":
        L, al, wc = U(r, 0.08, 0.2), U(r, 30, 65) * D2R, U(r, 0.09, 0.15)
        dx, dy = np.sin(al), np.cos(al)
        S = np.array([sw, sdrop]); O = S + L * np.array([dx, dy]); I = O + wc * np.array([-dy, dx])
        s.sym([S, O, I, (bw, ya)])
    s.ell((0, 0), nw * 0.9, U(r, 0.02, 0.10) if kind != "strap" else U(r, 0.08, 0.18), "sub")
    return s


# ------------------------------------------------------------------ footwear (toe at the left, y measured upwards)
def _up(V):
    return [(x, -y) for x, y in V]


def sneaker(r):
    t = U(r, 0.10, 0.18)                         # toe-box height
    px, ph = U(r, 0.56, 0.78), U(r, 0.32, 0.56)  # tongue / collar peak
    hh = ph * U(r, 0.7, 1.0)                     # heel height
    dip = hh * U(r, 0.8, 1.0)
    s = Scene()
    s.poly(_up([(0.0, 0.05), (0.03, t * 0.8), (0.12, t), (0.32, t + (ph - t) * U(r, 0.25, 0.45)), (px - 0.08, ph * 0.93), (px, ph),
                (px + 0.08, dip + (ph - dip) * 0.3), (0.88, dip), (0.96, hh), (1.0, hh * 0.85), (1.0, 0.04), (0.96, 0.0), (0.05, 0.0)]))
    return s


def hightop(r):
    t = U(r, 0.12, 0.2); px = U(r, 0.55, 0.68); ph = U(r, 0.55, 0.8)
    s = Scene()
    s.poly(_up([(0.0, 0.05), (0.03, t * 0.8), (0.14, t), (0.36, t + (ph - t) * 0.4), (px, ph), (0.97, ph * U(r, 0.9, 1.0)),
                (1.0, ph * 0.5), (1.0, 0.04), (0.96, 0.0), (0.05, 0.0)]))
    return s


def boot(r):
    t = U(r, 0.15, 0.25)                          # toe height
    sx = U(r, 0.40, 0.60)                         # where the shaft front reaches the top
    top = U(r, 0.56, 1.05)                        # shaft height: a boot stands clearly taller than a sneaker's collar
    back = U(r, 0.93, 1.0)
    x1 = U(r, 0.24, 0.36)                         # where the instep starts to rise
    V = [(0.0, 0.06), (0.02, t * 0.75), (0.1, t), (x1, t + 0.04), (sx - 0.05, min(t + U(r, 0.2, 0.34), top * 0.85)),
         (sx, top), (back, top), (1.0, top * 0.5), (1.0, 0.0), (0.05, 0.0)]
    s = Scene(); s.poly(_up(V))
    if r.random() < 0.45:                         # raised heel: cut the arch between ball and heel
        ax0, ax1 = U(r, 0.40, 0.55), U(r, 0.68, 0.92); ah = U(r, 0.07, 0.40)
        s.poly(_up([(ax0, -0.2), (ax0 + 0.04, ah * 0.5), (ax1 - 0.03, ah), (ax1, -0.2)]), "sub")
    return s


def sandal_flat(r):
    th = U(r, 0.05, 0.12)                         # sole thickness
    s = Scene()
    s.poly(_up([(0.0, 0.0), (0.0, th), (1.0, th * U(r, 1.0, 2.0)), (1.0, 0.0)]))
    h = U(r, 0.16, 0.38)                          # strap height
    x0 = U(r, 0.08, 0.3); x1 = U(r, 0.75, 0.96); xm = U(r, 0.4, 0.65)
    rr = U(r, 0.7, 1.6)
    k = r.integers(0, 4)
    if k == 0:      # toe strap + ankle strap
        s.stroke(_up([(x0, th), (x0 + 0.12, h * 0.6), (x0 + 0.26, th)]), rr)
        s.stroke(_up([(xm, th), (x1 - 0.1, h), (x1, h * 0.9), (x1, th)]), rr)
    elif k == 1:    # arch from toe to heel
        s.stroke(_up([(x0, th), (xm, h), (x1, h * 0.8), (x1, th)]), rr)
        s.stroke(_up([(xm, h), (xm - 0.1, th)]), rr)
    elif k == 2:    # thong: triangle
        s.stroke(_up([(x0, th), (xm, h), (xm + 0.2, th)]), rr)
    else:           # closed vamp block + heel strap
        s.poly(_up([(x0, th), (x0 + 0.1, h * 0.7), (xm, h * 0.8), (xm + 0.05, th)]))
        s.stroke(_up([(xm, h * 0.7), (x1, h), (x1, th)]), rr)
    return s


def sandal_flat_soft(r):
    """Sole plus a soft zone where straps of unknown layout sit (low over the toes, higher towards the ankle)."""
    th = U(r, 0.05, 0.12)
    s = Scene()
    s.poly(_up([(0.0, 0.0), (0.0, th), (1.0, th * U(r, 1.0, 2.0)), (1.0, 0.0)]))
    h = U(r, 0.2, 0.46); xp = U(r, 0.5, 0.9); x0 = U(r, 0.03, 0.3)
    s.layer = 1; s.level = U(r, 0.3, 0.45)
    s.poly(_up([(x0, th), (0.5 * (x0 + xp), h * U(r, 0.5, 0.9)), (xp, h), (min(xp + 0.15, 1.0), h * U(r, 0.6, 1.0)),
                (U(r, 0.88, 1.0), th)]))
    return s


def sandal_chunky(r):
    """Chunky strap sandal: sole plus a mostly-filled mass over the middle of the foot; toe end and heel end stay low
    (no toe box, no heel counter), which is what separates it from a sneaker."""
    th = U(r, 0.06, 0.14)
    s = Scene()
    s.poly(_up([(0.0, 0.0), (0.0, th), (1.0, th * U(r, 1.0, 1.6)), (1.0, 0.0)]))
    x0 = U(r, 0.05, 0.3); x1 = U(r, 0.72, 0.9); h = U(r, 0.24, 0.48); xp = U(r, 0.45, 0.8)
    s.layer = 1; s.level = U(r, 0.55, 0.7)
    s.poly(_up([(x0, th), (x0 + 0.08, h * U(r, 0.5, 0.8)), (xp, h), (x1, h * U(r, 0.3, 0.7)), (x1 + 0.03, th)]))
    return s


def sandal_heel(r):
    hx = U(r, 0.72, 0.9); hy = U(r, 0.3, 0.75)    # heel seat
    bx = U(r, 0.25, 0.42)                         # ball of the foot
    th = U(r, 0.04, 0.2)                          # sole / platform thickness under the toes
    kind = r.integers(0, 3)                       # 0 thin heel, 1 block heel, 2 wedge
    s = Scene()
    sole = [(0.0, 0.0), (0.0, th), (bx, th), (hx, hy), (1.0, hy + U(r, 0, 0.06)), (1.0, hy - 0.08), (hx, hy - 0.1), (bx + 0.03, 0.0)]
    s.poly(_up(sole))
    hw = U(r, 0.05, 0.14) if kind == 0 else U(r, 0.14, 0.32)
    if kind == 2:
        s.poly(_up([(bx, 0.0), (hx, hy - 0.05), (1.0, hy - 0.05), (1.0, 0.0)]))
    else:
        s.poly(_up([(1.0 - hw, hy - 0.05), (1.0, hy - 0.05), (1.0, 0.0), (1.0 - hw * U(r, 0.5, 1.0), 0.0)]))
    ay = hy + U(r, 0.15, 0.45)                    # ankle strap height
    if r.random() < 0.5:        # straps of unknown layout: soft zone over the foot from the toes to the ankle
        if r.random() < 0.4:    # ankle cuff
            s.poly(_up([(U(r, 0.6, 0.75), ay), (1.0, ay), (1.0, ay - U(r, 0.08, 0.2)), (0.7, ay - 0.12)]))
        s.layer = 1; s.level = U(r, 0.3, 0.45)
        s.poly(_up([(U(r, 0.03, 0.15), th), (0.22, th + U(r, 0.1, 0.25)), (U(r, 0.5, 0.72), ay), (1.0, ay), (1.0, hy), (hx, hy), (bx, th)]))
        return s
    rr = U(r, 0.7, 1.6)
    s.stroke(_up([(U(r, 0.08, 0.2), th), (0.24, th + U(r, 0.08, 0.18)), (bx + 0.05, th + 0.02)]), rr)       # toe strap
    if r.random() < 0.8:                          # ankle strap
        xa = U(r, 0.55, 0.75)
        s.stroke(_up([(1.0 - 0.03, hy), (U(r, 0.86, 0.98), ay), (xa, ay - U(r, -0.05, 0.12))]), rr)
        if r.random() < 0.5:
            s.stroke(_up([(xa, ay - 0.05), (bx + 0.1, th + 0.1)]), rr)
    return s


# ------------------------------------------------------------------ bags (unit = body width, y downwards)
def bag(r, handle):
    hb = U(r, 0.45, 0.95)                         # body height
    tw = U(r, 0.8, 1.12)                          # top width relative to the middle
    bwid = U(r, 0.8, 1.0)
    s = Scene()
    c = U(r, 0.0, 0.1)
    s.poly([(-0.5 * tw + c, 0), (0.5 * tw - c, 0), (0.5 * tw, c), (0.5, hb * 0.5), (0.5 * bwid, hb - c), (0.5 * bwid - c, hb),
            (-0.5 * bwid + c, hb), (-0.5 * bwid, hb - c), (-0.5, hb * 0.5), (-0.5 * tw, c)])
    if handle:
        hx = U(r, 0.12, 0.40); hy = U(r, 0.15, 0.8); rr = U(r, 0.5, 1.2)
        a = np.linspace(0, np.pi, 9)
        pw = U(r, 0.6, 1.0)                       # < 1: more pointed arch
        s.stroke([(-hx * np.cos(t), 0.03 - hy * np.sin(t) ** pw) for t in a], rr)
    return s


# ============================== net.py ==============================
"""Turns the hand-typed shape models into a network: silhouette features -> bank of rendered prototypes -> best match per
shape group -> class scores from hand-written tables (+ a centre-line cue for zips / button plackets)."""
import json, os, sys, itertools
import numpy as np



SEED = 20240
GROUP = 60            # prototypes per shape group
UNIT = 4.0            # score units (multi-scale squared-distance units) given to a factor e in how typical a shape is
NEVER = -1.0e4        # score offset for a class that a shape group cannot be

# shape families: (model, attribute table name or None, groups per attribute combination, {class: offset} if no table)
FAMILIES = [
    (dome, "dome", 3, None),
    (short_top, "short", 12, None),
    (sleeveless_top, "sleeveless", 3, None),
    (trouser, None, 30, {1: 0.0}), (trouser_folded, None, 4, {1: -4.0}), (shorts, None, 4, {1: -6.0}),
    (lambda r: dress(r, "none"), None, 14, {3: 0.0}), (lambda r: dress(r, "sheath"), None, 10, {3: 0.0}),
    (lambda r: dress(r, "strap"), None, 10, {3: 0.0}), (lambda r: dress(r, "short"), None, 14, {3: 0.0}),
    (lambda r: dress(r, "long"), None, 12, {3: 1.5}),
    (sandal_flat, None, 14, {5: 0.0}), (sandal_flat_soft, None, 14, {5: 0.0}), (sandal_chunky, None, 16, {5: 0.0}),
    (sandal_heel, None, 36, {5: 0.0}),
    (sneaker, None, 48, {7: 0.0}), (hightop, None, 12, {7: -2.0}),
    (lambda r: bag(r, False), None, 26, {8: 0.0}), (lambda r: bag(r, True), None, 34, {8: 0.0}),
    (boot, None, 60, {9: 0.0}),
]
ZONE_HULL = 0.85     # value of the filled channels inside a soft zone

# silhouette threshold: grey level LO and below is background, HI and above is item
LO, HI = 0.07, 0.13
# weights of the three silhouette channels (mask, row-filled, column-filled) and of the comparison scales
CH_W = (1.0, 0.5, 0.5)
SCALE_W = (0.15, 0.6, 0.5)          # identity, Gaussian sigma 1, Gaussian sigma 2
# centre-line cue (zip / button placket): a thin vertical valley or ridge down the middle, 19 rows long.
# strength ramp in units of the image's standard deviation; class -> (offset, gain) in units of UNIT
LINE_T0, LINE_T1 = 0.10, 0.30
LINE_CLASS = {4: (-0.7, 2.6), 6: (-0.3, 1.5)}
LINE_VETO = 3.0        # a line that is an actual gap in the silhouette (trouser legs, open neck) does not count


def gauss_matrix(sigma):
    i = np.arange(G)
    g = np.exp(-0.5 * ((i[:, None] - i[None, :]) / sigma) ** 2)
    g /= np.exp(-0.5 * (np.arange(-30, 31) / sigma) ** 2).sum()
    return np.kron(g, g)             # zero-padded 2-D blur as a (784, 784) matrix


def hull_channels(M):
    """M (n, 28, 28) -> (n, 3, 28, 28): mask, row-filled (between leftmost and rightmost item pixel), column-filled."""
    r = lambda v: np.maximum(v, 0.0)
    uL = r(1 - np.cumsum(M, 2)); uR = r(1 - np.cumsum(M[:, :, ::-1], 2)[:, :, ::-1])
    uT = r(1 - np.cumsum(M, 1)); uB = r(1 - np.cumsum(M[:, ::-1], 1)[:, ::-1])
    return np.stack([M, r(1 - uL - uR), r(1 - uT - uB)], 1)


def features(scene, jit=None):
    """(3, 28, 28) prototype: silhouette, row-filled, column-filled; a soft zone gets partial values."""
    solid, zone = scene.render(jit=jit)
    f = hull_channels(solid[None])[0]
    if scene.level > 0:
        fz = hull_channels(np.maximum(solid, zone)[None])[0]
        f = np.maximum(f, np.stack([scene.level * zone, ZONE_HULL * fz[1], ZONE_HULL * fz[2]]))
    return f


def render_bank():
    """Returns prototypes (n, 3, 28, 28) laid out group by group, and the (groups, 10) table of class offsets."""
    rng = np.random.default_rng(SEED)
    feats, offs = [], []
    for fn, table, per, fixed in FAMILIES:
        if table is None:
            for _ in range(per):
                for _ in range(GROUP):
                    feats.append(features(fn(rng), rng.random(3)))
                o = np.full(10, NEVER)
                for k, v in fixed.items():
                    o[k] = v
                offs.append(o)
            continue
        names = [a for a in TOP_TABLES[table] if a != "share"]
        combos = list(itertools.product(*[list(TOP_TABLES[table][a]) for a in names]))
        need = per * GROUP
        got = {c: [] for c in combos}
        while min(len(v) for v in got.values()) < need:          # draw shapes until every attribute combination is filled
            sc = fn(rng)
            c = tuple(sc.attrs[a] for a in names)
            if len(got[c]) < need:
                got[c].append(sc)
        for c in combos:
            sco = table_scores(table, dict(zip(names, c)))
            best = max(sco.values())
            o = np.full(10, NEVER)
            for k, v in sco.items():
                o[k] = UNIT * (v - best)
            for gi in range(per):
                for sc in got[c][gi * GROUP:(gi + 1) * GROUP]:
                    feats.append(features(sc, rng.random(3)))
                offs.append(o)
    return np.array(feats), np.array(offs)


def build(out_dir):
    feats, offs = render_bank()
    N, NG = len(feats), len(offs)
    F = feats.reshape(N, 3, G * G)
    Q = SCALE_W[0] * np.eye(G * G)
    for w, sg in zip(SCALE_W[1:], (1.0, 2.0)):
        B = gauss_matrix(sg); Q = Q + w * (B @ B)
    QF = np.stack([CH_W[c] * (F[:, c] @ Q) for c in range(3)], 1)          # (n, 3, 784)
    Wt = (2.0 * QF).reshape(N, -1)
    bt = -(QF * F).sum((1, 2))
    W = {}
    layers = []

    def L(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=inputs, **kw)); return name

    # --- silhouette: m = clip((x - LO) / (HI - LO), 0, 1) = relu(a) - relu(a - 1)
    L("t", "conv", ["image"], out=2, kernel=[1, 1], stride=1, padding=0)
    W["t.W"] = np.full((2, 1, 1, 1), 1.0 / (HI - LO)); W["t.b"] = np.array([-LO / (HI - LO), -LO / (HI - LO) - 1.0])
    L("t_r", "relu", ["t"])
    # --- d1: mask copy and the four "nothing on this side yet" indicators u = relu(1 - running sum)
    n2 = G * G
    I = np.eye(G); T = np.tril(np.ones((G, G)))
    PL, PR, PT, PB = np.kron(I, T), np.kron(I, T.T), np.kron(T, I), np.kron(T.T, I)
    d1 = np.zeros((5 * n2, 2 * n2)); b1 = np.zeros(5 * n2)
    d1[:n2, :n2] = np.eye(n2); d1[:n2, n2:] = -np.eye(n2)
    for j, Pm in enumerate((PL, PR, PT, PB)):
        d1[(j + 1) * n2:(j + 2) * n2, :n2] = -Pm; d1[(j + 1) * n2:(j + 2) * n2, n2:] = Pm; b1[(j + 1) * n2:(j + 2) * n2] = 1.0
    L("d1", "dense", ["t_r"], out=5 * n2); W["d1.W"] = d1; W["d1.b"] = b1
    L("d1_r", "relu", ["d1"])
    # --- d2: mask, row-filled = relu(1 - uL - uR), column-filled = relu(1 - uT - uB)
    d2 = np.zeros((3 * n2, 5 * n2)); b2 = np.zeros(3 * n2)
    d2[:n2, :n2] = np.eye(n2)
    d2[n2:2 * n2, n2:2 * n2] = -np.eye(n2); d2[n2:2 * n2, 2 * n2:3 * n2] = -np.eye(n2); b2[n2:2 * n2] = 1.0
    d2[2 * n2:, 3 * n2:4 * n2] = -np.eye(n2); d2[2 * n2:, 4 * n2:] = -np.eye(n2); b2[2 * n2:] = 1.0
    L("d2", "dense", ["d1_r"], out=3 * n2); W["d2.W"] = d2; W["d2.b"] = b2
    L("d2_r", "relu", ["d2"])
    # --- prototype bank: score = -(multi-scale squared distance to the prototype), up to a term shared by all prototypes
    L("bank", "dense", ["d2_r"], out=N); W["bank.W"] = Wt; W["bank.b"] = bt
    # --- best prototype within each shape group
    L("bank_g", "reshape", ["bank"], shape=[1, NG, GROUP])
    L("grp", "maxpool", ["bank_g"], size=[1, GROUP], stride=[1, GROUP])                     # (1, NG, 1)
    # --- every class's score for every group: group score + table offset (row index = class * NG + group)
    cw = np.zeros((10 * NG, NG)); cb = np.zeros(10 * NG)
    for k in range(10):
        cw[k * NG:(k + 1) * NG] = np.eye(NG); cb[k * NG:(k + 1) * NG] = offs[:, k]
    L("cls", "dense", ["grp"], out=10 * NG); W["cls.W"] = cw; W["cls.b"] = cb
    # --- centre-line cue on the contrast-normalised image
    L("z", "norm", ["image"])
    K = 19
    k4 = np.zeros((4, 1, K, K))
    a = np.zeros((K, K)); a[:, 7] = 1.0 / K; a[:, 9] = -1.0 / K          # left neighbour minus centre, averaged over 19 rows
    b = np.zeros((K, K)); b[:, 11] = 1.0 / K; b[:, 9] = -1.0 / K         # right neighbour minus centre
    k4[0, 0], k4[1, 0], k4[2, 0], k4[3, 0] = a, -a, a - b, b - a
    L("ln1", "conv", ["z"], out=4, kernel=[K, K], stride=1, padding=0); W["ln1.W"] = k4; W["ln1.b"] = np.zeros(4)
    L("ln1_r", "relu", ["ln1"])
    # valley = relu(min(a, b)), ridge = relu(min(-a, -b))
    L("ln2", "conv", ["ln1_r"], out=2, kernel=[1, 1], stride=1, padding=0)
    W["ln2.W"] = np.array([[1, -1, -1, 0], [-1, 1, 0, -1]], float).reshape(2, 4, 1, 1); W["ln2.b"] = np.zeros(2)
    L("ln2_r", "relu", ["ln2"])
    L("ln3", "conv", ["ln2_r"], out=1, kernel=[1, 1], stride=1, padding=0); W["ln3.W"] = np.ones((1, 2, 1, 1)); W["ln3.b"] = np.zeros(1)
    # the same detector on the silhouette m = t_r[0] - t_r[1]: where the "line" is a real gap, cancel it
    km = np.concatenate([k4, -k4], 1)
    L("lm1", "conv", ["t_r"], out=4, kernel=[K, K], stride=1, padding=0); W["lm1.W"] = km; W["lm1.b"] = np.zeros(4)
    L("lm1_r", "relu", ["lm1"])
    L("lm2", "conv", ["lm1_r"], out=2, kernel=[1, 1], stride=1, padding=0); W["lm2.W"] = W["ln2.W"].copy(); W["lm2.b"] = np.zeros(2)
    L("lm2_r", "relu", ["lm2"])
    L("lm3", "conv", ["lm2_r"], out=1, kernel=[1, 1], stride=1, padding=0); W["lm3.W"] = -LINE_VETO * np.ones((1, 2, 1, 1)); W["lm3.b"] = np.zeros(1)
    L("ln4", "add", ["ln3", "lm3"])
    L("ln_p", "maxpool", ["ln4"], size=[3, 4], stride=[1, 1])              # (1, 8, 7)
    # window: kernel rows starting at 6..8 (below any neck opening), kernel-centre columns 12..15  -> pooled position (6, 3)
    c1 = np.zeros((2, 8 * 7)); c1[:, 6 * 7 + 3] = 1.0 / (LINE_T1 - LINE_T0)
    L("cue1", "dense", ["ln_p"], out=2); W["cue1.W"] = c1
    W["cue1.b"] = np.array([-LINE_T0 / (LINE_T1 - LINE_T0), -LINE_T0 / (LINE_T1 - LINE_T0) - 1.0])
    L("cue1_r", "relu", ["cue1"])
    c2 = np.zeros((10 * NG, 2)); c2b = np.zeros(10 * NG)
    for kcls, (off, gain) in LINE_CLASS.items():
        c2[kcls * NG:(kcls + 1) * NG, 0] = gain * UNIT; c2[kcls * NG:(kcls + 1) * NG, 1] = -gain * UNIT
        c2b[kcls * NG:(kcls + 1) * NG] = off * UNIT
    L("cue2", "dense", ["cue1_r"], out=10 * NG); W["cue2.W"] = c2; W["cue2.b"] = c2b
    # --- best group per class
    L("sum", "add", ["cls", "cue2"])
    L("rs", "reshape", ["sum"], shape=[1, 10, NG])
    L("mx", "maxpool", ["rs"], size=[1, NG], stride=[1, NG])
    L("out", "dense", ["mx"], out=10); W["out.W"] = np.eye(10); W["out.b"] = np.zeros(10)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k_: v.astype(np.float32) for k_, v in W.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
