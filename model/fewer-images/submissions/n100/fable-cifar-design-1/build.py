"""build.py - writes network.json and weights.npz for a hand-designed 32x32 photo classifier.

Nothing here is trained or fitted, and no file is read.  The network is
  (1) fixed feature layers written from formulas (soft colour-category steps, Gabor edge energy, colour saliency),
  (2) a bank of templates: each template is ONE procedurally drawn scene (sky, grass, water, a car made of
      boxes, a horse made of ellipses ...) typed from general knowledge of what the ten classes look like,
      pushed through the feature layers and stored as its own template, normalised by itself alone,
  (3) a fixed read-out: a photo's distances to all templates are standardised by the network itself (norm op),
      the closest templates vote for the class whose drawing routine produced them.
Usage: python3 build.py OUT_DIR
"""
import sys, os, json
import numpy as np

# =====================================================================================================
# 1. Drawing engine (numpy only).  Canvas is S x S, coordinates are in [0,1], x to the right, y downwards.
# =====================================================================================================
S = 64
PXc = (np.arange(S) + 0.5) / S
PY2, PX2 = np.meshgrid(PXc, PXc, indexing="ij")


def c3(*v):
    return np.array(v, float)


def U(rng, a, b):
    return float(rng.uniform(a, b))


def pick(rng, items, p=None):
    if p is not None:
        p = np.asarray(p, float); p = p / p.sum()
    return items[int(rng.choice(len(items), p=p))]


def jit(rng, col, dv=0.10, dc=0.025):
    """A colour with a random brightness factor and a little random tint."""
    return np.clip(np.asarray(col, float) * np.exp(rng.normal(0, dv)) + rng.normal(0, dc, 3), 0, 1)


def mixc(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


_NC = {}


def _interp(c):
    if c not in _NC:
        t = PXc * c
        i = np.minimum(t.astype(int), c - 1)
        w = t - i
        _NC[c] = (i, w * w * (3 - 2 * w))
    return _NC[c]


def noise(rng, cy, cx=None):
    """Smooth random field (S,S): a (cy+1)x(cx+1) grid of normal numbers, smoothly interpolated."""
    cx = cx or cy
    g = rng.standard_normal((cy + 1, cx + 1))
    iy, wy = _interp(cy)
    ix, wx = _interp(cx)
    r0, r1 = g[iy], g[iy + 1]
    wx = wx[None, :]; wy = wy[:, None]
    return (r0[:, ix] * (1 - wx) + r0[:, ix + 1] * wx) * (1 - wy) + (r1[:, ix] * (1 - wx) + r1[:, ix + 1] * wx) * wy


def fnoise(rng, cells=(2, 4, 8, 16), amps=(1, .5, .25, .125)):
    """Fractal noise with roughly unit spread."""
    out = 0.0
    for c, a in zip(cells, amps):
        out = out + a * noise(rng, c)
    return out / (0.7 * np.sqrt(sum(a * a for a in amps)))


def line1d(rng, cells=4, amp=1.0):
    """Smooth random function of x, shape (S,)."""
    g = rng.standard_normal(cells + 1)
    i, w = _interp(cells)
    return amp * (g[i] * (1 - w) + g[i + 1] * w)


class Canvas:
    def __init__(self, rng):
        self.rng = rng
        self.img = np.zeros((S, S, 3))

    def _put(self, i0, i1, j0, j1, m, col, alpha=1.0, shade=None):
        reg = self.img[j0:j1, i0:i1]
        col = np.asarray(col, float)
        c = col[j0:j1, i0:i1][m] if col.ndim == 3 else col[None, :]
        if shade is not None:
            c = c * shade[j0:j1, i0:i1][m][:, None]
        if alpha >= 1.0:
            reg[m] = c
        else:
            reg[m] = reg[m] * (1 - alpha) + alpha * c

    @staticmethod
    def _bb(x0, x1, y0, y1):
        return (max(int(np.floor(x0 * S)), 0), min(int(np.ceil(x1 * S)), S),
                max(int(np.floor(y0 * S)), 0), min(int(np.ceil(y1 * S)), S))

    def poly(self, pts, col, alpha=1.0, shade=None):
        pts = np.asarray(pts, float)
        i0, i1, j0, j1 = self._bb(pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
        if i1 <= i0 or j1 <= j0:
            return
        x = PXc[i0:i1][None, :, None]; y = PXc[j0:j1][:, None, None]
        x1 = pts[:, 0]; y1 = pts[:, 1]; x2 = np.roll(x1, -1); y2 = np.roll(y1, -1)
        dy = y2 - y1
        dy = np.where(dy == 0, 1e-12, dy)
        cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / dy + x1)
        m = (cond.sum(-1) & 1).astype(bool)
        self._put(i0, i1, j0, j1, m, col, alpha, shade)

    def seg(self, a, b, r0, r1=None, col=(0, 0, 0), alpha=1.0, shade=None):
        """Thick line from a to b with radius r0 at a and r1 at b (round ends)."""
        r1 = r0 if r1 is None else r1
        rm = max(r0, r1)
        i0, i1, j0, j1 = self._bb(min(a[0], b[0]) - rm, max(a[0], b[0]) + rm, min(a[1], b[1]) - rm, max(a[1], b[1]) + rm)
        if i1 <= i0 or j1 <= j0:
            return
        x = PXc[i0:i1][None, :]; y = PXc[j0:j1][:, None]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy + 1e-12
        t = np.clip(((x - a[0]) * dx + (y - a[1]) * dy) / L2, 0, 1)
        d2 = (x - a[0] - t * dx) ** 2 + (y - a[1] - t * dy) ** 2
        m = d2 < (r0 + t * (r1 - r0)) ** 2
        self._put(i0, i1, j0, j1, m, col, alpha, shade)

    def path(self, pts, radii, col, alpha=1.0, shade=None):
        for k in range(len(pts) - 1):
            self.seg(pts[k], pts[k + 1], radii[k], radii[k + 1], col, alpha, shade)

    def ell(self, a, b, rmin, col, alpha=1.0, shade=None):
        """Ellipse whose major axis runs from point a to point b, with half-width rmin across."""
        cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx, dy = (b[0] - a[0]) / 2, (b[1] - a[1]) / 2
        rmaj = np.hypot(dx, dy) + 1e-9
        rm = max(rmaj, rmin)
        i0, i1, j0, j1 = self._bb(cx - rm, cx + rm, cy - rm, cy + rm)
        if i1 <= i0 or j1 <= j0:
            return
        x = PXc[i0:i1][None, :] - cx; y = PXc[j0:j1][:, None] - cy
        ux, uy = dx / rmaj, dy / rmaj
        p = (x * ux + y * uy) / rmaj; q = (-x * uy + y * ux) / rmin
        self._put(i0, i1, j0, j1, p * p + q * q < 1, col, alpha, shade)

    def circ(self, c, r, col, alpha=1.0, shade=None):
        self.ell((c[0] - r, c[1]), (c[0] + r, c[1]), r, col, alpha, shade)

    def rect(self, x0, y0, x1, y1, col, alpha=1.0, shade=None):
        i0, i1, j0, j1 = self._bb(x0, x1, y0, y1)
        if i1 <= i0 or j1 <= j0:
            return
        self._put(i0, i1, j0, j1, np.ones((j1 - j0, i1 - i0), bool), col, alpha, shade)

    def where(self, m, layer, alpha=1.0):
        """Paste a full-canvas layer (S,S,3) or a colour where the mask is true."""
        self._put(0, S, 0, S, m, layer, alpha)


class Frame:
    """Places a flat (2-D) object: local (u, v) with u forwards and v downwards -> canvas coordinates."""

    def __init__(self, cx, cy, s, ang=0.0, flip=1):
        self.cx, self.cy, self.s, self.flip = cx, cy, s, flip
        self.c, self.sn = np.cos(ang), np.sin(ang)

    def p(self, u, v):
        u = u * self.flip
        return (self.cx + self.s * (self.c * u - self.sn * v), self.cy + self.s * (self.sn * u + self.c * v))

    def pp(self, pts):
        return [self.p(u, v) for u, v in pts]


def rot3(yaw, pitch, roll=0.0):
    """Object axes: x forwards, y up, z to the object's right. Camera axes: X right, Y up, Z towards the viewer.
    pitch > 0 looks down on the object from above."""
    cy, sy = np.cos(yaw), np.sin(yaw); cp, sp = np.cos(pitch), np.sin(pitch); cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


class Obj3:
    """A solid made of flat faces, tubes and wheels, drawn back to front in a simple orthographic view."""

    def __init__(self):
        self.it = []

    def face(self, pts, col, n=None, bias=0.0, alpha=1.0):
        """Flat polygon. n: outward normal (hidden when it points away from the viewer); None = always drawn."""
        self.it.append(("p", np.asarray(pts, float), np.asarray(col, float), None if n is None else np.asarray(n, float), bias, alpha))
        return len(self.it) - 1

    def quad(self, ax, c, a0, a1, b0, b1, col, sgn, bias=0.0, alpha=1.0):
        """Rectangle on the plane ax = c.  ax 'x': (a, b) = (y, z);  'y': (x, z);  'z': (x, y)."""
        if ax == "x":
            pts, n = [(c, a0, b0), (c, a1, b0), (c, a1, b1), (c, a0, b1)], (sgn, 0, 0)
        elif ax == "y":
            pts, n = [(a0, c, b0), (a1, c, b0), (a1, c, b1), (a0, c, b1)], (0, sgn, 0)
        else:
            pts, n = [(a0, b0, c), (a1, b0, c), (a1, b1, c), (a0, b1, c)], (0, 0, sgn)
        return self.face(pts, col, n, bias, alpha)

    def tube(self, p0, p1, r0, r1, col, bias=0.0):
        self.it.append(("t", np.array([p0, p1], float), np.asarray(col, float), (r0, r1), bias, 1.0))

    def box(self, lo, hi, col, bias=0.0, cols=None):
        """Axis-aligned box. cols may override the colour of faces by name: +x -x +y -y +z -z."""
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        cols = cols or {}
        F = {"+x": ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0)),
             "-x": ([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], (-1, 0, 0)),
             "+y": ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0)),
             "-y": ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0)),
             "+z": ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1)),
             "-z": ([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], (0, 0, -1))}
        for k, (pts, n) in F.items():
            self.face(pts, cols.get(k, col), n, bias)

    def prism(self, prof, z0, z1, col, ecol=None, bias=0.0, up=False):
        """Outline prof [(a, b), ...] pushed out along the third axis from z0 to z1.
        up=False: outline lies in the (x, y) side plane and is pushed along z (width).
        up=True : outline lies in the (x, z) ground plane and is pushed along y (height)."""
        prof = np.asarray(prof, float)
        area = 0.5 * np.sum(prof[:, 0] * np.roll(prof[:, 1], -1) - np.roll(prof[:, 0], -1) * prof[:, 1])
        if area < 0:
            prof = prof[::-1]
            if ecol:
                K = len(prof); ecol = {((K - 2 - k) % K if isinstance(k, int) else k): v for k, v in ecol.items()}
        ecol = ecol or {}

        def P(a, b, c):
            return (a, c, b) if up else (a, b, c)

        K = len(prof)
        for k in range(K):
            (a0, b0), (a1, b1) = prof[k], prof[(k + 1) % K]
            n = P(b1 - b0, -(a1 - a0), 0.0)
            nn = np.array(n) / (np.linalg.norm(n) + 1e-12)
            self.face([P(a0, b0, z0), P(a1, b1, z0), P(a1, b1, z1), P(a0, b0, z1)], ecol.get(k, col), nn, bias)
        self.face([P(a, b, z1) for a, b in prof], ecol.get("top", col), P(0, 0, 1), bias)
        self.face([P(a, b, z0) for a, b in prof], ecol.get("bot", col), P(0, 0, -1), bias)

    def wheel(self, c, r, wid, sgn, tyre, hub):
        """Wheel with its axle along z; c is the centre of its outer face, sgn says which side is outward."""
        self.it.append(("w", np.array([c, (c[0], c[1], c[2] - sgn * wid)], float), (np.asarray(tyre, float), np.asarray(hub, float)), (r, sgn), 0.0, 1.0))

    def _proj(self, R):
        out = []
        for kind, pts, col, extra, bias, alpha in self.it:
            out.append(pts @ R.T)
        return out

    def bbox(self, R):
        lo = np.array([1e9, 1e9]); hi = -lo
        for (kind, pts, col, extra, bias, alpha), P in zip(self.it, self._proj(R)):
            rad = 0.0
            if kind == "t":
                rad = max(extra)
            elif kind == "w":
                rad = extra[0]
            lo = np.minimum(lo, P[:, :2].min(0) - rad); hi = np.maximum(hi, P[:, :2].max(0) + rad)
        return lo, hi

    def fit(self, R, tx, ty, tw, th):
        """Scale and centre so the drawn object just fits a tw x th box centred on (tx, ty)."""
        lo, hi = self.bbox(R)
        s = min(tw / (hi[0] - lo[0]), th / (hi[1] - lo[1]))
        return s, tx - s * (lo[0] + hi[0]) / 2, ty + s * (lo[1] + hi[1]) / 2

    def render(self, sc, R, s, cx, cy, light=(0.3, 0.8, 0.5), amb=0.62, dif=0.5, shade=None):
        L = np.asarray(light, float); L = L / np.linalg.norm(L)
        todo = []
        for (kind, pts, col, extra, bias, alpha), P in zip(self.it, self._proj(R)):
            depth = P[:, 2].mean() + bias
            if kind == "p":
                f = 1.0
                if extra is not None:
                    nc = R @ extra
                    if nc[2] <= 0.03:
                        continue
                    f = amb + dif * max(0.0, float(nc @ L))
                todo.append((depth, "p", P, np.clip(col * f, 0, 1), None, alpha))
            elif kind == "t":
                todo.append((depth, "t", P, col, extra, alpha))
            else:
                r, sgn = extra
                nz = (R @ np.array([0, 0, sgn]))[2]
                todo.append((depth + (0.6 if nz > 0.3 else -3.0), "w", P, col, (r, nz), alpha))
        todo.sort(key=lambda t: t[0])
        X = lambda P: np.stack([cx + s * P[:, 0], cy - s * P[:, 1]], 1)
        for depth, kind, P, col, extra, alpha in todo:
            if kind == "p":
                sc.poly(X(P), col, alpha, shade)
            elif kind == "t":
                q = X(P)
                sc.seg(q[0], q[1], s * extra[0], s * extra[1], col, alpha, shade)
            else:
                r, nz = extra
                ang = np.linspace(0, 2 * np.pi, 14, endpoint=False)
                ring = np.stack([np.cos(ang) * r, np.sin(ang) * r, np.zeros(14)], 1) @ R.T
                ax = P[1] - P[0]
                e = np.cross(ax, [0, 0, 1.0]); e = e / (np.linalg.norm(e) + 1e-9) * r
                tyre, hub = col
                sc.poly(X(np.array([P[0] + e, P[1] + e, P[1] - e, P[0] - e])), tyre)
                sc.poly(X(P[1] + ring), tyre)
                sc.poly(X(P[0] + ring), tyre)
                if nz > 0.05:
                    sc.poly(X(P[0] + ring * 0.55), hub)


def finish(rng, img, blur=None):
    """Canvas -> a 32x32 'photo': shrink, soften, random exposure / contrast / colour cast, a little grain."""
    im = img.reshape(32, 2, 32, 2, 3).mean((1, 3))
    sig = U(rng, 0.25, 0.6) if blur is None else blur
    k = np.exp(-0.5 * (np.arange(-2, 3) / sig) ** 2); k /= k.sum()
    pad = np.pad(im, ((2, 2), (0, 0), (0, 0)), mode="edge"); im = sum(k[i] * pad[i:i + 32] for i in range(5))
    pad = np.pad(im, ((0, 0), (2, 2), (0, 0)), mode="edge"); im = sum(k[i] * pad[:, i:i + 32] for i in range(5))
    im = np.clip(im, 0, 1) ** U(rng, 0.8, 1.3)
    g = im.mean(2, keepdims=True)
    im = g + (im - g) * U(rng, 0.55, 1.05)                       # real photos are often less colourful than paint
    m = im.mean()
    im = (im - m) * U(rng, 0.8, 1.12) + m * U(rng, 0.85, 1.12)
    warm = rng.normal(0, 0.03)                                  # colour cast: mostly warm <-> cool
    im = im * (1 + np.array([warm, 0.2 * warm, -warm]) + rng.normal(0, 0.008, 3)) + rng.normal(0, 0.008, 3)
    im = im + rng.normal(0, U(rng, 0.004, 0.02), im.shape)
    return np.clip(im, 0, 1).transpose(2, 0, 1)


# =====================================================================================================
# 2. Backgrounds, typed from general knowledge of what skies, grass, roads, water and rooms look like.
# =====================================================================================================
def tex(rng, col, amp, cells=(3, 8, 24), amps=(1, .7, .5), col2=None, mix=0.5, grain=0.0):
    """A textured surface: colour col with brightness noise, drifting towards col2 in patches; grain = pixel-scale detail."""
    f = 1 + amp * fnoise(rng, cells, amps)
    if grain:
        f = f + grain * (noise(rng, 32) + .7 * rng.standard_normal((S, S)))
    f = np.clip(f, .15, 2.5)[..., None]
    img = np.asarray(col, float)[None, None, :] * f
    if col2 is not None:
        t = (np.clip(0.5 + 0.6 * fnoise(rng, (2, 5), (1, .6)), 0, 1) * mix)[..., None]
        img = img * (1 - t) + np.asarray(col2, float) * t * f
    return np.clip(img, 0, 1)


SKY = ["blue", "blue", "deep", "pale", "pale", "haze", "haze", "over", "over", "white", "sunset"]
SKY_AIR = ["blue", "deep", "pale", "pale", "haze", "haze", "haze", "over", "over", "over", "white", "white", "sunset", "dusk"]


def sky(rng, kind=None):
    kind = kind or pick(rng, SKY)
    if kind == "blue":
        top, bot = jit(rng, (.25, .45, .80), .08), jit(rng, (.62, .76, .92), .05)
    elif kind == "deep":
        top, bot = jit(rng, (.12, .28, .68), .10), jit(rng, (.40, .60, .88), .06)
    elif kind == "pale":
        top, bot = jit(rng, (.55, .70, .90), .06), jit(rng, (.84, .89, .95), .04)
    elif kind == "over":
        g = U(rng, .55, .88)
        top = jit(rng, (g * .98, g, g * 1.03), .03, .008); bot = np.clip(top * U(rng, .95, 1.12), 0, 1)
    elif kind == "white":
        top = jit(rng, (.93, .94, .96), .03, .01); bot = top
    elif kind == "haze":
        top, bot = jit(rng, (.50, .60, .75), .08, .015), jit(rng, (.76, .80, .85), .05, .015)
    elif kind == "sunset":
        top, bot = jit(rng, (.35, .33, .50), .15), jit(rng, (.95, .62, .32), .08)
    else:  # dusk / storm
        top, bot = jit(rng, (.14, .17, .28), .15), jit(rng, (.42, .42, .50), .1)
    img = top + (bot - top) * (PY2 ** U(rng, .7, 1.5))[..., None]
    if kind in ("blue", "deep", "pale") and rng.random() < .5:
        m = (np.clip((fnoise(rng, (2, 4, 8), (1, .6, .3)) + U(rng, -1.2, .3)) * U(rng, 1, 3), 0, 1) * U(rng, .5, 1))[..., None]
        img = img * (1 - m) + jit(rng, (.95, .95, .97), .04, .01) * m
    elif kind in ("over", "dusk"):
        img = img * (1 + .06 * fnoise(rng, (2, 4, 8)))[..., None]
    return np.clip(img, 0, 1)


GRAIN = {"grass": .22, "dry": .18, "dirt": .10, "sand": .06, "asph": .07, "conc": .05, "snow": .03, "litter": .22, "rock": .14, "mud": .10}
GROUND = {  # name: (main colours, texture strength, patch colours)
    "grass": ([(.30, .46, .16), (.36, .50, .20), (.24, .38, .13), (.42, .52, .25), (.33, .42, .18), (.28, .36, .18),
               (.38, .42, .24), (.22, .28, .14), (.40, .40, .22)], .20,
              [(.50, .55, .25), (.20, .32, .10), (.45, .45, .20), (.3, .3, .15), (.5, .48, .3)]),
    "dry": ([(.62, .56, .36), (.55, .50, .30), (.70, .62, .42), (.50, .45, .28)], .14, [(.45, .45, .25), (.70, .65, .50)]),
    "dirt": ([(.52, .40, .28), (.45, .34, .24), (.60, .48, .36), (.56, .46, .36)], .12, [(.40, .30, .20), (.62, .55, .45)]),
    "sand": ([(.80, .70, .52), (.72, .63, .48), (.85, .78, .62)], .06, [(.70, .60, .42)]),
    "asph": ([(.40, .40, .42), (.50, .50, .50), (.32, .32, .34), (.58, .57, .56), (.45, .44, .43)], .06,
             [(.50, .50, .50), (.35, .35, .36)]),
    "conc": ([(.70, .69, .66), (.62, .60, .58), (.78, .77, .74)], .05, [(.60, .60, .58)]),
    "snow": ([(.90, .92, .95), (.82, .85, .90), (.95, .95, .96)], .04, [(.80, .84, .90)]),
    "litter": ([(.42, .32, .20), (.35, .27, .18), (.50, .40, .28), (.38, .33, .22)], .32,
               [(.25, .20, .12), (.60, .50, .35), (.30, .33, .15)]),
    "rock": ([(.50, .48, .45), (.40, .39, .37), (.60, .57, .52)], .16, [(.35, .34, .32), (.62, .60, .56)]),
    "mud": ([(.33, .26, .18), (.28, .24, .18), (.40, .33, .24)], .14, [(.20, .16, .10)]),
}
WATER = [(.20, .38, .58), (.30, .45, .56), (.15, .25, .42), (.33, .48, .50), (.50, .56, .62), (.25, .42, .66),
         (.38, .50, .60), (.20, .30, .32), (.12, .20, .30), (.45, .60, .70), (.40, .45, .48), (.28, .36, .40),
         (.55, .62, .66), (.18, .26, .28), (.35, .42, .38)]


def water(rng, gy=0.0, col=None):
    col = jit(rng, col if col is not None else pick(rng, WATER), .10, .02)
    far = mixc(col, (.75, .82, .88), U(rng, .0, .5))
    g = np.clip((PY2 - gy) / max(1 - gy, 1e-3), 0, 1)[..., None]
    img = far * (1 - g) + col * g
    rip = .6 * noise(rng, 28, 4) + .5 * noise(rng, 14, 3) + .4 * noise(rng, 5, 5)
    return np.clip(img * (1 + U(rng, .04, .16) * rip)[..., None], 0, 1)


def ground(rng, kind, gy=0.0):
    if kind == "water":
        return water(rng, gy)
    cols, amp, c2 = GROUND[kind]
    cells = (4, 10, 28) if kind in ("grass", "dry", "litter", "rock") else (3, 8, 20)
    dc = .008 if kind in ("asph", "conc", "snow", "rock") else .02
    return tex(rng, jit(rng, pick(rng, cols), .12, dc), amp * U(rng, .6, 1.3), cells, (1, .8, .6),
               jit(rng, pick(rng, c2), .12, dc), U(rng, .2, .7), GRAIN[kind] * U(rng, .5, 1.5))


def foliage(rng):
    base = jit(rng, pick(rng, [(.18, .30, .12), (.24, .36, .15), (.14, .24, .10), (.30, .40, .20), (.22, .30, .18)]), .12, .02)
    c2 = jit(rng, pick(rng, [(.45, .55, .28), (.08, .12, .06), (.35, .40, .15), (.40, .35, .20), (.1, .14, .08)]), .1)
    return tex(rng, base, U(rng, .25, .45), (4, 9, 20), (1, .8, .5), c2, U(rng, .3, .7), U(rng, .1, .4))


def bokeh(rng, pal=None):
    """Out-of-focus background: two or three colours in big soft blobs."""
    pal = pal or [(.30, .45, .18), (.45, .52, .25), (.18, .28, .12), (.55, .50, .35), (.40, .33, .22), (.60, .62, .55),
                  (.50, .65, .85), (.70, .72, .70), (.35, .38, .30), (.62, .58, .30), (.25, .22, .18)]
    a, b, c = (jit(rng, pick(rng, pal), .12, .03) for _ in range(3))
    t = np.clip(.5 + .55 * fnoise(rng, (2, 3, 6), (1, .7, .3)), 0, 1)[..., None]
    img = a * (1 - t) + b * t
    t = (np.clip(.2 + .5 * fnoise(rng, (2, 4), (1, .5)), 0, 1) * U(rng, 0, .8))[..., None]
    img = img * (1 - t) + c * t
    if rng.random() < .5:
        img = img * (1 + U(rng, .05, .25) * fnoise(rng, (6, 12, 24), (1, .8, .6)))[..., None]
    return np.clip(img, 0, 1)


def plain(rng, col, amp=.06):
    """Studio backdrop: one colour with a soft lighting gradient."""
    return np.clip(np.asarray(col, float) * (1 + amp * fnoise(rng, (1, 2), (1, .5)))[..., None], 0, 1)


def outdoor(rng, gy, sk=None, back=None, gk="grass", bh=None):
    """Sky on top, optional backdrop (trees, forest, hills, buildings, wall) standing on the ground line gy,
    ground below gy."""
    img = sky(rng, sk)
    land = PY2 > gy
    bh = U(rng, .08, .4) if bh is None else bh
    if back in ("trees", "forest"):
        fo = foliage(rng)
        if back == "forest":
            top = np.full(S, -1.0) if rng.random() < .7 else gy - bh * 2 - np.abs(line1d(rng, 6, .1))
            for _ in range(int(rng.integers(2, 7))):
                x = U(rng, 0, 1); w = U(rng, .012, .05)
                tc = jit(rng, pick(rng, [(.22, .17, .12), (.35, .30, .25), (.12, .10, .08), (.5, .47, .42)]), .1)
                m = np.abs(PX2 - x - .03 * (PY2 - .5) * rng.normal()) < w
                fo[m] = tc * (1 + .15 * noise(rng, 12, 2))[m][:, None]
        else:
            top = gy - bh * (0.6 + 0.4 * np.abs(line1d(rng, 5))) - .03 * line1d(rng, 16)
        m = PY2 > top[None, :]
        img[m] = fo[m]; land = land | m
    elif back == "hills":
        top = gy - bh * (0.35 + 0.65 * np.clip(.5 + .5 * line1d(rng, 3), 0, 1.3))
        hc = jit(rng, pick(rng, [(.30, .38, .42), (.25, .35, .20), (.45, .42, .35), (.40, .48, .55), (.2, .26, .2)]), .1)
        m = PY2 > top[None, :]
        img[m] = tex(rng, hc, .08, (3, 8), (1, .5))[m]; land = land | m
    elif back == "buildings":
        x = U(rng, -.2, 0.05)
        while x < 1:
            w = U(rng, .12, .45); h = bh * U(rng, .5, 2.2)
            bc = jit(rng, pick(rng, [(.70, .68, .64), (.55, .40, .32), (.80, .76, .68), (.45, .45, .47), (.85, .85, .85),
                                     (.60, .55, .50), (.35, .30, .28)]), .1)
            m = (PX2 >= x) & (PX2 < x + w) & (PY2 > gy - h)
            img[m] = bc
            if rng.random() < .6:   # rows of dark windows
                wy = ((PY2 - gy) * U(rng, 10, 18)) % 1 < .45
                wx = ((PX2 - x) * U(rng, 10, 18)) % 1 < .5
                img[m & wy & wx] = bc * U(rng, .35, .7)
            x += w
    elif back == "wall":
        wc = jit(rng, pick(rng, [(.75, .72, .66), (.55, .42, .32), (.85, .85, .83), (.40, .38, .36), (.60, .30, .22),
                                 (.50, .52, .50), (.30, .22, .16), (.68, .6, .45)]), .1)
        img = tex(rng, wc, .07, (2, 6, 14), (1, .5, .4)); land = land | True
    m = PY2 > gy
    img[m] = ground(rng, gk, gy)[m]
    if rng.random() < .6:
        img[land] = (img * (1 + U(rng, .05, .3) * fnoise(rng, (2, 4, 9), (1, .6, .4)))[..., None])[land]
    return np.clip(img, 0, 1)


def clutter(sc, rng, y0, y1, n):
    """Small things standing in the background: signs, posts, people, other cars, as plain little blocks."""
    for _ in range(n):
        col = jit(rng, pick(rng, PAINT + WALL + [(.1, .1, .1)] * 3), .15, .03)
        x, y, w, h = U(rng, 0, 1), U(rng, y0, y1), U(rng, .02, .16), U(rng, .03, .2)
        sc.rect(x - w / 2, y - h, x + w / 2, y, col)


FABRIC = [(.50, .20, .20), (.25, .30, .45), (.30, .40, .30), (.70, .65, .55), (.20, .15, .12), (.80, .78, .72),
          (.55, .45, .35), (.60, .60, .62), (.35, .25, .40), (.75, .60, .30), (.15, .20, .30), (.85, .45, .45),
          (.40, .50, .60), (.9, .88, .85), (.3, .3, .32), (.65, .5, .4)]
WALL = [(.85, .80, .70), (.90, .90, .88), (.65, .55, .45), (.75, .78, .80), (.80, .70, .60), (.55, .60, .55),
        (.40, .35, .30), (.90, .85, .75), (.70, .72, .65), (.25, .22, .2), (.6, .62, .66)]
FLOOR = [(.55, .38, .22), (.62, .55, .48), (.40, .28, .18), (.70, .66, .60), (.45, .45, .45), (.30, .22, .16),
         (.75, .70, .60), (.50, .30, .25), (.8, .78, .74), (.35, .33, .3)]


def indoor(rng, gy=None):
    """A room: wall above, floor / sofa / bed below, a few big soft patches of fabric, uneven flash lighting."""
    gy = U(rng, .25, .8) if gy is None else gy
    img = tex(rng, jit(rng, pick(rng, WALL), .12, .03), .05, (2, 5), (1, .5), grain=U(rng, 0, .06))
    low = tex(rng, jit(rng, pick(rng, FLOOR + FABRIC), .12, .03), U(rng, .04, .14), (3, 8, 20), (1, .6, .5), grain=U(rng, .02, .2))
    edge = gy + .06 * line1d(rng, 2) + U(rng, -.25, .25) * (PXc - .5)
    m = PY2 > edge[None, :]
    img[m] = low[m]
    for _ in range(int(rng.integers(0, 4))):     # cushions, blankets, furniture
        col = jit(rng, pick(rng, FABRIC + WALL), .15, .03)
        f = fnoise(rng, (1, 2, 4), (1, .7, .3)) + 1.2 * rng.normal()
        m = f > U(rng, .5, 1.2)
        img[m] = (col * (1 + .08 * noise(rng, 10) + U(rng, 0, .15) * noise(rng, 32))[..., None])[m]
    if rng.random() < .3:                         # straight furniture edge / door frame / window
        x0 = U(rng, -.1, .9); col = jit(rng, pick(rng, FLOOR + WALL + [(.95, .95, .95)]), .15, .03)
        m = (PX2 > x0) & (PX2 < x0 + U(rng, .1, .5)) & (PY2 < U(rng, .2, 1.))
        img[m] = col
    light = (1 + U(rng, .0, .3) * fnoise(rng, (1, 2), (1, .5))) * (U(rng, .45, .9) if rng.random() < .35 else 1.0)
    return np.clip(img * light[..., None], 0, 1)


def litterbg(rng, kind=None):
    """Close-up natural ground: leaf litter, soil, moss, rock, a big green leaf, dark pond water."""
    kind = kind or pick(rng, ["litter", "litter", "soil", "moss", "rock", "leaf", "leaf", "pond", "sandy"])
    if kind == "litter":
        img = ground(rng, "litter")
        for _ in range(int(rng.integers(3, 12))):   # single leaves and twigs
            col = jit(rng, pick(rng, [(.55, .42, .25), (.30, .22, .14), (.65, .55, .38), (.35, .38, .18), (.2, .16, .12)]), .12)
            f = noise(rng, int(rng.integers(3, 7))) + .6 * noise(rng, 9)
            m = f > U(rng, 1.1, 1.8)
            img[m] = col
    elif kind == "soil":
        img = ground(rng, pick(rng, ["dirt", "mud", "dirt"]))
    elif kind == "sandy":
        img = ground(rng, pick(rng, ["sand", "dry"]))
    elif kind == "moss":
        img = tex(rng, jit(rng, pick(rng, [(.28, .38, .14), (.35, .42, .18), (.22, .30, .12)]), .1), .3, (4, 10, 26), (1, .8, .7),
                  jit(rng, pick(rng, [(.4, .34, .22), (.15, .18, .1), (.5, .52, .3)]), .1), .6)
    elif kind == "rock":
        img = ground(rng, "rock")
    elif kind == "leaf":
        col = jit(rng, pick(rng, [(.30, .55, .15), (.25, .45, .14), (.40, .60, .20), (.2, .38, .14), (.45, .58, .3)]), .1)
        img = tex(rng, col, .08, (2, 5), (1, .5), jit(rng, (.5, .65, .3), .1), .4)
        a = U(rng, 0, np.pi)
        d = (PX2 - U(rng, .2, .8)) * np.cos(a) + (PY2 - U(rng, .2, .8)) * np.sin(a)
        img[np.abs(d) < .012] *= 1.25                # midrib
        if rng.random() < .5:
            m = d > U(rng, .15, .5)
            img[m] = (bokeh(rng) * U(rng, .3, 1))[m]
    else:  # pond / night
        img = tex(rng, jit(rng, pick(rng, [(.10, .13, .10), (.16, .2, .18), (.08, .08, .08), (.2, .22, .15)]), .2), .25, (3, 8), (1, .6),
                  jit(rng, pick(rng, [(.3, .4, .2), (.3, .32, .3), (.2, .25, .3)]), .1), .5)
    return np.clip(img, 0, 1)


# =====================================================================================================
# 3. Vehicles, drawn as simple solids (boxes, pushed-out side outlines, wheels) seen from a random angle.
# =====================================================================================================
PAINT = [(.75, .10, .08), (.72, .73, .75), (.93, .93, .93), (.08, .08, .09), (.12, .25, .60), (.08, .12, .30),
         (.92, .78, .10), (.10, .35, .20), (.40, .40, .42), (.70, .62, .45), (.90, .45, .10), (.40, .08, .10), (.55, .65, .78)]
#        red  silver white black blue navy yellow green grey beige orange maroon light-blue
CAR_P = [.20, .16, .13, .12, .09, .05, .06, .03, .07, .04, .02, .03, .03]
TRUCK_P = [.22, .08, .24, .06, .12, .03, .10, .05, .03, .01, .06, .02, .04]
TYRE = (.07, .07, .07)


def glasscol(rng):
    return jit(rng, pick(rng, [(.12, .15, .20), (.20, .26, .32), (.40, .50, .60), (.08, .08, .10), (.30, .35, .40)]), .1)


def veh_view(rng, p_side, p_q_front, p_front, p_q_rear):
    """Random viewing direction (yaw) for a vehicle whose nose points along +x."""
    r = rng.random()
    d = np.pi / 180
    if r < p_side:
        yaw = rng.normal(0, 9) * d
    elif r < p_side + p_q_front:
        yaw = -U(rng, 22, 68) * d
    elif r < p_side + p_q_front + p_front:
        yaw = -U(rng, 72, 100) * d
    elif r < p_side + p_q_front + p_front + p_q_rear:
        yaw = U(rng, 22, 68) * d
    else:
        yaw = U(rng, 72, 100) * d
    return yaw if rng.random() < .5 else np.pi - yaw


def make_car(rng):
    o = Obj3()
    col = jit(rng, pick(rng, PAINT, CAR_P), .08, .02)
    glass = glasscol(rng)
    typ = pick(rng, ["sedan", "sedan", "hatch", "van", "sport", "wagon"])
    j = lambda v, a=.025: v + rng.normal(0, a)
    if typ == "sedan":
        prof = [(-1, .14), (-1, j(.42)), (-.62, j(.46)), (j(-.40), .66), (j(.12), .66), (.45, j(.46)), (.95, j(.40)), (1, .30), (1, .14)]
        gl, win = (2, 4), [(-.55, .49), (-.39, .63), (.10, .63), (.38, .49)]
    elif typ == "hatch":
        prof = [(-1, .14), (-1, j(.45)), (j(-.82), .67), (j(.10), .67), (.45, j(.46)), (.95, j(.40)), (1, .30), (1, .14)]
        gl, win = (1, 3), [(-.9, .5), (-.8, .64), (.08, .64), (.38, .5)]
    elif typ == "van":
        prof = [(-1, .14), (-1, j(.74)), (j(.38), .74), (.66, j(.48)), (.96, j(.42)), (1, .14)]
        gl, win = (2,), [(-.9, .52), (-.9, .7), (.36, .7), (.58, .52)]
    elif typ == "sport":
        prof = [(-1, .12), (-1, j(.36)), (-.6, j(.42)), (j(-.3), .54), (j(.10), .54), (.45, j(.38)), (.98, j(.30)), (1, .12)]
        gl, win = (2, 4), [(-.5, .43), (-.3, .52), (.08, .52), (.36, .42)]
    else:
        prof = [(-1, .14), (-1, j(.48)), (j(-.92), .66), (j(.12), .66), (.45, j(.46)), (.95, j(.40)), (1, .30), (1, .14)]
        gl, win = (1, 3), [(-.9, .5), (-.86, .63), (.10, .63), (.38, .5)]
    w = U(rng, .37, .44)
    o.face([(-1.05, 0, -w - .06), (1.05, 0, -w - .06), (1.05, 0, w + .06), (-1.05, 0, w + .06)], (.05, .05, .05), (0, 1, 0), -30, U(rng, .4, .8))
    o.prism(prof, -w, w, col, {k: glass for k in gl})
    for sg in (-1, 1):
        o.face([(x, y, sg * (w + .004)) for x, y in win], glass, (0, 0, sg), .5)
        o.quad("x", 1.004, .30, .39, sg * .18, sg * .37, jit(rng, (.95, .93, .8), .05), 1, .5)     # headlights
        o.quad("x", -1.004, .32, .42, sg * .2, sg * .38, jit(rng, (.75, .1, .08), .1), -1, .5)    # tail lights
    o.quad("x", 1.004, .17, .29, -.2, .2, (.1, .1, .1), 1, .5)                                    # grille
    if rng.random() < .6:
        o.quad("x", -1.004, .22, .31, -.12, .12, (.9, .9, .88), -1, .5)                           # number plate
    r = U(rng, .15, .18)
    hub = jit(rng, pick(rng, [(.65, .65, .68), (.8, .8, .82), (.3, .3, .32)]), .08)
    for x in (j(-.62), j(.62)):
        for sg in (-1, 1):
            o.wheel((x, r, sg * (w + .01)), r, .2, sg, TYRE, hub)
    return o


def make_truck(rng):
    o = Obj3()
    col = jit(rng, pick(rng, PAINT, TRUCK_P), .08, .02)
    glass = glasscol(rng)
    chrome = jit(rng, (.72, .73, .75), .08, .01)
    dark = (.12, .12, .13)
    typ = pick(rng, ["box", "box", "semi", "semi", "semi", "dump", "dump", "fire", "tank", "flat"])
    j = lambda v, a=.025: v + rng.normal(0, a)
    wc = U(rng, .30, .36)                       # cab half-width
    hc = j(.84, .05)                            # cab roof height
    nose = rng.random() < .55                   # long bonnet in front of the cab, or flat-fronted cab
    x0 = {"box": .36, "semi": .34, "dump": .40, "fire": .42, "tank": .36, "flat": .40}[typ]
    if typ == "semi" and not nose:
        x0 = .52
    if nose:
        xh = j(.68)
        cab = [(x0, .24), (x0, hc), (xh - .04, hc), (xh, j(.60)), (.98, j(.56)), (1, .24)]
    else:
        cab = [(x0, .24), (x0, hc), (.94, hc), (1, j(.55)), (1, .24)]
    o.face([(-1.05, 0, -.42), (1.05, 0, -.42), (1.05, 0, .42), (-1.05, 0, .42)], (.05, .05, .05), (0, 1, 0), -30, U(rng, .4, .8))
    o.box((-1, .2, -.25), (x0, .32, .25), dark)                                           # chassis
    o.prism(cab, -wc, wc, col, {2: glass})
    xa = (x0 + .08, (xh - .06) if nose else .9)
    for sg in (-1, 1):
        o.quad("z", sg * (wc + .004), xa[0], xa[1], hc - .26, hc - .05, glass, sg, .5)                 # side window
        o.quad("x", 1.004, .34, .44, sg * .2, sg * (wc - .02), jit(rng, (.95, .93, .8), .05), 1, .5)  # headlights
    o.quad("x", 1.004, .3, j(.52), -.18, .18, chrome if rng.random() < .7 else dark, 1, .5)        # grille
    o.quad("x", 1.01, .2, .29, -wc, wc, chrome if rng.random() < .6 else dark, 1, .5)               # bumper
    hb = j(1.02, .06)                           # top of the load
    wb = U(rng, .33, .39)
    if typ in ("box", "semi"):
        bc = jit(rng, pick(rng, [(.93, .93, .93), (.75, .76, .78), col, pick(rng, PAINT)], [.5, .15, .2, .15]), .06, .015)
        o.box((-1, .32, -wb), (x0 - .03, hb, wb), bc)
        if rng.random() < .3:                   # lettering / stripe on the side of the box
            sc_ = jit(rng, pick(rng, PAINT), .1)
            for sg in (-1, 1):
                o.quad("z", sg * (wb + .004), -.8, x0 - .25, .55, .75, sc_, sg, .5)
    elif typ == "dump":
        bc = jit(rng, pick(rng, [col, (.92, .78, .10), (.90, .45, .10), (.4, .4, .42), (.12, .25, .6), (.75, .1, .08)]), .08)
        o.prism([(-1, .42), (-.96, j(.92)), (x0 + .04, j(.92)), (x0 - .03, .42)], -wb, wb, bc)
    elif typ == "fire":
        o.box((-1, .32, -wc), (x0, hc - .06, wc), col)
        for sg in (-1, 1):
            o.quad("z", sg * (wc + .004), -.95, x0, .5, .58, (.92, .92, .9), sg, .5)
            o.quad("z", sg * (wc + .004), -.8, -.2, .36, .5, chrome, sg, .5)
        o.box((-.95, hc - .06, -.12), (.6, hc + .03, .12), chrome)
    elif typ == "tank":
        bc = jit(rng, pick(rng, [(.8, .81, .83), (.93, .93, .93), col]), .06, .01)
        o.tube((-.92, .66, 0), (x0 - .1, .66, 0), .3, .3, bc)
    else:                                       # flat bed with a load
        o.box((-1, .32, -wb), (x0 - .03, .42, wb), jit(rng, pick(rng, [col, (.3, .3, .3), (.5, .4, .3)]), .1))
        if rng.random() < .7:
            o.box((-.9, .42, -wb * .8), (j(.1, .15), j(.75, .1), wb * .8), jit(rng, pick(rng, [(.5, .38, .25), (.6, .6, .6), (.75, .7, .5), (.3, .4, .3)]), .12))
    if typ == "semi" and rng.random() < .5:     # exhaust stacks
        for sg in (-1, 1):
            o.tube((x0 - .02, .4, sg * wc), (x0 - .02, hc + .12, sg * wc), .025, .025, chrome)
    r = U(rng, .17, .2) if typ == "dump" else U(rng, .125, .155)
    hub = jit(rng, pick(rng, [(.7, .7, .72), (.85, .85, .86), (.3, .3, .32), col]), .08)
    xs = [j(.76)] + {"box": [-.55], "semi": [.2, -.7, -.86], "dump": [-.45, -.75], "fire": [-.55], "tank": [.15, -.7, -.86],
                     "flat": [-.5, -.75]}[typ]
    for x in xs:
        for sg in (-1, 1):
            o.wheel((x, r, sg * (wb + .01)), r, .2, sg, TYRE, hub)
    lenf = U(rng, .45, 1.0)                     # everything behind the cab is squeezed towards it
    for it in o.it:
        x = it[1][:, 0]
        it[1][:, 0] = np.where(x < x0, x0 - (x0 - x) * lenf, x)
    return o


def make_plane(rng):
    """Aeroplane with its nose along +x: tube body, flat wings and tail. Returns (solid, kind)."""
    o = Obj3()
    typ = pick(rng, ["liner", "liner", "liner", "liner", "liner", "fighter", "fighter", "prop", "prop", "wing"])
    j = lambda v, a=.03: v + rng.normal(0, a)
    body = jit(rng, pick(rng, [(.93, .93, .94), (.80, .81, .83), (.55, .57, .60), (.25, .27, .30), (.75, .1, .08), (.9, .8, .2), (.2, .3, .6)],
                         [.40, .2, .15, .08, .06, .06, .05]), .06, .015)
    wingc = np.clip(body * U(rng, .7, .95), 0, 1)
    under = np.clip(wingc * U(rng, .6, .85), 0, 1)

    def flat(pts, top=wingc, bot=under):
        """A thin plate lying in the y = const plane: lighter on top, darker underneath; mirrored left / right."""
        for sg in (-1, 1):
            q = [(x, y, sg * z) for x, y, z in pts]
            o.face(q, top, (0, 1, 0)); o.face(q, bot, (0, -1, 0))
            o.tube(q[0], q[-1], .022, .012, bot)      # leading edge, so a wing seen edge-on is still a line

    def fin(pts, col):
        o.face([(x, y, 0) for x, y in pts], col, (0, 0, 1)); o.face([(x, y, 0) for x, y in pts], col, (0, 0, -1))

    if typ == "liner":
        r = U(rng, .115, .165)
        o.tube((-.97, r * .4, 0), (-.5, 0, 0), r * .4, r, body); o.tube((-.5, 0, 0), (.8, 0, 0), r, r, body)
        o.tube((.8, 0, 0), (.96, -r * .15, 0), r, r * .55, body)
        sp = U(rng, .8, 1.05); yw = -r * .5
        flat([(j(.30), yw, r * .5), (-.20, yw, r * .5), (j(-.56), yw + .05, sp), (j(-.34), yw + .05, sp)])
        flat([(-.68, r * .3, .04), (-.94, r * .3, .04), (-1.04, r * .3, j(.40)), (-.9, r * .3, j(.40))])
        finc = jit(rng, pick(rng, [body, (.12, .25, .6), (.75, .1, .08), (.1, .15, .35), (.9, .7, .1), (.1, .4, .3)], [.4, .2, .15, .1, .08, .07]), .08)
        fin([(-.60, r * .7), (j(-.86), j(.42)), (-1.03, j(.42)), (-.95, r * .7)], finc)
        ec = np.clip(body * U(rng, .5, .9), 0, 1)
        for z in ([.36] if rng.random() < .7 else [.34, .62]):
            for sg in (-1, 1):
                o.tube((.14 - .25 * z, yw - .07, sg * z), (-.08 - .25 * z, yw - .07, sg * z), .05, .045, ec)
        if rng.random() < .35:                    # coloured stripe along the windows
            o.tube((-.6, 0, 0), (.8, 0, 0), r * .22, r * .22, finc, .3)
    elif typ == "fighter":
        body = jit(rng, pick(rng, [(.50, .53, .56), (.35, .38, .40), (.65, .67, .70), (.25, .28, .25), (.4, .42, .35)]), .08, .015)
        wingc = np.clip(body * U(rng, .8, 1.0), 0, 1); under = np.clip(body * U(rng, .65, .9), 0, 1)
        o.tube((-.92, 0, 0), (.25, 0, 0), .12, .14, body); o.tube((.25, 0, 0), (1.0, 0, 0), .14, .025, body)
        o.tube((.25, .07, 0), (.5, .05, 0), .05, .03, (.15, .18, .22), .3)
        flat([(j(.25), 0, .06), (-.72, 0, .06), (-.78, 0, j(.64)), (j(-.5), 0, j(.64))], wingc, under)
        flat([(-.72, 0, .06), (-.98, 0, .06), (-1.04, 0, .36), (-.9, 0, .36)], wingc, under)
        fin([(-.5, .06), (-.84, j(.38)), (-.99, j(.38)), (-.92, .06)], wingc)
    elif typ == "prop":
        body = jit(rng, pick(rng, [(.93, .93, .94), (.9, .8, .2), (.75, .1, .08), (.2, .3, .6), (.7, .72, .75), (.3, .4, .3)]), .08, .02)
        wingc = np.clip(body * U(rng, .8, 1.0), 0, 1); under = np.clip(body * U(rng, .6, .85), 0, 1)
        r = U(rng, .12, .17)
        o.tube((-.95, r * .3, 0), (-.2, 0, 0), r * .35, r, body); o.tube((-.2, 0, 0), (.75, 0, 0), r, r * .9, body)
        o.tube((.75, 0, 0), (.9, 0, 0), r * .9, r * .5, (.2, .2, .2))
        yw = r * pick(rng, [-.6, .9]); sp = U(rng, .85, 1.1)
        flat([(.36, yw, 0), (-.04, yw, 0), (.0, yw, sp), (.28, yw, sp)], wingc, under)
        flat([(-.70, r * .3, 0), (-.96, r * .3, 0), (-.96, r * .3, .38), (-.78, r * .3, .38)], wingc, under)
        fin([(-.66, r * .5), (-.86, j(.36)), (-1.0, j(.36)), (-.96, r * .5)], wingc)
        if rng.random() < .6:
            o.tube((.2, .08, 0), (.4, .06, 0), .06, .04, (.2, .25, .3), .3)
    else:                                         # flying wing / delta seen as one dark triangle
        body = jit(rng, pick(rng, [(.22, .23, .26), (.12, .12, .14), (.4, .42, .45), (.55, .6, .65)]), .1, .015)
        sp = U(rng, .85, 1.1)
        o.tube((-.3, 0, 0), (.6, 0, 0), .07, .05, body)
        flat([(.92, 0, 0), (j(-.45), 0, sp), (-.62, 0, sp * .95), (j(-.3), 0, sp * .45), (-.5, 0, 0)], body, np.clip(body * .8, 0, 1))
    return o, typ


def plane_gear(o, rng):
    """Wheels down, for a plane standing on the ground; returns the height of the ground below the body."""
    h = U(rng, .16, .24)
    for x, z in ((.6, 0), (-.1, .16), (-.1, -.16)):
        o.tube((x, 0, z), (x, -h, z), .018, .018, (.25, .25, .27))
        o.tube((x, -h, z), (x, -h, z), .04, .04, TYRE)
    return -h - .04


HULL = [(.08, .08, .09), (.10, .14, .30), (.60, .10, .08), (.10, .30, .20), (.45, .47, .50), (.92, .92, .92), (.15, .3, .6), (.85, .5, .1)]


def make_ship(rng):
    """Ship floating on the plane y = 0 with its bow along +x. Returns (solid, kind)."""
    o = Obj3()
    typ = pick(rng, ["cargo", "cargo", "cruise", "sail", "sail", "speed", "speed", "speed", "fish", "fish", "navy", "tug"])
    j = lambda v, a=.02: v + rng.normal(0, a)
    white = jit(rng, (.93, .93, .92), .04, .01)

    def hull(b, h, col, deck, bowx=.7, stern=.6):
        prof = [(1, 0), (bowx, b), (-.92, b), (-1, b * stern), (-1, -b * stern), (-.92, -b), (bowx, -b)]
        o.face([(x * 1.04, 0, z * 1.25) for x, z in prof], (.06, .08, .1), (0, 1, 0), -30, U(rng, .15, .45))   # dark water under the hull
        if rng.random() < .5:                                                                                 # red / dark band at the waterline
            o.prism(prof, 0, h * .3, jit(rng, pick(rng, [(.55, .1, .08), (.1, .1, .1), (.1, .2, .1)]), .1), up=True)
            o.prism(prof, h * .3, h, col, {"top": deck}, up=True)
        else:
            o.prism(prof, 0, h, col, {"top": deck}, up=True)

    def wake(b, big):
        fc = jit(rng, (.9, .93, .95), .05, .01)
        a = U(rng, .5, .9)
        o.face([(-.9, 0, b), (-1.0 - 1.2 * big, 0, b * (1 + 2 * big)), (-1.0 - 1.2 * big, 0, -b * (1 + 2 * big)), (-.9, 0, -b)], fc, (0, 1, 0), -29, a)
        for sg in (-1, 1):
            o.face([(.8, 0, sg * b * .5), (.2, 0, sg * b * (1.2 + big)), (-.9, 0, sg * b * (1.3 + 1.5 * big)), (-.9, 0, sg * b)], fc, (0, 1, 0), -29, a * .8)

    if typ == "cargo":
        b, h = U(rng, .12, .17), U(rng, .14, .24)
        hull(b, h, jit(rng, pick(rng, HULL[:5] + [HULL[6]]), .1), jit(rng, pick(rng, [(.45, .25, .2), (.4, .42, .4), (.2, .35, .25)]), .1))
        xs = pick(rng, [-.85, -.85, .45])         # deck house at the stern (usually) or forward
        o.box((xs, h, -b * .8), (xs + .28, h + j(.3, .04), b * .8), white)
        o.box((xs + .08, h + .3, -b * .3), (xs + .2, h + j(.44, .03), b * .3), jit(rng, pick(rng, [(.7, .1, .08), (.1, .1, .1), (.1, .2, .5), (.9, .7, .1)]), .1))
        if rng.random() < .6:                      # containers or hatch covers
            x = -.5 if xs < 0 else -.9
            while x < (.65 if xs < 0 else .35):
                wd = U(rng, .1, .25)
                o.box((x, h, -b * .8), (x + wd, h + U(rng, .04, .2), b * .8), jit(rng, pick(rng, PAINT + [(.5, .25, .15)] * 3), .1))
                x += wd + .02
        for x in ([j(0, .2), j(.5, .1)] if rng.random() < .6 else []):
            o.tube((x, h, 0), (x, h + j(.32, .05), 0), .012, .012, jit(rng, pick(rng, [(.9, .8, .3), (.3, .3, .3), (.9, .9, .9)]), .1))
    elif typ == "cruise":
        b, h = U(rng, .13, .17), U(rng, .13, .18)
        hull(b, h, white if rng.random() < .8 else jit(rng, pick(rng, HULL), .1), white)
        x0, x1, y = -.88, .72, h
        for k in range(int(rng.integers(2, 5))):
            hh = U(rng, .07, .11)
            o.box((x0, y, -b * .9), (x1, y + hh, b * .9), white)
            for sg in (-1, 1):
                o.quad("z", sg * (b * .9 + .003), x0 + .03, x1 - .03, y + hh * .35, y + hh * .7, (.2, .25, .3), sg, .5)
            y += hh; x0 += U(rng, .03, .12); x1 -= U(rng, .08, .2)
        o.box((-.35, y, -b * .35), (-.15, y + j(.13), b * .35), jit(rng, pick(rng, [(.7, .1, .08), (.1, .15, .4), (.9, .7, .1), (.93, .93, .93), (.1, .1, .1)]), .1))
    elif typ == "sail":
        b, h = U(rng, .1, .15), U(rng, .07, .12)
        hull(b, h, jit(rng, pick(rng, [(.92, .92, .92)] * 3 + HULL), .08), jit(rng, pick(rng, [(.8, .75, .6), (.9, .9, .9), (.5, .35, .2)]), .1), bowx=.55)
        mh = U(rng, 1.2, 1.8); xm = j(.1, .05)
        sail = jit(rng, pick(rng, [(.95, .95, .93), (.9, .86, .75), (.85, .87, .9), (.8, .3, .2), (.3, .4, .7)], [.5, .2, .15, .08, .07]), .05, .01)
        o.tube((xm, h, 0), (xm, h + mh, 0), .012, .01, (.3, .28, .25))
        sw = rng.normal(0, .12)                    # sails swing a little out of the centre line
        if rng.random() < .85:
            o.face([(xm - .02, h + .12, 0), (xm - .02, h + mh * .97, 0), (xm - j(.8, .08), h + .12, sw)], sail)
        if rng.random() < .7:
            o.face([(xm + .03, h + .1, 0), (xm + .03, h + mh * j(.85, .05), 0), (.92, h + .06, -sw)], np.clip(sail * U(rng, .9, 1.05), 0, 1))
        if rng.random() < .3:                      # second mast
            o.tube((-.5, h, 0), (-.5, h + mh * .8, 0), .01, .01, (.3, .28, .25))
            o.face([(-.52, h + .1, 0), (-.52, h + mh * .77, 0), (-.98, h + .1, sw)], sail)
    elif typ == "speed":
        b, h = U(rng, .2, .27), U(rng, .11, .16)
        hull(b, h, jit(rng, pick(rng, [(.93, .93, .93)] * 4 + HULL), .06), jit(rng, pick(rng, [(.93, .93, .9), (.85, .8, .7), (.3, .4, .6)]), .06), bowx=.45, stern=1.0)
        if rng.random() < .8:
            cx0, cx1, ch = j(-.35, .08), j(.25, .08), U(rng, .1, .2)
            o.prism([(cx0, h), (cx0 + .05, h + ch), (cx1 - .15, h + ch), (cx1, h)], -b * .7, b * .7,
                    white, {2: glasscol(rng), "top": white if rng.random() < .6 else jit(rng, pick(rng, PAINT), .1)})
            for sg in (-1, 1):
                o.quad("z", sg * (b * .7 + .003), cx0 + .08, cx1 - .14, h + ch * .4, h + ch * .85, glasscol(rng), sg, .5)
        wake(b, U(rng, .2, 1.0))
    elif typ in ("fish", "tug"):
        b, h = U(rng, .18, .24), U(rng, .14, .2)
        hull(b, h, jit(rng, pick(rng, HULL), .1), jit(rng, pick(rng, [(.5, .35, .2), (.6, .6, .58), (.3, .4, .35)]), .1), bowx=.55, stern=.85)
        xs = j(.05, .25)
        o.box((xs, h, -b * .65), (xs + j(.36, .05), h + j(.3, .05), b * .65), white)
        for sg in (-1, 1):
            o.quad("z", sg * (b * .65 + .003), xs + .04, xs + .3, h + .16, h + .26, (.15, .2, .25), sg, .5)
        o.quad("x", xs + .365, h + .16, h + .26, -b * .5, b * .5, (.15, .2, .25), 1, .5)
        if rng.random() < .7:
            o.tube((xs + .1, h + .3, 0), (xs + .1, h + j(.75, .1), 0), .012, .01, (.3, .3, .3))
        if rng.random() < .4:
            o.box((xs + .12, h + .3, -b * .2), (xs + .24, h + .42, b * .2), jit(rng, pick(rng, [(.7, .1, .08), (.1, .1, .1), (.9, .7, .1)]), .1))
        if rng.random() < .5:
            wake(b, U(rng, .1, .5))
    else:                                          # grey warship
        b, h = U(rng, .09, .12), U(rng, .11, .15)
        g = jit(rng, (.55, .58, .62), .12, .01)
        hull(b, h, g, np.clip(g * .85, 0, 1), bowx=.6)
        x0, x1, y = -.45, .35, h
        for k in range(3):
            hh = U(rng, .07, .12)
            o.box((x0, y, -b * .7), (x1, y + hh, b * .7), np.clip(g * U(rng, .9, 1.1), 0, 1))
            y += hh; x0 += U(rng, .08, .2); x1 -= U(rng, .08, .2)
        o.tube((0, y, 0), (0, y + j(.3, .05), 0), .012, .008, np.clip(g * .7, 0, 1))
        if rng.random() < .5:
            wake(b, U(rng, .1, .4))
    return o, typ


# =====================================================================================================
# 4. Animals, drawn flat from ellipses, thick lines and triangles (body, neck, head, legs, ears, tail).
# =====================================================================================================
def coat(rng, base, F=None, kind="plain", col2=None, amp=.12, light=.25):
    """Full-canvas colour field for fur / skin: soft texture, light from above, optional patches, spots, stripes."""
    f = 1 + amp * fnoise(rng, (4, 10, 24), (1, .8, .6)) + amp * .8 * noise(rng, 32)
    if F is not None:
        f = f * (1 + light * np.clip((F.cy - PY2) / (F.s + 1e-6), -1, 1))
    f = f[..., None]
    img = np.asarray(base, float)[None, None, :] * f
    if kind == "patch":
        m = fnoise(rng, (2, 3, 5), (1, .8, .4)) > U(rng, -.4, .7)
        img[m] = (np.asarray(col2, float) * f)[m]
    elif kind == "spots":
        m = fnoise(rng, (7, 12, 20), (1, .8, .5)) > U(rng, .5, 1.1)
        img[m] = (np.asarray(col2, float) * f)[m]
    elif kind == "stripes":
        a = U(rng, 0, np.pi)
        ph = (PX2 * np.cos(a) + PY2 * np.sin(a)) * U(rng, 40, 70) + 2.5 * noise(rng, 5)
        m = np.sin(ph) > U(rng, .0, .5)
        img[m] = (np.asarray(col2, float) * f)[m]
    return np.clip(img, 0, 1)


def quadruped(sc, rng, F, sp, ct, c):
    """Four-legged animal seen from the side, head towards +u. sp: proportions. ct: coat field. c: extra colours."""
    P = F.p; s = F.s
    far = np.clip(ct * .72, 0, 1)
    G = sp["G"]; r0, r1 = sp["legr"]
    g = sp.get("gait", .07)

    def leg(u0, col, hind):
        d1 = rng.normal(0, g) - (.05 if hind else 0); d2 = d1 + rng.normal(0, g) + (.05 if hind else 0)
        hip, knee, foot = (u0, .05), (u0 + d1, .05 + (G - .05) * .52), (u0 + d2, G)
        sc.seg(P(*hip), P(*knee), s * r0 * 1.25, s * (r0 + r1) / 2, col)
        lc = c.get("sock", col)
        sc.seg(P(*knee), P(*foot), s * (r0 + r1) / 2, s * r1, lc if lc is not col else col)

    leg(sp["fl"] - .08, far, False); leg(sp["hl"] + .09, far, True)
    if "tail" in sp:
        pts, rad = sp["tail"]
        sc.path(F.pp(pts), [s * r for r in rad], c.get("tail", ct))
    bx, ry = sp["bx"], sp["ry"]
    sc.ell(P(-bx, 0), P(bx, 0), s * ry, ct)
    sc.ell(P(sp["fl"] - .2, -.02), P(sp["fl"] + .12, -.02), s * ry * 1.12, ct)       # chest
    sc.ell(P(sp["hl"] - .14, -.02), P(sp["hl"] + .2, -.02), s * ry * 1.1, ct)        # rump
    if "belly" in c:
        sc.ell(P(-bx * .7, ry * .62), P(bx * .7, ry * .62), s * ry * .38, c["belly"])
    if "rump" in c:
        sc.ell(P(-bx - .02, -.08), P(-bx + .14, .06), s * ry * .5, c["rump"])
    leg(sp["fl"], ct, False); leg(sp["hl"], ct, True)
    (nb, nt), (nr0, nr1) = sp["neck"], sp["neckr"]
    if "mane" in c:
        sc.seg(P(nb[0] - .1, nb[1] - .06), P(nt[0] - .07, nt[1] - .05), s * nr0 * .8, s * nr1 * .8, c["mane"])
    sc.seg(P(*nb), P(*nt), s * nr0, s * nr1, ct)
    hd = sp["head"]                                   # head end point (muzzle), radii
    hr0, hr1 = sp["headr"]
    for e in sp.get("ears", []):                      # ears: (du, dv, length-u, length-v, half-width)
        sc.ell(P(nt[0] + e[0], nt[1] + e[1]), P(nt[0] + e[0] + e[2], nt[1] + e[1] + e[3]), s * e[4], c.get("ear", ct))
    if "antler" in c:
        for side in (0, 1):
            b0 = (nt[0] - .02 + .04 * side, nt[1] - hr0 * .8)
            lean = rng.normal(-.06, .06) + .1 * side
            hgt = U(rng, .3, .55)
            pts = [b0, (b0[0] + lean, b0[1] - hgt * .5), (b0[0] + lean * 1.5 + .06, b0[1] - hgt)]
            sc.path(F.pp(pts), [s * .022, s * .018, s * .012], c["antler"])
            for k in range(int(rng.integers(1, 4))):
                t = U(rng, .3, .9)
                q = (b0[0] + lean * 1.2 * t, b0[1] - hgt * t)
                sc.seg(P(*q), P(q[0] + U(rng, .06, .16), q[1] - U(rng, .05, .16)), s * .014, s * .01, c["antler"])
    sc.seg(P(*nt), P(*hd), s * hr0, s * hr1, ct)
    if "snout" in sp:                                 # dogs: round skull plus a muzzle
        (su, sv), (sr0, sr1) = sp["snout"]
        sc.seg(P(nt[0] + hr0 * .5, nt[1] + .02), P(nt[0] + su, nt[1] + sv), s * sr0, s * sr1, c.get("muzzle", ct))
        sc.circ(P(nt[0] + su + sr1 * .5, nt[1] + sv - sr1 * .3), s * sr1 * .55, (.08, .07, .07))
    for e in sp.get("ears2", []):                     # ears drawn over the head (floppy dog ears)
        sc.ell(P(nt[0] + e[0], nt[1] + e[1]), P(nt[0] + e[0] + e[2], nt[1] + e[1] + e[3]), s * e[4], c.get("ear", ct))
    if "nose" in c:
        sc.circ(P(*hd), s * hr1 * .8, c["nose"])
    if "blaze" in c:
        sc.seg(P(nt[0] + .05, nt[1] - .03), P(hd[0], hd[1] - .03), s * hr1 * .45, s * hr1 * .45, c["blaze"])
    ex, ey = nt[0] + (hd[0] - nt[0]) * .28, nt[1] + (hd[1] - nt[1]) * .28 - hr0 * .3
    sc.circ(P(ex, ey), s * .022, (.05, .04, .04))
    if "rider" in c:
        shirt, pants, skin = c["rider"]
        sc.seg(P(.02, -ry * .6), P(.16, G * .45), s * .06, s * .045, pants)
        sc.seg(P(.0, -ry * .9), P(.06, -ry - .42), s * .09, s * .08, shirt)
        sc.seg(P(.06, -ry - .36), P(.3, -ry - .12), s * .035, s * .03, shirt)
        sc.circ(P(.09, -ry - .56), s * .075, skin)
        if rng.random() < .6:
            sc.ell(P(.02, -ry - .63), P(.16, -ry - .63), s * .05, (.1, .1, .12))     # hat / helmet


def spec_horse(rng):
    j = lambda v, a=.03: v + rng.normal(0, a)
    graze = rng.random() < .15
    nt = (j(.78), j(.22)) if graze else (j(.70), j(-.50, .06))
    hd = (nt[0] + j(.08), nt[1] + j(.3)) if graze else (nt[0] + j(.22), nt[1] + j(.2, .06))
    return dict(bx=j(.50), ry=j(.22, .015), fl=.36, hl=-.40, G=j(.92, .04), legr=(.06, .034), gait=.08,
                neck=((.40, -.08), nt), neckr=(.16, .105), head=hd, headr=(.105, .062),
                ears=[(-.02, -.1, .03, -.1, .03)],
                tail=([(-.52, -.1), (j(-.7), j(.12)), (j(-.68), j(.5, .08))], (.06, .055, .03)))


def spec_deer(rng):
    j = lambda v, a=.03: v + rng.normal(0, a)
    graze = rng.random() < .15
    nt = (j(.68), j(.2)) if graze else (j(.52), j(-.55, .06))
    hd = (nt[0] + j(.07), nt[1] + j(.24)) if graze else (nt[0] + j(.2), nt[1] + j(.07, .05))
    return dict(bx=j(.46), ry=j(.185, .015), fl=.33, hl=-.36, G=j(.9, .04), legr=(.04, .022), gait=.07,
                neck=((.36, -.08), nt), neckr=(.105, .068), head=hd, headr=(.078, .042),
                ears=[(-.03, -.04, -.1, -.1, .04), (-.01, -.05, -.03, -.15, .04)],
                tail=([(-.47, -.1), (-.54, -.02)], (.035, .03)))


def spec_dog(rng):
    j = lambda v, a=.03: v + rng.normal(0, a)
    nt = (j(.56), j(-.3, .05))
    up = rng.random() < .5
    tail = [(-.45, -.1), (j(-.62), j(-.3)), (j(-.58), j(-.5))] if up else [(-.45, -.08), (j(-.6), j(.08)), (j(-.66), j(.3))]
    flop = rng.random() < .65
    return dict(bx=j(.45), ry=j(.2, .02), fl=.33, hl=-.36, G=j(.62, .06), legr=(.058, .042), gait=.07,
                neck=((.36, -.08), nt), neckr=(.15, .13), head=(nt[0] + .03, nt[1]), headr=(.16, .15),
                snout=((j(.26), j(.06)), (.078, .06)),
                ears=[] if flop else [(-.06, -.1, .02, -.14, .05)],
                ears2=[(-.07, -.08, -.03, .2, .055)] if flop else [],
                tail=(tail, (.045, .035, .02)))


def spec_cat(rng):
    j = lambda v, a=.03: v + rng.normal(0, a)
    nt = (j(.55), j(-.16, .05))
    up = rng.random() < .5
    tail = [(-.45, -.05), (j(-.7), j(-.2)), (j(-.82), j(-.5))] if up else [(-.45, 0), (j(-.75), j(.1)), (j(-.98), j(.0))]
    return dict(bx=j(.45), ry=j(.17, .02), fl=.33, hl=-.36, G=j(.5, .05), legr=(.046, .036), gait=.08,
                neck=((.38, -.05), nt), neckr=(.14, .13), head=(nt[0] + .04, nt[1] + .01), headr=(.165, .15),
                ears=[(-.08, -.1, .0, -.13, .05), (.06, -.11, .01, -.12, .05)],
                tail=(tail, (.036, .03, .025)))


def dog_face(sc, rng, F, ct, c, chest=True):
    """Dog's head seen from the front; local units: head half-width about 0.85."""
    P = F.p; s = F.s
    if chest:
        sc.ell(P(0, .4), P(0, 2.4), s * U(rng, .7, 1.0), ct)
    ear = c.get("ear", ct)
    flop = c.get("flop", True)
    if flop:
        for sg in (-1, 1):
            sc.ell(P(sg * .72, -.45), P(sg * U(rng, .85, 1.05), U(rng, .4, .75)), s * U(rng, .22, .32), ear)
    else:
        for sg in (-1, 1):
            sc.poly([P(sg * .3, -.6), P(sg * U(rng, .7, .9), -U(rng, 1.15, 1.4)), P(sg * .9, -.3)], ear)
    sc.ell(P(0, -.8), P(0, .72), s * .84, ct)
    mz = c.get("muzzle", ct)
    sc.ell(P(0, .0), P(0, .82), s * U(rng, .36, .46), mz)
    for sg in (-1, 1):
        sc.circ(P(sg * .36, -.2), s * .09, (.06, .05, .05))
    sc.ell(P(-.17, .28), P(.17, .28), s * .11, (.08, .07, .07))
    if rng.random() < .3:
        sc.ell(P(0, .55), P(0, .98), s * .13, jit(rng, (.85, .45, .5), .08))
    else:
        sc.seg(P(-.2, .58), P(.2, .58), s * .025, s * .025, (.12, .08, .08))
    if not flop:
        return
    if "patch" in c:                                  # darker patch round one or both eyes
        for sg in ((-1, 1) if rng.random() < .5 else (pick(rng, [-1, 1]),)):
            sc.ell(P(sg * .36, -.5), P(sg * .36, .0), s * .2, c["patch"], .85)
            sc.circ(P(sg * .36, -.2), s * .09, (.06, .05, .05))


def cat_face(sc, rng, F, ct, c, chest=True):
    """Cat's head seen from the front; local units: head half-width about 0.95."""
    P = F.p; s = F.s
    if chest:
        sc.ell(P(0, .4), P(0, 2.4), s * U(rng, .75, 1.05), ct)
    for sg in (-1, 1):
        tip = (sg * U(rng, .68, .85), -U(rng, 1.15, 1.35))
        sc.poly([P(sg * .22, -.55), P(*tip), P(sg * .95, -.3)], ct)
        sc.poly([P(sg * .42, -.6), P(tip[0] * .95, tip[1] * .88), P(sg * .8, -.45)], c.get("inner", (.8, .6, .58)))
    sc.ell(P(-.95, -.02), P(.95, -.02), s * .76, ct)
    if "muzzle" in c:
        sc.ell(P(-.36, .38), P(.36, .38), s * .26, c["muzzle"])
        if rng.random() < .5:
            sc.ell(P(0, -.1), P(0, .5), s * .13, c["muzzle"])
    ec = c.get("eye", (.6, .7, .3))
    for sg in (-1, 1):
        sc.ell(P(sg * .38 - .17, -.12), P(sg * .38 + .17, -.12), s * .12, ec)
        sc.ell(P(sg * .38, -.23), P(sg * .38, -.01), s * U(rng, .04, .09), (.04, .04, .04))
    sc.poly([P(-.09, .2), P(.09, .2), P(0, .32)], c.get("nosec", (.8, .5, .5)))
    sc.seg(P(0, .32), P(0, .46), s * .02, s * .02, (.2, .15, .15))


def sitting(sc, rng, F, ct, c, face, hs):
    """Sitting cat or dog seen from the front: pear-shaped body, front legs, head on top."""
    P = F.p; s = F.s
    lean = rng.normal(0, .08)
    if rng.random() < .6:                             # tail lying to one side
        sg = pick(rng, [-1, 1])
        sc.path(F.pp([(sg * .3, .95), (sg * .75, 1.0), (sg * U(rng, .8, 1.0), U(rng, .5, .9))]), [s * .08, s * .07, s * .05], c.get("tail", ct))
    sc.ell(P(-.5, .78), P(.5, .78), s * .33, ct)
    sc.ell(P(lean, -.1), P(0, 1.0), s * .42, ct)
    if "belly" in c:
        sc.ell(P(lean * .7, .1), P(0, .9), s * .2, c["belly"])
    for sg in (-1, 1):
        sc.seg(P(sg * .16 + lean * .3, .35), P(sg * .18, 1.05), s * .09, s * .085, c.get("sock", ct))
    face(sc, rng, Frame(*P(lean * 1.2 + rng.normal(0, .05), -.35), s * hs, rng.normal(0, .12), F.flip), ct, c, False)


def lying(sc, rng, F, ct, c, face, hs):
    """Cat or dog lying down, body along u, head raised at the +u end (or curled up)."""
    P = F.p; s = F.s
    curl = rng.random() < .3
    tl = [(-.7, .15), (-.95, .3), (-.6, .45)] if rng.random() < .5 else [(-.7, .1), (-1.0, .0), (-1.1, -.25)]
    sc.path(F.pp(tl), [s * .07, s * .06, s * .045], c.get("tail", ct))
    sc.ell(P(-.75, .1), P(.5, .1), s * U(rng, .3, .4), ct)
    sc.ell(P(-.8, .12), P(-.3, .2), s * .3, ct)
    if "belly" in c:
        sc.ell(P(-.4, .3), P(.4, .3), s * .12, c["belly"])
    if not curl:
        for dv in (.0, .1):
            sc.seg(P(.35, .3 + dv), P(U(rng, .75, 1.0), .36 + dv), s * .075, s * .07, c.get("sock", ct))
        face(sc, rng, Frame(*P(U(rng, .45, .65), -.15 + rng.normal(0, .06)), s * hs, rng.normal(0, .2), F.flip), ct, c, False)
    else:
        face(sc, rng, Frame(*P(U(rng, .2, .45), .22), s * hs, rng.normal(.6, .3), F.flip), ct, c, False)


def horse_head(sc, rng, F, ct, c):
    """Horse's head and neck, in profile, filling the frame."""
    P = F.p; s = F.s
    j = lambda v, a=.05: v + rng.normal(0, a)
    poll, muz = (j(-.15), j(-.4)), (j(.75), j(.42))
    if "mane" in c:
        sc.seg(P(-1.45, 1.2), P(poll[0] - .1, poll[1] - .05), s * .55, s * .2, c["mane"])
    sc.seg(P(-1.3, 1.5), P(poll[0], poll[1] + .1), s * .62, s * .3, ct)
    for du in (-.12, .02):
        sc.poly([P(poll[0] + du - .08, poll[1] - .15), P(poll[0] + du, poll[1] - .52), P(poll[0] + du + .1, poll[1] - .15)], ct)
    sc.seg(P(*poll), P(*muz), s * .29, s * .15, ct)
    sc.seg(P(poll[0] + .1, poll[1] + .22), P(muz[0] - .25, muz[1] + .08), s * .2, s * .13, ct)   # jaw
    if "blaze" in c:
        sc.seg(P(poll[0] + .3, poll[1] - .1), P(muz[0] - .02, muz[1] - .12), s * .03, s * .035, c["blaze"])
    sc.circ(P(poll[0] + .27, poll[1] + .1), s * .045, (.05, .04, .04))
    sc.circ(P(muz[0] + .04, muz[1] - .02), s * .03, (.08, .06, .06))
    if "tack" in c:
        t = c["tack"]
        sc.seg(P(muz[0] - .28, muz[1] - .32), P(muz[0] - .38, muz[1] + .1), s * .025, s * .025, t)
        sc.seg(P(poll[0] + .05, poll[1] - .2), P(poll[0] + .2, poll[1] + .35), s * .025, s * .025, t)
        sc.seg(P(poll[0] + .2, poll[1] + .35), P(muz[0] - .36, muz[1] + .05), s * .02, s * .02, t)


def deer_face(sc, rng, F, ct, c, chest=True):
    """Deer's head seen from the front: narrow face, big ears out to the sides, dark nose, maybe antlers."""
    P = F.p; s = F.s
    if chest:
        sc.seg(P(0, .3), P(rng.normal(0, .2), 2.4), s * .45, s * .8, ct)
    if "antler" in c:
        for sg in (-1, 1):
            b0 = (sg * .2, -.65)
            pts = [b0, (sg * U(rng, .4, .6), -U(rng, 1.0, 1.3)), (sg * U(rng, .3, .6), -U(rng, 1.5, 1.9))]
            sc.path(F.pp(pts), [s * .06, s * .05, s * .035], c["antler"])
            for k in range(int(rng.integers(1, 4))):
                t = U(rng, .2, .9)
                q = (pts[0][0] + (pts[2][0] - pts[0][0]) * t, pts[0][1] + (pts[2][1] - pts[0][1]) * t)
                sc.seg(P(*q), P(q[0] + sg * U(rng, -.3, .1), q[1] - U(rng, .2, .45)), s * .04, s * .03, c["antler"])
    for sg in (-1, 1):
        tip = (sg * U(rng, .95, 1.2), -U(rng, .6, .95))
        sc.ell(P(sg * .35, -.42), P(*tip), s * .2, ct)
        sc.ell(P(sg * .5, -.48), P(tip[0] * .92, tip[1] * .95), s * .1, c.get("inner", (.85, .75, .7)))
    sc.ell(P(0, -.72), P(0, .68), s * .45, ct)
    if "muzzle" in c:
        sc.ell(P(0, .2), P(0, .7), s * .27, c["muzzle"])
    sc.ell(P(-.15, .5), P(.15, .5), s * .1, (.07, .06, .06))
    for sg in (-1, 1):
        sc.circ(P(sg * .3, -.2), s * .085, (.05, .04, .04))


def front_quad(sc, rng, F, ct, c, face, hs, legr, G):
    """Deer or horse standing and facing the camera: narrow chest, four thin legs, head on top."""
    P = F.p; s = F.s
    far = np.clip(ct * .72, 0, 1)
    for sg in (-1, 1):
        sc.seg(P(sg * .22 + rng.normal(0, .04), .3), P(sg * .25 + rng.normal(0, .05), G), s * legr * 1.2, s * legr, far)
    sc.ell(P(rng.normal(0, .05), -.3), P(0, .5), s * U(rng, .26, .34), ct)
    for sg in (-1, 1):
        sc.seg(P(sg * .13, .3), P(sg * .14 + rng.normal(0, .03), G), s * legr * 1.3, s * legr, c.get("sock", ct))
    sc.seg(P(0, -.15), P(0, -.6), s * .17, s * .13, ct)
    face(sc, rng, Frame(*P(rng.normal(0, .04), -.78), s * hs, rng.normal(0, .1), F.flip), ct, c, False)


def bird_side(sc, rng, F, c, legs=.35, neck=0.0):
    """Bird in profile, head towards +u. c: colours body, wing, belly, beak, leg. neck > 0 gives a long-necked bird."""
    P = F.p; s = F.s
    th = U(rng, .15, 1.0) if neck == 0 else U(rng, .0, .35)       # how upright the body is held
    cu, cv = np.cos(th), -np.sin(th)
    rear, front = (-.55 * cu, -.55 * cv), (.5 * cu, .5 * cv)
    bw = U(rng, .36, .46) if neck == 0 else U(rng, .34, .42)
    ta = th - U(rng, -.2, .5)                                     # tail direction
    tl = U(rng, .3, .65)
    for dx in (-.08, .1):                                         # legs
        fx = dx + rng.normal(0, .04)
        sc.seg(P(dx, bw * .7), P(fx, bw + legs), s * .03, s * .025, c["leg"])
        sc.seg(P(fx, bw + legs), P(fx + .12, bw + legs + .02), s * .022, s * .02, c["leg"])
    sc.seg(P(rear[0] * .7, rear[1] * .7), P(rear[0] - tl * np.cos(ta), rear[1] + tl * np.sin(ta)), s * .13, s * U(rng, .05, .1), c["wing"])
    sc.ell(P(*rear), P(*front), s * bw, c["body"])
    sc.ell(P(rear[0] * .3 - .0, rear[1] * .3 + bw * .5), P(front[0] * .9, front[1] * .9 + bw * .45), s * bw * .5, c["belly"])
    sc.ell(P(rear[0] * 1.05, rear[1] * 1.05 - .04), P(front[0] * .35, front[1] * .35 - bw * .35), s * bw * .55, c["wing"])
    if neck > 0:
        nb = (front[0] * .8, front[1] * .8)
        hd = (nb[0] + U(rng, .0, .35), nb[1] - neck)
        sc.seg(P(*nb), P(*hd), s * .13, s * .07, c.get("neck", c["body"]))
        hr = .13
    else:
        hd = (front[0] + .12 * cu + .05, front[1] + .12 * cv - .12)
        hr = U(rng, .25, .31)
    sc.circ(P(*hd), s * hr, c.get("head", c["body"]))
    bl = U(rng, .14, .4)
    sc.poly([P(hd[0] + hr * .7, hd[1] - hr * .35), P(hd[0] + hr * .7, hd[1] + hr * .35), P(hd[0] + hr + bl, hd[1] + rng.normal(.03, .04))], c["beak"])
    sc.circ(P(hd[0] + hr * .3, hd[1] - hr * .2), s * .04, (.04, .04, .04))


def bird_fly(sc, rng, F, c):
    """Bird in flight: small body, two spread wings."""
    P = F.p; s = F.s
    up = U(rng, -.5, .6)
    for sg in (-1, 1):
        tip = (rng.normal(-.1, .1), sg * U(rng, .8, 1.1))
        sc.poly([P(.22, sg * .05), P(tip[0] + .15, tip[1] * .6 - sg * up * .1), P(*tip), P(tip[0] - .25, tip[1] * .55), P(-.25, sg * .05)], c["wing"])
    sc.poly([P(-.3, -.07), P(-.3, .07), P(-.75, U(rng, .1, .2)), P(-.75, -U(rng, .1, .2))], c["wing"])
    sc.ell(P(-.4, 0), P(.4, 0), s * .13, c["body"])
    sc.circ(P(.45, 0), s * .1, c.get("head", c["body"]))
    sc.poly([P(.52, -.04), P(.52, .04), P(.72, 0)], c["beak"])


def bird_head(sc, rng, F, c):
    """Head and long neck of a big bird (ostrich, emu, heron ...) in close-up, beak towards +u."""
    P = F.p; s = F.s
    j = lambda v, a=.06: v + rng.normal(0, a)
    hd = (j(.1), j(-.35))
    sc.path(F.pp([(j(-.3, .2), 1.6), (j(-.15, .1), .6), hd]), [s * U(rng, .3, .5), s * .24, s * .2], c.get("neck", c["body"]))
    sc.ell(P(hd[0] - .3, hd[1] - .02), P(hd[0] + .32, hd[1] + .04), s * U(rng, .26, .32), c.get("head", c["body"]))
    bl = U(rng, .4, .8)
    sc.poly([P(hd[0] + .22, hd[1] - .12), P(hd[0] + .22, hd[1] + .2), P(hd[0] + .3 + bl, hd[1] + j(.12, .04))], c["beak"])
    sc.circ(P(hd[0] + .08, hd[1] - .06), s * .085, (.05, .04, .04))
    sc.circ(P(hd[0] + .1, hd[1] - .08), s * .025, (.8, .8, .8))


def frog_side(sc, rng, F, ct, c):
    """Frog sitting, seen from the side / three-quarters; head towards +u."""
    P = F.p; s = F.s
    j = lambda v, a=.03: v + rng.normal(0, a)
    tilt = j(.25, .08)
    leg = c.get("leg", ct)
    sc.seg(P(-.35, .6), P(j(.1, .08), .68), s * .06, s * .045, leg)                 # hind foot
    sc.ell(P(-.8, j(.5)), P(-.2, .6), s * .11, leg)                                # shank
    sc.ell(P(-.7, .25 + tilt), P(.55, .25 - tilt * 2), s * U(rng, .34, .42), ct)   # body
    sc.ell(P(-.78, j(.2)), P(-.18, j(.48)), s * U(rng, .15, .2), leg)              # thigh
    sc.ell(P(.2, -.02 - tilt), P(.88, .1 - tilt), s * U(rng, .22, .28), ct)        # head
    if "belly" in c:
        sc.ell(P(.0, .35), P(.8, .18 - tilt), s * .15, c["belly"])
    ex, ey = j(.5), -.2 - tilt
    sc.circ(P(ex, ey), s * .135, ct)
    sc.circ(P(ex + .02, ey), s * .09, c.get("eye", (.75, .6, .2)))
    sc.ell(P(ex - .05, ey), P(ex + .09, ey), s * .035, (.04, .04, .04))
    sc.seg(P(.42, .12 - tilt), P(.9, .1 - tilt), s * .014, s * .014, (.1, .08, .05))
    kx = j(.48, .05)
    sc.seg(P(.38, .2), P(kx, .6), s * .085, s * .06, leg)                          # front leg
    sc.seg(P(kx, .62), P(kx + j(.2, .05), .66), s * .045, s * .035, leg)


def frog_top(sc, rng, F, ct, c):
    """Frog seen from above, head towards +u, legs folded or stretched."""
    P = F.p; s = F.s
    j = lambda v, a=.05: v + rng.normal(0, a)
    leg = c.get("leg", ct)
    stretch = rng.random() < .3
    for sg in (-1, 1):
        if stretch:
            k1 = (j(-.8), sg * j(.5)); k2 = (j(-1.3), sg * j(.45)); k3 = (j(-1.6), sg * j(.6))
        else:
            k1 = (j(-.15), sg * j(.75)); k2 = (j(-.7), sg * j(.62)); k3 = (j(-.6), sg * j(.95))
        sc.seg(P(-.45, sg * .28), P(*k1), s * .15, s * .1, leg)
        sc.seg(P(*k1), P(*k2), s * .09, s * .07, leg)
        sc.seg(P(*k2), P(*k3), s * .05, s * .035, leg)
        e = (j(.5), sg * j(.62)); f = (j(.72), sg * j(.75))
        sc.seg(P(.3, sg * .28), P(*e), s * .075, s * .055, leg)
        sc.seg(P(*e), P(*f), s * .045, s * .035, leg)
    sc.ell(P(-.62, 0), P(.45, 0), s * U(rng, .36, .45), ct)
    sc.ell(P(.15, 0), P(.9, 0), s * U(rng, .28, .34), ct)
    for sg in (-1, 1):
        sc.circ(P(.56, sg * .27), s * .1, ct)
        sc.circ(P(.58, sg * .29), s * .06, (.05, .05, .04))
    if "stripe" in c:
        sc.seg(P(-.5, 0), P(.6, 0), s * .04, s * .03, c["stripe"])


def frog_front(sc, rng, F, ct, c):
    """Frog facing the camera: wide body, two eye bumps on top, pale throat, feet at the lower corners."""
    P = F.p; s = F.s
    j = lambda v, a=.04: v + rng.normal(0, a)
    leg = c.get("leg", ct)
    for sg in (-1, 1):
        sc.ell(P(sg * .7, .25), P(sg * j(1.0), j(.72)), s * .17, leg)
    sc.ell(P(-.78, .2), P(.78, .2), s * U(rng, .45, .55), ct)
    if "belly" in c:
        sc.ell(P(-.48, .36), P(.48, .36), s * U(rng, .24, .32), c["belly"])
    for sg in (-1, 1):
        sc.circ(P(sg * .4, -.3), s * .175, ct)
        sc.circ(P(sg * .4, -.32), s * .11, c.get("eye", (.75, .6, .2)))
        sc.ell(P(sg * .4 - .08, -.32), P(sg * .4 + .08, -.32), s * .04, (.04, .04, .04))
        sc.seg(P(sg * .45, .45), P(sg * j(.55), .88), s * .1, s * .075, leg)
        sc.seg(P(sg * .55, .9), P(sg * j(.8), .93), s * .05, s * .04, leg)
    sc.seg(P(-.62, .02), P(.62, .02), s * .016, s * .016, (.1, .08, .05))


def horse_face(sc, rng, F, ct, c, chest=True):
    """Horse's head seen from the front: long narrow face, small ears, maybe a white blaze."""
    P = F.p; s = F.s
    if chest:
        sc.seg(P(0, .3), P(rng.normal(0, .2), 2.6), s * .5, s * .9, ct)
    if "mane" in c:
        sc.ell(P(-.1, -1.0), P(.15, -.3), s * .3, c["mane"])
    for sg in (-1, 1):
        sc.poly([P(sg * .12, -.85), P(sg * U(rng, .3, .42), -U(rng, 1.25, 1.45)), P(sg * .42, -.8)], ct)
    sc.seg(P(0, -.6), P(0, .75), s * .42, s * .27, ct)
    if "blaze" in c:
        sc.seg(P(0, -.6), P(0, .7), s * U(rng, .06, .16), s * U(rng, .08, .2), c["blaze"])
    for sg in (-1, 1):
        sc.circ(P(sg * .33, -.38), s * .07, (.05, .04, .04))
        sc.circ(P(sg * .12, .8), s * .05, (.08, .06, .06))


# =====================================================================================================
# 5. One scene per call for each class: background + object in a typical pose, colour and size.
# =====================================================================================================
DEG = np.pi / 180
DARK = (.09, .08, .07)
SKIN = [(.85, .65, .55), (.75, .55, .45), (.55, .38, .28), (.92, .75, .65)]
CLOTH = [(.8, .1, .1), (.15, .2, .5), (.9, .9, .9), (.1, .1, .12), (.2, .45, .25), (.9, .75, .2), (.5, .5, .55), (.85, .4, .6)]


def fence(sc, rng, gy):
    col = jit(rng, pick(rng, [(.9, .9, .88), (.45, .33, .22), (.6, .58, .55), (.3, .25, .2)]), .08)
    n = int(rng.integers(2, 5)); h = U(rng, .1, .28); t = U(rng, .008, .016)
    for k in range(n):
        y = gy - h * (k + .6) / n
        sc.rect(0, y - t, 1, y + t, col)
    x = U(rng, 0, .3)
    while x < 1:
        sc.rect(x - t * 1.3, gy - h * 1.05, x + t * 1.3, gy + .02, col)
        x += U(rng, .2, .4)


def shadow(sc, F, G, a=.35, w=.6):
    sc.ell(F.p(-w, G), F.p(w, G), F.s * .07, (.05, .05, .04), a)


def vehicle_scene(rng, o, R, s, cx, cy, p_plain, backs, bp, grounds, gp, lo_hi):
    sc = Canvas(rng)
    lo, hi = o.bbox(R)
    ytop, ybot = cy - s * hi[1], cy - s * lo[1]
    if rng.random() < p_plain:      # cut-out / studio picture
        sc.img = plain(rng, jit(rng, pick(rng, [(.96, .96, .96), (.96, .96, .96), (.82, .82, .84), (.08, .08, .08), (.5, .5, .55)]), .04, .01))
    else:
        gy = ybot - U(rng, *lo_hi) * (ybot - ytop)
        sc.img = outdoor(rng, gy, None, pick(rng, backs, bp), pick(rng, grounds, gp))
        clutter(sc, rng, gy - .05, gy + .03, int(rng.integers(0, 6)))
    o.render(sc, R, s, cx, cy, light=(rng.normal(0, .4), .8, .5))
    return sc.img


def scene_airplane(rng):
    sc = Canvas(rng)
    o, typ = make_plane(rng)
    if rng.random() < .36:          # standing or rolling on the ground
        gb = plane_gear(o, rng)
        R = rot3(veh_view(rng, .4, .25, .07, .2), U(rng, 1, 10) * DEG)
        s, cx, cy = o.fit(R, .5 + rng.normal(0, .04), U(rng, .45, .62), U(rng, .8, 1.05), U(rng, .4, .75))
        gy = cy - s * (R @ np.array([0, gb, 0]))[1] - U(rng, .0, .1)
        sc.img = outdoor(rng, gy, pick(rng, SKY_AIR), pick(rng, [None, None, "hills", "trees", "buildings", "wall"]),
                         pick(rng, ["asph", "conc", "conc", "grass", "grass", "dry"]), bh=U(rng, .04, .25))
    else:
        R = rot3(veh_view(rng, .5, .2, .07, .15), float(np.clip(rng.normal(-15, 30), -75, 40)) * DEG, rng.normal(0, 18) * DEG)
        big = rng.random() < .88
        s, cx, cy = o.fit(R, .5 + rng.normal(0, .06), .5 + rng.normal(0, .07),
                          U(rng, .8, 1.1) if big else U(rng, .45, .75), U(rng, .65, 1.0) if big else U(rng, .3, .55))
        q = rng.random()
        if q < .12:
            sc.img = outdoor(rng, U(rng, .7, .95), pick(rng, SKY_AIR), pick(rng, [None, "hills", "trees"]), pick(rng, ["grass", "dry", "water", "dirt"]), bh=U(rng, .03, .1))
        elif q < .26:                 # looking down on the plane: land, fields or sea behind it
            R = rot3(U(rng, 0, 2 * np.pi), U(rng, 35, 80) * DEG, rng.normal(0, 10) * DEG)
            s, cx, cy = o.fit(R, .5 + rng.normal(0, .06), .5 + rng.normal(0, .07), U(rng, .75, 1.05), U(rng, .6, .95))
            sc.img = ground(rng, pick(rng, ["grass", "grass", "dry", "dirt", "asph", "conc", "water", "sand"]))
        else:
            sc.img = sky(rng, pick(rng, SKY_AIR))
    o.render(sc, R, s, cx, cy, light=(rng.normal(0, .4), .8, .5))
    return sc.img


def scene_automobile(rng):
    o = make_car(rng)
    R = rot3(veh_view(rng, .25, .40, .12, .15), U(rng, 4, 26) * DEG)
    s, cx, cy = o.fit(R, .5 + rng.normal(0, .03), U(rng, .52, .62), U(rng, .85, 1.08), U(rng, .48, .78))
    return vehicle_scene(rng, o, R, s, cx, cy, .12, ["buildings", "trees", "wall", "hills", None, "forest"], [.25, .32, .15, .05, .15, .08],
                         ["asph", "conc", "grass", "dirt", "sand", "snow", "dry"], [.5, .15, .13, .1, .04, .03, .05], (.25, 1.0))


def scene_truck(rng):
    o = make_truck(rng)
    R = rot3(veh_view(rng, .15, .58, .15, .08), U(rng, 0, 12) * DEG)
    s, cx, cy = o.fit(R, .5 + rng.normal(0, .03), U(rng, .48, .55), U(rng, .88, 1.1), U(rng, .72, 1.0))
    return vehicle_scene(rng, o, R, s, cx, cy, .16, [None, "trees", "buildings", "hills", "wall", "forest"], [.32, .28, .18, .1, .07, .05],
                         ["asph", "conc", "dirt", "grass", "sand", "snow", "dry"], [.45, .1, .22, .1, .05, .04, .04], (.12, .55))


def scene_ship(rng):
    sc = Canvas(rng)
    o, typ = make_ship(rng)
    high = rng.random() < (.45 if typ == "speed" else .22)
    R = rot3(veh_view(rng, .45, .25, .08, .15), (U(rng, 16, 50) if high else U(rng, 2, 13)) * DEG)
    s, cx, cy = o.fit(R, .5 + rng.normal(0, .05), U(rng, .5, .62), U(rng, .7, 1.05), U(rng, .45, .9))
    gy = -1.0 if (high and rng.random() < .7) else cy - U(rng, .02, .28)
    sc.img = outdoor(rng, gy, None, pick(rng, [None, None, None, "hills", "hills", "buildings", "trees"]), "water", bh=U(rng, .03, .15))
    o.render(sc, R, s, cx, cy, light=(rng.normal(0, .4), .8, .5))
    return sc.img


HORSE_COAT = [(.42, .25, .13), (.55, .30, .15), (.15, .11, .09), (.86, .85, .81), (.80, .62, .38), (.65, .50, .32), (.30, .20, .14), (.55, .52, .50)]
HORSE_P = [.26, .20, .14, .14, .06, .07, .09, .04]


def horse_cols(rng):
    k = int(rng.choice(len(HORSE_COAT), p=HORSE_P))
    base = jit(rng, HORSE_COAT[k], .1, .02)
    c = {}
    if k in (0, 5, 6):
        c["mane"] = c["tail"] = DARK
        if rng.random() < .6:
            c["sock"] = DARK
    elif k in (3, 7):
        c["mane"] = c["tail"] = np.clip(base * U(rng, .8, 1.05), 0, 1)
    elif k == 4 or rng.random() < .3:
        c["mane"] = c["tail"] = jit(rng, (.9, .85, .7), .05)
    else:
        c["mane"] = c["tail"] = np.clip(base * U(rng, .6, .9), 0, 1)
    if rng.random() < .2:
        c["sock"] = (.9, .9, .88)
    if rng.random() < .3:
        c["blaze"] = (.92, .92, .9)
    kind, col2 = ("patch", (.92, .92, .9)) if rng.random() < .08 else ("plain", None)
    return base, c, kind, col2


def animal_ground(rng, sc, feet_y, size, opts, p, allg=.0, sk=None):
    """Outdoor background for a standing animal: (backdrop, ground) pairs with probabilities p."""
    back, gk = pick(rng, opts, p)
    gy = -1.0 if rng.random() < allg else feet_y - U(rng, .1, 1.1) * size
    sc.img = outdoor(rng, gy, sk, back, gk)
    return gy


def scene_horse(rng):
    sc = Canvas(rng)
    base, c, kind, col2 = horse_cols(rng)
    r = rng.random()
    flip = pick(rng, [-1, 1])
    out = [("trees", "grass"), (None, "grass"), ("hills", "grass"), ("trees", "dirt"), ("wall", "dirt"), ("buildings", "dirt"), ("trees", "dry"),
           (None, "sand"), ("trees", "snow"), ("forest", "grass"), ("wall", "conc"), ("hills", "dry"), ("trees", "sand")]
    outp = [.26, .08, .08, .12, .08, .04, .08, .03, .03, .06, .05, .05, .04]
    if r < .64:                      # whole horse from the side
        s = U(rng, .38, .56)
        F = Frame(.5 + rng.normal(0, .05), U(rng, .42, .52), s, rng.normal(0, .05), flip)
        sp = spec_horse(rng)
        gy = animal_ground(rng, sc, F.cy + s * sp["G"], s, out, outp)
        if rng.random() < .22:
            fence(sc, rng, gy)
        shadow(sc, F, sp["G"])
        if rng.random() < .28:
            c["rider"] = (jit(rng, pick(rng, CLOTH), .1), jit(rng, pick(rng, [(.15, .15, .2), (.85, .82, .75), (.3, .3, .4), (.1, .1, .1)]), .1), jit(rng, pick(rng, SKIN), .08))
        quadruped(sc, rng, F, sp, coat(rng, base, F, kind, col2), c)
    elif r < .88:                    # head and neck
        s = U(rng, .34, .5)
        F = Frame(.5 + rng.normal(0, .08), U(rng, .42, .55), s, rng.normal(0, .25), flip)
        q = rng.random()
        if q < .35:
            sc.img = bokeh(rng)
        elif q < .8:
            animal_ground(rng, sc, .9, .5, out, outp)
        else:
            sc.img = plain(rng, jit(rng, pick(rng, [(.1, .09, .08), (.3, .22, .15), (.8, .8, .8), (.5, .4, .3)]), .1))
        if rng.random() < .35:
            c["tack"] = jit(rng, pick(rng, [(.1, .08, .06), (.4, .25, .15), (.7, .1, .1), (.15, .2, .5)]), .1)
        horse_head(sc, rng, F, coat(rng, base, F, kind, col2), c)
    else:                            # facing the camera
        s = U(rng, .3, .42)
        F = Frame(.5 + rng.normal(0, .06), U(rng, .45, .55), s, rng.normal(0, .04), flip)
        animal_ground(rng, sc, F.cy + s * .95, s, out, outp)
        front_quad(sc, rng, F, coat(rng, base, F, kind, col2), c, horse_face, .22, .055, .95)
    return sc.img


DEER_COAT = [(.60, .42, .26), (.50, .36, .22), (.62, .38, .20), (.45, .40, .33), (.25, .18, .12), (.70, .55, .40)]


def deer_cols(rng):
    base = jit(rng, pick(rng, DEER_COAT, [.28, .22, .16, .14, .1, .1]), .1, .02)
    c = {"nose": DARK}
    if rng.random() < .6:
        c["belly"] = mixc(base, (.92, .88, .8), U(rng, .4, .9))
    if rng.random() < .5:
        c["rump"] = (.92, .9, .85)
    if rng.random() < .55:
        c["antler"] = jit(rng, pick(rng, [(.75, .68, .55), (.5, .4, .3), (.85, .8, .7), (.35, .27, .2)]), .08)
    if rng.random() < .5:
        c["muzzle"] = mixc(base, (.9, .86, .8), U(rng, .3, .8))
    kind, col2 = ("spots", (.92, .9, .85)) if rng.random() < .1 else ("plain", None)
    return base, c, kind, col2


def scene_deer(rng):
    sc = Canvas(rng)
    base, c, kind, col2 = deer_cols(rng)
    r = rng.random()
    flip = pick(rng, [-1, 1])
    out = [("forest", "grass"), ("trees", "grass"), ("forest", "litter"), ("trees", "dry"), ("hills", "dry"), ("forest", "snow"), ("trees", "snow"),
           (None, "grass"), ("hills", "grass"), ("forest", "dry"), ("trees", "dirt"), ("trees", "litter")]
    outp = [.18, .2, .08, .13, .07, .04, .03, .05, .05, .09, .04, .04]
    if r < .14:                      # face
        s = U(rng, .26, .4)
        F = Frame(.5 + rng.normal(0, .06), U(rng, .42, .55), s, rng.normal(0, .12), flip)
        if rng.random() < .5:
            sc.img = bokeh(rng)
        else:
            animal_ground(rng, sc, .9, .5, out, outp, .3)
        deer_face(sc, rng, F, coat(rng, base, F, kind, col2), c)
    elif r < .3:                     # standing, facing the camera
        s = U(rng, .3, .46)
        F = Frame(.5 + rng.normal(0, .08), U(rng, .48, .58), s, rng.normal(0, .04), flip)
        animal_ground(rng, sc, F.cy + s * .95, s, out, outp, .25)
        front_quad(sc, rng, F, coat(rng, base, F, kind, col2), c, deer_face, .2, .04, .95)
    else:                            # whole animal from the side
        s = U(rng, .32, .54)
        F = Frame(.5 + rng.normal(0, .08), U(rng, .44, .58), s, rng.normal(0, .05), flip)
        sp = spec_deer(rng)
        if rng.random() < .12:
            sc.img = bokeh(rng)
        else:
            animal_ground(rng, sc, F.cy + s * sp["G"], s, out, outp, .35)
        quadruped(sc, rng, F, sp, coat(rng, base, F, kind, col2), c)
    return sc.img


DOG_COAT = [(.92, .90, .86), (.80, .60, .35), (.70, .50, .30), (.40, .26, .15), (.10, .10, .10), (.50, .48, .46), (.88, .80, .65), (.60, .35, .18)]
DOG_P = [.2, .18, .1, .14, .18, .06, .08, .06]


def dog_cols(rng):
    base = jit(rng, pick(rng, DOG_COAT, DOG_P), .08, .02)
    c = {"flop": rng.random() < .65}
    if rng.random() < .4:
        c["ear"] = np.clip(base * U(rng, .45, .85), 0, 1)
    if rng.random() < .4:
        c["muzzle"] = mixc(base, pick(rng, [(.93, .92, .9), (.85, .75, .6), (.1, .1, .1)]), U(rng, .5, 1))
    if rng.random() < .25:
        c["belly"] = c["sock"] = (.92, .91, .88)
    if rng.random() < .2:
        c["patch"] = jit(rng, pick(rng, [(.1, .1, .1), (.4, .26, .15), (.6, .35, .18)]), .1)
    kind, col2 = ("patch", jit(rng, pick(rng, [(.93, .92, .9), (.1, .1, .1), (.4, .26, .15)]), .05)) if rng.random() < .28 else ("plain", None)
    return base, c, kind, col2


def pet_bg(rng, sc, p_in, p_grass, p_plain, feet_y=.85):
    r = rng.random()
    if r < p_in:
        sc.img = indoor(rng)
    elif r < p_in + p_grass:
        gy = -1.0 if rng.random() < .55 else feet_y - U(rng, .15, .7)
        sc.img = outdoor(rng, gy, None, pick(rng, ["trees", None, "wall", "trees", "buildings"]), pick(rng, ["grass", "grass", "grass", "dry"]))
    elif r < p_in + p_grass + p_plain:
        sc.img = plain(rng, jit(rng, pick(rng, [(.95, .95, .95), (.1, .1, .1), (.6, .6, .65), (.3, .4, .6), (.8, .75, .65), (.6, .2, .2)]), .08, .02))
    else:
        q = rng.random()
        if q < .4:
            sc.img = bokeh(rng)
        else:
            gy = -1.0 if rng.random() < .5 else feet_y - U(rng, .15, .7)
            sc.img = outdoor(rng, gy, None, pick(rng, ["wall", "buildings", None, "trees"]), pick(rng, ["asph", "conc", "sand", "snow", "dirt", "conc"]))


def scene_dog(rng):
    sc = Canvas(rng)
    base, c, kind, col2 = dog_cols(rng)
    r = rng.random()
    flip = pick(rng, [-1, 1])
    if r < .38:                      # face
        s = U(rng, .3, .5)
        F = Frame(.5 + rng.normal(0, .07), U(rng, .38, .55), s, rng.normal(0, .15), flip)
        pet_bg(rng, sc, .45, .25, .1)
        dog_face(sc, rng, F, coat(rng, base, F, kind, col2, amp=.15), c)
    elif r < .58:
        s = U(rng, .36, .52)
        F = Frame(.5 + rng.normal(0, .07), U(rng, .36, .48), s, rng.normal(0, .06), flip)
        pet_bg(rng, sc, .42, .3, .1, F.cy + s)
        sitting(sc, rng, F, coat(rng, base, F, kind, col2, amp=.15), c, dog_face, U(rng, .32, .42))
    elif r < .75:
        s = U(rng, .4, .58)
        F = Frame(.5 + rng.normal(0, .06), U(rng, .45, .62), s, rng.normal(0, .12), flip)
        pet_bg(rng, sc, .5, .28, .07, F.cy + s * .4)
        lying(sc, rng, F, coat(rng, base, F, kind, col2, amp=.15), c, dog_face, U(rng, .28, .36))
    else:
        s = U(rng, .4, .56)
        F = Frame(.5 + rng.normal(0, .05), U(rng, .42, .55), s, rng.normal(0, .06), flip)
        sp = spec_dog(rng)
        pet_bg(rng, sc, .3, .42, .08, F.cy + s * sp["G"])
        shadow(sc, F, sp["G"])
        quadruped(sc, rng, F, sp, coat(rng, base, F, kind, col2, amp=.15), c)
    return sc.img


def cat_cols(rng):
    k = int(rng.choice(8, p=[.25, .18, .1, .15, .1, .07, .08, .07]))
    c = {"eye": jit(rng, pick(rng, [(.6, .7, .3), (.8, .7, .2), (.4, .6, .7), (.5, .6, .2)]), .08)}
    kind, col2 = "plain", None
    if k == 0:
        base = jit(rng, (.50, .44, .36), .1, .02); kind, col2 = "stripes", np.clip(base * U(rng, .35, .6), 0, 1)
    elif k == 1:
        base = jit(rng, (.80, .50, .22), .08, .02); kind, col2 = "stripes", np.clip(base * U(rng, .7, .9), 0, 1)
    elif k == 2:
        base = jit(rng, (.50, .50, .52), .1, .01)
    elif k == 3:
        base = jit(rng, (.10, .09, .09), .1, .01)
    elif k == 4:
        base = jit(rng, (.92, .90, .88), .04, .01)
    elif k == 5:
        base = jit(rng, (.85, .75, .60), .06, .02); c["ear"] = c["tail"] = (.35, .28, .22)
    elif k == 6:
        base = jit(rng, (.92, .90, .88), .04, .01); kind, col2 = "patch", jit(rng, pick(rng, [(.8, .5, .22), (.12, .1, .1), (.5, .44, .36)]), .08)
    else:
        base = jit(rng, (.10, .09, .09), .1, .01); c["muzzle"] = c["belly"] = c["sock"] = (.92, .91, .9)
    if k in (0, 1, 2) and rng.random() < .45:
        c["muzzle"] = c["belly"] = (.92, .91, .9)
    return base, c, kind, col2


def scene_cat(rng):
    sc = Canvas(rng)
    base, c, kind, col2 = cat_cols(rng)
    r = rng.random()
    flip = pick(rng, [-1, 1])
    if r < .4:
        s = U(rng, .3, .5)
        F = Frame(.5 + rng.normal(0, .07), U(rng, .4, .58), s, rng.normal(0, .18), flip)
        pet_bg(rng, sc, .68, .1, .08)
        cat_face(sc, rng, F, coat(rng, base, F, kind, col2), c)
    elif r < .6:
        s = U(rng, .36, .52)
        F = Frame(.5 + rng.normal(0, .07), U(rng, .36, .48), s, rng.normal(0, .06), flip)
        pet_bg(rng, sc, .68, .1, .08, F.cy + s)
        sitting(sc, rng, F, coat(rng, base, F, kind, col2), c, cat_face, U(rng, .3, .4))
    elif r < .88:
        s = U(rng, .4, .58)
        F = Frame(.5 + rng.normal(0, .06), U(rng, .45, .62), s, rng.normal(0, .2), flip)
        pet_bg(rng, sc, .72, .08, .07, F.cy + s * .4)
        lying(sc, rng, F, coat(rng, base, F, kind, col2), c, cat_face, U(rng, .27, .36))
    else:
        s = U(rng, .38, .55)
        F = Frame(.5 + rng.normal(0, .05), U(rng, .45, .58), s, rng.normal(0, .06), flip)
        sp = spec_cat(rng)
        pet_bg(rng, sc, .55, .2, .07, F.cy + s * sp["G"])
        quadruped(sc, rng, F, sp, coat(rng, base, F, kind, col2), c)
    return sc.img


BIRD_COL = [(.45, .35, .25), (.50, .50, .50), (.10, .10, .10), (.92, .92, .90), (.20, .40, .75), (.80, .20, .12), (.90, .80, .20),
            (.50, .50, .25), (.85, .50, .15), (.30, .35, .42), (.70, .58, .42)]
BIRD_P = [.22, .14, .1, .08, .08, .06, .08, .08, .05, .06, .05]


def bird_cols(rng):
    body = jit(rng, pick(rng, BIRD_COL, BIRD_P), .1, .02)
    c = {"body": body, "wing": np.clip(body * U(rng, .45, .95), 0, 1),
         "belly": mixc(body, pick(rng, [(.93, .92, .88), (.85, .78, .6), (.9, .8, .3), (.85, .5, .2)], [.5, .25, .13, .12]), U(rng, .2, 1)),
         "beak": jit(rng, pick(rng, [(.15, .14, .13), (.9, .7, .15), (.85, .45, .1), (.5, .48, .45), (.8, .3, .2)], [.45, .2, .12, .15, .08]), .08),
         "leg": jit(rng, pick(rng, [(.2, .18, .16), (.7, .5, .45), (.8, .65, .2), (.45, .42, .4)]), .1)}
    if rng.random() < .3:
        c["head"] = jit(rng, pick(rng, BIRD_COL), .1, .02)
    return c


def scene_bird(rng):
    sc = Canvas(rng)
    c = bird_cols(rng)
    r = rng.random()
    flip = pick(rng, [-1, 1])
    if r < .42:                      # small bird on a perch
        s = U(rng, .28, .48)
        F = Frame(.5 + rng.normal(0, .07), U(rng, .4, .55), s, rng.normal(0, .08), flip)
        q = rng.random()
        if q < .4:
            sc.img = bokeh(rng)
        elif q < .6:
            sc.img = sky(rng)
        elif q < .78:
            sc.img = foliage(rng)
        elif q < .92:
            sc.img = litterbg(rng, pick(rng, ["litter", "soil", "moss", "rock", "sandy"]))
        else:
            sc.img = plain(rng, jit(rng, pick(rng, [(.95, .95, .95), (.6, .6, .6), (.8, .85, .8)]), .05, .02))
        py = F.cy + s * (.42 + .35)
        pc = jit(rng, pick(rng, [(.35, .25, .17), (.5, .45, .4), (.25, .2, .15), (.6, .5, .38), (.3, .35, .2)]), .1)
        k = rng.random()
        if k < .6:                   # branch
            a = rng.normal(0, .25)
            sc.seg((-.1, py - a * .6 + .03), (1.1, py + a * .6 + .03), U(rng, .015, .05), U(rng, .012, .04), pc)
        elif k < .8:                 # post or stump
            w = U(rng, .05, .14)
            sc.rect(F.cx - w, py, F.cx + w, 1.0, pc)
        elif k < .9:                 # wire
            sc.seg((-.1, py), (1.1, py + rng.normal(0, .05)), .008, .008, (.15, .15, .15))
        else:                        # rock / ground
            sc.ell((-.2, py + .3), (1.2, py + .3), .3, pc)
        bird_side(sc, rng, F, c)
    elif r < .54:                    # on the ground
        s = U(rng, .26, .44)
        F = Frame(.5 + rng.normal(0, .07), U(rng, .4, .55), s, rng.normal(0, .06), flip)
        lg = U(rng, .35, .7)
        animal_ground(rng, sc, F.cy + s * (.42 + lg), s, [(None, "grass"), ("trees", "grass"), (None, "sand"), (None, "dirt"), (None, "dry"), (None, "snow"),
                                                          ("trees", "dry"), (None, "conc"), (None, "water")], [.25, .15, .12, .12, .12, .05, .08, .06, .05], .5)
        bird_side(sc, rng, F, c, legs=lg)
    elif r < .66:                    # tall long-necked bird (ostrich, emu, heron, crane)
        s = U(rng, .26, .38)
        F = Frame(.5 + rng.normal(0, .07), U(rng, .5, .6), s, rng.normal(0, .05), flip)
        lg = U(rng, .6, 1.0)
        animal_ground(rng, sc, F.cy + s * (.4 + lg), s, [(None, "grass"), ("trees", "grass"), ("trees", "dry"), (None, "dry"), ("hills", "dry"), (None, "sand"),
                                                         (None, "water"), ("wall", "dirt")], [.2, .2, .15, .12, .1, .08, .1, .05], .3)
        if rng.random() < .6:
            g = U(rng, .2, .5); c["body"] = c["wing"] = jit(rng, (g, g * .95, g * .9), .1); c["neck"] = c["head"] = jit(rng, pick(rng, [(.7, .6, .55), (.5, .45, .4), (.85, .85, .85)]), .1)
        bird_side(sc, rng, F, c, legs=lg, neck=U(rng, .7, 1.2))
    elif r < .80:                    # head of a big bird in close-up
        s = U(rng, .32, .5)
        F = Frame(.5 + rng.normal(0, .08), U(rng, .45, .6), s, rng.normal(0, .15), flip)
        q = rng.random()
        if q < .5:
            sc.img = bokeh(rng)
        elif q < .85:
            animal_ground(rng, sc, 1.0, .5, [(None, "grass"), ("trees", "grass"), ("trees", "dry"), ("wall", "dirt"), (None, "dry")], [.25, .25, .2, .15, .15], .3)
        else:
            sc.img = sky(rng)
        if rng.random() < .7:
            g = U(rng, .35, .7); c["body"] = jit(rng, (g, g * .93, g * .88), .1, .03)
        bird_head(sc, rng, F, c)
    elif r < .90:                    # flying
        s = U(rng, .25, .46)
        F = Frame(.5 + rng.normal(0, .08), .5 + rng.normal(0, .08), s, U(rng, 0, 2 * np.pi), flip)
        if rng.random() < .8:
            sc.img = sky(rng)
        else:
            sc.img = outdoor(rng, U(rng, .5, .9), None, pick(rng, [None, "hills", "trees"]), pick(rng, ["water", "grass", "water"]))
        if rng.random() < .4:
            c["body"] = c["wing"] = jit(rng, pick(rng, [(.12, .12, .13), (.92, .92, .92), (.35, .3, .25)]), .1)
        bird_fly(sc, rng, F, c)
    else:                            # swimming
        s = U(rng, .24, .4)
        F = Frame(.5 + rng.normal(0, .07), U(rng, .45, .6), s, rng.normal(0, .04), flip)
        gy = -1.0 if rng.random() < .7 else F.cy - U(rng, .15, .4)
        sc.img = outdoor(rng, gy, None, pick(rng, [None, "trees", "hills"]), "water")
        wat = sc.img.copy()
        bird_side(sc, rng, F, c, legs=.0, neck=U(rng, .0, .6) if rng.random() < .6 else 0.0)
        m = PY2 > F.cy + s * U(rng, .12, .25)       # the water hides the underside
        sc.img[m] = wat[m]
    return sc.img


FROG_COAT = [(.35, .60, .15), (.50, .62, .20), (.20, .35, .12), (.42, .42, .20), (.45, .33, .20), (.45, .42, .36), (.72, .50, .25),
             (.20, .18, .12), (.50, .50, .48), (.80, .30, .10)]
FROG_P = [.18, .08, .1, .15, .17, .11, .08, .06, .04, .03]


def frog_cols(rng):
    base = jit(rng, pick(rng, FROG_COAT, FROG_P), .1, .02)
    c = {"eye": jit(rng, pick(rng, [(.75, .6, .2), (.85, .45, .1), (.7, .15, .1), (.15, .12, .1), (.6, .55, .4)]), .08)}
    if rng.random() < .7:
        c["belly"] = jit(rng, pick(rng, [(.90, .88, .72), (.93, .93, .9), (.85, .80, .40), (.75, .7, .55)]), .06)
    if rng.random() < .25:
        c["leg"] = np.clip(base * U(rng, .6, 1.2), 0, 1)
    if rng.random() < .1:
        c["stripe"] = jit(rng, (.8, .75, .5), .1)
    r = rng.random()
    kind, col2 = "plain", None
    if r < .5:
        kind, col2 = "spots", np.clip(base * U(rng, .3, .65), 0, 1)
    elif r < .6:
        kind, col2 = "patch", jit(rng, pick(rng, FROG_COAT), .1)
    return base, c, kind, col2


def scene_frog(rng):
    sc = Canvas(rng)
    base, c, kind, col2 = frog_cols(rng)
    q = rng.random()
    if q < .6:
        sc.img = litterbg(rng)
    elif q < .72:
        sc.img = plain(rng, jit(rng, pick(rng, [(.96, .96, .96), (.96, .96, .96), (.08, .08, .08), (.6, .6, .6)]), .04, .01))
    elif q < .95:
        sc.img = bokeh(rng, [(.30, .45, .18), (.45, .52, .25), (.18, .28, .12), (.40, .33, .22), (.25, .22, .18), (.5, .45, .3)])
    else:
        sc.img = tex(rng, jit(rng, pick(rng, SKIN), .08), .06, (2, 5, 12), (1, .6, .4))
    r = rng.random()
    flip = pick(rng, [-1, 1])
    if r < .5:
        s = U(rng, .34, .5) if rng.random() < .85 else U(rng, .5, .7)
        F = Frame(.5 + rng.normal(0, .06), U(rng, .42, .56), s, rng.normal(0, .15), flip)
        ct = coat(rng, base, F, kind, col2, amp=.2)
        shadow(sc, F, .68, .4, .8)
        frog_side(sc, rng, F, ct, c)
    elif r < .78:
        s = U(rng, .22, .36)
        F = Frame(.5 + rng.normal(0, .06), .5 + rng.normal(0, .06), s, U(rng, 0, 2 * np.pi), flip)
        frog_top(sc, rng, F, coat(rng, base, None, kind, col2, amp=.2), c)
    else:
        s = U(rng, .32, .48)
        F = Frame(.5 + rng.normal(0, .06), U(rng, .42, .56), s, rng.normal(0, .08), flip)
        shadow(sc, F, .9, .4, .9)
        frog_front(sc, rng, F, coat(rng, base, F, kind, col2, amp=.2), c)
    return sc.img


SCENES = [scene_airplane, scene_automobile, scene_bird, scene_cat, scene_deer, scene_dog, scene_frog, scene_horse, scene_ship, scene_truck]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def draw(k, rng):
    """One 'photo' (3,32,32) of class k."""
    return finish(rng, SCENES[k](rng))


# =====================================================================================================
# 6. The network.  Fixed feature layers from formulas, then one row of weights per drawn scene.
# =====================================================================================================
class Net:
    def __init__(self):
        self.layers, self.w = [], {}

    def op(self, name, op, inputs, **kw):
        self.layers.append(dict(name=name, op=op, inputs=list(inputs), **kw))
        return name

    def conv(self, name, inp, W, b, stride=1, padding=0):
        W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
        self.w[name + ".W"], self.w[name + ".b"] = W, b
        return self.op(name, "conv", [inp], out=int(W.shape[0]), kernel=[int(W.shape[2]), int(W.shape[3])], stride=stride, padding=padding)

    def dense(self, name, inp, W, b):
        self.w[name + ".W"], self.w[name + ".b"] = np.asarray(W), np.asarray(b)
        return self.op(name, "dense", [inp], out=int(W.shape[0]))

    def relu(self, name, inp):
        return self.op(name, "relu", [inp])

    def norm(self, name, inp):
        return self.op(name, "norm", [inp])

    def pool(self, name, inp, kind, size, stride):
        return self.op(name, kind, [inp], size=list(size), stride=list(stride))

    def reshape(self, name, inp, shape):
        return self.op(name, "reshape", [inp], shape=list(shape))

    def add(self, name, inps):
        return self.op(name, "add", inps)


def _conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def _pool(x, size, stride, kind):
    n, c, h, w = x.shape
    (sh, sw), (th, tw) = size, stride
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if kind == "maxpool" else win.mean(0)


def forward(net, images, want):
    """Run the layers defined so far (same arithmetic as the scorer) and return the named activations."""
    vals = {"image": images}
    for L in net.layers:
        ins = [vals[i] for i in L["inputs"]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = _conv(ins[0], net.w[name + ".W"], net.w[name + ".b"], L["stride"], L["padding"])
        elif op == "dense":
            vals[name] = ins[0].reshape(len(ins[0]), -1) @ net.w[name + ".W"].T + net.w[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op in ("maxpool", "avgpool"):
            vals[name] = _pool(ins[0], L["size"], L["stride"], op)
        elif op == "add":
            vals[name] = sum(ins)
        elif op == "reshape":
            vals[name] = ins[0].reshape((len(ins[0]),) + tuple(L["shape"]))
        elif op == "norm":
            x = ins[0]
            mu = x.mean(axis=(2, 3), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
    return [vals[k] for k in want]


def w1x1(M):
    return np.asarray(M, float)[:, :, None, None]


def gabor_bank(n_or=6, k=7, sigma=1.4, lam=4.5):
    """Odd (edge) and even (line) Gabor filters at n_or orientations, zero mean, sum of |weights| = 1."""
    r = np.arange(k) - k // 2
    Y, X = np.meshgrid(r, r, indexing="ij")
    odd, even = [], []
    for i in range(n_or):
        th = np.pi * i / n_or
        u = X * np.cos(th) + Y * np.sin(th)
        env = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        o = env * np.sin(2 * np.pi * u / lam)
        e = env * np.cos(2 * np.pi * u / lam); e = e - env * (e.sum() / env.sum())
        odd.append(o / np.abs(o).sum()); even.append(e / np.abs(e).sum())
    return np.array(odd), np.array(even)


def front_end(net):
    """Feature layers. Returns the names (and sizes) of the feature maps the templates are compared on."""
    feats = []
    # ---- (A) colour: soft 'how much of this cell is ...' steps along 12 hue directions and 6 brightness levels
    dirs = []
    for a, b_ in ((0, 1), (0, 2), (1, 2)):               # R-G, R-B, G-B and their opposites
        v = np.zeros(3); v[a], v[b_] = 1, -1; dirs += [v, -v]
    for a in range(3):                                   # R-(G+B)/2 ... and their opposites
        v = np.full(3, -.5); v[a] = 1; dirs += [v, -v]
    units = [(d, t, w) for d in dirs for t, w in ((.04, .06), (.12, .10), (.25, .15))]
    units += [(np.full(3, 1 / 3), t, .15) for t in (.12, .27, .42, .57, .72, .87)]
    nu = len(units)
    W = np.zeros((2 * nu, 3)); b = np.zeros(2 * nu)
    for i, (d, t, wd) in enumerate(units):               # step(u) = relu((u-t)/w) - relu((u-t)/w - 1), between 0 and 1
        W[2 * i] = W[2 * i + 1] = d / wd
        b[2 * i], b[2 * i + 1] = -t / wd, -t / wd - 1
    net.conv("col_a", "image", w1x1(W), b)
    net.relu("col_b", "col_a")
    M = np.zeros((nu, 2 * nu))
    for i in range(nu):
        M[i, 2 * i], M[i, 2 * i + 1] = 1, -1
    net.conv("col_c", "col_b", w1x1(M), np.zeros(nu))
    net.pool("col_p", "col_c", "avgpool", (8, 8), (8, 8))                 # (42, 4, 4)
    feats.append(("col_p", (nu, 4, 4), "A"))
    # ---- (B) shape: Gabor edge + line energy of the contrast-normalised grey image, two scales
    net.conv("lum", "image", w1x1([[.299, .587, .114]]), np.zeros(1))
    net.norm("lumn", "lum")
    odd, even = gabor_bank()
    G = np.concatenate([odd, -odd, even, -even])[:, None]                 # (24, 1, 7, 7): +/- so relu gives |.|
    no = len(odd)
    clip = 0.5
    E = np.zeros((2 * no, 4 * no)); eb = np.zeros(2 * no)
    for i in range(no):
        E[i, [i, no + i, 2 * no + i, 3 * no + i]] = 1                     # energy = |odd| + |even|
        E[no + i] = E[i]; eb[no + i] = -clip                              # energy - clip
    C = np.zeros((no, 2 * no))
    for i in range(no):
        C[i, i], C[i, no + i] = 1, -1                                     # min(energy, clip)
    net.pool("lum2", "lumn", "avgpool", (2, 2), (2, 2))
    for tag, src, ps, pst, shape in (("f", "lumn", (6, 6), (4, 4), (no, 6, 6)), ("c", "lum2", (4, 4), (2, 2), (no, 4, 4))):
        net.conv("gab_" + tag, src, G, np.zeros(4 * no))
        net.relu("gab_r" + tag, "gab_" + tag)
        net.conv("en_" + tag, "gab_r" + tag, w1x1(E), eb)
        net.relu("en_r" + tag, "en_" + tag)
        net.conv("ec_" + tag, "en_r" + tag, w1x1(C), np.zeros(no))
        net.pool("ep_" + tag, "ec_" + tag, "avgpool", ps, pst)
        net.reshape("eq_" + tag, "ep_" + tag, [1, shape[0] * shape[1], shape[2]])
        net.norm("en_n" + tag, "eq_" + tag)                               # same numbers, standardised within the photo
        feats.append(("en_n" + tag, shape, "B"))
    # ---- (C) layout: where the unusual colours are (figure against ground) and where it is light or dark
    net.norm("imn", "image")
    net.conv("sal_a", "imn", w1x1(np.concatenate([np.eye(3), -np.eye(3)])), np.zeros(6))
    net.relu("sal_b", "sal_a")
    net.conv("sal_c", "sal_b", w1x1([[1 / 3] * 6, [1 / 3] * 6]), np.array([0, -1.5]))
    net.relu("sal_d", "sal_c")
    net.conv("sal_e", "sal_d", w1x1([[1, -1]]), np.zeros(1))              # min(mean |z|, 1.5)
    net.pool("sal_p", "sal_e", "avgpool", (8, 8), (4, 4))                 # (1, 7, 7)
    feats.append(("sal_p", (1, 7, 7), "C"))
    net.pool("lay_p", "lumn", "avgpool", (8, 8), (4, 4))                  # (1, 7, 7)
    feats.append(("lay_p", (1, 7, 7), "C"))
    return feats


def pyramid_metric(h, w, blocks):
    """Matrix M with  d^T M d = sum over levels of (mean over that level's regions of the squared region-average of d).
    blocks: list of (bh, bw) cell sizes, e.g. (1,1) = every cell, (h,w) = the whole map; or "ring" = two regions,
    the middle of the picture (where the object usually is) and the frame around it."""
    M = np.zeros((h * w, h * w))
    for blk in blocks:
        rows = []
        if blk == "ring":
            a = np.zeros((h, w)); a[1:h - 1, 1:w - 1] = 1
            rows = [(a / a.sum()).ravel(), ((1 - a) / (1 - a).sum()).ravel()]
            blk = (h + 1, w + 1)
        bh, bw = blk
        for y in range(0, h - bh + 1, bh):
            for x in range(0, w - bw + 1, bw):
                a = np.zeros((h, w)); a[y:y + bh, x:x + bw] = 1.0 / (bh * bw)
                rows.append(a.ravel())
        P = np.array(rows) / np.sqrt(len(rows))
        M += P.T @ P
    return M


METRIC = {"col_p": [(1, 1), (2, 2), (1, 4), (4, 4), "ring"],     # cells, quarters, horizontal bands, whole picture, middle / frame
          "en_nf": [(1, 1), (2, 2), (6, 6), "ring"],
          "en_nc": [(1, 1), (2, 2), (4, 4), "ring"],
          "sal_p": [(1, 1)],
          "lay_p": [(1, 1)]}
GROUP_WEIGHT = {"A": 2, "B": 1, "C": 1}                  # colour layout counts double
BETA, KNOTS = 5.0, [0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
GROUP = 4                                                # scenes per group; a group votes with its best-matching scene


def build(out_dir, per_class=6000, seed=20261120, verbose=True):
    net = Net()
    feats = front_end(net)
    # ---- draw the scenes, one template each
    rng = np.random.default_rng(seed)
    N = 10 * per_class
    labels = np.repeat(np.arange(10), per_class)
    F = {name: np.zeros((N, int(np.prod(shape)))) for name, shape, g in feats}
    B = 500
    for s0 in range(0, N, B):
        ims = np.stack([draw(int(k), rng) for k in labels[s0:s0 + B]])
        outs = forward(net, ims.astype(np.float32).astype(np.float64), [name for name, _, _ in feats])
        for (name, _, _), o in zip(feats, outs):
            F[name][s0:s0 + B] = o.reshape(len(o), -1)
        if verbose and (s0 // B) % 10 == 0:
            print("drawn", s0 + B, "of", N, flush=True)
    # ---- one dense row per scene: output = -(squared distance to that scene) + a term that is the same for all rows
    groups = {}
    for name, shape, g in feats:
        c, h, w = shape
        M = pyramid_metric(h, w, METRIC[name])
        T = F[name].reshape(N, c, h * w)
        TM = T @ M                                                      # each template times the metric, by itself
        net.dense("bank_" + name, name, (2 * TM).reshape(N, -1), -np.einsum("ncp,ncp->n", TM, T))
        groups.setdefault(g, []).append("bank_" + name)
    zs = []
    for g, names in groups.items():
        src = names[0] if len(names) == 1 else net.add("sum_" + g, names)
        net.reshape("row_" + g, src, [1, N, 1])
        net.norm("z_" + g, "row_" + g)                                  # standardise over the templates, per photo
        zs += ["z_" + g] * GROUP_WEIGHT[g]
    net.add("z_sum", zs)
    net.norm("z", "z_sum")
    assert per_class % GROUP == 0
    net.pool("zg", "z", "maxpool", (GROUP, 1), (GROUP, 1))             # templates are stored class by class, so groups are pure
    NG = N // GROUP
    glabels = labels[::GROUP]
    # ---- votes: a convex piecewise-linear stand-in for exp(BETA * (z - 3)), summed per class.
    #      It is zero below the first knot; the best-matching scene always has z above it.
    kn = KNOTS + [KNOTS[-1] + .5]
    v = [0.0] + [float(np.exp(BETA * (k - 3))) for k in kn[1:]]
    slopes = [(v[i + 1] - v[i]) / (kn[i + 1] - kn[i]) for i in range(len(KNOTS))]
    coef = [slopes[0]] + [slopes[i] - slopes[i - 1] for i in range(1, len(KNOTS))]
    knots = KNOTS
    J = len(knots)
    net.conv("vote_a", "zg", np.ones((J, 1, 1, 1)), -np.array(knots))
    net.relu("vote_b", "vote_a")
    Wc = np.zeros((10, J, NG), np.float32)
    for j in range(J):
        Wc[glabels, j, np.arange(NG)] = coef[j]
    net.dense("scores", "vote_b", Wc.reshape(10, J * NG), np.zeros(10))
    # ---- save
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": net.layers}, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: np.asarray(v_, np.float32) for k, v_ in net.w.items()})
    if verbose:
        print("saved", sum(int(np.prod(v_.shape)) for v_ in net.w.values()), "weights", flush=True)
    return net


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
