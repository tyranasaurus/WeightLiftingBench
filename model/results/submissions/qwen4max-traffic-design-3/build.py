"""Hand-designed traffic-sign recognition network.

Every weight comes from geometry / formulas written out below: analytic colour
opponency + Gaussian high-pass filters, and procedurally drawn models of the 43
German road signs (shape models typed from general knowledge of the signs).
No training data, no fitting, no statistics over images.

Usage:  python3 build.py OUT_DIR
"""
import sys, os, json, math
import numpy as np

# ----------------------------------------------------------------------------
# canonical geometry
# ----------------------------------------------------------------------------
N = 32
CEN = 15.5                 # centre of the 32x32 frame (pixel centres 0..31)
D0 = 32.0 / 1.2            # sign box size: crop keeps a 10% border -> 26.67 px
RES = 256                  # canonical render resolution (8 samples per pixel)
SS = RES // N

# colours of the sign materials (linear-ish sRGB values in 0..1)
RED = (0.72, 0.13, 0.15)
BLUE = (0.10, 0.28, 0.60)
YEL = (0.95, 0.72, 0.10)
WHT = (0.93, 0.93, 0.93)
BLK = (0.10, 0.10, 0.10)
GRY = (0.42, 0.42, 0.42)
BG = (0.42, 0.42, 0.42)

_g = (np.arange(RES) + 0.5) / SS - 0.5
GX, GY = np.meshgrid(_g, _g)          # GX[y,x] = x coordinate, GY[y,x] = y


# ----------------------------------------------------------------------------
# mask primitives (all coordinates in 32-space)
# ----------------------------------------------------------------------------
def _bbox(x0, x1, y0, y1):
    ix0 = max(0, int(math.floor((x0 + 0.5) * SS)) - 1)
    ix1 = min(RES, int(math.ceil((x1 + 0.5) * SS)) + 2)
    iy0 = max(0, int(math.floor((y0 + 0.5) * SS)) - 1)
    iy1 = min(RES, int(math.ceil((y1 + 0.5) * SS)) + 2)
    return iy0, iy1, ix0, ix1


def _put(iy0, iy1, ix0, ix1, m):
    out = np.zeros((RES, RES), bool)
    if iy1 > iy0 and ix1 > ix0:
        out[iy0:iy1, ix0:ix1] = m
    return out


def disc(cx, cy, r):
    iy0, iy1, ix0, ix1 = _bbox(cx - r, cx + r, cy - r, cy + r)
    x = GX[iy0:iy1, ix0:ix1] - cx
    y = GY[iy0:iy1, ix0:ix1] - cy
    return _put(iy0, iy1, ix0, ix1, x * x + y * y <= r * r)


def ell(cx, cy, rx, ry, ang=0.0):
    r = max(rx, ry)
    iy0, iy1, ix0, ix1 = _bbox(cx - r, cx + r, cy - r, cy + r)
    x = GX[iy0:iy1, ix0:ix1] - cx
    y = GY[iy0:iy1, ix0:ix1] - cy
    ca, sa = math.cos(ang), math.sin(ang)
    u = (ca * x + sa * y) / max(rx, 1e-9)
    v = (-sa * x + ca * y) / max(ry, 1e-9)
    return _put(iy0, iy1, ix0, ix1, u * u + v * v <= 1.0)


def ringm(cx, cy, r, t):
    return disc(cx, cy, r) & ~disc(cx, cy, max(r - t, 0.0))


def ellring(cx, cy, rx, ry, t, ang=0.0):
    return ell(cx, cy, rx, ry, ang) & ~ell(cx, cy, max(rx - t, 1e-4), max(ry - t, 1e-4), ang)


def seg(p0, p1, r):
    """capsule"""
    x0, y0 = p0; x1, y1 = p1
    iy0, iy1, ix0, ix1 = _bbox(min(x0, x1) - r, max(x0, x1) + r,
                               min(y0, y1) - r, max(y0, y1) + r)
    x = GX[iy0:iy1, ix0:ix1]; y = GY[iy0:iy1, ix0:ix1]
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        t = np.zeros_like(x)
    else:
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / L2, 0.0, 1.0)
    ex = x - (x0 + t * dx); ey = y - (y0 + t * dy)
    return _put(iy0, iy1, ix0, ix1, ex * ex + ey * ey <= r * r)


def stroke(pts, w):
    m = np.zeros((RES, RES), bool)
    for a, b in zip(pts[:-1], pts[1:]):
        m |= seg(a, b, w * 0.5)
    return m


def arc(cx, cy, r, a0, a1, w, n=16):
    pts = [(cx + r * math.cos(a0 + (a1 - a0) * i / n),
            cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
    return stroke(pts, w)


def poly(pts):
    """convex polygon, vertices in order"""
    P = np.asarray(pts, float)
    a = 0.0
    for i in range(len(P)):
        j = (i + 1) % len(P)
        a += P[i, 0] * P[j, 1] - P[j, 0] * P[i, 1]
    if a < 0:
        P = P[::-1]
    iy0, iy1, ix0, ix1 = _bbox(P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max())
    x = GX[iy0:iy1, ix0:ix1]; y = GY[iy0:iy1, ix0:ix1]
    m = np.ones(x.shape, bool)
    for i in range(len(P)):
        j = (i + 1) % len(P)
        ex, ey = P[j, 0] - P[i, 0], P[j, 1] - P[i, 1]
        m &= (ex * (y - P[i, 1]) - ey * (x - P[i, 0])) >= 0
    return _put(iy0, iy1, ix0, ix1, m)


def rpoly(pts, r, n=5):
    """convex polygon with corners rounded by radius r"""
    P = np.asarray(pts, float)
    a = 0.0
    for i in range(len(P)):
        j = (i + 1) % len(P)
        a += P[i, 0] * P[j, 1] - P[j, 0] * P[i, 1]
    if a < 0:
        P = P[::-1]
    K = len(P)
    out = []
    for i in range(K):
        v = P[i]; prev = P[(i - 1) % K]; nxt = P[(i + 1) % K]
        e1 = prev - v; e1 /= np.linalg.norm(e1)
        e2 = nxt - v; e2 /= np.linalg.norm(e2)
        bis = e1 + e2; bis /= np.linalg.norm(bis)
        cosal = float(np.dot(e1, bis))
        sinal = math.sqrt(max(1e-9, 1 - cosal * cosal))
        d = r / sinal
        c = v + bis * d
        t1 = v + e1 * (d * cosal)
        t2 = v + e2 * (d * cosal)
        a1 = math.atan2(t1[1] - c[1], t1[0] - c[0])
        a2 = math.atan2(t2[1] - c[1], t2[0] - c[0])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        for k in range(n + 1):
            ang = a1 + da * k / n
            out.append((c[0] + r * math.cos(ang), c[1] + r * math.sin(ang)))
    return poly(out)


def rrect(cx, cy, w, h, r=0.0, ang=0.0):
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        u, v = sx * w / 2, sy * h / 2
        pts.append((cx + ca * u - sa * v, cy + sa * u + ca * v))
    if r <= 1e-6:
        return poly(pts)
    return rpoly(pts, r)


def head(tip, direction, length, halfw):
    dx, dy = direction
    n = math.hypot(dx, dy); dx, dy = dx / n, dy / n
    bx, by = tip[0] - dx * length, tip[1] - dy * length
    return poly([tip, (bx - dy * halfw, by + dx * halfw), (bx + dy * halfw, by - dx * halfw)])


# ----------------------------------------------------------------------------
# canvas
# ----------------------------------------------------------------------------
class Scene:
    def __init__(self):
        self.img = np.empty((RES, RES, 3), np.float32)
        self.img[:] = np.array(BG, np.float32)
        self.clip = None

    def paint(self, m, col):
        if self.clip is not None:
            m = m & self.clip
        self.img[m] = np.array(col, np.float32)

    def add(self, m, col):
        self.img[m] = np.array(col, np.float32)


def P(u, v):
    return (CEN + u * D0, CEN + v * D0)


def L(uv):
    return [P(u, v) for u, v in uv]


# ----------------------------------------------------------------------------
# digit font (stroke models, unit box 0..1 x 0..1)
# ----------------------------------------------------------------------------
def digit_mask(ch, x0, y0, w, h):
    t = 0.185 * h

    def q(u, v):
        return (x0 + u * w, y0 + v * h)

    def st(uvs, thick=None):
        return stroke([q(u, v) for u, v in uvs], t if thick is None else thick)

    if ch == '0':
        return ellring(x0 + w / 2, y0 + h / 2, w * 0.48, h * 0.49, t)
    if ch == '1':
        return st([(0.24, 0.22), (0.60, 0.04), (0.60, 0.97)])
    if ch == '2':
        return st([(0.08, 0.22), (0.20, 0.08), (0.48, 0.03), (0.78, 0.10),
                   (0.87, 0.28), (0.68, 0.50), (0.12, 0.92), (0.92, 0.92)])
    if ch == '3':
        return st([(0.08, 0.16), (0.30, 0.04), (0.62, 0.05), (0.85, 0.19),
                   (0.78, 0.36), (0.50, 0.46), (0.79, 0.56), (0.89, 0.74),
                   (0.70, 0.93), (0.36, 0.97), (0.09, 0.85)])
    if ch == '4':
        return st([(0.68, 0.97), (0.68, 0.04), (0.08, 0.66), (0.94, 0.66)])
    if ch == '5':
        return st([(0.86, 0.05), (0.20, 0.05), (0.16, 0.42), (0.46, 0.37),
                   (0.76, 0.46), (0.88, 0.68), (0.71, 0.92), (0.38, 0.97), (0.09, 0.85)])
    if ch == '6':
        return st([(0.82, 0.08), (0.52, 0.04), (0.26, 0.20), (0.13, 0.50),
                   (0.13, 0.74), (0.29, 0.94), (0.56, 0.97), (0.80, 0.85),
                   (0.85, 0.65), (0.68, 0.49), (0.41, 0.46), (0.17, 0.57)])
    if ch == '7':
        return st([(0.08, 0.05), (0.90, 0.05), (0.42, 0.97)])
    if ch == '8':
        return (ellring(x0 + w / 2, y0 + h * 0.26, w * 0.41, h * 0.245, t) |
                ellring(x0 + w / 2, y0 + h * 0.72, w * 0.47, h * 0.275, t))
    if ch == 'S':
        return st([(0.86, 0.13), (0.56, 0.03), (0.22, 0.11), (0.22, 0.32),
                   (0.52, 0.45), (0.80, 0.58), (0.78, 0.85), (0.44, 0.97), (0.12, 0.86)])
    if ch == 'T':
        return st([(0.05, 0.05), (0.95, 0.05)]) | st([(0.50, 0.05), (0.50, 0.97)])
    if ch == 'O':
        return ellring(x0 + w / 2, y0 + h / 2, w * 0.47, h * 0.48, t)
    if ch == 'P':
        return st([(0.16, 0.97), (0.16, 0.05), (0.68, 0.05), (0.88, 0.24),
                   (0.68, 0.46), (0.16, 0.46)])
    raise ValueError(ch)


def number_mask(s, cx, cy, h, wfrac, gap):
    """digits centred on (cx,cy); each digit is wfrac*h wide"""
    w = wfrac * h
    total = len(s) * w + (len(s) - 1) * gap
    x = cx - total / 2
    m = np.zeros((RES, RES), bool)
    for ch in s:
        m |= digit_mask(ch, x, cy - h / 2, w, h)
        x += w + gap
    return m


# ----------------------------------------------------------------------------
# shared sign parts
# ----------------------------------------------------------------------------
def tri_pts(up=True):
    if up:
        return L([(0.0, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    return L([(0.0, 0.5), (-0.5, -0.5), (0.5, -0.5)])


def tri_inner(up=True, s=0.665):
    Pts = np.asarray(tri_pts(up))
    a = np.linalg.norm(Pts[1] - Pts[2])
    b = np.linalg.norm(Pts[0] - Pts[2])
    c = np.linalg.norm(Pts[0] - Pts[1])
    ic = (a * Pts[0] + b * Pts[1] + c * Pts[2]) / (a + b + c)
    return [tuple(ic + s * (p - ic)) for p in Pts]


def diamond_pts(s=1.0):
    return L([(0.0, -0.5 * s), (0.5 * s, 0.0), (0.0, 0.5 * s), (-0.5 * s, 0.0)])


def octagon_pts():
    R = 0.5 * D0 / math.cos(math.pi / 8)
    return [(CEN + R * math.cos(math.pi / 8 + i * math.pi / 4),
             CEN + R * math.sin(math.pi / 8 + i * math.pi / 4)) for i in range(8)]


def car_mask(cu, cv, w=0.240, h=0.140):
    """rear view of a car: body + cabin"""
    body = rrect(*P(cu, cv + 0.028), w * D0, h * D0 * 0.60, r=0.018 * D0)
    cab = rrect(*P(cu, cv - 0.032), w * D0 * 0.68, h * D0 * 0.62, r=0.020 * D0)
    return body | cab


def truck_rear(cu, cv, w=0.225, h=0.215):
    box = rrect(*P(cu, cv - 0.012), w * D0, h * D0 * 0.80, r=0.012 * D0)
    base = rrect(*P(cu, cv + 0.075), w * D0 * 0.94, h * D0 * 0.26, r=0.010 * D0)
    return box | base


def truck_side(cu, cv, s=1.0):
    box = rrect(*P(cu - 0.075 * s, cv - 0.030 * s), 0.290 * s * D0, 0.175 * s * D0, r=0.010 * D0)
    cab = rrect(*P(cu + 0.135 * s, cv + 0.020 * s), 0.125 * s * D0, 0.105 * s * D0, r=0.016 * D0)
    wind = rrect(*P(cu + 0.150 * s, cv - 0.005 * s), 0.075 * s * D0, 0.045 * s * D0, r=0.008 * D0)
    wheels = disc(*P(cu - 0.145 * s, cv + 0.075 * s), 0.036 * s * D0) | \
             disc(*P(cu + 0.130 * s, cv + 0.075 * s), 0.036 * s * D0)
    return (box | cab | wheels), wind


def stripes_mask(r=0.455, n=5, sp=0.062, t=0.034):
    m = np.zeros((RES, RES), bool)
    d = math.sqrt(0.5)
    for i in range(n):
        off = (i - (n - 1) / 2.0) * sp
        cx, cy = CEN + off * d * D0, CEN + off * d * D0
        m |= seg((cx - 0.7 * D0 * d, cy + 0.7 * D0 * d),
                 (cx + 0.7 * D0 * d, cy - 0.7 * D0 * d), t * D0 / 2)
    return m & disc(CEN, CEN, r * D0)


def person(cu, cv, s=1.0, lean=0.0):
    hd = disc(*P(cu + lean * 0.02, cv - 0.115 * s), 0.046 * s * D0)
    body = stroke(L([(cu + lean * 0.02, cv - 0.065 * s), (cu, cv + 0.045 * s)]), 0.072 * s * D0)
    legs = stroke(L([(cu, cv + 0.045 * s), (cu - 0.075 * s, cv + 0.165 * s)]), 0.056 * s * D0) | \
           stroke(L([(cu, cv + 0.045 * s), (cu + 0.070 * s, cv + 0.160 * s)]), 0.056 * s * D0)
    arms = stroke(L([(cu - 0.075 * s, cv + 0.010 * s), (cu + 0.010 * s, cv - 0.035 * s),
                     (cu + 0.080 * s, cv + 0.005 * s)]), 0.048 * s * D0)
    return hd | body | legs | arms


# ----------------------------------------------------------------------------
# the 43 sign models.  Each returns (scene, inner_mask_getter)
# ----------------------------------------------------------------------------
def inner_disc(r=0.385):
    return disc(CEN, CEN, r * D0)


def inner_tri(up=True):
    return rpoly(tri_inner(up, 0.635), 0.02 * D0)


def base_ring(sc):
    sc.paint(disc(CEN, CEN, 0.5 * D0), RED)
    sc.paint(disc(CEN, CEN, 0.400 * D0), WHT)


def base_tri(sc, up=True):
    sc.paint(rpoly(tri_pts(up), 0.055 * D0), RED)
    sc.paint(rpoly(tri_inner(up, 0.665), 0.022 * D0), WHT)


def tri_base(sc):
    """warning triangle: white field, pictograms are clipped to it"""
    base_tri(sc, True)
    sc.clip = rpoly(tri_inner(True, 0.665), 0.022 * D0)
    return inner_tri(True)


def base_blue(sc):
    sc.paint(disc(CEN, CEN, 0.5 * D0), WHT)
    sc.paint(disc(CEN, CEN, 0.478 * D0), BLUE)


def base_end(sc):
    sc.paint(disc(CEN, CEN, 0.5 * D0), BLK)
    sc.paint(disc(CEN, CEN, 0.478 * D0), WHT)


def build_sign(k):
    """returns (rgb float image at RES, inner mask at RES)"""
    sc = Scene()
    inner = inner_disc()
    if k in (0, 1, 2, 3, 4, 5, 7, 8):
        base_ring(sc)
        txt = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}[k]
        if len(txt) == 2:
            m = number_mask(txt, CEN, CEN + 0.005 * D0, 0.500 * D0, 0.560, 0.055 * D0)
        else:
            m = number_mask(txt, CEN, CEN + 0.005 * D0, 0.455 * D0, 0.480, 0.035 * D0)
        sc.paint(m, BLK)
    elif k == 6:
        base_end(sc)
        m = number_mask('80', CEN, CEN, 0.470 * D0, 0.560, 0.050 * D0)
        sc.paint(m, GRY)
        sc.paint(stripes_mask(), GRY)
    elif k == 9:
        base_ring(sc)
        sc.paint(car_mask(-0.135, 0.010), RED)
        sc.paint(car_mask(+0.135, 0.010), BLK)
    elif k == 10:
        base_ring(sc)
        sc.paint(truck_rear(-0.140, 0.000), RED)
        sc.paint(car_mask(+0.140, 0.030), BLK)
    elif k == 11:
        inner = tri_base(sc)
        sc.paint(rrect(*P(0.0, 0.105), 0.100 * D0, 0.470 * D0, r=0.010 * D0), BLK)
        sc.paint(rrect(*P(0.0, 0.000), 0.370 * D0, 0.052 * D0, r=0.006 * D0), BLK)
    elif k == 12:
        sc.paint(rpoly(diamond_pts(1.0), 0.030 * D0), WHT)
        sc.paint(rpoly(diamond_pts(0.70), 0.020 * D0), YEL)
        inner = poly(diamond_pts(0.66))
    elif k == 13:
        base_tri(sc, False); inner = inner_tri(False)
    elif k == 14:
        sc.paint(rpoly(octagon_pts(), 0.030 * D0), WHT)
        R = 0.5 * D0 / math.cos(math.pi / 8)
        pts = [(CEN + (R - 0.022 * D0) * math.cos(math.pi / 8 + i * math.pi / 4),
                CEN + (R - 0.022 * D0) * math.sin(math.pi / 8 + i * math.pi / 4)) for i in range(8)]
        sc.paint(rpoly(pts, 0.024 * D0), RED)
        m = number_mask('STOP', CEN, CEN, 0.215 * D0, 0.640, 0.045 * D0)
        sc.paint(m, WHT)
        inner = disc(CEN, CEN, 0.385 * D0)
    elif k == 15:
        base_ring(sc)
    elif k == 16:
        base_ring(sc)
        tm, wind = truck_side(0.0, 0.0, 1.0)
        sc.paint(tm, BLK)
        sc.paint(wind, WHT)
    elif k == 17:
        sc.paint(disc(CEN, CEN, 0.5 * D0), WHT)
        sc.paint(disc(CEN, CEN, 0.487 * D0), RED)
        sc.paint(rrect(CEN, CEN, 0.720 * D0, 0.130 * D0, r=0.012 * D0), WHT)
        inner = disc(CEN, CEN, 0.44 * D0)
    elif k == 18:
        inner = tri_base(sc)
        sc.paint(rrect(*P(0.0, 0.075), 0.072 * D0, 0.300 * D0, r=0.014 * D0), BLK)
        sc.paint(disc(*P(0.0, 0.310), 0.046 * D0), BLK)
    elif k == 19:
        inner = tri_base(sc)
        sc.paint(stroke(L([(-0.155, -0.045), (0.055, -0.045), (0.095, 0.020), (0.095, 0.300)]),
                        0.082 * D0), BLK)
    elif k == 20:
        inner = tri_base(sc)
        sc.paint(stroke(L([(0.155, -0.045), (-0.055, -0.045), (-0.095, 0.020), (-0.095, 0.300)]),
                        0.082 * D0), BLK)
    elif k == 21:
        inner = tri_base(sc)
        sc.paint(stroke(L([(-0.075, -0.140), (-0.075, 0.010), (0.075, 0.075), (0.075, 0.300)]),
                        0.078 * D0), BLK)
    elif k == 22:
        inner = tri_base(sc)
        for cu in (-0.105, 0.105):
            cx, cy = P(cu, 0.300)
            m = disc(cx, cy, 0.088 * D0) & (GY <= cy)
            sc.paint(m, BLK)
        sc.paint(rrect(*P(0.0, 0.303), 0.400 * D0, 0.022 * D0), BLK)
    elif k == 23:
        inner = tri_base(sc)
        ang = -0.28
        sc.paint(rrect(*P(0.010, 0.030), 0.250 * D0, 0.095 * D0, r=0.022 * D0, ang=ang), BLK)
        sc.paint(rrect(*P(-0.010, -0.020), 0.130 * D0, 0.060 * D0, r=0.016 * D0, ang=ang), BLK)
        for cu in (-0.105, 0.055):
            sc.paint(stroke(L([(cu, 0.115), (cu - 0.045, 0.175), (cu + 0.020, 0.235),
                               (cu - 0.040, 0.300)]), 0.038 * D0), BLK)
    elif k == 24:
        inner = tri_base(sc)
        sc.paint(stroke(L([(-0.058, -0.075), (-0.058, 0.300)]), 0.050 * D0), BLK)
        sc.paint(stroke(L([(0.098, -0.075), (0.048, 0.055), (0.048, 0.300)]), 0.050 * D0), BLK)
    elif k == 25:
        inner = tri_base(sc)
        sc.paint(poly(L([(0.020, 0.305), (0.145, 0.150), (0.270, 0.305)])), BLK)
        sc.paint(disc(*P(-0.105, -0.075), 0.044 * D0), BLK)
        sc.paint(stroke(L([(-0.100, -0.025), (-0.055, 0.115)]), 0.058 * D0), BLK)
        sc.paint(stroke(L([(-0.055, 0.115), (-0.150, 0.290)]), 0.042 * D0), BLK)
        sc.paint(stroke(L([(-0.055, 0.115), (0.010, 0.280)]), 0.042 * D0), BLK)
        sc.paint(stroke(L([(-0.175, 0.020), (-0.020, 0.075), (0.095, 0.215)]), 0.032 * D0), BLK)
        sc.paint(rrect(*P(0.105, 0.235), 0.075 * D0, 0.055 * D0, r=0.010 * D0, ang=0.7), BLK)
    elif k == 26:
        inner = tri_base(sc)
        for cv, col in ((-0.045, RED), (0.075, YEL), (0.195, (0.10, 0.55, 0.25))):
            sc.paint(disc(*P(0.0, cv), 0.049 * D0), col)
    elif k == 27:
        inner = tri_base(sc)
        for cv in (0.205, 0.255, 0.305):
            sc.paint(rrect(*P(0.0, cv), 0.360 * D0, 0.028 * D0), BLK)
        sc.paint(person(0.0, 0.045, 1.05, 0.4), BLK)
    elif k == 28:
        inner = tri_base(sc)
        sc.paint(person(0.085, 0.045, 1.05, 0.6), BLK)
        sc.paint(person(-0.120, 0.115, 0.82, 0.6), BLK)
    elif k == 29:
        inner = tri_base(sc)
        for cu in (-0.115, 0.115):
            sc.paint(ringm(*P(cu, 0.205), 0.090 * D0, 0.036 * D0), BLK)
        sc.paint(stroke(L([(-0.115, 0.205), (-0.020, 0.055), (0.115, 0.205)]), 0.034 * D0), BLK)
        sc.paint(stroke(L([(-0.020, 0.055), (0.055, 0.055), (0.115, 0.205)]), 0.034 * D0), BLK)
        sc.paint(stroke(L([(0.055, 0.055), (0.140, 0.010)]), 0.034 * D0), BLK)
        sc.paint(stroke(L([(-0.070, 0.020), (0.000, 0.020)]), 0.034 * D0), BLK)
        sc.paint(stroke(L([(-0.020, 0.055), (-0.045, 0.020)]), 0.032 * D0), BLK)
    elif k == 30:
        inner = tri_base(sc)
        cx, cy = P(0.0, 0.105)
        for i in range(6):
            a = i * math.pi / 3 + math.pi / 6
            ex, ey = cx + 0.180 * D0 * math.cos(a), cy + 0.180 * D0 * math.sin(a)
            sc.paint(seg((cx, cy), (ex, ey), 0.030 * D0), BLK)
            mx, my = cx + 0.110 * D0 * math.cos(a), cy + 0.110 * D0 * math.sin(a)
            for da in (0.9, -0.9):
                sc.paint(seg((mx, my), (mx + 0.062 * D0 * math.cos(a + da),
                                        my + 0.062 * D0 * math.sin(a + da)), 0.022 * D0), BLK)
    elif k == 31:
        inner = tri_base(sc)
        sc.paint(seg(P(-0.045, 0.020), P(0.115, 0.150), 0.048 * D0), BLK)
        sc.paint(seg(P(-0.045, 0.020), P(-0.115, -0.095), 0.030 * D0), BLK)
        sc.paint(disc(*P(-0.130, -0.120), 0.034 * D0), BLK)
        sc.paint(stroke(L([(-0.135, -0.150), (-0.185, -0.245)]), 0.026 * D0), BLK)
        sc.paint(stroke(L([(-0.115, -0.150), (-0.100, -0.255)]), 0.026 * D0), BLK)
        sc.paint(stroke(L([(-0.030, 0.050), (-0.115, 0.150), (-0.150, 0.245)]), 0.036 * D0), BLK)
        sc.paint(stroke(L([(-0.010, 0.060), (-0.045, 0.185), (-0.020, 0.275)]), 0.036 * D0), BLK)
        sc.paint(stroke(L([(0.115, 0.150), (0.215, 0.215), (0.250, 0.290)]), 0.036 * D0), BLK)
        sc.paint(stroke(L([(0.105, 0.160), (0.140, 0.260), (0.105, 0.310)]), 0.036 * D0), BLK)
    elif k == 32:
        base_end(sc)
        sc.paint(stripes_mask(), GRY)
    elif k in (33, 34, 35, 36, 37, 38, 39, 40):
        base_blue(sc)
        inner = inner_disc(0.43)
        sgn = 1.0
        if k in (34, 37, 39):
            sgn = -1.0
        if k == 35:
            sc.paint(stroke(L([(0.0, 0.300), (0.0, -0.110)]), 0.105 * D0), WHT)
            sc.paint(head(P(0.0, -0.330), (0, -1), 0.215 * D0, 0.165 * D0), WHT)
        elif k in (33, 34):
            sc.paint(stroke(L([(sgn * -0.055, 0.300), (sgn * -0.055, -0.030),
                               (sgn * 0.010, -0.130), (sgn * 0.115, -0.155)]), 0.100 * D0), WHT)
            sc.paint(head(P(sgn * 0.330, -0.155), (sgn, 0), 0.200 * D0, 0.155 * D0), WHT)
        elif k in (36, 37):
            sc.paint(stroke(L([(sgn * -0.045, 0.300), (sgn * -0.045, -0.110)]), 0.098 * D0), WHT)
            sc.paint(head(P(sgn * -0.045, -0.330), (0, -1), 0.205 * D0, 0.150 * D0), WHT)
            sc.paint(stroke(L([(sgn * -0.045, 0.090), (sgn * 0.050, 0.030), (sgn * 0.125, 0.015)]),
                            0.098 * D0), WHT)
            sc.paint(head(P(sgn * 0.345, 0.015), (sgn, 0), 0.205 * D0, 0.155 * D0), WHT)
        elif k in (38, 39):
            d = (sgn * math.sqrt(0.5), math.sqrt(0.5))
            sc.paint(stroke(L([(sgn * -0.235, -0.245), (sgn * 0.075, 0.065)]), 0.110 * D0), WHT)
            sc.paint(head(P(sgn * 0.290, 0.280), d, 0.215 * D0, 0.165 * D0), WHT)
        elif k == 40:
            r = 0.315 * D0
            for i in range(3):
                a0 = math.radians(60 + 120 * i)
                a1 = a0 - math.radians(96)
                sc.paint(arc(CEN, CEN, r, a0, a1, 0.092 * D0, 14), WHT)
                tipa = a1 - math.radians(11)
                tip = (CEN + r * math.cos(tipa), CEN + r * math.sin(tipa))
                tang = (math.sin(a1), -math.cos(a1))
                sc.paint(head(tip, tang, 0.135 * D0, 0.120 * D0), WHT)
    elif k == 41:
        base_end(sc)
        sc.paint(car_mask(-0.135, 0.010), GRY)
        sc.paint(car_mask(+0.135, 0.010), GRY)
        sc.paint(stripes_mask(), GRY)
    elif k == 42:
        base_end(sc)
        sc.paint(truck_rear(-0.140, 0.000), GRY)
        sc.paint(car_mask(+0.140, 0.030), GRY)
        sc.paint(stripes_mask(), GRY)
    else:
        raise ValueError(k)
    return sc.img, inner


# ----------------------------------------------------------------------------
# family (outer shape / colour) models
# ----------------------------------------------------------------------------
FAM = {}
for c in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16):
    FAM[c] = 0
FAM[17] = 1
FAM[14] = 2
FAM[13] = 3
for c in (11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
    FAM[c] = 4
FAM[12] = 5
for c in (33, 34, 35, 36, 37, 38, 39, 40):
    FAM[c] = 6
for c in (6, 32, 41, 42):
    FAM[c] = 7
NFAM = 8


def build_family(f):
    sc = Scene()
    if f == 0:
        base_ring(sc)
    elif f == 1:
        sc.paint(disc(CEN, CEN, 0.5 * D0), WHT)
        sc.paint(disc(CEN, CEN, 0.487 * D0), RED)
        sc.paint(rrect(CEN, CEN, 0.720 * D0, 0.130 * D0, r=0.012 * D0), WHT)
    elif f == 2:
        sc.paint(rpoly(octagon_pts(), 0.030 * D0), WHT)
        R = 0.5 * D0 / math.cos(math.pi / 8)
        pts = [(CEN + (R - 0.022 * D0) * math.cos(math.pi / 8 + i * math.pi / 4),
                CEN + (R - 0.022 * D0) * math.sin(math.pi / 8 + i * math.pi / 4)) for i in range(8)]
        sc.paint(rpoly(pts, 0.024 * D0), RED)
        sc.paint(rrect(CEN, CEN, 0.640 * D0, 0.190 * D0, r=0.020 * D0), WHT)
    elif f == 3:
        base_tri(sc, False)
    elif f == 4:
        base_tri(sc, True)
    elif f == 5:
        sc.paint(rpoly(diamond_pts(1.0), 0.030 * D0), WHT)
        sc.paint(rpoly(diamond_pts(0.70), 0.020 * D0), YEL)
    elif f == 6:
        base_blue(sc)
    elif f == 7:
        base_end(sc)
    return sc.img


# ----------------------------------------------------------------------------
# feature filters (the network's first conv), also applied to the renders
# ----------------------------------------------------------------------------
KS = 11
CH = 7          # red, green, blue, lum, hp0, hp1, hp2
SIG = (0.85, 1.70, 3.20)


def gauss(sigma):
    r = KS // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return g / g.sum()


def feature_kernels():
    W = np.zeros((CH, 3, KS, KS), np.float64)
    r = KS // 2
    W[0, :, r, r] = [1.0, -0.5, -0.5]
    W[1, :, r, r] = [-0.5, 1.0, -0.5]
    W[2, :, r, r] = [-0.5, -0.5, 1.0]
    W[3, :, r, r] = [1 / 3, 1 / 3, 1 / 3]
    for j, s in enumerate(SIG):
        g = gauss(s)
        for c in range(3):
            W[4 + j, c] = -g / 3.0
            W[4 + j, c, r, r] += 1 / 3.0
    return W


FW = feature_kernels()


def features(rgb32):
    """rgb32: (32,32,3) float in 0..1 -> (CH,32,32), zero padded like the net"""
    x = np.pad(rgb32.transpose(2, 0, 1), ((0, 0), (KS // 2, KS // 2), (KS // 2, KS // 2)))
    out = np.zeros((CH, N, N), np.float64)
    for dy in range(KS):
        for dx in range(KS):
            patch = x[:, dy:dy + N, dx:dx + N]
            out += np.einsum('oc,chw->ohw', FW[:, :, dy, dx], patch)
    return out


def blur32(img, sigma):
    r = max(1, int(round(3 * sigma)))
    t = np.arange(-r, r + 1)
    k = np.exp(-t * t / (2 * sigma * sigma)); k /= k.sum()
    p = np.pad(img, ((r, r), (0, 0), (0, 0)), mode='edge')
    tmp = np.zeros_like(img)
    for i, w in enumerate(k):
        tmp += w * p[i:i + N]
    p = np.pad(tmp, ((0, 0), (r, r), (0, 0)), mode='edge')
    out = np.zeros_like(img)
    for i, w in enumerate(k):
        out += w * p[:, i:i + N]
    return out


# ----------------------------------------------------------------------------
# warp a canonical render into a variant and downsample to 32x32
# ----------------------------------------------------------------------------
SUB = 4
_off = (np.arange(SUB) + 0.5) / SUB - 0.5
_oy, _ox = np.meshgrid(_off, _off, indexing='ij')
SY, SX = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
QX = (SX[:, :, None, None] + _ox[None, None, :, :]).reshape(-1)
QY = (SY[:, :, None, None] + _oy[None, None, :, :]).reshape(-1)


def warp(canon, A):
    """A maps canonical local coords -> variant local coords."""
    Ai = np.linalg.inv(A)
    px = Ai[0, 0] * (QX - CEN) + Ai[0, 1] * (QY - CEN) + CEN
    py = Ai[1, 0] * (QX - CEN) + Ai[1, 1] * (QY - CEN) + CEN
    fx = (px + 0.5) * SS - 0.5
    fy = (py + 0.5) * SS - 0.5
    x0 = np.floor(fx).astype(np.int32); y0 = np.floor(fy).astype(np.int32)
    tx = fx - x0; ty = fy - y0
    x0c = np.clip(x0, 0, RES - 2); x1c = np.clip(x0 + 1, 0, RES - 1)
    y0c = np.clip(y0, 0, RES - 2); y1c = np.clip(y0 + 1, 0, RES - 1)
    tx = np.clip(tx, 0, 1); ty = np.clip(ty, 0, 1)
    flat = canon.reshape(-1, 3)
    i00 = y0c * RES + x0c; i01 = y0c * RES + x1c
    i10 = y1c * RES + x0c; i11 = y1c * RES + x1c
    a = flat[i00] * (1 - tx)[:, None] + flat[i01] * tx[:, None]
    b = flat[i10] * (1 - tx)[:, None] + flat[i11] * tx[:, None]
    v = (a * (1 - ty)[:, None] + b * ty[:, None])
    return v.reshape(N, N, SUB * SUB, 3).mean(axis=2)


def variant_A(scale, rot_deg, aspect):
    th = math.radians(rot_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return R @ np.diag([scale * aspect, scale])


# ----------------------------------------------------------------------------
# content front end: rectified, blurred "ink density" maps.
# relu(-hp) is a darkness detector, relu(+hp) a brightness detector; blurring
# turns them into smooth density maps that tolerate small misalignment.  A
# uniform (blank) sign interior gives ~0 in both, which is what makes the
# "no vehicles" / blank-interior rule work.
# ----------------------------------------------------------------------------
CCH = 6                    # red, green, blue, dark-fine, dark-coarse, bright-coarse
INK_SIG = (0.80, 1.40, 1.40)
INK_SRC = (4, 5, 5)        # which hp channel feeds each ink map
INK_SGN = (-1.0, -1.0, +1.0)
BK = 9


def norm_np(x):
    mu = x.mean(axis=(1, 2), keepdims=True)
    v = ((x - mu) ** 2).mean(axis=(1, 2), keepdims=True)
    return (x - mu) / np.sqrt(v + 1e-5)


def blur_zero(x, sigma, K=BK):
    """separable gaussian, zero padding -- exactly what the net's conv does"""
    r = K // 2
    t = np.arange(-r, r + 1)
    k = np.exp(-t * t / (2 * sigma * sigma)); k /= k.sum()
    p = np.pad(x, ((r, r), (0, 0)), mode='constant')
    a = np.zeros_like(x)
    for i, w in enumerate(k):
        a += w * p[i:i + x.shape[0]]
    p = np.pad(a, ((0, 0), (r, r)), mode='constant')
    b = np.zeros_like(x)
    for i, w in enumerate(k):
        b += w * p[:, i:i + x.shape[1]]
    return b


def content_maps(fn):
    """fn: normalised (CH,32,32) -> normalised (CCH,32,32)"""
    out = np.zeros((CCH, N, N))
    out[0], out[1], out[2] = fn[0], fn[1], fn[2]
    for j in range(3):
        r = np.maximum(INK_SGN[j] * fn[INK_SRC[j]], 0.0)
        out[3 + j] = blur_zero(r, INK_SIG[j])
    return norm_np(out)


def content_kernels():
    """the three convs that build the content front end inside the network"""
    WA = np.zeros((CCH, CH, 1, 1))
    for c in range(3):
        WA[c, c, 0, 0] = 1.0
    WB = np.zeros((CCH, CH, 1, 1))
    for j in range(3):
        WB[3 + j, INK_SRC[j], 0, 0] = INK_SGN[j]
    WC = np.zeros((CCH, CCH, BK, BK))
    r = BK // 2
    for j in range(3):
        t = np.arange(-r, r + 1)
        k = np.exp(-t * t / (2 * INK_SIG[j] ** 2)); k /= k.sum()
        WC[3 + j, 3 + j] = np.outer(k, k)
    return WA, WB, WC


# ----------------------------------------------------------------------------
# template construction: each render normalised by itself alone
# ----------------------------------------------------------------------------
G_SHAPE = np.array([0.26, 0.09, 0.20, 0.08, 0.04, 0.12, 0.21])
G_CONT = np.array([0.18, 0.06, 0.16, 0.20, 0.20, 0.20])
G_CONT26 = np.array([0.18, 0.20, 0.12, 0.17, 0.17, 0.16])


def make_template(feat, win, mask, gains):
    """feat: (C,32,32) already normalised like the net; mask: (32,32) in 0..1"""
    o = (N - win) // 2
    f = feat[:, o:o + win, o:o + win]
    m = mask[o:o + win, o:o + win]
    tot = m.sum()
    if tot < 1e-6:
        return np.zeros((f.shape[0], win, win))
    t = np.zeros_like(f)
    gs = []
    for c in range(f.shape[0]):
        if gains[c] <= 0:
            gs.append(0.0); continue
        v = f[c] * m
        mu = v.sum() / tot
        v = (f[c] - mu) * m
        n = np.sqrt((v * v).sum())
        if n < 1e-7:
            gs.append(0.0); continue
        t[c] = v / n
        gs.append(gains[c])
    s = sum(gs)
    if s <= 0:
        return np.zeros_like(t)
    for c in range(f.shape[0]):
        t[c] *= gs[c] / s
    return t / math.sqrt(tot)


def mask32(mres):
    return mres.reshape(N, SS, N, SS).mean(axis=(1, 3))


# ----------------------------------------------------------------------------
# assemble the network
# ----------------------------------------------------------------------------
SCALES = (0.860, 0.930, 1.000)
ROTS = (-7.0, 0.0, 7.0)
ASPECTS = (1.00, 0.88)
SHAPE_K = 30
CONT_K = 28
EMPTY_TH = 0.30
A_SHAPE = 2.0
B_CONT = 1.0
SOLO_OFF = 0.60
SOLO_FAMS = (3, 5)

CONTENT_ORDER = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 16, 15,
                 6, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
F0_N = 12


def main(outdir):
    layers = []
    weights = {}

    def conv(name, inp, W, b, k, s, p):
        layers.append({"name": name, "op": "conv", "inputs": [inp], "out": int(W.shape[0]),
                       "kernel": [k, k], "stride": s, "padding": p})
        weights[name + ".W"] = W.astype(np.float32)
        weights[name + ".b"] = b.astype(np.float32)

    def dense(name, inp, W, b):
        layers.append({"name": name, "op": "dense", "inputs": [inp], "out": int(W.shape[0])})
        weights[name + ".W"] = W.astype(np.float32)
        weights[name + ".b"] = b.astype(np.float32)

    def simple(name, op, inp, **kw):
        d = {"name": name, "op": op, "inputs": [inp]}
        d.update(kw)
        layers.append(d)

    # ---- front end -----------------------------------------------------
    conv("feat", "image", FW, np.zeros(CH), KS, 1, KS // 2)
    simple("fn", "norm", "feat")
    WA, WB, WC = content_kernels()
    conv("cA", "fn", WA, np.zeros(CCH), 1, 1, 0)
    conv("cBp", "fn", WB, np.zeros(CCH), 1, 1, 0)
    simple("cBr", "relu", "cBp")
    conv("cB", "cBr", WC, np.zeros(CCH), BK, 1, BK // 2)
    simple("cin", "add", "cA")
    layers[-1]["inputs"] = ["cA", "cB"]
    simple("cinn", "norm", "cin")

    # ---- shape bank ----------------------------------------------------
    shape_variants = [(s, r, a) for s in SCALES for r in ROTS for a in ASPECTS]
    nsv = len(shape_variants)
    fam_rgb = [build_family(f) for f in range(NFAM)]
    SW = np.zeros((NFAM * nsv, CH, SHAPE_K, SHAPE_K))
    full = np.ones((RES, RES), bool)
    for f in range(NFAM):
        for vi, (s, r, a) in enumerate(shape_variants):
            img = blur32(warp(fam_rgb[f], variant_A(s, r, a)), 0.70)
            ft = norm_np(features(img))
            t = make_template(ft, SHAPE_K, mask32(full), G_SHAPE)
            SW[f * nsv + vi] = t
    conv("sconv", "fn", SW, np.zeros(NFAM * nsv), SHAPE_K, 2, 1)
    simple("srelu", "relu", "sconv")
    simple("spool", "maxpool", "srelu", size=[3, 3], stride=[3, 3])
    simple("sresh", "reshape", "spool", shape=[NFAM, nsv, 1])
    simple("smax", "maxpool", "sresh", size=[nsv, 1], stride=[nsv, 1])
    simple("sflat", "reshape", "smax", shape=[NFAM])

    # ---- content bank --------------------------------------------------
    cont_variants = [(s, r, a) for s in SCALES for r in ROTS for a in ASPECTS]
    ncv = len(cont_variants)
    CW = np.zeros((43 * ncv, CCH, CONT_K, CONT_K))
    for ci, k in enumerate(CONTENT_ORDER):
        rgb, inner = build_sign(k)
        m32 = mask32(inner)
        gains = G_CONT26 if k == 26 else G_CONT
        if k == 15:
            continue                      # "empty" is handled by the max rule
        for vi, (s, r, a) in enumerate(cont_variants):
            img = blur32(warp(rgb, variant_A(s, r, a)), 0.70)
            ft = content_maps(norm_np(features(img)))
            t = make_template(ft, CONT_K, m32, gains)
            CW[ci * ncv + vi] = t
    conv("cconv", "cinn", CW, np.zeros(43 * ncv), CONT_K, 2, 0)
    simple("crelu", "relu", "cconv")
    simple("cpool", "maxpool", "crelu", size=[3, 3], stride=[3, 3])
    simple("cresh", "reshape", "cpool", shape=[43, ncv, 1])
    simple("cmax", "maxpool", "cresh", size=[ncv, 1], stride=[ncv, 1])
    simple("cflat", "reshape", "cmax", shape=[43])

    # ---- empty-interior rule for class 15 ------------------------------
    simple("cf2", "reshape", "cmax", shape=[1, 43, 1])
    simple("f0max", "maxpool", "cf2", size=[F0_N, 1], stride=[F0_N, 1])
    simple("f0flat", "reshape", "f0max", shape=[3])
    dense("f0sel", "f0flat", np.array([[1.0, 0.0, 0.0]]), np.zeros(1))
    We = np.zeros((43, 1)); be = np.zeros(43)
    We[15, 0] = -B_CONT
    be[15] = 2 * EMPTY_TH * B_CONT
    dense("empty", "f0sel", We, be)

    # ---- combine -------------------------------------------------------
    Ws = np.zeros((43, NFAM))
    for c in range(43):
        Ws[c, FAM[c]] = A_SHAPE
    dense("sexp", "sflat", Ws, np.zeros(43))

    Wc = np.zeros((43, 43)); bc = np.zeros(43)
    for c in range(43):
        f = FAM[c]
        if f in SOLO_FAMS:
            bc[c] = B_CONT * SOLO_OFF
        elif c != 15:
            Wc[c, CONTENT_ORDER.index(c)] = B_CONT
    dense("cexp", "cflat", Wc, bc)

    layers.append({"name": "score", "op": "add", "inputs": ["sexp", "cexp", "empty"]})

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "network.json"), "w") as fh:
        json.dump({"layers": layers}, fh)
    np.savez(os.path.join(outdir, "weights.npz"), **weights)
    print("layers", len(layers), "weights", sum(v.size for v in weights.values()))


if __name__ == "__main__":
    main(sys.argv[1])
