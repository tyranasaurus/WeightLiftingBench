# ---------------------------------------------------------------------------------------------
# Part 1: drawing the 43 signs from general knowledge of the German road sign catalogue.
# Coordinates: the sign's bounding box is [-1, 1] x [-1, 1], x to the right, y DOWN.
# Each sign is drawn once on a fine "master" grid as paint (r, g, b) plus coverage alpha.
# ---------------------------------------------------------------------------------------------
import math
import numpy as np

M = 300            # master grid size
EXT = 1.2          # master grid covers [-EXT, EXT]^2
_xs = (np.arange(M) + 0.5) / M * 2 * EXT - EXT
X, Y = np.meshgrid(_xs, _xs)

# paint colours as they typically look in a photograph (general knowledge, not measured)
WHITE = (0.92, 0.92, 0.92)
RED = (0.72, 0.13, 0.13)
BLACK = (0.08, 0.08, 0.08)
BLUE = (0.10, 0.30, 0.66)
YELLOW = (0.92, 0.72, 0.10)
DGREY = (0.30, 0.30, 0.30)
GREEN = (0.10, 0.60, 0.35)
AMBER = (0.95, 0.70, 0.10)


def _box(x0, x1, y0, y1):
    """index slices of the master grid covering the coordinate box (clipped)"""
    f = M / (2 * EXT)
    i0 = max(0, int(math.floor((x0 + EXT) * f)) - 1); i1 = min(M, int(math.ceil((x1 + EXT) * f)) + 1)
    j0 = max(0, int(math.floor((y0 + EXT) * f)) - 1); j1 = min(M, int(math.ceil((y1 + EXT) * f)) + 1)
    return slice(j0, j1), slice(i0, i1)


def disc(cx, cy, r, ry=None):
    ry = r if ry is None else ry
    return ((X - cx) / r) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0


def rect(x0, x1, y0, y1):
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


def poly(pts):
    pts = np.asarray(pts, float)
    n = len(pts)
    out = np.zeros(X.shape, bool)
    sl = _box(pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
    xx, yy = X[sl], Y[sl]
    ins = np.zeros(xx.shape, bool)
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        ins ^= ((y1 > yy) != (y2 > yy)) & (xx < (x2 - x1) * (yy - y1) / (y2 - y1) + x1)
    out[sl] = ins
    return out


def stroke(pts, w, closed=False, butt=(False, False)):
    """thick polyline of width w with round joins; butt=(start, end) gives flat ends"""
    pts = [tuple(map(float, p)) for p in pts]
    if closed:
        pts = pts + [pts[0]]
    out = np.zeros(X.shape, bool)
    h = w / 2.0
    ns = len(pts) - 1
    for i in range(ns):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 < 1e-14:
            continue
        sl = _box(min(x1, x2) - h, max(x1, x2) + h, min(y1, y2) - h, max(y1, y2) + h)
        xx, yy = X[sl] - x1, Y[sl] - y1
        t = (xx * dx + yy * dy) / L2
        tc = np.clip(t, 0.0, 1.0)
        m = (xx - tc * dx) ** 2 + (yy - tc * dy) ** 2 <= h * h
        if i == 0 and butt[0] and not closed:
            m &= t >= 0.0
        if i == ns - 1 and butt[1] and not closed:
            m &= t <= 1.0
        out[sl] |= m
    return out


def earc(cx, cy, rx, ry, a0, a1, n=24):
    """points on an ellipse arc; angle in degrees, 0 = right, 90 = DOWN (y is down)"""
    a = np.radians(np.linspace(a0, a1, n))
    return list(zip(cx + rx * np.cos(a), cy + ry * np.sin(a)))


def rounded_poly(pts, rho):
    return poly(pts) | stroke(pts, 2 * rho, closed=True)


# ------------------------------------------------------------------ digits and letters
def glyph(ch, w, t):
    """list of (points, butt) strokes for a DIN-like character in the box [0, w] x [0, 1], stroke width t"""
    a = t / 2.0
    r = (w - t) / 2.0
    cx = w / 2.0
    S = []
    if ch == '0' or ch == 'O':
        p = earc(cx, a + r, r, r, 180, 360, 16) + earc(cx, 1 - a - r, r, r, 0, 180, 16)
        S.append((p + [p[0]], (False, False)))
    elif ch == '1':
        S.append(([(w - a, 0.0), (w - a, 1.0)], (True, True)))
        S.append(([(w - a, a), (a * 0.8, 0.30)], (False, True)))
    elif ch == '2':
        ry = 0.27
        p = earc(cx, a + ry, r, ry, 190, 395, 26)
        p += [(a, 1 - a), (w - a * 0.6, 1 - a)]
        S.append((p, (True, True)))
    elif ch == '3':
        ym = 0.47
        ryu = (ym - a) / 2.0; ryl = (1 - a - ym) / 2.0
        S.append((earc(cx - 0.01, a + ryu, r * 0.92, ryu, 205, 450, 26), (True, False)))
        S.append((earc(cx, ym + ryl, r, ryl, -90, 160, 26), (False, True)))
    elif ch == '5':
        yb = 0.40
        ry = (1 - a - yb) / 2.0
        p = [(w - a * 1.3, a), (a + 0.03, a), (a + 0.01, yb + 0.07)]
        S.append((p, (True, False)))
        S.append((earc(cx, yb + ry, r, ry, 215, 520, 30), (False, True)))
    elif ch == '6':
        ry = 0.29
        cyl = 1 - a - ry
        p = earc(cx, cyl, r, ry, 0, 360, 32)
        S.append((p, (False, False)))
        ryu = 0.46
        S.append((earc(cx, a + ryu, r, ryu, 303, 180, 20) + [(a, cyl)], (True, False)))
    elif ch == '7':
        S.append(([(a * 0.6, a), (w - a, a), (0.36 * w, 1.0)], (True, True)))
    elif ch == '8':
        ym = 0.47
        ryu = (ym - a) / 2.0; ryl = (1 - a - ym) / 2.0
        S.append((earc(cx, a + ryu, r * 0.90, ryu, 0, 360, 28), (False, False)))
        S.append((earc(cx, ym + ryl, r, ryl, 0, 360, 28), (False, False)))
    elif ch == 'S':
        ym = 0.5
        ryu = (ym - a) / 2.0; ryl = (1 - a - ym) / 2.0
        S.append((earc(cx, a + ryu, r, ryu, 330, 90, 22), (True, False)))
        S.append((earc(cx, ym + ryl, r, ryl, -90, 150, 22), (False, True)))
    elif ch == 'T':
        S.append(([(0.0, a), (w, a)], (True, True)))
        S.append(([(cx, a), (cx, 1.0)], (False, True)))
    elif ch == 'P':
        ry = 0.26
        S.append(([(a, 0.0), (a, 1.0)], (True, True)))
        S.append(([(a, a)] + earc(w - a - ry * 0.9, a + ry, ry * 0.9, ry, -90, 90, 14) + [(a, a + 2 * ry)], (False, False)))
    return S


def text(chars, widths, gap, height, t, cx=0.0, cy=0.0):
    """mask of a row of glyphs centred at (cx, cy); widths/gap/t are in units of the glyph height"""
    total = sum(widths) + gap * (len(chars) - 1)
    x0 = cx - total * height / 2.0
    y0 = cy - height / 2.0
    out = np.zeros(X.shape, bool)
    for ch, w in zip(chars, widths):
        for pts, butt in glyph(ch, w, t):
            q = [(x0 + px * height, y0 + py * height) for px, py in pts]
            out |= stroke(q, t * height, butt=butt)
        x0 += (w + gap) * height
    return out


# ------------------------------------------------------------------ canvas
class Canvas:
    def __init__(self):
        self.col = np.zeros((M, M, 3))
        self.alp = np.zeros((M, M))

    def paint(self, mask, colour):
        self.col[mask] = colour
        self.alp[mask] = 1.0

    def mirror(self):
        self.col = self.col[:, ::-1].copy()
        self.alp = self.alp[:, ::-1].copy()
        return self


# ------------------------------------------------------------------ sign bodies
RING_IN = 0.70     # inner radius of the red ring of a prohibition sign
RING_OUT = 0.97    # a thin white rim lies outside it


def prohibition():
    c = Canvas()
    c.paint(disc(0, 0, 1.0), WHITE)
    c.paint(disc(0, 0, RING_OUT) & ~disc(0, 0, RING_IN), RED)
    return c


TRI_RHO = 0.07     # corner rounding of triangles


def _tri(scale, up=True):
    """triangle filling the bounding box, scaled about its centroid; up: apex at the top"""
    cy = 1.0 / 3.0
    pts = [(0.0, cy - scale * 4.0 / 3.0), (scale, cy + scale * 2.0 / 3.0), (-scale, cy + scale * 2.0 / 3.0)]
    if not up:
        pts = [(x, -y) for x, y in pts]
    return pts


def _tri_mask(scale, up=True):
    k = 1.0 - TRI_RHO * 1.6        # skeleton so that the rounded shape still fills the box
    cy = (1.0 / 3.0) * (1 if up else -1)
    pts = [(x * k, cy + (y - cy) * k) for x, y in _tri(scale, up)]
    return rounded_poly(pts, TRI_RHO * scale ** 0.5)


def danger():
    c = Canvas()
    c.paint(_tri_mask(1.0), WHITE)
    c.paint(_tri_mask(0.955), RED)
    c.paint(_tri_mask(0.64), WHITE)
    return c


def blue_disc():
    c = Canvas()
    c.paint(disc(0, 0, 1.0), WHITE)
    c.paint(disc(0, 0, 0.955), BLUE)
    return c


def end_disc():
    c = Canvas()
    c.paint(disc(0, 0, 1.0), WHITE)
    c.paint(disc(0, 0, 0.985) & ~disc(0, 0, 0.945), DGREY)
    return c


def end_band(c):
    """five thin black lines running from lower left to upper right"""
    d = (X + Y) / math.sqrt(2.0)      # distance across the band
    m = np.zeros(X.shape, bool)
    for k in range(-2, 3):
        m |= np.abs(d - k * 0.085) <= 0.025
    c.paint(m & disc(0, 0, 0.93), BLACK)


def arrow_head(tip, direction, length, halfwidth):
    """triangle with its tip at `tip`, pointing along `direction`"""
    dx, dy = direction
    n = math.hypot(dx, dy); dx, dy = dx / n, dy / n
    bx, by = tip[0] - dx * length, tip[1] - dy * length
    return poly([tip, (bx - dy * halfwidth, by + dx * halfwidth), (bx + dy * halfwidth, by - dx * halfwidth)])


def car_front(cx, cy, s=1.0):
    """small car seen from the front: body bar, cabin outline with a light windscreen, two wheels"""
    w, m = 0.26 * s, np.zeros(X.shape, bool)
    m |= rect(cx - w, cx + w, cy - 0.01 * s, cy + 0.15 * s)                      # body
    m |= stroke([(cx - w * 0.92, cy), (cx - w * 0.62, cy - 0.2 * s), (cx + w * 0.62, cy - 0.2 * s), (cx + w * 0.92, cy)], 0.04 * s)
    m |= rect(cx - w * 0.92, cx - w * 0.5, cy + 0.14 * s, cy + 0.22 * s)         # wheels
    m |= rect(cx + w * 0.5, cx + w * 0.92, cy + 0.14 * s, cy + 0.22 * s)
    return m


def truck_front(cx, cy, s=1.0):
    """lorry seen from the front: a tall box on wheels"""
    w, m = 0.24 * s, np.zeros(X.shape, bool)
    m |= rect(cx - w, cx + w, cy - 0.36 * s, cy + 0.14 * s)
    m |= rect(cx - w * 0.9, cx - w * 0.45, cy + 0.14 * s, cy + 0.22 * s)
    m |= rect(cx + w * 0.45, cx + w * 0.9, cy + 0.14 * s, cy + 0.22 * s)
    return m


def truck_side(cx, cy, s=1.0):
    """lorry seen from the side, driving to the left: cab on the left, box body on the right"""
    m = np.zeros(X.shape, bool)
    m |= rect(cx - 0.22 * s, cx + 0.60 * s, cy - 0.30 * s, cy + 0.12 * s)                  # box
    m |= poly([(cx - 0.58 * s, cy + 0.12 * s), (cx - 0.58 * s, cy - 0.02 * s), (cx - 0.48 * s, cy - 0.16 * s),
               (cx - 0.27 * s, cy - 0.16 * s), (cx - 0.27 * s, cy + 0.12 * s)])            # cab
    m |= rect(cx - 0.58 * s, cx + 0.60 * s, cy + 0.10 * s, cy + 0.18 * s)                  # chassis
    m |= disc(cx - 0.40 * s, cy + 0.22 * s, 0.10 * s) | disc(cx + 0.36 * s, cy + 0.22 * s, 0.10 * s)
    return m


def person(cx, top, h, lean=0.0, stride=0.42, arm=0.24):
    """walking figure: head, trunk, two legs apart, two arms; h = total height"""
    m = disc(cx + lean * 0.9 * h, top + 0.085 * h, 0.085 * h)
    sh = (cx + lean * 0.7 * h, top + 0.22 * h)        # shoulders
    hip = (cx, top + 0.55 * h)
    m |= stroke([sh, hip], 0.17 * h)
    m |= stroke([hip, (cx - stride * 0.5 * h, top + h)], 0.105 * h)
    m |= stroke([hip, (cx + stride * 0.5 * h, top + h)], 0.105 * h)
    m |= stroke([sh, (cx - arm * h + lean * 0.3 * h, top + 0.50 * h)], 0.075 * h)
    m |= stroke([sh, (cx + arm * h + lean * 0.3 * h, top + 0.46 * h)], 0.075 * h)
    return m


# ------------------------------------------------------------------ the 43 classes
D2 = dict(widths=[0.565, 0.565], gap=0.13, height=0.84, t=0.125)           # two-digit speed limits
D3 = dict(widths=[0.25, 0.44, 0.44], gap=0.10, height=0.80, t=0.115)     # three-digit speed limits


def speed(digits):
    c = prohibition()
    p = D2 if len(digits) == 2 else D3
    c.paint(text(digits, **p), BLACK)
    return c


def sign(k):
    if k == 0: return speed('20')
    if k == 1: return speed('30')
    if k == 2: return speed('50')
    if k == 3: return speed('60')
    if k == 4: return speed('70')
    if k == 5: return speed('80')
    if k == 7: return speed('100')
    if k == 8: return speed('120')
    if k == 6:                                   # end of 80
        c = end_disc()
        c.paint(text('80', **D2), DGREY)
        end_band(c)
        return c
    if k == 9:                                   # no passing: red car left, black car right
        c = prohibition()
        c.paint(car_front(-0.31, 0.0), RED)
        c.paint(car_front(0.31, 0.0), BLACK)
        return c
    if k == 10:                                  # no passing for lorries: red lorry left, black car right
        c = prohibition()
        c.paint(truck_front(-0.31, 0.06), RED)
        c.paint(car_front(0.31, 0.06), BLACK)
        return c
    if k == 11:                                  # priority at next junction: fat upright arrow, thin cross bar
        c = danger()
        m = rect(-0.12, 0.12, 0.00, 0.70) | poly([(0.0, -0.26), (0.27, 0.06), (-0.27, 0.06)])
        m |= rect(-0.36, 0.36, 0.24, 0.40)
        c.paint(m, BLACK)
        return c
    if k == 12:                                  # priority road: yellow diamond, white frame, thin dark edge
        c = Canvas()
        dm = lambda s: rounded_poly([(0, -s * 0.93), (s * 0.93, 0), (0, s * 0.93), (-s * 0.93, 0)], 0.07 * s)
        c.paint(dm(1.0), DGREY)
        c.paint(dm(0.955), WHITE)
        c.paint(dm(0.66), YELLOW)
        return c
    if k == 13:                                  # yield: point-down triangle, broad red border
        c = Canvas()
        c.paint(_tri_mask(1.0, up=False), WHITE)
        c.paint(_tri_mask(0.955, up=False), RED)
        c.paint(_tri_mask(0.58, up=False), WHITE)
        return c
    if k == 14:                                  # stop: red octagon, white edge line, white letters
        c = Canvas()
        q = math.tan(math.radians(22.5))
        octa = lambda s: poly([(s * x, s * y) for x, y in [(q, -1), (1, -q), (1, q), (q, 1), (-q, 1), (-1, q), (-1, -q), (-q, -1)]])
        c.paint(octa(1.0), WHITE)
        c.paint(octa(0.94), RED)
        c.paint(text('STOP', widths=[0.50, 0.50, 0.52, 0.50], gap=0.11, height=0.66, t=0.15), WHITE)
        return c
    if k == 15:                                  # no vehicles: empty red ring
        return prohibition()
    if k == 16:                                  # no lorries
        c = prohibition()
        c.paint(truck_side(0.0, -0.02), BLACK)
        return c
    if k == 17:                                  # no entry: red disc, white bar
        c = Canvas()
        c.paint(disc(0, 0, 1.0), WHITE)
        c.paint(disc(0, 0, 0.96), RED)
        c.paint(rect(-0.72, 0.72, -0.165, 0.165), WHITE)
        return c
    if k == 18:                                  # general danger: exclamation mark
        c = danger()
        c.paint(poly([(-0.095, -0.24), (0.095, -0.24), (0.06, 0.40), (-0.06, 0.40)]) | disc(0, 0.585, 0.09), BLACK)
        return c
    if k in (19, 20):                            # bend to the left (19); 20 is its mirror image
        c = danger()
        p = [(0.20, 0.70), (0.20, 0.36)] + earc(-0.08, 0.36, 0.28, 0.28, 0, -62, 10)
        e = p[-1]; a = math.radians(-62 - 90)
        d = (math.cos(a), math.sin(a))
        m = stroke(p, 0.15, butt=(True, False)) | arrow_head((e[0] + d[0] * 0.22, e[1] + d[1] * 0.22), d, 0.27, 0.17)
        c.paint(m, BLACK)
        return c.mirror() if k == 20 else c
    if k == 21:                                  # double bend, first to the left
        c = danger()
        p = [(0.13, 0.70), (0.13, 0.38), (-0.13, 0.20), (-0.13, -0.10)]
        m = stroke(p, 0.14, butt=(True, True))
        c.paint(m, BLACK)
        return c
    if k == 22:                                  # bumpy road: two humps on a base line
        c = danger()
        top = [(-0.50, 0.50)] + [(x, 0.50 - 0.15 * max(0.0, math.cos((abs(x) - 0.20) / 0.17 * math.pi / 2)) ** 1.0)
                                 for x in np.linspace(-0.40, 0.40, 41)] + [(0.50, 0.50)]
        c.paint(poly(top + [(0.50, 0.63), (-0.50, 0.63)]), BLACK)
        return c
    if k == 23:                                  # slippery road: car above two crossing skid marks
        c = danger()
        m = rect(-0.25, 0.25, 0.08, 0.26) | poly([(-0.19, 0.08), (-0.13, -0.08), (0.13, -0.08), (0.19, 0.08)])
        m |= rect(-0.25, -0.13, 0.26, 0.36) | rect(0.13, 0.25, 0.26, 0.36)
        ys = np.linspace(0.40, 0.68, 13)
        for x0 in (-0.19, 0.19):
            m |= stroke([(x0 + 0.08 * math.sin((y - 0.40) / 0.28 * 2 * math.pi), y) for y in ys], 0.05)
        c.paint(m, BLACK)
        return c
    if k == 24:                                  # road narrows on the right
        c = danger()
        m = stroke([(-0.12, -0.14), (-0.12, 0.70)], 0.13, butt=(True, True))
        m |= stroke([(0.09, -0.14), (0.09, 0.16), (0.27, 0.38), (0.27, 0.70)], 0.13, butt=(True, True))
        c.paint(m, BLACK)
        return c
    if k == 25:                                  # road works: digging man on the left, heap on the right
        c = danger()
        m = disc(-0.03, -0.10, 0.075)
        m |= stroke([(-0.06, 0.00), (-0.19, 0.33)], 0.14)                       # leaning trunk
        m |= stroke([(-0.19, 0.33), (-0.33, 0.70)], 0.09) | stroke([(-0.19, 0.33), (-0.06, 0.70)], 0.09)
        m |= stroke([(-0.07, 0.06), (0.10, 0.24)], 0.07)                        # arms
        m |= stroke([(0.00, 0.02), (0.20, 0.44)], 0.045)                        # shovel
        m |= poly([(0.06, 0.70), (0.22, 0.45), (0.36, 0.44), (0.54, 0.70)])     # heap
        c.paint(m, BLACK)
        return c
    if k == 26:                                  # traffic lights: red, amber, green lamps
        c = danger()
        c.paint(disc(0, -0.03, 0.105), RED)
        c.paint(disc(0, 0.25, 0.105), AMBER)
        c.paint(disc(0, 0.53, 0.105), GREEN)
        return c
    if k == 27:                                  # pedestrians
        c = danger()
        c.paint(person(0.0, -0.22, 0.92, lean=-0.03, stride=0.30, arm=0.14), BLACK)
        return c
    if k == 28:                                  # children: two running figures
        c = danger()
        c.paint(person(0.17, -0.08, 0.76, lean=-0.08, stride=0.55) | person(-0.20, 0.10, 0.58, lean=-0.08, stride=0.55), BLACK)
        return c
    if k == 29:                                  # cyclists: bicycle seen from the side
        c = danger()
        m = np.zeros(X.shape, bool)
        for wx in (-0.27, 0.27):
            m |= disc(wx, 0.50, 0.20) & ~disc(wx, 0.50, 0.135)
        m |= stroke([(-0.27, 0.50), (-0.10, 0.22), (0.17, 0.22), (0.27, 0.50)], 0.05)
        m |= stroke([(-0.10, 0.22), (0.00, 0.50), (0.17, 0.22)], 0.05)
        m |= stroke([(-0.16, 0.14), (-0.04, 0.14)], 0.06) | stroke([(0.17, 0.22), (0.14, 0.08), (0.24, 0.08)], 0.05)
        c.paint(m, BLACK)
        return c
    if k == 30:                                  # ice or snow: snow crystal
        c = danger()
        m = np.zeros(X.shape, bool)
        cy, L = 0.34, 0.33
        for i in range(6):
            a = math.radians(30 + 60 * i)
            ux, uy = math.cos(a), math.sin(a)
            m |= stroke([(0, cy), (L * ux, cy + L * uy)], 0.065)
            bx, by = 0.62 * L * ux, cy + 0.62 * L * uy
            for sgn in (-1, 1):
                b = a + sgn * math.radians(55)
                m |= stroke([(bx, by), (bx + 0.13 * math.cos(b), by + 0.13 * math.sin(b))], 0.05)
        c.paint(m, BLACK)
        return c
    if k == 31:                                  # wild animals: deer leaping up to the left
        c = danger()
        m = stroke([(0.26, 0.40), (-0.06, 0.20)], 0.17)                         # body
        m |= stroke([(-0.06, 0.18), (-0.20, 0.00)], 0.10)                       # neck
        m |= stroke([(-0.20, 0.00), (-0.31, 0.04)], 0.08)                       # head
        m |= stroke([(-0.20, -0.02), (-0.17, -0.17)], 0.04) | stroke([(-0.18, -0.10), (-0.09, -0.16)], 0.04)   # antlers
        m |= stroke([(-0.08, 0.26), (-0.26, 0.34), (-0.36, 0.28)], 0.055)       # fore legs
        m |= stroke([(-0.02, 0.28), (-0.16, 0.44)], 0.055)
        m |= stroke([(0.28, 0.44), (0.42, 0.52), (0.52, 0.68)], 0.06)           # hind legs
        m |= stroke([(0.22, 0.46), (0.34, 0.62), (0.40, 0.70)], 0.055)
        c.paint(m, BLACK)
        return c
    if k == 32:                                  # end of all limits
        c = end_disc()
        end_band(c)
        return c
    if k in (33, 34):                            # turn right ahead (33); 34 is its mirror image
        c = blue_disc()
        p = [(-0.30, 0.66), (-0.30, 0.10)] + earc(0.0, 0.10, 0.30, 0.30, 180, 270, 10) + [(0.26, -0.20)]
        m = stroke(p, 0.20, butt=(True, True)) | arrow_head((0.72, -0.20), (1, 0), 0.48, 0.36)
        c.paint(m, WHITE)
        return c.mirror() if k == 34 else c
    if k == 35:                                  # ahead only
        c = blue_disc()
        c.paint(rect(-0.105, 0.105, -0.20, 0.70) | arrow_head((0, -0.74), (0, -1), 0.56, 0.38), WHITE)
        return c
    if k in (36, 37):                            # straight or right (36); 37 is its mirror image
        c = blue_disc()
        m = rect(-0.32, -0.14, -0.26, 0.70) | arrow_head((-0.23, -0.74), (0, -1), 0.50, 0.30)
        p = [(-0.23, 0.50)] + earc(0.05, 0.42, 0.28, 0.28, 180, 270, 10) + [(0.36, 0.14)]
        m |= stroke(p, 0.16, butt=(False, True)) | arrow_head((0.76, 0.14), (1, 0), 0.42, 0.28)
        c.paint(m, WHITE)
        return c.mirror() if k == 37 else c
    if k in (38, 39):                            # keep right (38): arrow pointing down to the right; 39 mirror
        c = blue_disc()
        m = stroke([(-0.44, -0.44), (0.20, 0.20)], 0.20, butt=(True, True)) | arrow_head((0.56, 0.56), (1, 1), 0.52, 0.36)
        c.paint(m, WHITE)
        return c.mirror() if k == 39 else c
    if k == 40:                                  # roundabout: three arrows chasing each other anticlockwise
        c = blue_disc()
        m = np.zeros(X.shape, bool)
        for a0 in (125, 245, 365):               # each arc runs anticlockwise on the sign = decreasing screen angle
            m |= stroke(earc(0, 0, 0.50, 0.50, a0, a0 - 80, 14), 0.18, butt=(True, True))
            a = math.radians(a0 - 80)
            d = (math.sin(a), -math.cos(a))      # direction of travel for decreasing angle
            base = (0.50 * math.cos(a), 0.50 * math.sin(a))
            m |= arrow_head((base[0] + d[0] * 0.29, base[1] + d[1] * 0.29), d, 0.31, 0.25)
        c.paint(m, WHITE)
        return c
    if k == 41:                                  # end of no passing
        c = end_disc()
        c.paint(car_front(-0.31, 0.0) | car_front(0.31, 0.0), DGREY)
        end_band(c)
        return c
    if k == 42:                                  # end of no passing for lorries
        c = end_disc()
        c.paint(truck_front(-0.31, 0.06) | car_front(0.31, 0.06), DGREY)
        end_band(c)
        return c
    raise ValueError(k)
# ---------------------------------------------------------------------------------------------
# Part 2: turning each drawing into matched-filter templates and writing the network.
#
# The network is a hand-built template matcher:
#   colour path : blurred [luminance, red-ness, blue-ness] -> region-normalised -> correlation with
#                 drawn templates over +-3 pixel shifts (three window sizes: full, middle, inner)
#   edge path   : rectified oriented derivatives (8 directions) pooled like HOG cells -> normalised
#                 -> correlation with the same drawings passed through the same formula
#   every (class, size) drawing is its own template; scores of the cues are summed per shift,
#   then max over shifts, then max over the size variants of a class.
#   Each drawing is split into an outer part (border, outline: w1, w2, h1) and an inner part
#   (digits, pictogram: w3, h2, h3) that are matched and normalised separately, so that an empty
#   interior does not out-score a faint pictogram. Luminance is matched as contrast (mean removed),
#   the two colour channels as absolute colour (grey = 0).
# ---------------------------------------------------------------------------------------------
import json
import os
import sys

NCLS = 43
IMG = 32
SCALES = [18.75, 20.25, 21.75, 23.25, 24.75, 26.25, 27.75]     # width / height of the sign in pixels
VARIANTS = [(i, j) for i in range(len(SCALES)) for j in range(len(SCALES)) if abs(i - j) <= 1]
SIG_CAM = 0.6        # optical blur of the camera, pixels
SIG_COL = 0.7        # blur of the colour path
CHROMA = 1.5         # gain of the two colour-opponent channels relative to luminance
SIG_DER = 0.8        # scale of the derivative filters
SIG_PF = 1.0         # pooling of the fine edge maps
SIG_PC = 2.0         # pooling of the coarse edge maps (stride 2)
NDIR = 8
DELTA = 1            # the inner window sits one pixel below the centre (triangle pictograms sit low)
INNER_U, INNER_V0, INNER_V1 = 0.66, -0.52, 0.80     # inner region in sign coordinates
SIL = 0.5            # assumed contrast of the sign outline against an unknown background
CUE_W = dict(w1=1.0, w2=1.0, w3=1.0, h1=1.0, h2=1.0, h3=1.0)
EPS_T = dict(col=0.03, hog=0.02)   # floor on a template's own spread (blank templates stay small)


def _phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def resample_matrix(D):
    """(32, M): box pixel of the 32 grid, blurred by the camera, sampled from the master grid for a sign of size D"""
    xm = (IMG - 1) / 2.0 + _xs * D / 2.0
    i = np.arange(IMG)[:, None]
    A = _phi((i + 0.5 - xm[None, :]) / SIG_CAM) - _phi((i - 0.5 - xm[None, :]) / SIG_CAM)
    return A * (xm[1] - xm[0])


def gauss1d(sig, k):
    x = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-0.5 * (x / sig) ** 2)
    return g / g.sum()


def conv(x, W, b=None, stride=1, padding=0):
    """same arithmetic as the scorer's conv, for (N, C, H, W) float64"""
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    out = np.zeros((n, o, ho, wo))
    for dy in range(kh):
        for dx in range(kw):
            patch = xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
            wk = W[:, :, dy, dx]
            if np.any(wk):
                out += np.einsum('nchw,oc->nohw', patch, wk)
    if b is not None:
        out += b[None, :, None, None]
    return out


def avgpool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


# ------------------------------------------------------------------ fixed front-end filters
def frontend():
    Wt = {}
    # colour path: [luminance, red-ness, blue-ness], Gaussian blurred
    g = gauss1d(SIG_COL, 5); G2 = np.outer(g, g)
    mix = np.array([[1 / 3.0, 1 / 3.0, 1 / 3.0],
                    [CHROMA, -CHROMA / 2, -CHROMA / 2],
                    [-CHROMA / 2, -CHROMA / 2, CHROMA]])
    Wt['colf'] = mix[:, :, None, None] * G2[None, None]
    # edge path: derivative of Gaussian in 8 directions for each of R, G, B
    x = np.arange(5) - 2.0
    g = np.exp(-0.5 * (x / SIG_DER) ** 2); g /= g.sum()
    d = -x / SIG_DER ** 2 * np.exp(-0.5 * (x / SIG_DER) ** 2); d /= np.abs(d * x).sum()     # unit response to a unit ramp
    Gx = np.outer(g, d)      # derivative along x (columns)
    Gy = np.outer(d, g)      # derivative along y (rows, downward)
    Wg = np.zeros((3 * NDIR, 3, 5, 5))
    for c in range(3):
        for j in range(NDIR):
            a = 2 * math.pi * j / NDIR
            Wg[c * NDIR + j, c] = math.cos(a) * Gx + math.sin(a) * Gy
    Wt['hg'] = Wg
    # sharpen the direction tuning: response minus the two perpendicular ones (zero beyond 45 degrees)
    Ws = np.zeros((3 * NDIR, 3 * NDIR, 1, 1))
    for c in range(3):
        for j in range(NDIR):
            Ws[c * NDIR + j, c * NDIR + j] = 1.0
            Ws[c * NDIR + j, c * NDIR + (j + NDIR // 4) % NDIR] = -1.0
            Ws[c * NDIR + j, c * NDIR + (j - NDIR // 4) % NDIR] = -1.0
    Wt['hs'] = Ws
    Wq = np.zeros((NDIR, 3 * NDIR, 1, 1))
    for c in range(3):
        for j in range(NDIR):
            Wq[j, c * NDIR + j] = 1.0
    Wt['hq'] = Wq
    g = gauss1d(SIG_PF, 5); Wf = np.zeros((NDIR, NDIR, 5, 5))
    gc = gauss1d(SIG_PC, 10); Wc = np.zeros((NDIR, NDIR, 10, 10))
    for j in range(NDIR):
        Wf[j, j] = np.outer(g, g)
        Wc[j, j] = np.outer(gc, gc)
    Wt['hpf'] = Wf
    Wt['hpc'] = Wc
    return Wt


FE = frontend()


def edge_maps(x):
    """x (N, 3, 32, 32) -> direction energies summed over colour, (N, 8, 28, 28) for pixels 2..29"""
    r = np.maximum(conv(x, FE['hg']), 0.0)
    q = np.maximum(conv(r, FE['hs']), 0.0)
    return conv(q, FE['hq'])


def pool_fine(q):       # (N, C, 28, 28) -> (N, C, 24, 24), pixels 4..27
    return conv(q, FE['hpf'][:q.shape[1], :q.shape[1]])


def pool_coarse(q):     # (N, C, 28, 28) -> (N, C, 14, 14), cells centred on pixels 2.5, 4.5, ...
    return conv(q, FE['hpc'][:q.shape[1], :q.shape[1]], stride=2, padding=4)


# ------------------------------------------------------------------ templates
def standardise(t, w, eps, colour):
    """t (C, h, w) values, w (h, w) support weights -> kernel whose response to a standardised copy of t is 1.
    colour=True : channel 0 (luminance) loses its own mean, the colour channels stay absolute
    colour=False: one joint mean over all channels is removed (edge energies)"""
    C = t.shape[0]
    sw = w.sum()
    if sw < 1e-6:
        return np.zeros_like(t)
    z = t.copy()
    if colour:
        z[0] = z[0] - (z[0] * w).sum() / sw
    else:
        z = z - (z * w).sum() / (C * sw)
    s = math.sqrt((z * z * w).sum() / (C * sw) + eps * eps)
    return z * w / (s * C * sw)


def make_templates(masters):
    """masters: list of 43 Canvas. Returns dict cue -> (K, C, k, k) kernels, K = len(VARIANTS) * 43, index v * 43 + class"""
    A = [resample_matrix(D) for D in SCALES]
    out = {c: [] for c in ('w1', 'w2', 'w3', 'h1', 'h2', 'h3')}
    gw = gauss1d(2.5, 13); Wwide = np.zeros((4, 4, 13, 13))
    for c in range(4):
        Wwide[c, c] = np.outer(gw, gw)
    gb = gauss1d(SIG_COL, 5); Wblur = np.outer(gb, gb)[None, None]
    pix = np.arange(IMG) - (IMG - 1) / 2.0
    for (ix, iy) in VARIANTS:
        Ax, Ay = A[ix], A[iy]
        P = np.zeros((NCLS, 3, IMG, IMG)); a = np.zeros((NCLS, 1, IMG, IMG))
        for k, cv in enumerate(masters):
            for c in range(3):
                P[k, c] = Ay @ (cv.col[:, :, c] * cv.alp) @ Ax.T
            a[k, 0] = Ay @ cv.alp @ Ax.T
        # inner region of this size variant
        u = pix[None, :] / (SCALES[ix] / 2.0); v = pix[:, None] / (SCALES[iy] / 2.0)
        inner = ((np.abs(u) <= INNER_U) & (v >= INNER_V0) & (v <= INNER_V1)).astype(float)
        r0 = 8 + DELTA
        win = np.zeros((IMG, IMG)); win[r0:r0 + 16, 8:24] = 1.0
        inner = inner * win                                  # inner part: region of the pictogram, inside the inner window
        inner14 = inner[2:30, 2:30].reshape(14, 2, 14, 2).mean((1, 3))
        outer, outer14 = 1.0 - inner, 1.0 - inner14
        # ---- colour cues
        Fp = conv(P, FE['colf'], padding=2)
        Fa = conv(a, Wblur, padding=2)
        tc = Fp / np.maximum(Fa, 0.05)
        wc = np.clip((Fa[:, 0] - 0.3) / 0.5, 0.0, 1.0)
        Fp1 = avgpool2(Fp[:, :, 2:30, 2:30]); Fa1 = avgpool2(Fa[:, :, 2:30, 2:30])
        t1 = Fp1 / np.maximum(Fa1, 0.05)
        w1 = np.clip((Fa1[:, 0] - 0.3) / 0.5, 0.0, 1.0)
        # ---- edge cues: sign continued outwards (no outline edge) plus an unsigned outline of assumed contrast
        Pa = np.concatenate([P, a], 1)
        wide = conv(Pa, Wwide, padding=6)
        fill = np.where(wide[:, 3:4] > 1e-3, wide[:, :3] / np.maximum(wide[:, 3:4], 1e-3), 0.45)
        q_int = edge_maps(P + (1.0 - a) * fill)
        q_a = edge_maps(np.repeat(a, 3, 1) * SIL)
        q_sil = 0.5 * (q_a + np.roll(q_a, NDIR // 2, axis=1))
        q = q_int + q_sil
        Ef = pool_fine(q); Ec = pool_coarse(q)
        a28 = a[:, :, 2:30, 2:30]
        af = pool_fine(a28)[:, 0]; ac = pool_coarse(a28)[:, 0]
        wf = np.clip((af - 0.05) / 0.2, 0.0, 1.0); wcs = np.clip((ac - 0.05) / 0.2, 0.0, 1.0)
        for k in range(NCLS):
            out['w1'].append(standardise(t1[k], w1[k] * outer14, EPS_T['col'], True))
            out['w2'].append(standardise(tc[k][:, 5:27, 5:27], (wc[k] * outer)[5:27, 5:27], EPS_T['col'], True))
            out['w3'].append(standardise(tc[k][:, r0:r0 + 16, 8:24], (wc[k] * inner)[r0:r0 + 16, 8:24], EPS_T['col'], True))
            out['h1'].append(standardise(Ec[k], wcs[k] * outer14, EPS_T['hog'], False))
            out['h2'].append(standardise(Ec[k][:, 2:12, 2:12], (wcs[k] * inner14)[2:12, 2:12], EPS_T['hog'], False))
            out['h3'].append(standardise(Ef[k][:, r0 - 4:r0 + 12, 4:20], (wf[k] * inner[4:28, 4:28])[r0 - 4:r0 + 12, 4:20], EPS_T['hog'], False))
    return {c: np.stack(v) * CUE_W[c] for c, v in out.items()}


# ------------------------------------------------------------------ network assembly
def delta_conv(C, k, dy, dx):
    W = np.zeros((C, C, k, k))
    for c in range(C):
        W[c, c, dy, dx] = 1.0
    return W


def build(out_dir):
    masters = [sign(k) for k in range(NCLS)]
    T = make_templates(masters)
    K = len(VARIANTS) * NCLS
    layers, weights = [], {}

    def L(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=list(inputs), **kw))
        return name

    def C(name, inp, W, stride=1, padding=0, b=None):
        weights[name + '.W'] = W.astype(np.float32)
        weights[name + '.b'] = (np.zeros(W.shape[0]) if b is None else b).astype(np.float32)
        return L(name, 'conv', [inp], out=int(W.shape[0]), kernel=[int(W.shape[2]), int(W.shape[3])], stride=stride, padding=padding)

    def joint_norm(name, inp, shape):
        c, h, w = shape
        L(name + '_r', 'reshape', [inp], shape=[1, c * h, w])
        L(name + '_n', 'norm', [name + '_r'])
        return L(name, 'reshape', [name + '_n'], shape=[c, h, w])

    def colour_norm(name, inp, shape):
        """luminance minus its mean over the map; then all three channels divided by their joint r.m.s.
        (the colour channels keep their zero: the map is stacked with its negative so the joint mean is 0)"""
        c, h, w = shape
        L(name + '_m', 'avgpool', [inp], size=[h, w], stride=[h, w])
        Wd = np.zeros((c * h * w, c))
        Wd[0:h * w, 0] = -1.0
        weights[name + '_d.W'] = Wd.astype(np.float32); weights[name + '_d.b'] = np.zeros(c * h * w, np.float32)
        L(name + '_d', 'dense', [name + '_m'], out=c * h * w)
        L(name + '_e', 'reshape', [name + '_d'], shape=[c, h, w])
        L(name + '_z', 'add', [inp, name + '_e'])
        Wpm = np.zeros((2 * c, c, 1, 1)); Wsel = np.zeros((c, 2 * c, 1, 1))
        for i in range(c):
            Wpm[i, i] = 1.0; Wpm[c + i, i] = -1.0; Wsel[i, i] = 1.0
        C(name + '_pm', name + '_z', Wpm)
        L(name + '_r', 'reshape', [name + '_pm'], shape=[1, 2 * c * h, w])
        L(name + '_n', 'norm', [name + '_r'])
        L(name + '_b', 'reshape', [name + '_n'], shape=[2 * c, h, w])
        return C(name, name + '_b', Wsel)

    # ---- colour path
    C('colf', 'image', FE['colf'], padding=2)                                   # (3, 32, 32)
    C('c28', 'colf', delta_conv(3, 5, 2, 2))                                    # pixels 2..29 -> (3, 28, 28)
    colour_norm('c28n', 'c28', (3, 28, 28))
    C('w2', 'c28n', T['w2'])                                                    # (K, 7, 7)
    C('c22', 'colf', delta_conv(3, 11, 5 + DELTA, 5))                           # (3, 22, 22)
    colour_norm('c22n', 'c22', (3, 22, 22))
    C('w3', 'c22n', T['w3'])                                                    # (K, 7, 7)
    L('c14', 'avgpool', ['c28'], size=[2, 2], stride=[2, 2])                    # (3, 14, 14)
    colour_norm('c14n', 'c14', (3, 14, 14))
    C('w1', 'c14n', T['w1'], padding=1)                                         # (K, 3, 3)
    # ---- edge path
    C('hg', 'image', FE['hg'])                                                  # (24, 28, 28)
    L('hg_r', 'relu', ['hg'])
    C('hs', 'hg_r', FE['hs'])
    L('hs_r', 'relu', ['hs'])
    C('hq', 'hs_r', FE['hq'])                                                   # (8, 28, 28)
    C('hf', 'hq', FE['hpf'])                                                    # (8, 24, 24)
    C('hf22', 'hf', delta_conv(NDIR, 3, 1 + DELTA, 1))                          # (8, 22, 22)
    joint_norm('hf22n', 'hf22', (NDIR, 22, 22))
    C('h3', 'hf22n', T['h3'])                                                   # (K, 7, 7)
    C('hc', 'hq', FE['hpc'], stride=2, padding=4)                               # (8, 14, 14)
    joint_norm('hcn', 'hc', (NDIR, 14, 14))
    C('h1', 'hcn', T['h1'], padding=1)                                          # (K, 3, 3)
    C('hc12', 'hc', delta_conv(NDIR, 3, 1, 1))                                  # (8, 12, 12)
    joint_norm('hc12n', 'hc12', (NDIR, 12, 12))
    C('h2', 'hc12n', T['h2'])                                                   # (K, 3, 3)
    # ---- combine: fine cues per pixel shift, then with the coarse cues per 2-pixel shift
    L('fine', 'add', ['w2', 'w3', 'h3'])
    L('fine_p', 'maxpool', ['fine'], size=[3, 3], stride=[2, 2])                # (K, 3, 3)
    L('all', 'add', ['fine_p', 'w1', 'h1', 'h2'])
    L('best', 'maxpool', ['all'], size=[3, 3], stride=[3, 3])                   # (K, 1, 1)
    L('byvar', 'reshape', ['best'], shape=[1, len(VARIANTS), NCLS])
    L('cls', 'maxpool', ['byvar'], size=[len(VARIANTS), 1], stride=[len(VARIANTS), 1])
    L('scores', 'reshape', ['cls'], shape=[NCLS])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'network.json'), 'w') as f:
        json.dump({'layers': layers}, f, indent=1)
    np.savez(os.path.join(out_dir, 'weights.npz'), **weights)
    return layers, weights


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
