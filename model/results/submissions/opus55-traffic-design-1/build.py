# build.py: writes network.json and weights.npz for a hand-designed GTSRB classifier (no data read).
# Vector drawings of the 43 German traffic sign classes, from general knowledge of the StVO designs.
# Sign coordinates: u to the right, v downwards; the sign's bounding box spans u in [-1, 1].
import numpy as np

WHITE = (0.85, 0.85, 0.85)
RED = (0.72, 0.12, 0.12)
BLACK = (0.07, 0.07, 0.07)
BLUE = (0.08, 0.22, 0.55)
YELLOW = (0.92, 0.62, 0.08)
GREY = (0.45, 0.45, 0.45)
DGREY = (0.28, 0.28, 0.28)
GREEN = (0.05, 0.55, 0.35)
AMBER = (0.95, 0.65, 0.05)
BG = (0.45, 0.45, 0.45)


# ---------------------------------------------------------------- geometry on point arrays
def seg_dist(U, V, a, b):
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((U - ax) * dx + (V - ay) * dy) / max(L2, 1e-12), 0.0, 1.0)
    return np.hypot(U - ax - t * dx, V - ay - t * dy)


def poly_dist(U, V, pts, closed=False):
    pts = list(pts)
    if closed:
        pts = pts + [pts[0]]
    d = np.full(U.shape, np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        d = np.minimum(d, seg_dist(U, V, a, b))
    return d


def poly_inside(U, V, pts):
    inside = np.zeros(U.shape, bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        cond = ((y1 > V) != (y2 > V)) & (U < (x2 - x1) * (V - y1) / (y2 - y1) + x1)
        inside ^= cond
    return inside


def arc(cx, cy, rx, ry, a0, a1, n=None):
    """points on an elliptical arc; angles in degrees, 0 = right, 90 = down (v grows downwards)."""
    if n is None:
        n = max(4, int(abs(a1 - a0) / 6) + 1)
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]


def tri_pts(apex_up=True, inr=None):
    """equilateral triangle with side 2 centred on its bounding box (true geometry, height sqrt(3)).
    inr: inradius of the (possibly shrunk) triangle, same incentre."""
    h = np.sqrt(3.0)
    if apex_up:
        cen = (0.0, h / 2 - h / 3)       # incentre lies 1/3 height above the base
        base = [(0.0, -h / 2), (1.0, h / 2), (-1.0, h / 2)]
    else:
        cen = (0.0, -h / 2 + h / 3)
        base = [(0.0, h / 2), (-1.0, -h / 2), (1.0, -h / 2)]
    r0 = 1.0 / np.sqrt(3.0)
    if inr is None:
        inr = r0
    k = inr / r0
    return [(cen[0] + k * (x - cen[0]), cen[1] + k * (y - cen[1])) for x, y in base], cen


class Canvas:
    def __init__(self, U, V):
        self.U, self.V = U, V
        self.rgb = np.empty((3,) + U.shape)
        for c in range(3):
            self.rgb[c] = BG[c]
        self.alpha = np.zeros(U.shape)

    def paint(self, m, col):
        for c in range(3):
            self.rgb[c][m] = col[c]

    # shapes -> boolean masks
    def disc(self, cx, cy, r):
        return (self.U - cx) ** 2 + (self.V - cy) ** 2 <= r * r

    def ellipse(self, cx, cy, rx, ry, ang=0.0):
        a = np.radians(ang)
        du, dv = self.U - cx, self.V - cy
        x = du * np.cos(a) + dv * np.sin(a); y = -du * np.sin(a) + dv * np.cos(a)
        return (x / rx) ** 2 + (y / ry) ** 2 <= 1

    def ring(self, r0, r1, cx=0.0, cy=0.0):
        d2 = (self.U - cx) ** 2 + (self.V - cy) ** 2
        return (d2 >= r0 * r0) & (d2 <= r1 * r1)

    def poly(self, pts):
        return poly_inside(self.U, self.V, pts)

    def rpoly(self, pts, r):
        """polygon pts dilated by r (rounded corners)."""
        return poly_inside(self.U, self.V, pts) | (poly_dist(self.U, self.V, pts, closed=True) <= r)

    def rect(self, u0, v0, u1, v1):
        return (self.U >= u0) & (self.U <= u1) & (self.V >= v0) & (self.V <= v1)

    def stroke(self, pts, w, closed=False):
        return poly_dist(self.U, self.V, pts, closed) <= w / 2

    def strokes(self, paths, w):
        m = np.zeros(self.U.shape, bool)
        for p in paths:
            m |= self.stroke(p, w)
        return m


# ---------------------------------------------------------------- digits and letters (DIN 1451 style, drawn as strokes)
SW = 0.14   # stroke width relative to the cap height


def glyph_paths(ch, w):
    """centre lines of a glyph in a box [0, w] x [0, 1] (y down)."""
    h = SW / 2
    rx = w / 2 - h
    if ch == '0':
        ry = 0.27
        top, bot = h + ry, 1 - h - ry
        p = arc(w / 2, top, rx, ry, 180, 360) + arc(w / 2, bot, rx, ry, 0, 180)
        return [p + [p[0]]]
    if ch == '1':
        x = 0.62 * w
        return [[(x, 1 - h), (x, h), (0.12 * w, 0.30)]]
    if ch == '2':
        p = arc(w / 2, 0.30, rx, 0.30 - h, 195, 345)
        q = arc(w / 2, 0.30, rx, 0.30 - h, 345, 380)
        return [p + q + [(h, 1 - h), (w - h, 1 - h)]]
    if ch == '3':
        a = arc(0.48 * w, 0.265, rx - 0.02, 0.265 - h, 195, 450)
        b = arc(w / 2, 0.72, rx, 0.28 - h, 270, 520)
        return [a, b]
    if ch == '5':
        x0 = 0.16 * w
        top = [(w - h, h), (x0, h), (x0 - 0.01, 0.47)]
        b = arc(w / 2, 0.67, rx, 0.33 - h, 215, 515)
        return [top, b]
    if ch == '6':
        loop = arc(w / 2, 0.68, rx, 0.32 - h, 0, 360)
        stem = [(h, 0.68), (h, 0.40)] + arc(w / 2, 0.40, rx, 0.40 - h, 180, 305)
        return [loop, stem]
    if ch == '7':
        return [[(h, h), (w - h, h), (0.32 * w, 1 - h)]]
    if ch == '8':
        a = arc(w / 2, 0.255, rx - 0.03, 0.255 - h, 0, 360)
        b = arc(w / 2, 0.72, rx, 0.28 - h, 0, 360)
        return [a, b]
    if ch == 'S':
        a = arc(w / 2, 0.265, rx, 0.265 - h, 335, 90)
        b = arc(w / 2, 0.735, rx, 0.265 - h, 270, 515)
        return [a, b]
    if ch == 'T':
        return [[(h, h), (w - h, h)], [(w / 2, h), (w / 2, 1 - h)]]
    if ch == 'O':
        ry = 0.5 - h
        return [arc(w / 2, 0.5, rx, ry, 0, 360)]
    if ch == 'P':
        bowl = [(h, 1 - h), (h, h), (w / 2 - 0.05, h)] + arc(w / 2 - 0.05, 0.29, rx - 0.0, 0.29 - h, 270, 450) + [(h, 0.58)]
        return [bowl]
    raise ValueError(ch)


def text(cv, s, H, cx=0.0, cy=0.0, widths=None, gap=0.08, sw=SW):
    """mask of string s with cap height H centred at (cx, cy) in sign coordinates."""
    if widths is None:
        widths = [0.56] * len(s)
    total = sum(widths) + gap * (len(s) - 1)
    x = cx - H * total / 2
    m = np.zeros(cv.U.shape, bool)
    for ch, w in zip(s, widths):
        paths = glyph_paths(ch, w)
        for p in paths:
            pts = [(x + H * a, cy - H / 2 + H * b) for a, b in p]
            m |= cv.stroke(pts, sw * H)
        x += H * (w + gap)
    return m


# ---------------------------------------------------------------- vehicles
def car_rear(cv, cx, cy, s):
    m = cv.rpoly([(cx - 0.48 * s, cy - 0.02 * s), (cx + 0.48 * s, cy - 0.02 * s), (cx + 0.48 * s, cy + 0.20 * s), (cx - 0.48 * s, cy + 0.20 * s)], 0.04 * s)
    m |= cv.poly([(cx - 0.36 * s, cy), (cx + 0.36 * s, cy), (cx + 0.24 * s, cy - 0.22 * s), (cx - 0.24 * s, cy - 0.22 * s)])
    m |= cv.rect(cx - 0.46 * s, cy + 0.18 * s, cx - 0.28 * s, cy + 0.32 * s)
    m |= cv.rect(cx + 0.28 * s, cy + 0.18 * s, cx + 0.46 * s, cy + 0.32 * s)
    return m


def truck_rear(cv, cx, cy, s):
    m = cv.rect(cx - 0.42 * s, cy - 0.5 * s, cx + 0.42 * s, cy + 0.32 * s)
    m |= cv.rect(cx - 0.40 * s, cy + 0.30 * s, cx - 0.18 * s, cy + 0.46 * s)
    m |= cv.rect(cx + 0.18 * s, cy + 0.30 * s, cx + 0.40 * s, cy + 0.46 * s)
    return m


def truck_side(cv, cx, cy, s):
    m = cv.rect(cx - 0.30 * s, cy - 0.30 * s, cx + 0.62 * s, cy + 0.14 * s)          # cargo box
    m |= cv.rpoly([(cx - 0.60 * s, cy - 0.12 * s), (cx - 0.36 * s, cy - 0.16 * s), (cx - 0.34 * s, cy + 0.16 * s), (cx - 0.62 * s, cy + 0.16 * s)], 0.02 * s)
    m |= cv.rect(cx - 0.62 * s, cy + 0.10 * s, cx + 0.62 * s, cy + 0.19 * s)          # chassis
    for wx in (-0.46, 0.18, 0.44):
        m |= cv.disc(cx + wx * s, cy + 0.22 * s, 0.10 * s)
    win = cv.rect(cx - 0.56 * s, cy - 0.08 * s, cx - 0.42 * s, cy + 0.0 * s)
    return m & ~win


# ---------------------------------------------------------------- sign families
def red_ring(cv, rin=0.78):
    o = cv.disc(0, 0, 1.0)
    cv.alpha = o.astype(float)
    cv.paint(o, RED)
    cv.paint(cv.disc(0, 0, rin), WHITE)


def end_disc(cv):
    o = cv.disc(0, 0, 1.0)
    cv.alpha = o.astype(float)
    cv.paint(o, DGREY)
    cv.paint(cv.disc(0, 0, 0.95), WHITE)


def end_stripes(cv):
    # five thin black stripes in a band running from lower left to upper right ("/")
    m = np.zeros(cv.U.shape, bool)
    d = (cv.U + cv.V) / np.sqrt(2)              # distance across the band
    for k in range(5):
        c = (k - 2) * 0.085
        m |= np.abs(d - c) <= 0.021
    m &= cv.disc(0, 0, 0.95)
    cv.paint(m, BLACK)


def blue_disc(cv):
    o = cv.disc(0, 0, 1.0)
    cv.alpha = o.astype(float)
    cv.paint(o, WHITE)
    cv.paint(cv.disc(0, 0, 0.92), BLUE)


def arrow_head(tip, direction, length, half):
    dx, dy = direction
    n = np.hypot(dx, dy); dx, dy = dx / n, dy / n
    bx, by = tip[0] - length * dx, tip[1] - length * dy
    return [tip, (bx - half * dy, by + half * dx), (bx + half * dy, by - half * dx)]


def tri_frame(cv, apex_up=True, border=0.17, corner=0.08, col=RED):
    pts, cen = tri_pts(apex_up, 1 / np.sqrt(3) - corner)
    o = cv.rpoly(pts, corner)
    cv.alpha = o.astype(float)
    cv.paint(o, col)
    ip, _ = tri_pts(apex_up, 1 / np.sqrt(3) - border - 0.03)
    cv.paint(cv.rpoly(ip, 0.03), WHITE)
    return cen


def person(cv, hx, hy, s, lean=0.0, flip=1.0):
    """black walking figure with the head at (hx, hy); s = height scale; flip mirrors."""
    m = cv.disc(hx, hy, 0.085 * s)
    sx, sy = hx + flip * lean * s, hy + 0.12 * s
    hipx, hipy = hx + flip * (lean * 0.5) * s - flip * 0.02 * s, hy + 0.48 * s
    m |= cv.stroke([(sx, sy), (hipx, hipy)], 0.14 * s)
    m |= cv.stroke([(hipx, hipy), (hipx - flip * 0.18 * s, hy + 0.88 * s)], 0.10 * s)
    m |= cv.stroke([(hipx, hipy), (hipx + flip * 0.10 * s, hy + 0.66 * s), (hipx + flip * 0.20 * s, hy + 0.88 * s)], 0.10 * s)
    m |= cv.stroke([(sx, sy + 0.06 * s), (sx - flip * 0.17 * s, hy + 0.40 * s)], 0.08 * s)
    m |= cv.stroke([(sx, sy + 0.06 * s), (sx + flip * 0.16 * s, hy + 0.36 * s)], 0.08 * s)
    return m


def draw(cv, cls, var):
    """draw class cls with variant dict var on canvas cv. Returns the family name."""
    mir = -1.0 if var.get('mirror') else 1.0
    if cls in (0, 1, 2, 3, 4, 5, 7, 8):
        red_ring(cv)
        s = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}[cls]
        H = var.get('H', 0.92)
        if len(s) == 2:
            m = text(cv, s, H, widths=[0.56, 0.56], gap=0.10)
        else:
            m = text(cv, s, H * 0.92, widths=[0.34, 0.47, 0.47], gap=0.07)
        cv.paint(m, BLACK)
        return 'circle'
    if cls == 6:
        end_disc(cv)
        H = var.get('H', 0.92)
        cv.paint(text(cv, '80', H, widths=[0.56, 0.56], gap=0.10), DGREY)
        end_stripes(cv)
        return 'circle'
    if cls == 9:
        red_ring(cv)
        cv.paint(car_rear(cv, -0.34, 0.0, 0.52), RED)
        cv.paint(car_rear(cv, 0.34, 0.0, 0.52), BLACK)
        return 'circle'
    if cls == 10:
        red_ring(cv)
        cv.paint(truck_rear(cv, -0.32, 0.0, 0.56), RED)
        cv.paint(car_rear(cv, 0.36, 0.04, 0.50), BLACK)
        return 'circle'
    if cls == 15:
        red_ring(cv)
        return 'circle'
    if cls == 16:
        red_ring(cv)
        cv.paint(truck_side(cv, 0.0, 0.03, 0.90), BLACK)
        return 'circle'
    if cls == 14:
        a = 1.0 / np.cos(np.pi / 8)
        oct_pts = [(a * np.cos(np.pi / 8 + k * np.pi / 4), a * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)]
        o = cv.poly(oct_pts)
        cv.alpha = o.astype(float)
        cv.paint(o, WHITE)
        cv.paint(cv.poly([(0.92 * x, 0.92 * y) for x, y in oct_pts]), RED)
        if not var.get('frame'):
            cv.paint(text(cv, 'STOP', 0.54, widths=[0.62, 0.62, 0.68, 0.62], gap=0.09, sw=0.17), WHITE)
        return 'circle'
    if cls == 17:
        o = cv.disc(0, 0, 1.0)
        cv.alpha = o.astype(float)
        cv.paint(o, WHITE)
        cv.paint(cv.disc(0, 0, 0.94), RED)
        if not var.get('frame'):
            cv.paint(cv.rect(-0.72, -0.16, 0.72, 0.16), WHITE)
        return 'circle'
    if cls == 32:
        end_disc(cv)
        end_stripes(cv)
        return 'circle'
    if cls == 41:
        end_disc(cv)
        cv.paint(car_rear(cv, -0.34, 0.0, 0.52), GREY)
        cv.paint(car_rear(cv, 0.34, 0.0, 0.52), GREY)
        end_stripes(cv)
        return 'circle'
    if cls == 42:
        end_disc(cv)
        cv.paint(truck_rear(cv, -0.32, 0.0, 0.56), GREY)
        cv.paint(car_rear(cv, 0.36, 0.04, 0.50), GREY)
        end_stripes(cv)
        return 'circle'
    if 33 <= cls <= 40:
        blue_disc(cv)
        if var.get('frame'):
            return 'circle'
        W = 0.21; L, HW = 0.40, 0.30
        m = np.zeros(cv.U.shape, bool)
        f = -1.0 if cls in (34, 37, 39) else 1.0
        if cls in (33, 34):
            x0 = -0.18 * f
            m |= cv.stroke([(x0, 0.75)] + arc(x0 + 0.30 * f, -0.02, 0.30, 0.30, 180 if f > 0 else 0, 270 if f > 0 else -90, 8) + [(0.30 * f, -0.32)], W)
            m |= cv.poly(arrow_head((0.74 * f, -0.32), (f, 0), L, HW))
        elif cls == 35:
            m |= cv.stroke([(0, 0.75), (0, -0.30)], W)
            m |= cv.poly(arrow_head((0, -0.76), (0, -1), L, HW))
        elif cls in (36, 37):
            x0 = -0.16 * f
            m |= cv.stroke([(x0, 0.76), (x0, -0.30)], W)
            m |= cv.poly(arrow_head((x0, -0.78), (0, -1), 0.36, 0.27))
            m |= cv.stroke([(x0, 0.45)] + arc(x0 + 0.24 * f, 0.45, 0.24, 0.26, 180 if f > 0 else 0, 270 if f > 0 else -90, 8) + [(0.32 * f, 0.19)], W * 0.95)
            m |= cv.poly(arrow_head((0.76 * f, 0.19), (f, 0), 0.36, 0.27))
        elif cls in (38, 39):
            m |= cv.stroke([(-0.48 * f, -0.48), (0.22 * f, 0.22)], W * 1.05)
            m |= cv.poly(arrow_head((0.56 * f, 0.56), (f, 1), 0.44, 0.33))
        elif cls == 40:
            r = 0.55
            for k in range(3):
                a0 = 30 + 120 * k
                a1 = a0 - 62
                m |= cv.stroke(arc(0, 0, r, r, a0, a1, 12), 0.15)
                t = np.radians(a1)
                tip = (r * np.cos(t - np.radians(26)), r * np.sin(t - np.radians(26)))
                m |= cv.poly(arrow_head(tip, (np.sin(t - np.radians(13)), -np.cos(t - np.radians(13))), 0.28, 0.21))
        cv.paint(m, WHITE)
        return 'circle'
    if cls == 12:
        o = np.abs(cv.U) + np.abs(cv.V) <= 1.0
        cv.alpha = o.astype(float)
        cv.paint(o, BLACK)
        cv.paint(np.abs(cv.U) + np.abs(cv.V) <= 0.95, WHITE)
        cv.paint(np.abs(cv.U) + np.abs(cv.V) <= var.get('inner', 0.60), YELLOW)
        return 'diamond'
    if cls == 13:
        tri_frame(cv, apex_up=False, border=0.22, corner=0.07)
        return 'down'
    # ------------------------------------------------ danger triangles (apex up)
    cen = tri_frame(cv, apex_up=True, border=0.17, corner=0.08)
    if var.get('frame'):
        return 'up'
    m = np.zeros(cv.U.shape, bool)
    if cls == 11:
        m |= cv.poly([(-0.08, -0.26), (0.08, -0.26), (0.19, 0.62), (-0.19, 0.62)])
        m |= cv.rect(-0.34, 0.04, 0.34, 0.18)
    elif cls == 18:
        m |= cv.rpoly([(-0.07, -0.28), (0.07, -0.28), (0.04, 0.32), (-0.04, 0.32)], 0.015)
        m |= cv.disc(0, 0.50, 0.075)
    elif cls in (19, 20):
        f = 1.0 if cls == 19 else -1.0       # 19 bends to the left; compact symbol inside the incircle
        q = Canvas(cv.U / 0.34, (cv.V - 0.29) / 0.34)
        m |= q.stroke([(0.35 * f, 0.95), (0.35 * f, 0.05)] + arc(-0.05 * f, 0.05, 0.40, 0.40, 0 if f > 0 else 180, -90 if f > 0 else 270, 8)
                      + [(-0.30 * f, -0.35)], 0.42)
        m |= q.poly(arrow_head((-0.95 * f, -0.35), (-f, 0), 0.65, 0.45))
    elif cls == 21:
        f = mir
        q = Canvas(cv.U / 0.34, (cv.V - 0.29) / 0.34)
        m |= q.stroke([(0.30 * f, 0.95), (0.30 * f, 0.45), (-0.25 * f, 0.0), (-0.25 * f, -0.40)], 0.38)
        m |= q.poly(arrow_head((-0.25 * f, -0.98), (0, -1), 0.62, 0.40))
    elif cls == 22:
        m |= cv.rect(-0.46, 0.48, 0.46, 0.56)
        m |= cv.disc(-0.21, 0.50, 0.13) & (cv.V <= 0.52)
        m |= cv.disc(0.21, 0.50, 0.13) & (cv.V <= 0.52)
    elif cls == 23:
        m |= car_rear(cv, 0.0, 0.17, 0.62)
        m |= cv.stroke([(-0.10, 0.36), (-0.20, 0.45), (-0.08, 0.53), (-0.18, 0.62)], 0.055)
        m |= cv.stroke([(0.14, 0.36), (0.04, 0.45), (0.16, 0.53), (0.06, 0.62)], 0.055)
    elif cls == 24:
        m |= cv.stroke([(-0.15, -0.02), (-0.15, 0.62)], 0.10)
        m |= cv.stroke([(0.20, 0.62), (0.20, 0.36), (0.05, 0.20), (0.05, -0.02)], 0.10)
    elif cls == 25:
        f = mir
        m |= person(cv, -0.10 * f, -0.18, 0.78, lean=0.10, flip=f)
        m |= cv.stroke([(-0.14 * f, 0.06), (-0.34 * f, 0.40), (-0.40 * f, 0.58)], 0.06)
        m |= cv.ellipse(0.30 * f, 0.62, 0.22, 0.14) & (cv.V <= 0.62)
    elif cls == 26:
        for yy in (-0.13, 0.18, 0.49):
            m |= cv.disc(0, yy, 0.145)
        cv.paint(m, BLACK)
        cv.paint(cv.disc(0, -0.13, 0.115), RED)
        cv.paint(cv.disc(0, 0.18, 0.115), AMBER)
        cv.paint(cv.disc(0, 0.49, 0.115), GREEN)
        return 'up'
    elif cls == 27:
        f = mir
        m |= person(cv, 0.02 * f, -0.24, 0.92, lean=0.04, flip=f)
    elif cls == 28:
        f = mir
        m |= person(cv, -0.20 * f, -0.18, 0.80, lean=0.06, flip=f)
        m |= person(cv, 0.20 * f, 0.02, 0.62, lean=0.06, flip=f)
    elif cls == 29:
        f = mir
        for wx in (-0.22, 0.22):
            m |= cv.ring(0.095, 0.15, wx, 0.47)
        m |= cv.stroke([(-0.22 * f, 0.47), (-0.02 * f, 0.47), (0.12 * f, 0.28), (-0.10 * f, 0.28), (-0.02 * f, 0.47)], 0.045)
        m |= cv.stroke([(0.12 * f, 0.28), (0.22 * f, 0.47)], 0.045)
        m |= cv.disc(0.04 * f, 0.0, 0.065)
        m |= cv.stroke([(0.02 * f, 0.07), (-0.06 * f, 0.25)], 0.09)
        m |= cv.stroke([(-0.06 * f, 0.25), (0.04 * f, 0.36), (-0.01 * f, 0.47)], 0.06)
        m |= cv.stroke([(0.01 * f, 0.10), (0.12 * f, 0.22)], 0.05)
    elif cls == 30:
        c0 = (0.0, 0.28)
        for k in range(3):
            a = np.radians(90 + 60 * k)
            dx, dy = 0.28 * np.cos(a), 0.28 * np.sin(a)
            m |= cv.stroke([(c0[0] - dx, c0[1] - dy), (c0[0] + dx, c0[1] + dy)], 0.065)
            for sgn in (-1, 1):
                ex, ey = c0[0] + sgn * 0.18 * np.cos(a), c0[1] + sgn * 0.18 * np.sin(a)
                for b in (-1, 1):
                    bb = a + b * np.radians(45) + (np.pi if sgn < 0 else 0)
                    m |= cv.stroke([(ex, ey), (ex + 0.10 * np.cos(bb), ey + 0.10 * np.sin(bb))], 0.05)
    elif cls == 31:
        f = mir
        q = Canvas(cv.U / 0.75, (cv.V - 0.29) / 0.75 + 0.11)     # deer drawn in its own frame, placed in the incircle
        m |= q.ellipse(0.0, 0.24, 0.27, 0.10, -25 * f)
        m |= q.stroke([(0.18 * f, 0.14), (0.30 * f, -0.06)], 0.09)
        m |= q.ellipse(0.34 * f, -0.10, 0.08, 0.05, -20 * f)
        m |= q.stroke([(0.30 * f, -0.12), (0.26 * f, -0.30), (0.20 * f, -0.36)], 0.035)
        m |= q.stroke([(0.30 * f, -0.12), (0.38 * f, -0.30), (0.46 * f, -0.33)], 0.035)
        m |= q.stroke([(0.16 * f, 0.20), (0.34 * f, 0.30), (0.46 * f, 0.26)], 0.05)
        m |= q.stroke([(0.14 * f, 0.22), (0.26 * f, 0.38)], 0.05)
        m |= q.stroke([(-0.20 * f, 0.32), (-0.36 * f, 0.46), (-0.48 * f, 0.58)], 0.05)
        m |= q.stroke([(-0.18 * f, 0.30), (-0.40 * f, 0.36), (-0.52 * f, 0.44)], 0.05)
    cv.paint(m, BLACK)
    return 'up'


FAMILY = {}
for _c in range(43):
    if _c == 12:
        FAMILY[_c] = 'diamond'
    elif _c == 13:
        FAMILY[_c] = 'down'
    elif _c in (11,) or 18 <= _c <= 31:
        FAMILY[_c] = 'up'
    else:
        FAMILY[_c] = 'circle'

# template variants per class (each variant is rendered as its own template)
def variants(cls):
    if cls in (0, 1, 2, 3, 4, 5, 7, 8, 6):
        return [dict(H=0.86), dict(H=0.98)]
    if cls in (21, 25, 27, 28, 29, 31):
        return [dict(), dict(mirror=True)]
    return [dict()]


# frame group of every class: (group name, class drawn as the empty frame, frame variant)
def frame_of(cls):
    if cls in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16):
        return 'RR', 15, {}
    if cls in (6, 32, 41, 42):
        return 'EN', 32, {}
    if 33 <= cls <= 40:
        return 'BL', 33, {'frame': True}
    if cls == 14:
        return 'ST', 14, {'frame': True}
    if cls == 17:
        return 'NE', 17, {'frame': True}
    if cls == 13:
        return 'YD', 13, {}
    if cls == 12:
        return 'DM', 12, {}
    return 'TR', 18, {'frame': True}

EMPTY = (15, 32, 13, 12)     # classes whose interior is just the frame


import json, os, sys

# ============================================================== design constants (all chosen by hand)
DC = 22.0                      # canonical sign size (bounding-box width) in canonical pixels
RC = DC / 2
SCALES = [19.0 * 1.07 ** k for k in range(7)]   # sign width in the 32x32 photo, pose hypotheses
SUBS = [0.965, 1.0, 1.036]     # fine scale steps inside each pose (template variants)
SHIFTS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]   # fine shifts in canonical px (template variants)
ASPECTS = {'up': [1.0, 1.1], 'down': [1.0, 1.1], 'circle': [1.0], 'diamond': [1.0]}   # vertical stretch of triangles
MARGIN = 2.5                   # dilation of the per-family normalisation window (canonical px)
SIG_T = 0.8                    # blur of templates (canonical px): camera blur + resampling blur
SIG_R = 0.55                   # resampling Gaussian, in canonical px
G_L, G_C, G_I = 8.0, 10.0, 8.0  # gains of luminance, chroma and ink features
OFF = 1.5                      # offset that keeps linear features positive through the relu
FAMS = ['circle', 'up', 'down', 'diamond']
BETA = 0.5                     # weight of the interior pictogram correlation relative to the frame correlation
THETA = 0.3                    # pictogram correlation credited to classes whose interior is empty
CAP = 1.0 + BETA               # a class score may not exceed CAP times the best frame score of its group

FINE = 0.25                    # template render grid step (canonical px)
FX = np.arange(-18.0, 18.0 + 1e-9, FINE)
NF = len(FX)


# ============================================================== feature definitions (shared by image path and templates)
def feat_from_rgb_fine(rgb):
    """rgb (3, NF, NF) on the fine grid -> (4, NF, NF) features [L, Cr, Cb, Ink] (without offsets)."""
    R, G, B = rgb
    L = (R + G + B) / 3
    Cr = R - (G + B) / 2
    Cb = B - (R + G) / 2
    st = int(round(1 / FINE))     # one canonical pixel on the fine grid

    def shift(a, dy, dx):
        p = np.pad(a, 2 * st + 1, mode='edge')
        o = 2 * st + 1
        return p[o + dy:o + dy + a.shape[0], o + dx:o + dx + a.shape[1]]
    box = sum(shift(L, dy * st, dx * st) for dy in range(-2, 3) for dx in range(-2, 3)) / 25
    g1 = np.array([1, 2, 1]) / 4
    g3 = sum(g1[a + 1] * g1[b + 1] * shift(L, a * st, b * st) for a in (-1, 0, 1) for b in (-1, 0, 1))
    ink = np.maximum(box - g3, 0)
    return np.stack([G_L * L, G_I * ink, G_C * Cr, G_C * Cb])


def feature_conv():
    """5x5 conv on the RGB photo giving [L, Ink-before-relu, Cr, Cb] (+offsets); a relu follows."""
    W = np.zeros((4, 3, 5, 5)); b = np.zeros(4)
    lum = np.array([1, 1, 1]) / 3
    cr = np.array([1, -0.5, -0.5]); cb = np.array([-0.5, -0.5, 1])
    W[0, :, 2, 2] = G_L * lum; b[0] = G_L * OFF
    g1 = np.array([1, 2, 1]) / 4
    k = np.full((5, 5), 1 / 25)
    k[1:4, 1:4] -= np.outer(g1, g1)
    for c in range(3):
        W[1, c] = G_I * k / 3
    W[2, :, 2, 2] = G_C * cr; b[2] = G_C * OFF
    W[3, :, 2, 2] = G_C * cb; b[3] = G_C * OFF
    return W, b


def chroma_conv():
    """1x1 linear conv giving [0, 0, Cr, Cb] (no offsets, no relu): used to restore absolute chroma."""
    W = np.zeros((4, 3, 1, 1)); b = np.zeros(4)
    W[2, :, 0, 0] = G_C * np.array([1, -0.5, -0.5])
    W[3, :, 0, 0] = G_C * np.array([-0.5, -0.5, 1])
    return W, b


# ============================================================== family masks and sample points
def tri_dist(x, y, apex_up, sv, scale):
    pts, _ = tri_pts(apex_up)
    pts = [(px * RC * scale, py * RC * scale * sv) for px, py in pts]
    ins = poly_inside(x, y, pts)
    d = poly_dist(x, y, pts, closed=True)
    return np.where(ins, 0.0, d)


def family_samples(fam):
    g = np.arange(-16, 16) + 0.5
    X, Y = np.meshgrid(g, g)
    x, y = X.ravel(), Y.ravel()
    big = max(SUBS)
    if fam == 'circle':
        ok = np.hypot(x, y) <= RC * big + MARGIN
    elif fam == 'diamond':
        s = RC * big
        # distance from point to the diamond |x|+|y| <= s
        l1 = np.abs(x) + np.abs(y)
        ok = (l1 <= s) | (np.minimum(poly_dist(x, y, [(0, -s), (s, 0), (0, s), (-s, 0)], closed=True), 99) <= MARGIN)
    else:
        up = fam == 'up'
        d = np.minimum.reduce([tri_dist(x, y, up, sv, big) for sv in ASPECTS[fam]])
        ok = d <= MARGIN
    x, y = x[ok], y[ok]
    m = len(x)
    n = int(np.ceil(np.sqrt(m)))
    return x, y, n


# ============================================================== resampling weights (photo -> canonical samples)
def resample_weights(x, y, n):
    """returns (P*n*n, 1, 32, 32): for each pose and sample, Gaussian weights over the photo pixels with the
    per-block mean row subtracted (so every channel of every block has zero mean); padding rows are zero."""
    m = len(x)
    N = n * n
    W = np.zeros((len(SCALES), N, 32, 32))
    Mrow = np.zeros((len(SCALES), 32, 32))
    ext = np.arange(-10, 42)
    idx = np.clip(ext, 0, 31)
    for p, d in enumerate(SCALES):
        k = d / DC
        X = 16 + x * k; Y = 16 + y * k
        s = SIG_R * k
        wx = np.exp(-((ext[None, :] + 0.5 - X[:, None]) ** 2) / (2 * s * s))
        wy = np.exp(-((ext[None, :] + 0.5 - Y[:, None]) ** 2) / (2 * s * s))
        wx[wx < 1e-4] = 0; wy[wy < 1e-4] = 0
        cx = np.zeros((m, 32)); cy = np.zeros((m, 32))
        for j in range(len(ext)):
            cx[:, idx[j]] += wx[:, j]; cy[:, idx[j]] += wy[:, j]
        cx /= cx.sum(1, keepdims=True); cy /= cy.sum(1, keepdims=True)
        w = cy[:, :, None] * cx[:, None, :]        # (m, 32 rows, 32 cols)
        Mrow[p] = w.mean(0)
        w -= w.mean(0, keepdims=True)
        W[p, :m] = w
    return W.reshape(len(SCALES) * N, 1, 32, 32), Mrow.reshape(len(SCALES), 1, 32, 32)


# ============================================================== templates
def blur_matrix(sig_fine):
    r = int(np.ceil(3 * sig_fine))
    B = np.zeros((NF, NF))
    for i in range(NF):
        for j in range(max(0, i - r), min(NF, i + r + 1)):
            B[i, j] = np.exp(-((i - j) ** 2) / (2 * sig_fine ** 2))
    return B / B.sum(1, keepdims=True)


def bilinear(img, x, y):
    """img (..., NF, NF) on grid FX; x, y canonical coords -> (..., len(x))"""
    fx = (x - FX[0]) / FINE; fy = (y - FX[0]) / FINE
    x0 = np.clip(np.floor(fx).astype(int), 0, NF - 2); y0 = np.clip(np.floor(fy).astype(int), 0, NF - 2)
    ax = np.clip(fx - x0, 0, 1); ay = np.clip(fy - y0, 0, 1)
    return (img[..., y0, x0] * (1 - ax) * (1 - ay) + img[..., y0, x0 + 1] * ax * (1 - ay)
            + img[..., y0 + 1, x0] * (1 - ax) * ay + img[..., y0 + 1, x0 + 1] * ax * ay)


def render_fine(cls, var, sub, sv, B):
    X, Y = np.meshgrid(FX, FX)
    cv = Canvas(X / (RC * sub), Y / (RC * sub * sv))
    draw(cv, cls, var)
    rgb = np.stack([B @ cv.rgb[c] @ B.T for c in range(3)])
    alpha = B @ cv.alpha @ B.T
    return feat_from_rgb_fine(rgb), alpha


def class_variants(cls):
    fam = FAMILY[cls]
    out = []
    for v in variants(cls):
        for sv in ASPECTS[fam]:
            out.append((v, sv))
    return out


def template_vec(F, A, x, y, dx, dy, absolute_chroma=True):
    """sample feature maps at the shifted points; L and Ink are made relative (alpha-weighted mean removed),
    chroma stays absolute; everything is weighted by the sign's coverage alpha."""
    f = bilinear(F, x - dx, y - dy)          # (4, m)
    a = bilinear(A, x - dx, y - dy)          # (m,)
    t = a * f
    for c in (0, 1):
        mu = (f[c] * a).sum() / max(a.sum(), 1e-9)
        t[c] = a * (f[c] - mu)
        t[c] -= t[c].mean()
    return t


def unit(v):
    return v / max(np.sqrt((v ** 2).sum()), 1e-12)


def interior_samples(win):
    """sample points of an interior (pictogram) window, in canonical px.
    win = ('disc', radius in sign radii) or ('tri', inradius in sign units of the true geometry)."""
    g = np.arange(-16, 16) + 0.5
    X, Y = np.meshgrid(g, g)
    x, y = X.ravel(), Y.ravel()
    if win[0] == 'disc':
        ok = np.hypot(x, y) <= win[1] * RC
    else:
        pts, cen = tri_pts(True, win[1])
        pts = [(px * RC, py * RC * 1.05) for px, py in pts]
        ok = poly_inside(x, y, pts)
    x, y = x[ok], y[ok]
    n = int(np.ceil(np.sqrt(len(x))))
    return x, y, n


def geo_variants(fam):
    """fine geometric variants (aspect, sub-scale, shift) shared by frames and pictograms."""
    return [(sv, sub, dx, dy) for sv in ASPECTS[fam] for sub in SUBS for dy in SHIFTS for dx in SHIFTS]


GROUPS = {   # group -> (family, frame class, frame variant, pictogram classes, empty classes, interior window)
    'RR': ('circle', 15, {}, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 16], [15], ('disc', 0.64)),
    'BL': ('circle', 33, {'frame': True}, [33, 34, 35, 36, 37, 38, 39, 40], [], ('disc', 0.73)),
    'EN': ('circle', 32, {}, [6, 41, 42], [32], ('disc', 0.73)),
    'TR': ('up', 18, {'frame': True}, [11] + list(range(18, 32)), [], ('tri', 0.30)),
}
SINGLES = {14: 'circle', 17: 'circle', 13: 'down', 12: 'diamond'}   # classes scored by one complete template


class Renders:
    def __init__(self):
        self.B = blur_matrix(SIG_T / FINE); self.cache = {}

    def get(self, cls, var, sub, sv):
        key = (cls, tuple(sorted(var.items())), sub, sv)
        if key not in self.cache:
            self.cache[key] = render_fine(cls, var, sub, sv, self.B)
        return self.cache[key]


def block_template(F, A, x, y, dx, dy, N):
    t = unit(template_vec(F, A, x, y, dx, dy))
    t[0] -= t.sum() / len(x)          # zero total sum: cancels the joint mean removed by norm
    out = np.zeros((N, 4)); out[:len(x)] = t.T
    return out / np.sqrt(4 * N)


def frame_bank(R, fam, fcls, fvar, x, y, n):
    N = n * n
    T = [block_template(*R.get(fcls, fvar, sub, sv), x, y, dx, dy, N) for sv, sub, dx, dy in geo_variants(fam)]
    return np.array(T).reshape(len(T), 1, 2 * n, 2 * n)


def pict_variants(cls):
    return variants(cls)


def pictogram_bank(R, fam, fcls, fvar, classes, xi, yi, ni):
    """pictogram templates on the interior window: (class render - frame render), relative features, unit norm."""
    N = ni * ni
    nv = max(len(pict_variants(c)) for c in classes)
    T = []
    for c in classes:
        vs = (pict_variants(c) * nv)[:nv]
        for v in vs:
            for sv, sub, dx, dy in geo_variants(fam):
                Fc, _ = R.get(c, v, sub, sv)
                Ff, _ = R.get(fcls, fvar, sub, sv)
                d = bilinear(Fc - Ff, xi - dx, yi - dy)        # (4, m)
                d -= d.mean(1, keepdims=True)
                d = unit(d)
                out = np.zeros((N, 4)); out[:len(xi)] = d.T
                T.append(BETA * out / np.sqrt(4 * N))
    return np.array(T).reshape(len(T), 1, 2 * ni, 2 * ni), nv


# ============================================================== network assembly
def build(out_dir):
    layers = []; weights = {}
    P = len(SCALES)

    def conv(name, inp, W, b, stride=1, padding=0):
        layers.append(dict(name=name, op='conv', inputs=[inp], out=int(W.shape[0]), kernel=[int(W.shape[2]), int(W.shape[3])],
                           stride=stride, padding=padding))
        weights[name + '.W'] = np.asarray(W, np.float32); weights[name + '.b'] = np.asarray(b, np.float32)

    def lay(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=inputs, **kw))

    def place(name, inp, classes, bias=0.0, sign=1.0):
        Wd = np.zeros((43, len(classes))); bd = np.zeros(43)
        for i, c in enumerate(classes):
            Wd[c, i] = sign; bd[c] = bias
        lay(name, 'dense', [inp], out=43)
        weights[name + '.W'] = Wd.astype(np.float32); weights[name + '.b'] = bd.astype(np.float32)
        outs.append(name)

    R = Renders()
    Wf, bf = feature_conv()
    conv('feat', 'image', Wf, bf, padding=2)
    lay('featr', 'relu', ['feat'])
    lay('stack', 'reshape', ['featr'], shape=[1, 128, 32])
    Wc, bc = chroma_conv()
    conv('chroma', 'image', Wc, bc)
    lay('cstack', 'reshape', ['chroma'], shape=[1, 128, 32])
    outs = []
    geo = {}
    for fam in FAMS:
        # ---- block window: L/Ink relative, chroma absolute, normalised per pose
        x, y, n = family_samples(fam)
        N = n * n
        Wr, Mr = resample_weights(x, y, n)
        conv(f'{fam}_rs', 'stack', Wr, np.zeros(P * N), stride=32)
        conv(f'{fam}_cm', 'cstack', Mr, np.zeros(P), stride=32)          # block mean of chroma, (P, 4, 1)
        Wb = np.zeros((P * N, P, 1, 1))
        for p in range(P):
            Wb[p * N:(p + 1) * N, p] = 1.0
        conv(f'{fam}_cb', f'{fam}_cm', Wb, np.zeros(P * N))              # broadcast to every sample
        lay(f'{fam}_abs', 'add', [f'{fam}_rs', f'{fam}_cb'])
        lay(f'{fam}_blk', 'reshape', [f'{fam}_abs'], shape=[P, 2 * n, 2 * n])
        lay(f'{fam}_nrm', 'norm', [f'{fam}_blk'])
        lay(f'{fam}_col', 'reshape', [f'{fam}_nrm'], shape=[1, P * 2 * n, 2 * n])
        geo[fam] = (x, y, n)
    for win in sorted(set(v[5] for v in GROUPS.values())):
        # ---- interior window: all features relative, normalised per pose
        wn = f'{win[0]}{int(round(win[1] * 100))}'
        xi, yi, ni = interior_samples(win)
        Wi, _ = resample_weights(xi, yi, ni)
        conv(f'{wn}_irs', 'stack', Wi, np.zeros(P * ni * ni), stride=32)
        lay(f'{wn}_iblk', 'reshape', [f'{wn}_irs'], shape=[P, 2 * ni, 2 * ni])
        lay(f'{wn}_inrm', 'norm', [f'{wn}_iblk'])
        lay(f'{wn}_icol', 'reshape', [f'{wn}_inrm'], shape=[1, P * 2 * ni, 2 * ni])
        geo[win] = (xi, yi, ni, wn)
    # ---- groups with pictograms: score = frame correlation + BETA * interior pictogram correlation, per pose
    def dense(name, inp, W, b):
        layers.append(dict(name=name, op='dense', inputs=[inp], out=int(W.shape[0])))
        weights[name + '.W'] = np.asarray(W, np.float32); weights[name + '.b'] = np.asarray(b, np.float32)

    def group_out(g, S, pcls, G, empties):
        """S: (1, M, 1) pictogram class scores, G: (1, 1, 1) best frame score.  Empty classes score G + BETA*THETA.
        The group's best score is capped at CAP * G by subtracting the same excess from every class of the group,
        so the order inside the group is kept while weak-frame groups cannot win on pictogram evidence alone."""
        allc = list(pcls) + list(empties)
        if pcls:
            place(f'{g}_out', S, pcls)
            lay(f'{g}_mx', 'maxpool', [S], size=[len(pcls), 1], stride=[len(pcls), 1])      # (1, 1, 1)
            lay(f'{g}_mxv', 'reshape', [f'{g}_mx'], shape=[1])
            top = f'{g}_mxv'
        if empties:
            dense(f'{g}_se', G, np.ones((1, 1)), np.full(1, BETA * THETA))                    # empty score (1,)
            for e in empties:
                place(f'{g}_eout{e}', f'{g}_se', [e])
            if pcls:
                dense(f'{g}_nt', top, -np.ones((1, 1)), np.zeros(1))
                lay(f'{g}_df', 'add', [f'{g}_se', f'{g}_nt'])
                lay(f'{g}_dfr', 'relu', [f'{g}_df'])
                lay(f'{g}_top', 'add', [top, f'{g}_dfr'])                                     # max(top, empty)
                top = f'{g}_top'
            else:
                top = f'{g}_se'
        dense(f'{g}_cG', G, np.full((1, 1), -CAP), np.zeros(1))
        lay(f'{g}_ex', 'add', [top, f'{g}_cG'])
        lay(f'{g}_exr', 'relu', [f'{g}_ex'])
        Wd = np.zeros((43, 1))
        for c in allc:
            Wd[c, 0] = -1.0
        dense(f'{g}_cut', f'{g}_exr', Wd, np.zeros(43))
        outs.append(f'{g}_cut')

    for g, (fam, fcls, fvar, pcls, empties, win) in GROUPS.items():
        x, y, n = geo[fam]; xi, yi, ni, wn = geo[win]
        Tf = frame_bank(R, fam, fcls, fvar, x, y, n)
        J = Tf.shape[0]
        conv(f'{g}_fr', f'{fam}_col', Tf, np.zeros(J), stride=2 * n)       # (J, P, 1)
        lay(f'{g}_frr', 'reshape', [f'{g}_fr'], shape=[1, J, P])
        lay(f'{g}_G', 'maxpool', [f'{g}_frr'], size=[J, P], stride=[J, P])  # best frame score (1, 1, 1)
        Tp, nv = pictogram_bank(R, fam, fcls, fvar, pcls, xi, yi, ni)
        MV = len(pcls) * nv
        conv(f'{g}_pi', f'{wn}_icol', Tp, np.zeros(Tp.shape[0]), stride=2 * ni)   # (MV*J, P, 1)
        lay(f'{g}_pir', 'reshape', [f'{g}_pi'], shape=[MV, J, P])
        conv(f'{g}_rep', f'{g}_frr', np.ones((MV, 1, 1, 1)), np.zeros(MV))         # frame score copied per class
        lay(f'{g}_sum', 'add', [f'{g}_rep', f'{g}_pir'])
        lay(f'{g}_mp', 'maxpool', [f'{g}_sum'], size=[J, P], stride=[J, P])        # (MV, 1, 1)
        lay(f'{g}_grp', 'reshape', [f'{g}_mp'], shape=[1, len(pcls), nv])
        lay(f'{g}_cls', 'maxpool', [f'{g}_grp'], size=[1, nv], stride=[1, nv])
        group_out(g, f'{g}_cls', pcls, f'{g}_G', empties)
    # ---- classes scored by a single complete template (stop, no entry, yield, priority road)
    for c, fam in SINGLES.items():
        x, y, n = geo[fam]
        T = frame_bank(R, fam, c, {}, x, y, n)
        J = T.shape[0]
        conv(f's{c}_tm', f'{fam}_col', T, np.zeros(J), stride=2 * n)
        lay(f's{c}_r', 'reshape', [f's{c}_tm'], shape=[1, J, P])
        lay(f's{c}_mp', 'maxpool', [f's{c}_r'], size=[J, P], stride=[J, P])
        group_out(f's{c}', None, [], f's{c}_mp', [c])
    lay('scores', 'add', outs)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'network.json'), 'w') as f:
        json.dump({'layers': layers}, f, indent=1)
    np.savez(os.path.join(out_dir, 'weights.npz'), **weights)


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
