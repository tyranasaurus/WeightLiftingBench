"""Tiny anti-aliased drawing library on float RGB canvases (values 0..1).
All shapes specified in canvas pixel coordinates. Anti-aliasing comes from
rendering at high resolution and area-downsampling later."""
import numpy as np

RED = (0.72, 0.08, 0.10)
WHITE = (0.95, 0.95, 0.95)
BLACK = (0.06, 0.06, 0.06)
BLUE = (0.05, 0.25, 0.60)
YELLOW = (0.95, 0.75, 0.05)
LGREY = (0.80, 0.80, 0.80)
DGREY = (0.30, 0.30, 0.30)
BG = (0.40, 0.42, 0.45)


class Canvas:
    def __init__(self, S, bg=BG):
        self.S = S
        y, x = np.mgrid[0:S, 0:S].astype(np.float64)
        self.X = x + 0.5
        self.Y = y + 0.5
        self.img = np.empty((S, S, 3))
        self.img[:] = bg

    def fill(self, mask, color):
        self.img[mask] = color

    def disc(self, cx, cy, r, color):
        self.fill((self.X - cx) ** 2 + (self.Y - cy) ** 2 <= r * r, color)

    def ring(self, cx, cy, r0, r1, color):
        d2 = (self.X - cx) ** 2 + (self.Y - cy) ** 2
        self.fill((d2 >= r0 * r0) & (d2 <= r1 * r1), color)

    def poly(self, pts, color):
        self.fill(poly_mask(self.X, self.Y, pts), color)

    def seg(self, p, q, w, color):
        self.fill(seg_mask(self.X, self.Y, p, q, w), color)

    def polyline(self, pts, w, color, close=False):
        m = np.zeros((self.S, self.S), bool)
        P = list(pts) + ([pts[0]] if close else [])
        for a, b in zip(P[:-1], P[1:]):
            m |= seg_mask(self.X, self.Y, a, b, w)
        for p in P:
            m |= (self.X - p[0]) ** 2 + (self.Y - p[1]) ** 2 <= (w / 2) ** 2
        self.fill(m, color)

    def ellipse_stroke(self, cx, cy, rx, ry, w, color, n=48):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]
        self.polyline(pts, w, color, close=True)

    def clip_disc(self, cx, cy, r):
        d2 = (self.X - cx) ** 2 + (self.Y - cy) ** 2
        self.img[d2 > r * r] = BG


def poly_mask(X, Y, pts):
    inside = np.zeros(X.shape, bool)
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if yi == yj:
            j = i
            continue
        cond = ((yi > Y) != (yj > Y)) & (X < (xj - xi) * (Y - yi) / (yj - yi) + xi)
        inside ^= cond
        j = i
    return inside


def seg_mask(X, Y, p, q, w):
    px, py = p
    qx, qy = q
    vx, vy = qx - px, qy - py
    L2 = vx * vx + vy * vy + 1e-12
    t = np.clip(((X - px) * vx + (Y - py) * vy) / L2, 0, 1)
    dx = X - (px + t * vx)
    dy = Y - (py + t * vy)
    return (dx * dx + dy * dy) <= (w / 2) ** 2


def downsample(img, k):
    """area-average an (S,S,3) image to (k,k,3)."""
    S = img.shape[0]
    # pad to multiple of k
    m = int(np.ceil(S / k)) * k
    if m != S:
        out = np.empty((m, m, 3))
        out[:] = BG
        out[:S, :S] = img
        img = out
        S = m
    f = S // k
    return img.reshape(k, f, k, f, 3).mean(axis=(1, 3))


"""Idealized renderings of the 43 German traffic sign classes, drawn from
general knowledge of the sign designs (StVO). No training data involved."""
import numpy as np

S = 160
C = S / 2.0
D = S / 1.2          # sign diameter (crop keeps ~10% border each side)
R = D / 2.0

DIGITS = {
    '0': [('ell', 0.5, 0.5, 0.34, 0.46)],
    '1': [('poly', [(0.30, 0.16), (0.56, 0.02), (0.56, 0.98)])],
    '2': [('poly', [(0.14, 0.22), (0.30, 0.04), (0.62, 0.02), (0.86, 0.18),
                    (0.84, 0.38), (0.16, 0.98), (0.88, 0.98)])],
    '3': [('poly', [(0.14, 0.10), (0.45, 0.02), (0.80, 0.14), (0.82, 0.34),
                    (0.55, 0.47), (0.82, 0.60), (0.84, 0.84), (0.50, 0.98),
                    (0.16, 0.90)])],
    '4': [('poly', [(0.62, 0.98), (0.62, 0.02), (0.12, 0.70), (0.88, 0.70)])],
    '5': [('poly', [(0.84, 0.04), (0.22, 0.04), (0.18, 0.44), (0.50, 0.38),
                    (0.80, 0.50), (0.86, 0.76), (0.62, 0.96), (0.24, 0.94)])],
    '6': [('poly', [(0.78, 0.04), (0.42, 0.16), (0.20, 0.46), (0.16, 0.74),
                    (0.36, 0.96), (0.66, 0.96), (0.84, 0.76), (0.80, 0.56),
                    (0.56, 0.46), (0.28, 0.52), (0.18, 0.66)])],
    '7': [('poly', [(0.14, 0.04), (0.86, 0.04), (0.42, 0.98)])],
    '8': [('ell', 0.5, 0.27, 0.28, 0.24), ('ell', 0.5, 0.72, 0.33, 0.25)],
    '9': [('ell', 0.5, 0.30, 0.31, 0.26), ('poly', [(0.81, 0.30), (0.81, 0.98)])],
}


def draw_digit(cv, ch, x0, y0, w, h, thick, color):
    for el in DIGITS[ch]:
        if el[0] == 'poly':
            pts = [(x0 + u * w, y0 + v * h) for (u, v) in el[1]]
            cv.polyline(pts, thick, color)
        else:
            _, cx, cy, rx, ry = el
            cv.ellipse_stroke(x0 + cx * w, y0 + cy * h, rx * w, ry * h, thick, color)


def head(cv, tip, dirv, size, color):
    dx, dy = dirv
    L = (dx * dx + dy * dy) ** 0.5
    dx, dy = dx / L, dy / L
    px, py = -dy, dx
    bx, by = tip[0] - dx * size, tip[1] - dy * size
    hw = size * 0.62
    cv.poly([(tip[0], tip[1]), (bx + px * hw, by + py * hw),
             (bx - px * hw, by - py * hw)], color)


def arrow(cv, pts, w, color, headsize=None):
    cv.polyline(pts, w, color)
    hs = headsize or w * 2.2
    (ax, ay), (bx, by) = pts[-1], pts[-2]
    head(cv, (ax, ay), (ax - bx, ay - by), hs, color)


def tri_up(cv, scale=1.0, color=RED):
    v = [(C, C - R), (C - R * 0.98, C + R * 0.74), (C + R * 0.98, C + R * 0.74)]
    cx = sum(p[0] for p in v) / 3
    cy = sum(p[1] for p in v) / 3
    pts = [(cx + (x - cx) * scale, cy + (y - cy) * scale) for (x, y) in v]
    cv.poly(pts, color)


def car_front(cv, cx, cy, w, color):
    h = w * 0.62
    cv.fill(seg_mask(cv.X, cv.Y, (cx - w / 2, cy), (cx + w / 2, cy), h * 0.62), color)
    cv.fill(seg_mask(cv.X, cv.Y, (cx - w * 0.34, cy - h * 0.42), (cx + w * 0.34, cy - h * 0.42), h * 0.5), color)
    for sx in (-0.34, 0.34):
        cv.disc(cx + sx * w, cy + h * 0.34, w * 0.11, color)


def truck_side(cv, cx, cy, w, color):
    h = w * 0.6
    cv.fill(seg_mask(cv.X, cv.Y, (cx - w / 2, cy - h * 0.1), (cx + w * 0.12, cy - h * 0.1), h * 0.7), color)
    cv.fill(seg_mask(cv.X, cv.Y, (cx + w * 0.3, cy), (cx + w / 2, cy), h * 0.5), color)
    for sx in (-0.32, 0.0, 0.36):
        cv.disc(cx + sx * w, cy + h * 0.4, w * 0.10, color)


def person(cv, cx, cy, h, color, walking=True):
    cv.disc(cx, cy - h * 0.38, h * 0.13, color)
    cv.seg((cx, cy - h * 0.25), (cx, cy + h * 0.12), h * 0.16, color)
    if walking:
        cv.seg((cx, cy + h * 0.10), (cx - h * 0.22, cy + h * 0.5), h * 0.13, color)
        cv.seg((cx, cy + h * 0.10), (cx + h * 0.20, cy + h * 0.5), h * 0.13, color)
        cv.seg((cx, cy - h * 0.15), (cx - h * 0.22, cy + h * 0.05), h * 0.11, color)
        cv.seg((cx, cy - h * 0.15), (cx + h * 0.22, cy - h * 0.02), h * 0.11, color)
    else:
        cv.seg((cx, cy + h * 0.10), (cx - h * 0.12, cy + h * 0.5), h * 0.13, color)
        cv.seg((cx, cy + h * 0.10), (cx + h * 0.12, cy + h * 0.5), h * 0.13, color)


def stripes(cv, cx, cy, r, color, n=5, thick=None, angle=np.pi / 4):
    thick = thick or r * 0.16
    dx, dy = np.cos(angle), np.sin(angle)
    px, py = -dy, dx
    for i in range(n):
        off = (i - (n - 1) / 2) * thick * 2.0
        a = (cx + px * off - dx * r * 1.5, cy + py * off - dy * r * 1.5)
        b = (cx + px * off + dx * r * 1.5, cy + py * off + dy * r * 1.5)
        m = seg_mask(cv.X, cv.Y, a, b, thick) & (((cv.X - cx) ** 2 + (cv.Y - cy) ** 2) <= r * r)
        cv.fill(m, color)


def render(cls):
    cv = Canvas(S)
    t = 0.28 * R          # red ring thickness
    Ri = R - t            # inner radius
    if cls <= 10 or cls in (15, 16):
        cv.disc(C, C, R, RED)
        cv.disc(C, C, Ri, WHITE)
    if cls in (0, 1, 2, 3, 4, 5, 7, 8):
        txt = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}[cls]
        n = len(txt)
        H = 1.40 * Ri if n == 2 else 1.15 * Ri
        if n == 2:
            w, gap = 0.60 * H, 0.10 * H
        else:
            w, gap = 0.44 * H, 0.06 * H
        total = n * w + (n - 1) * gap
        x = C - total / 2
        y0 = C - H / 2
        for ch in txt:
            draw_digit(cv, ch, x, y0, w, H, 0.18 * H, BLACK)
            x += w + gap
    elif cls == 9:
        car_front(cv, C - Ri * 0.42, C, Ri * 0.62, BLACK)
        car_front(cv, C + Ri * 0.42, C, Ri * 0.62, (0.70, 0.10, 0.10))
    elif cls == 10:
        truck_side(cv, C - Ri * 0.42, C, Ri * 0.72, BLACK)
        car_front(cv, C + Ri * 0.48, C, Ri * 0.5, BLACK)
    elif cls == 15:
        pass
    elif cls == 16:
        truck_side(cv, C, C, Ri * 1.0, BLACK)
    elif cls == 17:
        cv.disc(C, C, R, RED)
        cv.fill(seg_mask(cv.X, cv.Y, (C - R * 0.78, C), (C + R * 0.78, C), R * 0.34), WHITE)
    elif cls == 11:
        tri_up(cv, 1.0, RED)
        tri_up(cv, 0.68, WHITE)
        cv.seg((C, C + R * 0.42), (C, C - R * 0.28), R * 0.16, BLACK)
        cv.seg((C - R * 0.30, C - R * 0.02), (C + R * 0.30, C - R * 0.02), R * 0.13, BLACK)
        head(cv, (C, C - R * 0.42), (0, -1), R * 0.22, BLACK)
    elif cls == 12:
        cv.poly([(C, C - R), (C + R, C), (C, C + R), (C - R, C)], WHITE)
        cv.poly([(C, C - R * 0.78), (C + R * 0.78, C), (C, C + R * 0.78), (C - R * 0.78, C)], YELLOW)
    elif cls == 13:
        v = [(C, C + R), (C - R * 0.98, C - R * 0.74), (C + R * 0.98, C - R * 0.74)]
        cx = sum(p[0] for p in v) / 3
        cy = sum(p[1] for p in v) / 3
        cv.poly(v, RED)
        cv.poly([(cx + (x - cx) * 0.66, cy + (y - cy) * 0.66) for (x, y) in v], WHITE)
    elif cls == 14:
        oct_ = [(C + R * np.cos(a), C + R * np.sin(a)) for a in np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 8, endpoint=False)]
        cv.poly(oct_, WHITE)
        cv.poly([(C + (x - C) * 0.9, C + (y - C) * 0.9) for (x, y) in oct_], RED)
        # STOP text
        w, h, th = R * 0.24, R * 0.32, R * 0.09
        xs = C - 1.95 * w
        for ch in 'STOP':
            x = xs
            if ch == 'S':
                cv.polyline([(x + w, C - h), (x, C - h), (x, C - h * 0.1), (x + w, C + h * 0.1), (x + w, C + h), (x, C + h)], th, WHITE)
            elif ch == 'T':
                cv.polyline([(x, C - h), (x + w, C - h)], th, WHITE)
                cv.polyline([(x + w / 2, C - h), (x + w / 2, C + h)], th, WHITE)
            elif ch == 'O':
                cv.ellipse_stroke(x + w / 2, C, w / 2, h, th, WHITE)
            else:
                cv.polyline([(x, C + h), (x, C - h), (x + w, C)], th, WHITE)
            xs += w * 1.35
    elif cls in (6, 32, 41, 42):
        cv.disc(C, C, R, LGREY)
        cv.disc(C, C, R * 0.97, (0.88, 0.88, 0.88))
        stripes(cv, C, C, R * 0.99, (0.45, 0.45, 0.45), n=4, thick=R * 0.11)
        if cls == 6:
            H = 1.25 * R
            w, gap = 0.60 * H, 0.10 * H
            x = C - (2 * w + gap) / 2
            for ch in '80':
                draw_digit(cv, ch, x, C - H / 2, w, H, 0.17 * H, (0.15, 0.15, 0.15))
                x += w + gap
        if cls == 41:
            car_front(cv, C - R * 0.40, C, R * 0.55, (0.15, 0.15, 0.15))
            car_front(cv, C + R * 0.40, C, R * 0.55, (0.15, 0.15, 0.15))
        if cls == 42:
            truck_side(cv, C - R * 0.40, C, R * 0.62, (0.15, 0.15, 0.15))
            car_front(cv, C + R * 0.45, C, R * 0.45, (0.15, 0.15, 0.15))
    elif cls >= 33 and cls <= 40:
        cv.disc(C, C, R, BLUE)
        w = R * 0.22
        if cls == 35:
            arrow(cv, [(C, C + R * 0.55), (C, C - R * 0.30)], w, WHITE)
        elif cls == 33:
            arrow(cv, [(C - R * 0.12, C + R * 0.55), (C - R * 0.12, C - R * 0.18), (C + R * 0.30, C - R * 0.18)], w, WHITE)
        elif cls == 34:
            arrow(cv, [(C + R * 0.12, C + R * 0.55), (C + R * 0.12, C - R * 0.18), (C - R * 0.30, C - R * 0.18)], w, WHITE)
        elif cls == 36:
            arrow(cv, [(C - R * 0.16, C + R * 0.55), (C - R * 0.16, C - R * 0.30)], w, WHITE)
            arrow(cv, [(C - R * 0.16, C + R * 0.05), (C + R * 0.34, C + R * 0.05)], w, WHITE)
        elif cls == 37:
            arrow(cv, [(C + R * 0.16, C + R * 0.55), (C + R * 0.16, C - R * 0.30)], w, WHITE)
            arrow(cv, [(C + R * 0.16, C + R * 0.05), (C - R * 0.34, C + R * 0.05)], w, WHITE)
        elif cls == 38:
            arrow(cv, [(C - R * 0.38, C - R * 0.38), (C + R * 0.28, C + R * 0.28)], w, WHITE)
        elif cls == 39:
            arrow(cv, [(C + R * 0.38, C - R * 0.38), (C - R * 0.28, C + R * 0.28)], w, WHITE)
        elif cls == 40:
            rr = R * 0.45
            for k in range(3):
                a0 = k * 2 * np.pi / 3
                pts = [(C + rr * np.cos(a0 + u), C + rr * np.sin(a0 + u)) for u in np.linspace(0.3, 1.7, 8)]
                cv.polyline(pts, w * 0.8, WHITE)
                (ex, ey), (px, py) = pts[-1], pts[-2]
                head(cv, (ex, ey), (ex - px, ey - py), w * 1.8, WHITE)
    else:
        # warning triangles 18..31
        tri_up(cv, 1.0, RED)
        tri_up(cv, 0.70, WHITE)
        s = R * 0.5
        cy = C + R * 0.16
        if cls == 18:
            cv.seg((C, cy - s * 0.75), (C, cy + s * 0.25), s * 0.28, BLACK)
            cv.disc(C, cy + s * 0.65, s * 0.16, BLACK)
        elif cls in (19, 20):
            sgn = 1 if cls == 19 else -1
            pts = [(C + sgn * s * 0.45, cy + s * 0.7), (C + sgn * s * 0.45, cy - s * 0.15), (C - sgn * s * 0.45, cy - s * 0.15)]
            arrow(cv, pts, s * 0.30, BLACK)
        elif cls == 21:
            pts = [(C + s * 0.45, cy + s * 0.75), (C + s * 0.45, cy + s * 0.15), (C - s * 0.45, cy + s * 0.15), (C - s * 0.45, cy - s * 0.45), (C + s * 0.2, cy - s * 0.45)]
            arrow(cv, pts, s * 0.26, BLACK)
        elif cls == 22:
            cv.polyline([(C - s * 0.8, cy + s * 0.4), (C - s * 0.4, cy - s * 0.3), (C, cy + s * 0.4), (C + s * 0.4, cy - s * 0.3), (C + s * 0.8, cy + s * 0.4)], s * 0.26, BLACK)
        elif cls == 23:
            cv.fill(seg_mask(cv.X, cv.Y, (C - s * 0.5, cy - s * 0.1), (C + s * 0.3, cy - s * 0.1), s * 0.4), BLACK)
            cv.fill(seg_mask(cv.X, cv.Y, (C + s * 0.35, cy - s * 0.05), (C + s * 0.6, cy - s * 0.05), s * 0.3), BLACK)
            cv.polyline([(C - s * 0.7, cy + s * 0.45), (C - s * 0.4, cy + s * 0.3), (C - s * 0.1, cy + s * 0.45)], s * 0.12, BLACK)
            cv.polyline([(C + s * 0.1, cy + s * 0.45), (C + s * 0.4, cy + s * 0.3), (C + s * 0.7, cy + s * 0.45)], s * 0.12, BLACK)
        elif cls == 24:
            cv.seg((C - s * 0.45, cy - s * 0.7), (C - s * 0.45, cy + s * 0.7), s * 0.22, BLACK)
            cv.polyline([(C + s * 0.45, cy + s * 0.7), (C + s * 0.45, cy + s * 0.0), (C + s * 0.05, cy - s * 0.7)], s * 0.22, BLACK)
        elif cls == 25:
            cv.disc(C - s * 0.1, cy - s * 0.55, s * 0.16, BLACK)
            cv.seg((C - s * 0.1, cy - s * 0.4), (C + s * 0.15, cy + s * 0.05), s * 0.22, BLACK)
            cv.seg((C + s * 0.15, cy + s * 0.05), (C + s * 0.05, cy + s * 0.6), s * 0.18, BLACK)
            cv.seg((C + s * 0.15, cy + s * 0.05), (C + s * 0.45, cy + s * 0.45), s * 0.18, BLACK)
            cv.seg((C - s * 0.5, cy - s * 0.1), (C + s * 0.6, cy + s * 0.6), s * 0.12, BLACK)
        elif cls == 26:
            cv.fill(seg_mask(cv.X, cv.Y, (C, cy - s * 0.6), (C, cy + s * 0.6), s * 0.55), BLACK)
            cv.disc(C, cy - s * 0.38, s * 0.16, (0.8, 0.1, 0.1))
            cv.disc(C, cy, s * 0.16, (0.9, 0.7, 0.1))
            cv.disc(C, cy + s * 0.38, s * 0.16, (0.1, 0.7, 0.2))
        elif cls == 27:
            person(cv, C, cy, s * 1.5, BLACK)
        elif cls == 28:
            person(cv, C - s * 0.35, cy + s * 0.1, s * 1.1, BLACK)
            person(cv, C + s * 0.35, cy + s * 0.05, s * 1.0, BLACK)
        elif cls == 29:
            for sx in (-0.45, 0.45):
                cv.ellipse_stroke(C + sx * s, cy + s * 0.35, s * 0.28, s * 0.28, s * 0.10, BLACK)
            cv.polyline([(C - s * 0.45, cy + s * 0.35), (C - s * 0.1, cy - s * 0.2), (C + s * 0.25, cy - s * 0.2), (C + s * 0.45, cy + s * 0.35), (C - s * 0.1, cy + s * 0.35)], s * 0.10, BLACK)
            cv.seg((C - s * 0.1, cy - s * 0.2), (C - s * 0.2, cy - s * 0.5), s * 0.10, BLACK)
        elif cls == 30:
            for k in range(3):
                a = k * np.pi / 3
                cv.seg((C - s * 0.7 * np.cos(a), cy - s * 0.7 * np.sin(a)), (C + s * 0.7 * np.cos(a), cy + s * 0.7 * np.sin(a)), s * 0.14, BLACK)
        elif cls == 31:
            cv.fill(seg_mask(cv.X, cv.Y, (C - s * 0.4, cy), (C + s * 0.3, cy - s * 0.1), s * 0.4), BLACK)
            cv.seg((C + s * 0.3, cy - s * 0.2), (C + s * 0.55, cy - s * 0.6), s * 0.2, BLACK)
            cv.seg((C - s * 0.35, cy + s * 0.1), (C - s * 0.6, cy + s * 0.6), s * 0.12, BLACK)
            cv.seg((C - s * 0.1, cy + s * 0.15), (C - s * 0.2, cy + s * 0.65), s * 0.12, BLACK)
            cv.seg((C + s * 0.2, cy + s * 0.1), (C + s * 0.35, cy + s * 0.6), s * 0.12, BLACK)
    return cv.img


def interior_mask(cls):
    """0/1 map of the symbol-bearing interior of each sign (outline excluded)."""
    cv = Canvas(S, bg=(0.0, 0.0, 0.0))
    t = 0.28 * R
    Ri = R - t
    if cls <= 10 or cls in (15, 16):
        cv.disc(C, C, Ri * 0.98, (1, 1, 1))
    elif cls == 17 or cls in (6, 32, 41, 42) or (33 <= cls <= 40):
        cv.disc(C, C, R * 0.95, (1, 1, 1))
    elif cls in (11,) or (18 <= cls <= 31):
        tri_up(cv, 0.60, (1, 1, 1))
    elif cls == 13:
        v = [(C, C + R), (C - R * 0.98, C - R * 0.74), (C + R * 0.98, C - R * 0.74)]
        cx = sum(p[0] for p in v) / 3
        cy = sum(p[1] for p in v) / 3
        cv.poly([(cx + (x - cx) * 0.60, cy + (y - cy) * 0.60) for (x, y) in v], (1, 1, 1))
    elif cls == 14:
        oct_ = [(C + R * np.cos(a), C + R * np.sin(a)) for a in np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 8, endpoint=False)]
        cv.poly([(C + (x - C) * 0.80, C + (y - C) * 0.80) for (x, y) in oct_], (1, 1, 1))
    elif cls == 12:
        # whole diamond: the yellow/white-border contrast is the identity here
        cv.poly([(C, C - R), (C + R, C), (C, C + R), (C - R, C)], (1, 1, 1))
    return cv.img[:, :, 0]


def support_mask(cls):
    """0/1 silhouette of the sign plate itself (photo background excluded)."""
    cv = Canvas(S, bg=(0.0, 0.0, 0.0))
    if cls <= 17 or (32 <= cls <= 42):
        cv.disc(C, C, R, (1, 1, 1))
    elif cls in (11,) or (18 <= cls <= 31):
        tri_up(cv, 1.0, (1, 1, 1))
    elif cls == 13:
        cv.poly([(C, C + R), (C - R * 0.98, C - R * 0.74), (C + R * 0.98, C - R * 0.74)], (1, 1, 1))
    elif cls == 14:
        oct_ = [(C + R * np.cos(a), C + R * np.sin(a)) for a in np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 8, endpoint=False)]
        cv.poly(oct_, (1, 1, 1))
    elif cls == 12:
        cv.poly([(C, C - R), (C + R, C), (C, C + R), (C - R, C)], (1, 1, 1))
    return cv.img[:, :, 0]


def render_all():
    return {c: render(c) for c in range(43)}


import json
import os
import sys

import numpy as np

SCALES = [16, 20, 24, 28, 32]   # template windows; sign diameter = k/1.2, covering
                                # the 13..27 px range the crops produce


def fe_weights():
    g = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8
    ky = kx.T
    d45 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], float) / 4
    d135 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], float) / 4
    cols = [(1 / 3, 1 / 3, 1 / 3), (1, -.5, -.5), (-.5, -.5, 1), (.5, .5, -1), (-.5, 1, -.5)]
    W = np.zeros((9, 3, 3, 3))
    for i, c in enumerate(cols):
        for ch in range(3):
            W[i, ch] = g * c[ch]
    for i, k in enumerate([kx, ky, d45, d135]):
        for ch in range(3):
            W[5 + i, ch] = k * (1 / 3)
    return W, np.zeros(9)


def conv3_pad1(x, W, b):
    """x (H,W,C) -> (H,W,O), 3x3 pad 1."""
    H, Wd, C = x.shape
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    O = W.shape[0]
    out = np.zeros((H, Wd, O))
    for o in range(O):
        acc = np.zeros((H, Wd))
        for i in range(C):
            for dy in range(3):
                for dx in range(3):
                    acc += W[o, i, dy, dx] * xp[dy:dy + H, dx:dx + Wd, i]
        out[:, :, o] = acc + b[o]
    return out


def per_channel_norm(f):
    mu = f.mean(axis=(0, 1), keepdims=True)
    var = ((f - mu) ** 2).mean(axis=(0, 1), keepdims=True)
    return (f - mu) / np.sqrt(var + 1e-5)


def masked_norm(f, m):
    """per-channel mean/var over masked positions only; zero outside mask."""
    w = m[:, :, None]
    tot = w.sum() + 1e-9
    mu = (f * w).sum(axis=(0, 1), keepdims=True) / tot
    var = (((f - mu) ** 2) * w).sum(axis=(0, 1), keepdims=True) / tot
    return (f - mu) * w / np.sqrt(var + 1e-5)


W_FULL = 1.0     # gain of whole-sign templates
W_INT = 2.5      # gain of interior (symbol) templates: the symbol carries identity


def gauss_blur(f, sigma):
    """separable gaussian blur of a (C,H,W) array, reflect-free zero pad."""
    if sigma <= 0:
        return f
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=float)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    C, H, W = f.shape

    def shift(a, sh, axis):
        p = [(0, 0)] * 3
        p[axis] = (max(sh, 0), max(-sh, 0))
        a = np.pad(a, p)
        sl = [slice(None)] * 3
        if sh >= 0:
            sl[axis] = slice(0, a.shape[axis] - sh) if sh else slice(None)
        else:
            sl[axis] = slice(-sh, a.shape[axis])
        return a[tuple(sl)]

    out = np.zeros_like(f)
    for i, gv in enumerate(g):
        out += gv * shift(f, i - r, 1)
    f2 = out
    out = np.zeros_like(f2)
    for i, gv in enumerate(g):
        out += gv * shift(f2, i - r, 2)
    return out


def make_template(feat, m=None, sigma=0.0):
    """One generated sign image -> one template filter.
    Per-channel mean removed over the (masked) support, then a SINGLE scalar
    (joint L2) normalisation, so channels that carry no structure inside the
    support stay at ~0 instead of being blown up into noise."""
    if m is None:
        w = np.ones_like(feat)
        empty = False
    else:
        # erode 1 px so the outline/anti-alias annulus stays out of the support
        me = (m > 0.9)
        er = (me[1:-1, 1:-1] & me[:-2, 1:-1] & me[2:, 1:-1] & me[1:-1, :-2] & me[1:-1, 2:])
        er = np.pad(er, 1)
        if er.sum() == 0:
            er = me
        m = m * er
        w = m[:, :, None]
        tot = w.sum() + 1e-9
        mu0 = (feat[:, :, 0] * m).sum() / tot
        var0 = (((feat[:, :, 0] - mu0) ** 2) * m).sum() / tot
        empty = var0 < 1e-4
    tot = w.sum() + 1e-9
    mu = (feat * w).sum(axis=(0, 1), keepdims=True) / tot
    var = (((feat - mu) ** 2) * w).sum(axis=(0, 1), keepdims=True) / tot
    # per-channel equalisation, but with the variance floored at 10% of the
    # strongest channel's, so channels with no structure in the support stay
    # at ~0 instead of being amplified into noise
    scale = np.sqrt(var + 0.1 * var.max() + 1e-6)
    f = (feat - mu) * w / scale
    f = f.transpose(2, 0, 1)
    nrm = np.sqrt((f ** 2).sum())
    if empty or nrm < 1e-7 * f.size:
        # empty interior (e.g. "no vehicles"): a uniform-brightness detector,
        # so any dark symbol ink actively lowers this class's score.
        f = np.zeros_like(f)
        f[0] = m if m is not None else np.ones(f.shape[1:])
        nrm = np.sqrt((f ** 2).sum())
    f = gauss_blur(f / nrm, sigma)
    return f / (np.sqrt((f ** 2).sum()) + 1e-12)


def build(out_dir):
    FE_W, FE_B = fe_weights()
    full, interior = {}, {}
    for cls in range(43):
        hi = render(cls)
        mk = interior_mask(cls)
        sk = support_mask(cls)
        for k in SCALES:
            small = downsample(hi, k)                      # (k,k,3)
            feat = conv3_pad1(small, FE_W, FE_B)           # (k,k,9)
            sigma = 1.0 * k / 32.0
            full[(cls, k)] = make_template(feat, downsample(np.stack([sk] * 3, -1), k)[:, :, 0], sigma) * W_FULL
            # interiors get a coarser blur: the ink *layout* (not pixel-exact
            # strokes) is what survives blur, dark light and 1 px misalignment
            interior[(cls, k)] = make_template(feat, downsample(np.stack([mk] * 3, -1), k)[:, :, 0], 2.0 * k / 32.0) * W_INT

    layers = [{"name": "fe", "op": "conv", "inputs": ["image"], "out": 9,
               "kernel": [3, 3], "stride": 1, "padding": 1},
              {"name": "nfe", "op": "norm", "inputs": ["fe"]}]
    weights = {"fe.W": FE_W.astype(np.float32), "fe.b": FE_B.astype(np.float32)}
    pool_names = []
    for k in SCALES:
        P = 32 - k + 1
        for tag, store in (("t", full), ("i", interior)):
            name = f"{tag}{k}"
            W = np.stack([store[(c, k)] for c in range(43)]).astype(np.float32)
            layers.append({"name": name, "op": "conv", "inputs": ["nfe"], "out": 43,
                           "kernel": [k, k], "stride": 1, "padding": 0})
            layers.append({"name": name + "p", "op": "maxpool", "inputs": [name],
                           "size": [P, P], "stride": [P, P]})
            weights[name + ".W"] = W
            weights[name + ".b"] = np.zeros(43, np.float32)
            pool_names.append(name + "p")
    layers.append({"name": "sum", "op": "add", "inputs": pool_names})
    layers.append({"name": "scores", "op": "reshape", "inputs": ["sum"], "shape": [43]})

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
