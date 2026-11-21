"""Procedural idealized German traffic sign renderer (general knowledge only).
Renders in a 32x32 coordinate frame at 4x supersampling, returns (3,32,32) float 0..1.
No training data involved: every shape/colour is typed from knowledge of the sign designs.
"""
import numpy as np
import sys as _sys
signs = _sys.modules[__name__]  # self-reference: the sign renderer lives in this same file

S = 4  # supersample
N = 32

# palette (typical sign colours)
RED = (0.72, 0.07, 0.10)
BLUE = (0.05, 0.24, 0.60)
YELLOW = (0.95, 0.72, 0.06)
WHITE = (0.93, 0.93, 0.90)
BLACK = (0.09, 0.09, 0.09)
GRAY = (0.48, 0.48, 0.48)
DGRAY = (0.28, 0.28, 0.28)
GREEN = (0.10, 0.55, 0.20)
BG = (0.45, 0.45, 0.45)


class Canvas:
    def __init__(self, bg=BG, warp=None):
        self.a = np.zeros((N * S, N * S, 3))
        self.a[:] = bg
        self.painted = np.zeros((N * S, N * S), bool)
        yy, xx = np.mgrid[0:N * S, 0:N * S]
        px, py = (xx + 0.5) / S, (yy + 0.5) / S
        if warp is None:
            self.X, self.Y = px, py
        else:
            sx, sy, rot, tx, ty = warp
            u, v = px - 16 - tx, py - 16 - ty
            c, s = np.cos(-rot), np.sin(-rot)
            u, v = c * u - s * v, s * u + c * v
            self.X, self.Y = 16 + u / sx, 16 + v / sy

    def fill(self, mask, color):
        self.a[mask] = color
        self.painted |= mask

    @property
    def painted3(self):
        m = self.painted.reshape(N, S, N, S).any((1, 3))
        return m[None, None]

    def img(self, blur=0.0):
        a = self.a.reshape(N, S, N, S, 3).mean((1, 3)).transpose(2, 0, 1)
        if blur > 0:
            k = int(np.ceil(blur * 3)) * 2 + 1
            g = np.exp(-0.5 * (np.arange(k) - k // 2) ** 2 / blur ** 2)
            g /= g.sum()
            for c in range(3):
                a[c] = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 1, a[c])
                a[c] = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 0, a[c])
        return np.clip(a, 0, 1)

    # ---- primitives ----
    def disc(self, cx, cy, r, color):
        self.fill((self.X - cx) ** 2 + (self.Y - cy) ** 2 <= r * r, color)

    def ring(self, cx, cy, r1, r2, color):
        d2 = (self.X - cx) ** 2 + (self.Y - cy) ** 2
        self.fill((d2 >= r1 * r1) & (d2 <= r2 * r2), color)

    def convexpoly(self, pts, color):
        """pts: list of (x,y), any winding; fill via half planes."""
        pts = list(pts)
        area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                   for i in range(len(pts)))
        if area > 0:
            pts = pts[::-1]
        m = np.ones_like(self.X, bool)
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            # left side of edge (CCW) is inside
            m &= ((self.X - x1) * (y2 - y1) - (self.Y - y1) * (x2 - x1)) >= 0
        self.fill(m, color)

    def seg(self, p1, p2, t, color):
        (x1, y1), (x2, y2) = p1, p2
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy + 1e-9
        tt = ((self.X - x1) * dx + (self.Y - y1) * dy) / L2
        tt = np.clip(tt, 0, 1)
        px, py = x1 + tt * dx, y1 + tt * dy
        self.fill((self.X - px) ** 2 + (self.Y - py) ** 2 <= (t / 2) ** 2, color)

    def poly(self, pts, t, color, close=False):
        pts = list(pts)
        if close:
            pts = pts + [pts[0]]
        for i in range(len(pts) - 1):
            self.seg(pts[i], pts[i + 1], t, color)

    def arc(self, cx, cy, r, a0, a1, t, color):
        """angles in degrees, screen coords (y down); drawn from a0 to a1."""
        d = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
        ang = np.degrees(np.arctan2(self.Y - cy, self.X - cx))
        m = (d >= r - t / 2) & (d <= r + t / 2)
        delta = (a1 - a0) % 360
        m &= ((ang - a0) % 360) <= delta
        self.fill(m, color)

    def rect(self, x0, y0, x1, y1, color):
        self.fill((self.X >= x0) & (self.X <= x1) & (self.Y >= y0) & (self.Y <= y1), color)


# ---------- digits (DIN-like stroke font), unit box (0..w, 0..1) ----------
def _arcpts(cx, cy, rx, ry, a0, a1, n=10):
    return [(cx + rx * np.cos(np.radians(a)), cy + ry * np.sin(np.radians(a)))
            for a in np.linspace(a0, a1, n)]


DIGITS = {
    "0": lambda: _arcpts(0.5, 0.5, 0.34, 0.48, -90, 270, 14) + [_arcpts(0.5, 0.5, 0.34, 0.48, -90, 270, 14)[0]],
    "1": lambda: [(0.32, 0.26), (0.58, 0.06), (0.58, 1.0)],
    "2": lambda: _arcpts(0.5, 0.3, 0.32, 0.28, 190, 355, 9) + [(0.8, 0.45), (0.2, 1.0), (0.82, 1.0)],
    "3": lambda: _arcpts(0.48, 0.27, 0.32, 0.25, 185, 375, 9) + [(0.52, 0.5)] + _arcpts(0.48, 0.73, 0.34, 0.26, 335, 555, 9),
    "5": lambda: [(0.8, 0.06), (0.24, 0.06), (0.22, 0.46), (0.52, 0.44)] + _arcpts(0.5, 0.71, 0.32, 0.27, 330, 560, 9),
    "6": lambda: [(0.72, 0.04), (0.42, 0.3), (0.24, 0.62)] + _arcpts(0.5, 0.72, 0.28, 0.26, 180, -180, 12),
    "7": lambda: [(0.2, 0.06), (0.84, 0.06), (0.42, 1.0)],
    "8": lambda: _arcpts(0.5, 0.27, 0.28, 0.24, -90, 270, 12) + [_arcpts(0.5, 0.27, 0.28, 0.24, -90, 270, 12)[0]] +
         _arcpts(0.5, 0.74, 0.32, 0.25, -90, 270, 12) + [_arcpts(0.5, 0.74, 0.32, 0.25, -90, 270, 12)[0]],
    "9": lambda: _arcpts(0.5, 0.28, 0.28, 0.26, 180, -180, 12) + [(0.78, 0.4), (0.7, 1.0)],
}
DIG_W = 0.62
DIG_H = 10.6
DIG_T = 1.7


def draw_digits(cv, text, cx=16.0, cy=16.6, h=DIG_H, color=BLACK, t=DIG_T):
    n = len(text)
    w = h * DIG_W
    gap = h * 0.14
    total = n * w + (n - 1) * gap
    x0 = cx - total / 2
    for i, ch in enumerate(text):
        pts = DIGITS[ch]()
        ox, oy = x0 + i * (w + gap), cy - h / 2
        P = [(ox + px * w, oy + py * h) for px, py in pts]
        cv.poly(P, t, color)


def arrow_head(cv, tip, back_c, half_w, t, color):
    """triangle head: tip point, back centre, half width; back perpendicular auto."""
    (tx, ty), (bx, by) = tip, back_c
    dx, dy = tx - bx, ty - by
    L = np.hypot(dx, dy) + 1e-9
    px, py = -dy / L, dx / L
    cv.convexpoly([(bx + px * half_w, by + py * half_w), (tx, ty), (bx - px * half_w, by - py * half_w)], color)


# ---------- sign families ----------
def circle_base(cv, r=13.0, ring_t=2.7):
    cv.disc(16, 16, r, WHITE)
    cv.ring(16, 16, r - ring_t, r, RED)


def triangle_up(cv, half=13.0, top=4.6, bot=27.4, border=3.0):
    pts = [(16, top), (16 + half, bot), (16 - half, bot)]
    cv.convexpoly(pts, RED)
    f = 1 - border / ((bot - top) * 0.72)
    cyy = (top + bot + bot) / 3
    inner = [(16 + (x - 16) * f, cyy + (y - cyy) * f) for x, y in pts]
    cv.convexpoly(inner, WHITE)


def triangle_down(cv, half=13.0, top=5.0, bot=27.0, border=3.2):
    pts = [(16 - half, top), (16 + half, top), (16, bot)]
    cv.convexpoly(pts, RED)
    f = 1 - border / ((bot - top) * 0.72)
    cyy = (top + top + bot) / 3
    inner = [(16 + (x - 16) * f, cyy + (y - cyy) * f) for x, y in pts]
    cv.convexpoly(inner, WHITE)


def diamond(cv, r=13.2, border=2.6):
    cv.convexpoly([(16, 16 - r), (16 + r, 16), (16, 16 + r), (16 - r, 16)], WHITE)
    ri = r - border
    cv.convexpoly([(16, 16 - ri), (16 + ri, 16), (16, 16 + ri), (16 - ri, 16)], YELLOW)


def octagon(cv, r=13.2):
    c = r * np.cos(np.radians(22.5))
    pts = [(16 + c * np.cos(np.radians(a)), 16 + c * np.sin(np.radians(a))) for a in range(22, 361, 45)]
    cv.convexpoly(pts, RED)


def blue_disc(cv, r=13.0):
    cv.disc(16, 16, r, BLUE)


# ---------- pictograms (warning triangle interiors) ----------
def pic_excl(cv):
    cv.seg((16, 12.6), (16, 19.6), 2.3, BLACK)
    cv.disc(16, 22.4, 1.35, BLACK)


def pic_curve(cv, direction):
    s = 1 if direction == "right" else -1
    cv.poly([(16 - 1.6 * s, 23.2), (16 - 1.6 * s, 18.0), (16 + 2.6 * s, 15.6)], 2.3, BLACK)
    arrow_head(cv, (16 + 4.6 * s, 14.6), (16 + 1.6 * s, 15.9), 2.0, 1.0, BLACK)


def pic_double(cv):
    cv.poly([(13.2, 23.2), (15.2, 23.2), (15.2, 19.4), (17.4, 19.4), (17.4, 15.4)], 2.1, BLACK)
    arrow_head(cv, (17.4, 12.9), (17.4, 15.8), 1.9, 1.0, BLACK)


def pic_bump(cv):
    cv.arc(13.2, 22.6, 2.5, 180, 360, 4.6, BLACK)
    cv.arc(18.8, 22.6, 2.5, 180, 360, 4.6, BLACK)


def _car(cv, cx, cy, L, color, cab=True):
    cv.rect(cx - L / 2, cy - L * 0.16, cx + L / 2, cy + L * 0.16, color)
    if cab:
        cv.rect(cx - L * 0.28, cy - L * 0.36, cx + L * 0.18, cy - L * 0.1, color)
    cv.disc(cx - L * 0.3, cy + L * 0.18, L * 0.11, color)
    cv.disc(cx + L * 0.3, cy + L * 0.18, L * 0.11, color)


def pic_slip(cv):
    _car(cv, 16.2, 16.0, 8.6, BLACK)
    cv.poly([(11.5, 20.4), (13.2, 21.4), (11.8, 22.6)], 1.5, BLACK)
    cv.poly([(15.5, 21.4), (17.2, 22.4), (15.8, 23.6)], 1.5, BLACK)


def pic_narrow(cv):
    cv.seg((13.6, 13.6), (13.6, 23.0), 2.0, BLACK)
    cv.poly([(18.6, 23.0), (18.6, 18.4), (16.4, 13.6)], 2.0, BLACK)


def pic_work(cv):
    cv.disc(14.6, 13.6, 1.25, BLACK)
    cv.poly([(14.6, 15.0), (16.2, 18.6)], 2.0, BLACK)
    cv.poly([(16.2, 18.6), (14.4, 23.0)], 1.7, BLACK)
    cv.poly([(16.2, 18.6), (18.2, 23.0)], 1.7, BLACK)
    cv.poly([(15.2, 16.4), (19.0, 17.8)], 1.5, BLACK)
    cv.seg((20.2, 14.2), (17.6, 22.4), 1.3, BLACK)
    cv.rect(16.9, 21.6, 19.0, 23.0, BLACK)


def pic_tlight(cv):
    cv.rect(14.2, 11.8, 17.8, 22.6, BLACK)
    cv.disc(16, 14.2, 1.3, RED)
    cv.disc(16, 17.2, 1.3, YELLOW)
    cv.disc(16, 20.2, 1.3, GREEN)


def pic_ped(cv):
    cv.disc(15.6, 13.2, 1.25, BLACK)
    cv.seg((15.6, 14.8), (16.0, 19.0), 2.0, BLACK)
    cv.poly([(16.0, 19.0), (13.6, 23.4)], 1.6, BLACK)
    cv.poly([(16.0, 19.0), (18.2, 23.4)], 1.6, BLACK)
    cv.poly([(15.8, 15.8), (18.2, 17.6)], 1.4, BLACK)
    cv.poly([(15.8, 15.8), (13.6, 18.2)], 1.4, BLACK)


def pic_child(cv):
    cv.disc(18.2, 14.2, 1.15, BLACK)
    cv.seg((18.2, 15.6), (18.2, 19.4), 1.8, BLACK)
    cv.poly([(18.2, 19.4), (17.0, 23.2)], 1.4, BLACK)
    cv.poly([(18.2, 19.4), (19.4, 23.2)], 1.4, BLACK)
    cv.disc(13.8, 16.0, 1.0, BLACK)
    cv.seg((13.8, 17.2), (13.8, 20.4), 1.6, BLACK)
    cv.poly([(13.8, 20.4), (12.8, 23.2)], 1.3, BLACK)
    cv.poly([(13.8, 20.4), (14.8, 23.2)], 1.3, BLACK)
    cv.seg((14.4, 17.8), (17.6, 16.8), 1.2, BLACK)


def pic_bike(cv):
    for wx in (12.6, 19.4):
        cv.ring(wx, 20.6, 1.7, 2.6, BLACK)
    cv.poly([(12.6, 20.6), (15.2, 16.4), (17.6, 20.6), (15.0, 20.6), (12.6, 20.6)], 1.2, BLACK)
    cv.poly([(17.6, 20.6), (18.6, 15.8), (17.4, 15.4)], 1.2, BLACK)
    cv.seg((14.2, 15.6), (16.0, 15.6), 1.2, BLACK)


def pic_ice(cv):
    for a in (0, 60, 120):
        r = np.radians(a)
        cv.seg((16 - 3.0 * np.cos(r), 14.6 - 3.0 * np.sin(r)), (16 + 3.0 * np.cos(r), 14.6 + 3.0 * np.sin(r)), 1.2, BLACK)
    _car(cv, 16.0, 20.6, 8.0, BLACK)


def pic_deer(cv):
    cv.poly([(12.6, 18.4), (18.4, 17.4)], 3.0, BLACK)
    cv.poly([(18.4, 17.4), (20.4, 14.2)], 2.0, BLACK)
    cv.disc(20.6, 13.8, 1.2, BLACK)
    cv.poly([(20.6, 13.0), (21.6, 11.4)], 0.9, BLACK)
    cv.poly([(20.0, 12.8), (20.2, 11.2)], 0.9, BLACK)
    cv.poly([(18.0, 18.6), (20.2, 22.6)], 1.3, BLACK)
    cv.poly([(16.8, 18.8), (18.0, 22.8)], 1.3, BLACK)
    cv.poly([(13.6, 18.8), (11.6, 22.4)], 1.3, BLACK)
    cv.poly([(14.6, 19.0), (13.8, 22.8)], 1.3, BLACK)


# ---------- red-circle contents ----------
def cont_cars(cv, left=RED, right=BLACK):
    _car(cv, 11.8, 16.2, 6.4, left)
    _car(cv, 20.2, 16.8, 6.0, right)


def cont_truckcar(cv, truck=RED, car=BLACK):
    cv.rect(8.8, 13.2, 15.2, 18.0, truck)
    cv.rect(8.8, 15.4, 11.0, 18.6, truck)
    cv.disc(10.6, 18.6, 0.9, truck)
    cv.disc(13.6, 18.6, 0.9, truck)
    _car(cv, 20.4, 17.0, 5.6, car)


def cont_truck(cv):
    cv.rect(11.4, 13.0, 21.0, 17.8, BLACK)
    cv.rect(10.6, 15.2, 13.4, 18.4, BLACK)
    cv.disc(12.6, 18.4, 1.0, BLACK)
    cv.disc(16.4, 18.4, 1.0, BLACK)
    cv.disc(19.4, 18.4, 1.0, BLACK)


def cont_bar(cv):
    cv.disc(16, 16, 13.0, RED)
    cv.rect(7.0, 13.9, 25.0, 18.1, WHITE)


def stripes(cv, color=DGRAY, n=3, t=2.0, gap=4.0):
    for i in range(n):
        o = (i - (n - 1) / 2) * gap
        cv.seg((16 + o - 8.5, 16 + o + 8.5), (16 + o + 8.5, 16 + o - 8.5), t, color)


def cont_stop_text(cv):
    y0, y1 = 13.2, 18.8
    t = 1.5
    x = 8.6
    # S
    cv.poly([(x + 3.0, y0 + 0.2), (x + 0.4, y0 + 0.2), (x + 0.4, y0 + 2.6), (x + 3.0, y0 + 2.9), (x + 3.0, y1 - 0.2), (x + 0.4, y1 - 0.2)], t, WHITE)
    x += 4.4
    # T
    cv.seg((x, y0 + 0.2), (x + 3.2, y0 + 0.2), t, WHITE)
    cv.seg((x + 1.6, y0 + 0.2), (x + 1.6, y1 - 0.2), t, WHITE)
    x += 4.6
    # O
    cv.poly([(x + 1.6, y0 + 0.1), (x + 3.1, y0 + 1.4), (x + 3.1, y1 - 1.4), (x + 1.6, y1 - 0.1), (x + 0.1, y1 - 1.4), (x + 0.1, y0 + 1.4), (x + 1.6, y0 + 0.1)], t, WHITE)
    x += 4.6
    # P
    cv.seg((x + 0.4, y0 + 0.2), (x + 0.4, y1 - 0.2), t, WHITE)
    cv.poly([(x + 0.4, y0 + 0.2), (x + 2.8, y0 + 0.2), (x + 2.8, y0 + 2.8), (x + 0.4, y0 + 2.8)], t, WHITE)


# ---------- blue-disc arrows ----------
def arr_up(cv):
    cv.seg((16, 22.5), (16, 12.5), 3.0, WHITE)
    arrow_head(cv, (16, 8.2), (16, 13.4), 3.4, 1.0, WHITE)


def arr_bend(cv, d):
    s = 1 if d == "right" else -1
    cv.poly([(16 - 2.0 * s, 22.5), (16 - 2.0 * s, 14.5), (16 + 3.0 * s, 14.5)], 3.0, WHITE)
    arrow_head(cv, (16 + 7.6 * s, 14.5), (16 + 2.6 * s, 14.5), 3.4, 1.0, WHITE)


def arr_straight_or(cv, d):
    s = 1 if d == "right" else -1
    cv.seg((16 - 1.6 * s, 22.5), (16 - 1.6 * s, 12.5), 2.8, WHITE)
    arrow_head(cv, (16 - 1.6 * s, 8.4), (16 - 1.6 * s, 13.4), 3.2, 1.0, WHITE)
    cv.seg((16 - 1.6 * s, 16.0), (16 + 3.4 * s, 16.0), 2.8, WHITE)
    arrow_head(cv, (16 + 7.8 * s, 16.0), (16 + 2.8 * s, 16.0), 3.2, 1.0, WHITE)


def arr_keep(cv, d):
    s = 1 if d == "right" else -1
    cv.seg((16 - 4.0 * s, 11.5), (16 + 2.6 * s, 18.1), 3.2, WHITE)
    arrow_head(cv, (16 + 6.4 * s, 21.9), (16 + 2.2 * s, 17.7), 3.6, 1.0, WHITE)


def arr_round(cv):
    r = 6.8
    for base in (15, 135, 255):
        th0 = np.radians(-base)          # tail
        th1 = th0 - np.radians(80)       # head end (counter-clockwise on screen)
        cv.arc(16, 16, r, np.degrees(th1), np.degrees(th0), 2.6, WHITE)
        hx, hy = 16 + r * np.cos(th1), 16 + r * np.sin(th1)
        dx, dy = np.sin(th1), -np.cos(th1)   # direction of travel (ccw)
        arrow_head(cv, (hx + dx * 2.6, hy + dy * 2.6), (hx - dx * 1.6, hy - dy * 1.6), 2.0, 1.0, WHITE)


# ---------- class renderers ----------
SPEED_TXT = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}


def draw_class(cv, k):
    if k in SPEED_TXT:
        circle_base(cv)
        draw_digits(cv, SPEED_TXT[k])
    elif k == 6:
        cv.disc(16, 16, 13.0, WHITE)
        cv.ring(16, 16, 12.2, 13.0, GRAY)
        draw_digits(cv, "80", color=(0.32, 0.32, 0.32))
        stripes(cv, t=1.4)
    elif k == 9:
        circle_base(cv)
        cont_cars(cv)
    elif k == 10:
        circle_base(cv)
        cont_truckcar(cv)
    elif k == 11 or k == 12:
        diamond(cv)
    elif k == 13:
        triangle_down(cv)
    elif k == 14:
        octagon(cv)
        cont_stop_text(cv)
    elif k == 15:
        circle_base(cv)
    elif k == 16:
        circle_base(cv)
        cont_truck(cv)
    elif k == 17:
        cont_bar(cv)
    elif k == 18:
        triangle_up(cv)
        pic_excl(cv)
    elif k == 19:
        triangle_up(cv)
        pic_curve(cv, "left")
    elif k == 20:
        triangle_up(cv)
        pic_curve(cv, "right")
    elif k == 21:
        triangle_up(cv)
        pic_double(cv)
    elif k == 22:
        triangle_up(cv)
        pic_bump(cv)
    elif k == 23:
        triangle_up(cv)
        pic_slip(cv)
    elif k == 24:
        triangle_up(cv)
        pic_narrow(cv)
    elif k == 25:
        triangle_up(cv)
        pic_work(cv)
    elif k == 26:
        triangle_up(cv)
        pic_tlight(cv)
    elif k == 27:
        triangle_up(cv)
        pic_ped(cv)
    elif k == 28:
        triangle_up(cv)
        pic_child(cv)
    elif k == 29:
        triangle_up(cv)
        pic_bike(cv)
    elif k == 30:
        triangle_up(cv)
        pic_ice(cv)
    elif k == 31:
        triangle_up(cv)
        pic_deer(cv)
    elif k == 32:
        cv.disc(16, 16, 13.0, WHITE)
        cv.ring(16, 16, 12.2, 13.0, GRAY)
        stripes(cv, t=1.4)
    elif k == 33:
        blue_disc(cv)
        arr_bend(cv, "right")
    elif k == 34:
        blue_disc(cv)
        arr_bend(cv, "left")
    elif k == 35:
        blue_disc(cv)
        arr_up(cv)
    elif k == 36:
        blue_disc(cv)
        arr_straight_or(cv, "right")
    elif k == 37:
        blue_disc(cv)
        arr_straight_or(cv, "left")
    elif k == 38:
        blue_disc(cv)
        arr_keep(cv, "right")
    elif k == 39:
        blue_disc(cv)
        arr_keep(cv, "left")
    elif k == 40:
        blue_disc(cv)
        arr_round(cv)
    elif k == 41:
        cv.disc(16, 16, 13.0, WHITE)
        cv.ring(16, 16, 12.2, 13.0, GRAY)
        cont_cars(cv, left=GRAY, right=GRAY)
        stripes(cv, t=1.4)
    elif k == 42:
        cv.disc(16, 16, 13.0, WHITE)
        cv.ring(16, 16, 12.2, 13.0, GRAY)
        cont_truckcar(cv)
        stripes(cv, t=1.4)
    else:
        raise ValueError(k)


def render(k, blur=0.55):
    cv = Canvas()
    draw_class(cv, k)
    return cv.img(blur=blur)


# ---------- field renders (family bases, used as template backgrounds) ----------
def _finish(cv, blur=0.55):
    return cv.img(blur=blur)


def field_white_disc():
    cv = Canvas()
    cv.disc(16, 16, 13.0, WHITE)
    return _finish(cv)


def field_red_disc():
    cv = Canvas()
    cv.disc(16, 16, 13.0, RED)
    return _finish(cv)


def field_blue_disc():
    cv = Canvas()
    cv.disc(16, 16, 13.0, BLUE)
    return _finish(cv)


def field_octagon():
    cv = Canvas()
    octagon(cv)
    return _finish(cv)


def field_tri_up():
    cv = Canvas()
    triangle_up(cv)
    return _finish(cv)


def field_white_tri():
    cv = Canvas()
    cv.convexpoly([(16, 4.6), (16 + 13.0, 27.4), (16 - 13.0, 27.4)], WHITE)
    return _finish(cv)


def field_white_tridown():
    cv = Canvas()
    cv.convexpoly([(16 - 13.0, 5.0), (16 + 13.0, 5.0), (16, 27.0)], WHITE)
    return _finish(cv)

"""Build the hand-designed network (final architecture).

  image -> norm (per-channel standardisation: lighting invariance)
        -> conv "paint": 14 fixed filters (colour opponents, luminance, 4 oriented
           edges x2 polarities, green) -> relu
        -> conv FULL: whole-sign matched filters, one per class (11/12 shared):
           kernel = paint of the procedurally drawn ideal sign, averaged over two
           blur levels, 29x29 crop, unit L2 norm
        -> relu -> maxpool 2x2 (translation tolerance) -> dense (rule table)
        -> conv PART: symbol matched filters on the sign centre, 17x17 crops:
           * red-ring / triangle / blue-disc families: raw paint crop (symbol + local
             field context)
           * grey "end of" family: ink crop (class paint minus stripes-disc paint, so
             the shared stripes cancel)
           * EMPTY: hand formula (bright, ink-free interior)
        -> relu -> maxpool 4x4 -> dense (rule table)
        -> add -> 43 class scores

All weights derive from formulas and hand-drawn idealized sign geometry (general
knowledge of the German sign designs). No training data is used anywhere.
"""
import numpy as np

NCH = 14
BLURS = (0.4, 1.0)


def gauss2d(k, sigma):
    ax = np.arange(k) - (k - 1) / 2
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    G = np.outer(g, g)
    return G / G.sum()


def deriv2d(k, sigma, angle):
    ax = np.arange(k) - (k - 1) / 2
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    d = -ax / sigma ** 2 * np.exp(-0.5 * (ax / sigma) ** 2)
    c, s = np.cos(angle), np.sin(angle)
    return c * np.outer(g, d) + s * np.outer(d, g)


def paint_filters():
    g = gauss2d(5, 1.0)
    W = np.zeros((NCH, 3, 5, 5))
    W[0, 0] = g;  W[0, 1] = -g                              # RED
    W[1, 2] = g;  W[1, 0] = -0.5 * g; W[1, 1] = -0.5 * g    # BLUE
    W[2, 1] = g;  W[2, 2] = -g                              # YELLOW (G-B)
    W[3] = g / 3.0                                          # BRIGHT
    W[4] = -g / 3.0                                         # DARK
    for j, a in enumerate([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
        d = deriv2d(5, 1.0, a)
        W[5 + 2 * j] = d / 3.0
        W[6 + 2 * j] = -d / 3.0
    W[13, 1] = g; W[13, 0] = -0.5 * g; W[13, 2] = -0.5 * g  # GREEN
    return W


def conv_np(x, W, b, stride=1, padding=0):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
                     for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def norm_np(x):
    mu = x.mean(axis=(2, 3), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)


def blur32(img, sigma):
    if sigma <= 0:
        return img
    k = int(np.ceil(sigma * 3)) * 2 + 1
    g = np.exp(-0.5 * (np.arange(k) - k // 2) ** 2 / sigma ** 2)
    g /= g.sum()
    out = img.copy()
    for c in range(3):
        out[c] = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 1, out[c])
        out[c] = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 0, out[c])
    return out


PAINT_W = paint_filters()


def ideal_paint(img):
    """relu paint, averaged over two blur levels (sharp + motion-blurred)."""
    ps = [np.maximum(conv_np(norm_np(blur32(img, b)[None]), PAINT_W, np.zeros(NCH), 1, 2)[0], 0)
          for b in BLURS]
    return np.mean(ps, 0)


# ---------------- template catalogue ----------------
FULL_CLASSES = [k for k in range(43) if k != 12]      # 11 and 12 look identical
FULL_ID = {k: i for i, k in enumerate(FULL_CLASSES)}
NF = len(FULL_CLASSES)
FULL_K = 29

# part templates: (name, kind, class, centre x, centre y)
#   kind "raw": paint crop;  kind "ink": paint(class) - paint(base)
PARTS = (
    [(f"D{t}", "raw", k, 16.0, 16.6)
     for t, k in [("20", 0), ("30", 1), ("50", 2), ("60", 3), ("70", 4), ("80", 5), ("100", 7), ("120", 8)]] +
    [("CARS9", "raw", 9, 16.0, 16.5), ("TRUCKCAR10", "raw", 10, 16.0, 16.5),
     ("TRUCK16", "raw", 16, 16.0, 16.3), ("BAR17", "raw", 17, 16.0, 16.0),
     ("STOP", "raw", 14, 16.0, 16.0)] +
    [("D80G", "ink", 6, 16.0, 16.6, 32), ("CARS9G", "ink", 41, 16.0, 16.5, 32),
     ("TRUCKCAR10G", "ink", 42, 16.0, 16.5, 32), ("TRUCKG", "ink", 42, 13.5, 15.5, 41),
     ("STRIPES", "ink", 32, 16.0, 16.0, 15)] +
    [(f"P{k}", "raw", k, 16.0, 18.0) for k in range(18, 32)] +
    [("A33", "raw", 33, 16.0, 15.5), ("A34", "raw", 34, 16.0, 15.5),
     ("A35", "raw", 35, 16.0, 15.5), ("A36", "raw", 36, 16.0, 15.5),
     ("A37", "raw", 37, 16.0, 15.5), ("A38", "raw", 38, 16.0, 16.0),
     ("A39", "raw", 39, 16.0, 16.0), ("A40", "raw", 40, 16.0, 16.0)]
)
PART_CX = {"P19": 14.0, "P20": 18.5}
PART_ID = {n: i for i, t in enumerate(PARTS) for n in [t[0]]}
PART_ID["EMPTY"] = len(PARTS)
NP = len(PARTS) + 1
PART_K = 13


def build_templates():
    FW = np.zeros((NF, NCH, FULL_K, FULL_K))
    for k in FULL_CLASSES:
        FW[FULL_ID[k]] = ideal_paint(signs.render(k))[:, 2:2 + FULL_K, 2:2 + FULL_K]
    PW = np.zeros((NP, NCH, PART_K, PART_K))
    for i, spec in enumerate(PARTS):
        n, kind, k = spec[0], spec[1], spec[2]
        cx = PART_CX.get(n, spec[3])
        cy = spec[4]
        y0, x0 = int(round(cy)) - PART_K // 2, int(round(cx)) - PART_K // 2
        pc = ideal_paint(signs.render(k))
        if kind == "raw":
            crop = pc[:, y0:y0 + PART_K, x0:x0 + PART_K]
        else:
            base = spec[5]
            crop = (pc - ideal_paint(signs.render(base)))[:, y0:y0 + PART_K, x0:x0 + PART_K]
        if n == "A35":   # white side branches mean it is NOT "ahead only"
            for other in (36, 37):
                br = np.maximum(ideal_paint(signs.render(other)) - pc, 0)
                crop = crop - 0.6 * br[:, y0:y0 + PART_K, x0:x0 + PART_K]
        PW[i] = crop
    # EMPTY: hand formula - bright, ink-free interior
    yy, xx = np.mgrid[0:PART_K, 0:PART_K]
    m = (((yy - 8) ** 2 + (xx - 8) ** 2) <= 9.0 ** 2).astype(float)
    e = np.zeros((NCH, PART_K, PART_K))
    e[3] = m
    e[4] = -m
    PW[NP - 1] = e
    FW /= np.linalg.norm(FW.reshape(NF, -1), axis=1)[:, None, None, None]
    PW /= np.linalg.norm(PW.reshape(NP, -1), axis=1)[:, None, None, None]
    return FW, PW


PRIOR = np.array([.05, .22, .22, .14, .18, .18, .05, .14, .14, .10, .14, .00, .10, .14, .05,
                  .05, .05, .10, .10, .08, .08, .05, .05, .05, .03, .14, .05, .03, .05, .03,
                  .05, .08, .03, .06, .04, .10, .05, .03, .12, .03, .06, .02, .02])


def rules():
    F, C = FULL_ID, PART_ID
    R = {}
    sp = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}
    for k, t in sp.items():
        R[k] = [("F", F[k], 1.0), ("P", C[f"D{t}"], 0.9)]
    R[6] = [("F", F[6], 1.0), ("P", C["D80G"], 0.9), ("P", C["STRIPES"], 0.5),
            ("P", C["CARS9G"], -0.4), ("P", C["TRUCKCAR10G"], -0.4), ("P", C["TRUCKG"], -0.5)]
    R[9] = [("F", F[9], 1.0), ("P", C["CARS9"], 0.9)]
    R[10] = [("F", F[10], 1.0), ("P", C["TRUCKCAR10"], 0.9)]
    R[11] = [("F", F[11], 1.0)]
    R[12] = [("F", F[11], 1.0)]
    R[13] = [("F", F[13], 1.15)]
    R[14] = [("F", F[14], 1.0), ("P", C["STOP"], 0.9), ("P", C["BAR17"], -0.5)]
    R[15] = [("F", F[15], 1.0), ("P", C["EMPTY"], 0.7)]
    R[16] = [("F", F[16], 1.0), ("P", C["TRUCK16"], 0.9)]
    R[17] = [("F", F[17], 1.0), ("P", C["BAR17"], 1.0), ("P", C["STOP"], -0.5)]
    for k in range(18, 32):
        R[k] = [("F", F[k], 1.0), ("P", C[f"P{k}"], 0.9)]
    R[18] += [("P", C[f"P{k}"], -0.3) for k in (24, 25, 26, 27, 28, 29, 31)]
    R[26] += [("P", C["P18"], -0.3)]
    R[32] = [("F", F[32], 1.0), ("P", C["STRIPES"], 0.8),
             ("P", C["D80G"], -0.4), ("P", C["CARS9G"], -0.4), ("P", C["TRUCKCAR10G"], -0.4),
             ("P", C["TRUCKG"], -0.6)]
    for k in range(33, 41):
        R[k] = [("F", F[k], 1.0), ("P", C[f"A{k}"], 0.9)]
    R[33] += [("P", C["A36"], -0.5)]
    R[34] += [("P", C["A37"], -0.5)]
    R[35] += [("P", C["A36"], -0.5), ("P", C["A37"], -0.5)]
    R[36] += [("P", C["A35"], -0.4)]
    R[37] += [("P", C["A35"], -0.4)]
    R[41] = [("F", F[41], 1.0), ("P", C["CARS9G"], 0.9), ("P", C["STRIPES"], 0.5),
             ("P", C["TRUCKG"], -0.8), ("P", C["D80G"], -0.5)]
    R[42] = [("F", F[42], 1.0), ("P", C["TRUCKCAR10G"], 0.9), ("P", C["STRIPES"], 0.5),
             ("P", C["TRUCKG"], 1.0), ("P", C["D80G"], -0.5)]
    return R


FULL_CELLS = [0, 1, 2, 3]


def part_cells(name):
    spec = {t[0]: t for t in PARTS}.get(name)
    cy = spec[4] if spec else 16.0
    cx = PART_CX.get(name, spec[3] if spec else 16.0)
    cells = set()
    for oy in range(int(round(cy)) - 6 - 2, int(round(cy)) - 6 + 3):
        for ox in range(int(round(cx)) - 6 - 2, int(round(cx)) - 6 + 3):
            r, c = oy // 5, ox // 5
            if 0 <= r < 4 and 0 <= c < 4:
                cells.add(r * 4 + c)
    return sorted(cells)


def build_network():
    FW, PW = build_templates()
    layers = [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "c1", "op": "conv", "inputs": ["n0"], "out": NCH, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "cf", "op": "conv", "inputs": ["r1"], "out": NF, "kernel": [FULL_K, FULL_K], "stride": 1, "padding": 0},
        {"name": "rf", "op": "relu", "inputs": ["cf"]},
        {"name": "pf", "op": "maxpool", "inputs": ["rf"], "size": [2, 2], "stride": [2, 2]},
        {"name": "cp", "op": "conv", "inputs": ["r1"], "out": NP, "kernel": [PART_K, PART_K], "stride": 1, "padding": 0},
        {"name": "rp", "op": "relu", "inputs": ["cp"]},
        {"name": "pp", "op": "maxpool", "inputs": ["rp"], "size": [5, 5], "stride": [5, 5]},
        {"name": "df", "op": "dense", "inputs": ["pf"], "out": 43},
        {"name": "dp", "op": "dense", "inputs": ["pp"], "out": 43},
        {"name": "scores", "op": "add", "inputs": ["df", "dp"]},
    ]
    W = {"c1.W": PAINT_W.astype(np.float32), "c1.b": np.zeros(NCH, np.float32),
         "cf.W": FW.astype(np.float32), "cf.b": np.zeros(NF, np.float32),
         "cp.W": PW.astype(np.float32), "cp.b": np.zeros(NP, np.float32)}
    R = rules()
    WF = np.zeros((43, NF * 4))
    WP = np.zeros((43, NP * 16))
    for k in range(43):
        for br, t, w in R[k]:
            if br == "F":
                for c in FULL_CELLS:
                    WF[k, t * 4 + c] += w * 0.25
            else:
                cs = part_cells([n for n in PART_ID if PART_ID[n] == t][0])
                for c in cs:
                    WP[k, t * 16 + c] += w / len(cs)
    W["df.W"] = WF.astype(np.float32); W["df.b"] = PRIOR.astype(np.float32)
    W["dp.W"] = WP.astype(np.float32); W["dp.b"] = np.zeros(43, np.float32)
    return {"layers": layers}, W


# ---------------- entry point: write the answer files ----------------
def main(out_dir):
    import json, os
    net, W = build_network()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **W)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
