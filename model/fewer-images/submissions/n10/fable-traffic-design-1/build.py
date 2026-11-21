#!/usr/bin/env python3
"""Builds a hand-designed traffic sign network: python3 build.py OUT_DIR

Nothing here is trained or fitted and no file is read.  The program
  1. draws each of the 43 German signs from geometric descriptions typed below (rings, triangles, digits as strokes,
     arrows, pictograms made of a few strokes and blobs),
  2. places each drawing at a grid of poses (size, small rotation, aspect, half-pixel shift),
  3. passes each drawing through the fixed front end of the network (formula filters only) and keeps the result,
     normalised by itself alone, as one matched filter (template) of the network.
The network takes, for each class, the best match over all its templates and positions.
"""
import json
import sys
import numpy as np

F32 = np.float32

# ----------------------------------------------------------------------------------------------------------------------
# 1. Drawing the signs
# ----------------------------------------------------------------------------------------------------------------------
NC = 448                # canonical raster size
CEXT = 1.4              # canonical raster covers [-CEXT, CEXT]^2 ; the sign's bounding box is [-1, 1]^2
CPX = 2 * CEXT / NC     # size of one canonical raster cell

WHITE = (0.92, 0.92, 0.92)
RED = (0.72, 0.10, 0.12)
BLUE = (0.08, 0.27, 0.62)
YELLOW = (0.96, 0.76, 0.05)
BLACK = (0.07, 0.07, 0.07)
DGRAY = (0.25, 0.25, 0.25)
TL_RED = (0.85, 0.12, 0.10)
TL_YEL = (0.95, 0.75, 0.10)
TL_GRN = (0.10, 0.60, 0.25)


def sd_circle(X, Y, cx, cy, r):
    return np.hypot(X - cx, Y - cy) - r


def sd_ellipse(X, Y, cx, cy, rx, ry):
    return (np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2) - 1.0) * min(rx, ry)


def d_seg(X, Y, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return np.hypot(X - ax, Y - ay)
    t = np.clip(((X - ax) * vx + (Y - ay) * vy) / L2, 0.0, 1.0)
    return np.hypot(X - ax - t * vx, Y - ay - t * vy)


def sd_line(X, Y, pts, hw):
    """thick polyline with round joins and caps; pts list of (x, y); hw half width"""
    d = None
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        s = d_seg(X, Y, ax, ay, bx, by)
        d = s if d is None else np.minimum(d, s)
    return d - hw


def sd_poly(X, Y, pts):
    """signed distance to a simple polygon (negative inside)"""
    n = len(pts)
    d = None
    inside = np.zeros(X.shape, bool)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        s = d_seg(X, Y, ax, ay, bx, by)
        d = s if d is None else np.minimum(d, s)
        cond = (ay > Y) != (by > Y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = ax + (Y - ay) * (bx - ax) / (by - ay if by != ay else 1e-12)
        inside ^= cond & (X < xi)
    return np.where(inside, -d, d)


def sd_box(X, Y, cx, cy, hw, hh, r=0.0):
    qx = np.abs(X - cx) - (hw - r)
    qy = np.abs(Y - cy) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def arc(cx, cy, rx, ry, a0, a1, n=24):
    """points on an ellipse arc; angles in degrees, counter-clockwise as seen on screen (y axis points down)"""
    a = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(t), cy - ry * np.sin(t)) for t in a]


def arrow_head(tipx, tipy, dirx, diry, length, halfw):
    n = np.hypot(dirx, diry)
    dx, dy = dirx / n, diry / n
    bx, by = tipx - dx * length, tipy - dy * length
    return [(tipx, tipy), (bx - dy * halfw, by + dx * halfw), (bx + dy * halfw, by - dx * halfw)]


class Canvas:
    """paints layers given as signed distances; keeps colour image and the signed distance to the sign outline"""

    def __init__(self, sx=1.0, sy=1.0, oy=0.0):
        # canonical (u, v) -> drawing coordinates (X, Y) = (u * sx, v * sy + oy)
        c = (np.arange(NC) + 0.5) * CPX - CEXT
        U, V = np.meshgrid(c, c)
        self.X, self.Y = U * sx, V * sy + oy
        self.px = CPX * 0.5 * (sx + sy)
        self.scale = 0.5 * (sx + sy)
        self.img = np.zeros((3, NC, NC))   # colour of the sign, premultiplied by its coverage
        self.alpha = np.zeros((NC, NC))    # coverage of the sign's silhouette
        self.outline = None

    def paint(self, sd, col):
        a = np.clip(0.5 - sd / self.px, 0.0, 1.0)
        for i in range(3):
            self.img[i] = self.img[i] * (1 - a) + col[i] * a

    def set_outline(self, sd):
        self.outline = sd / self.scale     # in canonical units (approximately)
        self.alpha = np.clip(0.5 - sd / self.px, 0.0, 1.0)

    def pen(self, s=1.0, cx=0.0, cy=0.0):
        """a pen that draws the symbol scaled by s about (cx, cy)"""
        return Pen(self, s, cx, cy)


class Pen:
    def __init__(self, cv, s, cx, cy):
        self.cv, self.s = cv, s
        self.X = (cv.X - cx) / s + cx
        self.Y = (cv.Y - cy) / s + cy

    def paint(self, sd, col):
        self.cv.paint(sd * self.s, col)


# ---- digits and letters as strokes: glyph box is [0, w] x [0, 1], stroke width t -------------------------------------
def glyph(ch, w, t):
    m = t / 2
    rx = w / 2 - m
    if ch == "0" or ch == "O":
        ry = min(rx * 1.15, 0.5 - m)
        p = arc(w / 2, m + ry, rx, ry, 180, 0) + arc(w / 2, 1 - m - ry, rx, ry, 0, -180)
        return [p + [p[0]]]
    if ch == "1":
        xs = w - m
        return [[(m, 0.30), (xs, m), (xs, 1 - m)]]
    if ch == "2":
        ry = 0.25
        return [arc(w / 2, m + ry, rx, ry, 170, -35) + [(m, 1 - m), (w - m, 1 - m)]]
    if ch == "3":
        ym = 0.48
        ry1, ry2 = (ym - m) / 2, (1 - m - ym) / 2
        return [arc(w / 2, m + ry1, rx * 0.95, ry1, 150, -90), arc(w / 2, ym + ry2, rx, ry2, 90, -150)]
    if ch == "5":
        ry = 0.275
        cy = 1 - m - ry
        return [[(w - m - 0.03, m), (m + 0.04, m), (m + 0.02, 0.44)] + arc(w / 2 - 0.01, cy, rx, ry, 150, -155)]
    if ch == "6":
        ry = 0.265
        cy = 1 - m - ry
        ryt = 0.29
        bowl = arc(w / 2, cy, rx, ry, 0, 360, 40)
        return [bowl, arc(w / 2, m + ryt, rx, ryt, 40, 180) + [(m, cy)]]
    if ch == "7":
        return [[(m, m), (w - m, m), (0.36 * w, 1 - m)]]
    if ch == "8":
        ym = 0.48
        ry1, ry2 = (ym - m) / 2, (1 - m - ym) / 2
        return [arc(w / 2, m + ry1, rx * 0.92, ry1, 0, 360, 40), arc(w / 2, ym + ry2, rx, ry2, 0, 360, 40)]
    if ch == "S":
        ry1, ry2 = (0.5 - m) / 2, (0.5 - m) / 2
        return [arc(w / 2, m + ry1, rx, ry1, 30, 270) + arc(w / 2, 0.5 + ry2, rx, ry2, 90, -150)]
    if ch == "T":
        return [[(m, m), (w - m, m)], [(w / 2, m), (w / 2, 1 - m)]]
    if ch == "P":
        ry = (0.54 - m) / 2
        return [[(m, 1 - m), (m, m), (w / 2, m)] + arc(w / 2, m + ry, rx, ry, 90, -90) + [(m, m + 2 * ry)]]
    raise ValueError(ch)


def draw_text(cv, text, widths, H, t, gap, col, cy=0.0, cx=0.0):
    """text centred at (cx, cy); widths and stroke t relative to the glyph height H"""
    total = sum(w * H for w in widths) + gap * (len(text) - 1)
    x0 = cx - total / 2
    y0 = cy - H / 2
    for ch, w in zip(text, widths):
        for line in glyph(ch, w, t):
            pts = [(x0 + px * H, y0 + py * H) for px, py in line]
            cv.paint(sd_line(cv.X, cv.Y, pts, t * H / 2), col)
        x0 += w * H + gap


def draw_number(cv, text, col, cy=0.0):
    if len(text) == 2:
        draw_text(cv, text, [0.56, 0.56], 0.84, 0.145, 0.10, col, cy)
    else:
        widths = [0.30 if ch == "1" else 0.47 for ch in text]
        draw_text(cv, text, widths, 0.78, 0.135, 0.075, col, cy)


# ---- vehicles ---------------------------------------------------------------------------------------------------------
def car_rear(cv, cx, cy, W, H, col):
    X, Y = cv.X, cv.Y
    cv.paint(sd_box(X, Y, cx, cy + 0.12 * H, W / 2, 0.26 * H, 0.06 * H), col)
    cab = [(cx - 0.30 * W, cy - 0.5 * H), (cx + 0.30 * W, cy - 0.5 * H), (cx + 0.40 * W, cy - 0.1 * H), (cx - 0.40 * W, cy - 0.1 * H)]
    cv.paint(sd_poly(X, Y, cab), col)
    for s in (-1, 1):
        cv.paint(sd_box(X, Y, cx + s * 0.33 * W, cy + 0.42 * H, 0.09 * W, 0.08 * H), col)


def truck_rear(cv, cx, cy, W, H, col):
    X, Y = cv.X, cv.Y
    cv.paint(sd_box(X, Y, cx, cy - 0.07 * H, W / 2, 0.43 * H, 0.02), col)
    for s in (-1, 1):
        cv.paint(sd_box(X, Y, cx + s * 0.30 * W, cy + 0.42 * H, 0.11 * W, 0.08 * H), col)


def truck_side(cv, col):
    X, Y = cv.X, cv.Y
    cv.paint(sd_box(X, Y, 0.17, -0.09, 0.37, 0.215, 0.02), col)                      # load box
    cab = [(-0.54, 0.13), (-0.54, -0.02), (-0.44, -0.16), (-0.25, -0.16), (-0.25, 0.13)]
    cv.paint(sd_poly(X, Y, cab), col)                                                # cab with sloping screen
    cv.paint(sd_box(X, Y, 0.0, 0.15, 0.54, 0.03), col)                               # chassis
    for wx in (-0.37, 0.14, 0.36):
        cv.paint(sd_circle(X, Y, wx, 0.21, 0.085), col)


def stripes(cv, col, halfband=0.15, n=5, lw=0.018):
    """diagonal band of thin lines from lower left to upper right"""
    X, Y = cv.X, cv.Y
    dist = (X + Y) / np.sqrt(2.0)                 # distance from the diagonal through the centre
    inside = sd_circle(X, Y, 0, 0, 0.93)
    for k in range(n):
        off = -halfband + (k + 0.5) * (2 * halfband / n)
        cv.paint(np.maximum(np.abs(dist - off) - lw * halfband / 0.15, inside), col)


# ---- sign bodies ------------------------------------------------------------------------------------------------------
def red_ring(cv, inner=0.72):
    X, Y = cv.X, cv.Y
    o = sd_circle(X, Y, 0, 0, 1.0)
    cv.set_outline(o)
    cv.paint(o, WHITE)
    cv.paint(sd_circle(X, Y, 0, 0, 0.975), RED)
    cv.paint(sd_circle(X, Y, 0, 0, inner), WHITE)


def blue_disc(cv):
    X, Y = cv.X, cv.Y
    o = sd_circle(X, Y, 0, 0, 1.0)
    cv.set_outline(o)
    cv.paint(o, WHITE)
    cv.paint(sd_circle(X, Y, 0, 0, 0.965), BLUE)


def end_disc(cv):
    X, Y = cv.X, cv.Y
    o = sd_circle(X, Y, 0, 0, 1.0)
    cv.set_outline(o)
    cv.paint(o, BLACK)
    cv.paint(sd_circle(X, Y, 0, 0, 0.955), WHITE)


TRI_R = 0.10                                   # corner rounding of triangles (side length 2)
TRI_SX = 1 - (np.sqrt(3) - 1) * TRI_R          # half width of the rounded triangle
TRI_TOP = -2 / np.sqrt(3) + TRI_R              # top of the rounded triangle (apex up), centroid at the origin
TRI_BOT = 1 / np.sqrt(3)
TRI_SY = (TRI_BOT - TRI_TOP) / 2
TRI_OY = (TRI_BOT + TRI_TOP) / 2


def tri_pts(inset):
    """equilateral triangle, side 2, apex up, centroid at origin, shrunk by `inset` (distance of each side)"""
    f = 1 - inset * np.sqrt(3)
    return [(0.0, -2 / np.sqrt(3) * f), (f, f / np.sqrt(3)), (-f, f / np.sqrt(3))]


def warning_triangle():
    """canvas whose drawing coordinates have the triangle's centroid at the origin and side length 2"""
    cv = Canvas(TRI_SX, TRI_SY, TRI_OY)
    X, Y = cv.X, cv.Y
    o = sd_poly(X, Y, tri_pts(TRI_R)) - TRI_R
    cv.set_outline(o)
    cv.paint(o, WHITE)
    cv.paint(o + 0.02, RED)
    cv.paint(sd_poly(X, Y, tri_pts(0.18 + 0.03)) - 0.03, WHITE)
    return cv


def yield_triangle(border=0.21):
    cv = Canvas(TRI_SX, TRI_SY, -TRI_OY)
    X, Y = cv.X, -cv.Y
    o = sd_poly(X, Y, tri_pts(TRI_R)) - TRI_R
    cv.set_outline(o)
    cv.paint(o, WHITE)
    cv.paint(o + 0.025, RED)
    cv.paint(sd_poly(X, Y, tri_pts(border + 0.03)) - 0.03, WHITE)
    return cv


def person(cv, x, y, s, col, lean=0.0, stride=0.16):
    """simple walking figure; (x, y) is the hip, s the size of the figure (hip to top of head about 0.5 s)"""
    X, Y = cv.X, cv.Y
    sh = (x + lean * s, y - 0.30 * s)                         # shoulders
    cv.paint(sd_circle(X, Y, sh[0] + 0.02 * s, sh[1] - 0.13 * s, 0.085 * s), col)
    cv.paint(sd_line(X, Y, [sh, (x, y)], 0.075 * s), col)
    cv.paint(sd_line(X, Y, [(x, y), (x - stride * s, y + 0.36 * s)], 0.05 * s), col)
    cv.paint(sd_line(X, Y, [(x, y), (x + stride * s, y + 0.36 * s)], 0.05 * s), col)
    cv.paint(sd_line(X, Y, [sh, (sh[0] - 0.17 * s, sh[1] + 0.2 * s)], 0.04 * s), col)
    cv.paint(sd_line(X, Y, [sh, (sh[0] + 0.16 * s, sh[1] + 0.18 * s)], 0.04 * s), col)


class Mirror:
    """pen with the x axis mirrored"""

    def __init__(self, p):
        self.p, self.X, self.Y = p, -p.X, p.Y

    def paint(self, sd, col):
        self.p.paint(sd, col)


def blue_arrow_turn(p):
    """arrow rising from the bottom and turning to the right (209)"""
    X, Y = p.X, p.Y
    xs, yt, rc, hw = -0.28, -0.26, 0.30, 0.12
    pts = [(xs, 0.62), (xs, yt + rc)] + arc(xs + rc, yt + rc, rc, rc, 180, 90, 12) + [(0.34, yt)]
    p.paint(sd_line(X, Y, pts, hw), WHITE)
    p.paint(sd_poly(X, Y, arrow_head(0.72, yt, 1, 0, 0.40, 0.34)), WHITE)


def blue_arrow_straight_side(p):
    """arrow straight on with a branch to the right (214)"""
    X, Y = p.X, p.Y
    xs, hw = -0.22, 0.10
    p.paint(sd_line(X, Y, [(xs, 0.64), (xs, -0.34)], hw), WHITE)
    p.paint(sd_poly(X, Y, arrow_head(xs, -0.72, 0, -1, 0.38, 0.27)), WHITE)
    rc, yb = 0.30, 0.02
    pts = [(xs, yb + rc + 0.05), (xs, yb + rc)] + arc(xs + rc, yb + rc, rc, rc, 180, 90, 12) + [(0.36, yb)]
    p.paint(sd_line(X, Y, pts, hw), WHITE)
    p.paint(sd_poly(X, Y, arrow_head(0.70, yb, 1, 0, 0.36, 0.27)), WHITE)


def blue_arrow_keep(p):
    """arrow pointing down to the right (222)"""
    X, Y = p.X, p.Y
    p.paint(sd_line(X, Y, [(-0.40, -0.40), (0.22, 0.22)], 0.13), WHITE)
    p.paint(sd_poly(X, Y, arrow_head(0.55, 0.55, 1, 1, 0.46, 0.37)), WHITE)


DESIGN_SCALE = (1.0, 0.92, 1.08)     # my proportions are from memory: also try the inner symbol a little smaller / larger


def draw_class(k, dv=0):
    """returns the canvas with class k drawn in its bounding box; dv picks one of three proportion variants"""
    ps = DESIGN_SCALE[dv]
    speed = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}
    if k in speed:
        cv = Canvas()
        red_ring(cv)
        draw_number(cv.pen(ps), speed[k], BLACK)
        return cv
    if k == 6:                                    # end of 80
        cv = Canvas()
        end_disc(cv)
        draw_number(cv.pen(ps), "80", DGRAY)
        stripes(cv, BLACK)
        return cv
    if k == 9:                                    # no passing: red car left, black car right
        cv = Canvas()
        red_ring(cv)
        p = cv.pen(ps)
        car_rear(p, -0.30, 0.0, 0.50, 0.38, RED)
        car_rear(p, 0.30, 0.0, 0.50, 0.38, BLACK)
        return cv
    if k == 10:                                   # no passing for trucks: red truck left, black car right
        cv = Canvas()
        red_ring(cv)
        p = cv.pen(ps)
        truck_rear(p, -0.30, -0.04, 0.50, 0.60, RED)
        car_rear(p, 0.32, 0.08, 0.46, 0.36, BLACK)
        return cv
    if k == 15:
        cv = Canvas()
        red_ring(cv, (0.72, 0.76, 0.68)[dv])
        return cv
    if k == 16:
        cv = Canvas()
        red_ring(cv)
        truck_side(cv.pen(ps), BLACK)
        return cv
    if k == 17:                                   # no entry
        cv = Canvas()
        X, Y = cv.X, cv.Y
        o = sd_circle(X, Y, 0, 0, 1.0)
        cv.set_outline(o)
        cv.paint(o, WHITE)
        cv.paint(sd_circle(X, Y, 0, 0, 0.965), RED)
        cv.paint(sd_box(X, Y, 0, 0, 0.78 * ps, 0.17 * ps), WHITE)
        return cv
    if k == 14:                                   # stop
        cv = Canvas()
        X, Y = cv.X, cv.Y
        c = np.tan(np.pi / 8)
        octa = [(c, -1), (1, -c), (1, c), (c, 1), (-c, 1), (-1, c), (-1, -c), (-c, -1)]
        o = sd_poly(X, Y, octa)
        cv.set_outline(o)
        cv.paint(o, WHITE)
        cv.paint(o + 0.065, RED)
        draw_text(cv.pen(ps), "STOP", [0.44, 0.44, 0.44, 0.44], 0.68, 0.17, 0.075, WHITE)
        return cv
    if k == 12:                                   # priority road
        cv = Canvas()
        X, Y = cv.X, cv.Y
        r = 0.09
        e = 1 - r * (np.sqrt(2) - 1)

        def diamond(size, rr):
            s = size - rr * np.sqrt(2)
            return sd_poly(X, Y, [(0, -s), (s, 0), (0, s), (-s, 0)]) - rr
        o = diamond(1 / e * 1.0, r)               # rounded diamond whose bounding box is [-1, 1]^2
        cv.set_outline(o)
        cv.paint(o, BLACK)
        cv.paint(o + 0.03, WHITE)
        cv.paint(diamond((0.60, 0.55, 0.65)[dv], 0.05), YELLOW)
        return cv
    if k == 13:
        return yield_triangle((0.21, 0.185, 0.235)[dv])
    if k == 32:
        cv = Canvas()
        end_disc(cv)
        stripes(cv, BLACK, (0.15, 0.12, 0.18)[dv])
        return cv
    if k == 41:
        cv = Canvas()
        end_disc(cv)
        p = cv.pen(ps)
        car_rear(p, -0.30, 0.0, 0.50, 0.38, DGRAY)
        car_rear(p, 0.30, 0.0, 0.50, 0.38, DGRAY)
        stripes(cv, BLACK)
        return cv
    if k == 42:
        cv = Canvas()
        end_disc(cv)
        p = cv.pen(ps)
        truck_rear(p, -0.30, -0.04, 0.50, 0.60, DGRAY)
        car_rear(p, 0.32, 0.08, 0.46, 0.36, DGRAY)
        stripes(cv, BLACK)
        return cv
    if 33 <= k <= 40:
        cv = Canvas()
        blue_disc(cv)
        p = cv.pen(ps)
        X, Y = p.X, p.Y
        if k == 33:
            blue_arrow_turn(p)
        elif k == 34:
            blue_arrow_turn(Mirror(p))
        elif k == 35:
            p.paint(sd_line(X, Y, [(0, 0.56), (0, -0.30)], 0.13), WHITE)
            p.paint(sd_poly(X, Y, arrow_head(0, -0.71, 0, -1, 0.44, 0.38)), WHITE)
        elif k == 36:
            blue_arrow_straight_side(p)
        elif k == 37:
            blue_arrow_straight_side(Mirror(p))
        elif k == 38:
            blue_arrow_keep(p)
        elif k == 39:
            blue_arrow_keep(Mirror(p))
        else:                                     # roundabout: three arrows running counter-clockwise
            rr = 0.48
            for a0 in (235, 355, 115):
                pts = arc(0, 0, rr, rr, a0, a0 + 62, 14)
                p.paint(sd_line(X, Y, pts, 0.095), WHITE)
                a = np.radians(a0 + 100)
                tip = (rr * np.cos(a), -rr * np.sin(a))
                b = np.radians(a0 + 62)
                base = (rr * np.cos(b), -rr * np.sin(b))
                p.paint(sd_poly(X, Y, arrow_head(tip[0], tip[1], tip[0] - base[0], tip[1] - base[1], 0.32, 0.21)), WHITE)
        return cv
    # ---- warning triangles: coordinates with the centroid at the origin, side 2; the white inside spans about
    #      y in [-0.79, 0.40], half width 0.46 at y = 0.  The official symbols are bold.
    cv = warning_triangle()
    p = cv.pen(ps, 0.0, -0.05)
    X, Y = p.X, p.Y
    K = BLACK
    if k == 18:                                   # exclamation mark
        p.paint(sd_poly(X, Y, [(-0.065, -0.50), (0.065, -0.50), (0.045, 0.08), (-0.045, 0.08)]), K)
        p.paint(sd_circle(X, Y, 0, 0.24, 0.07), K)
    elif k == 11:                                 # priority at next junction: broad arrow with a thinner cross bar
        p.paint(sd_poly(X, Y, [(0, -0.57), (0.25, -0.13), (0.12, -0.13), (0.12, 0.34), (-0.12, 0.34), (-0.12, -0.13), (-0.25, -0.13)]), K)
        p.paint(sd_box(X, Y, 0, 0.06, 0.37, 0.055), K)
    elif k in (19, 20):                           # bend to the left / right: bold bent arrow
        q = p if k == 20 else Mirror(p)
        rc = 0.17
        pts = [(-0.17, 0.30), (-0.17, -0.12 + rc)] + arc(-0.17 + rc, -0.12 + rc, rc, rc, 180, 90, 8) + [(0.09, -0.12)]
        q.paint(sd_line(q.X, q.Y, pts, 0.09), K)
        q.paint(sd_poly(q.X, q.Y, arrow_head(0.34, -0.12, 1, 0, 0.27, 0.19)), K)
    elif k == 21:                                 # double bend, first to the left
        pts = [(0.15, 0.32), (0.15, 0.10), (-0.10, -0.10), (-0.10, -0.27)]
        p.paint(sd_line(X, Y, pts, 0.085), K)
        p.paint(sd_poly(X, Y, arrow_head(-0.10, -0.52, 0, -1, 0.26, 0.18)), K)
    elif k == 22:                                 # bumpy road
        p.paint(sd_box(X, Y, 0, 0.20, 0.43, 0.05), K)
        for bx in (-0.2, 0.2):
            p.paint(np.maximum(sd_ellipse(X, Y, bx, 0.16, 0.16, 0.15), Y - 0.16), K)
    elif k == 23:                                 # slippery road: car above two wavy tracks
        car_rear(p, 0.0, -0.20, 0.50, 0.42, K)
        for s in (-1, 1):
            x0 = 0.165 * s
            pts = [(x0, 0.03), (x0 - 0.08, 0.12), (x0 + 0.06, 0.22), (x0 - 0.04, 0.32)]
            p.paint(sd_line(X, Y, pts, 0.04), K)
    elif k == 24:                                 # road narrows on the right
        p.paint(sd_line(X, Y, [(-0.11, -0.40), (-0.11, 0.31)], 0.05), K)
        p.paint(sd_line(X, Y, [(0.06, -0.40), (0.06, -0.12), (0.24, 0.08), (0.24, 0.31)], 0.05), K)
    elif k == 25:                                 # road works: digging man and heap
        p.paint(sd_circle(X, Y, -0.04, -0.34, 0.075), K)
        p.paint(sd_line(X, Y, [(-0.07, -0.22), (-0.15, 0.04)], 0.065), K)
        p.paint(sd_line(X, Y, [(-0.15, 0.04), (-0.27, 0.31)], 0.04), K)
        p.paint(sd_line(X, Y, [(-0.15, 0.04), (-0.03, 0.31)], 0.04), K)
        p.paint(sd_line(X, Y, [(-0.06, -0.16), (0.10, 0.02), (0.22, 0.18)], 0.03), K)
        p.paint(np.maximum(sd_ellipse(X, Y, 0.30, 0.34, 0.21, 0.19), Y - 0.34), K)
    elif k == 26:                                 # traffic lights
        p.paint(sd_box(X, Y, 0, -0.07, 0.13, 0.39, 0.05), K)
        p.paint(sd_circle(X, Y, 0, -0.33, 0.105), TL_RED)
        p.paint(sd_circle(X, Y, 0, -0.07, 0.105), TL_YEL)
        p.paint(sd_circle(X, Y, 0, 0.19, 0.105), TL_GRN)
    elif k == 27:                                 # pedestrian
        person(p, 0.0, -0.03, 0.95, K, lean=0.02, stride=0.17)
    elif k == 28:                                 # children: a taller and a smaller figure
        person(p, 0.13, 0.0, 0.85, K, lean=0.03, stride=0.15)
        person(p, -0.19, 0.06, 0.66, K, lean=0.03, stride=0.15)
    elif k == 29:                                 # bicycle (front wheel to the left)
        for wx in (-0.25, 0.25):
            p.paint(np.abs(sd_circle(X, Y, wx, 0.16, 0.14)) - 0.025, K)
        frame = [(-0.25, 0.16), (-0.16, -0.10), (0.07, -0.10), (0.25, 0.16), (0.0, 0.16), (-0.16, -0.10)]
        p.paint(sd_line(X, Y, frame, 0.022), K)
        p.paint(sd_line(X, Y, [(0.0, 0.16), (0.08, -0.16)], 0.022), K)
        p.paint(sd_line(X, Y, [(0.02, -0.17), (0.14, -0.17)], 0.03), K)
        p.paint(sd_line(X, Y, [(-0.16, -0.10), (-0.19, -0.21), (-0.09, -0.23)], 0.022), K)
    elif k == 30:                                 # ice or snow: snow crystal
        c0 = (0.0, -0.02)
        for a in (90, 30, -30):
            t = np.radians(a)
            dx, dy = 0.30 * np.cos(t), -0.30 * np.sin(t)
            p.paint(sd_line(X, Y, [(c0[0] - dx, c0[1] - dy), (c0[0] + dx, c0[1] + dy)], 0.03), K)
            for s in (-1, 1):
                bx, by = c0[0] + s * dx * 0.62, c0[1] + s * dy * 0.62
                for q in (-1, 1):
                    t2 = t + q * np.radians(55)
                    ex, ey = bx + s * 0.11 * np.cos(t2), by - s * 0.11 * np.sin(t2)
                    p.paint(sd_line(X, Y, [(bx, by), (ex, ey)], 0.024), K)
    elif k == 31:                                 # wild animals: deer leaping to the left
        p.paint(sd_line(X, Y, [(-0.10, -0.10), (0.18, 0.02)], 0.085), K)             # body
        p.paint(sd_line(X, Y, [(-0.12, -0.13), (-0.23, -0.30)], 0.045), K)           # neck
        p.paint(sd_line(X, Y, [(-0.23, -0.31), (-0.34, -0.27)], 0.04), K)            # head
        p.paint(sd_line(X, Y, [(-0.21, -0.34), (-0.19, -0.49)], 0.02), K)            # antlers
        p.paint(sd_line(X, Y, [(-0.20, -0.41), (-0.10, -0.46)], 0.02), K)
        p.paint(sd_line(X, Y, [(-0.09, -0.05), (-0.24, 0.07), (-0.33, 0.04)], 0.028), K)   # fore legs
        p.paint(sd_line(X, Y, [(0.18, 0.03), (0.30, 0.13), (0.41, 0.29)], 0.03), K)        # hind legs
        p.paint(sd_line(X, Y, [(0.12, 0.06), (0.22, 0.20), (0.30, 0.30)], 0.026), K)
    else:
        raise ValueError(k)
    return cv


# ----------------------------------------------------------------------------------------------------------------------
# 2. Poses: placing a drawing into a 32 x 32 image
# ----------------------------------------------------------------------------------------------------------------------
SS = 4   # supersampling per pixel


def bilinear(img, u, v, fill):
    """img (C, NC, NC) or (NC, NC) sampled at canonical coordinates u, v (arrays); outside -> fill"""
    fx = (u + CEXT) / CPX - 0.5
    fy = (v + CEXT) / CPX - 0.5
    ok = (fx >= 0) & (fx <= NC - 1) & (fy >= 0) & (fy <= NC - 1)
    fx = np.clip(fx, 0, NC - 1.001)
    fy = np.clip(fy, 0, NC - 1.001)
    x0 = fx.astype(int)
    y0 = fy.astype(int)
    ax, ay = fx - x0, fy - y0
    if img.ndim == 2:
        img = img[None]
    out = (img[:, y0, x0] * (1 - ax) * (1 - ay) + img[:, y0, x0 + 1] * ax * (1 - ay)
           + img[:, y0 + 1, x0] * (1 - ax) * ay + img[:, y0 + 1, x0 + 1] * ax * ay)
    fillv = np.asarray(fill, float).reshape(-1, *([1] * u.ndim))
    return np.where(ok[None], out, fillv)


def gauss1d(sigma, radius):
    x = np.arange(-radius, radius + 1)
    g = np.exp(-x * x / (2.0 * sigma * sigma))
    return g / g.sum()


def blur_hw(a, sigma):
    """separable Gaussian blur over the last two axes with edge replication"""
    r = max(1, int(np.ceil(3 * sigma)))
    g = gauss1d(sigma, r)
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(r, r), (r, r)], mode="edge")
    out = sum(g[i] * p[..., i:i + a.shape[-2], :] for i in range(2 * r + 1))
    out = sum(g[i] * out[..., :, i:i + a.shape[-1]] for i in range(2 * r + 1))
    return out


SURROUND = {            # (grey level around the sign, brightness of the sign itself)
    "mid": (0.62, 1.0),     # neutral surround
    "dark": (0.15, 1.0),    # sign in front of something dark (trees, buildings in shade)
    "bright": (0.95, 0.55),  # sign in shade in front of a bright sky
}


def place(cv, d, aspect, rot_deg, sx, sy, surround="mid", blur=0.5):
    """image (3, 32, 32) of the drawing with bounding box width d px, height d * aspect px, rotated, centre shifted
    by (sx, sy) px from the image centre; also the signed distance (px) to the sign outline at each pixel centre"""
    hx, hy = d / 2.0, d * aspect / 2.0
    t = np.radians(rot_deg)
    c, s = np.cos(t), np.sin(t)
    bg, gain = SURROUND[surround]

    def to_canon(xs, ys):
        px, py = xs - (16.0 + sx), ys - (16.0 + sy)
        return (c * px + s * py) / hx, (-s * px + c * py) / hy
    g = (np.arange(32 * SS) + 0.5) / SS
    xs, ys = np.meshgrid(g, g)
    u, v = to_canon(xs, ys)
    fg = bilinear(cv.img, u, v, 0.0)
    al = bilinear(cv.alpha, u, v, 0.0)
    hi = gain * fg + (1.0 - al) * bg
    hi = blur_hw(hi, blur * SS)
    img = hi.reshape(3, 32, SS, 32, SS).mean(axis=(2, 4))
    g1 = np.arange(32) + 0.5
    xs, ys = np.meshgrid(g1, g1)
    u, v = to_canon(xs, ys)
    dist = bilinear(cv.outline, u, v, 1.0)[0] * (hx + hy) / 2.0
    return img, dist


# ----------------------------------------------------------------------------------------------------------------------
# 3. The network: fixed front end (formula filters), then templates
# ----------------------------------------------------------------------------------------------------------------------
N_ORI = 8
KAPPA = 1.0          # orientation tuning: response falls to zero 45 degrees away from the preferred direction
SIGMA_G = 0.8        # Gaussian derivative scale (pixels)
FLOOR_O = 0.010      # semi-saturation of the local normalisation of edge maps (see method.md for the reasoning)
FLOOR_I = 0.10       # the same for local brightness
FLOOR_C = 0.10       # the same for colour opponent maps
EPS = 1e-5           # epsilon inside the norm operation
W_ORI, W_INT, W_COL = 1.0, 0.7, 0.5     # weights of the three kinds of evidence

CELLS = 16           # 16 x 16 cells of 2 px stride
NB = 7               # 7 x 7 overlapping blocks of 4 x 4 cells


def tent4():
    k = np.array([1.0, 3.0, 3.0, 1.0]) / 8.0
    return np.outer(k, k)


def front_end_weights():
    W = {}
    # luminance
    W["lum.W"] = np.full((1, 3, 1, 1), 1.0 / 3.0)
    W["lum.b"] = np.zeros(1)
    # Gaussian derivative filters in 8 directions
    r = 2
    x = np.arange(-r, r + 1)
    g = np.exp(-x * x / (2 * SIGMA_G ** 2))
    g /= g.sum()
    dg = -x / SIGMA_G ** 2 * g
    gx = np.outer(g, dg)       # derivative along x (columns)
    gy = np.outer(dg, g)       # derivative along y (rows)
    K = np.zeros((N_ORI, 1, 5, 5))
    for j in range(N_ORI):
        a = 2 * np.pi * j / N_ORI
        # correlation with a kernel that is odd gives minus the derivative; flip the sign so that the output is the
        # derivative of brightness along direction a
        K[j, 0] = -(np.cos(a) * gx + np.sin(a) * gy)
    W["grad.W"] = K
    W["grad.b"] = np.zeros(N_ORI)
    # sharpen the orientation tuning: d_j - kappa * (d_{j+90} + d_{j-90})
    S = np.zeros((N_ORI, N_ORI, 1, 1))
    q = N_ORI // 4
    for j in range(N_ORI):
        S[j, j, 0, 0] = 1.0
        S[j, (j + q) % N_ORI, 0, 0] -= KAPPA
        S[j, (j - q) % N_ORI, 0, 0] -= KAPPA
    W["sharp.W"] = S
    W["sharp.b"] = np.zeros(N_ORI)
    # pooling into cells (tent window 4 x 4, stride 2); the gain sets the floor of the following normalisation
    t4 = tent4()
    P = np.zeros((N_ORI, N_ORI, 4, 4))
    for j in range(N_ORI):
        P[j, j] = t4 * (np.sqrt(EPS) / FLOOR_O)
    W["ocell.W"] = P
    W["ocell.b"] = np.zeros(N_ORI)
    W["icell.W"] = (t4 * (np.sqrt(EPS) / FLOOR_I))[None, None]
    W["icell.b"] = np.zeros(1)
    opp = np.array([[1.0, -1.0, 0.0], [-1.0, -1.0, 2.0]])
    opp /= np.linalg.norm(opp, axis=1, keepdims=True)
    C = np.zeros((2, 3, 4, 4))
    for o in range(2):
        for i in range(3):
            C[o, i] = t4 * opp[o, i] * (np.sqrt(EPS) / FLOOR_C)
    W["ccell.W"] = C
    W["ccell.b"] = np.zeros(2)
    # gather cells into blocks (one block per row) and blend the normalised blocks back
    for name, ch in (("o", N_ORI), ("i", 1)):
        nin = ch * CELLS * CELLS
        per = ch * 16
        G = np.zeros((NB * NB * per, nin))
        Sc = np.zeros((nin, NB * NB * per))
        wt = np.array([0.25, 0.75, 0.75, 0.25])
        wsum = np.zeros((CELLS, CELLS))
        for by in range(NB):
            for bx in range(NB):
                for iy in range(4):
                    for ix in range(4):
                        wsum[2 * by + iy, 2 * bx + ix] += wt[iy] * wt[ix]
        for by in range(NB):
            for bx in range(NB):
                b = by * NB + bx
                for c in range(ch):
                    for iy in range(4):
                        for ix in range(4):
                            yy, xx = 2 * by + iy, 2 * bx + ix
                            src = c * CELLS * CELLS + yy * CELLS + xx
                            row = b * per + c * 16 + iy * 4 + ix
                            G[row, src] = 1.0
                            Sc[src, row] = wt[iy] * wt[ix] / wsum[yy, xx]
        W[name + "gather.W"] = G
        W[name + "gather.b"] = np.zeros(G.shape[0])
        W[name + "blend.W"] = Sc
        W[name + "blend.b"] = np.zeros(nin)
    return {k: v.astype(F32) for k, v in W.items()}


def front_end_layers():
    L = []

    def add(name, op, inputs, **kw):
        d = {"name": name, "op": op, "inputs": inputs}
        d.update(kw)
        L.append(d)
    add("lum", "conv", ["image"], out=1, kernel=[1, 1], stride=1, padding=0)
    add("lumn", "norm", ["lum"])
    add("rgbn", "norm", ["image"])
    # edge orientation maps
    add("grad", "conv", ["lumn"], out=N_ORI, kernel=[5, 5], stride=1, padding=0)
    add("gradr", "relu", ["grad"])
    add("sharp", "conv", ["gradr"], out=N_ORI, kernel=[1, 1], stride=1, padding=0)
    add("ori", "relu", ["sharp"])
    add("ocell", "conv", ["ori"], out=N_ORI, kernel=[4, 4], stride=2, padding=3)
    add("ogather", "dense", ["ocell"], out=NB * NB * N_ORI * 16)
    add("oblock", "reshape", ["ogather"], shape=[NB * NB, N_ORI * 16, 1])
    add("oblockn", "norm", ["oblock"])
    add("oblend", "dense", ["oblockn"], out=N_ORI * CELLS * CELLS)
    add("omap", "reshape", ["oblend"], shape=[N_ORI, CELLS, CELLS])
    # local brightness
    add("icell", "conv", ["lumn"], out=1, kernel=[4, 4], stride=2, padding=1)
    add("igather", "dense", ["icell"], out=NB * NB * 16)
    add("iblock", "reshape", ["igather"], shape=[NB * NB, 16, 1])
    add("iblockn", "norm", ["iblock"])
    add("iblend", "dense", ["iblockn"], out=CELLS * CELLS)
    add("imap", "reshape", ["iblend"], shape=[1, CELLS, CELLS])
    # colour opponents
    add("ccell", "conv", ["rgbn"], out=2, kernel=[4, 4], stride=2, padding=1)
    add("cflat", "reshape", ["ccell"], shape=[1, 2 * CELLS, CELLS])
    add("cflatn", "norm", ["cflat"])
    add("cmap", "reshape", ["cflatn"], shape=[2, CELLS, CELLS])
    return L


# ---- numpy interpreter of the same operations (used here only to turn drawings into templates) -----------------------
def op_conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def op_norm(x):
    mu = x.mean(axis=(2, 3), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + EPS)


def run_layers(layers, weights, images):
    vals = {"image": images}
    for L in layers:
        ins = [vals[i] for i in L["inputs"]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = op_conv(ins[0], weights[name + ".W"], weights[name + ".b"], L.get("stride", 1), L.get("padding", 0))
        elif op == "dense":
            vals[name] = ins[0].reshape(len(ins[0]), -1) @ weights[name + ".W"].T + weights[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op == "reshape":
            vals[name] = ins[0].reshape((len(ins[0]),) + tuple(L["shape"]))
        elif op == "norm":
            vals[name] = op_norm(ins[0])
        else:
            raise ValueError(op)
    return vals


# ----------------------------------------------------------------------------------------------------------------------
# 4. Templates
# ----------------------------------------------------------------------------------------------------------------------
SIZES = [19.0, 20.5, 22.0, 23.5, 25.0, 26.5, 28.0]                       # width of the sign's bounding box in pixels
SUBSHIFTS = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]         # the conv adds shifts of -2, 0, +2 pixels
# appearance variants: (rotation in degrees, height / width, surround, extra blur in pixels, proportion variant)
LOOKS = [
    (0.0, 1.0, "mid", 0.0, 0),
    (-8.0, 1.0, "mid", 0.0, 0),
    (8.0, 1.0, "mid", 0.0, 0),
    (0.0, 0.94, "mid", 0.0, 0),
    (0.0, 1.06, "mid", 0.0, 0),
    (0.0, 1.0, "mid", 0.6, 0),
    (0.0, 1.0, "dark", 0.0, 0),
    (0.0, 1.0, "bright", 0.0, 0),
    (0.0, 1.0, "mid", 0.0, 1),
    (0.0, 1.0, "mid", 0.0, 2),
]
POSES = [(d, look, sx, sy) for d in SIZES for look in LOOKS for (sx, sy) in SUBSHIFTS]
V = len(POSES)
NCLS = 43


def base_blur(d):
    """small signs come from small photos that were enlarged, so they are softer; large ones were shrunk and are sharp"""
    return float(np.clip(0.8 - 0.05 * (d - 19.0), 0.5, 0.8))


def smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def pool_cells(m):
    """(n, 32, 32) pixel maps -> (n, 16, 16) cell maps with the same tent window as the network"""
    t4 = tent4()
    p = np.pad(m, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((m.shape[0], CELLS, CELLS))
    for dy in range(4):
        for dx in range(4):
            out += t4[dy, dx] * p[:, dy:dy + 2 * CELLS:2, dx:dx + 2 * CELLS:2]
    return out


def unit(t, wmask):
    """one template, normalised by itself alone: zero weighted mean over its own mask, unit length"""
    mu = t.sum() / max(wmask.sum() * t.shape[0], 1e-9)
    t = t - mu * wmask[None]
    n = np.sqrt((t * t).sum())
    return t / n if n > 1e-6 else t * 0.0


def render_class(k, poses=None):
    """all drawings of class k: images (n, 3, 32, 32), distance-to-outline maps (n, 32, 32), surround names"""
    poses = POSES if poses is None else poses
    canv = {}
    imgs = np.empty((len(poses), 3, 32, 32))
    dist = np.empty((len(poses), 32, 32))
    for i, (d, (rot, asp, sur, xblur, dv), sx, sy) in enumerate(poses):
        if dv not in canv:
            canv[dv] = draw_class(k, dv)
        imgs[i], dist[i] = place(canv[dv], d, asp, rot, sx, sy, sur, base_blur(d) + xblur)
    return imgs, dist, [p[1][2] for p in poses]


def build_templates(layers, weights, classes=range(NCLS), poses=None):
    """returns template kernels (T, 8, 16, 16), (T, 1, 16, 16), (T, 2, 16, 16) ordered class by class"""
    w64 = {k: v.astype(np.float64) for k, v in weights.items()}
    TO, TI, TC = [], [], []
    for k in classes:
        imgs, dist, sur = render_class(k, poses)
        vals = run_layers(layers, w64, imgs)
        O, I, C = vals["omap"], vals["imap"], vals["cmap"]
        # masks from the distance to the sign's outline (pixels, negative inside)
        w_in = pool_cells(smooth01((-dist - 0.5) / 1.5))            # inside, away from the outline
        w_all = pool_cells(smooth01((1.75 - dist) / 1.0))           # inside plus a narrow band outside the outline
        w_band = np.clip(w_all - w_in, 0, None)                     # band around the outline
        w_int = pool_cells(smooth01((-dist - 1.0) / 1.5))           # inside, clear of the surround
        w_col = pool_cells(smooth01((-dist + 0.0) / 1.0))           # whole sign
        # neutral surround: what is behind the sign is unknown, so the outline edge may have either polarity there:
        # use the template's own response with opposite directions merged
        Osym = 0.5 * (O + np.roll(O, N_ORI // 2, axis=1))
        for i in range(len(sur)):
            if sur[i] == "mid":
                TO.append(unit(w_in[i][None] * O[i] + w_band[i][None] * Osym[i], w_all[i]))
                TI.append(unit(I[i] * w_int[i][None], w_int[i]))
            else:                                                   # surround drawn explicitly: keep the outline edge
                TO.append(unit(w_all[i][None] * O[i], w_all[i]))
                TI.append(unit(I[i] * w_col[i][None], w_col[i]))
            TC.append(unit(C[i] * w_col[i][None], w_col[i]))
    return np.array(TO), np.array(TI), np.array(TC)


def build(out_dir):
    layers = front_end_layers()
    weights = front_end_weights()
    TO, TI, TC = build_templates(layers, weights)
    T = TO.shape[0]
    assert T == NCLS * V
    weights["mo.W"] = (W_ORI * TO).astype(F32)
    weights["mo.b"] = np.zeros(T, F32)
    weights["mi.W"] = (W_INT * TI).astype(F32)
    weights["mi.b"] = np.zeros(T, F32)
    weights["mc.W"] = (W_COL * TC).astype(F32)
    weights["mc.b"] = np.zeros(T, F32)
    layers = layers + [
        {"name": "mo", "op": "conv", "inputs": ["omap"], "out": T, "kernel": [CELLS, CELLS], "stride": 1, "padding": 1},
        {"name": "mi", "op": "conv", "inputs": ["imap"], "out": T, "kernel": [CELLS, CELLS], "stride": 1, "padding": 1},
        {"name": "mc", "op": "conv", "inputs": ["cmap"], "out": T, "kernel": [CELLS, CELLS], "stride": 1, "padding": 1},
        {"name": "match", "op": "add", "inputs": ["mo", "mi", "mc"]},
        {"name": "best_shift", "op": "maxpool", "inputs": ["match"], "size": [3, 3], "stride": [3, 3]},
        {"name": "by_class", "op": "reshape", "inputs": ["best_shift"], "shape": [1, NCLS, V]},
        {"name": "best_pose", "op": "maxpool", "inputs": ["by_class"], "size": [1, V], "stride": [1, V]},
        {"name": "scores", "op": "reshape", "inputs": ["best_pose"], "shape": [NCLS]},
    ]
    with open(out_dir + "/network.json", "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(out_dir + "/weights.npz", **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
