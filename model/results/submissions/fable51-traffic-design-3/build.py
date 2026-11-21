#!/usr/bin/env python3
"""
build.py  --  writes network.json and weights.npz for a hand-designed German traffic sign classifier.

Nothing here is learned from data.  The network is:
  1. joint brightness / contrast normalisation of the RGB image (reshape -> norm -> reshape),
  2. a fixed 5x5 filter bank: 8 signed Gaussian-derivative orientation channels on luminance and
     5 colour channels (red, blue, yellow, bright, dark) from hand-chosen colour formulas.  A second norm
     layer fed with tiny values only subtracts each channel's image mean (so colour casts cancel); every
     channel is then half-wave rectified with a threshold and clipped (two relus combined by the next conv),
  3. a Gaussian smoothing conv (also applies the clipping combination, per-channel gains and adds one
     coarsely smoothed "dark mass" channel),
  4. two big convs whose kernels are templates: every German sign class is drawn from general knowledge of
     its design (ring widths, digit strokes, pictograms) at several sizes and two palettes, passed through the
     same filters, masked to the sign area and normalised by its own norm alone.  One conv holds the whole
     sign, the other the pictogram alone (drawn on a plain canvas of the interior colour); the pictogram score
     may shift by one extra pixel, is clipped to a small range and added to the whole-sign score,
  5. max over positions and over the size / palette variants -> 43 class scores.
Usage: python3 build.py OUT_DIR
"""
import json, os, sys
import numpy as np

# ----------------------------------------------------------------------------------------------
# rendering: 32x32 frame, supersampled UP times, coordinates in frame pixels (y down)
# ----------------------------------------------------------------------------------------------
UP = 8
N = 32
S = N * UP
_yy, _xx = np.mgrid[0:S, 0:S]
X = ((_xx + 0.5) / UP).astype(np.float64)
Y = ((_yy + 0.5) / UP).astype(np.float64)


def m_disc(cx, cy, r):
    return ((X - cx) ** 2 + (Y - cy) ** 2 <= r * r).astype(np.float32)


def m_ring(cx, cy, r_out, r_in):
    d2 = (X - cx) ** 2 + (Y - cy) ** 2
    return ((d2 <= r_out ** 2) & (d2 >= r_in ** 2)).astype(np.float32)


def m_poly(pts):
    pts = np.asarray(pts, float)
    n = len(pts)
    inside = np.zeros(X.shape, bool)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if y0 == y1:
            continue
        cond = (y0 > Y) != (y1 > Y)
        xint = x0 + (Y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cond & (X < xint)
    return inside.astype(np.float32)


def m_seg(x0, y0, x1, y1, w):
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 <= 0:
        return m_disc(x0, y0, w / 2)
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1)
    px, py = x0 + t * dx, y0 + t * dy
    return ((X - px) ** 2 + (Y - py) ** 2 <= (w / 2) ** 2).astype(np.float32)


def m_arc(cx, cy, r, a0, a1, w):
    """arc of radius r between angles a0..a1 (degrees, screen coordinates, y down), stroke w, round caps"""
    lo, hi = min(a0, a1), max(a0, a1)
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    ang = np.degrees(np.arctan2(Y - cy, X - cx))
    a = (ang - lo) % 360.0
    m = ((np.abs(d - r) <= w / 2) & (a <= hi - lo)).astype(np.float32)
    for aa in (a0, a1):
        ex, ey = cx + r * np.cos(np.radians(aa)), cy + r * np.sin(np.radians(aa))
        m = np.maximum(m, m_disc(ex, ey, w / 2))
    return m


def m_rect(x0, y0, x1, y1):
    return ((X >= min(x0, x1)) & (X <= max(x0, x1)) & (Y >= min(y0, y1)) & (Y <= max(y0, y1))).astype(np.float32)


def m_union(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = np.maximum(out, m)
    return out


def m_halfdisc_up(cx, cy, r):
    return m_disc(cx, cy, r) * (Y <= cy)


# colours of sign paints (general knowledge; mid-grey surroundings).  Two palettes: a sign in good light,
# and a dim one as in under-exposed frames where the red border is as dark as the surroundings.
PALETTES = {
    "normal": dict(RED=(0.72, 0.08, 0.10), WHITE=(0.92, 0.92, 0.92), BLACK=(0.06, 0.06, 0.06), BLUE=(0.08, 0.22, 0.62),
                   YELLOW=(0.95, 0.72, 0.05), GREY=(0.45, 0.45, 0.45), BG=(0.42, 0.42, 0.42), RIM=(0.25, 0.25, 0.25), K=1.0),
    "dim": dict(RED=(0.22, 0.05, 0.06), WHITE=(0.50, 0.50, 0.50), BLACK=(0.03, 0.03, 0.03), BLUE=(0.04, 0.09, 0.26),
                YELLOW=(0.45, 0.33, 0.04), GREY=(0.28, 0.28, 0.28), BG=(0.20, 0.20, 0.20), RIM=(0.12, 0.12, 0.12), K=0.5),
}
C = dict(PALETTES["normal"])


def col(c):
    """palette colour by name, or an explicit rgb tuple scaled by the palette's brightness factor"""
    if isinstance(c, str):
        return C[c]
    return tuple(v * C["K"] for v in c)


RED, WHITE, BLACK, BLUE, YELLOW, GREY, BG = "RED", "WHITE", "BLACK", "BLUE", "YELLOW", "GREY", "BG"
PICTO_ONLY = [False]     # when True, only the pictogram is drawn on a canvas of the sign's interior colour


class Canvas:
    def __init__(self, bg):
        self.img = np.empty((S, S, 3), np.float32)
        self.img[:] = np.asarray(col(bg), np.float32)
        self.sign = np.zeros((S, S), np.float32)       # union of everything drawn as part of the sign
        self.interior = np.zeros((S, S), np.float32)   # region inside the border where the pictogram lives

    def paint(self, mask, color, part_of_sign=True):
        m = mask[..., None]
        self.img = self.img * (1 - m) + np.asarray(col(color), np.float32) * m
        if part_of_sign:
            self.sign = np.maximum(self.sign, mask)

    def set_interior(self, mask):
        self.interior = np.maximum(self.interior, mask)

    def down(self):
        img = self.img.reshape(N, UP, N, UP, 3).mean(axis=(1, 3))
        sign = self.sign.reshape(N, UP, N, UP).mean(axis=(1, 3))
        inter = self.interior.reshape(N, UP, N, UP).mean(axis=(1, 3))
        return img.transpose(2, 0, 1).astype(np.float32), sign.astype(np.float32), inter.astype(np.float32)


# ----------------------------------------------------------------------------------------------
# glyphs: digits and letters as strokes in a unit-height box (DIN 1451 like shapes)
# ----------------------------------------------------------------------------------------------
def glyph(ch, x0, y0, h, t=0.17, narrow=False):
    """mask of character ch with top-left corner (x0, y0) and height h; returns (mask, advance width)"""
    wd = {'1': 0.40, 'S': 0.60, 'T': 0.60, 'O': 0.60, 'P': 0.58}.get(ch, 0.56)
    if narrow:
        wd = {'1': 0.34}.get(ch, 0.50)
    T = t * h
    parts = []

    def P(x, y):
        return (x0 + x * h, y0 + y * h)

    def seg(xa, ya, xb, yb, w=None):
        parts.append(m_seg(*P(xa, ya), *P(xb, yb), (w or t) * h))

    def arc(cx, cy, r, a0, a1, w=None):
        parts.append(m_arc(*P(cx, cy), r * h, a0, a1, (w or t) * h))

    cx = wd / 2
    if ch in ('0', 'O'):
        r = (wd - t) / 2
        arc(cx, t / 2 + r, r, 180, 360)
        arc(cx, 1 - t / 2 - r, r, 0, 180)
        seg(t / 2, t / 2 + r, t / 2, 1 - t / 2 - r)
        seg(wd - t / 2, t / 2 + r, wd - t / 2, 1 - t / 2 - r)
    elif ch == '1':
        seg(wd - t / 2, t / 2, wd - t / 2, 1 - t / 2)
        seg(wd - t / 2, t / 2, 0.05, 0.30)
    elif ch == '2':
        r = (wd - t) / 2
        arc(cx, t / 2 + r, r, 180, 395)
        ex, ey = cx + r * np.cos(np.radians(35)), t / 2 + r + r * np.sin(np.radians(35))
        seg(ex, ey, t / 2, 1 - t / 2)
        seg(t / 2, 1 - t / 2, wd - t / 2, 1 - t / 2)
    elif ch == '3':
        r = (wd - t) / 2 * 0.95
        arc(cx, 0.265, r, 200, 440)
        arc(cx, 0.735, r, -80, 160)
        seg(cx - 0.1, 0.5, cx + 0.03, 0.5)
    elif ch == '5':
        seg(0.10, t / 2, wd - 0.06, t / 2)
        seg(0.11, t / 2, 0.11, 0.47)
        arc(cx, 0.68, (wd - t) / 2 + 0.04, -130, 150)
    elif ch == '6':
        arc(cx, 0.715, (wd - t) / 2 + 0.005, 0, 360)
        arc(cx + 0.05, 0.41, (wd - t) / 2 + 0.05, 180, 300)
        seg(t / 2, 0.41, t / 2, 0.715)
    elif ch == '7':
        seg(0.07, t / 2, wd - 0.07, t / 2)
        seg(wd - 0.07, t / 2, 0.20, 1 - t / 2)
    elif ch == '8':
        arc(cx, 0.265, (wd - t) / 2 * 0.92, 0, 360)
        arc(cx, 0.72, (wd - t) / 2 + 0.005, 0, 360)
    elif ch == 'S':
        arc(0.3, 0.26, 0.19, 120, 340)
        arc(0.3, 0.74, 0.19, 300, 560)
    elif ch == 'T':
        seg(0.0, t / 2, wd, t / 2)
        seg(wd / 2, t / 2, wd / 2, 1 - t / 2)
    elif ch == 'P':
        seg(t / 2, t / 2, t / 2, 1 - t / 2)
        seg(t / 2, t / 2, 0.33, t / 2)
        arc(0.33, 0.31, 0.21, -90, 90)
        seg(0.33, 0.52, t / 2, 0.52)
    else:
        raise ValueError(ch)
    return m_union(*parts), wd * h


def text_mask(txt, cx, cy, h, gap=0.10, t=0.17, narrow=False):
    """centred string; cx, cy centre of the text block in frame pixels, h digit height in pixels"""
    widths = [glyph(c, 0, 0, 1.0, t, narrow)[1] for c in txt]
    total = sum(widths) * h + gap * h * (len(txt) - 1)
    x = cx - total / 2
    ms = []
    for c, w in zip(txt, widths):
        m, _ = glyph(c, x, cy - h / 2, h, t, narrow)
        ms.append(m)
        x += w * h + gap * h
    return m_union(*ms)


# ----------------------------------------------------------------------------------------------
# sign parts
# ----------------------------------------------------------------------------------------------
TRI_SY = 1.13     # the crops are resized to a square without keeping the aspect ratio: triangles get taller


def tri_pts(cx, yc, R, up=True, sy=1.0):
    """equilateral triangle with centroid (cx, yc) and circumradius R, stretched vertically by sy"""
    pts = []
    for k in range(3):
        a = np.radians(-90 + 120 * k) if up else np.radians(90 + 120 * k)
        pts.append((cx + R * np.cos(a), yc + sy * R * np.sin(a)))
    return pts


def draw_warning_triangle(cv, cx, cy, D, up=True, border=0.075):
    """red bordered white triangle whose bounding box is D wide (and nearly D tall after the stretch);
    returns (centroid x, centroid y, white interior mask)"""
    R = D / np.sqrt(3)
    yc = cy + 0.144 * D * TRI_SY if up else cy - 0.144 * D * TRI_SY
    outer = m_poly(tri_pts(cx, yc, R, up, TRI_SY))
    inner = m_poly(tri_pts(cx, yc, R - 2 * border * D, up, TRI_SY))
    core = m_poly(tri_pts(cx, yc, R - 2 * border * D - 4.0, up, TRI_SY))   # 2 px inside the border
    if not PICTO_ONLY[0]:
        cv.paint(outer, RED)
        cv.paint(inner, WHITE)
    cv.set_interior(core)
    return cx, yc, inner


def draw_prohibition_disc(cv, cx, cy, D, ring=0.15):
    if not PICTO_ONLY[0]:
        cv.paint(m_disc(cx, cy, 0.5 * D), WHITE)
        cv.paint(m_ring(cx, cy, 0.475 * D, (0.475 - ring) * D), RED)
    cv.set_interior(m_disc(cx, cy, (0.475 - ring) * D - 2.0))


def draw_blue_disc(cv, cx, cy, D):
    if not PICTO_ONLY[0]:
        cv.paint(m_disc(cx, cy, 0.5 * D), WHITE)
        cv.paint(m_disc(cx, cy, 0.47 * D), BLUE)
    cv.set_interior(m_disc(cx, cy, 0.40 * D))
    return m_disc(cx, cy, 0.45 * D)


def draw_end_disc(cv, cx, cy, D):
    if not PICTO_ONLY[0]:
        cv.paint(m_disc(cx, cy, 0.5 * D), "RIM")
        cv.paint(m_disc(cx, cy, 0.48 * D), WHITE)
    cv.set_interior(m_disc(cx, cy, 0.42 * D))


def draw_end_stripes(cv, cx, cy, D):
    """the group of thin diagonal lines rising from lower left to upper right; at 32 pixels it reads as one
    dark grey band, so it is drawn as a band, clipped to the disc"""
    clip = m_disc(cx, cy, 0.47 * D)
    band = m_seg(cx - 0.5 * D, cy + 0.5 * D, cx + 0.5 * D, cy - 0.5 * D, 0.20 * D)
    cv.paint(band * clip, (0.30, 0.30, 0.30))


def car_front(cx, cy, D):
    body = m_rect(cx - 0.12 * D, cy - 0.03 * D, cx + 0.12 * D, cy + 0.09 * D)
    roof = m_rect(cx - 0.08 * D, cy - 0.11 * D, cx + 0.08 * D, cy - 0.03 * D)
    return m_union(body, roof)


def truck_front(cx, cy, D):
    return m_rect(cx - 0.12 * D, cy - 0.15 * D, cx + 0.12 * D, cy + 0.13 * D)


def truck_side(cx, cy, D):
    cab = m_rect(cx - 0.32 * D, cy - 0.06 * D, cx - 0.11 * D, cy + 0.11 * D)
    box = m_rect(cx - 0.11 * D, cy - 0.17 * D, cx + 0.32 * D, cy + 0.11 * D)
    w1 = m_disc(cx - 0.22 * D, cy + 0.15 * D, 0.05 * D)
    w2 = m_disc(cx + 0.19 * D, cy + 0.15 * D, 0.05 * D)
    return m_union(cab, box, w1, w2)


def arrow_head(tip, base_c, half):
    """triangle with apex tip, base centre base_c, half base width half"""
    tx, ty = tip
    bx, by = base_c
    dx, dy = tx - bx, ty - by
    L = np.hypot(dx, dy)
    px, py = -dy / L, dx / L
    return m_poly([(tx, ty), (bx + px * half, by + py * half), (bx - px * half, by - py * half)])


# ----------------------------------------------------------------------------------------------
# the 43 classes
# ----------------------------------------------------------------------------------------------
SPEED = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}
INTERIOR_COLOUR = {12: YELLOW, 14: RED, 17: RED, 33: BLUE, 34: BLUE, 35: BLUE, 36: BLUE, 37: BLUE, 38: BLUE, 39: BLUE, 40: BLUE}
NO_PICTOGRAM = {12, 13, 15}     # signs whose interior carries nothing to match
TRI_PICTO_SCALE = 1.15          # pictogram unit relative to the triangle's bounding width
BLUE_PICTO_SCALE = 1.20         # arrow unit relative to the blue disc diameter


def render_sign(k, D, cx=16.0, cy=16.0, bg=BG, palette="normal", picto_only=False):
    """draw class k with sign size D (pixels) centred at (cx, cy) in the given palette;
    with picto_only, only the pictogram is drawn on a plain canvas of the sign's interior colour;
    returns (image (3,32,32), sign mask (32,32), interior mask (32,32))"""
    C.clear()
    C.update(PALETTES[palette])
    PICTO_ONLY[0] = bool(picto_only)
    if picto_only:
        bg = INTERIOR_COLOUR.get(k, WHITE)
    cv = Canvas(bg)

    def P(x, y, ox, oy, U):   # pictogram coordinates (units of U) relative to an origin
        return (ox + x * U, oy + y * U)

    if k in SPEED:
        draw_prohibition_disc(cv, cx, cy, D)
        txt = SPEED[k]
        if len(txt) == 2:
            cv.paint(text_mask(txt, cx, cy, 0.52 * D, gap=0.08, t=0.18), BLACK)
        else:
            cv.paint(text_mask(txt, cx, cy, 0.48 * D, gap=0.05, t=0.17, narrow=True), BLACK)
    elif k == 6:    # end of speed limit 80
        draw_end_disc(cv, cx, cy, D)
        cv.paint(text_mask('80', cx, cy, 0.50 * D, gap=0.08, t=0.18), GREY)
        draw_end_stripes(cv, cx, cy, D)
    elif k == 9:    # no passing
        draw_prohibition_disc(cv, cx, cy, D)
        cv.paint(car_front(cx - 0.14 * D, cy, D), RED)
        cv.paint(car_front(cx + 0.14 * D, cy, D), BLACK)
    elif k == 10:   # no passing for trucks
        draw_prohibition_disc(cv, cx, cy, D)
        cv.paint(truck_front(cx - 0.14 * D, cy - 0.01 * D, D), RED)
        cv.paint(car_front(cx + 0.14 * D, cy, D), BLACK)
    elif k == 12:   # priority road
        if not PICTO_ONLY[0]:
            cv.paint(m_poly([(cx, cy - 0.5 * D), (cx + 0.5 * D, cy), (cx, cy + 0.5 * D), (cx - 0.5 * D, cy)]), WHITE)
            cv.paint(m_poly([(cx, cy - 0.37 * D), (cx + 0.37 * D, cy), (cx, cy + 0.37 * D), (cx - 0.37 * D, cy)]), YELLOW)
        cv.set_interior(m_poly([(cx, cy - 0.37 * D + 3), (cx + 0.37 * D - 3, cy), (cx, cy + 0.37 * D - 3), (cx - 0.37 * D + 3, cy)]))
    elif k == 13:   # yield
        draw_warning_triangle(cv, cx, cy, D, up=False, border=0.14)
    elif k == 14:   # stop
        R = 0.5413 * D
        oct_pts = [(cx + R * np.cos(np.radians(22.5 + 45 * i)), cy + R * np.sin(np.radians(22.5 + 45 * i))) for i in range(8)]
        R2 = 0.495 * D
        oct2 = [(cx + R2 * np.cos(np.radians(22.5 + 45 * i)), cy + R2 * np.sin(np.radians(22.5 + 45 * i))) for i in range(8)]
        if not PICTO_ONLY[0]:
            cv.paint(m_poly(oct_pts), WHITE)
            cv.paint(m_poly(oct2), RED)
        cv.set_interior(m_disc(cx, cy, 0.42 * D))
        cv.paint(text_mask('STOP', cx, cy, 0.30 * D, gap=0.10, t=0.2), WHITE)
    elif k == 15:   # no vehicles
        draw_prohibition_disc(cv, cx, cy, D)
    elif k == 16:   # no trucks
        draw_prohibition_disc(cv, cx, cy, D)
        cv.paint(truck_side(cx, cy, D), BLACK)
    elif k == 17:   # no entry
        if not PICTO_ONLY[0]:
            cv.paint(m_disc(cx, cy, 0.5 * D), WHITE)
            cv.paint(m_disc(cx, cy, 0.475 * D), RED)
        cv.set_interior(m_disc(cx, cy, 0.42 * D))
        cv.paint(m_rect(cx - 0.32 * D, cy - 0.10 * D, cx + 0.32 * D, cy + 0.10 * D), WHITE)
    elif k in (11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):   # warning triangles
        ox, oy, inner = draw_warning_triangle(cv, cx, cy, D)
        U = TRI_PICTO_SCALE * D

        def Q(x, y):
            return (ox + x * U, oy + y * U * TRI_SY)

        colour_parts = []     # (mask, colour) for the traffic lights
        if k == 11:   # priority at next junction
            m = m_union(m_seg(*Q(0, -0.22), *Q(0, 0.16), 0.13 * U),
                        m_seg(*Q(-0.18, -0.02), *Q(0.18, -0.02), 0.065 * U))
        elif k == 18:   # general danger
            m = m_union(m_seg(*Q(0, -0.24), *Q(0, 0.03), 0.09 * U), m_disc(*Q(0, 0.13), 0.052 * U))
        elif k in (19, 20):   # bend left / right
            s = 1.0 if k == 19 else -1.0
            w = 0.09 * U
            a0, a1 = (270, 360) if k == 19 else (180, 270)
            m = m_union(m_seg(*Q(0.07 * s, 0.15), *Q(0.07 * s, -0.04), w),
                        m_arc(*Q(-0.03 * s, -0.04), 0.10 * U, a0, a1, w),
                        m_seg(*Q(-0.03 * s, -0.14), *Q(-0.15 * s, -0.14), w))
        elif k == 21:   # double bend, first left
            w = 0.08 * U
            pts = [(0.06, 0.17), (0.06, 0.02), (-0.06, 0.02), (-0.06, -0.14), (0.04, -0.14)]
            m = m_union(*[m_seg(*Q(*pts[i]), *Q(*pts[i + 1]), w) for i in range(len(pts) - 1)])
        elif k == 22:   # bumpy road
            m = m_union(m_seg(*Q(-0.2, 0.12), *Q(0.2, 0.12), 0.045 * U),
                        m_halfdisc_up(*Q(-0.09, 0.12), 0.075 * U), m_halfdisc_up(*Q(0.09, 0.12), 0.075 * U))
        elif k == 23:   # slippery road
            m = m_union(m_seg(*Q(-0.13, -0.02), *Q(0.11, -0.12), 0.12 * U),
                        m_seg(*Q(-0.05, -0.10), *Q(0.05, -0.16), 0.07 * U),
                        m_seg(*Q(-0.14, 0.14), *Q(-0.02, 0.02), 0.04 * U),
                        m_seg(*Q(0.0, 0.16), *Q(0.10, 0.04), 0.04 * U))
        elif k == 24:   # road narrows on the right
            w = 0.06 * U
            m = m_union(m_seg(*Q(-0.09, 0.15), *Q(-0.09, -0.20), w), m_seg(*Q(0.09, 0.15), *Q(0.09, 0.0), w),
                        m_seg(*Q(0.09, 0.0), *Q(0.03, -0.20), w))
        elif k == 25:   # road works
            m = m_union(m_disc(*Q(-0.06, -0.20), 0.05 * U), m_seg(*Q(-0.05, -0.14), *Q(0.0, 0.0), 0.10 * U),
                        m_seg(*Q(0.0, 0.0), *Q(-0.07, 0.15), 0.06 * U), m_seg(*Q(0.0, 0.0), *Q(0.08, 0.14), 0.06 * U),
                        m_seg(*Q(-0.03, -0.08), *Q(0.10, 0.02), 0.045 * U), m_seg(*Q(0.10, 0.02), *Q(0.16, 0.10), 0.035 * U),
                        m_halfdisc_up(*Q(0.14, 0.15), 0.11 * U))
        elif k == 26:   # traffic lights: three outlined coloured discs
            m = m_union(*[m_disc(*Q(0, yy), 0.075 * U) for yy in (-0.19, -0.03, 0.13)])
            colour_parts = [(m_disc(*Q(0, yy), 0.055 * U), c) for yy, c in
                            ((-0.19, (0.85, 0.15, 0.10)), (-0.03, (0.90, 0.75, 0.10)), (0.13, (0.10, 0.60, 0.35)))]
        elif k == 27:   # pedestrians
            m = m_union(m_disc(*Q(0.01, -0.20), 0.05 * U), m_seg(*Q(0.0, -0.15), *Q(-0.01, 0.0), 0.10 * U),
                        m_seg(*Q(-0.01, 0.0), *Q(-0.09, 0.15), 0.06 * U), m_seg(*Q(-0.01, 0.0), *Q(0.07, 0.15), 0.06 * U),
                        m_seg(*Q(-0.01, -0.10), *Q(-0.09, 0.0), 0.04 * U), m_seg(*Q(-0.01, -0.10), *Q(0.07, -0.03), 0.04 * U))
        elif k == 28:   # children crossing
            m = m_union(m_disc(*Q(0.07, -0.19), 0.045 * U), m_seg(*Q(0.07, -0.15), *Q(0.04, -0.01), 0.085 * U),
                        m_seg(*Q(0.04, -0.01), *Q(-0.03, 0.15), 0.05 * U), m_seg(*Q(0.04, -0.01), *Q(0.12, 0.14), 0.05 * U),
                        m_disc(*Q(-0.09, -0.10), 0.04 * U), m_seg(*Q(-0.09, -0.07), *Q(-0.11, 0.03), 0.07 * U),
                        m_seg(*Q(-0.11, 0.03), *Q(-0.17, 0.15), 0.045 * U), m_seg(*Q(-0.11, 0.03), *Q(-0.05, 0.15), 0.045 * U))
        elif k == 29:   # cyclists
            m = m_union(m_ring(*Q(-0.11, 0.06), 0.10 * U, 0.07 * U), m_ring(*Q(0.11, 0.06), 0.10 * U, 0.07 * U),
                        m_seg(*Q(-0.11, 0.06), *Q(-0.02, -0.08), 0.035 * U), m_seg(*Q(-0.02, -0.08), *Q(0.11, 0.06), 0.035 * U),
                        m_seg(*Q(-0.02, -0.08), *Q(0.07, -0.08), 0.035 * U), m_seg(*Q(-0.02, -0.08), *Q(0.0, 0.06), 0.035 * U))
        elif k == 30:   # ice or snow
            c = Q(0, -0.03)
            ms = []
            for a in (90, 30, 150):
                dx, dy = np.cos(np.radians(a)), np.sin(np.radians(a))
                ms.append(m_seg(c[0] - 0.19 * U * dx, c[1] - 0.19 * U * dy, c[0] + 0.19 * U * dx, c[1] + 0.19 * U * dy, 0.045 * U))
                for sgn in (1, -1):
                    tx, ty = c[0] + sgn * 0.13 * U * dx, c[1] + sgn * 0.13 * U * dy
                    for b in (50, -50):
                        bx, by = np.cos(np.radians(a + b)), np.sin(np.radians(a + b))
                        ms.append(m_seg(tx, ty, tx + sgn * 0.07 * U * bx, ty + sgn * 0.07 * U * by, 0.035 * U))
            m = m_union(*ms)
        else:   # 31 wild animals (deer leaping to the left)
            m = m_union(m_seg(*Q(-0.08, -0.01), *Q(0.12, -0.05), 0.12 * U), m_seg(*Q(-0.08, -0.01), *Q(-0.16, -0.15), 0.07 * U),
                        m_seg(*Q(-0.16, -0.15), *Q(-0.22, -0.12), 0.05 * U), m_seg(*Q(-0.16, -0.15), *Q(-0.20, -0.26), 0.03 * U),
                        m_seg(*Q(-0.16, -0.15), *Q(-0.10, -0.25), 0.03 * U), m_seg(*Q(-0.06, 0.03), *Q(-0.16, 0.16), 0.04 * U),
                        m_seg(*Q(-0.04, 0.03), *Q(-0.10, 0.17), 0.04 * U), m_seg(*Q(0.10, -0.01), *Q(0.20, 0.13), 0.04 * U),
                        m_seg(*Q(0.12, -0.01), *Q(0.22, 0.09), 0.04 * U))
        cv.paint(m * inner, BLACK)
        for cm, cc in colour_parts:
            cv.paint(cm * inner, cc)
    elif k == 32:   # end of all limits
        draw_end_disc(cv, cx, cy, D)
        draw_end_stripes(cv, cx, cy, D)
    elif k in (33, 34, 35, 36, 37, 38, 39, 40):   # blue mandatory signs
        inner = draw_blue_disc(cv, cx, cy, D)
        U = BLUE_PICTO_SCALE * D

        def Q(x, y):
            return P(x, y, cx, cy, U)

        if k in (33, 34):   # turn right / left ahead
            s = 1.0 if k == 33 else -1.0
            a0, a1 = (180, 270) if k == 33 else (270, 360)
            m = m_union(m_seg(*Q(-0.10 * s, 0.30), *Q(-0.10 * s, 0.0), 0.10 * U),
                        m_arc(*Q(0.07 * s, 0.0), 0.17 * U, a0, a1, 0.10 * U),
                        arrow_head(Q(0.30 * s, -0.17), Q(0.08 * s, -0.17), 0.14 * U))
        elif k == 35:   # ahead only
            m = m_union(m_seg(*Q(0, 0.30), *Q(0, -0.10), 0.11 * U), arrow_head(Q(0, -0.33), Q(0, -0.08), 0.20 * U))
        elif k in (36, 37):   # straight or right / left
            s = 1.0 if k == 36 else -1.0
            m = m_union(m_seg(*Q(-0.07 * s, 0.30), *Q(-0.07 * s, -0.10), 0.10 * U),
                        arrow_head(Q(-0.07 * s, -0.33), Q(-0.07 * s, -0.09), 0.17 * U),
                        m_seg(*Q(-0.07 * s, 0.10), *Q(0.10 * s, 0.10), 0.09 * U),
                        arrow_head(Q(0.30 * s, 0.10), Q(0.10 * s, 0.10), 0.14 * U))
        elif k in (38, 39):   # keep right / left
            s = 1.0 if k == 38 else -1.0
            m = m_union(m_seg(*Q(-0.21 * s, -0.21), *Q(0.06 * s, 0.06), 0.11 * U),
                        arrow_head(Q(0.23 * s, 0.23), Q(0.06 * s, 0.06), 0.18 * U))
        else:   # 40 roundabout
            r = 0.27 * U
            ms = []
            for a_start in (30, 150, 270):
                a_end = a_start - 80
                ms.append(m_arc(cx, cy, r, a_end, a_start, 0.07 * U))
                ex, ey = cx + r * np.cos(np.radians(a_end)), cy + r * np.sin(np.radians(a_end))
                tx, ty = np.sin(np.radians(a_end)), -np.cos(np.radians(a_end))   # tangent of decreasing angle
                ms.append(arrow_head((ex + 0.11 * U * tx, ey + 0.11 * U * ty), (ex, ey), 0.085 * U))
            m = m_union(*ms)
        cv.paint(m * inner, WHITE)
    elif k == 41:   # end of no passing
        draw_end_disc(cv, cx, cy, D)
        cv.paint(car_front(cx - 0.14 * D, cy, D), GREY)
        cv.paint(car_front(cx + 0.14 * D, cy, D), GREY)
        draw_end_stripes(cv, cx, cy, D)
    elif k == 42:   # end of no passing for trucks
        draw_end_disc(cv, cx, cy, D)
        cv.paint(truck_front(cx - 0.14 * D, cy - 0.01 * D, D), GREY)
        cv.paint(car_front(cx + 0.14 * D, cy, D), GREY)
        draw_end_stripes(cv, cx, cy, D)
    else:
        raise ValueError(k)
    PICTO_ONLY[0] = False
    return cv.down()


# ----------------------------------------------------------------------------------------------
# fixed filter bank (layers 1-3) -- same code builds the weights and computes template features
# ----------------------------------------------------------------------------------------------
N_ORI = 8
SIGMA_D = 1.0        # Gaussian derivative scale
SIGMA_C = 0.8        # smoothing of colour channels inside the first conv
SIGMA_S = 1.0        # smoothing of all feature maps in the second conv
SIGMA_COARSE = 2.0   # smoothing of the extra coarse "dark mass" channel
K1 = 5
K2 = 7
TINY = 1e-4          # scale fed to the second norm layer so that it only subtracts the per-channel mean
NORM_EPS = 1e-5
ORI_T, ORI_C, ORI_G = 0.20, 1.00, 1.0
COLOR_SPEC = [  # name, rgb weights (applied to normalised rgb), threshold, clip, gain, centred weight, uncentred weight
    ("red", (1.0, -1.5, 0.5), 0.5, 1.5, 0.8, 0.7, 0.3),
    ("blue", (-0.5, -0.5, 1.0), 0.5, 1.5, 0.8, 0.4, 0.6),
    ("yellow", (0.0, 1.0, -1.0), 0.5, 1.5, 0.8, 0.5, 0.5),
    ("bright", (0.299, 0.587, 0.114), 0.3, 1.3, 0.5, 1.0, 0.0),
    ("dark", (-0.299, -0.587, -0.114), 0.3, 1.3, 0.5, 1.0, 0.0),
]
N_BASE = N_ORI + len(COLOR_SPEC)     # channels of the first stage
COARSE_FROM, COARSE_GAIN = N_BASE - 1, 0.7   # the coarse channel is the dark channel smoothed more
N_FEAT = N_BASE + 1                  # channels seen by the templates
LUM = np.array([0.299, 0.587, 0.114])


def gauss1d(sigma, k):
    x = np.arange(k) - (k - 1) / 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def gauss2d(sigma, k):
    g = gauss1d(sigma, k)
    return np.outer(g, g)


def deriv_kernels(sigma, k):
    """Gx, Gy (k x k): derivative of Gaussian, scaled so that a unit step gives response 1"""
    x = np.arange(k) - (k - 1) / 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    dg = x * g                                   # positive on the positive side
    Gx = np.outer(g, dg)                         # [dy, dx]
    Gx = Gx / Gx[:, x > 0].sum()                 # step response = 1
    Gy = Gx.T.copy()
    return Gx, Gy


def build_filters():
    """weights of the fixed filter bank:
       c1  (B,3,5,5)   projections scaled by TINY  -> norm layer = per-channel mean subtraction
       c1b (2B,B,1,1)  undo the scale, weight the centred value, thresholds / clip levels as biases
       c1u (2B,3,5,5)  uncentred colour projections added in parallel
       c2  (F,2B,7,7)  Gaussian smoothing, clip combination (a - b), per-channel gain, coarse channel"""
    Bc = N_BASE
    proj = np.zeros((Bc, 3, K1, K1), np.float64)
    Gx, Gy = deriv_kernels(SIGMA_D, K1)
    gc = gauss2d(SIGMA_C, K1)
    thr, clip, gain, wc, wu = (np.zeros(Bc) for _ in range(5))
    for j in range(N_ORI):
        th = 2 * np.pi * j / N_ORI
        ker = np.cos(th) * Gx + np.sin(th) * Gy
        for c in range(3):
            proj[j, c] = LUM[c] * ker
        thr[j], clip[j], gain[j], wc[j], wu[j] = ORI_T, ORI_C, ORI_G, 1.0, 0.0
    for i, (_, rgb, t, cl, g, a, b) in enumerate(COLOR_SPEC):
        j = N_ORI + i
        for c in range(3):
            proj[j, c] = rgb[c] * gc
        thr[j], clip[j], gain[j], wc[j], wu[j] = t, cl, g, a, b
    W1 = proj * TINY
    b1 = np.zeros(Bc)
    W1b = np.zeros((2 * Bc, Bc, 1, 1))
    b1b = np.zeros(2 * Bc)
    W1u = np.zeros((2 * Bc, 3, K1, K1))
    b1u = np.zeros(2 * Bc)
    undo = 1.0 / (TINY / np.sqrt(NORM_EPS))     # norm divides the tiny values by ~sqrt(NORM_EPS)
    for j in range(Bc):
        W1b[j, j, 0, 0] = W1b[Bc + j, j, 0, 0] = wc[j] * undo
        b1b[j], b1b[Bc + j] = -thr[j], -clip[j]
        W1u[j] = W1u[Bc + j] = wu[j] * proj[j]
    gs = gauss2d(SIGMA_S, K2)
    gcoarse = gauss2d(SIGMA_COARSE, K2)
    W2 = np.zeros((N_FEAT, 2 * Bc, K2, K2))
    for j in range(Bc):
        W2[j, j] = gain[j] * gs
        W2[j, Bc + j] = -gain[j] * gs
    W2[Bc, COARSE_FROM] = COARSE_GAIN * gcoarse
    W2[Bc, Bc + COARSE_FROM] = -COARSE_GAIN * gcoarse
    b2 = np.zeros(N_FEAT)
    return dict(W1=W1, b1=b1, W1b=W1b, b1b=b1b, W1u=W1u, b1u=b1u, W2=W2, b2=b2)


def conv2d(x, W, b, stride=1, padding=0):
    """x (C,H,W) -> (O,Ho,Wo); same arithmetic as the scorer's interpreter"""
    c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=1)
    out = W.reshape(o, -1) @ cols.reshape(c * kh * kw, ho * wo)
    return out.reshape(o, ho, wo) + b[:, None, None]


def norm_layer(x):
    mu = x.mean(axis=(1, 2), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(1, 2), keepdims=True) + NORM_EPS)


def features(img, flt):
    """img (3,32,32) in [0,1] -> (F,32,32) feature maps, exactly as the network computes them"""
    x = norm_layer(img.astype(np.float64).reshape(1, 3 * N, N)).reshape(3, N, N)
    y = norm_layer(conv2d(x, flt["W1"], flt["b1"], 1, K1 // 2))
    z = conv2d(y, flt["W1b"], flt["b1b"], 1, 0) + conv2d(x, flt["W1u"], flt["b1u"], 1, K1 // 2)
    z = np.maximum(z, 0)
    return conv2d(z, flt["W2"], flt["b2"], 1, K2 // 2)


def blur_image(img, sigma=0.6, k=5):
    g = gauss2d(sigma, k)
    W = np.zeros((3, 3, k, k))
    for c in range(3):
        W[c, c] = g
    return conv2d(np.pad(img, ((0, 0), (k // 2, k // 2), (k // 2, k // 2)), mode='edge'), W, np.zeros(3), 1, 0)


def dilate(mask, r):
    """grey dilation of a (32,32) mask with a disc of radius r pixels"""
    rr = int(np.ceil(r))
    out = np.zeros_like(mask)
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            if dx * dx + dy * dy <= r * r:
                sh = np.roll(np.roll(mask, dy, 0), dx, 1)
                if dy > 0: sh[:dy] = 0
                if dy < 0: sh[dy:] = 0
                if dx > 0: sh[:, :dx] = 0
                if dx < 0: sh[:, dx:] = 0
                out = np.maximum(out, sh)
    return out


# ----------------------------------------------------------------------------------------------
# templates
# ----------------------------------------------------------------------------------------------
SCALES = [22.0, 24.5, 27.0, 29.5, 32.0]   # sign sizes in pixels covered by the template bank
PALETTE_NAMES = ["normal", "dim"]
PAD_W = 2                              # whole-sign conv: shifts of +-2 px
PAD_I = 3                              # interior conv: +-3 px, then a 3x3 max -> one extra pixel of freedom
INTERIOR_WEIGHT = 1.0                  # weight of the clipped interior (pictogram) score
INTERIOR_LO, INTERIOR_HI = -1.0, 2.0   # the interior score is clipped to this range before it is added, so that it
#                                        can break ties inside a group but never overturn the whole-sign evidence
CLASS_BIAS = np.zeros(43)
CLASS_BIAS[15] = 0.3                   # "no vehicles": an empty interior gives no interior evidence either way,
#                                        so it needs a small head start over the digit templates (hand-chosen)
# the whole-sign part is the sum of two separately normalised parts: the colour layout and the edge structure
STRUCT_W = np.array([1.0] * N_ORI + [0.0] * (N_FEAT - N_ORI))
COLOUR_W = np.array([0.0] * N_ORI + [1.0, 1.0, 1.0, 1.0, 0.5, 0.0])   # red, blue, yellow, bright, dark (generic: half), coarse (interior only)


def masked_part(f, m_ori, m_col, centre_colour, chan_w=None):
    """orientation channels: zero mean over m_ori and cut to it; colour channels: cut to m_col and, if
    centre_colour, zero mean over it as well; optional per-channel weights; unit norm (normalised by itself alone)"""
    t = np.zeros_like(f)
    if m_ori.sum() > 0:
        for j in range(N_ORI):
            mu = (f[j] * m_ori).sum() / m_ori.sum()
            t[j] = (f[j] - mu) * m_ori
    if m_col.sum() > 0:
        for j in range(N_ORI, N_FEAT):
            mu = (f[j] * m_col).sum() / m_col.sum() if centre_colour else 0.0
            t[j] = (f[j] - mu) * m_col
    if chan_w is not None:
        t = t * np.asarray(chan_w, np.float64)[:, None, None]
    nrm = np.sqrt((t ** 2).sum())
    return t / nrm if nrm > 1e-9 else t


def make_template(k, D, palette, flt):
    """returns (whole-sign template, interior template), each of unit norm (or zero)"""
    img, sign, interior = render_sign(k, D, palette=palette)
    f = features(blur_image(img), flt)             # (F,32,32)
    m_ori = (dilate(sign, 2.5) > 0.5).astype(np.float64)
    m_col = (dilate(sign, 1.0) > 0.5).astype(np.float64)
    whole = masked_part(f, m_ori, m_col, False, STRUCT_W) + masked_part(f, m_ori, m_col, False, COLOUR_W)
    m_in = (interior > 0.5).astype(np.float64)
    if k in NO_PICTOGRAM:
        # nothing inside: the interior part only penalises dark mass (coarse dark channel), normalised
        inner = np.zeros_like(f)
        inner[N_BASE] = -m_in
        inner /= np.sqrt((inner ** 2).sum()) + 1e-12
        return whole, inner
    # the pictogram alone, drawn on a plain canvas of the interior colour, so that no border edge leaks in
    img_p, _, _ = render_sign(k, D, palette=palette, picto_only=True)
    f_p = features(blur_image(img_p), flt)
    inner = masked_part(f_p, m_in, m_in, True)
    return whole, inner


def build_all():
    flt = build_filters()
    variants = [(D, p) for D in SCALES for p in PALETTE_NAMES]
    nV = len(variants)
    V = 43 * nV
    W3w = np.zeros((V, N_FEAT, N, N), np.float64)
    b3w = np.zeros(V, np.float64)
    W3i = np.zeros((V, N_FEAT, N, N), np.float64)
    b3i = np.zeros(V, np.float64)
    for k in range(43):
        for v, (D, p) in enumerate(variants):
            whole, inner = make_template(k, D, p, flt)
            W3w[k * nV + v] = whole
            W3i[k * nV + v] = inner
            b3w[k * nV + v] = CLASS_BIAS[k]
    # c4: two shifted copies of every interior score for the clipping relus
    W4 = np.zeros((2 * V, V, 1, 1))
    b4 = np.zeros(2 * V)
    for i in range(V):
        W4[i, i, 0, 0] = 1.0;      b4[i] = -INTERIOR_LO
        W4[V + i, i, 0, 0] = 1.0;  b4[V + i] = -INTERIOR_HI
    # c5: clipped interior = lo + relu(x - lo) - relu(x - hi), weighted
    W5 = np.zeros((V, 2 * V, 1, 1))
    b5 = np.zeros(V)
    for i in range(V):
        W5[i, i, 0, 0] = INTERIOR_WEIGHT
        W5[i, V + i, 0, 0] = -INTERIOR_WEIGHT
        b5[i] = INTERIOR_WEIGHT * INTERIOR_LO
    hw = (N + 2 * PAD_W - N) + 1          # 5
    hi = (N + 2 * PAD_I - N) + 1          # 7 -> 3x3 max -> 5
    F = N_FEAT
    Bc = N_BASE
    layers = [
        {"name": "rs1", "op": "reshape", "inputs": ["image"], "shape": [1, 3 * N, N]},
        {"name": "n1", "op": "norm", "inputs": ["rs1"]},
        {"name": "rs2", "op": "reshape", "inputs": ["n1"], "shape": [3, N, N]},
        {"name": "c1", "op": "conv", "inputs": ["rs2"], "out": Bc, "kernel": [K1, K1], "stride": 1, "padding": K1 // 2},
        {"name": "n2", "op": "norm", "inputs": ["c1"]},
        {"name": "c1b", "op": "conv", "inputs": ["n2"], "out": 2 * Bc, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "c1u", "op": "conv", "inputs": ["rs2"], "out": 2 * Bc, "kernel": [K1, K1], "stride": 1, "padding": K1 // 2},
        {"name": "a1", "op": "add", "inputs": ["c1b", "c1u"]},
        {"name": "r1", "op": "relu", "inputs": ["a1"]},
        {"name": "c2", "op": "conv", "inputs": ["r1"], "out": F, "kernel": [K2, K2], "stride": 1, "padding": K2 // 2},
        {"name": "c3w", "op": "conv", "inputs": ["c2"], "out": V, "kernel": [N, N], "stride": 1, "padding": PAD_W},
        {"name": "c3i", "op": "conv", "inputs": ["c2"], "out": V, "kernel": [N, N], "stride": 1, "padding": PAD_I},
        {"name": "p3i", "op": "maxpool", "inputs": ["c3i"], "size": [3, 3], "stride": [1, 1]},
        {"name": "c4", "op": "conv", "inputs": ["p3i"], "out": 2 * V, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "r4", "op": "relu", "inputs": ["c4"]},
        {"name": "c5", "op": "conv", "inputs": ["r4"], "out": V, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "a2", "op": "add", "inputs": ["c3w", "c5"]},
        {"name": "p1", "op": "maxpool", "inputs": ["a2"], "size": [hw, hw], "stride": [hw, hw]},
        {"name": "rs3", "op": "reshape", "inputs": ["p1"], "shape": [1, 43, nV]},
        {"name": "p2", "op": "maxpool", "inputs": ["rs3"], "size": [1, nV], "stride": [1, nV]},
        {"name": "rs4", "op": "reshape", "inputs": ["p2"], "shape": [43]},
    ]
    assert hi - 2 == hw
    f32 = lambda a: np.asarray(a, np.float32)
    weights = {
        "c1.W": f32(flt["W1"]), "c1.b": f32(flt["b1"]),
        "c1b.W": f32(flt["W1b"]), "c1b.b": f32(flt["b1b"]),
        "c1u.W": f32(flt["W1u"]), "c1u.b": f32(flt["b1u"]),
        "c2.W": f32(flt["W2"]), "c2.b": f32(flt["b2"]),
        "c3w.W": f32(W3w), "c3w.b": f32(b3w),
        "c3i.W": f32(W3i), "c3i.b": f32(b3i),
        "c4.W": f32(W4), "c4.b": f32(b4),
        "c5.W": f32(W5), "c5.b": f32(b5),
    }
    return {"layers": layers}, weights


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    net, weights = build_all()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(net, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **weights)
    print("wrote", out, "total weights", sum(int(v.size) for v in weights.values()))


if __name__ == "__main__":
    main()
