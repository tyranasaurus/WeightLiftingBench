"""build.py: writes network.json and weights.npz for a hand-designed traffic-sign classifier.
All numbers come from this file: procedural drawings of the 43 German sign classes (general knowledge of their
designs), Gaussian-derivative / bar filters from formulas, and template matching by cosine similarity.
No training data is read or used. Usage: python3 build.py OUT_DIR"""
import json
import numpy as np

# ---------------- colours (RGB in [0,1]) ----------------
RED = (0.72, 0.10, 0.10)
WHITE = (0.92, 0.92, 0.92)
BLACK = (0.08, 0.08, 0.08)
BLUE = (0.10, 0.25, 0.65)
YELLOW = (0.95, 0.75, 0.12)
GREY = (0.55, 0.55, 0.55)      # grey pictograms on "end of" signs
LGREY = (0.82, 0.82, 0.82)     # light disc of "end of" signs
DGREY = (0.28, 0.28, 0.28)     # the diagonal stripe
TLRED = (0.85, 0.15, 0.10); TLYEL = (0.95, 0.75, 0.1); TLGRN = (0.1, 0.6, 0.35)


# ---------------- primitives (boolean masks on the (u, v) grids) ----------------
def disc(u, v, c, rad):
    return (u - c[0]) ** 2 + (v - c[1]) ** 2 <= rad * rad


def ellipse(u, v, c, a, b):
    return ((u - c[0]) / a) ** 2 + ((v - c[1]) / b) ** 2 <= 1.0


def ellipse_ring(u, v, c, a, b, sw):
    d = np.sqrt(((u - c[0]) / a) ** 2 + ((v - c[1]) / b) ** 2)
    return np.abs(d - 1.0) * min(a, b) <= sw / 2


def segment(u, v, p0, p1, sw):
    px, py = p1[0] - p0[0], p1[1] - p0[1]
    L2 = px * px + py * py
    if L2 < 1e-12:
        return disc(u, v, p0, sw / 2)
    t = np.clip(((u - p0[0]) * px + (v - p0[1]) * py) / L2, 0, 1)
    dx = u - (p0[0] + t * px); dy = v - (p0[1] + t * py)
    return dx * dx + dy * dy <= (sw / 2) ** 2


def polyline(u, v, pts, sw):
    m = np.zeros(u.shape, bool)
    for a, b in zip(pts[:-1], pts[1:]):
        m |= segment(u, v, a, b, sw)
    return m


def arc(u, v, c, rad, a0, a1, sw):
    """Arc of a circle: angles in degrees with atan2(v, u) (0 = right, 90 = down, -90 = up), from a0 to a1 clockwise."""
    d = np.hypot(u - c[0], v - c[1])
    ang = np.degrees(np.arctan2(v - c[1], u - c[0]))
    span = (a1 - a0) % 360.0
    rel = (ang - a0) % 360.0
    return (np.abs(d - rad) <= sw / 2) & (rel <= span)


def polygon(u, v, pts):
    """Even-odd point-in-polygon test for an arbitrary polygon."""
    inside = np.zeros(u.shape, bool)
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        cond = (v > min(y0, y1)) & (v <= max(y0, y1))
        if y1 != y0:
            xint = x0 + (v - y0) * (x1 - x0) / (y1 - y0)
            inside ^= cond & (u < xint)
    return inside


def regular_polygon(u, v, c, rad, n, rot_deg):
    pts = [(c[0] + rad * np.cos(np.radians(rot_deg + 360.0 * i / n)), c[1] + rad * np.sin(np.radians(rot_deg + 360.0 * i / n))) for i in range(n)]
    return polygon(u, v, pts)


def rect(u, v, x0, y0, x1, y1):
    return (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)


def round_rect(u, v, x0, y0, x1, y1, r):
    cx = np.clip(u, x0 + r, x1 - r); cy = np.clip(v, y0 + r, y1 - r)
    return (u - cx) ** 2 + (v - cy) ** 2 <= r * r


def triangle_up(u, v, half_w, inset=0.0):
    """Equilateral triangle, apex up, half-width half_w, bounding box vertically centred on v=0, shrunk by inset (perpendicular)."""
    h = half_w * np.sqrt(3.0)
    top, bot = -h / 2, h / 2
    # inset: move each edge inward by 'inset'; for an equilateral triangle that scales about the incentre
    inc_v = bot - h / 3.0  # incentre
    k = 1.0 - inset / (h / 3.0)
    pts = [(0.0, inc_v + (top - inc_v) * k), (half_w * k, inc_v + (bot - inc_v) * k), (-half_w * k, inc_v + (bot - inc_v) * k)]
    return polygon(u, v, pts)


def triangle_down(u, v, half_w, inset=0.0):
    return triangle_up(u, -v, half_w, inset)


# ---------------- digits (DIN-1451-like, stroke based) ----------------
def digit(u, v, ch, c, w, h, sw):
    """Digit ch centred at c, box width w, height h, stroke width sw."""
    uu = u - c[0]; vv = v - c[1]
    L, R, T, B = -w / 2 + sw / 2, w / 2 - sw / 2, -h / 2 + sw / 2, h / 2 - sw / 2
    m = np.zeros(u.shape, bool)
    if ch == '0':
        m |= ellipse_ring(uu, vv, (0, 0), w / 2 - sw / 2, h / 2 - sw / 2, sw)
    elif ch == '1':
        x = 0.12 * w
        m |= segment(uu, vv, (x, T), (x, B), sw)
        m |= segment(uu, vv, (x, T), (x - 0.42 * w, T + 0.30 * h), sw)
    elif ch == '2':
        r = (R - L) / 2
        cy = T + r
        m |= arc(uu, vv, (0, cy), r, 160, 390, sw)       # from upper-left over the top to lower-right
        end = (r * np.cos(np.radians(30)), cy + r * np.sin(np.radians(30)))
        m |= segment(uu, vv, end, (L, B), sw)
        m |= segment(uu, vv, (L, B), (R, B), sw)
    elif ch == '3':
        r1 = (R - L) / 2 * 0.92; cy1 = T + r1
        r2 = (R - L) / 2; cy2 = B - r2
        m |= arc(uu, vv, (0, cy1), r1, 200, 450, sw)     # upper bowl (open at lower-left)
        m |= arc(uu, vv, (0, cy2), r2, 270, 520, sw)     # lower bowl (open at upper-left)
    elif ch == '5':
        m |= segment(uu, vv, (L, T), (R, T), sw)
        m |= segment(uu, vv, (L, T), (L, -0.02 * h), sw)
        r = (R - L) / 2; cy = B - r
        m |= arc(uu, vv, (0, cy), r, 215, 500, sw)       # lower bowl
        m |= segment(uu, vv, (L, cy - r * 0.0 - 0.0), (L, -0.02 * h), sw * 0.9) if False else m
        m |= segment(uu, vv, (L, -0.02 * h), (0, cy - r), sw)
    elif ch == '6':
        r = (R - L) / 2; cy = B - r
        m |= ellipse_ring(uu, vv, (0, cy), r, r, sw)
        m |= arc(uu, vv, (0, T + r), r, 180, 330, sw)   # top hook
        m |= segment(uu, vv, (L, T + r), (L, cy), sw)
    elif ch == '7':
        m |= segment(uu, vv, (L, T), (R, T), sw)
        m |= segment(uu, vv, (R, T), (-0.12 * w, B), sw)
    elif ch == '8':
        r1 = (R - L) / 2 * 0.88; cy1 = T + r1 * 1.0
        r2 = (R - L) / 2; cy2 = B - r2
        m |= ellipse_ring(uu, vv, (0, cy1), r1, (cy2 - r2 - T) / 2 + sw * 0.3, sw)
        m |= ellipse_ring(uu, vv, (0, cy2), r2, r2, sw)
    return m


def number(u, v, text, c, dh, sw, gap=0.10):
    """Digits of 'text' centred at c with digit height dh. '1' is narrower."""
    wide = 0.56 if len(text) <= 2 else 0.50
    widths = [0.28 * dh if ch == '1' else wide * dh for ch in text]
    total = sum(widths) + gap * dh * (len(text) - 1)
    x = c[0] - total / 2
    m = np.zeros(u.shape, bool)
    for ch, w in zip(text, widths):
        m |= digit(u, v, ch, (x + w / 2, c[1]), w, dh, sw)
        x += w + gap * dh
    return m


# ---------------- pictogram parts ----------------
def car_back(u, v, c, w, h):
    """Rear view of a car: body with window, as a filled silhouette with a lighter window."""
    body = round_rect(u, v, c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2, 0.12 * h)
    body |= rect(u, v, c[0] - w * 0.45, c[1] + h * 0.35, c[0] - w * 0.3, c[1] + h * 0.62)
    body |= rect(u, v, c[0] + w * 0.3, c[1] + h * 0.35, c[0] + w * 0.45, c[1] + h * 0.62)
    win = round_rect(u, v, c[0] - w * 0.32, c[1] - h * 0.42, c[0] + w * 0.32, c[1] - h * 0.02, 0.05 * h)
    return body, win


def truck_side(u, v, c, w, h):
    """Side view of a lorry facing left: cab (left) + box (right) + wheels."""
    x0, x1 = c[0] - w / 2, c[0] + w / 2
    y0, y1 = c[1] - h / 2, c[1] + h / 2
    box = rect(u, v, x0 + 0.30 * w, y0, x1, y1 - 0.18 * h)
    cab = round_rect(u, v, x0, y0 + 0.30 * h, x0 + 0.33 * w, y1 - 0.18 * h, 0.08 * h)
    wheels = disc(u, v, (x0 + 0.17 * w, y1 - 0.12 * h), 0.14 * h) | disc(u, v, (x1 - 0.2 * w, y1 - 0.12 * h), 0.14 * h)
    return box | cab | wheels


def arrow_head(u, v, tip, direction_deg, length, half_w):
    """Filled triangular arrow head with the tip at 'tip', pointing in direction_deg (atan2 convention)."""
    d = np.radians(direction_deg)
    dx, dy = np.cos(d), np.sin(d)
    nx, ny = -dy, dx
    base = (tip[0] - dx * length, tip[1] - dy * length)
    pts = [tip, (base[0] + nx * half_w, base[1] + ny * half_w), (base[0] - nx * half_w, base[1] - ny * half_w)]
    return polygon(u, v, pts)


def person(u, v, c, s, walking=True):
    """Walking figure, height ~s, centred at c (c is the body centre)."""
    m = disc(u, v, (c[0] + 0.02 * s, c[1] - 0.40 * s), 0.11 * s)
    m |= segment(u, v, (c[0], c[1] - 0.27 * s), (c[0] + 0.02 * s, c[1] + 0.05 * s), 0.17 * s)
    if walking:
        m |= segment(u, v, (c[0], c[1] + 0.0 * s), (c[0] - 0.22 * s, c[1] + 0.45 * s), 0.11 * s)
        m |= segment(u, v, (c[0], c[1] + 0.0 * s), (c[0] + 0.20 * s, c[1] + 0.45 * s), 0.11 * s)
        m |= segment(u, v, (c[0], c[1] - 0.2 * s), (c[0] + 0.25 * s, c[1] + 0.0 * s), 0.09 * s)
        m |= segment(u, v, (c[0], c[1] - 0.2 * s), (c[0] - 0.22 * s, c[1] - 0.02 * s), 0.09 * s)
    else:
        m |= segment(u, v, (c[0], c[1]), (c[0] - 0.08 * s, c[1] + 0.45 * s), 0.11 * s)
        m |= segment(u, v, (c[0], c[1]), (c[0] + 0.10 * s, c[1] + 0.45 * s), 0.11 * s)
    return m


# ---------------- sign drawings ----------------
SPEED = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}
TRIANGLE_CLASSES = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
CIRCLE_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42}


def paint(img, mask, colour):
    img[mask] = colour


def draw_sign(k, u, v, bg):
    """Returns an (H, W, 3) float image of class k on the (u, v) grids with background colour bg."""
    img = np.empty(u.shape + (3,), np.float32)
    img[:] = bg
    RING_IN = 0.76
    if k in SPEED or k in (9, 10, 15, 16):
        paint(img, disc(u, v, (0, 0), 1.0), RED)
        paint(img, disc(u, v, (0, 0), RING_IN), WHITE)
        if k in SPEED:
            txt = SPEED[k]
            dh = 0.95 if len(txt) == 2 else 0.92
            sw = 0.13 if len(txt) == 2 else 0.12
            paint(img, number(u, v, txt, (0, 0.0), dh, sw, gap=0.10 if len(txt) == 2 else 0.05), BLACK)
        elif k == 9:
            b1, w1 = car_back(u, v, (-0.33, 0.03), 0.52, 0.42)
            b2, w2 = car_back(u, v, (0.33, 0.03), 0.52, 0.42)
            paint(img, b1, RED); paint(img, w1, WHITE)
            paint(img, b2, BLACK); paint(img, w2, WHITE)
        elif k == 10:
            paint(img, round_rect(u, v, -0.62, -0.42, -0.08, 0.30, 0.04), RED)      # lorry seen from behind (tall box)
            paint(img, rect(u, v, -0.55, 0.30, -0.45, 0.40) | rect(u, v, -0.25, 0.30, -0.15, 0.40), RED)
            b2, w2 = car_back(u, v, (0.33, 0.08), 0.50, 0.40)
            paint(img, b2, BLACK); paint(img, w2, WHITE)
        elif k == 16:
            paint(img, truck_side(u, v, (0.0, 0.0), 1.15, 0.72), BLACK)
    elif k == 17:
        paint(img, disc(u, v, (0, 0), 1.0), RED)
        paint(img, rect(u, v, -0.62, -0.13, 0.62, 0.13), WHITE)
    elif k in (6, 32, 41, 42):
        paint(img, disc(u, v, (0, 0), 1.0), LGREY)
        if k == 6:
            paint(img, number(u, v, '80', (0, 0), 0.95, 0.13), DGREY)
        elif k == 41:
            b1, w1 = car_back(u, v, (-0.33, 0.03), 0.52, 0.42)
            b2, w2 = car_back(u, v, (0.33, 0.03), 0.52, 0.42)
            paint(img, b1, GREY); paint(img, w1, LGREY)
            paint(img, b2, GREY); paint(img, w2, LGREY)
        elif k == 42:
            paint(img, round_rect(u, v, -0.62, -0.42, -0.08, 0.30, 0.04), GREY)
            b2, w2 = car_back(u, v, (0.33, 0.08), 0.50, 0.40)
            paint(img, b2, GREY); paint(img, w2, LGREY)
        # diagonal stripe from upper-right to lower-left
        if k == 32:
            for off in (-0.16, 0.0, 0.16):
                paint(img, segment(u, v, (0.75 + off, -0.75 + off), (-0.75 + off, 0.75 + off), 0.09) & disc(u, v, (0, 0), 0.97), DGREY)
        else:
            paint(img, segment(u, v, (0.75, -0.75), (-0.75, 0.75), 0.16) & disc(u, v, (0, 0), 0.97), DGREY)
    elif k in (33, 34, 35, 36, 37, 38, 39, 40):
        paint(img, disc(u, v, (0, 0), 1.0), WHITE)
        paint(img, disc(u, v, (0, 0), 0.94), BLUE)
        sw = 0.17; hl = 0.32; hw = 0.26
        if k == 35:
            paint(img, segment(u, v, (0, 0.62), (0, -0.30), sw) | arrow_head(u, v, (0, -0.66), -90, hl, hw), WHITE)
        elif k in (33, 34):
            s = 1 if k == 33 else -1
            m = segment(u, v, (-0.22 * s, 0.62), (-0.22 * s, 0.0), sw)
            m |= arc(u, v, (0.08 * s, 0.0), 0.30, 180 if s == 1 else 270, 270 if s == 1 else 360, sw)
            m |= segment(u, v, (0.08 * s, -0.30), (0.30 * s, -0.30), sw)
            m |= arrow_head(u, v, (0.66 * s, -0.30), 0 if s == 1 else 180, hl, hw)
            paint(img, m, WHITE)
        elif k in (36, 37):
            s = 1 if k == 36 else -1
            m = segment(u, v, (-0.18 * s, 0.62), (-0.18 * s, -0.32), sw) | arrow_head(u, v, (-0.18 * s, -0.68), -90, hl, hw)
            m |= segment(u, v, (-0.18 * s, 0.18), (0.30 * s, 0.18), sw) | arrow_head(u, v, (0.66 * s, 0.18), 0 if s == 1 else 180, hl, hw)
            paint(img, m, WHITE)
        elif k in (38, 39):
            s = 1 if k == 38 else -1
            ang = 45 if s == 1 else 135
            m = segment(u, v, (-0.42 * s, -0.45), (0.22 * s, 0.19), sw) | arrow_head(u, v, (0.50 * s, 0.47), ang, hl * 1.1, hw)
            paint(img, m, WHITE)
        elif k == 40:
            m = np.zeros(u.shape, bool)
            for a0 in (-150, -30, 90):
                m |= arc(u, v, (0, 0), 0.50, a0, a0 + 75, 0.13)
                # arrow head at the counter-clockwise end (angle a0): tangent direction = a0 - 90
                tip_a = np.radians(a0 - 10)
                m |= arrow_head(u, v, (0.50 * np.cos(tip_a), 0.50 * np.sin(tip_a)), a0 - 100, 0.24, 0.20)
            paint(img, m, WHITE)
    elif k == 12:
        paint(img, regular_polygon(u, v, (0, 0), 1.0, 4, 0), WHITE)
        paint(img, regular_polygon(u, v, (0, 0), 0.62, 4, 0), YELLOW)
    elif k == 13:
        paint(img, triangle_down(u, v, 1.0), RED)
        paint(img, triangle_down(u, v, 1.0, inset=0.18), WHITE)
    elif k == 14:
        paint(img, regular_polygon(u, v, (0, 0), 1.0, 8, 22.5), WHITE)
        paint(img, regular_polygon(u, v, (0, 0), 0.94, 8, 22.5), RED)
        # STOP: four letters, white strokes
        lh, lw, sw = 0.46, 0.30, 0.10
        xs = [-0.57, -0.19, 0.19, 0.57]
        m = np.zeros(u.shape, bool)
        # S
        cx = xs[0]
        m |= arc(u, v, (cx, -lh / 4), lh / 4 - sw / 2 + 0.02, 0, 270, sw) if False else m
        m |= arc(u, v, (cx, -lh / 4 + 0.01), lh / 4 - 0.02, 40, 300, sw)
        m |= arc(u, v, (cx, lh / 4 - 0.01), lh / 4 - 0.02, 220, 480, sw)
        # T
        cx = xs[1]
        m |= segment(u, v, (cx - lw / 2, -lh / 2 + sw / 2), (cx + lw / 2, -lh / 2 + sw / 2), sw)
        m |= segment(u, v, (cx, -lh / 2 + sw / 2), (cx, lh / 2 - sw / 2), sw)
        # O
        cx = xs[2]
        m |= ellipse_ring(u, v, (cx, 0), lw / 2 - sw / 2, lh / 2 - sw / 2, sw)
        # P
        cx = xs[3]
        m |= segment(u, v, (cx - lw / 2 + sw / 2, -lh / 2 + sw / 2), (cx - lw / 2 + sw / 2, lh / 2 - sw / 2), sw)
        m |= ellipse_ring(u, v, (cx - lw / 2 + sw / 2 + 0.02, -lh / 4 + 0.0), lw / 2 - sw / 2 + 0.02, lh / 4 - sw / 2 + 0.01, sw) & (u >= cx - lw / 2 + sw / 2)
        paint(img, m, WHITE)
    elif k in TRIANGLE_CLASSES:
        paint(img, triangle_up(u, v, 1.0), RED)
        paint(img, triangle_up(u, v, 1.0, inset=0.17), WHITE)
        m = np.zeros(u.shape, bool)
        cy = 0.27  # pictogram centre (around the incentre)
        if k == 11:
            m |= segment(u, v, (0, cy - 0.42), (0, cy + 0.38), 0.30)
            m |= segment(u, v, (-0.38, cy - 0.02), (0.38, cy - 0.02), 0.09)
        elif k == 18:
            m |= segment(u, v, (0, cy - 0.50), (0, cy + 0.16), 0.15)
            m |= disc(u, v, (0, cy + 0.42), 0.09)
        elif k in (19, 20):
            s = -1 if k == 19 else 1   # 19 bends left
            m |= segment(u, v, (-0.16 * s, cy + 0.45), (-0.16 * s, cy - 0.02), 0.13)
            m |= arc(u, v, (0.12 * s, cy - 0.02), 0.28, 180 if s == 1 else 270, 270 if s == 1 else 360, 0.13)
            m |= segment(u, v, (0.12 * s, cy - 0.30), (0.40 * s, cy - 0.30), 0.13)
        elif k == 21:
            pts = [(0.10, cy + 0.48), (0.10, cy + 0.20), (-0.16, cy + 0.04), (-0.16, cy - 0.14), (0.10, cy - 0.30), (0.10, cy - 0.50)]
            m |= polyline(u, v, pts, 0.14)
        elif k == 22:
            m |= rect(u, v, -0.50, cy + 0.26, 0.50, cy + 0.36)
            m |= (disc(u, v, (-0.25, cy + 0.30), 0.19) | disc(u, v, (0.25, cy + 0.30), 0.19)) & (v <= cy + 0.30)
        elif k == 23:
            # small skewed car upper-left, two long wavy skid marks running down to the right
            cu = u + 0.10; cv = v - (cy - 0.22)
            ang = np.radians(-18); ru = cu * np.cos(ang) - cv * np.sin(ang); rv = cu * np.sin(ang) + cv * np.cos(ang)
            m |= round_rect(ru, rv, -0.26, -0.10, 0.26, 0.10, 0.05)
            for x0 in (-0.30, -0.08):
                pts = [(x0, cy - 0.02), (x0 + 0.14, cy + 0.12), (x0 + 0.12, cy + 0.26), (x0 + 0.28, cy + 0.40), (x0 + 0.30, cy + 0.48)]
                m |= polyline(u, v, pts, 0.07)
        elif k == 24:
            m |= segment(u, v, (-0.20, cy - 0.45), (-0.20, cy + 0.48), 0.14)
            m |= polyline(u, v, [(0.22, cy + 0.48), (0.22, cy + 0.08), (0.06, cy - 0.45)], 0.14)
        elif k == 25:
            # worker leaning right with a shovel; heap of earth at the lower right
            m |= disc(u, v, (-0.12, cy - 0.44), 0.09)
            m |= segment(u, v, (-0.10, cy - 0.32), (0.02, cy + 0.02), 0.17)
            m |= segment(u, v, (0.0, cy + 0.02), (-0.18, cy + 0.46), 0.10) | segment(u, v, (0.0, cy + 0.02), (0.16, cy + 0.44), 0.10)
            m |= segment(u, v, (-0.30, cy + 0.40), (0.24, cy - 0.16), 0.06)
            m |= ellipse(u, v, (0.34, cy + 0.42), 0.20, 0.10)
        elif k == 26:
            paint(img, round_rect(u, v, -0.17, cy - 0.48, 0.17, cy + 0.46, 0.08) & ~round_rect(u, v, -0.12, cy - 0.43, 0.12, cy + 0.41, 0.06), BLACK)
            paint(img, disc(u, v, (0, cy - 0.30), 0.13), TLRED)
            paint(img, disc(u, v, (0, cy - 0.0), 0.13), TLYEL)
            paint(img, disc(u, v, (0, cy + 0.30), 0.13), TLGRN)
        elif k == 27:
            m |= person(u, v, (0.0, cy + 0.0), 0.95, walking=True)
        elif k == 28:
            m |= person(u, v, (-0.18, cy + 0.08), 0.72, walking=True)
            m |= person(u, v, (0.18, cy + 0.03), 0.82, walking=True)
        elif k == 29:
            m |= ellipse_ring(u, v, (-0.28, cy + 0.24), 0.21, 0.21, 0.06) | ellipse_ring(u, v, (0.28, cy + 0.24), 0.21, 0.21, 0.06)
            m |= polyline(u, v, [(-0.28, cy + 0.24), (-0.05, cy - 0.12), (0.28, cy + 0.24), (0.0, cy + 0.24), (-0.05, cy - 0.12)], 0.06)
            m |= segment(u, v, (-0.05, cy - 0.12), (0.12, cy - 0.28), 0.06)
        elif k == 30:
            c = (0.0, cy + 0.02)
            for a in (0, 60, 120):
                d = np.radians(a)
                m |= segment(u, v, (c[0] - 0.38 * np.cos(d), c[1] - 0.38 * np.sin(d)), (c[0] + 0.38 * np.cos(d), c[1] + 0.38 * np.sin(d)), 0.08)
            for a in (30, 90, 150):
                d = np.radians(a)
                m |= disc(u, v, (c[0] + 0.22 * np.cos(d), c[1] + 0.22 * np.sin(d)), 0.055) | disc(u, v, (c[0] - 0.22 * np.cos(d), c[1] - 0.22 * np.sin(d)), 0.055)
        elif k == 31:
            m |= ellipse(u, v, (0.08, cy + 0.05), 0.34, 0.13)
            m |= segment(u, v, (-0.20, cy + 0.0), (-0.40, cy - 0.28), 0.12)
            m |= segment(u, v, (-0.40, cy - 0.28), (-0.52, cy - 0.26), 0.09)
            m |= segment(u, v, (-0.40, cy - 0.30), (-0.30, cy - 0.50), 0.05) | segment(u, v, (-0.40, cy - 0.30), (-0.50, cy - 0.50), 0.05)
            m |= segment(u, v, (-0.15, cy + 0.12), (-0.30, cy + 0.46), 0.06) | segment(u, v, (0.32, cy + 0.12), (0.45, cy + 0.44), 0.06)
            m |= segment(u, v, (0.0, cy + 0.14), (-0.1, cy + 0.46), 0.06) | segment(u, v, (0.22, cy + 0.14), (0.3, cy + 0.46), 0.06)
        paint(img, m, BLACK)
    return img


def render(k, r=13.0, cx=16.0, cy=16.0, rot=0.0, squash=1.0, bg=(0.45, 0.45, 0.45), size=32, S=4):
    """Render class k into a (3, size, size) image. r: sign half-size in pixels; (cx, cy): centre in pixels."""
    n = size * S
    xs = (np.arange(n) + 0.5) / S
    X, Y = np.meshgrid(xs, xs)
    u = (X - cx) / r; v = (Y - cy) / r
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    u, v = c * u + s * v, -s * u + c * v
    u = u / squash
    img = draw_sign(k, u, v, bg)
    img = img.reshape(size, S, size, S, 3).mean(axis=(1, 3))
    return img.transpose(2, 0, 1)


# ---------------- numpy implementation of the scorer's ops (used only to compute template features) ----------------
NORM_EPS = 1e-5
def _conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, c, kh * kw).reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]

def _pool(x, size, stride, reduce):
    n, c, h, w = x.shape
    (sh, sw), (th, tw) = size, stride
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if reduce == "max" else win.mean(0)

def net_run(net, weights, images):
    vals = {"image": images}
    for L in net["layers"]:
        ins = [vals[i] for i in L["inputs"]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = _conv(ins[0], weights[name + ".W"], weights[name + ".b"], L.get("stride", 1), L.get("padding", 0))
        elif op == "dense":
            vals[name] = ins[0].reshape(len(ins[0]), -1) @ weights[name + ".W"].T + weights[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op in ("maxpool", "avgpool"):
            vals[name] = _pool(ins[0], L["size"], L["stride"], op[:3])
        elif op == "add":
            vals[name] = sum(ins)
        elif op == "reshape":
            vals[name] = ins[0].reshape((len(ins[0]),) + tuple(L["shape"]))
        elif op == "norm":
            x = ins[0]; mu = x.mean(axis=(2, 3), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + NORM_EPS)
    return vals[net["layers"][-1]["name"]]


# ---------------- parameters ----------------
P = dict(
    sigma=1.0,          # Gaussian first-derivative scale (pixels)
    bar_sigma=1.0,      # second-derivative (bar) scale
    ksize=7,
    n_orient=4,
    grad_gain=3.0,      # gradient channels gain
    bar_gain=1.5,       # bar channels gain (0 = no bar channels)
    bar_channels=(0,),
    grad_channels=(0, 1),
    col_gate=0.15,      # scale of the pure colour channels before the norm (soft noise gate via the norm epsilon)
    col_weight=2.0,     # weight of the raw colour channels in the templates
    raw_weight=1.0,     # weight of the raw intensity channels in the templates
    pool=4, pool_stride=2,
    scales=(10.5, 12.0, 13.5),
    shifts=(-2, 0, 2),
    rots=(0.0,),
    bgs=((0.2, 0.2, 0.2), (0.75, 0.75, 0.75)),
    squashes=(0.92, 1.0, 1.08),
    blurs=(0.6,),
    two_stage=False,
    norm_gamma=1.0,     # templates divided by |w|^gamma (1 = cosine similarity)
    empty_penalty=0.0,  # negative interior weights for empty-interior classes (15 no vehicles, 13 yield, 12 priority)
    mu=1.0,             # weight of the interior (within-group) term
    int_rho_scale=1.0,  # scale factor of the interior radius
)

GROUPS = [
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16],   # red ring circles
    [6, 32, 41, 42],                            # grey "end of" discs
    [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],  # warning triangles
    [33, 34, 35, 36, 37, 38, 39, 40],           # blue discs
    [12], [13], [14], [17],                     # singletons
]


def interior_geometry(k):
    """(vertical offset of the pictogram centre in units of r, radius of the pictogram region in units of r)."""
    if k in (11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
        return 0.27, 0.55
    if k == 13:
        return -0.27, 0.55
    return 0.0, 0.72


# Crops were resized to 32x32 without keeping the aspect ratio, so wide-and-low signs (triangles) appear stretched
# vertically to roughly square proportions. Per-class aspect correction: (radius factor, horizontal squash factor).
def class_aspect(k):
    if k in (11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
        return 1.155, 0.866      # equilateral triangle (height 0.866 * width) stretched to height == width
    if k == 13:
        return 1.08, 0.926       # yield: slightly less stretched
    return 1.0, 1.0


def gaussian_blur(img, sigma):
    """Separable Gaussian blur of a (C, H, W) image with edge padding."""
    if sigma < 0.05:
        return img
    r = int(np.ceil(3 * sigma)); x = np.arange(-r, r + 1)
    g = np.exp(-x * x / (2 * sigma * sigma)); g /= g.sum()
    pad = np.pad(img, ((0, 0), (r, r), (r, r)), mode='edge')
    t = sum(g[i] * pad[:, i:i + img.shape[1], :] for i in range(2 * r + 1))
    out = sum(g[i] * t[:, :, i:i + img.shape[2]] for i in range(2 * r + 1))
    return out


def gauss_deriv_filters(sigma, size, n_orient):
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gx = -x / sigma ** 2 * g; gy = -y / sigma ** 2 * g
    gx /= np.abs(gx).sum() / 2; gy /= np.abs(gy).sum() / 2
    return np.stack([np.cos(np.pi * i / n_orient) * gx + np.sin(np.pi * i / n_orient) * gy for i in range(n_orient)])


def bar_filters(sigma, size, n_orient):
    """Second directional derivatives of a Gaussian; positive response on a dark bar whose normal is the direction.
    Normalised so a 2-pixel-wide dark bar of unit contrast gives a response of 1 at its centre."""
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    out = []
    for i in range(n_orient):
        phi = np.pi * i / n_orient
        d = x * np.cos(phi) + y * np.sin(phi)
        f = (d * d / sigma ** 4 - 1.0 / sigma ** 2) * g
        f -= f.mean()
        bar = (np.abs(d) > 1.0).astype(np.float64)
        out.append(f / (f * bar).sum())
    return np.stack(out)


def colour_matrix(p):
    """Rows: intensity, gated red-green opponent, gated blue-yellow opponent."""
    I = np.array([1 / 3, 1 / 3, 1 / 3]); RG = np.array([1.0, -1.0, 0.0]); BY = np.array([-0.5, -0.5, 1.0])
    c = p["col_gate"]
    return np.stack([I, c * RG, c * BY])


def feature_layers(p):
    """Returns (layers, weights, rows) for the feature extraction part: image -> pooled features (C, Hp, Wp)."""
    layers, W = [], {}
    M = colour_matrix(p); nc = M.shape[0]
    col = np.zeros((nc, 3, 1, 1), np.float32); col[:, :, 0, 0] = M
    layers.append({"name": "col", "op": "conv", "inputs": ["image"], "out": nc, "kernel": [1, 1], "stride": 1, "padding": 0})
    W["col.W"] = col; W["col.b"] = np.zeros(nc, np.float32)
    layers.append({"name": "cn", "op": "norm", "inputs": ["col"]})
    k = p["ksize"]; c = k // 2
    fg = gauss_deriv_filters(p["sigma"], k, p["n_orient"]) * p["grad_gain"]
    fb = bar_filters(p["bar_sigma"], k, p["n_orient"]) * p["bar_gain"]
    rows = []   # (type, channel, orient, sign)
    for ch in p["grad_channels"]:
        for s in (1.0, -1.0):
            for i in range(p["n_orient"]):
                rows.append(("g", ch, i, s))
    if p["bar_gain"] > 0:
        for ch in p["bar_channels"]:
            for s in (1.0, -1.0):
                for i in range(p["n_orient"]):
                    rows.append(("b", ch, i, s))
    for ch in range(nc):
        rows.append(("r", ch, 0, 1.0)); rows.append(("r", ch, 0, -1.0))
    Wf = np.zeros((len(rows), nc, k, k), np.float32)
    for j, (t, ch, i, s) in enumerate(rows):
        if t == "g":
            Wf[j, ch] = s * fg[i]
        elif t == "b":
            Wf[j, ch] = s * fb[i]
        else:
            Wf[j, ch, c, c] = s
    layers.append({"name": "feat", "op": "conv", "inputs": ["cn"], "out": len(rows), "kernel": [k, k], "stride": 1, "padding": c})
    W["feat.W"] = Wf; W["feat.b"] = np.zeros(len(rows), np.float32)
    layers.append({"name": "fr", "op": "relu", "inputs": ["feat"]})
    layers.append({"name": "fp", "op": "avgpool", "inputs": ["fr"], "size": [p["pool"], p["pool"]], "stride": [p["pool_stride"], p["pool_stride"]]})
    return layers, W, rows


def template_variants(p):
    out = []
    for r in p["scales"]:
        for dx in p["shifts"]:
            for dy in p["shifts"]:
                for rot in p["rots"]:
                    for bg in p["bgs"]:
                        for sq in p["squashes"]:
                            for bl in p["blurs"]:
                                out.append(dict(r=r, cx=16.0 + dx, cy=16.0 + dy, rot=rot, bg=bg, squash=sq, blur=bl))
    return out


def render_variant(k, var):
    rf, sf = class_aspect(k)
    img = render(k, r=var["r"] * rf, cx=var["cx"], cy=var["cy"], rot=var["rot"], squash=var["squash"] * sf, bg=var["bg"])
    return gaussian_blur(img, var["blur"])


def unit(F, axis=-1):
    return F / (np.linalg.norm(F, axis=axis, keepdims=True) + 1e-9)


def build_network(p, verbose=False):
    layers, W, rows = feature_layers(p)
    variants = template_variants(p)
    V = len(variants)
    feat_net = {"layers": layers}
    imgs = np.stack([render_variant(k, var) for k in range(43) for var in variants]).astype(np.float32)
    feats = np.concatenate([net_run(feat_net, W, imgs[i:i + 512]) for i in range(0, len(imgs), 512)])
    C, Hp, Wp = feats.shape[1:]
    hp = Hp * Wp
    F = feats.reshape(43, V, -1).astype(np.float64)
    # channel weights
    chan_w = np.array([(p["col_weight"] if ch > 0 else p["raw_weight"]) if t == "r" else 1.0 for (t, ch, i, s) in rows])
    F = F * np.repeat(chan_w, hp)[None, None, :]
    nrm = np.linalg.norm(F, axis=2, keepdims=True) + 1e-9
    F = F / nrm ** p.get("norm_gamma", 1.0)
    F = F / np.median(np.linalg.norm(F, axis=2))   # common scale only
    D = F.shape[2]
    # interior masks per (class, variant)
    ys = np.arange(Hp) * p["pool_stride"] + p["pool"] / 2.0; xs = np.arange(Wp) * p["pool_stride"] + p["pool"] / 2.0
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    Mint = np.zeros((43, V, hp))
    for k in range(43):
        off, rho = interior_geometry(k)
        for vi, var in enumerate(variants):
            d = np.hypot(XX - var["cx"], YY - (var["cy"] + off * var["r"])) / var["r"]
            Mint[k, vi] = (d <= rho * p["int_rho_scale"]).ravel()
    MintF = np.tile(Mint, (1, 1, C))
    kappa = p.get("empty_penalty", 0.0)
    if kappa > 0:
        # "no vehicles", "yield" and "priority road" have nothing drawn inside: penalise interior edge/bar features,
        # using the average interior of the other classes of the same shape as the pattern of what should be absent.
        edge_ch = np.array([1.0 if t in ("g", "b") else 0.0 for (t, ch, i, s) in rows])
        edge_mask = np.repeat(edge_ch, hp)[None, :]
        for k, others in ((15, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 16]), (13, [11, 18, 25, 27, 28]), (12, [11, 18, 25, 27, 28])):
            avg_int = (F[others] * MintF[others]).mean(axis=0) * edge_mask          # (V, D)
            own = F[k]
            scale = np.linalg.norm(own, axis=1, keepdims=True) / (np.linalg.norm(avg_int, axis=1, keepdims=True) + 1e-9)
            F[k] = own - kappa * scale * avg_int
        F = unit(F)
    if not p["two_stage"]:
        layers.append({"name": "tm", "op": "dense", "inputs": ["fp"], "out": 43 * V})
        W["tm.W"] = F.reshape(43 * V, D).astype(np.float32); W["tm.b"] = np.zeros(43 * V, np.float32)
        layers.append({"name": "rs", "op": "reshape", "inputs": ["tm"], "shape": [43, V, 1]})
        layers.append({"name": "mx", "op": "maxpool", "inputs": ["rs"], "size": [V, 1], "stride": [V, 1]})
        layers.append({"name": "out", "op": "reshape", "inputs": ["mx"], "shape": [43]})
        return {"layers": layers}, W
    # ---- stage 1: group templates (mean of the group's class templates, per variant) ----
    ng = len(GROUPS)
    Gt = np.stack([unit(F[g].mean(axis=0)) for g in GROUPS])          # (ng, V, D)
    layers.append({"name": "g1", "op": "dense", "inputs": ["fp"], "out": ng * V})
    W["g1.W"] = Gt.reshape(ng * V, D).astype(np.float32); W["g1.b"] = np.zeros(ng * V, np.float32)
    layers.append({"name": "g1r", "op": "reshape", "inputs": ["g1"], "shape": [ng, V, 1]})
    layers.append({"name": "g1m", "op": "maxpool", "inputs": ["g1r"], "size": [V, 1], "stride": [V, 1]})
    layers.append({"name": "g1o", "op": "reshape", "inputs": ["g1m"], "shape": [ng]})
    A = np.zeros((43, ng), np.float32)
    for gi, g in enumerate(GROUPS):
        for k in g:
            A[k, gi] = 1.0
    layers.append({"name": "g2", "op": "dense", "inputs": ["g1o"], "out": 43})
    W["g2.W"] = A; W["g2.b"] = np.zeros(43, np.float32)
    # ---- stage 2: interior templates, mean-subtracted within the group ----
    Dk = np.zeros_like(F)
    for g in GROUPS:
        Ig = F[g] * MintF[g]
        if len(g) > 1:
            Ig = Ig - Ig.mean(axis=0, keepdims=True)
        Dk[g] = unit(Ig)
    layers.append({"name": "d1", "op": "dense", "inputs": ["fp"], "out": 43 * V})
    W["d1.W"] = (p["mu"] * Dk).reshape(43 * V, D).astype(np.float32); W["d1.b"] = np.zeros(43 * V, np.float32)
    layers.append({"name": "d1r", "op": "reshape", "inputs": ["d1"], "shape": [43, V, 1]})
    layers.append({"name": "d1m", "op": "maxpool", "inputs": ["d1r"], "size": [V, 1], "stride": [V, 1]})
    layers.append({"name": "d1o", "op": "reshape", "inputs": ["d1m"], "shape": [43]})
    layers.append({"name": "out", "op": "add", "inputs": ["g2", "d1o"]})
    if verbose:
        print(f"templates {43 * V} (+{ng * V} group), feature dim {D}, dense weights {(43 + ng) * V * D:,}")
    return {"layers": layers}, W


def save(net, W, out_dir):
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "network.json").write_text(json.dumps(net))
    np.savez(out_dir / "weights.npz", **{k: v.astype(np.float32) for k, v in W.items()})

if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    CONFIG = dict(P)
    CONFIG.update(grad_channels=(0,), col_weight=1.5, col_gate=0.1, bar_gain=1.5, two_stage=False, empty_penalty=0.0, norm_gamma=1.0)
    net, W = build_network(CONFIG)
    save(net, W, out_dir)
