import math, json, zlib, struct, sys
from pathlib import Path
import numpy as np

OUT = Path('.')

S = 32
HR = 6
HS = S * HR
BG = np.array([0.55, 0.55, 0.55], np.float32)
YY, XX = np.meshgrid((np.arange(HS) + 0.5) / HR - S / 2, (np.arange(HS) + 0.5) / HR - S / 2, indexing='ij')
X = XX
Y = YY

RED = np.array([0.92, 0.12, 0.12], np.float32)
BLUE = np.array([0.14, 0.36, 0.95], np.float32)
CYAN = np.array([0.33, 0.85, 0.98], np.float32)
YELLOW = np.array([0.96, 0.83, 0.18], np.float32)
WHITE = np.array([0.97, 0.97, 0.97], np.float32)
BLACK = np.array([0.08, 0.08, 0.08], np.float32)
GRAY = np.array([0.55, 0.55, 0.55], np.float32)
DARK = np.array([0.25, 0.25, 0.25], np.float32)
LIGHT = np.array([0.84, 0.84, 0.84], np.float32)
GREEN = np.array([0.18, 0.66, 0.22], np.float32)
ORANGE = np.array([0.95, 0.66, 0.14], np.float32)


def write_png(path, arr):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    raw = b''.join(b'\x00' + arr[y].tobytes() for y in range(h))
    def chunk(tag, data):
        return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag + data) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('!IIBBBBB', w, h, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw, 9))
    png += chunk(b'IEND', b'')
    Path(path).write_bytes(png)


def blank(rgb=BG):
    return np.broadcast_to(np.asarray(rgb, np.float32), (HS, HS, 3)).copy()


def downsample(img):
    return img.reshape(S, HR, S, HR, 3).mean((1, 3))


def fill(img, mask, color):
    img[mask] = color


def circle_mask(r, cx=0.0, cy=0.0, rx=None, ry=None):
    rx = r if rx is None else rx
    ry = r if ry is None else ry
    return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0


def ring_mask(r, t, cx=0.0, cy=0.0, rx=None, ry=None):
    rx = r if rx is None else rx
    ry = r if ry is None else ry
    outer = circle_mask(r, cx, cy, rx, ry)
    inner = circle_mask(max(0.01, r - t), cx, cy, max(0.01, rx - t), max(0.01, ry - t))
    return outer & (~inner)


def polygon_mask(points):
    pts = np.asarray(points, np.float32)
    inside = np.zeros((HS, HS), bool)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        cond = ((yi > Y) != (yj > Y)) & (X < (xj - xi) * (Y - yi) / (yj - yi + 1e-9) + xi)
        inside ^= cond
        j = i
    return inside


def segment_mask(p1, p2, w):
    x1, y1 = p1
    x2, y2 = p2
    vx, vy = x2 - x1, y2 - y1
    den = vx * vx + vy * vy + 1e-9
    t = ((X - x1) * vx + (Y - y1) * vy) / den
    t = np.clip(t, 0.0, 1.0)
    px = x1 + t * vx
    py = y1 + t * vy
    return (X - px) ** 2 + (Y - py) ** 2 <= (w * 0.5) ** 2


def polyline_mask(points, w, closed=False):
    pts = list(points)
    if closed:
        pts = pts + [pts[0]]
    m = np.zeros((HS, HS), bool)
    for a, b in zip(pts[:-1], pts[1:]):
        m |= segment_mask(a, b, w)
    return m


def arc_points(cx, cy, rx, ry, a0, a1, n=24):
    ang = np.linspace(a0, a1, n)
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in ang]


def tf(points, ax=1.0, ay=1.0, dx=0.0, dy=0.0):
    return [(x * ax + dx, y * ay + dy) for x, y in points]


def line(img, p1, p2, w, color):
    fill(img, segment_mask(p1, p2, w), color)


def stroke(img, pts, w, color, closed=False):
    fill(img, polyline_mask(pts, w, closed), color)


def poly(img, pts, color):
    fill(img, polygon_mask(pts), color)


def disc(img, r, color, cx=0.0, cy=0.0, rx=None, ry=None):
    fill(img, circle_mask(r, cx, cy, rx, ry), color)


def ring(img, r, t, color, cx=0.0, cy=0.0, rx=None, ry=None):
    fill(img, ring_mask(r, t, cx, cy, rx, ry), color)


def arrow_path(img, pts, w, color, head_len=3.6, head_w=3.4):
    stroke(img, pts, w, color)
    x2, y2 = pts[-1]
    x1, y1 = pts[-2]
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) + 1e-9
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    tip = (x2, y2)
    left = (x2 - head_len * ux + 0.5 * head_w * px, y2 - head_len * uy + 0.5 * head_w * py)
    right = (x2 - head_len * ux - 0.5 * head_w * px, y2 - head_len * uy - 0.5 * head_w * py)
    poly(img, [tip, left, right], color)


def seven_seg_digit(img, d, cx, cy, w, h, t, color):
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    ym = cy
    segs = {
        'a': ((x1, y1), (x2, y1)),
        'b': ((x2, y1), (x2, ym)),
        'c': ((x2, ym), (x2, y2)),
        'd': ((x1, y2), (x2, y2)),
        'e': ((x1, ym), (x1, y2)),
        'f': ((x1, y1), (x1, ym)),
        'g': ((x1, ym), (x2, ym)),
    }
    uses = {
        '2': 'abged',
        '3': 'abgcd',
        '5': 'afgcd',
        '6': 'afgecd',
        '7': 'abc',
        '8': 'abcdefg',
    }[d]
    for s in uses:
        line(img, *segs[s], t, color)


def digit(img, d, cx, cy, s, color):
    if d == '0':
        ring(img, 2.6 * s, 1.15 * s, color, cx, cy, 2.1 * s, 3.0 * s)
    elif d == '1':
        line(img, (cx, cy - 3.4 * s), (cx, cy + 3.4 * s), 1.15 * s, color)
        line(img, (cx - 1.6 * s, cy - 1.8 * s), (cx, cy - 3.4 * s), 1.0 * s, color)
        line(img, (cx - 1.7 * s, cy + 3.4 * s), (cx + 1.7 * s, cy + 3.4 * s), 1.0 * s, color)
    elif d in '235678':
        seven_seg_digit(img, d, cx, cy, 4.2 * s, 6.8 * s, 1.05 * s, color)
    else:
        raise ValueError(d)


def digits(img, text, cx, cy, s, color):
    n = len(text)
    spacing = 5.5 * s if n == 1 else (5.0 * s if n == 2 else 4.8 * s)
    start = cx - spacing * (n - 1) / 2
    for i, ch in enumerate(text):
        digit(img, ch, start + i * spacing, cy, s, color)


def car(img, cx, cy, s, color):
    poly(img, tf([(-3.8, 2.0), (4.0, 2.0), (3.2, 4.8), (-4.0, 4.8)], s, s, cx, cy), color)
    poly(img, tf([(-2.8, 2.0), (-1.0, -0.4), (1.8, -0.4), (3.4, 2.0)], s, s, cx, cy), color)


def truck(img, cx, cy, s, color):
    poly(img, tf([(-4.6, 1.0), (1.8, 1.0), (1.8, 4.8), (-4.6, 4.8)], s, s, cx, cy), color)
    poly(img, tf([(1.8, 1.0), (4.2, 1.0), (4.8, 2.2), (4.8, 4.8), (1.8, 4.8)], s, s, cx, cy), color)


def pedestrian(img, cx, cy, s, color):
    disc(img, 0.95 * s, color, cx, cy - 4.2 * s)
    line(img, (cx, cy - 3.0 * s), (cx, cy + 1.2 * s), 1.0 * s, color)
    line(img, (cx, cy - 1.0 * s), (cx - 2.0 * s, cy + 0.5 * s), 0.9 * s, color)
    line(img, (cx, cy - 0.8 * s), (cx + 1.8 * s, cy + 0.6 * s), 0.9 * s, color)
    line(img, (cx, cy + 1.2 * s), (cx - 2.0 * s, cy + 4.8 * s), 1.0 * s, color)
    line(img, (cx, cy + 1.2 * s), (cx + 1.6 * s, cy + 4.2 * s), 1.0 * s, color)


def children(img, cx, cy, s, color):
    pedestrian(img, cx - 1.6 * s, cy + 0.2 * s, 0.95 * s, color)
    pedestrian(img, cx + 1.6 * s, cy - 0.8 * s, 0.78 * s, color)
    line(img, (cx - 0.3 * s, cy - 0.3 * s), (cx + 0.8 * s, cy - 0.4 * s), 0.7 * s, color)


def bicycle(img, cx, cy, s, color):
    ring(img, 1.8 * s, 0.45 * s, color, cx - 3.5 * s, cy + 2.4 * s)
    ring(img, 1.8 * s, 0.45 * s, color, cx + 3.5 * s, cy + 2.4 * s)
    line(img, (cx - 3.5 * s, cy + 2.4 * s), (cx - 0.7 * s, cy + 0.7 * s), 0.6 * s, color)
    line(img, (cx - 0.7 * s, cy + 0.7 * s), (cx + 1.2 * s, cy + 2.4 * s), 0.6 * s, color)
    line(img, (cx + 1.2 * s, cy + 2.4 * s), (cx - 1.0 * s, cy + 2.4 * s), 0.6 * s, color)
    line(img, (cx - 1.0 * s, cy + 2.4 * s), (cx - 3.5 * s, cy + 2.4 * s), 0.6 * s, color)
    line(img, (cx - 0.7 * s, cy + 0.7 * s), (cx + 0.0 * s, cy - 1.0 * s), 0.6 * s, color)
    line(img, (cx + 0.0 * s, cy - 1.0 * s), (cx + 1.7 * s, cy - 1.0 * s), 0.6 * s, color)
    line(img, (cx + 1.2 * s, cy + 2.4 * s), (cx + 2.5 * s, cy + 0.2 * s), 0.6 * s, color)


def snowflake(img, cx, cy, s, color):
    for ang in np.linspace(0, math.pi, 3, endpoint=False):
        dx, dy = math.cos(ang) * 4.0 * s, math.sin(ang) * 4.0 * s
        line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), 0.7 * s, color)
        for sign in (-1, 1):
            ex, ey = cx + sign * dx, cy + sign * dy
            px, py = -dy, dx
            pn = math.hypot(px, py) + 1e-9
            px, py = px / pn, py / pn
            bx, by = -dx / (math.hypot(dx, dy) + 1e-9), -dy / (math.hypot(dx, dy) + 1e-9)
            line(img, (ex, ey), (ex + 1.4 * s * (px + bx), ey + 1.4 * s * (py + by)), 0.55 * s, color)
            line(img, (ex, ey), (ex + 1.4 * s * (-px + bx), ey + 1.4 * s * (-py + by)), 0.55 * s, color)


def worker(img, cx, cy, s, color):
    disc(img, 0.85 * s, color, cx - 0.5 * s, cy - 4.2 * s)
    line(img, (cx - 0.5 * s, cy - 3.3 * s), (cx - 0.4 * s, cy - 0.6 * s), 0.95 * s, color)
    line(img, (cx - 0.4 * s, cy - 1.8 * s), (cx + 2.0 * s, cy - 0.2 * s), 0.8 * s, color)
    line(img, (cx - 0.4 * s, cy - 0.6 * s), (cx - 2.0 * s, cy + 2.6 * s), 0.9 * s, color)
    line(img, (cx - 0.4 * s, cy - 0.6 * s), (cx + 1.0 * s, cy + 3.2 * s), 0.9 * s, color)
    line(img, (cx + 1.8 * s, cy - 0.2 * s), (cx + 3.9 * s, cy + 4.6 * s), 0.8 * s, color)
    poly(img, tf([(2.0, 3.8), (4.7, 3.0), (5.2, 4.6)], s, s, cx, cy), color)
    poly(img, tf([(-3.4, 5.2), (0.6, 3.9), (3.7, 5.2)], s, s, cx, cy), color)


def traffic_light(img, cx, cy, s, dark_color=BLACK):
    poly(img, tf([(-1.6, -5.0), (1.6, -5.0), (1.6, 4.8), (-1.6, 4.8)], s, s, cx, cy), dark_color)
    disc(img, 0.95 * s, RED, cx, cy - 3.2 * s)
    disc(img, 0.95 * s, ORANGE, cx, cy - 0.1 * s)
    disc(img, 0.95 * s, GREEN, cx, cy + 3.0 * s)


def narrows_right(img, cx, cy, s, color):
    line(img, (cx - 3.4 * s, cy - 5.2 * s), (cx - 3.4 * s, cy + 5.1 * s), 1.0 * s, color)
    line(img, (cx + 3.5 * s, cy - 5.0 * s), (cx + 0.7 * s, cy + 5.1 * s), 1.0 * s, color)


def exclamation(img, cx, cy, s, color):
    line(img, (cx, cy - 5.2 * s), (cx, cy + 1.8 * s), 1.4 * s, color)
    disc(img, 1.0 * s, color, cx, cy + 5.0 * s)


def priority_cross(img, cx, cy, s, color):
    line(img, (cx - 4.6 * s, cy - 4.6 * s), (cx + 4.6 * s, cy + 4.6 * s), 1.1 * s, color)
    line(img, (cx + 4.6 * s, cy - 4.6 * s), (cx - 4.6 * s, cy + 4.6 * s), 1.1 * s, color)


def bend(img, cx, cy, s, color, right=False, double=False):
    if double:
        pts = [(-1.8, 5.0), (-1.8, 1.8), (1.2, -0.6), (1.2, -4.8)]
        pts = tf(pts, s, s, cx, cy)
        stroke(img, pts, 1.4 * s, color)
    else:
        pts = [(-2.6, 5.0), (-2.6, 1.5), (-0.8, -1.0), (2.0, -4.6)]
        if right:
            pts = [(-x, y) for x, y in pts]
        pts = tf(pts, s, s, cx, cy)
        stroke(img, pts, 1.4 * s, color)


def bumps(img, cx, cy, s, color):
    pts = []
    for u in np.linspace(-4.8, 4.8, 24):
        pts.append((cx + u * s, cy + 2.4 * s - 1.4 * s * math.sin((u + 4.8) / 9.6 * math.pi * 3)))
    stroke(img, pts, 1.0 * s, color)


def slippery(img, cx, cy, s, color):
    car(img, cx + 1.5 * s, cy - 1.0 * s, 0.8 * s, color)
    pts1 = tf([(-3.8, 3.8), (-2.2, 2.8), (-3.0, 1.2), (-1.2, 0.2)], s, s, cx, cy)
    pts2 = tf([(-1.6, 5.2), (0.2, 4.0), (-0.2, 2.5), (1.7, 1.3)], s, s, cx, cy)
    stroke(img, pts1, 0.7 * s, color)
    stroke(img, pts2, 0.7 * s, color)


def deer(img, cx, cy, s, color):
    poly(img, tf([(-3.6, 1.0), (0.5, -1.2), (3.8, 0.0), (2.0, 2.6), (-1.5, 2.4)], s, s, cx, cy), color)
    poly(img, tf([(2.0, -1.0), (3.6, -3.8), (4.4, -2.7), (3.1, -0.6)], s, s, cx, cy), color)
    line(img, (cx - 2.0 * s, cy + 2.2 * s), (cx - 2.8 * s, cy + 5.4 * s), 0.8 * s, color)
    line(img, (cx - 0.4 * s, cy + 2.2 * s), (cx - 0.9 * s, cy + 5.0 * s), 0.8 * s, color)
    line(img, (cx + 1.2 * s, cy + 1.8 * s), (cx + 1.0 * s, cy + 4.8 * s), 0.8 * s, color)
    line(img, (cx + 3.0 * s, cy + 1.2 * s), (cx + 2.6 * s, cy + 4.0 * s), 0.8 * s, color)
    line(img, (cx + 4.0 * s, cy - 3.2 * s), (cx + 5.2 * s, cy - 4.8 * s), 0.6 * s, color)
    line(img, (cx + 4.0 * s, cy - 3.2 * s), (cx + 5.6 * s, cy - 3.6 * s), 0.6 * s, color)


def roundabout(img, cx, cy, s, color):
    for ang in (0.2, 2.3, 4.4):
        pts = []
        for a in np.linspace(ang, ang + 1.7, 18):
            pts.append((cx + 4.6 * s * math.cos(a), cy + 4.6 * s * math.sin(a)))
        arrow_path(img, pts, 1.0 * s, color, 2.1 * s, 2.0 * s)


def arrow_kind(img, kind, cx, cy, s, color):
    if kind == 'right':
        pts = tf([(0.0, 4.8), (0.0, -1.2), (4.0, -5.0)], s, s, cx, cy)
        arrow_path(img, pts, 1.5 * s, color)
    elif kind == 'left':
        pts = tf([(0.0, 4.8), (0.0, -1.2), (-4.0, -5.0)], s, s, cx, cy)
        arrow_path(img, pts, 1.5 * s, color)
    elif kind == 'ahead':
        pts = tf([(0.0, 5.2), (0.0, -5.2)], s, s, cx, cy)
        arrow_path(img, pts, 1.8 * s, color)
    elif kind == 'ahead_right':
        pts = tf([(0.0, 5.2), (0.0, -0.6), (3.8, -4.2)], s, s, cx, cy)
        arrow_path(img, pts, 1.5 * s, color)
        arrow_path(img, tf([(0.0, 5.2), (0.0, -5.2)], s, s, cx, cy), 1.2 * s, color, 2.6 * s, 2.4 * s)
    elif kind == 'ahead_left':
        pts = tf([(0.0, 5.2), (0.0, -0.6), (-3.8, -4.2)], s, s, cx, cy)
        arrow_path(img, pts, 1.5 * s, color)
        arrow_path(img, tf([(0.0, 5.2), (0.0, -5.2)], s, s, cx, cy), 1.2 * s, color, 2.6 * s, 2.4 * s)
    elif kind == 'keep_right':
        pts = tf([(3.5, -4.2), (0.6, -1.2), (-1.6, 2.0), (-3.8, 4.8)], s, s, cx, cy)
        arrow_path(img, pts, 1.7 * s, color)
    elif kind == 'keep_left':
        pts = tf([(-3.5, -4.2), (-0.6, -1.2), (1.6, 2.0), (3.8, 4.8)], s, s, cx, cy)
        arrow_path(img, pts, 1.7 * s, color)
    else:
        raise ValueError(kind)


def warning_triangle(img, ax=1.0, ay=1.0):
    outer = tf([(0, -11.7), (-10.8, 8.8), (10.8, 8.8)], ax, ay)
    inner = tf([(0, -8.3), (-7.8, 6.3), (7.8, 6.3)], ax, ay)
    poly(img, outer, RED)
    poly(img, inner, WHITE)


def yield_triangle(img, ax=1.0, ay=1.0):
    outer = tf([(0, 11.4), (-11.7, -8.8), (11.7, -8.8)], ax, ay)
    inner = tf([(0, 7.8), (-8.3, -6.0), (8.3, -6.0)], ax, ay)
    poly(img, outer, RED)
    poly(img, inner, WHITE)


def octagon_points(r=11.4, ax=1.0, ay=1.0):
    pts = []
    for a in np.linspace(math.pi / 8, 2 * math.pi + math.pi / 8, 8, endpoint=False):
        pts.append((r * math.cos(a) * ax, r * math.sin(a) * ay))
    return pts


def diamond(img, ax=1.0, ay=1.0):
    outer = tf([(0, -12.4), (12.4, 0), (0, 12.4), (-12.4, 0)], ax, ay)
    mid = tf([(0, -11.2), (11.2, 0), (0, 11.2), (-11.2, 0)], ax, ay)
    inner = tf([(0, -8.5), (8.5, 0), (0, 8.5), (-8.5, 0)], ax, ay)
    poly(img, outer, BLACK)
    poly(img, mid, WHITE)
    poly(img, inner, YELLOW)


def speed_base(img, ax=1.0, ay=1.0):
    disc(img, 12.1, WHITE, 0, 0, 12.1 * ax, 12.1 * ay)
    ring(img, 12.1, 2.7, RED, 0, 0, 12.1 * ax, 12.1 * ay)


def end_base(img, ax=1.0, ay=1.0):
    disc(img, 12.0, LIGHT, 0, 0, 12.0 * ax, 12.0 * ay)
    ring(img, 12.0, 1.1, DARK, 0, 0, 12.0 * ax, 12.0 * ay)


def mandatory_base(img, ax=1.0, ay=1.0, bright=False):
    c = CYAN if bright else BLUE
    disc(img, 12.0, c, 0, 0, 12.0 * ax, 12.0 * ay)


def no_entry_base(img, ax=1.0, ay=1.0):
    disc(img, 12.0, RED, 0, 0, 12.0 * ax, 12.0 * ay)
    poly(img, tf([(-8.0, -2.3), (8.0, -2.3), (8.0, 2.3), (-8.0, 2.3)], ax, ay), WHITE)


def draw_sign(cls, ax=1.0, ay=1.0):
    img = blank()
    if cls in (0, 1, 2, 3, 4, 5, 7, 8):
        speed_base(img, ax, ay)
        txt = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}[cls]
        digits(img, txt, 0, 1.5 * ay, 1.15 * min(ax, ay), BLACK)
    elif cls == 6:
        end_base(img, ax, ay)
        digits(img, '80', 0, 1.2 * ay, 1.05 * min(ax, ay), DARK)
        for off in (-7.8, -1.5, 4.8):
            line(img, (-10.5 * ax + off * 0.02, -9.0 * ay + off), (9.5 * ax + off * 0.02, 11.0 * ay + off), 1.3 * min(ax, ay), DARK)
    elif cls == 9:
        speed_base(img, ax, ay)
        car(img, -3.5 * ax, 1.0 * ay, 1.0 * min(ax, ay), DARK)
        car(img, 3.5 * ax, 0.4 * ay, 1.0 * min(ax, ay), RED)
    elif cls == 10:
        speed_base(img, ax, ay)
        truck(img, -3.3 * ax, 0.8 * ay, 0.9 * min(ax, ay), DARK)
        car(img, 3.6 * ax, 1.0 * ay, 0.8 * min(ax, ay), RED)
    elif cls == 11:
        warning_triangle(img, ax, ay)
        priority_cross(img, 0, 0.8 * ay, 0.95 * min(ax, ay), BLACK)
    elif cls == 12:
        diamond(img, ax, ay)
    elif cls == 13:
        yield_triangle(img, ax, ay)
    elif cls == 14:
        poly(img, octagon_points(11.3, ax, ay), RED)
    elif cls == 15:
        speed_base(img, ax, ay)
    elif cls == 16:
        speed_base(img, ax, ay)
        truck(img, 0.0, 0.8 * ay, 1.0 * min(ax, ay), DARK)
    elif cls == 17:
        no_entry_base(img, ax, ay)
    elif cls == 18:
        warning_triangle(img, ax, ay)
        exclamation(img, 0, 0.3 * ay, 1.0 * min(ax, ay), BLACK)
    elif cls == 19:
        warning_triangle(img, ax, ay)
        bend(img, 0, 1.0 * ay, 1.0 * min(ax, ay), BLACK, right=False)
    elif cls == 20:
        warning_triangle(img, ax, ay)
        bend(img, 0, 1.0 * ay, 1.0 * min(ax, ay), BLACK, right=True)
    elif cls == 21:
        warning_triangle(img, ax, ay)
        bend(img, 0, 1.0 * ay, 1.0 * min(ax, ay), BLACK, double=True)
    elif cls == 22:
        warning_triangle(img, ax, ay)
        bumps(img, 0, 0.6 * ay, 1.0 * min(ax, ay), BLACK)
    elif cls == 23:
        warning_triangle(img, ax, ay)
        slippery(img, 0, 0.4 * ay, 0.95 * min(ax, ay), BLACK)
    elif cls == 24:
        warning_triangle(img, ax, ay)
        narrows_right(img, 0, 0.8 * ay, 0.95 * min(ax, ay), BLACK)
    elif cls == 25:
        warning_triangle(img, ax, ay)
        worker(img, 0, 0.4 * ay, 0.95 * min(ax, ay), BLACK)
    elif cls == 26:
        warning_triangle(img, ax, ay)
        traffic_light(img, 0, 0.8 * ay, 0.85 * min(ax, ay), BLACK)
    elif cls == 27:
        warning_triangle(img, ax, ay)
        pedestrian(img, 0, 0.5 * ay, 0.95 * min(ax, ay), BLACK)
    elif cls == 28:
        warning_triangle(img, ax, ay)
        children(img, 0, 0.9 * ay, 0.95 * min(ax, ay), BLACK)
    elif cls == 29:
        warning_triangle(img, ax, ay)
        bicycle(img, 0, 1.0 * ay, 0.8 * min(ax, ay), BLACK)
    elif cls == 30:
        warning_triangle(img, ax, ay)
        snowflake(img, 0, 0.8 * ay, 0.9 * min(ax, ay), BLACK)
    elif cls == 31:
        warning_triangle(img, ax, ay)
        deer(img, 0, 1.2 * ay, 0.9 * min(ax, ay), BLACK)
    elif cls == 32:
        end_base(img, ax, ay)
        for off in (-7.8, -1.5, 4.8):
            line(img, (-10.5 * ax + off * 0.02, -9.0 * ay + off), (9.5 * ax + off * 0.02, 11.0 * ay + off), 1.6 * min(ax, ay), DARK)
    elif cls == 33:
        mandatory_base(img, ax, ay)
        arrow_kind(img, 'right', 0, 0.2 * ay, 0.95 * min(ax, ay), WHITE)
    elif cls == 34:
        mandatory_base(img, ax, ay)
        arrow_kind(img, 'left', 0, 0.2 * ay, 0.95 * min(ax, ay), WHITE)
    elif cls == 35:
        mandatory_base(img, ax, ay, bright=True)
        arrow_kind(img, 'ahead', 0, 0.2 * ay, 0.95 * min(ax, ay), WHITE)
    elif cls == 36:
        mandatory_base(img, ax, ay)
        arrow_kind(img, 'ahead_right', 0, 0.2 * ay, 0.9 * min(ax, ay), WHITE)
    elif cls == 37:
        mandatory_base(img, ax, ay)
        arrow_kind(img, 'ahead_left', 0, 0.2 * ay, 0.9 * min(ax, ay), WHITE)
    elif cls == 38:
        mandatory_base(img, ax, ay)
        arrow_kind(img, 'keep_right', 0, 0.2 * ay, 0.95 * min(ax, ay), WHITE)
    elif cls == 39:
        mandatory_base(img, ax, ay)
        arrow_kind(img, 'keep_left', 0, 0.2 * ay, 0.95 * min(ax, ay), WHITE)
    elif cls == 40:
        mandatory_base(img, ax, ay)
        roundabout(img, 0, 0.2 * ay, 0.95 * min(ax, ay), WHITE)
    elif cls == 41:
        end_base(img, ax, ay)
        car(img, -3.4 * ax, 0.8 * ay, 0.95 * min(ax, ay), DARK)
        car(img, 3.4 * ax, 0.4 * ay, 0.95 * min(ax, ay), GRAY)
        line(img, (-10.5 * ax, -9.0 * ay), (9.5 * ax, 11.0 * ay), 1.4 * min(ax, ay), DARK)
    elif cls == 42:
        end_base(img, ax, ay)
        truck(img, -3.2 * ax, 0.8 * ay, 0.85 * min(ax, ay), DARK)
        car(img, 3.8 * ax, 1.0 * ay, 0.75 * min(ax, ay), GRAY)
        line(img, (-10.5 * ax, -9.0 * ay), (9.5 * ax, 11.0 * ay), 1.4 * min(ax, ay), DARK)
    else:
        raise ValueError(cls)
    return np.clip(downsample(img), 0, 1)


def contact_sheet(images, cols=6, pad=2, scale=4):
    imgs = [np.repeat(np.repeat((im * 255).astype(np.uint8), scale, 0), scale, 1) for im in images]
    h, w = imgs[0].shape[:2]
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.full((pad + rows * (h + pad), pad + cols * (w + pad), 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = im
    return canvas

VARIANTS = [(1.0, 1.0), (1.10, 0.92), (0.92, 1.10), (0.90, 0.90)]
RAW_SCALE = 1.0
NORM_SCALE = 0.7
BG_CHW = BG.reshape(3, 1, 1)


def chw(img):
    return np.transpose(img, (2, 0, 1)).astype(np.float32)


def channel_norm(img_chw):
    mu = img_chw.mean(axis=(1, 2), keepdims=True)
    var = ((img_chw - mu) ** 2).mean(axis=(1, 2), keepdims=True)
    return (img_chw - mu) / np.sqrt(var + 1e-5)


def build_weights():
    raw_rows = []
    norm_rows = []
    for cls in range(43):
        for ax, ay in VARIANTS:
            p = chw(draw_sign(cls, ax, ay))
            q = p - BG_CHW
            q = q.reshape(-1)
            q = RAW_SCALE * q / (np.linalg.norm(q) + 1e-6)
            pn = channel_norm(p).reshape(-1)
            pn = NORM_SCALE * pn / (np.linalg.norm(pn) + 1e-6)
            raw_rows.append(q.astype(np.float32))
            norm_rows.append(pn.astype(np.float32))
    raw_W = np.stack(raw_rows).astype(np.float32)
    norm_W = np.stack(norm_rows).astype(np.float32)
    raw_b = np.zeros(raw_W.shape[0], np.float32)
    norm_b = np.zeros(norm_W.shape[0], np.float32)
    return raw_W, raw_b, norm_W, norm_b


def write_answer(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_W, raw_b, norm_W, norm_b = build_weights()
    net = {
        'layers': [
            {'name': 'raw', 'op': 'dense', 'inputs': ['image'], 'out': 172},
            {'name': 'inorm', 'op': 'norm', 'inputs': ['image']},
            {'name': 'nrm', 'op': 'dense', 'inputs': ['inorm'], 'out': 172},
            {'name': 'sum', 'op': 'add', 'inputs': ['raw', 'nrm']},
            {'name': 'resh', 'op': 'reshape', 'inputs': ['sum'], 'shape': [43, 1, 4]},
            {'name': 'pick', 'op': 'maxpool', 'inputs': ['resh'], 'size': [1, 4], 'stride': [1, 4]},
            {'name': 'scores', 'op': 'reshape', 'inputs': ['pick'], 'shape': [43]},
        ]
    }
    (out / 'network.json').write_text(json.dumps(net, separators=(',', ':')))
    np.savez(out / 'weights.npz', **{
        'raw.W': raw_W,
        'raw.b': raw_b,
        'nrm.W': norm_W,
        'nrm.b': norm_b,
    })


if __name__ == '__main__':
    write_answer(sys.argv[1] if len(sys.argv) > 1 else '.')
