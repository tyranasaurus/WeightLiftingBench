import json
import sys
from pathlib import Path

import numpy as np

RES = 16
SS = 8
N = RES * SS
CX = CY = RES / 2.0

MID = np.array([0.50, 0.50, 0.50], np.float32)
VRED = np.array([0.92, 0.12, 0.10], np.float32)
VBLUE = np.array([0.14, 0.34, 0.92], np.float32)
VWHITE = np.array([0.95, 0.95, 0.95], np.float32)
VBLACK = np.array([0.10, 0.10, 0.10], np.float32)
VYELLOW = np.array([0.94, 0.80, 0.12], np.float32)
VGREY = np.array([0.72, 0.72, 0.72], np.float32)
VDARK = np.array([0.30, 0.30, 0.30], np.float32)
VGREEN = np.array([0.12, 0.76, 0.16], np.float32)

u = (np.arange(N, dtype=np.float32) + 0.5) / SS
Y, X = np.meshgrid(u, u, indexing="ij")

FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
}


def canvas():
    return np.broadcast_to(MID[:, None, None], (3, N, N)).copy()


def downsample(img):
    return img.reshape(3, RES, SS, RES, SS).mean(axis=(2, 4)).astype(np.float32)


def rect(x1, y1, x2, y2):
    return (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)


def circle(r, cx=CX, cy=CY):
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r


def ring(r1, r0, cx=CX, cy=CY):
    d = (X - cx) ** 2 + (Y - cy) ** 2
    return (d <= r1 * r1) & (d >= r0 * r0)


def regular_polygon(n, r, rot=0.0, cx=CX, cy=CY):
    return [
        (cx + r * np.cos(rot + 2 * np.pi * i / n), cy + r * np.sin(rot + 2 * np.pi * i / n))
        for i in range(n)
    ]


def scale_points(points, factor, cx=CX, cy=CY):
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in points]


def polygon(points):
    pts = np.asarray(points, np.float32)
    c = pts.mean(axis=0)
    m = np.ones_like(X, dtype=bool)
    for p1, p2 in zip(pts, np.roll(pts, -1, axis=0)):
        cross = (X - p1[0]) * (p2[1] - p1[1]) - (Y - p1[1]) * (p2[0] - p1[0])
        ref = (c[0] - p1[0]) * (p2[1] - p1[1]) - (c[1] - p1[1]) * (p2[0] - p1[0])
        m &= cross * ref >= 0
    return m


def dist_seg(x1, y1, x2, y2):
    vx = x2 - x1
    vy = y2 - y1
    t = ((X - x1) * vx + (Y - y1) * vy) / (vx * vx + vy * vy + 1e-6)
    t = np.clip(t, 0.0, 1.0)
    px = x1 + t * vx
    py = y1 + t * vy
    return np.sqrt((X - px) ** 2 + (Y - py) ** 2)


def stroke(points, width):
    d = np.full_like(X, 1e9, dtype=np.float32)
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        d = np.minimum(d, dist_seg(x1, y1, x2, y2))
    return d <= width * 0.5


def paint(img, mask, color, alpha=1.0):
    m = np.asarray(mask, np.float32) * float(alpha)
    img *= (1.0 - m)[None, :, :]
    img += np.asarray(color, np.float32)[:, None, None] * m[None, :, :]


def draw_char(img, ch, x, y, s, color):
    pat = FONT[ch]
    for r, row in enumerate(pat):
        for c, v in enumerate(row):
            if v == "1":
                paint(img, rect(x + c * s, y + r * s, x + (c + 1) * s, y + (r + 1) * s), color)


def draw_text_center(img, text, cx, cy, height, color, gap=1.0):
    s = height / 7.0
    total = s * (5 * len(text) + gap * (len(text) - 1))
    x = cx - total / 2.0
    y = cy - height / 2.0
    for i, ch in enumerate(text):
        draw_char(img, ch, x, y, s, color)
        x += s * (5 + gap)


def draw_arrow(img, points, width=1.0, head_len=1.6, head_w=1.2, color=VWHITE):
    paint(img, stroke(points, width), color)
    x2, y2 = points[-1]
    x1, y1 = points[-2]
    vx = x2 - x1
    vy = y2 - y1
    n = (vx * vx + vy * vy) ** 0.5 + 1e-6
    ux, uy = vx / n, vy / n
    px, py = -uy, ux
    head = [
        (x2, y2),
        (x2 - head_len * ux + head_w * px, y2 - head_len * uy + head_w * py),
        (x2 - head_len * ux - head_w * px, y2 - head_len * uy - head_w * py),
    ]
    paint(img, polygon(head), color)


def draw_car(img, cx, cy, s, color):
    paint(img, rect(cx - 1.6 * s, cy - 0.3 * s, cx + 1.6 * s, cy + 0.6 * s), color)
    roof = [(cx - 0.8 * s, cy - 0.3 * s), (cx - 0.2 * s, cy - 1.0 * s), (cx + 0.9 * s, cy - 1.0 * s), (cx + 1.3 * s, cy - 0.3 * s)]
    paint(img, polygon(roof), color)


def draw_truck(img, cx, cy, s, color):
    paint(img, rect(cx - 1.9 * s, cy - 0.4 * s, cx + 0.8 * s, cy + 0.7 * s), color)
    paint(img, rect(cx + 0.8 * s, cy - 0.1 * s, cx + 1.7 * s, cy + 0.7 * s), color)
    paint(img, rect(cx + 1.1 * s, cy - 0.6 * s, cx + 1.7 * s, cy - 0.1 * s), color)


def circle_sign(img, radius=6.4, border=1.4, edge=VRED, fill=VWHITE):
    paint(img, circle(radius), edge)
    paint(img, circle(radius - border), fill)


def blue_disc(img):
    paint(img, circle(6.4), VBLUE)


def warning_triangle(img):
    outer = [(8.0, 1.8), (1.7, 13.8), (14.3, 13.8)]
    inner = scale_points(outer, 0.74)
    paint(img, polygon(outer), VRED)
    paint(img, polygon(inner), VWHITE)


def yield_sign(img):
    outer = [(8.0, 14.3), (1.8, 3.0), (14.2, 3.0)]
    inner = scale_points(outer, 0.72)
    paint(img, polygon(outer), VRED)
    paint(img, polygon(inner), VWHITE)


def priority_road(img):
    outer = [(8.0, 1.4), (14.6, 8.0), (8.0, 14.6), (1.4, 8.0)]
    middle = scale_points(outer, 0.90)
    inner = scale_points(outer, 0.70)
    paint(img, polygon(outer), VBLACK)
    paint(img, polygon(middle), VWHITE)
    paint(img, polygon(inner), VYELLOW)


def stop_sign(img):
    pts = regular_polygon(8, 5.9, rot=np.pi / 8.0)
    paint(img, polygon(pts), VRED)
    draw_text_center(img, "STOP", 8.0, 8.1, 4.2, VWHITE, gap=0.9)


def no_entry(img):
    paint(img, circle(6.2), VRED)
    paint(img, rect(3.6, 7.0, 12.4, 9.0), VWHITE)


def end_circle_base(img):
    paint(img, circle(6.4), VGREY)
    paint(img, circle(5.2), VWHITE)


def diag_band(offset=0.0, width=1.2):
    return np.abs((X - Y) - offset) <= width * 0.5


def end_stripes(img, offsets):
    for off in offsets:
        paint(img, diag_band(off, 1.1), VDARK)


def sign_speed(text):
    img = canvas()
    circle_sign(img)
    h = 5.0 if len(text) == 2 else 4.2
    draw_text_center(img, text, 8.0, 8.5, h, VBLACK, gap=1.0)
    return downsample(img)


def sign_end_speed_80():
    img = canvas()
    end_circle_base(img)
    draw_text_center(img, "80", 8.0, 8.5, 5.0, VGREY, gap=1.0)
    end_stripes(img, [-2.2, 0.0, 2.2])
    return downsample(img)


def sign_no_passing(truck=False, end=False):
    img = canvas()
    if end:
        end_circle_base(img)
    else:
        circle_sign(img)
    if truck:
        draw_truck(img, 6.1, 9.1, 1.1, VDARK if not end else VGREY)
        draw_car(img, 10.1, 9.1, 0.9, VRED if not end else VBLACK)
    else:
        draw_car(img, 6.2, 9.0, 1.0, VRED if not end else VGREY)
        draw_car(img, 10.0, 9.0, 1.0, VDARK)
    if end:
        end_stripes(img, [-1.8, 1.4])
    return downsample(img)


def sign_priority_next():
    img = canvas()
    warning_triangle(img)
    paint(img, stroke([(8.0, 5.0), (8.0, 11.8)], 1.1), VBLACK)
    paint(img, stroke([(5.4, 7.0), (8.0, 8.4), (10.7, 6.7)], 1.0), VBLACK)
    return downsample(img)


def sign_general_danger():
    img = canvas()
    warning_triangle(img)
    paint(img, rect(7.4, 5.1, 8.6, 10.4), VBLACK)
    paint(img, circle(0.7, 8.0, 12.1), VBLACK)
    return downsample(img)


def sign_bend(left=True):
    img = canvas()
    warning_triangle(img)
    pts = [(10.5, 4.8), (8.7, 5.8), (7.7, 7.3), (7.1, 9.4), (7.0, 11.7)]
    if not left:
        pts = [(16.0 - x, y) for x, y in pts]
    paint(img, stroke(pts, 1.2), VBLACK)
    return downsample(img)


def sign_double_bend():
    img = canvas()
    warning_triangle(img)
    pts = [(10.8, 4.9), (9.0, 5.7), (8.0, 6.9), (8.8, 8.5), (9.9, 10.2), (9.0, 11.8)]
    paint(img, stroke(pts, 1.1), VBLACK)
    return downsample(img)


def sign_bumpy():
    img = canvas()
    warning_triangle(img)
    pts = [(4.9, 10.9), (6.3, 9.1), (7.8, 10.9), (9.3, 9.1), (10.8, 10.9)]
    paint(img, stroke(pts, 1.0), VBLACK)
    return downsample(img)


def sign_slippery():
    img = canvas()
    warning_triangle(img)
    draw_car(img, 9.1, 7.8, 0.7, VBLACK)
    paint(img, stroke([(5.3, 10.8), (6.4, 9.8), (7.2, 10.8)], 0.6), VBLACK)
    paint(img, stroke([(6.0, 9.3), (7.2, 8.4), (8.0, 9.3)], 0.6), VBLACK)
    return downsample(img)


def sign_narrows_right():
    img = canvas()
    warning_triangle(img)
    paint(img, stroke([(5.9, 11.7), (6.2, 5.2)], 1.0), VBLACK)
    paint(img, stroke([(10.8, 11.7), (8.5, 5.2)], 1.0), VBLACK)
    return downsample(img)


def sign_roadworks():
    img = canvas()
    warning_triangle(img)
    paint(img, circle(0.7, 8.0, 5.2), VBLACK)
    paint(img, stroke([(8.0, 6.0), (7.3, 8.3), (6.7, 10.3)], 0.9), VBLACK)
    paint(img, stroke([(8.4, 7.0), (10.5, 9.6)], 0.8), VBLACK)
    paint(img, stroke([(6.0, 11.2), (10.8, 11.2)], 0.8), VBLACK)
    return downsample(img)


def sign_traffic_lights():
    img = canvas()
    warning_triangle(img)
    paint(img, rect(6.7, 4.8, 9.3, 11.8), VBLACK)
    paint(img, circle(0.75, 8.0, 6.0), VRED)
    paint(img, circle(0.75, 8.0, 8.3), VYELLOW)
    paint(img, circle(0.75, 8.0, 10.6), VGREEN)
    return downsample(img)


def sign_pedestrians(two=False):
    img = canvas()
    warning_triangle(img)
    if two:
        paint(img, circle(0.55, 6.5, 6.0), VBLACK)
        paint(img, stroke([(6.5, 6.7), (6.1, 8.4), (5.5, 10.2)], 0.7), VBLACK)
        paint(img, stroke([(6.4, 8.0), (7.5, 9.0)], 0.6), VBLACK)
        paint(img, stroke([(6.1, 8.6), (7.0, 10.6)], 0.6), VBLACK)
        paint(img, circle(0.55, 9.3, 5.4), VBLACK)
        paint(img, stroke([(9.3, 6.1), (8.8, 7.9), (8.4, 9.8)], 0.7), VBLACK)
        paint(img, stroke([(9.1, 7.4), (10.4, 8.4)], 0.6), VBLACK)
        paint(img, stroke([(8.8, 8.3), (10.1, 10.2)], 0.6), VBLACK)
    else:
        paint(img, circle(0.65, 7.7, 5.8), VBLACK)
        paint(img, stroke([(7.7, 6.5), (7.1, 8.3), (6.7, 10.1)], 0.8), VBLACK)
        paint(img, stroke([(7.5, 7.7), (8.8, 8.8)], 0.7), VBLACK)
        paint(img, stroke([(7.0, 8.7), (8.2, 10.8)], 0.7), VBLACK)
        paint(img, stroke([(7.0, 8.7), (5.8, 10.6)], 0.7), VBLACK)
    return downsample(img)


def sign_bicycle():
    img = canvas()
    warning_triangle(img)
    paint(img, ring(1.2, 0.7, 6.0, 10.0), VBLACK)
    paint(img, ring(1.2, 0.7, 10.5, 10.0), VBLACK)
    paint(img, stroke([(6.0, 10.0), (8.2, 8.0), (9.0, 10.0), (7.2, 10.0), (8.2, 8.0)], 0.6), VBLACK)
    paint(img, stroke([(8.1, 8.1), (8.6, 6.8)], 0.5), VBLACK)
    paint(img, stroke([(8.6, 6.8), (9.7, 6.8)], 0.5), VBLACK)
    return downsample(img)


def sign_snow():
    img = canvas()
    warning_triangle(img)
    for pts in [
        [(8.0, 5.4), (8.0, 11.2)],
        [(5.4, 6.6), (10.6, 10.0)],
        [(10.6, 6.6), (5.4, 10.0)],
    ]:
        paint(img, stroke(pts, 0.8), VBLACK)
    return downsample(img)


def sign_deer():
    img = canvas()
    warning_triangle(img)
    body = [(5.4, 8.8), (8.8, 7.4), (11.2, 8.0), (9.8, 9.5), (7.1, 10.1)]
    paint(img, polygon(body), VBLACK)
    paint(img, stroke([(8.6, 7.7), (9.8, 6.0)], 0.6), VBLACK)
    paint(img, stroke([(9.9, 6.0), (10.8, 5.2)], 0.5), VBLACK)
    paint(img, stroke([(9.9, 6.0), (10.6, 6.8)], 0.5), VBLACK)
    paint(img, stroke([(6.5, 9.7), (6.2, 11.4)], 0.5), VBLACK)
    paint(img, stroke([(8.0, 9.5), (7.8, 11.4)], 0.5), VBLACK)
    paint(img, stroke([(9.4, 9.2), (9.1, 11.0)], 0.5), VBLACK)
    return downsample(img)


def sign_end_all_limits():
    img = canvas()
    end_circle_base(img)
    end_stripes(img, [-3.0, -1.0, 1.0, 3.0])
    return downsample(img)


def sign_blue_right_turn(left=False):
    img = canvas()
    blue_disc(img)
    pts = [(8.0, 11.7), (8.0, 8.7), (8.4, 6.8), (10.8, 6.1)]
    if left:
        pts = [(16.0 - x, y) for x, y in pts]
    draw_arrow(img, pts, width=1.1, head_len=1.8, head_w=1.4)
    return downsample(img)


def sign_blue_ahead():
    img = canvas()
    blue_disc(img)
    draw_arrow(img, [(8.0, 11.7), (8.0, 5.2)], width=1.2, head_len=1.9, head_w=1.5)
    return downsample(img)


def sign_blue_straight_or_turn(left=False):
    img = canvas()
    blue_disc(img)
    draw_arrow(img, [(8.0, 11.8), (8.0, 5.6)], width=1.0, head_len=1.6, head_w=1.3)
    pts = [(8.0, 9.1), (8.6, 7.9), (10.8, 7.0)]
    if left:
        pts = [(16.0 - x, y) for x, y in pts]
    draw_arrow(img, pts, width=0.9, head_len=1.5, head_w=1.2)
    return downsample(img)


def sign_keep_side(left=False):
    img = canvas()
    blue_disc(img)
    pts = [(6.0, 4.8), (8.1, 7.5), (10.3, 11.2)]
    if left:
        pts = [(16.0 - x, y) for x, y in pts]
    draw_arrow(img, pts, width=1.1, head_len=1.7, head_w=1.4)
    return downsample(img)


def sign_roundabout():
    img = canvas()
    blue_disc(img)
    draw_arrow(img, [(8.3, 4.5), (11.3, 5.6), (11.0, 8.0)], width=0.85, head_len=1.2, head_w=1.0)
    draw_arrow(img, [(12.0, 8.7), (10.3, 11.5), (7.9, 10.8)], width=0.85, head_len=1.2, head_w=1.0)
    draw_arrow(img, [(6.0, 10.8), (4.8, 7.8), (6.8, 5.7)], width=0.85, head_len=1.2, head_w=1.0)
    return downsample(img)


def build_templates():
    T = [None] * 43
    T[0] = sign_speed("20")
    T[1] = sign_speed("30")
    T[2] = sign_speed("50")
    T[3] = sign_speed("60")
    T[4] = sign_speed("70")
    T[5] = sign_speed("80")
    T[6] = sign_end_speed_80()
    T[7] = sign_speed("100")
    T[8] = sign_speed("120")
    T[9] = sign_no_passing(False, False)
    T[10] = sign_no_passing(True, False)
    T[11] = sign_priority_next()
    T[12] = downsample((lambda img: (priority_road(img), img)[1])(canvas()))
    T[13] = downsample((lambda img: (yield_sign(img), img)[1])(canvas()))
    T[14] = downsample((lambda img: (stop_sign(img), img)[1])(canvas()))
    T[15] = downsample((lambda img: (circle_sign(img), img)[1])(canvas()))
    T[16] = downsample((lambda img: (circle_sign(img), draw_truck(img, 8.0, 8.8, 1.35, VBLACK), img)[2])(canvas()))
    T[17] = downsample((lambda img: (no_entry(img), img)[1])(canvas()))
    T[18] = sign_general_danger()
    T[19] = sign_bend(True)
    T[20] = sign_bend(False)
    T[21] = sign_double_bend()
    T[22] = sign_bumpy()
    T[23] = sign_slippery()
    T[24] = sign_narrows_right()
    T[25] = sign_roadworks()
    T[26] = sign_traffic_lights()
    T[27] = sign_pedestrians(False)
    T[28] = sign_pedestrians(True)
    T[29] = sign_bicycle()
    T[30] = sign_snow()
    T[31] = sign_deer()
    T[32] = sign_end_all_limits()
    T[33] = sign_blue_right_turn(False)
    T[34] = sign_blue_right_turn(True)
    T[35] = sign_blue_ahead()
    T[36] = sign_blue_straight_or_turn(False)
    T[37] = sign_blue_straight_or_turn(True)
    T[38] = sign_keep_side(False)
    T[39] = sign_keep_side(True)
    T[40] = sign_roundabout()
    T[41] = sign_no_passing(False, True)
    T[42] = sign_no_passing(True, True)
    return np.stack(T).astype(np.float32)


def channel_norm(x):
    mu = x.mean(axis=(1, 2), keepdims=True)
    var = ((x - mu) ** 2).mean(axis=(1, 2), keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5)


def make_weights(templates):
    yy, xx = np.mgrid[0:RES, 0:RES].astype(np.float32) + 0.5
    rr = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    center = np.clip(1.0 - np.maximum(rr - 5.8, 0.0) / 2.4, 0.0, 1.0).astype(np.float32)

    wr = []
    wn = []
    for t in templates:
        raw = (t - 0.5) * center[None, :, :]
        nr = channel_norm(t) * center[None, :, :]
        raw /= np.sqrt((raw * raw).sum()) + 1e-6
        nr /= np.sqrt((nr * nr).sum()) + 1e-6
        wr.append((1.4 * raw).reshape(-1))
        wn.append((1.0 * nr).reshape(-1))
    return np.stack(wr).astype(np.float32), np.zeros(43, np.float32), np.stack(wn).astype(np.float32), np.zeros(43, np.float32)


def write_answer(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = build_templates()
    raw_W, raw_b, norm_W, norm_b = make_weights(templates)

    net = {
        "layers": [
            {"name": "pool", "op": "avgpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]},
            {"name": "pnorm", "op": "norm", "inputs": ["pool"]},
            {"name": "raw", "op": "dense", "inputs": ["pool"], "out": 43},
            {"name": "norm", "op": "dense", "inputs": ["pnorm"], "out": 43},
            {"name": "scores", "op": "add", "inputs": ["raw", "norm"]},
        ]
    }

    (out_dir / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(
        out_dir / "weights.npz",
        **{
            "raw.W": raw_W,
            "raw.b": raw_b,
            "norm.W": norm_W,
            "norm.b": norm_b,
        },
    )


if __name__ == "__main__":
    write_answer(sys.argv[1])
