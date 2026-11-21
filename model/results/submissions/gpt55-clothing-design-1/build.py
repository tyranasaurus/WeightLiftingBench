import json, sys
from pathlib import Path
import numpy as np

H = W = 28
NPIX = H * W
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)


def soft_inside(d, margin=1.0):
    return np.clip(d / margin + 0.5, 0.0, 1.0)


def trap(cx, y0, y1, w0, w1):
    y0 = float(y0); y1 = float(y1)
    t = np.clip((Y - y0) / max(1e-6, (y1 - y0)), 0.0, 1.0)
    half = 0.5 * (w0 + (w1 - w0) * t)
    d = np.minimum.reduce([Y - y0, y1 - Y, half - np.abs(X - cx)])
    return soft_inside(d, 1.0)


def rect(cx, cy, w, h):
    d = np.minimum(0.5 * w - np.abs(X - cx), 0.5 * h - np.abs(Y - cy))
    return soft_inside(d, 1.0)


def ellipse(cx, cy, rx, ry):
    v = ((X - cx) / max(rx, 1e-6)) ** 2 + ((Y - cy) / max(ry, 1e-6)) ** 2
    return np.clip((1.0 - v) * 2.0, 0.0, 1.0)


def capsule(x0, y0, x1, y1, r):
    vx = x1 - x0; vy = y1 - y0
    den = vx * vx + vy * vy + 1e-6
    t = np.clip(((X - x0) * vx + (Y - y0) * vy) / den, 0.0, 1.0)
    px = x0 + t * vx; py = y0 + t * vy
    dist = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
    return np.clip((r + 0.55 - dist) / 1.1, 0.0, 1.0)


def poly(points):
    pts = list(points)
    inside = np.zeros((H, W), dtype=bool)
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        cond = ((y0 > Y) != (y1 > Y)) & (X < (x1 - x0) * (Y - y0) / (y1 - y0 + 1e-6) + x0)
        inside ^= cond
    return inside.astype(np.float32)


def blur(m, rounds=1):
    m = np.asarray(m, dtype=np.float32)
    for _ in range(rounds):
        p = np.pad(m, 1)
        m = (4 * p[1:-1, 1:-1] + 2 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) +
             (p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:])) / 16.0
    return np.clip(m, 0.0, 1.0)


def add_template(lst, m):
    m = blur(np.clip(m, 0.0, 1.0), 1)
    if float(m.sum()) > 8.0:
        lst.append(m.astype(np.float32))


def make_top(short=True, tank=False, cx=13.5, y0=5, yb=22, shoulder=16, waist=11, sleeve_len=4.0, sleeve_drop=4.0):
    m = np.maximum(trap(cx, y0 + 3, yb, shoulder * 0.62, waist), trap(cx, y0, y0 + 7, shoulder, shoulder * 0.65))
    if tank:
        m = np.maximum(m, capsule(cx - shoulder * 0.22, y0 + 1, cx - waist * 0.33, yb - 1, 1.0))
        m = np.maximum(m, capsule(cx + shoulder * 0.22, y0 + 1, cx + waist * 0.33, yb - 1, 1.0))
    elif short:
        m = np.maximum(m, capsule(cx - shoulder * 0.48, y0 + 2, cx - shoulder * 0.48 - sleeve_len, y0 + 2 + sleeve_drop, 2.2))
        m = np.maximum(m, capsule(cx + shoulder * 0.48, y0 + 2, cx + shoulder * 0.48 + sleeve_len, y0 + 2 + sleeve_drop, 2.2))
    else:
        m = np.maximum(m, capsule(cx - shoulder * 0.45, y0 + 2, cx - shoulder * 0.62, yb - 1, 2.1))
        m = np.maximum(m, capsule(cx + shoulder * 0.45, y0 + 2, cx + shoulder * 0.62, yb - 1, 2.1))
    neck = ellipse(cx, y0 + 1.5, 2.2, 1.3)
    m *= (1.0 - 0.85 * neck)
    return m


def make_trouser(cx=13.5, y0=4, yb=27, waist=15, legw=4.5, gap=2.5, flare=0.0):
    m = trap(cx, y0, y0 + 7, waist, waist * 0.75)
    crotch = y0 + 8
    left_top_outer = cx - gap * 0.5 - legw
    left_top_inner = cx - gap * 0.5
    right_top_inner = cx + gap * 0.5
    right_top_outer = cx + gap * 0.5 + legw
    left_bottom_outer = left_top_outer - flare
    left_bottom_inner = left_top_inner - 0.15 * flare
    right_bottom_inner = right_top_inner + 0.15 * flare
    right_bottom_outer = right_top_outer + flare
    ml = poly([(left_top_outer, crotch), (left_top_inner, crotch), (left_bottom_inner, yb), (left_bottom_outer, yb)])
    mr = poly([(right_top_inner, crotch), (right_top_outer, crotch), (right_bottom_outer, yb), (right_bottom_inner, yb)])
    m = np.maximum(m, np.maximum(ml, mr))
    m = np.maximum(m, capsule(cx - gap * 0.5 - legw * 0.5, crotch, cx - gap * 0.5 - legw * 0.5 - 0.4 * flare, yb - 1, legw * 0.48))
    m = np.maximum(m, capsule(cx + gap * 0.5 + legw * 0.5, crotch, cx + gap * 0.5 + legw * 0.5 + 0.4 * flare, yb - 1, legw * 0.48))
    return m


def make_pullover(cx=13.5, y0=5, yb=25, shoulder=17, bottom=13, sleeve_spread=2.5):
    m = trap(cx, y0 + 2, yb, shoulder * 0.65, bottom)
    m = np.maximum(m, trap(cx, y0, y0 + 7, shoulder, shoulder * 0.68))
    m = np.maximum(m, capsule(cx - shoulder * 0.48, y0 + 2, cx - shoulder * 0.53 - sleeve_spread, yb - 2, 2.4))
    m = np.maximum(m, capsule(cx + shoulder * 0.48, y0 + 2, cx + shoulder * 0.53 + sleeve_spread, yb - 2, 2.4))
    m = np.maximum(m, rect(cx - shoulder * 0.54 - sleeve_spread, yb - 2, 4.0, 2.4))
    m = np.maximum(m, rect(cx + shoulder * 0.54 + sleeve_spread, yb - 2, 4.0, 2.4))
    m *= (1.0 - 0.8 * ellipse(cx, y0 + 1.5, 2.3, 1.2))
    return m


def make_dress(cx=13.5, y0=5, yb=27, shoulder=12, waist=9, hem=21, sleeves=0):
    m = trap(cx, y0, y0 + 8, shoulder, waist)
    m = np.maximum(m, trap(cx, y0 + 7, yb, waist, hem))
    if sleeves == 1:
        m = np.maximum(m, capsule(cx - shoulder * 0.48, y0 + 2, cx - shoulder * 0.7, y0 + 7, 1.8))
        m = np.maximum(m, capsule(cx + shoulder * 0.48, y0 + 2, cx + shoulder * 0.7, y0 + 7, 1.8))
    elif sleeves == 2:
        m = np.maximum(m, capsule(cx - shoulder * 0.45, y0 + 2, cx - shoulder * 0.62, y0 + 13, 1.8))
        m = np.maximum(m, capsule(cx + shoulder * 0.45, y0 + 2, cx + shoulder * 0.62, y0 + 13, 1.8))
    m *= (1.0 - 0.85 * ellipse(cx, y0 + 1.2, 2.0, 1.2))
    return m


def make_coat(cx=13.5, y0=4, yb=27, shoulder=18, hem=18, gap=2.4, spread=2.0):
    outer = trap(cx, y0, yb, shoulder, hem)
    sleeves = np.maximum(capsule(cx - shoulder * 0.48, y0 + 2, cx - shoulder * 0.52 - spread, yb - 2, 2.5),
                         capsule(cx + shoulder * 0.48, y0 + 2, cx + shoulder * 0.52 + spread, yb - 2, 2.5))
    m = np.maximum(outer, sleeves)
    if gap > 0.1:
        openline = rect(cx, (y0 + yb) / 2 + 2, gap, yb - y0 - 8)
        m *= (1.0 - 0.72 * openline)
    collar = np.maximum(capsule(cx - 2.2, y0 + 2, cx - 0.5, y0 + 5, 1.2), capsule(cx + 2.2, y0 + 2, cx + 0.5, y0 + 5, 1.2))
    m = np.maximum(m, collar)
    return m


def make_shirt(cx=13.5, y0=5, yb=23, shoulder=17, bottom=13, sleeve=0, open_front=False):
    m = np.maximum(trap(cx, y0 + 3, yb, shoulder * 0.62, bottom), trap(cx, y0, y0 + 7, shoulder, shoulder * 0.7))
    if sleeve == 0:
        m = np.maximum(m, capsule(cx - shoulder * 0.48, y0 + 2, cx - shoulder * 0.72, y0 + 8, 2.1))
        m = np.maximum(m, capsule(cx + shoulder * 0.48, y0 + 2, cx + shoulder * 0.72, y0 + 8, 2.1))
    elif sleeve == 1:
        m = np.maximum(m, capsule(cx - shoulder * 0.47, y0 + 2, cx - shoulder * 0.55, yb - 2, 2.0))
        m = np.maximum(m, capsule(cx + shoulder * 0.47, y0 + 2, cx + shoulder * 0.55, yb - 2, 2.0))
    else:
        m = np.maximum(m, capsule(cx - shoulder * 0.48, y0 + 2, cx - shoulder * 0.72, y0 + 12, 2.0))
        m = np.maximum(m, capsule(cx + shoulder * 0.48, y0 + 2, cx + shoulder * 0.72, y0 + 12, 2.0))
    # collar leaves a V-shaped dark opening and two small lapels
    neck = np.maximum(ellipse(cx, y0 + 1.3, 2.0, 1.4), poly([(cx - 2.5, y0 + 2), (cx, y0 + 7), (cx + 2.5, y0 + 2)]))
    m *= (1.0 - 0.75 * neck)
    lapels = np.maximum(capsule(cx - 2.8, y0 + 2, cx - 0.8, y0 + 5.5, 0.9), capsule(cx + 2.8, y0 + 2, cx + 0.8, y0 + 5.5, 0.9))
    m = np.maximum(m, lapels)
    if open_front:
        m *= (1.0 - 0.55 * rect(cx, (y0 + yb) / 2 + 2, 1.5, yb - y0 - 8))
    return m


def make_sandal(cx=13.5, ybase=23, length=23, orient=1, heel=False, high=False):
    x0 = cx - length / 2; x1 = cx + length / 2
    if orient < 0:
        x0, x1 = x1, x0
    m = capsule(x0, ybase - 0.5, x1, ybase - 0.2, 1.0)
    m = np.maximum(m, capsule(x0 + orient * length * 0.15, ybase - 2.0, x1 - orient * length * 0.10, ybase - 2.6, 1.0))
    m = np.maximum(m, capsule(x0 + orient * length * 0.25, ybase - 6, x0 + orient * length * 0.55, ybase - 2.0, 0.8))
    m = np.maximum(m, capsule(x0 + orient * length * 0.50, ybase - 6, x0 + orient * length * 0.78, ybase - 2.0, 0.8))
    m = np.maximum(m, capsule(x0 + orient * length * 0.35, ybase - 4.5, x0 + orient * length * 0.72, ybase - 4.5, 0.75))
    if heel:
        hx = x0 + orient * length * 0.05
        m = np.maximum(m, capsule(hx, ybase - 0.5, hx - orient * (2.5 if high else 0.8), ybase - (10.0 if high else 6.0), 0.65))
        m = np.maximum(m, capsule(hx - orient * (2.5 if high else 0.8), ybase - (10.0 if high else 6.0), x0 + orient * length * 0.35, ybase - 5.5, 0.7))
    return m


def make_sneaker(cx=13.5, ybase=24, length=24, height=8, orient=1):
    xheel = cx - orient * length / 2
    xtoe = cx + orient * length / 2
    pts = [(xheel, ybase - 2), (xheel + orient * length * 0.10, ybase - height * 0.65),
           (xheel + orient * length * 0.34, ybase - height), (xheel + orient * length * 0.68, ybase - height * 0.75),
           (xtoe - orient * 1.0, ybase - height * 0.40), (xtoe + orient * 0.7, ybase - 2.0),
           (xtoe - orient * 1.5, ybase), (xheel - orient * 0.7, ybase)]
    m = poly(pts)
    m = np.maximum(m, capsule(xheel, ybase, xtoe, ybase, 1.15))
    m *= (1.0 - 0.45 * ellipse(xheel + orient * length * 0.36, ybase - height * 0.70, length * 0.16, height * 0.20))
    return m


def make_bag(cx=13.5, ytop=11, yb=26, width=20, taper=0, handle=1, round_body=False):
    if round_body:
        body = ellipse(cx, (ytop + yb) / 2, width / 2, (yb - ytop) / 2)
        body = np.maximum(body, rect(cx, (ytop + yb) / 2 + 1, width * 0.78, (yb - ytop) * 0.70))
    else:
        body = trap(cx, ytop, yb, width - taper, width + taper)
    m = body
    if handle:
        rx = width * (0.30 if handle == 1 else 0.38)
        ry = 5.0 if handle == 1 else 6.5
        cy = ytop + 0.8
        ring = ellipse(cx, cy, rx, ry) * (1.0 - ellipse(cx, cy + 0.4, max(0.5, rx - 1.8), max(0.5, ry - 1.8)))
        ring *= (Y < ytop + 2.5)
        straps = np.maximum(capsule(cx - rx * 0.65, ytop + 1, cx - width * 0.30, ytop + 5, 0.9),
                            capsule(cx + rx * 0.65, ytop + 1, cx + width * 0.30, ytop + 5, 0.9))
        m = np.maximum(m, np.maximum(ring, straps))
    return m


def make_boot(cx=13.5, ybase=25, length=21, shaft_h=14, shaft_w=8, orient=1, heel=0):
    xheel = cx - orient * length / 2
    xtoe = cx + orient * length / 2
    shaft_cx = xheel + orient * length * 0.23
    shaft = trap(shaft_cx, ybase - shaft_h - 2, ybase - 5, shaft_w * 0.95, shaft_w)
    foot_pts = [(xheel - orient * 0.5, ybase - 5), (xheel + orient * length * 0.38, ybase - 6),
                (xtoe - orient * 1.0, ybase - 3.2), (xtoe + orient * 0.6, ybase - 1.4),
                (xtoe - orient * 1.0, ybase), (xheel - orient * 0.8, ybase)]
    foot = poly(foot_pts)
    m = np.maximum(shaft, foot)
    m = np.maximum(m, capsule(xheel, ybase, xtoe, ybase, 1.1))
    if heel:
        hx = xheel + orient * 1.0
        m = np.maximum(m, capsule(hx, ybase - 0.4, hx - orient * 1.3, ybase - 4.5, 0.85))
    return m


def class_templates():
    T = [[] for _ in range(10)]
    # 0: T-shirt/top, short sleeves and tank-like tops
    for sh in [-1.0, 0.0, 1.0]:
        for yb in [20.5, 22.0, 23.5]:
            for shoulder in [13.5, 15.5, 17.5]:
                for waist in [9.5, 11.5, 13.5]:
                    add_template(T[0], make_top(True, False, 13.5 + sh, 5.0, yb, shoulder, waist, 3.5, 4.0))
        for shoulder in [11.5, 13.5, 15.5]:
            for waist in [9.0, 11.0, 13.0]:
                add_template(T[0], make_top(False, True, 13.5 + sh, 5.2, 22.5, shoulder, waist, 0, 0))
    # 1: trousers
    for sh in [-1.0, 0.0, 1.0]:
        for waist in [13.5, 15.5, 17.5]:
            for legw in [3.4, 4.2, 5.0]:
                for gap in [1.8, 2.8, 3.8]:
                    for flare in [-0.6, 0.2, 1.0]:
                        add_template(T[1], make_trouser(13.5 + sh, 4.2, 27.0, waist, legw, gap, flare))
    # 2: pullover
    for sh in [-1.0, 0.0, 1.0]:
        for yb in [23.5, 25.0, 26.0]:
            for shoulder in [15.0, 17.0, 19.0]:
                for bottom in [11.5, 13.5, 15.5]:
                    for spread in [0.5, 2.0, 3.5]:
                        add_template(T[2], make_pullover(13.5 + sh, 5.0, yb, shoulder, bottom, spread))
    # 3: dresses
    for sh in [-1.0, 0.0, 1.0]:
        for yb in [24.5, 26.0, 27.0]:
            for shoulder in [9.5, 11.5, 13.5]:
                for waist in [7.5, 9.5, 11.5]:
                    for hem in [16.0, 19.0, 22.0, 24.0]:
                        for sl in [0, 1]:
                            add_template(T[3], make_dress(13.5 + sh, 5.2, yb, shoulder, waist, hem, sl))
    # 4: coats
    for sh in [-1.0, 0.0, 1.0]:
        for shoulder in [15.5, 17.5, 19.5]:
            for hem in [14.5, 17.5, 20.5]:
                for gap in [0.0, 1.7, 2.8, 3.8]:
                    for spread in [0.5, 2.0, 3.5]:
                        add_template(T[4], make_coat(13.5 + sh, 4.5, 27.0, shoulder, hem, gap, spread))
    # 5: sandals
    for sh in [-1.5, 0.0, 1.5]:
        for orient in [-1, 1]:
            for yb in [22.0, 23.5, 25.0]:
                for length in [19.0, 22.0, 25.0]:
                    for heel in [False, True]:
                        add_template(T[5], make_sandal(13.5 + sh, yb, length, orient, heel, False))
                    add_template(T[5], make_sandal(13.5 + sh, yb, length, orient, True, True))
    # 6: shirts
    for sh in [-1.0, 0.0, 1.0]:
        for yb in [21.5, 23.0, 24.5]:
            for shoulder in [14.5, 16.5, 18.5]:
                for bottom in [11.0, 13.0, 15.0]:
                    for sleeve in [0, 1, 2]:
                        add_template(T[6], make_shirt(13.5 + sh, 5.0, yb, shoulder, bottom, sleeve, sleeve == 1))
    # 7: sneakers
    for sh in [-1.5, 0.0, 1.5]:
        for orient in [-1, 1]:
            for yb in [23.0, 24.5, 26.0]:
                for length in [20.0, 23.0, 26.0]:
                    for height in [6.5, 8.5, 10.5]:
                        add_template(T[7], make_sneaker(13.5 + sh, yb, length, height, orient))
    # 8: bags
    for sh in [-1.0, 0.0, 1.0]:
        for ytop in [9.5, 11.5, 13.0]:
            for width in [16.0, 19.0, 22.0, 24.5]:
                for taper in [-2.0, 0.0, 2.0]:
                    for handle in [0, 1, 2]:
                        add_template(T[8], make_bag(13.5 + sh, ytop, 26.0, width, taper, handle, False))
                add_template(T[8], make_bag(13.5 + sh, ytop, 25.5, width, 0.0, 1, True))
    # 9: ankle boots
    for sh in [-1.5, 0.0, 1.5]:
        for orient in [-1, 1]:
            for yb in [24.0, 25.5, 26.5]:
                for length in [18.0, 21.0, 24.0]:
                    for shaft_h in [10.0, 13.0, 16.0, 19.0]:
                        for shaft_w in [6.5, 8.5, 10.5]:
                            add_template(T[9], make_boot(13.5 + sh, yb, length, shaft_h, shaft_w, orient, heel=1))
    return T


def template_to_weight(m, beta=1.0, scale=1.0):
    m = np.clip(m.astype(np.float32), 0.0, 1.0)
    s = float(m.sum())
    if s < 1.0:
        s = 1.0
    outside = float(NPIX) - s
    if outside < 1.0:
        outside = 1.0
    # Average brightness in the drawn object minus average brightness elsewhere.
    # This is a hand-made similarity score, not a statistic from the data.
    w = m / s - beta * (1.0 - m) / outside
    return (scale * w.reshape(-1)).astype(np.float32)


def norm_mask(m):
    m = np.clip(np.asarray(m, dtype=np.float32), 0.0, 1.0)
    s = float(m.sum())
    return m / (s if s > 1e-6 else 1.0)


def edge_filters():
    fx = np.array([[-0.25, 0.0, 0.25], [-0.5, 0.0, 0.5], [-0.25, 0.0, 0.25]], dtype=np.float32)
    fy = fx.T
    fd = np.array([[-0.5, -0.25, 0.0], [-0.25, 0.0, 0.25], [0.0, 0.25, 0.5]], dtype=np.float32)
    fa = np.array([[0.0, 0.25, 0.5], [-0.25, 0.0, 0.25], [-0.5, -0.25, 0.0]], dtype=np.float32)
    lap = np.array([[0.0, -0.7, 0.0], [-0.7, 2.8, -0.7], [0.0, -0.7, 0.0]], dtype=np.float32)
    return np.stack([fx, -fx, fy, -fy, fd, -fd, lap, -lap], axis=0).astype(np.float32)


def conv2_same(m, k):
    p = np.pad(m.astype(np.float32), 1)
    out = np.zeros_like(m, dtype=np.float32)
    for yy in range(3):
        for xx in range(3):
            out += float(k[yy, xx]) * p[yy:yy + H, xx:xx + 28]
    return out


def edge_template_to_weight(m, scale=3.0):
    fs = edge_filters()
    maps = []
    for k in fs:
        e = np.maximum(conv2_same(m, k), 0.0)
        maps.append(e)
    e = np.stack(maps, axis=0)
    s = float(e.sum())
    if s < 1e-6:
        s = 1.0
    return (scale * e.reshape(-1) / s).astype(np.float32)


def edge_bank_templates():
    E = [[] for _ in range(10)]
    params = [
        (0, [make_top(True, False, 13.5, 5, 22, 16, 11.5, 4, 4), make_top(False, True, 13.5, 5, 22, 13, 10, 0, 0)]),
        (1, [make_trouser(13.5, 4, 27, 15, 4.2, 3, 0), make_trouser(13.5, 4, 27, 17, 5, 2, 1)]),
        (2, [make_pullover(13.5, 5, 25, 17, 13.5, 2), make_pullover(13.5, 5, 25, 19, 15, 3.5)]),
        (3, [make_dress(13.5, 5, 27, 11.5, 9, 22, 0), make_dress(13.5, 5, 26, 13, 10, 18, 1)]),
        (4, [make_coat(13.5, 4.5, 27, 18, 18, 2.5, 2), make_coat(13.5, 4.5, 27, 19.5, 20, 0, 3)]),
        (5, [make_sandal(13.5, 24, 23, 1, True, False), make_sandal(13.5, 24, 23, -1, True, False)]),
        (6, [make_shirt(13.5, 5, 23, 17, 13, 0, False), make_shirt(13.5, 5, 24, 18, 15, 1, True)]),
        (7, [make_sneaker(13.5, 24.5, 24, 8.5, 1), make_sneaker(13.5, 24.5, 24, 8.5, -1)]),
        (8, [make_bag(13.5, 11.5, 26, 21, 0, 1, False), make_bag(13.5, 12.5, 25.5, 19, 0, 0, True)]),
        (9, [make_boot(13.5, 25.5, 21, 14, 8, 1, 1), make_boot(13.5, 25.5, 21, 14, 8, -1, 1)]),
    ]
    for k, arr in params:
        for dx in [-1.0, 0.0, 1.0]:
            for m in arr:
                # simple horizontal roll approximates the small translations seen after centering
                mm = np.roll(m, int(dx), axis=1)
                E[k].append(blur(mm, 1))
    return E


def class_linear_weights(scale=9.0):
    # A second, deliberately coarse branch: class-specific region evidence.
    # Terms are geometric priors such as "trousers have two bright lower legs
    # with a dark gap" and "shoes are bright only near the bottom".
    terms = [[] for _ in range(10)]
    upper_all = rect(13.5, 9.0, 27.0, 14.0)
    very_top = rect(13.5, 3.5, 25.0, 7.0)
    bottom_all = rect(13.5, 24.5, 27.0, 7.0)
    shoe_band = rect(13.5, 22.5, 26.0, 9.0)
    low_center_gap = rect(13.5, 20.0, 3.2, 14.0)
    low_side_legs = np.maximum(rect(9.2, 20.5, 5.5, 14.0), rect(17.8, 20.5, 5.5, 14.0))
    # 0: short upper garment/top
    terms[0] += [(1.3, make_top(True, False, 13.5, 5.0, 22.0, 16.0, 11.5, 3.8, 4.0)),
                 (0.7, make_top(False, True, 13.5, 5.0, 22.0, 13.0, 10.5, 0, 0)),
                 (-0.9, rect(13.5, 25.0, 24.0, 5.0)),
                 (-0.7, low_side_legs),
                 (-0.4, rect(13.5, 14.0, 5.0, 16.0))]
    # 1: trouser/pants
    terms[1] += [(1.5, make_trouser(13.5, 4.2, 27.0, 15.5, 4.2, 3.0, 0.3)),
                 (0.9, low_side_legs),
                 (-1.2, low_center_gap),
                 (-0.7, np.maximum(rect(4.0, 12.0, 8.0, 12.0), rect(23.0, 12.0, 8.0, 12.0))),
                 (-0.5, rect(13.5, 7.0, 22.0, 7.0))]
    # 2: pullover, a closed upper garment with long sleeves
    terms[2] += [(1.35, make_pullover(13.5, 5.0, 24.8, 17.5, 13.5, 2.0)),
                 (0.8, np.maximum(capsule(5.5, 8.0, 4.2, 23.0, 2.0), capsule(21.5, 8.0, 22.8, 23.0, 2.0))),
                 (-0.6, low_center_gap),
                 (-0.7, rect(13.5, 26.5, 24.0, 3.0)),
                 (-0.4, very_top)]
    # 3: dress, narrow top and broad continuous skirt
    terms[3] += [(1.6, make_dress(13.5, 5.2, 27.0, 11.5, 9.0, 22.0, 0)),
                 (0.7, trap(13.5, 15.0, 27.0, 10.0, 23.0)),
                 (-0.9, low_center_gap),
                 (-0.6, np.maximum(rect(4.0, 12.0, 8.0, 12.0), rect(23.0, 12.0, 8.0, 12.0))),
                 (-0.5, rect(13.5, 24.5, 7.0, 6.0))]
    # 4: coat, long upper garment with sleeves and length to the hem
    terms[4] += [(1.45, make_coat(13.5, 4.5, 27.0, 18.0, 18.0, 2.5, 2.0)),
                 (0.8, np.maximum(capsule(5.0, 7.5, 3.8, 25.0, 2.4), capsule(22.0, 7.5, 23.2, 25.0, 2.4))),
                 (0.45, rect(13.5, 24.0, 17.0, 6.0)),
                 (-0.6, low_center_gap),
                 (-0.55, rect(13.5, 10.0, 6.0, 9.0))]
    # 5: sandal, bottom sole plus open straps, little filled upper
    sandal_shape = make_sandal(13.5, 24.0, 23.0, 1, True, False)
    sandal_shape = np.maximum(sandal_shape, make_sandal(13.5, 24.0, 23.0, -1, True, False))
    terms[5] += [(1.6, sandal_shape),
                 (0.7, capsule(2.5, 24.0, 25.5, 24.0, 1.0)),
                 (-1.15, upper_all),
                 (-0.75, rect(13.5, 19.0, 22.0, 6.0)),
                 (-0.5, rect(13.5, 12.0, 26.0, 12.0))]
    # 6: shirt, collared/structured upper garment; kept conservative because it overlaps many tops
    terms[6] += [(0.95, make_shirt(13.5, 5.0, 23.0, 17.0, 13.0, 0, False)),
                 (0.45, np.maximum(capsule(10.8, 7.0, 12.7, 11.5, 0.8), capsule(16.2, 7.0, 14.3, 11.5, 0.8))),
                 (-0.9, bottom_all),
                 (-0.45, low_center_gap),
                 (-0.3, rect(13.5, 17.0, 8.0, 13.0))]
    # 7: sneaker, solid low shoe body and sole
    sneaker_shape = np.maximum(make_sneaker(13.5, 24.5, 24.0, 8.5, 1), make_sneaker(13.5, 24.5, 24.0, 8.5, -1))
    terms[7] += [(1.8, sneaker_shape),
                 (0.8, capsule(2.0, 25.0, 26.0, 25.0, 1.2)),
                 (-1.3, upper_all),
                 (-0.9, rect(7.5, 13.5, 8.5, 14.0)),
                 (-0.45, rect(13.5, 19.0, 5.0, 8.0))]
    # 8: bag, compact body with possible handle; not shoulders or legs
    terms[8] += [(1.55, make_bag(13.5, 11.5, 26.0, 21.0, 0.0, 1, False)),
                 (0.7, make_bag(13.5, 12.5, 25.5, 19.0, 0.0, 0, True)),
                 (-0.9, very_top),
                 (-0.8, low_center_gap),
                 (-0.65, np.maximum(capsule(5.0, 7.5, 4.0, 24.5, 2.0), capsule(22.0, 7.5, 23.0, 24.5, 2.0))),
                 (-0.55, rect(13.5, 24.8, 24.0, 3.5))]
    # 9: ankle boot, low shoe plus a tall shaft
    boot_shape = np.maximum(make_boot(13.5, 25.5, 21.0, 14.0, 8.0, 1, 1), make_boot(13.5, 25.5, 21.0, 14.0, 8.0, -1, 1))
    terms[9] += [(1.75, boot_shape),
                 (0.75, shoe_band),
                 (0.65, np.maximum(rect(8.8, 14.5, 7.0, 15.0), rect(18.2, 14.5, 7.0, 15.0))),
                 (-1.0, very_top),
                 (-0.7, rect(13.5, 10.0, 10.0, 10.0)),
                 (-0.45, rect(13.5, 20.0, 5.0, 9.0))]
    Wlist = []
    blist = []
    for k in range(10):
        ww = np.zeros((H, 28), dtype=np.float32)
        bias = 0.0
        for coef, m in terms[k]:
            ww += float(coef) * norm_mask(m)
        Wlist.append((scale * ww.reshape(-1)).astype(np.float32))
        blist.append(bias)
    return np.stack(Wlist, axis=0), np.asarray(blist, dtype=np.float32)


def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = class_templates()
    V = max(len(t) for t in T)
    # keep a common block size for pooling; duplicate early variants for classes with fewer masks
    betas = [1.03, 1.08, 1.00, 1.00, 1.00, 1.35, 1.00, 1.18, 1.03, 1.15]
    weights = []
    bias = []
    for k in range(10):
        arr = T[k]
        for j in range(V):
            m = arr[j % len(arr)]
            weights.append(template_to_weight(m, betas[k], 8.0))
            # Tiny fixed preferences chosen from class geometry, not data: sandals sparse, bags compact, boots tall.
            bias.append(0.0)
    Wtmpl = np.stack(weights, axis=0).astype(np.float32)
    btmpl = np.asarray(bias, dtype=np.float32)
    E = edge_bank_templates()
    EV = max(len(e) for e in E)
    edge_weights = []
    edge_bias = []
    for k in range(10):
        arr = E[k]
        for j in range(EV):
            edge_weights.append(edge_template_to_weight(arr[j % len(arr)], 4.0))
            edge_bias.append(0.0)
    Wedge_tmpl = np.stack(edge_weights, axis=0).astype(np.float32)
    bedge_tmpl = np.asarray(edge_bias, dtype=np.float32)
    Wlin, blin = class_linear_weights(scale=3.0)
    # Class score calibration is intentionally coarse and hand-set.
    scales = np.array([1.05, 0.98, 1.05, 0.90, 1.05, 1.06, 1.05, 1.10, 0.82, 0.96], dtype=np.float32)
    cbias = np.array([0.15, 0.05, 0.25, -0.10, 0.20, 0.25, 0.35, 0.35, -0.30, 0.00], dtype=np.float32)
    Wtmpl_out = np.diag(scales).astype(np.float32)
    btmpl_out = cbias.astype(np.float32)

    # A compact hand-written feature bank.  Each feature is either a regional
    # brightness integral or a contrast between expected bright and dark areas.
    feats = []
    def addf(mask, bias=0.0, gain=1.0):
        feats.append((gain * norm_mask(mask).reshape(-1), float(bias)))
    def addc(pos, neg, bias=0.0, gain=1.0):
        feats.append((gain * (norm_mask(pos) - norm_mask(neg)).reshape(-1), float(bias)))
    bands = [rect(13.5, 2.5 + 5*i, 28, 5) for i in range(6)]
    cols = [rect(2.5 + 5*i, 13.5, 5, 28) for i in range(6)]
    left = rect(6.5, 13.5, 13, 28); right = rect(20.5, 13.5, 13, 28); mid = rect(13.5, 13.5, 8, 28)
    top = rect(13.5, 7.0, 28, 14); bottom = rect(13.5, 21.0, 28, 14)
    upper = rect(13.5, 10.0, 28, 18); lower = rect(13.5, 22.0, 28, 12)
    center_low = rect(13.5, 21.0, 4, 14)
    side_low = np.maximum(rect(8.5, 21.0, 6, 14), rect(18.5, 21.0, 6, 14))
    side_upper = np.maximum(rect(4.0, 13.5, 8, 18), rect(23.0, 13.5, 8, 18))
    wide_low = rect(13.5, 24.0, 24, 8)
    shoe_low = rect(13.5, 24.0, 28, 8)
    boot_shaft_lr = np.maximum(rect(8.0, 14.0, 8, 14), rect(19.0, 14.0, 8, 14))
    handle = make_bag(13.5, 9.5, 24.0, 18.0, 0.0, 1, False) * (Y < 13)
    masks_for_feats = [top, bottom, upper, lower, center_low, side_low, side_upper, wide_low, shoe_low,
                       boot_shaft_lr, handle, make_top(True, False), make_trouser(), make_pullover(),
                       make_dress(), make_coat(), np.maximum(make_sandal(13.5,24,23,1,True,False), make_sandal(13.5,24,23,-1,True,False)),
                       np.maximum(make_sneaker(13.5,24.5,24,8.5,1), make_sneaker(13.5,24.5,24,8.5,-1)),
                       make_bag(), np.maximum(make_boot(13.5,25.5,21,14,8,1,1), make_boot(13.5,25.5,21,14,8,-1,1))]
    for m in bands + cols + masks_for_feats:
        addf(m)
    addc(lower, upper); addc(top, bottom); addc(side_low, center_low); addc(center_low, side_low)
    addc(side_upper, mid); addc(wide_low, center_low); addc(shoe_low, upper); addc(boot_shaft_lr, shoe_low)
    F = len(feats)
    Wfeat = np.stack([f[0] for f in feats]).astype(np.float32)
    bfeat = np.array([f[1] for f in feats], dtype=np.float32)
    Wfeat_out = np.zeros((10, F), dtype=np.float32)
    # feature indices after six horizontal and six vertical bands
    TOP,BOT,UP,LOW,CGAP,LEGS,USIDE,WLOW,SHOE,SHAFT,HANDLE,FTOP,FPANT,FPULL,FDRESS,FCOAT,FSAND,FSNEAK,FBAG,FBOOT = range(12,32)
    LOWER_UPPER, TOP_BOTTOM, LEGS_GAP, GAP_LEGS, USIDE_MID, WLOW_GAP, SHOE_UPPER, SHAFT_SHOE = range(32,40)
    def setw(k, pairs):
        for i,v in pairs:
            Wfeat_out[k,i] = v
    setw(0, [(FTOP,1.4),(TOP,0.2),(BOT,-0.8),(LEGS_GAP,-0.6),(USIDE,0.2)])
    setw(1, [(FPANT,1.7),(LEGS,0.8),(LEGS_GAP,1.0),(CGAP,-0.8),(USIDE,-0.5),(TOP,-0.4)])
    setw(2, [(FPULL,1.4),(USIDE,0.6),(FCOAT,0.15),(BOT,-0.3),(CGAP,-0.4),(TOP,0.1)])
    setw(3, [(FDRESS,1.7),(WLOW,0.7),(WLOW_GAP,0.8),(USIDE,-0.6),(LEGS_GAP,-0.6),(FTOP,0.2)])
    setw(4, [(FCOAT,1.6),(USIDE,0.8),(WLOW,0.4),(CGAP,-0.35),(FTOP,-0.25),(FDRESS,0.2)])
    setw(5, [(FSAND,1.8),(SHOE,0.7),(SHOE_UPPER,1.2),(UP,-1.0),(SHAFT,-0.5),(FSNEAK,-0.2)])
    setw(6, [(FTOP,0.5),(FPULL,0.25),(FCOAT,0.25),(FDRESS,-0.2),(BOT,-0.6),(LEGS_GAP,-0.45)])
    setw(7, [(FSNEAK,1.9),(SHOE,0.8),(SHOE_UPPER,1.1),(UP,-1.0),(SHAFT,-0.5),(FSAND,-0.15)])
    setw(8, [(FBAG,1.2),(HANDLE,0.6),(WLOW,0.15),(UP,0.05),(CGAP,-0.35),(USIDE,-0.3),(SHOE,-0.7)])
    setw(9, [(FBOOT,1.7),(SHAFT,0.8),(SHOE,0.45),(SHAFT_SHOE,0.55),(UP,-0.45),(FSNEAK,-0.2)])
    Wfeat_out *= 0.45
    bfeat_out = np.array([0.08,0.04,0.10,-0.03,0.10,0.08,0.18,0.15,-0.18,0.02], dtype=np.float32)
    Wedge_out = np.diag(np.array([0.25,0.30,0.25,0.25,0.25,0.35,0.22,0.35,0.20,0.35], dtype=np.float32))
    bedge_out = np.zeros(10, dtype=np.float32)

    # Hinge features give a few if/then-style shape cues: e.g. shoes have
    # bright lower bands and little mass above them; trousers have two lower
    # legs and a central gap.  Thresholds are hand-set gray fractions.
    hdefs = []
    def addh(mask, thresh, high=True):
        if high:
            hdefs.append((norm_mask(mask).reshape(-1), -float(thresh)))
        else:
            hdefs.append((-norm_mask(mask).reshape(-1), float(thresh)))
    body_mid = rect(13.5, 17.0, 20.0, 14.0)
    center_low_hi = center_low
    addh(upper, 0.10, False)       # 0 upper dark
    addh(upper, 0.12, True)        # 1 upper bright
    addh(shoe_low, 0.10, True)     # 2 shoe band bright
    addh(bottom, 0.08, False)      # 3 bottom dark
    addh(bottom, 0.11, True)       # 4 bottom bright
    addh(center_low, 0.08, False)  # 5 central lower gap dark
    addh(side_low, 0.09, True)     # 6 lower side legs bright
    addh(wide_low, 0.12, True)     # 7 wide hem/body bright
    addh(side_upper, 0.10, True)   # 8 sleeves/upper sides bright
    addh(handle, 0.045, True)      # 9 handle bright
    addh(boot_shaft_lr, 0.09, True)# 10 boot shaft bright
    addh(top, 0.08, False)         # 11 top dark
    addh(top, 0.11, True)          # 12 top bright
    addh(body_mid, 0.12, True)     # 13 compact middle body bright
    addh(center_low_hi, 0.10, True)# 14 lower middle bright
    HF = len(hdefs)
    Whinge = np.stack([h[0] for h in hdefs]).astype(np.float32)
    bhinge = np.array([h[1] for h in hdefs], dtype=np.float32)
    Whinge_out = np.zeros((10, HF), dtype=np.float32)
    def seth(k, pairs):
        for i,v in pairs:
            Whinge_out[k,i] = v
    seth(0, [(1,0.8),(3,0.9),(6,-0.7),(7,-0.3),(8,0.2)])
    seth(1, [(6,1.0),(5,0.9),(4,0.4),(8,-0.5),(7,-0.3)])
    seth(2, [(1,0.9),(8,0.6),(3,0.2),(5,-0.25)])
    seth(3, [(7,1.0),(14,0.7),(1,0.3),(5,-0.6),(8,-0.4)])
    seth(4, [(1,0.7),(8,0.8),(7,0.5),(3,-0.2),(5,-0.2)])
    seth(5, [(0,1.0),(2,0.8),(11,0.6),(10,-0.5),(13,-0.4)])
    seth(6, [(1,0.6),(8,0.3),(3,0.5),(7,-0.25)])
    seth(7, [(0,1.0),(2,1.0),(11,0.6),(10,-0.5),(13,-0.2)])
    seth(8, [(9,0.9),(13,0.8),(7,0.3),(11,0.2),(2,-0.45)])
    seth(9, [(0,0.6),(2,0.7),(10,1.0),(11,0.2),(7,-0.15)])
    Whinge_out *= 1.25
    bhinge_out = np.array([0.03,0.00,0.05,-0.03,0.03,0.05,0.07,0.08,-0.03,0.03], dtype=np.float32)

    net = {"layers": [
        {"name": "tmpl", "op": "dense", "inputs": ["image"], "out": int(10 * V)},
        {"name": "group", "op": "reshape", "inputs": ["tmpl"], "shape": [10, int(V), 1]},
        {"name": "best", "op": "maxpool", "inputs": ["group"], "size": [int(V), 1], "stride": [int(V), 1]},
        {"name": "flat", "op": "reshape", "inputs": ["best"], "shape": [10]},
        {"name": "tmpl_scores", "op": "dense", "inputs": ["flat"], "out": 10},
        {"name": "feat_raw", "op": "dense", "inputs": ["image"], "out": int(F)},
        {"name": "feat_scores", "op": "dense", "inputs": ["feat_raw"], "out": 10},
        {"name": "hinge_raw", "op": "dense", "inputs": ["image"], "out": int(HF)},
        {"name": "hinge", "op": "relu", "inputs": ["hinge_raw"]},
        {"name": "hinge_scores", "op": "dense", "inputs": ["hinge"], "out": 10},
        {"name": "edge", "op": "conv", "inputs": ["image"], "out": 8, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "edge_pos", "op": "relu", "inputs": ["edge"]},
        {"name": "edge_tmpl", "op": "dense", "inputs": ["edge_pos"], "out": int(10 * EV)},
        {"name": "edge_group", "op": "reshape", "inputs": ["edge_tmpl"], "shape": [10, int(EV), 1]},
        {"name": "edge_best", "op": "maxpool", "inputs": ["edge_group"], "size": [int(EV), 1], "stride": [int(EV), 1]},
        {"name": "edge_flat", "op": "reshape", "inputs": ["edge_best"], "shape": [10]},
        {"name": "edge_scores", "op": "dense", "inputs": ["edge_flat"], "out": 10},
        {"name": "scores", "op": "add", "inputs": ["tmpl_scores", "feat_scores", "hinge_scores", "edge_scores"]}
    ]}
    (out_dir / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(out_dir / "weights.npz", **{
        "tmpl.W": Wtmpl, "tmpl.b": btmpl, "tmpl_scores.W": Wtmpl_out, "tmpl_scores.b": btmpl_out,
        "feat_raw.W": Wfeat, "feat_raw.b": bfeat, "feat_scores.W": Wfeat_out, "feat_scores.b": bfeat_out,
        "hinge_raw.W": Whinge, "hinge_raw.b": bhinge, "hinge_scores.W": Whinge_out, "hinge_scores.b": bhinge_out,
        "edge.W": edge_filters()[:, None, :, :].astype(np.float32), "edge.b": np.zeros(8, dtype=np.float32),
        "edge_tmpl.W": Wedge_tmpl, "edge_tmpl.b": bedge_tmpl, "edge_scores.W": Wedge_out, "edge_scores.b": bedge_out})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
