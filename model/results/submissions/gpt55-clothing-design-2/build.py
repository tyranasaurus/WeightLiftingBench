import json, os, sys
import numpy as np

SIZE = 28
T = 192

Y, X = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
XX = X + 0.5
YY = Y + 0.5


def add_poly(a, pts, val=1.0):
    pts = np.asarray(pts, dtype=np.float32)
    x = XX; y = YY
    inside = np.zeros((SIZE, SIZE), dtype=bool)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]; xj, yj = pts[j]
        hit = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-6) + xi)
        inside ^= hit
        j = i
    a[inside] = np.maximum(a[inside], val)


def cut_poly(a, pts):
    pts = np.asarray(pts, dtype=np.float32)
    x = XX; y = YY
    inside = np.zeros((SIZE, SIZE), dtype=bool)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]; xj, yj = pts[j]
        hit = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-6) + xi)
        inside ^= hit
        j = i
    a[inside] = 0.0


def add_rect(a, x0, y0, x1, y1, val=1.0):
    m = (XX >= x0) & (XX <= x1) & (YY >= y0) & (YY <= y1)
    a[m] = np.maximum(a[m], val)


def cut_rect(a, x0, y0, x1, y1):
    m = (XX >= x0) & (XX <= x1) & (YY >= y0) & (YY <= y1)
    a[m] = 0.0


def add_ellipse(a, cx, cy, rx, ry, val=1.0):
    m = ((XX - cx) / (rx + 1e-6)) ** 2 + ((YY - cy) / (ry + 1e-6)) ** 2 <= 1.0
    a[m] = np.maximum(a[m], val)


def cut_ellipse(a, cx, cy, rx, ry):
    m = ((XX - cx) / (rx + 1e-6)) ** 2 + ((YY - cy) / (ry + 1e-6)) ** 2 <= 1.0
    a[m] = 0.0


def add_line(a, x1, y1, x2, y2, width=1.0, val=1.0):
    vx = x2 - x1; vy = y2 - y1
    t = ((XX - x1) * vx + (YY - y1) * vy) / (vx * vx + vy * vy + 1e-6)
    t = np.clip(t, 0.0, 1.0)
    px = x1 + t * vx; py = y1 + t * vy
    d = np.sqrt((XX - px) ** 2 + (YY - py) ** 2)
    m = d <= width
    a[m] = np.maximum(a[m], val * (1.0 - 0.25 * d[m] / (width + 1e-6)))


def cut_line(a, x1, y1, x2, y2, width=1.0):
    vx = x2 - x1; vy = y2 - y1
    t = ((XX - x1) * vx + (YY - y1) * vy) / (vx * vx + vy * vy + 1e-6)
    t = np.clip(t, 0.0, 1.0)
    px = x1 + t * vx; py = y1 + t * vy
    d = np.sqrt((XX - px) ** 2 + (YY - py) ** 2)
    a[d <= width] = 0.0


def blur(a, n=1):
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    b = a.astype(np.float32)
    for _ in range(n):
        p = np.pad(b, ((1, 1), (1, 1)), mode='constant')
        b = sum(k[i, j] * p[i:i+SIZE, j:j+SIZE] for i in range(3) for j in range(3))
    return b


def norm_template(a):
    a = blur(np.clip(a, 0.0, 1.0), 1)
    v = a.reshape(-1).astype(np.float32)
    v = v - v.mean(dtype=np.float32)
    n = np.sqrt(np.sum(v * v, dtype=np.float32)) + 1e-6
    return v / n


def raw_template(a):
    a = blur(np.clip(a, 0.0, 1.0), 1)
    v = a.reshape(-1).astype(np.float32)
    n = np.sqrt(np.sum(v * v, dtype=np.float32)) + 1e-6
    return v / n


def mirror(a):
    return a[:, ::-1].copy()


def shift_img(a, dx, dy):
    b = np.zeros_like(a)
    xs0 = max(0, -dx); xs1 = min(SIZE, SIZE - dx)
    ys0 = max(0, -dy); ys1 = min(SIZE, SIZE - dy)
    b[ys0+dy:ys1+dy, xs0+dx:xs1+dx] = a[ys0:ys1, xs0:xs1]
    return b


def tshirt(rng):
    a = np.zeros((SIZE, SIZE), np.float32)
    cx = 14 + rng.uniform(-1.0, 1.0)
    top = rng.uniform(2.0, 4.0); sh_y = top + rng.uniform(2.0, 3.5)
    bottom = rng.uniform(18.5, 23.0)
    hw_sh = rng.uniform(4.8, 7.0); hw_bot = hw_sh + rng.uniform(-1.2, 0.8)
    add_poly(a, [(cx-hw_sh, sh_y), (cx+hw_sh, sh_y), (cx+hw_bot, bottom), (cx-hw_bot, bottom)], 0.95)
    mode = rng.integers(0, 4)
    if mode == 0:  # tank/vest
        add_poly(a, [(cx-hw_sh, sh_y), (cx-hw_sh-1.2, sh_y+3), (cx-hw_sh+0.6, sh_y+6), (cx-hw_sh+1.2, sh_y+1)], 0.8)
        add_poly(a, [(cx+hw_sh, sh_y), (cx+hw_sh+1.2, sh_y+3), (cx+hw_sh-0.6, sh_y+6), (cx+hw_sh-1.2, sh_y+1)], 0.8)
    else:
        sl = rng.uniform(3.5, 7.0); sd = rng.uniform(2.0, 5.5)
        add_poly(a, [(cx-hw_sh, sh_y), (cx-hw_sh-sl, sh_y+1.0), (cx-hw_sh-sl+0.7, sh_y+sd+2.0), (cx-hw_sh+0.8, sh_y+sd)], 0.9)
        add_poly(a, [(cx+hw_sh, sh_y), (cx+hw_sh+sl, sh_y+1.0), (cx+hw_sh+sl-0.7, sh_y+sd+2.0), (cx+hw_sh-0.8, sh_y+sd)], 0.9)
    # neckline holes: round, vee, or scoop
    if rng.random() < 0.45:
        cut_ellipse(a, cx, sh_y + 0.1, rng.uniform(1.4, 2.7), rng.uniform(1.0, 2.0))
    else:
        cut_poly(a, [(cx-2.1, sh_y-0.2), (cx+2.1, sh_y-0.2), (cx, sh_y+3.0)])
    if rng.random() < 0.30:
        add_line(a, cx-hw_bot+0.5, bottom-1.0, cx+hw_bot-0.5, bottom-1.0, 0.45, 0.65)
    return a


def trouser(rng):
    a = np.zeros((SIZE, SIZE), np.float32)
    cx = 14 + rng.uniform(-0.8, 0.8)
    top = rng.uniform(2.0, 5.0); crotch = rng.uniform(10.0, 14.0); bot = rng.uniform(26.0, 28.2)
    waist = rng.uniform(5.0, 7.5); hip = waist + rng.uniform(0.0, 1.8)
    gap = rng.uniform(1.3, 3.3)
    ankle = rng.uniform(1.7, 3.4); thigh = rng.uniform(2.5, 4.2)
    add_poly(a, [(cx-waist, top), (cx+waist, top), (cx+hip, crotch), (cx+gap/2+thigh, crotch), (cx+gap/2+ankle, bot), (cx+gap/2-ankle, bot), (cx+gap/2-thigh, crotch), (cx-gap/2, crotch), (cx-hip, crotch)], 0.95)
    add_poly(a, [(cx-hip, crotch), (cx-gap/2-thigh, crotch), (cx-gap/2-ankle, bot), (cx-waist*0.85, bot), (cx-waist, top)], 0.95)
    add_poly(a, [(cx+hip, crotch), (cx+gap/2+thigh, crotch), (cx+gap/2+ankle, bot), (cx+waist*0.85, bot), (cx+waist, top)], 0.95)
    cut_poly(a, [(cx-gap/2, crotch-1.0), (cx+gap/2, crotch-1.0), (cx+rng.uniform(0.7,1.8), bot), (cx-rng.uniform(0.7,1.8), bot)])
    add_rect(a, cx-waist, top, cx+waist, top+rng.uniform(1.0,2.0), 0.9)
    if rng.random() < 0.35:
        add_line(a, cx-waist+2, top+2, cx-waist+3, bot-1, 0.45, 0.45)
        add_line(a, cx+waist-2, top+2, cx+waist-3, bot-1, 0.45, 0.45)
    return a


def pullover(rng):
    a = np.zeros((SIZE, SIZE), np.float32)
    cx = 14 + rng.uniform(-1.0, 1.0)
    top = rng.uniform(1.5, 3.5); sh_y = top + rng.uniform(2.0, 3.4)
    bottom = rng.uniform(21.0, 25.0)
    hw_sh = rng.uniform(5.2, 7.4); hw_bot = hw_sh + rng.uniform(-1.0, 1.0)
    add_poly(a, [(cx-hw_sh, sh_y), (cx+hw_sh, sh_y), (cx+hw_bot, bottom), (cx-hw_bot, bottom)], 0.95)
    wrist_y = rng.uniform(20.0, 25.0); wrist_w = rng.uniform(1.3, 2.4)
    left_wx = cx - rng.uniform(9.0, 12.0); right_wx = cx + rng.uniform(9.0, 12.0)
    add_poly(a, [(cx-hw_sh, sh_y), (cx-hw_sh-2.5, sh_y+2.0), (left_wx-wrist_w, wrist_y), (left_wx+wrist_w, wrist_y+0.5), (cx-hw_sh+1.0, sh_y+5.5)], 0.92)
    add_poly(a, [(cx+hw_sh, sh_y), (cx+hw_sh+2.5, sh_y+2.0), (right_wx+wrist_w, wrist_y), (right_wx-wrist_w, wrist_y+0.5), (cx+hw_sh-1.0, sh_y+5.5)], 0.92)
    if rng.random() < 0.55:
        cut_ellipse(a, cx, sh_y, rng.uniform(1.5, 2.8), rng.uniform(0.9, 1.6))
    else:
        cut_poly(a, [(cx-2.4, sh_y-0.2), (cx+2.4, sh_y-0.2), (cx, sh_y+2.4)])
    if rng.random() < 0.45:
        add_line(a, cx-hw_bot+0.4, bottom-1.0, cx+hw_bot-0.4, bottom-1.0, 0.6, 0.65)
        add_line(a, left_wx-wrist_w, wrist_y, left_wx+wrist_w, wrist_y, 0.6, 0.65)
        add_line(a, right_wx-wrist_w, wrist_y, right_wx+wrist_w, wrist_y, 0.6, 0.65)
    return a


def dress(rng):
    a = np.zeros((SIZE, SIZE), np.float32)
    cx = 14 + rng.uniform(-0.8, 0.8)
    top = rng.uniform(1.5, 4.0); sh_y = top + rng.uniform(2.0, 3.5)
    waist_y = rng.uniform(9.5, 14.0); bot = rng.uniform(23.5, 28.0)
    hw_top = rng.uniform(3.7, 6.2); hw_waist = rng.uniform(3.0, 5.5); hw_hem = rng.uniform(6.5, 12.5)
    add_poly(a, [(cx-hw_top, sh_y), (cx+hw_top, sh_y), (cx+hw_waist, waist_y), (cx+hw_hem, bot), (cx-hw_hem, bot), (cx-hw_waist, waist_y)], 0.95)
    if rng.random() < 0.42:
        sl = rng.uniform(2.5, 5.5); sd = rng.uniform(2.0, 5.0)
        add_poly(a, [(cx-hw_top, sh_y), (cx-hw_top-sl, sh_y+1.0), (cx-hw_top-sl+0.5, sh_y+sd+1.5), (cx-hw_top+0.7, sh_y+sd)], 0.85)
        add_poly(a, [(cx+hw_top, sh_y), (cx+hw_top+sl, sh_y+1.0), (cx+hw_top+sl-0.5, sh_y+sd+1.5), (cx+hw_top-0.7, sh_y+sd)], 0.85)
    if rng.random() < 0.55:
        cut_ellipse(a, cx, sh_y, rng.uniform(1.4, 2.7), rng.uniform(1.0, 1.9))
    else:
        cut_poly(a, [(cx-2.3, sh_y-0.2), (cx+2.3, sh_y-0.2), (cx, sh_y+3.1)])
    if rng.random() < 0.35:
        add_line(a, cx-hw_waist, waist_y, cx+hw_waist, waist_y, 0.45, 0.65)
    return a


def coat(rng):
    a = np.zeros((SIZE, SIZE), np.float32)
    cx = 14 + rng.uniform(-0.8, 0.8)
    top = rng.uniform(1.5, 3.5); sh_y = top + rng.uniform(2.0, 3.5)
    bot = rng.uniform(23.5, 28.0)
    hw_sh = rng.uniform(5.8, 8.0); hw_bot = hw_sh + rng.uniform(-0.5, 2.0)
    add_poly(a, [(cx-hw_sh, sh_y), (cx+hw_sh, sh_y), (cx+hw_bot, bot), (cx-hw_bot, bot)], 0.95)
    wy = rng.uniform(21.0, 26.5); ww = rng.uniform(1.4, 2.7)
    lwx = cx - rng.uniform(9.0, 12.5); rwx = cx + rng.uniform(9.0, 12.5)
    add_poly(a, [(cx-hw_sh, sh_y), (cx-hw_sh-2.5, sh_y+2.0), (lwx-ww, wy), (lwx+ww, wy+0.5), (cx-hw_sh+1.2, sh_y+6.0)], 0.92)
    add_poly(a, [(cx+hw_sh, sh_y), (cx+hw_sh+2.5, sh_y+2.0), (rwx+ww, wy), (rwx-ww, wy+0.5), (cx+hw_sh-1.2, sh_y+6.0)], 0.92)
    # open front and lapel/collar holes, characteristic of coats
    cut_line(a, cx + rng.uniform(-0.4,0.4), sh_y+1.0, cx + rng.uniform(-0.4,0.4), bot-1.0, rng.uniform(0.45, 0.9))
    cut_poly(a, [(cx-3.5, sh_y-0.2), (cx-0.3, sh_y+3.6), (cx-1.2, sh_y+7.0)])
    cut_poly(a, [(cx+3.5, sh_y-0.2), (cx+0.3, sh_y+3.6), (cx+1.2, sh_y+7.0)])
    if rng.random() < 0.45:
        add_line(a, cx-1.5, sh_y+5.0, cx-1.5, bot-2.0, 0.35, 0.55)
        add_line(a, cx+1.5, sh_y+5.0, cx+1.5, bot-2.0, 0.35, 0.55)
    return a


def sandal(rng):
    a = np.zeros((SIZE, SIZE), np.float32)
    flip = rng.random() < 0.5
    # foot extends along the bottom, with open straps and sometimes a heel
    ysole = rng.uniform(21.5, 24.5); x0 = rng.uniform(1.0, 3.0); x1 = rng.uniform(24.0, 27.0)
    add_line(a, x0, ysole+0.5, x1, ysole+rng.uniform(-0.6,0.4), rng.uniform(0.8,1.4), 0.95)
    add_line(a, x0+1.5, ysole-0.5, x1-1.5, ysole-0.9+rng.uniform(-0.5,0.5), rng.uniform(0.45,0.9), 0.75)
    # toe/ankle straps
    for _ in range(rng.integers(2, 5)):
        x = rng.uniform(5.0, 22.0)
        h = rng.uniform(4.0, 11.0)
        sl = rng.uniform(-3.5, 3.5)
        add_line(a, x, ysole-0.3, x+sl, ysole-h, rng.uniform(0.55,1.05), 0.9)
    add_line(a, rng.uniform(15,22), ysole-rng.uniform(7,12), rng.uniform(20,26), ysole-rng.uniform(5,9), 0.7, 0.9)
    if rng.random() < 0.45:
        hx = rng.uniform(21.0,25.0); ht = rng.uniform(5.0,12.0)
        add_line(a, hx, ysole, hx+rng.uniform(0.5,2.0), ysole+ht*0.45, rng.uniform(0.6,1.1), 0.85)
        add_line(a, hx-1.0, ysole-ht, hx+1.0, ysole-ht, 0.6, 0.75)
    if flip:
        a = mirror(a)
    return a


def shirt(rng):
    a = np.zeros((SIZE, SIZE), np.float32)
    cx = 14 + rng.uniform(-1.0, 1.0)
    top = rng.uniform(1.5, 3.8); sh_y = top + rng.uniform(2.0, 3.4)
    bottom = rng.uniform(19.5, 24.5)
    hw_sh = rng.uniform(4.8, 7.2); hw_bot = hw_sh + rng.uniform(-1.0, 1.2)
    add_poly(a, [(cx-hw_sh, sh_y), (cx+hw_sh, sh_y), (cx+hw_bot, bottom), (cx-hw_bot, bottom)], 0.92)
    if rng.random() < 0.55:
        # long sleeves
        wy = rng.uniform(18.5, 25.0); ww = rng.uniform(1.1, 2.3)
        lwx = cx - rng.uniform(8.5, 11.5); rwx = cx + rng.uniform(8.5, 11.5)
        add_poly(a, [(cx-hw_sh, sh_y), (cx-hw_sh-2.2, sh_y+2.0), (lwx-ww, wy), (lwx+ww, wy+0.5), (cx-hw_sh+1.0, sh_y+5.4)], 0.9)
        add_poly(a, [(cx+hw_sh, sh_y), (cx+hw_sh+2.2, sh_y+2.0), (rwx+ww, wy), (rwx-ww, wy+0.5), (cx+hw_sh-1.0, sh_y+5.4)], 0.9)
    else:
        sl = rng.uniform(3.5, 6.5); sd = rng.uniform(2.2, 5.2)
        add_poly(a, [(cx-hw_sh, sh_y), (cx-hw_sh-sl, sh_y+1.0), (cx-hw_sh-sl+0.5, sh_y+sd+1.6), (cx-hw_sh+0.8, sh_y+sd)], 0.9)
        add_poly(a, [(cx+hw_sh, sh_y), (cx+hw_sh+sl, sh_y+1.0), (cx+hw_sh+sl-0.5, sh_y+sd+1.6), (cx+hw_sh-0.8, sh_y+sd)], 0.9)
    # collars, placket, buttons/opening
    cut_poly(a, [(cx-2.6, sh_y-0.2), (cx, sh_y+3.0), (cx-0.6, sh_y+5.2), (cx-4.0, sh_y+1.2)])
    cut_poly(a, [(cx+2.6, sh_y-0.2), (cx, sh_y+3.0), (cx+0.6, sh_y+5.2), (cx+4.0, sh_y+1.2)])
    add_poly(a, [(cx-3.2, sh_y), (cx-0.4, sh_y+3.6), (cx-1.4, sh_y+4.6)], 0.8)
    add_poly(a, [(cx+3.2, sh_y), (cx+0.4, sh_y+3.6), (cx+1.4, sh_y+4.6)], 0.8)
    cut_line(a, cx, sh_y+2.0, cx, bottom-1.0, rng.uniform(0.3,0.65))
    if rng.random() < 0.5:
        for by in np.linspace(sh_y+5, bottom-3, 3):
            cut_ellipse(a, cx, by, 0.55, 0.55)
    return a


def sneaker(rng):
    a = np.zeros((SIZE, SIZE), np.float32)
    flip = rng.random() < 0.5
    x0 = rng.uniform(1.0, 3.0); x1 = rng.uniform(24.0, 27.0)
    yb = rng.uniform(22.0, 25.0); h = rng.uniform(6.0, 10.5)
    # filled low shoe upper plus thick sole
    add_poly(a, [(x0, yb-1.0), (x0+2.0, yb-3.0), (x0+7.0, yb-h), (x0+13.0, yb-h+1.0), (x1-2.5, yb-4.0), (x1, yb-1.0), (x1-1.0, yb+1.6), (x0+0.5, yb+1.8)], 0.95)
    add_poly(a, [(x0, yb+0.2), (x1, yb-0.4), (x1-0.6, yb+2.0), (x0+0.4, yb+2.5)], 0.9)
    cut_ellipse(a, x0+8.0, yb-h+1.2, rng.uniform(2.0,3.4), rng.uniform(1.0,1.8))
    for k in range(rng.integers(2, 5)):
        xx = x0 + rng.uniform(8.0, 17.0)
        add_line(a, xx, yb-h+2.0, xx+rng.uniform(2.0,4.0), yb-2.5, 0.35, 0.55)
    if rng.random() < 0.45:
        add_line(a, x0+2, yb+2.0, x1-2, yb+1.4, 0.5, 0.55)
    if flip:
        a = mirror(a)
    return a


def bag(rng):
    a = np.zeros((SIZE, SIZE), np.float32)
    cx = 14 + rng.uniform(-1.0, 1.0)
    top = rng.uniform(8.5, 13.0); bot = rng.uniform(22.5, 27.0)
    hw_top = rng.uniform(5.5, 9.5); hw_bot = hw_top + rng.uniform(-1.5, 1.5)
    style = rng.integers(0, 4)
    if style == 0:
        add_poly(a, [(cx-hw_top, top), (cx+hw_top, top), (cx+hw_bot, bot), (cx-hw_bot, bot)], 0.95)
    elif style == 1:
        add_rect(a, cx-hw_top, top, cx+hw_top, bot, 0.95)
        add_ellipse(a, cx-hw_top, top+2, 2.5, 2.5, 0.95); add_ellipse(a, cx+hw_top, top+2, 2.5, 2.5, 0.95)
    elif style == 2:
        add_ellipse(a, cx, (top+bot)/2, hw_top, (bot-top)/2, 0.95)
        add_rect(a, cx-hw_top, (top+bot)/2, cx+hw_top, bot, 0.95)
    else:
        add_poly(a, [(cx-hw_top*0.8, top), (cx+hw_top*0.8, top), (cx+hw_top, bot-1), (cx+hw_bot*0.7, bot), (cx-hw_bot*0.7, bot), (cx-hw_top, bot-1)], 0.95)
    # handle loop: two straps or arch
    hy = top - rng.uniform(5.0, 8.0)
    add_line(a, cx-hw_top*0.55, top+0.5, cx-hw_top*0.25, hy, rng.uniform(0.55,1.1), 0.9)
    add_line(a, cx+hw_top*0.55, top+0.5, cx+hw_top*0.25, hy, rng.uniform(0.55,1.1), 0.9)
    add_line(a, cx-hw_top*0.25, hy, cx+hw_top*0.25, hy, rng.uniform(0.55,1.1), 0.9)
    cut_rect(a, cx-hw_top*0.25, hy+0.5, cx+hw_top*0.25, top-0.6)
    if rng.random() < 0.60:
        cut_line(a, cx-hw_top*0.8, top+rng.uniform(3.5,6.5), cx+hw_top*0.8, top+rng.uniform(3.5,6.5), rng.uniform(0.35,0.7))
    if rng.random() < 0.45:
        cut_ellipse(a, cx, top+rng.uniform(4.5,8.5), rng.uniform(0.7,1.2), rng.uniform(0.5,1.0))
    return a


def boot(rng):
    a = np.zeros((SIZE, SIZE), np.float32)
    flip = rng.random() < 0.5
    x0 = rng.uniform(1.5, 4.0); x1 = rng.uniform(23.0, 27.0)
    ysole = rng.uniform(22.0, 25.0)
    shaft_top = rng.uniform(5.0, 11.5); shaft_x0 = rng.uniform(8.0, 13.0); shaft_x1 = shaft_x0 + rng.uniform(5.0, 8.5)
    add_poly(a, [(shaft_x0, shaft_top), (shaft_x1, shaft_top+0.4), (shaft_x1+0.8, ysole-5.0), (x1-2.0, ysole-2.5), (x1, ysole-0.2), (x1-0.8, ysole+1.7), (x0+2.0, ysole+2.0), (x0, ysole+0.5), (shaft_x0-0.8, ysole-5.0)], 0.95)
    # heel and sole
    add_poly(a, [(x0+2.0, ysole+1.2), (shaft_x0+1.0, ysole+1.2), (shaft_x0+0.3, ysole+4.0), (x0+1.0, ysole+4.0)], 0.88)
    add_line(a, x0, ysole+1.8, x1-1.0, ysole+1.1, 0.9, 0.85)
    if rng.random() < 0.45:
        cut_line(a, shaft_x0+1.2, shaft_top+1.5, shaft_x0+1.7, ysole-3.5, 0.45)
    if rng.random() < 0.35:
        add_line(a, shaft_x0+0.8, shaft_top+1.2, shaft_x1-0.8, shaft_top+1.2, 0.45, 0.65)
    if flip:
        a = mirror(a)
    return a


GENS = [tshirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, boot]
# Small hand offsets make the max-correlation scores use a comparable prior scale.
BIASES = np.array([-0.20, -0.55, -0.30, -0.20, -0.45, 0.65, -0.70, 0.70, 0.05, 0.60], dtype=np.float32)
TEMPLATE_SCALE = 0.55


def mrect(x0, y0, x1, y1):
    m = ((XX >= x0) & (XX < x1) & (YY >= y0) & (YY < y1)).astype(np.float32)
    s = float(m.sum())
    return (m / s).reshape(-1) if s else m.reshape(-1)


def mline(x1, y1, x2, y2, width):
    vx = x2 - x1; vy = y2 - y1
    t = ((XX - x1) * vx + (YY - y1) * vy) / (vx * vx + vy * vy + 1e-6)
    t = np.clip(t, 0.0, 1.0)
    px = x1 + t * vx; py = y1 + t * vy
    m = (np.sqrt((XX - px) ** 2 + (YY - py) ** 2) <= width).astype(np.float32)
    s = float(m.sum())
    return (m / s).reshape(-1) if s else m.reshape(-1)


def build_rule():
    R = np.zeros((10, SIZE * SIZE), dtype=np.float32)
    b = np.array([-0.6, -0.8, -0.7, -0.7, -0.9, -0.6, -0.8, -0.7, -0.8, -0.8], dtype=np.float32)
    allm = mrect(0, 0, 28, 28)
    top5 = mrect(0, 0, 28, 5); top10 = mrect(0, 0, 28, 10); upper = mrect(0, 4, 28, 15)
    mid = mrect(0, 8, 28, 20); lower = mrect(0, 16, 28, 28); bottom = mrect(0, 21, 28, 28); sole = mrect(0, 23, 28, 28)
    torso = mrect(6, 5, 22, 20); torso_low = mrect(6, 12, 22, 24)
    short_sleeves = mrect(0, 5, 8, 15) + mrect(20, 5, 28, 15)
    long_sleeves = mrect(0, 7, 8, 24) + mrect(20, 7, 28, 24)
    side_legs = mrect(5, 10, 12, 28) + mrect(16, 10, 23, 28)
    leg_gap = mrect(12, 12, 16, 28)
    waist = mrect(6, 4, 22, 10)
    hem = mrect(3, 20, 25, 28); center_low = mrect(9, 16, 19, 28)
    bag_body = mrect(4, 10, 24, 26); bag_handle = mrect(7, 3, 21, 11) - 0.5 * mrect(10, 6, 18, 12)
    shoe_low = mrect(2, 17, 27, 27); low_upper = mrect(4, 14, 24, 23); shaft = mrect(9, 6, 19, 22)
    diag_sandal = mline(5, 22, 20, 13, 1.1) + mline(9, 22, 23, 14, 1.1) + mline(3, 24, 26, 23, 1.2)
    # Hand-written broad shape scores.  Positive terms are object parts typical of a class;
    # negative terms suppress broad categories that share one part but not the whole shape.
    R[0] += 5.0 * torso + 3.0 * short_sleeves - 4.0 * lower - 2.5 * sole
    R[1] += 7.5 * side_legs + 2.5 * waist - 6.0 * leg_gap - 3.0 * top5 - 2.0 * torso
    R[2] += 5.0 * torso + 5.0 * long_sleeves + 1.5 * torso_low - 3.0 * sole - 1.0 * hem
    R[3] += 4.0 * torso + 5.5 * hem + 3.0 * center_low - 2.0 * leg_gap - 2.0 * sole
    R[4] += 5.0 * torso_low + 5.0 * long_sleeves + 2.0 * center_low - 2.0 * sole
    R[5] += 10.0 * sole + 5.0 * diag_sandal - 7.0 * upper - 2.5 * allm
    R[6] += 4.0 * torso + 3.0 * short_sleeves + 2.0 * long_sleeves - 3.0 * lower - 1.5 * sole
    R[7] += 11.0 * shoe_low + 5.0 * low_upper - 8.0 * top10 - 4.0 * shaft - 2.0 * allm
    R[8] += 8.0 * bag_body + 4.0 * bag_handle - 4.0 * top5 - 3.0 * sole - 1.5 * long_sleeves
    R[9] += 10.0 * shoe_low + 7.0 * shaft + 2.0 * sole - 5.0 * top5 - 3.0 * torso
    return (0.03 * R).astype(np.float32), (0.03 * b).astype(np.float32)


def build_weights():
    rng = np.random.default_rng(12345)
    rows = []
    bias = []
    for c, gen in enumerate(GENS):
        for i in range(T):
            a = gen(rng)
            # A few templates are shifted by one pixel to make the hand shapes tolerant to centering.
            if i % 16 == 13:
                a = shift_img(a, 1, 0)
            elif i % 16 == 14:
                a = shift_img(a, -1, 0)
            elif i % 16 == 15:
                a = shift_img(a, 0, 1)
            rows.append(norm_template(a) * TEMPLATE_SCALE)
            bias.append(BIASES[c])
    W = np.stack(rows, axis=0).astype(np.float32)
    b = np.array(bias, dtype=np.float32)
    return W, b


def write(outdir):
    os.makedirs(outdir, exist_ok=True)
    layers = [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "match", "op": "dense", "inputs": ["n0"], "out": 10 * T},
        {"name": "grid", "op": "reshape", "inputs": ["match"], "shape": [10, T, 1]},
        {"name": "best", "op": "maxpool", "inputs": ["grid"], "size": [T, 1], "stride": [T, 1]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]},
    ]
    with open(os.path.join(outdir, 'network.json'), 'w') as f:
        json.dump({"layers": layers}, f, separators=(',', ':'))
    W, b = build_weights()
    np.savez(os.path.join(outdir, 'weights.npz'), **{'match.W': W, 'match.b': b})


if __name__ == '__main__':
    write(sys.argv[1] if len(sys.argv) > 1 else '.')
