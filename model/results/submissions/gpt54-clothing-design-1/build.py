
import json
import sys
from pathlib import Path

import numpy as np

H = W = 28
Y, X = np.mgrid[0:H, 0:W]
XN = (X + 0.5 - W / 2) / (W / 2)
YN = (Y + 0.5 - H / 2) / (H / 2)


def f32(x):
    return np.asarray(x, dtype=np.float32)


def union(*masks):
    out = np.zeros((H, W), np.float32)
    for m in masks:
        out = np.maximum(out, f32(m))
    return out


def subtract(a, b):
    return np.clip(f32(a) - f32(b), 0.0, 1.0)


def box(cx, cy, wx, hy):
    return ((np.abs(XN - cx) <= wx / 2) & (np.abs(YN - cy) <= hy / 2)).astype(np.float32)


def ellipse(cx, cy, rx, ry):
    return ((((XN - cx) / rx) ** 2 + ((YN - cy) / ry) ** 2) <= 1.0).astype(np.float32)


def trapezoid(y0, y1, w0, w1, cx=0.0):
    t = np.clip((YN - y0) / (y1 - y0 + 1e-12), 0.0, 1.0)
    w = w0 + (w1 - w0) * t
    return (((YN >= y0) & (YN <= y1)) & (np.abs(XN - cx) <= w / 2)).astype(np.float32)


def segment(x0, y0, x1, y1, thickness):
    vx, vy = x1 - x0, y1 - y0
    den = vx * vx + vy * vy + 1e-12
    t = ((XN - x0) * vx + (YN - y0) * vy) / den
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    d2 = (XN - px) ** 2 + (YN - py) ** 2
    return (d2 <= (thickness / 2) ** 2).astype(np.float32)


def ring_ellipse(cx, cy, rx, ry, thick, top_only=False):
    outer = ellipse(cx, cy, rx, ry)
    inner = ellipse(cx, cy, max(rx - thick, 1e-3), max(ry - thick, 1e-3))
    ring = subtract(outer, inner)
    if top_only:
        ring *= (YN <= cy).astype(np.float32)
    return ring


def pool4(mask):
    return f32(mask).reshape(7, 4, 7, 4).mean((1, 3))


def pool2(mask):
    return f32(mask).reshape(14, 2, 14, 2).mean((1, 3))


def mirror(mask):
    return f32(mask)[:, ::-1]


def norm_mask(mask):
    mask = f32(mask)
    s = float(mask.sum())
    return mask / s if s > 0 else mask


def detector_from(fg, bg, *, pool, pos, neg, thresh):
    fg_p = pool(fg)
    bg_p = pool(bg) if bg is not None else None
    w = pos * norm_mask(fg_p)
    if bg_p is not None and float(bg_p.sum()) > 0:
        w -= neg * norm_mask(bg_p)
    return w.reshape(-1), np.float32(-thresh)


# --- templates ---

def tee_template():
    body = trapezoid(-0.72, 0.50, 0.36, 0.50)
    sl = box(-0.46, -0.42, 0.24, 0.20)
    sr = box(0.46, -0.42, 0.24, 0.20)
    neck = ellipse(0.0, -0.80, 0.12, 0.07)
    fg = subtract(union(body, sl, sr), neck)
    bg = subtract(box(0.0, -0.08, 1.00, 1.36), fg)
    return fg, bg


def tank_template():
    body = trapezoid(-0.72, 0.56, 0.30, 0.46)
    straps = union(box(-0.15, -0.78, 0.10, 0.14), box(0.15, -0.78, 0.10, 0.14))
    neck = ellipse(0.0, -0.74, 0.11, 0.08)
    fg = subtract(union(body, straps), neck)
    bg = subtract(box(0.0, -0.04, 0.86, 1.44), fg)
    return fg, bg


def pullover_template():
    body = trapezoid(-0.76, 0.62, 0.40, 0.52)
    sl = segment(-0.22, -0.62, -0.72, 0.38, 0.20)
    sr = segment(0.22, -0.62, 0.72, 0.38, 0.20)
    neck = ellipse(0.0, -0.82, 0.13, 0.06)
    fg = subtract(union(body, sl, sr), neck)
    bg = subtract(box(0.0, -0.02, 1.18, 1.62), fg)
    return fg, bg


def coat_template():
    body = trapezoid(-0.78, 0.90, 0.46, 0.60)
    sl = segment(-0.24, -0.64, -0.78, 0.56, 0.20)
    sr = segment(0.24, -0.64, 0.78, 0.56, 0.20)
    neck = ellipse(0.0, -0.84, 0.13, 0.06)
    slit = box(0.0, 0.18, 0.08, 1.24)
    fg = subtract(subtract(union(body, sl, sr), neck), slit)
    bg = subtract(box(0.0, 0.04, 1.28, 1.86), fg)
    return fg, bg


def shirt_short_template():
    body = trapezoid(-0.76, 0.58, 0.40, 0.54)
    sl = box(-0.48, -0.38, 0.22, 0.22)
    sr = box(0.48, -0.38, 0.22, 0.22)
    collar = union(segment(-0.10, -0.82, -0.26, -0.64, 0.08), segment(0.10, -0.82, 0.26, -0.64, 0.08))
    slit = box(0.0, -0.16, 0.06, 0.74)
    fg = subtract(union(body, sl, sr, collar), slit)
    bg = subtract(box(0.0, -0.02, 1.10, 1.48), fg)
    return fg, bg


def shirt_long_template():
    body = trapezoid(-0.76, 0.72, 0.42, 0.56)
    sl = segment(-0.22, -0.62, -0.70, 0.46, 0.18)
    sr = segment(0.22, -0.62, 0.70, 0.46, 0.18)
    collar = union(segment(-0.08, -0.82, -0.24, -0.64, 0.08), segment(0.08, -0.82, 0.24, -0.64, 0.08))
    slit = box(0.0, -0.10, 0.06, 0.98)
    fg = subtract(union(body, sl, sr, collar), slit)
    bg = subtract(box(0.0, 0.00, 1.18, 1.68), fg)
    return fg, bg


def dress_flared_template():
    top = trapezoid(-0.78, -0.08, 0.32, 0.22)
    skirt = trapezoid(-0.08, 0.96, 0.24, 0.74)
    sleeves = union(box(-0.34, -0.48, 0.12, 0.16), box(0.34, -0.48, 0.12, 0.16))
    neck = ellipse(0.0, -0.82, 0.10, 0.06)
    fg = subtract(union(top, skirt, sleeves), neck)
    bg = subtract(box(0.0, 0.08, 1.14, 1.88), fg)
    return fg, bg


def dress_straight_template():
    top = trapezoid(-0.78, -0.02, 0.28, 0.22)
    skirt = trapezoid(-0.02, 0.92, 0.22, 0.40)
    straps = union(box(-0.12, -0.80, 0.08, 0.12), box(0.12, -0.80, 0.08, 0.12))
    neck = ellipse(0.0, -0.78, 0.09, 0.07)
    fg = subtract(union(top, skirt, straps), neck)
    bg = subtract(box(0.0, 0.04, 0.86, 1.84), fg)
    return fg, bg


def trouser_template():
    waist = trapezoid(-0.82, -0.28, 0.34, 0.52)
    leg_l = trapezoid(-0.28, 0.98, 0.24, 0.18, cx=-0.18)
    leg_r = trapezoid(-0.28, 0.98, 0.24, 0.18, cx=0.18)
    gap = box(0.0, 0.44, 0.12, 1.00)
    fg = subtract(union(waist, leg_l, leg_r), gap)
    bg = subtract(box(0.0, 0.08, 0.82, 1.92), fg)
    return fg, bg


def bag_tote_template():
    body = trapezoid(-0.06, 0.84, 0.54, 0.70)
    handle = ring_ellipse(0.0, -0.20, 0.26, 0.20, 0.10, top_only=True)
    posts = union(box(-0.16, -0.08, 0.06, 0.14), box(0.16, -0.08, 0.06, 0.14))
    fg = union(body, handle, posts)
    bg = subtract(box(0.0, 0.12, 0.96, 1.48), fg)
    return fg, bg


def bag_satchel_template():
    body = trapezoid(0.02, 0.72, 0.80, 0.62)
    flap = trapezoid(-0.12, 0.18, 0.70, 0.82)
    handle = ring_ellipse(0.0, -0.18, 0.22, 0.18, 0.08, top_only=True)
    posts = union(box(-0.12, -0.08, 0.06, 0.14), box(0.12, -0.08, 0.06, 0.14))
    fg = union(body, flap, handle, posts)
    bg = subtract(box(0.0, 0.12, 1.06, 1.22), fg)
    return fg, bg


def sandal_flat_template():
    sole = segment(-0.68, 0.54, 0.60, 0.52, 0.18)
    back = segment(-0.58, 0.44, -0.70, 0.08, 0.10)
    strap1 = segment(-0.12, 0.42, 0.12, 0.28, 0.12)
    strap2 = segment(0.16, 0.40, 0.42, 0.26, 0.12)
    fg = union(sole, back, strap1, strap2)
    bg = subtract(box(-0.02, 0.38, 1.36, 0.88), fg)
    return fg, bg


def sandal_heel_template():
    heel = segment(-0.62, 0.52, -0.46, -0.10, 0.10)
    sole = segment(-0.40, 0.54, 0.52, 0.50, 0.14)
    ankle = segment(-0.38, -0.04, -0.18, -0.28, 0.10)
    front = segment(0.10, 0.34, 0.34, 0.24, 0.12)
    fg = union(heel, sole, ankle, front)
    bg = subtract(box(-0.06, 0.24, 1.26, 1.18), fg)
    return fg, bg


def sneaker_template():
    sole = segment(-0.70, 0.56, 0.58, 0.54, 0.20)
    body = ellipse(-0.04, 0.34, 0.58, 0.24)
    collar = box(-0.34, 0.12, 0.24, 0.20)
    cut = box(-0.66, 0.06, 0.22, 0.42)
    fg = subtract(union(sole, body, collar), cut)
    bg = subtract(box(-0.04, 0.34, 1.42, 0.94), fg)
    return fg, bg


def boot_template():
    sole = segment(-0.70, 0.56, 0.56, 0.54, 0.20)
    foot = ellipse(-0.04, 0.34, 0.58, 0.24)
    shaft = trapezoid(-0.48, 0.34, 0.28, 0.34, cx=-0.28)
    heel = box(-0.60, 0.36, 0.14, 0.34)
    fg = union(sole, foot, shaft, heel)
    bg = subtract(box(-0.06, 0.18, 1.42, 1.34), fg)
    return fg, bg


def make_specs():
    tee = tee_template()
    tank = tank_template()
    pullover = pullover_template()
    coat = coat_template()
    shirt_short = shirt_short_template()
    shirt_long = shirt_long_template()
    dress_flared = dress_flared_template()
    dress_straight = dress_straight_template()
    trouser = trouser_template()
    bag_tote = bag_tote_template()
    bag_satchel = bag_satchel_template()
    sandal_flat = sandal_flat_template()
    sandal_heel = sandal_heel_template()
    sneaker = sneaker_template()
    boot = boot_template()

    sandal_flat_r = (mirror(sandal_flat[0]), mirror(sandal_flat[1]))
    sandal_heel_r = (mirror(sandal_heel[0]), mirror(sandal_heel[1]))
    sneaker_r = (mirror(sneaker[0]), mirror(sneaker[1]))
    boot_r = (mirror(boot[0]), mirror(boot[1]))

    branch7 = [
        ('b7_ttop', union(tee[0], tank[0]), subtract(box(0.0, -0.04, 1.00, 1.42), union(tee[0], tank[0])), dict(pos=2.2, neg=1.2, thresh=0.48)),
        ('b7_trouser', trouser[0], trouser[1], dict(pos=2.4, neg=1.2, thresh=0.52)),
        ('b7_pullover', pullover[0], pullover[1], dict(pos=2.2, neg=1.1, thresh=0.48)),
        ('b7_dress', union(dress_flared[0], dress_straight[0]), subtract(box(0.0, 0.08, 1.14, 1.88), union(dress_flared[0], dress_straight[0])), dict(pos=2.3, neg=1.1, thresh=0.50)),
        ('b7_coat', coat[0], coat[1], dict(pos=2.3, neg=1.2, thresh=0.50)),
        ('b7_sandal', union(sandal_flat[0], sandal_heel[0], sandal_flat_r[0], sandal_heel_r[0]), subtract(box(-0.04, 0.30, 1.34, 1.12), union(sandal_flat[0], sandal_heel[0], sandal_flat_r[0], sandal_heel_r[0])), dict(pos=2.4, neg=1.2, thresh=0.52)),
        ('b7_shirt', union(shirt_short[0], shirt_long[0]), subtract(box(0.0, 0.00, 1.14, 1.66), union(shirt_short[0], shirt_long[0])), dict(pos=2.2, neg=1.1, thresh=0.48)),
        ('b7_sneaker', union(sneaker[0], sneaker_r[0]), subtract(box(-0.04, 0.34, 1.42, 0.94), union(sneaker[0], sneaker_r[0])), dict(pos=2.5, neg=1.2, thresh=0.52)),
        ('b7_bag', union(bag_tote[0], bag_satchel[0]), subtract(box(0.0, 0.12, 1.06, 1.44), union(bag_tote[0], bag_satchel[0])), dict(pos=2.4, neg=1.2, thresh=0.54)),
        ('b7_boot', union(boot[0], boot_r[0]), subtract(box(-0.06, 0.18, 1.42, 1.34), union(boot[0], boot_r[0])), dict(pos=2.5, neg=1.2, thresh=0.52)),
    ]

    # coarse features
    branch7 += [
        ('f7_two_legs', union(box(-0.22, 0.52, 0.20, 0.92), box(0.22, 0.52, 0.20, 0.92), box(0.0, -0.52, 0.48, 0.56)), box(0.0, 0.48, 0.16, 0.84), dict(pos=2.5, neg=1.6, thresh=0.90)),
        ('f7_handle', union(ring_ellipse(0.0, -0.20, 0.24, 0.18, 0.08, top_only=True), box(-0.14, -0.06, 0.06, 0.14), box(0.14, -0.06, 0.06, 0.14)), box(0.0, -0.18, 0.18, 0.10), dict(pos=2.6, neg=1.8, thresh=0.74)),
        ('f7_long_sleeves', union(segment(-0.24, -0.58, -0.72, 0.34, 0.18), segment(0.24, -0.58, 0.72, 0.34, 0.18), box(0.0, -0.06, 0.48, 0.96)), union(box(-0.68, -0.42, 0.16, 0.20), box(0.68, -0.42, 0.16, 0.20)), dict(pos=2.2, neg=1.0, thresh=0.70)),
        ('f7_open_front', union(box(-0.18, -0.12, 0.28, 1.20), box(0.18, -0.12, 0.28, 1.20)), box(0.0, -0.10, 0.12, 1.10), dict(pos=2.2, neg=2.0, thresh=0.44)),
        ('f7_long_body', union(box(0.0, 0.40, 0.56, 0.86), box(0.0, -0.28, 0.44, 0.76)), box(0.0, -0.70, 1.10, 0.24), dict(pos=2.1, neg=0.9, thresh=0.64)),
        ('f7_skirt_flare', union(box(-0.30, 0.58, 0.22, 0.52), box(0.30, 0.58, 0.22, 0.52), box(0.0, -0.42, 0.34, 0.54)), box(0.0, 0.42, 0.18, 0.56), dict(pos=2.6, neg=1.6, thresh=0.78)),
        ('f7_low_profile', union(box(0.0, 0.50, 1.08, 0.26), box(0.10, 0.30, 0.56, 0.20)), box(0.0, -0.24, 1.18, 0.56), dict(pos=2.2, neg=1.4, thresh=0.66)),
        ('f7_ankle_shaft', union(box(-0.34, -0.02, 0.26, 0.62), segment(-0.62, 0.54, 0.44, 0.52, 0.18)), box(0.06, 0.10, 0.32, 0.56), dict(pos=2.3, neg=1.0, thresh=0.76)),
        ('f7_bag_box', box(0.0, 0.28, 0.72, 0.74), union(box(0.0, -0.64, 1.06, 0.22), box(0.0, 0.88, 1.06, 0.18)), dict(pos=2.2, neg=1.0, thresh=0.68)),
        ('f7_short_length', union(box(0.0, -0.28, 0.58, 0.78), box(-0.42, -0.38, 0.22, 0.18), box(0.42, -0.38, 0.22, 0.18)), box(0.0, 0.78, 1.00, 0.24), dict(pos=2.2, neg=1.8, thresh=0.70)),
    ]

    branch14 = [
        ('p14_tee', tee[0], tee[1], dict(pos=2.6, neg=1.2, thresh=0.56)),
        ('p14_tank', tank[0], tank[1], dict(pos=2.6, neg=1.2, thresh=0.56)),
        ('p14_pullover', pullover[0], pullover[1], dict(pos=2.6, neg=1.2, thresh=0.56)),
        ('p14_coat', coat[0], coat[1], dict(pos=2.6, neg=1.2, thresh=0.58)),
        ('p14_shirt_short', shirt_short[0], shirt_short[1], dict(pos=2.6, neg=1.2, thresh=0.56)),
        ('p14_shirt_long', shirt_long[0], shirt_long[1], dict(pos=2.6, neg=1.2, thresh=0.56)),
        ('p14_dress_flared', dress_flared[0], dress_flared[1], dict(pos=2.6, neg=1.2, thresh=0.58)),
        ('p14_dress_straight', dress_straight[0], dress_straight[1], dict(pos=2.6, neg=1.2, thresh=0.58)),
        ('p14_trouser', trouser[0], trouser[1], dict(pos=2.8, neg=1.2, thresh=0.60)),
        ('p14_bag_tote', bag_tote[0], bag_tote[1], dict(pos=2.7, neg=1.2, thresh=0.58)),
        ('p14_bag_satchel', bag_satchel[0], bag_satchel[1], dict(pos=2.7, neg=1.2, thresh=0.58)),
        ('p14_sandal_flat_l', sandal_flat[0], sandal_flat[1], dict(pos=2.7, neg=1.3, thresh=0.60)),
        ('p14_sandal_flat_r', sandal_flat_r[0], sandal_flat_r[1], dict(pos=2.7, neg=1.3, thresh=0.60)),
        ('p14_sandal_heel_l', sandal_heel[0], sandal_heel[1], dict(pos=2.7, neg=1.3, thresh=0.60)),
        ('p14_sandal_heel_r', sandal_heel_r[0], sandal_heel_r[1], dict(pos=2.7, neg=1.3, thresh=0.60)),
        ('p14_sneaker_l', sneaker[0], sneaker[1], dict(pos=2.8, neg=1.3, thresh=0.58)),
        ('p14_sneaker_r', sneaker_r[0], sneaker_r[1], dict(pos=2.8, neg=1.3, thresh=0.58)),
        ('p14_boot_l', boot[0], boot[1], dict(pos=2.8, neg=1.3, thresh=0.58)),
        ('p14_boot_r', boot_r[0], boot_r[1], dict(pos=2.8, neg=1.3, thresh=0.58)),
    ]

    branch14 += [
        ('f14_collar_v', union(segment(-0.06, -0.80, -0.22, -0.62, 0.06), segment(0.06, -0.80, 0.22, -0.62, 0.06)), box(0.0, -0.72, 0.08, 0.18), dict(pos=2.8, neg=1.4, thresh=0.38)),
        ('f14_short_sleeves', union(box(-0.48, -0.40, 0.22, 0.20), box(0.48, -0.40, 0.22, 0.20), box(0.0, -0.12, 0.46, 0.84)), union(box(-0.72, 0.24, 0.14, 0.42), box(0.72, 0.24, 0.14, 0.42)), dict(pos=2.4, neg=1.1, thresh=0.62)),
        ('f14_open_front', union(box(-0.18, -0.12, 0.28, 1.20), box(0.18, -0.12, 0.28, 1.20)), box(0.0, -0.10, 0.12, 1.10), dict(pos=2.5, neg=2.2, thresh=0.42)),
        ('f14_long_sleeves', union(segment(-0.24, -0.58, -0.72, 0.34, 0.18), segment(0.24, -0.58, 0.72, 0.34, 0.18), box(0.0, -0.06, 0.48, 0.96)), union(box(-0.68, -0.42, 0.16, 0.20), box(0.68, -0.42, 0.16, 0.20)), dict(pos=2.5, neg=1.0, thresh=0.64)),
        ('f14_handle', union(ring_ellipse(0.0, -0.20, 0.24, 0.18, 0.08, top_only=True), box(-0.14, -0.06, 0.06, 0.14), box(0.14, -0.06, 0.06, 0.14)), box(0.0, -0.18, 0.18, 0.10), dict(pos=2.8, neg=1.9, thresh=0.62)),
        ('f14_straps_top', union(box(-0.15, -0.78, 0.10, 0.14), box(0.15, -0.78, 0.10, 0.14)), box(0.0, -0.76, 0.12, 0.10), dict(pos=2.6, neg=1.7, thresh=0.42)),
        ('f14_leg_gap', union(box(-0.18, 0.46, 0.16, 0.94), box(0.18, 0.46, 0.16, 0.94), box(0.0, -0.54, 0.34, 0.54)), box(0.0, 0.48, 0.12, 0.92), dict(pos=2.7, neg=1.8, thresh=0.66)),
        ('f14_airy_sandal', union(segment(-0.68, 0.56, 0.58, 0.54, 0.18), box(-0.56, 0.20, 0.10, 0.40), box(0.26, 0.24, 0.12, 0.20)), box(0.04, 0.28, 0.36, 0.26), dict(pos=2.8, neg=1.8, thresh=0.68)),
        ('f14_closed_shoe', union(segment(-0.68, 0.56, 0.58, 0.54, 0.18), ellipse(-0.06, 0.32, 0.50, 0.20)), box(0.0, -0.18, 1.18, 0.50), dict(pos=2.6, neg=1.3, thresh=0.62)),
        ('f14_left_shaft', union(box(-0.34, -0.04, 0.24, 0.58), box(-0.58, 0.44, 0.18, 0.22), segment(-0.60, 0.56, 0.48, 0.54, 0.14)), box(0.18, 0.06, 0.24, 0.54), dict(pos=2.7, neg=1.1, thresh=0.70)),
        ('f14_right_shaft', mirror(union(box(-0.34, -0.04, 0.24, 0.58), box(-0.58, 0.44, 0.18, 0.22), segment(-0.60, 0.56, 0.48, 0.54, 0.14))), mirror(box(0.18, 0.06, 0.24, 0.54)), dict(pos=2.7, neg=1.1, thresh=0.70)),
        ('f14_boxy_mid', union(box(-0.30, 0.30, 0.12, 0.66), box(0.30, 0.30, 0.12, 0.66), box(0.0, 0.60, 0.60, 0.10)), union(box(0.0, -0.56, 1.06, 0.26), box(0.0, 0.90, 1.06, 0.16)), dict(pos=2.5, neg=1.1, thresh=0.66)),
    ]

    return branch7, branch14


def add_coeffs(W, idx, cls, coeffs):
    for name, val in coeffs.items():
        W[cls, idx[name]] += np.float32(val)


def make_weights():
    branch7, branch14 = make_specs()

    names7 = [name for name, _, _, _ in branch7]
    names14 = [name for name, _, _, _ in branch14]

    W7 = np.zeros((len(branch7), 49), np.float32)
    b7 = np.zeros((len(branch7),), np.float32)
    for i, (name, fg, bg, params) in enumerate(branch7):
        W7[i], b7[i] = detector_from(fg, bg, pool=pool4, **params)

    W14 = np.zeros((len(branch14), 196), np.float32)
    b14 = np.zeros((len(branch14),), np.float32)
    for i, (name, fg, bg, params) in enumerate(branch14):
        W14[i], b14[i] = detector_from(fg, bg, pool=pool2, **params)

    idx7 = {n: i for i, n in enumerate(names7)}
    idx14 = {n: i for i, n in enumerate(names14)}

    S7 = np.zeros((10, len(branch7)), np.float32)
    sb7 = np.array([-0.30, -0.10, -0.22, -0.22, -0.20, -0.14, -0.34, -0.14, -0.65, -0.14], np.float32)
    add_coeffs(S7, idx7, 0, {'b7_ttop': 3.2, 'f7_short_length': 1.2, 'f7_long_sleeves': -1.0, 'f7_open_front': -1.2, 'f7_long_body': -1.1, 'b7_shirt': -1.0, 'b7_coat': -0.8, 'b7_bag': -0.8, 'b7_dress': -0.8})
    add_coeffs(S7, idx7, 1, {'b7_trouser': 3.2, 'f7_two_legs': 2.1, 'f7_bag_box': -1.2, 'f7_low_profile': -1.0, 'f7_skirt_flare': -1.2, 'b7_bag': -1.2, 'b7_dress': -0.8})
    add_coeffs(S7, idx7, 2, {'b7_pullover': 3.0, 'f7_long_sleeves': 1.3, 'f7_open_front': -1.2, 'f7_long_body': 0.2, 'f7_short_length': -0.4, 'b7_coat': -0.8, 'b7_shirt': -0.8, 'b7_bag': -1.0})
    add_coeffs(S7, idx7, 3, {'b7_dress': 3.2, 'f7_long_body': 1.2, 'f7_skirt_flare': 1.7, 'f7_two_legs': -1.8, 'f7_low_profile': -1.0, 'b7_bag': -1.4, 'b7_coat': -0.7, 'b7_ttop': -0.8})
    add_coeffs(S7, idx7, 4, {'b7_coat': 3.0, 'f7_long_sleeves': 1.4, 'f7_open_front': 1.2, 'f7_long_body': 1.4, 'f7_short_length': -1.0, 'b7_pullover': -0.8, 'b7_shirt': -0.8, 'b7_dress': -0.6, 'b7_bag': -0.8})
    add_coeffs(S7, idx7, 5, {'b7_sandal': 3.0, 'f7_low_profile': 1.3, 'f7_ankle_shaft': -1.6, 'f7_bag_box': -0.8, 'b7_sneaker': -0.9, 'b7_boot': -1.0})
    add_coeffs(S7, idx7, 6, {'b7_shirt': 1.8, 'f7_open_front': 1.5, 'f7_long_sleeves': 0.3, 'f7_short_length': 0.2, 'b7_ttop': -1.0, 'b7_bag': -0.7, 'b7_coat': -0.4})
    add_coeffs(S7, idx7, 7, {'b7_sneaker': 3.2, 'f7_low_profile': 1.4, 'f7_ankle_shaft': -1.3, 'f7_bag_box': -1.1, 'b7_bag': -1.5, 'b7_sandal': -1.0, 'b7_boot': -0.8})
    add_coeffs(S7, idx7, 8, {'b7_bag': 1.6, 'f7_handle': 2.6, 'f7_bag_box': 1.8, 'f7_low_profile': -1.7, 'f7_two_legs': -1.3, 'f7_long_sleeves': -1.0, 'b7_ttop': -0.8, 'b7_shirt': -0.8, 'b7_trouser': -1.0, 'b7_sneaker': -1.2, 'b7_boot': -1.2, 'b7_dress': -0.8})
    add_coeffs(S7, idx7, 9, {'b7_boot': 3.2, 'f7_ankle_shaft': 2.1, 'f7_low_profile': 0.3, 'f7_bag_box': -0.9, 'b7_bag': -1.4, 'b7_sandal': -1.0, 'b7_sneaker': -0.8})

    S14 = np.zeros((10, len(branch14)), np.float32)
    sb14 = np.array([-0.10, -0.04, -0.08, -0.08, -0.08, -0.04, -0.26, -0.04, -0.52, -0.04], np.float32)
    add_coeffs(S14, idx14, 0, {'p14_tee': 2.9, 'p14_tank': 2.6, 'f14_short_sleeves': 1.1, 'f14_long_sleeves': -1.0, 'f14_open_front': -1.2, 'f14_collar_v': -0.8, 'f14_straps_top': 0.3, 'p14_shirt_short': -0.9, 'p14_shirt_long': -0.9, 'p14_dress_flared': -0.5})
    add_coeffs(S14, idx14, 1, {'p14_trouser': 3.1, 'f14_leg_gap': 2.0, 'f14_boxy_mid': -0.7, 'f14_closed_shoe': -1.0, 'p14_dress_flared': -0.6, 'p14_bag_tote': -0.8, 'p14_bag_satchel': -0.8})
    add_coeffs(S14, idx14, 2, {'p14_pullover': 2.8, 'f14_long_sleeves': 1.1, 'f14_open_front': -1.2, 'f14_collar_v': -0.5, 'p14_coat': -0.8, 'p14_shirt_long': -0.8})
    add_coeffs(S14, idx14, 3, {'p14_dress_flared': 2.7, 'p14_dress_straight': 2.5, 'f14_straps_top': 0.5, 'f14_leg_gap': -0.9, 'p14_bag_tote': -1.0, 'p14_bag_satchel': -1.0, 'p14_tee': -0.6, 'p14_tank': -0.6})
    add_coeffs(S14, idx14, 4, {'p14_coat': 2.9, 'f14_open_front': 1.2, 'f14_long_sleeves': 1.1, 'f14_collar_v': 0.3, 'p14_pullover': -0.7, 'p14_shirt_long': -0.5})
    add_coeffs(S14, idx14, 5, {'p14_sandal_flat_l': 1.5, 'p14_sandal_flat_r': 1.5, 'p14_sandal_heel_l': 1.5, 'p14_sandal_heel_r': 1.5, 'f14_airy_sandal': 1.6, 'f14_closed_shoe': -1.5, 'f14_left_shaft': -0.9, 'f14_right_shaft': -0.9, 'p14_sneaker_l': -0.8, 'p14_sneaker_r': -0.8, 'p14_boot_l': -1.0, 'p14_boot_r': -1.0})
    add_coeffs(S14, idx14, 6, {'p14_shirt_short': 1.8, 'p14_shirt_long': 1.8, 'f14_collar_v': 1.3, 'f14_open_front': 1.3, 'f14_short_sleeves': 0.2, 'f14_long_sleeves': 0.2, 'p14_tee': -1.0, 'p14_tank': -1.0})
    add_coeffs(S14, idx14, 7, {'p14_sneaker_l': 1.7, 'p14_sneaker_r': 1.7, 'f14_closed_shoe': 1.3, 'f14_airy_sandal': -1.1, 'f14_left_shaft': -0.8, 'f14_right_shaft': -0.8, 'f14_boxy_mid': -0.8, 'p14_bag_tote': -1.2, 'p14_bag_satchel': -1.2})
    add_coeffs(S14, idx14, 8, {'p14_bag_tote': 1.6, 'p14_bag_satchel': 1.6, 'f14_handle': 2.3, 'f14_boxy_mid': 1.4, 'f14_closed_shoe': -1.3, 'f14_leg_gap': -0.8, 'p14_tee': -0.8, 'p14_shirt_short': -0.8, 'p14_sneaker_l': -1.0, 'p14_sneaker_r': -1.0, 'p14_boot_l': -1.0, 'p14_boot_r': -1.0})
    add_coeffs(S14, idx14, 9, {'p14_boot_l': 1.8, 'p14_boot_r': 1.8, 'f14_left_shaft': 1.5, 'f14_right_shaft': 1.5, 'f14_closed_shoe': 0.9, 'f14_airy_sandal': -1.3, 'f14_handle': -0.4, 'p14_bag_tote': -0.8, 'p14_bag_satchel': -0.8, 'p14_sneaker_l': -0.7, 'p14_sneaker_r': -0.7})

    return {
        'W7': W7, 'b7': b7, 'S7': S7, 'sb7': sb7,
        'W14': W14, 'b14': b14, 'S14': S14, 'sb14': sb14,
    }


def build_network_and_weights():
    w = make_weights()
    net = {
        'layers': [
            {'name': 'p7', 'op': 'avgpool', 'inputs': ['image'], 'size': [4, 4], 'stride': [4, 4]},
            {'name': 'h7', 'op': 'dense', 'inputs': ['p7'], 'out': int(w['W7'].shape[0])},
            {'name': 'r7', 'op': 'relu', 'inputs': ['h7']},
            {'name': 's7', 'op': 'dense', 'inputs': ['r7'], 'out': 10},
            {'name': 'p14', 'op': 'avgpool', 'inputs': ['image'], 'size': [2, 2], 'stride': [2, 2]},
            {'name': 'h14', 'op': 'dense', 'inputs': ['p14'], 'out': int(w['W14'].shape[0])},
            {'name': 'r14', 'op': 'relu', 'inputs': ['h14']},
            {'name': 's14', 'op': 'dense', 'inputs': ['r14'], 'out': 10},
            {'name': 'scores', 'op': 'add', 'inputs': ['s7', 's14']},
        ]
    }
    arrays = {
        'h7.W': w['W7'].astype(np.float32),
        'h7.b': w['b7'].astype(np.float32),
        's7.W': w['S7'].astype(np.float32),
        's7.b': w['sb7'].astype(np.float32),
        'h14.W': w['W14'].astype(np.float32),
        'h14.b': w['b14'].astype(np.float32),
        's14.W': w['S14'].astype(np.float32),
        's14.b': w['sb14'].astype(np.float32),
    }
    return net, arrays


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net, arrays = build_network_and_weights()
    (out / 'network.json').write_text(json.dumps(net, separators=(',', ':')))
    np.savez(out / 'weights.npz', **arrays)


if __name__ == '__main__':
    main(sys.argv[1])
