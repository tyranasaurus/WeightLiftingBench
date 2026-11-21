import json
import sys
from pathlib import Path

import numpy as np

CLASSES = 10
SCENE_CH = 6
SHAPE_K = 9
MASKS_PER_CLASS = 4
POLARITIES = 2
SHAPE_OUT = CLASSES * MASKS_PER_CLASS * POLARITIES
SCENE_ALPHA = 1.45
SHAPE_ALPHA = 2.30


def f32(x):
    return np.asarray(x, np.float32)


def grid(res):
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    x = (xx + 0.5) / res * 2.0 - 1.0
    y = (yy + 0.5) / res * 2.0 - 1.0
    return x, y


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_inside(level, soft=0.10):
    return smooth((1.0 - level) / soft + 0.5)


def rot(x, y, cx, cy, ang):
    ca, sa = np.cos(ang), np.sin(ang)
    xr = ca * (x - cx) + sa * (y - cy)
    yr = -sa * (x - cx) + ca * (y - cy)
    return xr, yr


def ellipse(x, y, cx, cy, rx, ry, ang=0.0, soft=0.10):
    xr, yr = rot(x, y, cx, cy, ang)
    level = np.sqrt((xr / max(rx, 1e-4)) ** 2 + (yr / max(ry, 1e-4)) ** 2)
    return soft_inside(level, soft)


def rect(x, y, cx, cy, rx, ry, ang=0.0, soft=0.10):
    xr, yr = rot(x, y, cx, cy, ang)
    level = np.maximum(np.abs(xr) / max(rx, 1e-4), np.abs(yr) / max(ry, 1e-4))
    return soft_inside(level, soft)


def segment(x, y, x0, y0, x1, y1, width=0.05, soft=0.04):
    vx, vy = x1 - x0, y1 - y0
    denom = vx * vx + vy * vy + 1e-6
    t = ((x - x0) * vx + (y - y0) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
    return smooth((width - dist) / soft + 0.5)


def union(*masks):
    out = masks[0]
    for m in masks[1:]:
        out = np.maximum(out, m)
    return out


def top_band(y, edge=0.0, soft=0.18):
    return smooth((edge - y) / soft + 0.5)


def bottom_band(y, edge=0.0, soft=0.18):
    return smooth((y - edge) / soft + 0.5)


def quadruped(x, y, cx, cy, s, ang, body=(0.34, 0.17), head=(0.11, 0.08), neck=(0.28, 0.20), leg=0.38,
              tail=0.16, slim=1.0, antlers=False, pointed_ears=False, long_muzzle=False):
    xb, yb = rot(x, y, cx, cy, ang)
    body_m = ellipse(xb, yb, 0.00, 0.00, body[0] * s, body[1] * s * slim, 0.0, 0.09)
    chest = ellipse(xb, yb, 0.16 * s, -0.01 * s, body[0] * 0.72 * s, body[1] * 0.92 * s * slim, 0.0, 0.09)
    neck_m = segment(xb, yb, 0.18 * s, -0.04 * s, neck[0] * s, -neck[1] * s, 0.08 * s, 0.05)
    head_m = ellipse(xb, yb, (neck[0] + 0.10) * s, (-neck[1] - 0.02) * s, head[0] * s, head[1] * s * slim, -0.15, 0.09)
    if long_muzzle:
        head_m = np.maximum(head_m, ellipse(xb, yb, (neck[0] + 0.20) * s, (-neck[1] - 0.01) * s, head[0] * 0.78 * s, head[1] * 0.48 * s, -0.15, 0.09))
    legs = 0.0
    for j, lx in enumerate((-0.16, -0.02, 0.12, 0.24)):
        dx = 0.02 * s if j % 2 == 0 else -0.02 * s
        legs = np.maximum(legs, segment(xb, yb, lx * s, 0.10 * s, lx * s + dx, leg * s, 0.04 * s, 0.04))
    tail_m = segment(xb, yb, -body[0] * s, -0.02 * s, -(body[0] + tail) * s, -0.12 * s, 0.03 * s, 0.04)
    mask = union(body_m, chest, neck_m, head_m, legs, tail_m)
    if pointed_ears:
        e1 = segment(xb, yb, (neck[0] + 0.10) * s, (-neck[1] - 0.10) * s, (neck[0] + 0.04) * s, (-neck[1] - 0.22) * s, 0.02 * s, 0.03)
        e2 = segment(xb, yb, (neck[0] + 0.16) * s, (-neck[1] - 0.10) * s, (neck[0] + 0.21) * s, (-neck[1] - 0.22) * s, 0.02 * s, 0.03)
        mask = np.maximum(mask, np.maximum(e1, e2))
    if antlers:
        a1 = segment(xb, yb, (neck[0] + 0.08) * s, (-neck[1] - 0.10) * s, (neck[0] - 0.01) * s, (-neck[1] - 0.28) * s, 0.015 * s, 0.03)
        a2 = segment(xb, yb, (neck[0] + 0.16) * s, (-neck[1] - 0.10) * s, (neck[0] + 0.25) * s, (-neck[1] - 0.26) * s, 0.015 * s, 0.03)
        mask = np.maximum(mask, np.maximum(a1, a2))
    return mask


def airplane_mask(x, y, s=1.0, ang=0.0, cx=0.0, cy=0.0):
    xr, yr = rot(x, y, cx, cy, ang)
    fus = ellipse(xr, yr, 0.0, 0.0, 0.44 * s, 0.08 * s, 0.0, 0.09)
    wing = ellipse(xr, yr, -0.02 * s, 0.0, 0.12 * s, 0.48 * s, 0.0, 0.09)
    tail = ellipse(xr, yr, -0.28 * s, 0.0, 0.06 * s, 0.18 * s, 0.0, 0.09)
    nose = ellipse(xr, yr, 0.34 * s, 0.0, 0.08 * s, 0.08 * s, 0.0, 0.09)
    return union(fus, wing, tail, nose)


def car_mask(x, y, s=1.0, ang=0.0, cx=0.0, cy=0.0, front=False, truck=False):
    if not front:
        xr, yr = rot(x, y, cx, cy, ang)
        if truck:
            body = rect(xr, yr, -0.04 * s, 0.00, 0.34 * s, 0.16 * s, 0.0, 0.09)
            cab = rect(xr, yr, 0.32 * s, -0.01 * s, 0.16 * s, 0.16 * s, 0.0, 0.09)
            roof = rect(xr, yr, 0.28 * s, -0.18 * s, 0.10 * s, 0.05 * s, 0.0, 0.09)
            w1 = ellipse(xr, yr, -0.20 * s, 0.18 * s, 0.09 * s, 0.09 * s, 0.0, 0.09)
            w2 = ellipse(xr, yr, 0.06 * s, 0.18 * s, 0.09 * s, 0.09 * s, 0.0, 0.09)
            w3 = ellipse(xr, yr, 0.32 * s, 0.18 * s, 0.09 * s, 0.09 * s, 0.0, 0.09)
            return union(body, cab, roof, w1, w2, w3)
        body = rect(xr, yr, 0.00, 0.02 * s, 0.38 * s, 0.12 * s, 0.0, 0.09)
        roof = rect(xr, yr, -0.04 * s, -0.12 * s, 0.18 * s, 0.10 * s, 0.0, 0.09)
        frontm = ellipse(xr, yr, 0.28 * s, -0.01 * s, 0.12 * s, 0.10 * s, 0.0, 0.09)
        w1 = ellipse(xr, yr, -0.22 * s, 0.16 * s, 0.09 * s, 0.09 * s, 0.0, 0.09)
        w2 = ellipse(xr, yr, 0.22 * s, 0.16 * s, 0.09 * s, 0.09 * s, 0.0, 0.09)
        return union(body, roof, frontm, w1, w2)
    body = rect(x, y, cx, cy, 0.22 * s if not truck else 0.28 * s, 0.18 * s if not truck else 0.24 * s, 0.0, 0.09)
    top = rect(x, y, cx, cy - 0.18 * s, 0.15 * s if not truck else 0.20 * s, 0.06 * s, 0.0, 0.09)
    wheels = union(ellipse(x, y, cx - 0.14 * s, cy + (0.18 if not truck else 0.23) * s, 0.08 * s, 0.08 * s, 0.0, 0.09),
                   ellipse(x, y, cx + 0.14 * s, cy + (0.18 if not truck else 0.23) * s, 0.08 * s, 0.08 * s, 0.0, 0.09))
    return union(body, top, wheels)


def bird_mask(x, y, kind=0, s=1.0, cx=0.0, cy=0.0):
    if kind == 0:
        body = ellipse(x, y, cx, cy, 0.18 * s, 0.12 * s, 0.12, 0.09)
        head = ellipse(x, y, cx + 0.16 * s, cy - 0.08 * s, 0.06 * s, 0.06 * s, 0.0, 0.09)
        tail = segment(x, y, cx - 0.14 * s, cy + 0.02 * s, cx - 0.32 * s, cy + 0.10 * s, 0.05 * s, 0.04)
        beak = segment(x, y, cx + 0.21 * s, cy - 0.07 * s, cx + 0.34 * s, cy - 0.05 * s, 0.015 * s, 0.03)
        return union(body, head, tail, beak)
    if kind == 1:
        body = ellipse(x, y, cx, cy, 0.15 * s, 0.10 * s, 0.0, 0.09)
        neck = segment(x, y, cx + 0.04 * s, cy - 0.02 * s, cx + 0.10 * s, cy - 0.22 * s, 0.03 * s, 0.03)
        head = ellipse(x, y, cx + 0.11 * s, cy - 0.27 * s, 0.06 * s, 0.06 * s, 0.0, 0.09)
        leg1 = segment(x, y, cx - 0.03 * s, cy + 0.08 * s, cx - 0.04 * s, cy + 0.32 * s, 0.02 * s, 0.03)
        leg2 = segment(x, y, cx + 0.05 * s, cy + 0.08 * s, cx + 0.05 * s, cy + 0.32 * s, 0.02 * s, 0.03)
        return union(body, neck, head, leg1, leg2)
    if kind == 2:
        body = ellipse(x, y, cx, cy, 0.10 * s, 0.08 * s, 0.0, 0.09)
        wing1 = ellipse(x, y, cx - 0.18 * s, cy - 0.02 * s, 0.30 * s, 0.09 * s, 0.55, 0.09)
        wing2 = ellipse(x, y, cx + 0.18 * s, cy - 0.02 * s, 0.30 * s, 0.09 * s, -0.55, 0.09)
        return union(body, wing1, wing2)
    body = ellipse(x, y, cx, cy, 0.16 * s, 0.11 * s, 0.10, 0.09)
    branch = segment(x, y, cx - 0.48 * s, cy + 0.18 * s, cx + 0.52 * s, cy + 0.12 * s, 0.06 * s, 0.04)
    head = ellipse(x, y, cx + 0.13 * s, cy - 0.08 * s, 0.05 * s, 0.05 * s, 0.0, 0.09)
    return union(body, head, branch)


def cat_mask(x, y, kind=0, s=1.0, cx=0.0, cy=0.0):
    if kind == 0:
        head = ellipse(x, y, cx, cy, 0.28 * s, 0.25 * s, 0.0, 0.09)
        ear1 = rect(x, y, cx - 0.18 * s, cy - 0.25 * s, 0.06 * s, 0.10 * s, -0.55, 0.09)
        ear2 = rect(x, y, cx + 0.18 * s, cy - 0.25 * s, 0.06 * s, 0.10 * s, 0.55, 0.09)
        muzzle = ellipse(x, y, cx, cy + 0.10 * s, 0.13 * s, 0.08 * s, 0.0, 0.09)
        return union(head, ear1, ear2, muzzle)
    if kind == 1:
        body = ellipse(x, y, cx, cy, 0.26 * s, 0.18 * s, 0.08, 0.09)
        head = ellipse(x, y, cx + 0.20 * s, cy - 0.16 * s, 0.11 * s, 0.10 * s, 0.0, 0.09)
        ear1 = rect(x, y, cx + 0.14 * s, cy - 0.25 * s, 0.03 * s, 0.06 * s, -0.45, 0.09)
        ear2 = rect(x, y, cx + 0.25 * s, cy - 0.25 * s, 0.03 * s, 0.06 * s, 0.45, 0.09)
        tail = segment(x, y, cx - 0.22 * s, cy - 0.02 * s, cx - 0.38 * s, cy - 0.24 * s, 0.04 * s, 0.04)
        return union(body, head, ear1, ear2, tail)
    if kind == 2:
        body = ellipse(x, y, cx, cy, 0.24 * s, 0.20 * s, 0.0, 0.09)
        head = ellipse(x, y, cx + 0.12 * s, cy - 0.10 * s, 0.10 * s, 0.09 * s, 0.0, 0.09)
        tail = ellipse(x, y, cx - 0.08 * s, cy + 0.10 * s, 0.18 * s, 0.07 * s, 0.9, 0.09)
        ears = union(rect(x, y, cx + 0.05 * s, cy - 0.18 * s, 0.025 * s, 0.05 * s, -0.5, 0.09),
                     rect(x, y, cx + 0.18 * s, cy - 0.18 * s, 0.025 * s, 0.05 * s, 0.5, 0.09))
        return union(body, head, tail, ears)
    return cat_mask(x, y, 0, s * 0.82, cx, cy - 0.04)


def dog_mask(x, y, kind=0, s=1.0, cx=0.0, cy=0.0):
    if kind == 0:
        head = ellipse(x, y, cx, cy, 0.25 * s, 0.23 * s, 0.0, 0.09)
        muzzle = ellipse(x, y, cx + 0.06 * s, cy + 0.12 * s, 0.15 * s, 0.09 * s, 0.0, 0.09)
        ear1 = ellipse(x, y, cx - 0.19 * s, cy - 0.03 * s, 0.07 * s, 0.16 * s, 0.2, 0.09)
        ear2 = ellipse(x, y, cx + 0.19 * s, cy - 0.03 * s, 0.07 * s, 0.16 * s, -0.2, 0.09)
        return union(head, muzzle, ear1, ear2)
    if kind == 1:
        return quadruped(x, y, cx, cy, s, 0.03, body=(0.28, 0.15), head=(0.10, 0.08), neck=(0.26, 0.16), leg=0.34, tail=0.13, slim=1.0, long_muzzle=True)
    if kind == 2:
        body = ellipse(x, y, cx, cy, 0.22 * s, 0.16 * s, 0.0, 0.09)
        head = ellipse(x, y, cx - 0.12 * s, cy - 0.16 * s, 0.11 * s, 0.10 * s, 0.0, 0.09)
        ear = ellipse(x, y, cx - 0.18 * s, cy - 0.16 * s, 0.05 * s, 0.10 * s, 0.3, 0.09)
        leg = union(segment(x, y, cx - 0.06 * s, cy + 0.08 * s, cx - 0.06 * s, cy + 0.28 * s, 0.025 * s, 0.03),
                    segment(x, y, cx + 0.08 * s, cy + 0.08 * s, cx + 0.10 * s, cy + 0.28 * s, 0.025 * s, 0.03))
        return union(body, head, ear, leg)
    return dog_mask(x, y, 0, s * 0.84, cx, cy - 0.05)


def frog_mask(x, y, kind=0, s=1.0, cx=0.0, cy=0.0):
    body = ellipse(x, y, cx, cy, 0.24 * s, 0.16 * s, 0.0, 0.09)
    head = ellipse(x, y, cx + 0.10 * s, cy - 0.08 * s, 0.14 * s, 0.10 * s, 0.0, 0.09)
    eye1 = ellipse(x, y, cx + 0.08 * s, cy - 0.14 * s, 0.04 * s, 0.04 * s, 0.0, 0.09)
    eye2 = ellipse(x, y, cx + 0.18 * s, cy - 0.12 * s, 0.04 * s, 0.04 * s, 0.0, 0.09)
    if kind == 0:
        leg1 = segment(x, y, cx - 0.12 * s, cy + 0.02 * s, cx - 0.30 * s, cy + 0.22 * s, 0.05 * s, 0.04)
        leg2 = segment(x, y, cx + 0.04 * s, cy + 0.06 * s, cx + 0.24 * s, cy + 0.28 * s, 0.05 * s, 0.04)
        fore1 = segment(x, y, cx + 0.06 * s, cy - 0.02 * s, cx + 0.24 * s, cy + 0.08 * s, 0.04 * s, 0.04)
        fore2 = segment(x, y, cx - 0.02 * s, cy - 0.02 * s, cx - 0.18 * s, cy + 0.10 * s, 0.04 * s, 0.04)
        return union(body, head, eye1, eye2, leg1, leg2, fore1, fore2)
    if kind == 1:
        leg1 = segment(x, y, cx - 0.18 * s, cy + 0.02 * s, cx - 0.36 * s, cy + 0.12 * s, 0.05 * s, 0.04)
        leg2 = segment(x, y, cx + 0.08 * s, cy + 0.10 * s, cx + 0.28 * s, cy + 0.18 * s, 0.05 * s, 0.04)
        return union(body, head, eye1, eye2, leg1, leg2)
    if kind == 2:
        body2 = ellipse(x, y, cx, cy, 0.28 * s, 0.18 * s, 0.0, 0.09)
        return union(body2, eye1, eye2)
    return frog_mask(x, y, 0, s * 0.86, cx, cy + 0.02)


def horse_mask(x, y, kind=0, s=1.0, cx=0.0, cy=0.0):
    if kind == 0:
        return quadruped(x, y, cx, cy, s, -0.02, body=(0.34, 0.17), head=(0.11, 0.08), neck=(0.30, 0.22), leg=0.40, tail=0.20, slim=1.0, pointed_ears=True)
    if kind == 1:
        return quadruped(x, y, cx, cy, s, 0.08, body=(0.34, 0.16), head=(0.11, 0.08), neck=(0.30, 0.24), leg=0.42, tail=0.18, slim=1.0, pointed_ears=True)
    if kind == 2:
        body = ellipse(x, y, cx, cy, 0.26 * s, 0.18 * s, 0.0, 0.09)
        neck = segment(x, y, cx + 0.10 * s, cy - 0.05 * s, cx + 0.24 * s, cy - 0.28 * s, 0.06 * s, 0.04)
        head = ellipse(x, y, cx + 0.30 * s, cy - 0.34 * s, 0.12 * s, 0.08 * s, -0.15, 0.09)
        return union(body, neck, head)
    return horse_mask(x, y, 0, s * 0.84, cx, cy)


def deer_mask(x, y, kind=0, s=1.0, cx=0.0, cy=0.0):
    if kind == 0:
        return quadruped(x, y, cx, cy, s, -0.04, body=(0.28, 0.14), head=(0.10, 0.07), neck=(0.28, 0.22), leg=0.42, tail=0.10, slim=0.90, antlers=True)
    if kind == 1:
        return quadruped(x, y, cx, cy, s, 0.06, body=(0.28, 0.14), head=(0.10, 0.07), neck=(0.28, 0.22), leg=0.42, tail=0.10, slim=0.90, antlers=False)
    if kind == 2:
        return quadruped(x, y, cx, cy, s * 0.90, 0.14, body=(0.28, 0.13), head=(0.09, 0.06), neck=(0.26, 0.20), leg=0.40, tail=0.08, slim=0.86, antlers=False)
    return deer_mask(x, y, 0, s * 0.82, cx, cy)


def ship_mask(x, y, kind=0, s=1.0, cx=0.0, cy=0.0):
    if kind == 0:
        hull = rect(x, y, cx, cy, 0.40 * s, 0.10 * s, 0.0, 0.09)
        bow = ellipse(x, y, cx + 0.30 * s, cy, 0.14 * s, 0.10 * s, 0.0, 0.09)
        cabin = rect(x, y, cx + 0.08 * s, cy - 0.15 * s, 0.10 * s, 0.07 * s, 0.0, 0.09)
        return union(hull, bow, cabin)
    if kind == 1:
        hull = rect(x, y, cx, cy, 0.28 * s, 0.09 * s, 0.0, 0.09)
        mast = segment(x, y, cx - 0.02 * s, cy - 0.02 * s, cx - 0.02 * s, cy - 0.48 * s, 0.02 * s, 0.03)
        sail = rect(x, y, cx + 0.10 * s, cy - 0.28 * s, 0.11 * s, 0.18 * s, -0.2, 0.09)
        return union(hull, mast, sail)
    if kind == 2:
        hull = rect(x, y, cx, cy, 0.46 * s, 0.10 * s, 0.0, 0.09)
        cabin = rect(x, y, cx - 0.10 * s, cy - 0.13 * s, 0.12 * s, 0.08 * s, 0.0, 0.09)
        stack = rect(x, y, cx + 0.10 * s, cy - 0.18 * s, 0.04 * s, 0.08 * s, 0.0, 0.09)
        return union(hull, cabin, stack)
    hull = ellipse(x, y, cx, cy + 0.02 * s, 0.18 * s, 0.16 * s, 0.0, 0.09)
    cabin = rect(x, y, cx, cy - 0.15 * s, 0.10 * s, 0.07 * s, 0.0, 0.09)
    return union(hull, cabin)


def shape_mask(cls, kind, x, y):
    if cls == 0:
        params = [(0.90, -0.10, 0.02, -0.02), (0.82, 0.38, 0.00, 0.04), (0.80, -0.55, -0.02, -0.02), (0.72, 0.00, 0.00, 0.02)]
        s, a, cx, cy = params[kind]
        return airplane_mask(x, y, s, a, cx, cy)
    if cls == 1:
        if kind == 0:
            return car_mask(x, y, 0.92, 0.03, -0.02, 0.18, front=False, truck=False)
        if kind == 1:
            return car_mask(x, y, 0.84, -0.05, 0.00, 0.20, front=False, truck=False)
        if kind == 2:
            return car_mask(x, y, 0.95, 0.00, 0.00, 0.18, front=True, truck=False)
        return car_mask(x, y, 0.78, 0.00, 0.02, 0.16, front=False, truck=False)
    if cls == 2:
        if kind == 0:
            return bird_mask(x, y, 0, 0.94, -0.02, 0.10)
        if kind == 1:
            return bird_mask(x, y, 1, 0.94, -0.06, 0.10)
        if kind == 2:
            return bird_mask(x, y, 2, 0.96, 0.00, 0.02)
        return bird_mask(x, y, 3, 0.88, 0.00, 0.08)
    if cls == 3:
        if kind == 0:
            return cat_mask(x, y, 0, 0.96, 0.00, -0.02)
        if kind == 1:
            return cat_mask(x, y, 1, 0.90, -0.04, 0.18)
        if kind == 2:
            return cat_mask(x, y, 2, 0.92, -0.02, 0.20)
        return cat_mask(x, y, 3, 0.78, 0.00, -0.04)
    if cls == 4:
        if kind == 0:
            return deer_mask(x, y, 0, 0.88, -0.06, 0.18)
        if kind == 1:
            return deer_mask(x, y, 1, 0.86, -0.06, 0.18)
        if kind == 2:
            return deer_mask(x, y, 2, 0.84, -0.02, 0.20)
        return deer_mask(x, y, 3, 0.76, -0.04, 0.20)
    if cls == 5:
        if kind == 0:
            return dog_mask(x, y, 0, 0.96, 0.00, 0.00)
        if kind == 1:
            return dog_mask(x, y, 1, 0.92, -0.04, 0.18)
        if kind == 2:
            return dog_mask(x, y, 2, 0.90, 0.02, 0.22)
        return dog_mask(x, y, 3, 0.82, 0.00, -0.04)
    if cls == 6:
        if kind == 0:
            return frog_mask(x, y, 0, 0.96, -0.04, 0.14)
        if kind == 1:
            return frog_mask(x, y, 1, 0.92, -0.06, 0.16)
        if kind == 2:
            return frog_mask(x, y, 2, 0.92, -0.02, 0.16)
        return frog_mask(x, y, 3, 0.86, -0.02, 0.18)
    if cls == 7:
        if kind == 0:
            return horse_mask(x, y, 0, 0.94, -0.04, 0.18)
        if kind == 1:
            return horse_mask(x, y, 1, 0.90, -0.06, 0.18)
        if kind == 2:
            return horse_mask(x, y, 2, 0.92, -0.06, 0.16)
        return horse_mask(x, y, 3, 0.82, -0.02, 0.18)
    if cls == 8:
        if kind == 0:
            return ship_mask(x, y, 0, 0.96, 0.00, 0.26)
        if kind == 1:
            return ship_mask(x, y, 1, 0.94, 0.00, 0.24)
        if kind == 2:
            return ship_mask(x, y, 2, 0.96, 0.00, 0.26)
        return ship_mask(x, y, 3, 0.88, 0.00, 0.24)
    if cls == 9:
        if kind == 0:
            return car_mask(x, y, 0.98, 0.00, -0.02, 0.20, front=False, truck=True)
        if kind == 1:
            return car_mask(x, y, 0.94, -0.02, 0.00, 0.20, front=True, truck=True)
        if kind == 2:
            return car_mask(x, y, 0.86, 0.00, 0.00, 0.18, front=False, truck=True)
        return car_mask(x, y, 0.82, 0.00, 0.02, 0.18, front=False, truck=True)
    raise ValueError(cls)


def scene_object_mask(cls, x, y):
    if cls == 0:
        return airplane_mask(x, y, 0.70, 0.0, 0.00, 0.00)
    if cls == 1:
        return car_mask(x, y, 0.72, 0.0, -0.02, 0.18, front=False, truck=False)
    if cls == 2:
        return bird_mask(x, y, 0, 0.72, 0.00, 0.06)
    if cls == 3:
        return cat_mask(x, y, 0, 0.74, 0.00, -0.02)
    if cls == 4:
        return deer_mask(x, y, 1, 0.72, -0.04, 0.18)
    if cls == 5:
        return dog_mask(x, y, 0, 0.74, 0.00, 0.02)
    if cls == 6:
        return frog_mask(x, y, 0, 0.72, -0.04, 0.16)
    if cls == 7:
        return horse_mask(x, y, 0, 0.76, -0.02, 0.18)
    if cls == 8:
        return ship_mask(x, y, 0, 0.80, 0.00, 0.28)
    if cls == 9:
        return car_mask(x, y, 0.80, 0.0, -0.02, 0.20, front=False, truck=True)
    raise ValueError(cls)


def make_scene_weights():
    x, y = grid(8)
    top = top_band(y, 0.0, 0.20)
    bottom = bottom_band(y, 0.0, 0.20)
    mid = ellipse(x, y, 0.0, 0.05, 0.75, 0.45, 0.0, 0.25)
    scene = np.zeros((CLASSES, SCENE_CH, 8, 8), np.float32)
    for cls in range(CLASSES):
        obj = scene_object_mask(cls, x, y)
        if cls == 0:
            scene[cls, 0] += 0.95 * top + 0.35 * mid
            scene[cls, 4] += 0.45 * obj
            scene[cls, 5] += 0.12 * obj
            scene[cls, 1] -= 0.25 * bottom
        elif cls == 1:
            scene[cls, 5] += 0.35 * bottom
            scene[cls, 2] += 0.38 * obj
            scene[cls, 3] += 0.22 * obj
            scene[cls, 4] += 0.10 * obj
            scene[cls, 0] -= 0.10 * top
        elif cls == 2:
            scene[cls, 1] += 0.18 * mid
            scene[cls, 0] += 0.12 * top
            scene[cls, 2] += 0.14 * obj
        elif cls == 3:
            scene[cls, 2] += 0.18 * mid
            scene[cls, 4] += 0.10 * obj
            scene[cls, 5] += 0.20 * obj
            scene[cls, 0] -= 0.15 * top
        elif cls == 4:
            scene[cls, 1] += 0.28 * mid + 0.10 * bottom
            scene[cls, 2] += 0.26 * obj
            scene[cls, 0] += 0.06 * top
        elif cls == 5:
            scene[cls, 1] += 0.08 * bottom
            scene[cls, 2] += 0.16 * mid + 0.12 * obj
            scene[cls, 5] += 0.18 * obj
            scene[cls, 0] -= 0.08 * top
        elif cls == 6:
            scene[cls, 1] += 0.34 * mid + 0.10 * bottom
            scene[cls, 2] += 0.08 * obj
            scene[cls, 5] += 0.12 * obj
        elif cls == 7:
            scene[cls, 1] += 0.28 * bottom + 0.08 * mid
            scene[cls, 2] += 0.30 * obj
            scene[cls, 0] += 0.05 * top
        elif cls == 8:
            scene[cls, 0] += 0.40 * top + 0.60 * bottom
            scene[cls, 5] += 0.16 * obj
            scene[cls, 4] += 0.12 * obj
            scene[cls, 1] -= 0.22 * mid
        elif cls == 9:
            scene[cls, 5] += 0.46 * bottom
            scene[cls, 4] += 0.22 * obj
            scene[cls, 2] += 0.14 * obj
            scene[cls, 3] += 0.08 * obj
            scene[cls, 0] -= 0.08 * top
    W = np.zeros((CLASSES, SCENE_CH * 8 * 8), np.float32)
    for cls in range(CLASSES):
        w = scene[cls].reshape(-1)
        n = float(np.linalg.norm(w))
        if n > 1e-6:
            w = w / n
        W[cls] = w * SCENE_ALPHA
    b = np.array([-0.22, -0.18, -0.12, -0.10, -0.10, -0.10, -0.08, -0.10, -0.22, -0.18], np.float32)
    return W, b


def shape_filter(mask, bright=True):
    inside = mask.astype(np.float32)
    outside = 1.0 - inside
    W = np.zeros((2, SHAPE_K, SHAPE_K), np.float32)
    if bright:
        W[0] = 1.20 * inside - 0.10 * outside
        W[1] = -0.68 * inside - 0.03 * outside
    else:
        W[0] = -0.68 * inside - 0.03 * outside
        W[1] = 1.20 * inside - 0.10 * outside
    n = float(np.linalg.norm(W.ravel()))
    if n > 1e-6:
        W /= n
    return (W * SHAPE_ALPHA).astype(np.float32)


def make_shape_weights():
    x, y = grid(SHAPE_K)
    W = np.zeros((SHAPE_OUT, 2, SHAPE_K, SHAPE_K), np.float32)
    b = np.full((SHAPE_OUT,), -0.56, np.float32)
    k = 0
    for cls in range(CLASSES):
        for kind in range(MASKS_PER_CLASS):
            m = shape_mask(cls, kind, x, y)
            W[k] = shape_filter(m, True)
            k += 1
            W[k] = shape_filter(m, False)
            k += 1
    return W, b


def build_network():
    return {
        "layers": [
            {"name": "sem", "op": "conv", "inputs": ["image"], "out": SCENE_CH, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "sem_relu", "op": "relu", "inputs": ["sem"]},
            {"name": "sem_pool", "op": "avgpool", "inputs": ["sem_relu"], "size": [4, 4], "stride": [4, 4]},
            {"name": "scene_scores", "op": "dense", "inputs": ["sem_pool"], "out": CLASSES},
            {"name": "gray", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "gray_norm", "op": "norm", "inputs": ["gray"]},
            {"name": "split", "op": "conv", "inputs": ["gray_norm"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "split_relu", "op": "relu", "inputs": ["split"]},
            {"name": "shape_pool", "op": "avgpool", "inputs": ["split_relu"], "size": [2, 2], "stride": [2, 2]},
            {"name": "shape_conv", "op": "conv", "inputs": ["shape_pool"], "out": SHAPE_OUT, "kernel": [SHAPE_K, SHAPE_K], "stride": 1, "padding": 1},
            {"name": "shape_relu", "op": "relu", "inputs": ["shape_conv"]},
            {"name": "shape_pos", "op": "maxpool", "inputs": ["shape_relu"], "size": [10, 10], "stride": [10, 10]},
            {"name": "shape_grid", "op": "reshape", "inputs": ["shape_pos"], "shape": [CLASSES, 1, MASKS_PER_CLASS * POLARITIES]},
            {"name": "shape_max", "op": "maxpool", "inputs": ["shape_grid"], "size": [1, MASKS_PER_CLASS * POLARITIES], "stride": [1, MASKS_PER_CLASS * POLARITIES]},
            {"name": "shape_best", "op": "reshape", "inputs": ["shape_max"], "shape": [CLASSES]},
            {"name": "shape_aff", "op": "dense", "inputs": ["shape_best"], "out": CLASSES},
            {"name": "scores", "op": "add", "inputs": ["scene_scores", "shape_aff"]},
            {"name": "out", "op": "dense", "inputs": ["scores"], "out": CLASSES},
        ]
    }


def build_weights():
    weights = {}

    sem_W = np.zeros((SCENE_CH, 3, 1, 1), np.float32)
    sem_b = np.zeros((SCENE_CH,), np.float32)
    sem_W[0, :, 0, 0] = (-0.35, -0.10, 1.00)
    sem_b[0] = -0.28
    sem_W[1, :, 0, 0] = (-0.30, 1.00, -0.25)
    sem_b[1] = -0.26
    sem_W[2, :, 0, 0] = (0.74, 0.34, -0.72)
    sem_b[2] = -0.16
    sem_W[3, :, 0, 0] = (1.00, -0.46, -0.46)
    sem_b[3] = -0.20
    sem_W[4, :, 0, 0] = (0.34, 0.34, 0.34)
    sem_b[4] = -0.70
    sem_W[5, :, 0, 0] = (-0.34, -0.34, -0.34)
    sem_b[5] = 0.44
    weights["sem.W"] = sem_W
    weights["sem.b"] = sem_b

    scene_W, scene_b = make_scene_weights()
    weights["scene_scores.W"] = scene_W
    weights["scene_scores.b"] = scene_b

    gray_W = np.zeros((1, 3, 1, 1), np.float32)
    gray_W[0, :, 0, 0] = (0.33, 0.34, 0.33)
    weights["gray.W"] = gray_W
    weights["gray.b"] = np.zeros((1,), np.float32)

    split_W = np.zeros((2, 1, 1, 1), np.float32)
    split_W[0, 0, 0, 0] = 1.0
    split_W[1, 0, 0, 0] = -1.0
    weights["split.W"] = split_W
    weights["split.b"] = np.zeros((2,), np.float32)

    shape_W, shape_b = make_shape_weights()
    weights["shape_conv.W"] = shape_W
    weights["shape_conv.b"] = shape_b

    shape_scale = np.array([1.30, 1.12, 1.10, 1.18, 1.18, 1.18, 1.22, 1.18, 1.14, 1.16], np.float32)
    weights["shape_aff.W"] = np.diag(shape_scale).astype(np.float32)
    weights["shape_aff.b"] = np.array([-0.02, -0.04, -0.06, -0.06, -0.06, -0.06, -0.04, -0.06, -0.02, -0.04], np.float32)

    out_W = np.diag(np.array([1.08, 1.00, 1.08, 1.18, 1.18, 1.12, 1.12, 1.12, 1.08, 0.58], np.float32)).astype(np.float32)
    weights["out.W"] = out_W
    weights["out.b"] = np.array([0.02, 0.04, 0.06, 0.10, 0.10, 0.08, 0.08, 0.08, 0.04, -0.36], np.float32)

    return weights


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = build_network()
    weights = build_weights()
    (out / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(out / "weights.npz", **weights)


if __name__ == "__main__":
    main(sys.argv[1])
