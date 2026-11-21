import json
import os
import sys
import numpy as np

AIRPLANE, AUTOMOBILE, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK = range(10)
SKY, WATER, GREEN, BROWN, TAN, DARK, LIGHT, WHITE, RED, YELLOW, COOL, FUR = range(12)


def coords(k):
    xs = np.linspace(-1.0, 1.0, k, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, k, dtype=np.float32)
    return np.meshgrid(xs, ys)


def clip01(x):
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def blob(X, Y, cx, cy, rx, ry):
    z = 1.0 - ((X - cx) / rx) ** 2 - ((Y - cy) / ry) ** 2
    return clip01(z)


def box(X, Y, x0, x1, y0, y1, edge=0.18):
    mx = np.minimum((X - x0) / edge, (x1 - X) / edge)
    my = np.minimum((Y - y0) / edge, (y1 - Y) / edge)
    return clip01(np.minimum(mx, my))


def line(X, Y, x0, y0, x1, y1, width):
    vx = x1 - x0
    vy = y1 - y0
    px = X - x0
    py = Y - y0
    denom = vx * vx + vy * vy + 1e-6
    t = np.clip((px * vx + py * vy) / denom, 0.0, 1.0)
    dx = x0 + t * vx - X
    dy = y0 + t * vy - Y
    d = np.sqrt(dx * dx + dy * dy)
    return clip01(1.0 - d / width)


def flatten_mask(mask):
    return mask.astype(np.float32).reshape(-1)


def add_dense(W, cls, ch, mask, scale):
    k = mask.shape[0]
    start = ch * k * k
    W[cls, start:start + k * k] += scale * flatten_mask(mask)


def add_template(W, t, ch, mask, scale):
    W[t, ch] += scale * mask.astype(np.float32)


def make_semantic_weights():
    W = np.zeros((12, 3, 1, 1), np.float32)
    b = np.zeros(12, np.float32)
    W[SKY, :, 0, 0] = [-0.90, 0.15, 1.05]; b[SKY] = -0.16
    W[WATER, :, 0, 0] = [-0.55, 0.05, 0.90]; b[WATER] = -0.12
    W[GREEN, :, 0, 0] = [-0.65, 1.05, -0.45]; b[GREEN] = -0.06
    W[BROWN, :, 0, 0] = [0.95, 0.45, -0.35]; b[BROWN] = -0.48
    W[TAN, :, 0, 0] = [0.70, 0.60, -0.45]; b[TAN] = -0.56
    W[DARK, :, 0, 0] = [-0.3333, -0.3333, -0.3333]; b[DARK] = 0.38
    W[LIGHT, :, 0, 0] = [0.3333, 0.3333, 0.3333]; b[LIGHT] = -0.58
    W[WHITE, :, 0, 0] = [0.3333, 0.3333, 0.3333]; b[WHITE] = -0.78
    W[RED, :, 0, 0] = [1.00, -0.75, -0.75]; b[RED] = -0.04
    W[YELLOW, :, 0, 0] = [0.75, 0.75, -0.95]; b[YELLOW] = -0.30
    W[COOL, :, 0, 0] = [-0.10, 0.35, 0.80]; b[COOL] = -0.42
    W[FUR, :, 0, 0] = [0.85, 0.35, -0.55]; b[FUR] = -0.16
    return W, b


def make_color_readout():
    k = 8
    X, Y = coords(k)
    W = np.zeros((10, 12 * k * k), np.float32)
    b = np.zeros(10, np.float32)

    whole = np.ones((k, k), np.float32)
    top = box(X, Y, -1.0, 1.0, -1.0, -0.12)
    bottom = box(X, Y, -1.0, 1.0, 0.12, 1.0)
    center = blob(X, Y, 0.0, 0.02, 0.66, 0.56)
    face = blob(X, Y, 0.0, 0.02, 0.62, 0.58)
    wide_low = blob(X, Y, 0.0, 0.24, 0.96, 0.30)
    tall_low = blob(X, Y, 0.0, 0.18, 0.56, 0.62)
    small_mid = blob(X, Y, 0.0, -0.04, 0.32, 0.24)
    deer_body = blob(X, Y, -0.06, 0.12, 0.46, 0.24)
    horse_body = blob(X, Y, -0.04, 0.10, 0.62, 0.28)
    head_small = blob(X, Y, 0.48, -0.02, 0.18, 0.16)
    wheel_l = blob(X, Y, -0.42, 0.58, 0.16, 0.14)
    wheel_r = blob(X, Y, 0.42, 0.58, 0.16, 0.14)
    wheel_c = blob(X, Y, 0.00, 0.58, 0.14, 0.12)
    ship_band = box(X, Y, -0.75, 0.75, 0.06, 0.32)
    cat_ears = blob(X, Y, -0.34, -0.46, 0.14, 0.18) + blob(X, Y, 0.34, -0.46, 0.14, 0.18)
    eye_pair = blob(X, Y, -0.20, -0.06, 0.09, 0.07) + blob(X, Y, 0.20, -0.06, 0.09, 0.07)
    dog_muzzle = blob(X, Y, 0.0, 0.16, 0.22, 0.18)
    frog_eyes = blob(X, Y, -0.28, -0.02, 0.11, 0.09) + blob(X, Y, 0.28, -0.02, 0.11, 0.09)
    bird_bg = clip01(0.5 * top + 0.7 * whole)

    # Context branch only: small weights, broad priors.
    add_dense(W, AIRPLANE, SKY, whole, 0.26)
    add_dense(W, AIRPLANE, SKY, top, 0.18)
    add_dense(W, AIRPLANE, LIGHT, small_mid, 0.18)
    add_dense(W, AIRPLANE, WHITE, small_mid, 0.18)
    add_dense(W, AIRPLANE, GREEN, whole, -0.20)
    add_dense(W, AIRPLANE, WATER, bottom, -0.18)
    b[AIRPLANE] = -0.06

    add_dense(W, AUTOMOBILE, DARK, bottom, 0.20)
    add_dense(W, AUTOMOBILE, LIGHT, wide_low, 0.16)
    add_dense(W, AUTOMOBILE, COOL, wide_low, 0.18)
    add_dense(W, AUTOMOBILE, RED, wide_low, 0.10)
    add_dense(W, AUTOMOBILE, YELLOW, wide_low, 0.06)
    add_dense(W, AUTOMOBILE, DARK, wheel_l + wheel_r, 0.16)
    add_dense(W, AUTOMOBILE, GREEN, whole, -0.14)
    add_dense(W, AUTOMOBILE, WATER, whole, -0.08)

    add_dense(W, BIRD, GREEN, whole, 0.10)
    add_dense(W, BIRD, SKY, top, 0.12)
    add_dense(W, BIRD, BROWN, small_mid, 0.20)
    add_dense(W, BIRD, DARK, small_mid, 0.18)
    add_dense(W, BIRD, LIGHT, small_mid, 0.06)
    add_dense(W, BIRD, WATER, whole, -0.08)
    b[BIRD] = 0.06

    add_dense(W, CAT, FUR, face, 0.18)
    add_dense(W, CAT, TAN, face, 0.14)
    add_dense(W, CAT, BROWN, face, 0.06)
    add_dense(W, CAT, DARK, cat_ears, 0.12)
    add_dense(W, CAT, DARK, eye_pair, 0.12)
    add_dense(W, CAT, GREEN, whole, -0.14)
    add_dense(W, CAT, SKY, whole, -0.14)
    add_dense(W, CAT, WATER, whole, -0.08)
    b[CAT] = 0.04

    add_dense(W, DEER, GREEN, whole, 0.14)
    add_dense(W, DEER, BROWN, deer_body, 0.18)
    add_dense(W, DEER, TAN, deer_body, 0.14)
    add_dense(W, DEER, DARK, bottom, 0.06)
    add_dense(W, DEER, BROWN, head_small, 0.06)
    add_dense(W, DEER, WATER, whole, -0.06)
    b[DEER] = 0.02

    add_dense(W, DOG, FUR, face, 0.18)
    add_dense(W, DOG, TAN, center, 0.10)
    add_dense(W, DOG, BROWN, center, 0.08)
    add_dense(W, DOG, DARK, dog_muzzle, 0.14)
    add_dense(W, DOG, GREEN, bottom, 0.06)
    add_dense(W, DOG, WATER, whole, -0.08)
    b[DOG] = 0.03

    add_dense(W, FROG, GREEN, whole, 0.16)
    add_dense(W, FROG, GREEN, center, 0.16)
    add_dense(W, FROG, DARK, frog_eyes, 0.10)
    add_dense(W, FROG, BROWN, center, 0.04)
    add_dense(W, FROG, SKY, whole, -0.16)
    add_dense(W, FROG, WATER, whole, -0.10)
    b[FROG] = 0.04

    add_dense(W, HORSE, BROWN, horse_body, 0.20)
    add_dense(W, HORSE, TAN, horse_body, 0.10)
    add_dense(W, HORSE, GREEN, bottom, 0.10)
    add_dense(W, HORSE, SKY, top, 0.06)
    add_dense(W, HORSE, BROWN, head_small, 0.06)
    add_dense(W, HORSE, WATER, whole, -0.04)
    b[HORSE] = 0.02

    add_dense(W, SHIP, SKY, top, 0.24)
    add_dense(W, SHIP, WATER, bottom, 0.30)
    add_dense(W, SHIP, DARK, ship_band, 0.10)
    add_dense(W, SHIP, LIGHT, ship_band, 0.10)
    add_dense(W, SHIP, COOL, ship_band, 0.12)
    add_dense(W, SHIP, GREEN, whole, -0.22)
    add_dense(W, SHIP, BROWN, whole, -0.10)
    b[SHIP] = 0.10

    add_dense(W, TRUCK, DARK, bottom, 0.22)
    add_dense(W, TRUCK, LIGHT, tall_low, 0.18)
    add_dense(W, TRUCK, COOL, tall_low, 0.22)
    add_dense(W, TRUCK, RED, tall_low, 0.06)
    add_dense(W, TRUCK, YELLOW, tall_low, 0.06)
    add_dense(W, TRUCK, DARK, wheel_l + wheel_r + wheel_c, 0.18)
    add_dense(W, TRUCK, GREEN, whole, -0.16)
    add_dense(W, TRUCK, WATER, whole, -0.10)
    add_dense(W, TRUCK, SKY, top, -0.04)
    b[TRUCK] = -0.06

    W *= 0.55
    b *= 0.55
    return W, b


def make_small_template_bank():
    k = 9
    X, Y = coords(k)
    T = 14
    W = np.zeros((T, 12, k, k), np.float32)
    b = np.zeros(T, np.float32)

    whole = np.ones((k, k), np.float32)
    center = blob(X, Y, 0.0, 0.04, 0.68, 0.58)
    surround = clip01(whole - 0.85 * center)
    top = box(X, Y, -1.0, 1.0, -1.0, -0.12)
    bottom = box(X, Y, -1.0, 1.0, 0.12, 1.0)

    plane_wing = line(X, Y, -0.82, 0.0, 0.82, 0.0, 0.18)
    plane_body = line(X, Y, 0.0, -0.60, 0.0, 0.60, 0.10)
    plane_diag_a = line(X, Y, -0.82, 0.26, 0.82, -0.26, 0.18)
    plane_diag_b = line(X, Y, -0.18, 0.72, 0.18, -0.72, 0.10)
    plane_diag_c = line(X, Y, -0.82, -0.26, 0.82, 0.26, 0.18)
    plane_diag_d = line(X, Y, -0.18, -0.72, 0.18, 0.72, 0.10)

    ship_hull = box(X, Y, -0.78, 0.78, 0.08, 0.34)
    ship_close = box(X, Y, -0.88, 0.88, -0.02, 0.34)
    mast = line(X, Y, 0.0, -0.62, 0.0, 0.10, 0.08)
    sail = line(X, Y, 0.04, -0.46, 0.04, 0.08, 0.09) + line(X, Y, -0.26, -0.10, 0.24, -0.40, 0.11)

    car_body = box(X, Y, -0.66, 0.66, 0.04, 0.34)
    car_roof = box(X, Y, -0.28, 0.28, -0.22, 0.06)
    truck_body = box(X, Y, -0.66, 0.40, 0.02, 0.36)
    truck_cab = box(X, Y, 0.18, 0.76, -0.18, 0.20)
    tall_box = box(X, Y, -0.36, 0.36, -0.14, 0.44)
    grille = box(X, Y, -0.42, 0.42, 0.18, 0.54)

    wheel_l = blob(X, Y, -0.45, 0.58, 0.16, 0.13)
    wheel_r = blob(X, Y, 0.45, 0.58, 0.16, 0.13)
    wheel_c = blob(X, Y, 0.02, 0.58, 0.14, 0.12)
    underbody = box(X, Y, -0.82, 0.82, 0.22, 0.46)

    bird_body = blob(X, Y, 0.0, -0.06, 0.24, 0.18)
    bird_head = blob(X, Y, 0.24, -0.10, 0.10, 0.09)
    perch = line(X, Y, -0.74, 0.18, 0.74, 0.14, 0.08)
    wing_v = line(X, Y, -0.50, 0.24, 0.0, -0.10, 0.10) + line(X, Y, 0.0, -0.10, 0.50, 0.24, 0.10)
    bird_big = blob(X, Y, 0.0, 0.00, 0.34, 0.26)

    frog_body = blob(X, Y, 0.0, 0.18, 0.60, 0.28)
    frog_eyes = blob(X, Y, -0.28, -0.02, 0.10, 0.08) + blob(X, Y, 0.28, -0.02, 0.10, 0.08)

    # 0-2: airplane
    add_template(W, 0, SKY, surround, 0.18)
    add_template(W, 0, SKY, plane_wing + plane_body, -0.34)
    add_template(W, 0, LIGHT, plane_wing + plane_body, 0.40)
    add_template(W, 0, WHITE, plane_wing + plane_body, 0.38)
    add_template(W, 0, COOL, plane_body, 0.12)
    add_template(W, 0, GREEN, whole, -0.06)
    add_template(W, 0, BROWN, whole, -0.04)

    add_template(W, 1, SKY, surround, 0.18)
    add_template(W, 1, SKY, plane_diag_a + plane_diag_b, -0.34)
    add_template(W, 1, LIGHT, plane_diag_a + plane_diag_b, 0.40)
    add_template(W, 1, WHITE, plane_diag_a + plane_diag_b, 0.36)
    add_template(W, 1, COOL, plane_diag_b, 0.12)
    add_template(W, 1, GREEN, whole, -0.06)

    add_template(W, 2, SKY, surround, 0.18)
    add_template(W, 2, SKY, plane_diag_c + plane_diag_d, -0.34)
    add_template(W, 2, LIGHT, plane_diag_c + plane_diag_d, 0.40)
    add_template(W, 2, WHITE, plane_diag_c + plane_diag_d, 0.36)
    add_template(W, 2, COOL, plane_diag_d, 0.12)
    add_template(W, 2, GREEN, whole, -0.06)

    # 3-4: ship
    add_template(W, 3, SKY, top, 0.26)
    add_template(W, 3, WATER, bottom, 0.30)
    add_template(W, 3, SKY, ship_hull, -0.20)
    add_template(W, 3, WATER, ship_hull, -0.22)
    add_template(W, 3, DARK, ship_hull, 0.24)
    add_template(W, 3, LIGHT, ship_hull, 0.22)
    add_template(W, 3, COOL, ship_hull, 0.20)
    add_template(W, 3, GREEN, whole, -0.12)

    add_template(W, 4, SKY, top, 0.22)
    add_template(W, 4, WATER, bottom, 0.28)
    add_template(W, 4, WHITE, sail, 0.34)
    add_template(W, 4, LIGHT, sail, 0.24)
    add_template(W, 4, COOL, mast, 0.12)
    add_template(W, 4, DARK, ship_hull, 0.18)
    add_template(W, 4, GREEN, whole, -0.10)

    # 5-6: side vehicles
    add_template(W, 5, LIGHT, car_body + car_roof, 0.22)
    add_template(W, 5, COOL, car_body + car_roof, 0.24)
    add_template(W, 5, RED, car_body, 0.10)
    add_template(W, 5, YELLOW, car_body, 0.06)
    add_template(W, 5, DARK, wheel_l + wheel_r + underbody, 0.32)
    add_template(W, 5, GREEN, whole, -0.06)
    add_template(W, 5, WATER, whole, -0.04)

    add_template(W, 6, LIGHT, truck_body + truck_cab, 0.22)
    add_template(W, 6, COOL, truck_body + truck_cab, 0.28)
    add_template(W, 6, RED, truck_cab, 0.06)
    add_template(W, 6, DARK, wheel_l + wheel_c + wheel_r + underbody, 0.34)
    add_template(W, 6, GREEN, whole, -0.06)
    add_template(W, 6, WATER, whole, -0.04)

    # 7-8: birds
    add_template(W, 7, GREEN, surround, 0.10)
    add_template(W, 7, SKY, top, 0.08)
    add_template(W, 7, BROWN, bird_body + bird_head, 0.26)
    add_template(W, 7, DARK, bird_body + bird_head, 0.22)
    add_template(W, 7, LIGHT, bird_body, 0.08)
    add_template(W, 7, BROWN, perch, 0.14)
    add_template(W, 7, WATER, whole, -0.06)

    add_template(W, 8, SKY, surround, 0.14)
    add_template(W, 8, DARK, wing_v, 0.26)
    add_template(W, 8, LIGHT, wing_v, 0.14)
    add_template(W, 8, WHITE, wing_v, 0.10)
    add_template(W, 8, GREEN, whole, -0.04)

    # 9: front vehicle
    add_template(W, 9, LIGHT, tall_box, 0.24)
    add_template(W, 9, COOL, tall_box, 0.24)
    add_template(W, 9, DARK, grille, 0.26)
    add_template(W, 9, DARK, wheel_l + wheel_r, 0.14)
    add_template(W, 9, GREEN, whole, -0.06)
    add_template(W, 9, WATER, whole, -0.04)

    # 10: frog
    add_template(W, 10, GREEN, frog_body + 0.35 * surround, 0.22)
    add_template(W, 10, DARK, frog_eyes, 0.24)
    add_template(W, 10, BROWN, frog_body, 0.08)
    add_template(W, 10, SKY, whole, -0.16)
    add_template(W, 10, WATER, whole, -0.08)

    # 11: close ship / boat body
    add_template(W, 11, SKY, top, 0.16)
    add_template(W, 11, WATER, bottom, 0.20)
    add_template(W, 11, DARK, ship_close, 0.18)
    add_template(W, 11, LIGHT, ship_close, 0.22)
    add_template(W, 11, COOL, ship_close, 0.18)
    add_template(W, 11, GREEN, whole, -0.08)

    # 12: generic vehicle with wheels
    add_template(W, 12, DARK, wheel_l + wheel_r + underbody, 0.28)
    add_template(W, 12, LIGHT, car_body + car_roof, 0.10)
    add_template(W, 12, COOL, car_body + truck_cab, 0.14)
    add_template(W, 12, GREEN, whole, -0.04)

    # 13: larger small bird / colorful central bird
    add_template(W, 13, GREEN, surround, 0.10)
    add_template(W, 13, SKY, top, 0.06)
    add_template(W, 13, BROWN, bird_big, 0.22)
    add_template(W, 13, TAN, bird_big, 0.10)
    add_template(W, 13, DARK, bird_big, 0.16)
    add_template(W, 13, WATER, whole, -0.04)

    return W, b


def make_large_template_bank():
    k = 17
    X, Y = coords(k)
    T = 12
    W = np.zeros((T, 12, k, k), np.float32)
    b = np.zeros(T, np.float32)

    whole = np.ones((k, k), np.float32)
    center = blob(X, Y, 0.0, 0.04, 0.72, 0.60)
    surround = clip01(whole - 0.90 * center)
    top = box(X, Y, -1.0, 1.0, -1.0, -0.12)
    bottom = box(X, Y, -1.0, 1.0, 0.12, 1.0)

    cat_face = blob(X, Y, 0.0, 0.04, 0.66, 0.62)
    cat_ears = blob(X, Y, -0.36, -0.54, 0.16, 0.20) + blob(X, Y, 0.36, -0.54, 0.16, 0.20)
    cat_eyes = blob(X, Y, -0.22, -0.08, 0.08, 0.06) + blob(X, Y, 0.22, -0.08, 0.08, 0.06)

    dog_face = blob(X, Y, 0.0, 0.10, 0.74, 0.58)
    dog_ears = blob(X, Y, -0.56, -0.06, 0.18, 0.28) + blob(X, Y, 0.56, -0.06, 0.18, 0.28)
    dog_nose = blob(X, Y, 0.0, 0.18, 0.12, 0.10)

    deer_body = blob(X, Y, -0.06, 0.06, 0.44, 0.22)
    horse_body = blob(X, Y, -0.02, 0.05, 0.60, 0.26)
    head = blob(X, Y, 0.52, -0.02, 0.18, 0.14)
    neck = line(X, Y, 0.24, 0.00, 0.46, -0.10, 0.08)
    legs = (
        line(X, Y, -0.34, 0.16, -0.34, 0.70, 0.05)
        + line(X, Y, -0.10, 0.16, -0.10, 0.72, 0.05)
        + line(X, Y, 0.15, 0.16, 0.15, 0.72, 0.05)
        + line(X, Y, 0.40, 0.12, 0.40, 0.68, 0.05)
    )

    frog_body = blob(X, Y, 0.0, 0.16, 0.58, 0.28)
    frog_eyes = blob(X, Y, -0.28, -0.02, 0.10, 0.08) + blob(X, Y, 0.28, -0.02, 0.10, 0.08)
    furry_close = blob(X, Y, 0.0, 0.06, 0.84, 0.76)
    generic_quad = horse_body + head + 0.8 * legs

    truck_front = box(X, Y, -0.44, 0.44, -0.12, 0.50)
    truck_grille = box(X, Y, -0.48, 0.48, 0.20, 0.56)
    car_front = box(X, Y, -0.36, 0.36, -0.12, 0.44)
    car_grille = box(X, Y, -0.40, 0.40, 0.20, 0.50)
    wheel_pair = blob(X, Y, -0.28, 0.54, 0.12, 0.10) + blob(X, Y, 0.28, 0.54, 0.12, 0.10)

    ship_hull = box(X, Y, -0.76, 0.76, 0.06, 0.32)
    plane_wing = line(X, Y, -0.82, 0.0, 0.82, 0.0, 0.12)
    plane_body = line(X, Y, 0.0, -0.64, 0.0, 0.64, 0.08)

    bird_body = blob(X, Y, 0.0, -0.02, 0.28, 0.20)
    bird_head = blob(X, Y, 0.24, -0.08, 0.12, 0.10)
    perch = line(X, Y, -0.72, 0.18, 0.72, 0.14, 0.07)

    # 0: cat face
    add_template(W, 0, FUR, cat_face, 0.24)
    add_template(W, 0, TAN, cat_face, 0.18)
    add_template(W, 0, BROWN, cat_face, 0.08)
    add_template(W, 0, DARK, cat_ears + cat_eyes, 0.26)
    add_template(W, 0, GREEN, whole, -0.10)
    add_template(W, 0, SKY, whole, -0.10)

    # 1: dog face
    add_template(W, 1, FUR, dog_face, 0.24)
    add_template(W, 1, TAN, dog_face, 0.12)
    add_template(W, 1, BROWN, dog_face, 0.10)
    add_template(W, 1, DARK, dog_ears + dog_nose, 0.24)
    add_template(W, 1, GREEN, whole, -0.08)
    add_template(W, 1, SKY, whole, -0.06)

    # 2: deer side
    add_template(W, 2, GREEN, bottom, 0.12)
    add_template(W, 2, BROWN, deer_body + head + neck, 0.24)
    add_template(W, 2, TAN, deer_body, 0.14)
    add_template(W, 2, DARK, legs, 0.18)
    add_template(W, 2, WATER, whole, -0.06)

    # 3: horse side
    add_template(W, 3, GREEN, bottom, 0.10)
    add_template(W, 3, SKY, top, 0.08)
    add_template(W, 3, BROWN, horse_body + head + neck, 0.26)
    add_template(W, 3, TAN, horse_body, 0.12)
    add_template(W, 3, DARK, legs, 0.16)
    add_template(W, 3, WATER, whole, -0.04)

    # 4: large frog
    add_template(W, 4, GREEN, frog_body + 0.30 * surround, 0.24)
    add_template(W, 4, DARK, frog_eyes, 0.24)
    add_template(W, 4, BROWN, frog_body, 0.08)
    add_template(W, 4, SKY, whole, -0.14)
    add_template(W, 4, WATER, whole, -0.08)

    # 5: furry close-up
    add_template(W, 5, FUR, furry_close, 0.22)
    add_template(W, 5, TAN, furry_close, 0.14)
    add_template(W, 5, DARK, cat_eyes + dog_nose, 0.10)
    add_template(W, 5, GREEN, whole, -0.08)
    add_template(W, 5, SKY, whole, -0.08)

    # 6: generic quadruped
    add_template(W, 6, BROWN, generic_quad, 0.22)
    add_template(W, 6, TAN, generic_quad, 0.10)
    add_template(W, 6, DARK, legs, 0.16)
    add_template(W, 6, GREEN, bottom, 0.08)
    add_template(W, 6, WATER, whole, -0.04)

    # 7: truck front
    add_template(W, 7, LIGHT, truck_front, 0.22)
    add_template(W, 7, COOL, truck_front, 0.26)
    add_template(W, 7, DARK, truck_grille + wheel_pair, 0.26)
    add_template(W, 7, GREEN, whole, -0.08)
    add_template(W, 7, WATER, whole, -0.04)

    # 8: car front
    add_template(W, 8, LIGHT, car_front, 0.22)
    add_template(W, 8, COOL, car_front, 0.20)
    add_template(W, 8, RED, car_front, 0.06)
    add_template(W, 8, DARK, car_grille + wheel_pair, 0.22)
    add_template(W, 8, WATER, whole, -0.04)

    # 9: large ship
    add_template(W, 9, SKY, top, 0.24)
    add_template(W, 9, WATER, bottom, 0.30)
    add_template(W, 9, SKY, ship_hull, -0.18)
    add_template(W, 9, WATER, ship_hull, -0.20)
    add_template(W, 9, DARK, ship_hull, 0.20)
    add_template(W, 9, LIGHT, ship_hull, 0.20)
    add_template(W, 9, COOL, ship_hull, 0.18)
    add_template(W, 9, GREEN, whole, -0.10)

    # 10: large airplane
    add_template(W, 10, SKY, surround, 0.16)
    add_template(W, 10, SKY, plane_wing + plane_body, -0.26)
    add_template(W, 10, LIGHT, plane_wing + plane_body, 0.30)
    add_template(W, 10, WHITE, plane_wing + plane_body, 0.28)
    add_template(W, 10, COOL, plane_body, 0.10)
    add_template(W, 10, GREEN, whole, -0.08)

    # 11: large bird / perch
    add_template(W, 11, GREEN, surround, 0.10)
    add_template(W, 11, SKY, top, 0.06)
    add_template(W, 11, BROWN, bird_body + bird_head, 0.22)
    add_template(W, 11, DARK, bird_body + bird_head, 0.18)
    add_template(W, 11, BROWN, perch, 0.12)
    add_template(W, 11, WATER, whole, -0.04)

    return W, b


def make_small_readout(num_templates=14):
    W = np.zeros((10, num_templates), np.float32)
    b = np.zeros(10, np.float32)
    baseline = 1.9

    def link(cls, t, w):
        W[cls, t] += w
        if w > 0:
            b[cls] -= baseline * w

    # class links
    link(AIRPLANE, 0, 1.00)
    link(AIRPLANE, 1, 0.92)
    link(AIRPLANE, 2, 0.92)

    link(AUTOMOBILE, 5, 0.86)
    link(AUTOMOBILE, 9, 0.30)
    link(AUTOMOBILE, 12, 0.44)

    link(BIRD, 7, 0.92)
    link(BIRD, 8, 0.72)
    link(BIRD, 13, 0.78)

    link(FROG, 10, 1.04)

    link(SHIP, 3, 0.96)
    link(SHIP, 4, 0.74)
    link(SHIP, 11, 0.56)

    link(TRUCK, 6, 0.92)
    link(TRUCK, 9, 0.40)
    link(TRUCK, 12, 0.42)

    # discouragements
    W[AIRPLANE, 3] -= 0.12
    W[AIRPLANE, 4] -= 0.08
    W[BIRD, 0] -= 0.14
    W[BIRD, 1] -= 0.14
    W[BIRD, 2] -= 0.14
    W[SHIP, 0] -= 0.14
    W[SHIP, 1] -= 0.12
    W[SHIP, 2] -= 0.12
    W[AUTOMOBILE, 6] -= 0.18
    W[TRUCK, 5] -= 0.14
    W[AUTOMOBILE, 3] -= 0.08
    W[TRUCK, 3] -= 0.08
    W[FROG, 7] -= 0.08

    b += np.array([-0.10, 0.00, 0.18, 0.00, 0.00, 0.00, 0.10, 0.00, 0.22, -0.06], np.float32)
    return W, b


def make_large_readout(num_templates=12):
    W = np.zeros((10, num_templates), np.float32)
    b = np.zeros(10, np.float32)
    baseline = 1.9

    def link(cls, t, w):
        W[cls, t] += w
        if w > 0:
            b[cls] -= baseline * w

    link(AIRPLANE, 10, 0.72)
    link(AUTOMOBILE, 8, 0.66)
    link(BIRD, 11, 0.48)
    link(CAT, 0, 0.94)
    link(CAT, 5, 0.40)
    link(DEER, 2, 0.88)
    link(DEER, 6, 0.38)
    link(DOG, 1, 0.92)
    link(DOG, 5, 0.34)
    link(DOG, 6, 0.20)
    link(FROG, 4, 0.84)
    link(HORSE, 3, 0.90)
    link(HORSE, 6, 0.42)
    link(SHIP, 9, 0.78)
    link(TRUCK, 7, 0.82)

    W[CAT, 1] -= 0.12
    W[DOG, 0] -= 0.12
    W[DEER, 3] -= 0.14
    W[HORSE, 2] -= 0.14
    W[TRUCK, 8] -= 0.14
    W[AUTOMOBILE, 7] -= 0.14
    W[SHIP, 10] -= 0.12
    W[AIRPLANE, 9] -= 0.10
    W[BIRD, 10] -= 0.10

    b += np.array([0.04, 0.00, 0.10, 0.08, 0.10, 0.08, 0.08, 0.10, 0.12, -0.04], np.float32)
    return W, b


def write_network_and_weights(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    network = {
        'layers': [
            {'name': 'sem', 'op': 'conv', 'inputs': ['image'], 'out': 12, 'kernel': [1, 1], 'stride': 1, 'padding': 0},
            {'name': 'semr', 'op': 'relu', 'inputs': ['sem']},
            {'name': 'cpool', 'op': 'avgpool', 'inputs': ['semr'], 'size': [4, 4], 'stride': [4, 4]},
            {'name': 'clogits', 'op': 'dense', 'inputs': ['cpool'], 'out': 10},
            {'name': 'small', 'op': 'conv', 'inputs': ['semr'], 'out': 14, 'kernel': [9, 9], 'stride': 4, 'padding': 4},
            {'name': 'smalln', 'op': 'norm', 'inputs': ['small']},
            {'name': 'smallr', 'op': 'relu', 'inputs': ['smalln']},
            {'name': 'smallp', 'op': 'maxpool', 'inputs': ['smallr'], 'size': [8, 8], 'stride': [8, 8]},
            {'name': 'slogits', 'op': 'dense', 'inputs': ['smallp'], 'out': 10},
            {'name': 'large', 'op': 'conv', 'inputs': ['semr'], 'out': 12, 'kernel': [17, 17], 'stride': 4, 'padding': 8},
            {'name': 'largen', 'op': 'norm', 'inputs': ['large']},
            {'name': 'larger', 'op': 'relu', 'inputs': ['largen']},
            {'name': 'largep', 'op': 'maxpool', 'inputs': ['larger'], 'size': [8, 8], 'stride': [8, 8]},
            {'name': 'llogits', 'op': 'dense', 'inputs': ['largep'], 'out': 10},
            {'name': 'out', 'op': 'add', 'inputs': ['clogits', 'slogits', 'llogits']},
        ]
    }

    sem_W, sem_b = make_semantic_weights()
    color_W, color_b = make_color_readout()
    small_W, small_b = make_small_template_bank()
    large_W, large_b = make_large_template_bank()
    slogits_W, slogits_b = make_small_readout(small_W.shape[0])
    llogits_W, llogits_b = make_large_readout(large_W.shape[0])

    weights = {
        'sem.W': sem_W,
        'sem.b': sem_b,
        'clogits.W': color_W,
        'clogits.b': color_b.astype(np.float32),
        'small.W': small_W,
        'small.b': small_b.astype(np.float32),
        'slogits.W': slogits_W,
        'slogits.b': slogits_b.astype(np.float32),
        'large.W': large_W,
        'large.b': large_b.astype(np.float32),
        'llogits.W': llogits_W,
        'llogits.b': llogits_b.astype(np.float32),
    }

    with open(os.path.join(out_dir, 'network.json'), 'w') as f:
        json.dump(network, f, separators=(',', ':'))
    np.savez(os.path.join(out_dir, 'weights.npz'), **weights)


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else '.'
    write_network_and_weights(target)
