import json, math, sys
from pathlib import Path

import numpy as np


CLASSES = 10


def make_network_and_weights():
    # First layer: hand-written local colour, brightness and edge features.
    K = 7
    C = 3
    pad = K // 2
    names = []
    W_list = []
    b_list = []

    def add_center(name, coeffs, bias=0.0):
        W = np.zeros((C, K, K), np.float32)
        for c, v in enumerate(coeffs):
            W[c, pad, pad] = v
        names.append(name)
        W_list.append(W)
        b_list.append(float(bias))

    def add_spatial(name, coeffs, kernel, bias=0.0, scale=1.0):
        W = np.zeros((C, K, K), np.float32)
        kh, kw = kernel.shape
        y0 = pad - kh // 2
        x0 = pad - kw // 2
        ker = kernel.astype(np.float32) * float(scale)
        for c, v in enumerate(coeffs):
            W[c, y0:y0 + kh, x0:x0 + kw] += float(v) * ker
        names.append(name)
        W_list.append(W)
        b_list.append(float(bias))

    ycoef = (0.30, 0.59, 0.11)
    avgcoef = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    # raw colour/brightness and simple thresholded colours
    add_center("R", (1, 0, 0), 0)
    add_center("G", (0, 1, 0), 0)
    add_center("B", (0, 0, 1), 0)
    add_center("Y", ycoef, 0)
    add_center("dark", tuple(-v for v in ycoef), 0.55)
    add_center("verydark", tuple(-v for v in ycoef), 0.35)
    add_center("black", tuple(-v for v in ycoef), 0.23)
    add_center("light", ycoef, -0.50)
    add_center("verylight", ycoef, -0.70)
    add_center("red", (1.00, -0.55, -0.35), -0.05)
    add_center("orange", (0.95, 0.35, -0.75), -0.28)
    add_center("yellow", (0.70, 0.70, -0.90), -0.35)
    add_center("green", (-0.55, 1.00, -0.35), -0.03)
    add_center("vivid_green", (-0.70, 1.25, -0.55), -0.06)
    add_center("blue", (-0.35, -0.20, 1.00), -0.05)
    add_center("sky", (-0.25, 0.45, 0.80), -0.42)
    add_center("water", (-0.15, 0.25, 0.80), -0.34)
    add_center("brown", (0.90, 0.45, -0.75), -0.32)
    add_center("tan", (0.65, 0.55, -0.75), -0.20)
    add_center("lowblue", (0, 0, -1.0), 0.62)
    add_center("lowgreen", (0, -1.0, 0), 0.62)
    add_center("lowred", (-1.0, 0, 0), 0.62)
    add_center("rg", (1, -1, 0), 0)
    add_center("gr", (-1, 1, 0), 0)
    add_center("rb", (1, 0, -1), 0)
    add_center("br", (-1, 0, 1), 0)
    add_center("gb", (0, 1, -1), 0)
    add_center("bg", (0, -1, 1), 0)

    # Smooth colour fields: useful for sky, water, grass and large bodies.
    ax = np.arange(K, dtype=np.float32) - pad
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    g = np.exp(-(xx * xx + yy * yy) / (2 * 2.0 * 2.0)).astype(np.float32)
    g /= g.sum()
    add_spatial("smooth_Y", ycoef, g, 0)
    add_spatial("smooth_dark", tuple(-v for v in ycoef), g, 0.55)
    add_spatial("smooth_light", ycoef, g, -0.50)
    add_spatial("smooth_sky", (-0.25, 0.45, 0.80), g, -0.42)
    add_spatial("smooth_water", (-0.15, 0.25, 0.80), g, -0.34)
    add_spatial("smooth_green", (-0.55, 1.00, -0.35), g, -0.03)
    add_spatial("smooth_brown", (0.90, 0.45, -0.75), g, -0.32)

    # Edge and texture filters.  Both signs are included because a later ReLU is used.
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 4.0
    sy = sx.T
    d1 = (sx + sy) / math.sqrt(2.0)
    d2 = (sx - sy) / math.sqrt(2.0)
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32) / 3.0
    for nm, ker in (("edge_x", sx), ("edge_y", sy), ("edge_d1", d1), ("edge_d2", d2), ("lap", lap)):
        add_spatial(nm + "_p", ycoef, ker, 0)
        add_spatial(nm + "_n", ycoef, -ker, 0)

    def gabor(theta, wave=4.0, sxg=2.4, syg=1.0):
        ct, st = math.cos(theta), math.sin(theta)
        xr = xx * ct + yy * st
        yr = -xx * st + yy * ct
        ker = np.exp(-(xr * xr / (2 * sxg * sxg) + yr * yr / (2 * syg * syg))) * np.cos(2 * math.pi * xr / wave)
        ker = ker.astype(np.float32)
        ker -= ker.mean()
        ker /= np.sum(np.abs(ker)) + 1e-6
        return ker

    for nm, th in (("gabor_h", 0.0), ("gabor_d1", math.pi / 4), ("gabor_v", math.pi / 2), ("gabor_d2", 3 * math.pi / 4)):
        ker = gabor(th)
        add_spatial(nm + "_p", ycoef, ker, 0)
        add_spatial(nm + "_n", ycoef, -ker, 0)

    W0 = np.stack(W_list, axis=0).astype(np.float32)
    b0 = np.array(b_list, np.float32)
    fidx = {n: i for i, n in enumerate(names)}

    # Second layer: pass through the first features and add hand-made part detectors.
    TK = 9
    tpad = TK // 2
    pnames = list(names)
    P = len(names)
    W1 = np.zeros((P, len(names), TK, TK), np.float32)
    b1 = np.zeros((P,), np.float32)
    for i in range(P):
        W1[i, i, tpad, tpad] = 1.0

    ax2 = np.arange(TK, dtype=np.float32) - tpad
    yy2, xx2 = np.meshgrid(ax2, ax2, indexing="ij")
    rr = np.sqrt(xx2 * xx2 + yy2 * yy2)

    def norm_mask(m):
        m = m.astype(np.float32)
        s = float(np.sum(np.abs(m)))
        if s > 0:
            m = m / s
        return m

    disk = norm_mask(rr <= 3.2)
    small_disk = norm_mask(rr <= 2.2)
    ring = norm_mask((rr >= 2.0) & (rr <= 4.2))
    hrect = norm_mask((np.abs(yy2) <= 1.3) & (np.abs(xx2) <= 4.1))
    hwide = norm_mask((np.abs(yy2) <= 2.2) & (np.abs(xx2) <= 4.1))
    vrect = norm_mask((np.abs(xx2) <= 1.2) & (np.abs(yy2) <= 4.1))
    diag1 = norm_mask((np.abs(yy2 - xx2) <= 1.1) & (rr <= 4.6))
    diag2 = norm_mask((np.abs(yy2 + xx2) <= 1.1) & (rr <= 4.6))
    cross = norm_mask(((np.abs(yy2) <= 1.0) | (np.abs(xx2) <= 1.0)) & (rr <= 4.4))
    lower = norm_mask((yy2 >= 0) & (rr <= 4.2))

    def add_template(name, terms, bias=0.0):
        nonlocal W1, b1, P
        W = np.zeros((len(names), TK, TK), np.float32)
        for fname, coef, mask in terms:
            if fname in fidx:
                W[fidx[fname]] += float(coef) * mask.astype(np.float32)
        W1 = np.concatenate([W1, W[None]], axis=0)
        b1 = np.concatenate([b1, np.array([float(bias)], np.float32)])
        pnames.append(name)
        P += 1

    edge_terms = [("edge_x_p", 0.5, disk), ("edge_x_n", 0.5, disk), ("edge_y_p", 0.5, disk), ("edge_y_n", 0.5, disk),
                  ("edge_d1_p", 0.35, disk), ("edge_d1_n", 0.35, disk), ("edge_d2_p", 0.35, disk), ("edge_d2_n", 0.35, disk)]

    add_template("wheel", [("black", 1.4, disk), ("verydark", 0.9, disk), ("lap_p", 0.4, ring), ("lap_n", 0.4, ring)], -0.17)
    add_template("small_dark", [("black", 1.0, small_disk), ("verydark", 0.7, small_disk), ("dark", 0.4, small_disk)], -0.12)
    add_template("round_dark", [("dark", 1.0, disk), ("verydark", 0.6, disk)] + edge_terms, -0.24)
    add_template("round_light", [("light", 0.9, disk), ("verylight", 0.7, disk)] + edge_terms, -0.18)
    add_template("white_blob", [("verylight", 1.2, disk), ("light", 0.8, disk), ("blue", -0.2, disk), ("green", -0.2, disk)], -0.18)
    add_template("green_blob", [("green", 1.1, disk), ("vivid_green", 1.2, disk), ("smooth_green", 0.8, disk)], -0.16)
    add_template("brown_blob", [("brown", 1.1, disk), ("tan", 0.8, disk), ("smooth_brown", 0.7, disk)], -0.16)
    add_template("blue_blob", [("blue", 0.9, disk), ("water", 1.0, disk), ("smooth_water", 0.8, disk)], -0.16)
    add_template("sky_blob", [("sky", 1.1, disk), ("smooth_sky", 1.0, disk), ("light", 0.35, disk)], -0.16)
    add_template("texture", edge_terms + [("lap_p", 0.7, disk), ("lap_n", 0.7, disk)], -0.12)
    add_template("fur_face", edge_terms + [("brown", 0.7, disk), ("tan", 0.7, disk), ("dark", 0.4, disk), ("verylight", 0.25, disk)], -0.22)
    add_template("hbody_dark", [("dark", 0.9, hwide), ("brown", 0.7, hwide), ("tan", 0.45, hwide), ("gabor_h_p", 0.5, hrect), ("gabor_h_n", 0.5, hrect)], -0.18)
    add_template("hbody_light", [("light", 0.9, hwide), ("verylight", 0.8, hwide), ("gabor_h_p", 0.45, hrect), ("gabor_h_n", 0.45, hrect)], -0.17)
    add_template("hbody_color", [("red", 0.8, hwide), ("orange", 0.65, hwide), ("yellow", 0.55, hwide), ("blue", 0.6, hwide), ("light", 0.3, hwide)], -0.14)
    add_template("vleg_dark", [("dark", 0.75, vrect), ("brown", 0.65, vrect), ("edge_x_p", 0.45, vrect), ("edge_x_n", 0.45, vrect), ("gabor_v_p", 0.4, vrect), ("gabor_v_n", 0.4, vrect)], -0.16)
    add_template("box_part", [("light", 0.55, hwide), ("verylight", 0.45, hwide), ("dark", 0.35, hwide), ("edge_x_p", 0.55, cross), ("edge_x_n", 0.55, cross), ("edge_y_p", 0.55, cross), ("edge_y_n", 0.55, cross)], -0.20)
    add_template("diagonal_part", [("edge_d1_p", 0.8, diag1), ("edge_d1_n", 0.8, diag1), ("edge_d2_p", 0.8, diag2), ("edge_d2_n", 0.8, diag2), ("gabor_d1_p", 0.4, diag1), ("gabor_d2_p", 0.4, diag2)], -0.13)
    add_template("cross_part", [("edge_x_p", 0.45, cross), ("edge_x_n", 0.45, cross), ("edge_y_p", 0.45, cross), ("edge_y_n", 0.45, cross), ("diagonal_part" if False else "edge_d1_p", 0.2, cross)], -0.12)
    add_template("wing_part", [("diagonal_part" if False else "edge_d1_p", 0.55, diag1), ("edge_d1_n", 0.55, diag1), ("edge_d2_p", 0.55, diag2), ("edge_d2_n", 0.55, diag2), ("sky", 0.25, disk), ("light", 0.25, disk), ("dark", 0.25, disk)], -0.12)
    add_template("hull_part", [("hbody_dark", 0.0, hwide), ("dark", 0.65, hrect), ("light", 0.45, hrect), ("verylight", 0.45, hrect), ("water", 0.35, lower), ("gabor_h_p", 0.45, hrect), ("gabor_h_n", 0.45, hrect)], -0.16)
    add_template("animal_body", [("brown", 0.75, hwide), ("tan", 0.55, hwide), ("dark", 0.45, hwide), ("green", 0.20, lower), ("vivid_green", 0.10, lower), ("gabor_h_p", 0.35, hrect), ("gabor_h_n", 0.35, hrect)], -0.17)

    # Rule layer over an 8x8 grid after average-pooling the part maps.  Each
    # hidden unit is a hand-made conjunction such as "water below + sky above +
    # horizontal hull" or "wheels low + coloured horizontal body".
    H = Wd = 8
    y = (np.arange(H, dtype=np.float32) + 0.5) / H
    x = (np.arange(Wd, dtype=np.float32) + 0.5) / Wd
    gy, gx = np.meshgrid(y, x, indexing="ij")

    def gauss(cx, cy, sx=0.24, sy=0.24):
        return np.exp(-(((gx - cx) / sx) ** 2 + ((gy - cy) / sy) ** 2) / 2).astype(np.float32)

    allm = np.ones((H, Wd), np.float32)
    top = np.clip((0.63 - gy) / 0.55, 0, 1).astype(np.float32)
    upper = gauss(0.5, 0.34, 0.42, 0.24)
    mid = gauss(0.5, 0.52, 0.36, 0.22)
    center = gauss(0.5, 0.55, 0.30, 0.30)
    lower = gauss(0.5, 0.72, 0.40, 0.23)
    bottom = np.clip((gy - 0.42) / 0.50, 0, 1).astype(np.float32)
    lower_band = ((gy > 0.48) & (gy < 0.94)).astype(np.float32)
    left_lower = gauss(0.30, 0.76, 0.18, 0.16)
    right_lower = gauss(0.70, 0.76, 0.18, 0.16)
    wheelpos = (left_lower + right_lower).astype(np.float32)
    horizon = gauss(0.5, 0.52, 0.52, 0.12)
    left = np.clip((0.64 - gx) / 0.55, 0, 1).astype(np.float32)
    right = np.clip((gx - 0.36) / 0.55, 0, 1).astype(np.float32)
    vertical_center = gauss(0.5, 0.58, 0.20, 0.34)
    everywhere_center = np.maximum(center, 0.45 * allm).astype(np.float32)

    masks = {"all": allm, "top": top, "upper": upper, "mid": mid, "center": center,
             "lower": lower, "bottom": bottom, "lower_band": lower_band, "wheelpos": wheelpos,
             "horizon": horizon, "left": left, "right": right, "vertical": vertical_center,
             "wide_center": everywhere_center}
    pidx = {n: i for i, n in enumerate(pnames)}

    rule_W, rule_b, out_cols, rule_names = [], [], [], []

    def add_rule(name, terms, bias, cls_weights):
        w = np.zeros((len(pnames), H, Wd), np.float32)
        for fname, coef, mask_name in terms:
            if fname not in pidx:
                continue
            m = masks[mask_name].astype(np.float32)
            m = m / (float(m.sum()) + 1e-6)
            w[pidx[fname]] += float(coef) * m
        col = np.zeros(CLASSES, np.float32)
        for cls, val in cls_weights.items():
            col[int(cls)] = float(val)
        rule_W.append(w.reshape(-1))
        rule_b.append(float(bias))
        out_cols.append(col)
        rule_names.append(name)

    def ev(fname, coef=1.0, mask="all"):
        return (fname, coef, mask)

    def pos(cls, name, terms, bias=-0.20, strength=1.0, suppress=()):
        cw = {cls: strength}
        for j, v in suppress:
            cw[j] = cw.get(j, 0.0) - float(v)
        add_rule(name, terms, bias, cw)

    # Very broad scene/background evidence used as weak supporting cues.
    pos(0, "open_sky_air", [ev("sky", 4.0, "top"), ev("smooth_sky", 3.0, "top"), ev("wing_part", 2.4, "upper"), ev("diagonal_part", 1.5, "upper"), ev("light", 1.0, "mid")], -0.34, 2.6, suppress=[(8, 0.5), (6, 0.5)])
    pos(0, "dark_plane_silhouette", [ev("sky", 3.5, "top"), ev("small_dark", 2.8, "upper"), ev("wing_part", 3.4, "upper"), ev("cross_part", 1.4, "upper")], -0.32, 2.4, suppress=[(2, 0.5)])
    pos(0, "white_plane", [ev("verylight", 2.8, "upper"), ev("hbody_light", 2.2, "mid"), ev("diagonal_part", 2.8, "upper"), ev("sky_blob", 2.2, "top")], -0.34, 2.3, suppress=[(8, 0.8)])
    pos(0, "plane_no_ground", [ev("smooth_sky", 3.0, "all"), ev("smooth_green", -2.5, "bottom"), ev("water", -2.2, "bottom"), ev("wing_part", 3.0, "mid")], -0.22, 1.8)

    # Automobiles: small coloured or shiny horizontal body with two dark wheels low.
    pos(1, "car_two_wheels", [ev("wheel", 9.5, "wheelpos"), ev("hbody_color", 5.5, "lower"), ev("hbody_light", 4.0, "lower"), ev("box_part", 2.5, "lower"), ev("water", -4.0, "bottom"), ev("fur_face", -2.5, "center")], -0.20, 3.8, suppress=[(3, 0.8), (5, 0.8), (8, 0.8)])
    pos(1, "car_bright_low", [ev("wheel", 6.5, "wheelpos"), ev("red", 3.0, "lower_band"), ev("orange", 2.4, "lower_band"), ev("yellow", 2.0, "lower_band"), ev("blue", 2.0, "lower_band"), ev("verylight", 2.2, "lower"), ev("hbody_color", 3.0, "lower")], -0.22, 3.3)
    pos(1, "car_dark_low", [ev("wheel", 8.5, "wheelpos"), ev("black", 3.0, "lower"), ev("dark", 2.5, "lower"), ev("edge_x_p", 1.8, "lower"), ev("edge_x_n", 1.8, "lower"), ev("hbody_dark", 3.0, "lower")], -0.24, 2.9)
    pos(1, "car_road_box", [ev("box_part", 4.0, "lower"), ev("hbody_light", 3.2, "lower"), ev("hbody_color", 3.2, "lower"), ev("wheel", 4.0, "wheelpos"), ev("smooth_green", -1.4, "bottom")], -0.19, 2.8)

    # Birds: a small, textured or winged object in upper/central natural scenery.
    pos(2, "bird_in_sky", [ev("sky", 3.2, "top"), ev("small_dark", 4.6, "upper"), ev("wing_part", 3.0, "upper"), ev("diagonal_part", 2.2, "upper"), ev("wheel", -4.0, "bottom")], -0.25, 3.0, suppress=[(0, 0.3)])
    pos(2, "bird_perched", [ev("texture", 3.8, "center"), ev("small_dark", 3.4, "upper"), ev("brown", 2.8, "center"), ev("tan", 1.8, "center"), ev("green", 1.4, "all"), ev("hbody_color", -3.0, "lower")], -0.24, 2.9)
    pos(2, "bird_colored", [ev("texture", 3.0, "center"), ev("green", 2.0, "center"), ev("yellow", 1.8, "center"), ev("red", 1.4, "center"), ev("small_dark", 2.0, "upper"), ev("wing_part", 1.8, "upper")], -0.23, 2.3)
    pos(2, "bird_not_vehicle", [ev("fur_face", 1.8, "upper"), ev("round_dark", 1.8, "upper"), ev("diagonal_part", 2.4, "upper"), ev("wheel", -5.0, "wheelpos"), ev("water", -3.0, "bottom")], -0.16, 1.6)

    # Cats: close, central furry round shapes; less sky/water/vehicle evidence.
    pos(3, "cat_face", [ev("fur_face", 6.0, "center"), ev("round_light", 3.5, "center"), ev("round_dark", 3.2, "center"), ev("texture", 4.0, "center"), ev("water", -4.0, "all"), ev("wheel", -5.0, "wheelpos")], -0.28, 3.6, suppress=[(1, 0.9), (8, 0.8)])
    pos(3, "cat_warm_close", [ev("tan", 3.8, "center"), ev("brown_blob", 3.6, "center"), ev("verylight", 2.0, "center"), ev("fur_face", 4.5, "wide_center"), ev("smooth_green", -1.2, "bottom"), ev("sky", -2.0, "top")], -0.25, 3.0)
    pos(3, "cat_dark_close", [ev("dark", 3.0, "center"), ev("black", 2.5, "center"), ev("fur_face", 4.0, "center"), ev("texture", 3.0, "center"), ev("vleg_dark", -1.8, "lower"), ev("hbody_dark", -1.2, "lower")], -0.22, 2.4)
    pos(3, "cat_indoor", [ev("fur_face", 3.8, "center"), ev("round_light", 2.5, "center"), ev("green", -2.0, "bottom"), ev("sky", -2.0, "top"), ev("water", -3.0, "bottom"), ev("box_part", -1.6, "lower")], -0.18, 2.0)

    # Deer: tan/brown slender animal plus vegetation.
    pos(4, "deer_grass_tan", [ev("brown", 4.2, "lower"), ev("tan", 3.8, "center"), ev("animal_body", 4.6, "lower"), ev("vleg_dark", 3.2, "lower"), ev("smooth_green", 3.0, "bottom"), ev("wheel", -5.0, "wheelpos")], -0.30, 3.4)
    pos(4, "deer_small_field", [ev("brown_blob", 4.0, "center"), ev("hbody_dark", 2.0, "lower"), ev("vleg_dark", 2.8, "lower"), ev("green", 3.0, "all"), ev("sky", 1.2, "top"), ev("hbody_color", -3.0, "lower")], -0.28, 3.0)
    pos(4, "deer_natural", [ev("tan", 3.0, "lower"), ev("brown", 3.0, "lower"), ev("smooth_brown", 2.0, "all"), ev("smooth_green", 2.8, "bottom"), ev("vivid_green", -1.2, "center"), ev("water", -3.0, "bottom")], -0.22, 2.0)

    # Dogs: central furry subject, often larger/lower than cats and with warm/white/dark fur.
    pos(5, "dog_face_body", [ev("fur_face", 5.4, "center"), ev("animal_body", 2.8, "lower"), ev("brown_blob", 3.2, "center"), ev("tan", 3.0, "center"), ev("texture", 3.4, "center"), ev("wheel", -5.0, "wheelpos")], -0.27, 3.4)
    pos(5, "dog_white", [ev("fur_face", 4.5, "center"), ev("round_light", 3.4, "center"), ev("white_blob", 2.8, "center"), ev("animal_body", 2.2, "lower"), ev("green", 0.9, "bottom"), ev("water", -3.5, "all")], -0.25, 2.8)
    pos(5, "dog_dark", [ev("fur_face", 4.3, "center"), ev("dark", 2.8, "center"), ev("black", 2.2, "center"), ev("texture", 3.0, "center"), ev("animal_body", 2.0, "lower"), ev("sky", -1.8, "top")], -0.22, 2.5)
    pos(5, "dog_outdoor", [ev("fur_face", 3.2, "center"), ev("brown", 2.8, "center"), ev("tan", 2.8, "center"), ev("smooth_green", 1.8, "bottom"), ev("vleg_dark", 0.8, "lower"), ev("hbody_dark", -0.8, "lower")], -0.19, 1.9)

    # Frogs: vivid green/yellow central blob with texture, often on vegetation/water.
    pos(6, "frog_green_close", [ev("vivid_green", 7.0, "center"), ev("green_blob", 6.0, "center"), ev("green", 5.0, "center"), ev("round_dark", 1.8, "center"), ev("texture", 2.0, "center"), ev("wheel", -5.0, "wheelpos")], -0.30, 4.2)
    pos(6, "frog_leaf", [ev("vivid_green", 5.5, "all"), ev("green_blob", 4.0, "wide_center"), ev("yellow", 2.2, "center"), ev("smooth_green", 3.2, "all"), ev("fur_face", -1.0, "center"), ev("hbody_color", -2.2, "lower")], -0.28, 3.4)
    pos(6, "frog_water", [ev("green", 4.0, "center"), ev("vivid_green", 4.0, "center"), ev("water", 1.8, "bottom"), ev("blue_blob", 1.2, "bottom"), ev("round_dark", 1.4, "center"), ev("sky", -2.0, "top")], -0.23, 2.4)

    # Horses: large dark/brown horizontal body with legs and field/sky context.
    pos(7, "horse_field", [ev("animal_body", 5.2, "lower"), ev("hbody_dark", 4.2, "lower"), ev("vleg_dark", 5.2, "lower"), ev("brown", 3.5, "lower"), ev("smooth_green", 2.6, "bottom"), ev("wheel", -5.0, "wheelpos")], -0.32, 3.8)
    pos(7, "horse_dark", [ev("hbody_dark", 4.5, "lower"), ev("vleg_dark", 4.0, "lower"), ev("dark", 3.0, "lower"), ev("black", 2.0, "lower"), ev("green", 2.0, "bottom"), ev("fur_face", -0.6, "center")], -0.29, 3.1)
    pos(7, "horse_brown_sky", [ev("brown_blob", 3.6, "lower"), ev("animal_body", 4.0, "lower"), ev("vleg_dark", 3.4, "lower"), ev("sky", 1.4, "top"), ev("smooth_green", 1.8, "bottom"), ev("hbody_color", -3.0, "lower")], -0.28, 2.7)

    # Ships: water below, sky above and an approximately horizontal hull/boat at the horizon.
    pos(8, "ship_water_hull", [ev("water", 7.0, "bottom"), ev("smooth_water", 7.0, "bottom"), ev("hull_part", 5.5, "horizon"), ev("hbody_light", 3.0, "horizon"), ev("sky", 2.2, "top"), ev("smooth_green", -5.0, "bottom")], -0.38, 4.4, suppress=[(0, 0.6)])
    pos(8, "ship_white_water", [ev("water", 6.0, "bottom"), ev("blue_blob", 3.5, "bottom"), ev("white_blob", 4.0, "horizon"), ev("verylight", 2.0, "horizon"), ev("gabor_h_p", 2.0, "horizon"), ev("wheel", -5.0, "wheelpos")], -0.34, 3.6)
    pos(8, "ship_dark_hull", [ev("smooth_water", 6.0, "bottom"), ev("hull_part", 5.0, "lower"), ev("hbody_dark", 2.8, "horizon"), ev("dark", 2.0, "horizon"), ev("sky_blob", 2.0, "top"), ev("green", -4.0, "bottom")], -0.34, 3.3)
    pos(8, "ship_blue_scene", [ev("smooth_water", 5.0, "bottom"), ev("smooth_sky", 3.0, "top"), ev("hbody_light", 2.0, "horizon"), ev("hull_part", 3.0, "horizon"), ev("vivid_green", -4.0, "all")], -0.31, 2.5)

    # Trucks: boxier and taller than cars; strong vertical/horizontal edges and wheels low.
    pos(9, "truck_box_wheels", [ev("wheel", 8.5, "wheelpos"), ev("box_part", 6.0, "lower"), ev("hbody_light", 4.2, "lower"), ev("hbody_dark", 3.6, "lower"), ev("edge_y_p", 2.2, "mid"), ev("edge_y_n", 2.2, "mid"), ev("water", -4.0, "bottom")], -0.26, 3.9)
    pos(9, "truck_tall_front", [ev("box_part", 5.5, "vertical"), ev("hbody_light", 3.0, "mid"), ev("verylight", 2.4, "lower_band"), ev("dark", 2.2, "lower_band"), ev("wheel", 5.0, "wheelpos"), ev("fur_face", -2.5, "center")], -0.24, 3.2)
    pos(9, "truck_colored", [ev("wheel", 6.0, "wheelpos"), ev("red", 2.6, "lower_band"), ev("orange", 2.6, "lower_band"), ev("hbody_color", 3.2, "lower"), ev("box_part", 4.0, "mid"), ev("edge_x_p", 1.6, "lower"), ev("edge_y_p", 1.6, "lower")], -0.24, 2.9)
    pos(9, "truck_large_dark", [ev("wheel", 6.0, "wheelpos"), ev("box_part", 4.6, "lower"), ev("black", 2.5, "lower"), ev("dark", 2.6, "lower"), ev("hbody_dark", 3.6, "lower"), ev("smooth_green", -1.8, "bottom")], -0.23, 2.7)

    # Negative/context rules: when one scene cue is very strong, gently push away
    # implausible classes.  These are deliberately weak and generic.
    add_rule("water_scene", [ev("water", 5.0, "bottom"), ev("smooth_water", 5.0, "bottom"), ev("blue_blob", 2.0, "bottom")], -0.22, {8: 1.4, 0: -0.8, 1: -1.0, 3: -1.2, 4: -1.2, 5: -1.2, 7: -1.2, 9: -1.0})
    add_rule("green_nature_scene", [ev("smooth_green", 5.0, "bottom"), ev("green", 3.0, "all"), ev("vivid_green", 1.5, "all")], -0.22, {4: 0.9, 6: 1.0, 7: 0.9, 2: 0.4, 1: -1.0, 8: -1.0, 9: -1.0, 0: -0.6})
    add_rule("vehicle_evidence", [ev("wheel", 7.0, "wheelpos"), ev("box_part", 4.0, "lower"), ev("hbody_color", 3.0, "lower")], -0.20, {1: 1.4, 9: 1.3, 3: -1.1, 5: -1.1, 2: -0.8, 4: -0.8, 6: -0.8, 7: -0.8, 8: -0.4})
    add_rule("furry_close_evidence", [ev("fur_face", 6.0, "center"), ev("texture", 3.5, "center"), ev("round_light", 2.0, "center"), ev("round_dark", 2.0, "center")], -0.24, {3: 1.2, 5: 1.2, 1: -1.2, 8: -1.0, 9: -1.0, 0: -0.8})
    add_rule("large_quadruped_evidence", [ev("animal_body", 5.0, "lower"), ev("vleg_dark", 5.0, "lower"), ev("hbody_dark", 3.0, "lower"), ev("smooth_green", 2.0, "bottom")], -0.28, {4: 0.8, 7: 1.2, 1: -1.0, 8: -1.0, 9: -1.0, 0: -0.8})

    rule_W = np.stack(rule_W, axis=0).astype(np.float32)
    rule_b = np.array(rule_b, np.float32)
    Wout = np.stack(out_cols, axis=1).astype(np.float32)
    # Hand calibration: the animal/scene rules have different natural numeric
    # ranges, so the readout is scaled to keep the ten scores comparable.
    class_scale = np.array([1.20, 2.10, 1.55, 0.90, 1.85, 0.78, 1.25, 1.35, 0.72, 1.40], np.float32)
    Wout *= class_scale[:, None]
    # Small hand priors: CIFAR-10 is balanced, but several handcrafted evidence
    # streams (water/sky and fur texture) otherwise dominate the score range.
    bout = np.array([0.22, 0.60, 0.35, -0.08, 0.55, -0.34, 0.08, 0.18, -0.62, 0.28], np.float32)

    weights = {
        "f.W": W0.astype(np.float32),
        "f.b": b0.astype(np.float32),
        "p.W": W1.astype(np.float32),
        "p.b": b1.astype(np.float32),
        "rule.W": rule_W.astype(np.float32),
        "rule.b": rule_b.astype(np.float32),
        "out.W": Wout.astype(np.float32),
        "out.b": bout.astype(np.float32),
    }
    network = {"layers": [
        {"name": "f", "op": "conv", "inputs": ["image"], "out": int(W0.shape[0]), "kernel": [K, K], "stride": 1, "padding": pad},
        {"name": "fr", "op": "relu", "inputs": ["f"]},
        {"name": "p", "op": "conv", "inputs": ["fr"], "out": int(W1.shape[0]), "kernel": [TK, TK], "stride": 1, "padding": tpad},
        {"name": "pr", "op": "relu", "inputs": ["p"]},
        {"name": "pool", "op": "avgpool", "inputs": ["pr"], "size": [4, 4], "stride": [4, 4]},
        {"name": "rule", "op": "dense", "inputs": ["pool"], "out": int(rule_W.shape[0])},
        {"name": "rrelu", "op": "relu", "inputs": ["rule"]},
        {"name": "out", "op": "dense", "inputs": ["rrelu"], "out": CLASSES},
    ]}
    return network, weights


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    net, weights = make_network_and_weights()
    (out_dir / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(out_dir / "weights.npz", **weights)


if __name__ == "__main__":
    main()
