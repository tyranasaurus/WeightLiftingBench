"""Builds a hand-designed German traffic sign classifier.

Architecture:
  image (3,32,32)
    -> conv 1x1 (8 color-opponent channels)
    -> relu
    -> avgpool 4x4 stride 4  -> (8,8,8)
    -> norm per channel      -> contrast-normalised spatial patterns
    -> reshape (512,)
    -> dense (43,)

All weights are set from analytic shape/color templates and general road-sign
knowledge; no training image or label statistic is used.
"""
import json, os, sys
import numpy as np

CLASSES = 43
GH = GW = 8
NC = 8


def build_masks():
    y_g, x_g = np.mgrid[:GH, :GW].astype(np.float64)
    cy, cx = (GH - 1) / 2.0, (GW - 1) / 2.0
    d_g = np.sqrt((y_g - cy) ** 2 + (x_g - cx) ** 2)

    ring    = ((d_g > 2.3) & (d_g < 3.35)).astype(np.float32)
    inner   = (d_g < 2.3).astype(np.float32)
    center  = (d_g < 1.2).astype(np.float32)
    mid     = ((d_g >= 1.2) & (d_g < 2.3)).astype(np.float32)
    disc    = (d_g < 3.35).astype(np.float32)

    # Circle-specific ring cells (the mid-height sides)
    circ_sides = np.zeros((GH, GW), np.float32)
    for yy, xx in [(2, 1), (2, 6), (3, 1), (3, 6), (4, 1), (4, 6), (5, 1), (5, 6)]:
        circ_sides[yy, xx] = 1.0

    tri_up_corners = np.zeros((GH, GW), np.float32)
    for yy, xx in [(6, 0), (6, 1), (6, 6), (6, 7)]:
        tri_up_corners[yy, xx] = 1.0

    tri_dn_corners = np.zeros((GH, GW), np.float32)
    for yy, xx in [(0, 0), (0, 1), (0, 6), (0, 7)]:
        tri_dn_corners[yy, xx] = 1.0

    tri_dn_top = np.zeros((GH, GW), np.float32)
    for xx in range(GW):
        tri_dn_top[0, xx] = 1.0
    tri_up_bot = np.zeros((GH, GW), np.float32)
    for xx in range(GW):
        tri_up_bot[6, xx] = 1.0

    tri_up = np.zeros((GH, GW), np.float32)
    tri_up_e = np.zeros((GH, GW), np.float32)
    for yy in range(GH):
        for xx in range(GW):
            yf = float(yy); xf = float(xx)
            if yf < 0.4 or yf > 6.7:
                continue
            t = (yf - 0.5) / 6.0
            x_l = cx - 3.0 * t
            x_r = cx + 3.0 * t
            if x_l <= xf <= x_r:
                tri_up[yy, xx] = 1.0
            edge = 0.0
            if abs(xf - x_l) < 0.7: edge = 1.0
            if abs(xf - x_r) < 0.7: edge = 1.0
            if yf > 5.7 and 0.3 <= xf <= 6.7: edge = 1.0
            tri_up_e[yy, xx] = edge

    tri_dn = np.zeros((GH, GW), np.float32)
    tri_dn_e = np.zeros((GH, GW), np.float32)
    for yy in range(GH):
        for xx in range(GW):
            yf = float(yy); xf = float(xx)
            if yf < 0.3 or yf > 6.7:
                continue
            t = (yf - 0.5) / 6.0
            x_l = 0.5 + 3.0 * t
            x_r = 6.5 - 3.0 * t
            if x_l <= xf <= x_r:
                tri_dn[yy, xx] = 1.0
            edge = 0.0
            if abs(xf - x_l) < 0.7: edge = 1.0
            if abs(xf - x_r) < 0.7: edge = 1.0
            if yf < 1.2 and 0.3 <= xf <= 6.7: edge = 1.0
            tri_dn_e[yy, xx] = edge

    mh = np.abs(y_g - cy) + np.abs(x_g - cx)
    diamond_full = (mh < 3.4).astype(np.float32)
    diamond_in   = (mh < 2.4).astype(np.float32)
    diamond_edge = (diamond_full - diamond_in).clip(0).astype(np.float32)

    h_stripe = ((y_g > 2.7) & (y_g < 4.3) & (x_g > 0.5) & (x_g < 6.5)).astype(np.float32)
    v_stripe = ((x_g > 2.7) & (x_g < 4.3) & (y_g > 0.5) & (y_g < 6.5)).astype(np.float32)

    diag_dr = np.zeros((GH, GW), np.float32)
    diag_dl = np.zeros((GH, GW), np.float32)
    for yy in range(GH):
        for xx in range(GW):
            if disc[yy, xx] < 0.5:
                continue
            dr = (yy - cy) - (xx - cx)
            dl = (yy - cy) + (xx - cx)
            if abs(dr) < 1.1:
                diag_dr[yy, xx] = 1.0
            if abs(dl) < 1.1:
                diag_dl[yy, xx] = 1.0

    top_half = (y_g < cy).astype(np.float32)
    bot_half = (y_g > cy).astype(np.float32)
    left_half = (x_g < cx).astype(np.float32)
    right_half = (x_g > cx).astype(np.float32)

    return {
        "ring": ring, "inner": inner, "center": center, "mid": mid, "disc": disc,
        "circ_sides": circ_sides, "tri_up_corners": tri_up_corners,
        "tri_dn_corners": tri_dn_corners,
        "tri_dn_top": tri_dn_top, "tri_up_bot": tri_up_bot,
        "tri_up": tri_up, "tri_up_e": tri_up_e,
        "tri_dn": tri_dn, "tri_dn_e": tri_dn_e,
        "diamond_full": diamond_full, "diamond_in": diamond_in, "diamond_edge": diamond_edge,
        "h_stripe": h_stripe, "v_stripe": v_stripe,
        "diag_dr": diag_dr, "diag_dl": diag_dl,
        "top": top_half, "bot": bot_half, "left": left_half, "right": right_half,
        "y_g": y_g.astype(np.float32), "x_g": x_g.astype(np.float32),
    }


def build_weights():
    M = build_masks()
    ring = M["ring"]; inner = M["inner"]; center = M["center"]; mid = M["mid"]
    disc = M["disc"]
    circ_sides = M["circ_sides"]; tri_up_corners = M["tri_up_corners"]; tri_dn_corners = M["tri_dn_corners"]
    tri_dn_top = M["tri_dn_top"]; tri_up_bot = M["tri_up_bot"]
    tri_up = M["tri_up"]; tri_up_e = M["tri_up_e"]; tri_dn = M["tri_dn"]; tri_dn_e = M["tri_dn_e"]
    diamond_full = M["diamond_full"]; diamond_in = M["diamond_in"]; diamond_edge = M["diamond_edge"]
    h_stripe = M["h_stripe"]; v_stripe = M["v_stripe"]
    diag_dr = M["diag_dr"]; diag_dl = M["diag_dl"]
    tri_up_interior = (tri_up - tri_up_e).clip(0, 1)
    tri_dn_interior = (tri_dn - tri_dn_e).clip(0, 1)

    # 1x1 color conv
    col_W = np.array([
        [ 2.0, -1.0, -1.0],   # 0 red
        [-1.0, -1.0,  2.0],   # 1 blue
        [ 1.0,  1.0, -2.0],   # 2 yellow
        [ 1.0,  1.0,  1.0],   # 3 bright
        [-1.0, -1.0, -1.0],   # 4 dark
        [ 2.0, -2.0,  0.0],   # 5 red_pure (not yellow)
        [ 0.3,  1.3, -2.0],   # 6 yellow_pure (not red)
        [-0.3, -0.7,  1.5],   # 7 blue_dim
    ], dtype=np.float32).reshape(NC, 3, 1, 1)
    col_b = np.array([
        -0.10, -0.10, -0.10, -0.60,  0.55, -0.15, -0.10, -0.10,
    ], dtype=np.float32)

    D = np.zeros((CLASSES, NC, GH, GW), np.float32)

    def circle_red_base(k, strength=1.0):
        D[k, 0] += ring * 1.0 * strength
        D[k, 5] += ring * 0.9 * strength
        D[k, 0] -= tri_up_corners * 1.2 * strength
        D[k, 5] -= tri_up_corners * 1.0 * strength
        D[k, 0] -= tri_dn_corners * 1.2 * strength
        D[k, 5] -= tri_dn_corners * 1.0 * strength
        D[k, 0] -= tri_dn_top * 0.5 * strength
        D[k, 0] -= tri_up_bot * 0.5 * strength
        D[k, 1] -= disc * 0.3 * strength
        D[k, 2] -= disc * 0.2 * strength

    def triangle_up_base(k, strength=1.0):
        D[k, 0] += tri_up_e * 1.1 * strength
        D[k, 5] += tri_up_e * 1.0 * strength
        D[k, 0] += tri_up_corners * 0.7 * strength
        D[k, 5] += tri_up_corners * 0.6 * strength
        D[k, 0] -= circ_sides * 1.2 * strength
        D[k, 5] -= circ_sides * 1.0 * strength
        D[k, 0] -= tri_dn_corners * 1.0 * strength
        D[k, 5] -= tri_dn_corners * 0.8 * strength
        D[k, 0] -= tri_dn_top * 0.4 * strength
        D[k, 1] -= disc * 0.3 * strength
        D[k, 2] -= disc * 0.2 * strength

    def triangle_dn_base(k, strength=1.0):
        D[k, 0] += tri_dn_e * 1.1 * strength
        D[k, 5] += tri_dn_e * 1.0 * strength
        D[k, 0] += tri_dn_corners * 0.7 * strength
        D[k, 5] += tri_dn_corners * 0.6 * strength
        D[k, 0] -= circ_sides * 1.2 * strength
        D[k, 5] -= circ_sides * 1.0 * strength
        D[k, 0] -= tri_up_corners * 1.0 * strength
        D[k, 5] -= tri_up_corners * 0.8 * strength
        D[k, 0] -= tri_up_bot * 0.4 * strength
        D[k, 1] -= disc * 0.3 * strength
        D[k, 2] -= disc * 0.2 * strength

    # ------------- Speed limits 0-5, 7, 8: red ring, bright white middle, some digit darkness
    for k in [0, 1, 2, 3, 4, 5, 7, 8]:
        circle_red_base(k)
        D[k, 3] += inner * 0.5
        D[k, 3] += mid * 0.4        # extra bright at mid (less in center where digits are)
        D[k, 4] += center * 0.5     # some digit darkness, but weaker than vehicle signs
        D[k, 3] -= ring * 0.4       # ring NOT bright

    # Wider dark band for 3-digit numbers (100, 120)
    wide_dark = np.zeros((GH, GW), np.float32)
    wide_dark[3:5, 2:6] = 1.0
    D[7, 4] += wide_dark * 0.25
    D[8, 4] += wide_dark * 0.25

    # 6: end of 80 (dim grey circle with a black diagonal)
    D[6, 4] += disc * 0.8
    D[6, 3] += disc * 0.2
    D[6, 0] -= disc * 0.9
    D[6, 5] -= disc * 0.7
    D[6, 1] -= disc * 0.6
    D[6, 2] -= disc * 0.5
    D[6, 3] -= ring * 0.3

    # 9: no passing (red ring, two cars — one red, one dark)
    circle_red_base(9)
    D[9, 4] += center * 0.5
    D[9, 0] += center * 0.4
    D[9, 5] += center * 0.3
    D[9, 3] += mid * 0.2
    D[9, 3] -= center * 0.3     # less bright at centre than speed limits

    # 10: no passing trucks (red ring, two dark vehicles filling middle)
    circle_red_base(10)
    D[10, 4] += center * 1.1
    D[10, 4] += mid * 0.5
    D[10, 3] -= inner * 0.4     # strong anti-bright inside
    D[10, 3] -= ring * 0.3

    # 11: priority at next junction (red triangle, white interior, dark cross)
    triangle_up_base(11)
    D[11, 3] += tri_up_interior * 0.5
    D[11, 4] += center * 0.4

    # 12: priority road (yellow diamond, white border)
    D[12, 2] += diamond_full * 1.6
    D[12, 6] += diamond_full * 1.5
    D[12, 3] += diamond_edge * 0.6
    D[12, 0] -= disc * 0.5
    D[12, 5] -= disc * 0.5
    D[12, 1] -= disc * 0.5
    D[12, 4] -= diamond_in * 0.3

    # 13: yield (inverted red triangle, very big white interior)
    triangle_dn_base(13)
    D[13, 3] += tri_dn_interior * 1.0
    D[13, 4] -= tri_dn_interior * 0.3

    # 14: stop (red octagon with white "STOP" text, text takes a BIG central white area)
    D[14, 0] += disc * 1.0
    D[14, 5] += disc * 0.9
    D[14, 3] += center * 0.9
    D[14, 3] += mid * 0.4
    D[14, 1] -= disc * 0.4
    D[14, 2] -= disc * 0.3
    D[14, 0] -= tri_up_corners * 0.4
    D[14, 0] -= tri_dn_corners * 0.4
    # Stop's white area is broader (not just a thin strip like no-entry's bar)
    D[14, 3] -= h_stripe * 0.2  # anti-thin-bar pattern (push away from no-entry)

    # 15: no vehicles (red ring, VERY empty bright interior)
    circle_red_base(15)
    D[15, 3] += inner * 0.7
    D[15, 3] += center * 0.9        # strong bright at centre
    D[15, 4] -= center * 1.6        # STRONG anti-dark at centre (nothing there)
    D[15, 4] -= inner * 0.4
    D[15, 3] -= ring * 0.4

    # 16: no trucks (red ring, dark truck in centre)
    circle_red_base(16)
    D[16, 4] += center * 0.9
    D[16, 4] += mid * 0.2
    D[16, 3] += (inner - center).clip(0) * 0.3
    D[16, 3] -= center * 0.3

    # 17: no entry (solid red disc + thin white horizontal bar)
    D[17, 0] += disc * 1.1
    D[17, 5] += disc * 1.0
    D[17, 3] += h_stripe * 1.6     # specifically the horizontal bar
    D[17, 1] -= disc * 0.4
    D[17, 2] -= disc * 0.3
    D[17, 3] -= center * 0.6       # NOT a central white blob (would confuse with stop)
    D[17, 3] -= mid * 0.3          # NOT bright around centre either
    D[17, 0] -= tri_up_corners * 0.5
    D[17, 0] -= tri_dn_corners * 0.5
    D[17, 0] += circ_sides * 0.3    # extra red at middle-sides (solid disc)

    # 18-31: warning triangles, with small pictogram-position variations
    for k in range(18, 32):
        triangle_up_base(k)
        D[k, 3] += tri_up_interior * 0.25
        D[k, 4] += tri_up_interior * 0.25

    # Fine differentiation by pictogram position (general knowledge of sign pictograms)
    y_g = M["y_g"]; x_g = M["x_g"]

    # 18 general danger: vertical exclamation, dark on centre vertical line
    D[18, 4] += (v_stripe * tri_up_interior) * 0.3
    # 19 bend to the left: curved arrow going left
    D[19, 4] += ((x_g < 3.5) & (tri_up_interior > 0)).astype(np.float32) * 0.3
    # 20 bend to the right
    D[20, 4] += ((x_g > 3.5) & (tri_up_interior > 0)).astype(np.float32) * 0.3
    # 21 double bend starts left: left/centre mixed
    D[21, 4] += ((x_g < 4) & (tri_up_interior > 0)).astype(np.float32) * 0.2
    # 22 bumpy road: horizontal wave mid
    D[22, 4] += (((y_g > 3) & (y_g < 5)) & (tri_up_interior > 0)).astype(np.float32) * 0.3
    # 23 slippery road: curved marks
    D[23, 4] += ((tri_up_interior > 0) & ((y_g > 3) | (x_g < 3))).astype(np.float32) * 0.2
    # 24 road narrows on the right
    D[24, 4] += ((x_g > 3) & (tri_up_interior > 0)).astype(np.float32) * 0.2
    # 25 road works: dark worker figure vertical centre
    D[25, 4] += (v_stripe * tri_up_interior) * 0.3
    # 26 traffic lights: visible coloured lights
    D[26, 0] += center * 0.2
    D[26, 1] += center * 0.2
    D[26, 2] += center * 0.2
    # 27 pedestrians
    D[27, 4] += (v_stripe * tri_up_interior) * 0.25
    # 28 children crossing
    D[28, 4] += (v_stripe * tri_up_interior) * 0.2
    D[28, 4] += ((y_g > 4) & (tri_up_interior > 0)).astype(np.float32) * 0.15
    # 29 cyclists
    D[29, 4] += ((y_g > 3) & (tri_up_interior > 0)).astype(np.float32) * 0.2
    # 30 ice/snow: sparse pattern, slightly brighter interior
    D[30, 3] += tri_up_interior * 0.15
    D[30, 4] -= center * 0.15
    # 31 wild animals: large dark pictogram across middle
    D[31, 4] += tri_up_interior * 0.15

    # 32: end of all limits (grey circle with dark diagonals, overall dim)
    D[32, 4] += disc * 0.9
    D[32, 3] += ring * 0.2
    D[32, 0] -= disc * 1.0
    D[32, 5] -= disc * 0.9
    D[32, 1] -= disc * 0.6
    D[32, 2] -= disc * 0.5
    D[32, 3] -= center * 0.3
    D[32, 3] -= inner * 0.2

    # 33-40: blue mandatory signs
    for k in range(33, 41):
        D[k, 1] += disc * 1.1
        D[k, 7] += disc * 0.7
        D[k, 0] -= disc * 0.5
        D[k, 5] -= disc * 0.4
        D[k, 2] -= disc * 0.4
        D[k, 1] -= tri_up_corners * 0.3
        D[k, 1] -= tri_dn_corners * 0.3

    # 33 turn right ahead
    arrow_33 = np.zeros((GH, GW), np.float32)
    arrow_33[3:6, 3:5] = 0.6
    arrow_33[2:4, 3:6] = 0.7
    arrow_33[2:4, 5:7] = 0.7
    D[33, 3] += arrow_33 * 0.9

    # 34 turn left ahead
    D[34, 3] += arrow_33[:, ::-1].copy() * 0.9

    # 35 ahead only
    arrow_35 = np.zeros((GH, GW), np.float32)
    arrow_35[1:7, 3:5] = 0.9
    arrow_35[1:3, 2:6] = 0.6
    D[35, 3] += arrow_35 * 1.0

    # 36 straight or right
    arrow_36 = np.zeros((GH, GW), np.float32)
    arrow_36[1:7, 3:5] = 0.6
    arrow_36[3:5, 4:7] = 0.7
    D[36, 3] += arrow_36 * 0.9

    # 37 straight or left
    arrow_37 = np.zeros((GH, GW), np.float32)
    arrow_37[1:7, 3:5] = 0.6
    arrow_37[3:5, 1:4] = 0.7
    D[37, 3] += arrow_37 * 0.9

    # 38 keep right (diagonal down-right)
    D[38, 3] += diag_dr * 1.1
    D[38, 3] -= diag_dl * 0.5

    # 39 keep left (diagonal down-left)
    D[39, 3] += diag_dl * 1.1
    D[39, 3] -= diag_dr * 0.5

    # 40 roundabout (three curved arrows in a middle ring, blue centre)
    D[40, 3] += mid * 1.0
    D[40, 3] -= center * 0.6
    D[40, 3] -= ring * 0.3
    D[40, 1] += center * 0.6

    # 41, 42: end of no passing (bright grey circle, grey diagonal, cars)
    for k in [41, 42]:
        D[k, 3] += disc * 0.6
        D[k, 4] += disc * 0.25
        D[k, 0] -= disc * 0.8
        D[k, 5] -= disc * 0.7
        D[k, 1] -= disc * 0.5
        D[k, 2] -= disc * 0.4
    D[42, 4] += center * 0.2

    # L2-normalise each class template so classes are directly comparable
    out_W = D.reshape(CLASSES, -1).astype(np.float32)
    norms = np.linalg.norm(out_W, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    out_W = out_W / norms

    # Small class biases from general road-sign-frequency knowledge (common urban/road signs).
    # These are rounded, from general knowledge, not from training-set counts.
    bias = np.zeros(CLASSES, np.float32)
    bias[1] = 0.20   # 30 km/h is a very common urban zone limit
    bias[2] = 0.25   # 50 km/h is the standard German urban speed limit (most common)
    bias[4] = 0.15   # 70 km/h common on rural roads
    bias[5] = 0.15   # 80 km/h common on rural roads
    bias[7] = 0.10   # 100 km/h on bundesstrasse
    bias[12] = 0.10  # priority road common on main streets
    bias[13] = 0.10  # yield common
    bias[14] = 0.05  # stop less common
    bias[18] = 0.08  # general danger common
    bias[25] = 0.05  # road works common
    bias[38] = 0.08  # keep right common
    bias[10] = -0.05 # specialised no-passing-for-trucks is less common
    bias[15] = -0.10 # all-vehicles-prohibited is rare
    bias[16] = -0.05 # no-trucks is rare
    bias[0] = -0.05  # 20 km/h is rare
    bias[8] = -0.05  # 120 km/h is less common
    bias[41] = -0.05
    bias[42] = -0.10
    bias[27] = -0.05
    out_b = bias

    return col_W, col_b, out_W, out_b


def build_network_json():
    return {"layers": [
        {"name": "col", "op": "conv", "inputs": ["image"], "out": NC, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "col_r", "op": "relu", "inputs": ["col"]},
        {"name": "pool", "op": "avgpool", "inputs": ["col_r"], "size": [4, 4], "stride": [4, 4]},
        {"name": "pool_n", "op": "norm", "inputs": ["pool"]},
        {"name": "flat", "op": "reshape", "inputs": ["pool_n"], "shape": [NC * GH * GW]},
        {"name": "out", "op": "dense", "inputs": ["flat"], "out": CLASSES},
    ]}


def main(out_dir):
    net = build_network_json()
    col_W, col_b, out_W, out_b = build_weights()
    weights = {"col.W": col_W, "col.b": col_b, "out.W": out_W, "out.b": out_b}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/out"
    main(out_dir)
