"""Build a hand-designed network that classifies German traffic signs.

Architecture
------------
  1x1 colour conv : converts RGB -> 4 colour channels using STRICT formulas
                    (red = R - 2G, blue = B - 2R, yellow = G - 2B,
                     brightness = (R+G+B)/3).  "Strict" means the red
                    channel does NOT fire on orange / yellow, and vice
                    versa - so the orange priority-road sign does not
                    look like a red sign to the network.
  relu            : rectifies colour maps
  norm            : zero-mean, unit-variance per channel across the 32x32
                    image, so features are brightness-invariant
  32x32 template  : per-class template of shape (4, 32, 32); the dot
                    product with the colour map gives one class score
  reshape         : (43, 1, 1) -> (43,)

All weights are written by the code; no training data is read.  Templates
come from general knowledge of what each German traffic sign looks like
(red ring, yellow diamond, red triangle, blue disc with an arrow, etc.).
"""
import json
import os
import sys

import numpy as np

H = W = 32
NC = 4
NCL = 43
CY = CX = 15.5

R_IDX = 0        # red = R - 2G (fires on red, NOT on orange/yellow)
B_IDX = 1        # blue = B - 2R (fires on blue only)
Y_IDX = 2        # yellow = G - 2B (fires on yellow/orange, NOT red)
BR_IDX = 3       # brightness = (R + G + B)/3


def nrm(mask):
    s = float(mask.sum())
    return mask.astype(np.float32) / s if s > 0 else mask.astype(np.float32)


def make_masks():
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    r = np.sqrt((yy - CY) ** 2 + (xx - CX) ** 2)
    m = {"yy": yy, "xx": xx, "r": r}

    m["disc"] = (r < 13.0).astype(np.float32)
    m["disc_wide"] = (r < 15.0).astype(np.float32)
    m["disc_inner"] = (r < 8.0).astype(np.float32)
    m["ring_wide"] = ((r >= 6.0) & (r < 15.0)).astype(np.float32)
    m["ring"] = ((r >= 9.5) & (r < 14.0)).astype(np.float32)
    m["centre"] = (r < 5.0).astype(np.float32)
    m["centre_wide"] = (r < 7.0).astype(np.float32)
    m["outside"] = (r > 15.0).astype(np.float32)

    # Triangle (apex at top)
    dy_u = yy - 3.0
    hw_u = np.maximum(dy_u * 0.5 + 0.5, 0.0)
    tri_up_full = ((dy_u >= 0) & (dy_u <= 25.5) &
                   (np.abs(xx - CX) <= hw_u)).astype(np.float32)
    dy_u2 = yy - 6.0
    hw_u2 = np.maximum(dy_u2 * 0.5 - 1.0, 0.0)
    tri_up_inner = ((dy_u2 >= 0) & (dy_u2 <= 22) &
                    (np.abs(xx - CX) <= hw_u2)).astype(np.float32)
    m["tri_up_full"] = tri_up_full
    m["tri_up_inner"] = tri_up_inner
    m["tri_up_border"] = tri_up_full - tri_up_inner
    m["tri_up_out"] = 1 - tri_up_full

    # Triangle (apex at bottom)
    dy_d = 28.0 - yy
    hw_d = np.maximum(dy_d * 0.5 + 0.5, 0.0)
    tri_dn_full = ((dy_d >= 0) & (dy_d <= 25.5) &
                   (np.abs(xx - CX) <= hw_d)).astype(np.float32)
    dy_d2 = 25.0 - yy
    hw_d2 = np.maximum(dy_d2 * 0.5 - 1.0, 0.0)
    tri_dn_full = ((dy_d >= 0) & (dy_d <= 25.5) &
                   (np.abs(xx - CX) <= hw_d)).astype(np.float32)
    tri_dn_inner = ((dy_d2 >= 0) & (dy_d2 <= 22) &
                    (np.abs(xx - CX) <= hw_d2)).astype(np.float32)
    m["tri_dn_full"] = tri_dn_full
    m["tri_dn_inner"] = tri_dn_inner
    m["tri_dn_border"] = tri_dn_full - tri_dn_inner
    m["tri_dn_out"] = 1 - tri_dn_full

    d_abs = np.abs(yy - CY) + np.abs(xx - CX)
    diamond_full = (d_abs < 14.0).astype(np.float32)
    diamond_inner = (d_abs < 11.0).astype(np.float32)
    m["diamond_full"] = diamond_full
    m["diamond_inner"] = diamond_inner
    m["diamond_out"] = 1 - diamond_full

    # No-entry vs STOP discriminators
    m["bar_core"] = ((yy >= 14) & (yy <= 17) &
                     (np.abs(xx - CX) <= 10)).astype(np.float32)
    m["bar_ends"] = ((yy >= 14) & (yy <= 17) &
                     (np.abs(xx - CX) > 9) & (r < 14)).astype(np.float32)

    disc12 = (r < 12).astype(bool)
    m["q_ul"] = ((yy < CY) & (xx < CX) & disc12).astype(np.float32)
    m["q_ur"] = ((yy < CY) & (xx >= CX) & disc12).astype(np.float32)
    m["q_ll"] = ((yy >= CY) & (xx < CX) & disc12).astype(np.float32)
    m["q_lr"] = ((yy >= CY) & (xx >= CX) & disc12).astype(np.float32)

    m["vstrip"] = ((np.abs(xx - CX) < 2.5) & (r < 12)).astype(np.float32)
    m["vstrip_top"] = ((np.abs(xx - CX) < 2.5) &
                       (yy < CY) & (r < 12)).astype(np.float32)
    m["diag_ul_lr"] = ((np.abs((yy - CY) - (xx - CX)) < 2.5) &
                       (r < 12)).astype(np.float32)
    m["diag_ur_ll"] = ((np.abs((yy - CY) + (xx - CX)) < 2.5) &
                       (r < 12)).astype(np.float32)
    m["diag_ur_ll_thin"] = ((np.abs((yy - CY) + (xx - CX)) < 1.8) &
                            (r < 13)).astype(np.float32)

    m["tri_mid"] = ((yy >= 12) & (yy <= 20) & (tri_up_inner > 0)).astype(np.float32)
    m["tri_bot"] = ((yy >= 18) & (yy <= 26) & (tri_up_inner > 0)).astype(np.float32)
    m["tri_left"] = ((xx < CX - 1) & (tri_up_inner > 0)).astype(np.float32)
    m["tri_right"] = ((xx > CX + 1) & (tri_up_inner > 0)).astype(np.float32)

    m["bang_dot"] = ((np.abs(xx - CX) < 2) &
                     (yy >= 19) & (yy <= 23)).astype(np.float32)
    m["bang_line"] = ((np.abs(xx - CX) < 2) &
                      (yy >= 10) & (yy <= 17)).astype(np.float32)

    return m


MASKS = make_masks()


def set_colour_conv(col_W, col_b):
    # red = R - 2G (stays off on orange / yellow / white / pink)
    col_W[R_IDX, :, 0, 0] = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    # blue = B - 2R (stays off on sky / pink / white)
    col_W[B_IDX, :, 0, 0] = np.array([-2.0, 0.0, 1.0], dtype=np.float32)
    # yellow = G - 2B (stays off on red / blue / white)
    col_W[Y_IDX, :, 0, 0] = np.array([0.0, 1.0, -2.0], dtype=np.float32)
    col_W[BR_IDX, :, 0, 0] = np.array([1 / 3.0, 1 / 3.0, 1 / 3.0], dtype=np.float32)


# --------------------------------------------------------------------
# Base templates
# --------------------------------------------------------------------

def red_ring_base():
    """Red at an annular ring, bright interior, no red in the centre."""
    t = np.zeros((NC, H, W), dtype=np.float32)
    t[R_IDX] += 10.0 * nrm(MASKS["ring_wide"])
    t[R_IDX] -= 10.0 * nrm(MASKS["centre"])        # no red at centre
    t[R_IDX] -= 6.0 * nrm(MASKS["outside"])
    t[BR_IDX] += 4.0 * nrm(MASKS["centre_wide"])   # bright interior
    t[BR_IDX] -= 2.0 * nrm(MASKS["ring_wide"])
    return t


def red_disc_base():
    """Red fills the whole disc, including the centre."""
    t = np.zeros((NC, H, W), dtype=np.float32)
    t[R_IDX] += 10.0 * nrm(MASKS["disc"])
    t[R_IDX] += 10.0 * nrm(MASKS["centre"])        # red at centre
    t[R_IDX] -= 6.0 * nrm(MASKS["outside"])
    return t


def red_tri_up_base():
    t = np.zeros((NC, H, W), dtype=np.float32)
    t[R_IDX] += 10.0 * nrm(MASKS["tri_up_border"])
    t[R_IDX] -= 6.0 * nrm(MASKS["tri_up_inner"])
    t[R_IDX] -= 6.0 * nrm(MASKS["tri_up_out"])
    t[BR_IDX] += 4.0 * nrm(MASKS["tri_up_inner"])
    t[BR_IDX] -= 2.0 * nrm(MASKS["tri_up_border"])
    return t


def red_tri_dn_base():
    t = np.zeros((NC, H, W), dtype=np.float32)
    t[R_IDX] += 10.0 * nrm(MASKS["tri_dn_border"])
    t[R_IDX] -= 6.0 * nrm(MASKS["tri_dn_inner"])
    t[R_IDX] -= 6.0 * nrm(MASKS["tri_dn_out"])
    t[BR_IDX] += 4.0 * nrm(MASKS["tri_dn_inner"])
    return t


def yellow_diamond_base():
    t = np.zeros((NC, H, W), dtype=np.float32)
    t[Y_IDX] += 12.0 * nrm(MASKS["diamond_inner"])
    t[Y_IDX] -= 8.0 * nrm(MASKS["diamond_out"])
    return t


def blue_disc_base():
    t = np.zeros((NC, H, W), dtype=np.float32)
    t[B_IDX] += 12.0 * nrm(MASKS["disc"])
    t[B_IDX] -= 8.0 * nrm(MASKS["outside"])
    return t


def grey_stripe_base():
    """Grey/white disc with a dark diagonal stripe - end-of-limit signs."""
    t = np.zeros((NC, H, W), dtype=np.float32)
    t[BR_IDX] += 10.0 * nrm(MASKS["disc"])
    t[BR_IDX] -= 6.0 * nrm(MASKS["outside"])
    t[R_IDX] -= 6.0 * nrm(MASKS["disc"])
    t[B_IDX] -= 4.0 * nrm(MASKS["disc"])
    t[Y_IDX] -= 4.0 * nrm(MASKS["disc"])
    t[BR_IDX] -= 4.0 * nrm(MASKS["diag_ur_ll_thin"])
    return t


# --------------------------------------------------------------------
# Per-class templates
# --------------------------------------------------------------------

def set_class_templates(tmpl_W, tmpl_b):
    for c in (0, 1, 2, 3, 4, 5, 7, 8):
        tmpl_W[c] = red_ring_base()

    tmpl_W[6] = grey_stripe_base()

    for c in (9, 10):
        tmpl_W[c] = red_ring_base()

    # 11: priority at next junction
    tmpl_W[11] = red_tri_up_base()
    cross_v = ((np.abs(MASKS["xx"] - CX) < 2) &
               (MASKS["yy"] >= 8) & (MASKS["yy"] <= 24) &
               (MASKS["tri_up_inner"] > 0)).astype(np.float32)
    tmpl_W[11, BR_IDX] -= 3.0 * nrm(cross_v)

    tmpl_W[12] = yellow_diamond_base()
    tmpl_W[13] = red_tri_dn_base()

    # STOP: red disc with text; anti-cue against the no-entry bar
    tmpl_W[14] = red_disc_base()
    tmpl_W[14, BR_IDX] -= 8.0 * nrm(MASKS["bar_ends"])

    tmpl_W[15] = red_ring_base()
    tmpl_W[16] = red_ring_base()

    # No entry: red disc + white bar that spans the FULL width of the disc
    tmpl_W[17] = red_disc_base()
    tmpl_W[17, BR_IDX] += 6.0 * nrm(MASKS["bar_core"])
    tmpl_W[17, BR_IDX] += 10.0 * nrm(MASKS["bar_ends"])

    # 18: general danger ("!")
    tmpl_W[18] = red_tri_up_base()
    tmpl_W[18, BR_IDX] -= 3.0 * nrm(MASKS["bang_line"])
    tmpl_W[18, BR_IDX] -= 2.0 * nrm(MASKS["bang_dot"])

    tmpl_W[19] = red_tri_up_base()
    tmpl_W[19, BR_IDX] -= 2.0 * nrm(MASKS["tri_right"])

    tmpl_W[20] = red_tri_up_base()
    tmpl_W[20, BR_IDX] -= 2.0 * nrm(MASKS["tri_left"])

    tmpl_W[21] = red_tri_up_base()

    tmpl_W[22] = red_tri_up_base()
    bump = ((MASKS["yy"] >= 17) & (MASKS["yy"] <= 21) &
            (np.abs(MASKS["xx"] - CX) < 6) &
            (MASKS["tri_up_inner"] > 0)).astype(np.float32)
    tmpl_W[22, BR_IDX] -= 3.0 * nrm(bump)

    tmpl_W[23] = red_tri_up_base()
    tmpl_W[23, BR_IDX] -= 2.0 * nrm(MASKS["tri_bot"])

    tmpl_W[24] = red_tri_up_base()
    tmpl_W[25] = red_tri_up_base()

    tmpl_W[26] = red_tri_up_base()

    tmpl_W[27] = red_tri_up_base()
    tmpl_W[28] = red_tri_up_base()

    tmpl_W[29] = red_tri_up_base()
    wheels = ((MASKS["yy"] >= 20) & (MASKS["yy"] <= 24) &
              (np.abs(MASKS["xx"] - CX) < 6) &
              (MASKS["tri_up_inner"] > 0)).astype(np.float32)
    tmpl_W[29, BR_IDX] -= 2.0 * nrm(wheels)

    tmpl_W[30] = red_tri_up_base()
    tmpl_W[31] = red_tri_up_base()

    tmpl_W[32] = grey_stripe_base()

    # Blue signs
    tmpl_W[33] = blue_disc_base()
    tmpl_W[33, BR_IDX] += 6.0 * nrm(MASKS["q_ur"])
    tmpl_W[33, BR_IDX] -= 4.0 * nrm(MASKS["q_ll"])

    tmpl_W[34] = blue_disc_base()
    tmpl_W[34, BR_IDX] += 6.0 * nrm(MASKS["q_ul"])
    tmpl_W[34, BR_IDX] -= 4.0 * nrm(MASKS["q_lr"])

    tmpl_W[35] = blue_disc_base()
    tmpl_W[35, BR_IDX] += 8.0 * nrm(MASKS["vstrip"])
    sides = ((np.abs(MASKS["xx"] - CX) > 4) & (MASKS["r"] < 11)).astype(np.float32)
    tmpl_W[35, BR_IDX] -= 3.0 * nrm(sides)

    tmpl_W[36] = blue_disc_base()
    tmpl_W[36, BR_IDX] += 5.0 * nrm(MASKS["vstrip_top"])
    tmpl_W[36, BR_IDX] += 5.0 * nrm(MASKS["q_lr"])
    tmpl_W[36, BR_IDX] -= 3.0 * nrm(MASKS["q_ll"])

    tmpl_W[37] = blue_disc_base()
    tmpl_W[37, BR_IDX] += 5.0 * nrm(MASKS["vstrip_top"])
    tmpl_W[37, BR_IDX] += 5.0 * nrm(MASKS["q_ll"])
    tmpl_W[37, BR_IDX] -= 3.0 * nrm(MASKS["q_lr"])

    tmpl_W[38] = blue_disc_base()
    tmpl_W[38, BR_IDX] += 8.0 * nrm(MASKS["diag_ul_lr"])
    tmpl_W[38, BR_IDX] -= 4.0 * nrm(MASKS["diag_ur_ll"])

    tmpl_W[39] = blue_disc_base()
    tmpl_W[39, BR_IDX] += 8.0 * nrm(MASKS["diag_ur_ll"])
    tmpl_W[39, BR_IDX] -= 4.0 * nrm(MASKS["diag_ul_lr"])

    tmpl_W[40] = blue_disc_base()
    ra_ring = ((MASKS["r"] >= 5.0) & (MASKS["r"] < 10.5)).astype(np.float32)
    tmpl_W[40, BR_IDX] += 5.0 * nrm(ra_ring)
    tmpl_W[40, BR_IDX] -= 3.0 * nrm(MASKS["centre"])

    tmpl_W[41] = grey_stripe_base()
    tmpl_W[42] = grey_stripe_base()

    # Biases: within-group priors.  Numbers come from the knowledge that
    # 50 km/h, 30 km/h, 70 km/h, priority road, yield, general danger, keep
    # right, and no entry are the most common signs on German roads.
    tmpl_b[:] = 0.0
    tmpl_b[0] = 0.00
    tmpl_b[1] = 0.65
    tmpl_b[2] = 0.70
    tmpl_b[3] = 0.30
    tmpl_b[4] = 0.55
    tmpl_b[5] = 0.50
    tmpl_b[7] = 0.25
    tmpl_b[8] = 0.25
    tmpl_b[9] = 0.30
    tmpl_b[10] = 0.50
    tmpl_b[11] = 0.10
    tmpl_b[12] = 0.70
    tmpl_b[13] = 0.55
    tmpl_b[14] = 0.20
    tmpl_b[15] = 0.00
    tmpl_b[16] = 0.00
    tmpl_b[17] = 0.30
    tmpl_b[18] = 0.55
    tmpl_b[25] = 0.15
    tmpl_b[6] = 0.20
    tmpl_b[33] = 0.10
    tmpl_b[34] = 0.15
    tmpl_b[35] = 0.30
    tmpl_b[38] = 0.65


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    network = {
        "layers": [
            {"name": "col", "op": "conv", "inputs": ["image"],
             "out": NC, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "col_r", "op": "relu", "inputs": ["col"]},
            {"name": "col_n", "op": "norm", "inputs": ["col_r"]},
            {"name": "tmpl", "op": "conv", "inputs": ["col_n"],
             "out": NCL, "kernel": [H, W], "stride": 1, "padding": 0},
            {"name": "out", "op": "reshape", "inputs": ["tmpl"],
             "shape": [NCL]},
        ],
    }
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(network, f)

    col_W = np.zeros((NC, 3, 1, 1), dtype=np.float32)
    col_b = np.zeros((NC,), dtype=np.float32)
    set_colour_conv(col_W, col_b)

    tmpl_W = np.zeros((NCL, NC, H, W), dtype=np.float32)
    tmpl_b = np.zeros((NCL,), dtype=np.float32)
    set_class_templates(tmpl_W, tmpl_b)

    np.savez(os.path.join(out_dir, "weights.npz"),
             **{"col.W": col_W.astype(np.float32),
                "col.b": col_b.astype(np.float32),
                "tmpl.W": tmpl_W.astype(np.float32),
                "tmpl.b": tmpl_b.astype(np.float32)})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
