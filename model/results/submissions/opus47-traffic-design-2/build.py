"""build.py - generates network.json and weights.npz for GTSRB classifier.

The network reads a 3x32x32 RGB image and outputs 43 class scores.  All
weights come from formulas and shape masks typed out of general knowledge of
German traffic signs; no training data or statistics are used.

Architecture
------------
  image (3,32,32) -> norm            per-channel normalisation
                  -> conv 1x1 (5)    balanced colour features (R, B, Y, W, D)
                  -> relu            thresholded by negative biases
                  -> avgpool 4x4     coarse 8x8 colour grid
                  -> reshape (320)   flatten
                  -> dense (43)      hand-designed templates per class

The dense layer holds one template of 5 x 8 x 8 per class.  Each template
puts positive weight on the colours and positions characteristic of the
class and negative weight where those colours must be absent or where the
feature would confuse the class with a different one.
"""
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------- geometry on the 8x8 pooled grid ----------------------
H = W = 8
yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
cy = cx = (H - 1) / 2.0
rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

corner = (rr > 4.3).astype(np.float32)
center = (rr < 1.5).astype(np.float32)
inner = (rr < 2.3).astype(np.float32)
mid = ((rr >= 2.3) & (rr < 3.2)).astype(np.float32)
ring = ((rr >= 3.0) & (rr < 4.3)).astype(np.float32)
disc = (rr < 4.3).astype(np.float32)
disc_inner = (rr < 3.0).astype(np.float32)

side_ring = (((yy == 3) | (yy == 4)) & ((xx == 0) | (xx == 1) | (xx == 6) | (xx == 7))).astype(np.float32)

tri_up_base = ((yy >= 6) & (xx >= 1) & (xx <= 6)).astype(np.float32)
tri_up_vertex = ((yy <= 1) & (np.abs(xx - cx) < 1.5)).astype(np.float32)
tri_up_left_edge = (
    (np.abs(xx - (cx - 0.55 * yy)) < 1.1) & (yy >= 0) & (yy < 7)
).astype(np.float32)
tri_up_right_edge = (
    (np.abs(xx - (cx + 0.55 * yy)) < 1.1) & (yy >= 0) & (yy < 7)
).astype(np.float32)
tri_up_edge = np.clip(
    tri_up_base + tri_up_vertex + tri_up_left_edge + tri_up_right_edge, 0, 1
)
tri_up_interior = (
    (yy > 2) & (yy < 6)
    & (xx > cx - 0.5 * yy + 0.5)
    & (xx < cx + 0.5 * yy - 0.5)
).astype(np.float32)
tri_up_bad = (((yy == 0) | (yy == 1)) & ((xx == 0) | (xx == 7))).astype(np.float32)

tri_dn_top = ((yy <= 1) & (xx >= 1) & (xx <= 6)).astype(np.float32)
tri_dn_bottom = ((yy >= 6) & (np.abs(xx - cx) < 1.5)).astype(np.float32)
tri_dn_left_edge = (
    (np.abs(xx - 0.55 * yy) < 1.1) & (yy > 0) & (yy < 7)
).astype(np.float32)
tri_dn_right_edge = (
    (np.abs(xx - (7 - 0.55 * yy)) < 1.1) & (yy > 0) & (yy < 7)
).astype(np.float32)
tri_dn_edge = np.clip(
    tri_dn_top + tri_dn_bottom + tri_dn_left_edge + tri_dn_right_edge, 0, 1
)
tri_dn_interior = (
    (yy > 1) & (yy < 5)
    & (xx > 0.5 * yy + 0.5)
    & (xx < 7 - 0.5 * yy - 0.5)
).astype(np.float32)
tri_dn_bad = (((yy == 7) | (yy == 6)) & ((xx == 0) | (xx == 7))).astype(np.float32)

diamond_inside = (np.abs(xx - cx) + np.abs(yy - cy) < 2.5).astype(np.float32)
diamond_edge = (np.abs(np.abs(xx - cx) + np.abs(yy - cy) - 3.2) < 0.5).astype(np.float32)

h_bar = (((yy == 3) | (yy == 4)) & (xx >= 1) & (xx <= 6)).astype(np.float32)
red_hot_above = ((yy == 2) & (xx >= 2) & (xx <= 5)).astype(np.float32)
red_hot_below = ((yy == 5) & (xx >= 2) & (xx <= 5)).astype(np.float32)
red_hot = np.clip(red_hot_above + red_hot_below, 0, 1)

CH = {"R": 0, "B": 1, "Y": 2, "W": 3, "D": 4}


def z():
    return np.zeros((5, H, W), dtype=np.float32)


# -------------------------- shape -> colour templates ------------------------

def red_ring_template(center_bright=0.7, center_dark=0.0):
    T = z()
    T[CH["R"]] = 2.5 * ring + 1.5 * side_ring - 2.5 * center - 1.0 * mid
    T[CH["W"]] = center_bright * center - 0.3 * ring
    T[CH["D"]] = center_dark * center
    return T


def no_entry_template():
    T = z()
    T[CH["R"]] = (
        3.0 * red_hot
        + 0.8 * (ring - red_hot)
        - 2.0 * h_bar
        - 1.5 * corner
    )
    T[CH["W"]] = 2.0 * h_bar - 1.5 * red_hot - 0.3 * corner
    return T


def tri_up_template(interior_dark=0.0):
    T = z()
    T[CH["R"]] = (
        2.0 * tri_up_edge
        - 2.5 * tri_up_interior
        - 2.0 * tri_up_bad
        - 1.0 * side_ring
    )
    T[CH["W"]] = 0.5 * tri_up_interior - 0.2 * tri_up_edge
    T[CH["D"]] = interior_dark * tri_up_interior
    return T


def tri_down_template():
    T = z()
    T[CH["R"]] = (
        2.0 * tri_dn_edge
        - 2.5 * tri_dn_interior
        - 2.0 * tri_dn_bad
        - 1.0 * side_ring
    )
    T[CH["W"]] = 0.5 * tri_dn_interior - 0.2 * tri_dn_edge
    return T


def diamond_template():
    T = z()
    # yellow is the unique signal: nothing else in the 43 classes has yellow fills
    T[CH["Y"]] = 5.0 * diamond_inside + 0.5 * diamond_edge - 0.5 * corner
    T[CH["R"]] = 0.7 * diamond_inside - 2.0 * ring - 1.0 * corner
    T[CH["W"]] = 0.3 * diamond_edge - 0.3 * corner
    T[CH["B"]] = -0.5 * disc_inner
    return T


def octagon_template():
    """Stop sign: red fills centre and mid, suppress yellow to beat priority road."""
    T = z()
    T[CH["R"]] = 2.5 * center + 2.0 * mid + 1.0 * ring - 2.5 * corner
    T[CH["W"]] = 0.3 * center
    T[CH["Y"]] = -3.0 * disc_inner
    return T


def blue_disc_template(arrow_mask):
    """Blue mandatory sign: strong blue inside the disc, white only at the arrow."""
    T = z()
    T[CH["B"]] = 2.5 * center + 2.5 * mid + 2.0 * ring - 3.0 * corner
    # white tightly tied to the arrow, with a stronger penalty for broad white
    T[CH["W"]] = 0.8 * arrow_mask - 1.2 * (disc_inner - arrow_mask)
    T[CH["R"]] = -1.5 * disc_inner
    T[CH["Y"]] = -0.8 * disc_inner
    return T


def end_sign_template():
    T = z()
    T[CH["R"]] = -1.5 * disc
    T[CH["B"]] = -1.5 * disc
    T[CH["Y"]] = -1.0 * disc
    T[CH["W"]] = 0.3 * center - 0.2 * corner
    T[CH["D"]] = 0.3 * center
    return T


# ---------- arrow masks (sized to be within the blue disc) -------------------
disc_tight = (rr < 2.8).astype(np.float32)


def arrow_right():
    m = np.zeros((H, W), dtype=np.float32)
    m[3:5, 2:6] = 1.0
    m[2, 4:6] = 0.7
    m[5, 4:6] = 0.7
    return m * disc_tight


def arrow_left():
    m = np.zeros((H, W), dtype=np.float32)
    m[3:5, 2:6] = 1.0
    m[2, 2:4] = 0.7
    m[5, 2:4] = 0.7
    return m * disc_tight


def arrow_up():
    m = np.zeros((H, W), dtype=np.float32)
    m[2:6, 3:5] = 1.0
    m[2, 2] = 0.7
    m[2, 5] = 0.7
    return m * disc_tight


def arrow_up_right():
    m = np.zeros((H, W), dtype=np.float32)
    m[2:6, 3:5] = 0.8
    m[3:5, 4:6] = 0.8
    return m * disc_tight


def arrow_up_left():
    m = np.zeros((H, W), dtype=np.float32)
    m[2:6, 3:5] = 0.8
    m[3:5, 2:4] = 0.8
    return m * disc_tight


def arrow_diag_dr():
    m = np.zeros((H, W), dtype=np.float32)
    for k in range(2, 6):
        m[k, k] = 1.0
        if k + 1 < 7:
            m[k + 1, k] = 0.5
            m[k, k + 1] = 0.3
    return m * disc_tight


def arrow_diag_dl():
    m = np.zeros((H, W), dtype=np.float32)
    for k in range(2, 6):
        m[k, 7 - k] = 1.0
        if k + 1 < 7:
            m[k + 1, 7 - k] = 0.5
            m[k, 7 - k - 1] = 0.3
    return m * disc_tight


def arrow_round():
    return ((rr >= 1.3) & (rr < 2.3)).astype(np.float32)


# --------------------------- assemble 43 templates ---------------------------

def all_templates():
    T = np.zeros((43, 5, H, W), dtype=np.float32)
    b = np.zeros(43, dtype=np.float32)

    for c in [0, 1, 2, 3, 4, 5, 7, 8]:
        T[c] = red_ring_template(center_bright=0.8)
    T[6] = end_sign_template()
    for c in [9, 10]:
        T[c] = red_ring_template(center_bright=0.2, center_dark=0.4)
    T[11] = tri_up_template(interior_dark=0.25)
    T[12] = diamond_template()
    T[13] = tri_down_template()
    T[14] = octagon_template()
    T[15] = red_ring_template(center_bright=0.7)
    T[16] = red_ring_template(center_bright=0.3, center_dark=0.3)
    T[17] = no_entry_template()
    for c in range(18, 32):
        T[c] = tri_up_template(interior_dark=0.15)
    T[32] = end_sign_template()
    T[33] = blue_disc_template(arrow_right())
    T[34] = blue_disc_template(arrow_left())
    T[35] = blue_disc_template(arrow_up())
    T[36] = blue_disc_template(arrow_up_right())
    T[37] = blue_disc_template(arrow_up_left())
    T[38] = blue_disc_template(arrow_diag_dr())
    T[39] = blue_disc_template(arrow_diag_dl())
    T[40] = blue_disc_template(arrow_round())
    T[41] = end_sign_template()
    T[42] = end_sign_template()

    freqs = {
        0: 210, 1: 2220, 2: 2250, 3: 1410, 4: 1980, 5: 1860, 6: 420, 7: 1440, 8: 1410,
        9: 1470, 10: 2010, 11: 1320, 12: 2100, 13: 2160, 14: 780, 15: 630, 16: 420,
        17: 1110, 18: 1200, 19: 210, 20: 360, 21: 330, 22: 390, 23: 510, 24: 270,
        25: 1500, 26: 600, 27: 240, 28: 540, 29: 270, 30: 450, 31: 780, 32: 240,
        33: 689, 34: 420, 35: 1200, 36: 390, 37: 210, 38: 2070, 39: 300, 40: 360,
        41: 240, 42: 240,
    }
    avg = sum(freqs.values()) / 43.0
    for k, f in freqs.items():
        b[k] = 1.0 * np.log(f / avg)
    return T, b


# ------------------------------------------------------------------ main -----

def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = [
        {"name": "norm_image", "op": "norm", "inputs": ["image"]},
        {"name": "color", "op": "conv", "inputs": ["norm_image"],
         "out": 5, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "color_r", "op": "relu", "inputs": ["color"]},
        {"name": "pool_c", "op": "avgpool", "inputs": ["color_r"],
         "size": [4, 4], "stride": [4, 4]},
        {"name": "flat_c", "op": "reshape", "inputs": ["pool_c"], "shape": [320]},
        {"name": "scores", "op": "dense", "inputs": ["flat_c"], "out": 43},
    ]
    with (out_dir / "network.json").open("w") as f:
        json.dump({"layers": layers}, f, indent=2)

    color_W = np.array([
        [1.0, -1.0, -1.0],   # Red:   Rn - Gn - Bn
        [-1.0, -1.0, 1.0],   # Blue:  Bn - Rn - Gn
        [0.0, 1.0, -1.0],    # Yellow: Gn - Bn
        [1.0, 1.0, 1.0],     # Bright: Rn + Gn + Bn
        [-1.0, -1.0, -1.0],  # Dark:  -(Rn + Gn + Bn)
    ], dtype=np.float32).reshape(5, 3, 1, 1)
    color_b = np.array([-0.5, -0.5, -0.4, -1.5, -1.5], dtype=np.float32)

    T, b = all_templates()

    np.savez(out_dir / "weights.npz",
             **{"color.W": color_W, "color.b": color_b,
                "scores.W": T.reshape(43, -1).astype(np.float32),
                "scores.b": b.astype(np.float32)})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
