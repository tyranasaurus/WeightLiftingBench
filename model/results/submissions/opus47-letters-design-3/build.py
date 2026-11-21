"""Build ASL letter-classifier network weights by hand.

Idea:
  * The hand boundary produces strong gradients no matter whether the hand is
    brighter or darker than the background, or what fraction of the frame it
    fills.  So the hand-mask signal comes from |grad_x| + |grad_y|, not from
    |normalised image|.
  * 24 hand-drawn silhouette templates are converted to thickened outlines
    (the boundary of the drawn region), since the edge map concentrates mass
    along the silhouette boundary.
  * Each outline template is weighted +1/P on its positive pixels and -1/N on
    the complementary region (inside a fixed bounding box of plausible hand
    positions).  The convolution therefore computes
    mean(edge magnitude along the drawn outline) -
    mean(edge magnitude elsewhere inside the bbox).
  * Max-pool over all spatial positions for translation tolerance.

Each generated template is normalised using only itself. No training image or
cross-template statistic sets a weight; every number comes from this file.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
OUT.mkdir(parents=True, exist_ok=True)

K = 20


def new():
    return np.zeros((K, K), np.float32)


def set_rect(t, cy, cx, length, width, value, angle=0.0):
    rad = np.deg2rad(angle)
    cs, sn = np.cos(rad), np.sin(rad)
    ys, xs = np.indices((K, K), dtype=np.float32)
    dy = ys - cy
    dx = xs - cx
    along = -dy * cs + dx * sn
    across = dy * sn + dx * cs
    mask = (np.abs(along) <= length / 2) & (np.abs(across) <= width / 2)
    t[mask] = value


def set_ellipse(t, cy, cx, ry, rx, value, angle=0.0):
    rad = np.deg2rad(angle)
    cs, sn = np.cos(rad), np.sin(rad)
    ys, xs = np.indices((K, K), dtype=np.float32)
    dy = ys - cy
    dx = xs - cx
    u = -dy * cs + dx * sn
    v = dy * sn + dx * cs
    mask = (u / ry) ** 2 + (v / rx) ** 2 <= 1
    t[mask] = value


def set_disc(t, cy, cx, r, value):
    set_ellipse(t, cy, cx, r, r, value)


def set_ring(t, cy, cx, r_outer, r_inner, value):
    ys, xs = np.indices((K, K), dtype=np.float32)
    d2 = (ys - cy) ** 2 + (xs - cx) ** 2
    mask = (d2 <= r_outer ** 2) & (d2 >= r_inner ** 2)
    t[mask] = value


def set_arc(t, cy, cx, r_outer, r_inner, ang_min, ang_max, value):
    ys, xs = np.indices((K, K), dtype=np.float32)
    dy = ys - cy
    dx = xs - cx
    d2 = dy * dy + dx * dx
    ang = np.degrees(np.arctan2(dy, dx))
    mask = (d2 <= r_outer ** 2) & (d2 >= r_inner ** 2) & (ang >= ang_min) & (ang <= ang_max)
    t[mask] = value


def finger_rect(t, by, bx, length, width, value, angle=0.0):
    rad = np.deg2rad(angle)
    dy = -np.cos(rad)
    dx = np.sin(rad)
    cy = by + length / 2 * dy
    cx = bx + length / 2 * dx
    set_rect(t, cy, cx, length, width, value, angle)


# ===== Palm anchors =====

def add_upright_palm(t):
    set_ellipse(t, 14, 10, 3.5, 4, 1.0)
    set_ellipse(t, 18, 10, 2.8, 3.2, 1.0)


def add_right_palm(t):
    set_ellipse(t, 10, 6, 3.5, 4, 1.0)
    set_ellipse(t, 10, 2, 3, 2.5, 1.0)


def add_down_palm(t):
    set_ellipse(t, 6, 10, 3.5, 4, 1.0)
    set_ellipse(t, 2, 10, 2.8, 3.2, 1.0)


# ===== Silhouette drawings (one per class) =====

def tpl_A():
    t = new()
    add_upright_palm(t)
    set_ellipse(t, 12, 10, 4, 4.5, 1.0)
    set_disc(t, 13, 5, 1.8, 1.0)
    return t


def tpl_B():
    t = new()
    add_upright_palm(t)
    set_rect(t, 6, 10, 10, 7, 1.0)
    return t


def tpl_C():
    t = new()
    set_ellipse(t, 18, 10, 2.5, 3, 1.0)
    set_arc(t, 11, 11, 7.5, 3.5, 70, 290, 1.0)
    return t


def tpl_D():
    t = new()
    add_upright_palm(t)
    finger_rect(t, 11, 10, 10, 2.2, 1.0)
    return t


def tpl_E():
    t = new()
    add_upright_palm(t)
    set_ellipse(t, 12, 10, 4.2, 4.6, 1.0)
    set_disc(t, 9, 7, 1.3, 1.0)
    set_disc(t, 9, 10, 1.3, 1.0)
    set_disc(t, 9, 13, 1.3, 1.0)
    return t


def tpl_F():
    t = new()
    add_upright_palm(t)
    finger_rect(t, 11, 10, 8, 2, 1.0)
    finger_rect(t, 11, 13, 8, 2, 1.0)
    finger_rect(t, 12, 15, 7, 2, 1.0)
    set_ring(t, 11, 6, 2.8, 1.3, 1.0)
    return t


def tpl_G():
    t = new()
    add_right_palm(t)
    set_rect(t, 10, 14, 2.2, 10, 1.0, angle=90)
    return t


def tpl_H():
    t = new()
    add_right_palm(t)
    set_rect(t, 9, 14, 2.2, 10, 1.0, angle=90)
    set_rect(t, 11, 14, 2.2, 10, 1.0, angle=90)
    return t


def tpl_I():
    t = new()
    add_upright_palm(t)
    finger_rect(t, 11, 13, 9, 2.2, 1.0)
    return t


def tpl_K():
    t = new()
    add_upright_palm(t)
    finger_rect(t, 11, 8, 10, 2.2, 1.0)
    finger_rect(t, 11, 11, 9, 2.2, 1.0, angle=55)
    return t


def tpl_L():
    t = new()
    add_upright_palm(t)
    finger_rect(t, 11, 7, 10, 2.2, 1.0)
    set_rect(t, 13, 14, 2.2, 7, 1.0, angle=90)
    return t


def tpl_M():
    t = new()
    add_upright_palm(t)
    set_ellipse(t, 12, 10, 4, 4.4, 1.0)
    set_disc(t, 9, 7, 1.6, 1.0)
    set_disc(t, 9, 10, 1.6, 1.0)
    set_disc(t, 9, 13, 1.6, 1.0)
    return t


def tpl_N():
    t = new()
    add_upright_palm(t)
    set_ellipse(t, 12, 10, 4, 4.4, 1.0)
    set_disc(t, 9, 8, 1.7, 1.0)
    set_disc(t, 9, 12, 1.7, 1.0)
    return t


def tpl_O():
    t = new()
    set_ellipse(t, 18, 10, 2.5, 3, 1.0)
    set_ring(t, 11, 10, 6.5, 2.5, 1.0)
    return t


def tpl_P():
    t = new()
    add_down_palm(t)
    finger_rect(t, 9, 9, 10, 2.2, 1.0, angle=180)
    finger_rect(t, 9, 11, 9, 2.2, 1.0, angle=140)
    return t


def tpl_Q():
    t = new()
    add_down_palm(t)
    set_rect(t, 14, 10, 10, 2.2, 1.0)
    return t


def tpl_R():
    t = new()
    add_upright_palm(t)
    finger_rect(t, 11, 9, 9, 2, 1.0, angle=8)
    finger_rect(t, 11, 11, 9, 2, 1.0, angle=-8)
    return t


def tpl_S():
    t = new()
    add_upright_palm(t)
    set_ellipse(t, 12, 10, 4, 4.5, 1.0)
    set_rect(t, 11, 10, 2.5, 8, 1.0, angle=90)
    return t


def tpl_T():
    t = new()
    add_right_palm(t)
    set_rect(t, 10, 12, 2.2, 6, 1.0, angle=90)
    return t


def tpl_U():
    t = new()
    add_upright_palm(t)
    set_rect(t, 6, 10, 10, 3.5, 1.0)
    return t


def tpl_V():
    t = new()
    add_upright_palm(t)
    finger_rect(t, 11, 8, 10, 2, 1.0, angle=-15)
    finger_rect(t, 11, 12, 10, 2, 1.0, angle=15)
    return t


def tpl_W():
    t = new()
    add_upright_palm(t)
    finger_rect(t, 11, 7, 9, 1.8, 1.0, angle=-12)
    finger_rect(t, 11, 10, 9, 1.8, 1.0, angle=0)
    finger_rect(t, 11, 13, 9, 1.8, 1.0, angle=12)
    return t


def tpl_X():
    t = new()
    add_upright_palm(t)
    set_rect(t, 9, 10, 4, 2.2, 1.0)
    set_rect(t, 7, 11, 2, 3, 1.0)
    return t


def tpl_Y():
    t = new()
    add_upright_palm(t)
    set_rect(t, 13, 5, 2.2, 7, 1.0, angle=90)
    finger_rect(t, 11, 14, 9, 2.2, 1.0)
    return t


TEMPLATES = [
    tpl_A, tpl_B, tpl_C, tpl_D, tpl_E, tpl_F, tpl_G, tpl_H,
    tpl_I, tpl_K, tpl_L, tpl_M, tpl_N, tpl_O, tpl_P, tpl_Q,
    tpl_R, tpl_S, tpl_T, tpl_U, tpl_V, tpl_W, tpl_X, tpl_Y,
]


# ===== Convert silhouette to fat outline =====
def outline(silhouette, fat=2):
    mask = silhouette > 0
    eroded = mask.copy()
    for _ in range(fat):
        shift_up = np.roll(eroded, 1, axis=0); shift_up[0] = False
        shift_dn = np.roll(eroded, -1, axis=0); shift_dn[-1] = False
        shift_lt = np.roll(eroded, 1, axis=1); shift_lt[:, 0] = False
        shift_rt = np.roll(eroded, -1, axis=1); shift_rt[:, -1] = False
        eroded = eroded & shift_up & shift_dn & shift_lt & shift_rt
    return (mask & ~eroded).astype(np.float32)


# Bounding box for "plausible hand" region: everywhere inside here that
# isn't on the outline is anti-region (-1/N).
def anti_mask():
    m = np.zeros((K, K), bool)
    m[0:20, 2:18] = True
    m[0:3, 2:4] = False
    m[0:3, 15:18] = False
    return m


ANTI = anti_mask()


def finalize(silhouette):
    out = outline(silhouette, fat=2)
    pos = (out > 0)
    neg = ANTI & ~pos
    weights = np.zeros_like(out)
    P = int(pos.sum())
    N = int(neg.sum())
    if P > 0:
        weights[pos] = 1.0 / P
    if N > 0:
        weights[neg] = -1.0 / N
    return weights.astype(np.float32)


templates = np.zeros((24, 1, K, K), np.float32)
for i, fn in enumerate(TEMPLATES):
    templates[i, 0] = finalize(fn())

templates_b = np.zeros(24, np.float32)


# ===== Simple edge filters =====
# Sobel-x: horizontal derivative (positive at bright-right / dark-left transitions)
sobel_x = np.array([[-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0]], dtype=np.float32)
sobel_y = np.array([[-1.0, -2.0, -1.0],
                    [ 0.0,  0.0,  0.0],
                    [ 1.0,  2.0,  1.0]], dtype=np.float32)

sx_W = sobel_x.reshape(1, 1, 3, 3)
sx_b = np.zeros(1, np.float32)
sy_W = sobel_y.reshape(1, 1, 3, 3)
sy_b = np.zeros(1, np.float32)

# duplicate to 2 channels [+x, -x] then ReLU gives |x| after summation
dup_W = np.array([[+1.0], [-1.0]], np.float32).reshape(2, 1, 1, 1)
dup_b = np.zeros(2, np.float32)

sum2_W = np.array([+1.0, +1.0], np.float32).reshape(1, 2, 1, 1)
sum2_b = np.zeros(1, np.float32)

# threshold edge magnitude: only keep strong edges
THR = 0.4
thr_W = np.array([[1.0]], np.float32).reshape(1, 1, 1, 1)
thr_b = np.array([-THR], np.float32)


np.savez(
    OUT / "weights.npz",
    **{
        "sx.W": sx_W, "sx.b": sx_b,
        "dup_x.W": dup_W, "dup_x.b": dup_b,
        "abs_x.W": sum2_W, "abs_x.b": sum2_b,
        "sy.W": sy_W, "sy.b": sy_b,
        "dup_y.W": dup_W, "dup_y.b": dup_b,
        "abs_y.W": sum2_W, "abs_y.b": sum2_b,
        "thr.W": thr_W, "thr.b": thr_b,
        "tpl.W": templates, "tpl.b": templates_b,
    },
)

network = {
    "layers": [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        # Horizontal gradient branch
        {"name": "sx", "op": "conv", "inputs": ["n0"], "out": 1, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "dup_x", "op": "conv", "inputs": ["sx"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "rx", "op": "relu", "inputs": ["dup_x"]},
        {"name": "abs_x", "op": "conv", "inputs": ["rx"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        # Vertical gradient branch
        {"name": "sy", "op": "conv", "inputs": ["n0"], "out": 1, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "dup_y", "op": "conv", "inputs": ["sy"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "ry", "op": "relu", "inputs": ["dup_y"]},
        {"name": "abs_y", "op": "conv", "inputs": ["ry"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        # Combine edges
        {"name": "edge", "op": "add", "inputs": ["abs_x", "abs_y"]},
        # Threshold low edges
        {"name": "thr", "op": "conv", "inputs": ["edge"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "rt", "op": "relu", "inputs": ["thr"]},
        # Template matching
        {"name": "tpl", "op": "conv", "inputs": ["rt"], "out": 24, "kernel": [K, K], "stride": 1, "padding": 0},
        {"name": "pool", "op": "maxpool", "inputs": ["tpl"], "size": [28 - K + 1, 28 - K + 1], "stride": [1, 1]},
        {"name": "flat", "op": "reshape", "inputs": ["pool"], "shape": [24]},
    ]
}

with open(OUT / "network.json", "w") as f:
    json.dump(network, f, indent=2)

print("saved network.json and weights.npz to", OUT)
