"""Hand-designed person / no-person network for 96x96 colour photos.

All numbers come from formulas and from general knowledge of how people appear
in everyday photographs (skin-tone relations, head above shoulders, upright
body, dark hair, compact skin regions).  No training data is read or used.

Stage 1 (conv 3x3 stride 2, then relu) makes 11 rectified channels at 48x48:
  0 skin   = relu(blur(R-2G+B) - 0.03), cleaned by -3*relu(blur(R-G)-0.35)
  1 warm   = relu(blur(R-B) - 0.18)
  2 cool   = relu(blur(B-R) - 0.12)
  3 green  = relu(blur(G-(R+B)/2) - 0.08)
  4 bright = relu(blur(Y) - 0.45)      5 dark = relu(0.45 - blur(Y))
  6..9     = relu(+- dY/dx), relu(+- dY/dy)
  10 oversat = relu(blur(R-G) - 0.35)  (red objects; removed from skin by the 1x1)
Colour channels stay in physical units so that "absent" is exactly zero; the
structure channels (4,5,6+7,8+9) are contrast-normalised per image with `norm`.

At three pyramid levels (24x24, 12x12, 6x6) a bank of hand-drawn part
templates (face, head-and-shoulders, narrow upright body, compact skin blob,
dark figure, pair of legs) is applied as likelihood-ratio matched filters:
weights = expected-activation / channel-noise^2, normalised so a perfect match
scores 1.  Templates are drawn in per-pixel SNR units, so a match that lacks a
cue (e.g. no skin) scores only that cue's share of the template energy.
Each template score is soft-thresholded, max-pooled over position, and the
evidence is summed over scales and levels; a hand-set linear readout turns the
six evidence totals (plus a small global-colour context term) into two scores.
"""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/app/out"

# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------


def mesh(k):
    y, x = np.mgrid[0:k, 0:k].astype(np.float64)
    return y, x


def disk(k, cy, cx, ry, rx):
    y, x = mesh(k)
    return np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2) <= 1.0


def ring(k, cy, cx, ry, rx, iry, irx):
    return disk(k, cy, cx, ry, rx) & ~disk(k, cy, cx, iry, irx)


def band_y(k, y0, y1, x0, x1):
    y, x = mesh(k)
    return (y >= y0) & (y <= y1) & (x >= x0) & (x <= x1)


def vstrips(k, c, a, b, y0, y1):
    """two vertical strips at c+-[a,b]"""
    y, x = mesh(k)
    return (((x >= c - b) & (x <= c - a)) | ((x >= c + a) & (x <= c + b))) & (y >= y0) & (y <= y1)


# ---------------------------------------------------------------------------
# hand-set cue table: per-pixel SNR (expected activation / channel noise).
# Colour channels are physical with noise ~0.02 after blurring: bare skin
# excess ~0.12 gives SNR 6.  Structure channels are per-image standardised
# (noise 1): a strong contour activates ~3, hair/dark clothing ~3, lit face ~1.5.
# ---------------------------------------------------------------------------
SKIN = 6.0
DARK = 3.0
BRIGHT = 1.5
EDGE = 3.0
SMOOTH = -0.7
SIGMA = np.array([0.02, 0.03, 0.03, 0.03, 1.0, 1.0, 1.0, 1.0])

C_SKIN, C_WARM, C_COOL, C_GREEN, C_BRIGHT, C_DARK, C_EV, C_EH = range(8)


def new_t(k):
    return np.zeros((8, k, k))


def ksize(d):
    return int(round(2.9 * d)) | 1


def face_template(d):
    """frontal head: skin ellipse, dark hair cap, dark eye band, bright cheeks"""
    k = ksize(d)
    t = new_t(k)
    c = k / 2.0
    face = disk(k, 1.35 * d, c, 0.55 * d, 0.42 * d)
    eyes = disk(k, 1.28 * d, c - 0.20 * d, 0.10 * d, 0.11 * d) | \
        disk(k, 1.28 * d, c + 0.20 * d, 0.10 * d, 0.11 * d)
    mouth = disk(k, 1.70 * d, c, 0.09 * d, 0.13 * d)
    hair = disk(k, 0.78 * d, c, 0.38 * d, 0.46 * d)
    t[C_SKIN][face & ~eyes & ~mouth] = SKIN
    t[C_SKIN][ring(k, 1.35 * d, c, 0.95 * d, 0.75 * d, 0.62 * d, 0.50 * d)] = -0.35 * SKIN
    t[C_DARK][hair] = DARK
    t[C_DARK][eyes] = DARK
    t[C_DARK][mouth] = 0.6 * DARK
    t[C_BRIGHT][face & ~eyes & ~mouth & ~hair] = BRIGHT
    rg = ring(k, 1.35 * d, c, 0.62 * d, 0.48 * d, 0.50 * d, 0.36 * d)
    t[C_EV][rg] = EDGE
    t[C_EH][rg] = EDGE
    return t


def head_shoulders_template(d):
    """head blob sitting on a wider shoulder line with a smooth chest below"""
    k = ksize(d)
    t = new_t(k)
    c = k / 2.0
    t[C_SKIN][disk(k, 0.95 * d, c, 0.45 * d, 0.38 * d)] = SKIN
    t[C_DARK][disk(k, 0.68 * d, c, 0.32 * d, 0.40 * d)] = DARK
    t[C_EH][band_y(k, 0.30 * d, 0.48 * d, c - 0.35 * d, c + 0.35 * d)] = 0.8 * EDGE   # crown
    t[C_EH][band_y(k, 1.45 * d, 1.68 * d, c - 1.00 * d, c + 1.00 * d)] = EDGE         # shoulder line
    t[C_EV][vstrips(k, c, 0.92 * d, 1.15 * d, 1.5 * d, 2.7 * d)] = EDGE               # armsides
    t[C_EV][band_y(k, 1.8 * d, 2.75 * d, c - 0.75 * d, c + 0.75 * d)] = SMOOTH        # smooth chest
    t[C_EH][band_y(k, 1.8 * d, 2.75 * d, c - 0.75 * d, c + 0.75 * d)] = SMOOTH
    t[C_DARK][band_y(k, 1.8 * d, 2.75 * d, c - 0.75 * d, c + 0.75 * d)] = 0.5 * DARK  # clothed chest
    return t


def narrow_body_template(d):
    """head plus a narrow upright torso: vertical contours, smooth inside"""
    k = ksize(d)
    t = new_t(k)
    c = k / 2.0
    t[C_SKIN][disk(k, 0.75 * d, c, 0.38 * d, 0.32 * d)] = SKIN
    t[C_DARK][disk(k, 0.52 * d, c, 0.28 * d, 0.34 * d)] = DARK
    t[C_EV][vstrips(k, c, 0.52 * d, 0.72 * d, 1.1 * d, 2.8 * d)] = EDGE
    t[C_EH][band_y(k, 1.05 * d, 1.25 * d, c - 0.6 * d, c + 0.6 * d)] = EDGE
    t[C_EV][band_y(k, 1.4 * d, 2.8 * d, c - 0.45 * d, c + 0.45 * d)] = SMOOTH
    t[C_EH][band_y(k, 1.4 * d, 2.8 * d, c - 0.45 * d, c + 0.45 * d)] = SMOOTH
    return t


def skin_blob_template(d):
    """compact skin region with non-skin surround (face, arm, hand, leg)"""
    k = ksize(d)
    t = new_t(k)
    c = k / 2.0
    t[C_SKIN][disk(k, c, c, 0.55 * d, 0.50 * d)] = SKIN
    t[C_SKIN][ring(k, c, c, 1.05 * d, 1.00 * d, 0.68 * d, 0.63 * d)] = -0.4 * SKIN
    rg = ring(k, c, c, 0.66 * d, 0.61 * d, 0.52 * d, 0.47 * d)
    t[C_EV][rg] = 0.6 * EDGE
    t[C_EH][rg] = 0.6 * EDGE
    return t


def dark_figure_template(d):
    """dark upright silhouette (dark jacket / trousers / hair mass)"""
    k = ksize(d)
    t = new_t(k)
    c = k / 2.0
    body = band_y(k, 0.6 * d, 2.8 * d, c - 0.55 * d, c + 0.55 * d) | disk(k, 0.7 * d, c, 0.45 * d, 0.5 * d)
    t[C_DARK][body] = DARK
    sur = disk(k, 1.6 * d, c, 1.45 * d, 1.15 * d) & \
        ~band_y(k, 0.1 * d, 2.9 * d, c - 0.75 * d, c + 0.75 * d)
    t[C_DARK][sur] = -0.45 * DARK
    t[C_EV][vstrips(k, c, 0.50 * d, 0.68 * d, 0.6 * d, 2.8 * d)] = 0.7 * EDGE
    return t


def legs_template(d):
    """pair of vertical bars (legs / trousers) with a gap between"""
    k = ksize(d)
    t = new_t(k)
    c = k / 2.0
    legs = vstrips(k, c, 0.42 * d, 0.62 * d, 0.2 * d, 2.8 * d)
    t[C_EV][legs] = EDGE
    t[C_DARK][legs] = 0.6 * DARK
    t[C_EV][band_y(k, 0.2 * d, 2.8 * d, c - 0.14 * d, c + 0.14 * d)] = 0.5 * EDGE   # gap shows bg
    t[C_EH][band_y(k, 0.0 * d, 0.25 * d, c - 0.6 * d, c + 0.6 * d)] = 0.7 * EDGE    # hip / coat hem
    return t


TEMPLATE_FUNCS = [face_template, head_shoulders_template, narrow_body_template,
                  skin_blob_template, dark_figure_template, legs_template]
TYPE_WEIGHT = [3.0, 2.6, 1.2, 1.2, 0.5, 0.4]   # hand-set readout importance


def threshold_for(d96):
    """soft threshold on the matched-filter score; blurred small templates dilute"""
    if d96 <= 14:
        return 0.72
    if d96 <= 26:
        return 0.65
    return 0.58


LEVELS = [("L24", 4, [2.0, 3.0, 4.0, 6.0]),
          ("L12", 8, [2.0, 3.0, 4.0, 6.0]),
          ("L6", 16, [3.0, 4.0, 5.0, 6.0])]

layers = []
weights = {}


def add_layer(**kw):
    layers.append(kw)


# ---------------------------------------------------------------------------
# stage 1
# ---------------------------------------------------------------------------
G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16.0
DX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8.0
DY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float) / 8.0
YW = np.array([0.299, 0.587, 0.114])

W1 = np.zeros((11, 3, 3, 3))
b1 = np.zeros(11)


def put(o, filt, colorvec, bias=0.0):
    W1[o] = filt[None, :, :] * colorvec[:, None, None]
    b1[o] = bias


put(0, G, np.array([1, -2, 1.0]), -0.03)          # skin excess
put(1, G, np.array([1, 0, -1.0]), -0.18)          # warm
put(2, G, np.array([-1, 0, 1.0]), -0.12)          # cool
put(3, G, np.array([-0.5, 1, -0.5]), -0.08)       # green
put(4, G, YW, -0.45)                              # bright
put(5, G, -YW, 0.45)                              # dark
put(6, DX, YW)
put(7, -DX, YW)
put(8, DY, YW)
put(9, -DY, YW)
put(10, G, np.array([1, -1, 0.0]), -0.35)         # over-saturated red
weights["s1.W"], weights["s1.b"] = W1, b1
add_layer(name="s1", op="conv", inputs=["image"], out=11, kernel=[3, 3], stride=2, padding=1)
add_layer(name="s1r", op="relu", inputs=["s1"])

prev = "s1r"
pyr = {48: "s1r"}
for f in (24, 12, 6):
    add_layer(name=f"p{f}", op="avgpool", inputs=[prev], size=[2, 2], stride=[2, 2])
    pyr[f] = f"p{f}"
    prev = f"p{f}"

# colour branch (physical units, skin cleaned of over-saturated red)
SEL_C = np.zeros((4, 11))
SEL_C[0, 0] = 1.0
SEL_C[0, 10] = -3.0
SEL_C[1, 1] = SEL_C[2, 2] = SEL_C[3, 3] = 1.0
# structure branch (contrast-normalised)
SEL_S = np.zeros((4, 11))
SEL_S[0, 4] = 1.0
SEL_S[1, 5] = 1.0
SEL_S[2, 6] = SEL_S[2, 7] = 1.0
SEL_S[3, 8] = SEL_S[3, 9] = 1.0
PAD_C = np.zeros((8, 4))
PAD_C[0:4, 0:4] = np.eye(4)
PAD_S = np.zeros((8, 4))
PAD_S[4:8, 0:4] = np.eye(4)

n_scales = max(len(sc) for _, _, sc in LEVELS)
T = len(TEMPLATE_FUNCS) * n_scales
acc_names = []
for lname, factor, scales in LEVELS:
    src = pyr[96 // factor]
    add_layer(name=lname + "c", op="conv", inputs=[src], out=4, kernel=[1, 1], stride=1, padding=0)
    weights[lname + "c.W"] = SEL_C[:, :, None, None]
    weights[lname + "c.b"] = np.zeros(4)
    add_layer(name=lname + "s", op="conv", inputs=[src], out=4, kernel=[1, 1], stride=1, padding=0)
    weights[lname + "s.W"] = SEL_S[:, :, None, None]
    weights[lname + "s.b"] = np.zeros(4)
    add_layer(name=lname + "n", op="norm", inputs=[lname + "s"])
    add_layer(name=lname + "pc", op="conv", inputs=[lname + "c"], out=8, kernel=[1, 1], stride=1, padding=0)
    weights[lname + "pc.W"] = PAD_C[:, :, None, None]
    weights[lname + "pc.b"] = np.zeros(8)
    add_layer(name=lname + "ps", op="conv", inputs=[lname + "n"], out=8, kernel=[1, 1], stride=1, padding=0)
    weights[lname + "ps.W"] = PAD_S[:, :, None, None]
    weights[lname + "ps.b"] = np.zeros(8)
    add_layer(name=lname + "cat", op="add", inputs=[lname + "pc", lname + "ps"])

    kmax = 0
    bank = []
    thr = []
    for si, d in enumerate(scales):
        for fi, fn in enumerate(TEMPLATE_FUNCS):
            t = fn(d)
            kmax = max(kmax, t.shape[1])
            bank.append((si, fi, t))
            thr.append(threshold_for(d * factor))
    kmax = kmax | 1
    Wt = np.zeros((T, 8, kmax, kmax))
    for idx, (si, fi, t) in enumerate(bank):
        kh = t.shape[1]
        oy = (kmax - kh) // 2
        Wt[idx, :, oy:oy + kh, oy:oy + kh] = t
    # likelihood-ratio matched filter: w = x_exp/sigma^2, perfect match -> 1
    P = Wt.copy()
    denom = (P ** 2).sum(axis=(1, 2, 3))
    denom[denom == 0] = 1.0
    Wt = (P / SIGMA[None, :, None, None]) / denom[:, None, None, None]
    weights[lname + "t.W"] = Wt
    weights[lname + "t.b"] = np.zeros(T)
    pad = (kmax - 1) // 2
    add_layer(name=lname + "t", op="conv", inputs=[lname + "cat"], out=T, kernel=[kmax, kmax],
              stride=2, padding=pad)
    weights[lname + "th.W"] = np.eye(T)[:, :, None, None]
    weights[lname + "th.b"] = -np.array(thr)
    add_layer(name=lname + "th", op="conv", inputs=[lname + "t"], out=T, kernel=[1, 1], stride=1, padding=0)
    add_layer(name=lname + "e", op="relu", inputs=[lname + "th"])
    h_in = 96 // factor
    ho = (h_in + 2 * pad - kmax) // 2 + 1
    add_layer(name=lname + "m", op="maxpool", inputs=[lname + "e"], size=[ho, ho], stride=[1, 1])
    add_layer(name=lname + "v", op="reshape", inputs=[lname + "m"], shape=[T])
    acc_names.append(lname + "v")

add_layer(name="acc", op="add", inputs=acc_names)
Wro = np.zeros((2, T))
for si in range(n_scales):
    for fi, w in enumerate(TYPE_WEIGHT):
        Wro[1, si * len(TEMPLATE_FUNCS) + fi] = w / 2.0   # evidence spans ~2 neighbouring scales
weights["ro.W"] = Wro
weights["ro.b"] = np.array([0.0, -0.55])
add_layer(name="ro", op="dense", inputs=["acc"], out=2)

# global colour context: frame-filling warm colour (food/wood close-ups) and
# frame-filling green (vegetation) argue against a person being the subject
add_layer(name="gmean", op="avgpool", inputs=["s1r"], size=[48, 48], stride=[1, 1])
add_layer(name="gv", op="reshape", inputs=["gmean"], shape=[11])
Wg = np.zeros((2, 11))
Wg[1] = [0.0, -6.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
weights["g.W"] = Wg
weights["g.b"] = np.zeros(2)
add_layer(name="g", op="dense", inputs=["gv"], out=2)
add_layer(name="logits", op="add", inputs=["ro", "g"])

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": layers}, f)
np.savez(os.path.join(OUT, "weights.npz"), **{k: v.astype(np.float32) for k, v in weights.items()})
print("layers:", len(layers), "weights:", sum(v.size for v in weights.values()))
