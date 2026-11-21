"""Builds a hand-designed sign-letter classifier: python3 build.py OUT_DIR

Nothing here is read from data and nothing is fitted. Every template is one synthetic drawing of a hand shape, typed
from general knowledge of the American Sign Language alphabet (palm boxes, finger capsules, thumb, forearm), rendered
at 28 x 28 on a fixed grid of sizes / tilts / widths, and turned into weights using that drawing alone.

Network:  norm -> 8-orientation edge filters (Gaussian derivatives, both signs) -> |edge| with a soft clip ->
2 x 2 average pool -> global norm  (+ a blurred-intensity channel on the same 11 x 11 grid)  ->  one convolution
holding every template, searched over +-4 px shifts -> max over shifts -> max over each letter's templates.
Template = drawing's edge maps, 0 inside its hand, a penalty on its background, plus +-|silhouette| (hand vs
background) in the intensity channel.
"""
import json, sys, os
import numpy as np

N = 28          # image size
SS = 4          # supersampling for rendering
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

# ----------------------------------------------------------------------------------------------------------------
# 2-D shape primitives (signed distance: negative inside).  Model coordinates: x to the right, y downwards, frame 0..1
# ----------------------------------------------------------------------------------------------------------------

def sd_capsule(X, Y, a, b, r):
    ax, ay = a; bx, by = b
    px, py, dx, dy = X - ax, Y - ay, bx - ax, by - ay
    h = np.clip((px * dx + py * dy) / (dx * dx + dy * dy + 1e-12), 0.0, 1.0)
    return np.hypot(px - h * dx, py - h * dy) - r


def sd_chain(X, Y, pts, r):
    d = None
    for a, b in zip(pts[:-1], pts[1:]):
        e = sd_capsule(X, Y, a, b, r)
        d = e if d is None else np.minimum(d, e)
    return d


def sd_rbox(X, Y, c, hw, hh, r, ang=0.0):
    t = np.deg2rad(ang)
    px, py = X - c[0], Y - c[1]
    u = np.cos(t) * px + np.sin(t) * py
    v = -np.sin(t) * px + np.cos(t) * py
    qx, qy = np.abs(u) - (hw - r), np.abs(v) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def sd_ellipse(X, Y, c, rx, ry, ang=0.0):
    t = np.deg2rad(ang)
    px, py = X - c[0], Y - c[1]
    u = np.cos(t) * px + np.sin(t) * py
    v = -np.sin(t) * px + np.cos(t) * py
    k = np.hypot(u / rx, v / ry)
    return (k - 1.0) * min(rx, ry)


def sd_circle(X, Y, c, r):
    return np.hypot(X - c[0], Y - c[1]) - r


# A part: (kind, params, mode).  mode: "fill" = hand surface, "line" = hand surface whose rim is drawn as a dark crease
# where it lies over hand drawn earlier, "hole" = background seen through the hand.
def P(kind, *params, mode="line"):
    return (kind, params, mode)


SD = {"cap": sd_capsule, "chain": sd_chain, "rbox": sd_rbox, "ell": sd_ellipse, "circ": sd_circle}

HAND, BACK, CREASE, CREASE_W = 1.0, 0.0, 0.5, 0.022


def render(parts, scale=1.0, rot=0.0, shift=(0.0, 0.0), aspect=1.0, with_mask=False):
    """Draw parts in painter's order on a supersampled grid, then box-average down to 28 x 28."""
    n = N * SS
    g = (np.arange(n) + 0.5) / n
    Xi, Yi = np.meshgrid(g, g)
    cx, cy = 0.5 + shift[0], 0.55 + shift[1]
    t = np.deg2rad(rot)
    px, py = (Xi - cx) / (scale * aspect), (Yi - cy) / scale
    X = 0.5 + np.cos(t) * px + np.sin(t) * py          # inverse rotation: image -> model coordinates
    Y = 0.55 - np.sin(t) * px + np.cos(t) * py
    img = np.full((n, n), BACK)
    hand = np.zeros((n, n), bool)
    for kind, params, mode in parts:
        d = SD[kind](X, Y, *params)
        inside = d < 0
        if mode == "hole":
            img[inside] = BACK; hand[inside] = False; continue
        rim = inside & (d > -CREASE_W) & hand
        img[inside] = HAND
        if mode == "line":
            img[rim] = CREASE
        hand |= inside
    img = img.reshape(N, SS, N, SS).mean(axis=(1, 3))
    if with_mask:
        return img, hand.astype(float).reshape(N, SS, N, SS).mean(axis=(1, 3))
    return img


# ----------------------------------------------------------------------------------------------------------------
# Hand models, one per letter.  Right hand, palm towards the viewer unless the sign turns it; thumb on the viewer's
# right, little finger on the viewer's left, forearm entering from the bottom edge.  v holds the per-variant knobs.
# ----------------------------------------------------------------------------------------------------------------

FR, TR = 0.064, 0.068      # finger and thumb radius (a finger is about 3.5 px wide at 28 x 28)


def forearm(top=(0.53, 0.74), ang=0.0, r=0.22):
    t = np.deg2rad(ang)
    return P("cap", top, (top[0] + np.sin(t) * 0.9, top[1] + np.cos(t) * 0.9), r, mode="fill")


def tipof(base, ang, length):
    t = np.deg2rad(ang)
    return (base[0] + np.sin(t) * length, base[1] - np.cos(t) * length)


def finger(base, ang, length, r=FR):
    return P("cap", base, tipof(base, ang, length), r)


def bumps(xs, tops, y1, r=FR, lean=0.0):
    """Curled fingers seen from the front: a row of short bars with rounded knuckle tops."""
    return [P("cap", (x, y0), (x + lean, y1), r) for x, y0 in zip(xs, tops)]


def palm(c=(0.53, 0.62), hw=0.25, hh=0.22, r=0.11, ang=0.0):
    return P("rbox", c, hw, hh, r, ang, mode="fill")


def m_A(v):     # fist, knuckles facing out, thumb straight up along the index side
    t, w = v.get("t", 0.0), v.get("w", 0.0)
    return [forearm((0.53, 0.72)), palm((0.53, 0.47), 0.27 + w, 0.29, 0.14),
            *bumps([0.53 + (x - 0.53) * (1 + 3 * w) for x in (0.33, 0.455, 0.58, 0.70)], (0.24, 0.19, 0.17, 0.20), 0.47),
            P("cap", (0.79 + w, 0.64), (0.81 + w, 0.24 + t), TR)]


def m_B(v):     # flat hand, four fingers up and together, thumb folded across the palm
    s, ty = v.get("spread", 0.0), v.get("ty", 0.0)
    xs, lens = (0.355, 0.47, 0.585, 0.70), (0.30, 0.39, 0.42, 0.38)
    return [forearm((0.53, 0.80)), palm((0.53, 0.65), 0.25, 0.22, 0.1),
            *[finger((x, 0.47), (i - 1.5) * s, L) for i, (x, L) in enumerate(zip(xs, lens))],
            P("cap", (0.74, 0.74 + ty), (0.45, 0.60 + ty), TR)]


def m_C(v):     # side view, fingers and thumb curved into a C opening to the viewer's right
    o, th = v.get("open", 0.0), v.get("th", 0.0)
    return [forearm((0.30, 0.70), -18, 0.22), P("cap", (0.28, 0.74), (0.31, 0.30), 0.20, mode="fill"),
            P("chain", [(0.32, 0.22), (0.54, 0.11), (0.75, 0.15), (0.86, 0.30 - o)], 0.10),
            P("chain", [(0.38, 0.70), (0.62, 0.70), (0.82, 0.62 + o - th)], 0.075)]


def m_D(v):     # index straight up, the other fingers curled round to meet the thumb
    a, tx = v.get("ang", 0.0), v.get("tx", 0.0)
    return [forearm((0.55, 0.80)), palm((0.53, 0.60), 0.26, 0.22, 0.13),
            *bumps((0.34, 0.45, 0.56), (0.42, 0.40, 0.41), 0.55),
            P("cap", (0.77, 0.70), (0.52 + tx, 0.60), TR),
            finger((0.66, 0.44), a, 0.42)]


def m_E(v):     # fingertips bent down onto the thumb, which lies across under them
    h, ty = v.get("h", 0.0), v.get("ty", 0.0)
    return [forearm((0.53, 0.78)), palm((0.53, 0.52), 0.26, 0.28, 0.12),
            *bumps((0.34, 0.455, 0.57, 0.685), (0.20 + h, 0.15 + h, 0.14 + h, 0.17 + h), 0.36 + h * 0.5, 0.058),
            P("cap", (0.76, 0.50 + ty), (0.35, 0.45 + ty), TR)]


def m_F(v):     # index and thumb make a ring on the thumb side, the other three fingers up and spread
    s, hy = v.get("spread", 12.0), v.get("hy", 0.0)
    return [forearm((0.50, 0.80), 5), palm((0.50, 0.62), 0.24, 0.22, 0.11),
            finger((0.58, 0.44), s * 0.5, 0.40), finger((0.46, 0.44), -s * 0.4, 0.40), finger((0.34, 0.47), -s * 1.3, 0.33, 0.058),
            P("chain", [(0.64, 0.45 + hy), (0.77, 0.36 + hy), (0.86, 0.48 + hy)], 0.06),
            P("chain", [(0.64, 0.74), (0.82, 0.64 + hy), (0.86, 0.50 + hy)], TR),
            P("circ", (0.75, 0.53 + hy), 0.065, mode="hole")]


def m_G(v):     # hand on its side, index pointing to the viewer's right, thumb parallel below it
    a, tl = v.get("ang", -4.0), v.get("tl", 0.0)
    return [forearm((0.26, 0.70), -25, 0.22), palm((0.30, 0.52), 0.23, 0.24, 0.11),
            finger((0.40, 0.33), 90 + a, 0.57),
            finger((0.44, 0.47), 90 + a, 0.28 + tl, TR)]


def m_H(v):     # index and middle together, pointing to the viewer's right
    a, tl = v.get("ang", -2.0), v.get("tl", 0.0)
    return [forearm((0.26, 0.72), -25, 0.22), palm((0.30, 0.55), 0.23, 0.24, 0.11),
            finger((0.40, 0.30), 90 + a, 0.55), finger((0.40, 0.42), 90 + a, 0.51),
            finger((0.42, 0.55), 90 + a, 0.2 + tl, TR)]


def m_I(v):     # little finger up, the rest a fist with the thumb across
    a, ty = v.get("ang", -4.0), v.get("ty", 0.0)
    return [forearm((0.56, 0.80)), palm((0.56, 0.56), 0.26, 0.24, 0.13),
            *bumps((0.48, 0.60, 0.72), (0.34, 0.33, 0.36), 0.50),
            P("cap", (0.78, 0.64 + ty), (0.48, 0.58 + ty), TR),
            finger((0.36, 0.44), a, 0.38, 0.058)]


def m_K(v):     # index up, middle angled out, thumb up between them
    a, b = v.get("ang", -25.0), v.get("ia", 2.0)
    return [forearm((0.55, 0.80)), palm((0.53, 0.62), 0.25, 0.22, 0.11),
            *bumps((0.34, 0.45), (0.45, 0.44), 0.57),
            finger((0.55, 0.45), a, 0.34),
            finger((0.66, 0.45), b, 0.42),
            P("cap", (0.64, 0.64), (0.56, 0.42), TR)]


def m_L(v):     # index up, thumb straight out to the viewer's right
    a, ia = v.get("ang", 72.0), v.get("ia", 0.0)
    return [forearm((0.47, 0.80)), palm((0.47, 0.62), 0.24, 0.2, 0.11),
            *bumps((0.30, 0.41, 0.52), (0.44, 0.42, 0.43), 0.57),
            finger((0.60, 0.46), ia, 0.42),
            finger((0.66, 0.64), a, 0.36, TR)]


def m_M(v):     # fist with three fingers folded over the thumb
    n, lean, ty = v.get("n", 3), v.get("lean", 0.0), v.get("ty", 0.0)
    xs = (0.38, 0.51, 0.64) if n == 3 else (0.45, 0.59)
    return [forearm((0.53, 0.78)), palm((0.53, 0.50), 0.27, 0.29, 0.14),
            *[P("cap", (x + 0.07 + lean, 0.24), (x - 0.05 - lean, 0.56), FR) for x in xs],
            P("cap", (0.80, 0.62 + ty), (0.64, 0.62 + ty), TR)]


def m_N(v):     # like M with two fingers over the thumb
    return m_M(dict(v, n=2))


def m_O(v):     # all fingertips meet the thumb, leaving a round opening
    hs, hy = v.get("hole", 0.0), v.get("hy", 0.0)
    return [forearm((0.38, 0.76), -12, 0.22), P("ell", (0.54, 0.47), 0.33, 0.35, 0, mode="fill"),
            P("chain", [(0.28, 0.42), (0.39, 0.19), (0.61, 0.14), (0.79, 0.29), (0.78, 0.46)], 0.09),
            P("chain", [(0.40, 0.70), (0.64, 0.65), (0.78, 0.50)], 0.07),
            P("circ", (0.59, 0.44 + hy), 0.095 + hs, mode="hole")]


def m_P(v):     # K turned so the index points out to the viewer's left and the middle finger down
    a, b = v.get("ang", -70.0), v.get("mid", -115.0)
    return [forearm((0.70, 0.76), 18, 0.22), palm((0.67, 0.53), 0.24, 0.25, 0.12, 20),
            finger((0.56, 0.36), a, 0.52),
            finger((0.56, 0.52), b, 0.30),
            P("cap", (0.68, 0.44), (0.50, 0.46), 0.055)]


def m_Q(v):     # G turned to point down: arm rises at the left, hand arches over, index and thumb hang down
    dx, tl = v.get("dx", 0.0), v.get("tl", 0.0)
    return [P("cap", (0.24, 1.40), (0.30, 0.50), 0.2, mode="fill"),
            palm((0.53, 0.30), 0.34, 0.22, 0.16),
            P("cap", (0.48, 0.42), (0.42 + dx, 0.84), FR),
            P("cap", (0.74, 0.42), (0.80 + dx, 0.82 + tl), TR)]


def m_R(v):     # index and middle crossed, pointing up
    c = v.get("cross", 9.0)
    return [forearm((0.55, 0.80)), palm((0.53, 0.62), 0.25, 0.22, 0.11),
            *bumps((0.34, 0.45), (0.46, 0.45), 0.58),
            P("cap", (0.74, 0.68), (0.50, 0.60), TR),
            finger((0.65, 0.46), -3.0 + v.get("a", 0.0), 0.42), finger((0.58, 0.46), c + v.get("a", 0.0), 0.38, 0.06)]


def m_S(v):     # fist with the thumb across the front of the fingers
    ty, tx = v.get("t", 0.0), v.get("tx", 0.0)
    return [forearm((0.53, 0.78)), palm((0.53, 0.50), 0.27, 0.28, 0.15),
            *bumps((0.34, 0.455, 0.57, 0.685), (0.27, 0.23, 0.22, 0.25), 0.45),
            P("cap", (0.80, 0.62 + ty), (0.40 + tx, 0.52 + ty), TR)]


def m_T(v):     # as signed here: index bent across to the viewer's right, thumb tip poking up behind it
    t, a = v.get("t", 0.0), v.get("ang", 92.0)
    return [forearm((0.44, 0.80)), palm((0.44, 0.62), 0.26, 0.26, 0.11),
            finger((0.46, 0.39), a, 0.45),
            finger((0.50, 0.41), 0.0, 0.26 + t, 0.06)]


def m_U(v):     # index and middle up together
    a = v.get("a", 0.0)
    return [forearm((0.55, 0.80)), palm((0.53, 0.62), 0.25, 0.22, 0.11),
            *bumps((0.34, 0.45), (0.46, 0.45), 0.58),
            P("cap", (0.74, 0.68), (0.48, 0.60 + v.get("ty", 0.0)), TR),
            finger((0.66, 0.46), a, 0.41), finger((0.54, 0.46), a, 0.43)]


def m_V(v):     # index and middle up and spread
    s, a = v.get("spread", 14.0), v.get("a", 0.0)
    return [forearm((0.55, 0.80)), palm((0.53, 0.62), 0.25, 0.22, 0.11),
            *bumps((0.34, 0.45), (0.46, 0.45), 0.58),
            P("cap", (0.74, 0.68), (0.48, 0.60), TR),
            finger((0.64, 0.46), s + a, 0.42), finger((0.53, 0.46), -s + a, 0.42)]


def m_W(v):     # index, middle and ring up and spread, thumb holding the little finger
    s, a = v.get("spread", 14.0), v.get("a", 0.0)
    return [forearm((0.55, 0.80)), palm((0.55, 0.62), 0.25, 0.22, 0.11),
            P("cap", (0.74, 0.68), (0.40, 0.60), TR),
            finger((0.69, 0.46), s + a, 0.39), finger((0.57, 0.45), a, 0.43), finger((0.45, 0.46), -s + a, 0.39)]


def m_X(v):     # index raised and hooked, the rest a fist
    h, a = v.get("h", 0.0), v.get("a", 0.0)
    return [forearm((0.53, 0.80)), palm((0.53, 0.61), 0.26, 0.22, 0.12),
            *bumps((0.35, 0.46), (0.44, 0.43), 0.57),
            P("chain", [(0.59, 0.44), (0.59 + a, 0.23), (0.69 + a, 0.13 + h), (0.77 + a, 0.22 + 2 * h)], FR),
            P("cap", (0.76, 0.68), (0.56, 0.60), TR)]


def m_Y(v):     # little finger up and thumb out: the "hang loose" shape
    a, pa = v.get("ang", 60.0), v.get("pa", -6.0)
    return [forearm((0.51, 0.80)), palm((0.51, 0.62), 0.24, 0.2, 0.11),
            *bumps((0.44, 0.55, 0.66), (0.44, 0.43, 0.45), 0.58),
            finger((0.33, 0.47), pa, 0.37, 0.058),
            finger((0.68, 0.64), a, 0.40, TR)]


MODELS = {L: globals()["m_" + L] for L in LETTERS}

# Four shape variants per letter, typed by hand: the knobs a signer naturally varies (finger spread and angle, thumb
# placement, how far the fingers curl).
VARIANTS = {
    "A": [{}, {"t": 0.08}, {"t": -0.04, "w": 0.02}, {"w": -0.02, "t": 0.04}],
    "B": [{}, {"spread": 3.0}, {"ty": -0.06}, {"spread": 2.0, "ty": 0.04}],
    "C": [{}, {"open": 0.05}, {"open": -0.04}, {"th": 0.08}],
    "D": [{}, {"ang": -6.0}, {"ang": 6.0}, {"tx": 0.08}],
    "E": [{}, {"h": 0.05}, {"ty": 0.05}, {"h": -0.03, "ty": -0.03}],
    "F": [{}, {"spread": 7.0}, {"spread": 18.0}, {"hy": 0.06}],
    "G": [{}, {"ang": -12.0}, {"ang": 5.0}, {"tl": 0.1}],
    "H": [{}, {"ang": -10.0}, {"ang": 6.0}, {"tl": 0.12}],
    "I": [{}, {"ang": -12.0}, {"ang": 5.0}, {"ty": 0.06}],
    "K": [{}, {"ang": -35.0}, {"ang": -15.0}, {"ia": 8.0}],
    "L": [{}, {"ang": 60.0}, {"ang": 85.0}, {"ia": -6.0}],
    "M": [{}, {"lean": 0.04}, {"lean": -0.03}, {"ty": -0.06}],
    "N": [{}, {"lean": 0.04}, {"lean": -0.03}, {"ty": -0.06}],
    "O": [{}, {"hole": 0.03}, {"hole": -0.03}, {"hy": 0.05}],
    "P": [{}, {"ang": -82.0}, {"ang": -55.0}, {"mid": -140.0}],
    "Q": [{}, {"dx": -0.08}, {"dx": 0.06}, {"tl": -0.12}],
    "R": [{}, {"cross": 5.0}, {"cross": 14.0}, {"a": 6.0}],
    "S": [{}, {"t": -0.06}, {"t": 0.05}, {"tx": 0.1}],
    "T": [{}, {"t": -0.07}, {"ang": 82.0}, {"ang": 100.0}],
    "U": [{}, {"a": -6.0}, {"a": 6.0}, {"ty": -0.08}],
    "V": [{}, {"spread": 9.0}, {"spread": 20.0}, {"a": -6.0}],
    "W": [{}, {"spread": 9.0}, {"spread": 20.0}, {"a": -6.0}],
    "X": [{}, {"h": 0.05}, {"a": -0.06}, {"a": 0.05, "h": -0.03}],
    "Y": [{}, {"ang": 48.0}, {"ang": 75.0}, {"pa": -16.0}],
}


# ----------------------------------------------------------------------------------------------------------------
# Fixed front end, all from formulas
# ----------------------------------------------------------------------------------------------------------------
NORI = 8                    # edge orientations over 180 degrees
EK = 7                      # edge kernel size (valid convolution: 28 -> 22, so the frame border makes no edge)
SIG_ACROSS, SIG_ALONG = 1.0, 2.2   # elongated along the edge for sharper orientation tuning
CLIP, SLOPE = 1.0, 0.3      # soft clip of edge strength (in units of the image's own standard deviation)
FM = (N - EK + 1) // 2      # feature map size after 2 x 2 pooling: 11
SHIFT = 2                   # template search: +-2 cells = +-4 pixels each way
TBLUR = 0.8                 # template blur (cells) for tolerance to small shape differences
BGPEN = 0.25                # penalty for edges where the drawing has background, relative to its strongest edge cell
SILK, SILSIG = 8, 1.2       # silhouette channel: 8 x 8 Gaussian blur (sigma 1.2 px), stride 2, on the same 11 x 11 grid
SILW = 0.6                  # weight of the silhouette match next to the (unit-norm) edge match


def edge_filters():
    """Elongated first derivative of a Gaussian, 8 normal directions over 180 degrees, both signs: (16, 1, 7, 7)."""
    h = EK // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(float)
    W = np.zeros((2 * NORI, 1, EK, EK))
    for k in range(NORI):
        t = np.pi * k / NORI
        u = xx * np.cos(t) + yy * np.sin(t)          # across the edge
        w = -xx * np.sin(t) + yy * np.cos(t)         # along the edge
        g = -u * np.exp(-u ** 2 / (2 * SIG_ACROSS ** 2) - w ** 2 / (2 * SIG_ALONG ** 2))
        g -= g.mean()
        g /= np.sum(np.where(u > 0, -g, 0.0))        # a unit step across the edge gives a response of 1
        W[k, 0], W[k + NORI, 0] = -g, g
    return W


def front_end_layers():
    """Layers and weights shared by the network and by the template maker."""
    layers = [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "e1", "op": "conv", "inputs": ["n0"], "out": 2 * NORI, "kernel": [EK, EK], "stride": 1, "padding": 0},
        {"name": "e1r", "op": "relu", "inputs": ["e1"]},
        # |edge| and |edge| - CLIP for each orientation
        {"name": "e2", "op": "conv", "inputs": ["e1r"], "out": 2 * NORI, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "e2r", "op": "relu", "inputs": ["e2"]},
        {"name": "p1", "op": "avgpool", "inputs": ["e2r"], "size": [2, 2], "stride": [2, 2]},
        # soft clip: |e| - (1 - SLOPE) * relu(|e| - CLIP)
        {"name": "e3", "op": "conv", "inputs": ["p1"], "out": NORI, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "f1", "op": "reshape", "inputs": ["e3"], "shape": [1, NORI * FM, FM]},
        {"name": "f2", "op": "norm", "inputs": ["f1"]},
        {"name": "f3", "op": "reshape", "inputs": ["f2"], "shape": [NORI, FM, FM]},
    ]
    w = {"e1.W": edge_filters(), "e1.b": np.zeros(2 * NORI)}
    W2 = np.zeros((2 * NORI, 2 * NORI, 1, 1)); b2 = np.zeros(2 * NORI)
    for k in range(NORI):
        for j in (k, k + NORI):
            W2[k, j] = 1.0; W2[k + NORI, j] = 1.0
        b2[k + NORI] = -CLIP
    w["e2.W"], w["e2.b"] = W2, b2
    W3 = np.zeros((NORI, 2 * NORI, 1, 1))
    for k in range(NORI):
        W3[k, k] = 1.0; W3[k, k + NORI] = -(1.0 - SLOPE)
    w["e3.W"], w["e3.b"] = W3, np.zeros(NORI)
    return layers, w


# ----------------------------------------------------------------------------------------------------------------
# A small forward pass (same arithmetic as the scorer's interpreter) used to turn each drawing into a template
# ----------------------------------------------------------------------------------------------------------------

def _conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def forward(layers, w, x):
    vals = {"image": x}
    for L in layers:
        a = vals[L["inputs"][0]]
        op = L["op"]
        if op == "conv":
            r = _conv(a, w[L["name"] + ".W"], w[L["name"] + ".b"], L["stride"], L["padding"])
        elif op == "dense":
            r = a.reshape(len(a), -1) @ w[L["name"] + ".W"].T + w[L["name"] + ".b"]
        elif op == "relu":
            r = np.maximum(a, 0)
        elif op in ("maxpool", "avgpool"):
            (sh, sw), (th, tw) = L["size"], L["stride"]
            n, c, h, ww = a.shape
            ho, wo = (h - sh) // th + 1, (ww - sw) // tw + 1
            win = np.stack([a[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], 0)
            r = win.max(0) if op == "maxpool" else win.mean(0)
        elif op == "add":
            r = sum(vals[i] for i in L["inputs"])
        elif op == "reshape":
            r = a.reshape((len(a),) + tuple(L["shape"]))
        elif op == "norm":
            mu = a.mean(axis=(2, 3), keepdims=True)
            r = (a - mu) / np.sqrt(((a - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        vals[L["name"]] = r
    return vals[layers[-1]["name"]]


def blur_maps(f, s):
    h = int(np.ceil(3 * s))
    k = np.exp(-np.arange(-h, h + 1) ** 2 / (2 * s * s)); k /= k.sum()
    out = np.zeros_like(f)
    pad = np.pad(f, ((0, 0), (0, 0), (h, h), (0, 0)))
    for i, kv in enumerate(k):
        out += kv * pad[:, :, i:i + f.shape[2], :]
    pad = np.pad(out, ((0, 0), (0, 0), (0, 0), (h, h)))
    out2 = np.zeros_like(f)
    for i, kv in enumerate(k):
        out2 += kv * pad[:, :, :, i:i + f.shape[3]]
    return out2


# ----------------------------------------------------------------------------------------------------------------
# Template set: every letter drawn at a few sizes, tilts and widths (a fixed grid typed by hand, no randomness)
# ----------------------------------------------------------------------------------------------------------------
SCALES = (0.9, 1.0, 1.1)
ROTS = (-10.0, 0.0, 10.0)
ASPECTS = (0.9, 1.0, 1.1)


def all_drawings():
    """Each drawing, its coverage by hand and forearm, and its coverage by the hand alone (every model's first part is
    the forearm or arm, which is left out of the silhouette match because sleeves often make it darker than the hand)."""
    imgs, masks, hands, labels = [], [], [], []
    for c, L in enumerate(LETTERS):
        for v in VARIANTS[L]:
            parts = MODELS[L](v)
            for s in SCALES:
                for r in ROTS:
                    for a in ASPECTS:
                        im, m = render(parts, scale=s, rot=r, aspect=a, with_mask=True)
                        _, h = render(parts[1:], scale=s, rot=r, aspect=a, with_mask=True)
                        imgs.append(im); masks.append(m); hands.append(h); labels.append(c)
    return np.array(imgs), np.array(masks), np.array(hands), np.array(labels)


def silhouette_filter():
    c = (SILK - 1) / 2.0
    yy, xx = np.mgrid[0:SILK, 0:SILK].astype(float)
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * SILSIG ** 2))
    return g / g.sum()


def build(out_dir):
    layers, w = front_end_layers()
    imgs, masks, hands, labels = all_drawings()
    T = len(imgs); n_per = 2 * T // len(LETTERS)
    # Each template is made from its own drawing alone:
    #   + the drawing's edge maps (before the global normalisation, so empty places are exactly 0),
    #   0 inside the drawn hand (real hands are full of creases and shading, so edges there are neither good nor bad),
    #   - a penalty wherever the drawing has background (walls are mostly smooth, so edges there speak against it).
    edge_layers = layers[:[L["name"] for L in layers].index("e3") + 1]
    edges = blur_maps(forward(edge_layers, w, imgs[:, None].astype(np.float64)), TBLUR)   # (T, 8, 11, 11)
    o = EK // 2
    cov = masks[:, o:o + 2 * FM, o:o + 2 * FM].reshape(T, FM, 2, FM, 2).mean(axis=(2, 4))  # hand coverage per cell
    bg = blur_maps((1.0 - cov)[:, None], TBLUR)
    peak = edges.max(axis=(1, 2, 3), keepdims=True)
    feats = edges - BGPEN * peak * bg
    feats = feats - feats.mean(axis=(1, 2, 3), keepdims=True)
    feats = feats / np.sqrt((feats ** 2).sum(axis=(1, 2, 3), keepdims=True))
    # Silhouette part, also from the drawing alone: +1 on the hand, -1 on the background, 0 on the forearm.
    hcov = hands[:, o:o + 2 * FM, o:o + 2 * FM].reshape(T, FM, 2, FM, 2).mean(axis=(2, 4))
    sil = hcov - (1.0 - cov)
    sil = sil - sil.mean(axis=(1, 2), keepdims=True)
    sil = SILW * sil / np.sqrt((sil ** 2).sum(axis=(1, 2), keepdims=True))
    # The hand may be lighter or darker than the background, so every template comes in both signs; the max over a
    # class's templates then gives  edge match + SILW * |silhouette match|.
    W = np.zeros((T, 2, NORI + 1, FM, FM))
    W[:, :, :NORI] = feats[:, None]
    W[:, 0, NORI], W[:, 1, NORI] = sil, -sil
    W = W.reshape(2 * T, NORI + 1, FM, FM)
    # 9-channel input: the 8 edge maps plus the blurred intensity on the same grid
    Wid = np.zeros((NORI + 1, NORI, 1, 1))
    for k in range(NORI):
        Wid[k, k] = 1.0
    Ws = np.zeros((NORI + 1, 1, SILK, SILK)); Ws[NORI, 0] = silhouette_filter()
    w["f4.W"], w["f4.b"] = Wid, np.zeros(NORI + 1)
    w["s1.W"], w["s1.b"] = Ws, np.zeros(NORI + 1)
    layers += [
        {"name": "f4", "op": "conv", "inputs": ["f3"], "out": NORI + 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "s1", "op": "conv", "inputs": ["n0"], "out": NORI + 1, "kernel": [SILK, SILK], "stride": 2, "padding": 0},
        {"name": "f5", "op": "add", "inputs": ["f4", "s1"]},
        {"name": "tm", "op": "conv", "inputs": ["f5"], "out": 2 * T, "kernel": [FM, FM], "stride": 1, "padding": SHIFT},
        {"name": "tmax", "op": "maxpool", "inputs": ["tm"], "size": [2 * SHIFT + 1, 2 * SHIFT + 1], "stride": [1, 1]},
        {"name": "tr", "op": "reshape", "inputs": ["tmax"], "shape": [len(LETTERS), n_per, 1]},
        {"name": "cmax", "op": "maxpool", "inputs": ["tr"], "size": [n_per, 1], "stride": [n_per, 1]},
        {"name": "scores", "op": "reshape", "inputs": ["cmax"], "shape": [len(LETTERS)]},
    ]
    w["tm.W"], w["tm.b"] = W, np.zeros(2 * T)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: np.asarray(v, np.float32) for k, v in w.items()})
    return layers, w, imgs, masks, labels, hands


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
