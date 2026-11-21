"""Build a hand-designed ASL-letter classifier network (no training data, no fitting).

Idea: chamfer-style template matching on contour-presence maps.
  * n0 norm makes the image contrast-invariant; c1 = 12 oriented edge filters (6
    orientations x 2 polarities, 1st derivative of Gaussian, from formulas).
  * relu + clip at TAU (hand-set) + 2x2 maxpool + 3x3 avgpool -> smoothed contour
    energy f (6 orientations, 12x12); b = min(f, TAU) is a soft presence map.
  * d = 3x3 maxpool(b): contour presence within +-1 cell (+-2 px).
  * For each class and each of 5 drawn rotation variants, one synthetic hand drawing
    (palm ellipse + finger capsules at typed angles, from knowledge of the ASL
    alphabet) gives, by itself alone:
       P = cells where the drawing has contour (b > 0.5*TAU)   -> weight +1/|P| on d
       Q = cells where the drawing has no contour (b < 0.15*TAU) -> weight -GAMMA/|Q| on b
    plus a uniform -GAMMA2 penalty on window contour amount (clutter / extra fingers).
  * score = coverage of the template contour by the image contour (within 2 px)
    - gamma * mean image contour where the template expects none - gamma2 * amount;
    max over 3x3 shifts gives +-4 px translation tolerance; the 5 variants give
    rotation tolerance; fc averages variant scores per class.
Nothing is fitted; every number is a formula, a typed shape constant or a hand-set rule.
"""
import sys
import json
import numpy as np

N_ORIENT = 6
FSIZE = 7
SIGMA = 1.3
TAU = 4.0        # contour-energy clip, tames only extreme edges (hand-set)
P_REL = 0.45     # presence threshold as fraction of the template's own peak (hand-set)
Q_REL = 0.12     # absence threshold as fraction of the template's own peak (hand-set)
GAMMA_C = 2.5    # penalty for contour where template expects none (hand-set)
GAMMA_R = 1.2    # same, for ridge channels (hand-set)
W_RIDGE = 0.7    # ridge coverage weight relative to contour (hand-set)
GAMMA2 = 0.2     # penalty for overall contour amount (hand-set)
TK = 10          # template window on the 12x12 grid
TOFF = 1
V_ANGLES = (-6.0, 0.0, 6.0)
V_SCALES = (0.93, 1.0, 1.07)
N_VAR = len(V_ANGLES) * len(V_SCALES)


def edge_filters():
    """24 filters: per orientation [odd+, odd-] (contour) then [even+, even-] (ridge)."""
    ax = np.arange(FSIZE) - (FSIZE - 1) / 2.0
    X, Y = np.meshgrid(ax, ax)
    odd_f, even_f = [], []
    for k in range(N_ORIENT):
        th = k * np.pi / N_ORIENT
        s = X * np.cos(th) + Y * np.sin(th)
        t = -X * np.sin(th) + Y * np.cos(th)
        g = np.exp(-(s * s + t * t) / (2 * SIGMA * SIGMA))
        odd = -(s / (SIGMA * SIGMA)) * g
        odd /= np.sqrt((odd * odd).sum())
        odd_f += [odd, -odd]
        ev = (s * s / SIGMA ** 4 - 1.0 / SIGMA ** 2) * g
        ev = ev - ev.mean()
        ev /= np.sqrt((ev * ev).sum())
        even_f += [ev, -ev]
    return np.stack(odd_f + even_f)


def conv2d(x, W):
    O, kh, kw = W.shape
    p = kh // 2
    xp = np.pad(x, p, mode="constant")
    H, Wd = x.shape
    out = np.zeros((O, H, Wd))
    for o in range(O):
        Wo = W[o]
        acc = np.zeros((H, Wd))
        for dy in range(kh):
            for dx in range(kw):
                acc += Wo[dy, dx] * xp[dy:dy + H, dx:dx + Wd]
        out[o] = acc
    return out


def maxpool2(x):
    O, H, W = x.shape
    h, w = H // 2, W // 2
    return x[:, :2 * h, :2 * w].reshape(O, h, 2, w, 2).max(axis=(2, 4))


def avgpool3(x):
    return (x[:, :-2, :-2] + x[:, 1:-1, :-2] + x[:, 2:, :-2]
            + x[:, :-2, 1:-1] + x[:, 1:-1, 1:-1] + x[:, 2:, 1:-1]
            + x[:, :-2, 2:] + x[:, 1:-1, 2:] + x[:, 2:, 2:]) / 9.0


def dil3(x):
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), constant_values=x.min())
    m = x.copy()
    for dy in range(3):
        for dx in range(3):
            m = np.maximum(m, xp[:, dy:dy + x.shape[1], dx:dx + x.shape[2]])
    return m


def contour(img):
    """(28,28) -> clipped smoothed energy (12, 12, 12): 6 contour + 6 ridge maps."""
    img = (img - img.mean()) / np.sqrt(((img - img.mean()) ** 2).mean() + 1e-5)
    r = np.minimum(np.maximum(conv2d(img, FILTERS), 0.0), TAU)
    r = maxpool2(r)
    r = avgpool3(r)
    return r[0::2] + r[1::2]


def scale_img(img, sc, cx=13.5, cy=14.0):
    if sc == 1.0:
        return img
    xs = (XX - cx) / sc + cx
    ys = (YY - cy) / sc + cy
    x0 = np.clip(np.floor(xs).astype(int), 0, 27)
    y0 = np.clip(np.floor(ys).astype(int), 0, 27)
    x1 = np.clip(x0 + 1, 0, 27)
    y1 = np.clip(y0 + 1, 0, 27)
    fx = np.clip(xs - x0, 0, 1)
    fy = np.clip(ys - y0, 0, 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def variant_img(cls, vi):
    ang = V_ANGLES[vi // len(V_SCALES)]
    sc = V_SCALES[vi % len(V_SCALES)]
    return scale_img(rot_img(draw(SPECS[cls]), ang), sc)


# ---------------------------------------------------------------- synthetic hand drawings
BG, SKIN, CREASE, SLEEVE = 0.35, 0.78, 0.50, 0.18
YY, XX = np.mgrid[0:28, 0:28].astype(float)


def cap(x0, y0, x1, y1, r):
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    if L2 < 1e-9:
        d = np.hypot(XX - x0, YY - y0)
    else:
        u = np.clip(((XX - x0) * vx + (YY - y0) * vy) / L2, 0, 1)
        d = np.hypot(XX - (x0 + u * vx), YY - (y0 + u * vy))
    return d <= r


def ell(cx, cy, rx, ry):
    return ((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2 <= 1.0


def ann(cx, cy, rx, ry, irx, iry):
    return ell(cx, cy, rx, ry) & ~ell(cx, cy, irx, iry)


def draw(spec):
    img = np.full((28, 28), BG)
    dark = np.zeros((28, 28), bool)
    for pr in spec:
        kind = pr[0]
        if kind == "cap":
            img[cap(*pr[1:])] = SKIN
        elif kind == "ell":
            img[ell(*pr[1:])] = SKIN
        elif kind == "ann":
            img[ann(*pr[1:])] = SKIN
        elif kind == "dcap":
            dark |= cap(*pr[1:])
        elif kind == "sleeve":
            img[cap(*pr[1:])] = SLEEVE
    img[dark] = CREASE
    return img


def rot_img(img, deg, cx=13.5, cy=14.0):
    if deg == 0:
        return img
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    xs = ca * (XX - cx) - sa * (YY - cy) + cx
    ys = sa * (XX - cx) + ca * (YY - cy) + cy
    x0 = np.clip(np.floor(xs).astype(int), 0, 27)
    y0 = np.clip(np.floor(ys).astype(int), 0, 27)
    x1 = np.clip(x0 + 1, 0, 27)
    y1 = np.clip(y0 + 1, 0, 27)
    fx = np.clip(xs - x0, 0, 1)
    fy = np.clip(ys - y0, 0, 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


WRIST_D = ("cap", 13, 22, 13, 29, 4.0)
SLEEVE_D = ("sleeve", 13, 27, 13, 31, 4.6)
WRIST_L = ("cap", 8, 18, -2, 21, 3.6)
SLEEVE_L = ("sleeve", 1, 20, -4, 21.5, 4.2)
WRIST_R = ("cap", 20, 15, 30, 17, 3.6)
SLEEVE_R = ("sleeve", 27, 16, 32, 17.5, 4.2)
WRIST_TR = ("cap", 19, 10, 28, 2, 3.6)
SLEEVE_TR = ("sleeve", 26, 4, 30, 0, 4.2)
WRIST_BL = ("cap", 9, 20, 3, 28, 3.6)
SLEEVE_BL = ("sleeve", 4, 26, 1, 30, 4.2)

SPECS = {
    0: [("ell", 13, 16, 6.5, 6.5),
        ("cap", 9, 10.5, 9.5, 13, 1.9), ("cap", 12.5, 9.8, 13, 13, 1.9),
        ("cap", 16, 10.2, 16.5, 13, 1.9),
        ("cap", 19.5, 11, 20.5, 19, 2.2), WRIST_D, SLEEVE_D],
    1: [("ell", 13, 18, 6.0, 5.0),
        ("cap", 7.5, 16, 6.5, 4.5, 1.4), ("cap", 11.1, 15, 10.6, 3.0, 1.4),
        ("cap", 14.7, 15, 14.7, 3.0, 1.4), ("cap", 18.3, 16, 18.8, 4.5, 1.4),
        ("dcap", 9, 17.5, 17, 19.0, 0.8), WRIST_D, SLEEVE_D],
    2: [("cap", 9, 9, 14, 5.5, 2.2), ("cap", 14, 5.5, 18.5, 8.5, 2.0),
        ("cap", 18.5, 8.5, 19.0, 11.5, 1.8),
        ("ell", 9.5, 14, 3.6, 4.2),
        ("cap", 10, 18.5, 16.5, 16.5, 2.0), ("cap", 16.5, 16.5, 19.0, 15.0, 1.7),
        WRIST_BL, SLEEVE_BL],
    3: [("cap", 16, 14, 17.2, 3.0, 1.6), ("ann", 12.5, 17.0, 5.2, 4.6, 2.4, 2.0),
        ("cap", 15.5, 13.5, 14.5, 15.0, 1.6), WRIST_D, SLEEVE_D],
    4: [("ell", 13, 17, 6.0, 5.5),
        ("cap", 8, 10.5, 9, 14, 1.7), ("cap", 11.2, 9.6, 12.2, 13.5, 1.7),
        ("cap", 14.6, 9.6, 15.6, 13.5, 1.7), ("cap", 17.8, 10.5, 18.8, 14, 1.7),
        ("dcap", 8.5, 14.5, 18.5, 15.0, 0.7), WRIST_D, SLEEVE_D],
    5: [("ann", 17, 17, 4.6, 4.4, 2.0, 1.9),
        ("cap", 14, 14, 9.5, 5.0, 1.5), ("cap", 15.5, 13, 13.5, 4.0, 1.5),
        ("cap", 17.5, 13, 18.2, 5.0, 1.5), WRIST_BL, SLEEVE_BL],
    6: [("ell", 12, 16, 5.5, 5.0),
        ("cap", 14, 9.5, 24.5, 8.5, 1.6),
        ("cap", 13, 12.5, 21.0, 11.8, 1.4), ("dcap", 14, 11.0, 21, 10.4, 0.6),
        WRIST_L, SLEEVE_L],
    7: [("ell", 11, 16, 5.0, 4.6),
        ("cap", 13, 8.5, 24.0, 7.5, 1.4), ("cap", 13, 11.6, 24.0, 10.6, 1.4),
        WRIST_L, SLEEVE_L],
    8: [("ell", 14, 17, 5.5, 5.5),
        ("cap", 9.5, 14, 7.5, 4.0, 1.5),
        ("cap", 15, 11, 15.5, 14, 1.8), ("cap", 18, 12, 18.5, 15, 1.8),
        ("cap", 20.0, 13, 21.0, 19, 2.0), WRIST_D, SLEEVE_D],
    9: [("ell", 13, 18, 5.5, 4.6),
        ("cap", 15, 15, 18.5, 4.0, 1.5), ("cap", 11, 15, 7.0, 5.0, 1.5),
        ("cap", 13, 16, 13.2, 8.0, 1.4), WRIST_D, SLEEVE_D],
    10: [("ell", 13, 18, 5.5, 5.0),
         ("cap", 12, 15, 11.0, 3.0, 1.6), ("cap", 16, 16.5, 25.0, 15.0, 1.8),
         WRIST_D, SLEEVE_D],
    11: [("ell", 13, 17, 6.0, 5.5),
         ("cap", 9, 10, 10, 14, 1.8), ("cap", 12.5, 9, 13.5, 13, 1.8),
         ("cap", 16, 10, 17, 14, 1.8),
         ("dcap", 9, 14.5, 17, 15.2, 0.9), WRIST_D, SLEEVE_D],
    12: [("ell", 13, 17, 6.0, 5.5),
         ("cap", 10.5, 10, 11.5, 14, 1.8), ("cap", 14.5, 9.5, 15.5, 13.5, 1.8),
         ("dcap", 10, 14.5, 16, 15.0, 0.9), WRIST_D, SLEEVE_D],
    13: [("ann", 14, 14, 7.0, 7.0, 3.2, 3.2), WRIST_BL, SLEEVE_BL],
    14: [("ell", 16, 14, 5.0, 4.6),
         ("cap", 14, 9.5, 4.0, 8.5, 1.5),
         ("cap", 13.5, 15, 11.0, 22.5, 1.5),
         ("cap", 14, 12.2, 6.0, 11.4, 1.3), ("dcap", 13, 10.8, 6, 10.0, 0.6),
         WRIST_R, SLEEVE_R],
    15: [("ell", 16, 12, 5.0, 4.6),
         ("cap", 14, 15, 9.5, 23.0, 1.7),
         ("cap", 16.5, 16, 13.0, 23.5, 1.4), ("dcap", 14.6, 16.5, 11.5, 22.5, 0.6),
         WRIST_TR, SLEEVE_TR],
    16: [("ell", 13, 18, 5.5, 4.6),
         ("cap", 12, 15, 16.0, 4.0, 1.5), ("cap", 15, 15, 11.5, 4.0, 1.5),
         WRIST_D, SLEEVE_D],
    17: [("ell", 13, 16, 6.5, 6.0),
         ("cap", 9, 10.0, 9.5, 12.5, 1.8), ("cap", 12.5, 9.4, 13, 12.5, 1.8),
         ("cap", 16, 9.8, 16.5, 12.5, 1.8),
         ("dcap", 8, 15.0, 17, 16.0, 1.0), WRIST_D, SLEEVE_D],
    18: [("ell", 13, 16, 6.5, 6.0),
         ("cap", 9, 10.0, 9.5, 12.5, 1.8), ("cap", 12.5, 9.4, 13, 12.5, 1.8),
         ("cap", 16, 9.8, 16.5, 12.5, 1.8),
         ("cap", 17, 10.0, 23.0, 9.2, 1.8), WRIST_D, SLEEVE_D],
    19: [("ell", 13, 18, 5.5, 4.6),
         ("cap", 11, 15, 10.0, 3.0, 1.5), ("cap", 14.6, 15, 14.2, 3.0, 1.5),
         WRIST_D, SLEEVE_D],
    20: [("ell", 13, 18, 5.5, 4.6),
         ("cap", 11, 15, 7.0, 3.0, 1.5), ("cap", 15, 15, 19.0, 3.0, 1.5),
         WRIST_D, SLEEVE_D],
    21: [("ell", 13, 18, 5.5, 4.6),
         ("cap", 10.5, 15, 6.0, 4.0, 1.4), ("cap", 13.5, 14, 13.0, 3.0, 1.4),
         ("cap", 16.5, 15, 20.0, 4.0, 1.4), WRIST_D, SLEEVE_D],
    22: [("ell", 13, 17, 6.0, 5.5),
         ("cap", 12, 12, 13.5, 8.0, 1.7), ("cap", 13.5, 8.0, 17.0, 10.0, 1.7),
         WRIST_D, SLEEVE_D],
    23: [("ell", 13, 17, 5.5, 5.0),
         ("cap", 10, 14, 6.0, 5.0, 1.6), ("cap", 16, 16.5, 25.0, 14.0, 1.8),
         WRIST_D, SLEEVE_D],
}

FILTERS = edge_filters()


BORDER = np.ones((TK, TK), bool)
BORDER[1:-1, 1:-1] = False


def templates_for(cls):
    """-> list over variants of (P (12,10,10), Q (12,10,10)) weight maps."""
    out = []
    for vi in range(N_VAR):
        b = contour(variant_img(cls, vi))
        w = b[:, TOFF:TOFF + TK, TOFF:TOFF + TK]
        mx = max(w.max(), 1e-6)                  # self-normalisation by its own peak
        Pc, Pr = w[:6] > P_REL * mx, w[6:] > P_REL * mx
        den = max(Pc.sum() + W_RIDGE * Pr.sum(), 1.0)   # mean-coverage normalisation
        Tp = np.zeros_like(w)
        Tp[:6] = np.where(Pc, 1.0 / den, 0.0)
        Tp[6:] = np.where(Pr, W_RIDGE / den, 0.0)
        Qc = (w[:6] < Q_REL * mx) & (dil3(Pc.astype(float)) < 0.5) & ~BORDER
        Qr = (w[6:] < Q_REL * mx) & (dil3(Pr.astype(float)) < 0.5) & ~BORDER
        Tq = np.zeros_like(w)
        Tq[:6] = np.where(Qc, -GAMMA_C / max(Qc.sum(), 1.0), 0.0)
        Tq[6:] = np.where(Qr, -GAMMA_R / max(Qr.sum(), 1.0), 0.0)
        out.append((Tp, Tq))
    return out


def build():
    rows = 24 * N_VAR
    W2 = np.zeros((rows, 24, TK, TK))   # on dilated presence d
    W2n = np.zeros((rows, 24, TK, TK))  # on presence b (absence penalty)
    W3 = np.zeros((rows, 24, TK, TK))   # on b: uniform amount penalty
    for c in range(24):
        for v, (Tp, Tq) in enumerate(templates_for(c)):
            row = c * N_VAR + v
            W2[row, 0::2] = Tp
            W2[row, 1::2] = Tp
            W2n[row, 0::2] = Tq
            W2n[row, 1::2] = Tq
            W3[row] = -GAMMA2 / (12 * TK * TK)
    b2 = np.zeros(rows)

    W1 = FILTERS.reshape(24, 1, FSIZE, FSIZE).copy()
    b1 = np.zeros(24)
    Wcl1 = np.eye(24).reshape(24, 24, 1, 1)
    bcl1 = np.full(24, -TAU)
    Wcl3 = -np.eye(24).reshape(24, 24, 1, 1)
    bcl3 = np.zeros(24)

    Wf = np.zeros((24, rows))
    for c in range(24):
        for v in range(N_VAR):
            Wf[c, c * N_VAR + v] = 1.0 / N_VAR
    bf = np.zeros(24)

    net = {"layers": [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "c1", "op": "conv", "inputs": ["n0"], "out": 24,
         "kernel": [FSIZE, FSIZE], "stride": 1, "padding": FSIZE // 2},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "cl1", "op": "conv", "inputs": ["r1"], "out": 24,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "cl2", "op": "relu", "inputs": ["cl1"]},
        {"name": "cl3", "op": "conv", "inputs": ["cl2"], "out": 24,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "q1", "op": "add", "inputs": ["r1", "cl3"]},
        {"name": "p1", "op": "maxpool", "inputs": ["q1"], "size": [2, 2], "stride": [2, 2]},
        {"name": "s1", "op": "avgpool", "inputs": ["p1"], "size": [3, 3], "stride": [1, 1]},
        {"name": "d1", "op": "maxpool", "inputs": ["p1"], "size": [3, 3], "stride": [1, 1]},
        {"name": "c2", "op": "conv", "inputs": ["d1"], "out": rows,
         "kernel": [TK, TK], "stride": 1, "padding": 0},
        {"name": "c2n", "op": "conv", "inputs": ["s1"], "out": rows,
         "kernel": [TK, TK], "stride": 1, "padding": 0},
        {"name": "c3", "op": "conv", "inputs": ["s1"], "out": rows,
         "kernel": [TK, TK], "stride": 1, "padding": 0},
        {"name": "a2", "op": "add", "inputs": ["c2", "c2n", "c3"]},
        {"name": "p2", "op": "maxpool", "inputs": ["a2"], "size": [3, 3], "stride": [3, 3]},
        {"name": "r2", "op": "reshape", "inputs": ["p2"], "shape": [rows]},
        {"name": "fc", "op": "dense", "inputs": ["r2"], "out": 24},
    ]}
    weights = {"c1.W": W1.astype(np.float32), "c1.b": b1.astype(np.float32),
               "c2.W": W2.astype(np.float32), "c2.b": b2.astype(np.float32),
               "c2n.W": W2n.astype(np.float32), "c2n.b": np.zeros(rows, np.float32),
               "c3.W": W3.astype(np.float32), "c3.b": np.zeros(rows, np.float32),
               "cl1.W": Wcl1.astype(np.float32), "cl1.b": bcl1.astype(np.float32),
               "cl3.W": Wcl3.astype(np.float32), "cl3.b": bcl3.astype(np.float32),
               "fc.W": Wf.astype(np.float32), "fc.b": bf.astype(np.float32)}
    return net, weights


def main(dst):
    import os
    net, weights = build()
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(dst, "weights.npz"), **weights)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
