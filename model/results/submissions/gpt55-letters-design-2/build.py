import json, sys, os, math
import numpy as np

N = 28
CLASSES = 24
SHIFTS = [(dx, dy) for dy in (-0.12, 0.04, 0.18) for dx in (-0.18, 0.0, 0.18)]
SCALES = [0.86, 1.0, 1.14]
ROTS = [-10.0, 0.0, 10.0]
POLARITIES = [1.0, -1.0]
MIRRORS = [1.0, -1.0]
VPER = len(SHIFTS) * len(SCALES) * len(ROTS) * len(POLARITIES) * len(MIRRORS)
NEDGE = 16


def grid(shift=(0.0, 0.0), scale=1.0, rot=0.0):
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    x = (xx + 0.5 - N / 2) / (N / 2)
    y = (yy + 0.5 - N / 2) / (N / 2)
    x = x - shift[0]
    y = y - shift[1]
    a = -rot * math.pi / 180.0
    ca, sa = math.cos(a), math.sin(a)
    xr = (ca * x - sa * y) / scale
    yr = (sa * x + ca * y) / scale
    return xr, yr


def capsule_field(x, y, x1, y1, x2, y2, r):
    vx, vy = x2 - x1, y2 - y1
    den = vx * vx + vy * vy + 1e-9
    t = ((x - x1) * vx + (y - y1) * vy) / den
    t = np.clip(t, 0.0, 1.0)
    px, py = x1 + t * vx, y1 + t * vy
    d = np.sqrt((x - px) ** 2 + (y - py) ** 2)
    return np.exp(-0.5 * (d / r) ** 4)


def ellipse_field(x, y, cx, cy, rx, ry, ang=0.0):
    a = ang * math.pi / 180.0
    ca, sa = math.cos(a), math.sin(a)
    X, Y = x - cx, y - cy
    xr = ca * X + sa * Y
    yr = -sa * X + ca * Y
    q = (xr / rx) ** 2 + (yr / ry) ** 2
    return np.exp(-0.5 * q ** 2)


def add_caps(pos, x, y, p1, p2, r, amp=1.0):
    pos += amp * capsule_field(x, y, p1[0], p1[1], p2[0], p2[1], r)


def add_ell(pos, x, y, cx, cy, rx, ry, amp=1.0, ang=0.0):
    pos += amp * ellipse_field(x, y, cx, cy, rx, ry, ang)


def vertical_finger(pos, x, y, x0, tip=-0.76, base=0.12, r=0.070, amp=1.0, slant=0.0):
    add_caps(pos, x, y, (x0, base), (x0 + slant, tip), r, amp)


def palm(pos, x, y, cx=0.0, cy=0.36, rx=0.40, ry=0.42, amp=0.85):
    add_ell(pos, x, y, cx, cy, rx, ry, amp)
    add_caps(pos, x, y, (-0.12, 0.72), (0.02, 1.12), 0.18, 0.45)


def fist(pos, x, y, cx=0.0, cy=0.25, amp=1.0):
    add_ell(pos, x, y, cx, cy + 0.08, 0.42, 0.42, 0.85 * amp)
    for fx in [-0.26, -0.08, 0.10, 0.27]:
        add_ell(pos, x, y, cx + fx, cy - 0.23, 0.12, 0.13, 0.65 * amp)
    add_caps(pos, x, y, (cx - 0.12, cy + 0.45), (cx + 0.02, 1.12), 0.18, 0.38 * amp)


def neg_absent_vertical(neg, x, y, xs, amp=0.55):
    for x0 in xs:
        add_caps(neg, x, y, (x0, 0.07), (x0, -0.75), 0.060, amp)


def gaps_between(neg, x, y, xs, amp=0.45, tip=-0.72, base=0.12):
    for x0 in xs:
        add_caps(neg, x, y, (x0, base), (x0, tip), 0.026, amp)


def signed_model(c, x, y):
    pos = np.zeros_like(x, dtype=np.float32)
    neg = np.zeros_like(x, dtype=np.float32)
    if c == 0:
        fist(pos, x, y); add_caps(pos, x, y, (-0.42, 0.38), (-0.42, -0.10), 0.090, 0.90)
        add_caps(neg, x, y, (-0.18, 0.08), (0.28, -0.05), 0.045, 0.35); neg_absent_vertical(neg, x, y, [-0.26, -0.08, 0.10, 0.27], 0.35)
    elif c == 1:
        palm(pos, x, y)
        for fx in [-0.25, -0.08, 0.09, 0.25]: vertical_finger(pos, x, y, fx, tip=-0.83 if fx in [-0.08, 0.09] else -0.75, r=0.075, amp=1.05)
        add_caps(pos, x, y, (-0.40, 0.32), (-0.15, 0.06), 0.080, 0.55); gaps_between(neg, x, y, [-0.165, 0.005, 0.17], 0.18)
    elif c == 2:
        add_ell(pos, x, y, -0.08, 0.05, 0.43, 0.73, 1.15); add_ell(neg, x, y, 0.02, 0.05, 0.26, 0.48, 1.25); add_ell(neg, x, y, 0.38, 0.05, 0.23, 0.38, 1.15); add_caps(pos, x, y, (-0.34, 0.68), (-0.18, 1.08), 0.19, 0.35)
    elif c == 3:
        palm(pos, x, y, cy=0.38, amp=0.65); vertical_finger(pos, x, y, 0.03, tip=-0.82, base=0.04, r=0.073, amp=1.25)
        add_ell(pos, x, y, -0.03, 0.24, 0.34, 0.30, 0.75); add_ell(neg, x, y, -0.03, 0.22, 0.16, 0.14, 0.95); neg_absent_vertical(neg, x, y, [-0.28, -0.13, 0.20, 0.34], 0.65)
    elif c == 4:
        fist(pos, x, y, cy=0.30); add_caps(pos, x, y, (-0.37, 0.36), (0.30, 0.18), 0.075, 0.80)
        for fx in [-0.24, -0.08, 0.08, 0.23]: add_ell(pos, x, y, fx, -0.05, 0.13, 0.11, 0.55)
        add_ell(neg, x, y, 0.00, -0.38, 0.38, 0.25, 0.80)
    elif c == 5:
        palm(pos, x, y)
        for fx, sl in [(-0.03, -0.05), (0.18, 0.02), (0.34, 0.05)]: vertical_finger(pos, x, y, fx, tip=-0.77, base=0.03, r=0.068, amp=1.0, slant=sl)
        add_ell(pos, x, y, -0.30, 0.05, 0.18, 0.21, 0.85); add_ell(neg, x, y, -0.30, 0.05, 0.080, 0.095, 1.25); gaps_between(neg, x, y, [0.08, 0.26], 0.45)
    elif c == 6:
        add_ell(pos, x, y, 0.33, 0.28, 0.34, 0.32, 0.85, -8); add_caps(pos, x, y, (0.20, -0.04), (-0.82, -0.09), 0.075, 1.20); add_caps(pos, x, y, (0.16, 0.20), (-0.55, 0.25), 0.055, 0.60); add_caps(pos, x, y, (0.25, 0.58), (0.48, 1.05), 0.17, 0.42); add_caps(neg, x, y, (0.18, 0.08), (-0.68, 0.06), 0.050, 0.80)
    elif c == 7:
        add_ell(pos, x, y, 0.32, 0.27, 0.34, 0.31, 0.80, -8); add_caps(pos, x, y, (0.18, -0.12), (-0.82, -0.18), 0.070, 1.10); add_caps(pos, x, y, (0.18, 0.04), (-0.80, -0.01), 0.070, 1.05); add_caps(pos, x, y, (0.25, 0.58), (0.48, 1.05), 0.17, 0.38); add_caps(neg, x, y, (0.12, -0.04), (-0.72, -0.09), 0.030, 0.55)
    elif c == 8:
        fist(pos, x, y); vertical_finger(pos, x, y, 0.27, tip=-0.78, base=0.05, r=0.066, amp=1.15); neg_absent_vertical(neg, x, y, [-0.25, -0.08, 0.08], 0.65); add_caps(pos, x, y, (-0.38, 0.38), (-0.12, 0.18), 0.080, 0.45)
    elif c == 9:
        palm(pos, x, y); vertical_finger(pos, x, y, -0.13, tip=-0.78, base=0.08, r=0.070, amp=1.05, slant=-0.12); vertical_finger(pos, x, y, 0.12, tip=-0.76, base=0.08, r=0.070, amp=1.05, slant=0.12); add_caps(pos, x, y, (-0.35, 0.38), (0.18, -0.10), 0.075, 0.85); gaps_between(neg, x, y, [0.0], 0.45); neg_absent_vertical(neg, x, y, [0.31], 0.45)
    elif c == 10:
        fist(pos, x, y); vertical_finger(pos, x, y, 0.03, tip=-0.82, base=0.02, r=0.074, amp=1.20); add_caps(pos, x, y, (-0.04, 0.26), (-0.78, 0.15), 0.085, 1.00); neg_absent_vertical(neg, x, y, [-0.25, 0.22, 0.34], 0.60)
    elif c == 11:
        fist(pos, x, y, cy=0.27)
        for fx in [-0.22, -0.03, 0.16]: add_caps(pos, x, y, (fx, -0.17), (fx + 0.03, 0.25), 0.080, 0.85)
        add_caps(neg, x, y, (-0.34, 0.30), (0.34, 0.12), 0.040, 0.45); add_caps(pos, x, y, (-0.45, 0.38), (-0.14, 0.30), 0.070, 0.35)
    elif c == 12:
        fist(pos, x, y, cy=0.28)
        for fx in [-0.13, 0.10]: add_caps(pos, x, y, (fx, -0.17), (fx + 0.03, 0.25), 0.085, 0.90)
        add_caps(neg, x, y, (-0.28, 0.28), (0.28, 0.12), 0.040, 0.40); add_ell(neg, x, y, 0.28, -0.06, 0.13, 0.18, 0.45)
    elif c == 13:
        add_ell(pos, x, y, 0.00, 0.06, 0.42, 0.58, 1.15); add_ell(neg, x, y, 0.00, 0.06, 0.22, 0.33, 1.45); add_caps(pos, x, y, (-0.14, 0.65), (0.02, 1.08), 0.18, 0.35)
    elif c == 14:
        add_ell(pos, x, y, 0.18, -0.03, 0.34, 0.32, 0.78, -20); add_caps(pos, x, y, (0.07, 0.10), (-0.24, 0.82), 0.070, 1.15); add_caps(pos, x, y, (0.26, 0.08), (0.02, 0.78), 0.070, 1.05); add_caps(pos, x, y, (-0.18, 0.00), (0.32, 0.42), 0.075, 0.70); add_caps(pos, x, y, (0.45, -0.05), (0.72, -0.60), 0.16, 0.35); add_caps(neg, x, y, (0.15, 0.22), (-0.08, 0.75), 0.030, 0.55)
    elif c == 15:
        add_ell(pos, x, y, 0.16, 0.03, 0.34, 0.32, 0.82, -20); add_caps(pos, x, y, (0.05, 0.12), (-0.32, 0.86), 0.075, 1.20); add_caps(pos, x, y, (-0.10, 0.00), (0.32, 0.38), 0.060, 0.58); add_caps(pos, x, y, (0.43, -0.05), (0.72, -0.58), 0.16, 0.35); add_caps(neg, x, y, (0.20, 0.14), (-0.12, 0.78), 0.055, 0.70)
    elif c == 16:
        palm(pos, x, y); add_caps(pos, x, y, (-0.11, 0.08), (0.10, -0.78), 0.064, 1.08); add_caps(pos, x, y, (0.10, 0.08), (-0.08, -0.76), 0.064, 1.08); add_caps(neg, x, y, (0.00, 0.02), (0.00, -0.62), 0.026, 0.40); neg_absent_vertical(neg, x, y, [-0.30, 0.30], 0.55)
    elif c == 17:
        fist(pos, x, y, cy=0.28); add_caps(pos, x, y, (-0.38, 0.34), (0.25, 0.18), 0.090, 0.95); add_ell(neg, x, y, -0.42, -0.02, 0.12, 0.24, 0.40); neg_absent_vertical(neg, x, y, [-0.25, -0.08, 0.10, 0.27], 0.32)
    elif c == 18:
        fist(pos, x, y, cy=0.27); add_caps(pos, x, y, (-0.42, 0.08), (0.05, 0.02), 0.090, 0.95); add_ell(pos, x, y, -0.12, -0.06, 0.13, 0.18, 0.45); add_caps(neg, x, y, (0.08, -0.15), (0.08, 0.18), 0.035, 0.45); neg_absent_vertical(neg, x, y, [0.24, -0.28], 0.35)
    elif c == 19:
        palm(pos, x, y); vertical_finger(pos, x, y, -0.08, tip=-0.81, base=0.07, r=0.073, amp=1.12); vertical_finger(pos, x, y, 0.08, tip=-0.81, base=0.07, r=0.073, amp=1.12); gaps_between(neg, x, y, [0.0], 0.20); neg_absent_vertical(neg, x, y, [-0.28, 0.28], 0.60)
    elif c == 20:
        palm(pos, x, y); vertical_finger(pos, x, y, -0.10, tip=-0.78, base=0.08, r=0.070, amp=1.10, slant=-0.18); vertical_finger(pos, x, y, 0.10, tip=-0.78, base=0.08, r=0.070, amp=1.10, slant=0.18); add_caps(neg, x, y, (0.00, 0.03), (0.00, -0.65), 0.050, 0.80); neg_absent_vertical(neg, x, y, [-0.32, 0.32], 0.45)
    elif c == 21:
        palm(pos, x, y); vertical_finger(pos, x, y, -0.20, tip=-0.76, base=0.08, r=0.066, amp=1.05, slant=-0.12); vertical_finger(pos, x, y, 0.00, tip=-0.83, base=0.04, r=0.066, amp=1.12); vertical_finger(pos, x, y, 0.20, tip=-0.76, base=0.08, r=0.066, amp=1.05, slant=0.12); gaps_between(neg, x, y, [-0.10, 0.10], 0.52); neg_absent_vertical(neg, x, y, [0.36], 0.35)
    elif c == 22:
        fist(pos, x, y); add_caps(pos, x, y, (0.02, 0.02), (0.00, -0.42), 0.075, 0.95); add_caps(pos, x, y, (0.00, -0.42), (-0.20, -0.56), 0.075, 0.78); add_ell(neg, x, y, 0.10, -0.68, 0.26, 0.18, 0.70); neg_absent_vertical(neg, x, y, [-0.25, 0.23], 0.50)
    elif c == 23:
        fist(pos, x, y); add_caps(pos, x, y, (-0.05, 0.28), (-0.80, 0.14), 0.085, 1.05); vertical_finger(pos, x, y, 0.28, tip=-0.76, base=0.04, r=0.068, amp=1.10, slant=0.08); neg_absent_vertical(neg, x, y, [-0.12, 0.08], 0.65)
    w = pos - 0.55 * neg
    w *= np.exp(-0.18 * (x * x + y * y))
    return w.astype(np.float32)


def make_edge_filters():
    k = 5
    ax = np.arange(k, dtype=np.float32) - (k - 1) / 2
    yy, xx = np.meshgrid(ax, ax, indexing='ij')
    sig = 1.05
    g = np.exp(-(xx * xx + yy * yy) / (2 * sig * sig))
    filters = []
    for j in range(NEDGE):
        th = 2 * math.pi * j / NEDGE
        # derivative of a Gaussian along the chosen direction; zero mean.
        f = (math.cos(th) * xx + math.sin(th) * yy) * g
        f = f - f.mean()
        f = f / (np.sqrt((f * f).sum()) + 1e-9)
        filters.append(f)
    W = np.asarray(filters, dtype=np.float32)[:, None, :, :]
    b = np.zeros((NEDGE,), dtype=np.float32)
    return W, b


def conv_single(x, W):
    # x is 28x28, W is (C,1,5,5); same padding.
    p = 2
    xp = np.pad(x, ((p, p), (p, p)))
    out = []
    for f in W[:, 0]:
        s = np.zeros_like(x, dtype=np.float32)
        for dy in range(5):
            for dx in range(5):
                s += f[dy, dx] * xp[dy:dy+N, dx:dx+N]
        out.append(s)
    return np.asarray(out, dtype=np.float32)


def pool2(z):
    c, h, w = z.shape
    return z.reshape(c, h//2, 2, w//2, 2).max(axis=(2,4))


def norm_img(x):
    x = x.astype(np.float32)
    return (x - x.mean()) / np.sqrt(((x - x.mean()) ** 2).mean() + 1e-5)


def norm_ch(z):
    mu = z.mean(axis=(1,2), keepdims=True)
    return (z - mu) / np.sqrt(((z - mu) ** 2).mean(axis=(1,2), keepdims=True) + 1e-5)


def template_feature(c, sh, sc, rt, pol, mir, edgeW):
    x, y = grid(sh, sc, rt)
    x = x * mir
    t = signed_model(c, x, y)
    # Signed fields include inner holes and interdigital gaps, not just outer outlines.
    t = pol * norm_img(t)
    e = conv_single(t, edgeW)
    e = np.maximum(e, 0.0)
    e = pool2(e)
    e = norm_ch(e)
    v = e.reshape(-1)
    v = v / (np.sqrt((v * v).sum()) + 1e-9)
    return v.astype(np.float32)



def make_weights():
    edgeW, edgeb = make_edge_filters()
    erows = []
    bias = []
    # Hand-set offsets: simple rounded outlines are less trusted, highly articulated signs are boosted.
    class_bias = np.array([0.15,0.35,-0.05,0.65,0.10,0.45,0.95,0.95,0.65,0.90,0.85,0.20,0.20,-1.60,-0.80,-0.60,0.95,-0.10,0.25,0.80,0.95,0.95,0.85,0.80], dtype=np.float32)
    for c in range(CLASSES):
        for sh in SHIFTS:
            for sc in SCALES:
                for rt in ROTS:
                    for pol in POLARITIES:
                        for mir in MIRRORS:
                            erows.append(template_feature(c, sh, sc, rt, pol, mir, edgeW))
                            bias.append(class_bias[c])
    EW = np.stack(erows, axis=0).astype(np.float32)
    b = np.asarray(bias, dtype=np.float32)
    return edgeW, edgeb, EW, b


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    net = {"layers": [
        {"name": "n", "op": "norm", "inputs": ["image"]},
        {"name": "edge", "op": "conv", "inputs": ["n"], "out": NEDGE, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "er", "op": "relu", "inputs": ["edge"]},
        {"name": "ep", "op": "maxpool", "inputs": ["er"], "size": [2, 2], "stride": [2, 2]},
        {"name": "en", "op": "norm", "inputs": ["ep"]},
        {"name": "template", "op": "dense", "inputs": ["en"], "out": CLASSES * VPER},
        {"name": "byclass", "op": "reshape", "inputs": ["template"], "shape": [CLASSES, VPER, 1]},
        {"name": "best", "op": "maxpool", "inputs": ["byclass"], "size": [VPER, 1], "stride": [VPER, 1]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [CLASSES]},
    ]}
    edgeW, edgeb, W, b = make_weights()
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, separators=(",", ":"))
    np.savez(os.path.join(out_dir, "weights.npz"), **{"edge.W": edgeW, "edge.b": edgeb, "template.W": W, "template.b": b})

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
