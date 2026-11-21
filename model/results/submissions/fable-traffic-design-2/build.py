"""Writes network.json and weights.npz for a hand-designed traffic sign template network.
Every number comes from this file: sign shapes typed from general knowledge of German road signs,
rendered with simple geometry, and turned into matched filters. No data is read, nothing is fitted."""
import sys, json, os
import numpy as np

N, SS = 32, 4
RED = (0.72, 0.12, 0.14); BLUE = (0.10, 0.28, 0.68); YEL = (0.95, 0.72, 0.10)
WHT = (0.93, 0.93, 0.93); BLK = (0.08, 0.08, 0.08); GRY = (0.40, 0.40, 0.40)
DGR = (0.30, 0.30, 0.30); GRN = (0.10, 0.60, 0.30)
CW = 1.3                                             # weight of the two colour-opponent channels
SCALES = [9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5]   # half-size of the sign box in pixels
BLURS = [0.7, 1.3]                                   # template blur in pixels


def grid(a):
    c = (np.arange(N * SS) + 0.5) / SS - 0.5
    u = (c - 15.5) / a
    return np.meshgrid(u, u)


def disc(U, V, cx, cy, r): return (U - cx) ** 2 + (V - cy) ** 2 <= r * r
def rect(U, V, x0, y0, x1, y1): return (U >= x0) & (U <= x1) & (V >= y0) & (V <= y1)


def cap(U, V, x0, y0, x1, y1, t):
    dx, dy = x1 - x0, y1 - y0; L2 = dx * dx + dy * dy
    s = np.clip(((U - x0) * dx + (V - y0) * dy) / L2, 0, 1) if L2 > 0 else 0.0
    return (U - x0 - s * dx) ** 2 + (V - y0 - s * dy) ** 2 <= (t / 2) ** 2


def line(U, V, pts, t):
    m = np.zeros(U.shape, bool)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]): m |= cap(U, V, x0, y0, x1, y1, t)
    return m


def poly(U, V, pts):
    pos = np.ones(U.shape, bool); neg = np.ones(U.shape, bool)
    for i in range(len(pts)):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % len(pts)]
        c = (x1 - x0) * (V - y0) - (y1 - y0) * (U - x0)
        pos &= c >= 0; neg &= c <= 0
    return pos | neg


def ell(cx, cy, rx, ry, n=16):
    return [(cx + rx * np.cos(2 * np.pi * k / n), cy + ry * np.sin(2 * np.pi * k / n)) for k in range(n + 1)]


# glyphs as strokes in a unit box (x right, y down)
G = {
    '0': [ell(0.5, 0.5, 0.5, 0.5)],
    '1': [[(0.0, 0.28), (1.0, 0.0), (1.0, 1.0)]],
    '2': [[(0.0, 0.25), (0.15, 0.06), (0.5, 0.0), (0.85, 0.06), (1.0, 0.27), (0.85, 0.5), (0.0, 1.0), (1.0, 1.0)]],
    '3': [[(0.0, 0.12), (0.3, 0.0), (0.7, 0.0), (1.0, 0.22), (0.7, 0.46), (0.35, 0.46)],
          [(0.7, 0.46), (1.0, 0.73), (0.7, 1.0), (0.3, 1.0), (0.0, 0.88)]],
    '5': [[(0.95, 0.0), (0.1, 0.0), (0.05, 0.45), (0.5, 0.38), (0.9, 0.5), (1.0, 0.7), (0.85, 0.93), (0.5, 1.0), (0.15, 0.95), (0.0, 0.85)]],
    '6': [[(0.9, 0.08), (0.55, 0.0), (0.2, 0.15), (0.0, 0.5), (0.0, 0.75), (0.2, 0.95), (0.5, 1.0), (0.8, 0.95), (1.0, 0.72),
           (0.8, 0.48), (0.5, 0.42), (0.2, 0.5), (0.0, 0.7)]],
    '7': [[(0.0, 0.0), (1.0, 0.0), (0.35, 1.0)]],
    '8': [ell(0.5, 0.25, 0.42, 0.25), ell(0.5, 0.73, 0.5, 0.27)],
    'S': [[(1.0, 0.12), (0.6, 0.0), (0.2, 0.05), (0.0, 0.25), (0.3, 0.48), (0.7, 0.55), (1.0, 0.75), (0.8, 0.95), (0.4, 1.0), (0.0, 0.88)]],
    'T': [[(0.0, 0.0), (1.0, 0.0)], [(0.5, 0.0), (0.5, 1.0)]],
    'O': [ell(0.5, 0.5, 0.5, 0.5)],
    'P': [[(0.0, 1.0), (0.0, 0.0), (0.7, 0.0), (1.0, 0.25), (0.7, 0.5), (0.0, 0.5)]],
}


def text(U, V, s, t, h, w, gap, w1=None):
    widths = [(w1 if (ch == '1' and w1) else w) for ch in s]
    x = -(sum(widths) + gap * (len(s) - 1)) / 2; m = np.zeros(U.shape, bool)
    for ch, wd in zip(s, widths):
        for pl in G[ch]:
            m |= line(U, V, [(x + px * wd, -h / 2 + py * h) for px, py in pl], t)
        x += wd + gap
    return m


def number(U, V, s):
    if len(s) == 2: return text(U, V, s, 0.13, 0.68, 0.34, 0.20)
    return text(U, V, s, 0.11, 0.62, 0.27, 0.17, w1=0.14)


def car(U, V, x0, x1):
    return rect(U, V, x0, 0.0, x1, 0.2) | rect(U, V, x0 + 0.08, -0.14, x1 - 0.08, 0.0)


def band(U, V): return cap(U, V, -0.68, 0.68, 0.68, -0.68, 0.2)


def sign(c, U, V):
    """Returns (silhouette, [(mask, colour), ...]) for class c in box coordinates -1..1 (v points down)."""
    if c in (34, 37, 39, 20):                      # mirror images of 33, 36, 38, 19
        return sign({34: 33, 37: 36, 39: 38, 20: 19}[c], -U, V)
    if c in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16):   # red ring, white disc
        sil = disc(U, V, 0, 0, 1.0); L = [(sil, RED), (disc(U, V, 0, 0, 0.76), WHT)]
        if c in (0, 1, 2, 3, 4, 5, 7, 8):
            L.append((number(U, V, {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}[c]), BLK))
        elif c == 9:
            L += [(car(U, V, -0.56, -0.10), RED), (car(U, V, 0.10, 0.56), BLK)]
        elif c == 10:
            L += [(rect(U, V, -0.60, -0.30, -0.08, 0.20), RED), (car(U, V, 0.10, 0.56), BLK)]
        elif c == 16:
            L.append((rect(U, V, -0.15, -0.30, 0.55, 0.15) | rect(U, V, -0.55, -0.08, -0.15, 0.15)
                      | disc(U, V, -0.35, 0.2, 0.1) | disc(U, V, 0.3, 0.2, 0.1), BLK))
        return sil, L
    if c in (6, 32, 41, 42):                       # "end of" signs: white disc, grey content, diagonal band
        sil = disc(U, V, 0, 0, 1.0); L = [(sil, GRY), (disc(U, V, 0, 0, 0.93), WHT)]
        if c == 6: L.append((number(U, V, '80'), GRY))
        if c == 41: L += [(car(U, V, -0.56, -0.10), DGR), (car(U, V, 0.10, 0.56), DGR)]
        if c == 42: L += [(rect(U, V, -0.60, -0.30, -0.08, 0.20), DGR), (car(U, V, 0.10, 0.56), DGR)]
        L.append((band(U, V), DGR))
        return sil, L
    if c == 17:                                    # no entry
        sil = disc(U, V, 0, 0, 1.0)
        return sil, [(sil, RED), (rect(U, V, -0.72, -0.15, 0.72, 0.15), WHT)]
    if c == 14:                                    # stop
        A, B = np.abs(U), np.abs(V)
        sil = (A <= 1) & (B <= 1) & (A + B <= 1.414)
        return sil, [(sil, WHT), ((A <= 0.92) & (B <= 0.92) & (A + B <= 1.301), RED),
                     (text(U, V, 'STOP', 0.11, 0.52, 0.27, 0.16), WHT)]
    if c == 12:                                    # priority road
        D = np.abs(U) + np.abs(V); sil = D <= 1.0
        return sil, [(sil, WHT), (D <= 0.60, YEL)]
    if c == 13:                                    # yield
        sil = poly(U, V, [(-1, -1), (1, -1), (0, 1)]); k = 0.58
        return sil, [(sil, RED), (poly(U, V, [(-k, -1 / 3 - k * 2 / 3), (k, -1 / 3 - k * 2 / 3), (0, -1 / 3 + k * 4 / 3)]), WHT)]
    if 33 <= c <= 40:                              # blue discs with white arrows
        sil = disc(U, V, 0, 0, 1.0); L = [(sil, WHT), (disc(U, V, 0, 0, 0.94), BLUE)]
        if c == 33:
            m = line(U, V, [(-0.22, 0.62), (-0.22, 0.0), (-0.1, -0.17), (0.08, -0.22), (0.3, -0.22)], 0.2) \
                | poly(U, V, [(0.68, -0.22), (0.26, -0.56), (0.26, 0.12)])
        elif c == 35:
            m = line(U, V, [(0, 0.68), (0, -0.2)], 0.2) | poly(U, V, [(0, -0.72), (0.36, -0.18), (-0.36, -0.18)])
        elif c == 36:
            m = line(U, V, [(-0.22, 0.65), (-0.22, -0.25)], 0.16) | poly(U, V, [(-0.22, -0.72), (0.06, -0.28), (-0.5, -0.28)]) \
                | line(U, V, [(-0.22, 0.35), (-0.1, 0.12), (0.1, 0.05), (0.32, 0.05)], 0.16) \
                | poly(U, V, [(0.68, 0.05), (0.3, -0.24), (0.3, 0.34)])
        elif c == 38:
            m = line(U, V, [(-0.48, -0.48), (0.2, 0.2)], 0.24) | poly(U, V, [(0.6, 0.6), (0.52, -0.08), (-0.08, 0.52)])
        else:                                      # 40 roundabout: three curved arrows
            m = np.zeros(U.shape, bool)
            for k in range(3):
                t0 = np.deg2rad(k * 120 + 100)
                arc = [(0.48 * np.cos(t0 - np.deg2rad(d)), 0.48 * np.sin(t0 - np.deg2rad(d))) for d in range(0, 81, 16)]
                t1, t2 = t0 - np.deg2rad(80), t0 - np.deg2rad(108)
                m |= line(U, V, arc, 0.16) | poly(U, V, [(0.48 * np.cos(t2), 0.48 * np.sin(t2)),
                                                           (0.26 * np.cos(t1), 0.26 * np.sin(t1)), (0.70 * np.cos(t1), 0.70 * np.sin(t1))])
        L.append((m, WHT))
        return sil, L
    # red-bordered warning triangles with a black symbol
    sil = poly(U, V, [(0, -1), (1, 1), (-1, 1)]); k = 0.62
    L = [(sil, RED), (poly(U, V, [(0, 1 / 3 - k * 4 / 3), (k, 1 / 3 + k * 2 / 3), (-k, 1 / 3 + k * 2 / 3)]), WHT)]
    if c == 11:
        m = line(U, V, [(0, 0.7), (0, 0.0)], 0.2) | poly(U, V, [(0, -0.36), (0.24, 0.02), (-0.24, 0.02)]) \
            | line(U, V, [(-0.36, 0.36), (0.36, 0.36)], 0.13)
    elif c == 18:
        m = line(U, V, [(0, -0.2), (0, 0.38)], 0.17) | disc(U, V, 0, 0.63, 0.1)
    elif c == 19:
        m = line(U, V, [(0.2, 0.72), (0.2, 0.22), (0.08, 0.1), (-0.3, 0.1)], 0.16)
    elif c == 21:
        m = line(U, V, [(0.17, 0.72), (0.17, 0.45), (-0.15, 0.18), (-0.15, -0.15)], 0.15)
    elif c == 22:
        m = rect(U, V, -0.5, 0.56, 0.5, 0.68) | ((disc(U, V, -0.2, 0.56, 0.19) | disc(U, V, 0.2, 0.56, 0.19)) & (V <= 0.6))
    elif c == 23:
        m = rect(U, V, -0.22, -0.05, 0.22, 0.28) | line(U, V, [(-0.12, 0.36), (-0.25, 0.52), (-0.12, 0.7)], 0.08) \
            | line(U, V, [(0.2, 0.36), (0.07, 0.52), (0.2, 0.7)], 0.08)
    elif c == 24:
        m = line(U, V, [(-0.16, -0.12), (-0.16, 0.72)], 0.13) | line(U, V, [(0.08, -0.12), (0.08, 0.18), (0.27, 0.42), (0.27, 0.72)], 0.13)
    elif c == 25:
        m = disc(U, V, -0.05, -0.1, 0.09) | line(U, V, [(-0.02, 0.04), (-0.2, 0.36)], 0.15) | line(U, V, [(-0.2, 0.36), (-0.32, 0.7)], 0.09) \
            | line(U, V, [(-0.2, 0.36), (-0.03, 0.7)], 0.09) | line(U, V, [(0.02, 0.14), (0.28, 0.5)], 0.07) \
            | (disc(U, V, 0.33, 0.72, 0.2) & (V <= 0.72))
    elif c == 26:
        L += [(rect(U, V, -0.13, -0.18, 0.13, 0.7), (0.2, 0.2, 0.2)), (disc(U, V, 0, -0.02, 0.09), RED),
              (disc(U, V, 0, 0.26, 0.09), YEL), (disc(U, V, 0, 0.54, 0.09), GRN)]
        return sil, L
    elif c == 27:
        m = disc(U, V, 0, -0.12, 0.09) | line(U, V, [(0, 0.03), (0, 0.36)], 0.16) | line(U, V, [(0, 0.36), (-0.18, 0.72)], 0.09) \
            | line(U, V, [(0, 0.36), (0.16, 0.72)], 0.09) | line(U, V, [(0, 0.06), (-0.2, 0.3)], 0.07) | line(U, V, [(0, 0.06), (0.2, 0.3)], 0.07)
    elif c == 28:
        m = disc(U, V, -0.18, 0.02, 0.08) | line(U, V, [(-0.18, 0.14), (-0.18, 0.42)], 0.14) | line(U, V, [(-0.18, 0.42), (-0.3, 0.72)], 0.08) \
            | line(U, V, [(-0.18, 0.42), (-0.06, 0.72)], 0.08) | disc(U, V, 0.2, 0.14, 0.07) | line(U, V, [(0.2, 0.25), (0.2, 0.46)], 0.12) \
            | line(U, V, [(0.2, 0.46), (0.1, 0.72)], 0.07) | line(U, V, [(0.2, 0.46), (0.32, 0.72)], 0.07) | line(U, V, [(-0.1, 0.25), (0.12, 0.32)], 0.06)
    elif c == 29:
        m = (disc(U, V, -0.27, 0.5, 0.225) & ~disc(U, V, -0.27, 0.5, 0.155)) | (disc(U, V, 0.27, 0.5, 0.225) & ~disc(U, V, 0.27, 0.5, 0.155)) \
            | line(U, V, [(-0.27, 0.5), (-0.08, 0.22), (0.18, 0.22), (0.27, 0.5)], 0.06) | line(U, V, [(-0.08, 0.22), (0.03, 0.5), (0.18, 0.22)], 0.06) \
            | line(U, V, [(-0.14, 0.14), (-0.02, 0.14)], 0.06) | line(U, V, [(0.12, 0.1), (0.24, 0.12)], 0.06)
    elif c == 30:
        m = disc(U, V, 0, 0.32, 0.1)
        for ang in (90, 30, 150):
            dx, dy = 0.3 * np.cos(np.deg2rad(ang)), 0.3 * np.sin(np.deg2rad(ang))
            m |= line(U, V, [(-dx, 0.32 - dy), (dx, 0.32 + dy)], 0.09)
    else:                                          # 31 leaping deer
        m = line(U, V, [(-0.2, 0.2), (0.18, 0.38)], 0.2) | line(U, V, [(-0.2, 0.2), (-0.36, 0.0)], 0.1) | line(U, V, [(-0.36, 0.0), (-0.3, -0.15)], 0.06) \
            | line(U, V, [(-0.18, 0.28), (-0.42, 0.42)], 0.07) | line(U, V, [(0.2, 0.42), (0.42, 0.7)], 0.08) | line(U, V, [(0.1, 0.42), (0.2, 0.68)], 0.07)
    L.append((m, BLK))
    return sil, L


def gauss(sig):
    i = np.arange(N)
    Gm = np.exp(-(i[:, None] - i[None, :]) ** 2 / (2 * sig * sig))
    return Gm / Gm.sum(1, keepdims=True)


def template(c, a, sig):
    U, V = grid(a)
    sil, layers = sign(c, U, V)
    img = np.zeros((3,) + U.shape)
    for m, col in layers:
        for k in range(3): img[k][m] = col[k]
    pool = lambda x: x.reshape(N, SS, N, SS).mean((1, 3))
    Gm = gauss(sig)
    P = np.stack([Gm @ pool(img[k]) @ Gm.T for k in range(3)])      # blurred colour times coverage
    w = Gm @ pool(sil.astype(float)) @ Gm.T                         # blurred coverage
    rgb = P / np.maximum(w, 1e-6)
    Lum = rgb.mean(0); C1 = rgb[0] - (rgb[1] + rgb[2]) / 2; C2 = rgb[2] - (rgb[0] + rgb[1]) / 2
    Lum = Lum - (w * Lum).sum() / w.sum()                           # brightness offset does not matter inside the sign
    T = np.stack([w * Lum, w * CW * C1, w * CW * C2])
    return T / np.sqrt((T ** 2).sum())


def main(out):
    os.makedirs(out, exist_ok=True)
    V = len(SCALES) * len(BLURS)
    W = np.zeros((43 * V, 3, N, N), np.float32)
    for c in range(43):
        v = 0
        for a in SCALES:
            for sig in BLURS:
                W[c * V + v] = template(c, a, sig); v += 1
    fw = np.array([[1 / 3, 1 / 3, 1 / 3], [CW, -CW / 2, -CW / 2], [-CW / 2, -CW / 2, CW]], np.float32).reshape(3, 3, 1, 1)
    weights = {"feat.W": fw, "feat.b": np.zeros(3, np.float32), "tm.W": W, "tm.b": np.zeros(43 * V, np.float32)}
    net = {"layers": [
        {"name": "flat", "op": "reshape", "inputs": ["image"], "shape": [1, 96, 32]},
        {"name": "gn", "op": "norm", "inputs": ["flat"]},
        {"name": "imgn", "op": "reshape", "inputs": ["gn"], "shape": [3, 32, 32]},
        {"name": "feat", "op": "conv", "inputs": ["imgn"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "tm", "op": "conv", "inputs": ["feat"], "out": 43 * V, "kernel": [32, 32], "stride": 1, "padding": 2},
        {"name": "pmax", "op": "maxpool", "inputs": ["tm"], "size": [5, 5], "stride": [5, 5]},
        {"name": "grp", "op": "reshape", "inputs": ["pmax"], "shape": [1, 43, V]},
        {"name": "vmax", "op": "maxpool", "inputs": ["grp"], "size": [1, V], "stride": [1, V]},
        {"name": "out", "op": "reshape", "inputs": ["vmax"], "shape": [43]},
    ]}
    with open(os.path.join(out, "network.json"), "w") as f: json.dump(net, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **weights)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
