"""Builds a hand-designed template-matching network for German traffic signs.
No data is read: every weight comes from sign drawings typed below from general knowledge of the signs
(red ring, triangle, blue disc with arrow, digits as strokes ...) and from formula filters.
Usage: python3 build.py OUT_DIR
"""
import sys, json
import numpy as np
from pathlib import Path

N, SS = 32, 3
RED = (0.70, 0.10, 0.12); WHITE = (0.92, 0.92, 0.92); BLACK = (0.12, 0.12, 0.12)
BLUE = (0.10, 0.28, 0.65); YELLOW = (0.95, 0.75, 0.10); BG = 0.45
DGRAY = (0.30, 0.30, 0.30); MGRAY = (0.42, 0.42, 0.42)
LRED = (0.80, 0.15, 0.10); LYEL = (0.90, 0.75, 0.10); LGRN = (0.10, 0.60, 0.20)
SCALES = [0.58, 0.66, 0.74, 0.82, 0.90]          # sign half-size as a fraction of half the frame
SHIFTS = [-3.0, -1.5, 0.0, 1.5, 3.0]             # centre offsets in pixels
K = len(SCALES) * len(SHIFTS) ** 2
WC = 1.0                                         # weight of the colour-opponent channels


class Cv:
    def __init__(s, cx, cy, half, mirror=False):
        c = (np.arange(N * SS) + 0.5) / SS
        X, Y = np.meshgrid((c - cx) / half, (c - cy) / half)
        s.X = -X if mirror else X; s.Y = Y
        s.rgb = np.full((3,) + X.shape, BG); s.m = np.zeros(X.shape, bool)

    def paint(s, m, col):
        for i in range(3): s.rgb[i][m] = col[i]

    def disc(s, cx, cy, r): return (s.X - cx) ** 2 + (s.Y - cy) ** 2 < r * r

    def rect(s, x0, y0, x1, y1): return (s.X >= x0) & (s.X <= x1) & (s.Y >= y0) & (s.Y <= y1)

    def poly(s, pts):
        ins = np.zeros(s.X.shape, bool); n = len(pts)
        for i in range(n):
            (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
            if y1 == y2: continue
            ins ^= ((y1 > s.Y) != (y2 > s.Y)) & (s.X < (x2 - x1) * (s.Y - y1) / (y2 - y1) + x1)
        return ins

    def seg(s, p, q, w):
        dx, dy = q[0] - p[0], q[1] - p[1]; L = dx * dx + dy * dy + 1e-12
        t = np.clip(((s.X - p[0]) * dx + (s.Y - p[1]) * dy) / L, 0, 1)
        return (s.X - p[0] - t * dx) ** 2 + (s.Y - p[1] - t * dy) ** 2 < (w / 2) ** 2

    def line(s, pts, w):
        m = np.zeros(s.X.shape, bool)
        for i in range(len(pts) - 1): m |= s.seg(pts[i], pts[i + 1], w)
        return m

    def ell(s, cx, cy, rx, ry, w):
        a = np.linspace(0, 2 * np.pi, 25)
        return s.line([(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in a], w)

    def head(s, tip, d, length, hw):
        n = np.hypot(*d); dx, dy = d[0] / n, d[1] / n
        bx, by = tip[0] - dx * length, tip[1] - dy * length
        return s.poly([tip, (bx - dy * hw, by + dx * hw), (bx + dy * hw, by - dx * hw)])


# glyphs as strokes in a unit box (x right, y down)
GLYPH = {
    '1': [[(0.1, 0.3), (0.75, 0.0), (0.75, 1.0)]],
    '2': [[(0.0, 0.25), (0.15, 0.06), (0.5, 0.0), (0.85, 0.07), (1.0, 0.28), (0.9, 0.48), (0.0, 1.0), (1.0, 1.0)]],
    '3': [[(0.0, 0.12), (0.3, 0.0), (0.7, 0.0), (1.0, 0.2), (0.9, 0.42), (0.5, 0.48), (0.9, 0.55), (1.0, 0.75), (0.75, 1.0), (0.3, 1.0), (0.0, 0.88)]],
    '5': [[(0.95, 0.0), (0.15, 0.0), (0.08, 0.45), (0.5, 0.38), (0.85, 0.45), (1.0, 0.68), (0.85, 0.93), (0.5, 1.0), (0.15, 0.97), (0.0, 0.85)]],
    '6': [[(0.9, 0.06), (0.55, 0.0), (0.2, 0.15), (0.0, 0.5), (0.0, 0.72), (0.15, 0.95), (0.5, 1.0), (0.85, 0.93), (1.0, 0.7), (0.85, 0.45), (0.5, 0.38), (0.15, 0.48), (0.0, 0.65)]],
    '7': [[(0.0, 0.0), (1.0, 0.0), (0.4, 1.0)]],
    'S': [[(1.0, 0.15), (0.7, 0.0), (0.3, 0.0), (0.0, 0.2), (0.3, 0.45), (0.7, 0.55), (1.0, 0.8), (0.7, 1.0), (0.3, 1.0), (0.0, 0.85)]],
    'T': [[(0.0, 0.0), (1.0, 0.0)], [(0.5, 0.0), (0.5, 1.0)]],
    'P': [[(0.0, 1.0), (0.0, 0.0), (0.7, 0.0), (1.0, 0.15), (1.0, 0.4), (0.7, 0.55), (0.0, 0.55)]],
}


def glyph(cv, ch, x0, y0, w, h, sw):
    if ch in ('0', 'O'): return cv.ell(x0 + w / 2, y0 + h / 2, w / 2, h / 2, sw)
    if ch == '8': return cv.ell(x0 + w / 2, y0 + 0.25 * h, 0.42 * w, 0.25 * h, sw) | cv.ell(x0 + w / 2, y0 + 0.72 * h, 0.5 * w, 0.28 * h, sw)
    m = np.zeros(cv.X.shape, bool)
    for pl in GLYPH[ch]: m |= cv.line([(x0 + u * w, y0 + v * h) for u, v in pl], sw)
    return m


def text(cv, t, h, dw, gap, sw, cy=0.0):
    ws = [dw * (0.5 if ch == '1' else 1.0) for ch in t]
    x = -(sum(ws) + gap * (len(t) - 1)) / 2; m = np.zeros(cv.X.shape, bool)
    for ch, w in zip(t, ws):
        m |= glyph(cv, ch, x, cy - h / 2, w, h, sw); x += w + gap
    return m


def car(cv, cx, cy, w):   # car seen from behind
    return (cv.rect(cx - 0.32 * w, cy - 0.42 * w, cx + 0.32 * w, cy - 0.05 * w) | cv.rect(cx - 0.5 * w, cy - 0.1 * w, cx + 0.5 * w, cy + 0.28 * w)
            | cv.rect(cx - 0.45 * w, cy + 0.28 * w, cx - 0.22 * w, cy + 0.42 * w) | cv.rect(cx + 0.22 * w, cy + 0.28 * w, cx + 0.45 * w, cy + 0.42 * w))


def truck(cv, cx, cy, w):  # truck seen from behind
    return (cv.rect(cx - 0.5 * w, cy - 0.6 * w, cx + 0.5 * w, cy + 0.3 * w)
            | cv.rect(cx - 0.45 * w, cy + 0.3 * w, cx - 0.2 * w, cy + 0.45 * w) | cv.rect(cx + 0.2 * w, cy + 0.3 * w, cx + 0.45 * w, cy + 0.45 * w))


def truck_side(cv):
    return (cv.rect(-0.2, -0.3, 0.5, 0.15) | cv.rect(-0.5, -0.12, -0.24, 0.15) | cv.rect(-0.5, 0.15, 0.5, 0.22)
            | cv.disc(-0.33, 0.26, 0.1) | cv.disc(0.3, 0.26, 0.1))


def figure(cv, cx, top, h):
    return (cv.disc(cx, top + 0.1 * h, 0.1 * h) | cv.line([(cx, top + 0.24 * h), (cx, top + 0.6 * h)], 0.2 * h)
            | cv.line([(cx - 0.22 * h, top + h), (cx, top + 0.6 * h), (cx + 0.22 * h, top + h)], 0.1 * h)
            | cv.line([(cx - 0.25 * h, top + 0.52 * h), (cx, top + 0.3 * h), (cx + 0.25 * h, top + 0.52 * h)], 0.08 * h))


def scaled(pts, c, k): return [(c[0] + k * (x - c[0]), c[1] + k * (y - c[1])) for x, y in pts]


def red_ring(cv):
    cv.m = cv.disc(0, 0, 1.0); cv.paint(cv.m, WHITE); cv.paint(cv.disc(0, 0, 0.96), RED); cv.paint(cv.disc(0, 0, 0.70), WHITE)


def up_tri(cv):
    T = [(0.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]; c = (0.0, 1 / 3)
    cv.m = cv.poly(T); cv.paint(cv.m, WHITE); cv.paint(cv.poly(scaled(T, c, 0.94)), RED); cv.paint(cv.poly(scaled(T, c, 0.64)), WHITE)


def end_disc(cv):
    cv.m = cv.disc(0, 0, 1.0); cv.paint(cv.m, WHITE); cv.paint(cv.disc(0, 0, 0.97), DGRAY); cv.paint(cv.disc(0, 0, 0.90), WHITE)


def band(cv, hw, col):
    cv.paint((np.abs(cv.X + cv.Y) / np.sqrt(2) < hw) & cv.disc(0, 0, 0.90), col)


def blue_disc(cv):
    cv.m = cv.disc(0, 0, 1.0); cv.paint(cv.m, WHITE); cv.paint(cv.disc(0, 0, 0.94), BLUE)


SPEED = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}


def speed_text(cv, t):
    return text(cv, t, 0.74, 0.40, 0.13, 0.13) if len(t) == 2 else text(cv, t, 0.66, 0.32, 0.11, 0.11)


def draw(k, cx, cy, half):
    cv = Cv(cx, cy, half, mirror=k in (20, 34, 37, 39))
    if k in SPEED:
        red_ring(cv); cv.paint(speed_text(cv, SPEED[k]), BLACK)
    elif k == 9:
        red_ring(cv); cv.paint(car(cv, -0.3, 0.0, 0.5), RED); cv.paint(car(cv, 0.3, 0.0, 0.5), BLACK)
    elif k == 10:
        red_ring(cv); cv.paint(truck(cv, -0.3, 0.0, 0.5), RED); cv.paint(car(cv, 0.32, 0.06, 0.44), BLACK)
    elif k == 15:
        red_ring(cv)
    elif k == 16:
        red_ring(cv); cv.paint(truck_side(cv), BLACK)
    elif k == 17:
        cv.m = cv.disc(0, 0, 1.0); cv.paint(cv.m, WHITE); cv.paint(cv.disc(0, 0, 0.96), RED); cv.paint(cv.rect(-0.72, -0.17, 0.72, 0.17), WHITE)
    elif k == 14:
        o = lambda r: (np.abs(cv.X) < r) & (np.abs(cv.Y) < r) & (np.abs(cv.X) + np.abs(cv.Y) < r * np.sqrt(2))
        cv.m = o(1.0); cv.paint(cv.m, WHITE); cv.paint(o(0.93), RED); cv.paint(text(cv, 'STOP', 0.62, 0.32, 0.11, 0.10), WHITE)
    elif k == 13:
        T = [(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]; c = (0.0, -1 / 3)
        cv.m = cv.poly(T); cv.paint(cv.m, WHITE); cv.paint(cv.poly(scaled(T, c, 0.94)), RED); cv.paint(cv.poly(scaled(T, c, 0.56)), WHITE)
    elif k == 12:
        d = np.abs(cv.X) + np.abs(cv.Y)
        cv.m = d < 1.0; cv.paint(cv.m, WHITE); cv.paint(d < 0.62, YELLOW)
    elif k in (6, 32, 41, 42):
        end_disc(cv)
        if k == 6: cv.paint(text(cv, '80', 0.74, 0.40, 0.13, 0.13), DGRAY); band(cv, 0.13, MGRAY)
        elif k == 32: band(cv, 0.13, DGRAY)
        elif k == 41: cv.paint(car(cv, -0.3, 0.0, 0.5) | car(cv, 0.3, 0.0, 0.5), MGRAY); band(cv, 0.13, DGRAY)
        else: cv.paint(truck(cv, -0.3, 0.0, 0.5) | car(cv, 0.32, 0.06, 0.44), MGRAY); band(cv, 0.13, DGRAY)
    elif 33 <= k <= 40:
        blue_disc(cv)
        if k in (33, 34):
            m = cv.line([(-0.3, 0.62), (-0.3, 0.05), (-0.2, -0.12), (-0.02, -0.18), (0.27, -0.18)], 0.2) | cv.head((0.68, -0.18), (1, 0), 0.42, 0.33)
        elif k == 35:
            m = cv.line([(0.0, 0.66), (0.0, -0.25)], 0.2) | cv.head((0.0, -0.72), (0, -1), 0.5, 0.36)
        elif k in (36, 37):
            m = (cv.line([(-0.22, 0.66), (-0.22, -0.3)], 0.16) | cv.head((-0.22, -0.72), (0, -1), 0.42, 0.27)
                 | cv.line([(-0.22, 0.5), (-0.1, 0.2), (0.08, 0.1), (0.32, 0.1)], 0.16) | cv.head((0.7, 0.1), (1, 0), 0.4, 0.27))
        elif k in (38, 39):
            m = cv.line([(-0.5, -0.5), (0.2, 0.2)], 0.22) | cv.head((0.6, 0.6), (1, 1), 0.55, 0.36)
        else:
            m = np.zeros(cv.X.shape, bool)
            for j in range(3):
                a0 = np.pi / 2 + j * 2 * np.pi / 3
                aa = np.linspace(a0, a0 - np.radians(75), 8)     # arcs running anticlockwise as seen on the sign
                m |= cv.line([(0.48 * np.cos(a), 0.48 * np.sin(a)) for a in aa], 0.15)
                e = aa[-1]; p = (0.48 * np.cos(e), 0.48 * np.sin(e)); t = (np.sin(e), -np.cos(e))
                m |= cv.head((p[0] + 0.26 * t[0], p[1] + 0.26 * t[1]), t, 0.3, 0.2)
        cv.paint(m, WHITE)
    else:
        up_tri(cv)
        if k == 11:
            m = cv.poly([(0.0, -0.32), (-0.22, 0.1), (0.22, 0.1)]) | cv.rect(-0.1, 0.0, 0.1, 0.66) | cv.rect(-0.36, 0.28, 0.36, 0.39)
        elif k == 18:
            m = cv.rect(-0.075, -0.22, 0.075, 0.38) | cv.disc(0.0, 0.55, 0.09)
        elif k in (19, 20):
            m = cv.line([(0.2, 0.66), (0.2, 0.25), (0.1, 0.1), (-0.28, 0.05)], 0.14)
        elif k == 21:
            m = cv.line([(0.16, 0.66), (0.16, 0.45), (-0.14, 0.2), (-0.14, -0.08)], 0.13)
        elif k == 22:
            m = cv.rect(-0.42, 0.45, 0.42, 0.58) | ((cv.disc(-0.2, 0.47, 0.17) | cv.disc(0.2, 0.47, 0.17)) & (cv.Y < 0.5))
        elif k == 23:
            m = (cv.rect(-0.2, -0.05, 0.2, 0.3) | cv.rect(-0.13, -0.2, 0.13, -0.05)
                 | cv.line([(-0.14, 0.34), (-0.26, 0.48), (-0.16, 0.56), (-0.28, 0.68)], 0.07) | cv.line([(0.2, 0.34), (0.08, 0.48), (0.18, 0.56), (0.06, 0.68)], 0.07))
        elif k == 24:
            m = cv.line([(-0.15, -0.1), (-0.15, 0.66)], 0.1) | cv.line([(0.3, 0.66), (0.3, 0.38), (0.1, 0.12), (0.1, -0.1)], 0.1)
        elif k == 25:
            m = (cv.disc(0.0, -0.1, 0.08) | cv.line([(-0.02, 0.02), (-0.12, 0.35)], 0.14) | cv.line([(-0.3, 0.66), (-0.12, 0.35), (0.0, 0.66)], 0.09)
                 | cv.line([(-0.02, 0.1), (0.22, 0.45)], 0.07) | (cv.disc(0.32, 0.68, 0.2) & (cv.Y < 0.7)))
        elif k == 26:
            m = cv.rect(-0.11, -0.2, 0.11, 0.64)
        elif k == 27:
            m = figure(cv, 0.0, -0.25, 0.92)
        elif k == 28:
            m = figure(cv, -0.17, -0.15, 0.82) | figure(cv, 0.2, 0.05, 0.62) | cv.line([(-0.1, 0.25), (0.15, 0.3)], 0.06)
        elif k == 29:
            m = (cv.ell(-0.25, 0.47, 0.16, 0.16, 0.06) | cv.ell(0.25, 0.47, 0.16, 0.16, 0.06)
                 | cv.line([(-0.25, 0.47), (-0.05, 0.2), (0.2, 0.2), (0.03, 0.47), (-0.05, 0.2)], 0.06) | cv.line([(0.25, 0.47), (0.17, 0.08), (0.08, 0.08)], 0.06)
                 | cv.line([(-0.12, 0.12), (0.02, 0.12)], 0.06))
        elif k == 30:
            m = cv.disc(0.0, 0.32, 0.1)
            for a in (0.0, np.pi / 3, 2 * np.pi / 3):
                m |= cv.line([(-0.3 * np.cos(a), 0.32 - 0.3 * np.sin(a)), (0.3 * np.cos(a), 0.32 + 0.3 * np.sin(a))], 0.08)
        else:  # 31 leaping deer
            m = (cv.line([(-0.3, 0.45), (0.18, 0.18)], 0.17) | cv.line([(0.18, 0.18), (0.32, -0.02)], 0.09) | cv.disc(0.36, -0.02, 0.07)
                 | cv.line([(0.15, 0.22), (0.4, 0.42)], 0.06) | cv.line([(-0.28, 0.45), (-0.44, 0.68)], 0.06) | cv.line([(0.3, -0.05), (0.26, -0.25)], 0.05))
        cv.paint(m, BLACK)
        if k == 26:
            cv.paint(cv.disc(0.0, -0.07, 0.08), LRED); cv.paint(cv.disc(0.0, 0.22, 0.08), LYEL); cv.paint(cv.disc(0.0, 0.51, 0.08), LGRN)
    ds = lambda a: a.reshape(a.shape[:-2] + (N, SS, N, SS)).mean(axis=(-3, -1))
    return ds(cv.rgb), ds(cv.m.astype(float))


B3 = np.outer([1, 2, 1], [1, 2, 1]) / 16.0                      # binomial blur
A = np.array([[1 / 3, 1 / 3, 1 / 3], [WC, -WC / 2, -WC / 2], [-WC / 2, -WC / 2, WC]])   # luminance, red and blue opponent


def features(rgb):
    p = np.pad(rgb, ((0, 0), (1, 1), (1, 1)))
    bl = sum(B3[dy, dx] * p[:, dy:dy + N, dx:dx + N] for dy in range(3) for dx in range(3))
    return np.tensordot(A, bl, axes=(1, 0))


def template(k, cx, cy, half):
    rgb, m = draw(k, cx, cy, half)
    f = features(rgb - rgb.mean())                              # the network removes the global mean before the features
    f[0] -= (m * f[0]).sum() / m.sum()                          # brightness offset inside the sign does not matter
    f *= m[None]
    return (f / np.sqrt((f ** 2).sum())).reshape(-1)


def build(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    W = np.zeros((43 * K, 3 * N * N), np.float32)
    for k in range(43):
        v = 0
        for s in SCALES:
            for dy in SHIFTS:
                for dx in SHIFTS:
                    W[k * K + v] = template(k, N / 2 + dx, N / 2 + dy, s * N / 2); v += 1
    Wf = (A[:, :, None, None] * B3[None, None]).astype(np.float32)
    net = {"layers": [
        {"name": "flat", "op": "reshape", "inputs": ["image"], "shape": [1, 3 * N, N]},
        {"name": "gnorm", "op": "norm", "inputs": ["flat"]},
        {"name": "img", "op": "reshape", "inputs": ["gnorm"], "shape": [3, N, N]},
        {"name": "feat", "op": "conv", "inputs": ["img"], "out": 3, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "match", "op": "dense", "inputs": ["feat"], "out": 43 * K},
        {"name": "poses", "op": "reshape", "inputs": ["match"], "shape": [1, 43, K]},
        {"name": "best", "op": "maxpool", "inputs": ["poses"], "size": [1, K], "stride": [1, K]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [43]},
    ]}
    (out / "network.json").write_text(json.dumps(net, indent=1))
    np.savez(out / "weights.npz", **{"feat.W": Wf, "feat.b": np.zeros(3, np.float32), "match.W": W, "match.b": np.zeros(43 * K, np.float32)})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
