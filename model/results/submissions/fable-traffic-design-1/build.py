"""Builds a hand-designed template-matching network for German traffic signs (no data, no training).
Every template is one sign drawn by this code from general knowledge of the sign shapes, normalised by itself alone."""
import sys, json, os
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
SS = 4                      # supersampling
N = 32
RED = (0.70, 0.10, 0.10); BLUE = (0.10, 0.25, 0.65); YEL = (0.90, 0.70, 0.10)
WHITE = (0.90, 0.90, 0.90); BLACK = (0.10, 0.10, 0.10); GREY = (0.32, 0.32, 0.32); GREEN = (0.1, 0.6, 0.3)
BG = (0.45, 0.45, 0.45)

DIG = {
    '0': [([(0.5, 0), (1, 0.22), (1, 0.78), (0.5, 1), (0, 0.78), (0, 0.22)], True)],
    '1': [([(0, 0.22), (1, 0), (1, 1)], False)],
    '2': [([(0, 0.22), (0.3, 0), (0.7, 0), (1, 0.22), (1, 0.45), (0, 1), (1, 1)], False)],
    '3': [([(0, 0.1), (0.4, 0), (0.7, 0), (1, 0.2), (1, 0.35), (0.6, 0.5), (1, 0.65), (1, 0.8), (0.7, 1), (0.3, 1), (0, 0.88)], False),
          ([(0.3, 0.5), (0.6, 0.5)], False)],
    '5': [([(1, 0), (0.1, 0), (0.05, 0.45), (0.6, 0.4), (1, 0.58), (1, 0.8), (0.7, 1), (0.3, 1), (0, 0.86)], False)],
    '6': [([(0.9, 0.1), (0.6, 0), (0.35, 0), (0, 0.3), (0, 0.75), (0.3, 1), (0.7, 1), (1, 0.8), (1, 0.6), (0.7, 0.45), (0.3, 0.45), (0, 0.6)], False)],
    '7': [([(0, 0), (1, 0), (0.4, 1)], False)],
    '8': [([(0.5, 0), (0.92, 0.1), (0.92, 0.38), (0.5, 0.48), (0.08, 0.38), (0.08, 0.1)], True),
          ([(0.5, 0.48), (1, 0.6), (1, 0.88), (0.5, 1), (0, 0.88), (0, 0.6)], True)],
    'S': [([(1, 0.15), (0.7, 0), (0.3, 0), (0, 0.2), (0.2, 0.45), (0.8, 0.55), (1, 0.8), (0.7, 1), (0.3, 1), (0, 0.85)], False)],
    'T': [([(0, 0), (1, 0)], False), ([(0.5, 0), (0.5, 1)], False)],
    'O': [([(0.5, 0), (1, 0.22), (1, 0.78), (0.5, 1), (0, 0.78), (0, 0.22)], True)],
    'P': [([(0, 1), (0, 0), (0.7, 0), (1, 0.15), (1, 0.4), (0.7, 0.55), (0, 0.55)], False)],
}


class Canvas:
    def __init__(self, cx, cy, s):
        g = (np.arange(N * SS) + 0.5) / SS
        X, Y = np.meshgrid(g, g)
        self.U = (X - cx) / s; self.V = (Y - cy) / s
        self.col = np.empty((N * SS, N * SS, 3)); self.col[:] = BG
        self.al = np.zeros((N * SS, N * SS))

    def fill(self, m, c, sign=True):
        self.col[m] = c
        if sign: self.al[m] = 1.0

    def disc(self, cx, cy, r): return (self.U - cx) ** 2 + (self.V - cy) ** 2 <= r * r
    def rect(self, x0, x1, y0, y1): return (self.U >= x0) & (self.U <= x1) & (self.V >= y0) & (self.V <= y1)

    def seg(self, x0, y0, x1, y1, w):
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy + 1e-12
        t = np.clip(((self.U - x0) * dx + (self.V - y0) * dy) / L2, 0, 1)
        return (self.U - x0 - t * dx) ** 2 + (self.V - y0 - t * dy) ** 2 <= (w / 2) ** 2

    def pline(self, pts, w, closed=False):
        m = np.zeros(self.U.shape, bool)
        P = list(pts) + ([pts[0]] if closed else [])
        for a, b in zip(P[:-1], P[1:]): m |= self.seg(a[0], a[1], b[0], b[1], w)
        return m

    def poly(self, pts):
        m = np.zeros(self.U.shape, bool)
        for i in range(len(pts)):
            x0, y0 = pts[i]; x1, y1 = pts[i - 1]
            if y0 == y1: continue
            m ^= ((y0 > self.V) != (y1 > self.V)) & (self.U < (x1 - x0) * (self.V - y0) / (y1 - y0) + x0)
        return m

    def glyph(self, ch, x0, y0, w, h, sw):
        m = np.zeros(self.U.shape, bool)
        for pts, closed in DIG[ch]:
            m |= self.pline([(x0 + px * w, y0 + py * h) for px, py in pts], sw, closed)
        return m

    def text(self, s, h=0.76, w=0.40, gap=0.12, sw=0.14, w1=0.13):
        ws = [w1 if ch == '1' else w for ch in s]
        tot = sum(ws) + gap * (len(s) - 1)
        x = -tot / 2; m = np.zeros(self.U.shape, bool)
        for ch, cw in zip(s, ws):
            m |= self.glyph(ch, x, -h / 2, cw, h, sw); x += cw + gap
        return m

    def car(self, cx):
        return self.rect(cx - 0.26, cx + 0.26, -0.02, 0.2) | self.rect(cx - 0.15, cx + 0.15, -0.2, 0.0) | self.disc(cx - 0.15, 0.22, 0.07) | self.disc(cx + 0.15, 0.22, 0.07)

    def truck(self, x0, x1):
        return self.rect(x0, x1, -0.3, 0.2) | self.disc(x0 + 0.12, 0.22, 0.08) | self.disc(x1 - 0.12, 0.22, 0.08)

    def out(self):
        c = self.col.reshape(N, SS, N, SS, 3).mean((1, 3)).transpose(2, 0, 1)
        a = self.al.reshape(N, SS, N, SS).mean((1, 3))
        return c, a


def scaled(pts, c, f): return [(c[0] + f * (x - c[0]), c[1] + f * (y - c[1])) for x, y in pts]
def mirror(pts): return [(-x, y) for x, y in pts]


def ring_sign(cv):
    cv.fill(cv.disc(0, 0, 1.0), RED); cv.fill(cv.disc(0, 0, 0.72), WHITE)


def end_sign(cv):
    cv.fill(cv.disc(0, 0, 1.0), BLACK); cv.fill(cv.disc(0, 0, 0.90), WHITE)


def band(cv): cv.fill(cv.seg(-0.62, 0.62, 0.62, -0.62, 0.25) & cv.disc(0, 0, 0.9), GREY)


def tri_sign(cv):
    T = [(0, -1), (1, 1), (-1, 1)]
    cv.fill(cv.poly(T), RED); cv.fill(cv.poly(scaled(T, (0, 1 / 3), 0.60)), WHITE)


def blue_sign(cv):
    cv.fill(cv.disc(0, 0, 1.0), WHITE); cv.fill(cv.disc(0, 0, 0.92), BLUE)


def arrow_turn(cv, sgn):
    P = [(-0.2, 0.6), (-0.2, -0.1), (0.2, -0.1)]; H = [(0.2, -0.45), (0.65, -0.1), (0.2, 0.25)]
    if sgn < 0: P, H = mirror(P), mirror(H)
    cv.fill(cv.pline(P, 0.2) | cv.poly(H), WHITE)


def arrow_fork(cv, sgn):
    S = [(-0.2, 0.65), (-0.2, -0.2)]; H1 = [(-0.5, -0.2), (0.1, -0.2), (-0.2, -0.7)]
    B = [(-0.2, 0.4), (0.1, 0.15), (0.35, 0.15)]; H2 = [(0.35, -0.1), (0.7, 0.15), (0.35, 0.4)]
    if sgn < 0: S, H1, B, H2 = mirror(S), mirror(H1), mirror(B), mirror(H2)
    cv.fill(cv.pline(S, 0.18) | cv.poly(H1) | cv.pline(B, 0.15) | cv.poly(H2), WHITE)


def arrow_keep(cv, sgn):
    S = [(-0.45, -0.45), (0.2, 0.2)]; H = [(0.55, 0.55), (0.40, -0.10), (-0.10, 0.40)]
    if sgn < 0: S, H = mirror(S), mirror(H)
    cv.fill(cv.pline(S, 0.22) | cv.poly(H), WHITE)


def draw(k, cx, cy, s):
    cv = Canvas(cx, cy, s)
    speed = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}
    if k in speed:
        ring_sign(cv); t = speed[k]
        cv.fill(cv.text(t) if len(t) == 2 else cv.text(t, w=0.33, gap=0.10, sw=0.12), BLACK)
    elif k == 6:
        end_sign(cv); cv.fill(cv.text('80'), GREY); band(cv)
    elif k == 9:
        ring_sign(cv); cv.fill(cv.car(-0.31), RED); cv.fill(cv.car(0.31), BLACK)
    elif k == 10:
        ring_sign(cv); cv.fill(cv.truck(-0.62, -0.08), RED); cv.fill(cv.car(0.33), BLACK)
    elif k == 15:
        ring_sign(cv)
    elif k == 16:
        ring_sign(cv)
        cv.fill(cv.rect(-0.15, 0.52, -0.3, 0.15) | cv.rect(-0.5, -0.15, -0.1, 0.15) | cv.disc(-0.3, 0.2, 0.1) | cv.disc(0.3, 0.2, 0.1), BLACK)
    elif k == 32:
        end_sign(cv); band(cv)
    elif k == 41:
        end_sign(cv); cv.fill(cv.car(-0.31) | cv.car(0.31), GREY); band(cv)
    elif k == 42:
        end_sign(cv); cv.fill(cv.truck(-0.62, -0.08) | cv.car(0.33), GREY); band(cv)
    elif k == 12:
        D = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        cv.fill(cv.poly(D), BLACK); cv.fill(cv.poly(scaled(D, (0, 0), 0.93)), WHITE); cv.fill(cv.poly(scaled(D, (0, 0), 0.55)), YEL)
    elif k == 13:
        T = [(-1, -1), (1, -1), (0, 1)]
        cv.fill(cv.poly(T), RED); cv.fill(cv.poly(scaled(T, (0, -1 / 3), 0.50)), WHITE)
    elif k == 14:
        O = [(1, 0.414), (0.414, 1), (-0.414, 1), (-1, 0.414), (-1, -0.414), (-0.414, -1), (0.414, -1), (1, -0.414)]
        cv.fill(cv.poly(O), WHITE); cv.fill(cv.poly(scaled(O, (0, 0), 0.90)), RED)
        m = np.zeros(cv.U.shape, bool)
        for ch, x0 in zip('STOP', (-0.75, -0.35, 0.05, 0.45)): m |= cv.glyph(ch, x0, -0.33, 0.30, 0.66, 0.10)
        cv.fill(m, WHITE)
    elif k == 17:
        cv.fill(cv.disc(0, 0, 1.0), RED); cv.fill(cv.rect(-0.70, 0.70, -0.17, 0.17), WHITE)
    elif k in (33, 34, 35, 36, 37, 38, 39, 40):
        blue_sign(cv)
        if k == 33: arrow_turn(cv, 1)
        elif k == 34: arrow_turn(cv, -1)
        elif k == 35: cv.fill(cv.pline([(0, 0.65), (0, -0.15)], 0.2) | cv.poly([(-0.35, -0.15), (0.35, -0.15), (0, -0.7)]), WHITE)
        elif k == 36: arrow_fork(cv, 1)
        elif k == 37: arrow_fork(cv, -1)
        elif k == 38: arrow_keep(cv, 1)
        elif k == 39: arrow_keep(cv, -1)
        else:
            m = cv.disc(0, 0, 0.54) & ~cv.disc(0, 0, 0.34)
            for a in (90, 210, 330):
                r = np.deg2rad(a); m &= ~cv.seg(0.25 * np.cos(r), 0.25 * np.sin(r), 0.65 * np.cos(r), 0.65 * np.sin(r), 0.14)
            cv.fill(m, WHITE)
    else:                                   # red warning triangles
        tri_sign(cv)
        if k == 11:
            cv.fill(cv.pline([(0, 0.62), (0, -0.05)], 0.2) | cv.poly([(0, -0.38), (-0.2, 0.0), (0.2, 0.0)]) | cv.pline([(-0.33, 0.25), (0.33, 0.25)], 0.12), BLACK)
        elif k == 18:
            cv.fill(cv.pline([(0, -0.2), (0, 0.38)], 0.13) | cv.disc(0, 0.60, 0.085), BLACK)
        elif k == 19:
            cv.fill(cv.pline([(0.15, 0.65), (0.15, 0.15), (-0.3, 0.15)], 0.14), BLACK)
        elif k == 20:
            cv.fill(cv.pline([(-0.15, 0.65), (-0.15, 0.15), (0.3, 0.15)], 0.14), BLACK)
        elif k == 21:
            cv.fill(cv.pline([(-0.15, -0.1), (-0.15, 0.2), (0.15, 0.35), (0.15, 0.65)], 0.13), BLACK)
        elif k == 22:
            cv.fill(cv.rect(-0.45, 0.45, 0.45, 0.56) | ((cv.disc(-0.2, 0.47, 0.16) | cv.disc(0.2, 0.47, 0.16)) & (cv.V <= 0.5)), BLACK)
        elif k == 23:
            cv.fill(cv.rect(-0.22, 0.22, 0.05, 0.32) | cv.rect(-0.14, 0.14, -0.1, 0.05) | cv.pline([(-0.25, 0.42), (-0.12, 0.66)], 0.07) | cv.pline([(0.1, 0.42), (0.25, 0.66)], 0.07), BLACK)
        elif k == 24:
            cv.fill(cv.pline([(-0.15, -0.1), (-0.15, 0.65)], 0.1) | cv.pline([(0.08, -0.1), (0.08, 0.2), (0.25, 0.4), (0.25, 0.65)], 0.1), BLACK)
        elif k == 25:
            cv.fill(cv.pline([(-0.22, 0.62), (0.0, 0.12)], 0.12) | cv.disc(0.05, -0.05, 0.09) | cv.poly([(0.1, 0.64), (0.52, 0.64), (0.36, 0.36)]) | cv.pline([(0.05, 0.2), (0.3, 0.5)], 0.06), BLACK)
        elif k == 26:
            cv.fill(cv.rect(-0.13, 0.13, -0.2, 0.7), BLACK)
            cv.fill(cv.disc(0, -0.05, 0.1), RED); cv.fill(cv.disc(0, 0.25, 0.1), YEL); cv.fill(cv.disc(0, 0.55, 0.1), GREEN)
        elif k == 27:
            cv.fill(cv.disc(0, -0.1, 0.09) | cv.pline([(0, 0.05), (0, 0.35)], 0.14) | cv.pline([(-0.18, 0.68), (0, 0.35), (0.18, 0.68)], 0.08) | cv.pline([(-0.2, 0.3), (0, 0.1), (0.2, 0.3)], 0.06), BLACK)
        elif k == 28:
            m = cv.disc(0.17, -0.05, 0.08) | cv.pline([(0.17, 0.08), (0.17, 0.38)], 0.12) | cv.pline([(0.05, 0.68), (0.17, 0.38), (0.32, 0.68)], 0.07)
            m |= cv.disc(-0.2, 0.12, 0.07) | cv.pline([(-0.2, 0.22), (-0.2, 0.44)], 0.11) | cv.pline([(-0.32, 0.68), (-0.2, 0.44), (-0.1, 0.68)], 0.07) | cv.pline([(-0.2, 0.3), (0.17, 0.2)], 0.05)
            cv.fill(m, BLACK)
        elif k == 29:
            m = (cv.disc(-0.25, 0.47, 0.18) & ~cv.disc(-0.25, 0.47, 0.11)) | (cv.disc(0.25, 0.47, 0.18) & ~cv.disc(0.25, 0.47, 0.11))
            m |= cv.pline([(-0.25, 0.47), (-0.05, 0.2), (0.2, 0.2), (0.25, 0.47)], 0.06) | cv.pline([(-0.05, 0.2), (0.0, 0.0)], 0.1) | cv.disc(0.02, -0.1, 0.07)
            cv.fill(m, BLACK)
        elif k == 30:
            m = np.zeros(cv.U.shape, bool)
            for a in (90, 30, 150):
                r = np.deg2rad(a); m |= cv.seg(-0.36 * np.cos(r), 0.3 - 0.36 * np.sin(r), 0.36 * np.cos(r), 0.3 + 0.36 * np.sin(r), 0.08)
            cv.fill(m, BLACK)
        elif k == 31:
            cv.fill(cv.pline([(-0.3, 0.42), (0.22, 0.15)], 0.16) | cv.pline([(0.22, 0.15), (0.34, -0.08)], 0.08) | cv.pline([(-0.3, 0.42), (-0.46, 0.64)], 0.06) | cv.pline([(0.18, 0.2), (0.4, 0.42)], 0.06), BLACK)
    return cv.out()


# ---- front end: 3x3 binomial blur + colour opponent transform (formula filters) ----
BLUR = np.outer([1, 2, 1], [1, 2, 1]) / 16.0
MIX = np.array([[1 / 3, 1 / 3, 1 / 3],      # luminance
                [1.0, -1.0, 0.0],           # red - green
                [-0.5, -0.5, 1.0]])         # blue - yellow
W1 = (MIX[:, :, None, None] * BLUR[None, None]).astype(np.float64)


def front(img):
    p = np.pad(img, ((0, 0), (1, 1), (1, 1)))
    b = sum(BLUR[dy, dx] * p[:, dy:dy + N, dx:dx + N] for dy in range(3) for dx in range(3))
    return np.einsum('oc,chw->ohw', MIX, b)


SIZES = (9.5, 10.5, 11.5, 12.5, 13.5)       # half sizes of the sign in pixels (sign 19..27 px wide, from the border rule)
SHIFTS = (-1.0, 0.0, 1.0)
K = len(SIZES) * len(SHIFTS) ** 2
T = np.zeros((43 * K, 3 * N * N), np.float32)
for k in range(43):
    v = 0
    for s in SIZES:
        for dy in SHIFTS:
            for dx in SHIFTS:
                img, al = draw(k, 16.0 + dx, 16.0 + dy, s)
                f = front(img)
                m = (al > 0.05).astype(np.float64)
                f = (f - (f * m).sum((1, 2), keepdims=True) / m.sum()) * m     # zero mean inside its own sign mask, zero outside
                f = f / np.sqrt((f ** 2).sum())                                 # each template normalised by itself alone
                T[k * K + v] = f.reshape(-1); v += 1

net = {"layers": [
    {"name": "c1", "op": "conv", "inputs": ["image"], "out": 3, "kernel": [3, 3], "stride": 1, "padding": 1},
    {"name": "tm", "op": "dense", "inputs": ["c1"], "out": 43 * K},
    {"name": "r1", "op": "reshape", "inputs": ["tm"], "shape": [1, 43, K]},
    {"name": "mx", "op": "maxpool", "inputs": ["r1"], "size": [1, K], "stride": [1, K]},
    {"name": "scores", "op": "reshape", "inputs": ["mx"], "shape": [43]},
]}
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "network.json"), "w") as fh: json.dump(net, fh)
np.savez(os.path.join(OUT, "weights.npz"), **{"c1.W": W1.astype(np.float32), "c1.b": np.zeros(3, np.float32),
                                              "tm.W": T, "tm.b": np.zeros(43 * K, np.float32)})
