"""Hand-designed traffic-sign network: every weight comes from formulas and sign drawings typed here from
general knowledge of German road signs. Reads no files.   python3 build.py OUT_DIR"""
import sys, json
from pathlib import Path
import numpy as np

RED = (0.78, 0.10, 0.12); WHITE = (0.95, 0.95, 0.95); BLACK = (0.08, 0.08, 0.08); BLUE = (0.10, 0.30, 0.70)
YEL = (0.95, 0.75, 0.05); GREEN = (0.10, 0.60, 0.20); GRAY = (0.50, 0.50, 0.50); DARK = (0.22, 0.22, 0.22)
BG = (0.40, 0.40, 0.40)


# ---------------------------------------------------------------- drawing primitives (masks over coordinate grids)
def disc(X, Y, cx, cy, r): return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
def ell(X, Y, cx, cy, rx, ry): return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
def rect(X, Y, x0, y0, x1, y1): return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


def seg(X, Y, x0, y0, x1, y1, w):
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / (dx * dx + dy * dy), 0, 1)
    return (X - x0 - t * dx) ** 2 + (Y - y0 - t * dy) ** 2 <= (w / 2) ** 2


def pline(X, Y, pts, w):
    m = np.zeros(X.shape, bool)
    for (a, b), (c, d) in zip(pts[:-1], pts[1:]):
        m |= seg(X, Y, a, b, c, d, w)
    return m


def poly(X, Y, pts):
    m = np.zeros(X.shape, bool); n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        if y0 == y1: continue
        m ^= (Y >= min(y0, y1)) & (Y < max(y0, y1)) & (X < x0 + (Y - y0) * (x1 - x0) / (y1 - y0))
    return m


def ering(X, Y, cx, cy, rx, ry, t):
    return ell(X, Y, cx, cy, rx + t / 2, ry + t / 2) & ~ell(X, Y, cx, cy, max(rx - t / 2, 1e-3), max(ry - t / 2, 1e-3))


# ---------------------------------------------------------------- glyphs drawn as strokes in a unit box
GLYPH = {
    '1': [[(0.2, 0.3), (0.65, 0.0), (0.65, 1.0)]],
    '2': [[(0.05, 0.25), (0.2, 0.05), (0.5, 0.0), (0.8, 0.05), (0.95, 0.27), (0.85, 0.5), (0.05, 1.0), (1.0, 1.0)]],
    '3': [[(0.05, 0.12), (0.3, 0.0), (0.7, 0.0), (0.95, 0.22), (0.7, 0.47), (0.4, 0.47)],
          [(0.7, 0.47), (0.97, 0.73), (0.7, 1.0), (0.3, 1.0), (0.03, 0.86)]],
    '4': [[(0.75, 1.0), (0.75, 0.0), (0.0, 0.7), (1.0, 0.7)]],
    '5': [[(0.9, 0.0), (0.15, 0.0), (0.1, 0.45), (0.4, 0.38), (0.7, 0.4), (0.95, 0.65), (0.7, 1.0), (0.3, 1.0), (0.03, 0.85)]],
    '6': [[(0.85, 0.05), (0.5, 0.0), (0.15, 0.25), (0.03, 0.65), (0.25, 1.0), (0.7, 1.0), (0.97, 0.7), (0.7, 0.42),
           (0.3, 0.42), (0.05, 0.65)]],
    '7': [[(0.0, 0.0), (1.0, 0.0), (0.35, 1.0)]],
    'S': [[(1, 0.12), (0.75, 0), (0.25, 0), (0, 0.2), (0.25, 0.47), (0.75, 0.53), (1, 0.8), (0.75, 1), (0.25, 1), (0, 0.88)]],
    'T': [[(0, 0), (1, 0)], [(0.5, 0), (0.5, 1)]],
    'P': [[(0, 1), (0, 0), (0.7, 0), (1, 0.15), (1, 0.4), (0.7, 0.55), (0, 0.55)]],
}


def glyph(X, Y, ch, x0, y0, w, h, t):
    if ch in ('0', 'O'):
        return ering(X, Y, x0 + 0.5 * w, y0 + 0.5 * h, 0.5 * w, 0.5 * h, t)
    if ch == '8':
        return ering(X, Y, x0 + 0.5 * w, y0 + 0.26 * h, 0.42 * w, 0.26 * h, t) | ering(X, Y, x0 + 0.5 * w, y0 + 0.73 * h, 0.5 * w, 0.27 * h, t)
    m = np.zeros(X.shape, bool)
    for stroke in GLYPH[ch]:
        m |= pline(X, Y, [(x0 + a * w, y0 + b * h) for a, b in stroke], t)
    return m


def number(X, Y, s):
    if len(s) == 2:
        return glyph(X, Y, s[0], -0.43, -0.36, 0.36, 0.72, 0.13) | glyph(X, Y, s[1], 0.07, -0.36, 0.36, 0.72, 0.13)
    return (glyph(X, Y, s[0], -0.56, -0.32, 0.20, 0.64, 0.11) | glyph(X, Y, s[1], -0.22, -0.32, 0.27, 0.64, 0.11)
            | glyph(X, Y, s[2], 0.20, -0.32, 0.27, 0.64, 0.11))


def car(X, Y, cx, cy):  # car seen from behind
    return (rect(X, Y, cx - 0.22, cy - 0.02, cx + 0.22, cy + 0.20)
            | poly(X, Y, [(cx - 0.15, cy - 0.02), (cx - 0.10, cy - 0.20), (cx + 0.10, cy - 0.20), (cx + 0.15, cy - 0.02)])
            | rect(X, Y, cx - 0.20, cy + 0.20, cx - 0.10, cy + 0.28) | rect(X, Y, cx + 0.10, cy + 0.20, cx + 0.20, cy + 0.28))


def lorry(X, Y, cx, cy):  # truck seen from behind
    return (rect(X, Y, cx - 0.24, cy - 0.30, cx + 0.24, cy + 0.20)
            | rect(X, Y, cx - 0.22, cy + 0.20, cx - 0.10, cy + 0.28) | rect(X, Y, cx + 0.10, cy + 0.20, cx + 0.22, cy + 0.28))


def person(X, Y, cx, cy, s):
    return (disc(X, Y, cx, cy - 0.7 * s, 0.15 * s) | seg(X, Y, cx, cy - 0.5 * s, cx, cy + 0.1 * s, 0.28 * s)
            | seg(X, Y, cx, cy + 0.1 * s, cx - 0.35 * s, cy + 0.9 * s, 0.16 * s) | seg(X, Y, cx, cy + 0.1 * s, cx + 0.35 * s, cy + 0.9 * s, 0.16 * s)
            | seg(X, Y, cx, cy - 0.4 * s, cx - 0.4 * s, cy, 0.12 * s) | seg(X, Y, cx, cy - 0.4 * s, cx + 0.4 * s, cy, 0.12 * s))


# ---------------------------------------------------------------- the 43 signs, in sign coordinates (sign spans [-1, 1])
class Canvas:
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.rgb = np.empty((3,) + X.shape)
        for c in range(3): self.rgb[c] = BG[c]
        self.alpha = np.zeros(X.shape, bool)

    def paint(self, m, col):
        for c in range(3): self.rgb[c][m] = col[c]


TRI_H = 1.85  # triangle height in a crop stretched towards square (equilateral would be 1.73)


def red_ring(cv):
    d = disc(cv.X, cv.Y, 0, 0, 1.0); cv.alpha |= d
    cv.paint(d, RED); cv.paint(disc(cv.X, cv.Y, 0, 0, 0.74), WHITE)


def end_disc(cv):
    d = disc(cv.X, cv.Y, 0, 0, 1.0); cv.alpha |= d
    cv.paint(d, DARK); cv.paint(disc(cv.X, cv.Y, 0, 0, 0.94), WHITE)


def end_band(cv):
    cv.paint(seg(cv.X, cv.Y, -0.68, 0.68, 0.68, -0.68, 0.26) & disc(cv.X, cv.Y, 0, 0, 0.94), DARK)


def blue_disc(cv):
    d = disc(cv.X, cv.Y, 0, 0, 1.0); cv.alpha |= d
    cv.paint(d, WHITE); cv.paint(disc(cv.X, cv.Y, 0, 0, 0.95), BLUE)


def warn_tri(cv):
    h = TRI_H; outer = [(0, -h / 2), (-1, h / 2), (1, h / 2)]; cy = h / 6
    inner = [(x * 0.6, cy + (y - cy) * 0.6) for x, y in outer]
    m = poly(cv.X, cv.Y, outer); cv.alpha |= m
    cv.paint(m, RED); cv.paint(poly(cv.X, cv.Y, inner), WHITE)
    return cv.X / 0.38, (cv.Y - 0.27) / 0.38  # pictogram coordinates


SPEED = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}
MIRROR = {20: 19, 34: 33, 37: 36, 39: 38}


def draw(k, X, Y):
    if k in MIRROR:
        return draw(MIRROR[k], -X, Y)
    cv = Canvas(X, Y); P = cv.paint
    if k in SPEED:
        red_ring(cv); P(number(X, Y, SPEED[k]), BLACK)
    elif k == 6:
        end_disc(cv); P(number(X, Y, '80'), GRAY); end_band(cv)
    elif k == 9:
        red_ring(cv); P(car(X, Y, -0.30, 0.0), RED); P(car(X, Y, 0.30, 0.0), BLACK)
    elif k == 10:
        red_ring(cv); P(lorry(X, Y, -0.30, 0.0), RED); P(car(X, Y, 0.32, 0.05), BLACK)
    elif k == 12:
        m = np.abs(X) + np.abs(Y) <= 1.0; cv.alpha |= m
        P(m, WHITE); P(np.abs(X) + np.abs(Y) <= 0.66, YEL)
    elif k == 13:
        h = TRI_H; outer = [(-1, -h / 2), (1, -h / 2), (0, h / 2)]; cy = -h / 6
        m = poly(X, Y, outer); cv.alpha |= m
        P(m, RED); P(poly(X, Y, [(x * 0.6, cy + (y - cy) * 0.6) for x, y in outer]), WHITE)
    elif k == 14:
        octa = lambda s: (np.abs(X) <= s) & (np.abs(Y) <= s) & (np.abs(X) + np.abs(Y) <= 1.4142 * s)
        m = octa(1.0); cv.alpha |= m
        P(m, WHITE); P(octa(0.93), RED)
        t = np.zeros(X.shape, bool)
        for ch, x0 in zip('STOP', (-0.72, -0.34, 0.04, 0.42)):
            t |= glyph(X, Y, ch, x0, -0.31, 0.30, 0.62, 0.10)
        P(t, WHITE)
    elif k == 15:
        red_ring(cv)
    elif k == 16:
        red_ring(cv)
        P(rect(X, Y, -0.20, -0.30, 0.55, 0.15) | rect(X, Y, -0.55, -0.12, -0.25, 0.15)
          | disc(X, Y, -0.35, 0.22, 0.10) | disc(X, Y, 0.30, 0.22, 0.10), BLACK)
    elif k == 17:
        d = disc(X, Y, 0, 0, 1.0); cv.alpha |= d
        P(d, WHITE); P(disc(X, Y, 0, 0, 0.94), RED); P(rect(X, Y, -0.72, -0.16, 0.72, 0.16), WHITE)
    elif k == 32:
        end_disc(cv); end_band(cv)
    elif k == 41:
        end_disc(cv); P(car(X, Y, -0.30, 0.0) | car(X, Y, 0.30, 0.0), GRAY); end_band(cv)
    elif k == 42:
        end_disc(cv); P(lorry(X, Y, -0.30, 0.0) | car(X, Y, 0.32, 0.05), GRAY); end_band(cv)
    elif k == 33:
        blue_disc(cv)
        P(pline(X, Y, [(-0.3, 0.62), (-0.3, 0.05), (-0.2, -0.15), (0.0, -0.25), (0.3, -0.25)], 0.22)
          | poly(X, Y, [(0.72, -0.25), (0.28, -0.62), (0.28, 0.12)]), WHITE)
    elif k == 35:
        blue_disc(cv)
        P(seg(X, Y, 0, 0.68, 0, -0.2, 0.24) | poly(X, Y, [(0, -0.75), (-0.4, -0.18), (0.4, -0.18)]), WHITE)
    elif k == 36:
        blue_disc(cv)
        P(seg(X, Y, -0.22, 0.68, -0.22, -0.25, 0.20) | poly(X, Y, [(-0.22, -0.75), (-0.55, -0.22), (0.11, -0.22)])
          | pline(X, Y, [(-0.22, 0.45), (-0.10, 0.20), (0.10, 0.08), (0.35, 0.08)], 0.18)
          | poly(X, Y, [(0.75, 0.08), (0.33, -0.25), (0.33, 0.41)]), WHITE)
    elif k == 38:
        blue_disc(cv)
        P(seg(X, Y, -0.45, -0.45, 0.2, 0.2, 0.24) | poly(X, Y, [(0.62, 0.62), (0.47, -0.07), (-0.07, 0.47)]), WHITE)
    elif k == 40:
        blue_disc(cv)
        ang = np.degrees(np.arctan2(Y, X)) % 120.0
        r2 = X * X + Y * Y
        P((r2 >= 0.36 ** 2) & (r2 <= 0.62 ** 2) & (ang > 28.0), WHITE)
    else:  # warning triangles with a black pictogram
        LX, LY = warn_tri(cv)
        if k == 11:
            P(rect(LX, LY, -0.14, -0.3, 0.14, 0.9) | poly(LX, LY, [(0, -0.95), (-0.4, -0.25), (0.4, -0.25)])
              | rect(LX, LY, -0.7, 0.15, 0.7, 0.33), BLACK)
        elif k == 18:
            P(poly(LX, LY, [(-0.16, -0.95), (0.16, -0.95), (0.10, 0.35), (-0.10, 0.35)]) | disc(LX, LY, 0, 0.72, 0.17), BLACK)
        elif k == 19:
            P(pline(LX, LY, [(0.35, 0.9), (0.35, 0.1), (0.15, -0.25), (-0.35, -0.4)], 0.26)
              | poly(LX, LY, [(-0.7, -0.45), (-0.3, -0.8), (-0.3, -0.05)]), BLACK)
        elif k == 21:
            P(pline(LX, LY, [(0.3, 0.9), (0.3, 0.4), (-0.3, -0.2), (-0.3, -0.85)], 0.26), BLACK)
        elif k == 22:
            P(rect(LX, LY, -0.9, 0.3, 0.9, 0.5) | ((ell(LX, LY, -0.4, 0.3, 0.32, 0.35) | ell(LX, LY, 0.4, 0.3, 0.32, 0.35)) & (LY <= 0.3)), BLACK)
        elif k == 23:
            P(rect(LX, LY, -0.4, -0.5, 0.4, -0.1) | poly(LX, LY, [(-0.3, -0.5), (-0.2, -0.8), (0.2, -0.8), (0.3, -0.5)])
              | rect(LX, LY, -0.4, -0.1, -0.22, 0.08) | rect(LX, LY, 0.22, -0.1, 0.4, 0.08)
              | pline(LX, LY, [(-0.35, 0.15), (-0.05, 0.5), (-0.45, 0.85)], 0.12)
              | pline(LX, LY, [(0.25, 0.15), (0.55, 0.5), (0.15, 0.85)], 0.12), BLACK)
        elif k == 24:
            P(seg(LX, LY, -0.35, -0.85, -0.35, 0.9, 0.2) | pline(LX, LY, [(0.1, -0.85), (0.1, -0.25), (0.5, 0.25), (0.5, 0.9)], 0.2), BLACK)
        elif k == 25:
            P(disc(LX, LY, -0.15, -0.6, 0.14) | seg(LX, LY, -0.15, -0.45, -0.3, 0.1, 0.25)
              | seg(LX, LY, -0.3, 0.1, -0.6, 0.85, 0.16) | seg(LX, LY, -0.3, 0.1, 0.0, 0.85, 0.16)
              | seg(LX, LY, -0.15, -0.3, 0.35, 0.3, 0.10) | (ell(LX, LY, 0.55, 0.85, 0.4, 0.4) & (LY <= 0.85)), BLACK)
        elif k == 26:
            P(disc(LX, LY, 0, -0.6, 0.24), RED); P(disc(LX, LY, 0, 0.0, 0.24), YEL); P(disc(LX, LY, 0, 0.6, 0.24), GREEN)
        elif k == 27:
            P(person(LX, LY, 0.0, 0.0, 1.0), BLACK)
        elif k == 28:
            P(person(LX, LY, -0.38, 0.2, 0.75) | person(LX, LY, 0.35, 0.08, 0.9), BLACK)
        elif k == 29:
            P((disc(LX, LY, -0.5, 0.45, 0.38) & ~disc(LX, LY, -0.5, 0.45, 0.24)) | (disc(LX, LY, 0.5, 0.45, 0.38) & ~disc(LX, LY, 0.5, 0.45, 0.24))
              | pline(LX, LY, [(-0.5, 0.45), (-0.15, -0.15), (0.35, -0.15), (0.0, 0.45), (-0.15, -0.15)], 0.1)
              | seg(LX, LY, 0.5, 0.45, 0.3, -0.4, 0.1) | seg(LX, LY, 0.15, -0.4, 0.4, -0.4, 0.1) | seg(LX, LY, -0.3, -0.3, 0.0, -0.3, 0.12), BLACK)
        elif k == 30:
            m = np.zeros(X.shape, bool)
            for a in (90.0, 30.0, 150.0):
                c, s = 0.75 * np.cos(np.radians(a)), 0.75 * np.sin(np.radians(a))
                m |= seg(LX, LY, -c, 0.1 - s, c, 0.1 + s, 0.16)
            P(m, BLACK)
        elif k == 31:
            P(ell(LX, LY, 0.05, 0.1, 0.5, 0.22) | seg(LX, LY, -0.3, 0.0, -0.5, -0.5, 0.2) | seg(LX, LY, -0.5, -0.5, -0.7, -0.42, 0.16)
              | seg(LX, LY, -0.5, -0.5, -0.35, -0.85, 0.1) | seg(LX, LY, -0.3, 0.2, -0.6, 0.75, 0.12) | seg(LX, LY, -0.2, 0.25, -0.3, 0.85, 0.12)
              | seg(LX, LY, 0.4, 0.2, 0.75, 0.75, 0.12) | seg(LX, LY, 0.3, 0.25, 0.45, 0.85, 0.12), BLACK)
    return cv.rgb, cv.alpha


# ---------------------------------------------------------------- first layer: colour-opponent, blurred and band-pass filters
def gauss(sigma):
    r = np.arange(7) - 3.0
    g = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * sigma * sigma))
    return g / g.sum()


K = np.zeros((4, 3, 7, 7))
K[0] = gauss(0.8)[None] * np.array([1 / 3, 1 / 3, 1 / 3])[:, None, None]                 # brightness
K[1] = gauss(1.0)[None] * np.array([1.0, -0.5, -0.5])[:, None, None]                     # red against green/blue
K[2] = gauss(1.0)[None] * np.array([-0.5, -0.5, 1.0])[:, None, None]                     # blue against red/green
K[3] = (gauss(0.7) - gauss(2.0))[None] * np.array([1 / 3, 1 / 3, 1 / 3])[:, None, None]  # fine detail (difference of Gaussians)


def feats(imgs):  # same arithmetic as the conv layer, padding 3
    xp = np.pad(imgs, ((0, 0), (0, 0), (3, 3), (3, 3)))
    out = np.zeros((len(imgs), 4, 32, 32))
    for dy in range(7):
        for dx in range(7):
            out += np.einsum('oi,nihw->nohw', K[:, :, dy, dx], xp[:, :, dy:dy + 32, dx:dx + 32])
    return out


# ---------------------------------------------------------------- templates: one drawing each, normalised by itself alone
FRACS = (0.62, 0.70, 0.78, 0.86)            # sign size as a share of the frame (border is about 10 percent, at least 5 px)
SHIFTS = (-3.0, -1.5, 0.0, 1.5, 3.0)        # pixels
SS = 3                                      # supersampling per pixel
CH_W = (1.0, 1.0, 1.0, 1.0)
V = len(FRACS) * len(SHIFTS) ** 2


def template(k, frac, dx, dy):
    p = (np.arange(32 * SS) + 0.5) / SS
    R = 16.0 * frac
    X, Y = np.meshgrid((p - 16.0 - dx) / R, (p - 16.0 - dy) / R)
    rgb, alpha = draw(k, X, Y)
    rgb = rgb.reshape(3, 32, SS, 32, SS).mean(axis=(2, 4))
    alpha = alpha.astype(float).reshape(32, SS, 32, SS).mean(axis=(1, 3))
    return rgb, alpha


def build_templates():
    W = np.zeros((43 * V, 4 * 32 * 32), np.float32)
    for k in range(43):
        imgs, alphas = [], []
        for frac in FRACS:
            for dy in SHIFTS:
                for dx in SHIFTS:
                    rgb, a = template(k, frac, dx, dy); imgs.append(rgb); alphas.append(a)
        F = feats(np.stack(imgs)); A = np.stack(alphas)
        mu = F.mean(axis=(2, 3), keepdims=True)
        F = (F - mu) / np.sqrt(((F - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)   # each template by itself
        for c in (0, 3):   # brightness channels: only inside the sign (background brightness is unknown), zero sum there
            m = (F[:, c] * A).sum(axis=(1, 2), keepdims=True) / A.sum(axis=(1, 2), keepdims=True)
            F[:, c] = (F[:, c] - m) * A
        for c in range(4):
            F[:, c] *= CH_W[c]
        F = F.reshape(len(F), -1)
        F /= np.sqrt((F ** 2).sum(axis=1, keepdims=True))                                  # unit length, each by itself
        W[k * V:(k + 1) * V] = F.astype(np.float32)
    return W


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    net = {"layers": [
        {"name": "feat", "op": "conv", "inputs": ["image"], "out": 4, "kernel": [7, 7], "stride": 1, "padding": 3},
        {"name": "fn", "op": "norm", "inputs": ["feat"]},
        {"name": "match", "op": "dense", "inputs": ["fn"], "out": 43 * V},
        {"name": "grp", "op": "reshape", "inputs": ["match"], "shape": [1, 43, V]},
        {"name": "best", "op": "maxpool", "inputs": ["grp"], "size": [1, V], "stride": [1, V]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [43]},
    ]}
    weights = {"feat.W": K.astype(np.float32), "feat.b": np.zeros(4, np.float32),
               "match.W": build_templates(), "match.b": np.zeros(43 * V, np.float32)}
    (out / "network.json").write_text(json.dumps(net, indent=1))
    np.savez(out / "weights.npz", **weights)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
