"""Hand-designed traffic sign matcher. Writes network.json and weights.npz; reads no files.

Every weight comes from formulas and sign drawings typed here from general knowledge of German road signs:
each class is drawn as a vector graphic (ring, triangle, digits as strokes, arrows ...), rendered at several
sizes and offsets, and each render becomes one matched-filter template normalised by itself alone.
"""
import sys, os, json
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
N, S = 32, 3                                   # image size, supersampling per pixel
P = (np.arange(N * S) + 0.5) / S               # sample positions in pixel units

RED, WHITE, BLACK = (0.72, 0.10, 0.12), (0.95, 0.95, 0.95), (0.08, 0.08, 0.08)
BLUE, YELLOW, GREY, GREEN = (0.10, 0.28, 0.65), (0.95, 0.75, 0.05), (0.40, 0.40, 0.40), (0.10, 0.60, 0.25)
DARK = (0.2, 0.2, 0.2)

# colour features: luminance, red opponent, blue opponent
A = np.array([[1 / 3, 1 / 3, 1 / 3], [1.0, -0.5, -0.5], [-0.5, -0.5, 1.0]])
CW = np.array([1.0, 0.7, 0.7])                 # hand-chosen channel weights

SHIFTS = [-3.0, -1.5, 0.0, 1.5, 3.0]
VAR = [(d, dx, dy, 1.0) for d in (19.0, 22.0, 25.0, 28.0) for dy in SHIFTS for dx in SHIFTS]
VAR31 = [(d, dx, dy, f) for f in (1.0, -1.0) for d in (21.0, 26.0) for dy in SHIFTS for dx in SHIFTS]
NV = len(VAR)
MIRROR = {20: 19, 34: 33, 37: 36, 39: 38}


def arc(cx, cy, rx, ry, t0, t1, n=14):
    t = np.radians(np.linspace(t0, t1, n))
    return [(cx + rx * float(np.round(np.cos(a), 6)), cy + ry * float(np.round(np.sin(a), 6))) for a in t]


class Cv:
    def __init__(s, X, Y):
        s.X, s.Y = X, Y
        s.img = np.zeros(X.shape + (3,))
        s.mask = np.zeros(X.shape, bool)

    def paint(s, m, col):
        s.img[m] = col

    def disc(s, cx, cy, r):
        return (s.X - cx) ** 2 + (s.Y - cy) ** 2 <= r * r

    def ring(s, cx, cy, r, w):
        return s.disc(cx, cy, r + w / 2) & ~s.disc(cx, cy, r - w / 2)

    def rect(s, x0, y0, x1, y1):
        return (s.X >= x0) & (s.X <= x1) & (s.Y >= y0) & (s.Y <= y1)

    def seg(s, x0, y0, x1, y1, w):
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            return s.disc(x0, y0, w / 2)
        t = np.clip(((s.X - x0) * dx + (s.Y - y0) * dy) / L2, 0, 1)
        return (s.X - x0 - t * dx) ** 2 + (s.Y - y0 - t * dy) ** 2 <= (w / 2) ** 2

    def line(s, pts, w):
        m = np.zeros(s.X.shape, bool)
        for a, b in zip(pts[:-1], pts[1:]):
            m |= s.seg(a[0], a[1], b[0], b[1], w)
        return m

    def poly(s, pts):
        m = np.zeros(s.X.shape, bool)
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if y1 == y2:
                continue
            m ^= ((y1 > s.Y) != (y2 > s.Y)) & (s.X < (x2 - x1) * (s.Y - y1) / (y2 - y1) + x1)
        return m

    def head(s, tx, ty, dx, dy, l, hw):
        n = (dx * dx + dy * dy) ** 0.5
        dx, dy = dx / n, dy / n
        bx, by = tx - dx * l, ty - dy * l
        return s.poly([(tx, ty), (bx - dy * hw, by + dx * hw), (bx + dy * hw, by - dx * hw)])


def glyph(ch):
    """Stroke centre lines of a character in a box [-1, 1] x [-1, 1] (y down)."""
    if ch in "0O":
        return [arc(0, 0, 1, 1, 0, 360, 20)]
    if ch == "1":
        return [[(-0.6, -0.5), (0.3, -1), (0.3, 1)]]
    if ch == "2":
        return [arc(0, -0.45, 1, 0.55, 170, 390, 12) + [(-1, 1), (1, 1)]]
    if ch == "3":
        return [arc(0, -0.5, 0.9, 0.5, 200, 450, 12) + arc(0, 0.5, 1, 0.5, 270, 520, 12)]
    if ch == "4":
        return [[(0.45, 1), (0.45, -1), (-1, 0.4), (1, 0.4)]]
    if ch == "5":
        return [[(0.85, -1), (-0.8, -1), (-0.85, -0.1)] + arc(0, 0.3, 1, 0.7, 215, 510, 14)]
    if ch == "6":
        return [arc(0, 0.35, 1, 0.65, 0, 360, 18),
                [(-1, 0.35), (-0.95, -0.2), (-0.7, -0.65), (-0.2, -0.95), (0.3, -1), (0.75, -0.85)]]
    if ch == "7":
        return [[(-1, -1), (1, -1), (-0.2, 1)]]
    if ch == "8":
        return [arc(0, -0.5, 0.85, 0.5, 0, 360, 16), arc(0, 0.5, 1, 0.5, 0, 360, 16)]
    if ch == "9":
        return [[(-x, -y) for x, y in pl] for pl in glyph("6")]
    if ch == "S":
        return [arc(0, -0.5, 1, 0.5, -30, -270, 12) + arc(0, 0.5, 1, 0.5, -90, 150, 12)]
    if ch == "T":
        return [[(-1, -1), (1, -1)], [(0, -1), (0, 1)]]
    if ch == "P":
        return [[(-1, 1), (-1, -1), (0.2, -1)] + arc(0.2, -0.45, 0.8, 0.55, -90, 90, 10) + [(-1, 0.1)]]
    raise ValueError(ch)


def text(cv, s, cx, cy, h, sw, wd, gap, w1=None):
    ws = [w1 if (ch == "1" and w1) else wd for ch in s]
    x = cx - (sum(ws) + gap * (len(s) - 1)) / 2
    m = np.zeros(cv.X.shape, bool)
    for ch, w in zip(s, ws):
        ax, ay, mx = (w - sw) / 2, (h - sw) / 2, x + w / 2
        for pl in glyph(ch):
            m |= cv.line([(mx + px * ax, cy + py * ay) for px, py in pl], sw)
        x += w + gap
    return m


def car(cv, cx, cy):
    """Car seen from the rear: body, roof, two wheels."""
    m = cv.rect(cx - 0.26, cy - 0.04, cx + 0.26, cy + 0.17)
    m |= cv.poly([(cx - 0.2, cy - 0.04), (cx - 0.14, cy - 0.24), (cx + 0.14, cy - 0.24), (cx + 0.2, cy - 0.04)])
    m |= cv.rect(cx - 0.24, cy + 0.17, cx - 0.12, cy + 0.27) | cv.rect(cx + 0.12, cy + 0.17, cx + 0.24, cy + 0.27)
    return m


def truck(cv, cx, cy):
    """Truck seen from the rear: tall box, two wheels."""
    m = cv.rect(cx - 0.27, cy - 0.32, cx + 0.27, cy + 0.17)
    m |= cv.rect(cx - 0.25, cy + 0.17, cx - 0.12, cy + 0.28) | cv.rect(cx + 0.12, cy + 0.17, cx + 0.25, cy + 0.28)
    return m


SPEED = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}
TRI = [(0.0, -0.93), (1.0, 0.93), (-1.0, 0.93)]


def speed_text(cv, s):
    if len(s) == 2:
        return text(cv, s, 0, 0, 0.80, 0.13, 0.46, 0.08)
    return text(cv, s, 0, 0, 0.74, 0.115, 0.37, 0.07, w1=0.22)


def draw(k, cv):
    X, Y = cv.X, cv.Y
    if k in SPEED or k in (9, 10, 15, 16):                    # white disc, red ring
        d = cv.disc(0, 0, 1.0)
        cv.mask |= d
        cv.paint(d, RED)
        cv.paint(cv.disc(0, 0, 0.72), WHITE)
        if k in SPEED:
            cv.paint(speed_text(cv, SPEED[k]), BLACK)
        elif k == 9:
            cv.paint(car(cv, -0.31, 0.0), RED)
            cv.paint(car(cv, 0.31, 0.0), BLACK)
        elif k == 10:
            cv.paint(truck(cv, -0.31, 0.03), RED)
            cv.paint(car(cv, 0.33, 0.05), BLACK)
        elif k == 16:
            m = cv.rect(-0.18, -0.30, 0.56, 0.14) | cv.disc(-0.36, 0.2, 0.11) | cv.disc(0.3, 0.2, 0.11)
            m |= cv.poly([(-0.56, 0.14), (-0.56, -0.02), (-0.44, -0.2), (-0.22, -0.2), (-0.22, 0.14)])
            cv.paint(m, BLACK)
    elif k in (6, 32, 41, 42):                                # "end of" signs: white disc, diagonal band
        d = cv.disc(0, 0, 1.0)
        cv.mask |= d
        cv.paint(d, WHITE)
        cv.paint(d & ~cv.disc(0, 0, 0.92), GREY)
        if k == 6:
            cv.paint(speed_text(cv, "80"), GREY)
        elif k == 41:
            cv.paint(car(cv, -0.31, 0.0) | car(cv, 0.31, 0.0), GREY)
        elif k == 42:
            cv.paint(truck(cv, -0.31, 0.03) | car(cv, 0.33, 0.05), GREY)
        cv.paint(cv.seg(-0.68, 0.68, 0.68, -0.68, 0.30 if k == 32 else 0.20), DARK if k != 32 else BLACK)
    elif k == 12:                                             # priority road
        d = np.abs(X) + np.abs(Y) <= 1.0
        cv.mask |= d
        cv.paint(d, WHITE)
        cv.paint(np.abs(X) + np.abs(Y) <= 0.62, YELLOW)
    elif k == 13:                                             # yield
        pts = [(x, -y) for x, y in TRI]
        t = cv.poly(pts)
        cv.mask |= t
        cv.paint(t, RED)
        cv.paint(cv.poly([(x * 0.62, -0.31 + (y + 0.31) * 0.62) for x, y in pts]), WHITE)
    elif k == 14:                                             # stop
        d = (np.abs(X) <= 1.0) & (np.abs(Y) <= 1.0) & (np.abs(X) + np.abs(Y) <= 1.414)
        cv.mask |= d
        cv.paint(d, RED)
        cv.paint(text(cv, "STOP", 0, 0, 0.62, 0.10, 0.32, 0.07), WHITE)
    elif k == 17:                                             # no entry
        d = cv.disc(0, 0, 1.0)
        cv.mask |= d
        cv.paint(d, RED)
        cv.paint(cv.rect(-0.75, -0.16, 0.75, 0.16), WHITE)
    elif k in (11, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):   # warning triangles
        t = cv.poly(TRI)
        cv.mask |= t
        cv.paint(t, RED)
        cv.paint(cv.poly([(x * 0.62, 0.31 + (y - 0.31) * 0.62) for x, y in TRI]), WHITE)
        if k == 11:
            m = cv.rect(-0.08, 0.0, 0.08, 0.62) | cv.head(0, -0.25, 0, -1, 0.3, 0.2) | cv.rect(-0.36, 0.27, 0.36, 0.38)
        elif k == 18:
            m = cv.rect(-0.075, -0.2, 0.075, 0.36) | cv.disc(0, 0.54, 0.085)
        elif k == 19:
            m = cv.line([(0.15, 0.62), (0.15, 0.32), (0.06, 0.17), (-0.08, 0.10)], 0.13)
            m |= cv.poly([(-0.27, 0.06), (-0.06, -0.05), (-0.04, 0.24)])
        elif k == 21:
            m = cv.line([(0.14, 0.62), (0.14, 0.42), (-0.12, 0.2), (-0.12, -0.08)], 0.12)
        elif k == 22:
            m = cv.rect(-0.42, 0.42, 0.42, 0.52) | ((cv.disc(-0.2, 0.42, 0.17) | cv.disc(0.2, 0.42, 0.17)) & (Y <= 0.42))
        elif k == 23:
            m = cv.poly([(-0.2, -0.02), (0.14, -0.1), (0.2, 0.2), (-0.14, 0.28)])
            m |= cv.line([(-0.2, 0.3), (-0.08, 0.42), (-0.22, 0.52), (-0.1, 0.62)], 0.06)
            m |= cv.line([(0.08, 0.28), (0.2, 0.4), (0.06, 0.5), (0.18, 0.62)], 0.06)
        elif k == 24:
            m = cv.line([(-0.13, 0.0), (-0.13, 0.62)], 0.1)
            m |= cv.line([(0.05, 0.0), (0.05, 0.17), (0.24, 0.37), (0.24, 0.62)], 0.1)
        elif k == 25:
            m = cv.disc(-0.06, -0.08, 0.075) | cv.seg(-0.08, 0.02, 0.0, 0.3, 0.13)
            m |= cv.seg(0.0, 0.3, -0.16, 0.62, 0.08) | cv.seg(0.0, 0.3, 0.1, 0.62, 0.08) | cv.seg(-0.04, 0.08, 0.2, 0.36, 0.06)
            m |= (((X - 0.3) / 0.2) ** 2 + ((Y - 0.62) / 0.17) ** 2 <= 1.0) & (Y <= 0.62)
        elif k == 26:
            m = cv.rect(-0.12, -0.25, 0.12, 0.6)
        elif k == 27:
            m = cv.disc(0, -0.12, 0.075) | cv.seg(0, 0.0, 0, 0.3, 0.13)
            m |= cv.seg(0, 0.3, -0.16, 0.62, 0.08) | cv.seg(0, 0.3, 0.16, 0.62, 0.08)
            m |= cv.seg(0, 0.03, -0.16, 0.24, 0.06) | cv.seg(0, 0.03, 0.16, 0.24, 0.06)
        elif k == 28:
            m = cv.disc(-0.14, -0.02, 0.07) | cv.seg(-0.14, 0.08, -0.12, 0.34, 0.12)
            m |= cv.seg(-0.12, 0.34, -0.28, 0.62, 0.07) | cv.seg(-0.12, 0.34, 0.0, 0.62, 0.07)
            m |= cv.disc(0.2, 0.12, 0.06) | cv.seg(0.2, 0.2, 0.2, 0.4, 0.1)
            m |= cv.seg(0.2, 0.4, 0.1, 0.62, 0.06) | cv.seg(0.2, 0.4, 0.32, 0.62, 0.06) | cv.seg(-0.12, 0.16, 0.2, 0.26, 0.05)
        elif k == 29:
            m = cv.ring(-0.24, 0.44, 0.15, 0.05) | cv.ring(0.24, 0.44, 0.15, 0.05)
            m |= cv.line([(-0.24, 0.44), (-0.06, 0.16), (0.16, 0.16), (0.24, 0.44)], 0.05)
            m |= cv.line([(-0.06, 0.16), (0.04, 0.44), (0.16, 0.16)], 0.05)
            m |= cv.seg(-0.12, 0.1, 0.0, 0.1, 0.05) | cv.seg(0.16, 0.16, 0.12, 0.04, 0.05)
        elif k == 30:
            m = cv.disc(0, 0.28, 0.08)
            for cx_, sy_ in ((0.0, 1.0), (0.866, 0.5), (-0.866, 0.5)):
                m |= cv.seg(-0.27 * cx_, 0.28 - 0.27 * sy_, 0.27 * cx_, 0.28 + 0.27 * sy_, 0.07)
        else:  # 31: leaping deer (both facings are rendered through the flip variants)
            m = cv.seg(0.2, 0.36, -0.1, 0.16, 0.16) | cv.seg(-0.1, 0.16, -0.19, -0.02, 0.08) | cv.seg(-0.19, -0.02, -0.26, 0.06, 0.07)
            m |= cv.seg(-0.19, -0.02, -0.13, -0.16, 0.04) | cv.seg(-0.08, 0.2, -0.28, 0.4, 0.05)
            m |= cv.seg(0.2, 0.36, 0.36, 0.6, 0.06) | cv.seg(0.14, 0.38, 0.2, 0.62, 0.05)
        cv.paint(m, BLACK)
        if k == 26:
            cv.paint(cv.disc(0, -0.11, 0.09), RED)
            cv.paint(cv.disc(0, 0.17, 0.09), YELLOW)
            cv.paint(cv.disc(0, 0.45, 0.09), GREEN)
    else:                                                     # blue discs with white arrows
        d = cv.disc(0, 0, 1.0)
        cv.mask |= d
        cv.paint(d, BLUE)
        if k == 35:
            m = cv.line([(0, 0.68), (0, -0.3)], 0.2) | cv.head(0, -0.75, 0, -1, 0.5, 0.38)
        elif k == 33:
            m = cv.line([(-0.3, 0.65), (-0.3, 0.05)] + arc(0.0, 0.05, 0.3, 0.3, 180, 270, 8) + [(0.25, -0.25)], 0.2)
            m |= cv.head(0.74, -0.25, 1, 0, 0.5, 0.38)
        elif k == 36:
            m = cv.line([(-0.22, 0.68), (-0.22, -0.3)], 0.17) | cv.head(-0.22, -0.75, 0, -1, 0.42, 0.3)
            m |= cv.line([(-0.22, 0.45), (-0.1, 0.2), (0.1, 0.08), (0.34, 0.05)], 0.15) | cv.head(0.76, 0.05, 1, 0, 0.42, 0.3)
        elif k == 38:
            m = cv.line([(-0.45, -0.45), (0.2, 0.2)], 0.2) | cv.head(0.6, 0.6, 1, 1, 0.55, 0.38)
        else:  # 40: roundabout, three arrows running anticlockwise
            m = np.zeros(X.shape, bool)
            for a0 in (90.0, 210.0, 330.0):
                m |= cv.line(arc(0, 0, 0.5, 0.5, a0, a0 - 75, 8), 0.16)
                te = a0 - 75
                pe = arc(0, 0, 1, 1, te, te, 1)[0]
                pt = arc(0, 0, 0.5, 0.5, te - 34, te - 34, 1)[0]
                m |= cv.poly([pt, (pe[0] * 0.26, pe[1] * 0.26), (pe[0] * 0.74, pe[1] * 0.74)])
        cv.paint(m, WHITE)


def blur(T, sigma=0.7):
    x = np.arange(-2, 3)
    g = np.exp(-x * x / (2 * sigma * sigma))
    g /= g.sum()
    for ax in (2, 3):
        Tp = np.pad(T, [(0, 0), (0, 0)] + [(2, 2) if a == ax else (0, 0) for a in (2, 3)])
        T = sum(g[i] * (Tp[:, :, i:i + N, :] if ax == 2 else Tp[:, :, :, i:i + N]) for i in range(5))
    return T


def templates(k):
    var = np.array(VAR31 if k == 31 else VAR)
    d, dx, dy, fl = (var[:, i][:, None, None] for i in range(4))
    X = np.ascontiguousarray(np.broadcast_to(fl * (P[None, None, :] - (16 + dx)) / (d / 2), (NV, N * S, N * S)))
    Y = np.ascontiguousarray(np.broadcast_to((P[None, :, None] - (16 + dy)) / (d / 2), (NV, N * S, N * S)))
    if k in MIRROR:
        X, k = -X, MIRROR[k]
    cv = Cv(X, Y)
    draw(k, cv)
    F = cv.img @ A.T                                           # colour features of the drawing
    m = cv.mask.astype(np.float64)
    mu = (F * m[..., None]).sum((1, 2)) / m.sum((1, 2))[:, None]
    H = (F - mu[:, None, None, :]) * m[..., None]              # zero mean inside the sign, zero outside it
    T = H.reshape(NV, N, S, N, S, 3).mean((2, 4)).transpose(0, 3, 1, 2)
    T = blur(T) * CW[None, :, None, None]
    T /= np.sqrt((T ** 2).sum((1, 2, 3)))[:, None, None, None]  # each template normalised by itself alone
    return T.reshape(NV, -1)


def main():
    os.makedirs(OUT, exist_ok=True)
    W = np.concatenate([templates(k) for k in range(43)]).astype(np.float32)
    weights = {
        "feat.W": A.reshape(3, 3, 1, 1).astype(np.float32), "feat.b": np.zeros(3, np.float32),
        "match.W": W, "match.b": np.zeros(len(W), np.float32),
    }
    net = {"layers": [
        {"name": "feat", "op": "conv", "inputs": ["image"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "z", "op": "norm", "inputs": ["feat"]},
        {"name": "match", "op": "dense", "inputs": ["z"], "out": 43 * NV},
        {"name": "grid", "op": "reshape", "inputs": ["match"], "shape": [43, NV, 1]},
        {"name": "best", "op": "maxpool", "inputs": ["grid"], "size": [NV, 1], "stride": [NV, 1]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [43]},
    ]}
    with open(os.path.join(OUT, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(OUT, "weights.npz"), **weights)


if __name__ == "__main__":
    main()
