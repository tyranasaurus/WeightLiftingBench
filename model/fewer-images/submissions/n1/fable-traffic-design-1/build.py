"""Builds a hand-designed traffic-sign network: python3 build.py OUT_DIR
No data is read. Every weight comes from sign drawings typed below from general knowledge of German road signs.
Network: 1x1 conv to (luminance, red opponent, blue opponent) -> norm -> one 32x32 template per class and per sign size,
slid over a 7x7 window of shifts -> max over shifts -> max over sizes -> 43 scores."""
import sys, json
import numpy as np
from pathlib import Path

N, SS = 32, 3                       # frame size, supersampling of the drawings
RADII = (9.0, 10.5, 12.0, 13.5)     # half-width of the sign in pixels (sign fills 56% .. 84% of the frame)
PAD = 3                             # templates are tried at shifts of -3..3 pixels
LAM = 1.0                           # extra weight of the inner-symbol template
RED, WHITE, BLUE = (0.80, 0.10, 0.12), (0.95, 0.95, 0.95), (0.10, 0.30, 0.75)
YEL, BLACK, GRAY, DARK = (1.00, 0.80, 0.10), (0.08, 0.08, 0.08), (0.50, 0.50, 0.50), (0.25, 0.25, 0.25)
GREEN = (0.10, 0.60, 0.20)


# ---------- drawing primitives (coordinates: sign centre 0,0; sign half-width 1; y points down) ----------
def seg(X, Y, x0, y0, x1, y1, w):
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / (dx * dx + dy * dy + 1e-12), 0, 1)
    return (X - x0 - t * dx) ** 2 + (Y - y0 - t * dy) ** 2 <= (w / 2) ** 2


def line(X, Y, pts, w):
    m = np.zeros(X.shape, bool)
    for (a, b), (c, d) in zip(pts[:-1], pts[1:]):
        m |= seg(X, Y, a, b, c, d, w)
    return m


def poly(X, Y, pts):
    m = np.zeros(X.shape, bool)
    n = len(pts)
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        if y0 == y1:
            continue
        cond = ((y0 <= Y) & (Y < y1)) | ((y1 <= Y) & (Y < y0))
        m ^= cond & (X < x0 + (Y - y0) * (x1 - x0) / (y1 - y0))
    return m


def disc(X, Y, cx, cy, r):
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r


def rect(X, Y, x0, y0, x1, y1):
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


def ell(X, Y, cx, cy, rx, ry, w):
    a = np.linspace(0, 2 * np.pi, 25)
    return line(X, Y, [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in a], w)


def scale_pts(pts, c, s):
    return [(c[0] + s * (x - c[0]), c[1] + s * (y - c[1])) for x, y in pts]


# ---------- digits and letters as strokes in a unit box ----------
STROKES = {
    "1": [[(0.0, 0.28), (1.0, 0.0), (1.0, 1.0)]],
    "2": [[(0.05, 0.28), (0.2, 0.06), (0.5, 0.0), (0.8, 0.06), (0.95, 0.28), (0.85, 0.5), (0.05, 1.0), (1.0, 1.0)]],
    "3": [[(0.05, 0.14), (0.3, 0.0), (0.7, 0.0), (0.95, 0.22), (0.7, 0.47), (0.4, 0.47)],
          [(0.7, 0.47), (0.95, 0.72), (0.7, 1.0), (0.3, 1.0), (0.05, 0.86)]],
    "4": [[(0.75, 1.0), (0.75, 0.0), (0.0, 0.7), (1.0, 0.7)]],
    "5": [[(0.9, 0.0), (0.15, 0.0), (0.1, 0.45), (0.5, 0.38), (0.85, 0.5), (0.95, 0.72), (0.7, 1.0), (0.3, 1.0), (0.05, 0.86)]],
    "6": [[(0.85, 0.1), (0.6, 0.0), (0.3, 0.05), (0.08, 0.35), (0.05, 0.7), (0.25, 1.0), (0.7, 1.0), (0.95, 0.75),
           (0.8, 0.48), (0.4, 0.42), (0.08, 0.6)]],
    "7": [[(0.0, 0.0), (1.0, 0.0), (0.4, 1.0)]],
    "S": [[(1.0, 0.1), (0.5, 0.0), (0.0, 0.15), (0.1, 0.42), (0.9, 0.58), (1.0, 0.85), (0.5, 1.0), (0.0, 0.9)]],
    "T": [[(0.0, 0.0), (1.0, 0.0)], [(0.5, 0.0), (0.5, 1.0)]],
    "P": [[(0.0, 1.0), (0.0, 0.0), (0.7, 0.0), (1.0, 0.15), (1.0, 0.4), (0.7, 0.52), (0.0, 0.52)]],
}


def glyph(X, Y, ch, x0, y0, w, h, sw):
    if ch in ("0", "O"):
        return ell(X, Y, x0 + w / 2, y0 + h / 2, w / 2, h / 2, sw)
    if ch == "8":
        return ell(X, Y, x0 + w / 2, y0 + 0.25 * h, 0.4 * w, 0.25 * h, sw) | ell(X, Y, x0 + w / 2, y0 + 0.73 * h, 0.48 * w, 0.27 * h, sw)
    m = np.zeros(X.shape, bool)
    for s in STROKES[ch]:
        m |= line(X, Y, [(x0 + w * a, y0 + h * b) for a, b in s], sw)
    return m


def number(X, Y, text):
    if len(text) == 2:
        return glyph(X, Y, text[0], -0.47, -0.40, 0.42, 0.80, 0.12) | glyph(X, Y, text[1], 0.05, -0.40, 0.42, 0.80, 0.12)
    return (glyph(X, Y, text[0], -0.54, -0.35, 0.20, 0.70, 0.10) | glyph(X, Y, text[1], -0.26, -0.35, 0.36, 0.70, 0.10)
            | glyph(X, Y, text[2], 0.18, -0.35, 0.36, 0.70, 0.10))


def car(X, Y, x0, x1):
    return rect(X, Y, x0, 0.0, x1, 0.22) | rect(X, Y, x0 + 0.08, -0.2, x1 - 0.08, 0.0)


def person(X, Y, cx, top, s):
    m = disc(X, Y, cx, top + 0.09 * s, 0.09 * s)
    m |= seg(X, Y, cx, top + 0.2 * s, cx, top + 0.5 * s, 0.15 * s)
    m |= seg(X, Y, cx, top + 0.5 * s, cx - 0.2 * s, top + 0.86 * s, 0.1 * s) | seg(X, Y, cx, top + 0.5 * s, cx + 0.2 * s, top + 0.86 * s, 0.1 * s)
    m |= seg(X, Y, cx, top + 0.22 * s, cx - 0.2 * s, top + 0.45 * s, 0.07 * s) | seg(X, Y, cx, top + 0.22 * s, cx + 0.2 * s, top + 0.45 * s, 0.07 * s)
    return m


SPEED = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}
TRI_UP = [(0.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
TRI_DOWN = [(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]


def flipx(m):
    return m[:, ::-1]


def tri_symbol(X, Y, k):
    """Black symbol inside a danger triangle (white area: apex (0,-0.49), base corners (+-0.62, 0.75))."""
    if k == 11:   # priority at next junction: thick upward arrow crossed by a thin bar
        return poly(X, Y, [(0.0, -0.45), (-0.22, 0.05), (0.22, 0.05)]) | rect(X, Y, -0.09, 0.0, 0.09, 0.70) | rect(X, Y, -0.33, 0.30, 0.33, 0.43)
    if k == 18:   # exclamation mark
        return seg(X, Y, 0.0, -0.22, 0.0, 0.36, 0.15) | disc(X, Y, 0.0, 0.60, 0.09)
    if k == 19:   # bend to the left
        return line(X, Y, [(0.15, 0.66), (0.15, 0.22), (-0.2, -0.05)], 0.13) | poly(X, Y, [(-0.34, -0.16), (-0.08, -0.14), (-0.22, 0.10)])
    if k == 20:
        return flipx(tri_symbol(X, Y, 19))
    if k == 21:   # double bend, first to the left
        return line(X, Y, [(0.18, 0.66), (0.18, 0.42), (-0.18, 0.12), (-0.18, -0.15)], 0.13)
    if k == 22:   # bumpy road: two humps on a line
        return (rect(X, Y, -0.45, 0.45, 0.45, 0.56) | disc(X, Y, -0.2, 0.47, 0.17) | disc(X, Y, 0.2, 0.47, 0.17)) & (Y < 0.56)
    if k == 23:   # slippery road: car above two wavy skid marks
        return (poly(X, Y, [(-0.22, -0.02), (0.14, -0.10), (0.22, 0.22), (-0.14, 0.30)])
                | line(X, Y, [(-0.12, 0.30), (-0.26, 0.45), (-0.14, 0.56), (-0.28, 0.68)], 0.07)
                | line(X, Y, [(0.16, 0.24), (0.04, 0.42), (0.16, 0.54), (0.04, 0.68)], 0.07))
    if k == 24:   # road narrows on the right: straight left line, right line stepping inwards
        return seg(X, Y, -0.16, -0.10, -0.16, 0.66, 0.11) | line(X, Y, [(0.10, -0.10), (0.10, 0.20), (0.28, 0.42), (0.28, 0.66)], 0.11)
    if k == 25:   # road works: digging man and a heap
        return (disc(X, Y, -0.10, -0.12, 0.09) | seg(X, Y, -0.10, 0.02, 0.0, 0.30, 0.14)
                | seg(X, Y, 0.0, 0.30, -0.20, 0.66, 0.10) | seg(X, Y, 0.0, 0.30, 0.10, 0.66, 0.10)
                | seg(X, Y, -0.05, 0.08, 0.25, 0.36, 0.07) | poly(X, Y, [(0.14, 0.70), (0.52, 0.70), (0.34, 0.44)]))
    if k == 27:   # pedestrian
        return person(X, Y, 0.0, -0.22, 1.0)
    if k == 28:   # children crossing: two figures
        return person(X, Y, -0.20, 0.0, 0.76) | person(X, Y, 0.20, -0.10, 0.88)
    if k == 29:   # bicycle
        return (ell(X, Y, -0.25, 0.46, 0.16, 0.16, 0.06) | ell(X, Y, 0.25, 0.46, 0.16, 0.16, 0.06)
                | line(X, Y, [(-0.25, 0.46), (-0.06, 0.16), (0.18, 0.16), (0.25, 0.46)], 0.06)
                | line(X, Y, [(-0.06, 0.16), (0.04, 0.46), (0.18, 0.16)], 0.06) | seg(X, Y, 0.18, 0.16, 0.14, 0.02, 0.06)
                | seg(X, Y, -0.12, 0.08, 0.0, 0.08, 0.06))
    if k == 30:   # snowflake
        m = disc(X, Y, 0.0, 0.30, 0.08)
        for a in (0.0, np.pi / 3, 2 * np.pi / 3):
            m |= seg(X, Y, -0.32 * np.cos(a), 0.30 - 0.32 * np.sin(a), 0.32 * np.cos(a), 0.30 + 0.32 * np.sin(a), 0.08)
        return m
    if k == 31:   # leaping deer, facing left
        return (seg(X, Y, -0.18, 0.14, 0.24, 0.34, 0.20) | seg(X, Y, -0.18, 0.14, -0.30, -0.10, 0.10)
                | seg(X, Y, -0.30, -0.10, -0.42, -0.04, 0.08) | seg(X, Y, -0.28, -0.12, -0.20, -0.26, 0.05)
                | seg(X, Y, 0.24, 0.34, 0.42, 0.64, 0.08) | seg(X, Y, -0.14, 0.22, -0.36, 0.48, 0.08))
    raise ValueError(k)


def blue_symbol(X, Y, k):
    """White arrows on a blue disc."""
    if k == 33:   # turn right ahead: stem from the bottom left bending to an arrow pointing right
        return line(X, Y, [(-0.35, 0.55), (-0.35, 0.10), (-0.20, -0.10), (0.22, -0.10)], 0.20) | poly(X, Y, [(0.20, -0.42), (0.66, -0.10), (0.20, 0.22)])
    if k == 34:
        return flipx(blue_symbol(X, Y, 33))
    if k == 35:   # ahead only
        return seg(X, Y, 0.0, 0.62, 0.0, -0.2, 0.20) | poly(X, Y, [(-0.38, -0.15), (0.0, -0.70), (0.38, -0.15)])
    if k == 36:   # straight or right
        return (seg(X, Y, -0.20, 0.62, -0.20, -0.25, 0.17) | poly(X, Y, [(-0.47, -0.20), (-0.20, -0.68), (0.07, -0.20)])
                | line(X, Y, [(-0.20, 0.42), (-0.05, 0.14), (0.30, 0.12)], 0.15) | poly(X, Y, [(0.28, -0.12), (0.64, 0.12), (0.28, 0.36)]))
    if k == 37:
        return flipx(blue_symbol(X, Y, 36))
    if k == 38:   # keep right: arrow pointing down and to the right
        return seg(X, Y, -0.45, -0.45, 0.20, 0.20, 0.20) | poly(X, Y, [(0.58, 0.58), (0.36, -0.08), (-0.08, 0.36)])
    if k == 39:
        return flipx(blue_symbol(X, Y, 38))
    if k == 40:   # roundabout: ring of three arrows
        m = ell(X, Y, 0.0, 0.0, 0.45, 0.45, 0.16)
        for a in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3):
            m &= ~seg(X, Y, 0.25 * np.cos(a), 0.25 * np.sin(a), 0.70 * np.cos(a), 0.70 * np.sin(a), 0.12)
        return m
    raise ValueError(k)


def draw(k, r):
    """Returns (rgb image, sign mask, inner-symbol mask) on the supersampled grid for class k at half-width r pixels."""
    c = ((np.arange(N * SS) + 0.5) / SS - 0.5 - (N - 1) / 2) / r
    X, Y = np.meshgrid(c, c)
    img = np.full((3,) + X.shape, 0.5)

    def paint(m, col):
        for i in range(3):
            img[i][m] = col[i]

    D = disc(X, Y, 0, 0, 1.0)
    none = np.zeros(X.shape, bool)
    if k in SPEED or k in (9, 10, 15, 16):                      # red ring, white centre, black symbol
        mask, inner = D, disc(X, Y, 0, 0, 0.66)
        paint(D, RED); paint(disc(X, Y, 0, 0, 0.74), WHITE)
        if k in SPEED:
            paint(number(X, Y, SPEED[k]), BLACK)
        elif k == 9:
            paint(car(X, Y, -0.55, -0.07), RED); paint(car(X, Y, 0.07, 0.55), BLACK)
        elif k == 10:
            paint(rect(X, Y, -0.60, -0.30, -0.08, 0.24), RED); paint(car(X, Y, 0.07, 0.55), BLACK)
        elif k == 16:
            paint(rect(X, Y, -0.15, -0.30, 0.55, 0.15) | rect(X, Y, -0.55, -0.10, -0.15, 0.15)
                  | disc(X, Y, -0.35, 0.22, 0.10) | disc(X, Y, 0.30, 0.22, 0.10), BLACK)
    elif k in (6, 32, 41, 42):                                  # "end of" signs: white disc, grey symbol, dark diagonal band
        mask, inner = D, disc(X, Y, 0, 0, 0.80)
        paint(D, DARK); paint(disc(X, Y, 0, 0, 0.93), WHITE)
        if k == 6:
            paint(number(X, Y, "80"), GRAY)
        elif k == 41:
            paint(car(X, Y, -0.55, -0.07) | car(X, Y, 0.07, 0.55), GRAY)
        elif k == 42:
            paint(rect(X, Y, -0.60, -0.30, -0.08, 0.24) | car(X, Y, 0.07, 0.55), GRAY)
        paint(seg(X, Y, -0.64, 0.64, 0.64, -0.64, 0.22) & disc(X, Y, 0, 0, 0.93), DARK)
    elif k == 12:                                               # priority road: white diamond, yellow centre
        dia = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
        mask, inner = poly(X, Y, dia), none
        paint(mask, WHITE); paint(poly(X, Y, scale_pts(dia, (0, 0), 0.60)), YEL)
    elif k == 13:                                               # yield: downward triangle, red border, white centre
        mask, inner = poly(X, Y, TRI_DOWN), none
        paint(mask, RED); paint(poly(X, Y, scale_pts(TRI_DOWN, (0, -1 / 3), 0.58)), WHITE)
    elif k == 14:                                               # stop: red octagon with white letters
        octa = [(1.082 * np.cos(np.pi / 8 + j * np.pi / 4), 1.082 * np.sin(np.pi / 8 + j * np.pi / 4)) for j in range(8)]
        mask, inner = poly(X, Y, octa), none
        paint(mask, WHITE); paint(poly(X, Y, scale_pts(octa, (0, 0), 0.93)), RED)
        m = none.copy()
        for j, ch in enumerate("STOP"):
            m |= glyph(X, Y, ch, -0.665 + j * 0.35, -0.25, 0.28, 0.50, 0.09)
        paint(m, WHITE)
    elif k == 17:                                               # no entry: red disc, white bar
        mask, inner = D, none
        paint(D, RED); paint(rect(X, Y, -0.70, -0.16, 0.70, 0.16), WHITE)
    elif k == 26 or 18 <= k <= 31 or k == 11:                   # danger triangles
        mask, inner = poly(X, Y, TRI_UP), poly(X, Y, scale_pts(TRI_UP, (0, 1 / 3), 0.52))
        paint(mask, RED); paint(poly(X, Y, scale_pts(TRI_UP, (0, 1 / 3), 0.62)), WHITE)
        if k == 26:                                             # traffic lights: red, yellow, green discs
            paint(disc(X, Y, 0, -0.06, 0.12), RED); paint(disc(X, Y, 0, 0.22, 0.12), YEL); paint(disc(X, Y, 0, 0.50, 0.12), GREEN)
        else:
            paint(tri_symbol(X, Y, k), BLACK)
    elif 33 <= k <= 40:                                         # blue discs with white arrows
        mask, inner = D, disc(X, Y, 0, 0, 0.80)
        paint(D, WHITE); paint(disc(X, Y, 0, 0, 0.95), BLUE); paint(blue_symbol(X, Y, k), WHITE)
    else:
        raise ValueError(k)
    return img, mask.astype(float), inner.astype(float)


def down(a):
    return a.reshape(a.shape[:-2] + (N, SS, N, SS)).mean(axis=(-3, -1))


def blur(a):
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    a = (p[..., :-2, :] + 2 * p[..., 1:-1, :] + p[..., 2:, :]) / 4
    return (a[..., :, :-2] + 2 * a[..., :, 1:-1] + a[..., :, 2:]) / 4


FEAT = np.array([[1 / 3, 1 / 3, 1 / 3], [1.0, -0.5, -0.5], [-0.5, -0.5, 1.0]])   # luminance, red opponent, blue opponent


def template(k, r):
    """One drawing -> one template, normalised by itself alone."""
    img, mask, inner = draw(k, r)
    img, m, inn = blur(down(img)), down(mask), down(inner)
    F = np.einsum("fc,chw->fhw", FEAT, img)
    T = np.zeros_like(F)
    for i in range(3):
        d = F[i] - (F[i] * m).sum() / m.sum()
        T[i] = m * d / (np.sqrt((d * d * m).sum() / m.sum()) + 0.1)
    T /= np.sqrt((T * T).sum())
    if inn.sum() > 0:                                           # extra weight on the symbol inside the sign
        d = F[0] - (F[0] * inn).sum() / inn.sum()
        if np.sqrt((d * d * inn).sum() / inn.sum()) > 0.05:
            G = inn * d
            T[0] += LAM * G / np.sqrt((G * G).sum())
    return T


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = len(RADII)
    W = np.zeros((43 * S, 3, N, N), np.float32)
    for k in range(43):
        for s, r in enumerate(RADII):
            W[k * S + s] = template(k, r)
    weights = {
        "feat.W": FEAT.reshape(3, 3, 1, 1).astype(np.float32), "feat.b": np.zeros(3, np.float32),
        "match.W": W, "match.b": np.zeros(43 * S, np.float32),
    }
    side = 2 * PAD + 1
    net = {"layers": [
        {"name": "feat", "op": "conv", "inputs": ["image"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "featn", "op": "norm", "inputs": ["feat"]},
        {"name": "match", "op": "conv", "inputs": ["featn"], "out": 43 * S, "kernel": [N, N], "stride": 1, "padding": PAD},
        {"name": "shift", "op": "maxpool", "inputs": ["match"], "size": [side, side], "stride": [side, side]},
        {"name": "grid", "op": "reshape", "inputs": ["shift"], "shape": [1, 43, S]},
        {"name": "size", "op": "maxpool", "inputs": ["grid"], "size": [1, S], "stride": [1, S]},
        {"name": "scores", "op": "reshape", "inputs": ["size"], "shape": [43]},
    ]}
    (out_dir / "network.json").write_text(json.dumps(net, indent=1))
    np.savez(out_dir / "weights.npz", **weights)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
