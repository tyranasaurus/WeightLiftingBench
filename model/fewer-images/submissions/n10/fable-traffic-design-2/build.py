"""Hand-designed traffic sign network: every weight comes from sign drawings typed here from general knowledge.
Usage: python3 build.py OUT_DIR   (reads no files)"""
import sys, os, json
import numpy as np

N, SS = 32, 4                              # frame size, supersampling used when drawing
SCALES = [17.0, 20.0, 23.0, 26.0, 29.0]    # sign diameters in pixels (signs fill about 55% to 90% of the frame)
RED = (0.75, 0.10, 0.10); WHITE = (0.95, 0.95, 0.95); BLACK = (0.08, 0.08, 0.08); BLUE = (0.10, 0.30, 0.75)
YELLOW = (0.95, 0.75, 0.10); GREY = (0.35, 0.35, 0.35); GREEN = (0.10, 0.60, 0.30); LIGHT = (0.90, 0.90, 0.90)


class Cv:
    """Canvas in sign units: the sign's bounding box is u, v in [-1, 1] (v points down)."""
    def __init__(s, d):
        c = ((np.arange(N * SS) + 0.5) / SS - N / 2.0) / (d / 2.0)
        s.U, s.V = np.meshgrid(c, c)
        s.rgb = np.zeros((3, N * SS, N * SS)); s.mask = np.zeros((N * SS, N * SS), bool)

    def paint(s, m, col):
        for k in range(3): s.rgb[k][m] = col[k]

    def disc(s, cx, cy, r): return (s.U - cx) ** 2 + (s.V - cy) ** 2 <= r * r

    def rect(s, x0, y0, x1, y1): return (s.U >= x0) & (s.U <= x1) & (s.V >= y0) & (s.V <= y1)

    def poly(s, pts):
        inside = np.zeros(s.U.shape, bool); n = len(pts)
        for i in range(n):
            (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
            if y1 == y2: continue
            inside ^= ((y1 > s.V) != (y2 > s.V)) & (s.U < (x2 - x1) * (s.V - y1) / (y2 - y1) + x1)
        return inside

    def stroke(s, pts, w):
        m = np.zeros(s.U.shape, bool)
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            dx, dy = x2 - x1, y2 - y1
            t = np.clip(((s.U - x1) * dx + (s.V - y1) * dy) / max(dx * dx + dy * dy, 1e-12), 0, 1)
            m |= (s.U - x1 - t * dx) ** 2 + (s.V - y1 - t * dy) ** 2 <= (w / 2.0) ** 2
        return m


def ell(cx, cy, rx, ry, n=24):
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in np.linspace(0, 2 * np.pi, n + 1)]


GLYPH = {  # strokes in a unit box, x right, y down
    '0': [ell(0.5, 0.5, 0.5, 0.5)],
    '1': [[(0.0, 0.25), (1.0, 0.0), (1.0, 1.0)]],
    '2': [[(0.02, 0.27), (0.15, 0.07), (0.5, 0.0), (0.85, 0.07), (0.98, 0.27), (0.9, 0.48), (0.0, 1.0), (1.0, 1.0)]],
    '3': [[(0.03, 0.13), (0.3, 0.0), (0.7, 0.0), (0.95, 0.12), (0.95, 0.36), (0.7, 0.48), (0.4, 0.48)],
          [(0.7, 0.48), (0.97, 0.62), (0.97, 0.87), (0.7, 1.0), (0.3, 1.0), (0.03, 0.86)]],
    '5': [[(0.95, 0.0), (0.12, 0.0), (0.08, 0.45), (0.35, 0.38), (0.7, 0.38), (0.97, 0.55), (0.97, 0.85), (0.7, 1.0), (0.3, 1.0), (0.03, 0.87)]],
    '6': [[(0.9, 0.1), (0.65, 0.0), (0.35, 0.0), (0.08, 0.2), (0.03, 0.5), (0.03, 0.8), (0.3, 1.0), (0.7, 1.0), (0.97, 0.82), (0.97, 0.6),
           (0.7, 0.42), (0.35, 0.42), (0.05, 0.6)]],
    '7': [[(0.0, 0.0), (1.0, 0.0), (0.4, 1.0)]],
    '8': [ell(0.5, 0.25, 0.42, 0.25), ell(0.5, 0.74, 0.5, 0.26)],
    'S': [[(0.95, 0.12), (0.7, 0.0), (0.3, 0.0), (0.05, 0.12), (0.05, 0.38), (0.3, 0.5), (0.7, 0.5), (0.95, 0.62), (0.95, 0.88), (0.7, 1.0),
           (0.3, 1.0), (0.05, 0.88)]],
    'T': [[(0.0, 0.0), (1.0, 0.0)], [(0.5, 0.0), (0.5, 1.0)]],
    'O': [ell(0.5, 0.5, 0.5, 0.5)],
    'P': [[(0.05, 1.0), (0.05, 0.0), (0.7, 0.0), (0.95, 0.12), (0.95, 0.4), (0.7, 0.52), (0.05, 0.52)]],
}


def text(cv, chars, boxes, sw, col):
    for ch, (cx, cy, w, h) in zip(chars, boxes):
        for pl in GLYPH[ch]:
            cv.paint(cv.stroke([(cx + (x - 0.5) * w, cy + (y - 0.5) * h) for x, y in pl], sw), col)


def speed(cv, num, col=BLACK):
    if len(num) == 2:
        text(cv, num, [(-0.29, 0.0, 0.40, 0.78), (0.29, 0.0, 0.40, 0.78)], 0.13, col)
    else:
        text(cv, num, [(-0.52, 0.0, 0.13, 0.70), (-0.13, 0.0, 0.34, 0.70), (0.36, 0.0, 0.34, 0.70)], 0.11, col)


def ring(cv):
    m = cv.disc(0, 0, 1.0); cv.paint(m, RED); cv.paint(cv.disc(0, 0, 0.74), WHITE); cv.mask = m


def white_disc(cv):
    m = cv.disc(0, 0, 1.0); cv.paint(m, LIGHT); cv.mask = m


def blue_disc(cv):
    m = cv.disc(0, 0, 1.0); cv.paint(m, BLUE); cv.mask = m


def car(cv, cx, cy, hw, col):
    cv.paint(cv.rect(cx - hw, cy, cx + hw, cy + 0.2), col)
    cv.paint(cv.rect(cx - 0.62 * hw, cy - 0.17, cx + 0.62 * hw, cy), col)


def cars(cv, left, right):          # no passing: two cars side by side
    car(cv, -0.31, -0.02, 0.26, left); car(cv, 0.31, -0.02, 0.26, right)


def car_truck(cv, left, right):     # no passing for trucks: car on the left, truck on the right
    car(cv, -0.36, 0.02, 0.22, left)
    cv.paint(cv.rect(0.0, -0.3, 0.6, 0.2), right)
    cv.paint(cv.disc(0.14, 0.24, 0.08), right); cv.paint(cv.disc(0.46, 0.24, 0.08), right)


def slash(cv, w, col):
    cv.paint(cv.stroke([(-0.68, 0.68), (0.68, -0.68)], w) & cv.disc(0, 0, 1.0), col)


def triangle(cv):
    out = [(0.0, -0.95), (1.0, 0.85), (-1.0, 0.85)]; cx, cy, k = 0.0, 0.25, 0.62
    m = cv.poly(out); cv.paint(m, RED); cv.paint(cv.poly([(cx + k * (x - cx), cy + k * (y - cy)) for x, y in out]), WHITE); cv.mask = m


def arrow(cv, pts, w, head, col=WHITE):
    cv.paint(cv.stroke(pts, w), col); cv.paint(cv.poly(head), col)


def person(cv, x, top, size, col=BLACK):
    z = size
    cv.paint(cv.disc(x, top, 0.075 * z), col)
    cv.paint(cv.stroke([(x, top + 0.12 * z), (x, top + 0.42 * z)], 0.13 * z), col)
    cv.paint(cv.stroke([(x, top + 0.42 * z), (x - 0.15 * z, top + 0.71 * z)], 0.08 * z), col)
    cv.paint(cv.stroke([(x, top + 0.42 * z), (x + 0.14 * z, top + 0.71 * z)], 0.08 * z), col)
    cv.paint(cv.stroke([(x, top + 0.16 * z), (x - 0.15 * z, top + 0.36 * z)], 0.06 * z), col)
    cv.paint(cv.stroke([(x, top + 0.16 * z), (x + 0.15 * z, top + 0.34 * z)], 0.06 * z), col)


MIRROR = {19: 20, 34: 33, 37: 36, 39: 38}
SPEEDS = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}


def draw(c, cv):
    if c in MIRROR:
        cv.U = -cv.U; c = MIRROR[c]
    if c in SPEEDS:
        ring(cv); speed(cv, SPEEDS[c])
    elif c == 6:
        white_disc(cv); speed(cv, '80', GREY); slash(cv, 0.14, BLACK)
    elif c == 9:
        ring(cv); cars(cv, RED, BLACK)
    elif c == 10:
        ring(cv); car_truck(cv, RED, BLACK)
    elif c == 11:
        triangle(cv)
        cv.paint(cv.stroke([(0.0, -0.12), (0.0, 0.52)], 0.15), BLACK)
        cv.paint(cv.poly([(0.0, -0.3), (0.15, 0.0), (-0.15, 0.0)]), BLACK)
        cv.paint(cv.stroke([(-0.3, 0.25), (0.3, 0.25)], 0.13), BLACK)
    elif c == 12:
        m = cv.poly([(0, -1), (1, 0), (0, 1), (-1, 0)]); cv.paint(m, WHITE)
        cv.paint(cv.poly([(0, -0.6), (0.6, 0), (0, 0.6), (-0.6, 0)]), YELLOW); cv.mask = m
    elif c == 13:
        out = [(-1.0, -0.85), (1.0, -0.85), (0.0, 0.95)]; cx, cy, k = 0.0, -0.25, 0.62
        m = cv.poly(out); cv.paint(m, RED); cv.paint(cv.poly([(cx + k * (x - cx), cy + k * (y - cy)) for x, y in out]), WHITE); cv.mask = m
    elif c == 14:
        def octa(ap):
            r = ap / np.cos(np.pi / 8)
            return [(r * np.cos(np.pi / 8 + k * np.pi / 4), r * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)]
        m = cv.poly(octa(1.0)); cv.paint(m, WHITE); cv.paint(cv.poly(octa(0.9)), RED); cv.mask = m
        text(cv, 'STOP', [(-0.57, 0.0, 0.26, 0.56), (-0.19, 0.0, 0.26, 0.56), (0.19, 0.0, 0.26, 0.56), (0.57, 0.0, 0.26, 0.56)], 0.10, WHITE)
    elif c == 15:
        ring(cv)
    elif c == 16:
        ring(cv)
        cv.paint(cv.rect(-0.55, -0.08, -0.27, 0.18), BLACK); cv.paint(cv.rect(-0.22, -0.3, 0.55, 0.18), BLACK)
        cv.paint(cv.disc(-0.35, 0.24, 0.1), BLACK); cv.paint(cv.disc(0.35, 0.24, 0.1), BLACK)
    elif c == 17:
        m = cv.disc(0, 0, 1.0); cv.paint(m, RED); cv.paint(cv.rect(-0.72, -0.16, 0.72, 0.16), WHITE); cv.mask = m
    elif c == 18:
        triangle(cv); cv.paint(cv.stroke([(0.0, -0.2), (0.0, 0.28)], 0.14), BLACK); cv.paint(cv.disc(0.0, 0.47, 0.085), BLACK)
    elif c == 20:
        triangle(cv)
        arrow(cv, [(-0.1, 0.55), (-0.1, 0.2), (0.0, 0.1), (0.2, 0.1)], 0.13, [(0.2, -0.07), (0.2, 0.27), (0.4, 0.1)], BLACK)
    elif c == 21:
        triangle(cv); cv.paint(cv.stroke([(0.14, 0.55), (0.14, 0.38), (-0.12, 0.2), (-0.12, -0.05)], 0.12), BLACK)
    elif c == 22:
        triangle(cv); cv.paint(cv.rect(-0.4, 0.42, 0.4, 0.52), BLACK)
        cv.paint((cv.disc(-0.18, 0.42, 0.15) | cv.disc(0.18, 0.42, 0.15)) & (cv.V <= 0.42), BLACK)
    elif c == 23:
        triangle(cv); cv.paint(cv.rect(-0.17, 0.0, 0.17, 0.22), BLACK); cv.paint(cv.rect(-0.1, -0.12, 0.1, 0.0), BLACK)
        cv.paint(cv.stroke([(-0.12, 0.28), (-0.03, 0.4), (-0.16, 0.54)], 0.06), BLACK)
        cv.paint(cv.stroke([(0.12, 0.28), (0.21, 0.4), (0.08, 0.54)], 0.06), BLACK)
    elif c == 24:
        triangle(cv); cv.paint(cv.stroke([(-0.13, -0.05), (-0.13, 0.55)], 0.1), BLACK)
        cv.paint(cv.stroke([(0.24, 0.55), (0.24, 0.32), (0.08, 0.15), (0.08, -0.05)], 0.1), BLACK)
    elif c == 25:
        triangle(cv); cv.paint(cv.disc(-0.02, -0.1, 0.08), BLACK)
        cv.paint(cv.stroke([(-0.02, 0.0), (-0.14, 0.28)], 0.13), BLACK)
        cv.paint(cv.stroke([(-0.14, 0.28), (-0.27, 0.54)], 0.08), BLACK); cv.paint(cv.stroke([(-0.14, 0.28), (-0.02, 0.54)], 0.08), BLACK)
        cv.paint(cv.stroke([(0.0, 0.1), (0.2, 0.38)], 0.06), BLACK)
        cv.paint(cv.disc(0.27, 0.54, 0.17) & (cv.V <= 0.54), BLACK)
    elif c == 26:
        triangle(cv); cv.paint(cv.disc(0.0, -0.08, 0.11), RED); cv.paint(cv.disc(0.0, 0.18, 0.11), YELLOW); cv.paint(cv.disc(0.0, 0.44, 0.11), GREEN)
    elif c == 27:
        triangle(cv); person(cv, 0.0, -0.16, 1.0)
    elif c == 28:
        triangle(cv); person(cv, 0.13, -0.1, 0.92); person(cv, -0.2, 0.1, 0.63)
        cv.paint(cv.stroke([(-0.12, 0.24), (0.05, 0.16)], 0.05), BLACK)
    elif c == 29:
        triangle(cv)
        cv.paint(cv.stroke(ell(-0.22, 0.4, 0.15, 0.15), 0.05), BLACK); cv.paint(cv.stroke(ell(0.22, 0.4, 0.15, 0.15), 0.05), BLACK)
        cv.paint(cv.stroke([(-0.22, 0.4), (-0.06, 0.15), (0.14, 0.15), (0.22, 0.4)], 0.05), BLACK)
        cv.paint(cv.stroke([(-0.06, 0.15), (0.02, 0.4), (0.14, 0.15)], 0.05), BLACK)
    elif c == 30:
        triangle(cv)
        for a in (90.0, 30.0, 150.0):
            dx, dy = 0.28 * np.cos(np.radians(a)), 0.28 * np.sin(np.radians(a))
            cv.paint(cv.stroke([(-dx, 0.25 - dy), (dx, 0.25 + dy)], 0.07), BLACK)
        cv.paint(cv.disc(0.0, 0.25, 0.08), BLACK)
    elif c == 31:
        triangle(cv); cv.paint(cv.stroke([(-0.22, 0.33), (0.12, 0.15)], 0.15), BLACK)
        cv.paint(cv.stroke([(0.12, 0.15), (0.24, -0.08), (0.34, -0.04)], 0.07), BLACK)
        cv.paint(cv.stroke([(0.1, 0.2), (0.3, 0.36)], 0.05), BLACK); cv.paint(cv.stroke([(-0.22, 0.35), (-0.38, 0.54)], 0.06), BLACK)
    elif c == 32:
        white_disc(cv); slash(cv, 0.26, (0.25, 0.25, 0.25))
    elif c == 33:
        blue_disc(cv)
        arrow(cv, [(-0.22, 0.58), (-0.22, 0.05), (-0.12, -0.12), (0.05, -0.18), (0.25, -0.18)], 0.18, [(0.22, -0.48), (0.22, 0.12), (0.62, -0.18)])
    elif c == 35:
        blue_disc(cv); arrow(cv, [(0.0, 0.62), (0.0, -0.2)], 0.2, [(-0.33, -0.18), (0.33, -0.18), (0.0, -0.68)])
    elif c == 36:
        blue_disc(cv)
        arrow(cv, [(-0.18, 0.62), (-0.18, -0.25)], 0.16, [(-0.42, -0.25), (0.06, -0.25), (-0.18, -0.68)])
        arrow(cv, [(-0.18, 0.35), (0.0, 0.12), (0.3, 0.12)], 0.14, [(0.3, -0.12), (0.3, 0.36), (0.64, 0.12)])
    elif c == 38:
        blue_disc(cv); arrow(cv, [(-0.42, -0.42), (0.15, 0.15)], 0.2, [(0.42, -0.12), (-0.12, 0.42), (0.55, 0.55)])
    elif c == 40:
        blue_disc(cv)
        for a0 in (90.0, 210.0, 330.0):
            ang = np.radians(np.linspace(a0, a0 + 75.0, 12))
            cv.paint(cv.stroke([(0.48 * np.cos(t), -0.48 * np.sin(t)) for t in ang], 0.15), WHITE)
            a1, a2 = np.radians(a0 + 75.0), np.radians(a0 + 100.0)
            cv.paint(cv.poly([(0.28 * np.cos(a1), -0.28 * np.sin(a1)), (0.68 * np.cos(a1), -0.68 * np.sin(a1)),
                              (0.48 * np.cos(a2), -0.48 * np.sin(a2))]), WHITE)
    elif c == 41:
        white_disc(cv); cars(cv, GREY, GREY); slash(cv, 0.14, BLACK)
    elif c == 42:
        white_disc(cv); car_truck(cv, GREY, GREY); slash(cv, 0.14, BLACK)
    else:
        raise ValueError(c)


def features(rgb):
    """Luminance, red-ness, blue-ness: the same three features the first conv computes from the photo."""
    return np.stack([rgb.mean(0), rgb[0] - (rgb[1] + rgb[2]) / 2.0, rgb[2] - (rgb[0] + rgb[1]) / 2.0])


def template(c, d):
    """One drawing -> one template, normalised by itself alone: zero mean inside the sign, zero outside, unit length."""
    cv = Cv(d); draw(c, cv)
    rgb = cv.rgb.reshape(3, N, SS, N, SS).mean((2, 4))
    cov = cv.mask.astype(np.float64).reshape(N, SS, N, SS).mean((1, 3))
    F = features(rgb)
    mu = (F * cov).sum((1, 2)) / cov.sum()
    T = (F - mu[:, None, None]) * cov
    return T / np.sqrt((T ** 2).sum())


def main(out_dir):
    S = len(SCALES)
    g = np.outer([1.0, 2.0, 1.0], [1.0, 2.0, 1.0]) / 16.0          # light smoothing of the jagged nearest-neighbour pixels
    W1 = np.zeros((3, 3, 3, 3))
    W1[0, :] = g / 3.0
    W1[1, 0] = g; W1[1, 1] = -g / 2.0; W1[1, 2] = -g / 2.0
    W1[2, 2] = g; W1[2, 0] = -g / 2.0; W1[2, 1] = -g / 2.0
    W2 = np.zeros((43 * S, 3, N, N))
    for c in range(43):
        for j, d in enumerate(SCALES):
            W2[c * S + j] = template(c, d)
    layers = [
        {"name": "feat", "op": "conv", "inputs": ["image"], "out": 3, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "match", "op": "conv", "inputs": ["feat"], "out": 43 * S, "kernel": [N, N], "stride": 1, "padding": 3},
        {"name": "shiftmax", "op": "maxpool", "inputs": ["match"], "size": [7, 7], "stride": [7, 7]},
        {"name": "grid", "op": "reshape", "inputs": ["shiftmax"], "shape": [1, 43, S]},
        {"name": "scalemax", "op": "maxpool", "inputs": ["grid"], "size": [1, S], "stride": [1, S]},
        {"name": "scores", "op": "reshape", "inputs": ["scalemax"], "shape": [43]},
    ]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{
        "feat.W": W1.astype(np.float32), "feat.b": np.zeros(3, np.float32),
        "match.W": W2.astype(np.float32), "match.b": np.zeros(43 * S, np.float32)})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
