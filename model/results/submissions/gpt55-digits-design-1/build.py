import json, sys
from pathlib import Path
import numpy as np

# Hand-written procedural template matcher for centered 28x28 handwritten digits.
# The numbers below are generic stroke-model choices, not fitted to the training set.
T_RAW = 260
T_COARSE = 760
RAW_GRID = (20, 13)      # product = T_RAW
COARSE_GRID = (38, 20)   # product = T_COARSE
SEED = 1729

Y, X = np.mgrid[0:28, 0:28].astype(np.float32)
X = X + 0.5
Y = Y + 0.5


def line(p0, p1, n=18):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    p0 = np.array(p0, dtype=np.float32); p1 = np.array(p1, dtype=np.float32)
    return p0[None, :] * (1 - t[:, None]) + p1[None, :] * t[:, None]


def quad(p0, p1, p2, n=28):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    p0 = np.array(p0, dtype=np.float32); p1 = np.array(p1, dtype=np.float32); p2 = np.array(p2, dtype=np.float32)
    return (1 - t[:, None]) ** 2 * p0 + 2 * (1 - t[:, None]) * t[:, None] * p1 + t[:, None] ** 2 * p2


def cubic(p0, p1, p2, p3, n=36):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    p0 = np.array(p0, dtype=np.float32); p1 = np.array(p1, dtype=np.float32); p2 = np.array(p2, dtype=np.float32); p3 = np.array(p3, dtype=np.float32)
    u = 1 - t
    return (u[:, None] ** 3) * p0 + 3 * (u[:, None] ** 2) * t[:, None] * p1 + 3 * u[:, None] * (t[:, None] ** 2) * p2 + (t[:, None] ** 3) * p3


def arc(cx, cy, rx, ry, a0, a1, n=64):
    aa = np.linspace(a0, a1, n, dtype=np.float32)
    return np.stack([cx + rx * np.cos(aa), cy + ry * np.sin(aa)], axis=1).astype(np.float32)


def transform(pts, rng, kind):
    # Global writing variation: size, translation, slant, and a small rotation.
    dx = rng.uniform(-1.35, 1.35)
    dy = rng.uniform(-1.25, 1.25)
    sx = rng.uniform(0.90, 1.11)
    sy = rng.uniform(0.90, 1.11)
    rot = rng.uniform(-0.13, 0.13)
    sl = rng.uniform(-0.20, 0.20)
    if kind in (1, 7):
        rot += rng.uniform(-0.06, 0.06)
        sl += rng.uniform(-0.09, 0.09)
    c = np.cos(rot); s = np.sin(rot)
    q = pts.astype(np.float32) - np.array([14.0, 14.0], dtype=np.float32)
    xr = c * q[:, 0] - s * q[:, 1]
    yr = s * q[:, 0] + c * q[:, 1]
    x = 14.0 + dx + sx * xr + sl * yr
    y = 14.0 + dy + sy * yr
    return np.stack([x, y], axis=1).astype(np.float32)


def draw_path(canvas, pts, sigma):
    if len(pts) < 2:
        return
    sig2 = np.float32(2.0 * sigma * sigma)
    for a, b in zip(pts[:-1], pts[1:]):
        ax, ay = float(a[0]), float(a[1]); bx, by = float(b[0]), float(b[1])
        vx = bx - ax; vy = by - ay
        den = vx * vx + vy * vy + 1e-6
        t = ((X - ax) * vx + (Y - ay) * vy) / den
        t = np.clip(t, 0.0, 1.0)
        dx = X - (ax + t * vx)
        dy = Y - (ay + t * vy)
        d2 = dx * dx + dy * dy
        val = np.exp(-d2 / sig2)
        canvas[:] = np.maximum(canvas, val)


def render(paths, rng, digit):
    # A little width/intensity variation, still normalized per resulting template.
    sigma = rng.uniform(0.62, 1.12)
    if digit in (0, 6, 8, 9):
        sigma += rng.uniform(0.00, 0.18)
    canvas = np.zeros((28, 28), dtype=np.float32)
    for p in paths:
        draw_path(canvas, transform(p, rng, digit), sigma)
    # Mild pen saturation curve, chosen by hand.
    gamma = rng.uniform(0.82, 1.18)
    canvas = np.clip(canvas, 0.0, 1.0) ** gamma
    return canvas.astype(np.float32)


def digit_paths(d, rng):
    j = lambda a: rng.uniform(-a, a)
    paths = []
    if d == 0:
        rx = rng.uniform(5.5, 7.4); ry = rng.uniform(7.6, 9.7)
        cx = 14 + j(0.45); cy = 14 + j(0.55)
        # Mostly oval, sometimes slightly pointy through transform and unequal radii.
        paths.append(arc(cx, cy, rx, ry, rng.uniform(-0.25, 0.25), 2*np.pi + rng.uniform(-0.25, 0.25), 76))
    elif d == 1:
        style = rng.randint(0, 4)
        x_top = rng.uniform(12.0, 16.2); x_bot = x_top + rng.uniform(-2.3, 1.8)
        y_top = rng.uniform(4.7, 7.5); y_bot = rng.uniform(21.0, 24.0)
        if style == 0:
            paths.append(line((x_top, y_top), (x_bot, y_bot), 30))
        elif style == 1:
            paths.append(np.vstack([quad((x_top-3.8+j(.6), y_top+4.0+j(.5)), (x_top-1.0+j(.4), y_top+0.7+j(.4)), (x_top, y_top), 12), line((x_top, y_top), (x_bot, y_bot), 30)]))
        elif style == 2:
            paths.append(line((x_top, y_top), (x_bot, y_bot), 30))
            paths.append(line((x_bot-3.4+j(.5), y_bot), (x_bot+3.8+j(.5), y_bot+j(.3)), 12))
        else:
            paths.append(np.vstack([line((x_top-3.0+j(.5), y_top+3.0+j(.4)), (x_top, y_top), 10), line((x_top, y_top), (x_bot, y_bot), 30), line((x_bot-3.2+j(.5), y_bot), (x_bot+3.5+j(.5), y_bot), 10)]))
    elif d == 2:
        ytop = rng.uniform(6.0, 8.5); ymid = rng.uniform(13.0, 16.0); ybot = rng.uniform(21.0, 23.5)
        xl = rng.uniform(6.5, 9.0); xr = rng.uniform(18.6, 21.5)
        style = rng.randint(0, 3)
        if style == 0:
            paths.append(np.vstack([
                quad((xl, ytop+2.0), (13.8+j(.5), 3.8+j(.6)), (xr, ytop+1.0), 24),
                quad((xr, ytop+1.0), (22.0+j(.6), 11.8+j(.7)), (15.5+j(.8), ymid), 18),
                quad((15.5+j(.8), ymid), (10.5+j(.8), 18.0+j(.7)), (7.1+j(.7), ybot), 18),
                line((7.1+j(.7), ybot), (21.0+j(.8), ybot+j(.6)), 20)]))
        elif style == 1:
            paths.append(np.vstack([
                quad((xl, ytop+1.5), (13.5+j(.6), 4.0+j(.6)), (20.0+j(.7), ytop+0.8), 24),
                cubic((20.0+j(.7), ytop+0.8), (23.0+j(.5), 13.0+j(.7)), (12.5+j(.7), 17.0+j(.7)), (7.0+j(.7), ybot), 36),
                line((7.0+j(.7), ybot), (21.8+j(.8), ybot+j(.6)), 20)]))
        else:
            paths.append(np.vstack([
                quad((8.0+j(.7), 8.5+j(.8)), (13.0+j(.8), 4.2+j(.5)), (19.7+j(.7), 7.3+j(.6)), 22),
                quad((19.7+j(.7), 7.3+j(.6)), (22.0+j(.6), 11.0+j(.7)), (17.0+j(.8), 14.5+j(.6)), 18),
                line((17.0+j(.8), 14.5+j(.6)), (7.5+j(.8), ybot), 24),
                quad((7.5+j(.8), ybot), (14.5+j(.6), 22.7+j(.5)), (22.0+j(.7), 22.0+j(.5)), 20)]))
    elif d == 3:
        style = rng.randint(0, 3)
        midy = rng.uniform(13.2, 15.2)
        if style == 0:
            paths.append(np.vstack([
                quad((7.5+j(.7), 7.8+j(.7)), (14.5+j(.6), 3.8+j(.6)), (20.4+j(.6), 8.7+j(.7)), 25),
                quad((20.4+j(.6), 8.7+j(.7)), (18.8+j(.7), 13.0+j(.6)), (13.1+j(.7), midy), 18),
                quad((13.1+j(.7), midy), (21.2+j(.8), 15.0+j(.6)), (20.0+j(.7), 20.8+j(.7)), 22),
                quad((20.0+j(.7), 20.8+j(.7)), (14.0+j(.7), 25.0+j(.6)), (7.2+j(.7), 21.0+j(.7)), 25)]))
        elif style == 1:
            paths.append(np.vstack([
                line((8.0+j(.7), 6.5+j(.7)), (19.5+j(.8), 6.8+j(.7)), 14),
                quad((19.5+j(.8), 6.8+j(.7)), (21.0+j(.7), 12.0+j(.7)), (13.0+j(.8), midy), 22),
                quad((13.0+j(.8), midy), (22.0+j(.7), 15.2+j(.7)), (19.5+j(.7), 21.0+j(.7)), 22),
                quad((19.5+j(.7), 21.0+j(.7)), (13.0+j(.8), 24.2+j(.7)), (7.5+j(.8), 20.5+j(.7)), 20)]))
        else:
            paths.append(np.vstack([
                quad((8.2+j(.7), 8.0+j(.7)), (16.0+j(.8), 4.0+j(.6)), (20.5+j(.6), 9.2+j(.7)), 24),
                cubic((20.5+j(.6), 9.2+j(.7)), (17.0+j(.6), 12.5+j(.5)), (16.0+j(.6), 13.2+j(.4)), (13.0+j(.7), midy), 20),
                cubic((13.0+j(.7), midy), (20.0+j(.8), 15.3+j(.5)), (22.0+j(.7), 20.5+j(.7)), (15.0+j(.8), 23.2+j(.7)), 30),
                quad((15.0+j(.8), 23.2+j(.7)), (10.8+j(.7), 23.8+j(.6)), (7.0+j(.8), 21.0+j(.8)), 14)]))
    elif d == 4:
        style = rng.randint(0, 4)
        if style == 0:
            paths.append(line((17.8+j(.8), 5.0+j(.8)), (8.0+j(.7), 17.0+j(.7)), 24))
            paths.append(line((8.0+j(.7), 17.0+j(.7)), (21.5+j(.7), 17.0+j(.6)), 18))
            paths.append(line((18.0+j(.7), 5.2+j(.7)), (17.5+j(.7), 23.0+j(.8)), 30))
        elif style == 1:
            paths.append(line((9.2+j(.7), 5.8+j(.8)), (8.6+j(.7), 17.2+j(.7)), 22))
            paths.append(line((8.6+j(.7), 17.2+j(.7)), (21.5+j(.7), 17.2+j(.6)), 18))
            paths.append(line((18.5+j(.7), 5.5+j(.7)), (18.0+j(.7), 23.0+j(.8)), 30))
        elif style == 2:
            paths.append(np.vstack([line((17.5+j(.7), 5.0+j(.8)), (9.0+j(.7), 16.5+j(.7)), 22), line((9.0+j(.7), 16.5+j(.7)), (21.8+j(.7), 17.0+j(.7)), 18)]))
            paths.append(line((18.0+j(.7), 5.2+j(.8)), (18.0+j(.7), 23.5+j(.7)), 30))
        else:
            paths.append(line((10.0+j(.8), 5.5+j(.8)), (8.5+j(.7), 17.5+j(.7)), 22))
            paths.append(line((8.5+j(.7), 17.5+j(.7)), (21.2+j(.7), 17.0+j(.6)), 18))
            paths.append(line((20.0+j(.7), 6.0+j(.8)), (15.8+j(.7), 23.0+j(.8)), 30))
    elif d == 5:
        style = rng.randint(0, 3)
        if style == 0:
            paths.append(np.vstack([
                line((20.5+j(.7), 6.0+j(.6)), (8.3+j(.7), 6.0+j(.6)), 17),
                line((8.3+j(.7), 6.0+j(.6)), (7.7+j(.7), 13.5+j(.7)), 15),
                quad((7.7+j(.7), 13.5+j(.7)), (13.5+j(.8), 12.2+j(.6)), (17.8+j(.7), 14.3+j(.7)), 18),
                quad((17.8+j(.7), 14.3+j(.7)), (22.0+j(.7), 18.8+j(.7)), (16.8+j(.8), 22.5+j(.7)), 24),
                quad((16.8+j(.8), 22.5+j(.7)), (11.8+j(.8), 24.0+j(.6)), (7.2+j(.8), 20.5+j(.8)), 18)]))
        elif style == 1:
            paths.append(np.vstack([
                line((20.5+j(.7), 6.2+j(.6)), (9.0+j(.7), 6.1+j(.6)), 16),
                line((9.0+j(.7), 6.1+j(.6)), (8.0+j(.7), 13.4+j(.7)), 14),
                line((8.0+j(.7), 13.4+j(.7)), (16.5+j(.8), 13.2+j(.7)), 13),
                cubic((16.5+j(.8), 13.2+j(.7)), (23.0+j(.6), 15.8+j(.7)), (21.0+j(.8), 24.0+j(.7)), (9.0+j(.8), 21.0+j(.8)), 34)]))
        else:
            paths.append(np.vstack([
                line((20.0+j(.8), 6.0+j(.6)), (9.0+j(.8), 6.2+j(.6)), 16),
                quad((9.0+j(.8), 6.2+j(.6)), (7.2+j(.6), 10.0+j(.7)), (8.5+j(.8), 13.8+j(.7)), 16),
                quad((8.5+j(.8), 13.8+j(.7)), (14.0+j(.8), 11.5+j(.6)), (18.3+j(.7), 14.8+j(.7)), 18),
                quad((18.3+j(.7), 14.8+j(.7)), (22.0+j(.7), 19.0+j(.7)), (16.0+j(.8), 22.5+j(.8)), 24),
                line((16.0+j(.8), 22.5+j(.8)), (8.2+j(.8), 21.0+j(.8)), 13)]))
    elif d == 6:
        style = rng.randint(0, 3)
        if style == 0:
            paths.append(quad((18.5+j(.8), 5.5+j(.8)), (9.0+j(.8), 7.0+j(.8)), (8.5+j(.7), 15.5+j(.6)), 28))
            paths.append(arc(14.0+j(.5), 17.0+j(.5), rng.uniform(5.1, 6.7), rng.uniform(5.4, 6.9), -0.15, 2*np.pi+0.20, 64))
        elif style == 1:
            paths.append(np.vstack([
                cubic((18.5+j(.8), 5.5+j(.8)), (10.0+j(.8), 5.5+j(.8)), (6.8+j(.7), 13.5+j(.7)), (9.5+j(.6), 18.0+j(.5)), 36),
                arc(14.2+j(.5), 17.2+j(.5), rng.uniform(5.2, 6.8), rng.uniform(5.1, 6.7), np.pi*0.85, np.pi*2.85, 60)]))
        else:
            paths.append(cubic((18.0+j(.8), 5.5+j(.8)), (10.0+j(.8), 6.0+j(.8)), (8.0+j(.7), 12.0+j(.7)), (8.8+j(.6), 16.5+j(.5)), 32))
            paths.append(arc(14.0+j(.5), 17.0+j(.5), rng.uniform(5.2, 6.9), rng.uniform(5.4, 7.0), 0.0, 2*np.pi, 66))
    elif d == 7:
        style = rng.randint(0, 4)
        top_y = rng.uniform(6.0, 8.4)
        left_x = rng.uniform(6.5, 9.0); right_x = rng.uniform(19.0, 22.0)
        bot_x = rng.uniform(10.0, 14.2); bot_y = rng.uniform(21.5, 24.2)
        paths.append(line((left_x, top_y), (right_x, top_y+j(.4)), 18))
        if style == 0:
            paths.append(line((right_x, top_y+j(.4)), (bot_x, bot_y), 32))
        else:
            paths.append(quad((right_x, top_y+j(.4)), (17.0+j(.8), 13.0+j(.8)), (bot_x, bot_y), 32))
        if style == 2:
            paths.append(line((11.0+j(.7), 14.8+j(.7)), (18.0+j(.7), 14.2+j(.7)), 12))
        elif style == 3:
            paths.append(line((9.0+j(.7), top_y+1.5+j(.5)), (left_x, top_y), 8))
    elif d == 8:
        style = rng.randint(0, 3)
        if style == 0:
            paths.append(arc(14.0+j(.4), 10.0+j(.5), rng.uniform(4.6, 6.0), rng.uniform(4.6, 5.9), 0, 2*np.pi, 54))
            paths.append(arc(14.0+j(.4), 18.0+j(.5), rng.uniform(5.0, 6.6), rng.uniform(5.4, 6.8), 0, 2*np.pi, 62))
        elif style == 1:
            paths.append(np.vstack([
                arc(14.0+j(.4), 10.0+j(.5), rng.uniform(4.8, 6.2), rng.uniform(4.6, 5.9), -np.pi*0.15, np.pi*1.85, 54),
                arc(14.0+j(.4), 18.0+j(.5), rng.uniform(5.1, 6.8), rng.uniform(5.3, 6.8), np.pi*0.85, np.pi*2.85, 62)]))
        else:
            paths.append(np.vstack([
                cubic((14+j(.5), 4.8+j(.5)), (5.8+j(.7), 7.0+j(.7)), (8.0+j(.7), 14.0+j(.5)), (14+j(.5), 14.0+j(.4)), 34),
                cubic((14+j(.5), 14.0+j(.4)), (22.0+j(.7), 14.5+j(.6)), (22.0+j(.7), 23.0+j(.7)), (14+j(.5), 23.4+j(.6)), 34),
                cubic((14+j(.5), 23.4+j(.6)), (5.5+j(.7), 21.5+j(.7)), (7.0+j(.7), 15.0+j(.6)), (14+j(.5), 14.0+j(.4)), 34),
                cubic((14+j(.5), 14.0+j(.4)), (21.0+j(.7), 13.0+j(.6)), (21.0+j(.7), 6.0+j(.7)), (14+j(.5), 4.8+j(.5)), 34)]))
    elif d == 9:
        style = rng.randint(0, 3)
        if style == 0:
            paths.append(arc(14.0+j(.5), 10.8+j(.5), rng.uniform(5.1, 6.7), rng.uniform(5.2, 6.8), -0.1, 2*np.pi+0.1, 64))
            paths.append(quad((18.6+j(.6), 13.0+j(.6)), (18.2+j(.8), 20.5+j(.8)), (11.5+j(.8), 23.2+j(.7)), 30))
        elif style == 1:
            paths.append(np.vstack([
                arc(14.0+j(.5), 10.8+j(.5), rng.uniform(5.2, 6.8), rng.uniform(5.2, 6.8), -0.2, 2*np.pi-0.05, 60),
                line((18.8+j(.6), 13.5+j(.6)), (17.0+j(.8), 23.0+j(.8)), 24)]))
        else:
            paths.append(arc(14.0+j(.5), 10.8+j(.5), rng.uniform(5.0, 6.6), rng.uniform(5.0, 6.7), -0.1, 2*np.pi+0.1, 62))
            paths.append(cubic((18.8+j(.6), 13.0+j(.6)), (19.5+j(.7), 18.0+j(.8)), (15.5+j(.8), 22.0+j(.8)), (11.0+j(.8), 23.5+j(.7)), 30))
    return paths


def avg2(a):
    return a.reshape(14, 2, 14, 2).mean(axis=(1, 3))


def normalize_template(a):
    z = a.astype(np.float32) - np.float32(a.mean())
    n = np.sqrt(np.float32((z * z).sum()) + np.float32(1e-6))
    return (z / n).astype(np.float32)


def make_weights():
    rng = np.random.RandomState(SEED)
    wr = np.empty((10 * T_RAW, 1 * 28 * 28), dtype=np.float32)
    wc = np.empty((10 * T_COARSE, 1 * 14 * 14), dtype=np.float32)
    for d in range(10):
        for i in range(max(T_RAW, T_COARSE)):
            # Independent generated template; if both banks use it, each normalization is still by itself alone.
            im = render(digit_paths(d, rng), rng, d)
            if i < T_RAW:
                wr[d * T_RAW + i] = normalize_template(im).reshape(-1)
            if i < T_COARSE:
                wc[d * T_COARSE + i] = (2.0 * normalize_template(avg2(im))).reshape(-1)
    weights = {
        'raw_match.W': wr,
        'raw_match.b': np.zeros((10 * T_RAW,), dtype=np.float32),
        'coarse_match.W': wc,
        'coarse_match.b': np.zeros((10 * T_COARSE,), dtype=np.float32),
    }
    return weights


def build(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = [
        {'name': 'raw_norm', 'op': 'norm', 'inputs': ['image']},
        {'name': 'raw_match', 'op': 'dense', 'inputs': ['raw_norm'], 'out': 10 * T_RAW},
        {'name': 'raw_grid', 'op': 'reshape', 'inputs': ['raw_match'], 'shape': [10, RAW_GRID[0], RAW_GRID[1]]},
        {'name': 'raw_best', 'op': 'maxpool', 'inputs': ['raw_grid'], 'size': [RAW_GRID[0], RAW_GRID[1]], 'stride': [RAW_GRID[0], RAW_GRID[1]]},
        {'name': 'raw_scores', 'op': 'reshape', 'inputs': ['raw_best'], 'shape': [10]},
        {'name': 'coarse_pool', 'op': 'avgpool', 'inputs': ['image'], 'size': [2, 2], 'stride': [2, 2]},
        {'name': 'coarse_norm', 'op': 'norm', 'inputs': ['coarse_pool']},
        {'name': 'coarse_match', 'op': 'dense', 'inputs': ['coarse_norm'], 'out': 10 * T_COARSE},
        {'name': 'coarse_grid', 'op': 'reshape', 'inputs': ['coarse_match'], 'shape': [10, COARSE_GRID[0], COARSE_GRID[1]]},
        {'name': 'coarse_best', 'op': 'maxpool', 'inputs': ['coarse_grid'], 'size': [COARSE_GRID[0], COARSE_GRID[1]], 'stride': [COARSE_GRID[0], COARSE_GRID[1]]},
        {'name': 'coarse_scores', 'op': 'reshape', 'inputs': ['coarse_best'], 'shape': [10]},
        {'name': 'scores', 'op': 'add', 'inputs': ['raw_scores', 'coarse_scores']},
    ]
    (out / 'network.json').write_text(json.dumps({'layers': layers}, separators=(',', ':')))
    np.savez(out / 'weights.npz', **make_weights())

if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '/app/out')
