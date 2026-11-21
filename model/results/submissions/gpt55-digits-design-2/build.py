import json, sys
from pathlib import Path
import numpy as np

T = 11264
GRID = np.arange(28, dtype=np.float32)
Y, X = np.meshgrid(GRID, GRID, indexing="ij")
EPS = 1e-5


def line(p0, p1, n=60):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    p0 = np.array(p0, dtype=np.float32); p1 = np.array(p1, dtype=np.float32)
    return (1 - t) * p0 + t * p1


def quad(p0, p1, p2, n=70):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    p0 = np.array(p0, dtype=np.float32); p1 = np.array(p1, dtype=np.float32); p2 = np.array(p2, dtype=np.float32)
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def cubic(p0, p1, p2, p3, n=90):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    p0 = np.array(p0, dtype=np.float32); p1 = np.array(p1, dtype=np.float32); p2 = np.array(p2, dtype=np.float32); p3 = np.array(p3, dtype=np.float32)
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def arc(cx, cy, rx, ry, a0, a1, n=120):
    th = np.linspace(a0, a1, n, dtype=np.float32)
    return np.stack([cx + rx * np.cos(th), cy + ry * np.sin(th)], axis=1).astype(np.float32)


def digit_paths(d, rng):
    R = rng.uniform
    strokes = []
    if d == 0:
        rx, ry = R(0.48, 0.72), R(0.72, 1.02)
        th = np.linspace(0, 2*np.pi, 150, dtype=np.float32)
        wob = R(-0.08, 0.08)
        x = rx * np.cos(th) * (1 + wob * np.sin(th))
        y = ry * np.sin(th) * (1 + R(-0.05, 0.05) * np.cos(th))
        strokes.append(np.stack([x, y], 1).astype(np.float32))
    elif d == 1:
        top = (R(-0.10, 0.22), R(-1.02, -0.78)); bot = (R(-0.15, 0.18), R(0.75, 1.05))
        midx = R(-0.10, 0.12)
        strokes.append(cubic(top, (midx+R(-.08,.08), -0.35), (midx+R(-.08,.08), 0.25), bot, 85))
        if rng.random() < 0.65:
            strokes.append(line((top[0]-R(0.25,0.55), top[1]+R(0.15,0.35)), top, 35))
        if rng.random() < 0.28:
            y = bot[1] + R(-0.04, 0.05)
            strokes.append(line((bot[0]-R(0.25,0.45), y), (bot[0]+R(0.22,0.42), y), 35))
    elif d == 2:
        st = (R(-0.65,-0.35), R(-0.62,-0.38)); rt = (R(0.35,0.72), R(-0.60,-0.34))
        lo = (R(-0.65,-0.35), R(0.48,0.78)); en = (R(0.42,0.78), R(0.62,0.90))
        p = cubic(st, (R(-0.55,-0.15), R(-1.08,-0.82)), (R(0.45,0.75), R(-1.05,-0.75)), rt, 70)
        q = cubic(rt, (R(0.72,0.85), R(-0.02,0.20)), (R(-0.18,0.10), R(0.20,0.45)), lo, 80)
        r = line(lo, en, 45)
        if rng.random() < 0.35:
            # a straighter Z-like two
            q = line(rt, lo, 70)
        strokes.append(np.vstack([p, q[1:], r[1:]]))
    elif d == 3:
        p0 = (R(-0.62,-0.35), R(-0.86,-0.58)); mid = (R(-0.12,0.18), R(-0.10,0.12)); p3 = (R(-0.65,-0.34), R(0.48,0.78))
        p = cubic(p0, (R(0.18,0.55), R(-1.05,-0.78)), (R(0.72,0.86), R(-0.52,-0.18)), mid, 80)
        q = cubic(mid, (R(0.72,0.88), R(0.10,0.42)), (R(0.42,0.70), R(0.85,1.03)), p3, 85)
        strokes.append(np.vstack([p, q[1:]]))
    elif d == 4:
        if rng.random() < 0.68:
            xv = R(0.22,0.52); yc = R(-0.02,0.22)
            strokes.append(line((xv, R(-0.92,-0.65)), (xv+R(-.08,.08), R(0.78,1.05)), 80))
            strokes.append(line((R(-0.70,-0.42), yc), (xv, R(-0.90,-0.62)), 55))
            strokes.append(line((R(-0.68,-0.42), yc), (R(0.52,0.82), yc+R(-.06,.06)), 60))
        else:
            left = R(-0.55,-0.30); xv = R(0.20,0.48); yc=R(-0.02,0.20)
            strokes.append(line((left, R(-0.88,-0.62)), (left, yc), 45))
            strokes.append(line((left, yc), (R(0.55,0.78), yc+R(-.05,.05)), 65))
            strokes.append(line((xv, R(-0.85,-0.55)), (xv, R(0.75,1.00)), 75))
    elif d == 5:
        top_y = R(-0.92,-0.68); left_x=R(-0.68,-0.42)
        p = line((R(0.38,0.72), top_y), (left_x, top_y+R(-.04,.05)), 45)
        q = line((left_x, top_y), (left_x+R(-.05,.08), R(-0.24,-0.02)), 45)
        start = q[-1]
        r = cubic(tuple(start), (R(-0.05,0.30), R(-0.35,-0.12)), (R(0.70,0.86), R(-0.03,0.30)), (R(0.42,0.65), R(0.25,0.48)), 70)
        s = cubic(tuple(r[-1]), (R(0.30,0.55), R(0.80,1.00)), (R(-0.35,-0.08), R(0.90,1.05)), (R(-0.66,-0.36), R(0.48,0.72)), 75)
        strokes.append(np.vstack([p, q[1:], r[1:], s[1:]]))
    elif d == 6:
        # A six is a lower loop with an open upper hook, not a full-height oval.
        cx, cy = R(-0.08,0.08), R(0.36,0.55); rx, ry = R(0.36,0.54), R(0.34,0.54)
        a0 = R(0.95*np.pi, 1.20*np.pi)
        loop = arc(cx, cy, rx, ry, a0, a0 + 2*np.pi, 135)
        hook = cubic((R(0.05,0.42), R(-1.04,-0.76)), (R(-0.38,-0.08), R(-0.86,-0.54)), (R(-0.72,-0.46), R(-0.08,0.20)), tuple(loop[0]), 90)
        strokes.append(np.vstack([hook, loop[1:]]))
    elif d == 7:
        ytop = R(-0.92,-0.66); xr=R(0.42,0.75); xl=R(-0.72,-0.42)
        strokes.append(line((xl, ytop), (xr, ytop+R(-.05,.06)), 55))
        strokes.append(line((xr, ytop), (R(-0.30,0.10), R(0.74,1.06)), 80))
        if rng.random() < 0.48:
            strokes.append(line((R(-0.35,-0.08), R(-0.02,0.20)), (R(0.28,0.56), R(-0.10,0.12)), 35))
    elif d == 8:
        if rng.random() < 0.72:
            strokes.append(arc(R(-.04,.05), R(-0.42,-0.24), R(0.30,0.50), R(0.30,0.48), 0, 2*np.pi, 105))
            strokes.append(arc(R(-.03,.05), R(0.34,0.52), R(0.35,0.56), R(0.35,0.56), 0, 2*np.pi, 120))
        else:
            th = np.linspace(0, 2*np.pi, 170, dtype=np.float32)
            x = R(0.30,0.50) * np.sin(th)
            y = np.where(np.cos(th) >= 0,
                         R(-0.42,-0.24) + R(0.30,0.48) * np.sin(th),
                         R(0.34,0.52) + R(0.35,0.56) * np.sin(th))
            strokes.append(np.stack([x, y], 1).astype(np.float32))
    elif d == 9:
        cx, cy = R(-0.06,0.14), R(-0.44,-0.16); rx, ry = R(0.39,0.64), R(0.38,0.62)
        strokes.append(arc(cx, cy, rx, ry, 0, 2*np.pi, 125))
        x0 = cx + rx*R(0.58,0.98)
        if rng.random() < 0.55:
            stem = cubic((x0, cy+R(0.00,0.24)), (R(0.60,0.84), R(0.10,0.42)), (R(0.38,0.68), R(0.58,0.96)), (R(-0.02,0.34), R(0.76,1.08)), 85)
        else:
            stem = cubic((x0, cy+R(0.00,0.24)), (R(0.34,0.76), R(0.18,0.50)), (R(0.05,0.58), R(0.60,0.98)), (R(-0.30,0.28), R(0.78,1.10)), 85)
        strokes.append(stem)
    return strokes


def transform_paths(strokes, rng):
    sc = rng.uniform(8.65, 10.55)
    ax = rng.uniform(0.86, 1.16); ay = rng.uniform(0.92, 1.10)
    ang = rng.uniform(-0.18, 0.18)
    sh = rng.uniform(-0.22, 0.22)
    c, s = np.cos(ang), np.sin(ang)
    out = []
    for p in strokes:
        x = p[:,0].copy(); y = p[:,1].copy()
        # gentle personal-style curvature, not fitted to any image
        x = x + rng.uniform(-0.055, 0.055) * y * y + rng.uniform(-0.035, 0.035) * y
        x *= ax; y *= ay
        xr = c*x - s*y
        yr = s*x + c*y
        xr = xr + sh*yr
        out.append(np.stack([13.5 + sc*xr, 13.5 + sc*yr], 1).astype(np.float32))
    return out


def render(strokes, width):
    img = np.zeros((28,28), dtype=np.float32)
    for p in strokes:
        d2 = np.full((28,28), 1e9, dtype=np.float32)
        # Chunking keeps temporary arrays small and deterministic.
        for a in range(0, len(p), 64):
            q = p[a:a+64]
            dd = (X[...,None] - q[:,0])**2 + (Y[...,None] - q[:,1])**2
            d2 = np.minimum(d2, dd.min(axis=2))
        stroke = np.exp(-d2 / (2 * width * width)).astype(np.float32)
        img = np.maximum(img, stroke)
    return img


def shift_bilinear(img, dx, dy):
    # output[y,x] = img[y-dy, x-dx]
    yy = Y - np.float32(dy); xx = X - np.float32(dx)
    x0 = np.floor(xx).astype(np.int32); y0 = np.floor(yy).astype(np.int32)
    x1 = x0 + 1; y1 = y0 + 1
    wx = xx - x0; wy = yy - y0
    out = np.zeros((28,28), dtype=np.float32)
    for xa, wa in ((x0, 1-wx), (x1, wx)):
        for ya, wb in ((y0, 1-wy), (y1, wy)):
            m = (xa >= 0) & (xa < 28) & (ya >= 0) & (ya < 28)
            out[m] += img[ya[m], xa[m]] * (wa[m] * wb[m]).astype(np.float32)
    return out


def make_digit(d, rng):
    strokes = transform_paths(digit_paths(d, rng), rng)
    width = rng.uniform(0.68, 1.22)
    img = render(strokes, width)
    mass = float(img.sum())
    if mass > 1e-6:
        cy = float((img * Y).sum() / mass); cx = float((img * X).sum() / mass)
        img = shift_bilinear(img, 13.5 - cx + rng.normal(0, 0.30), 13.5 - cy + rng.normal(0, 0.30))
    # A very light soft clip makes rendered strokes closer to grey antialiased ink.
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img


def gaussian_kernel():
    ax = np.arange(-2, 3, dtype=np.float32)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    k = np.exp(-(xx*xx + yy*yy) / (2 * 0.85 * 0.85)).astype(np.float32)
    k /= k.sum()
    return k

K = gaussian_kernel()


def blur(img):
    pad = np.pad(img, ((2,2),(2,2)), mode='constant')
    out = np.zeros_like(img, dtype=np.float32)
    for dy in range(5):
        for dx in range(5):
            out += K[dy,dx] * pad[dy:dy+28, dx:dx+28]
    return out


def avg_pool(img, k):
    n = 28 // k
    return img.reshape(n, k, n, k).mean(axis=(1,3)).astype(np.float32)


def row_from(feat, alpha):
    v = feat.reshape(-1).astype(np.float32)
    v = v - v.mean()
    sd = np.sqrt(np.mean(v*v) + EPS).astype(np.float32)
    return (alpha * v / (sd * v.size)).astype(np.float32)


def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = 64, 176
    layers = [
        {"name":"blur", "op":"conv", "inputs":["image"], "out":1, "kernel":[5,5], "stride":1, "padding":2},
        {"name":"n28", "op":"norm", "inputs":["blur"]},
        {"name":"d28", "op":"dense", "inputs":["n28"], "out":10*T},
        {"name":"r28", "op":"reshape", "inputs":["d28"], "shape":[10,h,w]},
        {"name":"m28", "op":"maxpool", "inputs":["r28"], "size":[h,w], "stride":[h,w]},
        {"name":"s28", "op":"reshape", "inputs":["m28"], "shape":[10]},
        {"name":"p14", "op":"avgpool", "inputs":["blur"], "size":[2,2], "stride":[2,2]},
        {"name":"n14", "op":"norm", "inputs":["p14"]},
        {"name":"d14", "op":"dense", "inputs":["n14"], "out":10*T},
        {"name":"r14", "op":"reshape", "inputs":["d14"], "shape":[10,h,w]},
        {"name":"m14", "op":"maxpool", "inputs":["r14"], "size":[h,w], "stride":[h,w]},
        {"name":"s14", "op":"reshape", "inputs":["m14"], "shape":[10]},
        {"name":"p7", "op":"avgpool", "inputs":["blur"], "size":[4,4], "stride":[4,4]},
        {"name":"n7", "op":"norm", "inputs":["p7"]},
        {"name":"d7", "op":"dense", "inputs":["n7"], "out":10*T},
        {"name":"r7", "op":"reshape", "inputs":["d7"], "shape":[10,h,w]},
        {"name":"m7", "op":"maxpool", "inputs":["r7"], "size":[h,w], "stride":[h,w]},
        {"name":"s7", "op":"reshape", "inputs":["m7"], "shape":[10]},
        {"name":"score", "op":"add", "inputs":["s28","s14","s7"]},
    ]
    W28 = np.empty((10*T, 28*28), dtype=np.float32)
    W14 = np.empty((10*T, 14*14), dtype=np.float32)
    W7 = np.empty((10*T, 7*7), dtype=np.float32)
    rng = np.random.default_rng(928374)
    for d in range(10):
        for j in range(T):
            img = make_digit(d, rng)
            b = blur(img)
            row = d*T + j
            W28[row] = row_from(b, 1.00)
            W14[row] = row_from(avg_pool(b, 2), 0.82)
            W7[row] = row_from(avg_pool(b, 4), 0.48)
    weights = {
        "blur.W": K.reshape(1,1,5,5).astype(np.float32),
        "blur.b": np.zeros(1, dtype=np.float32),
        "d28.W": W28, "d28.b": np.zeros(10*T, dtype=np.float32),
        "d14.W": W14, "d14.b": np.zeros(10*T, dtype=np.float32),
        "d7.W": W7, "d7.b": np.zeros(10*T, dtype=np.float32),
    }
    (out_dir / "network.json").write_text(json.dumps({"layers": layers}, separators=(",", ":")))
    np.savez(out_dir / "weights.npz", **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
