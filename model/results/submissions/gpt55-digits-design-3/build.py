import json
import sys
from pathlib import Path

import numpy as np

# Data-free handwritten digit recognizer.  The weights are not learned: build.py
# renders a bank of synthetic pen strokes from hand-written geometric digit
# descriptions, then stores them as fixed template-match filters.

H = WID = 28
PN = 1000   # normalized full-resolution templates per class
PR = 500    # raw L2-style full-resolution templates per class
PC = 1000   # normalized 14x14 coarse templates per class
SEED = 20261119

Y, X = np.mgrid[0:H, 0:WID].astype(np.float32)


def line(a, b, n=24):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return a * (1.0 - t) + b * t


def cubic(p0, p1, p2, p3, n=40):
    p0 = np.asarray(p0, dtype=np.float32); p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32); p3 = np.asarray(p3, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    u = 1.0 - t
    return (u**3) * p0 + 3 * (u**2) * t * p1 + 3 * u * (t**2) * p2 + (t**3) * p3


def ellipse(cx, cy, rx, ry, start=0.0, end=2*np.pi, n=96):
    th = np.linspace(start, end, n, dtype=np.float32)
    return np.stack([cx + rx*np.cos(th), cy + ry*np.sin(th)], axis=1).astype(np.float32)


def arc(cx, cy, rx, ry, deg0, deg1, n=40):
    return ellipse(cx, cy, rx, ry, np.deg2rad(deg0), np.deg2rad(deg1), n)


def cat(*parts):
    out = []
    for p in parts:
        if len(out) and len(p):
            out.append(p[1:])
        else:
            out.append(p)
    return np.concatenate(out, axis=0).astype(np.float32)


def transform_paths(paths, rng):
    # Global handwriting variations: aspect ratio, rotation, slant and centering.
    sx = rng.uniform(0.84, 1.16)
    sy = rng.uniform(0.88, 1.12)
    rot = rng.uniform(-0.17, 0.17)
    shear = rng.uniform(-0.30, 0.30)
    tx = rng.normal(0.0, 0.90)
    ty = rng.normal(0.0, 0.75)
    c, s = np.cos(rot), np.sin(rot)
    cen = np.array([13.5, 13.5], dtype=np.float32)
    out = []
    for p in paths:
        q = p.astype(np.float32) - cen
        q[:, 0] *= sx
        q[:, 1] *= sy
        q[:, 0] += shear * q[:, 1]
        x = c*q[:, 0] - s*q[:, 1]
        y = s*q[:, 0] + c*q[:, 1]
        out.append(np.stack([x + cen[0] + tx, y + cen[1] + ty], axis=1).astype(np.float32))
    return out


def render(paths, sigma):
    # Gaussian pen: every segment contributes by distance to the nearest point on
    # the segment.  Max-compositing keeps crossings from becoming too bright.
    img = np.zeros((H, WID), dtype=np.float32)
    sig2 = np.float32(2.0 * sigma * sigma)
    for p in paths:
        if len(p) < 2:
            continue
        for a, b in zip(p[:-1], p[1:]):
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            vx, vy = bx - ax, by - ay
            den = vx*vx + vy*vy + 1e-6
            t = ((X - ax)*vx + (Y - ay)*vy) / den
            t = np.clip(t, 0.0, 1.0)
            dx = X - (ax + t*vx)
            dy = Y - (ay + t*vy)
            img = np.maximum(img, np.exp(-(dx*dx + dy*dy) / sig2).astype(np.float32))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def digit_paths(d, rng):
    # Generic pen-path models of the ten classes.  Constants are hand chosen
    # from ordinary digit shapes rather than measured from the supplied images.
    if d == 0:
        cx, cy = rng.uniform(13.3, 14.7), rng.uniform(13.8, 15.1)
        rx, ry = rng.uniform(4.8, 7.4), rng.uniform(7.0, 9.6)
        if rng.random() < 0.18:
            # Pointed fast zero.
            p = cat(cubic((cx, cy-ry), (cx+rx*1.2, cy-ry*0.9), (cx+rx*1.1, cy+ry*0.8), (cx, cy+ry), 42),
                    cubic((cx, cy+ry), (cx-rx*1.1, cy+ry*0.9), (cx-rx*1.2, cy-ry*0.8), (cx, cy-ry), 42))
        else:
            p = ellipse(cx, cy, rx, ry, n=120)
        return [p]

    if d == 1:
        x_top = rng.uniform(13.0, 17.2)
        x_bot = x_top + rng.uniform(-4.0, 1.0)
        y_top = rng.uniform(4.5, 7.8); y_bot = rng.uniform(21.2, 24.4)
        stem = cubic((x_top, y_top), (x_top-0.6, 10.0), (x_bot+0.3, 17.0), (x_bot, y_bot), 54)
        paths = [stem]
        if rng.random() < 0.55:
            paths.append(line((x_top-rng.uniform(2.2, 5.8), y_top+rng.uniform(1.0, 3.8)), (x_top+0.2, y_top), 22))
        if rng.random() < 0.25:
            paths.append(line((x_bot-rng.uniform(2.5, 5.0), y_bot+0.1), (x_bot+rng.uniform(2.0, 4.4), y_bot+0.1), 20))
        return paths

    if d == 2:
        style = rng.random()
        top_y = rng.uniform(5.3, 8.6)
        left = rng.uniform(7.0, 10.8); right = rng.uniform(18.0, 22.2)
        if style < 0.65:
            top = cubic((left, top_y+rng.uniform(1.0,2.4)), (rng.uniform(10.0,13.0), top_y-rng.uniform(2.0,3.8)),
                        (right, top_y-rng.uniform(0.7,2.5)), (right+rng.uniform(-0.6,1.0), rng.uniform(9.4,12.6)), 40)
            mid = cubic(top[-1], (rng.uniform(21.0,23.0), rng.uniform(13.0,16.0)),
                        (rng.uniform(10.0,14.0), rng.uniform(16.0,19.8)),
                        (rng.uniform(7.0,10.8), rng.uniform(20.0,22.8)), 48)
            base = line(mid[-1], (rng.uniform(19.0,23.0), rng.uniform(20.5,23.6)), 34)
            return [cat(top, mid, base)]
        elif style < 0.86:
            # Z-shaped two with a straighter diagonal.
            top = arc((left+right)/2, top_y+2.6, (right-left)/2, rng.uniform(3.0,4.8), 195, 25, 42)
            diag = line(top[-1], (rng.uniform(8.0,11.0), rng.uniform(20.0,22.5)), 44)
            base = line(diag[-1], (rng.uniform(20.0,23.0), rng.uniform(21.0,23.5)), 34)
            return [cat(top, diag, base)]
        else:
            # Curly lower tail, like a handwritten 2 that almost loops.
            top = cubic((left, top_y+1.8), (left+3, top_y-3.2), (right, top_y-1.5), (right, top_y+4.0), 42)
            tail = cubic(top[-1], (right, top_y+8.0), (left+1.5, top_y+9.0), (left+2.0, top_y+13.5), 40)
            base = cubic(tail[-1], (left+7, top_y+15.0), (right-1, top_y+13.0), (right+0.5, top_y+14.8), 32)
            return [cat(top, tail, base)]

    if d == 3:
        left = rng.uniform(6.8, 10.5); right = rng.uniform(17.8, 22.2)
        top_y = rng.uniform(5.4, 8.7); mid_y = rng.uniform(12.7, 15.6); bot_y = rng.uniform(20.0, 23.4)
        style = rng.random()
        if style < 0.72:
            top = cubic((left, top_y+0.7), (rng.uniform(11.0,15.0), top_y-3.1),
                        (right, top_y-0.8), (rng.uniform(14.5,18.5), mid_y), 46)
            bot = cubic(top[-1], (right+2.0, mid_y+1.4), (right, bot_y+1.4),
                        (left+rng.uniform(-0.3,2.4), bot_y), 52)
            if rng.random() < 0.38:
                kink = line((rng.uniform(13.5,17.0), mid_y), (rng.uniform(18.0,20.8), mid_y+rng.uniform(-0.3,0.5)), 14)
                return [cat(top, kink, bot)]
            return [cat(top, bot)]
        else:
            # More angular 3, useful for forms with a clear middle shelf.
            top = line((left, top_y), (right, top_y+rng.uniform(-0.1,1.0)), 24)
            diag = line(top[-1], (rng.uniform(13.5,16.5), mid_y), 22)
            mid = line(diag[-1], (right, mid_y+rng.uniform(-0.4,0.7)), 18)
            low = cubic(mid[-1], (right+1.0, mid_y+4.0), (right-2.0, bot_y+1.5), (left, bot_y), 42)
            return [cat(top, diag, mid, low)]

    if d == 4:
        xv = rng.uniform(16.2, 20.0)
        ytop = rng.uniform(4.8, 8.2); ycross = rng.uniform(13.7, 17.5); ybot = rng.uniform(21.5, 24.5)
        style = rng.random()
        if style < 0.70:
            diag = line((xv-rng.uniform(0.0,3.2), ytop), (rng.uniform(6.8,10.8), ycross), 38)
            cross = line((rng.uniform(6.8,9.8), ycross), (rng.uniform(20.0,23.0), ycross), 36)
            vert = line((xv, ytop+rng.uniform(0.0,2.0)), (xv+rng.uniform(-1.0,0.8), ybot), 50)
            return [diag, cross, vert]
        else:
            # Open-top / h-like four.
            leftv = line((rng.uniform(9.5,12.5), ytop+2.0), (rng.uniform(8.0,10.5), ycross), 28)
            cross = line(leftv[-1], (rng.uniform(20.0,22.8), ycross+rng.uniform(-0.4,0.5)), 38)
            vert = line((xv, ytop), (xv+rng.uniform(-0.8,0.8), ybot), 52)
            return [cat(leftv, cross), vert]

    if d == 5:
        left = rng.uniform(7.2, 10.8); right = rng.uniform(18.8, 22.8)
        ytop = rng.uniform(5.0, 7.8); ym = rng.uniform(11.8, 15.3); ybot = rng.uniform(20.0, 23.5)
        top = line((right, ytop), (left, ytop+rng.uniform(-0.2,0.8)), 34)
        down = line(top[-1], (left+rng.uniform(-0.8,1.4), ym), 28)
        mid = cubic(down[-1], (rng.uniform(11.5,15.5), ym-1.2),
                    (rng.uniform(16.0,20.0), ym-0.3), (rng.uniform(17.0,20.5), ym+1.4), 30)
        low = cubic(mid[-1], (right+1.8, rng.uniform(16.0,18.8)),
                    (rng.uniform(17.0,20.0), ybot+1.4), (left+rng.uniform(-0.5,3.0), ybot), 50)
        if rng.random() < 0.20:
            # Some fives look like a single S below a short top stroke.
            low = cubic(down[-1], (right, ym), (right, ybot), (left+1.5, ybot), 60)
        return [cat(top, down, mid, low)]

    if d == 6:
        cx, cy = rng.uniform(13.0, 15.4), rng.uniform(16.6, 19.0)
        rx, ry = rng.uniform(4.6, 6.6), rng.uniform(4.6, 6.6)
        style = rng.random()
        loop = ellipse(cx, cy, rx, ry, n=100)
        if style < 0.75:
            start = (rng.uniform(16.5,21.0), rng.uniform(4.6,8.0))
            hook = cubic(start, (rng.uniform(11.0,15.0), rng.uniform(6.5,9.5)),
                         (rng.uniform(7.5,10.5), rng.uniform(12.0,16.0)),
                         (cx-rx+rng.uniform(-0.2,1.0), cy-rng.uniform(1.2,3.3)), 58)
            return [cat(hook, loop)] if rng.random() < 0.45 else [hook, loop]
        else:
            # Tall C-shaped top flowing into lower loop.
            hook = cubic((rng.uniform(17.0,20.5), rng.uniform(4.8,7.5)),
                         (rng.uniform(9.0,12.5), rng.uniform(5.8,9.0)),
                         (rng.uniform(7.0,9.5), rng.uniform(14.0,17.0)),
                         (cx-rx*0.8, cy+0.2), 62)
            return [cat(hook, loop)]

    if d == 7:
        ytop = rng.uniform(5.3, 8.8)
        xleft = rng.uniform(6.0, 9.8); xright = rng.uniform(19.0, 22.8)
        xbot = rng.uniform(10.8, 15.8); ybot = rng.uniform(21.8, 24.7)
        top = line((xleft, ytop), (xright, ytop+rng.uniform(-0.3,1.0)), 38)
        diag = line(top[-1], (xbot, ybot), 52)
        paths = [cat(top, diag)]
        if rng.random() < 0.20:
            paths.append(line((rng.uniform(10.0,12.5), rng.uniform(12.5,15.5)),
                              (rng.uniform(16.5,19.5), rng.uniform(12.0,15.2)), 18))
        return paths

    if d == 8:
        cx = rng.uniform(13.0, 15.0)
        top_cy = rng.uniform(8.4, 11.2); bot_cy = rng.uniform(16.5, 19.2)
        style = rng.random()
        if style < 0.45:
            top = ellipse(cx+rng.uniform(-0.6,0.6), top_cy, rng.uniform(4.0,6.2), rng.uniform(3.7,5.4), n=86)
            bot = ellipse(cx+rng.uniform(-0.5,0.5), bot_cy, rng.uniform(4.8,6.8), rng.uniform(4.7,6.8), n=98)
            return [top, bot]
        elif style < 0.75:
            # Continuous figure-eight with a crossing waist.
            p = cat(ellipse(cx, top_cy, rng.uniform(4.0,5.9), rng.uniform(3.7,5.3), start=-np.pi/2, end=1.5*np.pi, n=86),
                    ellipse(cx, bot_cy, rng.uniform(4.7,6.6), rng.uniform(4.7,6.7), start=-np.pi/2, end=1.5*np.pi, n=98))
            return [p]
        else:
            # S-like 8: right upper arc, left waist, right lower arc, with a
            # connecting left side; useful for quickly written eights.
            p1 = cubic((cx-4.5, top_cy+0.2), (cx-5.2, top_cy-4.0), (cx+5.8, top_cy-4.0), (cx+5.0, top_cy+1.0), 42)
            p2 = cubic(p1[-1], (cx+4.2, top_cy+5.0), (cx-4.8, top_cy+3.0), (cx-3.8, bot_cy-0.5), 40)
            p3 = cubic(p2[-1], (cx-4.8, bot_cy+5.5), (cx+6.0, bot_cy+5.5), (cx+5.0, bot_cy-0.2), 48)
            p4 = cubic(p3[-1], (cx+3.2, bot_cy-4.6), (cx-4.5, bot_cy-3.0), (cx-4.5, top_cy+0.2), 44)
            return [cat(p1, p2, p3, p4)]

    if d == 9:
        cx, cy = rng.uniform(13.0, 15.6), rng.uniform(9.8, 12.7)
        rx, ry = rng.uniform(4.6, 6.7), rng.uniform(4.4, 6.3)
        loop = ellipse(cx, cy, rx, ry, n=96)
        style = rng.random()
        if style < 0.55:
            tail_top = (cx+rx*rng.uniform(0.55,0.95), cy+rng.uniform(1.0,3.0))
            tail = cubic(tail_top, (rng.uniform(20.0,22.3), rng.uniform(14.0,17.5)),
                         (rng.uniform(17.0,20.0), rng.uniform(19.0,22.5)),
                         (rng.uniform(13.5,17.5), rng.uniform(21.8,24.6)), 56)
        else:
            tail_top = (cx+rx*rng.uniform(0.50,0.95), cy+rng.uniform(0.8,2.5))
            tail = line(tail_top, (tail_top[0]+rng.uniform(-1.8,0.8), rng.uniform(21.8,24.6)), 58)
        return [cat(loop, tail)] if rng.random() < 0.45 else [loop, tail]
    raise ValueError(d)


def make_template(d, rng):
    paths = transform_paths(digit_paths(d, rng), rng)
    sigma = rng.uniform(0.72, 1.55)
    if d == 1:
        sigma *= rng.uniform(0.80, 1.05)
    if d in (0, 6, 8, 9):
        sigma *= rng.uniform(0.95, 1.22)
    img = render(paths, sigma)
    # Pen pressure and antialiasing style.
    gamma = rng.uniform(0.70, 1.35)
    img = np.clip(img, 0.0, 1.0) ** np.float32(gamma)
    # Some handwritten images are blocky/saturated; a mild threshold-like curve
    # gives the template bank both soft and hard-edged examples.
    if rng.random() < 0.35:
        img = np.clip((img - rng.uniform(0.04, 0.12)) * rng.uniform(1.08, 1.35), 0.0, 1.0).astype(np.float32)
    return img.astype(np.float32)


def norm_flat(a):
    f = a.reshape(-1).astype(np.float32)
    f = (f - np.float32(f.mean())).astype(np.float32)
    return (f / np.float32(np.sqrt(np.mean(f * f) + 1e-5))).astype(np.float32)


def coarse14(a):
    return a.reshape(14, 2, 14, 2).mean(axis=(1, 3)).astype(np.float32)


def coarse7(a):
    return a.reshape(7, 4, 7, 4).mean(axis=(1, 3)).astype(np.float32)


def fill_branch(rng, per_class, mode):
    if mode == "coarse":
        size = 196
    elif mode == "tiny":
        size = 49
    else:
        size = 784
    n = 10 * per_class
    W = np.zeros((n, size), dtype=np.float32)
    b = np.zeros((n,), dtype=np.float32)
    idx = 0
    for d in range(10):
        for _ in range(per_class):
            t = make_template(d, rng)
            if mode == "norm":
                W[idx] = norm_flat(t)
                b[idx] = 0.0
            elif mode == "raw":
                f = t.reshape(-1).astype(np.float32)
                W[idx] = 2.0 * f
                b[idx] = -float(np.dot(f, f))
            elif mode == "coarse":
                W[idx] = norm_flat(coarse14(t))
                b[idx] = 0.0
            elif mode == "tiny":
                W[idx] = norm_flat(coarse7(t))
                b[idx] = 0.0
            idx += 1
    return W, b


def diag_dense(scale=1.0, bias=None):
    W = np.eye(10, dtype=np.float32) * np.float32(scale)
    b = np.zeros(10, dtype=np.float32) if bias is None else np.asarray(bias, dtype=np.float32)
    return W, b


def build_weights():
    rng = np.random.default_rng(SEED)
    Wn, bn = fill_branch(rng, PN, "norm")
    Wr, br = fill_branch(rng, PR, "raw")
    Wc, bc = fill_branch(rng, PC, "coarse")
    Wsn, bsn = diag_dense(1.00)
    Wsr, bsr = diag_dense(3.50)
    Wsc, bsc = diag_dense(2.05)
    # Hand calibration from general error tendencies of template matching: simple
    # strokes (1 and 7) otherwise win too often, while looped 8/9 and 2 need a
    # little prior support.
    final_bias = np.array([-5.0, -48.0, 25.0, 2.0, 15.0, 0.0, 8.0, -42.0, 70.0, 42.0], dtype=np.float32)
    Wf, bf = diag_dense(1.0, final_bias)
    return Wn, bn, Wr, br, Wc, bc, Wsn, bsn, Wsr, bsr, Wsc, bsc, Wf, bf


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = {"layers": [
        {"name": "zn", "op": "norm", "inputs": ["image"]},
        {"name": "matchn", "op": "dense", "inputs": ["zn"], "out": 10*PN},
        {"name": "classn", "op": "reshape", "inputs": ["matchn"], "shape": [10, PN, 1]},
        {"name": "bestn", "op": "maxpool", "inputs": ["classn"], "size": [PN, 1], "stride": [PN, 1]},
        {"name": "flatn", "op": "reshape", "inputs": ["bestn"], "shape": [10]},
        {"name": "scoren", "op": "dense", "inputs": ["flatn"], "out": 10},

        {"name": "matchr", "op": "dense", "inputs": ["image"], "out": 10*PR},
        {"name": "classr", "op": "reshape", "inputs": ["matchr"], "shape": [10, PR, 1]},
        {"name": "bestr", "op": "maxpool", "inputs": ["classr"], "size": [PR, 1], "stride": [PR, 1]},
        {"name": "flatr", "op": "reshape", "inputs": ["bestr"], "shape": [10]},
        {"name": "scorer", "op": "dense", "inputs": ["flatr"], "out": 10},

        {"name": "small", "op": "avgpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]},
        {"name": "smalln", "op": "norm", "inputs": ["small"]},
        {"name": "matchc", "op": "dense", "inputs": ["smalln"], "out": 10*PC},
        {"name": "classc", "op": "reshape", "inputs": ["matchc"], "shape": [10, PC, 1]},
        {"name": "bestc", "op": "maxpool", "inputs": ["classc"], "size": [PC, 1], "stride": [PC, 1]},
        {"name": "flatc", "op": "reshape", "inputs": ["bestc"], "shape": [10]},
        {"name": "scorec", "op": "dense", "inputs": ["flatc"], "out": 10},

        {"name": "sumscore", "op": "add", "inputs": ["scoren", "scorer", "scorec"]},
        {"name": "score", "op": "dense", "inputs": ["sumscore"], "out": 10},
    ]}
    Wn, bn, Wr, br, Wc, bc, Wsn, bsn, Wsr, bsr, Wsc, bsc, Wf, bf = build_weights()
    (out / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(out / "weights.npz",
             **{"matchn.W": Wn, "matchn.b": bn,
                "matchr.W": Wr, "matchr.b": br,
                "matchc.W": Wc, "matchc.b": bc,
                "scoren.W": Wsn, "scoren.b": bsn,
                "scorer.W": Wsr, "scorer.b": bsr,
                "scorec.W": Wsc, "scorec.b": bsc,
                "score.W": Wf, "score.b": bf})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
