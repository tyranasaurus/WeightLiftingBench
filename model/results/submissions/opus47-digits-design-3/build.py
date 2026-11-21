"""build.py: generates /app/out/network.json and /app/out/weights.npz.

Architecture: a single 28x28 convolution of hand-designed digit templates,
followed by a per-class max-pool. Templates are generated from parametric
stroke paths that describe each digit (shape only, no reference to any image
or label). We sweep styles, slants and stroke widths to cover natural
handwriting variation. All weights come from these formulas alone.
"""
from __future__ import annotations
import json
import sys
import numpy as np
from pathlib import Path

CANVAS = 28
BOX = 40.0
CENTER = 13.5
NUM_TEMPLATES_PER_CLASS = 48


# -------- rendering helpers --------------------------------------------------

def polyline_dist(points, X, Y):
    d2 = None
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            seg = (X - x0) ** 2 + (Y - y0) ** 2
        else:
            t = ((X - x0) * dx + (Y - y0) * dy) / L2
            t = np.clip(t, 0.0, 1.0)
            px = x0 + t * dx
            py = y0 + t * dy
            seg = (X - px) ** 2 + (Y - py) ** 2
        d2 = seg if d2 is None else np.minimum(d2, seg)
    return np.sqrt(d2)


def render_strokes(strokes, pen_sigma, slant=0.0, aspect=1.0, scale=1.0):
    yy, xx = np.indices((CANVAS, CANVAS)).astype(np.float64)
    dy = (yy - CENTER) / (BOX * scale)
    dx = (xx - CENTER) / (BOX * scale * aspect) - slant * dy
    canvas = np.zeros((CANVAS, CANVAS))
    for path in strokes:
        if len(path) < 2:
            continue
        d = polyline_dist(path, dx, dy)
        canvas = np.maximum(canvas, np.exp(-0.5 * (d / pen_sigma) ** 2))
    return canvas


def render_blobs(blobs, slant=0.0):
    yy, xx = np.indices((CANVAS, CANVAS)).astype(np.float64)
    dy = (yy - CENTER) / BOX
    dx = (xx - CENTER) / BOX - slant * dy
    canvas = np.zeros((CANVAS, CANVAS))
    for (bx, by, bsig) in blobs:
        d2 = (dx - bx) ** 2 + (dy - by) ** 2
        canvas = np.maximum(canvas, np.exp(-0.5 * d2 / (bsig * bsig)))
    return canvas


def arc_pts(cx, cy, rx, ry, a0_deg, a1_deg, n=36):
    out = []
    for i in range(n + 1):
        t = i / n
        a = np.radians(a0_deg + (a1_deg - a0_deg) * t)
        out.append((cx + rx * np.cos(a), cy + ry * np.sin(a)))
    return out


# -------- digit stroke designs ----------------------------------------------

def paths_0(style):
    s = style % 4
    if s == 0:
        return [arc_pts(0.0, 0.0, 0.20, 0.25, 0, 360, n=48)]
    if s == 1:
        return [arc_pts(0.0, 0.0, 0.17, 0.24, 0, 360, n=48)]
    if s == 2:
        return [arc_pts(0.0, 0.0, 0.22, 0.25, 0, 360, n=48)]
    if s == 3:
        return [arc_pts(0.0, 0.0, 0.19, 0.22, 0, 360, n=48)]


def paths_1(style):
    s = style % 4
    if s == 0:
        return [[(0.0, -0.25), (0.0, 0.25)]]
    if s == 1:
        return [[(-0.10, -0.14), (0.0, -0.25), (0.0, 0.25)]]
    if s == 2:
        serif = [(-0.10, -0.14), (0.0, -0.25), (0.0, 0.25)]
        base = [(-0.12, 0.25), (0.12, 0.25)]
        return [serif, base]
    if s == 3:
        return [[(0.0, -0.22), (0.0, 0.25)]]  # slightly shorter


def paths_2(style):
    s = style % 4
    if s == 0:
        top = arc_pts(0.0, -0.10, 0.17, 0.14, 200, 360 + 60, n=24)
        diag = [(0.08, 0.0), (-0.18, 0.22)]
        base = [(-0.19, 0.22), (0.18, 0.22)]
        return [top + diag, base]
    if s == 1:
        top = arc_pts(0.0, -0.12, 0.18, 0.13, 180, 360, n=24)
        diag = [(0.18, -0.12), (-0.18, 0.22)]
        base = [(-0.19, 0.22), (0.19, 0.22)]
        return [top, diag, base]
    if s == 2:
        top = arc_pts(0.0, -0.10, 0.16, 0.12, 210, 360 + 60, n=24)
        diag = [(0.07, 0.0), (-0.06, 0.10), (-0.18, 0.22)]
        base = [(-0.19, 0.22), (0.17, 0.22)]
        return [top + diag, base]
    if s == 3:
        # z-like angular 2
        return [[(-0.17, -0.18), (0.0, -0.22), (0.18, -0.18), (0.15, -0.05), (-0.18, 0.22), (0.18, 0.22)]]


def paths_3(style):
    s = style % 4
    if s == 0:
        top = arc_pts(-0.02, -0.11, 0.18, 0.13, 150, 390, n=24)
        bot = arc_pts(-0.02, 0.11, 0.19, 0.14, 330, 210, n=24)
        bridge = [(0.00, -0.02), (0.00, 0.02)]
        return [top, bridge, bot]
    if s == 1:
        return [[(-0.17, -0.22), (0.17, -0.22), (0.00, -0.02), (0.17, 0.22), (-0.17, 0.22)]]
    if s == 2:
        top = arc_pts(0.00, -0.12, 0.17, 0.13, 170, 20, n=24)
        bot = arc_pts(0.00, 0.12, 0.20, 0.14, 340, 180, n=24)
        bridge = [top[-1], (0.00, 0.0), bot[0]]
        return [top, bridge, bot]
    if s == 3:
        # 3 with open-to-left both bumps
        top = arc_pts(-0.04, -0.11, 0.17, 0.13, 180, 420, n=24)
        bot = arc_pts(-0.04, 0.11, 0.20, 0.13, 300, 180, n=24)
        bridge = [(0.0, 0.0), (0.0, 0.03)]
        return [top, bridge, bot]


def paths_4(style):
    s = style % 4
    if s == 0:
        left = [(0.00, -0.22), (-0.17, 0.07)]
        bar = [(-0.17, 0.07), (0.11, 0.07)]
        right = [(0.11, -0.18), (0.11, 0.24)]
        return [left, bar, right]
    if s == 1:
        left = [(0.11, -0.22), (-0.17, 0.07)]
        bar = [(-0.17, 0.07), (0.17, 0.07)]
        right = [(0.11, -0.22), (0.11, 0.24)]
        return [left, bar, right]
    if s == 2:
        left = [(0.03, -0.22), (-0.06, -0.08), (-0.18, 0.07)]
        bar = [(-0.17, 0.07), (0.14, 0.07)]
        right = [(0.13, -0.18), (0.13, 0.24)]
        return [left, bar, right]
    if s == 3:
        # tall right vertical with triangular top
        left = [(0.10, -0.24), (-0.18, 0.08)]
        bar = [(-0.18, 0.08), (0.18, 0.08)]
        right = [(0.11, -0.24), (0.14, 0.24)]
        return [left, bar, right]


def paths_5(style):
    s = style % 4
    if s == 0:
        top = [(-0.16, -0.22), (0.17, -0.22)]
        left = [(-0.16, -0.22), (-0.16, 0.02)]
        curve = [(-0.16, 0.02), (0.02, -0.02)] + arc_pts(0.0, 0.10, 0.18, 0.14, -75, 180, n=24)
        return [top, left, curve]
    if s == 1:
        top = [(-0.18, -0.22), (0.17, -0.22)]
        left = [(-0.18, -0.22), (-0.18, 0.02)]
        curve = [(-0.18, 0.02)] + arc_pts(0.0, 0.10, 0.18, 0.14, -75, 180, n=24)
        return [top, left, curve]
    if s == 2:
        top = [(-0.17, -0.22), (0.18, -0.19)]
        left = [(-0.17, -0.22), (-0.17, 0.02)]
        curve = [(-0.17, 0.02)] + arc_pts(0.0, 0.10, 0.18, 0.14, -75, 180, n=24)
        return [top, left, curve]
    if s == 3:
        # compact 5
        top = [(-0.14, -0.22), (0.15, -0.22)]
        left = [(-0.14, -0.22), (-0.14, 0.00)]
        curve = [(-0.14, 0.00)] + arc_pts(0.02, 0.10, 0.17, 0.14, -80, 180, n=24)
        return [top, left, curve]


def paths_6(style):
    s = style % 4
    if s == 0:
        descent = arc_pts(0.0, 0.0, 0.20, 0.24, -55, 195, n=36)
        loop = arc_pts(0.0, 0.10, 0.18, 0.13, 0, 360, n=32)
        return [descent, loop]
    if s == 1:
        descent = arc_pts(0.02, -0.02, 0.19, 0.24, -65, 190, n=36)
        loop = arc_pts(0.0, 0.11, 0.17, 0.13, 0, 360, n=32)
        return [descent, loop]
    if s == 2:
        descent = arc_pts(0.0, 0.0, 0.21, 0.22, -45, 200, n=36)
        loop = arc_pts(0.0, 0.12, 0.19, 0.12, 0, 360, n=32)
        return [descent, loop]
    if s == 3:
        # tight loop with longer descent
        descent = arc_pts(0.0, -0.05, 0.20, 0.22, -55, 190, n=36)
        loop = arc_pts(-0.01, 0.11, 0.17, 0.12, 0, 360, n=32)
        return [descent, loop]


def paths_7(style):
    s = style % 4
    if s == 0:
        top = [(-0.19, -0.22), (0.19, -0.22)]
        diag = [(0.19, -0.22), (-0.06, 0.25)]
        return [top, diag]
    if s == 1:
        top = [(-0.19, -0.22), (0.19, -0.22)]
        diag = [(0.19, -0.22), (-0.08, 0.25)]
        bar = [(-0.11, 0.03), (0.09, 0.03)]
        return [top, diag, bar]
    if s == 2:
        top = [(-0.19, -0.18), (-0.19, -0.22), (0.19, -0.22)]
        diag = [(0.19, -0.22), (0.00, 0.25)]
        return [top, diag]
    if s == 3:
        top = [(-0.17, -0.22), (0.19, -0.22)]
        diag = [(0.19, -0.22), (-0.03, 0.25)]
        return [top, diag]


def paths_8(style):
    s = style % 4
    if s == 0:
        top = arc_pts(0.0, -0.13, 0.14, 0.12, 0, 360, n=36)
        bot = arc_pts(0.0, 0.13, 0.18, 0.12, 0, 360, n=36)
        return [top, bot]
    if s == 1:
        top = arc_pts(0.0, -0.12, 0.12, 0.12, 0, 360, n=36)
        bot = arc_pts(0.0, 0.13, 0.19, 0.12, 0, 360, n=36)
        return [top, bot]
    if s == 2:
        top = arc_pts(0.0, -0.13, 0.16, 0.13, 0, 360, n=36)
        bot = arc_pts(0.0, 0.13, 0.17, 0.13, 0, 360, n=36)
        return [top, bot]
    if s == 3:
        top = arc_pts(0.0, -0.14, 0.13, 0.11, 0, 360, n=36)
        bot = arc_pts(0.0, 0.12, 0.17, 0.13, 0, 360, n=36)
        return [top, bot]


def paths_9(style):
    s = style % 4
    if s == 0:
        loop = arc_pts(0.0, -0.10, 0.16, 0.13, 0, 360, n=36)
        down = [(0.16, -0.10), (0.14, 0.25)]
        return [loop, down]
    if s == 1:
        loop = arc_pts(0.0, -0.09, 0.17, 0.13, 0, 360, n=36)
        down = [(0.17, -0.09), (0.17, 0.10), (0.08, 0.25)]
        return [loop, down]
    if s == 2:
        loop = arc_pts(0.0, -0.10, 0.16, 0.14, 0, 360, n=36)
        down = [(0.16, -0.10), (0.03, 0.25)]
        return [loop, down]
    if s == 3:
        loop = arc_pts(0.0, -0.11, 0.15, 0.12, 0, 360, n=36)
        down = [(0.15, -0.11), (0.14, 0.15), (0.08, 0.25)]
        return [loop, down]


PATHS = [paths_0, paths_1, paths_2, paths_3, paths_4, paths_5, paths_6, paths_7, paths_8, paths_9]


# Anti-ink Gaussian blobs: positions where this digit has NO ink. We bake
# these into each positive template with a small weight so they still look
# like the digit but gain a mild negative signal against look-alikes.
ANTI = {
    0: [(0.0, 0.0, 0.11)],
    1: [(-0.18, 0.0, 0.10), (0.18, 0.0, 0.10)],
    2: [(0.15, 0.03, 0.06), (-0.17, -0.18, 0.05), (-0.18, 0.05, 0.05)],
    3: [(-0.17, 0.00, 0.07), (-0.18, -0.20, 0.05), (-0.22, 0.10, 0.05),
        (-0.22, -0.10, 0.05)],
    4: [(0.00, -0.14, 0.07), (-0.18, 0.22, 0.05), (0.19, 0.22, 0.05),
        (-0.18, -0.20, 0.06)],
    5: [(0.19, -0.10, 0.07), (0.18, 0.00, 0.06)],
    6: [(0.18, -0.20, 0.07), (0.17, -0.10, 0.06)],
    7: [(-0.18, 0.18, 0.08), (-0.18, 0.00, 0.06), (-0.18, -0.10, 0.06),
        (0.00, -0.05, 0.05)],
    8: [(0.00, -0.24, 0.05), (0.00, 0.24, 0.05),
        (0.22, -0.20, 0.05), (-0.22, -0.20, 0.05),
        (0.22, 0.20, 0.05), (-0.22, 0.20, 0.05)],
    9: [(-0.17, 0.18, 0.08), (-0.17, 0.08, 0.06), (-0.19, -0.05, 0.05)],
}

ANTI_WEIGHT = 0.0    # Checks showed anti hurt accuracy; keep positive-only.


# -------- template generation ------------------------------------------------

def _digit_settings():
    out = []
    styles = [0, 1, 2, 3]            # 4 styles
    slants = [-0.20, -0.07, 0.05, 0.18]  # 4 slants
    pens = [0.030, 0.055, 0.085]     # 3 pens
    for st in styles:
        for sl in slants:
            for pw in pens:
                out.append((st, sl, pw))
    assert len(out) == NUM_TEMPLATES_PER_CLASS
    return out


def generate_templates():
    settings = _digit_settings()
    T = np.zeros((10, NUM_TEMPLATES_PER_CLASS, CANVAS, CANVAS), dtype=np.float64)
    for d in range(10):
        anti_pts = ANTI[d]
        for i, (st, sl, pw) in enumerate(settings):
            strokes = PATHS[d](st)
            pos = render_strokes(strokes, pw, slant=sl)
            if anti_pts and ANTI_WEIGHT > 0:
                anti = render_blobs(anti_pts, slant=sl)
                T[d, i] = pos - ANTI_WEIGHT * anti
            else:
                T[d, i] = pos
    # Zero-mean and unit L2 per template
    flat = T.reshape(-1, CANVAS * CANVAS)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.sqrt((flat * flat).sum(axis=1, keepdims=True)) + 1e-8
    flat = flat / norm
    return flat.reshape(T.shape)


# -------- main ---------------------------------------------------------------

def build(out_dir: Path):
    T = generate_templates()
    K = NUM_TEMPLATES_PER_CLASS
    total = 10 * K
    W = T.reshape(total, 1, CANVAS, CANVAS).astype(np.float32)
    b = np.zeros((total,), dtype=np.float32)

    net = {
        "layers": [
            {"name": "match", "op": "conv", "inputs": ["image"],
             "out": total, "kernel": [CANVAS, CANVAS], "stride": 1, "padding": 0},
            {"name": "r1", "op": "reshape", "inputs": ["match"], "shape": [1, 10, K]},
            {"name": "pick", "op": "maxpool", "inputs": ["r1"],
             "size": [1, K], "stride": [1, K]},
            {"name": "out", "op": "reshape", "inputs": ["pick"], "shape": [10]},
        ]
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "network.json").write_text(json.dumps(net, indent=2))
    np.savez(out_dir / "weights.npz", **{"match.W": W, "match.b": b})


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    build(out)
