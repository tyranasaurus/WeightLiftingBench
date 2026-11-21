"""Writes network.json and weights.npz for a hand-designed digit classifier. Reads no files.

Design: blur the image, standardise it (norm), correlate it with stroke-drawn digit templates at 5 x 5 small
shifts, take the best shift, then the best template of each class.
Every template is one image drawn here from pen paths typed from general knowledge of how digits are written,
and is normalised by itself alone (own mean removed, own length divided out). No data, no fitting.
"""
import sys, os, json
import numpy as np


def ell(cx, cy, rx, ry, n=36):
    t = np.radians(np.linspace(0.0, 360.0, n + 1))
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]


# Pen paths in a unit box: x to the right, y downwards. Each shape is a list of strokes (polylines).
SHAPES = {
    0: [[ell(0.5, 0.5, 0.5, 0.5)],
        [ell(0.5, 0.5, 0.38, 0.5)],
        [ell(0.5, 0.5, 0.3, 0.5)]],
    1: [[[(0.5, 0.0), (0.5, 1.0)]],
        [[(0.2, 0.25), (0.55, 0.0), (0.55, 1.0)]],
        [[(0.2, 0.25), (0.55, 0.0), (0.55, 1.0)], [(0.2, 1.0), (0.9, 1.0)]]],
    2: [[[(0.1, 0.25), (0.22, 0.08), (0.5, 0.0), (0.78, 0.08), (0.9, 0.27), (0.8, 0.48), (0.5, 0.72), (0.08, 1.0), (0.95, 1.0)]],
        [[(0.1, 0.25), (0.22, 0.08), (0.5, 0.0), (0.78, 0.08), (0.9, 0.27), (0.8, 0.48), (0.45, 0.78), (0.15, 0.95),
          (0.05, 0.85), (0.2, 0.78), (0.5, 0.88), (0.95, 1.0)]],
        [[(0.15, 0.2), (0.35, 0.03), (0.65, 0.03), (0.85, 0.22), (0.75, 0.5), (0.4, 0.8), (0.08, 1.0), (0.5, 0.92), (0.95, 0.95)]]],
    3: [[[(0.12, 0.1), (0.4, 0.0), (0.7, 0.03), (0.86, 0.2), (0.72, 0.4), (0.4, 0.48), (0.72, 0.56), (0.9, 0.75),
          (0.72, 0.95), (0.4, 1.0), (0.08, 0.9)]],
        [[(0.12, 0.0), (0.88, 0.0), (0.45, 0.42), (0.72, 0.5), (0.9, 0.72), (0.72, 0.95), (0.4, 1.0), (0.08, 0.9)]],
        [[(0.15, 0.08), (0.45, 0.0), (0.75, 0.05), (0.88, 0.22), (0.7, 0.42), (0.25, 0.48), (0.7, 0.54), (0.9, 0.75),
          (0.7, 0.96), (0.35, 1.0), (0.05, 0.88)]]],
    4: [[[(0.15, 0.0), (0.08, 0.55), (0.95, 0.55)], [(0.72, 0.0), (0.72, 1.0)]],
        [[(0.7, 0.0), (0.05, 0.65), (0.98, 0.65)], [(0.7, 0.0), (0.7, 1.0)]],
        [[(0.2, 0.0), (0.1, 0.5), (0.85, 0.5)], [(0.8, 0.1), (0.75, 1.0)]]],
    5: [[[(0.9, 0.0), (0.25, 0.0), (0.18, 0.42), (0.5, 0.36), (0.8, 0.46), (0.92, 0.68), (0.78, 0.92), (0.45, 1.0), (0.08, 0.9)]],
        [[(0.95, 0.0), (0.35, 0.03), (0.2, 0.4), (0.55, 0.4), (0.85, 0.55), (0.85, 0.82), (0.55, 1.0), (0.1, 0.95)]],
        [[(0.3, 0.05), (0.2, 0.45), (0.55, 0.4), (0.85, 0.55), (0.88, 0.8), (0.6, 1.0), (0.1, 0.92)], [(0.3, 0.05), (0.95, 0.0)]]],
    6: [[[(0.8, 0.0), (0.5, 0.1), (0.25, 0.35), (0.12, 0.65), (0.2, 0.9), (0.5, 1.0), (0.8, 0.9), (0.9, 0.7), (0.75, 0.5),
          (0.45, 0.46), (0.15, 0.65)]],
        [[(0.55, 0.0), (0.3, 0.3), (0.15, 0.65), (0.22, 0.9), (0.5, 1.0), (0.8, 0.9), (0.88, 0.7), (0.7, 0.52), (0.4, 0.5),
          (0.17, 0.7)]],
        [[(0.85, 0.05), (0.45, 0.08), (0.15, 0.4), (0.08, 0.7), (0.25, 0.95), (0.6, 1.0), (0.9, 0.8), (0.85, 0.55),
          (0.55, 0.42), (0.2, 0.55)]]],
    7: [[[(0.08, 0.0), (0.92, 0.0), (0.4, 1.0)]],
        [[(0.08, 0.0), (0.92, 0.0), (0.4, 1.0)], [(0.3, 0.5), (0.85, 0.5)]],
        [[(0.1, 0.18), (0.1, 0.02), (0.9, 0.0), (0.6, 0.5), (0.45, 1.0)]]],
    8: [[ell(0.5, 0.24, 0.38, 0.24), ell(0.5, 0.74, 0.46, 0.26)],
        [[(0.85, 0.12), (0.6, 0.0), (0.3, 0.03), (0.15, 0.2), (0.3, 0.4), (0.7, 0.58), (0.88, 0.78), (0.7, 0.97), (0.35, 1.0),
          (0.12, 0.8), (0.3, 0.58), (0.7, 0.4), (0.88, 0.2), (0.85, 0.12)]],
        [ell(0.5, 0.25, 0.42, 0.25), ell(0.5, 0.75, 0.42, 0.25)]],
    9: [[ell(0.45, 0.27, 0.4, 0.27), [(0.85, 0.15), (0.85, 0.5), (0.7, 1.0)]],
        [ell(0.45, 0.27, 0.4, 0.27), [(0.85, 0.05), (0.85, 1.0)]],
        [ell(0.48, 0.27, 0.4, 0.27), [(0.88, 0.2), (0.85, 0.6), (0.65, 0.92), (0.35, 1.0), (0.1, 0.9)]]],
}
SLANTS = [-0.1, 0.15, 0.4]      # sideways lean of the top relative to the bottom, as a share of the height
ASPECTS = [0.6, 0.85]           # width of the unit box relative to its height
THICKS = [2.0, 3.5]             # pen width in pixels
SIGMA = 1.0                     # blur applied to both image and templates
YY, XX = np.mgrid[0:28, 0:28].astype(np.float64)


def gauss_kernel():
    r = np.arange(-3, 4, dtype=np.float64)
    g = np.exp(-r * r / (2 * SIGMA * SIGMA))
    g /= g.sum()
    return np.outer(g, g)


def blur(img, k):
    p = np.pad(img, 3)
    out = np.zeros_like(img)
    for dy in range(7):
        for dx in range(7):
            out += k[dy, dx] * p[dy:dy + 28, dx:dx + 28]
    return out


def render(strokes, aspect, slant, thick):
    """Draw the strokes so the ink fits a 20 x 20 box and its centre of mass sits at the centre of the frame."""
    segs = []
    for st in strokes:
        p = np.array(st, dtype=np.float64)
        x = (p[:, 0] - 0.5) * aspect + slant * (0.5 - p[:, 1])
        y = p[:, 1] - 0.5
        segs.append(np.stack([x, y], 1))
    allp = np.concatenate(segs)
    span = max(allp[:, 0].max() - allp[:, 0].min(), allp[:, 1].max() - allp[:, 1].min())
    sc = (20.0 - thick) / span
    segs = [q * sc for q in segs]
    mid = (allp.min(0) + allp.max(0)) / 2 * sc

    def draw(off):
        d = np.full((28, 28), 1e9)
        for q in segs:
            q = q + off
            for a, b in zip(q[:-1], q[1:]):
                ab = b - a
                l2 = float((ab ** 2).sum())
                if l2 < 1e-12:
                    t = np.zeros_like(XX)
                else:
                    t = np.clip(((XX - a[0]) * ab[0] + (YY - a[1]) * ab[1]) / l2, 0.0, 1.0)
                d = np.minimum(d, np.hypot(XX - a[0] - t * ab[0], YY - a[1] - t * ab[1]))
        return np.clip(thick / 2 + 0.5 - d, 0.0, 1.0)

    off = np.array([13.5, 13.5]) - mid
    img = draw(off)
    m = img.sum()
    off = off + np.array([13.5 - (img * XX).sum() / m, 13.5 - (img * YY).sum() / m])
    return draw(off)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    k = gauss_kernel()
    temps = []
    for c in range(10):
        for shape in SHAPES[c]:
            for sl in SLANTS:
                for asp in ASPECTS:
                    for th in THICKS:
                        t = blur(render(shape, asp, sl, th), k)
                        t = t - t.mean()                       # each template normalised by itself alone
                        t = t / np.sqrt((t ** 2).sum())
                        temps.append(t / 28.0)                 # so a perfect match scores 1
    T = len(temps)
    V = T // 10
    weights = {
        "blur.W": k.reshape(1, 1, 7, 7).astype(np.float32),
        "blur.b": np.zeros(1, np.float32),
        "tm.W": np.stack(temps).reshape(T, 1, 28, 28).astype(np.float32),
        "tm.b": np.zeros(T, np.float32),
    }
    net = {"layers": [
        {"name": "blur", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [7, 7], "stride": 1, "padding": 3},
        {"name": "nrm", "op": "norm", "inputs": ["blur"]},
        {"name": "tm", "op": "conv", "inputs": ["nrm"], "out": T, "kernel": [28, 28], "stride": 1, "padding": 2},
        {"name": "shift", "op": "maxpool", "inputs": ["tm"], "size": [5, 5], "stride": [5, 5]},
        {"name": "byclass", "op": "reshape", "inputs": ["shift"], "shape": [1, 10, V]},
        {"name": "best", "op": "maxpool", "inputs": ["byclass"], "size": [1, V], "stride": [1, V]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]},
    ]}
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
