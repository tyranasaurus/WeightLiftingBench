"""Writes network.json and weights.npz for a hand-designed clothing classifier.

Every number comes from this file: silhouettes typed from general knowledge of what a T-shirt, trouser,
pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot look like in a front / side product photo
that fills a 28 x 28 frame. No data is read, nothing is fitted.

Network:  image -> norm -> dense (30 silhouette templates, 3 per class) -> reshape (1, 10, 3)
          -> maxpool over the 3 templates of each class -> reshape (10,)
"""
import json, sys
from pathlib import Path
import numpy as np

S = 4            # supersampling factor for drawing
N = 28


def grid():
    c = (np.arange(N * S) + 0.5) / S
    return np.meshgrid(c, c)          # X, Y in pixel units, y downwards


def poly(pts):
    """Filled polygon (even-odd rule) on the supersampled grid."""
    X, Y = grid()
    inside = np.zeros(X.shape, bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        cond = (y1 > Y) != (y2 > Y)
        xint = (x2 - x1) * (Y - y1) / (y2 - y1) + x1
        inside ^= cond & (X < xint)
    return inside


def ring(cx, cy, rx, ry, t, upper_only=True):
    """Elliptical ring of thickness t (a handle loop); only the part above cy if upper_only."""
    X, Y = grid()
    outer = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0
    inner = ((X - cx) / (rx - t)) ** 2 + ((Y - cy) / (ry - t)) ** 2 <= 1.0
    m = outer & ~inner
    if upper_only:
        m &= Y <= cy
    return m


def render(mask):
    img = mask.astype(np.float64).reshape(N, S, N, S).mean(axis=(1, 3))
    k = np.array([1.0, 2.0, 1.0]) / 4.0
    for _ in range(2):                 # light binomial blur: tolerance to small shifts
        p = np.pad(img, 1)
        img = k[0] * p[:-2, 1:-1] + k[1] * p[1:-1, 1:-1] + k[2] * p[2:, 1:-1]
        p = np.pad(img, 1)
        img = k[0] * p[1:-1, :-2] + k[1] * p[1:-1, 1:-1] + k[2] * p[1:-1, 2:]
    return img


def union(*masks):
    m = masks[0].copy()
    for q in masks[1:]:
        m |= q
    return m


def shapes():
    T = {}
    # 0: T-shirt / top -- short sleeves sticking out at the shoulders, or a sleeveless tank top
    T[0] = [
        poly([(10, 1), (7, 2), (1, 7), (3, 11), (7, 8), (7, 27), (21, 27), (21, 8), (25, 11), (27, 7), (21, 2), (18, 1), (14, 3)]),
        poly([(9, 0), (11, 0), (12, 5), (16, 5), (17, 0), (19, 0), (20, 8), (21, 27), (7, 27), (8, 8)]),
        poly([(9, 1), (5, 2), (0, 8), (2, 12), (5, 10), (5, 27), (23, 27), (23, 10), (26, 12), (28, 8), (23, 2), (19, 1), (14, 3)]),
    ]
    # 1: trouser -- tall, narrow, two legs with a gap below the crotch
    T[1] = [
        poly([(9, 0), (19, 0), (20, 27), (15, 27), (14, 7), (13, 27), (8, 27)]),
        poly([(9, 0), (19, 0), (19, 27), (14.6, 27), (14, 9), (13.4, 27), (9, 27)]),
        poly([(8, 0), (20, 0), (22, 27), (17, 27), (14, 6), (11, 27), (6, 27)]),
    ]
    # 2: pullover -- body with long sleeves angled away from the body, full image width
    T[2] = [
        poly([(10, 1), (6, 2), (2, 10), (0, 26), (4, 27), (6, 12), (7, 27), (21, 27), (22, 12), (24, 27), (28, 26), (26, 10), (22, 2), (18, 1), (14, 2.5)]),
        poly([(10, 1), (6, 2), (3, 8), (1, 27), (5, 27), (7, 11), (7, 24), (21, 24), (21, 11), (23, 27), (27, 27), (25, 8), (22, 2), (18, 1), (14, 2.5)]),
        poly([(9, 1), (5, 2), (1, 9), (0, 24), (4, 25), (6, 13), (6, 26), (22, 26), (22, 13), (24, 25), (28, 24), (27, 9), (23, 2), (19, 1), (14, 3)]),
    ]
    # 3: dress -- narrow top, waist, skirt that widens (or stays narrow) down to the hem
    T[3] = [
        poly([(11, 0), (12.5, 0), (13, 3), (15, 3), (15.5, 0), (17, 0), (18, 6), (17, 11), (21, 27), (7, 27), (11, 11), (10, 6)]),
        poly([(11, 0), (17, 0), (18, 5), (17.5, 10), (18.5, 27), (9.5, 27), (10.5, 10), (10, 5)]),
        poly([(11, 0), (17, 0), (21, 4), (19.5, 7), (17.5, 6), (17, 11), (20, 27), (8, 27), (11, 11), (10.5, 6), (8.5, 7), (7, 4)]),
    ]
    # 4: coat -- long body with a collar, sleeves hanging straight down against the body
    T[4] = [
        poly([(11, 0), (6, 2), (4, 8), (3, 27), (7, 27), (7.5, 12), (8, 27), (20, 27), (20.5, 12), (21, 27), (25, 27), (24, 8), (22, 2), (17, 0), (14, 3)]),
        poly([(11, 0), (8, 2), (6, 8), (5, 24), (8, 24), (8, 27), (20, 27), (20, 24), (23, 24), (22, 8), (20, 2), (17, 0), (14, 3)]),
        poly([(10, 0), (5, 2), (3, 9), (2, 27), (26, 27), (25, 9), (23, 2), (18, 0), (14, 3)]),
    ]
    # 5: sandal -- thin sole with a few straps, mostly empty; flat or high-heeled, toe to the left
    sole_flat = poly([(0, 16), (28, 15), (28, 18), (0, 19)])
    T[5] = [
        union(sole_flat, poly([(5, 16), (9, 12), (12, 12), (13, 16)]), poly([(19, 15), (21, 9), (27, 9), (27, 15), (25, 15), (25, 11), (22, 11), (21, 15)])),
        union(poly([(0, 22), (9, 22), (9, 26), (0, 26)]), poly([(8, 23), (21, 10), (23, 12), (10, 25)]), poly([(22, 10), (27, 10), (27, 27), (24, 27)]),
              poly([(21, 3), (27, 3), (27, 10), (25, 10), (25, 5), (21, 5)]), poly([(3, 22), (6, 17), (9, 17), (10, 22)])),
        union(poly([(0, 19), (28, 17), (28, 21), (0, 22)]), poly([(3, 19), (8, 13), (12, 13), (13, 19)]),
              poly([(18, 18), (20, 8), (27, 8), (27, 18), (25, 18), (24, 10), (21, 11), (20, 18)])),
    ]
    # 6: shirt -- collar with an open V neck, sleeves long or elbow length, or a loose sleeveless top
    T[6] = [
        poly([(11, 0), (7, 2), (4, 9), (3, 26), (6, 26), (7.5, 10), (7, 27), (21, 27), (20.5, 10), (22, 26), (25, 26), (24, 9), (21, 2), (17, 0), (14, 5)]),
        poly([(11, 0), (6, 2), (2, 12), (5, 14), (7, 9), (7, 27), (21, 27), (21, 9), (23, 14), (26, 12), (22, 2), (17, 0), (14, 5)]),
        poly([(9, 1), (19, 1), (21, 6), (22, 27), (6, 27), (7, 6)]),
    ]
    # 7: sneaker -- low wedge, toe at the left, opening at the right, vertically centred
    T[7] = [
        poly([(0, 17), (3, 14), (10, 12), (15, 8), (19, 9), (24, 9), (27, 7), (28, 10), (28, 20), (0, 20)]),
        poly([(0, 18), (4, 14), (12, 11), (16, 5), (27, 5), (28, 22), (0, 22)]),
        poly([(0, 16), (8, 13), (16, 10), (22, 11), (27, 9), (28, 19), (1, 19)]),
    ]
    # 8: bag -- wide box, often with a handle loop above it
    T[8] = [
        union(poly([(3, 10), (25, 10), (25, 27), (3, 27)]), ring(14, 10, 7, 9, 1.5)),
        poly([(2, 6), (26, 6), (27, 8), (27, 21), (26, 22), (2, 22), (1, 21), (1, 8)]),
        union(poly([(5, 12), (23, 12), (24, 27), (4, 27)]), ring(14, 12, 6, 12, 1.5)),
    ]
    # 9: ankle boot -- shoe with a tall shaft at the right, toe at the left, often a heel
    T[9] = [
        poly([(0, 20), (3, 16), (10, 13), (14, 9), (15, 2), (27, 2), (28, 24), (20, 24), (19, 22), (12, 23), (0, 24)]),
        poly([(0, 19), (4, 15), (11, 12), (14, 6), (27, 6), (28, 22), (0, 23)]),
        poly([(1, 22), (6, 17), (12, 14), (14, 0), (26, 0), (27, 27), (22, 27), (21, 22), (14, 23), (8, 26), (1, 26)]),
    ]
    return T


def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = shapes()
    rows = []
    for k in range(10):
        assert len(T[k]) == 3
        for m in T[k]:
            t = render(m)
            t = t - t.mean()                      # each template normalised by itself alone
            t = t / np.sqrt((t ** 2).sum())
            rows.append(t.reshape(-1))
    W = np.stack(rows).astype(np.float32)          # (30, 784)
    b = np.zeros(30, np.float32)
    net = {"layers": [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "templates", "op": "dense", "inputs": ["n0"], "out": 30},
        {"name": "group", "op": "reshape", "inputs": ["templates"], "shape": [1, 10, 3]},
        {"name": "best", "op": "maxpool", "inputs": ["group"], "size": [1, 3], "stride": [1, 3]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]},
    ]}
    (out_dir / "network.json").write_text(json.dumps(net, indent=1))
    np.savez(out_dir / "weights.npz", **{"templates.W": W, "templates.b": b})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
