import json
import sys
from pathlib import Path

import numpy as np

H = W = 14


def blank():
    return np.zeros((H, W), dtype=bool)


def rect(mask, y0, y1, x0, x1):
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    if y0 < y1 and x0 < x1:
        mask[y0:y1, x0:x1] = True


def trapezoid(mask, y0, y1, c0, c1, hw0, hw1):
    ys = list(range(max(0, int(y0)), min(H, int(y1))))
    if not ys:
        return
    n = len(ys)
    for j, y in enumerate(ys):
        t = 0.0 if n == 1 else j / (n - 1)
        c = (1.0 - t) * c0 + t * c1
        hw = (1.0 - t) * hw0 + t * hw1
        left = int(np.floor(c - hw))
        right = int(np.ceil(c + hw)) + 1
        rect(mask, y, y + 1, left, right)


def ellipse(mask, cy, cx, ry, rx, *, fill=True, thickness=1.0, y_max=None, y_min=None):
    yy, xx = np.ogrid[:H, :W]
    outer = ((yy - cy) / max(ry, 0.1)) ** 2 + ((xx - cx) / max(rx, 0.1)) ** 2 <= 1.0
    if fill:
        hit = outer
    else:
        iry = max(ry - thickness, 0.15)
        irx = max(rx - thickness, 0.15)
        inner = ((yy - cy) / iry) ** 2 + ((xx - cx) / irx) ** 2 <= 1.0
        hit = outer & ~inner
    if y_max is not None:
        hit &= yy <= y_max
    if y_min is not None:
        hit &= yy >= y_min
    mask |= hit


def subtract(a, *bs):
    out = a.copy()
    for b in bs:
        out &= ~b
    return out


def dilate(mask, margin=1):
    out = mask.copy()
    for dy in range(-margin, margin + 1):
        for dx in range(-margin, margin + 1):
            if dy == 0 and dx == 0:
                continue
            ys = slice(max(0, dy), H + min(0, dy))
            yd = slice(max(0, -dy), H + min(0, -dy))
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            out[yd, xd] |= mask[ys, xs]
    return out


def template_match_weights(mask, pos_gain, neg_gain):
    mask = mask.astype(bool)
    outer = dilate(mask, 1)
    neg = ~outer
    w = np.zeros((H, W), dtype=np.float32)
    pos_n = int(mask.sum())
    neg_n = int(neg.sum())
    if pos_n:
        w[mask] = np.float32(pos_gain / pos_n)
    if neg_n:
        w[neg] = np.float32(-neg_gain / neg_n)
    return w


def template_params(name):
    footwear = {"sneaker", "sandal", "sandal_heel", "boot_ankle", "boot_heel"}
    bags = {"bag_tote", "bag_clutch"}
    dresses = {"dress_flare", "dress_slim"}
    trousers = {"trouser", "trouser_wide"}
    if name in footwear:
        return 1.20, 0.45, 0.28
    if name in bags:
        return 1.15, 0.45, 0.30
    if name in dresses:
        return 1.10, 0.50, 0.34
    if name in trousers:
        return 1.10, 0.50, 0.35
    return 1.10, 0.55, 0.37


def make_templates():
    items = []

    m = blank()
    trapezoid(m, 3, 13, 6.5, 6.5, 2.7, 2.2)
    rect(m, 3, 6, 1, 5)
    rect(m, 3, 6, 9, 13)
    notch = blank()
    ellipse(notch, 2.5, 6.5, 1.2, 1.5, fill=True, y_max=3)
    m = subtract(m, notch)
    items.append(("tshirt", m))

    m = blank()
    trapezoid(m, 2, 13, 6.5, 6.5, 2.2, 2.6)
    rect(m, 1, 3, 4, 5)
    rect(m, 1, 3, 8, 9)
    notch = blank()
    ellipse(notch, 2.1, 6.5, 1.0, 1.2, fill=True, y_max=3)
    m = subtract(m, notch)
    items.append(("tank", m))

    m = blank()
    trapezoid(m, 3, 13, 6.5, 6.5, 3.2, 3.4)
    trapezoid(m, 3, 11, 2.4, 1.6, 1.3, 0.9)
    trapezoid(m, 3, 11, 10.6, 11.4, 1.3, 0.9)
    notch = blank()
    ellipse(notch, 2.2, 6.5, 0.9, 0.9, fill=True, y_max=3)
    m = subtract(m, notch)
    items.append(("pullover", m))

    m = blank()
    trapezoid(m, 3, 13, 6.5, 6.5, 3.1, 3.3)
    trapezoid(m, 3, 11, 2.5, 1.7, 1.3, 1.0)
    trapezoid(m, 3, 11, 10.5, 11.3, 1.3, 1.0)
    hood = blank()
    ellipse(hood, 2.5, 6.5, 1.6, 2.0, fill=False, thickness=0.8, y_max=4)
    m |= hood
    notch = blank()
    ellipse(notch, 2.3, 6.5, 0.7, 0.8, fill=True, y_max=3)
    m = subtract(m, notch)
    items.append(("hoodie", m))

    m = blank()
    trapezoid(m, 2, 13, 6.5, 6.5, 3.0, 3.6)
    trapezoid(m, 3, 11, 2.3, 1.5, 1.3, 1.0)
    trapezoid(m, 3, 11, 10.7, 11.5, 1.3, 1.0)
    split = blank()
    trapezoid(split, 2, 6, 5.3, 5.9, 0.5, 0.1)
    trapezoid(split, 2, 6, 7.7, 7.1, 0.5, 0.1)
    rect(split, 6, 13, 6, 8)
    m = subtract(m, split)
    items.append(("coat", m))

    m = blank()
    trapezoid(m, 3, 13, 6.5, 6.5, 2.6, 2.5)
    rect(m, 3, 6, 1, 4)
    rect(m, 3, 6, 10, 13)
    collar = blank()
    trapezoid(collar, 2, 5, 5.6, 6.0, 0.6, 0.1)
    trapezoid(collar, 2, 5, 7.4, 7.0, 0.6, 0.1)
    rect(collar, 5, 13, 6, 8)
    m = subtract(m, collar)
    items.append(("shirt_short", m))

    m = blank()
    trapezoid(m, 3, 13, 6.5, 6.5, 2.8, 2.9)
    trapezoid(m, 3, 11, 2.6, 1.9, 1.2, 0.9)
    trapezoid(m, 3, 11, 10.4, 11.1, 1.2, 0.9)
    collar = blank()
    trapezoid(collar, 2, 5, 5.6, 6.0, 0.6, 0.1)
    trapezoid(collar, 2, 5, 7.4, 7.0, 0.6, 0.1)
    rect(collar, 5, 13, 6, 8)
    m = subtract(m, collar)
    items.append(("shirt_long", m))

    m = blank()
    trapezoid(m, 2, 13, 6.5, 6.5, 1.7, 4.4)
    rect(m, 3, 5, 2, 4)
    rect(m, 3, 5, 10, 12)
    notch = blank()
    ellipse(notch, 2.2, 6.5, 0.9, 0.9, fill=True, y_max=3)
    m = subtract(m, notch)
    items.append(("dress_flare", m))

    m = blank()
    trapezoid(m, 1, 13, 6.5, 6.5, 1.5, 2.0)
    rect(m, 1, 3, 4, 5)
    rect(m, 1, 3, 8, 9)
    notch = blank()
    ellipse(notch, 1.9, 6.5, 0.8, 1.0, fill=True, y_max=2)
    m = subtract(m, notch)
    items.append(("dress_slim", m))

    m = blank()
    rect(m, 0, 2, 4, 10)
    trapezoid(m, 2, 14, 4.4, 4.0, 1.4, 1.2)
    trapezoid(m, 2, 14, 8.6, 9.0, 1.4, 1.2)
    gap = blank()
    trapezoid(gap, 4, 14, 6.5, 6.5, 0.6, 1.1)
    m = subtract(m, gap)
    items.append(("trouser", m))

    m = blank()
    rect(m, 0, 2, 4, 10)
    trapezoid(m, 2, 14, 4.4, 3.9, 1.5, 1.6)
    trapezoid(m, 2, 14, 8.6, 9.1, 1.5, 1.6)
    gap = blank()
    trapezoid(gap, 4, 14, 6.5, 6.5, 0.7, 1.0)
    m = subtract(m, gap)
    items.append(("trouser_wide", m))

    m = blank()
    trapezoid(m, 7, 10, 5.3, 7.8, 3.4, 4.9)
    rect(m, 10, 12, 2, 12)
    rect(m, 8, 10, 2, 4)
    toe = blank()
    trapezoid(toe, 6, 8, 8.8, 9.8, 2.0, 1.2)
    m |= toe
    items.append(("sneaker", m))

    m = blank()
    rect(m, 9, 11, 2, 12)
    rect(m, 7, 9, 4, 6)
    rect(m, 7, 9, 8, 10)
    rect(m, 6, 8, 10, 12)
    gap = blank()
    rect(gap, 7, 9, 6, 8)
    rect(gap, 8, 10, 2, 4)
    m = subtract(m, gap)
    items.append(("sandal", m))

    m = blank()
    rect(m, 9, 11, 3, 12)
    rect(m, 6, 10, 2, 3)
    rect(m, 7, 9, 5, 7)
    rect(m, 6, 8, 10, 12)
    gap = blank()
    rect(gap, 7, 9, 7, 10)
    m = subtract(m, gap)
    items.append(("sandal_heel", m))

    m = blank()
    trapezoid(m, 5, 13, 6.5, 6.5, 4.2, 4.5)
    handle = blank()
    ellipse(handle, 4.0, 6.5, 3.0, 2.2, fill=False, thickness=0.9, y_max=6)
    m |= handle
    hole = blank()
    rect(hole, 5, 7, 5, 9)
    m = subtract(m, hole)
    items.append(("bag_tote", m))

    m = blank()
    trapezoid(m, 7, 13, 6.5, 6.5, 4.5, 4.2)
    rect(m, 6, 8, 5, 9)
    items.append(("bag_clutch", m))

    m = blank()
    rect(m, 9, 11, 2, 12)
    trapezoid(m, 5, 10, 5.2, 5.0, 2.2, 2.0)
    rect(m, 4, 6, 4, 8)
    toe = blank()
    trapezoid(toe, 7, 9, 8.5, 9.8, 2.0, 1.4)
    m |= toe
    items.append(("boot_ankle", m))

    m = blank()
    rect(m, 9, 11, 3, 12)
    rect(m, 5, 10, 4, 8)
    rect(m, 4, 6, 4, 7)
    rect(m, 7, 10, 2, 3)
    toe = blank()
    trapezoid(toe, 7, 9, 9.0, 10.0, 1.8, 1.2)
    m |= toe
    items.append(("boot_heel", m))

    return items


def build_answer():
    templates = make_templates()
    names = [name for name, _ in templates]

    w1_rows = []
    b1 = []
    for name, mask in templates:
        pos_gain, neg_gain, threshold = template_params(name)
        w1_rows.append(template_match_weights(mask, pos_gain, neg_gain).reshape(-1))
        b1.append(-threshold)
    w1 = np.stack(w1_rows).astype(np.float32)
    b1 = np.array(b1, dtype=np.float32)

    name_to_i = {name: i for i, name in enumerate(names)}
    w2 = np.zeros((10, len(names)), dtype=np.float32)
    b2 = np.zeros((10,), dtype=np.float32)

    def add(class_id, weight, *template_names):
        for t in template_names:
            w2[class_id, name_to_i[t]] += np.float32(weight)

    add(0, 1.20, "tshirt")
    add(0, 1.00, "tank")

    add(1, 1.20, "trouser")
    add(1, 1.10, "trouser_wide")

    add(2, 1.15, "pullover")
    add(2, 0.95, "hoodie")

    add(3, 1.15, "dress_flare")
    add(3, 1.10, "dress_slim")

    add(4, 1.20, "coat")

    add(5, 1.20, "sandal")
    add(5, 1.10, "sandal_heel")

    add(6, 1.10, "shirt_short")
    add(6, 1.15, "shirt_long")

    add(7, 1.30, "sneaker")

    add(8, 1.20, "bag_tote")
    add(8, 1.10, "bag_clutch")

    add(9, 1.20, "boot_ankle")
    add(9, 1.10, "boot_heel")

    network = {
        "layers": [
            {
                "name": "pool",
                "op": "maxpool",
                "inputs": ["image"],
                "size": [2, 2],
                "stride": [2, 2],
            },
            {
                "name": "match",
                "op": "dense",
                "inputs": ["pool"],
                "out": int(w1.shape[0]),
            },
            {
                "name": "match_relu",
                "op": "relu",
                "inputs": ["match"],
            },
            {
                "name": "scores",
                "op": "dense",
                "inputs": ["match_relu"],
                "out": 10,
            },
        ]
    }
    weights = {
        "match.W": w1,
        "match.b": b1,
        "scores.W": w2,
        "scores.b": b2,
    }
    return network, weights


def main(out_dir):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    network, weights = build_answer()
    with (out_path / "network.json").open("w") as f:
        json.dump(network, f)
    np.savez(out_path / "weights.npz", **weights)


if __name__ == "__main__":
    main(sys.argv[1])
