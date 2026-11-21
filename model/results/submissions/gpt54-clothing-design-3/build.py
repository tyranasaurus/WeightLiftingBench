import json
import sys
from pathlib import Path

import numpy as np


CLASSES = 10


def blank(size):
    return np.zeros((size, size), np.float32)


def add_rect(mask, y0, y1, x0, x1, value=1.0):
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], np.float32(value))


def make_mask(size, rects):
    m = blank(size)
    for y0, y1, x0, x1 in rects:
        add_rect(m, y0, y1, x0, x1)
    return m


def append_template(specs, name, size, pos_rects, neg_rects, votes, thr=0.4, pos_gain=1.0, neg_gain=1.5, bias=0.0):
    specs.append(
        {
            "name": name,
            "pos": make_mask(size, pos_rects),
            "neg": make_mask(size, neg_rects),
            "votes": dict(votes),
            "thr": float(thr),
            "pos_gain": float(pos_gain),
            "neg_gain": float(neg_gain),
            "bias": float(bias),
        }
    )


def linear_dense(specs, size):
    n = size * size
    W = np.zeros((len(specs), n), np.float32)
    b = np.zeros(len(specs), np.float32)
    meta = []
    for row, spec in enumerate(specs):
        pos = spec["pos"].reshape(-1).astype(np.float32)
        neg = spec["neg"].reshape(-1).astype(np.float32)
        pos_gain = np.float32(spec.get("pos_gain", 1.0))
        neg_gain = np.float32(spec.get("neg_gain", 1.5))
        W[row] = pos_gain * pos - neg_gain * neg
        b[row] = np.float32(-spec.get("thr", 0.4) * pos_gain * float(pos.sum()) + spec.get("bias", 0.0))
        meta.append((spec["name"], spec["votes"]))
    return W, b, meta


def output_dense(meta, classes=CLASSES):
    W = np.zeros((classes, len(meta)), np.float32)
    b = np.zeros(classes, np.float32)
    for j, (_, votes) in enumerate(meta):
        for cls, w in votes.items():
            W[int(cls), j] = np.float32(w)
    return W, b


def prototype_dense(specs, size, classes=CLASSES):
    W = np.zeros((classes, size * size), np.float32)
    b = np.zeros(classes, np.float32)
    for spec in specs:
        cls = int(spec["class"])
        pos = make_mask(size, spec.get("pos", [])).reshape(-1).astype(np.float32)
        neg = make_mask(size, spec.get("neg", [])).reshape(-1).astype(np.float32)
        pos_gain = np.float32(spec.get("pos_gain", 2.0))
        neg_gain = np.float32(spec.get("neg_gain", 1.5))
        if pos.sum() > 0:
            W[cls] += pos_gain * pos / pos.sum()
        if neg.sum() > 0:
            W[cls] -= neg_gain * neg / neg.sum()
        b[cls] += np.float32(spec.get("bias", 0.0))
    return W, b


def proto_n14():
    return [
        {
            "class": 0,
            "pos": [(0, 3, 3, 11), (2, 5, 1, 13), (4, 10, 4, 10)],
            "neg": [(10, 14, 0, 14), (4, 10, 0, 2), (4, 10, 12, 14)],
            "pos_gain": 2.4,
            "neg_gain": 1.8,
            "bias": 0.1,
        },
        {
            "class": 1,
            "pos": [(0, 2, 4, 10), (2, 5, 3, 11), (5, 14, 3, 6), (5, 14, 8, 11)],
            "neg": [(5, 14, 6, 8), (0, 14, 0, 2), (0, 14, 12, 14)],
            "pos_gain": 3.0,
            "neg_gain": 2.5,
        },
        {
            "class": 2,
            "pos": [(0, 3, 3, 11), (1, 10, 0, 3), (1, 10, 11, 14), (2, 12, 4, 10)],
            "neg": [(12, 14, 0, 14)],
            "pos_gain": 2.6,
            "neg_gain": 1.5,
        },
        {
            "class": 3,
            "pos": [(0, 2, 4, 10), (2, 6, 5, 9), (6, 14, 2, 12)],
            "neg": [(0, 3, 0, 3), (0, 3, 11, 14)],
            "pos_gain": 2.6,
            "neg_gain": 1.8,
        },
        {
            "class": 4,
            "pos": [(0, 3, 3, 11), (1, 11, 0, 3), (1, 11, 11, 14), (2, 14, 4, 10), (10, 14, 3, 11)],
            "neg": [],
            "pos_gain": 2.7,
            "neg_gain": 1.0,
        },
        {
            "class": 5,
            "pos": [(10, 12, 1, 13), (7, 9, 2, 5), (7, 9, 8, 11), (8, 11, 1, 3)],
            "neg": [(0, 6, 0, 14), (6, 10, 5, 8)],
            "pos_gain": 2.8,
            "neg_gain": 2.0,
        },
        {
            "class": 6,
            "pos": [(0, 3, 3, 11), (1, 8, 1, 13), (3, 11, 4, 10)],
            "neg": [(2, 10, 6, 8), (11, 14, 0, 14)],
            "pos_gain": 2.3,
            "neg_gain": 1.6,
        },
        {
            "class": 7,
            "pos": [(10, 12, 1, 13), (8, 10, 1, 4), (7, 10, 4, 11), (8, 10, 11, 13)],
            "neg": [(0, 6, 0, 14), (4, 8, 1, 4)],
            "pos_gain": 2.8,
            "neg_gain": 1.8,
        },
        {
            "class": 8,
            "pos": [(6, 12, 1, 13), (3, 6, 2, 12), (1, 5, 3, 5), (1, 5, 9, 11), (1, 3, 5, 9)],
            "neg": [(2, 5, 5, 9), (0, 2, 0, 2), (0, 2, 12, 14)],
            "pos_gain": 2.8,
            "neg_gain": 1.8,
            "bias": 0.05,
        },
        {
            "class": 9,
            "pos": [(10, 12, 1, 13), (8, 10, 4, 11), (3, 10, 2, 7), (6, 10, 1, 4)],
            "neg": [(0, 4, 8, 14)],
            "pos_gain": 2.9,
            "neg_gain": 1.8,
        },
    ]


def specs_a7():
    s = []
    append_template(
        s,
        "upper_short",
        7,
        [(0, 2, 1, 6), (1, 3, 0, 7), (3, 7, 2, 5)],
        [(3, 7, 0, 1), (3, 7, 6, 7)],
        {0: 2.6, 6: 0.9, 2: -0.4, 4: -0.6, 5: -2.0, 7: -2.0, 9: -2.0, 8: -1.0},
        thr=0.40,
        neg_gain=1.4,
    )
    append_template(
        s,
        "upper_long",
        7,
        [(0, 2, 1, 6), (1, 5, 0, 2), (1, 5, 5, 7), (2, 7, 2, 5)],
        [],
        {2: 2.0, 4: 2.0, 6: 0.8, 0: -0.6, 5: -2.0, 7: -2.0, 9: -2.0, 8: -1.0},
        thr=0.40,
        neg_gain=1.3,
    )
    append_template(
        s,
        "dress",
        7,
        [(0, 2, 2, 5), (2, 4, 2, 5), (4, 7, 1, 6)],
        [(0, 2, 0, 2), (0, 2, 5, 7)],
        {3: 2.4, 4: 0.2, 1: -1.0, 5: -2.0, 7: -2.0, 9: -2.0},
        thr=0.38,
        neg_gain=1.4,
    )
    append_template(
        s,
        "trouser",
        7,
        [(0, 2, 2, 5), (2, 7, 1, 3), (2, 7, 4, 6)],
        [(3, 7, 3, 4), (0, 7, 0, 1), (0, 7, 6, 7)],
        {1: 3.0, 3: -1.4, 5: -2.0, 7: -2.0, 9: -1.8, 8: -0.8},
        thr=0.36,
        neg_gain=1.8,
    )
    append_template(
        s,
        "bag",
        7,
        [(2, 6, 1, 6), (0, 2, 2, 5)],
        [(1, 3, 3, 4)],
        {8: 2.8, 5: -1.5, 7: -1.5, 9: -1.5, 0: -0.6, 2: -0.6, 4: -0.6},
        thr=0.34,
        neg_gain=1.8,
    )
    append_template(
        s,
        "shoe_any",
        7,
        [(5, 7, 0, 7), (4, 6, 1, 6)],
        [(0, 4, 0, 7)],
        {5: 0.8, 7: 1.0, 9: 1.0, 8: 0.3, 0: -1.8, 2: -1.8, 3: -1.8, 4: -1.8, 6: -1.8, 1: -1.4},
        thr=0.32,
        neg_gain=1.6,
    )
    append_template(
        s,
        "shoe_open",
        7,
        [(5, 7, 0, 7), (4, 5, 1, 3), (4, 5, 4, 6)],
        [(3, 5, 3, 4), (0, 4, 0, 7)],
        {5: 1.6, 7: -0.4, 9: -0.4},
        thr=0.30,
        neg_gain=1.8,
    )
    append_template(
        s,
        "shoe_high",
        7,
        [(5, 7, 0, 7), (3, 6, 1, 5)],
        [(0, 2, 0, 7)],
        {9: 1.6, 7: 0.3, 5: -0.3},
        thr=0.32,
        neg_gain=1.5,
    )
    append_template(
        s,
        "upper_open",
        7,
        [(0, 2, 1, 6), (2, 6, 1, 3), (2, 6, 4, 6)],
        [(2, 6, 3, 4), (6, 7, 0, 7)],
        {6: 1.2, 4: 0.4, 0: 0.2},
        thr=0.34,
        neg_gain=1.8,
    )
    return s


def specs_a14():
    s = []
    append_template(
        s,
        "top_short",
        14,
        [(0, 2, 3, 11), (1, 4, 1, 13), (3, 14, 4, 10)],
        [(4, 14, 0, 2), (4, 14, 12, 14)],
        {0: 3.2, 6: 0.8, 2: -0.6, 4: -0.8, 5: -2.5, 7: -2.5, 9: -2.5, 8: -1.0},
        thr=0.42,
        neg_gain=1.4,
    )
    append_template(
        s,
        "top_tank",
        14,
        [(0, 2, 4, 10), (2, 14, 4, 10)],
        [(0, 4, 0, 3), (0, 4, 11, 14)],
        {0: 2.6, 6: 0.5, 3: 0.3, 2: -0.5, 4: -0.8, 5: -2.4, 7: -2.4, 9: -2.4, 8: -0.8},
        thr=0.42,
        neg_gain=1.4,
    )
    append_template(
        s,
        "pullover",
        14,
        [(0, 2, 3, 11), (1, 12, 0, 3), (1, 12, 11, 14), (2, 14, 4, 10)],
        [],
        {2: 3.0, 4: 0.6, 6: 0.4, 0: -0.8, 5: -2.8, 7: -2.8, 9: -2.8, 8: -1.0},
        thr=0.42,
        neg_gain=1.3,
    )
    append_template(
        s,
        "hoodie",
        14,
        [(0, 3, 2, 12), (2, 12, 0, 3), (2, 12, 11, 14), (3, 14, 4, 10)],
        [],
        {2: 2.7, 4: 0.7, 6: 0.3, 0: -0.6, 5: -2.6, 7: -2.6, 9: -2.6, 8: -1.0},
        thr=0.42,
        neg_gain=1.3,
    )
    append_template(
        s,
        "coat_open",
        14,
        [(0, 2, 3, 11), (1, 14, 0, 3), (1, 14, 11, 14), (2, 14, 3, 6), (2, 14, 8, 11)],
        [(3, 14, 6, 8)],
        {4: 3.2, 6: 0.6, 2: 0.2, 0: -0.8, 5: -2.8, 7: -2.8, 9: -2.8, 8: -0.8},
        thr=0.38,
        neg_gain=1.7,
    )
    append_template(
        s,
        "coat_closed",
        14,
        [(0, 2, 3, 11), (1, 14, 0, 3), (1, 14, 11, 14), (2, 14, 4, 10)],
        [],
        {4: 2.8, 2: 0.4, 6: 0.3, 0: -0.7, 5: -2.7, 7: -2.7, 9: -2.7, 8: -0.7},
        thr=0.43,
    )
    append_template(
        s,
        "shirt_open_short",
        14,
        [(0, 2, 3, 11), (1, 4, 1, 13), (2, 14, 4, 6), (2, 14, 8, 10)],
        [(2, 14, 6, 8)],
        {6: 2.8, 0: 0.6, 4: 0.4, 2: -0.2, 5: -2.4, 7: -2.4, 9: -2.4, 8: -0.6},
        thr=0.38,
        neg_gain=1.7,
    )
    append_template(
        s,
        "shirt_open_long",
        14,
        [(0, 2, 3, 11), (1, 8, 0, 3), (1, 8, 11, 14), (2, 10, 4, 6), (2, 10, 8, 10)],
        [(2, 10, 6, 8), (11, 14, 0, 14)],
        {6: 2.6, 4: 0.5, 2: 0.1, 0: -0.2, 5: -2.2, 7: -2.2, 9: -2.2, 8: -0.6},
        thr=0.38,
        neg_gain=1.7,
    )
    append_template(
        s,
        "generic_upper",
        14,
        [(0, 3, 3, 11), (2, 10, 4, 10)],
        [(11, 14, 0, 14)],
        {6: 1.6, 0: 0.6, 2: 0.8, 4: 0.8, 3: 0.3, 5: -2.2, 7: -2.2, 8: -0.8, 9: -2.2},
        thr=0.40,
        neg_gain=1.4,
    )
    append_template(
        s,
        "dress_flare",
        14,
        [(0, 2, 4, 10), (2, 5, 5, 9), (5, 14, 2, 12)],
        [(0, 2, 0, 3), (0, 2, 11, 14)],
        {3: 3.4, 4: 0.4, 1: -1.5, 5: -2.8, 7: -2.8, 8: -1.2, 9: -2.8, 0: -0.4},
        thr=0.42,
        neg_gain=1.4,
    )
    append_template(
        s,
        "dress_straight",
        14,
        [(0, 2, 4, 10), (2, 6, 5, 9), (6, 14, 3, 11)],
        [(0, 2, 0, 3), (0, 2, 11, 14)],
        {3: 2.8, 4: 0.4, 1: -1.0, 5: -2.6, 7: -2.6, 8: -1.0, 9: -2.6},
        thr=0.42,
        neg_gain=1.4,
    )
    append_template(
        s,
        "trouser_narrow",
        14,
        [(0, 2, 4, 10), (2, 5, 3, 11), (5, 14, 3, 6), (5, 14, 8, 11)],
        [(5, 14, 6, 8), (0, 14, 0, 2), (0, 14, 12, 14)],
        {1: 4.0, 3: -1.5, 5: -2.5, 7: -2.5, 9: -2.0, 8: -1.0, 0: -0.8},
        thr=0.40,
        neg_gain=1.8,
    )
    append_template(
        s,
        "trouser_wide",
        14,
        [(0, 2, 4, 10), (2, 5, 2, 12), (5, 14, 2, 6), (5, 14, 8, 12)],
        [(5, 14, 6, 8)],
        {1: 3.6, 3: -1.3, 5: -2.3, 7: -2.3, 9: -1.8, 8: -0.8, 0: -0.6},
        thr=0.42,
        neg_gain=1.8,
    )
    append_template(
        s,
        "bag_handle",
        14,
        [(5, 13, 2, 12), (1, 5, 3, 5), (1, 5, 9, 11), (1, 3, 5, 9)],
        [(2, 5, 5, 9)],
        {8: 4.0, 5: -1.6, 7: -1.6, 9: -1.6, 0: -0.8, 2: -0.8, 4: -0.8, 3: -0.5},
        thr=0.36,
        neg_gain=1.9,
    )
    append_template(
        s,
        "bag_box",
        14,
        [(6, 12, 1, 13), (2, 6, 4, 6), (2, 6, 8, 10)],
        [(0, 2, 0, 14)],
        {8: 3.2, 5: -1.4, 7: -1.4, 9: -1.4, 0: -0.6, 2: -0.6, 4: -0.6},
        thr=0.38,
        neg_gain=1.5,
    )
    append_template(
        s,
        "bag_flat",
        14,
        [(7, 12, 1, 13)],
        [(0, 5, 0, 14)],
        {8: 2.2, 5: -0.8, 7: -0.8, 9: -0.8, 0: -0.5, 2: -0.5, 4: -0.5},
        thr=0.34,
        neg_gain=1.6,
    )
    append_template(
        s,
        "sneaker_low",
        14,
        [(10, 12, 1, 13), (8, 10, 1, 4), (7, 10, 4, 11), (8, 10, 11, 13)],
        [(0, 6, 0, 14)],
        {7: 3.2, 9: 0.6, 5: -1.2, 8: -0.8, 0: -2.5, 2: -2.5, 3: -2.5, 4: -2.5, 6: -2.5},
        thr=0.38,
        neg_gain=1.7,
    )
    append_template(
        s,
        "sneaker_high",
        14,
        [(10, 12, 1, 13), (7, 10, 1, 4), (5, 9, 2, 5), (7, 10, 4, 11)],
        [(0, 3, 0, 6)],
        {7: 2.8, 9: 0.8, 5: -1.0, 8: -0.7, 0: -2.3, 2: -2.3, 3: -2.3, 4: -2.3, 6: -2.3},
        thr=0.38,
        neg_gain=1.5,
    )
    append_template(
        s,
        "boot_flat",
        14,
        [(10, 12, 1, 13), (6, 10, 1, 4), (3, 10, 3, 7), (6, 10, 7, 11)],
        [(0, 2, 8, 14)],
        {9: 3.2, 7: -0.8, 5: -0.8, 8: -0.6, 0: -2.4, 2: -2.4, 3: -2.4, 4: -2.4, 6: -2.4},
        thr=0.38,
        neg_gain=1.5,
    )
    append_template(
        s,
        "boot_heel",
        14,
        [(7, 12, 1, 3), (10, 12, 3, 13), (3, 10, 3, 8), (6, 10, 8, 11)],
        [(0, 2, 8, 14)],
        {9: 3.4, 5: 0.2, 7: -1.0, 8: -0.6, 0: -2.4, 2: -2.4, 3: -2.4, 4: -2.4, 6: -2.4},
        thr=0.36,
        neg_gain=1.5,
    )
    append_template(
        s,
        "sandal_flat",
        14,
        [(10, 12, 1, 13), (7, 9, 3, 5), (7, 9, 8, 10)],
        [(6, 10, 5, 8), (0, 6, 0, 14)],
        {5: 3.2, 7: -1.0, 9: -1.0, 8: -0.6, 0: -2.2, 2: -2.2, 3: -2.2, 4: -2.2, 6: -2.2},
        thr=0.32,
        neg_gain=1.9,
    )
    append_template(
        s,
        "sandal_heel",
        14,
        [(7, 12, 1, 3), (10, 12, 3, 13), (7, 9, 7, 9), (8, 10, 10, 12)],
        [(6, 10, 4, 7), (0, 6, 0, 14)],
        {5: 3.4, 9: -0.5, 7: -0.9, 8: -0.5, 0: -2.2, 2: -2.2, 3: -2.2, 4: -2.2, 6: -2.2},
        thr=0.32,
        neg_gain=1.9,
    )
    append_template(
        s,
        "footwear_any",
        14,
        [(9, 12, 1, 13), (7, 10, 4, 11)],
        [(0, 6, 0, 14)],
        {5: 1.0, 7: 1.0, 9: 1.0, 8: 0.3, 0: -2.0, 2: -2.0, 3: -2.0, 4: -2.0, 6: -2.0, 1: -1.5},
        thr=0.34,
        neg_gain=1.6,
    )
    return s


def specs_m14():
    s = []
    append_template(
        s,
        "bag_handle_max",
        14,
        [(5, 13, 2, 12), (1, 5, 3, 5), (1, 5, 9, 11), (1, 3, 5, 9)],
        [(2, 5, 5, 9)],
        {8: 2.0, 5: -0.5, 7: -0.5, 9: -0.5},
        thr=0.42,
        neg_gain=1.8,
    )
    append_template(
        s,
        "shoe_any_max",
        14,
        [(9, 12, 1, 13), (7, 10, 4, 11)],
        [(0, 6, 0, 14)],
        {5: 0.8, 7: 0.8, 9: 0.8, 0: -1.5, 2: -1.5, 3: -1.5, 4: -1.5, 6: -1.5, 1: -1.0},
        thr=0.42,
        neg_gain=1.5,
    )
    append_template(
        s,
        "sandal_open_max",
        14,
        [(10, 12, 1, 13), (7, 9, 3, 5), (7, 9, 8, 10)],
        [(6, 10, 5, 8), (0, 6, 0, 14)],
        {5: 2.0, 7: -0.7, 9: -0.7},
        thr=0.38,
        neg_gain=1.9,
    )
    append_template(
        s,
        "sneaker_closed_max",
        14,
        [(10, 12, 1, 13), (8, 10, 1, 4), (7, 10, 4, 11), (8, 10, 11, 13)],
        [(0, 6, 0, 14)],
        {7: 2.0, 5: -0.7, 9: -0.3},
        thr=0.42,
        neg_gain=1.6,
    )
    append_template(
        s,
        "boot_tall_max",
        14,
        [(10, 12, 1, 13), (6, 10, 1, 4), (3, 10, 3, 7), (6, 10, 7, 11)],
        [(0, 2, 8, 14)],
        {9: 2.0, 7: -0.6, 5: -0.6},
        thr=0.40,
        neg_gain=1.5,
    )
    append_template(
        s,
        "shirt_open_max",
        14,
        [(0, 2, 3, 11), (1, 4, 1, 13), (2, 10, 4, 6), (2, 10, 8, 10)],
        [(2, 10, 6, 8)],
        {6: 1.2, 4: 0.3, 0: 0.2},
        thr=0.42,
        neg_gain=1.8,
    )
    append_template(
        s,
        "coat_open_max",
        14,
        [(0, 2, 3, 11), (1, 11, 0, 3), (1, 11, 11, 14), (2, 14, 3, 6), (2, 14, 8, 11)],
        [(3, 14, 6, 8)],
        {4: 1.2, 6: 0.4},
        thr=0.42,
        neg_gain=1.8,
    )
    append_template(
        s,
        "trouser_gap_max",
        14,
        [(0, 2, 4, 10), (2, 5, 3, 11), (5, 14, 3, 6), (5, 14, 8, 11)],
        [(5, 14, 6, 8)],
        {1: 1.8},
        thr=0.42,
        neg_gain=2.0,
    )
    append_template(
        s,
        "dress_max",
        14,
        [(0, 2, 4, 10), (2, 5, 5, 9), (5, 14, 2, 12)],
        [(0, 2, 0, 3), (0, 2, 11, 14)],
        {3: 1.4},
        thr=0.44,
        neg_gain=1.5,
    )
    return s


def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a7 = specs_a7()
    a14 = specs_a14()
    m14 = specs_m14()
    n14 = proto_n14()

    net = {
        "layers": [
            {"name": "a7", "op": "avgpool", "inputs": ["image"], "size": [4, 4], "stride": [4, 4]},
            {"name": "a7_t", "op": "dense", "inputs": ["a7"], "out": len(a7)},
            {"name": "a7_r", "op": "relu", "inputs": ["a7_t"]},
            {"name": "a7_out", "op": "dense", "inputs": ["a7_r"], "out": 10},
            {"name": "a14", "op": "avgpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]},
            {"name": "a14_t", "op": "dense", "inputs": ["a14"], "out": len(a14)},
            {"name": "a14_r", "op": "relu", "inputs": ["a14_t"]},
            {"name": "a14_out", "op": "dense", "inputs": ["a14_r"], "out": 10},
            {"name": "m14", "op": "maxpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]},
            {"name": "m14_t", "op": "dense", "inputs": ["m14"], "out": len(m14)},
            {"name": "m14_r", "op": "relu", "inputs": ["m14_t"]},
            {"name": "m14_out", "op": "dense", "inputs": ["m14_r"], "out": 10},
            {"name": "n14", "op": "avgpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]},
            {"name": "n14_norm", "op": "norm", "inputs": ["n14"]},
            {"name": "n14_score", "op": "dense", "inputs": ["n14_norm"], "out": 10},
            {"name": "out", "op": "add", "inputs": ["a7_out", "a14_out", "m14_out", "n14_score"]},
        ]
    }

    weights = {}

    W, b, meta = linear_dense(a7, 7)
    weights["a7_t.W"] = W
    weights["a7_t.b"] = b
    W, b = output_dense(meta)
    weights["a7_out.W"] = (0.45 * W).astype(np.float32)
    weights["a7_out.b"] = (0.45 * b).astype(np.float32)

    W, b, meta = linear_dense(a14, 14)
    weights["a14_t.W"] = W
    weights["a14_t.b"] = b
    W, b = output_dense(meta)
    weights["a14_out.W"] = (0.30 * W).astype(np.float32)
    weights["a14_out.b"] = (0.30 * b).astype(np.float32)

    W, b, meta = linear_dense(m14, 14)
    weights["m14_t.W"] = W
    weights["m14_t.b"] = b
    W, b = output_dense(meta)
    weights["m14_out.W"] = (0.35 * W).astype(np.float32)
    weights["m14_out.b"] = (0.35 * b).astype(np.float32)

    W, b = prototype_dense(n14, 14)
    weights["n14_score.W"] = (4.0 * W).astype(np.float32)
    weights["n14_score.b"] = (4.0 * b).astype(np.float32)

    with (out_dir / "network.json").open("w") as f:
        json.dump(net, f)
    np.savez(out_dir / "weights.npz", **{k: v.astype(np.float32) for k, v in weights.items()})


if __name__ == "__main__":
    build(sys.argv[1])
