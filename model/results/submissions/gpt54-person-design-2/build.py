import json
import sys
from pathlib import Path

import numpy as np


LUMA = np.array([0.30, 0.59, 0.11], np.float32)


def grid(k):
    r = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k].astype(np.float32)
    if r <= 0:
        return x * 0.0, y * 0.0
    return (x - r) / r, (y - r) / r


def normalize_pos(m):
    m = np.maximum(np.asarray(m, np.float32), 0.0)
    s = float(m.sum())
    return m / s if s > 1e-8 else m


def core(mask, power=1.0):
    return normalize_pos(np.maximum(mask, 0.0) ** power)


def ellipse(x, y, cx, cy, rx, ry, sharp=3.0):
    return np.exp(-0.5 * sharp * (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)).astype(np.float32)


def capsule(x, y, ax, ay, bx, by, r, sharp=4.0):
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy + 1e-8
    t = ((x - ax) * dx + (y - ay) * dy) / den
    t = np.clip(t, 0.0, 1.0)
    px, py = ax + t * dx, ay + t * dy
    d2 = ((x - px) / r) ** 2 + ((y - py) / r) ** 2
    return np.exp(-0.5 * sharp * d2).astype(np.float32)


def blur3(m, rounds=2):
    m = m.astype(np.float32)
    for _ in range(rounds):
        p = np.pad(m, 1, mode="edge")
        m = (
            p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]
            + 2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:]
            + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
        ) / 16.0
    return m.astype(np.float32)


def union(*parts):
    return np.maximum.reduce([np.asarray(p, np.float32) for p in parts])


def ring(mask, inner=0.58, rounds=3):
    return normalize_pos(np.clip(blur3(mask, rounds=rounds) - inner * mask, 0.0, None))


def add_mass(T, out, ch, mask, gain, power=1.0):
    T[out, ch] += np.float32(gain) * core(mask, power)


def add_edge_ring(T, out, mask, gain):
    m = ring(mask)
    for ch in range(7, 15):
        T[out, ch] += np.float32(gain) * m


def first_layer():
    W = np.zeros((16, 3, 3, 3), np.float32)
    b = np.zeros(16, np.float32)

    def center(out, rgb, bias):
        W[out, :, 1, 1] = np.array(rgb, np.float32)
        b[out] = np.float32(bias)

    center(0, [1.20, 0.55, -1.15], -0.18)  # warm loose
    center(1, [0.95, 0.25, -0.95], -0.28)  # warm strict
    center(2, [-0.30, -0.59, -0.11], 0.42)  # dark
    center(3, [0.30, 0.59, 0.11], -0.75)  # light
    center(4, [1.00, -0.35, -0.35], -0.22)  # red
    center(5, [0.70, 0.20, -1.00], -0.05)  # low blue / yellow-brown
    center(6, [0.30, 0.59, 0.11], 0.00)  # luma
    center(15, [0.30, 0.59, 0.11], -0.28)  # medium+ brightness

    sobx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0
    soby = sobx.T
    diag1 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], np.float32) / 8.0
    diag2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], np.float32) / 8.0

    def edge(out, kern):
        for c, a in enumerate(LUMA):
            W[out, c] = a * kern

    edge(7, sobx)
    edge(8, -sobx)
    edge(9, soby)
    edge(10, -soby)
    edge(11, diag1)
    edge(12, -diag1)
    edge(13, diag2)
    edge(14, -diag2)
    return W, b


def template_layer():
    k = 15
    x, y = grid(k)

    head = ellipse(x, y, 0.0, -0.52, 0.22, 0.22, sharp=4.5)
    portrait = ellipse(x, y, 0.0, 0.05, 0.44, 0.63, sharp=2.2)
    eye_l = ellipse(x, y, -0.12, -0.55, 0.08, 0.04, sharp=8.0)
    eye_r = ellipse(x, y, 0.12, -0.55, 0.08, 0.04, sharp=8.0)
    eyes = union(eye_l, eye_r)
    mouth = ellipse(x, y, 0.0, -0.38, 0.10, 0.04, sharp=7.0)
    portrait_eyes = capsule(x, y, -0.32, -0.15, 0.32, -0.15, 0.09, sharp=5.5)
    portrait_mouth = ellipse(x, y, 0.0, 0.18, 0.14, 0.06, sharp=6.0)

    shoulders = capsule(x, y, -0.52, -0.10, 0.52, -0.10, 0.18, sharp=3.0)
    torso = ellipse(x, y, 0.0, 0.18, 0.28, 0.38, sharp=2.8)
    hips = ellipse(x, y, 0.0, 0.50, 0.24, 0.16, sharp=3.2)
    leg_l = capsule(x, y, -0.14, 0.30, -0.22, 0.98, 0.11, sharp=4.0)
    leg_r = capsule(x, y, 0.14, 0.30, 0.22, 0.98, 0.11, sharp=4.0)
    arm_l = capsule(x, y, -0.24, 0.04, -0.55, 0.38, 0.09, sharp=4.0)
    arm_r = capsule(x, y, 0.24, 0.04, 0.55, 0.38, 0.09, sharp=4.0)

    upper = union(head, shoulders, torso)
    full = union(upper, hips, leg_l, leg_r, arm_l, arm_r)

    head_l = ellipse(x, y, -0.28, -0.42, 0.20, 0.20, sharp=4.5)
    head_r = ellipse(x, y, 0.28, -0.42, 0.20, 0.20, sharp=4.5)
    group_body = ellipse(x, y, 0.0, 0.20, 0.38, 0.36, sharp=2.5)
    group_shoulders = capsule(x, y, -0.58, -0.08, 0.58, -0.08, 0.22, sharp=3.0)
    group = union(head_l, head_r, group_shoulders, group_body)
    group_eyes = union(
        ellipse(x, y, -0.33, -0.43, 0.08, 0.04, sharp=8.0),
        ellipse(x, y, 0.33, -0.43, 0.08, 0.04, sharp=8.0),
    )

    T = np.zeros((6, 16, k, k), np.float32)
    b = np.zeros(6, np.float32)

    # 0: face
    add_mass(T, 0, 0, head, 1.7, power=1.5)
    add_mass(T, 0, 1, head, 2.0, power=1.7)
    add_mass(T, 0, 5, head, 0.6, power=1.3)
    add_mass(T, 0, 15, head, 0.45, power=1.0)
    add_mass(T, 0, 2, union(eyes, mouth), 1.25, power=1.0)
    add_edge_ring(T, 0, head, 0.85)
    b[0] = -0.92

    # 1: large portrait / close face
    add_mass(T, 1, 0, portrait, 1.25, power=1.2)
    add_mass(T, 1, 1, portrait, 1.45, power=1.3)
    add_mass(T, 1, 5, portrait, 0.55, power=1.0)
    add_mass(T, 1, 3, portrait, 0.35, power=1.0)
    add_mass(T, 1, 2, union(portrait_eyes, portrait_mouth), 1.10, power=1.0)
    add_edge_ring(T, 1, portrait, 0.70)
    b[1] = -0.90

    # 2: head and shoulders
    add_mass(T, 2, 0, head, 0.90, power=1.4)
    add_mass(T, 2, 1, head, 1.10, power=1.5)
    add_mass(T, 2, 15, union(shoulders, torso), 0.95, power=1.0)
    add_mass(T, 2, 2, union(eyes, mouth), 0.55, power=1.0)
    add_mass(T, 2, 2, union(shoulders, torso), 0.18, power=1.0)
    add_mass(T, 2, 3, union(shoulders, torso), 0.18, power=1.0)
    add_edge_ring(T, 2, upper, 1.10)
    b[2] = -1.00

    # 3: standing body silhouette
    add_mass(T, 3, 0, head, 0.65, power=1.3)
    add_mass(T, 3, 1, head, 0.78, power=1.5)
    add_mass(T, 3, 15, full, 0.95, power=1.0)
    add_mass(T, 3, 2, union(torso, hips), 0.18, power=1.0)
    add_mass(T, 3, 3, union(torso, hips), 0.18, power=1.0)
    add_edge_ring(T, 3, full, 1.45)
    b[3] = -1.10

    # 4: two adjacent heads / small group
    add_mass(T, 4, 0, union(head_l, head_r), 1.00, power=1.4)
    add_mass(T, 4, 1, union(head_l, head_r), 1.18, power=1.5)
    add_mass(T, 4, 5, union(head_l, head_r), 0.40, power=1.1)
    add_mass(T, 4, 15, group, 0.62, power=1.0)
    add_mass(T, 4, 2, group_eyes, 0.75, power=1.0)
    add_edge_ring(T, 4, group, 0.90)
    b[4] = -0.98

    # 5: generic human-like silhouette with weak colour assumptions
    add_mass(T, 5, 2, upper, 0.36, power=1.0)
    add_mass(T, 5, 3, upper, 0.36, power=1.0)
    add_mass(T, 5, 15, full, 0.55, power=1.0)
    add_edge_ring(T, 5, full, 1.70)
    b[5] = -1.15

    return T, b


def dense_layers():
    W0 = np.zeros((2, 6), np.float32)
    b0 = np.array([0.0, -0.58], np.float32)
    W0[1] = np.array([0.75, 0.35, 0.70, 0.80, 0.55, 0.35], np.float32)

    W1 = np.zeros((2, 6), np.float32)
    b1 = np.array([0.0, -0.55], np.float32)
    W1[1] = np.array([0.85, 0.55, 0.80, 0.75, 0.60, 0.45], np.float32)

    W2 = np.zeros((2, 6), np.float32)
    b2 = np.array([0.0, -0.48], np.float32)
    W2[1] = np.array([1.00, 0.95, 0.45, 0.35, 0.25, 0.55], np.float32)
    return (W0, b0), (W1, b1), (W2, b2)


def network_dict():
    return {
        "layers": [
            {"name": "f0", "op": "conv", "inputs": ["image"], "out": 16, "kernel": [3, 3], "stride": 1, "padding": 1},
            {"name": "r0", "op": "relu", "inputs": ["f0"]},
            {"name": "t0", "op": "conv", "inputs": ["r0"], "out": 6, "kernel": [15, 15], "stride": 1, "padding": 7},
            {"name": "h0", "op": "relu", "inputs": ["t0"]},
            {"name": "g0", "op": "maxpool", "inputs": ["h0"], "size": [96, 96], "stride": [96, 96]},
            {"name": "p1", "op": "avgpool", "inputs": ["r0"], "size": [2, 2], "stride": [2, 2]},
            {"name": "t1", "op": "conv", "inputs": ["p1"], "out": 6, "kernel": [15, 15], "stride": 1, "padding": 7},
            {"name": "h1", "op": "relu", "inputs": ["t1"]},
            {"name": "g1", "op": "maxpool", "inputs": ["h1"], "size": [48, 48], "stride": [48, 48]},
            {"name": "p2", "op": "avgpool", "inputs": ["p1"], "size": [2, 2], "stride": [2, 2]},
            {"name": "t2", "op": "conv", "inputs": ["p2"], "out": 6, "kernel": [15, 15], "stride": 1, "padding": 7},
            {"name": "h2", "op": "relu", "inputs": ["t2"]},
            {"name": "g2", "op": "maxpool", "inputs": ["h2"], "size": [24, 24], "stride": [24, 24]},
            {"name": "s0", "op": "reshape", "inputs": ["g0"], "shape": [6]},
            {"name": "s1", "op": "reshape", "inputs": ["g1"], "shape": [6]},
            {"name": "s2", "op": "reshape", "inputs": ["g2"], "shape": [6]},
            {"name": "d0", "op": "dense", "inputs": ["s0"], "out": 2},
            {"name": "d1", "op": "dense", "inputs": ["s1"], "out": 2},
            {"name": "d2", "op": "dense", "inputs": ["s2"], "out": 2},
            {"name": "out", "op": "add", "inputs": ["d0", "d1", "d2"]},
        ]
    }


def build(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    net = network_dict()
    fW, fb = first_layer()
    tW, tb = template_layer()
    (d0W, d0b), (d1W, d1b), (d2W, d2b) = dense_layers()

    with (out / "network.json").open("w") as f:
        json.dump(net, f)

    np.savez(
        out / "weights.npz",
        **{
            "f0.W": fW.astype(np.float32),
            "f0.b": fb.astype(np.float32),
            "t0.W": tW.astype(np.float32),
            "t0.b": tb.astype(np.float32),
            "t1.W": tW.astype(np.float32),
            "t1.b": tb.astype(np.float32),
            "t2.W": tW.astype(np.float32),
            "t2.b": tb.astype(np.float32),
            "d0.W": d0W.astype(np.float32),
            "d0.b": d0b.astype(np.float32),
            "d1.W": d1W.astype(np.float32),
            "d1.b": d1b.astype(np.float32),
            "d2.W": d2W.astype(np.float32),
            "d2.b": d2b.astype(np.float32),
        }
    )


if __name__ == "__main__":
    build(sys.argv[1])
