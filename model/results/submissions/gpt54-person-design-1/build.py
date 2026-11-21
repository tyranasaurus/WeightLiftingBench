from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

NORM_EPS = 1e-5


def conv(x: np.ndarray, W: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho = (h + 2 * padding - kh) // stride + 1
    wo = (w + 2 * padding - kw) // stride + 1
    cols = np.stack(
        [x[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)],
        axis=2,
    )
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def avgpool2(x: np.ndarray) -> np.ndarray:
    return (x[:, :, 0::2, 0::2] + x[:, :, 1::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]) * 0.25


def norm(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=(2, 3), keepdims=True)
    z = x - mu
    return z / np.sqrt((z * z).mean(axis=(2, 3), keepdims=True) + NORM_EPS)


def make_edge_filters() -> np.ndarray:
    s = np.array([1.0, 4.0, 6.0, 4.0, 1.0], np.float32)
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0], np.float32)
    gx = np.outer(s, d)
    gy = np.outer(d, s)
    g45 = (gx + gy) / np.sqrt(2.0)
    g135 = (gx - gy) / np.sqrt(2.0)
    lap = np.array(
        [
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, -2.0, -1.0, 0.0],
            [-1.0, -2.0, 16.0, -2.0, -1.0],
            [0.0, -1.0, -2.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
        ],
        np.float32,
    )
    base = [gx, g45, gy, g135, lap]
    filt = []
    for f in base:
        f = f.astype(np.float32)
        f /= np.sqrt((f * f).sum()) + 1e-12
        filt.append(f)
    all_filters = filt + [-f for f in filt]
    W = np.stack(all_filters)[:, None, :, :].astype(np.float32)
    return W


EDGE_W = make_edge_filters()
EDGE_B = np.zeros((EDGE_W.shape[0],), np.float32)


PAIR_INDEX = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]


def circle_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def ellipse_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def capsule_mask(h: int, w: int, p0: tuple[float, float], p1: tuple[float, float], r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x0, y0 = p0
    x1, y1 = p1
    vx, vy = x1 - x0, y1 - y0
    denom = vx * vx + vy * vy + 1e-6
    t = ((xx - x0) * vx + (yy - y0) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    return ((xx - px) ** 2 + (yy - py) ** 2) <= r * r


GAUSS3 = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], np.float32) / 16.0


def blur(img: np.ndarray, rounds: int = 2) -> np.ndarray:
    out = img.astype(np.float32)
    for _ in range(rounds):
        p = np.pad(out, 1)
        out = sum(GAUSS3[i, j] * p[i:i + out.shape[0], j:j + out.shape[1]] for i in range(3) for j in range(3))
    return out


FULL_POSES = {"frontal", "stride", "side", "armsout"}


def render_pose(kind: str, size: int, *, aspect: float = 1.0, mirror: bool = False, x_shift: float = 0.0, y_shift: float = 0.0) -> np.ndarray:
    H = W = int(size)
    m = np.zeros((H, W), np.float32)
    cx = W * (0.5 + x_shift)
    cy_shift = H * y_shift
    sx = aspect * W * 0.22
    head_r = H * 0.10
    head_c = (cx, H * 0.16 + cy_shift)
    shoulder_y = H * 0.28 + cy_shift
    hip_y = H * 0.50 + cy_shift
    foot_y = H * 0.94 + cy_shift
    shoulder_dx = sx * 0.95
    hip_dx = sx * 0.60
    arm_r = H * 0.045
    leg_r = H * 0.048
    torso_rx = sx * 0.78
    torso_ry = H * 0.22

    if kind == "frontal":
        arms = [
            ((cx - shoulder_dx, shoulder_y), (cx - shoulder_dx * 1.05, hip_y + H * 0.10)),
            ((cx + shoulder_dx, shoulder_y), (cx + shoulder_dx * 1.05, hip_y + H * 0.10)),
        ]
        legs = [
            ((cx - hip_dx * 0.55, hip_y), (cx - hip_dx * 0.9, foot_y)),
            ((cx + hip_dx * 0.55, hip_y), (cx + hip_dx * 0.9, foot_y)),
        ]
        torso = (cx, (shoulder_y + hip_y) * 0.52, torso_rx * 1.0, torso_ry)
    elif kind == "stride":
        arms = [
            ((cx - shoulder_dx, shoulder_y), (cx - shoulder_dx * 0.2, hip_y + H * 0.02)),
            ((cx + shoulder_dx, shoulder_y), (cx + shoulder_dx * 1.35, hip_y - H * 0.02)),
        ]
        legs = [
            ((cx - hip_dx * 0.45, hip_y), (cx - hip_dx * 1.1, foot_y)),
            ((cx + hip_dx * 0.25, hip_y), (cx + hip_dx * 0.2, foot_y - H * 0.10)),
        ]
        torso = (cx, (shoulder_y + hip_y) * 0.52, torso_rx * 0.95, torso_ry)
    elif kind == "side":
        head_c = (cx + sx * 0.10, H * 0.16 + cy_shift)
        arms = [
            ((cx + shoulder_dx * 0.2, shoulder_y), (cx + shoulder_dx * 1.2, hip_y - H * 0.02)),
            ((cx - shoulder_dx * 0.05, shoulder_y), (cx - shoulder_dx * 0.2, hip_y + H * 0.12)),
        ]
        legs = [
            ((cx, hip_y), (cx - sx * 0.55, foot_y - H * 0.04)),
            ((cx, hip_y + H * 0.04), (cx + sx * 0.25, foot_y - H * 0.18)),
        ]
        torso = (cx, (shoulder_y + hip_y) * 0.52, torso_rx * 0.58, torso_ry)
    elif kind == "armsout":
        arms = [
            ((cx - shoulder_dx, shoulder_y), (cx - shoulder_dx * 1.7, shoulder_y + H * 0.05)),
            ((cx + shoulder_dx, shoulder_y), (cx + shoulder_dx * 1.7, shoulder_y + H * 0.05)),
        ]
        legs = [
            ((cx - hip_dx * 0.45, hip_y), (cx - hip_dx * 1.0, foot_y)),
            ((cx + hip_dx * 0.45, hip_y), (cx + hip_dx * 1.0, foot_y)),
        ]
        torso = (cx, (shoulder_y + hip_y) * 0.52, torso_rx * 0.9, torso_ry)
    elif kind == "upper":
        arms = [
            ((cx - shoulder_dx, shoulder_y), (cx - shoulder_dx * 1.05, hip_y - H * 0.03)),
            ((cx + shoulder_dx, shoulder_y), (cx + shoulder_dx * 1.05, hip_y - H * 0.03)),
        ]
        legs = []
        torso = (cx, (shoulder_y + hip_y - H * 0.10) * 0.53, torso_rx * 1.15, torso_ry * 1.20)
    elif kind == "seated":
        arms = [
            ((cx - shoulder_dx, shoulder_y), (cx - shoulder_dx * 0.4, hip_y + H * 0.04)),
            ((cx + shoulder_dx, shoulder_y), (cx + shoulder_dx * 0.9, hip_y - H * 0.03)),
        ]
        legs = [
            ((cx - hip_dx * 0.2, hip_y), (cx + sx * 0.25, hip_y + H * 0.15)),
            ((cx + sx * 0.18, hip_y + H * 0.15), (cx + sx * 0.55, hip_y + H * 0.15)),
        ]
        torso = (cx, (shoulder_y + hip_y) * 0.50, torso_rx * 0.9, torso_ry * 0.95)
    elif kind == "close":
        head_c = (cx, H * 0.22 + cy_shift)
        head_r = H * 0.16
        arms = [
            ((cx - shoulder_dx * 1.3, H * 0.42 + cy_shift), (cx - shoulder_dx * 1.8, H * 0.82 + cy_shift)),
            ((cx + shoulder_dx * 1.3, H * 0.42 + cy_shift), (cx + shoulder_dx * 1.8, H * 0.82 + cy_shift)),
        ]
        legs = []
        torso = (cx, H * 0.58 + cy_shift, torso_rx * 1.6, H * 0.22)
    else:
        raise ValueError(kind)

    if mirror:
        def flip_pt(pt: tuple[float, float]) -> tuple[float, float]:
            x, y = pt
            return (2.0 * cx - x, y)

        head_c = flip_pt(head_c)
        arms = [(flip_pt(b), flip_pt(a)) for a, b in arms]
        legs = [(flip_pt(b), flip_pt(a)) for a, b in legs]

    m[circle_mask(H, W, head_c[0], head_c[1], head_r)] = 1.0
    m[ellipse_mask(H, W, torso[0], torso[1], torso[2], torso[3])] = 1.0
    for a, b in arms:
        m[capsule_mask(H, W, a, b, arm_r)] = 1.0
    for a, b in legs:
        m[capsule_mask(H, W, a, b, leg_r)] = 1.0
    m[capsule_mask(H, W, (cx, head_c[1] + head_r * 0.8), (cx, shoulder_y + H * 0.03), H * 0.035)] = 1.0

    if kind in {"upper", "close"}:
        ex = sx * 0.38
        ey = head_c[1] + head_r * 0.05
        er = max(1.0, H * 0.03)
        m[circle_mask(H, W, head_c[0] - ex, ey, er)] *= 0.25
        m[circle_mask(H, W, head_c[0] + ex, ey, er)] *= 0.25
        my = head_c[1] + head_r * 0.55
        m[capsule_mask(H, W, (head_c[0] - ex * 0.6, my), (head_c[0] + ex * 0.6, my), er * 0.55)] *= 0.45

    return blur(m, rounds=2)


def template_feature_from_mask(mask: np.ndarray) -> np.ndarray:
    x = mask.astype(np.float32)[None, None, :, :]
    x = norm(x)
    y = conv(x, EDGE_W, EDGE_B, stride=1, padding=2)
    y = np.maximum(y, 0.0)
    y = avgpool2(y)
    y = norm(y)[0]
    parts = []
    for a, b in PAIR_INDEX:
        p = np.maximum(y[a], 0.0) + np.maximum(y[b], 0.0)
        parts.append(p)
        parts.append(p.copy())
    w = np.stack(parts, axis=0).astype(np.float32)
    for c in range(w.shape[0]):
        w[c] -= w[c].mean()
    n = float(np.sqrt((w * w).sum()) + 1e-12)
    return (w / n).astype(np.float32)


def branch_specs() -> list[dict]:
    branches = []
    for k, thresh, branch_bias in [
        (8, 4.9, -1.0),
        (12, 4.7, -1.0),
        (16, 4.5, -0.9),
        (20, 4.3, -0.8),
        (24, 4.1, -0.7),
    ]:
        templates = []
        aspects = [0.86, 1.00, 1.16]
        for aspect in aspects:
            templates.append(("frontal", aspect, False, 0.0, 0.0))
            templates.append(("armsout", aspect, False, 0.0, 0.0))
            templates.append(("upper", aspect, False, 0.0, 0.02))
            templates.append(("close", aspect, False, 0.0, 0.02))
            templates.append(("seated", aspect, False, 0.0, 0.02))
            for mirror in (False, True):
                templates.append(("stride", aspect, mirror, 0.0, 0.0))
                templates.append(("side", aspect, mirror, 0.0, 0.0))
        if k >= 16:
            for shift in (-0.12, 0.12):
                templates.append(("upper", 1.0, False, shift, 0.02))
                templates.append(("close", 1.0, False, shift, 0.02))
                templates.append(("side", 1.0, shift < 0, shift, 0.00))
        if k <= 8:
            templates = [t for t in templates if t[0] not in {"close", "seated"}]
        if k == 24:
            templates += [
                ("close", 1.25, False, 0.0, 0.03),
                ("close", 1.25, False, -0.15, 0.03),
                ("close", 1.25, False, 0.15, 0.03),
            ]
        branches.append({"k": k, "threshold": thresh, "bias": branch_bias, "templates": templates})
    return branches


def make_branch_weights(k: int, templates: list[tuple[str, float, bool, float, float]], threshold: float, branch_bias: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orig = k * 2
    bank = []
    for kind, aspect, mirror, x_shift, y_shift in templates:
        mask = render_pose(kind, orig, aspect=aspect, mirror=mirror, x_shift=x_shift, y_shift=y_shift)
        bank.append(template_feature_from_mask(mask))
    W = np.stack(bank, axis=0).astype(np.float32)
    b = np.full((len(bank),), -float(threshold), np.float32)
    dW = np.zeros((2, len(bank)), np.float32)
    dW[1, :] = 1.0
    db = np.array([0.0, branch_bias], np.float32)
    return W, b, dW, db


def build_answer() -> tuple[dict, dict[str, np.ndarray]]:
    weights: dict[str, np.ndarray] = {}
    layers: list[dict] = []

    weights["gray.W"] = np.array([[[[0.299]], [[0.587]], [[0.114]]]], np.float32)
    weights["gray.b"] = np.zeros((1,), np.float32)
    layers.append({"name": "gray", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0})
    layers.append({"name": "gray_n", "op": "norm", "inputs": ["gray"]})

    weights["edge.W"] = EDGE_W.astype(np.float32)
    weights["edge.b"] = EDGE_B.astype(np.float32)
    layers.append({"name": "edge", "op": "conv", "inputs": ["gray_n"], "out": int(EDGE_W.shape[0]), "kernel": [5, 5], "stride": 1, "padding": 2})
    layers.append({"name": "edge_r", "op": "relu", "inputs": ["edge"]})
    layers.append({"name": "edge_p", "op": "avgpool", "inputs": ["edge_r"], "size": [2, 2], "stride": [2, 2]})
    layers.append({"name": "edge_n", "op": "norm", "inputs": ["edge_p"]})

    logit_inputs = []
    for idx, spec in enumerate(branch_specs()):
        name = f"t{idx}"
        W, b, dW, db = make_branch_weights(spec["k"], spec["templates"], spec["threshold"], spec["bias"])
        weights[name + ".W"] = W
        weights[name + ".b"] = b
        layers.append({"name": name, "op": "conv", "inputs": ["edge_n"], "out": int(W.shape[0]), "kernel": [spec["k"], spec["k"]], "stride": 1, "padding": 0})
        layers.append({"name": name + "_r", "op": "relu", "inputs": [name]})
        out_hw = 49 - spec["k"]
        layers.append({"name": name + "_m", "op": "maxpool", "inputs": [name + "_r"], "size": [out_hw, out_hw], "stride": [out_hw, out_hw]})
        weights[name + "_d.W"] = dW
        weights[name + "_d.b"] = db
        layers.append({"name": name + "_d", "op": "dense", "inputs": [name + "_m"], "out": 2})
        logit_inputs.append(name + "_d")

    layers.append({"name": "score", "op": "add", "inputs": logit_inputs})
    return {"layers": layers}, weights


def write_answer(out_dir: Path) -> None:
    net, weights = build_answer()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(out_dir / "weights.npz", **weights)


def main(argv: list[str]) -> None:
    out_dir = Path(argv[1]) if len(argv) > 1 else Path(".")
    write_answer(out_dir)


if __name__ == "__main__":
    main(sys.argv)
