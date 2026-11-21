#!/usr/bin/env python3
"""build.py NEW_FOLDER  -> writes NEW_FOLDER/network.json and NEW_FOLDER/weights.npz

Hand-designed network for 24-class ASL fingerspelling (28x28 greyscale, sign-language-MNIST style).
No training data is used anywhere.  Every number comes from this file:
  * geometric hand models (palm / fist ellipse, wrist, finger and thumb capsules, rings for C and O)
    typed from general knowledge of the ASL alphabet as seen from the front,
  * the network normalises the image, average-pools it to 14x14 and normalises again; a dense layer
    holds the normalised 14x14 silhouettes of the rendered hand models at many shifts / scales /
    rotations and in both polarities (hand lighter or darker than the background), so each output is a
    cosine similarity; the class score is the maximum over its variants (reshape + maxpool).
The renderer (section 1), the hand models (section 2) and a small interpreter with the scorer's
semantics (section 3) are followed by the classifier construction (section 4).  Section 2 also keeps
the models split into body and distinctive parts; only their union is used by the final network.
"""
import json, sys, os
import numpy as np

# ----------------------------------------------------------------------------------------------
# 1. Rendering of geometric hand models
# ----------------------------------------------------------------------------------------------
SIZE = 28
SS = 4  # supersampling factor for anti-aliased rendering


def _grid():
    n = SIZE * SS
    c = (np.arange(n) + 0.5) / SS
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return xx, yy


XX, YY = _grid()


def m_ellipse(cx, cy, rx, ry, ang=0.0):
    a = np.deg2rad(ang)
    dx, dy = XX - cx, YY - cy
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def m_capsule(x0, y0, x1, y1, r):
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return (XX - x0) ** 2 + (YY - y0) ** 2 <= r * r
    t = ((XX - x0) * dx + (YY - y0) * dy) / L2
    t = np.clip(t, 0.0, 1.0)
    px, py = x0 + t * dx, y0 + t * dy
    return (XX - px) ** 2 + (YY - py) ** 2 <= r * r


def m_ring(cx, cy, r_out, r_in, a0=None, a1=None):
    """annulus; optional arc from angle a0 to a1 (degrees, counter-clockwise on screen, 0 = +x, 90 = up)."""
    d2 = (XX - cx) ** 2 + (YY - cy) ** 2
    m = (d2 <= r_out * r_out) & (d2 >= r_in * r_in)
    if a0 is not None:
        ang = np.rad2deg(np.arctan2(-(YY - cy), XX - cx)) % 360.0
        a0m, a1m = a0 % 360.0, a1 % 360.0
        if a0m <= a1m:
            m &= (ang >= a0m) & (ang <= a1m)
        else:
            m &= (ang >= a0m) | (ang <= a1m)
    return m


def transform(parts, scale=1.0, dx=0.0, dy=0.0, rot=0.0, sx=1.0, cx=14.0, cy=14.0):
    """scale (and horizontal stretch sx) / rotate about the image centre, then shift."""
    a = np.deg2rad(rot)
    ca, sa = np.cos(a), np.sin(a)

    def pt(x, y):
        x, y = (x - cx) * sx, y - cy
        x, y = scale * (ca * x - sa * y), scale * (sa * x + ca * y)
        return x + cx + dx, y + cy + dy

    out = []
    for kind, args in parts:
        if kind in ("ellipse", "hole"):
            x, y, rx, ry, ang = args
            x, y = pt(x, y)
            out.append((kind, (x, y, rx * scale * (sx if abs(ang) < 45 else 1.0), ry * scale * (1.0 if abs(ang) < 45 else sx), ang + rot)))
        elif kind in ("capsule", "crease"):
            x0, y0, x1, y1, r = args
            x0, y0 = pt(x0, y0)
            x1, y1 = pt(x1, y1)
            out.append((kind, (x0, y0, x1, y1, r * scale)))
        elif kind == "ring":
            x, y, ro, ri, a0, a1 = args
            x, y = pt(x, y)
            out.append((kind, (x, y, ro * scale, ri * scale, None if a0 is None else a0 + rot, None if a1 is None else a1 + rot)))
        else:
            raise ValueError(kind)
    return out


def render(parts, skin, bg, crease_depth=0.22, return_mask=False):
    """Render a hand model to a 28x28 float image.  Solid parts build the hand mask, 'hole' cuts it out,
    'crease' paints darker lines inside the hand (skin folds / gaps between fingers are always darker)."""
    hand = np.zeros_like(XX, dtype=bool)
    holes = np.zeros_like(XX, dtype=bool)
    creases = np.zeros_like(XX, dtype=bool)
    for kind, args in parts:
        if kind == "ellipse":
            hand |= m_ellipse(*args)
        elif kind == "capsule":
            hand |= m_capsule(*args)
        elif kind == "ring":
            hand |= m_ring(*args)
        elif kind == "hole":
            holes |= m_ellipse(*args)
        elif kind == "crease":
            creases |= m_capsule(*args)
    hand &= ~holes
    img = np.where(hand, skin, bg).astype(np.float64)
    img = img - crease_depth * (creases & hand)
    img = img.reshape(SIZE, SS, SIZE, SS).mean(axis=(1, 3))
    if return_mask:
        return img, hand.reshape(SIZE, SS, SIZE, SS).mean(axis=(1, 3))
    return img


# ----------------------------------------------------------------------------------------------
# 2. Hand models, one per class (general knowledge of the ASL alphabet as seen from the front;
#    right hand, palm mostly toward the viewer, so the thumb is on the viewer's right).
#    Coordinates: x to the right, y down, in pixels of the 28x28 frame; wrist leaves through the bottom
#    (through the top for P and Q, which point down).
# ----------------------------------------------------------------------------------------------
def E(x, y, rx, ry, ang=0.0):
    return ("ellipse", (x, y, rx, ry, ang))


def C(x0, y0, x1, y1, r):
    return ("capsule", (x0, y0, x1, y1, r))


def K(x0, y0, x1, y1, r=0.55):
    return ("crease", (x0, y0, x1, y1, r))


def R(x, y, ro, ri, a0=None, a1=None):
    return ("ring", (x, y, ro, ri, a0, a1))


def H(x, y, rx, ry, ang=0.0):
    return ("hole", (x, y, rx, ry, ang))


def knuckles(xs, y0, y1):
    return [K(x, y0, x, y1) for x in xs]


FR = 1.5  # finger radius (fingers are about 3 px wide at this resolution)

# Each class: (BODY, PARTS).  BODY = palm / fist ellipse and the wrist (shared, undistinctive structure);
# PARTS = the distinctive elements of the sign: extended fingers, thumb, knuckle / finger-fold lines,
# the ring of C and O.  Knuckle and fold lines are drawn as creases (darker lines on the skin) of about
# 2 px width; extended fingers as capsules of about 3 px width.
KR = 0.9  # radius of knuckle / fold lines


def kn(xs, y0, y1):
    return [K(x, y0, x, y1, KR) for x in xs]


MODELS2 = {
    # A: fist, four knuckles along the top, thumb upright along the right side
    0: ([E(14, 15, 6.5, 7), C(13, 20, 12, 31, 5)],
        [C(20.5, 17, 20.5, 10, 1.8), K(19.0, 17, 19.0, 10.5, 0.7)] + kn([10.5, 13.5, 16.5], 8.5, 12.5)),
    # B: flat hand, four fingers up and together, thumb folded across the palm
    1: ([E(14, 17, 5.8, 6.5), C(14, 21, 14, 31, 4.5)],
        [C(9.5, 12, 9.5, 4, FR), C(12.5, 12, 12.5, 2.5, FR), C(15.5, 12, 15.5, 3, FR), C(18.5, 12, 18.5, 5.5, FR),
         K(11, 4, 11, 12, 0.7), K(14, 3, 14, 12, 0.7), K(17, 4, 17, 12, 0.7), K(8.5, 19, 13, 14.5, 0.8)]),
    # C: curved hand, opening to the viewer's right; fingers form the upper arc, thumb the lower arc
    2: ([C(9, 20, 5, 31, 4.5)],
        [R(13, 14, 9.5, 5.0, 40, 300), R(13, 14, 9.0, 5.5, 300, 330), K(16, 5.5, 20, 7.5, 0.7), K(19, 7.5, 21.5, 10.5, 0.7)]),
    # D: index finger straight up, other fingers curled touching the thumb (small ring)
    3: ([E(13, 17, 6, 6), C(13, 21, 12, 31, 4.5)],
        [H(15, 17, 1.6, 1.6), C(13.5, 12, 13.5, 2, FR), K(10, 13, 13, 11.5, 0.7)]),
    # E: fist with curled fingers, fingertips as a row of bumps at the top, thumb across underneath
    4: ([E(14, 15.5, 7, 6.5), C(14, 20, 14, 31, 5)],
        [K(9.5, 14, 19, 14, 0.8)] + kn([10, 13, 16, 19], 9, 13.5)),
    # F: thumb and index make a ring (lower left), the other three fingers up and spread
    5: ([E(14, 16.5, 5.5, 5.5), C(13, 21, 12, 31, 4.5)],
        [R(11, 16.5, 4.8, 2.3), C(14, 12, 13.5, 2, FR), C(16.5, 12, 18, 3, FR), C(18.5, 13, 22, 6, FR)]),
    # G: hand horizontal, index pointing to the viewer's right, thumb parallel under it
    6: ([E(9, 15, 6, 6), C(8, 18, 4, 31, 5)],
        [C(13, 12, 25.5, 11.5, FR), C(13, 15.5, 21, 15.5, FR), K(13, 13.7, 21, 13.5, 0.7)]),
    # H: like G with index and middle together pointing right
    7: ([E(9, 15, 6, 6), C(8, 18, 5, 31, 5)],
        [C(13, 11, 26.5, 10.5, FR), C(13, 14, 26.5, 13.5, FR), K(13, 12.5, 26, 12, 0.7)]),
    # I: fist with the little finger up on the viewer's left
    8: ([E(15, 16, 6.5, 6.5), C(15, 21, 16, 31, 5)],
        [C(9.8, 13, 9, 3, 1.3)] + kn([13, 16, 19], 10, 14)),
    # K: index up, middle finger angled up-right, thumb between them
    9: ([E(13, 17, 5.5, 5.5), C(13, 21, 12, 31, 4.5)],
        [C(13, 12, 12.5, 2, FR), C(15, 12, 21, 5, FR), C(14, 13, 17, 9, 1.3)]),
    # L: index up, thumb out to the viewer's right
    10: ([E(12, 17, 5.5, 5.5), C(12, 21, 11, 31, 4.5)],
         [C(14, 12, 14, 2, FR), C(16, 15, 26, 14, FR)]),
    # M: fist, three fingers folded down over the thumb (three finger columns), thumb tip peeking at the bottom
    11: ([E(14, 15, 7, 7), C(14, 21, 13, 31, 5)],
         [C(9.5, 20, 8.5, 22, 1.2)] + kn([10.5, 14, 17.5], 8.5, 17)),
    # N: fist, two fingers folded over the thumb
    12: ([E(14, 15.5, 6.5, 6.5), C(14, 21, 13, 31, 5)],
         [C(10.5, 20, 9.5, 22, 1.2)] + kn([12, 16], 9.5, 17)),
    # O: fingers and thumb form a round O, hole toward the right
    13: ([C(12, 21, 10, 31, 5)],
         [E(14, 15, 7.5, 7.5), H(16, 14.5, 2.6, 2.4)]),
    # P: like K pointing down; the arm enters from the top right, index points down-left, middle down
    14: ([E(15, 11, 6, 5.5), C(18, 8, 27, -4, 6)],
         [C(13, 14, 7, 24, FR), C(16, 15, 16.5, 24, FR), C(14, 14, 12, 20, 1.3)]),
    # Q: like G pointing down; arm from the top, index and thumb hanging down
    15: ([E(15, 10, 6, 5.5), C(16, 6, 19, -5, 6)],
         [C(12, 14, 10, 25, FR), C(17, 14, 16, 24, FR)]),
    # R: index and middle crossed, pointing up
    16: ([E(13, 17, 5.5, 5.5), C(13, 21, 12, 31, 4.5)],
         [C(13, 12, 14.5, 2, 1.3), C(14.5, 12, 12.5, 2.5, 1.3), K(13.8, 12, 13.8, 4, 0.6)]),
    # S: fist, knuckles on top, thumb across the front
    17: ([E(14, 15, 6.5, 7), C(14, 21, 14, 31, 5)],
         [K(9, 16.5, 19, 15.5, 0.9)] + kn([10.5, 13.5, 16.5], 8.5, 13)),
    # T: fist, index bent over the thumb with its knuckle pointing right, thumb tip up
    18: ([E(13, 15, 6, 6), C(13, 20, 13, 31, 5)],
         [C(14, 10.5, 19.5, 10.5, 1.6), C(12, 10, 12, 7, 1.4)]),
    # U: index and middle together pointing up
    19: ([E(13, 17, 5.5, 5.5), C(13, 21, 12, 31, 4.5)],
         [C(12.5, 12, 12.5, 2, FR), C(15.5, 12, 15.5, 2, FR), K(14, 3, 14, 12, 0.7)]),
    # V: index and middle up and spread
    20: ([E(13, 17, 5.5, 5.5), C(13, 21, 12, 31, 4.5)],
         [C(12, 12, 9, 2, FR), C(15, 12, 18, 2, FR)]),
    # W: index, middle and ring up and spread
    21: ([E(14, 17, 5.8, 5.5), C(14, 21, 13, 31, 4.5)],
         [C(11, 12, 8, 2, 1.4), C(14, 12, 14, 1, 1.4), C(17, 12, 20, 2, 1.4)]),
    # X: index finger hooked, pointing up with a bent tip
    22: ([E(13, 16, 6, 6), C(13, 21, 12, 31, 4.5)],
         [C(13, 11, 13, 5, 1.4), C(13, 5, 16.5, 7, 1.3)]),
    # Y: little finger up-left and thumb out right ("hang loose")
    23: ([E(14, 16, 6, 6), C(14, 21, 15, 31, 5)],
         [C(9, 12.5, 6, 3, 1.3), C(19, 14, 26, 9, FR)] + kn([12, 15, 18], 10, 14)),
}
WRIST_WIDEN = 1.0  # forearms are wide where they leave the frame


def _widen(parts):
    out = []
    for k, a in parts:
        if k == "capsule" and (a[3] >= SIZE or a[3] < 0):
            a = (a[0], a[1], a[2], a[3], a[4] + WRIST_WIDEN)
        out.append((k, a))
    return out


MODELS2 = {k: (_widen(b), p) for k, (b, p) in MODELS2.items()}
MODELS = {k: body + parts for k, (body, parts) in MODELS2.items()}


def parts_silhouette(parts, dilate=1.5):
    """Mask (28x28 in [0,1]) of the region around the distinctive parts, slightly dilated."""
    m = np.zeros_like(XX, dtype=bool)
    for kind, args in parts:
        if kind == "ellipse":
            x, y, rx, ry, ang = args
            m |= m_ellipse(x, y, rx + dilate, ry + dilate, ang)
        elif kind in ("capsule", "crease"):
            x0, y0, x1, y1, r = args
            m |= m_capsule(x0, y0, x1, y1, r + dilate)
        elif kind == "ring":
            x, y, ro, ri, a0, a1 = args
            m |= m_ring(x, y, ro + dilate, max(ri - dilate, 0.0), a0, a1)
        elif kind == "hole":
            x, y, rx, ry, ang = args
            m |= m_ellipse(x, y, rx + dilate, ry + dilate, ang)
    return m.reshape(SIZE, SS, SIZE, SS).mean(axis=(1, 3))


# ----------------------------------------------------------------------------------------------
# 3. A small numpy interpreter with exactly the scorer's semantics (single image, shape (C,H,W))
# ----------------------------------------------------------------------------------------------
NORM_EPS = 1e-5


def np_conv(x, W, b, stride, pad):
    Cc, Hh, Ww = x.shape
    O, _, k, _ = W.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    ho, wo = (Hh + 2 * pad - k) // stride + 1, (Ww + 2 * pad - k) // stride + 1
    out = np.zeros((O, ho, wo))
    for dy in range(k):
        for dx in range(k):
            patch = xp[:, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
            out += np.einsum("oc,chw->ohw", W[:, :, dy, dx], patch)
    return out + b[:, None, None]


def np_pool(x, size, stride, reduce):
    Cc, Hh, Ww = x.shape
    (sh, sw), (th, tw) = size, stride
    ho, wo = (Hh - sh) // th + 1, (Ww - sw) // tw + 1
    win = np.stack([x[:, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if reduce == "max" else win.mean(0)


def np_run(layers, weights, img, upto=None):
    """Run layers on one image (1,28,28); returns dict of all layer outputs."""
    vals = {"image": img}
    for L in layers:
        ins = [vals[i] for i in L["inputs"]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = np_conv(ins[0], weights[name + ".W"], weights[name + ".b"], L.get("stride", 1), L.get("padding", 0))
        elif op == "dense":
            vals[name] = weights[name + ".W"] @ ins[0].reshape(-1) + weights[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op in ("maxpool", "avgpool"):
            vals[name] = np_pool(ins[0], L["size"], L["stride"], op[:3])
        elif op == "add":
            vals[name] = sum(ins)
        elif op == "reshape":
            vals[name] = ins[0].reshape(tuple(L["shape"]))
        elif op == "norm":
            x = ins[0]
            mu = x.mean(axis=(1, 2), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(1, 2), keepdims=True) + NORM_EPS)
        if upto is not None and name == upto:
            break
    return vals


# ----------------------------------------------------------------------------------------------
# 4. Classifier: silhouette matching on the coarse intensity layout
#    norm(image) -> 2x2 average pooling (14x14) -> whole-vector normalisation -> dense layer whose rows are
#    the normalised 14x14 silhouettes of the rendered hand models (hand = 1, background = 0) at many
#    shifts / scales / rotations and in both polarities (hand lighter or darker than the background);
#    the class score is the maximum cosine similarity over its variants.
# ----------------------------------------------------------------------------------------------
SHIFTS = [(dx, dy) for dy in (-4, -2, 0, 2, 4) for dx in (-4, -2, 0, 2, 4)]
SCALES = [1.0, 1.15, 1.3, 1.45, 1.6]
ROTATIONS = [-10.0, 0.0, 10.0]
STRETCH = [1.0]
POOL = 2
CREASE = 0.0


def feature_layers():
    n = SIZE // POOL
    layers = [{"name": "n0", "op": "norm", "inputs": ["image"]},
              {"name": "p0", "op": "avgpool", "inputs": ["n0"], "size": [POOL, POOL], "stride": [POOL, POOL]},
              {"name": "flat", "op": "reshape", "inputs": ["p0"], "shape": [1, n * n, 1]},
              {"name": "n1", "op": "norm", "inputs": ["flat"]}]
    return layers, {}, "p0"


def variants():
    out = []
    for s in SCALES:
        for sx in STRETCH:
            for rot in ROTATIONS:
                for (dx, dy) in SHIFTS:
                    out.append(dict(scale=s, sx=sx, rot=rot, dx=dx, dy=dy))
    return out


def build(verbose=False):
    layers, weights, feat = feature_layers()
    VARS = variants()
    nv = len(VARS)
    T = []
    for cls in range(24):
        for V in VARS:
            img = render(transform(MODELS[cls], **V), 1.0, 0.0, CREASE)
            T.append(np_run(layers, weights, img[None], upto=feat)[feat].reshape(-1))
    T = np.array(T).reshape(24, nv, -1)
    T = T - T.mean(axis=2, keepdims=True)
    T = T / (np.linalg.norm(T, axis=2, keepdims=True) + 1e-9)
    D = T.shape[2]
    W = np.concatenate([T, -T], axis=1).reshape(24 * 2 * nv, D) / np.sqrt(D)   # both polarities
    layers.append({"name": "d", "op": "dense", "inputs": ["n1"], "out": int(24 * 2 * nv)})
    weights["d.W"] = W
    weights["d.b"] = np.zeros(24 * 2 * nv)
    layers.append({"name": "v1", "op": "reshape", "inputs": ["d"], "shape": [24, int(2 * nv), 1]})
    layers.append({"name": "p1", "op": "maxpool", "inputs": ["v1"], "size": [int(2 * nv), 1], "stride": [1, 1]})
    layers.append({"name": "out", "op": "reshape", "inputs": ["p1"], "shape": [24]})
    weights = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
    return {"layers": layers}, weights


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    net, weights = build()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(net, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **weights)
    print("wrote", out, {k: v.shape for k, v in weights.items()})
