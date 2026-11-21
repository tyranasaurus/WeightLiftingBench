"""Hand-designed classifier for 28x28 greyscale clothing images (Fashion-MNIST style).

Architecture:
    image (1, 28, 28)
    -> norm (whiten spatially)
    -> dense "feat" to 20 feature activations (2 per class):
           row 2k   = POSITIVE silhouette of class k (ink regions)
           row 2k+1 = ANTI silhouette of class k    (ink regions typical of
                      the confusers of k but NOT of k itself)
    -> relu
    -> dense "out" to 10 class scores:
           out[k] = feat[2k] - feat[2k+1]

Both template groups are drawn from simple geometric primitives (rectangles)
encoding general knowledge about how clothing/footwear/bags look. No training,
no data: every number comes from the code itself.
"""

import sys, os, json
import numpy as np


def _add(t, r0, r1, c0, c1, v):
    r0, r1 = max(0, r0), min(28, r1)
    c0, c1 = max(0, c0), min(28, c1)
    if r0 < r1 and c0 < c1:
        t[r0:r1, c0:c1] = np.maximum(t[r0:r1, c0:c1], v)


def make_P_N():
    """Return (P, N) each of shape (10, 28, 28), values in [0, 1]."""
    P = np.zeros((10, 28, 28), dtype=np.float32)
    N = np.zeros((10, 28, 28), dtype=np.float32)

    # --- 0: T-shirt ---
    p = P[0]
    _add(p, 4, 7, 6, 22, 0.55)
    _add(p, 6, 9, 4, 24, 0.55)
    _add(p, 7, 23, 9, 19, 0.95)
    n = N[0]
    _add(n, 4, 7, 10, 18, 0.80)    # dark round neck (NOT bright collar)
    _add(n, 5, 11, 0, 4, 0.75)     # NOT full-width sleeves (pullover/coat)
    _add(n, 5, 11, 24, 28, 0.75)
    _add(n, 24, 28, 0, 28, 0.55)   # NOT long bottom (dress) / shoe
    _add(n, 12, 23, 13, 15, 0.20)  # mild NOT dark center stripe (coat)
    _add(n, 0, 3, 0, 28, 0.20)

    # --- 1: trouser ---
    p = P[1]
    _add(p, 2, 5, 10, 18, 0.60)
    _add(p, 5, 26, 7, 12, 0.95)
    _add(p, 5, 26, 16, 21, 0.95)
    n = N[1]
    _add(n, 8, 26, 13, 15, 0.75)   # STRONG dark gap
    _add(n, 2, 7, 0, 10, 0.45)
    _add(n, 2, 7, 18, 28, 0.45)
    _add(n, 5, 11, 0, 6, 0.60)
    _add(n, 5, 11, 22, 28, 0.60)
    _add(n, 10, 24, 10, 18, 0.35)  # NOT solid torso across center (clothing)
    _add(n, 0, 2, 0, 28, 0.20)

    # --- 2: pullover ---
    p = P[2]
    _add(p, 5, 11, 0, 28, 0.75)
    _add(p, 4, 6, 1, 27, 0.55)
    _add(p, 11, 24, 6, 22, 0.90)
    _add(p, 12, 23, 10, 18, 0.98)  # SOLID center
    n = N[2]
    _add(n, 25, 28, 0, 28, 0.55)
    _add(n, 0, 4, 0, 28, 0.25)
    # anti: coat-like dark center stripe -> if visible, image is NOT a pullover
    # (we flip this logic: anti of pullover = dark center, meaning image shows coat pattern)
    # But our N represents "confuser ink regions". Coat has dark center = absence of ink,
    # so we can't detect it via a positive template. Instead, we rely on the center of
    # the pullover P being strong -> a dark center kills the pos score, lowering output.
    _add(n, 20, 26, 7, 12, 0.35)   # NOT trouser legs
    _add(n, 20, 26, 16, 21, 0.35)

    # --- 3: dress ---
    p = P[3]
    _add(p, 3, 6, 11, 17, 0.55)
    for r in range(6, 27):
        frac = (r - 6) / 20.0
        half = max(3, int(round(4 + frac * 8)))
        _add(p, r, r + 1, 14 - half, 14 + half, 0.85)
    _add(p, 22, 27, 7, 21, 0.45)
    n = N[3]
    _add(n, 5, 11, 0, 7, 0.70)
    _add(n, 5, 11, 21, 28, 0.70)
    _add(n, 0, 3, 0, 28, 0.30)
    _add(n, 12, 25, 13, 15, 0.20)
    _add(n, 20, 26, 0, 4, 0.25)    # NOT wide corners (shoes extend here)
    _add(n, 20, 26, 24, 28, 0.25)

    # --- 4: coat ---
    p = P[4]
    _add(p, 5, 11, 0, 28, 0.75)
    _add(p, 4, 6, 1, 27, 0.55)
    _add(p, 11, 24, 6, 13, 0.92)
    _add(p, 11, 24, 15, 22, 0.92)
    _add(p, 4, 8, 7, 12, 0.60)
    _add(p, 4, 8, 16, 21, 0.60)
    n = N[4]
    # The "solid center" of pullover is a confuser pattern for coat: if image has bright
    # center at cols 13-15 rows 11-24, it's NOT a coat (coat has dark center).
    _add(n, 11, 24, 13, 15, 0.85)  # bright center stripe ink = pullover-like, so anti for coat
    _add(n, 25, 28, 0, 28, 0.55)
    _add(n, 0, 3, 0, 28, 0.25)
    _add(n, 20, 26, 7, 12, 0.30)
    _add(n, 20, 26, 16, 21, 0.30)

    # --- 5: sandal ---
    p = P[5]
    _add(p, 23, 27, 3, 25, 0.85)
    _add(p, 14, 23, 6, 10, 0.55)
    _add(p, 14, 23, 18, 22, 0.55)
    _add(p, 18, 21, 10, 18, 0.30)
    _add(p, 20, 24, 3, 7, 0.50)
    n = N[5]
    _add(n, 0, 14, 0, 28, 0.60)    # STRONG NOT top (clothing)
    _add(n, 16, 23, 11, 17, 0.50)  # NOT solid shoe upper (sneaker/boot)
    _add(n, 13, 20, 7, 11, 0.35)   # NOT ankle shaft (boot)
    _add(n, 20, 27, 13, 17, 0.30)

    # --- 6: shirt ---
    p = P[6]
    _add(p, 4, 7, 5, 23, 0.55)
    _add(p, 6, 9, 3, 25, 0.55)
    _add(p, 7, 24, 9, 19, 0.75)
    _add(p, 3, 7, 10, 18, 0.80)    # BRIGHT collar
    _add(p, 9, 23, 12, 16, 0.90)   # BRIGHT button placket
    n = N[6]
    _add(n, 4, 6, 13, 15, 0.35)    # v-notch
    _add(n, 5, 11, 0, 3, 0.60)
    _add(n, 5, 11, 25, 28, 0.60)
    _add(n, 24, 28, 0, 28, 0.50)
    _add(n, 12, 24, 7, 9, 0.20)
    _add(n, 12, 24, 19, 21, 0.20)
    _add(n, 0, 3, 0, 28, 0.20)

    # --- 7: sneaker ---
    p = P[7]
    _add(p, 19, 25, 3, 25, 0.90)
    _add(p, 17, 22, 10, 25, 0.80)
    _add(p, 24, 27, 3, 25, 0.60)
    _add(p, 20, 24, 20, 26, 0.80)
    n = N[7]
    _add(n, 0, 17, 0, 28, 0.65)    # STRONG NOT top
    _add(n, 12, 18, 4, 11, 0.55)   # NOT ankle shaft (boot)
    _add(n, 13, 18, 6, 10, 0.40)

    # --- 8: bag ---
    p = P[8]
    _add(p, 11, 24, 8, 20, 0.85)
    _add(p, 3, 11, 9, 12, 0.70)
    _add(p, 3, 11, 16, 19, 0.70)
    _add(p, 3, 5, 10, 18, 0.65)
    n = N[8]
    _add(n, 5, 11, 13, 15, 0.70)   # HOLE in handle (strongly distinctive)
    _add(n, 5, 11, 0, 8, 0.75)     # NOT full-width sleeves
    _add(n, 5, 11, 20, 28, 0.75)
    _add(n, 4, 8, 3, 9, 0.45)
    _add(n, 4, 8, 19, 25, 0.45)
    _add(n, 24, 28, 0, 28, 0.60)   # NOT shoe bottom
    _add(n, 11, 24, 0, 7, 0.45)
    _add(n, 11, 24, 21, 28, 0.45)
    _add(n, 0, 3, 0, 28, 0.20)

    # --- 9: ankle boot ---
    p = P[9]
    _add(p, 15, 25, 7, 22, 0.85)
    _add(p, 12, 20, 6, 12, 0.95)   # STRONG shaft
    _add(p, 20, 25, 18, 25, 0.85)
    _add(p, 23, 26, 4, 24, 0.55)
    n = N[9]
    _add(n, 0, 12, 0, 28, 0.65)    # STRONG NOT top
    _add(n, 20, 26, 0, 5, 0.35)    # NOT sneaker-style extension
    _add(n, 14, 20, 15, 25, 0.20)

    return P, N


def _normalize(T):
    """Center spatially and L2-normalize each template to unit norm."""
    T_c = T - T.mean(axis=(1, 2), keepdims=True)
    norms = np.sqrt((T_c ** 2).sum(axis=(1, 2), keepdims=True))
    return T_c / (norms + 1e-8)


def main(out_dir):
    P, N = make_P_N()
    P_n = _normalize(P.astype(np.float32))
    N_n = _normalize(N.astype(np.float32))

    # Build feat dense: interleave rows 2k=P_n[k], rows 2k+1=N_n[k]
    W_feat = np.zeros((20, 784), dtype=np.float32)
    b_feat = np.zeros(20, dtype=np.float32)
    for k in range(10):
        W_feat[2 * k] = P_n[k].flatten()
        W_feat[2 * k + 1] = N_n[k].flatten()
        # Small negative bias on anti so ReLU triggers only for strong confuser matches.
        b_feat[2 * k] = 0.0
        b_feat[2 * k + 1] = -0.3

    # Build out dense: out[k] = feat[2k] - feat[2k+1]
    W_out = np.zeros((10, 20), dtype=np.float32)
    b_out = np.zeros(10, dtype=np.float32)
    for k in range(10):
        W_out[k, 2 * k] = 1.0
        W_out[k, 2 * k + 1] = -1.0

    net = {"layers": [
        {"name": "n", "op": "norm", "inputs": ["image"]},
        {"name": "feat", "op": "dense", "inputs": ["n"], "out": 20},
        {"name": "rfeat", "op": "relu", "inputs": ["feat"]},
        {"name": "out", "op": "dense", "inputs": ["rfeat"], "out": 10},
    ]}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, indent=2)
    np.savez(os.path.join(out_dir, "weights.npz"),
             **{"feat.W": W_feat, "feat.b": b_feat,
                "out.W": W_out, "out.b": b_out})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
