"""Data-free build: writes network.json and weights.npz from code alone.

Architecture:
  image (1,28,28)
    -> norm                                          (1,28,28)
    -> conv (24 out, kernel 20x20, pad 0, stride 1)  (24,9,9)
    -> maxpool 9x9 stride 9                          (24,1,1)
    -> reshape                                       (24,)

The 24 convolution filters are hand-designed 20x20 templates for each ASL
letter sign, sparse by construction: each template marks only the positions
that are discriminative for that class. Every template cell is one of:
    -1  the letter's hand is expected to lie here (darker than the mean in the
        normalised image, which is how the data almost always looks)
    +1  this position is NOT a hand position for this letter, and it IS a hand
        position for one or more competing letters (so "bright here" is helpful
        evidence for this letter)
     0  don't care — no cell forces the template to disagree with another class
After construction each template is normalised using only its own mean and L2.

No training data and no cross-template statistic is read: every number is set
either by my knowledge of the ASL hand alphabet or by a geometric drawing rule.
"""
import json
import sys
from pathlib import Path

import numpy as np

K = 20                              # template / kernel size
POOL = 28 - K + 1                   # 9 (maxpool size for translation search)
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"


# -------- drawing helpers (overwrite cells with the given value) --------

def _disk(img, cy, cx, r, v):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = v


def _ellipse(img, cy, cx, ay, ax, v):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    img[((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2 <= 1] = v


def _rect(img, y0, y1, x0, x1, v):
    H, W = img.shape
    img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = v


def _line(img, y0, x0, y1, x1, thickness, v):
    H, W = img.shape
    n = int(max(abs(y1 - y0), abs(x1 - x0))) * 3 + 2
    ys = np.linspace(y0, y1, n)
    xs = np.linspace(x0, x1, n)
    yy, xx = np.ogrid[:H, :W]
    for y, x in zip(ys, xs):
        img[(yy - y) ** 2 + (xx - x) ** 2 <= thickness ** 2] = v


def _ring(img, cy, cx, r_out, r_in, v, cut_side=None, cut_thr=None):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    m = (r2 <= r_out ** 2) & (r2 >= r_in ** 2)
    if cut_side == 'right':
        m = m & ~((xx > cut_thr) & (np.abs(yy - cy) < r_out * 0.7))
    img[m] = v


def make_templates():
    """Return 24 (K, K) float32 templates. See module docstring for the +-1/0 convention.

    Within 20x20, the hand sits roughly centred: palm around row 11, col 10;
    fingers reach up to rows 1-10 for upright signs and to the right (cols 13-19)
    for sideways ones. Wrist enters from the bottom between cols 8 and 12."""
    T = []

    # Define reusable positions:
    PCY, PCX = 11, 10        # palm centre
    WY0 = 15                 # wrist starts here
    WX0, WX1 = 8, 12         # wrist column band

    # ---- 0: A — closed fist, thumb along the side; NO fingers up and NO side extensions.
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, PCY, PCX, 5.0, -1)             # fist mass
    _rect(t, WY0, K, WX0, WX1, -1)          # wrist
    _rect(t, 0, 6, 5, 15, +1)               # strong anti: no fingers up
    _rect(t, 7, 15, 0, 4, +1)               # anti: no sideways on the left
    _rect(t, 7, 15, 16, K, +1)              # anti: no sideways on the right
    T.append(t)

    # ---- 1: B — four fingers extended up, thumb across the palm (TALL mass centrally)
    t = np.zeros((K, K), dtype=np.float32)
    _rect(t, 1, 11, 5, 16, -1)              # fingers filling the top broadly
    _disk(t, 13, 10, 4.5, -1)               # palm
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 7, 15, 0, 4, +1)               # anti sideways left
    _rect(t, 7, 15, 16, K, +1)              # anti sideways right
    T.append(t)

    # ---- 2: C — open-C arc (opening to the right)
    t = np.zeros((K, K), dtype=np.float32)
    _ring(t, 10, 9, 8, 4, -1, cut_side='right', cut_thr=12)
    _rect(t, 7, 14, 13, 18, +1)             # strong anti: opening must be bg
    _rect(t, 7, 14, 7, 11, +1)              # anti: centre of the ring is bg
    T.append(t)

    # ---- 3: D — one thin finger straight up, round fist below
    t = np.zeros((K, K), dtype=np.float32)
    _rect(t, 1, 13, 9, 12, -1)              # index finger up
    _disk(t, 13, 10, 4.5, -1)               # palm
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 1, 10, 4, 8, +1)               # anti: no extra fingers at upper left
    _rect(t, 1, 10, 13, 17, +1)             # anti: no extra fingers at upper right
    _rect(t, 7, 15, 0, 4, +1)
    _rect(t, 7, 15, 16, K, +1)
    T.append(t)

    # ---- 4: E — fist with knuckles visible (small bumps across the top of the fist)
    t = np.zeros((K, K), dtype=np.float32)
    _ellipse(t, 10, 10, 6, 5.5, -1)
    _rect(t, WY0, K, WX0, WX1, -1)
    _disk(t, 5, 7, 1.2, -1)                 # knuckle bumps
    _disk(t, 5, 10, 1.2, -1)
    _disk(t, 5, 13, 1.2, -1)
    _rect(t, 0, 3, 5, 15, +1)               # anti fingers up above knuckles
    T.append(t)

    # ---- 5: F — three fingers up + a pinched circle to one side
    t = np.zeros((K, K), dtype=np.float32)
    _rect(t, 1, 11, 7, 9, -1)
    _rect(t, 1, 11, 10, 12, -1)
    _rect(t, 1, 11, 13, 15, -1)
    _disk(t, 13, 11, 4, -1)
    _ring(t, 12, 5, 3.0, 1.2, -1)           # pinch circle at lower left
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 7, 15, 16, K, +1)              # anti sideways right
    T.append(t)

    # ---- 6: G — sideways fist on the left, one index finger extending right
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 10, 6, 4.5, -1)                # fist
    _rect(t, 9, 12, 6, 19, -1)              # index pointing right
    _rect(t, 8, 12, 0, 4, -1)               # arm from the left edge
    _rect(t, 0, 5, 0, K, +1)                # anti top
    _rect(t, 15, K, 0, K, +1)               # anti bottom (hand is horizontal)
    T.append(t)

    # ---- 7: H — sideways fist, two fingers pointing right
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 10, 6, 4.5, -1)
    _rect(t, 7, 10, 6, 19, -1)
    _rect(t, 11, 14, 6, 19, -1)
    _rect(t, 8, 12, 0, 4, -1)
    _rect(t, 0, 4, 0, K, +1)
    _rect(t, 16, K, 0, K, +1)
    T.append(t)

    # ---- 8: I — a single thin pinky extended up on one side of the fist
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 13, 10, 4.5, -1)
    _rect(t, 1, 14, 6, 9, -1)               # pinky on one side
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 1, 10, 10, 17, +1)             # anti: no other fingers up
    T.append(t)

    # ---- 9: K — index up + middle angled to the side; thumb between
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 13, 10, 4.5, -1)
    _rect(t, 1, 13, 8, 10, -1)              # index up
    _line(t, 10, 10, 1, 16, 1.2, -1)        # middle angled up-right
    _disk(t, 9, 12, 1.5, -1)                # thumb bump
    _rect(t, WY0, K, WX0, WX1, -1)
    T.append(t)

    # ---- 10: L — index up and thumb to the side (L shape)
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 13, 10, 4.5, -1)
    _rect(t, 1, 13, 7, 10, -1)              # index up
    _rect(t, 12, 15, 10, 19, -1)            # thumb to the right
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 1, 10, 11, 17, +1)             # anti: no other fingers up near the index
    T.append(t)

    # ---- 11: M — fist with three fingers folded over thumb (three small bumps on top)
    t = np.zeros((K, K), dtype=np.float32)
    _ellipse(t, 10, 10, 6, 6, -1)
    _rect(t, WY0, K, WX0, WX1, -1)
    _disk(t, 4, 6, 1.3, -1)
    _disk(t, 4, 10, 1.3, -1)
    _disk(t, 4, 14, 1.3, -1)
    _rect(t, 0, 3, 5, 15, +1)
    T.append(t)

    # ---- 12: N — fist with two fingers folded over thumb (two bumps)
    t = np.zeros((K, K), dtype=np.float32)
    _ellipse(t, 10, 10, 6, 6, -1)
    _rect(t, WY0, K, WX0, WX1, -1)
    _disk(t, 4, 8, 1.3, -1)
    _disk(t, 4, 12, 1.3, -1)
    _rect(t, 0, 3, 5, 15, +1)
    T.append(t)

    # ---- 13: O — closed ring; centre is strongly background
    t = np.zeros((K, K), dtype=np.float32)
    _ring(t, 10, 10, 7, 4, -1)
    _disk(t, 10, 10, 2.5, +1)               # strong bg at centre
    _rect(t, WY0, K, WX0, WX1, -1)          # wrist stub
    T.append(t)

    # ---- 14: P — downward-pointing K: fist at top, fingers point down
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 6, 10, 4.5, -1)                # fist on top
    _rect(t, 6, 18, 8, 10, -1)              # index pointing down
    _line(t, 9, 10, 18, 16, 1.2, -1)        # middle angled down-right
    _disk(t, 10, 12, 1.5, -1)               # thumb between
    _rect(t, 0, 5, WX0, WX1, -1)            # wrist from top
    _rect(t, 17, K, 0, 6, +1)
    T.append(t)

    # ---- 15: Q — downward G: fist at top, index + thumb point down
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 6, 10, 4.5, -1)
    _rect(t, 6, 18, 8, 10, -1)              # index down
    _rect(t, 6, 15, 11, 13, -1)             # thumb down (shorter)
    _rect(t, 0, 5, WX0, WX1, -1)
    _rect(t, 0, 5, 0, 5, +1)                # anti sideways top
    _rect(t, 0, 5, 15, K, +1)
    T.append(t)

    # ---- 16: R — two tight fingers up (crossed, so tighter than V)
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 13, 10, 4.5, -1)
    _rect(t, 1, 13, 8, 12, -1)              # tight pair of fingers
    _line(t, 1, 8, 10, 11, 1.0, -1)         # one cross
    _line(t, 1, 11, 10, 8, 1.0, -1)         # the other
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 1, 10, 13, 17, +1)             # anti third+ fingers
    _rect(t, 1, 10, 3, 7, +1)
    T.append(t)

    # ---- 17: S — fist with thumb across the front (short horizontal mass mid-front)
    t = np.zeros((K, K), dtype=np.float32)
    _ellipse(t, 10, 10, 6, 6, -1)
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 7, 10, 4, 11, -1)              # thumb across the front top
    _rect(t, 0, 3, 5, 15, +1)
    T.append(t)

    # ---- 18: T — fist with thumb between index and middle (small bump near top)
    t = np.zeros((K, K), dtype=np.float32)
    _ellipse(t, 10, 10, 6, 6, -1)
    _rect(t, WY0, K, WX0, WX1, -1)
    _disk(t, 6, 10, 1.7, -1)                # the thumb tip
    _rect(t, 0, 3, 5, 15, +1)
    T.append(t)

    # ---- 19: U — two fingers up, together (one wide column)
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 13, 10, 4.5, -1)
    _rect(t, 1, 13, 8, 13, -1)              # two fingers side by side
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 1, 10, 3, 7, +1)                # anti other fingers
    _rect(t, 1, 10, 14, 18, +1)
    T.append(t)

    # ---- 20: V — two fingers up, spread (peace sign)
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 13, 10, 4.5, -1)
    _line(t, 1, 6, 11, 9, 1.0, -1)          # left finger
    _line(t, 1, 14, 11, 11, 1.0, -1)        # right finger
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 1, 8, 9, 12, +1)               # anti middle: between the fingers must be bg
    T.append(t)

    # ---- 21: W — three fingers up, spread apart
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 13, 10, 4.5, -1)
    _rect(t, 1, 13, 5, 7, -1)
    _rect(t, 1, 13, 9, 11, -1)
    _rect(t, 1, 13, 13, 15, -1)
    _rect(t, WY0, K, WX0, WX1, -1)
    T.append(t)

    # ---- 22: X — a short bent/hooked index finger (short stub, not long)
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 13, 10, 4.5, -1)
    _rect(t, 6, 11, 8, 10, -1)              # short stub
    _disk(t, 6, 9, 1.4, -1)                 # hooked tip
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 0, 5, 5, 15, +1)               # anti a full-length finger
    T.append(t)

    # ---- 23: Y — pinky out one way, thumb out the other (hang-loose)
    t = np.zeros((K, K), dtype=np.float32)
    _disk(t, 13, 10, 4.5, -1)
    _rect(t, 7, 14, 3, 6, -1)               # pinky extending up-left
    _rect(t, 12, 16, 11, 19, -1)            # thumb extending right
    _rect(t, WY0, K, WX0, WX1, -1)
    _rect(t, 0, 6, 7, 13, +1)                # anti: no finger up at centre
    T.append(t)

    return T


def normalise(v):
    v = v.astype(np.float64)
    v = v - v.mean()
    n = float(np.sqrt((v * v).sum()))
    if n > 1e-9:
        v = v / n
    return v.astype(np.float32)


def network_json():
    return {
        "layers": [
            {"name": "norm_in", "op": "norm", "inputs": ["image"]},
            {"name": "conv1", "op": "conv", "inputs": ["norm_in"],
             "out": 24, "kernel": [K, K], "stride": 1, "padding": 0},
            {"name": "pool1", "op": "maxpool", "inputs": ["conv1"],
             "size": [POOL, POOL], "stride": [POOL, POOL]},
            {"name": "scores", "op": "reshape", "inputs": ["pool1"], "shape": [24]},
        ]
    }


def build_weights():
    templates = make_templates()
    W = np.zeros((24, 1, K, K), dtype=np.float32)
    b = np.zeros((24,), dtype=np.float32)
    for k, t in enumerate(templates):
        W[k, 0] = normalise(t)
    return {"conv1.W": W, "conv1.b": b}


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "network.json").write_text(json.dumps(network_json()))
    np.savez(str(out_dir / "weights.npz"), **build_weights())


if __name__ == "__main__":
    main()
