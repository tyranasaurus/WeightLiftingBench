"""Builds a hand-designed CNN for CIFAR-10.

Architecture
------------
image (3, 32, 32)
  -> conv c1 (16 feature maps, 3x3 kernel, stride 1, pad 1)
  -> relu
  -> avgpool 8x8 stride 8   ->  (16, 4, 4)
  -> reshape                ->  (256,)
  -> dense                  ->  (10,)

The 16 conv filters are hand-designed low-level detectors:
  0..2  raw R, G, B at the centre pixel
  3     blueness 2B - R - G  (sky, water)
  4     greenness 2G - R - B (grass, foliage)
  5     warmth   R + G - 2B  (earth, fur, skin)
  6     redness  2R - G - B  (red paint, red cabs)
  7     bright pixels     (brightness - 0.45)
  8     white  pixels     (brightness - 0.75)
  9     dark   pixels     (0.35 - brightness)
 10..13 Sobel horizontal and vertical edges, both signs
 14     centre-surround blob detector
 15     metallic / grey   (brightness - 0.55)

The dense layer's weights come from per-class rules that encode where and
what each class is expected to look like, based on general knowledge of the
classes (sky above airplanes and ships, grass below horses and deer, pure
green for frogs, dark small subject centred for birds, bright boxy shapes
for cars and trucks, warm furry subjects for cats and dogs).

A per-class bias then corrects for the expected ``baseline'' score each class
would get on a generic natural photo (with hand-chosen expected feature
activations) so that an average image gives roughly the same score to every
class and the discriminating features decide the winner.

No training data, class averages or any other fitted statistic is used:
every number in this file is written by hand from general knowledge.
"""

import json
import sys
from pathlib import Path
import numpy as np


# ------------------------------------------------------------------
# conv1: 16 hand-designed feature detectors
# ------------------------------------------------------------------

F_R, F_G, F_B   = 0, 1, 2
F_BLUE          = 3
F_GREEN         = 4
F_WARM          = 5
F_RED           = 6
F_BRIGHT        = 7
F_WHITE         = 8
F_DARK          = 9
F_VEDGE_P       = 10
F_VEDGE_N       = 11
F_HEDGE_P       = 12
F_HEDGE_N       = 13
F_BLOB          = 14
F_METAL         = 15
F               = 16


def build_conv1():
    W = np.zeros((F, 3, 3, 3), dtype=np.float32)
    b = np.zeros((F,),          dtype=np.float32)

    for i, ch in enumerate((0, 1, 2)):
        W[i, ch, 1, 1] = 1.0

    W[F_BLUE,  2, 1, 1] =  2.0; W[F_BLUE, 0, 1, 1] = -1.0; W[F_BLUE, 1, 1, 1] = -1.0
    W[F_GREEN, 1, 1, 1] =  2.0; W[F_GREEN, 0, 1, 1] = -1.0; W[F_GREEN, 2, 1, 1] = -1.0
    W[F_WARM,  0, 1, 1] =  1.0; W[F_WARM,  1, 1, 1] =  1.0; W[F_WARM,  2, 1, 1] = -2.0
    W[F_RED,   0, 1, 1] =  2.0; W[F_RED,   1, 1, 1] = -1.0; W[F_RED,   2, 1, 1] = -1.0

    for ch in (0, 1, 2):
        W[F_BRIGHT, ch, 1, 1] =  1.0 / 3.0
        W[F_WHITE,  ch, 1, 1] =  1.0 / 3.0
        W[F_DARK,   ch, 1, 1] = -1.0 / 3.0
        W[F_METAL,  ch, 1, 1] =  1.0 / 3.0
    b[F_BRIGHT] = -0.45
    b[F_WHITE]  = -0.75
    b[F_DARK]   =  0.35
    b[F_METAL]  = -0.55

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32) / 8.0
    sobel_y = sobel_x.T
    for ch in (0, 1, 2):
        W[F_VEDGE_P, ch] =  sobel_x / 3.0
        W[F_VEDGE_N, ch] = -sobel_x / 3.0
        W[F_HEDGE_P, ch] =  sobel_y / 3.0
        W[F_HEDGE_N, ch] = -sobel_y / 3.0

    # Centre-surround blob: centre weight +8/(3*8), surround weights -1/(3*8).
    for ch in (0, 1, 2):
        W[F_BLOB, ch, 1, 1] =  8.0 / 24.0
        for y in range(3):
            for x in range(3):
                if not (y == 1 and x == 1):
                    W[F_BLOB, ch, y, x] = -1.0 / 24.0

    return W.astype(np.float32), b.astype(np.float32)


# ------------------------------------------------------------------
# dense: hand-chosen per-class rules over (feature, region)
# ------------------------------------------------------------------

TOP     = (0,)
UPPER   = (0, 1)
MIDDLE  = (1, 2)
LOWER   = (2, 3)
BOTTOM  = (3,)
ALLY    = (0, 1, 2, 3)
LEFT    = (0, 1)
CENTER  = (1, 2)
RIGHT   = (2, 3)
ALLX    = (0, 1, 2, 3)
SIDES   = (0, 3)


def add(W, c, f, ys, xs, v):
    for y in ys:
        for x in xs:
            W[c, f, y, x] += v


def build_dense():
    W = np.zeros((10, F, 4, 4), dtype=np.float32)

    # -------- 0: airplane --------
    # Dominantly blue sky top to bottom, with a bright/white metallic plane
    # body in the middle rows.  Rarely green, warm or dark.
    add(W, 0, F_BLUE,   UPPER,  ALLX,   +1.8)
    add(W, 0, F_BLUE,   LOWER,  ALLX,   +0.9)
    add(W, 0, F_WHITE,  MIDDLE, CENTER, +1.4)
    add(W, 0, F_METAL,  MIDDLE, CENTER, +0.9)
    add(W, 0, F_BRIGHT, UPPER,  ALLX,   +0.2)
    add(W, 0, F_GREEN,  ALLY,   ALLX,   -2.2)
    add(W, 0, F_WARM,   ALLY,   ALLX,   -1.3)
    add(W, 0, F_RED,    ALLY,   ALLX,   -0.5)
    add(W, 0, F_DARK,   ALLY,   ALLX,   -0.7)

    # -------- 1: automobile --------
    # Metallic body in the middle rows, dark tyres at the bottom, little
    # green, boxy horizontal edges below the hood line.  Smaller than truck.
    add(W, 1, F_METAL,   MIDDLE, CENTER, +1.8)
    add(W, 1, F_BRIGHT,  MIDDLE, CENTER, +0.6)
    add(W, 1, F_DARK,    BOTTOM, CENTER, +1.1)
    add(W, 1, F_HEDGE_P, LOWER,  ALLX,   +0.7)
    add(W, 1, F_HEDGE_N, LOWER,  ALLX,   +0.7)
    add(W, 1, F_VEDGE_P, MIDDLE, ALLX,   +0.3)
    add(W, 1, F_VEDGE_N, MIDDLE, ALLX,   +0.3)
    add(W, 1, F_RED,     MIDDLE, CENTER, +0.3)
    add(W, 1, F_GREEN,   ALLY,   ALLX,   -1.8)
    add(W, 1, F_BLUE,    UPPER,  ALLX,   +0.3)
    add(W, 1, F_BLUE,    LOWER,  ALLX,   -0.6)
    add(W, 1, F_WARM,    ALLY,   ALLX,   -0.2)

    # -------- 2: bird --------
    # A small centred subject on sky or foliage, often dark against a light
    # or green background.  The blob detector is the main signal.
    add(W, 2, F_BLUE,    UPPER,  ALLX,   +0.9)
    add(W, 2, F_GREEN,   ALLY,   ALLX,   +0.7)
    add(W, 2, F_GREEN,   MIDDLE, SIDES,  +0.5)
    add(W, 2, F_BLOB,    MIDDLE, CENTER, +1.6)
    add(W, 2, F_DARK,    MIDDLE, CENTER, +1.0)
    add(W, 2, F_WARM,    MIDDLE, CENTER, +0.4)
    add(W, 2, F_VEDGE_P, MIDDLE, CENTER, +0.4)
    add(W, 2, F_VEDGE_N, MIDDLE, CENTER, +0.4)
    add(W, 2, F_HEDGE_P, MIDDLE, CENTER, +0.3)
    add(W, 2, F_HEDGE_N, MIDDLE, CENTER, +0.3)
    add(W, 2, F_METAL,   ALLY,   ALLX,   -0.5)

    # -------- 3: cat --------
    # Centred furry animal in warm tones.  Cats are typically indoors or
    # close-up, with little pure green grass and no blue sky dominating.
    add(W, 3, F_WARM,    MIDDLE, CENTER, +1.1)
    add(W, 3, F_BRIGHT,  MIDDLE, CENTER, +0.4)
    add(W, 3, F_DARK,    MIDDLE, CENTER, +0.6)
    add(W, 3, F_WHITE,   MIDDLE, CENTER, +0.5)
    add(W, 3, F_RED,     MIDDLE, CENTER, +0.3)
    add(W, 3, F_VEDGE_P, MIDDLE, CENTER, +0.3)  # fur texture
    add(W, 3, F_VEDGE_N, MIDDLE, CENTER, +0.3)
    add(W, 3, F_HEDGE_P, MIDDLE, CENTER, +0.3)
    add(W, 3, F_HEDGE_N, MIDDLE, CENTER, +0.3)
    add(W, 3, F_BLOB,    MIDDLE, CENTER, +0.4)
    add(W, 3, F_BLUE,    UPPER,  ALLX,   -1.4)
    add(W, 3, F_GREEN,   ALLY,   ALLX,   -1.0)
    add(W, 3, F_METAL,   ALLY,   ALLX,   -0.5)

    # -------- 4: deer --------
    # Brown-tan body on a green forest/grass background, thin legs below.
    add(W, 4, F_WARM,    MIDDLE, CENTER, +1.3)
    add(W, 4, F_GREEN,   LOWER,  ALLX,   +1.3)
    add(W, 4, F_GREEN,   UPPER,  ALLX,   +0.5)
    add(W, 4, F_GREEN,   MIDDLE, SIDES,  +0.6)
    add(W, 4, F_VEDGE_P, LOWER,  CENTER, +0.4)  # thin legs
    add(W, 4, F_VEDGE_N, LOWER,  CENTER, +0.4)
    add(W, 4, F_RED,     MIDDLE, CENTER, +0.2)
    add(W, 4, F_BLUE,    UPPER,  ALLX,   -1.5)
    add(W, 4, F_METAL,   ALLY,   ALLX,   -0.5)
    add(W, 4, F_WHITE,   ALLY,   ALLX,   -0.4)
    add(W, 4, F_BRIGHT,  UPPER,  ALLX,   -0.3)

    # -------- 5: dog --------
    # Centred warm or dark furry animal, usually standing on grass with the
    # body extending into the lower rows.  Rarely blue sky.
    add(W, 5, F_WARM,    MIDDLE, CENTER, +1.5)
    add(W, 5, F_WARM,    LOWER,  CENTER, +0.4)
    add(W, 5, F_BRIGHT,  MIDDLE, CENTER, +0.4)
    add(W, 5, F_DARK,    MIDDLE, CENTER, +0.8)
    add(W, 5, F_WHITE,   MIDDLE, CENTER, +0.4)
    add(W, 5, F_BLOB,    MIDDLE, CENTER, +0.5)
    add(W, 5, F_GREEN,   BOTTOM, ALLX,   +1.0)  # grass below
    add(W, 5, F_RED,     MIDDLE, CENTER, +0.3)
    add(W, 5, F_VEDGE_P, MIDDLE, CENTER, +0.3)
    add(W, 5, F_VEDGE_N, MIDDLE, CENTER, +0.3)
    add(W, 5, F_HEDGE_P, MIDDLE, CENTER, +0.3)
    add(W, 5, F_HEDGE_N, MIDDLE, CENTER, +0.3)
    add(W, 5, F_BLUE,    UPPER,  ALLX,   -1.2)
    add(W, 5, F_GREEN,   UPPER,  ALLX,   -0.5)
    add(W, 5, F_METAL,   ALLY,   ALLX,   -0.5)

    # -------- 6: frog --------
    # Dominantly green, almost never blue, often dark green tone.
    add(W, 6, F_GREEN,   ALLY,   ALLX,   +1.6)
    add(W, 6, F_GREEN,   MIDDLE, CENTER, +0.6)
    add(W, 6, F_BLUE,    ALLY,   ALLX,   -1.8)
    add(W, 6, F_WARM,    ALLY,   ALLX,   -0.9)
    add(W, 6, F_WHITE,   ALLY,   ALLX,   -1.1)
    add(W, 6, F_METAL,   ALLY,   ALLX,   -1.1)
    add(W, 6, F_RED,     ALLY,   ALLX,   -0.6)
    add(W, 6, F_BRIGHT,  UPPER,  ALLX,   -0.5)

    # -------- 7: horse --------
    # Large brown subject across the middle, grass below, strong body line.
    add(W, 7, F_WARM,    MIDDLE, CENTER, +1.4)
    add(W, 7, F_WARM,    LOWER,  CENTER, +0.5)
    add(W, 7, F_GREEN,   LOWER,  ALLX,   +1.0)
    add(W, 7, F_GREEN,   UPPER,  ALLX,   +0.3)
    add(W, 7, F_HEDGE_P, MIDDLE, ALLX,   +0.4)
    add(W, 7, F_HEDGE_N, MIDDLE, ALLX,   +0.4)
    add(W, 7, F_VEDGE_P, LOWER,  CENTER, +0.3)
    add(W, 7, F_VEDGE_N, LOWER,  CENTER, +0.3)
    add(W, 7, F_BLUE,    UPPER,  ALLX,   -1.2)
    add(W, 7, F_METAL,   ALLY,   ALLX,   -0.6)
    add(W, 7, F_WHITE,   ALLY,   ALLX,   -0.4)

    # -------- 8: ship --------
    # Blue sky above and blue/grey water below, long horizontal waterline
    # and a bright/grey hull in the middle rows.
    add(W, 8, F_BLUE,    UPPER,  ALLX,   +2.0)
    add(W, 8, F_BLUE,    LOWER,  ALLX,   +2.0)
    add(W, 8, F_METAL,   MIDDLE, CENTER, +0.7)
    add(W, 8, F_WHITE,   MIDDLE, CENTER, +0.5)
    add(W, 8, F_HEDGE_P, MIDDLE, ALLX,   +0.5)
    add(W, 8, F_HEDGE_N, MIDDLE, ALLX,   +0.5)
    add(W, 8, F_GREEN,   ALLY,   ALLX,   -2.0)
    add(W, 8, F_WARM,    ALLY,   ALLX,   -1.4)
    add(W, 8, F_RED,     ALLY,   ALLX,   -0.4)

    # -------- 9: truck --------
    # Big boxy vehicle, bright/white or red cab on top with strong horizontal
    # roof edge and vertical side edges, dark under-chassis below.
    add(W, 9, F_METAL,   UPPER,  CENTER, +1.0)
    add(W, 9, F_METAL,   MIDDLE, CENTER, +0.7)
    add(W, 9, F_WHITE,   UPPER,  CENTER, +0.8)
    add(W, 9, F_BRIGHT,  MIDDLE, CENTER, +0.4)
    add(W, 9, F_RED,     MIDDLE, CENTER, +0.5)
    add(W, 9, F_DARK,    BOTTOM, CENTER, +0.9)
    add(W, 9, F_HEDGE_P, UPPER,  ALLX,   +0.6)
    add(W, 9, F_HEDGE_N, UPPER,  ALLX,   +0.6)
    add(W, 9, F_VEDGE_P, MIDDLE, ALLX,   +0.5)
    add(W, 9, F_VEDGE_N, MIDDLE, ALLX,   +0.5)
    add(W, 9, F_GREEN,   ALLY,   ALLX,   -1.8)
    add(W, 9, F_BLUE,    UPPER,  ALLX,   -0.4)
    add(W, 9, F_WARM,    ALLY,   ALLX,   -0.3)

    # -------- per-class bias so an average natural image scores ~ 0 --------
    # Hand-chosen expected activations on a typical natural photograph (not
    # measured on training images).  The bias then equals  - W[c] . E_feat
    # so that every class starts from zero on an "average" photo and the
    # discriminating features decide which class wins.
    E = np.zeros(F, dtype=np.float32)
    E[F_R]       = 0.50
    E[F_G]       = 0.48
    E[F_B]       = 0.46
    E[F_BLUE]    = 0.08
    E[F_GREEN]   = 0.09
    E[F_WARM]    = 0.11
    E[F_RED]     = 0.05
    E[F_BRIGHT]  = 0.10
    E[F_WHITE]   = 0.04
    E[F_DARK]    = 0.07
    E[F_VEDGE_P] = 0.03
    E[F_VEDGE_N] = 0.03
    E[F_HEDGE_P] = 0.03
    E[F_HEDGE_N] = 0.03
    E[F_BLOB]    = 0.04
    E[F_METAL]   = 0.04

    baseline = np.einsum('cfyx,f->c', W, E)
    b = -baseline.astype(np.float32)

    return W.reshape(10, F * 4 * 4).astype(np.float32), b


# ------------------------------------------------------------------
# network.json / weights.npz
# ------------------------------------------------------------------

def build_network_json():
    return {
        "layers": [
            {"name": "c1",    "op": "conv",    "inputs": ["image"],
             "out": F, "kernel": [3, 3], "stride": 1, "padding": 1},
            {"name": "r1",    "op": "relu",    "inputs": ["c1"]},
            {"name": "p1",    "op": "avgpool", "inputs": ["r1"],
             "size": [8, 8], "stride": [8, 8]},
            {"name": "flat",  "op": "reshape", "inputs": ["p1"],
             "shape": [F * 4 * 4]},
            {"name": "dense", "op": "dense",   "inputs": ["flat"], "out": 10},
        ]
    }


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cW, cB = build_conv1()
    dW, dB = build_dense()
    np.savez(out / "weights.npz",
             **{"c1.W": cW, "c1.b": cB, "dense.W": dW, "dense.b": dB})
    (out / "network.json").write_text(json.dumps(build_network_json(), indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
