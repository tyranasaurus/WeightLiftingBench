"""Hand-designed CIFAR-10 network.

Architecture:
  conv1   (1x1, 3 -> 12 colour-feature channels, threshold in bias)
  relu
  avgpool (4, 4) stride (4, 4)   -> (12, 8, 8)
  dense   (768 -> 10)

Colour channels use a ReLU on linear combinations of RGB with carefully-chosen
biases, so each one fires only for pixels that strongly show that colour.
avgpool over 4x4 blocks then gives the fraction of such pixels per region.

The dense layer is a hand-designed template per class.  All templates refer to
the same four geometric regions:

  CENTRE : y 2..5, x 2..5   -- the object itself
  TOP    : y 0..1           -- sky area
  BOTTOM : y 6..7           -- ground / water
  SIDES  : y 2..5, x in {0, 1, 6, 7} -- left/right background

Per-class design philosophy:
  * Each class has at most 25-30 as a peak weight on its signature feature.
  * Classes that need a background cue (deer/horse/ship/frog) rely on BOTTOM
    or TOP penalties/bonuses so they only win when that cue is present.
  * Classes without strong background cues (cat, dog, auto, truck) differ in
    which centre feature wins (saturated / warm / white / dark).
  * A small per-class bias and shared penalties for cross-category features
    keep baseline scores comparable on a neutral (grey) image.

All numbers come from general knowledge of the class visuals; no training
data is read; no fitting is performed.
"""
import json
import os
import sys

import numpy as np


ARCH = {
    "layers": [
        {"name": "colors", "op": "conv", "inputs": ["image"], "out": 12,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "colors_relu", "op": "relu", "inputs": ["colors"]},
        {"name": "pool", "op": "avgpool", "inputs": ["colors_relu"],
         "size": [4, 4], "stride": [4, 4]},
        {"name": "fc", "op": "dense", "inputs": ["pool"], "out": 10},
    ]
}

# -------------- 12 colour channels --------------
R, G, B, BR_MID, BLUE, GREEN, RED, PRED, YEL, DARK, WHITE, DBLUE = range(12)
NF = 12
H = Wg = 8


def build_colors():
    cw = np.zeros((NF, 3, 1, 1), dtype=np.float32)
    cb = np.zeros(NF, dtype=np.float32)

    cw[R, 0, 0, 0] = 1.0
    cw[G, 1, 0, 0] = 1.0
    cw[B, 2, 0, 0] = 1.0

    cw[BR_MID, :, 0, 0] = 1.0 / 3.0; cb[BR_MID] = -0.40

    cw[BLUE, 0, 0, 0] = -0.5; cw[BLUE, 1, 0, 0] = -0.5; cw[BLUE, 2, 0, 0] = 1.0
    cb[BLUE] = -0.04
    cw[GREEN, 0, 0, 0] = -0.5; cw[GREEN, 1, 0, 0] = 1.0; cw[GREEN, 2, 0, 0] = -0.5
    cb[GREEN] = -0.03
    cw[RED, 0, 0, 0] = 1.0; cw[RED, 1, 0, 0] = -0.5; cw[RED, 2, 0, 0] = -0.5
    cb[RED] = -0.04
    cw[PRED, 0, 0, 0] = 1.0; cw[PRED, 1, 0, 0] = -1.0; cw[PRED, 2, 0, 0] = 0.0
    cb[PRED] = -0.15
    cw[YEL, 0, 0, 0] = 0.5; cw[YEL, 1, 0, 0] = 0.5; cw[YEL, 2, 0, 0] = -1.0
    cb[YEL] = -0.05
    cw[DARK, :, 0, 0] = -1.0 / 3.0; cb[DARK] = 0.30
    cw[WHITE, :, 0, 0] = 1.0 / 3.0; cb[WHITE] = -0.70
    cw[DBLUE, 0, 0, 0] = -1.0; cw[DBLUE, 1, 0, 0] = -1.0; cw[DBLUE, 2, 0, 0] = 1.5
    cb[DBLUE] = -0.05

    return cw, cb


CENTRE = [(y, x) for y in range(2, 6) for x in range(2, 6)]
TOP    = [(y, x) for y in range(0, 2) for x in range(0, 8)]
BOTTOM = [(y, x) for y in range(6, 8) for x in range(0, 8)]
SIDES  = [(y, x) for y in range(2, 6) for x in (0, 1, 6, 7)]
ALL    = [(y, x) for y in range(0, 8) for x in range(0, 8)]


def add(T, cls, feat, cells, value):
    for y, x in cells:
        T[cls, feat, y, x] += value


def build_fc():
    T = np.zeros((10, NF, H, Wg), dtype=np.float32)

    # ===== AIRPLANE (0) =====
    # Strong blue sky top; pale plane in centre; no grass anywhere.
    add(T, 0, DBLUE, TOP, 18)
    add(T, 0, BLUE,  TOP, 8)
    add(T, 0, WHITE, CENTRE, 10)
    add(T, 0, BR_MID, CENTRE, 2)
    add(T, 0, GREEN, ALL, -10)
    add(T, 0, YEL,   CENTRE, -5)
    add(T, 0, PRED,  CENTRE, -8)
    add(T, 0, DBLUE, BOTTOM, -6)
    add(T, 0, BLUE,  BOTTOM, -4)

    # ===== AUTOMOBILE (1) =====
    # Saturated-colour centre (PRED or BLUE or DARK), DARK road below, no sky.
    add(T, 1, PRED, CENTRE, 22)
    add(T, 1, BLUE, CENTRE, 10)
    add(T, 1, DARK, CENTRE, 14)
    add(T, 1, DBLUE, TOP, -12)
    add(T, 1, BLUE,  TOP, -8)
    add(T, 1, GREEN, ALL, -6)
    add(T, 1, DARK,  BOTTOM, 8)
    add(T, 1, YEL,   CENTRE, -8)
    add(T, 1, WHITE, CENTRE, -4)
    add(T, 1, DARK,  SIDES, -4)       # confine darkness to centre / road

    # ===== BIRD (2) =====
    # Generic small animal — backup for warm/dark small object with sky or green.
    add(T, 2, DBLUE, TOP, 7)
    add(T, 2, GREEN, TOP, 6)
    add(T, 2, GREEN, BOTTOM, 4)
    add(T, 2, GREEN, SIDES, 4)
    add(T, 2, YEL,   CENTRE, 14)
    add(T, 2, DARK,  CENTRE, 10)
    add(T, 2, WHITE, CENTRE, 6)
    add(T, 2, RED,   CENTRE, 6)
    add(T, 2, DBLUE, BOTTOM, -6)

    # ===== CAT (3) =====
    # Fur centre (warm / dark / white) indoors, no heavy green or blue sky.
    # Cat body is compact: penalise WHITE/DARK in the sides and top (truck-sized
    # bright blocks are not cats).
    add(T, 3, YEL,   CENTRE, 20)
    add(T, 3, DARK,  CENTRE, 8)
    add(T, 3, WHITE, CENTRE, 10)
    add(T, 3, R,     CENTRE, 0.8)
    add(T, 3, GREEN, ALL, -5)
    add(T, 3, DBLUE, TOP, -10)
    add(T, 3, BLUE,  TOP, -5)
    add(T, 3, WHITE, SIDES, -4)
    add(T, 3, WHITE, TOP, -3)
    add(T, 3, DARK,  SIDES, -3)

    # ===== DEER (4) =====
    # Brown body + STRONG grass below + leaves above.
    add(T, 4, YEL,   CENTRE, 18)
    add(T, 4, GREEN, BOTTOM, 22)
    add(T, 4, GREEN, TOP, 8)
    add(T, 4, GREEN, SIDES, 6)
    add(T, 4, DBLUE, TOP, -10)
    add(T, 4, DBLUE, BOTTOM, -10)
    add(T, 4, WHITE, CENTRE, -5)
    add(T, 4, PRED,  CENTRE, -6)
    add(T, 4, R,     CENTRE, 0.3)

    # ===== DOG (5) =====
    # Warm/dark fur, some grass, generic outdoor-friendly pet.
    add(T, 5, YEL,   CENTRE, 18)
    add(T, 5, DARK,  CENTRE, 10)
    add(T, 5, WHITE, CENTRE, 7)
    add(T, 5, R,     CENTRE, 0.6)
    add(T, 5, GREEN, BOTTOM, 10)
    add(T, 5, GREEN, SIDES, 4)
    add(T, 5, DBLUE, TOP, -8)
    add(T, 5, DBLUE, BOTTOM, -6)
    add(T, 5, PRED,  CENTRE, -4)

    # ===== FROG (6) =====
    # Centre is the strongest frog signal; surrounds add moderate weight.
    add(T, 6, GREEN, CENTRE, 32)
    add(T, 6, GREEN, TOP, 6)
    add(T, 6, GREEN, BOTTOM, 6)
    add(T, 6, GREEN, SIDES, 8)
    add(T, 6, YEL,   CENTRE, 3)
    add(T, 6, G,     CENTRE, 1.5)       # push even moderately-green centres up
    add(T, 6, DBLUE, ALL, -12)
    add(T, 6, BLUE,  ALL, -4)
    add(T, 6, PRED,  CENTRE, -6)
    add(T, 6, WHITE, CENTRE, -4)

    # ===== HORSE (7) =====
    # Darker brown body + STRONG grass below.  Overlaps deer heavily.
    add(T, 7, YEL,   CENTRE, 13)
    add(T, 7, DARK,  CENTRE, 8)
    add(T, 7, GREEN, BOTTOM, 24)
    add(T, 7, GREEN, TOP, 6)
    add(T, 7, GREEN, SIDES, 6)
    add(T, 7, DBLUE, TOP, -10)
    add(T, 7, DBLUE, BOTTOM, -10)
    add(T, 7, WHITE, CENTRE, -5)
    add(T, 7, PRED,  CENTRE, -5)
    add(T, 7, R,     CENTRE, 0.3)

    # ===== SHIP (8) =====
    # Water in lower half + sky in upper half + bright hull centre.
    add(T, 8, DBLUE, BOTTOM, 22)
    add(T, 8, DBLUE, TOP, 8)
    add(T, 8, BLUE,  TOP, 6)
    add(T, 8, BLUE,  BOTTOM, 6)
    add(T, 8, WHITE, CENTRE, 8)
    add(T, 8, GREEN, ALL, -10)
    add(T, 8, YEL,   CENTRE, -6)
    add(T, 8, PRED,  CENTRE, -5)

    # ===== TRUCK (9) =====
    # Large WHITE/pale body across middle AND sides, dark road below.
    add(T, 9, WHITE, CENTRE, 14)
    add(T, 9, WHITE, SIDES, 6)             # body extends to sides (vs cat)
    add(T, 9, BR_MID, CENTRE, 2)
    add(T, 9, DARK, BOTTOM, 6)
    add(T, 9, BLUE, TOP, 2)
    add(T, 9, PRED, CENTRE, -8)
    add(T, 9, YEL,  CENTRE, -4)
    add(T, 9, GREEN, ALL, -6)
    add(T, 9, DBLUE, BOTTOM, -6)

    # per-class scalar bias to balance baseline scores
    b = np.zeros(10, dtype=np.float32)
    b[0] = 0.2
    b[1] = 0.0
    b[2] = 1.0     # modest bird fallback
    b[3] = 0.5
    b[4] = 0.0
    b[5] = 0.0
    b[6] = 0.0
    b[7] = 0.0
    b[8] = 0.3     # boost ship a little
    b[9] = 0.0

    return T.reshape(10, -1), b


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/out"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(ARCH, f, indent=2)

    cw, cb = build_colors()
    fw, fb = build_fc()

    np.savez(
        os.path.join(out_dir, "weights.npz"),
        **{
            "colors.W": cw.astype(np.float32),
            "colors.b": cb.astype(np.float32),
            "fc.W":     fw.astype(np.float32),
            "fc.b":     fb.astype(np.float32),
        },
    )


if __name__ == "__main__":
    main()
