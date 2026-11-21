"""Hand-crafted CIFAR-10 classifier.

Writes network.json and weights.npz into the folder given on the command line.
Every weight is set from general knowledge of how the ten classes look to a
human: blue dominates the top of airplane scenes, blue dominates both top
(sky) and bottom (water) of ship scenes, frogs are green and not bright,
deer and horses are warm-brown bodies against green surrounds, cats and
dogs are furry warm-dark bodies with no outdoor cues, automobiles and
trucks have dark strips of road under them.  No training data is read.

Architecture
------------
image (3,32,32)
  -> c1  conv 3 -> 6, 1x1     colour-opponent features (Blue, Green, Brown,
                              Red, Dark, VeryBright) designed so brown does
                              not fire on pure green pixels and red does not
                              fire on warm tones with large B.
  -> r1  relu
  -> p1  avgpool 4x4 s4       -> (6, 8, 8) 8x8 regional feature map
  -> rs  reshape [384]
  -> fc  dense 384 -> 10      per-class spatial templates with per-class
                              bias calibrated to a "typical" feature level.
"""

import json
import sys
from pathlib import Path

import numpy as np


# feature indices into the (6, 8, 8) pooled feature map
BLU, GRN, BRN, RED, DRK, VBR = range(6)
N_FEAT = 6


# General-knowledge priors for the typical regional value of each feature
# over an arbitrary natural photo.  Used only to calibrate per-class biases
# so no class wins just by having larger raw weights on common features
# (e.g. DRK, BRN).  These are intuitive estimates, not measured values.
MU = np.array([
    0.050,   # BLU
    0.050,   # GRN
    0.075,   # BRN  warm/brown is common
    0.020,   # RED  pure red / orange is rare
    0.090,   # DRK  dark pixels are common
    0.030,   # VBR  very bright pixels
], dtype=np.float32)


def build_c1():
    """Six colour-opponent features on raw RGB.

    BLU - saturated blue   : B above average of R and G.
    GRN - saturated green  : G above average of R and B.
    BRN - warm / brown     : R + G large, B small.  The G coefficient is
                             deliberately smaller than the R coefficient so
                             pure-green pixels (grass, frog) do NOT fire BRN.
    RED - saturated red    : R dominant, B explicitly small.
    DRK - brightness below ~0.30 (active on dark pixels).
    VBR - brightness above ~0.75 (active on bright highlights).
    """
    rows = [
        ([-0.5, -0.5,  1.0], -0.03),   # BLU
        ([-0.5,  1.0, -0.5], -0.03),   # GRN
        ([ 0.4,  0.3, -0.7], -0.03),   # BRN
        ([ 1.0, -0.4, -0.6], -0.03),   # RED
        ([-1/3, -1/3, -1/3],  0.30),   # DRK
        ([ 1/3,  1/3,  1/3], -0.75),   # VBR
    ]
    W = np.zeros((6, 3, 1, 1), dtype=np.float32)
    b = np.zeros(6, dtype=np.float32)
    for i, (w, bias) in enumerate(rows):
        W[i, :, 0, 0] = w
        b[i] = bias
    return W, b


# ---- spatial masks on the 8x8 regional grid ----
H = 8
def mkmask(fn):
    m = np.zeros((H, H), dtype=np.float32)
    for y in range(H):
        for x in range(H):
            m[y, x] = fn(y, x)
    return m

ALL     = mkmask(lambda y, x: 1.0)
TOP     = mkmask(lambda y, x: 1.0 if y < 4 else 0.0)
BOTTOM  = mkmask(lambda y, x: 1.0 if y >= 4 else 0.0)
UPPER   = mkmask(lambda y, x: 1.0 if y < 2 else 0.0)
LOWER   = mkmask(lambda y, x: 1.0 if y >= 6 else 0.0)
MIDBAND = mkmask(lambda y, x: 1.0 if 3 <= y <= 4 else 0.0)
CENTER  = mkmask(lambda y, x: 1.0 if 2 <= y < 6 and 2 <= x < 6 else 0.0)
EDGES   = ALL - CENTER
BOT_CEN = mkmask(lambda y, x: 1.0 if y >= 4 and 2 <= x < 6 else 0.0)
TOP_EDGES = TOP * (1 - CENTER)


def build_fc():
    """Class-specific (6, 8, 8) templates based on what the class looks like."""
    W = np.zeros((10, N_FEAT, H, H), dtype=np.float32)

    # 0: AIRPLANE - blue sky dominates upper half; plane body is bright;
    # no green or warm fur anywhere
    W[0, BLU] = 2.4 * TOP + 0.3 * BOTTOM
    W[0, GRN] = -1.8 * ALL
    W[0, BRN] = -1.2 * ALL
    W[0, RED] = -0.9 * ALL
    W[0, DRK] = -0.8 * ALL
    W[0, VBR] = 0.8 * TOP

    # 1: AUTOMOBILE - small centre body (varied colour), dark road strip at
    # the bottom, no grass
    W[1, BLU] = -0.4 * BOTTOM + 0.3 * UPPER
    W[1, GRN] = -1.9 * ALL
    W[1, BRN] = 0.5 * CENTER
    W[1, RED] = 0.9 * CENTER
    W[1, DRK] = 2.6 * LOWER + 0.4 * CENTER
    W[1, VBR] = 0.3 * CENTER

    # 2: BIRD - small coloured object in centre, natural surround, moderate
    # saturation at centre
    W[2, BLU] = 0.4 * TOP_EDGES
    W[2, GRN] = 0.4 * EDGES
    W[2, BRN] = 1.2 * CENTER
    W[2, RED] = 1.3 * CENTER
    W[2, DRK] = 0.5 * CENTER
    W[2, VBR] = 0.5 * CENTER

    # 3: CAT - warm or dark furry face in middle; no outdoor cues
    W[3, BLU] = -1.6 * ALL
    W[3, GRN] = -1.1 * ALL
    W[3, BRN] = 1.3 * CENTER
    W[3, RED] = 0.7 * CENTER
    W[3, DRK] = 0.8 * CENTER
    W[3, VBR] = 0.5 * CENTER

    # 4: DEER - tan body in centre, green surround (forest/grass)
    W[4, BLU] = -2.4 * ALL
    W[4, GRN] = 1.3 * EDGES + 0.6 * BOTTOM
    W[4, BRN] = 2.2 * CENTER + 0.4 * MIDBAND
    W[4, RED] = 0.4 * CENTER
    W[4, VBR] = -0.3 * ALL

    # 5: DOG - warm fur in centre, varied background; emphasise warmth in
    # centre more than cat's "indoor dark" profile
    W[5, BLU] = -0.9 * ALL
    W[5, GRN] = -0.2 * ALL
    W[5, BRN] = 1.9 * CENTER
    W[5, RED] = 1.0 * CENTER
    W[5, DRK] = 0.6 * CENTER
    W[5, VBR] = 0.6 * CENTER

    # 6: FROG - green and muted; frogs are not bright or sky-blue
    W[6, BLU] = -1.8 * ALL
    W[6, GRN] = 1.7 * ALL
    W[6, RED] = -1.3 * ALL
    W[6, BRN] = -0.1 * ALL
    W[6, VBR] = -1.3 * ALL
    W[6, DRK] = 0.3 * ALL

    # 7: HORSE - brown or dark body in middle, grass below
    W[7, BLU] = -1.3 * ALL
    W[7, GRN] = 1.2 * BOTTOM + 0.4 * EDGES
    W[7, BRN] = 1.8 * CENTER
    W[7, DRK] = 0.9 * CENTER
    W[7, RED] = 0.3 * CENTER
    W[7, VBR] = -0.3 * ALL

    # 8: SHIP - blue at both top (sky) and bottom (water); no green
    W[8, BLU] = 2.5 * TOP + 4.0 * BOTTOM
    W[8, GRN] = -2.0 * ALL
    W[8, BRN] = -1.3 * ALL
    W[8, RED] = -0.8 * ALL
    W[8, DRK] = -0.2 * BOTTOM
    W[8, VBR] = 0.3 * TOP

    # 9: TRUCK - boxy body, dark strip under; often outdoor sky above
    W[9, BLU] = 0.4 * UPPER - 0.4 * BOTTOM
    W[9, GRN] = -1.5 * ALL
    W[9, BRN] = 0.5 * CENTER
    W[9, RED] = 0.6 * CENTER
    W[9, DRK] = 3.0 * LOWER + 0.3 * CENTER
    W[9, VBR] = 0.4 * TOP

    # Per-class bias calibration: zero expected score at "typical" feature
    # levels (f at MU[f] everywhere).  Keeps classes comparable so the
    # argmax is driven by the image's actual deviation pattern.
    mu = MU.reshape(1, N_FEAT, 1, 1)
    expected = (W * mu).reshape(10, -1).sum(axis=1).astype(np.float32)
    fc_b = -expected

    fc_W = W.reshape(10, N_FEAT * H * H).astype(np.float32)
    return fc_W, fc_b


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    net = {"layers": [
        {"name": "c1", "op": "conv", "inputs": ["image"],
         "out": 6, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "p1", "op": "avgpool", "inputs": ["r1"],
         "size": [4, 4], "stride": [4, 4]},
        {"name": "rs", "op": "reshape", "inputs": ["p1"], "shape": [384]},
        {"name": "fc", "op": "dense", "inputs": ["rs"], "out": 10},
    ]}
    (out / "network.json").write_text(json.dumps(net))

    c1_W, c1_b = build_c1()
    fc_W, fc_b = build_fc()

    np.savez(out / "weights.npz",
             **{"c1.W": c1_W, "c1.b": c1_b,
                "fc.W": fc_W, "fc.b": fc_b})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
