"""Builds network.json and weights.npz for a hand-designed person/no-person classifier.

Design rationale
----------------
From general knowledge of COCO-style natural photos:
- No-person photos in COCO are frequently close-ups of warm/organic subjects
  (food, pets, foliage, warm vehicles) or wide nature scenes (sky above grass or
  earth below).  These tend to have a high fraction of pixels where R+G exceeds
  2B (warm / green cast) *and* a strong top-to-bottom luminance gradient: a
  bright sky overhead and a darker ground / subject below.
- Person photos cover varied indoor and outdoor scenes and introduce clothing,
  limbs and background objects, which raise image complexity (edge density)
  and weaken the clean sky-above / ground-below gradient because a person's
  body occupies the middle of the frame.

The network therefore extracts three complementary signals:
1. The average "warm-greenness" excess (R+G-2B)+ pooled over the image.
2. The average |luminance gradient| pooled over the image (edge density).
3. The top-half minus bottom-half mean luminance (sky-over-ground asymmetry).
All three are pooled from an upper-half window and a lower-half window so a
single dense layer can form the asymmetry from the two halves.

The dense readout scores class 0 (no-person) with the warm signal plus the
top-over-bottom brightness drop, and scores class 1 (person) with the edge
density, with scales chosen so typical images see comparable magnitudes for
both scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    net = {
        "layers": [
            {"name": "c1", "op": "conv",   "inputs": ["image"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "r1", "op": "relu",   "inputs": ["c1"]},
            {"name": "c2", "op": "conv",   "inputs": ["r1"],   "out": 6, "kernel": [3, 3], "stride": 1, "padding": 1},
            {"name": "r2", "op": "relu",   "inputs": ["c2"]},
            {"name": "p1", "op": "avgpool","inputs": ["r2"],   "size": [48, 96], "stride": [48, 96]},
            {"name": "fl", "op": "reshape","inputs": ["p1"],   "shape": [12]},
            {"name": "out","op": "dense",  "inputs": ["fl"],   "out": 2},
        ]
    }

    # --- conv1: 3 colour channels -> 2 colour features (1x1 kernel) ---
    # ch0: luminance Y = 0.299 R + 0.587 G + 0.114 B
    # ch1: warm/green excess over blue = R + G - 2 B (ReLU in r1 keeps the
    #      non-blue-dominated pixels).
    W_c1 = np.zeros((2, 3, 1, 1), dtype=np.float32)
    W_c1[0, 0, 0, 0] = 0.299
    W_c1[0, 1, 0, 0] = 0.587
    W_c1[0, 2, 0, 0] = 0.114
    W_c1[1, 0, 0, 0] = 1.0
    W_c1[1, 1, 0, 0] = 1.0
    W_c1[1, 2, 0, 0] = -2.0
    b_c1 = np.zeros(2, dtype=np.float32)

    # --- conv2: 2 -> 6 (3x3 kernels) ---
    # Sobel-like gradient detectors on luminance (channel 0 of r1) and simple
    # 3x3 mean filters on luminance and warm channels.  ReLU on signed Sobel
    # halves keeps one direction of gradient per channel; the pair +Sobel /
    # -Sobel then sums to the full |gradient| average.
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32) / 8.0
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32) / 8.0
    mean3 = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)

    W_c2 = np.zeros((6, 2, 3, 3), dtype=np.float32)
    W_c2[0, 0] =  sobel_x   # ch0: +dY/dx
    W_c2[1, 0] = -sobel_x   # ch1: -dY/dx
    W_c2[2, 0] =  sobel_y   # ch2: +dY/dy
    W_c2[3, 0] = -sobel_y   # ch3: -dY/dy
    W_c2[4, 0] =  mean3     # ch4: smoothed Y (per-pixel brightness)
    W_c2[5, 1] =  mean3     # ch5: smoothed warm
    b_c2 = np.zeros(6, dtype=np.float32)

    # --- pooling: split 96x96 into two 48x96 halves -> (6, 2, 1) ---
    # reshape to (12,) with channel-major ordering, so the 12 features are:
    #   idx 0,1 = ch0 top,bot (+dY/dx)
    #   idx 2,3 = ch1 top,bot (-dY/dx)
    #   idx 4,5 = ch2 top,bot (+dY/dy)
    #   idx 6,7 = ch3 top,bot (-dY/dy)
    #   idx 8,9 = ch4 top,bot (mean Y)
    #   idx 10,11= ch5 top,bot (mean warm)

    # --- dense: 12 features -> 2 class scores ---
    # Class 0 (no-person) = warm_top + warm_bot                (total warmness)
    #                     + (mean_Y_top - mean_Y_bot)          (top-brighter asymmetry)
    # Class 1 (person)    = K_e * (sum of 8 edge features)     (edge density)
    #
    # Typical natural-photo magnitudes:
    #   edge per half ~ 0.016 => 8 halves total ~ 0.13
    #   warm per half ~ 0.15  => sum ~ 0.30 (we use sum, so expect ~0.3)
    #   mean_Y per half ~ 0.45; asymmetry (top-bot) ~ 0.03-0.07 for landscapes
    # The warm sum would dominate, so we weight warm by 0.5 to keep scales
    # comparable, giving typical class-0 ~ 0.15 + 0.03 = 0.18.  We set K_e so
    # the edge-density contribution for a typical image sits near 0.17.
    W_out = np.zeros((2, 12), dtype=np.float32)
    # Class 0 weights -- warm total + luminance top-bot asymmetry
    W_out[0, 8]  =  1.3   # +mean_Y_top
    W_out[0, 9]  = -1.3   # -mean_Y_bot    (contribution = 1.3 * (top-bot asymmetry))
    W_out[0, 10] =  1.0   # +warm_top
    W_out[0, 11] =  1.0   # +warm_bot      (contribution = warm_total_sum)
    # Class 1 weights: edge density (8 edge-half features, each weighted 2.0)
    K_e = 2.0
    for idx in (0, 1, 2, 3, 4, 5, 6, 7):
        W_out[1, idx] = K_e
    b_out = np.zeros(2, dtype=np.float32)

    weights = {
        "c1.W":  W_c1,  "c1.b":  b_c1,
        "c2.W":  W_c2,  "c2.b":  b_c2,
        "out.W": W_out, "out.b": b_out,
    }

    with open(out_dir / "network.json", "w") as f:
        json.dump(net, f, indent=2)
    np.savez(out_dir / "weights.npz", **weights)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/out")
    build(out)
