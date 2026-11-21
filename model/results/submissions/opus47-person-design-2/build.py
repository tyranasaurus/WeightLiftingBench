"""Builds network.json and weights.npz for person/no-person classifier.

Architecture: simple single-feature classifier based on edge density.

  l1: 3x3 conv, 3->4, padding 1
       4 channels compute forward differences (dx+, dx-, dy+, dy-) on brightness.
  r1: relu -> gives |dx|, |dy| when summed.
  l2: 1x1 conv, 4->1
       output = |dx| + |dy| - 0.1 (per-pixel gradient above threshold).
  r2: relu -> edge intensity map (only positive values survive).
  pool: avgpool 96x96 (global) -> average edge intensity in image.
  flat: reshape to (1,)
  cls: dense 1->2
       Person photos have more texture/edges than no-person scenes (which often
       contain large uniform regions: sky, walls, water, uniform animal hides).
       Threshold 0.03 reflects typical edge-density split.

All numbers come from this file; no training data is read.
"""
import json, sys, os, numpy as np


def main(out_dir):
    net = {"layers": [
        {"name": "l1", "op": "conv", "inputs": ["image"], "out": 4,
         "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "r1", "op": "relu", "inputs": ["l1"]},
        {"name": "l2", "op": "conv", "inputs": ["r1"], "out": 1,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "r2", "op": "relu", "inputs": ["l2"]},
        {"name": "pool", "op": "avgpool", "inputs": ["r2"],
         "size": [96, 96], "stride": [1, 1]},
        {"name": "flat", "op": "reshape", "inputs": ["pool"], "shape": [1]},
        {"name": "cls", "op": "dense", "inputs": ["flat"], "out": 2},
    ]}

    W_l1 = np.zeros((4, 3, 3, 3), dtype=np.float32)
    b_l1 = np.zeros(4, dtype=np.float32)

    # Forward differences of brightness in 4 directions.
    for c in range(3):
        W_l1[0, c, 1, 1] = -1.0 / 3.0; W_l1[0, c, 1, 2] = 1.0 / 3.0  # dx+
        W_l1[1, c, 1, 1] = 1.0 / 3.0;  W_l1[1, c, 1, 2] = -1.0 / 3.0  # dx-
        W_l1[2, c, 1, 1] = -1.0 / 3.0; W_l1[2, c, 2, 1] = 1.0 / 3.0  # dy+
        W_l1[3, c, 1, 1] = 1.0 / 3.0;  W_l1[3, c, 2, 1] = -1.0 / 3.0  # dy-

    # Edge pixel: |dx| + |dy| - 0.1 (strong edges only)
    W_l2 = np.ones((1, 4, 1, 1), dtype=np.float32)
    b_l2 = np.array([-0.1], dtype=np.float32)

    # Dense classifier: 1 -> 2
    # Score difference = 2 * (edge_density - 0.035).
    # Threshold chosen by hand reflecting that typical natural photos have
    # a small fraction of strong-edge pixels.
    W_cls = np.array([[-100.0], [100.0]], dtype=np.float32)
    b_cls = np.array([3.5, -3.5], dtype=np.float32)

    weights = {
        "l1.W": W_l1, "l1.b": b_l1,
        "l2.W": W_l2, "l2.b": b_l2,
        "cls.W": W_cls, "cls.b": b_cls,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
