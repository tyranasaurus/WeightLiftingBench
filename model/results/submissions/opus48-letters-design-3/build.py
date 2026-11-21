"""Builds an ASL-letter classifier's weights from general knowledge of hand shapes.

The image is contrast-normalised, then a bank of oriented ridge filters (oriented second-derivative-of-
Gaussian) measures bar energy at four orientations.  Each filter is kept with its negative and both pass
through relu, giving polarity-invariant bar energy (the hand may be lighter OR darker than its background).
maxpool over 4x4 blocks records, for a 7x7 grid and each orientation, the peak oriented structure there.

The last layer is a hand-written template over (orientation, grid-cell).  Letters are told apart by WHICH
orientation of edge sits WHERE, chosen so the cues are disjoint between easily-confused letters:
  * a fist/blob has a rounded HORIZONTAL top & bottom edge and no vertical reach up;
  * fingers-up letters have VERTICAL finger edges in the top rows (B wide, D one, U two, V a gap, W three,
    I to a side, K/R with a thumb/cross, X short, F three-on-the-right, L a left post + horizontal thumb);
  * sideways G,H have a HORIZONTAL bar reaching the far LEFT;
  * C/O are rings (vertical sides + horizontal top & bottom), O closed, C open on the right;
  * P,Q point down; Y reaches out to both sides.
The bottom two grid rows (the wrist/forearm entering from the bottom edge) carry no weight.  No training
data or statistics are used as weights; viewing the data only guided which rows/columns each handshape
occupies.
"""
import sys, json, os
import numpy as np

K = 9              # ridge filter size
SA, SL = 1.1, 2.3  # across / along sigmas (tuned to a ~3px finger)
G = 7              # pooled grid
PHI = [0.0, 90.0, 45.0, 135.0]   # across-bar angle: 0 vertical bar, 1 horizontal bar, 2 "\", 3 "/"


def ridge(phi_deg):
    c = (K - 1) / 2.0
    ys, xs = np.mgrid[0:K, 0:K].astype(np.float64)
    x, y = xs - c, ys - c
    phi = np.deg2rad(phi_deg)
    u = x * np.cos(phi) + y * np.sin(phi)
    v = -x * np.sin(phi) + y * np.cos(phi)
    g = np.exp(-(u * u / (2 * SA * SA) + v * v / (2 * SL * SL)))
    f = ((SA * SA - u * u) / SA ** 4) * g
    f -= f.mean()
    f /= np.sqrt((f * f).sum())
    return f.astype(np.float32)


def build_filters():
    W = np.zeros((8, 1, K, K), np.float32)
    for o in range(4):
        f = ridge(PHI[o])
        W[o, 0] = f
        W[o + 4, 0] = -f
    return W, np.zeros((8,), np.float32)


Vo, Ho, Do = [0], [1], [2, 3]   # orientation groups (V vertical bar, H horizontal bar, D diagonals)


def build_readout():
    w = np.zeros((24, 8, G, G), np.float64)

    def add(cls, orients, gys, gxs, val=1.0):
        for o in orients:
            for gy in gys:
                for gx in gxs:
                    w[cls, o, gy, gx] += val
                    w[cls, o + 4, gy, gx] += val

    # ---- fist / blob letters: horizontal domed top & bottom, compact (no vertical reach up) ----
    add(0, Ho, [2], [2, 3]); add(0, Ho, [4], [2, 3]); add(0, Do, [3], [1], 0.6)                        # A
    add(4, Ho, [2], [2, 3, 4]); add(4, Ho, [3], [2, 3, 4])                                             # E
    add(17, Ho, [2], [2, 3, 4]); add(17, Ho, [4], [2, 3, 4])                                           # S
    add(11, Ho, [2], [2, 3, 4]); add(11, Ho, [4], [2, 3, 4]); add(11, Vo, [3], [2, 3, 4], 0.5)         # M
    add(12, Ho, [2], [3, 4]); add(12, Ho, [4], [3, 4]); add(12, Vo, [3], [3, 4], 0.5)                  # N
    add(18, Ho, [3], [1, 2, 3]); add(18, Do, [3], [1], 0.6)                                            # T

    # ---- fingers up: vertical edges reaching the top rows ----
    add(1, Vo, [0], [2, 3, 4]); add(1, Vo, [1], [2, 3, 4, 5]); add(1, Vo, [2], [2, 3, 4], 0.5)         # B wide
    add(3, Vo, [0], [3]); add(3, Vo, [1], [3]); add(3, Vo, [2], [3], 0.5)                              # D one, centre
    add(19, Vo, [0], [3, 4]); add(19, Vo, [1], [3, 4]); add(19, Vo, [2], [3, 4], 0.5)                  # U two together
    add(20, Vo, [0], [2]); add(20, Vo, [1], [2]); add(20, Vo, [0], [4]); add(20, Vo, [1], [4])
    add(20, Do, [1], [3], 0.6)                                                                         # V two spread (gap gx3)
    add(21, Vo, [0], [2, 3, 4]); add(21, Vo, [1], [2, 3, 4])                                           # W three
    add(9, Vo, [0], [3, 4]); add(9, Vo, [1], [3, 4]); add(9, Do, [1, 2], [2, 3], 0.6)                  # K spread + thumb
    add(16, Vo, [0], [3]); add(16, Vo, [1], [3]); add(16, Do, [1], [2, 3], 0.7)                        # R crossed
    add(8, Vo, [0], [4]); add(8, Vo, [1], [4]); add(8, Vo, [2], [4], 0.5)                              # I little finger (right)
    add(22, Vo, [1], [3]); add(22, Vo, [2], [3]); add(22, Do, [1], [3], 0.7)                           # X bent, short
    add(5, Vo, [0], [4, 5]); add(5, Vo, [1], [4, 5]); add(5, Do, [3], [1, 2], 0.6)                     # F three up right + loop
    add(10, Vo, [0, 1, 2], [1]); add(10, Ho, [4], [2, 3], 0.9)                                         # L up-left + thumb

    # ---- sideways (left) ----
    add(6, Ho, [3], [0, 1, 2]); add(6, Ho, [2], [1, 2], 0.5)                                           # G one finger left
    add(7, Ho, [3], [0, 1, 2]); add(7, Ho, [4], [0, 1, 2])                                             # H two fingers left

    # ---- pointing down (forearm band excluded below) ----
    add(14, Vo, [4], [3, 4]); add(14, Do, [4], [3], 0.5)                                               # P
    add(15, Vo, [4], [3]); add(15, Vo, [4], [2], 0.5)                                                  # Q

    # ---- rings / spread ----
    add(2, Ho, [1], [2, 3]); add(2, Ho, [4], [2, 3]); add(2, Vo, [2, 3], [1], 0.9)                     # C open right
    add(13, Ho, [1], [2, 3]); add(13, Ho, [4], [2, 3]); add(13, Vo, [2, 3], [1], 0.8); add(13, Vo, [2, 3], [4], 0.8)  # O closed
    add(23, Vo, [3], [1], 0.9); add(23, Do, [3], [1], 0.7); add(23, Vo, [3], [5], 0.9); add(23, Do, [3], [5], 0.7)    # Y both sides

    w[:, :, 5:, :] = 0.0                               # forearm band carries no weight
    for c in range(24):
        n = np.sqrt((w[c] ** 2).sum())
        if n > 0:
            w[c] /= n
    return w.reshape(24, -1).astype(np.float32), np.zeros((24,), np.float32)


NETWORK = {
    "layers": [
        {"name": "z", "op": "norm", "inputs": ["image"]},
        {"name": "edge", "op": "conv", "inputs": ["z"], "out": 8, "kernel": [K, K], "stride": 1, "padding": (K - 1) // 2},
        {"name": "re", "op": "relu", "inputs": ["edge"]},
        {"name": "pool", "op": "maxpool", "inputs": ["re"], "size": [4, 4], "stride": [4, 4]},
        {"name": "out", "op": "dense", "inputs": ["pool"], "out": 24},
    ]
}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    eW, eb = build_filters()
    oW, ob = build_readout()
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(NETWORK, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **{"edge.W": eW, "edge.b": eb, "out.W": oW, "out.b": ob})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
