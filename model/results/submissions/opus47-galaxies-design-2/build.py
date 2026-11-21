"""Builds network.json and weights.npz for a hand-designed galaxy classifier.

The network reads a 128x128 RGB image, derives per-pixel luminance /
blueness / redness channels with a 1x1 conv, downsamples (two 2x2 avgpool
steps) to 32x32, then spatial-normalises each channel (per-channel mean
subtracted, divided by std) so the sky background sits near zero. A ReLU
keeps only the galaxy pixels.

Two 32x32 convolutions act on this 32x32 feature map (full-image kernels):

  * `feats` computes 16 non-oriented templates drawn from general galaxy
    knowledge: isotropic Gaussians of several sizes (compact centre vs
    extended), annular rings (extent), differences of Gaussians
    (concentration contrast), and color-channel templates (blue centre /
    blue ring for spirals, red centre / extended for red ellipticals).

  * `oriented` computes 8 narrow elongated Gaussians at angles 0..7*pi/8.
    A max-pool over the 8 outputs gives a rotation-invariant "strongest
    axis" strength; an avg-pool gives the mean axis strength. Their
    difference (max - avg), expressed through two parallel dense layers
    whose outputs are added, is a scale-invariant elongation indicator
    that identifies edge-on (8, 9) and cigar-shaped (4) galaxies.

Dense weights for each class were chosen by hand from general knowledge:
e.g. round smooth (2) gets large central luminance + red centre +
moderate extent and dislikes elongation; barred / unbarred spirals
(5, 6, 7) favour blue extent / blue ring / blue centre; edge-on with
bulge (9) likes a red central bulge on top of strong elongation, while
edge-on without bulge (8) is the preferred edge-on default; cigar (4)
is small and red with mild elongation. Biases slightly favour the modal
class so ambiguous galaxies default to plausible predictions.
"""
import sys, os, json
import numpy as np


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    F = 16

    network = {
        "layers": [
            {"name": "down1", "op": "avgpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]},
            {"name": "color", "op": "conv", "inputs": ["down1"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "down2", "op": "avgpool", "inputs": ["color"], "size": [2, 2], "stride": [2, 2]},
            {"name": "nrm", "op": "norm", "inputs": ["down2"]},
            {"name": "r", "op": "relu", "inputs": ["nrm"]},
            {"name": "feats", "op": "conv", "inputs": ["r"], "out": F, "kernel": [32, 32], "stride": 1, "padding": 0},
            {"name": "flat_a", "op": "reshape", "inputs": ["feats"], "shape": [F]},
            {"name": "dense_a", "op": "dense", "inputs": ["flat_a"], "out": 10},
            {"name": "oriented", "op": "conv", "inputs": ["r"], "out": 8, "kernel": [32, 32], "stride": 1, "padding": 0},
            {"name": "resh_b", "op": "reshape", "inputs": ["oriented"], "shape": [1, 1, 8]},
            {"name": "maxor", "op": "maxpool", "inputs": ["resh_b"], "size": [1, 8], "stride": [1, 8]},
            {"name": "avgor", "op": "avgpool", "inputs": ["resh_b"], "size": [1, 8], "stride": [1, 8]},
            {"name": "flat_max", "op": "reshape", "inputs": ["maxor"], "shape": [1]},
            {"name": "flat_avg", "op": "reshape", "inputs": ["avgor"], "shape": [1]},
            {"name": "dense_max", "op": "dense", "inputs": ["flat_max"], "out": 10},
            {"name": "dense_avg", "op": "dense", "inputs": ["flat_avg"], "out": 10},
            {"name": "out", "op": "add", "inputs": ["dense_a", "dense_max", "dense_avg"]},
        ]
    }

    with open(os.path.join(out_dir, "network.json"), "w") as fjson:
        json.dump(network, fjson, indent=1)

    weights = {}

    # Color 1x1 conv: (R, G, B) -> (lum, blueness = B-R, redness = R-B)
    Wc = np.zeros((3, 3, 1, 1), dtype=np.float32)
    Wc[0, 0, 0, 0] = 1.0 / 3; Wc[0, 1, 0, 0] = 1.0 / 3; Wc[0, 2, 0, 0] = 1.0 / 3
    Wc[1, 0, 0, 0] = -1.0;    Wc[1, 2, 0, 0] =  1.0
    Wc[2, 0, 0, 0] =  1.0;    Wc[2, 2, 0, 0] = -1.0
    weights['color.W'] = Wc
    weights['color.b'] = np.zeros(3, dtype=np.float32)

    # Grid for 32x32 templates, centre at (15.5, 15.5)
    y, x = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing='ij')
    r2 = x * x + y * y

    def gauss_iso(sig):
        g = np.exp(-r2 / (2.0 * sig * sig))
        return (g / g.sum()).astype(np.float64)

    def gauss_oriented(smajor, sminor, angle):
        c, s = np.cos(angle), np.sin(angle)
        xr =  x * c + y * s
        yr = -x * s + y * c
        g = np.exp(-(xr * xr / (2.0 * smajor * smajor) + yr * yr / (2.0 * sminor * sminor)))
        return (g / g.sum()).astype(np.float64)

    def ring(r_in, r_out):
        m = ((r2 >= r_in * r_in) & (r2 < r_out * r_out)).astype(np.float64)
        return m / m.sum()

    LUM, BLUE, RED = 0, 1, 2

    Wf = np.zeros((F, 3, 32, 32), dtype=np.float32)
    bf = np.zeros(F, dtype=np.float32)

    Wf[0, LUM]   = gauss_iso(1.5)                            # tight centre spot
    Wf[1, LUM]   = gauss_iso(3.0)                            # small central
    Wf[2, LUM]   = gauss_iso(5.0)                            # medium
    Wf[3, LUM]   = gauss_iso(8.0)                            # large
    Wf[4, LUM]   = gauss_iso(12.0)                           # full extent
    Wf[5, LUM]   = ring(3, 7)                                # inner ring
    Wf[6, LUM]   = ring(6, 12)                               # outer ring
    Wf[7, LUM]   = gauss_iso(2.0) - gauss_iso(6.0)           # concentration
    Wf[8, LUM]   = gauss_iso(6.0) - gauss_iso(2.0)           # diffuse
    Wf[9, BLUE]  = gauss_iso(3.0)                            # blue centre
    Wf[10, BLUE] = ring(2, 7)                                # blue ring
    Wf[11, BLUE] = gauss_iso(7.0)                            # blue extended
    Wf[12, RED]  = gauss_iso(3.0)                            # red centre
    Wf[13, RED]  = gauss_iso(7.0)                            # red extended
    Wf[14, BLUE] = gauss_iso(5.0)
    Wf[14, RED]  = -gauss_iso(5.0)                           # blue-dominant indicator
    Wf[15, LUM]  = gauss_iso(1.0) - gauss_iso(3.0)           # sharp vs halo

    weights['feats.W'] = Wf
    weights['feats.b'] = bf

    # 8 oriented narrow Gaussians on luminance
    Wo = np.zeros((8, 3, 32, 32), dtype=np.float32)
    for k in range(8):
        angle = k * np.pi / 8.0
        Wo[k, LUM] = gauss_oriented(smajor=8.0, sminor=1.5, angle=angle)
    weights['oriented.W'] = Wo
    weights['oriented.b'] = np.zeros(8, dtype=np.float32)

    # Dense A: 16 features -> 10 classes
    WA = np.zeros((10, F), dtype=np.float32)
    bA = np.zeros(10, dtype=np.float32)

    # 0: disturbed -- hard to detect by hand
    WA[0, 15] = +0.15
    bA[0]     = -0.2

    # 1: merging -- extended light, less concentrated; often reddish
    WA[1, 4]  = +0.5
    WA[1, 8]  = +0.3
    WA[1, 12] = +0.3
    bA[1]     = +0.0

    # 2: round smooth -- big red elliptical (modal class)
    WA[2, 3]  = +1.3
    WA[2, 12] = +1.1
    WA[2, 14] = -2.5
    bA[2]     = +0.45

    # 3: in-between round smooth -- medium red, more compact than 2
    WA[3, 1]  = +0.25
    WA[3, 12] = +0.6
    WA[3, 14] = -1.5
    bA[3]     = +0.15

    # 4: cigar-shaped smooth -- small, red, moderately elongated
    WA[4, 1]  = +0.2
    WA[4, 12] = +0.3
    WA[4, 14] = -0.6
    bA[4]     = -0.1

    # 5: barred spiral -- blue, extended with structure
    WA[5, 10] = +1.5
    WA[5, 14] = +2.3
    WA[5, 11] = +0.4
    bA[5]     = +0.3

    # 6: unbarred tight spiral -- compact blue core
    WA[6, 9]  = +2.5
    WA[6, 14] = +0.8
    bA[6]     = -0.1

    # 7: unbarred loose spiral -- blue extended with ring
    WA[7, 10] = +2.0
    WA[7, 11] = +1.0
    WA[7, 14] = +1.2
    bA[7]     = 0.0

    # 8: edge-on no bulge -- strongly elongated, no bright centre
    WA[8, 0]  = -0.3
    WA[8, 12] = -0.5
    bA[8]     = +0.7

    # 9: edge-on with bulge -- elongated + strong red central bulge
    WA[9, 12] = +2.2
    WA[9, 0]  = +0.1
    bA[9]     = -1.6

    weights['dense_a.W'] = WA
    weights['dense_a.b'] = bA

    # Dense max: (max over orientations) -> 10
    # Elongation = max - avg; strong boost for edge-on / cigar
    Wmax = np.zeros((10, 1), dtype=np.float32)
    Wmax[4, 0] = +3.0
    Wmax[8, 0] = +4.0
    Wmax[9, 0] = +2.2
    Wmax[2, 0] = -1.0
    Wmax[3, 0] = -0.4
    Wmax[1, 0] = +0.3
    weights['dense_max.W'] = Wmax
    weights['dense_max.b'] = np.zeros(10, dtype=np.float32)

    # Dense avg: (mean over orientations) -> 10
    # Elongated classes pair with opposite sign so contribution is X*(max-avg)
    Wavg = np.zeros((10, 1), dtype=np.float32)
    Wavg[4, 0] = -3.0
    Wavg[8, 0] = -3.0
    Wavg[9, 0] = -1.5
    Wavg[2, 0] = +0.8
    Wavg[3, 0] = +0.5
    Wavg[5, 0] = +0.3
    Wavg[7, 0] = +0.3
    Wavg[1, 0] = +0.2
    weights['dense_avg.W'] = Wavg
    weights['dense_avg.b'] = np.zeros(10, dtype=np.float32)

    np.savez(os.path.join(out_dir, "weights.npz"), **weights)


if __name__ == "__main__":
    main()
