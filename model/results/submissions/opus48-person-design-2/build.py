"""
build.py -- regenerates network.json and weights.npz for a hand-designed
person / no-person classifier.  NO training, NO data: every number below comes
from formulas and from general knowledge about photographs of people.

What the network measures (all from general knowledge, nothing fitted):

1. FINE CENTRAL DETAIL.  A person is the sharp, high-contrast subject of a photo
   (faces with eyes, hands, textured clothing, skin highlights) and sits in the
   centre of the frame.  Many "no person" photos are smooth / low-contrast
   (sky, sea, fog, plain walls, empty rooms, blurred backgrounds).  So we run a
   bank of small luminance centre-surround contrast detectors (horizontal and
   vertical, two scales, both polarities), keep the strongest response in each
   local 12x12 cell (max-pool = "a sharp feature exists here"), and add up that
   detail over a centre-weighted (slightly upper) region.  High central detail
   -> person.

   The decision threshold T is grounded in a general property of natural images:
   typical local luminance contrast is about 0.2, and summing the 8 detectors
   gives a typical central-detail level near 0.2 * 8 ~= 1.6; a person makes the
   centre sharper than a typical scene, so the boundary is set just below that,
   T = 1.3.  (It is a hand-chosen threshold, not fitted to the data.)

2. GREENNESS tie-breaker.  Foliage, grass, food and many natural "no person"
   scenes are green (G > B); people / indoor / urban scenes much less so.  Mean
   relu(G - B) therefore votes toward "no person" with a modest weight g = 6.

Decision:  person  iff  central_detail  >  T + g * greenness.

Nothing here reads the training images or labels; the constants are design
choices from the ideas above.
"""
import sys, os, json
import numpy as np

LUM = np.array([0.299, 0.587, 0.114], dtype=np.float64)  # standard luminance

KS = 15          # square conv kernel (square is required by the interpreter)
POOL = 12        # max-pool window / stride -> 96/12 = 8x8 grid
GRID = 96 // POOL
T = 1.3          # detail threshold (natural-image contrast level, hand chosen)
GW = 6.0         # greenness vote weight
CY, CX = 3.2, 3.5  # centre of the 8x8 grid, nudged slightly upward (faces)
SIG = 1.6          # central Gaussian width


def band_kernel(center, flanks, rows, horizontal=True):
    """Separable centre-surround band: a bright 3-wide centre vs two darker
    flank bands (each weighted -0.5), averaged over a 3-wide perpendicular band.
    Zero-sum -> responds to local contrast, not absolute brightness."""
    h = np.zeros(KS)
    h[center] = 1.0 / len(center)
    for fb in flanks:
        h[fb] = -0.5 / len(fb)
    v = np.zeros(KS)
    v[rows] = 1.0 / len(rows)
    k = np.outer(v, h)                 # horizontal centre-surround
    return k if horizontal else k.T


def build_network():
    layers = [
        {"name": "det", "op": "conv", "inputs": ["image"],
         "out": 8, "kernel": [KS, KS], "stride": 1, "padding": KS // 2},
        {"name": "detr", "op": "relu", "inputs": ["det"]},
        {"name": "detp", "op": "maxpool", "inputs": ["detr"],
         "size": [POOL, POOL], "stride": [POOL, POOL]},
        {"name": "detf", "op": "reshape", "inputs": ["detp"], "shape": [8 * GRID * GRID]},
        {"name": "scoreDetail", "op": "dense", "inputs": ["detf"], "out": 2},

        {"name": "col", "op": "conv", "inputs": ["image"],
         "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "colr", "op": "relu", "inputs": ["col"]},
        {"name": "colp", "op": "avgpool", "inputs": ["colr"],
         "size": [96, 96], "stride": [96, 96]},
        {"name": "colf", "op": "reshape", "inputs": ["colp"], "shape": [1]},
        {"name": "scoreCol", "op": "dense", "inputs": ["colf"], "out": 2},

        {"name": "out", "op": "add", "inputs": ["scoreDetail", "scoreCol"]},
    ]
    return {"layers": layers}


def build_weights():
    W = {}

    # ---- conv "det": 8 luminance centre-surround contrast detectors ----
    # small scale: flanks +-3 ; large scale: flanks +-6 ; horizontal & vertical.
    kHs = band_kernel([6, 7, 8], [[3, 4, 5], [9, 10, 11]], [6, 7, 8], horizontal=True)
    kHl = band_kernel([6, 7, 8], [[0, 1, 2], [12, 13, 14]], [6, 7, 8], horizontal=True)
    kVs = band_kernel([6, 7, 8], [[3, 4, 5], [9, 10, 11]], [6, 7, 8], horizontal=False)
    kVl = band_kernel([6, 7, 8], [[0, 1, 2], [12, 13, 14]], [6, 7, 8], horizontal=False)
    bases = [kHs, kHl, kVs, kVl]
    kernels = []
    for k in bases:
        kernels.append(k)       # bright-centre polarity
        kernels.append(-k)      # dark-centre polarity
    det_W = np.zeros((8, 3, KS, KS), dtype=np.float64)
    for o, k in enumerate(kernels):
        for i in range(3):
            det_W[o, i] = LUM[i] * k
    W["det.W"] = det_W
    W["det.b"] = np.zeros(8, dtype=np.float64)

    # ---- dense "scoreDetail": centre-weighted sum of pooled detail ----
    ii, jj = np.mgrid[0:GRID, 0:GRID].astype(np.float64)
    gc = np.exp(-(((ii - CY) ** 2 + (jj - CX) ** 2) / (2 * SIG * SIG)))
    gc = gc / gc.sum()                                   # central mask, sums to 1
    A = np.zeros((2, 8 * GRID * GRID), dtype=np.float64)
    for c in range(8):
        base = c * GRID * GRID
        for i in range(GRID):
            for j in range(GRID):
                A[1, base + i * GRID + j] = gc[i, j]     # person score = central detail
    W["scoreDetail.W"] = A
    W["scoreDetail.b"] = np.array([T, 0.0], dtype=np.float64)   # no-person baseline T

    # ---- conv "col": greenness channel (G - B), 1x1 ----
    col_W = np.zeros((1, 3, 1, 1), dtype=np.float64)
    col_W[0, :, 0, 0] = [0.0, 1.0, -1.0]
    W["col.W"] = col_W
    W["col.b"] = np.zeros(1, dtype=np.float64)

    # ---- dense "scoreCol": greenness votes toward no-person ----
    B = np.zeros((2, 1), dtype=np.float64)
    B[0, 0] = GW
    W["scoreCol.W"] = B
    W["scoreCol.b"] = np.zeros(2, dtype=np.float64)

    return {k: v.astype(np.float32) for k, v in W.items()}


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(build_network(), f, indent=2)
    np.savez(os.path.join(out_dir, "weights.npz"), **build_weights())
    print("wrote network.json and weights.npz to", out_dir)


if __name__ == "__main__":
    main()
