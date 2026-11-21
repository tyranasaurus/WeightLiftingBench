"""
build.py -- writes network.json and weights.npz for a hand-designed CIFAR-10 classifier.

NO training, no fitting, no data is read.  Every number is chosen by hand from general
knowledge of what these ten things look like in small 32x32 photos.

Design: a COLOUR voting network.  Colour-opponent channels are zero-centred, so a ReLU at 0
is a natural, self-calibrating threshold (no data-dependent constant needed).  I deliberately
use no edge/texture detector: its scale cannot be set without measuring the data.

  image(3,32,32)
   color : 1x1 conv -> LUM, BLU(blue-excess), GRN(green-excess), RMG(red-excess)
   avgpool to a 4x4 grid of regions (top/bottom/centre/surround = groups of cells)
   -> 9 ReLU "evidence detectors", each a self-calibrating comparison:
        bluesky      = relu(mean BLU over the top rows)            blue sky overhead
        bluewater    = relu(mean BLU over the bottom rows)         blue water below
        brighttop    = relu(mean LUM over the top rows - 0.5)      bright sky over object
        greenground  = relu(mean GRN over the bottom rows)         grass / foliage below
        greenall     = relu(mean GRN over the whole frame)         green everywhere (e.g. frog)
        animalongrass= relu(mean GRN surround - mean GRN centre)   non-green body on green grass
        redobj       = relu(mean RMG over the centre)              warm / red object in centre
        warm         = relu(-mean BLU over the whole frame)        warm, not blue (indoor animal)
        dark         = relu(0.46 - mean LUM)                       dark overall (e.g. frog)
   -> a vote layer: each class adds a balanced budget of +votes from the detectors that
      support it and a few -votes from detectors that argue against it.

Why this is balanced and cannot collapse: every detector output is >=0 and bounded, and every
class is given a similar total +vote budget, so the prediction is decided by WHICH detectors
fire, not by one class having huge weights.  Detector gains come from the natural size of each
formula (opponent colours are a few tenths at most; luminance is 0..1 with 0.5 mid-grey).
"""
import sys, json, os
import numpy as np

# ----------------------------------------------------------------------------- colour conv
LUMW = [0.299, 0.587, 0.114]
color_W = np.zeros((4, 3, 1, 1), np.float32)
color_W[0, :, 0, 0] = LUMW                 # 0 LUM
color_W[1, :, 0, 0] = [-0.5, -0.5, 1.0]    # 1 BLU blue-excess
color_W[2, :, 0, 0] = [-0.5, 1.0, -0.5]    # 2 GRN green-excess
color_W[3, :, 0, 0] = [1.0, -0.5, -0.5]    # 3 RMG red-excess
color_b = np.zeros(4, np.float32)

# ----------------------------------------------------------------------------- cell groups (4x4 grid)
TOP = [(r, c) for r in (0, 1) for c in range(4)]
BOT = [(r, c) for r in (2, 3) for c in range(4)]
ALL = [(r, c) for r in range(4) for c in range(4)]
CEN = [(r, c) for r in (1, 2) for c in (1, 2)]
SUR = [rc for rc in ALL if rc not in CEN]

NEXP = 9
Ec = np.zeros((NEXP, 4, 4, 4), np.float32)
eb = np.zeros(NEXP, np.float32)
LUM, BLU, GRN, RMG = 0, 1, 2, 3


def addc(e, ch, cells, coef):
    for (r, c) in cells:
        Ec[e, ch, r, c] += coef


# 0 bluesky = relu(mean BLU top) * 7
addc(0, BLU, TOP, 7.0 / len(TOP))
# 1 bluewater = relu(mean BLU bottom) * 7
addc(1, BLU, BOT, 7.0 / len(BOT))
# 2 brighttop = relu(mean LUM top - 0.5) * 5
addc(2, LUM, TOP, 5.0 / len(TOP)); eb[2] = -5.0 * 0.5
# 3 greenground = relu(mean GRN bottom) * 7
addc(3, GRN, BOT, 7.0 / len(BOT))
# 4 greenall = relu(mean GRN all) * 7
addc(4, GRN, ALL, 7.0 / len(ALL))
# 5 animalongrass = relu(mean GRN surround - mean GRN centre) * 7
addc(5, GRN, SUR, 7.0 / len(SUR)); addc(5, GRN, CEN, -7.0 / len(CEN))
# 6 redobj = relu(mean RMG centre) * 6
addc(6, RMG, CEN, 6.0 / len(CEN))
# 7 warm = relu(-mean BLU all) * 5
addc(7, BLU, ALL, -5.0 / len(ALL))
# 8 dark = relu(0.46 - mean LUM all) * 4
addc(8, LUM, ALL, -4.0 / len(ALL)); eb[8] = 4.0 * 0.46

# ----------------------------------------------------------------------------- vote table
# detectors: 0 bluesky 1 bluewater 2 brighttop 3 greenground 4 greenall
#            5 animalongrass 6 redobj 7 warm 8 dark
VOTES = {
 'airplane': {0: 1.3, 2: 1.0, 4: -0.8, 7: -0.5},
 'auto'    : {6: 1.0, 2: 0.4, 4: -0.9, 1: -0.6, 7: -0.2},
 'bird'    : {0: 0.6, 3: 0.6, 6: 0.4, 5: 0.4, 2: 0.3},
 'cat'     : {7: 0.8, 6: 0.8, 4: -0.5, 1: -0.4},
 'deer'    : {5: 1.3, 3: 0.7, 6: 0.5, 2: 0.4, 1: -0.4},
 'dog'     : {7: 0.8, 6: 0.7, 3: 0.3, 1: -0.4},
 'frog'    : {4: 1.3, 8: 0.8, 3: 0.4, 2: -0.5, 0: -0.4},
 'horse'   : {5: 1.3, 3: 0.6, 6: 0.5, 8: 0.4, 1: -0.4},
 'ship'    : {1: 1.5, 0: 0.6, 2: 0.5, 4: -0.8, 7: -0.5},
 'truck'   : {2: 0.8, 6: 0.6, 0: 0.3, 4: -0.9, 1: -0.6, 3: -0.5},
}
ORDER = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
vote_W = np.zeros((10, NEXP), np.float32)
for k, cls in enumerate(ORDER):
    for j, w in VOTES[cls].items():
        vote_W[k, j] = w
vote_b = np.zeros(10, np.float32)

det_W = Ec.reshape(NEXP, -1).astype(np.float32)   # (9, 64)
det_b = eb.astype(np.float32)

layers = [
    {"name": "color", "op": "conv",    "inputs": ["image"], "out": 4, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "poolc", "op": "avgpool", "inputs": ["color"], "size": [8, 8], "stride": [8, 8]},
    {"name": "det",   "op": "dense",   "inputs": ["poolc"], "out": NEXP},
    {"name": "drelu", "op": "relu",    "inputs": ["det"]},
    {"name": "out",   "op": "dense",   "inputs": ["drelu"], "out": 10},
]
weights = {
    "color.W": color_W, "color.b": color_b,
    "det.W": det_W, "det.b": det_b,
    "out.W": vote_W, "out.b": vote_b,
}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=2)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in weights.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
