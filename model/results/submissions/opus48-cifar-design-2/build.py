"""build.py -- writes network.json and weights.npz for a hand-designed CIFAR-10 classifier.

No training data is read and no fitting is done. Every number is either a fixed image filter
(colour-opponent combinations, Sobel / Laplacian stencils) or part of a hand-written rule table
that says, from general knowledge of these ten everyday objects, whether a class tends to score
high (+) or low (-) on each of nine interpretable cues:

  cues  = [lumT, lumB, grn, grnBot, bluT, bluB, redC, sat, tex]
          brightness top / bottom, greenness overall / at the bottom (grass), blueness at
          top / bottom (sky & water), warm colour in the centre, overall colourfulness, and
          high-frequency texture energy.

The rule table R (10 classes x 9 cues) is my own small integer judgement (e.g. planes and ships
sit in blue sky so bluT = +2; frogs are warm-green, dark and busy; cars/trucks are grey and
edgy). It is turned into a classifier in a self-balancing way:
  * subtract each cue's mean over classes  -> every cue is zero-sum, so no cue biases the vote;
  * standardise each cue by a general-knowledge neutral point mu and spread s
    (gray-world prior: colour opponents mu=0, luminance mu=0.5);
  * normalise each class's template to unit length, so no class wins just by having bigger
    numbers.  score_c = unit_template_c . standardised_cues .
The cues are computed by fixed conv + pool layers; three dense branches (colour, colourfulness,
texture) each emit a (10,) score and are summed.
"""
import sys, json, os
import numpy as np

NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CUES = ["lumT", "lumB", "grn", "grnBot", "bluT", "bluB", "redC", "sat", "tex"]

# --- hand-written rule table: class x cue, each entry in {-2,-1,0,1,2} from general knowledge ---
#              lumT lumB  grn grnB bluT bluB redC  sat  tex
R = np.array([
    [ 2,  0,  0,  0,  2,  1, -2,  1, -2],   # airplane : bright blue sky, cool, smooth
    [ 0, -1, -1, -1,  0,  0,  0, -2,  2],   # automobile: grey, edgy, little green/blue
    [ 0,  1,  1,  1, -1, -1,  0,  1, -2],   # bird : greenish/varied, smooth sky, colourful
    [-1,  1,  0,  0, -1,  0,  1, -1,  0],   # cat : neutral, warmish, indoor
    [-1,  1,  2,  2, -1, -2,  1,  1,  0],   # deer : very green (grass), brown body, no blue
    [-1,  1,  0,  0, -1, -1,  1, -1,  0],   # dog : warm neutral fur
    [-2, -1,  1,  1, -2, -1,  2,  2,  1],   # frog : dark, warm, colourful, no blue, busy skin
    [ 0,  1,  1,  1, -1, -2,  1,  0,  1],   # horse : brown on grass, no blue at bottom
    [ 2, -1,  0,  0,  2,  2, -1,  0, -1],   # ship : bright blue sky and water, dark bottom, cool
    [ 1, -1, -1, -1,  0,  0,  0, -2,  2],   # truck : bright top, grey, edgy
], dtype=np.float64)

# general-knowledge neutral point (mu) and spread (s) for each cue -- NOT measured from the data.
# Colour opponents are neutral at 0 (gray-world prior), luminance at mid-grey 0.5; colourfulness
# and texture at round typical levels for natural photos in [0,1].
MU = np.array([0.50, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.10])
S  = np.array([0.12, 0.12, 0.04, 0.05, 0.07, 0.07, 0.07, 0.06, 0.04])

GRID = 6          # pooled resolution (6x6 cells over the 32x32 image)
POOL = 5          # avgpool window / stride -> 6x6

# ----------------------------------------------------------------------------- architecture
def make_network():
    P = {"size": [POOL, POOL], "stride": [POOL, POOL]}
    return {"layers": [
        {"name": "opp",  "op": "conv", "inputs": ["image"], "out": 4, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "cpool", "op": "avgpool", "inputs": ["opp"], **P},
        {"name": "oppr", "op": "conv", "inputs": ["image"], "out": 6, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "oppr_r", "op": "relu", "inputs": ["oppr"]},
        {"name": "spool", "op": "avgpool", "inputs": ["oppr_r"], **P},
        {"name": "edge", "op": "conv", "inputs": ["image"], "out": 6, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "erl",  "op": "relu", "inputs": ["edge"]},
        {"name": "tpool", "op": "avgpool", "inputs": ["erl"], **P},
        {"name": "scoreC", "op": "dense", "inputs": ["cpool"], "out": 10},
        {"name": "scoreS", "op": "dense", "inputs": ["spool"], "out": 10},
        {"name": "scoreT", "op": "dense", "inputs": ["tpool"], "out": 10},
        {"name": "out", "op": "add", "inputs": ["scoreC", "scoreS", "scoreT"]},
    ]}

# ----------------------------------------------------------------------------- fixed filters
def opp_weights():
    W = np.array([[1/3, 1/3, 1/3], [1., -.5, -.5], [-.5, 1., -.5], [-.5, -.5, 1.]])[:, :, None, None]
    return W, np.zeros(4)

def oppr_weights():
    base = [[1., -.5, -.5], [-1., .5, .5], [-.5, 1., -.5], [.5, -1., .5], [-.5, -.5, 1.], [.5, .5, -1.]]
    return np.array(base)[:, :, None, None], np.zeros(6)

def edge_weights():
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8.0
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float) / 8.0
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float) / 4.0
    stencils = [gx, -gx, gy, -gy, lap, -lap]
    W = np.zeros((6, 3, 3, 3))
    for o, s in enumerate(stencils):
        for i in range(3):
            W[o, i] = s / 3.0            # applied to luminance = mean of RGB
    return W, np.zeros(6)

# ----------------------------------------------------------------------------- cue functionals
def region_mask(rows, cols):
    m = np.zeros((GRID, GRID))
    for r in rows:
        for c in cols:
            m[r, c] = 1.0
    return m

ALL = list(range(GRID)); TOP = [0, 1]; BOT = [4, 5]; CEN = [2, 3]

def colour_functionals():
    """linear functional (coeff over cpool (4,GRID,GRID)) for each colour cue -> value = mean over a region."""
    def f(ch, rows, cols):
        a = np.zeros((4, GRID, GRID)); m = region_mask(rows, cols); a[ch] = m / m.sum(); return a
    return {
        "lumT": f(0, TOP, ALL), "lumB": f(0, BOT, ALL),
        "grn": f(2, ALL, ALL), "grnBot": f(2, BOT, ALL),
        "bluT": f(3, TOP, ALL), "bluB": f(3, BOT, ALL),
        "redC": f(1, CEN, CEN),
    }

def sat_functional():
    a = np.ones((6, GRID, GRID)); return a / (GRID * GRID)      # mean over all cells & 6 rectified channels

def tex_functional():
    a = np.zeros((6, GRID, GRID)); a[4:6] = 1.0; return a / (GRID * GRID)  # |laplacian| channels

# ----------------------------------------------------------------------------- readout
def build_weights():
    wd = {}
    wd["opp.W"], wd["opp.b"] = opp_weights()
    wd["oppr.W"], wd["oppr.b"] = oppr_weights()
    wd["edge.W"], wd["edge.b"] = edge_weights()

    Rc = R - R.mean(axis=0, keepdims=True)            # zero-sum each cue
    norms = np.linalg.norm(Rc, axis=1, keepdims=True)
    U = Rc / np.where(norms > 0, norms, 1.0)          # unit template per class

    cf = colour_functionals()
    cidx = [CUES.index(k) for k in ["lumT", "lumB", "grn", "grnBot", "bluT", "bluB", "redC"]]
    sidx = CUES.index("sat"); tidx = CUES.index("tex")
    sf = sat_functional(); tf = tex_functional()

    WC = np.zeros((10, 4 * GRID * GRID)); bC = np.zeros(10)
    WS = np.zeros((10, 6 * GRID * GRID)); bS = np.zeros(10)
    WT = np.zeros((10, 6 * GRID * GRID)); bT = np.zeros(10)
    for c in range(10):
        acc = np.zeros((4, GRID, GRID))
        for j in cidx:
            acc += (U[c, j] / S[j]) * cf[CUES[j]]
        WC[c] = acc.reshape(-1)
        WS[c] = ((U[c, sidx] / S[sidx]) * sf).reshape(-1)
        WT[c] = ((U[c, tidx] / S[tidx]) * tf).reshape(-1)
        # whole-classifier bias so a neutral image (cue == mu) scores zero
        bC[c] = -float(np.sum(U[c] * MU / S))
    wd["scoreC.W"], wd["scoreC.b"] = WC, bC
    wd["scoreS.W"], wd["scoreS.b"] = WS, bS
    wd["scoreT.W"], wd["scoreT.b"] = WT, bT
    return wd

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(make_network(), f, indent=2)
    wd = {k: np.asarray(v, dtype=np.float32) for k, v in build_weights().items()}
    np.savez(os.path.join(out_dir, "weights.npz"), **wd)

if __name__ == "__main__":
    main()
