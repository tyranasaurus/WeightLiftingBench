"""Build network.json and weights.npz for ASL letter classifier.

Hand-designed oriented-edge network. Four Sobel-like filters at 4 orientations
compute edge energy; pos/neg + ReLU gives absolute gradient; 4x4 avg pool gives
one orientation-vector per cell. A dense layer with carefully designed per-class
weights classifies the sign.

The 4 Sobel-like 3x3 filters have measurable cross-orientation response:
  Vertical edge:   V=0.50 H=0.00 D1=0.38 D2=0.38
  Horizontal edge: V=0.00 H=0.50 D1=0.38 D2=0.38
  backslash edge:  V=0.38 H=0.38 D1=0.50 D2=0.00
  slash edge:      V=0.38 H=0.38 D1=0.00 D2=0.50

So "vertical edges" also light up D1 and D2 at 76% strength. Patterns are
designed with this in mind: letters with specific vertical fingers use only
the V channel (not D1/D2). Letters with diagonal content use D1 and/or D2.
Letters with horizontal content use H.

Design principles:
 - Wrist row (row 3) is zeroed out -- it is a constant across classes.
 - Positive weights where a letter expects edge energy.
 - Weights by orientation are kept orthogonal to avoid confusion:
   a pattern either uses V OR D1/D2 in a given area, not both.
 - Patterns NOT L2-normalized to preserve carefully-chosen weight magnitudes.
"""
import sys, json, os
import numpy as np

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/app/out"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- 4 oriented 3x3 Sobel-like filters --------
S_V  = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],   dtype=np.float32) / 8.0
S_H  = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],   dtype=np.float32) / 8.0
S_D1 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],   dtype=np.float32) / 8.0  # backslash edge
S_D2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]],   dtype=np.float32) / 8.0  # slash edge
ORIENT_FILTERS = np.stack([S_V, S_H, S_D1, S_D2])[:, None, :, :]  # (4, 1, 3, 3)

V, HO, D1, D2 = 0, 1, 2, 3
ROWS, COLS, ORIENTS = 4, 4, 4

def zeros():
    return np.zeros((ORIENTS, ROWS, COLS), np.float32)

# -------- 24 letter patterns --------

def pat_A():
    w = zeros()
    # fist, no fingers up; small thumb bump on left
    w[V, 0, 1] -= 0.4; w[V, 0, 2] -= 0.4
    w[HO, 1, 1] += 0.3; w[HO, 1, 2] += 0.3
    w[V, 2, 0] += 0.3
    return w

def pat_B():
    """4 fingers up, spread wide: V at all top-row cols."""
    w = zeros()
    w[V, 0, 0] += 0.6; w[V, 0, 1] += 0.8; w[V, 0, 2] += 0.8; w[V, 0, 3] += 0.6
    w[V, 1, 0] += 0.4; w[V, 1, 1] += 0.6; w[V, 1, 2] += 0.6; w[V, 1, 3] += 0.4
    return w

def pat_C():
    """Curved: strong diagonals at corners, strong vertical on one side."""
    w = zeros()
    w[V, 0, 1] -= 0.4; w[V, 0, 2] -= 0.4
    # Top of C curves: / at top-left, \ at top-right of C
    w[D2, 0, 0] += 0.4; w[D2, 0, 1] += 0.4  # top-left curve (/)
    w[D1, 0, 2] += 0.4; w[D1, 0, 3] += 0.4  # top-right curve (\)
    # Bottom of C curves: inverted
    w[D1, 2, 0] += 0.4; w[D1, 2, 1] += 0.4  # bottom-left
    w[D2, 2, 2] += 0.4; w[D2, 2, 3] += 0.4  # bottom-right
    return w

def pat_D():
    """Single index finger up (center): V at top-center only."""
    w = zeros()
    w[V, 0, 1] += 0.8; w[V, 0, 2] += 0.8
    w[V, 1, 1] += 0.6; w[V, 1, 2] += 0.6
    w[V, 0, 0] -= 0.4; w[V, 0, 3] -= 0.4
    # ring at bottom
    w[HO, 2, 1] += 0.3; w[HO, 2, 2] += 0.3
    return w

def pat_E():
    """Compact fist with thumb across front. Extra horizontal middle."""
    w = zeros()
    w[V, 0, 1] -= 0.5; w[V, 0, 2] -= 0.5
    w[HO, 1, 1] += 0.4; w[HO, 1, 2] += 0.4
    w[HO, 2, 1] += 0.4; w[HO, 2, 2] += 0.4
    return w

def pat_F():
    """3 fingers up (right bias) + loop at bottom-left."""
    w = zeros()
    w[V, 0, 1] += 0.5; w[V, 0, 2] += 0.7; w[V, 0, 3] += 0.5
    w[V, 1, 1] += 0.3; w[V, 1, 2] += 0.5; w[V, 1, 3] += 0.3
    w[V, 0, 0] -= 0.3  # no finger far left
    # loop at bottom-left  -- round shape
    w[HO, 2, 0] += 0.4; w[D1, 2, 0] += 0.2; w[D2, 2, 0] += 0.2
    return w

def pat_G():
    """Horizontal hand: index extends sideways. H dominant."""
    w = zeros()
    w[V, 0, 1] -= 0.5; w[V, 0, 2] -= 0.5
    w[V, 0, 0] -= 0.3; w[V, 0, 3] -= 0.3
    w[HO, 1, 0] += 0.5; w[HO, 1, 1] += 0.6; w[HO, 1, 2] += 0.6; w[HO, 1, 3] += 0.5
    return w

def pat_H():
    """Horizontal, 2 fingers extending to side. Stronger H than G."""
    w = zeros()
    w[V, 0, 1] -= 0.5; w[V, 0, 2] -= 0.5
    w[HO, 1, 0] += 0.5; w[HO, 1, 1] += 0.7; w[HO, 1, 2] += 0.7; w[HO, 1, 3] += 0.5
    w[HO, 2, 0] += 0.4; w[HO, 2, 1] += 0.6; w[HO, 2, 2] += 0.6; w[HO, 2, 3] += 0.4
    return w

def pat_I():
    """Fist with pinky on RIGHT: V at top-right only."""
    w = zeros()
    w[V, 0, 2] += 0.6; w[V, 0, 3] += 0.8
    w[V, 1, 2] += 0.4; w[V, 1, 3] += 0.6
    w[V, 0, 0] -= 0.5; w[V, 0, 1] -= 0.5
    return w

def pat_K():
    """Index up at left-center + middle finger angled diagonally."""
    w = zeros()
    w[V, 0, 1] += 0.6
    w[V, 1, 1] += 0.4
    # Diagonal \ in top-right (angled middle finger)
    w[D1, 0, 2] += 0.5; w[D1, 0, 3] += 0.4
    w[V, 0, 2] -= 0.3; w[V, 0, 3] -= 0.3
    return w

def pat_L():
    """Index up (center-left) + thumb horizontal to the side."""
    w = zeros()
    w[V, 0, 1] += 0.7
    w[V, 1, 1] += 0.5
    w[HO, 2, 1] += 0.5; w[HO, 2, 2] += 0.5; w[HO, 2, 3] += 0.4
    w[V, 0, 2] -= 0.4; w[V, 0, 3] -= 0.4
    return w

def pat_M():
    """Fist with 3 knuckles on top: horizontal edges + weak V bumps."""
    w = zeros()
    w[HO, 0, 1] += 0.5; w[HO, 0, 2] += 0.5
    w[V, 0, 1] += 0.1; w[V, 0, 2] += 0.1
    w[HO, 1, 1] += 0.3; w[HO, 1, 2] += 0.3
    return w

def pat_N():
    """Fist with 2 knuckles: weaker horizontal top than M."""
    w = zeros()
    w[HO, 0, 1] += 0.4; w[HO, 0, 2] += 0.4
    w[HO, 1, 1] += 0.3; w[HO, 1, 2] += 0.3
    return w

def pat_O():
    """Ring: curved boundary all around. Horizontal top and bottom."""
    w = zeros()
    w[V, 0, 1] -= 0.5; w[V, 0, 2] -= 0.5
    w[HO, 0, 1] += 0.3; w[HO, 0, 2] += 0.3
    w[HO, 2, 1] += 0.4; w[HO, 2, 2] += 0.4
    # diagonal curves at corners of ring
    w[D1, 1, 0] += 0.3; w[D2, 1, 3] += 0.3
    w[D2, 2, 0] += 0.3; w[D1, 2, 3] += 0.3
    return w

def pat_P():
    """Pointing DOWN like K. Palm on top. Index points down."""
    w = zeros()
    w[V, 0, 1] -= 0.5; w[V, 0, 2] -= 0.5
    w[HO, 0, 1] += 0.4; w[HO, 0, 2] += 0.4
    # Vertical index pointing down (row 2)
    w[V, 2, 1] += 0.4; w[V, 2, 2] += 0.4
    # Diagonal \ for angled middle finger
    w[D1, 2, 2] += 0.3; w[D1, 2, 3] += 0.3
    return w

def pat_Q():
    """Pointing DOWN like G. Palm on top. Two parallel down-pointers."""
    w = zeros()
    w[V, 0, 1] -= 0.5; w[V, 0, 2] -= 0.5
    w[HO, 0, 1] += 0.4; w[HO, 0, 2] += 0.4
    w[V, 2, 1] += 0.5; w[V, 2, 2] += 0.5
    return w

def pat_R():
    """Crossed fingers: BOTH diagonals strong in top center."""
    w = zeros()
    # NO vertical weight -- crossed fingers produce diagonal edges
    w[D1, 0, 1] += 0.5; w[D1, 0, 2] += 0.5
    w[D2, 0, 1] += 0.5; w[D2, 0, 2] += 0.5
    w[D1, 1, 1] += 0.3; w[D2, 1, 2] += 0.3
    return w

def pat_S():
    """Compact fist, thumb across front."""
    w = zeros()
    w[V, 0, 1] -= 0.5; w[V, 0, 2] -= 0.5
    w[HO, 0, 1] += 0.4; w[HO, 0, 2] += 0.4  # top of fist is slightly flat
    w[HO, 1, 1] += 0.3; w[HO, 1, 2] += 0.3
    return w

def pat_T():
    """Fist with small thumb tip at top center."""
    w = zeros()
    w[V, 0, 1] += 0.4; w[V, 0, 2] += 0.4
    w[V, 0, 0] -= 0.5; w[V, 0, 3] -= 0.5
    w[HO, 1, 1] += 0.3; w[HO, 1, 2] += 0.3
    return w

def pat_U():
    """Index + middle up and TOGETHER: V in center top."""
    w = zeros()
    w[V, 0, 1] += 0.8; w[V, 0, 2] += 0.8
    w[V, 1, 1] += 0.6; w[V, 1, 2] += 0.6
    w[V, 0, 0] -= 0.5; w[V, 0, 3] -= 0.5
    return w

def pat_V():
    """Index + middle in a V shape: diagonals diverging."""
    w = zeros()
    # Left finger goes up and left -> \ edge at top-left
    w[D1, 0, 1] += 0.5; w[D1, 1, 1] += 0.3
    # Right finger goes up and right -> / edge at top-right
    w[D2, 0, 2] += 0.5; w[D2, 1, 2] += 0.3
    # NO straight vertical weight -- real V shape fingers are angled
    w[V, 0, 0] -= 0.4; w[V, 0, 3] -= 0.4
    return w

def pat_W():
    """3 fingers up: V across cols 0-3."""
    w = zeros()
    w[V, 0, 0] += 0.5; w[V, 0, 1] += 0.7; w[V, 0, 2] += 0.7; w[V, 0, 3] += 0.5
    w[V, 1, 0] += 0.3; w[V, 1, 1] += 0.5; w[V, 1, 2] += 0.5; w[V, 1, 3] += 0.3
    # W is like B but slightly narrower; distinguish by weaker outer col
    return w

def pat_X():
    """Fist with bent index. Horizontal hook top, weak vertical shaft."""
    w = zeros()
    w[HO, 0, 1] += 0.5; w[HO, 0, 2] += 0.5
    w[V, 1, 1] += 0.2; w[V, 1, 2] += 0.2
    w[V, 0, 0] -= 0.3; w[V, 0, 3] -= 0.3
    return w

def pat_Y():
    """Thumb + pinky extended outward. Diagonals away from fist."""
    w = zeros()
    w[V, 0, 1] -= 0.3; w[V, 0, 2] -= 0.3
    # Pinky going up-right: / edge in top-right -> D2 at (0, 2-3)
    w[D2, 0, 2] += 0.4; w[D2, 0, 3] += 0.5
    # Thumb going down-left: / edge in middle-left (since thumb comes from palm going down) -- actually use D1 or D2
    # Thumb from palm (center) to lower-left = \ edge -> D1 at (2, 0) or (1, 0)
    w[D1, 1, 0] += 0.3; w[D1, 2, 0] += 0.4
    return w

LETTERS = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
PATTERN_FNS = [pat_A, pat_B, pat_C, pat_D, pat_E, pat_F, pat_G, pat_H, pat_I, pat_K,
               pat_L, pat_M, pat_N, pat_O, pat_P, pat_Q, pat_R, pat_S, pat_T, pat_U,
               pat_V, pat_W, pat_X, pat_Y]

def build_class_weights():
    patterns = np.stack([fn() for fn in PATTERN_FNS])  # (24, 4, 4, 4)
    patterns[:, :, 3, :] = 0.0  # wrist row zero
    W = patterns.reshape(24, -1).astype(np.float32)
    # Zero mean per class
    W = W - W.mean(axis=1, keepdims=True)
    # NOT L2 normalized - keep designed magnitudes
    b = np.zeros(24, np.float32)
    return W, b

NETWORK = {
    "layers": [
        {"name": "n1",     "op": "norm",    "inputs": ["image"]},
        {"name": "g_pos",  "op": "conv",    "inputs": ["n1"], "out": 4, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "r_pos",  "op": "relu",    "inputs": ["g_pos"]},
        {"name": "g_neg",  "op": "conv",    "inputs": ["n1"], "out": 4, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "r_neg",  "op": "relu",    "inputs": ["g_neg"]},
        {"name": "edges",  "op": "add",     "inputs": ["r_pos", "r_neg"]},
        {"name": "pooled", "op": "avgpool", "inputs": ["edges"], "size": [7, 7], "stride": [7, 7]},
        {"name": "flat",   "op": "reshape", "inputs": ["pooled"], "shape": [64]},
        {"name": "out",    "op": "dense",   "inputs": ["flat"], "out": 24},
    ]
}

def main():
    W_cls, b_cls = build_class_weights()
    gW = ORIENT_FILTERS.astype(np.float32)
    weights = {
        "g_pos.W": gW,
        "g_pos.b": np.zeros(4, np.float32),
        "g_neg.W": (-gW).astype(np.float32),
        "g_neg.b": np.zeros(4, np.float32),
        "out.W":   W_cls,
        "out.b":   b_cls,
    }
    with open(os.path.join(OUT_DIR, "network.json"), "w") as f:
        json.dump(NETWORK, f)
    np.savez(os.path.join(OUT_DIR, "weights.npz"), **weights)
    print(f"wrote network.json and weights.npz to {OUT_DIR}")

if __name__ == "__main__":
    main()
