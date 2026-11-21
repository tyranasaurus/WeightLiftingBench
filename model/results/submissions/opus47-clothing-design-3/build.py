"""Builds network.json and weights.npz for the Fashion-MNIST hand-designed classifier.
Usage: python3 build.py OUT_DIR

Architecture:
  image (1,28,28) -> avgpool 2x2/2 -> (1,14,14)
                  -> reshape (196,)
                  -> dense 196 -> 10   (10 class silhouette templates)

All 1970 weights are hand-designed 14x14 silhouette templates, one per class.
Each template emphasises the discriminating regions of its class (trouser legs
and dark centre gap, bag handle hole, dress's strongly-dark sides, boot's
ankle rise on the right, footwear's dark upper half) and leaves cells that
are shared across classes near zero.  The templates are zero-centred and
L2-normalised so overall brightness doesn't bias any class.
"""
from __future__ import annotations
import json, sys, os
import numpy as np

HR = 14
N = HR * HR  # 196


def tplate():
    return np.zeros((HR, HR), np.float32)


def rect(t, r1, r2, c1, c2, val):
    t[r1:r2, c1:c2] += val


# =============================================================================
# 10 class-silhouette templates (14x14).
# Each 14x14 cell = 2x2 patch of the 28x28 input.  "Positive" = reliably
# bright for this class, "negative" = reliably dark, "0" = uncertain/shared.
# =============================================================================
T = [tplate() for _ in range(10)]

# ----- 0: T-shirt --------------------------------------------------------------
# Signature: WIDE shoulders over a NARROW body that reaches near the bottom.
rect(T[0], 1, 3, 2, 12, 1.5)    # wide top rows (shoulders + short sleeves)
rect(T[0], 3, 12, 4, 10, 1.0)   # narrow body
rect(T[0], 3, 12, 0, 3, -1.8)   # mid-rows sides EMPTY (short sleeves)
rect(T[0], 3, 12, 11, 14, -1.8) # mid-rows sides EMPTY
rect(T[0], 0, 1, 0, 14, -0.8)   # very top row empty
rect(T[0], 13, 14, 0, 14, -0.5) # very bottom row empty-ish

# ----- 1: Trouser --------------------------------------------------------------
# Signature: narrow silhouette with a dark CENTRE GAP in the lower half.
# De-emphasise the outside (shared dark with dress etc.) and focus on the gap.
rect(T[1], 1, 4, 4, 10, 1.0)    # waistband
rect(T[1], 4, 13, 3, 6, 1.5)    # left leg (strong)
rect(T[1], 4, 13, 8, 11, 1.5)   # right leg (strong)
rect(T[1], 6, 13, 6, 8, -2.5)   # STRONG CENTRE GAP (distinguisher)
rect(T[1], 4, 13, 0, 3, -1.0)   # mid-rows outer-left dark
rect(T[1], 4, 13, 11, 14, -1.0) # mid-rows outer-right dark
rect(T[1], 0, 1, 0, 14, -0.5)   # top row empty
rect(T[1], 13, 14, 0, 14, -0.3) # bottom row empty
rect(T[1], 0, 4, 0, 3, -0.8)    # top-left corner
rect(T[1], 0, 4, 11, 14, -0.8)  # top-right corner

# ----- 2: Pullover -------------------------------------------------------------
# Signature: broad body throughout; sleeves part of the body down to mid-rows;
# reaches ALL the way to the bottom (wider than shirt's bottom).
rect(T[2], 1, 4, 2, 12, 1.2)    # wide shoulders
rect(T[2], 4, 14, 1, 13, 1.0)   # WIDE body down to the LAST row (incl 13)
rect(T[2], 4, 14, 0, 1, -0.5)   # just outside the body
rect(T[2], 4, 14, 13, 14, -0.5)
rect(T[2], 0, 1, 0, 14, -0.8)   # very top row empty

# ----- 3: Dress ----------------------------------------------------------------
# Signature: narrow central column spanning the FULL height; strongly empty sides.
# Top MUST also have content (narrow shoulders) so pure-bottom items (footwear)
# don't match.
rect(T[3], 1, 13, 5, 9, 1.5)    # narrow central column (strong)
rect(T[3], 0, 4, 5, 9, 1.0)     # narrow top - REQUIRED (vs empty-top footwear)
rect(T[3], 0, 13, 0, 4, -1.5)   # LEFT SIDE STRONGLY DARK (all rows)
rect(T[3], 0, 13, 10, 14, -1.5) # RIGHT SIDE STRONGLY DARK
rect(T[3], 1, 13, 4, 5, 0.3)    # slight ok at col 4
rect(T[3], 1, 13, 9, 10, 0.3)   # slight ok at col 9
rect(T[3], 13, 14, 0, 2, -0.5)  # bottom corners empty
rect(T[3], 13, 14, 12, 14, -0.5)

# ----- 4: Coat -----------------------------------------------------------------
# Signature: large body AND outer columns BRIGHT at mid-rows (full sleeves).
rect(T[4], 1, 4, 2, 12, 1.0)    # top shoulders
rect(T[4], 4, 13, 0, 14, 0.9)   # wide body incl outer cols (long sleeves)
rect(T[4], 4, 10, 0, 2, 0.6)    # emphasise outer-left sleeves (coat specific)
rect(T[4], 4, 10, 12, 14, 0.6)  # outer-right sleeves
rect(T[4], 0, 1, 0, 14, -0.8)   # very top row empty
rect(T[4], 4, 10, 6, 8, -0.5)   # faint centre-opening

# ----- 5: Sandal ---------------------------------------------------------------
# Signature: empty upper half, strappy content extending higher than a sneaker,
# solid sole; distinctive pattern in the foot region.
rect(T[5], 0, 8, 0, 14, -1.3)   # STRONG empty top half
rect(T[5], 8, 12, 1, 13, 0.8)   # strap region (extended reach)
rect(T[5], 8, 11, 5, 9, -0.7)   # center gap in foot area (distinguishes vs sneaker)
rect(T[5], 12, 14, 1, 13, 1.4)  # SOLID sole (stronger)

# ----- 6: Shirt ----------------------------------------------------------------
# Signature: broad shoulders + sleeves; body narrows at the bottom.
# STRONG upper content required so footwear (empty top) doesn't trigger shirt.
rect(T[6], 1, 4, 2, 12, 1.5)    # wide shoulders (stronger)
rect(T[6], 4, 9, 1, 13, 1.2)    # wide sleeves (stronger)
rect(T[6], 9, 13, 3, 11, 1.0)   # narrower body
rect(T[6], 9, 13, 0, 2, -1.0)   # bottom-left empty (stronger)
rect(T[6], 9, 13, 12, 14, -1.0) # bottom-right empty
rect(T[6], 3, 11, 6, 8, -0.3)   # faint button line
rect(T[6], 0, 1, 0, 14, -0.8)   # very top row empty
rect(T[6], 13, 14, 0, 2, -0.8)  # strong empty bottom corners
rect(T[6], 13, 14, 12, 14, -0.8)

# ----- 7: Sneaker --------------------------------------------------------------
# Signature: empty top ~3/4, SOLID compact shoe in last 2-3 rows (no straps).
rect(T[7], 0, 10, 0, 14, -1.5)  # STRONG empty top (more than sandal)
rect(T[7], 10, 14, 1, 13, 1.5)  # solid bottom (stronger)
rect(T[7], 10, 12, 4, 10, 0.6)  # strong SOLID middle (no gap unlike sandal)
rect(T[7], 13, 14, 2, 12, 0.4)  # extra bright sole centre

# ----- 8: Bag ------------------------------------------------------------------
# Signature: HANDLE on top (two bright uprights with a dark hole between them)
# + rectangular body; sides DARK (no sleeves, unlike upper garments).
rect(T[8], 0, 4, 4, 6, 2.0)     # handle left upright (VERY STRONG)
rect(T[8], 0, 4, 8, 10, 2.0)    # handle right upright (VERY STRONG)
rect(T[8], 0, 4, 6, 8, -2.5)    # HANDLE HOLE between uprights (dominant)
rect(T[8], 0, 4, 0, 4, -1.0)    # outside handle (left)
rect(T[8], 0, 4, 10, 14, -1.0)  # outside handle (right)
rect(T[8], 4, 13, 2, 12, 1.0)   # body
rect(T[8], 4, 13, 0, 2, -0.8)   # body sides strongly dark (no sleeves)
rect(T[8], 4, 13, 12, 14, -0.8)

# ----- 9: Ankle boot -----------------------------------------------------------
# Signature: empty top, ANKLE on the right rising from the sole, solid sole.
rect(T[9], 0, 4, 0, 14, -1.3)    # empty very top
rect(T[9], 4, 10, 0, 6, -1.0)    # left-upper empty (no ankle left)
rect(T[9], 4, 10, 8, 13, 1.3)    # ANKLE on the RIGHT (strong)
rect(T[9], 10, 13, 2, 13, 1.0)   # shoe body
rect(T[9], 10, 13, 0, 2, -0.3)
rect(T[9], 13, 14, 0, 14, 1.2)   # sole

# =============================================================================
# Normalise.
# =============================================================================
stacked = np.stack([t.reshape(N) for t in T]).astype(np.float32)   # (10, 196)
stacked = stacked - stacked.mean(axis=1, keepdims=True)
norms = np.sqrt((stacked ** 2).sum(axis=1, keepdims=True))
stacked = stacked / np.where(norms == 0, 1.0, norms)
stacked = stacked * 10.0

W = stacked.astype(np.float32)
b = np.zeros(10, dtype=np.float32)

# =============================================================================
# Assemble and write.
# =============================================================================
network = {
    "layers": [
        {"name": "pool", "op": "avgpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]},
        {"name": "flat", "op": "reshape", "inputs": ["pool"], "shape": [N]},
        {"name": "out",  "op": "dense",   "inputs": ["flat"], "out": 10},
    ]
}


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/out"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(network, f)
    np.savez(os.path.join(out_dir, "weights.npz"),
             **{"out.W": W, "out.b": b})


if __name__ == "__main__":
    main()
