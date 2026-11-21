"""Hand-designed person / no-person network. Every number comes from this file (no data is read).
Usage: python3 build.py OUT_DIR
Idea: (1) per-pixel soft skin-colour membership from linear colour rules typed from general knowledge
(reddish-orange hue, moderate saturation, not too dark); (2) centre-minus-surround blob detectors on the
pooled skin map at 4, 8 and 16 pixel scales (faces, hands, arms are compact skin patches; wood, sand and
food surfaces are large and cancel); (3) decision: person if there is some compact skin evidence but the
photo is not flooded with skin-coloured pixels."""
import json, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else "."); out.mkdir(parents=True, exist_ok=True)
W = {}

# ---- 1. skin colour: violations of linear rules on (R, G, B); membership = relu(1 - sum of violations)
K = 12.0
rules = [  # (cR, cG, cB, const): violation = cR*R + cG*G + cB*B + const, counted when positive
    (-1.0, 1.12, 0.0, 0.0),    # R should exceed 1.12 G (red dominant)
    (1.0, -1.9, 0.0, 0.0),     # R below 1.9 G (not pure red)
    (0.0, -1.0, 0.95, 0.0),    # G at least 0.95 B (not purple / blue)
    (-0.5, 1.0, -0.5, 0.0),    # hue below about 30 degrees: G no closer to R than to B (rejects yellow-brown)
    (-1.0, 0.0, 0.0, 0.28),    # R at least 0.28 (not too dark)
    (0.22, 0.0, -1.0, 0.0),    # B at least 0.22 R (not over-saturated orange)
    (-1.0, 0.0, 1.0, 0.07),    # R - B at least 0.07 (not grey / white)
    (0.0, 0.0, 0.0, 0.0),      # unused
]
W["v.W"] = np.array([[r[0], r[1], r[2]] for r in rules], np.float32).reshape(8, 3, 1, 1) * K
W["v.b"] = np.array([r[3] for r in rules], np.float32) * K
W["m.W"] = -np.ones((1, 8, 1, 1), np.float32)
W["m.b"] = np.ones(1, np.float32)

# ---- 2. blob kernels (9 x 9): centre minus ring, threshold 0.5 so straight edges of big regions stay silent
def ring(d):
    k = np.zeros((9, 9), np.float32)
    for y in range(9):
        for x in range(9):
            if max(abs(y - 4), abs(x - 4)) == d: k[y, x] = 1.0
    return k / k.sum()
small = -ring(2); small[4, 4] = 1.0
medium = -ring(4); medium[3:6, 3:6] = 1.0 / 9.0
ident = np.zeros((9, 9), np.float32); ident[4, 4] = 1.0
THR = 0.5
W["bf.W"] = np.stack([small, medium, ident])[:, None].astype(np.float32)   # fine level: 47 x 47 cells of 4 px, step 2
W["bf.b"] = np.array([-THR, -THR, 0.0], np.float32)
W["bc.W"] = np.stack([small, medium])[:, None].astype(np.float32)          # coarse level: 23 x 23 cells of 8 px, step 4
W["bc.b"] = np.array([-THR, -THR], np.float32)

# ---- 3. decision. B = blob evidence in units of "about one clean blob"; f = mean skin membership of the photo
NF, NC = 47 * 47, 23 * 23
B_LO, B_HI, F_LO, F_HI = 0.3, 1.5, 0.12, 0.30
W["da.W"] = np.array([[NF, NF, 0], [NF, NF, 0], [0, 0, 1], [0, 0, 1]], np.float32)
W["da.b"] = np.array([-B_LO, -B_HI, -F_LO, -F_HI], np.float32)
W["db.W"] = np.array([[NC, NC], [NC, NC], [0, 0], [0, 0]], np.float32)
W["db.b"] = np.zeros(4, np.float32)
W["o.W"] = np.array([[0, 0, 0, 0],
                     [1 / (B_HI - B_LO), -1 / (B_HI - B_LO), -1 / (F_HI - F_LO), 1 / (F_HI - F_LO)]], np.float32)
W["o.b"] = np.array([0.0, -0.5], np.float32)

layers = [
    {"name": "v", "op": "conv", "inputs": ["image"], "out": 8, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "vr", "op": "relu", "inputs": ["v"]},
    {"name": "m", "op": "conv", "inputs": ["vr"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "skin", "op": "relu", "inputs": ["m"]},
    {"name": "pf", "op": "avgpool", "inputs": ["skin"], "size": [4, 4], "stride": [2, 2]},
    {"name": "bf", "op": "conv", "inputs": ["pf"], "out": 3, "kernel": [9, 9], "stride": 1, "padding": 4},
    {"name": "bfr", "op": "relu", "inputs": ["bf"]},
    {"name": "gf", "op": "avgpool", "inputs": ["bfr"], "size": [47, 47], "stride": [47, 47]},
    {"name": "da", "op": "dense", "inputs": ["gf"], "out": 4},
    {"name": "pc", "op": "avgpool", "inputs": ["skin"], "size": [8, 8], "stride": [4, 4]},
    {"name": "bc", "op": "conv", "inputs": ["pc"], "out": 2, "kernel": [9, 9], "stride": 1, "padding": 4},
    {"name": "bcr", "op": "relu", "inputs": ["bc"]},
    {"name": "gc", "op": "avgpool", "inputs": ["bcr"], "size": [23, 23], "stride": [23, 23]},
    {"name": "db", "op": "dense", "inputs": ["gc"], "out": 4},
    {"name": "h", "op": "add", "inputs": ["da", "db"]},
    {"name": "hr", "op": "relu", "inputs": ["h"]},
    {"name": "o", "op": "dense", "inputs": ["hr"], "out": 2},
]
(out / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
np.savez(out / "weights.npz", **{k: v.astype(np.float32) for k, v in W.items()})
