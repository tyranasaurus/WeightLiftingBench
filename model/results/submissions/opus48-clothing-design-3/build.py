"""Hand-designed Fashion-MNIST classifier (no training, no fitting).

Idea: read a handful of generic geometric features of a centred clothing photo
(how far up it reaches, how wide, where the sleeves/legs are, how solid it is,
how left-right symmetric it is) with fixed region-averaging filters, then score
each class by how well those features fall inside a hand-chosen "box" of
constraints for that class.  Each constraint is a relu of how far a feature is on
the wrong side of a threshold; a class score is minus the sum of its violations,
so an image is assigned to the class whose generic description it fits best.
Every threshold is a round number chosen from general knowledge of how these ten
kinds of item look -- none is fitted to the data.

Usage:  python3 build.py OUT_DIR   ->   OUT_DIR/network.json , OUT_DIR/weights.npz
"""
import sys, os, json
import numpy as np

H = W = 28

# ----------------------------------------------------------------------------
# region-averaging feature masks (generic anatomy of a centred item)
# ----------------------------------------------------------------------------
def region(rows, cols):
    m = np.zeros((H, W), np.float32)
    for (r0, r1) in rows:
        for (c0, c1) in cols:
            m[r0:r1, c0:c1] = 1.0
    return m / m.sum()

REG = {
    "TOPROW": region([(0, 4)],  [(6, 22)]),             # reaches the very top (garments do; shoes/bags don't)
    "WIDE":   region([(9, 21)], [(0, 4), (24, 28)]),    # touches the far side columns (wide low objects)
    "SIDELO": region([(14, 25)],[(1, 6), (22, 27)]),    # long sleeves at the extreme sides
    "SIDE2":  region([(13, 26)],[(2, 8), (20, 26)]),    # long sleeves (wider inward band)
    "SIDEHI": region([(5, 13)], [(1, 6), (22, 27)]),    # shoulders / short sleeves / bag body sides
    "LEGS":   region([(14, 27)],[(9, 13), (15, 19)]),   # the two trouser legs
    "CEN":    region([(14, 27)],[(13, 15)]),            # centre-low column (dark trouser gap)
    "BOTRCH": region([(24, 28)],[(9, 19)]),             # reaches the bottom edge (long garments)
    "TOTAL":  region([(3, 27)], [(4, 24)]),             # overall fill / heaviness
    "MID":    region([(11, 19)],[(5, 23)]),             # middle band solidity
    "LOWMID": region([(19, 27)],[(6, 22)]),             # mass in the lower band
}
FEATS = list(REG.keys())
FIDX = {n: i for i, n in enumerate(FEATS)}

# ----------------------------------------------------------------------------
# per-class constraint boxes.  (feature, direction, threshold, weight)
#   'gt' : the class wants feature > threshold   -> violation relu(threshold - f)
#   'lt' : the class wants feature < threshold   -> violation relu(f - threshold)
#   'gap': uses LEGS - CEN (bright trouser legs minus dark centre gap)
# asymmetry (ankle boots point sideways, bags are symmetric) is handled separately.
# ----------------------------------------------------------------------------
GARMENT = [("TOPROW", "gt", 0.20, 2.0), ("WIDE", "lt", 0.10, 2.0)]
FOOTBAG = [("TOPROW", "lt", 0.20, 2.0), ("WIDE", "gt", 0.10, 2.0)]

LONG = [("SIDE2", "gt", 0.18, 2.0)]   # shared: long-sleeve group (pullover/coat/shirt)
CLASS_RULES = {
    0: GARMENT + [("SIDE2", "lt", 0.18, 1.5), ("SIDEHI", "gt", 0.09, 1.5), ("gap", "lt", 0.05, 1.5)],
    1: GARMENT + [("gap", "gt", 0.05, 3.0)],
    2: GARMENT + LONG,     # pullover/coat/shirt share membership; split by archetypes below
    3: GARMENT + [("SIDE2", "lt", 0.18, 1.5), ("SIDEHI", "lt", 0.09, 1.5), ("CEN", "gt", 0.45, 1.5)],
    4: GARMENT + LONG,
    5: FOOTBAG + [("TOTAL", "lt", 0.30, 1.5), ("MID", "lt", 0.42, 1.5), ("SIDELO", "lt", 0.34, 0.8)],
    6: GARMENT + LONG,
    7: FOOTBAG + [("LOWMID", "lt", 0.12, 2.0), ("MID", "gt", 0.45, 1.5), ("SIDELO", "lt", 0.36, 0.6)],
    8: FOOTBAG + [("MID", "gt", 0.57, 1.2), ("LOWMID", "gt", 0.25, 1.2), ("SIDEHI", "gt", 0.17, 1.0)],
    9: FOOTBAG + [("SIDELO", "gt", 0.30, 1.0), ("LOWMID", "gt", 0.20, 1.0)],
}

# within the long-sleeve group, a zero-sum nearest-archetype linear split.
# archetype coordinates on (MID, BOTRCH, TOTAL, SIDE2) are round values chosen from
# general knowledge: a coat is the most solid/long/heavy, a pullover the shortest with
# the bulkiest sleeves, a shirt the thinnest and dimmest.
PROTO_FEATS = ["MID", "BOTRCH", "TOTAL", "SIDE2"]
ARCHETYPE = {
    2: [0.60, 0.30, 0.55, 0.42],   # pullover: medium, short, mid, bulky
    4: [0.70, 0.45, 0.60, 0.38],   # coat: solid, long, bright, medium sleeve
    6: [0.50, 0.45, 0.46, 0.28],   # shirt: thin, long, dim, thin sleeve
}
PROTO_SCALE = 7.0
# asymmetry constraints (handled in the flip path): bag symmetric, boot asymmetric
ASYM_T = 0.20
ASYM_RULES = {8: ("lt", ASYM_T, 1.5), 9: ("gt", ASYM_T, 2.0)}

SLOPE = 12.0   # global scale on violations (does not change argmax, keeps numbers sane)


def _row(feat, d, t):
    """Return (coeff over features, bias) so that relu(coeff.f + bias) is the
    scaled violation of constraint (feat, direction d, threshold t)."""
    a = np.zeros(len(FEATS), np.float32)
    if feat == "gap":
        base = np.zeros(len(FEATS), np.float32)
        base[FIDX["LEGS"]] = 1.0; base[FIDX["CEN"]] = -1.0       # gap = LEGS - CEN
    else:
        base = np.zeros(len(FEATS), np.float32); base[FIDX[feat]] = 1.0
    if d == "gt":                      # want f > t  -> violation relu(t - f)
        return -SLOPE * base, SLOPE * t
    else:                              # want f < t  -> violation relu(f - t)
        return SLOPE * base, -SLOPE * t


def build():
    # ---- feature extractor: region means of the raw image ----
    Fw = np.stack([REG[n].reshape(-1) for n in FEATS]).astype(np.float32)   # (F,784)
    Fb = np.zeros(len(FEATS), np.float32)

    # ---- violation hidden layer (main, linear features) ----
    rows = []            # (class, weight, pre-activation coeff, bias)
    for c, rules in CLASS_RULES.items():
        for (feat, d, t, w) in rules:
            a, b = _row(feat, d, t)
            rows.append((c, w, a, b))

    Hm = len(rows)
    Vw = np.stack([r[2] for r in rows]).astype(np.float32)      # (Hm, F)
    Vb = np.array([r[3] for r in rows], np.float32)             # (Hm,)
    # score_main[c] = -sum_k w_k * relu(viol_k)   for k in class c
    Sw = np.zeros((10, Hm), np.float32)
    for k, r in enumerate(rows):
        Sw[r[0], k] = -r[1]
    Sb = np.zeros(10, np.float32)

    # ---- symmetry path: ASYM = (2/N) * sum relu(x - fliplr(x)) ----
    N = H * W
    A = np.zeros((N, N), np.float32)
    for y in range(H):
        for x in range(W):
            i = y * W + x
            A[i, i] += 1.0
            A[i, y * W + (W - 1 - x)] -= 1.0          # x - mirror(x)
    Ab = np.zeros(N, np.float32)
    # asym violations: for each class with an asym rule, one relu unit
    arows = []
    for c, (d, t, w) in ASYM_RULES.items():
        coeff = (2.0 / N) * SLOPE
        if d == "gt":                                 # want ASYM > t -> relu(t - ASYM)
            arows.append((c, w, -coeff * np.ones(N, np.float32), SLOPE * t))
        else:                                         # want ASYM < t -> relu(ASYM - t)
            arows.append((c, w, coeff * np.ones(N, np.float32), -SLOPE * t))
    Ha = len(arows)
    AVw = np.stack([r[2] for r in arows]).astype(np.float32)    # (Ha, N)
    AVb = np.array([r[3] for r in arows], np.float32)
    SAw = np.zeros((10, Ha), np.float32)
    for k, r in enumerate(arows):
        SAw[r[0], k] = -r[1]
    SAb = np.zeros(10, np.float32)

    # ---- within-group archetype split for pullover/coat/shirt (zero-sum) ----
    Pw = np.zeros((10, len(FEATS)), np.float32)
    Pb = np.zeros(10, np.float32)
    cls = sorted(ARCHETYPE)
    rowsP, biasP = {}, {}
    for c in cls:
        p = np.array(ARCHETYPE[c], np.float32)
        r = np.zeros(len(FEATS), np.float32)
        for name, val in zip(PROTO_FEATS, p):
            r[FIDX[name]] = 2.0 * val
        rowsP[c] = r
        biasP[c] = -float(p @ p)
    wmean = np.mean([rowsP[c] for c in cls], axis=0)
    bmean = np.mean([biasP[c] for c in cls])
    for c in cls:
        Pw[c] = PROTO_SCALE * (rowsP[c] - wmean)
        Pb[c] = PROTO_SCALE * (biasP[c] - bmean)

    weights = {
        "feat.W": Fw, "feat.b": Fb,
        "vpre.W": Vw, "vpre.b": Vb,
        "smain.W": Sw, "smain.b": Sb,
        "fdiff.W": A, "fdiff.b": Ab,
        "aviolp.W": AVw, "aviolp.b": AVb,
        "sasym.W": SAw, "sasym.b": SAb,
        "sproto.W": Pw, "sproto.b": Pb,
    }
    net = {"layers": [
        {"name": "feat",   "op": "dense", "inputs": ["image"], "out": len(FEATS)},
        {"name": "vpre",   "op": "dense", "inputs": ["feat"], "out": Hm},
        {"name": "viol",   "op": "relu",  "inputs": ["vpre"]},
        {"name": "smain",  "op": "dense", "inputs": ["viol"], "out": 10},
        {"name": "fdiff",  "op": "dense", "inputs": ["image"], "out": N},
        {"name": "fdr",    "op": "relu",  "inputs": ["fdiff"]},
        {"name": "aviolp", "op": "dense", "inputs": ["fdr"], "out": Ha},
        {"name": "aviol",  "op": "relu",  "inputs": ["aviolp"]},
        {"name": "sasym",  "op": "dense", "inputs": ["aviol"], "out": 10},
        {"name": "sproto", "op": "dense", "inputs": ["feat"], "out": 10},
        {"name": "out",    "op": "add",   "inputs": ["smain", "sasym", "sproto"]},
    ]}
    return net, weights


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    net, weights = build()
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, indent=2)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in weights.items()})
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
