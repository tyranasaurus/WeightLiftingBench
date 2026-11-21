"""Hand-designed person / no-person network. No training data is read: every weight
comes from colour-science formulas, a hand-written face model, and hand-chosen thresholds.

"No person" is the majority and is full of skin-coloured, face-structured confounders
(brown animals, wood, bread/food).  So fire for "person" only on a HUMAN face:
  (1) a PINKISH human-skin region  (blue not too low vs red -> excludes orange wood /
      brown animals / food), AND
  (2) a darker horizontal EYE-BAND inside it.
The pinkish-skin gate is what removes the animal/food faces that a bare eye-band fires on.

  image -> S  = pinkish-skin box membership in [0,1]
        -> Y  = luminance
  for scales with face ~ 12 / 24 / 48 px:
     band    = eye-band response on Y  (mean(forehead+cheeks) - mean(eye row))
     skinctx = local pinkish-skin fraction over the face-sized window
     AND     = min( relu(a*(band-m1)), relu(b*(skinctx-m2)) )
     face_s  = global max of AND
  face_ev = sum over scales ; person = ALPHA*face_ev - T ; no-person = 0
"""
import sys, os, json
import numpy as np

# pinkish human-skin box
CRLO, CRHI = 0.52, 0.66
CBLO, CBHI = 0.34, 0.49
YLO,  YHI  = 0.12, 0.97
RGMIN = 0.08
GBMIN = 0.03
RHO   = 0.55        # B/R floor: pinkish skin, not orange wood/animal/food
K     = 40.0
# eye-band face model
A_BAND = 8.0
M1     = 0.05
B_SKIN = 6.0
M2     = 0.30
ALPHA  = 1.0
T      = 0.35
SCALES = (2, 4, 8)   # downsample factors -> faces ~12/24/48 px


def eye_kernel():
    K6 = np.zeros((6, 6), np.float64)
    K6[[0, 1, 4, 5], :] = 1.0 / 24.0
    K6[[2, 3], :] = -1.0 / 12.0
    return K6


def skin_conv():
    wY  = np.array([0.299, 0.587, 0.114]); bY  = 0.0
    wCb = np.array([-0.168736, -0.331264, 0.5]); bCb = 0.5
    wCr = np.array([0.5, -0.418688, -0.081312]); bCr = 0.5
    W = np.zeros((9, 3, 1, 1), np.float64); b = np.zeros((9,), np.float64)
    W[0, :, 0, 0] = -wCr; b[0] = CRLO - bCr
    W[1, :, 0, 0] =  wCr; b[1] = bCr - CRHI
    W[2, :, 0, 0] = -wCb; b[2] = CBLO - bCb
    W[3, :, 0, 0] =  wCb; b[3] = bCb - CBHI
    W[4, :, 0, 0] = -wY;  b[4] = YLO - bY
    W[5, :, 0, 0] =  wY;  b[5] = bY - YHI
    W[6, 0, 0, 0] = -1.0; W[6, 1, 0, 0] = 1.0; b[6] = RGMIN
    W[7, 1, 0, 0] = -1.0; W[7, 2, 0, 0] = 1.0; b[7] = GBMIN
    W[8, 0, 0, 0] = RHO;  W[8, 2, 0, 0] = -1.0; b[8] = 0.0
    return W, b


def build():
    W1, b1 = skin_conv()
    weights = {
        "c_feat.W": W1, "c_feat.b": b1,
        "c_skin.W": np.full((1, 9, 1, 1), -K), "c_skin.b": np.array([1.0]),
        "c_lum.W": np.array([[[[0.299]], [[0.587]], [[0.114]]]]).reshape(1, 3, 1, 1), "c_lum.b": np.array([0.0]),
        "d_out.W": None, "d_out.b": None,  # set below
    }
    K6 = eye_kernel()
    layers = [
        {"name": "c_feat", "op": "conv", "inputs": ["image"], "out": 9, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "r_feat", "op": "relu", "inputs": ["c_feat"]},
        {"name": "c_skin", "op": "conv", "inputs": ["r_feat"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "r_skin", "op": "relu", "inputs": ["c_skin"]},
        {"name": "c_lum", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
    ]
    face_inputs = []
    for f in SCALES:
        s = str(f)
        ps = (96 // f) - 6 + 1  # side of the band/skinctx map = (96/f - 6) + 1; global max over it
        Wb = np.zeros((1, 1, 6, 6)); Wb[0, 0] = A_BAND * K6
        weights["band%s.W" % s] = Wb; weights["band%s.b" % s] = np.array([-A_BAND * M1])
        weights["sbin%s.W" % s] = np.array([[[[B_SKIN]]]]); weights["sbin%s.b" % s] = np.array([-B_SKIN * M2])
        weights["negs%s.W" % s] = np.array([[[[-1.0]]]]); weights["negs%s.b" % s] = np.array([0.0])
        weights["negr%s.W" % s] = np.array([[[[-1.0]]]]); weights["negr%s.b" % s] = np.array([0.0])
        layers += [
            {"name": "y%s" % s, "op": "avgpool", "inputs": ["c_lum"],  "size": [f, f], "stride": [f, f]},
            {"name": "sk%s" % s, "op": "avgpool", "inputs": ["r_skin"], "size": [f, f], "stride": [f, f]},
            {"name": "band%s" % s, "op": "conv", "inputs": ["y%s" % s], "out": 1, "kernel": [6, 6], "stride": 1, "padding": 0},
            {"name": "bbin%s" % s, "op": "relu", "inputs": ["band%s" % s]},
            {"name": "sctx%s" % s, "op": "avgpool", "inputs": ["sk%s" % s], "size": [6, 6], "stride": [1, 1]},
            {"name": "sbin%s" % s, "op": "conv", "inputs": ["sctx%s" % s], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "rsbin%s" % s, "op": "relu", "inputs": ["sbin%s" % s]},
            {"name": "negs%s" % s, "op": "conv", "inputs": ["rsbin%s" % s], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "diff%s" % s, "op": "add", "inputs": ["bbin%s" % s, "negs%s" % s]},
            {"name": "rd%s" % s, "op": "relu", "inputs": ["diff%s" % s]},
            {"name": "negr%s" % s, "op": "conv", "inputs": ["rd%s" % s], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "and%s" % s, "op": "add", "inputs": ["bbin%s" % s, "negr%s" % s]},
            {"name": "face%s" % s, "op": "maxpool", "inputs": ["and%s" % s], "size": [ps, ps], "stride": [ps, ps]},
        ]
        face_inputs.append("face%s" % s)
    layers += [
        {"name": "face_ev", "op": "add", "inputs": face_inputs},
        {"name": "d_out", "op": "dense", "inputs": ["face_ev"], "out": 2},
    ]
    weights["d_out.W"] = np.array([[0.0], [ALPHA]]); weights["d_out.b"] = np.array([0.0, -T])
    weights = {k: v.astype(np.float32) for k, v in weights.items()}
    return {"layers": layers}, weights


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    net, weights = build()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out, "weights.npz"), **weights)


if __name__ == "__main__":
    main()
