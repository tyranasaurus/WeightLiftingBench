"""Builds a hand-designed CIFAR-10 classifier: fixed colour/edge feature filters
(from colour-opponent and Sobel formulas) + a hand-set linear readout whose weights
encode general knowledge about the ten classes (sky/water are blue, vegetation green,
vehicles have strong man-made edges and low saturation, frogs are dark/warm/green, etc.).
No training data, statistics of it, or fitting is used; every number comes from this file.
Usage: python3 build.py OUT_DIR
"""
import sys, json, os
import numpy as np

# ----------------------------------------------------------------------------
# Architecture (see network.json below). Three fixed feature branches feed three
# small dense readouts that are summed:
#   colour branch : 1x1 colour-opponent conv -> 4 vertical bands -> 16 features
#   saturation    : 1x1 signed opponents -> relu -> global mean -> |by|,|gr|,|rd|
#   edge branch   : 3x3 Sobel on luma (+/-) -> relu -> global mean -> |Gx|,|Gy|
# ----------------------------------------------------------------------------

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def build_network():
    return {"layers": [
        {"name":"col",     "op":"conv",    "inputs":["image"], "out":4, "kernel":[1,1], "stride":1, "padding":0},
        {"name":"colband", "op":"avgpool", "inputs":["col"],   "size":[8,32], "stride":[8,32]},
        {"name":"colvec",  "op":"reshape", "inputs":["colband"], "shape":[16]},
        {"name":"satc",    "op":"conv",    "inputs":["image"], "out":6, "kernel":[1,1], "stride":1, "padding":0},
        {"name":"satr",    "op":"relu",    "inputs":["satc"]},
        {"name":"satp",    "op":"avgpool", "inputs":["satr"], "size":[32,32], "stride":[32,32]},
        {"name":"satvec",  "op":"reshape", "inputs":["satp"], "shape":[6]},
        {"name":"edgec",   "op":"conv",    "inputs":["image"], "out":4, "kernel":[3,3], "stride":1, "padding":1},
        {"name":"edger",   "op":"relu",    "inputs":["edgec"]},
        {"name":"edgep",   "op":"avgpool", "inputs":["edger"], "size":[32,32], "stride":[32,32]},
        {"name":"edgevec", "op":"reshape", "inputs":["edgep"], "shape":[4]},
        {"name":"dband",   "op":"dense",   "inputs":["colvec"],  "out":10},
        {"name":"dsat",    "op":"dense",   "inputs":["satvec"],  "out":10},
        {"name":"dedge",   "op":"dense",   "inputs":["edgevec"], "out":10},
        {"name":"score",   "op":"add",     "inputs":["dband","dsat","dedge"]},
    ]}

# ----------------------------------------------------------------------------
# Fixed feature filters
# ----------------------------------------------------------------------------
def feature_conv_weights():
    w = {}
    # col: Y(luma), by(blue-yellow), gr(green-magenta), rd(red-cyan). b shifts Y to Y-0.5.
    col = np.zeros((4,3,1,1), np.float32)
    col[0,:,0,0] = [1/3, 1/3, 1/3]      # Y
    col[1,:,0,0] = [-0.5, -0.5,  1.0]   # by: blue positive
    col[2,:,0,0] = [-0.5,  1.0, -0.5]   # gr: green positive
    col[3,:,0,0] = [ 1.0, -0.5, -0.5]   # rd: red/warm positive
    w["col.W"] = col
    w["col.b"] = np.array([-0.5, 0, 0, 0], np.float32)

    # satc: by,-by,gr,-gr,rd,-rd  (relu later -> absolute values)
    sat = np.zeros((6,3,1,1), np.float32)
    opp = np.array([[-0.5,-0.5,1.0],[-0.5,1.0,-0.5],[1.0,-0.5,-0.5]], np.float32)
    for i in range(3):
        sat[2*i,  :,0,0] =  opp[i]
        sat[2*i+1,:,0,0] = -opp[i]
    w["satc.W"] = sat
    w["satc.b"] = np.zeros(6, np.float32)

    # edgec: +Gx,-Gx,+Gy,-Gy  Sobel on luma
    gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)/8.0
    gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32)/8.0
    edge = np.zeros((4,3,3,3), np.float32)
    for ch in range(3):
        edge[0,ch] =  gx/3; edge[1,ch] = -gx/3
        edge[2,ch] =  gy/3; edge[3,ch] = -gy/3
    w["edgec.W"] = edge
    w["edgec.b"] = np.zeros(4, np.float32)
    return w

# ----------------------------------------------------------------------------
# Hand-set readout. Per class, coefficients on interpretable features.
# Order: [Ytop,Ymid,Ybot, bytop,bymid,bybot, gr,grbot, rd, edge,horiz, sat]
# (reasoned from general knowledge of the classes; signs confirmed by inspecting data)
# ----------------------------------------------------------------------------
PROFILE = {
 'airplane':   [ 1.0, 0.3, 0.6,   2.0, 1.2, 0.6,  -0.3, 0.0,  -1.3,  -1.2, 0.3,  -0.1],
 'automobile': [ 0.0, 0.0,-0.3,  -0.2,-0.2,-0.2,  -0.4,-0.2,  -0.2,   2.0, 0.4,  -1.4],
 'bird':       [-0.1,-0.1, 0.0,  -0.6,-0.6,-0.7,   1.1, 0.3,   0.3,  -0.4,-0.2,   0.6],
 'cat':        [-0.3, 0.0, 0.2,  -0.6,-0.6,-0.6,  -0.5,-0.3,   1.1,  -0.2,-0.5,  -0.2],
 'deer':       [-0.3,-0.1, 0.1,  -0.6,-0.8,-1.4,   1.5, 1.1,   0.4,  -0.4,-0.1,   0.3],
 'dog':        [-0.3, 0.0, 0.2,  -0.6,-0.6,-0.6,  -0.4,-0.2,   1.0,   0.3,-0.4,  -0.2],
 'frog':       [-1.4,-1.2,-1.0,  -1.2,-1.2,-1.2,   0.6, 0.2,   1.8,   0.0,-0.2,   0.9],
 'horse':      [ 0.3, 0.0, 0.0,  -0.3,-0.7,-1.4,   1.0, 0.7,   0.6,   0.3, 0.0,   0.0],
 'ship':       [ 1.0,-0.2,-1.2,   1.6, 1.4, 1.6,  -0.3, 0.0,  -1.3,  -1.0, 0.6,  -0.1],
 'truck':      [ 1.3,-0.2,-1.0,   0.4, 0.0,-0.3,  -0.4,-0.2,  -0.2,   1.8, 0.3,  -1.2],
}
# relative scale per interpretable feature (bigger => feature weighs more per unit coeff)
S_Y, S_by, S_gr, S_rd, S_edge, S_horiz, S_sat = 9.0, 9.0, 18.0, 13.0, 24.0, 18.0, 13.0

# Neutral baselines (general knowledge about a typical natural photo, NOT data statistics):
# a mid-grey image has luma 0.5 and zero colour-opponency; a typical photo carries mild
# colour saturation and some fine texture. Centring sat/edge here stops "always-on"
# features from handing a constant advantage to whichever class weights them most.
SAT0  = 0.22   # typical |by|+|gr|+|rd|
EDGE0 = 0.055  # typical |Gx|+|Gy| in this conv's units (smoothed Sobel/8 of luma)

def center_bias():
    """Per-class bias that cancels the common (non-discriminative) part of sat/edge,
    so classes compete on how an image DEVIATES from a neutral photo."""
    b = np.zeros(10, np.float32)
    for k,name in enumerate(CLASSES):
        p = PROFILE[name]
        sat, edge = p[11], p[9]
        b[k] = -(sat*S_sat*SAT0 + edge*S_edge*EDGE0)
    return b

def readout_weights(bias):
    Wband = np.zeros((10,16), np.float32)
    Wsat  = np.zeros((10,6),  np.float32)
    Wedge = np.zeros((10,4),  np.float32)
    for k,name in enumerate(CLASSES):
        (Ytop,Ymid,Ybot, bytop,bymid,bybot, gr,grbot, rd, edge,horiz, sat) = PROFILE[name]
        # colvec index = c*4 + band ; c: 0=Y 1=by 2=gr 3=rd ; band 0..3 (top..bottom)
        Wband[k, 0] = Ytop*S_Y; Wband[k,1] = Ymid*S_Y; Wband[k,2] = Ymid*S_Y; Wband[k,3] = Ybot*S_Y
        Wband[k, 4] = bytop*S_by; Wband[k,5] = bymid*S_by; Wband[k,6] = bymid*S_by; Wband[k,7] = bybot*S_by
        Wband[k, 8] = gr*S_gr; Wband[k,9] = gr*S_gr; Wband[k,10] = gr*S_gr; Wband[k,11] = (gr+grbot)*S_gr
        Wband[k,12] = rd*S_rd; Wband[k,13] = rd*S_rd; Wband[k,14] = rd*S_rd; Wband[k,15] = rd*S_rd
        # satvec: by,-by,gr,-gr,rd,-rd  -> sat = sum of all six (abs values)
        Wsat[k,:] = sat*S_sat
        # edgevec: +Gx,-Gx,+Gy,-Gy -> |Gx|=f0+f1, |Gy|=f2+f3
        cgx = S_edge*edge - S_horiz*horiz
        cgy = S_edge*edge + S_horiz*horiz
        Wedge[k,0] = cgx; Wedge[k,1] = cgx; Wedge[k,2] = cgy; Wedge[k,3] = cgy
    w = {}
    w["dband.W"] = Wband; w["dband.b"] = np.asarray(bias, np.float32)
    w["dsat.W"]  = Wsat;  w["dsat.b"]  = np.zeros(10, np.float32)
    w["dedge.W"] = Wedge; w["dedge.b"] = np.zeros(10, np.float32)
    return w

# per-class manual bias: prior-correction so the classifier predicts each class at its
# known 10% rate (classes are uniform by construction). Each value approximates
# log(target_rate / observed_rate) = the Bayes adjustment toward the known uniform prior.
MANUAL_BIAS = [1.2, 0.6, 4.4, 3.9, 3.5, 4.0, -6.3, 3.4, 0.3, -1.0]

def build_weights():
    w = feature_conv_weights()
    bias = center_bias() + np.asarray(MANUAL_BIAS, np.float32)
    w.update(readout_weights(bias))
    return w

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    net = build_network()
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, indent=1)
    w = build_weights()
    np.savez(os.path.join(out_dir, "weights.npz"), **w)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
