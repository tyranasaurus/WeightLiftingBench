#!/usr/bin/env python3
"""
Hand-designed person / no-person network for 96x96 COCO thumbnails.

Every number below is typed from general knowledge of what people look like in
photographs (skin chromaticity, a face as a skin patch with two dark eyes beside a
brighter nose bridge under a bright forehead, dark hair above a skin blob, a standing
figure as a narrow vertical blob against uniform snow / water / grass, sports courts as
thin white lines on coloured ground).  Nothing is measured from, fitted to, or copied
from any image.

Usage:  python3 build.py OUT_DIR      -> writes OUT_DIR/network.json, OUT_DIR/weights.npz
"""
import sys, os, json
import numpy as np

# --------------------------------------------------------------------------------------
# tiny graph builder
# --------------------------------------------------------------------------------------
layers = []
weights = {}
_names = set()


def _reg(name):
    assert name not in _names, name
    _names.add(name)


def conv(name, inp, W, b, stride=1, padding=0):
    _reg(name)
    W = np.asarray(W, np.float32)
    b = np.asarray(b, np.float32)
    assert W.ndim == 4 and W.shape[2] == W.shape[3] and b.shape == (W.shape[0],)
    weights[name + ".W"] = W
    weights[name + ".b"] = b
    layers.append({"name": name, "op": "conv", "inputs": [inp], "out": int(W.shape[0]),
                   "kernel": [int(W.shape[2]), int(W.shape[3])], "stride": int(stride), "padding": int(padding)})
    return name


def dense(name, inp, W, b):
    _reg(name)
    W = np.asarray(W, np.float32)
    b = np.asarray(b, np.float32)
    assert W.ndim == 2 and b.shape == (W.shape[0],)
    weights[name + ".W"] = W
    weights[name + ".b"] = b
    layers.append({"name": name, "op": "dense", "inputs": [inp], "out": int(W.shape[0])})
    return name


def relu(name, inp):
    _reg(name)
    layers.append({"name": name, "op": "relu", "inputs": [inp]})
    return name


def maxpool(name, inp, size, stride):
    _reg(name)
    layers.append({"name": name, "op": "maxpool", "inputs": [inp], "size": list(size), "stride": list(stride)})
    return name


def avgpool(name, inp, size, stride):
    _reg(name)
    layers.append({"name": name, "op": "avgpool", "inputs": [inp], "size": list(size), "stride": list(stride)})
    return name


def add(name, inps):
    _reg(name)
    layers.append({"name": name, "op": "add", "inputs": list(inps)})
    return name


# --------------------------------------------------------------------------------------
# Stage 1: per-pixel colour evidence (1x1 convs).  Soft indicators are built as
#   indicator = relu(1 - sum_i violation_i),  violation_i = relu((threshold_i - x_i) / ramp_i)
# so each indicator is 1 when all conditions hold and fades linearly to 0.
# --------------------------------------------------------------------------------------
CR = np.array([0.5, -0.4187, -0.0813])        # YCbCr chroma (red difference), on 0..1 RGB
CB = np.array([-0.1687, -0.3313, 0.5])        # YCbCr chroma (blue difference)
YL = np.array([0.299, 0.587, 0.114])          # luminance
R = np.array([1.0, 0, 0]); G_ = np.array([0, 1.0, 0]); B_ = np.array([0, 0, 1.0])

cf_rows = []   # (weight vector over [r,g,b], bias)


def ramp_below(vec, thr, width):
    """channel = (thr - vec.x) / width : positive when x is below thr (a violation of x >= thr)."""
    cf_rows.append((-vec / width, thr / width))
    return len(cf_rows) - 1


def ramp_above(vec, thr, width):
    """channel = (vec.x - thr) / width : positive when x is above thr."""
    cf_rows.append((vec / width, -thr / width))
    return len(cf_rows) - 1


# skin: YCbCr skin box (Cr in ~[0.06,0.17], Cb in ~[-0.18,-0.01]), not too dark,
#       not pink (g >= b), not yellow (hue below ~30 degrees: g-b must not exceed 1.2(r-g))
SKIN_V = [ramp_below(CR, 0.06, 0.03), ramp_above(CR, 0.17, 0.04),
          ramp_above(CB, -0.01, 0.03), ramp_below(CB, -0.18, 0.04),
          ramp_below(YL, 0.25, 0.10), ramp_below(G_ - B_, -0.02, 0.04),
          ramp_above((G_ - B_) - 1.2 * (R - G_), 0.03, 0.05)]
# luminance and darkness
cf_rows.append((YL, 0.0)); Y_CH = len(cf_rows) - 1
cf_rows.append((-YL / 0.35, 1.0)); DARK_CH = len(cf_rows) - 1          # dark = (0.35 - Y)/0.35
# bright cool / neutral white (snow, overcast sky, white walls in daylight)
WHITE_V = [ramp_below(YL, 0.75, 0.12), ramp_above(CR, 0.02, 0.03),
           ramp_below(CB, -0.03, 0.03), ramp_above(CB, 0.08, 0.05)]
# saturated warm colours (red / orange): clipped ramp on Cr
SW_LO, SW_HI = ramp_above(CR, 0.18, 0.06), ramp_above(CR, 0.24, 0.06)
# green (grass, vegetation, green courts)
GREEN_V = [ramp_below(G_ - R, 0.05, 0.05), ramp_below(G_ - B_, 0.10, 0.05), ramp_below(YL, 0.15, 0.10)]
# blue (water, sky, blue courts): clearly bluer than red, not just a cool white
BLUE_V = [ramp_below(B_ - R, 0.15, 0.07), ramp_below(B_ - G_, 0.0, 0.05), ramp_below(YL, 0.20, 0.10)]

n_cf = len(cf_rows)
W = np.zeros((n_cf, 3, 1, 1), np.float32)
b = np.zeros((n_cf,), np.float32)
for i, (w, bb) in enumerate(cf_rows):
    W[i, :, 0, 0] = w
    b[i] = bb
cf = relu("cf_r", conv("cf", "image", W, b))

# Stage 2: combine into 7 pixel maps: S (skin), Y, D (dark), W (white), SW (sat. warm), G (green), B (blue)
CH = dict(S=0, Y=1, D=2, W=3, SW=4, G=5, B=6)
NCH = 7
W = np.zeros((NCH, n_cf, 1, 1), np.float32)
b = np.zeros((NCH,), np.float32)
for i in SKIN_V: W[CH["S"], i] = -1
b[CH["S"]] = 1
W[CH["Y"], Y_CH] = 1
W[CH["D"], DARK_CH] = 1
for i in WHITE_V: W[CH["W"], i] = -1
b[CH["W"]] = 1
W[CH["SW"], SW_LO] = 1; W[CH["SW"], SW_HI] = -1
for i in GREEN_V: W[CH["G"], i] = -1
b[CH["G"]] = 1
for i in BLUE_V: W[CH["B"], i] = -1
b[CH["B"]] = 1
pf = relu("pf_r", conv("pf", cf, W, b))      # (7, 96, 96), all values in [0, 1]

# --------------------------------------------------------------------------------------
# Stage 3: cell pyramids.  q-levels: cell s, stride s/2 (overlapping); p-levels: cell c, stride c.
# --------------------------------------------------------------------------------------
p2 = avgpool("p2", pf, [2, 2], [2, 2])                 # (7,48,48)
q6 = avgpool("q6", pf, [6, 6], [3, 3])                 # (7,31,31)   6px cells, stride 3
q8 = avgpool("q8", p2, [4, 4], [2, 2])                 # (7,23,23)   8px cells, stride 4
q12 = avgpool("q12", p2, [6, 6], [3, 3])               # (7,15,15)   12px cells, stride 6
q16 = avgpool("q16", p2, [8, 8], [4, 4])               # (7,11,11)   16px cells, stride 8
q32 = avgpool("q32", p2, [16, 16], [8, 8])             # (7,5,5)     32px cells, stride 16
p3 = avgpool("p3", pf, [3, 3], [3, 3])                 # (7,32,32)
p4 = avgpool("p4", p2, [2, 2], [2, 2])                 # (7,24,24)
p6 = avgpool("p6", pf, [6, 6], [6, 6])                 # (7,16,16)
p8 = avgpool("p8", p4, [2, 2], [2, 2])                 # (7,12,12)
p12 = avgpool("p12", pf, [12, 12], [12, 12])           # (7,8,8)
gavg = avgpool("gavg", pf, [96, 96], [96, 96])         # (7,1,1)     whole-image fractions

# --------------------------------------------------------------------------------------
# Slot bookkeeping for the readout.  Every branch ends in a global max (or mean) and a
# "placement" dense layer that drops its values into named slots of one shared vector; the
# vectors are summed (add) and passed through relu (stage-1 units).  A second dense+relu forms
# clipped ORs across scales and ANDs with whole-image gates; the last dense gives the 2 scores.
# --------------------------------------------------------------------------------------
slot_names = []


def slot(name):
    if name not in slot_names:
        slot_names.append(name)
    return slot_names.index(name)


# OR-groups: group value = min(sum of members, 1) = sum - relu(sum - 1): two slots "g" and "g-1".
GROUPS = ["blob", "bigblob", "head", "face", "snow", "water", "field", "lines", "wfrac", "bfrac", "cfrac", "gfrac",
          "white_excess", "skin_amount"]
for g in GROUPS:
    slot(g); slot(g + "-1")
slot("skin_excess"); slot("warm_excess")
NSLOT = len(slot_names)

placements = []
_bias_done = [False]


def place(name, inp, n_in, entries, extra_bias=None):
    """entries: list of (slot_name, input_index, coefficient).  Group '-1' biases are added once."""
    Wp = np.zeros((NSLOT, n_in), np.float32)
    bp = np.zeros((NSLOT,), np.float32)
    for sname, j, c in entries:
        Wp[slot(sname), j] += c
    if not _bias_done[0]:
        for g in GROUPS:
            bp[slot(g + "-1")] = -1.0
        _bias_done[0] = True
    if extra_bias:
        for sname, v in extra_bias.items():
            bp[slot(sname)] += v
    placements.append(dense(name, inp, Wp, bp))


# --------------------------------------------------------------------------------------
# Detector kernels on a 7-channel pyramid level (5x5 cells, index 2 = centre)
# --------------------------------------------------------------------------------------
def K():
    return np.zeros((NCH, 5, 5), np.float32)


def clip_pair(k, lo, hi):
    """two pre-activations giving clip((k.x - lo)/(hi - lo), 0, 1) = relu(a) - relu(a - 1)."""
    w = hi - lo
    return [(k / w, -lo / w), (k / w, -lo / w - 1.0)]


def blob_kernel():
    # compact skin blob: centre cell minus the ring of 16 cells at distance 2 (touching, not overlapping)
    k = K()
    k[CH["S"], 2, 2] = 1.0
    for i in range(5):
        for j in range(5):
            if max(abs(i - 2), abs(j - 2)) == 2:
                k[CH["S"], i, j] = -1.0 / 16
    return k


def head_pairs():
    """Hair-cap template on an overlapping grid (cell s, stride s/2): the face is the centre column
    over rows 2-3 (s wide, 1.5s tall) and must be skin; the row above it (row 0, columns 1-3, i.e. the
    hair, hat or helmet) must be dark; the cells beside the face (columns 0 and 4, rows 2-3) must not
    be skin."""
    kt = K(); kt[CH["D"], 0, 1:4] = 1.0 / 3                       # hair fills ~half of the cell above
    krel = K(); krel[CH["Y"], 2:4, 2] = 1.0 / 2; krel[CH["Y"], 0, 1:4] = -1.0 / 3   # hair darker than face
    ks = K(); ks[CH["S"], 2:4, 2] = 1.0 / 2
    kside = K(); kside[CH["S"], 2:4, 0] = 1.0 / 4; kside[CH["S"], 2:4, 4] = 1.0 / 4
    return (clip_pair(kt, 0.2, 0.45) + clip_pair(krel, 0.1, 0.25) + clip_pair(ks, 0.5, 0.8)
            + clip_pair(-kside, -0.4, -0.1)), 4


def face_pairs():
    """Two-eye template on a non-overlapping grid with cell c.  Columns 0 and 4 hold the two eyes:
    row 2 is darker than row 4 (cheeks) and than row 0 (forehead) unless row 0 is dark hair (a fringe);
    the band is thin (darker than the mean of rows 1 and 3); the cheek row is skin; the nose bridge
    (row 2, column 2) is brighter than the two eyes.  Returns (pairs, number of AND terms)."""
    pairs = []
    for j in (0, 4):
        ka = K(); ka[CH["Y"], 0, j] = 1.0; ka[CH["Y"], 2, j] = -1.0
        kd = K(); kd[CH["D"], 0, j] = 1.0
        kb = K(); kb[CH["Y"], 4, j] = 1.0; kb[CH["Y"], 2, j] = -1.0
        pairs += clip_pair(ka, 0.06, 0.15) + clip_pair(kd, 0.4, 0.7)      # one AND term (an OR of the two)
        pairs += clip_pair(kb, 0.06, 0.15)
    km = K()
    for j in (0, 4):
        km[CH["Y"], 1, j] = 1.0 / 4; km[CH["Y"], 3, j] = 1.0 / 4; km[CH["Y"], 2, j] = -1.0 / 2
    pairs += clip_pair(km, 0.02, 0.06)
    kg = K()
    for j in (0, 2, 4):
        kg[CH["S"], 4, j] = 1.0 / 3
    pairs += clip_pair(kg, 0.5, 0.8)
    kn = K(); kn[CH["Y"], 2, 2] = 1.0; kn[CH["Y"], 2, 0] = -0.5; kn[CH["Y"], 2, 4] = -0.5
    pairs += clip_pair(kn, 0.03, 0.08)
    return pairs, 7


def fig_kernels():
    # vertical figure: centre column of 5 cells versus the two side columns (one cell away)
    kv = K()
    kv[CH["Y"], :, 2] = 1.0 / 5
    kv[CH["Y"], :, 0] = -1.0 / 10
    kv[CH["Y"], :, 4] = -1.0 / 10
    gates = {}
    for nm, chans in [("snow", ["W"]), ("water", ["B"]), ("field", ["G", "SW"])]:
        kg = K()
        for c in chans:
            kg[CH[c], :, 0] = 1.0 / 10
            kg[CH[c], :, 4] = 1.0 / 10
        gates[nm] = kg
    # compact dark blob (a crouching surfer): centre cell darker than the ring, ring blue or white
    kb = K()
    kb[CH["Y"], 2, 2] = -1.0
    kr = K()
    for i in range(5):
        for j in range(5):
            if max(abs(i - 2), abs(j - 2)) == 2:
                kb[CH["Y"], i, j] = 1.0 / 16
                kr[CH["B"], i, j] = 1.0 / 16; kr[CH["W"], i, j] = 1.0 / 16
    return kv, gates, kb, kr


FIG_NAMES = ("snow", "water", "field")
LOWER_ONLY = {"snow", "water"}     # read these from the lower half of the image only
BLOB_SCALE_W = {"L8": 1.0, "L12": 0.8, "L16": 1.0, "L32": 1.0}
BLOB_GROUP = {"L8": "blob", "L12": "blob", "L16": "bigblob", "L32": "bigblob"}   # 8-12 px: hands, small faces;
#                                                                                 16-32 px: large skin-coloured objects


def level_detectors(tag, inp, map_hw, want):
    """conv5x5 -> relu -> conv1x1 -> relu [-> isolation conv -> relu] -> global max -> placement."""
    pre = []      # (kernel (7,5,5), bias)
    comb = []     # (name, [(index, coeff)], bias)

    def add_det(name, pairs, bias):
        idx0 = len(pre)
        pre.extend(pairs)
        comb.append((name, [(idx0 + t, (1.0 if t % 2 == 0 else -1.0)) for t in range(len(pairs))], bias))

    if "blob" in want:
        add_det("blob", clip_pair(blob_kernel(), 0.5, 0.9), 0.0)
    if "head" in want:
        pairs, nterms = head_pairs()
        add_det("head", pairs, -(nterms - 1.0))
    if "face" in want:
        pairs, nterms = face_pairs()
        add_det("face", pairs, -(nterms - 1.0))
    if any(w in want for w in FIG_NAMES):
        kv, gates, kb, kr = fig_kernels()
        absv = clip_pair(kv, 0.10, 0.25) + clip_pair(-kv, 0.10, 0.25)   # |V| clipped (sum of two clips)
        if "snow" in want:
            add_det("snow", absv + clip_pair(gates["snow"], 0.35, 0.65), -1.0)
        if "water" in want:
            add_det("water", absv + clip_pair(gates["water"], 0.35, 0.65), -1.0)
            add_det("water", clip_pair(kb, 0.15, 0.35) + clip_pair(kr, 0.45, 0.75), -1.0)
        if "field" in want:
            add_det("field", absv + clip_pair(gates["field"], 0.35, 0.65), -1.0)
    n_pre = len(pre)
    W1 = np.zeros((n_pre, NCH, 5, 5), np.float32)
    b1 = np.zeros((n_pre,), np.float32)
    for i, (k, bb) in enumerate(pre):
        W1[i] = k; b1[i] = bb
    r1 = relu(tag + "_d_r", conv(tag + "_d", inp, W1, b1))
    n_out = len(comb)
    W2 = np.zeros((n_out, n_pre, 1, 1), np.float32)
    b2 = np.zeros((n_out,), np.float32)
    for o, (name, terms, bb) in enumerate(comb):
        for idx, c in terms:
            W2[o, idx, 0, 0] = c
        b2[o] = bb
    r2 = relu(tag + "_c_r", conv(tag + "_c", r1, W2, b2))
    h, w = map_hw[0] - 4, map_hw[1] - 4
    names = [c[0] for c in comb]
    fig_idx = [o for o, nm in enumerate(names) if nm in FIG_NAMES]
    if fig_idx:
        # vertical isolation: a figure response must not continue 3 cells above and below (poles,
        # door frames, tree trunks and towers do; people do not).  Zero padding = nothing there.
        W3 = np.zeros((n_out, n_out, 7, 7), np.float32); b3 = np.zeros((n_out,), np.float32)
        for o in range(n_out):
            W3[o, o, 3, 3] = 1.0
            if o in fig_idx:
                W3[o, o, 0, 3] = -0.5; W3[o, o, 6, 3] = -0.5
        r2 = relu(tag + "_i_r", conv(tag + "_i", r2, W3, b3, padding=3))
    full = [o for o, nm in enumerate(names) if nm not in LOWER_ONLY]
    lower = [o for o, nm in enumerate(names) if nm in LOWER_ONLY]
    if full:
        gmax = maxpool(tag + "_max", r2, [h, w], [h, w])          # (n_out,1,1)
        entries = []
        for o in full:
            nm = names[o]
            coef = BLOB_SCALE_W.get(tag, 1.0) if nm == "blob" else 1.0
            grp = BLOB_GROUP.get(tag, "blob") if nm == "blob" else nm
            entries.append((grp, o, coef)); entries.append((grp + "-1", o, coef))
        place(tag + "_pl", gmax, n_out, entries)
    if lower:
        hh = (h + 1) // 2
        st = h - hh
        hv = maxpool(tag + "_half", r2, [hh, w], [st, w])      # (n_out, 2, 1): upper / lower half maxima
        entries = []
        for o in lower:
            j = o * 2 + 1
            entries.append((names[o], j, 1.0)); entries.append((names[o] + "-1", j, 1.0))
        place(tag + "_plw", hv, n_out * 2, entries)


def head_level(tag, inp, map_hw):
    """Hair-cap head template as a 9x9 kernel on an overlapping grid (cell s, stride s/2; index 4 =
    centre).  Face: rows 4-8 x columns 3-5 (2s wide, 3s tall) must be skin.  Hair / hat / helmet:
    row 2 x columns 3-5 (an s-tall strip directly above the face) must be dark and darker than the
    face.  Beside the face (columns 0-1 and 7-8, rows 4-8) there must be no skin."""
    def K9():
        return np.zeros((NCH, 9, 9), np.float32)
    kt = K9(); kt[CH["D"], 2, 3:6] = 1.0 / 3
    krel = K9(); krel[CH["Y"], 4:9, 3:6] = 1.0 / 15; krel[CH["Y"], 2, 3:6] = -1.0 / 3
    ks = K9(); ks[CH["S"], 4:9, 3:6] = 1.0 / 15
    kside = K9(); kside[CH["S"], 4:9, 0:2] = 1.0 / 20; kside[CH["S"], 4:9, 7:9] = 1.0 / 20
    keye = K9(); keye[CH["D"], 4:6, 3:6] = 1.0 / 6                 # eyes / brows: some darkness in the upper face
    pairs = (clip_pair(kt, 0.3, 0.55) + clip_pair(krel, 0.1, 0.25) + clip_pair(ks, 0.5, 0.8)
             + clip_pair(-kside, -0.4, -0.1) + clip_pair(keye, 0.03, 0.10))
    W1 = np.zeros((len(pairs), NCH, 9, 9), np.float32); b1 = np.zeros((len(pairs),), np.float32)
    for i, (k, bb) in enumerate(pairs):
        W1[i] = k; b1[i] = bb
    r1 = relu(tag + "_d_r", conv(tag + "_d", inp, W1, b1))
    W2 = np.zeros((1, len(pairs), 1, 1), np.float32); b2 = np.array([-4.0], np.float32)
    for t in range(len(pairs)):
        W2[0, t, 0, 0] = 1.0 if t % 2 == 0 else -1.0
    r2 = relu(tag + "_c_r", conv(tag + "_c", r1, W2, b2))
    h, w = map_hw[0] - 8, map_hw[1] - 8
    gmax = maxpool(tag + "_max", r2, [h, w], [h, w])
    place(tag + "_pl", gmax, 1, [("head", 0, 1.0), ("head-1", 0, 1.0)])


head_level("H6", q6, (31, 31))
head_level("H8", q8, (23, 23))
head_level("H12", q12, (15, 15))
head_level("H16", q16, (11, 11))
level_detectors("L8", q8, (23, 23), {"blob", "snow", "water", "field"})
level_detectors("L12", q12, (15, 15), {"blob", "snow", "water", "field"})
level_detectors("L16", q16, (11, 11), {"blob", "snow", "water", "field"})
level_detectors("L32", q32, (5, 5), {"blob"})
level_detectors("F3", p3, (32, 32), {"face"})
level_detectors("F4", p4, (24, 24), {"face"})
level_detectors("F6", p6, (16, 16), {"face"})
level_detectors("F8", p8, (12, 12), {"face"})
level_detectors("F12", p12, (8, 8), {"face"})

# --------------------------------------------------------------------------------------
# Court lines: thin bright ridges (4 orientations) on green / blue ground, sustained over 9 px
# --------------------------------------------------------------------------------------
pre = []
for nm, cen, sa, sb in [
        ("h", [(2, j) for j in range(5)], [(0, j) for j in range(5)], [(4, j) for j in range(5)]),
        ("v", [(i, 2) for i in range(5)], [(i, 0) for i in range(5)], [(i, 4) for i in range(5)]),
        ("d", [(t, t) for t in (1, 2, 3)], [(0, 2), (1, 3), (2, 4)], [(2, 0), (3, 1), (4, 2)]),
        ("a", [(t, 4 - t) for t in (1, 2, 3)], [(0, 2), (1, 1), (2, 0)], [(2, 4), (3, 3), (4, 2)])]:
    for side in (sa, sb):
        k = K()
        for (i, j) in cen: k[CH["Y"], i, j] += 1.0 / len(cen)
        for (i, j) in side: k[CH["Y"], i, j] -= 1.0 / len(side)
        pre.extend(clip_pair(k, 0.05, 0.15))
kc = K()
for c in ("G", "B", "SW"):
    kc[CH[c], :, :] = 1.0 / 25
pre.extend(clip_pair(kc, 0.3, 0.6))
W1 = np.zeros((len(pre), NCH, 5, 5), np.float32); b1 = np.zeros((len(pre),), np.float32)
for i, (k, bb) in enumerate(pre):
    W1[i] = k; b1[i] = bb
rr = relu("ridge_r", conv("ridge", pf, W1, b1))                       # (18, 92, 92)
W2 = np.zeros((4, len(pre), 1, 1), np.float32); b2 = np.full((4,), -2.0, np.float32)
for o in range(4):
    base = o * 4
    for t in range(4):
        W2[o, base + t, 0, 0] = 1.0 if t % 2 == 0 else -1.0           # clip(side a) + clip(side b)
    W2[o, 16, 0, 0] = 1.0; W2[o, 17, 0, 0] = -1.0                      # + clip(colour gate)
lr = relu("lines_r", conv("lines", rr, W2, b2))                       # (4, 92, 92) in [0,1]
L = 9
W3 = np.zeros((4, 4, L, L), np.float32); b3 = np.full((4,), -0.25 / 0.4, np.float32)
for t in range(L):
    W3[0, 0, L // 2, t] = 1.0 / L / 0.4
    W3[1, 1, t, L // 2] = 1.0 / L / 0.4
    W3[2, 2, t, t] = 1.0 / L / 0.4
    W3[3, 3, t, L - 1 - t] = 1.0 / L / 0.4
ll = relu("linelen_r", conv("linelen", lr, W3, b3))                   # (4, 84, 84)
W4 = np.ones((1, 4, 1, 1), np.float32); b4 = np.zeros((1,), np.float32)
lsum = conv("linesum", ll, W4, b4)                                    # (1, 84, 84)
lmean = avgpool("linemean", lsum, [84, 84], [84, 84])                 # (1,1,1): fraction of line pixels
LINE_LO, LINE_HI = 0.008, 0.03                                        # line amount: clip((m - lo)/(hi - lo))
place("lines_pl", lmean, 1, [("lines", 0, 1 / (LINE_HI - LINE_LO)), ("lines-1", 0, 1 / (LINE_HI - LINE_LO))],
      extra_bias={"lines": -LINE_LO / (LINE_HI - LINE_LO), "lines-1": -LINE_LO / (LINE_HI - LINE_LO)})

# --------------------------------------------------------------------------------------
# Whole-image fractions: gates for the context cues, and penalties for too much skin-coloured
# area (wood, sand, bread), saturated warm colour (food, signs, vehicles) or white (bathrooms,
# kitchens, plates) without anything else speaking for a person.
# --------------------------------------------------------------------------------------
Wg = np.zeros((NSLOT, NCH), np.float32); bg = np.zeros((NSLOT,), np.float32)


def frac_gate(name, chans, lo, hi):
    for c in chans:
        Wg[slot(name), CH[c]] = 1 / (hi - lo); Wg[slot(name + "-1"), CH[c]] = 1 / (hi - lo)
    bg[slot(name)] += -lo / (hi - lo); bg[slot(name + "-1")] += -lo / (hi - lo)


frac_gate("wfrac", ["W"], 0.20, 0.40)              # snow scenes are largely bright white
frac_gate("bfrac", ["B"], 0.20, 0.40)              # surf scenes are largely water
frac_gate("cfrac", ["G", "B", "SW"], 0.30, 0.50)   # courts fill much of the picture
frac_gate("gfrac", ["G", "SW"], 0.25, 0.45)        # fields fill much of the picture
frac_gate("white_excess", ["W"], 0.40, 0.70)
frac_gate("skin_amount", ["S"], 0.02, 0.10)        # some skin-coloured area at all
Wg[slot("skin_excess"), CH["S"]] = 1 / 0.3; bg[slot("skin_excess")] = -0.55 / 0.3
Wg[slot("warm_excess"), CH["SW"]] = 1 / 0.25; bg[slot("warm_excess")] = -0.12 / 0.25
placements.append(dense("gavg_pl", gavg, Wg, bg))

# --------------------------------------------------------------------------------------
# Readout: stage-1 units -> stage-2 (clipped ORs, ANDs with the whole-image gates) -> scores
# --------------------------------------------------------------------------------------
summed = add("slots", placements)
u1 = relu("slots_r", summed)

stage2 = ["blob", "bigblob", "head", "face", "field", "snow", "water", "lines", "skin_excess", "warm_excess",
          "white_excess", "skin_amount", "animal", "plate"]
W2 = np.zeros((len(stage2), NSLOT), np.float32); b2 = np.zeros((len(stage2),), np.float32)


def orval(row, g, c=1.0):
    W2[row, slot(g)] += c; W2[row, slot(g + "-1")] -= c


for r, nm in enumerate(stage2):
    if nm in ("blob", "bigblob", "head", "face", "white_excess", "skin_amount"):
        orval(r, nm)
    elif nm == "animal":
        orval(r, "bigblob"); orval(r, "gfrac"); b2[r] = -1.0    # large skin-coloured blob on grass: an animal
    elif nm == "plate":
        orval(r, "wfrac"); orval(r, "skin_amount"); b2[r] = -1.0  # brown food on a white plate / cloth
    elif nm == "field":
        orval(r, "field"); orval(r, "gfrac"); b2[r] = -1.0      # figure on grass AND mostly-green picture
    elif nm == "snow":
        orval(r, "snow"); orval(r, "wfrac"); b2[r] = -1.0       # figure on snow AND mostly-white picture
    elif nm == "water":
        orval(r, "water"); orval(r, "bfrac"); b2[r] = -1.0      # figure on water AND mostly-blue picture
    elif nm == "lines":
        orval(r, "lines"); orval(r, "cfrac"); b2[r] = -1.0      # court lines AND mostly green/blue picture
    else:
        W2[r, slot(nm)] = 1.0
u2 = relu("stage2_r", dense("stage2", u1, W2, b2))

FINAL_W = {
    "blob": 0.7, "bigblob": -0.3, "head": 1.3, "face": 2.1, "field": 1.5, "snow": 2.4, "water": 2.1, "lines": 2.1,
    "skin_excess": -0.8, "warm_excess": -1.0, "white_excess": -0.8, "skin_amount": 0.4, "animal": -1.0, "plate": -0.4,
}
THETA = 1.7
Wf = np.zeros((2, len(stage2)), np.float32); bf = np.zeros((2,), np.float32)
for r, nm in enumerate(stage2):
    Wf[1, r] = FINAL_W[nm]
bf[0] = THETA
dense("out", u2, Wf, bf)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **weights)
    n = sum(int(np.prod(v.shape)) for v in weights.values())
    print(f"wrote {len(layers)} layers, {n:,} weights, {NSLOT} stage-1 units")


if __name__ == "__main__":
    main()
