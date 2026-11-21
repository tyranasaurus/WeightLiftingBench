#!/usr/bin/env python3
"""build.py NEW_FOLDER  ->  writes NEW_FOLDER/network.json and NEW_FOLDER/weights.npz

Hand-designed person / no-person network for 96x96 colour photos. Every number below is typed
from general knowledge (skin-colour rules, face and body proportions, logic gates built from
ReLUs); nothing is read from any file and nothing was fitted to data.

Overview
  pixel stage (1x1 convs, full resolution)
      skin map s   : soft AND of linear colour inequalities (R>G, warm hue <= ~30deg,
                     saturation 0.15..0.65, not too dark, not red/magenta)
      colour maps  : R, G, B carried along for luminance / chroma contrasts
  pyramid         : 96 -> 48 -> 24 -> 12 by 2x2 average pooling
  per level (window convs, then 1x1 "logic" convs built from ReLU min / max / clip gates)
      blob  : compact skin region: 3x3 centre mostly skin, and the max over the four 3-wide
              surround bands (above / below / left / right) mostly not skin
      face  : blob gated by face structure: a cap region directly above that is darker than
              the blob (hair, hat) and / or a dark horizontal line through the blob (eyes, mouth)
      fig   : vertical figure: a 3-wide, 7-tall body whose upper and lower parts both contrast
              (same sign) with both side backgrounds; background directly above the head, on
              both flanks and below the feet; and a smooth surround (low edge energy in a ring)
  readout         : global max pooling, clipping to [0, 1], hand-set linear combination
  Border safety   : zero padding outside the image can only suppress a detection, never create
                    one (contrasts are mins over sides / body parts, surround a max over bands,
                    the eye line a min of two half differences, the cap needs non-zero luminance).
"""
import json, sys, os
import numpy as np


# ----------------------------------------------------------------------------- builder
class Net:
    def __init__(self):
        self.layers, self.W, self.ch = [], {}, {"image": 3}

    def _add(self, name, op, inputs, ch, **kw):
        L = {"name": name, "op": op, "inputs": list(inputs)}
        L.update(kw)
        self.layers.append(L)
        self.ch[name] = ch
        return name

    def conv(self, name, inp, W, b=None, stride=1, padding=0):
        W = np.asarray(W, np.float64)
        o, i, k, k2 = W.shape
        assert k == k2 and i == self.ch[inp], (name, W.shape, self.ch[inp])
        if b is None:
            b = np.zeros(o)
        self.W[name + ".W"] = W.astype(np.float32)
        self.W[name + ".b"] = np.asarray(b, np.float64).astype(np.float32)
        return self._add(name, "conv", [inp], o, out=int(o), kernel=[int(k), int(k)], stride=int(stride), padding=int(padding))

    def dense(self, name, inp, W, b=None):
        W = np.asarray(W, np.float64)
        if b is None:
            b = np.zeros(W.shape[0])
        self.W[name + ".W"] = W.astype(np.float32)
        self.W[name + ".b"] = np.asarray(b, np.float64).astype(np.float32)
        return self._add(name, "dense", [inp], W.shape[0], out=int(W.shape[0]))

    def relu(self, name, inp):
        return self._add(name, "relu", [inp], self.ch[inp])

    def maxpool(self, name, inp, size, stride):
        return self._add(name, "maxpool", [inp], self.ch[inp], size=list(size), stride=list(stride))

    def avgpool(self, name, inp, size, stride):
        return self._add(name, "avgpool", [inp], self.ch[inp], size=list(size), stride=list(stride))

    def add(self, name, inputs):
        return self._add(name, "add", list(inputs), self.ch[inputs[0]])

    def conv_relu(self, name, inp, W, b=None, stride=1, padding=0):
        self.conv(name, inp, W, b, stride, padding)
        return self.relu(name + "_r", name)

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "network.json"), "w") as f:
            json.dump({"layers": self.layers}, f, indent=1)
        np.savez(os.path.join(folder, "weights.npz"), **self.W)


class Chain:
    """Sequence of 1x1 conv + ReLU stages written with named channels.
    stage(exprs): exprs = [(out_name, {in_name: coef, ...}, bias), ...]; every output passes a ReLU,
    so a pass-through ('x': {'x': 1}) is exact only for non-negative values (all our values are)."""

    def __init__(self, net, pfx, src, names):
        self.net, self.pfx, self.src, self.names, self.n = net, pfx, src, list(names), 0
        assert len(names) == net.ch[src], (pfx, len(names), net.ch[src])

    def stage(self, exprs, relu=True):
        W = np.zeros((len(exprs), len(self.names), 1, 1)); b = np.zeros(len(exprs))
        for o, (oname, terms, bias) in enumerate(exprs):
            for iname, coef in terms.items():
                W[o, self.names.index(iname), 0, 0] += coef
            b[o] = bias
        self.n += 1
        name = f"{self.pfx}_c{self.n}"
        if relu:
            self.src = self.net.conv_relu(name, self.src, W, b)
        else:
            self.src = self.net.conv(name, self.src, W, b)
        self.names = [e[0] for e in exprs]
        return self.src


def mask(k, rows, cols):
    m = np.zeros((k, k), bool)
    m[np.ix_(list(rows), list(cols))] = True
    return m


def mk(cin, k, ch, m):
    """k x k kernel over cin channels: mean of channel ch over the boolean mask m."""
    K = np.zeros((cin, k, k))
    K[ch][m] = 1.0 / m.sum()
    return K


def tap(cin, k, ch, dy, dx, val=1.0):
    K = np.zeros((cin, k, k))
    K[ch, dy, dx] = val
    return K


def sel(cin, chans, cout=None, offset=0):
    """1x1 conv weights copying input channels `chans` to output channels offset.. (zeros elsewhere)."""
    cout = cout or len(chans)
    W = np.zeros((cout, cin, 1, 1))
    for i, c in enumerate(chans):
        W[offset + i, c, 0, 0] = 1.0
    return W


# ----------------------------------------------------------------------------- constants
LUMA = np.array([0.299, 0.587, 0.114])      # Rec. 601 luminance

# Skin colour model: constraints  a.(R,G,B) + d >= 0 ; violation = relu(-gain * (a.x + d)) ;
# gain = 1 / soft margin.  Typed from the usual RGB / HSV skin rules.
SKIN_RULES = [
    # (a_R,   a_G,   a_B,    d,     gain)   meaning
    (1.0,  -1.0,   0.0,  -0.06,  25.0),  # R - G >= 0.06          : skin is clearly redder than green
    (0.0,   1.0,  -1.0,   0.02,  25.0),  # G - B >= -0.02         : warm, not magenta / pink
    (0.45, -1.0,   0.55,  0.00,  20.0),  # (G - B) <= 0.45 (R - B): hue <= ~27 deg (soft to ~37)
    (0.80,  0.0,  -1.0,   0.00,  25.0),  # (R - B) >= 0.20 R      : saturation >= 0.20 (not grey / beige / white)
    (-0.35, 0.0,   1.0,   0.00,  33.0),  # (R - B) <= 0.65 R      : saturation <= 0.65 (not orange)
    (1.0,   0.0,   0.0,  -0.30,  12.5),  # R >= 0.30              : bright enough to judge colour
    (-0.09, 1.0,  -0.91,  0.02,  25.0),  # (G - B) >= 0.09 (R - B): hue >= ~5 deg (not pure red)
]
N_RULES = len(SKIN_RULES)

# blob logic (means of the skin map over a 3x3 centre and four 3-wide surround bands)
BLOB_C0, BLOB_CG = 0.45, 1.0 / 0.35      # centre fullness ramp 0.45 -> 0.80
RING_T, RING_G = 0.30, 1.0 / 0.20        # band penalty ramp 0.30 -> 0.50
# face structure
CAP_T, CAP_G = 0.10, 1.0 / 0.20          # cap darker than the blob by 0.10 -> 0.30 in luminance
CAP_DARK_T, CAP_DARK_G = 0.40, 1.0 / 0.15   # and dark in absolute terms: luminance 0.40 -> 0.25
CAP_PRESENT = 50.0                       # cap luminance must exceed 0.02 (zero means outside the image)
EYE_T, EYE_G = 0.04, 1.0 / 0.08          # dark horizontal line: dip of 0.04 -> 0.12 in luminance on both sides
FACE_W = 0.7                             # face = blob - relu(1 - 0.7 cap - 0.7 eye)

# figure logic
FIG_ABOVE = 0.8                          # penalty on contrast directly above the head
FIG_FLANK = 0.35                         # penalty on contrast in the columns flanking the body (each side)
FIG_BELOW = 0.3                          # penalty on contrast directly below the feet
FIG_T = 0.10                             # contrast threshold
FIG_GAIN = 5.0                           # response reaches 1 at net contrast 0.30
EDGE_T, EDGE_SLOPE = 0.04, 30.0          # smooth-surround gate: penalty 30 * relu(ring edge energy - 0.04)

# snow: bright, colourless, smooth cells (48 x 48 level) in the lower 70 % of the picture
SNOW_Y, SNOW_YG = 0.72, 10.0             # luminance >= 0.72 (ramp to 0.62)
SNOW_C1, SNOW_C2, SNOW_CG = 0.06, 0.04, 25.0   # |R - B| <= 0.12, |2G - R - B| <= 0.16
SNOW_E, SNOW_EG = 0.035, 40.0            # edge energy <= 0.035
SNOW_T, SNOW_SCALE = 0.30, 0.30          # fraction 0.30 -> 0.60 ramps the feature 0 -> 1

# readout
W_BLOB = {1: 0.10, 2: 0.25, 4: 0.25, 8: 0.20}
W_FACE = {2: 1.00, 4: 1.00, 8: 1.00}
FACE_SCALE = 0.8                         # face feature reaches 1 at raw value 0.8
W_FIG = {2: 1.00, 4: 1.00}
W_SNOW = 0.60
W_SKIN_LO, SKIN_LO_SCALE = 0.10, 0.10                     # skin fraction up to 10 % counts a little
W_SKIN_HI, SKIN_HI_T, SKIN_HI_SCALE = -0.50, 0.30, 0.20   # skin fraction above 30 % counts against
THETA = 0.95

FEAT = ["bm1", "bm2", "bm4", "bm8", "fm2", "fm4", "fm8", "gm2", "gm4", "skin", "snow"]
NF = len(FEAT)


def slot(name):
    return FEAT.index(name)


def feat_dense(net, name, inp, assign):
    """dense layer scattering input channels into feature slots: assign = {slot_name: channel}."""
    Wd = np.zeros((NF, net.ch[inp]))
    for s, c in assign.items():
        Wd[slot(s), c] = 1.0
    net.dense(name, inp, Wd)
    return name


# ----------------------------------------------------------------------------- network
def build():
    net = Net()

    # ---- pixel stage ---------------------------------------------------------------
    W = np.zeros((N_RULES, 3, 1, 1)); b = np.zeros(N_RULES)
    for i, (ar, ag, ab, d, g) in enumerate(SKIN_RULES):
        W[i, :, 0, 0] = -g * np.array([ar, ag, ab])
        b[i] = -g * d
    net.conv_relu("pix", "image", W, b)                         # skin violations
    W = np.zeros((4, N_RULES, 1, 1)); b = np.zeros(4)
    W[0, :, 0, 0] = -1.0; b[0] = 1.0
    net.conv("maps_a", "pix_r", W, b)
    net.conv("maps_b", "image", sel(3, [0, 1, 2], cout=4, offset=1))
    net.add("maps", ["maps_a", "maps_b"])
    net.relu("maps_r", "maps")          # [s, R, G, B] all in [0, 1]

    # ---- pyramid -------------------------------------------------------------------
    net.avgpool("L2", "maps_r", [2, 2], [2, 2])   # 48 x 48
    net.avgpool("L4", "L2", [2, 2], [2, 2])       # 24 x 24
    net.avgpool("L8", "L4", [2, 2], [2, 2])       # 12 x 12

    feats = []
    net.avgpool("skin_avg", "maps_r", [96, 96], [96, 96])     # (4, 1, 1)
    feats.append(feat_dense(net, "feat_skin", "skin_avg", {"skin": 0}))

    # ---- level 1: skin blobs at full resolution (3x3 px centre), stride 2 ----------
    net.conv("s1", "maps_r", sel(4, [0]))                       # (1, 96, 96)
    net.conv_relu("b1_big", "s1", blob_kernels(1, 9, 0), stride=2, padding=4)       # (5, 48, 48)
    blob1 = blob_chain(net, "b1", "b1_big_r", BLOB_NAMES)
    net.maxpool("b1_max", blob1, [48, 48], [48, 48])
    feats.append(feat_dense(net, "b1_fmax", "b1_max", {"bm1": 0}))

    feats += level2_branch(net)
    feats += level_branch(net, "L4", 4, with_fig=True)
    feats += level_branch(net, "L8", 8, with_fig=False)

    net.add("feat", feats)                                      # (NF,)
    readout(net, "feat")
    return net


def pre_convs(net, pfx, lname):
    """3x3 conv on [s, R, G, B] -> [s, Y, c1', c2', ex+, ex-, ey+, ey-], then 1x1 -> [s, Y, c1', c2', e]."""
    W = np.zeros((8, 4, 3, 3)); b = np.zeros(8)
    W[0] = tap(4, 3, 0, 1, 1)
    for c in range(3):
        W[1, 1 + c, 1, 1] = LUMA[c]
    W[2, 1, 1, 1], W[2, 3, 1, 1], b[2] = 0.5, -0.5, 0.5                       # c1' = (R - B + 1) / 2
    W[3, 2, 1, 1], W[3, 1, 1, 1], W[3, 3, 1, 1], b[3] = 0.5, -0.25, -0.25, 0.5  # c2' = (2G - R - B + 2) / 4
    for c in range(3):   # central differences of Y, split into positive / negative parts
        W[4, 1 + c, 1, 2] += 0.5 * LUMA[c]; W[4, 1 + c, 1, 0] -= 0.5 * LUMA[c]
        W[5, 1 + c, 1, 2] -= 0.5 * LUMA[c]; W[5, 1 + c, 1, 0] += 0.5 * LUMA[c]
        W[6, 1 + c, 2, 1] += 0.5 * LUMA[c]; W[6, 1 + c, 0, 1] -= 0.5 * LUMA[c]
        W[7, 1 + c, 2, 1] -= 0.5 * LUMA[c]; W[7, 1 + c, 0, 1] += 0.5 * LUMA[c]
    net.conv_relu(pfx + "_pre", lname, W, b, padding=1)
    W = np.zeros((5, 8, 1, 1))
    for i in range(4):
        W[i, i] = 1.0
    W[4, 4:8] = 1.0
    net.conv_relu(pfx + "_pre2", pfx + "_pre_r", W)
    return pfx + "_pre2_r"


# ---- window kernels -----------------------------------------------------------------
BLOB_NAMES = ["cen", "band_t", "band_b", "band_l", "band_r"]
FACE_NAMES = ["caprel", "ycap", "e3a", "e3b", "e4a", "e4b", "e5a", "e5b"]
FIG_PARTS = ["top", "bot", "tor"]                      # upper box, lower box, torso outer columns 4 and 6 (rows 4-6)
FIG_POS = [f"{p}{s}" for p in FIG_PARTS for s in "lr"]  # 10 (part, side) contrasts
FIG_NAMES = FIG_POS + ["n" + n for n in FIG_POS] + ["ab", "nab", "fl", "nfl", "fr", "nfr", "be", "nbe"]


def blob_kernels(cin, k, ch):
    """[centre 3x3 mean, band top, band bottom, band left, band right] of channel ch in a k x k window
    (k = 9: centre rows 3-5, bands at the border; k = 11: centre rows 4-6, bands one cell in)."""
    c = k // 2
    cen = mask(k, (c - 1, c, c + 1), (c - 1, c, c + 1))
    o = 1 if k == 11 else 0
    span = range(o, k - o)
    bands = [mask(k, (o, o + 1, o + 2), span), mask(k, (k - 3 - o, k - 2 - o, k - 1 - o), span),
             mask(k, span, (o, o + 1, o + 2)), mask(k, span, (k - 3 - o, k - 2 - o, k - 1 - o))]
    return np.stack([mk(cin, k, ch, cen)] + [mk(cin, k, ch, m) for m in bands])


def face_kernels(cin, k, ch):
    """Luminance channel ch of an 11x11 window: [caprel = centre - cap, ycap = cap mean,
    for r in 3,4,5: (row r-1 - row r), (row r+1 - row r)] (eye line = min of the two halves)."""
    cen = mask(k, (4, 5, 6), (4, 5, 6))
    cap = mask(k, (1, 2, 3), (4, 5, 6))
    outs = [mk(cin, k, ch, cen) - mk(cin, k, ch, cap), mk(cin, k, ch, cap)]
    for r in (3, 4, 5):
        row = lambda rr: mk(cin, k, ch, mask(k, (rr,), (4, 5, 6)))
        outs.append(row(r - 1) - row(r))
        outs.append(row(r + 1) - row(r))
    return outs


def fig_kernels(cin, k, ch):
    """One contrast channel of an 11x11 window. Body columns 4-6; upper box rows 2-3, lower box rows 7-8,
    torso rows 4-6 in the two outer body columns (a thin pole fails this); side references left cols 0-2 / right cols 8-10 (rows 2-8).
    Outputs (FIG_NAMES): +(part - side) for the 10 (part, side) pairs, their negatives, then
    +-(above - sides) [rows 0-1, cols 3-7]; +-(flank col 3 - left side); +-(flank col 7 - right side);
    +-(below - sides) [rows 9-10, cols 3-7]."""
    parts = {"top": mask(k, (2, 3), (4, 5, 6)), "bot": mask(k, (7, 8), (4, 5, 6)), "tor": mask(k, (4, 5, 6), (4, 6))}
    left, right = mask(k, range(2, 9), (0, 1, 2)), mask(k, range(2, 9), (8, 9, 10))
    above, below = mask(k, (0, 1), (3, 4, 5, 6, 7)), mask(k, (9, 10), (3, 4, 5, 6, 7))
    flank_l, flank_r = mask(k, range(2, 9), (3,)), mask(k, range(2, 9), (7,))
    M = lambda m: mk(cin, k, ch, m)
    sides = 0.5 * (M(left) + M(right))
    pos = [M(parts[p]) - M(sd) for p in FIG_PARTS for sd in (left, right)]
    outs = pos + [-p for p in pos]
    for d in (M(above) - sides, M(flank_l) - M(left), M(flank_r) - M(right), M(below) - sides):
        outs += [d, -d]
    return outs


def ring_kernel(cin, k, ch):
    outer = np.zeros((k, k), bool); outer[0, :] = outer[k - 1, :] = True; outer[:, 0] = outer[:, k - 1] = True
    return mk(cin, k, ch, outer)


# ---- logic chains (1x1 convs) ---------------------------------------------------------
def blob_chain(net, pfx, src, names):
    """blob = clip((cen - 0.45)/0.35) - max over the 4 bands of relu((band - 0.30)/0.20). Returns a 1-channel layer."""
    c = Chain(net, pfx, src, names)
    s1 = [("u", {"cen": BLOB_CG}, -BLOB_C0 * BLOB_CG), ("u1", {"cen": BLOB_CG}, -BLOB_C0 * BLOB_CG - 1.0)]
    s1 += [(f"r{b}", {f"band_{b}": RING_G}, -RING_T * RING_G) for b in "tblr"]
    c.stage(s1)
    c.stage([("cu", {"u": 1.0, "u1": -1.0}, 0.0), ("rt", {"rt": 1.0}, 0.0), ("rl", {"rl": 1.0}, 0.0),
             ("dtb", {"rb": 1.0, "rt": -1.0}, 0.0), ("dlr", {"rr": 1.0, "rl": -1.0}, 0.0)])
    c.stage([("cu", {"cu": 1.0}, 0.0), ("M1", {"rt": 1.0, "dtb": 1.0}, 0.0),
             ("dM", {"rl": 1.0, "dlr": 1.0, "rt": -1.0, "dtb": -1.0}, 0.0)])
    return c.stage([("blob", {"cu": 1.0, "M1": -1.0, "dM": -1.0}, 0.0)])


def face_gap_chain(net, pfx, src, names):
    """Face-structure evidence from the luminance window outputs (FACE_NAMES):
      cap = clip(ramp(caprel)) AND clip(ramp(0.40 - ycap)) AND ycap > 0.02 ;  eye = clip(max_r ramp(min(e_ra, e_rb)))
      gap = relu(1 - 0.7 cap - 0.7 eye)        (face = relu(blob - gap), so face <= blob)
    Returns a 1-channel layer."""
    c = Chain(net, pfx, src, names)
    s1 = [("cap", {"caprel": CAP_G}, -CAP_T * CAP_G), ("cap1", {"caprel": CAP_G}, -CAP_T * CAP_G - 1.0),
          ("capd", {"ycap": -CAP_DARK_G}, CAP_DARK_T * CAP_DARK_G), ("capd1", {"ycap": -CAP_DARK_G}, CAP_DARK_T * CAP_DARK_G - 1.0),
          ("capmiss", {"ycap": -CAP_PRESENT}, 1.0)]
    for r in (3, 4, 5):
        s1 += [(f"e{r}a", {f"e{r}a": 1.0}, 0.0), (f"e{r}b", {f"e{r}b": 1.0}, 0.0), (f"e{r}d", {f"e{r}b": 1.0, f"e{r}a": -1.0}, 0.0)]
    c.stage(s1)
    s2 = [("crel", {"cap": 1.0, "cap1": -1.0, "capmiss": -1.0}, 0.0), ("cnd", {"capd": -1.0, "capd1": 1.0}, 1.0)]
    s2 += [(f"m{r}", {f"e{r}b": 1.0, f"e{r}d": -1.0}, 0.0) for r in (3, 4, 5)]          # min(e_ra, e_rb)
    c.stage(s2)
    s3 = [("ccap", {"crel": 1.0, "cnd": -1.0}, 0.0)]
    s3 += [(f"g{r}", {f"m{r}": EYE_G}, -EYE_T * EYE_G) for r in (3, 4, 5)]
    s3 += [(f"g{r}1", {f"m{r}": EYE_G}, -EYE_T * EYE_G - 1.0) for r in (3, 4, 5)]
    c.stage(s3)
    c.stage([("ccap", {"ccap": 1.0}, 0.0), ("c3", {"g3": 1.0, "g31": -1.0}, 0.0),
             ("d43", {"g4": 1.0, "g41": -1.0, "g3": -1.0, "g31": 1.0}, 0.0), ("c5", {"g5": 1.0, "g51": -1.0}, 0.0)])
    c.stage([("ccap", {"ccap": 1.0}, 0.0), ("E1", {"c3": 1.0, "d43": 1.0}, 0.0), ("d5", {"c5": 1.0, "c3": -1.0, "d43": -1.0}, 0.0)])
    return c.stage([("gap", {"ccap": -FACE_W, "E1": -FACE_W, "d5": -FACE_W}, 1.0)])


def blob_face_merge(net, pfx, blob, gap):
    """[blob, face] with face = relu(blob - gap)."""
    net.conv(pfx + "_bb", blob, np.array([[[[1.0]]], [[[1.0]]]]))
    net.conv(pfx + "_gg", gap, np.array([[[[0.0]]], [[[-1.0]]]]))
    net.add(pfx + "_bf", [pfx + "_bb", pfx + "_gg"])
    return net.relu(pfx + "_bf_r", pfx + "_bf")


def min_tree(chain, groups, carry):
    """groups: list of lists of channel names (all non-negative). Reduces every group to its minimum by
    repeated pairing (min(a, b) = b - relu(b - a)); `carry` channels are passed through. Returns the list of
    final channel names (one per group)."""
    groups = [list(g) for g in groups]
    while max(len(g) for g in groups) > 1:
        # stage A: pass everything, add differences for each pair
        sA, pairs = [], []
        for gi, g in enumerate(groups):
            for j in range(0, len(g) - 1, 2):
                a, b = g[j], g[j + 1]
                sA.append((f"{b}~d", {b: 1.0, a: -1.0}, 0.0)); pairs.append((gi, a, b))
            for n in g:
                sA.append((n, {n: 1.0}, 0.0))
        sA += [(n, {n: 1.0}, 0.0) for n in carry]
        chain.stage(sA)
        # stage B: combine pairs
        sB, new_groups = [], []
        for gi, g in enumerate(groups):
            ng = []
            for j in range(0, len(g) - 1, 2):
                a, b = g[j], g[j + 1]
                sB.append((f"{b}~m", {b: 1.0, f"{b}~d": -1.0}, 0.0)); ng.append(f"{b}~m")
            if len(g) % 2 == 1:
                sB.append((g[-1], {g[-1]: 1.0}, 0.0)); ng.append(g[-1])
            new_groups.append(ng)
        sB += [(n, {n: 1.0}, 0.0) for n in carry]
        chain.stage(sB)
        groups = new_groups
    return [g[0] for g in groups]


def fig_chain(net, pfx, src, names, chans, with_edge):
    """names: channel names of src; for each contrast channel name X in chans the FIG_NAMES prefixed X_.
    fig = GAIN * (sum_X sum_sign min over the 10 (part, side) contrasts - ABOVE ab - FLANK (fl + fr) - BELOW be - T)
          - SLOPE relu(edge - EDGE_T)       (edge channel 'edge' if with_edge)
    Returns the pre-ReLU 1-channel layer."""
    c = Chain(net, pfx, src, names)
    groups, pen = [], {}
    s1 = []
    for X in chans:
        for sg in ("", "n"):
            g = [f"{X}_{sg}{n}" for n in FIG_POS]
            groups.append(g)
            s1 += [(n, {n: 1.0}, 0.0) for n in g]
        for t, w in (("ab", FIG_ABOVE), ("fl", FIG_FLANK), ("fr", FIG_FLANK), ("be", FIG_BELOW)):
            pen[f"{X}_{t}"] = w; pen[f"{X}_n{t}"] = w
    s1.append(("pen", pen, 0.0))
    carry = ["pen"]
    if with_edge:
        s1.append(("edge", {"edge": 1.0}, -EDGE_T)); carry.append("edge")
    c.stage(s1)
    mins = min_tree(c, groups, carry)
    terms = {m: FIG_GAIN for m in mins}
    terms["pen"] = -FIG_GAIN
    if with_edge:
        terms["edge"] = -EDGE_SLOPE
    return c.stage([("fig", terms, -FIG_GAIN * FIG_T)], relu=False)


# ---- levels --------------------------------------------------------------------------
def level2_branch(net):
    """48 x 48 level at stride 1 with single-channel window convs (memory friendly)."""
    pre2 = pre_convs(net, "v2", "L2")                    # [s, Y, c1', c2', e]
    net.conv("v2_s", pre2, sel(5, [0]))
    net.conv("v2_Y", pre2, sel(5, [1]))
    # snow cells: violations of bright / colourless / smooth, then snow = relu(1 - sum)
    W = np.zeros((6, 5, 1, 1)); b = np.zeros(6)
    W[0, 1], b[0] = -SNOW_YG, SNOW_Y * SNOW_YG
    W[1, 2], b[1] = SNOW_CG, -(0.5 + SNOW_C1) * SNOW_CG
    W[2, 2], b[2] = -SNOW_CG, (0.5 - SNOW_C1) * SNOW_CG
    W[3, 3], b[3] = SNOW_CG, -(0.5 + SNOW_C2) * SNOW_CG
    W[4, 3], b[4] = -SNOW_CG, (0.5 - SNOW_C2) * SNOW_CG
    W[5, 4], b[5] = SNOW_EG, -SNOW_E * SNOW_EG
    net.conv_relu("v2_snowv", pre2, W, b)
    net.conv_relu("v2_snow", "v2_snowv_r", -np.ones((1, 6, 1, 1)), np.ones(1))       # (1, 48, 48)
    Wd = np.zeros((NF, 48 * 48))
    Wd[slot("snow"), 14 * 48:] = 1.0 / (34 * 48)                                        # mean over rows 14..47
    net.dense("feat_snow", "v2_snow_r", Wd)
    net.conv_relu("v2s_big", "v2_s", blob_kernels(1, 9, 0), padding=4)                 # 5 ch
    ynames = FACE_NAMES + ["Y_" + n for n in FIG_NAMES]
    net.conv_relu("v2y_big", "v2_Y", np.stack(face_kernels(1, 11, 0) + fig_kernels(1, 11, 0)), padding=5)
    blob = blob_chain(net, "v2b", "v2s_big_r", BLOB_NAMES)
    gap = face_gap_chain(net, "v2g", "v2y_big_r", ynames)
    bf = blob_face_merge(net, "v2", blob, gap)                                         # [blob, face] 48 x 48
    net.maxpool("v2_bf_max", bf, [48, 48], [48, 48])
    out = ["feat_snow", feat_dense(net, "v2_bf_fmax", "v2_bf_max", {"bm2": 0, "fm2": 1})]

    fig_pre = fig_chain(net, "v2f", "v2y_big_r", ynames, ["Y"], with_edge=False)       # (1, 48, 48)
    net.relu("v2f_raw_r", fig_pre)
    net.maxpool("v2f_pool", "v2f_raw_r", [2, 2], [2, 2])                               # (1, 24, 24)
    pre4 = pre_convs(net, "v4", "L4")                                                  # 24 x 24
    net.conv("v4_e", pre4, sel(5, [4]))
    net.conv_relu("v2f_ring", "v4_e", ring_kernel(1, 7, 0)[None], np.array([-EDGE_T]), padding=3)
    net.conv("v2f_pen", "v2f_ring_r", np.array([[[[-EDGE_SLOPE]]]]))
    net.add("v2f_fig", ["v2f_pool", "v2f_pen"])
    net.relu("v2f_fig_r", "v2f_fig")
    net.maxpool("v2f_max", "v2f_fig_r", [24, 24], [24, 24])
    out.append(feat_dense(net, "v2f_fmax", "v2f_max", {"gm2": 0}))
    return out


def level_branch(net, lname, level, with_fig):
    """24 x 24 and 12 x 12 levels: one 5-channel 11x11 window conv at stride 1 feeding the logic chains."""
    pfx = "v%d" % level
    H = 96 // level
    pre2 = "v4_pre2_r" if level == 4 else pre_convs(net, pfx, lname)      # [s, Y, c1', c2', e]
    k = 11
    outs = list(blob_kernels(5, k, 0)) + face_kernels(5, k, 1)
    names = BLOB_NAMES + FACE_NAMES
    chans = []
    if with_fig:
        for ch, X in ((1, "Y"), (2, "C1"), (3, "C2")):
            outs += fig_kernels(5, k, ch)
            names += [f"{X}_{n}" for n in FIG_NAMES]
            chans.append(X)
        outs.append(ring_kernel(5, k, 4)); names.append("edge")
    net.conv_relu(pfx + "_big", pre2, np.stack(outs), padding=5)
    blob = blob_chain(net, pfx + "b", pfx + "_big_r", names)
    gap = face_gap_chain(net, pfx + "g", pfx + "_big_r", names)
    bf = blob_face_merge(net, pfx, blob, gap)
    net.maxpool(pfx + "_bf_max", bf, [H, H], [H, H])
    out = [feat_dense(net, pfx + "_bf_fmax", pfx + "_bf_max", {"bm%d" % level: 0, "fm%d" % level: 1})]
    if with_fig:
        fig_pre = fig_chain(net, pfx + "f", pfx + "_big_r", names, chans, with_edge=True)
        net.relu(pfx + "f_fig_r", fig_pre)
        net.maxpool(pfx + "f_max", pfx + "f_fig_r", [H, H], [H, H])
        out.append(feat_dense(net, pfx + "f_fmax", pfx + "f_max", {"gm%d" % level: 0}))
    return out


def readout(net, feat):
    """clip each max feature to [0, 1] (skin fraction: low ramp and high ramp), then a weighted sum."""
    rows = []     # (weight_vector over FEAT, bias, readout weight)

    def ramp(name, scale, offset, w):
        v = np.zeros(NF); v[slot(name)] = 1.0 / scale
        rows.append((v, -offset / scale, w))            # relu(f/scale - offset/scale)
        rows.append((v, -offset / scale - 1.0, -w))     # minus relu(... - 1)  -> clip at 1

    for lv, w in W_BLOB.items():
        ramp("bm%d" % lv, 1.0, 0.0, w)
    for lv, w in W_FACE.items():
        ramp("fm%d" % lv, FACE_SCALE, 0.0, w)
    for lv, w in W_FIG.items():
        ramp("gm%d" % lv, 1.0, 0.0, w)
    ramp("skin", SKIN_LO_SCALE, 0.0, W_SKIN_LO)
    ramp("skin", SKIN_HI_SCALE, SKIN_HI_T, W_SKIN_HI)
    ramp("snow", SNOW_SCALE, SNOW_T, W_SNOW)
    W1 = np.stack([r[0] for r in rows]); b1 = np.array([r[1] for r in rows])
    net.dense("clip", feat, W1, b1)
    net.relu("clip_r", "clip")
    W2 = np.zeros((2, len(rows))); b2 = np.zeros(2)
    W2[1] = [r[2] for r in rows]
    b2[0] = THETA
    net.dense("scores", "clip_r", W2, b2)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    net = build()
    net.save(folder)
    print("wrote", folder, "layers:", len(net.layers), "weights:", sum(int(v.size) for v in net.W.values()))
