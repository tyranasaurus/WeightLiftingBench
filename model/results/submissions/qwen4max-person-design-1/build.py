"""Hand-designed person / no-person network for 96x96 COCO photos.

All weights come from formulas (Gaussian / difference-of-Gaussian kernels) and
from typed general knowledge about how people look in photos: a skin chroma
cone, a face = skin blob with darker hair above and two dark eyes, a head above
an upright torso, face/hand-sized skin blobs surrounded by non-skin.  Large
uniform skin sheets (food, wood, sand) and horizontal skin blobs (animals) are
suppressors.  No training data, no fitting, no image-derived numbers.

Usage: python3 build.py OUT_DIR
"""
import sys, os, json
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ----------------------------------------------------------------- builder
layers = []
weights = {}
shapes = {"image": (3, 96, 96)}


def _shape(name, op, ins, st):
    s = shapes[ins[0]]
    if op == "conv":
        k = st["kernel"]; p = st.get("padding", 0); stride = st.get("stride", 1)
        return (st["out"], (s[1] + 2 * p - k[0]) // stride + 1,
                (s[2] + 2 * p - k[1]) // stride + 1)
    if op == "dense":
        return (st["out"],)
    if op in ("relu", "norm"):
        return s
    if op in ("maxpool", "avgpool"):
        (sh, sw), (th, tw) = st["size"], st["stride"]
        return (s[0], (s[1] - sh) // th + 1, (s[2] - sw) // tw + 1)
    if op == "add":
        return s
    if op == "reshape":
        return tuple(st["shape"])
    raise ValueError(op)


def L(name, op, inputs, **st):
    layers.append({"name": name, "op": op, "inputs": inputs, **st})
    shapes[name] = _shape(name, op, inputs, st)
    return name


def conv(name, inp, W, b, stride=1, padding=0):
    W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
    o, c, kh, kw = W.shape
    weights[name + ".W"] = W; weights[name + ".b"] = b
    return L(name, "conv", [inp], out=o, kernel=[kh, kw], stride=stride, padding=padding)


def relu(name, inp):
    return L(name, "relu", [inp])


def pool(name, inp, size, stride, kind):
    return L(name, kind, [inp], size=list(size), stride=list(stride))


def add(name, ins):
    return L(name, "add", ins)


def dense(name, inp, W, b):
    W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
    weights[name + ".W"] = W; weights[name + ".b"] = b
    return L(name, "dense", [inp], out=W.shape[0])


def reshape(name, inp, shape):
    return L(name, "reshape", [inp], shape=list(shape))


def zeros(*sh):
    return np.zeros(sh, np.float32)


# ----------------------------------------------------------------- kernels
def gk(k, sx, sy=None, dx=0.0, dy=0.0):
    """Gaussian window k x k, std (sx,sy), offset (dx,dy), normalised to sum 1."""
    if sy is None:
        sy = sx
    r = (k - 1) / 2.0
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    g = np.exp(-((x - dx) ** 2 / (2 * sx * sx) + (y - dy) ** 2 / (2 * sy * sy)))
    return g / g.sum()


def dogk(k, ax=1.0, ay=1.0, ratio=0.5):
    """Zero-sum centre-surround DoG sized to the kernel: surround std = k/4.5
    (mild truncation), centre std = ratio * surround.  Detects blobs of radius
    ~ (0.5 .. 2) * surround_std."""
    ss = k / 4.5
    return gk(k, ss * ratio * ax, ss * ratio * ay) - gk(k, ss * ax, ss * ay)


# =================================================================
# 1. colour front end: skin chroma cone + luma + chroma magnitude
# =================================================================
Wcol = zeros(5, 3, 1, 1)
Wcol[0, 0, 0, 0] = 0.299; Wcol[0, 1, 0, 0] = 0.587; Wcol[0, 2, 0, 0] = 0.114
Wcol[1, 0, 0, 0] = 1.0;  Wcol[1, 1, 0, 0] = -1.0
Wcol[2, 1, 0, 0] = 1.0;  Wcol[2, 2, 0, 0] = -1.0
Wcol[3, 0, 0, 0] = -0.1687; Wcol[3, 1, 0, 0] = -0.3313; Wcol[3, 2, 0, 0] = 0.5
Wcol[4, 0, 0, 0] = 0.5;  Wcol[4, 1, 0, 0] = -0.4187; Wcol[4, 2, 0, 0] = -0.0813
conv("col", "image", Wcol, [0, 0, 0, 0.5, 0.5])

pen_terms = [
    ([0.08, -1.0, 0.0, 0.0, 0.0], 0.0, 12.0),    # u >= 0.08 Y
    ([-0.60, 1.0, 0.0, 0.0, 0.0], 0.0, 8.0),     # u <= 0.60 Y
    ([0.0, 0.25, -1.0, 0.0, 0.0], 0.0, 12.0),    # v >= 0.25 u
    ([0.0, -1.25, 1.0, 0.0, 0.0], 0.0, 6.0),     # v <= 1.25 u
    ([-1.0, 0.0, 0.0, 0.0, 0.0], 0.12, 6.0),     # Y >= 0.12
    ([1.0, 0.0, 0.0, 0.0, 0.0], -1.02, 12.0),    # Y <= 1.02
    ([0.0, 0.0, 0.0, 0.0, -1.0], 0.50, 10.0),    # Cr >= 0.50
    ([0.0, 0.0, 0.0, 0.0, 1.0], -0.74, 10.0),    # Cr <= 0.74
    ([0.0, 0.0, 0.0, 1.0, 0.0], -0.52, 10.0),    # Cb <= 0.52
    ([0.0, 0.0, 0.0, -1.0, 0.0], 0.26, 10.0),    # Cb >= 0.26
    ([0.0, -1.0, 0.0, 0.0, 0.0], 0.025, 20.0),   # u >= 0.025 (not grey)
    ([0.0, 0.0, -1.0, 0.0, 0.0], 0.012, 20.0),   # v >= 0.012
]
NP = len(pen_terms)
Wpen = zeros(NP + 5, 5, 1, 1); bpen = zeros(NP + 5)
for i, (c, b, g) in enumerate(pen_terms):
    for j in range(5):
        Wpen[i, j, 0, 0] = c[j]
    bpen[i] = b
Wpen[NP, 0, 0, 0] = 1.0
Wpen[NP + 1, 1, 0, 0] = 1.0
Wpen[NP + 2, 1, 0, 0] = -1.0
Wpen[NP + 3, 2, 0, 0] = 1.0
Wpen[NP + 4, 2, 0, 0] = -1.0
conv("pen", "col", Wpen, bpen)
relu("penr", "pen")

Wskl = zeros(3, NP + 5, 1, 1); bskl = zeros(3)
for i, (c, b, g) in enumerate(pen_terms):
    Wskl[0, i, 0, 0] = -g
bskl[0] = 1.0
Wskl[1, NP, 0, 0] = 1.0
for i in range(4):
    Wskl[2, NP + 1 + i, 0, 0] = 1.0
conv("skl", "penr", Wskl, bskl)
relu("SY0", "skl")                      # (3,96,96) [S, Y, chroma]

# =================================================================
# 2. pyramid  (0 skin, 1 luma, 2 chroma)
# =================================================================
pool("P1", "SY0", [2, 2], [2, 2], "avgpool")   # (3,48,48)
pool("P2", "P1", [2, 2], [2, 2], "avgpool")    # (3,24,24)
pool("P3", "P2", [2, 2], [2, 2], "avgpool")    # (3,12,12)


def sel(name, inp, ch, nch):
    W = zeros(1, nch, 1, 1); W[0, ch, 0, 0] = 1.0
    return conv(name, inp, W, [0.0])


sel("S0", "SY0", 0, 3)
sel("C0", "SY0", 2, 3)
sel("S2", "P2", 0, 3)

# =================================================================
# 3. skin-blob detectors (zero-sum centre-surround), valid conv
# =================================================================
def blob(name, inp, nch, k, gain, thr, ch=0, ax=1.0, ay=1.0):
    W = zeros(1, nch, k, k)
    W[0, ch] = gain * dogk(k, ax, ay)
    conv(name, inp, W, [-thr])
    return relu(name + "r", name)


Bl1a = blob("bl1a", "P1", 3, 7, 2.0, 0.45)
Bl1b = blob("bl1b", "P1", 3, 11, 2.0, 0.45)
Bl2a = blob("bl2a", "P2", 3, 9, 2.0, 0.45)
Bl2b = blob("bl2b", "P2", 3, 13, 2.0, 0.45)
Bl3 = blob("bl3", "P3", 3, 9, 2.0, 0.45)
Vblob = blob("vbl", "P2", 3, 13, 2.0, 0.40, ax=0.55, ay=1.6)
Hblob = blob("hbl", "P2", 3, 13, 2.0, 0.40, ax=1.6, ay=0.55)

# =================================================================
# 4. head = skin blob + darker hair above   ([S, Y])
# =================================================================
def head(name, inp, nch, k, d, gain_s, gain_y, thr):
    ss = k / 4.5; sc = 0.5 * ss
    W = zeros(1, nch, k, k)
    W[0, 0] = gain_s * dogk(k)
    W[0, 1] = gain_y * (gk(k, sc) - gk(k, sc, dy=-d))
    conv(name, inp, W, [-thr])
    return relu(name + "r", name)


Hd1 = head("hd1", "P1", 3, 9, 2.5, 1.6, 0.9, 0.45)
Hd2 = head("hd2", "P2", 3, 11, 3.0, 1.6, 0.9, 0.50)

# =================================================================
# 5. face with two dark eyes  ([S, Y])
# =================================================================
def face(name, inp, nch, k, sf, se, de, dy, sb, thr):
    W = zeros(1, nch, k, k)
    W[0, 1] = 3.0 * (gk(k, sf) - 0.5 * gk(k, se, dx=-de, dy=-dy)
                     - 0.5 * gk(k, se, dx=de, dy=-dy) - 0.5 * gk(k, sb))
    W[0, 0] = 0.9 * gk(k, sf) - 0.9 * gk(k, sb)
    conv(name, inp, W, [-thr])
    return relu(name + "r", name)


Fc1 = face("fc1", "P1", 3, 13, 2.0, 0.8, 1.8, 0.9, 4.0, 0.40)
Fc1b = face("fc1b", "P1", 3, 15, 2.6, 1.0, 2.4, 1.2, 5.0, 0.45)


def fullface(name, inp, nch, k, sf, se, de, dy, dh, sb, thr):
    """skin blob + darker hair above + two dark eyes inside: a whole face."""
    W = zeros(1, nch, k, k)
    W[0, 0] = 1.0 * gk(k, sf) - 1.0 * gk(k, sb)
    W[0, 1] = (1.0 * gk(k, sf) - 0.5 * gk(k, se, dx=-de, dy=-dy)
               - 0.5 * gk(k, se, dx=de, dy=-dy)
               - 0.6 * gk(k, sf * 0.8, dy=-dh) - 0.5 * gk(k, sb))
    conv(name, inp, 1.5 * W, [-thr])
    return relu(name + "r", name)


Fh1 = fullface("fh1", "P1", 3, 15, 2.2, 0.8, 1.8, 0.9, 4.0, 4.5, 0.70)
Fh2 = fullface("fh2", "P2", 3, 13, 1.8, 0.7, 1.5, 0.8, 3.2, 3.8, 0.70)

# =================================================================
# 6. upright figure: head blob above torso blob (luma, both polarities)
# =================================================================
def upright(tag, inp, nch, kdog, ktmp, shc, stc, d, thr):
    # signed DoG: head (round) and torso (tall, narrow).  A solid dark figure
    # gives negative responses at head and torso centres; a light figure positive.
    W = zeros(2, nch, kdog, kdog)
    W[0, 1] = gk(kdog, shc) - gk(kdog, 2.0 * shc)                       # round head
    W[1, 1] = gk(kdog, stc * 2.5, stc * 0.45) - gk(kdog, stc * 5.0, stc * 0.9)  # wide flat shoulders
    conv(tag + "dg", inp, W, [0, 0])
    Kh = gk(ktmp, 0.9)
    Kt = gk(ktmp, 2.2, 0.6, dy=d)
    Ka = gk(ktmp, 0.9, dy=-d)
    Kl = gk(ktmp, 1.0, 0.6, dx=-3.5, dy=d)
    Kr = gk(ktmp, 1.0, 0.6, dx=3.5, dy=d)
    hd = -Kh + 0.6 * Ka
    td = -Kt + 0.5 * Kl + 0.5 * Kr
    Wt = zeros(2, 2, ktmp, ktmp)
    Wt[0, 0] = hd; Wt[0, 1] = td          # dark figure on light ground
    Wt[1, 0] = -hd; Wt[1, 1] = -td        # light figure on dark ground
    conv(tag + "up", tag + "dg", 2.0 * Wt, [-thr, -thr])
    relu(tag + "upr", tag + "up")
    Ws = zeros(1, 2, 1, 1); Ws[0, 0, 0, 0] = 1; Ws[0, 1, 0, 0] = 1
    conv(tag + "sum", tag + "upr", Ws, [0.0])
    return tag + "sum"


Up1 = upright("u1", "P1", 3, 11, 9, 0.7, 1.1, 2.0, 0.35)
Up2 = upright("u2", "P2", 3, 13, 9, 0.9, 1.4, 3.0, 0.35)

# =================================================================
# 7. two skin blobs stacked vertically (face above arm / leg), zero-sum
# =================================================================
Wsp = zeros(1, 3, 17, 17)
Wsp[0, 0] = 1.6 * (gk(17, 1.6) + gk(17, 1.6, dy=7.0) - 2.0 * gk(17, 6.0))
conv("spair", "P1", Wsp, [-0.45])
Spair = relu("spairr", "spair")

# =================================================================
# 7b. global texture activity (a face must stand out from clutter)
# =================================================================
Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0
Sy = Sx.T
Wg = zeros(4, 3, 3, 3)
Wg[0, 1] = Sx; Wg[1, 1] = -Sx; Wg[2, 1] = Sy; Wg[3, 1] = -Sy
conv("grad", "P1", Wg, [0, 0, 0, 0])
relu("gradr", "grad")
Wgs = zeros(1, 4, 1, 1)
for i in range(4):
    Wgs[0, i, 0, 0] = 1.0
conv("gradm", "gradr", Wgs, [0.0])           # gradient magnitude map

# =================================================================
# 7c. vertical mirror symmetry of the skin map (faces are symmetric)
# =================================================================
Wsd = zeros(2, 3, 7, 7)
Wsd[0, 0, 3, 0] = 1.0; Wsd[0, 0, 3, 6] = -1.0     # S(x-3) - S(x+3)
Wsd[1, 0, 3, 0] = -1.0; Wsd[1, 0, 3, 6] = 1.0
conv("symd", "P1", Wsd, [0, 0])
relu("symdr", "symd")
Wss = zeros(1, 2, 1, 1); Wss[0, 0, 0, 0] = 1; Wss[0, 1, 0, 0] = 1
conv("syma", "symdr", Wss, [0.0])                 # |left-right skin difference|
conv("symc", "syma", np.array([[[[-1.0]]]], np.float32), [0.35])
relu("symr", "symc")                              # high where skin is mirror-symmetric
pool("symm", "symr", [7, 7], [1, 1], "avgpool")   # smooth over a face-sized window

# =================================================================
# 7d. tall foreground blob on luma (upright body; rejects horizontal animals)
# =================================================================
def vbody(tag, inp, nch, k, sx, sy):
    W = zeros(2, nch, k, k)
    K = gk(k, sx, sy) - gk(k, 2 * sx, 2 * sy)
    W[0, 1] = K; W[1, 1] = -K
    conv(tag + "c", inp, W, [0, 0])
    relu(tag + "r", tag + "c")
    Ws = zeros(1, 2, 1, 1); Ws[0, 0, 0, 0] = 1; Ws[0, 1, 0, 0] = 1
    conv(tag + "s", tag + "r", Ws, [0.0])
    return tag + "s"


Vb1 = vbody("vb1", "P1", 3, 11, 1.0, 2.6)
Vb2 = vbody("vb2", "P2", 3, 13, 1.3, 3.2)

# =================================================================
# 8. scalar features -> slots of one vector
# =================================================================
def gmax(name, inp):
    s = shapes[inp]
    pool(name + "m", inp, [s[1], s[2]], [1, 1], "maxpool")
    return reshape(name + "s", name + "m", [1])


def gmean(name, inp):
    s = shapes[inp]
    pool(name + "m", inp, [s[1], s[2]], [1, 1], "avgpool")
    return reshape(name + "s", name + "m", [1])


def dens(name, inp, t):
    W = zeros(1, shapes[inp][0], 1, 1); W[0, 0, 0, 0] = 1.0
    conv(name + "c", inp, W, [-t])
    relu(name + "r", name + "c")
    return gmean(name, name + "r")


pool("sgrid", "S0", [16, 16], [16, 16], "avgpool")   # (1,6,6)


def region(name, cells):
    W = zeros(1, 36); b = zeros(1)
    for (r, c) in cells:
        W[0, r * 6 + c] = 1.0 / len(cells)
    return dense(name, "sgrid", W, b)


F = 55
feat_maps = []


def addfeat(nm, j):
    feat_maps.append((nm, j))


maxlist = [("bl1a", Bl1a), ("bl1b", Bl1b), ("bl2a", Bl2a), ("bl2b", Bl2b),
           ("bl3", Bl3), ("hd1", Hd1), ("hd2", Hd2), ("fc1", Fc1),
           ("fc1b", Fc1b), ("fh1", Fh1), ("fh2", Fh2), ("up1", Up1),
           ("up2", Up2), ("vbl", Vblob), ("hbl", Hblob), ("spair", Spair)]
for j, (nm, mp) in enumerate(maxlist):
    addfeat(gmax("mx_" + nm, mp), j)                     # 0..15

denslist = [("bl1a", Bl1a, 0.25), ("bl1b", Bl1b, 0.25), ("bl2a", Bl2a, 0.25),
            ("bl2b", Bl2b, 0.25), ("bl3", Bl3, 0.25), ("hd1", Hd1, 0.25),
            ("hd2", Hd2, 0.25), ("fc1", Fc1, 0.25), ("fc1b", Fc1b, 0.25),
            ("fh1", Fh1, 0.25), ("fh2", Fh2, 0.25), ("up1", Up1, 0.20),
            ("up2", Up2, 0.20), ("vbl", Vblob, 0.25), ("hbl", Hblob, 0.25),
            ("spair", Spair, 0.25)]
for j, (nm, mp, t) in enumerate(denslist, start=16):
    addfeat(dens("dn_" + nm, mp, t), j)                  # 16..31

addfeat(gmean("sk_frac", "S0"), 32)
addfeat(region("sk_center", [(r, c) for r in range(1, 5) for c in range(1, 5)]), 33)
addfeat(region("sk_top", [(r, c) for r in range(0, 3) for c in range(6)]), 34)
addfeat(region("sk_bot", [(r, c) for r in range(3, 6) for c in range(6)]), 35)
Wu = zeros(1, 1, 1, 1); Wu[0, 0, 0, 0] = 1.0 / 0.15
conv("skun_c", "S2", Wu, [-0.85 / 0.15]); relu("skun_r", "skun_c")
addfeat(gmean("sk_uniform", "skun_r"), 36)
conv("skmd_c", "S2", Wu, [-0.45 / 0.15]); relu("skmd_r", "skmd_c")
addfeat(gmean("sk_mid", "skmd_r"), 37)
addfeat(gmax("chr_max", "C0"), 38)
addfeat(gmean("chr_mean", "C0"), 39)
addfeat(gmax("sk_any", "S0"), 40)
pool("bl1a_g", Bl1a, [14, 14], [14, 14], "maxpool")   # (1,3,3)
Wc = zeros(1, 1, 3, 3); Wc[0, 0, 1, 1] = 1.0
conv("bl1a_ctr", "bl1a_g", Wc, [0.0])
addfeat(reshape("bl1a_ctrs", "bl1a_ctr", [1]), 41)
addfeat(region("sk_border", [(r, c) for r in range(6) for c in range(6)
                               if r in (0, 5) or c in (0, 5)]), 42)
addfeat(gmean("activity", "gradm"), 43)
addfeat(gmax("mx_sym", "symm"), 44)
addfeat(dens("dn_sym", "symm", 0.15), 45)
addfeat(gmax("mx_vb1", Vb1), 46)
addfeat(gmax("mx_vb2", Vb2), 47)
addfeat(dens("dn_vb1", Vb1, 0.20), 48)
addfeat(dens("dn_vb2", Vb2, 0.20), 49)
# global means of the face/head maps: a real face is an isolated strong peak,
# clutter produces many comparable peaks (max - mean separates them)
addfeat(gmean("mn_fh1", Fh1), 50)
addfeat(gmean("mn_fh2", Fh2), 51)
addfeat(gmean("mn_fc1", Fc1), 52)
addfeat(gmean("mn_hd1", Hd1), 53)
addfeat(gmean("mn_hd2", Hd2), 54)

slot_names = []
for nm, j in feat_maps:
    W = zeros(F, 1); W[j, 0] = 1.0
    slot_names.append(dense("slot%d" % j, nm, W, zeros(F)))
add("feats", slot_names)

# =================================================================
# 9. hand-written decision rules.  Conservative: with no strong, specific
#    person structure every unit is 0 and the tie goes to "no person".
# =================================================================
H = 10
Wh = zeros(H, F); bh = zeros(H)


def rule(i, terms, bias):
    for j, w in terms:
        Wh[i, j] = w
    bh[i] = bias


rule(0, [(9, 2.5), (10, 2.5), (25, 3.0), (26, 3.0), (50, -1.5), (51, -1.5)], -0.50)  # whole face
rule(1, [(7, 2.5), (8, 2.0), (23, 2.5), (52, -1.5)], -0.55)            # eyes+skin
rule(2, [(5, 2.5), (6, 2.5), (21, 2.0), (22, 2.0), (53, -1.2), (54, -1.2)], -0.55)  # head+hair
rule(3, [(25, 4.0), (26, 4.0), (21, 3.0), (22, 3.0), (23, 3.0)], -0.30)  # several people
rule(4, [(11, 2.0), (12, 2.0), (27, 2.0), (28, 2.0)], -0.85)           # upright figure
rule(5, [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 0.8),
         (16, 1.5), (17, 1.5), (18, 1.5), (19, 1.5), (41, 0.5)], -1.20)  # skin blob
rule(6, [(36, 3.0), (32, 1.5), (42, 1.0), (9, -1.2), (10, -1.2), (7, -1.2)], -0.85)  # uniform sheet, unless a face
rule(7, [(14, 2.5), (30, 3.5)], -0.60)                                 # horizontal skin (animal/food)
rule(8, [(13, 1.5), (29, 2.5), (15, 1.0), (31, 2.0), (33, 0.8)], -0.95)  # vertical skin structure
rule(9, [(46, 2.0), (47, 2.0), (48, 2.5), (49, 2.5)], -0.95)           # tall foreground figure

dense("hidden", "feats", Wh, bh)
relu("hiddenr", "hidden")

Wout = zeros(2, H); bout = zeros(2)
Wout[1] = [1.5, 1.2, 0.7, 0.8, 0.5, 0.2, -1.3, -1.0, 0.4, 0.5]
Wout[0] = 0.0
dense("score", "hiddenr", Wout, bout)

# =================================================================
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": layers}, f)
np.savez(os.path.join(OUT, "weights.npz"), **weights)
print("layers:", len(layers), "weights:", sum(v.size for v in weights.values()))
