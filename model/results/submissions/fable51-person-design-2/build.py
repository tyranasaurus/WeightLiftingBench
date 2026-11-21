"""build.py -- writes network.json and weights.npz for a hand-designed person / no-person network.

Every number below is chosen from general knowledge of skin colour, faces, clothing, the proportions of a
standing person and of photographs; nothing is read from any file.  Usage: python3 build.py OUT_DIR
"""
import json, sys
import numpy as np

# ----------------------------------------------------------------------------------------------
# helpers for assembling the layer list and the weight dictionary
# ----------------------------------------------------------------------------------------------
layers, weights = [], {}


def add_layer(name, op, inputs, **settings):
    L = {"name": name, "op": op, "inputs": list(inputs)}
    L.update(settings)
    layers.append(L)
    return name


def conv(name, inp, W, b, stride=1, padding=0):
    W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
    weights[name + ".W"], weights[name + ".b"] = W, b
    return add_layer(name, "conv", [inp], out=int(W.shape[0]), kernel=[int(W.shape[2]), int(W.shape[3])],
                     stride=stride, padding=padding)


def conv1x1(name, inp, rows):
    """rows: list of (coefficient vector over input channels, bias)."""
    W = np.zeros((len(rows), len(rows[0][0]), 1, 1), np.float32)
    b = np.zeros(len(rows), np.float32)
    for i, (coef, bias) in enumerate(rows):
        W[i, :, 0, 0] = coef; b[i] = bias
    return conv(name, inp, W, b)


def dense(name, inp, W, b):
    W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
    weights[name + ".W"], weights[name + ".b"] = W, b
    return add_layer(name, "dense", [inp], out=int(W.shape[0]))


def relu(name, inp):
    return add_layer(name, "relu", [inp])


def vec(nin, terms):
    c = np.zeros(nin, np.float32)
    for i, v in terms.items():
        c[i] += v
    return c


def ramp(nin, terms, t, w):
    """Two 1x1 rows (u, u-1) with u = (sum_i terms[i]*x_i - t) / w.  After relu, row0 - row1 equals
    clip((q - t)/w, 0, 1): a soft indicator rising from 0 at q = t to 1 at q = t + w."""
    c = vec(nin, terms)
    return [(c / w, -t / w), (c / w, -t / w - 1.0)]


def combine(nin, pairs, extra=None, bias=0.0):
    """Row summing clipped indicators given as (row of u, row of u-1) pairs, plus extra linear terms."""
    c = np.zeros(nin, np.float32)
    for a, b in pairs:
        c[a] += 1.0; c[b] -= 1.0
    for i, v in (extra or {}).items():
        c[i] += v
    return (c, bias)


def absdiff_rows(nin, i, j):
    """Two rows whose relu outputs sum to |x_i - x_j|."""
    return [(vec(nin, {i: 1.0, j: -1.0}), 0.0), (vec(nin, {i: -1.0, j: 1.0}), 0.0)]


# ----------------------------------------------------------------------------------------------
# Stage 1: per-pixel colour analysis (1x1 convolutions + relu).  R, G, B in [0, 1].
# ----------------------------------------------------------------------------------------------
# Photographed skin is a reddish-orange of moderate saturation: hue roughly 5..30 degrees, saturation
# roughly 0.15..0.58, not too dark.  Each bound is a linear inequality in RGB with a soft ramp; the skin
# map is the soft AND of all of them.  (hue = 60 * (G-B)/(R-B) when R >= G >= B.)
#   c1  R - G                  >= 0.04 .. 0.08    clearly redder than green (excludes grey / beige)
#   c2  -0.083R + G - 0.917B   >= -0.03 .. 0.01   hue >= ~5 deg  (excludes pink / magenta / brick red)
#   c3  0.5R - G + 0.5B        >= -0.03 .. 0.01   hue <= ~30 deg (excludes sand, pine, bread, dirt)
#   c4  0.85R - B              >= -0.02 .. 0.02   saturation >= ~0.15 (excludes white / grey / pale walls)
#   c5  B - 0.42R              >= -0.04 .. 0.00   saturation <= ~0.58 (excludes oranges, dark wood, fur)
#   c6  R                      >= 0.20 .. 0.30    bright enough for hue to be meaningful
# Luminance Y = 0.299R + 0.587G + 0.114B; dark = clip((0.40 - Y)/0.20): 1 for Y <= 0.20, 0 for Y >= 0.40.
# chroma = |R-G| + |G-B| is used to recognise black-and-white photographs; warm = relu(R - B) to
# recognise sepia / toned photographs (warm tint everywhere but no vivid colour).
Ylum = {0: 0.299, 1: 0.587, 2: 0.114}
rows = []
rows += ramp(3, {0: 1.0, 1: -1.0}, 0.04, 0.04)                 # c1   -> 0,1
rows += ramp(3, {0: -0.083, 1: 1.0, 2: -0.917}, -0.03, 0.04)   # c2   -> 2,3
rows += ramp(3, {0: 0.5, 1: -1.0, 2: 0.5}, -0.03, 0.04)        # c3   -> 4,5
rows += ramp(3, {0: 0.85, 2: -1.0}, -0.02, 0.04)               # c4   -> 6,7
rows += ramp(3, {0: -0.42, 2: 1.0}, -0.04, 0.04)               # c5   -> 8,9
rows += ramp(3, {0: 1.0}, 0.20, 0.10)                          # c6   -> 10,11
rows += ramp(3, {k: -v for k, v in Ylum.items()}, -0.40, 0.20)  # dark -> 12,13  u = (0.40 - Y)/0.20
rows += absdiff_rows(3, 0, 1) + absdiff_rows(3, 1, 2)          # chroma parts -> 14..17
rows += [(vec(3, {0: 1.0, 2: -1.0}), 0.0)]                      # warm = relu(R - B) -> 18
g1 = conv1x1("g1", "image", rows)
r1 = relu("r1", g1)
n1 = len(rows)
g2 = conv1x1("g2", r1, [
    combine(n1, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)], bias=-5.0),   # skin  (soft AND of six)
    combine(n1, [(12, 13)]),                                                      # dark
    combine(n1, [], {14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0}),                        # chroma
    combine(n1, [], {18: 1.0}),                                                   # warm tint: relu(R - B)
])
r2 = relu("r2", g2)                      # (4, 96, 96): skin, dark, chroma, warm

# ----------------------------------------------------------------------------------------------
# Stage 2: pyramids.  (skin, dark) and (R, G, B) averaged into cells of 2..12 pixels.
# ----------------------------------------------------------------------------------------------
p3 = conv1x1("p3", r2, [(vec(4, {0: 1}), 0.0), (vec(4, {1: 1}), 0.0)])          # skin, dark
pyr = {}
for pool in (2, 3, 4, 5, 6, 8, 12):
    L = 96 // pool
    pyr[L] = add_layer(f"pyr{L}", "avgpool", [p3], size=[pool, pool], stride=[pool, pool])
    pyr[(L, "rgb")] = add_layer(f"pyrgb{L}", "avgpool", ["image"], size=[pool, pool], stride=[pool, pool])

NFEAT = 24   # 0-5 face at 6 scales, 6-11 isolated skin blob at 6 scales, 12-16 standing figure at 5 scales,
             # 17 mean skin, 18 max chroma, 19 mean warm tint
K = 9


def box(r0, r1, c0, c1):
    m = np.zeros((K, K), np.float32); m[r0:r1, c0:c1] = 1
    return m / m.sum()


def template(nout, channel_rows):
    W = np.zeros((nout, 1, K, K), np.float32)
    for i, m in channel_rows.items():
        W[i, 0] = m
    return W


def embed(slots, nin):
    W = np.zeros((NFEAT, nin), np.float32)
    for i, s in enumerate(slots):
        W[s, i] = 1.0
    return W, np.zeros(NFEAT, np.float32)


def region_means(tag, L, nout, skin_rows, dark_rows, rgb_rows, padding=K // 2):
    """Box means of pyramid maps at level L.  skin_rows / dark_rows: {out channel: box};
    rgb_rows: {out channel: box} applied to each of R, G, B with channel offsets 0, 1, 2."""
    parts = []
    if skin_rows:
        sel = conv1x1(f"{tag}_sel_skin", pyr[L], [(vec(2, {0: 1}), 0.0)])
        parts.append(conv(f"{tag}_tp_skin", sel, template(nout, skin_rows), np.zeros(nout, np.float32), padding=padding))
    if dark_rows:
        sel = conv1x1(f"{tag}_sel_dark", pyr[L], [(vec(2, {1: 1}), 0.0)])
        parts.append(conv(f"{tag}_tp_dark", sel, template(nout, dark_rows), np.zeros(nout, np.float32), padding=padding))
    for c in range(3):
        sel = conv1x1(f"{tag}_sel_rgb{c}", pyr[(L, "rgb")], [(vec(3, {c: 1}), 0.0)])
        parts.append(conv(f"{tag}_tp_rgb{c}", sel, template(nout, {i + c: m for i, m in rgb_rows.items()}),
                          np.zeros(nout, np.float32), padding=padding))
    return add_layer(f"{tag}_t1", "add", parts)


branch_outputs = []

# ----------------------------------------------------------------------------------------------
# Stage 3a: face templates.  In pyramid cells, a face is a 3x3 box of skin; above it a 3x2 cap that is
# not skin and is either dark (hair) or at least a different colour from the background beside the
# head (hat, light hair against a darker scene); left and right of it 2-cell flanks that are not skin;
# one cell below the chin a 3x3 "body" box that is not skin (clothing) and whose colour differs from the
# background beside the head.  Applied at every position of 6 pyramid levels: faces of about 6, 9, 12,
# 18, 24 and 36 pixels.  Grid (9x9, centre (4,4)): cap rows 0-1 cols 3-5; face rows 2-4 cols 3-5;
# flanks rows 2-4 cols 1-2 and 6-7; neck row 5; body rows 6-8 cols 3-5.
# ----------------------------------------------------------------------------------------------
FACE, CAP, BODY = box(2, 5, 3, 6), box(0, 2, 3, 6), box(6, 9, 3, 6)
FLANK = (box(2, 5, 1, 3) + box(2, 5, 6, 8)) / 2
# template channels: 0 A(skin face) 1 F(skin flank) 2 U(skin cap) 3 B(skin body) 4 D(dark cap)
#                    5-7 RGB body, 8-10 RGB flank, 11-13 RGB cap
NT = 14
for s_idx, L in enumerate([48, 32, 24, 16, 12, 8]):
    tag = f"f{s_idx}"
    t1 = region_means(tag, L, NT, {0: FACE, 1: FLANK, 2: CAP, 3: BODY}, {4: CAP}, {5: BODY, 8: FLANK, 11: CAP})
    r = []
    r += ramp(NT, {0: 1.0}, 0.40, 0.30)            # iA   face box mostly skin            -> 0,1
    r += ramp(NT, {1: -1.0}, -0.35, 0.25)          # iF   flanks not skin                 -> 2,3
    r += ramp(NT, {2: -1.0}, -0.45, 0.30)          # iU   cap not skin                    -> 4,5
    r += ramp(NT, {3: -1.0}, -0.35, 0.25)          # iB   body box not skin (clothing)    -> 6,7
    r += ramp(NT, {4: 1.0}, 0.15, 0.30)            # iD   cap dark (hair)                 -> 8,9
    for c in range(3):
        r += absdiff_rows(NT, 5 + c, 8 + c)        # |body - flank| per colour            -> 10..15
    for c in range(3):
        r += absdiff_rows(NT, 11 + c, 8 + c)       # |cap - flank| per colour             -> 16..21
    t2 = relu(f"{tag}_t2r", conv1x1(f"{tag}_t2", t1, r))
    t3 = relu(f"{tag}_t3r", conv1x1(f"{tag}_t3", t2, [
        combine(22, [(0, 1), (2, 3), (4, 5), (6, 7)]),                 # 0: s4 = iA + iF + iU + iB
        combine(22, [(8, 9)]),                                         # 1: iD
    ] + ramp(22, {i: 1.0 for i in range(10, 16)}, 0.12, 0.20)          # 2,3: body contrast  clip((sum|d| - 0.12)/0.2)
      + ramp(22, {i: 1.0 for i in range(16, 22)}, 0.30, 0.30)))        # 4,5: cap contrast   clip((sum|d| - 0.30)/0.30)
    t4 = relu(f"{tag}_t4r", conv1x1(f"{tag}_t4", t3, [
        (vec(6, {0: 1, 2: 1, 3: -1}), -4.0),         # 0: iso  = relu(s4 + iBodyC - 4)
        (vec(6, {0: 1, 2: 1, 3: -1}), 0.0),          # 1: s4 + iBodyC
        (vec(6, {1: 1, 4: 1, 5: -1}), 0.0),          # 2: iD + iCapC
        (vec(6, {1: 1, 4: 1, 5: -1}), -1.0),         # 3: iD + iCapC - 1
    ]))
    t5 = relu(f"{tag}_t5r", conv1x1(f"{tag}_t5", t4, [
        (vec(4, {0: 1}), 0.0),                       # iso
        (vec(4, {1: 1, 2: 1, 3: -1}), -5.0),         # face = relu(s4 + iBodyC + clip(iD + iCapC) - 5)
    ]))
    mx = add_layer(f"{tag}_max", "maxpool", [t5], size=[L, L], stride=[L, L])
    rs = add_layer(f"{tag}_rs", "reshape", [mx], shape=[2])
    Wd, bd = embed([6 + s_idx, s_idx], 2)
    branch_outputs.append(dense(f"{tag}_emb", rs, Wd, bd))

# ----------------------------------------------------------------------------------------------
# Stage 3b: standing-figure templates (no skin needed).  A person seen whole is about 8 cells tall and
# 3 wide: a 1-cell-wide head (rows 1-2, col 4) over a 3-cell-wide torso (rows 3-5) and legs (rows 6-8).
# Conditions, all as colour differences against the background flanks (cols 1-2 and 6-7) at the same
# rows: the row above the head is uniform (nothing continues upward: not a pole, tree or tower); the head
# differs from the background while the cells beside the head (cols 3 and 5) do not (narrow head on wide
# shoulders); torso and legs differ from the background; the background itself is uniform (left and right
# sides alike), as on a field, court, slope, road or wall.  Levels 32..12 cover
# people 24..64 pixels tall.  No padding here: zero padding would fake colour contrast at the image border.
# ----------------------------------------------------------------------------------------------
ABOVE, ABOVE_FL = box(0, 1, 3, 6), (box(0, 1, 1, 3) + box(0, 1, 6, 8)) / 2
HEAD, HEAD_SIDES, HEAD_FL = box(1, 3, 4, 5), (box(1, 3, 3, 4) + box(1, 3, 5, 6)) / 2, (box(1, 3, 1, 3) + box(1, 3, 6, 8)) / 2
TORSO, TORSO_FL = box(3, 6, 3, 6), (box(3, 6, 1, 3) + box(3, 6, 6, 8)) / 2
LEGS, LEGS_FL = box(6, 9, 3, 6), (box(6, 9, 1, 3) + box(6, 9, 6, 8)) / 2
BODY_L, BODY_R = box(3, 9, 1, 3), box(3, 9, 6, 8)        # left / right background beside torso and legs
regions = [ABOVE, ABOVE_FL, HEAD, HEAD_SIDES, HEAD_FL, TORSO, TORSO_FL, LEGS, LEGS_FL, BODY_L, BODY_R]
NG = 3 * len(regions)       # channel 3*i + c = colour c of region i
# pairs of regions whose colour difference is measured: pair p occupies t2 channels 6p..6p+5
pairs = [(0, 1), (2, 4), (3, 4), (5, 6), (7, 8), (9, 10)]   # ..., left/right background alike
for g_idx, L in enumerate([32, 24, 19, 16, 12]):
    tag = f"g{g_idx}"
    t1 = region_means(tag, L, NG, {}, {}, {3 * i: m for i, m in enumerate(regions)}, padding=0)
    r = []
    for (i, j) in pairs:
        for c in range(3):
            r += absdiff_rows(NG, 3 * i + c, 3 * j + c)
    t2 = relu(f"{tag}_t2r", conv1x1(f"{tag}_t2", t1, r))
    NP = 6 * len(pairs)
    r3 = []
    r3 += ramp(NP, {i: -1.0 for i in range(0, 6)}, -0.20, 0.12)     # above uniform:  clip((0.20 - d)/0.12)
    r3 += ramp(NP, {i: 1.0 for i in range(6, 12)}, 0.20, 0.20)      # head differs:   clip((d - 0.20)/0.20)
    r3 += ramp(NP, {i: -1.0 for i in range(12, 18)}, -0.50, 0.20)   # head sides uniform: clip((0.50 - d)/0.20)
    r3 += ramp(NP, {i: 1.0 for i in range(18, 24)}, 0.20, 0.20)     # torso differs
    r3 += ramp(NP, {i: 1.0 for i in range(24, 30)}, 0.15, 0.20)     # legs differ
    r3 += ramp(NP, {i: -1.0 for i in range(30, 36)}, -0.25, 0.15)   # left and right background alike
    t3 = relu(f"{tag}_t3r", conv1x1(f"{tag}_t3", t2, r3))
    t4 = relu(f"{tag}_t4r", conv1x1(f"{tag}_t4", t3, [
        combine(12, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)], bias=-5.0),   # figure = relu(sum of 6 - 5)
    ]))
    Lo = L - K + 1
    mx = add_layer(f"{tag}_max", "maxpool", [t4], size=[Lo, Lo], stride=[Lo, Lo])
    rs = add_layer(f"{tag}_rs", "reshape", [mx], shape=[1])
    Wd, bd = embed([12 + g_idx], 1)
    branch_outputs.append(dense(f"{tag}_emb", rs, Wd, bd))

# ----------------------------------------------------------------------------------------------
# Stage 3c: global cues.  Mean skin fraction and mean warm tint; maximum chroma (zero only for
# black-and-white photos, small for toned ones).
# ----------------------------------------------------------------------------------------------
gsel = conv1x1("gsel", r2, [(vec(4, {0: 1}), 0.0), (vec(4, {3: 1}), 0.0)])
gm = add_layer("gmean", "avgpool", [gsel], size=[96, 96], stride=[96, 96])
gmr = add_layer("gmean_rs", "reshape", [gm], shape=[2])
Wd, bd = embed([17, 19], 2)
branch_outputs.append(dense("gmean_emb", gmr, Wd, bd))
chs = conv1x1("chsel", r2, [(vec(4, {2: 1}), 0.0)])
chm = add_layer("chmax", "maxpool", [chs], size=[96, 96], stride=[96, 96])
chr_ = add_layer("chmax_rs", "reshape", [chm], shape=[1])
Wd, bd = embed([18], 1)
branch_outputs.append(dense("chmax_emb", chr_, Wd, bd))

feat = add_layer("feat", "add", branch_outputs)

# ----------------------------------------------------------------------------------------------
# Stage 4: readout.  Hidden units are clipped indicators; the output is a hand-weighted sum.
# ----------------------------------------------------------------------------------------------
face_w = np.array([0.5, 0.9, 1.0, 1.0, 1.0, 1.0], np.float32)    # the smallest faces cannot fire alone
iso_w = np.array([0.5, 0.9, 1.0, 1.0, 1.0, 1.0], np.float32)
fig_w = np.array([0.6, 1.0, 1.0, 1.0, 1.0], np.float32)          # nor the smallest figures
H = []
c = np.zeros(NFEAT, np.float32); c[0:6] = face_w; H += [(c, 0.0), (c, -1.0)]        # 0,1 faceE = clip(sum)
c = np.zeros(NFEAT, np.float32); c[6:12] = iso_w;  H += [(c, 0.0), (c, -1.0)]       # 2,3 isoE  = clip(sum)
c = np.zeros(NFEAT, np.float32); c[12:17] = fig_w; H += [(c, 0.0), (c, -1.0)]       # 4,5 figE  = clip(sum)
H += ramp(NFEAT, {17: 1.0}, 0.25, 0.20)        # 6,7  areaBig = clip((mean skin - 0.25)/0.20)
H += ramp(NFEAT, {18: -1.0}, -0.05, 0.03)      # 8,9  grey = clip((0.05 - max chroma)/0.03): black-and-white photo
H += ramp(NFEAT, {18: -1.0}, -0.40, 0.10)      # 10,11 lowchroma = clip((0.40 - max chroma)/0.10): no vivid colour anywhere
H += ramp(NFEAT, {19: 1.0}, 0.08, 0.08)        # 12,13 warm = clip((mean warm - 0.08)/0.08): warm tint over the image
Wh = np.stack([h[0] for h in H]); bh = np.array([h[1] for h in H], np.float32)
hidr = relu("hidr", dense("hid", feat, Wh, bh))
# second hidden stage: sepia = relu(lowchroma + warm - 1) (a warm-tinted photo without vivid colour: an old
# or toned photograph, almost always of people); bigpen = relu(areaBig - lowchroma) (large skin-like area in
# a colourful image: wood, sand, orange light, food -- but not a toned photograph)
nh = len(H)
def ind(k):
    return {2 * k: 1.0, 2 * k + 1: -1.0}
Wk = []
Wk.append((vec(nh, ind(0)), 0.0))                                   # faceE
Wk.append((vec(nh, ind(1)), 0.0))                                   # isoE
Wk.append((vec(nh, ind(2)), 0.0))                                   # figE
Wk.append((vec(nh, ind(4)), 0.0))                                   # grey
Wk.append((vec(nh, {**ind(5), **{2 * 6: 1.0, 2 * 6 + 1: -1.0}}), -1.0))   # sepia = relu(lowchroma + warm - 1)
Wk.append((vec(nh, {**ind(3), **{2 * 5: -1.0, 2 * 5 + 1: 1.0}}), 0.0))   # bigpen = relu(areaBig - lowchroma)
Wk2 = np.stack([w_[0] for w_ in Wk]); bk2 = np.array([w_[1] for w_ in Wk], np.float32)
hid2 = relu("hid2r", dense("hid2", hidr, Wk2, bk2))

W_FACE, W_ISO, W_FIG, W_GREY, W_SEPIA, W_BIG, THRESH = 1.0, 0.3, 0.85, 0.9, 0.9, 0.6, 0.75
Wo = np.zeros((2, 6), np.float32); bo = np.zeros(2, np.float32)
Wo[1] = [W_FACE, W_ISO, W_FIG, W_GREY, W_SEPIA, -W_BIG]
bo[0] = THRESH
out = dense("out", hid2, Wo, bo)

# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as fh:
        json.dump({"layers": layers}, fh, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in weights.items()})
    print(f"wrote {len(layers)} layers, {sum(int(v.size) for v in weights.values())} weights to {out_dir}")
