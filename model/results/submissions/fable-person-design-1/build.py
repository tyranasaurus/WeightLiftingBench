"""Hand-designed person / no-person network for 96x96 photos.

Every number below is typed from general knowledge about how people look in photos
(skin colour as a cone in RGB, a face as a compact skin blob that stands clear of other warm things,
darker hair above it, a clothed body below it that differs from the background beside the head).
Nothing is read from any file and nothing is fitted:  python3 build.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)

layers = []
weights = {}


def L(name, op, inputs, **kw):
    layers.append(dict(name=name, op=op, inputs=list(inputs), **kw))
    return name


def conv(name, inp, W, b, stride=1, pad=0):
    W = np.asarray(W, np.float32)
    b = np.asarray(b, np.float32)
    weights[name + ".W"] = W
    weights[name + ".b"] = b
    return L(name, "conv", [inp], out=int(W.shape[0]), kernel=[int(W.shape[2]), int(W.shape[3])], stride=int(stride), padding=int(pad))


def dense(name, inp, W, b):
    W = np.asarray(W, np.float32)
    b = np.asarray(b, np.float32)
    weights[name + ".W"] = W
    weights[name + ".b"] = b
    return L(name, "dense", [inp], out=int(W.shape[0]))


def relu(name, inp):
    return L(name, "relu", [inp])


def avgpool(name, inp, s):
    return L(name, "avgpool", [inp], size=[s, s], stride=[s, s])


def maxpool(name, inp, size, stride):
    return L(name, "maxpool", [inp], size=list(size), stride=list(stride))


def add(name, inps):
    return L(name, "add", inps)


def reshape(name, inp, shape):
    return L(name, "reshape", [inp], shape=list(shape))


def norm(name, inp):
    return L(name, "norm", [inp])


def pw(rows, bias):
    """1x1 conv weights from a list of rows."""
    W = np.asarray(rows, np.float64)
    return W[:, :, None, None], np.asarray(bias, np.float64)


# ----------------------------------------------------------------------------------------------
# 1. Colour: skin as a cone in RGB.
#    With u = R - B and w = G - B (R > G > B for skin), hue = 60 deg * w / u and saturation = u / R.
#    Skin of every complexion sits near hue 5..28 deg, saturation 0.16..0.66; wood, bread, sand, dry grass
#    are yellower (hue 30..50 deg).  Each limit is a half-space; a violation is relu(linear form).
# ----------------------------------------------------------------------------------------------
R = np.array([1.0, 0.0, 0.0])
G = np.array([0.0, 1.0, 0.0])
B = np.array([0.0, 0.0, 1.0])
U = R - B
Wc = G - B
LUM = 0.30 * R + 0.59 * G + 0.11 * B

SKIN_HUE_LO, SKIN_HUE_HI = 0.08, 0.47     # w/u limits: hue 5 deg .. 28 deg
SKIN_SAT_LO, SKIN_SAT_HI = 0.16, 0.66
SKIN_VAL_LO = 0.28
WARM_HUE_LO, WARM_HUE_HI = 0.05, 0.88     # earthy warm colours: hue 3 deg .. 53 deg (pure red is clothing, not wood or fur)
WARM_SAT_LO = 0.12
WARM_VAL_LO = 0.20

lin_rows = [
    (SKIN_HUE_LO * U - Wc, 0.0),          # 0 skin: too red / pink
    (Wc - SKIN_HUE_HI * U, 0.0),          # 1 skin: too yellow
    (SKIN_SAT_LO * R - U, 0.0),           # 2 skin: too grey
    (U - SKIN_SAT_HI * R, 0.0),           # 3 skin: too saturated
    (-R, SKIN_VAL_LO),                    # 4 skin: too dark
    (WARM_HUE_LO * U - Wc, 0.0),          # 5 warm: too red / magenta
    (Wc - WARM_HUE_HI * U, 0.0),          # 6 warm: too yellow-green
    (WARM_SAT_LO * R - U, 0.0),           # 7 warm: too grey
    (-R, WARM_VAL_LO),                    # 8 warm: too dark
    (LUM, 0.0),                           # 9 luminance (never negative, so relu passes it through)
]
SKIN_SLOPE = [22.0, 22.0, 20.0, 14.0, 10.0]   # membership falls from 1 to 0 over about 0.05 of violation
WARM_SLOPE = [20.0, 20.0, 20.0, 10.0]

conv("lin", "image", *pw([r for r, _ in lin_rows], [b for _, b in lin_rows]))
relu("viol", "lin")

row_skin = np.zeros(10); row_skin[0:5] = -np.array(SKIN_SLOPE)
row_warm = np.zeros(10); row_warm[5:9] = -np.array(WARM_SLOPE)
row_inv = np.zeros(10); row_inv[9] = -1.0
z10 = np.zeros(10)

# fine feature: skin membership (zero padding = "not skin")
conv("fine_lin", "viol", *pw([row_skin], [1.0]))
relu("fine", "fine_lin")

# context features: [warm, not warm, skin]
#   warm + notwarm = 1 inside the photo and 0 in the zero padding, so their sum tells how much of a region is inside.
conv("ctx_lin", "viol", *pw([row_warm, z10, row_skin], [1.0, 0.0, 1.0]))
relu("ctx_a", "ctx_lin")                   # [warm, 0, skin], each in [0, 1]
conv("ctx_b", "ctx_a", *pw([[0, 0, 0], [-1.0, 0, 0], [0, 0, 0]], [0.0, 1.0, 0.0]))    # [0, 1 - warm, 0]
add("ctx", ["ctx_a", "ctx_b"])

# inverse luminance (zero padding = "bright", so nothing outside the photo counts as dark hair)
conv("inv_lin", "viol", *pw([row_inv], [1.0]))
relu("inv", "inv_lin")

# colour features for "does this region look different from that one":
#   [normalised luminance, red-green, blue-yellow, 1]   (the constant 1 is 0 in the zero padding: it marks the photo)
conv("lum1", "image", *pw([LUM], [0.0]))
norm("lumn", "lum1")                       # zero mean, unit variance over the photo (zero padding = average brightness)
conv("lum4", "lumn", *pw([[1.0], [0.0], [0.0], [0.0]], [0.0] * 4))
CHROMA_GAIN = 4.0                          # a colour difference of 0.25 counts like one std of luminance
conv("chroma4", "image", *pw([np.zeros(3), CHROMA_GAIN * (R - G), CHROMA_GAIN * (B - 0.5 * R - 0.5 * G), np.zeros(3)], [0.0, 0.0, 0.0, 1.0]))
add("col", ["lum4", "chroma4"])


# ----------------------------------------------------------------------------------------------
# 2. The head model, in units of the face radius r (x to the right, y downwards, origin = face centre).
# ----------------------------------------------------------------------------------------------
def ell(x, y, a, b):
    return np.sqrt((x / a) ** 2 + (y / b) ** 2)


def ang_from_up(x, y):                     # 0 = straight up, 180 = straight down, sign = side
    return np.rad2deg(np.arctan2(x, -y))


def m_centre(x, y):                        # the face: an upright ellipse
    return ell(x, y, 0.80, 0.95) <= 1.0


def m_hair(x, y):                          # cap over the top of the face, within 65 degrees of straight up
    rho = ell(x, y, 0.85, 1.05)
    return (rho >= 0.95) & (rho <= 1.60) & (np.abs(ang_from_up(x, y)) <= 65.0)


def ring(x, y, lo=1.65, hi=2.60):          # starts well outside the face so a face half a cell off-grid stays clear
    rho = ell(x, y, 0.85, 1.05)
    return (rho >= lo) & (rho <= hi)


def m_top(x, y):                           # surround above the head
    return ring(x, y) & (np.abs(ang_from_up(x, y)) <= 45.0)


def m_left(x, y):                          # surround beside the head
    a = ang_from_up(x, y)
    return ring(x, y) & (a < -45.0) & (a >= -125.0)


def m_right(x, y):
    a = ang_from_up(x, y)
    return ring(x, y) & (a > 45.0) & (a <= 125.0)


def m_under(x, y):                         # under the chin, below the neck: collar and chest
    return ring(x, y, 1.70, 2.90) & (np.abs(ang_from_up(x, y)) >= 150.0)


def m_body(x, y):                          # shoulders and chest under the head
    return (np.abs(x) <= 1.4) & (y >= 1.7) & (y <= 3.6)


def m_flank(x, y):                         # background beside the head, above the shoulders
    return (np.abs(x) >= 1.9) & (np.abs(x) <= 3.3) & (y >= -1.3) & (y <= 0.9)


def render(mask, r_cells, half, ss=8):
    """Mean-filter weights for a mask: fraction of each cell covered, normalised to sum to 1."""
    k = 2 * half + 1
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    c = (np.arange(-half, half + 1)[:, None] + offs[None, :]).ravel() / r_cells
    Y, X = np.meshgrid(c, c, indexing="ij")
    m = mask(X, Y).astype(np.float64).reshape(k, ss, k, ss).mean(axis=(1, 3))
    return m / m.sum()


# ----------------------------------------------------------------------------------------------
# 3. Head detectors at face radii from 2 to 25 pixels, in four groups.
#    Each group reads the skin map pooled to cells of c pixels (face ellipse, then the best of each 2x2 block)
#    and the context / colour / luminance maps pooled to cells of 2c pixels.
#    evidence = skin in the face ellipse
#               - warm colour in the surround (left, right, top: a head stands clear of other warm things)
#               - skin colour continuing under the chin (animals, toys, posts, furniture; people wear clothes)
#               + bonus for a dark cap of hair
#               + bonus for a body under the head that differs from the background beside the head
#               - TAU0
# ----------------------------------------------------------------------------------------------
GROUPS = [(1, [2.0, 2.5, 3.15, 4.0]),      # (fine cell size in pixels, face radii in fine cells)
          (2, [2.5, 3.15, 4.0]),
          (4, [2.5, 3.15, 4.0]),
          (8, [2.5, 3.15])]
HALF_F, HALF_C = 4, 6                      # kernel half sizes (9x9 fine, 13x13 context)

HAIR_RATIO = 0.60                          # hair counts when the cap is darker than 60% of the face brightness
HAIR_GAIN = 10.0
A_HAIR = 0.20                              # bonus for a dark cap
A_BODY = 0.20                              # bonus for a body under the head
BODY_T0, BODY_T1 = 0.35, 0.95              # body-vs-flank difference where the bonus starts / is full
BODY_INVALID = 2.0                         # no body bonus when the body or flank regions fall outside the photo
PAD_WARM = 0.5                             # surround outside the photo counts as half warm
SIDE_TOL, TOP_TOL, UNDER_TOL = 0.15, 0.15, 0.25   # how much of each part of the surround may be warm / skin
SECTOR_SLOPE = 2.0
TAU0 = 0.55                                # what a bare skin blob must beat

NCH = 13                                   # context responses per template
group_out = []
for c, radii in GROUPS:
    T = len(radii)
    f_in = "fine" if c == 1 else avgpool(f"fine_p{c}", "fine", c)
    x_in = avgpool(f"ctx_p{2 * c}", "ctx", 2 * c)
    c_in = avgpool(f"col_p{2 * c}", "col", 2 * c)
    i_in = avgpool(f"inv_p{2 * c}", "inv", 2 * c)

    # fine conv: mean skin in the face ellipse, then the best position of each 2x2 block
    kf = 2 * HALF_F + 1
    Wf = np.zeros((T, 1, kf, kf))
    for t, rr in enumerate(radii):
        Wf[t, 0] = render(m_centre, rr, HALF_F)
    conv(f"hf{c}", f_in, Wf, np.zeros(T), pad=HALF_F)
    maxpool(f"hf{c}m", f"hf{c}", [2, 2], [2, 2])

    # context responses per template:
    #   0..2   warm excess of the left / right / top surround
    #   3      skin excess under the chin (padding counts as not skin)
    #   4      how much of the body and flank regions lies outside the photo
    #   5..10  +-(body - flank) for luminance, red-green, blue-yellow
    #   11,12  hair, hair - 1
    kc = 2 * HALF_C + 1
    Wx = np.zeros((NCH * T, 3, kc, kc)); bx = np.zeros(NCH * T)
    Wc2 = np.zeros((NCH * T, 4, kc, kc))
    Wi = np.zeros((NCH * T, 1, kc, kc)); bi = np.zeros(NCH * T)
    for t, rr in enumerate(radii):
        rc = rr / 2.0
        o = NCH * t
        for s, (m, tol) in enumerate(((m_left, SIDE_TOL), (m_right, SIDE_TOL), (m_top, TOP_TOL))):
            w = render(m, rc, HALF_C)
            # fill = warm + PAD_WARM * outside,  outside = 1 - warm - notwarm
            Wx[o + s, 0] = (1.0 - PAD_WARM) * w; Wx[o + s, 1] = -PAD_WARM * w; bx[o + s] = PAD_WARM - tol
        Wx[o + 3, 2] = render(m_under, rc, HALF_C); bx[o + 3] = -UNDER_TOL
        body = render(m_body, rc, HALF_C)
        flank = render(m_flank, rc, HALF_C)
        Wx[o + 4, 0] = -(body + flank); Wx[o + 4, 1] = -(body + flank); bx[o + 4] = 2.0
        for j in range(3):
            Wc2[o + 5 + 2 * j, j] = body - flank
            Wc2[o + 6 + 2 * j, j] = flank - body
        # hair = gain * (ratio * lum_face - lum_cap), written with inverse luminance (1 - lum)
        hw = HAIR_GAIN * (render(m_hair, rc, HALF_C) - HAIR_RATIO * render(m_centre, rc, HALF_C))
        Wi[o + 11, 0] = hw; bi[o + 11] = -HAIR_GAIN * (1.0 - HAIR_RATIO)
        Wi[o + 12, 0] = hw; bi[o + 12] = -HAIR_GAIN * (1.0 - HAIR_RATIO) - 1.0
    conv(f"hx{c}", x_in, Wx, bx, pad=HALF_C)
    conv(f"hc{c}", c_in, Wc2, np.zeros(NCH * T), pad=HALF_C)
    conv(f"hi{c}", i_in, Wi, bi, pad=HALF_C)
    add(f"hg{c}", [f"hx{c}", f"hc{c}", f"hi{c}"])
    relu(f"hg{c}r", f"hg{c}")
    # per template: [penalty, hair in 0..1, body ramp, body ramp - 1]
    Wb = np.zeros((4 * T, NCH * T)); bb = np.zeros(4 * T)
    for t in range(T):
        o = NCH * t
        Wb[4 * t, o:o + 4] = SECTOR_SLOPE
        Wb[4 * t + 1, o + 11] = 1.0; Wb[4 * t + 1, o + 12] = -1.0
        for q in (2, 3):
            Wb[4 * t + q, o + 5:o + 11] = 1.0 / (BODY_T1 - BODY_T0)
            Wb[4 * t + q, o + 4] = -BODY_INVALID
            bb[4 * t + q] = -BODY_T0 / (BODY_T1 - BODY_T0) - (q - 2)
    conv(f"hb{c}", f"hg{c}r", *pw(Wb, bb))
    relu(f"hb{c}r", f"hb{c}")
    Wq = np.zeros((T, 4 * T))
    for t in range(T):
        Wq[t, 4 * t] = -1.0; Wq[t, 4 * t + 1] = A_HAIR; Wq[t, 4 * t + 2] = A_BODY; Wq[t, 4 * t + 3] = -A_BODY
    conv(f"hq{c}", f"hb{c}r", *pw(Wq, -TAU0 * np.ones(T)))
    add(f"hz{c}", [f"hf{c}m", f"hq{c}"])
    relu(f"hz{c}r", f"hz{c}")
    n = 96 // (2 * c)
    reshape(f"hz{c}f", f"hz{c}r", [1, T * n, n])
    maxpool(f"hmax{c}", f"hz{c}f", [T * n, n], [T * n, n])          # best head of this group: (1, 1, 1)
    group_out.append(f"hmax{c}")

# best head over all groups
NG = len(GROUPS)
emb = []
for i, name in enumerate(group_out):
    Wd = np.zeros((NG, 1)); Wd[i, 0] = 1.0
    emb.append(dense(f"emb{i}", name, Wd, np.zeros(NG)))
add("heads", emb)
reshape("heads_r", "heads", [1, NG, 1])
maxpool("best", "heads_r", [NG, 1], [NG, 1])

# ----------------------------------------------------------------------------------------------
# 4. Standing figure, for people whose skin does not show (far away, back view, winter clothes, black and white).
#    A figure is a column of head / torso / legs.  In units of the torso width w (origin: top centre of the head):
#    the torso and the legs both differ from the background on their left and right, the column has a second tone
#    (legs or head whose colour is not a mixture of torso and background; a plain post has one tone), the background
#    is the same on both sides at every level, and the column ends: above the head is background again.
# ----------------------------------------------------------------------------------------------
def box(x0, x1, y0, y1):
    return lambda x, y: (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


FIG_YC = 1.45                               # the kernel is centred here (in w) so that the whole figure fits
P_HEAD = box(-0.35, 0.35, 0.05, 0.75)
P_TORSO = box(-0.50, 0.50, 0.90, 2.20)
P_LEGS = box(-0.45, 0.45, 2.40, 3.80)
P_ABOVE = box(-0.50, 0.50, -0.90, -0.20)
FLANK_X0, FLANK_X1 = 0.95, 1.75
LEVELS = [(-0.30, 0.80), (0.90, 2.20), (2.40, 3.80)]         # head, torso and legs levels of the flanks
P_FL = [box(-FLANK_X1, -FLANK_X0, y0, y1) for y0, y1 in LEVELS]
P_FR = [box(FLANK_X0, FLANK_X1, y0, y1) for y0, y1 in LEVELS]


def render_fig(mask, w_cells, half, ss=8):
    k = 2 * half + 1
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    c = (np.arange(-half, half + 1)[:, None] + offs[None, :]).ravel() / w_cells
    Y, X = np.meshgrid(c + FIG_YC, c, indexing="ij")
    m = mask(X, Y).astype(np.float64).reshape(k, ss, k, ss).mean(axis=(1, 3))
    return m / m.sum()


FIG_CELLS = [3, 4, 6, 8]                    # cell sizes in pixels
FIG_W = [1.7, 2.0, 2.6]                     # torso widths in cells: 5 .. 21 pixels, figures 19 .. 80 pixels tall
FIG_HALF = 6
LEG_GAIN = 1.4                              # legs fill their box less than a torso does
TT_GAIN = 1.5                               # two-tone evidence against the torso / legs contrasts
SKIN_TT = 5.0                               # skin-coloured legs or head over a torso that is not: bare legs, a face
SKIN_TORSO = 4.0                            # ... and the torso must really be free of skin colour (not half a wooden post)
FIG_CAP = 2.5
FIG_K_SIDES, FIG_K_TOP, FIG_K_ABOVE = 0.5, 0.25, 0.5     # penalties for a background that is not uniform
FIG_K_INVALID = 6.0                         # figures cut by the frame do not count
FIG_TAU = 1.2                               # in units of one std of luminance (or 0.25 of colour difference)
ND = 9                                      # signed differences per template
NF = 6 * ND + 3
FT = len(FIG_W)
fig_out = []
for c in FIG_CELLS:
    src = f"col_p{c}" if any(l["name"] == f"col_p{c}" for l in layers) else avgpool(f"col_p{c}", "col", c)
    ssrc = f"fine_p{c}" if any(l["name"] == f"fine_p{c}" for l in layers) else avgpool(f"fine_p{c}", "fine", c)
    kf = 2 * FIG_HALF + 1
    Wf = np.zeros((NF * FT, 4, kf, kf)); bf = np.zeros(NF * FT)
    Ws = np.zeros((NF * FT, 1, kf, kf))
    for t, wc in enumerate(FIG_W):
        rn = lambda m: render_fig(m, wc, FIG_HALF)
        head, torso, legs, above = rn(P_HEAD), rn(P_TORSO), rn(P_LEGS), rn(P_ABOVE)
        fl = [rn(m) for m in P_FL]; fr = [rn(m) for m in P_FR]
        ft, fm, fb = (0.5 * (fl[i] + fr[i]) for i in range(3))
        diffs = [torso - fm,                             # 0: torso against its flanks
                 legs - fb,                              # 1: legs against their flanks
                 torso - fb,                             # 2: torso against the legs' flanks
                 head - ft,                              # 3: head against its flanks
                 torso - ft,                             # 4: torso against the head's flanks
                 fl[0] - fr[0], fl[1] - fr[1], fl[2] - fr[2],   # 5, 6, 7: left against right background
                 above - ft]                             # 8: above the head against the background
        o = NF * t
        for d, w in enumerate(diffs):
            for j in range(3):
                Wf[o + 6 * d + 2 * j, j] = w             # channel order per difference: +lum, -lum, +rg, -rg, +by, -by
                Wf[o + 6 * d + 2 * j + 1, j] = -w
        allr = (head + torso + legs + above + sum(fl) + sum(fr)) / 10.0
        Wf[o + 6 * ND, 3] = -allr; bf[o + 6 * ND] = 1.0  # fraction of the template outside the photo
        Ws[o + 6 * ND + 1, 0] = legs - SKIN_TORSO * torso     # skin on the legs, none on the torso
        Ws[o + 6 * ND + 2, 0] = head - SKIN_TORSO * torso     # skin on the head, none on the torso
    conv(f"fgc{c}", src, Wf, bf, pad=FIG_HALF)
    conv(f"fgs{c}", ssrc, Ws, np.zeros(NF * FT), pad=FIG_HALF)
    add(f"fg{c}", [f"fgc{c}", f"fgs{c}"])
    relu(f"fg{c}r", f"fg{c}")
    # A second tone.  With g = legs - background and t = torso - background (per colour channel, per sign):
    #   LB = sum relu(g - t) = "legs go beyond the torso",  TB = sum relu(t - g) = "torso goes beyond the legs".
    # When one box only holds a mixture of the other colour and the background (the end of a plain post, a box
    # that is half off the object) only one of LB, TB is non-zero.  Two real tones give both: min(LB, TB) > 0.
    # The same for head against torso.
    # layer 1: [A, A - B, P, 6 x LB, 6 x TB, 6 x LB(head), 6 x TB(head), skin second tone]
    N1 = 28
    W1 = np.zeros((N1 * FT, NF * FT))
    for t in range(FT):
        o = NF * t
        sl = lambda d: slice(o + 6 * d, o + 6 * d + 6)
        W1[N1 * t, sl(0)] = 1.0
        W1[N1 * t + 1, sl(0)] = 1.0; W1[N1 * t + 1, sl(1)] = -LEG_GAIN
        W1[N1 * t + 2, sl(5)] = FIG_K_TOP
        W1[N1 * t + 2, sl(6)] = FIG_K_SIDES; W1[N1 * t + 2, sl(7)] = FIG_K_SIDES
        W1[N1 * t + 2, sl(8)] = FIG_K_ABOVE
        W1[N1 * t + 2, o + 6 * ND] = FIG_K_INVALID
        for q in range(6):
            W1[N1 * t + 3 + q, o + 6 * 1 + q] = 1.0; W1[N1 * t + 3 + q, o + 6 * 2 + q] = -1.0      # LB: legs beyond torso
            W1[N1 * t + 9 + q, o + 6 * 2 + q] = 1.0; W1[N1 * t + 9 + q, o + 6 * 1 + q] = -1.0      # TB: torso beyond legs
            W1[N1 * t + 15 + q, o + 6 * 3 + q] = 1.0; W1[N1 * t + 15 + q, o + 6 * 4 + q] = -1.0    # head beyond torso
            W1[N1 * t + 21 + q, o + 6 * 4 + q] = 1.0; W1[N1 * t + 21 + q, o + 6 * 3 + q] = -1.0    # torso beyond head
        W1[N1 * t + 27, o + 6 * ND + 1] = SKIN_TT; W1[N1 * t + 27, o + 6 * ND + 2] = SKIN_TT
    conv(f"fa{c}", f"fg{c}r", *pw(W1, np.zeros(N1 * FT)))
    relu(f"fa{c}r", f"fa{c}")
    # layer 2: [m1 = min(A, B), LB, LB - TB, LBh, LBh - TBh, P, skin second tone]
    N2 = 7
    W2 = np.zeros((N2 * FT, N1 * FT))
    for t in range(FT):
        o = N1 * t
        W2[N2 * t, o] = 1.0; W2[N2 * t, o + 1] = -1.0
        W2[N2 * t + 1, o + 3:o + 9] = 1.0
        W2[N2 * t + 2, o + 3:o + 9] = 1.0; W2[N2 * t + 2, o + 9:o + 15] = -1.0
        W2[N2 * t + 3, o + 15:o + 21] = 1.0
        W2[N2 * t + 4, o + 15:o + 21] = 1.0; W2[N2 * t + 4, o + 21:o + 27] = -1.0
        W2[N2 * t + 5, o + 2] = 1.0
        W2[N2 * t + 6, o + 27] = 1.0
    conv(f"fb{c}", f"fa{c}r", *pw(W2, np.zeros(N2 * FT)))
    relu(f"fb{c}r", f"fb{c}")
    # layer 3: TT = gain * (min(LB, TB) + min(LBh, TBh)) + skin second tone;  [m1, m1 - TT, P]
    W3 = np.zeros((3 * FT, N2 * FT))
    for t in range(FT):
        o = N2 * t
        tt = np.zeros(N2 * FT); tt[o + 1] = 1.0; tt[o + 2] = -1.0; tt[o + 3] = 1.0; tt[o + 4] = -1.0
        W3[3 * t, o] = 1.0
        W3[3 * t + 1, o] = 1.0; W3[3 * t + 1] -= TT_GAIN * tt; W3[3 * t + 1, o + 6] -= 1.0
        W3[3 * t + 2, o + 5] = 1.0
    conv(f"fc{c}", f"fb{c}r", *pw(W3, np.zeros(3 * FT)))
    relu(f"fc{c}r", f"fc{c}")
    # layer 4: m2 = min(m1, TT) = m1 - relu(m1 - TT);  [m2, m2 - cap, P]
    W4 = np.zeros((3 * FT, 3 * FT)); b4 = np.zeros(3 * FT)
    for t in range(FT):
        W4[3 * t, 3 * t] = 1.0; W4[3 * t, 3 * t + 1] = -1.0
        W4[3 * t + 1, 3 * t] = 1.0; W4[3 * t + 1, 3 * t + 1] = -1.0; b4[3 * t + 1] = -FIG_CAP
        W4[3 * t + 2, 3 * t + 2] = 1.0
    conv(f"fe{c}", f"fc{c}r", *pw(W4, b4))
    relu(f"fe{c}r", f"fe{c}")
    # layer 5: figure evidence = min(m2, cap) - P - tau
    W5 = np.zeros((FT, 3 * FT))
    for t in range(FT):
        W5[t, 3 * t] = 1.0; W5[t, 3 * t + 1] = -1.0; W5[t, 3 * t + 2] = -1.0
    conv(f"fd{c}", f"fe{c}r", *pw(W5, -FIG_TAU * np.ones(FT)))
    relu(f"fd{c}r", f"fd{c}")
    n = 96 // c
    reshape(f"fd{c}f", f"fd{c}r", [1, FT * n, n])
    maxpool(f"fmax{c}", f"fd{c}f", [FT * n, n], [FT * n, n])
    fig_out.append(f"fmax{c}")

NFG = len(fig_out)
emb = []
for i, name in enumerate(fig_out):
    Wd = np.zeros((NFG, 1)); Wd[i, 0] = 1.0
    emb.append(dense(f"femb{i}", name, Wd, np.zeros(NFG)))
add("figs", emb)
reshape("figs_r", "figs", [1, NFG, 1])
maxpool("fbest", "figs_r", [NFG, 1], [NFG, 1])

# ----------------------------------------------------------------------------------------------
# 5. Decision.  A photo that is mostly warm-coloured (wood, food, sand, sepia) needs a stronger head.
#    person  <=>  a head that clears its threshold, or a standing figure.
# ----------------------------------------------------------------------------------------------
MODE = "final"                              # "final", or "diag_fig" = figure fires while the head detector is silent
avgpool("glob", "ctx", 96)                  # (3, 1, 1): [warm fraction, 1 - warm fraction, skin fraction]
WARM_F0, WARM_LAMBDA = 0.35, 0.60
THETA = 0.05
SCORE_GAIN = 10.0
# hidden units: [best head, warm excess, best figure]
dense("h_best", "best", [[1.0], [0.0], [0.0]], [0.0, 0.0, 0.0])
dense("h_glob", "glob", [[0.0] * 3, [1.0, 0.0, 0.0], [0.0] * 3], [0.0, -WARM_F0, 0.0])
dense("h_fig", "fbest", [[0.0], [0.0], [1.0]], [0.0, 0.0, 0.0])
add("h_sum", ["h_best", "h_glob", "h_fig"])
relu("h", "h_sum")
dense("ev_lin", "h", [[1.0, -WARM_LAMBDA, 0.0], [0.0, 0.0, 1.0]], [-THETA, 0.0])
relu("ev", "ev_lin")                        # [head evidence, figure evidence], both >= 0
if MODE == "final":
    dense("scores", "ev", [[0.0, 0.0], [SCORE_GAIN, SCORE_GAIN]], [0.0, -0.001])
else:
    dense("scores", "ev", [[0.0, 0.0], [-1000.0, SCORE_GAIN]], [0.0, -0.001])

# ----------------------------------------------------------------------------------------------
(OUT / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
np.savez(OUT / "weights.npz", **weights)
print("layers", len(layers), "weights", sum(int(v.size) for v in weights.values()))
