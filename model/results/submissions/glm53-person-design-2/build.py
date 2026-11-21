"""Builds a hand-designed person/no-person network for 96x96 COCO photos.

No training, no data: every weight comes from formulas and general knowledge
of (a) skin colour statistics (YCbCr / R-G chroma windows from the
skin-detection literature, centred on the chroma that survives shrinking a
photo to 96x96), (b) the geometry of human faces, and (c) centre-surround
contrast.

Pipeline:
  image -> Y, Cr, Cb, R-G colour features (per pixel, 1x1 conv)
       -> relu "bump" units: triangular membership functions of chroma
          (skin likelihood; |chroma| units give a neutral-grey likelihood for
          black-and-white photos)
       -> skin map s, strong-skin map ss, dark-spot map d (5x5 ring contrast),
          local skin density sA (9x9 mean; faces are isolated warm blobs),
          grey map g -- at 96, 48 and 24 px (avgpool pyramid of the colours).
  For every head width, an "eye-pair" stage computes, at each position,
  d(left eye) + d(right eye), the asymmetry |dL - dR|, and d(nose bridge):
  two dark eyes with a BRIGHT nose bridge between them is the human pattern;
  a dark band or a single dark blob (wood grain, flower centres) is not.
  Face/head templates at 12 head widths (4..60 px at 96-scale), fixed kernels
  over [s, d, g, sA, ss, sumd, asym, dmid]:
  +skin and +strong-skin on the face oval (forehead discounted for hair),
  +eye-pair evidence, +mouth, +dark hair band above, -skin on side margins,
  -density on the surroundings (a face is an isolated warm blob).
  Global max-pool of every template map + co-localised scale groups +
  strong-skin block cues -> dense (2,) readout with hand-set weights.
"""
import json
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# hand-chosen constants (from general knowledge, not fitted to data)
# ----------------------------------------------------------------------------
# YCbCr of a pixel in [0,1]:  Cr = 0.713*(R-Y), Cb = 0.564*(B-Y).
RG_LO, RG_MID, RG_HI = 0.050, 0.135, 0.220
CR_LO, CR_MID, CR_HI = 0.030, 0.095, 0.160
CB_LO, CB_MID, CB_HI = -0.180, -0.095, -0.010
A_RG, A_CR, A_CB = 0.40, 0.35, 0.25          # weights of the three memberships
SS_RG, SS_CR, SS_T = 1.00, 0.90, 1.00        # strong skin: membership sum > 1

G_ABS_W = 3.3                                # grey (B&W photo) likelihood
G_BRIGHT_T, G_BRIGHT_W = 0.82, 6.0           # kill blown-out whites
G_DARK_T, G_DARK_W = 0.12, 2.0               # kill near-black

DK, DK_MARGIN = 4.0, 0.08                    # dark-spot: ring mean - centre

# face template weights (tiered by head width Wf: small <8, mid <12, large >=12)
W_SKIN = 0.62                                 # skin cell weight on the face oval
W_SS = 0.30                                   # strong-skin bonus on the oval
W_FORE = 0.75                                 # forehead cells count this much
W_EYE = {0: 0.17, 1: 0.35, 2: 0.55}           # eye evidence weight
W_MOUTH = {0: 0.05, 1: 0.08, 2: 0.12}
W_HAIR = {0: 0.05, 1: 0.08, 2: 0.08}          # dark hair band above the face
W_MARGIN = {0: 0.15, 1: 0.30, 2: 0.30}        # penalty for skin beside the face
W_ISO = 0.35                                  # penalty for warm surroundings
W_GREY = 0.30                                 # grey-channel weight on skin cells
# eye-pair channel weights at the face-centre cell (tier >= 1 only)
P_SUM, P_ASYM, P_MID = 0.0, 0.0, 0.45
BIAS = {0: 0.28, 1: 0.38, 2: 0.42}            # template threshold
EYE_PATCH_FRAC = 1.0                          # eye weight on the dark patches


def tier(Wf):
    return 0 if Wf < 8 else (1 if Wf < 12 else 2)


# final readout (person logit), one weight per cue slot
PERSON_W = [0.50, 0.60, 0.70,              # max face resp, head widths 4, 5.5, 7
            0.70, 0.80, 0.80,              # 10, 14, 18
            0.80, 0.80,                    # 22, 28
            0.75, 0.75, 0.75, 0.70,        # 31, 38, 48, 60
            0.70,                          # max over all 12 face scales
            0.90, 0.90, 0.90, 0.90,        # co-localised scale groups A-D
            0.50, 0.60,                    # strong-skin block cues (4x4, 8x8)
            0.15,                          # mean skin fraction
            2.00,                          # strong-skin fraction
            -1.00,                         # excess skin fraction (>0.25) penalty
            -0.80]                         # colourful-object discount (chroma)
PERSON_B = -0.55

CHROMA_XT = 0.22     # colourful-photo threshold (food/flowers vs people)

SKIN_T = 0.25          # threshold for the strong-skin block cues
SKINFRAC_XT = 0.25     # "too much skin" threshold
COLOC_THR = {"A": 0.45, "B": 0.35, "C": 0.40, "D": 0.40}

NF = 9  # feature channels [s, d, g, sA, ss, sumd, asymP, asymM, dmid]


# ----------------------------------------------------------------------------
def conv_l(name, inp, out, k, stride=1, pad=0):
    return {"name": name, "op": "conv", "inputs": [inp], "out": out,
            "kernel": [k, k], "stride": stride, "padding": pad}


def relu_l(name, inp):
    return {"name": name, "op": "relu", "inputs": [inp]}


def pool_l(name, inp, op, size, stride):
    return {"name": name, "op": op, "inputs": [inp],
            "size": list(size), "stride": list(stride)}


def add_l(name, ins):
    return {"name": name, "op": "add", "inputs": list(ins)}


# ---- stage weights ---------------------------------------------------------
def ycc_weights():
    """image(R,G,B) -> [Y, Cr, Cb, R-G]"""
    W = np.zeros((4, 3, 1, 1))
    W[0, 0], W[0, 1], W[0, 2] = 0.299, 0.587, 0.114          # Y
    W[1, 0], W[1, 1], W[1, 2] = 0.500, -0.4186, -0.0813      # Cr = 0.713*(R-Y)
    W[2, 0], W[2, 1], W[2, 2] = -0.1687, -0.3313, 0.500      # Cb = 0.564*(B-Y)
    W[3, 0], W[3, 1] = 1.0, -1.0                             # R-G
    return W, np.zeros(4)


def bump_weights():
    """[Y,Cr,Cb,RG] -> 17 threshold units (pre-relu).
    0-2: Cr - lo/mid/hi; 3-5: Cb - lo/mid/hi; 6-8: RG - lo/mid/hi;
    9-14: |Cr|,|Cb|,|RG| parts; 15: Y-0.82; 16: 0.12-Y."""
    W = np.zeros((17, 4, 1, 1))
    b = np.zeros(17)
    for r, (ch, t) in enumerate([(1, CR_LO), (1, CR_MID), (1, CR_HI),
                                 (2, CB_LO), (2, CB_MID), (2, CB_HI),
                                 (3, RG_LO), (3, RG_MID), (3, RG_HI)]):
        W[r, ch] = 1.0
        b[r] = -t
    for r, ch, s in [(9, 1, 1), (10, 1, -1), (11, 2, 1), (12, 2, -1),
                     (13, 3, 1), (14, 3, -1)]:
        W[r, ch] = s
    W[15, 0] = 1.0
    b[15] = -G_BRIGHT_T
    W[16, 0] = -1.0
    b[16] = G_DARK_T
    return W, b


def skin_rows():
    """bumpr -> two conv rows over the 17 units: skin s and strong skin ss."""
    w_s = np.zeros(17)
    n_rg = (RG_HI - RG_LO) / 2.0
    n_cr = (CR_HI - CR_LO) / 2.0
    n_cb = (CB_HI - CB_LO) / 2.0
    w_s[0], w_s[1], w_s[2] = A_CR / n_cr, -2 * A_CR / n_cr, A_CR / n_cr
    w_s[3], w_s[4], w_s[5] = A_CB / n_cb, -2 * A_CB / n_cb, A_CB / n_cb
    w_s[6], w_s[7], w_s[8] = A_RG / n_rg, -2 * A_RG / n_rg, A_RG / n_rg
    w_ss = np.zeros(17)
    w_ss[0], w_ss[1], w_ss[2] = SS_CR / n_cr, -2 * SS_CR / n_cr, SS_CR / n_cr
    w_ss[6], w_ss[7], w_ss[8] = SS_RG / n_rg, -2 * SS_RG / n_rg, SS_RG / n_rg
    return w_s, (w_ss, -SS_T)


def grey_row():
    """bumpr -> grey-likelihood raw (one conv row)."""
    w = np.zeros(17)
    for r in range(9, 15):
        w[r] = -G_ABS_W
    w[15] = -G_BRIGHT_W
    w[16] = -G_DARK_W
    return w, 1.0


def dark_kernel(K):
    """Y -> d = DK*(ring_mean(Y) - Y - DK_MARGIN); ring = KxK minus centre."""
    Wk = np.zeros((1, 1, K, K))
    n = K * K - 1
    for dy in range(K):
        for dx in range(K):
            Wk[0, 0, dy, dx] = DK / n
    Wk[0, 0, K // 2, K // 2] = -DK
    return Wk, np.array([-DK * DK_MARGIN])


def pair_kernel(Wf):
    """Eye-pair stage kernel on the dark map for head width Wf (level px).
    Output channels (into a 9-ch stack): 5: dL+dR, 6: dL-dR, 7: -(dL-dR),
    8: d(nose bridge). Eye offsets: +-0.21*Wf horizontally, 0.108*Wf up."""
    a = max(2, int(round(0.21 * Wf)))
    e = max(1, int(round(0.108 * Wf)))
    m = max(a, e)
    K = 2 * m + 1
    W = np.zeros((NF, NF, K, K))
    b = np.zeros(NF)
    dy = m - e
    W[8, 1, dy, m] = 1.0            # nose bridge darkness (between the eyes)
    return W, b, K


# ---- face template ---------------------------------------------------------
def face_kernel(K, Wf, with_g, with_pair):
    """Kernel over the 9-channel stack for a frontal head of width Wf px."""
    Hf = 1.35 * Wf
    c = (K - 1) / 2.0
    a, b = Wf / 2.0, Hf / 2.0
    ker = np.zeros((NF, K, K))      # [s, d, g, sA, ss, sumd, asymP, asymM, dmid]
    t = tier(Wf)
    we, wm, wh, wmar = W_EYE[t], W_MOUTH[t], W_HAIR[t], W_MARGIN[t]

    skin, eye, mouth, hair, margin, iso = [], [], [], [], [], []
    for y in range(K):
        for x in range(K):
            ex, ey = (x - c) / a, (y - c) / b
            if ex * ex + ey * ey <= 1.0:
                in_eye_y = (c - 0.13 * Hf) <= y <= (c - 0.03 * Hf)
                if t == 0:
                    in_eye_x = abs(x - c) <= 0.30 * Wf
                else:
                    in_eye_x = (c - 0.35 * Wf <= x <= c - 0.12 * Wf) or \
                               (c + 0.12 * Wf <= x <= c + 0.35 * Wf)
                if in_eye_y and in_eye_x:
                    eye.append((y, x))
                    continue
                if (c + 0.18 * Hf <= y <= c + 0.30 * Hf) and abs(x - c) <= 0.13 * Wf:
                    mouth.append((y, x))
                    continue
                f = W_FORE if y < c - 0.20 * Hf else 1.0
                skin.append((y, x, f))
            else:
                dil = ((x - c) / (a + 2.2)) ** 2 + ((y - c) / (b + 2.2)) ** 2
                if dil <= 1.0 and (c - 0.6 * b) <= y <= (c + b + 2.2) and abs(x - c) > 0.92 * a:
                    margin.append((y, x))
                if (c - b - 2.5) <= y < (c - b) and abs(x - c) <= 0.55 * Wf and y >= 0:
                    hair.append((y, x))
                dil3 = ((x - c) / (a + 3.0)) ** 2 + ((y - c) / (b + 3.0)) ** 2
                if dil3 > 1.0:
                    iso.append((y, x))

    n_eff = sum(f for _, _, f in skin) or 1.0
    for y, x, f in skin:
        ker[0, y, x] = W_SKIN * f / n_eff
        ker[4, y, x] = W_SS * f / n_eff
        if with_g:
            ker[2, y, x] = W_GREY * f / n_eff
    # eyes: robust patch average on d; eye-pair weights returned separately
    for y, x in eye:
        ker[1, y, x] = EYE_PATCH_FRAC * we / max(len(eye), 1)
    for y, x in mouth:
        ker[1, y, x] += wm / max(len(mouth), 1)
    for y, x in hair:
        ker[1, y, x] += wh / max(len(hair), 1)
    for y, x in margin:
        ker[0, y, x] = -wmar / max(len(margin), 1)
    for y, x in iso:
        ker[3, y, x] = -W_ISO / max(len(iso), 1)

    pairw = None
    if with_pair:
        pairw = [P_SUM * we, -P_ASYM * we, -P_ASYM * we, -P_MID * we]
    return ker, -BIAS[t], pairw


# ----------------------------------------------------------------------------
def build():
    layers = []
    W = {}

    def add_conv(name, inp, out, k, w, b, stride=1, pad=0):
        layers.append(conv_l(name, inp, out, k, stride, pad))
        W[name + ".W"] = np.asarray(w, dtype=np.float32)
        W[name + ".b"] = np.asarray(b, dtype=np.float32)

    # ---- colour stage ----
    w, b = ycc_weights()
    add_conv("ycc", "image", 4, 1, w, b)
    wye, bye = np.zeros((1, 3, 1, 1)), np.zeros(1)
    wye[0, 0], wye[0, 1], wye[0, 2] = 0.299, 0.587, 0.114
    add_conv("yonly", "image", 1, 1, wye, bye)

    wb, bb = bump_weights()
    w_s, (w_ss, b_ss) = skin_rows()
    grow, gbias = grey_row()
    dk5, dk5b = dark_kernel(5)

    # ---- per-level maps (9-channel stacks, pair channels stay zero here) ----
    def level(tag, color_in, y_in, with_g):
        add_conv("bump" + tag, color_in, 17, 1, wb, bb)
        layers.append(relu_l("bumpr" + tag, "bump" + tag))
        ws = np.zeros((NF, 17, 1, 1))
        ws[0] = w_s.reshape(1, 17, 1, 1)
        add_conv("skin" + tag, "bumpr" + tag, NF, 1, ws, np.zeros(NF))
        wss = np.zeros((NF, 17, 1, 1))
        wss[4] = np.array(w_ss).reshape(1, 17, 1, 1)
        bss = np.zeros(NF)
        bss[4] = b_ss
        add_conv("ssn" + tag, "bumpr" + tag, NF, 1, wss, bss)
        layers.append(relu_l("ssr" + tag, "ssn" + tag))          # strong skin
        parts = ["skin" + tag, "ssr" + tag]
        wd = np.zeros((NF, 1, 5, 5))
        wd[1, 0] = dk5[0, 0]
        bd = np.zeros(NF)
        bd[1] = dk5b[0]
        add_conv("dark" + tag, y_in, NF, 5, wd, bd, pad=2)
        layers.append(relu_l("darkr" + tag, "dark" + tag))
        parts.append("darkr" + tag)
        if with_g:
            wg = np.zeros((NF, 17, 1, 1))
            wg[2] = np.array(grow).reshape(1, 17, 1, 1)
            add_conv("grey" + tag, "bumpr" + tag, NF, 1, wg,
                     np.array([0.0, 0.0, gbias] + [0.0] * (NF - 3)))
            layers.append(relu_l("greyr" + tag, "grey" + tag))
            parts.append("greyr" + tag)
        layers.append(pool_l("pool9" + tag, "skin" + tag, "avgpool", [9, 9], [1, 1]))
        wsa = np.zeros((NF, NF, 1, 1))
        wsa[3, 0] = 1.0
        add_conv("sAl" + tag, "pool9" + tag, NF, 1, wsa, np.zeros(NF), pad=4)
        parts.append("sAl" + tag)
        layers.append(add_l("feat" + tag, parts))
        return "feat" + tag, "darkr" + tag

    f96, d96 = level("96", "ycc", "yonly", False)
    layers.append(pool_l("ap1", "ycc", "avgpool", [2, 2], [2, 2]))
    layers.append(pool_l("y1", "yonly", "avgpool", [2, 2], [2, 2]))
    f48, d48 = level("48", "ap1", "y1", True)
    layers.append(pool_l("ap2", "ap1", "avgpool", [2, 2], [2, 2]))
    layers.append(pool_l("y2", "y1", "avgpool", [2, 2], [2, 2]))
    f24, d24 = level("24", "ap2", "y2", True)

    # ---- face templates ----
    # (conv name, feature layer, dark layer, K, [head widths], size, with_pair)
    templates = [
        ("cA1", f96, d96, 9, [4.0], 96, False),
        ("cA2", f96, d96, 11, [5.5], 96, False),
        ("cA3", f96, d96, 13, [7.0], 96, False),
        ("cB1", f48, d48, 11, [5.0], 48, True),
        ("cB2", f48, d48, 15, [7.0, 9.0], 48, True),
        ("cC1", f48, d48, 19, [11.0], 48, True),
        ("cC2", f48, d48, 23, [14.0], 48, True),
        ("cD1", f24, d24, 15, [7.75], 24, True),
        ("cD2", f24, d24, 17, [9.5], 24, True),
        ("cD3", f24, d24, 21, [12.0], 24, True),
        ("cD4", f24, d24, 25, [15.0], 24, True),
    ]
    max_names = []
    for cname, feat, dark, K, widths, size, with_pair in templates:
        with_g = feat != f96
        C = len(widths)
        inp = feat
        if with_pair:
            # one eye-pair stage per head width; stack them into wide channels
            for wi, wf in enumerate(widths):
                Wp, bp, Kp = pair_kernel(wf)
                pname = "pair%s_%d" % (cname, wi)
                add_conv(pname, dark, NF, Kp, Wp, bp, pad=(Kp - 1) // 2)
                layers.append(relu_l("pairr%s_%d" % (cname, wi), pname))
            NCW = 5 + 4 * len(widths)
            ex_parts = []
            wfz = np.zeros((NCW, NF, 1, 1))
            for ci in range(NF):
                wfz[ci, ci] = 1.0
            add_conv("fw" + cname, feat, NCW, 1, wfz, np.zeros(NCW))
            ex_parts.append("fw" + cname)
            for wi in range(len(widths)):
                wpp = np.zeros((NCW, NF, 1, 1))
                for ch in range(5, 9):
                    wpp[5 + 4 * wi + (ch - 5), ch] = 1.0
                add_conv("fp%s_%d" % (cname, wi), "pairr%s_%d" % (cname, wi),
                         NCW, 1, wpp, np.zeros(NCW))
                ex_parts.append("fp%s_%d" % (cname, wi))
            layers.append(add_l("fs" + cname, ex_parts))
            inp = "fs" + cname
            ker = np.zeros((C, NCW, K, K))
            bias = np.zeros(C)
            for i, wf in enumerate(widths):
                k, bb, pairw = face_kernel(K, wf, with_g, True)
                ker[i, :NF] = k
                for j, pv in enumerate(pairw):
                    ker[i, 5 + 4 * i + j] = pv
                bias[i] = bb
        else:
            ker = np.zeros((C, NF, K, K))
            bias = np.zeros(C)
            for i, wf in enumerate(widths):
                ker[i], bias[i], _ = face_kernel(K, wf, with_g, False)
        add_conv(cname, inp, C, K, ker, bias, stride=1, pad=(K - 1) // 2)
        layers.append(relu_l("r" + cname, cname))
        mname = "m" + cname
        layers.append(pool_l(mname, "r" + cname, "maxpool", [size, size], [size, size]))
        max_names.append((mname, C))

    # ---- max over the 12 face scales ----
    NC = sum(C for _, C in max_names)
    fx_names = []
    ch = 0
    for mname, C in max_names:
        wex = np.zeros((NC, C, 1, 1))
        for i in range(C):
            wex[ch + i, i] = 1.0
        ex = "fx" + mname
        add_conv(ex, mname, NC, 1, wex, np.zeros(NC))
        fx_names.append(ex)
        ch += C
    layers.append(add_l("fcues", fx_names))                     # (NC,1,1)
    layers.append({"name": "fc12", "op": "reshape", "inputs": ["fcues"], "shape": [1, NC, 1]})
    layers.append(pool_l("fmax", "fc12", "maxpool", [NC, 1], [NC, 1]))  # (1,1,1)

    # ---- co-localisation cues: adjacent scales summed at the SAME position ----
    layers.append(add_l("sumA", ["rcA1", "rcA2", "rcA3"]))      # (1,96,96)
    layers.append(pool_l("mSA", "sumA", "maxpool", [96, 96], [96, 96]))
    add_conv("eB1", "rcB1", 1, 1, np.ones((1, 1, 1, 1)), np.zeros(1))
    add_conv("eB2", "rcB2", 1, 1, np.ones((1, 2, 1, 1)), np.zeros(1))
    layers.append(add_l("sumB", ["eB1", "eB2"]))                # W10+W14+W18
    layers.append(pool_l("mSB", "sumB", "maxpool", [48, 48], [48, 48]))
    layers.append(add_l("sumC", ["rcC1", "rcC2"]))              # W22+W28
    layers.append(pool_l("mSC", "sumC", "maxpool", [48, 48], [48, 48]))
    layers.append(add_l("sumD", ["rcD1", "rcD2", "rcD3", "rcD4"]))  # W31..W60
    layers.append(pool_l("mSD", "sumD", "maxpool", [24, 24], [24, 24]))
    for g in ("A", "B", "C", "D"):
        add_conv("thr" + g, "mS" + g, 1, 1, np.ones((1, 1, 1, 1)),
                 np.array([-COLOC_THR[g]]))
        layers.append(relu_l("coloc" + g, "thr" + g))

    # ---- strong-skin block cues and fractions ----
    wst = np.zeros((1, NF, 1, 1))
    wst[0, 4] = 1.0
    add_conv("skT", f96, 1, 1, wst, np.array([-SKIN_T]))
    layers.append(relu_l("skTr", "skT"))
    layers.append(pool_l("bA1", "skTr", "avgpool", [4, 4], [4, 4]))
    layers.append(pool_l("bA", "bA1", "maxpool", [24, 24], [24, 24]))
    layers.append(pool_l("bB1", "skTr", "avgpool", [8, 8], [8, 8]))
    layers.append(pool_l("bB", "bB1", "maxpool", [12, 12], [12, 12]))
    layers.append(pool_l("sf5", "skin96", "avgpool", [96, 96], [96, 96]))  # (9,1,1)
    layers.append(pool_l("sfss", "ssr96", "avgpool", [96, 96], [96, 96]))  # (9,1,1)

    wsx = np.zeros((1, NF, 1, 1))
    wsx[0, 0] = 1.0
    add_conv("sXp", "sf5", 1, 1, wsx, np.array([-SKINFRAC_XT]))
    layers.append(relu_l("sX", "sXp"))
    # colourful-photo discount: mean chroma energy |Cr| + |R-G|
    wce = np.zeros((1, 17, 1, 1))
    for u in (9, 10, 13, 14):
        wce[0, u] = 1.0
    add_conv("ceP", "bumpr96", 1, 1, wce, np.zeros(1))
    layers.append(pool_l("ceM", "ceP", "avgpool", [96, 96], [96, 96]))
    add_conv("ceXp", "ceM", 1, 1, np.ones((1, 1, 1, 1)), np.array([-CHROMA_XT]))
    layers.append(relu_l("ceX", "ceXp"))

    NCUE = len(PERSON_W)

    # ---- expand every (C,1,1) cue tensor to (NCUE,1,1) and add ----
    mapping = []
    slot = 0
    for mname, C in max_names:
        for i in range(C):
            mapping.append((mname, i, slot))
            slot += 1
    mapping.append(("fmax", 0, slot)); slot += 1
    for g in ("A", "B", "C", "D"):
        mapping.append(("coloc" + g, 0, slot)); slot += 1
    for nm in ("bA", "bB"):
        mapping.append((nm, 0, slot)); slot += 1
    mapping.append(("sf5", 0, slot)); slot += 1   # mean skin fraction
    mapping.append(("sfss", 4, slot)); slot += 1  # strong-skin fraction
    mapping.append(("sX", 0, slot)); slot += 1
    mapping.append(("ceX", 0, slot)); slot += 1
    assert slot == NCUE, (slot, NCUE)
    ex_names = []
    src_ch = dict(max_names + [("fmax", 1), ("colocA", 1), ("colocB", 1), ("colocC", 1),
                               ("colocD", 1), ("bA", 1), ("bB", 1), ("sf5", NF),
                               ("sfss", NF), ("sX", 1), ("ceX", 1)])
    for mname in sorted(set(m for m, _, _ in mapping)):
        trip = [t for t in mapping if t[0] == mname]
        C = src_ch[mname]
        we = np.zeros((NCUE, C, 1, 1))
        for _, chn, sl in trip:
            we[sl, chn] = 1.0
        ex = "ex" + mname
        add_conv(ex, mname, NCUE, 1, we, np.zeros(NCUE))
        ex_names.append(ex)
    layers.append(add_l("cues", ex_names))

    # ---- readout ----
    Wo = np.zeros((2, NCUE))
    Wo[1] = PERSON_W
    bo = np.array([0.0, PERSON_B])
    layers.append({"name": "out", "op": "dense", "inputs": ["cues"], "out": 2})
    W["out.W"] = np.asarray(Wo, dtype=np.float32)
    W["out.b"] = np.asarray(bo, dtype=np.float32)
    return layers, W


def main(outdir):
    outdir = Path(outdir)
    layers, W = build()
    (outdir / "network.json").write_text(json.dumps({"layers": layers}))
    np.savez(outdir / "weights.npz", **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
