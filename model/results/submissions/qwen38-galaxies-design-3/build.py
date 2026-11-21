#!/usr/bin/env python3
"""Builds network.json + weights.npz for a hand-designed galaxy morphology classifier.

Everything below comes from formulas and from general knowledge of how the ten
Galaxy Zoo morphology classes look (smooth ellipticals of increasing ellipticity,
barred/unbarred spirals with tight/loose arms, edge-on discs with/without a bulge,
disturbed systems and mergers).  No data are read; no fitting is performed.

Architecture sketch
-------------------
image -> 1x1 conv (luminance L, raw colour r-b)
        -> per-channel spatial norm + relu threshold  (sky-subtracted, contrast-normalised light map Lp)
        -> two avgpool scales (64, 32)
        -> oriented ridge filter banks (8 angles, narrow & wide cross-profile, surround-suppressed)
           + channel-max via reshape/maxpool trick  (orientation-invariant ridge maps)
        -> dense layers with analytic masks: radial ring means, ring x sector means,
           regional colour means, light centroid, regional ridge/texture means
        -> maxpool grids: central peak, off-centre peak, per-band sector max
        -> small dense/relu chains: sector max/mean, opposite-sector asymmetry, opposite-pair minimum
        -> one fused 80-vector of interpretable measurements
        -> dense+relu "indicator" layer: hand-chosen thresholded comparisons (elongation,
           steepness, thinness, bulge, bar dominance, texture, colour, asymmetry, pairs, peaks)
        -> final dense = hand-written rule table over indicators and raw measurements (zero bias,
           so the decision is invariant to a global rescaling of every indicator).
"""
import json
import math
import os
import sys

import numpy as np

F = 80          # fused measurement vector length
IND = 43        # number of indicator units
BOTH = 160      # indicators (0..46) + raw measurements (80..159)


# ----------------------------------------------------------------------------- helpers
def gauss2(k, sx, sy, theta, cx=None):
    """(k,k) anisotropic gaussian, u along theta."""
    c = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k].astype(float) - c
    u = x * math.cos(theta) + y * math.sin(theta)
    v = -x * math.sin(theta) + y * math.cos(theta)
    return np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))


def ridge(k, sa, sb, sbs, sbump, theta):
    """oriented ridge filter: G(along,across) - G(across,along) - c*isotropic surround.

    cross-orientation suppression kills any radially symmetric profile (cores,
    profile shoulders); the surround is tuned (c) so that a broad isotropic bump of
    scale `sbump` also gives zero response.  What survives is genuinely
    one-dimensional bright structure narrower than `sbump` (spiral arms, dust lanes,
    bars, edge-on discs).  Normalised to unit positive mass, so the response to a
    matched ridge is of order its surface brightness."""
    g = gauss2(k, sa, sb, theta)
    gc = gauss2(k, sb, sa, theta)
    s = gauss2(k, sbs, sbs, theta)
    b = gauss2(k, sbump, sbump, 0.0)
    c = ((g - gc) * b).sum() / (s * b).sum()
    w = g - gc - c * s
    return w / w[w > 0].sum()


def rad(h, w, rc, sr, cx=None, cy=None):
    """radial gaussian weight map."""
    cy = (h - 1) / 2.0 if cy is None else cy
    cx = (w - 1) / 2.0 if cx is None else cx
    y, x = np.mgrid[0:h, 0:w].astype(float)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return np.exp(-0.5 * ((r - rc) / sr) ** 2)


def wedge(h, w, rc, sr, phi, kp=2.0):
    """ring x von-Mises sector weight map, normalised to sum 1."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.mgrid[0:h, 0:w].astype(float)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    a = np.arctan2(y - cy, x - cx)
    m = np.exp(-0.5 * ((r - rc) / sr) ** 2) * np.exp(kp * (np.cos(a - phi) - 1))
    return m / m.sum()


# ----------------------------------------------------------------------------- build
def build(out_dir):
    layers = []
    W = {}

    def L(name, op, inputs, **set_):
        layers.append(dict(name=name, op=op, inputs=inputs, **set_))

    # ---- stage 1: photometric pre-processing ------------------------------------
    w = np.zeros((3, 3, 1, 1))
    w[0, 0, 0, 0] = w[0, 1, 0, 0] = w[0, 2, 0, 0] = 1.0 / 3.0   # L
    w[1, 0, 0, 0] = 1.0                                        # r - b  (raw colour)
    w[2, 2, 0, 0] = 1.0                                        # b (raw, unused later but keeps norm honest)
    W["lum.W"], W["lum.b"] = w, np.zeros(3)
    L("lum", "conv", ["image"], out=3, kernel=[1, 1], stride=1, padding=0)
    L("nrm", "norm", ["lum"])
    # Lp = relu(L_norm - 0.15): sky and noise removed, galaxy light kept, contrast-normalised
    wl = np.zeros((1, 3, 1, 1)); wl[0, 0, 0, 0] = 1.0
    W["lsel.W"], W["lsel.b"] = wl, np.array([-0.15])
    L("lsel", "conv", ["nrm"], out=1, kernel=[1, 1], stride=1, padding=0)
    L("pos", "relu", ["lsel"])
    # raw luminance (un-normalised) for scale-free colour ratios
    wr = np.zeros((1, 3, 1, 1)); wr[0, 0, 0, 0] = 1.0
    W["lraw.W"], W["lraw.b"] = wr, np.zeros(1)
    L("lraw", "conv", ["lum"], out=1, kernel=[1, 1], stride=1, padding=0)
    wc = np.zeros((1, 3, 1, 1)); wc[0, 1, 0, 0] = 1.0
    W["rbraw.W"], W["rbraw.b"] = wc, np.zeros(1)
    L("rbraw", "conv", ["lum"], out=1, kernel=[1, 1], stride=1, padding=0)

    L("p64", "avgpool", ["pos"], size=[2, 2], stride=[2, 2])
    L("p32", "avgpool", ["p64"], size=[2, 2], stride=[2, 2])

    # ---- stage 2: oriented ridge banks ------------------------------------------
    NO = 8
    k1, k2 = 11, 15
    Wn = np.zeros((NO, 1, k1, k1))
    Ww = np.zeros((NO, 1, k2, k2))
    for o in range(NO):
        th = math.pi * o / NO
        Wn[o, 0] = ridge(k1, 2.2, 0.9, 3.0, 2.5, th)   # thin ridges (arms, lanes, edge-on discs)
        Ww[o, 0] = ridge(k2, 3.5, 1.4, 4.0, 3.0, th)   # thicker ridges (bars, cigar bodies)
    W["rn.W"], W["rn.b"] = Wn, np.zeros(NO)
    L("rn", "conv", ["p64"], out=NO, kernel=[k1, k1], stride=1, padding=k1 // 2)
    L("rnr", "relu", ["rn"])
    W["rw.W"], W["rw.b"] = Ww, np.zeros(NO)
    L("rw", "conv", ["p32"], out=NO, kernel=[k2, k2], stride=1, padding=k2 // 2)
    L("rwr", "relu", ["rw"])

    # channel-max over orientations via a running-max chain:
    # m = E_0;  m = m + relu(E_o - m)   (each linear step is a 1x1 conv selector + add + relu)
    def cmax(src, pre, h):
        for o in range(NO):
            ws = np.zeros((1, NO, 1, 1)); ws[0, o, 0, 0] = 1.0
            W[pre + "s%d.W" % o], W[pre + "s%d.b" % o] = ws, np.zeros(1)
            L(pre + "s%d" % o, "conv", [src], out=1, kernel=[1, 1], stride=1, padding=0)
        prev = pre + "s0"
        for o in range(1, NO):
            wn = np.zeros((1, NO, 1, 1)); wn[0, o, 0, 0] = -1.0
            W[pre + "n%d.W" % o], W[pre + "n%d.b" % o] = wn, np.zeros(1)
            L(pre + "n%d" % o, "conv", [src], out=1, kernel=[1, 1], stride=1, padding=0)
            L(pre + "d%d" % o, "add", [prev, pre + "n%d" % o])
            L(pre + "r%d" % o, "relu", [pre + "d%d" % o])
            cur = pre + "m%d" % o
            L(cur, "add", [pre + "r%d" % o, pre + "s%d" % o])
            prev = cur
        return prev

    rnmax = cmax("rnr", "rn", 64)
    rwmax = cmax("rwr", "rw", 32)
    wm = np.full((1, NO, 1, 1), -1.0 / NO)
    W["negm.W"], W["negm.b"] = wm, np.zeros(1)
    L("negm", "conv", ["rnr"], out=1, kernel=[1, 1], stride=1, padding=0)
    L("rnm32", "avgpool", ["rnm7"], size=[2, 2], stride=[2, 2])
    L("neg32", "avgpool", ["negm"], size=[2, 2], stride=[2, 2])
    L("dom32", "add", ["rnm32", "neg32"])          # max_over_orient - mean_over_orient

    # ---- stage 3: peak grid ------------------------------------------------------
    L("lmax", "maxpool", ["pos"], size=[8, 8], stride=[8, 8])   # (1,16,16) local maxima

    # ---- stage 4: analytic photometric masks (dense) -----------------------------
    H = 128
    rings = [2, 4, 7, 11, 16, 23, 32, 45, 60]
    bands = [(5, 2.2), (10, 3.5), (20, 6.0), (38, 11.0)]
    y, x = np.mgrid[0:H, 0:H].astype(float)
    rr = np.sqrt((y - 63.5) ** 2 + (x - 63.5) ** 2)
    wcen = np.exp(-0.5 * (rr / 26.0) ** 2)

    def slotvec():
        return np.zeros(F)

    DL = np.zeros((F, 1 * H * H))          # from Lp
    for i, rc in enumerate(rings[:8]):
        m = rad(H, H, rc, max(1.5, 0.35 * rc)) / rad(H, H, rc, max(1.5, 0.35 * rc)).sum()
        DL[i] = m.ravel()
    m8 = rad(H, H, rings[8], 0.35 * rings[8]); DL[8] = (m8 / m8.sum()).ravel()
    DL[9] = ((x - 63.5) * wcen / wcen.sum()).ravel()          # cx
    DL[10] = ((y - 63.5) * wcen / wcen.sum()).ravel()         # cy
    DL[11] = (wcen / wcen.sum()).ravel()                      # mean light in body window
    for b, (rc, sr) in enumerate(bands):
        for j in range(8):
            DL[16 + 8 * b + j] = wedge(H, H, rc, sr, math.pi * j / 4, 4.0).ravel()
    W["descL.W"], W["descL.b"] = DL, np.zeros(F)
    L("descL", "dense", ["pos"], out=F)

    gin = np.exp(-0.5 * (rr / 8.0) ** 2)
    gout = np.exp(-0.5 * ((rr - 22.0) / 9.0) ** 2)
    DL2 = np.zeros((F, 1 * H * H))
    DL2[12] = (gin / gin.sum()).ravel()
    DL2[14] = (gout / gout.sum()).ravel()
    W["descL2.W"], W["descL2.b"] = DL2, np.zeros(F)
    L("descL2", "dense", ["lraw"], out=F)
    DR2 = np.zeros((F, 1 * H * H))
    DR2[13] = (gin / gin.sum()).ravel()
    DR2[15] = (gout / gout.sum()).ravel()
    W["descR2.W"], W["descR2.b"] = DR2, np.zeros(F)
    L("descR2", "dense", ["rbraw"], out=F)

    # texture / dominance regional means at 32-scale (1 px = 4 original px)
    h2 = 32
    T1 = np.zeros((F, h2 * h2))
    T1[48] = rad(h2, h2, 3.2, 1.8).ravel() / rad(h2, h2, 3.2, 1.8).sum()      # thin-ridge, inner  (orig ~6-20)
    T1[49] = rad(h2, h2, 7.5, 2.5).ravel() / rad(h2, h2, 7.5, 2.5).sum()      # thin-ridge, mid    (orig ~20-40)
    T1[50] = rad(h2, h2, 12.5, 2.5).ravel() / rad(h2, h2, 12.5, 2.5).sum()    # thin-ridge, outer
    T1[51] = rad(h2, h2, 0.0, 1.5).ravel() / rad(h2, h2, 0.0, 1.5).sum()      # thin-ridge, core
    W["tex1.W"], W["tex1.b"] = T1, np.zeros(F)
    L("tex1", "dense", ["rnm32"], out=F)
    T2 = np.zeros((F, h2 * h2))
    T2[52] = rad(h2, h2, 3.5, 2.5).ravel() / rad(h2, h2, 3.5, 2.5).sum()      # wide-ridge, body
    T2[53] = rad(h2, h2, 0.0, 1.5).ravel() / rad(h2, h2, 0.0, 1.5).sum()      # wide-ridge, core
    W["tex2.W"], W["tex2.b"] = T2, np.zeros(F)
    L("tex2", "dense", ["rwm7"], out=F)
    T3 = np.zeros((F, h2 * h2))
    T3[54] = rad(h2, h2, 3.5, 2.0).ravel() / rad(h2, h2, 3.5, 2.0).sum()      # orient. dominance, inner
    T3[55] = rad(h2, h2, 9.0, 3.0).ravel() / rad(h2, h2, 9.0, 3.0).sum()      # orient. dominance, mid
    W["tex3.W"], W["tex3.b"] = T3, np.zeros(F)
    L("tex3", "dense", ["dom32"], out=F)

    # peaks: central 2x2 cells and off-centre annulus cells of the 8x8 maxpool grid
    gc = np.mgrid[0:16, 0:16]
    crad = np.sqrt((gc[0] * 8 + 3.5 - 63.5) ** 2 + (gc[1] * 8 + 3.5 - 63.5) ** 2)
    cen = [(7, 7), (7, 8), (8, 7), (8, 8)]
    ann = [(i, j) for i in range(16) for j in range(16) if 12 <= crad[i, j] <= 44]
    PK = np.zeros((4, 256))
    for n, (i, j) in enumerate(cen):
        PK[n, i * 16 + j] = 1.0
    W["pkc.W"], W["pkc.b"] = PK, np.zeros(4)
    L("pkc", "dense", ["lmax"], out=4)
    L("pkc_sh", "reshape", ["pkc"], shape=[1, 4, 1])
    L("pkc_mp", "maxpool", ["pkc_sh"], size=[4, 1], stride=[4, 1])
    PC = np.zeros((F, 1)); PC[56] = 1.0
    W["pcpad.W"], W["pcpad.b"] = PC, np.zeros(F)
    L("pcpad", "dense", ["pkc_mp"], out=F)
    KA = len(ann)
    PA = np.zeros((KA, 256))
    for n, (i, j) in enumerate(ann):
        PA[n, i * 16 + j] = 1.0
    W["pka.W"], W["pka.b"] = PA, np.zeros(KA)
    L("pka", "dense", ["lmax"], out=KA)
    L("pka_sh", "reshape", ["pka"], shape=[1, KA, 1])
    L("pka_mp", "maxpool", ["pka_sh"], size=[KA, 1], stride=[KA, 1])
    PA2 = np.zeros((F, 1)); PA2[57] = 1.0
    W["papad.W"], W["papad.b"] = PA2, np.zeros(F)
    L("papad", "dense", ["pka_mp"], out=F)

    L("feat0", "add", ["descL", "descL2", "descR2", "tex1", "tex2", "tex3", "pcpad", "papad"])

    # ---- stage 5: sector statistics ----------------------------------------------
    L("f_sh", "reshape", ["feat0"], shape=[10, 8, 1])
    L("f_mp", "maxpool", ["f_sh"], size=[8, 1], stride=[8, 1])     # (10,1,1) group maxima
    SM = np.zeros((F, 10))
    for b in range(4):
        SM[64 + b, 2 + b] = 1.0                                    # maxS per band
    W["smax.W"], W["smax.b"] = SM, np.zeros(F)
    L("smax", "dense", ["f_mp"], out=F)
    SN = np.zeros((F, F))
    for b in range(4):
        for j in range(8):
            SN[68 + b, 16 + 8 * b + j] = 1.0 / 8.0                # meanS per band
    W["smean.W"], W["smean.b"] = SN, np.zeros(F)
    L("smean", "dense", ["feat0"], out=F)
    AA = np.zeros((32, F))
    for b in range(4):
        for j in range(4):
            AA[b * 4 + j, 16 + 8 * b + j] = 1.0
            AA[b * 4 + j, 16 + 8 * b + j + 4] = -1.0
            AA[16 + b * 4 + j, 16 + 8 * b + j] = -1.0
            AA[16 + b * 4 + j, 16 + 8 * b + j + 4] = 1.0
    W["asymA.W"], W["asymA.b"] = AA, np.zeros(32)
    L("asymA", "dense", ["feat0"], out=32)
    L("asymR", "relu", ["asymA"])
    AB = np.zeros((F, 32))
    for b in range(4):
        for j in range(4):
            AB[72 + b, b * 4 + j] = 0.25
            AB[72 + b, 16 + b * 4 + j] = 0.25
    W["asymB.W"], W["asymB.b"] = AB, np.zeros(F)
    L("asymB", "dense", ["asymR"], out=F)
    C1 = np.zeros((16, F))
    for b in range(4):
        for j in range(4):
            C1[b * 4 + j, 16 + 8 * b + j] = 0.5
            C1[b * 4 + j, 16 + 8 * b + j + 4] = 0.5
    W["pairA.W"], W["pairA.b"] = C1, np.zeros(16)
    L("pairA", "dense", ["feat0"], out=16)
    C2 = np.zeros((16, 32))
    for b in range(4):
        for j in range(4):
            C2[b * 4 + j, b * 4 + j] = -0.5
            C2[b * 4 + j, 16 + b * 4 + j] = -0.5
    W["pairB.W"], W["pairB.b"] = C2, np.zeros(16)
    L("pairB", "dense", ["asymR"], out=16)
    L("pairS", "add", ["pairA", "pairB"])                          # min(S_j, S_j+4)
    L("pair_sh", "reshape", ["pairS"], shape=[4, 4, 1])
    L("pair_mp", "maxpool", ["pair_sh"], size=[4, 1], stride=[4, 1])
    PM = np.zeros((F, 4))
    for b in range(4):
        PM[76 + b, b] = 1.0
    W["pairC.W"], W["pairC.b"] = PM, np.zeros(F)
    L("pairC", "dense", ["pair_mp"], out=F)

    L("feat", "add", ["feat0", "smax", "smean", "asymB", "pairC"])

    # ---- stage 6: hand-set indicator layer ----------------------------------------
    # slot names
    S = {}
    for i in range(8):
        S["m%d" % i] = i
    S.update(m8=8, cx=9, cy=10, lw=11, lr_in=12, cr_in=13, lr_out=14, cr_out=15)
    for b in range(4):
        for j in range(8):
            S["S%d_%d" % (b, j)] = 16 + 8 * b + j
    S.update(t_in=48, t_mid=49, t_out=50, rn_c=51, rw_in=52, rw_c=53, dom_in=54, dom_mid=55)
    S.update(pc=56, pa=57)
    for b in range(4):
        S["mxS%d" % b] = 64 + b
        S["mnS%d" % b] = 68 + b
        S["as%d" % b] = 72 + b
        S["pm%d" % b] = 76 + b

    rows = []   # (name, {slot: coeff})

    def R(name, **kw):
        rows.append((name, {S[k]: v for k, v in kw.items()}))

    # sector-elongation ladder (scale-free ratios of max to mean sector light)
    for b in range(4):
        R("e1_%d" % b, **{"mxS%d" % b: 1.0, "mnS%d" % b: -1.20})
    for b in range(4):
        R("e2_%d" % b, **{"mxS%d" % b: 1.0, "mnS%d" % b: -1.55})
    for b in range(4):
        R("e3_%d" % b, **{"mxS%d" % b: 1.0, "mnS%d" % b: -1.90})
    R("e2b_2", mxS2=1.0, mnS2=-1.65)
    R("e2b_3", mxS3=1.0, mnS3=-1.70)
    R("e3b_2", mxS2=1.0, mnS2=-1.35)
    R("e3b_3", mxS3=1.0, mnS3=-1.60)
    # roundness ladder
    R("rd_1", mnS1=1.12, mxS1=-1.0)
    R("rd2s", mnS2=1.10, mxS2=-1.0)
    R("rd3s", mnS3=1.25, mxS3=-1.0)
    # profile steepness / concentration
    R("st1", m2=1.0, m4=-1.8)
    R("st2", m4=1.0, m6=-1.8)
    R("conc", m0=1.0, lw=-3.0)
    # ridge (arm / disc) texture, relative to local light
    R("texr1", t_mid=1.0, m3=-0.12)
    R("thin1", t_mid=1.0, m3=-0.27)
    R("thg", t_mid=1.0, m3=-0.20)
    R("thinout", t_out=1.0, m4=-0.025)
    # thick-ridge (bar / cigar body / edge-on disc) contrasts
    R("rwbar", rw_c=1.0, rw_in=-1.35)
    R("rwmid", rw_c=1.0, rw_in=-1.00)
    R("rwblo", rw_c=1.0, rw_in=-0.92)
    R("ebar", mxS1=1.0, mnS1=-1.24)
    # colours (raw r-b against raw luminance: scale free)
    R("redc", cr_in=1.0, lr_in=-0.12)
    R("bluec", cr_in=-1.0, lr_in=-0.02)
    R("blueo", cr_out=-1.0, lr_out=-0.02)
    R("redgrad", cr_in=1.0, lr_in=-0.10, cr_out=-1.0, lr_out=0.10)
    # lopsidedness / multiplicity
    for b in (1, 2, 3):
        R("as1_%d" % b, **{"as%d" % b: 1.0, "mnS%d" % b: -0.35, "lw": -0.08})
    R("pmx0", mxS0=0.90, pm0=-1.0)
    R("pmx1", mxS1=0.93, pm1=-1.0)
    R("offxp", cx=1.0, lw=-1.5)
    R("offxn", cx=-1.0, lw=-1.5)
    R("offyp", cy=1.0, lw=-1.5)
    R("offyn", cy=-1.0, lw=-1.5)
    assert len(rows) == IND, len(rows)

    IW = np.zeros((IND, F))
    for i, (nm, cf) in enumerate(rows):
        for sl, v in cf.items():
            IW[i, sl] = v
    W["indr.W"], W["indr.b"] = IW, np.zeros(IND)
    L("indr", "dense", ["feat"], out=IND)
    L("ind", "relu", ["indr"])

    IP = np.zeros((BOTH, IND))
    for i in range(IND):
        IP[i, i] = 1.0
    W["indp.W"], W["indp.b"] = IP, np.zeros(BOTH)
    L("indp", "dense", ["ind"], out=BOTH)
    RP = np.zeros((BOTH, F))
    for i in range(F):
        RP[80 + i, i] = 1.0
    W["rawp.W"], W["rawp.b"] = RP, np.zeros(BOTH)
    L("rawp", "dense", ["feat"], out=BOTH)
    L("both", "add", ["indp", "rawp"])

    # ---- stage 7: hand-written rule table -----------------------------------------
    # weights over [indicators 0..46 | raw slots 80+slot]
    RULE = {
        0: dict(as1_1=5.0, as1_3=1.2, pmx1=0.2, bluec=0.6, blueo=0.5,
                offxp=0.3, offxn=0.3, offyp=0.3, offyn=0.3,
                pmx0=-2.5, e3_3=-0.4, thinout=-1.5, texr1=-0.5),
        1: dict(pmx0=1.6, pmx1=1.0, as1_2=1.2, as1_3=0.8, e2_1=0.5,
                thinout=-1.5, as1_1=-0.8, rd_1=-0.5),
        2: dict(rd_1=1.0, rd2s=1.5, rd3s=1.5, st1=0.1, st2=0.1, conc=0.1, redc=0.2,
                texr1=-4.0, thinout=-2.0, e1_2=-2.5, e2_2=-1.5, as1_2=-0.8),
        3: dict(e1_2=3.0, e1_3=1.5, e2b_2=-6.0, st1=0.1,
                texr1=-2.0, thinout=-2.0, as1_2=-0.8),
        4: dict(e2b_2=6.0, e2b_3=1.5, e2_2=1.0, rwmid=0.8, st1=0.1, e3_2=-0.5,
                thinout=-3.0, texr1=-2.0, as1_2=-4.0, pmx0=-4.0, thg=-30.0),
        5: dict(ebar=3.0, texr1=2.0, e1_2=0.3, thinout=-1.5, thin1=-4.0,
                blueo=-0.6, as1_2=-0.5, pmx0=-0.8, e2_1=-2.0, e2_2=-2.0,
                e2_3=-2.0, e3b_2=-2.0, e3b_3=-2.0, e2b_2=-3.0),
        6: dict(texr1=2.5, thinout=-1.5, ebar=-1.0, rwblo=-1.0, redgrad=1.0,
                e2_2=-0.6, thin1=-1.0),
        7: dict(texr1=1.5, thinout=0.5, rwblo=1.0, blueo=0.9, bluec=0.5, as1_2=0.3,
                ebar=-0.8, e2_2=-0.4, thin1=-1.0, redgrad=-0.5),
        8: dict(rwbar=1.6, thin1=2.0, thinout=1.0, e2_2=0.6, e2b_2=-2.0,
                pmx0=-1.0, pmx1=-0.8, as1_2=-0.5),
        9: dict(e3b_3=9.0, e3b_2=4.0, thinout=4.0, rwbar=-1.5, rwmid=-1.0,
                thin1=-1.5, texr1=-1.0, pmx0=-1.0, st1=0.1, e2b_2=-6.0),
    }
    name2i = {nm: i for i, (nm, _) in enumerate(rows)}
    FW = np.zeros((10, BOTH))
    for k, rule in RULE.items():
        for nm, v in rule.items():
            if nm in name2i:
                FW[k, name2i[nm]] = v
            else:
                FW[k, 80 + S[nm]] = v
    W["scores.W"], W["scores.b"] = FW, np.zeros(10)
    L("scores", "dense", ["both"], out=10)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
    return layers, W


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
