#!/usr/bin/env python3
"""Hand-designed galaxy morphology network (Galaxy10 DECaLS classes), weights written from formulas only.

Usage: python3 build.py OUT_DIR   -> writes OUT_DIR/network.json and OUT_DIR/weights.npz

Design (no training data is used anywhere in this file):
  1. 3x3 binomial smoothing of the image; background (0.13) subtracted; luminance, red (z) and blue (g) maps;
     a band-pass map (Gaussian sigma 1 minus Gaussian sigma 3) for clumpiness.
  2. Polar resampling (dense layers whose weights are the pixel/cell overlap areas): 32 sectors x 24 radial
     bins of 2 px, for luminance, red and blue.
  3. Isophotal ray extents at four surface-brightness levels; major/minor axes by max/min over sectors
     (maxpool); isophotal axis ratios, colours, 180-degree asymmetry and clumpiness turned into ratios with the
     'norm' layer applied to (a, b, 0) triplets, which is a monotonic function of a/b.
  4. Soft threshold indicators (relu ramps), combined with Lukasiewicz AND (relu(a+b-1)) into a
     hand-written decision tree over: colour (red/blue), axis ratio (edge-on / cigar / in-between / round),
     bulge, asymmetry (mergers), bar (inner isophote more elongated than outer), concentration, clumpiness.
All thresholds are chosen by hand from general knowledge of galaxy morphology and of the image rendering.
"""
import json
import os
import sys

import numpy as np

H = W = 128
CY = CX = 63.5          # image centre (pixel coordinates)
BG = 0.13               # sky background level in the 0..1 images
NSEC = 32               # angular sectors
NRAD = 24               # radial bins
DR = 2.0                # radial bin width in pixels
ISO = [0.06, 0.16, 0.30, 0.50]   # isophote levels above background: low, mid1, mid2, high
RAMP_W = 0.04           # softness of the isophote ramps (luminance units)


# ----------------------------------------------------------------------------------------------------------
# small graph builder
# ----------------------------------------------------------------------------------------------------------
class Net:
    def __init__(self):
        self.layers = []
        self.weights = {}
        self.shapes = {"image": (3, H, W)}

    def _add(self, name, op, inputs, out_shape, **settings):
        assert name not in self.shapes, name
        layer = {"name": name, "op": op, "inputs": list(inputs)}
        layer.update(settings)
        self.layers.append(layer)
        self.shapes[name] = tuple(int(v) for v in out_shape)
        return name

    def conv(self, name, inp, Wt, b, stride=1, padding=0):
        Wt = np.asarray(Wt, np.float32)
        b = np.asarray(b, np.float32)
        c, h, w = self.shapes[inp]
        o, i, k, k2 = Wt.shape
        assert i == c and k == k2 and b.shape == (o,)
        ho = (h + 2 * padding - k) // stride + 1
        wo = (w + 2 * padding - k) // stride + 1
        self.weights[name + ".W"] = Wt
        self.weights[name + ".b"] = b
        return self._add(name, "conv", [inp], (o, ho, wo), out=o, kernel=[k, k], stride=stride, padding=padding)

    def dense(self, name, inp, Wt, b=None):
        Wt = np.asarray(Wt, np.float32)
        n_in = int(np.prod(self.shapes[inp]))
        assert Wt.shape[1] == n_in, (Wt.shape, n_in)
        if b is None:
            b = np.zeros(Wt.shape[0], np.float32)
        b = np.asarray(b, np.float32)
        self.weights[name + ".W"] = Wt
        self.weights[name + ".b"] = b
        return self._add(name, "dense", [inp], (Wt.shape[0],), out=int(Wt.shape[0]))

    def relu(self, name, inp):
        return self._add(name, "relu", [inp], self.shapes[inp])

    def maxpool(self, name, inp, size, stride):
        c, h, w = self.shapes[inp]
        ho = (h - size[0]) // stride[0] + 1
        wo = (w - size[1]) // stride[1] + 1
        return self._add(name, "maxpool", [inp], (c, ho, wo), size=list(size), stride=list(stride))

    def avgpool(self, name, inp, size, stride):
        c, h, w = self.shapes[inp]
        ho = (h - size[0]) // stride[0] + 1
        wo = (w - size[1]) // stride[1] + 1
        return self._add(name, "avgpool", [inp], (c, ho, wo), size=list(size), stride=list(stride))

    def reshape(self, name, inp, shape):
        assert int(np.prod(shape)) == int(np.prod(self.shapes[inp]))
        return self._add(name, "reshape", [inp], tuple(shape), shape=[int(v) for v in shape])

    def norm(self, name, inp):
        assert len(self.shapes[inp]) == 3
        return self._add(name, "norm", [inp], self.shapes[inp])

    def add(self, name, inputs):
        return self._add(name, "add", inputs, self.shapes[inputs[0]])

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "network.json"), "w") as f:
            json.dump({"layers": self.layers}, f)
        np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in self.weights.items()})


# ----------------------------------------------------------------------------------------------------------
# formula-based building blocks
# ----------------------------------------------------------------------------------------------------------
def gauss2d(sigma, k):
    a = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-a ** 2 / (2.0 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def binom3():
    g = np.array([1.0, 2.0, 1.0]) / 4.0
    return np.outer(g, g)


def embed(kernel, k):
    """Zero-pad a small square kernel into a k x k kernel."""
    out = np.zeros((k, k))
    s = kernel.shape[0]
    o = (k - s) // 2
    out[o:o + s, o:o + s] = kernel
    return out


def polar_matrix():
    """(NSEC*NRAD, H*W) matrix: row c holds the mean-over-cell weights of polar cell c. Cell index = j*NSEC + k."""
    sub = (np.arange(4) + 0.5) / 4.0 - 0.5          # 4x4 sub-samples per pixel
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    Wm = np.zeros((NSEC * NRAD, H * W))
    for dy in sub:
        for dx in sub:
            y = yy + dy - CY
            x = xx + dx - CX
            r = np.hypot(y, x)
            th = np.mod(np.arctan2(y, x), 2 * np.pi)
            j = np.floor(r / DR).astype(int)
            k = np.floor(th / (2 * np.pi / NSEC)).astype(int)
            k = np.minimum(k, NSEC - 1)
            ok = j < NRAD
            cell = (j * NSEC + k)[ok]
            pix = (yy * W + xx)[ok]
            np.add.at(Wm, (cell, pix), 1.0 / 16.0)
    area = Wm.sum(1, keepdims=True)                 # pixel area of each cell
    Wm = Wm / np.maximum(area, 1e-9)
    return Wm, area[:, 0]


def g_of_ratio(rho):
    """Value of the first entry of norm([a, b, 0]) as a function of rho = a / b (a, b >= 0)."""
    rho = np.asarray(rho, float)
    return (2 * rho - 1) / np.sqrt(2 * (rho ** 2 - rho + 1))


# ----------------------------------------------------------------------------------------------------------
# the network
# ----------------------------------------------------------------------------------------------------------
def build():
    net = Net()
    K = 11
    sm = embed(binom3(), K)                       # 3x3 smoothing inside an 11x11 kernel
    bp = (gauss2d(1.0, K) - gauss2d(3.0, K))      # band-pass (sums to zero -> background cancels)

    # ---- stage 1: smoothed maps, all linear in the image -----------------------------------------------
    # channels: 0 L (luminance - bg), 1 R'-smoothed, 2 B'-smoothed, 3 bp, 4 -bp,
    #           5 s*(L - iso1), 6 s*(L - iso1) - 1   (steep ramp pieces for the pixel mask at the mid1 isophote)
    #           7 L - 0.05  (noise-thresholded luminance for the asymmetry map)
    s_pix = 50.0
    W1 = np.zeros((8, 3, K, K))
    b1 = np.zeros(8)
    for c in range(3):
        W1[0, c] = sm / 3.0
        W1[3, c] = bp / 3.0
        W1[4, c] = -bp / 3.0
        W1[5, c] = s_pix * sm / 3.0
        W1[6, c] = s_pix * sm / 3.0
        W1[7, c] = sm / 3.0
    W1[1, 0] = sm
    W1[2, 2] = sm
    b1[0] = -BG
    b1[1] = -BG
    b1[2] = -BG
    b1[5] = -s_pix * (BG + ISO[1])
    b1[6] = -s_pix * (BG + ISO[1]) - 1.0
    b1[7] = -BG - 0.05
    net.conv("f1", "image", W1, b1, padding=K // 2)
    net.relu("f1r", "f1")
    # 1x1 combine: 0 L+, 1 R+, 2 B+, 3 |bp|, 4 mask m (= relu(x) - relu(x-1)), 5 L-0.05 (+)
    W2 = np.zeros((6, 8, 1, 1))
    W2[0, 0] = 1
    W2[1, 1] = 1
    W2[2, 2] = 1
    W2[3, 3] = 1
    W2[3, 4] = 1
    W2[4, 5] = 1
    W2[4, 6] = -1
    W2[5, 7] = 1
    net.conv("f2", "f1r", W2, np.zeros(6))
    # gated maps: |bp| + m - 1 and L + m - 1 (relu -> value where the mask is on, 0 elsewhere)
    W3 = np.zeros((2, 6, 1, 1))
    W3[0, 3] = 1
    W3[0, 4] = 1
    W3[1, 0] = 1
    W3[1, 4] = 1
    net.conv("f3", "f2", W3, np.array([-1.0, -1.0]))
    net.relu("f3r", "f3")

    def select(name, src, ch, nch):
        Ws = np.zeros((1, nch, 1, 1))
        Ws[0, ch] = 1
        return net.conv(name, src, Ws, np.zeros(1))

    select("selL", "f2", 0, 6)
    select("selR", "f2", 1, 6)
    select("selB", "f2", 2, 6)
    select("selA", "f2", 5, 6)
    select("selC", "f3r", 0, 2)
    select("selD", "f3r", 1, 2)

    # ---- stage 2: polar resampling -----------------------------------------------------------------------
    Pm, area = polar_matrix()
    net.dense("polL", "selL", Pm)
    net.dense("polR", "selR", Pm)
    net.dense("polB", "selB", Pm)
    NC = NSEC * NRAD
    cell_j = np.arange(NC) // NSEC                 # radial bin of each cell
    cell_k = np.arange(NC) % NSEC                  # sector of each cell
    r_cell = (cell_j + 0.5) * DR

    # ---- stage 3a: isophote ramps on luminance cells -----------------------------------------------------
    # for each level t and cell c: u = (P - iso_t)/RAMP_W ; ramp = relu(u) - relu(u - 1)
    nT = len(ISO)
    Wr = np.zeros((2 * nT * NC, NC))
    br = np.zeros(2 * nT * NC)
    for t, iso in enumerate(ISO):
        for part in range(2):
            rows = slice((2 * t + part) * NC, (2 * t + part + 1) * NC)
            Wr[rows, :] = np.eye(NC) / RAMP_W
            br[rows] = -iso / RAMP_W - part
    net.dense("isoU", "polL", Wr, br)
    net.relu("isoUr", "isoU")
    # ramps m_t(c) = u_part0 - u_part1  -> (nT*NC)
    Wm = np.zeros((nT * NC, 2 * nT * NC))
    for t in range(nT):
        Wm[t * NC:(t + 1) * NC, (2 * t) * NC:(2 * t + 1) * NC] = np.eye(NC)
        Wm[t * NC:(t + 1) * NC, (2 * t + 1) * NC:(2 * t + 2) * NC] = -np.eye(NC)
    net.dense("isoM", "isoUr", Wm)
    # ray extents E_t(k) = sum_j m_t(j,k) ; layout rows: [E_0..E_3, -E_0..-E_3] each NSEC wide
    We = np.zeros((2 * nT * NSEC, nT * NC))
    for t in range(nT):
        for c in range(NC):
            We[t * NSEC + cell_k[c], t * NC + c] = 1.0
            We[(nT + t) * NSEC + cell_k[c], t * NC + c] = -1.0
    net.dense("ext", "isoM", We)
    net.reshape("extR", "ext", (2 * nT, NSEC, 1))
    net.maxpool("extMax", "extR", [NSEC, 1], [NSEC, 1])       # (2nT,1,1): [maj_t ..., -min_t ...]
    net.reshape("axes", "extMax", (2 * nT,))

    # mean extents and colour sums come from the same isoM / polar layers ------------------------------------
    # colour masking at the cell level: gated_B = relu(P_B + m_low - 1), gated_R = relu(P_R + m_low - 1)
    # Build a combined vector: [P_R (NC), P_B (NC), m_low (NC)] via three dense layers then add.
    I_NC = np.eye(NC)
    Z = np.zeros((NC, NC))
    # pack R: rows 0..NC-1 ; B: rows NC..2NC-1 ; from isoM take m_low (t=0) into both blocks
    WpR = np.vstack([I_NC, Z])
    WpB = np.vstack([Z, I_NC])
    net.dense("packR", "polR", WpR, np.full(2 * NC, -1.0))
    net.dense("packB", "polB", WpB)
    WpM = np.zeros((2 * NC, nT * NC))
    WpM[:NC, :NC] = I_NC
    WpM[NC:, :NC] = I_NC
    net.dense("packM", "isoM", WpM)
    net.add("colGate", ["packR", "packB", "packM"])
    net.relu("colGateR", "colGate")               # [gated R (NC), gated B (NC)]

    # ---- stage 3b: asymmetry (180 degree rotation about the centre) at 64x64 ------------------------------
    net.relu("selAr", "selA")
    net.avgpool("asyP", "selAr", [2, 2], [2, 2])              # (1,64,64)
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    rr = np.hypot(yy - 31.5, xx - 31.5)
    half = (rr < 18) & ((yy < 32))                            # one half-disk (rows 0..31)
    idx = np.flatnonzero(half)
    Wd = np.zeros((2 * len(idx), 64 * 64))
    for n, p in enumerate(idx):
        y, x = divmod(int(p), 64)
        q = (63 - y) * 64 + (63 - x)
        Wd[n, p] = 1.0
        Wd[n, q] = -1.0
        Wd[len(idx) + n, p] = -1.0
        Wd[len(idx) + n, q] = 1.0
    net.dense("asyD", "asyP", Wd)
    net.relu("asyDr", "asyD")
    # sums: asy_num = sum |D| (both halves -> factor 2 covered by using both signs over one half ... we
    # sum |D| over the half disk, i.e. half of the full-disk sum), asy_den = sum L over the full disk
    Ws = np.zeros((1, 2 * len(idx)))
    Ws[0, :] = 1.0
    net.dense("asyNum", "asyDr", Ws)
    disk = (rr < 18).astype(float).reshape(1, -1)
    net.dense("asyDen", "asyP", disk)

    # ---- stage 3c: clumpiness sums ------------------------------------------------------------------------
    yy2, xx2 = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rpix = np.hypot(yy2 - CY, xx2 - CX)
    cw = ((rpix >= 4.0) & (rpix < 44.0)).astype(float).reshape(1, -1)
    net.dense("clNum", "selC", cw)
    net.dense("clDen", "selD", cw)

    # ---- stage 4: gather scalar features into one vector --------------------------------------------------
    # feature vector F layout:
    #  0..3 maj_t, 4..7 min_t (positive), 8 sumR_out, 9 sumB_out, 10 sumR_in, 11 sumB_in,
    #  12 asyNum, 13 asyDen, 14 clNum, 15 clDen, 16 sumR_tot, 17 sumB_tot, 18 meanext_low, 19 meanext_mid1
    NF = 20
    Wa = np.zeros((NF, 2 * nT))
    for t in range(nT):
        Wa[t, t] = 1.0
        Wa[4 + t, nT + t] = -1.0
    net.dense("gA", "axes", Wa)
    Wc = np.zeros((NF, 2 * NC))
    inner = r_cell < 6.0
    for c in range(NC):
        a = area[c]
        if inner[c]:
            Wc[10, c] = a
            Wc[11, NC + c] = a
        else:
            Wc[8, c] = a
            Wc[9, NC + c] = a
        Wc[16, c] = a
        Wc[17, NC + c] = a
    net.dense("gC", "colGateR", Wc)
    Wn = np.zeros((NF, 1))
    Wn[12, 0] = 1.0
    net.dense("gAN", "asyNum", Wn)
    Wn = np.zeros((NF, 1))
    Wn[13, 0] = 1.0
    net.dense("gAD", "asyDen", Wn)
    Wn = np.zeros((NF, 1))
    Wn[14, 0] = 1.0
    net.dense("gCN", "clNum", Wn)
    Wn = np.zeros((NF, 1))
    Wn[15, 0] = 1.0
    net.dense("gCD", "clDen", Wn)
    Wx = np.zeros((NF, 2 * nT * NSEC))
    Wx[18, 0:NSEC] = 1.0 / NSEC
    Wx[19, NSEC:2 * NSEC] = 1.0 / NSEC
    net.dense("gX", "ext", Wx)
    net.add("F", ["gA", "gC", "gAN", "gAD", "gCN", "gCD", "gX"])

    # ---- stage 5: ratios through norm on (a, b, 0) triplets ------------------------------------------------
    # triplet list: (name, index_a, index_b, scale_a)  -> value g(scale_a * a / b)
    triplets = [
        ("q_low", 4, 0, 1.0),
        ("q_m1", 5, 1, 1.0),
        ("q_m2", 6, 2, 1.0),
        ("q_high", 7, 3, 1.0),
        ("col_out", 9, 8, 1.0),      # B/R outer
        ("col_in", 11, 10, 1.0),     # B/R inner
        ("asym", 12, 13, 4.0),       # 4 * (half-disk sum|D|) / sum L  = 2 * A_classic  (A = sum|I-I180|/sum I)
        ("clump", 14, 15, 3.0),      # 3 * clumpiness
        ("conc", 3, 0, 2.0),         # 2 * maj_high / maj_low
        ("col_tot", 17, 16, 1.0),
        ("ext_ratio", 19, 18, 2.0),  # 2 * mean extent at mid1 / mean extent at low
    ]
    nTr = len(triplets)
    Wt = np.zeros((3 * nTr, NF))
    for i, (_, ia, ib, sa) in enumerate(triplets):
        Wt[3 * i, ia] = sa
        Wt[3 * i + 1, ib] = 1.0
    net.dense("trip", "F", Wt)
    net.reshape("tripR", "trip", (nTr, 3, 1))
    net.norm("tripN", "tripR")
    net.reshape("ratios", "tripN", (3 * nTr,))
    ridx = {name: 3 * i for i, (name, _, _, _) in enumerate(triplets)}
    rscale = {name: sa for (name, _, _, sa) in triplets}

    # ---- stage 6: indicator ramps ---------------------------------------------------------------------------
    # Each indicator = clip((x - a)/(b - a), 0, 1) where x is either a ratio (via g) or a raw feature.
    # We build a combined input vector [ratios (3nTr), F (NF)] by two dense layers + add.
    NR = 3 * nTr
    Wv1 = np.vstack([np.eye(NR), np.zeros((NF, NR))])
    Wv2 = np.vstack([np.zeros((NR, NF)), np.eye(NF)])
    net.dense("v1", "ratios", Wv1)
    net.dense("v2", "F", Wv2)
    net.add("V", ["v1", "v2"])
    NV = NR + NF

    ramps = []   # (name, input index in V, a, b) : indicator rises from 0 at a to 1 at b (b may be < a)

    def ratio_ramp(name, rname, rho_a, rho_b):
        s = rscale[rname]
        ramps.append((name, ridx[rname], float(g_of_ratio(s * rho_a)), float(g_of_ratio(s * rho_b))))

    def raw_ramp(name, fidx, a, b):
        ramps.append((name, NR + fidx, float(a), float(b)))

    # shape classes from the low-isophote axis ratio q = minor/major
    ratio_ramp("edge", "q_low", 0.40, 0.28)          # 1 when q < 0.28
    ratio_ramp("notedge", "q_low", 0.28, 0.40)
    ratio_ramp("cigar_lo", "q_low", 0.28, 0.36)      # q above ~0.3
    ratio_ramp("cigar_hi", "q_low", 0.52, 0.44)      # q below ~0.5
    ratio_ramp("inb_lo", "q_low", 0.44, 0.52)
    ratio_ramp("inb_hi", "q_low", 0.84, 0.72)        # q below ~0.78
    ratio_ramp("round", "q_low", 0.72, 0.84)         # q above ~0.78
    # colour
    ratio_ramp("blue", "col_out", 0.78, 0.92)
    ratio_ramp("yellow", "col_out", 0.92, 0.78)
    ratio_ramp("blue_in", "col_in", 0.80, 0.95)
    ratio_ramp("yellow_in", "col_in", 0.95, 0.80)
    # asymmetry (classic A = sum|I-I180| / sum I ; our ratio is 2A)
    ratio_ramp("asym", "asym", 2 * 0.22, 2 * 0.34)
    ratio_ramp("sym", "asym", 2 * 0.34, 2 * 0.22)
    ratio_ramp("asym_mild", "asym", 2 * 0.13, 2 * 0.20)
    # bulge for edge-on galaxies: high isophote exists, is wide and roundish
    ratio_ramp("q_high_round", "q_high", 0.35, 0.60)
    raw_ramp("high_exists", 3, 1.5, 2.5)             # maj_high in cells (2 px each)
    raw_ramp("min_high_wide", 7, 1.2, 2.2)           # min_high in cells
    # bar: inner isophote (mid1 or mid2) more elongated than the outer one
    ratio_ramp("q_m1_elong", "q_m1", 0.62, 0.45)
    ratio_ramp("q_m2_elong", "q_m2", 0.62, 0.45)
    ratio_ramp("q_low_roundish", "q_low", 0.55, 0.70)
    raw_ramp("m1_exists", 1, 2.0, 3.0)
    raw_ramp("m2_exists", 2, 2.0, 3.0)
    # concentration / clumpiness
    ratio_ramp("conc_hi", "conc", 0.30, 0.45)        # bright isophote extends far relative to the faint one
    ratio_ramp("conc_lo", "conc", 0.45, 0.30)
    ratio_ramp("clumpy", "clump", 0.22, 0.34)
    ratio_ramp("smooth", "clump", 0.34, 0.22)

    nRm = len(ramps)
    Wrm = np.zeros((2 * nRm, NV))
    brm = np.zeros(2 * nRm)
    for i, (_, vi, a, b) in enumerate(ramps):
        k = 1.0 / (b - a)
        Wrm[2 * i, vi] = k
        brm[2 * i] = -a * k
        Wrm[2 * i + 1, vi] = k
        brm[2 * i + 1] = -a * k - 1.0
    net.dense("rampU", "V", Wrm, brm)
    net.relu("rampUr", "rampU")
    Wrc = np.zeros((nRm, 2 * nRm))
    for i in range(nRm):
        Wrc[i, 2 * i] = 1.0
        Wrc[i, 2 * i + 1] = -1.0
    net.dense("ind", "rampUr", Wrc)
    I = {name: i for i, (name, _, _, _) in enumerate(ramps)}

    # ---- stage 7: AND-combinations (Lukasiewicz): relu(sum - (n-1)) -------------------------------------------
    ands = [
        ("bulge", ["q_high_round", "high_exists", "min_high_wide"]),
        ("bar1", ["q_m1_elong", "q_low_roundish", "m1_exists"]),
        ("bar2", ["q_m2_elong", "q_low_roundish", "m2_exists"]),
        ("cigar", ["cigar_lo", "cigar_hi"]),
        ("inbetween", ["inb_lo", "inb_hi"]),
        ("g_yel", ["yellow", "notedge", "sym"]),
        ("g_blu", ["blue", "notedge", "sym"]),
        ("g_mrg", ["asym", "notedge"]),
        ("g_edge", ["edge"]),
        ("mild_only", ["asym_mild", "sym"]),
    ]
    nA = len(ands)
    Wand = np.zeros((nA, nRm))
    band = np.zeros(nA)
    for i, (_, names) in enumerate(ands):
        for nm in names:
            Wand[i, I[nm]] = 1.0
        band[i] = -(len(names) - 1)
    net.dense("andU", "ind", Wand, band)
    net.relu("A1", "andU")
    A = {name: i for i, (name, _) in enumerate(ands)}

    # carry indicators forward as well: V2 = [A1 (nA), ind (nRm)]
    Wc1 = np.vstack([np.eye(nA), np.zeros((nRm, nA))])
    Wc2 = np.vstack([np.zeros((nA, nRm)), np.eye(nRm)])
    net.dense("c1", "A1", Wc1)
    net.dense("c2", "ind", Wc2)
    net.add("V2", ["c1", "c2"])
    NV2 = nA + nRm

    def v2(name):
        return A[name] if name in A else nA + I[name]

    # second AND stage: bar = bar1 OR bar2 ~ min(1, bar1 + bar2) -> use relu(bar1 + bar2) then clip later
    ands2 = [
        ("bar", ["bar1", "bar2"], 0.0),                      # OR (sum, clipped by the following ramp)
        ("nobar_hint", [], 0.0),
        ("yel_round", ["g_yel", "round"], -1.0),
        ("yel_inb", ["g_yel", "inbetween"], -1.0),
        ("yel_cigar", ["g_yel", "cigar"], -1.0),
        ("edge_bulge", ["g_edge", "bulge"], -1.0),
        ("edge_nobulge", ["g_edge"], 0.0),                   # minus bulge handled in scores
        ("blu_conc", ["g_blu", "conc_hi"], -1.0),
        ("blu_yelin", ["g_blu", "yellow_in"], -1.0),
        ("blu_clumpy", ["g_blu", "clumpy"], -1.0),
        ("blu_mild", ["g_blu", "mild_only"], -1.0),
        ("blu_bar", ["g_blu", "bar1"], -1.0),
        ("blu_bar2", ["g_blu", "bar2"], -1.0),
    ]
    nA2 = len(ands2)
    Wand2 = np.zeros((nA2, NV2))
    band2 = np.zeros(nA2)
    for i, (_, names, bias) in enumerate(ands2):
        for nm in names:
            Wand2[i, v2(nm)] = 1.0
        band2[i] = bias
    net.dense("andU2", "V2", Wand2, band2)
    net.relu("A2", "andU2")
    A2 = {name: i for i, (name, _, _) in enumerate(ands2)}

    # carry V2 forward too: V3 = [A2 (nA2), V2 (NV2)]
    Wd1 = np.vstack([np.eye(nA2), np.zeros((NV2, nA2))])
    Wd2 = np.vstack([np.zeros((nA2, NV2)), np.eye(NV2)])
    net.dense("d1", "A2", Wd1)
    net.dense("d2", "V2", Wd2)
    net.add("V3", ["d1", "d2"])

    def v3(name):
        return A2[name] if name in A2 else nA2 + v2(name)

    # ---- stage 8: class scores ----------------------------------------------------------------------------
    prior = np.array([0.061, 0.104, 0.149, 0.114, 0.019, 0.115, 0.103, 0.148, 0.080, 0.106])
    Wsc = np.zeros((10, nA2 + NV2))
    bsc = 0.02 * prior / prior.max()

    def w(cls, name, val):
        Wsc[cls, v3(name)] += val

    # edge-on
    w(8, "edge_nobulge", 1.0)
    w(8, "edge_bulge", -0.7)
    w(8, "blue", 0.25)
    w(9, "edge_nobulge", 0.55)
    w(9, "edge_bulge", 0.7)
    w(9, "yellow", 0.25)
    # yellow smooth group
    w(2, "yel_round", 1.0)
    w(3, "yel_inb", 1.0)
    w(4, "yel_cigar", 1.0)
    # mergers
    w(1, "g_mrg", 1.0)
    # blue group: barred / tight / loose / disturbed
    w(5, "g_blu", 0.30)
    w(5, "blu_bar", 0.7)
    w(5, "blu_bar2", 0.7)
    w(6, "g_blu", 0.40)
    w(6, "blu_conc", 0.35)
    w(6, "blu_yelin", 0.30)
    w(6, "blu_clumpy", -0.25)
    w(7, "g_blu", 0.50)
    w(7, "blu_clumpy", 0.25)
    w(7, "blu_conc", -0.20)
    w(7, "blu_yelin", -0.15)
    w(0, "g_blu", 0.10)
    w(0, "blu_mild", 0.55)
    w(0, "blu_clumpy", 0.15)
    net.dense("scores", "V3", Wsc, bsc)
    return net


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    net = build()
    net.save(out)
    total = sum(int(np.prod(v.shape)) for v in net.weights.values())
    print(f"wrote {len(net.layers)} layers, {total:,} weights to {out}")
