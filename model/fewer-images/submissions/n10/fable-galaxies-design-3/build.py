"""Hand-designed galaxy morphology network. Every number below comes from formulas and from general knowledge of
galaxy morphology and of the Galaxy10 DECaLS class definitions; nothing is read from, fitted to or measured on images.

Pipeline
  c1    5x5 Gaussian blur, stride 2 -> 64x64 maps: L = relu(luminance - sky), RED = relu(R-B - t), BLUE = relu(B-R - t)
  dq    centred quadrupole moments of L at 24 orientations (3 Gaussian windows, 5 rings, ring total); maxpool -> |Q|
  dd    centred dipole moments of L at 12 directions (2 windows); maxpool -> |D|
  ds    scalar moments: aperture fluxes, second radial moments, colour sum
  ratio every feature is a ratio a/b; the pair is laid out as [a, b, -a, -b] in one channel and "norm" turns it into
        sqrt(2) a / sqrt(a^2 + b^2), a monotone function of a/b that does not depend on the galaxy's brightness
  hid   soft steps (pairs of relu hinges) on the ratios; out = hand-written additive rule table + log class priors
"""
import sys, os, json
import numpy as np


def nrm(rho):
    """value the norm layer gives for a ratio a/b = rho"""
    return np.sqrt(2.0) * rho / np.sqrt(1.0 + rho * rho)


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    W, layers = {}, []

    # ---------------------------------------------------------------- c1: blur, downsample, sky / colour thresholds
    T_LUM, T_COL = 0.15, 0.04           # sky is "near black": keep only pixels clearly above it
    ax = np.arange(5) - 2.0
    g = np.exp(-ax ** 2 / 2.0)
    g2 = np.outer(g, g); g2 /= g2.sum()
    w = np.zeros((3, 3, 5, 5)); b = np.zeros(3)
    w[0, :] = g2 / 3.0; b[0] = -T_LUM                   # luminance above sky
    w[1, 0] = g2; w[1, 2] = -g2; b[1] = -T_COL          # red excess  (z band over g band)
    w[2, 0] = -g2; w[2, 2] = g2; b[2] = -T_COL          # blue excess (g band over z band)
    W["c1.W"], W["c1.b"] = w, b
    layers.append({"name": "c1", "op": "conv", "inputs": ["image"], "out": 3, "kernel": [5, 5], "stride": 2, "padding": 2})
    layers.append({"name": "r1", "op": "relu", "inputs": ["c1"]})

    # ---------------------------------------------------------------- geometry of the 64x64 maps (1 px = 1.04 arcsec)
    N, c = 64, 31.75                    # output pixel i sits at input pixel 2i; the image centre 63.5 -> 31.75
    yy, xx = np.meshgrid(np.arange(N) - c, np.arange(N) - c, indexing="ij")
    r2 = xx ** 2 + yy ** 2; r = np.sqrt(r2); phi = np.arctan2(yy, xx)

    def row(m, ch=0):
        v = np.zeros((3, N, N)); v[ch] = m
        return v.reshape(-1)

    def win(s):
        return np.exp(-r2 / (2.0 * s * s))

    # ---------------------------------------------------------------- dq: quadrupoles at 24 orientations
    NQ = 24
    thetas = np.arange(NQ) * np.pi / NQ
    ring_r, ring_d = [2.5, 5.0, 7.5, 10.0, 12.5], 1.25
    rings = [np.exp(-(r - rr) ** 2 / (2.0 * ring_d ** 2)) for rr in ring_r]
    ring_tot = sum(rings)
    qwin = [win(4.0) * r2 / 16.0, win(8.0) * r2 / 64.0, win(16.0) * r2 / 256.0] + [m * r2 / 64.0 for m in rings] + [ring_tot * r2 / 64.0]
    NG = len(qwin)                      # 9 groups
    wq = np.zeros((NG * NQ, 3 * N * N))
    for gi, m in enumerate(qwin):
        for k, th in enumerate(thetas):
            wq[gi * NQ + k] = row(m * np.cos(2.0 * (phi - th)))
    W["dq.W"], W["dq.b"] = wq, np.zeros(NG * NQ)
    layers.append({"name": "dq", "op": "dense", "inputs": ["r1"], "out": NG * NQ})
    layers.append({"name": "dq_r", "op": "reshape", "inputs": ["dq"], "shape": [NG, 1, NQ]})
    layers.append({"name": "dq_m", "op": "maxpool", "inputs": ["dq_r"], "size": [1, NQ], "stride": [1, NQ]})

    # ---------------------------------------------------------------- dd: dipoles at 12 directions
    ND = 12
    psis = np.arange(ND) * 2.0 * np.pi / ND
    dwin = [win(8.0) * r / 8.0, win(16.0) * r / 16.0]
    wd = np.zeros((2 * ND, 3 * N * N))
    for gi, m in enumerate(dwin):
        for k, ps in enumerate(psis):
            wd[gi * ND + k] = row(m * np.cos(phi - ps))
    W["dd.W"], W["dd.b"] = wd, np.zeros(2 * ND)
    layers.append({"name": "dd", "op": "dense", "inputs": ["r1"], "out": 2 * ND})
    layers.append({"name": "dd_r", "op": "reshape", "inputs": ["dd"], "shape": [2, 1, ND]})
    layers.append({"name": "dd_m", "op": "maxpool", "inputs": ["dd_r"], "size": [1, ND], "stride": [1, ND]})

    # ---------------------------------------------------------------- ds: scalar moments
    S_M2, S_M4, S_M8, S_M16, S_R4, S_R8, S_R16, S_COL, S_RTOT = range(9)
    ws = np.zeros((9, 3 * N * N))
    ws[S_M2] = row(win(2.0)); ws[S_M4] = row(win(4.0)); ws[S_M8] = row(win(8.0)); ws[S_M16] = row(win(16.0))
    ws[S_R4] = row(win(4.0) * r2 / 16.0); ws[S_R8] = row(win(8.0) * r2 / 64.0); ws[S_R16] = row(win(16.0) * r2 / 256.0)
    ws[S_COL] = row(win(8.0), 1) - row(win(8.0), 2)
    ws[S_RTOT] = row(ring_tot * r2 / 64.0)
    W["ds.W"], W["ds.b"] = ws, np.zeros(9)
    layers.append({"name": "ds", "op": "dense", "inputs": ["r1"], "out": 9})

    # ---------------------------------------------------------------- ratios a/b laid out as [a, b, -a, -b]
    J_E4, J_E8, J_E16, J_A8, J_A16, J_C1, J_C3, J_COL, J_TW = range(9)
    NJ = 9
    eq = np.zeros((NJ * 4, NG)); ed = np.zeros((NJ * 4, 2)); es = np.zeros((NJ * 4, 9))

    def put(mat, j, pos, src, coef=1.0):   # pos 0: numerator, 1: denominator
        mat[j * 4 + pos, src] += coef
        mat[j * 4 + pos + 2, src] -= coef

    put(eq, J_E4, 0, 0); put(es, J_E4, 1, S_R4)             # ellipticity, 4 px window
    put(eq, J_E8, 0, 1); put(es, J_E8, 1, S_R8)             # ellipticity, 8 px window
    put(eq, J_E16, 0, 2); put(es, J_E16, 1, S_R16)          # ellipticity, 16 px window
    put(ed, J_A8, 0, 0); put(es, J_A8, 1, S_M8)             # lopsidedness, 8 px window
    put(ed, J_A16, 0, 1); put(es, J_A16, 1, S_M16)          # lopsidedness, 16 px window
    put(es, J_C1, 0, S_M2); put(es, J_C1, 1, S_M4)          # core concentration
    put(es, J_C3, 0, S_M8); put(es, J_C3, 1, S_M16)         # compactness (small galaxy -> 1, large -> 0.25)
    put(es, J_COL, 0, S_COL); put(es, J_COL, 1, S_M8)       # colour: red minus blue excess per unit light
    for gi in range(3, 8):
        put(eq, J_TW, 0, gi)                                # sum of ring quadrupole amplitudes ...
    put(eq, J_TW, 0, 8, -1.0)                               # ... minus amplitude of their vector sum = position-angle twist
    put(es, J_TW, 1, S_RTOT)
    W["eq.W"], W["eq.b"] = eq, np.zeros(NJ * 4)
    W["ed.W"], W["ed.b"] = ed, np.zeros(NJ * 4)
    W["es.W"], W["es.b"] = es, np.zeros(NJ * 4)
    layers.append({"name": "eq", "op": "dense", "inputs": ["dq_m"], "out": NJ * 4})
    layers.append({"name": "ed", "op": "dense", "inputs": ["dd_m"], "out": NJ * 4})
    layers.append({"name": "es", "op": "dense", "inputs": ["ds"], "out": NJ * 4})
    layers.append({"name": "cat", "op": "add", "inputs": ["eq", "ed", "es"]})
    layers.append({"name": "pairs", "op": "reshape", "inputs": ["cat"], "shape": [NJ, 4, 1]})
    layers.append({"name": "ratio", "op": "norm", "inputs": ["pairs"]})

    # ---------------------------------------------------------------- rule table: soft steps and class weights
    # classes: 0 disturbed, 1 merging, 2 round, 3 in-between, 4 cigar, 5 barred, 6 tight spiral, 7 loose spiral,
    #          8 edge-on no bulge, 9 edge-on bulge
    # ellipticity e = (1-q^2)/(1+q^2) for axis ratio q; region scores for e8 (round / mildly / clearly / extremely elongated)
    A = np.array([-0.5, -1.0, 2.0, -1.5, -4.0, 0.0, 0.3, 0.0, -5.0, -5.0])
    B = np.array([0.0, 0.0, -2.5, 1.5, -1.5, 0.3, 0.3, 0.3, -3.0, -2.5])
    C = np.array([0.2, 0.5, -5.0, 0.0, 2.5, -0.3, -0.5, -0.5, 0.0, 0.5])
    D = np.array([-1.0, -1.0, -6.0, -4.0, 0.5, -3.0, -3.0, -3.0, 3.0, 2.5])
    steps = []  # (coefficients over the 9 ratios, n_lo, n_hi, class weights)

    def one(j, sign=1.0):
        v = np.zeros(NJ); v[j] = sign
        return v

    steps.append((one(J_E8), nrm(0.08), nrm(0.18), B - A))
    steps.append((one(J_E8), nrm(0.40), nrm(0.60), C - B))
    steps.append((one(J_E8), nrm(0.68), nrm(0.84), D - C))
    # inner region as flat as the whole: a thin disc with no bulge
    steps.append((one(J_E4), nrm(0.55), nrm(0.75), np.array([0, 0, 0, 0, -0.5, 0, 0, 0, 1.5, -0.8])))
    # inner region more elongated than the outer: a bar inside a rounder disc
    steps.append((one(J_E4) - one(J_E8), 0.03, 0.20, np.array([0, 0.3, 0, 0, 0, 1.0, 0, 0, -0.3, -0.5])))
    # elongated on the largest scale too
    steps.append((one(J_E16), nrm(0.60), nrm(0.80), np.array([-0.3, 0, -1.0, -0.7, -0.3, -0.5, -0.5, -0.5, 0.7, 0.7])))
    # bright compact core (bulge / elliptical) against flat low-surface-brightness discs
    steps.append((one(J_C1), nrm(0.40), nrm(0.60), np.array([-0.3, 0, 1.0, 0.8, 0.3, 0, 0.3, -1.2, -0.8, 0.8])))
    # small on the sky: featureless-looking galaxies; spirals and edge-on discs are picked when well resolved
    steps.append((one(J_C3), nrm(0.65), nrm(0.85), np.array([-0.5, -0.5, 0.8, 0.5, 0.3, -0.5, -0.2, -0.8, -0.6, -0.3])))
    # red (old stars: ellipticals, bulges) and blue (star-forming discs)
    steps.append((one(J_COL), nrm(0.08), nrm(0.28), np.array([-0.3, 0, 0.8, 0.8, 0.3, -0.2, -0.4, -1.0, -0.8, 0.6])))
    steps.append((one(J_COL, -1.0), nrm(0.0), nrm(0.15), np.array([0.6, 0.3, -1.2, -1.0, -0.3, 0.3, 0.6, 1.2, 0.9, -0.8])))
    # lopsided light: second nucleus, tidal features
    steps.append((one(J_A8), nrm(0.10), nrm(0.25), np.array([1.2, 2.0, -1.0, -0.7, -0.5, -0.2, -0.5, 0.3, -0.5, -0.7])))
    steps.append((one(J_A16), nrm(0.12), nrm(0.30), np.array([0.6, 1.0, -0.3, -0.2, -0.2, -0.1, -0.2, 0.2, -0.2, -0.2])))
    # position angle twisting with radius: spiral arms, bars, irregular structure
    steps.append((one(J_TW), nrm(0.03), nrm(0.10), np.array([0.8, 0.6, -1.0, -0.8, -0.8, 1.0, 0.8, 1.2, -0.8, -0.8])))

    H = 2 * len(steps)
    wh = np.zeros((H, NJ * 4)); bh = np.zeros(H); wo = np.zeros((10, H))
    for i, (coef, lo, hi, cw) in enumerate(steps):
        d = hi - lo
        for u, edge, sgn in ((2 * i, lo, 1.0), (2 * i + 1, hi, -1.0)):
            wh[u, np.arange(NJ) * 4] = coef / d       # position 0 of each channel holds the normalised ratio
            bh[u] = -edge / d
            wo[:, u] = sgn * cw
    # class frequencies of the published Galaxy10 DECaLS catalogue (general knowledge of the data set)
    counts = np.array([1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873], dtype=float)
    bo = np.log(counts / counts.sum()) + A
    W["hid.W"], W["hid.b"] = wh, bh
    W["out.W"], W["out.b"] = wo, bo
    layers.append({"name": "hid", "op": "dense", "inputs": ["ratio"], "out": H})
    layers.append({"name": "hid_r", "op": "relu", "inputs": ["hid"]})
    layers.append({"name": "out", "op": "dense", "inputs": ["hid_r"], "out": 10})

    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
