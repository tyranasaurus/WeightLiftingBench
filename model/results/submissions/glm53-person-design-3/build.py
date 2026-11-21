#!/usr/bin/env python3
"""Builds a hand-designed person/no-person classifier network (no training).

Pipeline, all weights typed from general knowledge of what people look like
in everyday photos:
  1. colour channels L, U=R-G, V=(R+G)/2-B, Gaussian-smoothed for noise
     robustness (JPEG artefacts must not eat the chroma tolerances)
  2. soft skin mask: L1-ball membership around typical skin chroma
  3. "solid" skin gate, then subtract "rough" skin (skin-coloured pixels with
     high local gradient = wood/straw/brick/food texture) -> smooth skin
  4. multi-scale density grids (6/12/24 px cells) + centre-surround blobs
  5. eye-pair detector at two scales: two small dark spots with a LIGHT gap
     between them, surrounded by smooth skin (eyes+face).  The light-gap term
     kills big single holes (food), dark lines (grill marks) and shadows.
  6. head cue: an isolated smooth-skin blob (local 30 px context mostly
     non-skin colour, few other blobs nearby) with a dark mass directly above
     (hair) or below (clothes/neck shadow)
  7. vertical-edge column concentration (standing figure cue, via norm)
  8. scalar features -> dense(2) readout

Every constant comes from general knowledge (skin chroma ranges, eye spacing
~ 0.4 x face width, faces smooth, heads sit on darker bodies), not from
statistics of the training images.
"""
import sys, json
import numpy as np
from pathlib import Path

# ---------------- tunable constants (hand-chosen) ----------------
SK = dict(u0=0.12, du=0.14, v0=0.13, dv=0.13, l0=0.55, dl=0.32, ball=2.3)
DARK_T = 0.45                 # brightness below which a pixel is dark
SOLID_GAIN, SOLID_B = 3.0, 1.05
ROUGH_GAIN = 6.0              # gradient level that counts as textured
EYE_LR_B = 0.3                # per-eye "darker than bridge" bias
EYE_CAP = 1.2
EYE_SKIN_GAIN, EYE_SKIN_B = 2.6, 1.25   # smooth-skin-surround gate
EYE_SKIN_CAP = 0.8            # cap so surround alone can never fire the gate
EYE_GATE = 1.35
CS_B = 0.04                   # centre-surround bias for blob detectors
SHIFT_GAIN = 2.4              # dark-above / dark-below gain on 6px grid
BLOB_CELL_T = 0.15            # 6px cell counts as a blob above this
SPARSE_SLOPE = 2.5            # local blob density slope for sparse gate
VCOL_T = 1.5                  # z threshold for hot edge columns

W_PERSON = np.array([0.15, 0.2, 0.5, 0.2, 2.6, 0.3, 0.3, 0.2, 0.0, -3.0,
                     0.1, 0.1, 0.2, -1.5, -1.0, 1.5, 0.6, 1.0, -0.7], np.float64)
B_PERSON = -0.55
HEADMEAN_GAIN = 8.0            # gain for the mean (group) head cue

# ---------------- kernel helpers ----------------
def box(k, inner):
    W = np.zeros((k, k), np.float64)
    o = (k - inner) // 2
    W[o:o + inner, o:o + inner] = 1.0 / (inner * inner)
    return W

def gauss5(sigma=1.2):
    y, x = np.mgrid[0:5, 0:5]
    G = np.exp(-((y - 2) ** 2 + (x - 2) ** 2) / (2.0 * sigma * sigma))
    return G / G.sum()

def sobel(axis):
    S = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64) / 4.0
    return S if axis == "x" else S.T

def _unused_eyepair_kernel(k, inner, gain):
    """Two dark boxes with a lighter box between them (eye pair)."""
    W = np.zeros((k, k), np.float64)
    r0 = (k - inner) // 2
    rows = slice(r0, r0 + inner)
    L = slice(0, inner); M = slice(r0 + 1, r0 + 1 + inner); R = slice(k - inner, k)
    W[rows, L] += gain / (inner * inner)
    W[rows, R] += gain / (inner * inner)
    W[rows, M] -= 2.0 * gain / (inner * inner)
    return W

# ---------------- build ----------------
def main(outdir):
    L, W = [], {}
    def conv(name, inp, Wv, bv, k=1, pad=0):
        Wv = np.asarray(Wv, np.float32)
        bv = np.atleast_1d(np.asarray(bv, np.float32))
        assert Wv.ndim == 4 and Wv.shape[2] == Wv.shape[3] == k, (name, Wv.shape)
        L.append(dict(name=name, op="conv", inputs=[inp], out=int(Wv.shape[0]),
                      kernel=[k, k], stride=1, padding=pad))
        W[name + ".W"], W[name + ".b"] = Wv, bv
    def conv1(name, inp, w, b):
        conv(name, inp, np.asarray(w, np.float32).reshape(1, 1, 1, 1),
             np.float32([b]))
    def relu(name, inp): L.append(dict(name=name, op="relu", inputs=[inp]))
    def add(name, *ins): L.append(dict(name=name, op="add", inputs=list(ins)))
    def pool(name, inp, kind, size, stride):
        L.append(dict(name=name, op=kind, inputs=[inp], size=list(size), stride=list(stride)))
    def norm(name, inp): L.append(dict(name=name, op="norm", inputs=[inp]))
    def minlayer(tag, a, b):
        """min(a,b) via a - relu(a-b); a,b are same-shape layers."""
        conv1(tag + "_nb", b, -1, 0)
        add(tag + "_ab", a, tag + "_nb")
        relu(tag + "_r", tag + "_ab")
        conv1(tag + "_nr", tag + "_r", -1, 0)
        add(tag + "_m", a, tag + "_nr")
    def caplayer(tag, a, cap):
        """min(a, cap) for a >= 0."""
        conv1(tag + "_c", a, 1, -cap); relu(tag + "_cr", tag + "_c")
        conv1(tag + "_cn", tag + "_cr", -1, 0)
        add(tag + "_cap", a, tag + "_cn")

    # --- 1. colour channels and smoothed copy ---
    colW = np.zeros((3, 3, 1, 1), np.float32)
    colW[0, :, 0, 0] = [1/3., 1/3., 1/3.]
    colW[1, :, 0, 0] = [1., -1., 0.]
    colW[2, :, 0, 0] = [.5, .5, -1.]
    conv("col", "image", colW, np.zeros(3, np.float32))
    G = gauss5(1.2)
    colSW = np.zeros((3, 3, 5, 5), np.float32)
    for o in range(3): colSW[o, o] = G
    conv("cols", "col", colSW, np.zeros(3, np.float32), k=5, pad=2)

    # --- 2. soft skin mask on smoothed chroma ---
    u0, du, v0, dv, l0, dl, ball = (SK[k] for k in ("u0","du","v0","dv","l0","dl","ball"))
    ad1W = np.zeros((6, 3, 1, 1), np.float32)
    ad1W[0, 1], ad1W[1, 1] = 1/du, -1/du
    ad1W[2, 2], ad1W[3, 2] = 1/dv, -1/dv
    ad1W[4, 0], ad1W[5, 0] = 1/dl, -1/dl
    ad1b = np.float32([ -u0/du, u0/du, -v0/dv, v0/dv, -l0/dl, l0/dl ])
    conv("ad1", "cols", ad1W, ad1b); relu("ad1r", "ad1")
    conv("dist", "ad1r", -np.ones((1, 6, 1, 1), np.float32), np.float32(ball))
    relu("skin", "dist")                       # [0, ball]
    conv("skins", "skin", (gauss5(1.2) / ball).reshape(1, 1, 5, 5).astype(np.float32),
         np.float32(0), k=5, pad=2)            # ~[0,1]

    # --- 3. solid / rough / smooth skin ---
    conv1("sol3", "skins", SOLID_GAIN, -SOLID_B); relu("solid", "sol3")
    gx = np.zeros((1, 3, 3, 3), np.float64); gx[0, 0] = sobel("x")
    gy = np.zeros((1, 3, 3, 3), np.float64); gy[0, 0] = sobel("y")
    conv("gxs", "cols", gx.astype(np.float32), np.float32(0), k=3, pad=1)
    conv("gys", "cols", gy.astype(np.float32), np.float32(0), k=3, pad=1)
    conv1("gxn", "gxs", -1, 0); relu("axp", "gxs"); relu("axn", "gxn"); add("amx", "axp", "axn")
    conv1("gyn", "gys", -1, 0); relu("ayp", "gys"); relu("ayn", "gyn"); add("amy", "ayp", "ayn")
    add("gmag", "amx", "amy")
    conv1("ng3", "gmag", -ROUGH_GAIN, 0)
    add("amb", "solid", "ng3"); relu("ramb", "amb")
    conv1("nramb", "ramb", -1, 0); add("rough", "solid", "nramb")   # min(solid, g*ROUGH)
    conv1("nrough", "rough", -1, 0); add("smooth", "solid", "nrough")

    # --- 4. grids and centre-surround blobs ---
    pool("s6", "smooth", "avgpool", [6, 6], [6, 6])
    pool("s12", "smooth", "avgpool", [12, 12], [12, 12])
    pool("s24", "smooth", "avgpool", [24, 24], [24, 24])
    pool("s96", "smooth", "avgpool", [96, 96], [96, 96])
    def csk():
        K = np.full((3, 3), -1.0 / 8); K[1, 1] = 1.0
        return K.reshape(1, 1, 3, 3).astype(np.float32)
    for nm, src, kmx in (("cs6", "s6", 16), ("cs12", "s12", 8), ("cs24", "s24", 4)):
        conv(nm, src, csk(), np.float32(-CS_B), k=3, pad=1)
        relu(nm + "r", nm)
        pool("b" + nm[2:] + "max", nm + "r", "maxpool", [kmx, kmx], [kmx, kmx])

    # --- 5. dark map + eye-pair cue ---
    darkW = np.zeros((1, 3, 1, 1), np.float32); darkW[0, 0, 0, 0] = -1.0
    conv("darkc", "cols", darkW, np.float32(DARK_T))
    relu("dark", "darkc")                      # [0, 0.45]

    def eye_scale(tag, k, inner, gain):
        # left eye darker than bridge AND right eye darker than bridge
        for side, colsl in (("L", (0, inner)), ("R", (k - inner, k))):
            KL = np.zeros((k, k), np.float64)
            r0 = (k - inner) // 2
            rows = slice(r0, r0 + inner)
            KL[rows, colsl[0]:colsl[1]] = gain / (inner * inner)
            m0 = r0 + 1
            KL[rows, m0:m0 + inner] = -gain / (inner * inner)
            conv("eye" + side + tag, "dark", KL.reshape(1, 1, k, k).astype(np.float32),
                 np.float32(-EYE_LR_B), k=k, pad=k // 2)
            relu("eye" + side + tag + "r", "eye" + side + tag)
        minlayer("eyemin" + tag, "eyeL" + tag + "r", "eyeR" + tag + "r")
        caplayer("eyecap" + tag, "eyemin" + tag + "_m", EYE_CAP)
        conv("eyeB" + tag, "smooth", (box(k, k) * EYE_SKIN_GAIN).reshape(1, 1, k, k).astype(np.float32),
             np.float32(-EYE_SKIN_B), k=k, pad=k // 2)
        relu("eyeBr" + tag, "eyeB" + tag)
        caplayer("eyebcap" + tag, "eyeBr" + tag, EYE_SKIN_CAP)
        add("eyep" + tag, "eyecap" + tag + "_cap", "eyebcap" + tag + "_cap")
        conv1("eyeg" + tag, "eyep" + tag, 1, -EYE_GATE)
        relu("eyegr" + tag, "eyeg" + tag)
        pool("eyem8" + tag, "eyegr" + tag, "maxpool", [8, 8], [8, 8])   # (1,12,12)
    eye_scale("1", 9, 3, 8.0)    # ~12-20 px faces
    eye_scale("2", 17, 5, 12.0)  # ~25-45 px faces
    add("eyesum", "eyem81", "eyem82")
    pool("fhole", "eyesum", "maxpool", [12, 12], [12, 12])

    # --- 6. head cue: isolated skin blob with dark above or below ---
    pool("db6", "dark", "avgpool", [6, 6], [6, 6])            # (1,16,16)
    for tag, dy in (("l", 2), ("a", 0)):                      # below / above
        K = np.zeros((3, 3), np.float32); K[dy, 1] = 1.0
        conv("db" + tag, "db6", K.reshape(1, 1, 3, 3), np.float32(0), k=3, pad=1)
        conv1("db" + tag + "3", "db" + tag, SHIFT_GAIN, 0)
    # fhs / fha (ungated head-shoulders, low weight)
    add("hspre", "cs6r", "dbl3"); conv1("hsc", "hspre", 1, -0.70); relu("hsr", "hsc")
    pool("fhs", "hsr", "maxpool", [16, 16], [16, 16])
    add("hapre", "cs6r", "dba3"); conv1("hac", "hapre", 1, -0.65); relu("har", "hac")
    pool("fha", "har", "maxpool", [16, 16], [16, 16])
    # isolated blob: local 30px context mostly non-skin colour, few blobs nearby
    pool("sk6", "skins", "avgpool", [6, 6], [6, 6])
    conv("ctx5", "sk6", np.full((1, 1, 5, 5), 1/25., np.float32), np.float32(0), k=5, pad=2)
    conv1("nsc", "ctx5", -2.5, 1.0); relu("nsr", "nsc")
    conv1("b6c", "cs6r", 1, -BLOB_CELL_T); relu("b6cr", "b6c")
    pool("fbcount", "b6cr", "avgpool", [16, 16], [16, 16])
    conv("spc", "b6cr", np.full((1, 1, 5, 5), 1/25., np.float32), np.float32(0), k=5, pad=2)
    conv1("spg", "spc", -SPARSE_SLOPE, 1.0); relu("spgr", "spg")
    conv1("c2", "cs6r", 3, 0)
    minlayer("m_ns", "c2", "nsr")                # min(3*cs6r, ns)
    minlayer("m_sp", "m_ns_m", "spgr")           # min(.., sparse)
    # dark above OR below the blob: max(dba3, dbl3) then min with isolation
    add("ab_sum", "dba3", "dbl3")
    minlayer("m_ab", "dba3", "dbl3")
    conv1("n_mab", "m_ab_m", -1, 0)
    add("ab_max", "ab_sum", "n_mab")             # max(above, below)
    minlayer("m_head", "m_sp_m", "ab_max")       # head cue per cell
    pool("headm8", "m_head_m", "maxpool", [8, 8], [8, 8])
    pool("fstripe", "headm8", "maxpool", [2, 2], [2, 2])
    conv1("headmn", "m_head_m", HEADMEAN_GAIN, 0)
    pool("fheadmean", "headmn", "avgpool", [16, 16], [16, 16])
    # vertical limb stripe: narrow (vs left/right) skin with vertical continuity
    HK = np.zeros((3, 3), np.float64)
    HK[1, 0], HK[1, 1], HK[1, 2] = -0.5, 1.0, -0.5
    conv("hcsk", "s6", HK.reshape(1, 1, 3, 3).astype(np.float32), np.float32(-0.06), k=3, pad=1)
    relu("hcsr", "hcsk")
    VK = np.zeros((3, 3), np.float64)
    VK[0, 1] = VK[1, 1] = VK[2, 1] = 1/3.
    conv("vvk", "s6", VK.reshape(1, 1, 3, 3).astype(np.float32), np.float32(-0.25), k=3, pad=1)
    relu("vvr", "vvk")
    conv1("hcs3", "hcsr", 1.5, 0)
    minlayer("m_limb", "hcs3", "vvr")
    pool("flimb", "m_limb_m", "maxpool", [16, 16], [16, 16])

    # --- 6b. blue/bright dominance (sky, water, seascape -> no person) ---
    lW = np.zeros((1, 3, 1, 1), np.float32); lW[0, 0, 0, 0] = 1.0
    vW = np.zeros((1, 3, 1, 1), np.float32); vW[0, 2, 0, 0] = 1.0
    conv("Lc", "cols", lW, np.float32(0)); conv("Vc", "cols", vW, np.float32(0))
    pool("L12", "Lc", "avgpool", [12, 12], [12, 12])
    pool("V12", "Vc", "avgpool", [12, 12], [12, 12])
    conv1("bl1", "V12", -4.0, 0.4); relu("bl1r", "bl1")     # bluish: V < 0.1
    conv1("bl2", "L12", 2.5, -1.25); relu("bl2r", "bl2")    # bright: L > 0.5
    minlayer("m_blue", "bl1r", "bl2r")
    pool("fblue", "m_blue_m", "avgpool", [8, 8], [8, 8])

    # --- 7. vertical edge column concentration ---
    pool("colE", "amx", "avgpool", [96, 1], [96, 1])
    norm("cz", "colE")
    conv1("czt", "cz", 1, -VCOL_T); relu("chot", "czt")
    pool("fvconc", "chot", "avgpool", [1, 96], [1, 96])

    # --- 8. remaining scalars ---
    pool("fedge", "gmag", "avgpool", [96, 96], [96, 96])
    pool("frough", "rough", "avgpool", [96, 96], [96, 96])
    pool("fskinraw", "skins", "avgpool", [96, 96], [96, 96])
    CK = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64) / 16.0
    conv("centk", "s12", CK.reshape(1, 1, 3, 3).astype(np.float32), np.float32(0), k=3, pad=1)
    pool("fcent", "centk", "avgpool", [8, 8], [8, 8])
    UD = np.zeros((8, 8), np.float32)
    UD[:4] = 1/32.; UD[4:] = -1/32.
    conv("fuld", "s12", UD.reshape(1, 1, 8, 8), np.float32(0), k=8, pad=0)
    conv1("foverc", "fskinraw", 2.5, -1.125); relu("foverr", "foverc")
    caplayer("fovercap", "foverr", 1.2)

    # --- 9. assemble and read out ---
    sources = ["s96", "b6max", "b12max", "b24max", "fhole", "fhs", "fha",
               "fvconc", "fedge", "frough", "fcent", "fskinraw", "fuld",
               "fovercap_cap", "fbcount", "fstripe", "flimb", "fheadmean",
               "fblue"]
    n = len(sources)
    for i, src in enumerate(sources):
        Wv = np.zeros((n, 1, 1, 1), np.float32); Wv[i, 0, 0, 0] = 1.0
        conv("asm" + str(i), src, Wv, np.zeros(n, np.float32))
    add("feat", *["asm" + str(i) for i in range(n)])
    dW = np.zeros((2, n), np.float32)
    dW[1] = W_PERSON[:n]
    L.append(dict(name="out", op="dense", inputs=["feat"], out=2))
    W["out.W"], W["out.b"] = dW, np.float32([0.0, B_PERSON])

    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "network.json").write_text(json.dumps({"layers": L}))
    np.savez(out / "weights.npz", **{k: v.astype(np.float32) for k, v in W.items()})
    print(f"wrote {len(L)} layers, {sum(int(np.prod(v.shape)) for v in W.values())} weights")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
