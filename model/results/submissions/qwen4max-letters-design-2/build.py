"""Hand-designed ASL fingerspelling network.  Every weight comes from a formula or
from a hand-written geometric model of the 24 static ASL letters.  No data is read.

Usage: python3 build.py OUT_DIR
"""
import json, os, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

K = 11
C = (K - 1) // 2
yy, xx = np.mgrid[0:K, 0:K].astype(np.float64)
U = xx - C
V = yy - C

LAYERS = []
W = {}


def layer(name, op, inputs, **settings):
    L = {"name": name, "op": op, "inputs": inputs}
    L.update(settings)
    LAYERS.append(L)
    return name


def put(name, w, b):
    W[name + ".W"] = np.asarray(w, np.float32)
    W[name + ".b"] = np.asarray(b, np.float32)


def pq(theta_deg):
    t = np.deg2rad(theta_deg)
    return U * np.cos(t) + V * np.sin(t), -U * np.sin(t) + V * np.cos(t)


def edge_filter(theta, sigma=1.25):
    """derivative-of-Gaussian edge filter, unit response on an ideal unit step."""
    p, q = pq(theta)
    f = -p / sigma ** 2 * np.exp(-(p ** 2 + q ** 2) / (2 * sigma ** 2))
    return f / f[f > 0].sum()


def ridge_filter(theta, sc=1.0, ss=2.0, sq=4.0, width=3.0):
    """elongated difference-of-Gaussians line detector; unit response on an ideal
    infinite stripe of `width` pixels and contrast 1 (analytic target, no data)."""
    p, q = pq(theta)
    f = (np.exp(-p ** 2 / (2 * sc ** 2)) - (sc / ss) * np.exp(-p ** 2 / (2 * ss ** 2))) \
        * np.exp(-q ** 2 / (2 * sq ** 2))
    f -= f.mean()                      # exact zero sum despite kernel truncation
    target = (np.abs(p) <= width / 2.0).astype(np.float64)
    return f / (f * target).sum()


def blob_filter(sc=2.0, ss=4.0, radius=3.0):
    f = np.exp(-(U ** 2 + V ** 2) / (2 * sc ** 2)) \
        - (sc / ss) ** 2 * np.exp(-(U ** 2 + V ** 2) / (2 * ss ** 2))
    f -= f.mean()
    target = (np.sqrt(U ** 2 + V ** 2) <= radius).astype(np.float64)
    return f / (f * target).sum()


def smooth_filter(s=1.5):
    g = np.exp(-(U ** 2 + V ** 2) / (2 * s * s))
    return g / g.sum()


# ---------------------------------------------------------------- filter bank
# channel pairs (2i, 2i+1) are +/- of the same filter, rectified and summed -> |response|
BANK = []
for th in (0, 90, 45, 135):
    BANK.append(("edge%d" % th, edge_filter(th, 1.25)))
for th in (0, 90, 45, 135):
    BANK.append(("ridge%d" % th, ridge_filter(th, 1.0, 2.0, 4.0, 3.0)))
for th in (0, 90):
    BANK.append(("gap%d" % th, ridge_filter(th, 2.2, 4.4, 4.0, 6.0)))
BANK.append(("blob", blob_filter(2.0, 4.0, 3.0)))

NPAIR = len(BANK)                     # 11 absolute-response channels
SM = smooth_filter(1.5)               # then bright / dark half-maps

f1w = np.zeros((2 * NPAIR + 2, 1, K, K))
for i, (_, f) in enumerate(BANK):
    f1w[2 * i, 0] = f
    f1w[2 * i + 1, 0] = -f
f1w[2 * NPAIR, 0] = SM
f1w[2 * NPAIR + 1, 0] = -SM

layer("n0", "norm", ["image"])
layer("f1", "conv", ["n0"], out=2 * NPAIR + 2, kernel=[K, K], stride=1, padding=C)
put("f1", f1w, np.zeros(2 * NPAIR + 2))
layer("r1", "relu", ["f1"])
mix = np.zeros((NPAIR + 2, 2 * NPAIR + 2, 1, 1))
for i in range(NPAIR):
    mix[i, 2 * i, 0, 0] = 1.0
    mix[i, 2 * i + 1, 0, 0] = 1.0
mix[NPAIR, 2 * NPAIR, 0, 0] = 1.0
mix[NPAIR + 1, 2 * NPAIR + 1, 0, 0] = 1.0
layer("a1", "conv", ["r1"], out=NPAIR + 2, kernel=[1, 1], stride=1, padding=0)
put("a1", mix, np.zeros(NPAIR + 2))

# absolute-response channel indices
AV, AH, AD, AE, RV, RH, RD, RE, GV, GH, BL, BR, DK = range(13)

# ---------------------------------------------------------------- pooling views
# four row bands (rows 0-6, 7-13, 14-20, 21-27) keeping every column
layer("bv", "maxpool", ["a1"], size=[7, 1], stride=[7, 1])    # (13,4,28)
# four column bands (cols 0-6, 7-13, 14-20, 21-27) keeping every row
layer("bh", "maxpool", ["a1"], size=[1, 7], stride=[1, 7])    # (13,28,4)
layer("g7", "avgpool", ["a1"], size=[4, 4], stride=[4, 4])    # (13,7,7)
layer("g4", "maxpool", ["a1"], size=[7, 7], stride=[7, 7])    # (13,4,4)

VIEW_SHAPE = {"bv": (13, 4, 28), "bh": (13, 28, 4), "g7": (13, 7, 7), "g4": (13, 4, 4)}
FEATS = []
FC = {v: [] for v in VIEW_SHAPE}


def feat(name, view, coef):
    FC[view].append((len(FEATS), np.asarray(coef, np.float64).reshape(VIEW_SHAPE[view])))
    FEATS.append(name)


def zeros(view):
    return np.zeros(VIEW_SHAPE[view])


xs = np.arange(28) - 13.5
PROF = (("tot", np.ones(28)), ("l", (xs < -3.5)), ("c", (np.abs(xs) <= 3.5)),
        ("r", (xs > 3.5)), ("fl", (xs < -6.5)), ("fr", (xs > 6.5)),
        ("m1", xs), ("m2", xs ** 2))
REG7 = {"TL": (0, 3, 0, 3), "TC": (0, 3, 2, 5), "TR": (0, 3, 4, 7),
        "ML": (2, 5, 0, 3), "MC": (2, 5, 2, 5), "MR": (2, 5, 4, 7),
        "BL": (4, 7, 0, 3), "BC": (4, 7, 2, 5), "BR": (4, 7, 4, 7),
        "AL": (0, 7, 0, 7),
        "T0": (0, 1, 0, 7), "T1": (1, 2, 0, 7), "B0": (6, 7, 0, 7),
        "L0": (0, 7, 0, 1), "R0": (0, 7, 6, 7),
        "T0L": (0, 1, 0, 3), "T0C": (0, 1, 2, 5), "T0R": (0, 1, 4, 7),
        "B0L": (6, 7, 0, 3), "B0C": (6, 7, 2, 5), "B0R": (6, 7, 4, 7)}
GCH = ((AV, "eV"), (AH, "eH"), (AD, "eD"), (AE, "eE"), (RV, "rV"), (RH, "rH"),
       (RD, "rD"), (RE, "rE"), (GV, "gV"), (GH, "gH"), (BL, "ob"))

# vertical structure, one profile per row band
for band in range(4):
    for ch, tag in ((RV, "r"), (GV, "g"), (AV, "e"), (BL, "o")):
        for nm, w in PROF:
            if tag == "o" and nm in ("m1", "m2", "fl", "fr"):
                continue
            c = zeros("bv"); c[ch, band, :] = w
            feat("v%d_%s_%s" % (band, tag, nm), "bv", c)

# horizontal structure, one profile per column band
for band in range(4):
    for ch, tag in ((RH, "r"), (GH, "g"), (AH, "e"), (BL, "o")):
        for nm, w in PROF:
            if tag == "o" and nm in ("m1", "m2", "ft", "fb"):
                continue
            nm2 = {"l": "t", "c": "m", "r": "b", "fl": "ft", "fr": "fb"}.get(nm, nm)
            c = zeros("bh"); c[ch, :, band] = w
            feat("h%d_%s_%s" % (band, tag, nm2), "bh", c)

for rn, (r0, r1, c0, c1) in REG7.items():
    for ch, cn in GCH:
        c = zeros("g7"); c[ch, r0:r1, c0:c1] = 16.0
        feat("g_%s_%s" % (rn, cn), "g7", c)

for ch, cn in ((RV, "rV"), (RH, "rH"), (BL, "ob")):
    for r in range(4):
        for cc in range(4):
            c = zeros("g4"); c[ch, r, cc] = 1.0
            feat("m_%s_%d%d" % (cn, r, cc), "g4", c)
for ch, cn in ((AV, "eV"), (AH, "eH")):
    for r0, r1, c0, c1, qn in ((0, 2, 0, 2, "TL"), (0, 2, 2, 4, "TR"), (2, 4, 0, 2, "BL"), (2, 4, 2, 4, "BR")):
        c = zeros("g4"); c[ch, r0:r1, c0:c1] = 1.0
        feat("q_%s_%s" % (cn, qn), "g4", c)

NF = len(FEATS)
FIDX = {n: i for i, n in enumerate(FEATS)}

for v in VIEW_SHAPE:
    mat = np.zeros((NF, int(np.prod(VIEW_SHAPE[v]))))
    for j, row in FC[v]:
        mat[j] = row.ravel()
    nm = "d_" + v
    layer(nm, "dense", [v], out=NF)
    put(nm, mat, np.zeros(NF))
layer("F", "add", ["d_bv", "d_bh", "d_g7", "d_g4"])

# ---------------------------------------------------------------- detectors
# Every detector is a positive homogeneous function of the filter responses:
# relu(energy_A - lambda * energy_B).  Both terms scale with image contrast, so
# the whole score vector scales together and the argmax is contrast invariant.
def comb(*items):
    out = {}
    for it in items:
        d, c = it if isinstance(it, tuple) else (it, 1.0)
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + c * v
    return out


def tot(*names):
    return {n: 1.0 for n in names}


DET = []


def det(name, terms, thr=0.0, scale=1.0):
    v = np.zeros(NF)
    for k, c in terms.items():
        v[FIDX[k]] += c
    DET.append((name, v * scale, thr * scale))
    return name


VS = tot("v0_r_tot", "v1_r_tot", "v2_r_tot", "v3_r_tot")
HS = tot("h0_r_tot", "h1_r_tot", "h2_r_tot", "h3_r_tot")
RS = comb(VS, HS)
E2 = tot("g_AL_eV", "g_AL_eH")
OB = tot("g_AL_ob")
RVH = tot("g_AL_rV", "g_AL_rH")
MLE = tot("g_ML_eV", "g_ML_eH")
MRE = tot("g_MR_eV", "g_MR_eH")
MCE = tot("g_MC_eV", "g_MC_eH")
T0E = tot("g_T0_eV", "g_T0_eH")
B0E = tot("g_B0_eV", "g_B0_eH")
V0 = tot("v0_r_tot"); V3 = tot("v3_r_tot")
H3 = tot("h3_r_tot"); H0 = tot("h0_r_tot")
HR = tot("h2_r_tot", "h3_r_tot"); HL = tot("h0_r_tot", "h1_r_tot")

# --- where the vertical prongs sit along the image height
det("uA", comb(V0, (VS, -0.16)))
det("uB", comb(V0, (VS, -0.26)))
det("uC", comb(V0, (VS, -0.36)))
det("dA", comb(V3, (VS, -0.16)))
det("dB", comb(V3, (VS, -0.26)))
det("midA", comb(tot("v1_r_tot", "v2_r_tot"), (VS, -0.60)))
det("midB", comb(tot("v1_r_tot", "v2_r_tot"), (VS, -0.72)))
det("vDom", comb(VS, (RS, -0.55)))
det("hDom", comb(HS, (RS, -0.55)))

# --- prong-ness of the top band: narrow ridges versus plain boundary
det("tRidge", comb(V0, (tot("v0_e_tot"), -0.75)))
det("tRidge2", comb(V0, (tot("v0_e_tot"), -1.05)))
det("bRidge", comb(V3, (tot("v3_e_tot"), -0.75)))
det("mRidge", comb(tot("v1_r_tot", "v2_r_tot"), (tot("v1_e_tot", "v2_e_tot"), -0.85)))

# --- wide gaps between prongs
det("gA", comb(tot("v0_g_tot"), (V0, -0.25)))
det("gB", comb(tot("v0_g_tot"), (V0, -0.50)))
det("gC", comb(tot("v0_g_tot"), (V0, -0.80)))
det("bgA", comb(tot("v3_g_tot"), (V3, -0.25)))
det("bgB", comb(tot("v3_g_tot"), (V3, -0.50)))

# --- spread and position of the top-band prongs (dimensionless, in px / px^2)
det("spN", comb((V0, 10.0), (tot("v0_r_m2"), -1.0)), 0.0, 0.05)
det("spM", comb((V0, 22.0), (tot("v0_r_m2"), -1.0)), 0.0, 0.05)
det("spW", comb(tot("v0_r_m2"), (V0, -18.0)), 0.0, 0.05)
det("spVW", comb(tot("v0_r_m2"), (V0, -34.0)), 0.0, 0.05)
det("pL", comb((tot("v0_r_m1"), -1.0), (V0, -2.0)), 0.0, 0.15)
det("pLL", comb((tot("v0_r_m1"), -1.0), (V0, -4.5)), 0.0, 0.15)
det("pR", comb(tot("v0_r_m1"), (V0, -2.0)), 0.0, 0.15)
det("pRR", comb(tot("v0_r_m1"), (V0, -4.5)), 0.0, 0.15)
det("fL", comb(tot("v0_r_fl"), (V0, -0.30)))
det("fR", comb(tot("v0_r_fr"), (V0, -0.30)))
det("ctr", comb(tot("v0_r_c"), (V0, -0.45)))
det("bspW", comb(tot("v3_r_m2"), (V3, -18.0)), 0.0, 0.05)
det("bpL", comb((tot("v3_r_m1"), -1.0), (V3, -2.0)), 0.0, 0.15)
det("bpR", comb(tot("v3_r_m1"), (V3, -2.0)), 0.0, 0.15)
det("bfL", comb(tot("v3_r_fl"), (V3, -0.30)))

# --- horizontal prongs: far right / far left column bands
det("rA", comb(H3, (HS, -0.18)))
det("rB", comb(H3, (HS, -0.30)))
det("lA", comb(H0, (HS, -0.18)))
det("lB", comb(H0, (HS, -0.30)))
det("rT", comb(tot("h3_r_t"), (H3, -0.55)))
det("rM", comb(tot("h3_r_m"), (H3, -0.45)))
det("rBo", comb(tot("h3_r_b"), (H3, -0.45)))
det("rFT", comb(tot("h3_r_ft"), (H3, -0.30)))
det("lT", comb(tot("h0_r_t"), (H0, -0.55)))
det("lM", comb(tot("h0_r_m"), (H0, -0.45)))
det("lBo", comb(tot("h0_r_b"), (H0, -0.45)))
det("rAsym", comb(HR, (HL, -1.30)))
det("lAsym", comb(HL, (HR, -1.30)))
det("rGap", comb(tot("h2_g_tot", "h3_g_tot"), (HR, -0.25)))
det("rRidge", comb(HR, (tot("h2_e_tot", "h3_e_tot"), -0.75)))
det("lRidge", comb(HL, (tot("h0_e_tot", "h1_e_tot"), -0.75)))
det("rMidLow", comb(tot("h3_r_m", "h3_r_b"), (H3, -0.55)))
det("lMidLow", comb(tot("h0_r_m", "h0_r_b"), (H0, -0.55)))

# --- coarse regional evidence, as fractions of the whole-image energy
det("topStrip", comb(T0E, (E2, -0.085)))
det("noTopStrip", comb((E2, 0.085), (T0E, -1.0)))
det("botStrip", comb(B0E, (E2, -0.10)))
det("noBotStrip", comb((E2, 0.10), (B0E, -1.0)))
det("holeC", comb(tot("g_MC_ob"), (OB, -0.22)))
det("holeML", comb(tot("g_ML_ob"), (OB, -0.18)))
det("holeMR", comb(tot("g_MR_ob"), (OB, -0.18)))
det("holeBC", comb(tot("g_BC_ob"), (OB, -0.18)))
det("holeT", comb(tot("g_T0_ob", "g_T1_ob"), (OB, -0.24)))
det("Cgap", comb(MLE, (MRE, -1.30)))
det("Oring", comb(MRE, (MLE, -0.78)))
det("leftArc", comb(tot("g_L0_eV", "g_L0_eH"), (E2, -0.09)))
det("rightArc", comb(tot("g_R0_eV", "g_R0_eH"), (E2, -0.09)))
det("ghAsym", comb(tot("g_BL_eV", "g_BL_eH"), (tot("g_BR_eV", "g_BR_eH"), -1.25)))
det("pqAsym", comb(tot("g_TL_eV", "g_TL_eH", "g_TR_eV", "g_TR_eH"),
                   (tot("g_BL_eV", "g_BL_eH", "g_BR_eV", "g_BR_eH"), -0.90)))
det("creaseT1", comb(tot("g_T1_eH"), (E2, -0.055)))
det("creaseMC", comb(tot("g_MC_eH"), (E2, -0.065)))
det("fistMid", comb(MCE, MLE, MRE, (E2, -0.50)))
det("diagDtr", comb(tot("g_TR_rD", "g_MR_rD"), (RVH, -0.10)))
det("diagDbl", comb(tot("g_BL_rD", "g_BC_rD"), (RVH, -0.10)))
det("topWide", comb(tot("g_T0_eV", "g_T1_eV"), (E2, -0.11)))
det("topR", comb(tot("g_T0R_eV", "g_T0R_eH"), (E2, -0.030)))
det("topL", comb(tot("g_T0L_eV", "g_T0L_eH"), (E2, -0.030)))
det("topC", comb(tot("g_T0C_eV", "g_T0C_eH"), (E2, -0.035)))
det("botL", comb(tot("g_B0L_eV", "g_B0L_eH"), (E2, -0.030)))
det("botC", comb(tot("g_B0C_eV", "g_B0C_eH"), (E2, -0.035)))

# --- coarse max-grid presence, relative to neighbouring cells
det("mRVr", comb(tot("m_rV_03"), (tot("m_rV_02"), -0.60)))
det("mRVl", comb(tot("m_rV_00"), (tot("m_rV_01"), -0.60)))
det("mOBc", comb(tot("m_ob_11", "m_ob_12", "m_ob_21", "m_ob_22"),
                 (tot("m_ob_00", "m_ob_03", "m_ob_30", "m_ob_33"), -0.50)))
det("mRHt", comb(tot("m_rH_02", "m_rH_03"), (tot("m_rH_22", "m_rH_23"), -0.60)))
det("mRHm", comb(tot("m_rH_12", "m_rH_13"), (tot("m_rH_02", "m_rH_03"), -0.60)))

NH = len(DET)
HIDX = {d[0]: i for i, d in enumerate(DET)}
hw = np.zeros((NH, NF)); hb = np.zeros(NH)
for i, (n, v, t) in enumerate(DET):
    hw[i] = v; hb[i] = -t
layer("dh", "dense", ["F"], out=NH)
put("dh", hw, hb)
layer("H", "relu", ["dh"])

# ---------------------------------------------------------------- pose layer
POSE = []


def pose(name, terms):
    v = np.zeros(NH)
    for k, c in terms.items():
        v[HIDX[k]] += c
    POSE.append((name, v))


pose("pUp", {"uA": 1.0, "uB": 0.6, "tRidge": 0.8, "topStrip": 0.5, "vDom": 0.3,
             "noTopStrip": -0.8, "dA": -0.6, "rA": -0.5, "midB": -0.5})
pose("pDown", {"dA": 1.0, "dB": 0.6, "bRidge": 0.8, "botStrip": 0.4, "vDom": 0.3,
               "noBotStrip": -0.8, "uA": -0.6, "rA": -0.5, "midB": -0.5})
pose("pRight", {"rA": 1.0, "rB": 0.6, "rRidge": 0.7, "rT": 0.4, "ghAsym": 0.4,
                "diagDtr": 0.3, "uA": -0.7, "dA": -0.7, "noTopStrip": -0.3})
pose("pFist", {"noTopStrip": 0.8, "midA": 0.7, "midB": 0.6, "fistMid": 0.4,
               "uA": -1.2, "uB": -0.8, "dA": -1.0, "rA": -0.8, "tRidge": -0.8,
               "topStrip": -0.6, "bRidge": -0.6})
pose("pRing", {"holeC": 1.0, "mOBc": 0.5, "Oring": 0.5, "holeML": 0.3,
               "uB": -0.6, "rA": -0.5})
pose("pCres", {"Cgap": 1.0, "leftArc": 0.6, "holeC": -0.8, "Oring": -0.5,
               "uB": -0.6, "rA": -0.5})
pose("pSpread", {"spW": 0.6, "gA": 0.5, "gB": 0.4, "spVW": 0.3, "spN": -0.6})
pose("pNarrow", {"spN": 0.7, "spM": 0.4, "spW": -0.6, "gA": -0.5})
pose("pThumbL", {"lA": 0.7, "lMidLow": 0.5, "lAsym": 0.4, "lRidge": 0.4, "rA": -0.5})
pose("pThumbR", {"rA": 0.6, "rMidLow": 0.6, "rM": 0.4, "rRidge": 0.4,
                 "rFT": -0.6, "uB": -0.5})
pose("pLeftPos", {"pLL": 0.8, "fL": 0.6, "mRVl": 0.4, "pR": -0.6})
pose("pCentrePos", {"ctr": 0.6, "pL": -0.4, "pR": -0.4, "fL": -0.5})

NP = len(POSE)
PIDX = {p[0]: i for i, p in enumerate(POSE)}
pw = np.zeros((NP, NH))
for i, (n, v) in enumerate(POSE):
    pw[i] = v
layer("dp", "dense", ["H"], out=NP)
put("dp", pw, np.zeros(NP))
layer("P", "relu", ["dp"])

# ---------------------------------------------------------------- class readout
# Hand-written rule table.  Geometry: thumb side = image right, pinky = image left.
SCORE = {}


def sc(k, poses, terms):
    SCORE[k] = (poses, terms)


FISTP = {"pFist": 3.0, "pUp": -1.2, "pRight": -1.6, "pDown": -1.6, "pRing": -1.0, "pCres": -1.0}
UPP = {"pUp": 2.6, "pFist": -2.2, "pRight": -1.6, "pDown": -1.6}
RIGHTP = {"pRight": 3.4, "pUp": -1.5, "pDown": -1.6, "pFist": -1.0, "pRing": -0.5, "pCres": -0.5}
DOWNP = {"pDown": 3.4, "pUp": -1.6, "pRight": -1.6, "pFist": -1.0}

# 0 A : fist, thumb standing vertically along the right side, four knuckles on top
sc(0, dict(FISTP, pThumbL=-0.6),
   {"fR": 0.9, "mRVr": 0.9, "creaseT1": 0.2, "mRHt": -0.5,
    "ctr": -0.3, "holeML": -0.3})
# 1 B : four straight fingers together, wide flat top, thumb folded across the palm
sc(1, dict(UPP, pSpread=0.3, pNarrow=-0.5, pRing=-0.8),
   {"uC": 0.8, "topWide": 1.0, "topC": 0.6, "spM": 0.4, "gB": -1.0, "gC": -1.0,
    "fL": -0.5, "fR": -0.5, "spN": -0.4})
# 2 C : open crescent, thick arc on the left, gap at middle right
sc(2, {"pCres": 3.4, "pUp": -1.5, "pFist": -0.6, "pRing": -1.6, "pRight": -1.2, "pDown": -1.2},
   {"Cgap": 0.8, "leftArc": 0.8, "rightArc": -0.6, "holeC": -0.7, "mOBc": -0.4})
# 3 D : one index finger up near the centre, ring of curled fingers + thumb below
sc(3, dict(UPP, pRing=1.3, pNarrow=1.0, pSpread=-1.0, pThumbR=-1.0, pCentrePos=0.6),
   {"spN": 0.6, "ctr": 0.6, "holeC": 0.6, "holeML": 0.5, "uB": -0.9, "gA": -0.8,
    "pLL": -0.7, "fL": -0.7})
# 4 E : tight fist, four fingertips curled down onto the thumb, strong crease
sc(4, FISTP, {"creaseT1": 0.9, "creaseMC": 0.6, "fR": -0.6, "mRHt": -0.5, "mRVr": -0.4,
              "lMidLow": -0.4})
# 5 F : three spread fingers up + closed index/thumb ring at middle right
sc(5, dict(UPP, pSpread=1.0, pRing=1.1, pNarrow=-0.8),
   {"gA": 0.9, "gB": 0.6, "holeMR": 1.1, "holeC": 0.4, "uB": 0.5, "uC": 0.3,
    "spW": 0.4, "gC": -0.4, "Cgap": -0.5})
# 6 G : index pointing right and slightly up, hand mass at lower left
sc(6, RIGHTP, {"rT": 0.8, "rFT": 0.6, "diagDtr": 0.6, "topR": 0.5, "rGap": 0.4,
               "rB": -0.4, "lA": -0.4})
# 7 H : index + middle together pointing right, thicker horizontal bar
sc(7, RIGHTP, {"rT": 0.8, "rB": 0.7, "diagDtr": 0.5, "topR": 0.4, "rGap": -0.7, "lA": -0.4})
# 8 I : pinky alone up on the far left, fist to its right
sc(8, dict(UPP, pNarrow=1.2, pSpread=-1.2, pRing=-0.5, pThumbR=-1.2, pLeftPos=1.3),
   {"spN": 0.7, "pLL": 0.9, "fL": 0.9, "mRVl": 0.6, "uB": -0.9, "gA": -0.8, "pR": -0.7})
# 9 K : index + middle in a wide V, thumb pushed out to the left between them
sc(9, dict(UPP, pSpread=0.7, pThumbL=1.2, pRing=-0.3),
   {"gA": 0.7, "spW": 0.4, "lMidLow": 0.6, "lA": 0.5, "uB": 0.5, "uC": -0.6,
    "gC": -0.6, "spN": -0.5})
# 10 L : index up, thumb straight out to the right at mid height
sc(10, dict(UPP, pNarrow=0.8, pThumbR=1.5, pSpread=-0.9, pRing=-0.5, pCentrePos=0.4),
   {"spN": 0.6, "rMidLow": 0.8, "rM": 0.6, "uB": -0.9, "gA": -0.8, "pLL": -0.7,
    "fL": -0.7, "rB": -0.4, "rFT": -0.5})
# 11 M : three fingers folded over the thumb, thumb tip low on the left
sc(11, FISTP, {"lMidLow": 0.6, "lBo": 0.4, "creaseMC": 0.3, "midA": 0.3,
               "fR": -0.7, "mRHt": -0.5, "uB": -0.3})

# 12 N : two fingers folded over the thumb, thumb tip low on the left
sc(12, FISTP, {"lMidLow": 0.5, "lBo": 0.35, "creaseMC": 0.3, "fR": -0.7, "mRHt": -0.5,
               "uB": -0.5, "midA": 0.2})
# 13 O : all fingertips meet the thumb, one closed ring with a hole in the middle
sc(13, {"pRing": 3.4, "pFist": 0.4, "pUp": -1.5, "pRight": -1.5, "pDown": -1.5, "pCres": -2.0},
   {"holeC": 1.0, "mOBc": 0.8, "Oring": 0.8, "leftArc": 0.4, "rightArc": 0.4,
    "Cgap": -0.9, "gA": -0.7})
# 14 P : K turned downwards, two prongs down plus the thumb, wrist from the top
sc(14, dict(DOWNP, pRing=0.3),
   {"dB": 0.8, "bgA": 0.6, "bspW": 0.4, "botC": 0.4, "bpL": 0.3, "dA": 0.3})
# 15 Q : G turned downwards, one prong down with the thumb beside it
sc(15, DOWNP, {"dA": 0.8, "diagDbl": 0.6, "botL": 0.4, "bfL": 0.4, "bpL": 0.4,
               "dB": -0.7, "bgA": -0.4})
# 16 R : index and middle crossed, one narrow prong, loop of curled fingers below
sc(16, dict(UPP, pNarrow=1.0, pRing=1.0, pSpread=-1.0, pThumbR=-1.0, pCentrePos=0.4),
   {"spN": 0.6, "holeML": 0.8, "holeC": 0.5, "ctr": 0.4, "uB": -0.5, "gA": -0.9,
    "spW": -0.5, "pLL": -0.5})
# 17 S : fist with the thumb lying horizontally across the front, tip to the left
sc(17, FISTP, {"lMidLow": 0.9, "lM": 0.7, "lRidge": 0.4, "mRHm": 0.4,
               "fR": -0.8, "mRHt": -0.6, "lBo": -0.3})
# 18 T : thumb tip pokes up between index and middle, index folded to the right
sc(18, dict(FISTP, pFist=2.2, pRight=0.4, pUp=-0.5),
   {"rT": 0.7, "mRHt": 0.7, "topL": 0.5, "creaseT1": 0.3, "rB": -0.8,
    "ghAsym": -0.7, "rFT": -0.4, "gA": -0.7})
# 19 U : index and middle straight up, touching, one narrow block
sc(19, dict(UPP, pNarrow=1.5, pSpread=-1.5, pRing=-0.4, pCentrePos=0.5),
   {"spN": 1.0, "spM": 0.6, "ctr": 0.5, "topC": 0.4, "uB": 0.4, "gA": -1.1,
    "gB": -0.7, "spW": -0.7, "pLL": -0.6, "fL": -0.6})
# 20 V : index and middle straight up, spread apart, one wide gap
sc(20, dict(UPP, pSpread=1.5, pNarrow=-1.0, pThumbL=-0.7, pRing=-0.5),
   {"gA": 1.1, "gB": 0.4, "spW": 0.6, "uB": 0.5, "uC": -0.7, "gC": -0.8,
    "spN": -0.7, "pLL": -0.5})
# 21 W : index, middle and ring spread up, two wide gaps
sc(21, dict(UPP, pSpread=1.8, pNarrow=-1.2, pRing=-0.9),
   {"gB": 1.1, "gC": 0.7, "spVW": 0.6, "spW": 0.4, "uB": 0.6, "uC": 0.4,
    "spN": -0.7, "holeMR": -0.8, "holeC": -0.4, "Cgap": -0.5})
# 22 X : index bent into a hook above a fist, small loop under the hook
sc(22, dict(FISTP, pFist=2.0, pUp=0.2, pRing=0.3, pSpread=-0.8),
   {"holeT": 1.1, "creaseT1": 0.5, "topC": 0.3, "mRHt": 0.3, "uB": -0.8,
    "gA": -0.8, "spW": -0.5, "rMidLow": -0.5})
# 23 Y : pinky up on the far left and thumb out to the right, three fingers folded
sc(23, dict(UPP, pNarrow=0.8, pThumbR=1.5, pSpread=-1.0, pFist=-1.0, pLeftPos=1.3),
   {"pLL": 1.0, "fL": 0.9, "mRVl": 0.6, "rMidLow": 0.8, "rM": 0.6, "rBo": 0.4,
    "uB": -0.9, "gA": -0.8, "pR": -0.7, "rB": -0.4, "rFT": -0.6})

# ---------------------------------------------------------------- output layers
rw = np.zeros((24, NH)); qw = np.zeros((24, NP))
for k, (poses, terms) in SCORE.items():
    for n, c in terms.items():
        rw[k, HIDX[n]] += c
    for n, c in poses.items():
        qw[k, PIDX[n]] += c
layer("s1", "dense", ["H"], out=24)
put("s1", rw, np.zeros(24))
layer("s2", "dense", ["P"], out=24)
put("s2", qw, np.zeros(24))
layer("scores", "add", ["s1", "s2"])

# ---------------------------------------------------------------- write answer
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": LAYERS}, f, indent=1)
np.savez(os.path.join(OUT, "weights.npz"), **W)
print("layers %d  features %d  detectors %d  poses %d  weights %d"
      % (len(LAYERS), NF, NH, NP, sum(int(v.size) for v in W.values())))
