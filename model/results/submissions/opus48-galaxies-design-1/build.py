"""Builds a hand-designed galaxy-classifier network: writes network.json and weights.npz.

Every weight comes from an analytic formula (Gaussian windows, quadrupole-moment
templates, a difference-of-directions anisotropy, raw colour) or from a hand-reasoned
class scorecard built from general knowledge of what the ten galaxy classes look like.
No training data, no files and no randomness are read or used.

Idea.  Each galaxy sits at the image centre.  From the contrast-normalised light we
measure three scale/rotation/contrast-robust quantities and classify with an explicit
rule table encoded as a tiny relu network:
  * colour  -- smooth ellipticals are red, star-forming disks are blue;
  * elongation ratio r = ANISO/Tr of the (core-clipped, isophotal) light, rotation
    invariant via a max over directions;  round -> in-between -> cigar/edge-on;
  * size baseline Tr.
All score terms are homogeneous (proportional to galaxy size), so the argmax depends
only on the scale-free ratio r and the colour sign, not on the galaxy's apparent size.

Run:  python3 build.py OUT_DIR
"""
import sys, os, json
import numpy as np

# ---- analysis grid (128 -> 64 after avgpool [2,2]) -------------------------
G = 64
CEN = (G - 1) / 2.0
_i, _j = np.mgrid[0:G, 0:G].astype(np.float64)
X = _j - CEN
Y = _i - CEN
R = np.sqrt(X * X + Y * Y)


def gauss(sig):
    return np.exp(-(R * R) / (2.0 * sig * sig))


def window_unit(sig):
    w = gauss(sig)
    return w / w.sum()


def moment_templates(sig):
    """Quadrupole weights with one shared normalisation so ANISO/Tr = (1-q^2)/(1+q^2)."""
    g = gauss(sig)
    wT = (X * X + Y * Y) * g
    norm = wT.sum()
    return wT / norm, (X * X - Y * Y) * g / norm, (2.0 * X * Y) * g / norm


# ---- layer / weight assembly ----------------------------------------------
layers, W = [], {}


def add(name, op, inputs, **kw):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(kw)
    layers.append(d)


def conv1x1(name, inp, mat, bias):
    mat = np.asarray(mat, np.float64)
    add(name, "conv", [inp], out=mat.shape[0], kernel=[1, 1], stride=1, padding=0)
    W[name + ".W"] = mat.reshape(mat.shape[0], mat.shape[1], 1, 1)
    W[name + ".b"] = np.asarray(bias, np.float64)


# luminance and raw redness (R=z band is channel 0, B=g band is channel 2)
conv1x1("lum", "image", [[1 / 3, 1 / 3, 1 / 3]], [0.0])
conv1x1("red", "image", [[1.0, 0.0, -1.0]], [0.0])

# contrast/background-normalised light, then its positive part
add("lumn", "norm", ["lum"])
add("Mpos", "relu", ["lumn"])

# clip the galaxy light to an isophotal plateau: Mclip = min(Mpos, C) = Mpos - relu(Mpos - C).
# A low clip makes the quadrupole reflect the outline (so thin edge-on disks read as very
# elongated) instead of the round bright core.
C_CLIP = 0.4
conv1x1("mhi", "Mpos", [[1.0]], [-C_CLIP])
add("mhir", "relu", ["mhi"])
conv1x1("mneg", "mhir", [[-1.0]], [0.0])
add("Mclip", "add", ["Mpos", "mneg"])
add("Md", "avgpool", ["Mclip"], size=[2, 2], stride=[2, 2])          # (1,64,64)

# raw luminance / redness, downsampled, for the colour test
add("Ld", "avgpool", ["lum"], size=[2, 2], stride=[2, 2])            # (1,64,64)
add("Rd", "avgpool", ["red"], size=[2, 2], stride=[2, 2])            # (1,64,64)

# ---- shape moments: mfeat = [Tr, A, B] ------------------------------------
SIG_MOM, SIG_COL = 9.0, 8.0
wT, wA, wB = moment_templates(SIG_MOM)
Mw = np.stack([wT.reshape(-1), wA.reshape(-1), wB.reshape(-1)])
add("mfeat", "dense", ["Md"], out=3)
W["mfeat.W"], W["mfeat.b"] = Mw, np.zeros(3)

# anisotropy magnitude = max over directions of cos(a)*A + sin(a)*B = sqrt(A^2+B^2)
NPHI = 24
Pw = np.zeros((NPHI, 3), np.float64)
for jj in range(NPHI):
    a = 2.0 * np.pi * jj / NPHI
    Pw[jj, 1] = np.cos(a); Pw[jj, 2] = np.sin(a)
add("proj", "dense", ["mfeat"], out=NPHI)
W["proj.W"], W["proj.b"] = Pw, np.zeros(NPHI)
add("projR", "reshape", ["proj"], shape=[1, 1, NPHI])
add("anisoP", "maxpool", ["projR"], size=[1, NPHI], stride=[1, 1])
add("aniso", "reshape", ["anisoP"], shape=[1])

# ---- feature vector F = [Tr, ANISO, COLraw, LUMcol] ------------------------
NF = 4
fm = np.zeros((NF, 3), np.float64); fm[0, 0] = 1.0          # Tr
add("fm", "dense", ["mfeat"], out=NF)
W["fm.W"], W["fm.b"] = fm, np.zeros(NF)

fa = np.zeros((NF, 1), np.float64); fa[1, 0] = 1.0          # ANISO
add("fa", "dense", ["aniso"], out=NF)
W["fa.W"], W["fa.b"] = fa, np.zeros(NF)

wcol = window_unit(SIG_COL).reshape(-1)
fc = np.zeros((NF, G * G), np.float64); fc[2] = wcol        # COLraw (central redness)
add("fc", "dense", ["Rd"], out=NF)
W["fc.W"], W["fc.b"] = fc, np.zeros(NF)

fL = np.zeros((NF, G * G), np.float64); fL[3] = wcol        # LUMcol (central luminance)
add("fL", "dense", ["Ld"], out=NF)
W["fL.W"], W["fL.b"] = fL, np.zeros(NF)

add("F", "add", ["fm", "fa", "fc", "fL"])

# ---- gates (relu of homogeneous linear comparisons) ------------------------
# F indices: Tr=0, ANISO=1, COLraw=2, LUMcol=3
TT = [0.0, 0.2, 0.35, 0.5, 0.65]     # r=ANISO/Tr thresholds across the axis-ratio range
THETA = 0.12                         # redness-per-luminance threshold (red vs blue)
GATES = [("g%d" % k, {1: 1.0, 0: -t}) for k, t in enumerate(TT)]   # g_t = Tr*max(r-t,0)
GATES += [
    ("red",   {2: 1.0, 3: -THETA}),          # COLraw - THETA*LUMcol  (>0 red)
    ("blue",  {2: -1.0, 3: THETA}),           # THETA*LUMcol - COLraw  (>0 blue)
    ("trace", {0: 1.0}),                      # Tr (size baseline, >=0)
]
NG = len(GATES)
gpre = np.zeros((NG, NF), np.float64)
for gi, (_, coeffs) in enumerate(GATES):
    for idx, c in coeffs.items():
        gpre[gi, idx] = c
add("gpre", "dense", ["F"], out=NG)
W["gpre.W"], W["gpre.b"] = gpre, np.zeros(NG)
add("gates", "relu", ["gpre"])

# ---- class scorecards ------------------------------------------------------
# Colour (CW, large) picks the red/blue group and is identical within a group, so it
# cancels in the intra-group argmax; the elongation terms then route by the ratio r.
GI = {name: k for k, (name, _) in enumerate(GATES)}
CW = 3.0
ROUND = {"trace": 0.5, "g0": -2.0}                 # prefers low r (round)
INBET = {"g1": 2.0, "g2": -4.0, "g3": 2.0}         # triangular bump, mid-low r
CIGAR = {"g2": 2.0, "g3": -4.0, "g4": 2.0}         # triangular bump, mid r
EDGE = {"g4": 3.0}                                 # ramp up, high r
DORMANT = {"trace": -0.5}                           # never wins (no reliable feature)


def merged(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    return out


scard = {
    2: merged(ROUND, {"red": CW}),      # round smooth
    3: merged(INBET, {"red": CW}),      # in-between round smooth
    4: merged(CIGAR, {"red": CW}),      # cigar-shaped smooth
    9: merged(EDGE, {"red": CW}),       # edge-on with bulge
    5: merged(ROUND, {"blue": CW}),     # barred spiral
    6: merged(ROUND, {"blue": CW}),     # (unbarred) tight spiral  -- ties to 5
    7: merged(INBET, {"blue": CW}),     # unbarred loose spiral
    8: merged(EDGE, {"blue": CW}),      # edge-on without bulge
    1: dict(DORMANT),                   # merging (dormant)
    0: dict(DORMANT),                   # disturbed (dormant)
}
scg = np.zeros((10, NG), np.float64)
for cls, d in scard.items():
    for gname, val in d.items():
        scg[cls, GI[gname]] = val
add("scores", "dense", ["gates"], out=10)
W["scores.W"], W["scores.b"] = scg, np.zeros(10)


# ---------------------------------------------------------------------------
def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f)
    np.savez(os.path.join(out_dir, "weights.npz"),
             **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
    print("wrote network.json and weights.npz")
