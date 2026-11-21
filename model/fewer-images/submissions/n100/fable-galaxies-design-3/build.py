"""Hand-designed galaxy classifier. Writes network.json and weights.npz into the folder given as argv[1].

Every number here is typed by hand from general knowledge (the Legacy Survey asinh colour rendering puts the
sky at about 0.13; axis-ratio formulas for ellipses; red-sequence / blue-cloud colours) -- nothing is fitted.

Pipeline
  p    4x4 box average, stride 2 (noise suppression)            -> (3, 63, 63)
  c1   1x1 conv + relu: sky-subtracted brightness, three soft isophote masks (low / mid / high level), R and B
  fp   dense: window-weighted first and second moments of each mask projected on 16 directions
  fpm  max over the 16 directions = vector magnitudes (ellipticity numerator, centroid offset)
  fs   dense: mask areas, traces of second moments, flux, red-minus-blue flux
  gn   norm over groups [a, -a, b, -b]  ->  sqrt(2) * (a/b) / sqrt(1 + (a/b)^2): scale-free ratios
  h    ramps (two relus each) on the ratios, with hand-chosen thresholds
  out  hand-written rule table: class score = bias + sum of ramp votes
"""
import json, sys
from pathlib import Path
import numpy as np

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out_dir.mkdir(parents=True, exist_ok=True)

N = 63                       # side of the pooled map
CEN = 31.0                   # image centre in pooled-pixel index
SKY_T = 0.17                 # sky (0.13 from the rendering formula) + 0.04 margin
ND = 16                      # directions for vector magnitudes
SIGMA_W = 10.0               # central Gaussian window (pooled pixels = 2 original pixels)

W, layers = {}, []

def add_layer(name, op, inputs, **kw):
    layers.append({"name": name, "op": op, "inputs": inputs, **kw})

# ---------------------------------------------------------------- pixel stage
add_layer("p", "avgpool", ["image"], size=[4, 4], stride=[2, 2])
C1 = 9
w = np.zeros((C1, 3, 1, 1)); b = np.zeros(C1)
mean = np.array([1 / 3, 1 / 3, 1 / 3])
w[0, :, 0, 0] = mean;            b[0] = -SKY_T                      # brightness above sky
LEVELS = [(SKY_T, 0.04), (0.35, 0.10), (0.55, 0.10)]                # (start, width) of the soft masks
for k, (t0, wd) in enumerate(LEVELS):
    w[1 + 2 * k, :, 0, 0] = mean / wd;  b[1 + 2 * k] = -t0 / wd
    w[2 + 2 * k, :, 0, 0] = mean / wd;  b[2 + 2 * k] = -t0 / wd - 1.0
w[7, 0, 0, 0] = 1.0;             b[7] = -SKY_T                      # red channel above sky
w[8, 2, 0, 0] = 1.0;             b[8] = -SKY_T                      # blue channel above sky
W["c1.W"], W["c1.b"] = w, b
add_layer("c1", "conv", ["p"], out=C1, kernel=[1, 1], stride=1, padding=0)
add_layer("r1", "relu", ["c1"])

# ---------------------------------------------------------------- moments (dense on the pooled map)
yy, xx = np.meshgrid(np.arange(N) - CEN, np.arange(N) - CEN, indexing="ij")
win = np.exp(-(xx ** 2 + yy ** 2) / (2 * SIGMA_W ** 2))
npx = N * N

def put(mat, row, chan, img, sign=1.0):
    mat[row, chan * npx:(chan + 1) * npx] += sign * img.reshape(-1)

def put_mask(mat, row, level, img):          # mask_k = relu(z) - relu(z - 1)
    put(mat, row, 1 + 2 * level, img, +1.0)
    put(mat, row, 2 + 2 * level, img, -1.0)

fp = np.zeros((6 * ND, C1 * npx))            # groups: E_low, E_mid, E_high, O_low, O_mid, O_high
for lev in range(3):
    for d in range(ND):
        ang = 2 * np.pi * d / ND
        put_mask(fp, lev * ND + d, lev, win * ((xx ** 2 - yy ** 2) * np.cos(ang) + 2 * xx * yy * np.sin(ang)))
        put_mask(fp, (3 + lev) * ND + d, lev, win * (xx * np.cos(ang) + yy * np.sin(ang)))
W["fp.W"], W["fp.b"] = fp, np.zeros(6 * ND)
add_layer("fp", "dense", ["r1"], out=6 * ND)
add_layer("fpr", "reshape", ["fp"], shape=[1, 6, ND])
add_layer("fpm", "maxpool", ["fpr"], size=[1, ND], stride=[1, ND])

fs = np.zeros((8, C1 * npx))                 # A_low, T_low, A_mid, T_mid, A_high, T_high, F, Cr
for lev in range(3):
    put_mask(fs, 2 * lev, lev, win)
    put_mask(fs, 2 * lev + 1, lev, win * (xx ** 2 + yy ** 2))
put(fs, 6, 0, win)
put(fs, 7, 7, win, +1.0)
put(fs, 7, 8, win, -1.0)
W["fs.W"], W["fs.b"] = fs, np.zeros(8)
add_layer("fs", "dense", ["r1"], out=8)

# ---------------------------------------------------------------- scale-free ratios via norm
# group g = [a, -a, b, -b];  source of a / b: ("m", i) = fpm[i], ("s", i) = fs[i]
GROUPS = [(("m", 0), ("s", 1)),   # 0 ellipticity of low mask
          (("m", 1), ("s", 3)),   # 1 ellipticity of mid mask
          (("m", 2), ("s", 5)),   # 2 ellipticity of high mask
          (("m", 3), ("s", 0)),   # 3 centroid offset of low mask
          (("m", 4), ("s", 2)),   # 4 centroid offset of mid mask
          (("m", 5), ("s", 4)),   # 5 centroid offset of high mask
          (("s", 7), ("s", 6)),   # 6 colour (R - B) / brightness
          (("s", 4), ("s", 0)),   # 7 bright-core area fraction
          (("s", 2), ("s", 0))]   # 8 mid area fraction (spare)
NG = len(GROUPS)
ga, gb = np.zeros((4 * NG, 6)), np.zeros((4 * NG, 8))
for g, (a, bsrc) in enumerate(GROUPS):
    for (kind, idx), base in ((a, 0), (bsrc, 2)):
        mat = ga if kind == "m" else gb
        mat[4 * g + base, idx] = 1.0
        mat[4 * g + base + 1, idx] = -1.0
W["ga.W"], W["ga.b"] = ga, np.zeros(4 * NG)
W["gb.W"], W["gb.b"] = gb, np.zeros(4 * NG)
add_layer("ga", "dense", ["fpm"], out=4 * NG)
add_layer("gb", "dense", ["fs"], out=4 * NG)
add_layer("g", "add", ["ga", "gb"])
add_layer("gr", "reshape", ["g"], shape=[NG, 4, 1])
add_layer("gn", "norm", ["gr"])

# ---------------------------------------------------------------- ramps  clip((sign*x - t0) / (t1 - t0), 0, 1)
def psi(x):                                   # what the norm group returns for a ratio x
    return np.sqrt(2.0) * x / np.sqrt(1.0 + x * x)

def psi_q(q):                                 # low-mask ellipticity reading of an ellipse with axis ratio q
    return psi((1 - q * q) / (1 + q * q))

RAMPS = [  # name, source, sign, t0, t1
    ("NR",  ("n", 0), 1, psi_q(0.87), psi_q(0.77)),   # not round
    ("NR2", ("n", 0), 1, psi_q(0.77), psi_q(0.56)),   # clearly elongated outline
    ("EL",  ("n", 0), 1, psi_q(0.53), psi_q(0.41)),   # cigar / edge-on outline
    ("TH",  ("n", 0), 1, psi_q(0.38), psi_q(0.28)),   # thin outline
    ("LG",  ("s", 0), 1, 90.0, 220.0),                # large (windowed area; radius about 11 -> 19 pixels)
    ("RD",  ("n", 6), 1, 0.15, 0.50),                 # red (old stellar population)
    ("BL",  ("n", 6), -1, -0.10, 0.25),               # blue (star forming)
    ("CO",  ("n", 7), 1, 0.03, 0.20),                 # has a bright core
    ("EM",  ("n", 1), 1, 0.45, 0.80),                 # mid isophote elongated
    ("EH",  ("n", 2), 1, 0.45, 0.85),                 # bright core elongated
    ("EH2", ("n", 2), 1, 0.80, 0.95),                 # bright core very elongated (double nucleus / thin disc)
    ("OH",  ("n", 5), 1, 0.90, 1.25),                 # bright core off centre
    ("OM",  ("n", 4), 1, 0.90, 1.25),                 # mid isophote off centre
    ("OL",  ("n", 3), 1, 1.00, 1.30),                 # outline off centre (lopsided)
]
NRAMP = len(RAMPS)
ha, hb, hbias = np.zeros((2 * NRAMP, 4 * NG)), np.zeros((2 * NRAMP, 8)), np.zeros(2 * NRAMP)
for k, (_, (kind, idx), sign, t0, t1) in enumerate(RAMPS):
    s = sign / (t1 - t0)
    for j in range(2):
        if kind == "n":
            ha[2 * k + j, 4 * idx] = s
        else:
            hb[2 * k + j, idx] = s
        hbias[2 * k + j] = -t0 / (t1 - t0) - j
W["ha.W"], W["ha.b"] = ha, hbias
W["hb.W"], W["hb.b"] = hb, np.zeros(2 * NRAMP)
add_layer("ha", "dense", ["gn"], out=2 * NRAMP)
add_layer("hb", "dense", ["fs"], out=2 * NRAMP)
add_layer("h", "add", ["ha", "hb"])
add_layer("hr", "relu", ["h"])

# ---------------------------------------------------------------- rule table (votes per class)
RULES = {
    0: (-1.2, dict(OL=1.5, BL=0.8, LG=0.5, EL=-1.5)),                                   # disturbed
    1: (-1.0, dict(OH=2.0, OM=1.2, EH=1.0, EH2=2.0, OL=0.6, EL=-2.5)),                  # merging
    2: (1.5,  dict(NR=-5.0, LG=-2.0, BL=-1.0, EM=-1.0)),                                # round smooth
    3: (-3.5, dict(NR=5.0, EL=-5.0, LG=-2.0, BL=-0.8)),                                 # in-between smooth
    4: (-4.7, dict(EL=5.0, TH=-3.5, LG=-3.0)),                                          # cigar smooth
    5: (-1.5, dict(LG=2.2, EM=1.2, NR2=-1.2, EL=-3.0, RD=0.3, CO=0.3)),                 # barred spiral
    6: (-1.3, dict(LG=2.2, NR=-0.6, CO=0.5, RD=0.3, EL=-3.0, EM=-0.4)),                 # tight spiral
    7: (-1.2, dict(LG=2.0, BL=1.2, EL=-3.0, OL=0.4, CO=-0.4)),                          # loose spiral
    8: (-4.5, dict(EL=3.0, TH=3.0, BL=1.0, CO=-1.2, RD=-0.8, LG=0.5, EH2=1.5)),         # edge-on, no bulge
    9: (-4.0, dict(EL=3.0, TH=2.0, CO=1.0, RD=0.8, BL=-0.5, EH2=-1.2)),                 # edge-on, bulge
}
names = [r[0] for r in RAMPS]
ow, ob = np.zeros((10, 2 * NRAMP)), np.zeros(10)
for c, (bias, votes) in RULES.items():
    ob[c] = bias
    for nm, v in votes.items():
        k = names.index(nm)
        ow[c, 2 * k] = v
        ow[c, 2 * k + 1] = -v
W["out.W"], W["out.b"] = ow, ob
add_layer("out", "dense", ["hr"], out=10)

(out_dir / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
np.savez(out_dir / "weights.npz", **{k: v.astype(np.float32) for k, v in W.items()})
print("wrote", out_dir, "weights:", sum(v.size for v in W.values()))
