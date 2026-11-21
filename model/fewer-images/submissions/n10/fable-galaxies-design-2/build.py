"""Hand-designed galaxy classifier. Every weight comes from the formulas below (no data, no fitting).

Idea: matched filtering. The image is halved, turned into a brightness map and a blue-minus-red map, cropped to the
central 48 x 48 and normalised (norm op).  Dense layers hold analytic shape templates (ellipses of given axis ratio,
edge-on disk + bulge, two blobs, disk + bar), each made zero-mean / unit-norm by itself, so each dense output is the
Pearson correlation of the image with that shape.  A max-pool over each template family gives "how well does the best
round / in-between / cigar / thin / disk+bulge / double / barred model fit".  A tiny hand-written rule layer combines
those fits with a colour cue (spirals blue, smooth galaxies yellow-red) and general class priors into 10 scores.
"""
import sys, json, os
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

HW = 48                      # side of the cropped half-resolution map
N = HW * HW
yy, xx = np.mgrid[0:HW, 0:HW].astype(np.float64)
X = xx - (HW - 1) / 2.0
Y = yy - (HW - 1) / 2.0


def ell(sa, sb, th, x0=0.0, y0=0.0):
    """Elliptical radius: major scale sa along angle th, minor scale sb, centre offset (x0, y0)."""
    u = (X - x0) * np.cos(th) + (Y - y0) * np.sin(th)
    v = -(X - x0) * np.sin(th) + (Y - y0) * np.cos(th)
    return np.sqrt((u / sa) ** 2 + (v / sb) ** 2)


def prof(rho, kind):
    """kind 0: Gaussian profile, kind 1: exponential (disk-like, cuspy) profile."""
    return np.exp(-0.5 * rho ** 2) if kind == 0 else np.exp(-rho)


def finish(t):
    """Normalise one template by itself: zero mean, unit norm; /sqrt(N) so that dense(norm(x)) = Pearson r."""
    t = t - t.mean()
    t = t / np.sqrt((t ** 2).sum())
    return t / np.sqrt(N)


S = [1.5, 2.1, 3.0, 4.2, 6.0, 8.5, 12.0]          # major-axis scales in half-resolution pixels
TH = [np.pi * k / 18.0 for k in range(18)]        # 18 orientations, 10 degrees apart
PSF = 0.55                                        # smallest believable minor-axis scale (seeing)


def ellipse_group(qs):
    T = []
    for q in qs:
        for s in S:
            if q * s < 0.5:
                continue                           # unresolved: shape would not have this axis ratio
            ths = [0.0] if q >= 0.999 else TH
            for th in ths:
                for kind in (0, 1):
                    T.append(finish(prof(ell(s, max(q * s, PSF), th), kind)))
    return T


def bulge_group():
    """Edge-on disk with a central bulge: thin exponential disk + rounder Gaussian bulge."""
    T = []
    for qd in (0.14, 0.22):
        for s in S:
            if qd * s < 0.5:
                continue
            for th in TH:
                disk = np.exp(-ell(s, max(qd * s, PSF), th))
                for fb in (0.25, 0.4):
                    sb = max(fb * s, 0.7)
                    bulge = np.exp(-0.5 * ell(sb, 0.8 * sb, th) ** 2)
                    T.append(finish(disk + 1.5 * bulge))
    return T


def double_group():
    """Two separate cores: one at the centre plus a neighbour, or a pair straddling the centre."""
    T = []
    for s in (1.5, 2.1, 3.0, 4.2):
        for dm in (2.5, 3.5):
            d = dm * s
            for k in range(12):
                ph = 2 * np.pi * k / 12.0
                dx, dy = d * np.cos(ph), d * np.sin(ph)
                for kind in (0, 1):
                    T.append(finish(prof(ell(s, s, 0.0), kind) + 0.8 * prof(ell(0.8 * s, 0.8 * s, 0.0, dx, dy), kind)))
            for k in range(6):
                ph = np.pi * k / 6.0
                dx, dy = 0.5 * d * np.cos(ph), 0.5 * d * np.sin(ph)
                for kind in (0, 1):
                    T.append(finish(prof(ell(s, s, 0.0, dx, dy), kind) + prof(ell(s, s, 0.0, -dx, -dy), kind)))
    return T


def bar_group():
    """Barred spiral: round or mildly inclined faint disk with a short, narrow, brighter bar through the centre."""
    T = []
    for s in S[1:]:
        for th in TH:
            for qd, dth in ((1.0, 0.0), (0.8, np.pi / 2)):
                disk = np.exp(-0.5 * ell(s, qd * s, th + dth) ** 2)
                bar = np.exp(-0.5 * ell(0.6 * s, max(0.17 * s, PSF), th) ** 2)
                T.append(finish(disk + 1.2 * bar))
    return T


groups = [                                   # (name, templates), feature index = position in this list
    ("round", ellipse_group([1.0, 0.88])),          # 0 -> class 2
    ("mid", ellipse_group([0.76, 0.64, 0.54])),     # 1 -> class 3
    ("cigar", ellipse_group([0.45, 0.37])),         # 2 -> class 4
    ("thin", ellipse_group([0.28, 0.20, 0.14])),    # 3 -> class 8
    ("bulge", bulge_group()),                       # 4 -> class 9
    ("double", double_group()),                     # 5 -> class 1
    ("bar", bar_group()),                           # 6 -> class 5
]
NF = 8                                       # features: 7 best-fit correlations + colour
W = {}
layers = [{"name": "pool", "op": "avgpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]}]

# brightness L = (R+G+B)/3 and colour D = B - 0.9 R (positive = blue), then crop 64 -> 48 with identity 5x5 convs
w = np.zeros((2, 3, 5, 5)); w[0, :, 2, 2] = 1.0 / 3.0; w[1, 0, 2, 2] = -0.9; w[1, 2, 2, 2] = 1.0
W["mix.W"], W["mix.b"] = w, np.zeros(2)
layers.append({"name": "mix", "op": "conv", "inputs": ["pool"], "out": 2, "kernel": [5, 5], "stride": 1, "padding": 0})
prev = "mix"
for i in range(3):
    w = np.zeros((2, 2, 5, 5)); w[0, 0, 2, 2] = 1.0; w[1, 1, 2, 2] = 1.0
    W[f"crop{i}.W"], W[f"crop{i}.b"] = w, np.zeros(2)
    layers.append({"name": f"crop{i}", "op": "conv", "inputs": [prev], "out": 2, "kernel": [5, 5], "stride": 1, "padding": 0})
    prev = f"crop{i}"
layers.append({"name": "nrm", "op": "norm", "inputs": [prev]})

embeds = []
for gi, (name, T) in enumerate(groups):
    K = len(T)
    w = np.zeros((K, 2 * N))
    w[:, :N] = np.stack([t.ravel() for t in T])          # brightness channel only
    W[f"t_{name}.W"], W[f"t_{name}.b"] = w, np.zeros(K)
    layers.append({"name": f"t_{name}", "op": "dense", "inputs": ["nrm"], "out": K})
    layers.append({"name": f"r_{name}", "op": "reshape", "inputs": [f"t_{name}"], "shape": [1, 1, K]})
    layers.append({"name": f"m_{name}", "op": "maxpool", "inputs": [f"r_{name}"], "size": [1, K], "stride": [1, K]})
    e = np.zeros((NF, 1)); e[gi, 0] = 1.0
    W[f"e_{name}.W"], W[f"e_{name}.b"] = e, np.zeros(NF)
    layers.append({"name": f"e_{name}", "op": "dense", "inputs": [f"m_{name}"], "out": NF})
    embeds.append(f"e_{name}")

# colour cue: centre-weighted mean of the normalised blue-minus-red map
win = np.exp(-0.5 * (X ** 2 + Y ** 2) / 5.0 ** 2); win = win / win.sum()
e = np.zeros((NF, 2 * N)); e[7, N:] = win.ravel()
W["e_col.W"], W["e_col.b"] = e, np.zeros(NF)
layers.append({"name": "e_col", "op": "dense", "inputs": ["nrm"], "out": NF})
embeds.append("e_col")
layers.append({"name": "feat", "op": "add", "inputs": embeds})

# rule layer.  hidden: h0..h6 = r_i + 1, h7 = relu(r_round - r_mid), h8 = relu(c + 1), h9 = relu(c - 1)
#   -> clipped colour cb = h8 - h9 - 1 in [-1, 1];  face-on fit rface = max(r_round, r_mid) = h1 - 1 + h7
NH = 10
hw, hb = np.zeros((NH, NF)), np.zeros(NH)
for i in range(7):
    hw[i, i] = 1.0; hb[i] = 1.0
hw[7, 0], hw[7, 1] = 1.0, -1.0
hw[8, 7], hb[8] = 1.0, 1.0
hw[9, 7], hb[9] = 1.0, -1.0
W["hid.W"], W["hid.b"] = hw, hb
layers.append({"name": "hid", "op": "dense", "inputs": ["feat"], "out": NH})
layers.append({"name": "hr", "op": "relu", "inputs": ["hid"]})

G = 40.0        # points per unit of correlation
WC = 3.0        # points for a fully blue (+) or fully red (-) galaxy
KL = 10.0       # loose spirals fit a smooth model worse than tight ones
TAU = 0.9
T0 = 0.5        # "nothing fits": disturbed
prior = {0: 0.0, 1: -1.0, 2: 0.3, 3: 0.0, 4: -1.5, 5: -0.5, 6: -0.5, 7: -0.2, 8: 0.0, 9: 0.0}  # cigar is rare
ow, ob = np.zeros((10, NH)), np.zeros(10)


def fit(cls, h, gain=G):            # + gain * r_h   (h_h = r_h + 1)
    ow[cls, h] += gain; ob[cls] -= gain


def colour(cls, wt):                # + wt * cb
    ow[cls, 8] += wt; ow[cls, 9] -= wt; ob[cls] -= wt


ob[0] += G * T0; colour(0, 0.5 * WC)                      # disturbed
fit(1, 5)                                                 # merging: two cores
fit(2, 0); colour(2, -WC)                                 # round smooth
fit(3, 1); colour(3, -WC)                                 # in-between smooth
fit(4, 2); colour(4, -WC)                                 # cigar smooth
fit(5, 6); colour(5, 0.3 * WC)                            # barred spiral
fit(6, 1); ow[6, 7] += G; colour(6, WC)                   # tight spiral: face-on disk, blue
fit(7, 1, G - KL); ow[7, 7] += G - KL; ob[7] += KL * TAU; colour(7, WC)   # loose spiral: blue, lumpier
fit(8, 3); colour(8, 0.5 * WC)                            # edge-on, no bulge: thin, bluish
fit(9, 4); colour(9, -0.5 * WC)                           # edge-on with bulge
for k, p in prior.items():
    ob[k] += p
W["out.W"], W["out.b"] = ow, ob
layers.append({"name": "out", "op": "dense", "inputs": ["hr"], "out": 10})

with open(os.path.join(out_dir, "network.json"), "w") as f:
    json.dump({"layers": layers}, f)
np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
