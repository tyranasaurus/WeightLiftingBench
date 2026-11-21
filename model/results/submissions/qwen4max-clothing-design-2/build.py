"""Hand-designed Fashion-MNIST classifier. Writes network.json + weights.npz into argv[1].

Every number comes from formulas (Gaussian/center-surround/Laplacian kernels, region
indicators on the canonical 28x28 centred canvas) and a hand-written table of what each
class looks like in general. No training data is read; nothing is fitted.

Architecture:
  image -> soft foreground mask M = relu(image - 0.18)
  structural features from M: regional means / ink fractions (dense), occupied row/col
    extents and per-band widths (avgpool + saturating threshold + pool)
  detectors on image: enclosed-hole (center-surround), vertical dark line, Laplacian
    texture; hole / opening aggregated by region MAX (presence), texture by region mean
  -> feature vector F (28)
  -> stage 1: saturating part / group gates S in [0,1] (shoe, tall-garment, bag-handle,
     leg-gap, flare, long-sleeve, front-opening, texture, sparse, holey, shaft, solid)
  -> stage 2: signed linear combination of gates per class (last layer). A class only
     scores inside its own group; wrong-group evidence is driven negative.
All thresholds / weights are hand-chosen from general knowledge of the classes.
"""
import sys, json
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
H = W = 28
N = H * W

layers = []
weights = {}


def layer(name, op, inputs, **settings):
    L = {"name": name, "op": op, "inputs": inputs}
    L.update(settings)
    layers.append(L)
    return name


def setw(name, W=None, b=None):
    if W is not None:
        weights[name + ".W"] = np.asarray(W, np.float32)
    if b is not None:
        weights[name + ".b"] = np.asarray(b, np.float32)


def rect_mask(boxes):
    m = np.zeros((H, W), np.float64)
    for (y0, y1, x0, x1) in boxes:
        m[y0:y1 + 1, x0:x1 + 1] = 1.0
    return m


def region_row(boxes, mode):
    m = rect_mask(boxes)
    if mode == "mean":
        return (m / m.sum()).reshape(-1)
    elif mode == "frac":
        return (m / N).reshape(-1)
    elif mode == "sum":
        return m.reshape(-1)
    raise ValueError(mode)


def placement(n_in, n_out, mapping):
    P = np.zeros((n_out, n_in), np.float64)
    for o, i in mapping.items():
        P[o, i] = 1.0
    return P


# ---------------------------------------------------------------- mask
layer("mask_conv", "conv", ["image"], out=1, kernel=[1, 1], stride=1, padding=0)
setw("mask_conv", np.ones((1, 1, 1, 1)), [-0.18])
layer("mask", "relu", ["mask_conv"])

# ---------------------------------------------------------------- occupied extents
def saturate(inp, c, K, t, prefix):
    """min(1, relu(K*(x - t))) per channel, via conv/relu/conv/relu/conv."""
    eye = np.eye(c).reshape(c, c, 1, 1)
    layer(prefix + "_s1", "conv", [inp], out=c, kernel=[1, 1], stride=1, padding=0)
    setw(prefix + "_s1", K * eye, np.full(c, -K * t))
    layer(prefix + "_r1", "relu", [prefix + "_s1"])
    layer(prefix + "_s2", "conv", [prefix + "_r1"], out=c, kernel=[1, 1], stride=1, padding=0)
    setw(prefix + "_s2", -eye, np.ones(c))
    layer(prefix + "_r2", "relu", [prefix + "_s2"])
    layer(prefix + "_s3", "conv", [prefix + "_r2"], out=c, kernel=[1, 1], stride=1, padding=0)
    setw(prefix + "_s3", -eye, np.ones(c))
    return prefix + "_s3"


layer("rowmean", "avgpool", ["mask"], size=[1, 28], stride=[1, 28])          # (1,28,1)
rowcap = saturate("rowmean", 1, 8.0, 0.04, "rowocc")
layer("height_frac", "avgpool", [rowcap], size=[28, 1], stride=[28, 1])      # (1,1,1)

layer("colmean", "avgpool", ["mask"], size=[28, 1], stride=[28, 1])          # (1,1,28)
colcap = saturate("colmean", 1, 8.0, 0.04, "colocc")
layer("width_frac", "avgpool", [colcap], size=[1, 28], stride=[1, 28])       # (1,1,1)

# per-band occupied widths (top / mid / bottom)
bands = [(2, 11), (10, 19), (16, 27)]
Bs = np.zeros((3 * 28, N), np.float64)
for bi, (y0, y1) in enumerate(bands):
    for x in range(28):
        Bs[bi * 28 + x] = region_row([(y0, y1, x, x)], "sum")
layer("band_sums", "dense", ["mask"], out=84)
setw("band_sums", Bs, np.zeros(84))
layer("band_res", "reshape", ["band_sums"], shape=[3, 1, 28])
bandcap = saturate("band_res", 3, 2.0, 1.0, "bandocc")
Bw = np.zeros((3, 84), np.float64)
for bi in range(3):
    Bw[bi, bi * 28:(bi + 1) * 28] = 1.0 / 28
layer("band_w", "dense", [bandcap], out=3)
setw("band_w", Bw, np.zeros(3))

# ---------------------------------------------------------------- regional features (19)
REG = [
    ([(0, 27, 0, 27)], "mean"),   # 0 total_mean
    ([(0, 13, 0, 27)], "mean"),   # 1 top_mean
    ([(14, 27, 0, 27)], "mean"),  # 2 bot_mean
    ([(14, 27, 0, 27)], "frac"),  # 3 bot_sum_frac
    ([(0, 13, 0, 27)], "frac"),   # 4 top_sum_frac
    ([(16, 27, 11, 16)], "mean"),                          # 5 center_bot_mean
    ([(16, 27, 4, 10), (16, 27, 17, 23)], "mean"),         # 6 side_bot_mean
    ([(12, 20, 0, 6), (12, 20, 21, 27)], "mean"),          # 7 side_mid_mean
    ([(3, 9, 0, 6), (3, 9, 21, 27)], "mean"),              # 8 shoulder_mean
    ([(8, 22, 8, 19)], "mean"),   # 9 body_mean
    ([(1, 8, 9, 18)], "mean"),    # 10 topcenter_mean
    ([(0, 27, 0, 13)], "mean"),   # 11 left_mean
    ([(0, 27, 14, 27)], "mean"),  # 12 right_mean
    ([(3, 10, 0, 27)], "mean"),   # 13 topmost_mean (boot shaft / bag handle zone)
    ([(6, 12, 8, 19)], "mean"),   # 14 waist_mean
    ([(22, 27, 0, 27)], "mean"),  # 15 hem_mean
    ([(2, 10, 0, 7), (2, 10, 20, 27)], "mean"),            # 16 upper_side_mean
    ([(14, 22, 0, 7), (14, 22, 20, 27)], "mean"),          # 17 lower_side_mean
    ([(4, 14, 0, 27)], "mean"),   # 18 upper_mean
]
R = np.stack([region_row(b, m) for (b, m) in REG])
layer("reg", "dense", ["mask"], out=len(REG))
setw("reg", R, np.zeros(len(REG)))

# ---------------------------------------------------------------- detectors on image
def center_surround():
    k = np.zeros((9, 9), np.float64)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            r = (dy * dy + dx * dx) ** 0.5
            if r < 1.6:
                k[dy + 4, dx + 4] = -1.0
            elif 2.4 < r < 4.4:
                k[dy + 4, dx + 4] = 0.3
    pos = k[k > 0].sum()
    return k / pos


def vertical_dark_line():
    k = np.zeros((9, 9), np.float64)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if dx == 0:
                k[dy + 4, dx + 4] = -1.0
            elif abs(dx) in (1, 2):
                k[dy + 4, dx + 4] = 0.5
    pos = k[k > 0].sum()
    return k / pos


LAP = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float64) * 0.25

layer("hole_conv", "conv", ["image"], out=1, kernel=[9, 9], stride=1, padding=4)
setw("hole_conv", center_surround().reshape(1, 1, 9, 9), [0.0])
layer("hole", "relu", ["hole_conv"])
layer("open_conv", "conv", ["image"], out=1, kernel=[9, 9], stride=1, padding=4)
setw("open_conv", vertical_dark_line().reshape(1, 1, 9, 9), [0.0])
layer("open", "relu", ["open_conv"])
layer("tex_conv", "conv", ["image"], out=1, kernel=[3, 3], stride=1, padding=1)
setw("tex_conv", LAP.reshape(1, 1, 3, 3), [0.0])
layer("tex", "relu", ["tex_conv"])

# Presence (max) aggregation: a thin handle loop / placket line must not be diluted
# by averaging over a whole region.
layer("hole_top", "maxpool", ["hole"], size=[10, 20], stride=[10, 20])   # (1,2,1)
layer("handle_hole", "dense", ["hole_top"], out=1)
setw("handle_hole", np.array([[1.0, 0.0]]), np.zeros(1))
layer("hole_allp", "maxpool", ["hole"], size=[28, 28], stride=[28, 28])  # (1,1,1)
layer("hole_all", "reshape", ["hole_allp"], shape=[1])
layer("open_maxp", "maxpool", ["open"], size=[21, 18], stride=[21, 18])  # (1,1,1)
layer("opening", "reshape", ["open_maxp"], shape=[1])
layer("tex_feat", "dense", ["tex"], out=1)
setw("tex_feat", region_row([(8, 22, 8, 19)], "mean").reshape(1, -1), np.zeros(1))

# ---------------------------------------------------------------- assemble F (28)
layer("wf_res", "reshape", ["width_frac"], shape=[1])
layer("hf_res", "reshape", ["height_frac"], shape=[1])
layer("ext_wf", "dense", ["wf_res"], out=5); setw("ext_wf", placement(1, 5, {0: 0}), np.zeros(5))
layer("ext_hf", "dense", ["hf_res"], out=5); setw("ext_hf", placement(1, 5, {1: 0}), np.zeros(5))
layer("ext_bw", "dense", ["band_w"], out=5); setw("ext_bw", placement(3, 5, {2: 0, 3: 1, 4: 2}), np.zeros(5))
layer("ext", "add", ["ext_wf", "ext_hf", "ext_bw"])

layer("det_hh", "dense", ["handle_hole"], out=4); setw("det_hh", placement(1, 4, {0: 0}), np.zeros(4))
layer("det_ha", "dense", ["hole_all"], out=4); setw("det_ha", placement(1, 4, {1: 0}), np.zeros(4))
layer("det_o", "dense", ["opening"], out=4); setw("det_o", placement(1, 4, {2: 0}), np.zeros(4))
layer("det_t", "dense", ["tex_feat"], out=4); setw("det_t", placement(1, 4, {3: 0}), np.zeros(4))
layer("det", "add", ["det_hh", "det_ha", "det_o", "det_t"])

layer("F_reg", "dense", ["reg"], out=28); setw("F_reg", placement(19, 28, {i: i for i in range(19)}), np.zeros(28))
layer("F_ext", "dense", ["ext"], out=28); setw("F_ext", placement(5, 28, {19 + i: i for i in range(5)}), np.zeros(28))
layer("F_det", "dense", ["det"], out=28); setw("F_det", placement(4, 28, {24 + i: i for i in range(4)}), np.zeros(28))
layer("F", "add", ["F_reg", "F_ext", "F_det"])

# ---------------------------------------------------------------- cascade readout
# Stage 1: saturating part / group gates S in [0,1] computed from the raw features F.
# Each unit is min(1, relu(K*(coef.F + c0 - t))). Stage 2 combines them with signed
# weights so a class only scores inside its own group (wrong-group evidence < 0).
# F indices: 0 total,5 center_bot,6 side_bot,7 side_mid,13 topmost,19 width_frac,
#            20 height_frac,21 width_top,23 width_bot,24 handle_hole,25 hole_all,
#            26 opening,27 texture
UNITS = [
    ({19: 2.0, 20: -1.6}, -0.3, 3.0, 0.10),   # 0 shoe   (wide & short)
    ({20: 3.6, 19: -2.0}, -1.1, 3.0, 0.10),   # 1 tall garment (tall, not shoe)
    ({24: 2.0}, -0.3, 3.0, 0.10),             # 2 bag    (handle loop hole)
    ({6: 2.0, 5: -2.0}, -0.2, 3.0, 0.05),     # 3 leg gap
    ({23: 2.0, 21: -2.0}, -0.1, 3.0, 0.10),   # 4 flare  (wide hem)
    ({7: 2.0}, -0.3, 3.0, 0.10),              # 5 long sleeves
    ({26: 2.0}, -0.3, 3.0, 0.05),             # 6 front opening
    ({27: 2.0}, -0.1, 3.0, 0.05),             # 7 texture (pattern / knit)
    ({0: -2.0}, 0.5, 3.0, 0.05),              # 8 sparse (low coverage)
    ({25: 2.0}, -0.35, 3.0, 0.05),            # 9 holey (strong strap gaps only)
    ({13: 4.0}, -0.2, 3.0, 0.0),              # 10 shaft (top-zone mass: boot shaft)
    ({0: 2.0}, -0.3, 3.0, 0.05),              # 11 solid (high coverage)
]
NU = len(UNITS)
Cw = np.zeros((NU, 28), np.float64)
Cb = np.zeros(NU, np.float64)
for ui, (coef, c0, K, t) in enumerate(UNITS):
    for j, v in coef.items():
        Cw[ui, j] = K * v
    Cb[ui] = K * (c0 - t)
layer("s1", "dense", ["F"], out=NU)
setw("s1", Cw, Cb)
layer("s1r", "relu", ["s1"])


def vcap(inp, n, prefix):
    eye = np.eye(n)
    layer(prefix + "_c1", "dense", [inp], out=n); setw(prefix + "_c1", -eye, np.ones(n))
    layer(prefix + "_r", "relu", [prefix + "_c1"])
    layer(prefix + "_c2", "dense", [prefix + "_r"], out=n); setw(prefix + "_c2", -eye, np.ones(n))
    return prefix + "_c2"


S = vcap("s1r", NU, "s1cap")

# Stage 2: class evidence = signed sum of gates - threshold (last layer, linear scores).
EVID = [
    ({1: 1, 5: -1, 6: -1, 3: -1, 4: -1}, 0.5),          # 0 T-shirt
    ({1: 1, 3: 1, 4: -1, 5: -1, 6: -1}, 1.0),           # 1 trouser
    ({1: 1, 5: 1, 6: -1, 3: -1, 4: -1, 7: -1}, 1.0),    # 2 pullover
    ({1: 1, 4: 1, 3: -1, 5: -0.5, 2: -1}, 1.0),         # 3 dress
    ({1: 1, 5: 1, 6: 1, 7: -2, 3: -1, 4: -1}, 1.5),     # 4 coat
    ({0: 1, 8: 1, 9: 1, 11: -1, 10: -1}, 1.0),          # 5 sandal
    ({1: 1, 6: 0.5, 7: 2, 5: 0.5, 3: -1, 4: -1}, 2.0),  # 6 shirt
    ({0: 1, 11: 1, 8: -1, 9: -1, 10: -1}, 1.0),         # 7 sneaker
    ({2: 1, 0: -1, 1: -0.5}, 0.3),                      # 8 bag
    ({0: 1, 10: 1, 11: 0.5, 8: -1, 9: -1}, 1.0),        # 9 ankle boot
]
Ew = np.zeros((10, NU), np.float64)
Eb = np.zeros(10, np.float64)
for c, (coef, T) in enumerate(EVID):
    for j, v in coef.items():
        Ew[c, j] = v
    Eb[c] = -T
layer("readout", "dense", [S], out=10)
setw("readout", Ew, Eb)

# ---------------------------------------------------------------- write
net = {"layers": layers}
with open(OUT + "/network.json", "w") as f:
    json.dump(net, f, indent=1)
np.savez(OUT + "/weights.npz", **weights)
print("wrote", len(layers), "layers,", sum(v.size for v in weights.values()), "weights")
