"""Hand-designed person / no-person network for 96 x 96 photos.  Usage: python3 build.py OUT_DIR

Every weight below comes from a formula or from a constant typed by hand from general knowledge about how
people look in photographs (skin colour ranges, face and body proportions, how clothing contrasts with the
background).  Nothing is read from disk and nothing is fitted: there is no training data, no optimiser, no
statistics of images.  The program is deterministic (it uses no random numbers).

Evidence for "person" comes from three kinds of detectors, each a small stack of conv / relu / pool layers:

  1. SKIN BLOB   a compact skin-coloured blob (face, hand) with little skin colour in the ring around it, at
                 seven sizes.  Ignored in scenes dominated by warm colours (wood, food, lamp light), where
                 skin colour says little.
  2. FACE        a larger skin blob that also has a darker eye band between forehead and cheeks and darker
                 hair above it.  Not subject to the warm-scene rule.
  3. FIGURE      a standing figure: an upright (or leaning) ribbon about 2.6 body-widths long that is brighter /
                 darker / redder / ... than the background on BOTH sides, ends above and below, has nothing
                 similar beside it, and changes colour between its upper and lower half (clothing).

The final layer says "person" when any of them fires.  Exactly grey (black-and-white) photos, where colour
evidence cannot exist, are called "person" (black-and-white photography is mostly of people).
"""
import json, os, sys
import numpy as np

# =============================================================================================== constants
# ---- skin colour: a wedge in RGB (hue between red and orange, moderate saturation, not dark)
HUE_LO, HUE_HI = 6.0, 36.0            # degrees
SAT_LO, SAT_HI = 0.18, 0.68           # (max - min) / max
VAL_LO = 0.33                         # red channel at least this
SOFT = 12.0                           # membership falls from 1 to 0 over a violation of 1 / SOFT (RGB units)
# ---- skin blob templates
LAMBDA, THETA, K = 2.0, 0.5, 13       # ring weight, required excess of core over ring, kernel size
BLOB_SIZES = (1.5, 2.12)              # face half-widths in pixels of the pyramid level
# ---- decision constants for the skin blob
F0, ETA, GAMMA = 0.20, 0.30, 2.0      # warm-scene rule: starts at warm fraction F0 (+ ETA * green/blue fraction), slope GAMMA
TAU = 0.17                            # blob evidence needed
ZONE_PEN = (0.0, 0.0, 0.10)           # top / middle / bottom third of the picture (faces are rarely at the bottom)
#            a = 2.1   3    4.2   6    8.5   12    17   (face half-width in pixels)
SCALE_PEN = (0.05, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10)
# ---- structured face
FACE = dict(
    EYE_T=0.03, EYE_SAT=0.05,         # eye band darker than forehead AND cheeks by EYE_T .. EYE_T + EYE_SAT (luma)
    HAIR_T=0.06, HAIR_SAT=0.10,       # hair darker than the middle of the face by HAIR_T .. HAIR_T + HAIR_SAT
    BLOB_SAT=0.15,                    # blob response that counts as a full skin blob
    NEED=2.4,                         # blob + hair + eyes (each 0..1) must reach this
)
SF_MIN = 0.05
# ---- standing figure
OPP = np.array([[0.299, 0.587, 0.114],      # luma
                [1.0, -1.0, 0.0],           # red - green
                [-0.5, -0.5, 1.0]])         # blue - yellow
WT = 3.75                                   # template unit (level pixels); nominal body width = 1.2 WT
KR = 11                                     # ridge kernel size
FIG = dict(
    RIDGE_T=(0.12, 0.09, 0.09),             # contrast needed in luma, red-green, blue-yellow
    RIDGE_SAT=0.05,                         # a ridge counts fully once it exceeds its threshold by this much
    FIG_COVER=0.60,                         # share of the axis that must be ridge
    FIG_END=0.5,                            # weight of ridge found beyond either end (poles, trunks)
    FIG_SIDE=0.7,                           # weight of ridge found beside the figure (fences, clutter)
    DIV_MIN=0.5, DIV_W=1.0,                 # required change of ridge type between upper and lower half; penalty weight
    ROWS=0.6,                               # half height of the ridge patches (level pixels)
)
FIG_LEVELS = (2, 3, 4, 6)                   # pooling factors: body widths of about 9, 13, 18, 27 pixels
FIG_MIN = 0.05
SLANTS = (-0.5, 0.0, 0.5)                   # lean of the figure axis (columns per row)
WN = 1.2 * WT / 2.0                         # nominal body width in pooled units (2 level pixels)
HH = 1.3 * WN                               # half length of the figure axis
E0 = 2.1 * WN + 0.25; E1 = E0 + 2.0         # end windows start beyond the tallest figure
S0, S1 = 1.2 * WN, 2.4 * WN                 # side zones
KF = 15                                     # figure kernel size


# =============================================================================================== graph builder
class Net:
    def __init__(self):
        self.L, self.Wt, self.shape = [], {}, {"image": (3, 96, 96)}

    def conv(self, name, inp, W, b, stride=1, padding=0):
        W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
        c, h, w = self.shape[inp]
        assert W.shape[1] == c, (name, W.shape, self.shape[inp])
        k = W.shape[2]
        self.L.append({"name": name, "op": "conv", "inputs": [inp], "out": int(W.shape[0]), "kernel": [k, k],
                       "stride": stride, "padding": padding})
        self.Wt[name + ".W"], self.Wt[name + ".b"] = W.astype(np.float32), b.astype(np.float32)
        self.shape[name] = (W.shape[0], (h + 2 * padding - k) // stride + 1, (w + 2 * padding - k) // stride + 1)
        return name

    def dense(self, name, inp, W, b):
        W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
        assert W.shape[1] == int(np.prod(self.shape[inp])), (name, W.shape, self.shape[inp])
        self.L.append({"name": name, "op": "dense", "inputs": [inp], "out": int(W.shape[0])})
        self.Wt[name + ".W"], self.Wt[name + ".b"] = W.astype(np.float32), b.astype(np.float32)
        self.shape[name] = (W.shape[0],)
        return name

    def relu(self, name, inp):
        self.L.append({"name": name, "op": "relu", "inputs": [inp]}); self.shape[name] = self.shape[inp]; return name

    def pool(self, name, kind, inp, size, stride):
        c, h, w = self.shape[inp]
        self.L.append({"name": name, "op": kind, "inputs": [inp], "size": list(size), "stride": list(stride)})
        self.shape[name] = (c, (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1)
        return name

    def add(self, name, inputs):
        assert all(self.shape[i] == self.shape[inputs[0]] for i in inputs), [self.shape[i] for i in inputs]
        self.L.append({"name": name, "op": "add", "inputs": list(inputs)}); self.shape[name] = self.shape[inputs[0]]; return name

    def reshape(self, name, inp, shape):
        assert int(np.prod(shape)) == int(np.prod(self.shape[inp]))
        self.L.append({"name": name, "op": "reshape", "inputs": [inp], "shape": list(shape)}); self.shape[name] = tuple(shape); return name

    def concat_vec(self, name, inputs):
        """Concatenate flat tensors into one vector: embed each with a dense selection matrix, then add."""
        sizes = [int(np.prod(self.shape[i])) for i in inputs]
        tot, off, parts = sum(sizes), 0, []
        for j, (i, n) in enumerate(zip(inputs, sizes)):
            Wd = np.zeros((tot, n)); Wd[off:off + n] = np.eye(n); off += n
            parts.append(self.dense(f"{name}_p{j}", i, Wd, np.zeros(tot)))
        return self.add(name, parts) if len(parts) > 1 else parts[0]

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "network.json"), "w") as f:
            json.dump({"layers": self.L}, f, indent=1)
        np.savez(os.path.join(out_dir, "weights.npz"), **self.Wt)


# =============================================================================================== shapes (anti-aliased masks)
def _grid(k, ss):
    c = (k - 1) / 2.0
    g = (np.arange(k * ss) + 0.5) / ss - 0.5 - c
    return np.meshgrid(g, g, indexing="ij")


def ell(k, ax, ay, ss=8):
    """Ellipse mask (k, k), centred, semi-axes ax (horizontal), ay (vertical)."""
    yy, xx = _grid(k, ss)
    m = ((xx / ax) ** 2 + (yy / ay) ** 2) <= 1.0
    return m.reshape(k, ss, k, ss).mean(axis=(1, 3))


def box(k, x0, x1, y0, y1, ss=8):
    """Rectangle mask (k, k) in coordinates relative to the kernel centre (pixel i covers [i - .5, i + .5])."""
    yy, xx = _grid(k, ss)
    m = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    return m.reshape(k, ss, k, ss).mean(axis=(1, 3))


def shear_box(k, x0, x1, y0, y1, s, ss=8):
    """Rectangle whose column offset is measured from the leaning axis x = s * y."""
    yy, xx = _grid(k, ss)
    u = xx - s * yy
    m = (u >= x0) & (u <= x1) & (yy >= y0) & (yy <= y1)
    return m.reshape(k, ss, k, ss).mean(axis=(1, 3))


def sector_weight(k, half_angle_deg, inside, ss=8):
    """Weight map: `inside` within half_angle of straight down from the centre, 1 elsewhere."""
    yy, xx = _grid(k, ss)
    ang = np.degrees(np.arctan2(np.abs(xx), yy))
    w = np.where(ang < half_angle_deg, inside, 1.0)
    return w.reshape(k, ss, k, ss).mean(axis=(1, 3))


def unit(m):
    return m / m.sum()


def safe_unit(m):
    return m / max(m.sum(), 1e-9)


# =============================================================================================== colour stage
def colour_stage(n):
    """1x1 convolutions: soft memberships of skin / warm / green / blue, chroma magnitude, luma."""
    tl, th = HUE_LO / 60.0, HUE_HI / 60.0
    Y = [0.299, 0.587, 0.114]
    rows = [
        # skin wedge violations:       R      G      B         bias
        ([tl, -1.0, 1.0 - tl], 0.0),                  # hue below HUE_LO:  tl (R-B) - (G-B)
        ([-th, 1.0, th - 1.0], 0.0),                  # hue above HUE_HI:  (G-B) - th (R-B)
        ([SAT_LO - 1.0, 0.0, 1.0], 0.0),              # too grey:          SAT_LO R - (R-B)
        ([1.0 - SAT_HI, 0.0, -1.0], 0.0),             # too saturated:     (R-B) - SAT_HI R
        ([-1.0, 0.0, 0.0], VAL_LO),                   # too dark
        # warm (red .. yellow, saturation above about 0.2): violations
        ([-0.85, 0.0, 1.0], 0.03), ([-0.15, -1.0, 1.15], 0.0), ([-1.0, 1.0, 0.0], -0.02), ([-1.0, 0.0, 0.0], 0.20),
        # vegetation green: violations
        ([1.0, -1.0, 0.0], 0.03), ([0.0, -1.0, 1.0], 0.05), ([-Y[0], -Y[1], -Y[2]], 0.12),
        # sky / water blue: violations
        ([1.0, 0.0, -1.0], 0.06), ([0.0, 1.0, -1.0], -0.02), ([-Y[0], -Y[1], -Y[2]], 0.30),
        # chroma magnitude pieces |R-G|, |G-B|
        ([1.0, -1.0, 0.0], 0.0), ([-1.0, 1.0, 0.0], 0.0), ([0.0, 1.0, -1.0], 0.0), ([0.0, -1.0, 1.0], 0.0),
        # luma
        (Y, 0.0),
    ]
    W1 = np.array([r[0] for r in rows])[:, :, None, None]; b1 = np.array([r[1] for r in rows])
    n.conv("c1", "image", W1, b1); n.relu("r1", "c1")
    Ws = np.zeros((1, 20, 1, 1)); Ws[0, 0:5] = -SOFT
    n.conv("c2s", "r1", Ws, [1.0]); n.relu("skin0", "c2s")                         # skin membership in [0, 1]
    Wg = np.zeros((4, 20, 1, 1)); bg = np.array([1.0, 1.0, 1.0, 0.0])
    Wg[0, 5:9] = -SOFT; Wg[1, 9:12] = -SOFT; Wg[2, 12:15] = -SOFT; Wg[3, 15:19] = 1.0
    n.conv("c2g", "r1", Wg, bg); n.relu("glob0", "c2g")                            # warm, green, blue, chroma
    n.pool("gstat", "avgpool", "glob0", [96, 96], [96, 96])                        # their shares of the picture


# =============================================================================================== skin blobs
def blob_template(a, k=K):
    """Core ellipse (should be skin) minus a surrounding ring (should not be); the ring counts less below (neck)."""
    core = unit(ell(k, 0.85 * a, 1.1 * a))
    ring = unit(np.clip(ell(k, 2.6 * a, 3.0 * a) - ell(k, 1.6 * a, 2.0 * a), 0, None) * sector_weight(k, 50.0, 0.4))
    return core - LAMBDA * ring


def blob_stage(n):
    n.pool("skin1", "avgpool", "skin0", [2, 2], [2, 2])
    n.pool("skin2", "avgpool", "skin1", [2, 2], [2, 2])
    n.pool("skin3", "avgpool", "skin2", [2, 2], [2, 2])
    plan = [("skin0", 2, (2.12,)), ("skin1", 1, BLOB_SIZES), ("skin2", 1, BLOB_SIZES), ("skin3", 1, BLOB_SIZES)]
    outs = []
    for li, (src, stride, sizes) in enumerate(plan):
        Wb = np.stack([blob_template(a) for a in sizes])[:, None]
        n.conv(f"blob{li}", src, Wb, np.full(len(sizes), -THETA), stride=stride, padding=K // 2)
        n.relu(f"blob{li}r", f"blob{li}")
        hw = n.shape[f"blob{li}r"][1]
        outs.append(n.pool(f"blob{li}z", "maxpool", f"blob{li}r", [hw // 3, hw], [hw // 3, hw]))   # (sizes, 3 zones, 1)
    return outs


# =============================================================================================== structured face
def eye_kernels(a, k=7):
    """(forehead - eye band, forehead - cheeks) for a face of half-width a (pixels of the finer level)."""
    t = 0.4 * a; hw = 0.75 * a
    eye = unit(box(k, -hw, hw, -t / 2, t / 2))
    fore = unit(box(k, -hw, hw, -t / 2 - 0.45 * a, -t / 2))
    cheek = unit(box(k, -hw, hw, t / 2, t / 2 + 0.5 * a))
    return fore - eye, fore - cheek


def hair_kernels(a, k=9):
    """Middle of the face minus the band above the forehead: (round face, tall face), relative to the blob centre."""
    hw = 0.6 * a
    centre = unit(box(k, -hw, hw, -0.3 * a, 0.5 * a))
    round_ = unit(box(k, -hw, hw, -1.5 * a, -0.95 * a))
    tall = unit(box(k, -hw, hw, -2.3 * a, -1.5 * a))
    return centre - round_, centre - tall


def face_branch(n, blob_r, luma_fine, luma_here, tag):
    """blob_r: relu'd blob maps at one level (channels = BLOB_SIZES). Returns (2, 1, 1): best structured face per size."""
    P = FACE
    We = np.zeros((4, 1, 7, 7))
    for j, a in enumerate(BLOB_SIZES):
        We[2 * j, 0], We[2 * j + 1, 0] = eye_kernels(2 * a)
    n.conv(f"{tag}_e", luma_fine, We, np.zeros(4), padding=3); n.relu(f"{tag}_er", f"{tag}_e")
    # min(forehead - eyes, cheeks - eyes) - T  =  relu(f - e) - relu(f - c) - T
    n.conv(f"{tag}_em", f"{tag}_er", np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])[:, :, None, None],
           np.full(2, -P["EYE_T"])); n.relu(f"{tag}_emr", f"{tag}_em")
    s = 1.0 / P["EYE_SAT"]
    n.conv(f"{tag}_ec", f"{tag}_emr", np.array([[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]])[:, :, None, None], [0.0, -1.0, 0.0, -1.0])
    n.relu(f"{tag}_ecr", f"{tag}_ec")
    n.conv(f"{tag}_es", f"{tag}_ecr", np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])[:, :, None, None], [0.0, 0.0])
    n.pool(f"{tag}_ep", "maxpool", f"{tag}_es", [2, 2], [2, 2])                    # onto the blob grid, values in [0, 1]
    Wn = np.zeros((8, 2, 3, 3))                                                    # eye band in the row above the blob centre
    for j in range(2):
        Wn[2 + j, j, 0:2, :] = 1.0
    n.conv(f"{tag}_en", f"{tag}_ep", Wn, np.zeros(8), padding=1)                   # channels 2, 3
    Wh = np.zeros((8, 1, 9, 9)); bh = np.zeros(8)
    for j, a in enumerate(BLOB_SIZES):
        r, t = hair_kernels(a)
        Wh[4 + 2 * j, 0], Wh[5 + 2 * j, 0] = r / P["HAIR_SAT"], t / P["HAIR_SAT"]
        bh[4 + 2 * j] = bh[5 + 2 * j] = -P["HAIR_T"] / P["HAIR_SAT"]
    n.conv(f"{tag}_h", luma_here, Wh, bh, padding=4)                               # channels 4 .. 7
    Wb = np.zeros((8, 2, 1, 1)); Wb[0, 0] = Wb[1, 1] = 1.0 / P["BLOB_SAT"]
    n.conv(f"{tag}_b", blob_r, Wb, np.zeros(8))                                    # channels 0, 1
    n.add(f"{tag}_cat", [f"{tag}_b", f"{tag}_en", f"{tag}_h"]); n.relu(f"{tag}_catr", f"{tag}_cat")
    Wc = np.zeros((12, 8, 1, 1)); bc = np.zeros(12)                                # clip blob, eyes, hair to [0, 1]
    for j in range(2):
        o = 6 * j
        Wc[o + 0, j] = Wc[o + 1, j] = 1.0; bc[o + 1] = -1.0
        Wc[o + 2, 2 + j] = Wc[o + 3, 2 + j] = 1.0; bc[o + 3] = -1.0
        Wc[o + 4, [4 + 2 * j, 5 + 2 * j]] = 1.0; Wc[o + 5, [4 + 2 * j, 5 + 2 * j]] = 1.0; bc[o + 5] = -1.0
    n.conv(f"{tag}_c", f"{tag}_catr", Wc, bc); n.relu(f"{tag}_cr", f"{tag}_c")
    Wo = np.zeros((2, 12, 1, 1))
    for j in range(2):
        Wo[j, 6 * j:6 * j + 6] = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])[:, None, None]
    n.conv(f"{tag}_o", f"{tag}_cr", Wo, np.full(2, -P["NEED"])); n.relu(f"{tag}_or", f"{tag}_o")
    _, H, W = n.shape[f"{tag}_or"]
    return n.pool(f"{tag}_max", "maxpool", f"{tag}_or", [H, W], [H, W])


# =============================================================================================== standing figure
def ridge_kernels(k=KR):
    """Patch differences: (8, k, k) kernels [Cm-L, L-Cm, Cm-Cl, Cl-Cm, Cm-Cr, Cr-Cm, R-L, L-R]
    (Cm, Cl, Cr: middle / left / right part of the body core; L, R: background flanks)."""
    a, b = 0.12 * WT, 0.36 * WT
    f0, f1 = 0.85 * WT, 1.35 * WT
    r = FIG["ROWS"]
    Cm = unit(box(k, -a, a, -r, r)); Cl = unit(box(k, -b, -a, -r, r)); Cr = unit(box(k, a, b, -r, r))
    Lf = unit(box(k, -f1, -f0, -r, r)); Rf = unit(box(k, f0, f1, -r, r))
    return np.stack([Cm - Lf, Lf - Cm, Cm - Cl, Cl - Cm, Cm - Cr, Cr - Cm, Rf - Lf, Lf - Rf])


def figure_kernels(s, k=KF):
    """Templates on pooled ridge maps: (on the 'any ridge' map, on its widened copy)."""
    centre = safe_unit(shear_box(k, -0.5, 0.5, -HH, HH, s))
    above = safe_unit(shear_box(k, -1.0, 1.0, -E1, -E0, s)); below = safe_unit(shear_box(k, -1.0, 1.0, E0, E1, s))
    side = safe_unit(shear_box(k, -S1, -S0, -HH, HH, s) + shear_box(k, S0, S1, -HH, HH, s))
    return centre - FIG["FIG_SIDE"] * side, -FIG["FIG_END"] * (above + below)


def diversity_kernel(s, k=7):
    return safe_unit(shear_box(k, -0.5, 0.5, -HH, 0, s)) - safe_unit(shear_box(k, -0.5, 0.5, 0, HH, s))


def figure_branch(n, src, tag, const_src, split):
    """src: RGB at one pyramid level, (3, H, W). Returns (3, 1, 1): best figure score for each lean.
    split=True computes the three opponent channels in separate convolutions (less memory at the finest level)."""
    P = FIG
    _, H, W = n.shape[src]
    k = KR
    sp = ridge_kernels(k)
    if split:
        parts = []
        for c in range(3):
            n.conv(f"{tag}_op{c}", src, OPP[c][None, :, None, None], [0.0])
            Wr = np.zeros((24, 1, k, k)); Wr[c * 8:(c + 1) * 8, 0] = sp
            parts.append(n.conv(f"{tag}_d{c}", f"{tag}_op{c}", Wr, np.zeros(24), padding=k // 2))
        n.add(f"{tag}_d", parts)
    else:
        Wr = np.zeros((24, 3, k, k))
        for c in range(3):
            for j in range(8):
                Wr[c * 8 + j] = OPP[c][:, None, None] * sp[j][None]
        n.conv(f"{tag}_d", src, Wr, np.zeros(24), padding=k // 2)
    n.relu(f"{tag}_dr", f"{tag}_d")
    # six ridge types.  (channel c, +): (Cm-L) - relu(Cm-Cl) - relu(Cm-Cr) - relu(R-L) - T   = all three core parts
    # exceed both flanks by T;  (c, -): the mirror image (all three below both flanks)
    W2 = np.zeros((6, 24, 1, 1)); b2 = np.zeros(6)
    for c in range(3):
        o = c * 8
        W2[c * 2, [o + 0, o + 2, o + 4, o + 6]] = np.array([1.0, -1.0, -1.0, -1.0])[:, None, None]
        W2[c * 2 + 1, [o + 1, o + 3, o + 5, o + 7]] = np.array([1.0, -1.0, -1.0, -1.0])[:, None, None]
        b2[c * 2] = b2[c * 2 + 1] = -P["RIDGE_T"][c]
    n.conv(f"{tag}_q", f"{tag}_dr", W2, b2)
    # beyond the picture the flank reads as black, which fakes a ridge: switch off columns next to the left/right edges
    # (a constant map: dense layer with zero weights whose bias is the map)
    m = 5
    mask = np.zeros((6, H, W)); mask[:, :, :m] = -10.0; mask[:, :, W - m:] = -10.0
    n.dense(f"{tag}_maskv", const_src, np.zeros((6 * H * W, int(np.prod(n.shape[const_src])))), mask.reshape(-1))
    n.reshape(f"{tag}_mask", f"{tag}_maskv", [6, H, W])
    n.add(f"{tag}_qm", [f"{tag}_q", f"{tag}_mask"]); n.relu(f"{tag}_r", f"{tag}_qm")
    sat = P["RIDGE_SAT"]                                             # clip each type, and their sum, to [0, 1]
    W3 = np.zeros((14, 6, 1, 1)); b3 = np.zeros(14)
    for t in range(6):
        W3[2 * t, t] = W3[2 * t + 1, t] = 1.0 / sat; b3[2 * t + 1] = -1.0
    W3[12, :] = W3[13, :] = 1.0 / sat; b3[13] = -1.0
    n.conv(f"{tag}_s", f"{tag}_r", W3, b3); n.relu(f"{tag}_sr", f"{tag}_s")
    W4 = np.zeros((7, 14, 1, 1))
    for t in range(7):
        W4[t, 2 * t] = 1.0; W4[t, 2 * t + 1] = -1.0
    n.conv(f"{tag}_u", f"{tag}_sr", W4, np.zeros(7))                 # 6 types + 'any', each in [0, 1]
    n.pool(f"{tag}_uh", "maxpool", f"{tag}_u", [1, 3], [1, 1])       # tolerate a slightly wavy axis
    n.pool(f"{tag}_v", "avgpool", f"{tag}_uh", [2, 2], [2, 2])       # pooled unit = 2 level pixels
    # 'any ridge' and a widened copy (ridge within +-1 column) used for the end windows, so that a long upright
    # pole cannot escape through a leaning template
    Wa = np.zeros((3, 7, 3, 3)); Wa[0, 6, 1, 1] = 1.0; Wa[1, 6, 1, :] = 1.0; Wa[2, 6, 1, :] = 1.0
    n.conv(f"{tag}_a", f"{tag}_v", Wa, [0.0, 0.0, -1.0], padding=1); n.relu(f"{tag}_ar", f"{tag}_a")
    n.conv(f"{tag}_aw", f"{tag}_ar", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]])[:, :, None, None], [0.0, 0.0])
    Wf = np.zeros((3, 2, KF, KF))
    for i, s in enumerate(SLANTS):
        Wf[i, 0], Wf[i, 1] = figure_kernels(s)
    n.conv(f"{tag}_f", f"{tag}_aw", Wf, np.full(3, -P["FIG_COVER"]), padding=KF // 2)
    # change of ridge type between the upper and lower half of the axis (six types), minus 3x the change of 'any'
    # (so that a plain post seen half-way off the template does not count as a change)
    kd = np.zeros((42, 7, 7, 7))
    for i, s in enumerate(SLANTS):
        d7 = diversity_kernel(s)
        for t in range(7):
            kd[i * 14 + 2 * t, t] = d7; kd[i * 14 + 2 * t + 1, t] = -d7
    n.conv(f"{tag}_dv", f"{tag}_v", kd, np.zeros(42), padding=3); n.relu(f"{tag}_dvr", f"{tag}_dv")
    W5 = np.zeros((3, 42, 1, 1))
    for i in range(3):
        W5[i, i * 14:i * 14 + 12] = -1.0; W5[i, i * 14 + 12:i * 14 + 14] = 3.0
    n.conv(f"{tag}_dg", f"{tag}_dvr", W5, np.full(3, P["DIV_MIN"]))  # shortfall of type change
    We = np.zeros((6, 3, 1, 1)); Wd = np.zeros((6, 3, 1, 1))
    for i in range(3):
        We[i, i] = 1.0; Wd[3 + i, i] = 1.0
    n.conv(f"{tag}_fe", f"{tag}_f", We, np.zeros(6)); n.conv(f"{tag}_de", f"{tag}_dg", Wd, np.zeros(6))
    n.add(f"{tag}_g", [f"{tag}_fe", f"{tag}_de"]); n.relu(f"{tag}_gr", f"{tag}_g")
    W6 = np.zeros((3, 6, 1, 1))
    for i in range(3):
        W6[i, i] = 1.0; W6[i, 3 + i] = -P["DIV_W"]
    n.conv(f"{tag}_o", f"{tag}_gr", W6, np.zeros(3)); n.relu(f"{tag}_or", f"{tag}_o")
    _, Hv, Wv = n.shape[f"{tag}_or"]
    return n.pool(f"{tag}_max", "maxpool", f"{tag}_or", [Hv, Wv], [Hv, Wv])


# =============================================================================================== whole network
def build(out_dir):
    n = Net()
    colour_stage(n)
    zones = blob_stage(n)
    Wl = np.zeros((1, 20, 1, 1)); Wl[0, 19] = 1.0
    n.conv("luma0", "r1", Wl, [0.0])
    n.pool("luma1", "avgpool", "luma0", [2, 2], [2, 2]); n.pool("luma2", "avgpool", "luma1", [2, 2], [2, 2]); n.pool("luma3", "avgpool", "luma2", [2, 2], [2, 2])
    faces = [face_branch(n, "blob2r", "luma1", "luma2", "sf2"), face_branch(n, "blob3r", "luma2", "luma3", "sf3")]
    for f in FIG_LEVELS:
        n.pool(f"rgb{f}", "avgpool", "image", [f, f], [f, f])
    figs = [figure_branch(n, f"rgb{f}", f"fig{i}", "gstat", split=(f == 2)) for i, f in enumerate(FIG_LEVELS)]
    # ---- decision head.  gvec = 21 blob maxima (size-major, zone-minor), warm, green, blue, chroma, figures, faces
    g = n.concat_vec("gvec", zones + ["gstat"] + figs + faces)
    nb, nf, ns = 21, 3 * len(figs), 2 * len(faces)
    tot = nb + 4 + nf + ns
    nh = nb + 2 + nf + ns
    W = np.zeros((nh, tot)); b = np.zeros(nh)
    for s in range(7):
        for z in range(3):
            W[s * 3 + z, s * 3 + z] = 1.0; b[s * 3 + z] = -SCALE_PEN[s] - ZONE_PEN[z]
    W[nb, nb + 0] = 1.0; W[nb, nb + 1] = -ETA; W[nb, nb + 2] = -ETA; b[nb] = -F0          # warm-scene excess
    W[nb + 1, nb + 3] = -100.0; b[nb + 1] = 1.0                                              # 1 for an exactly grey photo
    for j in range(nf):
        W[nb + 2 + j, nb + 4 + j] = 1.0; b[nb + 2 + j] = -FIG_MIN
    for j in range(ns):
        W[nb + 2 + nf + j, nb + 4 + nf + j] = 1.0; b[nb + 2 + nf + j] = -SF_MIN
    n.dense("h1", g, W, b); n.relu("h1r", "h1")

    def group_max(name, lo, cnt):
        Ws = np.zeros((cnt, nh)); Ws[:, lo:lo + cnt] = np.eye(cnt)
        n.dense(name + "sel", "h1r", Ws, np.zeros(cnt)); n.reshape(name + "sel3", name + "sel", [1, 1, cnt])
        return n.pool(name, "maxpool", name + "sel3", [1, cnt], [1, cnt])

    group_max("emax", 0, nb); group_max("fmax", nb + 2, nf); group_max("smax", nb + 2 + nf, ns)
    Wo = np.zeros((2, nh)); Wo[0, nb] = 1.0; Wo[1, nb + 1] = 1.0
    n.dense("osel", "h1r", Wo, np.zeros(2))
    v = n.concat_vec("hvec", ["emax", "osel", "fmax", "smax"])           # best blob, warm excess, grey flag, best figure, best face
    n.dense("h2", v, [[1.0, -GAMMA, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
            [-TAU, 0.0, 0.0, 0.0]); n.relu("h2r", "h2")
    # class 0 score is 0; class 1 score is positive as soon as one detector fires
    n.dense("out", "h2r", [[0.0, 0.0, 0.0, 0.0], [10.0, 0.1, 10.0, 10.0]], [0.0, -0.05])
    n.save(out_dir)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
