"""Hand-designed (no training, no data) network for 28x28 clothing classification.

Everything here comes from formulas (Gaussians, Gabor filters, geometric shape
models) and hand-written rules about what garments, shoes and bags look like.
Run:  python3 build.py OUT_DIR   ->  OUT_DIR/network.json + OUT_DIR/weights.npz
"""
import json, math, os, sys
import numpy as np

# ----------------------------------------------------------------- tiny net API
class Net:
    def __init__(self):
        self.layers = []
        self.w = {}
        self.shapes = {"image": (1, 28, 28)}

    def _rec(self, name, op, inputs, settings, shape):
        self.layers.append(dict(name=name, op=op, inputs=inputs, **settings))
        self.shapes[name] = shape
        return name

    def conv(self, name, inp, W, b, stride=1, padding=0):
        W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
        o, c, kh, kw = W.shape
        C, H, Wd = self.shapes[inp]
        assert c == C, (name, c, C)
        ho = (H + 2 * padding - kh) // stride + 1
        wo = (Wd + 2 * padding - kw) // stride + 1
        self.w[name + ".W"] = W; self.w[name + ".b"] = b
        return self._rec(name, "conv", [inp], dict(out=o, kernel=[kh, kw], stride=stride, padding=padding), (o, ho, wo))

    def dense(self, name, inp, W, b):
        W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
        n = int(np.prod(self.shapes[inp]))
        assert W.shape[1] == n, (name, W.shape, n)
        self.w[name + ".W"] = W; self.w[name + ".b"] = b
        return self._rec(name, "dense", [inp], dict(out=W.shape[0]), (W.shape[0],))

    def relu(self, name, inp):
        return self._rec(name, "relu", [inp], {}, self.shapes[inp])

    def pool(self, name, inp, size, stride, kind="avg"):
        C, H, W = self.shapes[inp]
        ho = (H - size[0]) // stride[0] + 1
        wo = (W - size[1]) // stride[1] + 1
        return self._rec(name, kind + "pool", [inp], dict(size=list(size), stride=list(stride)), (C, ho, wo))

    def add(self, name, *ins):
        return self._rec(name, "add", list(ins), {}, self.shapes[ins[0]])

    def reshape(self, name, inp, shape):
        return self._rec(name, "reshape", [inp], dict(shape=list(shape)), tuple(shape))

    def norm(self, name, inp):
        return self._rec(name, "norm", [inp], {}, self.shapes[inp])

    def save(self, out):
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "network.json"), "w") as f:
            json.dump({"layers": self.layers}, f)
        np.savez(os.path.join(out, "weights.npz"), **self.w)


# ----------------------------------------------------------------- filter maths
def gauss2d(k, sigma):
    r = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k]
    g = np.exp(-((x - r) ** 2 + (y - r) ** 2) / (2 * sigma * sigma))
    return g / g.sum()


def log2d(k, sigma):
    """Laplacian of Gaussian, zero-sum, unit positive lobe."""
    r = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k]
    d2 = (x - r) ** 2 + (y - r) ** 2
    g = (d2 - 2 * sigma ** 2) * np.exp(-d2 / (2 * sigma ** 2))
    g = g - g.mean()
    return g / np.abs(g).sum() * 2.0


def gabor2d(k, sigma, theta, lam):
    """Even (cosine) Gabor: a bar/edge pair detector at orientation theta."""
    r = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k]
    xr = (x - r) * math.cos(theta) + (y - r) * math.sin(theta)
    yr = -(x - r) * math.sin(theta) + (y - r) * math.cos(theta)
    g = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.cos(2 * math.pi * xr / lam)
    g = g - g.mean()
    s = np.abs(g).sum()
    return g / s * 2.0 if s > 0 else g


def eye_conv(c, scale=1.0, bias=0.0):
    W = np.zeros((c, c, 1, 1), np.float32)
    for i in range(c):
        W[i, i, 0, 0] = scale
    return W, np.full(c, bias, np.float32)


def pick_conv(cin, cout, pairs):
    """cout x cin 1x1 conv; pairs = list of (out_index, in_index, weight)."""
    W = np.zeros((cout, cin, 1, 1), np.float32)
    for o, i, v in pairs:
        W[o, i, 0, 0] = v
    return W, np.zeros(cout, np.float32)


# ----------------------------------------------------------------- front end
net = Net()

# Contrast standardisation for the texture branch.
g = net.norm("g", "image")                                   # (1,28,28) zero mean unit var

# Scale space on the RAW photo: three Gaussian blur levels.
SIG = [0.9, 1.6, 2.6]
Wsm = np.stack([gauss2d(5, s) for s in SIG])[:, None, :, :]   # (3,1,5,5)
rsm = net.conv("rsm", "image", Wsm, np.zeros(3), padding=2)   # (3,28,28)

# Silhouette = "clearly brighter than the backdrop".  The backdrop is black, so
# the right reference is the photo's own brightest blurred value, not its mean
# (a mean threshold swallows the gaps between thin sandal straps).
TFRAC = 0.30
GAIN = 10.0
rmx = net.pool("rmx", rsm, [28, 28], [28, 28], kind="max")    # (3,1,1)
rmxv = net.reshape("rmxv", rmx, [3])
We = np.zeros((3 * 784, 3), np.float32)
for c in range(3):
    We[c * 784:(c + 1) * 784, c] = 1.0
rmxe = net.dense("rmxe", rmxv, We, np.zeros(3 * 784))
rmxe = net.reshape("rmxe3", rmxe, [3, 28, 28])
Wn = np.zeros((3, 3, 1, 1), np.float32)
for c in range(3):
    Wn[c, c, 0, 0] = -TFRAC
rmxs = net.conv("rmxs", rmxe, Wn, np.zeros(3))
dif = net.add("dif", rsm, rmxs)                               # blur - TFRAC*max

Wsat = np.zeros((6, 3, 1, 1), np.float32); bsat = np.zeros(6, np.float32)
for i in range(3):
    Wsat[2 * i, i, 0, 0] = GAIN
    Wsat[2 * i + 1, i, 0, 0] = GAIN
    bsat[2 * i + 1] = -1.0
sat = net.conv("sat", dif, Wsat, bsat)
satr = net.relu("satr", sat)
Wm = np.zeros((3, 6, 1, 1), np.float32)
for i in range(3):
    Wm[i, 2 * i, 0, 0] = 1.0
    Wm[i, 2 * i + 1, 0, 0] = -1.0
msk = net.conv("msk", satr, Wm, np.zeros(3))                  # (3,28,28) in [0,1]

# Occupancy grids at two resolutions (fraction of each cell covered by the item).
m14 = net.pool("m14", msk, [2, 2], [2, 2])                    # (3,14,14)
m7 = net.pool("m7", msk, [4, 4], [4, 4])                      # (3,7,7)
m4 = net.pool("m4", msk, [7, 7], [7, 7])                      # (3,4,4)

# Row / column occupancy profiles of the finest silhouette.
rp = net.pool("rp", msk, [1, 28], [1, 28])                    # (3,28,1) width per row
cp = net.pool("cp", msk, [28, 1], [28, 1])                    # (3,1,28) height per col

# Texture: isotropic + four oriented band-pass energies.  Background is flat, so
# these fire only on the item (its fabric pattern, seams, edges).
K = 7
filters = [log2d(K, 1.1)] + [gabor2d(K, 1.5, t, 4.0) for t in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)]
filters += [gabor2d(K, 1.0, 0.0, 3.0), gabor2d(K, 1.0, math.pi / 2, 3.0)]
NF = len(filters)
Wbp = np.zeros((2 * NF, 1, K, K), np.float32)
for i, f in enumerate(filters):
    Wbp[2 * i, 0] = f; Wbp[2 * i + 1, 0] = -f
bp = net.conv("bp", g, Wbp, np.zeros(2 * NF), padding=3)
bpr = net.relu("bpr", bp)
Wte = np.zeros((NF, 2 * NF, 1, 1), np.float32)
for i in range(NF):
    Wte[i, 2 * i, 0, 0] = 1.0; Wte[i, 2 * i + 1, 0, 0] = 1.0
tex = net.conv("tex", bpr, Wte, np.zeros(NF))                 # (NF,28,28) >= 0
t7 = net.pool("t7", tex, [4, 4], [4, 4])                      # (NF,7,7)
t14 = net.pool("t14", tex, [2, 2], [2, 2])                    # (NF,14,14)

# ------------------------------------------------- hand-drawn diagnostic features
# Every feature is a difference of mean occupancies over rectangles that I place
# from general knowledge of where garment parts sit in a centred 28x28 photo.
N = 28


def reg(r0, r1, c0, c1):
    """Mean-over-rectangle operator (normalised indicator)."""
    a = np.zeros((N, N), np.float32)
    a[r0:r1, c0:c1] = 1.0 / max(1, (r1 - r0) * (c1 - c0))
    return a


def mirror(pat):
    return pat[:, ::-1].copy()


FEATS = []          # (name, channel, 28x28 pattern)


def feat(name, ch, terms):
    """terms: list of (weight, r0,r1,c0,c1)."""
    p = np.zeros((N, N), np.float32)
    for w, r0, r1, c0, c1 in terms:
        p += w * reg(r0, r1, c0, c1)
    FEATS.append((name, ch, p))


def featm(name, ch, terms):
    """Same, plus a left/right mirrored twin (shoes and boots face either way)."""
    p = np.zeros((N, N), np.float32)
    for w, r0, r1, c0, c1 in terms:
        p += w * reg(r0, r1, c0, c1)
    FEATS.append((name + "_L", ch, p))
    FEATS.append((name + "_R", ch, mirror(p)))


# --- global geometry -----------------------------------------------------------
def featx(name, terms):
    """terms: (channel, weight, r0,r1,c0,c1) - spans several mask scales."""
    p = np.zeros((3, N, N), np.float32)
    for ch, w, r0, r1, c0, c1 in terms:
        p[ch] += w * reg(r0, r1, c0, c1)
    FEATS.append((name, -1, p))


feat("fill", 0, [(1, 0, 28, 0, 28)])
feat("fill_c", 2, [(1, 0, 28, 0, 28)])
feat("fill_m", 1, [(1, 0, 28, 0, 28)])
# thinness: fine silhouette minus coarse silhouette (thin straps vanish when blurred)
featx("thin", [(0, 1, 0, 28, 0, 28), (2, -1, 0, 28, 0, 28)])
featx("thin_m", [(0, 1, 0, 28, 0, 28), (1, -1, 0, 28, 0, 28)])
feat("top", 0, [(1, 0, 7, 0, 28)])
feat("bot", 0, [(1, 21, 28, 0, 28)])
feat("mid", 0, [(1, 10, 18, 0, 28)])
feat("left", 0, [(1, 0, 28, 0, 7)])
feat("right", 0, [(1, 0, 28, 21, 28)])
feat("ctr", 0, [(1, 0, 28, 10, 18)])
feat("w_top", 0, [(1, 4, 12, 0, 28)])
feat("w_mid", 0, [(1, 13, 19, 0, 28)])
feat("w_bot", 0, [(1, 21, 27, 0, 28)])
feat("h_left", 0, [(1, 0, 28, 2, 9)])
feat("h_mid", 0, [(1, 0, 28, 11, 17)])
feat("h_right", 0, [(1, 0, 28, 19, 26)])
# sleeve length: outer columns low down (long sleeves) vs at shoulder height
feat("slv_low", 0, [(1, 13, 23, 0, 6), (1, 13, 23, 22, 28)])
feat("slv_low2", 0, [(1, 15, 25, 0, 5), (1, 15, 25, 23, 28)])
feat("slv_top", 0, [(1, 4, 12, 0, 6), (1, 4, 12, 22, 28)])
feat("slv_drop", 0, [(1, 4, 12, 0, 6), (1, 4, 12, 22, 28), (-1, 13, 23, 0, 6), (-1, 13, 23, 22, 28)])
feat("body_w", 0, [(1, 14, 24, 7, 21)])
feat("shoulder_w", 0, [(1, 4, 12, 0, 28)])
feat("collar_w", 0, [(1, 0, 3, 0, 28)])
feat("collar_step", 0, [(1, 3, 9, 0, 28), (-1, 0, 3, 0, 28)])
# hem narrower than the sleeve zone => short sleeves (T-shirt)
feat("sleevedrop", 0, [(1, 20, 27, 0, 28), (-1, 4, 12, 0, 28)])
feat("sleevedrop2", 0, [(1, 16, 24, 0, 28), (-1, 3, 11, 0, 28)])
feat("shoulder", 0, [(1, 4, 12, 0, 28), (-1, 15, 24, 0, 28)])
# cuffs below the body hem: bright sides, dark centre in the last rows
feat("hem_gap", 0, [(1, 25, 28, 1, 8), (1, 25, 28, 20, 27), (-2, 25, 28, 9, 19)])
feat("hem_sides", 0, [(1, 25, 28, 1, 8), (1, 25, 28, 20, 27)])
feat("hem_ctr", 0, [(1, 25, 28, 9, 19)])
feat("hem_gap2", 0, [(1, 23, 28, 0, 7), (1, 23, 28, 21, 28), (-2, 23, 28, 8, 20)])

# --- trousers: two legs with a dark gap between them ---------------------------
for tag, r0, r1, gl, gr, ll, lr in (
        ("a", 19, 28, 12, 16, 7, 21),
        ("b", 15, 24, 11, 17, 5, 23),
        ("c", 22, 28, 12, 16, 8, 20)):
    feat("leggap_" + tag, 0, [(1, r0, r1, ll, gl), (1, r0, r1, gr, lr), (-2, r0, r1, gl, gr)])
feat("leggap_top", 0, [(1, 12, 19, 7, 12), (1, 12, 19, 16, 21), (-2, 12, 19, 12, 16)])
feat("waist", 0, [(1, 0, 5, 7, 21)])
feat("waist_narrow", 0, [(1, 0, 5, 7, 21), (-1, 0, 5, 0, 7), (-1, 0, 5, 21, 28)])

# --- upper body: sleeves hanging beside the torso (pullover / coat) ------------
featm("armgap", 0, [(1, 11, 25, 1, 5), (1, 11, 25, 9, 14), (-2, 11, 25, 5, 9)])
featm("armgap2", 0, [(1, 13, 26, 0, 4), (1, 13, 26, 8, 13), (-2, 13, 26, 4, 8)])
feat("bodycol", 0, [(1, 10, 26, 9, 19)])
feat("sleevecol", 0, [(1, 10, 26, 0, 6), (1, 10, 26, 22, 28)])
feat("sleeve_vs_body", 0, [(1, 10, 26, 0, 6), (1, 10, 26, 22, 28), (-1, 10, 26, 9, 19)])

# --- front opening / placket down the centre (coat, shirt) ---------------------
feat("centreline", 0, [(1, 8, 25, 9, 13), (1, 8, 25, 15, 19), (-2, 8, 25, 13, 15)])
feat("centreline_lo", 0, [(1, 15, 27, 9, 13), (1, 15, 27, 15, 19), (-2, 15, 27, 13, 15)])
feat("centreline_hi", 0, [(1, 4, 14, 9, 13), (1, 4, 14, 15, 19), (-2, 4, 14, 13, 15)])
feat("centre_dark", 0, [(-1, 6, 26, 12, 16), (1, 6, 26, 6, 12), (1, 6, 26, 16, 22)])
# neck opening at the top centre
feat("neck", 0, [(1, 0, 4, 5, 11), (1, 0, 4, 17, 23), (-2, 0, 4, 11, 17)])
feat("neck_deep", 0, [(1, 0, 7, 5, 11), (1, 0, 7, 17, 23), (-2, 0, 7, 11, 17)])
# collar / lapel: bright wedges either side of a top-centre notch
feat("collar", 0, [(1, 1, 7, 8, 12), (1, 1, 7, 16, 20), (-2, 1, 7, 12, 16)])

# --- bags: handle loop with a hole inside it -----------------------------------
feat("handlehole", 0, [(1, 0, 4, 9, 19), (1, 4, 11, 6, 10), (1, 4, 11, 18, 22),
                       (-3, 5, 11, 11, 17)])
feat("handlehole2", 0, [(1, 0, 5, 8, 20), (1, 5, 13, 4, 9), (1, 5, 13, 19, 24),
                        (-3, 6, 13, 10, 18)])
feat("bagbody", 0, [(1, 11, 26, 5, 23)])
feat("bagtop", 0, [(1, 0, 6, 0, 28)])
feat("bag_compact", 0, [(1, 8, 26, 4, 24), (-1, 0, 6, 0, 28)])
feat("holetop", 0, [(-1, 1, 8, 10, 18), (1, 8, 24, 6, 22)])

# --- footwear ------------------------------------------------------------------
# Photos are centred, so a wide low shoe sits in the MIDDLE rows (y ~ 6..22),
# while an ankle boot is tall and reaches the top and bottom borders.
feat("sole_mid", 0, [(1, 18, 23, 0, 28)])
feat("sole_lo", 0, [(1, 23, 28, 0, 28)])
feat("empty_top", 0, [(-1, 0, 6, 0, 28)])
feat("empty_bot", 0, [(-1, 23, 28, 0, 28)])
feat("empty_ends", 0, [(-1, 0, 6, 0, 28), (-1, 23, 28, 0, 28)])
feat("midband", 0, [(1, 9, 21, 0, 28)])
feat("lowprofile", 0, [(1, 10, 22, 0, 28), (-1, 0, 7, 0, 28), (-1, 22, 28, 0, 28)])
featm("shaft", 0, [(1, 2, 15, 3, 13), (-1, 2, 15, 16, 27)])
featm("shaft_hi", 0, [(1, 0, 11, 4, 14), (-1, 0, 11, 17, 28)])
featm("toe", 0, [(1, 14, 24, 15, 28), (-1, 14, 24, 0, 12)])
featm("toe_lo", 0, [(1, 18, 27, 15, 28), (-1, 18, 27, 0, 12)])
featm("heelstem", 0, [(1, 14, 26, 3, 7), (-1, 14, 26, 8, 15), (-1, 8, 18, 3, 7)])
feat("solid_mid", 0, [(1, 10, 22, 3, 25)])
feat("shoe_solid", 0, [(1, 8, 22, 2, 26), (-1, 0, 6, 0, 28), (-1, 23, 28, 0, 28)])
# dress: flare (hem wider than bust) and tall narrow column
feat("flare", 0, [(1, 22, 28, 0, 28), (-1, 6, 12, 0, 28)])
feat("flare2", 0, [(1, 24, 28, 0, 28), (-1, 8, 14, 0, 28)])
feat("tall", 0, [(1, 0, 28, 9, 19), (-1, 0, 28, 0, 6), (-1, 0, 28, 22, 28)])
feat("narrow_top", 0, [(1, 4, 11, 9, 19), (-1, 4, 11, 0, 8), (-1, 4, 11, 20, 28)])
feat("top_occ", 0, [(1, 0, 4, 0, 28)])
feat("bot_occ", 0, [(1, 24, 28, 0, 28)])
feat("leggap_d", 0, [(1, 20, 28, 9, 13), (1, 20, 28, 15, 19), (-2, 20, 28, 13, 15)])
feat("leggap_e", 0, [(1, 16, 26, 8, 12), (1, 16, 26, 16, 20), (-2, 16, 26, 12, 16)])

# --- bounding-box extent from saturated occupancy profiles ---------------------
# sat(p) = clip(20*(p-0.05), 0, 1) is ~1 for an occupied row/col, ~0 for an empty
# one; summing gives the object's height / width in pixels.
def sat_profile(inp, name):
    C = net.shapes[inp][0]
    W1 = np.zeros((2 * C, C, 1, 1), np.float32); b1 = np.zeros(2 * C, np.float32)
    for i in range(C):
        W1[2 * i, i, 0, 0] = 20.0; W1[2 * i + 1, i, 0, 0] = 20.0
        b1[2 * i] = -1.0; b1[2 * i + 1] = -2.0
    a = net.conv(name + "_s", inp, W1, b1)
    a = net.relu(name + "_r", a)
    W2 = np.zeros((C, 2 * C, 1, 1), np.float32)
    for i in range(C):
        W2[i, 2 * i, 0, 0] = 1.0; W2[i, 2 * i + 1, 0, 0] = -1.0
    return net.conv(name + "_m", a, W2, np.zeros(C))


rpb = sat_profile(rp, "rpb")                       # (3,28,1) occupied rows
cpb = sat_profile(cp, "cpb")                       # (3,1,28) occupied cols
rsum = net.pool("rsum", rpb, [28, 1], [28, 1])     # (3,1,1)
csum = net.pool("csum", cpb, [1, 28], [1, 28])     # (3,1,1)
rsumv = net.reshape("rsumv", rsum, [3])
csumv = net.reshape("csumv", csum, [3])
rpbv = net.reshape("rpbv", rpb, [3 * 28])
cpbv = net.reshape("cpbv", cpb, [3 * 28])

# --- assemble the diagnostic conv (kernel 28 -> 1x1, i.e. global linear readout)
FN = [f[0] for f in FEATS]
Wf = np.zeros((len(FEATS), 3, N, N), np.float32)
for i, (nm, ch, p) in enumerate(FEATS):
    if ch >= 0:
        Wf[i, ch] = p
    else:
        Wf[i] = p
fb = np.zeros(len(FEATS), np.float32)
diag = net.conv("diag", msk, Wf, fb, stride=1, padding=0)      # (F,1,1)
diagv = net.reshape("diagv", diag, [len(FEATS)])

# --- texture readout on the 7x7 energy grid ------------------------------------
TF = []


def tfeat(name, terms):
    """terms: (weight, channel, r0,r1,c0,c1) on the 7x7 grid (4px cells)."""
    p = np.zeros((NF, 7, 7), np.float32)
    for w, ch, r0, r1, c0, c1 in terms:
        a = np.zeros((7, 7), np.float32)
        a[r0:r1, c0:c1] = 1.0 / max(1, (r1 - r0) * (c1 - c0))
        p[ch] += w * a
    TF.append((name, p))


for ch, nm in enumerate(["iso", "h0", "h45", "h90", "h135", "fine0", "fine90"]):
    tfeat("tex_" + nm, [(1, ch, 1, 6, 1, 6)])
    tfeat("texin_" + nm, [(1, ch, 2, 5, 2, 5)])
tfeat("tex_all", [(1, c, 0, 7, 0, 7) for c in range(NF)])
tfeat("tex_plaid", [(1, 1, 1, 6, 1, 6), (1, 3, 1, 6, 1, 6), (-1, 2, 1, 6, 1, 6), (-1, 4, 1, 6, 1, 6)])
tfeat("tex_stripe", [(1, 1, 1, 6, 1, 6), (1, 3, 1, 6, 1, 6), (-2, 0, 1, 6, 1, 6)])
tfeat("tex_edge", [(1, 0, 0, 7, 0, 7), (-1, 0, 2, 5, 2, 5)])
tfeat("tex_top", [(1, c, 0, 2, 1, 6) for c in range(NF)])
tfeat("tex_bot", [(1, c, 5, 7, 1, 6) for c in range(NF)])
# centre-strip texture: a button placket is a short-period vertical rhythm
tfeat("tex_ctr_fine90", [(1, 6, 1, 6, 3, 4)])
tfeat("tex_ctr_fine0", [(1, 5, 1, 6, 3, 4)])
tfeat("tex_ctr_iso", [(1, 0, 1, 6, 3, 4)])
tfeat("tex_ctr_all", [(1, c, 1, 6, 3, 4) for c in range(NF)])
tfeat("tex_side_fine90", [(1, 6, 1, 6, 1, 3), (1, 6, 1, 6, 4, 6)])
tfeat("tex_placket", [(1, 6, 1, 6, 3, 4), (-1, 6, 1, 6, 1, 3), (-1, 6, 1, 6, 4, 6)])
tfeat("tex_knit", [(1, 2, 1, 6, 1, 6), (1, 4, 1, 6, 1, 6), (-1, 1, 1, 6, 1, 6), (-1, 3, 1, 6, 1, 6)])
TN = [t[0] for t in TF]
Wt = np.stack([t[1] for t in TF])
texv = net.conv("texf", t7, Wt, np.zeros(len(TF)), stride=1, padding=0)
texv = net.reshape("texv", texv, [len(TF)])

# --- grey-level centre structure (placket / opening before thresholding) -------
GF = []


def gfeat(name, terms):
    p = np.zeros((N, N), np.float32)
    for w, r0, r1, c0, c1 in terms:
        p += w * reg(r0, r1, c0, c1)
    GF.append((name, p))


gfeat("g_ctr", [(1, 6, 26, 8, 12), (1, 6, 26, 16, 20), (-2, 6, 26, 12, 16)])
gfeat("g_ctr_lo", [(1, 16, 27, 8, 12), (1, 16, 27, 16, 20), (-2, 16, 27, 12, 16)])
gfeat("g_ctr_hi", [(1, 3, 13, 8, 12), (1, 3, 13, 16, 20), (-2, 3, 13, 12, 16)])
gfeat("g_neck", [(1, 0, 4, 5, 11), (1, 0, 4, 17, 23), (-2, 0, 4, 11, 17)])
gfeat("g_bright", [(1, 4, 26, 5, 23)])
gfeat("g_collar", [(1, 1, 6, 8, 12), (1, 1, 6, 16, 20), (-2, 1, 6, 12, 16)])
gfeat("g_hem", [(1, 24, 28, 4, 24), (-1, 18, 24, 4, 24)])
GN = [x[0] for x in GF]
gv = net.conv("gf", g, np.stack([x[1] for x in GF])[:, None, :, :], np.zeros(len(GF)), padding=0)
gv = net.reshape("gv", gv, [len(GF)])

# ------------------------------------------------- hand-drawn silhouette models
# Shapes are typed from general knowledge of clothing: a torso rectangle with
# sleeve quads and a neck notch, two trouser legs with a crotch gap, an A-line
# skirt, a sole line with an upper curve, a shaft plus foot for a boot, a bag
# body with a handle loop.  Rasterised softly onto the 14x14 occupancy grid.
def raster(fn, n=14, ss=4):
    yy, xx = np.mgrid[0:n * ss, 0:n * ss]
    y = (yy + 0.5) * 28.0 / (n * ss); x = (xx + 0.5) * 28.0 / (n * ss)
    m = fn(x, y).astype(np.float32)
    return m.reshape(n, ss, n, ss).mean(axis=(1, 3))


def poly(pts):
    p = np.asarray(pts, float)
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    s = 1.0 if area > 0 else -1.0

    def f(x, y):
        ok = np.ones(np.shape(x), bool)
        for i in range(len(p)):
            x1, y1 = p[i]; x2, y2 = p[(i + 1) % len(p)]
            ok &= (s * ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))) >= -1e-9
        return ok
    return f


def rect(x0, x1, y0, y1):
    return poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def ell(cx, cy, rx, ry):
    return lambda x, y: ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def U(*fs):
    return lambda x, y: np.logical_or.reduce([f(x, y) for f in fs])


def MINUS(a, *bs):
    return lambda x, y: a(x, y) & ~np.logical_or.reduce([b(x, y) for b in bs])


def MIR(f):
    return lambda x, y: f(28.0 - x, y)


def BOTH(f):
    return U(f, MIR(f))


def neck(cx=14.0, cy=1.0, rx=3.6, ry=2.4):
    return ell(cx, cy, rx, ry)


TEMPL = []          # (class, name, 14x14 pattern)


def T(cls, name, fn, mirrored=False):
    TEMPL.append((cls, name, raster(fn)))
    if mirrored:
        TEMPL.append((cls, name + "_m", raster(MIR(fn))))


def tee(body=(8, 20), hem=28, slen=12, sout=3, cwid=(10, 18), ctop=3, flare=0):
    """T-shirt: narrow collar at the very top, a shoulder/sleeve trapezoid that
    widens down to the sleeve hem (~row 12), then a plain torso to the hem."""
    b0, b1 = body
    collar = rect(cwid[0], cwid[1], 0, ctop)
    sleeves = poly([(b0 - 1, ctop), (b1 + 1, ctop), (28 - sout, slen), (sout, slen)])
    torso = poly([(b0, ctop), (b1, ctop), (b1 + flare, hem), (b0 - flare, hem)])
    return U(collar, sleeves, torso)


def knit(body=(4, 24), hem=26, cwid=(10, 18), ctop=2, shoulder=8, ends=True, flare=0):
    """Pullover: collar, then a boxy block (torso + long sleeves hanging straight
    down beside it), with the sleeve cuffs poking out below the body hem."""
    b0, b1 = body
    collar = rect(cwid[0], cwid[1], 0, ctop)
    block = poly([(b0 + 2, ctop), (b1 - 2, ctop), (b1, shoulder), (b1 + flare, hem),
                  (b0 - flare, hem), (b0, shoulder)])
    out = U(collar, block)
    if ends:
        out = U(out, rect(b0, b0 + 4, hem - 1, 28), rect(b1 - 4, b1, hem - 1, 28))
    return out


def coat(top=(10, 18), hemw=(2, 26), hem=27, ctop=3, shoulder=8, ends=True, hood=False):
    """Coat: a long trapezoid, narrow at the collar and widening steadily to the
    hem, with sleeve cuffs below the body hem."""
    t0, t1 = top; h0, h1 = hemw
    body = poly([(t0, 0), (t1, 0), (h1, hem), (h0, hem)])
    out = body
    if hood:
        out = U(out, ell(14, ctop, 7.0, 4.0))
    if ends:
        out = U(out, rect(h0, h0 + 5, hem - 1, 28), rect(h1 - 5, h1, hem - 1, 28))
    return out


def shirt(top=(9, 19), hemw=(3, 25), hem=26, ctop=2, ends=True, short=False, slen=12):
    """Shirt: like a coat but a little shorter and less flared; short-sleeved
    versions have the T-shirt sleeve trapezoid instead of hanging cuffs."""
    t0, t1 = top; h0, h1 = hemw
    if short:
        return tee(body=(8, 20), hem=hem, slen=slen, sout=h0, cwid=(t0, t1), ctop=ctop)
    body = poly([(t0, 0), (t1, 0), (h1, hem), (h0, hem)])
    out = body
    if ends:
        out = U(out, rect(h0, h0 + 5, hem - 1, 28), rect(h1 - 5, h1, hem - 1, 28))
    return out


def dress(bust=(9, 19), hemw=(4, 24), top=1, hem=28, waist=11, sleeves=0, cwid=(11, 17)):
    b0, b1 = bust; h0, h1 = hemw
    collar = rect(cwid[0], cwid[1], 0, top + 1)
    bod = poly([(b0, top), (b1, top), (b1, waist), (b0, waist)])
    sk = poly([(b0, waist), (b1, waist), (h1, hem), (h0, hem)])
    sh = U(collar, bod, sk)
    if sleeves == 1:
        sh = U(sh, poly([(b0, top), (b1, top), (b1 + 5, top + 7), (b0 - 5, top + 7)]))
    elif sleeves == 2:
        sh = U(sh, BOTH(poly([(b0, top), (b0 - 6, top + 4), (b0 - 5, 22), (b0 - 1, 22)])))
    return sh


def trouser(waist=(7, 21), crotch=11, hem=28, legw=6, gap=2, taper=0):
    w0, w1 = waist
    cx = 14.0
    hip = rect(w0, w1, 0, crotch)
    l0 = w0; l1 = cx - gap / 2
    r0 = cx + gap / 2; r1 = w1
    ll = poly([(l0, crotch), (l1, crotch), (l1 - taper, hem), (l0 + taper, hem)])
    rr = poly([(r0, crotch), (r1, crotch), (r1 - taper, hem), (r0 + taper, hem)])
    return U(hip, ll, rr)


def sneaker(h=15, sole=3, toe_up=3, high=False):
    """Low shoe: flat sole + rounded upper, toe to the left.  Centred vertically."""
    y0 = 14 - h / 2.0; y1 = 14 + h / 2.0
    ys = y1 - sole
    solef = poly([(0, ys), (28, ys), (28, y1), (0, y1)])
    up = poly([(0.5, ys), (1, y0 + toe_up + 2), (5, y0 + toe_up), (11, y0),
               (17, y0 + (0 if high else 1)), (23, y0 + 3), (27.5, y0 + 6), (27.5, ys)])
    return U(solef, up)


def boot(shaft=(6, 16), top=2, foot=(2, 26), fy=17, sole=23, heel=True, hx=(4, 9)):
    s0, s1 = shaft; f0, f1 = foot
    sh = rect(s0, s1, top, fy + 1)
    ft = poly([(f0, fy - 2), (s1, fy - 2), (f1, fy + 2), (f1, sole), (f0, sole)])
    sl = rect(f0, f1, sole - 1, sole + 1)
    out = U(sh, ft, sl)
    if heel:
        out = U(out, rect(hx[0], hx[1], sole, sole + 4))
    return out


def sandal(kind=0):
    if kind == 0:      # flat slide: sole + one broad strap
        return U(rect(1, 27, 19, 22),
                 poly([(6, 19), (16, 12), (20, 14), (11, 19)]),
                 rect(2, 6, 17, 22))
    if kind == 1:      # high heel: thin sole, heel stem, ankle strap
        return U(poly([(2, 18), (20, 16), (21, 19), (3, 21)]),
                 rect(19, 22, 12, 26),
                 poly([(8, 16), (14, 8), (17, 9), (11, 17)]),
                 rect(13, 20, 6, 9))
    if kind == 2:      # strappy: several thin bands over a sole
        return U(rect(1, 27, 20, 23),
                 poly([(4, 20), (10, 13), (12, 14), (6, 20)]),
                 poly([(12, 20), (18, 12), (20, 13), (14, 20)]),
                 poly([(20, 20), (25, 14), (27, 15), (22, 20)]))
    if kind == 3:      # flip-flop: sole + thin Y strap
        return U(rect(2, 26, 19, 22),
                 poly([(12, 19), (14, 11), (16, 11), (15, 19)]),
                 poly([(6, 18), (14, 12), (15, 14), (8, 19)]),
                 poly([(22, 18), (15, 12), (14, 14), (20, 19)]))
    if kind == 4:      # wedge: solid triangular heel under a strap
        return U(poly([(2, 18), (26, 16), (27, 24), (2, 24)]),
                 poly([(6, 18), (16, 10), (20, 12), (11, 18)]))
    return U(rect(1, 27, 18, 21), rect(10, 18, 10, 18))


def bag(kind=0):
    if kind == 0:      # tote: rectangular body + handle loop with a hole
        body = rect(4, 24, 11, 26)
        handle = MINUS(ell(14, 12, 8, 10), ell(14, 12, 5.8, 7.6))
        return U(body, MINUS(handle, rect(0, 28, 12, 28)))
    if kind == 1:      # clutch: plain wide rectangle, no handle
        return rect(2, 26, 9, 20)
    if kind == 2:      # shoulder bag: body + long thin strap arch
        body = rect(5, 23, 13, 25)
        strap = MINUS(ell(14, 14, 9, 12), ell(14, 14, 7.4, 10.4))
        return U(body, MINUS(strap, rect(0, 28, 14, 28)))
    if kind == 3:      # bucket bag: rounded body, short handles
        body = U(rect(5, 23, 10, 24), ell(14, 22, 9, 5))
        handle = MINUS(ell(14, 10, 6, 7), ell(14, 10, 4.2, 5.2))
        return U(body, MINUS(handle, rect(0, 28, 10, 28)))
    if kind == 4:      # small purse: compact body, tall thin handle
        body = rect(7, 21, 14, 25)
        handle = MINUS(ell(14, 14, 5, 11), ell(14, 14, 3.6, 9.4))
        return U(body, MINUS(handle, rect(0, 28, 14, 28)))
    if kind == 5:      # wide shopper, handles at the sides
        body = poly([(3, 10), (25, 10), (23, 26), (5, 26)])
        return U(body, BOTH(poly([(4, 10), (7, 3), (9, 4), (7, 10)])))
    return rect(4, 24, 8, 24)


# ---- class 0: T-shirt / top
T(0, "tee1", tee())
T(0, "tee2", tee(body=(7, 21), sout=1, slen=13, cwid=(9, 19)))
T(0, "tee3", tee(body=(9, 19), sout=4, slen=11, hem=27))
T(0, "tee4", tee(body=(6, 22), sout=0, slen=14, cwid=(8, 20), flare=1))
T(0, "tee5", tee(body=(8, 20), sout=3, slen=10, ctop=4, cwid=(11, 17)))
T(0, "tank1", U(rect(10, 18, 0, 2), rect(8, 20, 2, 28), BOTH(rect(8, 11, 0, 5))))
T(0, "tank2", U(rect(11, 17, 0, 2), rect(9, 19, 2, 28)))
T(0, "top1", tee(body=(9, 19), hem=26, slen=9, sout=5, cwid=(10, 18), ctop=4))
# ---- class 2: pullover
T(2, "knit1", knit())
T(2, "knit2", knit(body=(3, 25), hem=27, shoulder=7))
T(2, "knit3", knit(body=(5, 23), hem=25, ctop=3, shoulder=9))
T(2, "knit4", knit(body=(4, 24), hem=22, ctop=3, shoulder=8, flare=1))
T(2, "knit5", knit(body=(6, 22), hem=27, shoulder=10, ends=False))
T(2, "knit6", U(knit(body=(5, 23), hem=26), ell(14, 2, 6.5, 3.5)))
T(2, "knit7", knit(body=(7, 21), hem=26, shoulder=9, cwid=(11, 17)))
T(2, "knit8", knit(body=(3, 25), hem=28, ctop=1, shoulder=6, flare=2))
# ---- class 4: coat
T(4, "coat1", coat())
T(4, "coat2", coat(top=(9, 19), hemw=(3, 25), hem=28))
T(4, "coat3", coat(top=(11, 17), hemw=(1, 27), hem=27))
T(4, "coat4", coat(top=(8, 20), hemw=(4, 24), hem=24, ends=False))
T(4, "coat5", coat(hood=True, hemw=(3, 25)))
T(4, "coat6", coat(top=(10, 18), hemw=(5, 23), hem=28, ends=False))
T(4, "coat7", coat(top=(11, 17), hemw=(6, 22), hem=26))
T(4, "coat8", coat(top=(9, 19), hemw=(1, 27), hem=28))
# ---- class 6: shirt
T(6, "shirt1", shirt())
T(6, "shirt2", shirt(top=(10, 18), hemw=(4, 24), hem=27))
T(6, "shirt3", shirt(short=True, hemw=(2, 26)))
T(6, "shirt4", shirt(top=(8, 20), hemw=(2, 26), hem=28))
T(6, "shirt5", shirt(short=True, hemw=(4, 24), slen=10, hem=25))
T(6, "shirt6", shirt(top=(11, 17), hemw=(5, 23), hem=26))
T(6, "shirt7", shirt(top=(9, 19), hemw=(3, 25), hem=25, ends=False))
T(6, "shirt8", shirt(short=True, hemw=(1, 27), slen=13))
# ---- class 3: dress
T(3, "dress1", dress())
T(3, "dress2", dress(bust=(9, 19), hemw=(8, 20), waist=8))
T(3, "dress3", dress(bust=(10, 18), hemw=(3, 25), waist=12, sleeves=1))
T(3, "dress4", dress(bust=(9, 19), hemw=(6, 22), waist=10, sleeves=2))
T(3, "dress5", dress(bust=(10, 18), hemw=(10, 18), top=0, hem=28, waist=6))
T(3, "dress6", dress(bust=(8, 20), hemw=(2, 26), waist=13, sleeves=1))
T(3, "dress7", dress(bust=(9, 19), hemw=(7, 21), waist=9, sleeves=2, top=2))
T(3, "dress8", dress(bust=(11, 17), hemw=(5, 23), waist=14))
# ---- class 1: trouser
T(1, "tr1", trouser(gap=4))
T(1, "tr2", trouser(waist=(5, 23), crotch=10, gap=5))
T(1, "tr3", trouser(waist=(8, 20), crotch=12, gap=3, taper=1))
T(1, "tr4", trouser(waist=(6, 22), crotch=9, hem=28, gap=6))
T(1, "tr5", trouser(waist=(7, 21), crotch=14, gap=3))
T(1, "tr6", trouser(waist=(6, 22), crotch=8, hem=20, gap=4))
T(1, "tr7", trouser(waist=(9, 19), crotch=11, gap=4, taper=-1))
T(1, "tr8", trouser(waist=(7, 21), crotch=15, gap=5))
# ---- class 7: sneaker
for i, h in enumerate((13, 15, 17, 19)):
    T(7, f"snk{i}", sneaker(h=h), mirrored=True)
T(7, "snk_hi", sneaker(h=19, high=True), mirrored=True)
T(7, "snk_thick", sneaker(h=16, sole=5), mirrored=True)
# ---- class 9: ankle boot
T(9, "boot1", boot(), mirrored=True)
T(9, "boot2", boot(shaft=(7, 17), top=4, fy=18, sole=24, heel=False), mirrored=True)
T(9, "boot3", boot(shaft=(5, 15), top=1, fy=16, sole=22, hx=(3, 8)), mirrored=True)
T(9, "boot4", boot(shaft=(8, 18), top=6, fy=19, sole=25, heel=False), mirrored=True)
T(9, "boot5", boot(shaft=(6, 16), top=2, foot=(1, 27), fy=18, sole=24), mirrored=True)
T(9, "boot6", boot(shaft=(9, 19), top=8, fy=20, sole=26, heel=False), mirrored=True)
# ---- class 5: sandal
for k in range(6):
    T(5, f"san{k}", sandal(k), mirrored=True)
# ---- class 8: bag
for k in range(6):
    T(8, f"bag{k}", bag(k))
T(8, "bag6", U(rect(3, 25, 12, 26), MINUS(ell(14, 12, 9, 11), ell(14, 12, 7.0, 9.0))))
T(8, "bag7", rect(6, 22, 6, 22))

# ------------------------------------------------- template matching readout
# score_t = [ <T,obs> - BETA * <1-T,obs> ] / <T,1>   (a Tversky-style overlap).
# obs is the 14x14 occupancy grid; a perfect match scores 1, a superset scores
# low, a subset scores low.  Linear in obs -> one dense layer.
BETA = 0.6
THETA = 0.50
NT = len(TEMPL)
Wtm = np.zeros((NT, 3 * 14 * 14), np.float32)
btm = np.full(NT, -THETA, np.float32)
for i, (cls, name, pat) in enumerate(TEMPL):
    s = pat.sum()
    ker = ((1 + BETA) * pat - BETA) / s
    Wtm[i, 0:196] = (0.6 * ker).ravel()
    Wtm[i, 196:392] = (0.4 * ker).ravel()
tscore = net.dense("tscore", m14, Wtm, btm)
tact = net.relu("tact", tscore)                       # (NT,) template evidence

# mirror-invariant: max over each left/right pair, then one value per shape model
groups = {}
for i, (cls, name, pat) in enumerate(TEMPL):
    base = name[:-2] if name.endswith("_m") else name
    groups.setdefault((cls, base), []).append(i)
gkeys = sorted(groups.keys())
pairs = [g for g in gkeys if len(groups[g]) == 2]
NB = len(gkeys)
bidx = {g: j for j, g in enumerate(gkeys)}
Wd = np.zeros((len(pairs), NT), np.float32)
for j, g in enumerate(pairs):
    a, b = groups[g]
    Wd[j, a] = 1.0; Wd[j, b] = -1.0
pd = net.dense("pdiff", tact, Wd, np.zeros(len(pairs)))
pr = net.relu("prelu", pd)
Wsel = np.zeros((NB, NT), np.float32)
Wexp = np.zeros((NB, len(pairs)), np.float32)
for g in gkeys:
    j = bidx[g]; mem = groups[g]
    Wsel[j, mem[-1]] = 1.0
    if len(mem) == 2:
        Wexp[j, pairs.index(g)] = 1.0
ps = net.dense("psel", tact, Wsel, np.zeros(NB))
pe = net.dense("pexp", pr, Wexp, np.zeros(NB))
bact = net.add("bact", ps, pe)                        # (NB,) best evidence per model

# per-class shape evidence: average over that class's shape models
Wbc = np.zeros((10, NB), np.float32)
for g in gkeys:
    cls, base = g
    n = sum(1 for gg in gkeys if gg[0] == cls)
    Wbc[cls, bidx[g]] = 10.0 / n
bcls = net.dense("bcls", bact, Wbc, np.zeros(10))     # (10,) shape score per class

# ------------------------------------------------- hand-written rule readout
# V collects every diagnostic; h1 holds hand-set thresholds and 0..1 ramps;
# h2 holds a few AND / max combinations; the class score is the sum of four
# linear heads (rules, combinations, raw diagnostics, shape templates).
VN = list(FN) + list(TN) + list(GN) + [f"rsum{i}" for i in range(3)] + \
     [f"csum{i}" for i in range(3)] + [f"b{c}_{b}" for (c, b) in gkeys]
Vparts = [diagv, texv, gv, rsumv, csumv, bact]
# concatenate by projecting each part into a common width and summing (no concat op)
NV = len(VN)
proj = []
off = 0
for p in Vparts:
    n = int(np.prod(net.shapes[p]))
    Wp = np.zeros((NV, n), np.float32)
    for i in range(n):
        Wp[off + i, i] = 1.0
    proj.append(net.dense("cat_" + p, p, Wp, np.zeros(NV)))
    off += n
V = net.add("V", *proj)
VIDX = {n: i for i, n in enumerate(VN)}

H1 = []          # (name, [(feature, coef)...], bias)
H1IDX = {}


def u1(nm, terms, bias=0.0):
    H1IDX[nm] = len(H1); H1.append((nm, terms, bias))


def ramp(nm, f, lo, hi):
    """clip((f-lo)/(hi-lo),0,1) as two relu units; use rterm() to apply it."""
    k = 1.0 / (hi - lo)
    u1(nm + "p", [(f, k)], -k * lo)
    u1(nm + "n", [(f, k)], -k * lo - 1.0)
    return nm


def thr(nm, f, t, gain=1.0):
    u1(nm, [(f, gain)], -gain * t)
    return nm


O1 = [dict() for _ in range(10)]
O2 = [dict() for _ in range(10)]
O0 = [dict() for _ in range(10)]
BIAS = [0.0] * 10


def o(k, nm, w, layer=1):
    d = {1: O1, 2: O2, 0: O0}[layer][k]
    d[nm] = d.get(nm, 0.0) + w


def rt(k, nm, w, layer=1):
    o(k, nm + "p", w, layer); o(k, nm + "n", -w, layer)


def raw(k, f, w):
    o(k, f, w, 0)


# ---- geometry ramps ----------------------------------------------------------
rH = ramp("rH", "rsum0", 20 / 28, 27 / 28)          # object reaches top+bottom
rHn = ramp("rHn", "rsum0", 24 / 28, 17 / 28)        # object is short
rW = ramp("rW", "csum0", 20 / 28, 27 / 28)          # object reaches left+right
rWn = ramp("rWn", "csum0", 25 / 28, 18 / 28)        # object is narrow
rFill = ramp("rFill", "fill", 0.14, 0.42)
rSparse = ramp("rSparse", "fill", 0.30, 0.16)
rThin = ramp("rThin", "thin", 0.015, 0.10)
rThin2 = ramp("rThin2", "thin_m", 0.01, 0.07)

# ---- trousers ----------------------------------------------------------------
rLegA = ramp("rLegA", "leggap_a", 0.55, 1.25)
rLegB = ramp("rLegB", "leggap_b", 0.45, 1.10)
rLegC = ramp("rLegC", "leggap_c", 0.45, 1.10)
rLegT = ramp("rLegT", "leggap_top", 0.30, 0.90)
rNeck = ramp("rNeck", "neck", 0.10, 0.45)
rWaist = ramp("rWaist", "waist", 0.30, 0.60)

# ---- upper body --------------------------------------------------------------
rSh = ramp("rSh", "shoulder", 0.04, 0.28)           # short sleeves stick out
rSh2 = ramp("rSh2", "shoulder", 0.16, 0.40)
rDrop = ramp("rDrop", "sleevedrop", -0.08, -0.32)   # hem narrower than shoulders
rDrop2 = ramp("rDrop2", "sleevedrop2", -0.05, -0.28)
rAgL = ramp("rAgL", "armgap_L", 0.35, 0.95)
rAgR = ramp("rAgR", "armgap_R", 0.35, 0.95)
rAg2L = ramp("rAg2L", "armgap2_L", 0.35, 0.95)
rAg2R = ramp("rAg2R", "armgap2_R", 0.35, 0.95)
rCl = ramp("rCl", "centreline", 0.15, 0.70)         # dark front opening
rClLo = ramp("rClLo", "centreline_lo", 0.15, 0.70)
rClHi = ramp("rClHi", "centreline_hi", 0.15, 0.70)
rGcl = ramp("rGcl", "g_ctr", 0.10, 0.55)
rGclLo = ramp("rGclLo", "g_ctr_lo", 0.10, 0.55)
rCol = ramp("rCol", "collar", 0.08, 0.40)
rGcol = ramp("rGcol", "g_collar", 0.08, 0.40)
rSvb = ramp("rSvb", "sleeve_vs_body", -0.10, 0.25)
# corrected sleeve cues: long sleeves keep the outer columns occupied low down
rSlv = ramp("rSlv", "slv_low", 0.10, 0.50)
rSlv2 = ramp("rSlv2", "slv_low2", 0.10, 0.50)
rSlvDrop = ramp("rSlvDrop", "slv_drop", 0.05, 0.40)
rHemGap = ramp("rHemGap", "hem_gap", 0.10, 0.80)
rHemGap2 = ramp("rHemGap2", "hem_gap2", 0.10, 0.80)
rHemCtr = ramp("rHemCtr", "hem_ctr", 0.25, 0.60)
rColStep = ramp("rColStep", "collar_step", 0.05, 0.35)
rLegD = ramp("rLegD", "leggap_d", 0.40, 1.10)
rLegE = ramp("rLegE", "leggap_e", 0.40, 1.10)
rTopOcc = ramp("rTopOcc", "top_occ", 0.03, 0.25)
rBotOcc = ramp("rBotOcc", "bot_occ", 0.03, 0.25)

# ---- dress -------------------------------------------------------------------
rFlare = ramp("rFlare", "flare", -0.02, 0.22)
rFlare2 = ramp("rFlare2", "flare2", -0.02, 0.22)
rTall = ramp("rTall", "tall", 0.10, 0.45)
rNTop = ramp("rNTop", "narrow_top", 0.05, 0.35)

# ---- footwear ----------------------------------------------------------------
rSole = ramp("rSole", "sole_mid", 0.35, 0.75)
rEmptyTop = ramp("rEmptyTop", "top", 0.16, 0.03)
rEmptyBot = ramp("rEmptyBot", "bot", 0.16, 0.03)
rMid = ramp("rMid", "midband", 0.40, 0.75)
rSolid = ramp("rSolid", "shoe_solid", 0.30, 0.65)
rShaftL = ramp("rShaftL", "shaft_L", 0.15, 0.55)
rShaftR = ramp("rShaftR", "shaft_R", 0.15, 0.55)
rShaftHL = ramp("rShaftHL", "shaft_hi_L", 0.10, 0.45)
rShaftHR = ramp("rShaftHR", "shaft_hi_R", 0.10, 0.45)
rHeelL = ramp("rHeelL", "heelstem_L", 0.10, 0.45)
rHeelR = ramp("rHeelR", "heelstem_R", 0.10, 0.45)
rLow = ramp("rLow", "lowprofile", 0.10, 0.45)

# ---- bags --------------------------------------------------------------------
rHole = ramp("rHole", "handlehole", 0.15, 0.70)
rHole2 = ramp("rHole2", "handlehole2", 0.15, 0.70)
rBagBody = ramp("rBagBody", "bagbody", 0.45, 0.80)
rBagComp = ramp("rBagComp", "bag_compact", 0.15, 0.45)
rHoleTop = ramp("rHoleTop", "holetop", 0.10, 0.50)

# ---- texture -----------------------------------------------------------------
rPlaid = ramp("rPlaid", "tex_plaid", 0.00, 0.14)
rStripe = ramp("rStripe", "tex_stripe", 0.00, 0.14)
rPlk = ramp("rPlk", "tex_placket", 0.00, 0.07)
rKnit = ramp("rKnit", "tex_knit", 0.00, 0.09)
rTexAll = ramp("rTexAll", "tex_all", 0.10, 0.45)
rTexLo = ramp("rTexLo", "tex_all", 0.22, 0.06)
rCtrFine = ramp("rCtrFine", "tex_ctr_fine90", 0.02, 0.16)
rFine = ramp("rFine", "tex_fine90", 0.03, 0.22)

# ---- h2: combinations (AND / max) -------------------------------------------
H2 = []
H2IDX = {}


def u2(nm, terms, bias=0.0):
    H2IDX[nm] = len(H2); H2.append((nm, terms, bias))


def ramp2(nm, terms, lo, hi):
    k = 1.0 / (hi - lo)
    u2(nm + "p", [(t, k * c) for t, c in terms], -k * lo)
    u2(nm + "n", [(t, k * c) for t, c in terms], -k * lo - 1.0)
    return nm


def rt2(k, nm, w):
    o(k, nm + "p", w, 2); o(k, nm + "n", -w, 2)


def max2(nm, a, b):
    """max(a,b) = b + relu(a-b); returns (unit_to_add, base_unit)."""
    u2(nm, [(a, 1.0), (b, -1.0)], 0.0)
    return nm, b


def R(nm, w=1.0):
    """clipped ramp value as h1 terms"""
    return [(nm + "p", w), (nm + "n", -w)]


def SUM(*nms):
    t = []
    for nm in nms:
        t += R(nm)
    return t


# leg gap: evidence summed over three placements of the crotch gap
legT = SUM("rLegA", "rLegB", "rLegC")
rTr = ramp2("rTr", legT + R("rWn"), 1.70, 2.90)
rTr2 = ramp2("rTr2", legT, 1.10, 2.30)
# short sleeves: shoulder bump AND a narrower hem
rTee = ramp2("rTee", R("rSh") + R("rDrop"), 1.05, 1.85)
rTee2 = ramp2("rTee2", R("rSh2") + R("rDrop2"), 1.05, 1.85)
# long sleeves hanging free of the torso (pullover)
rKn = ramp2("rKn", SUM("rAgL", "rAgR", "rAg2L", "rAg2R"), 1.10, 2.60)
# front opening (coat): mask line or grey line, upper or lower torso
rCo = ramp2("rCo", SUM("rCl", "rGcl", "rClHi", "rGclLo"), 0.90, 2.40)
# footwear gates
rShoe = ramp2("rShoe", R("rW") + R("rHn"), 1.10, 1.80)
rShoeSolid = ramp2("rShoeSolid", R("rW") + R("rHn") + R("rSolid"), 1.70, 2.60)
rSandal = ramp2("rSandal", R("rW") + R("rHn") + R("rSparse"), 1.60, 2.50)
rSandal2 = ramp2("rSandal2", R("rThin2") + R("rSparse"), 1.05, 1.85)
# boot: a shaft on one side
rBoot = ramp2("rBoot", SUM("rShaftL", "rShaftR", "rShaftHL", "rShaftHR"), 0.65, 1.60)
# bag: handle hole, or a compact solid block
rBag = ramp2("rBag", SUM("rHole", "rHole2"), 0.65, 1.60)
rBag2 = ramp2("rBag2", R("rBagBody") + R("rBagComp"), 1.10, 1.85)
# dress: flare and height
rDress = ramp2("rDress", R("rFlare") + R("rH"), 1.05, 1.85)
rDress2 = ramp2("rDress2", R("rFlare2") + R("rTall"), 1.05, 1.85)
# shirt: placket rhythm or plaid, with a collar
rShirt = ramp2("rShirt", R("rPlk") + R("rCol"), 0.85, 1.70)
rShirt2 = ramp2("rShirt2", R("rPlaid") + R("rGcol"), 0.85, 1.70)
rShirt3 = ramp2("rShirt3", R("rCtrFine") + R("rPlk"), 0.75, 1.60)

# ---- class scores ------------------------------------------------------------
# Positive terms are the evidence FOR a class; cross-penalties are kept small so
# that a single strong cue is not double-counted against every other class.
# 0 T-shirt / top
rt(0, rTee, 2.6, 2); rt(0, rTee2, 1.6, 2)
rt(0, rSh, 1.2); rt(0, rDrop, 0.8)
rt(0, rKn, -0.7, 2); rt(0, rCo, -0.7, 2)
rt(0, rDress, -0.6, 2); rt(0, rTr, -1.3, 2)
rt(0, rAgL, -0.3); rt(0, rAgR, -0.3)
rt(0, rShirt, -0.4, 2); rt(0, rShirt2, -0.3, 2)
rt(0, rBoot, -0.6, 2); rt(0, rShoe, -0.9, 2)
BIAS[0] = 0.2

# 1 trouser
rt(1, rTr, 3.0, 2); rt(1, rTr2, 1.6, 2)
rt(1, rLegA, 0.8); rt(1, rLegB, 0.6); rt(1, rLegC, 0.5)
rt(1, rWn, 0.7); rt(1, rH, 0.5)
rt(1, rNeck, -0.9); rt(1, rSh, -0.7)
rt(1, rShoe, -1.2, 2); rt(1, rBag, -0.9, 2)
BIAS[1] = -0.1

# 2 pullover
rt(2, rKn, 2.6, 2)
rt(2, rAgL, 0.7); rt(2, rAgR, 0.7)
rt(2, rSvb, 0.5); rt(2, rKnit, 0.5)
rt(2, rSh, -0.8); rt(2, rDrop, -0.5)
rt(2, rCo, -0.8, 2); rt(2, rCl, -0.4)
rt(2, rDress, -0.5, 2); rt(2, rTr, -1.3, 2)
rt(2, rShirt, -0.4, 2); rt(2, rShirt3, -0.3, 2)
rt(2, rShoe, -0.9, 2); rt(2, rBoot, -0.6, 2)
BIAS[2] = 0.2

# 3 dress
rt(3, rDress, 2.4, 2); rt(3, rDress2, 1.6, 2)
rt(3, rFlare, 0.9); rt(3, rFlare2, 0.6)
rt(3, rTall, 0.7); rt(3, rNTop, 0.5); rt(3, rH, 0.4)
rt(3, rCo, -0.6, 2); rt(3, rCl, -0.3)
rt(3, rKn, -0.5, 2); rt(3, rTr, -1.4, 2)
rt(3, rSh, -0.4)
rt(3, rShoe, -1.0, 2); rt(3, rBag, -0.6, 2)
BIAS[3] = 0.0

# 4 coat
rt(4, rCo, 2.6, 2)
rt(4, rCl, 0.7); rt(4, rClHi, 0.5); rt(4, rGcl, 0.5)
rt(4, rH, 0.7); rt(4, rCol, 0.4)
rt(4, rSh, -0.7); rt(4, rDrop, -0.4)
rt(4, rDress, -0.5, 2); rt(4, rTr, -1.3, 2)
rt(4, rKn, -0.4, 2)
rt(4, rShirt, -0.4, 2); rt(4, rPlaid, -0.2)
rt(4, rShoe, -1.0, 2); rt(4, rBoot, -0.6, 2)
BIAS[4] = 0.05

# 5 sandal
rt(5, rSandal, 2.8, 2); rt(5, rSandal2, 2.0, 2)
rt(5, rThin, 1.3); rt(5, rThin2, 0.9); rt(5, rSparse, 1.1)
rt(5, rHeelL, 0.7); rt(5, rHeelR, 0.7)
rt(5, rSole, 0.5); rt(5, rEmptyTop, 0.5); rt(5, rEmptyBot, 0.5)
rt(5, rH, -1.4); rt(5, rFill, -0.6)
rt(5, rSolid, -0.7); rt(5, rBoot, -0.7, 2)
rt(5, rNeck, -0.8)
BIAS[5] = 0.0

# 6 shirt
rt(6, rShirt, 2.2, 2); rt(6, rShirt2, 1.6, 2); rt(6, rShirt3, 1.4, 2)
rt(6, rPlk, 0.9); rt(6, rPlaid, 0.7); rt(6, rCol, 0.5); rt(6, rGcol, 0.4)
rt(6, rCtrFine, 0.5)
rt(6, rCo, -0.5, 2); rt(6, rCl, -0.3)
rt(6, rKn, -0.4, 2)
rt(6, rDress, -0.5, 2); rt(6, rTr, -1.3, 2)
rt(6, rShoe, -1.0, 2); rt(6, rBoot, -0.6, 2)
BIAS[6] = 0.15

# 7 sneaker
rt(7, rShoeSolid, 3.0, 2); rt(7, rShoe, 1.4, 2)
rt(7, rSolid, 1.0); rt(7, rSole, 0.7); rt(7, rMid, 0.5)
rt(7, rEmptyTop, 0.6); rt(7, rEmptyBot, 0.6); rt(7, rLow, 0.6)
rt(7, rH, -1.6); rt(7, rThin, -1.0); rt(7, rSparse, -0.8)
rt(7, rBoot, -1.0, 2); rt(7, rBag, -0.7, 2)
rt(7, rHeelL, -0.4); rt(7, rHeelR, -0.4)
rt(7, rNeck, -0.8)
BIAS[7] = 0.0

# 8 bag
rt(8, rBag, 2.8, 2); rt(8, rBag2, 1.8, 2)
rt(8, rHole, 0.9); rt(8, rHole2, 0.7); rt(8, rHoleTop, 0.5)
rt(8, rBagBody, 0.6); rt(8, rBagComp, 0.5)
rt(8, rNeck, -0.4); rt(8, rSh, -0.5)
rt(8, rTr, -1.3, 2); rt(8, rDress, -0.6, 2)
rt(8, rShoeSolid, -0.7, 2)
rt(8, rThin, -0.4)
BIAS[8] = 0.0

# 9 ankle boot
rt(9, rBoot, 2.8, 2)
rt(9, rShaftL, 0.7); rt(9, rShaftR, 0.7)
rt(9, rShaftHL, 0.4); rt(9, rShaftHR, 0.4)
rt(9, rSole, 0.4); rt(9, rSolid, 0.5)
rt(9, rH, 0.4); rt(9, rW, 0.4)
rt(9, rShoeSolid, -0.9, 2); rt(9, rSandal, -1.0, 2)
rt(9, rThin, -0.7); rt(9, rBag, -0.7, 2)
rt(9, rKn, -0.5, 2); rt(9, rCo, -0.5, 2)
rt(9, rNeck, -1.0); rt(9, rCol, -0.6); rt(9, rSh, -0.6)
BIAS[9] = 0.0

# ---- corrected sleeve / hem evidence (added on top of the rules above) -------
# short sleeves: outer columns empty low down, hem solid in the centre
rt(0, rSlvDrop, 1.6); rt(0, rHemCtr, 0.8)
rt(0, rSlv, -1.0); rt(0, rHemGap, -0.8)
# long sleeves hanging to the hem: pullover / coat / shirt
for k in (2, 4, 6):
    rt(k, rSlv, 1.1); rt(k, rSlv2, 0.6); rt(k, rHemGap, 0.9)
    rt(k, rSlvDrop, -0.8); rt(k, rColStep, 0.4)
# trousers: narrow, no shoulder step, centre leg gap, no cuffs at the sides
rt(1, rLegD, 1.0); rt(1, rLegE, 0.8)
rt(1, rHemGap, -1.0); rt(1, rSlv, -0.8); rt(1, rColStep, -0.6)
# dress: no sleeves in the outer columns at mid height, flared hem
rt(3, rSlv, -0.7); rt(3, rHemGap, -0.6); rt(3, rHemCtr, 0.5)
# footwear: boots reach the top and bottom borders, sneakers do not.  Garments
# also reach both borders, so the boot bonus must be gated against garment cues.
rt(9, rTopOcc, 0.3); rt(9, rBotOcc, 0.3)
rt(9, rColStep, -1.6); rt(9, rSlv, -1.0); rt(9, rHemGap, -0.8)
rt(9, rLegD, -1.0); rt(9, rLegE, -0.8); rt(9, rFlare, -0.8)
rt(7, rTopOcc, -0.9); rt(7, rBotOcc, -0.9)
rt(5, rTopOcc, -0.5); rt(5, rBotOcc, -0.5)
# solidity proxy: fill minus 0.6 * (occupied height fraction)
for k, w in ((7, 1.6), (5, -1.4), (9, -0.3)):
    raw(k, "fill", w); raw(k, "rsum0", -0.6 * w)

# ---- final balance: shirt was absorbing every long-sleeved garment ----------
rt(6, rPlk, -0.5); rt(6, rPlaid, -0.4); rt(6, rCtrFine, -0.3)
rt(6, rShirt, -0.9, 2); rt(6, rShirt2, -0.7, 2); rt(6, rShirt3, -0.6, 2)
rt(4, rCo, 0.9, 2); rt(4, rGcl, 0.5); rt(4, rClHi, 0.4)
rt(2, rKn, 0.8, 2); rt(2, rSlv, 0.5)
rt(9, rBoot, 0.9, 2); rt(9, rH, 0.6)
rt(7, rH, -0.8); rt(7, rShoeSolid, -0.7, 2)

# ---- assemble the four heads -------------------------------------------------
W1 = np.zeros((len(H1), NV), np.float32); b1 = np.zeros(len(H1), np.float32)
for i, (nm, terms, bias) in enumerate(H1):
    for f, c in terms:
        W1[i, VIDX[f]] += c
    b1[i] = bias
h1 = net.dense("h1", V, W1, b1)
h1r = net.relu("h1r", h1)

W2 = np.zeros((len(H2), len(H1)), np.float32); b2 = np.zeros(len(H2), np.float32)
for i, (nm, terms, bias) in enumerate(H2):
    for t, c in terms:
        W2[i, H1IDX[t]] += c
    b2[i] = bias
h2 = net.dense("h2", h1r, W2, b2)
h2r = net.relu("h2r", h2)


def head(nm, src, OD, names, n):
    W = np.zeros((10, n), np.float32); b = np.zeros(10, np.float32)
    for k in range(10):
        for t, w in OD[k].items():
            W[k, names[t]] += w
    return net.dense(nm, src, W, b)


o1 = head("o1", h1r, O1, H1IDX, len(H1))
o2 = head("o2", h2r, O2, H2IDX, len(H2))
o0 = head("o0", V, O0, VIDX, NV)
for k in range(10):
    net.w["o1.b"][k] = BIAS[k]
out = net.add("out", o1, o2, o0, bcls)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--shapes":
        for L in net.layers:
            print(f"{L['name']:12s} {L['op']:8s} {net.shapes[L['name']]}")
        print("weights:", sum(v.size for v in net.w.values()))
        print("diag:", len(FEATS), "tex:", len(TF), "templates:", NT, "models:", NB,
              "h1:", len(H1), "h2:", len(H2))
    else:
        dest = sys.argv[1] if len(sys.argv) > 1 else "."
        net.save(dest)
        print("wrote", dest, "layers:", len(net.layers),
              "weights:", sum(v.size for v in net.w.values()))
