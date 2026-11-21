"""Build weights for a hand-designed person detector on 96x96 colour photos.

Every number below comes from a formula or from general knowledge about people
in everyday photos: luminance weights, Sobel / Gaussian-derivative edge
filters, centre-surround (DoG) blob filters, the classic skin-chromaticity
rules (R>G>B with R-G and R-B in a moderate band, not too dark / not blown
out), and a stick-figure shape model -- a head blob above a shoulder line,
vertical torso sides, legs below, or a compact skin-coloured face patch.
No file is read and no statistic of any image is used.

Architecture
------------
Pixel front end (conv + relu + min(x,1) caps, so each feature channel lies in
[0,1] and reads as "fraction of this cue present"):
  * 6 unsigned gradient-orientation bins + total edge energy,
  * a skin mask from chromaticity gates,
  * dark/bright centre-surround blob masks at four head-size scales,
  * colour saturation and mean luminance.
Two HOG-like cell grids (avgpool 4x4 and 8x8).  Six detector branches with
windows of 16, 24, 32, 48, 72 and 96 px correlate the cell grids with hand
stamped person templates (standing, sitting/upper-body, face, skin patch).
Each template response S becomes a fire map
    fire = relu(min(S,1) + 0.6*min(relu((z(S)-t)/1.6),1) - 0.9)
where z() is the per-channel spatial norm: a detection needs an absolute
template match and some salience inside the image.  Global max- and mean-
pooling of each fire map gives per-template evidence; a hand-picked linear
readout (dense layers summed by add) maps the evidence vector to the two
class scores, the class-0 score being the constant 0.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

LAYERS = []
WEIGHTS = {}


def layer(name, op, inputs, **settings):
    d = {"name": name, "op": op, "inputs": list(inputs)}
    d.update(settings)
    LAYERS.append(d)
    return name


def conv(name, inp, weights, bias, stride=1, padding=0):
    weights = np.asarray(weights, np.float64)
    o, c, kh, kw = weights.shape
    layer(name, "conv", [inp], out=o, kernel=[kh, kw], stride=stride, padding=padding)
    WEIGHTS[name + ".W"] = weights.astype(np.float32)
    WEIGHTS[name + ".b"] = np.asarray(bias, np.float64).astype(np.float32)
    return name


def dense(name, inp, w, b):
    w = np.asarray(w, np.float64)
    layer(name, "dense", [inp], out=w.shape[0])
    WEIGHTS[name + ".W"] = w.astype(np.float32)
    WEIGHTS[name + ".b"] = np.asarray(b, np.float64).astype(np.float32)
    return name


def relu(name, inp):
    return layer(name, "relu", [inp])


def pool(name, inp, size, stride, kind):
    return layer(name, kind, [inp], size=list(size), stride=list(stride))


def add(name, inputs):
    return layer(name, "add", inputs)


def cap(name, inp, c):
    """min(max(x,0),1) channelwise: relu(x) - relu(x-1)."""
    r = relu(name + "_a", inp)
    o = conv(name + "_b", inp, np.eye(c).reshape(c, c, 1, 1), -np.ones(c))
    r2 = relu(name + "_c", o)
    n = conv(name + "_d", r2, -np.eye(c).reshape(c, c, 1, 1), np.zeros(c))
    return add(name, [r, n])


def cap1(name, inp, c):
    """min(max(x,0),1) for a flat (c,) vector."""
    r = relu(name + "_a", inp)
    o = dense(name + "_b", inp, np.eye(c), -np.ones(c))
    r2 = relu(name + "_c", o)
    n = dense(name + "_d", r2, -np.eye(c), np.zeros(c))
    return add(name, [r, n])


# ---------------------------------------------------------------- front end
LUM = np.array([0.299, 0.587, 0.114])

# 1x1 colour projections, each with its own bias so relu() applies a gate
lin_w = np.zeros((23, 3, 1, 1))
lin_b = np.zeros(23)
def put1x1(i, wr, wg, wb, b=0.0):
    lin_w[i, 0, 0, 0] = wr
    lin_w[i, 1, 0, 0] = wg
    lin_w[i, 2, 0, 0] = wb
    lin_b[i] = b
# hat(band) gates: relu(x-r0) - relu(x-r1) - relu(x-f0) + relu(x-f1)
put1x1(0, 1, -1, 0, -0.03)    # R-G hat: skin band, rejects yellow (banana)
put1x1(1, 1, -1, 0, -0.09)
put1x1(2, 1, -1, 0, -0.22)
put1x1(3, 1, -1, 0, -0.30)
put1x1(4, 1, 0, -1, -0.05)    # R-B hat: rejects bread / wood / sand tones
put1x1(5, 1, 0, -1, -0.12)
put1x1(6, 1, 0, -1, -0.30)
put1x1(7, 1, 0, -1, -0.40)
put1x1(8, 0, 1, -1, -0.01)    # G-B hat
put1x1(9, 0, 1, -1, -0.03)
put1x1(10, 0, 1, -1, -0.10)
put1x1(11, 0, 1, -1, -0.16)
put1x1(12, *LUM, -0.10)       # L hat: not too dark, not blown out
put1x1(13, *LUM, -0.18)
put1x1(14, *LUM, -0.80)
put1x1(15, *LUM, -0.92)
put1x1(16, 1, -1, 0)          # |R-G| parts
put1x1(17, -1, 1, 0)
put1x1(18, 0, 1, -1)          # |G-B| parts
put1x1(19, 0, -1, 1)
put1x1(20, 1, 0, -1)          # |R-B| parts
put1x1(21, -1, 0, 1)
put1x1(22, *LUM)              # luminance
lin = conv("lin", "image", lin_w, lin_b)
linr = relu("linr", lin)

GX = np.array([[-1., 0, 1], [-2., 0, 2], [-1., 0, 1]]) / 8.0
GY = GX.T
gw = np.zeros((2, 3, 3, 3))
for c in range(3):
    gw[0, c] = LUM[c] * GX
    gw[1, c] = LUM[c] * GY
grad = conv("grad", "image", gw, np.zeros(2), padding=1)

dirs_w = np.zeros((12, 2, 1, 1))
for i in range(12):
    th = i * np.pi / 6.0
    dirs_w[i, 0, 0, 0] = np.cos(th)
    dirs_w[i, 1, 0, 0] = np.sin(th)
dirs = conv("dirs", grad, dirs_w, np.zeros(12))
dirr = relu("dirr", dirs)
bins_w = np.zeros((7, 12, 1, 1))
for j in range(6):
    bins_w[j, j, 0, 0] = 1.0
    bins_w[j, j + 6, 0, 0] = 1.0
for d in (0, 6, 3, 9):
    bins_w[6, d, 0, 0] = 0.5
bins = conv("bins", dirr, bins_w, np.zeros(7))
binsc = cap("binsc", bins, 7)          # orientation bins + energy, in [0,1]

# skin: sum of four hat gates (each in [0,1]), thresholded at 2.6 of 4
sk_w = np.zeros((1, 23, 1, 1))
sk_b = np.zeros(1)
A, B, C, E = 16.7 / 1.4, 14.3 / 1.4, 12.5 / 1.4, 50.0 / 1.4
for base, g in ((0, A), (4, B), (8, E), (12, C)):
    sk_w[0, base, 0, 0], sk_w[0, base + 1, 0, 0] = g, -g
    sk_w[0, base + 2, 0, 0], sk_w[0, base + 3, 0, 0] = -g, g
sk_b[0] = -2.6 / 1.4
skin = relu("skin", conv("skinpre", linr, sk_w, sk_b))
skinc = cap("skinc", skin, 1)

sl_w = np.zeros((2, 23, 1, 1))
for i in range(16, 22):
    sl_w[0, i, 0, 0] = 1.0 / 3.0
sl_w[1, 22, 0, 0] = 1.0
satl = conv("satl", linr, sl_w, np.zeros(2))
satlc = cap("satlc", satl, 2)


def gauss(k, s):
    ax = np.arange(k) - k // 2
    g = np.exp(-(ax ** 2) / (2.0 * s * s))
    G = np.outer(g, g)
    return G / G.sum()


def dog_kernel(k, sc, ss):
    return gauss(k, sc) - gauss(k, ss)


def blob_branch(tag, inp, k, sigma_c, sigma_s, gain=None, thr_contrast=0.15):
    """+- centre-surround blob mask, capped to [0,1]."""
    d = dog_kernel(k, sigma_c, sigma_s)
    disc = np.zeros((k, k))
    r = 2.0 * sigma_c
    yy, xx = np.mgrid[0:k, 0:k]
    disc[((yy - k // 2) ** 2 + (xx - k // 2) ** 2) <= r * r] = 1.0
    resp = float((d * disc).sum())          # response to a unit-contrast disc
    g = 1.0 / (resp * 0.35) if gain is None else gain
    t = g * resp * thr_contrast
    w = np.zeros((2, 1, k, k))
    w[0, 0] = d
    w[1, 0] = -d
    raw = conv(tag, inp, w, np.zeros(2), padding=k // 2)
    rr = relu(tag + "r", raw)
    sc_ = conv(tag + "s", rr, g * np.eye(2).reshape(2, 2, 1, 1), -t * np.ones(2))
    sr = relu(tag + "sr", sc_)
    return cap(tag + "c", sr, 2)


l1 = conv("l1", "image", LUM.reshape(1, 3, 1, 1), np.zeros(1))
im2 = pool("im2", "image", [2, 2], [2, 2], "avgpool")
l2 = conv("l2", im2, LUM.reshape(1, 3, 1, 1), np.zeros(1))
im4 = pool("im4", "image", [4, 4], [4, 4], "avgpool")
l4 = conv("l4", im4, LUM.reshape(1, 3, 1, 1), np.zeros(1))
im8 = pool("im8", "image", [8, 8], [8, 8], "avgpool")
l8 = conv("l8", im8, LUM.reshape(1, 3, 1, 1), np.zeros(1))

bs = blob_branch("bs", l1, 3, 0.8, 1.6)     # heads ~4-10 px
bm = blob_branch("bm", l2, 3, 0.8, 1.6)     # ~10-24 px
bl = blob_branch("bl", l4, 3, 0.8, 1.6)     # ~24-48 px
bx = blob_branch("bx", l8, 3, 0.8, 1.6)     # ~48-96 px

# ------------------------------------------------------------ cell grids
c4E = pool("c4E", binsc, [4, 4], [4, 4], "avgpool")
c8E = pool("c8E", binsc, [8, 8], [8, 8], "avgpool")
c4S = pool("c4S", skinc, [4, 4], [4, 4], "avgpool")
c8S = pool("c8S", skinc, [8, 8], [8, 8], "avgpool")
c4L = pool("c4L", satlc, [4, 4], [4, 4], "avgpool")
c8L = pool("c8L", satlc, [8, 8], [8, 8], "avgpool")
bs4 = pool("bs4", bs, [4, 4], [4, 4], "avgpool")
bs8 = pool("bs8", bs, [8, 8], [8, 8], "avgpool")
bm4 = pool("bm4", bm, [2, 2], [2, 2], "avgpool")
bm8 = pool("bm8", bm, [4, 4], [4, 4], "avgpool")
bl8 = pool("bl8", bl, [2, 2], [2, 2], "avgpool")

CH = {"E": 7, "S": 1, "L": 2, "bs": 2, "bm": 2, "bl": 2, "bx": 2}
GRID4 = {"E": c4E, "S": c4S, "L": c4L, "bs": bs4, "bm": bm4, "bl": bl}
GRID8 = {"E": c8E, "S": c8S, "L": c8L, "bs": bs8, "bm": bm8, "bl": bl8, "bx": bx}

# ------------------------------------------------------- person templates
def blank(k):
    return {g: np.zeros((CH[g], k, k)) for g in CH}


def head_taps(T, k, blobs, w_dark=0.9, w_bright=0.6, w_skin=0.5):
    hc, hw = k // 2, max(1, k // 8)
    hr = max(1, (k + 2) // 5)
    for r in range(hr):
        for c in range(hc - hw, hc + hw + 1):
            for g in blobs:
                T[g][0, r, c] += w_dark               # hair / head blob
                T[g][1, r, c] += w_bright
            T["E"][6, r, c] += 0.3
            T["S"][0, r, c] += w_skin                 # face
        for c in list(range(max(0, hc - hw - 2), hc - hw)) + \
                 list(range(hc + hw + 1, min(k, hc + hw + 3))):
            T["E"][6, r, c] -= 0.4                    # head narrower than body
    return hr


def background(T, k, cl, cr, rs):
    for r in range(k):
        for c in list(range(0, max(0, cl - 1))) + list(range(min(k, cr + 1), k)):
            T["E"][6, r, c] -= 0.25                   # figure-ground contrast
    for g in ("bs", "bm", "bl", "bx"):
        if g in T:
            for r in range(k):
                for c in list(range(0, max(0, cl - 1))) + list(range(min(k, cr + 1), k)):
                    T[g][0, r, c] -= 0.15
                    T[g][1, r, c] -= 0.15


def t_stand(k, blobs):
    T = blank(k)
    cl, cr, hc = k // 4, k - k // 4, k // 2
    rs = head_taps(T, k, blobs)
    for c in range(cl, cr):
        T["E"][3, rs, c] += 0.6                       # shoulder line
    for r in range(rs, k):
        T["E"][0, r, cl] += 0.4                       # torso / leg sides
        T["E"][0, r, cr - 1] += 0.4
    for r in range(rs, 3 * k // 4):
        for c in range(cl + 1, cr - 1):
            T["E"][6, r, c] += 0.1                    # clothing texture
            T["S"][0, r, c] += 0.2                    # arms / hands
    for r in range(3 * k // 4, k):
        for c in (cl, hc, cr - 1):
            T["E"][0, r, c] += 0.35                   # leg edges + gap
    background(T, k, cl, cr, rs)
    return T


def t_sit(k, blobs):
    T = blank(k)
    cl, cr = k // 4, k - k // 4
    rs = head_taps(T, k, blobs)
    for c in range(max(0, cl - 1), min(k, cr + 1)):
        T["E"][3, rs, c] += 0.6                       # broad shoulder line
    for r in range(rs, k):
        T["E"][0, r, cl] += 0.35
        T["E"][0, r, cr - 1] += 0.35
        for c in range(cl + 1, cr - 1):
            T["E"][6, r, c] += 0.1
            T["S"][0, r, c] += 0.3                    # hands on lap
    for c in range(cl, cr):
        T["E"][3, k - 1, c] += 0.3                    # knees / seat line
    background(T, k, cl, cr, rs)
    return T


def t_face(k, blobs):
    T = blank(k)
    c0, c1 = k // 3, k - k // 3
    for r in range(c0, c1 + 1):
        for c in range(c0, c1 + 1):
            T["S"][0, r, c] += 1.0                    # skin-coloured core
    for r in range(0, c0):
        for c in range(c0, c1 + 1):
            for g in blobs:
                T[g][0, r, c] += 0.8                  # hair above forehead
                T[g][1, r, c] += 0.3
    for c in range(c0, c1 + 1):
        T["E"][6, (c0 + c1) // 2, c] += 0.3           # eyes / mouth texture
    for r in range(0, c0 + 1):
        for c in list(range(0, c0)) + list(range(c1 + 1, k)):
            T["S"][0, r, c] -= 0.8                    # no skin at top corners
    for r in range(c0, c1 + 1):
        for c in list(range(0, max(0, c0 - 1))) + list(range(min(k, c1 + 2), k)):
            T["S"][0, r, c] -= 0.4                    # face is a compact patch
    return T


def t_skin(k):
    T = blank(k)
    c0, c1 = k // 3, k - k // 3
    for r in range(c0, c1 + 1):
        for c in range(c0, c1 + 1):
            T["S"][0, r, c] += 1.0
    for r in range(max(0, c0 - 1), min(k, c1 + 2)):
        for c in range(max(0, c0 - 1), min(k, c1 + 2)):
            if r < c0 or r > c1 or c < c0 or c > c1:
                T["S"][0, r, c] -= 0.4                # compact, not a wall
    return T


# ------------------------------------------------------------ branches
THETA = 0.75
final_inputs = []


def presence(name, tensor, c):
    """Bounded per-cell cue presence: p = min(max(z(channel), 0)/3, 1).

    z-scoring makes every cue scale-free; capping at 3 sigma keeps a single
    salient cell from dominating a template sum, so a template only fires on
    a *configuration* of cells matching the person model.
    """
    z = layer(name + "_z", "norm", [tensor])
    zr = relu(name + "_zr", z)
    zs = conv(name + "_zs", zr, np.eye(c).reshape(c, c, 1, 1) / 2.5, np.zeros(c))
    return cap(name + "_p", zs, c)


def branch(tag, k, grid, temps, h):
    """Majority-vote matched filters on bounded presence maps.

    theta = 0.4 * (sum of positive template weights): a window fires once
    ~40% of the expected person evidence is present in the right layout.
    """
    n = len(temps)
    raw = {g: np.stack([t[0][g] for t in temps]) for g in grid}
    pos = np.zeros(n)
    for g in grid:
        pos += np.maximum(raw[g], 0).sum(axis=(1, 2, 3))
    theta = np.maximum(0.5 * pos, 1e-6)
    parts = []
    for g in grid:
        parts.append(conv(f"{tag}_{g}", grid[g], raw[g] / theta[:, None, None, None],
                          np.zeros(n)))
    S = add(tag + "_sum", parts)
    f0 = conv(tag + "_f0", S, np.eye(n).reshape(n, n, 1, 1), np.full(n, -0.55))
    fire = cap(tag + "_fire", relu(tag + "_fr", f0), n)
    mx = pool(tag + "_max", fire, [h, h], [1, 1], "maxpool")
    av = pool(tag + "_avg", fire, [h, h], [1, 1], "avgpool")
    dm = dense(tag + "_dm", mx, np.stack([np.zeros(n), np.array([t[1] for t in temps])]), np.zeros(2))
    da = dense(tag + "_da", av, np.stack([np.zeros(n), np.array([t[2] for t in temps])]), np.zeros(2))
    final_inputs.extend([dm, da])


PC4 = {g: presence("p4" + g, GRID4[g], CH[g]) for g in GRID4}
PC8 = {g: presence("p8" + g, GRID8[g], CH[g]) for g in GRID8}

BR = [
    ("b1", 4, PC4, [(t_stand(4, ["bs"]), 0.6, 0.3), (t_skin(4), 0.7, 0.4)], 21),
    ("b2", 6, PC4, [(t_stand(6, ["bs", "bm"]), 0.6, 0.3), (t_sit(6, ["bs", "bm"]), 0.6, 0.3),
                    (t_face(6, ["bs", "bm"]), 1.0, 0.4)], 19),
    ("b3", 8, PC4, [(t_stand(8, ["bm", "bl"]), 0.6, 0.3), (t_sit(8, ["bm", "bl"]), 0.6, 0.3),
                    (t_face(8, ["bm", "bl"]), 1.0, 0.4)], 17),
    ("b4", 6, PC8, [(t_stand(6, ["bl", "bx"]), 0.6, 0.3), (t_sit(6, ["bl", "bx"]), 0.6, 0.3),
                    (t_face(6, ["bl", "bx"]), 1.0, 0.4)], 7),
    ("b5", 9, PC8, [(t_stand(9, ["bx"]), 0.6, 0.3), (t_sit(9, ["bx"]), 0.6, 0.3)], 4),
]
for tag, k, grid, temps, h in BR:
    branch(tag, k, grid, temps, h)

# global skin-fraction gist
gp = pool("gist", skinc, [96, 96], [96, 96], "avgpool")
g1 = dense("g1", gp, np.array([[8.0]]), np.zeros(1))
g1r = relu("g1r", g1)
gc = cap1("gc", g1r, 1)
g2 = dense("g2", gc, np.array([[0.0], [0.8]]), np.array([0.0, -THETA]))
# (THETA set below in the readout constants)
final_inputs.append(g2)

scores = add("scores", final_inputs)

net = {"layers": LAYERS}
(OUT / "network.json").write_text(json.dumps(net))
np.savez(OUT / "weights.npz", **WEIGHTS)
total = sum(int(v.size) for v in WEIGHTS.values())
print(f"wrote {len(LAYERS)} layers, {total} weights to {OUT}")
