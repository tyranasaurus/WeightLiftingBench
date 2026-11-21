# build.py: writes network.json and weights.npz for a hand-designed clothing classifier. Reads no files.
# Hand-made shape models of the 10 clothing classes, drawn from polygons (general knowledge of garment shapes).
# Every coordinate is typed by hand in a normalised item box: u in [0,1] left->right, v in [0,1] top->bottom.
# The box is then placed on the 28x28 canvas the way the shop pipeline does: longest side 28, centred.
import numpy as np

SS = 4  # supersampling per pixel side


def _grid():
    g = (np.arange(28 * SS) + 0.5) / SS
    return np.meshgrid(g, g)  # X (cols), Y (rows)


GX, GY = _grid()


def poly_cover(pts):
    """Coverage (28x28, 0..1) of a closed polygon given in canvas coordinates (x, y)."""
    pts = np.asarray(pts, float)
    inside = np.zeros(GX.shape, bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        cond = (GY >= min(y1, y2)) & (GY < max(y1, y2))
        xc = x1 + (GY - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (GX < xc)
    return inside.reshape(28, SS, 28, SS).mean(axis=(1, 3))


def sym(R):
    """Right-half outline from top centre clockwise to a centre point -> full symmetric outline."""
    return list(R) + [(1 - u, v) for (u, v) in reversed(R[1:-1])]


def mirror(P):
    return [(1 - u, v) for (u, v) in P]


def box(r):
    """Item box for aspect r = width / height, longest side 28, centred."""
    if r >= 1:
        w, h = 28.0, 28.0 / r
    else:
        w, h = 28.0 * r, 28.0
    return 14 - w / 2, 14 - h / 2, w, h


def place(P, r):
    x0, y0, w, h = box(r)
    return [(x0 + u * w, y0 + v * h) for (u, v) in P]


def blur(a, s):
    if s <= 0:
        return a
    k = np.arange(-3, 4)
    g = np.exp(-k ** 2 / (2 * s * s)); g /= g.sum()
    p = np.pad(a, ((0, 0), (3, 3)))
    t = sum(g[i] * p[:, i:i + 28] for i in range(7))          # blur along x
    p = np.pad(t, ((3, 3), (0, 0)))
    return sum(g[i] * p[i:i + 28, :] for i in range(7))       # blur along y


def render(parts, r, s=0.9):
    """parts: list of (outline in (u,v), density). Positive density adds item (union); negative density cuts
    (multiplies the probability by 1 - |d| * coverage), used for places where the item may or may not show."""
    q = np.ones((28, 28)); cut = np.ones((28, 28))
    for P, d in parts:
        c = poly_cover(place(P, r))
        if d > 0:
            q *= 1 - d * c
        else:
            cut *= 1 + d * c
    return np.clip(blur((1 - q) * cut, s), 0, 1)


def arc(cx, cy, a, b, t0, t1, n=7):
    t = np.linspace(t0, t1, n)
    return [(cx + a * np.sin(x), cy - b * np.cos(x)) for x in t]


NECK = [(0.39, 0.0), (0.61, 0.0), (0.56, 0.07), (0.44, 0.07)]     # neck opening: may show the back of the garment


# ---------------------------------------------------------------- class shape models
def trouser(r, taper, gap, crotch):
    R = [(0.5, 0.0), (0.98, 0.0), (1.0, 0.18), (1.0 - taper, 1.0), (0.5 + gap / 2, 1.0), (0.5, crotch)]
    return [(sym(R), 0.97)]


def dress(r, kind, sleeve):
    if kind == "aline":       # fitted top, flared hem; the hem is the widest part
        sh, bw, ww = 0.20, 0.30, 0.25
        R = [(0.5, 0.0), (0.5 + sh, 0.0), (0.5 + bw, 0.17), (0.5 + ww, 0.36), (1.0, 1.0), (0.5, 1.0)]
    elif kind == "sheath":    # nearly straight sides
        sh, bw, ww = 0.32, 0.50, 0.44
        R = [(0.5, 0.0), (0.5 + sh, 0.0), (0.5 + bw, 0.17), (0.5 + ww, 0.38), (0.5 + bw, 1.0), (0.5, 1.0)]
    else:                     # shift / tunic: straight from the armpit, slight flare
        sh, bw, ww = 0.28, 0.40, 0.40
        R = [(0.5, 0.0), (0.5 + sh, 0.0), (0.5 + bw, 0.17), (0.5 + ww, 0.4), (0.5 + 0.5, 1.0), (0.5, 1.0)]
    parts = [(sym(R), 0.97), (NECK, -0.6)]
    if sleeve:                # short cap sleeves at the shoulders
        S = [(0.5 + sh - 0.02, 0.0), (0.5 + bw + 0.16, 0.12), (0.5 + bw + 0.10, 0.2), (0.5 + bw, 0.2)]
        S = [(min(u, 1.0), v) for (u, v) in S]
        parts += [(S, 0.9), (mirror(S), 0.9)]
    return parts


def tee(r, bw, sl):
    """Short-sleeved top: body of half-width bw, sleeves angled down and out to the box edge."""
    st, sb = sl            # sleeve top-outer and bottom-outer corner heights
    ap = sb - 0.05
    R = [(0.5, 0.0), (0.62, 0.0), (0.7, 0.03), (1.0, st), (0.95, sb), (0.5 + bw, ap), (0.5 + bw, 1.0), (0.5, 1.0)]
    return [(sym(R), 0.97), (NECK, -0.5)]


def tank(r, deep):
    """Sleeveless top: narrow straps, deep scooped neck and armholes, straight sides to the hem."""
    nh, nd = (0.2, 0.22) if deep else (0.15, 0.13)
    s = nh + 0.08
    R = [(0.5, nd), (0.5 + nh, 0.0), (0.5 + s, 0.0), (0.5 + s + 0.02, 0.12), (0.86, 0.24), (1.0, 0.32), (1.0, 1.0), (0.5, 1.0)]
    return [(sym(R), 0.97)]


def longtop(r, cw, nd, sd, se=1.0, bw=0.35, hem="straight", gapline=False):
    """Long-sleeved top seen flat: domed shoulders (collar or hood fills the neck), sleeves hanging along the body.
    se: where the sleeves end (1 = at the bottom edge). With se = 1 the body hem may end nd above the cuffs, leaving a
    notch between the sleeves; with se < 1 (long coats) the body continues below the cuffs down to the hem."""
    R = [(0.5, 0.0)] + arc(0.6, sd, 0.4, sd, 0.0, np.pi / 2)
    if hem == "tail":     # shirt: cuffs a little above the bottom, rounded shirt-tail hem lowest at the centre
        R += [(1.0, 0.9), (1 - cw, 0.91), (1 - cw - 0.04, 0.95), (0.62, 0.99), (0.5, 1.0)]
    elif se < 0.99:
        R += [(1.0, se), (1 - cw, se + 0.02), (0.5 + bw, se + 0.02), (0.5 + bw + 0.02, 1.0), (0.5, 1.0)]
    else:
        R += [(1.0, 1.0), (1 - cw, 1.0)] + ([(1 - cw, 1 - nd), (0.5, 1 - nd)] if nd > 0 else [(0.5, 1.0)])
    parts = [(sym(R), 0.97)]
    if gapline:           # a dark seam or gap between sleeve and body
        G = [(1 - cw - 0.03, 0.4), (1 - cw + 0.005, 0.4), (1 - cw + 0.005, 0.97), (1 - cw - 0.03, 0.97)]
        parts += [(G, -0.45), (mirror(G), -0.45)]
    return parts


def sneaker(r, toe, tongue):
    P = [(0.0, 0.72), (0.06, toe), (0.35, toe - 0.12), (0.6, tongue), (0.72, 0.03), (0.82, 0.1), (0.93, 0.0), (1.0, 0.12),
         (1.0, 0.84), (0.97, 1.0), (0.04, 1.0), (0.0, 0.88)]
    return [(P, 0.97)]


def boot(r, front, heel):
    """Ankle boot: toe at the left, concave instep rising to the shaft on the right, which reaches the top."""
    top = [(0.0, 0.8), (0.05, 0.62), (0.22, 0.52), (front - 0.1, 0.36), (front - 0.02, 0.16), (front, 0.0), (0.95, 0.0),
           (0.95, 0.45), (1.0, 0.82)]
    if heel:   # raised heel: a gap under the arch
        bot = [(0.98, 1.0), (0.80, 1.0), (0.78, 0.86), (0.45, 0.9), (0.30, 1.0), (0.03, 1.0), (0.0, 0.92)]
    else:
        bot = [(0.97, 1.0), (0.03, 1.0), (0.0, 0.92)]
    return [(top + bot, 0.97)]


def sandal(r, heel, dens):
    if not heel:   # flat: a sole strip with sparse straps above it
        sole = [(0.0, 0.78), (1.0, 0.76), (1.0, 1.0), (0.0, 1.0)]
        straps = [(0.05, 0.8), (0.3, 0.35), (0.6, 0.0), (0.9, 0.0), (1.0, 0.3), (1.0, 0.78)]
        return [(sole, 0.92), (straps, dens)]
    # heeled: sole climbs from the toe to the heel seat, heel post on the right, sparse straps above
    sole = [(0.0, 0.86), (0.32, 0.84), (0.78, 0.42), (1.0, 0.38), (1.0, 0.5), (0.8, 0.54), (0.36, 1.0), (0.0, 1.0)]
    post = [(0.84, 0.45), (0.98, 0.45), (0.97, 1.0), (0.88, 1.0)]
    straps = [(0.06, 0.84), (0.3, 0.45), (0.62, 0.0), (1.0, 0.0), (1.0, 0.4), (0.78, 0.42), (0.32, 0.84)]
    return [(sole, 0.9), (post, 0.75), (straps, dens)]


def bag(r, body_top, trap, handle):
    """Bag: solid body from v = body_top down, optional top inset (trapezoid), optional handle loop above."""
    B = [(trap, body_top), (1 - trap, body_top), (1.0, 1.0), (0.0, 1.0)]
    parts = [(B, 0.97)]
    if handle and body_top > 0.1:
        t = np.linspace(0, np.pi, 13)
        hw, th = 0.25, 0.05
        outer = [(0.5 - (hw + th) * np.cos(a), body_top - (body_top) * np.sin(a)) for a in t]
        inner = [(0.5 - hw * np.cos(a), body_top - (body_top - th * 1.5) * np.sin(a)) for a in t[::-1]]
        parts.append((outer + inner, 0.75))
    return parts


# class-level offsets (log-likelihood units) for the long-sleeved shapes shared by pullover, coat and shirt: the
# outline alone does not tell them apart, so a plain long-sleeved top is a pullover unless interior cues (a zip or
# front-opening line, a check pattern) added later in the network say coat or shirt. The same offsets are used for
# every shoulder slope (a rule giving sloped shoulders to coats was tried and did not hold).
_LONG = {2: 0.0, 4: -20.0, 6: -25.0}
SHARED_BIAS = {0.12: _LONG, 0.2: _LONG, 0.3: _LONG}


def prototypes():
    """Returns {class: [(prob map, bias), ...]} of hand-made shape models."""
    out = {k: [] for k in range(10)}
    add = lambda c, parts, r, bias=0.0: out[c].append((render(parts, r), bias))
    for r in (0.34, 0.4, 0.46, 0.52, 0.58, 0.65, 0.72):
        for taper in (0.0, 0.12):
            for gap in (0.18, 0.32):
                add(1, trouser(r, taper, gap, 0.33), r)
    for kind, rs in (("aline", (0.5, 0.57, 0.64, 0.72, 0.8)), ("sheath", (0.34, 0.4, 0.46, 0.52)), ("shift", (0.46, 0.54, 0.62, 0.7))):
        for r in rs:
            for sleeve in (False, True):
                add(3, dress(r, kind, sleeve), r)
    for r in (0.8, 0.88, 0.96, 1.04, 1.12, 1.2):
        for bw in (0.28, 0.34):
            for sl in ((0.14, 0.32), (0.2, 0.42)):
                add(0, tee(r, bw, sl), r)
    for r in (0.68, 0.75):                    # fitted tees: sleeves stick out only a little past the body
        for bw in (0.36, 0.4):
            add(0, tee(r, bw, (0.12, 0.3)), r)
    for r in (0.48, 0.55, 0.62):
        for deep in (False, True):
            add(0, tank(r, deep), r)
    # long-sleeved tops: one shared family for pullover, coat and shirt
    for r in (0.7, 0.78, 0.86, 0.94, 1.02, 1.1, 1.18):
        for sd in (0.12, 0.2, 0.3):
            for kw in (dict(nd=0.0), dict(nd=0.08, gapline=True), dict(nd=0.0, se=0.88)):
                for c in (2, 4, 6):
                    add(c, longtop(r, 0.15, sd=sd, **kw), r, SHARED_BIAS[sd][c])
    for r in (0.78, 0.86, 0.94, 1.02):        # shirts: shirt-tail hem; short-sleeved shirts
        for sd in (0.2, 0.3):
            add(6, longtop(r, 0.15, 0.0, sd, hem="tail"), r, -10.0)
        add(6, tee(r, 0.33, (0.2, 0.42)), r, -25.0)
    for r in (1.8, 2.05, 2.35, 2.7, 3.1):
        for toe, tongue in ((0.45, 0.12), (0.55, 0.2)):
            add(7, sneaker(r, toe, tongue), r)
    for r in (0.75, 0.87, 1.0, 1.12, 1.25, 1.4, 1.55, 1.72):
        for front in (0.45, 0.56):
            for heel in (False, True):
                add(9, boot(r, front, heel), r)
    for r in (2.4, 3.0, 3.6, 4.3):
        for dens in (0.3, 0.5):
            add(5, sandal(r, False, dens), r)
    for r in (1.0, 1.2, 1.4, 1.65, 1.95, 2.3):
        for dens in (0.3, 0.5):
            add(5, sandal(r, True, dens), r)
    for r in (0.75, 0.88, 1.0, 1.15, 1.32, 1.5, 1.75):
        for body_top, handle in ((0.0, False), (0.25, True), (0.42, True)):
            for trap in (0.0, 0.1):
                add(8, bag(r, body_top, trap, handle), r)
    return out


# ---------------------------------------------------------------- network
import json, sys, os

LAYERS, WEIGHTS = [], {}


def layer(name, op, inputs, **kw):
    LAYERS.append(dict(name=name, op=op, inputs=list(inputs), **kw))
    return name


def conv(name, inp, W, b, pad):
    WEIGHTS[name + ".W"] = np.asarray(W, np.float32); WEIGHTS[name + ".b"] = np.asarray(b, np.float32)
    k = W.shape[2]
    return layer(name, "conv", [inp], out=int(W.shape[0]), kernel=[k, k], stride=1, padding=pad)


def dense(name, inp, W, b):
    WEIGHTS[name + ".W"] = np.asarray(W, np.float32); WEIGHTS[name + ".b"] = np.asarray(b, np.float32)
    return layer(name, "dense", [inp], out=int(W.shape[0]))


def relu(name, inp):
    return layer(name, "relu", [inp])


EPS = 0.06          # floor / ceiling of the hand-made foreground probabilities
LAM = 1.0           # scale of the shape log-likelihood


def ramp_rows(lo, hi):
    """Two pre-activations whose relus give ramp = (relu(f-lo) - relu(f-hi)) / (hi-lo)."""
    return np.array([-lo, -hi]), np.array([1.0, -1.0]) / (hi - lo)


def build():
    # 1. per-pixel channels: soft silhouette mask m = a0 - a1 (ramp from grey 0.02 to 0.06) and the intensity p
    W = np.zeros((3, 1, 1, 1)); b = np.zeros(3)
    W[0, 0, 0, 0], b[0] = 25.0, -0.5
    W[1, 0, 0, 0], b[1] = 25.0, -1.5
    W[2, 0, 0, 0] = 1.0
    t = relu("thr_r", conv("thr", "image", W, b, 0))

    # 2. shape log-likelihood of every hand-made prototype: sum over pixels of m*logit(P) + log(1-P)
    protos = prototypes()
    K = max(len(v) for v in protos.values())
    Wp = np.zeros((10 * K, 3 * 784)); bp = np.zeros(10 * K)
    for c in range(10):
        L = protos[c]
        for j in range(K):
            M, bias = L[j % len(L)]
            P = EPS + (1 - 2 * EPS) * M
            lo = np.log(P / (1 - P)).ravel()
            Wp[c * K + j, :784] = LAM * lo
            Wp[c * K + j, 784:1568] = -LAM * lo
            bp[c * K + j] = LAM * np.log(1 - P).sum() - np.log(len(L)) + bias
    pr = dense("proto", t, Wp, bp)
    pr = layer("proto_g", "reshape", [pr], shape=[1, 10, K])
    pr = layer("proto_max", "maxpool", [pr], size=[1, K], stride=[1, K])
    shape_score = layer("shape_score", "reshape", [pr], shape=[10])

    # 3. interior cues, 13x13 kernels centred at tap (6, 6)
    C = 6
    W = np.zeros((21, 3, 13, 13)); b = np.zeros(21)
    W[0, 2, C, C] = 1.0                                        # p
    W[1, 2, C, C - 2], W[1, 2, C, C + 2] = 1.0, -1.0           # relu(p(x-2) - p(x+2))
    W[2, 2, C, C - 2], W[2, 2, C, C + 2] = -1.0, 1.0           # relu(p(x+2) - p(x-2))
    W[3, 0, C - 1:C + 2, C - 1:C + 2], W[3, 1, C - 1:C + 2, C - 1:C + 2], b[3] = 1.0, -1.0, -8.0   # inside (3x3)
    for o, s in ((4, 1.0), (5, -1.0)):                          # +- Laplacian of p
        W[o, 2, C, C] = s
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            W[o, 2, C + dy, C + dx] = -0.25 * s
    W[6, 0, C, C], W[6, 1, C, C] = 1.0, -1.0                   # m
    for d in range(1, 7):     # item within 6 px to the left / right / above / below: count p and relu(count - 1)
        for o, (dy, dx) in zip((7, 9, 11, 13), ((0, -d), (0, d), (-d, 0), (d, 0))):
            W[o, 0, C + dy, C + dx] += 1; W[o, 1, C + dy, C + dx] -= 1
            W[o + 1, 0, C + dy, C + dx] += 1; W[o + 1, 1, C + dy, C + dx] -= 1
    b[[8, 10, 12, 14]] = -1.0
    W[15, 0, C - 1:C + 2, C - 1:C + 2], W[15, 1, C - 1:C + 2, C - 1:C + 2], b[15] = -1.0, 1.0, 1.0   # 3x3 all background
    for o, s, taps in ((16, 1, ((0, -1), (0, 1))), (17, -1, ((0, -1), (0, 1))), (18, 1, ((-1, 0), (1, 0))), (19, -1, ((-1, 0), (1, 0)))):
        W[o, 2, C, C] = s                                       # +- second difference across x (vertical lines) / y
        for dy, dx in taps:
            W[o, 2, C + dy, C + dx] = -0.5 * s
    W[20, 0, C - 2:C + 3, C - 2:C + 3], W[20, 1, C - 2:C + 3, C - 2:C + 3], b[20] = 1.0, -1.0, -24.0   # inside (5x5)
    f1 = relu("cue1_r", conv("cue1", t, W, b, C))

    TL, TT, TP = 0.03, 0.12, 0.08
    W = np.zeros((8, 21, 5, 5)); b = np.zeros(8)
    for ry in range(5):                                        # dark / bright vertical line, averaged over 5 rows
        W[0, 0, ry, 0] += 0.2; W[0, 0, ry, 2] -= 0.2; W[0, 1, ry, 2] -= 0.2
        W[1, 0, ry, 2] += 0.2; W[1, 0, ry, 0] -= 0.2; W[1, 2, ry, 2] -= 0.2
    b[0] = b[1] = -TL
    W[2, 4, 2, 2] = W[2, 5, 2, 2] = 1.0; W[2, 3, 2, 2] = 2.0; b[2] = -(TT + 2.0)   # |lap| inside the item
    # background with item on both sides (left and right / above and below) within 6 px: clamp + clamp - 1 - 2m
    W[3, 7, 2, 2], W[3, 8, 2, 2], W[3, 9, 2, 2], W[3, 10, 2, 2], W[3, 6, 2, 2], b[3] = 1, -1, 1, -1, -2, -1
    W[4, 11, 2, 2], W[4, 12, 2, 2], W[4, 13, 2, 2], W[4, 14, 2, 2], W[4, 6, 2, 2], b[4] = 1, -1, 1, -1, -2, -1
    W[5, 15, 2, 2] = 1.0                                       # pass the background core on
    W[6, 16, 2, 2] = W[6, 17, 2, 2] = 1.0; W[6, 20, 2, 2] = 2.0; b[6] = -(TP + 2.0)   # thin vertical lines inside
    W[7, 18, 2, 2] = W[7, 19, 2, 2] = 1.0; W[7, 20, 2, 2] = 2.0; b[7] = -(TP + 2.0)   # thin horizontal lines inside
    f2 = relu("cue2_r", conv("cue2", f1, W, b, 2))

    W = np.zeros((8, 8, 7, 7)); b = np.zeros(8)
    for o in (0, 1, 2, 6, 7):
        W[o, o, 3, 3] = 1.0                                    # line, texture and line-energy maps passed on
    W[3, 3, :, 3], b[3] = 1.0, -6.0                            # leg gap that runs for 7 rows
    W[4, 3, 3, 3] = W[4, 4, 3, 3] = W[4, 5, 3, 3] = 1.0; b[4] = -2.0   # wide hole enclosed on all four sides
    W[5, 3, 3, 3] = W[5, 4, 3, 3] = 1.0; b[5] = -1.0           # any background pocket enclosed on all four sides
    f3 = relu("cue3_r", conv("cue3", f2, W, b, 3))

    def chan(c, rows=slice(0, 28), cols=slice(0, 28), wt=1.0):
        m = np.zeros((8, 28, 28)); m[c, rows, cols] = wt; return m
    line = chan(0, slice(5, 26), slice(11, 17)) + chan(1, slice(5, 26), slice(11, 17), 0.5)
    cues = [(line, 0.5, 3.0),                         # 0 centre line (zip, opening, placket)
            (chan(2), 5.0, 20.0),                     # 1 general texture inside the item
            (chan(3, slice(10, 28), slice(11, 17)), 2.0, 8.0),   # 2 leg gap
            (chan(4), 3.0, 15.0),                     # 3 wide enclosed holes
            (chan(5), 6.0, 25.0),                     # 4 enclosed pockets of background (straps)
            (chan(6), 4.0, 16.0),                     # 5 vertical line energy inside
            (chan(7), 4.0, 16.0)]                     # 6 horizontal line energy inside
    n = len(cues)
    Wf = np.zeros((2 * n, 8 * 784)); bf = np.zeros(2 * n); R = np.zeros((n, 2 * n))
    for i, (mp, lo, hi) in enumerate(cues):
        Wf[2 * i] = Wf[2 * i + 1] = mp.ravel()
        bf[2 * i:2 * i + 2], R[i, 2 * i:2 * i + 2] = ramp_rows(lo, hi)
    f4 = relu("cue4_r", dense("cue4", f3, Wf, bf))
    # ramps r0..r6 and one extra unit relu(r5 - r6), so that min(r5, r6) = r5 - relu(r5 - r6): a check pattern
    # (plaid) needs lines in both directions, unlike stripes
    R2 = np.vstack([R, R[5] - R[6]])
    f5 = relu("cue5_r", dense("cue5", f4, R2, np.zeros(n + 1)))
    Bk = np.zeros((10, n + 1))
    Bk[4, 0] = 45.0                                  # coat: long centre line (zip or front opening)
    Bk[6, 0], Bk[6, 1] = 20.0, 15.0                  # shirt: button placket, patterned cloth
    Bk[6, 5], Bk[6, 7] = 45.0, -45.0                 # shirt: check pattern = min(vertical, horizontal line energy)
    Bk[1, 2] = 50.0                                  # trouser: background between the legs
    Bk[5, 3], Bk[5, 4] = 20.0, 40.0                  # sandal: open spaces between straps
    bonus = dense("bonus", f5, Bk, np.zeros(10))
    layer("scores", "add", [shape_score, bonus])


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    build()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump({"layers": LAYERS}, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **WEIGHTS)
