"""Hand-designed traffic-sign network.  Every number comes from geometry / formulas / my own
knowledge of German road signs -- no training data is read, no fitting is done.

Usage:  python3 build.py OUT_DIR
"""
import json, os, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

LAYERS = []
WEIGHTS = {}


def layer(name, op, inputs, **st):
    d = {"name": name, "op": op, "inputs": list(inputs)}
    d.update(st)
    LAYERS.append(d)
    return name


def conv(name, inp, w, b=None, stride=1, padding=0):
    w = np.asarray(w, np.float32)
    if b is None:
        b = np.zeros(w.shape[0], np.float32)
    b = np.asarray(b, np.float32)
    assert w.ndim == 4 and w.shape[2] == w.shape[3]
    WEIGHTS[name + ".W"] = w
    WEIGHTS[name + ".b"] = b
    return layer(name, "conv", [inp], out=int(w.shape[0]), kernel=[int(w.shape[2])] * 2,
                 stride=int(stride), padding=int(padding))


def dense(name, inp, w, b=None):
    w = np.asarray(w, np.float32)
    if b is None:
        b = np.zeros(w.shape[0], np.float32)
    WEIGHTS[name + ".W"] = w
    WEIGHTS[name + ".b"] = np.asarray(b, np.float32)
    return layer(name, "dense", [inp], out=int(w.shape[0]))


def relu(name, inp):
    return layer(name, "relu", [inp])


def pool(name, inp, size, stride=None, kind="max"):
    if stride is None:
        stride = size
    return layer(name, kind + "pool", [inp], size=list(size), stride=list(stride))


def reshape(name, inp, shape):
    return layer(name, "reshape", [inp], shape=list(shape))


def add(name, ins):
    return layer(name, "add", ins)


def norm(name, inp):
    return layer(name, "norm", [inp])


# ---------------------------------------------------------------- small geometry helpers
def grid(k, c=None):
    """centred coordinate grids for a k x k kernel"""
    if c is None:
        c = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k].astype(np.float64)
    return y - c, x - c


def blur(a, sigma):
    """separable gaussian, edge-clamped"""
    if sigma <= 0:
        return a.copy()
    r = max(1, int(3 * sigma))
    t = np.arange(-r, r + 1)
    g = np.exp(-0.5 * (t / sigma) ** 2)
    g /= g.sum()
    out = np.array(a, np.float64)
    for ax in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[ax] = (r, r)
        p = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, gi in enumerate(g):
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(i, i + out.shape[ax])
            acc += gi * p[tuple(sl)]
        out = acc
    return out


def unit(a):
    """scale a mask so its positive part sums to 1"""
    s = np.abs(a).sum()
    return a / s if s > 1e-9 else a


def posnorm(a):
    s = a[a > 0].sum()
    return a / s if s > 1e-9 else a


def stack_kernels(maps, nch, idx):
    """maps: list of (dict channel->2D mask).  returns (len(maps), nch, k, k)"""
    k = list(maps[0].values())[0].shape[0]
    W = np.zeros((len(maps), nch, k, k), np.float64)
    for i, m in enumerate(maps):
        for ch, v in m.items():
            W[i, ch] = v
    return W

# ================================================================ 1. colour front end
# norm() removes a global multiplicative lighting change exactly (mean and std both scale),
# which is what the very dark / washed-out frames need.
n0 = norm("n0", "image")

# colour-opponent channels, thresholds chosen by hand in units of per-channel std
CW = np.zeros((5, 3, 1, 1), np.float64)
CW[0, :, 0, 0] = [1.0, -0.5, -0.5]     # redness
CW[1, :, 0, 0] = [-0.5, -0.5, 1.0]     # blueness
CW[2, :, 0, 0] = [0.5, 0.5, -1.0]      # yellowness
CW[3, :, 0, 0] = [1 / 3, 1 / 3, 1 / 3]  # brightness
CW[4, :, 0, 0] = [-1 / 3, -1 / 3, -1 / 3]  # dark ink
CSC = 2.5   # opponent channels cancel the common luminance swing, so they need gain
CW[:3] *= CSC
CB = np.array([-0.35 * CSC, -0.35 * CSC, -0.35 * CSC, -0.10, -0.10])
conv("c1", n0, CW, CB)
p1 = relu("p1", "c1")                       # [5,32,32]  red blue yellow lum ink
h1 = pool("h1", p1, [2, 2], kind="avg")     # [5,16,16]  half resolution for outer shape

RED, BLUE, YEL, LUM, INK = 0, 1, 2, 3, 4


# ---------------------------------------------------------------- shape masks (half res)
def poly_masks(k, verts, tborder, tcore):
    """inside / border / core / outside masks of a convex polygon"""
    dy, dx = np.mgrid[0:k, 0:k].astype(np.float64)   # absolute pixel coords, verts are absolute
    V = np.array(verts, np.float64)
    crosses, dists = [], []
    n = len(V)
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        e = b - a
        cr = e[0] * (dy - a[1]) - e[1] * (dx - a[0])
        crosses.append(cr)
        dists.append(np.abs(cr) / np.hypot(*e))
    C = np.stack(crosses)
    inside = (C.min(0) >= 0) | (C.max(0) <= 0)
    dmin = np.stack(dists).min(0)
    border = inside & (dmin < tborder)
    core = inside & (dmin >= tcore)
    outside = ~inside & (dmin > 0.7)
    return inside.astype(float), border.astype(float), core.astype(float), outside.astype(float)


def disc_masks(k, R):
    dy, dx = grid(k)
    r = np.hypot(dy, dx)
    ring = ((r >= 0.70 * R) & (r <= 1.06 * R)).astype(float)
    core = (r <= 0.60 * R).astype(float)
    full = (r <= 1.02 * R).astype(float)
    out = (r >= 1.18 * R).astype(float)
    return ring, core, full, out


def tri_verts(k, R, up=True):
    c = (k - 1) / 2.0
    h = 1.02 * R          # half height
    w = 1.08 * R          # half base width
    if up:
        return [(c, c - h), (c - w, c + 0.86 * h), (c + w, c + 0.86 * h)]
    return [(c, c + h), (c - w, c - 0.86 * h), (c + w, c - 0.86 * h)]


KS = 15
shape_maps = []
shape_names = []


def add_shape(nm, m):
    shape_names.append(nm)
    shape_maps.append({c: blur(v, 0.55) for c, v in m.items()})


for R in (6.6, 5.2, 7.5):
    ring, core, full, out = disc_masks(KS, R)
    ring, core, out = posnorm(ring), posnorm(core), posnorm(out)
    add_shape("ring%.1f" % R, {RED: ring - 0.70 * core - 0.55 * out, LUM: 0.15 * core})
for R in (6.6, 5.2):
    ring, core, full, out = disc_masks(KS, R)
    full, out = posnorm(full), posnorm(out)
    inner = posnorm((np.hypot(*grid(KS)) <= 0.62 * R).astype(float))
    add_shape("disc%.1f" % R, {RED: inner - 0.55 * out})
for R in (6.6, 5.3):
    ins, bor, cor, out = poly_masks(KS, tri_verts(KS, R, True), 1.45, 2.3)
    bor, cor, out = posnorm(bor), posnorm(cor), posnorm(out)
    add_shape("tri%.1f" % R, {RED: bor - 0.60 * out, LUM: 0.30 * cor})
    ins, bor, cor, out = poly_masks(KS, tri_verts(KS, R, False), 1.45, 2.3)
    bor, cor, out = posnorm(bor), posnorm(cor), posnorm(out)
    add_shape("inv%.1f" % R, {RED: bor - 0.60 * out, LUM: 0.30 * cor})
for R in (6.8, 5.4):
    dy, dx = grid(KS)
    d = np.abs(dy) + np.abs(dx)
    cor = posnorm((d <= 0.68 * R).astype(float))
    bor = posnorm(((d > 0.72 * R) & (d <= 1.05 * R)).astype(float))
    out = posnorm((d > 1.25 * R).astype(float))
    add_shape("dia%.1f" % R, {YEL: cor - 0.6 * out, LUM: 0.35 * bor})
for R in (6.6, 5.2):
    ring, core, full, out = disc_masks(KS, R)
    core, full = posnorm(core), posnorm(full)
    add_shape("grey%.1f" % R, {LUM: core, RED: -2.0 * full, BLUE: -2.0 * full, YEL: -2.0 * full})
for R in (6.6, 5.2, 7.5):
    ring, core, full, out = disc_masks(KS, R)
    full, core, out = posnorm(full), posnorm(core), posnorm(out)
    add_shape("blu%.1f" % R, {BLUE: full - 0.55 * out, LUM: 0.20 * core})

NS = len(shape_maps)
conv("shp", h1, stack_kernels(shape_maps, 5, None), np.zeros(NS), stride=1, padding=3)
shpR = relu("shpR", "shp")
shpM = pool("shpM", shpR, [8, 8])           # [NS,1,1] best alignment near the centre
shpV = reshape("shpV", shpM, [NS])
SI = {n: i for i, n in enumerate(shape_names)}

# ================================================================ 2. stroke drawing helpers
# Everything below is drawn from my own knowledge of the German sign shapes (DIN pictograms),
# rendered analytically.  No image is ever read.


def _seg_dist(px, py, pts):
    """min distance from every grid point to a polyline given in pixel coords"""
    d = np.full(px.shape, 1e9)
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0, 1) if L2 > 1e-12 else np.zeros_like(px)
        d = np.minimum(d, np.hypot(px - (ax + t * vx), py - (ay + t * vy)))
    return d


def draw(prims, w, h, ss=3):
    """render primitives in a unit box (x right, y down) into an (h,w) antialiased mask.
    prim kinds: ('s', x0,y0,x1,y1, r)  capsule
                ('a', cx,cy,rx,ry, a0,a1, r) elliptical arc, degrees, y-down angles
                ('e', cx,cy,rx,ry, r) full ellipse ring
                ('f', cx,cy,rx,ry) filled ellipse
                ('p', pts, r) polyline through unit points
    r is a half-thickness as a fraction of the box WIDTH."""
    w, h = int(round(w)), int(round(h))
    W, H = w * ss, h * ss
    gy, gx = np.mgrid[0:H, 0:W]
    px = (gx + 0.5) / ss
    py = (gy + 0.5) / ss
    m = np.zeros((H, W))
    for p in prims:
        if p[0] == 's':
            _, x0, y0, x1, y1, r = p
            pts = [(x0 * w, y0 * h), (x1 * w, y1 * h)]
            d = _seg_dist(px, py, pts)
            m = np.maximum(m, np.clip(r * w + 0.5 - d, 0, 1))
        elif p[0] in ('a', 'e'):
            if p[0] == 'a':
                _, cx, cy, rx, ry, a0, a1, r = p
            else:
                _, cx, cy, rx, ry, r = p
                a0, a1 = 0.0, 360.0
            n = max(8, int(abs(a1 - a0) / 4))
            th = np.radians(np.linspace(a0, a1, n + 1))
            pts = [(cx * w + rx * w * np.cos(t), cy * h + ry * h * np.sin(t)) for t in th]
            if p[0] == 'e':
                pts.append(pts[0])
            d = _seg_dist(px, py, pts)
            m = np.maximum(m, np.clip(r * w + 0.5 - d, 0, 1))
        elif p[0] == 'f':
            _, cx, cy, rx, ry = p
            q = ((px - cx * w) / (rx * w)) ** 2 + ((py - cy * h) / (ry * h)) ** 2
            m = np.maximum(m, np.clip((1.0 - np.sqrt(q)) * min(rx * w, ry * h) + 0.5, 0, 1))
        elif p[0] == 't':
            _, x0, y0, x1, y1, x2, y2 = p
            V = [(x0 * w, y0 * h), (x1 * w, y1 * h), (x2 * w, y2 * h)]
            cs = []
            for i in range(3):
                ax, ay = V[i]
                bx, by = V[(i + 1) % 3]
                cs.append((bx - ax) * (py - ay) - (by - ay) * (px - ax))
            C = np.stack(cs)
            m = np.maximum(m, ((C.min(0) >= 0) | (C.max(0) <= 0)).astype(float))
        elif p[0] == 'p':
            _, pts, r = p
            d = _seg_dist(px, py, [(a * w, b * h) for a, b in pts])
            m = np.maximum(m, np.clip(r * w + 0.5 - d, 0, 1))
    # box downsample
    m = m.reshape(h, ss, w, ss).mean(axis=(1, 3))
    return m


DIGITS = {
    '0': [('e', 0.50, 0.50, 0.31, 0.40, 0.10)],
    '1': [('s', 0.60, 0.06, 0.60, 0.95, 0.105), ('s', 0.60, 0.07, 0.30, 0.26, 0.095)],
    '2': [('a', 0.50, 0.30, 0.31, 0.25, 165, 400, 0.10),
          ('s', 0.775, 0.44, 0.16, 0.90, 0.10),
          ('s', 0.13, 0.925, 0.88, 0.925, 0.10)],
    '3': [('a', 0.46, 0.27, 0.30, 0.235, 150, 450, 0.10),
          ('a', 0.46, 0.72, 0.32, 0.245, 270, 575, 0.10)],
    '5': [('s', 0.17, 0.075, 0.85, 0.075, 0.10),
          ('s', 0.19, 0.075, 0.19, 0.46, 0.10),
          ('s', 0.19, 0.45, 0.58, 0.43, 0.10),
          ('a', 0.50, 0.68, 0.32, 0.255, 280, 565, 0.10)],
    '6': [('a', 0.56, 0.32, 0.33, 0.27, 172, 325, 0.10),
          ('s', 0.235, 0.33, 0.235, 0.70, 0.10),
          ('e', 0.50, 0.705, 0.28, 0.235, 0.10)],
    '7': [('s', 0.13, 0.075, 0.87, 0.075, 0.10),
          ('s', 0.845, 0.10, 0.37, 0.95, 0.10)],
    '8': [('e', 0.50, 0.265, 0.265, 0.215, 0.095),
          ('e', 0.50, 0.725, 0.315, 0.235, 0.095)],
}


def digit_mask(ch, w, h):
    return draw(DIGITS[ch], w, h)


def place(dst, m, x0, y0):
    """paste mask m into dst with its top-left at (x0,y0) (floats, rounded)"""
    h, w = m.shape
    H, W = dst.shape
    xa, ya = int(round(x0)), int(round(y0))
    xs, ys = max(0, xa), max(0, ya)
    xe, ye = min(W, xa + w), min(H, ya + h)
    if xe > xs and ye > ys:
        dst[ys:ye, xs:xe] = np.maximum(dst[ys:ye, xs:xe], m[ys - ya:ye - ya, xs - xa:xe - xa])
    return dst


def dilate(a, r):
    """binary-ish dilation by a disc of radius r (pixels)"""
    if r <= 0:
        return a.copy()
    ri = int(np.ceil(r))
    H, W = a.shape
    out = np.zeros_like(a)
    for dy in range(-ri, ri + 1):
        for dx in range(-ri, ri + 1):
            if dx * dx + dy * dy > r * r + 0.25:
                continue
            out = np.maximum(out, np.roll(np.roll(a, dy, 0), dx, 1))
    return out


def ink_template(S, box, k):
    """zero-sum matched filter: dark strokes inside a bright box.
    S, box are (k,k) soft masks; returns kernel for the signed-luminance channel."""
    Sd = dilate(S, 1.0)
    B = np.clip(box - Sd, 0, None)
    s = S.sum()
    b = B.sum()
    if s < 1e-6 or b < 1e-6:
        return np.zeros_like(S)
    return -(S / s) + (B / b)

# ================================================================ 3. speed-limit digit blocks
# signed colour channels at full resolution (no relu: matched filters need the sign)
SW = np.zeros((4, 3, 1, 1), np.float64)
SW[0, :, 0, 0] = [1 / 3, 1 / 3, 1 / 3]      # L  luminance
SW[1, :, 0, 0] = [1.0, -0.5, -0.5]          # Rc redness
SW[2, :, 0, 0] = [-0.5, -0.5, 1.0]          # Bc blueness
SW[3, :, 0, 0] = [-0.5, 1.0, -0.5]          # Gc greenness
conv("sg", n0, SW, np.zeros(4))
LCH, RCH, BCH, GCH = 0, 1, 2, 3
# single-channel views so the big matched-filter banks stay cheap
conv("vL", "sg", np.eye(4)[0:1].reshape(1, 4, 1, 1))
conv("vLB", "sg", np.eye(4)[0:2].reshape(2, 4, 1, 1))     # L, Rc
conv("vLBb", "sg", np.eye(4)[[0, 2]].reshape(2, 4, 1, 1))  # L, Bc
conv("vLG", "sg", np.eye(4)[[0, 3]].reshape(2, 4, 1, 1))   # L, Gc

KD = 21
BLOCKS = ["20", "30", "50", "60", "70", "80", "100", "120"]
DSCALE = [(8.5, 0.56, 0.14), (10.3, 0.56, 0.14), (12.1, 0.56, 0.14), (13.9, 0.56, 0.14),
          (7.5, 0.50, 0.12), (9.0, 0.50, 0.12), (10.5, 0.50, 0.12)]
dmaps, dnames = [], []
c = (KD - 1) / 2.0
for si, (hd, wr, gr) in enumerate(DSCALE):
    for blk in BLOCKS:
        n = len(blk)
        if (n == 3) != (si >= 4):
            continue
        wd, gap = wr * hd, gr * hd
        tot = n * wd + (n - 1) * gap
        x0, y0 = c + 0.5 - tot / 2, c + 0.5 - hd / 2
        S = np.zeros((KD, KD))
        for i, ch in enumerate(blk):
            place(S, digit_mask(ch, wd, hd), x0 + i * (wd + gap), y0)
        box = np.zeros((KD, KD))
        yy, xx = np.mgrid[0:KD, 0:KD]
        box[(xx >= x0 - 1.3) & (xx <= x0 + tot + 0.3) & (yy >= y0 - 1.3) & (yy <= y0 + hd + 0.3)] = 1.0
        T = blur(ink_template(S, box, KD), 0.5)
        dmaps.append({0: T, 1: -1.2 * posnorm(box)})
        dnames.append("%s_%d" % (blk, si % 4))
ND = len(dmaps)
conv("dig", "vLB", stack_kernels(dmaps, 2, None), np.full(ND, -0.45), stride=1, padding=0)
digR = relu("digR", "dig")
digM = pool("digM", digR, [12, 12])
digV = reshape("digV", digM, [ND])
DI = {n: i for i, n in enumerate(dnames)}

# ================================================================ 4. blue mandatory-disc arrows
def mirror(prims):
    out = []
    for p in prims:
        if p[0] == 's':
            out.append(('s', 1 - p[1], p[2], 1 - p[3], p[4], p[5]))
        elif p[0] == 't':
            out.append(('t', 1 - p[1], p[2], 1 - p[3], p[4], 1 - p[5], p[6]))
        elif p[0] in ('a',):
            out.append(('a', 1 - p[1], p[2], p[3], p[4], 180 - p[6], 180 - p[5], p[7]))
        elif p[0] in ('e', 'f'):
            out.append((p[0], 1 - p[1], p[2], p[3], p[4], *p[5:]))
        else:
            out.append(p)
    return out


A_UP = [('s', 0.50, 0.93, 0.50, 0.30, 0.075),
        ('t', 0.50, 0.05, 0.235, 0.35, 0.765, 0.35)]
A_RIGHT = [('s', 0.355, 0.93, 0.355, 0.47, 0.075),
           ('a', 0.595, 0.47, 0.24, 0.21, 180, 270, 0.075),
           ('s', 0.595, 0.26, 0.70, 0.26, 0.075),
           ('t', 0.955, 0.26, 0.685, 0.075, 0.685, 0.445)]
A_UPRIGHT = [('s', 0.355, 0.93, 0.355, 0.32, 0.072),
             ('t', 0.355, 0.055, 0.115, 0.345, 0.595, 0.345),
             ('s', 0.355, 0.585, 0.70, 0.585, 0.070),
             ('t', 0.955, 0.585, 0.685, 0.40, 0.685, 0.77)]
A_KEEP = [('s', 0.235, 0.215, 0.615, 0.595, 0.082),
          ('t', 0.855, 0.835, 0.735, 0.435, 0.455, 0.715)]
_round = []
for a0 in (95, 215, 335):
    _round.append(('a', 0.50, 0.50, 0.30, 0.30, a0, a0 + 78, 0.068))
    th = np.radians(a0 + 78)
    hx, hy = 0.50 + 0.30 * np.cos(th), 0.50 + 0.30 * np.sin(th)
    tx, ty = -np.sin(th), np.cos(th)          # tangent (direction of travel)
    nx, ny = np.cos(th), np.sin(th)
    _round.append(('t', hx + 0.17 * tx, hy + 0.17 * ty,
                   hx - 0.02 * tx + 0.13 * nx, hy - 0.02 * ty + 0.13 * ny,
                   hx - 0.02 * tx - 0.13 * nx, hy - 0.02 * ty - 0.13 * ny))
A_ROUND = _round

ARROWS = [("r", A_RIGHT), ("l", mirror(A_RIGHT)), ("u", A_UP),
          ("ur", A_UPRIGHT), ("ul", mirror(A_UPRIGHT)),
          ("kr", A_KEEP), ("kl", mirror(A_KEEP)), ("rb", A_ROUND)]

KA = 21
amaps, anames = [], []
for Rd in (13.0, 10.5, 14.5):
    A = 1.30 * Rd
    ca = (KA - 1) / 2.0
    yy, xx = np.mgrid[0:KA, 0:KA]
    rr = np.hypot(yy - ca, xx - ca)
    disc = (rr <= 0.97 * Rd).astype(float)
    for nm, prims in ARROWS:
        S = np.zeros((KA, KA))
        place(S, draw(prims, A, A), ca + 0.5 - A / 2, ca + 0.5 - A / 2)
        Sd = dilate(S, 1.0)
        B = np.clip(disc - Sd, 0, None)
        s, b = S.sum(), B.sum()
        TL = blur(S / s - 0.85 * B / b, 0.55)
        TB = blur(0.45 * disc / disc.sum(), 0.55)
        amaps.append({0: TL, 1: TB})
        anames.append("%s_%.0f" % (nm, Rd))
NA = len(amaps)
conv("arr", "vLBb", stack_kernels(amaps, 2, None), np.full(NA, -0.95), stride=2, padding=0)
arrR = relu("arrR", "arr")
arrM = pool("arrM", arrR, [6, 6])
arrV = reshape("arrV", arrM, [NA])
AI = {n: i for i, n in enumerate(anames)}

# ================================================================ 5. warning-triangle pictograms
PRI = [('s', 0.50, 0.03, 0.50, 0.97, 0.135), ('s', 0.06, 0.38, 0.94, 0.38, 0.075)]
EXC = [('s', 0.50, 0.04, 0.50, 0.62, 0.19), ('f', 0.50, 0.88, 0.21, 0.095)]
BL = [('s', 0.64, 0.96, 0.64, 0.44, 0.115),
      ('a', 0.40, 0.44, 0.24, 0.26, 270, 360, 0.115),
      ('s', 0.40, 0.18, 0.10, 0.18, 0.115)]
DB = [('p', [(0.64, 0.04), (0.30, 0.40), (0.68, 0.58), (0.34, 0.96)], 0.115)]
BMP = [('f', 0.27, 0.86, 0.185, 0.62), ('f', 0.73, 0.86, 0.185, 0.62)]
SLP = [('f', 0.50, 0.24, 0.29, 0.115), ('f', 0.47, 0.14, 0.16, 0.085),
       ('p', [(0.20, 0.60), (0.32, 0.70), (0.20, 0.80), (0.32, 0.96)], 0.055),
       ('p', [(0.68, 0.60), (0.80, 0.70), (0.68, 0.80), (0.80, 0.96)], 0.055)]
NR = [('s', 0.22, 0.04, 0.22, 0.96, 0.10),
      ('p', [(0.78, 0.96), (0.78, 0.50), (0.56, 0.04)], 0.10)]
RW = [('f', 0.32, 0.12, 0.075, 0.075), ('s', 0.32, 0.22, 0.30, 0.55, 0.062),
      ('s', 0.30, 0.55, 0.19, 0.92, 0.05), ('s', 0.30, 0.55, 0.43, 0.92, 0.05),
      ('s', 0.18, 0.34, 0.56, 0.64, 0.045), ('t', 0.60, 0.96, 0.97, 0.96, 0.80, 0.60)]
PED = [('f', 0.52, 0.10, 0.10, 0.08), ('s', 0.52, 0.20, 0.45, 0.52, 0.07),
       ('s', 0.45, 0.52, 0.20, 0.94, 0.055), ('s', 0.45, 0.52, 0.74, 0.90, 0.055),
       ('s', 0.28, 0.30, 0.74, 0.42, 0.05)]
CH = [('f', 0.66, 0.12, 0.075, 0.085), ('s', 0.66, 0.22, 0.60, 0.55, 0.055),
      ('s', 0.60, 0.55, 0.44, 0.96, 0.045), ('s', 0.60, 0.55, 0.83, 0.92, 0.045),
      ('s', 0.47, 0.32, 0.86, 0.40, 0.04),
      ('f', 0.24, 0.28, 0.06, 0.07), ('s', 0.24, 0.36, 0.20, 0.62, 0.045),
      ('s', 0.20, 0.62, 0.09, 0.96, 0.04), ('s', 0.20, 0.62, 0.35, 0.92, 0.04),
      ('s', 0.09, 0.44, 0.39, 0.50, 0.035)]
BIC = [('e', 0.22, 0.72, 0.185, 0.30, 0.042), ('e', 0.78, 0.72, 0.185, 0.30, 0.042),
       ('p', [(0.22, 0.72), (0.45, 0.36), (0.62, 0.72), (0.78, 0.72)], 0.033),
       ('s', 0.45, 0.36, 0.70, 0.36, 0.033), ('s', 0.62, 0.72, 0.56, 0.28, 0.033),
       ('s', 0.66, 0.18, 0.82, 0.30, 0.033)]
SNOW = []
for _a in range(6):
    _t = np.radians(_a * 60 + 90)
    _cx, _cy = 0.5 + 0.44 * np.cos(_t), 0.5 + 0.44 * np.sin(_t)
    SNOW.append(('s', 0.5, 0.5, _cx, _cy, 0.062))
    _mx, _my = 0.5 + 0.28 * np.cos(_t), 0.5 + 0.28 * np.sin(_t)
    for _s in (-1, 1):
        _u = _t + _s * np.radians(55)
        SNOW.append(('s', _mx, _my, _mx + 0.16 * np.cos(_u), _my + 0.16 * np.sin(_u), 0.045))
DEER = [('s', 0.30, 0.36, 0.74, 0.62, 0.098), ('s', 0.30, 0.36, 0.19, 0.14, 0.06),
        ('f', 0.165, 0.115, 0.07, 0.055),
        ('s', 0.16, 0.07, 0.09, 0.00, 0.028), ('s', 0.19, 0.07, 0.27, 0.00, 0.028),
        ('s', 0.35, 0.46, 0.21, 0.86, 0.045), ('s', 0.41, 0.49, 0.39, 0.92, 0.045),
        ('s', 0.72, 0.60, 0.86, 0.94, 0.045), ('s', 0.70, 0.62, 0.61, 0.94, 0.045),
        ('s', 0.76, 0.56, 0.88, 0.46, 0.038)]

# name, prims, box width, box height, vertical offset of box centre from sign centre
TLP = [('f', 0.50, 0.15, 0.20, 0.11), ('f', 0.50, 0.50, 0.20, 0.11), ('f', 0.50, 0.85, 0.20, 0.11)]
# (name, strokes, stroke-box w, h, y offset, extra box margins L,R,U,D)
# wide margins make a template reject signs whose extra ink falls just outside its own symbol
PICTO = [("pri", PRI, 11, 13, 2.5, (1.2, 1.2, 1.2, 1.2)),
         ("exc", EXC, 6, 14, 2.0, (3.0, 3.0, 1.2, 1.2)),
         ("bl", BL, 10, 11, 2.5, (1.6, 1.6, 1.2, 1.2)),
         ("br", mirror(BL), 10, 11, 2.5, (1.6, 1.6, 1.2, 1.2)),
         ("db", DB, 10, 13, 2.5, (1.6, 1.6, 1.2, 1.2)),
         ("bmp", BMP, 13, 7, 5.5, (1.2, 1.2, 7.0, 1.2)),
         ("slp", SLP, 13, 13, 2.5, (1.2, 1.2, 1.2, 1.2)),
         ("nr", NR, 9, 13, 2.5, (1.8, 1.8, 1.2, 1.2)),
         ("rw", RW, 14, 13, 2.5, (1.2, 1.2, 1.2, 1.2)),
         ("ped", PED, 10, 13, 2.5, (2.6, 2.6, 1.2, 1.2)),
         ("ch", CH, 15, 12, 3.0, (1.2, 1.2, 1.2, 1.2)),
         ("bic", BIC, 15, 9, 4.0, (1.2, 1.2, 3.0, 1.2)),
         ("snow", SNOW, 11, 11, 3.0, (1.6, 1.6, 1.6, 1.6)),
         ("deer", DEER, 14, 11, 3.5, (1.2, 1.2, 1.2, 1.2)),
         ("tl", TLP, 7, 13, 2.0, (2.6, 2.6, 1.2, 1.2))]

KT = 19
tmaps, tnames = [], []
ct = (KT - 1) / 2.0
yy, xx = np.mgrid[0:KT, 0:KT]
for sc in (1.0, 0.82):
    for nm, prims, pw, ph, oy, bp in PICTO:
        pw, ph = pw * sc, ph * sc
        S = np.zeros((KT, KT))
        place(S, draw(prims, pw, ph), ct + 0.5 - pw / 2, ct + 0.5 + oy * sc - ph / 2)
        x0, y0 = ct + 0.5 - pw / 2, ct + 0.5 + oy * sc - ph / 2
        bl, br, bu, bd = [v * sc for v in bp]
        box = ((xx >= x0 - bl) & (xx <= x0 + pw + br - 1.0) &
               (yy >= y0 - bu) & (yy <= y0 + ph + bd - 1.0)).astype(float)
        tmaps.append({0: blur(ink_template(S, box, KT), 0.5)})
        tnames.append("%s_%.2f" % (nm, sc))
NT = len(tmaps)
conv("tri", "vL", stack_kernels(tmaps, 1, None), np.full(NT, -0.45), stride=1, padding=0)
triR = relu("triR", "tri")
triM = pool("triM", triR, [14, 14])
triV = reshape("triV", triM, [NT])
TI = {n: i for i, n in enumerate(tnames)}

# ---- traffic lights: the only warning sign with coloured dots inside (red / yellow / green)
SW5 = np.zeros((5, 3, 1, 1), np.float64)
SW5[0] = SW[0]; SW5[1] = SW[1]; SW5[2] = SW[2]; SW5[3] = SW[3]
SW5[4, :, 0, 0] = [0.5, 0.5, -1.0]
conv("sg5", n0, SW5, np.zeros(5))
conv("vLGY", "sg5", np.eye(5)[[0, 1, 3, 4]].reshape(4, 5, 1, 1))   # L, Rc, Gc, Yc
gmaps, gnames = [], []
for sc in (1.0, 0.82):
    pw, ph, oy = 7 * sc, 13 * sc, 2.0 * sc
    x0, y0 = ct + 0.5 - pw / 2, ct + 0.5 + oy - ph / 2
    box = ((xx >= x0 - 1.0) & (xx <= x0 + pw) & (yy >= y0 - 1.0) & (yy <= y0 + ph)).astype(float)
    dots, surr = [], []
    for fy in (0.15, 0.50, 0.85):
        d = np.zeros((KT, KT))
        dd = 3.4 * sc
        cxp, cyp = ct + 0.5, y0 + fy * ph
        place(d, draw([('f', 0.5, 0.5, 0.45, 0.45)], dd, dd), cxp - dd / 2, cyp - dd / 2)
        dots.append(posnorm(blur(d, 0.4)))
        rr2 = np.hypot(yy - cyp + 0.5, xx - cxp + 0.5)
        surr.append(posnorm(blur(((rr2 >= 2.3 * sc) & (rr2 <= 4.2 * sc)).astype(float), 0.4)))
    allsur = sum(surr) / 3.0
    for j, (cn, ch) in enumerate((("tlR", 1), ("tlY", 3), ("tlG", 2))):
        gmaps.append({0: -0.30 * dots[j], ch: dots[j] - 1.0 * allsur})
        gnames.append("%s_%.2f" % (cn, sc))
NG = len(gmaps)
conv("tlc", "vLGY", stack_kernels(gmaps, 4, None), np.full(NG, -0.25), stride=2, padding=0)
tlcR = relu("tlcR", "tlc")
tlcM = pool("tlcM", tlcR, [7, 7])
tlcV = reshape("tlcV", tlcM, [NG])
GI = {n: i for i, n in enumerate(gnames)}

# ================================================================ 6. interior region contrasts
# Used for: empty disc / truck / two cars / truck+car, stop vs no-entry, and the grey
# "end of" discs (stripes only vs digits vs cars).
KR = 21
cr = (KR - 1) / 2.0
ry, rx = np.mgrid[0:KR, 0:KR]
ry, rx = ry - cr, rx - cr
rmaps, rnames = [], []


def radd(nm, ch, A, B, wa=1.0, wb=1.0):
    a, b = posnorm(blur(A.astype(float), 0.5)), posnorm(blur(B.astype(float), 0.5))
    rmaps.append({ch: wa * a - wb * b})
    rnames.append(nm)


for R in (13.0, 10.5):
    s = "%.0f" % R
    rr = np.hypot(ry, rx)
    core = rr <= 0.58 * R
    ref = (rr > 0.62 * R) & (rr <= 0.74 * R)
    L = core & (rx < -0.08 * R)
    Rt = core & (rx > 0.08 * R)
    U = core & (ry < -0.10 * R)
    D = core & (ry >= -0.10 * R)
    regs = [("all", core), ("l", L), ("r", Rt), ("lu", L & U), ("ll", L & D),
            ("ru", Rt & U), ("rl", Rt & D),
            ("luu", L & (ry < -0.28 * R)), ("ruu", Rt & (ry < -0.28 * R)),
            ("mid", core & (np.abs(ry) < 0.20 * R)),
            ("top", core & (ry < -0.26 * R)), ("bot", core & (ry > 0.26 * R))]
    for nm, m in regs:
        if m.sum() < 3:
            m = core
        radd("ink_%s_%s" % (nm, s), 0, ref, m)
    for nm, m in [("l", L), ("lu", L & U), ("ll", L & D), ("r", Rt), ("ru", Rt & U),
                  ("luu", L & (ry < -0.28 * R))]:
        if m.sum() < 3:
            m = core
        radd("red_%s_%s" % (nm, s), 1, m, ref)
    # stop vs no-entry: a white horizontal bar has bright centre rows and dark rows off it
    band = (np.abs(ry) <= 0.13 * R) & (np.abs(rx) <= 0.62 * R)
    off = (np.abs(ry) > 0.24 * R) & (np.abs(ry) <= 0.46 * R) & (np.abs(rx) <= 0.62 * R)
    radd("bar_%s" % s, 0, band, off)
    radd("barw_%s" % s, 0, band, ref)
NRG = len(rmaps)
conv("reg", "vLB", stack_kernels(rmaps, 2, None), np.zeros(NRG), stride=2, padding=0)
# keep the region contrasts jointly aligned: average the four central conv positions
SEL = np.zeros((NRG, NRG * 36))
for i in range(NRG):
    for (py, px) in ((2, 2), (2, 3), (3, 2), (3, 3)):
        SEL[i, i * 36 + py * 6 + px] = 0.25
dense("regV", "reg", SEL)
RI = {n: i for i, n in enumerate(rnames)}

# ================================================================ 6b. vertical-edge energy
# STOP lettering has many vertical edges; a no-entry bar has almost none.
EG = np.zeros((2, 1, 3, 3))
EG[0, 0, 1] = [-1.0, 0.0, 1.0]
EG[1, 0, 1] = [1.0, 0.0, -1.0]
conv("egx", "vL", EG, np.zeros(2), stride=1, padding=1)
egr = relu("egr", "egx")
conv("ega", egr, np.ones((1, 2, 1, 1)))
emaps, enames = [], []
for R in (13.0, 10.5):
    s = "%.0f" % R
    rr = np.hypot(ry, rx)
    emaps.append({0: posnorm((rr <= 0.60 * R).astype(float))})
    enames.append("core_" + s)
    emaps.append({0: posnorm(((np.abs(ry) <= 0.22 * R) & (np.abs(rx) <= 0.66 * R)).astype(float))})
    enames.append("band_" + s)
NE = len(emaps)
conv("egc", "ega", stack_kernels(emaps, 1, None), np.zeros(NE), stride=2, padding=0)
ESEL = np.zeros((NE, NE * 36))
for i in range(NE):
    for (py, px) in ((2, 2), (2, 3), (3, 2), (3, 3)):
        ESEL[i, i * 36 + py * 6 + px] = 0.25
dense("egV", "egc", ESEL)
EI = {n: i for i, n in enumerate(enames)}

# ================================================================ 7. exact max over scale groups
# max(a,b) = b + relu(a-b), built from dense / relu / add.
def groupmax(prefix, vec, n, groups):
    cur = vec
    depth = max(len(g) for g in groups)
    for step in range(depth - 1):
        Wa = np.zeros((n, n))
        Wb = np.eye(n)
        used = False
        for g in groups:
            if len(g) > step + 1:
                i0, i1 = g[0], g[step + 1]
                Wa[i0] = 0.0
                Wa[i0, i0], Wa[i0, i1] = 1.0, -1.0
                Wb[i0] = 0.0
                Wb[i0, i1] = 1.0
                used = True
        if not used:
            break
        a = relu(prefix + "a%d" % step, dense(prefix + "d%d" % step, cur, Wa))
        b = dense(prefix + "p%d" % step, cur, Wb)
        cur = add(prefix + "m%d" % step, [a, b])
    return cur


def groups_of(names, idx):
    """group template indices by their base name (scale suffix after the last '_')"""
    d = {}
    for n, i in idx.items():
        d.setdefault(n.rsplit("_", 1)[0], []).append(i)
    return [sorted(v) for v in d.values()]


shpG = [[SI["ring6.6"], SI["ring5.2"], SI["ring7.5"]],
        [SI["disc6.6"], SI["disc5.2"]],
        [SI["tri6.6"], SI["tri5.3"]],
        [SI["inv6.6"], SI["inv5.3"]],
        [SI["dia6.8"], SI["dia5.4"]],
        [SI["grey6.6"], SI["grey5.2"]],
        [SI["blu6.6"], SI["blu5.2"], SI["blu7.5"]]]
shpX = groupmax("mxS", shpV, NS, shpG)
digX = groupmax("mxD", digV, ND, groups_of(dnames, DI))
arrX = groupmax("mxA", arrV, NA, groups_of(anames, AI))
triX = groupmax("mxT", triV, NT, groups_of(tnames, TI))
tlcX = groupmax("mxG", tlcV, NG, groups_of(gnames, GI))

CAT = {c: g[0] for c, g in zip(("ring", "disc", "tri", "inv", "dia", "grey", "blu"), shpG)}

# scaled category strengths, then "best competing category" for each category.
# A class only wins if its own outer shape beats the alternatives -- this stops a
# dark core (blue disc, pictogram) from being read as a truck / car sign.
CATNAME = ["ring", "disc", "tri", "inv", "dia", "grey", "blu"]
CATSCALE = {"ring": 1.0, "disc": 0.85, "tri": 0.9, "inv": 0.95, "dia": 1.0, "grey": 1.5, "blu": 1.8}
CS = np.zeros((7, NS))
for i, cn in enumerate(CATNAME):
    CS[i, CAT[cn]] = CATSCALE[cn]
catV = dense("catV", shpX, CS)
OTH = np.zeros((42, 7))
for c in range(7):
    j = 0
    for d in range(7):
        if d != c:
            OTH[c * 6 + j, d] = 1.0
            j += 1
othM = groupmax("mxO", dense("othV", catV, OTH), 42, [[c * 6 + j for j in range(6)] for c in range(7)])
PICK = np.zeros((7, 42))
for c in range(7):
    PICK[c, c * 6] = 1.0
othX = dense("othX", othM, PICK)

# "some speed-limit number matched well" -- used to explain away the car / truck / empty discs
drep = sorted(g[0] for g in groups_of(dnames, DI))
DR = np.zeros((len(drep), ND))
for i, j in enumerate(drep):
    DR[i, j] = 1.0
digAny = dense("digAny", groupmax("mxA2", dense("digRep", digX, DR), len(drep),
                                 [list(range(len(drep)))]), np.eye(len(drep))[0:1])

# ================================================================ 8. hand-written rule table
# score(class) = gate * category + evidence for that class's symbol.
# All coefficients below are chosen by hand from the geometry of the signs.
G = 3.0
WD = 2.8
WA = 2.2
WT = 2.6
K = 43

CATOF = {}
for k in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16):
    CATOF[k] = "ring"
for k in (14, 17):
    CATOF[k] = "disc"
for k in (11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
    CATOF[k] = "tri"
CATOF[13] = "inv"
CATOF[12] = "dia"
for k in (6, 32, 41, 42):
    CATOF[k] = "grey"
for k in range(33, 41):
    CATOF[k] = "blu"

HINH = 0.7
Wc = np.zeros((K, 7))
Wo = np.zeros((K, 7))
for k in range(K):
    c = CATNAME.index(CATOF[k])
    Wc[k, c] = G
    Wo[k, c] = -G * HINH
dense("dShape", catV, Wc)
dense("dShape2", othX, Wo)

# ---- speed limits
BLK = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}
Wd = np.zeros((K, ND))
for k, b in BLK.items():
    Wd[k, DI[b + "_0"]] = WD
Wd[6, DI["80_0"]] = 0.9 * WD          # end-of-80 carries the same grey digits
dense("dDigit", digX, Wd)

# ---- blue mandatory discs
ARR = {33: "r", 34: "l", 35: "u", 36: "ur", 37: "ul", 38: "kr", 39: "kl", 40: "rb"}
Wa = np.zeros((K, NA))
for k, b in ARR.items():
    Wa[k, AI[b + "_13"]] = WA
dense("dArrow", arrX, Wa)

# ---- warning triangles
PIC = {11: "pri", 18: "exc", 19: "bl", 20: "br", 21: "db", 22: "bmp", 23: "slp",
       24: "nr", 25: "rw", 26: "tl", 27: "ped", 28: "ch", 29: "bic", 30: "snow", 31: "deer"}
Wt = np.zeros((K, NT))
for k, b in PIC.items():
    Wt[k, TI[b + "_1.00"]] = WT
dense("dTri", triX, Wt)

Wg = np.zeros((K, NG))
Wg[26, GI["tlR_1.00"]] = 1.2
Wg[26, GI["tlG_1.00"]] = 1.5
Wg[26, GI["tlY_1.00"]] = 0.8
for k in PIC:
    if k != 26:
        Wg[k, GI["tlG_1.00"]] = -0.6
        Wg[k, GI["tlR_1.00"]] = -0.3
dense("dTlc", tlcX, Wg)

# ---- interior regions
Wr = np.zeros((K, NRG))


def rw(k, nm, w):
    for s in ("13", "10"):
        j = RI.get(nm + "_" + s)
        if j is not None:
            Wr[k, j] += w


def midx(k, w):
    rw(k, "ink_mid", w)
    rw(k, "ink_top", -0.5 * w)
    rw(k, "ink_bot", -0.5 * w)


rw(15, "ink_all", -2.5)
rw(16, "ink_all", 1.2); rw(16, "red_l", -4.0); midx(16, 1.0)
rw(9, "red_l", 2.5); rw(9, "ink_r", 2.0); rw(9, "ink_all", 1.0); rw(9, "red_luu", -2.0)
rw(10, "red_l", 2.5); rw(10, "ink_r", 2.0); rw(10, "ink_all", 1.0); rw(10, "red_luu", 2.5)
rw(17, "bar", 2.5)
rw(14, "bar", -2.5)
midx(6, 1.0); rw(6, "ink_luu", -1.5)
midx(32, -1.5)
rw(41, "ink_l", 2.0); rw(41, "ink_r", 2.0); rw(41, "ink_all", 1.0); rw(41, "ink_luu", -2.0)
rw(42, "ink_l", 2.0); rw(42, "ink_r", 2.0); rw(42, "ink_all", 1.0); rw(42, "ink_luu", 2.5)
dense("dReg", "regV", Wr)

Wa2 = np.zeros((K, 1))
for k in (9, 10, 15, 16):
    Wa2[k, 0] = -1.6
for k in (32, 41, 42):
    Wa2[k, 0] = -1.2
dense("dDigAny", digAny, Wa2)

We = np.zeros((K, NE))
for s in ("13", "10"):
    We[14, EI["band_" + s]] = 1.2
    We[17, EI["band_" + s]] = -1.0
    We[15, EI["core_" + s]] = -1.5
    We[16, EI["core_" + s]] = 1.0
    We[32, EI["core_" + s]] = -1.5
dense("dEdge", "egV", We)

add("scores", ["dShape", "dShape2", "dDigit", "dArrow", "dTri", "dTlc", "dReg", "dEdge", "dDigAny"])

# ---------------------------------------------------------------- write
net = {"layers": LAYERS}
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump(net, f, indent=1)
np.savez(os.path.join(OUT, "weights.npz"), **{k: v.astype(np.float32) for k, v in WEIGHTS.items()})
print("layers", len(LAYERS), "weights", sum(v.size for v in WEIGHTS.values()))
