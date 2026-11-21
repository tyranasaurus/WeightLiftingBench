"""build.py: writes network.json and weights.npz for the sign-language letter classifier.

Everything is made from formulas and from hand-typed shape models of a right hand making each static ASL letter
(palm box, finger capsules at typed angles, thumb, forearm).  No data is read.

  features : per-image normalisation -> 16 steered Gaussian-derivative filters -> rectified, sharpened into
             8 unsigned edge orientations -> clipped -> Gaussian pooling to a 12 x 12 grid -> global normalisation
  templates: every (letter, pose, size, tilt, width, position) drawing of the hand model is rendered, passed through
             the same feature formulas and normalised by itself alone; it becomes one row of a dense matching layer
  readout  : max over the templates of each letter.
"""
import json, os, sys
import numpy as np

# ----------------------------------------------------------------------------------------------------------------
# feature formulas (shared by the network weights and by the template maker)
# ----------------------------------------------------------------------------------------------------------------
FRAME = 28            # image side
CANVAS = 60           # a hand is drawn once on this larger canvas with its palm centre in the middle, then the frame is
                      # cut out at every allowed position
NDIR = 16             # signed gradient directions
NORI = 8              # unsigned edge orientations
GK = 5                # gradient kernel side
GSIG = 0.9            # gradient Gaussian sigma (pixels)
DEAD = 0.03           # dead zone for edge energy (in units of image standard deviations per pixel)
CLIP = 0.25           # edge energy saturates at about the response to a step of half an image standard deviation, so the
                      # faint outline of a lit finger against a wall of similar brightness counts nearly as much as a deep shadow
PK = 8                # pooling kernel side
PSIG = 1.75           # pooling Gaussian sigma (pixels)
PSTR = 2              # pooling stride
PPAD = 3              # pooling padding
CELLS = (FRAME - GK + 1 + 2 * PPAD - PK) // PSTR + 1            # 12
NCLASS = 24
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"


def gradient_filters():
    """16 first-derivative-of-Gaussian filters, direction k * 22.5 degrees; unit response to a unit ramp."""
    r = np.arange(GK) - (GK - 1) / 2.0
    yy, xx = np.meshgrid(r, r, indexing="ij")
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * GSIG ** 2))
    dx = xx * g
    dx = dx / np.sum(dx * xx)
    dy = yy * g
    dy = dy / np.sum(dy * yy)
    W = np.zeros((NDIR, 1, GK, GK))
    for k in range(NDIR):
        t = 2 * np.pi * k / NDIR
        W[k, 0] = np.cos(t) * dx + np.sin(t) * dy
    return W


def mix_weights():
    """orientation k: |r_k| - |r_perpendicular| - dead zone   (|r| = relu(r) + relu(-r))"""
    W = np.zeros((NORI, NDIR, 1, 1))
    for k in range(NORI):
        W[k, k, 0, 0] += 1.0
        W[k, k + NORI, 0, 0] += 1.0
        p = (k + NORI // 2) % NORI
        W[k, p, 0, 0] -= 1.0
        W[k, p + NORI, 0, 0] -= 1.0
    return W, np.full(NORI, -DEAD)


def pool_kernel():
    r = np.arange(PK) - (PK - 1) / 2.0
    g = np.exp(-r ** 2 / (2 * PSIG ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def conv2d(x, W, b, stride=1, padding=0):
    """x (N, C, H, W); W (O, C, k, k).  Plain correlation, same formula as the task statement."""
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    out = np.zeros((n, o, ho, wo))
    for dy in range(kh):
        for dx in range(kw):
            patch = xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
            out += np.einsum("nchw,oc->nohw", patch, W[:, :, dy, dx], optimize=True)
    return out + np.asarray(b)[None, :, None, None]


def edge_features(img, mean, std):
    """img (N, S, S) grey images; mean, std (N,) normalisation constants.  Returns pooled orientation maps (N, 8, c, c)."""
    x = ((img - mean[:, None, None]) / std[:, None, None])[:, None]
    r = np.maximum(conv2d(x, gradient_filters(), np.zeros(NDIR)), 0)
    Wm, bm = mix_weights()
    e = np.maximum(conv2d(r, Wm, bm), 0)
    e = CLIP - np.maximum(CLIP - e, 0)
    Wp = np.zeros((NORI, NORI, PK, PK))
    for k in range(NORI):
        Wp[k, k] = pool_kernel()
    return conv2d(e, Wp, np.zeros(NORI), stride=PSTR, padding=PPAD)


# region cue: how far a pixel is from the background level, whichever way (lighter or darker).
# The background level is taken as the mean of the top rim of the picture, where the hand rarely is
# (the forearm comes in from below or from the side).
RIM = 2               # rim thickness
RIM_DOWN = 10         # the rim runs along the whole top edge and this far down both sides
FIG_LO = 0.25         # |pixel - background| below this (in image standard deviations) counts as background
FIG_HI = 1.00         # ... above this counts fully as figure
RPAD = 1              # region pooling padding (so region cells sit on the edge cells)
REGION_SD = 0.45      # standard deviation of a clean figure map (about a half-covered frame)
REGION_WEIGHT = 1.0   # region match counts as much as edge match
CUT_SLACK = 1.0       # the crop is made around the hand: only positions that keep the hand box (palm and fingers, not the
                      # forearm) inside the frame, give or take this many pixels, are used ...
CENTRE_TOL = 4.0      # ... and keep the centre of that box within this many pixels of the frame centre
TEXTURE = 0.03        # skin is textured (creases, shading) and the background is plain: a template expects this much edge
                      # energy of every orientation all over the hand and forearm, on top of its drawn edges


def rim_mask():
    m = np.zeros((FRAME, FRAME))
    m[:RIM, :] = 1
    m[:RIM_DOWN, :RIM] = 1
    m[:RIM_DOWN, -RIM:] = 1
    return m / m.sum()


def pool_cells(x, padding):
    """Gaussian pooling of (N, S, S) maps to cells."""
    return conv2d(x[:, None], pool_kernel()[None, None], np.zeros(1), stride=PSTR, padding=padding)[:, 0]


# ----------------------------------------------------------------------------------------------------------------
# drawing: a hand is a list of layers painted back to front; a layer is a union of capsules / rounded boxes,
# filled with a skin tone and edged with a line: against the ground this is just the hand's outline step, on top of
# another part it is the faint shadow line between the two.
# ----------------------------------------------------------------------------------------------------------------
BG, SKIN = 0.55, 0.80
LINE_SOFT = 0.75      # the faint line where one part of the hand lies on another: much weaker than the hand's outline step,
                      # because what goes on inside a real hand (shading, creases) is not something these drawings can
                      # predict, while the outline against a plain wall is
LINE_W = 0.9          # width of that line, frame pixels
SS = 2                # supersampling
SOFT = 0.6            # blur (frame pixels) standing for the softness of a shrunk photo


def cap(a, b, r0, r1=None, arm=False):
    """capsule from a to b with end radii r0, r1; arm=True marks the forearm (not counted in the hand's bounding box)"""
    return ["cap", np.array(a, float), np.array(b, float), float(r0), float(r0 if r1 is None else r1), arm]


def rbox(c, hw, hh, rad, ang=0.0):
    return ["box", np.array(c, float), float(hw), float(hh), float(rad), float(ang)]


def stroke(pts, r, r1=None):
    r1 = r if r1 is None else r1
    n = len(pts) - 1
    return [cap(pts[i], pts[i + 1], r + (r1 - r) * i / n, r + (r1 - r) * (i + 1) / n) for i in range(n)]


def _sdf(prim, X, Y):
    if prim[0] == "cap":
        a, b, r0, r1 = prim[1:5]
        ab = b - a
        L2 = float(ab @ ab) + 1e-12
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0, 1)
        return np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1]) - (r0 + t * (r1 - r0))
    _, c, hw, hh, rad, ang = prim
    ca, sa = np.cos(ang), np.sin(ang)
    u = (X - c[0]) * ca + (Y - c[1]) * sa
    v = -(X - c[0]) * sa + (Y - c[1]) * ca
    qx, qy = np.abs(u) - (hw - rad), np.abs(v) - (hh - rad)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rad


def place(hand, palm_px, tilt, wide, canvas=CANVAS):
    """Map a hand (typed in palm-width units, y up, palm centre at the origin) into canvas pixels: stretch it by `wide`
    across its width (x, or y for a hand typed pointing sideways; fingers running along the hand get thicker, a thumb
    lying across it gets longer, the forearm thickens too), rotate by `tilt` degrees about the palm centre, scale to
    `palm_px` pixels per palm width, palm centre at the canvas centre.
    Returns the placed layers and the hand box (x0, y0, x1, y1) in canvas pixels (palm and fingers, not the forearm)."""
    t = np.deg2rad(tilt)
    sideways = hand.get("across") == "y"
    M = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ np.diag([1.0, wide] if sideways else [wide, 1.0])
    s = float(palm_px)
    lo, hi = np.array([1e9, 1e9]), np.array([-1e9, -1e9])
    out = []
    for layer in hand["layers"]:
        new = []
        for p in layer:
            def px(q):
                z = (M @ q) * s
                return np.array([canvas / 2.0 + z[0], canvas / 2.0 - z[1]])
            if p[0] == "cap":
                d = p[2] - p[1]
                L = float(np.hypot(d[0], d[1]))
                if p[5]:
                    k = s * wide
                elif L < 1e-9:
                    k = s * np.sqrt(wide)
                else:
                    nx, ny = -d[1] / L, d[0] / L                 # normal of the capsule's axis
                    k = s / (np.hypot(nx, ny / wide) if sideways else np.hypot(nx / wide, ny))
                new.append(["cap", px(p[1]), px(p[2]), p[3] * k, p[4] * k, p[5]])
                if not p[5]:
                    for q, r in ((new[-1][1], new[-1][3]), (new[-1][2], new[-1][4])):
                        lo, hi = np.minimum(lo, q - r), np.maximum(hi, q + r)
            else:
                new.append(["box", px(p[1]), p[2] * s * (1.0 if sideways else wide), p[3] * s * (wide if sideways else 1.0),
                            p[4] * s, -(p[5] + t)])
                ca, sa = np.cos(p[5]), np.sin(p[5])
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        q = px(p[1] + np.array([ca * sx * p[2] - sa * sy * p[3], sa * sx * p[2] + ca * sy * p[3]]))
                        lo, hi = np.minimum(lo, q), np.maximum(hi, q)
        out.append(new)
    return out, (lo[0], lo[1], hi[0], hi[1])


def _soften(img):
    n = img.shape[0]
    r = np.arange(-2, 3)
    k = np.exp(-r ** 2 / (2 * SOFT ** 2))
    k /= k.sum()
    p = np.pad(img, 2, mode="edge")
    img = sum(k[i] * p[i:i + n, :] for i in range(5))
    p = np.pad(img, ((0, 0), (2, 2)), mode="edge")
    return sum(k[i] * p[:, i:i + n] for i in range(5))


def paint(layers, canvas=CANVAS, ss=SS):
    """returns (grey picture, figure mask, hand-without-forearm mask), all canvas x canvas"""
    n = canvas * ss
    g = (np.arange(n) + 0.5) / ss
    Y, X = np.meshgrid(g, g, indexing="ij")
    img = np.full((n, n), BG)
    fig = np.zeros((n, n))
    hand = np.zeros((n, n))
    for layer in layers:
        line = LINE_SOFT
        d = np.full((n, n), 1e9)
        for p in layer:
            dp = _sdf(p, X, Y)
            d = np.minimum(d, dp)
            if not (p[0] == "cap" and p[5]):
                hand = np.maximum(hand, np.clip(0.5 - dp * ss, 0, 1))
        a_out = np.clip(0.5 - (d - LINE_W / 2) * ss, 0, 1)
        a_in = np.clip(0.5 - (d + LINE_W / 2) * ss, 0, 1)
        img = img * (1 - a_out) + line * a_out
        img = img * (1 - a_in) + SKIN * a_in
        fig = np.maximum(fig, np.clip(0.5 - d * ss, 0, 1))
    img = img.reshape(canvas, ss, canvas, ss).mean(axis=(1, 3))
    fig = fig.reshape(canvas, ss, canvas, ss).mean(axis=(1, 3))
    hand = hand.reshape(canvas, ss, canvas, ss).mean(axis=(1, 3))
    return _soften(img), _soften(fig), _soften(hand)


# ----------------------------------------------------------------------------------------------------------------
# the hand model.  Units: palm width = 1, y up.  A right hand seen from the front with the palm to the viewer:
# thumb on the viewer's right (+x), then index, middle, ring, little finger towards -x.
# Proportions are ordinary hand anatomy: palm a little longer than wide, knuckle line at y = 0.5, fingers 0.7 - 0.95
# long and 0.23 wide, thumb 0.27 wide, wrist 0.8 wide.  A curled finger is seen as a short column hanging from its
# bend at about the knuckle line (a little above it in a clenched fist); a straight finger stands a full finger
# length above that.
# ----------------------------------------------------------------------------------------------------------------
FR = 0.115
FX = {"p": -0.375, "r": -0.125, "m": 0.125, "i": 0.375}
FLEN = {"p": 0.72, "r": 0.92, "m": 1.00, "i": 0.90}
CURL_TOP = {"p": 0.44, "r": 0.52, "m": 0.56, "i": 0.52}      # top of a loosely curled finger (curling round to the thumb)
FIST_TOP = {"p": 0.54, "r": 0.62, "m": 0.66, "i": 0.62}      # top of a finger in a clenched fist


def unit(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a)])


def forearm(at, deg):
    """forearm leaving the hand at `at`, running off the picture in direction `deg` (degrees, 0 = +x, -90 = down)"""
    at = np.array(at, float)
    return cap(at, at + 3.4 * unit(deg), 0.40, 0.44, arm=True)


def palm(arm=0.0, bottom=-0.45):
    """palm box plus forearm leaving its lower edge; arm = degrees the forearm swings from straight down (positive: to +x).
    The palm of a fist leans back and is partly covered by the fingers, so it shows shorter than a flat palm (bottom=-0.6)."""
    return [rbox((0, (0.5 + bottom) / 2), 0.5, (0.5 - bottom) / 2, 0.22), forearm((0.0, bottom + 0.2), -90 + arm)]


def col(f, top=None, bot=0.10, r=0.12):
    """a loosely curled finger seen from the front: a column from the bend down to the tip (top, bot: its two ends)"""
    top = CURL_TOP[f] if top is None else top
    return [cap((FX[f], top - r), (FX[f], bot + r), r)]


def fist(f, top=None, bot=0.05, r=0.12):
    """a finger of a clenched fist: a longer column"""
    top = FIST_TOP[f] if top is None else top
    return [cap((FX[f], top - r), (FX[f], bot + r), r)]


def ext(f, lean=0.0, length=None, base=0.40, r=FR):
    """a straight finger; lean in degrees from upright (positive leans to +x, the thumb side)"""
    L = (FLEN[f] if length is None else length) + (0.5 - base) - r
    a = np.array([FX[f], base])
    return [cap(a, a + L * unit(90 - lean), r, 0.9 * r)]


def thumb(kind):
    if kind == "side":      # lying along the side of the index finger, tip up (A)
        return [cap((0.47, -0.28), (0.62, 0.44), 0.15, 0.125)]
    if kind == "cross":     # across the front of the curled fingers (S, and the fist under I, R, U, V ...)
        return [cap((0.50, -0.20), (-0.02, 0.14), 0.145, 0.125)]
    if kind == "palm":      # folded across the palm (B, W)
        return [cap((0.48, -0.30), (-0.05, -0.02), 0.15, 0.125)]
    if kind == "out":       # straight out sideways (L)
        return [cap((0.42, -0.28), (1.15, -0.02), 0.15, 0.12)]
    if kind == "diag":      # out and up (Y)
        return [cap((0.42, -0.28), (1.02, 0.38), 0.15, 0.12)]
    if kind == "ring":      # reaching to the curled middle finger tip (D)
        return [cap((0.48, -0.26), (0.10, 0.14), 0.145, 0.125)]
    raise ValueError(kind)


def front(parts, arm=0.0, bottom=-0.45):
    return {"layers": [palm(arm, bottom)] + parts}


def clenched(parts, arm=0.0):
    """a fist seen from the front: the heel of the palm shows below the fingers"""
    return front(parts, arm, bottom=-0.6)


def side_hand(n_ext, below=8.0, thumb_seen=False):
    """G / H: the hand turned so the fingers point to +x, back of the hand to the viewer, index finger on top.
    n_ext fingers are straight, the rest are curled with their knuckles to the right.
    below: the forearm comes in from the left, this many degrees below the horizontal."""
    layers = [[rbox((-0.05, 0.0), 0.55, 0.5, 0.22), forearm((-0.40, 0.0), 180 + below)]]
    rows = [0.375, 0.125, -0.125, -0.375]
    names = ["i", "m", "r", "p"]
    for j in range(3, n_ext - 1, -1):
        layers.append([cap((0.25, rows[j]), (0.60, rows[j]), 0.12)])
    if thumb_seen:
        layers.append([cap((0.10, 0.30), (0.82, 0.20), 0.135, 0.12)])
    for j in range(n_ext - 1, -1, -1):
        layers.append([cap((0.25, rows[j]), (0.5 + FLEN[names[j]] - FR, rows[j]), FR, 0.9 * FR)])
    return {"layers": layers, "across": "y"}


ARMS = (0.0, -15.0)     # forearm straight below the hand, or coming up from the lower left (the signer's right elbow)


def hands(letter):
    """the poses typed for each letter: two hand shapes (v = 0, 1), each with two forearm directions"""
    def both(mk, arms=ARMS):
        return [mk(a, v) for v in (0, 1) for a in arms]

    pick = lambda v, a, b: a if v == 0 else b
    if letter == "A":       # fist, thumb upright against the side of the index finger
        return both(lambda arm, v: clenched([fist("p"), fist("r"), fist("m"), fist("i"),
                                          pick(v, thumb("side"), [cap((0.47, -0.28), (0.68, 0.34), 0.15, 0.125)])], arm))
    if letter == "S":       # fist, thumb across the front of the fingers
        return both(lambda arm, v: clenched([fist("p"), fist("r"), fist("m"), fist("i"),
                                          pick(v, thumb("cross"), [cap((0.52, -0.14), (-0.10, 0.06), 0.145, 0.125)])], arm))
    if letter == "E":       # fingers upright from the knuckles then bent down onto the thumb, which lies across the palm:
        # the tallest of the fists, short finger columns on top, thumb bar under them
        return both(lambda arm, v: clenched([fist(f, FIST_TOP[f] + pick(v, 0.26, 0.18), 0.40) for f in "prmi"]
                                         + [[cap((0.50, 0.02), (-0.28, pick(v, 0.22, 0.16)), 0.145, 0.125)]], arm))
    if letter == "M":       # three fingers folded down over the thumb, thumb tip peeping out beyond the ring finger
        return both(lambda arm, v: clenched([fist("p"), [cap((-0.27, -0.14), (-0.30, -0.20), 0.12)],
                                          fist("r", bot=pick(v, -0.14, -0.06)), fist("m", bot=pick(v, -0.14, -0.06)),
                                          fist("i", bot=pick(v, -0.14, -0.06))], arm))
    if letter == "N":       # two fingers folded down over the thumb, thumb tip between middle and ring finger
        return both(lambda arm, v: clenched([fist("p"), fist("r"), [cap((-0.02, -0.14), (-0.05, -0.20), 0.12)],
                                          fist("m", bot=pick(v, -0.14, -0.06)), fist("i", bot=pick(v, -0.14, -0.06))], arm))
    if letter == "T":       # thumb pushed up between index and middle finger; the bent index finger juts out sideways over it
        def mk(arm, v):
            if v == 0:      # fist from the front, the index knuckle sticking out to the side, thumb tip on top
                return clenched([fist("p"), fist("r"), fist("m"), [cap((0.20, 0.52), (0.24, 0.84), 0.12)],
                              [cap((0.10, 0.44), (1.05, 0.50), 0.14, 0.125)]], arm)
            # hand turned sideways like G, but the index finger is short and bent and the thumb tip stands up behind it
            layers = [[rbox((-0.05, 0.0), 0.55, 0.5, 0.22), forearm((-0.25, -0.30), -90 + arm - 25)]]
            for y in (-0.375, -0.125, 0.125):
                layers.append([cap((0.25, y), (0.60, y), 0.12)])
            layers.append([cap((0.05, 0.45), (0.10, 0.74), 0.12)])
            layers.append([cap((-0.10, 0.38), (1.10, 0.40), 0.14, 0.125)])
            return {"layers": layers, "across": "y"}
        return both(mk)
    if letter == "B":       # flat hand, four fingers up together, thumb folded across the palm
        return both(lambda arm, v: front([ext("p", pick(v, 0, 4) - 2, base=0.42) + ext("r", pick(v, 0, 4) - 1, base=0.42)
                                          + ext("m", pick(v, 0, 4), base=0.42) + ext("i", pick(v, 0, 4) + 1, base=0.42),
                                          ext("r", pick(v, 0, 4) - 1, base=0.42), ext("i", pick(v, 0, 4) + 1, base=0.42),
                                          thumb("palm")], arm, bottom=-0.75))
    if letter == "D":       # index finger up; the other three curl round to meet the thumb
        return both(lambda arm, v: front([ext("i", pick(v, 0, 6), base=0.30), col("p"), col("r"), col("m"), thumb("ring")], arm))
    if letter == "I":       # fist with the little finger up
        return both(lambda arm, v: front([ext("p", pick(v, -3, -12), base=0.30, r=0.10), col("r"), col("m"), col("i"), thumb("cross")], arm))
    if letter == "L":       # index finger up, thumb straight out sideways
        return both(lambda arm, v: front([ext("i", 0, base=0.30), col("p"), col("r"), col("m"),
                                          pick(v, thumb("out"), [cap((0.42, -0.28), (1.10, 0.18), 0.15, 0.12)])], arm))
    if letter == "Y":       # little finger and thumb spread out, the three middle fingers curled
        return both(lambda arm, v: front([ext("p", pick(v, -25, -12), base=0.30, r=0.10), col("r"), col("m"), col("i"), thumb("diag")], arm))
    if letter == "R":       # index and middle finger up and crossed: one thick finger
        return both(lambda arm, v: front([ext("m", pick(v, 8, 10), base=0.30), ext("i", pick(v, -8, -4), base=0.30),
                                          col("p"), col("r"), thumb("cross")], arm))
    if letter == "U":       # index and middle finger up, side by side
        return both(lambda arm, v: front([ext("m", pick(v, 0, 5), base=0.30), ext("i", pick(v, 0, 5), base=0.30),
                                          col("p"), col("r"), thumb("cross")], arm))
    if letter == "V":       # index and middle finger up and spread
        return both(lambda arm, v: front([ext("m", pick(v, -14, -10), base=0.34), ext("i", pick(v, 16, 22), base=0.34),
                                          col("p"), col("r"), thumb("cross")], arm))
    if letter == "W":       # index, middle and ring finger up and spread; thumb holds the little finger down
        return both(lambda arm, v: front([ext("r", -pick(v, 20, 14), base=0.36), ext("m", 0, base=0.40), ext("i", pick(v, 20, 14), base=0.36),
                                          col("p"), thumb("palm")], arm))
    if letter == "K":       # index up, middle finger pointing up and forward (so it looks shorter), thumb tip between them
        # (v = 1: the middle finger points out to the side, away from the thumb)
        return both(lambda arm, v: front([ext("m", pick(v, -22, -60), length=pick(v, 0.60, 0.70), base=0.34), ext("i", 4, base=0.34),
                                          col("p"), col("r"), [cap((0.48, -0.25), (0.22, 0.56), 0.145, 0.125)]], arm))
    if letter == "F":       # index finger tip on thumb tip (a ring at the side), the other three fingers up and spread
        return both(lambda arm, v: front([ext("p", -2 * pick(v, 14, 20), base=0.34, r=0.10), ext("r", -pick(v, 14, 20), base=0.38),
                                          ext("m", pick(v, 4, 0), base=0.40),
                                          stroke([(0.36, 0.36), (0.76, 0.62), (0.98, 0.32)], 0.11),
                                          stroke([(0.44, -0.28), (0.86, -0.08), (0.98, 0.26)], 0.14, 0.12)], arm))
    if letter == "X":       # fist with the index finger raised and hooked: a short thick arch standing on the fist, its tip
        # curling back down towards the thumb
        return both(lambda arm, v: front([col("p"), col("r"), col("m"),
                                          stroke(pick(v, [(0.12, 0.40), (0.14, 0.86), (0.38, 1.00), (0.62, 0.86), (0.62, 0.62)],
                                                      [(0.20, 0.40), (0.22, 0.80), (0.44, 0.92), (0.66, 0.78), (0.66, 0.58)]), 0.12),
                                          thumb("cross")], arm))
    if letter == "G":       # index finger pointing sideways
        return both(lambda below, v: side_hand(1, below, thumb_seen=(v == 1)), arms=(25.0, 55.0))
    if letter == "H":       # index and middle finger pointing sideways
        return both(lambda below, v: side_hand(2, below, thumb_seen=(v == 1)), arms=(25.0, 55.0))
    if letter == "C":       # curved hand seen from the side, opening to the right: fingers arch over, thumb below
        def mk(arm, v):
            base = [rbox((-0.05, 0.0), 0.55, 0.52, 0.25), forearm((-0.18, -0.32), -90 + arm)]   # the cupped palm fills most of the C
            fingers = stroke([(-0.28, 0.40), (-0.04, 0.80), (0.42, 0.90), pick(v, (0.82, 0.64), (0.76, 0.52))], 0.27, 0.20)
            return {"layers": [base, fingers, [cap((-0.10, -0.40), pick(v, (0.70, -0.34), (0.64, -0.24)), 0.19, 0.16)]]}
        return both(mk, arms=(-12.0, -25.0))
    if letter == "O":       # like C but the finger tips come down onto the thumb tip: a closed ring with a hole
        def mk(arm, v):
            meet = pick(v, (0.74, 0.12), (0.66, 0.16))
            base = [rbox((-0.24, 0.0), 0.36, 0.48, 0.24), forearm((-0.20, -0.32), -90 + arm)]
            fingers = stroke([(-0.28, 0.40), (-0.02, 0.78), (0.42, 0.82), (0.72, 0.50), meet], 0.25, 0.18)
            th = stroke([(-0.12, -0.40), (0.44, -0.24), meet], 0.19, 0.16)
            return {"layers": [base, fingers, th]}
        return both(mk, arms=(-12.0, -25.0))
    if letter == "P":       # K hand tipped over: index finger points away to the left, middle finger down, forearm from lower right
        def mk(arm, v):
            base = [rbox((0.15, 0.0), 0.50, 0.48, 0.24), forearm((0.40, -0.08), arm)]
            return {"layers": [base, [cap((0.0, -0.05), pick(v, (-0.50, -0.66), (-0.30, -0.72)), FR)],
                               [cap((0.25, -0.12), (-0.30, 0.04), 0.14, 0.125)],
                               [cap((0.0, 0.34), pick(v, (-1.05, 0.78), (-1.00, 0.45)), 0.125, 0.11)]]}
        return both(mk, arms=(-60.0, -35.0))
    if letter == "Q":       # G hand tipped over: wrist arched on top, index finger and thumb hang down, forearm from below on the left
        def mk(arm, v):
            base = [rbox((0.10, 0.36), 0.56, 0.42, 0.30), forearm((-0.30, 0.25), arm)]
            return {"layers": [base, [cap((0.12, 0.10), pick(v, (0.28, -0.60), (0.40, -0.56)), 0.14, 0.125)],
                               [cap((0.46, 0.22), pick(v, (0.72, -0.78), (0.86, -0.66)), FR, 0.9 * FR)]]}
        return both(mk, arms=(-100.0, 140.0))   # the forearm rises from below on the left, or comes down from the upper left
    raise ValueError(letter)


# ----------------------------------------------------------------------------------------------------------------
# template bank and network
# ----------------------------------------------------------------------------------------------------------------
PALMS = (10.5, 12.0, 13.5, 15.5)         # palm width in pixels: the frame is about two to two and a half palm widths wide
TILTS = (-24.0, -12.0, 0.0, 12.0, 24.0)  # in-plane turn of hand and forearm about the palm centre, degrees (negative: clockwise)
TILTS_FOR = {"M": (-48.0, -36.0, -24.0, -12.0, 0.0),   # M, N and S are made with the fist rolled over towards the thumb
             "N": (-48.0, -36.0, -24.0, -12.0, 0.0),   # side, knuckles up and to the right
             "S": (-36.0, -24.0, -12.0, 0.0, 12.0)}
WIDES = (1.00, 1.20)                     # slim / broad hands (build, the bulk of thumb and curled fingers)
GRID = 2                                 # palm centres lie on this pixel grid, the same grid for every letter


def variants(letter):
    return [(h, p, t, w) for h in hands(letter) for p in PALMS for t in TILTS_FOR.get(letter, TILTS) for w in WIDES]


def make_templates(hand_list):
    """hand_list: list of (hand, palm_px, tilt, wide).  Every drawing is cut out of its canvas at each allowed position
    of the palm centre; each cut-out is one template, normalised by its own numbers only.
    Returns edge rows (N, 8 * 12 * 12) and region rows (N, 12 * 12), float32."""
    rows_e, rows_r = [], []
    pics, figs = [], []
    mid = CANVAS // 2

    def flush():
        if not pics:
            return
        P, F = np.array(pics), np.array(figs)
        mean = P.mean(axis=(1, 2))
        std = np.sqrt(P.var(axis=(1, 2)) + 1e-5)
        R = pool_cells(F, RPAD)
        E = edge_features(P, mean, std) + TEXTURE * R[:, None]
        E = E.reshape(len(P), -1)
        E = E - E.mean(axis=1, keepdims=True)
        E = E / (np.sqrt((E ** 2).mean(axis=1, keepdims=True)) * E.shape[1])      # (frame features, zero mean unit variance) . row = correlation
        R = R.reshape(len(F), -1)
        R = R - R.mean(axis=1, keepdims=True)
        R = REGION_WEIGHT * R / (np.sqrt((R ** 2).mean(axis=1, keepdims=True)) * REGION_SD * R.shape[1])
        rows_e.append(E.astype(np.float32))
        rows_r.append(R.astype(np.float32))
        del pics[:], figs[:]

    for h, p, t, w in hand_list:
        layers, (x0, y0, x1, y1) = place(h, p, t, w)
        img, fig, _ = paint(layers)
        bx, by = (x0 + x1) / 2 - mid, (y0 + y1) / 2 - mid      # hand box centre relative to the palm centre
        for cy in range(0, FRAME + 1, GRID):                    # palm centre in the frame
            if abs(cy + by - FRAME / 2) > CENTRE_TOL or cy + y0 - mid < -CUT_SLACK or cy + y1 - mid > FRAME + CUT_SLACK:
                continue
            for cx in range(0, FRAME + 1, GRID):
                if abs(cx + bx - FRAME / 2) > CENTRE_TOL or cx + x0 - mid < -CUT_SLACK or cx + x1 - mid > FRAME + CUT_SLACK:
                    continue
                a, b = mid - cy, mid - cx
                pics.append(img[a:a + FRAME, b:b + FRAME])
                figs.append(fig[a:a + FRAME, b:b + FRAME])
        if len(pics) >= 400:
            flush()
    flush()
    return np.concatenate(rows_e), np.concatenate(rows_r)


def build(out_dir):
    W = {}
    layers = []

    def add(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=inputs, **kw))

    eye = np.eye(NORI)[:, :, None, None]
    add("n0", "norm", ["image"])
    # --- edge branch
    add("grad", "conv", ["n0"], out=NDIR, kernel=[GK, GK], stride=1, padding=0)
    W["grad.W"], W["grad.b"] = gradient_filters(), np.zeros(NDIR)
    add("grad_r", "relu", ["grad"])
    add("ori", "conv", ["grad_r"], out=NORI, kernel=[1, 1], stride=1, padding=0)
    W["ori.W"], W["ori.b"] = mix_weights()
    add("ori_r", "relu", ["ori"])
    add("sat_a", "conv", ["ori_r"], out=NORI, kernel=[1, 1], stride=1, padding=0)       # CLIP - e
    W["sat_a.W"], W["sat_a.b"] = -eye, np.full(NORI, CLIP)
    add("sat_r", "relu", ["sat_a"])
    add("sat_b", "conv", ["sat_r"], out=NORI, kernel=[1, 1], stride=1, padding=0)       # min(e, CLIP)
    W["sat_b.W"], W["sat_b.b"] = -eye, np.full(NORI, CLIP)
    add("cells", "conv", ["sat_b"], out=NORI, kernel=[PK, PK], stride=PSTR, padding=PPAD)
    Wp = np.zeros((NORI, NORI, PK, PK))
    for k in range(NORI):
        Wp[k, k] = pool_kernel()
    W["cells.W"], W["cells.b"] = Wp, np.zeros(NORI)
    add("cells_flat", "reshape", ["cells"], shape=[1, NORI * CELLS, CELLS])
    add("feat", "norm", ["cells_flat"])                                                 # zero mean, unit variance over all cells
    # --- region branch
    add("bgsub", "dense", ["n0"], out=FRAME * FRAME)                                   # pixel - mean of the top rim
    W["bgsub.W"] = np.eye(FRAME * FRAME) - np.ones((FRAME * FRAME, 1)) * rim_mask().reshape(1, -1)
    W["bgsub.b"] = np.zeros(FRAME * FRAME)
    add("diff", "reshape", ["bgsub"], shape=[1, FRAME, FRAME])
    add("pm", "conv", ["diff"], out=2, kernel=[1, 1], stride=1, padding=0)
    W["pm.W"], W["pm.b"] = np.array([1.0, -1.0]).reshape(2, 1, 1, 1), np.zeros(2)
    add("pm_r", "relu", ["pm"])
    add("dev", "conv", ["pm_r"], out=1, kernel=[1, 1], stride=1, padding=0)            # |diff| - FIG_LO
    W["dev.W"], W["dev.b"] = np.ones((1, 2, 1, 1)), np.array([-FIG_LO])
    add("dev_r", "relu", ["dev"])
    span = FIG_HI - FIG_LO
    add("fig_a", "conv", ["dev_r"], out=1, kernel=[1, 1], stride=1, padding=0)
    W["fig_a.W"], W["fig_a.b"] = -np.ones((1, 1, 1, 1)), np.array([span])
    add("fig_r", "relu", ["fig_a"])
    add("fig_b", "conv", ["fig_r"], out=1, kernel=[1, 1], stride=1, padding=0)         # min(|diff| - FIG_LO, span) >= 0
    W["fig_b.W"], W["fig_b.b"] = -np.ones((1, 1, 1, 1)), np.array([span])
    add("fig", "conv", ["fig_b"], out=1, kernel=[PK, PK], stride=PSTR, padding=RPAD)   # pooled figure map in 0..1
    W["fig.W"], W["fig.b"] = pool_kernel()[None, None] / span, np.zeros(1)
    # --- templates: one dense row per drawing; letters padded to the same count by repeating their first row
    per = [make_templates(variants(letter)) for letter in LETTERS]
    V = max(len(e) for e, _ in per)
    pad = lambda x: np.concatenate([x, np.repeat(x[:1], V - len(x), axis=0)])
    KE = np.concatenate([pad(e) for e, _ in per])
    KR = np.concatenate([pad(r) for _, r in per])
    add("match_e", "dense", ["feat"], out=NCLASS * V)
    W["match_e.W"], W["match_e.b"] = KE, np.zeros(NCLASS * V)
    add("match_r", "dense", ["fig"], out=NCLASS * V)
    W["match_r.W"], W["match_r.b"] = KR, np.zeros(NCLASS * V)
    add("match", "add", ["match_e", "match_r"])
    add("by_class", "reshape", ["match"], shape=[1, NCLASS, V])
    add("best", "maxpool", ["by_class"], size=[1, V], stride=[1, V])
    add("scores", "reshape", ["best"], shape=[NCLASS])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
    return [len(e) for e, _ in per]


if __name__ == "__main__":
    counts = build(sys.argv[1] if len(sys.argv) > 1 else ".")
    print("templates per letter:", counts)
