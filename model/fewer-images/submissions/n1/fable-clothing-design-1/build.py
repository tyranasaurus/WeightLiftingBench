#!/usr/bin/env python3
"""build.py -- writes network.json and weights.npz for the clothing classifier.   usage: python3 build.py OUT_DIR

Reads no files.  Every number comes from this source:
  * formula filters: soft-threshold units, a Gaussian blur, eight derivative-of-Gaussian edge filters, finite
    differences, a long thin vertical-line filter;
  * procedural drawings of garments, shoes and bags typed from general knowledge of what these things look like in a
    front / side product photo (polygons for body and sleeves, a sole line and an ankle curve, a bag with a handle
    loop, a sneaker with a pale rubber sole ...).  Each drawing becomes ONE template, normalised by itself alone;
  * a few hand-chosen constants: thresholds, weights of two hand rules, a small handicap for look-alike garments.
No training, no fitting, no statistics over images of any kind.

The network:  image -> soft mask -> (a) blurred, standardised silhouette  and  (b) standardised map of pooled edge
directions -> correlation of each with every template -> sum -> best template of each class -> + two hand rules.
"""
import sys, json, math
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# hand-chosen constants
# ----------------------------------------------------------------------------------------------------------------------
SEED = 20240611
K_PER_CLASS = 5000          # templates drawn per class
CANVAS = 168                # drawing canvas (6 samples per final pixel)
INK = 0.5                   # grey level given to a drawn item (a mid-grey garment)
# soft mask: m = W_LOW*clip((x-T0)/(T1-T0),0,1) + (1-W_LOW)*clip(x/T2,0,1).  The first term says "something is here"
# (so that pale items still give a silhouette), the second "it is solidly here" (so that the half-dark gaps between
# straps, legs, or sleeve and body stay visible at this resolution).
T0, T1, T2 = 0.015, 0.10, 0.5
W_LOW = 0.5
BLUR_SIGMA = 1.0            # blur of the mask before silhouette correlation
EDGE_SIGMA = 1.0            # scale of the derivative-of-Gaussian edge filters
EDGE_POOL, EDGE_STRIDE = 4, 2
LAMBDA_EDGE = 0.5           # weight of the edge-layout correlation relative to the silhouette correlation
# rule for strappy footwear: outline length (total variation of the mask) versus filled area
SPARSE_C = 0.6              # "strappy" when total variation > SPARSE_C * area
SPARSE_S0 = 40.0            # soft range of that comparison, in pixels
SPARSE_GAIN = 0.08          # bonus / penalty given to the sandal score
# rule for a full-length front opening (zip, button band): a long thin vertical line down the middle of the picture,
# stronger than the same kind of line beside the middle.  Coats and shirts open down the front; pullovers and T-shirts do not.
LINE_ROWS = 11              # length of the line filter, rows
LINE_T0 = 0.12              # line strength (in standard deviations of the picture) where the rule starts to act
LINE_T1 = 0.20              # ... and the further strength over which it reaches full effect
LINE_GAIN = 0.04            # bonus given to the coat and shirt scores
# Some kinds of garment have the silhouette of a more common kind from another class (a short-sleeved shirt looks like a
# T-shirt, a short jacket like a pullover).  Their templates carry a small handicap, so that they win only when they fit
# clearly better or when the front-opening rule speaks for them.  Smaller than LINE_GAIN on purpose.
HANDICAP = 0.03
HANDICAP_SMALL = 0.015


# ----------------------------------------------------------------------------------------------------------------------
# tiny drawing library: polygons on a fine canvas, union / subtraction, trim to the item, shrink so the longest side is 28
# ----------------------------------------------------------------------------------------------------------------------
def _fill(poly, R):
    """poly: (n,2) array of (x,y) canvas coordinates. Returns bool (R,R) mask (non-zero winding)."""
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    yc = np.arange(R) + 0.5
    dy = (y1 - y0)[:, None]
    valid = (yc[None, :] >= np.minimum(y0, y1)[:, None]) & (yc[None, :] < np.maximum(y0, y1)[:, None])
    t = (yc[None, :] - y0[:, None]) / np.where(dy == 0, 1.0, dy)
    xc = x0[:, None] + t * (x1 - x0)[:, None]
    col = np.clip(np.ceil(xc - 0.5), 0, R).astype(np.int64)
    sgn = np.where(dy > 0, 1.0, -1.0) * valid
    idx = (np.arange(R)[None, :] * (R + 1) + col).ravel()
    acc = np.bincount(idx, weights=sgn.ravel(), minlength=R * (R + 1)).reshape(R, R + 1)
    return np.abs(np.cumsum(acc, axis=1)[:, :R]) > 0.5


def _resample_matrix(n_in, n_out):
    """area-averaging matrix (n_out, n_in)"""
    r = n_in / n_out
    lo = np.arange(n_out)[:, None] * r
    hi = lo + r
    j = np.arange(n_in)[None, :]
    return np.clip(np.minimum(hi, j + 1) - np.maximum(lo, j), 0, None) / r


def render(add, sub=(), rng=None, dim=()):
    """add / sub: lists of polygons in any units (x right, y down). Draws union(add) minus union(sub) in grey INK,
    with the regions in dim = [(polygon, ink), ...] drawn in their own paler ink where they lie on the item;
    trims to the item, shrinks so that the longest side is 28 pixels, centres it. Returns the (28,28) picture."""
    allp = np.concatenate(add, axis=0)
    lo, hi = allp.min(0), allp.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    s = (CANVAS - 8) / span
    R = CANVAS
    mask = np.zeros((R, R), bool)
    for p in add:
        mask |= _fill((p - lo) * s + 4.0, R)
    for p in sub:
        mask &= ~_fill((p - lo) * s + 4.0, R)
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    out = np.zeros((28, 28))
    if len(rows) == 0:
        return out
    grey = mask * INK
    for p, ink in dim:
        grey = np.where(mask & _fill((p - lo) * s + 4.0, R), ink, grey)
    crop = grey[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    hc, wc = crop.shape
    f = 28.0 / max(hc, wc)
    h2, w2 = int(max(1, round(hc * f))), int(max(1, round(wc * f)))
    small = _resample_matrix(hc, h2) @ crop @ _resample_matrix(wc, w2).T
    oy, ox = (28 - h2) // 2, (28 - w2) // 2
    if rng is not None:                      # an odd leftover can sit on either side
        if (28 - h2) % 2 == 1 and rng.random() < 0.5: oy += 1
        if (28 - w2) % 2 == 1 and rng.random() < 0.5: ox += 1
    out[oy:oy + h2, ox:ox + w2] = small
    return out


def P(*pts):
    return np.array(pts, dtype=np.float64)


def mirror(right_pts):
    """right_pts: outline of the right half from top centre-ish to bottom (x >= 0). Returns closed symmetric polygon."""
    r = np.array(right_pts, dtype=np.float64)
    l = r[::-1].copy(); l[:, 0] *= -1
    return np.concatenate([r, l], axis=0)


def flipx(poly):
    q = np.array(poly, dtype=np.float64).copy(); q[:, 0] *= -1
    return q


def chaikin(poly, iters=1):
    """corner cutting of a closed polygon: rounds corners"""
    p = np.array(poly, dtype=np.float64)
    for _ in range(iters):
        q = np.roll(p, -1, axis=0)
        a, b = 0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q
        p = np.stack([a, b], axis=1).reshape(-1, 2)
    return p


def stroke(pts, width):
    """thick open polyline as one polygon. width: scalar or per-point array."""
    p = np.array(pts, dtype=np.float64)
    d = np.gradient(p, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    w = (np.broadcast_to(np.asarray(width, dtype=np.float64), (len(p),)) / 2.0)[:, None]
    return np.concatenate([p + nrm * w, (p - nrm * w)[::-1]], axis=0)


def curve(ctrl, n=12):
    """smooth open curve through control points (Catmull-Rom)"""
    c = np.array(ctrl, dtype=np.float64)
    if len(c) < 3:
        t = np.linspace(0, 1, n)[:, None]
        return c[0] * (1 - t) + c[-1] * t
    ext = np.concatenate([[2 * c[0] - c[1]], c, [2 * c[-1] - c[-2]]], axis=0)
    out = []
    m = max(2, n // (len(c) - 1))
    for i in range(len(c) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t = np.linspace(0, 1, m, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(c[-1][None, :])
    return np.concatenate(out, axis=0)


def arc(cx, cy, rx, ry, a0, a1, n=14):
    """ellipse arc, angles in degrees, measured from +x towards -y (up on the page)"""
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy - ry * np.sin(a)], axis=1)


def rect(x0, y0, x1, y1):
    return P((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def disc(cx, cy, rx, ry=None, n=14):
    return arc(cx, cy, rx, rx if ry is None else ry, 0, 360, n + 1)[:-1]


def rotate(polys, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    M = np.array([[c, -s], [s, c]])
    return [p @ M.T for p in polys]


def up(poly):
    """shoe drawings use y up from the ground; turn them into page coordinates (y down)"""
    q = np.array(poly, dtype=np.float64).copy(); q[:, 1] *= -1
    return q


class Rnd:
    def __init__(self, rng): self.g = rng
    def u(self, a, b): return a + (b - a) * self.g.random()
    def p(self, prob): return self.g.random() < prob
    def pick(self, n): return int(self.g.integers(0, n))
    def random(self): return self.g.random()


# ----------------------------------------------------------------------------------------------------------------------
# garments.  Units: pixels of a drawing whose main side is about 28.  x = 0 is the centre line, y grows downwards.
# ----------------------------------------------------------------------------------------------------------------------
def sleeve(S, ang_deg, length, top_w, cuff):
    """A sleeve is a tapering strip hanging from the shoulder tip S (right side, x > 0).  Its outer edge leaves S at
    ang_deg from straight down (0 = hangs down, 90 = sticks out sideways) and is `length` long; the strip is top_w wide at
    the shoulder (that end lies over the body) and `cuff` wide at its end."""
    a = math.radians(ang_deg)
    d = np.array([math.sin(a), math.cos(a)]); n = np.array([math.cos(a), -math.sin(a)])
    S = np.array(S, dtype=np.float64)
    co = S + length * d
    return P(S, co, co - n * cuff, S - n * top_w)


def top(r, L, bw, hw, nw, nd, sy, sw, ay, slv=None, waist=None, tail=0.0, collar=0.0, hood=0.0, vneck=None, rib=0.0):
    """Generic upper-body garment, front view.
    L length, bw half-width at the armpit, hw half-width at the hem, nw half-width of the neck opening, nd depth of the
    neck dip, sy how far the shoulder tip is below the neck, sw shoulder half-width, ay armpit height (y),
    slv = (angle, length, width at the shoulder, cuff width) or None, waist = (half-width, y), tail = how much the side seams are shorter than
    the centre (shirt tails), collar = height of a stand-up collar, hood = height of a hood bump, vneck = (half-width,
    depth) of an open front, rib = how much a ribbed hem pulls in."""
    right = [(0.0, nd), (nw, 0.0), (sw, sy), (bw, ay)]
    if waist is not None:
        right.append((waist[0], waist[1]))
    if rib > 0:
        right += [(hw, L - 2.6), (hw - rib, L - 2.2), (hw - rib, L)]
    elif tail > 0:
        right += [(hw, L - tail), (hw * 0.55, L - 0.15 * tail)]
    else:
        right.append((hw, L))
    right.append((0.0, L))
    add = [mirror(right)]
    sub = []
    if slv is not None:
        sp = sleeve((sw, sy), slv[0], slv[1], slv[2], slv[3])
        add += [sp, flipx(sp)]
    if collar > 0:
        add.append(mirror([(0.0, -collar), (nw * 0.95, -collar), (nw + 0.5, 0.8), (0.0, 0.8)]))
    if hood > 0:
        add.append(arc(0.0, 1.0, nw + 1.4, hood + 1.0, 0, 180, 12))
    if vneck is not None:
        sub.append(P((-vneck[0], -collar - 0.5), (vneck[0], -collar - 0.5), (0.0, vneck[1])))
    return add, sub


def gen_tshirt(r):
    L = 28.0
    t = r.random()
    if t < 0.76:                                   # ordinary T-shirt: short sleeves pointing out and down
        bw = r.u(6.2, 10.6); hw = bw * r.u(0.9, 1.08)
        nw = r.u(2.4, 4.4); nd = r.u(0.3, 2.4)
        sy = r.u(0.8, 3.0); sw = bw + r.u(-0.3, 1.2); ay = sy + r.u(6.0, 9.0)
        wt = r.u(5.5, 8.0)
        slv = (r.u(28, 80), r.u(4.0, 9.0), wt, wt * r.u(0.75, 1.0))
        return top(r, L, bw, hw, nw, nd, sy, sw, ay, slv)
    if t < 0.90:                                   # sleeveless top / vest: straps, deep neck and arm holes
        bw = r.u(5.8, 9.0); hw = bw * r.u(0.9, 1.1)
        sn = r.u(1.8, 3.6); so = sn + r.u(1.2, 3.0)
        nd = r.u(2.0, 7.0); ay = r.u(6.5, 11.0)
        right = [(0.0, nd), (sn * 0.9, nd * 0.45), (sn, 0.0), (so, 0.0), (so + r.u(0.0, 0.8), ay * 0.55), (bw, ay)]
        if r.p(0.4):
            right.append((bw * r.u(0.86, 0.97), r.u(14, 18)))
        right += [(hw, L), (0.0, L)]
        return [mirror(right)], []
    # fitted / long T-shirt with small cap sleeves
    bw = r.u(5.4, 7.6); hw = bw * r.u(0.95, 1.12)
    nw = r.u(2.4, 4.0); nd = r.u(0.5, 3.5)
    sy = r.u(0.8, 2.5); sw = bw + r.u(0.0, 1.0); ay = sy + r.u(5.0, 7.5)
    wt = r.u(4.5, 6.5)
    slv = (r.u(40, 85), r.u(2.0, 4.5), wt, wt * r.u(0.8, 1.0))
    waist = (bw * r.u(0.84, 0.96), r.u(13, 17)) if r.p(0.6) else None
    return top(r, L, bw, hw, nw, nd, sy, sw, ay, slv, waist=waist)


def gen_pullover(r):
    L = 28.0
    bw = r.u(7.0, 10.0)
    nw = r.u(2.6, 4.6); nd = r.u(0.4, 2.2)
    sy = r.u(1.5, 3.6); sw = bw + r.u(0.0, 1.6); ay = sy + r.u(6.5, 9.5)
    if r.p(0.65):                                  # sleeves hanging beside the body, a little apart from it
        ang = r.u(6, 18)
    else:                                          # sleeves spread out (laid flat)
        ang = r.u(18, 38)
    down = r.u(19.5, 25.5)                         # how far below the shoulder the cuff ends: about as low as the hem
    sl = down / math.cos(math.radians(ang))
    slv = (ang, sl, r.u(5.5, 8.0), r.u(3.0, 4.6))
    rib = r.u(0.3, 1.2) if r.p(0.6) else 0.0
    hw = bw * r.u(0.9, 1.03)
    collar = r.u(1.0, 2.8) if r.p(0.15) else 0.0
    hood = r.u(2.0, 4.0) if (collar == 0 and r.p(0.08)) else 0.0
    return top(r, L, bw, hw, nw, nd, sy, sw, ay, slv, rib=rib, collar=collar, hood=hood)


def gen_coat(r):
    L = 28.0
    jacket = r.p(0.25)
    nw = r.u(2.3, 4.2); nd = r.u(0.0, 1.5)
    if jacket:                                     # short jacket: wide, thick sleeves that end near the hem
        bw = r.u(7.5, 10.4); hw = bw * r.u(0.92, 1.06)
        sy = r.u(1.5, 3.8); sw = bw + r.u(0.0, 1.5); ay = sy + r.u(7.0, 10.0)
        ang = r.u(4.0, 15.0)
        down = r.u(19.5, 24.5)
        sl = down / math.cos(math.radians(ang))
        slv = (ang, sl, r.u(6.8, 9.0), r.u(3.8, 5.6))
    else:                                          # coat: longer than its sleeves, which hang straight down close to the body
        bw = r.u(6.4, 9.4); hw = bw * r.u(0.96, 1.2)
        sy = r.u(1.5, 3.8); sw = bw + r.u(0.0, 1.5); ay = sy + r.u(7.0, 10.0)
        ang = r.u(1.5, 10.0)
        down = r.u(16.0, 22.5)
        sl = down / math.cos(math.radians(ang))
        slv = (ang, sl, r.u(6.0, 8.5), r.u(3.4, 5.4))
    t = r.random()
    collar = hood = 0.0; vneck = None
    if t < 0.45:
        collar = r.u(0.8, 2.6)                     # stand-up collar
    elif t < 0.6:
        hood = r.u(2.0, 4.5)                       # hood
    elif t < 0.75:
        vneck = (r.u(1.0, 2.6), r.u(2.5, 8.0))     # worn open: a gap between the lapels
    waist = (bw * r.u(0.88, 0.98), r.u(13, 17)) if r.p(0.25) else None
    return top(r, L, bw, hw, nw, nd, sy, sw, ay, slv, waist=waist, collar=collar, hood=hood, vneck=vneck) + (({"handicap": HANDICAP},) if jacket else ())


def gen_shirt(r):
    L = 28.0
    t = r.random()
    if t >= 0.95:                                  # shirt folded as in its packaging: a rectangle with a collar on top
        fw = r.u(7.0, 9.8); nw = r.u(2.2, 3.6); c = r.u(0.6, 1.8)
        right = [(0.0, 0.0), (nw, 0.0), (fw, r.u(0.5, 2.0)), (fw * r.u(0.97, 1.03), L), (0.0, L)]
        add = [mirror(right), mirror([(0.0, -c), (nw * 0.95, -c), (nw + 0.5, 0.8), (0.0, 0.8)])]
        return add, [], {"handicap": HANDICAP_SMALL}
    if t < 0.54:                                   # long-sleeved button shirt / blouse: thin sleeves, collar, shirt tails
        bw = r.u(6.3, 9.0); hw = bw * r.u(0.94, 1.1)
        nw = r.u(2.2, 3.8); nd = r.u(0.3, 1.5)
        sy = r.u(1.2, 3.2); sw = bw + r.u(0.0, 1.3); ay = sy + r.u(6.0, 8.5)
        ang = r.u(5, 22)
        down = r.u(19.0, 25.5)
        sl = down / math.cos(math.radians(ang))
        slv = (ang, sl, r.u(4.8, 6.8), r.u(2.6, 4.0))
        tail = r.u(0.8, 3.5) if r.p(0.65) else 0.0
        vneck = (r.u(0.9, 2.2), r.u(2.0, 5.5)) if r.p(0.15) else None
        collar = r.u(0.4, 1.6) if r.p(0.65) else 0.0
        return top(r, L, bw, hw, nw, nd, sy, sw, ay, slv, tail=tail, vneck=vneck, collar=collar) + ({"handicap": HANDICAP_SMALL},)
    if t < 0.83:                                   # short-sleeved shirt / polo: longer, straighter sleeves than a T-shirt, collar
        bw = r.u(6.2, 9.8); hw = bw * r.u(0.95, 1.1)
        nw = r.u(2.2, 3.8); nd = r.u(0.3, 1.5)
        sy = r.u(1.0, 3.0); sw = bw + r.u(0.0, 1.3); ay = sy + r.u(6.0, 8.5)
        wt = r.u(5.5, 7.5)
        slv = (r.u(18, 55), r.u(6.0, 11.0), wt, wt * r.u(0.75, 1.0))
        tail = r.u(0.8, 3.5) if r.p(0.6) else 0.0
        vneck = (r.u(0.9, 2.2), r.u(2.0, 5.5)) if r.p(0.15) else None
        collar = r.u(0.4, 1.6) if r.p(0.7) else 0.0
        return top(r, L, bw, hw, nw, nd, sy, sw, ay, slv, tail=tail, vneck=vneck, collar=collar) + ({"handicap": HANDICAP},)
    # sleeveless blouse: wide shoulders, small arm holes, collar
    bw = r.u(6.0, 8.6); hw = bw * r.u(0.95, 1.15)
    nw = r.u(2.0, 3.4); so = bw * r.u(0.8, 1.0)
    ay = r.u(6.0, 9.0)
    right = [(0.0, r.u(1.0, 4.0)), (nw, 0.0), (so, r.u(0.8, 2.2)), (so - r.u(0.0, 0.8), ay * 0.6), (bw, ay)]
    tail = r.u(0.8, 3.0) if r.p(0.5) else 0.0
    if tail > 0:
        right += [(hw, L - tail), (hw * 0.55, L - 0.15 * tail)]
    else:
        right.append((hw, L))
    right.append((0.0, L))
    add = [mirror(right)]
    if r.p(0.5):
        c = r.u(0.4, 1.2)
        add.append(mirror([(0.0, -c), (nw * 0.95, -c), (nw + 0.5, 0.8), (0.0, 0.8)]))
    return add, [], {"handicap": HANDICAP}


def gen_dress(r):
    L = 28.0
    t = r.random()
    bu = r.u(3.5, 5.6)                             # bust half-width
    uy = r.u(5.0, 8.0)                             # under-arm height
    wy = uy + r.u(3.0, 6.0); wa = bu * r.u(0.72, 0.98)
    hy = wy + r.u(4.0, 7.0); hp = max(wa, bu * r.u(0.95, 1.2)) + r.u(0.0, 1.2)
    if r.p(0.45):
        hm = hp + r.u(-0.8, 1.0)                   # straight / sheath
    else:
        hm = hp + r.u(1.0, 4.5)                    # flared skirt
    lower = [(wa, wy), (hp, hy), (hm, L), (0.0, L)]
    if t < 0.58:                                   # straps or sleeveless
        sn = r.u(1.2, 2.8); so = sn + r.u(0.7, 2.3); nd = r.u(2.5, 7.0)
        right = [(0.0, nd), (sn * 0.9, nd * 0.45), (sn, 0.0), (so, 0.0), (so + r.u(-0.2, 0.6), uy * 0.55), (bu, uy)] + lower
        return [mirror(right)], []
    nw = r.u(1.8, 3.2); nd = r.u(0.8, 4.5)
    sy = r.u(0.8, 2.2); sw = bu + r.u(-0.2, 0.9)
    right = [(0.0, nd), (nw, 0.0), (sw, sy), (bu, uy)] + lower
    add = [mirror(right)]
    if t < 0.76:                                   # short sleeves
        wt = r.u(3.5, 5.0)
        sp = sleeve((sw, sy), r.u(25, 70), r.u(2.5, 6.0), wt, wt * r.u(0.75, 1.0))
    else:                                          # long thin sleeves hanging down
        ang = r.u(3, 16); down = r.u(12.0, 19.0)
        sp = sleeve((sw, sy), ang, down / math.cos(math.radians(ang)), r.u(3.0, 4.2), r.u(1.8, 3.0))
    add += [sp, flipx(sp)]
    return add, []


def gen_trouser(r):
    L = 28.0
    t = r.random()
    if t < 0.88:                                   # front view, two legs
        ww = r.u(4.0, 6.6)                         # waist half-width
        hip = ww + r.u(0.0, 1.3); yh = r.u(4.0, 7.5)
        yc = r.u(6.5, 11.5)                        # crotch
        lw = r.u(2.3, 4.4)                         # width of one leg at the ankle
        if r.p(0.25):
            gap = r.u(0.0, 0.5)                    # legs together
        else:
            gap = r.u(0.5, 3.6)                    # half of the gap between the ankles
        ox = gap + lw
        knee = None
        if r.p(0.3):                               # tapered then flared, or shaped leg
            ky = r.u(15, 20); kx_o = hip + (ox - hip) * (ky - yh) / (L - yh) + r.u(-0.7, 0.3)
            knee = (kx_o, ky)
        right = [(0.0, 0.0), (ww, 0.0), (hip, yh)]
        if knee: right.append(knee)
        right += [(ox, L), (gap, L), (0.0, yc)]
        poly = mirror(right)
        return [poly], []
    # side view / folded: one tapering leg shape
    w0 = r.u(5.5, 8.5); w1 = r.u(2.6, 4.6); lean = r.u(-1.5, 1.5)
    yh = r.u(5, 9); bulge = r.u(0.0, 1.2)
    poly = P((0, 0), (w0, 0), (w0 + bulge, yh), (lean + w0 / 2 + w1 / 2, L), (lean + w0 / 2 - w1 / 2, L), (-bulge * 0.3, yh))
    return [poly], []


# ----------------------------------------------------------------------------------------------------------------------
# footwear, side view, toe to the left.  Units: pixels of a shoe 28 long; y measured UP from the ground (flipped by up()).
# ----------------------------------------------------------------------------------------------------------------------
def gen_sneaker(r):
    high = r.p(0.18)
    H = r.u(15.0, 20.0) if high else r.u(8.6, 14.6)
    tx = r.u(13.5, 17.5) if high else r.u(14.5, 19.5)       # where the tongue / collar starts
    toe = H * (r.u(0.24, 0.38) if high else r.u(0.34, 0.50))  # height of the toe box
    dip = r.u(0.0, 0.14) * H
    bx = 27.4 - r.u(0.0, 1.2)
    spring = r.u(0.6, 1.8)
    pts = [(1.2, 0.0), (27.0, 0.0), (28.0, 0.9), (27.9, H * 0.55), (bx, H),
           ((tx + bx) / 2.0, H - dip), (tx, H - r.u(0.0, 0.9)),
           (tx - (tx - 7.0) * 0.5, toe + (H - toe) * r.u(0.34, 0.5)),
           (5.5, toe * 1.03), (1.8, toe * 0.94), (0.2, toe * 0.62), (0.0, spring)]
    dim = []
    if r.p(0.45):                                  # white rubber sole: almost as pale as the background of the photo
        ink = r.u(0.03, 0.16); st = r.u(1.8, 3.6)
        dim.append((up(rect(-2.0, -2.0, 30.0, st)), ink))
        if r.p(0.3):                               # ... and a white toe cap
            dim.append((up(rect(-2.0, -2.0, r.u(3.0, 5.5), toe * 1.2)), ink))
    return [up(chaikin(P(*pts), 2))], [], {"dim": dim}


def gen_boot(r):
    H = r.u(19.0, 30.0)
    toe = r.u(5.3, 8.6)
    xf = r.u(10.5, 16.5)                           # front of the shaft
    xb = r.u(25.6, 27.8)                           # back of the shaft
    hf = H + r.u(-2.2, 1.5)                        # the top line may slant
    xi, yi = xf - r.u(0.0, 1.8), toe + r.u(2.5, 7.0)   # where the instep meets the shaft
    yi = min(yi, hf - 3.0)
    hh = r.u(0.4, 6.5)                             # heel height
    hl = r.u(5.0, 9.0)                             # heel length
    pts = [(0.0, 1.4), (0.5, toe * 0.62), (3.0, toe * 0.96), (6.5, toe), ((6.5 + xi) / 2 + 0.3, (toe + yi) / 2 - r.u(0.0, 0.8)),
           (xi, yi), (xf + r.u(-1.0, 1.0), (yi + hf) / 2), (xf, hf), (xb, H), (xb + r.u(0.0, 0.6), H * 0.5), (28.0, hh + 1.5), (28.0, 0.0)]
    if hh > 1.6:
        pts += [(28.0 - hl, 0.0), (28.0 - hl - 0.6, hh * 0.85), (13.5, 0.4), (1.5, 0.0)]
    else:
        pts += [(1.5, 0.0)]
    return [up(chaikin(P(*pts), 2))], []


def _toe_strap(r, x0, x1, base, rise, w, filled):
    """a band over the forefoot seen from the side: a hollow arch, or a filled hump"""
    xm = (x0 + x1) / 2.0 + r.u(-0.8, 1.5)
    if filled:
        return curve([(x0, base - 0.3), ((x0 + xm) / 2, base + rise * 0.8), (xm, base + rise), (x1, base + rise * r.u(0.3, 0.7)), (x1 + 0.5, base - 0.3)], 16)
    return stroke(curve([(x0, base), (xm, base + rise), (x1, base)], 12), w)


def _ankle_strap(r, add, bx, by, w, front_x, front_y):
    """heel post going up from the back of the sole (bx, by), a band round the ankle, and a strap coming forward"""
    hy = by + r.u(4.0, 9.5)
    hx = bx - r.u(0.0, 2.0)
    add.append(stroke(P((bx, by), (hx, hy)), w))
    fx = hx - r.u(4.5, 9.0); fy = hy - r.u(0.0, 2.0)
    add.append(stroke(P((hx, hy), (fx, fy)), w))
    if r.p(0.65):
        add.append(stroke(P((fx, fy), (front_x, front_y)), w))


def gen_sandal(r):
    t = r.random()
    add = []
    dim = []
    w = r.u(1.2, 2.5)                              # strap width
    if t < 0.25:                                   # flat strappy sandal: thin sole, straps over the toes, ankle strap
        ts = r.u(1.4, 4.5); lift = r.u(0.0, 1.5); hk = r.u(0.0, 1.5)
        add.append(P((0.0, lift), (3.0, 0.0), (28.0, 0.0), (28.0, ts + hk), (18.0, ts + 0.3 * hk), (3.0, ts), (0.0, ts + lift)))
        x0 = r.u(1.5, 5.0); x1 = x0 + r.u(6.0, 10.0)
        add.append(_toe_strap(r, x0, x1, ts, r.u(2.5, 5.5), w, r.p(0.45)))
        if r.p(0.3):
            add.append(_toe_strap(r, x1 + r.u(0.0, 2.0), x1 + r.u(4.0, 7.0), ts, r.u(3.0, 6.0), w, False))
        if r.p(0.85):
            _ankle_strap(r, add, 27.0 - r.u(0.0, 1.2), ts + hk, w, r.u(10.0, 16.0), ts + r.u(0.0, 2.5))
        if r.p(0.4):
            add.append(P((22.0, ts), (28.0, ts + hk), (27.6, ts + hk + r.u(2.5, 5.5)), (23.5, ts + 1.0)))   # heel cup
    elif t < 0.40:                                 # slide / flip-flop: sole and one band (or thong) over the forefoot
        ts = r.u(1.8, 5.5); lift = r.u(0.0, 1.5); hk = r.u(0.0, 2.0)
        add.append(P((0.0, lift), (3.0, 0.0), (28.0, 0.0), (28.0, ts + hk), (18.0, ts + 0.4 * hk), (3.0, ts), (0.0, ts + lift)))
        x0 = r.u(2.0, 6.0); x1 = x0 + r.u(8.0, 13.0)
        add.append(_toe_strap(r, x0, x1, ts, r.u(3.0, 6.5), w, r.p(0.6)))
    elif t < 0.70:                                 # high heel: sole rising to the right, heel under it, straps
        ts = r.u(1.1, 2.4)
        xh = r.u(21.0, 26.5); yh = r.u(5.0, 14.0)
        ball = r.u(8.0, 12.0)
        c = curve([(0.0, 0.6 + ts / 2), (ball * 0.5, ts / 2), (ball, ts / 2 + 0.3), ((ball + xh) / 2, yh * 0.55), (xh, yh), (28.0, yh + r.u(0.0, 1.0))], 18)
        add.append(stroke(c, ts))
        if r.p(0.6):                               # stiletto
            wt = r.u(2.5, 4.5); wb = r.u(0.9, 2.2)
            add.append(P((28.0 - wt, yh), (28.0, yh + 0.5), (27.6, 0.0), (27.6 - wb, 0.0)))
        else:                                      # block heel
            wt = r.u(4.5, 7.5); wb = wt * r.u(0.6, 1.0)
            add.append(P((28.0 - wt, yh * 0.9), (28.0, yh + 0.5), (28.0, 0.0), (28.0 - wb, 0.0)))
        if r.p(0.85):
            x0 = r.u(1.5, 5.0); x1 = x0 + r.u(5.0, 8.0)
            add.append(_toe_strap(r, x0, x1, ts, r.u(2.5, 5.0), w, r.p(0.45)))
        if r.p(0.85):
            _ankle_strap(r, add, 27.3, yh + 0.3, w, ball + r.u(0.0, 4.0), yh * 0.45)
        if r.p(0.4):
            add.append(P((xh - 2.0, yh - 0.3), (28.0, yh + 0.5), (27.8, yh + r.u(2.5, 5.0)), (xh, yh + 1.5)))   # heel cup
    elif t < 0.84:                                 # wedge / platform sole with straps
        t0 = r.u(1.5, 3.8); t1 = r.u(5.0, 11.0)
        add.append(P((0.0, 0.8), (2.0, 0.0), (28.0, 0.0), (28.0, t1), (17.0, t1 * 0.72 + t0 * 0.28), (9.0, t0 + 0.4), (0.0, t0 + 0.6)))
        x0 = r.u(1.5, 5.0); x1 = x0 + r.u(6.0, 9.0)
        add.append(_toe_strap(r, x0, x1, t0, r.u(3.0, 5.5), w, r.p(0.5)))
        if r.p(0.85):
            _ankle_strap(r, add, 27.2, t1, w, r.u(12.0, 17.0), t1 * 0.6 + r.u(0.0, 2.0))
    else:                                          # sporty / fisherman sandal: a shoe-shaped cage of straps over a solid sole
        H = r.u(8.5, 13.5); ts = r.u(1.8, 3.4)
        add.append(P((0.0, 1.0), (2.0, 0.0), (28.0, 0.0), (28.0, ts), (0.0, ts + 0.6)))
        w = max(w, 1.5)
        tx = r.u(15.0, 20.0); toe = ts + r.u(1.5, 3.5)
        rim = curve([(1.0, ts + 0.5), (4.0, toe), (10.0, toe + (H - toe) * 0.45), (tx, H), (26.5, H - r.u(0.0, 2.5)), (27.3, ts)], 20)
        if r.p(0.5):                               # the straps drawn one by one, with holes between them
            add.append(stroke(rim, w))
            nst = 2 + r.pick(3)
            for k in range(nst):
                x = 5.0 + (k + r.u(0.2, 0.8)) * (tx + 3.0 - 5.0) / nst
                ytop = toe + (H - toe) * min(1.0, max(0.0, (x - 4.0) / (tx - 4.0)))
                add.append(stroke(P((x + r.u(-2.0, 2.0), ts), (x, ytop)), w))
            if r.p(0.5):
                add.append(stroke(P((3.0, ts + 1.0), (26.0, ts + (H - ts) * r.u(0.35, 0.6))), w * 0.9))
        else:                                      # the cage seen from afar: straps and holes blend into a half-tone
            cage = np.concatenate([rim, P((27.3, ts - 0.2), (1.0, ts - 0.2))], axis=0)
            add.append(cage)
            dim.append((cage, r.u(0.09, 0.30)))
            if r.p(0.5):
                rs = stroke(rim, w)
                add.append(rs); dim.append((rs, INK))
    return [up(p) for p in add], [], {"dim": [(up(p), ink) for p, ink in dim]}


# ----------------------------------------------------------------------------------------------------------------------
# bags, front view.  x = 0 centre, y down, the top edge of the body at y = 0.
# ----------------------------------------------------------------------------------------------------------------------
def bag_body(r, bb, bt, hb, soft):
    """trapezoid body (top half-width bt, bottom half-width bb, height hb) with cut corners; soft = rounded bottom"""
    ct = r.u(0.4, 2.0)                                           # corner cut at the top
    cb = r.u(2.5, 6.0) if soft else r.u(0.4, 2.2)                # corner cut at the bottom
    bulge = bb * r.u(1.0, 1.06) if soft else bb
    pts = [(-bt + ct, 0.0), (bt - ct, 0.0), (bt, ct), (bulge, hb * 0.6), (bb, hb - cb), (bb - cb, hb),
           (-bb + cb, hb), (-bb, hb - cb), (-bulge, hb * 0.6), (-bt, ct)]
    return chaikin(P(*pts), 1)


def gen_bag(r):
    t = r.random()
    add = []
    if t < 0.42:                                   # hand bag / tote: body with one handle loop on top
        bb = 14.0; bt = bb * r.u(0.7, 1.08); hb = r.u(13.0, 22.0)
        add.append(bag_body(r, bb, bt, hb, r.p(0.3)))
        hx = r.u(3.5, 0.85 * min(bt, bb)); hy = r.u(3.5, 13.0); w = r.u(1.0, 2.3)
        add.append(stroke(arc(r.u(-0.5, 0.5), 0.6, hx, hy, 0, 180, 16), w))
        if r.p(0.25):                              # second handle seen behind the first
            add.append(stroke(arc(r.u(-1.5, 1.5), 0.6, hx * r.u(0.75, 1.1), hy * r.u(0.7, 0.95), 0, 180, 16), w))
    elif t < 0.66:                                 # clutch / purse / wallet / case: no handle
        bb = 14.0; bt = bb * r.u(0.85, 1.05); hb = r.u(12.0, 24.0)
        add.append(bag_body(r, bb, bt, hb, r.p(0.3)))
        if r.p(0.3):                               # short wrist strap or tab on top
            add.append(stroke(arc(r.u(-6.0, 6.0), 0.5, r.u(1.5, 3.0), r.u(1.5, 3.5), 0, 180, 10), r.u(0.9, 1.6)))
    elif t < 0.94:                                 # shoulder bag: long thin strap meeting high above the body
        bb = 14.0; bt = bb * r.u(0.75, 1.05); hb = r.u(14.0, 22.0)
        add.append(bag_body(r, bb, bt, hb, r.p(0.3)))
        sx = bt * r.u(0.55, 0.98); apex = r.u(8.0, 20.0); w = r.u(0.9, 1.7)
        if r.p(0.5):
            add.append(stroke(P((-sx, 0.5), (r.u(-1.5, 1.5), -apex), (sx, 0.5)), w))
        else:
            add.append(stroke(arc(0.0, 0.5, sx, apex, 0, 180, 18), w))
    else:                                          # back pack / bucket bag: tall rounded body, small loop on top
        hb = 26.0; bb = r.u(8.0, 12.0); bt = bb * r.u(0.55, 0.85)
        add.append(bag_body(r, bb, bt, hb, r.p(0.6)))
        if r.p(0.7):
            add.append(stroke(arc(0.0, 0.6, r.u(1.5, 3.5), r.u(1.8, 4.0), 0, 180, 10), r.u(0.9, 1.6)))
    return add, []


GENERATORS = [gen_tshirt, gen_trouser, gen_pullover, gen_dress, gen_coat, gen_sandal, gen_shirt, gen_sneaker, gen_bag, gen_boot]
TILT = [2.0, 1.5, 2.0, 1.5, 2.0, 2.5, 2.0, 2.5, 2.0, 2.5]      # small random rotation of the whole drawing, degrees
WOBBLE = 0.22                                                   # random displacement of every corner, pixels


def draw_class(k, count, seed=SEED):
    """count drawings of class k -> (count, 28, 28) grey pictures, and the handicap of each"""
    rng = np.random.default_rng([seed, k])
    r = Rnd(rng)
    out = np.zeros((count, 28, 28))
    handicap = np.zeros(count)
    for i in range(count):
        g = GENERATORS[k](r)
        add, sub = g[0], g[1]
        extra = g[2] if len(g) > 2 else {}
        handicap[i] = extra.get("handicap", 0.0)
        dim = extra.get("dim", [])
        add = [p + rng.normal(0.0, WOBBLE, p.shape) for p in add]
        ang = r.u(-TILT[k], TILT[k])
        add = rotate(add, ang); sub = rotate(list(sub), ang)
        dim = list(zip(rotate([p for p, _ in dim], ang), [ink for _, ink in dim]))
        out[i] = render(add, sub, rng, dim)
    return out, handicap


# ----------------------------------------------------------------------------------------------------------------------
# the network (same operations as the scorer's interpreter, re-implemented here so that this file needs nothing else)
# ----------------------------------------------------------------------------------------------------------------------
def op_conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def op_pool(x, size, stride, reduce):
    n, c, h, w = x.shape
    (sh, sw), (th, tw) = size, stride
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if reduce == "max" else win.mean(0)


def forward(layers, weights, images, want):
    """runs the layers on images (N,1,28,28) and returns the outputs named in want"""
    vals = {"image": images}
    for L in layers:
        ins = [vals[i] for i in L["inputs"]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = op_conv(ins[0], weights[name + ".W"], weights[name + ".b"], L.get("stride", 1), L.get("padding", 0))
        elif op == "dense":
            vals[name] = ins[0].reshape(len(ins[0]), -1) @ weights[name + ".W"].T + weights[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op in ("maxpool", "avgpool"):
            vals[name] = op_pool(ins[0], L["size"], L["stride"], op[:3])
        elif op == "add":
            vals[name] = sum(ins)
        elif op == "reshape":
            vals[name] = ins[0].reshape((len(ins[0]),) + tuple(L["shape"]))
        elif op == "norm":
            x = ins[0]
            mu = x.mean(axis=(2, 3), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
    return [vals[w] for w in want]


def front_end():
    """layers and weights that turn the image into (a) a blurred, standardised soft mask and (b) a standardised map of
    pooled edge directions.  All weights are formulas."""
    f32 = np.float32
    W = {}
    layers = []
    # soft mask from four soft-threshold units
    w1 = np.zeros((4, 1, 1, 1)); b1 = np.zeros(4)
    g = 1.0 / (T1 - T0)
    w1[0, 0, 0, 0], b1[0] = g, -T0 * g
    w1[1, 0, 0, 0], b1[1] = g, -T0 * g - 1.0
    w1[2, 0, 0, 0], b1[2] = 1.0 / T2, 0.0
    w1[3, 0, 0, 0], b1[3] = 1.0 / T2, -1.0
    W["pre1.W"], W["pre1.b"] = w1, b1
    w2 = np.array([W_LOW, -W_LOW, 1.0 - W_LOW, -(1.0 - W_LOW)]).reshape(1, 4, 1, 1)
    W["pre2.W"], W["pre2.b"] = w2, np.zeros(1)
    layers += [
        {"name": "pre1", "op": "conv", "inputs": ["image"], "out": 4, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "pre1r", "op": "relu", "inputs": ["pre1"]},
        {"name": "mask", "op": "conv", "inputs": ["pre1r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
    ]
    W["mask.W"], W["mask.b"] = W.pop("pre2.W"), W.pop("pre2.b")
    # (a) silhouette branch: Gaussian blur, standardise
    ax = np.arange(-2, 3)
    g1 = np.exp(-ax ** 2 / (2 * BLUR_SIGMA ** 2)); g1 /= g1.sum()
    W["blur.W"], W["blur.b"] = np.outer(g1, g1).reshape(1, 1, 5, 5), np.zeros(1)
    layers += [
        {"name": "blur", "op": "conv", "inputs": ["mask"], "out": 1, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "sil", "op": "norm", "inputs": ["blur"]},
    ]
    # (b) edge branch: derivative of a Gaussian in 8 directions, half-wave rectified, max-pooled, standardised together
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    G = np.exp(-(xx ** 2 + yy ** 2) / (2 * EDGE_SIGMA ** 2))
    we = np.zeros((8, 1, 5, 5))
    for k in range(8):
        th = k * math.pi / 4
        d = (xx * math.cos(th) + yy * math.sin(th)) * G
        d = d / np.sum(d * (xx * math.cos(th) + yy * math.sin(th)))      # unit response to a unit ramp
        we[k, 0] = d
    W["edge.W"], W["edge.b"] = we, np.zeros(8)
    n_pool = (28 - EDGE_POOL) // EDGE_STRIDE + 1
    layers += [
        {"name": "edge", "op": "conv", "inputs": ["mask"], "out": 8, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "edger", "op": "relu", "inputs": ["edge"]},
        {"name": "edgep", "op": "maxpool", "inputs": ["edger"], "size": [EDGE_POOL, EDGE_POOL], "stride": [EDGE_STRIDE, EDGE_STRIDE]},
        {"name": "edgef", "op": "reshape", "inputs": ["edgep"], "shape": [1, 8 * n_pool, n_pool]},
        {"name": "edg", "op": "norm", "inputs": ["edgef"]},
    ]
    return layers, {k: v.astype(f32) for k, v in W.items()}, 8 * n_pool * n_pool


def build(out_dir, k_per_class=K_PER_CLASS, verbose=True):
    f32 = np.float32
    layers, W, n_edge = front_end()
    N = 10 * k_per_class
    WA = np.zeros((N, 784), f32)
    WB = np.zeros((N, n_edge), f32)
    BA = np.zeros(N, f32)
    for k in range(10):
        pic, handicap = draw_class(k, k_per_class)                # each drawing: a mid-grey item on black
        BA[k * k_per_class:(k + 1) * k_per_class] = -handicap
        img = pic.astype(f32)[:, None, :, :]
        for s in range(0, k_per_class, 500):
            sil, edg = forward(layers, W, img[s:s + 500], ["sil", "edg"])
            # each drawing becomes its own template: its own standardised maps, nothing shared between drawings
            WA[k * k_per_class + s:k * k_per_class + s + len(sil)] = sil.reshape(len(sil), -1) / 784.0
            WB[k * k_per_class + s:k * k_per_class + s + len(sil)] = LAMBDA_EDGE * edg.reshape(len(edg), -1) / float(n_edge)
        if verbose:
            print("class", k, "drawn", flush=True)
    W["tsil.W"], W["tsil.b"] = WA, BA
    W["tedg.W"], W["tedg.b"] = WB, np.zeros(N, f32)
    layers += [
        {"name": "tsil", "op": "dense", "inputs": ["sil"], "out": N},
        {"name": "tedg", "op": "dense", "inputs": ["edg"], "out": N},
        {"name": "tsum", "op": "add", "inputs": ["tsil", "tedg"]},
        {"name": "tgrid", "op": "reshape", "inputs": ["tsum"], "shape": [1, 10, k_per_class]},
        {"name": "tbest", "op": "maxpool", "inputs": ["tgrid"], "size": [1, k_per_class], "stride": [1, k_per_class]},
        {"name": "tcls", "op": "reshape", "inputs": ["tbest"], "shape": [10]},
    ]
    # hand rule: strappy footwear.  Total variation of the mask (outline length, holes included) against its area.
    wd = np.zeros((4, 1, 3, 3))
    wd[0, 0, 1, 1], wd[0, 0, 1, 2] = -1.0, 1.0      # +d/dx
    wd[1, 0, 1, 1], wd[1, 0, 1, 2] = 1.0, -1.0      # -d/dx
    wd[2, 0, 1, 1], wd[2, 0, 2, 1] = -1.0, 1.0      # +d/dy
    wd[3, 0, 1, 1], wd[3, 0, 2, 1] = 1.0, -1.0      # -d/dy
    W["tv.W"], W["tv.b"] = wd.astype(f32), np.zeros(4, f32)
    # u = (TV - c*area)/s0 ; units carry (u+1)/2 and (u+1)/2 - 1 so that relu(a) - relu(b) = clip((u+1)/2, 0, 1)
    W["rtv.W"], W["rtv.b"] = np.full((2, 4 * 28 * 28), 0.5 / SPARSE_S0, f32), np.array([0.5, -0.5], f32)
    W["rar.W"], W["rar.b"] = np.full((2, 784), -0.5 * SPARSE_C / SPARSE_S0, f32), np.zeros(2, f32)
    wr = np.zeros((10, 2), f32); br = np.zeros(10, f32)
    wr[5] = [2 * SPARSE_GAIN, -2 * SPARSE_GAIN]; br[5] = -SPARSE_GAIN
    W["rule.W"], W["rule.b"] = wr, br
    layers += [
        {"name": "tv", "op": "conv", "inputs": ["mask"], "out": 4, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "tvr", "op": "relu", "inputs": ["tv"]},
        {"name": "rtv", "op": "dense", "inputs": ["tvr"], "out": 2},
        {"name": "rar", "op": "dense", "inputs": ["mask"], "out": 2},
        {"name": "rsum", "op": "add", "inputs": ["rtv", "rar"]},
        {"name": "rsumr", "op": "relu", "inputs": ["rsum"]},
        {"name": "rule", "op": "dense", "inputs": ["rsumr"], "out": 10},
    ]
    # hand rule: front opening.  Standardise the picture, look for a long vertical line (bright or dark) with a
    # [-1/2, 1, -1/2] profile averaged over LINE_ROWS rows, keep the strongest response in each group of 4 columns,
    # and compare the middle group (columns 12..15) with the two groups beside it (8..11 and 16..19).
    hl = LINE_ROWS // 2
    wl = np.zeros((2, 1, LINE_ROWS, LINE_ROWS))
    wl[0, 0, :, hl - 1], wl[0, 0, :, hl], wl[0, 0, :, hl + 1] = -0.5 / LINE_ROWS, 1.0 / LINE_ROWS, -0.5 / LINE_ROWS
    wl[1] = -wl[0]
    W["vl.W"], W["vl.b"] = wl.astype(f32), np.zeros(2, f32)
    rows = np.arange(10, 22)                                   # filter centres: the filter then spans rows 5..26, below the neck opening
    wf = np.zeros((2, 28, 7))
    wf[:, rows, 3] = 1.0 / len(rows)
    wf[:, rows, 2] = -0.5 / len(rows)
    wf[:, rows, 4] = -0.5 / len(rows)
    wf = wf.reshape(-1) / LINE_T1
    W["lin.W"], W["lin.b"] = np.stack([wf, wf]).astype(f32), np.array([-LINE_T0 / LINE_T1, -LINE_T0 / LINE_T1 - 1.0], f32)
    wq = np.zeros((10, 2), f32)
    wq[4] = [LINE_GAIN, -LINE_GAIN]; wq[6] = [LINE_GAIN, -LINE_GAIN]
    W["lrule.W"], W["lrule.b"] = wq, np.zeros(10, f32)
    layers += [
        {"name": "z", "op": "norm", "inputs": ["image"]},
        {"name": "vl", "op": "conv", "inputs": ["z"], "out": 2, "kernel": [LINE_ROWS, LINE_ROWS], "stride": 1, "padding": hl},
        {"name": "vlr", "op": "relu", "inputs": ["vl"]},
        {"name": "vlm", "op": "maxpool", "inputs": ["vlr"], "size": [1, 4], "stride": [1, 4]},
        {"name": "lin", "op": "dense", "inputs": ["vlm"], "out": 2},
        {"name": "linr", "op": "relu", "inputs": ["lin"]},
        {"name": "lrule", "op": "dense", "inputs": ["linr"], "out": 10},
        {"name": "scores", "op": "add", "inputs": ["tcls", "rule", "lrule"]},
    ]
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(f32) for k, v in W.items()})
    if verbose:
        print("wrote", out_dir, "weights:", sum(int(v.size) for v in W.values()))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
