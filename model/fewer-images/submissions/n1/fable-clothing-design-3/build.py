#!/usr/bin/env python3
"""Builds network.json and weights.npz for the clothing classifier.  Usage: python3 build.py OUT_DIR

Everything here comes from typed-in shape models (general knowledge of what T-shirts, trousers, shoes, bags ... look
like in a centred, trimmed product photo) and from formula filters.  No file is read.

Architecture (see method.md):
  image -> three hand-made maps:  soft silhouette (blurred, 28x28)
                                  silhouette with enclosed holes filled in (blurred, 14x14)
                                  bright parts only (blurred, 14x14)
        -> norm (each map standardised) -> dense layers whose rows are individual rendered pictures, each passed
           through the same three maps and normalised by itself alone  (output = weighted correlation)
        -> max over groups of templates of a class, then mean of the group maxima  -> class score
  plus a small hand-set branch: a centre-line detector (front opening of coats / shirts, absent in T-shirts / pullovers).
"""
import sys, os, json
import numpy as np

SEED = 20240611
N = 28                    # image side
SS = 4                    # supersampling of the rasteriser
W = N * SS

# ----------------------------------------------------------------------------------------------------------------
# hand-chosen feature constants
T0, T1 = 0.03, 0.13       # soft mask: 0 below T0 (background noise), 1 above T1 (even a pale item is "item")
BLUR_SIGMA = 0.8          # pixels
FILL_TAU = 1.0            # mask mass needed on each of the four sides for a pixel to count as enclosed
B0, B1 = 0.3, 0.6         # "bright parts" map: 0 below B0, 1 above B1.  Thin straps and handles cover only part
                          # of a pixel and come out mid-grey, so they fade here while solid dark-coloured items stay
W_RAW, W_FILL, W_BRIGHT = 0.4, 0.3, 0.3   # weights of the three correlations
GROUPS = 16               # groups of templates per class (mean of group maxima)
PER_GROUP = 615           # templates per group

# centre-line detector (front opening of coats, button placket of shirts)
CL_ROWS = 17              # the line has to run along this many rows
CL_KAPPA = 3.0            # the centre response must exceed this many times the mean response of the cloth beside it
CL_FLOOR = 0.02           # ... and this much contrast, so that a plain garment's pixel noise cannot fire
CL_SAT = 0.02             # margin at which the bonus is fully on
CL_BONUS = 0.03           # score taken from T-shirt and pullover when a centre line is clearly there

CLASSES = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]


# ----------------------------------------------------------------------------------------------------------------
# geometry helpers (all shapes are lists of polygons; x to the right, y downwards)
def P(*pts):
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def chaikin(p, n=2):
    """Corner cutting of a closed polygon: rounds corners the way cloth and leather do."""
    p = np.asarray(p, dtype=np.float64)
    for _ in range(n):
        q = np.roll(p, -1, axis=0)
        a = 0.75 * p + 0.25 * q
        b = 0.25 * p + 0.75 * q
        p = np.stack([a, b], axis=1).reshape(-1, 2)
    return p


def mirror(right):
    """right: points of the right half (x >= 0) from the top centre to the bottom centre. Returns the closed outline."""
    r = np.asarray(right, dtype=np.float64)
    l = r[::-1].copy()
    l[:, 0] *= -1
    return np.vstack([r, l])


def flipx(poly):
    q = np.asarray(poly, dtype=np.float64).copy()
    q[:, 0] *= -1
    return q


def ellipse(cx, cy, rx, ry, n=20, a0=0.0, a1=2 * np.pi):
    t = np.linspace(a0, a1, n, endpoint=(a1 - a0) < 2 * np.pi - 1e-9)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)


def stroke(pts, w, w_end=None):
    """Thick polyline as a list of quads plus octagons at the joints. w may taper linearly to w_end."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    ws = np.linspace(w, w if w_end is None else w_end, n)
    out = []
    for i in range(n - 1):
        a, b = pts[i], pts[i + 1]
        d = b - a
        L = np.hypot(d[0], d[1])
        if L < 1e-9:
            continue
        nrm = np.array([-d[1], d[0]]) / L
        out.append(np.array([a + nrm * ws[i] / 2, b + nrm * ws[i + 1] / 2, b - nrm * ws[i + 1] / 2, a - nrm * ws[i] / 2]))
    for i in range(n):
        out.append(ellipse(pts[i, 0], pts[i, 1], ws[i] / 2, ws[i] / 2, n=8))
    return out


def arc_pts(x0, x1, y_base, height, n=12, power=2.0):
    """Arch from (x0, y_base) up to y_base - height and down to (x1, y_base). power 2 = parabola, larger = boxier."""
    t = np.linspace(-1, 1, n)
    return np.stack([(x0 + x1) / 2 + t * (x1 - x0) / 2, y_base - height * (1 - np.abs(t) ** power)], axis=1)


def rot(polys, ang, c=(0.0, 0.0)):
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]])
    c = np.asarray(c, dtype=np.float64)
    return [(np.asarray(p) - c) @ R.T + c for p in polys]


# ----------------------------------------------------------------------------------------------------------------
# rasteriser: union of polygons by non-zero winding on a supersampled grid
def _wind(polys):
    x0s, y0s, x1s, y1s = [], [], [], []
    for p in polys:
        p = np.asarray(p, dtype=np.float64)
        q = np.roll(p, -1, axis=0)
        if np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]) < 0:      # same orientation for all
            p = p[::-1]
            q = np.roll(p, -1, axis=0)
        x0s.append(p[:, 0]); y0s.append(p[:, 1]); x1s.append(q[:, 0]); y1s.append(q[:, 1])
    x0 = np.concatenate(x0s) * SS; y0 = np.concatenate(y0s) * SS
    x1 = np.concatenate(x1s) * SS; y1 = np.concatenate(y1s) * SS
    ys = (np.arange(W) + 0.5)[:, None]
    up = (y0[None] <= ys) & (y1[None] > ys)
    dn = (y1[None] <= ys) & (y0[None] > ys)
    dy = y1 - y0
    dy = np.where(np.abs(dy) < 1e-12, 1.0, dy)
    xc = x0[None] + (ys - y0[None]) / dy[None] * (x1 - x0)[None]
    col = np.clip(np.ceil(xc - 0.5), 0, W).astype(np.int64)
    sgn = up.astype(np.int64) - dn.astype(np.int64)
    rows = np.broadcast_to(np.arange(W)[:, None], col.shape)
    acc = np.bincount((rows * (W + 1) + col).ravel(), weights=sgn.ravel().astype(np.float64), minlength=W * (W + 1))
    return np.cumsum(acc.reshape(W, W + 1), axis=1)[:, :W] != 0


def render(shape, rs, fit=True):
    """shape: dict with 'add' (list of polygons), optional 'sub' (holes) and 'add2' (drawn over the holes).
    The shape is trimmed to its bounding box, scaled so that the longer side is 28 pixels and centred, the way the
    product photos were prepared.  Returns the pixel coverage (28, 28) in [0, 1]."""
    add = [np.asarray(p, dtype=np.float64) for p in shape["add"]]
    sub = [np.asarray(p, dtype=np.float64) for p in shape.get("sub", [])]
    add2 = [np.asarray(p, dtype=np.float64) for p in shape.get("add2", [])]
    ang = shape.get("tilt", 0.0)
    if ang != 0.0:
        add, sub, add2 = rot(add, ang), rot(sub, ang), rot(add2, ang)
    allp = np.vstack(add + add2)
    lo, hi = allp.min(0), allp.max(0)
    size = hi - lo
    s = (N - shape.get("margin", 0.0)) / max(size[0], size[1])
    off = (N - size * s) / 2 - lo * s + shape.get("shift", np.zeros(2))
    tr = lambda ps: [p * s + off for p in ps]
    m = _wind(tr(add))
    if sub:
        m &= ~_wind(tr(sub))
    if add2:
        m |= _wind(tr(add2))
    lev = m.astype(np.float64)
    for polys, factor in shape.get("shade", []):             # parts that are paler than the rest (white soles ...)
        polys = [np.asarray(p, dtype=np.float64) for p in polys]
        if ang != 0.0:
            polys = rot(polys, ang)
        lev = np.where(_wind(tr(polys)), lev * factor, lev)
    return lev.reshape(N, SS, N, SS).mean(axis=(1, 3))


def U(rs, a, b):
    return a + (b - a) * rs.random_sample()


def jitter(rs, shape, tilt=0.03, margin=0.8, shift=0.4):
    shape["tilt"] = rs.normal() * tilt
    shape["margin"] = U(rs, 0.0, margin)
    shape["shift"] = np.array([U(rs, -shift, shift), U(rs, -shift, shift)])
    return shape


# ----------------------------------------------------------------------------------------------------------------
# upper-body garments.  Unit: garment length from the neck point (y = 0) to the hem (y = 1).
def top_shape(rs, bw, hem_w, nw, nd, sd, ap, sleeve, waist=1.0, hem_curve=0.0, collar=None, hood=None, vneck=None,
              strap=None, sleeve_bend=0.0):
    """bw: half width of the body at the armpit; hem_w: half width at the hem; nw: half width of the neck opening;
    nd: how far the neckline dips in the silhouette; sd: drop of the shoulder tip; ap: height of the armpit;
    sleeve: None or (theta from vertical, length, width at the root, width at the cuff)."""
    add, sub = [], []
    A = np.array([bw, ap])
    if strap is not None:
        # sleeveless: shoulder strap of width strap from the neck point, then an armhole curving down to the armpit
        sx = nw + strap
        right = [(0, nd), (nw * 0.55, nd * 0.85), (nw, 0), (sx, 0.005), (sx + (bw - sx) * 0.25, ap * 0.55), (bw, ap)]
    else:
        th, L, wr, wc = sleeve
        d = np.array([np.sin(th), np.cos(th)])
        n_in = np.array([-np.cos(th), np.sin(th)])
        Pt = A - wr * n_in                                   # point of the outer sleeve edge opposite the armpit
        sd = min(sd, Pt[1] - 0.01)
        t = (Pt[1] - sd) / max(np.cos(th), 0.15)
        S = Pt - t * d                                       # shoulder tip
        S[0] = max(S[0], nw + 0.03)
        co = S + L * d                                       # outer corner of the cuff
        ci = co + wc * n_in                                  # inner corner of the cuff
        if sleeve_bend != 0.0:                               # slight elbow: sleeves rarely hang perfectly straight
            mid_o = S + 0.5 * L * d - sleeve_bend * n_in
            mid_i = (A + ci) / 2 - sleeve_bend * n_in
            sl = P(S, mid_o, co, ci, mid_i, A)
        else:
            sl = P(S, co, ci, A)
        add += [sl, flipx(sl)]
        right = [(0, nd), (nw * 0.55, nd * 0.85), (nw, 0), (S[0], S[1]), (bw, ap)]
    right += [(bw * waist, 0.62), (hem_w, 1.0), (hem_w * 0.5, 1.0 + hem_curve * 0.8), (0, 1.0 + hem_curve)]
    add.append(mirror(right))
    if collar is not None:                                   # stand-up / shirt collar: a bump above the neck points
        cb, ct, ch = collar
        add.append(P((-cb, 0.03), (-ct, -ch), (ct, -ch), (cb, 0.03)))
    if hood is not None:                                     # hood lying flat above the shoulders
        hw, hh = hood
        add.append(ellipse(0, 0.02, hw, hh, n=16))
    if vneck is not None:                                    # open front / V neck seen as a notch
        vw, vd, top = vneck
        sub.append(P((-vw, top - 0.02), (vw, top - 0.02), (0, top + vd)))
    return {"add": add, "sub": sub}


# Product shots show garments on an invisible mannequin or hanger, so a body projects narrower than its flat-lay
# width and sleeves hang at the sides.  Classes that are easily confused share the same ranges for everything that
# does not tell them apart (body width, shoulder slope, armhole, sleeve width); they differ only in the cues named
# in the comments.
def _short_family(rs):
    return dict(bw=U(rs, 0.2, 0.35), sd=U(rs, 0.04, 0.14), ap=U(rs, 0.22, 0.34), waist=U(rs, 0.9, 1.0),
                th=np.radians(U(rs, 15, 75)), L=U(rs, 0.15, 0.4), wr=U(rs, 0.16, 0.26), taper=U(rs, 0.7, 1.0))


def _long_family(rs):
    return dict(bw=U(rs, 0.19, 0.35), sd=U(rs, 0.04, 0.14), ap=U(rs, 0.2, 0.34), waist=U(rs, 0.92, 1.02),
                wr=U(rs, 0.12, 0.24), taper=U(rs, 0.5, 0.85), bend=U(rs, -0.025, 0.025))


def _shirt_collar(rs):
    ct = U(rs, 0.05, 0.1)                                    # a shirt collar is narrower than a T-shirt neck opening
    return (ct + U(rs, 0.01, 0.04), ct, U(rs, 0.03, 0.09))


def gen_tshirt(rs):
    kind = rs.random_sample()
    nw = U(rs, 0.1, 0.2)                                     # wide round neck, no collar
    nd = U(rs, 0.0, 0.1)
    hc = U(rs, -0.01, 0.03)
    if kind < 0.62:        # classic short-sleeved tee
        g = _short_family(rs)
        return top_shape(rs, g["bw"], g["bw"] * U(rs, 0.92, 1.1), nw, nd, g["sd"], g["ap"],
                         (g["th"], g["L"], g["wr"], g["wr"] * g["taper"]), g["waist"], hc)
    bw = U(rs, 0.2, 0.35)
    hem = bw * U(rs, 0.92, 1.1)
    waist = U(rs, 0.9, 1.0)
    if kind < 0.78:        # sleeveless top / vest
        return top_shape(rs, bw * U(rs, 0.85, 1.0), hem, U(rs, 0.08, 0.18), U(rs, 0.03, 0.2), 0.0, U(rs, 0.24, 0.42), None,
                         waist, hc, strap=U(rs, 0.03, 0.11))
    if kind < 0.94:        # cap sleeves / wide boxy tops
        th = np.radians(U(rs, 45, 80)); L = U(rs, 0.05, 0.16); wr = U(rs, 0.15, 0.24); wc = wr * U(rs, 0.8, 1.0)
        return top_shape(rs, bw, hem, nw, nd, U(rs, 0.02, 0.08), U(rs, 0.2, 0.3), (th, L, wr, wc), waist, hc)
    # tops with longer sleeves
    th = np.radians(U(rs, 8, 35)); L = U(rs, 0.4, 0.9); wr = U(rs, 0.12, 0.2); wc = wr * U(rs, 0.55, 0.8)
    return top_shape(rs, bw, hem, nw, nd, U(rs, 0.04, 0.14), U(rs, 0.22, 0.32), (th, L, wr, wc), waist, hc)


def gen_shirt(rs):
    kind = rs.random_sample()
    hc = U(rs, 0.0, 0.06)                                    # shirt tail: the hem hangs lower in the middle
    collar = _shirt_collar(rs) if rs.random_sample() < 0.85 else None
    vneck = (U(rs, 0.015, 0.045), U(rs, 0.04, 0.14), -(collar[2] if collar else 0.0)) if rs.random_sample() < 0.3 else None
    nw = U(rs, 0.08, 0.14)
    nd = U(rs, 0.0, 0.04)
    if kind < 0.52:        # long-sleeved shirt or blouse: slimmer sleeves than knitwear or outerwear
        g = _long_family(rs)
        th = np.radians(U(rs, 3, 30)); L = U(rs, 0.65, 1.0); wr = g["wr"] * 0.85
        return top_shape(rs, g["bw"], g["bw"] * U(rs, 0.94, 1.1), nw, nd, g["sd"], g["ap"], (th, L, wr, wr * g["taper"]),
                         g["waist"], hc, collar, vneck=vneck, sleeve_bend=g["bend"])
    if kind < 0.76:        # short-sleeved shirt, polo
        g = _short_family(rs)
        return top_shape(rs, g["bw"], g["bw"] * U(rs, 0.92, 1.1), nw, nd, g["sd"], g["ap"],
                         (g["th"], g["L"], g["wr"], g["wr"] * g["taper"]), g["waist"], hc, collar, vneck=vneck)
    bw = U(rs, 0.2, 0.33)
    hem = bw * U(rs, 0.94, 1.12)
    waist = U(rs, 0.9, 1.0)
    if kind < 0.88:        # sleeveless blouse
        return top_shape(rs, bw, hem, nw, U(rs, 0.02, 0.15), 0.0, U(rs, 0.22, 0.38), None, waist, hc, collar, vneck=vneck,
                         strap=U(rs, 0.05, 0.13))
    # three-quarter or rolled sleeves
    th = np.radians(U(rs, 8, 40)); L = U(rs, 0.32, 0.62); wr = U(rs, 0.13, 0.21); wc = wr * U(rs, 0.6, 0.9)
    return top_shape(rs, bw, hem, nw, nd, U(rs, 0.04, 0.14), U(rs, 0.22, 0.33), (th, L, wr, wc), waist, hc, collar, vneck=vneck)


def gen_pullover(rs):
    g = _long_family(rs)
    bw = g["bw"] + 0.01
    hem = bw * U(rs, 0.82, 1.02)                             # ribbed hem pulls in
    nw = U(rs, 0.1, 0.2)                                     # round neck
    nd = U(rs, 0.0, 0.1)
    th = np.radians(abs(rs.normal()) * 10 + 2)               # sleeves hang down, often a little spread
    L = U(rs, 0.75, 1.1)                                     # cuffs reach the hem or hang below it
    k = rs.random_sample()
    collar = hood = vneck = None
    if k < 0.12:
        collar = (U(rs, 0.09, 0.14), U(rs, 0.08, 0.13), U(rs, 0.04, 0.11))      # roll neck
    elif k < 0.2:
        hood = (U(rs, 0.11, 0.17), U(rs, 0.08, 0.16))
    elif k < 0.3:
        vneck = (U(rs, 0.04, 0.09), U(rs, 0.06, 0.2), 0.0)
    return top_shape(rs, bw, hem, nw, nd, g["sd"], g["ap"], (th, L, g["wr"], g["wr"] * g["taper"]), g["waist"],
                     U(rs, -0.01, 0.02), collar, hood, vneck, sleeve_bend=g["bend"])


def gen_coat(rs):
    g = _long_family(rs)
    bw = g["bw"] - 0.01                                      # longer garment: the body is slimmer relative to its length
    hem = bw * U(rs, 0.95, 1.2)                              # straight or flared hem
    nw = U(rs, 0.08, 0.16)
    th = np.radians(abs(rs.normal()) * 7 + 2)                # sleeves hang close to the body
    L = U(rs, 0.58, 1.08)                                    # long coats: sleeves end well above the hem
    k = rs.random_sample()
    collar = hood = vneck = None
    if k < 0.45:
        collar = (U(rs, 0.09, 0.16), U(rs, 0.07, 0.14), U(rs, 0.03, 0.1))       # stand-up collar
    elif k < 0.7:
        hood = (U(rs, 0.11, 0.19), U(rs, 0.08, 0.18))
    if rs.random_sample() < 0.5:
        vneck = (U(rs, 0.02, 0.07), U(rs, 0.05, 0.22), -(collar[2] if collar else 0.0))   # lapels / open front
    return top_shape(rs, bw, hem, nw, U(rs, 0.0, 0.03), g["sd"], g["ap"], (th, L, g["wr"], g["wr"] * g["taper"]),
                     g["waist"], U(rs, -0.01, 0.02), collar, hood, vneck, sleeve_bend=g["bend"])


def gen_dress(rs):
    kind = rs.random_sample()
    bust = U(rs, 0.09, 0.17)                                 # half widths, in units of the dress length
    yb = U(rs, 0.16, 0.24)
    waist = bust * U(rs, 0.72, 0.98)
    yw = U(rs, 0.3, 0.44)
    hemw = max(waist * U(rs, 1.0, 1.3), U(rs, 0.1, 0.36))
    hip = waist + (hemw - waist) * U(rs, 0.35, 0.8)          # bell or A-line
    yh = U(rs, 0.52, 0.68)
    hc = U(rs, -0.01, 0.03)
    add, sub = [], []
    skirt = [(waist, yw), (hip, yh), (hemw, 1.0), (hemw * 0.5, 1.0 + hc * 0.8), (0, 1.0 + hc)]
    if kind < 0.42:        # straps or sleeveless
        nw = U(rs, 0.04, 0.11); strap = U(rs, 0.015, 0.07); nd = U(rs, 0.04, 0.2)
        sx = min(nw + strap, bust)
        right = [(0, nd), (nw * 0.6, nd * 0.8), (nw, 0), (sx, 0.005), (sx + (bust - sx) * 0.3, yb * 0.6), (bust, yb)] + skirt
        add.append(mirror(right))
    elif kind < 0.57:      # strapless / tube
        dip = U(rs, 0.0, 0.04)
        right = [(0, dip), (bust * 0.6, 0.0), (bust, 0.01), (bust, yb)] + skirt
        add.append(mirror(right))
    else:                  # with sleeves
        nw = U(rs, 0.05, 0.11); nd = U(rs, 0.0, 0.08)
        if kind < 0.8:     # short sleeves
            th = np.radians(U(rs, 25, 75)); L = U(rs, 0.05, 0.2); wr = U(rs, 0.09, 0.15); wc = wr * U(rs, 0.7, 1.0)
        else:              # long sleeves
            th = np.radians(U(rs, 3, 22)); L = U(rs, 0.35, 0.68); wr = U(rs, 0.08, 0.13); wc = wr * U(rs, 0.5, 0.85)
        ap = U(rs, 0.15, 0.23)
        d = np.array([np.sin(th), np.cos(th)]); n_in = np.array([-np.cos(th), np.sin(th)])
        A = np.array([bust, ap])
        Pt = A - wr * n_in
        sd = min(U(rs, 0.02, 0.06), Pt[1] - 0.01)
        S = Pt - (Pt[1] - sd) / max(np.cos(th), 0.15) * d
        S[0] = max(S[0], nw + 0.02)
        co = S + L * d; ci = co + wc * n_in
        sl = P(S, co, ci, A)
        add += [sl, flipx(sl)]
        right = [(0, nd), (nw * 0.6, nd * 0.8), (nw, 0), (S[0], S[1]), (bust, ap)] + skirt
        add.append(mirror(right))
    return {"add": add, "sub": sub}


def gen_trouser(rs):
    kind = rs.random_sample()
    a = U(rs, 0.14, 0.23)                                    # half width of the waist, unit = trouser length
    if kind < 0.1:         # folded in half / seen from the side: one tapering leg
        a = U(rs, 0.09, 0.15)
        b = a * U(rs, 0.5, 0.95)
        off = U(rs, -0.03, 0.03)
        right = P((-a, 0), (a, 0), (a * 1.05, 0.25), (b + off, 1.0), (-b + off, 1.0), (-a * 1.05, 0.25))
        return {"add": [right]}
    hip = a * U(rs, 1.0, 1.18)
    yh = U(rs, 0.14, 0.26)
    cy = U(rs, 0.24, 0.38)                                   # crotch
    lw = U(rs, 0.11, 0.21)                                   # leg width at the hem
    if kind < 0.3:
        xo = lw + U(rs, 0.0, 0.012)                          # legs together: at most a hairline between them
    else:
        xo = max(lw + 0.01, hip + U(rs, -0.06, 0.1))         # legs apart: an inverted V opens below the crotch
    knee = U(rs, 0.0, 0.03)                                  # slim legs curve in at the knee
    yk = 0.62
    xk_o = hip + (xo - hip) * (yk - yh) / (1 - yh) - knee
    xi = xo - lw                                             # inner hem
    wb = U(rs, 0.0, 0.015)
    right = [(0, wb), (a, 0), (hip, yh), (xk_o, yk), (xo, 1.0), (xi, 1.0), (xi * 0.45, cy + (1 - cy) * 0.45), (0, cy)]
    return {"add": [mirror(right)]}


# ----------------------------------------------------------------------------------------------------------------
# footwear, seen from the side with the toe to the left.  Unit: shoe length; yy is the height above the ground.
def _up(pts):
    q = P(*pts)
    q[:, 1] *= -1
    return q


def sneaker_outline(rs, hc=None, high=False):
    ht = U(rs, 0.1, 0.18)                                    # height of the toe box (sole included)
    if hc is None:
        hc = U(rs, 0.3, 0.48)                                # height at the ankle opening
    xt = U(rs, 0.5, 0.68)                                    # where the laces reach the opening
    xl = U(rs, 0.2, 0.4)                                     # where the laces start; in front of it the toe box is low
    lift = U(rs, 0.0, 0.07)
    hh = hc * U(rs, 0.85, 1.12)                              # heel tab
    dip = U(rs, 0.0, 0.07)
    ts = U(rs, 0.0, 0.05)                                    # toe spring
    pts = [(0.0, ts + 0.03), (0.015, ht * 0.8), (0.08, ht), (xl, ht + lift)]
    if high:                                                 # laces slope up, then the padded collar rises steeply
        xt = U(rs, 0.45, 0.62)
        pts += [(xt - U(rs, 0.04, 0.12), ht + lift + (hc - ht - lift) * U(rs, 0.3, 0.7))]
    pts += [(xt, hc), ((xt + 0.97) / 2, min(hc, hh) - dip), (0.97, hh), (1.0, hh * 0.6), (0.99, 0.04), (0.96, 0.0),
            (0.5, 0.0), (0.06, 0.0)]
    return chaikin(_up(pts), 2)


def gen_sneaker(rs):
    high = rs.random_sample() < 0.3                          # mid and high tops
    s = {"add": [sneaker_outline(rs, U(rs, 0.44, 0.66) if high else None, high)]}
    if rs.random_sample() < 0.4:                             # white / contrast sole: pale band along the bottom
        t = U(rs, 0.05, 0.11)
        s["shade"] = [([_up([(-0.1, -0.1), (-0.1, t), (1.1, t), (1.1, -0.1)])], U(rs, 0.15, 0.6))]
    return s


def gen_boot(rs):
    ht = U(rs, 0.13, 0.22)
    xs = U(rs, 0.34, 0.56)                                   # front of the shaft
    hs = 0.6 + 0.42 * np.sqrt(rs.random_sample())            # height of the shaft: mostly well above the ankle
    lean = U(rs, -0.05, 0.06)
    hi = ht + U(rs, 0.12, 0.3)                               # instep, where the shaft starts
    xb = U(rs, 0.9, 1.0)
    slant = U(rs, -0.1, 0.1)
    heel = U(rs, 0.0, 0.16) if rs.random_sample() < 0.6 else 0.0
    xh = U(rs, 0.66, 0.8)
    ts = U(rs, 0.0, 0.04)
    pts = [(0.0, ts + 0.03), (0.02, ht * 0.8), (0.11, ht), (xs - 0.1, ht + (hi - ht) * 0.45), (xs, hi),
           (xs + lean, hs), (xb, hs + slant), (min(1.0, xb + 0.02), hs * 0.5), (1.0, heel + 0.06), (0.99, 0.0)]
    if heel > 0.03:
        pts += [(xh, 0.0), (xh - 0.01, heel * U(rs, 0.6, 1.0)), (0.5, heel * 0.15), (0.38, 0.0)]
    else:
        pts += [(0.5, 0.0)]
    pts += [(0.06, 0.0)]
    return {"add": [chaikin(_up(pts), 2)]}


def gen_sandal(rs):
    kind = rs.random_sample()
    add, sub, add2 = [], [], []
    if kind < 0.3:         # flat strappy sandal / flip-flop
        t = U(rs, 0.035, 0.09)                               # sole thickness
        ts = U(rs, 0.0, 0.06)
        sole = _up([(0.0, ts + t / 2), (0.15, t / 2 + ts * 0.3), (0.4, t / 2), (1.0, t / 2 + U(rs, 0.0, 0.03))])
        add += stroke(sole, t)
        sw = U(rs, 0.025, 0.07)
        hstrap = U(rs, 0.1, 0.24)                            # toe strap arching over the forefoot
        x0 = U(rs, 0.08, 0.25); x1 = x0 + U(rs, 0.2, 0.4)
        add += stroke(arc_pts(x0, x1, -t / 2, hstrap, power=U(rs, 1.5, 3.0)), sw)
        if rs.random_sample() < 0.7:                         # heel strap / ankle strap
            ha = U(rs, 0.16, 0.4)
            xa = U(rs, 0.78, 0.95)
            add += stroke(_up([(xa + U(rs, -0.05, 0.05), t / 2), (xa, ha)]), sw)
            add += stroke(_up([(xa, ha), (xa - U(rs, 0.12, 0.3), ha + U(rs, -0.04, 0.04))]), sw)
            if rs.random_sample() < 0.5:
                add += stroke(_up([(xa - 0.2, ha), (x1 - 0.05, t)]), sw)
        if rs.random_sample() < 0.35:                        # heel cup
            add.append(_up([(0.86, t), (0.88, U(rs, 0.12, 0.25)), (1.0, U(rs, 0.12, 0.25)), (1.0, t)]))
    elif kind < 0.68:      # heeled sandal
        hh = U(rs, 0.2, 0.62)                                # heel height
        t = U(rs, 0.03, 0.07)
        xball = U(rs, 0.28, 0.42)
        plat = U(rs, 0.0, 0.08) if rs.random_sample() < 0.4 else 0.0
        sole_pts = [(0.0, plat + t / 2 + U(rs, 0.0, 0.03)), (xball, plat + t / 2), (0.75, plat + hh * 0.72), (0.99, plat + hh)]
        add += stroke(_up(sole_pts), t)
        if plat > 0.01:
            add.append(_up([(0.0, 0.0), (0.0, plat + t / 2), (xball, plat + t / 2), (xball + 0.05, 0.0)]))
        hk = rs.random_sample()
        if hk < 0.45:      # stiletto
            wt = U(rs, 0.08, 0.16); wb = U(rs, 0.02, 0.05); xh = U(rs, 0.88, 0.97)
            add.append(_up([(xh - wt, plat + hh * 0.9), (1.0, plat + hh), (xh + wb / 2, 0.0), (xh - wb / 2, 0.0)]))
        elif hk < 0.75:    # block heel
            wt = U(rs, 0.16, 0.28); wb = wt * U(rs, 0.6, 1.0)
            add.append(_up([(1.0 - wt, plat + hh * 0.85), (1.0, plat + hh), (1.0, 0.0), (1.0 - wb, 0.0)]))
        else:              # wedge
            add.append(_up([(xball * U(rs, 0.3, 1.0), 0.0), (xball, plat + t / 2), (0.75, plat + hh * 0.72),
                            (1.0, plat + hh), (1.0, 0.0)]))
        sw = U(rs, 0.025, 0.08)
        x0 = U(rs, 0.05, 0.2); x1 = x0 + U(rs, 0.18, 0.35)
        add += stroke(arc_pts(x0, x1, -(plat + t / 2), U(rs, 0.1, 0.22), power=U(rs, 1.5, 3.0)), sw)
        if rs.random_sample() < 0.75:                        # ankle strap above the heel
            ha = plat + hh + U(rs, 0.12, 0.34)
            xa = U(rs, 0.84, 0.97)
            add += stroke(_up([(xa, plat + hh), (xa, ha)]), sw)
            add += stroke(_up([(xa, ha), (xa - U(rs, 0.15, 0.32), ha - U(rs, -0.03, 0.1))]), sw)
            if rs.random_sample() < 0.5:
                add += stroke(_up([(xa - 0.22, ha - 0.03), (x1, plat + t + 0.08)]), sw)
        if rs.random_sample() < 0.4:                         # closed heel counter
            add.append(_up([(0.8, plat + hh * 0.8), (0.84, plat + hh + U(rs, 0.1, 0.2)), (1.0, plat + hh + U(rs, 0.1, 0.2)), (1.0, plat + hh)]))
    else:                  # sporty / trekking / fisherman sandal: thick sole, broad straps, open toe
        t = U(rs, 0.06, 0.14)                                # sole thickness at the toe
        th = t * U(rs, 1.0, 2.0)                             # ... and at the heel
        ts = U(rs, 0.0, 0.05)
        add.append(chaikin(_up([(0.0, ts), (0.0, ts + t), (0.5, t), (1.0, th), (1.0, 0.0), (0.4, 0.0)]), 1))
        hc = U(rs, 0.24, 0.46)                               # height of the upper at the instep
        xt = U(rs, 0.5, 0.72)
        x0 = U(rs, 0.08, 0.3)                                # the toes stay uncovered in front of this
        hh = hc * U(rs, 0.7, 1.15)
        top = lambda x: np.interp(x, [x0, xt, 0.97], [t + 0.3 * (hc - t), hc, hh])
        upper = [(x0, t * 0.5)] + [(x, float(top(x))) for x in np.linspace(x0, 0.97, 7)] + [(1.0, hh * 0.6), (1.0, th * 0.5)]
        add.append(_up(upper))
        shade = [([_up([(x0 - 0.02, t)] + upper[1:-1] + [(1.02, th)])], U(rs, 0.3, 0.8))]   # webbing, not solid leather
        nh = rs.randint(1, 4)                                # gaps between the straps
        edges = np.sort(x0 + (0.95 - x0) * rs.random_sample(2 * nh))
        for i in range(nh):
            xa, xb = edges[2 * i], edges[2 * i + 1]
            if xb - xa < 0.04:
                continue
            ya = float(top(xa)) * U(rs, 0.55, 1.2); yb = float(top(xb)) * U(rs, 0.55, 1.2)
            lean = U(rs, -0.04, 0.06)
            sub.append(_up([(xa, t + (th - t) * xa), (xa + lean, ya), (xb + lean, yb), (xb, t + (th - t) * xb)]))
        return {"add": add, "sub": sub, "shade": shade}
    return {"add": add, "sub": sub, "add2": add2}


# ----------------------------------------------------------------------------------------------------------------
# bags.  Unit: body width.
def gen_bag(rs):
    kind = rs.random_sample()
    add, sub, add2 = [], [], []
    hb = U(rs, 0.5, 1.15)                                    # body height
    if kind < 0.18:
        hb = U(rs, 0.4, 0.75)                                # clutch / purse: wide, no handle
    top = U(rs, 0.72, 1.06)                                  # top width relative to the bottom
    bulge = U(rs, 0.0, 0.05)
    r = U(rs, 0.0, 0.12)                                     # corner rounding
    body = P((-0.5 * top, 0), (0.5 * top, 0), (0.5 * (top + 1) / 2 + bulge, hb * 0.5), (0.5, hb - r), (0.5 - r, hb),
             (-0.5 + r, hb), (-0.5, hb - r), (-0.5 * (top + 1) / 2 - bulge, hb * 0.5))
    body = chaikin(body, 1 if rs.random_sample() < 0.5 else 2)
    add.append(body)
    if 0.18 <= kind < 0.72:      # one or two top handles
        hx = U(rs, 0.14, 0.42) * top
        hh = U(rs, 0.15, 0.75)
        w = U(rs, 0.025, 0.07)
        pw = U(rs, 1.6, 4.0)
        add += stroke(arc_pts(-hx, hx, 0.03, hh, n=14, power=pw), w)
        if rs.random_sample() < 0.3:
            add += stroke(arc_pts(-hx * 0.8, hx * 0.8, 0.03, hh * U(rs, 0.6, 0.9), n=14, power=pw), w)
    elif 0.72 <= kind < 0.88:    # long shoulder strap
        hx = U(rs, 0.3, 0.5) * top
        hh = U(rs, 0.5, 1.3)
        w = U(rs, 0.02, 0.05)
        if rs.random_sample() < 0.5:
            add += stroke(P((-hx, 0.03), (U(rs, -0.1, 0.1), -hh), (hx, 0.03)), w)
        else:
            add += stroke(arc_pts(-hx, hx, 0.03, hh, n=14, power=U(rs, 1.5, 2.5)), w)
    elif kind >= 0.88:           # backpack / bucket: taller body, small loop on top
        add = [p * np.array([U(rs, 0.65, 0.9), 1.25]) for p in add]
        add += stroke(arc_pts(-0.1, 0.1, 0.03, U(rs, 0.08, 0.2), n=8), U(rs, 0.03, 0.06))
    return {"add": add, "sub": sub, "add2": add2}


GENERATORS = {"tshirt": gen_tshirt, "trouser": gen_trouser, "pullover": gen_pullover, "dress": gen_dress, "coat": gen_coat,
              "sandal": gen_sandal, "shirt": gen_shirt, "sneaker": gen_sneaker, "bag": gen_bag, "boot": gen_boot}


# ----------------------------------------------------------------------------------------------------------------
# features (the same arithmetic as the first layers of the network)
def gauss_kernel(sigma, k=5):
    r = np.arange(k) - k // 2
    g = np.exp(-0.5 * (r / sigma) ** 2)
    g /= g.sum()
    return np.outer(g, g)


def conv_same(x, ker):
    k = ker.shape[0]
    p = k // 2
    xp = np.pad(x, p)
    out = np.zeros_like(x, dtype=np.float64)
    for dy in range(k):
        for dx in range(k):
            out += ker[dy, dx] * xp[dy:dy + x.shape[0], dx:dx + x.shape[1]]
    return out


def soft_mask(img):
    return np.clip((img - T0) / (T1 - T0), 0.0, 1.0)


def fill_mask(m):
    """Fills what the item encloses: a pixel with item to its left, right, above and below counts as item.
    Pale or patterned items keep their outline but lose their inside; this puts the inside back."""
    c = np.cumsum(m, axis=1); left = c - m; right = c[:, -1:] - c
    r = np.cumsum(m, axis=0); up = r - m; down = r[-1:, :] - r
    q = sum(np.maximum(1.0 - d / FILL_TAU, 0.0) for d in (left, right, up, down))
    e = np.maximum(1.0 - q, 0.0)
    return e + np.maximum(m - e, 0.0)


def features(img):
    """Returns the three standardised feature maps of a picture: blurred mask (28x28), blurred filled mask (14x14)
    and blurred bright-parts map (14x14)."""
    m = soft_mask(img)
    g = gauss_kernel(BLUR_SIGMA)
    half = lambda z: z.reshape(N // 2, 2, N // 2, 2).mean(axis=(1, 3))
    a = conv_same(m, g)
    b = half(conv_same(fill_mask(m), g))
    c = half(conv_same(np.clip((img - B0) / (B1 - B0), 0.0, 1.0), g))
    return standardise(a), standardise(b), standardise(c)


def standardise(f):
    f = f - f.mean()
    return f / np.sqrt((f ** 2).mean() + 1e-5)


def make_image(rs, cls):
    """One synthetic picture: a flat-shaded silhouette of random brightness."""
    shape = jitter(rs, GENERATORS[cls](rs))
    cov = render(shape, rs)
    bright = U(rs, 0.25, 1.0)
    return cov * bright


def build_templates(n_per_class, seed=SEED):
    TA = np.zeros((len(CLASSES), n_per_class, N * N), dtype=np.float32)
    TB = np.zeros((len(CLASSES), n_per_class, N * N // 4), dtype=np.float32)
    TC = np.zeros((len(CLASSES), n_per_class, N * N // 4), dtype=np.float32)
    for c, cls in enumerate(CLASSES):
        rs = np.random.RandomState(seed + 1000 * c)
        for i in range(n_per_class):
            a, b, d = features(make_image(rs, cls))          # each picture is its own template, normalised by itself
            TA[c, i] = W_RAW * a.ravel() / a.size
            TB[c, i] = W_FILL * b.ravel() / b.size
            TC[c, i] = W_BRIGHT * d.ravel() / d.size
    return TA, TB, TC


# ----------------------------------------------------------------------------------------------------------------
def build(out_dir):
    npc = GROUPS * PER_GROUP
    TA, TB, TC = build_templates(npc)
    layers, weights = [], {}

    def conv(name, inp, Wt, b, pad=0):
        Wt = np.asarray(Wt, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
        layers.append({"name": name, "op": "conv", "inputs": [inp], "out": int(Wt.shape[0]), "kernel": [int(Wt.shape[2]), int(Wt.shape[3])],
                       "stride": 1, "padding": int(pad)})
        weights[name + ".W"], weights[name + ".b"] = Wt, b

    def dense(name, inp, Wt, b):
        Wt = np.asarray(Wt, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
        layers.append({"name": name, "op": "dense", "inputs": [inp], "out": int(Wt.shape[0])})
        weights[name + ".W"], weights[name + ".b"] = Wt, b

    def op(name, kind, inp, **kw):
        layers.append({"name": name, "op": kind, "inputs": inp if isinstance(inp, list) else [inp], **kw})

    # soft mask m = 1 - relu(1 - relu((x - T0) / (T1 - T0)))
    conv("ramp", "image", np.full((1, 1, 1, 1), 1.0 / (T1 - T0)), [-T0 / (T1 - T0)])
    op("ramp_relu", "relu", "ramp")
    conv("inv", "ramp_relu", np.full((1, 1, 1, 1), -1.0), [1.0])
    op("inv_relu", "relu", "inv")
    conv("mask", "inv_relu", np.full((1, 1, 1, 1), -1.0), [1.0])
    conv("blur", "mask", gauss_kernel(BLUR_SIGMA)[None, None], [0.0], pad=2)
    op("blur_norm", "norm", "blur")
    # enclosure fill: saturating sums of the mask to the left / right / above / below of every pixel
    # (a dense layer, because a convolution spanning the whole image would be very slow in the reference code)
    D = np.zeros((4, N, N, N, N))
    for i in range(N):
        for j in range(N):
            D[0, i, j, i, :j] = 1.0 / FILL_TAU
            D[1, i, j, i, j + 1:] = 1.0 / FILL_TAU
            D[2, i, j, :i, j] = 1.0 / FILL_TAU
            D[3, i, j, i + 1:, j] = 1.0 / FILL_TAU
    dense("sides_flat", "mask", D.reshape(4 * N * N, N * N), np.zeros(4 * N * N))
    op("sides", "reshape", "sides_flat", shape=[4, N, N])
    conv("sides_inv", "sides", -np.eye(4).reshape(4, 4, 1, 1), np.ones(4))          # 1 - sum / tau
    op("sides_lack", "relu", "sides_inv")                                           # what each side lacks
    conv("enc", "sides_lack", -np.ones((1, 4, 1, 1)), [1.0])
    op("enclosed", "relu", "enc")                                                   # 1 only if all four sides are there
    conv("enclosed_neg", "enclosed", np.full((1, 1, 1, 1), -1.0), [0.0])
    op("mask_extra", "add", ["mask", "enclosed_neg"])
    op("mask_extra_relu", "relu", "mask_extra")
    op("filled", "add", ["enclosed", "mask_extra_relu"])                            # max(mask, enclosed)
    conv("fblur", "filled", gauss_kernel(BLUR_SIGMA)[None, None], [0.0], pad=2)
    op("fpool", "avgpool", "fblur", size=[2, 2], stride=[2, 2])
    op("fill_norm", "norm", "fpool")
    # bright parts: 1 - relu(1 - relu((x - B0) / (B1 - B0)))
    conv("bramp", "image", np.full((1, 1, 1, 1), 1.0 / (B1 - B0)), [-B0 / (B1 - B0)])
    op("bramp_relu", "relu", "bramp")
    conv("binv", "bramp_relu", np.full((1, 1, 1, 1), -1.0), [1.0])
    op("binv_relu", "relu", "binv")
    conv("bright", "binv_relu", np.full((1, 1, 1, 1), -1.0), [1.0])
    conv("bblur", "bright", gauss_kernel(BLUR_SIGMA)[None, None], [0.0], pad=2)
    op("bpool", "avgpool", "bblur", size=[2, 2], stride=[2, 2])
    op("bright_norm", "norm", "bpool")
    # template bank: correlation of the standardised maps with each rendered picture
    nt = TA.shape[0] * TA.shape[1]
    dense("templates_raw", "blur_norm", TA.reshape(nt, -1), np.zeros(nt))
    dense("templates_fill", "fill_norm", TB.reshape(nt, -1), np.zeros(nt))
    dense("templates_bright", "bright_norm", TC.reshape(nt, -1), np.zeros(nt))
    op("templates", "add", ["templates_raw", "templates_fill", "templates_bright"])
    op("bank", "reshape", "templates", shape=[1, len(CLASSES) * GROUPS, PER_GROUP])
    op("group_max", "maxpool", "bank", size=[1, PER_GROUP], stride=[1, PER_GROUP])
    op("groups", "reshape", "group_max", shape=[1, len(CLASSES), GROUPS])
    op("class_mean", "avgpool", "groups", size=[1, GROUPS], stride=[1, GROUPS])
    op("scores", "reshape", "class_mean", shape=[len(CLASSES)])

    # centre-line branch: mean over CL_ROWS rows of a horizontal second difference (1 and 2 pixel wide lines),
    # both polarities (the opening may be darker or lighter than the cloth), compared with the same response
    # a few columns to either side, which measures how busy the cloth itself is (prints, checks, stripes).
    k = CL_ROWS
    ker = {0: (-1, np.array([-0.5, 1.0, -0.5]) / k), 1: (-1, np.array([-0.5, 0.5, 0.5, -0.5]) / k)}
    side = (9, 10, 11, 16, 17, 18)
    rows = (11, 14, 17)                                      # centres of the row windows: rows 3..19, 6..22, 9..25
    cols = tuple(range(9, 19))
    unit = {}
    Wc = []
    for f in (0, 1):
        for r in rows:
            for cc in cols:
                w = np.zeros((N, N))
                off, kk = ker[f]
                w[r - k // 2:r + k // 2 + 1, cc + off:cc + off + len(kk)] = kk[None, :]
                for sg in (0, 1):                            # the two polarities, rectified separately
                    unit[(f, sg, r, cc)] = len(Wc)
                    Wc.append((w if sg == 0 else -w).ravel())
    dense("cl", "image", np.array(Wc), np.zeros(len(Wc)))
    op("cl_abs", "relu", "cl")
    picks = [(f, r, cc) for f in (0, 1) for r in rows for cc in ((12, 13, 14, 15) if f == 0 else (12, 13, 14))]
    Wp = np.zeros((2 * len(picks), len(Wc)))
    for j, (f, r, cc) in enumerate(picks):
        for sg in (0, 1):
            Wp[2 * j + sg, unit[(f, sg, r, cc)]] = 1.0
            for c2 in side:
                for s2 in (0, 1):
                    Wp[2 * j + sg, unit[(f, s2, r, c2)]] = -CL_KAPPA / len(side)
    dense("cl_pick", "cl_abs", Wp, np.full(2 * len(picks), -CL_FLOOR))
    op("cl_relu", "relu", "cl_pick")
    op("cl_row", "reshape", "cl_relu", shape=[1, 1, 2 * len(picks)])
    op("cl_max", "maxpool", "cl_row", size=[1, 2 * len(picks)], stride=[1, 2 * len(picks)])
    dense("cl_cap", "cl_max", [[-1.0]], [CL_SAT])            # CL_SAT - s
    op("cl_cap_relu", "relu", "cl_cap")
    # A front opening speaks against T-shirt and pullover (and so, among tops, for coat and shirt).  It is written as
    # a penalty on those two rather than a bonus for the others, so that things which merely have something vertical
    # in the middle (the gap between trouser legs, a boot shaft) are not pushed towards coat or shirt.
    gain = np.zeros((len(CLASSES), 1))
    gain[CLASSES.index("tshirt"), 0] = -CL_BONUS
    gain[CLASSES.index("pullover"), 0] = -CL_BONUS
    dense("cl_bonus", "cl_cap_relu", -gain / CL_SAT, gain[:, 0])   # gain * min(s, CL_SAT) / CL_SAT
    op("out", "add", ["scores", "cl_bonus"])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
