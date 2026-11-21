#!/usr/bin/env python3
"""Hand-designed clothing classifier: writes network.json and weights.npz.

No data is read.  Every weight comes from this file:
  * a fixed intensity curve + Gaussian blur (formulas),
  * silhouette templates DRAWN by the parametric shape models below (typed from general
    knowledge of what T-shirts, trousers, shoes, bags ... look like in a shop photo that is
    trimmed to the item, scaled so its longest side is 28 px and centred).  Every drawing
    becomes one row of a dense layer, normalised by itself alone (zero mean, unit norm),
  * a few hand-chosen thresholds and rule tables: a morphological "thin / strappy" cue for sandals and a
    "front opening" (zip / button line) cue for coats and shirts versus pullovers and T-shirts.
Usage: python3 build.py OUT_DIR
"""
import json
import math
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------------------
# hand-chosen settings
# ----------------------------------------------------------------------------------------
N_PER_CLASS = 10000         # drawings per class
SEED = 20261120
BLUR_SIGMA = 1.0            # px, applied to image feature and to drawings alike
NOMINAL_INK = 0.6           # grey level a drawn item is given before the intensity curve
SS = 4                      # supersampling of the rasteriser
G = 28 * SS
YC = np.arange(G) + 0.5

# intensity curve  F(x) = sum_i w_i * clamp((x - lo_i) / (hi_i - lo_i), 0, 1)   (concave: mostly "is there ink")
CURVE = [(0.4, 0.02, 0.10), (0.3, 0.0, 0.25), (0.3, 0.0, 0.70)]
# hard-ish ink mask used by the morphological cue
MASK_LO, MASK_HI = 0.04, 0.08   # steep: a dim but solid item must still count as solid


def curve(x):
    out = np.zeros_like(x)
    for w, lo, hi in CURVE:
        out += w * np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return out


def gauss_kernel(sigma, radius=2):
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-ax * ax / (2 * sigma * sigma))
    k /= k.sum()
    return np.outer(k, k)


def blur_batch(T, K):
    """Same arithmetic as a 5x5 conv with zero padding 2.  T: (n, 28, 28)."""
    r = K.shape[0] // 2
    P = np.pad(T, ((0, 0), (r, r), (r, r)))
    out = np.zeros_like(T)
    for dy in range(K.shape[0]):
        for dx in range(K.shape[1]):
            out += K[dy, dx] * P[:, dy:dy + 28, dx:dx + 28]
    return out


# ----------------------------------------------------------------------------------------
# tiny vector drawing kit
# ----------------------------------------------------------------------------------------
def fill(mask, V):
    """Even-odd scanline fill of polygon V (n, 2) given in hi-res pixel coordinates (x, y)."""
    x1 = V[:, 0]; y1 = V[:, 1]
    x2 = np.roll(x1, -1); y2 = np.roll(y1, -1)
    cross = (y1[:, None] <= YC) != (y2[:, None] <= YC)
    e, rr = np.nonzero(cross)
    if len(e) == 0:
        return
    t = (YC[rr] - y1[e]) / (y2[e] - y1[e])
    xi = x1[e] + t * (x2[e] - x1[e])
    ci = np.clip(np.ceil(xi - 0.5), 0, G).astype(np.int64)
    tog = np.bincount(rr * (G + 1) + ci, minlength=G * (G + 1)).reshape(G, G + 1)
    mask |= (np.cumsum(tog, axis=1)[:, :G] & 1).astype(bool)


def raster(polys):
    """Trim to the drawing, scale its longest side to 28 px, centre it; returns (28, 28) coverage in [0, 1]."""
    polys = [np.asarray(p, dtype=np.float64) for p in polys]
    allv = np.concatenate(polys)
    lo = allv.min(0); hi = allv.max(0)
    s = G / float((hi - lo).max())
    off = G / 2.0 - s * (lo + hi) / 2.0
    mask = np.zeros((G, G), dtype=bool)
    for V in polys:
        fill(mask, V * s + off)
    return mask.reshape(28, SS, 28, SS).mean(axis=(1, 3))


def sym(R):
    """Mirror a right-half outline (top centre -> bottom centre) into a closed polygon."""
    R = [(float(x), float(y)) for x, y in R]
    L = [(-x, y) for x, y in reversed(R) if x > 1e-9]
    return np.array(R + L)


def mirror(P):
    P = np.asarray(P, dtype=np.float64).copy()
    P[:, 0] = -P[:, 0]
    return P[::-1]


def rnd(P, rad, n=3):
    """Round the corners of a closed polygon with radius-like cut `rad`."""
    P = np.asarray(P, dtype=np.float64); m = len(P); out = []
    for i in range(m):
        p0, p1, p2 = P[i - 1], P[i], P[(i + 1) % m]
        d0 = p0 - p1; d2 = p2 - p1
        l0 = math.hypot(d0[0], d0[1]); l2 = math.hypot(d2[0], d2[1])
        c = min(rad, 0.45 * l0, 0.45 * l2)
        if c < 1e-6:
            out.append(p1); continue
        a = p1 + d0 / l0 * c; b = p1 + d2 / l2 * c
        for k in range(n + 1):
            t = k / n
            out.append((1 - t) ** 2 * a + 2 * (1 - t) * t * p1 + t * t * b)
    return np.array(out)


def qbez(p0, p1, p2, n=6):
    """Quadratic curve from p0 to p2 with control p1; returns n points, excluding p0."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    return [tuple((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2) for t in (np.arange(1, n + 1) / n)]


def stroke(P, w):
    """Thick polyline as a list of quads."""
    P = np.asarray(P, dtype=np.float64); out = []
    for a, b in zip(P[:-1], P[1:]):
        d = b - a; L = math.hypot(d[0], d[1])
        if L < 1e-9:
            continue
        d = d / L; nrm = np.array([-d[1], d[0]]) * (w / 2.0)
        a2 = a - d * (w * 0.3); b2 = b + d * (w * 0.3)
        out.append(np.array([a2 + nrm, b2 + nrm, b2 - nrm, a2 - nrm]))
    return out


def arc(cx, cy, rx, ry, a0, a1, n=10):
    """Points on an ellipse; angles in degrees, 0 = +x, 90 = up (towards smaller y)."""
    t = np.radians(np.linspace(a0, a1, n + 1))
    return np.stack([cx + rx * np.cos(t), cy - ry * np.sin(t)], axis=1)


def U(lo, hi, r):
    return lo + (hi - lo) * float(r)


# ----------------------------------------------------------------------------------------
# upper-body garments (front view, height ~1, x centred on 0, y down)
# ----------------------------------------------------------------------------------------
def sleeve_polys(hs, ys, hb, ya, ang, ls, w_end, neck=None, rq=0.0):
    """Sleeve from shoulder tip (hs, ys); `ang` degrees away from straight down; armpit at (hb, ya).

    With `neck` (the point where the shoulder line starts) and rq > 0 the shoulder corner is rounded: the outline
    leaves the shoulder line rq before the tip and joins the sleeve edge rq after it.  Returns (polys, Q) where Q
    is the point on the shoulder line at which the torso outline should turn down to the armpit.
    """
    b = math.radians(ang)
    A = np.array([hs, ys]); dirv = np.array([math.sin(b), math.cos(b)])
    B = A + ls * dirv
    C = B + w_end * np.array([-math.cos(b), math.sin(b)])
    D = np.array([hb, ya])
    E = np.array([max(hb - 0.05, 0.02), ys + 0.05])
    if neck is not None and rq > 1e-6:
        v = np.asarray(neck, dtype=np.float64) - A; L = math.hypot(v[0], v[1])
        q = min(rq, 0.8 * L)
        Q = A + v / L * q
        right = np.array([Q] + [np.array(p) for p in qbez(Q, A, A + dirv * q, 5)] + [B, C, D, E])
    else:
        Q = A
        right = np.array([A, B, C, D, E])
    return [right, mirror(right)], (float(Q[0]), float(Q[1]))


def torso(top, hs, ys, hb, ya, fw, fh, yb, tail=0.0, armhole=None):
    R = list(top) + [(hs, ys)]
    if armhole:
        R += armhole
    R += [(hb, ya), (hb * fw, ya + 0.5 * (yb - ya)), (hb * fh, yb - tail)]
    if tail > 1e-6:
        R += [(hb * fh * 0.6, yb - 0.2 * tail)]
    R += [(0.0, yb)]
    return sym(R)


def neck_crew(hn, dn):
    return [(0.0, dn), (hn * 0.6, dn * 0.35), (hn, 0.0)]


def neck_stand(hc, ch):
    return [(0.0, 0.0), (hc, 0.0), (hc + 0.015, ch)]


def neck_collar(hn, dv, ch):
    return [(0.0, dv), (hn * 0.45, 0.0), (hn, 0.0), (hn + 0.02, ch)]


def neck_hood(hh, ht):
    pts = arc(0.0, ht, hh, ht, 90, 0, 6)
    return [tuple(p) for p in pts]


def scoop_top(si, so, dn):
    """Scoop neckline between straps: centre (0, dn) up to strap inner top (si, 0), strap top to (so, 0)."""
    return [(0.0, dn)] + qbez((0.0, dn), (si, dn), (si, 0.0), 6) + [(so, 0.0)]


def tee(r, sleeve_len=(0.10, 0.24), body=(0.24, 0.36), collar=False):
    hb = U(body[0], body[1], r[0])
    ys = U(0.03, 0.10, r[4])
    if collar:
        ch = U(0.02, 0.05, r[2])
        top = neck_collar(U(0.07, 0.12, r[1]), U(0.0, 0.035, r[13]), ch)
        ys = ch + U(0.03, 0.08, r[4])
    else:
        top = neck_crew(U(0.09, 0.16, r[1]), U(0.0, 0.05, r[2]))
    hs = hb + U(-0.01, 0.05, r[3])
    ya = ys + U(0.17, 0.27, r[8])
    polys = [torso(top, hs, ys, hb, ya, U(0.90, 1.02, r[9]), U(0.92, 1.08, r[10]), 1.0, tail=U(0.0, 0.03, r[12]))]
    polys += sleeve_polys(hs, ys, hb, ya, U(30, 70, r[5]), U(sleeve_len[0], sleeve_len[1], r[6]), U(0.10, 0.16, r[7]))[0]
    return polys


def tank(r, body=(0.20, 0.30), strap_w=(0.03, 0.08), strap_x=(0.12, 0.22), flare=(0.95, 1.15)):
    hb = U(body[0], body[1], r[0])
    so = min(U(strap_x[0], strap_x[1], r[1]), hb - 0.01)
    si = max(so - U(strap_w[0], strap_w[1], r[7]), 0.03)
    dn = U(0.10, 0.30, r[2])
    ya = U(0.22, 0.38, r[8])
    top = scoop_top(si, so, dn)
    arm = qbez((so, 0.0), (so, ya), (hb, ya), 6)[:-1]
    R = top + arm + [(hb, ya), (hb * U(0.90, 1.0, r[9]), 0.62), (hb * U(flare[0], flare[1], r[10]), 1.0), (0.0, 1.0)]
    return [sym(R)]


def long_top(r, kind):
    """Pullover / coat / shirt: torso plus long sleeves hanging beside it.

    Everything that does not depend on the kind of garment is drawn from the SAME ranges for all three: the
    sleeve angle (including sleeves lying flush against the body - a shaded sleeve gap is no gap for a
    silhouette), the upper sleeve width, and the shoulder line (how far it drops from the neck and how round the
    shoulder is: that is the wearer's or mannequin's shoulder, not the garment's).  The classes differ in what
    clothing knowledge says they differ in: neckline / collar, proportions (a ~68 cm pullover fills the frame
    widest, a ~76 cm shirt less, a long coat least), cuff width, where the cuffs end relative to the hem, and
    the hem line (shirt tails, flared coats).
    """
    ang = U(0, 12, r[5]); wu = U(0.11, 0.18, r[3])
    sd = U(0.05, 0.13, r[4])                 # shoulder drop from the neck base to the shoulder tip
    rq = U(0.03, 0.12, r[21])                # how much of the shoulder corner is rounded off
    wc = {"pullover": U(0.08, 0.13, r[7]), "shirt": U(0.07, 0.11, r[7])}.get(kind, U(0.10, 0.15, r[7]))   # cuff width
    if kind == "pullover":
        hb = U(0.24, 0.32, r[0]); ls = U(0.78, 1.0, r[6]); yb = U(0.84, 1.0, r[11]); tail = 0.0
        top = neck_crew(U(0.10, 0.19, r[1]), U(0.0, 0.02, r[2])); ys = sd     # flat top: the back of the neckband shows
        fw, fh = U(0.92, 1.02, r[9]), U(0.90, 1.04, r[10])
    elif kind in ("coat", "coat_hood", "coat_flat"):
        hb = U(0.18, 0.29, r[0]); ls = U(0.66, 0.95, r[6]); yb = U(0.86, 1.0, r[11]); tail = 0.0
        if kind == "coat":
            ch = U(0.04, 0.10, r[2]); top = neck_stand(U(0.08, 0.15, r[1]), ch); ys = ch + sd
        elif kind == "coat_hood":
            ht = U(0.08, 0.16, r[2]); top = neck_hood(U(0.10, 0.17, r[1]), ht); ys = ht + 0.5 * sd
        else:
            top = neck_crew(U(0.08, 0.14, r[1]), U(0.0, 0.04, r[2])); ys = sd
        fw, fh = U(0.95, 1.05, r[9]), U(0.95, 1.20, r[10])
    else:  # shirt
        hb = U(0.21, 0.29, r[0]); ls = U(0.76, 0.97, r[6]); yb = U(0.92, 1.0, r[11]); tail = U(0.02, 0.09, r[12])
        ch = U(0.03, 0.06, r[2]); top = neck_collar(U(0.07, 0.12, r[1]), U(0.0, 0.035, r[13]), ch); ys = ch + sd
        fw, fh = U(0.92, 1.02, r[9]), U(0.95, 1.08, r[10])
    hs = hb + wu * U(0.7, 1.0, r[14])
    ya = ys + U(0.19, 0.29, r[8])
    sl, Q = sleeve_polys(hs, ys, hb, ya, ang, ls, wc, neck=top[-1], rq=rq)
    return [torso(top, Q[0], Q[1], hb, ya, fw, fh, yb, tail=tail)] + sl


# ----------------------------------------------------------------------------------------
# dress, trousers
# ----------------------------------------------------------------------------------------
def dress(r, kind):
    hbust = U(0.12, 0.20, r[0]); ybust = U(0.16, 0.28, r[8])
    hw = hbust * U(0.72, 1.0, r[9]); yw = U(0.32, 0.46, r[15])
    hhem = max(U(0.14, 0.36, r[10]), hw)
    ymid = 0.5 * (yw + 1.0)
    hmid = hw + (hhem - hw) * U(0.4, 0.85, r[16])
    skirt = [(hw, yw), (hmid, ymid), (hhem, 1.0), (0.0, 1.0 + U(0.0, 0.02, r[12]))]
    polys = []
    if kind == "strap":
        so = min(U(0.07, 0.16, r[1]), hbust - 0.01); si = max(so - U(0.025, 0.07, r[7]), 0.02)
        top = scoop_top(si, so, U(0.06, 0.20, r[2]))
        arm = qbez((so, 0.0), (so, ybust), (hbust, ybust), 6)
        polys.append(sym(top + arm + skirt))
    elif kind == "tube":
        polys.append(sym([(0.0, 0.0), (hbust, 0.0), (hbust, max(ybust - 0.1, 0.05))] + skirt))
    else:
        ys = U(0.02, 0.07, r[4])
        top = neck_crew(U(0.07, 0.12, r[1]), U(0.0, 0.06, r[2]))
        if kind == "short":
            hs = hbust + U(0.0, 0.05, r[3]); ya = ys + U(0.12, 0.20, r[8])
            sl = sleeve_polys(hs, ys, hbust, ya, U(30, 70, r[5]), U(0.05, 0.14, r[6]), U(0.07, 0.11, r[7]))[0]
        else:  # long sleeves
            hs = hbust + U(0.06, 0.10, r[3]); ya = ys + U(0.14, 0.22, r[8])
            sl = sleeve_polys(hs, ys, hbust, ya, U(2, 14, r[5]), U(0.40, 0.66, r[6]), U(0.05, 0.08, r[7]))[0]
        polys.append(sym(top + [(hs, ys), (hbust, ya)] + skirt))
        polys += sl
    return polys


def trouser(r, kind):
    if kind == "single":   # folded / legs together: one tapering column
        ht = U(0.12, 0.20, r[0]); hbm = U(0.07, 0.15, r[1])
        return [sym([(0.0, 0.0), (ht, 0.0), (ht * U(1.0, 1.12, r[2]), U(0.12, 0.25, r[3])), (hbm, 1.0), (0.0, 1.0)])]
    hw = U(0.14, 0.23, r[0]); hh = hw * U(1.0, 1.2, r[2]); yh = U(0.12, 0.26, r[3]); yc = U(0.24, 0.40, r[4])
    xo = hh * U(0.70, 1.08, r[1]); wl = hh * U(0.42, 0.75, r[5])     # ankle width: roughly half the hip half-width or more
    xi = xo - wl
    ymid = 0.5 * (yh + 1.0); xmid = 0.5 * (hh + xo) + U(-0.01, 0.02, r[6])
    if xi < 0.012:
        return [sym([(0.0, 0.0), (hw, 0.0), (hh, yh), (xmid, ymid), (xo, 1.0), (0.0, 1.0)])]
    g0 = min(xi, U(0.0, 0.03, r[7]))
    return [sym([(0.0, 0.0), (hw, 0.0), (hh, yh), (xmid, ymid), (xo, 1.0), (xi, 1.0), (g0, yc), (0.0, yc)])]


# ----------------------------------------------------------------------------------------
# footwear: side view, toe on the left, length 1, drawn with y UP and flipped at the end
# ----------------------------------------------------------------------------------------
def flip(polys):
    out = []
    for p in polys:
        p = np.asarray(p, dtype=np.float64).copy(); p[:, 1] = -p[:, 1]; out.append(p)
    return out


def sneaker(r, kind):
    if kind == "high":
        ht = U(0.47, 0.68, r[0]); xt = U(0.58, 0.75, r[1]); dip = U(0.0, 0.04, r[2])
    else:
        ht = U(0.32, 0.47, r[0]); xt = U(0.54, 0.72, r[1]); dip = U(0.03, 0.10, r[2])
    P = [(0.0, U(0.04, 0.09, r[3])), (U(0.05, 0.12, r[4]), U(0.12, 0.19, r[5])),
         (U(0.28, 0.40, r[6]), U(0.18, 0.28, r[7])), (xt, ht),
         (min(xt + U(0.07, 0.13, r[8]), 0.9), ht - dip), (U(0.92, 0.98, r[9]), ht - U(-0.03, 0.08, r[10])),
         (1.0, U(0.12, 0.25, r[11])), (U(0.96, 1.0, r[12]), 0.0), (U(0.08, 0.15, r[13]), 0.0)]
    return flip([rnd(P, U(0.02, 0.06, r[14]))])


def boot(r, kind):
    H = U(0.48, 1.15, r[0]); xs = U(0.40, 0.60, r[1])
    yank = min(U(0.38, 0.52, r[2]), H - 0.08)
    P = [(0.0, U(0.04, 0.10, r[3])), (U(0.06, 0.14, r[4]), U(0.15, 0.25, r[5])),
         (U(0.28, 0.40, r[6]), min(U(0.24, 0.36, r[7]), yank - 0.03)), (xs, yank),
         (xs + U(-0.05, 0.08, r[8]), H), (U(0.90, 1.0, r[9]), H + U(-0.06, 0.04, r[10])),
         (U(0.96, 1.0, r[11]), U(0.15, 0.30, r[12])), (U(0.94, 1.0, r[13]), 0.0)]
    if kind == "heel":
        xh = U(0.66, 0.80, r[15]); hh = U(0.04, 0.16, r[16])
        P += [(xh, 0.0), (xh - U(0.01, 0.04, r[17]), hh), (U(0.42, 0.55, r[18]), U(0.0, 0.03, r[19]))]
    P += [(U(0.08, 0.15, r[20]), 0.0)]
    return flip([rnd(P, U(0.02, 0.06, r[14]))])


def sandal(r, kind):
    polys = []
    sw = U(0.03, 0.06, r[20])                      # strap width
    if kind == "flat":
        ts = U(0.03, 0.08, r[0])
        polys.append(np.array([(0.0, 0.0), (1.0, 0.0), (1.0, ts), (0.04, ts), (0.0, ts * 0.5)]))
        x1 = U(0.06, 0.22, r[1]); x2 = U(0.38, 0.58, r[2]); xm = U(x1, x2, r[3]); hm = ts + U(0.07, 0.20, r[4])
        if r[5] < 0.5:                             # toe band seen from the side: low filled hump
            polys.append(np.array([(x1, ts), (xm - 0.04, hm), (xm + 0.04, hm), (x2, ts)]))
        else:                                      # thin toe straps
            polys += stroke([(x1, ts), (xm, hm), (x2, ts)], sw)
        ha = ts + U(0.14, 0.36, r[6]); xa = U(0.60, 0.78, r[7])
        if r[8] < 0.5:                             # heel counter
            polys.append(np.array([(U(0.72, 0.86, r[9]), ts), (0.9, ha * U(0.6, 1.0, r[10])), (1.0, ha), (1.0, ts)]))
        else:
            polys += stroke([(0.97, ts), (0.97, ha)], sw)
        if r[11] < 0.75:                           # ankle strap
            polys += stroke([(0.97, ha), (xa, ha * U(0.7, 1.1, r[12]))], sw)
            if r[13] < 0.5:
                polys += stroke([(xa, ha), (U(0.60, 0.80, r[14]), ts)], sw)
        if r[15] < 0.35:                           # T-strap from toe band to ankle
            polys += stroke([(xm, hm), (xa, ha)], sw)
    elif kind == "heel":
        H = U(0.20, 0.62, r[0]); xb = U(0.28, 0.45, r[1]); xr = U(0.84, 0.96, r[2])
        pf = U(0.02, 0.12, r[3]) if r[4] < 0.4 else 0.03          # platform under the toes
        st = U(0.03, 0.07, r[5])
        polys.append(np.array([(0.0, 0.0), (xb, 0.0), (xb, pf), (0.02, pf)]))
        polys += stroke([(xb - 0.05, pf - st * 0.3), (xr, H), (1.0, H)], st)
        wt = U(0.05, 0.22, r[6]); wb = U(0.03, 0.9, r[7]) * wt
        polys.append(np.array([(1.0 - wt, H), (1.0, H), (1.0, 0.0), (1.0 - wb, 0.0)]))
        x1 = U(0.05, 0.18, r[8]); x2 = U(0.32, 0.52, r[9]); xm = U(x1, x2, r[10]); hm = pf + U(0.08, 0.22, r[11])
        if r[12] < 0.5:
            polys.append(np.array([(x1, pf), (xm - 0.04, hm), (xm + 0.04, hm), (x2, pf)]))
        else:
            polys += stroke([(x1, pf), (xm, hm), (x2, pf)], sw)
        ha = H + U(0.12, 0.34, r[13]); xa = U(0.58, 0.78, r[14])
        if r[15] < 0.8:
            polys += stroke([(0.97, H), (0.97, ha), (xa, ha * U(0.75, 1.05, r[16]))], sw)
        if r[17] < 0.4:
            polys += stroke([(xm, hm), (xa, ha)], sw)
    else:  # "solid": wedge / slide / clog / sport sandal with a mostly closed upper
        tf = U(0.04, 0.12, r[0]); Hh = tf + U(0.0, 0.30, r[1])
        polys.append(np.array([(0.0, 0.0), (1.0, 0.0), (1.0, Hh), (U(0.35, 0.55, r[2]), tf), (0.03, tf), (0.0, tf * 0.5)]))
        x1 = U(0.03, 0.15, r[3]); x2 = U(0.45, 0.70, r[4]); hm = tf + U(0.10, 0.24, r[5]); xm = U(0.25, 0.5, r[6])
        y2 = tf + (Hh - tf) * max(0.0, (x2 - 0.45)) / 0.55
        polys.append(rnd(np.array([(x1, tf), (xm - 0.06, hm), (xm + 0.08, hm), (x2, y2)]), 0.03))
        if r[7] < 0.6:                             # back part
            ha = Hh + U(0.10, 0.26, r[8])
            if r[9] < 0.5:
                polys.append(np.array([(U(0.70, 0.85, r[10]), Hh * 0.9), (0.9, ha), (1.0, ha), (1.0, Hh)]))
            else:
                polys += stroke([(0.97, Hh), (0.97, ha), (U(0.6, 0.78, r[10]), ha * U(0.8, 1.0, r[11]))], sw)
    return flip(polys)


# ----------------------------------------------------------------------------------------
# bags: front view, body width 1, y down, top of the body at y = 0
# ----------------------------------------------------------------------------------------
def bag(r, kind):
    polys = []
    hb = U(0.50, 0.90, r[0])
    if kind == "clutch":
        hb = U(0.48, 0.85, r[0])
    if kind == "tall":
        hb = U(1.0, 1.4, r[0])
    wt = U(0.72, 1.0, r[1]); wbm = U(0.80, 1.0, r[2])
    if r[3] < 0.5:
        wt = 1.0
    else:
        wbm = 1.0
    body = np.array([(-0.5 * wt, 0.0), (0.5 * wt, 0.0), (0.5 * wbm, hb), (-0.5 * wbm, hb)])
    polys.append(rnd(body, U(0.02, 0.14, r[4])))
    th = U(0.03, 0.07, r[5])
    if kind in ("handle", "tall"):
        xh = U(0.14, 0.36, r[6]) * wt; hh = U(0.14, 0.55, r[7])
        if kind == "tall":
            xh = U(0.10, 0.22, r[6]); hh = U(0.08, 0.2, r[7])
        if r[8] < 0.6:                              # round handle
            pts = arc(0.0, 0.03, xh, hh, 0, 180, 12)
        else:                                       # pointed / hanging strap
            xm = xh * U(0.0, 0.5, r[9])
            pts = np.array([(xh, 0.03), (xm, -hh), (-xm, -hh), (-xh, 0.03)])
        polys += stroke(pts, th)
    elif kind == "strap":                           # long shoulder strap looping high above the body
        xh = 0.5 * wt * U(0.75, 1.0, r[6]); hh = U(0.45, 1.0, r[7]); th = U(0.025, 0.05, r[5])
        if r[8] < 0.5:
            pts = arc(0.0, 0.05, xh, hh, 0, 180, 14)
        else:
            xm = xh * U(0.0, 0.4, r[9])
            pts = np.array([(xh, 0.05), (xm, -hh), (-xm, -hh), (-xh, 0.05)])
        polys += stroke(pts, th)
    return polys


# ----------------------------------------------------------------------------------------
# which drawings each class gets:  (share of the class's drawings, prior bias, generator)
# The bias is a small hand-chosen handicap for variants that are rare in their class and look like
# the typical member of another class (e.g. a sleeveless "shirt" versus a tank top).
# ----------------------------------------------------------------------------------------
CLASSES = [
    # 0 T-shirt / top
    [(0.60, 0.0, lambda r: tee(r)),
     (0.20, 0.0, lambda r: tank(r)),
     (0.20, 0.0, lambda r: tee(r, sleeve_len=(0.04, 0.10), body=(0.21, 0.30)))],
    # 1 trouser
    [(0.88, 0.0, lambda r: trouser(r, "legs")),
     (0.12, -0.01, lambda r: trouser(r, "single"))],
    # 2 pullover
    [(1.00, 0.0, lambda r: long_top(r, "pullover"))],
    # 3 dress
    [(0.42, 0.0, lambda r: dress(r, "strap")),
     (0.30, 0.0, lambda r: dress(r, "short")),
     (0.18, -0.01, lambda r: dress(r, "long")),
     (0.10, -0.01, lambda r: dress(r, "tube"))],
    # 4 coat
    [(0.60, 0.0, lambda r: long_top(r, "coat")),
     (0.15, 0.0, lambda r: long_top(r, "coat_hood")),
     (0.25, -0.015, lambda r: long_top(r, "coat_flat"))],
    # 5 sandal
    [(0.40, 0.0, lambda r: sandal(r, "flat")),
     (0.35, 0.0, lambda r: sandal(r, "heel")),
     (0.25, -0.01, lambda r: sandal(r, "solid"))],
    # 6 shirt
    [(0.60, 0.0, lambda r: long_top(r, "shirt")),
     (0.20, -0.02, lambda r: tee(r, sleeve_len=(0.10, 0.22), body=(0.24, 0.34), collar=True)),
     (0.20, -0.01, lambda r: tank(r, body=(0.22, 0.32), strap_w=(0.05, 0.12), strap_x=(0.16, 0.26), flare=(0.95, 1.1)))],
    # 7 sneaker
    [(0.82, 0.0, lambda r: sneaker(r, "low")),
     (0.18, -0.03, lambda r: sneaker(r, "high"))],
    # 8 bag
    [(0.50, 0.0, lambda r: bag(r, "handle")),
     (0.27, 0.0, lambda r: bag(r, "clutch")),
     (0.15, 0.0, lambda r: bag(r, "strap")),
     (0.08, -0.01, lambda r: bag(r, "tall"))],
    # 9 ankle boot
    [(0.55, 0.0, lambda r: boot(r, "flat")),
     (0.45, 0.0, lambda r: boot(r, "heel"))],
]


def draw_class(c, n, R):
    """Returns (n, 28, 28) coverage drawings and (n,) prior biases for class c.  R: (n, 32) uniforms."""
    subs = CLASSES[c]
    bounds = np.round(np.cumsum([s[0] for s in subs]) * n).astype(int); bounds[-1] = n
    imgs = np.zeros((n, 28, 28)); bias = np.zeros(n)
    j = 0
    for k in range(n):
        while k >= bounds[j]:
            j += 1
        imgs[k] = raster(subs[j][2](R[k]))
        bias[k] = subs[j][1]
    return imgs, bias


def make_templates(n=N_PER_CLASS):
    """All drawings -> dense rows.  Each drawing is normalised by itself alone."""
    R = np.random.default_rng(SEED).random((n, 32))   # the same parameter draws are reused by every class
    K = gauss_kernel(BLUR_SIGMA)
    W = np.zeros((10 * n, 784), dtype=np.float32); b = np.zeros(10 * n, dtype=np.float32)
    for c in range(10):
        imgs, bias = draw_class(c, n, R)
        T = blur_batch(curve(NOMINAL_INK * imgs), K).reshape(n, 784)
        T = T - T.mean(axis=1, keepdims=True)
        T = T / np.sqrt((T * T).sum(axis=1, keepdims=True))
        W[c * n:(c + 1) * n] = (T / 28.0).astype(np.float32)      # dense(norm(feature)) = Pearson correlation
        b[c * n:(c + 1) * n] = bias
    return W, b


# ----------------------------------------------------------------------------------------
# the network
# ----------------------------------------------------------------------------------------
def build(n=N_PER_CLASS):
    layers, wts = [], {}

    def conv(name, inp, W, b, stride=1, padding=0):
        W = np.asarray(W, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
        layers.append({"name": name, "op": "conv", "inputs": [inp], "out": int(W.shape[0]),
                       "kernel": [int(W.shape[2]), int(W.shape[3])], "stride": stride, "padding": padding})
        wts[name + ".W"] = W; wts[name + ".b"] = b

    def dense(name, inp, W, b):
        W = np.asarray(W, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
        layers.append({"name": name, "op": "dense", "inputs": [inp], "out": int(W.shape[0])})
        wts[name + ".W"] = W; wts[name + ".b"] = b

    def op(name, kind, inputs, **kw):
        layers.append(dict({"name": name, "op": kind, "inputs": list(inputs)}, **kw))

    # --- stage 1: per-pixel ramps; after the relu, (even - odd) channel pairs are clamps to [0, 1]
    ramps = [(lo, hi) for _, lo, hi in CURVE] + [(MASK_LO, MASK_HI)]
    W = np.zeros((2 * len(ramps), 1, 1, 1)); b = np.zeros(2 * len(ramps))
    for i, (lo, hi) in enumerate(ramps):
        k = 1.0 / (hi - lo)
        W[2 * i, 0, 0, 0] = k; b[2 * i] = -lo * k
        W[2 * i + 1, 0, 0, 0] = k; b[2 * i + 1] = -lo * k - 1.0
    conv("ramp", "image", W, b)
    op("ramp_r", "relu", ["ramp"])
    nch = 2 * len(ramps); m0 = 2 * len(CURVE)        # channels m0, m0+1 hold the ink mask ramp

    # --- stage 2: silhouette feature = blurred intensity curve, standardised over the image
    K = gauss_kernel(BLUR_SIGMA)
    W = np.zeros((1, nch, 5, 5))
    for i, (w, _, _) in enumerate(CURVE):
        W[0, 2 * i] = w * K; W[0, 2 * i + 1] = -w * K
    conv("feat", "ramp_r", W, [0.0], padding=2)
    op("feat_n", "norm", ["feat"])

    # --- stage 3: correlation with every drawing, best drawing per class
    TW, Tb = make_templates(n)
    dense("match", "feat_n", TW, Tb)
    op("match_g", "reshape", ["match"], shape=[1, 10, n])
    op("match_max", "maxpool", ["match_g"], size=[1, n], stride=[1, n])
    op("shape_score", "reshape", ["match_max"], shape=[10])

    # --- stage 4: "thin / strappy" cue for sandals from grey-scale morphology on the ink mask M
    def mask_conv(name, sign, pad):
        W = np.zeros((1, nch, 1, 1)); W[0, m0, 0, 0] = sign; W[0, m0 + 1, 0, 0] = -sign
        conv(name, "ramp_r", W, [0.0], padding=pad)

    def neg_conv(name, inp, pad):
        conv(name, inp, -np.ones((1, 1, 1, 1)), [0.0], padding=pad)

    mask_conv("m_neg", -1.0, 1)                                         # -M, zero border        (1, 30, 30)
    op("ero_neg", "maxpool", ["m_neg"], size=[3, 3], stride=[1, 1])    # -erode3(M)             (1, 28, 28)
    neg_conv("ero", "ero_neg", 1)                                       # erode3(M), zero border (1, 30, 30)
    op("opened", "maxpool", ["ero"], size=[3, 3], stride=[1, 1])       # open3(M)               (1, 28, 28)
    mask_conv("m_pad", 1.0, 4)                                          # M, zero border         (1, 36, 36)
    op("dil", "maxpool", ["m_pad"], size=[5, 5], stride=[1, 1])        # dilate5(M)             (1, 32, 32)
    neg_conv("dil_neg", "dil", 0)
    op("closed_neg", "maxpool", ["dil_neg"], size=[5, 5], stride=[1, 1])   # -close5(M)         (1, 28, 28)
    # s = thin + holes - 0.22 * ink - 5,  thin = sum(M - open3 M),  holes = sum(close5 M - M),  ink = sum(M)
    # (a solid shoe has almost no thin parts or holes; straps and cut-outs are all thin parts and holes)
    cM = 1.0 - 1.0 - 0.22
    W = np.zeros((1, nch, 28, 28)); W[0, m0] = cM; W[0, m0 + 1] = -cM
    dense("s_ink", "ramp_r", W.reshape(1, -1), [-5.0])
    dense("s_open", "opened", -np.ones((1, 784)), [0.0])
    dense("s_closed", "closed_neg", -np.ones((1, 784)), [0.0])
    op("s_sum", "add", ["s_ink", "s_open", "s_closed"])
    dense("s_ramp", "s_sum", np.array([[1.0 / 40.0], [1.0 / 40.0]]), [0.0, -1.0])
    op("s_ramp_r", "relu", ["s_ramp"])
    W = np.zeros((10, 2)); W[5, 0] = 0.25; W[5, 1] = -0.25             # up to +0.25 for "sandal"
    dense("strappy", "s_ramp_r", W, np.zeros(10))

    # --- stage 5: "front opening" cue.  A zip / button line runs down the middle of coats (and, fainter, shirts)
    # but not pullovers or T-shirts.  Units look for a 1- or 2-pixel wide vertical line, brighter or darker than
    # its flanks, held over rows 7..22 near the centre column, on the standardised raw image.
    op("img_n", "norm", ["image"])
    rows = list(range(7, 23)); units = []
    for c in (12, 13, 14, 15):
        u = np.zeros((28, 28)); u[rows, c] = 1.0; u[rows, c - 1] = -0.5; u[rows, c + 1] = -0.5; units.append(u / len(rows))
    for c in (12, 13, 14):
        u = np.zeros((28, 28)); u[rows, c] = 0.5; u[rows, c + 1] = 0.5; u[rows, c - 1] = -0.5; u[rows, c + 2] = -0.5
        units.append(u / len(rows))
    units = np.array(units + [-u for u in units]).reshape(-1, 784)
    nu = len(units)
    dense("line", "img_n", units, np.zeros(nu))
    op("line_r", "relu", ["line"])
    op("line_g", "reshape", ["line_r"], shape=[1, 1, nu])
    op("line_max", "maxpool", ["line_g"], size=[1, nu], stride=[1, nu])
    L0, L1 = 0.12, 0.30                                # line contrast (in image standard deviations): none .. clear
    k = 1.0 / (L1 - L0)
    dense("line_ramp", "line_max", np.array([[k], [k]]), [-L0 * k, -L0 * k - 1.0])
    op("line_ramp_r", "relu", ["line_ramp"])
    #            no line   clear line
    table = {4: (-0.006, +0.014),      # coat: zip or buttons, usually clear
             6: (-0.003, +0.010),      # shirt: button placket, fainter
             2: (+0.006, -0.014),      # pullover: closed front
             0: (0.0, -0.006)}         # T-shirt / top: closed front
    W = np.zeros((10, 2)); b = np.zeros(10)
    for c, (lo, hi) in table.items():
        b[c] = lo; W[c, 0] = hi - lo; W[c, 1] = -(hi - lo)
    dense("opening", "line_ramp_r", W, b)

    op("scores", "add", ["shape_score", "strappy", "opening"])
    return {"layers": layers}, wts


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    net, wts = build()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(net, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **wts)


if __name__ == "__main__":
    main()
