"""Builds the network and its weights from hand-written shape models only (no data is read).

Idea: every class gets a parametric drawing program written from general knowledge of the object (a T-shirt is a
body with two short sleeves hanging beside it, a boot is a sole line with a shaft at the heel end, a bag is a body
with a handle loop, ...).  Each drawing is rendered the way the task says the photos were made (trimmed to the item,
longest side 28 px, centred), and every single rendering becomes one template on its own: its silhouette is blurred
a little and turned into per-pixel log-odds "is this pixel part of the item" (a Bernoulli model of the mask).

The network
  1. divides the image by its own standard deviation and thresholds it softly into an item mask (a threshold relative
     to the item's own brightness, so that dark items survive and the faint sharpening halo around items does not);
  2. cuts the mask into four quadrants, the left ones mirrored so that symmetric garments share templates;
  3. scores each quadrant against every template of that quadrant type with one convolution (log-likelihood of the mask);
  4. takes, per class and quadrant, the best template with max-pooling, and adds the four quadrant scores of each class;
  5. adds one hand-set cue: a thin vertical line through the centre of the trunk (zip / button placket / open front).
Classes that share an outline (T-shirt vs short-sleeved shirt; pullover vs jacket vs long-sleeved shirt) are drawn from
the same body parameters and differ only in neckline, collar and hem, so the decision between them rests on exactly those.
"""
import json, sys, os
import numpy as np

"""Batched rasteriser for hand-drawn shape models (numpy only, float64 geometry).

A scene is a list of primitives in world coordinates (x right, y down):
  ('poly',   V (B,n,2),            sign, en (B,) bool)   filled polygon, nonzero winding
  ('stroke', P (B,n,2), r_px (B,), sign, en (B,) bool)   thick polyline, radius in output pixels
Positive primitives are OR-ed in, negative ones are cut out, in list order.
The scene is trimmed to its bounding box, scaled so the longest side is 28 px and centred,
as the task says the product photos were.
"""

N = 28
SS = 3


def _grid(sym):
    ys = (np.arange(N * SS) + 0.5) / SS
    xs = ys[N * SS // 2:] if sym else ys
    return xs, ys


def _fill(V, xs, ys, en):
    B, n, _ = V.shape
    PY = ys[None, :, None]
    PX = xs[None, None, :]
    wn = np.zeros((B, len(ys), len(xs)), np.int8)
    e8 = en.astype(np.int8)[:, None, None]
    for i in range(n):
        j = (i + 1) % n
        x1 = V[:, i, 0][:, None, None]; y1 = V[:, i, 1][:, None, None]
        x2 = V[:, j, 0][:, None, None]; y2 = V[:, j, 1][:, None, None]
        up = (y1 <= PY) & (y2 > PY)
        dn = (y2 <= PY) & (y1 > PY)
        dy = y2 - y1
        t = (PY - y1) / np.where(dy == 0, 1.0, dy)
        xc = x1 + t * (x2 - x1)
        s = (up.astype(np.int8) - dn.astype(np.int8)) * e8
        wn += s * (xc > PX)
    return wn != 0


def _stroke(P, r, xs, ys, en):
    B, n, _ = P.shape
    PY = ys[None, :, None]
    PX = xs[None, None, :]
    out = np.zeros((B, len(ys), len(xs)), bool)
    r2 = (r * r)[:, None, None]
    for i in range(n - 1):
        x1 = P[:, i, 0][:, None, None]; y1 = P[:, i, 1][:, None, None]
        dx = P[:, i + 1, 0][:, None, None] - x1; dy = P[:, i + 1, 1][:, None, None] - y1
        L2 = dx * dx + dy * dy
        ax = PX - x1; ay = PY - y1
        t = np.clip((ax * dx + ay * dy) / np.where(L2 == 0, 1.0, L2), 0.0, 1.0)
        ex = ax - t * dx; ey = ay - t * dy
        out |= (ex * ex + ey * ey) <= r2
    return out & en[:, None, None]


def render(scene, B, sym=False):
    """Returns coverage (B, 28, 28) in [0, 1]."""
    big = 1e9
    # --- bounding box of the positive primitives (stroke radii are in pixels, so iterate on the scale)
    s = np.ones(B)
    for it in range(6):
        x0 = np.full(B, big); x1 = np.full(B, -big); y0 = np.full(B, big); y1 = np.full(B, -big)
        for prim in scene:
            if prim[0] == 'poly':
                _, V, sign, en = prim
                rad = np.zeros(B)
            else:
                _, V, rpx, sign, en = prim
                rad = rpx / s if it > 0 else np.zeros(B)
            if sign < 0:
                continue
            lo_x = np.where(en, V[:, :, 0].min(1) - rad, big); hi_x = np.where(en, V[:, :, 0].max(1) + rad, -big)
            lo_y = np.where(en, V[:, :, 1].min(1) - rad, big); hi_y = np.where(en, V[:, :, 1].max(1) + rad, -big)
            x0 = np.minimum(x0, lo_x); x1 = np.maximum(x1, hi_x)
            y0 = np.minimum(y0, lo_y); y1 = np.maximum(y1, hi_y)
        if sym:
            x1 = np.maximum(x1, -x0); x0 = -x1
        s = N / np.maximum(x1 - x0, y1 - y0)
    ox = N / 2 - s * (x0 + x1) / 2
    oy = N / 2 - s * (y0 + y1) / 2
    xs, ys = _grid(sym)
    acc = np.zeros((B, len(ys), len(xs)), bool)
    for prim in scene:
        if prim[0] == 'poly':
            _, V, sign, en = prim
            Vp = np.stack([V[:, :, 0] * s[:, None] + ox[:, None], V[:, :, 1] * s[:, None] + oy[:, None]], -1)
            m = _fill(Vp, xs, ys, en)
        else:
            _, V, rpx, sign, en = prim
            Vp = np.stack([V[:, :, 0] * s[:, None] + ox[:, None], V[:, :, 1] * s[:, None] + oy[:, None]], -1)
            m = _stroke(Vp, rpx, xs, ys, en)
        if sign > 0:
            acc |= m
        else:
            acc &= ~m
    if sym:
        acc = np.concatenate([acc[:, :, ::-1], acc], axis=2)
    cov = acc.reshape(B, N, SS, N, SS).mean(axis=(2, 4))
    return cov


# ---------- helpers for building outlines ----------
def pts(*cols):
    """pts((x0,y0),(x1,y1),...) with each coordinate a (B,) array or scalar -> (B, n, 2)."""
    B = max(np.size(c) for p in cols for c in p)
    out = np.empty((B, len(cols), 2))
    for i, (x, y) in enumerate(cols):
        out[:, i, 0] = x; out[:, i, 1] = y
    return out


def mirror(P):
    """Mirror about x = 0 and reverse the order."""
    Q = P[:, ::-1].copy(); Q[:, :, 0] *= -1
    return Q


def spline(C, m=5, closed=False):
    """Catmull-Rom curve through control points C (B, n, 2); m samples per segment."""
    B, n, _ = C.shape
    if closed:
        Cp = np.concatenate([C[:, -1:], C, C[:, :2]], 1); nseg = n
    else:
        Cp = np.concatenate([C[:, :1], C, C[:, -1:]], 1); nseg = n - 1
    t = (np.arange(m) / m)[None, None, :, None]
    P0 = Cp[:, 0:nseg][:, :, None]; P1 = Cp[:, 1:nseg + 1][:, :, None]
    P2 = Cp[:, 2:nseg + 2][:, :, None]; P3 = Cp[:, 3:nseg + 3][:, :, None]
    Q = 0.5 * (2 * P1 + (-P0 + P2) * t + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t ** 2 + (-P0 + 3 * P1 - 3 * P2 + P3) * t ** 3)
    Q = Q.reshape(B, nseg * m, 2)
    if not closed:
        Q = np.concatenate([Q, C[:, -1:]], 1)
    return Q


def arc(cx, cy, rx, ry, a0, a1, m):
    """Elliptic arc, angles in radians measured from +x towards +y (down); returns (B, m, 2)."""
    B = max(np.size(v) for v in (cx, cy, rx, ry, a0, a1))
    u = np.linspace(0.0, 1.0, m)[None, :]
    a = np.broadcast_to(np.asarray(a0, float).reshape(-1, 1), (B, 1)) * (1 - u) + np.broadcast_to(np.asarray(a1, float).reshape(-1, 1), (B, 1)) * u
    x = np.asarray(cx, float).reshape(-1, 1) + np.asarray(rx, float).reshape(-1, 1) * np.cos(a)
    y = np.asarray(cy, float).reshape(-1, 1) + np.asarray(ry, float).reshape(-1, 1) * np.sin(a)
    return np.stack([np.broadcast_to(x, (B, m)), np.broadcast_to(y, (B, m))], -1)


def cat(*parts):
    return np.concatenate(parts, axis=1)

"""Shape models for the garment classes, typed from general knowledge of how these garments are cut.
Units: body length (shoulder/neck point to hem) = 1, x = 0 is the centre line, y grows downwards."""

D = np.pi / 180


def U(rng, B, lo, hi):
    return rng.uniform(lo, hi, B)


def Up(rng, B, lo, hi, p):
    """Skewed towards lo for p > 1."""
    return lo + (hi - lo) * rng.uniform(0, 1, B) ** p


def top_scene(B, bw_a, bw_w, yw, bw_h, hc, sw, sd, nw, yn, ah, sl, th, cw, sleeve,
              collar=None, ch=0.0, cwid=0.0, vnotch=None, vd=0.0, hood=None, hw=0.0, hh=0.0, hem_round=0.0):
    """Body + set-in sleeves (+ collar / hood). All arguments are (B,) arrays; sleeve/collar/vnotch/hood are bool masks."""
    T = np.ones(B, bool)
    ay = sd + ah
    # right half of the body, from the top centre clockwise
    half = pts((0 * nw, yn), (0.55 * nw, 0.8 * yn), (nw, 0 * nw), (sw, sd), (bw_a, ay), (bw_w, yw),
               (bw_h, 1 - hem_round), (bw_h - 1.5 * hem_round, 1 + 0 * nw), (0.5 * bw_h, 1 + 0.8 * hc), (0 * nw, 1 + hc))
    body = cat(half, mirror(half)[:, 1:-1])
    scene = [('poly', body, +1, T)]
    # sleeve: outer edge leaves the shoulder tip at angle th from the vertical, cuff is perpendicular to it
    Sx, Sy = sw, sd
    Ox, Oy = Sx + sl * np.sin(th), Sy + sl * np.cos(th)
    Ix, Iy = Ox - cw * np.cos(th), Oy + cw * np.sin(th)
    sl_r = pts((Sx - 0.07, Sy + 0.01), (Sx, Sy), (Ox, Oy), (Ix, Iy), (bw_a, ay), (bw_a - 0.07, ay))
    scene.append(('poly', sl_r, +1, sleeve))
    scene.append(('poly', mirror(sl_r), +1, sleeve))
    if collar is not None:
        col = pts((-nw, 0.04), (-cwid, -ch), (cwid, -ch), (nw, 0.04))
        scene.append(('poly', col, +1, collar))
    if hood is not None:
        hd = cat(pts((hw, 0.05)), arc(0.0, 0.0, hw, hh, 0 * D, -180 * D, 9), pts((-hw, 0.05)))
        scene.append(('poly', hd, +1, hood))
    if vnotch is not None:
        vx = 0.55 * cwid
        vn = pts((-vx, -ch - 0.02), (vx, -ch - 0.02), (0 * vx, -ch + vd))
        scene.append(('poly', vn, -1, vnotch))
    return scene


def tank_scene(B, bw, bw_h, nw, strap, yn, ay, hc, yn_top=None):
    """Sleeveless top with shoulder straps: deep neck and armholes."""
    T = np.ones(B, bool)
    xo = nw + strap
    half = pts((0 * nw, yn), (0.7 * nw, 0.75 * yn), (nw, 0.15 * yn), (nw, 0 * nw), (xo, 0 * nw), (xo + 0.01, 0.35 * ay),
               (0.5 * (xo + bw), 0.8 * ay), (bw, ay), (bw_h, 1 + 0 * nw), (0 * nw, 1 + hc))
    body = cat(half, mirror(half)[:, 1:-1])
    return [('poly', body, +1, T)]


# ---------------------------------------------------------------- sleeved tops: classes 0, 2, 4, 6
# T-shirt and short-sleeved shirt, and pullover / jacket / long-sleeved shirt, are drawn from the SAME body and sleeve
# parameters (same seed, same order of draws), so that the bank holds each body once per class.  The classes then differ
# only in what really tells them apart in outline:
#   T-shirt, pullover : round neck: a wide top edge, flat, slightly dipped or slightly raised (neck band)
#   coat              : stand-up collar (a narrower, taller bump on top) or a hood
#   shirt             : small collar with an open notch; often a curved shirt-tail hem
# Three-quarter sleeves on a top are most typical of blouses, so that sleeve length is drawn for the shirt class only.
NECK_ID = {'crew': 1, 'coat': 2, 'shirt': 3}


def neck_and_hem(B, rng, neck):
    """Class-specific parts, drawn from their own stream."""
    r = np.random.default_rng([int(rng.integers(1 << 30)), NECK_ID[neck]])
    F = np.zeros(B, bool)
    if neck == 'crew':
        # top edge between the shoulder points: from a ribbed neck band standing a little above them (garment worn or
        # hung: 2-3 cm on a 65-70 cm garment) to the shallow dip of a garment laid flat
        return dict(nw=U(r, B, 0.09, 0.18), yn=U(r, B, -0.045, 0.06), collar=F, ch=0.03 + 0 * U(r, B, 0, 1), cwid=0.1 + 0 * U(r, B, 0, 1),
                    vnotch=F, vd=np.zeros(B), hood=F, hw=np.full(B, 0.15), hh=np.full(B, 0.1), curved=F, hc=np.zeros(B), hr=np.zeros(B))
    if neck == 'coat':
        nw = U(r, B, 0.08, 0.15)
        hood = r.uniform(0, 1, B) < 0.35
        return dict(nw=nw, yn=U(r, B, -0.02, 0.01), collar=~hood, ch=U(r, B, 0.04, 0.10), cwid=nw * U(r, B, 0.6, 1.0),
                    vnotch=F, vd=np.zeros(B), hood=hood, hw=U(r, B, 0.11, 0.19), hh=U(r, B, 0.07, 0.16), curved=F, hc=np.zeros(B), hr=np.zeros(B))
    nw = U(r, B, 0.07, 0.14)
    curved = r.uniform(0, 1, B) < 0.5
    return dict(nw=nw, yn=U(r, B, -0.01, 0.02), collar=~F, ch=U(r, B, 0.025, 0.065), cwid=nw * U(r, B, 0.8, 1.15),
                vnotch=~F, vd=U(r, B, 0.03, 0.10), hood=F, hw=np.full(B, 0.15), hh=np.full(B, 0.1),
                curved=curved, hc=np.where(curved, U(r, B, 0.02, 0.07), 0.0), hr=np.where(curved, U(r, B, 0.02, 0.06), 0.0))


def gen_sleeved(B, rng, sleeves, neck):
    if sleeves == 'short':                           # sleeves end above the elbow and hang down beside the body
        bw = U(rng, B, 0.22, 0.41)
        sw = bw * U(rng, B, 0.98, 1.2)
        sd = U(rng, B, 0.03, 0.10)
        ah = U(rng, B, 0.20, 0.33)
        th = U(rng, B, 6, 58) * D
        sl = U(rng, B, 0.10, 0.42)
        cw = U(rng, B, 0.14, 0.27)
    elif sleeves == 'mid':                           # three-quarter sleeves: end between elbow and wrist, often flared
        bw = U(rng, B, 0.24, 0.42)
        sw = bw * U(rng, B, 0.95, 1.15)
        sd = U(rng, B, 0.03, 0.10)
        ah = U(rng, B, 0.19, 0.33)
        th = U(rng, B, 4, 40) * D
        sl = (U(rng, B, 0.48, 0.78) - sd) / np.cos(th)
        cw = U(rng, B, 0.10, 0.22)
    else:                                            # long sleeves hanging beside the body, cuffs about level with the hem
        bw = U(rng, B, 0.24, 0.44)
        sw = bw * U(rng, B, 0.92, 1.10)
        sd = U(rng, B, 0.03, 0.11)
        ah = U(rng, B, 0.19, 0.33)
        th = Up(rng, B, 1, 26, 1.5) * D
        sl = (U(rng, B, 0.80, 1.12) - sd) / np.cos(th)
        cw = U(rng, B, 0.08, 0.17)
    mid = bw * U(rng, B, 0.92, 1.04)
    yw = U(rng, B, 0.5, 0.7)
    hem = bw * U(rng, B, 0.88, 1.10)
    hr0 = U(rng, B, 0.0, 0.02)
    n = neck_and_hem(B, rng, neck)
    return top_scene(B, bw, mid, yw, hem, n['hc'], sw, sd, n['nw'], n['yn'], ah, sl, th, cw, np.ones(B, bool),
                     collar=n['collar'], ch=n['ch'], cwid=n['cwid'], vnotch=n['vnotch'], vd=n['vd'],
                     hood=n['hood'], hw=n['hw'], hh=n['hh'], hem_round=np.where(n['curved'], n['hr'], hr0))


def gen_tank(B, rng):
    bw = U(rng, B, 0.23, 0.40)
    nw = U(rng, B, 0.07, 0.17)
    strap = U(rng, B, 0.025, 0.13)
    yn = U(rng, B, 0.08, 0.34)
    ay = U(rng, B, 0.20, 0.40)
    return tank_scene(B, bw, bw * U(rng, B, 0.9, 1.2), nw, np.minimum(strap, bw - nw - 0.01), yn, ay, U(rng, B, 0.0, 0.04))


def gen_cami(B, rng):
    """Strappy top: straight or slightly dipped top edge hanging from two thin straps; wider and shorter than a strap dress."""
    T = np.ones(B, bool)
    bw = U(rng, B, 0.24, 0.40)
    ys = U(rng, B, 0.10, 0.30)
    dip = U(rng, B, 0.0, 0.10)
    ya = ys + U(rng, B, 0.02, 0.10)
    bh = bw * U(rng, B, 0.92, 1.3)
    hc = U(rng, B, 0.0, 0.04)
    half = pts((0 * bw, ys + dip), (0.6 * bw, ys + 0.15 * dip), (0.85 * bw, ys), (bw, ya), (0.5 * (bw + bh) * U(rng, B, 0.92, 1.0), 0.6 + 0 * bw), (bh, 1 + 0 * bw), (0 * bw, 1 + hc))
    body = cat(half, mirror(half)[:, 1:-1])
    xs_b = bw * U(rng, B, 0.5, 0.88)
    xs_t = xs_b * U(rng, B, 0.5, 1.0)
    rpx = U(rng, B, 0.45, 1.0)
    st = pts((xs_b, ys + 0.02), (xs_t, 0 * bw))
    return [('poly', body, +1, T), ('stroke', st, rpx, +1, T), ('stroke', mirror(st), rpx, +1, T)]


# ---------------------------------------------------------------- class 4: coat
def gen_coat(B, rng, long=False):
    if long:                                         # overcoat: long and narrow, cuffs well above the hem
        bw = U(rng, B, 0.18, 0.30)
        cuff = U(rng, B, 0.58, 0.86)
        flare = U(rng, B, 0.95, 1.25)
    else:                                            # jacket / parka: hip length, cuffs about level with the hem
        bw = U(rng, B, 0.22, 0.37)
        cuff = U(rng, B, 0.80, 1.04)
        flare = U(rng, B, 0.9, 1.08)
    sw = bw * U(rng, B, 0.92, 1.1)
    sd = U(rng, B, 0.03, 0.10)
    nw = U(rng, B, 0.08, 0.16)
    ah = U(rng, B, 0.19, 0.32)
    th = Up(rng, B, 1, 14, 1.5) * D
    sl = (cuff - sd) / np.cos(th)
    cw = U(rng, B, 0.09, 0.17)
    kind = rng.uniform(0, 1, B)
    hood = kind < 0.25
    collar = (kind >= 0.25) & (kind < 0.88)
    yn = np.where(kind >= 0.88, U(rng, B, -0.01, 0.03), U(rng, B, -0.02, 0.01))
    return top_scene(B, bw, bw * U(rng, B, 0.93, 1.05), U(rng, B, 0.45, 0.65), bw * flare, 0 * bw, sw, sd, nw, yn, ah, sl, th, cw,
                     np.ones(B, bool), collar=collar, ch=U(rng, B, 0.03, 0.09), cwid=nw * U(rng, B, 0.65, 1.0),
                     hood=hood, hw=U(rng, B, 0.11, 0.19), hh=U(rng, B, 0.07, 0.16), hem_round=U(rng, B, 0.0, 0.02))


# ---------------------------------------------------------------- class 6: shirt
def gen_shirt_sleeveless(B, rng):
    bw = U(rng, B, 0.25, 0.40)
    nw = U(rng, B, 0.08, 0.16)
    strap = U(rng, B, 0.08, 0.20)
    yn = U(rng, B, 0.02, 0.16)
    ay = U(rng, B, 0.20, 0.34)
    return tank_scene(B, bw, bw * U(rng, B, 0.95, 1.25), nw, np.minimum(strap, bw - nw - 0.01), yn, ay, U(rng, B, 0.0, 0.06))


# ---------------------------------------------------------------- class 3: dress
def dress_body(B, rng, top_y, bb, with_shoulders, sw=None, sd=None, nw=None, yn=None, ay=None):
    """Bodice from the bust line down: waist, hip, flared or straight skirt. Returns right-half outline points below the armpit."""
    yw = U(rng, B, 0.30, 0.44)
    bwst = bb * U(rng, B, 0.72, 1.0)
    column = rng.uniform(0, 1, B) < 0.35                        # hangs straight down; the others flare
    bh = np.where(column, bb * U(rng, B, 0.9, 1.15), np.maximum(bb * U(rng, B, 0.95, 1.1), U(rng, B, 0.13, 0.40)))
    yh = yw + U(rng, B, 0.10, 0.22)
    hipw = bwst + (bh - bwst) * U(rng, B, 0.35, 0.9)
    return yw, bwst, yh, hipw, bh


def gen_dress(B, rng, kind):
    T = np.ones(B, bool)
    bb = U(rng, B, 0.08, 0.23)                       # half bust width (dress length = 1)
    hc = U(rng, B, 0.0, 0.03)
    if kind == 'strap':
        ys = U(rng, B, 0.07, 0.20)                   # top of the bodice, straps above it
        yw, bwst, yh, hipw, bh = dress_body(B, rng, ys, bb, False)
        ya = ys + U(rng, B, 0.02, 0.08)
        dip = U(rng, B, 0.0, 0.06)
        half = pts((0 * bb, ys + dip), (0.6 * bb, ys), (0.85 * bb, ys), (bb, ya), (bwst, yw), (hipw, yh), (bh, 1 + 0 * bb), (0 * bb, 1 + hc))
        body = cat(half, mirror(half)[:, 1:-1])
        xs_b = bb * U(rng, B, 0.5, 0.85)
        xs_t = xs_b * U(rng, B, 0.45, 1.0)
        rpx = U(rng, B, 0.45, 0.9)
        st = pts((xs_b, ys + 0.02), (xs_t, 0 * bb))
        return [('poly', body, +1, T), ('stroke', st, rpx, +1, T), ('stroke', mirror(st), rpx, +1, T)]
    sd = U(rng, B, 0.02, 0.06)
    nw = U(rng, B, 0.05, 0.10)
    sw = np.maximum(bb * U(rng, B, 0.85, 1.1), nw + 0.03)
    ah = U(rng, B, 0.12, 0.2)
    yw, bwst, yh, hipw, bh = dress_body(B, rng, sd + ah, bb, True)
    yw = np.maximum(yw, sd + ah + 0.06)
    yh = np.maximum(yh, yw + 0.08)
    if kind == 'sleeveless':
        yn = U(rng, B, 0.02, 0.16)
        swn = np.maximum(sw * U(rng, B, 0.7, 1.0), nw + 0.025)
        half = pts((0 * bb, yn), (0.6 * nw, 0.8 * yn), (nw, 0 * bb), (swn, sd), (swn + 0.3 * (bb - swn), sd + 0.6 * ah), (bb, sd + ah), (bwst, yw), (hipw, yh),
                   (bh, 1 + 0 * bb), (0 * bb, 1 + hc))
        body = cat(half, mirror(half)[:, 1:-1])
        return [('poly', body, +1, T)]
    yn = U(rng, B, 0.0, 0.10)
    half = pts((0 * bb, yn), (0.6 * nw, 0.8 * yn), (nw, 0 * bb), (sw, sd), (bb, sd + ah), (bwst, yw), (hipw, yh), (bh, 1 + 0 * bb), (0 * bb, 1 + hc))
    body = cat(half, mirror(half)[:, 1:-1])
    if kind == 'short':
        sl = U(rng, B, 0.05, 0.22); th = U(rng, B, 30, 75) * D; cw = U(rng, B, 0.08, 0.15)
    else:
        sl = U(rng, B, 0.34, 0.68); th = Up(rng, B, 3, 28, 1.5) * D; cw = U(rng, B, 0.045, 0.09)
    Sx, Sy = sw, sd
    Ox, Oy = Sx + sl * np.sin(th), Sy + sl * np.cos(th)
    Ix, Iy = Ox - cw * np.cos(th), Oy + cw * np.sin(th)
    sl_r = pts((Sx - 0.04, Sy + 0.01), (Sx, Sy), (Ox, Oy), (Ix, Iy), (bb, sd + ah), (bb - 0.04, sd + ah))
    return [('poly', body, +1, T), ('poly', sl_r, +1, T), ('poly', mirror(sl_r), +1, T)]


# ---------------------------------------------------------------- class 1: trousers
def gen_trousers(B, rng, shorts=False):
    T = np.ones(B, bool)
    if shorts:
        Lr = U(rng, B, 0.38, 0.62)                  # shorts: same cut, cut off above the knee
    else:
        Lr = np.ones(B)
    ww = U(rng, B, 0.14, 0.24)                       # half waist (full length = 1)
    yh = U(rng, B, 0.12, 0.22)
    hw = ww * U(rng, B, 1.0, 1.22)
    yc = U(rng, B, 0.24, 0.37)                       # crotch
    lw = U(rng, B, 0.09, 0.19)                       # leg width at the hem
    xo = hw * U(rng, B, 0.7, 1.15)                   # outer hem position: tapered ... slightly spread
    xi = np.maximum(xo - lw, 0.03)
    xk = hw + (xo - hw) * U(rng, B, 0.35, 0.65)      # outer edge at the knee
    lean = 0 * ww
    if shorts:
        # cut the legs at Lr: interpolate the outer and inner edges
        f = (Lr - yh) / (1 - yh)
        xo = hw + (xo - hw) * f
        g = (Lr - yc) / (1 - yc)
        xi = np.maximum(xi * g, 0.03)
        half = pts((lean, 0 * ww), (ww, 0 * ww), (hw, yh), (xo, Lr), (xi, Lr), (0 * ww, yc))
    else:
        half = pts((lean, 0 * ww), (ww, 0 * ww), (hw, yh), (xk, 0.5 * (yh + 1)), (xo, 1 + 0 * ww), (xi, 1 + 0 * ww), (0 * ww, yc))
    body = cat(half, mirror(half)[:, 1:-1])
    return [('poly', body, +1, T)]

"""Shape models for bags and footwear, typed from general knowledge.
Bags: body width = 1, centre line x = 0, y grows downwards, y = 0 is the top edge of the body.
Shoes: side view, toe to the left; x from 0 (toe tip) to 1 (back of the heel), height measured upwards from the ground
(the helper up() flips it to image coordinates)."""



def up(P):
    Q = P.copy(); Q[:, :, 1] *= -1
    return Q


# ---------------------------------------------------------------- class 8: bag
def bag_body(B, rng, hb):
    a = 0.5 * U(rng, B, 0.72, 1.0)
    flip = rng.uniform(0, 1, B) < 0.3                      # some bags are wider at the top (bucket), most at the bottom
    wt = np.where(flip, 0.5, a); wb = np.where(flip, a, 0.5)
    rc = U(rng, B, 0.02, 0.22) * np.minimum(hb, 1.0)       # bottom corner rounding
    rt = U(rng, B, 0.0, 0.10) * np.minimum(hb, 1.0)        # top corner rounding
    sag = U(rng, B, -0.02, 0.05) * hb                      # top edge may sag (soft bags) or bulge slightly
    half = pts((0 * hb, sag), (wt - rt, 0 * hb), (wt, rt), (wb + (wt - wb) * rc / hb, hb - rc), (wb - 0.35 * rc, hb - 0.3 * rc), (wb - rc, hb), (0 * hb, hb))
    return cat(half, mirror(half)[:, 1:-1]), wt


def gen_bag(B, rng, kind):
    T = np.ones(B, bool)
    if kind == 'plain':                                    # clutch, wallet, briefcase, pouch: no handle showing
        hb = U(rng, B, 0.42, 0.92)
        body, wt = bag_body(B, rng, hb)
        return [('poly', body, +1, T)]
    if kind == 'handle':                                   # hand bag / tote: handle arc standing on the top edge
        hb = U(rng, B, 0.5, 1.1)
        body, wt = bag_body(B, rng, hb)
        rx = wt * U(rng, B, 0.3, 0.85)
        ry = U(rng, B, 0.15, 0.8)
        rpx = U(rng, B, 0.5, 1.2)
        point = U(rng, B, 0.0, 1.0)                        # 0: round arc, 1: more pointed (hanging from a hook)
        a = np.linspace(-180, 0, 13)[None, :] * D
        x = rx[:, None] * np.cos(a)
        yr = np.sin(a)
        yt = -(1 - np.abs(np.cos(a)))
        y = 0.03 + ry[:, None] * ((1 - point[:, None]) * yr + point[:, None] * yt)
        h = np.stack([x, y], -1)
        return [('poly', body, +1, T), ('stroke', h, rpx, +1, T)]
    if kind == 'strap':                                    # shoulder bag: long strap attached near the top corners
        hb = U(rng, B, 0.45, 1.0)
        body, wt = bag_body(B, rng, hb)
        rx = wt * U(rng, B, 0.8, 1.0)
        ry = U(rng, B, 0.35, 1.2)
        rpx = U(rng, B, 0.5, 1.1)
        point = U(rng, B, 0.0, 1.0)
        a = np.linspace(-180, 0, 13)[None, :] * D
        x = rx[:, None] * np.cos(a)
        y = 0.06 + ry[:, None] * ((1 - point[:, None]) * np.sin(a) - point[:, None] * (1 - np.abs(np.cos(a))))
        h = np.stack([x, y], -1)
        return [('poly', body, +1, T), ('stroke', h, rpx, +1, T)]
    if kind == 'two':                                      # two short handles / side loops
        hb = U(rng, B, 0.5, 1.0)
        body, wt = bag_body(B, rng, hb)
        cx = wt * U(rng, B, 0.35, 0.6)
        rx = wt * U(rng, B, 0.15, 0.3)
        ry = U(rng, B, 0.1, 0.4)
        rpx = U(rng, B, 0.5, 1.1)
        h = arc(cx, 0.03, rx, ry, -180 * D, 0 * D, 9)
        return [('poly', body, +1, T), ('stroke', h, rpx, +1, T), ('stroke', mirror(h), rpx, +1, T)]
    raise ValueError(kind)


# ---------------------------------------------------------------- class 7: sneaker
def gen_sneaker(B, rng, high=False, white_sole=False):
    T = np.ones(B, bool)
    if high:
        H = U(rng, B, 0.50, 0.80)
        xt = U(rng, B, 0.46, 0.64)
    else:
        H = U(rng, B, 0.23, 0.47)                           # low shoe: much longer than tall
        xt = U(rng, B, 0.52, 0.82)                          # front of the ankle opening (top of the tongue)
    ts = U(rng, B, 0.01, 0.06)                              # toe spring
    toe_h = np.minimum(U(rng, B, 0.09, 0.17), 0.6 * H)      # height of the toe box
    dip = U(rng, B, 0.0, 0.08)
    tab = U(rng, B, 0.0, 0.08)
    v1 = toe_h + (H - toe_h) * U(rng, B, 0.15, 0.5)
    v2 = v1 + (H - v1) * U(rng, B, 0.3, 0.7)
    top = pts((0 * H, ts + 0.02), (0.035 + 0 * H, toe_h * 0.75), (0.14 + 0 * H, toe_h + 0.01), (0.30 + 0 * H, v1), (0.5 * (0.30 + xt), v2),
              (xt, H), (0.5 * (xt + 0.95), H - dip), (0.95 + 0 * H, H - dip + tab), (0.995 + 0 * H, 0.6 * H), (1.0 + 0 * H, 0.05 + 0 * H))
    curve = spline(top, 4)
    bottom = pts((0.975 + 0 * H, 0 * H), (0.06 + 0 * H, 0 * H), (0.01 + 0 * H, ts * 0.5))
    scene = [('poly', up(cat(curve, bottom)), +1, T)]
    if white_sole:
        # a white rubber sole is dark in the inverted photo: a dark band above the thin line where the sole meets the ground
        b0 = U(rng, B, 0.018, 0.04)
        b1 = b0 + U(rng, B, 0.035, 0.12)
        x0 = U(rng, B, -0.02, 0.12); x1 = U(rng, B, 0.88, 1.02)
        lift = U(rng, B, 0.0, 0.04)                         # the band follows the toe spring a little
        band = pts((x0, b0 + lift), (0.3 + 0 * H, b0), (x1, b0), (x1, b1), (0.3 + 0 * H, b1), (x0, b1 + lift))
        scene.append(('poly', up(band), -1, T))
    return scene


# ---------------------------------------------------------------- class 9: ankle boot
def gen_boot(B, rng):
    T = np.ones(B, bool)
    H = Up(rng, B, 0.50, 1.22, 1.4)                         # total height relative to the length; many are only ankle high
    ts = U(rng, B, 0.01, 0.06)
    xs = U(rng, B, 0.30, 0.62)                              # where the front of the shaft stands
    lean = U(rng, B, -0.06, 0.06)
    tilt = U(rng, B, -0.08, 0.08)                           # the top may slope
    xb = U(rng, B, 0.93, 1.0)
    toe_h = U(rng, B, 0.12, 0.24)
    v1 = np.minimum(U(rng, B, 0.22, 0.40), 0.72 * H)
    na = np.where(rng.uniform(0, 1, B) < 0.6, U(rng, B, 0.03, 0.16), 0.0) * np.minimum(H, 1.0)   # arch under the waist of the sole
    xhf = U(rng, B, 0.64, 0.80)
    hy = np.maximum(0.5 * H, v1 + 0.05)
    top = pts((0 * H, ts + 0.02), (0.035 + 0 * H, toe_h * 0.75), (0.13 + 0 * H, toe_h + 0.02), (0.27 + 0 * H, v1), (xs - 0.07, 0.5 * (v1 + hy)),
              (xs, hy + 0.04), (xs + 0.5 * lean, 0.5 * (hy + H) + 0.03), (xs + lean, H))
    curve = spline(top, 4)
    rest = pts((xb, H + tilt), (1.0 + 0 * H, 0.55 * H), (1.0 + 0 * H, 0.05 + 0 * H), (0.985 + 0 * H, 0 * H), (xhf + 0.015, 0 * H), (xhf, na), (0.58 + 0 * H, na),
               (0.42 + 0 * H, 0 * H), (0.06 + 0 * H, 0 * H), (0.01 + 0 * H, ts * 0.5))
    return [('poly', up(cat(curve, rest)), +1, T)]


# ---------------------------------------------------------------- class 5: sandal
def gen_sandal_flat(B, rng, chunky=False):
    T = np.ones(B, bool)
    tk = U(rng, B, 0.035, 0.10) if not chunky else U(rng, B, 0.06, 0.14)
    ts = U(rng, B, 0.0, 0.05)
    rise = U(rng, B, 0.0, 0.07)                            # slight heel lift
    sole = pts((0 * tk, ts), (0.03 + 0 * tk, ts + tk), (0.4 + 0 * tk, tk), (1.0 + 0 * tk, tk + rise), (1.0 + 0 * tk, 0 * tk), (0.08 + 0 * tk, 0 * tk))
    scene = [('poly', up(sole), +1, T)]
    rlo, rhi = (0.45, 1.0) if not chunky else (0.8, 1.4)
    Ah = tk + U(rng, B, 0.14, 0.38)                        # ankle height
    Ax = U(rng, B, 0.68, 0.9)
    r1 = U(rng, B, rlo, rhi); r2 = U(rng, B, rlo, rhi); r3 = U(rng, B, rlo, rhi)
    pick = rng.uniform(0, 1, (5, B))
    # toe strap: arch over the forefoot
    x0 = U(rng, B, 0.08, 0.25); x1 = x0 + U(rng, B, 0.12, 0.28); th_ = tk + U(rng, B, 0.07, 0.2)
    toe = spline(pts((x0, tk), (0.5 * (x0 + x1) + 0.03, th_), (x1, tk)), 4)
    scene.append(('stroke', up(toe), r1, +1, pick[0] < 0.8))
    # strap from the forefoot back up to the ankle
    xi = U(rng, B, 0.25, 0.55)
    ins = pts((xi, tk), (Ax, Ah))
    scene.append(('stroke', up(ins), r2, +1, pick[1] < 0.6))
    # heel strap up to the ankle
    hl = pts((U(rng, B, 0.9, 1.0), tk + rise), (Ax + U(rng, B, 0.0, 0.1), Ah))
    scene.append(('stroke', up(hl), r2, +1, pick[2] < 0.75))
    # ankle strap, roughly level
    an = pts((Ax - U(rng, B, 0.0, 0.15), Ah + U(rng, B, -0.03, 0.03)), (np.minimum(Ax + U(rng, B, 0.08, 0.25), 1.0), Ah + U(rng, B, -0.03, 0.03)))
    scene.append(('stroke', up(an), r3, +1, pick[3] < 0.6))
    # second band over the instep
    y0 = U(rng, B, 0.3, 0.5); y1 = y0 + U(rng, B, 0.1, 0.25); mh = tk + U(rng, B, 0.1, 0.26)
    mid = spline(pts((y0, tk), (0.5 * (y0 + y1) + 0.04, mh), (y1, tk)), 4)
    scene.append(('stroke', up(mid), r3, +1, pick[4] < 0.45))
    return scene


def gen_sandal_heel(B, rng, wedge=False):
    T = np.ones(B, bool)
    hh = U(rng, B, 0.22, 0.75) if not wedge else U(rng, B, 0.30, 0.65)     # heel height
    tk = U(rng, B, 0.03, 0.07)
    pf = np.where(rng.uniform(0, 1, B) < 0.35, U(rng, B, 0.03, 0.14), 0.0)     # platform under the forefoot
    xb = U(rng, B, 0.28, 0.46)                             # ball of the foot: sole leaves the ground behind it
    xh = U(rng, B, 0.80, 0.90)                             # front of the heel seat
    # upper surface of the sole from toe to heel seat, then back underneath
    upper = spline(pts((0 * hh, pf + tk + 0.01), (0.15 + 0 * hh, pf + tk), (xb, pf + tk), (0.5 * (xb + xh) + 0.02, pf + tk + 0.42 * (hh - pf)), (xh, hh), (1.0 + 0 * hh, hh + 0.01)), 4)
    if wedge:
        under = pts((1.0 + 0 * hh, 0 * hh), (0.06 + 0 * hh, 0 * hh), (0 * hh, 0.02 + 0 * hh))
        scene = [('poly', up(cat(upper, under)), +1, T)]
    else:
        lower = spline(pts((1.0 + 0 * hh, hh - tk), (xh, hh - tk), (0.5 * (xb + xh) + 0.04, pf + 0.42 * (hh - pf) - 0.2 * tk), (xb + 0.02, pf + 0 * hh), (xb + 0.02, 0 * hh)), 4)
        under = pts((0.06 + 0 * hh, 0 * hh), (0 * hh, 0.02 + 0 * hh))
        scene = [('poly', up(cat(upper, lower, under)), +1, T)]
        # the heel itself: stiletto ... block
        wtop = U(rng, B, 0.08, 0.2); wbot = wtop * U(rng, B, 0.2, 1.0)
        xr = U(rng, B, 0.95, 1.0)
        heel = pts((xr - wtop, hh - 0.5 * tk), (1.0 + 0 * hh, hh - 0.5 * tk), (xr, 0 * hh), (xr - wbot, 0 * hh))
        scene.append(('poly', up(heel), +1, T))
    pick = rng.uniform(0, 1, (5, B))
    r1 = U(rng, B, 0.45, 1.1); r2 = U(rng, B, 0.45, 1.0); r3 = U(rng, B, 0.45, 1.0)
    if wedge:                                              # a wedge always carries broad straps over toe and ankle
        pick[:3] = 0.0
        r1 = U(rng, B, 0.8, 1.5); r2 = U(rng, B, 0.8, 1.5)
    base = pf + tk
    # toe strap
    x0 = U(rng, B, 0.06, 0.2); x1 = x0 + U(rng, B, 0.12, 0.26); th_ = base + U(rng, B, 0.08, 0.2)
    toe = spline(pts((x0, base), (0.5 * (x0 + x1) + 0.03, th_), (x1, base)), 4)
    scene.append(('stroke', up(toe), r1, +1, pick[0] < 0.85))
    # ankle point above the heel seat
    Ah = hh + U(rng, B, 0.16, 0.42)
    Ax = U(rng, B, 0.62, 0.82)
    # back strap / heel cup rising from the heel seat
    bk = pts((U(rng, B, 0.93, 1.0), hh), (U(rng, B, 0.9, 1.0), Ah))
    scene.append(('stroke', up(bk), r2, +1, pick[1] < 0.8))
    # ankle strap
    an = pts((Ax, Ah + U(rng, B, -0.05, 0.05)), (0.97 + 0 * hh, Ah))
    scene.append(('stroke', up(an), r2, +1, (pick[2] < 0.75) & (pick[1] < 0.8)))
    # strap from the forefoot up to the ankle
    xi = U(rng, B, 0.15, 0.45)
    ins = pts((xi, base + 0.02), (Ax, Ah))
    scene.append(('stroke', up(ins), r3, +1, (pick[3] < 0.45) & (pick[1] < 0.8) & (pick[2] < 0.75)))
    # band across the instep, landing on the sole further back
    y0 = U(rng, B, 0.3, 0.45); y1 = U(rng, B, 0.5, 0.7)
    f = (y1 - xb) / np.maximum(xh - xb, 0.05)
    mid = spline(pts((y0, base), (0.5 * (y0 + y1), base + U(rng, B, 0.15, 0.3)), (y1, base + np.clip(f, 0, 1) * 0.6 * (hh - pf))), 4)
    scene.append(('stroke', up(mid), r3, +1, pick[4] < 0.4))
    return scene


# ====================================================================== the template bank
# (class, generator, keyword arguments, number of drawings, left-right symmetric?, seed)
BANK = [
    (0, gen_sleeved, {'sleeves': 'short', 'neck': 'crew'}, 10000, True, 100),
    (6, gen_sleeved, {'sleeves': 'short', 'neck': 'shirt'}, 10000, True, 100),      # same seed: same bodies as the T-shirts
    (0, gen_tank, {}, 1500, True, 102),
    (0, gen_cami, {}, 1000, True, 103),
    (1, gen_trousers, {}, 5000, True, 111),
    (1, gen_trousers, {'shorts': True}, 500, True, 112),
    (2, gen_sleeved, {'sleeves': 'long', 'neck': 'crew'}, 10000, True, 120),
    (4, gen_sleeved, {'sleeves': 'long', 'neck': 'coat'}, 10000, True, 120),        # same seed: same bodies as the pullovers
    (6, gen_sleeved, {'sleeves': 'long', 'neck': 'shirt'}, 10000, True, 120),       # same seed again
    (3, gen_dress, {'kind': 'strap'}, 3000, True, 131),
    (3, gen_dress, {'kind': 'sleeveless'}, 3500, True, 132),
    (3, gen_dress, {'kind': 'short'}, 2500, True, 133),
    (3, gen_dress, {'kind': 'long'}, 3000, True, 134),
    (4, gen_coat, {'long': True}, 3500, True, 142),
    (5, gen_sandal_flat, {}, 4000, False, 151),
    (5, gen_sandal_flat, {'chunky': True}, 500, False, 152),
    (5, gen_sandal_heel, {}, 4000, False, 153),
    (5, gen_sandal_heel, {'wedge': True}, 500, False, 154),
    (6, gen_shirt_sleeveless, {}, 1000, True, 163),
    (6, gen_sleeved, {'sleeves': 'mid', 'neck': 'crew'}, 1500, True, 165),
    (6, gen_sleeved, {'sleeves': 'mid', 'neck': 'shirt'}, 1500, True, 166),
    (7, gen_sneaker, {}, 3500, False, 171),
    (7, gen_sneaker, {'white_sole': True}, 2500, False, 173),
    (8, gen_bag, {'kind': 'plain'}, 3000, True, 181),
    (8, gen_bag, {'kind': 'handle'}, 4000, True, 182),
    (8, gen_bag, {'kind': 'strap'}, 2500, True, 183),
    (8, gen_bag, {'kind': 'two'}, 1500, True, 184),
    (9, gen_boot, {}, 6000, False, 191),
]

U_ITEM = 2.3            # the input is divided by its own standard deviation; a typical item pixel is then about 2.3
MASK_U0 = 0.30          # ... and a pixel counts as background below 0.30 (about an eighth of the item's brightness: this
MASK_GAIN = 3.0         #     ignores the faint halo that sharpening leaves around every item), fully "item" from 0.63 on
SIGMA = 1.2             # blur of each template: how far its outline may be off, in pixels
EPS = 0.05              # chance that a pixel disagrees with its template for unmodelled reasons (prints, shadows)
BLOCK = 500             # templates are max-pooled in blocks of this many
SHIFT = 700.0           # added to every template score so that scores are positive and 0 can stand for "no template"
CHUNK = 500

# One extra cue that a silhouette cannot give, with hand-chosen weights (in the same log-odds units as the templates):
# a thin vertical line down the centre of the front (zip, button placket, the gap of an open front) that runs through the
# whole trunk.  Coats mostly have it, button shirts often, pullovers and T-shirts normally not; trousers show the leg gap.
LINE_T0, LINE_T1 = 0.18, 0.45        # line contrast (input / its std) where the cue starts to count / counts fully
LINE_BONUS = [-5.0, 8.0, 0.0, 3.0, 15.0, 0.0, 7.0, 0.0, 0.0, 0.0]


def blur(t):
    """Gaussian blur of (B, 28, 28) arrays, zero outside the frame; shifts and adds only."""
    r = 4
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / SIGMA) ** 2)
    k = k / k.sum()
    B = t.shape[0]
    p = np.zeros((B, N + 2 * r, N))
    p[:, r:r + N] = t
    a = np.zeros_like(t)
    for i in range(2 * r + 1):
        a += k[i] * p[:, i:i + N]
    p = np.zeros((B, N, N + 2 * r))
    p[:, :, r:r + N] = a
    b = np.zeros_like(t)
    for i in range(2 * r + 1):
        b += k[i] * p[:, :, i:i + N]
    return b


def template_logodds(cov):
    """Each rendering on its own -> per-pixel log-odds W and log(1 - p) C, both (B, 28, 28)."""
    t = np.clip(MASK_GAIN * (U_ITEM * cov - MASK_U0), 0.0, 1.0)     # the same soft threshold the network applies to photos
    p = EPS + (1 - 2 * EPS) * np.clip(blur(t), 0.0, 1.0)
    return np.log(p / (1 - p)), np.log(1 - p)


def render_bank(entry):
    cls, fn, kw, n, sym, seed = entry
    rng = np.random.default_rng(seed)
    covs = []
    for i in range(0, n, CHUNK):
        b = min(CHUNK, n - i)
        covs.append(render(fn(b, rng, **kw), b, sym=sym))
    return np.concatenate(covs)


def quadrant_filters(cov, sym):
    """Returns for 'top' and 'bottom': list of (filters (n,14,14), bias (n,), side) with side in {'both','left','right'}.
    Filters are stored in right-quadrant orientation (column 0 touches the centre line)."""
    W, C = template_logodds(cov)
    out = {}
    for name, rows in (('top', slice(0, 14)), ('bottom', slice(14, 28))):
        wr, cr = W[:, rows, 14:], C[:, rows, 14:]
        wl, cl = W[:, rows, :14][:, :, ::-1], C[:, rows, :14][:, :, ::-1]
        if sym:
            out[name] = [(wr, cr.sum(axis=(1, 2)), 'both')]
        else:
            out[name] = [(wl, cl.sum(axis=(1, 2)), 'left'), (wr, cr.sum(axis=(1, 2)), 'right')]
    return out


def f32(a):
    return np.asarray(a, np.float32)


def conv_layer(name, inp, out, k, pad=0, stride=1):
    return {"name": name, "op": "conv", "inputs": [inp], "out": out, "kernel": [k, k], "stride": stride, "padding": pad}


def front_end(layers, weights):
    """image -> u = image / its own std (zero stays zero) -> soft mask in [0, 1]."""
    layers += [{"name": "z", "op": "norm", "inputs": ["image"]},
               conv_layer("zneg", "z", 1, 1),
               {"name": "zmax", "op": "maxpool", "inputs": ["zneg"], "size": [28, 28], "stride": [28, 28]},
               {"name": "zb", "op": "dense", "inputs": ["zmax"], "out": 784},
               {"name": "zbr", "op": "reshape", "inputs": ["zb"], "shape": [1, 28, 28]},
               {"name": "u", "op": "add", "inputs": ["z", "zbr"]},
               conv_layer("m1", "u", 2, 1),
               {"name": "m1r", "op": "relu", "inputs": ["m1"]},
               conv_layer("mask", "m1r", 1, 1)]
    weights['zneg.W'] = f32([-1.0]).reshape(1, 1, 1, 1); weights['zneg.b'] = f32([0.0])
    weights['zb.W'] = np.ones((784, 1), np.float32); weights['zb.b'] = np.zeros(784, np.float32)
    weights['m1.W'] = np.full((2, 1, 1, 1), MASK_GAIN, np.float32)
    weights['m1.b'] = f32([-MASK_GAIN * MASK_U0, -MASK_GAIN * MASK_U0 - 1.0])
    weights['mask.W'] = f32([1.0, -1.0]).reshape(1, 2, 1, 1); weights['mask.b'] = f32([0.0])


def line_head(layers, weights):
    """Thin vertical line (brighter or darker than both sides) in the centre columns, present in each of three row bands."""
    W = np.zeros((1, 1, 5, 5), np.float32); W[0, 0, 2] = [-0.5, 0.0, 1.0, 0.0, -0.5]
    weights['lh.W'] = W; weights['lh.b'] = f32([0.0])
    weights['lhp.W'] = f32([1.0, -1.0]).reshape(2, 1, 1, 1); weights['lhp.b'] = np.zeros(2, np.float32)
    weights['lhn.W'] = f32([[-1.0, 0.0], [0.0, -1.0]]).reshape(2, 2, 1, 1); weights['lhn.b'] = np.zeros(2, np.float32)
    S = np.zeros((8, 2 * 2 * 28), np.float32)
    i = 0
    for ch in range(2):
        for c in (12, 13, 14, 15):
            S[i, ch * 56 + 28 + c] = -1.0; i += 1          # row 1 of the pooled map = bands of rows 6-11, 12-17, 18-23
    weights['lsel.W'] = S; weights['lsel.b'] = np.zeros(8, np.float32)
    s = 1.0 / (LINE_T1 - LINE_T0)
    weights['lf.W'] = f32([[s], [s]]); weights['lf.b'] = f32([-s * LINE_T0, -s * LINE_T0 - 1.0])
    weights['lcls.W'] = f32([[b, -b] for b in LINE_BONUS]); weights['lcls.b'] = np.zeros(10, np.float32)
    layers += [conv_layer("lh", "u", 1, 5, pad=2),
               {"name": "lhs", "op": "avgpool", "inputs": ["lh"], "size": [6, 1], "stride": [6, 1]},
               conv_layer("lhp", "lhs", 2, 1),
               {"name": "lhpr", "op": "relu", "inputs": ["lhp"]},
               conv_layer("lhn", "lhpr", 2, 1),
               {"name": "lhm", "op": "maxpool", "inputs": ["lhn"], "size": [3, 1], "stride": [1, 1]},
               {"name": "lsel", "op": "dense", "inputs": ["lhm"], "out": 8},
               {"name": "lselr", "op": "reshape", "inputs": ["lsel"], "shape": [1, 1, 8]},
               {"name": "lmax", "op": "maxpool", "inputs": ["lselr"], "size": [1, 8], "stride": [1, 8]},
               {"name": "lf", "op": "dense", "inputs": ["lmax"], "out": 2},
               {"name": "lfr", "op": "relu", "inputs": ["lf"]},
               {"name": "lcls", "op": "dense", "inputs": ["lfr"], "out": 10}]
    return "lcls"


def build(out_dir):
    groups = {'top': [], 'bottom': []}          # per quadrant type: list of (class, side, filters, bias)
    for entry in BANK:
        cov = render_bank(entry)
        q = quadrant_filters(cov, entry[4])
        for name in groups:
            for w, b, side in q[name]:
                groups[name].append((entry[0], side, w.astype(np.float32), b.astype(np.float32)))
    weights = {}
    layers = []
    front_end(layers, weights)
    finals = []
    for name, r0 in (('top', 0), ('bottom', 14)):
        tag = 'T' if name == 'top' else 'B'
        # rearrange half of the mask into (1, 14, 28): columns 0..13 = left quadrant mirrored, 14..27 = right quadrant
        P = np.zeros((392, 784), np.float32)
        for r in range(14):
            for c in range(28):
                src_c = 13 - c if c < 14 else c
                P[r * 28 + c, (r0 + r) * 28 + src_c] = 1.0
        weights[f'q{tag}.W'] = P
        weights[f'q{tag}.b'] = np.zeros(392, np.float32)
        Wf = np.concatenate([g[2] for g in groups[name]])[:, None]
        bf = np.concatenate([g[3] for g in groups[name]]) + np.float32(SHIFT)
        K = Wf.shape[0]
        assert K % BLOCK == 0
        weights[f'bank{tag}.W'] = Wf
        weights[f'bank{tag}.b'] = bf
        nb = K // BLOCK
        # which blocks may be used for which (class, side)
        slots = {(k, s): [] for k in range(10) for s in range(2)}
        j = 0
        for cls, side, w, b in groups[name]:
            assert len(w) % BLOCK == 0
            for _ in range(len(w) // BLOCK):
                if side in ('both', 'left'):
                    slots[(cls, 0)].append(j * 2 + 0)
                if side in ('both', 'right'):
                    slots[(cls, 1)].append(j * 2 + 1)
                j += 1
        R = max(len(v) for v in slots.values())
        S = np.zeros((20 * R, nb * 2), np.float32)
        for (k, s), v in slots.items():
            for i, src in enumerate(v):
                S[(k * 2 + s) * R + i, src] = 1.0
        weights[f'scat{tag}.W'] = S
        weights[f'scat{tag}.b'] = np.zeros(20 * R, np.float32)
        Cm = np.zeros((10, 20), np.float32)
        for k in range(10):
            Cm[k, 2 * k] = 1.0; Cm[k, 2 * k + 1] = 1.0
        weights[f'cls{tag}.W'] = Cm
        weights[f'cls{tag}.b'] = np.full(10, -2 * SHIFT, np.float32)
        layers += [
            {"name": f"q{tag}", "op": "dense", "inputs": ["mask"], "out": 392},
            {"name": f"q{tag}r", "op": "reshape", "inputs": [f"q{tag}"], "shape": [1, 14, 28]},
            conv_layer(f"bank{tag}", f"q{tag}r", K, 14, stride=14),
            {"name": f"bank{tag}r", "op": "reshape", "inputs": [f"bank{tag}"], "shape": [1, K, 2]},
            {"name": f"pool{tag}", "op": "maxpool", "inputs": [f"bank{tag}r"], "size": [BLOCK, 1], "stride": [BLOCK, 1]},
            {"name": f"scat{tag}", "op": "dense", "inputs": [f"pool{tag}"], "out": 20 * R},
            {"name": f"scat{tag}r", "op": "reshape", "inputs": [f"scat{tag}"], "shape": [1, 20, R]},
            {"name": f"best{tag}", "op": "maxpool", "inputs": [f"scat{tag}r"], "size": [1, R], "stride": [1, R]},
            {"name": f"cls{tag}", "op": "dense", "inputs": [f"best{tag}"], "out": 10},
        ]
        finals.append(f"cls{tag}")
    finals.append(line_head(layers, weights))
    layers.append({"name": "scores", "op": "add", "inputs": finals})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'network.json'), 'w') as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, 'weights.npz'), **weights)


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
