"""Builds network.json and weights.npz for the clothing classifier. Reads no files: every number comes from this code.

Idea: the network turns the image into a soft silhouette mask (hand-chosen grey-level ramp), blurs it with a Gaussian,
standardises it (norm) and correlates it with a bank of silhouette templates. Every template is one drawing produced by
the parametric shape models below (T-shirt = body + short sleeves, trousers = waist + two legs, boot = sole + shaft ...),
typed from general knowledge of what these items look like in a shop listing. Each drawing is its own template,
normalised by itself alone. The class score is the best correlation over that class's templates (maxpool).
Two hand-written rules are added to those scores: an open-shoe rule (a patchy picture is a sandal, a smooth one is not)
and a small handicap for the shirt class. Their constants are chosen by hand and explained where they are defined.
"""
import sys, json
import numpy as np

SIZE = 28
SS = 4                      # sub-rows per pixel row in the rasteriser
N_PER_CLASS = 6000          # drawings per class
SEED = 20240611
RAMP_LO, RAMP_HI = 0.05, 0.15   # grey level where the silhouette mask starts / saturates (hand-chosen)
SIGMA = 1.0                 # Gaussian blur of the mask, in pixels (hand-chosen)
KS = 7                      # blur kernel size
DRAW_BRIGHTNESS = 0.6       # grey level the drawings are painted with before they go through the same ramp
# Open-shoe rule (hand-chosen). A sandal is straps and gaps, so inside its outline the picture is patchy; sneakers and
# boots are closed surfaces and smooth. Patchiness = sum |image - smoothed image| divided by sum image, which does not
# depend on how bright the item is. Worked out by hand for ideal shapes: a sharp outline adds about 0.8 to 1.0 x its
# length x brightness, so a closed shoe (outline about 70 pixels, area about 225) gives about 0.3 to 0.4 with a sole
# seam and some detail, while straps one to two pixels wide with gaps between them give 0.6 and more. The limit is
# put between the two; above it the sandal score gains, below it the sandal score loses.
# Mildly patchy pictures are weak evidence and very patchy ones strong evidence (closed shoes almost never reach
# them), so the rule has two steps.
ROUGH_RATIO = 0.52          # patchiness above which a shoe counts as open; below it, as closed
ROUGH_RATIO_HIGH = 0.62     # patchiness above which it counts as clearly open
ROUGH_SCALE = 12.0          # grey-level sum over which each step goes from nothing to full strength
SANDAL_BONUS = 0.04         # first step, in units of correlation: added above ROUGH_RATIO, subtracted below it
SANDAL_BONUS_HIGH = 0.04    # second step: added on top above ROUGH_RATIO_HIGH
# Prior (hand-chosen): seen as a silhouette, a shirt nearly always has a more common look-alike (short sleeves are
# mostly T-shirts, long angled sleeves mostly pullovers, long straight sleeves mostly coats), so a shirt drawing has
# to match clearly better than the others to win.
SHIRT_HANDICAP = 0.015      # in units of correlation


# ----------------------------------------------------------------------------------------------------------------------
# rasteriser: polygons (pixel coordinates, x right, y down) -> 28 x 28 coverage, exact along x, SS sub-rows along y
# ----------------------------------------------------------------------------------------------------------------------
def _raster_batch(P):
    """P: (M, K, 2) with K even. Returns (M, SIZE, SIZE) coverage in 0..1 (even-odd fill)."""
    M, K, _ = P.shape
    ys = (np.arange(SIZE * SS) + 0.5) / SS
    x0, y0 = P[:, :, 0], P[:, :, 1]
    x1, y1 = np.roll(x0, -1, axis=1), np.roll(y0, -1, axis=1)
    Y = ys[None, :, None]
    cross = (y0[:, None, :] <= Y) != (y1[:, None, :] <= Y)
    dy = y1 - y0
    slope = (x1 - x0) / np.where(dy == 0, 1.0, dy)
    xc = x0[:, None, :] + (Y - y0[:, None, :]) * slope[:, None, :]
    xc = np.where(cross, xc, 1e9)
    xc.sort(axis=2)
    lo, hi = xc[:, :, 0::2], xc[:, :, 1::2]
    j = np.arange(SIZE, dtype=np.float64)
    cov = (np.clip(hi[..., None] - j, 0.0, 1.0) - np.clip(lo[..., None] - j, 0.0, 1.0)).sum(axis=2)
    return np.clip(cov, 0.0, 1.0).reshape(M, SIZE, SS, SIZE).mean(axis=2)


def render(shapes):
    """shapes: list (one per drawing) of lists of (sign, (K, 2) array in pixel coords). sign +1 adds, -1 cuts out.
    Returns (len(shapes), SIZE, SIZE) coverage: union of the added parts times (1 - union of the cut parts)."""
    n = len(shapes)
    pos = np.zeros((n, SIZE, SIZE)); neg = np.zeros((n, SIZE, SIZE))
    buckets = {}
    for i, parts in enumerate(shapes):
        for sign, pts in parts:
            k = len(pts)
            kb = 8
            while kb < k: kb *= 2
            buckets.setdefault(kb, []).append((i, sign, pts))
    for kb in sorted(buckets):
        items = buckets[kb]
        chunk = max(1, 60000 // kb)
        for s in range(0, len(items), chunk):
            part = items[s:s + chunk]
            P = np.empty((len(part), kb, 2))
            for r, (_, _, pts) in enumerate(part):
                P[r, :len(pts)] = pts
                P[r, len(pts):] = pts[-1]
            cov = _raster_batch(P)
            for r, (i, sign, _) in enumerate(part):
                if sign > 0: np.maximum(pos[i], cov[r], out=pos[i])
                else: np.maximum(neg[i], cov[r], out=neg[i])
    return pos * (1.0 - neg)


# ----------------------------------------------------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------------------------------------------------
def A(*pts):
    return np.array(pts, dtype=np.float64)


def mirror(right):
    """right: points of the right half listed from the top centre line to the bottom centre line -> closed outline."""
    R = np.asarray(right, dtype=np.float64)
    L = R[::-1].copy(); L[:, 0] *= -1
    return np.vstack([R, L])


def flipx(pts):
    P = np.asarray(pts, dtype=np.float64).copy(); P[:, 0] *= -1
    return P


def chaikin(pts, times=1):
    """Corner cutting on a closed outline (rounds the corners)."""
    P = np.asarray(pts, dtype=np.float64)
    for _ in range(times):
        Q = np.roll(P, -1, axis=0)
        P = np.stack([0.75 * P + 0.25 * Q, 0.25 * P + 0.75 * Q], axis=1).reshape(-1, 2)
    return P


def stroke(pts, w):
    """A line through pts with thickness w, as one outline."""
    P = np.asarray(pts, dtype=np.float64)
    d = np.diff(P, axis=0)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    vn = np.zeros_like(P)
    vn[0], vn[-1] = nrm[0], nrm[-1]
    if len(P) > 2:
        m = nrm[:-1] + nrm[1:]
        m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-6)
        c = (m * nrm[:-1]).sum(axis=1, keepdims=True)
        vn[1:-1] = m / np.maximum(c, 0.5)
    return np.vstack([P + vn * (w / 2), (P - vn * (w / 2))[::-1]])


def arch(xa, xb, y0, h, n=9):
    """Half-ellipse line from (xa, y0) over the top (y0 - h, y is down) to (xb, y0)."""
    t = np.linspace(np.pi, 0.0, n)
    cx, rx = (xa + xb) / 2, (xb - xa) / 2
    return np.stack([cx + rx * np.cos(t), y0 - h * np.sin(t)], axis=1)


def place(parts, rng, yup=False, rot=0.0):
    """Put a drawing into the 28 x 28 frame the way the photos were prepared: trimmed to the item, the longest side
    scaled to (about) 28 pixels, centred. rot: small rotation in degrees applied first."""
    out = []
    c, s = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    for sign, p in parts:
        p = np.asarray(p, dtype=np.float64).copy()
        if yup: p[:, 1] *= -1
        p = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], axis=1)
        out.append((sign, p))
    allp = np.vstack([p for sign, p in out if sign > 0])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    w, h = hi - lo
    long_side = rng.uniform(26.8, 28.0)
    sc = long_side / max(w, h)
    ctr = (lo + hi) / 2
    off = np.array([SIZE / 2, SIZE / 2], dtype=np.float64)
    jit = rng.uniform(-0.5, 0.5)
    if w >= h: off[1] += jit      # the short side sits on whole pixels, so its centre can be half a pixel off
    else: off[0] += jit
    return [(sign, (p - ctr) * sc + off) for sign, p in out]


# ----------------------------------------------------------------------------------------------------------------------
# shape models. Garments: x to the right from the centre line, y down, body length (neck point to hem) = 1.
# ----------------------------------------------------------------------------------------------------------------------
def sleeve_band(S, th, L, wt, cu):
    """A sleeve as a tapering band: outer edge from the shoulder tip S, th degrees away from straight down, length L,
    width wt at the top and cu at the cuff."""
    t = np.deg2rad(th)
    d = np.array([np.sin(t), np.cos(t)]); n = np.array([-np.cos(t), np.sin(t)])
    S = np.asarray(S, dtype=np.float64)
    E1 = S + L * d
    return A(S - 0.02 * d, E1, E1 + cu * n, S + wt * n)


def upper_parts(nw, sw, sd, nd, tc, ww, hw, rise, L=0.0, th=0.0, wt=0.0, cu=0.0, collar=0.0, hood=0.0, neckcol=0.0,
                notch=None):
    """Front view of a top on an invisible mannequin or laid flat.
    nw neck half-width, sw shoulder half-width, sd shoulder drop, nd depth of the neckline dip, tc half-width of the
    torso below the armpit, ww waist half-width, hw hem half-width, rise how much the side seams end above the centre
    of the hem (shirt tail), sleeve length L, angle th from straight down (degrees), widths wt (top) and cu (cuff);
    collar / hood / neckcol: height of a collar, hood or roll neck above the neck points.
    notch: (half-width, depth) of the open V between the two sides of a collar, cut out of the top centre."""
    body = mirror([(0.0, nd), (nw * 0.55, nd * 0.72), (nw, 0.0), (sw, sd), (tc, sd + 0.14), (ww, 0.62), (hw, 1.0 - rise),
                   (hw * 0.62, 1.0 - rise * 0.3), (0.0, 1.0)])
    parts = [(1, body)]
    if L > 0:
        slv = sleeve_band((sw, sd), th, L, wt, cu)
        parts += [(1, slv), (1, flipx(slv))]
    if collar > 0:
        parts.append((1, mirror([(0.0, -collar), (nw * 1.05, -collar), (nw * 1.25, sd * 0.5), (0.0, sd * 0.5 + 0.02)])))
    if neckcol > 0:
        parts.append((1, mirror([(0.0, -neckcol), (nw * 0.9, -neckcol), (nw, 0.02), (0.0, 0.03)])))
    if hood > 0:
        parts.append((1, arch(-nw * 1.25, nw * 1.25, 0.03, hood + 0.03, n=8)))
    if notch is not None:
        top = -collar - 0.02
        parts.append((-1, A((-notch[0], top), (notch[0], top), (0.0, -collar + notch[1]))))
    return parts


def gen_tshirt(rng):
    U = rng.uniform
    kind = rng.random()
    if kind < 0.74:                                   # short-sleeved T-shirt
        sw = U(0.27, 0.40); tc = sw * U(0.95, 1.15); wt = U(0.18, 0.27)
        parts = upper_parts(nw=U(0.10, 0.16), sw=sw, sd=U(0.03, 0.08), nd=U(0.0, 0.06), tc=tc,
                            ww=tc * U(0.92, 1.0), hw=tc * U(0.9, 1.08), rise=U(0.0, 0.03),
                            L=U(0.13, 0.36), th=U(35, 80), wt=wt, cu=wt * U(0.75, 1.0))
    elif kind < 0.94:                                 # sleeveless top with straps
        cw = U(0.25, 0.38); so = U(0.13, 0.25); st = U(0.04, 0.11); si = max(so - st, 0.04)
        nd = U(0.08, 0.30); ay = U(0.24, 0.40); ww = cw * U(0.88, 1.0); hw = cw * U(0.88, 1.12)
        body = mirror([(0.0, nd), (si * 0.6, nd * 0.9), (si, nd * 0.35), (si, 0.0), (so, 0.0), (so + 0.01, ay * 0.55),
                       (cw, ay), (ww, 0.62), (hw, 1.0), (0.0, 1.0)])
        parts = [(1, body)]
    else:                                             # wide, short top
        sw = U(0.38, 0.52); tc = sw * U(0.95, 1.1); wt = U(0.2, 0.32)
        parts = upper_parts(nw=U(0.13, 0.20), sw=sw, sd=U(0.03, 0.09), nd=U(0.0, 0.08), tc=tc,
                            ww=tc * U(0.94, 1.0), hw=tc * U(0.9, 1.05), rise=0.0,
                            L=U(0.12, 0.32), th=U(40, 85), wt=wt, cu=wt * U(0.75, 1.0))
    return place(parts, rng, rot=U(-2, 2))


def long_sleeve_length(rng, sw, sd, th, lo, hi, ymax):
    """Sleeve length from the shoulder width (an arm is about 2.7 half-shoulders long), not ending below ymax."""
    return min(2.7 * sw * rng.uniform(lo, hi), (ymax - sd) / np.cos(np.deg2rad(th)))


def gen_pullover(rng):
    U = rng.uniform
    sw = U(0.30, 0.42); tc = sw * U(0.84, 1.08); sd = U(0.03, 0.09)
    th = U(6, 30) if rng.random() < 0.8 else U(2, 6)   # mostly shown with the sleeves angled away from the body
    extra = rng.random()
    parts = upper_parts(nw=U(0.11, 0.17), sw=sw, sd=sd, nd=U(0.0, 0.05), tc=tc,
                        ww=tc * U(0.93, 1.0), hw=tc * U(0.84, 1.0), rise=0.0,
                        L=long_sleeve_length(rng, sw, sd, th, 0.86, 1.08, 1.1), th=th, wt=U(0.17, 0.27), cu=U(0.09, 0.15),
                        hood=U(0.05, 0.11) if extra < 0.12 else 0.0,
                        neckcol=U(0.04, 0.09) if 0.12 <= extra < 0.24 else 0.0)
    return place(parts, rng, rot=U(-2, 2))


def gen_coat(rng):
    U = rng.uniform
    extra = rng.random()
    sd = U(0.03, 0.08)
    if rng.random() < 0.55:                           # long coat: narrow for its length, sleeves end well above the hem
        sw = U(0.22, 0.31); tc = sw * U(0.85, 1.06); hw = tc * U(0.95, 1.3); th = U(0, 9)
        wt = U(0.14, 0.22); cu = U(0.09, 0.14)
    else:                                             # jacket: about as wide as long
        sw = U(0.30, 0.40); tc = sw * U(0.85, 1.06); hw = tc * U(0.9, 1.06); th = U(0, 12)
        wt = U(0.17, 0.26); cu = U(0.10, 0.17)
    collar = hood = 0.0; notch = None
    if extra < 0.25: collar = U(0.02, 0.06)                                   # closed stand-up collar
    elif extra < 0.50: hood = U(0.05, 0.12)                                   # hood
    elif extra < 0.78: collar = U(0.02, 0.06); notch = (U(0.03, 0.08), U(0.06, 0.18))   # open collar
    elif extra < 0.86: collar = U(0.01, 0.04); notch = (U(0.05, 0.10), U(0.18, 0.32))   # lapels, deep V
    parts = upper_parts(nw=U(0.10, 0.16), sw=sw, sd=sd, nd=U(0.0, 0.03), tc=tc,
                        ww=tc * U(0.9, 1.02), hw=hw, rise=0.0,
                        L=long_sleeve_length(rng, sw, sd, th, 0.88, 1.08, 1.08), th=th, wt=wt, cu=cu,
                        collar=collar, hood=hood, notch=notch)
    return place(parts, rng, rot=U(-2, 2))


def gen_shirt(rng):
    U = rng.uniform
    kind = rng.random()
    sd = U(0.04, 0.08)
    collar = U(0.02, 0.05)
    notch = (U(0.025, 0.06), U(0.05, 0.14)) if rng.random() < 0.5 else None   # collar open at the top button
    if kind < 0.68:                                   # long-sleeved button shirt: slim sleeves, collar, shirt tail
        sw = U(0.26, 0.34); tc = sw * U(0.9, 1.1); th = U(4, 28)
        parts = upper_parts(nw=U(0.09, 0.13), sw=sw, sd=sd, nd=0.0, tc=tc,
                            ww=tc * U(0.92, 1.0), hw=tc * U(0.95, 1.08), rise=U(0.02, 0.10),
                            L=long_sleeve_length(rng, sw, sd, th, 0.9, 1.08, 1.05), th=th, wt=U(0.13, 0.20),
                            cu=U(0.06, 0.10), collar=collar, notch=notch)
    elif kind < 0.92:                                 # short-sleeved shirt: longer and slimmer than a T-shirt, collar
        sw = U(0.26, 0.33); tc = sw * U(0.95, 1.1); wt = U(0.17, 0.24)
        parts = upper_parts(nw=U(0.09, 0.13), sw=sw, sd=sd, nd=0.0, tc=tc,
                            ww=tc * U(0.92, 1.0), hw=tc * U(0.95, 1.06), rise=U(0.02, 0.09),
                            L=U(0.18, 0.34), th=U(22, 55), wt=wt, cu=wt * U(0.75, 1.0), collar=collar, notch=notch)
    else:                                             # sleeveless blouse with a collar
        sw = U(0.20, 0.30); tc = sw * U(1.0, 1.25)
        parts = upper_parts(nw=U(0.09, 0.13), sw=sw, sd=U(0.03, 0.07), nd=0.0, tc=tc,
                            ww=tc * U(0.9, 1.0), hw=tc * U(0.95, 1.1), rise=U(0.0, 0.08), collar=collar, notch=notch)
    return place(parts, rng, rot=U(-2, 2))


def gen_dress(rng):
    U = rng.uniform
    bw = U(0.14, 0.27)                                # bust half-width, dress length = 1
    wy = U(0.28, 0.44); ww = bw * U(0.72, 0.97)
    hw = min(bw * U(0.85, 2.2), U(0.30, 0.45))
    f = U(0.35, 0.85)                                 # how early the skirt reaches its width
    hy = wy + (1.0 - wy) * U(0.3, 0.55)
    hipw = ww + (hw - ww) * f
    ay = U(0.14, 0.24)
    lower = [(bw, ay), (ww, wy), (hipw, hy), (hw, 1.0), (hw * 0.6, 1.0 + U(0.0, 0.02)), (0.0, 1.0 + U(0.0, 0.03))]
    kind = rng.random()
    parts = []
    if kind < 0.36:                                   # thin straps
        so = bw * U(0.55, 0.92); st = U(0.025, 0.07); si = max(so - st, 0.02); nd = U(0.07, 0.2)
        top = [(0.0, nd), (si * 0.6, nd * 0.92), (si, nd * 0.4), (si, 0.0), (so, 0.0), (so + 0.005, ay * 0.6)]
    elif kind < 0.62:                                 # sleeveless with shoulders
        sw = bw * U(0.72, 1.0); nw = sw * U(0.4, 0.7); nd = U(0.02, 0.14)
        top = [(0.0, nd), (nw * 0.6, nd * 0.8), (nw, 0.0), (sw, U(0.02, 0.05)), (sw * 0.96, ay * 0.6)]
    elif kind < 0.80:                                 # short sleeves
        sw = bw * U(0.85, 1.05); nw = sw * U(0.4, 0.65); nd = U(0.01, 0.10); sd = U(0.02, 0.05)
        top = [(0.0, nd), (nw * 0.6, nd * 0.8), (nw, 0.0), (sw, sd)]
        wt = U(0.10, 0.16)
        slv = sleeve_band((sw, sd), U(25, 70), U(0.07, 0.2), wt, wt * U(0.7, 1.0))
        parts += [(1, slv), (1, flipx(slv))]
    elif kind < 0.95:                                 # long sleeves
        sw = bw * U(0.85, 1.05); nw = sw * U(0.4, 0.65); nd = U(0.01, 0.10); sd = U(0.02, 0.05)
        top = [(0.0, nd), (nw * 0.6, nd * 0.8), (nw, 0.0), (sw, sd)]
        slv = sleeve_band((sw, sd), U(2, 22), U(0.38, 0.66), U(0.09, 0.14), U(0.04, 0.08))
        parts += [(1, slv), (1, flipx(slv))]
    else:                                             # strapless
        top = [(0.0, ay * U(0.3, 0.8)), (bw * 0.5, ay * 0.3), (bw * 0.95, ay * 0.45)]
    parts.append((1, mirror(top + lower)))
    return place(parts, rng, rot=U(-1.5, 1.5))


def gen_trouser(rng):
    U = rng.uniform
    if rng.random() < 0.9:                            # front view, two legs
        hp = U(0.17, 0.27); ww = hp * U(0.8, 1.0); hpy = U(0.14, 0.24); cy = U(0.24, 0.36)
        lw = hp * U(0.5, 0.95)                        # leg width at the hem
        gi = max(min(U(0.0, 0.09), 1.12 * hp - lw), 0.0)   # inner edge of the leg at the hem
        if rng.random() < 0.3: gi *= 0.2              # legs (almost) touching
        ho = gi + lw                                  # outer edge of the leg at the hem
        kn = U(0.55, 0.7); t = (kn - hpy) / (1.0 - hpy)
        ko = hp + (ho - hp) * t + U(-0.02, 0.015)     # knee, outer
        ki = gi * (kn - cy) / (1.0 - cy)              # knee, inner
        wb = U(0.0, 0.02)
        body = mirror([(0.0, 0.0), (ww, 0.0), (ww + wb, 0.04), (hp, hpy), (ko, kn), (ho, 1.0), (gi, 1.0), (ki, kn),
                       (0.0, cy)])
    else:                                             # folded in half / side view: one tapering leg
        wt = U(0.11, 0.17); hp = wt * U(1.0, 1.25); wk = hp * U(0.6, 0.85); wh = wk * U(0.75, 1.05)
        sh = U(-0.03, 0.03)
        body = A((-wt, 0.0), (wt, 0.0), (hp, 0.2), (wk + sh, 0.62), (wh + sh, 1.0), (-wh + sh, 1.0), (-wk + sh, 0.62),
                 (-hp, 0.2))
    return place([(1, body)], rng, rot=U(-1.5, 1.5))


# Shoes: side view, toe to the left. x from 0 (toe) to 1 (heel), y UP from the ground, shoe length = 1.
def gen_sneaker(rng):
    U = rng.uniform
    high = rng.random() < 0.22
    xt = U(0.56, 0.70)
    if high:
        ht = U(0.50, 0.72); hc = ht - U(0.0, 0.07); hh = hc + U(-0.02, 0.05)
    else:
        ht = U(0.33, 0.50); hc = ht - U(0.05, 0.14); hh = hc + U(0.0, 0.08)
    toe = U(0.10, 0.16); ts = U(0.0, 0.035)
    v1 = toe + U(0.03, 0.07); v2 = v1 + (ht - v1) * U(0.3, 0.5)
    out = A((0.0, 0.05 + ts), (0.03, toe), (0.16, v1), (0.36, v2), (xt - 0.10, ht - U(0.03, 0.09)), (xt, ht),
            (xt + 0.09, max(hc, ht - 0.05)), (0.86, hc), (0.97, hh), (1.0, hh * 0.55), (0.985, 0.02), (0.9, 0.0),
            (0.16, 0.0), (0.04, ts))
    return place([(1, chaikin(out, 1))], rng, yup=True, rot=U(-3, 3))


def gen_boot(rng):
    U = rng.uniform
    H = U(0.70, 1.15)                                  # shaft height relative to the shoe length
    xs = U(0.40, 0.60)                                 # where the front of the shaft starts
    toe = U(0.15, 0.24); v1 = toe + U(0.03, 0.08); ins = U(0.40, 0.56)
    xf = xs + U(-0.04, 0.06); xb = U(0.93, 1.0); tilt = U(-0.06, 0.08)
    an = U(0.04, 0.16) if rng.random() < 0.45 else 0.0   # notch under the arch (shoe with a heel)
    hx = U(0.66, 0.78)
    top = [(0.0, 0.07), (0.03, toe), (0.16, v1), (0.32, v1 + (ins - v1) * 0.45), (xs, ins),
           (xf, 0.5 * (ins + H)), (xf + U(-0.02, 0.02), H), (xb, H + tilt), (1.0, 0.5 * H), (1.0, 0.15), (0.985, 0.0)]
    if an > 0:
        bottom = [(hx, 0.0), (hx - 0.03, an), (0.45, an * 0.5), (0.36, 0.0), (0.14, 0.0), (0.03, 0.025)]
    else:
        bottom = [(0.14, 0.0), (0.03, 0.025)]
    return place([(1, chaikin(A(*(top + bottom)), 1))], rng, yup=True, rot=U(-3, 3))


def band(xa, xb, ybase_a, ybase_b, xta, xtb, yta, ytb):
    """A strap seen from the side: a four-cornered band from the sole (xa..xb) up to the top of the foot (xta..xtb)."""
    return A((xa, ybase_a), (xb, ybase_b), (xtb, ytb), (xta, yta))


def gen_sandal(rng):
    U = rng.uniform
    kind = rng.random()
    parts = []
    if kind < 0.30:                                   # flat sandal with straps
        ts = U(0.04, 0.09); hl = U(0.0, 0.07); sp = U(0.0, 0.03)
        parts.append((1, A((0.0, sp), (0.2, 0.0), (1.0, 0.0), (1.0, ts + hl), (0.55, ts), (0.02, ts + sp))))
        sw = U(0.04, 0.085)
        a = U(0.10, 0.22); wd = U(0.07, 0.17); h = U(0.08, 0.16); sl = U(0.0, 0.10)
        parts.append((1, band(a, a + wd, ts, ts, a + sl, a + sl + wd * 0.8, ts + h, ts + h * 1.15)))
        ah = ts + hl + U(0.22, 0.42); ax = U(0.58, 0.74)
        if rng.random() < 0.8: parts.append((1, stroke([(ax, ah), (0.96, ah + U(-0.03, 0.03))], sw)))      # ankle strap
        parts.append((1, stroke([(U(0.9, 0.97), ts), (0.96, ah)], sw * U(1.0, 2.5))))                       # heel strap
        if rng.random() < 0.7: parts.append((1, stroke([(ax, ah), (U(0.45, 0.62), ts)], sw)))              # instep strap
        if rng.random() < 0.4: parts.append((1, stroke([(a + sl + wd * 0.4, ts + h), (ax, ah)], sw * 0.8)))  # T strap
    elif kind < 0.60:                                 # high heel
        p = U(0.03, 0.10); hs = U(0.25, 0.62); xb = U(0.33, 0.46); t = U(0.045, 0.08)
        parts.append((1, A((0.0, 0.0), (xb + 0.04, 0.0), (xb + 0.08, p), (0.0, p + U(0.0, 0.02)))))
        parts.append((1, stroke([(xb - 0.05, p * 0.6), (xb + 0.1, p + 0.03), (0.8, hs - 0.03), (1.0, hs)], t)))
        bw = U(0.03, 0.17)
        parts.append((1, A((U(0.78, 0.88), hs - 0.02), (1.0, hs), (1.0, 0.0), (1.0 - bw, 0.0))))
        sw = U(0.04, 0.08)
        a = U(0.05, 0.18); wd = U(0.07, 0.2); h = U(0.07, 0.15); sl = U(0.0, 0.08)
        parts.append((1, band(a, a + wd, p, p, a + sl, a + sl + wd * 0.8, p + h, p + h * 1.2)))
        ha = U(0.14, 0.36)
        if rng.random() < 0.75:
            parts.append((1, stroke([(U(0.66, 0.78), hs + ha - U(0.0, 0.04)), (0.99, hs + ha)], sw)))
            parts.append((1, stroke([(0.97, hs), (0.985, hs + ha)], sw * U(1.0, 2.5))))
            if rng.random() < 0.5: parts.append((1, stroke([(0.72, hs + ha), (U(0.5, 0.62), p + (hs - p) * 0.35)], sw)))
        else:
            parts.append((1, A((0.84, hs - 0.02), (1.0, hs), (1.0, hs + ha * 0.6), (0.9, hs + ha * 0.5))))
    elif kind < 0.70:                                 # wedge
        p = U(0.03, 0.10); hs = U(0.22, 0.5); xb = U(0.3, 0.45)
        parts.append((1, A((0.0, 0.0), (1.0, 0.0), (1.0, hs), (0.8, hs - 0.04), (xb, p), (0.0, p))))
        sw = U(0.04, 0.08)
        a = U(0.06, 0.18); wd = U(0.08, 0.2); h = U(0.08, 0.15); sl = U(0.0, 0.08)
        parts.append((1, band(a, a + wd, p, p, a + sl, a + sl + wd * 0.8, p + h, p + h * 1.2)))
        ha = U(0.15, 0.33)
        parts.append((1, stroke([(U(0.66, 0.78), hs + ha - U(0.0, 0.04)), (0.99, hs + ha)], sw)))
        parts.append((1, stroke([(0.97, hs), (0.985, hs + ha)], sw * U(1.0, 2.5))))
        if rng.random() < 0.6: parts.append((1, stroke([(0.72, hs + ha), (U(0.45, 0.6), p + (hs - p) * 0.3)], sw)))
    elif kind < 0.82:                                 # flip-flop / thong: thin sole, one strap
        t0 = U(0.04, 0.08); t1 = t0 + U(0.0, 0.07); sp = U(0.0, 0.04)
        parts.append((1, A((0.0, sp), (0.2, 0.0), (1.0, 0.0), (1.0, t1), (0.5, 0.5 * (t0 + t1)), (0.02, t0 + sp))))
        xa = U(0.14, 0.28); xp = U(0.40, 0.56); xe = U(0.6, 0.78); h = U(0.09, 0.2); sw = U(0.04, 0.085)
        if rng.random() < 0.5:
            parts.append((1, stroke([(xa, t0), (xp, t0 + h), (xe, t0 + 0.01)], sw)))
        else:
            parts.append((1, A((xa, t0), (xp, t0 + h), (xe, t0 + 0.01))))
    elif kind < 0.95:                                 # sporty sandal: thick sole, wide bands, openings between them
        ts = U(0.08, 0.15); sp = U(0.0, 0.03)
        parts.append((1, A((0.0, sp + 0.02), (0.15, 0.0), (1.0, 0.0), (1.0, ts + U(0.0, 0.04)), (0.02, ts + sp))))
        h1 = U(0.10, 0.17); h2 = U(0.2, 0.3); h3 = U(0.26, 0.4)
        a = U(0.06, 0.14); w1 = U(0.12, 0.24)
        parts.append((1, band(a, a + w1, ts, ts, a + 0.03, a + w1 + 0.03, ts + h1, ts + h1 * 1.25)))
        m = a + w1 + U(0.0, 0.12); w2 = U(0.1, 0.22)          # opening between toe band and instep band: none to wide
        parts.append((1, band(m, m + w2, ts, ts, m + 0.06, m + w2 + 0.06, ts + h2 * 0.9, ts + h2)))
        hb = max(U(0.74, 0.86), m + w2 + 0.02)
        parts.append((1, band(hb, 0.99, ts, ts, hb + 0.02, 0.99, ts + h3, ts + h3)))
        sw = U(0.04, 0.09)
        parts.append((1, stroke([(m + w2 * 0.6, ts + h2), (hb + 0.05, ts + h3 - 0.02)], sw)))
        if rng.random() < 0.6: parts.append((1, stroke([(a + w1 * 0.6, ts + h1), (m + 0.08, ts + h2 * 0.95)], sw)))
    else:                                             # slide / mule: sole and a closed front
        ts = U(0.05, 0.12); hl = U(0.0, 0.12); xe = U(0.45, 0.62); h = U(0.18, 0.3)
        parts.append((1, A((0.0, 0.02), (0.15, 0.0), (1.0, 0.0), (1.0, ts + hl), (0.55, ts), (0.0, ts))))
        parts.append((1, A((0.0, ts * 0.5), (0.03, ts + h * 0.45), (0.2, ts + h * 0.7), (xe, ts + h), (xe + 0.03, ts))))
    return place(parts, rng, yup=True, rot=U(-3, 3))


# Bags: front view. x from the centre, y DOWN, body width = 1, top edge of the body at y = 0.
def gen_bag(rng):
    U = rng.uniform
    kind = rng.random()
    parts = []

    def body(hb, top_ratio, round_times):
        wt, wb = (0.5 * top_ratio, 0.5) if top_ratio <= 1 else (0.5, 0.5 / top_ratio)
        b = A((-wt, 0.0), (wt, 0.0), (0.5 * (wt + wb) + U(0.0, 0.02), hb * 0.5), (wb, hb), (-wb, hb),
              (-0.5 * (wt + wb) - U(0.0, 0.02), hb * 0.5))
        return chaikin(b, round_times), wt

    if kind < 0.47:                                   # handbag / tote with a handle loop on top
        hb = U(0.55, 1.05)
        b, wt = body(hb, U(0.72, 1.3), int(rng.integers(0, 3)))
        parts.append((1, b))
        a = wt * U(0.3, 0.85); hh = U(0.15, 0.7); t = U(0.025, 0.065)
        if rng.random() < 0.85:
            parts.append((1, stroke(arch(-a, a, 0.03, hh), t)))
            if rng.random() < 0.3:
                parts.append((1, stroke(arch(-a * U(0.6, 0.9), a * U(0.6, 0.9), 0.03, hh * U(0.75, 0.95)), t)))
    elif kind < 0.68:                                 # clutch / purse / pouch: wide body, no handle
        hb = U(0.42, 0.85)
        b, wt = body(hb, U(0.85, 1.15), int(rng.integers(0, 3)))
        parts.append((1, b))
        if rng.random() < 0.25:                       # small wrist loop at one side
            sx = wt * (1 if rng.random() < 0.5 else -1)
            parts.append((1, stroke(arch(sx - 0.12, sx + 0.02, 0.05, U(0.1, 0.25)), U(0.02, 0.04))))
    elif kind < 0.90:                                 # shoulder bag: body with a long strap rising above it
        hb = U(0.48, 0.9)
        b, wt = body(hb, U(0.8, 1.2), int(rng.integers(0, 3)))
        parts.append((1, b))
        a = wt * U(0.7, 1.0); hh = U(0.35, 1.0); t = U(0.02, 0.05)
        if rng.random() < 0.5:
            parts.append((1, stroke(arch(-a, a, 0.03, hh), t)))
        else:
            px = U(-0.15, 0.15)
            parts.append((1, stroke([(-a, 0.03), (px, -hh), (a, 0.03)], t)))
    else:                                             # bucket / hobo bag: sagging top edge, round bottom, strap
        hb = U(0.6, 0.95); dip = U(0.08, 0.25)
        b = A((-0.46, 0.0), (-0.2, dip * 0.8), (0.0, dip), (0.2, dip * 0.8), (0.46, 0.0), (0.5, hb * 0.5), (0.36, hb),
              (-0.36, hb), (-0.5, hb * 0.5))
        parts.append((1, chaikin(b, 2)))
        parts.append((1, stroke(arch(-0.44, 0.44, 0.03, U(0.3, 0.8)), U(0.025, 0.05))))
    return place(parts, rng, rot=U(-2, 2))


GENERATORS = [gen_tshirt, gen_trouser, gen_pullover, gen_dress, gen_coat, gen_sandal, gen_shirt, gen_sneaker, gen_bag,
              gen_boot]


def draw_class(k, n, seed=SEED):
    """n drawings of class k as grey images (N, 28, 28) in 0..1."""
    rng = np.random.default_rng([seed, k])
    shapes = [GENERATORS[k](rng) for _ in range(n)]
    return render(shapes) * DRAW_BRIGHTNESS


# ----------------------------------------------------------------------------------------------------------------------
# the network's fixed front end, and the same computation in numpy for the drawings
# ----------------------------------------------------------------------------------------------------------------------
def gauss_kernel():
    r = np.arange(KS) - KS // 2
    g = np.exp(-r ** 2 / (2 * SIGMA ** 2)); g /= g.sum()
    return np.outer(g, g)


def front_end(x):
    """x: (N, 28, 28) grey images -> standardised blurred masks (N, 784). Mirrors the network layers below."""
    u = np.maximum((x - RAMP_LO) / (RAMP_HI - RAMP_LO), 0.0)
    m = 1.0 - np.maximum(1.0 - u, 0.0)
    p = KS // 2
    mp = np.pad(m, ((0, 0), (p, p), (p, p)))
    G = gauss_kernel()
    b = np.zeros_like(m)
    for dy in range(KS):
        for dx in range(KS):
            b += G[dy, dx] * mp[:, dy:dy + SIZE, dx:dx + SIZE]
    mu = b.mean(axis=(1, 2), keepdims=True)
    z = (b - mu) / np.sqrt(((b - mu) ** 2).mean(axis=(1, 2), keepdims=True) + 1e-5)
    return z.reshape(len(x), -1)


def build(out_dir, n_per_class=N_PER_CLASS):
    D = SIZE * SIZE
    a = 1.0 / (RAMP_HI - RAMP_LO)
    f32 = np.float32
    weights = {
        "ramp.W": np.full((1, 1, 1, 1), a, f32), "ramp.b": np.full((1,), -a * RAMP_LO, f32),
        "inv.W": np.full((1, 1, 1, 1), -1.0, f32), "inv.b": np.ones((1,), f32),
        "mask.W": np.full((1, 1, 1, 1), -1.0, f32), "mask.b": np.ones((1,), f32),
        "blur.W": gauss_kernel().reshape(1, 1, KS, KS).astype(f32), "blur.b": np.zeros((1,), f32),
    }
    # --- template bank ---
    W = np.zeros((10 * n_per_class, D), f32)
    for k in range(10):
        z = front_end(draw_class(k, n_per_class))          # each drawing standardised by itself alone
        W[k * n_per_class:(k + 1) * n_per_class] = (z / D).astype(f32)   # so that W @ z_image = correlation
    weights["match.W"] = W
    weights["match.b"] = np.zeros((10 * n_per_class,), f32)

    # --- open-shoe rule ---
    dog = -gauss_kernel(); dog[KS // 2, KS // 2] += 1.0                     # image minus its Gaussian-smoothed copy
    weights["rough.W"] = np.stack([dog, -dog]).reshape(2, 1, KS, KS).astype(f32)   # both signs, so relu gives |.|
    weights["rough.b"] = np.zeros((2,), f32)
    weights["patchy.W"] = np.full((2, 2 * D), 1.0 / ROUGH_SCALE, f32); weights["patchy.b"] = np.zeros((2,), f32)
    weights["mass.W"] = np.stack([np.full((D,), -ROUGH_RATIO / ROUGH_SCALE, f32),
                                  np.full((D,), -ROUGH_RATIO_HIGH / ROUGH_SCALE, f32)])
    weights["mass.b"] = np.zeros((2,), f32)
    # openness = [g1, g2]. clip(g1, -1, 1) = relu(g1 + 1) - relu(g1 - 1) - 1, clip(g2, 0, 1) = relu(g2) - relu(g2 - 1)
    weights["open.W"] = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], f32); weights["open.b"] = np.array([1, -1, 0, -1], f32)
    B = np.zeros((10, 4), f32); B[5] = [SANDAL_BONUS, -SANDAL_BONUS, SANDAL_BONUS_HIGH, -SANDAL_BONUS_HIGH]
    bias = np.zeros((10,), f32); bias[5] = -SANDAL_BONUS; bias[6] = -SHIRT_HANDICAP
    weights["bonus.W"] = B; weights["bonus.b"] = bias

    one = {"kernel": [1, 1], "stride": 1, "padding": 0}
    layers = [
        # silhouette mask -> blur -> standardise -> correlate with every drawing -> best drawing per class
        {"name": "ramp", "op": "conv", "inputs": ["image"], "out": 1, **one},
        {"name": "ramp_relu", "op": "relu", "inputs": ["ramp"]},
        {"name": "inv", "op": "conv", "inputs": ["ramp_relu"], "out": 1, **one},
        {"name": "inv_relu", "op": "relu", "inputs": ["inv"]},
        {"name": "mask", "op": "conv", "inputs": ["inv_relu"], "out": 1, **one},
        {"name": "blur", "op": "conv", "inputs": ["mask"], "out": 1, "kernel": [KS, KS], "stride": 1, "padding": KS // 2},
        {"name": "std", "op": "norm", "inputs": ["blur"]},
        {"name": "match", "op": "dense", "inputs": ["std"], "out": 10 * n_per_class},
        {"name": "by_class", "op": "reshape", "inputs": ["match"], "shape": [1, 10, n_per_class]},
        {"name": "best", "op": "maxpool", "inputs": ["by_class"], "size": [1, n_per_class], "stride": [1, n_per_class]},
        {"name": "template_scores", "op": "reshape", "inputs": ["best"], "shape": [10]},
        # open-shoe rule: patchiness of the picture against its total brightness -> sandal score up or down;
        # the same layer carries the shirt handicap as a constant
        {"name": "rough", "op": "conv", "inputs": ["image"], "out": 2, "kernel": [KS, KS], "stride": 1, "padding": KS // 2},
        {"name": "rough_relu", "op": "relu", "inputs": ["rough"]},
        {"name": "patchy", "op": "dense", "inputs": ["rough_relu"], "out": 2},
        {"name": "mass", "op": "dense", "inputs": ["image"], "out": 2},
        {"name": "openness", "op": "add", "inputs": ["patchy", "mass"]},
        {"name": "open", "op": "dense", "inputs": ["openness"], "out": 4},
        {"name": "open_relu", "op": "relu", "inputs": ["open"]},
        {"name": "bonus", "op": "dense", "inputs": ["open_relu"], "out": 10},
        {"name": "scores", "op": "add", "inputs": ["template_scores", "bonus"]},
    ]
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
