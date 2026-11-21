#!/usr/bin/env python3
"""build.py -- writes network.json and weights.npz for a hand-designed digit classifier.

Reads no files.  Every number comes from this source:
  * a fixed feature extractor made of formula filters (a Gaussian blur and 8 oriented
    "line" filters = Gaussian along the stroke x negative second derivative of a Gaussian
    across it) and a ReLU, read out at two scales: a fine Gaussian stride-2 pooling
    (9 x 14 x 14, keeps gaps and loops) and a coarse one after a 3x3 max-pool dilation
    (9 x 13 x 13, tolerant to shifts and stroke thickness);
  * a template bank: synthetic digits rendered from pen paths typed from general
    knowledge of how people write the ten digits (loops, bars, hooks, bowls ...), with
    hand-chosen random variation of slant, rotation, aspect ratio, stroke width and a
    smooth warp.  Each rendered image is pushed through the same feature extractor and
    becomes ONE row of each dense layer, standardised by its own mean and std only;
  * the network sums the fine and coarse correlations and takes, per class, the maximum
    over that class's templates.

Usage: python3 build.py OUT_DIR [N_PER_CLASS]
"""
import sys, json, math
import numpy as np

SEED = 20251119
N_PER_CLASS = 3500
IMG = 28
BOX = 20.0                 # digits are scaled to fit a 20 x 20 box (as in the data description)
CENTRE = 13.75             # centre-of-mass target (between the two integer conventions 13.5 / 14)
CENTRE_JIT = 0.6

# ---- feature extractor settings (all hand chosen) -------------------------------------
K1, P1 = 9, 4
SIG_BLUR = 1.0             # intensity channel blur
N_ORI = 8
SIG_ACROSS = 1.4           # Mexican-hat width across the stroke (positive lobe ~ 2.8 px wide)
SIG_ALONG = 2.0            # Gaussian extent along the stroke
INT_GAIN = 0.5             # gain of the intensity channel relative to the orientation channels
K2, P2, S2 = 7, 3, 2
# two matching scales, summed:
#   fine   : Gaussian pooling (sigma 1.2) straight after the ReLU      -> 9 x 14 x 14 (keeps gaps / loops)
#   coarse : 3x3 max-pool dilation, then Gaussian pooling (sigma 1.6)  -> 9 x 13 x 13 (tolerant to shifts / thickness)
SIG_FINE = 1.2
MP = 3
SIG_COARSE = 1.6
C = 1 + N_ORI
HF = (IMG + 2 * P2 - K2) // S2 + 1               # 14
H1 = IMG - MP + 1                                # 26
HC = (H1 + 2 * P2 - K2) // S2 + 1                # 13
DF = C * HF * HF                                 # 1764
DC = C * HC * HC                                 # 1521
NORM_EPS = 1e-5


# =========================================================================================
# geometry helpers: all paths live in a unit box, x to the right, y DOWN, (0,0) top-left
# =========================================================================================
def arc(cx, cy, rx, ry, a0, a1, n=60, rmod=None):
    """Elliptic arc from angle a0 to a1 (degrees).  -90 = top, 0 = right, 90 = bottom, 180 = left;
    increasing angle runs clockwise on screen.  rmod(a) optionally modulates the radius."""
    a = np.radians(np.linspace(a0, a1, n))
    r = 1.0 if rmod is None else rmod(a)
    return np.stack([cx + rx * r * np.cos(a), cy + ry * r * np.sin(a)], 1)


def bez(ctrl, n=40):
    """Bezier curve of any order through De Casteljau."""
    P = np.asarray(ctrl, float)[None].repeat(n, 0)
    t = np.linspace(0, 1, n)[:, None, None]
    while P.shape[1] > 1:
        P = (1 - t) * P[:, :-1] + t * P[:, 1:]
    return P[:, 0]


def seg(p, q, n=20):
    return bez([p, q], n)


def cat(*parts):
    out = [np.asarray(parts[0], float)]
    for p in parts[1:]:
        p = np.asarray(p, float)
        if np.linalg.norm(p[0] - out[-1][-1]) < 1e-9:
            p = p[1:]
        out.append(p)
    return np.concatenate(out, 0)


def U(rng, a, b):
    return float(rng.uniform(a, b))


def pick(rng, items, probs):
    return items[int(rng.choice(len(items), p=np.asarray(probs) / np.sum(probs)))]


def polar(cx, cy, rx, ry, a):
    a = math.radians(a)
    return (cx + rx * math.cos(a), cy + ry * math.sin(a))


# =========================================================================================
# digit generators: each returns a list of strokes (arrays of (x, y)) and transform options
# =========================================================================================
def gen0(rng):
    rx = U(rng, 0.22, 0.5)
    egg = U(rng, -0.1, 0.16)                   # >0: narrower at the top
    tilt = U(rng, -0.06, 0.06)
    rmod = lambda a: 1 + egg * 0.5 * np.sin(a) + tilt * np.cos(a)
    a0 = U(rng, -140, -40)
    total = U(rng, 335, 400)
    s = arc(0.5, 0.5, rx, 0.5, a0, a0 + total, n=110, rmod=rmod)
    if rng.uniform() < 0.3:                    # small hook inside at the start of the loop
        p0 = s[0]
        inner = (0.5 + (p0[0] - 0.5) * U(rng, 0.4, 0.75), 0.5 + (p0[1] - 0.5) * U(rng, 0.4, 0.75))
        s = cat(seg(inner, p0, 12), s)
    return [s], {"aspect": (0.75, 1.2)}


def gen1(rng):
    style = pick(rng, ["bar", "flag", "flagbase", "base"], [0.64, 0.22, 0.06, 0.08])
    bow = U(rng, -0.06, 0.06)
    stem = bez([(0.5, 0.0), (0.5 + bow, 0.5), (0.5, 1.0)], 50)
    strokes = []
    if style in ("flag", "flagbase"):
        fx, fy = U(rng, 0.08, 0.26), U(rng, 0.06, 0.26)
        flag = bez([(0.5 - fx, fy), (0.5 - fx * 0.4, fy * 0.3), (0.5, 0.0)], 20)
        strokes.append(cat(flag, stem))
    else:
        strokes.append(stem)
    if style in ("base", "flagbase"):
        b = U(rng, 0.12, 0.3)
        strokes.append(seg((0.5 - b, 1.0), (0.5 + b, 1.0)))
    return strokes, {"rot": 18.0, "shear": (-0.5, 0.65), "aspect": (0.9, 1.1)}


def gen2(rng):
    style = pick(rng, ["plain", "loop", "zed", "ess", "curl"], [0.34, 0.26, 0.12, 0.14, 0.14])
    rx, ry = U(rng, 0.22, 0.45), U(rng, 0.13, 0.3)
    cx = U(rng, 0.45, 0.62)
    a0 = U(rng, -195, -150)                    # top arc spans at most ~255 degrees: never a closed loop
    a1 = U(rng, 15, 60) if style != "zed" else U(rng, -15, 30)
    top = arc(cx, ry, rx, ry, a0, a1, n=60)
    p0 = top[0]
    if style == "curl":                        # the head starts with a small inward curl
        crx, cry = U(rng, 0.05, 0.11), U(rng, 0.05, 0.1)
        top = cat(arc(p0[0] + crx, p0[1], crx, cry, -120 + U(rng, -40, 40), 180, n=25), top)
    elif rng.uniform() < 0.3:                  # or just a small hook at the very start
        top = cat(seg((p0[0] + U(rng, 0.03, 0.1), p0[1] + U(rng, 0.03, 0.1)), p0, 10), top)
    end = top[-1]
    bl = (U(rng, 0.0, 0.16), U(rng, 0.9, 1.0))
    br = (U(rng, 0.6, 1.0), U(rng, 0.84, 1.0))
    if style == "zed":
        return [cat(top, seg(end, bl, 40), seg(bl, br, 30))], {}
    mid = ((end[0] + bl[0]) / 2, (end[1] + bl[1]) / 2)
    bowamt = U(rng, -0.12, 0.15)
    ctrl = (mid[0] + bowamt, mid[1] + bowamt * 0.3)
    if style == "ess":                         # S-shaped body
        c1 = (end[0] + U(rng, -0.05, 0.1), end[1] + U(rng, 0.15, 0.3))
        c2 = (bl[0] + U(rng, -0.05, 0.15), bl[1] - U(rng, 0.1, 0.3))
        diag = bez([end, c1, c2, bl], 45)
        base = bez([bl, ((bl[0] + br[0]) / 2, bl[1] + U(rng, -0.05, 0.05)), br], 30)
        return [cat(top, diag, base)], {}
    if style in ("plain", "curl"):
        diag = bez([end, ctrl, bl], 45)
        base = bez([bl, ((bl[0] + br[0]) / 2, bl[1] + U(rng, -0.05, 0.05)), br], 30)
        return [cat(top, diag, base)], {}
    # loop style: diagonal into a small loop at the bottom left, then the base
    lcx, lcy = U(rng, 0.14, 0.28), U(rng, 0.8, 0.9)
    lrx, lry = U(rng, 0.06, 0.13), U(rng, 0.05, 0.1)
    la0 = U(rng, -85, -35)
    lstart = polar(lcx, lcy, lrx, lry, la0)
    diag = bez([end, ctrl, lstart], 45)
    loop = arc(lcx, lcy, lrx, lry, la0, la0 + 360 + U(rng, 0, 30), n=45)
    base = bez([loop[-1], ((loop[-1][0] + br[0]) / 2, br[1] + U(rng, -0.04, 0.04)), br], 30)
    return [cat(top, diag, loop, base)], {}


def gen3(rng):
    style = pick(rng, ["round", "flat"], [0.8, 0.2])
    jx, jy = U(rng, 0.28, 0.48), U(rng, 0.42, 0.54)        # the middle junction (points left)
    lrx, lry = U(rng, 0.28, 0.46), (1.0 - jy) / 2 * U(rng, 0.95, 1.1)
    lcx, lcy = 0.47 + U(rng, -0.06, 0.06), 1.0 - lry
    lower = arc(lcx, lcy, lrx, lry, U(rng, -110, -80), U(rng, 150, 190), n=70)   # open bowl, never a loop
    if style == "round":
        urx, ury = U(rng, 0.22, 0.42), jy / 2 * U(rng, 0.95, 1.1)
        ucx, ucy = 0.46 + U(rng, -0.06, 0.06), ury
        upper = arc(ucx, ucy, urx, ury, U(rng, -155, -120), U(rng, 75, 110), n=60)
        return [cat(upper, seg(upper[-1], (jx, jy), 8), seg((jx, jy), lower[0], 8), lower)], {}
    top = seg((U(rng, 0.05, 0.2), U(rng, 0.0, 0.06)), (U(rng, 0.7, 0.95), U(rng, 0.0, 0.05)), 30)
    diag = seg(top[-1], (jx, jy), 30)
    return [cat(top, diag, seg((jx, jy), lower[0], 8), lower)], {}


def gen4(rng):
    style = pick(rng, ["open", "closed", "openround"], [0.42, 0.3, 0.28])
    yc = U(rng, 0.45, 0.72)
    xl = U(rng, 0.0, 0.15)
    xr = U(rng, 0.7, 1.0)
    xb = U(rng, 0.55, 0.8)                                  # the right stem x
    stem_bottom = (xb + U(rng, -0.22, 0.06), 1.0)
    if style == "closed":
        top = (xb + U(rng, -0.05, 0.03), U(rng, 0.0, 0.05))
        left = seg(top, (xl, yc), 40)
        stem = seg((xb, 0.0), stem_bottom, 45)
    else:
        top = (U(rng, 0.15, 0.5), U(rng, 0.0, 0.06))
        if style == "openround":
            left = bez([top, (xl - 0.06, yc * U(rng, 0.8, 1.0)), (xl + 0.06, yc)], 40)
        else:
            left = seg(top, (xl, yc), 40)
        stem = seg((xb, U(rng, 0.0, 0.15)), stem_bottom, 45)
    bar = seg((xl, yc), (xr, yc + U(rng, -0.08, 0.06)), 40)
    return [cat(left, bar), stem], {}


def gen5(rng):
    style = pick(rng, ["joined", "separate", "sloppy", "ess"], [0.35, 0.28, 0.2, 0.17])
    xl, xr = U(rng, 0.05, 0.3), U(rng, 0.65, 0.95)
    yt = U(rng, 0.0, 0.08)
    ys = U(rng, 0.3, 0.56)
    xs = xl - U(rng, 0.05, 0.18) if style == "sloppy" else xl + U(rng, -0.08, 0.08)
    if style == "ess":                         # one smooth stroke: the stem bows out and flows into the bowl
        stem = bez([(xl, yt), (xl - U(rng, 0.02, 0.12), ys * 0.5), (xs + U(rng, 0.05, 0.15), ys)], 30)
    else:
        stem = bez([(xl, yt), (xl + U(rng, -0.05, 0.05), (yt + ys) / 2), (xs, ys)], 30)
    rx = U(rng, 0.3, 0.48)
    ry = (1.0 - ys) / 2 * U(rng, 0.95, 1.08)
    cx, cy = 0.5 + U(rng, -0.08, 0.06), 1.0 - ry
    bowl = arc(cx, cy, rx, ry, U(rng, -130, -95), U(rng, 118, 165), n=70)
    conn = bez([(xs, ys), ((xs + bowl[0][0]) / 2, ys - 0.03), bowl[0]], 12)
    body = cat(stem, conn, bowl)
    tilt = U(rng, -0.08, 0.08)
    if style == "separate":
        gap = U(rng, 0.02, 0.12)
        bar = seg((xl + gap, yt + U(rng, -0.03, 0.03)), (xr, yt + tilt), 30)
        return [bar, body], {}
    bar = seg((xl, yt), (xr, yt + tilt), 30)
    return [cat(bar[::-1], body)], {}


def gen6(rng):
    rx, ry = U(rng, 0.25, 0.44), U(rng, 0.18, 0.3)
    cx, cy = 0.5 + U(rng, -0.06, 0.06), 1.0 - ry
    xt, yt = U(rng, 0.45, 0.85), U(rng, 0.0, 0.06)
    straight = U(rng, 0.0, 1.0)                 # 0 = round C-like stem, 1 = straight stem
    lx = cx - rx
    c1 = (lx + 0.35 * (1 - straight) + 0.1, 0.02 + 0.1 * straight)
    c2 = (lx - U(rng, 0.0, 0.1) * (1 - straight), U(rng, 0.3, 0.5))
    stem = bez([(xt, yt), c1, c2, (lx, cy)], 50)
    loop = arc(cx, cy, rx, ry, 180, -180 - U(rng, -15, 45), n=80)
    return [cat(stem, loop)], {}


def gen7(rng):
    style = pick(rng, ["plain", "cross", "hook"], [0.68, 0.18, 0.14])
    xl, yl = U(rng, 0.0, 0.25), U(rng, 0.0, 0.12)
    xr, yr = U(rng, 0.85, 1.0), U(rng, 0.0, 0.06)
    bar = bez([(xl, yl), ((xl + xr) / 2, (yl + yr) / 2 + U(rng, -0.07, 0.07)), (xr, yr)], 30)
    xe = U(rng, 0.15, 0.85)
    mid = ((xr + xe) / 2, 0.5)
    bowamt = U(rng, -0.1, 0.12)
    diag = bez([(xr, yr), (mid[0] + bowamt, mid[1]), (xe, 1.0)], 50)
    strokes = []
    if style == "hook":
        hook = seg((xl + U(rng, -0.03, 0.03), yl + U(rng, 0.08, 0.2)), (xl, yl), 12)
        strokes.append(cat(hook, bar, diag))
    else:
        strokes.append(cat(bar, diag))
    if style == "cross":
        yc = U(rng, 0.4, 0.62)
        i = int(np.argmin(np.abs(diag[:, 1] - yc)))
        xc = diag[i, 0]
        half = U(rng, 0.1, 0.22)
        strokes.append(seg((xc - half, yc), (xc + half, yc - U(rng, -0.02, 0.04)), 20))
    return strokes, {"aspect": (0.6, 1.15)}


def gen8(rng):
    style = pick(rng, ["eight", "twoloops"], [0.8, 0.2])
    opts = {"shear": (-0.2, 0.65), "aspect": (0.75, 1.2)}
    if style == "twoloops":                    # two stacked rings, the top one smaller
        ry1 = U(rng, 0.18, 0.27)
        rx1 = U(rng, 0.14, 0.3)
        ry2 = (1.0 - 2 * ry1 + U(rng, 0.0, 0.08)) / 2
        rx2 = rx1 * U(rng, 1.05, 1.5)
        up = arc(0.5 + U(rng, -0.05, 0.08), ry1, rx1, ry1, 90, 90 + 370, n=70)
        lo = arc(0.5 + U(rng, -0.06, 0.04), 1.0 - ry2, rx2, ry2, -90, -90 - 370, n=70)
        return [up, lo], opts
    # one continuous figure of eight (Lissajous x = sin 2t, y = -cos t), with a narrower top loop
    # (factor 1 - k cos t), the crossing pushed up (c sin^2 t) and the top loop shifted sideways (dx cos t)
    A = U(rng, 0.14, 0.3)
    k = U(rng, 0.0, 0.45)
    c = U(rng, 0.0, 0.12)
    dx = U(rng, -0.04, 0.14)
    t0 = U(rng, -0.4, 0.3)                     # where the pen starts / ends near the top: open or overlapped
    t = np.linspace(t0, 2 * math.pi + U(rng, -0.2, 0.45), 160)
    x = 0.5 + A * (1 - k * np.cos(t)) * np.sin(2 * t) + dx * np.cos(t)
    y = 0.5 - 0.5 * np.cos(t) - c * np.sin(t) ** 2
    return [np.stack([x, y], 1)], opts


def gen9(rng):
    style = pick(rng, ["straight", "curved", "open"], [0.45, 0.35, 0.2])
    rx = U(rng, 0.2, 0.4)
    ry = U(rng, 0.17, 0.29)
    cx, cy = 0.5 + U(rng, -0.06, 0.06), ry
    a0 = U(rng, -5, 25)
    a1 = -360 + U(rng, -25, 15) if style != "open" else -300 + U(rng, -30, 20)
    loop = arc(cx, cy, rx, ry, a0, a1, n=80)
    sx, sy = cx + rx, cy
    if style == "curved":
        stem = bez([(sx, sy), (sx + U(rng, 0.0, 0.06), U(rng, 0.7, 0.95)), (U(rng, 0.05, 0.4), U(rng, 0.92, 1.0))], 50)
    else:
        stem = bez([(sx, sy), (sx + U(rng, -0.04, 0.04), 0.65), (sx + U(rng, -0.25, 0.03), 1.0)], 50)
    if style == "open":
        return [cat(loop[::-1], stem)], {}
    return [cat(loop, seg(loop[-1], (sx, sy), 6), stem)], {}


GEN = [gen0, gen1, gen2, gen3, gen4, gen5, gen6, gen7, gen8, gen9]


# =========================================================================================
# transform + render
# =========================================================================================
def transform(strokes, rng, opts):
    rot = opts.get("rot", 8.0)
    shear_lo, shear_hi = opts.get("shear", (-0.25, 0.55))
    asp_lo, asp_hi = opts.get("aspect", (0.62, 1.12))
    ax = U(rng, asp_lo, asp_hi)
    sh = U(rng, shear_lo, shear_hi)
    th = math.radians(rng.normal(0, rot / 2))
    A1, A2 = U(rng, 0, 0.06), U(rng, 0, 0.06)
    f1, f2 = U(rng, 0.5, 1.6), U(rng, 0.5, 1.6)
    p1, p2 = U(rng, 0, 2 * math.pi), U(rng, 0, 2 * math.pi)
    out = []
    for s in strokes:
        x, y = s[:, 0].copy(), s[:, 1].copy()
        x = x + A1 * np.sin(2 * math.pi * f1 * y + p1)
        y = y + A2 * np.sin(2 * math.pi * f2 * x + p2)
        x = (x - 0.5) * ax + 0.5
        x = x + sh * (0.5 - y)
        xc, yc = x - 0.5, y - 0.5
        xr = math.cos(th) * xc - math.sin(th) * yc
        yr = math.sin(th) * xc + math.cos(th) * yc
        out.append(np.stack([xr, yr], 1))
    return out


_GRID = None


def _grid():
    global _GRID
    if _GRID is None:
        yy, xx = np.mgrid[0:IMG, 0:IMG]
        _GRID = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    return _GRID


def render(strokes, w):
    """strokes in pixel coordinates (x = column, y = row).  Anti-aliased stroke of width w."""
    G = _grid()
    img = np.zeros(IMG * IMG)
    for s in strokes:
        a, b = s[:-1], s[1:]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-12)
        ap = G[:, None, :] - a[None]
        t = np.clip((ap * ab[None]).sum(2) / L2[None], 0, 1)
        d2 = ((ap - t[:, :, None] * ab[None]) ** 2).sum(2)
        d = np.sqrt(d2.min(1))
        img = np.maximum(img, np.clip(w / 2 + 0.5 - d, 0, 1))
    return img.reshape(IMG, IMG)


def make_template(digit, rng):
    strokes, opts = GEN[digit](rng)
    strokes = transform(strokes, rng, opts)
    w = U(rng, 1.5, 4.2)
    allp = np.concatenate(strokes, 0)
    lo, hi = allp.min(0), allp.max(0)
    ext = max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    scale = (BOX - w) / ext
    mid = (lo + hi) / 2
    px = [(s - mid) * scale + (IMG - 1) / 2 for s in strokes]
    img = render(px, w)
    G = _grid()
    com = (G * img.ravel()[:, None]).sum(0) / img.sum()
    target = CENTRE + rng.uniform(-CENTRE_JIT, CENTRE_JIT, 2)
    px = [s + (target - com) for s in px]
    return render(px, w)


# =========================================================================================
# feature extractor (same arithmetic as the scorer's blocks)
# =========================================================================================
def gauss2d(k, sig):
    r = np.arange(k) - (k - 1) / 2
    g = np.exp(-r ** 2 / (2 * sig ** 2))
    g2 = np.outer(g, g)
    return g2 / g2.sum()


def line_filter(k, theta, sig_across, sig_along):
    r = np.arange(k) - (k - 1) / 2
    yy, xx = np.meshgrid(r, r, indexing="ij")
    u = xx * math.cos(theta) + yy * math.sin(theta)       # along the stroke
    v = -xx * math.sin(theta) + yy * math.cos(theta)      # across the stroke
    f = (1 - v ** 2 / sig_across ** 2) * np.exp(-v ** 2 / (2 * sig_across ** 2)) * np.exp(-u ** 2 / (2 * sig_along ** 2))
    f = f - f.mean()                                      # exactly zero response to a flat region
    f = f / np.maximum(f, 0).sum()                        # a stroke covering the positive lobe gives ~1
    return f


def diag_gauss(sig):
    W = np.zeros((C, C, K2, K2))
    g = gauss2d(K2, sig)
    for c in range(C):
        W[c, c] = g
    return W


def feature_weights():
    W1 = np.zeros((C, 1, K1, K1))
    W1[0, 0] = INT_GAIN * gauss2d(K1, SIG_BLUR)
    for i in range(N_ORI):
        W1[1 + i, 0] = line_filter(K1, math.pi * i / N_ORI, SIG_ACROSS, SIG_ALONG)
    return {"feat.W": W1, "feat.b": np.zeros(C),
            "fine.W": diag_gauss(SIG_FINE), "fine.b": np.zeros(C),
            "coarse.W": diag_gauss(SIG_COARSE), "coarse.b": np.zeros(C)}


def conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def maxpool(x, size, stride):
    n, c, h, w = x.shape
    ho, wo = (h - size) // stride + 1, (w - size) // stride + 1
    win = np.stack([x[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(size) for dx in range(size)], axis=0)
    return win.max(0)


def features(imgs, fw):
    """Returns (fine, coarse) feature matrices, flattened in the network's own order."""
    h = np.maximum(conv(imgs, fw["feat.W"], fw["feat.b"], 1, P1), 0)
    fine = conv(h, fw["fine.W"], fw["fine.b"], S2, P2)
    coarse = conv(maxpool(h, MP, 1), fw["coarse.W"], fw["coarse.b"], S2, P2)
    return fine.reshape(len(imgs), -1), coarse.reshape(len(imgs), -1)


def standardise(f):
    """Each template normalised by itself alone: zero mean, unit variance (same formula as the norm layer)."""
    mu = f.mean(1, keepdims=True)
    z = f - mu
    return z / np.sqrt((z ** 2).mean(1, keepdims=True) + NORM_EPS)


# =========================================================================================
def network_json(T):
    n = T // 10
    return {"layers": [
        {"name": "feat", "op": "conv", "inputs": ["image"], "out": C, "kernel": [K1, K1], "stride": 1, "padding": P1},
        {"name": "feat_relu", "op": "relu", "inputs": ["feat"]},
        # fine scale
        {"name": "fine", "op": "conv", "inputs": ["feat_relu"], "out": C, "kernel": [K2, K2], "stride": S2, "padding": P2},
        {"name": "fine_vec", "op": "reshape", "inputs": ["fine"], "shape": [1, C, HF * HF]},
        {"name": "fine_norm", "op": "norm", "inputs": ["fine_vec"]},
        {"name": "fine_match", "op": "dense", "inputs": ["fine_norm"], "out": T},
        # coarse scale
        {"name": "dilate", "op": "maxpool", "inputs": ["feat_relu"], "size": [MP, MP], "stride": [1, 1]},
        {"name": "coarse", "op": "conv", "inputs": ["dilate"], "out": C, "kernel": [K2, K2], "stride": S2, "padding": P2},
        {"name": "coarse_vec", "op": "reshape", "inputs": ["coarse"], "shape": [1, C, HC * HC]},
        {"name": "coarse_norm", "op": "norm", "inputs": ["coarse_vec"]},
        {"name": "coarse_match", "op": "dense", "inputs": ["coarse_norm"], "out": T},
        # sum of the two correlations, then the best template of each class
        {"name": "match", "op": "add", "inputs": ["fine_match", "coarse_match"]},
        {"name": "bank", "op": "reshape", "inputs": ["match"], "shape": [1, 10, n]},
        {"name": "classmax", "op": "maxpool", "inputs": ["bank"], "size": [1, n], "stride": [1, n]},
        {"name": "scores", "op": "reshape", "inputs": ["classmax"], "shape": [10]},
    ]}


def build_templates(n_per_class, seed=SEED):
    rng = np.random.default_rng(seed)
    fw = feature_weights()
    rows_f, rows_c = [], []
    for digit in range(10):
        imgs = np.stack([make_template(digit, rng) for _ in range(n_per_class)])[:, None]
        for i in range(0, n_per_class, 500):
            f, c = features(imgs[i:i + 500], fw)
            rows_f.append(standardise(f))
            rows_c.append(standardise(c))
    return fw, np.concatenate(rows_f, 0) / DF, np.concatenate(rows_c, 0) / DC


def main():
    out = sys.argv[1]
    n_per_class = int(sys.argv[2]) if len(sys.argv) > 2 else N_PER_CLASS
    import os
    os.makedirs(out, exist_ok=True)
    fw, Wf, Wc = build_templates(n_per_class)
    T = Wf.shape[0]
    weights = {k: v.astype(np.float32) for k, v in fw.items()}
    weights.update({"fine_match.W": Wf.astype(np.float32), "fine_match.b": np.zeros(T, np.float32),
                    "coarse_match.W": Wc.astype(np.float32), "coarse_match.b": np.zeros(T, np.float32)})
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(network_json(T), f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **weights)
    print(f"wrote {T} templates, {sum(v.size for v in weights.values()):,} weights")


if __name__ == "__main__":
    main()
