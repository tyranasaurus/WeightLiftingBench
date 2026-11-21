"""Build a hand-designed digit classifier: a bank of matched filters.

Nothing here reads any file.  Every number comes from (a) formulas (Gaussian blur,
distance-to-pen-path rasterisation) and (b) parametric pen-path models of the ten
digit shapes that I typed from general knowledge of how digits are written, with
style parameters (slant, width, aspect, curvature, stroke thickness, alternative
stroke topologies) drawn from a fixed-seed RNG.

The network is:  blur -> normalise -> matched-filter bank -> max over each class's
filters.  A filter is one rendered digit variant, blurred, mean-removed and scaled
to unit L2 norm, so the dot product with the normalised image is a cosine
similarity.  The class whose best-matching variant scores highest wins.

Usage:  python3 build.py OUTPUT_FOLDER
"""

import json
import os
import sys

import numpy as np

TAU = 2.0 * np.pi

# ----------------------------------------------------------------- global knobs
SEED = 20261119
N_BASE = 3000                      # distinct pen-path variants per class
SHIFTS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))   # small alignment slack
SIGMA = 1.1                        # Gaussian blur, pixels
KS = 7                             # blur kernel size (odd)
BOX = 20.0                         # digits are size-normalised into a 20x20 box
FRAME = 28
CENTRE = 13.5                      # centre of mass target (0-indexed pixel grid)
JITTER = 0.6                       # sub-pixel placement jitter, +/- pixels


# ------------------------------------------------------------------- path tools
def cubic(p0, p1, p2, p3, n=14):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def seg(p0, p1, n=6):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) * np.asarray(p0, float) + t * np.asarray(p1, float)


class Pen:
    """A stroke: move / line / cubic, sampled into a polyline."""

    def __init__(self, p):
        self.cur = np.asarray(p, float)
        self.pts = [self.cur[None, :]]

    def l(self, p, n=6):
        p = np.asarray(p, float)
        self.pts.append(seg(self.cur, p, n)[1:])
        self.cur = p
        return self

    def c(self, c1, c2, p, n=14):
        p = np.asarray(p, float)
        self.pts.append(cubic(self.cur, c1, c2, p, n)[1:])
        self.cur = p
        return self

    def out(self):
        return np.concatenate(self.pts, 0)


def loop(cx, cy, rx, ry, n=80, power=2.0, harm=(), tilt=0.0):
    """Closed oval; power 2 = ellipse, >2 boxier, <2 pointier.  harm = (k, amp, phase)."""
    t = np.linspace(0.0, TAU, n)
    ct, st = np.cos(t), np.sin(t)
    if power != 2.0:
        e = 2.0 / power
        ct = np.sign(ct) * np.abs(ct) ** e
        st = np.sign(st) * np.abs(st) ** e
    r = np.ones(n)
    for k, a, ph in harm:
        r = r + a * np.cos(k * t + ph)
    x = rx * r * ct
    y = ry * r * st
    if tilt:
        x, y = x * np.cos(tilt) - y * np.sin(tilt), x * np.sin(tilt) + y * np.cos(tilt)
    return np.stack([cx + x, cy + y], 1)


def oval_harm(rng, a2=0.05, a3=0.035, a1=0.03):
    return [(1, rng.uniform(-a1, a1), rng.uniform(0, TAU)),
            (2, rng.uniform(-a2, a2), rng.uniform(0, TAU)),
            (3, rng.uniform(-a3, a3), rng.uniform(0, TAU))]


# ------------------------------------------------------- pen-path digit models
# Coordinates: x right, y down, unit box [0,1] x [0,1].  These are the shapes I
# know digits to have; the random parameters cover handwriting style, not data.

def m0(rng):
    """0: one closed oval, usually taller than wide, sometimes egg-shaped."""
    rx = rng.uniform(0.25, 0.40)
    ry = rng.uniform(0.43, 0.50)
    return [loop(0.5, 0.5, rx, ry, 88, rng.uniform(1.75, 2.55),
                 oval_harm(rng), rng.uniform(-0.10, 0.10))]


def m1(rng):
    """1: a tall bar, usually bare, sometimes bowed, occasionally flag / foot."""
    xt = 0.5 + rng.uniform(-0.05, 0.05)
    xb = 0.5 + rng.uniform(-0.05, 0.05)
    y0 = rng.uniform(0.00, 0.06)
    y1 = rng.uniform(0.94, 1.00)
    bow = rng.uniform(-0.11, 0.11)
    strokes = [cubic((xt, y0), (xt + bow, y0 + 0.33), (xb + bow, y1 - 0.33), (xb, y1), 22)]
    if rng.random() < 0.18:                       # small flag at the top
        fx = xt - rng.uniform(0.13, 0.30)
        fy = y0 + rng.uniform(0.10, 0.26)
        strokes.append(cubic((fx, fy), (fx + 0.05, fy - 0.07),
                             (xt - 0.05, y0 + 0.03), (xt, y0), 8))
    if rng.random() < 0.12:                       # foot serif
        w = rng.uniform(0.15, 0.28)
        strokes.append(seg((xb - w, y1), (xb + w, y1), 6))
    return strokes


def m2(rng):
    """2: top hook, diagonal down-left, flat base.  Sometimes a curled top."""
    u = rng.random()
    yb = rng.uniform(0.88, 0.95)
    xe = rng.uniform(0.82, 0.98)
    if u < 0.22:                                  # curled / looped top
        p = Pen((0.36, 0.70))
        p.c((0.14, 0.52), (0.10, 0.14), (0.44, 0.05))
        p.c((0.74, -0.02), (0.94, 0.20), (0.72, 0.42))
        p.c((0.50, 0.64), (0.30, 0.78), (0.16, yb))
    elif u < 0.38:                                # angular, Z-like
        p = Pen((0.15, rng.uniform(0.14, 0.24)))
        p.c((0.28, 0.01), (0.60, 0.00), (rng.uniform(0.72, 0.84), rng.uniform(0.12, 0.20)))
        p.l((rng.uniform(0.14, 0.24), yb))
    else:                                         # ordinary 2
        p = Pen((rng.uniform(0.10, 0.18), rng.uniform(0.24, 0.34)))
        p.c((0.11, 0.06), (0.50, -0.04), (rng.uniform(0.68, 0.80), rng.uniform(0.08, 0.18)))
        p.c((0.92, 0.26), (0.80, 0.36), (rng.uniform(0.52, 0.66), rng.uniform(0.38, 0.50)))
        p.l((rng.uniform(0.13, 0.22), yb))
    p.l((xe, yb + rng.uniform(-0.02, 0.02)))
    return [p.out()]


def m3(rng):
    """3: two stacked bowls, or a flat top bar into the lower bowl."""
    ym = rng.uniform(0.44, 0.52)
    xm = rng.uniform(0.36, 0.50)
    if rng.random() < 0.22:
        p = Pen((rng.uniform(0.12, 0.20), rng.uniform(0.04, 0.09)))
        p.l((rng.uniform(0.68, 0.82), rng.uniform(0.03, 0.08)))
        p.c((0.62, 0.22), (0.52, 0.34), (xm, ym))
    else:
        p = Pen((rng.uniform(0.11, 0.20), rng.uniform(0.13, 0.22)))
        p.c((0.26, -0.01), (0.68, -0.03), (rng.uniform(0.72, 0.84), rng.uniform(0.13, 0.22)))
        p.c((0.86, 0.34), (0.64, ym - 0.02), (xm, ym))
    p.c((0.66, ym + 0.01), (rng.uniform(0.82, 0.92), ym + 0.12),
        (rng.uniform(0.78, 0.88), rng.uniform(0.72, 0.80)))
    p.c((0.80, 0.96), (0.40, 1.02), (rng.uniform(0.09, 0.18), rng.uniform(0.82, 0.92)))
    return [p.out()]


def m4(rng):
    """4: diagonal + crossbar + right vertical.  Usually the top is open (the
    diagonal starts well left of the vertical); sometimes a closed triangle."""
    yh = rng.uniform(0.56, 0.74)
    xleft = rng.uniform(0.04, 0.18)
    xv = rng.uniform(0.60, 0.82)
    if rng.random() < 0.35:                       # closed triangle
        xtop, yvtop = xv, rng.uniform(0.02, 0.06)
    else:                                         # open top
        xtop = xv - rng.uniform(0.14, 0.36)
        yvtop = rng.uniform(0.02, 0.22)
    xtop = max(xtop, xleft + 0.10)
    diag = cubic((xtop, 0.03), (xtop - 0.10, 0.26), (xleft + 0.08, yh - 0.18), (xleft, yh), 18)
    bar = seg((xleft, yh), (rng.uniform(0.84, 0.99), yh + rng.uniform(-0.03, 0.03)), 8)
    xb = xv + rng.uniform(-0.06, 0.06)
    vert = cubic((xv, yvtop), (xv + 0.01, 0.45), (xb - 0.01, 0.75), (xb, 0.97), 14)
    return [diag, bar, vert]


def m5(rng):
    """5: top bar, short left stem, big lower bowl.  Sometimes S-like."""
    ybar = rng.uniform(0.03, 0.10)
    x0 = rng.uniform(0.12, 0.26)
    x1 = rng.uniform(0.66, 0.90)
    tilt = rng.uniform(-0.10, 0.10)
    ym = rng.uniform(0.34, 0.48)
    s_like = rng.random() < 0.38
    if s_like:
        top = cubic((x1, ybar - tilt), (x1 - 0.25, ybar - 0.05),
                    (x0 + 0.05, ybar - 0.02), (x0, ybar + 0.14), 12)
        stem = cubic((x0, ybar + 0.14), (x0 - 0.02, ybar + 0.22),
                     (x0 + 0.02, ym - 0.08), (x0 + 0.06, ym), 8)
    else:
        top = cubic((x0, ybar + tilt), (x0 + 0.25, ybar - 0.03),
                    (x1 - 0.25, ybar - 0.03), (x1, ybar - tilt), 10)
        stem = cubic((x0, ybar + tilt), (x0 - 0.02, ybar + 0.15),
                     (x0 - 0.02, ym - 0.12), (x0 + 0.02, ym), 10)
    p = Pen((x0 + 0.02 if not s_like else x0 + 0.06, ym))
    p.c((0.42, ym - 0.10), (rng.uniform(0.78, 0.88), ym - 0.02),
        (rng.uniform(0.80, 0.89), ym + 0.20))
    p.c((0.87, ym + 0.42), (0.50, 1.00), (rng.uniform(0.10, 0.20), rng.uniform(0.83, 0.93)))
    return [top, stem, p.out()]


def m6(rng):
    """6: ascender curving down-left into a closed lower loop."""
    cx = 0.5 + rng.uniform(-0.05, 0.05)
    cy = rng.uniform(0.64, 0.74)
    rx = rng.uniform(0.25, 0.35)
    ry = rng.uniform(0.21, 0.29)
    xt = rng.uniform(0.55, 0.85)
    bend = rng.uniform(0.0, 0.22)
    p = Pen((xt, rng.uniform(0.01, 0.07)))
    p.c((xt - 0.10 - bend, 0.20), (cx - rx - 0.04, cy - 0.32), (cx - rx, cy))
    return [p.out(), loop(cx, cy, rx, ry, 72, rng.uniform(1.8, 2.4),
                          oval_harm(rng, 0.04, 0.03, 0.02), rng.uniform(-0.1, 0.1))]


def m7(rng):
    """7: top bar and a descending diagonal; sometimes a crossbar or a serif."""
    ybar = rng.uniform(0.04, 0.13)
    x0 = rng.uniform(0.05, 0.32)
    x1 = rng.uniform(0.74, 0.95)
    xe = rng.uniform(0.26, 0.54)
    bar = cubic((x0, ybar + rng.uniform(0.0, 0.05)), (x0 + 0.25, ybar - 0.03),
                (x1 - 0.25, ybar - 0.02), (x1, ybar), 10)
    bow = rng.uniform(-0.08, 0.08)
    diag = cubic((x1, ybar), (x1 - 0.06 + bow, 0.40), (xe + 0.10 + bow, 0.70), (xe, 0.97), 18)
    strokes = [bar, diag]
    if rng.random() < 0.26:                       # crossbar
        yc = rng.uniform(0.42, 0.58)
        t = (yc - ybar) / (0.97 - ybar)
        xd = x1 + (xe - x1) * t
        strokes.append(seg((xd - rng.uniform(0.26, 0.38), yc + 0.02),
                           (xd + rng.uniform(0.02, 0.10), yc - 0.02), 6))
    if rng.random() < 0.14:                       # tick at the left of the bar
        strokes.append(seg((x0, ybar), (x0 - 0.02, ybar + rng.uniform(0.08, 0.18)), 5))
    return strokes


def m8(rng):
    """8: two stacked loops (upper smaller), or a single crossing figure-eight."""
    if rng.random() < 0.32:
        p = Pen((0.70, 0.14))
        p.c((0.52, -0.02), (0.22, 0.06), (0.28, 0.26))
        p.c((0.33, 0.44), (0.72, 0.52), (0.74, 0.72))
        p.c((0.76, 0.94), (0.40, 1.02), (0.24, 0.88))
        p.c((0.08, 0.72), (0.42, 0.52), (0.60, 0.38))
        p.c((0.76, 0.26), (0.80, 0.18), (0.70, 0.14))
        return [p.out()]
    cxu = 0.5 + rng.uniform(-0.05, 0.05)
    cyu = rng.uniform(0.22, 0.31)
    rxu = rng.uniform(0.16, 0.28)
    ryu = rng.uniform(0.17, 0.27)
    cxl = 0.5 + rng.uniform(-0.05, 0.05)
    cyl = rng.uniform(0.68, 0.77)
    rxl = rxu * rng.uniform(1.05, 1.40)
    ryl = ryu * rng.uniform(1.00, 1.30)
    return [loop(cxu, cyu, rxu, ryu, 64, rng.uniform(1.8, 2.4), oval_harm(rng, 0.05, 0.04, 0.03)),
            loop(cxl, cyl, rxl, ryl, 72, rng.uniform(1.8, 2.4), oval_harm(rng, 0.05, 0.04, 0.03))]


def m9(rng):
    """9: closed upper loop with a tail; tail straight, curved or hooked."""
    cx = 0.5 + rng.uniform(-0.05, 0.05)
    cy = rng.uniform(0.21, 0.32)
    rx = rng.uniform(0.22, 0.34)
    ry = rng.uniform(0.18, 0.28)
    sx, sy = cx + rx * rng.uniform(0.85, 1.0), cy + ry * rng.uniform(0.0, 0.5)
    u = rng.random()
    if u < 0.40:                                  # straight-ish tail
        tail = cubic((sx, sy), (sx + 0.02, sy + 0.25),
                     (sx - rng.uniform(0.0, 0.10), 0.75),
                     (sx - rng.uniform(0.02, 0.16), 0.97), 16)
    elif u < 0.80:                                # tail curving left at the bottom
        tail = cubic((sx, sy), (sx + 0.04, 0.62), (0.66, 0.90),
                     (rng.uniform(0.34, 0.56), rng.uniform(0.92, 0.99)), 16)
    else:                                         # hooked tail
        tail = cubic((sx, sy), (sx + 0.03, 0.60), (0.58, 0.86),
                     (rng.uniform(0.24, 0.38), rng.uniform(0.80, 0.90)), 16)
    return [loop(cx, cy, rx, ry, 72, rng.uniform(1.8, 2.4),
                 oval_harm(rng, 0.04, 0.03, 0.02), rng.uniform(-0.1, 0.1)), tail]


MODELS = (m0, m1, m2, m3, m4, m5, m6, m7, m8, m9)


# ------------------------------------------------------------------ style warp
def style_paths(paths, rng):
    """Aspect, slant, rotation and a smooth random bend, all about the box centre."""
    aspect = rng.uniform(0.78, 1.14)
    slant = float(np.clip(rng.normal(0.10, 0.17), -0.42, 0.48))
    rot = rng.normal(0.0, 0.055)
    a1, a2 = rng.uniform(-0.034, 0.034), rng.uniform(-0.020, 0.020)
    b1 = rng.uniform(-0.028, 0.028)
    a3, b2 = rng.uniform(-0.16, 0.16), rng.uniform(-0.14, 0.14)
    p1, p2, p3 = rng.uniform(0, 1, 3)
    cr, sr = np.cos(rot), np.sin(rot)
    out = []
    for pts in paths:
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        x = x + a1 * np.sin(TAU * (y + p1)) + a2 * np.sin(2 * TAU * (y + p2)) + a3 * (y - 0.5) ** 2
        y = y + b1 * np.sin(TAU * (x + p3)) + b2 * (x - 0.5) ** 2
        x = 0.5 + (x - 0.5) * aspect
        x = x + slant * (0.5 - y)
        dx, dy = x - 0.5, y - 0.5
        x = 0.5 + dx * cr - dy * sr
        y = 0.5 + dx * sr + dy * cr
        out.append(np.stack([x, y], 1))
    return out


# ------------------------------------------------------------------ rasteriser
_GRID = None


def grid():
    global _GRID
    if _GRID is None:
        yy, xx = np.mgrid[0:FRAME, 0:FRAME].astype(np.float64)
        _GRID = np.stack([xx.ravel(), yy.ravel()], 1)
    return _GRID


def ink(paths, r):
    """Anti-aliased union of capsules of radius r around the pen paths."""
    P = grid()
    best = np.full(len(P), 1e9)
    for pts in paths:
        A, B = pts[:-1], pts[1:]
        AB = B - A
        L2 = np.einsum("ij,ij->i", AB, AB)
        L2 = np.where(L2 < 1e-12, 1e-12, L2)
        AP = P[:, None, :] - A[None, :, :]
        t = np.clip(np.einsum("pkj,kj->pk", AP, AB) / L2, 0.0, 1.0)
        D = AP - t[:, :, None] * AB[None, :, :]
        d = np.sqrt(np.einsum("pkj,pkj->pk", D, D)).min(1)
        best = np.minimum(best, d)
    return np.clip(r + 0.5 - best, 0.0, 1.0).reshape(FRAME, FRAME)


def gauss_kernel(sigma, k):
    o = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-0.5 * (o / sigma) ** 2)
    return g / g.sum()


def blur(a, g):
    """Separable convolution with zero padding, exactly like the network's conv."""
    k = len(g)
    p = k // 2
    ap = np.pad(a, ((p, p), (0, 0)))
    tmp = sum(g[i] * ap[i:i + FRAME] for i in range(k))
    ap = np.pad(tmp, ((0, 0), (p, p)))
    return sum(g[i] * ap[:, i:i + FRAME] for i in range(k))


def translate(a, dx, dy):
    out = np.zeros_like(a)
    ys0, ys1 = max(0, dy), min(FRAME, FRAME + dy)
    xs0, xs1 = max(0, dx), min(FRAME, FRAME + dx)
    out[ys0:ys1, xs0:xs1] = a[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def com(a):
    s = a.sum()
    yy, xx = np.mgrid[0:FRAME, 0:FRAME]
    return np.array([(a * xx).sum() / s, (a * yy).sum() / s])


def render_variant(rng, k):
    """One digit variant -> raw 28x28 ink map, size- and centre-of-mass-normalised."""
    paths = style_paths(MODELS[k](rng), rng)
    r = float(np.clip(rng.normal(1.52, 0.34), 0.90, 2.40))
    box = BOX * rng.uniform(0.76, 1.02)      # size normalisation of handwriting is only approximate
    allp = np.concatenate(paths, 0)
    lo, hi = allp.min(0), allp.max(0)
    span = float((hi - lo).max())
    scale = (box - 2.0 * r) / max(span, 1e-6)
    mid = (lo + hi) / 2.0
    base = [(p - mid) * scale + CENTRE for p in paths]
    a = ink(base, r)
    shift = CENTRE - com(a) + rng.uniform(-JITTER, JITTER, 2)
    a = ink([p + shift for p in base], r)
    return a


def make_template(rng, k, g):
    return blur(render_variant(rng, k), g)


# --------------------------------------------------------------------- assembly
def build(out_dir, n_base=N_BASE, verbose=True):
    rng = np.random.default_rng(SEED)
    K = gauss_kernel(SIGMA, KS)
    nshift = len(SHIFTS)
    per_class = n_base * nshift
    total = 10 * per_class
    W = np.empty((total, FRAME * FRAME), np.float32)
    row = 0
    for k in range(10):
        for i in range(n_base):
            t = make_template(rng, k, K)
            for dx, dy in SHIFTS:
                v = translate(t, dx, dy) if (dx or dy) else t
                v = v - v.mean()
                nrm = np.sqrt((v * v).sum())
                W[row] = (v / max(nrm, 1e-9)).ravel().astype(np.float32)
                row += 1
        if verbose:
            print(f"class {k}: {per_class} filters", flush=True)
    b = np.zeros(total, np.float32)

    layers = [
        {"name": "blur", "op": "conv", "inputs": ["image"], "out": 1,
         "kernel": [KS, KS], "stride": 1, "padding": KS // 2},
        {"name": "unit", "op": "norm", "inputs": ["blur"]},
        {"name": "match", "op": "dense", "inputs": ["unit"], "out": total},
        {"name": "byclass", "op": "reshape", "inputs": ["match"], "shape": [1, 10, per_class]},
        {"name": "best", "op": "maxpool", "inputs": ["byclass"],
         "size": [1, per_class], "stride": [1, per_class]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]},
    ]
    net = {"layers": layers}
    bandW = (K[:, None] * K[None, :]).astype(np.float32)[None, None, :, :]
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "weights.npz"),
             **{"blur.W": bandW, "blur.b": np.zeros(1, np.float32),
                "match.W": W, "match.b": b})
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, indent=1)
    if verbose:
        print(f"wrote {total} filters, {W.size + bandW.size + 1 + total:,} weights")
    return net, W


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
