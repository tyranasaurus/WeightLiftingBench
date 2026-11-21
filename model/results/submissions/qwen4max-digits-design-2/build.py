"""Hand-designed digit classifier: weights come only from typed pen-path shape models.

Nothing here reads any file.  Every number is either a constant I chose from general
knowledge of how the ten digits are written, or a formula (spline, disk brush, Gaussian
distance falloff, sinusoidal elastic warp) evaluated on those typed paths.

Pipeline implemented by the network:
  image -> conv 28x28, one filter per synthetic prototype, padding 2 (5x5 alignments)
        -> maxpool over alignments -> reshape (10, n, 1) -> maxpool over each class -> (10,)

A prototype filter is a two-sided soft chamfer match against a brushed pen path:
  S(p)      = exp(-max(0, dist(p, prototype ink))^2 / 2 sigma^2)   image ink -> prototype
  relief(p) = Gaussian-blurred prototype ink, calibrated so a perfect self-match
              refunds the whole mass charge                          prototype ink -> image
  score     = <image, S + LAMBDA*relief> - LAMBDA * mass(prototype)
The reward covers image ink lying on the prototype's strokes; the charge is only paid for
prototype strokes the image does not supply.  Each class score is the best of its 1680
prototypes over 25 alignments.
"""
import sys, json, math
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
RNG = np.random.default_rng(20261119)

FRAME = 28.0
CENTER = 13.5
BOX = 20.0            # the digits are size-normalised into a 20x20 box
KSIZE = 28            # whole frame: a prototype must never be cropped
C0 = (28 - KSIZE) // 2


# ----------------------------------------------------------------------------- splines
def catmull(pts, closed=False, per=48):
    """Catmull-Rom spline through typed control points -> dense polyline."""
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([P[0], P, P[-1]])
    out = []
    t = np.linspace(0.0, 1.0, per, endpoint=False)[:, None]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    out.append(P[-2][None, :])
    return np.vstack(out)


def resample(poly, step=0.018):
    """Resample a polyline at ~constant arc length (unit-height coords); points + arc weights."""
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-9:
        return poly[:1].copy(), np.array([1.0])
    n = max(8, int(math.ceil(total / step)))
    u = np.linspace(0.0, total, n)
    pts = np.stack([np.interp(u, s, poly[:, 0]), np.interp(u, s, poly[:, 1])], axis=1)
    w = np.gradient(u)
    return pts, np.maximum(w, 1e-6)


class Stroke:
    """One pen stroke: a spline path plus a radius profile along it."""

    def __init__(self, pts, closed=False, taper=0.0, bulge=0.0, phase=0.0):
        self.pts = np.asarray(pts, float)
        self.closed = closed
        self.taper = taper      # thinning at the ends (fraction of radius)
        self.bulge = bulge      # sinusoidal width modulation
        self.phase = phase

    def sample(self, jitter=0.0, rng=None):
        P = self.pts.copy()
        if jitter > 0 and rng is not None:
            P = P + rng.normal(0, jitter, P.shape)
        poly = catmull(P, self.closed)
        pts, w = resample(poly)
        u = np.linspace(0.0, 1.0, len(pts))
        prof = 1.0 - self.taper * (2 * u - 1) ** 2
        prof = prof * (1.0 + self.bulge * np.sin(2 * math.pi * (u + self.phase)))
        return pts, w, np.maximum(prof, 0.25)


# ------------------------------------------------------------------- prototype geometry
def build_strokes(strokes, slant, aspect, radius, warp_amp, rng, jitter):
    """Typed paths -> points/radii in the 28x28 frame, normalised like the data."""
    P, W, R = [], [], []
    for st in strokes:
        p, w, r = st.sample(jitter, rng)
        P.append(p); W.append(w); R.append(r)
    pts = np.vstack(P); w = np.concatenate(W); prof = np.concatenate(R)

    # slant (top leans right for slant>0) and width aspect, in unit-height coordinates
    pts = pts.copy()
    pts[:, 0] += slant * (0.5 - pts[:, 1])
    pts[:, 0] *= aspect

    pts *= BOX                      # unit height -> 20 px
    rad = radius * prof

    # smooth elastic warp: a few low-frequency sinusoids, fixed-seed randomness
    if warp_amp > 0:
        for _ in range(3):
            fx, fy = rng.uniform(0.4, 1.4, 2)
            ph = rng.uniform(0, 2 * math.pi, 2)
            ax, ay = rng.normal(0, 1, 2)
            n = np.linalg.norm([ax, ay]) + 1e-9
            ax, ay = ax / n, ay / n
            t = 2 * math.pi * (fx * pts[:, 0] + fy * pts[:, 1]) / BOX
            pts[:, 0] += warp_amp * ax * np.sin(t + ph[0])
            pts[:, 1] += warp_amp * ay * np.sin(t + ph[1])

    # fit the ink (path + brush radius) into the 20x20 box, keeping proportions
    lo = (pts - rad[:, None]).min(0); hi = (pts + rad[:, None]).max(0)
    ext = (hi - lo).max()
    f = BOX / ext
    pts *= f; rad *= f
    lo = (pts - rad[:, None]).min(0); hi = (pts + rad[:, None]).max(0)

    # centre of mass of the ink -> centre of the frame
    ww = w * rad
    com = (pts * ww[:, None]).sum(0) / ww.sum()
    pts += CENTER - com
    return pts, rad


YY, XX = np.mgrid[0:KSIZE, 0:KSIZE].astype(np.float64)
GX = (XX + C0).ravel()
GY = (YY + C0).ravel()

# a slightly larger grid so the Gaussian blur below has background context
PAD = 4
GSZ = KSIZE + 2 * PAD
gy, gx = np.mgrid[C0 - PAD:C0 - PAD + GSZ, C0 - PAD:C0 - PAD + GSZ].astype(np.float64)
FX = gx.ravel(); FY = gy.ravel()


def gauss1d(rho):
    r = int(math.ceil(3 * rho))
    t = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-t * t / (2 * rho * rho))
    return k / k.sum()


def blur2d(a, rho):
    k = gauss1d(rho)
    r = len(k) // 2
    p = np.pad(a, r, mode="constant")
    b = np.zeros_like(a)
    for i, w in enumerate(k):
        b += w * p[i:i + a.shape[0], r:r + a.shape[1]]
    p = np.pad(b, r, mode="constant")
    c = np.zeros_like(a)
    for i, w in enumerate(k):
        c += w * p[r:r + a.shape[0], i:i + a.shape[1]]
    return c


def kernel_for(pts, rad, sigma):
    """Two-sided soft chamfer match between the image and a brushed pen path.

    S(p) = exp(-max(0, dist(p, prototype ink))^2 / 2 sigma^2)   (image ink -> prototype)
    relief(p) = blurred prototype ink                              (prototype ink -> image)
    score = <image, S + LAMBDA*relief> - LAMBDA*mass(prototype)

    The first term rewards image ink lying on the prototype's strokes; the second
    refunds the mass charge wherever the image really does supply ink, so a prototype
    is only charged for strokes the image lacks.  This is what stops a bare `1` bar
    from winning on every digit, and stops a thick prototype from losing to a thin one.
    """
    dx = FX[:, None] - pts[None, :, 0]
    dy = FY[:, None] - pts[None, :, 1]
    d = (np.sqrt(dx * dx + dy * dy) - rad[None, :]).min(1).reshape(GSZ, GSZ)
    S = np.exp(-(np.maximum(d, 0.0) ** 2) / (2.0 * sigma * sigma))
    T = np.clip(0.5 - d, 0.0, 1.0)
    relief = blur2d(T, RHO)
    sl = slice(PAD, PAD + KSIZE)
    Tw, Rw = T[sl, sl], relief[sl, sl]
    mass = float(Tw.sum())
    # calibrate relief so a perfect self-match refunds the whole mass charge
    c0 = float((Tw * Rw).sum()) / max(mass, 1e-6)
    K = S[sl, sl] + LAMBDA * Rw / max(c0, 1e-6)
    return K.astype(np.float32), mass


# ---------------------------------------------------------------- typed digit shape models
# Coordinates: x right, y down, height 1.0.  Widths are the natural aspect of each digit.
# These are pen paths I typed from general knowledge of how the digits are written;
# `aspect`, `slant`, brush radius and the elastic warp supply the writer-to-writer variation.

def S(pts, **kw):
    return Stroke(pts, **kw)


MODELS = {
 0: [
   lambda a: [S([(0.5 + 0.5 * math.cos(t), 0.5 + 0.5 * math.sin(t))
                 for t in np.linspace(-math.pi / 2, 3 * math.pi / 2, 17)[:-1]], closed=True)],
   lambda a: [S([(0.5 + 0.5 * (0.60 + 0.55 * (0.5 + 0.5 * math.sin(t))) * math.cos(t),
                 0.5 + 0.5 * math.sin(t))
                 for t in np.linspace(-math.pi / 2, 3 * math.pi / 2, 17)[:-1]], closed=True)],
   lambda a: [S([(0.5 + 0.5 * math.cos(t), 0.5 + 0.5 * math.sin(t))
                 for t in np.linspace(-math.pi / 2 + 0.32, 3 * math.pi / 2 - 0.14, 15)])],
   lambda a: [S([(0.30, 0.03), (0.70, 0.03), (0.94, 0.25), (0.96, 0.70),
                 (0.72, 0.97), (0.30, 0.97), (0.06, 0.72), (0.05, 0.26)], closed=True)],
   lambda a: [S([(0.5 + 0.5 * (0.75 + 0.4 * (0.5 - 0.5 * math.sin(t))) * math.cos(t),
                 0.5 + 0.5 * math.sin(t))
                 for t in np.linspace(-math.pi / 2, 3 * math.pi / 2, 17)[:-1]], closed=True)],
   lambda a: [S([(0.44, 0.02), (0.78, 0.10), (0.95, 0.42), (0.86, 0.80),
                 (0.56, 0.98), (0.22, 0.90), (0.05, 0.58), (0.14, 0.20)], closed=True)],
 ],
 1: [
   lambda a: [S([(0.50, 0.00), (0.50, 0.34), (0.50, 0.68), (0.50, 1.00)], taper=0.15)],
   lambda a: [S([(0.50, 0.00), (0.47, 0.34), (0.50, 0.68), (0.53, 1.00)], taper=0.10)],
   lambda a: [S([(0.20, 0.20), (0.38, 0.08), (0.50, 0.00)]),
              S([(0.50, 0.00), (0.50, 0.40), (0.50, 1.00)])],
   lambda a: [S([(0.22, 0.18), (0.40, 0.06), (0.50, 0.00)]),
              S([(0.50, 0.00), (0.50, 0.50), (0.50, 0.98)]),
              S([(0.16, 1.00), (0.50, 1.00), (0.84, 1.00)])],
   lambda a: [S([(0.50, 0.00), (0.50, 0.50), (0.50, 0.98)]),
              S([(0.18, 1.00), (0.50, 1.00), (0.82, 1.00)])],
   lambda a: [S([(0.54, 0.00), (0.44, 0.30), (0.46, 0.66), (0.56, 1.00)], taper=0.2)],
 ],
 2: [
   lambda a: [S([(0.12, 0.19), (0.20, 0.07), (0.40, 0.02), (0.60, 0.06), (0.70, 0.16),
                 (0.63, 0.28), (0.42, 0.52), (0.18, 0.78), (0.05, 0.92)]),
              S([(0.05, 0.92), (0.38, 0.97), (0.70, 0.95), (0.96, 0.90)])],
   lambda a: [S([(0.10, 0.22), (0.20, 0.07), (0.44, 0.02), (0.66, 0.08), (0.74, 0.22),
                 (0.62, 0.40), (0.36, 0.62), (0.14, 0.80), (0.08, 0.90), (0.22, 0.95),
                 (0.32, 0.86), (0.20, 0.80)]),
              S([(0.08, 0.93), (0.40, 0.98), (0.72, 0.95), (0.96, 0.90)])],
   lambda a: [S([(0.10, 0.18), (0.26, 0.05), (0.52, 0.02), (0.74, 0.10), (0.76, 0.26),
                 (0.48, 0.52), (0.10, 0.90)]),
              S([(0.10, 0.92), (0.45, 0.96), (0.94, 0.93)])],
   lambda a: [S([(0.16, 0.16), (0.32, 0.04), (0.54, 0.03), (0.70, 0.14), (0.64, 0.30),
                 (0.38, 0.56), (0.10, 0.90)]),
              S([(0.10, 0.92), (0.42, 0.97), (0.92, 0.94)])],
   lambda a: [S([(0.10, 0.24), (0.16, 0.09), (0.36, 0.01), (0.58, 0.04), (0.72, 0.14),
                 (0.66, 0.28), (0.46, 0.52), (0.22, 0.76), (0.06, 0.93)]),
              S([(0.06, 0.93), (0.36, 0.99), (0.68, 0.97), (0.97, 0.93)])],
   lambda a: [S([(0.12, 0.20), (0.24, 0.06), (0.46, 0.03), (0.66, 0.10), (0.70, 0.24),
                 (0.52, 0.44), (0.26, 0.66), (0.08, 0.88)]),
              S([(0.08, 0.90), (0.42, 0.94), (0.95, 0.90)])],
 ],
 3: [
   lambda a: [S([(0.06, 0.14), (0.24, 0.03), (0.50, 0.01), (0.74, 0.08), (0.80, 0.22),
                 (0.64, 0.36), (0.40, 0.44)]),
              S([(0.40, 0.44), (0.66, 0.50), (0.86, 0.62), (0.86, 0.80),
                 (0.64, 0.95), (0.34, 0.99), (0.08, 0.90)])],
   lambda a: [S([(0.08, 0.03), (0.42, 0.02), (0.76, 0.03)]),
              S([(0.76, 0.03), (0.56, 0.20), (0.40, 0.42)]),
              S([(0.40, 0.42), (0.66, 0.48), (0.86, 0.62), (0.86, 0.80), (0.62, 0.96),
                 (0.32, 0.99), (0.06, 0.88)])],
   lambda a: [S([(0.14, 0.10), (0.36, 0.02), (0.60, 0.06), (0.66, 0.20), (0.48, 0.34),
                 (0.34, 0.42)]),
              S([(0.34, 0.42), (0.62, 0.48), (0.86, 0.62), (0.88, 0.80), (0.64, 0.96),
                 (0.32, 0.99), (0.06, 0.88)])],
   lambda a: [S([(0.08, 0.12), (0.30, 0.02), (0.58, 0.03), (0.78, 0.14), (0.72, 0.30),
                 (0.46, 0.42), (0.28, 0.46)]),
              S([(0.28, 0.46), (0.56, 0.52), (0.82, 0.64), (0.84, 0.82),
                 (0.58, 0.97), (0.28, 0.98), (0.06, 0.86)])],
   lambda a: [S([(0.10, 0.10), (0.32, 0.02), (0.56, 0.04), (0.72, 0.16), (0.62, 0.32),
                 (0.42, 0.46)]),
              S([(0.42, 0.46), (0.68, 0.52), (0.88, 0.66), (0.84, 0.84), (0.58, 0.98),
                 (0.28, 0.97), (0.08, 0.84)])],
   lambda a: [S([(0.06, 0.20), (0.20, 0.06), (0.44, 0.02), (0.68, 0.08), (0.76, 0.22),
                 (0.58, 0.38), (0.36, 0.46)]),
              S([(0.36, 0.46), (0.64, 0.52), (0.84, 0.66), (0.80, 0.86), (0.52, 0.99),
                 (0.22, 0.96), (0.06, 0.82)])],
 ],
 4: [
   lambda a: [S([(0.58, 0.02), (0.36, 0.26), (0.14, 0.50), (0.05, 0.64)]),
              S([(0.05, 0.64), (0.35, 0.66), (0.68, 0.65), (0.97, 0.64)]),
              S([(0.74, 0.00), (0.72, 0.34), (0.70, 0.68), (0.68, 1.00)])],
   lambda a: [S([(0.72, 0.02), (0.48, 0.26), (0.22, 0.50), (0.06, 0.63)]),
              S([(0.06, 0.63), (0.36, 0.65), (0.70, 0.64), (0.97, 0.63)]),
              S([(0.74, 0.00), (0.72, 0.34), (0.70, 0.68), (0.68, 1.00)])],
   lambda a: [S([(0.42, 0.16), (0.26, 0.38), (0.10, 0.58), (0.05, 0.66)]),
              S([(0.05, 0.66), (0.38, 0.67), (0.70, 0.66), (0.97, 0.65)]),
              S([(0.70, 0.02), (0.68, 0.36), (0.66, 0.70), (0.64, 1.00)])],
   lambda a: [S([(0.12, 0.04), (0.18, 0.30), (0.26, 0.56), (0.30, 0.64)]),
              S([(0.30, 0.64), (0.60, 0.65), (0.97, 0.64)]),
              S([(0.72, 0.00), (0.70, 0.34), (0.68, 0.68), (0.66, 1.00)])],
   lambda a: [S([(0.55, 0.04), (0.34, 0.30), (0.12, 0.56), (0.04, 0.70)]),
              S([(0.04, 0.70), (0.36, 0.71), (0.70, 0.70), (0.97, 0.69)]),
              S([(0.72, 0.00), (0.70, 0.36), (0.68, 0.72), (0.66, 1.00)])],
   lambda a: [S([(0.62, 0.06), (0.40, 0.18), (0.20, 0.38), (0.10, 0.60), (0.08, 0.66)]),
              S([(0.08, 0.66), (0.40, 0.68), (0.72, 0.67), (0.97, 0.66)]),
              S([(0.76, 0.02), (0.74, 0.36), (0.72, 0.70), (0.70, 1.00)])],
 ],
 5: [
   lambda a: [S([(0.86, 0.03), (0.54, 0.02), (0.22, 0.04)]),
              S([(0.22, 0.04), (0.16, 0.22), (0.14, 0.42)]),
              S([(0.14, 0.42), (0.40, 0.35), (0.68, 0.40), (0.86, 0.56), (0.84, 0.78),
                 (0.60, 0.95), (0.30, 0.98), (0.06, 0.88)])],
   lambda a: [S([(0.16, 0.02), (0.52, 0.00), (0.90, 0.03)]),
              S([(0.22, 0.06), (0.16, 0.24), (0.14, 0.44)]),
              S([(0.14, 0.44), (0.42, 0.37), (0.70, 0.42), (0.86, 0.58), (0.82, 0.80),
                 (0.56, 0.96), (0.28, 0.98), (0.06, 0.86)])],
   lambda a: [S([(0.84, 0.06), (0.50, 0.02), (0.20, 0.08)]),
              S([(0.20, 0.08), (0.14, 0.26), (0.20, 0.42)]),
              S([(0.20, 0.42), (0.46, 0.38), (0.72, 0.44), (0.88, 0.60), (0.82, 0.82),
                 (0.54, 0.97), (0.24, 0.96), (0.06, 0.84)])],
   lambda a: [S([(0.88, 0.04), (0.52, 0.02), (0.20, 0.05)]),
              S([(0.20, 0.05), (0.15, 0.20), (0.13, 0.34)]),
              S([(0.13, 0.34), (0.42, 0.30), (0.72, 0.38), (0.90, 0.56), (0.86, 0.80),
                 (0.58, 0.97), (0.26, 0.98), (0.04, 0.86)])],
   lambda a: [S([(0.80, 0.02), (0.48, 0.01), (0.18, 0.03)]),
              S([(0.18, 0.03), (0.14, 0.18), (0.12, 0.36)]),
              S([(0.12, 0.36), (0.40, 0.32), (0.70, 0.40), (0.88, 0.58), (0.82, 0.82),
                 (0.54, 0.97), (0.24, 0.97), (0.05, 0.85)])],
   lambda a: [S([(0.88, 0.05), (0.56, 0.02), (0.24, 0.06)]),
              S([(0.24, 0.06), (0.18, 0.26), (0.16, 0.46)]),
              S([(0.16, 0.46), (0.44, 0.40), (0.72, 0.46), (0.88, 0.62), (0.84, 0.82),
                 (0.58, 0.96), (0.28, 0.98), (0.06, 0.88)])],
 ],
 6: [
   lambda a: [S([(0.84, 0.02), (0.60, 0.10), (0.34, 0.28), (0.16, 0.52), (0.10, 0.76),
                 (0.22, 0.93), (0.48, 0.99), (0.72, 0.93), (0.82, 0.78), (0.70, 0.64),
                 (0.44, 0.62), (0.20, 0.70), (0.11, 0.82)])],
   lambda a: [S([(0.88, 0.02), (0.62, 0.12), (0.36, 0.32), (0.20, 0.56), (0.14, 0.78),
                 (0.26, 0.94), (0.50, 0.99), (0.72, 0.93), (0.80, 0.79), (0.68, 0.67),
                 (0.44, 0.66), (0.22, 0.73), (0.15, 0.84)])],
   lambda a: [S([(0.76, 0.03), (0.48, 0.14), (0.24, 0.34), (0.12, 0.58), (0.10, 0.80),
                 (0.26, 0.95), (0.52, 0.99), (0.78, 0.92), (0.88, 0.76), (0.74, 0.62),
                 (0.46, 0.60), (0.20, 0.68), (0.10, 0.80)])],
   lambda a: [S([(0.62, 0.02), (0.44, 0.16), (0.26, 0.36), (0.16, 0.58), (0.12, 0.78),
                 (0.26, 0.94), (0.50, 0.99), (0.74, 0.91), (0.82, 0.72), (0.68, 0.56),
                 (0.42, 0.53), (0.20, 0.62), (0.12, 0.78)])],
   lambda a: [S([(0.78, 0.02), (0.52, 0.14), (0.28, 0.34), (0.14, 0.56), (0.10, 0.76),
                 (0.24, 0.93), (0.50, 0.99), (0.74, 0.92), (0.84, 0.74), (0.72, 0.58),
                 (0.46, 0.55), (0.22, 0.64), (0.12, 0.78)])],
   lambda a: [S([(0.68, 0.04), (0.42, 0.18), (0.22, 0.40), (0.12, 0.62), (0.14, 0.82),
                 (0.30, 0.96), (0.54, 0.99), (0.76, 0.90), (0.82, 0.70), (0.66, 0.55),
                 (0.40, 0.54), (0.18, 0.66), (0.13, 0.82)])],
 ],
 7: [
   lambda a: [S([(0.04, 0.06), (0.36, 0.03), (0.68, 0.04), (0.95, 0.05)]),
              S([(0.95, 0.05), (0.78, 0.35), (0.60, 0.66), (0.42, 1.00)])],
   lambda a: [S([(0.04, 0.06), (0.36, 0.03), (0.68, 0.04), (0.95, 0.05)]),
              S([(0.95, 0.05), (0.78, 0.35), (0.60, 0.66), (0.42, 1.00)]),
              S([(0.28, 0.53), (0.52, 0.51), (0.76, 0.50)])],
   lambda a: [S([(0.06, 0.18), (0.10, 0.07), (0.38, 0.03), (0.70, 0.04), (0.95, 0.06)]),
              S([(0.95, 0.06), (0.80, 0.36), (0.62, 0.68), (0.46, 1.00)])],
   lambda a: [S([(0.06, 0.05), (0.40, 0.02), (0.94, 0.05)]),
              S([(0.94, 0.05), (0.86, 0.30), (0.66, 0.62), (0.36, 0.98)])],
   lambda a: [S([(0.08, 0.04), (0.42, 0.02), (0.92, 0.04)]),
              S([(0.92, 0.04), (0.74, 0.36), (0.56, 0.68), (0.40, 1.00)])],
   lambda a: [S([(0.04, 0.10), (0.30, 0.04), (0.62, 0.03), (0.94, 0.06)]),
              S([(0.94, 0.06), (0.82, 0.34), (0.64, 0.66), (0.48, 1.00)]),
              S([(0.30, 0.50), (0.54, 0.48), (0.78, 0.47)])],
 ],
 8: [
   lambda a: [S([(0.50, 0.46), (0.28, 0.38), (0.20, 0.22), (0.30, 0.07), (0.52, 0.02),
                 (0.72, 0.08), (0.78, 0.24), (0.66, 0.38), (0.44, 0.48), (0.22, 0.60),
                 (0.14, 0.78), (0.28, 0.94), (0.52, 0.99), (0.76, 0.93), (0.86, 0.76),
                 (0.76, 0.58), (0.50, 0.46)])],
   lambda a: [S([(0.50, 0.42), (0.32, 0.34), (0.26, 0.18), (0.38, 0.05), (0.56, 0.02),
                 (0.72, 0.09), (0.74, 0.26), (0.60, 0.38), (0.42, 0.46), (0.20, 0.60),
                 (0.12, 0.80), (0.28, 0.95), (0.54, 0.99), (0.80, 0.92), (0.88, 0.74),
                 (0.76, 0.56), (0.50, 0.42)])],
   lambda a: [S([(0.50, 0.50), (0.24, 0.42), (0.14, 0.24), (0.26, 0.06), (0.50, 0.01),
                 (0.76, 0.07), (0.86, 0.26), (0.72, 0.44), (0.46, 0.52), (0.24, 0.64),
                 (0.16, 0.82), (0.32, 0.96), (0.56, 0.99), (0.78, 0.92), (0.86, 0.76),
                 (0.74, 0.60), (0.50, 0.50)])],
   lambda a: [S([(0.52, 0.44), (0.34, 0.36), (0.30, 0.20), (0.42, 0.06), (0.58, 0.03),
                 (0.72, 0.12), (0.70, 0.28), (0.56, 0.40), (0.44, 0.50), (0.28, 0.62),
                 (0.22, 0.80), (0.36, 0.95), (0.56, 0.98), (0.74, 0.90), (0.78, 0.72),
                 (0.66, 0.56), (0.52, 0.44)])],
   lambda a: [S([(0.48, 0.44), (0.26, 0.34), (0.22, 0.16), (0.36, 0.04), (0.56, 0.02),
                 (0.74, 0.10), (0.76, 0.28), (0.60, 0.40), (0.40, 0.48), (0.20, 0.62),
                 (0.14, 0.82), (0.32, 0.96), (0.56, 0.99), (0.80, 0.90), (0.84, 0.70),
                 (0.70, 0.54), (0.48, 0.44)])],
   lambda a: [S([(0.50, 0.48), (0.30, 0.40), (0.24, 0.24), (0.34, 0.08), (0.54, 0.03),
                 (0.72, 0.10), (0.76, 0.26), (0.64, 0.40), (0.46, 0.50), (0.26, 0.62),
                 (0.18, 0.80), (0.32, 0.95), (0.54, 0.99), (0.76, 0.92), (0.84, 0.74),
                 (0.72, 0.58), (0.50, 0.48)])],
 ],
 9: [
   lambda a: [S([(0.74, 0.26), (0.58, 0.36), (0.38, 0.37), (0.22, 0.28), (0.21, 0.14),
                 (0.37, 0.03), (0.58, 0.02), (0.74, 0.10), (0.76, 0.25), (0.76, 0.55),
                 (0.72, 0.80), (0.62, 1.00)])],
   lambda a: [S([(0.74, 0.28), (0.58, 0.40), (0.36, 0.40), (0.18, 0.30), (0.18, 0.15),
                 (0.36, 0.03), (0.58, 0.02), (0.76, 0.12), (0.78, 0.30), (0.78, 0.62),
                 (0.76, 1.00)])],
   lambda a: [S([(0.76, 0.30), (0.60, 0.42), (0.36, 0.43), (0.18, 0.32), (0.18, 0.15),
                 (0.36, 0.03), (0.58, 0.02), (0.76, 0.12), (0.78, 0.30), (0.74, 0.58),
                 (0.60, 0.86), (0.38, 0.98), (0.18, 0.94)])],
   lambda a: [S([(0.78, 0.34), (0.60, 0.45), (0.34, 0.46), (0.16, 0.35), (0.16, 0.17),
                 (0.36, 0.03), (0.62, 0.02), (0.80, 0.14), (0.82, 0.33), (0.78, 0.62),
                 (0.70, 0.88), (0.58, 1.00)])],
   lambda a: [S([(0.74, 0.26), (0.58, 0.36), (0.38, 0.37), (0.22, 0.28), (0.22, 0.14),
                 (0.38, 0.03), (0.58, 0.02), (0.74, 0.10), (0.76, 0.26), (0.76, 0.56),
                 (0.72, 0.82), (0.64, 1.00)])],
   lambda a: [S([(0.78, 0.32), (0.62, 0.44), (0.38, 0.45), (0.18, 0.34), (0.17, 0.17),
                 (0.35, 0.03), (0.59, 0.02), (0.77, 0.12), (0.80, 0.30), (0.78, 0.58),
                 (0.70, 0.84), (0.56, 0.98)])],
 ],
}

# natural width/height ratio ranges and slant ranges per digit (typed, general knowledge)
ASPECT = {0: (0.55, 1.00), 1: (0.90, 1.10), 2: (0.68, 1.00), 3: (0.62, 0.95),
          4: (0.68, 1.02), 5: (0.62, 0.95), 6: (0.58, 0.92), 7: (0.66, 1.00),
          8: (0.55, 0.90), 9: (0.55, 0.90)}
SLANT = {0: (-0.34, 0.44), 1: (-0.26, 0.46), 2: (-0.20, 0.34), 3: (-0.20, 0.34),
         4: (-0.20, 0.30), 5: (-0.20, 0.32), 6: (-0.20, 0.34), 7: (-0.18, 0.30),
         8: (-0.28, 0.40), 9: (-0.22, 0.34)}


# ------------------------------------------------------------------------- prototype bank
N_SLANT, N_ASPECT, N_RAD, N_WARP = 7, 3, 5, 16
RADII = [0.85, 1.15, 1.45, 1.80, 2.15]
SIGMA = 1.6           # how far image ink may stray from a prototype stroke
RHO = 1.0             # blur used for the reverse (prototype -> image) direction
LAMBDA = 0.60         # charge per unit of prototype ink the image does not supply
WARP_AMP = 1.10
JITTER = 0.030


def bank_for_class(k, rng):
    styles = MODELS[k]
    a0, a1 = ASPECT[k]
    s0, s1 = SLANT[k]
    slants = np.linspace(s0, s1, N_SLANT)
    aspects = np.linspace(a0, a1, N_ASPECT)
    ks, ms = [], []
    for si in range(N_SLANT):
        for ai in range(N_ASPECT):
            for ri in range(N_RAD):
                for wi in range(N_WARP):
                    st = styles[(si + ai + ri + wi) % len(styles)]
                    strokes = st(None)
                    pts, rad = build_strokes(strokes, slants[si], aspects[ai],
                                             RADII[ri], WARP_AMP, rng, JITTER)
                    K, m = kernel_for(pts, rad, SIGMA)
                    ks.append(K); ms.append(m)
    return np.asarray(ks, np.float32), np.asarray(ms, np.float32)


def make_bank():
    rng = np.random.default_rng(20261119)
    per = N_SLANT * N_ASPECT * N_RAD * N_WARP
    W = np.empty((10 * per, 1, KSIZE, KSIZE), np.float32)
    b = np.empty(10 * per, np.float32)
    for k in range(10):
        K, m = bank_for_class(k, rng)
        W[k * per:(k + 1) * per, 0] = K
        b[k * per:(k + 1) * per] = -LAMBDA * m
    return W, b, per


def main():
    W, b, per = make_bank()
    layers = [
        {"name": "match", "op": "conv", "inputs": ["image"], "out": int(W.shape[0]),
         "kernel": [KSIZE, KSIZE], "stride": 1, "padding": 2},
        {"name": "pos", "op": "maxpool", "inputs": ["match"], "size": [5, 5], "stride": [5, 5]},
        {"name": "grp", "op": "reshape", "inputs": ["pos"], "shape": [10, per, 1]},
        {"name": "best", "op": "maxpool", "inputs": ["grp"], "size": [per, 1], "stride": [per, 1]},
        {"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [10]},
    ]
    import os
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(OUT, "weights.npz"), **{"match.W": W, "match.b": b})
    print("templates per class:", per, "total weights:", W.size + b.size)


if __name__ == "__main__":
    main()
