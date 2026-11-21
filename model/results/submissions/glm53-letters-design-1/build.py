"""Designs a sign-language-letter classifier network and writes its weights.

Method (no training, no data):
  * A fixed formula front-end: anisotropic Gabor ridge/edge filters at several
    orientations and scales, difference-of-Gaussian blobs and annulus (ring)
    filters.  Each filter is applied twice with opposite sign and the two
    rectified responses are summed, which gives the absolute response, so the
    features do not care whether the hand is lighter or darker than the
    background.  A 2x2 maxpool gives mild translation tolerance.
  * A generative template bank: a parametric hand model (palm ellipse, finger
    and thumb capsules at anatomical proportions/angles, wrist stub) is posed
    for each of the 24 letters; each pose variant is rendered, pushed through
    the same feature pipeline, and its feature vector becomes one row of the
    final dense layer.  The per-class scores are the maximum over the class's
    pose variants (dense -> reshape -> maxpool), i.e. best-template matching.
  * Per-class biases equalise the false-alarm level of every class, measured
    on synthetic renders only (never on the training set).

Everything is derived from formulas and general knowledge of hands/ASL; no
training images, labels or statistics of them are used anywhere.
Run: python3 build.py OUT_DIR   ->  writes OUT_DIR/network.json, OUT_DIR/weights.npz
"""
import json, math, sys
import numpy as np

IMG = 28            # image size
SS = 4              # supersampling for the renderer
KSZ, PAD = 13, 6    # conv kernel size / padding
NORI_F = 12         # fine orientations
NCH = 38            # total filter channels
FEAT = NCH * 14 * 14

# ----------------------------------------------------------------------------
# 1. Filters (all formulas; each L2-normalised on the 13x13 grid)
# ----------------------------------------------------------------------------
def make_filters():
    fils = []
    def gab(theta, sig_n, sig_a, lam, odd):
        # bar/edge filter: elongated along direction theta, modulated across it
        a = math.radians(theta)
        # pixel-space direction of the bar (y down): world angle -> pixel angle
        ca, sa = math.cos(a), math.sin(a)
        K = np.zeros((KSZ, KSZ))
        for dj in range(KSZ):
            for di in range(KSZ):
                dx, dy = di - PAD, dj - PAD      # (dx right, dy down)
                n = -dx * sa + dy * ca           # across-bar coordinate
                u = dx * ca + dy * sa            # along-bar coordinate
                env = math.exp(-(n * n) / (2 * sig_n * sig_n) - (u * u) / (2 * sig_a * sig_a))
                ph = math.sin if odd else math.cos
                K[dj, di] = env * ph(2 * math.pi * n / lam)
        K -= K.mean()
        K /= np.sqrt((K * K).sum())
        return K
    def dog(s1, s2):
        K = np.zeros((KSZ, KSZ))
        for dj in range(KSZ):
            for di in range(KSZ):
                dx, dy = di - PAD, dj - PAD
                r2 = dx * dx + dy * dy
                K[dj, di] = math.exp(-r2 / (2 * s1 * s1)) / (2 * math.pi * s1 * s1) \
                          - math.exp(-r2 / (2 * s2 * s2)) / (2 * math.pi * s2 * s2)
        K /= np.sqrt((K * K).sum())
        return K
    def annulus(R, w):
        K = np.zeros((KSZ, KSZ))
        for dj in range(KSZ):
            for di in range(KSZ):
                dx, dy = di - PAD, dj - PAD
                r = math.hypot(dx, dy)
                K[dj, di] = math.exp(-((r - R) ** 2) / (2 * w * w))
        K -= K.mean()
        K /= np.sqrt((K * K).sum())
        return K
    for j in range(NORI_F):                     # fine ridges + edges
        t = 180.0 * j / NORI_F
        fils.append(gab(t, 1.2, 2.6, 4.2, False))
        fils.append(gab(t, 1.2, 2.6, 4.2, True))
    for j in range(6):                          # coarse ridges
        fils.append(gab(180.0 * j / 6, 2.0, 4.6, 7.0, False))
    for j in range(4):                          # very coarse bars (overall direction)
        fils.append(gab(180.0 * j / 4, 2.6, 4.8, 9.0, False))
    fils.append(dog(1.4, 3.0))                  # blobs, two scales
    fils.append(dog(2.8, 5.6))
    fils.append(annulus(3.5, 1.7))              # rings (O, C), two radii
    fils.append(annulus(5.5, 2.0))
    assert len(fils) == NCH
    return np.stack(fils)                       # (NCH, KSZ, KSZ)

# ----------------------------------------------------------------------------
# 2. Feature pipeline (numpy mirror of the network ops)
# ----------------------------------------------------------------------------
def _conv(x, W):
    # x: (N,1,28,28); W: (NCH,13,13); correlation, zero padding, stride 1
    n = x.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)))
    ho = (IMG + 2 * PAD - KSZ) + 1
    cols = np.stack([xp[:, 0, dy:dy + ho, dx:dx + ho].reshape(n, -1)
                     for dy in range(KSZ) for dx in range(KSZ)], axis=1)  # (n,169,784)
    Wm = W.reshape(NCH, KSZ * KSZ)
    out = np.matmul(Wm, cols)                              # (n,NCH,784)
    return out.transpose(0, 1, 2).reshape(n, NCH, IMG, IMG)

def _maxpool2(a):
    n, c = a.shape[0], a.shape[1]
    h = a.reshape(n, c, 14, 2, 14, 2)
    return np.maximum(np.maximum(h[:, :, :, 0, :, 0], h[:, :, :, 1, :, 0]),
                      np.maximum(h[:, :, :, 0, :, 1], h[:, :, :, 1, :, 1])).reshape(n, c, 14, 14)

FEAT_C = NCH * 7 * 7   # coarse path feature count

def features2(imgs):
    """imgs: (N,1,28,28) in [0,1] -> (fine (N,FEAT), coarse (N,FEAT_C))
    feature matrices, each std-normalised (mirrors the two network paths)."""
    x = imgs.astype(np.float64)
    mu = x.mean(axis=(2, 3), keepdims=True)
    x = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
    W = make_filters()
    fin, cou = [], []
    for i in range(0, x.shape[0], 64):
        r = _conv(x[i:i + 64], W)
        a = np.abs(r)                       # == relu(r)+relu(-r) of the network
        f = _maxpool2(a)                    # (n,38,14,14)
        fin.append(f.reshape(f.shape[0], -1))
        c = a.reshape(a.shape[0], NCH, 7, 4, 7, 4).mean(axis=(3, 5))   # 4x4 avg
        cou.append(c.reshape(c.shape[0], -1))
    def std(m):
        mu = m.mean(axis=1, keepdims=True)
        return (m - mu) / np.sqrt(((m - mu) ** 2).mean(axis=1, keepdims=True) + 1e-5)
    return std(np.concatenate(fin)), std(np.concatenate(cou))

# ----------------------------------------------------------------------------
# 3. Parametric hand renderer (general knowledge of hand anatomy + ASL poses)
# ----------------------------------------------------------------------------
def _cov_in(d, t, e):        # smooth coverage: 1 inside threshold t, 0 outside
    return np.clip((t - d) / e + 0.5, 0.0, 1.0)

def render(spec, rot=0.0, scale=1.2, cx=14.0, cy=17.4, mirror=False,
           hand=0.75, bg=0.35, blur=0.35, edge=0.30, shade=None):
    """spec: list of primitives in palm units (y up, palm centre at origin).
    ('e', x, y, rx, ry)  filled ellipse
    ('c', [(x,y),...], w)  capsule chain (fingers, thumb, wrist)
    Returns a 28x28 float image in [0,1]."""
    n = IMG * SS
    gx = ((np.arange(n) + 0.5) / SS)[None, :] * np.ones((n, 1))   # image-pixel coords
    gy = np.ones((n, 1)) * ((np.arange(n) + 0.5) / SS)[:, None]
    dx = (gx - cx) / scale
    dy = (cy - gy) / scale
    rho = math.radians(rot)
    x = dx * math.cos(rho) + dy * math.sin(rho)
    y = -dx * math.sin(rho) + dy * math.cos(rho)
    if mirror:
        x = -x
    m = np.zeros((n, n))
    if np.isscalar(bg):
        bgarr = np.full((n, n), float(bg))
    else:                                   # background image (28x28), hand occludes it
        bgarr = np.repeat(np.repeat(np.asarray(bg, dtype=float), SS, 0), SS, 1)
    v = bgarr.copy()
    amp = 0.25 * (hand - float(bgarr.mean()))   # internal tonal variation of the hand
    for pi, pr in enumerate(spec):
        # deterministic per-primitive brightness jitter (fingers/palm differ)
        hx, hy = (pr[1], pr[2]) if pr[0] == 'e' else (pr[1][0][0], pr[1][0][1])
        hv = hand + amp * math.sin(12.9898 * (hx + 0.31 * pi) + 78.233 * (hy - 0.17 * pi))
        if pr[0] == 'e':
            _, ex, ey, rx, ry = pr
            d = np.sqrt(((x - ex) / rx) ** 2 + ((y - ey) / ry) ** 2)
            cov = _cov_in(d, 1.0, edge / 4.5)
        else:
            _, pts, w = pr
            cov = np.zeros((n, n))
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                vx, vy = x1 - x0, y1 - y0
                L2 = vx * vx + vy * vy
                t = np.clip(((x - x0) * vx + (y - y0) * vy) / (L2 + 1e-9), 0, 1)
                px = x0 + t * vx
                py = y0 + t * vy
                d = np.hypot(x - px, y - py)
                cov = np.maximum(cov, _cov_in(d, w / 2.0, edge))
        m = np.maximum(m, cov)
        v = np.where(cov >= m, hv, v)      # brightness of the visible primitive
    img = bgarr + (v - bgarr) * m
    if shade is not None:                      # gentle internal shading of the hand
        sx, sy, amp = shade
        img = img + amp * m * np.clip((x - np.mean(x)) / 8.0, -1, 1) * sx \
                  + amp * m * np.clip((y - np.mean(y)) / 8.0, -1, 1) * sy
    img = img.reshape(IMG, SS, IMG, SS).mean(axis=(1, 3))
    if blur > 0:                                   # mild gaussian blur
        r = 2
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / blur) ** 2)
        k /= k.sum()
        img = np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), 0, img)
        img = np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), 1, img)
    return np.clip(img, 0, 1)

# --- hand anatomy (palm units: palm width 10; y up; thumb on -x side) --------
FI = dict(i=(-2.6, 4.8, 7.5, 2.4), m=(-0.9, 5.2, 8.6, 2.4),
          r=(0.8, 5.0, 7.8, 2.3), p=(2.5, 4.2, 5.6, 2.0))   # base, length, width
def _dir(a): return (math.sin(math.radians(a)), math.cos(math.radians(a)))
def _ext(f, a, bend=0.0, lf=1.0, wf=1.0):
    bx, by, L, w = FI[f]
    L *= lf; w *= wf
    d0 = _dir(a); d1 = _dir(a + bend)
    p1 = (bx + 0.55 * L * d0[0], by + 0.55 * L * d0[1])
    p2 = (p1[0] + 0.45 * L * d1[0], p1[1] + 0.45 * L * d1[1])
    return ('c', [(bx, by), p1, p2], w)
def _bump(f, a, L=2.5, wf=1.0):
    bx, by, _, w = FI[f]
    d = _dir(a)
    return ('c', [(bx, by), (bx + L * d[0], by + L * d[1])], w * wf)

PALM_O = ('e', 0.0, 0.2, 4.9, 5.1)          # open palm
PALM_F = ('e', 0.0, 0.9, 5.05, 6.6)         # fist
WRIST = ('c', [(0.0, -4.3), (0.0, -12.5)], 7.0)

def _arc_pts(cx, cy, rx, ry, a0, a1, npt):
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
            for a in np.linspace(a0, a1, npt)]

def letter_prims(k, g):
    """Primitives for class k with jitter dict g (all values hand-chosen ranges)."""
    P = []
    fb = g['bumpL']; fw = g['fw']
    if k == 0:      # A: fist, thumb up along the side
        P += [PALM_F, WRIST,
              ('c', [(-4.95 + g['tx'], -2.2 + g['ty']), (-4.5 + g['tx'], 5.6 + g['ty'])], 2.6 * fw),
              _bump('i', -12 + g['j1'], fb), _bump('m', -4 + g['j1'], fb),
              _bump('r', 5 + g['j2'], fb), _bump('p', 18 + g['j2'], fb * 0.9)]
    elif k == 1:    # B: four fingers up (adjacent), thumb folded across palm
        P += [PALM_O, WRIST,
              ('c', [(-4.35, 2.9), (1.7, 3.7)], 2.6 * fw),
              _ext('i', -6 + g['j1'], 2, g['lf'], fw), _ext('m', -1 + g['j1'], 2, g['lf'], fw),
              _ext('r', 4 + g['j2'], 2, g['lf'], fw), _ext('p', 10 + g['j2'], 3, g['lf'], fw)]
    elif k == 2:    # C: tall crescent opening right
        pts = _arc_pts(-3.0, 3.2, 5.4, 8.4, 42, 318, 9)
        P += [('c', pts, 3.1 * fw), ('c', [(-3.0, -4.0), (-2.0, -12.0)], 6.5)]
    elif k == 3:    # D: index up, others curled, thumb pinching middle
        P += [PALM_F, WRIST,
              _ext('i', -2 + g['j1'], 0, g['lf'], fw),
              _bump('m', 4, 2.2, fw), _bump('r', 10, 2.2, fw), _bump('p', 16, 2.0, fw),
              ('c', [(-4.45, -0.7), (-2.5, 1.5), (-0.5, 3.9)], 2.6 * fw)]
    elif k == 4:    # E: fingers curled down over thumb
        P += [PALM_F, WRIST]
        fbE = fb * 1.45
        for f, a, sg in [('i', -12, -1), ('m', -4, -1), ('r', 5, 1), ('p', 18, 1)]:
            bx, by, _, w = FI[f]
            d0 = _dir(a); d1 = _dir(a + sg * 120)
            p1 = (bx + fbE * d0[0], by + fbE * d0[1])
            p2 = (p1[0] + 1.9 * d1[0], p1[1] + 1.9 * d1[1])
            P.append(('c', [(bx, by), p1, p2], w * fw * 0.92))
    elif k == 5:    # F: three fingers up, index curled to meet thumb (circle)
        P += [('e', 0.0, 0.2, 4.7, 4.9), WRIST,
              ('c', [(-2.6, 4.8), (-2.3, 6.3), (-1.55, 4.9)], 2.4 * fw),
              ('c', [(-4.45, -0.7), (-2.8, 1.2), (-1.7, 2.9)], 2.6 * fw),
              _ext('m', -4 + g['j1'], 4, g['lf'], fw), _ext('r', 5 + g['j2'], 4, g['lf'], fw),
              _ext('p', 14 + g['j2'], 5, g['lf'], fw)]
    elif k == 6:    # G: index + thumb pointing sideways (rendered upright, rotated)
        P += [('e', 0.0, 0.4, 4.9, 5.4),
              ('c', [(1.2, -3.6), (6.0, -10.2)], 6.5),
              _ext('i', -2 + g['j1'], 0, g['lf'], fw),
              ('c', [(-6.05 + g['j1'] * 0.2, -0.8), (-5.9 + g['j1'] * 0.2, 3.2)], 2.35 * fw),
              _bump('m', 3, 2.0, fw), _bump('r', 9, 2.0, fw), _bump('p', 15, 1.9, fw)]
    elif k == 7:    # H: index + middle together pointing sideways
        seam = 0.22 if g['seam'] else 0.0
        P += [('e', 0.0, 0.4, 4.9, 5.4),
              ('c', [(1.2, -3.6), (6.0, -10.2)], 6.5),
              _ext('i', -3 + g['j1'] - seam * 12, 0, g['lf'], fw),
              _ext('m', 3 + g['j2'] + seam * 12, 0, g['lf'], fw),
              ('c', [(-4.2, -1.0), (0.5, -0.2)], 2.5 * fw),
              _bump('r', 8, 2.0, fw), _bump('p', 15, 1.9, fw)]
    elif k == 8:    # I: pinky up
        P += [PALM_F, WRIST,
              ('c', [(2.6 + g['j2'] * 0.1, 4.3),
                     (2.8 + g['j2'] * 0.1, 4.3 + 0.55 * 7.0 * g['lf']),
                     (3.05 + g['j2'] * 0.1, 4.3 + 7.0 * g['lf'])], 2.05 * fw),
              _bump('i', -12 + g['j1'], fb), _bump('m', -4 + g['j1'], fb), _bump('r', 5 + g['j2'], fb)]
    elif k == 9:    # K: index up, middle up-out, thumb between
        P += [PALM_O, WRIST,
              _ext('i', -4 + g['j1'] * 0.5, 0, g['lf'], fw),
              _ext('m', 26 + g['j2'] * 0.5, 6, g['lf'], fw),
              ('c', [(-4.45, -0.7), (-0.4 + g['j1'] * 0.15, 5.5)], 2.9 * fw),
              _bump('r', 8, 2.2, fw), _bump('p', 15, 2.0, fw)]
    elif k == 10:   # L: index up, thumb out to the side
        P += [('e', 0.0, 0.5, 4.95, 5.9), WRIST,
              _ext('i', -2 + g['j1'], 0, g['lf'], fw),
              ('c', [(-4.35, 2.7), (-8.9, 2.0)], 2.6 * fw),
              _bump('m', -4 + g['j1'], fb), _bump('r', 5 + g['j2'], fb), _bump('p', 16 + g['j2'], fb * 0.9)]
    elif k == 11:   # M: three fingers over thumb, thumb tip at ring-pinky gap
        P += [PALM_F, WRIST,
              _bump('i', -10 + g['j1'], fb + 0.8, fw), _bump('m', -2 + g['j1'], fb + 0.8, fw),
              _bump('r', 8 + g['j2'], fb + 0.8, fw), _bump('p', 17 + g['j2'], fb * 0.9, fw),
              ('c', [(0.25 + g['tx'], 1.5), (1.85 + g['tx'], 6.6 + g['ty'])], 2.45 * fw)]
    elif k == 12:   # N: two fingers over thumb, thumb tip at middle-ring gap
        P += [PALM_F, WRIST,
              _bump('i', -10 + g['j1'], fb + 0.8, fw), _bump('m', -2 + g['j1'], fb + 0.8, fw),
              _bump('r', 8 + g['j2'], fb, fw), _bump('p', 17 + g['j2'], fb * 0.9, fw),
              ('c', [(-0.15 + g['tx'], 1.3), (0.45 + g['tx'], 6.5 + g['ty'])], 2.4 * fw)]
    elif k == 13:   # O: closed ring
        P += [('c', _arc_pts(-1.4, 3.0, 4.15, 4.15, 0, 360, 10), 3.0 * fw),
              ('e', -1.2, -1.6, 3.5, 2.4), ('c', [(-1.2, -3.8), (-0.8, -12.0)], 6.6)]
    elif k == 14:   # P: K pointing down (whole hand rotated 180)
        P += [PALM_O, WRIST,
              _ext('i', -5 + g['j1'], 0, g['lf'], fw),
              _ext('m', 26 + g['j2'], 6, g['lf'], fw),
              ('c', [(-4.45, -0.7), (-0.5, 4.35)], 2.6 * fw),
              _bump('r', 8, 2.2, fw), _bump('p', 15, 2.0, fw)]
    elif k == 15:   # Q: G pointing down
        P += [('e', 0.0, 0.4, 4.9, 5.4),
              ('c', [(1.2, -3.6), (6.0, -10.2)], 6.5),
              _ext('i', -2 + g['j1'], 0, g['lf'], fw),
              ('c', [(-4.5, -0.8), (-4.35, 4.8)], 2.6 * fw),
              _bump('m', 3, 2.0, fw), _bump('r', 9, 2.0, fw), _bump('p', 15, 1.9, fw)]
    elif k == 16:   # R: index and middle crossed
        cxr, cyr = g['rx'], 8.35 + g['ry']      # crossing point (jittered)
        P += [PALM_O, WRIST,
              ('c', [(-2.6, 4.8), (cxr, cyr), (cxr + 1.8, cyr + 2.9)], 2.35 * fw),
              ('c', [(-0.9, 5.2), (cxr - 1.25, cyr + 0.1), (cxr - 2.85, cyr + 3.0)], 2.35 * fw),
              ('c', [(-4.35, 3.0), (1.7, 3.8)], 2.6 * fw),
              _bump('r', 8, 2.2, fw), _bump('p', 16, 2.0, fw)]
    elif k == 17:   # S: fist, thumb across the front of the fingers
        P += [PALM_F, WRIST,
              ('c', [(-4.2 + g['tx'], 6.9 + g['ty']), (2.3 + g['tx'], 7.3 + g['ty'])], 2.8 * fw),
              _bump('i', -12 + g['j1'], fb), _bump('m', -4 + g['j1'], fb),
              _bump('r', 5 + g['j2'], fb), _bump('p', 18 + g['j2'], fb * 0.9)]
    elif k == 18:   # T: thumb tip between index and middle
        P += [PALM_F, WRIST,
              ('c', [(-3.5 + g['tx'], 0.4), (-1.65 + g['tx'], 7.0 + g['ty'])], 2.45 * fw),
              _bump('i', -12 + g['j1'], fb * 0.9), _bump('m', -4 + g['j1'], fb * 0.9),
              _bump('r', 5 + g['j2'], fb * 0.9), _bump('p', 18 + g['j2'], fb * 0.8)]
    elif k == 19:   # U: index + middle up together
        P += [PALM_O, WRIST,
              ('c', [(-2.2, 4.9), (-2.05, 8.4), (-1.9, 12.2)], 2.05 * fw),
              ('c', [(0.3, 5.05), (0.45, 8.6), (0.65, 12.4)], 2.05 * fw),
              ('c', [(-4.35, 3.0), (1.7, 3.8)], 2.6 * fw),
              _bump('r', 8, 2.2, fw), _bump('p', 16, 2.0, fw)]
    elif k == 20:   # V: index + middle up, spread
        P += [PALM_O, WRIST,
              _ext('i', -14 + g['j1'] * 0.5, 3, g['lf'], fw), _ext('m', 14 + g['j2'] * 0.5, 3, g['lf'], fw),
              ('c', [(-4.35, 3.0), (1.7, 3.8)], 2.6 * fw),
              _bump('r', 8, 2.2, fw), _bump('p', 16, 2.0, fw)]
    elif k == 21:   # W: three fingers up, spread
        P += [PALM_O, WRIST,
              _ext('i', -19 + g['j1'], 3, g['lf'], fw), _ext('m', -1 + g['j1'], 3, g['lf'], fw),
              _ext('r', 17 + g['j2'], 3, g['lf'], fw),
              ('c', [(-4.3, 3.2), (2.0, 4.4)], 2.5 * fw), _bump('p', 18, 1.8, fw)]
    elif k == 22:   # X: index hooked
        P += [('e', 0.0, 0.5, 4.95, 5.9), WRIST,
              ('c', [(-2.6, 4.8), (-2.5, 8.2), (-0.45, 6.1)], 2.4 * fw),
              ('c', [(-4.35, 3.0), (1.7, 3.8)], 2.6 * fw),
              _bump('m', -4 + g['j1'], fb), _bump('r', 5 + g['j2'], fb), _bump('p', 16 + g['j2'], fb * 0.9)]
    else:           # Y: thumb out, pinky up
        P += [PALM_F, WRIST,
              ('c', [(-4.35, 2.6), (-8.8, 1.9)], 2.6 * fw),
              _ext('p', 16 + g['j2'], 2, g['lf'], fw),
              _bump('i', -12 + g['j1'], fb), _bump('m', -4 + g['j1'], fb), _bump('r', 5 + g['j2'], fb)]
    return P

# placement defaults per class: (base rotation deg, base cy)
PLACE = {0: (0, 17.2), 1: (0, 17.6), 2: (0, 17.4), 3: (0, 17.4), 4: (0, 17.2),
         5: (0, 17.6), 6: (-90, 15.0), 7: (-90, 15.0), 8: (0, 17.2), 9: (0, 17.6),
         10: (0, 17.2), 11: (0, 17.2), 12: (0, 17.2), 13: (0, 17.4), 14: (180, 11.0),
         15: (180, 11.5), 16: (0, 17.6), 17: (0, 17.2), 18: (0, 17.2), 19: (0, 17.6),
         20: (0, 17.6), 21: (0, 17.6), 22: (0, 17.2), 23: (0, 17.2)}

def jitter(rng):
    return dict(j1=rng.uniform(-4, 4), j2=rng.uniform(-4, 4),
                lf=rng.uniform(0.92, 1.08), fw=rng.uniform(0.88, 1.18),
                bumpL=rng.uniform(2.6, 3.4),
                rx=rng.uniform(-0.6, 0.9), ry=rng.uniform(-0.8, 0.8),
                tx=rng.uniform(-0.45, 0.45), ty=rng.uniform(-0.35, 0.35),
                seam=rng.random() < 0.5)

def _pose(k, rng, mode):
    """Sample a full hand pose + render style for class k."""
    g = jitter(rng)
    br, bcy = PLACE[k]
    if mode == 'wide':
        rot = br + rng.uniform(-10, 10)
        sc = rng.uniform(1.00, 1.45)
        cx = 14.0 + rng.uniform(-5.5, 5.5)
        cy = bcy + rng.uniform(-3.5, 3.5)
    else:
        rot = br + rng.uniform(-7, 7)
        sc = rng.uniform(1.08, 1.35)
        cx = 14.0 + rng.uniform(-2.5, 2.5)
        cy = bcy + rng.uniform(-2.0, 2.0)
    mir = rng.random() < 0.25
    blur = rng.uniform(0.15, 0.45)
    edge = rng.uniform(0.22, 0.40)
    shade = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 0.05))
    bgv = rng.uniform(0.30, 0.60)
    step = rng.uniform(0.25, 0.50)
    return dict(spec=letter_prims(k, g), rot=rot, sc=sc, cx=cx, cy=cy, mir=mir,
                blur=blur, edge=edge, shade=shade,
                hand=min(bgv + step, 0.95), bgv=bgv)

def draw(k, rng, mode='fine'):
    """One synthetic image of class k (flat background, flat-ish hand)."""
    p = _pose(k, rng, mode)
    return render(p['spec'], p['rot'], p['sc'], p['cx'], p['cy'], p['mir'],
                  blur=p['blur'], edge=p['edge'], shade=p['shade'],
                  hand=p['hand'], bg=p['bgv'])

def draw_noisy(k, rng, mode='fine'):
    """Synthetic image with realistic clutter: mostly flat background with
    occasional strong background edges (walls/objects) that the hand occludes,
    plus the described augmentation (noise, brightness/contrast)."""
    p = _pose(k, rng, mode)
    bg = np.full((28, 28), p['bgv'])
    gy, gx = np.mgrid[0:28, 0:28]
    nedge = rng.choice([0, 1, 1, 2], p=[0.35, 0.35, 0.15, 0.15])
    for _ in range(nedge):
        ang = rng.uniform(0, math.pi)
        px, py = rng.uniform(2, 26), rng.uniform(2, 26)
        dx, dy = math.cos(ang), math.sin(ang)
        sd = (gx - px) * (-dy) + (gy - py) * dx       # signed dist to the line
        h = rng.uniform(0.25, 0.8) * rng.choice([-1, 1])
        bg = bg + h * np.clip(sd / 1.2, -0.5, 0.5)    # step edge across it
    if rng.random() < 0.4:                            # gentle shading blobs
        amp = rng.uniform(0.03, 0.10)
        small = rng.uniform(-1, 1, (7, 7))
        b = np.kron(small, np.ones((4, 4)))[:28, :28]
        bg = bg + amp * b / (np.abs(b).max() + 1e-9)
    bg = np.clip(bg, 0, 1)
    img = render(p['spec'], p['rot'], p['sc'], p['cx'], p['cy'], p['mir'],
                 blur=p['blur'], edge=p['edge'], shade=p['shade'],
                 hand=p['hand'], bg=bg)
    contr = rng.uniform(0.85, 1.15)
    img = (img - 0.45) * contr + 0.45
    img += rng.normal(0, rng.uniform(0.012, 0.05), (28, 28))
    return np.clip(img, 0, 1)

# ----------------------------------------------------------------------------
# 4. Build the template bank and the weights
# ----------------------------------------------------------------------------
V = 64              # pose variants per class, fine path
VC = 96             # pose variants per class, coarse path (wide pose scatter)

def _smooth_template(t, alpha):
    """Blend a 14x14-per-channel template with its 2x2 block average."""
    if alpha <= 0:
        return t
    c = t.reshape(NCH, 14, 14)
    coarse = c.reshape(NCH, 7, 2, 7, 2).mean(axis=(2, 4))
    up = np.repeat(np.repeat(coarse, 2, axis=1), 2, axis=2)[:, :14, :14]
    return (1 - alpha) * c + alpha * up

def _region_weights(grid):
    """Hand-chosen emphasis: finger/thumb detail lives in the upper half of the
    frame for upright signs; wrist/clutter lives below.  Ring channels matter
    for O/C/F, coarse bars slightly less discriminative."""
    wch = np.array([1.0] * 24 + [0.85] * 6 + [0.85] * 4 + [1.0] * 2 + [1.1] * 2)
    nh = max(1, int(round(grid * 0.5)))       # top half: fingers and thumbs
    nm = max(1, int(round(grid * 0.25)))      # middle: palm
    wrow = np.array([1.35] * nh + [1.1] * nm + [0.8] * (grid - nh - nm))
    return (wch[:, None, None] * wrow[None, :, None] * np.ones((1, 1, grid))).ravel()

def build_all(out_dir):
    # --- template variants for both paths ----------------------------------
    wreg_f = _region_weights(14)
    wreg_c = _region_weights(7)
    temps = np.zeros((24, V, FEAT))
    tempsc = np.zeros((24, VC, FEAT_C))
    for k in range(24):
        rk = np.random.default_rng(1000 + k)
        for v in range(V):
            img = draw(k, rk, 'fine')[None, None]
            tf, tc = features2(img)
            alpha = rk.choice([0.0, 0.35, 0.7, 1.0], p=[0.3, 0.3, 0.25, 0.15])
            t = _smooth_template(tf[0], alpha).ravel() * wreg_f
            temps[k, v] = t * (math.sqrt(FEAT) / (np.linalg.norm(t) + 1e-9))
        rk = np.random.default_rng(2000 + k)
        for v in range(VC):
            img = draw(k, rk, 'wide')[None, None]
            tf, tc = features2(img)
            tcc = tc[0] * wreg_c
            tempsc[k, v] = tcc * (math.sqrt(FEAT_C) / (np.linalg.norm(tcc) + 1e-9))
    # --- probe images (synthetic, noisy/cluttered) for calibration ---------
    probs, plab = [], []
    for k in range(24):
        rk = np.random.default_rng(5000 + k)
        for j in range(16):
            probs.append(draw_noisy(k, rk, 'wide' if j % 2 else 'fine'))
            plab.append(k)
    PF, PC = features2(np.array(probs)[:, None, :, :])
    plab = np.array(plab)
    Wd = temps.reshape(24 * V, FEAT)
    Wdc = tempsc.reshape(24 * VC, FEAT_C)
    Sf = (PF @ Wd.T).reshape(len(PF), 24, V).max(axis=2)
    Sc = (PC @ Wdc.T).reshape(len(PF), 24, VC).max(axis=2)
    # equalise the discriminative margin of the two paths (synthetic probes)
    Tf = np.mean([Sf[plab == k, k].mean() for k in range(24)])
    Ff = np.mean([Sf[plab != k].mean()])
    Tc = np.mean([Sc[plab == k, k].mean() for k in range(24)])
    Fc = np.mean([Sc[plab != k].mean()])
    sc_c = (Tf - Ff) / (Tc - Fc + 1e-9)
    Wdc = Wdc * sc_c
    Sc = Sc * sc_c
    # per-class bias on the summed score: cancel mean false-alarm level
    S = Sf + Sc
    bd = np.zeros(24 * V)
    for k in range(24):
        bd[k * V:(k + 1) * V] = -np.mean(S[plab != k, k])
    # --- conv weights: filters and their negatives -------------------------
    F = make_filters()
    Wc = np.zeros((NCH, 1, KSZ, KSZ))
    Wc[:, 0] = F
    net = {"layers": [
        {"name": "nm", "op": "norm", "inputs": ["image"]},
        {"name": "cp", "op": "conv", "inputs": ["nm"], "out": NCH, "kernel": [KSZ, KSZ], "stride": 1, "padding": PAD},
        {"name": "cn", "op": "conv", "inputs": ["nm"], "out": NCH, "kernel": [KSZ, KSZ], "stride": 1, "padding": PAD},
        {"name": "rp", "op": "relu", "inputs": ["cp"]},
        {"name": "rn", "op": "relu", "inputs": ["cn"]},
        {"name": "ab", "op": "add", "inputs": ["rp", "rn"]},
        {"name": "mp", "op": "maxpool", "inputs": ["ab"], "size": [2, 2], "stride": [2, 2]},
        {"name": "ap", "op": "avgpool", "inputs": ["ab"], "size": [4, 4], "stride": [4, 4]},
        {"name": "rs", "op": "reshape", "inputs": ["mp"], "shape": [1, 1, FEAT]},
        {"name": "fn", "op": "norm", "inputs": ["rs"]},
        {"name": "rsc", "op": "reshape", "inputs": ["ap"], "shape": [1, 1, FEAT_C]},
        {"name": "fnc", "op": "norm", "inputs": ["rsc"]},
        {"name": "mt", "op": "dense", "inputs": ["fn"], "out": 24 * V},
        {"name": "mtc", "op": "dense", "inputs": ["fnc"], "out": 24 * VC},
        {"name": "vr", "op": "reshape", "inputs": ["mt"], "shape": [24, 1, V]},
        {"name": "vm", "op": "maxpool", "inputs": ["vr"], "size": [1, V], "stride": [1, V]},
        {"name": "vrc", "op": "reshape", "inputs": ["mtc"], "shape": [24, 1, VC]},
        {"name": "vmc", "op": "maxpool", "inputs": ["vrc"], "size": [1, VC], "stride": [1, VC]},
        {"name": "rf", "op": "reshape", "inputs": ["vm"], "shape": [24]},
        {"name": "rfc", "op": "reshape", "inputs": ["vmc"], "shape": [24]},
        {"name": "out", "op": "add", "inputs": ["rf", "rfc"]},
    ]}
    np.savez(out_dir + "/weights.npz",
             **{"cp.W": Wc.astype(np.float32), "cp.b": np.zeros(NCH, np.float32),
                "cn.W": (-Wc).astype(np.float32), "cn.b": np.zeros(NCH, np.float32),
                "mt.W": Wd.astype(np.float32), "mt.b": bd.astype(np.float32),
                "mtc.W": Wdc.astype(np.float32), "mtc.b": np.zeros(24 * VC, np.float32)})
    with open(out_dir + "/network.json", "w") as f:
        json.dump(net, f)
    return dict(temps=temps, tempsc=tempsc, sc_c=sc_c, bd=bd)

# ----------------------------------------------------------------------------
def classify(imgs, model):
    """Dev-only helper: score synthetic images with the same math as the net."""
    f, c = features2(imgs)
    Sf = (f @ model['temps'].reshape(24 * V, FEAT).T + model['bd']).reshape(len(f), 24, V).max(2)
    Sc = (c @ (model['tempsc'].reshape(24 * VC, FEAT_C) * model['sc_c']).T).reshape(len(f), 24, VC).max(2)
    return Sf + Sc

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/app/out"
    build_all(out)
    print("wrote", out + "/network.json and weights.npz")
