"""Hand-designed handwritten-digit classifier built with NO training data.

  image (1,28,28)
    -> conv 'feat' : 2 isotropic Gaussian blurs (shape/intensity, two scales)
                     + 8 oriented ridge detectors (stroke orientation),
                     all closed-form filters (no data).
    -> relu        : keep ink intensity and bright-stroke energy (non-negative).
    -> avgpool 4x4 stride 2 -> (10,13,13) feature map.
    -> dense 'match': each row is the (non-negative, unit-norm) feature vector
                      of ONE hand-drawn digit template.
    -> reshape (10,T,1) ; maxpool over the T templates of each class ; reshape (10,)

The score of a class is the largest dot product between the input feature map
and any of that class's templates.  Because features are non-negative (relu
energies) and every template is unit-norm, this is a cosine match on
non-negative vectors: a busy digit matches a busy template (more overlap) and a
thin digit a thin template, with no bias toward sparse prototypes.  relu
features are linear up to the image's contrast, so the match is automatically
contrast-invariant and needs no normalisation layer.

Every number comes from the code: the filters are closed-form (Gaussian and
oriented second-derivative-of-Gaussian ridge detectors); the templates are
digits drawn from typed pen paths (general knowledge of digit shape), each one
turned into a template normalised by itself alone, with slant / thickness /
style variety and smooth elastic warps (fixed seeds) to cover handwriting
variation.  No training image or label is ever read or used.
"""
import sys, os, json
import numpy as np

# ------------------------------------------------------------------ config
F = 3                 # supersample factor for rendering
SZ = 28
BOX = 20.0
K = 8                 # orientations
KS = 7                # filter size
PAD = 3
BLUR_SIGMAS = [0.8, 1.6]
BLUR_NORM = 2.0       # L2 norm of each blur filter (ridge filters are unit norm);
                      # gives the two intensity channels ~equal weight to the 8 ridge channels
# template parameter grid (slant, pen width, style, elastic warp) -- typed, not fit
SHEARS = [-0.35, -0.12, 0.12, 0.35]
PENS = [1.6, 2.4]
STYLES = [0, 1, 2]
ELASTICS = [(0, 0.0), (1, 0.035), (2, 0.045), (3, 0.045),
            (4, 0.055), (5, 0.055), (6, 0.06), (7, 0.04)]
NCH = len(BLUR_SIGMAS) + K

# ------------------------------------------------------------------ curves
def catmull(points, n=22):
    P = np.array(points, float)
    if len(P) == 2:
        t = np.linspace(0, 1, n)[:, None]
        return P[0][None] * (1 - t) + P[1][None] * t
    ext = np.vstack([P[0] + (P[0] - P[1]), P, P[-1] + (P[-1] - P[-2])])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t = np.linspace(0, 1, n)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                   (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                   (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)

def ellipse(cx, cy, rx, ry, a0=0, a1=360, n=48):
    a = np.linspace(np.radians(a0), np.radians(a1), n)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)

# ------------------------------------------------------------------ strokes (typed from knowledge)
def strokes_for(d, style=0):
    S = []
    def line(pts, w=1.0, smooth=True, n=22):
        a = catmull(pts, n) if smooth and len(pts) > 2 else np.array(pts, float)
        S.append((a, w))
    if d == 0:
        rx = [0.21, 0.25, 0.17][style % 3]
        line(ellipse(0.5, 0.5, rx, 0.37, 0, 360, 60), 1.0, smooth=False)
    elif d == 1:
        if style % 3 == 0:
            line([(0.5, 0.12), (0.5, 0.88)], 1.0, smooth=False)
        elif style % 3 == 1:
            line([(0.36, 0.25), (0.5, 0.12), (0.5, 0.88)], 1.0)
        else:
            line([(0.36, 0.25), (0.52, 0.12), (0.52, 0.88)], 1.0)
            line([(0.36, 0.88), (0.68, 0.88)], 1.0, smooth=False)
    elif d == 2:
        if style % 2 == 0:
            line([(0.29, 0.31), (0.37, 0.18), (0.55, 0.15), (0.68, 0.28),
                  (0.62, 0.45), (0.44, 0.63), (0.28, 0.81)], 1.0)
        else:
            line([(0.3, 0.33), (0.4, 0.19), (0.58, 0.16), (0.69, 0.32),
                  (0.58, 0.5), (0.4, 0.66), (0.3, 0.79)], 1.0)
        line([(0.27, 0.83), (0.74, 0.83)], 1.0, smooth=False)
    elif d == 3:
        line([(0.3, 0.22), (0.5, 0.14), (0.68, 0.26), (0.5, 0.46),
              (0.7, 0.62), (0.54, 0.82), (0.29, 0.78)], 1.0)
        if style % 2 == 1:
            line([(0.42, 0.46), (0.52, 0.46)], 1.0, smooth=False)
    elif d == 4:
        if style % 2 == 0:
            line([(0.57, 0.12), (0.26, 0.62), (0.75, 0.62)], 1.0)
            line([(0.62, 0.12), (0.62, 0.88)], 1.0, smooth=False)
        else:
            line([(0.3, 0.12), (0.27, 0.59), (0.73, 0.59)], 1.0)
            line([(0.64, 0.12), (0.6, 0.88)], 1.0, smooth=False)
    elif d == 5:
        line([(0.67, 0.16), (0.33, 0.16), (0.31, 0.46)], 1.0)
        line([(0.31, 0.46), (0.52, 0.41), (0.7, 0.56), (0.57, 0.79), (0.29, 0.81)], 1.0)
    elif d == 6:
        line([(0.63, 0.18), (0.42, 0.27), (0.31, 0.47), (0.3, 0.64)], 1.0)
        line(ellipse(0.49, 0.67, 0.2, 0.2, 0, 360, 48), 1.0, smooth=False)
    elif d == 7:
        line([(0.27, 0.19), (0.73, 0.17)], 1.0, smooth=False)
        line([(0.73, 0.17), (0.39, 0.88)], 1.0, smooth=False)
        if style % 2 == 1:
            line([(0.4, 0.52), (0.61, 0.5)], 0.9, smooth=False)
    elif d == 8:
        line(ellipse(0.5, 0.3, 0.16, 0.16, 0, 360, 40), 1.0, smooth=False)
        line(ellipse(0.5, 0.67, 0.21, 0.21, 0, 360, 48), 1.0, smooth=False)
    elif d == 9:
        line(ellipse(0.5, 0.33, 0.2, 0.2, 0, 360, 48), 1.0, smooth=False)
        line([(0.69, 0.33), (0.64, 0.62), (0.49, 0.87)], 1.0)
    return S

# ------------------------------------------------------------------ raster
def segdist(grid, Q):
    d = np.full(len(grid), 1e9)
    for i in range(len(Q) - 1):
        a = Q[i]; b = Q[i + 1]
        ab = b - a; L2 = ab @ ab
        if L2 < 1e-9:
            dd = np.hypot(grid[:, 0] - a[0], grid[:, 1] - a[1])
        else:
            t = np.clip(((grid - a) @ ab) / L2, 0, 1)
            proj = a[None] + t[:, None] * ab[None]
            dd = np.hypot(grid[:, 0] - proj[:, 0], grid[:, 1] - proj[:, 1])
        d = np.minimum(d, dd)
    return d

def warp_field(seed, amp, G=5):
    rng = np.random.default_rng(seed)
    return rng.normal(0, amp, (G, G)), rng.normal(0, amp, (G, G))

def bilerp(field, x, y):
    G = field.shape[0]
    fx = np.clip(x * (G - 1), 0, G - 1 - 1e-6)
    fy = np.clip(y * (G - 1), 0, G - 1 - 1e-6)
    x0 = fx.astype(int); y0 = fy.astype(int)
    tx = fx - x0; ty = fy - y0
    f00 = field[y0, x0]; f10 = field[y0, x0 + 1]
    f01 = field[y0 + 1, x0]; f11 = field[y0 + 1, x0 + 1]
    return (f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty) +
            f01 * (1 - tx) * ty + f11 * tx * ty)

def render(d, style=0, shear=0.0, pen=2.1, aspect=1.0, eseed=0, eamp=0.0):
    strokes = strokes_for(d, style)
    cs = SZ * F
    pr = pen * F / 2.0
    dgx, dgy = warp_field(eseed, eamp)
    def xf(P):
        x = P[:, 0].copy(); y = P[:, 1].copy()
        x = 0.5 + (x - 0.5) * aspect
        x = x + shear * (0.5 - y)
        if eamp > 0:
            x = x + bilerp(dgx, P[:, 0], P[:, 1])
            y = y + bilerp(dgy, P[:, 0], P[:, 1])
        return np.stack([x, y], 1)
    strokes = [(xf(p), w) for p, w in strokes]
    allpts = np.vstack([p for p, _ in strokes])
    mn = allpts.min(0); mx = allpts.max(0)
    ext = (mx - mn).max()
    sc = (BOX * F) / max(ext, 1e-6)
    cxy = (mn + mx) / 2.0
    center = np.array([cs / 2.0, cs / 2.0])
    ys, xs = np.mgrid[0:cs, 0:cs]
    grid = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    ink = np.zeros(cs * cs)
    soft = 0.8 * F
    for P, w in strokes:
        Q = (P - cxy) * sc + center
        r = pr * w
        seg = segdist(grid, Q)
        ink = np.maximum(ink, np.clip(r + soft - seg, 0, soft) / soft)
    img = ink.reshape(cs, cs).reshape(SZ, F, SZ, F).mean((1, 3))
    img = np.clip(img, 0, 1)
    s = img.sum()
    if s > 1e-6:
        yy, xx = np.mgrid[0:SZ, 0:SZ]
        cy = (img * yy).sum() / s; cx = (img * xx).sum() / s
        sy = int(round((SZ - 1) / 2.0 - cy)); sx = int(round((SZ - 1) / 2.0 - cx))
        img = np.roll(np.roll(img, sy, 0), sx, 1)
    return img.astype(np.float32)

# ------------------------------------------------------------------ filters (closed form)
def make_filters():
    r = KS // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    W = np.zeros((NCH, 1, KS, KS), np.float32)
    ch = 0
    for sigma in BLUR_SIGMAS:
        g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        g = g / np.sqrt((g ** 2).sum()) * BLUR_NORM
        W[ch, 0] = g; ch += 1
    sa, sb = 2.0, 1.1
    for k in range(K):
        th = np.pi * k / K
        a = xx * np.cos(th) + yy * np.sin(th)
        b = -xx * np.sin(th) + yy * np.cos(th)
        g = (1 - (b / sb) ** 2) * np.exp(-(b ** 2) / (2 * sb ** 2) - (a ** 2) / (2 * sa ** 2))
        g = g - g.mean()
        g = g / (np.sqrt((g ** 2).sum()) + 1e-9)
        W[ch, 0] = g; ch += 1
    return W, np.zeros(NCH, np.float32)

# ------------------------------------------------------------------ math ops (match the scorer)
def conv(x, W):
    o, c, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (PAD, PAD), (PAD, PAD)))
    H, Wd = x.shape[1], x.shape[2]
    ho = H + 2 * PAD - kh + 1; wo = Wd + 2 * PAD - kw + 1
    cols = np.stack([xp[:, dy:dy + ho, dx:dx + wo] for dy in range(kh) for dx in range(kw)], 1)
    cols = cols.reshape(c * kh * kw, ho * wo)
    Wm = W.reshape(o, c, kh * kw).reshape(o, c * kh * kw)
    return (Wm @ cols).reshape(o, ho, wo)

def avgpool(x, size, stride):
    c, H, Wd = x.shape
    ho = (H - size) // stride + 1; wo = (Wd - size) // stride + 1
    win = np.stack([x[:, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
                    for dy in range(size) for dx in range(size)], 0)
    return win.mean(0)

def features(img, W):
    y = conv(img[None].astype(np.float64), W)   # (NCH,28,28)
    y = np.maximum(y, 0.0)
    y = avgpool(y, 4, 2)                         # (NCH,13,13)
    return y.reshape(-1)

# ------------------------------------------------------------------ build
def build(out_dir):
    W, b = make_filters()
    params = [(st, sh, pe, es, ea) for st in STYLES for sh in SHEARS
              for pe in PENS for (es, ea) in ELASTICS]
    T = len(params)
    rows = []
    for d in range(10):
        for (st, sh, pe, es, ea) in params:
            img = render(d, style=st, shear=sh, pen=pe, eseed=es, eamp=ea)
            f = features(img, W)
            n = np.sqrt((f ** 2).sum())
            if n > 1e-9:
                f = f / n
            rows.append(f.astype(np.float32))
    matchW = np.stack(rows, 0)
    matchb = np.zeros(10 * T, np.float32)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "weights.npz"),
             **{"feat.W": W.astype(np.float32), "feat.b": b.astype(np.float32),
                "match.W": matchW, "match.b": matchb})

    net = {"layers": [
        {"name": "feat", "op": "conv", "inputs": ["image"], "out": NCH,
         "kernel": [KS, KS], "stride": 1, "padding": PAD},
        {"name": "r", "op": "relu", "inputs": ["feat"]},
        {"name": "pool", "op": "avgpool", "inputs": ["r"], "size": [4, 4], "stride": [2, 2]},
        {"name": "match", "op": "dense", "inputs": ["pool"], "out": 10 * T},
        {"name": "grid", "op": "reshape", "inputs": ["match"], "shape": [10, T, 1]},
        {"name": "mp", "op": "maxpool", "inputs": ["grid"], "size": [T, 1], "stride": [T, 1]},
        {"name": "out", "op": "reshape", "inputs": ["mp"], "shape": [10]},
    ]}
    with open(os.path.join(out_dir, "network.json"), "w") as fo:
        json.dump(net, fo)

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
