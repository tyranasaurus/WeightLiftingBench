"""Builds a hand-designed MNIST classifier: Gaussian blur -> per-image norm ->
bank of stroke-model digit templates (cosine matching) -> max over each class's
templates -> 10 class scores. Every number comes from typed stroke paths and
formulas (general knowledge of how the ten digits are drawn); no training data,
no fitting, no randomness. Usage: python3 build.py OUT_DIR
"""
import sys, os, json
import numpy as np

H = W = 28
YS, XS = np.mgrid[0:H, 0:W]

# ---- stroke rendering (round soft pen) ----------------------------------
def arc(cx, cy, rx, ry, a0, a1, n=64):
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))

def line(p0, p1, n=28):
    t = np.linspace(0, 1, n)
    return list(zip(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t))

def _stamp(pts, sigma):
    pts = np.array(pts)
    dx = XS[:, :, None] - pts[None, None, :, 0]
    dy = YS[:, :, None] - pts[None, None, :, 1]
    d2 = (dx * dx + dy * dy).min(axis=2)
    return np.clip(np.exp(-d2 / (2 * sigma * sigma)), 0, 1).astype(np.float32)


def _xform(strokes, shear, scale, tx, ty, asp=1.0):
    pts = []
    for s in strokes:
        for (x, y) in s:
            xx = 14 + (x - 14) * scale * asp
            yy = 14 + (y - 14) * scale
            pts.append((xx + shear * (14 - yy) + tx, yy + ty))
    return pts


BOX = ((np.abs(XS - 13.5) <= 10) & (np.abs(YS - 13.5) <= 10)).astype(np.float32)


def make_template(strokes, sigma=1.2, shear=0.0, beta=0.5, asp=1.0, box=20.0, center=13.5, thr=0.3):
    """Mimic the stated preprocessing on an ideal glyph: apply width factor `asp`,
    scale its ink bounding box to fit `box` x `box` (proportions kept), then shift
    so its centre of mass sits at `center`. Returns an ink-expected map minus a
    `beta`-weighted veto on the region inside the digit's box but away from its own
    strokes (penalises ink in wrong places)."""
    im = _stamp(_xform(strokes, shear, 1.0, 0.0, 0.0, asp), sigma)
    ys, xs = np.where(im > thr)
    scale = box / max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    im = _stamp(_xform(strokes, shear, scale, 0.0, 0.0, asp), sigma)
    tot = im.sum()
    tx, ty = center - (XS * im).sum() / tot, center - (YS * im).sum() / tot
    pts = _xform(strokes, shear, scale, tx, ty, asp)
    P = _stamp(pts, sigma)                      # ink expected here
    tol = _stamp(pts, sigma + 1.3) > 0.22       # tolerance band around strokes
    F = BOX * (~tol)                            # forbidden: in box, far from strokes
    return P - beta * F

# ---- the ten digits as typed pen paths, two style variants each ---------
def strokes_for(d, style=0):
    if d == 0:
        if style == 0:
            return [arc(14, 13.5, 5.4, 9.2, 0, 360)]
        return [arc(14, 13.5, 4.7, 9.0, 0, 360)]
    if d == 1:
        if style == 0:
            return [line((14, 4), (14, 23))]
        return [line((14, 4), (14, 23)), line((10.5, 7), (14, 4)), line((10, 23.3), (18, 23.3))]
    if d == 2:
        top = arc(13, 9.2, 5.0, 4.6, 200, 430)
        diag = line((17.5, 12.0), (9, 22))
        base = line((8.3, 22), (19.7, 22))
        if style == 0:
            return [top, diag, base]
        top2 = arc(13, 9.4, 5.3, 5.2, 188, 430)
        diag2 = line((17.8, 12.2), (9, 22))
        return [top2, diag2, base]
    if d == 3:
        up = arc(13, 9, 5.5, 4.5, -104, 102)
        lo = arc(12.5, 17.5, 5.8, 5.0, -97, 117)
        if style == 0:
            return [up, lo]
        up2 = arc(12.5, 9, 5.8, 4.3, -120, 100)
        lo2 = arc(12.5, 17.8, 6.0, 5.2, -100, 125)
        return [up2, lo2]
    if d == 4:
        vert = line((17, 4), (17, 24))
        bar = line((7, 15), (19.5, 15))
        if style == 0:                       # open top
            return [vert, bar, line((12.5, 4), (7, 15))]
        return [vert, bar, line((17, 4), (7, 15))]  # closed apex
    if d == 5:
        top = line((17.5, 5), (9, 5))
        vert = line((9, 5), (9, 13))
        if style == 0:
            belly = arc(10.5, 17.3, 6.3, 5.4, -70, 125)
            return [top, vert, belly]
        belly = arc(10.0, 17.6, 6.6, 5.6, -62, 130)
        return [top, vert, belly]
    if d == 6:
        if style == 0:
            hook = arc(13, 13, 6.2, 8.6, 300, 168)
            loop = arc(12.3, 17.6, 5.3, 5.3, 0, 360)
            return [hook, loop]
        hook = arc(13.5, 12.5, 6.6, 9.0, 305, 165)
        loop = arc(12.0, 18.0, 5.7, 5.5, 0, 360)
        return [hook, loop]
    if d == 7:
        top = line((7.8, 5), (19.2, 5))
        if style == 0:
            diag = line((19.2, 5), (10, 23))
            return [top, diag]
        diag = line((19.2, 5), (10.5, 23))
        cross = line((11.5, 14.5), (16.8, 14.5))
        return [top, diag, cross]
    if d == 8:
        if style == 0:
            return [arc(14, 9.2, 4.5, 4.8, 0, 360), arc(14, 17.8, 5.3, 5.4, 0, 360)]
        return [arc(14, 9.3, 4.0, 4.9, 0, 360), arc(14, 17.9, 4.8, 5.4, 0, 360)]
    if d == 9:
        loop = arc(14.5, 10, 4.9, 5.1, 0, 360)
        if style == 0:
            return [loop, line((18.9, 9.5), (12.5, 23))]
        return [loop, line((18.7, 10), (16.8, 23))]
    raise ValueError(d)

# ---- assemble ------------------------------------------------------------
SHEARS = [-0.30, -0.18, -0.06, 0.06, 0.18, 0.30, 0.42]
SIGMAS = [1.00, 1.45]
STYLES = [0, 1]
ASPECTS = [1.00]
BETA = 0.35
M = len(STYLES) * len(SHEARS) * len(SIGMAS) * len(ASPECTS)   # templates per class


def gaussian_kernel(k, sigma):
    a = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-(a[:, None] ** 2 + a[None, :] ** 2) / (2 * sigma * sigma))
    return (g / g.sum()).astype(np.float32)


PAD = 2                     # +/- PAD pixels of translation tolerance
SP = 2 * PAD + 1            # spatial positions per axis after the conv match


def build():
    # template filters (cosine matching): zero-mean, unit-norm 28x28 templates,
    # ordered class-major so channel = class*M + template.
    rows = []
    for c in range(10):
        for st in STYLES:
            for sh in SHEARS:
                for sg in SIGMAS:
                    for asp in ASPECTS:
                        im = make_template(strokes_for(c, st), sigma=sg, shear=sh,
                                           beta=BETA, asp=asp).reshape(-1)
                        im = im - im.mean()
                        rows.append((im / np.linalg.norm(im)).astype(np.float32))
    Wt = np.stack(rows, axis=0).reshape(10 * M, 1, H, W)   # (10*M, 1, 28, 28)
    bt = np.zeros(10 * M, np.float32)

    blurW = gaussian_kernel(5, 1.0).reshape(1, 1, 5, 5)
    blurb = np.zeros(1, np.float32)

    weights = {"blur.W": blurW, "blur.b": blurb, "tmpl.W": Wt, "tmpl.b": bt}
    net = {"layers": [
        {"name": "blur", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "nrm", "op": "norm", "inputs": ["blur"]},
        {"name": "tmpl", "op": "conv", "inputs": ["nrm"], "out": 10 * M, "kernel": [H, W], "stride": 1, "padding": PAD},
        {"name": "resh", "op": "reshape", "inputs": ["tmpl"], "shape": [10, M, SP * SP]},
        {"name": "pool", "op": "maxpool", "inputs": ["resh"], "size": [M, SP * SP], "stride": [M, SP * SP]},
        {"name": "out", "op": "reshape", "inputs": ["pool"], "shape": [10]},
    ]}
    return net, weights


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    net, weights = build()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out, "weights.npz"), **weights)


if __name__ == "__main__":
    main()
