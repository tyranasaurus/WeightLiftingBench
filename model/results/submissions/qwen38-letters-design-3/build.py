"""Build weights for a hand-designed ASL-fingerspelling classifier.

Everything here comes from code alone (no data files are read):

  * Front end: a fixed bank of oriented derivative-of-Gaussian edge filters at three
    scales (formulas only), rectified and recombined into polarity-insensitive
    oriented-edge-magnitude maps, then blurred/subsampled into three spatial-pyramid
    levels (a coarse HOG-like descriptor).
  * Read out: a template bank.  For each of the 24 letters a parametric hand model
    (palm ellipse + finger capsules at typed angles + folded-finger block + thumb +
    wrist, all from general knowledge of hands and the ASL alphabet) is drawn in 12
    poses (rotation / translation / scale).  Each drawing is pushed through the very
    same front end and becomes ONE template row, normalised by itself alone
    (per-channel mean removed, then unit L2 norm, per pyramid level).
  * Class score = max over that class's 12 template scores (template-bank matching,
    implemented with reshape + maxpool), summed over the three pyramid levels.

No fitting, no statistics across images, no numbers read off any image.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

K = 24  # classes
NAMES = "A B C D E F G H I K L M N O P Q R S T U V W X Y".split()

# ---------------------------------------------------------------- front end filters
ORI = 6  # unsigned orientation bins over 180 degrees
SCALES = [(7, 1.2, 1), (11, 2.2, 1), (17, 4.5, 2)]  # (kernel, sigma, stride) per filter scale


def bar_kernel(k, sigma, phi):
    """Unit-L2, zero-mean oriented second derivative of a Gaussian (bar/line detector):
    responds to bright or dark bars of width ~2*sigma whose axis is perpendicular to phi,
    and is blind to linear brightness ramps (shading)."""
    ax = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    X, Y = np.meshgrid(ax, ax)  # X: horizontal, Y: vertical (image row)
    u = X * np.cos(phi) + Y * np.sin(phi)
    v = -X * np.sin(phi) + Y * np.cos(phi)
    g = (u * u / sigma ** 4 - 1.0 / sigma ** 2) * np.exp(-(u * u + v * v) / (2 * sigma ** 2))
    g -= g.mean()
    return g / np.sqrt((g * g).sum())


def ori_kernels(k, sigma):
    """(2*ORI, k, k): for each bin, the filter and its exact negative (both polarities)."""
    ws = []
    for j in range(ORI):
        g = bar_kernel(k, sigma, np.pi * j / ORI)
        ws.append(g)
        ws.append(-g)
    return np.stack(ws)


def mixer():
    """(ORI, 2*ORI, 1, 1): sums the two rectified polarities of each bin -> |response|."""
    w = np.zeros((ORI, 2 * ORI, 1, 1))
    for j in range(ORI):
        w[j, 2 * j, 0, 0] = 1.0
        w[j, 2 * j + 1, 0, 0] = 1.0
    return w


# ---------------------------------------------------------------- synthetic hand model
BG, HAND, LINE = 0.35, 0.85, 0.42
S = 4  # supersampling factor of the drawing canvas


def _coords(n):
    yy, xx = np.mgrid[0:n, 0:n]
    return (xx + 0.5) / S, (yy + 0.5) / S


def _seg_mask(X, Y, p0, p1, w):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1) if L2 > 1e-9 else np.zeros_like(X)
    d = np.sqrt((X - x0 - t * dx) ** 2 + (Y - y0 - t * dy) ** 2)
    return d <= w / 2


def capsule(img, p0, p1, w, val=HAND):
    X, Y = _coords(img.shape[0])
    img[_seg_mask(X, Y, p0, p1, w)] = val
    return img


def darkline(img, p0, p1, w):
    return capsule(img, p0, p1, w, LINE)


def ellipse(img, c, rx, ry, rot=0.0):
    X, Y = _coords(img.shape[0])
    X = X - c[0]
    Y = Y - c[1]
    a = np.deg2rad(rot)
    u = X * np.cos(a) + Y * np.sin(a)
    v = -X * np.sin(a) + Y * np.cos(a)
    img[(u / rx) ** 2 + (v / ry) ** 2 <= 1] = HAND
    return img


def disc(img, c, r):
    return ellipse(img, c, r, r)


def arc(img, c, r_out, r_in, a0, a1):
    X, Y = _coords(img.shape[0])
    X = X - c[0]
    Y = -(Y - c[1])  # y up, math convention, 0 deg = east
    r = np.sqrt(X ** 2 + Y ** 2)
    th = np.degrees(np.arctan2(Y, X)) % 360
    ok = ((th >= a0) & (th <= a1)) if a0 <= a1 else ((th >= a0) | (th <= a1))
    img[ok & (r <= r_out) & (r >= r_in)] = HAND
    return img


def _dirf(ang):  # 0 deg = up, positive = clockwise
    a = np.deg2rad(ang)
    return np.sin(a), -np.cos(a)


_THIN = [0.0]  # drawing-style switch: finger thickness varies between people


def finger(img, base, ang, length, w):
    d = _dirf(ang)
    return capsule(img, base, (base[0] + d[0] * length, base[1] + d[1] * length), max(1.2, w - _THIN[0]))


def palm(img, c=(14.0, 18.0), rx=4.6, ry=5.0):
    return ellipse(img, c, rx, ry)


def wrist(img, where="bottom", c=(14.0, 17.5)):
    if where == "bottom":
        return capsule(img, (c[0], c[1] + 3.5), (c[0] + 1.0, 29.0), 5.5)
    if where == "left":
        return capsule(img, (c[0] - 2.5, c[1] + 1.5), (-1.5, c[1] + 3.5), 5.0)
    if where == "top":
        return capsule(img, (c[0], c[1] - 3), (c[0] + 0.5, -1.0), 5.0)
    return capsule(img, (c[0] - 2.0, c[1] + 3.5), (c[0] - 5.5, 29.0), 5.0)


def fistblock(img, c=(14.0, 17.5)):
    """folded fingers over the palm: block on top of the palm + crease lines"""
    ellipse(img, (c[0], c[1] - 2.6), 4.8, 3.4)
    for y in (c[1] - 4.6, c[1] - 3.1, c[1] - 1.6):
        darkline(img, (c[0] - 3.8, y), (c[0] + 3.8, y - 0.5), 0.8)
    return img


BASE = {"i": (11.0, 14.2), "m": (13.1, 13.7), "r": (15.2, 14.0), "p": (17.1, 14.8)}
LEN = {"i": 10.5, "m": 11.5, "r": 10.5, "p": 8.0}
WID = {"i": 2.3, "m": 2.4, "r": 2.3, "p": 2.0}


def up(img, which, ang, dl=0.0, dw=0.0, base=None):
    b = base or BASE[which]
    return finger(img, b, ang, LEN[which] + dl, WID[which] + dw)


def sep(img, b, a, l):  # dark separation line between two touching fingers
    d = _dirf(a)
    return darkline(img, b, (b[0] + d[0] * l, b[1] + d[1] * l), 0.8)


LETTERS = {}


def L(name):
    def deco(f):
        LETTERS[name] = f
        return f
    return deco


@L("A")  # fist, thumb vertical along the left side
def _(img):
    palm(img); fistblock(img)
    capsule(img, (9.6, 18.5), (9.0, 10.0), 2.5)
    darkline(img, (10.8, 17.5), (10.5, 11.0), 0.8)
    wrist(img)


@L("B")  # four fingers together up, thumb folded across the palm
def _(img):
    palm(img)
    up(img, "i", -4, dw=0.4); up(img, "m", 0, dw=0.4); up(img, "r", 4, dw=0.4); up(img, "p", 10, dw=0.3)
    sep(img, (12.05, 13.95), -2, 9.0); sep(img, (14.15, 13.85), 2, 9.0); sep(img, (16.15, 14.4), 7, 7.0)
    darkline(img, (10.0, 17.2), (14.4, 16.6), 0.9)
    wrist(img)


@L("C")  # open curve, gap toward the viewer's right
def _(img):
    arc(img, (14.0, 14.5), 7.6, 4.4, 35, 325)
    wrist(img, "bottomleft", (14.0, 14.5))


@L("D")  # index up, other fingertips meet the thumb in a small ring
def _(img):
    palm(img); fistblock(img)
    up(img, "i", -2, base=(11.4, 14.0))
    arc(img, (14.6, 16.2), 3.2, 1.3, 0, 360)
    wrist(img)


@L("E")  # fingers curled down over the thumb
def _(img):
    palm(img); fistblock(img)
    for x in (11.8, 14.0, 16.2):
        darkline(img, (x, 12.6), (x + 0.3, 15.6), 0.8)
    darkline(img, (10.0, 17.0), (16.6, 16.6), 0.9)
    wrist(img)


@L("F")  # thumb+index ring, three fingers up
def _(img):
    palm(img)
    arc(img, (11.2, 16.0), 3.0, 1.2, 0, 360)
    up(img, "m", 2); up(img, "r", 8); up(img, "p", 15)
    wrist(img)


@L("G")  # index horizontal to the viewer's right, thumb above it
def _(img):
    c = (11.5, 17.5)
    palm(img, c, 4.6, 5.0); fistblock(img, c)
    finger(img, (11.5, 14.6), 90, 10.5, 2.2)
    finger(img, (10.6, 12.4), 78, 6.0, 2.2)
    wrist(img, "left", c)


@L("H")  # index+middle together horizontal right
def _(img):
    c = (11.5, 18.0)
    palm(img, c, 4.6, 5.0); fistblock(img, c)
    finger(img, (11.5, 13.8), 90, 10.5, 2.2)
    finger(img, (11.5, 16.0), 92, 10.5, 2.2)
    darkline(img, (11.6, 14.9), (21.4, 15.3), 0.8)
    finger(img, (10.6, 11.8), 78, 6.0, 2.2)
    wrist(img, "left", c)


@L("I")  # pinky up only
def _(img):
    palm(img); fistblock(img)
    up(img, "p", 8)
    wrist(img)


@L("K")  # index+middle spread up, thumb up between them
def _(img):
    palm(img); fistblock(img)
    up(img, "i", -14, base=(11.2, 14.0))
    up(img, "m", 16, base=(14.4, 13.6))
    finger(img, (12.8, 14.6), 0, 7.0, 2.1)
    wrist(img)


@L("L")  # index up, thumb out to the viewer's left
def _(img):
    palm(img); fistblock(img)
    up(img, "i", -2, base=(11.2, 14.0))
    finger(img, (10.4, 17.0), -80, 7.5, 2.5)
    wrist(img)


@L("M")  # fist, thumb tucked under three fingers (tip at the pinky side)
def _(img):
    palm(img); fistblock(img)
    darkline(img, (10.0, 16.2), (17.0, 15.9), 0.9)
    disc(img, (16.8, 16.0), 1.1)
    wrist(img)


@L("N")  # fist, thumb tucked under two fingers
def _(img):
    palm(img); fistblock(img)
    darkline(img, (10.0, 15.2), (16.0, 14.9), 0.9)
    disc(img, (15.4, 15.0), 1.1)
    wrist(img)


@L("O")  # closed ring
def _(img):
    arc(img, (14.0, 15.0), 7.0, 4.1, 0, 360)
    wrist(img, "bottomleft", (14.0, 15.0))


@L("P")  # K rotated down: index right, middle down, thumb up
def _(img):
    c = (12.0, 17.0)
    palm(img, c, 4.6, 5.0); fistblock(img, c)
    finger(img, (12.0, 14.0), 95, 9.5, 2.1)
    finger(img, (13.6, 15.6), 160, 7.0, 2.1)
    finger(img, (11.4, 12.8), 55, 5.0, 2.1)
    wrist(img, "left", c)


@L("Q")  # G rotated down: index and thumb pointing down
def _(img):
    c = (14.0, 13.0)
    palm(img, c, 4.6, 5.0); fistblock(img, c)
    finger(img, (13.2, 16.5), 185, 9.5, 2.2)
    finger(img, (15.6, 15.8), 168, 6.0, 2.2)
    wrist(img, "top", c)


@L("R")  # index and middle crossed
def _(img):
    palm(img); fistblock(img)
    up(img, "i", -10, base=(11.5, 14.0))
    up(img, "m", 8, base=(12.9, 14.0))
    darkline(img, (12.3, 13.6), (14.4, 4.5), 0.8)
    wrist(img)


@L("S")  # fist, thumb across the front
def _(img):
    palm(img); fistblock(img)
    capsule(img, (9.6, 16.2), (15.4, 15.6), 2.4)
    darkline(img, (9.6, 14.8), (15.4, 14.2), 0.8)
    wrist(img)


@L("T")  # fist, thumb tip between index and middle
def _(img):
    palm(img); fistblock(img)
    disc(img, (12.2, 13.4), 1.2)
    darkline(img, (10.6, 14.4), (13.6, 13.8), 0.8)
    wrist(img)


@L("U")  # index+middle together up
def _(img):
    palm(img); fistblock(img)
    up(img, "i", -3, base=(11.4, 14.0), dw=0.3)
    up(img, "m", 3, base=(13.4, 13.6), dw=0.3)
    darkline(img, (12.4, 13.8), (12.8, 3.5), 0.8)
    wrist(img)


@L("V")  # index+middle spread up
def _(img):
    palm(img); fistblock(img)
    up(img, "i", -15, base=(11.3, 14.0))
    up(img, "m", 13, base=(13.5, 13.6))
    wrist(img)


@L("W")  # three fingers spread up
def _(img):
    palm(img); fistblock(img)
    up(img, "i", -15, base=(11.2, 14.0), dl=-0.5)
    up(img, "m", 0, base=(13.2, 13.6))
    up(img, "r", 15, base=(15.2, 14.0), dl=-0.5)
    wrist(img)


@L("X")  # fist with a hooked index
def _(img):
    palm(img); fistblock(img)
    capsule(img, (11.8, 14.4), (11.2, 10.2), 2.2)
    capsule(img, (11.2, 10.2), (13.8, 9.2), 2.1)
    wrist(img)


@L("Y")  # thumb left and pinky right, rest folded
def _(img):
    palm(img); fistblock(img)
    finger(img, (10.4, 17.5), -85, 8.5, 2.5)
    up(img, "p", 22)
    wrist(img)


# poses each letter is drawn in: (rotation deg, dx, dy, scale)
VARIANTS = [(r, dx, dy, sc, thin) for thin in (0.0, 0.5) for sc in (0.75, 0.90, 1.05, 1.20)
            for (dx, dy) in ((0, 0), (-3, 0), (3, 0), (0, -3), (0, 3), (-3, -3), (3, 3), (-3, 3), (3, -3))
            for r in (-12, -6, 0, 6, 12)]
V = len(VARIANTS)


def render(name, rot=0.0, dx=0.0, dy=0.0, scale=1.0, thin=0.0):
    img = np.full((28 * S, 28 * S), BG, np.float64)
    _THIN[0] = thin
    LETTERS[name](img)
    _THIN[0] = 0.0
    if (rot, dx, dy, scale) != (0.0, 0.0, 0.0, 1.0):
        n = img.shape[0]
        X, Y = _coords(n)
        X = X - 14.0
        Y = Y - 14.0
        a = np.deg2rad(-rot)
        u = (X * np.cos(a) - Y * np.sin(a)) / scale + 14.0 - dx
        v = (X * np.sin(a) + Y * np.cos(a)) / scale + 14.0 - dy
        ui = np.clip((u * S - 0.5).astype(int), 0, n - 1)
        vi = np.clip((v * S - 0.5).astype(int), 0, n - 1)
        img = img[vi, ui]
    img = img.reshape(28, S, 28, S).mean(axis=(1, 3))
    return gauss_blur(img, 0.8)  # photos downsampled from high res have soft edges


def gauss_blur(a, sigma):
    k = int(np.ceil(3 * sigma)) * 2 + 1
    ax = np.arange(k) - k // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2)); g /= g.sum()
    a = np.apply_along_axis(lambda r: np.convolve(r, g, "same"), 0, a)
    return np.apply_along_axis(lambda r: np.convolve(r, g, "same"), 1, a)


def highpass_kernel(k=15, sigma=3.5):
    """delta - Gaussian: removes low-frequency luminance (exposure vignette, shading
    gradients, smooth background ramps) while keeping edges, bars and creases."""
    ax = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    X, Y = np.meshgrid(ax, ax)
    g = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    g[k // 2, k // 2] -= 1.0
    return -g  # image - blur(image)


# ---------------------------------------------------------------- front end (numpy)
def conv(x, W, b, stride=1, padding=0):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
                     for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def pool(x, size, stride, kind):
    n, c, h, w = x.shape
    (sh, sw), (th, tw) = size, stride
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw]
                    for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if kind == "max" else win.mean(0)


def frontend(images):
    """images (N,1,28,28) -> list of three level feature arrays (N, dim_L)."""
    images = conv(images, highpass_kernel()[None, None], np.zeros(1), 1, 7)
    mags = []
    for (k, sigma, st) in SCALES:
        e = conv(images, ori_kernels(k, sigma)[:, None, :, :], np.zeros(2 * ORI), st, k // 2)
        mags.append(conv(np.maximum(e, 0.0), mixer(), np.zeros(ORI)))
    a = pool(mags[0], (2, 2), (2, 2), "avg")                               # (N,6,14,14)
    b = pool(pool(mags[1], (3, 3), (1, 1), "avg"), (4, 4), (3, 3), "avg")   # (N,6,8,8)
    c = pool(pool(mags[2], (3, 3), (1, 1), "avg"), (4, 4), (2, 2), "avg")   # (N,6,5,5)
    return [o.reshape(len(o), -1) for o in (a, b, c)]


def level_mask(h, w):
    """hand-chosen spatial weight: the wrist strip and the frame border carry no letter
    information (every hand has a wrist; clutter lives at the border), so templates ignore them."""
    m = np.ones((h, w))
    m[-1, :] = 0.0
    if h >= 8:
        m[-2, :] = 0.3
    m[0, :] *= 0.3
    m[:, 0] *= 0.3
    m[:, -1] *= 0.3
    return m


def nuisance_dirs(h, w):
    """Two analytic, letter-independent directions in (ORI, h, w) space, from formulas only:
    the tangent ring of any silhouette outline, and the vertical wrist bar at bottom centre.
    Removing each template's component along them stops the match being driven by structure
    that every hand shares."""
    gs = []
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(cx, cy)
    th = (np.degrees(np.arctan2(yy - cy, xx - cx)) % 180)
    ring = np.zeros((ORI, h, w))
    ok = (r > 0.3) & (r < 0.95)
    binn = np.round(th / 30).astype(int) % ORI
    ring[binn, yy, xx] = np.where(ok, 1.0, 0.0)[None][0] * 0 + np.where(ok, 1.0, 0.0)
    gs.append(ring)
    wrist = np.zeros((ORI, h, w))
    wrist[0, int(h * 0.6):, max(0, w // 2 - 1):w // 2 + 1] = 1.0
    gs.append(wrist)
    out = []
    for g in gs:
        g = g / (np.sqrt((g * g).sum()) + 1e-12)
        out.append(g.ravel())
    return out


def self_normalise(f, shape):
    """per template, by itself alone: apply the hand-chosen mask, remove the two analytic
    nuisance directions, remove its own per-channel mean, then unit L2 norm."""
    h, w = shape[1], shape[2]
    g = f.reshape((len(f),) + shape)
    g = g * level_mask(h, w)[None]
    g = g.reshape((len(f),) + shape)
    g = g - g.mean(axis=(1, 2), keepdims=True)
    g = g.reshape(len(f), -1)
    return g / (np.sqrt((g * g).sum(1, keepdims=True)) + 1e-12)


def main():
    # ---- templates: every generated drawing becomes one row, normalised by itself
    imgs = np.stack([render(NAMES[k], *v) for k in range(K) for v in VARIANTS])[:, None].astype(np.float32)
    feats = [np.concatenate([frontend(imgs[i:i + 400])[j] for i in range(0, len(imgs), 400)]) for j in range(3)]
    shapes = [(ORI, 14, 14), (ORI, 8, 8), (ORI, 5, 5)]
    tmpl = [self_normalise(f, s) for f, s in zip(feats, shapes)]
    T = K * V

    # ---- network graph
    layers = [
        {"name": "hp", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [15, 15], "stride": 1, "padding": 7},
        {"name": "e1", "op": "conv", "inputs": ["hp"], "out": 2 * ORI, "kernel": [7, 7], "stride": 1, "padding": 3},
        {"name": "r1", "op": "relu", "inputs": ["e1"]},
        {"name": "m1", "op": "conv", "inputs": ["r1"], "out": ORI, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "e2", "op": "conv", "inputs": ["hp"], "out": 2 * ORI, "kernel": [11, 11], "stride": 1, "padding": 5},
        {"name": "r2", "op": "relu", "inputs": ["e2"]},
        {"name": "m2", "op": "conv", "inputs": ["r2"], "out": ORI, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "e3", "op": "conv", "inputs": ["hp"], "out": 2 * ORI, "kernel": [17, 17], "stride": 2, "padding": 8},
        {"name": "r3", "op": "relu", "inputs": ["e3"]},
        {"name": "m3", "op": "conv", "inputs": ["r3"], "out": ORI, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "a2", "op": "avgpool", "inputs": ["m1"], "size": [2, 2], "stride": [2, 2]},
        {"name": "b1", "op": "avgpool", "inputs": ["m2"], "size": [3, 3], "stride": [1, 1]},
        {"name": "b2", "op": "avgpool", "inputs": ["b1"], "size": [4, 4], "stride": [3, 3]},
        {"name": "c1", "op": "avgpool", "inputs": ["m3"], "size": [3, 3], "stride": [1, 1]},
        {"name": "c2", "op": "avgpool", "inputs": ["c1"], "size": [4, 4], "stride": [2, 2]},
        {"name": "dA", "op": "dense", "inputs": ["a2"], "out": T},
        {"name": "dB", "op": "dense", "inputs": ["b2"], "out": T},
        {"name": "dC", "op": "dense", "inputs": ["c2"], "out": T},
        {"name": "sT", "op": "add", "inputs": ["dA", "dB", "dC"]},
        {"name": "rs", "op": "reshape", "inputs": ["sT"], "shape": [K, 1, V]},
        {"name": "mx", "op": "maxpool", "inputs": ["rs"], "size": [1, V], "stride": [1, V]},
        {"name": "scores", "op": "reshape", "inputs": ["mx"], "shape": [K]},
    ]

    weights = {
        "hp.W": highpass_kernel()[None, None].astype(np.float32), "hp.b": np.zeros(1, np.float32),
        "e1.W": ori_kernels(7, 1.2)[:, None].astype(np.float32), "e1.b": np.zeros(2 * ORI, np.float32),
        "e2.W": ori_kernels(11, 2.2)[:, None].astype(np.float32), "e2.b": np.zeros(2 * ORI, np.float32),
        "e3.W": ori_kernels(17, 4.5)[:, None].astype(np.float32), "e3.b": np.zeros(2 * ORI, np.float32),
        "m1.W": mixer().astype(np.float32), "m1.b": np.zeros(ORI, np.float32),
        "m2.W": mixer().astype(np.float32), "m2.b": np.zeros(ORI, np.float32),
        "m3.W": mixer().astype(np.float32), "m3.b": np.zeros(ORI, np.float32),
        "dA.W": tmpl[0].astype(np.float32), "dA.b": np.zeros(T, np.float32),
        "dB.W": tmpl[1].astype(np.float32), "dB.b": np.zeros(T, np.float32),
        "dC.W": tmpl[2].astype(np.float32), "dC.b": np.zeros(T, np.float32),
    }

    (OUT / "network.json").write_text(json.dumps({"layers": layers}))
    np.savez(OUT / "weights.npz", **weights)
    total = sum(int(np.prod(v.shape)) for v in weights.values())
    print(f"wrote {OUT/'network.json'} and {OUT/'weights.npz'}: {total:,} weights, {T} templates")


main()
