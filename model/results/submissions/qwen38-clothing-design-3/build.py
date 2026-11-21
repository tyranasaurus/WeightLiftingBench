"""Build weights for a hand-designed Fashion-MNIST-style classifier.

Everything here comes from formulas and from general knowledge of what clothing,
shoes and bags look like in a shop photo: silhouettes are drawn part by part
(rectangles / trapezoids / ellipses / arcs for body, sleeves, legs, soles,
shafts, handles), turned into zero-mean unit-norm templates, and used as
normalized-cross-correlation filters.  A second branch computes structural
measurements (extent profiles, openness/holes, texture, leg gap, asymmetry)
with fixed filters and pools them; a hand-written rule table turns those into
per-class bonuses.  No data is read and nothing is fitted.

Usage: python3 build.py OUT_DIR
"""
import json
import math
import sys

import numpy as np

# --------------------------------------------------------------------------
# tiny rasterizer: shapes drawn in normalised frame coordinates [0,1]x[0,1]
# (x right, y down), supersampled then box-downsampled for soft edges.
# --------------------------------------------------------------------------
SS = 4          # supersampling factor
N = 28          # frame size in pixels


def _grid():
    return np.zeros((N * SS, N * SS), np.float32)


def _coords():
    n = N * SS
    t = (np.arange(n) + 0.5) / n          # cell centres in [0,1]
    return t


def rect(g, x0, y0, x1, y1, v=1.0):
    t = _coords()
    mx = (t >= x0) & (t < x1)
    my = (t >= y0) & (t < y1)
    g[np.ix_(my, mx)] = v
    return g


def ell(g, cx, cy, rx, ry, v=1.0):
    t = _coords()
    dx = (t - cx) / rx
    dy = (t - cy) / ry
    m = (dx[None, :] ** 2 + dy[:, None] ** 2) <= 1.0
    g[m] = v
    return g


def poly(g, pts, v=1.0):
    """Even-odd scanline fill of a polygon given in normalised coords."""
    n = N * SS
    t = _coords()
    ys = [p[1] for p in pts]
    y0i = max(int(math.floor(min(ys) * n)) - 1, 0)
    y1i = min(int(math.ceil(max(ys) * n)) + 1, n)
    for iy in range(y0i, y1i):
        y = t[iy]
        xs = []
        for i in range(len(pts)):
            xa, ya = pts[i]
            xb, yb = pts[(i + 1) % len(pts)]
            if (ya <= y < yb) or (yb <= y < ya):
                xs.append(xa + (y - ya) * (xb - xa) / (yb - ya))
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            a = int(math.floor(xs[k] * n))
            b = int(math.ceil(xs[k + 1] * n))
            g[iy, max(a, 0):min(b, n)] = v
    return g


def bar(g, x0, y0, x1, y1, v=1.0):
    """Thick axis-aligned bar with rounded ends (same as rect, kept for clarity)."""
    return rect(g, x0, y0, x1, y1, v)


def strap(g, xa, ya, xb, yb, w, v=1.0):
    """Thick line segment (a strap / sole line / seam)."""
    n = N * SS
    t = _coords()
    yy, xx = np.meshgrid(t, t, indexing="ij")
    dx, dy = xb - xa, yb - ya
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        u = np.zeros_like(xx)
    else:
        u = np.clip(((xx - xa) * dx + (yy - ya) * dy) / L2, 0.0, 1.0)
    px, py = xa + u * dx, ya + u * dy
    d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    g[d <= w * 0.5] = v
    return g


def arc(g, cx, cy, r, w, a0, a1, v=1.0):
    """Circular arc band, angles in degrees, 0 = up, clockwise."""
    t = _coords()
    yy, xx = np.meshgrid(t, t, indexing="ij")
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ang = np.degrees(np.arctan2(xx - cx, -(yy - cy))) % 360.0
    m = (d >= r - w * 0.5) & (d <= r + w * 0.5)
    if a0 <= a1:
        m &= (ang >= a0) & (ang <= a1)
    else:
        m &= (ang >= a0) | (ang <= a1)
    g[m] = v
    return g


def cut(g, *a, **kw):
    """Same as rect but writes 0 (a notch / opening)."""
    kw["v"] = 0.0
    return rect(g, *a, **kw)


def down(g):
    return g.reshape(N, SS, N, SS).mean(axis=(1, 3))


def mirror_x(a):
    return a[:, ::-1]


def blur(a, sigma):
    if sigma <= 0:
        return a
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, a)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)
    return out


def finish(g, sigma=0.65):
    """Supersampled drawing -> 28x28 template, zero-mean, unit-norm for NCC."""
    a = blur(down(g), sigma).astype(np.float64)
    return a


def ncc_weights(t, n_pix):
    u = t - t.mean()
    s = np.sqrt((u * u).sum())
    if s < 1e-9:
        return np.zeros_like(u)
    return (u / s / math.sqrt(n_pix)).astype(np.float32)


# --------------------------------------------------------------------------
# silhouette models, drawn from general knowledge of each garment class.
# Each returns a list of 28x28 drawings (variants: cut, pose, orientation).
# --------------------------------------------------------------------------
def d_tshirt():
    out = []
    # classic tee: body + short angled sleeves + neck notch
    g = _grid()
    rect(g, 0.31, 0.02, 0.69, 1.00)
    poly(g, [(0.31, 0.02), (0.13, 0.10), (0.17, 0.32), (0.31, 0.28)])
    poly(g, [(0.69, 0.02), (0.87, 0.10), (0.83, 0.32), (0.69, 0.28)])
    ell(g, 0.5, 0.00, 0.09, 0.06, 0.0)
    out.append(finish(g))
    # boxy / wide tee
    g = _grid()
    rect(g, 0.26, 0.05, 0.74, 0.97)
    poly(g, [(0.26, 0.05), (0.07, 0.13), (0.11, 0.40), (0.26, 0.36)])
    poly(g, [(0.74, 0.05), (0.93, 0.13), (0.89, 0.40), (0.74, 0.36)])
    ell(g, 0.5, 0.03, 0.10, 0.06, 0.0)
    out.append(finish(g))
    # tank top / sleeveless
    g = _grid()
    rect(g, 0.33, 0.10, 0.67, 1.00)
    rect(g, 0.35, 0.00, 0.43, 0.14)
    rect(g, 0.57, 0.00, 0.65, 0.14)
    ell(g, 0.5, 0.10, 0.10, 0.10, 0.0)
    out.append(finish(g))
    # fitted tee, narrow body
    g = _grid()
    rect(g, 0.34, 0.02, 0.66, 1.00)
    poly(g, [(0.34, 0.02), (0.18, 0.09), (0.21, 0.26), (0.34, 0.23)])
    poly(g, [(0.66, 0.02), (0.82, 0.09), (0.79, 0.26), (0.66, 0.23)])
    ell(g, 0.5, 0.00, 0.08, 0.05, 0.0)
    out.append(finish(g))
    # sleeves angled down towards mid-body
    g = _grid()
    rect(g, 0.32, 0.02, 0.68, 1.00)
    poly(g, [(0.32, 0.02), (0.14, 0.14), (0.22, 0.42), (0.32, 0.34)])
    poly(g, [(0.68, 0.02), (0.86, 0.14), (0.78, 0.42), (0.68, 0.34)])
    ell(g, 0.5, 0.00, 0.09, 0.05, 0.0)
    out.append(finish(g))
    # raglan / wide-neck tee, boxy short body
    g = _grid()
    poly(g, [(0.30, 0.04), (0.70, 0.04), (0.72, 0.60), (0.68, 0.96), (0.32, 0.96), (0.28, 0.60)])
    poly(g, [(0.30, 0.04), (0.12, 0.16), (0.18, 0.36), (0.30, 0.30)])
    poly(g, [(0.70, 0.04), (0.88, 0.16), (0.82, 0.36), (0.70, 0.30)])
    strap(g, 0.34, 0.04, 0.44, 0.24, 0.03, 0.3)
    strap(g, 0.66, 0.04, 0.56, 0.24, 0.03, 0.3)
    out.append(finish(g))
    # long tee, narrow shoulders
    g = _grid()
    rect(g, 0.35, 0.00, 0.65, 1.00)
    poly(g, [(0.35, 0.00), (0.21, 0.06), (0.24, 0.22), (0.35, 0.19)])
    poly(g, [(0.65, 0.00), (0.79, 0.06), (0.76, 0.22), (0.65, 0.19)])
    ell(g, 0.5, 0.00, 0.07, 0.04, 0.0)
    out.append(finish(g))
    # polo: collar + short placket
    g = _grid()
    rect(g, 0.31, 0.03, 0.69, 0.98)
    poly(g, [(0.31, 0.03), (0.15, 0.10), (0.19, 0.30), (0.31, 0.26)])
    poly(g, [(0.69, 0.03), (0.85, 0.10), (0.81, 0.30), (0.69, 0.26)])
    poly(g, [(0.42, 0.03), (0.50, 0.12), (0.58, 0.03), (0.58, 0.00), (0.42, 0.00)], 0.0)
    rect(g, 0.49, 0.10, 0.51, 0.30, 0.4)
    out.append(finish(g))
    return out


def d_trouser():
    out = []
    # straight leg
    g = _grid()
    rect(g, 0.32, 0.00, 0.68, 0.34)
    rect(g, 0.32, 0.30, 0.47, 1.00)
    rect(g, 0.53, 0.30, 0.68, 1.00)
    out.append(finish(g))
    # skinny
    g = _grid()
    rect(g, 0.36, 0.00, 0.64, 0.32)
    poly(g, [(0.36, 0.28), (0.48, 0.28), (0.46, 1.00), (0.38, 1.00)])
    poly(g, [(0.52, 0.28), (0.64, 0.28), (0.62, 1.00), (0.54, 1.00)])
    out.append(finish(g))
    # wide leg
    g = _grid()
    rect(g, 0.28, 0.00, 0.72, 0.40)
    rect(g, 0.28, 0.36, 0.47, 1.00)
    rect(g, 0.53, 0.36, 0.72, 1.00)
    out.append(finish(g))
    # cropped / shorts-ish
    g = _grid()
    rect(g, 0.33, 0.12, 0.67, 0.45)
    rect(g, 0.33, 0.42, 0.47, 0.88)
    rect(g, 0.53, 0.42, 0.67, 0.88)
    out.append(finish(g))
    return out


def d_pullover():
    out = []
    # sweater: wide body + long sleeves hanging along the sides
    g = _grid()
    rect(g, 0.27, 0.04, 0.73, 0.96)
    rect(g, 0.10, 0.06, 0.27, 0.82)
    rect(g, 0.73, 0.06, 0.90, 0.82)
    ell(g, 0.5, 0.02, 0.09, 0.05, 0.0)
    out.append(finish(g))
    # hoodie: wider, sleeves to the hem, hood bump
    g = _grid()
    rect(g, 0.24, 0.06, 0.76, 0.94)
    rect(g, 0.05, 0.08, 0.24, 0.88)
    rect(g, 0.76, 0.08, 0.95, 0.88)
    ell(g, 0.5, 0.06, 0.14, 0.09, 1.0)
    ell(g, 0.5, 0.03, 0.08, 0.05, 0.0)
    out.append(finish(g))
    # cropped sweater
    g = _grid()
    rect(g, 0.28, 0.12, 0.72, 0.90)
    rect(g, 0.12, 0.14, 0.28, 0.86)
    rect(g, 0.72, 0.14, 0.88, 0.86)
    ell(g, 0.5, 0.10, 0.09, 0.05, 0.0)
    out.append(finish(g))
    return out


def d_dress():
    out = []
    # A-line: narrow bodice, flaring skirt
    g = _grid()
    poly(g, [(0.38, 0.00), (0.62, 0.00), (0.60, 0.34), (0.40, 0.34)])
    poly(g, [(0.40, 0.30), (0.60, 0.30), (0.76, 1.00), (0.24, 1.00)])
    poly(g, [(0.38, 0.02), (0.28, 0.10), (0.31, 0.24), (0.39, 0.20)])
    poly(g, [(0.62, 0.02), (0.72, 0.10), (0.69, 0.24), (0.61, 0.20)])
    out.append(finish(g))
    # column dress with straps
    g = _grid()
    rect(g, 0.36, 0.08, 0.64, 1.00)
    rect(g, 0.38, 0.00, 0.44, 0.10)
    rect(g, 0.56, 0.00, 0.62, 0.10)
    poly(g, [(0.36, 0.70), (0.64, 0.70), (0.68, 1.00), (0.32, 1.00)])
    out.append(finish(g))
    # fit-and-flare with sleeves
    g = _grid()
    poly(g, [(0.36, 0.00), (0.64, 0.00), (0.62, 0.36), (0.38, 0.36)])
    poly(g, [(0.38, 0.32), (0.62, 0.32), (0.72, 1.00), (0.28, 1.00)])
    poly(g, [(0.36, 0.02), (0.24, 0.10), (0.27, 0.30), (0.37, 0.26)])
    poly(g, [(0.64, 0.02), (0.76, 0.10), (0.73, 0.30), (0.63, 0.26)])
    out.append(finish(g))
    return out


def d_coat():
    out = []
    # long coat: wide body, full sleeves, front opening line
    g = _grid()
    rect(g, 0.25, 0.02, 0.75, 0.98)
    rect(g, 0.07, 0.04, 0.25, 0.88)
    rect(g, 0.75, 0.04, 0.93, 0.88)
    rect(g, 0.485, 0.14, 0.515, 0.98, 0.25)
    strap(g, 0.40, 0.04, 0.49, 0.28, 0.035, 0.3)   # lapels
    strap(g, 0.60, 0.04, 0.51, 0.28, 0.035, 0.3)
    ell(g, 0.5, 0.00, 0.08, 0.05, 0.0)
    out.append(finish(g))
    # jacket: shorter, boxy
    g = _grid()
    rect(g, 0.22, 0.08, 0.78, 0.92)
    rect(g, 0.04, 0.10, 0.22, 0.86)
    rect(g, 0.78, 0.10, 0.96, 0.86)
    rect(g, 0.485, 0.18, 0.515, 0.92, 0.25)
    out.append(finish(g))
    # slim long coat
    g = _grid()
    rect(g, 0.30, 0.00, 0.70, 1.00)
    rect(g, 0.15, 0.03, 0.30, 0.82)
    rect(g, 0.70, 0.03, 0.85, 0.82)
    rect(g, 0.485, 0.12, 0.515, 1.00, 0.25)
    poly(g, [(0.30, 0.60), (0.70, 0.60), (0.74, 1.00), (0.26, 1.00)])
    out.append(finish(g))
    return out


def d_shirt():
    out = []
    # button shirt: narrow body, slim separate sleeves, placket
    g = _grid()
    rect(g, 0.32, 0.02, 0.68, 0.98)
    rect(g, 0.18, 0.04, 0.30, 0.92)
    rect(g, 0.70, 0.04, 0.82, 0.92)
    rect(g, 0.49, 0.06, 0.51, 0.98, 0.55)
    poly(g, [(0.44, 0.02), (0.50, 0.09), (0.56, 0.02), (0.56, 0.00), (0.44, 0.00)], 0.0)
    out.append(finish(g))
    # long-sleeve shirt with pronounced collar V
    g = _grid()
    rect(g, 0.32, 0.02, 0.68, 0.98)
    rect(g, 0.18, 0.04, 0.30, 0.92)
    rect(g, 0.70, 0.04, 0.82, 0.92)
    poly(g, [(0.42, 0.02), (0.50, 0.16), (0.58, 0.02), (0.58, 0.00), (0.42, 0.00)], 0.0)
    rect(g, 0.49, 0.14, 0.51, 0.98, 0.55)
    out.append(finish(g))
    # blouse: soft wide body, long thin sleeves
    g = _grid()
    poly(g, [(0.30, 0.02), (0.70, 0.02), (0.74, 0.60), (0.68, 0.98), (0.32, 0.98), (0.26, 0.60)])
    rect(g, 0.16, 0.04, 0.27, 0.94)
    rect(g, 0.73, 0.04, 0.84, 0.94)
    out.append(finish(g))
    # slim shirt, sleeves separated from the body by a seam gap
    g = _grid()
    rect(g, 0.34, 0.02, 0.66, 0.98)
    rect(g, 0.20, 0.05, 0.31, 0.90)
    rect(g, 0.69, 0.05, 0.80, 0.90)
    rect(g, 0.315, 0.10, 0.335, 0.88, 0.0)
    rect(g, 0.665, 0.10, 0.685, 0.88, 0.0)
    poly(g, [(0.44, 0.02), (0.50, 0.10), (0.56, 0.02), (0.56, 0.00), (0.44, 0.00)], 0.0)
    rect(g, 0.49, 0.08, 0.51, 0.98, 0.5)
    out.append(finish(g))
    # oxford shirt: sleeves held slightly away from the body
    g = _grid()
    rect(g, 0.33, 0.02, 0.67, 0.98)
    poly(g, [(0.30, 0.03), (0.17, 0.06), (0.15, 0.90), (0.28, 0.90), (0.30, 0.30)])
    poly(g, [(0.70, 0.03), (0.83, 0.06), (0.85, 0.90), (0.72, 0.90), (0.70, 0.30)])
    poly(g, [(0.43, 0.02), (0.50, 0.12), (0.57, 0.02), (0.57, 0.00), (0.43, 0.00)], 0.0)
    rect(g, 0.49, 0.10, 0.51, 0.98, 0.5)
    out.append(finish(g))
    return out


def d_sandal():
    out = []
    # flip-flop / slide: thin sole + strap
    g = _grid()
    rect(g, 0.02, 0.56, 0.98, 0.64)
    strap(g, 0.30, 0.56, 0.50, 0.40, 0.05)
    strap(g, 0.70, 0.56, 0.50, 0.40, 0.05)
    out.append(finish(g))
    # high heel: sole ramp + heel post + ankle strap
    g = _grid()
    poly(g, [(0.04, 0.66), (0.72, 0.44), (0.76, 0.52), (0.10, 0.76)])
    rect(g, 0.74, 0.44, 0.88, 0.80)
    strap(g, 0.30, 0.44, 0.46, 0.24, 0.045)
    strap(g, 0.62, 0.40, 0.46, 0.24, 0.045)
    out.append(finish(g))
    # gladiator: sole + several straps
    g = _grid()
    rect(g, 0.03, 0.62, 0.97, 0.70)
    strap(g, 0.20, 0.62, 0.34, 0.30, 0.04)
    strap(g, 0.50, 0.62, 0.34, 0.30, 0.04)
    strap(g, 0.80, 0.62, 0.62, 0.32, 0.04)
    strap(g, 0.34, 0.30, 0.62, 0.32, 0.04)
    out.append(finish(g))
    # wedge sandal
    g = _grid()
    poly(g, [(0.04, 0.72), (0.90, 0.40), (0.94, 0.68), (0.10, 0.78)])
    strap(g, 0.28, 0.60, 0.44, 0.36, 0.05)
    strap(g, 0.66, 0.48, 0.50, 0.30, 0.05)
    out.append(finish(g))
    return out


def d_sneaker():
    out = []
    # low runner, ankle at right
    g = _grid()
    rect(g, 0.02, 0.58, 0.98, 0.68)
    poly(g, [(0.02, 0.58), (0.06, 0.46), (0.30, 0.38), (0.58, 0.32), (0.78, 0.34),
             (0.90, 0.40), (0.98, 0.46), (0.98, 0.58)])
    out.append(finish(g))
    # same, ankle at left
    out.append(mirror_x(out[-1]))
    # high top
    g = _grid()
    rect(g, 0.02, 0.62, 0.98, 0.72)
    poly(g, [(0.02, 0.62), (0.06, 0.50), (0.30, 0.42), (0.52, 0.38), (0.60, 0.36),
             (0.60, 0.14), (0.84, 0.14), (0.86, 0.40), (0.98, 0.50), (0.98, 0.62)])
    out.append(finish(g))
    out.append(mirror_x(out[-1]))
    # chunky trainer, taller upper
    g = _grid()
    rect(g, 0.02, 0.54, 0.98, 0.66)
    poly(g, [(0.02, 0.54), (0.08, 0.40), (0.34, 0.30), (0.62, 0.24), (0.82, 0.28),
             (0.94, 0.36), (0.98, 0.44), (0.98, 0.54)])
    out.append(finish(g))
    out.append(mirror_x(out[-1]))
    return out


def d_bag():
    out = []
    # tote: trapezoid body + two handle arcs
    g = _grid()
    poly(g, [(0.22, 0.34), (0.78, 0.34), (0.72, 0.96), (0.28, 0.96)])
    arc(g, 0.40, 0.36, 0.16, 0.045, 180, 360)
    arc(g, 0.60, 0.36, 0.16, 0.045, 180, 360)
    out.append(finish(g))
    # wide clutch / messenger: flat box, thin strap
    g = _grid()
    rect(g, 0.06, 0.36, 0.94, 0.68)
    strap(g, 0.10, 0.36, 0.06, 0.20, 0.03)
    out.append(finish(g))
    # handbag: rounded body + single handle
    g = _grid()
    ell(g, 0.5, 0.66, 0.30, 0.26)
    arc(g, 0.5, 0.42, 0.20, 0.05, 180, 360)
    out.append(finish(g))
    # backpack: tall rounded box + top loop
    g = _grid()
    poly(g, [(0.26, 0.22), (0.74, 0.22), (0.78, 0.60), (0.74, 0.92), (0.26, 0.92), (0.22, 0.60)])
    arc(g, 0.5, 0.22, 0.10, 0.045, 180, 360)
    out.append(finish(g))
    # satchel: flat rectangle with a diagonal shoulder strap
    g = _grid()
    rect(g, 0.12, 0.34, 0.88, 0.86)
    strap(g, 0.20, 0.34, 0.78, 0.86, 0.035, 0.45)
    out.append(finish(g))
    # messenger with flap seam
    g = _grid()
    rect(g, 0.16, 0.30, 0.84, 0.88)
    rect(g, 0.16, 0.56, 0.84, 0.60, 0.35)
    strap(g, 0.16, 0.30, 0.10, 0.14, 0.03)
    out.append(finish(g))
    # duffel: wide rounded barrel + two short handles
    g = _grid()
    poly(g, [(0.10, 0.42), (0.90, 0.42), (0.94, 0.62), (0.90, 0.80), (0.10, 0.80), (0.06, 0.62)])
    arc(g, 0.38, 0.44, 0.10, 0.04, 180, 360)
    arc(g, 0.62, 0.44, 0.10, 0.04, 180, 360)
    out.append(finish(g))
    # tall shopper tote
    g = _grid()
    rect(g, 0.24, 0.22, 0.76, 0.94)
    arc(g, 0.5, 0.24, 0.16, 0.045, 180, 360)
    out.append(finish(g))
    return out


def d_boot():
    out = []
    # heeled ankle boot, toe left, shaft right
    g = _grid()
    poly(g, [(0.56, 0.12), (0.82, 0.12), (0.84, 0.52), (0.88, 0.66), (0.90, 0.78),
             (0.10, 0.78), (0.12, 0.70), (0.34, 0.62), (0.52, 0.54)])
    rect(g, 0.62, 0.78, 0.82, 0.88)
    out.append(finish(g))
    out.append(mirror_x(out[-1]))
    # flat chelsea boot
    g = _grid()
    poly(g, [(0.58, 0.18), (0.84, 0.18), (0.86, 0.56), (0.92, 0.68), (0.94, 0.76),
             (0.08, 0.76), (0.10, 0.68), (0.36, 0.62), (0.54, 0.54)])
    rect(g, 0.08, 0.76, 0.94, 0.84)
    out.append(finish(g))
    out.append(mirror_x(out[-1]))
    # tall heeled boot
    g = _grid()
    poly(g, [(0.54, 0.04), (0.80, 0.04), (0.82, 0.50), (0.86, 0.66), (0.88, 0.80),
             (0.10, 0.80), (0.12, 0.72), (0.34, 0.64), (0.50, 0.54)])
    rect(g, 0.60, 0.80, 0.80, 0.92)
    out.append(finish(g))
    out.append(mirror_x(out[-1]))
    return out


DRAW = [d_tshirt, d_trouser, d_pullover, d_dress, d_coat,
        d_sandal, d_shirt, d_sneaker, d_bag, d_boot]
V = 8  # template slots per class (variants repeated to fill)


def all_templates():
    """(10*V, 28, 28) float drawings."""
    ts = []
    for c in range(10):
        vs = DRAW[c]()
        while len(vs) < V:
            vs = vs + vs
        ts.extend(vs[:V])
    return np.array(ts, np.float64)


# --------------------------------------------------------------------------
# hand-written rule table over structural measurements
# --------------------------------------------------------------------------
def band_profile(lo, hi, n=14, scale=1.0):
    """Zero-mean 'mass between rows/cols lo..hi' matched filter on n bins."""
    v = np.zeros(n)
    v[lo:hi] = 1.0
    v -= v.mean()
    return v * scale


RULE = {}
# row-profile weights (14 bins, top->bottom), on the contrast-normalised profile
RULE["r14"] = {
    0: band_profile(1, 13), 1: band_profile(0, 14), 2: band_profile(1, 12),
    3: band_profile(0, 14), 4: band_profile(0, 13), 5: band_profile(6, 11),
    6: band_profile(1, 13), 7: band_profile(5, 11), 8: band_profile(4, 11),
    9: band_profile(3, 12),
}
# column-profile weights (14 bins, left->right)
RULE["c14"] = {
    0: band_profile(2, 12), 1: band_profile(4, 10), 2: band_profile(1, 13),
    3: band_profile(3, 11), 4: band_profile(1, 13), 5: band_profile(0, 14),
    6: band_profile(2, 12), 7: band_profile(0, 14), 8: band_profile(2, 12),
    9: band_profile(1, 13),
}
# trouser: two legs -> mass in two narrow column bands, none at edges/centre
tr = np.zeros(14)
tr[4:7] = 1.0
tr[8:11] = 1.0
tr -= tr.mean()
RULE["c14"][1] = tr
# widths of top / middle / bottom row bands (presence cells), sum-zero so they
# respond to the width *profile*, not to overall size
RULE["wid"] = {
    0: (+0.020, -0.006, -0.014), 1: (0.0, 0.0, 0.0),
    2: (+0.010, 0.000, -0.010), 3: (-0.016, -0.006, +0.022),
    4: (+0.010, 0.000, -0.010), 5: (-0.016, +0.004, +0.012),
    6: (+0.012, -0.004, -0.008), 7: (-0.014, +0.002, +0.012),
    8: (-0.030, +0.014, +0.008), 9: (-0.012, +0.002, +0.010),
}
# sleeve / shoulder mass: (topL, topR, bottomL, bottomR) of outer column bands
RULE["sle"] = {
    0: (0.0, 0.0, -0.35, -0.35), 1: (0.0, 0.0, 0.0, 0.0),
    2: (0.0, 0.0, +0.30, +0.30), 3: (0.0, 0.0, -0.25, -0.25),
    4: (0.0, 0.0, +0.30, +0.30), 5: (0.0, 0.0, 0.0, 0.0),
    6: (0.0, 0.0, +0.25, +0.25), 7: (0.0, 0.0, 0.0, 0.0),
    8: (-0.15, -0.15, -0.10, -0.10), 9: (0.0, 0.0, 0.0, 0.0),
}
# aspect: mean width minus mean height (footwear wide+short, trouser narrow+tall)
RULE["aspect"] = {0: -0.012, 1: -0.020, 2: -0.010, 3: -0.005, 4: -0.008,
                  5: +0.045, 6: -0.012, 7: +0.040, 8: +0.020, 9: +0.025}
# scalar bonuses: leggap, open3, open5, openH, openV, tex, asym  (see layers)
RULE["leggap"] = {0: -0.01, 1: +0.05, 2: -0.01, 3: -0.03, 4: -0.02,
                  5: +0.01, 6: -0.01, 7: -0.02, 8: -0.01, 9: -0.02}
# opening deficit (opening(x) - x, <= 0): how much thin bright structure
# (straps, soles, sleeves, legs) a window of that size eats away
RULE["open3"] = {0: -0.06, 1: -0.04, 2: -0.04, 3: -0.02, 4: -0.04,
                 5: -0.55, 6: -0.08, 7: +0.20, 8: +0.12, 9: +0.20}
RULE["open5"] = {0: -0.07, 1: -0.15, 2: -0.07, 3: -0.05, 4: -0.07,
                 5: -0.60, 6: -0.07, 7: +0.20, 8: +0.15, 9: +0.20}
RULE["open9"] = {0: -0.09, 1: -0.12, 2: -0.09, 3: -0.09, 4: -0.09,
                 5: -0.18, 6: -0.09, 7: +0.06, 8: +0.06, 9: +0.06}
# hole fill (closing(x) - x, >= 0): dark gaps narrower than the window
RULE["hf5"] = {0: (-0.10, 0.00), 1: (-0.20, +1.20), 2: (-0.15, -0.10), 3: (-0.20, -0.20),
               4: (-0.15, -0.15), 5: (+0.50, +0.50), 6: (-0.20, -0.20), 7: (-0.20, -0.20),
               8: (+0.50, 0.00), 9: (-0.20, -0.20)}
RULE["hf9"] = {0: (-0.10, 0.00), 1: (-0.10, +0.30), 2: (-0.15, -0.10), 3: (+0.20, -0.20),
               4: (-0.15, -0.15), 5: (+0.40, +0.40), 6: (-0.30, -0.30), 7: (-0.20, -0.20),
               8: (+0.60, 0.00), 9: (-0.20, -0.20)}
# texture in the centre column band (button placket / collar of a shirt)
RULE["texmid"] = {0: -0.07, 1: -0.03, 2: +0.02, 3: -0.03, 4: +0.04,
                  5: -0.02, 6: +0.10, 7: -0.03, 8: -0.03, 9: -0.03}
# darkness of the centre vertical line (coat / shirt front opening)
RULE["cline"] = {0: -0.06, 1: -0.04, 2: -0.10, 3: -0.04, 4: +0.30,
                 5: -0.02, 6: +0.08, 7: -0.04, 8: -0.04, 9: -0.04}
RULE["tex"] = {0: -0.04, 1: -0.02, 2: +0.03, 3: +0.01, 4: +0.02,
               5: -0.02, 6: +0.04, 7: -0.02, 8: -0.03, 9: -0.02}
RULE["asym"] = {0: -0.07, 1: -0.07, 2: -0.07, 3: -0.05, 4: -0.07,
                5: +0.07, 6: -0.05, 7: +0.05, 8: -0.05, 9: +0.20}
# bottom-band edge energy (sole line)
RULE["texbot"] = {0: -0.10, 1: -0.10, 2: -0.10, 3: -0.10, 4: -0.10,
                  5: +0.20, 6: -0.10, 7: +0.30, 8: -0.08, 9: +0.20}

TEMPLATE_SCALE = 1.0
RULE_SCALE = 1.0


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------
def build(out_dir):
    layers = []
    W = {}

    def add_layer(name, op, inputs, **settings):
        d = {"name": name, "op": op, "inputs": inputs}
        d.update(settings)
        layers.append(d)

    # --- base maps -------------------------------------------------------
    # saturated presence map: 0 on background, ~constant on any item pixel,
    # so dark uppers and bright soles count equally (shape, not brightness)
    W["pa1.W"] = 5.0 * np.ones((1, 1, 1, 1), np.float32)
    W["pa1.b"] = np.array([-0.30], np.float32)
    add_layer("pa1", "conv", ["image"], out=1, kernel=[1, 1], stride=1, padding=0)
    add_layer("pr1", "relu", ["pa1"])
    W["pa2.W"] = 5.0 * np.ones((1, 1, 1, 1), np.float32)
    W["pa2.b"] = np.array([-1.30], np.float32)
    add_layer("pa2", "conv", ["image"], out=1, kernel=[1, 1], stride=1, padding=0)
    add_layer("pr2", "relu", ["pa2"])
    W["pr2n.W"] = -np.ones((1, 1, 1, 1), np.float32)
    W["pr2n.b"] = np.zeros(1, np.float32)
    add_layer("pr2n", "conv", ["pr2"], out=1, kernel=[1, 1], stride=1, padding=0)
    add_layer("pr", "add", ["pr1", "pr2n"])
    # smoothed 28x28 brightness map and its contrast normalisation
    gk = np.exp(-0.5 * (np.array([-1.0, 0.0, 1.0]) / 0.8) ** 2)
    gk /= gk.sum()
    kW = np.outer(gk, gk).reshape(1, 1, 3, 3).astype(np.float32)
    W["sm.W"] = kW
    W["sm.b"] = np.zeros(1, np.float32)
    add_layer("sm", "conv", ["image"], out=1, kernel=[3, 3], stride=1, padding=1)
    add_layer("n1", "norm", ["sm"])
    # smoothed presence map, normalised: the shape substrate
    W["psm.W"] = kW
    W["psm.b"] = np.zeros(1, np.float32)
    add_layer("psm", "conv", ["pr"], out=1, kernel=[3, 3], stride=1, padding=1)
    add_layer("n1p", "norm", ["psm"])
    add_layer("occ2", "avgpool", ["pr"], size=[2, 2], stride=[2, 2])
    add_layer("n2", "norm", ["occ2"])
    add_layer("occ7", "avgpool", ["pr"], size=[4, 4], stride=[4, 4])
    add_layer("n7", "norm", ["occ7"])

    # --- template branches ------------------------------------------------
    # difference-of-Gaussians map: outlines and interior structure, flat
    # background suppressed, so NCC is not dominated by background agreement
    r5 = np.arange(-2, 3)
    g1 = np.exp(-0.5 * (r5 / 0.8) ** 2)
    g2 = np.exp(-0.5 * (r5 / 2.2) ** 2)
    K = np.outer(g1, g1) / (g1.sum() ** 2) - np.outer(g2, g2) / (g2.sum() ** 2)
    K = K.astype(np.float32)
    W["dog.W"] = K.reshape(1, 1, 5, 5)
    W["dog.b"] = np.zeros(1, np.float32)
    add_layer("dog", "conv", ["n1p"], out=1, kernel=[5, 5], stride=1, padding=2)
    add_layer("n1d", "norm", ["dog"])

    def dog_filter(t):
        p = np.pad(t, 2, mode="constant")
        out = np.zeros_like(t)
        for dy in range(5):
            for dx in range(5):
                out += K[dy, dx] * p[dy:dy + t.shape[0], dx:dx + t.shape[1]]
        return out

    T = all_templates()                      # (10*V, 28, 28)
    t28 = np.array([ncc_weights(t, 28 * 28) for t in T])
    t28e = np.array([ncc_weights(dog_filter(t), 28 * 28) for t in T])
    t14 = np.array([ncc_weights(blur(t.reshape(14, 2, 14, 2).mean((1, 3)), 0.4), 14 * 14)
                    for t in T])
    t7 = np.array([ncc_weights(blur(t.reshape(7, 4, 7, 4).mean((1, 3)), 0.3), 7 * 7)
                   for t in T])
    W["t28.W"] = t28.reshape(t28.shape[0], -1).astype(np.float32)
    W["t28.b"] = np.zeros(t28.shape[0], np.float32)
    add_layer("t28", "dense", ["n1p"], out=t28.shape[0])
    W["t7.W"] = (0.8 * t7).reshape(t7.shape[0], -1).astype(np.float32)
    W["t7.b"] = np.zeros(t7.shape[0], np.float32)
    add_layer("t7", "dense", ["n7"], out=t7.shape[0])
    W["t28e.W"] = (0.6 * t28e).reshape(t28e.shape[0], -1).astype(np.float32)
    W["t28e.b"] = np.zeros(t28e.shape[0], np.float32)
    add_layer("t28e", "dense", ["n1d"], out=t28e.shape[0])
    W["t14.W"] = t14.reshape(t14.shape[0], -1).astype(np.float32)
    W["t14.b"] = np.zeros(t14.shape[0], np.float32)
    add_layer("t14", "dense", ["n2"], out=t14.shape[0])
    add_layer("tsum", "add", ["t28", "t28e", "t14", "t7"])
    add_layer("tsh", "reshape", ["tsum"], shape=[10, V, 1])
    add_layer("tmx", "maxpool", ["tsh"], size=[V, 1], stride=[V, 1])
    add_layer("tmax", "reshape", ["tmx"], shape=[10])
    final_inputs = ["tmax"]

    # --- structural measurements ------------------------------------------
    # 1-D profiles of the normalised 14x14 map
    add_layer("r14", "avgpool", ["n2"], size=[1, 14], stride=[1, 14])
    add_layer("r14f", "reshape", ["r14"], shape=[14])
    add_layer("c14", "avgpool", ["n2"], size=[14, 1], stride=[14, 1])
    add_layer("c14f", "reshape", ["c14"], shape=[14])
    Wr = np.array([RULE["r14"][c] for c in range(10)], np.float32) * 0.04
    W["gr.W"], W["gr.b"] = Wr, np.zeros(10, np.float32)
    add_layer("gr", "dense", ["r14f"], out=10)
    Wc = np.array([RULE["c14"][c] for c in range(10)], np.float32) * 0.04
    W["gc.W"], W["gc.b"] = Wc, np.zeros(10, np.float32)
    add_layer("gc", "dense", ["c14f"], out=10)
    final_inputs += ["gr", "gc"]

    # presence map and band widths / heights
    W["sh1.W"] = np.ones((1, 1, 1, 1), np.float32)
    W["sh1.b"] = np.array([-0.12], np.float32)
    add_layer("sh1", "conv", ["occ2"], out=1, kernel=[1, 1], stride=1, padding=0)
    add_layer("p14", "relu", ["sh1"])
    W["sh2.W"] = np.ones((1, 1, 1, 1), np.float32)
    W["sh2.b"] = np.array([-0.55], np.float32)
    add_layer("sh2", "conv", ["occ2"], out=1, kernel=[1, 1], stride=1, padding=0)
    add_layer("p14h", "relu", ["sh2"])
    add_layer("brow", "maxpool", ["p14"], size=[5, 1], stride=[4, 1])
    add_layer("browf", "reshape", ["brow"], shape=[42])
    Ww = np.zeros((3, 42), np.float32)
    for b in range(3):
        Ww[b, b * 14:(b + 1) * 14] = 1.0
    W["gw.W"] = Ww
    W["gw.b"] = np.zeros(3, np.float32)
    add_layer("gw", "dense", ["browf"], out=3)
    Wgw = np.array([RULE["wid"][c] for c in range(10)], np.float32) * RULE_SCALE * 2.5
    W["gw2.W"], W["gw2.b"] = Wgw, np.zeros(10, np.float32)
    add_layer("gw2", "dense", ["gw"], out=10)
    # sleeve / shoulder mass in the outer column bands, top vs bottom half
    add_layer("sle", "avgpool", ["p14"], size=[6, 4], stride=[8, 10])
    add_layer("slef", "reshape", ["sle"], shape=[4])
    Wsl = np.array([RULE["sle"][c] for c in range(10)], np.float32)
    W["gsl.W"], W["gsl.b"] = Wsl, np.zeros(10, np.float32)
    add_layer("gsl", "dense", ["slef"], out=10)
    # extent counts: how many rows / columns carry any item mass
    add_layer("rm", "avgpool", ["p14"], size=[1, 14], stride=[1, 14])
    add_layer("rmf", "reshape", ["rm"], shape=[14])
    W["drx.W"] = np.eye(14, dtype=np.float32)
    W["drx.b"] = np.full(14, -0.15, np.float32)
    add_layer("drx", "dense", ["rmf"], out=14)
    add_layer("drxr", "relu", ["drx"])
    add_layer("cm", "avgpool", ["p14"], size=[14, 1], stride=[14, 1])
    add_layer("cmf", "reshape", ["cm"], shape=[14])
    W["dcx.W"] = np.eye(14, dtype=np.float32)
    W["dcx.b"] = np.full(14, -0.15, np.float32)
    add_layer("dcx", "dense", ["cmf"], out=14)
    add_layer("dcxr", "relu", ["dcx"])
    Wa = np.array([[RULE["aspect"][c] * 0.35] * 14 for c in range(10)], np.float32)
    W["gax.W"], W["gax.b"] = Wa, np.zeros(10, np.float32)
    add_layer("gax", "dense", ["dcxr"], out=10)
    Wah = np.array([[-RULE["aspect"][c] * 0.35] * 14 for c in range(10)], np.float32)
    W["gah.W"], W["gah.b"] = Wah, np.zeros(10, np.float32)
    add_layer("gah", "dense", ["drxr"], out=10)
    final_inputs += ["gw2", "gsl", "gax", "gah"]

    # leg gap: bottom band, sides present vs centre empty
    add_layer("bbot", "maxpool", ["p14h"], size=[5, 1], stride=[3, 1])
    add_layer("bbotf", "reshape", ["bbot"], shape=[56])
    wl = np.zeros((1, 56), np.float32)
    wl[0, 46:48] = 1.0     # inner edges of the two legs
    wl[0, 50:52] = 1.0
    wl[0, 48:50] = -2.0    # the crotch gap between them
    W["gl.W"], W["gl.b"] = wl, np.zeros(1, np.float32)
    add_layer("gl", "dense", ["bbotf"], out=1)
    Wgl = np.array([[RULE["leggap"][c]] for c in range(10)], np.float32)
    W["gl2.W"], W["gl2.b"] = Wgl, np.zeros(10, np.float32)
    add_layer("gl2", "dense", ["gl"], out=10)
    final_inputs.append("gl2")

    # openness: morphological closing minus identity, at three window sizes.
    # closing = dilate(erode()) realised as -max(-x) then max of that; the
    # reference crop keeps both operands the same shape.
    def centre_one(k):
        w = np.zeros((1, 1, k, k), np.float32)
        w[0, 0, k // 2, k // 2] = 1.0
        return w

    gs = np.exp(-0.5 * (np.arange(-2, 3) / 1.6) ** 2)
    gs /= gs.sum()
    W["sm2.W"] = np.outer(gs, gs).reshape(1, 1, 5, 5).astype(np.float32)
    W["sm2.b"] = np.zeros(1, np.float32)
    add_layer("sm2", "conv", ["n1p"], out=1, kernel=[5, 5], stride=1, padding=2)
    W["crp.W"] = centre_one(5)
    W["crp.b"] = np.zeros(1, np.float32)
    add_layer("crp", "conv", ["sm2"], out=1, kernel=[5, 5], stride=1, padding=0)
    W["neg.W"] = -np.ones((1, 1, 1, 1), np.float32)
    W["neg.b"] = np.zeros(1, np.float32)
    add_layer("neg", "conv", ["crp"], out=1, kernel=[1, 1], stride=1, padding=0)
    for tag, k in (("3", 3), ("5", 5), ("9", 9)):
        add_layer("mx" + tag, "maxpool", ["neg"], size=[k, k], stride=[1, 1])
        W["mn" + tag + ".W"] = -np.ones((1, 1, 1, 1), np.float32)
        W["mn" + tag + ".b"] = np.zeros(1, np.float32)
        add_layer("mn" + tag, "conv", ["mx" + tag], out=1, kernel=[1, 1], stride=1, padding=0)
        add_layer("cl" + tag, "maxpool", ["mn" + tag], size=[k, k], stride=[1, 1])
        rk = 2 * k - 1
        W["rf" + tag + ".W"] = centre_one(rk)
        W["rf" + tag + ".b"] = np.zeros(1, np.float32)
        add_layer("rf" + tag, "conv", ["neg"], out=1, kernel=[rk, rk], stride=1, padding=0)
        add_layer("ho" + tag, "add", ["cl" + tag, "rf" + tag])
        m = 24 - 2 * k + 2
        add_layer("os" + tag, "avgpool", ["ho" + tag], size=[m, m], stride=[m, m])
        add_layer("os" + tag + "f", "reshape", ["os" + tag], shape=[1])
        wv = np.array([[RULE["open" + tag][c]] for c in range(10)], np.float32)
        W["go" + tag + ".W"], W["go" + tag + ".b"] = wv, np.zeros(10, np.float32)
        add_layer("go" + tag, "dense", ["os" + tag + "f"], out=10)
        final_inputs.append("go" + tag)
        if k in (5, 9):
            # true closing (dilate then erode) minus identity: dark hole filling
            add_layer("dl" + tag, "maxpool", ["crp"], size=[k, k], stride=[1, 1])
            W["nd" + tag + ".W"] = -np.ones((1, 1, 1, 1), np.float32)
            W["nd" + tag + ".b"] = np.zeros(1, np.float32)
            add_layer("nd" + tag, "conv", ["dl" + tag], out=1, kernel=[1, 1], stride=1, padding=0)
            add_layer("er" + tag, "maxpool", ["nd" + tag], size=[k, k], stride=[1, 1])
            W["cx" + tag + ".W"] = -np.ones((1, 1, 1, 1), np.float32)
            W["cx" + tag + ".b"] = np.zeros(1, np.float32)
            add_layer("cx" + tag, "conv", ["er" + tag], out=1, kernel=[1, 1], stride=1, padding=0)
            add_layer("hf" + tag, "add", ["cx" + tag, "rf" + tag])
            add_layer("hs" + tag, "avgpool", ["hf" + tag], size=[m // 2, m], stride=[m // 2, m])
            add_layer("hs" + tag + "f", "reshape", ["hs" + tag], shape=[2])
            wh = np.array([RULE["hf" + tag][c] for c in range(10)], np.float32)
            W["gh" + tag + ".W"], W["gh" + tag + ".b"] = wh, np.zeros(10, np.float32)
            add_layer("gh" + tag, "dense", ["hs" + tag + "f"], out=10)
            final_inputs.append("gh" + tag)

    # texture: |laplacian| of the normalised map, global and in the bottom band
    lap = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], np.float32)
    W["lap.W"] = lap.reshape(1, 1, 3, 3)
    W["lap.b"] = np.zeros(1, np.float32)
    add_layer("lap", "conv", ["n1"], out=1, kernel=[3, 3], stride=1, padding=1)
    W["lapn.W"] = -np.ones((1, 1, 1, 1), np.float32)
    W["lapn.b"] = np.zeros(1, np.float32)
    add_layer("lapn", "conv", ["lap"], out=1, kernel=[1, 1], stride=1, padding=0)
    add_layer("lapr", "relu", ["lap"])
    add_layer("lapnr", "relu", ["lapn"])
    add_layer("texm", "add", ["lapr", "lapnr"])
    add_layer("texg", "avgpool", ["texm"], size=[28, 28], stride=[28, 28])
    add_layer("texgf", "reshape", ["texg"], shape=[1])
    Wgt = np.array([[RULE["tex"][c]] for c in range(10)], np.float32)
    W["gt.W"], W["gt.b"] = Wgt, np.zeros(10, np.float32)
    add_layer("gt", "dense", ["texgf"], out=10)
    final_inputs.append("gt")
    add_layer("texb", "avgpool", ["texm"], size=[7, 28], stride=[7, 28])
    add_layer("texbf", "reshape", ["texb"], shape=[4])
    wb = np.zeros((10, 4), np.float32)
    for c in range(10):
        wb[c, 2] = RULE["texbot"][c] * 0.5
        wb[c, 3] = RULE["texbot"][c] * 0.5
        wb[c, 0] = -RULE["texbot"][c] * 0.5
        wb[c, 1] = -RULE["texbot"][c] * 0.5
    W["gtb.W"], W["gtb.b"] = wb, np.zeros(10, np.float32)
    add_layer("gtb", "dense", ["texbf"], out=10)
    final_inputs.append("gtb")
    # texture in the centre column band (placket / collar)
    add_layer("texc", "avgpool", ["texm"], size=[20, 8], stride=[20, 8])
    add_layer("texcf", "reshape", ["texc"], shape=[3])
    wc = np.zeros((10, 3), np.float32)
    for c in range(10):
        wc[c, 1] = RULE["texmid"][c]
        wc[c, 0] = -RULE["texmid"][c] * 0.5
        wc[c, 2] = -RULE["texmid"][c] * 0.5
    W["gtc.W"], W["gtc.b"] = wc, np.zeros(10, np.float32)
    add_layer("gtc", "dense", ["texcf"], out=10)
    final_inputs.append("gtc")
    # front-opening seam: centre column pair darker than the two neighbours,
    # measured over the upper-middle rows (coat / shirt placket)
    add_layer("clv", "avgpool", ["n1"], size=[8, 4], stride=[4, 4])
    add_layer("clvf", "reshape", ["clv"], shape=[42])
    wl2 = np.zeros((10, 42), np.float32)
    for c in range(10):
        wl2[c, 10] = -RULE["cline"][c]         # centre band cols 12..15, rows 4..11
        wl2[c, 9] = RULE["cline"][c] * 0.5
        wl2[c, 11] = RULE["cline"][c] * 0.5
    W["gcl.W"], W["gcl.b"] = wl2, np.zeros(10, np.float32)
    add_layer("gcl", "dense", ["clvf"], out=10)
    final_inputs.append("gcl")

    # left/right height asymmetry (boot shaft on one side)
    add_layer("lr", "avgpool", ["n2"], size=[14, 7], stride=[14, 7])
    add_layer("lrf", "reshape", ["lr"], shape=[2])
    Wd = np.array([[1.0, -1.0], [-1.0, 1.0]], np.float32)
    W["gas.W"], W["gas.b"] = Wd, np.zeros(2, np.float32)
    add_layer("gas", "dense", ["lrf"], out=2)
    add_layer("gasr", "relu", ["gas"])
    Was = np.array([[RULE["asym"][c], RULE["asym"][c]] for c in range(10)], np.float32)
    W["gas2.W"], W["gas2.b"] = Was, np.zeros(10, np.float32)
    add_layer("gas2", "dense", ["gasr"], out=10)
    final_inputs.append("gas2")

    # --- combine ----------------------------------------------------------
    add_layer("scores", "add", final_inputs)

    net = {"layers": layers}
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **W)
    return net, W


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
