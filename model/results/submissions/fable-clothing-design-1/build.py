"""Builds the hand-designed clothing classifier: python3 build.py OUT_DIR

No data is read.  Every weight comes from the shape models typed below:

  * each class is described by a few parametric outline models (a top is a body polygon plus two sleeve
    polygons, a trouser is two legs under a waistband, a shoe is a side profile with the toe on the left, a bag
    is a body with a handle loop, ...), written from general knowledge of how these items look in shop photos;
  * the models are drawn many times with their proportions varied by a seeded random number generator, and each
    drawing is put through the same framing as the photos (trimmed, longest side scaled to 28, centred);
  * the network turns an image into a soft silhouette, blurs it, and compares it with every drawing
    (a dense layer whose rows are the blurred drawings; the score is minus the squared distance, up to a term
    that is the same for every drawing).  A max-pool takes the best drawing of each class.
"""
import json
import os
import sys

import numpy as np

S = 28                                   # image side
SEED = 20261119
DEG = np.pi / 180.0

# pixel centres
_jj, _ii = np.meshgrid(np.arange(S) + 0.5, np.arange(S) + 0.5)
PX = _jj.reshape(1, -1)                  # x (column) coordinate of each pixel centre
PY = _ii.reshape(1, -1)                  # y (row) coordinate


# ----------------------------------------------------------------------------------------------------------
# geometry: signed distances (negative inside) evaluated at the 784 pixel centres, for a batch of shapes
# ----------------------------------------------------------------------------------------------------------
def poly_sdf(V):
    """V: (B, M, 2) polygon vertices in pixel units -> (B, 784) signed distance."""
    B, M, _ = V.shape
    d2 = np.full((B, S * S), 1e18)
    inside = np.zeros((B, S * S), bool)
    for i in range(M):
        j = (i - 1) % M
        ax, ay = V[:, i, 0:1], V[:, i, 1:2]
        bx, by = V[:, j, 0:1], V[:, j, 1:2]
        ex, ey = bx - ax, by - ay
        wx, wy = PX - ax, PY - ay
        t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey + 1e-12), 0.0, 1.0)
        qx, qy = wx - ex * t, wy - ey * t
        d2 = np.minimum(d2, qx * qx + qy * qy)
        c1 = PY >= ay
        c2 = PY < by
        c3 = ex * wy > ey * wx
        inside ^= (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
    d = np.sqrt(d2)
    return np.where(inside, -d, d)


def line_sdf(P, r):
    """P: (B, K, 2) polyline in pixel units, r: (B,) half thickness in pixels -> (B, 784)."""
    B, K, _ = P.shape
    d2 = np.full((B, S * S), 1e18)
    for i in range(K - 1):
        ax, ay = P[:, i, 0:1], P[:, i, 1:2]
        bx, by = P[:, i + 1, 0:1], P[:, i + 1, 1:2]
        ex, ey = bx - ax, by - ay
        wx, wy = PX - ax, PY - ay
        t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey + 1e-12), 0.0, 1.0)
        qx, qy = wx - ex * t, wy - ey * t
        d2 = np.minimum(d2, qx * qx + qy * qy)
    return np.sqrt(d2) - r[:, None]


def col(*xs):
    """Stack per-sample coordinate arrays into (B, M)."""
    n = max(np.size(x) for x in xs)
    return np.stack([np.broadcast_to(np.asarray(x, float), (n,)) for x in xs], axis=1)


def pts(xs, ys):
    """Lists of per-sample x and y arrays -> (B, M, 2)."""
    return np.stack([col(*xs), col(*ys)], axis=2)


def mirror(V):
    """Mirror a right-hand part to the left (x -> -x)."""
    W = V[:, ::-1].copy()
    W[:, :, 0] = -W[:, :, 0]
    return W


class Shape:
    """A batch of drawings: filled polygons, thick lines, and cut-outs, in the item's own units."""

    def __init__(self, n):
        self.n = n
        self.polys, self.lines, self.cuts = [], [], []

    def poly(self, V):
        self.polys.append(V)

    def line(self, P, r):
        self.lines.append((P, np.broadcast_to(np.asarray(r, float), (self.n,)).copy()))

    def cut(self, V):
        self.cuts.append(V)

    def render(self):
        """Frame like the photos (trim, longest side -> 28 pixels, centre) and return coverage (B, 28, 28)."""
        lo = np.full((self.n, 2), 1e18)
        hi = np.full((self.n, 2), -1e18)
        for V in self.polys:
            lo = np.minimum(lo, V.min(1)); hi = np.maximum(hi, V.max(1))
        for P, r in self.lines:
            lo = np.minimum(lo, P.min(1) - r[:, None]); hi = np.maximum(hi, P.max(1) + r[:, None])
        size = hi - lo
        sc = S / np.maximum(size[:, 0], size[:, 1])
        off = S / 2.0 - (lo + hi) / 2.0 * sc[:, None]

        def to_px(V):
            return V * sc[:, None, None] + off[:, None, :]

        d = np.full((self.n, S * S), 1e18)
        for V in self.polys:
            d = np.minimum(d, poly_sdf(to_px(V)))
        for P, r in self.lines:
            d = np.minimum(d, line_sdf(to_px(P), r * sc))
        for V in self.cuts:
            d = np.maximum(d, -poly_sdf(to_px(V)))
        return np.clip(0.5 - d, 0.0, 1.0).reshape(self.n, S, S)


class Rand:
    """Seeded uniform sampler."""

    def __init__(self, seed, n):
        self.rs = np.random.RandomState(seed)
        self.n = n

    def u(self, a, b):
        return a + (b - a) * self.rs.random_sample(self.n)

    def flag(self, p):
        return self.rs.random_sample(self.n) < p


# ----------------------------------------------------------------------------------------------------------
# upper-body garments.  Units: garment length (shoulder at the neck to hem) = 100.  x = 0 is the centre line,
# y grows downwards from the shoulder line at the neck.
# ----------------------------------------------------------------------------------------------------------
def top_body(sh, Wn, dn, Wsh, ysh, Wch, yap, Wwa, ywa, Whem, L, hemc):
    """Body: neckline dip, shoulders, armpits, waist, hem (optionally curved down in the middle)."""
    xs = [-Wn / 2, -Wn / 4, 0 * Wn, Wn / 4, Wn / 2, Wsh / 2, Wch / 2, Wwa / 2, Whem / 2, Whem / 4, 0 * Wn,
          -Whem / 4, -Whem / 2, -Wwa / 2, -Wch / 2, -Wsh / 2]
    ys = [0 * dn, 0.75 * dn, dn, 0.75 * dn, 0 * dn, ysh, yap, ywa, L, L + 0.75 * hemc, L + hemc,
          L + 0.75 * hemc, L, ywa, yap, ysh]
    sh.poly(pts(xs, ys))


def top_sleeves(sh, Wsh, ysh, Ls, th, wt, wc):
    """Two sleeves: bands that start at the shoulder point and run along an axis th away from straight down,
    Ls long, wt wide at the top and wc wide at the cuff (widths measured across the band)."""
    sx, sy = Wsh / 2, ysh
    cox, coy = sx + Ls * np.sin(th), sy + Ls * np.cos(th)          # outer cuff corner
    cix, ciy = cox - wc * np.cos(th), coy + wc * np.sin(th)        # inner cuff corner
    tix, tiy = sx - wt * np.cos(th), sy + wt * np.sin(th) + 2      # inner top corner (inside the body)
    V = pts([sx, cox, cix, tix], [sy, coy, ciy, tiy])
    sh.poly(V)
    sh.poly(mirror(V))


def top_collar(sh, Wc, Wct, hc, depth):
    """A collar standing above the shoulder line: a trapezoid, Wc wide at the base, Wct at the top, hc high."""
    sh.poly(pts([-Wc / 2, -Wct / 2, Wct / 2, Wc / 2, Wc / 2, -Wc / 2], [0 * hc, -hc, -hc, 0 * hc, depth, depth]))


def top_hood(sh, Wh, hh, depth):
    """A hood lying behind the neck: a half ellipse above the shoulder line."""
    a = np.linspace(0, np.pi, 9)
    xs = [-(Wh / 2) * np.cos(t) for t in a] + [Wh / 2, -Wh / 2]
    ys = [-hh * np.sin(t) for t in a] + [depth, depth]
    sh.poly(pts(xs, ys))


def gen_tshirt(n, seed):
    r = Rand(seed, n); sh = Shape(n); L = 100.0
    Wn = r.u(20, 32); dn = r.u(-4, 6)                # rib collar standing slightly proud ... open neckline
    Wsh = r.u(50, 66); ysh = r.u(3, 9)
    Wch = Wsh + r.u(-6, 6); yap = r.u(26, 38)
    Whem = Wch + r.u(-8, 5); Wwa = (Wch + Whem) / 2 - r.u(0, 4)
    top_body(sh, Wn, dn, Wsh, ysh, Wch, yap, Wwa, 65.0, Whem, L, r.u(0, 3))
    top_sleeves(sh, Wsh, ysh, r.u(15, 30), r.u(25, 58) * DEG, r.u(22, 30), r.u(17, 25))
    return sh


def gen_tank(n, seed):
    """Sleeveless top: straps or narrow shoulders, deep neckline, armholes."""
    r = Rand(seed, n); sh = Shape(n); L = 100.0
    Wn = r.u(16, 34); strap = r.u(4, 14); dn = r.u(8, 30)
    Wsh = Wn + 2 * strap; ysh = r.u(0.5, 4)
    Wch = np.maximum(Wsh + r.u(0, 14), r.u(40, 62)); yap = r.u(22, 36)
    Whem = Wch + r.u(-6, 10); Wwa = (Wch + Whem) / 2 - r.u(0, 5)
    # armhole scooped in: an extra point between shoulder and armpit
    xa = Wsh / 2 + 0.2 * (Wch - Wsh) / 2; ya = 0.65 * yap
    xs = [-Wn / 2, -Wn / 4, 0 * Wn, Wn / 4, Wn / 2, Wsh / 2, xa, Wch / 2, Wwa / 2, Whem / 2, -Whem / 2, -Wwa / 2,
          -Wch / 2, -xa, -Wsh / 2]
    ys = [0 * dn, 0.75 * dn, dn, 0.75 * dn, 0 * dn, ysh, ya, yap, 65.0, L, L, 65.0, yap, ya, ysh]
    sh.poly(pts(xs, ys))
    return sh


def gen_longtop(n, seed, kind):
    """Long-sleeved tops as worn on a dummy: the sleeves hang down beside the body, slightly outwards.
    Pullovers, coats and shirts share this outline; what differs is typed per kind:
    'pullover': open neckline (a dip between the shoulders), sleeves reach about as far down as the hem;
    'coat': bulkier sleeves, often longer than its sleeves, and a collar or hood added by the caller;
    'shirt': slimmer sleeves, a shirt-tail hem that curves down in the middle, a small collar."""
    r = Rand(seed, n); sh = Shape(n); L = 100.0
    Wsh = r.u(56, 76); ysh = r.u(3, 10)
    th = r.u(0, 12) * DEG
    if kind == 'pullover':
        Wn = r.u(20, 38); dn = r.u(-4, 8); wt = r.u(13, 21); Ls = r.u(84, 106); hemc = r.u(0, 2)
    elif kind == 'coat':
        Wn = r.u(18, 30); dn = r.u(0, 4); wt = r.u(15, 24); Ls = r.u(68, 104); hemc = r.u(0, 3)
    else:
        Wn = r.u(16, 26); dn = r.u(0, 5); wt = r.u(12, 17); Ls = r.u(82, 102); hemc = r.u(1, 8)
    wc = wt * r.u(0.6, 0.85)
    Wch = np.maximum(Wsh - 2 * wt * r.u(0.35, 1.0), r.u(38, 46))     # body seen between the sleeves
    Whem = Wch + r.u(-8, 6)
    yap = r.u(28, 40)
    Wwa = (Wch + Whem) / 2 - r.u(0, 4)
    top_body(sh, Wn, dn, Wsh, ysh, Wch, yap, Wwa, 65.0, Whem, L, hemc)
    top_sleeves(sh, Wsh, ysh, Ls, th, wt, wc)
    return r, sh, Wn, dn


def gen_pullover(n, seed):
    return gen_longtop(n, seed, 'pullover')[1]


def gen_pullover_hood(n, seed):
    r, sh, Wn, dn = gen_longtop(n, seed, 'pullover')
    top_hood(sh, Wn + r.u(4, 14), r.u(6, 14), dn + 2)
    return sh


def gen_coat(n, seed):
    """Coat or jacket with a collar standing above the shoulders."""
    r, sh, Wn, dn = gen_longtop(n, seed, 'coat')
    Wc = r.u(16, 28)
    top_collar(sh, Wc, Wc - r.u(1, 7), r.u(5, 12), dn + 2)
    return sh


def gen_coat_hood(n, seed):
    r, sh, Wn, dn = gen_longtop(n, seed, 'coat')
    top_hood(sh, Wn + r.u(6, 18), r.u(7, 16), dn + 2)
    return sh


def gen_coat_plain(n, seed):
    return gen_longtop(n, seed, 'coat')[1]


def gen_shirt_long(n, seed):
    r, sh, Wn, dn = gen_longtop(n, seed, 'shirt')
    Wc = Wn + r.u(-2, 4)
    top_collar(sh, Wc, Wc - r.u(2, 8), r.u(3, 8), dn + 2)
    return sh


def gen_shirt_short(n, seed):
    """Short-sleeved shirt or blouse: a T-shirt outline with a collar."""
    r = Rand(seed, n); sh = Shape(n); L = 100.0
    Wn = r.u(16, 28); dn = r.u(0, 6)
    Wsh = r.u(50, 66); ysh = r.u(3, 9)
    Wch = Wsh + r.u(-6, 7); yap = r.u(26, 38)
    Whem = Wch + r.u(-6, 8); Wwa = (Wch + Whem) / 2 - r.u(0, 4)
    top_body(sh, Wn, dn, Wsh, ysh, Wch, yap, Wwa, 65.0, Whem, L, r.u(1, 8))
    top_sleeves(sh, Wsh, ysh, r.u(16, 34), r.u(25, 58) * DEG, r.u(22, 30), r.u(17, 26))
    Wc = Wn + r.u(-2, 4)
    top_collar(sh, Wc, Wc - r.u(2, 8), r.u(3, 8), dn + 2)
    return sh


# ----------------------------------------------------------------------------------------------------------
# dresses.  Units: length = 100.
# ----------------------------------------------------------------------------------------------------------
def dress_body(r, sh, Wn, dn, Wsh, ysh):
    """Bodice and skirt below a given shoulder line; returns chest width and armpit depth for sleeves."""
    n = r.n; L = 100.0
    Wch = r.u(27, 42); yap = r.u(14, 24)
    Wwa = Wch - r.u(0, 9); ywa = r.u(28, 40)
    Whem = Wwa + r.u(2, 42)
    Wmid = Wwa + (Whem - Wwa) * r.u(0.45, 0.85); ymid = r.u(58, 70)
    Wsh = np.minimum(Wsh, Wch + 6)
    xa = Wsh / 2 + 0.25 * (Wch - Wsh) / 2; ya = 0.6 * yap
    xs = [-Wn / 2, -Wn / 4, 0 * Wn, Wn / 4, Wn / 2, Wsh / 2, xa, Wch / 2, Wwa / 2, Wmid / 2, Whem / 2, -Whem / 2,
          -Wmid / 2, -Wwa / 2, -Wch / 2, -xa, -Wsh / 2]
    ys = [0 * dn, 0.75 * dn, dn, 0.75 * dn, 0 * dn, ysh, ya, yap, ywa, ymid, L + 0 * dn, L + 0 * dn, ymid, ywa, yap,
          ya, ysh]
    sh.poly(pts(xs, ys))
    return Wsh, Wch, yap


def gen_dress_sleeveless(n, seed):
    r = Rand(seed, n); sh = Shape(n)
    Wn = r.u(10, 22); strap = r.u(2, 9)
    dress_body(r, sh, Wn, r.u(5, 18), Wn + 2 * strap, r.u(0.5, 3))
    return sh


def gen_dress_short(n, seed):
    r = Rand(seed, n); sh = Shape(n)
    Wn = r.u(10, 20); ysh = r.u(1.5, 5)
    Wsh, Wch, yap = dress_body(r, sh, Wn, r.u(2, 12), r.u(26, 40), ysh)
    top_sleeves(sh, Wsh, ysh, r.u(8, 22), r.u(25, 60) * DEG, r.u(12, 17), r.u(9, 14))
    return sh


def gen_dress_long(n, seed):
    """Dress with long or three-quarter sleeves."""
    r = Rand(seed, n); sh = Shape(n)
    Wn = r.u(10, 20); ysh = r.u(1.5, 5)
    Wsh, Wch, yap = dress_body(r, sh, Wn, r.u(2, 12), r.u(26, 40), ysh)
    top_sleeves(sh, Wsh, ysh, r.u(34, 66), r.u(2, 14) * DEG, r.u(8, 12), r.u(6, 9))
    return sh


# ----------------------------------------------------------------------------------------------------------
# trousers.  Units: length = 100, waistband at y = 0.
# ----------------------------------------------------------------------------------------------------------
def gen_trouser(n, seed):
    r = Rand(seed, n); sh = Shape(n)
    Ww = r.u(28, 44); Wh = Ww + r.u(0, 9); yh = r.u(12, 22)
    yc = r.u(22, 36)                                 # crotch
    wl = r.u(10, 21)                                 # leg width at the hem
    xi = r.u(0.3, 9)                                 # half gap between the legs at the hem
    xo = xi + wl
    xk = (Wh / 2 + xo) / 2 + r.u(-2, 1); yk = (yh + 100) / 2
    xs = [-Ww / 2, Ww / 2, Wh / 2, xk, xo, xi, 0 * Ww, -xi, -xo, -xk, -Wh / 2]
    ys = [0 * Ww, 0 * Ww, yh, yk, 100 + 0 * Ww, 100 + 0 * Ww, yc, 100 + 0 * Ww, 100 + 0 * Ww, yk, yh]
    sh.poly(pts(xs, ys))
    return sh


def gen_shorts(n, seed):
    r = Rand(seed, n); sh = Shape(n)
    Ww = r.u(70, 95); Wh = Ww + r.u(0, 12); yh = r.u(20, 35)
    yc = r.u(45, 70); wl = r.u(38, 52); xi = r.u(1, 8); xo = xi + wl
    xs = [-Ww / 2, Ww / 2, Wh / 2, xo, xi, 0 * Ww, -xi, -xo, -Wh / 2]
    ys = [0 * Ww, 0 * Ww, yh, 100 + 0 * Ww, 100 + 0 * Ww, yc, 100 + 0 * Ww, 100 + 0 * Ww, yh]
    sh.poly(pts(xs, ys))
    return sh


# ----------------------------------------------------------------------------------------------------------
# shoes, seen from the side with the toe on the left.  Units: length = 100; x = 0 at the toe, 100 at the heel;
# h = height above the ground (turned into image rows by up()).
# ----------------------------------------------------------------------------------------------------------
def up(xs, hs):
    return pts(xs, [-np.asarray(h, float) for h in hs])


def gen_sneaker(n, seed):
    """Low shoe: flat sole, low toe, upper rising to the tongue, a dip at the ankle opening, heel tab."""
    r = Rand(seed, n); sh = Shape(n); z = np.zeros(n)
    ty = r.u(0, 4)                                   # toe spring
    ht = r.u(30, 52)                                 # tongue top
    xt = r.u(60, 74)
    hc = ht - r.u(2, 12)                             # ankle opening
    hb = hc + r.u(0, 9)                              # heel tab
    hv = r.u(15, 24)                                 # vamp over the toes
    xs = [z + 3, z + 22, z + 97, z + 100, z + 100, z + 98, z + 88, xt + 6, xt, xt - 12, z + 34, z + 14, z + 3, z]
    hs = [ty, z, z, z + 3, z + 12, hb, hc, ht - 2, ht, ht - r.u(4, 9), hv, hv - r.u(4, 8), ty + r.u(5, 9), ty + 3]
    sh.poly(up(xs, hs))
    return sh


def gen_sneaker_darksole(n, seed):
    """Sneaker whose sole is too dark to show: the framing still includes it, the silhouette does not."""
    sh = gen_sneaker(n, seed)
    r = Rand(seed + 5, n); z = np.zeros(n); c = r.u(5, 11)
    sh.cut(up([z - 5, z + 105, z + 105, z - 5], [z - 6, z - 6, c, c]))
    return sh


def gen_sneaker_soleline(n, seed):
    """Sneaker whose sole shows only as a thin bottom line: a dark band separates it from the upper."""
    sh = gen_sneaker(n, seed)
    r = Rand(seed + 9, n); z = np.zeros(n); c1 = r.u(1.5, 4); c2 = c1 + r.u(4, 9)
    sh.cut(up([z - 5, z + 105, z + 105, z - 5], [c1, c1, c2, c2]))
    return sh


def gen_hightop(n, seed):
    """High-top sneaker: the same sole and toe with the upper carried up around the ankle."""
    r = Rand(seed, n); sh = Shape(n); z = np.zeros(n)
    ty = r.u(0, 4); ht = r.u(50, 72); xt = r.u(52, 66); hv = r.u(16, 26)
    xs = [z + 3, z + 22, z + 97, z + 100, z + 100, z + 99, xt, xt - 10, z + 30, z + 12, z + 3, z]
    hs = [ty, z, z, z + 3, z + 12, ht + r.u(-6, 4), ht, ht - r.u(10, 22), hv, hv - r.u(4, 8), ty + r.u(5, 9),
          ty + 3]
    sh.poly(up(xs, hs))
    return sh


def gen_boot(n, seed, heel):
    """Ankle boot: low toe on the left, shaft over the rear half; flat sole or a raised block heel."""
    r = Rand(seed, n); sh = Shape(n); z = np.zeros(n)
    Ht = r.u(44, 100)                                # shaft top above the ground
    xf = r.u(40, 60)                                 # where the shaft front meets the instep
    hi = r.u(26, 40)                                 # instep height
    xft = xf + r.u(-6, 8)                            # shaft front at the top
    xbt = r.u(94, 102)                               # shaft back at the top
    hv = r.u(14, 22)
    if heel:
        hh = r.u(10, 36); hw = r.u(14, 32); xb = r.u(38, 55)
        hi = hi + 0.45 * hh; Ht = Ht + 0.3 * hh; ar = r.u(0.5, 0.9)
        bx = [z + 3, xb, 100 - hw - 5, 100 - hw, z + 99, z + 100]
        bh = [z, z, hh * ar, z, z, hh + 4]
    else:
        ty = r.u(0, 3)
        bx = [z + 3, z + 20, z + 99, z + 100]
        bh = [ty, z, z, z + 5]
    Ht = np.maximum(Ht, hi + 12)
    xs = bx + [xbt, xft, xf, z + 26, z + 8, z]
    hs = bh + [Ht + r.u(-6, 6), Ht, hi, hv, hv - r.u(4, 8), z + 4]
    sh.poly(up(xs, hs))
    return sh


def gen_boot_flat(n, seed):
    return gen_boot(n, seed, False)


def gen_boot_heel(n, seed):
    return gen_boot(n, seed, True)


def gen_sandal_flat(n, seed):
    """Flat sandal: a thin sole, a band or arch over the toes, thin straps up to the ankle."""
    r = Rand(seed, n); sh = Shape(n); z = np.zeros(n)
    st = r.u(3, 9)                                   # sole thickness
    sh.poly(up([z, z + 100, z + 100, z + 2], [z, z, st, st + r.u(0, 3)]))
    x1 = r.u(8, 22); x2 = x1 + r.u(18, 34); hp = st + r.u(6, 18); tk = r.u(1.5, 3.5)
    sh.line(up([x1, (x1 + x2) / 2 - 4, (x1 + x2) / 2 + 4, x2], [st, hp, hp, st]), tk)
    xa = r.u(78, 98); ha = st + r.u(12, 32)          # ankle point
    sh.line(up([r.u(45, 68), xa], [st, ha]), tk)
    sh.line(up([r.u(88, 100), xa, xa - r.u(0, 25)], [st, ha, ha + r.u(-4, 6)]), tk)
    return sh


def gen_sandal_slide(n, seed):
    """Slide or mule: sole with one solid band over the forefoot, sometimes a thicker wedge sole."""
    r = Rand(seed, n); sh = Shape(n); z = np.zeros(n)
    sf = r.u(3, 9); sb = sf + r.u(0, 16)
    sh.poly(up([z, z + 100, z + 100, z + 2], [z, z, sb, sf]))
    x1 = r.u(6, 20); x2 = x1 + r.u(25, 45); hp = r.u(10, 24)
    sh.poly(up([x1, x1 + 6, x2 - 8, x2, x2, x1], [sf + 4, sf + hp, sf + hp, sf + 6, z + 1, z + 1]))
    return sh


def gen_sandal_wedge(n, seed):
    """Wedge or platform sandal: a sole that is thick under the heel, with straps over toes and ankle."""
    r = Rand(seed, n); sh = Shape(n); z = np.zeros(n)
    sf = r.u(4, 14); sb = sf + r.u(10, 34)
    sh.poly(up([z, z + 100, z + 100, r.u(40, 60), z + 2], [z, z, sb, sf + (sb - sf) * r.u(0.2, 0.5), sf]))
    tk = r.u(1.5, 3.5)
    x1 = r.u(8, 22); x2 = x1 + r.u(18, 34); hp = sf + r.u(8, 18)
    sh.line(up([x1, (x1 + x2) / 2, x2], [sf, hp, sf + 3]), tk)
    xa = r.u(80, 98); ha = sb + r.u(12, 30)
    sh.line(up([r.u(50, 70), xa], [sf + 0.5 * (sb - sf), ha]), tk)
    sh.line(up([r.u(92, 100), xa, xa - r.u(0, 25)], [sb, ha, ha + r.u(-4, 6)]), tk)
    return sh


def gen_sandal_chunky(n, seed):
    """Sandal with broad straps: a sole (sometimes a platform or wedge), a broad band over the toes, a band
    over the instep and a heel cup joined to it by an ankle strap, with open gaps between the parts."""
    r = Rand(seed, n); sh = Shape(n); z = np.zeros(n)
    sf = r.u(4, 12); sb = sf + r.u(0, 14)

    def s(x):
        return sf + (sb - sf) * x / 100.0
    sh.poly(up([z, z + 100, z + 100, z + 2], [z, z, sb, sf]))
    x1 = r.u(4, 14); x2 = x1 + r.u(14, 28); h1 = r.u(8, 18)
    sh.poly(up([x1, x1 + 3, x2 - 3, x2], [s(x1) - 1, s(x1) + h1, s(x2) + h1 + r.u(0, 6), s(x2) - 1]))
    x3 = x2 + r.u(5, 14); x4 = x3 + r.u(10, 22); h2 = r.u(16, 32)
    sh.poly(up([x3, x3 + 4, x4, x4 + 2], [s(x3) - 1, s(x3) + h2 - 4, s(x4) + h2, s(x4) - 1]))
    x5 = r.u(74, 88); hc = r.u(14, 30)
    sh.poly(up([x5, x5 + 2, z + 100, z + 100], [s(x5) - 1, s(x5) + hc, sb + hc + r.u(0, 6), sb - 1]))
    sh.line(up([z + 97, x4], [sb + hc, s(x4) + h2]), r.u(2, 4))
    return sh


def gen_sandal_heel(n, seed):
    """High-heeled sandal: footbed climbing from the ball of the foot to a raised heel seat, a thin or block
    heel underneath it, a toe strap and an ankle strap."""
    r = Rand(seed, n); sh = Shape(n); z = np.zeros(n)
    hh = r.u(38, 75)                                 # heel seat height
    xs_ = r.u(82, 96)                                # heel seat position
    pf = r.u(2, 12)                                  # platform under the toes
    tk = r.u(1.5, 3.5)
    xball = r.u(28, 42)
    # footbed
    sh.line(up([z + 2, xball, (xball + xs_) / 2 + 4, xs_ + 4], [pf, pf + 1, pf + 0.5 * hh, hh]), r.u(2, 4.5))
    sh.poly(up([z, xball + 4, xball, z + 2], [z, z, pf, pf]))
    hw = np.where(r.flag(0.55), r.u(3, 7), r.u(10, 26))            # stiletto or block heel
    sh.poly(up([xs_ + 4 - hw, xs_ + 4, xs_ + 5, xs_ + 5 - 0.6 * hw], [hh, hh, z, z]))
    x1 = r.u(6, 16); x2 = x1 + r.u(12, 26); hp = pf + r.u(8, 18)
    sh.line(up([x1, (x1 + x2) / 2, x2], [pf, hp, pf + 2]), tk)
    xa = xs_ + r.u(-10, 4); ha = hh + r.u(14, 40)
    sh.line(up([xs_ + 3, xa, xa - r.u(8, 30)], [hh, ha, ha + r.u(-8, 6)]), tk)
    return sh


# ----------------------------------------------------------------------------------------------------------
# bags.  Units: body width = 100; y grows downwards from the top edge of the body.
# ----------------------------------------------------------------------------------------------------------
def bag_body(r, sh, hmax=105):
    n = r.n; z = np.zeros(n)
    H = r.u(45, hmax)
    Wt = r.u(72, 100); Wb = r.u(80, 100)             # top and bottom widths
    ct = r.u(1, 22); cb = r.u(1, 16)                 # how far the corners are cut (rounded tops / bottoms)
    xs = [-Wt / 2 + ct, Wt / 2 - ct, Wt / 2, Wb / 2, Wb / 2 - cb, -Wb / 2 + cb, -Wb / 2, -Wt / 2]
    ys = [z, z, 0.8 * ct, H - cb, H, H, H - cb, 0.8 * ct]
    sh.poly(pts(xs, ys))
    return H, Wt


def arc(x0, x1, h, k=9):
    """Handle loop from (x0, 0) to (x1, 0) rising h above the top edge: half-ellipse polyline."""
    a = np.linspace(0, np.pi, k)
    xs = [(x0 + x1) / 2 - (x1 - x0) / 2 * np.cos(t) for t in a]
    ys = [-h * np.sin(t) + 2 for t in a]
    return pts(xs, ys)


def gen_bag_plain(n, seed):
    """Clutch, pouch or case: body only."""
    r = Rand(seed, n); sh = Shape(n)
    bag_body(r, sh, 88)
    return sh


def gen_bag_handle(n, seed):
    """Handbag: body with a short handle loop above the top edge."""
    r = Rand(seed, n); sh = Shape(n)
    H, Wt = bag_body(r, sh)
    hx = Wt / 2 * r.u(0.35, 0.8)
    sh.line(arc(-hx, hx, r.u(18, 62)), r.u(1.5, 3.5))
    return sh


def gen_bag_tote(n, seed):
    """Tote or shoulder bag: long strap rising well above a narrower body."""
    r = Rand(seed, n); sh = Shape(n)
    H, Wt = bag_body(r, sh)
    hx = Wt / 2 * r.u(0.45, 0.95)
    sh.line(arc(-hx, hx, r.u(60, 150), 11), r.u(1.5, 3.5))
    return sh


def gen_bag_strap(n, seed):
    """Cross-body bag: body with a thin strap looping out to one side and over the top."""
    r = Rand(seed, n); sh = Shape(n); z = np.zeros(n)
    H, Wt = bag_body(r, sh)
    side = np.where(r.flag(0.5), 1.0, -1.0)
    xo = side * (Wt / 2 + r.u(2, 10)); top = -r.u(4, 30)
    sh.line(pts([side * Wt / 2 * 0.9, xo, xo, -side * Wt / 2 * r.u(0.3, 0.9)], [H * r.u(0.3, 0.7), H * 0.3, top, top]),
            r.u(1.2, 3))
    return sh


# ----------------------------------------------------------------------------------------------------------
# the bank of drawings: (class, generator, share of the class's drawings, score offset)
# ----------------------------------------------------------------------------------------------------------
PER_CLASS = 5000
BANK = [
    (0, gen_tshirt, 0.70, 0.0), (0, gen_tank, 0.30, 0.0),
    (1, gen_trouser, 0.92, 0.0), (1, gen_shorts, 0.08, 0.0),
    (2, gen_pullover, 0.88, -1.5), (2, gen_pullover_hood, 0.12, -1.5),
    (3, gen_dress_sleeveless, 0.40, 0.0), (3, gen_dress_short, 0.30, 0.0), (3, gen_dress_long, 0.30, 0.0),
    # a long-sleeved outline is read as a pullover unless the collar or the centre line says coat; both sit a
    # little below the sleeveless / short-sleeved / dress outlines they are otherwise confused with
    (4, gen_coat, 0.45, -3.0), (4, gen_coat_hood, 0.40, -3.0), (4, gen_coat_plain, 0.15, -3.0),
    (5, gen_sandal_flat, 0.25, 0.0), (5, gen_sandal_slide, 0.12, 0.0), (5, gen_sandal_wedge, 0.18, 0.0),
    (5, gen_sandal_chunky, 0.20, 0.0), (5, gen_sandal_heel, 0.25, 0.0),
    # shirts share their outlines with pullovers, coats and T-shirts and are the rarer reading of each
    (6, gen_shirt_long, 0.60, -6.0), (6, gen_shirt_short, 0.40, -14.0),
    (7, gen_sneaker, 0.48, 0.0), (7, gen_sneaker_darksole, 0.15, 0.0), (7, gen_sneaker_soleline, 0.25, 0.0),
    (7, gen_hightop, 0.12, -1.5),
    (8, gen_bag_plain, 0.30, -1.5), (8, gen_bag_handle, 0.40, -1.5), (8, gen_bag_tote, 0.15, -1.5),
    (8, gen_bag_strap, 0.15, -1.5),
    (9, gen_boot_flat, 0.55, 0.0), (9, gen_boot_heel, 0.45, 0.0),
]

# silhouette: s = clip((x - SIL_T) / SIL_D, 0, 1)
SIL_T, SIL_D = 0.04, 0.08
BLUR_SIGMA, BLUR_K = 1.2, 7
DRAW_BRIGHT = 0.6            # grey level given to the drawings before they go through the silhouette step


def blur_kernel():
    a = np.arange(BLUR_K) - BLUR_K // 2
    g = np.exp(-a * a / (2.0 * BLUR_SIGMA ** 2)); g /= g.sum()
    return np.outer(g, g)


def silhouette(img):
    return np.clip((img - SIL_T) / SIL_D, 0.0, 1.0)


def blur(sil):
    """sil: (B, 28, 28) -> blurred with zero padding, same as the network's conv."""
    K = blur_kernel(); p = BLUR_K // 2
    pad = np.pad(sil, ((0, 0), (p, p), (p, p)))
    out = np.zeros_like(sil)
    for dy in range(BLUR_K):
        for dx in range(BLUR_K):
            out += K[dy, dx] * pad[:, dy:dy + S, dx:dx + S]
    return out


TOP_ROWS, TOP_COLS, TOP_WEIGHT = 5, (6, 22), 3.0


def pixel_weights():
    """How much each pixel counts in the comparison: 1 everywhere, more in the top-centre zone where a neckline
    dip, a collar or a hood tells tops apart."""
    lam = np.ones((S, S))
    lam[:TOP_ROWS, TOP_COLS[0]:TOP_COLS[1]] = TOP_WEIGHT
    return lam


def draw_bank(per_class=PER_CLASS, seed=SEED, chunk=1000):
    """Returns coverage drawings (P, 28, 28), class of each, offset of each; classes in order, equal counts."""
    imgs, cls, offs = [], [], []
    for c in range(10):
        rows = [b for b in BANK if b[0] == c]
        counts = [int(round(per_class * b[2])) for b in rows]
        counts[0] += per_class - sum(counts)
        for k, (b, m) in enumerate(zip(rows, counts)):
            done = 0
            while done < m:
                q = min(chunk, m - done)
                imgs.append(b[1](q, seed + 1000 * c + 37 * k + done).render())
                done += q
            cls += [c] * m; offs += [b[3]] * m
    return np.concatenate(imgs), np.array(cls), np.array(offs, float)


# centre-line detector (zip or button band of a coat or shirt), on the contrast-normalised image
LINE_ROWS = 10                       # the line must run through this many rows
LINE_PROFILE = [1.0, -3.0, 2.0, 2.0, -3.0, 1.0]   # across the line: blind to flat, sloping and curved shading
LINE_T, LINE_W = 0.06, 0.20          # response where the cue starts, and the range over which it saturates
LINE_REF = 1.5                       # texture test: the response of the 10-row filter must beat this many times
                                     # the mean response of the same filter taken one row at a time (a line that
                                     # runs straight down adds up over the rows, texture does not)
LINE_BONUS = {4: 12.0}               # class -> score added when the line is clearly there
LINE_GATE = 30.0                     # the line only counts inside a garment: the silhouette must be on over the
                                     # whole filter window and two pixels to either side of it (else narrow items'
                                     # own edges, or the gap between trouser legs, look like a line).  The response
                                     # loses LINE_GATE times the share of the window that is not garment.
# open-shoe detector (sandals): dark pixels that have silhouette above and below them in the same column, within
# GAP_REACH rows -- the openings between straps and sole.  A long horizontal run of such pixels is not counted:
# that is the dark sole band of a sneaker with its bright bottom edge, not a strap opening.
GAP_REACH = 4
BAND_COLS = 9                        # a run is "long" when most of 9 neighbouring columns are gap pixels
BAND_T, BAND_W = 0.55, 0.25          # share of those columns where "long" starts, and the range to fully long
GAP_T, GAP_W = 4.0, 10.0             # counted gap area where the cue starts, and the range over which it saturates
GAP_BONUS = {5: 12.0}

def build(out_dir, per_class=PER_CLASS, cues=True):
    cov, cls, offs = draw_bank(per_class)
    P = len(cov)
    mu = blur(silhouette(DRAW_BRIGHT * cov)).reshape(P, S * S)
    lam = pixel_weights().reshape(1, S * S)
    W = {}
    k = 1.0 / SIL_D
    W['sil.W'] = np.array([k, k], float).reshape(2, 1, 1, 1)
    W['sil.b'] = np.array([-SIL_T * k, -SIL_T * k - 1.0])
    G = blur_kernel()
    W['blur.W'] = np.stack([G, -G])[None]                      # (1, 2, 7, 7): blur of clip = relu(a) - relu(a - 1)
    W['blur.b'] = np.zeros(1)
    W['match.W'] = 2.0 * lam * mu                               # minus squared distance, without the image's own term
    W['match.b'] = -(lam * mu * mu).sum(1) + offs
    layers = [
        {"name": "sil", "op": "conv", "inputs": ["image"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "sil_r", "op": "relu", "inputs": ["sil"]},
        {"name": "blur", "op": "conv", "inputs": ["sil_r"], "out": 1, "kernel": [BLUR_K, BLUR_K], "stride": 1,
         "padding": BLUR_K // 2},
        {"name": "match", "op": "dense", "inputs": ["blur"], "out": P},
        {"name": "by_class", "op": "reshape", "inputs": ["match"], "shape": [1, 10, per_class]},
        {"name": "best", "op": "maxpool", "inputs": ["by_class"], "size": [1, per_class], "stride": [1, per_class]},
        {"name": "shape10", "op": "reshape", "inputs": ["best"], "shape": [10]},
    ]
    if cues:
        # ---- centre line ----
        kk = 11
        Wl = np.zeros((2, 1, kk, kk))
        prof = np.array(LINE_PROFILE) / 4.0 / LINE_ROWS
        for dy in range(LINE_ROWS):
            Wl[0, 0, dy, 2:8] = prof                            # bright line; taps at columns c-3 .. c+2
            Wl[1, 0, dy, 2:8] = -prof                           # dark line
        W['line.W'] = Wl
        W['line.b'] = np.zeros(2)
        Wg = np.zeros((2, 2, kk, kk))
        Wg[:, 0, :LINE_ROWS, 0:10] = LINE_GATE / (10 * LINE_ROWS)    # columns c-5 .. c+4, all rows of the filter
        Wg[:, 1, :LINE_ROWS, 0:10] = -LINE_GATE / (10 * LINE_ROWS)
        W['line_gate.W'] = Wg
        W['line_gate.b'] = np.full(2, -LINE_GATE)               # 0 when the window is all garment, negative otherwise
        # window rows r-5 .. r+4 for r = 13 .. 21 (rows 8 .. 25 of the garment), line between columns 13|14 +- 1
        ph, pw = 9, 3
        Wp = np.zeros((2, 2, 28 - ph + 1, 28 - pw + 1))
        Wp[:, :, 13, 13] = 1.0 / LINE_W                         # strongest response of either sign at the centre
        W['line_c.W'] = Wp.reshape(2, -1)
        W['line_c.b'] = np.array([-LINE_T / LINE_W, -LINE_T / LINE_W - 1.0])   # relu(v) - relu(v - 1) = clip(v, 0, 1)
        W1 = np.zeros((2, 1, kk, kk))
        W1[0, 0, kk // 2, 2:8] = np.array(LINE_PROFILE) / 4.0   # the same profile on a single row
        W1[1, 0, kk // 2, 2:8] = -np.array(LINE_PROFILE) / 4.0
        W['line1.W'] = W1
        W['line1.b'] = np.zeros(2)
        Wr = np.zeros((2, 2, S, S))
        Wr[:, :, 8:26, 13:16] = -LINE_REF / LINE_W / (18 * 3)
        W['line_o.W'] = Wr.reshape(2, -1)                       # minus the mean single-row response at the centre
        W['line_o.b'] = np.zeros(2)
        Wo = np.zeros((10, 2))
        for c, bonus in LINE_BONUS.items():
            Wo[c] = [bonus, -bonus]
        W['line_s.W'] = Wo
        W['line_s.b'] = np.zeros(10)
        # ---- strap openings ----
        R = GAP_REACH; ks = 2 * R + 1
        Ws = np.zeros((12, 2, ks, ks))
        for d in range(1, R + 1):
            Ws[d - 1, :, R - d, R] = [1.0, -1.0]                # the silhouette d rows above
            Ws[4 + d - 1, :, R + d, R] = [1.0, -1.0]            # the silhouette d rows below
        Ws[8, :, R, R] = [1.0, -1.0]                            # the silhouette here (channels 9-11 stay 0)
        W['shift.W'] = Ws
        W['shift.b'] = np.zeros(12)
        W['gap.W'] = np.array([1.0, 1.0, -2.0]).reshape(1, 3, 1, 1)   # silhouette above AND below, dark here
        W['gap.b'] = np.array([-1.0])
        hb = BAND_COLS // 2
        Wb = np.zeros((2, 1, BAND_COLS, BAND_COLS))
        Wb[:, 0, hb, :] = 1.0 / BAND_COLS / BAND_W              # share of gap pixels along the row
        W['band.W'] = Wb
        W['band.b'] = np.array([-BAND_T / BAND_W, -BAND_T / BAND_W - 1.0])
        Wd = np.zeros((2, 2, BAND_COLS, BAND_COLS))
        Wd[:, 0, hb - 1:hb + 2, :] = 1.0                        # spread "long run" over the whole run (+-4 columns,
        Wd[:, 1, hb - 1:hb + 2, :] = -1.0                       # +-1 row)
        W['bandw.W'] = Wd
        W['bandw.b'] = np.array([0.0, -1.0])
        W['sgap_g.W'] = np.ones((1, 1, 1, 1)); W['sgap_g.b'] = np.zeros(1)
        W['sgap_m.W'] = np.array([-1.0, 1.0]).reshape(1, 2, 1, 1); W['sgap_m.b'] = np.zeros(1)
        W['gap_h.W'] = np.full((2, 1), S * S / GAP_W)           # mean over the image * 784 = counted gap area
        W['gap_h.b'] = np.array([-GAP_T / GAP_W, -GAP_T / GAP_W - 1.0])
        Wg2 = np.zeros((10, 2))
        for c, bonus in GAP_BONUS.items():
            Wg2[c] = [bonus, -bonus]
        W['gap_s.W'] = Wg2
        W['gap_s.b'] = np.zeros(10)
        layers += [
            {"name": "xn", "op": "norm", "inputs": ["image"]},
            {"name": "line", "op": "conv", "inputs": ["xn"], "out": 2, "kernel": [kk, kk], "stride": 1, "padding": kk // 2},
            {"name": "line_gate", "op": "conv", "inputs": ["sil_r"], "out": 2, "kernel": [kk, kk], "stride": 1,
             "padding": kk // 2},
            {"name": "line_g", "op": "add", "inputs": ["line", "line_gate"]},
            {"name": "line_r", "op": "relu", "inputs": ["line_g"]},
            {"name": "line_p", "op": "maxpool", "inputs": ["line_r"], "size": [ph, pw], "stride": [1, 1]},
            {"name": "line_c", "op": "dense", "inputs": ["line_p"], "out": 2},
            {"name": "line1", "op": "conv", "inputs": ["xn"], "out": 2, "kernel": [kk, kk], "stride": 1, "padding": kk // 2},
            {"name": "line1_r", "op": "relu", "inputs": ["line1"]},
            {"name": "line_o", "op": "dense", "inputs": ["line1_r"], "out": 2},
            {"name": "line_h", "op": "add", "inputs": ["line_c", "line_o"]},
            {"name": "line_hr", "op": "relu", "inputs": ["line_h"]},
            {"name": "line_s", "op": "dense", "inputs": ["line_hr"], "out": 10},
            {"name": "shift", "op": "conv", "inputs": ["sil_r"], "out": 12, "kernel": [ks, ks], "stride": 1, "padding": R},
            {"name": "shift_g", "op": "reshape", "inputs": ["shift"], "shape": [3, 4, S * S]},
            {"name": "shift_m", "op": "maxpool", "inputs": ["shift_g"], "size": [4, 1], "stride": [4, 1]},
            {"name": "uds", "op": "reshape", "inputs": ["shift_m"], "shape": [3, S, S]},
            {"name": "gap", "op": "conv", "inputs": ["uds"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "gap_r", "op": "relu", "inputs": ["gap"]},
            {"name": "band", "op": "conv", "inputs": ["gap_r"], "out": 2, "kernel": [BAND_COLS, BAND_COLS], "stride": 1,
             "padding": hb},
            {"name": "band_r", "op": "relu", "inputs": ["band"]},
            {"name": "bandw", "op": "conv", "inputs": ["band_r"], "out": 2, "kernel": [BAND_COLS, BAND_COLS], "stride": 1,
             "padding": hb},
            {"name": "bandw_r", "op": "relu", "inputs": ["bandw"]},
            {"name": "sgap_g", "op": "conv", "inputs": ["gap_r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "sgap_m", "op": "conv", "inputs": ["bandw_r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
            {"name": "sgap", "op": "add", "inputs": ["sgap_g", "sgap_m"]},
            {"name": "sgap_r", "op": "relu", "inputs": ["sgap"]},
            {"name": "gap_a", "op": "avgpool", "inputs": ["sgap_r"], "size": [S, S], "stride": [S, S]},
            {"name": "gap_h", "op": "dense", "inputs": ["gap_a"], "out": 2},
            {"name": "gap_hr", "op": "relu", "inputs": ["gap_h"]},
            {"name": "gap_s", "op": "dense", "inputs": ["gap_hr"], "out": 10},
            {"name": "scores", "op": "add", "inputs": ["shape10", "line_s", "gap_s"]},
        ]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k_: v.astype(np.float32) for k_, v in W.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
