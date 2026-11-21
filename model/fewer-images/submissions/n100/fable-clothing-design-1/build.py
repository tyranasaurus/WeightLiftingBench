"""build.py - writes network.json and weights.npz for the clothing classifier. Reads no files.

Every weight comes from this code; nothing is fitted and no training image or statistic of one is used.

  1. Soft silhouette: clip((image - T0) / (T1 - T0), 0, 1), then a Gaussian blur from its formula.
  2. Template bank: 10,000 drawings per class made by the parametric shape models below. The models are typed from
     general knowledge: garment measurements in cm (chest width, length, sleeve length and angle, collar ...), shoe
     profiles in units of the shoe length (sole, toe box, collar or shaft height, heel), bags as a body with a handle
     loop. Like the dataset, each drawing is scaled so that its longest side fills the frame, and centred.
     Each drawing is ONE template, passed through the same soft silhouette and blur and normalised by itself alone:
     weight row = its blurred silhouette, bias = -|row|^2 / 2 (+ a hand-set offset for coat and shirt), so that the
     dense layer ranks templates by squared distance to the image. A max-pool takes the best template per class.
  3. Two hand-built detectors on the contrast-normalised image (norm layer), each adding a bounded term:
     - a thin vertical line (zip / button placket) down the middle that stands out against the same measurement a few
       pixels to either side counts against pullover (coat and shirt templates carry a small fixed handicap instead);
     - a dark gap with bright material on both sides in the lower middle (the gap between two legs) counts for
       trouser, its absence against.
Usage: python3 build.py OUT_DIR
"""
import sys, json, os
import numpy as np

# ---- rasteriser ----

SS = 4                      # vertical sub-rows per pixel for polygon scan conversion
_GX, _GY = np.meshgrid(np.arange(28) + 0.5, np.arange(28) + 0.5)


def raster_poly(P):
    """P: (N, V, 2) polygon vertices in pixel coordinates (x right, y down; pixel i covers [i, i+1]).
    Returns (N, 28, 28) area coverage in [0, 1] (non-zero winding)."""
    P = np.asarray(P, np.float64)
    N, V, _ = P.shape
    R = 28 * SS
    ys = ((np.arange(R) + 0.5) / SS)[None, :]
    pr = (np.arange(28) + 1.0)[None, None, :]
    acc = np.zeros((N, R, 28))
    for k in range(V):
        x0, y0 = P[:, k, 0][:, None], P[:, k, 1][:, None]
        x1, y1 = P[:, (k + 1) % V, 0][:, None], P[:, (k + 1) % V, 1][:, None]
        dy = y1 - y0
        hit = (ys >= np.minimum(y0, y1)) & (ys < np.maximum(y0, y1))
        t = (ys - y0) / np.where(dy == 0, 1.0, dy)
        xc = x0 + t * (x1 - x0)
        d = np.where(dy > 0, 1.0, -1.0) * hit
        acc += d[:, :, None] * np.clip(pr - xc[:, :, None], 0.0, 1.0)
    return np.clip(np.abs(acc), 0.0, 1.0).reshape(N, 28, SS, 28).mean(2)


def raster_stroke(A, B, r):
    """Thick line segment A->B ((N, 2) each, pixel coordinates) of half-width r (N,). Returns (N, 28, 28) coverage."""
    ax, ay = A[:, 0][:, None, None], A[:, 1][:, None, None]
    bx, by = B[:, 0][:, None, None], B[:, 1][:, None, None]
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((_GX[None] - ax) * dx + (_GY[None] - ay) * dy) / L2, 0.0, 1.0)
    dist = np.sqrt((_GX[None] - ax - t * dx) ** 2 + (_GY[None] - ay - t * dy) ** 2)
    r = r[:, None, None]
    return np.clip(r + 0.5 - dist, 0.0, np.minimum(1.0, 2.0 * r))


def fit_frame(polys, strokes, size):
    """polys: list of (N, V, 2) arrays in natural units; strokes: list of ((N, K, 2) polyline, (N,) half-width).
    Mimics the dataset preparation: the bounding box of the whole item is scaled so that its longest side is
    `size` pixels ((N,), about 28) and centred in the 28 x 28 frame. Returns (polys, strokes) in pixel coordinates."""
    lo = np.full((len(size), 2), np.inf); hi = -lo
    for P in polys:
        lo = np.minimum(lo, P.min(1)); hi = np.maximum(hi, P.max(1))
    for Q, r in strokes:
        lo = np.minimum(lo, (Q - r[:, None, None]).min(1)); hi = np.maximum(hi, (Q + r[:, None, None]).max(1))
    s = size / (hi - lo).max(1)
    c = (lo + hi) / 2
    tr = lambda X: (X - c[:, None, :]) * s[:, None, None] + 14.0
    return [tr(P) for P in polys], [(tr(Q), r * s) for Q, r in strokes]


def render(polys, strokes, size, min_r=0.3):
    """Union of filled polygons and stroked polylines, fitted to the frame. Returns (N, 28, 28) coverage in [0, 1]."""
    pp, ss = fit_frame(polys, strokes, size)
    cov = np.zeros((len(size), 28, 28))
    for P in pp:
        cov = np.maximum(cov, raster_poly(P))
    for Q, r in ss:
        r = np.maximum(r, min_r)
        for k in range(Q.shape[1] - 1):
            cov = np.maximum(cov, raster_stroke(Q[:, k], Q[:, k + 1], r))
    return cov


def V(*pts):
    """Stack vertices given as (x, y) pairs of (N,) arrays (or scalars) into (N, V, 2)."""
    n = max(np.size(c) for p in pts for c in p)
    return np.stack([np.stack([np.broadcast_to(np.asarray(p[0], float), (n,)), np.broadcast_to(np.asarray(p[1], float), (n,))], -1) for p in pts], 1)


def mirror(P):
    """Mirror a polygon/polyline (N, V, 2) in x."""
    return P * np.array([-1.0, 1.0])

# ---- shape models ----

RAD = np.pi / 180.0


class Rs:
    """Uniform / bell-shaped (sum of two uniforms) samplers with a fixed seed."""
    def __init__(self, seed, n):
        self.rng = np.random.default_rng(seed); self.n = n
    def u(self, lo, hi):
        return lo + (hi - lo) * self.rng.random(self.n)
    def t(self, lo, hi):
        return lo + (hi - lo) * (self.rng.random(self.n) + self.rng.random(self.n)) / 2.0
    def b(self, p):
        return self.rng.random(self.n) < p
    def size(self):
        return self.u(26.6, 28.0)


def body_poly(nw, nd, ch, sw, sd, cw, ah, ww, wy, hw, L, hemc):
    """Front outline of an upper-body garment without sleeves. x from the centre line, y down from the shoulder line.
    nw neck half-width, nd depth of the neckline dip, ch collar height above the shoulder line, (sw, sd) shoulder tip,
    (cw, ah) underarm point, (ww, wy) waist point, (hw, L) hem corner, hemc: how much lower the hem is at the centre."""
    right = [(0 * nw, nd - ch), (0.5 * nw, 0.75 * nd - ch), (0.85 * nw, 0.28 * nd - ch), (nw, -ch),
             (nw + 0.5 + 0.4 * ch, 0 * nw), (sw, sd), (cw, ah), (ww, wy), (hw, L - hemc), (0.6 * hw, L - 0.3 * hemc), (0 * nw, L)]
    R = V(*right)
    return np.concatenate([R, mirror(R[:, ::-1])[:, 1:-1]], 1)


def sleeve_polys(sw, sd, cw, ah, phi, sl, cuff):
    """Two sleeves. The outer edge leaves the shoulder tip at angle phi from the vertical (0 = hanging straight down,
    90 degrees = straight out sideways) and has length sl; the cuff (width `cuff`) is square to it."""
    ux, uy = np.sin(phi), np.cos(phi)
    Cx, Cy = sw + sl * ux, sd + sl * uy
    Dx, Dy = Cx - cuff * uy, Cy + cuff * ux
    ex = np.minimum(sw, cw) - 5.0
    R = V((sw, sd), (Cx, Cy), (Dx, Dy), (ex, ah), (ex, sd + 3.0))
    return [R, mirror(R)]


def gen_tshirt(seed, n):
    r = Rs(seed, n)
    cw = r.t(18.5, 25.5); L = r.t(58, 76)
    nw = r.t(7, 11); nd = r.t(1, 7); ch = 0 * nw
    sw = cw + r.u(-1, 3); sd = r.t(2.5, 6.5); ah = sd + r.t(17, 23)
    hw = cw + r.t(-2.5, 2.5); ww = (cw + hw) / 2 - r.u(0, 1.5); wy = ah + 0.5 * (L - ah)
    phi = r.t(25, 62) * RAD; sl = np.where(r.b(0.8), r.t(8, 20), r.t(2, 9)); cuff = r.t(12, 18)
    body = body_poly(nw, nd, ch, sw, sd, cw, ah, ww, wy, hw, L, r.u(0, 2))
    return render([body] + sleeve_polys(sw, sd, cw, ah, phi, sl, cuff), [], r.size())


def gen_tank(seed, n):
    """Sleeveless top: straps, scooped armholes."""
    r = Rs(seed, n)
    cw = r.t(12, 22); L = r.t(54, 76)
    si = r.t(5.5, 10)                  # inner edge of the strap at the top
    so = si + r.t(2, 8)                # outer edge
    nd = r.t(6, 20); ah = r.t(17, 25)
    hw = cw + r.t(-2, 4); ww = (cw + hw) / 2 - r.u(0, 2); wy = ah + 0.5 * (L - ah)
    right = [(0 * cw, nd), (0.6 * si, 0.9 * nd), (0.95 * si, 0.45 * nd), (si, 0 * cw), (so, 0 * cw),
             (so + 0.25 * (cw - so), 0.6 * ah), (cw, ah), (ww, wy), (hw, L), (0 * cw, L)]
    R = V(*right)
    return render([np.concatenate([R, mirror(R[:, ::-1])[:, 1:-1]], 1)], [], r.size())


def gen_pullover(seed, n):
    """Long sleeves hanging beside the body and reaching past the hem; round neckline."""
    r = Rs(seed, n)
    cw = r.t(17.5, 25); L = r.t(53, 69)
    nw = r.t(7.5, 12); nd = r.t(1, 7); ch = 0 * nw
    sw = cw + r.u(0.5, 5.5); sd = r.t(3, 7.5); ah = sd + r.t(19, 25)
    hw = cw - r.t(-1.5, 3.5); ww = (cw + hw) / 2; wy = ah + 0.5 * (L - ah)
    phi = r.t(-1, 14) * RAD; sl = r.t(57, 69); cuff = r.t(8.5, 13)
    body = body_poly(nw, nd, ch, sw, sd, cw, ah, ww, wy, hw, L, 0 * nw)
    return render([body] + sleeve_polys(sw, sd, cw, ah, phi, sl, cuff), [], r.size())


def gen_coat(seed, n):
    """Jacket or coat: stand-up collar or hood, straight sleeves close to a boxy body, hem at or below the cuffs."""
    r = Rs(seed, n)
    cw = r.t(19.5, 27.5); L = np.where(r.b(0.8), r.t(62, 84), r.t(78, 104))
    hood = r.b(0.25)
    nw = np.where(hood, r.t(9, 13), r.t(6, 11)); ch = np.where(hood, r.t(5, 12), r.t(1.5, 8)); nd = r.t(0, 4)
    sw = cw + r.u(0.5, 5); sd = r.t(2, 6.5); ah = sd + r.t(20, 26)
    hw = cw + r.t(-2.5, 4); ww = (cw + hw) / 2 - r.u(0, 2); wy = ah + 0.45 * (L - ah)
    phi = r.t(-1, 11) * RAD; sl = r.t(56, 68); cuff = r.t(10, 14.5)
    body = body_poly(nw, nd, ch, sw, sd, cw, ah, ww, wy, hw, L, 0 * nw)
    return render([body] + sleeve_polys(sw, sd, cw, ah, phi, sl, cuff), [], r.size())


def gen_shirt(seed, n):
    """Long-sleeved button shirt / blouse: small collar, slimmer sleeves hanging a little away from the body, tails."""
    r = Rs(seed, n)
    cw = r.t(17.5, 25); L = r.t(62, 80)
    nw = r.t(6.5, 10); ch = r.t(0.5, 4.5); nd = r.t(1, 8)
    sw = cw + r.u(0, 4.5); sd = r.t(3, 7); ah = sd + r.t(18, 24)
    hw = cw + r.t(-2, 4); ww = (cw + hw) / 2 - r.u(0, 2.5); wy = ah + 0.5 * (L - ah)
    phi = r.t(0, 22) * RAD; sl = r.t(54, 67); cuff = r.t(7, 11)
    body = body_poly(nw, nd, ch, sw, sd, cw, ah, ww, wy, hw, L, r.t(0, 7))
    return render([body] + sleeve_polys(sw, sd, cw, ah, phi, sl, cuff), [], r.size())


def gen_dress(seed, n):
    r = Rs(seed, n)
    L = r.t(82, 128)
    bw = r.t(15.5, 22.5)                           # bust half-width
    ah = r.t(18, 25)
    wy = r.t(36, 46); ww = bw - r.t(0, 7)          # waist
    hy = wy + r.t(16, 24)
    flare = r.b(0.5)
    hpw = np.where(flare, ww + r.t(3, 11), ww + r.t(2, 8))                 # hip half-width
    hw = np.where(flare, hpw + r.t(3, 20), hpw + r.t(-4, 4))               # hem half-width
    kind = r.u(0, 1)                                                       # <.4 straps / .4-.6 wide shoulders / >.6 sleeves
    strap = kind < 0.4
    nw = r.t(6, 11); nd = np.where(strap, r.t(8, 20), r.t(2, 12))
    sw = np.where(strap, nw + r.t(1.5, 5), bw + r.u(-3, 2)); sd = np.where(strap, 0 * L, r.t(2, 6))
    # armhole edge: straps go nearly straight down then out to the bust; shoulders slope straight to the underarm
    mx = np.where(strap, sw + 0.25 * (bw - sw), 0.5 * (sw + bw)); my = np.where(strap, 0.6 * ah, 0.5 * (sd + ah))
    right = [(0 * L, nd), (0.6 * nw, 0.9 * nd), (0.95 * nw, 0.4 * nd), (nw, 0 * L), (sw, sd), (mx, my), (bw, ah),
             (ww, wy), (hpw, hy), (hw, L), (0 * L, L)]
    R = V(*right)
    body = np.concatenate([R, mirror(R[:, ::-1])[:, 1:-1]], 1)
    longs = r.b(0.5)
    phi = np.where(longs, r.t(1, 17), r.t(25, 65)) * RAD
    sl = np.where(longs, r.t(40, 64), r.t(6, 22)); cuff = np.where(longs, r.t(8, 12), r.t(11, 17))
    sl = np.where(kind > 0.6, sl, 0.01)            # no sleeves: degenerate (zero-area) sleeve
    cuff = np.where(kind > 0.6, cuff, 0.01)
    return render([body] + sleeve_polys(sw, sd, bw, ah, phi, sl, cuff), [], r.size())


def gen_trouser(seed, n):
    r = Rs(seed, n)
    short = r.b(0.0)                                # shorts left out: a boxy outline with no long leg gap is not evidence for trousers
    L = np.where(short, r.t(38, 62), r.t(92, 110))
    wh = r.t(17, 24)                                # waist half-width
    hp = wh + r.t(0, 4.5); hy = r.t(14, 22)         # hip
    rise = r.t(24, 33)
    lw = np.where(short, r.t(22, 30), r.t(13, 23))  # leg width at the hem
    xo = np.where(short, hp + r.t(-1, 5), hp + r.t(-6, 5))    # outer edge of the leg at the hem
    xi = np.maximum(xo - lw, r.u(0.4, 1.6))         # inner edge of the leg at the hem (half of the gap between the legs)
    xo = np.maximum(xo, xi + 9.0)
    knee = 0.5 * (rise + L)
    right = [(0 * L, 0 * L), (wh, 0 * L), (hp, hy), (0.5 * (hp + xo) + r.u(-1, 1), knee), (xo, L), (xi, L),
             (0.45 * xi + r.u(0, 0.8), knee), (0 * L, rise)]
    R = V(*right)
    body = np.concatenate([R, mirror(R[:, ::-1])[:, 1:]], 1)
    return render([body], [], r.size())


def up(P):
    """Shoe and bag models are written with y up from the ground; the image has y down."""
    return P * np.array([1.0, -1.0])


def gen_sneaker(seed, n):
    """Side view, toe to the left: flat sole, low toe box, upper rising to the collar at the heel."""
    r = Rs(seed, n)
    high = r.b(0.04)
    h = np.where(high, r.t(0.44, 0.62), r.t(0.29, 0.46))          # height of the collar at the heel
    xt = np.where(high, r.t(0.50, 0.68), r.t(0.56, 0.76))         # where the tongue / top of the lacing ends
    ht = h + np.where(high, r.t(-0.06, 0.10), r.t(-0.04, 0.10))
    dip = r.t(0.0, 0.07)
    tb = r.t(0.12, 0.23)                                          # toe box height
    ty = r.t(0.03, 0.10)                                          # toe spring
    xb = r.u(0.965, 1.0)
    z = 0 * h
    P = V((z, ty + 0.04), (z + 0.03, ty * 0.5), (z + 0.12, z), (z + 0.9, z), (z + 1.0, z + 0.03), (z + 1.0, 0.5 * h), (xb, h),
          (0.5 * (xb + xt) + 0.06, h - dip), (xt + 0.05, ht), (xt, ht - 0.01), (0.55 * xt + 0.05, 0.45 * ht + 0.55 * tb),
          (z + 0.2, tb + 0.03), (z + 0.08, tb), (z + 0.015, 0.6 * tb + 0.4 * ty))
    return render([up(P)], [], r.size())


def gen_boot(seed, n):
    """Side view, toe to the left: foot part like a closed shoe, shaft over the heel half rising above the ankle."""
    r = Rs(seed, n)
    H = np.where(r.b(0.45), r.t(0.44, 0.74), r.t(0.64, 1.16))     # shaft height (in shoe lengths)
    xf = r.t(0.36, 0.62); xb = r.u(0.92, 1.0)
    Hf = H + r.t(-0.10, 0.05)
    heeled = r.b(0.5)
    hh = np.where(heeled, r.t(0.07, 0.36), r.t(0.0, 0.07))        # heel height
    hwid = np.where(heeled, r.t(0.10, 0.28), r.t(0.25, 0.35))     # heel block length
    tb = r.t(0.14, 0.27); ty = r.t(0.02, 0.09)
    ins = np.minimum(r.t(0.38, 0.58), 0.72 * H)                   # height where the vamp meets the shaft
    z = 0 * H
    P = V((z, ty + 0.04), (z + 0.03, ty * 0.5), (z + 0.12, z), (z + 0.45, z), (1.0 - hwid - 0.08, 0.75 * hh), (1.0 - hwid, 0.9 * hh),
          (1.0 - hwid + 0.02, z), (z + 0.985, z), (z + 1.0, hh + 0.05), (0.5 * (xb + 1.0), 0.5 * H), (xb, H), (xf, Hf),
          (xf + 0.01, 0.5 * (Hf + ins) + 0.03), (xf - 0.06, ins + hh * 0.5), (z + 0.2, tb + 0.05 + 0.3 * hh), (z + 0.07, tb), (z + 0.012, 0.6 * tb + 0.4 * ty))
    return render([up(P)], [], r.size())


def gen_sandal_flat(seed, n):
    """Flat or low sandal: thin sole, toe strap, strap(s) to the ankle, ankle / heel strap. Mostly empty space."""
    r = Rs(seed, n)
    ts = r.t(0.03, 0.10)                                          # sole thickness
    z = 0 * ts
    lift = r.t(0.0, 0.10)                                         # low heel / wedge
    sole = V((z, z + 0.02), (z + 0.04, z), (z + 1.0, z), (z + 1.0, ts + lift), (z + 0.6, ts + 0.5 * lift), (z + 0.05, ts + 0.01))
    w = r.t(0.012, 0.06)                                          # strap half-width
    ah = ts + lift + r.t(0.16, 0.42)                              # ankle strap height
    ax = r.t(0.62, 0.82)                                          # front of the ankle loop
    t1 = r.t(0.10, 0.25); t2 = t1 + r.t(0.12, 0.3)                # toe strap feet
    th = ts + r.t(0.08, 0.2)
    toe = V((t1, ts), (0.5 * (t1 + t2), th), (t2, ts))
    back = V((z + 0.97, ts + lift), (z + 0.95 + r.u(-0.04, 0.03), ah), (ax, ah - r.u(0, 0.08)), (ax - r.u(0.0, 0.15), ts + 0.3 * lift))
    vamp = V((0.5 * (t1 + t2), th), (0.5 * (t1 + t2) + 0.15, th + 0.3 * (ah - th)), (ax, ah - 0.03))
    has_vamp = r.b(0.6)
    vamp = np.where(has_vamp[:, None, None], vamp, toe)
    return render([up(sole)], [(up(toe), w), (up(back), w), (up(vamp), w)], r.size())


def gen_sandal_heel(seed, n):
    """High-heeled sandal: sole climbing from the ball of the foot to a raised heel seat, heel (pin, block or wedge),
    toe strap and ankle strap. A tall, mostly empty figure whose mass lies along a rising diagonal."""
    r = Rs(seed, n)
    z = r.u(0, 0)
    hh = r.t(0.28, 0.80)                                          # heel height
    xs = r.t(0.80, 0.96)                                          # heel seat x
    pf = r.t(0.0, 0.12)                                           # platform under the ball of the foot
    ts = r.t(0.025, 0.07)
    bx = r.t(0.28, 0.42)                                          # where the sole starts to climb
    wedge = r.b(0.35)
    hw = np.where(r.b(0.5), r.t(0.02, 0.06), r.t(0.06, 0.2))      # heel thickness at the ground
    hx = xs + r.u(-0.04, 0.04)                                    # heel foot position
    # sole / footbed band (polygon), from the toe along the ground then up to the heel seat
    sole = V((z, pf + 0.02), (z + 0.03, z), (bx, z), (np.where(wedge, hx - hw, bx + 0.03), np.where(wedge, z, pf + 0.01)),
             (np.where(wedge, hx + hw, xs - 0.12), np.where(wedge, z, 0.8 * hh - ts)), (xs + 0.04, hh - ts), (xs + 0.04, hh + ts),
             (xs - 0.1, 0.85 * hh + ts), (bx, pf + 2 * ts), (z + 0.03, pf + ts + 0.02))
    heel = V((xs - 0.07, hh - ts), (xs + 0.04, hh - ts), (hx + hw, z), (hx - hw, z))
    w = r.t(0.012, 0.07)
    ah = hh + r.t(0.15, 0.42)                                     # ankle strap height
    ax = xs - r.t(0.12, 0.38)
    back = V((xs + 0.02, hh), (xs + r.u(-0.03, 0.04), ah), (ax, ah - r.u(-0.03, 0.08)))
    t1 = r.t(0.06, 0.2); t2 = t1 + r.t(0.14, 0.3); th = pf + ts + r.t(0.08, 0.2)
    toe = V((t1, pf + ts), (0.5 * (t1 + t2), th), (t2, pf + 2 * ts + 0.3 * (t2 - bx).clip(0) * hh))
    diag = V((0.5 * (t1 + t2), th), (0.5 * (0.5 * (t1 + t2) + ax), 0.5 * (th + ah) - r.u(0, 0.08)), (ax, ah))
    kind = r.u(0, 1)
    diag = np.where((kind < 0.55)[:, None, None], diag, back)
    cup = np.where(r.b(0.45), r.t(0.08, 0.3), 0.001)
    cupp = V((xs - 0.16, 0.82 * hh), (xs + 0.04, hh - ts), (xs + 0.035, hh + cup), (xs - 0.07, hh + 0.75 * cup))
    return render([up(sole), up(heel), up(cupp)], [(up(toe), w), (up(back), w), (up(diag), w)], r.size())


def gen_sandal_chunky(seed, n):
    """Sporty / platform sandal: thick sole and broad bands over toes and ankle with gaps between them."""
    r = Rs(seed, n)
    z = r.u(0, 0)
    ts = r.t(0.05, 0.24); lift = r.t(0.0, 0.2)
    sole = V((z, z + 0.04), (z + 0.05, z), (z + 1.0, z), (z + 1.0, ts + lift), (z + 0.55, ts + 0.4 * lift), (z + 0.03, ts))
    b1 = r.t(0.05, 0.22); b2 = b1 + r.t(0.12, 0.34); bh = ts + r.t(0.08, 0.24)
    band = V((b1, ts - 0.01), (b1 + 0.03, bh), (b2 - 0.02, bh + 0.05), (b2 + 0.03, ts - 0.01))
    a1 = r.t(0.52, 0.74); ah = ts + lift + r.t(0.16, 0.42)
    ank = V((a1, ts), (a1 + 0.04, ah), (z + 0.97, ah + r.u(-0.04, 0.04)), (z + 1.0, ts + lift), (z + 0.9, ts + lift), (z + 0.88, ah - 0.1),
            (a1 + 0.14, ah - 0.1), (a1 + 0.12, ts))
    w = r.t(0.015, 0.045)
    link = V((b2 - 0.02, bh + 0.03), (a1 + 0.04, ah - 0.03))
    return render([up(sole), up(band), up(ank)], [(up(link), w)], r.size())


def gen_sandal_slide(seed, n):
    """Slide, mule or flip-flop: a sole (flat, platform or wedge) and one broad band over the forefoot; heel end open."""
    r = Rs(seed, n)
    z = r.u(0, 0)
    ts = r.t(0.04, 0.18); lift = np.where(r.b(0.5), r.t(0.0, 0.1), r.t(0.1, 0.45))
    sole = V((z, z + 0.03), (z + 0.05, z), (z + 1.0, z), (z + 1.0, ts + lift), (z + 0.5, ts + 0.35 * lift), (z + 0.03, ts))
    b1 = r.t(0.06, 0.22); b2 = b1 + r.t(0.2, 0.42); bh = r.t(0.1, 0.26)
    y2 = ts + 0.35 * lift * ((b2 - 0.05) / 0.45).clip(0, 1)
    band = V((b1, ts - 0.01), (b1 + 0.05, ts + 0.6 * bh), (0.5 * (b1 + b2), y2 + bh), (b2, y2 + 0.9 * bh), (b2 + 0.03, y2 - 0.01))
    return render([up(sole), up(band)], [], r.size())


def gen_bag(seed, n):
    """Front view: a box / trapezoid body, usually with a carrying handle or strap drawn as a thin loop above it."""
    r = Rs(seed, n)
    z = r.u(0, 0)
    bh = r.t(0.42, 1.0)                                           # body height relative to its width
    tw = r.t(0.34, 0.54)                                          # half-width of the top edge (bottom is 0.5)
    bwid = np.where(r.b(0.3), r.t(0.36, 0.5), z + 0.5)            # half-width of the bottom edge
    cr = r.t(0.02, 0.12)                                          # corner cut
    body = V((-bwid + cr, z), (bwid - cr, z), (bwid, cr), (np.maximum(tw, bwid) * 0.5 + 0.5 * tw, 0.6 * bh), (tw, bh - 0.5 * cr), (tw - 0.5 * cr, bh),
             (-tw + 0.5 * cr, bh), (-tw, bh - 0.5 * cr), (-np.maximum(tw, bwid) * 0.5 - 0.5 * tw, 0.6 * bh), (-bwid, cr))
    kind = r.u(0, 1)                                              # <.28 none, <.75 hand loop, else shoulder strap
    a = np.where(kind < 0.75, r.t(0.14, 0.40), np.minimum(tw, r.t(0.3, 0.52)))    # where the handle meets the top edge
    hh = np.where(kind < 0.75, r.t(0.14, 0.60), r.t(0.35, 1.0))                   # how far it rises
    peak = r.u(0.55, 1.0)                                         # 1 = rounded arch, lower = more pointed
    k = 9
    th = np.linspace(0, np.pi, k)[None, :]
    hx = -a[:, None] * np.cos(th) * (peak[:, None] + (1 - peak[:, None]) * np.abs(np.cos(th)))
    hy = bh[:, None] - 0.02 + hh[:, None] * np.sin(th) ** peak[:, None]
    handle = np.stack([hx, hy], -1)
    none = (kind < 0.28)[:, None, None]
    handle = np.where(none, np.stack([0 * hx, 0 * hy + 0.5 * bh[:, None]], -1), handle)
    w = r.t(0.010, 0.035)
    return render([up(body)], [(up(handle), w)], r.size())


# class -> list of (shape model, share of the class's templates, score offset of these templates)
# Coats and long-sleeved shirts open at the front. With no front line seen a pullover is the better guess, so the
# templates of both classes start the same small distance behind.
B_COAT, B_SHIRT = 2.0, 2.0
GENS = {0: [(gen_tshirt, 0.7, 0.0), (gen_tank, 0.3, 0.0)], 1: [(gen_trouser, 1.0, 0.0)], 2: [(gen_pullover, 1.0, 0.0)],
        3: [(gen_dress, 1.0, 0.0)], 4: [(gen_coat, 1.0, -B_COAT)],
        5: [(gen_sandal_flat, 0.3, 0.0), (gen_sandal_heel, 0.35, 0.0), (gen_sandal_chunky, 0.2, 0.0), (gen_sandal_slide, 0.15, 0.0)],
        6: [(gen_shirt, 1.0, -B_SHIRT)], 7: [(gen_sneaker, 1.0, 0.0)], 8: [(gen_bag, 1.0, 0.0)], 9: [(gen_boot, 1.0, 0.0)]}


# ----------------------------------------------------------------------------------------------------------------
# network
# ----------------------------------------------------------------------------------------------------------------
T0, T1 = 0.012, 0.04         # soft silhouette: 0 below T0, 1 above T1 (the background of the photos is black)
TEMPLATE_BRIGHTNESS = 0.25   # a drawing is passed through the same soft threshold as if it had this grey level
SIGMA = 1.0                  # blur (pixels) applied to silhouettes before comparing them
K = 10000                    # templates per class

# front-line detector (zip / button placket down the middle of a garment), all values chosen by hand
CMAX = 0.15                  # per-pixel line response is clipped here (image in units of its own standard deviation)
LINE_ROWS = 15               # the line must run over about this many rows
ROWS = range(11, 18)         # centres of the 15-row windows that are examined
CTR_COLS = (12, 13, 14, 15)  # the middle of the frame
REF_COLS = (9, 10, 17, 18)   # the same measurement a few pixels to either side: fabric texture, prints, folds
TH0 = 0.25 * CMAX            # the centre must beat the sides by this much before the cue starts ...
TH1 = 0.55 * CMAX            # ... and the cue is full here
DELTA = 6.0                  # a front line counts this much against pullover (the closed-front long-sleeved class)

# leg-gap detector: trousers are two legs with a dark gap between them in the lower middle of the frame.
# A gap pixel is much darker than BOTH the pixel d to its left and the pixel d to its right, for d = 2, 3 or 5
# (narrow gap between legs that almost touch ... legs standing apart); the largest of the three responses is used.
GAP_D = (2, 3, 5)
GAP_CLIP = 1.0               # clip of the gap response (image in units of its own standard deviation)
GAP_ROWS = 9                 # rows averaged
GAP_CENTRES = range(16, 23)  # centres of the 9-row windows: the lower half of the frame
GAP_COLS = range(10, 16)     # after the 3-wide horizontal max these cover image columns 10..17
GAP_T0, GAP_T1 = 0.5, 0.8    # no gap below T0, clear gap above T1: far darker than a seam, a fold or a zip
DELTA_GAP = 3.0              # trouser score: +DELTA_GAP with a clear gap between the legs, -DELTA_GAP with none


def gauss_kernel(sigma, radius):
    ax = np.arange(-radius, radius + 1.0)
    g = np.exp(-ax ** 2 / (2 * sigma ** 2)); g /= g.sum()
    return np.outer(g, g)


def soft_sil(x):
    return np.clip((x - T0) / (T1 - T0), 0.0, 1.0)


def blur(imgs, ker):
    r = ker.shape[0] // 2
    P = np.pad(imgs, ((0, 0), (r, r), (r, r)))
    out = np.zeros_like(imgs)
    for dy in range(ker.shape[0]):
        for dx in range(ker.shape[1]):
            out += ker[dy, dx] * P[:, dy:dy + 28, dx:dx + 28]
    return out


def class_templates(c, k):
    """k drawings for class c and the score offset of each."""
    parts = GENS[c]; out = []; off = []; left = k
    for i, (fn, frac, offset) in enumerate(parts):
        m = left if i == len(parts) - 1 else int(round(k * frac))
        left -= m
        out.append(fn(1000 * c + 17 * i + 5, m)); off.append(np.full(m, offset))
    return np.concatenate(out), np.concatenate(off)


def build(out_dir):
    ker = gauss_kernel(SIGMA, 2)
    W = np.zeros((10 * K, 784), np.float32); b = np.zeros(10 * K, np.float32)
    for c in range(10):
        cov, off = class_templates(c, K)                     # (K, 28, 28) drawings, coverage in [0, 1]
        feat = blur(soft_sil(TEMPLATE_BRIGHTNESS * cov), ker).reshape(K, 784)
        W[c * K:(c + 1) * K] = feat                          # each drawing is its own template ...
        b[c * K:(c + 1) * K] = -0.5 * (feat ** 2).sum(1) + off   # ... normalised by itself alone
    g = 1.0 / (T1 - T0)
    f32 = np.float32
    weights = {
        "sil1.W": np.full((1, 1, 1, 1), g, f32), "sil1.b": np.array([-T0 * g], f32),
        "sil2.W": np.full((1, 1, 1, 1), -1.0, f32), "sil2.b": np.array([1.0], f32),
        "sil3.W": np.full((1, 1, 1, 1), -1.0, f32), "sil3.b": np.array([1.0], f32),
        "blur.W": ker.reshape(1, 1, 5, 5).astype(f32), "blur.b": np.zeros(1, f32),
        "match.W": W, "match.b": b,
    }
    layers = [
        {"name": "sil1", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "sil1r", "op": "relu", "inputs": ["sil1"]},
        {"name": "sil2", "op": "conv", "inputs": ["sil1r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "sil2r", "op": "relu", "inputs": ["sil2"]},
        {"name": "sil3", "op": "conv", "inputs": ["sil2r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "blur", "op": "conv", "inputs": ["sil3"], "out": 1, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "match", "op": "dense", "inputs": ["blur"], "out": 10 * K},
        {"name": "bank", "op": "reshape", "inputs": ["match"], "shape": [1, 10, K]},
        {"name": "best", "op": "maxpool", "inputs": ["bank"], "size": [1, K], "stride": [1, K]},
        {"name": "shape_scores", "op": "reshape", "inputs": ["best"], "shape": [10]},
    ]

    # ---- front-line branch -------------------------------------------------------------------------------------
    # image in units of its own spread -> thin vertical line detector (bright or dark) -> clip -> average down 15 rows
    # -> strongest value in the middle columns minus strongest value in side columns -> bounded penalty for pullover
    ld = np.zeros((2, 1, 5, 5), f32); ld[0, 0, 2] = [-0.5, 0.0, 1.0, 0.0, -0.5]; ld[1] = -ld[0]
    eye = np.zeros((2, 2, 1, 1), f32); eye[0, 0] = eye[1, 1] = 1.0
    lv = np.zeros((2, 2, LINE_ROWS, LINE_ROWS), f32); lv[0, 0, :, LINE_ROWS // 2] = lv[1, 1, :, LINE_ROWS // 2] = 1.0 / LINE_ROWS
    roi = np.zeros((4 * 28, 2, 28, 28), f32); j = [0, 0, 0, 0]
    for grp, cols in ((0, CTR_COLS), (2, REF_COLS)):
        for ch in range(2):
            for rr in ROWS:
                for cc in cols:
                    roi[(grp + ch) * 28 + j[grp + ch], ch, rr, cc] = 1.0; j[grp + ch] += 1
    assert j == [28, 28, 28, 28]
    lq = np.array([[1, 0, -1, 0], [0, 1, 0, -1]], f32)
    width = TH1 - TH0
    # lwr = width - min(excess, width)  ->  line strength L = 1 - lwr / width;  pullover: -DELTA * L
    cue = np.zeros((10, 1), f32); cue[2, 0] = DELTA / width
    cue_b = np.zeros(10, f32); cue_b[2] = -DELTA
    # leg gap. gp: for each spacing d: p = x[c-d] - x[c] (kept as p+ and p-) and q = x[c-d] - x[c+d];
    # g_d = min(left, right) - centre = p - relu(q). Then max over the spacings (max(a, b) = a + relu(b - a)), clip,
    # max over 3 columns, average down 9 rows, strongest value in the lower middle of the frame.
    R = max(GAP_D); nd = len(GAP_D)
    gp = np.zeros((3 * nd, 1, 2 * R + 1, 2 * R + 1), f32)
    for i, d in enumerate(GAP_D):
        gp[3 * i, 0, R, R - d] = 1.0; gp[3 * i, 0, R, R] = -1.0
        gp[3 * i + 1] = -gp[3 * i]
        gp[3 * i + 2, 0, R, R - d] = 1.0; gp[3 * i + 2, 0, R, R + d] = -1.0
    gm = np.zeros((nd, 3 * nd, 1, 1), f32)
    for i in range(nd):
        gm[i, 3 * i] = 1.0; gm[i, 3 * i + 1] = -1.0; gm[i, 3 * i + 2] = -1.0
    assert nd == 3
    ga = np.zeros((3, 3, 1, 1), f32)                 # (g0, g1, g2) -> (g0, g1 - g0, g2)
    ga[0, 0] = 1.0; ga[1, 1] = 1.0; ga[1, 0] = -1.0; ga[2, 2] = 1.0
    gb = np.zeros((2, 3, 1, 1), f32)                 # m = g0 + relu(g1 - g0);  -> (m, g2 - m)
    gb[0, 0] = 1.0; gb[0, 1] = 1.0; gb[1, 2] = 1.0; gb[1, 0] = -1.0; gb[1, 1] = -1.0
    gv = np.zeros((1, 1, GAP_ROWS, GAP_ROWS), f32); gv[0, 0, :, GAP_ROWS // 2] = 1.0 / GAP_ROWS
    n_g = len(GAP_CENTRES) * len(GAP_COLS)
    groi = np.zeros((n_g, 1, 28, 26), f32); j = 0
    for rr in GAP_CENTRES:
        for cc in GAP_COLS:
            groi[j, 0, rr, cc] = 1.0; j += 1
    gw = GAP_T1 - GAP_T0
    # gwr = gw - min(relu(G - T0), gw) = gw * (1 - Lgap);  trouser: DELTA_GAP * (2 * Lgap - 1)
    gcue = np.zeros((10, 1), f32); gcue[1, 0] = -2.0 * DELTA_GAP / gw
    gcue_b = np.zeros(10, f32); gcue_b[1] = DELTA_GAP
    weights.update({
        "ld.W": ld, "ld.b": np.zeros(2, f32),
        "lc1.W": -eye, "lc1.b": np.full(2, CMAX, f32),
        "lc2.W": -eye, "lc2.b": np.full(2, CMAX, f32),
        "lv.W": lv, "lv.b": np.zeros(2, f32),
        "roi.W": roi.reshape(4 * 28, -1), "roi.b": np.zeros(4 * 28, f32),
        "lq.W": lq, "lq.b": np.full(2, -TH0, f32),
        "lw.W": np.full((1, 2), -1.0, f32), "lw.b": np.array([width], f32),
        "cue.W": cue, "cue.b": cue_b,
        "gp.W": gp, "gp.b": np.zeros(3 * nd, f32),
        "gm.W": gm, "gm.b": np.zeros(nd, f32),
        "ga.W": ga, "ga.b": np.zeros(3, f32),
        "gb.W": gb, "gb.b": np.zeros(2, f32),
        "gc1.W": np.full((1, 2, 1, 1), -1.0, f32), "gc1.b": np.array([GAP_CLIP], f32),   # clip - max(m, g2)
        "gc2.W": np.full((1, 1, 1, 1), -1.0, f32), "gc2.b": np.array([GAP_CLIP], f32),
        "gv.W": gv, "gv.b": np.zeros(1, f32),
        "groi.W": groi.reshape(n_g, -1), "groi.b": np.zeros(n_g, f32),
        "gq.W": np.ones((1, 1), f32), "gq.b": np.array([-GAP_T0], f32),
        "gw.W": np.full((1, 1), -1.0, f32), "gw.b": np.array([gw], f32),
        "gcue.W": gcue, "gcue.b": gcue_b,
    })
    layers += [
        {"name": "nrm", "op": "norm", "inputs": ["image"]},
        {"name": "ld", "op": "conv", "inputs": ["nrm"], "out": 2, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "ldr", "op": "relu", "inputs": ["ld"]},
        {"name": "lc1", "op": "conv", "inputs": ["ldr"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "lc1r", "op": "relu", "inputs": ["lc1"]},
        {"name": "lc2", "op": "conv", "inputs": ["lc1r"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "lv", "op": "conv", "inputs": ["lc2"], "out": 2, "kernel": [LINE_ROWS, LINE_ROWS], "stride": 1, "padding": LINE_ROWS // 2},
        {"name": "roi", "op": "dense", "inputs": ["lv"], "out": 4 * 28},
        {"name": "roi_r", "op": "reshape", "inputs": ["roi"], "shape": [4, 1, 28]},
        {"name": "roi_m", "op": "maxpool", "inputs": ["roi_r"], "size": [1, 28], "stride": [1, 28]},
        {"name": "lq", "op": "dense", "inputs": ["roi_m"], "out": 2},
        {"name": "lqr", "op": "relu", "inputs": ["lq"]},
        {"name": "lw", "op": "dense", "inputs": ["lqr"], "out": 1},
        {"name": "lwr", "op": "relu", "inputs": ["lw"]},
        {"name": "cue", "op": "dense", "inputs": ["lwr"], "out": 10},
        {"name": "gp", "op": "conv", "inputs": ["nrm"], "out": 3 * nd, "kernel": [2 * R + 1, 2 * R + 1], "stride": 1, "padding": R},
        {"name": "gpr", "op": "relu", "inputs": ["gp"]},
        {"name": "gm", "op": "conv", "inputs": ["gpr"], "out": nd, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "gmr", "op": "relu", "inputs": ["gm"]},
        {"name": "ga", "op": "conv", "inputs": ["gmr"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "gar", "op": "relu", "inputs": ["ga"]},
        {"name": "gb", "op": "conv", "inputs": ["gar"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "gbr", "op": "relu", "inputs": ["gb"]},
        {"name": "gc1", "op": "conv", "inputs": ["gbr"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "gc1r", "op": "relu", "inputs": ["gc1"]},
        {"name": "gc2", "op": "conv", "inputs": ["gc1r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "gmx", "op": "maxpool", "inputs": ["gc2"], "size": [1, 3], "stride": [1, 1]},
        {"name": "gv", "op": "conv", "inputs": ["gmx"], "out": 1, "kernel": [GAP_ROWS, GAP_ROWS], "stride": 1, "padding": GAP_ROWS // 2},
        {"name": "groi", "op": "dense", "inputs": ["gv"], "out": n_g},
        {"name": "groi_r", "op": "reshape", "inputs": ["groi"], "shape": [1, 1, n_g]},
        {"name": "groi_m", "op": "maxpool", "inputs": ["groi_r"], "size": [1, n_g], "stride": [1, n_g]},
        {"name": "gq", "op": "dense", "inputs": ["groi_m"], "out": 1},
        {"name": "gqr", "op": "relu", "inputs": ["gq"]},
        {"name": "gw", "op": "dense", "inputs": ["gqr"], "out": 1},
        {"name": "gwr", "op": "relu", "inputs": ["gw"]},
        {"name": "gcue", "op": "dense", "inputs": ["gwr"], "out": 10},
        {"name": "scores", "op": "add", "inputs": ["shape_scores", "cue", "gcue"]},
    ]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
