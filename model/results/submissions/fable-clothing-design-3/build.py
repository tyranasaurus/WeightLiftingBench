"""build.py -- writes network.json and weights.npz for the clothing classifier.  Reads no files: every number comes from this code.

Method: a bank of 80,000 procedurally drawn silhouettes (8,000 per class) of garments, shoes and bags, made from hand-written
shape models (general garment / shoe / bag proportions, fixed random seeds), is turned into the weights of dense layers that
compute, for every drawn prototype, minus half the squared distance between the photo's mask features and the prototype's.
Hand-designed front end: soft foreground mask, morphological closing / erosion / thin-vertical-gap maps (max-pooling), blur.
Hand-set tables add sub-type priors and two interior cues (centre front line, dim item).  The class score is its best prototype.
No training, fitting or data statistics are used anywhere.
"""
import numpy as np

# ----------------------------------------------------------------------------- rasteriser
class Shape:
    def __init__(self):
        self.prims = []

    def poly(self, pts, sign=1):
        self.prims.append(("poly", np.asarray(pts, float), sign)); return self

    def stroke(self, pts, width, sign=1):
        self.prims.append(("stroke", np.asarray(pts, float), sign, float(width))); return self

    def ellipse(self, cx, cy, rx, ry, sign=1, rot=0.0, n=20):
        t = np.arange(n) * 2 * np.pi / n
        x, y = rx * np.cos(t), ry * np.sin(t)
        c, s = np.cos(rot), np.sin(rot)
        return self.poly(np.stack([cx + c * x - s * y, cy + s * x + c * y], 1), sign)

    def transform(self, rot=0.0, shear=0.0, sx=1.0, sy=1.0):
        c, s = np.cos(rot), np.sin(rot)
        A = np.array([[c, -s], [s, c]]) @ np.array([[1.0, shear], [0.0, 1.0]]) @ np.diag([sx, sy])
        out = []
        for p in self.prims:
            out.append((p[0], p[1] @ A.T) + tuple(p[2:]))
        self.prims = out; return self


def _poly_mask(xs, ys, pts):
    x1, y1 = pts[:, 0], pts[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    cond = (y1[:, None] > ys[None, :]) != (y2[:, None] > ys[None, :])          # (E, gy)
    dy = np.where(y2 == y1, 1.0, y2 - y1)
    xint = x1[:, None] + (ys[None, :] - y1[:, None]) * ((x2 - x1) / dy)[:, None]   # (E, gy)
    cross = cond[:, :, None] & (xs[None, None, :] < xint[:, :, None])           # (E, gy, gx)
    return (cross.sum(0) & 1).astype(bool)


def _stroke_mask(xs, ys, pts, width):
    X, Y = xs[None, :], ys[:, None]
    m = np.zeros((len(ys), len(xs)), bool)
    r2 = (width / 2.0) ** 2
    for i in range(len(pts) - 1):
        ax, ay = pts[i]; bx, by = pts[i + 1]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy + 1e-12
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / l2, 0.0, 1.0)
        m |= ((X - ax - t * dx) ** 2 + (Y - ay - t * dy) ** 2) <= r2
    return m


def _resample_matrix(n, m):
    """area-averaging matrix (m, n): n source cells -> m target cells"""
    e = np.arange(m + 1) * (n / m)
    j = np.arange(n)
    lo = np.maximum(e[:-1, None], j[None, :]); hi = np.minimum(e[1:, None], j[None, :] + 1)
    return np.clip(hi - lo, 0, None) / (n / m)


def render(shape, G=96, odd=0):
    """coverage image (28, 28) in [0, 1] of the shape after trim / scale-longest-side-to-28 / centre."""
    lo = np.array([np.inf, np.inf]); hi = -lo
    for p in shape.prims:
        if p[2] > 0:
            pad = p[3] / 2 if p[0] == "stroke" else 0.0
            lo = np.minimum(lo, p[1].min(0) - pad); hi = np.maximum(hi, p[1].max(0) + pad)
    L = max(hi[0] - lo[0], hi[1] - lo[1])
    step = L / G
    xs = lo[0] + (np.arange(int(np.ceil((hi[0] - lo[0]) / step))) + 0.5) * step
    ys = lo[1] + (np.arange(int(np.ceil((hi[1] - lo[1]) / step))) + 0.5) * step
    mask = np.zeros((len(ys), len(xs)), bool)
    for p in shape.prims:
        m = _poly_mask(xs, ys, p[1]) if p[0] == "poly" else _stroke_mask(xs, ys, p[1], p[3])
        mask = (mask | m) if p[2] > 0 else (mask & ~m)
    rows = np.where(mask.any(1))[0]; cols = np.where(mask.any(0))[0]
    crop = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].astype(float)
    h, w = crop.shape
    if h >= w:
        H, W = 28, int(min(28, max(1, round(28.0 * w / h))))
    else:
        H, W = int(min(28, max(1, round(28.0 * h / w)))), 28
    small = _resample_matrix(h, H) @ crop @ _resample_matrix(w, W).T
    out = np.zeros((28, 28))
    oy = (28 - H + (odd if (28 - H) % 2 else 0)) // 2
    ox = (28 - W + (odd if (28 - W) % 2 else 0)) // 2
    out[oy:oy + H, ox:ox + W] = small
    return out


# ----------------------------------------------------------------------------- helpers
def U(rng, a, b):
    return a + (b - a) * rng.random()


def smooth(pts, n=6, closed=False):
    """Catmull-Rom curve through the control points"""
    P = np.asarray(pts, float)
    if closed:
        Q = np.vstack([P[-1], P, P[0], P[1]])
    else:
        Q = np.vstack([P[0], P, P[-1]])
    out = []
    t = (np.arange(n) / n)[:, None]
    for i in range(len(Q) - 3):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out = np.vstack(out)
    if not closed:
        out = np.vstack([out, P[-1]])
    return out


def mirror(pts):
    return [(-x, y) for x, y in pts]


# ----------------------------------------------------------------------------- upper-body garments
def bezier(p0, p1, p2, n=5):
    t = (np.arange(1, n + 1) / n)[:, None]
    p0, p1, p2 = (np.asarray(q, float) for q in (p0, p1, p2))
    return [tuple(q) for q in (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t ** 2 * p2]


def upper(rng, p):
    """Front view of a garment on a hanger / ghost mannequin.  Units: neck top y = 0, hem y = p['len'], centre x = 0.
    The neck is drawn filled (the back of the collar shows behind the front neckline) unless p['notch'] > 0.
    p: neck_hw notch(depth of an open neckline) sh_drop chest_hw armpit_y waist_hw waist_y hem_hw len hem_curve
       [hip_hw hip_y] [strap: shoulder half width for sleeveless cuts]
       sleeve: None | dict(kind='short', sh_ext, len, ang, w_cuff, asym) | dict(kind='long', sh_ext, w_cuff, gap, flare, y_cuff, round, bend)
       collar: None | dict(kind='stand'|'hood', h, w)"""
    s = Shape()
    nw = p["neck_hw"]; sd = p["sh_drop"]
    bw, ay = p["chest_hw"], p["armpit_y"]
    ww, wy = p["waist_hw"], p["waist_y"]
    hw, L, hc = p["hem_hw"], p["len"], p["hem_curve"]
    sl = p.get("sleeve")
    sides = {}
    for side in (1, -1):
        if sl is None:                                   # sleeveless: strap, armhole curving in and back out to the chest
            sw = p["strap"]
            top = [(nw, 0.0), (sw, sd), (sw + 0.012, sd + 0.45 * (ay - sd)), (0.5 * (sw + bw) + 0.01, sd + 0.85 * (ay - sd))]
            sleeve_poly = None
        elif sl["kind"] == "short":
            sw = bw + sl["sh_ext"]
            a = sl["ang"] * (1 + sl.get("asym", 0.0) * side)
            u = np.array([np.sin(a), np.cos(a)]); n_in = np.array([-np.cos(a), np.sin(a)])
            S = np.array([sw, sd]); Co = S + sl["len"] * u; Ci = Co + sl["w_cuff"] * n_in
            top = [(nw, 0.0), (sw, sd)]
            sleeve_poly = [(0.5 * (nw + sw), 0.5 * sd), tuple(S), tuple(Co), tuple(Ci), (bw - 0.02, ay)]
        else:                                            # long sleeve hanging beside the body
            sw = bw + sl["sh_ext"]
            fl = sl["flare"] * (1 + sl.get("asym", 0.0) * side)
            S = np.array([sw, sd])
            Co = np.array([bw + sl["gap"] + sl["w_cuff"] + fl, sl["y_cuff"]])
            Ci = np.array([bw + sl["gap"] + fl, sl["y_cuff"] + 0.01])
            D = S + sl["round"] * (Co - S)
            curve = bezier((nw, 0.0), S, D, n=5)
            top = [(nw, 0.0)] + curve
            if sl.get("bend", 0.0):                      # elbow pushed outward
                E = 0.5 * (D + Co) + np.array([sl["bend"], 0.0]); Ei = 0.5 * (np.array([bw, ay]) + Ci) + np.array([sl["bend"], 0.0])
                sleeve_poly = [curve[2], tuple(D), tuple(E), tuple(Co), tuple(Ci), tuple(Ei), (bw - 0.02, ay)]
            else:
                sleeve_poly = [curve[2], tuple(D), tuple(Co), tuple(Ci), (bw - 0.02, ay)]
        body = top + [(bw, ay), (ww, wy)]
        if p.get("hip_hw") is not None:
            body += [(p["hip_hw"], p["hip_y"])]
        body += [(hw, L)]
        sides[side] = [(side * x, y) for x, y in body]
        if sleeve_poly is not None:
            s.poly([(side * x, y) for x, y in sleeve_poly])
    hem = [(hw * t, L + hc * (1 - t * t)) for t in (0.6, 0.2, -0.2, -0.6)]
    nd = p.get("notch", 0.0)
    if nd > 0:
        if p.get("neck_kind") == "v":
            neck = [(0.0, nd)]
        else:
            neck = [(-nw * t, nd * (1 - t * t) ** 0.7) for t in (0.7, 0.35, 0.0, -0.35, -0.7)]
    else:
        neck = []
    s.poly(sides[1] + hem + sides[-1][::-1] + neck)
    col = p.get("collar")
    if col is not None:
        h, w = col["h"], col.get("w", nw)
        if col["kind"] == "stand":                       # collar / turtle neck standing above the shoulder line
            s.poly([(-w - 0.01, 0.04), (-w * 0.9, -h), (w * 0.9, -h), (w + 0.01, 0.04)])
        else:                                            # hood lying behind the neck: a rounded bump
            s.ellipse(0.0, 0.03 - 0.5 * h, w, 0.5 * h + 0.04)
    return s


def jitter(s, rng, rot=0.025, shear=0.025):
    return s.transform(rot=rng.normal(0, rot), shear=rng.normal(0, shear))


def _torso(rng, bw, L, taper=(-0.03, 0.02), hem=(-0.03, 0.03), sd=(0.05, 0.10), ay=(0.26, 0.36), nw=(0.10, 0.16), hc=(0.0, 0.02)):
    return dict(neck_hw=U(rng, *nw), sh_drop=U(rng, *sd), chest_hw=bw, armpit_y=U(rng, *ay), waist_hw=bw + U(rng, *taper), waist_y=0.62 * L,
                hem_hw=bw + U(rng, *hem), len=L, hem_curve=U(rng, *hc))


def _short(rng, ext=(0.02, 0.08), ln=(0.17, 0.30), ang=(35, 68), wc=(0.14, 0.20)):
    return dict(kind="short", sh_ext=U(rng, *ext), len=U(rng, *ln), ang=np.radians(U(rng, *ang)), w_cuff=U(rng, *wc), asym=rng.normal(0, 0.05))


def _long(rng, wc, gap, flare, y_cuff, rnd, bend_p=0.15):
    w = U(rng, *wc); g = U(rng, *gap)
    return dict(kind="long", sh_ext=w * U(rng, 0.75, 1.05) + max(g, 0.0) * 0.5, w_cuff=w, gap=g, flare=U(rng, *flare), y_cuff=y_cuff,
                round=U(rng, *rnd), bend=U(rng, 0.01, 0.04) if rng.random() < bend_p else 0.0, asym=rng.normal(0, 0.15))


# Each generator returns (shape, kind): kind names the sub-type, so that hand-set priors and cue tables can refer to it.


def gen_tshirt(rng):
    r = rng.random()
    kind = "tee"
    if r < 0.70:                               # classic short-sleeved tee
        p = _torso(rng, U(rng, 0.21, 0.33), 1.0, ay=(0.27, 0.38))
        p["sleeve"] = _short(rng, ext=(0.03, 0.09), ln=(0.16, 0.30), ang=(35, 66), wc=(0.13, 0.19))
        if rng.random() < 0.25:                # fitted cut
            p["waist_hw"] = p["chest_hw"] - U(rng, 0.02, 0.05); p["waist_y"] = U(rng, 0.5, 0.65)
    elif r < 0.86:                             # tank top / sleeveless top
        bw = U(rng, 0.20, 0.31); nw = U(rng, 0.08, 0.16)
        p = _torso(rng, bw, 1.0, hem=(-0.02, 0.05), sd=(0.0, 0.03), ay=(0.22, 0.36))
        p.update(neck_hw=nw, strap=nw + U(rng, 0.03, 0.10), notch=U(rng, 0.06, 0.28), neck_kind="round")
        if rng.random() < 0.35:
            p["notch"] = 0.0
        kind = "tank"
    elif r < 0.94:                             # boxy / cap-sleeved top
        p = _torso(rng, U(rng, 0.27, 0.38), U(rng, 0.8, 1.0), nw=(0.12, 0.20), ay=(0.20, 0.30))
        p["sleeve"] = _short(rng, ext=(0.03, 0.10), ln=(0.07, 0.18), ang=(45, 85), wc=(0.14, 0.22))
    else:                                      # elbow-length sleeves
        p = _torso(rng, U(rng, 0.22, 0.32), 1.0)
        p["sleeve"] = _short(rng, ext=(0.02, 0.06), ln=(0.32, 0.48), ang=(22, 50), wc=(0.11, 0.16))
        kind = "elbow"
    return jitter(upper(rng, p), rng), kind


def _hang(rng, p_band, p_open, band_drop, hem_drop):
    """How long sleeves hang: 'band' = the cuffs reach lower than the hem (a dark band shows between them under the hem);
    'open' = the sleeves stand clear of the body (a dark slit between sleeve and body).  Returns (tag, hem - cuff, gap, flare)."""
    band = rng.random() < p_band; opn = rng.random() < p_open
    dy = -U(rng, *band_drop) if band else U(rng, *hem_drop)
    if opn:
        gap, flare = U(rng, 0.02, 0.06), U(rng, 0.0, 0.06)
    else:
        gap, flare = U(rng, -0.03, 0.008), U(rng, 0.0, 0.015)
    return ("b" if band else "n") + ("o" if opn else "m"), dy, gap, flare


def gen_pullover(rng):
    r = rng.random()
    tag, dy, gap, flare = _hang(rng, 0.70, 0.60, (0.07, 0.22), (-0.04, 0.08))
    yc = U(rng, 0.92, 1.04)
    p = _torso(rng, U(rng, 0.24, 0.33), yc + dy, taper=(-0.02, 0.01), hem=(-0.05, 0.0), sd=(0.06, 0.15), ay=(0.30, 0.40), nw=(0.11, 0.17))
    p["sleeve"] = _long(rng, (0.10, 0.15), (gap, gap), (flare, flare), yc, (0.10, 0.30))
    if r < 0.15:
        p["collar"] = dict(kind="hood", h=U(rng, 0.06, 0.13), w=p["neck_hw"] + U(rng, 0.02, 0.06))
    elif r < 0.35:
        p["collar"] = dict(kind="stand", h=U(rng, 0.02, 0.07))
    return jitter(upper(rng, p), rng), "pullover_" + tag


def gen_coat(rng):
    r = rng.random()
    if r < 0.78:                               # jacket / short coat
        tag, dy, gap, flare = _hang(rng, 0.35, 0.30, (0.05, 0.16), (-0.02, 0.20))
        yc = U(rng, 0.92, 1.04); L = yc + dy; bw = U(rng, 0.25, 0.34); kind = "jacket_" + tag
    else:                                      # long coat: the body hangs well below the sleeves
        tag, dy, gap, flare = _hang(rng, 0.0, 0.30, (0.0, 0.0), (0.0, 0.0))
        yc = U(rng, 0.92, 1.06); L = yc * U(rng, 1.3, 1.9); bw = U(rng, 0.23, 0.32); kind = "longcoat"
    p = _torso(rng, bw, L, taper=(-0.03, 0.01), hem=(-0.02, 0.06), sd=(0.07, 0.17), ay=(0.30, 0.40), nw=(0.10, 0.17))
    p["sleeve"] = _long(rng, (0.11, 0.16), (gap, gap), (flare, flare), yc, (0.15, 0.35))
    q = rng.random()
    if q < 0.30:
        p["collar"] = dict(kind="hood", h=U(rng, 0.08, 0.16), w=p["neck_hw"] + U(rng, 0.03, 0.08))
    elif q < 0.85:
        p["collar"] = dict(kind="stand", h=U(rng, 0.03, 0.10))
    return jitter(upper(rng, p), rng), kind


def gen_shirt(rng):
    r = rng.random()
    if r < 0.66:                               # long-sleeved button shirt / blouse
        tag, dy, gap, flare = _hang(rng, 0.20, 0.50, (0.05, 0.16), (-0.03, 0.12))
        yc = U(rng, 0.92, 1.04)
        p = _torso(rng, U(rng, 0.24, 0.32), yc + dy, taper=(-0.03, 0.0), hem=(-0.03, 0.03), sd=(0.05, 0.12), ay=(0.28, 0.36), nw=(0.08, 0.13), hc=(0.0, 0.07))
        p["sleeve"] = _long(rng, (0.08, 0.12), (gap, gap), (flare, flare), yc, (0.05, 0.20), bend_p=0.3)
        if rng.random() < 0.8:
            p["collar"] = dict(kind="stand", h=U(rng, 0.02, 0.06))
        kind = "shirt_" + tag
    elif r < 0.88:                             # short-sleeved shirt / polo / blouse: the outline of a tee
        p = _torso(rng, U(rng, 0.22, 0.34), 1.0, nw=(0.08, 0.14), hc=(0.0, 0.06))
        p["sleeve"] = _short(rng, ext=(0.02, 0.08), ln=(0.20, 0.38), ang=(28, 62), wc=(0.13, 0.20))
        if rng.random() < 0.8:
            p["collar"] = dict(kind="stand", h=U(rng, 0.02, 0.06))
        kind = "shirt_short"
    else:                                      # sleeveless blouse / camisole
        bw = U(rng, 0.22, 0.34); nw = U(rng, 0.09, 0.18)
        p = _torso(rng, bw, 1.0, hem=(0.0, 0.10), sd=(0.0, 0.03), ay=(0.20, 0.34), hc=(0.0, 0.05))
        p.update(neck_hw=nw, strap=nw + U(rng, 0.02, 0.10), notch=U(rng, 0.05, 0.25), neck_kind="v" if rng.random() < 0.5 else "round")
        kind = "shirt_sleeveless"
    return jitter(upper(rng, p), rng), kind


def gen_dress(rng):
    r = rng.random()
    L = 1.0
    bw = U(rng, 0.09, 0.24)                     # dresses are long: the chest is narrow relative to the length
    wy = U(rng, 0.26, 0.40)
    waist = bw - U(rng, 0.0, 0.035)
    straight = rng.random() < 0.5
    hem = waist + (U(rng, 0.0, 0.05) if straight else U(rng, 0.05, 0.20))
    p = dict(chest_hw=bw, armpit_y=U(rng, 0.14, 0.24), waist_hw=waist, waist_y=wy, hem_hw=hem, len=L, hem_curve=U(rng, 0.0, 0.03), sh_drop=U(rng, 0.02, 0.06))
    if rng.random() < 0.5:                      # hips, then straight or flaring down
        p["hip_hw"] = waist + U(rng, 0.01, 0.04) + 0.3 * (hem - waist); p["hip_y"] = wy + U(rng, 0.15, 0.25)
    kind = "dress"
    if r < 0.45:                               # straps / sleeveless
        nw = min(bw * 0.8, U(rng, 0.05, 0.12))
        p.update(neck_hw=nw, strap=min(bw, nw + U(rng, 0.02, 0.08)), sh_drop=U(rng, 0.0, 0.02), notch=U(rng, 0.05, 0.22) if rng.random() < 0.75 else 0.0,
                 neck_kind="v" if rng.random() < 0.4 else "round")
        if straight and bw > 0.19:
            kind = "dress_top"                 # as wide and straight as a sleeveless top
    elif r < 0.75:                             # short sleeves
        p.update(neck_hw=min(bw * 0.7, U(rng, 0.06, 0.11)))
        p["sleeve"] = _short(rng, ext=(0.01, 0.05), ln=(0.08, 0.22), ang=(30, 70), wc=(0.08, 0.13))
        if bw > 0.19:
            kind = "dress_top"                 # as wide as a tee
    elif r < 0.93:                             # long sleeves
        p.update(neck_hw=min(bw * 0.7, U(rng, 0.06, 0.11)))
        p["sleeve"] = _long(rng, (0.055, 0.085), (0.0, 0.05), (0.0, 0.10), U(rng, 0.50, 0.72), (0.05, 0.2), bend_p=0.3)
        kind = "dress_long"
    else:                                      # strapless / tube
        p.update(neck_hw=bw * U(rng, 0.7, 0.95), strap=bw, sh_drop=0.0, armpit_y=0.03)
    return jitter(upper(rng, p), rng), kind


# ----------------------------------------------------------------------------- trousers
def gen_trouser(rng):
    s = Shape()
    r = rng.random()
    kind = "trouser"
    L = 1.0
    wh = U(rng, 0.15, 0.22)                     # waist half width
    hip = wh + U(rng, 0.0, 0.04)
    hy = U(rng, 0.12, 0.22)
    cy = U(rng, 0.24, 0.38)                     # crotch
    if r < 0.82:                               # two legs with a visible gap
        legw = U(rng, 0.085, 0.15)              # leg width at the hem
        gap = U(rng, 0.008, 0.06) if rng.random() < 0.8 else U(rng, 0.06, 0.18)   # half gap between the legs at the hem
        out = gap + legw
        kx = 0.5 * (hip + out) + U(rng, -0.02, 0.02)   # knee
        right = [(wh, 0.0), (hip, hy), (kx, 0.62), (out, L), (gap, L + U(rng, -0.01, 0.01)), (0.55 * gap, 0.62), (0.0, cy)]
        s.poly(right + mirror(right)[::-1][1:])
        if rng.random() < 0.12:                 # legs swung slightly to one side
            s.transform(shear=rng.normal(0, 0.06))
    elif r < 0.93:                             # legs together: a tall block, straight or tapering, flat waistband on top
        out = hip + U(rng, -0.07, 0.01)
        right = [(wh, 0.0), (hip, hy), (0.5 * (hip + out), 0.62), (out, L)]
        s.poly(right + mirror(right)[::-1])
        kind = "trouser_closed"
    elif r < 0.97:                             # wide legs / culottes with a slit
        out = U(rng, 0.20, 0.32)
        right = [(wh, 0.0), (hip, hy), (out, L)]
        s.poly(right + mirror(right)[::-1])
        g = U(rng, 0.008, 0.03)
        s.poly([(0.0, cy), (g, L + 0.01), (-g, L + 0.01)], sign=-1)
    else:                                      # shorts
        Ls = U(rng, 0.45, 0.75)
        out = hip + U(rng, 0.0, 0.08)
        right = [(wh, 0.0), (hip, hy * 0.8), (out, Ls), (U(rng, 0.02, 0.06), Ls), (0.0, Ls * U(rng, 0.55, 0.75))]
        s.poly(right + mirror(right)[::-1][1:])
    return jitter(s, rng, rot=0.02, shear=0.02), kind


# ----------------------------------------------------------------------------- bags
def gen_bag(rng):
    s = Shape()
    w = 1.0
    h = U(rng, 0.45, 1.05)                      # body height relative to its width
    kind = rng.random()
    top = U(rng, 0.78, 1.0) if rng.random() < 0.6 else 1.0     # top edge narrower than the bottom (trapezoid)
    if rng.random() < 0.25:
        top = 1.0 / U(rng, 0.8, 1.0)            # or wider at the top (tote)
    rc = U(rng, 0.03, 0.14)                     # corner rounding
    tw, bwd = 0.5 * w * min(top, 1.0) / max(1.0, 1.0), 0.5 * w
    if top > 1.0:
        tw, bwd = 0.5 * w, 0.5 * w / top
    if kind < 0.12:                            # round / hobo body
        s.ellipse(0.0, h * 0.5, 0.5 * w, 0.5 * h * U(rng, 1.0, 1.15))
        s.poly([(-tw * 0.9, 0.0), (tw * 0.9, 0.0), (bwd * 0.8, h * 0.6), (-bwd * 0.8, h * 0.6)])
    else:
        body = [(-tw + rc, 0.0), (tw - rc, 0.0), (tw, rc), (bwd, h - rc), (bwd - rc, h), (-bwd + rc, h), (-bwd, h - rc), (-tw, rc)]
        s.poly(body)
    hk = rng.random()
    th = U(rng, 0.05, 0.09)
    if hk < 0.30:
        pass                                   # clutch / no handle visible
    elif hk < 0.72:                            # one or two arched handles
        a = U(rng, 0.18, 0.42) * w; hh = U(rng, 0.25, 0.75)
        arc = smooth([(-a, 0.02), (-a * U(rng, 0.75, 1.0), -hh * 0.7), (0.0, -hh), (a * U(rng, 0.75, 1.0), -hh * 0.7), (a, 0.02)], n=5)
        s.stroke(arc, th)
        if rng.random() < 0.3:
            s.stroke(arc * np.array([U(rng, 0.7, 0.9), U(rng, 0.75, 0.95)]) + np.array([U(rng, -0.05, 0.05), 0.0]), th)
    elif hk < 0.90:                            # long shoulder strap looping above the body
        a = tw * U(rng, 0.85, 1.0); hh = U(rng, 0.5, 1.1)
        arc = smooth([(-a, 0.05), (-a * U(rng, 0.5, 0.9), -hh * 0.75), (U(rng, -0.1, 0.1), -hh), (a * U(rng, 0.5, 0.9), -hh * 0.75), (a, 0.05)], n=5)
        s.stroke(arc, th * 0.8)
    else:                                      # short grab handle on a flap + side strap
        a = U(rng, 0.10, 0.2); hh = U(rng, 0.08, 0.2)
        s.stroke(smooth([(-a, 0.0), (-a, -hh), (a, -hh), (a, 0.0)], n=4), th)
        if rng.random() < 0.5:
            sd = 1 if rng.random() < 0.5 else -1
            s.stroke([(sd * tw, 0.1), (sd * (bwd + U(rng, 0.03, 0.1)), h * U(rng, 0.4, 0.9))], th)
    return jitter(s, rng, rot=0.04, shear=0.04), "bag"


# ----------------------------------------------------------------------------- shoes (toe to the left, y up, ground y = 0)
def _flip(pts):
    P = np.asarray(pts, float).copy(); P[:, 1] *= -1
    return P


def gen_sneaker(rng):
    s = Shape()
    hi_top = rng.random() < 0.10
    sole = U(rng, 0.04, 0.09)
    toe_h = U(rng, 0.13, 0.20)
    if hi_top:
        ch = U(rng, 0.55, 0.85); cx = U(rng, 0.50, 0.62)
        up = [(0.0, sole * 0.7), (0.03, toe_h * 0.8), (0.14, toe_h + 0.03), (0.32, toe_h + 0.10), (cx - 0.04, ch * 0.72), (cx, ch * 0.97),
              (cx + 0.12, ch), (0.93, ch * 0.98), (0.985, ch * 0.80), (1.0, 0.25), (0.99, 0.0)]
    else:
        tx = U(rng, 0.52, 0.68); th = U(rng, 0.33, 0.48)        # tongue top
        cd = U(rng, 0.03, 0.10)                                  # dip of the collar behind the tongue
        hb = th - cd + U(rng, 0.0, 0.07)                         # heel tab
        up = [(0.0, sole * 0.7), (0.03, toe_h * 0.8), (0.14, toe_h + 0.02), (0.32, toe_h + 0.07), (tx - 0.10, th * 0.86), (tx, th),
              (tx + 0.07, th - cd * 0.6), (0.5 * (tx + 0.95), th - cd), (0.94, hb), (0.99, hb * 0.82), (1.0, 0.18), (0.99, 0.0)]
    bottom = [(0.99, 0.0), (0.06, 0.0), (0.0, sole * 0.7)]
    s.poly(_flip(np.vstack([smooth(up, n=4), bottom])))
    return s.transform(rot=rng.normal(0, 0.02), sy=U(rng, 0.92, 1.1)), ("hightop" if hi_top else "sneaker")


def gen_boot(rng):
    s = Shape()
    heel = rng.random()
    hh = 0.0 if heel < 0.45 else (U(rng, 0.05, 0.14) if heel < 0.75 else U(rng, 0.14, 0.34))   # heel height
    sole = U(rng, 0.03, 0.07)
    toe_h = U(rng, 0.12, 0.20)
    fx = U(rng, 0.40, 0.58)                                       # front of the shaft
    top = hh + U(rng, 0.52, 0.95)                                 # top of the shaft
    bx = U(rng, 0.94, 1.0)                                        # back of the shaft at the top
    lean = U(rng, -0.04, 0.06)
    instep = hh * 0.55 + toe_h + U(rng, 0.10, 0.20)
    up = [(0.0, sole * 0.6), (0.03, toe_h * 0.8), (0.13, toe_h + 0.02 + hh * 0.05), (0.30, toe_h + 0.07 + hh * 0.3), (fx - 0.05, instep),
          (fx + lean * 0.3, instep + 0.12), (fx + lean, top - 0.03), (fx + lean + 0.02, top)]
    back = [(bx, top + U(rng, -0.03, 0.04)), (bx + 0.01, top * 0.6), (1.0, hh + 0.12), (0.99, hh + 0.02)]
    if hh > 0.02:
        hw = U(rng, 0.05, 0.11) if hh > 0.2 and rng.random() < 0.5 else U(rng, 0.12, 0.26)
        bottom = [(0.98, 0.0), (0.98 - hw, 0.0), (0.98 - hw - 0.02, hh * 0.85), (0.55, hh * 0.35), (0.42, 0.0), (0.05, 0.0)]
    else:
        bottom = [(0.985, 0.0), (0.05, 0.0)]
    s.poly(_flip(np.vstack([smooth(up, n=4), back, bottom])))
    return s.transform(rot=rng.normal(0, 0.02)), "boot"


def gen_sandal(rng):
    s = Shape()
    r = rng.random()
    t = U(rng, 0.06, 0.11)                      # strap thickness: straps are a couple of pixels wide at this size
    if r < 0.45:                               # flat sandal / flip-flop / slide
        th = U(rng, 0.05, 0.12)
        back = th + (U(rng, 0.02, 0.10) if rng.random() < 0.35 else 0.0)        # slight wedge toward the heel
        s.poly(_flip([(0.0, 0.0), (1.0, 0.0), (1.0, back), (0.5, 0.5 * (th + back)), (0.03, th * 0.8)]))
        top = th
        k = rng.random()
        if k < 0.22:                           # chunky sport / trekking sandal: a low upper with openings
            uh = U(rng, 0.20, 0.36)
            up = [(0.02, top), (0.10, top + uh * 0.45), (0.35, top + uh * 0.75), (0.62, top + uh), (0.80, top + uh * 0.8), (0.97, top + uh * 0.95), (0.99, top)]
            s.poly(_flip(smooth(up, n=4)))
            for i in range(int(rng.integers(2, 5))):
                cx = U(rng, 0.15, 0.85)
                s.ellipse(cx, -(top + uh * U(rng, 0.25, 0.5) * (0.5 + cx)), U(rng, 0.05, 0.11), U(rng, 0.04, 0.08), sign=-1, rot=U(rng, -0.6, 0.6), n=10)
        else:
            if k < 0.45:                       # thong: strap from the toe post back to the sides
                px = U(rng, 0.12, 0.25); ph = U(rng, 0.10, 0.22)
                s.stroke(_flip([(px, top), (px + 0.15, top + ph), (U(rng, 0.55, 0.75), top)]), t)
            elif k < 0.70:                     # slide: one wide band over the forefoot
                a0 = U(rng, 0.10, 0.25); b0 = a0 + U(rng, 0.25, 0.42); bh = U(rng, 0.12, 0.24)
                s.poly(_flip([(a0, top), (a0 + 0.08, top + bh), (b0 - 0.05, top + bh * 1.1), (b0, top)]))
            else:                              # one to three toe straps
                for i in range(int(rng.integers(1, 4))):
                    a0 = U(rng, 0.06, 0.45)
                    s.stroke(_flip([(a0, top), (a0 + U(rng, 0.05, 0.16), top + U(rng, 0.10, 0.22)), (a0 + U(rng, 0.2, 0.3), top)]), t)
            if rng.random() < 0.65:            # heel strap / ankle loop at the back
                bh = U(rng, 0.14, 0.36)
                s.stroke(_flip([(U(rng, 0.9, 0.97), top), (U(rng, 0.86, 0.96), top + bh), (U(rng, 0.58, 0.75), top + bh * U(rng, 0.7, 1.1)),
                                (U(rng, 0.48, 0.62), top)][:int(rng.integers(3, 5))]), t)
    else:                                      # heeled sandal or wedge
        hh = U(rng, 0.20, 0.62)                 # heel height relative to the length
        th = U(rng, 0.05, 0.10)
        ball = U(rng, 0.30, 0.45)
        plat = U(rng, 0.03, 0.12) if rng.random() < 0.45 else 0.0
        sole_top = [(0.0, th * 0.7 + plat), (ball, th + plat), (0.72, hh * 0.72 + plat * 0.5), (0.97, hh)]
        if r > 0.82:                           # wedge: filled under the arch
            s.poly(_flip(np.vstack([smooth(sole_top, n=4), [(0.99, 0.0), (0.02, 0.0)]])))
        else:
            s.stroke(_flip(smooth(sole_top, n=4)), th * 1.3)
            s.poly(_flip([(0.0, 0.0), (ball + 0.04, 0.0), (ball + 0.08, th + plat), (0.0, th + plat)]))
            hw = U(rng, 0.04, 0.08) if rng.random() < 0.45 else U(rng, 0.10, 0.24)
            s.poly(_flip([(0.98, hh + 0.01), (0.98 - hw - 0.06, hh * 0.93), (0.98 - hw, 0.0), (0.98, 0.0)]))
        y0 = th + plat
        for i in range(int(rng.integers(1, 3))):   # toe straps
            a0 = U(rng, 0.04, 0.28)
            s.stroke(_flip([(a0, y0), (a0 + U(rng, 0.06, 0.16), y0 + U(rng, 0.08, 0.22)), (a0 + U(rng, 0.2, 0.32), y0 + 0.03)]), t * U(rng, 1.0, 1.8))
        k = rng.random()
        ah = hh + U(rng, 0.16, 0.42)
        if rng.random() < 0.6:                 # heel cup
            s.poly(_flip([(0.78, hh * 0.85), (0.98, hh), (0.99, hh + U(rng, 0.1, 0.22)), (0.84, hh + 0.06)]))
        if k < 0.8:                            # ankle strap: up the back of the heel, then forward
            s.stroke(_flip([(0.95, hh), (U(rng, 0.90, 0.98), ah), (U(rng, 0.55, 0.78), ah + U(rng, -0.05, 0.08))]), t)
        if k > 0.45:                           # strap from the ankle down to the toes
            s.stroke(_flip([(U(rng, 0.62, 0.78), ah), (U(rng, 0.40, 0.55), hh * 0.5 + 0.1), (U(rng, 0.3, 0.42), y0 + 0.05)]), t)
    return s.transform(rot=rng.normal(0, 0.02)), "sandal"


GENERATORS = [gen_tshirt, gen_trouser, gen_pullover, gen_dress, gen_coat, gen_sandal, gen_shirt, gen_sneaker, gen_bag, gen_boot]
NAMES = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]


# ============================================================================= network builder
import numpy as np, json, sys, time

T0, TS = 0.03, 0.035         # soft foreground threshold: mask = clip((I - T0) / TS, 0, 1)
B0, BS = 0.30, 0.20          # 'bright' mask = clip((I - B0) / BS, 0, 1)
V_TYP = 0.5                  # typical item brightness used to turn pixel coverage into the mask a real photo would give
SIGMA = 1.0                  # blur before 2x subsampling
W_C, W_E, W_G = 1.0, 0.7, 1.0   # weights of the closed, eroded and thin-gap channels relative to the plain mask
LINE_ROWS = (5, 25)          # rows over which a centre front line is looked for, in LINE_SEG segments
LINE_SEG = 4
SEG_T0, SEG_TS = 0.06, 0.12  # per-segment line strength (in units of the image's standard deviation) -> 0..1
LINE_T0, LINE_TS = 0.5, 0.4  # fraction of segments showing the line -> cue 0..1
LINE_VETO = 6.0              # a dark line whose centre is background does not count
# Hand-set tables, in units of half squared distance, looked up by sub-type prefix (default 0):
#   PRIOR: penalty for sub-types that are rare in their class but have the outline of a common sub-type of another class.
#          For long-sleeved tops the tag says how the sleeves hang: b/n = cuffs lower than the hem or not, o/m = sleeves clear of
#          the body or merged with it.  Pullovers mostly hang 'bo', jackets 'nm', shirts 'n?'.
#   LINE:  (bias, gain) added as bias + gain * cue for the centre-front-line cue (zips and button plackets)
#   DIM:   (bias, gain) for the 'item is mostly dim' cue (light-coloured cloth is dim in these inverted photos)
PRIOR = dict(elbow=-2.0, shirt_short=-2.0, shirt_sleeveless=-2.0, trouser_closed=-2.0, hightop=-2.0, dress_top=-1.5,
             pullover_bo=0.0, pullover_bm=-0.6, pullover_no=-1.3, pullover_nm=-1.9,
             jacket_nm=0.0, jacket_bm=-0.9, jacket_no=-1.3, jacket_bo=-2.2,
             shirt_no=0.0, shirt_nm=0.0, shirt_bo=-2.1, shirt_bm=-2.1)
LINE = dict(jacket=(-0.6, 3.6), longcoat=(-0.6, 3.6), shirt_b=(-0.6, 3.6), shirt_n=(-0.6, 3.6), shirt_short=(-0.6, 3.6),
            trouser=(0.0, 3.6), trouser_closed=(-1.0, 4.6))      # a hairline gap or crease between trouser legs is such a line too
DIM = dict(jacket=(0.1, -0.5), longcoat=(0.1, -0.5), shirt=(-0.4, 1.5))


def lookup(table, kind, default):
    best = None
    for key in table:
        if kind.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return default if best is None else table[best]


def gauss_kernel(sigma=SIGMA, r=2):
    a = np.arange(-r, r + 1)
    g = np.exp(-a * a / (2 * sigma * sigma)); g /= g.sum()
    return np.outer(g, g)


def soft_mask(I):
    return np.clip((I - T0) / TS, 0.0, 1.0)


def pad(X, p, value=0.0):
    return np.pad(X, ((0, 0), (p, p), (p, p)), constant_values=value)


def mpool(X, h, w):
    """max over h x w windows, stride 1, no padding"""
    n, H, Wd = X.shape
    return np.max(np.stack([X[:, dy:dy + H - h + 1, dx:dx + Wd - w + 1] for dy in range(h) for dx in range(w)]), 0)


def blur_s2(X, k):
    """5x5 blur, stride 2, padding 2"""
    n, h, w = X.shape
    P = pad(X, 2)
    ho, wo = (h - 1) // 2 + 1, (w - 1) // 2 + 1
    out = np.zeros((n, ho, wo))
    for dy in range(5):
        for dx in range(5):
            out += k[dy, dx] * P[:, dy:dy + 2 * ho:2, dx:dx + 2 * wo:2]
    return out


def features(M):
    """M: (n, 28, 28) soft masks -> the four flattened feature blocks the network computes (already weighted)"""
    k = gauss_kernel()
    C = 1.0 - mpool(1.0 - mpool(pad(M, 2), 3, 3), 3, 3)          # closing: holes and slits up to 2 px are filled
    E = 1.0 - mpool(pad(1.0 - C, 1, 1.0), 3, 3)                   # erosion of the closed mask: thin parts vanish
    T = C - M                                                     # what the closing filled
    G = mpool(mpool(-mpool(pad(-T, 4), 5, 1), 5, 1), 1, 3)        # ... kept where it runs >= 5 px vertically, widened to 3 px
    n = len(M)
    return (blur_s2(M, k).reshape(n, -1), np.sqrt(W_C) * blur_s2(C, k).reshape(n, -1),
            np.sqrt(W_E) * blur_s2(E, k).reshape(n, -1), np.sqrt(W_G) * blur_s2(G, k).reshape(n, -1))


def prototypes(K, seed=12345):
    cov = np.zeros((10, K, 28, 28), np.float32); kinds = []
    for c in range(10):
        rng = np.random.default_rng(seed + c)
        for i in range(K):
            shape, kind = GENERATORS[c](rng)
            cov[c, i] = render(shape, odd=int(rng.integers(0, 2))); kinds.append(kind)
    return cov, kinds


def build(out_dir, K=6000):
    t = time.time()
    cov, kinds = prototypes(K)
    N = 10 * K
    prior = np.array([lookup(PRIOR, k, 0.0) for k in kinds])
    line = np.array([lookup(LINE, k, (0.0, 0.0)) for k in kinds])
    dim = np.array([lookup(DIM, k, (0.0, 0.0)) for k in kinds])
    parts = [features(soft_mask(cov.reshape(N, 28, 28)[i:i + 2000].astype(np.float64) * V_TYP)) for i in range(0, N, 2000)]   # in chunks: memory
    fM, fC, fE, fG = (np.concatenate([p[j] for p in parts]) for j in range(4))
    W = {}
    k = gauss_kernel()
    c11 = lambda w, b: (np.array(w, float).reshape(1, -1, 1, 1), np.array([b], float))
    W["m1.W"] = np.array([1.0 / TS, 1.0 / BS]).reshape(2, 1, 1, 1); W["m1.b"] = np.array([-T0 / TS, -B0 / BS])
    W["m2.W"] = -np.eye(2).reshape(2, 2, 1, 1); W["m2.b"] = np.ones(2)             # relu -> 1 - mask, 1 - bright mask
    W["m3.W"] = -np.eye(2).reshape(2, 2, 1, 1); W["m3.b"] = np.ones(2)             # mask, bright mask
    kk = np.zeros((1, 2, 5, 5)); kk[0, 0] = k
    W["fM.W"] = kk; W["fM.b"] = np.zeros(1)
    W["cl0.W"], W["cl0.b"] = c11([1.0, 0.0], 0.0)                                   # mask, zero-padded by 2
    W["cl2.W"], W["cl2.b"] = c11([-1.0], 1.0)                                       # 1 - dilated
    W["cl4.W"], W["cl4.b"] = c11([-1.0], 1.0)                                       # closed mask
    W["fC.W"] = (k * np.sqrt(W_C)).reshape(1, 1, 5, 5); W["fC.b"] = np.zeros(1)
    W["e0.W"], W["e0.b"] = c11([-1.0], 1.0)                                         # 1 - closed, padded by 1 with ones
    W["e2.W"], W["e2.b"] = c11([-1.0], 1.0)                                         # eroded closed mask
    W["fE.W"] = (k * np.sqrt(W_E)).reshape(1, 1, 5, 5); W["fE.b"] = np.zeros(1)
    W["nm.W"], W["nm.b"] = c11([-1.0, 0.0], 0.0)                                    # - mask
    W["g1.W"], W["g1.b"] = c11([-1.0], 0.0)                                         # - filled, zero-padded by 4
    W["g3.W"], W["g3.b"] = c11([-1.0], 0.0)                                         # vertical erosion of filled
    W["fG.W"] = (k * np.sqrt(W_G)).reshape(1, 1, 5, 5); W["fG.b"] = np.zeros(1)
    W["pM.W"] = fM; W["pC.W"] = fC; W["pE.W"] = fE; W["pG.W"] = fG
    W["pM.b"] = -0.5 * ((fM ** 2).sum(1) + (fC ** 2).sum(1) + (fE ** 2).sum(1) + (fG ** 2).sum(1)) + prior + line.sum(1) + dim.sum(1)
    W["pC.b"] = np.zeros(N); W["pE.b"] = np.zeros(N); W["pG.b"] = np.zeros(N)
    # ---- cue: the item is mostly dim.  q = clip((0.45 * area - bright area) / 50, 0, 1)
    w = np.zeros((1, 2, 28, 28)); w[0, 0] = 0.45 / 50.0; w[0, 1] = -1.0 / 50.0
    W["q1.W"] = w.reshape(1, -1); W["q1.b"] = np.zeros(1)
    W["q2.W"] = -np.ones((1, 1)); W["q2.b"] = np.ones(1)                            # relu -> 1 - q
    W["qN.W"] = -dim[:, 1].reshape(N, 1); W["qN.b"] = np.zeros(N)
    # ---- cue: a vertical line down the centre front (zip, button placket), brighter or darker than the cloth on both sides.
    # For polarity p, centre column j and row segment s: L = centre column - the column 2 to its left, R = same to the right (row means);
    # strength = min(relu(L), relu(R)); the cue is the fraction of segments in which one and the same column shows the line.
    r0, r1 = LINE_ROWS; seg = (r1 - r0) // LINE_SEG; cols = (12, 13, 14, 15)
    nd = 2 * LINE_SEG * len(cols)                    # detectors, index = (p * 4 + j) * LINE_SEG + s
    A = np.zeros((2 * nd, 28, 28))
    for p, sign in enumerate((1.0, -1.0)):
        for j, c0 in enumerate(cols):
            for sgi in range(LINE_SEG):
                d = (p * len(cols) + j) * LINE_SEG + sgi
                rows = slice(r0 + sgi * seg, r0 + (sgi + 1) * seg)
                for side, row in ((-1, 2 * d), (1, 2 * d + 1)):
                    A[row, rows, c0] += sign / seg; A[row, rows, c0 + 2 * side] -= sign / seg
    W["l1.W"] = A.reshape(2 * nd, 784); W["l1.b"] = np.zeros(2 * nd)
    # a line must lie on the cloth: veto it where any of the five columns it spans is background (the open gap between trouser
    # legs, the bright rim beside such a gap, a thin leg, strap or sleeve standing alone)
    Gt = np.zeros((2 * nd, 2, 28, 28))
    for p in range(2):
        for j, c0 in enumerate(cols):
            for sgi in range(LINE_SEG):
                d = (p * len(cols) + j) * LINE_SEG + sgi
                Gt[2 * d:2 * d + 2, 0, r0 + sgi * seg:r0 + (sgi + 1) * seg, c0 - 2:c0 + 3] = -LINE_VETO / seg
    W["lg.W"] = Gt.reshape(2 * nd, -1); W["lg.b"] = np.zeros(2 * nd)
    B = np.zeros((2 * nd, 2 * nd)); Cm = np.zeros((nd, 2 * nd))
    for d in range(nd):                             # detector d uses relu outputs a = 2d, b = 2d + 1
        B[2 * d, 2 * d] = 1.0                       # a
        B[2 * d + 1, 2 * d] = 1.0; B[2 * d + 1, 2 * d + 1] = -1.0      # a - b
        Cm[d, 2 * d] = 1.0 / SEG_TS; Cm[d, 2 * d + 1] = -1.0 / SEG_TS  # min(a, b) = a - relu(a - b)
    W["l2.W"] = B; W["l2.b"] = np.zeros(2 * nd)
    W["l3.W"] = Cm; W["l3.b"] = np.full(nd, -SEG_T0 / SEG_TS)           # relu -> (strength - SEG_T0) / SEG_TS, clipped below
    W["l4.W"] = -np.eye(nd); W["l4.b"] = np.ones(nd)                    # relu -> 1 - clipped strength
    W["l5.W"] = -np.eye(nd); W["l5.b"] = np.ones(nd)                    # clipped strength in [0, 1]
    W["c1.W"] = np.full((1, 1), 1.0 / LINE_TS); W["c1.b"] = np.array([-LINE_T0 / LINE_TS])
    W["c2.W"] = -np.ones((1, 1)); W["c2.b"] = np.ones(1)                            # relu -> 1 - c
    W["cN.W"] = -line[:, 1].reshape(N, 1); W["cN.b"] = np.zeros(N)
    cv = lambda name, inp, out, ks=1, st=1, pad=0: dict(name=name, op="conv", inputs=[inp], out=out, kernel=[ks, ks], stride=st, padding=pad)
    relu = lambda name, inp: dict(name=name, op="relu", inputs=[inp])
    mp = lambda name, inp, size: dict(name=name, op="maxpool", inputs=[inp], size=size, stride=[1, 1] if size == [3, 3] else size)
    dn = lambda name, inp, out: dict(name=name, op="dense", inputs=[inp], out=out)
    layers = [
        cv("m1", "image", 2), relu("m1r", "m1"), cv("m2", "m1r", 2), relu("m2r", "m2"), cv("m3", "m2r", 2),
        cv("fM", "m3", 1, 5, 2, 2), dn("pM", "fM", N),
        cv("cl0", "m3", 1, 1, 1, 2), mp("cl1", "cl0", [3, 3]), cv("cl2", "cl1", 1), mp("cl3", "cl2", [3, 3]), cv("cl4", "cl3", 1),
        cv("fC", "cl4", 1, 5, 2, 2), dn("pC", "fC", N),
        cv("e0", "cl4", 1, 1, 1, 1), mp("e1", "e0", [3, 3]), cv("e2", "e1", 1), cv("fE", "e2", 1, 5, 2, 2), dn("pE", "fE", N),
        cv("nm", "m3", 1), dict(name="t", op="add", inputs=["cl4", "nm"]),
        cv("g1", "t", 1, 1, 1, 4), dict(name="g2", op="maxpool", inputs=["g1"], size=[5, 1], stride=[1, 1]), cv("g3", "g2", 1),
        dict(name="g4", op="maxpool", inputs=["g3"], size=[5, 1], stride=[1, 1]),
        dict(name="g5", op="maxpool", inputs=["g4"], size=[1, 3], stride=[1, 1]),
        cv("fG", "g5", 1, 5, 2, 2), dn("pG", "fG", N),
        dn("q1", "m3", 1), relu("q1r", "q1"), dn("q2", "q1r", 1), relu("q2r", "q2"), dn("qN", "q2r", N),
        dict(name="n0", op="norm", inputs=["image"]),
        dn("l1", "n0", 2 * nd), dn("lg", "m2r", 2 * nd), dict(name="l1g", op="add", inputs=["l1", "lg"]), relu("l1r", "l1g"), dn("l2", "l1r", 2 * nd), relu("l2r", "l2"), dn("l3", "l2r", nd), relu("l3r", "l3"),
        dn("l4", "l3r", nd), relu("l4r", "l4"), dn("l5", "l4r", nd),
        dict(name="l6", op="reshape", inputs=["l5"], shape=[1, 2 * len(cols), LINE_SEG]),
        dict(name="l9", op="avgpool", inputs=["l6"], size=[1, LINE_SEG], stride=[1, LINE_SEG]),            # fraction of segments, per column and polarity
        dict(name="l10", op="maxpool", inputs=["l9"], size=[2 * len(cols), 1], stride=[2 * len(cols), 1]),  # best column, brighter or darker
        dn("c1", "l10", 1), relu("c1r", "c1"), dn("c2", "c1r", 1), relu("c2r", "c2"), dn("cN", "c2r", N),
        dict(name="score", op="add", inputs=["pM", "pC", "pE", "pG", "qN", "cN"]),
        dict(name="byclass", op="reshape", inputs=["score"], shape=[10, 1, K]),
        dict(name="best", op="maxpool", inputs=["byclass"], size=[1, K], stride=[1, K]),          # best-matching prototype of each class
        dict(name="scores", op="reshape", inputs=["best"], shape=[10]),
    ]
    json.dump({"layers": layers}, open(f"{out_dir}/network.json", "w"))
    np.savez(f"{out_dir}/weights.npz", **{n: np.asarray(v, np.float32) for n, v in W.items()})
    print("built N=%d in %.1fs" % (N, time.time() - t))


if __name__ == "__main__":
    import os
    os.makedirs(sys.argv[1], exist_ok=True)
    build(sys.argv[1], K=8000)
