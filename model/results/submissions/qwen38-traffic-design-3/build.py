"""Hand-designed traffic-sign network: weights drawn from general knowledge of German signs.
Usage: python3 build.py OUT_DIR [--viz VIZDIR]"""
import sys, json
import numpy as np

CH = ["red", "blue", "yellow", "white", "dark"]
NC = len(CH)


def grid(n):
    ys, xs = np.mgrid[0:n, 0:n].astype(float)
    return ys - (n - 1) / 2.0, xs - (n - 1) / 2.0


def soft(v):
    return np.clip(v, 0.0, 1.0)


def disc(y, x, r, cx=0.0, cy=0.0):
    return soft(r + 0.5 - np.hypot(x - cx, y - cy))


def ann(y, x, r0, r1, cx=0.0, cy=0.0):
    return disc(y, x, r1, cx, cy) - disc(y, x, r0, cx, cy)


def diamond(y, x, r):
    return soft(r + 0.5 - (np.abs(x) + np.abs(y)))


def dband(y, x, r0, r1):
    return diamond(y, x, r1) - diamond(y, x, r0)


def octagon(y, x, r):
    return soft(r + 0.5 - np.maximum(np.maximum(np.abs(x), np.abs(y)), (np.abs(x) + np.abs(y)) / 1.414))


def tri(y, x, hw, up=True, cy=0.6):
    """equilateral-ish triangle, apex up (or down), half width hw at base, soft edges"""
    yy = (y - cy) if up else (-(y - cy))
    a, b = hw * 1.05, hw * 0.80          # apex y=-a, base y=+b
    t = (yy + a) / (b + a)
    halfw = hw * t
    m = soft((halfw - np.abs(x)) / 0.7) * soft((b - yy) / 0.7) * soft((yy + a) / 0.7)
    return soft(m)


def tri_border(y, x, hw, w, up=True, cy=0.6):
    return soft(tri(y, x, hw, up, cy) - tri(y, x, hw - w * 1.6, up, cy - (0 if up else 0) + (w * 0.55 if up else -w * 0.55)))


def seg(y, x, p0, p1, w):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((x - x0) * dx + (y - y0) * dy) / L2, 0.0, 1.0)
    d = np.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
    return soft(w / 2 + 0.5 - d)


def _normv(v):
    vx, vy = float(v[0]), float(v[1])
    L = np.hypot(vx, vy) + 1e-9
    return vx / L, vy / L


def head(y, x, tip, dirv, hl, hw):
    ux, uy = _normv(dirv)
    bx, by = tip[0] - ux * hl, tip[1] - uy * hl
    t = (x - bx) * ux + (y - by) * uy
    lat = (x - bx) * (-uy) + (y - by) * ux
    halfw = hw * (1.0 - t / hl)
    return soft((halfw - np.abs(lat)) / 0.7) * soft((t + 0.4) / 0.7) * soft((hl - t) / 0.7)


def arrow(y, x, tail, tip, sw, hw, hl):
    ux, uy = _normv((tip[0] - tail[0], tip[1] - tail[1]))
    base = (tip[0] - ux * hl, tip[1] - uy * hl)
    return soft(seg(y, x, tail, base, sw) + head(y, x, tip, (ux, uy), hl, hw))


def bent_arrow(y, x, p0, p1, tip, sw, hw, hl):
    ux, uy = _normv((tip[0] - p1[0], tip[1] - p1[1]))
    base = (tip[0] - ux * hl, tip[1] - uy * hl)
    m = seg(y, x, p0, p1, sw) + seg(y, x, p1, base, sw) + disc(y, x, sw * 0.75, p1[0], p1[1])
    m = m + head(y, x, tip, (ux, uy), hl, hw)
    return soft(m)


def dilate(m, r=2):
    out = m
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out = np.maximum(out, np.roll(np.roll(m, dy, 0), dx, 1))
    return out


def blur(m, passes=1):
    k = np.array([0.25, 0.5, 0.25])
    for _ in range(passes):
        m = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 0, m)
        m = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, m)
    return m


DIGITS = {
    "0": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
    "5": ["#####", "#....", "#....", "####.", "....#", "....#", "####."],
    "6": [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", "..#..", "..#..", "..#.."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
}


def ascii_mask(y, x, pat, h, w, cx=0.0, cy=0.0, thick=1.0):
    """render a 5x7 ascii glyph centred at (cx,cy) with height h, width w"""
    pr = np.array([[0.0 if c == "." else 1.0 for c in row] for row in pat])
    ph, pw = pr.shape
    ys = np.round(np.linspace(0, ph, h + 1)).astype(int)
    xs = np.round(np.linspace(0, pw, w + 1)).astype(int)
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            r0, r1 = min(ys[i], ph - 1), max(ys[i + 1] - 1, ys[i])
            c0, c1 = min(xs[j], pw - 1), max(xs[j + 1] - 1, xs[j])
            out[i, j] = pr[r0:r1 + 1, c0:c1 + 1].mean()
    out = blur(out, 1)
    return out, h, w


def place(n, sub, cx=0.0, cy=0.0):
    """put a (h,w) array into an (n,n) canvas centred at (cx,cy) with soft edges"""
    h, w = sub.shape
    canvas = np.zeros((n, n))
    y, x = grid(n)
    ys = y - cy + (h - 1) / 2.0
    xs = x - cx + (w - 1) / 2.0
    iy = np.round(ys).astype(int)
    ix = np.round(xs).astype(int)
    ok = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
    canvas[ok] = sub[iy[ok], ix[ok]]
    return canvas


def number(n, digits, h=None):
    """draw a multi-digit number centred, returns dark mask"""
    k = len(digits)
    if h is None:
        h = {1: 12, 2: 11, 3: 9}[k]
    w = int(round(h * 0.62))
    gap = 1.0
    total = k * w + (k - 1) * gap
    out = np.zeros((n, n))
    x0 = -total / 2.0 + w / 2.0
    for i, d in enumerate(digits):
        sub, hh, ww = ascii_mask(None, None, DIGITS[d], h, w)
        out = np.maximum(out, place(n, sub, cx=x0 + i * (w + gap), cy=0.0))
    return out

# ---------------------------------------------------------------- templates
def blank(n):
    return np.zeros((NC, n, n))


def put(t, ch, m, w=1.0):
    t[CH.index(ch)] += w * m


def norml1(t):
    s = np.abs(t).sum()
    return t / (s + 1e-9)


def car(y, x, cx, cy, s=1.0):
    m = seg(y, x, (cx - 2.6 * s, cy + 0.6 * s), (cx + 2.6 * s, cy + 0.6 * s), 1.8 * s)
    m = m + seg(y, x, (cx - 1.3 * s, cy - 0.9 * s), (cx + 1.3 * s, cy - 0.9 * s), 1.7 * s)
    m = m + disc(y, x, 0.9 * s, cx - 1.6 * s, cy + 1.9 * s) + disc(y, x, 0.9 * s, cx + 1.6 * s, cy + 1.9 * s)
    return soft(m)


def truck(y, x, cx, cy, s=1.0):
    m = seg(y, x, (cx - 2.8 * s, cy - 0.4 * s), (cx + 0.6 * s, cy - 0.4 * s), 2.8 * s)
    m = m + seg(y, x, (cx + 1.6 * s, cy + 0.2 * s), (cx + 2.8 * s, cy + 0.2 * s), 2.0 * s)
    m = m + disc(y, x, 0.8 * s, cx - 1.8 * s, cy + 1.9 * s) + disc(y, x, 0.8 * s, cx + 1.8 * s, cy + 1.9 * s)
    return soft(m)


def ped(y, x, cx, cy, s=1.0):
    m = disc(y, x, 0.9 * s, cx + 0.4 * s, cy - 3.6 * s)
    m = m + seg(y, x, (cx + 0.4 * s, cy - 2.6 * s), (cx, cy + 0.4 * s), 1.4 * s)
    m = m + seg(y, x, (cx, cy + 0.4 * s), (cx - 1.8 * s, cy + 3.8 * s), 1.2 * s)
    m = m + seg(y, x, (cx, cy + 0.4 * s), (cx + 1.6 * s, cy + 3.8 * s), 1.2 * s)
    m = m + seg(y, x, (cx + 0.4 * s, cy - 2.0 * s), (cx - 1.2 * s, cy - 0.6 * s), 1.0 * s)
    m = m + seg(y, x, (cx + 0.4 * s, cy - 2.0 * s), (cx + 1.8 * s, cy - 1.0 * s), 1.0 * s)
    return soft(m)


def build_G():
    n = 26
    y, x = grid(n)
    T = {}
    t = blank(n); put(t, "red", ann(y, x, 9.3, 12.4)); T["g_ring"] = t
    t = blank(n); put(t, "red", disc(y, x, 12.4)); put(t, "white", seg(y, x, (-8, 0), (8, 0), 4.2), 0.9)
    put(t, "white", -soft(disc(y, x, 10.2) - seg(y, x, (-8, 0), (8, 0), 4.6)), 0.5); T["g_noentry"] = t
    t = blank(n); put(t, "red", tri_border(y, x, 12.2, 2.6, True)); T["g_tri"] = t
    t = blank(n); put(t, "red", tri_border(y, x, 12.2, 2.6, False)); T["g_tridown"] = t
    t = blank(n); put(t, "red", soft(octagon(y, x, 12.4) - octagon(y, x, 10.0))); T["g_oct"] = t
    t = blank(n); put(t, "yellow", diamond(y, x, 11.0)); put(t, "white", dband(y, x, 11.0, 12.6), 0.6); T["g_dthin"] = t
    t = blank(n); put(t, "yellow", diamond(y, x, 7.4)); put(t, "white", dband(y, x, 7.4, 12.6), 0.9); T["g_dthick"] = t
    t = blank(n); put(t, "blue", disc(y, x, 12.4)); T["g_blue"] = t
    t = blank(n); put(t, "white", disc(y, x, 10.8), 0.9); put(t, "dark", ann(y, x, 10.8, 12.5), 0.6)
    put(t, "red", -ann(y, x, 9.0, 12.6), 0.7); T["g_wdisk"] = t
    return T


def build_A():
    n = 16
    y, x = grid(n)
    T = {}
    def mk(name, whitem):
        t = blank(n)
        put(t, "white", whitem, 1.0)
        put(t, "blue", soft(dilate(whitem, 2) - whitem), 0.8)
        T[name] = t
    mk("a_up", arrow(y, x, (0, 6.4), (0, -6.6), 2.2, 3.4, 3.6))
    mk("a_bendR", bent_arrow(y, x, (-1.2, 6.4), (-1.2, 0.8), (6.2, 0.8), 2.2, 3.2, 3.4))
    mk("a_bendL", bent_arrow(y, x, (1.2, 6.4), (1.2, 0.8), (-6.2, 0.8), 2.2, 3.2, 3.4))
    upR = arrow(y, x, (0, 6.4), (0, -6.6), 2.0, 3.0, 3.2)
    upR = soft(upR + seg(y, x, (0, 1.2), (3.6, 1.2), 1.8) + head(y, x, (6.4, 1.2), (1, 0), 3.0, 2.8))
    mk("a_upR", upR)
    upL = arrow(y, x, (0, 6.4), (0, -6.6), 2.0, 3.0, 3.2)
    upL = soft(upL + seg(y, x, (0, 1.2), (-3.6, 1.2), 1.8) + head(y, x, (-6.4, 1.2), (-1, 0), 3.0, 2.8))
    mk("a_upL", upL)
    mk("a_diagDR", arrow(y, x, (-4.6, -4.6), (4.9, 4.9), 2.2, 3.2, 3.4))
    mk("a_diagDL", arrow(y, x, (4.6, -4.6), (-4.9, 4.9), 2.2, 3.2, 3.4))
    r = 4.3
    m = np.zeros((n, n))
    for a0 in (90, 210, 330):
        ang = np.arctan2(-(y), x) * 180 / np.pi
        ang = np.mod(ang - a0, 360)
        arc = soft((ang) / 6) * soft((78 - ang) / 6) * ann(y, x, r - 1.0, r + 1.0)
        m = m + arc
        a1 = np.deg2rad(a0 + 78)
        tip = (r * np.cos(a1), -r * np.sin(a1))
        tang = (-np.sin(a1), -np.cos(a1))
        m = m + head(y, x, tip, tang, 2.6, 2.4)
    mk("a_round", soft(m))
    return T


def build_D():
    n = 18
    y, x = grid(n)
    T = {}
    for name, digs, h in [("n20", "20", 11), ("n30", "30", 11), ("n50", "50", 11), ("n60", "60", 11),
                          ("n70", "70", 11), ("n80", "80", 11), ("n100", "100", 9), ("n120", "120", 9)]:
        dm = number(n, digs, h)
        t = blank(n)
        put(t, "dark", dm, 1.0)
        put(t, "white", soft(dilate(dm, 2) - dm), 0.8)
        T[name] = t
    return T


def build_S():
    n = 16
    y, x = grid(n)
    T = {}
    def mk(name, darkm, redm=None, wht=0.8, ctxr=7.4):
        t = blank(n)
        put(t, "dark", darkm, 1.0)
        if redm is not None:
            put(t, "red", redm, 1.0)
        both = soft(darkm + (redm if redm is not None else 0))
        ctx = soft(dilate(both, 2) - both)
        put(t, "white", ctx, wht)
        T[name] = t
    cl, cr = car(y, x, -3.7, 0, 0.85), car(y, x, 3.7, 0, 0.85)
    mk("s_carcAR", cl, redm=cr)
    mk("s_truckcar", truck(y, x, -3.7, 0, 0.85), redm=car(y, x, 3.7, 0, 0.85))
    mk("s_truck", truck(y, x, 0, 0, 1.15))
    slash = seg(y, x, (-6.4, 6.4), (6.4, -6.4), 1.7)
    mk("s_carslash", soft(car(y, x, -0.5, 0.5, 0.95) + slash))
    mk("s_truckslash", soft(truck(y, x, -0.5, 0.5, 0.95) + slash))
    mk("s_slash", seg(y, x, (-6.6, 6.6), (6.6, -6.6), 2.8))
    t = blank(n)
    put(t, "white", disc(y, x, 6.5), 1.0)
    put(t, "dark", -disc(y, x, 6.5), 1.0)
    T["s_empty"] = t
    mk("w_excl", soft(seg(y, x, (0, -4.4), (0, 1.0), 1.7) + disc(y, x, 1.1, 0, 3.6)))
    bar = seg(y, x, (-7.6, 0), (7.6, 0), 4.0)
    t = blank(n); put(t, "white", bar, 1.0); put(t, "red", soft(dilate(bar, 2) - bar), 0.6); T["s_bar"] = t
    let = sum(disc(y, x, 1.4, lx, 0) for lx in (-4.5, -1.5, 1.5, 4.5))
    gaps = sum(disc(y, x, 0.8, gx, 0) for gx in (-3.0, 0.0, 3.0))
    t = blank(n); put(t, "white", soft(let), 1.0); put(t, "white", -gaps, 0.7)
    put(t, "red", soft(dilate(let, 2) - let), 0.6); T["s_stop"] = t
    mk("w_bendL", bent_arrow(y, x, (1.4, 5.0), (1.4, 0.2), (-4.8, 0.2), 1.7, 2.6, 2.8))
    mk("w_bendR", bent_arrow(y, x, (-1.4, 5.0), (-1.4, 0.2), (4.8, 0.2), 1.7, 2.6, 2.8))
    dbl = seg(y, x, (0.9, 5.2), (0.9, 1.0), 1.6) + seg(y, x, (0.9, 1.0), (-0.9, -1.0), 1.6)
    dbl = dbl + seg(y, x, (-0.9, -1.0), (-0.9, -3.2), 1.6) + head(y, x, (-0.9, -5.4), (0, -1), 2.2, 2.4)
    mk("w_dbl", soft(dbl))
    bump = seg(y, x, (-4.6, 2.8), (4.6, 2.8), 1.2)
    bump = bump + disc(y, x, 2.0, -2.3, 2.8) * soft((2.8 - y) / 0.7) + disc(y, x, 2.0, 2.3, 2.8) * soft((2.8 - y) / 0.7)
    mk("w_bump", soft(bump))
    skid = car(y, x, 1.2, -1.2, 0.8)
    skid = skid + seg(y, x, (-4.0, 1.6), (-1.6, 1.0), 1.0) + seg(y, x, (-1.6, 1.0), (-3.0, 2.8), 1.0)
    skid = skid + seg(y, x, (-0.6, 3.0), (1.8, 2.4), 1.0) + seg(y, x, (1.8, 2.4), (0.4, 4.2), 1.0)
    mk("w_skid", soft(skid))
    nar = seg(y, x, (-1.8, -4.4), (-1.8, 4.4), 1.4) + seg(y, x, (1.8, -4.4), (1.8, -0.2), 1.4)
    nar = nar + seg(y, x, (1.8, -0.2), (0.4, 4.4), 1.4)
    mk("w_narrow", soft(nar))
    wk = disc(y, x, 0.9, -1.4, -3.4) + seg(y, x, (-1.4, -2.4), (-1.0, 0.6), 1.4)
    wk = wk + seg(y, x, (-1.0, 0.6), (-2.2, 3.6), 1.2) + seg(y, x, (-1.0, 0.6), (0.2, 3.6), 1.2)
    wk = wk + seg(y, x, (-1.4, -1.6), (1.2, -0.4), 1.0) + seg(y, x, (1.2, -0.4), (2.6, 2.2), 0.9)
    wk = wk + disc(y, x, 1.9, 3.2, 3.4) * soft((y - 1.6) / 0.7)
    mk("w_work", soft(wk))
    li = seg(y, x, (0, -4.2), (0, 3.4), 2.8)
    t = blank(n)
    put(t, "dark", li, 1.0)
    put(t, "red", disc(y, x, 1.0, 0, -2.6), 1.0)
    put(t, "yellow", disc(y, x, 1.0, 0, -0.2), 1.0)
    put(t, "white", soft(dilate(li, 2) - li), 0.8)
    T["w_light"] = t
    mk("w_ped", ped(y, x, 0, 0, 1.0))
    mk("w_child", soft(ped(y, x, -2.3, 0.8, 0.78) + ped(y, x, 2.2, 1.0, 0.68)))
    bk = ann(y, x, 1.0, 2.0, -2.9, 2.5) + ann(y, x, 1.0, 2.0, 2.9, 2.5)
    bk = bk + seg(y, x, (-2.9, 2.5), (-0.6, -0.2), 0.9) + seg(y, x, (-0.6, -0.2), (2.9, 2.5), 0.9)
    bk = bk + seg(y, x, (-0.6, -0.2), (-1.7, -1.7), 0.9)
    bk = bk + seg(y, x, (-1.3, -1.9), (0.6, -3.4), 1.1) + disc(y, x, 0.8, 1.1, -4.1)
    mk("w_bike", soft(bk))
    sn = seg(y, x, (-4.4, 0), (4.4, 0), 1.1) + seg(y, x, (-2.2, -3.8), (2.2, 3.8), 1.1)
    sn = sn + seg(y, x, (-2.2, 3.8), (2.2, -3.8), 1.1)
    mk("w_snow", soft(sn))
    dr = seg(y, x, (-2.6, 0.8), (1.6, -0.6), 1.5) + seg(y, x, (1.6, -0.6), (2.8, -2.4), 1.2)
    dr = dr + seg(y, x, (2.8, -2.4), (3.9, -2.8), 1.0)
    dr = dr + seg(y, x, (3.0, -2.8), (2.6, -4.4), 0.8) + seg(y, x, (3.4, -2.8), (3.9, -4.4), 0.8)
    dr = dr + seg(y, x, (-2.0, 1.0), (-3.4, 3.4), 1.0) + seg(y, x, (-1.0, 1.0), (-1.6, 3.6), 1.0)
    dr = dr + seg(y, x, (0.8, 0.2), (1.8, 3.0), 1.0) + seg(y, x, (1.4, 0.0), (2.8, 2.6), 1.0)
    mk("w_deer", soft(dr))
    return T


TG, TA, TD, TS = build_G(), build_A(), build_D(), build_S()
BANKS = [("g", TG, 26, 2, 0.70), ("a", TA, 16, 1, 0.80), ("d", TD, 18, 1, 0.70), ("s", TS, 16, 1, 0.70)]
for _b, _T, _n, _s, _t in BANKS:
    for _k in _T:
        _T[_k] = norml1(_T[_k])

# ---------------------------------------------------------------- network
RULE = {
    0:  [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("d", "n20", 2.0)],
    1:  [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("d", "n30", 2.0)],
    2:  [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("d", "n50", 2.0)],
    3:  [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("d", "n60", 2.0)],
    4:  [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("d", "n70", 2.0)],
    5:  [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("d", "n80", 2.0)],
    6:  [("g", "g_wdisk", 2.0), ("g", "g_ring", -1.0), ("d", "n80", 1.2), ("s", "s_slash", 1.4)],
    7:  [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("d", "n100", 2.0)],
    8:  [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("d", "n120", 2.0)],
    9:  [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("s", "s_carcAR", 2.0)],
    10: [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("s", "s_truckcar", 2.0)],
    11: [("g", "g_dthin", 2.2), ("g", "g_dthick", -1.2)],
    12: [("g", "g_dthick", 2.2), ("g", "g_dthin", -1.2)],
    13: [("g", "g_tridown", 2.2)],
    14: [("g", "g_oct", 1.6), ("s", "s_stop", 1.6)],
    15: [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("s", "s_empty", 1.8)],
    16: [("g", "g_ring", 2.0), ("g", "g_wdisk", -1.0), ("s", "s_truck", 2.0)],
    17: [("g", "g_noentry", 1.6), ("s", "s_bar", 1.6)],
    18: [("g", "g_tri", 2.0), ("s", "w_excl", 2.0)],
    19: [("g", "g_tri", 2.0), ("s", "w_bendL", 2.0)],
    20: [("g", "g_tri", 2.0), ("s", "w_bendR", 2.0)],
    21: [("g", "g_tri", 2.0), ("s", "w_dbl", 2.0)],
    22: [("g", "g_tri", 2.0), ("s", "w_bump", 2.0)],
    23: [("g", "g_tri", 2.0), ("s", "w_skid", 2.0)],
    24: [("g", "g_tri", 2.0), ("s", "w_narrow", 2.0)],
    25: [("g", "g_tri", 2.0), ("s", "w_work", 2.0)],
    26: [("g", "g_tri", 2.0), ("s", "w_light", 2.0)],
    27: [("g", "g_tri", 2.0), ("s", "w_ped", 2.0)],
    28: [("g", "g_tri", 2.0), ("s", "w_child", 2.0)],
    29: [("g", "g_tri", 2.0), ("s", "w_bike", 2.0)],
    30: [("g", "g_tri", 2.0), ("s", "w_snow", 2.0)],
    31: [("g", "g_tri", 2.0), ("s", "w_deer", 2.0)],
    32: [("g", "g_wdisk", 2.0), ("g", "g_ring", -1.0), ("s", "s_slash", 1.6), ("d", "n80", -1.3), ("s", "s_carslash", -1.3), ("s", "s_truckslash", -1.3)],
    33: [("g", "g_blue", 2.0), ("a", "a_bendR", 2.0)],
    34: [("g", "g_blue", 2.0), ("a", "a_bendL", 2.0)],
    35: [("g", "g_blue", 2.0), ("a", "a_up", 2.0)],
    36: [("g", "g_blue", 2.0), ("a", "a_upR", 2.0)],
    37: [("g", "g_blue", 2.0), ("a", "a_upL", 2.0)],
    38: [("g", "g_blue", 2.0), ("a", "a_diagDR", 2.0)],
    39: [("g", "g_blue", 2.0), ("a", "a_diagDL", 2.0)],
    40: [("g", "g_blue", 2.0), ("a", "a_round", 2.0)],
    41: [("g", "g_wdisk", 2.0), ("g", "g_ring", -1.0), ("s", "s_carslash", 2.0)],
    42: [("g", "g_wdisk", 2.0), ("g", "g_ring", -1.0), ("s", "s_truckslash", 2.0)],
}

BMAP = {"g": 0, "a": 1, "d": 2, "s": 3}


def build_network():
    layers = []
    W = {}
    # colour / luminance front end
    w = np.zeros((5, 3, 3, 3))
    w[0] = 1.0 / 27.0
    w[1, 0] = 1.0 / 9.0; w[1, 1] = -1.0 / 18.0; w[1, 2] = -1.0 / 18.0
    w[2, 2] = 1.0 / 9.0; w[2, 0] = -1.0 / 18.0; w[2, 1] = -1.0 / 18.0
    w[3, 0] = 1.0 / 18.0; w[3, 1] = 1.0 / 18.0; w[3, 2] = -1.0 / 9.0
    w[4, 1] = 1.0 / 9.0; w[4, 0] = -1.0 / 18.0; w[4, 2] = -1.0 / 18.0
    W["c1.W"], W["c1.b"] = w, np.zeros(5)
    layers.append({"name": "c1", "op": "conv", "inputs": ["image"], "out": 5, "kernel": [3, 3], "stride": 1, "padding": 1})
    layers.append({"name": "n1", "op": "norm", "inputs": ["c1"]})
    w = np.zeros((5, 5, 1, 1))
    w[0, :, 0, 0] = [0, 1.2, 0, 0, 0]
    w[1, :, 0, 0] = [0, 0, 1.2, 0, 0]
    w[2, :, 0, 0] = [0, 0, 0, 1.2, 0]
    w[3, :, 0, 0] = [1.0, -0.5, -0.5, -0.5, -0.5]
    w[4, :, 0, 0] = [-1.0, 0, 0, 0, 0]
    W["c2.W"], W["c2.b"] = w, np.array([-0.90, -0.75, -1.60, -0.35, -0.10])
    layers.append({"name": "c2", "op": "conv", "inputs": ["n1"], "out": 5, "kernel": [1, 1], "stride": 1, "padding": 0})
    layers.append({"name": "r2", "op": "relu", "inputs": ["c2"]})
    k = np.outer([0.25, 0.5, 0.25], [0.25, 0.5, 0.25])
    w = np.zeros((5, 5, 3, 3))
    for i in range(5):
        w[i, i] = k
    W["c3.W"], W["c3.b"] = w, np.zeros(5)
    layers.append({"name": "c3", "op": "conv", "inputs": ["r2"], "out": 5, "kernel": [3, 3], "stride": 1, "padding": 1})
    # template banks
    pool_sizes = {2: 4, 1: None}
    for bch, T, n, st, thr in BANKS:
        names = list(T.keys())
        w = np.stack([T[kk] for kk in names]).astype(np.float64)
        W[f"t{bch}.W"], W[f"t{bch}.b"] = w, np.full(len(names), -thr)
        layers.append({"name": f"t{bch}", "op": "conv", "inputs": ["c3"], "out": len(names),
                       "kernel": [n, n], "stride": st, "padding": 0})
        layers.append({"name": f"r{bch}", "op": "relu", "inputs": [f"t{bch}"]})
        if st == 2:
            ho = (32 - n) // st + 1
            layers.append({"name": f"p{bch}", "op": "maxpool", "inputs": [f"r{bch}"], "size": [ho, ho], "stride": [ho, ho]})
        else:
            ho = 32 - n + 1
            layers.append({"name": f"p{bch}", "op": "maxpool", "inputs": [f"r{bch}"], "size": [ho, ho], "stride": [ho, ho]})
    # rule table read-out
    idx = {bch: {kk: i for i, kk in enumerate(T.keys())} for bch, T, n, st, thr in BANKS}
    dens = []
    for bch in ["g", "a", "d", "s"]:
        T = {"g": TG, "a": TA, "d": TD, "s": TS}[bch]
        wd = np.zeros((43, len(T)))
        for cls, terms in RULE.items():
            for b2, kk, wt in terms:
                if b2 == bch:
                    wd[cls, idx[bch][kk]] = wt
        W[f"d{bch}.W"], W[f"d{bch}.b"] = wd, np.zeros(43)
        layers.append({"name": f"d{bch}", "op": "dense", "inputs": [f"p{bch}"], "out": 43})
        dens.append(f"d{bch}")
    layers.append({"name": "out", "op": "add", "inputs": dens})
    return {"layers": layers}, W


def viz(path):
    import zlib, struct
    def write_png(p, arr):
        h, w, _ = arr.shape
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
        def chunk(tag, data):
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
        open(p, "wb").write(png)
    for bch, T, n, st, thr in BANKS:
        names = list(T.keys())
        cols = 8
        rows = (len(names) + cols - 1) // cols
        sc = 5
        sheet = np.full((rows * (n * sc + 4) + 4, cols * (n * sc + 4) + 4, 3), 200, np.uint8)
        for i, kk in enumerate(names):
            t = T[kk] / (np.abs(T[kk]).max() + 1e-12)
            r, c = divmod(i, cols)
            rgb = np.zeros((n, n, 3))
            rgb[:, :, 0] = t[0] + 0.9 * t[3] + 0.95 * t[2]
            rgb[:, :, 1] = 0.9 * t[3] + 0.75 * t[2]
            rgb[:, :, 2] = t[1] + 0.9 * t[3]
            rgb *= (1.0 - 0.85 * np.clip(t[4], 0, 1))[:, :, None]
            neg = np.clip(-t[4], 0, 1)
            rgb[:, :, 0] += neg * 0.6
            rgb[:, :, 2] += neg * 0.6
            big = np.kron(np.clip(rgb, 0, 1) * 255, np.ones((sc, sc, 1))).astype(np.uint8)
            y0 = 4 + r * (n * sc + 4)
            x0 = 4 + c * (n * sc + 4)
            sheet[y0:y0 + n * sc, x0:x0 + n * sc] = big
        write_png(f"{path}/bank_{bch}.png", sheet)
        print("wrote", bch, names)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    if "--viz" in sys.argv:
        viz(sys.argv[sys.argv.index("--viz") + 1])
        return
    import os
    os.makedirs(out, exist_ok=True)
    net, W = build_network()
    with open(f"{out}/network.json", "w") as f:
        json.dump(net, f)
    np.savez(f"{out}/weights.npz", **{k: v.astype(np.float32) for k, v in W.items()})
    tot = sum(v.size for v in W.values())
    print("weights:", tot)


main()
