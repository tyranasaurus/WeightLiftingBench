# build.py -- writes network.json and weights.npz for a hand-designed
# traffic-sign classifier.  All weights are drawn from general knowledge of
# German road-sign geometry (rings, triangles, octagons, digits, arrows...)
# rendered programmatically; no files are read.
import sys, json
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ---------------- geometry framework ----------------
K  = 28            # template kernel size (pixels)
SS = 8             # supersampling factor for drawing
SCALES = [10.5, 12.0, 13.5]          # sign outer radius, kernel pixels
TRI_ROTS = [-10.0, 0.0, 10.0]        # rotation variants for tilt-sensitive shapes
KNORM = 3.0        # template kernels are scaled to this Frobenius norm

CH_L, CH_RG, CH_BY, CH_D5, CH_B5, CH_D9, CH_B9 = range(7)
# relative strength of each feature channel inside a template
CHW = {CH_L: 0.9, CH_RG: 1.4, CH_BY: 1.4, CH_D5: 2.0, CH_B5: 2.0, CH_D9: 0.9, CH_B9: 0.9}

_gy, _gx = np.mgrid[0:K*SS, 0:K*SS]
CC = (K - 1) / 2.0
GY0 = (_gy + 0.5) / SS - CC     # coordinates centred on the sign (0,0 = kernel centre)
GX0 = (_gx + 0.5) / SS - CC

def down(m):
    return m.reshape(K, SS, K, SS).mean(axis=(1, 3)).astype(np.float64)

def rot_coords(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    x = GX0 * c + GY0 * s
    y = -GX0 * s + GY0 * c
    return y, x

def M_disc(Y, X, cx, cy, r):
    return (X - cx)**2 + (Y - cy)**2 <= r*r
def M_ring(Y, X, cx, cy, r0, r1):
    d = (X - cx)**2 + (Y - cy)**2
    return (d >= r0*r0) & (d <= r1*r1)
def M_ellring(Y, X, cx, cy, rx, ry, t):
    d1 = ((X - cx)/rx)**2 + ((Y - cy)/ry)**2
    rx2, ry2 = max(rx - t, 1e-3), max(ry - t, 1e-3)
    d2 = ((X - cx)/rx2)**2 + ((Y - cy)/ry2)**2
    return (d1 <= 1.0) & (d2 >= 1.0)
def M_seg(Y, X, x0, y0, x1, y1, w):
    vx, vy = x1 - x0, y1 - y0
    L2 = vx*vx + vy*vy + 1e-12
    t = np.clip(((X - x0)*vx + (Y - y0)*vy) / L2, 0, 1)
    dx = X - (x0 + t*vx); dy = Y - (y0 + t*vy)
    return dx*dx + dy*dy <= (w/2.0)**2
def M_poly(Y, X, pts):
    inside = np.zeros(Y.shape, bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i+1) % n]
        cond = (y1 > Y) != (y2 > Y)
        xint = (x2 - x1) * (Y - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= cond & (X < xint)
    return inside
def M_tri_edge(Y, X, v0, v1, v2, w):
    m = M_seg(Y, X, v0[0], v0[1], v1[0], v1[1], w)
    m |= M_seg(Y, X, v1[0], v1[1], v2[0], v2[1], w)
    m |= M_seg(Y, X, v2[0], v2[1], v0[0], v0[1], w)
    return m
def M_arc(Y, X, cx, cy, r0, r1, a0, a1):
    ang = np.degrees(np.arctan2(Y - cy, X - cx)) % 360.0
    insec = (ang >= a0) & (ang <= a1) if a1 >= a0 else (ang >= a0) | (ang <= a1)
    return M_ring(Y, X, cx, cy, r0, r1) & insec

def dilate(m, it=2):
    m = m.copy()
    for _ in range(it):
        m = np.maximum(m, np.roll(m, 1, 0)); m = np.maximum(m, np.roll(m, -1, 0))
        m = np.maximum(m, np.roll(m, 1, 1)); m = np.maximum(m, np.roll(m, -1, 1))
    return m

def pl(Y, X, pts, w):
    m = np.zeros(Y.shape, bool)
    for i in range(len(pts)-1):
        m |= M_seg(Y, X, pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], w)
    return m

# ---------------- digit font (unit box x[0,0.62] y[0,1]) ------------------
DIG = {
 '0': [('ell', (0.31, 0.50, 0.26, 0.46, 0.13))],
 '1': [('seg', (0.10, 0.06, 0.10, 1.00, 0.15)), ('seg', (0.10, 0.12, 0.36, 0.02, 0.12))],
 '2': [('pl', ([(0.05,0.12),(0.40,0.06),(0.42,0.35),(0.05,0.92),(0.45,0.92)], 0.13))],
 '3': [('pl', ([(0.05,0.10),(0.38,0.06),(0.42,0.28),(0.18,0.50),(0.42,0.68),(0.38,0.92),(0.05,0.94)], 0.13))],
 '5': [('pl', ([(0.42,0.06),(0.06,0.08),(0.05,0.48),(0.36,0.50),(0.42,0.72),(0.25,0.94),(0.04,0.90)], 0.13))],
 '6': [('pl', ([(0.40,0.07),(0.10,0.32),(0.06,0.72),(0.20,0.94),(0.40,0.86),(0.40,0.58),(0.14,0.54)], 0.13))],
 '7': [('pl', ([(0.05,0.08),(0.44,0.08),(0.16,0.96)], 0.13))],
 '8': [('ell', (0.31, 0.27, 0.20, 0.22, 0.10)), ('ell', (0.31, 0.74, 0.25, 0.24, 0.10))],
}
def digit_mask(Y, X, d, cx, cy, h, sw=1.0):
    m = np.zeros(Y.shape, bool)
    for kind, a in DIG[d]:
        if kind == 'ell':
            ex, ey, rx, ry, t = a
            m |= M_ellring(Y, X, cx + (ex-0.31)*h, cy + (ey-0.5)*h, rx*h, ry*h, t*h*sw)
        elif kind == 'seg':
            x0, y0, x1, y1, w = a
            m |= M_seg(Y, X, cx + (x0-0.31)*h, cy + (y0-0.5)*h,
                             cx + (x1-0.31)*h, cy + (y1-0.5)*h, w*h*sw)
        else:
            pts, w = a
            q = [(cx + (px-0.31)*h, cy + (py-0.5)*h) for px, py in pts]
            m |= pl(Y, X, q, w*h*sw)
    return m

# ---------------- sign frames --------------------------------------------
TRI_V  = [(0.0, -1.02), (-0.78, 0.97), (0.78, 0.97)]
DTRI_V = [(0.0, 1.02), (-0.78, -0.97), (0.78, -0.97)]

def tri_scale(v, R, s):
    gx = sum(p[0] for p in v) / 3.0; gy = sum(p[1] for p in v) / 3.0
    return [((gx + (p[0]-gx)*s)*R, (gy + (p[1]-gy)*s)*R) for p in v]


# ---------------- content helpers ----------------------------------------
# Feature maps (after relu): ch0 L+8, ch1 RG+8, ch2 BY+8 (level, planes must
# be zero-sum so the +8 cancels), ch3 dark-thin, ch4 bright-thin, ch5 dark-
# thick, ch6 bright-thick (all >= 0, so negative weights are true penalties).
def content_dark_planes(inner_ss, cmask_ss, wd=1.5, wk=0.7, wp=0.2):
    """dark pictogram: + on strokes; penalise ANY content (either polarity)
    elsewhere inside -- relu'd maps make both penalties truly negative."""
    inn = down(inner_ss); cm = down(cmask_ss)
    pen = ((inn > 0.45) & ~(dilate(cm, 2) > 0.25)).astype(np.float64)
    return [(CH_D5,  wd, cm, False), (CH_D9,  wk, cm, False),
            (CH_D5, -wp, pen, False), (CH_D9, -wp*0.5, pen, False),
            (CH_B5, -wp*0.6, pen, False), (CH_B9, -wp*0.3, pen, False)]

def content_bright_planes(inner_ss, cmask_ss, wb=1.5, wk=0.7, wp=0.2):
    inn = down(inner_ss); cm = down(cmask_ss)
    pen = ((inn > 0.45) & ~(dilate(cm, 2) > 0.25)).astype(np.float64)
    return [(CH_B5,  wb, cm, False), (CH_B9,  wk, cm, False),
            (CH_B5, -wp, pen, False), (CH_B9, -wp*0.5, pen, False),
            (CH_D5, -wp*0.6, pen, False), (CH_D9, -wp*0.3, pen, False)]

def empty_planes(inner_ss):
    inn = down(inner_ss)
    return [(CH_D5, -1.2, inn, False), (CH_D9, -0.6, inn, False),
            (CH_B5, -0.9, inn, False), (CH_B9, -0.45, inn, False)]

def pic_car(Y, X, R, cx, cy, s):
    m = M_poly(Y, X, [(cx-0.21*s, cy+0.05*s), (cx+0.21*s, cy+0.05*s),
                      (cx+0.17*s, cy-0.06*s), (cx+0.09*s, cy-0.14*s), (cx-0.09*s, cy-0.14*s),
                      (cx-0.17*s, cy-0.06*s)])
    m |= M_disc(Y, X, cx-0.12*s, cy+0.11*s, 0.07*s) | M_disc(Y, X, cx+0.12*s, cy+0.11*s, 0.07*s)
    return m

def pic_truck(Y, X, R, cx, cy, s):
    m = M_poly(Y, X, [(cx-0.26*s, cy+0.05*s), (cx+0.26*s, cy+0.05*s), (cx+0.26*s, cy-0.26*s),
                      (cx-0.02*s, cy-0.26*s), (cx-0.02*s, cy-0.16*s), (cx-0.14*s, cy-0.16*s),
                      (cx-0.14*s, cy-0.02*s), (cx-0.26*s, cy-0.02*s)])
    m |= M_disc(Y, X, cx-0.17*s, cy+0.10*s, 0.07*s) | M_disc(Y, X, cx+0.16*s, cy+0.10*s, 0.07*s)
    return m

def halfdisc(Y, X, cx, cy, r, up=True):
    m = M_disc(Y, X, cx, cy, r)
    return m & (Y < cy) if up else m & (Y > cy)

def arrow(Y, X, pts, w, heads):
    m = pl(Y, X, pts, w)
    for base, tip, hw in heads:
        d = np.array(tip) - np.array(base); L = np.hypot(*d) + 1e-9
        p = np.array([-d[1], d[0]]) / L
        b = np.array(base)
        m |= M_poly(Y, X, [tuple(tip), tuple(b + p*hw), tuple(b - p*hw)])
    return m

def hatch_stripes(Y, X, R):
    d = np.array([1.0, -1.0]) / np.sqrt(2.0); n = np.array([1.0, 1.0]) / np.sqrt(2.0)
    m = np.zeros(Y.shape, bool)
    for o in (-0.48, -0.16, 0.16, 0.48):
        c = n * (o * R)
        m |= M_seg(Y, X, c[0]-0.95*R*d[0], c[1]-0.95*R*d[1], c[0]+0.95*R*d[0], c[1]+0.95*R*d[1], 0.14*R)
    return m & M_disc(Y, X, 0, 0, 0.76*R)

# ---------------- pictograms (units of R, centre (0,0)) -------------------
def mk_digits(txt):
    def f(Y, X, R):
        if len(txt) == 2:
            h = 0.74*R; xs = (-0.28*R, 0.28*R); y = 0.05*R; sw = 1.25
        else:
            h = 0.62*R; xs = (-0.50*R, 0.0, 0.50*R); y = 0.05*R; sw = 1.35
        m = np.zeros(Y.shape, bool)
        for ch, xc in zip(txt, xs):
            m |= digit_mask(Y, X, ch, xc, y, h, sw)
        return m
    return f

def mk_cars():
    def f(Y, X, R):
        return pic_car(Y, X, R, -0.26*R, 0.10*R, 0.92*R) | pic_car(Y, X, R, 0.26*R, 0.10*R, 0.92*R)
    return f

def mk_cartruck():
    def f(Y, X, R):
        return pic_car(Y, X, R, -0.30*R, 0.14*R, 0.85*R) | pic_truck(Y, X, R, 0.22*R, 0.02*R, 0.95*R)
    return f

def mk_truck():
    def f(Y, X, R):
        return pic_truck(Y, X, R, 0.0, 0.06*R, 1.35*R)
    return f

def mk_x():
    def f(Y, X, R):
        return (M_seg(Y, X, -0.45*R, -0.40*R, 0.42*R, 0.38*R, 0.20*R) |
                M_seg(Y, X, 0.45*R, -0.40*R, -0.42*R, 0.38*R, 0.20*R))
    return f

def mk_excl():
    def f(Y, X, R):
        return M_seg(Y, X, -0.02*R, -0.32*R, -0.02*R, -0.04*R, 0.20*R) | M_disc(Y, X, -0.02*R, 0.30*R, 0.12*R)
    return f

def mk_bend(left=True):
    def f(Y, X, R):
        s = -1.0 if left else 1.0
        return (M_seg(Y, X, s*0.32*R, 0.05*R, -s*0.12*R, 0.05*R, 0.20*R) |
                M_seg(Y, X, -s*0.12*R, 0.05*R, -s*0.12*R, 0.44*R, 0.20*R))
    return f

def mk_dbend():
    def f(Y, X, R):
        return pl(Y, X, [(-0.26*R, -0.26*R), (-0.26*R, 0.02*R), (0.08*R, 0.14*R), (0.08*R, 0.42*R)], 0.19*R)
    return f

def mk_bumps():
    def f(Y, X, R):
        return halfdisc(Y, X, -0.36*R, 0.42*R, 0.20*R) | halfdisc(Y, X, 0.13*R, 0.42*R, 0.20*R)
    return f

def mk_slip():
    def f(Y, X, R):
        m = pic_car(Y, X, R, 0.0, -0.24*R, 0.75*R)
        m |= pl(Y, X, [(-0.16*R, 0.0), (-0.28*R, 0.13*R), (-0.12*R, 0.26*R), (-0.24*R, 0.40*R)], 0.09*R)
        m |= pl(Y, X, [(0.18*R, 0.0), (0.06*R, 0.13*R), (0.20*R, 0.26*R), (0.08*R, 0.40*R)], 0.09*R)
        return m
    return f

def mk_narrow():
    def f(Y, X, R):
        return (M_seg(Y, X, -0.26*R, -0.15*R, -0.26*R, 0.48*R, 0.12*R) |
                M_seg(Y, X, 0.30*R, -0.15*R, 0.22*R, 0.48*R, 0.12*R))
    return f

def mk_works():
    def f(Y, X, R):
        m = M_disc(Y, X, -0.28*R, -0.18*R, 0.09*R)
        m |= M_seg(Y, X, -0.26*R, -0.08*R, -0.18*R, 0.15*R, 0.13*R)
        m |= M_seg(Y, X, -0.18*R, 0.15*R, -0.30*R, 0.45*R, 0.10*R)
        m |= M_seg(Y, X, -0.18*R, 0.15*R, -0.06*R, 0.45*R, 0.10*R)
        m |= M_seg(Y, X, -0.24*R, 0.0, 0.06*R, 0.08*R, 0.09*R)
        m |= halfdisc(Y, X, 0.12*R, 0.42*R, 0.22*R)
        return m
    return f

def mk_lights():
    def f(Y, X, R):
        outer = M_poly(Y, X, [(-0.26*R, -0.62*R), (0.18*R, -0.62*R), (0.18*R, 0.44*R), (-0.26*R, 0.44*R)])
        inner = M_poly(Y, X, [(-0.17*R, -0.53*R), (0.09*R, -0.53*R), (0.09*R, 0.35*R), (-0.17*R, 0.35*R)])
        m = outer & ~inner
        for dy in (-0.44, -0.08, 0.28):
            m |= M_ring(Y, X, -0.04*R, dy*R, 0.06*R, 0.17*R)
        return m
    return f

def mk_ped():
    def f(Y, X, R):
        m = M_disc(Y, X, -0.06*R, -0.34*R, 0.10*R)
        m |= M_seg(Y, X, -0.06*R, -0.22*R, 0.0, 0.10*R, 0.13*R)
        m |= M_seg(Y, X, 0.0, 0.10*R, -0.16*R, 0.45*R, 0.10*R)
        m |= M_seg(Y, X, 0.0, 0.10*R, 0.14*R, 0.45*R, 0.10*R)
        m |= M_seg(Y, X, -0.06*R, -0.14*R, 0.12*R, -0.02*R, 0.08*R)
        m |= M_seg(Y, X, -0.06*R, -0.14*R, -0.20*R, 0.0, 0.08*R)
        return m
    return f

def mk_children():
    def f(Y, X, R):
        m = M_disc(Y, X, -0.30*R, -0.06*R, 0.09*R)
        m |= M_seg(Y, X, -0.30*R, 0.04*R, -0.28*R, 0.24*R, 0.11*R)
        m |= M_seg(Y, X, -0.28*R, 0.24*R, -0.38*R, 0.48*R, 0.09*R)
        m |= M_seg(Y, X, -0.28*R, 0.24*R, -0.18*R, 0.48*R, 0.09*R)
        m |= M_disc(Y, X, 0.14*R, 0.0, 0.08*R)
        m |= M_seg(Y, X, 0.14*R, 0.09*R, 0.15*R, 0.26*R, 0.10*R)
        m |= M_seg(Y, X, 0.15*R, 0.26*R, 0.06*R, 0.48*R, 0.08*R)
        m |= M_seg(Y, X, 0.15*R, 0.26*R, 0.24*R, 0.48*R, 0.08*R)
        return m
    return f

def mk_cycle():
    def f(Y, X, R):
        m = M_ring(Y, X, -0.22*R, 0.32*R, 0.09*R, 0.22*R) | M_ring(Y, X, 0.22*R, 0.32*R, 0.09*R, 0.22*R)
        m |= pl(Y, X, [(-0.22*R, 0.30*R), (-0.02*R, -0.02*R), (0.22*R, 0.30*R)], 0.07*R)
        m |= M_seg(Y, X, -0.02*R, -0.02*R, 0.12*R, -0.10*R, 0.07*R)
        m |= M_seg(Y, X, -0.02*R, -0.02*R, -0.16*R, -0.08*R, 0.07*R)
        return m
    return f

def mk_snow():
    def f(Y, X, R):
        m = np.zeros(Y.shape, bool)
        for a in (90, 150, 210, 270, 330, 30):
            t = np.deg2rad(a)
            m |= M_seg(Y, X, 0.10*R*np.cos(t), 0.10*R*np.sin(t),
                             0.48*R*np.cos(t), 0.48*R*np.sin(t), 0.10*R)
        return m
    return f

def mk_deer():
    def f(Y, X, R):
        m = M_seg(Y, X, -0.18*R, -0.08*R, 0.26*R, 0.06*R, 0.30*R)
        m |= M_seg(Y, X, -0.18*R, -0.08*R, -0.34*R, -0.28*R, 0.14*R)
        m |= M_disc(Y, X, -0.38*R, -0.32*R, 0.09*R)
        m |= M_seg(Y, X, -0.42*R, -0.36*R, -0.50*R, -0.50*R, 0.05*R)
        m |= M_seg(Y, X, -0.34*R, -0.40*R, -0.28*R, -0.52*R, 0.05*R)
        m |= M_seg(Y, X, 0.16*R, 0.14*R, 0.06*R, 0.46*R, 0.09*R)
        m |= M_seg(Y, X, 0.30*R, 0.10*R, 0.40*R, 0.44*R, 0.09*R)
        return m
    return f

def mk_arrow(kind):
    def f(Y, X, R):
        r = R
        if kind == 'right':
            return arrow(Y, X, [(-0.16*r, 0.45*r), (-0.16*r, 0.02*r), (0.12*r, -0.18*r)], 0.20*r,
                         [((0.12*r, -0.18*r), (0.38*r, -0.46*r), 0.17*r)])
        if kind == 'left':
            return arrow(Y, X, [(0.16*r, 0.45*r), (0.16*r, 0.02*r), (-0.12*r, -0.18*r)], 0.20*r,
                         [((-0.12*r, -0.18*r), (-0.38*r, -0.46*r), 0.17*r)])
        if kind == 'ahead':
            return arrow(Y, X, [(0.0, 0.50*r), (0.0, -0.10*r)], 0.22*r,
                         [((0.0, -0.10*r), (0.0, -0.52*r), 0.18*r)])
        if kind == 'strright':
            m = arrow(Y, X, [(-0.24*r, 0.50*r), (-0.24*r, -0.10*r)], 0.20*r,
                      [((-0.24*r, -0.10*r), (-0.24*r, -0.52*r), 0.17*r)])
            m |= arrow(Y, X, [(-0.24*r, 0.18*r), (0.12*r, 0.32*r)], 0.18*r,
                       [((0.12*r, 0.32*r), (0.42*r, 0.46*r), 0.16*r)])
            return m
        if kind == 'strleft':
            m = arrow(Y, X, [(0.24*r, 0.50*r), (0.24*r, -0.10*r)], 0.20*r,
                      [((0.24*r, -0.10*r), (0.24*r, -0.52*r), 0.17*r)])
            m |= arrow(Y, X, [(0.24*r, 0.18*r), (-0.12*r, 0.32*r)], 0.18*r,
                       [((-0.12*r, 0.32*r), (-0.42*r, 0.46*r), 0.16*r)])
            return m
        if kind == 'keepr':
            return arrow(Y, X, [(-0.36*r, -0.36*r), (-0.06*r, -0.06*r)], 0.24*r,
                         [((-0.06*r, -0.06*r), (0.38*r, 0.38*r), 0.21*r)])
        if kind == 'keepl':
            return arrow(Y, X, [(0.36*r, -0.36*r), (0.06*r, -0.06*r)], 0.24*r,
                         [((0.06*r, -0.06*r), (-0.38*r, 0.38*r), 0.21*r)])
        if kind == 'round':
            m = np.zeros(Y.shape, bool)
            for a0 in (60, 180, 300):
                a1 = a0 + 80
                m |= M_arc(Y, X, 0, 0, 0.34*r, 0.54*r, a0, a1)
                t0 = np.deg2rad(a1); t1 = np.deg2rad(a1 + 20)
                base = (0.36*r*np.cos(t0), 0.36*r*np.sin(t0))
                tip  = (0.62*r*np.cos(t1), 0.62*r*np.sin(t1))
                d = np.array(tip) - np.array(base); L = np.hypot(*d) + 1e-9
                pv = np.array([-d[1], d[0]]) / L
                m |= M_poly(Y, X, [tip, tuple(np.array(base)+pv*0.13*r), tuple(np.array(base)-pv*0.13*r)])
            return m
    return f

def mk_hatch_car():
    def f(Y, X, R):
        return (0.6 * hatch_stripes(Y, X, R).astype(np.float64)
                + 2.0 * pic_car(Y, X, R, -0.18*R, -0.02*R, 0.85*R).astype(np.float64))
    return f

def mk_hatch_truck():
    def f(Y, X, R):
        return (0.6 * hatch_stripes(Y, X, R).astype(np.float64)
                + 2.0 * pic_truck(Y, X, R, -0.20*R, 0.0, 1.15*R).astype(np.float64))
    return f

def mk_end80():
    def f(Y, X, R):
        m = mk_digits("80")(Y, X, R)
        m |= M_seg(Y, X, -0.42*R, 0.44*R, 0.42*R, -0.44*R, 0.16*R)
        return m
    return f

# ---------------- sign frames --------------------------------------------
TRI_V  = [(0.0, -1.02), (-0.78, 0.97), (0.78, 0.97)]
DTRI_V = [(0.0, 1.02), (-0.78, -0.97), (0.78, -0.97)]

def tri_scale(v, R, s):
    gx = sum(p[0] for p in v) / 3.0; gy = sum(p[1] for p in v) / 3.0
    return [((gx + (p[0]-gx)*s)*R, (gy + (p[1]-gy)*s)*R) for p in v]

def frame_ring(Y, X, R):
    # red annulus; white (not red) mid-radius; not blue at the ring
    return [(CH_RG,  1.60, M_ring(Y, X, 0, 0, 0.76*R, R)),
            (CH_RG, -0.50, M_ring(Y, X, 0, 0, 0.42*R, 0.68*R)),
            (CH_BY, -0.50, M_ring(Y, X, 0, 0, 0.76*R, R)),
            (CH_BY, -0.30, M_ring(Y, X, 0, 0, 0.42*R, 0.68*R)),
            (CH_L,   0.40, M_disc(Y, X, 0, 0, 0.62*R)),
            (CH_L,  -0.30, M_ring(Y, X, 0, 0, 0.80*R, R))]

def frame_tri(Y, X, R, v=TRI_V):
    vv  = [(p[0]*R, p[1]*R) for p in v]
    iv  = tri_scale(v, R, 0.58)
    border = M_tri_edge(Y, X, vv[0], vv[1], vv[2], 0.26*R)
    wide   = M_tri_edge(Y, X, vv[0], vv[1], vv[2], 0.44*R)
    anti   = M_ring(Y, X, 0, 0, 0.80*R, 1.04*R) & ~wide
    return [(CH_RG,  1.00, border),
            (CH_RG, -0.80, anti),
            (CH_BY, -0.40, anti),
            (CH_RG, -0.50, M_poly(Y, X, iv)),
            (CH_BY, -0.40, border),
            (CH_L,   0.50, M_poly(Y, X, iv)),
            (CH_L,  -0.30, border)]

def frame_bluedisc(Y, X, R):
    return [(CH_BY,  1.00, M_disc(Y, X, 0, 0, 0.95*R)),
            (CH_RG, -0.50, M_disc(Y, X, 0, 0, 0.95*R)),
            (CH_L,  -0.30, M_disc(Y, X, 0, 0, 0.95*R)),
            (CH_D5, -0.30, M_disc(Y, X, 0, 0, 0.88*R))]

def frame_whitedisc(Y, X, R):
    return [(CH_L,    0.60, M_disc(Y, X, 0, 0, 0.78*R)),
            (CH_RG,  -1.30, M_ring(Y, X, 0, 0, 0.76*R, 1.00*R)),
            (CH_BY,  -0.50, M_ring(Y, X, 0, 0, 0.76*R, 1.00*R)),
            (CH_RG,  -0.20, M_disc(Y, X, 0, 0, 0.74*R)),
            (CH_BY,  -0.40, M_disc(Y, X, 0, 0, R)),
            (CH_D5,   0.50, M_ring(Y, X, 0, 0, 0.90*R, 0.99*R)),
            (CH_D9,   0.30, M_ring(Y, X, 0, 0, 0.90*R, 0.99*R))]

def frame_reddisc(Y, X, R):
    return [(CH_RG,  1.10, M_disc(Y, X, 0, 0, 0.94*R)),
            (CH_BY, -0.50, M_disc(Y, X, 0, 0, 0.94*R)),
            (CH_L,  -1.00, M_disc(Y, X, 0, 0, 0.70*R))]

def frame_oct(Y, X, R):
    octv = [(R*np.cos(np.deg2rad(22.5 + 45*i)), R*np.sin(np.deg2rad(22.5 + 45*i))) for i in range(8)]
    return [(CH_RG,  1.10, M_poly(Y, X, octv)),
            (CH_BY, -0.45, M_poly(Y, X, octv)),
            (CH_L,  -0.80, M_disc(Y, X, 0, 0, 0.70*R))]

def frame_diamond(Y, X, R):
    vv = [(0.0, -0.74*R), (0.74*R, 0.0), (0.0, 0.74*R), (-0.74*R, 0.0)]
    edge = M_poly(Y, X, vv) & ~M_poly(Y, X, [(p[0]*0.80, p[1]*0.80) for p in vv])
    return [(CH_D5,  1.00, edge),
            (CH_D9,  0.40, edge),
            (CH_D5, -0.60, M_ring(Y, X, 0, 0, 0.86*R, 0.98*R)),
            (CH_L,   0.60, M_poly(Y, X, vv)),
            (CH_RG, -0.35, M_poly(Y, X, vv)),
            (CH_BY, -0.35, M_poly(Y, X, vv))]

def inner_ring(Y, X, R):  return M_disc(Y, X, 0, 0, 0.70*R)
def inner_tri(Y, X, R):   return M_poly(Y, X, tri_scale(TRI_V, R, 0.58))
def inner_dtri(Y, X, R):  return M_poly(Y, X, tri_scale(DTRI_V, R, 0.58))
def inner_blue(Y, X, R):  return M_disc(Y, X, 0, 0, 0.80*R)
def inner_white(Y, X, R): return M_disc(Y, X, 0, 0, 0.76*R)

# ---------------- template registry ---------------------------------------
FAM_SHAPE = {
    'ring':    (frame_ring, True),
    'tri':     (frame_tri, True),
    'dtri':    (lambda Y, X, R: frame_tri(Y, X, R, DTRI_V), True),
    'blue':    (frame_bluedisc, True),
    'white':   (frame_whitedisc, True),
    'reddisc': (frame_reddisc, True),
    'oct':     (frame_oct, True),
    'diamond': (frame_diamond, True),
}
ROTS9 = (-9.0, 0.0, 9.0)   # every template type gets 3 rotations x 3 scales

def bright_bar(Y, X, R):
    return M_poly(Y, X, [(-0.58*R, -0.14*R), (0.58*R, -0.14*R), (0.58*R, 0.14*R), (-0.58*R, 0.14*R)])

def bright_stop(Y, X, R):
    m = np.zeros(Y.shape, bool)
    for x in (-0.42, -0.14, 0.14, 0.42):
        m |= M_seg(Y, X, x*R, -0.15*R, x*R, 0.17*R, 0.16*R)
    return m

TEMPLATES = []
def add(name, fam, cls, mode):
    TEMPLATES.append((name, fam, cls, mode, ROTS9))

for fam in FAM_SHAPE:
    add('shp_' + fam, fam, None, 'shape')

for txt, cls in [("20", 0), ("30", 1), ("50", 2), ("60", 3), ("70", 4), ("80", 5), ("100", 7), ("120", 8)]:
    add('ring_' + txt, 'ring', cls, ('dark', mk_digits(txt)))
add('ring_cars', 'ring', 9, ('dark', mk_cars()))
add('ring_cartruck', 'ring', 10, ('dark', mk_cartruck()))
add('ring_truck', 'ring', 16, ('dark', mk_truck()))
for nm, cls, fn in [('x', 11, mk_x()), ('excl', 18, mk_excl()), ('bendl', 19, mk_bend(True)),
                    ('bendr', 20, mk_bend(False)), ('dbend', 21, mk_dbend()), ('bumps', 22, mk_bumps()),
                    ('slip', 23, mk_slip()), ('narrow', 24, mk_narrow()), ('works', 25, mk_works()),
                    ('lights', 26, mk_lights()), ('ped', 27, mk_ped()), ('children', 28, mk_children()),
                    ('cycle', 29, mk_cycle()), ('snow', 30, mk_snow()), ('deer', 31, mk_deer())]:
    add('tri_' + nm, 'tri', cls, ('dark', fn))
for nm, cls in [('right', 33), ('left', 34), ('ahead', 35), ('strright', 36), ('strleft', 37),
                ('keepr', 38), ('keepl', 39), ('round', 40)]:
    add('blue_' + nm, 'blue', cls, ('bright', mk_arrow(nm)))
add('hatch_plain', 'white', 32, ('dark', hatch_stripes))
add('hatch_car', 'white', 41, ('dark', mk_hatch_car()))
add('hatch_truck', 'white', 42, ('dark', mk_hatch_truck()))
add('end80', 'white', 6, ('dark', mk_end80()))
add('noentry', 'reddisc', 17, ('bright', bright_bar))
add('stop', 'oct', 14, ('bright', bright_stop))
add('diamondc', 'diamond', 12, ('dark', lambda Y, X, R: M_poly(Y, X,
    [(0.0, -0.74*R), (0.74*R, 0.0), (0.0, 0.74*R), (-0.74*R, 0.0)])))

# ---------------- kernel assembly ----------------------------------------
LEVEL_CH = (CH_L, CH_RG, CH_BY)

def build_kernel(entry, R, ang):
    name, fam, cls, mode, rots = entry
    Y, X = rot_coords(ang)
    if mode == 'shape':
        planes = list(FAM_SHAPE[fam][0](Y, X, R))
    elif isinstance(mode, tuple) and mode[0] == 'dark':
        cm = mode[1](Y, X, R)
        inn = {'ring': inner_ring, 'tri': inner_tri, 'dtri': inner_dtri,
               'white': inner_white, 'blue': inner_blue, 'diamond': inner_white,
               'reddisc': inner_ring, 'oct': inner_ring}[fam](Y, X, R)
        planes = content_dark_planes(inn, cm)
        if name == 'tri_lights':
            planes += [(CH_RG,  0.5, down(M_disc(Y, X, -0.04*R, -0.44*R, 0.15*R)), False),
                       (CH_RG, -0.4, down(M_disc(Y, X, -0.04*R, 0.28*R, 0.15*R)), False),
                       (CH_BY, -0.3, down(M_disc(Y, X, -0.04*R, -0.08*R, 0.14*R)), False)]
    elif isinstance(mode, tuple) and mode[0] == 'bright':
        cm = mode[1](Y, X, R)
        inn = {'blue': inner_blue, 'reddisc': lambda Y, X, R: M_disc(Y, X, 0, 0, 0.80*R),
               'oct': lambda Y, X, R: M_disc(Y, X, 0, 0, 0.70*R)}[fam](Y, X, R)
        planes = content_bright_planes(inn, cm, wb=0.9 if name == 'stop' else 1.5)
        if name == 'stop':   # the stop sign's white inner rim is content-like
            planes += [(CH_B5, 0.8, down(M_ring(Y, X, 0, 0, 0.74*R, 0.88*R)), False),
                       (CH_B9, 0.4, down(M_ring(Y, X, 0, 0, 0.74*R, 0.88*R)), False)]
    kern = np.zeros((7, K, K))
    for item in planes:
        ch, w, mask = item[0], item[1], item[2]
        is_ss = len(item) < 4 or item[3]
        p = w * (down(mask) if is_ss else mask)
        if ch in LEVEL_CH:
            p = p - p.mean()          # zero-sum: the level maps carry a +8 offset
        nrm = np.sqrt((p * p).sum())
        if nrm < 1e-9:
            continue
        kern[ch] += CHW[ch] * p / nrm
    n = np.sqrt((kern * kern).sum())
    if n > 1e-9:
        kern *= KNORM / n
    return kern

# ---------------- feature-extraction kernels ------------------------------
def gauss2d(n, sigma):
    c = (n - 1) / 2.0
    ax = np.arange(n) - c
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx*xx + yy*yy) / (2.0*sigma*sigma))
    return g / g.sum()

def make_feat_W():
    # out: 0 L(+8 bias), 1 RG(+8), 2 BY(+8), 3 -DoG5 (dark thin), 4 +DoG5,
    #      5 -DoG9 (dark thick), 6 +DoG9 -- relu afterwards makes 3..6 one-sided
    W = np.zeros((7, 3, 9, 9))
    ident = np.zeros((9, 9)); ident[4, 4] = 1.0
    for i in range(3):
        W[i, i] = ident
    dog5 = np.zeros((9, 9)); dog5[2:7, 2:7] += gauss2d(5, 0.7) - gauss2d(5, 1.6)
    dog9 = gauss2d(9, 1.4) - gauss2d(9, 3.2)
    for A in (dog5, dog9):
        A /= np.sqrt((A * A).sum())
    W[CH_D5, 0] = -dog5
    W[CH_B5, 0] =  dog5
    W[CH_D9, 0] = -dog9
    W[CH_B9, 0] =  dog9
    b = np.zeros(7); b[0] = b[1] = b[2] = 8.0
    return W, b

# ---------------- build ----------------------------------------------------
def main():
    chans = []
    kernels = []
    for entry in TEMPLATES:
        name, fam, cls, mode, rots = entry
        for R in SCALES:
            for ang in rots:
                kernels.append(build_kernel(entry, R, ang))
                chans.append(dict(name=name, fam=fam, cls=cls,
                                  shape=(mode == 'shape'), scale=R, ang=ang))
    T = len(chans)
    bankW = np.stack(kernels).astype(np.float32)
    bankb = np.zeros(T, np.float32)

    # dense 1: gates with global per-image common-mode removal
    THETA_SHAPE = 30.0
    THETA_CONTENT = 8.0
    W1 = np.full((T, T), -1.0 / T, np.float32); b1 = np.zeros(T, np.float32)
    for j, ch in enumerate(chans):
        W1[j, j] += 1.0
        b1[j] = -(THETA_SHAPE if ch['shape'] else THETA_CONTENT)

    # dense 2: class rules (every type has 9 channels -> uniform means)
    CLS_FAM = {15: 'ring', 13: 'dtri'}   # empty-interior classes (no content type)
    for ch in chans:
        if ch['cls'] is not None:
            CLS_FAM[ch['cls']] = ch['fam']
    PRIOR2 = [1, 2, 4, 5, 10, 12, 13, 35, 38]
    PRIOR1 = [3, 7, 8, 9, 11, 17, 18, 26, 28, 31, 33, 34, 36, 39, 40]
    EMPTY_BONUS = {15: 2.5, 13: 2.5}     # empty-interior classes win ties
    W2 = np.zeros((43, T), np.float32); b2 = np.zeros(43, np.float32)
    for k in range(43):
        fam = CLS_FAM.get(k)
        if fam is None:
            continue
        has_content = any((not c['shape']) and c['cls'] == k for c in chans)
        for j, ch in enumerate(chans):
            n = 9.0
            if not ch['shape'] and ch['cls'] == k:
                W2[k, j] += 3.0 / n
            elif not ch['shape'] and ch['fam'] == fam:
                W2[k, j] += -0.15 / n
            elif ch['shape'] and ch['fam'] != fam:
                W2[k, j] += -0.6 / n
            elif ch['shape'] and ch['fam'] == fam:
                W2[k, j] += 1.5 / n
        b2[k] = EMPTY_BONUS.get(k, 0.0)
        if k in PRIOR2:   b2[k] += 0.30
        elif k in PRIOR1: b2[k] += 0.15

    featW, featb = make_feat_W()
    colW = np.zeros((3, 3, 1, 1), np.float32)
    colW[0, :, 0, 0] = np.array([1/3., 1/3., 1/3.])
    colW[1, :, 0, 0] = np.array([1.0, -1.0, 0.0])
    colW[2, :, 0, 0] = np.array([-0.5, -0.5, 1.0])
    smW = np.zeros((3, 3, 3, 3), np.float32)
    binom = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64) / 16.0
    for i in range(3):
        smW[i, i] = binom

    weights = {
        'col.W': colW,  'col.b': np.zeros(3, np.float32),
        'sm.W':  smW,   'sm.b':  np.zeros(3, np.float32),
        'feat.W': featW.astype(np.float32), 'feat.b': featb.astype(np.float32),
        'bank.W': bankW, 'bank.b': bankb,
        'd1.W': W1, 'd1.b': b1,
        'd2.W': W2, 'd2.b': b2,
    }
    np.savez(OUT + '/weights.npz', **{k: v.astype(np.float32) for k, v in weights.items()})

    net = {"layers": [
        {"name": "col",  "op": "conv", "inputs": ["image"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "sm",   "op": "conv", "inputs": ["col"],   "out": 3, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "nrm",  "op": "norm", "inputs": ["sm"]},
        {"name": "feat", "op": "conv", "inputs": ["nrm"],   "out": 7, "kernel": [9, 9], "stride": 1, "padding": 4},
        {"name": "fr",   "op": "relu", "inputs": ["feat"]},
        {"name": "bank", "op": "conv", "inputs": ["fr"],    "out": T, "kernel": [K, K], "stride": 1, "padding": 3},
        {"name": "pool", "op": "maxpool", "inputs": ["bank"], "size": [11, 11], "stride": [11, 11]},
        {"name": "rs",   "op": "reshape", "inputs": ["pool"], "shape": [T]},
        {"name": "d1",   "op": "dense", "inputs": ["rs"],   "out": T},
        {"name": "g",    "op": "relu",  "inputs": ["d1"]},
        {"name": "d2",   "op": "dense", "inputs": ["g"],    "out": 43},
    ]}
    with open(OUT + '/network.json', 'w') as f:
        json.dump(net, f)
    total = sum(int(np.prod(v.shape)) for v in weights.values())
    print("templates:", T, " total weights:", total)

if __name__ == "__main__" or len(sys.argv) > 1:
    main()
