"""Hand-shape template library: filter bank, renderer, letter specs.
All numbers are typed from general knowledge of hands / ASL letters.
Coordinates: 28x28 frame, x right, y down, origin top-left. Hand drawn in 2x supersample."""
import numpy as np

# ---------------- filter bank (all 9x9, zero-mean, unit norm) ----------------
K = 9

def _grid(k):
    c = (k - 1) / 2.0
    x = np.arange(k) - c
    return np.meshgrid(x, x)  # X, Y

def make_bank():
    F = []
    # odd (edge) Gabors, two scales
    for sig, lam in [(1.5, 4.2), (2.5, 7.0)]:
        for th in range(6):
            t = th * np.pi / 6
            X, Y = _grid(K)
            xr = X * np.cos(t) + Y * np.sin(t)
            F.append(np.exp(-(X**2 + Y**2) / (2 * sig**2)) * np.sin(2 * np.pi * xr / lam))
    # even (bar) Gabors, two widths
    for sig, lam in [(1.8, 5.0), (2.6, 8.0)]:
        for th in range(6):
            t = th * np.pi / 6
            X, Y = _grid(K)
            xr = X * np.cos(t) + Y * np.sin(t)
            F.append(np.exp(-(X**2 + Y**2) / (2 * sig**2)) * np.cos(2 * np.pi * xr / lam))
    # LoG blobs, two scales
    for sig in [1.5, 2.8]:
        X, Y = _grid(K)
        r2 = X**2 + Y**2
        F.append((r2 - 2 * sig**2) / sig**4 * np.exp(-r2 / (2 * sig**2)))
    out = []
    for g in F:
        g = g - g.mean()
        out.append(g / np.sqrt((g**2).sum()))
    return np.stack(out).astype(np.float64)   # (26, 9, 9)

NF = 26

# ---------------- renderer ----------------
SS = 2
N = 28 * SS
GX, GY = np.meshgrid(np.arange(N, dtype=np.float64), np.arange(N, dtype=np.float64))  # x, y in 56-space

def seg_dist(px, py, x0, y0, x1, y1):
    vx, vy = x1 - x0, y1 - y0
    l2 = vx * vx + vy * vy
    t = np.clip(((px - x0) * vx + (py - y0) * vy) / max(l2, 1e-9), 0.0, 1.0)
    return np.hypot(px - (x0 + t * vx), py - (y0 + t * vy))

def cap(x0, y0, x1, y1, r):
    """capsule in 28-space, soft edge ~0.75 px (28-scale)"""
    d = seg_dist(GX, GY, x0 * SS, y0 * SS, x1 * SS, y1 * SS)
    return np.clip(1 - (d - r * SS) / 1.6, 0, 1)

def ell(cx, cy, rx, ry, ang=0.0):
    ca, sa = np.cos(ang), np.sin(ang)
    dx, dy = GX - cx * SS, GY - cy * SS
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    rho = np.sqrt((u / (rx * SS))**2 + (v / (ry * SS))**2)
    return np.clip(1 - (rho - 1) / 0.11, 0, 1)

def arc(cx, cy, r0, r1, a0, a1):
    """annulus sector; angles in degrees, y-down convention (atan2(dy,dx)).
    covers angles going from a0 up to a1 (a1 > a0, can pass 180/-180)."""
    dx, dy = GX - cx * SS, GY - cy * SS
    d = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))
    # distance (in degrees) to the arc interval [a0, a1] modulo 360
    t = (ang - a0) % 360.0
    span = (a1 - a0) % 360.0
    # t = 0 at a0, increasing with angle; distance to interval [0, span]
    raw = a1 - a0
    span = raw % 360.0
    if raw >= 359.9:
        span = 360.0
    if span >= 359.9:
        inside = np.ones_like(d)
    else:
        inside = np.clip(np.minimum(t, span - t) / 9.0, 0, 1)
    rad = np.clip(1 - (d - r1 * SS) / 1.6, 0, 1) * np.clip(1 - (r0 * SS - d) / 1.6, 0, 1)
    return rad * inside

def cut(m, c):
    return np.minimum(m, 1 - c)

def finish(m56):
    """blur + downsample to 28"""
    k = np.exp(-np.arange(-4, 5)**2 / (2 * 1.1**2))
    k /= k.sum()
    tmp = np.zeros_like(m56)
    for i, w in enumerate(k):
        if i == 4: tmp += w * m56
        else: tmp += w * np.roll(m56, i - 4, axis=1)
    out = np.zeros_like(m56)
    for i, w in enumerate(k):
        if i == 4: out += w * tmp
        else: out += w * np.roll(tmp, i - 4, axis=0)
    # np.roll wraps around; zero the wrap artifacts at borders
    out[:, :4] = np.minimum(out[:, :4], m56[:, :4]); out[:, -4:] = np.minimum(out[:, -4:], m56[:, -4:])
    out[:4, :] = np.minimum(out[:4, :], m56[:4, :]); out[-4:, :] = np.minimum(out[-4:, :], m56[-4:, :])
    return out.reshape(28, SS, 28, SS).mean(axis=(1, 3))

def rot_scale(m56, deg, sc, cx=14.0, cy=14.5, sx=1.0, sy=1.0):
    """rotate+scale (optionally anisotropic) a rendered 56-mask, bilinear"""
    t = np.radians(deg)
    ca, sa = np.cos(t), np.sin(t)
    px, py = np.meshgrid(np.arange(N, dtype=np.float64), np.arange(N, dtype=np.float64))
    # invert: source coords for each target pixel
    dx, dy = (px - cx * SS) / sc, (py - cy * SS) / sc
    dx = dx / sx; dy = dy / sy
    sx_ = (dx * ca + dy * sa) + cx * SS
    sy_ = (-dx * sa + dy * ca) + cy * SS
    sx, sy = sx_, sy_
    x0 = np.clip(np.floor(sx).astype(int), 0, N - 2)
    y0 = np.clip(np.floor(sy).astype(int), 0, N - 2)
    fx = np.clip(sx - x0, 0, 1); fy = np.clip(sy - y0, 0, 1)
    return (m56[y0, x0] * (1 - fx) * (1 - fy) + m56[y0, x0 + 1] * fx * (1 - fy)
            + m56[y0 + 1, x0] * (1 - fx) * fy + m56[y0 + 1, x0 + 1] * fx * fy)

# ---------------- letter specs ----------------
# Common building blocks (28-space). Wrist enters from bottom for upright hands.
def wrist(m, cx=13.5, cy=21.0, r=4.2, x1=13.8, y1=30.5):
    return np.maximum(m, cap(cx, cy, x1, y1, r))

def palm(m, cx=13.5, cy=17.5, rx=5.8, ry=6.0):
    return np.maximum(m, ell(cx, cy, rx, ry))

def draw_A(p):
    m = ell(14.0, 12.9, 9.2, 7.2)                       # fist (measured extent)
    m = np.maximum(m, cap(20.1, 7.6, 20.9, 15.6, 1.9))  # thumb ridge along right side
    for x in [10.6, 13.6, 16.6]:                        # knuckle bumps top
        m = np.maximum(m, cap(x - 1.2, 7.8, x + 1.3, 7.0, 1.5))
    m = cut(m, cap(18.9, 8.0, 19.1, 15.2, 0.45))        # slot between thumb and fist
    m = cut(m, cap(12.1, 8.4, 12.0, 10.4, 0.4))         # slots between knuckles
    m = cut(m, cap(15.1, 8.3, 15.0, 10.3, 0.4))
    return wrist(m, cx=13.8, cy=20.0, r=4.6, x1=14.2, y1=30.5)

def draw_B(p):
    sp = p.get('spread', 1.0)
    m = ell(13.8, 18.0, 5.6, 5.4)                      # palm
    fx = [16.6, 13.6, 10.7, 8.2]                       # index..pinky (index right)
    ft = [4.2, 3.0, 4.4, 8.2]                          # fingertip y
    fr = [1.85, 1.95, 1.85, 1.55]
    for x, t, r in zip(fx, ft, fr):
        xb = 14.2 + (x - 14.2) * sp
        m = np.maximum(m, cap(xb, 13.8, xb + (x - 14.2) * .12, t, r))
    for x in [15.15, 12.15, 9.45]:                     # thin gaps between fingers
        m = cut(m, cap(14.2 + (x - 14.2) * sp, 5.2, 14.2 + (x - 14.2) * sp, 12.4, 0.3))
    m = np.maximum(m, cap(8.6, 15.6, 7.0, 17.6, 1.7))  # thumb bump on left (pinky side)
    m = cut(m, cap(7.6, 16.8, 17.5, 16.4, 0.45))       # thumb crease line across palm
    return wrist(m)

def draw_C(p):
    m = arc(17.6, 14.2, 5.4, 11.6, 68, 292)            # back of C on left, opens right
    m = np.maximum(m, cap(8.5, 20.5, 6.0, 29.5, 3.4))  # wrist bottom-left
    m = cut(m, cap(11.5, 14.0, 12.5, 14.1, 0.4))       # faint heel line? keep light
    return m

def draw_D(p):
    m = ell(12.6, 17.6, 5.3, 5.1)                      # base (curled fingers+thumb ring)
    m = np.maximum(m, cap(15.8, 14.2, 16.3, 3.2, 1.9))  # index up
    m = cut(m, ell(12.6, 17.9, 2.6, 2.5))              # hole of the O
    m = cut(m, cap(14.3, 13.2, 14.5, 15.8, 0.4))       # slot between index and base
    return wrist(m, cx=12.2, cy=21.5, x1=12.0, y1=30.5)

def draw_E(p):
    m = ell(14.0, 12.6, 9.7, 7.4)                       # wide top mass
    m = np.maximum(m, ell(13.8, 18.6, 6.6, 4.0))        # taper to wrist
    for x in [8.9, 11.7, 14.5, 17.3]:                   # 4 curled fingertip ridges
        m = np.maximum(m, cap(x, 8.4, x, 4.6, 1.55))
    for x in [10.3, 13.1, 15.9]:
        m = cut(m, cap(x, 4.4, x, 9.8, 0.45))           # slots between ridges
    return wrist(m, cx=13.6, cy=21.5, r=4.2, x1=13.8, y1=30.5)

def draw_F(p):
    sp = p.get('spread', 1.0)
    m = ell(13.2, 17.8, 5.6, 5.4)                      # palm
    # three up fingers on the left (middle tallest, pinky shortest), spread apart
    for x, t, r in [(6.4, 9.2, 1.45), (10.6, 5.6, 1.7), (14.8, 3.4, 1.8)]:
        xb = 10.6 + (x - 10.6) * sp
        m = np.maximum(m, cap(xb, 14.4, xb, t, r))
    for x in [8.5, 12.7]:
        m = cut(m, cap(10.6 + (x - 10.6) * sp, 6.0, 10.6 + (x - 10.6) * sp, 13.4, 0.4))
    m = np.maximum(m, ell(17.4, 16.0, 3.5, 3.7))       # index+thumb ring at base right
    m = cut(m, ell(17.5, 16.2, 2.1, 2.25))             # its hole
    return wrist(m, cx=13.2, cy=21.8)

def draw_G(p):
    lo = p.get('low', 0.0)                             # alt pose: finger a bit lower
    m = ell(8.8, 14.2, 5.2, 6.0)                       # curled mass, left-middle
    m = np.maximum(m, cap(13.0, 8.6 + lo, 26.6, 8.0 + lo, 1.9))   # index pointing right
    m = np.maximum(m, cap(12.6, 15.2, 22.6, 16.8, 1.55))  # thumb below the gap
    m = np.maximum(m, cap(8.2, 18.5, 6.2, 29.5, 3.6))  # wrist bottom-left
    return m

def draw_H(p):
    lo = p.get('low', 0.0)
    m = ell(8.4, 15.8, 5.0, 6.4)                       # mass below-left
    m = np.maximum(m, cap(10.6, 9.9 + lo, 25.8, 9.3 + lo, 1.9))   # index pointing right
    m = np.maximum(m, cap(11.0, 13.1 + lo, 25.4, 12.7 + lo, 1.9))  # middle below it
    m = cut(m, cap(12.5, 11.5 + lo, 24.5, 11.1 + lo, 0.42))  # line between the fingers
    m = np.maximum(m, cap(8.0, 21.0, 6.4, 29.5, 3.6))  # wrist
    return m

def draw_I(p):
    m = ell(14.0, 13.6, 9.0, 7.1)                       # fist
    m = np.maximum(m, cap(15.6, 10.0, 16.3, 3.2, 1.65))  # pinky up
    m = cut(m, cap(14.0, 9.2, 14.4, 6.8, 0.42))         # slot pinky/fist
    return wrist(m, cx=13.6, cy=20.5, r=4.4, x1=14.0, y1=30.5)

def draw_K(p):
    sp = p.get('spread', 1.0)
    m = ell(14.0, 18.0, 5.5, 5.2)
    m = np.maximum(m, cap(16.9, 13.8, 17.5, 2.9, 1.9))            # index up (right)
    m = np.maximum(m, cap(12.6, 14.2, 12.6 - 3.1 * sp, 5.0, 1.9))  # middle angled out-left
    m = np.maximum(m, cap(14.9, 15.6, 14.6, 8.2, 1.45))           # thumb up between them
    m = cut(m, cap(16.0, 12.5, 15.9, 7.0, 0.4))                   # slot index/thumb
    m = cut(m, cap(13.6, 13.5, 13.4, 7.4, 0.4))                   # slot thumb/middle
    return wrist(m)

def draw_L(p):
    m = ell(12.8, 17.6, 5.2, 5.6)                      # palm/fist mass
    m = np.maximum(m, cap(15.6, 14.0, 16.1, 3.0, 1.9))  # index up
    m = np.maximum(m, cap(16.6, 12.6, 21.4, 11.2, 1.65))  # thumb out to the right
    m = cut(m, cap(17.6, 12.0, 22.0, 10.6, 0.5))       # crook between index and thumb
    return wrist(m, cx=12.2, cy=21.8)

def draw_M(p):
    m = ell(14.0, 12.9, 9.3, 7.2)                       # fist
    for x0, y0, x1, y1 in [(18.0, 6.8, 15.3, 13.6), (14.6, 6.3, 12.0, 13.2), (11.2, 5.9, 8.6, 12.6)]:
        m = np.maximum(m, cap(x0, y0, x1, y1, 1.6))     # three draped finger ridges
    m = cut(m, cap(16.5, 7.4, 13.7, 13.8, 0.42))
    m = cut(m, cap(13.1, 6.9, 10.5, 13.4, 0.42))
    return wrist(m, cx=13.6, cy=20.2, r=4.4, x1=14.0, y1=30.5)

def draw_N(p):
    m = ell(14.1, 13.0, 9.1, 7.1)
    for x0, y0, x1, y1 in [(17.6, 7.0, 14.9, 13.8), (13.9, 6.4, 11.2, 13.3)]:
        m = np.maximum(m, cap(x0, y0, x1, y1, 1.65))    # two draped ridges
    m = cut(m, cap(15.7, 7.6, 13.0, 14.0, 0.44))
    return wrist(m, cx=13.7, cy=20.3, r=4.4, x1=14.1, y1=30.5)

def draw_O(p):
    m = arc(14.0, 13.6, 3.4, 9.7, -20, 340)             # ring, seam upper-left
    m = np.maximum(m, cap(13.5, 21.5, 14.0, 30.5, 4.4))  # wrist
    m = cut(m, cap(8.5, 10.5, 9.0, 9.5, 0.45))          # seam line at left
    return m

def draw_P(p):
    m = ell(14.0, 7.8, 6.6, 5.4)                       # mass at top
    m = np.maximum(m, cap(13.0, -1.5, 13.8, 5.0, 3.6))  # arm from top
    m = np.maximum(m, cap(16.2, 11.8, 17.6, 24.0, 1.8))  # index down
    m = np.maximum(m, cap(11.6, 11.8, 10.0, 23.0, 1.8))  # middle down (spread)
    m = np.maximum(m, cap(14.1, 12.0, 14.4, 18.6, 1.4))  # thumb between
    m = cut(m, cap(15.3, 13.0, 15.7, 18.0, 0.4))
    m = cut(m, cap(12.8, 13.0, 12.5, 18.0, 0.4))
    return m

def draw_Q(p):
    m = ell(15.5, 8.2, 6.8, 5.6)                       # mass at top
    m = np.maximum(m, cap(16.8, 9.5, 17.6, 29.5, 3.1))  # arm/wrist descending
    m = np.maximum(m, cap(13.4, 11.5, 11.2, 25.0, 1.7))  # index hooking down-left
    m = np.maximum(m, cap(12.6, 12.0, 8.6, 20.0, 1.4))   # thumb further left, shorter
    m = cut(m, cap(15.0, 12.5, 13.0, 22.0, 0.5))       # gap between arm and hook
    return m

def draw_R(p):
    m = ell(14.0, 18.0, 5.6, 5.2)
    m = np.maximum(m, cap(16.3, 13.8, 13.5, 3.4, 1.85))  # index leaning left at top
    m = np.maximum(m, cap(12.5, 13.8, 15.7, 2.6, 1.9))  # middle leaning right at top
    m = cut(m, cap(14.3, 8.2, 14.4, 12.6, 0.4))         # slot below the crossing
    m = np.maximum(m, cap(9.6, 16.6, 8.2, 19.0, 2.1))   # folded ring/pinky bump left
    return wrist(m)

def draw_S(p):
    m = ell(14.0, 12.9, 9.9, 7.3)                       # fist, wide
    m = np.maximum(m, cap(21.8, 12.2, 4.6, 12.9, 1.9))  # thumb across the front
    m = cut(m, cap(20.6, 10.4, 5.4, 10.8, 0.48))        # shadow line above thumb
    m = cut(m, cap(20.8, 15.0, 5.6, 15.4, 0.48))        # shadow line below thumb
    for x in [10.8, 14.0, 17.0]:                        # knuckle bumps top
        m = np.maximum(m, cap(x - 1.1, 7.6, x + 1.1, 7.0, 1.4))
    return wrist(m, cx=13.8, cy=20.2, r=4.4, x1=14.2, y1=30.5)

def draw_T(p):
    m = ell(14.0, 13.0, 9.4, 7.2)                       # fist
    m = np.maximum(m, cap(15.9, 9.8, 16.3, 4.0, 1.65))  # thumb tip up
    m = cut(m, cap(14.6, 5.4, 14.7, 9.4, 0.42))         # slots around thumb
    m = cut(m, cap(17.4, 5.4, 17.5, 9.6, 0.42))
    for x in [10.2, 12.7]:                              # knuckle bumps top-left
        m = np.maximum(m, cap(x - 1.0, 8.0, x + 1.0, 7.4, 1.35))
    return wrist(m, cx=13.6, cy=20.4, r=4.4, x1=14.0, y1=30.5)

def draw_U(p):
    m = ell(14.2, 18.0, 5.8, 5.2)
    m = np.maximum(m, cap(12.5, 13.8, 12.3, 3.6, 1.95))  # middle (left)
    m = np.maximum(m, cap(15.9, 13.8, 16.2, 3.0, 1.95))  # index (right, taller? index shorter)
    m = cut(m, cap(14.2, 4.0, 14.25, 12.4, 0.38))       # thin line between fingers
    m = np.maximum(m, cap(9.4, 16.6, 8.2, 19.2, 2.2))   # folded ring/pinky bump left
    return wrist(m)

def draw_V(p):
    sp = p.get('spread', 1.0)
    m = ell(14.0, 18.0, 5.6, 5.2)
    m = np.maximum(m, cap(15.8, 13.8, 15.8 + 2.1 * sp, 4.4, 1.85))   # index up-right
    m = np.maximum(m, cap(12.6, 13.8, 12.6 - 2.0 * sp, 3.1, 1.9))    # middle up-left (taller)
    m = np.maximum(m, cap(9.6, 16.8, 8.3, 19.2, 2.1))  # folded ring/pinky bump left
    return wrist(m)

def draw_W(p):
    sp = p.get('spread', 1.0)
    m = ell(14.2, 18.2, 5.6, 5.0)
    m = np.maximum(m, cap(17.4, 14.0, 17.4 + 1.9 * sp, 6.4, 1.8))    # index
    m = np.maximum(m, cap(14.2, 13.8, 14.1, 3.0, 1.85))              # middle (tallest)
    m = np.maximum(m, cap(11.0, 14.0, 11.0 - 1.9 * sp, 6.0, 1.8))    # ring
    m = np.maximum(m, cap(8.6, 17.2, 7.2, 19.8, 1.9))  # thumb+pinky bump bottom-left
    return wrist(m)

def draw_X(p):
    m = ell(13.8, 17.8, 5.6, 5.2)
    m = np.maximum(m, cap(14.6, 14.0, 14.3, 4.2, 1.7))  # finger column up
    m = np.maximum(m, ell(15.2, 9.6, 2.9, 2.6))         # hooked bulge near top
    m = cut(m, cap(16.6, 8.0, 16.8, 11.4, 0.42))        # slot right of bulge
    return wrist(m)

def draw_Y(p):
    m = ell(13.6, 16.8, 5.7, 5.5)                       # fist
    m = np.maximum(m, cap(16.2, 12.8, 19.3, 4.8, 1.5))  # pinky up-right
    m = np.maximum(m, cap(11.0, 15.2, 5.2, 12.8, 1.65))  # thumb out-left, slightly up
    m = cut(m, cap(15.2, 12.6, 17.8, 8.0, 0.42))        # slot pinky/fist
    m = cut(m, cap(10.4, 14.4, 7.0, 13.0, 0.42))        # slot thumb/fist
    return wrist(m, cx=13.2, cy=21.5)

DRAW = [draw_A, draw_B, draw_C, draw_D, draw_E, draw_F, draw_G, draw_H, draw_I, draw_K,
        draw_L, draw_M, draw_N, draw_O, draw_P, draw_Q, draw_R, draw_S, draw_T, draw_U,
        draw_V, draw_W, draw_X, draw_Y]

# variant parameter sets: (nominal, alt)
ALT = {
    'B': dict(spread=1.28), 'F': dict(spread=1.3), 'K': dict(spread=1.35),
    'V': dict(spread=1.45), 'W': dict(spread=1.35), 'U': dict(), 'R': dict(),
    'G': dict(low=1.6), 'H': dict(low=1.4),
}
# per-class overall scale (hands fill most of the 28px frame) about (14.0, 14.8)
SCALE = {k: 1.30 for k in range(24)}
for k in [0, 4, 8, 11, 12, 13, 17, 18]:   # fist letters: geometry already at target
    SCALE[k] = 1.0
SCALE[2] = 1.12   # C already big

def elastic_warp56(m56, amp, freq, phase, shear):
    """smooth elastic + shear warp of a rendered 56-mask (bilinear)"""
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    k1, k2 = freq
    p1, p2, p3, p4 = phase
    dx = amp * np.sin(xx * k1 + p1) * np.cos(yy * k2 + p2)
    dy = amp * np.sin(yy * k2 + p3) * np.cos(xx * k1 + p4)
    sx = xx + dx + shear * (yy - N / 2) * 0.5
    sy = yy + dy
    x0 = np.clip(np.floor(sx).astype(int), 0, N - 2); y0 = np.clip(np.floor(sy).astype(int), 0, N - 2)
    fx = np.clip(sx - x0, 0, 1); fy = np.clip(sy - y0, 0, 1)
    return (m56[y0, x0] * (1 - fx) * (1 - fy) + m56[y0, x0 + 1] * fx * (1 - fy)
            + m56[y0 + 1, x0] * (1 - fx) * fy + m56[y0 + 1, x0 + 1] * fx * fy)

def letter_variants(k, nvar=16, seed=1234 + 0):
    """nvar pose variants: nominal, alt, mirrors, plus seeded elastic jitter clones."""
    fn = DRAW[k]
    name = "ABCDEFGHIKLMNOPQRSTUVWXY"[k]
    p0 = {}
    p1 = ALT.get(name, {})
    sc = SCALE[k]
    m0 = rot_scale(fn(p0), 0.0, sc, 14.0, 14.8)
    m1 = rot_scale(fn(p1), 0.0, sc, 14.0, 14.8)
    sym = name in ('E', 'O', 'S', 'M', 'N')
    if sym:
        base = [m0, m1,
                rot_scale(fn(p0), 5, sc * 0.96, 14.0, 14.8),
                rot_scale(fn(p1), -5, sc * 1.04, 14.0, 14.8)]
    else:
        base = [m0, m1, m0[:, ::-1], m1[:, ::-1]]
    rng = np.random.RandomState(seed * 1000 + 77 * k)
    outs = [finish(b) for b in base]
    j = 0
    while len(outs) < nvar:
        b = base[j % len(base)]
        amp = rng.uniform(0.8, 2.2) * SS
        freq = (rng.uniform(0.12, 0.3), rng.uniform(0.12, 0.3))
        phase = rng.uniform(0, 6.28, 4)
        shear = rng.uniform(-0.10, 0.10)
        rot = rng.uniform(-5, 5)
        scj = rng.uniform(0.95, 1.06)
        mm = rot_scale(b, rot, scj, 14.0, 14.8)
        mm = elastic_warp56(mm, amp, freq, phase, shear)
        outs.append(finish(mm))
        j += 1
    return outs[:nvar]



# =====================================================================
#  Network assembly v8: coarse 7x7 layout matcher, noise-direction removal
# =====================================================================
import json

CS_SET = [(1.8, 4.8), (3.2, 8.5)]     # mid + large center-surround scales only
NV = 16                               # variants per class (pose ensemble)
TM2_STRIDE = 2
TM1_W = 0.8                            # 14x14 stream weight
ORTHO_BASIS = False                   # (mean-direction removal hurts the coarse signal)

def gauss_k(k, sigma):
    c = (k - 1) / 2.0
    x = np.arange(k) - c
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))

def dog(si, so, k):
    gi, go = gauss_k(k, si), gauss_k(k, so)
    gi /= gi.sum(); go /= go.sum()
    d = gi - go
    return d / np.sqrt((d**2).sum())

KBIG = 29
def front_kernels(cs_set=CS_SET):
    W = np.zeros((len(cs_set), 1, KBIG, KBIG))
    for i, (si, so) in enumerate(cs_set):
        k = min(2 * int(np.ceil(so * 1.35)) + 1, KBIG)
        g = dog(si, so, k)
        o = (KBIG - k) // 2
        W[i, 0, o:o+k, o:o+k] = g
    return W

def taper2d(n):
    """regional taper: frame edges down, bottom (sleeve) strongly down"""
    t = np.ones((n, n))
    r = max(1, n // 6)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(r) + 0.5) / r)
    t[:r, :] *= ramp[:, None]
    t[:, :r] *= ramp[None, :]
    t[:, n-r:] *= ramp[::-1][None, :]
    b = max(1, n // 5)
    bramp = np.linspace(1.0, 0.15, b)
    t[n-b:, :] *= np.maximum(bramp, 0.15)[:, None]
    return t

def cenorm(x):
    mu = x.mean(axis=(2, 3), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)

def avgpool_hw(x, k):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // k, k, w // k, k).mean(axis=(3, 5))

def conv2(x, W, stride=1, pad=0):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
                     for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo)

def remove_basis(T, B):
    """T: (V,D) unit rows. Remove projections onto orthonormalised span of B rows."""
    Q, _ = np.linalg.qr(np.asarray(B).T)      # (D, m)
    T = T - (T @ Q) @ Q.T
    return T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)

def build_all(cs_set=CS_SET, ortho=ORTHO_BASIS, tm2_stride=TM2_STRIDE, tm1_w=TM1_W, nv=NV):
    W3 = front_kernels(cs_set)
    masks = []
    for k in range(24):
        masks += letter_variants(k, nvar=nv)
    masks = np.stack(masks).astype(np.float64)
    V = masks.shape[0]
    F3T = cenorm(conv2(masks[:, None], W3, pad=KBIG // 2))     # (V,C,28,28)
    C = F3T.shape[1]
    a2T = avgpool_hw(F3T, 4)                                    # (V,C,7,7)
    tap = taper2d(7)
    T2 = np.stack([((a2T[i]) * tap[None]).reshape(-1) for i in range(V)])
    T2 = T2 / np.maximum(np.linalg.norm(T2, axis=1, keepdims=True), 1e-9)
    if ortho:
        T2 = remove_basis(T2, [T2.mean(0)])

    def paired(T):
        Tf = np.zeros((2 * V,) + T.shape[1:]); Tf[0::2] = T; Tf[1::2] = -T
        return Tf

    w = {}
    w["f3.W"] = W3.astype(np.float32);                     w["f3.b"] = np.zeros(C, np.float32)
    if tm1_w > 0:
        a1T = avgpool_hw(F3T, 2)
        tap1 = taper2d(14)
        T1 = np.stack([((a1T[i]) * tap1[None]).reshape(-1) for i in range(V)])
        T1 = T1 / np.maximum(np.linalg.norm(T1, axis=1, keepdims=True), 1e-9)
        if ortho:
            T1 = remove_basis(T1, [T1.mean(0)])
        w["tm1.W"] = paired(T1.reshape(V, C, 14, 14) * tm1_w).astype(np.float32)
        w["tm1.b"] = np.zeros(2 * V, np.float32)
    w["tm2.W"] = paired(T2.reshape(V, C, 7, 7)).astype(np.float32)
    w["tm2.b"] = np.zeros(2 * V, np.float32)
    m2s = 3 if tm2_stride == 2 else 5
    layers = [
        {"name": "f3",  "op": "conv", "inputs": ["image"], "out": C, "kernel": [KBIG, KBIG], "stride": 1, "padding": KBIG // 2},
        {"name": "n1",  "op": "norm", "inputs": ["f3"]},
        {"name": "a2",  "op": "avgpool", "inputs": ["n1"], "size": [4, 4], "stride": [4, 4]},
        {"name": "tm2", "op": "conv", "inputs": ["a2"], "out": 2 * V, "kernel": [7, 7], "stride": tm2_stride, "padding": 2},
        {"name": "m2",  "op": "maxpool", "inputs": ["tm2"], "size": [m2s, m2s], "stride": [m2s, m2s]},
        *([{"name": "a1",  "op": "avgpool", "inputs": ["n1"], "size": [2, 2], "stride": [2, 2]},
           {"name": "tm1", "op": "conv", "inputs": ["a1"], "out": 2 * V, "kernel": [14, 14], "stride": 2, "padding": 2},
           {"name": "m1", "op": "maxpool", "inputs": ["tm1"], "size": [3, 3], "stride": [3, 3]}] if tm1_w > 0 else []),
        {"name": "mm",  "op": "add", "inputs": ["m2"] + (["m1"] if tm1_w > 0 else [])},
        {"name": "sr",  "op": "reshape", "inputs": ["mm"], "shape": [V, 1, 2]},
        {"name": "mx",  "op": "maxpool", "inputs": ["sr"], "size": [1, 2], "stride": [1, 2]},
        {"name": "sf",  "op": "reshape", "inputs": ["mx"], "shape": [V]},
        {"name": "out", "op": "dense", "inputs": ["sf"], "out": 24},
    ]
    Wd = np.zeros((24, V), np.float32)
    per = V // 24
    for k in range(24):
        Wd[k, per * k:per * k + per] = 1.0 / per
    w["out.W"] = Wd;                                        w["out.b"] = np.zeros(24, np.float32)
    return {"layers": layers}, w

def main(dest, **kw):
    import pathlib
    net, w = build_all(**kw)
    d = pathlib.Path(dest); d.mkdir(parents=True, exist_ok=True)
    (d / "network.json").write_text(json.dumps(net))
    np.savez(d / "weights.npz", **w)
    print("wrote network; total weights:", sum(int(np.prod(v.shape)) for v in w.values()))

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
