"""Parametric hand renderer (numpy only). Hand frame: x right, y up, units = pixels of the 28x28 image at unit scale,
origin at the palm centre. Shapes are typed from general knowledge of the ASL manual alphabet (right hand, palm toward
the viewer unless the letter is shown from the side), not from any training image."""
import numpy as np

SS = 4            # supersampling factor
EDGE = 0.2        # softness (in pixels at unit scale) of shape edges

# brightness levels (0..1) for the two lighting situations; creases stay dark in both
LIGHT = dict(bg=0.55, hand=0.80, fing=0.80, arm=0.72, gap=0.30, hole=0.55)
DARK = dict(bg=0.85, hand=0.60, fing=0.60, arm=0.52, gap=0.28, hole=0.85)
SHADE = 0.12      # left-to-right shading of the hand: left side brighter (light from the signer's right / viewer's left)

# material tags
HAND, FING, ARM, GAP, HOLE = "hand", "fing", "arm", "gap", "hole"


def cov_from_dist(d):
    return np.clip(0.5 - d / EDGE, 0.0, 1.0)


def sd_capsule(X, Y, p0, p1, r):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return np.hypot(X - x0, Y - y0) - r
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0.0, 1.0)
    return np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy)) - r


def sd_ellipse(X, Y, c, a, b, ang_deg=0.0):
    th = np.deg2rad(ang_deg)
    u = (X - c[0]) * np.cos(th) + (Y - c[1]) * np.sin(th)
    v = -(X - c[0]) * np.sin(th) + (Y - c[1]) * np.cos(th)
    rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    return (rho - 1.0) * min(a, b)


def sd_rrect(X, Y, c, hw, hh, r, ang_deg=0.0):
    th = np.deg2rad(ang_deg)
    u = (X - c[0]) * np.cos(th) + (Y - c[1]) * np.sin(th)
    v = -(X - c[0]) * np.sin(th) + (Y - c[1]) * np.cos(th)
    qx = np.abs(u) - (hw - r)
    qy = np.abs(v) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def arc(c, r, a0, a1, n=12):
    th = np.deg2rad(np.linspace(a0, a1, n))
    return [(c[0] + r * np.cos(t), c[1] + r * np.sin(t)) for t in th]


# ---------------------------------------------------------------- primitives: (kind, material, params)
def P_cap(p0, p1, r, m=HAND):
    return ("cap", m, (p0, p1, r))


def P_ell(c, a, b, ang=0.0, m=HAND):
    return ("ell", m, (c, a, b, ang))


def P_rr(c, hw, hh, r, ang=0.0, m=HAND):
    return ("rr", m, (c, hw, hh, r, ang))


def P_gap(p0, p1, r=0.38):
    return ("cap", GAP, (p0, p1, r))


def chain(pts, r, m=HAND):
    return [P_cap(pts[i], pts[i + 1], r, m) for i in range(len(pts) - 1)]


def seg(p, ang_deg, L):
    t = np.deg2rad(ang_deg)
    return (p[0] + L * np.cos(t), p[1] + L * np.sin(t))


# finger bases on the knuckle row (frontal view), x: pinky (left) .. index (right)
BASE = {"index": (4.1, 5.0), "middle": (1.4, 5.4), "ring": (-1.3, 5.1), "pinky": (-4.0, 4.3)}
LEN = {"index": 10.0, "middle": 11.0, "ring": 10.0, "pinky": 7.8}
RAD = {"index": 1.3, "middle": 1.3, "ring": 1.25, "pinky": 1.15}
ORDER = ["pinky", "ring", "middle", "index"]


def palm(hw=5.6, hh=5.9, c=(0.0, -0.3), ang=0.0):
    return [P_rr(c, hw, hh, 3.0, ang)]


def forearm(p0=(0.0, -5.0), ang=-90.0, r=5.4, L=30.0):
    return [P_cap(p0, seg(p0, ang, L), r, ARM)]


def ext_finger(name, lean=0.0, L=None, r=None, base=None):
    """extended straight finger; lean = degrees from vertical, positive leans right."""
    b = base or BASE[name]
    L = L or LEN[name]
    r = r or RAD[name]
    return [P_cap(b, seg(b, 90.0 - lean, L), r, FING)]


def curled(name, L=5.0, r=1.3, top=1.0, lean=0.0, base=None):
    """curled finger of a fist: short stub pointing down from the knuckle row"""
    b = base or BASE[name]
    p0 = seg(b, 90.0 - lean, top)
    p1 = seg(b, -90.0 - lean, L - top)
    return [P_cap(p0, p1, r)]


def stub_gaps(names, L=4.5, top=0.6, lean=0.0):
    out = []
    for i in range(len(names) - 1):
        a, b = BASE[names[i]], BASE[names[i + 1]]
        m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        out.append(P_gap(seg(m, 90 - lean, top), seg(m, -90 - lean, L - top)))
    return out


def finger_gap(nameA, nameB, leanA, leanB, frac=0.9, start=-0.5):
    """dark crease between two touching/near fingers"""
    a, b = BASE[nameA], BASE[nameB]
    La, Lb = LEN[nameA] * frac, LEN[nameB] * frac
    ea, eb = seg(a, 90 - leanA, La), seg(b, 90 - leanB, Lb)
    m0 = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + start)
    m1 = ((ea[0] + eb[0]) / 2, (ea[1] + eb[1]) / 2)
    return [P_gap(m0, m1)]


def fist(stub_L=5.0, lean=0.0):
    out = palm()
    for n in ORDER:
        out += curled(n, L=stub_L, lean=lean)
    out += stub_gaps(ORDER, L=stub_L - 0.5, lean=lean)
    return out


def thumb_side():
    return [P_cap((5.6, -3.0), (6.4, 4.4), 1.5, FING), P_gap((4.9, -2.0), (5.2, 3.8), 0.34)]


def thumb_out(ang=15.0, L=8.5):
    return [P_cap((5.0, -1.5), seg((5.0, -1.5), ang, L), 1.5, FING)]


def thumb_across():
    """thumb folded over the palm: a bump at the right edge and a diagonal crease across the lower palm"""
    return [P_cap((6.0, -1.5), (3.5, 0.5), 1.6, FING), P_gap((5.8, -3.6), (1.8, 1.2), 0.36)]


# ---------------------------------------------------------------- letters: each returns a list of primitives
def L_A(v=0):
    if v == 0:
        return forearm() + fist() + thumb_side()
    return forearm() + fist(stub_L=5.5, lean=-5) + [P_cap((5.4, -3.0), (6.6, 5.0), 1.5, FING), P_gap((4.6, -2.0), (5.4, 4.4), 0.34)]


def L_B(v=0):
    leans = {"index": 3.0, "middle": 0.0, "ring": -3.0, "pinky": -7.0} if v == 0 else {"index": 6.0, "middle": 2.0, "ring": -2.0, "pinky": -9.0}
    out = forearm() + palm(hw=5.9, hh=6.0)
    for n in ORDER:
        out += ext_finger(n, lean=leans[n])
    for i in range(3):
        out += finger_gap(ORDER[i], ORDER[i + 1], leans[ORDER[i]], leans[ORDER[i + 1]])
    out += thumb_across()
    return out


def L_C(v=0):
    """side view: back of the hand is a thick band rising from the wrist, fingers hook over the top to the right,
    short thumb at the bottom right; wide opening on the right."""
    tilt = 10.0 if v == 0 else 18.0            # lean of the band to the right (degrees from vertical)
    out = forearm(p0=(-1.0, -6.0), ang=-90.0 - tilt * 0.5, r=5.2)
    base = (-1.0, -6.0)
    top = seg(base, 90.0 - tilt, 13.0)
    out += [P_cap(base, top, 3.6)]                                    # back of hand / palm band
    c = (top[0] + 4.5, top[1] - 4.0)                                  # centre of the finger hook
    out += chain(arc(c, 5.0, 150.0, 60.0 - (0 if v == 0 else 15), 10), 2.6, FING)   # fingers curving over the top
    out += chain(arc(c, 5.0, 60.0 - (0 if v == 0 else 15), 10.0, 6), 2.2, FING)     # finger tips coming down
    for ang in (120.0, 95.0, 70.0, 45.0):
        out.append(P_gap(seg(c, ang, 2.6), seg(c, ang, 7.8), 0.34))
    tb = (base[0] + 3.5, base[1] + 5.5)
    out += [P_cap(tb, seg(tb, 25.0, 6.5), 1.6, FING)]                        # thumb pointing to the opening
    return out


def L_D(v=0):
    out = forearm() + palm()
    out += ext_finger("index", lean=3.0 if v == 0 else -3.0)
    for n in ("middle", "ring", "pinky"):
        out += curled(n, L=6.0, lean=-8.0)
    out += stub_gaps(["pinky", "ring", "middle"], L=5.5, lean=-8.0)
    out += [P_cap((5.8, -2.5), (3.0, 2.0), 1.4, FING), P_gap((2.2, 0.8), (3.4, 2.8), 0.34)]
    return out


def L_E(v=0):
    out = forearm() + palm(hw=5.8, hh=6.0)
    L, top = (4.5, 1.4) if v == 0 else (4.0, 1.8)
    for n in ORDER:
        out += curled(n, L=L, top=top)
    out += stub_gaps(ORDER, L=L - 0.5, top=top - 0.3)
    out += [P_gap((-5.0, 2.0), (5.2, 2.4), 0.34), P_cap((6.0, -2.0), (2.5, -1.0), 1.4, FING), P_gap((5.6, -0.6), (2.0, 0.2), 0.3)]
    return out


def L_F(v=0):
    out = forearm() + palm()
    leans = {"middle": 2.0, "ring": -12.0, "pinky": -28.0} if v == 0 else {"middle": -4.0, "ring": -18.0, "pinky": -36.0}
    for n in ("middle", "ring", "pinky"):
        out += ext_finger(n, lean=leans[n])
    out += finger_gap("ring", "middle", leans["ring"], leans["middle"], 0.55)
    loop_c = (5.6, 1.2)
    out += chain(arc(loop_c, 2.8, 0.0, 360.0, 20), 1.35, FING)
    out += [("cap", HOLE, (loop_c, loop_c, 1.3))]
    return out


def L_G(v=0):
    out = forearm(p0=(-3.0, -3.0), ang=-105.0, r=5.0)
    out += [P_rr((-2.0, 0.5), 5.0, 4.6, 2.5)]
    tilt = 2.0 if v == 0 else 8.0
    out += [P_cap((1.5, 2.4), seg((1.5, 2.4), tilt, 11.0), 1.4, FING)]          # index
    out += [P_cap((-0.5, 4.8), seg((-0.5, 4.8), tilt, 7.5), 1.35, FING)]        # thumb on top
    out += [P_gap((0.5, 3.7), seg((0.5, 3.7), tilt, 7.0), 0.34)]
    out += [P_gap((-5.0, -1.5), (3.0, -1.0), 0.34)]                        # crease of the curled fingers
    return out


def L_H(v=0):
    out = forearm(p0=(-3.0, -3.0), ang=-105.0, r=5.0)
    out += [P_rr((-2.5, 0.5), 5.0, 4.6, 2.5)]
    tilt = 2.0 if v == 0 else 8.0
    out += [P_cap((1.5, 3.4), seg((1.5, 3.4), tilt, 11.0), 1.3, FING)]
    out += [P_cap((1.5, 0.9), seg((1.5, 0.9), tilt, 11.0), 1.3, FING)]
    out += [P_gap((1.0, 2.15), seg((1.0, 2.15), tilt, 10.0), 0.36)]
    out += [P_gap((-5.0, -1.8), (2.5, -1.4), 0.34)]
    return out


def L_I(v=0):
    out = forearm() + palm()
    for n in ("ring", "middle", "index"):
        out += curled(n)
    out += stub_gaps(["ring", "middle", "index"], L=4.5)
    out += [P_cap(BASE["pinky"], seg(BASE["pinky"], 90 + (3 if v == 0 else 10), 8.5), 1.15, FING)]
    out += thumb_across()
    return out


def L_K(v=0):
    out = forearm() + palm(hw=5.0, hh=5.9, c=(0.3, -0.3))
    lean = 5.0 if v == 0 else 12.0
    out += [P_cap((1.8, 5.2), seg((1.8, 5.2), 90.0 - lean, 10.5), 1.3, FING)]          # index up
    out += [P_cap((3.2, 4.8), seg((3.2, 4.8), 48.0 - lean, 9.0), 1.3, FING)]           # middle slanted
    out += [P_cap((3.8, -0.5), (4.6, 5.3), 1.4, FING)]                                   # thumb between
    out += [P_gap((2.6, 1.5), (3.0, 6.2), 0.34)]
    out += curled("pinky", L=4.5) + curled("ring", L=4.5) + stub_gaps(["pinky", "ring"], L=4.0)
    return out


def L_L(v=0):
    out = forearm() + palm()
    out += ext_finger("index", lean=2.0 if v == 0 else -4.0, L=10.5)
    for n in ("middle", "ring", "pinky"):
        out += curled(n)
    out += stub_gaps(["pinky", "ring", "middle"], L=4.5)
    out += thumb_out(ang=10.0 if v == 0 else 22.0, L=9.0)
    out += [P_gap((5.2, 3.2), (5.4, 0.0), 0.34)]
    return out


def L_O(v=0):
    out = forearm()
    ang = 12.0 if v == 0 else 0.0
    out += [P_ell((0.5, 0.5), 5.8, 6.8, ang)]
    out += [("ell", HOLE, ((2.6, 1.8), 1.7, 2.1, ang))]
    out += [P_gap((-4.8, -0.5), (-1.2, 6.2), 0.34), P_gap((-1.2, 6.2), (3.6, 5.6), 0.34), P_gap((0.0, 2.0), (-3.0, 0.0), 0.3)]
    return out


def L_P(v=0):
    out = forearm(p0=(-3.5, 0.5), ang=-118.0, r=5.0)
    out += [P_rr((-1.5, 3.0), 5.0, 4.0, 2.5, -15.0)]
    if v == 0:
        out += [P_cap((2.5, 4.0), seg((2.5, 4.0), -8.0, 10.5), 1.35, FING)]    # index to the right, slightly down
        out += [P_cap((2.0, 2.0), seg((2.0, 2.0), -80.0, 9.5), 1.3, FING)]     # middle hanging down
        out += [P_cap((0.0, 1.0), seg((0.0, 1.0), -60.0, 5.0), 1.4, FING)]     # thumb
    else:
        out += [P_cap((2.5, 3.5), seg((2.5, 3.5), -25.0, 10.5), 1.35, FING)]
        out += [P_cap((1.5, 1.5), seg((1.5, 1.5), -88.0, 9.5), 1.3, FING)]
        out += [P_cap((-0.5, 0.5), seg((-0.5, 0.5), -65.0, 5.0), 1.4, FING)]
    out += [P_gap((-4.5, 1.5), (1.5, 0.5), 0.34)]
    return out


def L_Q(v=0):
    out = forearm(p0=(0.5, 6.0), ang=80.0 if v == 0 else 95.0, r=5.0)
    out += [P_rr((0.0, 4.0), 5.4, 4.6, 2.5)]
    out += [P_cap((-1.2, 1.5), seg((-1.2, 1.5), -92.0, 10.5), 1.35, FING)]     # index down
    out += [P_cap((2.8, 1.0), seg((2.8, 1.0), -85.0, 7.0), 1.4, FING)]         # thumb down, shorter
    out += [P_gap((0.8, 1.5), (1.0, -4.8), 0.36), P_gap((-4.5, 1.5), (-2.0, 1.0), 0.34)]
    return out


def L_R(v=0):
    out = forearm() + palm(hw=5.2, hh=5.9, c=(0.2, -0.3))
    lean = 10.0 if v == 0 else 16.0
    a0, a1 = (3.2, 5.0), (1.0, 5.3)
    # two crossed fingers, both leaning right, crossing about half-way up
    out += [P_cap(a0, seg(a0, 90 - lean + 9, 11.0), 1.2, FING), P_cap(a1, seg(a1, 90 - lean - 9, 11.0), 1.2, FING)]
    out += [P_gap(seg(a1, 90 - lean - 9, 6.5), seg(a1, 90 - lean - 9, 10.5), 0.3)]
    for n in ("ring", "pinky"):
        out += curled(n)
    out += stub_gaps(["pinky", "ring"], L=4.5)
    out += thumb_across()
    return out


def L_U(v=0):
    out = forearm() + palm(hw=5.4, hh=5.9, c=(0.2, -0.3))
    lean = 8.0 if v == 0 else 15.0
    out += [P_cap((3.2, 5.0), seg((3.2, 5.0), 90 - lean, 11.0), 1.3, FING)]
    out += [P_cap((0.7, 5.3), seg((0.7, 5.3), 90 - lean, 11.3), 1.3, FING)]
    out += [P_gap((1.95, 4.8), seg((1.95, 4.8), 90 - lean, 10.5), 0.36)]
    for n in ("ring", "pinky"):
        out += curled(n)
    out += stub_gaps(["pinky", "ring"], L=4.5)
    out += thumb_across()
    return out


def L_V(v=0):
    out = forearm() + palm()
    s = 14.0 if v == 0 else 20.0
    out += ext_finger("index", lean=s, L=10.5)
    out += ext_finger("middle", lean=-s + 3, L=11.0)
    for n in ("ring", "pinky"):
        out += curled(n)
    out += stub_gaps(["pinky", "ring"], L=4.5)
    out += thumb_across()
    return out


def L_W(v=0):
    out = forearm() + palm()
    s = 15.0 if v == 0 else 21.0
    out += ext_finger("index", lean=s, L=10.0)
    out += ext_finger("middle", lean=0.0, L=11.0)
    out += ext_finger("ring", lean=-s + 2, L=10.0)
    out += curled("pinky", L=4.5)
    out += thumb_across()
    return out


def L_X(v=0):
    out = forearm() + palm()
    b = BASE["index"]
    k = (b[0] + 0.3, b[1] + 5.8) if v == 0 else (b[0] - 0.4, b[1] + 6.4)
    out += [P_cap(b, k, 1.3, FING), P_cap(k, (k[0] + 3.0, k[1] - 1.6), 1.15, FING)]
    for n in ("middle", "ring", "pinky"):
        out += curled(n)
    out += stub_gaps(["pinky", "ring", "middle"], L=4.5)
    out += thumb_side()
    return out


def L_Y(v=0):
    out = forearm() + palm()
    for n in ("ring", "middle", "index"):
        out += curled(n)
    out += stub_gaps(["ring", "middle", "index"], L=4.5)
    pa = 112.0 if v == 0 else 124.0
    out += [P_cap(BASE["pinky"], seg(BASE["pinky"], pa, 8.5), 1.15, FING)]
    out += thumb_out(ang=18.0 if v == 0 else 30.0, L=9.0)
    return out


def L_M(v=0):
    lean = -16.0 if v == 0 else -6.0
    out = forearm() + palm()
    for n in ("ring", "middle", "index"):
        out += curled(n, L=6.5, top=1.2, lean=lean)
    out += curled("pinky", L=4.5, top=0.4, lean=lean)
    out += stub_gaps(ORDER, L=6.0, top=1.0, lean=lean)
    out += [P_cap((5.8, -2.5), (3.5, -1.0), 1.4)]
    return out


def L_N(v=0):
    lean = -10.0 if v == 0 else 0.0
    out = forearm() + palm()
    for n in ("middle", "index"):
        out += curled(n, L=6.5, top=1.2, lean=lean)
    for n in ("ring", "pinky"):
        out += curled(n, L=4.6, top=0.4, lean=lean)
    out += stub_gaps(ORDER, L=5.5, top=1.0, lean=lean)
    out += [P_cap((5.8, -2.5), (3.5, -1.0), 1.4)]
    return out


def L_S(v=0):
    out = forearm() + fist(stub_L=5.0)
    y = 1.0 if v == 0 else 1.8
    out += [P_gap((6.4, y + 1.9), (-1.8, y + 2.5), 0.34), P_cap((6.2, y), (-1.2, y + 0.6), 1.45, FING), P_gap((6.4, y - 1.9), (-1.8, y - 1.3), 0.34)]
    return out


def L_T(v=0):
    out = forearm(p0=(-2.5, -3.0), ang=-105.0, r=5.0)
    out += [P_rr((-1.5, 0.5), 5.2, 4.8, 2.5)]
    tilt = 4.0 if v == 0 else 12.0
    out += [P_cap((1.5, 2.8), seg((1.5, 2.8), tilt, 7.5), 1.4, FING)]        # index wrapped, pointing right
    out += [P_cap((-2.0, 3.5), seg((-2.0, 3.5), 95.0, 6.0), 1.4, FING)]      # thumb tip poking up
    out += [P_gap((0.3, 3.0), (-0.5, 6.2), 0.34), P_gap((-5.0, -1.5), (3.0, -1.0), 0.34)]
    return out


LETTERS = [L_A, L_B, L_C, L_D, L_E, L_F, L_G, L_H, L_I, L_K, L_L, L_M, L_N, L_O, L_P, L_Q, L_R, L_S, L_T, L_U, L_V, L_W, L_X, L_Y]
NAMES = list("ABCDEFGHIKLMNOPQRSTUVWXY")


def render(prims, cx, cy, scale=1.0, rot=0.0, size=28, levels=LIGHT):
    """Render primitives (hand frame) into a size x size image. (cx, cy) = image position (col,row) of the hand origin.
    rot: rotation in degrees (positive = counter-clockwise on screen)."""
    n = size * SS
    g = (np.arange(n) + 0.5) / SS
    XI, YI = np.meshgrid(g, g)
    u = (XI - cx) / scale
    v = -(YI - cy) / scale
    th = np.deg2rad(rot)
    X = u * np.cos(th) + v * np.sin(th)
    Y = -u * np.sin(th) + v * np.cos(th)
    shade = 1.0 + SHADE * np.clip(-X / 5.0, -1.0, 1.0)          # left of the hand brighter than the right
    img = np.full((n, n), levels["bg"], dtype=np.float64)
    for kind, mat, pr in prims:
        if kind == "cap":
            d = sd_capsule(X, Y, pr[0], pr[1], pr[2])
        elif kind == "ell":
            d = sd_ellipse(X, Y, *pr)
        else:
            d = sd_rrect(X, Y, *pr)
        cov = cov_from_dist(d)
        val = levels[mat]
        if mat in (HAND, FING, ARM):
            val = val * shade
        img = img * (1 - cov) + val * cov
    return img.reshape(size, SS, size, SS).mean(axis=(1, 3))


def dilate(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out = np.maximum(out, np.roll(np.roll(m, dy, 0), dx, 1))
    return out


def emphasis_map(prims, cx, cy, scale=1.0, rot=0.0, size=28, w_bg=0.25, w_hand=0.6, w_fing=1.0, r=2):
    """hand-chosen importance of each pixel for a template: distinctive parts (extended fingers, thumbs, hooks,
    holes) 1.0, the rest of the hand 0.6, forearm and background 0.25.  Built from the render's own geometry."""
    z = dict(bg=0.0, hand=0.0, fing=0.0, arm=0.0, gap=0.0, hole=0.0)
    lh = dict(z, hand=1.0, fing=1.0)
    lf = dict(z, fing=1.0, hole=1.0)
    hand = dilate(render([p for p in prims if p[1] in (HAND, FING)], cx, cy, scale, rot, size, lh) > 0.5, r)
    fing = dilate(render([p for p in prims if p[1] in (FING, HOLE)], cx, cy, scale, rot, size, lf) > 0.5, r)
    return np.where(fing, w_fing, np.where(hand, w_hand, w_bg)).astype(np.float64)


def hand_bbox(prims, cx, cy, scale=1.0, rot=0.0, size=28):
    """bounding box (rows/cols) of the hand (forearm excluded), from a render of the hand parts only"""
    hp = [p for p in prims if p[1] in (HAND, FING)]
    im = render(hp, cx, cy, scale, rot, size, levels=dict(bg=0.0, hand=1.0, fing=1.0, arm=1.0, gap=1.0, hole=1.0))
    rows, cols = np.where(im > 0.5)
    return rows.min(), rows.max(), cols.min(), cols.max()


# ====================================================================================================================
#  Network builder.  Everything below turns the renders above into weights.  No data files are read.
# ====================================================================================================================
import json, os, sys

NCLS = 24
ROTS = (-12.0, 0.0, 12.0)            # whole-hand rotations (degrees) per pose
SCALES = (1.06, 1.2, 1.34)           # whole-hand scales per pose (1.2 = nominal: palm about 13 px wide)
VARIANTS = (0, 1)                    # typed pose variants per letter
TOP_ROWS = (1.0, 4.0)                # row of the top of the hand after framing (shifts of +-2 px come from the conv)
X_CENTRE = 13.5
W_PIX = 0.5                          # weight of the pixel-correlation branch (hand-chosen)
W_STR = 1.0                          # weight of the bar/edge structure branch (hand-chosen)
NBAR = 8                             # bar (even Gabor) orientations over 180 degrees, at two widths
NEDGE = 4                            # edge (Gaussian derivative) orientations over 180 degrees
SAT = 0.3                            # saturation level of filter responses, in units of image standard deviations
GATE_K = 2.0                         # strength of the hand-polarity gate
SIG_BLUR = 1.0
EPS = 1e-5
KS = 13                              # filter size
NCELL = 12                           # cells per side after pooling


def gauss_kernel(size, sigma):
    r = (size - 1) / 2
    g = np.arange(size) - r
    k = np.exp(-(g[:, None] ** 2 + g[None, :] ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def filter_bank():
    """20 signed filters (13x13): 8 even-Gabor bar detectors tuned to one finger width, 8 tuned to two fingers
    side by side, 4 Gaussian-derivative edge detectors."""
    r = (KS - 1) / 2
    g = np.arange(KS) - r
    X, Y = np.meshgrid(g, g)                   # X: column offset, Y: row offset
    fs = []
    for lam, sa, sl in ((5.5, 1.5, 2.0), (10.0, 2.8, 3.0)):   # wavelength (~2 x bar width), across / along sigmas
        for k in range(NBAR):
            th = np.pi * k / NBAR
            xa = X * np.cos(th) + Y * np.sin(th)   # across the bar
            yl = -X * np.sin(th) + Y * np.cos(th)  # along the bar
            env = np.exp(-xa ** 2 / (2 * sa ** 2) - yl ** 2 / (2 * sl ** 2))
            f = env * np.cos(2 * np.pi * xa / lam)
            f = f - env * (f.sum() / env.sum())    # zero mean, keeping the envelope shape
            fs.append(f / np.abs(f).sum())
    sig = 1.2
    G = np.exp(-(X ** 2 + Y ** 2) / (2 * sig ** 2))
    for e in range(NEDGE):
        th = np.pi * e / NEDGE
        d = X * np.cos(th) + Y * np.sin(th)
        f = -d / sig ** 2 * G
        f = f - f.mean()
        fs.append(f / np.abs(f).sum())
    return np.array(fs)                        # (20, 13, 13)


def structure_weights():
    """conv1 (1->80): [+f_j (bias 0)] [-f_j (bias 0)] [+f_j (bias -SAT)] [-f_j (bias -SAT)];
    mix (80->36, 1x1): 16 bright bars, 16 dark bars, 4 unsigned edges, each response saturating at SAT."""
    fs = filter_bank()
    nf = len(fs)                               # 20
    nb = 2 * NBAR                              # 16 bar filters (two widths)
    W1 = np.concatenate([fs, -fs, fs, -fs], axis=0)[:, None]     # (80, 1, 13, 13)
    b1 = np.concatenate([np.zeros(2 * nf), -SAT * np.ones(2 * nf)])
    NOUT = 2 * nb + NEDGE                      # 36
    Wm = np.zeros((NOUT, 4 * nf))
    for k in range(nb):
        Wm[k, k] = 1.0;              Wm[k, 2 * nf + k] = -1.0               # min(relu(+f), SAT)  bright bar
        Wm[nb + k, nf + k] = 1.0;    Wm[nb + k, 3 * nf + k] = -1.0          # min(relu(-f), SAT)  dark bar
    for e in range(NEDGE):
        j = nb + e
        Wm[2 * nb + e, j] = 1.0;      Wm[2 * nb + e, 2 * nf + j] = -1.0
        Wm[2 * nb + e, nf + j] = 1.0; Wm[2 * nb + e, 3 * nf + j] = -1.0
    return W1, b1, Wm[:, :, None, None], np.zeros(NOUT)


def gate_masks():
    """polarity feature: mean of the central hand region minus mean of the top/side border (bottom excluded: forearm)."""
    rr, cc = np.mgrid[0:28, 0:28]
    centre = (((cc - 13.5) / 7.0) ** 2 + ((rr - 13.5) / 8.0) ** 2) <= 1.0
    border = ((rr < 3) | (cc < 3) | (cc > 24)) & (rr < 22)
    m = centre / centre.sum() - border / border.sum()
    return m


# ---- a tiny reference implementation of the ops (same formulas as the scorer's interpreter) ----
def op_norm(x):
    mu = x.mean(axis=(1, 2), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(1, 2), keepdims=True) + EPS)


def op_conv(x, W, b, stride, padding):
    c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=1)
    out = W.reshape(o, -1) @ cols.reshape(c * kh * kw, ho * wo)
    return out.reshape(o, ho, wo) + b[:, None, None]


def op_pool(x, size, stride, kind):
    c, h, w = x.shape
    ho, wo = (h - size) // stride + 1, (w - size) // stride + 1
    win = np.stack([x[:, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(size) for dx in range(size)], axis=0)
    return win.max(0) if kind == "max" else win.mean(0)


class FrontEnd:
    def __init__(self):
        self.blur = gauss_kernel(5, SIG_BLUR)[None, None].astype(np.float64)
        self.W1, self.b1, self.Wm, self.bm = structure_weights()
        self.zero1 = np.zeros(1)

    def pixel_vec(self, img):
        """norm -> blur -> norm, flattened (784,), zero mean unit variance"""
        n0 = op_norm(img[None])
        bl = op_conv(n0, self.blur, self.zero1, 1, 2)
        return op_norm(bl).reshape(-1)

    def structure_map(self, img):
        """norm -> filters -> relu -> saturating mix -> maxpool 3/1 -> avgpool 4/2 -> joint norm; (36, 12, 12)"""
        n0 = op_norm(img[None])
        a = np.maximum(op_conv(n0, self.W1, self.b1, 1, (KS - 1) // 2), 0.0)
        s = op_conv(a, self.Wm, self.bm, 1, 0)
        s = op_pool(s, 3, 1, "max")
        cells = op_pool(s, 4, 2, "avg")                    # (36, 12, 12)
        nout = cells.shape[0]
        flat = cells.reshape(1, nout, -1)
        return op_norm(flat).reshape(nout, NCELL, NCELL)

    @staticmethod
    def cell_weights(emph):
        """the emphasis map pooled the same way as the features (28x28 -> 12x12)"""
        return op_pool(op_pool(emph[None], 3, 1, "avg"), 4, 2, "avg")[0]


def weighted_template(t, w):
    """weighted, zero-sum, unit-norm template: tau_i = w_i (t_i - mu_w), mu_w = sum(w t)/sum(w)"""
    mu = (w * t).sum() / w.sum()
    tau = w * (t - mu)
    return tau / np.sqrt((tau ** 2).sum())


def placement(prims, scale, rot, top):
    """hand origin (cx, cy) such that the top of the hand (forearm excluded) is at row `top`, centred horizontally"""
    big = 64
    r0, r1, c0, c1 = hand_bbox(prims, big / 2, big / 2, scale, rot, big)
    return big / 2 + (X_CENTRE - (c0 + c1 + 1) / 2), big / 2 + (top - r0)


def framed_render(prims, scale, rot, levels=LIGHT, top=2.5):
    cx, cy = placement(prims, scale, rot, top)
    return render(prims, cx, cy, scale, rot, 28, levels)


def build_templates():
    fe = FrontEnd()
    pix, struct, dark = [], [], []
    per_class = len(VARIANTS) * len(ROTS) * len(SCALES) * len(TOP_ROWS) * 2
    for k, letter in enumerate(LETTERS):
        for v in VARIANTS:
            prims = letter(v)
            for rot in ROTS:
                for sc in SCALES:
                    for top in TOP_ROWS:
                        cx, cy = placement(prims, sc, rot, top)
                        emph = emphasis_map(prims, cx, cy, sc, rot, 28)
                        wcell = fe.cell_weights(emph)
                        for flag, levels in ((0, LIGHT), (1, DARK)):   # hand lighter / darker than the background
                            img = render(prims, cx, cy, sc, rot, 28, levels)
                            pix.append(weighted_template(fe.pixel_vec(img), emph.reshape(-1)))
                            f = fe.structure_map(img)                  # (36, 12, 12)
                            struct.append(weighted_template(f, np.broadcast_to(wcell, f.shape)))
                            dark.append(flag)
    return np.array(pix), np.array(struct), np.array(dark), per_class


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fe = FrontEnd()
    pix, struct, dark, M = build_templates()
    T = pix.shape[0]
    assert T == NCLS * M
    nfeat = struct[0].size                                 # 5184
    nstr = struct.shape[1]                                 # 36
    weights = {}
    weights["blur.W"] = fe.blur.astype(np.float32)
    weights["blur.b"] = np.zeros(1, np.float32)
    weights["filt.W"] = fe.W1.astype(np.float32)
    weights["filt.b"] = fe.b1.astype(np.float32)
    weights["mix.W"] = fe.Wm.astype(np.float32)
    weights["mix.b"] = fe.bm.astype(np.float32)
    # pixel branch: input has zero mean, unit variance over 784 values (norm 28); template is unit-norm and zero-sum.
    weights["pix.W"] = (W_PIX * pix / 28.0).reshape(T, 1, 28, 28).astype(np.float32)
    weights["pix.b"] = np.zeros(T, np.float32)
    # structure branch: nfeat jointly normalised values (norm sqrt(nfeat)); template is unit-norm and zero-sum.
    weights["str.W"] = (W_STR * struct / np.sqrt(float(nfeat))).astype(np.float32)
    weights["str.b"] = np.zeros(T, np.float32)
    # polarity gate: channel 0 = relu(-K p) (penalty for light-hand templates), channel 1 = relu(+K p) (for dark ones)
    m = gate_masks()
    weights["pol.W"] = np.stack([-GATE_K * m, GATE_K * m])[:, None].astype(np.float32)
    weights["pol.b"] = np.zeros(2, np.float32)
    Wg = np.zeros((T, 2, 1, 1), np.float32)
    Wg[dark == 0, 0, 0, 0] = -1.0
    Wg[dark == 1, 1, 0, 0] = -1.0
    weights["gate.W"] = Wg
    weights["gate.b"] = np.zeros(T, np.float32)
    layers = [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "blur", "op": "conv", "inputs": ["n0"], "out": 1, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "n1", "op": "norm", "inputs": ["blur"]},
        {"name": "pix", "op": "conv", "inputs": ["n1"], "out": T, "kernel": [28, 28], "stride": 2, "padding": 2},
        {"name": "pol", "op": "conv", "inputs": ["n1"], "out": 2, "kernel": [28, 28], "stride": 2, "padding": 2},
        {"name": "polr", "op": "relu", "inputs": ["pol"]},
        {"name": "gate", "op": "conv", "inputs": ["polr"], "out": T, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "filt", "op": "conv", "inputs": ["n0"], "out": int(fe.W1.shape[0]), "kernel": [KS, KS], "stride": 1, "padding": (KS - 1) // 2},
        {"name": "filtr", "op": "relu", "inputs": ["filt"]},
        {"name": "mix", "op": "conv", "inputs": ["filtr"], "out": nstr, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "local", "op": "maxpool", "inputs": ["mix"], "size": [3, 3], "stride": [1, 1]},
        {"name": "cells", "op": "avgpool", "inputs": ["local"], "size": [4, 4], "stride": [2, 2]},
        {"name": "flat", "op": "reshape", "inputs": ["cells"], "shape": [1, nstr, NCELL * NCELL]},
        {"name": "n2", "op": "norm", "inputs": ["flat"]},
        {"name": "back", "op": "reshape", "inputs": ["n2"], "shape": [nstr, NCELL, NCELL]},
        {"name": "str", "op": "conv", "inputs": ["back"], "out": T, "kernel": [NCELL, NCELL], "stride": 1, "padding": 1},
        {"name": "score", "op": "add", "inputs": ["pix", "str", "gate"]},
        {"name": "best_shift", "op": "maxpool", "inputs": ["score"], "size": [3, 3], "stride": [3, 3]},
        {"name": "by_class", "op": "reshape", "inputs": ["best_shift"], "shape": [NCLS, M, 1]},
        {"name": "best_template", "op": "maxpool", "inputs": ["by_class"], "size": [M, 1], "stride": [M, 1]},
        {"name": "out", "op": "reshape", "inputs": ["best_template"], "shape": [NCLS]},
    ]
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)
    print(f"wrote {T} templates ({M} per class), {sum(v.size for v in weights.values()):,} weights")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
