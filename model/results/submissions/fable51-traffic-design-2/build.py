#!/usr/bin/env python3
"""
build.py -- writes network.json and weights.npz for a hand-designed German traffic sign classifier.

No data is read. Every weight is produced from geometric descriptions of the 43 sign classes
(ring widths, triangle borders, digits drawn as strokes, arrows, simple pictograms) plus a
few hand-chosen constants.

Network (all numpy-interpretable ops):
  image (3,32,32)
   -> cmap  : 1x1 conv -> [I, red, blue, yellow, zero] cone tests, relu
   -> selI  : 1x1 conv picks I            -> normI : per-image contrast normalisation (grey)
   -> selC  : 1x1 conv picks all 5 maps   -> reshape to one channel -> normC (joint, brightness invariant) -> reshape back
   -> tA (conv 32x32, pad 2) on normI  +  tB (conv 32x32, pad 2) on colour maps   (prototype correlations at 5x5 shifts)
   -> maxpool over shifts -> reshape -> maxpool over the scale variants of each class -> dense identity readout (43)
"""
import sys, json, math
import numpy as np

N = 32          # image size
SS = 4          # supersampling for rendering
PAD = 3         # shift tolerance (+-3 px)
SCALES = [0.56, 0.64, 0.72, 0.80, 0.88, 0.96, 1.04]   # sign size as a fraction of the frame (real crops vary widely)
ASPECTS = [0.9, 1.0, 1.1]   # width / height of the sign (crops are only roughly square)
VARIANTS = [(f, a) for f in SCALES for a in ASPECTS]
BLUR_SIGMA = 1.1
BLUR_COLOUR = 1.0   # colour templates are blurred more: colour evidence should be tolerant
W_INNER = 2.0   # extra weight of the sign interior (pictogram region) in the whole-sign grey template
LAMBDA_INNER = 0.7  # weight of the separately normalised interior-only pattern inside the sign template
COL_EPS = 0.015  # small threshold on the colour cone tests (sensor noise); casts are handled by the cones
MU_EDGE = 0.8   # weight of the edge-layout evidence
LOG_SIGMA = 1.0  # scale of the centre-surround (Laplacian of Gaussian) filter, in pixels
W_PICT = 0.5    # weight of the pictogram-level evidence (own shift search, rectified)
PAD_P = 3       # shift tolerance of the pictogram search
C_COL = 0.45    # weight of colour evidence
C_YELLOW_BOOST = 1.2  # the yellow diamond is the only yellow sign; its grey structure is weak
C_LIGHTS_BOOST = 2.5  # the coloured dots of the traffic-light sign are tiny but decisive
KAPPA = 0.5     # penalty for a colour where it is not expected (relative to reward)
KAPPA_PICT = 1.0  # a coloured pictogram counts only if its surroundings do not share the colour
KAPPA_WHITE = 0.0  # a region expected to be coloured must not be bright white: penalty on intensity there
KAPPA_ABSENT = 0.5  # penalty for a colour on a sign that should not have it at all
A_MIN_FRAC = 0.15   # smallest colour area (fraction of the sign) used to normalise a colour reward

# colour indices on the canvas
BG, WHITE, BLACK, RED, BLUE, YELLOW, GREY, GREEN, RED_P, YELLOW_P = range(10)
NPAINT = 10
# expected intensity (mean of R,G,B) of each paint
INTENS = {WHITE: 1.0, BLACK: 0.0, RED: 0.35, BLUE: 0.30, YELLOW: 0.75, GREY: 0.55, GREEN: 0.40, RED_P: 0.35, YELLOW_P: 0.75}
# cone map used by each paint: (main paint, pictogram paint)
COLOUR_MAPS = {"red": (RED, RED_P), "blue": (BLUE, None), "yellow": (YELLOW, YELLOW_P)}
PICT_CLASSES = set(range(0, 12)) | {14, 16, 17} | set(range(18, 33)) | {41, 42}   # signs whose pictogram gets its own search
PICT_PAINTS = {17: (WHITE,), 14: (WHITE,)}   # default pictogram paints: black, grey, coloured pictogram parts

# ----------------------------------------------------------------------------------------------
# primitives: functions (u, v) -> bool mask. u right, v down, sign bounding box = [-0.5, 0.5]^2
# ----------------------------------------------------------------------------------------------
def P_disc(cu, cv, r):
    return lambda u, v: (u - cu) ** 2 + (v - cv) ** 2 <= r * r


def P_ring(cu, cv, r, w):
    return lambda u, v: np.abs(np.sqrt((u - cu) ** 2 + (v - cv) ** 2) - r) <= w / 2


def P_rect(u0, v0, u1, v1):
    return lambda u, v: (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)


def P_poly(pts):
    pts = [(float(a), float(b)) for a, b in pts]

    def f(u, v):
        inside = np.zeros(u.shape, bool)
        n = len(pts)
        for i in range(n):
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
            cond = (y0 > v) != (y1 > v)
            with np.errstate(divide="ignore", invalid="ignore"):
                xin = x0 + (v - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
            inside ^= cond & (u < xin)
        return inside
    return f


def P_seg(p, q, w):
    (x0, y0), (x1, y1) = p, q

    def f(u, v):
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy + 1e-12
        t = np.clip(((u - x0) * dx + (v - y0) * dy) / L2, 0, 1)
        return (u - x0 - t * dx) ** 2 + (v - y0 - t * dy) ** 2 <= (w / 2) ** 2
    return f


def P_path(pts, w):
    return P_union(*[P_seg(pts[i], pts[i + 1], w) for i in range(len(pts) - 1)])


def P_arc(cu, cv, r, a0, a1, w):
    """ring sector; angles in degrees, 0 = right, 90 = down, going from a0 increasing to a1"""
    span = (a1 - a0) % 360
    if span == 0:
        span = 360

    def f(u, v):
        d = np.sqrt((u - cu) ** 2 + (v - cv) ** 2)
        ang = (np.degrees(np.arctan2(v - cv, u - cu)) - a0) % 360
        return (np.abs(d - r) <= w / 2) & (ang <= span)
    return f


def P_union(*ps):
    def f(u, v):
        m = np.zeros(u.shape, bool)
        for p in ps:
            m |= p(u, v)
        return m
    return f


def P_inter(*ps):
    def f(u, v):
        m = np.ones(u.shape, bool)
        for p in ps:
            m &= p(u, v)
        return m
    return f


def P_diff(a, b):
    return lambda u, v: a(u, v) & ~b(u, v)


def P_xf(p, s=1.0, du=0.0, dv=0.0, su=None, sv=None, mirror=False):
    """place primitive p (defined around the origin) scaled by s (or su, sv) at (du, dv); mirror flips u"""
    su = s if su is None else su
    sv = s if sv is None else sv
    if mirror:
        return lambda u, v: p(-(u - du) / su, (v - dv) / sv)
    return lambda u, v: p((u - du) / su, (v - dv) / sv)


def P_mirror(p):
    return lambda u, v: p(-u, v)


def P_head(tip, direction, length, width):
    """triangular arrow head with its tip at `tip`, pointing along `direction`"""
    dx, dy = direction
    L = math.hypot(dx, dy) + 1e-12
    dx, dy = dx / L, dy / L
    bx, by = tip[0] - dx * length, tip[1] - dy * length
    nx, ny = -dy, dx
    return P_poly([tip, (bx + nx * width / 2, by + ny * width / 2), (bx - nx * width / 2, by - ny * width / 2)])


def P_arrow(pts, w, hl, hw):
    """polyline shaft + head at the last point"""
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    d = (x1 - x0, y1 - y0)
    L = math.hypot(*d) + 1e-12
    # shorten the shaft so it ends inside the head
    end = (x1 - d[0] / L * hl * 0.7, y1 - d[1] / L * hl * 0.7)
    shaft = P_path(list(pts[:-1]) + [end], w)
    return P_union(shaft, P_head((x1, y1), d, hl, hw))


# ----------------------------------------------------------------------------------------------
# digits as strokes (DIN-1451-like), height 1, width ~0.56, centred at origin
# ----------------------------------------------------------------------------------------------
T = 0.15   # stroke width (fraction of digit height)
A = 0.28   # half width of the digit box
RP = A - T / 2   # path radius for round parts


def digit(ch):
    r = RP
    if ch == "0":
        return P_union(P_arc(0, -0.5 + A, r, 180, 360, T), P_arc(0, 0.5 - A, r, 0, 180, T),
                       P_seg((-r, -0.5 + A), (-r, 0.5 - A), T), P_seg((r, -0.5 + A), (r, 0.5 - A), T))
    if ch == "1":
        return P_union(P_seg((0.08, -0.5 + T / 2), (0.08, 0.5 - T / 2), T),
                       P_seg((0.08, -0.5 + T / 2), (-0.16, -0.26), T * 0.9))
    if ch == "2":
        c = (0, -0.5 + A)
        e = (c[0] + r * math.cos(math.radians(35)), c[1] + r * math.sin(math.radians(35)))
        return P_union(P_arc(c[0], c[1], r, 185, 35, T), P_seg(e, (-r, 0.5 - T / 2), T),
                       P_seg((-r, 0.5 - T / 2), (r + 0.02, 0.5 - T / 2), T))
    if ch == "3":
        rt, rb = 0.19, 0.22
        return P_union(P_arc(0, -0.5 + rt + T / 2, rt, 200, 90, T),
                       P_arc(0, 0.5 - rb - T / 2, rb, 270, 160, T))
    if ch == "5":
        rb = 0.23
        c = (-0.02, 0.5 - rb - T / 2)
        return P_union(P_seg((-r + 0.02, -0.5 + T / 2), (r + 0.02, -0.5 + T / 2), T),
                       P_seg((-r + 0.02, -0.5 + T / 2), (-r + 0.02, -0.02), T),
                       P_seg((-r + 0.02, -0.04), (c[0], c[1] - rb), T),
                       P_arc(c[0], c[1], rb, 270, 150, T))
    if ch == "6":
        rb = 0.21
        cb = (0, 0.5 - rb - T / 2)
        return P_union(P_ring(cb[0], cb[1], rb, T),
                       P_arc(0.30, 0.0, 0.50, 180, 252, T),
                       P_seg((-0.20, 0.0), (-0.20, cb[1]), T))
    if ch == "7":
        return P_union(P_seg((-r - 0.02, -0.5 + T / 2), (r + 0.02, -0.5 + T / 2), T),
                       P_seg((r + 0.02, -0.5 + T / 2), (-0.1, 0.5 - T / 2), T))
    if ch == "8":
        rt, rb = 0.18, 0.21
        return P_union(P_ring(0, -0.5 + rt + T / 2, rt, T), P_ring(0, 0.5 - rb - T / 2, rb, T))
    if ch == "9":
        return P_mirror(P_xf(digit("6"), sv=-1.0))
    raise ValueError(ch)


def text_digits(s, h, cu=0.0, cv=0.0, colour=BLACK):
    """digits of string s with height h, centred at (cu, cv); returns list of (prim, colour)"""
    widths = [(0.34 if ch == "1" else 0.58) * h for ch in s]
    gap = 0.10 * h
    total = sum(widths) + gap * (len(s) - 1)
    x = cu - total / 2
    out = []
    for ch, w in zip(s, widths):
        out.append((P_xf(digit(ch), s=h, du=x + w / 2 - (0.02 * h if ch == "1" else 0.0), dv=cv), colour))
        x += w + gap
    return out


def letters_stop(h, cv=0.0):
    """very rough white STOP lettering: S T O P, height h"""
    t = 0.2 * h
    out = []
    # S
    xs = -0.36
    out.append(P_union(P_arc(xs, -0.22 * h, 0.17 * h, 200, 60, t), P_arc(xs, 0.22 * h, 0.17 * h, 20, 240, t)))
    # T
    xt = -0.12
    out.append(P_union(P_seg((xt - 0.18 * h, -0.5 * h + t / 2), (xt + 0.18 * h, -0.5 * h + t / 2), t),
                       P_seg((xt, -0.5 * h + t / 2), (xt, 0.5 * h - t / 2), t)))
    # O
    xo = 0.12
    out.append(P_union(P_arc(xo, -0.5 * h + 0.25 * h, 0.17 * h, 180, 360, t), P_arc(xo, 0.5 * h - 0.25 * h, 0.17 * h, 0, 180, t),
                       P_seg((xo - 0.17 * h, -0.25 * h), (xo - 0.17 * h, 0.25 * h), t), P_seg((xo + 0.17 * h, -0.25 * h), (xo + 0.17 * h, 0.25 * h), t)))
    # P
    xp = 0.36
    out.append(P_union(P_seg((xp - 0.15 * h, -0.5 * h + t / 2), (xp - 0.15 * h, 0.5 * h - t / 2), t),
                       P_arc(xp - 0.02 * h, -0.25 * h, 0.2 * h, 270, 90, t),
                       P_seg((xp - 0.15 * h, -0.5 * h + t / 2), (xp - 0.02 * h, -0.5 * h + t / 2), t),
                       P_seg((xp - 0.15 * h, -0.0 * h), (xp - 0.02 * h, -0.0 * h), t)))
    return [(P_xf(p, dv=cv), WHITE) for p in out]


# ----------------------------------------------------------------------------------------------
# sign descriptions: each returns (list of (primitive, colour) painted in order, inner-region primitive)
# ----------------------------------------------------------------------------------------------
RING_W = 0.10          # red ring width (fraction of the diameter)
TRI_B = 0.075          # warning triangle border width (fraction of the width)
YIELD_B = 0.085


def tri_up(scale=1.0, dv=0.0):
    # triangle filling the box: apex at top. (vertical stretch of the crop makes it about as tall as wide)
    return P_xf(P_poly([(-0.5, 0.5), (0.5, 0.5), (0, -0.5)]), s=scale, dv=dv)


def tri_down(scale=1.0, dv=0.0):
    return P_xf(P_poly([(-0.5, -0.5), (0.5, -0.5), (0, 0.5)]), s=scale, dv=dv)


def tri_inner_scale(b):
    # inset of a triangle of side 1 (and height 1) by border b: shrink about the centroid
    return 1.0 - 3.2 * b


def circle_sign(pict, ring=RING_W):
    prims = [(P_disc(0, 0, 0.5), RED), (P_disc(0, 0, 0.5 - ring), WHITE)] + pict
    return prims, P_disc(0, 0, 0.5 - ring)


def blue_sign(pict):
    prims = [(P_disc(0, 0, 0.5), WHITE), (P_disc(0, 0, 0.47), BLUE)] + pict
    return prims, P_disc(0, 0, 0.47)


def end_sign(pict):
    prims = [(P_disc(0, 0, 0.5), BLACK), (P_disc(0, 0, 0.47), WHITE)] + pict
    prims.append((P_seg((0.33, -0.33), (-0.33, 0.33), 0.10), BLACK))
    return prims, P_disc(0, 0, 0.47)


def warn_sign(pict):
    inner = tri_inner_scale(TRI_B)
    # centroid of the outer triangle is at v = 1/6; inner triangle shares it
    prims = [(tri_up(), RED), (tri_up(inner, dv=(1 - inner) / 6), WHITE)] + [(P_xf(p, dv=0.13), c) for p, c in pict]
    return prims, tri_up(inner, dv=(1 - inner) / 6)


def car_rear(cu, cv, w, colour):
    """small car seen from behind, width w, centred at (cu, cv)"""
    h = 0.85 * w
    body = P_rect(cu - w / 2, cv - 0.15 * h, cu + w / 2, cv + 0.45 * h)
    roof = P_rect(cu - 0.34 * w, cv - 0.55 * h, cu + 0.34 * w, cv - 0.15 * h)
    wheels = P_union(P_rect(cu - w / 2 + 0.05 * w, cv + 0.45 * h, cu - w / 2 + 0.3 * w, cv + 0.6 * h),
                     P_rect(cu + w / 2 - 0.3 * w, cv + 0.45 * h, cu + w / 2 - 0.05 * w, cv + 0.6 * h))
    return [(P_union(body, roof, wheels), colour)]


def truck_rear(cu, cv, w, h, colour):
    body = P_rect(cu - w / 2, cv - h / 2, cu + w / 2, cv + h / 2)
    wheels = P_union(P_rect(cu - w / 2 + 0.05 * w, cv + h / 2, cu - w / 2 + 0.3 * w, cv + h / 2 + 0.12 * h),
                     P_rect(cu + w / 2 - 0.3 * w, cv + h / 2, cu + w / 2 - 0.05 * w, cv + h / 2 + 0.12 * h))
    return [(P_union(body, wheels), colour)]


def truck_side(colour=BLACK):
    body = P_rect(-0.08, -0.17, 0.33, 0.10)
    cab = P_union(P_rect(-0.32, -0.06, -0.08, 0.10), P_poly([(-0.32, -0.06), (-0.26, -0.14), (-0.08, -0.14), (-0.08, -0.06)]))
    wheels = P_union(P_disc(-0.22, 0.13, 0.055), P_disc(0.12, 0.13, 0.055), P_disc(0.25, 0.13, 0.055))
    return [(P_union(body, cab, wheels), colour)]


def person(cu, cv, h, colour=BLACK, lean=0.0):
    """walking figure of height h, centred at (cu, cv), facing left"""
    head = P_disc(cu + 0.02 * h, cv - 0.40 * h, 0.11 * h)
    torso = P_poly([(cu - 0.12 * h, cv - 0.26 * h), (cu + 0.10 * h, cv - 0.26 * h), (cu + 0.12 * h, cv + 0.05 * h), (cu - 0.10 * h, cv + 0.05 * h)])
    leg1 = P_seg((cu - 0.04 * h, cv + 0.02 * h), (cu - 0.22 * h, cv + 0.5 * h), 0.12 * h)
    leg2 = P_seg((cu + 0.04 * h, cv + 0.02 * h), (cu + 0.16 * h, cv + 0.48 * h), 0.12 * h)
    arm = P_seg((cu + 0.0 * h, cv - 0.22 * h), (cu - 0.2 * h, cv - 0.05 * h), 0.09 * h)
    return [(P_union(head, torso, leg1, leg2, arm), colour)]


def pict_warning(c):
    """pictogram for warning class c, in a local frame centred on the pictogram"""
    if c == 18:   # general danger: exclamation mark
        return [(P_union(P_seg((0, -0.2), (0, 0.07), 0.08), P_disc(0, 0.19, 0.05)), BLACK)]
    if c == 11:   # priority at next junction: thick vertical, thin horizontal arms
        return [(P_union(P_rect(-0.06, -0.21, 0.06, 0.23), P_rect(-0.2, -0.03, 0.2, 0.03)), BLACK)]
    if c == 19:   # bend left
        return [(P_arrow([(0.07, 0.24), (0.07, -0.02), (-0.14, -0.13)], 0.08, 0.1, 0.17), BLACK)]
    if c == 20:   # bend right
        return [(P_mirror(pict_warning(19)[0][0]), BLACK)]
    if c == 21:   # double bend, first left
        return [(P_arrow([(0.07, 0.24), (0.07, 0.1), (-0.07, 0.02), (-0.07, -0.07), (0.06, -0.14), (0.06, -0.24)], 0.07, 0.09, 0.15), BLACK)]
    if c == 22:   # bumpy road
        base = P_rect(-0.24, 0.17, 0.24, 0.22)
        bumps = P_union(P_inter(P_disc(-0.11, 0.19, 0.09), P_rect(-0.3, 0.0, 0.3, 0.19)),
                        P_inter(P_disc(0.11, 0.19, 0.09), P_rect(-0.3, 0.0, 0.3, 0.19)))
        return [(P_union(base, bumps), BLACK)]
    if c == 23:   # slippery road: car (upper right) with skid marks down-left
        car = P_poly([(-0.04, -0.14), (0.2, -0.06), (0.16, 0.06), (-0.1, -0.02)])
        wheels = P_union(P_disc(-0.05, 0.0, 0.035), P_disc(0.14, 0.07, 0.035))
        s1 = P_path([(-0.06, 0.03), (-0.14, 0.09), (-0.08, 0.15), (-0.18, 0.23)], 0.035)
        s2 = P_path([(0.04, 0.08), (-0.04, 0.14), (0.02, 0.19), (-0.08, 0.26)], 0.035)
        return [(P_union(car, wheels, s1, s2), BLACK)]
    if c == 24:   # road narrows on the right
        left = P_seg((-0.1, -0.22), (-0.1, 0.24), 0.05)
        right = P_path([(0.11, 0.24), (0.11, 0.08), (0.03, -0.03), (0.03, -0.22)], 0.05)
        return [(P_union(left, right), BLACK)]
    if c == 25:   # road works
        head = P_disc(-0.07, -0.19, 0.05)
        torso = P_poly([(-0.13, -0.12), (-0.02, -0.12), (0.03, 0.04), (-0.1, 0.08)])
        legs = P_union(P_seg((-0.08, 0.06), (-0.13, 0.24), 0.05), P_seg((-0.02, 0.05), (0.05, 0.2), 0.05))
        shovel = P_seg((-0.05, -0.06), (0.15, 0.15), 0.035)
        mound = P_poly([(0.0, 0.24), (0.12, 0.1), (0.25, 0.24)])
        return [(P_union(head, torso, legs, shovel, mound), BLACK)]
    if c == 26:   # traffic lights
        return [(P_rect(-0.065, -0.22, 0.065, 0.22), BLACK), (P_disc(0, -0.14, 0.04), RED_P),
                (P_disc(0, 0.0, 0.04), YELLOW_P), (P_disc(0, 0.14, 0.04), GREEN)]
    if c == 27:   # pedestrians
        return person(0.0, 0.0, 0.48)
    if c == 28:   # children crossing: adult + child running left
        return person(0.07, -0.02, 0.46) + person(-0.12, 0.06, 0.34)
    if c == 29:   # cyclists crossing
        wheels = P_union(P_ring(-0.12, 0.15, 0.08, 0.035), P_ring(0.12, 0.15, 0.08, 0.035))
        frame = P_union(P_path([(-0.12, 0.15), (0.0, 0.02), (0.12, 0.15)], 0.03), P_seg((0.0, 0.02), (0.04, 0.15), 0.03),
                        P_seg((-0.02, 0.0), (0.1, 0.0), 0.03))
        rider = P_union(P_disc(0.04, -0.2, 0.045), P_seg((0.02, -0.14), (-0.03, -0.0), 0.06), P_seg((0.0, -0.1), (0.1, -0.02), 0.035))
        return [(P_union(wheels, frame, rider), BLACK)]
    if c == 30:   # ice / snow: snowflake
        arms = [P_seg((-0.2 * math.cos(math.radians(a)), -0.2 * math.sin(math.radians(a))),
                      (0.2 * math.cos(math.radians(a)), 0.2 * math.sin(math.radians(a))), 0.04) for a in (90, 30, 150)]
        tips = [P_disc(0.17 * math.cos(math.radians(a)), 0.17 * math.sin(math.radians(a)), 0.035) for a in range(30, 390, 60)]
        return [(P_union(*(arms + tips)), BLACK)]
    if c == 31:   # wild animals: leaping deer facing left
        body = P_poly([(-0.24, -0.02), (0.02, -0.14), (0.22, -0.04), (0.16, 0.08), (-0.14, 0.1)])
        head = P_union(P_disc(-0.25, -0.12, 0.06), P_seg((-0.25, -0.12), (-0.33, -0.07), 0.06))
        antlers = P_union(P_seg((-0.26, -0.16), (-0.31, -0.27), 0.03), P_seg((-0.22, -0.16), (-0.15, -0.27), 0.03))
        legs = P_union(P_seg((-0.14, 0.09), (-0.2, 0.26), 0.05), P_seg((-0.05, 0.1), (-0.03, 0.26), 0.05),
                       P_seg((0.12, 0.08), (0.2, 0.24), 0.05), P_seg((0.17, 0.05), (0.28, 0.22), 0.04))
        return [(P_union(body, head, antlers, legs), BLACK)]
    raise ValueError(c)


def sign(c):
    """returns (prims, inner_prim) for class c"""
    speed = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}
    if c in speed:
        s = speed[c]
        h = 0.50 if len(s) == 2 else 0.46
        return circle_sign(text_digits(s, h))
    if c == 9:    # no passing: red car left, black car right
        return circle_sign(car_rear(-0.16, 0.0, 0.25, RED_P) + car_rear(0.16, 0.0, 0.25, BLACK))
    if c == 10:   # no passing for trucks: red truck left, black car right
        return circle_sign(truck_rear(-0.17, -0.02, 0.25, 0.34, RED_P) + car_rear(0.17, 0.06, 0.23, BLACK))
    if c == 15:   # no vehicles
        return circle_sign([])
    if c == 16:   # no trucks
        return circle_sign(truck_side(BLACK))
    if c == 17:   # no entry
        return [(P_disc(0, 0, 0.5), RED), (P_rect(-0.37, -0.085, 0.37, 0.085), WHITE)], P_disc(0, 0, 0.5)
    if c == 14:   # stop
        R0, R1 = 0.5 / math.cos(math.radians(22.5)), 0.455 / math.cos(math.radians(22.5))
        octo = lambda R: P_poly([(R * math.cos(math.radians(22.5 + 45 * k)), R * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)])
        return [(octo(R0), WHITE), (octo(R1), RED)] + letters_stop(0.28, cv=0.0), octo(R1)
    if c == 13:   # yield
        inner = tri_inner_scale(YIELD_B)
        return [(tri_down(), RED), (tri_down(inner, dv=-(1 - inner) / 6), WHITE)], tri_down(inner, dv=-(1 - inner) / 6)
    if c == 12:   # priority road
        dia = lambda r: P_poly([(0, -r), (r, 0), (0, r), (-r, 0)])
        return [(dia(0.5), WHITE), (dia(0.30), YELLOW)], dia(0.5)
    if c in (11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
        return warn_sign(pict_warning(c))
    if c == 6:    # end of speed limit 80
        return end_sign(text_digits("80", 0.50, colour=GREY))
    if c == 32:   # end of all limits: five thin stripes
        stripes = [(P_seg((0.33 + k, -0.33 + k), (-0.33 + k, 0.33 + k), 0.03), BLACK) for k in (-0.11, -0.055, 0.0, 0.055, 0.11)]
        prims = [(P_disc(0, 0, 0.5), BLACK), (P_disc(0, 0, 0.47), WHITE)] + stripes
        return prims, P_disc(0, 0, 0.47)
    if c == 41:   # end of no passing
        return end_sign(car_rear(-0.16, 0.0, 0.24, GREY) + car_rear(0.16, 0.0, 0.24, GREY))
    if c == 42:   # end of no passing for trucks
        return end_sign(truck_rear(-0.17, -0.02, 0.25, 0.34, GREY) + car_rear(0.17, 0.06, 0.22, GREY))
    # blue mandatory signs
    if c == 35:   # ahead only
        return blue_sign([(P_arrow([(0, 0.33), (0, -0.34)], 0.12, 0.22, 0.38), WHITE)])
    if c == 33:   # turn right ahead
        return blue_sign([(P_arrow([(-0.1, 0.3), (-0.1, -0.06), (-0.03, -0.16), (0.08, -0.2), (0.31, -0.2)], 0.11, 0.17, 0.3), WHITE)])
    if c == 34:   # turn left ahead
        return blue_sign([(P_mirror(sign(33)[0][-1][0]), WHITE)])
    if c == 36:   # straight or right
        up = P_arrow([(-0.09, 0.32), (-0.09, -0.33)], 0.11, 0.18, 0.3)
        br = P_arrow([(-0.09, 0.04), (0.31, 0.04)], 0.1, 0.15, 0.26)
        return blue_sign([(P_union(up, br), WHITE)])
    if c == 37:   # straight or left
        return blue_sign([(P_mirror(sign(36)[0][-1][0]), WHITE)])
    if c == 38:   # keep right: arrow pointing down-right
        return blue_sign([(P_arrow([(-0.23, -0.23), (0.24, 0.24)], 0.11, 0.2, 0.32), WHITE)])
    if c == 39:   # keep left
        return blue_sign([(P_mirror(sign(38)[0][-1][0]), WHITE)])
    if c == 40:   # roundabout: three curved arrows
        parts = []
        for k in range(3):
            a0 = -70 + 120 * k
            a1 = a0 + 75
            parts.append(P_arc(0, 0, 0.28, a0, a1, 0.09))
            # head at the a0 end, pointing in the direction of decreasing angle (counter-clockwise on screen)
            tip = (0.28 * math.cos(math.radians(a0 - 25)), 0.28 * math.sin(math.radians(a0 - 25)))
            d = (math.sin(math.radians(a0)), -math.cos(math.radians(a0)))
            parts.append(P_head(tip, d, 0.14, 0.2))
        return blue_sign([(P_union(*parts), WHITE)])
    raise ValueError(c)


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
def render(c, W, H, cx=N / 2, cy=N / 2, rot=0.0, ss=SS, A=None):
    """render class c with bounding box W x H pixels centred at (cx, cy); returns
    fractions (8, N, N) of each colour index per pixel and the inner-region fraction (N, N)"""
    g = (np.arange(N * ss) + 0.5) / ss
    X, Y = np.meshgrid(g, g)
    x, y = X - cx, Y - cy
    if rot:
        cr, sr = math.cos(rot), math.sin(rot)
        x, y = cr * x + sr * y, -sr * x + cr * y
    if A is not None:
        x, y = A[0][0] * x + A[0][1] * y, A[1][0] * x + A[1][1] * y
    u, v = x / W, y / H
    prims, inner = sign(c)
    canvas = np.zeros(u.shape, np.int8)
    for p, col in prims:
        canvas[p(u, v)] = col
    fr = np.stack([(canvas == k).reshape(N, ss, N, ss).mean((1, 3)) for k in range(NPAINT)])
    inn = inner(u, v).reshape(N, ss, N, ss).mean((1, 3))
    return fr, inn


def gauss_blur(a, sigma):
    """separable gaussian blur of the last two axes (zero padded)"""
    if sigma <= 0:
        return a
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.zeros_like(a, dtype=np.float64)
    ap = np.pad(a.astype(np.float64), [(0, 0)] * (a.ndim - 2) + [(r, r), (r, r)])
    tmp = np.zeros_like(ap)
    for i, kv in enumerate(k):
        tmp[..., :, r:-r] += kv * ap[..., :, i:i + a.shape[-1]]
    for i, kv in enumerate(k):
        out += kv * tmp[..., i:i + a.shape[-2], r:-r]
    return out


# ----------------------------------------------------------------------------------------------
# templates
# ----------------------------------------------------------------------------------------------
def log_kernel(sigma=LOG_SIGMA, size=7):
    """negative Laplacian of Gaussian (centre positive), zero mean, scaled so that |response| to a unit step is about 1"""
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    q = (x * x + y * y) / (2 * sigma * sigma)
    k = (1 - q) * np.exp(-q)
    k -= k.mean()
    k /= np.abs(k).sum() / 2
    return k


def conv2d_same(a, k):
    """2-D correlation with zero padding (a: (..., H, W), k: (kh, kw))"""
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    ap = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(ph, ph), (pw, pw)])
    out = np.zeros(a.shape, np.float64)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * ap[..., i:i + a.shape[-2], j:j + a.shape[-1]]
    return out


def make_template(c, f, a=1.0):
    """grey, interior, edge and colour templates for class c at scale f (sign size = f*N) and aspect a"""
    fr0, inn0 = render(c, f * N * math.sqrt(a), f * N / math.sqrt(a))
    fr = gauss_blur(fr0, BLUR_SIGMA)
    inn = gauss_blur(inn0, BLUR_SIGMA)
    core = (gauss_blur(inn0, 1.0) > 0.92).astype(np.float64)   # interior eroded by ~1.4 px: no border leakage
    plate = fr[1:].sum(0)                      # coverage of the sign
    E = sum(fr[k] * INTENS[k] for k in INTENS)  # expected intensity (weighted by coverage)
    # whole-sign pattern, zero-mean over the sign, unit norm
    w = plate + (W_INNER - 1.0) * inn
    m = (w * E).sum() / max(w.sum(), 1e-9)
    G = w * (E - m * plate)
    G -= plate * (G.sum() / max(plate.sum(), 1e-9))
    G /= (np.sqrt((G ** 2).sum()) + 1e-9)
    # interior-only pattern (pictogram / digits), zero-mean over the interior, unit norm
    mi = (core * E).sum() / max(core.sum(), 1e-9)
    Gi = core * (E - mi)
    ni = np.sqrt((Gi ** 2).sum())
    Gi = Gi / ni if ni > 0.5 else 0.0 * Gi      # a plain interior carries no pattern evidence
    G = G + LAMBDA_INNER * Gi                   # one pattern per variant: whole sign plus interior emphasis
    G /= (np.sqrt((G ** 2).sum()) + 1e-9) * N   # unit norm: a cosine-like score against a unit-variance image
    # pictogram-level pattern: a tight box around the pictogram ink, zero-mean there (no gross layout)
    P = np.zeros((N, N))
    if c in PICT_CLASSES:
        paints = PICT_PAINTS.get(c, (BLACK, GREY, RED_P, YELLOW_P, GREEN))
        ink = sum(fr0[k] for k in paints) * (core > 0)
        if ink.sum() > 2:
            region = ((gauss_blur(ink, 1.5) > 0.02) & (core > 0)).astype(np.float64)
            P = region * (E - (region * E).sum() / max(region.sum(), 1e-9))
            P *= W_PICT / ((np.sqrt((P ** 2).sum()) + 1e-9) * N)
    Gi = P
    # expected edge energy: every paint boundary (including the outline of the sign) produces an edge
    M = 0.5 * np.abs(conv2d_same(fr, log_kernel())).sum(0)
    plate_d = (gauss_blur(plate, 1.0) > 0.05).astype(np.float64)
    Me = plate_d * (M - (plate_d * M).sum() / max(plate_d.sum(), 1e-9))
    Me /= (np.sqrt((Me ** 2).sum()) + 1e-9) * N
    Me *= MU_EDGE
    cols = {}
    A_plate = plate.sum()
    A_min = A_MIN_FRAC * A_plate
    frc = gauss_blur(fr0, BLUR_COLOUR)
    reward_total = np.zeros((N, N))
    for name, (k_main, k_pict) in COLOUR_MAPS.items():
        sharp = fr0[k_main] + (fr0[k_pict] if k_pict is not None else 0.0)
        if sharp.sum() > 0.5:
            Wc = np.zeros((N, N))
            for k in (k_main, k_pict):
                if k is not None and fr0[k].sum() > 0.5:
                    boost = C_LIGHTS_BOOST if (c == 26 and k == k_pict) else 1.0
                    A_k = max(fr0[k].sum(), A_min)
                    Wc += boost * frc[k] / A_k                        # reward, per unit of expected area
                    if name != "yellow":
                        reward_total += boost * frc[k] / A_k           # (yellow paint is bright by nature)
                    if k == k_pict:
                        near = gauss_blur(fr0[k], 0.7) > 0.05
                        far = gauss_blur(fr0[k], 2.0) > 0.02
                        surround = np.clip(far.astype(np.float64) - near, 0, None) * plate
                        Wc -= boost * KAPPA_PICT * surround / max(surround.sum(), 1.0) * (fr0[k].sum() / A_k)
            # penalise the colour where the sign should not have it, leaving a margin around the expected region
            dil = gauss_blur(sharp, 1.0) > 0.05
            neg = np.clip(plate - dil, 0, None)
            Wc -= KAPPA * neg / max(fr0[k_main].sum() if fr0[k_main].sum() > 0.5 else fr0[k_pict].sum(), A_min)
        else:
            Wc = -KAPPA_ABSENT * plate / (0.3 * A_plate)
        if name == "yellow" and c == 12:
            Wc = Wc * C_YELLOW_BOOST
        cols[name] = C_COL * Wc
    cols["I"] = -C_COL * KAPPA_WHITE * reward_total       # coloured regions must not be bright white
    return G, Gi, Me, cols


def build():
    layers, weights = [], {}
    # --- colour maps: I, red, blue, yellow, zero (cone tests; greys give 0 red / negative blue, yellow)
    cm = np.zeros((5, 3, 1, 1), np.float32)
    cm[0, :, 0, 0] = [1 / 3, 1 / 3, 1 / 3]            # intensity
    cm[1, :, 0, 0] = [1.0, -1.45, 0.3]                 # red (negative on greys, pinkish whites and yellow)
    cm[2, :, 0, 0] = [-0.5, -0.65, 1.0]                # blue (negative on bluish whites)
    cm[3, :, 0, 0] = [-0.15, 1.0, -1.2]                # yellow
    cm[4, :, 0, 0] = [0.0, 0.0, 0.0]                   # zero reference
    layers.append({"name": "cmap", "op": "conv", "inputs": ["image"], "out": 5, "kernel": [1, 1], "stride": 1, "padding": 0})
    weights["cmap.W"] = cm
    weights["cmap.b"] = np.array([0.0, -COL_EPS, -COL_EPS, -COL_EPS, 0.0], np.float32)
    layers.append({"name": "cmr", "op": "relu", "inputs": ["cmap"]})
    # grey branch
    sI = np.zeros((1, 5, 1, 1), np.float32); sI[0, 0] = 1
    layers.append({"name": "selI", "op": "conv", "inputs": ["cmr"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0})
    weights["selI.W"], weights["selI.b"] = sI, np.zeros(1, np.float32)
    layers.append({"name": "normI", "op": "norm", "inputs": ["selI"]})
    # edge energy branch: centre-surround filter, rectified both ways
    lk = log_kernel()
    ek = np.zeros((2, 1, lk.shape[0], lk.shape[1]), np.float32)
    ek[0, 0], ek[1, 0] = lk, -lk
    layers.append({"name": "edge", "op": "conv", "inputs": ["normI"], "out": 2, "kernel": list(lk.shape), "stride": 1, "padding": lk.shape[0] // 2})
    weights["edge.W"], weights["edge.b"] = ek, np.zeros(2, np.float32)
    layers.append({"name": "edger", "op": "relu", "inputs": ["edge"]})
    # colour branch: joint normalisation of the five maps (one shared mean and scale)
    sC = np.zeros((5, 5, 1, 1), np.float32)
    for i in range(5):
        sC[i, i] = 1
    layers.append({"name": "selC", "op": "conv", "inputs": ["cmr"], "out": 5, "kernel": [1, 1], "stride": 1, "padding": 0})
    weights["selC.W"], weights["selC.b"] = sC, np.zeros(5, np.float32)
    layers.append({"name": "rsC", "op": "reshape", "inputs": ["selC"], "shape": [1, 5 * N, N]})
    layers.append({"name": "normC", "op": "norm", "inputs": ["rsC"]})
    layers.append({"name": "rsC2", "op": "reshape", "inputs": ["normC"], "shape": [5, N, N]})
    # --- prototype templates
    S = len(VARIANTS)
    Tn = 43 * S
    WA = np.zeros((Tn, 1, N, N), np.float32)
    WP = np.zeros((Tn, 1, N, N), np.float32)
    WB = np.zeros((Tn, 5, N, N), np.float32)
    WE = np.zeros((Tn, 2, N, N), np.float32)
    for c in range(43):
        for si, (f, a) in enumerate(VARIANTS):
            G, Gi, Me, cols = make_template(c, f, a)
            t = c * S + si
            WA[t, 0] = G
            WP[t, 0] = Gi
            WB[t, 0] = cols["I"]
            WE[t, 0] = Me
            WE[t, 1] = Me
            WB[t, 1] = cols["red"]
            WB[t, 2] = cols["blue"]
            WB[t, 3] = cols["yellow"]
            WB[t, 4] = -(WB[t, 0] + WB[t, 1] + WB[t, 2] + WB[t, 3])   # cancels the shared offset of the joint norm
    layers.append({"name": "tA", "op": "conv", "inputs": ["normI"], "out": Tn, "kernel": [N, N], "stride": 1, "padding": PAD})
    weights["tA.W"], weights["tA.b"] = WA, np.zeros(Tn, np.float32)
    layers.append({"name": "tB", "op": "conv", "inputs": ["rsC2"], "out": Tn, "kernel": [N, N], "stride": 1, "padding": PAD})
    weights["tB.W"], weights["tB.b"] = WB, np.zeros(Tn, np.float32)
    layers.append({"name": "tE", "op": "conv", "inputs": ["edger"], "out": Tn, "kernel": [N, N], "stride": 1, "padding": PAD})
    weights["tE.W"], weights["tE.b"] = WE, np.zeros(Tn, np.float32)
    layers.append({"name": "tsum", "op": "add", "inputs": ["tA", "tB", "tE"]})
    k = 2 * PAD + 1
    layers.append({"name": "mpos", "op": "maxpool", "inputs": ["tsum"], "size": [k, k], "stride": [k, k]})
    layers.append({"name": "rsT", "op": "reshape", "inputs": ["mpos"], "shape": [1, Tn, 1]})
    layers.append({"name": "mscale", "op": "maxpool", "inputs": ["rsT"], "size": [S, 1], "stride": [S, 1]})
    layers.append({"name": "feat", "op": "reshape", "inputs": ["mscale"], "shape": [43]})
    # pictogram-level evidence: own shift search, max over variants, rectified
    kp = 2 * PAD_P + 1
    layers.append({"name": "tP", "op": "conv", "inputs": ["normI"], "out": Tn, "kernel": [N, N], "stride": 1, "padding": PAD_P})
    weights["tP.W"], weights["tP.b"] = WP, np.zeros(Tn, np.float32)
    layers.append({"name": "mposP", "op": "maxpool", "inputs": ["tP"], "size": [kp, kp], "stride": [kp, kp]})
    layers.append({"name": "rsP", "op": "reshape", "inputs": ["mposP"], "shape": [1, Tn, 1]})
    layers.append({"name": "mscaleP", "op": "maxpool", "inputs": ["rsP"], "size": [S, 1], "stride": [S, 1]})
    layers.append({"name": "featP", "op": "reshape", "inputs": ["mscaleP"], "shape": [43]})
    layers.append({"name": "featPr", "op": "relu", "inputs": ["featP"]})
    layers.append({"name": "featsum", "op": "add", "inputs": ["feat", "featPr"]})
    layers.append({"name": "out", "op": "dense", "inputs": ["featsum"], "out": 43})
    weights["out.W"], weights["out.b"] = np.eye(43, dtype=np.float32), np.zeros(43, np.float32)
    return {"layers": layers}, weights


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    import os
    os.makedirs(out, exist_ok=True)
    net, weights = build()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out, "weights.npz"), **{k: v.astype(np.float32) for k, v in weights.items()})
    print("wrote", out, "total weights", sum(int(np.prod(v.shape)) for v in weights.values()))


if __name__ == "__main__":
    main()
