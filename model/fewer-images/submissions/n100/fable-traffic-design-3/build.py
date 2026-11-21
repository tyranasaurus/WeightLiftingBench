"""Hand-designed traffic-sign network: writes network.json and weights.npz.

Usage: python3 build.py OUT_DIR

Nothing here is fitted to data.  The network is
  1. a fixed photometric front end (blur, piecewise-linear log, centre-weighted standardisation),
  2. hand-chosen feature maps (brightness, local contrast, three colour indicators, six edge orientations),
  3. one big convolution whose kernels are drawings of the 43 signs (typed below from general knowledge of the
     German sign catalogue), each drawn at several sizes, passed through the same front end and normalised by itself,
  4. a maximum over positions and sizes.
"""
import sys, json
import numpy as np

# ----------------------------------------------------------------------------------------------------------------
# 0. Hand-chosen constants
# ----------------------------------------------------------------------------------------------------------------
N = 32                      # image side
M = 30                      # side of all feature maps (the 3 x 3 blur is taken without padding)
EPS_LOG = 0.02              # offset inside the logarithm (about 5 grey levels)
N_KNOTS = 14                # hinges of the piecewise-linear logarithm
WIN_SIGMA = 6.5             # centre window (pixels) used to weight the standardisation statistics
C_A = 1.5                   # brightness clipped at +-1.5 standard deviations
C_B = 1.0                   # local contrast clipped at +-1 standard deviation
SIG_B = 2.0                 # surround blur of the local contrast channel
COL_LO, COL_HI = 0.4, 1.2   # colour indicator ramp (in standard deviations of the log image)
SIG_COL = 1.0               # blur of the colour differences (before thresholding, so colour fringes cancel)
N_ORI = 6                   # edge orientations (30 degree steps)
ORI_DEAD, ORI_CAP = 0.06, 0.30   # edge strength dead zone and saturation
SIG_ORI = 1.0               # blur of the orientation maps
W_A, W_B, W_COL, W_ORI = 1.0, 1.0, 1.0, 1.0      # weights of the four feature groups in the match
SIZES = [19.0, 20.5, 22.0, 23.5, 25.0, 26.5, 28.0]   # sign sizes (pixels) at which every sign is drawn
# every sign is drawn, at every size, in these (rotation in degrees, blur in pixels) versions: photos range from
# crisp to badly smeared, and a crisp drawing would lose against a simpler sign whenever detail is washed out
VERSIONS = [(0.0, 0.0), (0.0, 0.9), (0.0, 1.8), (-8.0, 0.9), (8.0, 0.9)]
KAPPA = 1.0                 # weight of the interior (content) match relative to the frame match
BLANK_TAU, BLANK_ETA = 3.0, 0.06   # signs with an empty interior: content score = TAU - ETA * (edge energy inside)
KT = 30                     # template kernel side (the whole feature map)
PAD_T = 2                   # template padding: shifts of -2 .. +2 pixels per axis
SS = 6                      # supersampling of the drawings
N_CLASSES = 43
NF = 4 + N_ORI              # feature channels: A, B, red-minus-blue, yellow, orientations
ORI0 = 4                    # first orientation channel

# drawing colours (typical photographed sign paints)
WHITE = (0.85, 0.85, 0.85)
RED = (0.62, 0.12, 0.13)
BLUE = (0.10, 0.27, 0.62)
YELLOW = (0.85, 0.66, 0.10)
BLACK = (0.14, 0.14, 0.14)
GREY = (0.40, 0.40, 0.40)
DGREY = (0.24, 0.24, 0.24)
GREEN = (0.15, 0.50, 0.30)
BG = (0.42, 0.42, 0.42)

# ----------------------------------------------------------------------------------------------------------------
# 1. Drawing primitives (all work on arrays of sample coordinates X, Y and return boolean masks)
# ----------------------------------------------------------------------------------------------------------------

def seg_dist(X, Y, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / max(L2, 1e-12), 0.0, 1.0)
    return np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))


def stroke(X, Y, pts, w, closed=False):
    pts = list(pts)
    if closed:
        pts = pts + [pts[0]]
    d = np.full(X.shape, 1e9)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        d = np.minimum(d, seg_dist(X, Y, x0, y0, x1, y1))
    return d <= w / 2.0


def poly(X, Y, pts):
    inside = np.zeros(X.shape, bool)
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if y0 == y1:
            continue
        cond = ((y0 > Y) != (y1 > Y)) & (X < (x1 - x0) * (Y - y0) / (y1 - y0) + x0)
        inside ^= cond
    return inside


def disc(X, Y, cx, cy, r):
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r


def ellipse(X, Y, cx, cy, rx, ry):
    return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0


def rect(X, Y, x0, y0, x1, y1):
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


def arc(cx, cy, rx, ry, a0, a1, n=24):
    """points on an ellipse arc; angles in radians, y axis pointing down (angle -pi/2 is the top)"""
    a = np.linspace(a0, a1, n)
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in a]


def convex_sdf(X, Y, pts):
    """signed distance to a convex polygon (negative inside)"""
    pts = np.asarray(pts, float)
    n = len(pts)
    area = 0.0
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    sgn = 1.0 if area > 0 else -1.0
    dmax = np.full(X.shape, -1e9)
    dseg = np.full(X.shape, 1e9)
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        L = np.hypot(ex, ey)
        nx, ny = sgn * ey / L, -sgn * ex / L          # outward normal
        dmax = np.maximum(dmax, (X - x0) * nx + (Y - y0) * ny)
        dseg = np.minimum(dseg, seg_dist(X, Y, x0, y0, x1, y1))
    return np.where(dmax <= 0, dmax, dseg)


def shrink_convex(pts, r):
    """move every edge of a convex polygon inwards by r"""
    pts = np.asarray(pts, float)
    n = len(pts)
    area = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))
    sgn = 1.0 if area > 0 else -1.0
    lines = []
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        L = np.hypot(ex, ey)
        nx, ny = sgn * ey / L, -sgn * ex / L
        lines.append((nx, ny, nx * x0 + ny * y0 - r))
    out = []
    for i in range(n):
        a1, b1, c1 = lines[i - 1]
        a2, b2, c2 = lines[i]
        det = a1 * b2 - a2 * b1
        out.append(((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det))
    return out


def rounded_sdf(X, Y, pts, r):
    return convex_sdf(X, Y, shrink_convex(pts, r)) - r


def arrow_head(tip, direction, length, half_width):
    """triangle with its tip at `tip`, pointing along `direction`"""
    dx, dy = direction
    L = np.hypot(dx, dy); dx, dy = dx / L, dy / L
    bx, by = tip[0] - dx * length, tip[1] - dy * length
    return [tip, (bx - dy * half_width, by + dx * half_width), (bx + dy * half_width, by - dx * half_width)]


def mirror(pts):
    return [(-x, y) for x, y in pts]


# ----------------------------------------------------------------------------------------------------------------
# 2. Lettering: digits and S, T, O, P as centre lines in a unit box (x right, y down)
# ----------------------------------------------------------------------------------------------------------------

def _superellipse(cx, cy, rx, ry, n=2.6, m=40):
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return [(cx + rx * np.sign(a) * abs(a) ** (2 / n), cy + ry * np.sign(b) * abs(b) ** (2 / n)) for a, b in zip(c, s)]


def glyph(ch):
    """list of (points, closed) centre lines in the unit box"""
    P = np.pi
    if ch in '0O':
        return [(_superellipse(0.5, 0.5, 0.5, 0.5), True)]
    if ch == '1':
        return [([(0.0, 0.27), (1.0, 0.0), (1.0, 1.0)], False)]
    if ch == '2':
        top = arc(0.5, 0.27, 0.5, 0.27, 1.05 * P, 2 * P + 0.6, 20)
        return [(top + [(0.0, 1.0), (1.0, 1.0)], False)]
    if ch == '3':
        up = arc(0.5, 0.26, 0.5, 0.26, 1.12 * P, 2 * P + 1.18, 20)
        lo = arc(0.5, 0.73, 0.5, 0.27, -1.3, 0.9 * P, 20)
        return [(up + [(0.42, 0.485)] + lo, False)]
    if ch == '5':
        bowl = arc(0.48, 0.68, 0.52, 0.32, -0.8 * P, 0.85 * P, 24)
        return [([(0.95, 0.0), (0.10, 0.0), (0.06, 0.47)] + bowl, False)]
    if ch == '6':
        loop = _superellipse(0.5, 0.68, 0.5, 0.32, 2.2)
        stem = [(0.92, 0.13), (0.72, 0.03), (0.5, 0.0), (0.27, 0.05), (0.09, 0.2), (0.0, 0.42), (0.0, 0.68)]
        return [(loop, True), (stem, False)]
    if ch == '7':
        return [([(0.0, 0.0), (1.0, 0.0), (0.36, 1.0)], False)]
    if ch == '8':
        return [(_superellipse(0.5, 0.25, 0.44, 0.25, 2.2), True), (_superellipse(0.5, 0.73, 0.5, 0.27, 2.2), True)]
    if ch == '9':
        return [([(1 - x, 1 - y) for x, y in pts], c) for pts, c in glyph('6')]
    if ch == 'S':
        pts = [(0.95, 0.15), (0.75, 0.02), (0.5, 0.0), (0.25, 0.02), (0.05, 0.14), (0.02, 0.27), (0.12, 0.40), (0.5, 0.5),
               (0.88, 0.60), (0.98, 0.73), (0.95, 0.86), (0.75, 0.98), (0.5, 1.0), (0.25, 0.98), (0.05, 0.85)]
        return [(pts, False)]
    if ch == 'T':
        return [([(0.0, 0.0), (1.0, 0.0)], False), ([(0.5, 0.0), (0.5, 1.0)], False)]
    if ch == 'P':
        bowl = arc(0.6, 0.27, 0.4, 0.27, -P / 2, P / 2, 14)
        return [([(0.0, 1.0), (0.0, 0.0)] + bowl + [(0.0, 0.54)], False)]
    raise ValueError(ch)


def text_mask(X, Y, chars, widths, height, gap, t, cx=0.0, cy=0.0):
    """ink mask of a row of glyphs centred at (cx, cy); widths are the outer widths of each glyph"""
    total = sum(widths) + gap * (len(chars) - 1)
    x = cx - total / 2.0
    m = np.zeros(X.shape, bool)
    for ch, w in zip(chars, widths):
        x0, y0 = x + t / 2, cy - height / 2 + t / 2
        sx, sy = w - t, height - t
        for pts, closed in glyph(ch):
            m |= stroke(X, Y, [(x0 + px * sx, y0 + py * sy) for px, py in pts], t, closed)
        x += w + gap
    return m


# ----------------------------------------------------------------------------------------------------------------
# 3. The 43 signs.  Coordinates: the sign is one unit wide, origin at the centre of the disc / diamond / octagon,
#    or at the centroid of the triangle (side length 1); x to the right, y downwards.
# ----------------------------------------------------------------------------------------------------------------
S3 = np.sqrt(3.0)
TRI_UP = [(0.0, -S3 / 3), (0.5, S3 / 6), (-0.5, S3 / 6)]
TRI_DN = [(0.0, S3 / 3), (-0.5, -S3 / 6), (0.5, -S3 / 6)]
DIAMOND = [(0.0, -0.5), (0.5, 0.0), (0.0, 0.5), (-0.5, 0.0)]
_o = 0.5 * np.tan(np.pi / 8)
OCTAGON = [(-_o, -0.5), (_o, -0.5), (0.5, -_o), (0.5, _o), (_o, 0.5), (-_o, 0.5), (-0.5, _o), (-0.5, -_o)]
R_TRI, R_DIA = 0.04, 0.035          # corner rounding
RING_W = 0.14                        # red ring of the prohibition discs
B_DANGER, B_YIELD = 0.09, 0.13       # red border of the triangles

SHAPE = {}
for _c in list(range(0, 11)) + [15, 16, 17, 32] + list(range(33, 43)):
    SHAPE[_c] = 'circle'
for _c in [11] + list(range(18, 32)):
    SHAPE[_c] = 'tri_up'
SHAPE[12] = 'diamond'; SHAPE[13] = 'tri_dn'; SHAPE[14] = 'octagon'


def shape_sdf(kind, X, Y):
    if kind == 'circle':
        return np.hypot(X, Y) - 0.5
    if kind == 'tri_up':
        return rounded_sdf(X, Y, TRI_UP, R_TRI)
    if kind == 'tri_dn':
        return rounded_sdf(X, Y, TRI_DN, R_TRI)
    if kind == 'diamond':
        return rounded_sdf(X, Y, DIAMOND, R_DIA)
    if kind == 'octagon':
        return convex_sdf(X, Y, OCTAGON)
    raise ValueError(kind)


def content_sdf(c, X, Y):
    """signed distance (sign units, negative inside) to the boundary of the area that carries the class-specific
    drawing: the white disc inside a red ring, the white triangle inside the red border, the blue disc, ..."""
    kind = SHAPE[c]
    if c == 12:
        return rounded_sdf(X, Y, [(0.0, -0.31), (0.31, 0.0), (0.0, 0.31), (-0.31, 0.0)], 0.02)
    if c == 14:
        return convex_sdf(X, Y, OCTAGON) + 0.03
    if kind == 'circle':
        if c == 17:
            r = 0.5
        elif c in (6, 32, 41, 42):
            r = 0.47
        elif c >= 33:
            r = 0.475
        else:
            r = 0.5 - RING_W
        return np.hypot(X, Y) - r
    if kind == 'tri_up':
        return rounded_sdf(X, Y, shrink_convex(TRI_UP, B_DANGER), 0.012)
    return rounded_sdf(X, Y, shrink_convex(TRI_DN, B_YIELD), 0.01)


BLANK = (13, 15)            # yield and no vehicles: nothing is drawn inside


def shape_bbox(kind):
    """true bounding box (x0, y0, x1, y1) of the sign outline"""
    if kind in ('circle', 'octagon'):
        return (-0.5, -0.5, 0.5, 0.5)
    if kind == 'diamond':
        e = 0.5 - R_DIA * (np.sqrt(2.0) - 1.0)
        return (-e, -e, e, e)
    hw = 0.5 - R_TRI * (S3 - 1.0)
    if kind == 'tri_up':
        return (-hw, -S3 / 3 + R_TRI, hw, S3 / 6)
    return (-hw, -S3 / 6, hw, S3 / 3 - R_TRI)


def car_rear(X, Y, cx, cy, w, h):
    """small car seen from behind: body, roof, two wheels; (cx, cy) is the centre of its box"""
    x0, y0 = cx - w / 2, cy - h / 2
    m = rect(X, Y, x0, y0 + 0.36 * h, x0 + w, y0 + 0.82 * h)
    m |= poly(X, Y, [(x0 + 0.2 * w, y0), (x0 + 0.8 * w, y0), (x0 + 0.93 * w, y0 + 0.4 * h), (x0 + 0.07 * w, y0 + 0.4 * h)])
    m |= rect(X, Y, x0 + 0.06 * w, y0 + 0.8 * h, x0 + 0.3 * w, y0 + h)
    m |= rect(X, Y, x0 + 0.7 * w, y0 + 0.8 * h, x0 + 0.94 * w, y0 + h)
    return m


def truck_rear(X, Y, cx, cy, w, h):
    x0, y0 = cx - w / 2, cy - h / 2
    m = rect(X, Y, x0, y0, x0 + w, y0 + 0.86 * h)
    m |= rect(X, Y, x0 + 0.1 * w, y0 + 0.84 * h, x0 + 0.34 * w, y0 + h)
    m |= rect(X, Y, x0 + 0.66 * w, y0 + 0.84 * h, x0 + 0.9 * w, y0 + h)
    return m


def truck_side(X, Y):
    """lorry seen from the side, driving to the left"""
    m = rect(X, Y, -0.09, -0.135, 0.27, 0.055)                                   # load
    m |= poly(X, Y, [(-0.27, 0.055), (-0.27, -0.01), (-0.225, -0.075), (-0.12, -0.075), (-0.12, 0.055)])   # cab
    m |= rect(X, Y, -0.27, 0.04, 0.27, 0.075)                                    # chassis
    m |= disc(X, Y, -0.185, 0.09, 0.05) | disc(X, Y, 0.16, 0.09, 0.05)
    return m


def band(X, Y, half_width):
    """diagonal band from lower left to upper right"""
    return np.abs(X + Y) / np.sqrt(2.0) <= half_width


def person(X, Y, cx, top, h, lean=0.0, stride=0.4, fat=1.0):
    """simple walking figure of height h whose head top is at `top`"""
    m = disc(X, Y, cx + lean * h * 0.1, top + 0.095 * h, 0.095 * h)
    sh = (cx + lean * h * 0.05, top + 0.25 * h); hip = (cx - lean * h * 0.05, top + 0.58 * h)
    m |= stroke(X, Y, [sh, hip], 0.2 * h * fat)
    m |= stroke(X, Y, [hip, (cx - stride * h * 0.5, top + h)], 0.11 * h * fat)
    m |= stroke(X, Y, [hip, (cx + stride * h * 0.5, top + h)], 0.11 * h * fat)
    m |= stroke(X, Y, [sh, (cx - 0.24 * h, top + 0.5 * h)], 0.085 * h * fat)
    m |= stroke(X, Y, [sh, (cx + 0.24 * h, top + 0.46 * h)], 0.085 * h * fat)
    return m


def draw_sign(c, X, Y):
    """returns an (..., 3) colour image of sign c sampled at X, Y (background BG)"""
    img = np.empty(X.shape + (3,))
    img[:] = BG
    def put(mask, col):
        img[mask] = col
    kind = SHAPE[c]
    sd = shape_sdf(kind, X, Y)
    inside = sd <= 0

    if kind == 'circle' and c not in (6, 32, 41, 42) and c < 33:        # red ring signs and no entry
        put(inside, RED)
        if c == 17:
            put(rect(X, Y, -0.38, -0.08, 0.38, 0.08), WHITE)
            return img
        put(np.hypot(X, Y) <= 0.5 - RING_W, WHITE)
        two = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80'}
        if c in two:
            put(text_mask(X, Y, two[c], [0.25, 0.25], 0.44, 0.045, 0.068), BLACK)
        elif c == 7:
            put(text_mask(X, Y, '100', [0.115, 0.195, 0.195], 0.40, 0.04, 0.058), BLACK)
        elif c == 8:
            put(text_mask(X, Y, '120', [0.115, 0.195, 0.195], 0.40, 0.04, 0.058), BLACK)
        elif c == 9:
            put(car_rear(X, Y, -0.125, 0.0, 0.21, 0.145), RED)
            put(car_rear(X, Y, 0.125, 0.0, 0.21, 0.145), BLACK)
        elif c == 10:
            put(truck_rear(X, Y, -0.145, -0.02, 0.24, 0.27), RED)
            put(car_rear(X, Y, 0.14, 0.045, 0.2, 0.14), BLACK)
        elif c == 16:
            put(truck_side(X, Y), BLACK)
        return img

    if c in (6, 32, 41, 42):                                             # end-of signs: white disc, grey band
        put(inside, WHITE)
        put((np.hypot(X, Y) > 0.47) & inside, DGREY)
        put(band(X, Y, 0.075) & (np.hypot(X, Y) <= 0.47), GREY)          # five thin lines read as one grey band
        if c == 6:
            put(text_mask(X, Y, '80', [0.25, 0.25], 0.44, 0.045, 0.068), DGREY)
        elif c == 41:
            put(car_rear(X, Y, -0.125, 0.0, 0.21, 0.145), DGREY)
            put(car_rear(X, Y, 0.125, 0.0, 0.21, 0.145), DGREY)
        elif c == 42:
            put(truck_rear(X, Y, -0.145, -0.02, 0.24, 0.27), DGREY)
            put(car_rear(X, Y, 0.14, 0.045, 0.2, 0.14), DGREY)
        return img

    if c >= 33:                                                          # blue discs with white arrows
        put(inside, WHITE)
        put(np.hypot(X, Y) <= 0.475, BLUE)
        m = np.zeros(X.shape, bool)
        if c == 35:
            m |= stroke(X, Y, [(0.0, 0.33), (0.0, -0.1)], 0.10)
            m |= poly(X, Y, arrow_head((0.0, -0.38), (0, -1), 0.31, 0.2))
        elif c in (33, 34):
            path = [(-0.15, 0.33), (-0.15, 0.0)] + arc(-0.03, 0.0, 0.12, 0.12, np.pi, 1.5 * np.pi, 10) + [(0.1, -0.12)]
            head = arrow_head((0.38, -0.12), (1, 0), 0.29, 0.2)
            if c == 34:
                path, head = mirror(path), mirror(head)
            m |= stroke(X, Y, path, 0.10) | poly(X, Y, head)
        elif c in (36, 37):
            main = [(-0.12, 0.35), (-0.12, -0.12)]
            head1 = arrow_head((-0.12, -0.38), (0, -1), 0.26, 0.15)
            br = [(-0.12, 0.3)] + arc(0.02, 0.19, 0.14, 0.14, np.pi, 1.5 * np.pi, 10) + [(0.16, 0.05)]
            head2 = arrow_head((0.38, 0.05), (1, 0), 0.23, 0.15)
            if c == 37:
                main, head1, br, head2 = mirror(main), mirror(head1), mirror(br), mirror(head2)
            m |= stroke(X, Y, main, 0.085) | poly(X, Y, head1) | stroke(X, Y, br, 0.085) | poly(X, Y, head2)
        elif c in (38, 39):
            path = [(-0.22, -0.22), (0.08, 0.08)]
            head = arrow_head((0.28, 0.28), (1, 1), 0.29, 0.2)
            if c == 39:
                path, head = mirror(path), mirror(head)
            m |= stroke(X, Y, path, 0.10) | poly(X, Y, head)
        elif c == 40:
            for k in range(3):                                           # three arrows chasing each other anticlockwise
                a0 = -np.pi / 2 + k * 2 * np.pi / 3
                pts = arc(0.0, 0.0, 0.25, 0.25, a0 + 0.35, a0 - 1.15, 14)
                m |= stroke(X, Y, pts, 0.075)
                tx, ty = 0.25 * np.cos(a0 - 1.15), 0.25 * np.sin(a0 - 1.15)
                dx, dy = np.sin(a0 - 1.15), -np.cos(a0 - 1.15)           # anticlockwise tangent (y down)
                m |= poly(X, Y, arrow_head((tx + 0.13 * dx, ty + 0.13 * dy), (dx, dy), 0.17, 0.115))
        put(m & (np.hypot(X, Y) <= 0.475), WHITE)
        return img

    if c == 12:                                                          # priority road
        put(inside, WHITE)
        put((sd > -0.03) & (sd <= -0.012), DGREY)
        put(rounded_sdf(X, Y, [(0.0, -0.31), (0.31, 0.0), (0.0, 0.31), (-0.31, 0.0)], 0.02) <= 0, YELLOW)
        return img

    if c == 13:                                                          # yield
        put(inside, RED)
        put(rounded_sdf(X, Y, shrink_convex(TRI_DN, B_YIELD), 0.01) <= 0, WHITE)
        return img

    if c == 14:                                                          # stop
        put(inside, WHITE)
        put(sd <= -0.03, RED)
        put(text_mask(X, Y, 'STOP', [0.155, 0.155, 0.165, 0.155], 0.35, 0.035, 0.055), WHITE)
        return img

    # ---- danger triangles ----
    put(inside, RED)
    put(rounded_sdf(X, Y, shrink_convex(TRI_UP, B_DANGER), 0.012) <= 0, WHITE)
    m = np.zeros(X.shape, bool)
    if c == 11:                                                          # crossroads with priority
        m |= rect(X, Y, -0.09, -0.1, 0.09, 0.175)                        # very broad upright arrow
        m |= poly(X, Y, arrow_head((0.0, -0.3), (0, -1), 0.22, 0.13))
        m |= rect(X, Y, -0.17, 0.01, 0.17, 0.085)                        # short stubby cross road
    elif c == 18:                                                        # general danger
        m |= poly(X, Y, [(-0.04, -0.27), (0.04, -0.27), (0.026, 0.02), (-0.026, 0.02)])
        m |= disc(X, Y, 0.0, 0.115, 0.04)
    elif c in (19, 20):                                                  # bend
        path = [(0.085, 0.15), (0.085, 0.02)] + arc(-0.035, 0.02, 0.12, 0.12, 0.0, -0.5 * np.pi, 8) + [(-0.06, -0.1)]
        head = arrow_head((-0.21, -0.1), (-1, 0), 0.15, 0.105)
        if c == 20:
            path, head = mirror(path), mirror(head)
        m |= stroke(X, Y, path, 0.105) | poly(X, Y, head)
    elif c == 21:                                                        # double bend, first to the left
        m |= stroke(X, Y, [(0.075, 0.16), (0.075, 0.035), (-0.065, -0.085), (-0.065, -0.25)], 0.095)
    elif c == 22:                                                        # uneven road
        m |= rect(X, Y, -0.22, 0.085, 0.22, 0.13)
        m |= ellipse(X, Y, -0.095, 0.09, 0.075, 0.075) & (Y <= 0.09)
        m |= ellipse(X, Y, 0.095, 0.09, 0.075, 0.075) & (Y <= 0.09)
    elif c == 23:                                                        # slippery road: car above two wavy tracks
        m |= car_rear(X, Y, 0.0, -0.035, 0.28, 0.16)                     # the car is the heavy part
        for sx in (-0.08, 0.08):                                         # the skid marks are thin
            pts = [(sx + 0.035 * np.sin(t * 2 * np.pi), 0.05 + 0.115 * t) for t in np.linspace(0, 1, 14)]
            m |= stroke(X, Y, pts, 0.028)
    elif c == 24:                                                        # road narrows on the right
        m |= stroke(X, Y, [(-0.07, -0.23), (-0.07, 0.16)], 0.045)
        m |= stroke(X, Y, [(0.015, -0.23), (0.015, -0.09), (0.095, 0.01), (0.095, 0.16)], 0.045)
    elif c == 25:                                                        # road works: digging man on the left, heap on the right
        m |= disc(X, Y, -0.045, -0.2, 0.038)
        m |= stroke(X, Y, [(-0.06, -0.14), (-0.115, 0.01)], 0.07)        # bent body
        m |= stroke(X, Y, [(-0.115, 0.01), (-0.175, 0.165)], 0.045)      # legs
        m |= stroke(X, Y, [(-0.115, 0.01), (-0.04, 0.165)], 0.045)
        m |= stroke(X, Y, [(-0.05, -0.12), (0.025, -0.02)], 0.035)       # arms
        m |= stroke(X, Y, [(-0.03, -0.1), (0.075, 0.09)], 0.025)         # shovel
        m |= ellipse(X, Y, 0.125, 0.165, 0.115, 0.095) & (Y <= 0.165)    # heap
    elif c == 26:                                                        # traffic lights
        put(disc(X, Y, 0.0, -0.215, 0.055), RED)
        put(disc(X, Y, 0.0, -0.075, 0.055), YELLOW)
        put(disc(X, Y, 0.0, 0.065, 0.055), GREEN)
    elif c == 27:                                                        # pedestrian
        m |= person(X, Y, 0.0, -0.27, 0.44, lean=0.3, stride=0.42, fat=1.15)
    elif c == 28:                                                        # children
        m |= person(X, Y, -0.09, -0.17, 0.33, lean=-0.3, stride=0.45, fat=1.5)
        m |= person(X, Y, 0.075, -0.235, 0.395, lean=-0.3, stride=0.45, fat=1.5)
    elif c == 29:                                                        # bicycle
        for wx in (-0.125, 0.125):
            m |= stroke(X, Y, arc(wx, 0.065, 0.095, 0.095, 0, 2 * np.pi, 20), 0.032)
        m |= stroke(X, Y, [(-0.125, 0.065), (-0.045, -0.075), (0.07, -0.075), (0.005, 0.065), (-0.125, 0.065)], 0.03)
        m |= stroke(X, Y, [(0.005, 0.065), (-0.045, -0.075), (-0.06, -0.12)], 0.03)
        m |= stroke(X, Y, [(0.125, 0.065), (0.065, -0.115), (0.02, -0.13)], 0.03)
        m |= stroke(X, Y, [(-0.1, -0.125), (-0.03, -0.125)], 0.035)
    elif c == 30:                                                        # snow flake
        for k in range(3):
            a = np.pi / 2 + k * np.pi / 3
            dx, dy = 0.185 * np.cos(a), 0.185 * np.sin(a)
            m |= stroke(X, Y, [(-dx, -dy), (dx, dy)], 0.045)
            for sgn in (-1, 1):
                bx, by = sgn * 0.62 * dx, sgn * 0.62 * dy
                for da in (-0.9, 0.9):
                    ex, ey = 0.065 * np.cos(a + da) * sgn, 0.065 * np.sin(a + da) * sgn
                    m |= stroke(X, Y, [(bx, by), (bx + ex, by + ey)], 0.035)
    elif c == 31:                                                        # leaping deer, heading up and to the left
        m |= stroke(X, Y, [(-0.07, -0.1), (0.1, 0.01)], 0.1)             # body
        m |= stroke(X, Y, [(-0.07, -0.1), (-0.12, -0.19)], 0.05)         # neck
        m |= stroke(X, Y, [(-0.12, -0.19), (-0.165, -0.165)], 0.04)      # head
        m |= stroke(X, Y, [(-0.115, -0.2), (-0.09, -0.27)], 0.022)       # antlers
        m |= stroke(X, Y, [(-0.1, -0.2), (-0.045, -0.25)], 0.022)
        m |= stroke(X, Y, [(-0.06, -0.08), (-0.14, -0.02), (-0.2, -0.03)], 0.03)   # fore legs
        m |= stroke(X, Y, [(-0.03, -0.05), (-0.09, 0.03)], 0.03)
        m |= stroke(X, Y, [(0.1, 0.01), (0.17, 0.03), (0.215, 0.11)], 0.035)       # hind legs
        m |= stroke(X, Y, [(0.08, 0.02), (0.14, 0.09), (0.16, 0.15)], 0.03)
    put(m, BLACK)
    return img


def blur_image(img, sigma):
    """Gaussian blur of a (3, n, n) image, edges replicated"""
    r = int(np.ceil(3 * sigma))
    g = gauss1d(sigma, r)
    p = np.pad(img, ((0, 0), (r, r), (r, r)), mode="edge")
    n = img.shape[1]
    t = sum(g[k] * p[:, k:k + n, :] for k in range(2 * r + 1))
    return sum(g[k] * t[:, :, k:k + n] for k in range(2 * r + 1))


def render(c, size, cx=15.5, cy=15.5, n=N, ss=SS, angle=0.0, blur=0.0):
    """Draw sign c with its bounding box `size` pixels wide and high, centred at pixel (cx, cy).
    Returns image (3, n, n) and two signed distances in pixels (negative inside), each (n, n): to the sign outline
    and to the boundary of the content area."""
    kind = SHAPE[c]
    bx0, by0, bx1, by1 = shape_bbox(kind)
    sub = (np.arange(n * ss) + 0.5) / ss - 0.5                     # pixel-centre coordinates of the samples
    PX, PY = np.meshgrid(sub, sub)
    u, v = (PX - cx) / size, (PY - cy) / size
    if angle:
        ca, sa = np.cos(angle), np.sin(angle)
        u, v = ca * u + sa * v, -sa * u + ca * v
    X = (bx0 + bx1) / 2 + u * (bx1 - bx0)
    Y = (by0 + by1) / 2 + v * (by1 - by0)
    img = draw_sign(c, X, Y)
    img = img.reshape(n, ss, n, ss, 3).mean(axis=(1, 3)).transpose(2, 0, 1)
    if blur > 0:
        img = blur_image(img, blur)
    px_per_unit = size / (0.5 * ((bx1 - bx0) + (by1 - by0)))
    sd = (shape_sdf(kind, X, Y) * px_per_unit).reshape(n, ss, n, ss).mean(axis=(1, 3))
    cd = (content_sdf(c, X, Y) * px_per_unit).reshape(n, ss, n, ss).mean(axis=(1, 3))
    return img, (sd, cd)


# ----------------------------------------------------------------------------------------------------------------
# 4. The network front end: layers and weights from formulas
# ----------------------------------------------------------------------------------------------------------------

def gauss1d(sigma, radius):
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


def gauss2d(sigma, radius):
    g = gauss1d(sigma, radius)
    return np.outer(g, g)


def build_front():
    """returns (layers, weights) of everything before the template convolution; the last layer is named 'feat'"""
    L, W = [], {}

    def conv(name, inp, w, b, pad=0, stride=1):
        o, i, k, _ = w.shape
        L.append({"name": name, "op": "conv", "inputs": [inp], "out": int(o), "kernel": [int(k), int(k)], "stride": stride, "padding": pad})
        W[name + ".W"] = w.astype(np.float32); W[name + ".b"] = np.asarray(b, np.float32)

    def op(name, kind, inputs, **kw):
        L.append(dict({"name": name, "op": kind, "inputs": list(inputs)}, **kw))

    # (a) light blur against the jagged nearest-neighbour resampling
    b1 = np.array([1.0, 4.0, 1.0]) / 6.0
    w = np.zeros((3, 3, 3, 3))
    for c in range(3):
        w[c, c] = np.outer(b1, b1)
    conv("pre", "image", w, np.zeros(3), pad=0)                       # 32 -> 30, no border artefact

    # (b) log(x + eps) as a sum of hinges with geometrically spaced knots
    r = ((1.0 + EPS_LOG) / EPS_LOG) ** (1.0 / N_KNOTS)
    knots = EPS_LOG * r ** np.arange(N_KNOTS + 1) - EPS_LOG          # 0 ... 1
    vals = np.log(knots + EPS_LOG)
    slopes = np.diff(vals) / np.diff(knots)
    dsl = np.diff(np.concatenate([[0.0], slopes]))                    # slope changes at knots 0 .. K-1
    w = np.zeros((3 * N_KNOTS, 3, 1, 1)); b = np.zeros(3 * N_KNOTS)
    w2 = np.zeros((3, 3 * N_KNOTS, 1, 1))
    for c in range(3):
        for k in range(N_KNOTS):
            w[c * N_KNOTS + k, c, 0, 0] = 1.0
            b[c * N_KNOTS + k] = -knots[k]
            w2[c, c * N_KNOTS + k, 0, 0] = dsl[k]
    conv("hinge", "pre", w, b)
    op("hinge_r", "relu", ["hinge"])
    conv("logim", "hinge_r", w2, np.full(3, vals[0]))

    # (c) standardise the whole log image with statistics weighted towards the centre:
    #     a second copy of the image in which centre pixels are repeated (Gaussian window) is appended,
    #     the pair is normalised together, and the first copy is kept.
    P = M * M
    mid = (M - 1) / 2.0
    yy, xx = np.mgrid[0:M, 0:M]
    g = np.exp(-((yy - mid) ** 2 + (xx - mid) ** 2) / (2 * WIN_SIGMA ** 2)).ravel()
    q = g / g.sum() * P
    m = np.floor(q + 1e-9).astype(int)
    rest = P - m.sum()
    order = np.argsort(-(np.round(q - m, 9)), kind="stable")
    m[order[:rest]] += 1                                              # largest remainders: the counts sum to P
    src = np.repeat(np.arange(P), m)
    rep = np.zeros((3 * P, 3 * P), np.float32)
    for c in range(3):
        rep[c * P + np.arange(P), c * P + src] = 1.0
    op("log_flat", "reshape", ["logim"], shape=[1, 1, 3 * P])
    L.append({"name": "rep", "op": "dense", "inputs": ["logim"], "out": 3 * P})
    W["rep.W"] = rep; W["rep.b"] = np.zeros(3 * P, np.float32)
    op("rep_flat", "reshape", ["rep"], shape=[1, 1, 3 * P])
    conv("row0", "log_flat", np.array([1.0, 0.0]).reshape(2, 1, 1, 1), np.zeros(2))
    conv("row1", "rep_flat", np.array([0.0, 1.0]).reshape(2, 1, 1, 1), np.zeros(2))
    op("rows", "add", ["row0", "row1"])
    op("rows2", "reshape", ["rows"], shape=[1, 2, 3 * P])
    op("rows_n", "norm", ["rows2"])
    op("first", "maxpool", ["rows_n"], size=[1, 1], stride=[2, 1])
    op("z", "reshape", ["first"], shape=[3, M, M])

    gA, gB, gC, gO = np.sqrt(W_A), np.sqrt(W_B), np.sqrt(W_COL), np.sqrt(W_ORI)
    lum = np.full(3, 1.0 / 3.0)

    # (d) "paint brightness" v = max(red, green): white, yellow and red paint are bright, blue and black are dark.
    #     (With plain luminance a red ring is dark, and any dark background passes for a ring.)
    w = np.zeros((3, 3, 1, 1))
    w[0, 1, 0, 0] = 1.0; w[0, 0, 0, 0] = -1.0          # g - r
    w[1, 0, 0, 0] = 1.0; w[2, 0, 0, 0] = -1.0          # r, -r
    conv("mx", "z", w, np.zeros(3))
    op("mx_r", "relu", ["mx"])
    conv("v", "mx_r", np.array([1.0, 1.0, -1.0]).reshape(1, 3, 1, 1), np.zeros(1))    # r + relu(g - r)

    # brightness A: v clipped
    conv("pa", "v", np.array([1.0 / C_A, 1.0 / C_A]).reshape(2, 1, 1, 1), np.array([1.0, -1.0]))
    op("pa_r", "relu", ["pa"])
    w = np.zeros((NF, 2, 1, 1)); b = np.zeros(NF)
    w[0, 0, 0, 0] = gA; w[0, 1, 0, 0] = -gA; b[0] = -gA
    conv("featA", "pa_r", w, b)

    # colour: the differences R-G, G-B, B-R are blurred first (colour fringes at black/white edges come in
    # opposite pairs and cancel), then passed through a ramp.  Red and blue share one signed channel, so that a
    # red ring counts against a blue-disc template and vice versa; yellow has its own channel.
    rc = 2
    gc = gauss2d(SIG_COL, rc)
    w = np.zeros((6, 3, 2 * rc + 1, 2 * rc + 1)); b = np.zeros(6)
    opp = [(0, 1), (1, 2), (2, 0)]
    for k, (p, n_) in enumerate(opp):
        for j, thr in enumerate((COL_LO, COL_HI)):
            w[2 * k + j, p] = gc; w[2 * k + j, n_] = -gc; b[2 * k + j] = -thr
    conv("pc", "z", w, b, pad=rc)
    op("pc_r", "relu", ["pc"])
    w = np.zeros((NF, 6, 1, 1))
    q = gC / (COL_HI - COL_LO)
    w[2, 0, 0, 0] = q; w[2, 1, 0, 0] = -q            # + red
    w[2, 4, 0, 0] = -q; w[2, 5, 0, 0] = q            # - blue
    w[3, 2, 0, 0] = q; w[3, 3, 0, 0] = -q            # yellow
    conv("featC", "pc_r", w, np.zeros(NF))

    # (e) local contrast B: v minus its Gaussian surround, clipped
    rb = 4
    hp = -gauss2d(SIG_B, rb); hp[rb, rb] += 1.0
    w = np.zeros((2, 1, 2 * rb + 1, 2 * rb + 1))
    w[0, 0] = hp / C_B; w[1, 0] = hp / C_B
    conv("bp", "v", w, np.array([1.0, -1.0]), pad=rb)
    op("bp_r", "relu", ["bp"])
    w = np.zeros((NF, 2, 1, 1)); b = np.zeros(NF)
    w[1, 0, 0, 0] = gB; w[1, 1, 0, 0] = -gB; b[1] = -gB
    conv("featB", "bp_r", w, b)

    # (f) unsigned edge orientation: |directional derivative| minus cot(30 deg) * |perpendicular derivative|
    sob_x = np.outer([1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]) / 8.0
    sob_y = sob_x.T
    w = np.zeros((2 * N_ORI, 3, 3, 3))
    for k in range(N_ORI):
        th = np.pi * k / N_ORI
        d = np.cos(th) * sob_x + np.sin(th) * sob_y
        for c in range(3):
            w[2 * k, c] = d * lum[c]; w[2 * k + 1, c] = -d * lum[c]
    conv("grad", "z", w, np.zeros(2 * N_ORI), pad=0)                  # 30 -> 28: no false edge at the frame
    op("grad_r", "relu", ["grad"])
    cot = 1.0 / np.tan(np.pi / N_ORI)
    w = np.zeros((2 * N_ORI, 2 * N_ORI, 1, 1)); b = np.zeros(2 * N_ORI)
    for k in range(N_ORI):
        kp = (k + N_ORI // 2) % N_ORI
        for j, thr in enumerate((ORI_DEAD, ORI_DEAD + ORI_CAP)):
            w[2 * k + j, 2 * k, 0, 0] = 1.0; w[2 * k + j, 2 * k + 1, 0, 0] = 1.0
            w[2 * k + j, 2 * kp, 0, 0] = -cot; w[2 * k + j, 2 * kp + 1, 0, 0] = -cot
            b[2 * k + j] = -thr
    conv("ori", "grad_r", w, b)
    op("ori_r", "relu", ["ori"])
    ro = 2
    go = gauss2d(SIG_ORI, ro)
    w = np.zeros((NF, 2 * N_ORI, 2 * ro + 1, 2 * ro + 1))
    for k in range(N_ORI):
        w[ORI0 + k, 2 * k] = gO * go / ORI_CAP
        w[ORI0 + k, 2 * k + 1] = -gO * go / ORI_CAP
    conv("featO", "ori_r", w, np.zeros(NF), pad=ro + 1)               # 28 -> 30, re-centred

    op("feat", "add", ["featA", "featC", "featB", "featO"])
    return L, W


# ----------------------------------------------------------------------------------------------------------------
# 5. A small interpreter with the same semantics as the scorer's building blocks (used to pass drawings through
#    the front end so that every drawing becomes its own template)
# ----------------------------------------------------------------------------------------------------------------

def run_layers(layers, weights, images, want):
    vals = {"image": images.astype(np.float64)}
    for Ly in layers:
        ins = [vals[i] for i in Ly["inputs"]]
        kind, name = Ly["op"], Ly["name"]
        if kind == "conv":
            w = weights[name + ".W"].astype(np.float64); b = weights[name + ".b"].astype(np.float64)
            x = ins[0]; p = Ly.get("padding", 0); s = Ly.get("stride", 1)
            n, c, h, wd = x.shape; o, _, kh, kw = w.shape
            xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
            ho, wo = (h + 2 * p - kh) // s + 1, (wd + 2 * p - kw) // s + 1
            out = np.zeros((n, o, ho, wo))
            for dy in range(kh):
                for dx in range(kw):
                    wk = w[:, :, dy, dx]
                    if not wk.any():
                        continue
                    out += np.einsum("nchw,oc->nohw", xp[:, :, dy:dy + s * ho:s, dx:dx + s * wo:s], wk, optimize=True)
            vals[name] = out + b[None, :, None, None]
        elif kind == "dense":
            vals[name] = ins[0].reshape(len(ins[0]), -1) @ weights[name + ".W"].astype(np.float64).T + weights[name + ".b"]
        elif kind == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif kind in ("maxpool", "avgpool"):
            x = ins[0]; (sh, sw), (th, tw) = Ly["size"], Ly["stride"]
            n, c, h, wd = x.shape
            ho, wo = (h - sh) // th + 1, (wd - sw) // tw + 1
            win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
            vals[name] = win.max(0) if kind == "maxpool" else win.mean(0)
        elif kind == "add":
            vals[name] = sum(ins)
        elif kind == "reshape":
            vals[name] = ins[0].reshape((len(ins[0]),) + tuple(Ly["shape"]))
        elif kind == "norm":
            x = ins[0]
            mu = x.mean(axis=(2, 3), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        else:
            raise ValueError(kind)
        if name == want:
            return vals[name]
    return vals[want]


# ----------------------------------------------------------------------------------------------------------------
# 6. Templates: every drawing -> features -> masked, mean-free, unit-norm kernel
# ----------------------------------------------------------------------------------------------------------------
MARGIN_IN = 0.75      # brightness / contrast channels: use the sign interior, this many pixels inside the outline
MARGIN_OUT = 2.0      # colour / edge channels: also use a rim this many pixels outside the outline
# the content area stops this many pixels inside the frame, more for the channels that smear further
MARGIN_CONTENT = {"A": 0.75, "B": 1.5, "COL": 0.75, "ORI": 2.0}
MARGIN_BLANK = 2.5    # area over which an empty interior is checked for edges


def soft_mask(sd, offset):
    """1 where the signed distance is below `offset`, with a one pixel linear ramp"""
    return np.clip(offset - sd + 0.5, 0.0, 1.0)


def unit_part(f, masks):
    """mask every channel, remove its own mean inside the mask, scale the whole part to unit norm"""
    t = np.zeros_like(f)
    for ch in range(NF):
        m = masks[ch]
        if m.sum() > 0:
            t[ch] = (f[ch] - (f[ch] * m).sum() / m.sum()) * m
    nrm = np.sqrt((t * t).sum())
    return t / nrm if nrm > 1e-9 else t


def channel_kind(ch):
    return "A" if ch == 0 else "B" if ch == 1 else "COL" if ch < ORI0 else "ORI"


def make_template(c, feat, dist):
    """One drawing -> one kernel.  feat (NF, 30, 30): features of the drawing; dist: its two signed-distance maps.
    The kernel is  frame part + KAPPA * content part.  Both parts come from this drawing alone and each is
    normalised by itself.  Frame: the sign outside its content area (outline, border, rim just outside) plus the
    colour layout of the whole sign.  Content: the inside (digits, pictogram, arrow).
    Normalising the inside separately means a pictogram is judged by its pattern, not by how much ink it has
    (otherwise the sign with the emptiest interior wins whenever a photo is soft).
    Returns (kernel, bias)."""
    o = (N - M) // 2
    sd = dist[0][o:o + M, o:o + M]
    cd = dist[1][o:o + M, o:o + M]
    base = {"A": soft_mask(sd, -MARGIN_IN), "B": soft_mask(sd, -MARGIN_IN),
            "COL": soft_mask(sd, MARGIN_OUT), "ORI": soft_mask(sd, MARGIN_OUT)}
    m_con, m_frame = [], []
    for ch in range(NF):
        k = channel_kind(ch)
        mc = soft_mask(cd, MARGIN_CONTENT[k] if c == 12 else -MARGIN_CONTENT[k])
        m_con.append(mc)
        m_frame.append(base[k] if k == "COL" else base[k] * (1.0 - mc))
    frame = unit_part(feat, m_frame)
    if c in BLANK:
        # nothing to match inside: reward an interior without edges instead
        con = np.zeros_like(feat)
        con[ORI0:] = -BLANK_ETA / np.sqrt(W_ORI) * soft_mask(cd, -MARGIN_BLANK)
        return frame + KAPPA * con, KAPPA * BLANK_TAU
    con = unit_part(feat, m_con)
    return frame + KAPPA * con, 0.0


def variants():
    """the list of (size, angle, blur) at which every class is drawn"""
    return [(s, np.deg2rad(a), b) for s in SIZES for a, b in VERSIONS]


def build():
    layers, weights = build_front()
    var = variants()
    V = len(var)
    imgs, sds, cls = [], [], []
    for c in range(N_CLASSES):
        for size, ang, bl in var:
            im, sd = render(c, size, angle=ang, blur=bl)
            imgs.append(im); sds.append(sd); cls.append(c)
    imgs = np.array(imgs)
    feats = np.concatenate([run_layers(layers, weights, imgs[i:i + 64], "feat") for i in range(0, len(imgs), 64)])
    parts = [make_template(cls[i], feats[i], sds[i]) for i in range(len(imgs))]
    kern = np.stack([p[0] for p in parts]); bias = np.array([p[1] for p in parts])
    nt = len(kern)
    layers.append({"name": "match", "op": "conv", "inputs": ["feat"], "out": nt, "kernel": [KT, KT], "stride": 1, "padding": PAD_T})
    weights["match.W"] = kern.astype(np.float32); weights["match.b"] = bias.astype(np.float32)
    npos = M + 2 * PAD_T - KT + 1
    layers.append({"name": "best_pos", "op": "maxpool", "inputs": ["match"], "size": [npos, npos], "stride": [npos, npos]})
    layers.append({"name": "by_class", "op": "reshape", "inputs": ["best_pos"], "shape": [1, N_CLASSES, V]})
    layers.append({"name": "best_var", "op": "maxpool", "inputs": ["by_class"], "size": [1, V], "stride": [1, V]})
    layers.append({"name": "scores", "op": "reshape", "inputs": ["best_var"], "shape": [N_CLASSES]})
    return {"layers": layers}, weights


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    net, weights = build()
    with open(out + "/network.json", "w") as f:
        json.dump(net, f)
    np.savez(out + "/weights.npz", **weights)
