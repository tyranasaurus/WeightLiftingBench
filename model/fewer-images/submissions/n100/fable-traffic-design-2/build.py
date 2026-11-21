"""build.py -- writes network.json and weights.npz for the traffic-sign network.

Everything here comes from the code itself: the 43 signs are drawn as vector shapes from the German road-sign designs
(rings, triangles, digits as strokes, arrows, pictograms), each drawing is rendered into the 32x32 frame at a grid of
sizes / aspect ratios / small rotations / two blur levels, passed through the hand-designed front end (log colour,
per-image normalisation, opponent colours, saturated oriented edges) and used as its own matched-filter template,
normalised by itself alone.  No file is read, no training data or statistics of any images are used, nothing is fitted.

usage: python3 build.py OUT_DIR
"""
import json, sys
import numpy as np

# ====================================================================================================
# draw.py
# ====================================================================================================
# ----------------------------------------------------------------------------------------------
# Tiny vector rasteriser.  Signs are drawn in "true" coordinates: x in [0, 1] across the sign's
# bounding box, y in [0, asp] downwards (asp = height / width of the real sign).
# ----------------------------------------------------------------------------------------------
NS = 400          # supersampled raster of the bounding box
NC = 200          # canonical raster kept per sign

WHITE = np.array([0.92, 0.92, 0.92]); BLACK = np.array([0.12, 0.12, 0.12])
RED = np.array([0.68, 0.13, 0.14]); BLUE = np.array([0.10, 0.28, 0.62])
YELLOW = np.array([0.90, 0.68, 0.08]); GREY = np.array([0.26, 0.26, 0.26])
TL_RED = np.array([0.80, 0.15, 0.15]); TL_YEL = np.array([0.92, 0.72, 0.10]); TL_GRN = np.array([0.15, 0.62, 0.32])


class Cv:
    def __init__(self, asp=1.0):
        self.asp = asp
        t = (np.arange(NS) + 0.5) / NS
        self.X, self.Y = np.meshgrid(t, t * asp)
        self.rgb = np.zeros((3, NS, NS)); self.alpha = np.zeros((NS, NS), bool); self.inner = np.zeros((NS, NS), bool)

    # --- masks ---
    def disc(self, cx, cy, r, ry=None):
        ry = r if ry is None else ry
        return ((self.X - cx) / r) ** 2 + ((self.Y - cy) / ry) ** 2 <= 1.0

    def poly(self, pts):
        pts = np.asarray(pts, float); n = len(pts); inside = np.zeros(self.X.shape, bool)
        for i in range(n):
            x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
            if y0 == y1: continue
            c = ((y0 <= self.Y) != (y1 <= self.Y)) & (self.X < x0 + (self.Y - y0) * (x1 - x0) / (y1 - y0))
            inside ^= c
        return inside

    def segd(self, p0, p1):
        x0, y0 = p0; x1, y1 = p1; dx, dy = x1 - x0, y1 - y0; L2 = dx * dx + dy * dy
        if L2 < 1e-18: return np.hypot(self.X - x0, self.Y - y0)
        t = np.clip(((self.X - x0) * dx + (self.Y - y0) * dy) / L2, 0, 1)
        return np.hypot(self.X - x0 - t * dx, self.Y - y0 - t * dy)

    def pathd(self, pts, closed=False):
        pts = [tuple(p) for p in pts]; d = np.full(self.X.shape, 1e9)
        segs = list(zip(pts[:-1], pts[1:])) + ([(pts[-1], pts[0])] if closed else [])
        for a, b in segs: d = np.minimum(d, self.segd(a, b))
        return d

    def stroke(self, pts, w, closed=False):
        return self.pathd(pts, closed) <= w / 2

    def polysd(self, pts):
        d = self.pathd(pts, closed=True); return np.where(self.poly(pts), -d, d)

    # --- painting ---
    def paint(self, mask, col):
        self.rgb[:, mask] = np.asarray(col, float)[:, None]

    def shape(self, mask):
        self.alpha |= mask

    def result(self):
        f = NS // NC
        a = self.alpha.astype(float)
        rgb = (self.rgb * a).reshape(3, NC, f, NC, f).mean((2, 4))
        a = a.reshape(NC, f, NC, f).mean((1, 3))
        inn = self.inner.astype(float).reshape(NC, f, NC, f).mean((1, 3))
        return np.concatenate([rgb, a[None], inn[None]], 0)      # premultiplied RGB, alpha, interior region: (5, NC, NC)


def arc(cx, cy, rx, ry, t0, t1, n=24):
    t = np.radians(np.linspace(t0, t1, n))
    return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))


def rot(pts, ang, c):
    a = np.radians(ang); ca, sa = np.cos(a), np.sin(a)
    return [(c[0] + (x - c[0]) * ca - (y - c[1]) * sa, c[1] + (x - c[0]) * sa + (y - c[1]) * ca) for x, y in pts]


def mirror(pts):
    return [(1.0 - x, y) for x, y in pts]


# ----------------------------------------------------------------------------------------------
# Glyphs (DIN-like digits and letters), drawn as centre-line strokes inside a box.
# ----------------------------------------------------------------------------------------------
def glyph_paths(ch, x0, y0, w, h, s):
    """Centre-line paths [(pts, closed)] for a character in the box (x0, y0, w, h) with stroke width s."""
    W, H = w - s, h - s
    r = H / W                                   # centre-line box is 1 x r in units of W

    def P(pts): return [(x0 + s / 2 + u * W, y0 + s / 2 + v * W) for u, v in pts]
    out = []
    if ch in '0O':
        pts = arc(.5, .5, .5, .5, 180, 360) + arc(.5, r - .5, .5, .5, 0, 180)
        out.append((P(pts), True))
    elif ch == '8':
        j = 0.47 * r
        out.append((P(arc(.5, j / 2, .44, j / 2, 0, 360, 40)), True))
        out.append((P(arc(.5, (j + r) / 2, .5, (r - j) / 2, 0, 360, 40)), True))
    elif ch == '3':
        j = 0.47 * r
        out.append((P(arc(.5, j / 2, .44, j / 2, 205, 450, 30)), False))
        out.append((P(arc(.5, (j + r) / 2, .5, (r - j) / 2, 270, 515, 30)), False))
        out.append((P([(.30, j), (.5, j)]), False))
    elif ch == '2':
        a = arc(.5, .5, .5, .5, 190, 395, 28)
        out.append((P(a + [(0.0, r), (1.0, r)]), False))
    elif ch == '5':
        cy, ry = 0.68 * r, 0.32 * r
        a = arc(.48, cy, .52, ry, 215, 510, 34)
        out.append((P([(.95, 0), (.08, 0)] + a), False))
    elif ch == '6':
        cy, ry = 0.69 * r, 0.31 * r
        out.append((P(arc(.5, cy, .5, ry, 0, 360, 40)), True))
        out.append((P([(0, cy)] + arc(.5, 0.3 * r, .5, 0.3 * r, 180, 322, 20)), False))
    elif ch == '7':
        out.append((P([(0, 0), (1, 0), (.32, r)]), False))
    elif ch == '1':
        out.append((P([(0.0, 0.30 * r), (1.0, 0), (1.0, r)]), False))
    elif ch == 'S':
        up = arc(.5, .25 * r, .5, .25 * r, 325, 90, 26)
        lo = arc(.5, .75 * r, .5, .25 * r, 270, 505, 26)
        out.append((P(up + lo), False))
    elif ch == 'T':
        out.append((P([(0, 0), (1, 0)]), False)); out.append((P([(.5, 0), (.5, r)]), False))
    elif ch == 'P':
        b = 0.27 * r
        out.append((P([(0, r), (0, 0), (.5, 0)] + arc(.5, b, .5, b, 270, 450, 16) + [(0, 2 * b)]), False))
    return out


def text(cv, chars, xc, yc, h, widths, s, gap, col):
    tot = sum(widths) + gap * (len(chars) - 1); x = xc - tot / 2
    for ch, w in zip(chars, widths):
        for pts, closed in glyph_paths(ch, x, yc - h / 2, w, h, s):
            cv.paint(cv.stroke(pts, s, closed), col)
        x += w + gap


def arrow_poly(tail, tip, sw, hl, hw):
    tail = np.asarray(tail, float); tip = np.asarray(tip, float)
    d = tip - tail; L = np.hypot(*d); d = d / L; n = np.array([-d[1], d[0]])
    b = tip - d * hl
    return [tuple(tail + n * sw / 2), tuple(b + n * sw / 2), tuple(b + n * hw / 2), tuple(tip),
            tuple(b - n * hw / 2), tuple(b - n * sw / 2), tuple(tail - n * sw / 2)]


# ====================================================================================================
# signs.py
# ====================================================================================================
TRI = 0.8660254


# ---------- base shapes ----------
BORDER = True          # False: draw the sign with its coloured border left white ("content only" drawing)


def base_ring():
    cv = Cv(1.0); R = np.hypot(cv.X - .5, cv.Y - .5)
    cv.shape(R <= .5); cv.paint(R <= .5, WHITE)
    if BORDER: cv.paint(R <= .483, RED); cv.paint(R <= .35, WHITE)
    cv.inner = R <= .34
    return cv


def base_end():
    cv = Cv(1.0); R = np.hypot(cv.X - .5, cv.Y - .5)
    cv.shape(R <= .5); cv.paint(R <= .5, WHITE)
    if BORDER: cv.paint((R <= .485) & (R >= .462), BLACK)
    cv.inner = R <= .34
    return cv


def end_stripes(cv):
    R = np.hypot(cv.X - .5, cv.Y - .5)
    u = (cv.X - .5 + cv.Y - .5) / np.sqrt(2)          # distance across the lower-left -> upper-right diagonal
    m = np.zeros(cv.X.shape, bool)
    for k in range(-2, 3): m |= np.abs(u - k * 0.045) <= 0.017
    cv.paint(m & (R <= .462), BLACK)


def base_blue():
    cv = Cv(1.0); R = np.hypot(cv.X - .5, cv.Y - .5)
    cv.shape(R <= .5); cv.paint(R <= .5, WHITE); cv.paint(R <= .483, BLUE)
    cv.inner = R <= .34
    return cv


def base_tri(up=True, band=0.085, rim=0.011, rho=0.04):
    cv = Cv(TRI)
    if up: v = [(.5, 0), (1, TRI), (0, TRI)]
    else: v = [(0, 0), (1, 0), (.5, TRI)]
    c = np.array([.5, TRI * (2 / 3 if up else 1 / 3)])
    # inner (sharp) triangle whose dilation by rho gives the rounded outline
    k = 1 - rho / (TRI / 3)
    vin = [tuple(c + (np.array(p) - c) * k) for p in v]
    sd = cv.polysd(vin) - rho                          # <= 0 inside the rounded triangle
    cv.shape(sd <= 0); cv.paint(sd <= 0, WHITE)
    if BORDER: cv.paint(sd <= -rim, RED); cv.paint(sd <= -(rim + band), WHITE)
    cv.inner = sd <= -(rim + band + 0.012)
    return cv


# ---------- pictogram parts ----------
def car_rear(cv, cx, yb, col, w=0.27, hh=0.20):
    """Car seen from behind: bottom of the wheels at yb."""
    yt = yb - hh
    cab = [(cx - 0.07, yt), (cx + 0.07, yt), (cx + 0.095, yt + 0.085), (cx - 0.095, yt + 0.085)]
    cv.paint(cv.poly(cab), col)
    body = cv.stroke([(cx - w / 2 + 0.03, yt + 0.12), (cx + w / 2 - 0.03, yt + 0.12)], 0.075)
    cv.paint(body, col)
    for sgn in (-1, 1):
        x = cx + sgn * (w / 2 - 0.045)
        cv.paint(cv.poly([(x - 0.03, yt + 0.13), (x + 0.03, yt + 0.13), (x + 0.03, yb), (x - 0.03, yb)]), col)


def truck_rear(cv, cx, yb, col, w=0.25, hh=0.28):
    yt = yb - hh
    cv.paint(cv.poly([(cx - w / 2, yt), (cx + w / 2, yt), (cx + w / 2, yb - 0.045), (cx - w / 2, yb - 0.045)]), col)
    for sgn in (-1, 1):
        x = cx + sgn * (w / 2 - 0.05)
        cv.paint(cv.poly([(x - 0.035, yb - 0.05), (x + 0.035, yb - 0.05), (x + 0.035, yb), (x - 0.035, yb)]), col)


def truck_side(cv, col):
    cv.paint(cv.poly([(.42, .355), (.80, .355), (.80, .575), (.42, .575)]), col)                     # cargo box
    cv.paint(cv.poly([(.27, .435), (.40, .435), (.40, .575), (.21, .575), (.21, .505)]), col)        # cab
    cv.paint(cv.poly([(.21, .56), (.80, .56), (.80, .605), (.21, .605)]), col)                       # chassis
    cv.paint(cv.disc(.32, .615, .043), col); cv.paint(cv.disc(.69, .615, .043), col)


def speed(chars):
    cv = base_ring()
    if len(chars) == 2:
        text(cv, chars, .5, .5, .42, [.24, .24], .062, .05, BLACK)
    else:
        ws = [.11 if c == '1' else .185 for c in chars]
        text(cv, chars, .5, .5, .39, ws, .052, .04, BLACK)
    return cv


def blue_arrow(cv, pts):
    cv.paint(cv.poly(pts), WHITE)


def s_turn(mir):
    cv = base_blue(); f = mirror if mir else (lambda p: p)
    cl = [(.37, .82), (.37, .52)] + arc(.51, .52, .14, .14, 180, 270, 12) + [(.62, .38)]
    cv.paint(cv.stroke(f(cl), .11), WHITE)
    cv.paint(cv.poly(f([(.37 - .055, .84), (.37 + .055, .84), (.37 + .055, .78), (.37 - .055, .78)])), WHITE)  # flat foot
    cv.paint(cv.poly(f([(.60, .19), (.86, .38), (.60, .57)])), WHITE)
    return cv


def s_ahead():
    cv = base_blue(); blue_arrow(cv, arrow_poly((.5, .85), (.5, .13), .115, .29, .40)); return cv


def s_straight_or(mir):
    cv = base_blue(); f = mirror if mir else (lambda p: p)
    blue_arrow(cv, f(arrow_poly((.40, .85), (.40, .13), .095, .25, .30)))
    cl = [(.40, .74)] + arc(.56, .74, .16, .16, 180, 270, 12) + [(.66, .58)]
    cv.paint(cv.stroke(f(cl), .085), WHITE)
    cv.paint(cv.poly(f([(.64, .445), (.87, .58), (.64, .715)])), WHITE)
    return cv


def s_keep(mir):
    cv = base_blue(); d = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)]); c = np.array([.5, .5])
    pts = arrow_poly(c - .36 * d, c + .36 * d, .115, .29, .40)
    blue_arrow(cv, mirror(pts) if mir else pts); return cv


def s_roundabout():
    cv = base_blue(); r = 0.27
    for t0 in (318.0, 198.0, 78.0):                   # tail angle; arrows run towards decreasing angle (counter-clockwise on screen)
        cv.paint(cv.stroke(arc(.5, .5, r, r, t0, t0 - 72, 16), .07), WHITE)
        tb, tt = np.radians(t0 - 68), np.radians(t0 - 104)
        b = np.array([.5 + r * np.cos(tb), .5 + r * np.sin(tb)]); rad = np.array([np.cos(tb), np.sin(tb)])
        tip = np.array([.5 + r * np.cos(tt), .5 + r * np.sin(tt)])
        cv.paint(cv.poly([tuple(b + rad * .09), tuple(b - rad * .09), tuple(tip)]), WHITE)
    return cv


# ---------- danger-sign pictograms (triangle, true coords: y in [0, 0.866]) ----------
def p_general(cv):
    cv.paint(cv.poly([(.462, .31), (.538, .31), (.524, .60), (.476, .60)]), BLACK)
    cv.paint(cv.disc(.5, .685, .04), BLACK)


def p_crossroad(cv):
    cv.paint(cv.poly([(.5, .285), (.65, .47), (.57, .47), (.57, .745), (.43, .745), (.43, .47), (.35, .47)]), BLACK)
    cv.paint(cv.poly([(.31, .545), (.69, .545), (.69, .615), (.31, .615)]), BLACK)


def bend_pts():
    """Bend to the left: stem lower right, turning towards the upper left."""
    ang = np.radians(12.0); d = np.array([-np.cos(ang), -np.sin(ang)])
    stem = [(.61, .745), (.61, .56)]
    a = arc(.55, .56, .06, .06, 0, -(90 - 12), 8)
    end = np.array(a[-1]); base = end + d * 0.065
    n = np.array([-d[1], d[0]])
    head = [tuple(base + n * .09), tuple(base + d * .13), tuple(base - n * .09)]
    return stem + a + [tuple(base)], head


def p_bend(cv, mir):
    cl, head = bend_pts(); f = mirror if mir else (lambda p: p)
    cv.paint(cv.stroke(f(cl), .078), BLACK); cv.paint(cv.poly(f(head)), BLACK)
    x = .61; cv.paint(cv.poly(f([(x - .039, .70), (x + .039, .70), (x + .039, .76), (x - .039, .76)])), BLACK)


def p_double(cv):
    cl = [(.585, .75), (.585, .625), (.435, .535), (.435, .44)]
    cv.paint(cv.stroke(cl, .085), BLACK)
    cv.paint(cv.poly([(.585 - .0425, .70), (.585 + .0425, .70), (.585 + .0425, .76), (.585 - .0425, .76)]), BLACK)
    cv.paint(cv.poly([(.435 - .085, .455), (.435, .33), (.435 + .085, .455)]), BLACK)


def p_bumpy(cv):
    cv.paint(cv.poly([(.25, .665), (.75, .665), (.75, .71), (.25, .71)]), BLACK)
    for cx in (.385, .615):
        cv.paint(cv.disc(cx, .672, .095, .085) & (cv.Y <= .69), BLACK)


def p_slippery(cv):
    c = (.5, .50); A = -9
    cv.paint(cv.poly(rot([(.375, .455), (.625, .455), (.635, .56), (.365, .56)], A, c)), BLACK)       # body
    cv.paint(cv.poly(rot([(.415, .385), (.585, .385), (.61, .46), (.39, .46)], A, c)), BLACK)         # cabin
    for x in (.40, .60):
        cv.paint(cv.poly(rot([(x - .032, .55), (x + .032, .55), (x + .032, .605), (x - .032, .605)], A, c)), BLACK)
    for x0 in (.41, .575):                             # two wavy skid marks
        t = np.linspace(0, 1, 14)
        pts = list(zip(x0 + 0.04 * np.sin(t * 2 * np.pi), .635 + .095 * t))
        cv.paint(cv.stroke(pts, .024), BLACK)


def p_narrows(cv):
    cv.paint(cv.poly([(.40, .40), (.455, .40), (.455, .745), (.40, .745)]), BLACK)
    cl = [(.635, .745), (.635, .625), (.565, .535), (.565, .40)]
    cv.paint(cv.stroke(cl, .055), BLACK)
    cv.paint(cv.poly([(.6075, .70), (.6625, .70), (.6625, .745), (.6075, .745)]), BLACK)


def p_roadworks(cv):
    cv.paint(cv.disc(.45, .375, .04), BLACK)                                      # head
    cv.paint(cv.stroke([(.455, .445), (.50, .56)], .09), BLACK)                   # torso, leaning forward
    cv.paint(cv.stroke([(.495, .575), (.43, .655), (.405, .74)], .055), BLACK)    # rear leg
    cv.paint(cv.stroke([(.505, .575), (.55, .655), (.54, .74)], .055), BLACK)     # front leg
    cv.paint(cv.stroke([(.47, .46), (.565, .53)], .042), BLACK)                   # arms
    cv.paint(cv.stroke([(.50, .455), (.64, .67)], .03), BLACK)                    # shovel
    cv.paint(cv.disc(.675, .748, .12, .10) & (cv.Y <= .748), BLACK)               # heap


def p_lights(cv):
    for y, col in ((.40, TL_RED), (.535, TL_YEL), (.67, TL_GRN)):
        cv.paint(cv.disc(.5, y, .052), col)


def person(cv, hx, hy, s, lean=0.0, stride=1.0):
    """Walking figure: head at (hx, hy), overall scale s (about the total height)."""
    cv.paint(cv.disc(hx, hy, .095 * s), BLACK)
    sh = (hx + lean * .05 * s, hy + .17 * s); hip = (hx + lean * .12 * s - .0 * s, hy + .52 * s)
    cv.paint(cv.stroke([sh, hip], .19 * s), BLACK)
    cv.paint(cv.stroke([hip, (hip[0] - .20 * s * stride, hy + .98 * s)], .11 * s), BLACK)
    cv.paint(cv.stroke([hip, (hip[0] + .22 * s * stride, hy + .98 * s)], .11 * s), BLACK)
    cv.paint(cv.stroke([sh, (sh[0] - .22 * s * stride, sh[1] + .30 * s)], .085 * s), BLACK)
    cv.paint(cv.stroke([sh, (sh[0] + .22 * s * stride, sh[1] + .28 * s)], .085 * s), BLACK)


def p_pedestrian(cv):
    person(cv, .5, .365, .385, lean=0.15)


def p_children(cv):
    person(cv, .575, .40, .34, lean=-0.5, stride=1.0)
    person(cv, .415, .465, .275, lean=-0.5, stride=1.0)
    cv.paint(cv.stroke([(.52, .50), (.45, .545)], .03), BLACK)


def p_bicycle(cv):
    for cx in (.355, .645):
        R = np.hypot(cv.X - cx, cv.Y - .63)
        cv.paint((R <= .108) & (R >= .07), BLACK)
    cv.paint(cv.stroke([(.355, .63), (.455, .485), (.60, .485), (.645, .63)], .034), BLACK)
    cv.paint(cv.stroke([(.455, .485), (.505, .63), (.60, .485)], .034), BLACK)
    cv.paint(cv.stroke([(.505, .63), (.355, .63)], .026), BLACK)
    cv.paint(cv.stroke([(.435, .45), (.485, .45)], .036), BLACK)                  # saddle
    cv.paint(cv.stroke([(.60, .485), (.585, .43), (.63, .42)], .03), BLACK)       # handlebar


def p_snow(cv):
    c = np.array([.5, .575]); L = .15
    for k in range(3):
        a = np.radians(90 + 60 * k); d = np.array([np.cos(a), np.sin(a)])
        cv.paint(cv.stroke([tuple(c - d * L), tuple(c + d * L)], .036), BLACK)
        for sgn in (-1, 1):
            p = c + sgn * d * L * .62
            for da in (-55, 55):
                b = np.radians(np.degrees(a) + da); e = np.array([np.cos(b), np.sin(b)]) * sgn
                cv.paint(cv.stroke([tuple(p), tuple(p + e * .055)], .03), BLACK)


def p_deer(cv):
    body = [(.41, .475), (.50, .47), (.615, .53), (.665, .60), (.60, .625), (.50, .585), (.42, .555)]
    cv.paint(cv.poly(body), BLACK)
    cv.paint(cv.stroke([(.43, .50), (.385, .435), (.345, .445)], .05), BLACK)            # neck and head (facing left)
    cv.paint(cv.stroke([(.385, .43), (.40, .355)], .024), BLACK)                         # antlers
    cv.paint(cv.stroke([(.395, .39), (.435, .365)], .022), BLACK)
    cv.paint(cv.stroke([(.445, .55), (.385, .60), (.335, .585)], .03), BLACK)            # forelegs, tucked forward
    cv.paint(cv.stroke([(.47, .565), (.42, .64)], .028), BLACK)
    cv.paint(cv.stroke([(.63, .595), (.69, .645), (.735, .70)], .034), BLACK)            # hind legs stretched back
    cv.paint(cv.stroke([(.60, .61), (.655, .675), (.69, .725)], .028), BLACK)


def danger(p):
    cv = base_tri(True); p(cv); return cv


def s_noentry():
    cv = Cv(1.0); R = np.hypot(cv.X - .5, cv.Y - .5)
    cv.shape(R <= .5); cv.paint(R <= .5, WHITE); cv.paint(R <= .483, RED)
    cv.paint(cv.poly([(.12, .422), (.88, .422), (.88, .578), (.12, .578)]), WHITE)
    cv.inner = R <= .34
    return cv


def s_stop():
    cv = Cv(1.0); k = 0.5 * np.tan(np.pi / 8)
    v = [(.5 - k, 0), (.5 + k, 0), (1, .5 - k), (1, .5 + k), (.5 + k, 1), (.5 - k, 1), (0, .5 + k), (0, .5 - k)]
    sd = cv.polysd(v)
    cv.shape(sd <= 0); cv.paint(sd <= 0, WHITE); cv.paint(sd <= -.028, RED)
    text(cv, 'STOP', .5, .5, .33, [.15, .15, .15, .15], .048, .03, WHITE)
    cv.inner = sd <= -.15
    return cv


def s_priority():
    cv = Cv(1.0); rho = .045; h0 = .5 - rho
    sd = cv.polysd([(.5, .5 - h0), (.5 + h0, .5), (.5, .5 + h0), (.5 - h0, .5)]) - rho
    cv.shape(sd <= 0); cv.paint(sd <= 0, WHITE)
    h1 = .30
    sd2 = cv.polysd([(.5, .5 - h1), (.5 + h1, .5), (.5, .5 + h1), (.5 - h1, .5)]) - .015
    cv.paint(sd2 <= 0, YELLOW)
    cv.inner = sd <= -.11
    return cv


def s_nopass(truck, grey=False):
    cv = base_end() if grey else base_ring()
    c1, c2 = (GREY, GREY) if grey else (RED, BLACK)
    if truck:
        truck_rear(cv, .355, .635, c1); car_rear(cv, .665, .635, c2, w=.25, hh=.19)
    else:
        car_rear(cv, .345, .60, c1); car_rear(cv, .655, .60, c2)
    if grey: end_stripes(cv)
    return cv


def s_end80():
    cv = base_end(); text(cv, '80', .5, .5, .42, [.24, .24], .062, .05, GREY); end_stripes(cv); return cv


def s_endall():
    cv = base_end(); end_stripes(cv); return cv


def s_notrucks():
    cv = base_ring(); truck_side(cv, BLACK); return cv


NAMES = ['20', '30', '50', '60', '70', '80', 'end80', '100', '120', 'nopass', 'nopass_truck', 'crossroad', 'priority', 'yield', 'stop',
         'novehicles', 'notrucks', 'noentry', 'general', 'bend_l', 'bend_r', 'double', 'bumpy', 'slippery', 'narrows', 'roadworks',
         'lights', 'pedestrian', 'children', 'bicycle', 'snow', 'deer', 'endall', 'right', 'left', 'ahead', 'ahead_right', 'ahead_left',
         'keep_r', 'keep_l', 'roundabout', 'end_nopass', 'end_nopass_truck']


def all_signs():
    S = [None] * 43
    for c, ch in zip((0, 1, 2, 3, 4, 5, 7, 8), ('20', '30', '50', '60', '70', '80', '100', '120')): S[c] = speed(ch)
    S[6] = s_end80(); S[9] = s_nopass(False); S[10] = s_nopass(True)
    S[11] = danger(p_crossroad); S[12] = s_priority(); S[13] = base_tri(False, band=0.09, rim=0.012); S[14] = s_stop()
    S[15] = base_ring(); S[16] = s_notrucks(); S[17] = s_noentry(); S[18] = danger(p_general)
    S[19] = danger(lambda cv: p_bend(cv, False)); S[20] = danger(lambda cv: p_bend(cv, True)); S[21] = danger(p_double)
    S[22] = danger(p_bumpy); S[23] = danger(p_slippery); S[24] = danger(p_narrows); S[25] = danger(p_roadworks)
    S[26] = danger(p_lights); S[27] = danger(p_pedestrian); S[28] = danger(p_children); S[29] = danger(p_bicycle)
    S[30] = danger(p_snow); S[31] = danger(p_deer); S[32] = s_endall()
    S[33] = s_turn(False); S[34] = s_turn(True); S[35] = s_ahead(); S[36] = s_straight_or(False); S[37] = s_straight_or(True)
    S[38] = s_keep(False); S[39] = s_keep(True); S[40] = s_roundabout(); S[41] = s_nopass(False, True); S[42] = s_nopass(True, True)
    return np.stack([s.result() for s in S])            # (43, 5, NC, NC)


BLANK = (13, 15)       # signs with nothing drawn inside the border


def draw_all():
    global BORDER
    BORDER = True; full = all_signs()
    BORDER = False; cont = all_signs()
    BORDER = True
    return full, cont


# ====================================================================================================
# pipe.py
# ====================================================================================================
# ----------------------------------------------------------------------------------------------
# Hand-chosen settings
# ----------------------------------------------------------------------------------------------
EPS = 0.025            # log(x + EPS): a few grey levels, so sensor noise in black areas is not amplified
NKNOT = 16             # relu pieces used for the log curve
SIG_A = 0.7            # blur of the linear colour channels
SIG_D = 0.7            # scale of the edge (Gaussian derivative) filters
ALPHA = 1.0            # orientation sharpening (subtract the response of the two orthogonal directions)
TAU = 1.5              # edge saturation level, in units of the normalised luminance step
SIG_P = 0.8            # spatial pooling of the edge maps
KER = 28               # template window; the 32x32 image leaves 5x5 placements
BG = 0.10              # dark neutral background behind a rendered sign (gives every sign a clear outline edge)
CHROMA = 2.0           # gain of the two colour-opponent channels relative to luminance
GAMMA = 2.0            # weight of the interior ("content only") template when ranking classes inside a family
DELTA = 0.2            # bonus that lets an empty sign win inside its family when no content template responds
SIZES = [19.0, 20.5, 22.0, 23.5, 25.0, 26.5, 28.0]
ASPECTS = [(0.0, 0.0), (-0.75, 0.75), (0.75, -0.75)]
ANGLES = [0.0, -7.0, 7.0]
SIG_R = 0.6            # optical blur of a rendered sign (sharp set)
SIG_R2 = 1.3           # blurred set (motion / defocus), upright only
VARIANTS = [(d + a, d + b, g, SIG_R) for d in SIZES for a, b in ASPECTS for g in ANGLES] + \
           [(d + a, d + b, 0.0, SIG_R2) for d in SIZES for a, b in ASPECTS]
# families of signs that share an outline: red ring, red triangle, blue disc, white "end of" disc, and four singletons
FAMILIES = [[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16], [11] + list(range(18, 32)), list(range(33, 41)), [6, 32, 41, 42], [12], [13], [14], [17]]


def gauss1(sig, r):
    x = np.arange(-r, r + 1.0); g = np.exp(-x * x / (2 * sig * sig)); return g / g.sum()


def gauss2(sig, r):
    g = gauss1(sig, r); return np.outer(g, g)


# ----------------------------------------------------------------------------------------------
# numpy interpreter for the layer list (same definitions as the task statement)
# ----------------------------------------------------------------------------------------------
def conv(x, W, b, stride=1, pad=0):
    n, c, h, w = x.shape; o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    out = np.zeros((n, o, ho, wo))
    for dy in range(kh):
        for dx in range(kw):
            out += np.einsum('nchw,oc->nohw', xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride], W[:, :, dy, dx], optimize=True)
    return out + b[None, :, None, None]


def run_layers(layers, weights, x, upto=None):
    vals = {'image': x}
    for L in layers:
        a = vals[L['inputs'][0]]; op = L['op']; nm = L['name']
        if op == 'conv': vals[nm] = conv(a, weights[nm + '.W'], weights[nm + '.b'], L.get('stride', 1), L.get('padding', 0))
        elif op == 'relu': vals[nm] = np.maximum(a, 0)
        elif op == 'reshape': vals[nm] = a.reshape((len(a),) + tuple(L['shape']))
        elif op == 'norm':
            mu = a.mean((2, 3), keepdims=True); vals[nm] = (a - mu) / np.sqrt(((a - mu) ** 2).mean((2, 3), keepdims=True) + 1e-5)
        elif op == 'add': vals[nm] = sum(vals[i] for i in L['inputs'])
        else: raise ValueError(op)
        if nm == upto: break
    return vals


# ----------------------------------------------------------------------------------------------
# Front end: log colour -> per-image normalisation -> (A) opponent colour maps, (B) saturated oriented edge maps
# ----------------------------------------------------------------------------------------------
def front_end():
    layers, W = [], {}
    # 1. log(x + EPS) per colour channel as a sum of relus (piecewise linear through geometric knots)
    t = EPS * ((1 + EPS) / EPS) ** (np.arange(NKNOT + 1) / NKNOT) - EPS; t[0] = 0.0; t[-1] = 1.0
    f = np.log(t + EPS); m = np.diff(f) / np.diff(t); dm = np.diff(np.concatenate([[0.0], m]))
    w1 = np.zeros((3 * NKNOT, 3, 1, 1)); b1 = np.zeros(3 * NKNOT); w2 = np.zeros((3, 3 * NKNOT, 1, 1))
    for c in range(3):
        for k in range(NKNOT):
            w1[c * NKNOT + k, c, 0, 0] = 1.0; b1[c * NKNOT + k] = -t[k]; w2[c, c * NKNOT + k, 0, 0] = dm[k]
    layers += [dict(name='k1', op='conv', inputs=['image'], out=3 * NKNOT, kernel=[1, 1], stride=1, padding=0),
               dict(name='k1r', op='relu', inputs=['k1']),
               dict(name='lg', op='conv', inputs=['k1r'], out=3, kernel=[1, 1], stride=1, padding=0),
               dict(name='lg_r', op='reshape', inputs=['lg'], shape=[1, 96, 32]),
               dict(name='lg_n', op='norm', inputs=['lg_r']),
               dict(name='z', op='reshape', inputs=['lg_n'], shape=[3, 32, 32])]
    W['k1.W'], W['k1.b'], W['lg.W'], W['lg.b'] = w1, b1, w2, np.zeros(3)
    # 2A. opponent colour (luminance, red-green, blue-yellow), lightly blurred; colour channels get the gain CHROMA
    opp = np.array([[1, 1, 1], [1, -1, 0], [-1, -1, 2]], float); opp /= np.linalg.norm(opp, axis=1, keepdims=True)
    opp *= np.array([1.0, CHROMA, CHROMA])[:, None] / np.sqrt(3.0)
    ga = gauss2(SIG_A, 2)
    W['fa.W'] = opp[:, :, None, None] * ga[None, None]; W['fa.b'] = np.zeros(3)
    layers += [dict(name='fa', op='conv', inputs=['z'], out=3, kernel=[5, 5], stride=1, padding=2)]
    # 2B. eight directional derivatives of the luminance; a unit step edge gives a peak response of 1
    r = 2; x = np.arange(-r, r + 1.0); g = np.exp(-x * x / (2 * SIG_D ** 2)); g /= g.sum()
    dg = -x * g; step = (np.arange(-r, r + 1.0) >= 0).astype(float)
    dg = dg / abs(np.sum(dg * step))                       # response of the 1-D derivative filter to a unit step at its centre
    gx = np.outer(g, dg); gy = np.outer(dg, g)             # gx: derivative along x (columns), gy: along y (rows)
    wg = np.zeros((8, 3, 5, 5))
    for k in range(8):
        a = np.pi * k / 4
        wg[k, :] = (np.cos(a) * gx + np.sin(a) * gy)[None] / np.sqrt(3.0)
    W['g.W'], W['g.b'] = wg, np.zeros(8)
    layers += [dict(name='g', op='conv', inputs=['z'], out=8, kernel=[5, 5], stride=1, padding=2), dict(name='gr', op='relu', inputs=['g'])]
    # orientation sharpening and saturation: e_k = clip(r_k - ALPHA (r_{k+2} + r_{k-2}), 0, TAU) = relu(s) - relu(s - TAU)
    wo = np.zeros((16, 8, 1, 1)); bo = np.zeros(16)
    for k in range(8):
        for j in (k, k + 8):
            wo[j, k, 0, 0] = 1.0; wo[j, (k + 2) % 8, 0, 0] -= ALPHA; wo[j, (k - 2) % 8, 0, 0] -= ALPHA
        bo[k + 8] = -TAU
    W['o.W'], W['o.b'] = wo, bo
    gp = gauss2(SIG_P, 3); wp = np.zeros((8, 16, 7, 7))
    for k in range(8): wp[k, k] = gp; wp[k, k + 8] = -gp
    W['p.W'], W['p.b'] = wp, np.zeros(8)
    layers += [dict(name='o', op='conv', inputs=['gr'], out=16, kernel=[1, 1], stride=1, padding=0), dict(name='or', op='relu', inputs=['o']),
               dict(name='p', op='conv', inputs=['or'], out=8, kernel=[7, 7], stride=1, padding=3),
               dict(name='p_r', op='reshape', inputs=['p'], shape=[1, 256, 32]), dict(name='p_n', op='norm', inputs=['p_r']),
               dict(name='fb', op='reshape', inputs=['p_n'], shape=[8, 32, 32])]
    # 2U. unsigned edge maps (opposite directions added): the outline of the sign whatever the background brightness
    wu = np.zeros((4, 8, 1, 1))
    for k in range(4): wu[k, k, 0, 0] = 1.0; wu[k, k + 4, 0, 0] = 1.0
    W['u.W'], W['u.b'] = wu, np.zeros(4)
    layers += [dict(name='u', op='conv', inputs=['p'], out=4, kernel=[1, 1], stride=1, padding=0),
               dict(name='u_r', op='reshape', inputs=['u'], shape=[1, 128, 32]), dict(name='u_n', op='norm', inputs=['u_r']),
               dict(name='fu', op='reshape', inputs=['u_n'], shape=[4, 32, 32])]
    return layers, W


# ----------------------------------------------------------------------------------------------
# Rendering a canonical sign into the 32x32 frame
# ----------------------------------------------------------------------------------------------
def blur_maps(x, sig, r):
    g = gauss1(sig, r); xp = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(r, r), (r, r)])
    n = x.shape[-1]
    y = sum(g[i] * xp[..., i:i + n, :] for i in range(2 * r + 1))
    return sum(g[i] * y[..., :, i:i + n] for i in range(2 * r + 1))


def render(C, w, h, ang, S=6, dx=0.0, dy=0.0):
    """C: (N, 4, NC, NC) premultiplied RGB + alpha of signs filling their bounding box.  Returns (N, 4, 32, 32)."""
    NC = C.shape[-1]; n = 32
    sub = (np.arange(n * S) + 0.5) / S; PX, PY = np.meshgrid(sub, sub)
    a = np.radians(ang); ca, sa = np.cos(a), np.sin(a)
    x = PX - 16.0 - dx; y = PY - 16.0 - dy
    qx = ca * x + sa * y; qy = -sa * x + ca * y
    u = (qx / w + 0.5) * NC - 0.5; v = (qy / h + 0.5) * NC - 0.5
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int); fu = u - u0; fv = v - v0

    def g(iy, ix):
        ok = (iy >= 0) & (iy < NC) & (ix >= 0) & (ix < NC)
        return C[:, :, np.clip(iy, 0, NC - 1), np.clip(ix, 0, NC - 1)] * ok
    val = g(v0, u0) * ((1 - fu) * (1 - fv)) + g(v0, u0 + 1) * (fu * (1 - fv)) + g(v0 + 1, u0) * ((1 - fu) * fv) + g(v0 + 1, u0 + 1) * (fu * fv)
    return val.reshape(len(C), C.shape[1], n, S, n, S).mean((3, 5))


def composite(R, sig):
    """(N, 4, 32, 32) premultiplied RGBA -> image on the neutral background, after the optical blur."""
    Rb = blur_maps(R, sig, 4)
    return Rb[:, :3] + BG * (1 - Rb[:, 3:4])


def unit(T):
    """Normalise each template by itself alone (an empty template stays zero)."""
    nrm = np.sqrt((T ** 2).sum((1, 2, 3), keepdims=True))
    return T / np.where(nrm > 1e-9, nrm, 1.0)


def masked(F, M):
    """Zero-mean each channel inside the mask, zero outside."""
    M = M[:, None].astype(float)
    mu = (F * M).sum((2, 3), keepdims=True) / np.maximum(M.sum((2, 3), keepdims=True), 1.0)
    return (F - mu) * M


def build(full, cont, blank=()):
    """full: (N, 5, NC, NC) drawn signs (premultiplied RGB, alpha, interior region); cont: the same signs drawn without their coloured border."""
    layers, W = front_end()
    lo = (32 - KER) // 2; N = len(full); V = len(VARIANTS); K = N * V; npos = (32 - KER + 1) ** 2
    isb = np.zeros((N, 1, 1, 1)); isb[list(blank)] = 1.0
    crop = lambda T: T[:, :, lo:lo + KER, lo:lo + KER]
    FA, FB, FU, EA, EB = [], [], [], [], []
    for (w, h, ang, sig) in VARIANTS:
        Rf = render(full, w, h, ang); Rc = render(cont[:, :4], w, h, ang)
        vf = run_layers(layers, W, composite(Rf[:, :4], sig)); vc = run_layers(layers, W, composite(Rc, sig))
        sigB = np.sqrt(sig ** 2 + 0.8 ** 2)
        a1 = blur_maps(Rf[:, 3], sig, 4); a2 = blur_maps(Rf[:, 3], sigB, 4)
        mA = a1 >= 0.95                      # colour maps: pixels the blurred sign covers
        mB = a2 >= 0.98                      # signed edge maps: about two pixels inside, where the background cannot be seen
        mU = (a2 >= 0.05) & (a2 <= 0.95)     # unsigned edge maps: the band around the outline
        iA = blur_maps(Rf[:, 4], sig, 4) >= 0.90; iB = blur_maps(Rf[:, 4], sigB, 4) >= 0.93     # interior region
        # whole-sign template: each feature group normalised by itself
        fA = np.concatenate([unit(masked(vf['fa'][:, :1], mA)), unit(masked(vf['fa'][:, 1:], mA))], 1)
        fB = unit(masked(vf['fb'], mB)) / np.sqrt(8.0); fU = unit(masked(vf['fu'], mU)) / 2.0
        # interior template from the border-free drawing (empty signs: a share DELTA of their whole-sign template instead)
        cA = np.concatenate([unit(masked(vc['fa'][:, :1], iA)), unit(masked(vc['fa'][:, 1:], iA))], 1)
        cB = unit(masked(vc['fb'], iB)) / np.sqrt(8.0)
        eA = GAMMA * (1 - isb) * cA + DELTA * isb * fA; eB = GAMMA * (1 - isb) * cB + DELTA * isb * fB
        FA.append(crop(fA)); FB.append(crop(fB)); FU.append(crop(fU)); EA.append(crop(eA)); EB.append(crop(eB))
    for nm, T, src in (('tfa', FA, 'fa'), ('tfb', FB, 'fb'), ('tfu', FU, 'fu'), ('tea', EA, 'fa'), ('teb', EB, 'fb')):
        T = np.stack(T, 1); T = T.reshape((K,) + T.shape[2:])                      # class-major order
        W[nm + '.W'], W[nm + '.b'] = T, np.zeros(K)
        layers.append(dict(name=nm, op='conv', inputs=[src], out=K, kernel=[KER, KER], stride=1, padding=0))
    # whole-sign score and family-ranking score for every template placement, then the best placement / variant per class
    layers += [dict(name='full', op='add', inputs=['tfa', 'tfb', 'tfu']), dict(name='rank', op='add', inputs=['full', 'tea', 'teb'])]
    for nm in ('full', 'rank'):
        layers += [dict(name=nm + '_r', op='reshape', inputs=[nm], shape=[1, K, npos]),
                   dict(name=nm + '_m', op='maxpool', inputs=[nm + '_r'], size=[V, npos], stride=[V, npos]),
                   dict(name=nm + '_v', op='reshape', inputs=[nm + '_m'], shape=[N])]
    # family logic: score(c) = best whole-sign score in c's family + rank(c) - best rank in c's family
    F = len(FAMILIES); S = max(len(f) for f in FAMILIES)
    sel = np.zeros((F * S, N)); back = np.zeros((N, F))
    for i, fam in enumerate(FAMILIES):
        for j in range(S): sel[i * S + j, fam[j] if j < len(fam) else fam[0]] = 1.0
        for c in fam: back[c, i] = 1.0
    for nm, sgn in (('full', 1.0), ('rank', -1.0)):
        W[nm + '_s.W'], W[nm + '_s.b'] = sel, np.zeros(F * S); W[nm + '_b.W'], W[nm + '_b.b'] = sgn * back, np.zeros(N)
        layers += [dict(name=nm + '_s', op='dense', inputs=[nm + '_v'], out=F * S),
                   dict(name=nm + '_sr', op='reshape', inputs=[nm + '_s'], shape=[1, F, S]),
                   dict(name=nm + '_sm', op='maxpool', inputs=[nm + '_sr'], size=[1, S], stride=[1, S]),
                   dict(name=nm + '_b', op='dense', inputs=[nm + '_sm'], out=N)]
    layers.append(dict(name='out', op='add', inputs=['full_b', 'rank_v', 'rank_b']))
    return layers, W


def save(out_dir, layers, W):
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'network.json'), 'w') as f: json.dump({'layers': layers}, f)
    np.savez(os.path.join(out_dir, 'weights.npz'), **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    full, cont = draw_all()
    layers, W = build(full, cont, BLANK)
    save(out_dir, layers, W)
