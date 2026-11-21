"""Data-free builder for a hand-designed German-traffic-sign classifier.

Run:  python3 build.py OUT_DIR   ->  writes OUT_DIR/network.json and OUT_DIR/weights.npz

No training, no data: every weight is derived from clean sign templates that this
file RENDERS from general knowledge of German road signs (shapes, colours, digit
strokes, arrows, symbols).  The network is a matched filter:
  pre (1x1 conv): opponent channels [luminance, red-green, yellow-blue]
  norm          : per-channel zero-mean unit-variance  (removes DC -> no template
                  can become a generic 'sink')
  corr (conv)   : one filter per (class, scale); filter = W_S * Shat[family] + Dhat[class]
                    Shat = shared family colour/shape template (masked to the sign
                           footprint, L2-normalised) -> identical within a sign family,
                           so it cancels in the within-family comparison;
                    Dhat = class-specific mark template (digits / arrows / symbol,
                           masked to the interior, per-channel-centred, L2-normalised)
                           -> decides which class within a family.
  maxpool over 7x7 shifts, then over the scale variants -> 43 class scores.
Each template is normalised only by itself; no statistic is taken across templates
or across any data.
"""
import numpy as np, json, sys, os

SS = 5
N = 32
G = N * SS
_ys = (np.arange(G) + 0.5) / SS
_xs = (np.arange(G) + 0.5) / SS
Y, X = np.meshgrid(_ys, _xs, indexing='ij')

RED    = (0.72, 0.10, 0.12)
BLUE   = (0.08, 0.19, 0.52)
YELLOW = (0.98, 0.78, 0.06)
WHITE  = (0.97, 0.97, 0.95)
BLACK  = (0.10, 0.10, 0.10)
GRAY   = (0.50, 0.50, 0.50)
DRED   = (0.40, 0.06, 0.08)
GREEN  = (0.10, 0.45, 0.18)
AMBER  = (0.95, 0.70, 0.08)
LIGHT  = (0.85, 0.85, 0.85)
CX = CY = 16.0
R_OUT, R_IN = 15.0, 12.6


class Canvas:
    def __init__(self, bg=GRAY):
        self.img = np.ones((G, G, 3)) * np.array(bg, float)

    def _put(self, mask, color):
        self.img[mask] = np.array(color, float)

    def disc(self, cx, cy, r, color):
        self._put((X - cx) ** 2 + (Y - cy) ** 2 <= r * r, color)

    def ring(self, cx, cy, r_out, r_in, color):
        d = (X - cx) ** 2 + (Y - cy) ** 2
        self._put((d <= r_out * r_out) & (d >= r_in * r_in), color)

    def poly(self, pts, color):
        pts = [(float(a), float(b)) for a, b in pts]
        n = len(pts)
        area = 0.0
        for i in range(n):
            x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
            area += x0 * y1 - x1 * y0
        s = 1.0 if area > 0 else -1.0
        inside = np.ones((G, G), bool)
        for i in range(n):
            x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
            cross = (x1 - x0) * (Y - y0) - (y1 - y0) * (X - x0)
            inside &= (s * cross >= 0)
        self._put(inside, color)

    def seg(self, p0, p1, w, color):
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy + 1e-9
        t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1)
        px = x0 + t * dx; py = y0 + t * dy
        self._put((X - px) ** 2 + (Y - py) ** 2 <= (w / 2) ** 2, color)

    def polyline(self, pts, w, color):
        for i in range(len(pts) - 1):
            self.seg(pts[i], pts[i + 1], w, color)

    def rect(self, x0, y0, x1, y1, color):
        self._put((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1), color)

    def down(self):
        return self.img.reshape(N, SS, N, SS, 3).mean((1, 3))


_YY, _XX = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5, indexing='ij')

def disc_mask(r, cx=CX, cy=CY):
    return (_XX - cx) ** 2 + (_YY - cy) ** 2 <= r * r

def tri_up_pts(scale=1.0):
    return [(CX, CY - 13.2 * scale), (CX - 14.0 * scale, CY + 11.0 * scale), (CX + 14.0 * scale, CY + 11.0 * scale)]

def tri_down_pts(scale=1.0):
    return [(CX - 14.0 * scale, CY - 11.0 * scale), (CX + 14.0 * scale, CY - 11.0 * scale), (CX, CY + 13.2 * scale)]

def diamond_pts(scale=1.0):
    r = 14.2 * scale
    return [(CX, CY - r), (CX + r, CY), (CX, CY + r), (CX - r, CY)]

def octagon_pts(r=14.6):
    a = r * 0.4142
    return [(CX - a, CY - r), (CX + a, CY - r), (CX + r, CY - a), (CX + r, CY + a),
            (CX + a, CY + r), (CX - a, CY + r), (CX - r, CY + a), (CX - r, CY - a)]

def poly_mask(pts):
    pts = [(float(a), float(b)) for a, b in pts]
    n = len(pts)
    area = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))
    s = 1.0 if area > 0 else -1.0
    inside = np.ones((N, N), bool)
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        cross = (x1 - x0) * (_YY - y0) - (y1 - y0) * (_XX - x0)
        inside &= (s * cross >= 0)
    return inside

def shrink(pts, d):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    out = []
    for x, y in pts:
        vx, vy = x - cx, y - cy
        L = (vx * vx + vy * vy) ** 0.5 + 1e-9
        out.append((x - vx / L * d, y - vy / L * d))
    return out


_SEG = {'a': ((0, 0), (1, 0)), 'b': ((1, 0), (1, 0.5)), 'c': ((1, 0.5), (1, 1)),
        'd': ((0, 1), (1, 1)), 'e': ((0, 0.5), (0, 1)), 'f': ((0, 0), (0, 0.5)),
        'g': ((0, 0.5), (1, 0.5))}
_DIG = {'0': 'abcdef', '1': 'bc', '2': 'abged', '3': 'abgcd', '4': 'fbgc',
        '5': 'afgcd', '6': 'afgedc', '7': 'abc', '8': 'abcdefg', '9': 'abcdfg'}

def draw_digit(cv, ch, x0, y0, w, h, sw, color=BLACK):
    for s in _DIG[ch]:
        (ax, ay), (bx, by) = _SEG[s]
        cv.seg((x0 + ax * w, y0 + ay * h), (x0 + bx * w, y0 + by * h), sw, color)

def draw_number(cv, text, cx, cy, h, w, sw, color=BLACK, gap=None):
    if gap is None:
        gap = w * 0.35
    total = len(text) * w + (len(text) - 1) * gap
    x = cx - total / 2; y0 = cy - h / 2
    for ch in text:
        draw_digit(cv, ch, x, y0, w, h, sw, color); x += w + gap


SUPER_OF = {}
for c in [0,1,2,3,4,5,7,8,9,10,15,16]: SUPER_OF[c] = 0
for c in [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]: SUPER_OF[c] = 1
SUPER_OF[13] = 2; SUPER_OF[12] = 3; SUPER_OF[14] = 4; SUPER_OF[17] = 5
for c in [33,34,35,36,37,38,39,40]: SUPER_OF[c] = 6
for c in [6,32,41,42]: SUPER_OF[c] = 7

def super_blank(s):
    cv = Canvas(bg=GRAY)
    if s == 0:
        cv.disc(CX, CY, R_OUT, RED); cv.disc(CX, CY, R_IN, WHITE)
    elif s == 1:
        p = tri_up_pts(); cv.poly(p, RED); cv.poly(shrink(p, 2.6), WHITE)
    elif s == 2:
        p = tri_down_pts(); cv.poly(p, RED); cv.poly(shrink(p, 2.6), WHITE)
    elif s == 3:
        p = diamond_pts(); cv.poly(p, WHITE); cv.poly(shrink(p, 2.0), BLACK); cv.poly(shrink(p, 2.8), YELLOW)
    elif s == 4:
        cv.poly(octagon_pts(), RED)
    elif s == 5:
        cv.disc(CX, CY, R_OUT, RED)
    elif s == 6:
        cv.disc(CX, CY, R_OUT, BLUE)
    elif s == 7:
        cv.disc(CX, CY, R_OUT, GRAY); cv.disc(CX, CY, R_IN, LIGHT)
    return cv

def _box(x0, x1, y0, y1, g):
    x0 = CX + (x0 - CX) * g; x1 = CX + (x1 - CX) * g
    y0 = CY + (y0 - CY) * g; y1 = CY + (y1 - CY) * g
    return (_XX > x0) & (_XX < x1) & (_YY > y0) & (_YY < y1)

def super_footprint(s, g=1.0):
    if s in (0, 5, 6, 7): return disc_mask(R_OUT * g)
    if s == 1: return poly_mask(tri_up_pts(g))
    if s == 2: return poly_mask(tri_down_pts(g))
    if s == 3: return poly_mask(diamond_pts(g))
    if s == 4: return poly_mask(octagon_pts(14.6 * g))

def detail_mask(s, g=1.0):
    if s == 0: return disc_mask(11.8 * g)
    if s == 1: return poly_mask(shrink(tri_up_pts(g), 2.8 * g))
    if s == 2: return poly_mask(shrink(tri_down_pts(g), 2.8 * g))
    if s == 3: return poly_mask(shrink(diamond_pts(g), 3.2 * g))
    if s == 4: return _box(4, 28, 11, 21, g)
    if s == 5: return _box(3, 29, 12.5, 19.5, g)
    if s == 6: return disc_mask(12.5 * g)
    if s == 7: return disc_mask(12.5 * g)

def scale_about_center(img, g, bg=GRAY):
    H, Wd, _ = img.shape
    cy = H / 2.0; cx = Wd / 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(Wd), indexing='ij')
    sy = cy + (yy - cy) / g; sx = cx + (xx - cx) / g
    y0 = np.floor(sy).astype(int); x0 = np.floor(sx).astype(int)
    fy = sy - y0; fx = sx - x0
    def samp(iy, ix):
        ok = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < Wd)
        v = np.ones((H, Wd, 3)) * np.array(bg, float)
        g2 = img[np.clip(iy, 0, H - 1), np.clip(ix, 0, Wd - 1)]
        v[ok] = g2[ok]
        return v
    return (samp(y0, x0) * ((1 - fy) * (1 - fx))[..., None]
            + samp(y0, x0 + 1) * ((1 - fy) * fx)[..., None]
            + samp(y0 + 1, x0) * (fy * (1 - fx))[..., None]
            + samp(y0 + 1, x0 + 1) * (fy * fx)[..., None])

def down32(img160):
    return img160.reshape(N, SS, N, SS, 3).mean((1, 3))


def draw_car(cv, cx, cy, L, H, color):
    cv.rect(cx - L / 2, cy - H * 0.12, cx + L / 2, cy + H * 0.30, color)
    cv.poly([(cx - L * 0.28, cy - H * 0.12), (cx + L * 0.02, cy - H * 0.55),
             (cx + L * 0.22, cy - H * 0.55), (cx + L * 0.30, cy - H * 0.12)], color)
    cv.disc(cx - L * 0.30, cy + H * 0.34, H * 0.22, color)
    cv.disc(cx + L * 0.30, cy + H * 0.34, H * 0.22, color)

def draw_truck(cv, cx, cy, L, H, color):
    cv.rect(cx - L / 2, cy - H * 0.30, cx + L * 0.18, cy + H * 0.28, color)
    cv.rect(cx + L * 0.18, cy - H * 0.05, cx + L / 2, cy + H * 0.28, color)
    cv.disc(cx - L * 0.28, cy + H * 0.34, H * 0.22, color)
    cv.disc(cx + L * 0.34, cy + H * 0.34, H * 0.22, color)

def draw_arrow(cv, pts, w, head_len, color, head_from=None):
    cv.polyline(pts, w, color)
    tip = pts[-1]; prev = head_from if head_from is not None else pts[-2]
    dx, dy = tip[0] - prev[0], tip[1] - prev[1]
    L = (dx * dx + dy * dy) ** 0.5 + 1e-9
    ux, uy = dx / L, dy / L; nx, ny = -uy, ux
    base = (tip[0] - ux * head_len, tip[1] - uy * head_len); hw = head_len * 0.72
    cv.poly([(tip[0], tip[1]), (base[0] + nx * hw, base[1] + ny * hw),
             (base[0] - nx * hw, base[1] - ny * hw)], color)

def add_mark(cv, c):
    if c in (0,1,2,3,4,5,7,8):
        txt = {0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[c]
        if len(txt) == 2:
            draw_number(cv, txt, CX, CY, 15.0, 6.8, 1.6, gap=3.2)
        else:
            draw_number(cv, txt, CX, CY, 14.0, 4.6, 1.4, gap=2.2)
    elif c == 9:
        draw_car(cv, CX - 4.5, CY, 9.0, 10.0, RED); draw_car(cv, CX + 4.5, CY, 9.0, 10.0, BLACK)
    elif c == 10:
        draw_car(cv, CX - 4.8, CY, 8.5, 9.5, RED); draw_truck(cv, CX + 4.8, CY, 9.5, 10.0, BLACK)
    elif c == 15:
        pass
    elif c == 16:
        draw_truck(cv, CX, CY, 15.0, 12.0, BLACK)
    elif c == 11:
        cv.seg((CX, CY - 7.5), (CX, CY + 7.0), 3.0, BLACK)
        cv.seg((CX - 5.5, CY - 1.5), (CX + 5.5, CY - 1.5), 2.4, BLACK)
    elif c == 18:
        cv.seg((CX, CY - 7.0), (CX, CY + 3.0), 2.4, BLACK); cv.disc(CX, CY + 6.6, 1.5, BLACK)
    elif c == 19:
        draw_arrow(cv, [(CX+1.5, CY+7), (CX+1.5, CY-1), (CX-3.5, CY-6)], 2.4, 4.5, BLACK)
    elif c == 20:
        draw_arrow(cv, [(CX-1.5, CY+7), (CX-1.5, CY-1), (CX+3.5, CY-6)], 2.4, 4.5, BLACK)
    elif c == 21:
        cv.polyline([(CX+2, CY+7), (CX+2, CY+2), (CX-2, CY-1), (CX-2, CY-4)], 2.3, BLACK)
        draw_arrow(cv, [(CX-2, CY-4), (CX-2, CY-6)], 2.3, 3.5, BLACK)
    elif c == 22:
        cv.seg((CX-6, CY+4),(CX-3,CY-1),2.2,BLACK); cv.seg((CX-3,CY-1),(CX-0.5,CY+4),2.2,BLACK)
        cv.seg((CX+0.5,CY+4),(CX+3,CY-1),2.2,BLACK); cv.seg((CX+3,CY-1),(CX+6,CY+4),2.2,BLACK)
    elif c == 23:
        cv.rect(CX-5, CY-5, CX+5, CY-1, BLACK)
        cv.polyline([(CX-6,CY+5),(CX-3,CY+3),(CX-1,CY+5),(CX+2,CY+3),(CX+5,CY+5)], 1.6, BLACK)
    elif c == 24:
        cv.seg((CX-4, CY-6),(CX-4,CY+6),2.2,BLACK); cv.seg((CX+4, CY-6),(CX+1.5,CY+6),2.2,BLACK)
    elif c == 25:
        cv.disc(CX-1, CY-6, 1.6, BLACK); cv.seg((CX-1,CY-4),(CX+1,CY+2),2.2,BLACK)
        cv.seg((CX-1,CY-2),(CX+4,CY-5),1.8,BLACK); cv.poly([(CX-6,CY+7),(CX+1,CY+7),(CX-2.5,CY+1)], BLACK)
    elif c == 26:
        cv.rect(CX-2.5, CY-7.5, CX+2.5, CY+7.5, BLACK)
        cv.disc(CX, CY-4.2, 1.7, DRED); cv.disc(CX, CY, 1.7, AMBER); cv.disc(CX, CY+4.2, 1.7, GREEN)
    elif c == 27:
        cv.disc(CX, CY-6, 1.7, BLACK); cv.seg((CX,CY-4),(CX-1,CY+2),2.0,BLACK)
        cv.seg((CX-1,CY+2),(CX-3,CY+7),1.8,BLACK); cv.seg((CX-1,CY+2),(CX+2,CY+7),1.8,BLACK)
        cv.seg((CX,CY-2),(CX+3,CY+1),1.6,BLACK)
    elif c == 28:
        for off in (-3.5, 3.5):
            cv.disc(CX+off, CY-6, 1.4, BLACK); cv.seg((CX+off,CY-4.5),(CX+off,CY+2),1.8,BLACK)
            cv.seg((CX+off,CY+2),(CX+off-2,CY+7),1.5,BLACK); cv.seg((CX+off,CY+2),(CX+off+2,CY+7),1.5,BLACK)
    elif c == 29:
        cv.ring(CX-4.5, CY+3, 3.4, 2.3, BLACK); cv.ring(CX+4.5, CY+3, 3.4, 2.3, BLACK)
        cv.polyline([(CX-4.5,CY+3),(CX-1,CY+3),(CX+1,CY-2),(CX+4.5,CY+3)], 1.4, BLACK)
        cv.seg((CX+1,CY-2),(CX+3,CY-2),1.4,BLACK)
    elif c == 30:
        for ang in (0, 60, 120):
            a = np.deg2rad(ang); dx, dy = np.cos(a)*6.5, np.sin(a)*6.5
            cv.seg((CX-dx, CY-dy),(CX+dx, CY+dy),1.5,BLACK)
    elif c == 31:
        cv.poly([(CX-6,CY+4),(CX-2,CY-3),(CX+3,CY-4),(CX+6,CY-6),(CX+5,CY-2),(CX+2,CY+1),
                 (CX+4,CY+6),(CX+1,CY+6),(CX-1,CY+1),(CX-3,CY+6),(CX-6,CY+6)], BLACK)
    elif c == 35:
        draw_arrow(cv, [(CX, CY+8), (CX, CY-6.5)], 3.0, 5.5, WHITE)
    elif c == 33:
        draw_arrow(cv, [(CX-3, CY+7), (CX-3, CY-1), (CX+5, CY-1)], 3.0, 5.0, WHITE)
    elif c == 34:
        draw_arrow(cv, [(CX+3, CY+7), (CX+3, CY-1), (CX-5, CY-1)], 3.0, 5.0, WHITE)
    elif c == 36:
        draw_arrow(cv, [(CX-2.5, CY+7.5), (CX-2.5, CY-6.5)], 2.6, 4.6, WHITE)
        draw_arrow(cv, [(CX-2.5, CY+1), (CX+5.5, CY+1)], 2.6, 4.2, WHITE)
    elif c == 37:
        draw_arrow(cv, [(CX+2.5, CY+7.5), (CX+2.5, CY-6.5)], 2.6, 4.6, WHITE)
        draw_arrow(cv, [(CX+2.5, CY+1), (CX-5.5, CY+1)], 2.6, 4.2, WHITE)
    elif c == 38:
        draw_arrow(cv, [(CX-5, CY-6.5), (CX+3.5, CY+6.5)], 3.0, 5.2, WHITE)
    elif c == 39:
        draw_arrow(cv, [(CX+5, CY-6.5), (CX-3.5, CY+6.5)], 3.0, 5.2, WHITE)
    elif c == 40:
        pts = []
        for ang in range(0, 360, 6):
            a = np.deg2rad(ang); pts.append((CX + 6.5*np.cos(a), CY + 6.5*np.sin(a)))
        for k in range(3):
            seg = pts[k*20: k*20+16]; cv.polyline(seg, 2.2, WHITE); draw_arrow(cv, seg[-2:], 2.2, 3.4, WHITE)
    elif c == 6:
        draw_number(cv, '80', CX, CY, 13.0, 6.5, 2.2, color=GRAY, gap=1.3)
        cv.seg((CX-9, CY+9),(CX+9, CY-9), 1.8, GRAY)
    elif c == 32:
        for off in (-3, 0, 3):
            cv.seg((CX-9+off, CY+9),(CX+9+off, CY-9), 1.3, GRAY)
    elif c == 41:
        draw_car(cv, CX-4.5, CY, 9.0, 10.0, GRAY); draw_car(cv, CX+4.5, CY, 9.0, 10.0, GRAY)
        cv.seg((CX-9, CY+9),(CX+9, CY-9), 1.8, (0.3,0.3,0.3))
    elif c == 42:
        draw_car(cv, CX-4.8, CY, 8.5, 9.5, GRAY); draw_truck(cv, CX+4.8, CY, 9.5, 10.0, GRAY)
        cv.seg((CX-9, CY+9),(CX+9, CY-9), 1.8, (0.3,0.3,0.3))

def _render_class_cv(c):
    cv = super_blank(SUPER_OF[c])
    if SUPER_OF[c] == 5:
        cv.rect(CX - 10.5, CY - 2.6, CX + 10.5, CY + 2.6, WHITE)
    if c == 14:
        cv.rect(CX-9, CY-3.2, CX+9, CY+3.2, RED)
        for bx in (-7.5,-3.0,2.0,6.5):
            cv.rect(CX+bx-1.3, CY-3.0, CX+bx+1.3, CY+3.0, WHITE)
        cv.rect(CX-9, CY+3.2, CX+9, CY+4.0, RED)
    add_mark(cv, c)
    return cv


# ---------------- assemble weights ----------------
SCALES = [0.64, 0.72, 0.80, 0.88, 0.96]
V = len(SCALES)
CW = 0.7
W_S = 2.5
NORM_EPS = 1e-5

def pre_transform(img):
    R, Gc, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return np.stack([(R + Gc + B) / 3.0, R - Gc, 0.5 * (R + Gc) - B], 0)

def norm_chan(x):
    mu = x.mean(axis=(1, 2), keepdims=True)
    y = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(1, 2), keepdims=True) + NORM_EPS)
    y[1] *= CW; y[2] *= CW
    return y

def unit(T):
    nrm = np.sqrt((T ** 2).sum())
    return T / nrm if nrm > 1e-6 else T * 0.0

def shat(img, mask):
    return unit(norm_chan(pre_transform(img)) * mask[None, :, :])

def dhat(img, mask):
    ch = norm_chan(pre_transform(img))
    T = np.zeros((3, N, N))
    for c in range(3):
        mu = ch[c][mask].mean() if mask.any() else 0.0
        T[c][mask] = ch[c][mask] - mu
    return unit(T)

def build_weights():
    super_imgs = {s: super_blank(s).img for s in range(8)}
    class_imgs = {c: _render_class_cv(c).img for c in range(43)}
    Shat = {}
    for s in range(8):
        for gi, g in enumerate(SCALES):
            img32 = down32(scale_about_center(super_imgs[s], g))
            Shat[(s, gi)] = shat(img32, super_footprint(s, g))
    W = np.zeros((43 * V, 3, N, N), np.float32)
    for c in range(43):
        s = SUPER_OF[c]
        for gi, g in enumerate(SCALES):
            img32 = down32(scale_about_center(class_imgs[c], g))
            D = dhat(img32, detail_mask(s, g))
            W[c * V + gi] = (W_S * Shat[(s, gi)] + D).astype(np.float32)
    pre = np.array([[1/3, 1/3, 1/3], [1, -1, 0], [0.5, 0.5, -1]], np.float32).reshape(3, 3, 1, 1)
    return {"pre.W": pre, "pre.b": np.zeros(3, np.float32),
            "corr.W": W, "corr.b": np.zeros(43 * V, np.float32)}

def network():
    return {"layers": [
        {"name": "pre", "op": "conv", "inputs": ["image"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "xn", "op": "norm", "inputs": ["pre"]},
        {"name": "corr", "op": "conv", "inputs": ["xn"], "out": 43 * V, "kernel": [32, 32], "stride": 1, "padding": 3},
        {"name": "shift", "op": "maxpool", "inputs": ["corr"], "size": [7, 7], "stride": [7, 7]},
        {"name": "grp", "op": "reshape", "inputs": ["shift"], "shape": [1, 43, V]},
        {"name": "scalepool", "op": "maxpool", "inputs": ["grp"], "size": [1, V], "stride": [1, V]},
        {"name": "scores", "op": "reshape", "inputs": ["scalepool"], "shape": [43]},
    ]}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    w = build_weights()
    np.savez(os.path.join(out, "weights.npz"), **w)
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(network(), f)
    print("wrote", out)
