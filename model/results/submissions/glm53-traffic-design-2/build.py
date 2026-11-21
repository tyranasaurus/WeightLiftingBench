#!/usr/bin/env python3
"""Hand-crafted traffic-sign classifier (GTSRB-like, 43 classes, 32x32 RGB).

No training of any kind: every weight is constructed in this file from
(a) geometric formulas (annuli, triangles, gratings, angular masks) and
(b) ASCII "shape models" of German traffic signs typed from general
knowledge of what the signs look like (digit strokes, pictograms, arrows).

Pipeline
  1. colour/brightness maps (redness, blueness, yellowness, greenness,
     local blackness, globally brightness-normalised value), each
     per-image normalised (norm op) and clamped to [-4, 4]
  2. banks of 32x32 template correlations (conv, stride 1, padding 2)
     at 5 scales (x 3 aspect variants for sign frames), shift-tolerant
     via 5x5 maxpool, scale-tolerant via maxpool over the scale channels
  3. a hand-written rule layer (dense + relu thresholds) turns the
     template scores into 43 class scores.
"""
import sys, json
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/app/out"

# ---------------------------------------------------------------- graph
LAYERS = []
WEIGHTS = {}
NAMES = set()

def layer(name, op, inputs, **settings):
    assert name and name != "image" and name not in NAMES, name
    NAMES.add(name)
    LAYERS.append({"name": name, "op": op, "inputs": inputs, **settings})
    return name

def conv(name, inp, w, b, stride=1, padding=0):
    w = np.asarray(w, np.float32); b = np.asarray(b, np.float32)
    assert w.ndim == 4
    layer(name, "conv", [inp], out=w.shape[0], kernel=[w.shape[2], w.shape[3]],
          stride=stride, padding=padding)
    WEIGHTS[name + ".W"] = w
    WEIGHTS[name + ".b"] = b
    return name

def dense(name, inp, w, b):
    w = np.asarray(w, np.float32); b = np.asarray(b, np.float32)
    layer(name, "dense", [inp], out=w.shape[0])
    WEIGHTS[name + ".W"] = w
    WEIGHTS[name + ".b"] = b
    return name

def relu(name, inp):
    return layer(name, "relu", [inp])

def c1(name, inp, w, b=0.0):
    """1x1 conv on a (C,H,W) tensor; w may be scalar applied to the single
    channel, or a (out,in) matrix."""
    inp_c = 1 if np.isscalar(w) else None
    if np.isscalar(w):
        Wm = np.array([[[[float(w)]]]], np.float32)
    else:
        Wm = np.asarray(w, np.float32).reshape(w.shape[0], w.shape[1], 1, 1)
    return conv(name, inp, Wm, [b])

def addn(name, *inps):
    return layer(name, "add", list(inps))

def norml(name, inp):
    return layer(name, "norm", [inp])

def reshapel(name, inp, shape):
    return layer(name, "reshape", [inp], shape=list(shape))

def maxpool(name, inp, size, stride):
    return layer(name, "maxpool", [inp], size=list(size), stride=list(stride))

# ---------------------------------------------------------------- maps
# image -> linear colour combinations
# channels: 0 V3=(R+G+B)/3, 1 dRG=R-G, 2 dRB=R-B, 3 dGB=G-B
linW = np.array([[1/3, 1/3, 1/3], [1, -1, 0], [1, 0, -1], [0, 1, -1]], np.float32)
conv("lin1", "image", linW.reshape(4, 3, 1, 1), np.zeros(4, np.float32))

def sel(name, ch):
    Wm = np.zeros((1, 4, 1, 1), np.float32); Wm[0, ch] = 1.0
    return conv(name, "lin1", Wm, [0.0])

v3   = sel("v3", 0)
drg  = sel("drg", 1)
dgb  = sel("dgb", 3)
c1("neg_drb", "lin1", np.array([[0, 0, -1, 0]], np.float32))   # B-R
c1("neg_drg", "lin1", np.array([[0, -1, 0, 0]], np.float32))   # G-R
c1("half_drb", "lin1", np.array([[0, 0, 0.5, 0]], np.float32)) # 0.5*(R-B)

relu("r_gb", dgb)          # relu(G-B)
relu("r_gr", "neg_drg")    # relu(G-R)
relu("r_rg", drg)          # relu(R-G)

# hue maps with small margins (hand-set) to reject weak/cross hues
# redness  = relu( dRG - relu(dGB) - M )          = relu(R - max(G,B) - M)
MARG = 0.03      # red/yellow margin
MARG_B = 0.07    # blue margin (dark noise is bluish)
c1("n_r_gb", "r_gb", -1.0, -MARG)
addn("red_pre", drg, "n_r_gb"); relu("red_raw", "red_pre")
# blueness = relu( (B-R) - relu(G-R) - M )        = relu(B - max(R,G) - M)
c1("n_r_gr", "r_gr", -1.0, -MARG_B)
addn("blue_pre", "neg_drb", "n_r_gr"); relu("blue_raw", "blue_pre")
# greenness= relu( dGB - relu(dRG) - 0.5*relu(dRB) - M )   (rejects yellow)
c1("t5", "r_rg", -1.0)
c1("t6", "lin1", np.array([[0, 0, -0.5, 0]], np.float32))   # -0.5*(R-B)
addn("green_pre", dgb, "t5", "t6")
c1("green_b", "green_pre", 1.0, -MARG); relu("green_raw", "green_b")
# yellowness = relu(0.5 dRB + 0.5 dGB - 1.25 relu(dRG) - 0.5 relu(G-R) - M)
#   (yellow: R,G >> B with R ~ G;  rejects red (R>>G) and green (G>>R))
c1("t2", dgb, 0.5)
c1("t3", "r_rg", -1.25)
c1("t4", "r_gr", -0.5)
addn("yell_pre", "half_drb", "t2", "t3", "t4")
c1("yell_b", "yell_pre", 1.0, -MARG); relu("yell_raw", "yell_b")

# local blackness: relu(mean5(V3) - V3)
meanW = np.full((1, 1, 5, 5), 1/25, np.float32)
conv("mean5", v3, meanW, [0.0], stride=1, padding=2)
c1("neg_v3", v3, -1.0, -0.15)
addn("black_pre", "mean5", "neg_v3"); relu("black_raw", "black_pre")

def make_norm_clamp(base, scale, out):
    c1(out + "_s", base, scale)
    norml(out + "_n", out + "_s")
    m = out + "_n"
    c1(out + "_a", m, 1.0, 4.0)     # m+4
    relu(out + "_ra", out + "_a")
    c1(out + "_b", m, 1.0, -4.0)    # m-4
    relu(out + "_rb", out + "_b")
    c1(out + "_nb", out + "_rb", -1.0)      # -relu(m-4)
    c1(out + "_c4", out + "_ra", 0.0, -4.0) # constant -4
    addn(out, out + "_ra", out + "_nb", out + "_c4")

make_norm_clamp("red_raw", 40.0, "nred")
make_norm_clamp("blue_raw", 40.0, "nblue")
make_norm_clamp("yell_raw", 40.0, "nyell")
make_norm_clamp("green_raw", 40.0, "ngreen")
make_norm_clamp("black_raw", 30.0, "nblack")
norml("nv0", v3)
c1("nv_a", "nv0", 1.0, 4.0);  relu("nv_ra", "nv_a")
c1("nv_b", "nv0", 1.0, -4.0); relu("nv_rb", "nv_b")
c1("nv_nb", "nv_rb", -1.0)
c1("nv_c4", "nv_ra", 0.0, -4.0)
addn("nv", "nv_ra", "nv_nb", "nv_c4")

# ---------------------------------------------------------------- templates
YS, XS = np.mgrid[0:32, 0:32]
DX = XS - 16.0
DY = YS - 16.0
RR = np.sqrt(DX ** 2 + DY ** 2)
THETA = np.arctan2(DY, DX)          # y down; "up" is -pi/2

def _rescale(k, f, a=1.0):
    """bilinear rescale about (16,16) by (f*a, f)"""
    fx, fy = f * a, f
    sx = 16.0 + (XS - 16.0) / fx
    sy = 16.0 + (YS - 16.0) / fy
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    tx = sx - x0; ty = sy - y0
    def gp(x, y):
        x = np.clip(x, 0, 31); y = np.clip(y, 0, 31)
        return k[y, x]
    return (gp(x0, y0) * (1 - tx) * (1 - ty) + gp(x0 + 1, y0) * tx * (1 - ty) +
            gp(x0, y0 + 1) * (1 - tx) * ty + gp(x0 + 1, y0 + 1) * tx * ty)

def mk(pos, neg):
    """kernel = mean(pos values) - mean(neg values): 0 on any uniform map,
    ~+3 for a clean match. pos/neg are float arrays (masks or graded)."""
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    sp = np.abs(pos).sum(); sn = np.abs(neg).sum()
    k = np.zeros((32, 32), np.float64)
    if sp > 1e-9:
        k += pos / sp
    if sn > 1e-9:
        k -= neg / sn
    return k

def finish(base, f, a=1.0):
    return _rescale(base, f, a).astype(np.float32)

def mask_disc(r0, r1, cx=16.0, cy=16.0, ax=1.0):
    r = np.sqrt(((XS - cx) * 1.0 / ax) ** 2 + (YS - cy) ** 2)
    return (r >= r0) & (r <= r1)

def poly_mask(pts):
    inside = np.zeros((32, 32), bool)
    x, y = XS, YS
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(all="ignore"):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= (cond & (x < xint))
    return inside

def tri_pts(R, up=True, s=1.0):
    if up:
        return [(16.0, 16.0 - 1.10 * R * s), (16.0 + 0.95 * R * s, 16.0 + 0.55 * R * s),
                (16.0 - 0.95 * R * s, 16.0 + 0.55 * R * s)]
    return [(16.0 - 0.95 * R * s, 16.0 - 0.55 * R * s), (16.0 + 0.95 * R * s, 16.0 - 0.55 * R * s),
            (16.0, 16.0 + 1.10 * R * s)]

def seg_dist(x0, y0, x1, y1):
    px, py = x1 - x0, y1 - y0
    L2 = px * px + py * py
    t = np.clip(((XS - x0) * px + (YS - y0) * py) / L2, 0, 1)
    return np.sqrt((XS - x0 - t * px) ** 2 + (YS - y0 - t * py) ** 2)

def art_kernel(art, bw, bh, cx=16.0, cy=16.0):
    h = len(art); w = len(art[0])
    k = np.zeros((32, 32), np.float64)
    x0, y0 = cx - bw / 2, cy - bh / 2
    for yy in range(32):
        for xx in range(32):
            u = int((xx + 0.5 - x0) / bw * w)
            v = int((yy + 0.5 - y0) / bh * h)
            if 0 <= u < w and 0 <= v < h:
                c = art[v][u]
                if c == '#': k[yy, xx] += 1.0
                elif c == 'o': k[yy, xx] += 0.55
                elif c == 'x': k[yy, xx] -= 3.0   # intentional strong negative
    return k

def dilate1(k):
    m = (k > 0.35).astype(np.float64)
    p = np.pad(m, 1)
    d = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            d = np.maximum(d, p[1 + dy:33 + dy, 1 + dx:33 + dx])
    return np.where(d > 0.5, np.maximum(k, 0.9), k)

def rect_mask(x0, x1, y0, y1):
    return (XS >= x0) & (XS <= x1) & (YS >= y0) & (YS <= y1)

def surround(k, region):
    """negative mask: region minus the positive strokes"""
    return (region & (k <= 0.001)).astype(np.float64)

def dilate_mask(m, w):
    d = m.astype(bool).copy()
    for _ in range(int(w)):
        p = np.pad(d, 1)
        n = d.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                n |= p[1 + dy:33 + dy, 1 + dx:33 + dx]
        d = n
    return d

def halo(k, region, w=2):
    """negative mask: a band of width w around the strokes, inside region"""
    m = k > 0.35
    return (dilate_mask(m, w) & ~m & region).astype(np.float64)

def art_kernel2(art, bw, bh, cx=16.0, cy=16.0):
    """art with '.' cells kept as separate negative map; returns (pos, dots)"""
    h = len(art); w = len(art[0])
    pos = np.zeros((32, 32), np.float64)
    dots = np.zeros((32, 32), np.float64)
    x0, y0 = cx - bw / 2, cy - bh / 2
    for yy in range(32):
        for xx in range(32):
            u = int((xx + 0.5 - x0) / bw * w)
            v = int((yy + 0.5 - y0) / bh * h)
            if 0 <= u < w and 0 <= v < h:
                c = art[v][u]
                if c == '#': pos[yy, xx] += 1.0
                elif c == 'o': pos[yy, xx] += 0.55
                elif c == 'x': dots[yy, xx] += 1.0
    return pos, dots

R0 = 12.0

# --- digit arts (5 wide, 12 tall) -------------------------------------
DIG = {
 '0': [".###.","##.##","#...#","#...#","#...#","#...#","#...#","#...#","#...#","#...#","##.##",".###."],
 '1': [".#...","##...",".#...",".#...",".#...",".#...",".#...",".#...",".#...",".#...",".#...","###.."],
 '2': [".###.","##.##","...##","..##.",".##..",".##..","##...","##...","##...","#####","#####","....."],
 '3': [".###.","##..#","....#","...##","..###","..###","...##","....#","#...#","#..##","##.##",".###."],
 '5': ["#####","##...","##...","#####","...##","...##","...##","...##","...##","#...#","#..##",".###."],
 '6': [".###.","##...","##...","####.","#..#.","#...#","#...#","#...#","#...#","#..#.",".###.",".###."],
 '7': ["#####","##..#","....#","...##","...##","..##.","..##.",".##..",".##..",".##..","##...","##..."],
 '8': [".###.","##.##","##.##","##.##",".###.",".###.","#...#","#...#","#...#","#...#","##.##",".###."],
}
def num_art(digits):
    rows = []
    for r in range(12):
        row = ""
        for i, d in enumerate(digits):
            row += DIG[d][r]
            if i < len(digits) - 1:
                row += ".."
        rows.append(row)
    return rows

# --- pictogram arts ---------------------------------------------------
P19 = ["..##.......",".####......",".#######...","..####.##..",".......##..",
       ".......##..",".......##..",".......##..",".......##..",".......##..",".......##.."]
P20 = ["......##...","......####.","...#######.","..##.####..","..##.......",
       "..##.......","..##.......","..##.......","..##.......","..##.......","..##......."]
P21 = [".....##....","....####...","...######..","...######..","....##.....",
       "...##......","..##.......","..##.......","..###......","...###.....","...####...."]
P22 = ["............","............","............","............","............",
       "............","............","..##....##..",".####..####.","#####..#####","#####..#####"]
P23 = ["....###.....","...#####....","...#####....","...#####....","....###.....",
       "...##..##...","..##....##..",".##......##.",".#........#.","##........##","#..........#"]
P24 = [".##...##....","###..####...","###..#####..",".##..#####..",".##....##...",
       ".##.....##..",".##.....##..",".##.....##..",".##.....##..",".##.....##..",".##.....##.."]
P25 = ["..##.......","..##.......","..####.....","...####....","....##.##..",
       "....##.##..","...##...##.","..##.....##","..#......##",".##....####",".##....####"]
P27 = ["....##.....","....##.....","...####....","...####....","..######...",
       "##.####..##","##..##...##","....##.....","....##.....","...##.##...","..##....##."]
P28 = [".##....##...",".##....##...",".####..###..",".####..###..","..##...##...",
       "..##...##...","..##...##...",".##.##.##.##",".##..#..#..#",".#...#..#..#",".#...#..#..#"]
P31 = ["..#...#....",".##.####...","..#######..","..########.",".##########",
       ".##########",".#########.","..#...#..#.","..#...#..#.",".##..##..##","..........."]
P11 = ["............","............","....##......","...#..#.....","..#....#....",
       ".#......#...","..#....#....","...#..#.....","....##......","............","............"]
TRK16 = ["..#######...","..#######...","..#######...","..#######...","..#######...",
         ".##..####...",".##..####...",".##..####...",".########...","....#..#....","............"]
CAR41 = ["..###..###..",".#####.#####",".#####.#####",".#####.#####","..#.#...#.#.",
         "............"]
TRK42 = ["...####.....","...####.....","..######....","..######....","..######....",
         "..##..##....","..##..##....","..######....","............","....#..#....","............"]
TL26 = ["...##.......","..####......","..####......","..####......","..####......",
        "..####......","..####......","..####......","..####......","..####......","..####......"]

def arrow_arts():
    """11-wide arts, centre column 5. White strokes on blue disc."""
    A35 = [".....#.....",
           "....###....",
           "...#####...",
           "..#######..",
           "....###....",
           "....###....",
           "....###....",
           "....###....",
           "....###....",
           "....###....",
           "....###...."]
    A33 = [".......###.",
           "......####.",
           "....#######",
           "....######.",
           "....###....",
           "....###....",
           "....###....",
           "....###....",
           "....###....",
           "....###....",
           "....###...."]
    def mirror(a):
        return [row[::-1] for row in a]
    A34 = mirror(A33)
    A36 = [".....#.....",
           "....###....",
           "...#####...",
           "..#######..",
           "....###....",
           "....###....",
           "....###....",
           "....###....",
           "....####...",
           "....######.",
           "....####..."]
    A37 = mirror(A36)
    return A33, A34, A35, A36, A37

def line_kernel(x0, y0, x1, y1, th):
    k = np.zeros((32, 32), np.float64)
    k[seg_dist(x0, y0, x1, y1) <= th] = 1.0
    return k

# ---------------------------------------------------------------- base kernels
def base_kernels():
    K = {}
    R = R0        # 12: red circle / triangle reference radius
    Rb = 9.0      # blue disc reference radius
    FI = 0.72     # inner-triangle scale (white interior)
    inner = poly_mask(tri_pts(R, True, FI))
    outer = poly_mask(tri_pts(R, True))
    inner_dn = poly_mask(tri_pts(R, False, FI))
    outer_dn = poly_mask(tri_pts(R, False))
    circ_in = mask_disc(0.0, 0.68 * R)

    # ---- sign frames (redness map)
    K['ring'] = mk(mask_disc(0.62 * R, 1.00 * R),
                   mask_disc(0.30 * R, 0.55 * R) | mask_disc(1.05 * R, 1.28 * R))
    outb = poly_mask(tri_pts(R, True, 1.12))
    K['tri'] = mk(outer & ~inner,
                  (outb & ~outer) | poly_mask(tri_pts(R, True, 0.50)))
    outb_dn = poly_mask(tri_pts(R, False, 1.12))
    K['tridn'] = mk(outer_dn & ~inner_dn,
                    (outb_dn & ~outer_dn) | poly_mask(tri_pts(R, False, 0.50)))
    g = np.cos(3 * (THETA + np.pi / 2))
    m3 = mask_disc(0.80 * R, 1.16 * R).astype(np.float64)
    K['sin3'] = mk(np.clip(g, 0, None) * m3, np.clip(-g, 0, None) * m3)
    K['reddisc'] = mk(mask_disc(0.12 * R, 0.95 * R), mask_disc(1.05 * R, 1.30 * R))
    # ---- blue disc
    K['bluedisc'] = mk(mask_disc(0.10 * Rb, 0.95 * Rb), mask_disc(1.02 * Rb, 1.25 * Rb))
    K['bluecen'] = mk(mask_disc(0.0, 0.24 * Rb), mask_disc(0.32 * Rb, 0.62 * Rb))
    # ---- yellow diamond (priority road)
    K['yelldisc'] = mk(mask_disc(0.0, 0.52 * R), mask_disc(0.58 * R, 0.85 * R))
    rh = np.abs(DX) / (0.72 * R) + np.abs(DY) / (0.95 * R)
    K['rhomb'] = mk((rh >= 0.74) & (rh <= 1.02),
                    (rh <= 0.55) | ((rh > 1.05) & (rh <= 1.30)))
    # ---- interior brightness (nv)
    K['ivbright'] = mk(mask_disc(0.30 * R, 0.60 * R), mask_disc(0.78 * R, 1.02 * R))
    K['ivtri'] = mk(inner, outer & ~inner)
    # ---- 45-degree stripe grating (end-of-limit), on nv
    lam = 0.42 * R
    u = (XS + YS) / lam
    mg = mask_disc(0.15 * R, 0.72 * R)
    for nm, ph in (('stripes', 0.0), ('stripec', np.pi / 2)):
        s = np.sin(2 * np.pi * u + ph) * mg
        K[nm] = mk(np.clip(s, 0, None), np.clip(-s, 0, None))
    # ---- no-entry bar / STOP letters, on nv
    K['bar'] = mk(rect_mask(16 - 0.55 * R, 16 + 0.55 * R, 16 - 0.13 * R, 16 + 0.13 * R),
                  rect_mask(16 - 0.42 * R, 16 + 0.42 * R, 16 - 0.34 * R, 16 - 0.18 * R) |
                  rect_mask(16 - 0.42 * R, 16 + 0.42 * R, 16 + 0.18 * R, 16 + 0.34 * R))
    pos = np.zeros((32, 32), bool); neg = np.zeros((32, 32), bool)
    for off in (-0.42, -0.14, 0.14, 0.42):
        pos |= rect_mask(16 + off * R - 0.10 * R, 16 + off * R + 0.10 * R,
                         16 - 0.17 * R, 16 + 0.17 * R)
    for off in (-0.28, 0.0, 0.28):
        neg |= rect_mask(16 + off * R - 0.05 * R, 16 + off * R + 0.05 * R,
                         16 - 0.15 * R, 16 + 0.15 * R)
    for y0, y1 in ((16 - 0.30 * R, 16 - 0.22 * R), (16 + 0.22 * R, 16 + 0.30 * R)):
        neg |= rect_mask(16 - 0.55 * R, 16 + 0.55 * R, y0, y1)
    K['stop'] = mk(pos, neg)
    # ---- roundabout: white ring with 3 gaps, dark centre, on nv
    mring = mask_disc(0.28 * Rb, 0.55 * Rb)
    ang = np.mod(THETA - np.pi / 2, 2 * np.pi / 3)
    ang = np.minimum(ang, 2 * np.pi / 3 - ang)
    K['wring'] = mk(mring & (ang > 0.45),
                    mask_disc(0.0, 0.22 * Rb) | mask_disc(0.60 * Rb, 0.72 * Rb))
    # ---- black-content generics (nblack)
    K['blacktot'] = mk(mask_disc(0.25 * R, 0.62 * R), mask_disc(0.72 * R, 0.95 * R))
    k = line_kernel(16 - 0.42 * R, 16 + 0.42 * R, 16 + 0.42 * R, 16 - 0.42 * R, 1.3)
    kf = dilate1(k)
    K['slash'] = mk(k, (dilate_mask(k > 0.35, 4) & ~dilate_mask(k > 0.35, 2)
                        & circ_in).astype(np.float64))
    # ---- class 9 / 10 blobs (halo surrounds)
    k = np.zeros((32, 32)); k[mask_disc(0.0, 0.22 * R, 16 - 0.33 * R, 16 + 0.08 * R)] = 1.0
    K['carL_red'] = mk(k, halo(k, mask_disc(0.0, 0.62 * R), 2))
    k = np.zeros((32, 32))
    k[rect_mask(16 - 0.28 * R, 16 + 0.28 * R, 16 - 0.35 * R, 16 - 0.02 * R)] = 1.0
    K['truck_red'] = mk(k, halo(k, mask_disc(0.0, 0.62 * R), 2))
    k = np.zeros((32, 32)); k[mask_disc(0.0, 0.22 * R, 16 + 0.33 * R, 16 + 0.08 * R)] = 1.0
    K['carR_blk'] = mk(k, halo(k, mask_disc(0.0, 0.62 * R), 2))
    k = np.zeros((32, 32))
    k[rect_mask(16 - 0.22 * R, 16 + 0.22 * R, 16 + 0.12 * R, 16 + 0.38 * R)] = 1.0
    K['carB_blk'] = mk(k, halo(k, mask_disc(0.0, 0.62 * R), 2))
    for nm, dy in (('dotred', -0.22), ('dotyel', -0.02), ('dotgrn', 0.30)):
        cy = 16 + dy * R
        k = np.zeros((32, 32)); k[mask_disc(0.0, 0.15 * R, 16, cy)] = 1.0
        K[nm] = mk(k, mask_disc(0.22 * R, 0.38 * R, 16, cy))
    # ---- speed-limit numbers: strokes + halo
    for num, nm in (("20", "n20"), ("30", "n30"), ("50", "n50"), ("60", "n60"),
                    ("70", "n70"), ("80", "n80")):
        k = dilate1(art_kernel(num_art(num), 0.96 * R, 1.00 * R))
        K[nm] = mk(k, halo(k, circ_in, 2))
    for num, nm in (("100", "n100"), ("120", "n120")):
        k = dilate1(art_kernel(num_art(num), 1.25 * R, 1.00 * R))
        K[nm] = mk(k, halo(k, circ_in, 2))
    # ---- triangle pictograms: strokes + halo inside inner triangle
    PICT = {
        'p11': (P11, 0.55, 0.55, 16.0), 'p19': (P19, 0.95, 0.80, 15.2),
        'p20': (P20, 0.95, 0.80, 15.2), 'p21': (P21, 0.95, 0.95, 16.0),
        'p22': (P22, 0.95, 0.50, 18.6), 'p23': (P23, 0.95, 0.85, 16.3),
        'p24': (P24, 0.95, 0.85, 16.0), 'p25': (P25, 0.95, 0.90, 16.5),
        'p27': (P27, 0.80, 0.90, 16.0), 'p28': (P28, 0.95, 0.90, 16.5),
        'p31': (P31, 0.95, 0.90, 16.0),
        'tlbox': (TL26, 0.65, 1.15, 16.5),
        'truck16': (TRK16, 0.85, 0.75, 16.3),
        'car41': (CAR41, 0.95, 0.42, 15.8), 'truck42': (TRK42, 0.55, 0.75, 16.5),
    }
    tri_big = poly_mask(tri_pts(R, True, 1.06))
    TRI_P = ('p11', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p27', 'p28', 'p31', 'tlbox')
    for nm, (art, bw, bh, cy) in PICT.items():
        k = dilate1(art_kernel(art, bw * R, bh * R, 16.0, cy))
        reg = tri_big if nm in TRI_P else mask_disc(0.0, 0.95 * R)
        K[nm] = mk(k, halo(k, reg, 2))
    # bicycle: two wheel rings + frame
    wl, wr_, wy = 16 - 0.26 * R, 16 + 0.26 * R, 16 + 0.14 * R
    k = np.zeros((32, 32))
    k[mask_disc(0.145 * R, 0.215 * R, wl, wy)] = 1.0
    k[mask_disc(0.145 * R, 0.215 * R, wr_, wy)] = 1.0
    for (x0, y0, x1, y1) in ((wl, wy, 16 + 0.03 * R, 16 - 0.28 * R),
                             (16 + 0.03 * R, 16 - 0.28 * R, wr_, wy),
                             (wl, wy, wr_, wy)):
        k[seg_dist(x0, y0, x1, y1) <= 0.075 * R] = 1.0
    K['p29'] = mk(dilate1(k), halo(k, tri_big, 2))
    # snowflake: 6 arms
    k = np.zeros((32, 32))
    for adeg in (90, 150, 210, 270, 330, 30):
        t = np.radians(adeg)
        k[seg_dist(16, 16.2, 16 + 0.42 * R * np.cos(t), 16.2 + 0.42 * R * np.sin(t)) <= 0.075 * R] = 1.0
    K['p30'] = mk(dilate1(k), halo(k, tri_big, 2))
    # ---- mandatory-arrow quadrant blobs (white-on-blue), on nv
    QB = {
        'aTL': (-0.50, -0.42), 'aTC': (0.0, -0.45), 'aTR': (+0.50, -0.42),
        'aML': (-0.55, 0.0), 'aC': (0.0, 0.0), 'aMR': (+0.55, 0.0),
        'aBL': (-0.50, +0.42), 'aBC': (0.0, +0.45), 'aBR': (+0.50, +0.42),
    }
    for qn, (ox, oy) in QB.items():
        cx, cy = 16 + ox * Rb, 16 + oy * Rb
        kq = np.zeros((32, 32)); kq[mask_disc(0.0, 0.17 * Rb, cx, cy)] = 1.0
        K[qn] = mk(kq, mask_disc(0.24 * Rb, 0.38 * Rb, cx, cy))
    return K

KB = base_kernels()

# ---------------------------------------------------------------- banks
SC_CIRC = [0.80, 0.90, 1.00, 1.10, 1.20]   # sign radius 9.6 .. 14.4 (base 12)
SC_TRI  = [0.80, 0.90, 1.00, 1.10, 1.20]
SC_SMALL= [0.62, 0.75, 0.87, 1.00, 1.12]   # radius 7.4 .. 13.4
SC_BLUE = [0.74, 0.87, 1.00, 1.15, 1.31]   # blue-disc radius 6.7 .. 11.8 (base 9)
ASP     = [0.85, 1.00, 1.18]

FEATIDX = {}
NBANKS = []

def build_bank(bname, mapname, K, specs, pool=5):
    """specs: list of (feature_name, base_kernel, scale_list, aspects|None)."""
    T = len(specs)
    kers = []
    for (nm, kb, scs, asps) in specs:
        vs = []
        for f in scs:
            for a in (asps or [1.0]):
                vs.append(finish(kb, f, a))
        assert len(vs) == K, (nm, len(vs), K)
        kers.extend(vs)
        FEATIDX[nm] = None   # fill after we know order
    W = np.stack(kers).astype(np.float32)[:, None, :, :]   # (T*K, 1, 32, 32)
    conv(bname, mapname, W, np.zeros(len(kers), np.float32), stride=1, padding=2)
    maxpool(bname + "_mp", bname, [pool, pool], [pool, pool])  # (T*K,1,1)
    reshapel(bname + "_rs", bname + "_mp", [T, K, 1])
    maxpool(bname + "_sc", bname + "_rs", [K, 1], [K, 1])  # (T,1,1)
    for i, (nm, _, _, _) in enumerate(specs):
        FEATIDX[nm] = (bname + "_sc", i)
    NBANKS.append(bname + "_sc")
    return bname + "_sc"

bk_red15 = build_bank("bkredA", "nred", len(SC_CIRC) * len(ASP), [
    ("ring", KB['ring'], SC_CIRC, ASP),
    ("tri", KB['tri'], SC_TRI, ASP),
    ("tridn", KB['tridn'], SC_TRI, ASP),
    ("sin3", KB['sin3'], SC_TRI, ASP),
])
bk_red5 = build_bank("bkredB", "nred", len(SC_SMALL), pool=3, specs=[
    ("reddisc", KB['reddisc'], SC_SMALL, None),
    ("carL_red", KB['carL_red'], [0.75, 0.85, 0.95, 1.0, 1.0], None),
    ("truck_red", KB['truck_red'], [0.75, 0.85, 0.95, 1.0, 1.0], None),
    ("dotred", KB['dotred'], SC_TRI, None),
])
bk_blue15 = build_bank("bkblueA", "nblue", len(SC_BLUE) * len(ASP), [
    ("bluedisc", KB['bluedisc'], SC_BLUE, ASP),
])
bk_blue5 = build_bank("bkblueB", "nblue", len(SC_BLUE), pool=3, specs=[
    ("bluecen", KB['bluecen'], SC_BLUE, None),
])
bk_yell5 = build_bank("bkyell", "nyell", len(SC_SMALL), pool=3, specs=[
    ("yelldisc", KB['yelldisc'], SC_SMALL, None),
    ("dotyel", KB['dotyel'], SC_TRI, None),
])
bk_grn5 = build_bank("bkgrn", "ngreen", len(SC_TRI), pool=3, specs=[
    ("dotgrn", KB['dotgrn'], SC_TRI, None),
])
bk_black5 = build_bank("bkblack", "nblack", len(SC_CIRC), [
    ("n20", KB['n20'], SC_CIRC, None), ("n30", KB['n30'], SC_CIRC, None),
    ("n50", KB['n50'], SC_CIRC, None), ("n60", KB['n60'], SC_CIRC, None),
    ("n70", KB['n70'], SC_CIRC, None), ("n80", KB['n80'], SC_CIRC, None),
    ("n100", KB['n100'], SC_CIRC, None), ("n120", KB['n120'], SC_CIRC, None),
    ("slash", KB['slash'], SC_CIRC, None),
    ("blacktot", KB['blacktot'], SC_CIRC, None),
    ("carR_blk", KB['carR_blk'], [0.75, 0.85, 0.95, 1.0, 1.0], None),
    ("carB_blk", KB['carB_blk'], [0.75, 0.85, 0.95, 1.0, 1.0], None),
    ("truck16", KB['truck16'], [0.75, 0.85, 0.95, 1.0, 1.0], None),
    ("p11", KB['p11'], SC_TRI, None), ("p19", KB['p19'], SC_TRI, None),
    ("p20", KB['p20'], SC_TRI, None), ("p21", KB['p21'], SC_TRI, None),
    ("p22", KB['p22'], SC_TRI, None), ("p23", KB['p23'], SC_TRI, None),
    ("p24", KB['p24'], SC_TRI, None), ("p25", KB['p25'], SC_TRI, None),
    ("p27", KB['p27'], SC_TRI, None), ("p28", KB['p28'], SC_TRI, None),
    ("p29", KB['p29'], SC_TRI, None), ("p30", KB['p30'], SC_TRI, None),
    ("p31", KB['p31'], SC_TRI, None), ("tlbox", KB['tlbox'], SC_TRI, None),
    ("car41", KB['car41'], SC_SMALL, None), ("truck42", KB['truck42'], SC_SMALL, None),
])
bk_nv15 = build_bank("bknvA", "nv", len(SC_TRI) * len(ASP), [
    ("ivtri", KB['ivtri'], SC_TRI, ASP),
])
bk_nv5 = build_bank("bknvB", "nv", len(SC_SMALL), pool=3, specs=[
    ("ivbright", KB['ivbright'], SC_CIRC, None),
    ("stripes", KB['stripes'], SC_SMALL, None),
    ("stripec", KB['stripec'], SC_SMALL, None),
    ("rhomb", KB['rhomb'], SC_SMALL, None),
    ("bar", KB['bar'], SC_SMALL, None),
    ("stop", KB['stop'], SC_SMALL, None),
    ("wring", KB['wring'], SC_BLUE, None),
    ("aTL", KB['aTL'], SC_BLUE, None), ("aTC", KB['aTC'], SC_BLUE, None),
    ("aTR", KB['aTR'], SC_BLUE, None), ("aML", KB['aML'], SC_BLUE, None),
    ("aC", KB['aC'], SC_BLUE, None), ("aMR", KB['aMR'], SC_BLUE, None),
    ("aBL", KB['aBL'], SC_BLUE, None), ("aBC", KB['aBC'], SC_BLUE, None),
    ("aBR", KB['aBR'], SC_BLUE, None),
])

# merge all banks into one feature tensor (F,1,1)
FEATS = list(FEATIDX.keys())
F = len(FEATS)
merges = []
for bi, bn in enumerate(NBANKS):
    src = bn
    T = sum(1 for nm in FEATS if FEATIDX[nm][0] == src)
    Wm = np.zeros((F, T, 1, 1), np.float32)
    tpos = {}
    for nm in FEATS:
        s, i = FEATIDX[nm]
        if s == src:
            if s not in tpos:
                tpos[s] = 0
            Wm[FEATS.index(nm), tpos[s], 0, 0] = 1.0
            tpos[s] += 1
    merges.append(conv("mrg%d" % bi, src, Wm, np.zeros(F, np.float32)))
addn("feat", *merges)
reshapel("featv", "feat", [F])

# ---------------------------------------------------------------- rules
# units: (feat, sgn, thr): sgn=+ -> relu(f - thr) "feature present"
#                      sgn=- -> relu(thr - f) "feature absent"
# class score = sum w * unit;  b2 auto-compensates absence units.
POS_THR, NEG_THR = 0.7, 0.35

UNITS = []
UNITKEY = {}

def unit(feat, sgn, thr):
    key = (feat, sgn, round(float(thr), 3))
    if key not in UNITKEY:
        UNITKEY[key] = len(UNITS)
        UNITS.append((FEATS.index(feat), sgn, float(thr)))
    return UNITKEY[key]

def R(feat, sgn, w, thr=None):
    thr = (POS_THR if sgn > 0 else NEG_THR) if thr is None else thr
    return (feat, sgn, float(w), float(thr))

def anti(feat_list, w):
    """rival suppression: penalty proportional to rivals being present"""
    return [R(f, +1, -w) for f in feat_list]

G1 = [R("ring", +1, 1.1), R("ivbright", +1, 0.5)] +      anti(["bluedisc", "tri", "sin3", "tridn", "reddisc", "yelldisc",
           "stripes", "stripec", "rhomb"], 0.8)
G2 = [R("tri", +1, 1.1), R("sin3", +1, 0.7), R("ivtri", +1, 0.5)] +      anti(["ring", "bluedisc", "tridn", "reddisc", "yelldisc",
           "stripes", "stripec", "rhomb", "ivbright"], 0.8)
G3 = [R("bluedisc", +1, 1.0), R("ring", +1, -2.0), R("tri", +1, -2.0),
      R("tridn", +1, -2.0), R("reddisc", +1, -2.0), R("yelldisc", +1, -2.0),
      R("stripes", +1, -1.0), R("stripec", +1, -1.0), R("rhomb", +1, -1.0)]
G4 = [R("stripes", +1, 0.9), R("stripec", +1, 0.9), R("ivbright", +1, 0.3)] +      anti(["ring", "bluedisc", "reddisc", "yelldisc", "tri", "tridn", "rhomb"], 0.8)

NUMS = ["n20", "n30", "n50", "n60", "n70", "n80", "n100", "n120"]
PICTS = ["p11", "p19", "p20", "p21", "p22", "p23", "p24", "p25", "p27",
         "p28", "p29", "p30", "p31", "tlbox"]
ARRS = ["arr33", "arr34", "arr35", "arr36", "arr37", "arr38", "arr39", "wring"]

RULES = {}
for ci, nf in enumerate(NUMS):
    cls = [0, 1, 2, 3, 4, 5, 7, 8][ci]
    RULES[cls] = G1 + [R(nf, +1, 2.2, 0.45)] + anti([f for f in NUMS if f != nf], 0.35) + [
        R("slash", +1, -0.6), R("blacktot", -1, 0.5, 0.8)] +         anti(["carL_red", "truck_red", "carR_blk", "carB_blk", "truck16"], 0.5)
RULES[6] = G1 + [R("slash", +1, 1.6), R("n80", +1, 0.9, 0.35)] +     anti([f for f in NUMS if f != "n80"], 0.3) +     anti(["carL_red", "truck_red", "carR_blk", "carB_blk", "truck16"], 0.5)
RULES[15] = G1 + [R("blacktot", -1, 1.3, 0.8), R("ivbright", +1, 0.3)] +     anti(NUMS, 0.45) + anti(["slash", "carL_red", "truck_red", "truck16",
                             "carR_blk", "carB_blk"], 0.5)
RULES[9] = G1 + [R("carL_red", +1, 1.6, 1.2), R("carR_blk", +1, 1.4, 1.9)] +     anti(NUMS, 0.3) + anti(["truck_red", "truck16", "carB_blk", "slash"], 0.6)
RULES[10] = G1 + [R("truck_red", +1, 1.6, 1.2), R("carB_blk", +1, 1.4, 1.9)] +     anti(NUMS, 0.3) + anti(["carL_red", "truck16", "carR_blk", "slash"], 0.6)
RULES[16] = G1 + [R("truck16", +1, 1.8, 1.0)] + anti(NUMS, 0.35) +     anti(["carL_red", "truck_red", "carR_blk", "carB_blk", "slash"], 0.6) +     [R("blacktot", -1, 0.4, 0.8)]
for cls, pf in [(11, "p11"), (19, "p19"), (20, "p20"), (21, "p21"), (22, "p22"),
                (23, "p23"), (24, "p24"), (25, "p25"), (27, "p27"), (28, "p28"),
                (29, "p29"), (30, "p30"), (31, "p31")]:
    RULES[cls] = G2 + [R(pf, +1, 1.8)] + anti([f for f in PICTS if f != pf], 0.3)
RULES[18] = G2 + [R("blacktot", -1, 0.8, 0.8), R("ivtri", +1, 0.3)] + anti(PICTS, 0.22)
RULES[26] = G2 + [R("tlbox", +1, 1.0), R("dotgrn", +1, 1.4), R("dotred", +1, 0.8),
                  R("dotyel", +1, 0.6)] + anti(PICTS, 0.25)
def blobs(req, forb):
    return [R("a" + b, +1, 1.6, 1.0) for b in req] + \
           [R("a" + b, +1, -1.6, 0.8) for b in forb]
RULES[33] = G3 + blobs("TR C BC".split(), "TL BL BR".split())
RULES[34] = G3 + blobs("TL C BC".split(), "TR BR BL".split())
RULES[35] = G3 + blobs("TC C BC".split(), "TL TR BL BR".split())
RULES[36] = G3 + blobs("TC C BC BR".split(), "TL TR BL".split())
RULES[37] = G3 + blobs("TC C BC BL".split(), "TR BR TL".split())
RULES[38] = G3 + blobs("TR BL".split(), "TL BR BC".split())
RULES[39] = G3 + blobs("TL BR".split(), "TR BL BC".split())

RULES[40] = G3 + [R("bluecen", +1, 2.6), R("wring", +1, 0.6)]
RULES[12] = [R("yelldisc", +1, 1.8), R("rhomb", +1, 0.5)] +     anti(["ring", "tri", "tridn", "bluedisc", "reddisc", "stripes", "stripec",
          "sin3", "ivbright"], 0.8)
RULES[13] = [R("tridn", +1, 1.6), R("ivbright", +1, 0.5), R("sin3", -1, 0.9),
             R("ivtri", +1, -0.5)] +     anti(["ring", "tri", "bluedisc", "yelldisc", "reddisc", "stripes", "stripec",
          "rhomb"], 0.8)
BASE14 = anti(["ring", "tri", "tridn", "bluedisc", "yelldisc", "sin3",
               "stripes", "stripec", "rhomb"], 0.5)
RULES[14] = [R("reddisc", +1, 1.6), R("stop", +1, 1.5), R("bar", +1, -0.9, 1.0)] + BASE14
RULES[17] = [R("reddisc", +1, 1.6), R("bar", +1, 1.4), R("stop", +1, -0.9)] + BASE14
RULES[32] = G4 + [R("blacktot", -1, 0.4, 1.0)] +     anti(["car41", "truck42", "slash"], 0.7)
RULES[41] = G4 + [R("car41", +1, 1.8), R("blacktot", +1, 0.4)] +     anti(["truck42", "slash"], 0.7)
RULES[42] = G4 + [R("truck42", +1, 1.8), R("blacktot", +1, 0.4)] +     anti(["car41", "slash"], 0.7)

assert sorted(RULES.keys()) == list(range(43)), sorted(RULES.keys())

# ---------------------------------------------------------------- dense
for c in range(43):                     # materialize all units first
    for (feat, sgn, w, thr) in RULES[c]:
        unit(feat, sgn, thr)
U = len(UNITS)
W1 = np.zeros((U, F), np.float32)
b1 = np.zeros(U, np.float32)
for i, (fi, sgn, thr) in enumerate(UNITS):
    W1[i, fi] = sgn
    b1[i] = -sgn * thr
W2 = np.zeros((43, U), np.float32)
b2 = np.zeros(43, np.float32)
for c in range(43):
    for (feat, sgn, w, thr) in RULES[c]:
        ui = unit(feat, sgn, thr)
        W2[c, ui] += w
        if sgn < 0:
            b2[c] -= w * thr
dense("u_lin", "featv", W1, b1)
relu("h", "u_lin")
dense("out", "h", W2, b2)

# ---------------------------------------------------------------- write
import os
os.makedirs(OUT, exist_ok=True)
net = {"layers": LAYERS}
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump(net, f)
np.savez(os.path.join(OUT, "weights.npz"), **{k: v.astype(np.float32) for k, v in WEIGHTS.items()})
tot = sum(int(np.prod(v.shape)) for v in WEIGHTS.values())
print("layers:", len(LAYERS), " features:", F, " units:", U, " weights:", tot)
