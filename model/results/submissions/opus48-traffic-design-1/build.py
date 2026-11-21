"""Build a hand-designed traffic-sign classifier: write network.json and weights.npz.

Every weight comes from geometric templates of the 43 German sign classes, drawn from general
knowledge of their standardised shapes/colours.  No training data is read or used.

Usage: python3 build.py OUT_DIR
"""
import sys, os, json
import numpy as np

N = 32
yy, xx = np.mgrid[0:N, 0:N].astype(float)   # yy = row (y, down), xx = col (x)
C = (N - 1) / 2.0                            # 15.5 centre

# ---------------------------------------------------------------- drawing utils
def disc(r, cx=C, cy=C):
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(float)

def annulus(r0, r1, cx=C, cy=C):
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return ((d >= r0) & (d <= r1)).astype(float)

def seg_dist(ax, ay, bx, by):
    """distance from every grid point to segment A-B"""
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby + 1e-9
    t = ((xx - ax) * abx + (yy - ay) * aby) / L2
    t = np.clip(t, 0.0, 1.0)
    px, py = ax + t * abx, ay + t * aby
    return np.sqrt((xx - px) ** 2 + (yy - py) ** 2)

def stroke(ax, ay, bx, by, w=1.6):
    """soft line segment of half-width ~w"""
    d = seg_dist(ax, ay, bx, by)
    return np.clip(w - d + 0.5, 0.0, 1.0)

def polyline(pts, w=1.6):
    m = np.zeros((N, N))
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        m = np.maximum(m, stroke(ax, ay, bx, by, w))
    return m

def convex_mask(verts):
    cxv = np.mean([v[0] for v in verts]); cyv = np.mean([v[1] for v in verts])
    inside = np.ones((N, N), bool)
    n = len(verts)
    for i in range(n):
        x0, y0 = verts[i]; x1, y1 = verts[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        cp = ex * (yy - y0) - ey * (xx - x0)
        cc = ex * (cyv - y0) - ey * (cxv - x0)
        inside &= (cp * np.sign(cc) >= -0.3)
    return inside.astype(float)

def shrink(verts, f):
    cxv = np.mean([v[0] for v in verts]); cyv = np.mean([v[1] for v in verts])
    return [(cxv + (x - cxv) * f, cyv + (y - cyv) * f) for (x, y) in verts]

def tri_up(a=2.0):     # warning triangle, apex up
    return [(C, a + 0.5), (a + 1.0, N - 2.0 - a), (N - 1.0 - a, N - 2.0 - a)]
def tri_down(a=2.0):   # yield, apex down
    return [(a + 1.0, a + 2.0), (N - 1.0 - a, a + 2.0), (C, N - 2.0 - a)]
def diamond(a=1.0):
    return [(C, a), (N - 1.0 - a, C), (C, N - 1.0 - a), (a, C)]
def octagon(r=13.5):
    import math
    return [(C + r * math.cos(math.radians(22.5 + 45 * k)),
             C + r * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]

def outline(verts, th=0.76):
    return np.clip(convex_mask(verts) - convex_mask(shrink(verts, th)), 0, 1)

def blur(a, sigma):
    if sigma <= 0:
        return a.astype(float)
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    out = a.astype(float)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), -1, out)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), -2, out)
    return out

# feature channel indices after the 1x1 conv
RED, BLUE, YEL, LUM, DRK = 0, 1, 2, 3, 4
NF = 5

def chan(d):
    """build a (NF,32,32) template from {channel: 2d map}"""
    T = np.zeros((NF, N, N))
    for c, m in d.items():
        T[c] = T[c] + m
    return T

def unit(T):
    v = T.reshape(-1).astype(float)
    n = np.sqrt((v * v).sum())
    return (v / n) if n > 1e-12 else v

# ---------------------------------------------------------------- arrow helper
def arrow(pts, w=1.9, head=4.2):
    m = polyline(pts, w)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy) + 1e-9; dx, dy = dx / L, dy / L
    px, py = -dy, dx
    b1 = (x1 - dx * head + px * head * 0.75, y1 - dy * head + py * head * 0.75)
    b2 = (x1 - dx * head - px * head * 0.75, y1 - dy * head - py * head * 0.75)
    m = np.maximum(m, stroke(x1, y1, b1[0], b1[1], w))
    m = np.maximum(m, stroke(x1, y1, b2[0], b2[1], w))
    return m

def rect(x0, x1, y0, y1):
    return (((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1))).astype(float)

# ---------------------------------------------------------------- digit kernels (13x13, dark channel)
def kstroke(K, ax, ay, bx, by, w):
    gy, gx = np.mgrid[0:K, 0:K].astype(float)
    abx, aby = bx - ax, by - ay; L2 = abx * abx + aby * aby + 1e-9
    t = np.clip(((gx - ax) * abx + (gy - ay) * aby) / L2, 0, 1)
    px, py = ax + t * abx, ay + t * aby
    d = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)
    return np.clip(w - d + 0.5, 0, 1)

def digit_glyph(dig):
    K = 13; m = np.zeros((K, K))
    x0, x1, y0, ym, y1 = 3.0, 9.0, 1.0, 6.0, 11.0
    seg = {'a': (x0, y0, x1, y0), 'b': (x1, y0, x1, ym), 'c': (x1, ym, x1, y1),
           'd': (x0, y1, x1, y1), 'e': (x0, ym, x0, y1), 'f': (x0, y0, x0, ym),
           'g': (x0, ym, x1, ym)}
    use = {0: 'abcdef', 2: 'abged', 3: 'abgcd', 5: 'afgcd',
           6: 'afgecd', 7: 'abc', 8: 'abcdefg'}
    if dig == 1:
        xm = (x0 + x1) / 2
        m = np.maximum(m, kstroke(K, xm, y0, xm, y1, 1.35))
    else:
        for s in use[dig]:
            ax, ay, bx, by = seg[s]
            m = np.maximum(m, kstroke(K, ax, ay, bx, by, 1.35))
    m = blur(m, 0.55)
    m = m - m.mean()
    n = np.sqrt((m * m).sum())
    return m / n if n > 1e-9 else m

# ---------------------------------------------------------------- 7-seg digit on 32x32
SEG = {'a': 'a', }
USE = {0: 'abcdef', 2: 'abged', 3: 'abgcd', 5: 'afgcd', 6: 'afgecd', 7: 'abc', 8: 'abcdefg', 9: 'abcdfg'}

def digit32(d, box, w=1.6):
    x0, x1, y0, ym, y1 = box
    if d == 1:
        xm = 0.5 * (x0 + x1)
        return stroke(xm, y0, xm, y1, w)
    seg = {'a': (x0, y0, x1, y0), 'b': (x1, y0, x1, ym), 'c': (x1, ym, x1, y1),
           'd': (x0, y1, x1, y1), 'e': (x0, ym, x0, y1), 'f': (x0, y0, x0, ym),
           'g': (x0, ym, x1, ym)}
    m = np.zeros((N, N))
    for s in USE[d]:
        ax, ay, bx, by = seg[s]
        m = np.maximum(m, stroke(ax, ay, bx, by, w))
    return m

# ---------------------------------------------------------------- build everything
def build():
    # ----- 1x1 feature conv -> red(pure), blue, yellow(gold), lum, dark
    featW = np.zeros((NF, 3, 1, 1), np.float32)
    featW[RED, :, 0, 0] = [1.0, -1.0, -1.0]           # traffic red: red+, gold~0, blue<0
    featW[BLUE, :, 0, 0] = [-0.5, -0.5, 1.0]
    featW[YEL, :, 0, 0] = [0.0, 1.0, -1.0]            # gold/yellow: G-B (rejects red)
    featW[LUM, :, 0, 0] = [1 / 3, 1 / 3, 1 / 3]
    featW[DRK, :, 0, 0] = [-1 / 3, -1 / 3, -1 / 3]
    featb = np.zeros(NF, np.float32)

    def T(d, sigma=1.0):
        A = np.zeros((NF, N, N))
        for c, m in d.items():
            A[c] = A[c] + m
        if sigma > 0:
            A = blur(A, sigma)
        return unit(A)

    interior = convex_mask(shrink(tri_up(), 0.74))

    # ===== branch A: global group / content detectors =====
    gA = []
    gA.append(T({RED: annulus(9.5, 15.5) - 0.6 * disc(7.5)}))   # 0 red ring
    gA.append(T({RED: outline(tri_up(), 0.74)}))                # 1 red triangle up (warning)
    gA.append(T({RED: outline(tri_down(), 0.62)}))              # 2 yield
    gA.append(T({YEL: convex_mask(diamond(3.0))}))              # 3 gold diamond (priority road)
    gA.append(T({BLUE: disc(13.0)}))                            # 4 blue disc
    gA.append(T({RED: disc(7.5)}))                              # 5 red fill centre (stop/no-entry)
    gA.append(T({DRK: stroke(24, 8, 8, 24, 2.3),
                 RED: -0.4 * disc(13), BLUE: -0.4 * disc(13), YEL: -0.4 * disc(13)}))  # 6 grey+diag
    gA.append(T({LUM: rect(5, 27, 14.2, 17.8)}))                # 7 white bar (no entry)
    st = np.zeros((N, N))
    for bx in (8.5, 12.5, 18.5, 22.5):
        st = np.maximum(st, rect(bx - 1.3, bx + 1.3, 13.5, 18.5))
    gA.append(T({LUM: st}))                                     # 8 stop text
    gA.append(T({RED: rect(9, 22, 13, 20)}))                    # 9 centre red (no-passing car)
    gA.append(T({DRK: rect(8, 24, 12, 21)}))                    # 10 solid dark centre (truck)
    gA.append(T({LUM: disc(7.0)}))                              # 11 clean bright centre (no vehicles)
    gA.append(T({DRK: rect(9, 23, 10, 22)}))                    # 12 end-of-80 digits
    gA.append(T({DRK: disc(8.5)}))                              # 13 ink amount in interior
    gA.append(T({DRK: rect(16, 24, 9, 15)}))                    # 14 upper-right dark (truck taller)
    gpA_W = np.stack(gA).astype(np.float32)
    Na = gpA_W.shape[0]

    WA = np.zeros((43, Na), np.float32)
    WG = 3.0
    ring = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
    tri = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    bluec = [33, 34, 35, 36, 37, 38, 39, 40]
    greyc = [6, 32, 41, 42]
    for c in ring: WA[c, 0] = WG
    for c in tri:  WA[c, 1] = WG
    WA[13, 2] = WG
    WA[12, 3] = WG
    for c in bluec: WA[c, 4] = WG
    WA[14, 5] = WG; WA[17, 5] = WG
    for c in greyc: WA[c, 6] = WG
    WA[17, 7] = 1.4          # white bar -> no entry
    WA[14, 8] = 1.0          # stop text -> stop
    WA[9, 9] = 1.4; WA[10, 9] = 1.4        # centre red -> no passing
    WA[9, 14] = -0.7; WA[10, 14] = 0.7     # tall right object -> truck variant (10)
    WA[16, 10] = 0.25        # solid dark centre -> no trucks
    WA[15, 11] = 0.0         # clean centre -> no vehicles DISABLED (was a sink absorbing most signs)
    WA[15, 13] = -1.5        # interior-ink penalty keeps 15 from winning
    WA[6, 12] = 0.8          # end digits -> end of 80
    bA = np.zeros(43, np.float32)

    # ===== branch B: speed-limit first digit (global templates, contrastive) =====
    boxA = (8.5, 15.0, 9.5, 16.0, 22.5)    # first digit of a 2-digit number
    box1 = (6.0, 11.5, 10.0, 16.0, 22.0)   # leading '1' of a 3-digit number
    boxC = (13.0, 19.5, 10.0, 16.0, 22.0)  # middle digit of a 3-digit number
    tplB = [digit32(2, boxA), digit32(3, boxA), digit32(5, boxA), digit32(6, boxA),
            digit32(7, boxA), digit32(8, boxA), digit32(1, box1), digit32(2, boxC)]
    gpB_W = np.stack([T({DRK: m}, sigma=1.3) for m in tplB]).astype(np.float32)
    nb = gpB_W.shape[0]            # 8
    WB = np.zeros((43, nb), np.float32)
    wB = 1.3
    two = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}   # class 0..5 -> template index of its first digit
    for cls, ti in two.items():
        WB[cls, ti] += wB
        for j in range(6):
            WB[cls, j] -= wB / 6.0
    w1, w1b = 0.9, 0.5
    for cls in (7, 8):
        WB[cls, 6] += w1
        for j in range(6):
            WB[cls, j] -= w1 / 6.0
    WB[7, 7] -= w1b          # 100: no '2' in the middle
    WB[8, 7] += w1b          # 120: a '2' in the middle
    for j in range(nb):      # suppress empty/truck when a digit shape is present
        WB[15, j] -= 0.5
        WB[16, j] -= 0.3
    bB = np.zeros(43, np.float32)

    # ===== branch C: blue arrows (global) =====
    def arrow_T(mask):
        return T({LUM: mask - 0.4 * disc(12.5)}, sigma=1.0)
    arr = [
        arrow_T(arrow([(C, 24), (C, 9)])),                                     # ahead 35
        arrow_T(arrow([(C - 1, 23), (C - 1, 13), (C + 6, 13)])),               # turn right 33
        arrow_T(arrow([(C + 1, 23), (C + 1, 13), (C - 6, 13)])),               # turn left 34
        arrow_T(np.maximum(arrow([(C, 24), (C, 9)]), arrow([(C, 16), (C + 6, 16)]))),  # str/right 36
        arrow_T(np.maximum(arrow([(C, 24), (C, 9)]), arrow([(C, 16), (C - 6, 16)]))),  # str/left 37
        arrow_T(arrow([(C - 5, 9), (C + 5, 21)])),                             # keep right 38
        arrow_T(arrow([(C + 5, 9), (C - 5, 21)])),                             # keep left 39
        T({LUM: annulus(5.0, 8.0) - 0.4 * disc(12.5)}, sigma=1.0),             # roundabout 40
    ]
    gpC_W = np.stack(arr).astype(np.float32)
    na = len(arr)
    WC = np.zeros((43, na), np.float32)
    wC = 1.3
    for k, c in enumerate([35, 33, 34, 36, 37, 38, 39, 40]):   # contrastive arrow readout
        WC[c, k] += wC
        for kk in range(na):
            WC[c, kk] -= wC / na
    bC = np.zeros(43, np.float32)

    # ===== branch D: warning-triangle symbols (global, contrastive) =====
    def sym_T(mask):
        return T({DRK: mask - 0.3 * interior}, sigma=0.8)
    S = []
    S.append(sym_T(np.maximum(stroke(C, 10, C, 23, 2.2), stroke(C - 6, 16, C + 6, 16, 1.5))))  # 11 cross
    S.append(sym_T(np.maximum(stroke(C, 10, C, 19, 1.8), disc(1.4, C, 22))))   # 18 exclamation
    S.append(sym_T(arrow([(C, 23), (C, 16), (C - 5, 12)])))                    # 19 bend left
    S.append(sym_T(arrow([(C, 23), (C, 16), (C + 5, 12)])))                    # 20 bend right
    S.append(sym_T(polyline([(C - 3, 23), (C - 3, 18), (C + 3, 14), (C + 3, 10)], 1.7)))  # 21 double
    m = polyline([(C - 8, 22), (C + 8, 22)], 1.4)
    m = np.maximum(m, polyline([(C - 7, 22), (C - 5, 18.5), (C - 2, 22)], 1.5))
    m = np.maximum(m, polyline([(C + 2, 22), (C + 5, 18.5), (C + 7, 22)], 1.5))
    S.append(sym_T(m))                                                         # 22 bumpy
    m = rect(C - 5, C + 5, 11, 15)
    m = np.maximum(m, polyline([(C - 6, 20), (C - 4, 18), (C - 2, 20)], 1.3))
    m = np.maximum(m, polyline([(C + 2, 20), (C + 4, 18), (C + 6, 20)], 1.3))
    S.append(sym_T(m))                                                         # 23 slippery
    S.append(sym_T(np.maximum(stroke(C - 6, 22, C - 3, 11, 1.5), stroke(C + 6, 22, C + 3, 11, 1.5))))  # 24
    m = disc(1.6, C - 1, 11)
    m = np.maximum(m, stroke(C - 1, 12, C - 1, 19, 1.7))
    m = np.maximum(m, convex_mask([(C + 1, 22), (C + 8, 22), (C + 8, 16)]))
    S.append(sym_T(m))                                                         # 25 road works
    S.append(sym_T(rect(C - 2.2, C + 2.2, 9, 23)))                             # 26 traffic lights
    m = disc(1.5, C, 11)
    m = np.maximum(m, stroke(C, 12, C, 18, 1.6))
    m = np.maximum(m, polyline([(C, 18), (C - 2.5, 22)], 1.3))
    m = np.maximum(m, polyline([(C, 18), (C + 2.5, 22)], 1.3))
    S.append(sym_T(m))                                                         # 27 pedestrian
    m = np.zeros((N, N))
    for cx in (C - 3.5, C + 3.5):
        m = np.maximum(m, disc(1.3, cx, 12))
        m = np.maximum(m, stroke(cx, 13, cx, 19, 1.4))
    S.append(sym_T(m))                                                         # 28 children
    m = np.maximum(annulus(1.6, 2.7, C - 4.5, 19), annulus(1.6, 2.7, C + 4.5, 19))
    m = np.maximum(m, stroke(C - 4.5, 19, C + 4.5, 19, 1.2))
    m = np.maximum(m, stroke(C, 19, C + 1, 13, 1.2))
    S.append(sym_T(m))                                                         # 29 cyclist
    m = np.zeros((N, N)); cy = 16
    for (ex, ey) in [(0, 7), (6, 3.5), (6, -3.5)]:
        m = np.maximum(m, stroke(C - ex, cy - ey, C + ex, cy + ey, 1.3))
    S.append(sym_T(m))                                                         # 30 snowflake
    m = stroke(C - 6, 17, C + 3, 15, 2.2)
    m = np.maximum(m, stroke(C + 3, 15, C + 7, 10, 1.5))
    for lx in (C - 5, C - 2, C + 1):
        m = np.maximum(m, stroke(lx, 17, lx, 22, 1.1))
    S.append(sym_T(m))                                                         # 31 deer
    gpD_W = np.stack(S).astype(np.float32)
    WD = np.zeros((43, len(S)), np.float32)
    wD = 0.9
    for k, c in enumerate(tri):
        WD[c, k] += wD
        for kk in range(len(tri)):
            WD[c, kk] -= wD / len(tri)
    bD = np.zeros(43, np.float32)

    weights = {
        "feat.W": featW, "feat.b": featb,
        "gpA.W": gpA_W, "gpA.b": np.zeros(Na, np.float32),
        "scoreA.W": WA, "scoreA.b": bA,
        "gpB.W": gpB_W, "gpB.b": np.zeros(nb, np.float32),
        "scoreB.W": WB, "scoreB.b": bB,
        "gpC.W": gpC_W, "gpC.b": np.zeros(len(arr), np.float32),
        "scoreC.W": WC, "scoreC.b": bC,
        "gpD.W": gpD_W, "gpD.b": np.zeros(len(S), np.float32),
        "scoreD.W": WD, "scoreD.b": bD,
    }
    layers = [
        {"name": "feat", "op": "conv", "inputs": ["image"], "out": NF, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "featn", "op": "norm", "inputs": ["feat"]},
        {"name": "gpA", "op": "dense", "inputs": ["featn"], "out": Na},
        {"name": "gpAr", "op": "relu", "inputs": ["gpA"]},
        {"name": "scoreA", "op": "dense", "inputs": ["gpAr"], "out": 43},
        {"name": "gpB", "op": "dense", "inputs": ["featn"], "out": nb},
        {"name": "gpBr", "op": "relu", "inputs": ["gpB"]},
        {"name": "scoreB", "op": "dense", "inputs": ["gpBr"], "out": 43},
        {"name": "gpC", "op": "dense", "inputs": ["featn"], "out": len(arr)},
        {"name": "gpCr", "op": "relu", "inputs": ["gpC"]},
        {"name": "scoreC", "op": "dense", "inputs": ["gpCr"], "out": 43},
        {"name": "gpD", "op": "dense", "inputs": ["featn"], "out": len(S)},
        {"name": "gpDr", "op": "relu", "inputs": ["gpD"]},
        {"name": "scoreD", "op": "dense", "inputs": ["gpDr"], "out": 43},
        {"name": "total", "op": "add", "inputs": ["scoreA", "scoreB", "scoreC", "scoreD"]},
    ]
    return layers, weights


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    layers, weights = build()
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
