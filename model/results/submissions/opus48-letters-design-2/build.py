"""Build a hand-designed ASL letter classifier's weights with NO training data.

Idea: a fixed formula-based front end (standardise, oriented Gaussian-derivative
and blur filters, per-channel standardise, rectify, coarse average pool) turns a
28x28 image into a magnitude descriptor.  For each letter we RENDER an idealised
hand silhouette from ASL geometry (typed from general knowledge), push it through
the same front end, and use the resulting descriptor as a matched filter (one dense
row).  Several shifted/scaled renders per letter give translation/scale tolerance;
a max-pool over a letter's variants keeps its best match.  Every number comes from
formulas or from geometry typed here -- no image, statistic or label is read.

Usage: python3 build.py OUT_DIR    (writes OUT_DIR/network.json, OUT_DIR/weights.npz)
"""
import sys, os, json
import numpy as np

# ------------------------------------------------------------------ front end
NORM_EPS = 1e-5
K = 5                # conv kernel size
PAD = 2
POOL = 2             # avgpool factor -> 14x14 grid
GRID = 28 // POOL
SIGMA = 1.1
LINE_ANGLES = [0.0, 45.0, 90.0, 135.0]
NBASE = 1 + len(LINE_ANGLES)     # 5
NCH = 2 * NBASE                  # 10


def gaussian_kernels():
    r = K // 2
    ax = np.arange(-r, r + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    s2 = SIGMA ** 2
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * s2)); g /= g.sum()
    e = np.exp(-(xx ** 2 + yy ** 2) / (2 * s2))
    Gxx = (xx ** 2 / s2 - 1.0) / s2 * e
    Gyy = (yy ** 2 / s2 - 1.0) / s2 * e
    Gxy = (xx * yy / s2 ** 2) * e
    ker = [g]
    bars = [Gxx, 0.5 * Gxx + Gxy + 0.5 * Gyy, Gyy, 0.5 * Gxx - Gxy + 0.5 * Gyy]
    for d in bars:
        d = d - d.mean(); d /= np.abs(d).sum() + 1e-12
        ker.append(d)
    return ker


def conv_weights():
    """conv with NCH out channels: for base j, channel 2j = Kj, channel 2j+1 = -Kj."""
    ker = gaussian_kernels()
    W = np.zeros((NCH, 1, K, K), np.float64)
    for j, kr in enumerate(ker):
        W[2 * j, 0] = kr
        W[2 * j + 1, 0] = -kr
    b = np.zeros((NCH,), np.float64)
    return W, b


def _standardize(x):
    m = x.mean()
    v = ((x - m) ** 2).mean()
    return (x - m) / np.sqrt(v + NORM_EPS)


def _conv_same(img, ker):
    r = K // 2
    xp = np.pad(img, ((r, r), (r, r)))
    out = np.zeros_like(img, dtype=np.float64)
    for dy in range(K):
        for dx in range(K):
            out += ker[dy, dx] * xp[dy:dy + 28, dx:dx + 28]
    return out


def _avgpool(x):
    return x.reshape(GRID, POOL, GRID, POOL).mean(axis=(1, 3))


def descriptor(img, ker):
    """Replicate the network front end and return the tied magnitude descriptor
    (NBASE, GRID, GRID): avgpool(|standardise(conv(standardise(img)))|)."""
    n0 = _standardize(img.astype(np.float64))
    tied = np.zeros((NBASE, GRID, GRID), np.float64)
    for j in range(NBASE):
        g = _conv_same(n0, ker[j])
        s = _standardize(g)
        tied[j] = _avgpool(np.abs(s))
    return tied


# ------------------------------------------------------------------ geometry
SS = 3
R = 28 * SS
_yy, _xx = np.mgrid[0:R, 0:R].astype(np.float64)
_YB = _yy / SS + 0.5 / SS
_XB = _xx / SS + 0.5 / SS

FX = {"index": 10.2, "middle": 13.1, "ring": 16.0, "pinky": 18.8}
PALM_C = (14.4, 17.2)
PALM_RX, PALM_RY = 6.3, 6.6
YTOP = 11.5
FW = 1.45
FULL_TIP_Y = 2.6


class Canvas:
    def __init__(self):
        self.m = np.zeros((R, R), np.float64)

    def _add(self, mask):
        self.m = np.maximum(self.m, mask)

    def capsule(self, p0, p1, r, aa=0.8):
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        if L2 < 1e-6:
            t = np.zeros_like(_XB)
        else:
            t = np.clip(((_XB - x0) * dx + (_YB - y0) * dy) / L2, 0.0, 1.0)
        px = x0 + t * dx; py = y0 + t * dy
        d = np.sqrt((_XB - px) ** 2 + (_YB - py) ** 2)
        self._add(np.clip((r - d) / aa + 0.5, 0.0, 1.0))

    def disk(self, c, rx, ry, aa=0.8):
        x0, y0 = c
        d = np.sqrt(((_XB - x0) / rx) ** 2 + ((_YB - y0) / ry) ** 2)
        self._add(np.clip((1.0 - d) * min(rx, ry) / aa + 0.5, 0.0, 1.0))

    def ring(self, c, rx, ry, inner, gap_right=False):
        x0, y0 = c
        d = np.sqrt(((_XB - x0) / rx) ** 2 + ((_YB - y0) / ry) ** 2)
        outer = np.clip((1.0 - d) * min(rx, ry) / 0.8 + 0.5, 0.0, 1.0)
        di = np.sqrt(((_XB - x0) / inner) ** 2 + ((_YB - y0) / inner) ** 2)
        hole = np.clip((1.0 - di) * inner / 0.8 + 0.5, 0.0, 1.0)
        r = np.clip(outer - hole, 0.0, 1.0)
        if gap_right:
            r = r * (1 - ((_XB > x0 + 0.3) & (np.abs(_YB - y0) < ry * 0.55)).astype(np.float64))
        self._add(r)

    def image(self):
        return self.m.reshape(28, SS, 28, SS).mean(axis=(1, 3))


def _finger(cv, name, angle_deg, length, width=FW, base_dy=0.0):
    bx = FX[name]; by = YTOP + base_dy
    a = np.radians(angle_deg)
    cv.capsule((bx, by), (bx + np.sin(a) * length, by - np.cos(a) * length), width)


def render(letter):
    cv = Canvas()
    palm = lambda c=PALM_C, rx=PALM_RX, ry=PALM_RY: cv.disk(c, rx, ry)
    wrist = lambda: cv.capsule((14.0, 23.0), (14.6, 29.5), 4.2)
    up = lambda name, ang=0.0, ln=None, w=FW: _finger(cv, name, ang, (YTOP - FULL_TIP_Y) if ln is None else ln, w)
    knuck = lambda ty: [cv.disk((FX[n], ty), 1.7, 1.9) for n in ("index", "middle", "ring", "pinky")]
    thumb_tuck = lambda: cv.capsule((8.0, 18.5), (13.5, 16.8), 1.7)
    thumb_side = lambda: cv.capsule((8.6, 19.0), (9.3, 12.5), 1.7)
    thumb_left = lambda: cv.capsule((9.5, 16.5), (3.2, 18.3), 1.7)
    thumb_front = lambda: cv.capsule((8.8, 16.0), (16.5, 16.8), 1.8)

    L = letter
    if L == "A":
        cv.disk((14.4, 15.5), 6.3, 7.0); knuck(10.6); thumb_side()
    elif L == "B":
        [up(n) for n in ("index", "middle", "ring", "pinky")]; thumb_tuck()
    elif L == "C":
        cv.ring((13.0, 15.5), 8.2, 8.6, 4.7, gap_right=True); wrist()
    elif L == "D":
        up("index"); cv.disk((14.8, 16.0), 5.0, 5.2); wrist()
    elif L == "E":
        cv.disk((14.4, 16.2), 6.3, 6.6); knuck(11.6); thumb_tuck()
    elif L == "F":
        up("middle", 3); up("ring", 8); up("pinky", 14)
        cv.ring((9.6, 13.2), 2.9, 2.9, 1.3); palm(); wrist()
    elif L == "G":
        cv.disk((17.5, 16.0), 5.6, 6.0); wrist()
        cv.capsule((13.0, 13.5), (2.5, 13.0), FW); cv.capsule((13.0, 15.8), (4.0, 15.8), 1.5)
    elif L == "H":
        cv.disk((18.0, 16.2), 5.4, 6.0); wrist()
        cv.capsule((13.5, 12.8), (2.5, 12.4), FW); cv.capsule((13.5, 15.4), (2.5, 15.6), FW)
    elif L == "I":
        cv.disk((13.8, 16.2), 6.0, 6.4); knuck(11.0); up("pinky"); thumb_side()
    elif L == "K":
        up("index", -12); up("middle", 14); cv.capsule((11.5, 16.5), (13.2, 12.5), 1.6); palm(); wrist()
    elif L == "L":
        up("index"); thumb_left()
    elif L == "M":
        cv.disk((14.4, 16.2), 6.6, 6.8); [cv.disk((FX[n], 11.8), 1.9, 2.0) for n in ("index", "middle", "ring")]
    elif L == "N":
        cv.disk((14.0, 16.2), 6.4, 6.8); [cv.disk((FX[n], 11.8), 2.0, 2.1) for n in ("index", "middle")]
    elif L == "O":
        cv.ring((14.0, 15.8), 7.8, 8.2, 4.0); wrist()
    elif L == "P":
        cv.disk((13.5, 11.5), 5.6, 5.8); wrist()
        cv.capsule((11.5, 13.5), (10.0, 24.0), FW); cv.capsule((14.0, 13.5), (16.5, 22.0), FW)
    elif L == "Q":
        cv.disk((14.0, 11.5), 5.6, 5.8); wrist()
        cv.capsule((12.5, 13.5), (11.0, 25.0), FW); cv.capsule((15.5, 13.5), (17.5, 23.0), 1.5)
    elif L == "R":
        cv.capsule((FX["index"], YTOP), (13.6, FULL_TIP_Y), FW)
        cv.capsule((FX["middle"], YTOP), (11.9, FULL_TIP_Y), FW); palm(); wrist(); thumb_tuck()
    elif L == "S":
        cv.disk((14.4, 16.0), 6.4, 6.9); knuck(10.8); thumb_front()
    elif L == "T":
        cv.disk((14.4, 16.2), 6.3, 6.8); knuck(11.0); cv.disk((11.8, 11.0), 1.7, 2.0)
    elif L == "U":
        up("index"); up("middle"); thumb_tuck()
    elif L == "V":
        up("index", -13); up("middle", 13); thumb_tuck()
    elif L == "W":
        up("index", -16); up("middle", 0); up("ring", 16); thumb_tuck()
    elif L == "X":
        cv.capsule((FX["index"], YTOP), (FX["index"], 7.0), FW)
        cv.capsule((FX["index"], 7.0), (12.6, 8.0), FW); cv.disk((14.4, 16.0), 5.6, 6.2)
    elif L == "Y":
        thumb_left(); up("pinky", 8)
    img = cv.image()
    # ensure palm/wrist present for letters that build only fingers
    return np.clip(img, 0.0, 1.0)


def resample(img, scale, dx, dy):
    """scale about centre (13.5,13.5) then shift by (dx,dy); bilinear, zero outside."""
    c = 13.5
    ys, xs = np.mgrid[0:28, 0:28].astype(np.float64)
    sx = (xs - c - dx) / scale + c
    sy = (ys - c - dy) / scale + c
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    fx = sx - x0; fy = sy - y0
    out = np.zeros((28, 28), np.float64)
    for oy in (0, 1):
        for ox in (0, 1):
            xi = x0 + ox; yi = y0 + oy
            w = (fx if ox else 1 - fx) * (fy if oy else 1 - fy)
            valid = (xi >= 0) & (xi < 28) & (yi >= 0) & (yi < 28)
            xi_c = np.clip(xi, 0, 27); yi_c = np.clip(yi, 0, 27)
            out += np.where(valid, img[yi_c, xi_c] * w, 0.0)
    return out


LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
SCALES = [0.7, 0.85, 1.0, 1.15, 1.3]
SHIFTS = [-3.0, 0.0, 3.0]
DIM = NCH * GRID * GRID


def tied_vec(tied):
    """(NBASE,GRID,GRID) -> (NCH*GRID*GRID,) with +/- channels tied to the same value."""
    w = np.zeros((NCH, GRID, GRID), np.float64)
    for j in range(NBASE):
        w[2 * j] = tied[j]
        w[2 * j + 1] = tied[j]
    return w.reshape(-1)


def render_core():
    cv = Canvas()
    cv.disk(PALM_C, PALM_RX, PALM_RY)
    cv.capsule((14.0, 23.0), (14.6, 29.5), 4.2)
    for i, n in enumerate(("index", "middle", "ring", "pinky")):
        _finger(cv, n, (i - 1.5) * 7.0, 6.0, FW)
    cv.capsule((8.6, 19.0), (9.3, 13.5), 1.7)
    return np.clip(cv.image(), 0.0, 1.0)


def _norm0(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def build_dense(ker, us):
    variants = [(s, dx, dy) for s in SCALES for dx in SHIFTS for dy in SHIFTS]
    T = len(variants)
    rows = []
    for L in LETTERS:
        base = render(L)
        for (s, dx, dy) in variants:
            im = resample(base, s, dx, dy) if (s != 1.0 or dx or dy) else base
            w = tied_vec(descriptor(im, ker))
            w = w - w.mean()
            for u in us:                       # make orthogonal to common-hand directions
                w = w - (w @ u) * u
            n = np.linalg.norm(w)
            if n > 1e-9:
                w = w / n
            rows.append(w)
    W = np.stack(rows, axis=0)                 # (24*T, DIM)
    b = np.zeros((W.shape[0],), np.float64)
    return W, b, T


def network_json(T):
    return {"layers": [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "feat", "op": "conv", "inputs": ["n0"], "out": NCH, "kernel": [K, K], "stride": 1, "padding": PAD},
        {"name": "n1", "op": "norm", "inputs": ["feat"]},
        {"name": "r1", "op": "relu", "inputs": ["n1"]},
        {"name": "p1", "op": "avgpool", "inputs": ["r1"], "size": [POOL, POOL], "stride": [POOL, POOL]},
        {"name": "match", "op": "dense", "inputs": ["p1"], "out": 24 * T},
        {"name": "rs", "op": "reshape", "inputs": ["match"], "shape": [24, T, 1]},
        {"name": "pick", "op": "maxpool", "inputs": ["rs"], "size": [T, 1], "stride": [T, 1]},
        {"name": "out", "op": "reshape", "inputs": ["pick"], "shape": [24]},
    ]}


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/out"
    os.makedirs(out_dir, exist_ok=True)
    ker = gaussian_kernels()
    cW, cb = conv_weights()
    # common-hand direction made from a single typed generic hand
    us = [_norm0(tied_vec(descriptor(render_core(), ker)))]
    dW, db, T = build_dense(ker, us)
    net = network_json(T)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out_dir, "weights.npz"),
             **{"feat.W": cW.astype(np.float32), "feat.b": cb.astype(np.float32),
                "match.W": dW.astype(np.float32), "match.b": db.astype(np.float32)})
    print("wrote", out_dir, "T=", T, "dense W", dW.shape)


if __name__ == "__main__":
    main()
