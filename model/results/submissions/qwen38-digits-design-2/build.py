"""Self-contained builder: writes network.json + weights.npz for a hand-designed
MNIST classifier. No files are read; every number comes from formulas (Gabor bar
filters, gaussian blurs) and typed pen-path shape models of the digit classes
(stroke specs below), rendered through the same feature pipeline the network uses.
Architecture: conv1 = 8 oriented even-Gabor bar filters (4 orientations x 2 stroke
widths) -> relu -> maxpool2; readout = dense template matching on the pooled
orientation-energy map (14x14x8) and on blurred-ink maps at 14x14 and 7x7; templates
are the pipeline responses of typed digit prototypes drawn over handwriting
variations (stroke width, slant, rotation, per-stroke jitter); final max over shape
variants per class."""
import sys, os, json
import numpy as np

# ==== from strokes.py ====
"""Stroke renderer: draws typed pen paths (lines, arcs, beziers) onto a 28x28 canvas.
All geometry is general-knowledge shape modelling - no data involved."""
import numpy as np


def _brush(canvas, x, y, w, amp=1.0):
    """Stamp a soft round brush of diameter w at (x,y) with additive ink, then clip."""
    r = int(np.ceil(w))
    x0, x1 = int(np.floor(x - r)), int(np.ceil(x + r)) + 1
    y0, y1 = int(np.floor(y - r)), int(np.ceil(y + r)) + 1
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    sig = max(w * 0.5, 0.35)
    g = amp * np.exp(-d2 / (2 * sig * sig))
    H, W = canvas.shape
    ys_c, xs_c = np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)
    np.maximum.at(canvas, (ys_c, xs_c), g)


def _polyline(pts, n=None):
    """Dense resampling of a polyline to ~0.4px steps."""
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    if n is None:
        n = int(max(2, np.ceil(L.sum() / 0.4)))
    t = np.linspace(0, L.sum(), n)
    out, acc = [], 0.0
    cs = np.concatenate([[0], np.cumsum(L)])
    for ti in t:
        i = min(np.searchsorted(cs, ti, "right") - 1, len(L) - 1)
        i = max(i, 0)
        f = (ti - cs[i]) / max(L[i], 1e-9)
        out.append(pts[i] * (1 - f) + pts[i + 1] * f)
    return np.array(out)


def stroke_line(canvas, p0, p1, w, amp=1.0, n=None):
    for (y, x) in _polyline([p0, p1], n):
        _brush(canvas, x, y, w, amp)


def stroke_arc(canvas, cy, cx, ry, rx, a0, a1, w, amp=1.0, n=None):
    """Elliptical arc, angles in degrees, y down. a0->a1 counterclockwise in screen coords means decreasing angle; just sample."""
    ts = np.linspace(np.radians(a0), np.radians(a1), n or max(8, int(abs(a1 - a0) / 3)))
    pts = [(cy + ry * np.sin(t), cx + rx * np.cos(t)) for t in ts]
    for (y, x) in _polyline(pts, n or max(8, int(abs(a1 - a0) / 2))):
        _brush(canvas, x, y, w, amp)


def stroke_bezier(canvas, ctrl, w, amp=1.0, n=None):
    """ctrl: list of control points (y,x). De Casteljau sampling."""
    ctrl = np.asarray(ctrl, float)
    m = len(ctrl) - 1
    n = n or max(16, int(m * 24))
    ts = np.linspace(0, 1, n)
    pts = []
    for t in ts:
        c = ctrl.copy()
        for k in range(m):
            c[:m - k] = c[:m - k] * (1 - t) + c[1:m - k + 1] * t
        pts.append(c[0])
    for (y, x) in _polyline(np.array(pts), n * 2):
        _brush(canvas, x, y, w, amp)


def stroke_poly(canvas, pts, w, amp=1.0, n=None):
    """Smooth path through waypoints: piecewise linear resampled densely (corners slightly rounded by brush)."""
    for (y, x) in _polyline(pts, n):
        _brush(canvas, x, y, w, amp)


def transform(img, scale=1.0, slant=0.0, dy=0.0, dx=0.0, rot=0.0):
    """Resample a 28x28 image: scale about centre, horizontal shear (slant, degrees), translate, small rotation."""
    H = W = img.shape[0]
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    cy = cx = 13.5
    # inverse map
    x = xs - cx - dx
    y = ys - cy - dy
    if rot:
        a = np.radians(rot)
        x, y = np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y
    x = x + np.tan(np.radians(slant)) * y
    x, y = x / scale, y / scale
    x, y = x + cx, y + cy
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    ok = (x0 >= 0) & (x0 < W - 1) & (y0 >= 0) & (y0 < H - 1)
    x0c, y0c = np.clip(x0, 0, W - 2), np.clip(y0, 0, H - 2)
    v = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy) +
         img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
    return np.where(ok, v, 0.0).astype(np.float32)

# ==== from digits.py ====
"""Typed pen-path shape models of the ten digit classes (general knowledge only).
Each variant is a list of stroke specs; the renderer can add small per-stroke jitter
so templates tolerate local shape noise. Coordinates (y, x) on a 28x28 canvas."""
import numpy as np


def stroke_smooth(canvas, wps, w, amp=1.0):
    """Smooth path through waypoints: quadratic beziers joining segment midpoints."""
    wps = np.asarray(wps, float)
    if len(wps) == 2:
        stroke_poly(canvas, wps, w, amp, n=200)
        return
    mids = (wps[:-1] + wps[1:]) / 2
    dense = [np.linspace(wps[0], mids[0], 24)]
    for i in range(len(mids) - 1):
        p0, c, p1 = mids[i], wps[i + 1], mids[i + 1]
        ts = np.linspace(0, 1, 40)[1:]
        q = np.outer((1 - ts) ** 2, p0) + np.outer(2 * (1 - ts) * ts, c) + np.outer(ts ** 2, p1)
        dense.append(q)
    dense.append(np.linspace(mids[-1], wps[-1], 24)[1:])
    path = np.concatenate(dense, 0)
    for (y, x) in _polyline(path, n=int(np.hypot(*np.diff(path, axis=0).T).sum() / 0.35)):
        _brush(canvas, x, y, w, amp)


L = lambda *pts: ("line", list(pts))          # pts: (y,x) pairs
S = lambda *pts: ("smooth", list(pts))
A = lambda cy, cx, ry, rx, a0, a1: ("arc", (cy, cx, ry, rx, a0, a1))

VARIANTS = {
    0: [
        [A(14, 14, 8.2, 7.2, -90, 270)],
        [A(14, 14, 8.2, 6.2, -90, 270)],
        [A(14, 14, 8.4, 4.9, -90, 270)],
    ],
    1: [
        [L((10.8, 9.0), (6.8, 14.0)), L((6.8, 14.0), (22.0, 14.2))],
        [L((10.8, 9.0), (6.8, 14.0)), L((6.8, 14.0), (22.0, 14.2)), L((22.0, 10.2), (22.0, 18.2))],
        [L((7.0, 14.0), (22.0, 14.0))],
        [L((11.5, 11.0), (7.0, 14.2)), L((7.0, 14.2), (22.0, 14.0))],
    ],
    2: [
        [S((10.4, 9.2), (7.9, 13.3), (11.4, 18.0), (15.0, 13.8), (18.4, 8.5),
           (18.7, 13.6), (18.7, 18.6), (13.6, 19.0), (8.6, 18.8))],
        [S((10.4, 9.2), (7.9, 13.3), (11.4, 18.0), (15.6, 14.2), (19.0, 9.6),
           (20.6, 14.0), (18.2, 19.6), (13.8, 19.2), (12.8, 16.6), (16.0, 16.0), (20.0, 18.2))],
    ],
    3: [
        [S((9.7, 10.9), (8.6, 15.0), (11.2, 17.1), (13.4, 16.4), (13.7, 13.3),
           (14.8, 16.4), (17.8, 17.1), (19.9, 13.8), (19.2, 10.9))],
        [S((8.4, 10.2), (8.5, 16.6), (11.8, 14.6), (13.4, 13.6), (15.6, 15.6),
           (17.6, 18.4), (15.6, 21.0), (11.6, 20.6), (9.6, 18.6))],
        [S((9.9, 11.6), (9.0, 14.8), (11.2, 16.5), (13.0, 15.8), (13.2, 13.4),
           (14.2, 15.8), (16.6, 16.5), (18.3, 13.6), (17.7, 11.2))],
    ],
    4: [
        [L((17.6, 7.0), (6.0, 15.8)), L((17.6, 5.8), (17.6, 20.6)), L((5.8, 17.6), (21.2, 17.6))],
        [L((17.4, 8.6), (10.8, 17.4)), L((17.0, 5.5), (17.0, 22.5)), L((7.0, 17.6), (21.5, 17.6))],
    ],
    5: [
        [L((7.8, 15.8), (8.4, 10.1)), L((8.4, 10.1), (12.6, 10.3)),
         S((12.6, 10.3), (13.0, 14.9), (14.6, 17.8), (17.2, 17.6), (19.3, 14.2), (18.0, 9.9))],
        [L((7.6, 16.6), (8.2, 9.6)), L((8.2, 9.6), (12.8, 9.8)),
         S((12.8, 9.8), (13.4, 15.0), (15.2, 18.4), (18.0, 18.0), (20.0, 14.2), (18.6, 9.6))],
    ],
    6: [
        [S((6.1, 15.6), (8.5, 10.6), (12.5, 8.2), (16.6, 8.4)), A(17.3, 11.6, 4.1, 3.9, 170, 530)],
        [S((6.1, 16.4), (8.5, 10.6), (12.5, 8.0), (16.8, 8.4)), A(17.3, 11.8, 4.2, 4.5, 170, 530)],
        [S((6.4, 14.8), (8.6, 10.4), (12.4, 8.2), (16.2, 8.6)), A(17.2, 11.4, 4.0, 3.4, 170, 490)],
    ],
    7: [
        [L((7.3, 8.3), (7.3, 19.7)), L((7.6, 19.1), (22.0, 11.3))],
        [L((7.3, 8.3), (7.3, 19.7)), S((7.6, 19.1), (14.5, 16.3), (22.0, 11.3)), L((15.0, 11.3), (14.3, 17.3))],
        [L((7.3, 8.3), (7.3, 19.7)), S((7.6, 19.1), (14.5, 16.3), (22.0, 11.3))],
    ],
    8: [
        [A(10.1, 14, 3.7, 4.6, -90, 270), A(17.9, 14, 4.1, 5.5, -90, 270)],
        [S((16.2, 9.4), (11.6, 7.6), (9.4, 10.0), (10.9, 12.9), (14.0, 14.0), (17.2, 15.6),
           (18.4, 18.8), (15.2, 21.6), (11.0, 20.9), (9.2, 18.0), (11.4, 15.2), (14.4, 14.1))],
        [A(10.1, 14, 3.9, 5.1, -90, 270), A(17.9, 14, 4.3, 6.0, -90, 270)],
    ],
    9: [
        [A(9.4, 11.5, 3.9, 3.6, -90, 270), S((10.3, 15.1), (15.5, 14.9), (19.9, 14.0), (21.6, 10.4))],
        [A(9.4, 11.5, 3.9, 3.6, -90, 270), S((10.3, 15.1), (16.0, 15.0), (22.0, 14.6))],
        [A(9.4, 11.2, 3.8, 3.1, -90, 270), S((10.2, 14.4), (15.4, 14.4), (19.8, 13.6), (21.5, 10.0))],
    ],
}


def draw_specs(specs, w, rng=None, jit=0.0):
    c = np.zeros((28, 28), np.float32)
    for sp in specs:
        if sp[0] == "line":
            pts = np.array(sp[1], float)
            if rng is not None and jit:
                pts = pts + rng.normal(0, jit, pts.shape)
            stroke_line(c, tuple(pts[0]), tuple(pts[1]), w)
        elif sp[0] == "smooth":
            pts = np.array(sp[1], float)
            if rng is not None and jit:
                pts = pts + rng.normal(0, jit, pts.shape)
            stroke_smooth(c, pts, w)
        else:
            cy, cx, ry, rx, a0, a1 = sp[1]
            if rng is not None and jit:
                cy, cx = cy + rng.normal(0, jit), cx + rng.normal(0, jit)
                ry, rx = ry * (1 + rng.normal(0, jit * 0.06)), rx * (1 + rng.normal(0, jit * 0.06))
                a0, a1 = a0 + rng.normal(0, jit * 4), a1 + rng.normal(0, jit * 4)
            stroke_arc(c, cy, cx, ry, rx, a0, a1, w)
    return c


def normalize20(img):
    """Mimic the dataset geometry: scale ink bbox so its largest side is 20 px
    (aspect kept), then move the centre of mass to the frame centre."""
    ink = img > 0.03
    if not ink.any():
        return img
    ys, xs = np.nonzero(ink)
    cy0, cx0 = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    s = 20.0 / max(h, w)
    c = transform(img, dy=14.0 - cy0, dx=14.0 - cx0)
    c = transform(c, scale=s)
    for _ in range(2):
        ink = c > 0.03
        ys, xs = np.nonzero(ink)
        h2, w2 = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        cy2, cx2 = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
        c = transform(c, dy=14.0 - cy2, dx=14.0 - cx2)
        c = transform(c, scale=20.0 / max(h2, w2))
        m = c.sum()
        yy, xx = np.mgrid[0:28, 0:28].astype(float)
        c = transform(c, dy=14.0 - (c * yy).sum() / m, dx=14.0 - (c * xx).sum() / m)
    return c


def render_variant(cls, vi, width=2.5, scale=1.0, slant=0.0, rot=0.0, dy=0.0, dx=0.0,
                   rng=None, jit=0.0):
    c = draw_specs(VARIANTS[cls][vi], width, rng, jit)
    c = normalize20(c)
    return transform(c, scale=scale, slant=slant, dy=dy, dx=dx, rot=rot)


    ims = []
    for d in range(10):
        for vi in range(len(VARIANTS[d])):
            ims.append(render_variant(d, vi))
    m = montage([upscale(im, 6) for im in ims], 17)
    write_png_gray("/app/work/protos.png", m, invert=True)
    print("saved", len(ims), "prototypes")

# ==== from pipeline.py ====
"""Feature pipeline: conv1 (oriented bar filters from formulas) -> relu -> maxpool ->
conv2 (structural primitives matched on conv1 maps) -> relu -> maxpool.
Simulator mirrors /app/blocks.py exactly."""
import numpy as np


def conv_np(x, Wt, b, stride=1, padding=0):
    """x: (C,H,W) single image. Wt: (O,C,k,k)."""
    c, h, w = x.shape
    o, _, kh, kw = Wt.shape
    xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=1)
    return (cols.reshape(c * kh * kw, ho * wo).T @ Wt.reshape(o, -1).T).T.reshape(o, ho, wo) + b[:, None, None]


def maxpool_np(x, size=2, stride=2):
    c, h, w = x.shape
    ho, wo = (h - size) // stride + 1, (w - size) // stride + 1
    win = np.stack([x[:, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(size) for dx in range(size)], 0)
    return win.max(0)


# ---------------- conv1: oriented even-Gabor bar filters, 2 width scales ----------------
K1 = 11

def gabor_bar(theta_deg, sig_t, sig_a, k=K1):
    """Even Gabor: bright bar along direction theta (degrees, 0 = horizontal).
    Positive central stripe, negative flanks; modulo-180 symmetric."""
    th = np.radians(theta_deg)
    yy, xx = np.mgrid[0:k, 0:k].astype(float)
    yy -= (k - 1) / 2.0; xx -= (k - 1) / 2.0
    t = xx * np.cos(th) + yy * np.sin(th)   # along stroke
    a = -xx * np.sin(th) + yy * np.cos(th)  # across stroke
    env = np.exp(-t * t / (2 * sig_t ** 2))
    car = np.exp(-a * a / (2 * sig_a ** 2)) - 0.42 * np.exp(-a * a / (2 * (2.1 * sig_a) ** 2))
    f = env * car
    f -= f.mean()
    return (f / np.abs(f).max() * 1.6).astype(np.float32)

ORIENTS = [0, 45, 90, 135]

def conv1_filters():
    """8 channels: 4 orientations x 2 scales (thin ~2px, thick ~3.5px strokes)."""
    fs = []
    for th in ORIENTS:
        fs.append(gabor_bar(th, 3.6, 1.05))
    for th in ORIENTS:
        fs.append(gabor_bar(th, 3.6, 1.9))
    Wt = np.stack(fs)[:, None, :, :]
    return Wt.astype(np.float32), np.zeros(Wt.shape[0], np.float32)

W1, B1 = conv1_filters()


def feat_fine(img):
    """conv1+relu+maxpool2 -> (8,14,14). img: (28,28) or (1,28,28)."""
    x = img.astype(np.float32)
    if x.ndim == 2:
        x = x[None]
    return maxpool_np(np.maximum(conv_np(x, W1, B1, 1, 5), 0))


    from png import write_png_gray, montage, upscale
    m = montage([upscale((f - f.min()) / (f.max() - f.min() + 1e-9), 6) for f in W1[:, 0]], 4)
    write_png_gray("/app/work/conv1.png", m, invert=True)
    print("conv1 filters saved; W1", W1.shape)

# ==== from prims.py ====
"""Layer-2 structural detectors: matched filters built by rendering typed stroke
primitives (lines, arcs, corners, junctions, ends, loops) and passing them through
the layer-1 pipeline. General-knowledge geometry only."""
import numpy as np

S = 24  # primitive canvas


def _c():
    return np.zeros((S, S), np.float32)


def prim_list():
    """Each entry: (tag, draw(canvas, w))."""
    P = []
    c0 = S / 2
    # straight lines, 4 orientations
    for th in [0, 45, 90, 135]:
        a = np.radians(th)
        dy, dx = np.sin(a) * 10, np.cos(a) * 10
        P.append((f"line{th}", lambda c, w, dy=dy, dx=dx: stroke_line(c, (c0 - dy, c0 - dx), (c0 + dy, c0 + dx), w)))
    # arcs turning +-60 deg over the window, 4 base orientations
    for th in [0, 45, 90, 135]:
        for sgn in (+1, -1):
            def dr(c, w, th=th, sgn=sgn):
                a = np.radians(th)
                # circle centre offset perpendicular to stroke start direction
                r = 9.0
                cy, cx = c0 - sgn * r * np.cos(a), c0 + sgn * r * np.sin(a)
                # start angle: point (c0,c0) relative to centre
                t0 = np.degrees(np.arctan2(c0 - cy, c0 - cx))
                stroke_arc(c, cy, cx, r, r, t0, t0 + sgn * 62, w)
            P.append((f"arc{th}{'p' if sgn>0 else 'm'}", dr))
    # sharp corners (90 deg turns), 4 kinds
    for th in [0, 45, 90, 135]:
        def dr(c, w, th=th):
            a = np.radians(th)
            p0 = (c0 - 9 * np.sin(a), c0 - 9 * np.cos(a))
            p1 = (c0, c0)
            p2 = (c0 - 9 * np.cos(a), c0 + 9 * np.sin(a))
            stroke_poly(c, [p0, p1, p2], w, n=200)
        P.append((f"corner{th}", dr))
    # stroke endings at centre, 4 orientations
    for th in [0, 45, 90, 135]:
        def dr(c, w, th=th):
            a = np.radians(th)
            dy, dx = np.sin(a) * 10, np.cos(a) * 10
            stroke_line(c, (c0 - dy, c0 - dx), (c0, c0), w)
        P.append((f"end{th}", dr))
    # T junctions, 4 rotations
    for th in [0, 45, 90, 135]:
        def dr(c, w, th=th):
            a = np.radians(th)
            dy, dx = np.sin(a) * 10, np.cos(a) * 10
            stroke_line(c, (c0 - dy, c0 - dx), (c0 + dy, c0 + dx), w)
            stroke_line(c, (c0, c0), (c0 - np.cos(a) * 9, c0 + np.sin(a) * 9), w)
        P.append((f"tee{th}", dr))
    # crossings
    def drx(c, w):
        stroke_line(c, (c0 - 7, c0 - 7), (c0 + 7, c0 + 7), w)
        stroke_line(c, (c0 - 7, c0 + 7), (c0 + 7, c0 - 7), w)
    P.append(("cross_x", drx))

    def drp(c, w):
        stroke_line(c, (c0, c0 - 9), (c0, c0 + 9), w)
        stroke_line(c, (c0 - 9, c0), (c0 + 9, c0), w)
    P.append(("cross_p", drp))
    # small closed loop (like the bowl of 6/9/8)
    P.append(("loop", lambda c, w: stroke_arc(c, c0, c0, 4.4, 4.4, -90, 270, w)))
    # gentle curve (30 deg turn)
    for th in [0, 45, 90, 135]:
        def dr(c, w, th=th):
            a = np.radians(th)
            r = 17.0
            cy, cx = c0 - r * np.cos(a), c0 + r * np.sin(a)
            t0 = np.degrees(np.arctan2(c0 - cy, c0 - cx))
            stroke_arc(c, cy, cx, r, r, t0, t0 + 32, w)
        P.append((f"curve{th}", dr))
    return P


def conv2_filters(width=2.4):
    P = prim_list()
    K = []
    tags = []
    for tag, dr in P:
        c = _c()
        dr(c, width)
        f = feat_fine(c)            # (8,12,12)
        k = f[:, 3:8, 3:8] if False else f[:, (12 - 5) // 2:(12 - 5) // 2 + 5, (12 - 5) // 2:(12 - 5) // 2 + 5]
        n = np.linalg.norm(k)
        if n < 1e-6:
            k = np.zeros_like(k); n = 1.0
        K.append(k / n)
        tags.append(tag)
    return np.stack(K).astype(np.float32), np.zeros(len(K), np.float32), tags


    W2, B2, tags = conv2_filters()
    print("conv2:", W2.shape, tags)
    ims = []
    for tag, dr in prim_list():
        c = _c(); dr(c, 2.4)
        ims.append(c)
    write_png_gray("/app/work/prims.png", montage([upscale(i, 5) for i in ims], 10), invert=True)

# ==== from build_core.py ====
"""Builds network.json + weights.npz for a hand-designed MNIST classifier.
Architecture:
  conv1  8 oriented even-Gabor bar filters (4 orientations x 2 stroke widths), 11x11
  relu, maxpool 2
  conv2  31 structural detectors (lines/arcs/corners/ends/junctions/crossings/loop),
         matched filters on conv1 maps, built by rendering typed stroke primitives
  relu, maxpool 2
  readout: two dense layers (fine 14x14x8 orientation map, coarse 7x7x31 structural
  map) whose rows are blurred, normalised feature maps of typed digit pen-path
  prototypes rendered through the same pipeline; max over shape variants per class.
No data is read anywhere: every number comes from formulas and typed shape models.
"""
import numpy as np

W2, B2, TAGS = conv2_filters()
FINE_BLUR = 1.2
INK14_BLUR = 1.0
F2 = W2.shape[0]

# ---------------- readout templates ----------------
# (width, scale, slant, rot, dy, dx) handwriting variations per prototype.
# Geometry (20px box, centred mass) is imposed by normalize20, so scale stays 1.
VARS = [
    (1.6, 1.0, 0, 0, 0, 0), (2.1, 1.0, 0, 0, 0, 0), (2.6, 1.0, 0, 0, 0, 0), (3.1, 1.0, 0, 0, 0, 0),
    (2.1, 1.0, -10, 0, 0, 0), (2.1, 1.0, 10, 0, 0, 0),
    (2.6, 1.0, -10, 0, 0, 0), (2.6, 1.0, 10, 0, 0, 0),
    (1.6, 1.0, -10, -4, 0, 0), (3.1, 1.0, 10, 4, 0, 0),
    (2.1, 1.0, 0, 5, 0, 0), (2.1, 1.0, 0, -5, 0, 0),
    (2.4, 1.0, 4, -3, 0, 0), (2.4, 1.0, -4, 3, 0, 0),
    (1.8, 1.0, -5, 2, 0, 0), (2.9, 1.0, 5, -2, 0, 0),
    (2.2, 1.0, -12, 0, 0, 0), (2.2, 1.0, 12, 0, 0, 0),
    (2.8, 1.0, 0, 6, 0, 0), (1.7, 1.0, 0, -6, 0, 0),
]

def gauss_blur(a, sig):
    """3x3 separable gaussian blur on last two axes of (C,H,W)."""
    k = np.exp(-np.arange(-1, 2) ** 2 / (2 * sig * sig)); k /= k.sum()
    ap = np.pad(a, ((0, 0), (1, 1), (0, 0)), mode="constant")
    out = np.zeros_like(a)
    for i in range(3):
        out += k[i] * ap[:, i:i + a.shape[1], :]
    ap = np.pad(out, ((0, 0), (0, 0), (1, 1)), mode="constant")
    out2 = np.zeros_like(a)
    for i in range(3):
        out2 += k[i] * ap[:, :, i:i + a.shape[2]]
    return out2


def blur2d(a, sig):
    """2D gaussian blur on (H,W) with 5-tap kernel."""
    t = np.arange(-2, 3)
    k = np.exp(-t ** 2 / (2 * sig * sig)); k /= k.sum()
    ap = np.pad(a, ((2, 2), (0, 0)), mode="constant")
    out = sum(k[i] * ap[i:i + a.shape[0], :] for i in range(5))
    ap = np.pad(out, ((0, 0), (2, 2)), mode="constant")
    return sum(k[i] * ap[:, i:i + a.shape[1]] for i in range(5))


def avgpool2(a):
    h, w = a.shape
    return (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) / 4.0


def variant_features(cls, vi):
    """Mean fine (8,14,14), coarse (F2,7,7) and blurred-ink (14,14)+(7,7) maps
    over handwriting variations."""
    fine = np.zeros((8, 14, 14)); coarse = np.zeros((F2, 7, 7))
    ink14 = np.zeros((14, 14)); ink7 = np.zeros((7, 7)); ink28 = np.zeros((28, 28))
    rng = np.random.RandomState(1000 + 17 * cls + vi)
    for (w, sc, sl, ro, dy, dx) in VARS:
        img = render_variant(cls, vi, width=w, scale=sc, slant=sl, rot=ro, dy=dy, dx=dx,
                             rng=rng, jit=0.6)
        f = feat_fine(img)
        c = maxpool_np(np.maximum(conv_np(f, W2, B2, 1, 2), 0))
        fine += f; coarse += c
        b = blur2d(img, 1.4)
        ink14 += maxpool_np(b[None])[0]
        ink7 += avgpool2(maxpool_np(blur2d(img, 1.8)[None])[0])[:7, :7]
        ink28 += blur2d(img, 1.2)
    n = len(VARS)
    return fine / n, coarse / n, ink14 / n, ink7 / n, ink28 / n


def build(out_dir):
    P = max(len(v) for v in VARIANTS.values())
    ALPHA, BETA, G1, G2, G3 = 0.55, 0.0, 0.50, 0.42, 0.0
    rows_f = np.zeros((10 * P, 8 * 14 * 14), np.float64)
    rows_c = np.zeros((10 * P, F2 * 7 * 7), np.float64)
    rows_i = np.zeros((10 * P, 14 * 14), np.float64)
    rows_7 = np.zeros((10 * P, 7 * 7), np.float64)
    rows_28 = np.zeros((10 * P, 28 * 28), np.float64)
    for cls in range(10):
        nvar = len(VARIANTS[cls])
        for p in range(P):
            vi = min(p, nvar - 1)
            f, c, i14, i7, i28 = variant_features(cls, vi)
            f = gauss_blur(f, FINE_BLUR).ravel()
            c = gauss_blur(c, 0.8).ravel()
            i14 = blur2d(i14, INK14_BLUR).ravel()
            i7 = blur2d(i7, 0.7).ravel()
            i28 = blur2d(i28, 0.5).ravel()
            for row, vec, g in ((rows_f, f, ALPHA), (rows_c, c, BETA),
                                (rows_i, i14, G1), (rows_7, i7, G2),
                                (rows_28, i28, G3)):
                n = np.linalg.norm(vec)
                row[cls * P + p] = g * vec / max(n, 1e-9)

    def gkernel(sig):
        t = np.arange(-2, 3)
        k = np.exp(-t ** 2 / (2 * sig * sig)); k /= k.sum()
        return np.outer(k, k).astype(np.float32)

    net = {"layers": [
        {"name": "c1", "op": "conv", "inputs": ["image"], "out": 8, "kernel": [11, 11], "stride": 1, "padding": 5},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "p1", "op": "maxpool", "inputs": ["r1"], "size": [2, 2], "stride": [2, 2]},
        {"name": "c2", "op": "conv", "inputs": ["p1"], "out": F2, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "r2", "op": "relu", "inputs": ["c2"]},
        {"name": "p2", "op": "maxpool", "inputs": ["r2"], "size": [2, 2], "stride": [2, 2]},
        {"name": "bi", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "pi", "op": "maxpool", "inputs": ["bi"], "size": [2, 2], "stride": [2, 2]},
        {"name": "b7", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "p7a", "op": "maxpool", "inputs": ["b7"], "size": [2, 2], "stride": [2, 2]},
        {"name": "p7", "op": "avgpool", "inputs": ["p7a"], "size": [2, 2], "stride": [2, 2]},
        {"name": "rf", "op": "reshape", "inputs": ["p1"], "shape": [8 * 14 * 14]},
        {"name": "df", "op": "dense", "inputs": ["rf"], "out": 10 * P},
        {"name": "rc", "op": "reshape", "inputs": ["p2"], "shape": [F2 * 7 * 7]},
        {"name": "dc", "op": "dense", "inputs": ["rc"], "out": 10 * P},
        {"name": "r28", "op": "reshape", "inputs": ["image"], "shape": [28 * 28]},
        {"name": "d28", "op": "dense", "inputs": ["r28"], "out": 10 * P},
        {"name": "ri", "op": "reshape", "inputs": ["pi"], "shape": [14 * 14]},
        {"name": "di", "op": "dense", "inputs": ["ri"], "out": 10 * P},
        {"name": "r7", "op": "reshape", "inputs": ["p7"], "shape": [7 * 7]},
        {"name": "d7", "op": "dense", "inputs": ["r7"], "out": 10 * P},
        {"name": "sc", "op": "add", "inputs": ["df", "dc", "di", "d7", "d28"]},
        {"name": "rs", "op": "reshape", "inputs": ["sc"], "shape": [10, P, 1]},
        {"name": "mp", "op": "maxpool", "inputs": ["rs"], "size": [P, 1], "stride": [P, 1]},
        {"name": "out", "op": "reshape", "inputs": ["mp"], "shape": [10]},
    ]}
    weights = {
        "c1.W": W1.astype(np.float32), "c1.b": B1.astype(np.float32),
        "c2.W": W2.astype(np.float32), "c2.b": B2.astype(np.float32),
        "bi.W": gkernel(1.4)[None, None], "bi.b": np.zeros(1, np.float32),
        "b7.W": gkernel(1.8)[None, None], "b7.b": np.zeros(1, np.float32),
        "df.W": rows_f.astype(np.float32), "df.b": np.zeros(10 * P, np.float32),
        "dc.W": rows_c.astype(np.float32), "dc.b": np.zeros(10 * P, np.float32),
        "di.W": rows_i.astype(np.float32), "di.b": np.zeros(10 * P, np.float32),
        "d7.W": rows_7.astype(np.float32), "d7.b": np.zeros(10 * P, np.float32),
        "d28.W": rows_28.astype(np.float32), "d28.b": np.zeros(10 * P, np.float32),
    }
    os.makedirs(out_dir, exist_ok=True)
    json.dump(net, open(os.path.join(out_dir, "network.json"), "w"))
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)
    tot = sum(v.size for v in weights.values())
    print(f"wrote {out_dir}: {tot} weights, P={P}, F2={F2}")

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
