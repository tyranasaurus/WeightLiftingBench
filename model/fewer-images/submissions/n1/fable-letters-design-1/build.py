"""build.py - writes network.json and weights.npz for the sign-language letter classifier.

Everything here comes from the code itself: no file is read.  The weights are
  (1) fixed formula filters: Sobel derivatives steered to 8 orientations, an orientation-sharpening 1x1 mix,
      Gaussian pooling, and a local contrast normalisation built from a dense "gather", the norm op and a
      strided max-pool;
  (2) a bank of matched filters.  Each one is the feature map of ONE image drawn by the small 3-D hand model
      below (palm slab, finger capsules, thumb, forearm) posed as one ASL letter from joint angles typed from
      general knowledge of the manual alphabet, seen from one randomly drawn (fixed seed) viewpoint / size /
      lighting.  Each template is normalised by itself alone (its own mean removed, its own norm divided out).
The class score is the best match over that class's templates and over small shifts (max-pooling).
"""
import sys, json, os
import numpy as np

D2R = np.pi / 180.0


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.sqrt((v * v).sum())


# ---------------------------------------------------------------------------------------------------------------
# Anatomy, in units of hand length (wrist crease to tip of the middle finger = 1).
# Hand frame: origin at the wrist centre, +y towards the fingers, +x towards the thumb, +z out of the palm.
# For a right hand held palm-forward, a viewer facing the signer sees +x on their right.
# ---------------------------------------------------------------------------------------------------------------
FNAMES = ("index", "middle", "ring", "pinky")
FINGER = {
    "index": dict(mcp=(0.148, 0.500, 0.0), L=(0.215, 0.125, 0.070), r=(0.051, 0.047, 0.043, 0.039)),
    "middle": dict(mcp=(0.050, 0.515, 0.0), L=(0.240, 0.140, 0.075), r=(0.052, 0.048, 0.044, 0.040)),
    "ring": dict(mcp=(-0.047, 0.490, 0.0), L=(0.220, 0.135, 0.075), r=(0.049, 0.045, 0.041, 0.037)),
    "pinky": dict(mcp=(-0.138, 0.435, 0.0), L=(0.175, 0.100, 0.065), r=(0.043, 0.039, 0.036, 0.033)),
}
THUMB_CMC = np.array([0.085, 0.09, 0.02])
THUMB_L = (0.24, 0.165, 0.095)
THUMB_R = (0.066, 0.055, 0.048, 0.043)
PALM_POLY = np.array([(-0.105, 0.03), (0.095, 0.03), (0.125, 0.25), (0.150, 0.50), (0.050, 0.515),
                      (-0.050, 0.49), (-0.145, 0.435), (-0.150, 0.20)])
SPACING = 0.02


def capsule(p0, p1, r0, r1, spacing=SPACING):
    p0 = np.asarray(p0, dtype=np.float64); p1 = np.asarray(p1, dtype=np.float64)
    n = int(np.ceil(np.sqrt(((p1 - p0) ** 2).sum()) / spacing)) + 1
    t = np.linspace(0.0, 1.0, n)[:, None]
    return p0[None, :] * (1 - t) + p1[None, :] * t, r0 * (1 - t[:, 0]) + r1 * t[:, 0]


def _inside(poly, x, y):
    inside = np.zeros(x.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]; x1, y1 = poly[(i + 1) % n]
        cond = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0)
        inside ^= cond
    return inside


def _palm():
    step = 0.028
    xs = np.arange(-0.16, 0.16, step); ys = np.arange(0.03, 0.53, step)
    X, Y = np.meshgrid(xs, ys)
    m = _inside(PALM_POLY, X, Y)
    pts = [np.stack([X[m], Y[m]], axis=1)]
    n = len(PALM_POLY)
    for i in range(n):
        a, b = PALM_POLY[i], PALM_POLY[(i + 1) % n]
        k = int(np.ceil(np.sqrt(((b - a) ** 2).sum()) / step)) + 1
        t = np.linspace(0, 1, k)[:, None]
        pts.append(a[None] * (1 - t) + b[None] * t)
    p = np.concatenate(pts, axis=0)
    # smooth thenar and hypothenar domes on the palmar side
    z = 0.035 * np.exp(-((p[:, 0] - 0.07) ** 2 + (p[:, 1] - 0.15) ** 2) / 0.012) + 0.022 * np.exp(-((p[:, 0] + 0.10) ** 2 + (p[:, 1] - 0.16) ** 2) / 0.012)
    c = np.concatenate([p, z[:, None]], axis=1)
    r = 0.076 - 0.024 * np.clip(p[:, 1] / 0.5, 0, 1)
    return c, r


PALM_C, PALM_R = _palm()


def finger_points(name, flex, abd):
    """flex = (mcp, pip, dip) flexion in degrees; abd in degrees (positive leans towards the thumb side)."""
    g = FINGER[name]
    a = abd * D2R
    b = np.array([np.sin(a), np.cos(a), 0.0]); nz = np.array([0.0, 0.0, 1.0])
    pts = [np.array(g["mcp"], dtype=np.float64)]
    phi = 0.0
    for L, f in zip(g["L"], flex):
        phi += f * D2R
        pts.append(pts[-1] + L * (np.cos(phi) * b + np.sin(phi) * nz))
    return pts


def thumb_points(spec):
    """spec: {'dirs': (d_mc, d_pp, d_dp)} or {'mc': d_mc, 'target': tip position, 'hint': bend direction}."""
    c0 = THUMB_CMC
    if "dirs" in spec:
        d = [unit(v) for v in spec["dirs"]]
        p1 = c0 + THUMB_L[0] * d[0]; p2 = p1 + THUMB_L[1] * d[1]; p3 = p2 + THUMB_L[2] * d[2]
        return [c0, p1, p2, p3]
    m = c0 + THUMB_L[0] * unit(spec["mc"])
    tgt = np.asarray(spec["target"], dtype=np.float64)
    dv = tgt - m; dist = np.sqrt((dv * dv).sum()); dh = dv / dist
    l1, l2 = THUMB_L[1], THUMB_L[2]
    if dist >= 0.985 * (l1 + l2):
        return [c0, m, m + l1 * dh, m + (l1 + l2) * dh]
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = np.sqrt(max(l1 * l1 - a * a, 0.0))
    hint = np.asarray(spec.get("hint", (0.3, 0.2, 1.0)), dtype=np.float64)
    hp = hint - (hint * dh).sum() * dh
    hp = hp / np.sqrt((hp * hp).sum())
    return [c0, m, m + a * dh + h * hp, tgt]


def hand_spheres(pose):
    """pose: dict with 'fingers' {name: (mcp, pip, dip, abd)}, 'thumb' spec, 'wrist' (flex, dev) in degrees.
    Returns centres (N,3), radii (N,), is_forearm (N,) in the hand frame."""
    C, R = [PALM_C], [PALM_R]
    for name in FNAMES:
        f = pose["fingers"][name]
        pts = finger_points(name, f[:3], f[3])
        rr = FINGER[name]["r"]
        for i in range(3):
            c, r = capsule(pts[i], pts[i + 1], rr[i], rr[i + 1]); C.append(c); R.append(r)
    tp = thumb_points(pose["thumb"])
    for i in range(3):
        c, r = capsule(tp[i], tp[i + 1], THUMB_R[i], THUMB_R[i + 1]); C.append(c); R.append(r)
    for t in np.linspace(0.12, 1.0, 7):  # web / thenar mass between the thumb metacarpal and the palm edge
        c, r = capsule((0.105 + 0.035 * t, 0.08 + 0.36 * t, 0.012), tp[0] + t * (tp[1] - tp[0]), 0.058 - 0.012 * t, 0.060 - 0.008 * t)
        C.append(c); R.append(r)
    nh = sum(len(r) for r in R)
    fl, dv = pose.get("wrist", (0.0, 0.0))
    f = unit([np.sin(dv * D2R), -np.cos(dv * D2R) * np.cos(fl * D2R), np.sin(fl * D2R)])
    w = np.array([1.0, 0.0, 0.0]); w = unit(w - (w * f).sum() * f)
    start = np.array([-0.005, 0.04, 0.0])
    t = np.linspace(0.0, 1.0, 34)
    for o, r0 in ((-0.078, 0.070), (-0.039, 0.088), (0.0, 0.094), (0.039, 0.088), (0.078, 0.070)):
        c = start[None, :] + (o * (1 + 0.55 * t))[:, None] * w[None, :] + (1.5 * t)[:, None] * f[None, :]
        C.append(c); R.append(r0 * (1 + 0.45 * t))
    C = np.concatenate(C, axis=0); R = np.concatenate(R, axis=0)
    fore = np.arange(len(R)) >= nh
    return C, R, fore


def render(C, Rad, scale, ox, oy, light, amb, albedo, bg, n=28, ss=3):
    """Orthographic z-buffer render of spheres. Image x = ox + scale*X, y = oy - scale*Y (pixels)."""
    S = n * ss
    zbuf = np.full((S, S), -1e9); sh = np.zeros((S, S)); hit = np.zeros((S, S), dtype=bool)
    px = (ox + scale * C[:, 0]) * ss - 0.5
    py = (oy - scale * C[:, 1]) * ss - 0.5
    pz = scale * C[:, 2] * ss
    pr = scale * Rad * ss
    lx, ly, lz = light
    x0s = np.maximum(np.ceil(px - pr), 0).astype(int); x1s = np.minimum(np.floor(px + pr), S - 1).astype(int)
    y0s = np.maximum(np.ceil(py - pr), 0).astype(int); y1s = np.minimum(np.floor(py + pr), S - 1).astype(int)
    grid = np.arange(S, dtype=np.float64)
    for i in range(len(pr)):
        x0, x1, y0, y1 = x0s[i], x1s[i], y0s[i], y1s[i]
        if x1 < x0 or y1 < y0:
            continue
        r = pr[i]
        dx = grid[x0:x1 + 1] - px[i]; dy = grid[y0:y1 + 1] - py[i]
        h2 = r * r - (dx * dx)[None, :] - (dy * dy)[:, None]
        h = np.sqrt(np.maximum(h2, 0.0))
        z = pz[i] + h
        zb = zbuf[y0:y1 + 1, x0:x1 + 1]
        win = (h2 > 0) & (z > zb)
        if not win.any():
            continue
        s = (lx * dx[None, :] - ly * dy[:, None] + lz * h) / r
        zb[win] = z[win]
        sh[y0:y1 + 1, x0:x1 + 1][win] = s[win]
        hit[y0:y1 + 1, x0:x1 + 1][win] = True
    img = np.where(hit, albedo * (amb + (1 - amb) * np.maximum(sh, 0.0)), bg)
    return img.reshape(n, ss, n, ss).mean(axis=(1, 3))



LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"


def tip_of(name, f):
    return finger_points(name, f[:3], f[3])[3]


def base_pose(ch):
    """Canonical pose of one letter: fingers, thumb, view (yaw, pitch, roll in degrees), wrist (flex, dev)."""
    cu = dict(index=(88, 100, 65, 0), middle=(88, 100, 65, 0), ring=(88, 100, 65, -2), pinky=(88, 100, 65, -4))
    ex = dict(index=(0, 0, 0, -1), middle=(0, 0, 0, 0), ring=(0, 0, 0, 1), pinky=(0, 0, 0, 3))
    fingers = dict(cu)
    view = (0.0, 0.0, 0.0)
    wrist = (0.0, 0.0)
    th_across = dict(mc=(0.45, 0.45, 0.77), target=(0.03, 0.40, 0.31))          # thumb over the front of a fist
    th_hold = dict(mc=(0.42, 0.45, 0.79), target=(-0.05, 0.38, 0.27))            # thumb holding ring / little finger
    thumb = th_across
    if ch == "A":
        fingers = {k: (85, 100, 60, v[3]) for k, v in cu.items()}
        thumb = dict(dirs=((0.55, 0.75, 0.35), (0.05, 1.0, 0.25), (-0.05, 1.0, 0.10)))
    elif ch == "B":
        fingers = dict(ex)
        thumb = dict(mc=(0.40, 0.55, 0.70), target=(-0.02, 0.36, 0.09))
    elif ch == "C":
        fingers = dict(index=(35, 45, 30, 2), middle=(35, 45, 30, 0), ring=(35, 45, 30, -2), pinky=(38, 45, 30, -5))
        thumb = dict(dirs=((0.40, 0.35, 0.85), (0.20, 0.30, 0.93), (0.0, 0.45, 0.90)))
        view = (75.0, 0.0, -10.0)
    elif ch == "D":
        fingers = dict(index=(0, 0, 0, 2), middle=(62, 62, 40, 0), ring=(62, 62, 40, -2), pinky=(65, 62, 40, -4))
        thumb = dict(mc=(0.40, 0.35, 0.85), target=tip_of("middle", fingers["middle"]) + np.array([0.04, -0.045, 0.0]), hint=(0.2, -1.0, 0.6))
        view = (55.0, 0.0, -5.0)
    elif ch == "E":
        fingers = {k: (40, 100, 70, v[3]) for k, v in cu.items()}
        thumb = dict(mc=(0.40, 0.50, 0.75), target=(-0.02, 0.30, 0.10))
    elif ch == "F":
        fingers = dict(index=(60, 70, 40, 0), middle=(0, 0, 0, 5), ring=(0, 0, 0, -8), pinky=(0, 0, 0, -21))
        thumb = dict(mc=(0.45, 0.50, 0.74), target=tip_of("index", fingers["index"]) + np.array([0.0, -0.03, 0.02]))
    elif ch == "G":
        fingers = dict(cu, index=(0, 0, 0, 0))
        thumb = dict(dirs=((0.72, 0.58, 0.38), (0.25, 0.93, 0.28), (0.0, 1.0, 0.10)))
        view = (180.0, 0.0, -75.0)
        wrist = (0.0, 15.0)
    elif ch == "H":
        fingers = dict(cu, index=(0, 0, 0, -2), middle=(0, 0, 0, 2))
        thumb = dict(dirs=((0.45, 0.60, 0.66), (-0.10, 0.95, 0.30), (-0.10, 1.0, 0.05)))
        view = (180.0, 0.0, -75.0)
        wrist = (0.0, 15.0)
    elif ch == "I":
        fingers = dict(cu, pinky=(0, 0, 0, -10))
    elif ch == "K":
        fingers = dict(cu, index=(0, 0, 0, 4), middle=(62, 0, 0, -3))
        thumb = dict(mc=(0.50, 0.60, 0.62), target=(0.09, 0.60, 0.17))
        view = (15.0, 0.0, 0.0)
    elif ch == "L":
        fingers = dict(cu, index=(0, 0, 0, 2))
        thumb = dict(dirs=((0.72, 0.60, 0.30), (0.93, 0.35, 0.10), (0.95, 0.30, 0.0)))
    elif ch == "M":
        fingers = dict(cu, index=(70, 100, 20, 0), middle=(70, 100, 20, 0), ring=(70, 100, 20, -2))
        thumb = dict(mc=(0.35, 0.50, 0.79), target=(-0.10, 0.40, 0.12))
    elif ch == "N":
        fingers = dict(cu, index=(70, 100, 20, 0), middle=(70, 100, 20, 0))
        thumb = dict(mc=(0.35, 0.50, 0.79), target=(0.0, 0.42, 0.12))
    elif ch == "O":
        fingers = dict(index=(62, 62, 40, 0), middle=(62, 62, 40, 0), ring=(62, 62, 40, -2), pinky=(65, 62, 40, -4))
        thumb = dict(mc=(0.40, 0.35, 0.85), target=tip_of("middle", fingers["middle"]) + np.array([0.04, -0.045, 0.0]), hint=(0.2, -1.0, 0.6))
        view = (75.0, 0.0, -10.0)
    elif ch == "P":
        fingers = dict(cu, index=(0, 0, 0, 4), middle=(62, 0, 0, -3))
        thumb = dict(mc=(0.50, 0.60, 0.62), target=(0.09, 0.60, 0.17))
        view = (-90.0, 0.0, 50.0)
        wrist = (20.0, 0.0)
    elif ch == "Q":
        fingers = dict(cu, index=(0, 0, 0, 0))
        thumb = dict(dirs=((0.72, 0.58, 0.38), (0.25, 0.93, 0.28), (0.0, 1.0, 0.10)))
        view = (180.0, 0.0, 135.0)
        wrist = (20.0, 0.0)
    elif ch == "R":
        fingers = dict(cu, index=(0, 0, 0, -7), middle=(8, 0, 0, 7))
        thumb = th_hold
    elif ch == "S":
        pass
    elif ch == "T":
        fingers = dict(cu, index=(80, 95, 60, 0))
        thumb = dict(mc=(0.40, 0.60, 0.69), target=(0.10, 0.53, 0.20))
    elif ch == "U":
        fingers = dict(cu, index=(0, 0, 0, -2), middle=(0, 0, 0, 2))
        thumb = th_hold
    elif ch == "V":
        fingers = dict(cu, index=(0, 0, 0, 13), middle=(0, 0, 0, -11))
        thumb = th_hold
    elif ch == "W":
        fingers = dict(cu, index=(0, 0, 0, 16), middle=(0, 0, 0, 0), ring=(0, 0, 0, -16), pinky=(70, 95, 55, -4))
        thumb = dict(mc=(0.38, 0.42, 0.82), target=(-0.13, 0.33, 0.20))
    elif ch == "X":
        fingers = dict(cu, index=(15, 95, 65, 0))
        view = (70.0, 0.0, -5.0)
    elif ch == "Y":
        fingers = dict(cu, pinky=(0, 0, 0, -28))
        thumb = dict(dirs=((0.78, 0.50, 0.30), (0.93, 0.36, 0.05), (0.93, 0.36, 0.0)))
    return dict(fingers=fingers, thumb=thumb, wrist=wrist), view


def place(C, Rad, fore, view, fill, dx=0.0, dy=0.0, n=28):
    """Rotate into view and choose scale / offset so that the hand's bounding box has its longer side = fill * n."""
    yaw, pitch, roll = view
    R = Rz(roll * D2R) @ Rx(pitch * D2R) @ Ry(yaw * D2R)
    Cw = C @ R.T
    h = ~fore
    x0 = (Cw[h, 0] - Rad[h]).min(); x1 = (Cw[h, 0] + Rad[h]).max()
    y0 = (Cw[h, 1] - Rad[h]).min(); y1 = (Cw[h, 1] + Rad[h]).max()
    scale = fill * n / max(x1 - x0, y1 - y0)
    ox = n / 2.0 + dx - scale * (x0 + x1) / 2.0
    oy = n / 2.0 + dy + scale * (y0 + y1) / 2.0
    return Cw, scale, ox, oy


# ---------------------------------------------------------------------------------------------------------------
# Random (fixed seed) variation of pose, viewpoint, size and lighting around each letter's canonical pose
# ---------------------------------------------------------------------------------------------------------------
FRONT = dict(yaw=(-20, 30), pitch=(-15, 20), roll=(-22, 10), flex=(-15, 15), dev=(-12, 12))
VIEWS = {
    "C": dict(yaw=(45, 95), roll=(-32, 5)),
    "O": dict(yaw=(45, 95), roll=(-32, 5)),
    "D": dict(yaw=(20, 80), roll=(-25, 8)),
    "X": dict(yaw=(30, 95), roll=(-25, 8)),
    "T": dict(yaw=(0, 85), roll=(-28, 8)),
    "K": dict(yaw=(-10, 45)),
    "M": dict(pitch=(-5, 35)),
    "N": dict(pitch=(-5, 35)),
    "G": dict(yaw=(150, 215), roll=(-100, -55), dev=(-35, 0), pitch=(-15, 15)),
    "H": dict(yaw=(150, 215), roll=(-100, -55), dev=(-35, 0), pitch=(-15, 15)),
    "P": dict(yaw=(-115, -60), roll=(15, 80), flex=(0, 45), pitch=(-15, 15)),
    "Q": dict(yaw=(150, 210), roll=(105, 165), flex=(0, 40), dev=(-15, 15), pitch=(-15, 15)),
}


def sample_variant(ch, rng):
    pose, _ = base_pose(ch)
    fingers = {}
    for name in FNAMES:
        m, p, d, a = pose["fingers"][name]
        straight = (m < 30 and p < 30)
        sd = 4.0 if straight else 7.0
        j = rng.normal(0.0, 1.0, 4)
        m2, p2, d2 = m + sd * j[0], p + sd * j[1], d + sd * j[2]
        if straight:
            m2, p2, d2 = max(m2, -5.0), max(p2, 0.0), max(d2, 0.0)
        fingers[name] = (m2, p2, d2, a + 2.5 * j[3])
    th = dict(pose["thumb"])
    j = rng.normal(0.0, 1.0, (3, 3))
    if "dirs" in th:
        th["dirs"] = tuple(unit(v) + 0.07 * j[i] for i, v in enumerate(th["dirs"]))
    else:
        th["mc"] = unit(th["mc"]) + 0.06 * j[0]
        th["target"] = np.asarray(th["target"], dtype=np.float64) + 0.015 * j[1]
    rngs = dict(FRONT); rngs.update(VIEWS.get(ch, {}))
    u = rng.uniform(0.0, 1.0, 14)
    lerp = lambda ab, t: ab[0] + (ab[1] - ab[0]) * t
    view = (lerp(rngs["yaw"], u[0]), lerp(rngs["pitch"], u[1]), lerp(rngs["roll"], u[2]))
    wrist = (lerp(rngs["flex"], u[3]), lerp(rngs["dev"], u[4]))
    par = dict(fill=0.74 + 0.26 * u[5], dx=-1.5 + 3.0 * u[6], dy=-1.5 + 3.0 * u[7],
               light=unit([-0.5 + 1.0 * u[8], -0.2 + 0.8 * u[9], 1.0]), amb=0.40 + 0.30 * u[10],
               albedo=0.65 + 0.35 * u[11], bg=0.25 + 0.65 * u[12], blur=0.3 + 0.4 * u[13])
    return dict(fingers=fingers, thumb=th, wrist=wrist), view, par


def blur_img(img, sigma):
    k = np.exp(-0.5 * (np.arange(-2, 3) / sigma) ** 2); k = k / k.sum()
    p = np.pad(img, 2, mode="edge")
    t = sum(k[i] * p[:, i:i + img.shape[1]] for i in range(5))
    return sum(k[i] * t[i:i + img.shape[0], :] for i in range(5))


def draw_variant(ch, rng, mirror=False):
    pose, view, par = sample_variant(ch, rng)
    C, Rad, fore = hand_spheres(pose)
    Cw, sc, ox, oy = place(C, Rad, fore, view, par["fill"], par["dx"], par["dy"])
    img = render(Cw, Rad, sc, ox, oy, par["light"], par["amb"], par["albedo"], par["bg"])
    img = blur_img(img, par["blur"])
    return img[:, ::-1].copy() if mirror else img


# ---------------------------------------------------------------------------------------------------------------
# Feature front-end (formula filters).  The same numbers are written into the network's first layers.
# ---------------------------------------------------------------------------------------------------------------
N_ORI = 8          # unsigned edge orientations, 22.5 degrees apart
SHARP = 1.0        # e_k = relu(|d_k| - SHARP * |d_k+90deg|): orientation tuning, zero beyond 45 degrees
POOL_SIGMA = 1.5   # pixels, Gaussian pooling of orientation energy (tolerance to small deformations)
POOL_K = 8         # pooling kernel size, stride 2 -> 13 x 13 cells
NB_SIGMA = 1.0     # cells, neighbourhood used by the local contrast normaliser
NB_BETA = 2.0      # weight of the neighbourhood energy in the normaliser
EPS_EDGE = 0.03    # pooled-gradient level below which the normaliser stops amplifying (image noise floor)
ALPHA = float(np.sqrt(16.0 * 1e-5) / EPS_EDGE)   # input gain that turns the norm op's 1e-5 into EPS_EDGE
G = 13             # cells per side
SOBX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
SOBY = SOBX.T.copy()
THETAS = np.arange(N_ORI) * np.pi / N_ORI


def pool_kernel():
    o = np.arange(POOL_K) - (POOL_K - 1) / 2.0
    k = np.exp(-0.5 * (o / POOL_SIGMA) ** 2); k = k / k.sum()
    return np.outer(k, k)


def nb_kernel():
    o = np.arange(-2, 3)
    k = np.exp(-0.5 * (o / NB_SIGMA) ** 2); k = k / k.sum()
    return np.outer(k, k)


def features(imgs):
    """imgs (B, 28, 28) -> locally normalised orientation maps (B, 8, 13, 13); mirrors the network exactly."""
    imgs = np.asarray(imgs, dtype=np.float64)
    B = imgs.shape[0]
    gx = np.zeros((B, 26, 26)); gy = np.zeros((B, 26, 26))
    for dy in range(3):
        for dx in range(3):
            win = imgs[:, dy:dy + 26, dx:dx + 26]
            gx += SOBX[dy, dx] * win; gy += SOBY[dy, dx] * win
    d = np.stack([np.cos(t) * gx + np.sin(t) * gy for t in THETAS], axis=1)
    a = np.abs(d)
    e = np.maximum(a - SHARP * np.roll(a, N_ORI // 2, axis=1), 0.0)
    pk = pool_kernel()
    ep = np.pad(e, ((0, 0), (0, 0), (3, 3), (3, 3)))
    F = np.zeros((B, N_ORI, G, G))
    for dy in range(POOL_K):
        for dx in range(POOL_K):
            F += pk[dy, dx] * ep[:, :, dy:dy + 2 * G:2, dx:dx + 2 * G:2]
    nk = nb_kernel()
    Fp = np.pad(F, ((0, 0), (0, 0), (2, 2), (2, 2)))
    Gn = np.zeros_like(F)
    for dy in range(5):
        for dx in range(5):
            Gn += nk[dy, dx] * Fp[:, :, dy:dy + G, dx:dx + G]
    var = (ALPHA ** 2) * ((F ** 2).sum(1, keepdims=True) + (NB_BETA ** 2) * (Gn ** 2).sum(1, keepdims=True)) / 16.0
    return ALPHA * F / np.sqrt(var + 1e-5)


# ---------------------------------------------------------------------------------------------------------------
# Template bank and network
# ---------------------------------------------------------------------------------------------------------------
V_PER_CLASS = 250      # drawings per letter (the last fifth are drawn mirrored: left hand / back-of-hand view)
N_MIRROR = 50
SHIFT_PAD = 2          # templates are also tried at shifts of up to +-2 cells (+-4 pixels)
SEED = 20240


def make_templates(v_per_class=V_PER_CLASS, n_mirror=N_MIRROR, seed=SEED, return_images=False):
    imgs = []
    for ci, ch in enumerate(LETTERS):
        rng = np.random.RandomState(seed + 101 * ci)
        for v in range(v_per_class):
            imgs.append(draw_variant(ch, rng, mirror=(v >= v_per_class - n_mirror)))
    imgs = np.array(imgs)
    Z = features(imgs)
    T = Z - Z.mean(axis=(1, 2, 3), keepdims=True)                      # each template: its own mean removed ...
    T = T / np.sqrt((T ** 2).sum(axis=(1, 2, 3), keepdims=True))       # ... and its own norm divided out
    return (T, imgs) if return_images else T


def build(out_dir, v_per_class=V_PER_CLASS, n_mirror=N_MIRROR):
    W = {}
    # g: steered Sobel derivatives, +d_k and -d_k for each of the 8 orientations
    w = np.zeros((2 * N_ORI, 1, 3, 3))
    for k, t in enumerate(THETAS):
        dk = np.cos(t) * SOBX + np.sin(t) * SOBY
        w[2 * k, 0] = dk; w[2 * k + 1, 0] = -dk
    W["g.W"] = w; W["g.b"] = np.zeros(2 * N_ORI)
    # ori: |d_k| - SHARP * |d_(k+90deg)|
    w = np.zeros((N_ORI, 2 * N_ORI, 1, 1))
    for k in range(N_ORI):
        kp = (k + N_ORI // 2) % N_ORI     # np.roll(a, 4)[k] = a[k-4] = a[(k+4) % 8]
        w[k, 2 * k, 0, 0] = 1.0; w[k, 2 * k + 1, 0, 0] = 1.0
        w[k, 2 * kp, 0, 0] -= SHARP; w[k, 2 * kp + 1, 0, 0] -= SHARP
    W["ori.W"] = w; W["ori.b"] = np.zeros(N_ORI)
    # pool: Gaussian, stride 2
    w = np.zeros((N_ORI, N_ORI, POOL_K, POOL_K)); pk = pool_kernel()
    for k in range(N_ORI):
        w[k, k] = pk
    W["pool.W"] = w; W["pool.b"] = np.zeros(N_ORI)
    # gather: per cell p, 32 numbers [ +aF_k, -aF_k, +abG_k, -abG_k ] for k = 0..7 (G = neighbourhood average of F)
    P = G * G
    w = np.zeros((P * 4 * N_ORI, N_ORI * P)); nk = nb_kernel()
    for i in range(G):
        for j in range(G):
            p = i * G + j
            for k in range(N_ORI):
                r = p * 4 * N_ORI + 4 * k
                w[r, k * P + p] = ALPHA; w[r + 1, k * P + p] = -ALPHA
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < G and 0 <= jj < G:
                            q = ii * G + jj
                            w[r + 2, k * P + q] = ALPHA * NB_BETA * nk[di + 2, dj + 2]
                            w[r + 3, k * P + q] = -ALPHA * NB_BETA * nk[di + 2, dj + 2]
    W["gather.W"] = w; W["gather.b"] = np.zeros(P * 4 * N_ORI)
    # back: (cell, orientation) -> (orientation, row, col)
    w = np.zeros((N_ORI * P, P * N_ORI))
    for p in range(P):
        for k in range(N_ORI):
            w[k * P + p, p * N_ORI + k] = 1.0
    W["back.W"] = w; W["back.b"] = np.zeros(N_ORI * P)
    # tm: matched filters, one per drawing
    T = make_templates(v_per_class, n_mirror)
    W["tm.W"] = T; W["tm.b"] = np.zeros(len(T))
    nt = len(T); ncls = len(LETTERS); sp = 2 * SHIFT_PAD + 1
    layers = [
        dict(name="g", op="conv", inputs=["image"], out=2 * N_ORI, kernel=[3, 3], stride=1, padding=0),
        dict(name="g_relu", op="relu", inputs=["g"]),
        dict(name="ori", op="conv", inputs=["g_relu"], out=N_ORI, kernel=[1, 1], stride=1, padding=0),
        dict(name="ori_relu", op="relu", inputs=["ori"]),
        dict(name="pool", op="conv", inputs=["ori_relu"], out=N_ORI, kernel=[POOL_K, POOL_K], stride=2, padding=3),
        dict(name="gather", op="dense", inputs=["pool"], out=P * 4 * N_ORI),
        dict(name="cells", op="reshape", inputs=["gather"], shape=[P, 4 * N_ORI, 1]),
        dict(name="lnorm", op="norm", inputs=["cells"]),
        dict(name="pick", op="maxpool", inputs=["lnorm"], size=[1, 1], stride=[4, 1]),
        dict(name="back", op="dense", inputs=["pick"], out=N_ORI * P),
        dict(name="maps", op="reshape", inputs=["back"], shape=[N_ORI, G, G]),
        dict(name="tm", op="conv", inputs=["maps"], out=nt, kernel=[G, G], stride=1, padding=SHIFT_PAD),
        dict(name="tm_max", op="maxpool", inputs=["tm"], size=[sp, sp], stride=[sp, sp]),
        dict(name="by_class", op="reshape", inputs=["tm_max"], shape=[1, ncls, nt // ncls]),
        dict(name="cls_max", op="maxpool", inputs=["by_class"], size=[1, nt // ncls], stride=[1, nt // ncls]),
        dict(name="scores", op="reshape", inputs=["cls_max"], shape=[ncls]),
    ]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
