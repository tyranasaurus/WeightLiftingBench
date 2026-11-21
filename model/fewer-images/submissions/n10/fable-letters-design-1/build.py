"""Builds the sign-language network: python3 build.py OUT_DIR  -> OUT_DIR/network.json, OUT_DIR/weights.npz

No data is read.  Every weight comes from code in this file:
  * a fixed edge-orientation front end (Gaussian-derivative filters, rectification, pooling, local contrast normalisation),
  * a bank of templates.  Each template is the front-end response to ONE synthetic picture of a 3D hand model
    (capsule bones with typical human proportions) posed in one manual-alphabet handshape typed in below from general
    knowledge of the ASL alphabet, drawn with a seeded random viewpoint / size / light.  Each template is normalised by
    itself alone; the class score is the best template match of that class (max-pool).
"""
import sys, json, os
import numpy as np

D2R = np.pi / 180.0
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
N_PER_CLASS = int(os.environ.get("HAND_N", "4000"))
SEED = 20261120

# =====================================================================================================================
# 1. hand model.  Units: palm width = 1.  Hand frame: x towards the thumb, y towards the fingertips, z out of the palm.
#    Camera frame: x right, y up, z towards the viewer (orthographic).  Right hand.
# =====================================================================================================================
FNAMES = ("index", "middle", "ring", "pinky")
MCP = {"index": (0.36, 1.00, 0.0), "middle": (0.12, 1.06, 0.0), "ring": (-0.12, 1.01, 0.0), "pinky": (-0.35, 0.90, 0.0)}
FLEN = {"index": (0.47, 0.27, 0.22), "middle": (0.52, 0.31, 0.23), "ring": (0.48, 0.29, 0.22), "pinky": (0.38, 0.21, 0.20)}
FRAD = {"index": (0.125, 0.120, 0.113, 0.105), "middle": (0.127, 0.122, 0.115, 0.107),
        "ring": (0.122, 0.117, 0.110, 0.102), "pinky": (0.108, 0.103, 0.097, 0.090)}
THUMB_CMC = np.array([0.27, 0.20, 0.08])
THUMB_LEN = (0.50, 0.37, 0.31)
THUMB_RAD = (0.19, 0.145, 0.125, 0.108)
PALM_BASE = {"index": (0.21, 0.12, -0.03), "middle": (0.07, 0.10, -0.03), "ring": (-0.07, 0.10, -0.03), "pinky": (-0.21, 0.12, -0.03)}


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


ROT = {"x": rot_x, "y": rot_y, "z": rot_z}


def unit(v):
    v = np.asarray(v, float)
    return v / np.sqrt((v * v).sum())


def hand_capsules(fingers, thumb, wrist):
    """fingers: {name: (spread, mcp, pip, dip)} degrees (spread + = towards the thumb, flexion + = curling into the palm)
       thumb: three bone directions (metacarpal, proximal, distal) in the hand frame
       wrist: (flex, dev) degrees, how the forearm leaves the hand (flex + = forearm swings to the palm side)
       returns A (n,3), B (n,3), rA (n,), rB (n,), is_hand (n,) in the hand frame."""
    A, B, rA, rB, part = [], [], [], [], []

    def add(a, b, ra, rb, p=1):
        A.append(a); B.append(b); rA.append(ra); rB.append(rb); part.append(p)

    for i in range(7):                                         # palm: bars from the wrist fanning out to the knuckle line
        t = i / 2.0
        j = min(int(t), 2)
        w = t - j
        f0, f1 = FNAMES[j], FNAMES[j + 1]
        m = [(1 - w) * MCP[f0][k] + w * MCP[f1][k] for k in range(3)]
        p0 = [(1 - w) * PALM_BASE[f0][k] + w * PALM_BASE[f1][k] for k in range(3)]
        add(p0, (m[0], m[1] - 0.05, m[2] - 0.05), 0.19, 0.17)
    add((-0.20, 0.14, -0.02), (0.20, 0.14, -0.02), 0.20, 0.20)   # heel of the hand
    add((-0.27, 0.17, 0.03), (-0.35, 0.78, 0.0), 0.19, 0.15)     # little-finger edge pad
    add((0.24, 0.30, 0.0), (0.34, 0.84, -0.02), 0.17, 0.15)      # edge below the index
    for f in FNAMES:
        sp, a1, a2, a3 = fingers[f]
        sp = sp * D2R
        p = np.array(MCP[f], float)
        th = 0.0
        rad = FRAD[f]
        for j, (L, a) in enumerate(zip(FLEN[f], (a1, a2, a3))):
            th += a * D2R
            c = np.cos(th)
            q = p + L * np.array([np.sin(sp) * c, np.cos(sp) * c, np.sin(th)])
            add(p, q, rad[j], rad[j + 1])
            p = q
    p = THUMB_CMC.copy()
    for j in range(3):
        q = p + THUMB_LEN[j] * unit(thumb[j])
        add(p, q, THUMB_RAD[j], THUMB_RAD[j + 1])
        p = q
    wf, wd = wrist                                             # forearm: three parallel bars (oval cross-section)
    fdir = rot_z(wd * D2R) @ np.array([0.0, -np.cos(wf * D2R), np.sin(wf * D2R)])
    side = np.cross(fdir, np.array([0.0, 0.0, 1.0]))
    side = unit(side) if (side * side).sum() > 0.01 else np.array([1.0, 0.0, 0.0])
    w0 = np.array([0.0, 0.10, -0.03])
    for sgn in (-1.0, -0.5, 0.5, 1.0):
        add(w0 + sgn * 0.13 * side, w0 + sgn * 0.22 * side + 4.0 * fdir, 0.19, 0.25, 0)
    add(w0, w0 + 4.0 * fdir, 0.20, 0.27, 0)
    return np.array(A, float), np.array(B, float), np.array(rA), np.array(rB), np.array(part, bool)


def rasterize(A, B, rA, rB, light, M):
    """Capsules in pixel coordinates (col, row, depth).  z-buffer; returns depth (M, M) (-1e9 = empty) and the
    Lambert term (surface normal . light) of the visible surface."""
    zbuf = np.full((M, M), -1e9)
    lbuf = np.zeros((M, M))
    lo = np.clip(np.floor(np.minimum(A[:, :2] - rA[:, None], B[:, :2] - rB[:, None])).astype(int), 0, M)
    hi = np.clip(np.ceil(np.maximum(A[:, :2] + rA[:, None], B[:, :2] + rB[:, None])).astype(int) + 1, 0, M)
    grid = np.arange(M) + 0.5
    lx, ly, lz = light
    for i in range(len(A)):
        x0, y0 = lo[i]; x1, y1 = hi[i]
        if x1 <= x0 or y1 <= y0:
            continue
        px = grid[x0:x1][None, :]; py = grid[y0:y1][:, None]
        ax, ay, az = A[i]
        dx, dy, dz = B[i] - A[i]
        dr = rB[i] - rA[i]
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-9), 0.0, 1.0)
        tt = np.stack([t, np.zeros_like(t), np.ones_like(t)])          # nearest point on the bone, and its two ends
        r = rA[i] + tt * dr
        ex = px - (ax + tt * dx); ey = py - (ay + tt * dy)
        q = r * r - ex * ex - ey * ey
        sq = np.sqrt(np.maximum(q, 0.0))
        z = np.where(q > 0, az + tt * dz + sq, -1e9)
        j = z.argmax(0)[None]
        zz = np.take_along_axis(z, j, 0)[0]
        lam = np.take_along_axis((ex * lx + ey * ly + sq * lz) / r, j, 0)[0]
        sub = zbuf[y0:y1, x0:x1]
        upd = zz > sub
        sub[upd] = zz[upd]
        lbuf[y0:y1, x0:x1][upd] = lam[upd]
    return zbuf, lbuf


def box_blur(a, k):
    """sum filter of half-width k, window clamped at the borders"""
    H, W = a.shape
    c = np.cumsum(np.pad(a, ((1, 0), (0, 0))), axis=0)
    lo = np.clip(np.arange(H) - k, 0, H); hi = np.clip(np.arange(H) + k + 1, 0, H)
    a = c[hi] - c[lo]
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1)
    lo = np.clip(np.arange(W) - k, 0, W); hi = np.clip(np.arange(W) + k + 1, 0, W)
    return c[:, hi] - c[:, lo]


def gauss_blur(img, sigma):
    x = np.arange(-2, 3)
    k = np.exp(-x * x / (2 * sigma * sigma)); k /= k.sum()
    p = np.pad(img, 2, mode="edge")
    t = sum(k[i] * p[i:i + img.shape[0], :] for i in range(5))
    return sum(k[i] * t[:, i:i + img.shape[1]] for i in range(5))


# =====================================================================================================================
# 2. the 24 handshapes (general knowledge of the ASL manual alphabet), as seen by a person facing a right-handed signer
# =====================================================================================================================
CURL = (0, 88, 100, 65)       # finger rolled into a fist


def fz(index, middle, ring, pinky):
    return {"index": index, "middle": middle, "ring": ring, "pinky": pinky}


TH_SIDE = [(0.56, 0.70, 0.44), (0.19, 0.96, 0.16), (-0.07, 0.99, 0.07)]      # thumb up along the side of the index (A)
TH_FRONT = [(0.45, 0.45, 0.77), (-0.50, 0.58, 0.64), (-0.95, 0.20, 0.25)]    # thumb wrapped across the front of the fingers (S)
TH_PALM = [(0.20, 0.85, 0.45), (-0.55, 0.75, 0.30), (-0.75, 0.65, 0.05)]     # thumb folded across the palm (B)
TH_HOLD = [(0.15, 0.90, 0.38), (-0.80, 0.50, 0.15), (-0.95, 0.30, -0.05)]    # thumb flat across the palm, holding a curled finger (W)
TH_TUCK = [(0.20, 0.80, 0.50), (-0.80, 0.40, 0.35), (-0.98, 0.10, 0.05)]     # thumb tucked under the fingertips (E)
TH_OUT = [(0.75, 0.60, 0.25), (0.88, 0.45, 0.10), (0.90, 0.42, 0.05)]        # thumb stuck out sideways (L)
TH_O = [(0.30, 0.30, 0.90), (-0.15, 0.55, 0.82), (-0.30, 0.90, -0.30)]       # thumb opposed, tip meeting the fingertips (O)
TH_RING = [(0.30, 0.30, 0.90), (-0.20, 0.60, 0.77), (-0.40, 0.85, 0.30)]     # thumb meeting the index tip in a ring (F)
TH_C = [(0.30, 0.25, 0.92), (0.00, 0.55, 0.83), (-0.10, 0.85, 0.50)]         # thumb opposed but apart (C)
TH_D = [(0.40, 0.50, 0.77), (-0.35, 0.65, 0.67), (-0.75, 0.55, 0.35)]        # thumb tip on the curled middle finger (D)
TH_K = [(0.40, 0.55, 0.73), (0.00, 0.85, 0.52), (-0.20, 0.95, 0.25)]         # thumb tip up against the middle finger (K)
TH_G = [(0.50, 0.75, 0.43), (0.20, 0.95, 0.25), (0.05, 1.00, 0.05)]          # thumb lying parallel to the index (G)
TH_Y = [(0.65, 0.70, 0.25), (0.60, 0.78, 0.10), (0.55, 0.83, 0.05)]          # thumb out and up (Y)
TH_T = [(0.20, 0.90, 0.40), (-0.10, 0.98, 0.15), (0.00, 1.00, 0.00)]         # thumb between index and middle, tip poking past the knuckles (T)
TH_UNDER3 = [(0.15, 0.90, 0.38), (-0.80, 0.50, 0.15), (-0.95, 0.30, -0.05)]  # thumb under three fingers (M)
TH_UNDER2 = [(0.15, 0.90, 0.38), (-0.60, 0.70, 0.20), (-0.80, 0.55, 0.00)]   # thumb under two fingers (N)
TH_Q = [(0.70, 0.60, 0.35), (0.45, 0.85, 0.20), (0.30, 0.95, 0.05)]          # thumb pointing along the index, held apart (Q)

# Views.  A right-handed signer's forearm comes up from the viewer's lower left, so an upright hand leans a little
# clockwise, and a palm held forward beside the shoulder is seen slightly from its thumb side.
FRONT = [("y", -8), ("z", -6)]            # palm to the viewer
FIST = [("y", 0), ("z", -26)]             # palm to the viewer, hand tipped over to the right at the wrist
SIDE = [("y", 75), ("z", -14)]            # palm turned to the viewer's right, seen from the little-finger side
BACK_SIDEWAYS = [("y", 180), ("z", -72)]  # back of the hand to the viewer, fingers pointing to the viewer's right
UP = (0, 0, 3, 3)                         # straight finger
HOLD = (0, 62, 100, 60)                   # finger folded down and held by the thumb (knuckle stays high)

# each variant: fingers, thumb, view = camera-frame rotations applied in order, wrist = (flex, dev),
# jit = half-ranges of the random (yaw, pitch, roll) added to the view, sj = spread jitter (degrees),
# fj = scale of the flexion jitter, fill = range of the fraction of the frame the hand spans
POSES = {
    "A": [dict(fingers=fz(CURL, CURL, CURL, CURL), thumb=TH_SIDE, view=[("y", 0), ("z", -8)])],
    "B": [dict(fingers=fz((0.5, 2, 3, 2), (0, 2, 3, 2), (-0.5, 2, 3, 2), (-2, 2, 3, 2)), thumb=TH_PALM, view=FRONT, sj=1.0)],
    "C": [dict(fingers=fz((0, 50, 45, 30), (0, 50, 45, 30), (0, 50, 45, 30), (0, 52, 45, 30)), thumb=TH_C, view=[("y", 65), ("z", -14)], jit=(18, 15, 14), sj=1.0, fj=0.5)],
    "D": [dict(fingers=fz((2, 0, 3, 3), (0, 50, 80, 50), (0, 50, 80, 50), (0, 53, 80, 50)), thumb=TH_D, view=[("y", -5), ("z", -4)])],
    "E": [dict(fingers=fz((0, 40, 110, 78), (0, 40, 110, 78), (0, 40, 110, 78), (0, 40, 110, 78)), thumb=TH_TUCK, view=FRONT)],
    "F": [dict(fingers=fz((0, 50, 65, 35), (10, 3, 5, 3), (-12, 3, 5, 3), (-34, 3, 5, 3)), thumb=TH_RING, view=[("y", 42)], jit=(20, 14, 14), sj=4.0)],
    "G": [dict(fingers=fz((0, 5, 5, 3), CURL, CURL, CURL), thumb=TH_G, view=BACK_SIDEWAYS, wrist=(0, -12), jit=(25, 15, 18))],
    "H": [dict(fingers=fz((1, 5, 5, 3), (-1, 5, 5, 3), CURL, CURL), thumb=TH_G, view=BACK_SIDEWAYS, wrist=(0, -12), jit=(25, 15, 18), sj=1.5)],
    "I": [dict(fingers=fz(HOLD, HOLD, HOLD, (-10, 0, 3, 3)), thumb=TH_SIDE, view=[("y", -5), ("z", -10)]),
          dict(fingers=fz(HOLD, HOLD, HOLD, (-10, 0, 3, 3)), thumb=TH_FRONT, view=[("y", -5), ("z", -10)])],
    "K": [dict(fingers=fz((8, 0, 3, 3), (-15, 50, 5, 5), HOLD, HOLD), thumb=TH_K, view=[("y", -15), ("z", -10)], jit=(22, 14, 14), sj=4.0, fill=(0.82, 1.05))],
    "L": [dict(fingers=fz(UP, HOLD, HOLD, HOLD), thumb=TH_OUT, view=[("y", -5), ("z", -4)])],
    "M": [dict(fingers=fz((0, 70, 85, 25), (0, 70, 85, 25), (0, 70, 85, 25), CURL), thumb=TH_UNDER3, view=FIST, wrist=(0, 15), jit=(20, 15, 14))],
    "N": [dict(fingers=fz((0, 70, 85, 25), (0, 70, 85, 25), CURL, CURL), thumb=TH_UNDER2, view=FIST, wrist=(0, 15), jit=(20, 15, 14))],
    "O": [dict(fingers=fz((0, 60, 65, 40), (0, 60, 65, 40), (0, 60, 65, 40), (0, 62, 65, 40)), thumb=TH_O, view=SIDE, jit=(15, 15, 14), sj=1.0, fj=0.5)],
    "P": [dict(fingers=fz((6, 0, 3, 3), (-3, 80, 10, 5), CURL, CURL), thumb=TH_K, view=[("y", -90), ("z", 75), ("x", -45)], wrist=(60, 0), jit=(20, 20, 20))],
    "Q": [dict(fingers=fz((0, 10, 5, 3), CURL, CURL, CURL), thumb=TH_Q, view=[("x", 140)], wrist=(75, 0), jit=(20, 22, 15))],
    "R": [dict(fingers=fz((-5, 0, 3, 3), (7, 8, 3, 3), HOLD, HOLD), thumb=TH_FRONT, view=FRONT, sj=1.5)],
    "S": [dict(fingers=fz(CURL, CURL, CURL, CURL), thumb=TH_FRONT, view=[("y", 0), ("z", -22)], wrist=(0, 12), jit=(20, 15, 14))],
    "T": [dict(fingers=fz((0, 85, 8, 5), (0, 88, 40, 25), (0, 90, 65, 35), (0, 92, 80, 45)), thumb=TH_T, view=[("y", 78)], wrist=(-38, 0), jit=(20, 15, 15))],
    "U": [dict(fingers=fz((0.5, 0, 3, 3), (-0.5, 0, 3, 3), HOLD, HOLD), thumb=TH_FRONT, view=FRONT, sj=1.2)],
    "V": [dict(fingers=fz((20, 0, 3, 3), (-6, 0, 3, 3), HOLD, HOLD), thumb=TH_FRONT, view=FRONT, sj=4.0)],
    "W": [dict(fingers=fz((18, 0, 3, 3), (0, 0, 3, 3), (-14, 0, 3, 3), HOLD), thumb=TH_HOLD, view=FRONT, sj=4.0)],
    "X": [dict(fingers=fz((0, 5, 85, 50), CURL, CURL, CURL), thumb=TH_FRONT, view=[("y", 55), ("z", -8)], jit=(25, 15, 12))],
    "Y": [dict(fingers=fz(HOLD, HOLD, HOLD, (-15, 0, 3, 3)), thumb=TH_Y, view=[("y", -5), ("z", -12)])],
}

# =====================================================================================================================
# 3. one random synthetic picture of a letter (28 x 28, values 0..1)
# =====================================================================================================================
IMG = 28
SS = 2            # supersampling factor of the renderer


def sample(letter, rng):
    variants = POSES[letter]
    v = variants[int(rng.integers(len(variants)))]
    fingers = {}
    sj = v.get("sj", 2.5)
    fj = v.get("fj", 1.0)
    for f in FNAMES:
        sp, a1, a2, a3 = v["fingers"][f]
        tight = a2 > 60
        e = rng.normal(0.0, 1.0, 4)
        fingers[f] = (sp + sj * e[0], a1 + fj * 7 * e[1], a2 + fj * (8 if tight else 5) * e[2], a3 + fj * (8 if tight else 4) * e[3])
    thumb = [unit(unit(d) + rng.normal(0.0, 0.10, 3)) for d in v["thumb"]]
    wf, wd = v.get("wrist", (0.0, 0.0))
    e = rng.normal(0.0, 1.0, 2)
    wrist = (wf + 10 * e[0], wd + 8 * e[1])
    R = np.eye(3)
    for ax, a in v["view"]:
        R = ROT[ax](a * D2R) @ R
    jy, jp, jr = v.get("jit", (18, 14, 14))
    u = rng.uniform(-1.0, 1.0, 3)
    R = rot_z(u[2] * jr * D2R) @ rot_x(u[1] * jp * D2R) @ rot_y(u[0] * jy * D2R) @ R
    A, B, rA, rB, is_hand = hand_capsules(fingers, thumb, wrist)
    A = A @ R.T; B = B @ R.T
    # The photo is a square crop around the hand: the hand (wrist to fingertips) about fills the frame, with the wrist
    # at or beyond the edge the forearm enters from; a compact handshape such as a fist is blown up a little less.
    P = np.concatenate([A[is_hand], B[is_hand]]); rr = np.concatenate([rA[is_hand], rB[is_hand]])
    x0, x1 = (P[:, 0] - rr).min(), (P[:, 0] + rr).max()
    y0, y1 = (P[:, 1] - rr).min(), (P[:, 1] + rr).max()
    M = IMG * SS
    u = rng.uniform(0.0, 1.0, 12)
    f0, f1 = v.get("fill", (0.92, 1.15))
    s = min(13.5 + 5.0 * u[0], (f0 + (f1 - f0) * u[1]) * IMG / max(x1 - x0, y1 - y0))      # pixels per palm width at 28 x 28
    # placement: along the forearm the fingertip side keeps a margin of 0..5 px; across it the hand is centred +-3.5 px
    fd = B[-1] - A[-1]
    fn = np.sqrt(fd[0] ** 2 + fd[1] ** 2)
    fcol, frow = (fd[0] / fn, -fd[1] / fn) if fn > 0.4 else (0.0, 1.0)
    off = []
    for ext, fa, uu, um in ((x1 - x0, fcol, u[2], u[10]), (y1 - y0, frow, u[9], u[11])):
        e = s * ext
        lim = max(IMG - e, 0.0) / 2 + 1.0
        centred = float(np.clip(7.0 * uu - 3.5, -lim, lim))
        pushed = (5.0 * um + e / 2 - IMG / 2) * (1.0 if fa > 0 else -1.0)
        off.append(fa * fa * pushed + (1.0 - fa * fa) * centred)
    s = s * SS
    cx = M / 2 + off[0] * SS - s * (x0 + x1) / 2
    cy = M / 2 + off[1] * SS + s * (y0 + y1) / 2
    flip = np.array([s, -s, s])
    A = A * flip + np.array([cx, cy, 0.0]); B = B * flip + np.array([cx, cy, 0.0])
    light = unit([-0.9 + 1.2 * u[3], -0.45 + 0.65 * u[4], 0.75])     # (col, row, depth): frontal, mostly from the viewer's left
    z, lam = rasterize(A, B, s * rA, s * rB, light, M)
    amb = 0.25 + 0.25 * u[5]
    mask = z > -1e8
    sh = amb + (1.0 - amb) * np.clip(lam, 0.0, None)
    k = max(1, int(round(0.12 * s)))                               # crease darkening where the surface dips below its surroundings
    m = mask.astype(float)
    zb = box_blur(np.where(mask, z, 0.0), k) / np.maximum(box_blur(m, k), 1e-6)
    sh = sh * (1.0 - (0.4 + 0.4 * u[6]) * np.clip((zb - z) / (0.10 * s), 0.0, 1.0))
    img = np.where(mask, (0.6 + 0.4 * u[7]) * sh, 0.35 + 0.6 * u[8])
    img = img.reshape(IMG, SS, IMG, SS).mean(axis=(1, 3))
    img = gauss_blur(img, 0.4 + 0.4 * rng.uniform())
    return np.clip(img, 0.0, 1.0)


# =====================================================================================================================
# 4. the network: fixed front end + template bank
# =====================================================================================================================
K = 8                 # edge orientations (unsigned), 22.5 degrees apart
GS = 24               # gradient map side (28 - 5 + 1)
CS = 11               # pooled cell grid side
NC = CS * CS
ROWS = 22             # rows of 8 numbers in each normalisation group
PED = 0.012           # pedestal: pooled response of an edge of about 5% contrast is half way to saturation
FDIM = NC * K


def front_end():
    layers, W = [], {}
    # (a) oriented Gaussian-derivative filters, both signs, sigma = 1
    x = np.arange(-2, 3, dtype=float)
    gx = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / 2.0)
    dx = x[None, :] * gx; dx /= (x[None, :] * dx).sum()
    dy = x[:, None] * gx; dy /= (x[:, None] * dy).sum()
    w = np.zeros((2 * K, 1, 5, 5))
    for k in range(K):
        th = np.pi * k / K
        f = np.cos(th) * dx + np.sin(th) * dy
        w[2 * k, 0] = f; w[2 * k + 1, 0] = -f
    layers.append({"name": "grad", "op": "conv", "inputs": ["image"], "out": 2 * K, "kernel": [5, 5], "stride": 1, "padding": 0})
    W["grad.W"] = w; W["grad.b"] = np.zeros(2 * K)
    layers.append({"name": "grad_r", "op": "relu", "inputs": ["grad"]})
    # (b) unsigned orientation energy, sharpened by subtracting the response at the perpendicular orientation
    w = np.zeros((K, 2 * K, 1, 1))
    for k in range(K):
        kp = (k + K // 2) % K
        w[k, 2 * k] = w[k, 2 * k + 1] = 1.0
        w[k, 2 * kp] = w[k, 2 * kp + 1] = -1.0
    layers.append({"name": "ori", "op": "conv", "inputs": ["grad_r"], "out": K, "kernel": [1, 1], "stride": 1, "padding": 0})
    W["ori.W"] = w; W["ori.b"] = np.zeros(K)
    layers.append({"name": "ori_r", "op": "relu", "inputs": ["ori"]})
    # (c) pooling into overlapping 4 x 4 cells, stride 2, binomial weights
    b = np.array([1.0, 3.0, 3.0, 1.0]) / 8.0
    w = np.zeros((K, K, 4, 4))
    for k in range(K):
        w[k, k] = np.outer(b, b)
    layers.append({"name": "cell", "op": "conv", "inputs": ["ori_r"], "out": K, "kernel": [4, 4], "stride": 2, "padding": 0})
    W["cell.W"] = w; W["cell.b"] = np.zeros(K)
    # (d) local contrast normalisation.  Re-order to (cell, orientation), then for every cell build a group holding
    #     +-its own 8 values, +-the values of its 3 x 3 neighbour cells and a +-pedestal; "norm" divides each group by its
    #     root mean square (the mean is 0 by construction); finally keep the cell's own 8 normalised values.
    w = np.zeros((FDIM, FDIM))
    for k in range(K):
        for c in range(NC):
            w[c * K + k, k * NC + c] = 1.0
    layers.append({"name": "swap", "op": "dense", "inputs": ["cell"], "out": FDIM})
    W["swap.W"] = w; W["swap.b"] = np.zeros(FDIM)
    layers.append({"name": "swap_r", "op": "reshape", "inputs": ["swap"], "shape": [NC, K, 1]})
    w = np.zeros((NC * ROWS, NC, 1, 1)); bias = np.zeros(NC * ROWS)
    for cy in range(CS):
        for cx in range(CS):
            c = cy * CS + cx
            nb = [(cy + a) * CS + (cx + b2) for a in (-1, 0, 1) for b2 in (-1, 0, 1) if 0 <= cy + a < CS and 0 <= cx + b2 < CS]
            g = np.sqrt(9.0 / len(nb))
            w[c * ROWS + 0, c] = 1.0
            w[c * ROWS + 1, c] = -1.0
            for i, n in enumerate(nb):
                w[c * ROWS + 2 + i, n] = g
                w[c * ROWS + 11 + i, n] = -g
            bias[c * ROWS + 20] = PED
            bias[c * ROWS + 21] = -PED
    layers.append({"name": "group", "op": "conv", "inputs": ["swap_r"], "out": NC * ROWS, "kernel": [1, 1], "stride": 1, "padding": 0})
    W["group.W"] = w; W["group.b"] = bias
    layers.append({"name": "group_r", "op": "reshape", "inputs": ["group"], "shape": [NC, ROWS * K, 1]})
    layers.append({"name": "lcn", "op": "norm", "inputs": ["group_r"]})
    layers.append({"name": "lcn_r", "op": "reshape", "inputs": ["lcn"], "shape": [NC * ROWS, K, 1]})
    w = np.zeros((NC, NC * ROWS, 1, 1))
    for c in range(NC):
        w[c, c * ROWS] = 1.0
    layers.append({"name": "feat", "op": "conv", "inputs": ["lcn_r"], "out": NC, "kernel": [1, 1], "stride": 1, "padding": 0})
    W["feat.W"] = w; W["feat.b"] = np.zeros(NC)
    return layers, W


def conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, a:a + stride * ho:stride, b2:b2 + stride * wo:stride] for a in range(kh) for b2 in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def features(images):
    """The front end applied to (n, 28, 28) images, written out directly (same arithmetic as the layers above).
    Returns (n, FDIM) in the (cell, orientation) order the template layer reads."""
    _, W = FRONT
    x = images[:, None].astype(np.float64)
    x = np.maximum(conv(x, W["grad.W"], W["grad.b"], 1, 0), 0.0)
    x = np.maximum(conv(x, W["ori.W"], W["ori.b"], 1, 0), 0.0)
    x = conv(x, W["cell.W"], W["cell.b"], 2, 0)                       # (n, K, CS, CS)
    n = len(x)
    e = (x * x).sum(1)                                                # energy of each cell
    ep = np.pad(e, ((0, 0), (1, 1), (1, 1)))
    cnt = np.pad(np.ones((CS, CS)), 1)
    es = sum(ep[:, a:a + CS, b2:b2 + CS] for a in range(3) for b2 in range(3))
    cn = sum(cnt[a:a + CS, b2:b2 + CS] for a in range(3) for b2 in range(3))
    var = (2.0 * e + 2.0 * es * (9.0 / cn) + 2.0 * K * PED * PED) / (ROWS * K)
    x = x / np.sqrt(var + 1e-5)[:, None]
    return x.transpose(0, 2, 3, 1).reshape(n, FDIM)


FRONT = front_end()


def build(out_dir):
    layers, W = FRONT
    layers = list(layers); W = dict(W)
    rng = np.random.default_rng(SEED)
    T = 24 * N_PER_CLASS
    bank = np.zeros((T, FDIM), np.float32)
    row = 0
    for ci, letter in enumerate(LETTERS):
        imgs = np.stack([sample(letter, rng) for _ in range(N_PER_CLASS)])
        f = features(imgs).reshape(N_PER_CLASS, CS, CS, K)
        # tolerance for where the model puts an edge: smooth each template over neighbouring cells with (1, 2, 1) / 4
        f = np.pad(f, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
        f = 0.25 * f[:, :-2] + 0.5 * f[:, 1:-1] + 0.25 * f[:, 2:]
        f = np.pad(f, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="edge")
        f = 0.25 * f[:, :, :-2] + 0.5 * f[:, :, 1:-1] + 0.25 * f[:, :, 2:]
        f = f.reshape(N_PER_CLASS, FDIM)
        f = f - f.mean(axis=1, keepdims=True)                         # each template normalised by itself alone
        f = f / np.sqrt((f * f).sum(axis=1, keepdims=True))
        bank[row:row + N_PER_CLASS] = f
        row += N_PER_CLASS
    layers.append({"name": "match", "op": "dense", "inputs": ["feat"], "out": T})
    W["match.W"] = bank; W["match.b"] = np.zeros(T)
    layers.append({"name": "match_r", "op": "reshape", "inputs": ["match"], "shape": [1, 24, N_PER_CLASS]})
    layers.append({"name": "best", "op": "maxpool", "inputs": ["match_r"], "size": [1, N_PER_CLASS], "stride": [1, N_PER_CLASS]})
    layers.append({"name": "scores", "op": "reshape", "inputs": ["best"], "shape": [24]})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as fh:
        json.dump({"layers": layers}, fh)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
