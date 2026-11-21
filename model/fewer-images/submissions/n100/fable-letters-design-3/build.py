#!/usr/bin/env python3
"""Builds a hand-designed sign-language-letter network (no data, no training).

Every number here comes from general knowledge: the anatomy of a hand, the ASL manual
alphabet handshapes, and formula filters.  A small 3-D hand model (spheres along bones) is
posed for each letter, drawn as a 28x28 picture, pushed through the fixed front-end of the
network (oriented-edge filters from formulas) and each drawing becomes ONE template
(normalised by itself alone).  The network scores an image by its best-matching template
per letter.
"""
import sys, json, math
import numpy as np

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
D2R = math.pi / 180.0

# ----------------------------------------------------------------------------------------------
# hand anatomy (cm).  Hand frame: wrist at origin, fingers along +y, thumb side +x,
# palm normal +z.  A right hand with its palm toward the viewer then has the thumb on the
# viewer's right, as a person facing a right-handed signer sees it.
# ----------------------------------------------------------------------------------------------
MCP = np.array([[2.9, 8.6, 0.0], [1.0, 9.0, 0.0], [-0.95, 8.6, 0.0], [-2.8, 7.8, 0.0]])
CARPAL = np.array([[2.2, 0.9, 0.0], [0.75, 0.9, 0.0], [-0.75, 0.9, 0.0], [-2.2, 0.9, 0.0]])
FLEN = np.array([[4.1, 2.4, 1.9], [4.6, 2.8, 2.0], [4.2, 2.6, 2.0], [3.3, 1.9, 1.8]])
FRAD = np.array([[0.98, 0.90, 0.83, 0.75], [1.00, 0.92, 0.85, 0.77], [0.95, 0.87, 0.80, 0.72], [0.85, 0.77, 0.70, 0.64]])
THUMB_BASE = np.array([2.1, 1.7, 0.9])
TLEN = np.array([4.4, 3.2, 2.4])
TRAD = np.array([1.35, 1.15, 1.05, 0.92])
PALM_R0, PALM_R1 = 1.6, 1.3


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def chain(p0, p1, r0, r1, out, step=0.5):
    """spheres from p0 to p1 with radius going from r0 to r1"""
    L = float(np.linalg.norm(p1 - p0))
    n = max(2, int(math.ceil(L / (step * min(r0, r1)))) + 1)
    for i in range(n):
        t = i / (n - 1.0)
        out.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]), p0[2] + t * (p1[2] - p0[2]), r0 + t * (r1 - r0)))


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-9)


def hand_spheres(pose):
    """pose: dict with
         fingers: 4 x (splay, mcp, pip, dip) in degrees (index, middle, ring, pinky);
                  splay > 0 leans toward the thumb side, flexion curls toward the palm
         lift:    4 values (cm), how far each finger base is pushed toward the palm side (optional)
         thumb:   3 direction vectors for metacarpal, proximal and distal bones (hand frame)
         arm:     direction of the forearm from the wrist (hand frame)
       returns (hand spheres (n,4), arm spheres (m,4)) in the hand frame"""
    S = []
    webs = []
    wide, flen, thick = pose.get("wide", 1.0), pose.get("flen", 1.0), pose.get("thick", 1.0)
    for f in range(4):
        splay, a1, a2, a3 = [v * D2R for v in pose["fingers"][f]]
        u = np.array([math.sin(splay), math.cos(splay), 0.0])
        n = np.array([0.0, 0.0, 1.0])
        p = MCP[f] * np.array([wide, 1.0, 1.0])
        if "lift" in pose:
            p[2] += pose["lift"][f]
        chain(CARPAL[f] * np.array([wide, 1.0, 1.0]), p, PALM_R0, PALM_R1, S, step=0.6)
        if f < 3:                         # fill between this ray and the next so the palm is smooth
            chain(0.5 * (CARPAL[f] + CARPAL[f + 1]) * np.array([wide, 1.0, 1.0]),
                  0.5 * (MCP[f] + MCP[f + 1]) * np.array([wide, 1.0, 1.0]), PALM_R0, PALM_R1, S, step=0.6)
        th = 0.0
        for i, a in enumerate((a1, a2, a3)):
            th += a
            q = p + flen * FLEN[f, i] * (math.cos(th) * u + math.sin(th) * n)
            if i == 0:
                webs.append(p + 0.33 * (q - p))
            chain(p, q, thick * FRAD[f, i], thick * FRAD[f, i + 1], S, step=0.7)
            p = q
    for f in range(3):                    # skin web between neighbouring finger bases
        chain(webs[f], webs[f + 1], 0.8, 0.8, S, step=0.8)
    p = THUMB_BASE * np.array([wide, 1.0, 1.0])
    for i in range(3):
        q = p + (flen if i else 1.0) * TLEN[i] * unit(pose["thumb"][i])
        chain(p, q, thick * TRAD[i], thick * TRAD[i + 1], S, step=0.7)
        if i == 0:                        # skin web between the thumb and the edge of the palm
            ip = MCP[0] * np.array([wide, 1.0, 1.0])
            ic = CARPAL[0] * np.array([wide, 1.0, 1.0])
            chain(p + 0.75 * (q - p), ic + 0.7 * (ip - ic), 1.0, 1.0, S, step=0.8)
            chain(p + 0.4 * (q - p), ic + 0.4 * (ip - ic), 1.15, 1.15, S, step=0.8)
        p = q
    A = []
    fdir = unit(pose["arm"])
    side = np.array([1.0, 0.0, 0.0]) - fdir[0] * fdir
    side = unit(side)
    w0 = np.array([0.0, 0.6, 0.0])
    for sgn in (-1.0, -0.5, 0.0, 0.5, 1.0):
        chain(w0 + sgn * 1.55 * side, w0 + sgn * 2.1 * side + 16.0 * fdir, 1.75, 2.35, A, step=0.6)
    return np.array(S), np.array(A)


# ----------------------------------------------------------------------------------------------
# drawing: orthographic view, depth buffer over spheres, simple light + crease darkening
# ----------------------------------------------------------------------------------------------
HI = 56          # drawing resolution (2x the 28-pixel frame)


def box_blur(a, r):
    """mean over a (2r+1)^2 window, edges replicated"""
    k = 2 * r + 1
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(p, axis=0)
    c = np.concatenate([c[k - 1:k], c[k:] - c[:-k]], axis=0)
    c2 = np.cumsum(c, axis=1)
    c2 = np.concatenate([c2[:, k - 1:k], c2[:, k:] - c2[:, :-k]], axis=1)
    return c2 / (k * k)


def render(pose, view, look, n=HI):
    """view: dict yaw, pitch, roll (degrees), fill_long, fill_short, dx, dy (pixels of the 28-frame)
       look: dict light (lx, ly), ambient, skin, bg, crease
       returns an n x n picture of the 28x28 frame, values 0..1"""
    S, A = hand_spheres(pose)
    R = rot_z(view["roll"] * D2R) @ rot_x(view["pitch"] * D2R) @ rot_y(view["yaw"] * D2R)
    Sc = S[:, :3] @ R.T
    Ac = A[:, :3] @ R.T
    # square crop around the hand: the long side of the hand's box fills most of the frame,
    # but the short side is never allowed to fill more than fill_short of it
    lo = (Sc[:, :2] - S[:, 3:4]).min(0)
    hi = (Sc[:, :2] + S[:, 3:4]).max(0)
    ext = hi - lo
    scale = min(view["fill_long"] * 28.0 / ext.max(), view["fill_short"] * 28.0 / ext.min())
    ctr = 0.5 * (lo + hi)
    up = n / 28.0
    s2 = scale * up
    allc = np.concatenate([Sc, Ac], 0)
    rad = np.concatenate([S[:, 3], A[:, 3]]) * s2
    cx = n / 2.0 + (view["dx"] + (allc[:, 0] - ctr[0]) * scale) * up
    cy = n / 2.0 + (view["dy"] - (allc[:, 1] - ctr[1]) * scale) * up
    cz = allc[:, 2] * s2
    keep = (cx + rad > 0) & (cx - rad < n) & (cy + rad > 0) & (cy - rad < n)
    cx, cy, cz, rad = cx[keep], cy[keep], cz[keep], rad[keep]
    # depth buffer: every sphere is evaluated only in a small window around its centre
    K = int(math.ceil(rad.max())) + 1
    off = np.arange(-K, K + 1)
    ix = np.floor(cx).astype(np.int64)[:, None] + off[None, :]
    iy = np.floor(cy).astype(np.int64)[:, None] + off[None, :]
    dx = (ix + 0.5) - cx[:, None]
    dy = (iy + 0.5) - cy[:, None]
    h2 = (rad * rad)[:, None, None] - (dy * dy)[:, :, None] - (dx * dx)[:, None, :]
    ok = (h2 > 0) & ((iy >= 0) & (iy < n))[:, :, None] & ((ix >= 0) & (ix < n))[:, None, :]
    si, wi, wj = np.nonzero(ok)
    hgt = np.sqrt(h2[si, wi, wj])
    z = cz[si] + hgt
    pix = iy[si, wi] * n + ix[si, wj]
    order = np.argsort(z, kind="stable")          # far to near; the nearest surface is written last
    si, pix, z, hgt = si[order], pix[order], z[order], hgt[order]
    rw = rad[si]
    lx, ly = look["light"]
    ln = math.sqrt(lx * lx + ly * ly + 1.0)
    lam = np.maximum((dx[si, wj[order]] * lx - dy[si, wi[order]] * ly + hgt) / (rw * ln), 0.0)
    zbuf = np.full(n * n, np.nan)
    sbuf = np.zeros(n * n)
    zbuf[pix] = z
    sbuf[pix] = look["ambient"] + (1.0 - look["ambient"]) * lam
    mask = ~np.isnan(zbuf)
    # crease darkening: pixels lying deeper than their surroundings (gaps between fingers,
    # the palm just under curled fingertips) get darker
    zfar = zbuf[mask].min() if mask.any() else 0.0
    zf = np.where(mask, zbuf, zfar).reshape(n, n)
    r = max(1, int(round(0.45 * s2)))
    dip = (box_blur(zf, r) - zf) / s2            # in cm
    ao = 1.0 - look["crease"] * np.clip(dip, 0.0, 1.0)
    img = np.where(mask.reshape(n, n), look["skin"] * sbuf.reshape(n, n) * ao, look["bg"])
    return np.clip(img, 0.0, 1.0)


def draw(pose, view, look):
    """28x28 picture (area-averaged from the 56x56 drawing)"""
    return render(pose, view, look, HI).reshape(28, 2, 28, 2).mean((1, 3))


# ----------------------------------------------------------------------------------------------
# the 24 handshapes, typed from knowledge of the ASL manual alphabet.
# Finger entries are (splay, MCP, PIP, DIP) flexion in degrees for index, middle, ring, pinky.
# Thumb entries are the directions of its three bones.  Each letter may have several
# variants (different ways the sign is held / seen); "w" is the share of templates it gets.
# View: yaw turns the hand about the vertical axis (+90: palm faces the viewer's right and the
# little-finger edge is toward the viewer; 180: back of the hand toward the viewer), pitch tips
# it about the horizontal axis, roll turns it in the picture plane (counter-clockwise).
# ----------------------------------------------------------------------------------------------
FIST = (0, 85, 100, 70)
EXT = (0, 3, 4, 3)
TH_SIDE = [(0.6, 0.75, 0.3), (0.05, 1.0, 0.25), (-0.05, 1.0, 0.1)]         # along the side of the index, pointing up
TH_PALM = [(0.2, 0.7, 0.7), (-0.7, 0.6, 0.4), (-0.9, 0.3, 0.2)]            # folded across the palm
TH_OVER = [(0.3, 0.5, 0.8), (-0.5, 0.55, 0.65), (-0.9, 0.25, 0.3)]         # wrapped over the front of curled fingers
TH_UNDER = [(0.2, 0.6, 0.75), (-0.75, 0.4, 0.5), (-0.95, 0.2, 0.2)]        # tucked under curled fingers
TH_OUT = [(0.8, 0.55, 0.2), (0.95, 0.3, 0.1), (1.0, 0.15, 0.05)]           # stuck out sideways
TH_OUT_UP = [(0.7, 0.65, 0.2), (0.75, 0.65, 0.1), (0.7, 0.7, 0.0)]         # stuck out sideways and upward
TH_ALONG = [(0.5, 0.6, 0.55), (0.2, 0.45, 0.85), (0.0, 0.35, 0.95)]        # pointing out of the palm, under a bent-forward index
UP = (-0.25, -1.0, 0.0)      # forearm below the hand; a raised right forearm comes up from the lower left
LEAN = -3                    # hands with raised fingers are held nearly upright
FLEAN = -12                  # fists are held leaning clockwise (top toward the thumb side)
TURN = -5                    # the palm is turned slightly toward the viewer's left


def V(fingers, thumb, yaw=TURN, pitch=0, roll=LEAN, arm=UP, w=1.0, jy=14, jp=12, jr=9, lift=None):
    d = dict(fingers=fingers, thumb=thumb, yaw=yaw, pitch=pitch, roll=roll, arm=arm, w=w, jy=jy, jp=jp, jr=jr)
    if lift is not None:
        d["lift"] = lift
    return d


BACK_RIGHT = dict(yaw=180, roll=-81, arm=(-0.5, -0.85, 0.0))   # back of the hand to the viewer, fingers pointing right, thumb side up
BACK_RIGHT_BENT = dict(yaw=180, roll=-81, arm=(-0.9, -0.45, 0.0))    # the same with the wrist bent, forearm coming up from below

SHAPES = {
    "A": [V([FIST, FIST, FIST, FIST], TH_SIDE, roll=FLEAN)],
    "B": [V([(1.5, 3, 4, 3), (0, 3, 4, 3), (-1.5, 3, 4, 3), (-3.5, 3, 4, 3)], TH_PALM)],
    "C": [V([(0, 35, 50, 35), (0, 35, 50, 35), (0, 38, 52, 35), (0, 42, 55, 35)],
            [(0.3, 0.15, 0.95), (0.1, 0.35, 0.93), (0.0, 0.5, 0.87)], yaw=78, roll=-24, jy=10)],
    "D": [V([EXT, (0, 60, 85, 45), (0, 62, 85, 45), (0, 65, 85, 45)],
            [(0.2, 0.5, 0.85), (-0.5, 0.5, 0.7), (-0.8, 0.5, 0.3)], yaw=-40, jy=12)],
    "E": [V([(0, 45, 110, 70), (0, 45, 110, 70), (0, 45, 110, 70), (0, 50, 110, 70)],
            [(0.3, 0.6, 0.75), (-0.6, 0.55, 0.55), (-1.0, 0.1, 0.15)], roll=FLEAN)],
    "F": [V([(0, 45, 55, 35), (14, 3, 4, 3), (-10, 3, 4, 3), (-34, 3, 4, 3)],
            [(0.35, 0.45, 0.82), (0.05, 0.55, 0.83), (-0.2, 0.75, 0.6)], yaw=48)],
    "G": [V([(0, 88, 5, 5), FIST, FIST, FIST], TH_ALONG, yaw=72, w=0.3),
          V([EXT, FIST, FIST, FIST], [(0.5, 0.8, 0.3), (0.1, 0.95, 0.3), (0.0, 1.0, 0.1)], w=0.35, **BACK_RIGHT),
          V([EXT, FIST, FIST, FIST], [(0.5, 0.8, 0.3), (0.1, 0.95, 0.3), (0.0, 1.0, 0.1)], w=0.35, **BACK_RIGHT_BENT)],
    "H": [V([(0, 88, 5, 5), (0, 88, 5, 5), FIST, FIST], TH_ALONG, yaw=72, w=0.3),
          V([EXT, EXT, FIST, FIST], TH_OVER, w=0.35, **BACK_RIGHT),
          V([EXT, EXT, FIST, FIST], TH_OVER, w=0.35, **BACK_RIGHT_BENT)],
    "I": [V([FIST, FIST, FIST, (-8, 3, 4, 3)], TH_OVER)],
    "K": [V([(4, 3, 4, 3), (-6, 65, 5, 5), FIST, FIST],
            [(0.3, 0.6, 0.75), (0.0, 0.8, 0.6), (-0.1, 0.9, 0.4)], yaw=-30)],
    "L": [V([EXT, FIST, FIST, FIST], TH_OUT), V([EXT, FIST, FIST, FIST], TH_OUT_UP)],
    "M": [V([(0, 80, 75, 30), (0, 80, 75, 30), (0, 80, 75, 30), FIST], TH_UNDER, roll=FLEAN)],
    "N": [V([(0, 80, 75, 30), (0, 80, 75, 30), FIST, FIST], TH_UNDER, roll=FLEAN)],
    "O": [V([(0, 60, 70, 45), (0, 60, 70, 45), (0, 62, 70, 45), (0, 66, 72, 45)],
            [(0.25, 0.5, 0.83), (-0.1, 0.8, 0.6), (-0.3, 0.9, -0.2)], yaw=64, roll=-20, jy=12)],
    "P": [V([EXT, (0, 85, 8, 5), FIST, FIST],
            [(0.4, 0.5, 0.75), (0.0, 0.5, 0.85), (-0.2, 0.6, 0.8)], yaw=-128, roll=82, arm=(0.0, -0.76, 0.64), jy=14, jr=14)],
    "Q": [V([(0, 45, 10, 5), FIST, FIST, FIST],
            [(0.4, 0.35, 0.85), (0.1, 0.5, 0.86), (0.0, 0.6, 0.8)], yaw=-128, pitch=0, roll=125, arm=(0.0, 0.5, 0.87), jr=12)],
    "R": [V([(-7, 0, 3, 3), (9, 10, 4, 4), FIST, FIST], TH_OVER)],
    "S": [V([FIST, FIST, FIST, FIST], TH_OVER, roll=FLEAN)],
    "T": [V([(0, 82, 15, 10), FIST, FIST, FIST],
            [(0.0, 0.9, 0.45), (0.0, 1.0, 0.05), (0.0, 1.0, -0.05)], yaw=70, jy=12),
          V([(0, 85, 45, 25), FIST, FIST, FIST],
            [(0.0, 0.9, 0.45), (0.0, 1.0, 0.05), (0.0, 1.0, -0.05)], yaw=70, jy=12)],
    "U": [V([(2, 3, 4, 3), (-2, 3, 4, 3), FIST, FIST], TH_OVER)],
    "V": [V([(12, 3, 4, 3), (-10, 3, 4, 3), FIST, FIST], TH_OVER)],
    "W": [V([(16, 3, 4, 3), (0, 3, 4, 3), (-16, 3, 4, 3), FIST], TH_UNDER)],
    "X": [V([(0, 15, 95, 60), FIST, FIST, FIST], [(0.25, 0.55, 0.8), (-0.45, 0.6, 0.55), (-0.85, 0.35, 0.1)], yaw=50)],
    "Y": [V([FIST, FIST, FIST, (-30, 3, 4, 3)], TH_OUT_UP)],
}


# ----------------------------------------------------------------------------------------------
# random variation of a handshape (fixed seed): how relaxed the fingers are, how the hand is
# turned, how big it is in the frame, the build of the hand, where the light comes from
# ----------------------------------------------------------------------------------------------
def sample(rs, v):
    common = rs.normal(0.0, 5.0)
    fingers = []
    for (sp, a1, a2, a3) in v["fingers"]:
        if a1 + a2 + a3 > 60:      # a curled finger
            j = common + rs.normal(0.0, 5.0, 3)
            fingers.append((sp + rs.normal(0.0, 2.0), a1 + j[0], max(a2 + j[1], 0.0), max(a3 + j[2], 0.0)))
        else:                      # a straight finger: may lean and bend a little
            j = np.abs(rs.normal(0.0, 6.0, 3))
            fingers.append((sp + rs.normal(0.0, 3.0), a1 + rs.normal(0.0, 6.0), a2 + j[1], a3 + j[2]))
    pose = dict(fingers=fingers,
                thumb=[unit(np.array(t, float) + rs.normal(0.0, 0.12, 3)) for t in v["thumb"]],
                arm=unit(np.array(v["arm"], float) + rs.normal(0.0, 1.0, 3) * np.array([0.2, 0.05, 0.2])),
                wide=rs.uniform(0.96, 1.1), flen=rs.uniform(0.84, 1.0), thick=rs.uniform(1.0, 1.22))
    if "lift" in v:
        pose["lift"] = v["lift"]
    view = dict(yaw=v["yaw"] + rs.normal(0.0, v["jy"]), pitch=v["pitch"] + rs.normal(0.0, v["jp"]),
                roll=v["roll"] + rs.normal(0.0, v["jr"]),
                fill_long=rs.uniform(0.88, 1.08), fill_short=rs.uniform(0.6, 0.78),
                dx=rs.uniform(-2.5, 2.5), dy=rs.uniform(-2.5, 1.5))
    # the hand is drawn light on a dark backdrop with modest creases, so that its outline is
    # the dominant feature of a template and inner detail counts for less
    look = dict(light=(rs.uniform(-0.7, 0.7), rs.uniform(-0.3, 0.7)), ambient=rs.uniform(0.45, 0.6),
                skin=rs.uniform(0.8, 0.95), bg=rs.uniform(0.1, 0.2), crease=rs.uniform(0.5, 0.9))
    return pose, view, look


def drawings(letter, count, seed):
    """count pictures (count, 28, 28) of one letter"""
    rs = np.random.RandomState(seed)
    vs = SHAPES[letter]
    w = np.array([v["w"] for v in vs], float)
    edges = np.cumsum(w) / w.sum()
    out = np.zeros((count, 28, 28))
    for i in range(count):
        v = vs[min(int(np.searchsorted(edges, (i + 0.5) / count)), len(vs) - 1)]
        pose, view, look = sample(rs, v)
        out[i] = draw(pose, view, look)
    return out


# ----------------------------------------------------------------------------------------------
# the network.  Front-end (all weights from formulas):
#   norm   : each image to zero mean / unit variance (removes brightness and contrast)
#   g      : 16 Gaussian-derivative filters (directions every 22.5 degrees over the full circle)
#   e      : unsigned, sharpened orientation energy  |d_k| - BETA*|d_k+90|  for 8 orientations,
#            with a noise floor EDGE_T and a ceiling (soft edge-present signal, so that faint
#            and strong outlines count alike and it does not matter which side is brighter)
#   p      : Gaussian pooling to an 11x11 grid of cells (tolerance to small misplacement)
# Matching: every drawing is a template (conv with an 11x11 kernel, padding 1 = 9 placements),
# best placement (maxpool), best template of each letter (maxpool).
# ----------------------------------------------------------------------------------------------
N_ORI = 8
G_SIGMA = 0.9          # pixels
BETA = 2.0             # orientation sharpening: response falls to zero 26.6 degrees off-axis
EDGE_T = 0.08          # noise floor, in units of image standard deviations per pixel
EDGE_C = 0.25          # range over which the edge signal rises before it saturates
POOL_SIGMA = 1.5       # pixels
POOL_K = 8
POOL_PAD = 2
CELLS = 11
PAD_M = 1


def frontend_weights():
    k = np.arange(-2, 3)
    yy, xx = np.meshgrid(k, k, indexing="ij")
    gauss = np.exp(-(xx ** 2 + yy ** 2) / (2 * G_SIGMA ** 2))
    gauss /= gauss.sum()
    gx = -xx * gauss / G_SIGMA ** 2
    gy = -yy * gauss / G_SIGMA ** 2
    Wg = np.zeros((2 * N_ORI, 1, 5, 5))
    for j in range(2 * N_ORI):
        a = j * math.pi / N_ORI
        Wg[j, 0] = math.cos(a) * gx + math.sin(a) * gy
    We = np.zeros((2 * N_ORI, 2 * N_ORI, 1, 1))
    be = np.zeros(2 * N_ORI)
    for o in range(N_ORI):
        perp = (o + N_ORI // 2) % N_ORI
        for half in (0, 1):                  # channel o: ramp start, channel o + 8: ramp end
            c = o + half * N_ORI
            We[c, o, 0, 0] = We[c, o + N_ORI, 0, 0] = 1.0
            We[c, perp, 0, 0] = We[c, perp + N_ORI, 0, 0] = -BETA
            be[c] = -EDGE_T - half * EDGE_C
    t = np.arange(POOL_K) - (POOL_K - 1) / 2.0
    g1 = np.exp(-t ** 2 / (2 * POOL_SIGMA ** 2))
    g1 /= g1.sum()
    Wp = np.zeros((N_ORI, 2 * N_ORI, POOL_K, POOL_K))
    for o in range(N_ORI):
        Wp[o, o] = np.outer(g1, g1)
        Wp[o, o + N_ORI] = -np.outer(g1, g1)
    return dict(Wg=Wg, bg=np.zeros(2 * N_ORI), We=We, be=be, Wp=Wp, bp=np.zeros(N_ORI))


def conv2d(x, W, b, stride=1, padding=0):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def frontend(imgs, fw):
    """imgs (N, 28, 28) -> pooled orientation maps (N, 8, 11, 11); the same arithmetic as the network layers"""
    x = imgs[:, None].astype(np.float64)
    mu = x.mean(axis=(2, 3), keepdims=True)
    x = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
    x = np.maximum(conv2d(x, fw["Wg"], fw["bg"]), 0)
    x = np.maximum(conv2d(x, fw["We"], fw["be"]), 0)
    return conv2d(x, fw["Wp"], fw["bp"], stride=2, padding=POOL_PAD)


def to_templates(feats):
    """each drawing's feature map becomes one template, normalised by itself alone:
       every orientation channel has its own mean removed, then the whole template gets unit length"""
    t = feats - feats.mean(axis=(2, 3), keepdims=True)
    t /= np.sqrt((t ** 2).sum(axis=(1, 2, 3), keepdims=True)) + 1e-9
    return t


PER_LETTER = 1000
SEED = 20261120


def build(out_dir, per_letter=PER_LETTER, verbose=True):
    fw = frontend_weights()
    nch = N_ORI
    bank = np.zeros((24 * per_letter, nch, CELLS, CELLS), np.float32)
    for li, letter in enumerate(LETTERS):
        pics = drawings(letter, per_letter, SEED + li)
        bank[li * per_letter:(li + 1) * per_letter] = to_templates(frontend(pics, fw))
        if verbose:
            print("letter", letter, "done", flush=True)
    T = 24 * per_letter
    layers = [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "g", "op": "conv", "inputs": ["n0"], "out": 2 * N_ORI, "kernel": [5, 5], "stride": 1, "padding": 0},
        {"name": "g_r", "op": "relu", "inputs": ["g"]},
        {"name": "e", "op": "conv", "inputs": ["g_r"], "out": 2 * N_ORI, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "e_r", "op": "relu", "inputs": ["e"]},
        {"name": "p", "op": "conv", "inputs": ["e_r"], "out": nch, "kernel": [POOL_K, POOL_K], "stride": 2, "padding": POOL_PAD},
        {"name": "m", "op": "conv", "inputs": ["p"], "out": T, "kernel": [CELLS, CELLS], "stride": 1, "padding": PAD_M},
        {"name": "m_best", "op": "maxpool", "inputs": ["m"], "size": [2 * PAD_M + 1, 2 * PAD_M + 1], "stride": [1, 1]},
        {"name": "by_letter", "op": "reshape", "inputs": ["m_best"], "shape": [1, 24, per_letter]},
        {"name": "letter_best", "op": "maxpool", "inputs": ["by_letter"], "size": [1, per_letter], "stride": [1, per_letter]},
        {"name": "scores", "op": "reshape", "inputs": ["letter_best"], "shape": [24]},
    ]
    weights = {
        "g.W": fw["Wg"].astype(np.float32), "g.b": fw["bg"].astype(np.float32),
        "e.W": fw["We"].astype(np.float32), "e.b": fw["be"].astype(np.float32),
        "p.W": fw["Wp"].astype(np.float32), "p.b": fw["bp"].astype(np.float32),
        "m.W": bank, "m.b": np.zeros(T, np.float32),
    }
    with open(out_dir + "/network.json", "w") as f:
        json.dump({"layers": layers}, f)
    np.savez(out_dir + "/weights.npz", **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
