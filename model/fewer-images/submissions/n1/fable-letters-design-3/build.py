#!/usr/bin/env python3
"""Writes network.json and weights.npz for the sign-language letter classifier.   python3 build.py OUT_DIR

No file is read and nothing is fitted.  Every weight comes from
  (a) formulas: derivative filters at 8 orientations, a local contrast normaliser, a binomial blur;
  (b) templates: a 3D hand (palm, finger and thumb segments typed from general anatomy) is posed for each of the 24
      letters with joint angles typed from knowledge of the ASL manual alphabet, drawn by a tiny renderer under
      randomly drawn (fixed seed) view, size, place and lighting, and each drawing, pushed through the fixed filters
      of (a) and normalised by itself alone, becomes one template (one convolution kernel).
The network scores a class by its best matching template.
"""
import json
import os
import sys

import numpy as np

DEG = np.pi / 180.0
SEED = 20261120
PER_CLASS = 800                # templates per letter
NAMES = "ABCDEFGHIKLMNOPQRSTUVWXY"

# ----------------------------------------------------------------------------------------------------------------
# Anatomy (cm).  Hand frame of a RIGHT hand: +x = thumb (radial) side, +y = towards the finger tips,
# +z = out of the palm.  With the identity view the palm faces the viewer and the thumb is on the viewer's right.
# ----------------------------------------------------------------------------------------------------------------
FING = ("index", "middle", "ring", "pinky")
MCP = {"index": (2.9, 8.3, 0.0), "middle": (0.95, 8.6, 0.0), "ring": (-0.95, 8.2, 0.0), "pinky": (-2.75, 7.5, 0.0)}
WRIST = {"index": (1.7, 0.6, 0.0), "middle": (0.57, 0.5, 0.0), "ring": (-0.57, 0.5, 0.0), "pinky": (-1.7, 0.6, 0.0)}
FLEN = {"index": (4.0, 2.3, 1.5), "middle": (4.4, 2.7, 1.6), "ring": (4.1, 2.5, 1.6), "pinky": (3.2, 1.8, 1.4)}
FRAD = {"index": 1.07, "middle": 1.1, "ring": 1.04, "pinky": 0.9}
THUMB_BASE = (2.0, 1.2, 0.6)
THUMB_LEN = (4.3, 3.0, 2.2)
THUMB_RAD = (1.5, 1.25, 1.12, 1.0)   # radius at CMC, MCP, IP, tip
PALM_R_WRIST, PALM_R_KNUCKLE = 1.3, 1.05

# finger joint presets: (MCP flexion, PIP flexion, DIP flexion) in degrees
EXT = (0, 0, 0)
CURL = (88, 100, 60)          # tight fist
DRAPE = (65, 85, 35)          # folded forward and hanging down over the thumb (M, N, T)
ECURL = (55, 105, 80)         # E: claw, finger tips come back down to the thumb
OCURVE = (58, 62, 38)         # O: finger tips meet the thumb tip
CCURVE = (45, 40, 25)         # C: open curve, tips well in front of the palm
DCURVE = (60, 70, 45)         # D: three fingers round to the thumb
HOOK = (10, 80, 60)           # X: hooked index

# thumb presets: direction of metacarpal, proximal and distal segment (hand frame, normalised later)
TH_SIDE = ((0.62, 0.74, 0.25), (0.05, 0.98, 0.2), (0.0, 1.0, 0.1))          # A: upright along the index side
TH_ACROSS = ((0.45, 0.55, 0.7), (-0.55, 0.6, 0.58), (-0.9, 0.35, 0.25))      # S, I, U, V, R ...: over the folded fingers
TH_PALM = ((0.35, 0.8, 0.45), (-0.8, 0.55, -0.1), (-0.9, 0.4, -0.1))         # B: flat across the palm
TH_E = ((0.35, 0.75, 0.55), (-0.85, 0.45, 0.25), (-0.98, 0.15, -0.1))        # E: across the palm under the tips
TH_C = ((0.55, 0.55, 0.63), (0.25, 0.35, 0.9), (0.0, 0.5, 0.87))             # C: opposed, pointing forward
TH_D = ((0.2, 0.45, 0.87), (-0.15, 0.6, 0.78), (-0.3, 0.9, 0.2))             # D: meets the middle finger tip
TH_F = ((0.4, 0.5, 0.77), (0.0, 0.8, 0.6), (-0.25, 0.95, 0.2))               # F: meets the index tip
TH_G = ((0.45, 0.6, 0.66), (0.05, 0.93, 0.35), (0.0, 0.98, 0.2))             # G: parallel to the index, in front
TH_K = ((0.35, 0.6, 0.72), (-0.3, 0.88, 0.37), (-0.3, 0.92, 0.25))           # K: up between index and middle
TH_L = ((0.85, 0.5, 0.15), (0.97, 0.2, 0.1), (0.98, 0.15, 0.05))             # L: straight out sideways
TH_M = ((0.2, 0.7, 0.68), (-0.8, 0.5, 0.2), (-0.95, 0.3, 0.0))               # M: under three fingers
TH_N = ((0.2, 0.7, 0.68), (-0.55, 0.7, 0.35), (-0.5, 0.8, 0.2))              # N: under two fingers
TH_T = ((0.3, 0.75, 0.55), (-0.3, 0.9, 0.25), (-0.3, 0.92, 0.2))             # T: under the index
TH_O = ((0.35, 0.45, 0.82), (0.0, 0.75, 0.66), (-0.25, 0.9, 0.3))            # O: meets the finger tips
TH_W = ((0.3, 0.7, 0.65), (-0.8, 0.45, 0.4), (-0.9, 0.35, 0.2))              # W: holds the little finger down
TH_Y = ((0.8, 0.55, 0.2), (0.85, 0.5, 0.1), (0.8, 0.6, 0.05))                # Y: out and up


def view(yaw=(0, 22), roll=(0, 14), pitch=(0, 14), dev=(-0.25, 0.25), flex=(-0.15, 0.25), frac=(0.72, 1.05), mirror=False):
    """yaw: about the hand's long axis (+90 = palm turned to the viewer's right so the little-finger edge faces the
    viewer, -90 = thumb edge faces the viewer, 180 = back of the hand); roll: in the image plane, anticlockwise
    (-90 = fingers point to the viewer's right, 180 = down); pitch: finger tips tipped towards the viewer.
    Each is (centre, half-range) in degrees.  dev / flex: range of the forearm direction (-y + dev*x + flex*z).
    frac: share of the frame taken by the hand's bounding box.  mirror: every second drawing is flipped left-right."""
    return dict(yaw=yaw, roll=roll, pitch=pitch, dev=dev, flex=flex, frac=frac, mirror=mirror)


FRONT = view()
SIDE = view(yaw=(58, 28), roll=(-12, 16), dev=(-0.2, 0.2), flex=(-0.2, 0.3))
TILT = view(roll=(0, 16), pitch=(20, 20))
# G and H: the hand lies sideways, palm down or towards the signer, fingers pointing across the frame
POINT = view(yaw=(130, 62), roll=(-85, 22), dev=(-0.9, 0.1), flex=(-0.2, 0.3), frac=(0.8, 1.3), mirror=True)
# fingers: index, middle, ring, pinky -> (flexion preset, abduction in degrees, + = towards the thumb side)
LETTERS = {
    "A": dict(f=[(CURL, 0), (CURL, 0), (CURL, 0), (CURL, 0)], th=TH_SIDE, view=FRONT),
    "B": dict(f=[(EXT, -2), (EXT, 0), (EXT, 2), (EXT, 5)], th=TH_PALM, view=FRONT),
    "C": dict(f=[(CCURVE, -1), (CCURVE, 0), (CCURVE, 1), (CCURVE, 3)], th=TH_C, view=SIDE),
    "D": dict(f=[(EXT, 0), (DCURVE, 0), (DCURVE, 0), (DCURVE, 0)], th=TH_D,
              view=view(yaw=(50, 38), roll=(-6, 14), dev=(-0.2, 0.2), flex=(-0.2, 0.3))),
    "E": dict(f=[(ECURL, 0), (ECURL, 0), (ECURL, 0), (ECURL, 0)], th=TH_E, view=FRONT),
    "F": dict(f=[((45, 65, 40), 0), ((4, 4, 2), 4), ((4, 4, 2), -9), ((4, 4, 2), -23)], th=TH_F, view=FRONT),
    "G": dict(f=[(EXT, 0), (CURL, 0), (CURL, 0), (CURL, 0)], th=TH_G, view=POINT),
    "H": dict(f=[(EXT, -2), (EXT, 1), (CURL, 0), (CURL, 0)], th=TH_ACROSS, view=POINT),
    "I": dict(f=[(CURL, 0), (CURL, 0), (CURL, 0), (EXT, -8)], th=TH_ACROSS, view=FRONT),
    "K": dict(f=[(EXT, 5), ((55, 5, 0), 0), (CURL, 0), (CURL, 0)], th=TH_K, view=view(yaw=(-45, 50))),
    "L": dict(f=[(EXT, 0), (CURL, 0), (CURL, 0), (CURL, 0)], th=TH_L, view=FRONT),
    "M": dict(f=[(DRAPE, 0), (DRAPE, 0), (DRAPE, 0), (CURL, 0)], th=TH_M, view=TILT),
    "N": dict(f=[(DRAPE, 0), (DRAPE, 0), (CURL, 0), (CURL, 0)], th=TH_N, view=TILT),
    "O": dict(f=[(OCURVE, -1), (OCURVE, 0), (OCURVE, 1), (OCURVE, 3)], th=TH_O, view=SIDE),
    # P: K hand hanging from a bent wrist, index roughly level, middle finger down, seen edge-on
    "P": dict(f=[(EXT, 5), ((60, 5, 0), 0), (CURL, 0), (CURL, 0)], th=TH_K,
              view=view(yaw=(-90, 28), roll=(90, 30), dev=(-0.2, 0.2), flex=(0.2, 1.4), frac=(0.75, 1.2), mirror=True)),
    # Q: G hand pointing down, seen edge-on
    "Q": dict(f=[(EXT, 0), (CURL, 0), (CURL, 0), (CURL, 0)], th=TH_G,
              view=view(yaw=(-90, 28), roll=(150, 30), dev=(-0.2, 0.2), flex=(0.0, 0.8), frac=(0.75, 1.2), mirror=True)),
    "R": dict(f=[(EXT, -9), ((10, 0, 0), 9), (CURL, 0), (CURL, 0)], th=TH_ACROSS, view=FRONT),
    "S": dict(f=[(CURL, 0), (CURL, 0), (CURL, 0), (CURL, 0)], th=TH_ACROSS, view=FRONT),
    "T": dict(f=[(DRAPE, 0), (CURL, 0), (CURL, 0), (CURL, 0)], th=TH_T, view=view(yaw=(45, 50), roll=(-4, 14))),
    "U": dict(f=[(EXT, -2), (EXT, 1), (CURL, 0), (CURL, 0)], th=TH_ACROSS, view=FRONT),
    "V": dict(f=[(EXT, 13), (EXT, -11), (CURL, 0), (CURL, 0)], th=TH_ACROSS, view=FRONT),
    "W": dict(f=[(EXT, 17), (EXT, 0), (EXT, -17), (CURL, 0)], th=TH_W, view=FRONT),
    "X": dict(f=[(HOOK, 0), (CURL, 0), (CURL, 0), (CURL, 0)], th=TH_ACROSS, view=SIDE),
    "Y": dict(f=[(CURL, 0), (CURL, 0), (CURL, 0), (EXT, -25)], th=TH_Y, view=FRONT),
}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.sqrt((v * v).sum())


def _capsule(a, b, ra, rb, out, tag):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = int(np.ceil(np.sqrt(((b - a) ** 2).sum()) / (0.45 * min(ra, rb)))) + 1
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]), ra + t * (rb - ra), tag))


def hand_spheres(fingers, thumb, dev=0.0, flex=0.0, flen=1.0, fthick=1.0):
    """fingers: 4 x (mcp, pip, dip, abd) degrees; thumb: 3 direction vectors; forearm direction = -y + dev*x + flex*z;
    flen, fthick: finger length and finger / thumb thickness relative to the average hand.
    Returns array (S, 5): x, y, z, r, tag (1 = hand, 0 = forearm) in the hand frame."""
    S = []
    keys = list(FING)
    w = [np.array(WRIST[k]) for k in keys]
    m = [np.array(MCP[k]) for k in keys]
    for i in range(4):                       # palm: metacarpals and fillers between them
        _capsule(w[i], m[i], PALM_R_WRIST, PALM_R_KNUCKLE, S, 1)
        if i < 3:
            _capsule((w[i] + w[i + 1]) / 2, (m[i] + m[i + 1]) / 2, PALM_R_WRIST, PALM_R_KNUCKLE, S, 1)
    webs = []
    zax = np.array([0.0, 0.0, 1.0])
    for k, (mcp, pip, dip, abd) in zip(keys, fingers):
        d0 = np.array([np.sin(abd * DEG), np.cos(abd * DEG), 0.0])
        p = np.array(MCP[k]); r = FRAD[k] * fthick; ang = 0.0
        rr = (r, 0.96 * r, 0.92 * r, 0.88 * r)
        for j, (a, L) in enumerate(zip((mcp, pip, dip), FLEN[k])):
            ang += a
            d = np.cos(ang * DEG) * d0 + np.sin(ang * DEG) * zax
            q = p + flen * L * d
            _capsule(p, q, rr[j], rr[j + 1], S, 1)
            if j == 0:
                webs.append((p.copy(), d.copy()))
            p = q
    for i in range(3):                       # skin web between neighbouring fingers (first 1.7 cm)
        (pa, da), (pb, db) = webs[i], webs[i + 1]
        _capsule((pa + pb) / 2, (pa + 1.7 * da + pb + 1.7 * db) / 2, 0.95, 0.85, S, 1)
    p = np.array(THUMB_BASE)                 # thumb; its metacarpal doubles as the ball of the thumb
    pts = [p]
    for d, L in zip(thumb, THUMB_LEN):
        p = p + (flen if len(pts) > 1 else 1.0) * L * _unit(d)
        pts.append(p)
    for j in range(3):
        _capsule(pts[j], pts[j + 1], THUMB_RAD[j] * (fthick if j else 1.0), THUMB_RAD[j + 1] * fthick, S, 1)
    for s in (0.25, 0.5, 0.75):              # web between thumb and index metacarpal
        _capsule(pts[0] + s * (pts[1] - pts[0]), w[0] + s * 0.8 * (m[0] - w[0]), 0.9, 0.9, S, 1)
    fd = _unit((dev, -1.0, flex))            # forearm: three sphere chains from the wrist
    side = _unit(np.cross(fd, (0.0, 0.0, 1.0)))
    if side[0] < 0:
        side = -side
    base = np.array([0.0, 0.9, 0.0])
    for off, r0, r1 in ((-1.45, 1.45, 2.0), (0.0, 1.6, 2.3), (1.45, 1.45, 2.0)):
        _capsule(base + off * side, base + 24.0 * fd + 1.45 * off * side, r0, r1, S, 0)
    return np.array(S)


def rot(yaw, pitch, roll):
    cy, sy = np.cos(yaw * DEG), np.sin(yaw * DEG)
    cp, sp = np.cos(pitch * DEG), np.sin(pitch * DEG)
    cr, sr = np.cos(roll * DEG), np.sin(roll * DEG)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


_OFF = {}


def _offsets(R):
    if R not in _OFF:
        o = np.arange(-R, R + 1)
        oy, ox = np.meshgrid(o, o, indexing="ij")
        _OFF[R] = (ox.ravel(), oy.ravel())
    return _OFF[R]


def _box(a, k):
    """box sum over (2k+1)^2 windows, zero outside"""
    n = a.shape[0]
    c = np.zeros((n + 2 * k + 1, n + 2 * k + 1))
    c[k + 1:k + 1 + n, k + 1:k + 1 + n] = a
    c = c.cumsum(0).cumsum(1)
    m = 2 * k + 1
    return c[m:, m:] - c[:-m, m:] - c[m:, :-m] + c[:-m, :-m]


G5 = np.exp(-0.5 * (np.arange(-2, 3) / 0.6) ** 2)
G5 /= G5.sum()


def render(S, R, frac, jx, jy, light, alb, amb, bg, bgrad, N=28, SS=3):
    """S: spheres in the hand frame (S,5); R: rotation; frac: share of the frame taken by the hand's bounding box;
    jx, jy in [-1, 1]: where the box sits inside the free margin.  Returns an (N, N) image in [0, 1]."""
    P = S[:, :3] @ R.T
    r = S[:, 3]
    hand = S[:, 4] > 0.5
    lo = (P[hand, :2] - r[hand, None]).min(0)
    hi = (P[hand, :2] + r[hand, None]).max(0)
    size = max(hi[0] - lo[0], hi[1] - lo[1])
    W = N * SS
    sc = frac * W / size                                   # supersampled pixels per cm
    free = (W - sc * (hi - lo)) / 2 + SS * 1.0             # the box may stick out by one pixel
    tip = R[:, 1]                                          # image direction of the fingers
    off = [0.0, 0.0]
    for a in (0, 1):
        j = (jx, jy)[a]
        if free[a] >= 0:
            off[a] = j * free[a]
        else:                                              # box larger than the frame: keep the finger-tip side inside
            sgn = 1.0 if tip[a] >= 0 else -1.0
            off[a] = sgn * (free[a] - SS * 1.5 * (j + 1.0))
    cx = W / 2 + off[0]
    cy = W / 2 - off[1]
    X = (P[:, 0] - (lo[0] + hi[0]) / 2) * sc + cx - 0.5
    Y = -(P[:, 1] - (lo[1] + hi[1]) / 2) * sc + cy - 0.5
    Z = P[:, 2] * sc
    rp = r * sc
    zbuf = np.full(W * W, -1e9)
    cand = []
    seen = (X + rp > -1) & (X - rp < W) & (Y + rp > -1) & (Y - rp < W)      # spheres that can touch the frame
    X, Y, Z, rp = X[seen], Y[seen], Z[seen], rp[seen]
    small = rp <= 4.5
    for sel in (small, ~small):
        if not sel.any():
            continue
        Rm = int(np.ceil(rp[sel].max())) + 1
        ox, oy = _offsets(Rm)
        x0, y0, z0, r0 = X[sel], Y[sel], Z[sel], rp[sel]
        px = np.round(x0).astype(np.int64)[:, None] + ox[None, :]
        py = np.round(y0).astype(np.int64)[:, None] + oy[None, :]
        dx = px - x0[:, None]
        dy = py - y0[:, None]
        h2 = r0[:, None] ** 2 - dx * dx - dy * dy
        ok = (h2 > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < W)
        dz = np.sqrt(h2[ok])
        idx = (py * W + px)[ok]
        zz = (z0[:, None] + np.zeros_like(h2))[ok] + dz
        rr = (r0[:, None] + np.zeros_like(h2))[ok]
        np.maximum.at(zbuf, idx, zz)
        cand.append((idx, zz, dx[ok] / rr, -dy[ok] / rr, dz / rr))
    shade = np.zeros(W * W)
    for idx, zz, nx, ny, nz in cand:
        win = zz >= zbuf[idx]
        # wrap lighting: skin is flatter than a matte cylinder, so limbs keep their width
        shade[idx[win]] = np.clip((nx[win] * light[0] + ny[win] * light[1] + nz[win] * light[2] + 0.6) / 1.6, 0.0, 1.0)
    mask = (zbuf > -1e8).reshape(W, W)
    z = np.where(mask, zbuf.reshape(W, W), 0.0)
    k = max(1, int(round(0.9 * sc)))
    zb = _box(z, k) / np.maximum(_box(mask.astype(float), k), 1.0)
    ao = np.clip(1.0 - 0.55 * (zb - z) / sc, 0.3, 1.0)     # hollows and creases between fingers are darker
    img = alb * (amb + (1 - amb) * shade.reshape(W, W)) * ao
    yy, xx = np.mgrid[0:W, 0:W] / (W - 1.0) - 0.5
    img = np.where(mask, img, bg + bgrad[0] * xx + bgrad[1] * yy)
    img = img.reshape(N, SS, N, SS).mean((1, 3))
    p = np.pad(img, 2, mode="edge")
    img = sum(G5[i] * p[i:i + N, 2:2 + N] for i in range(5))
    p = np.pad(img, 2, mode="edge")
    img = sum(G5[i] * p[2:2 + N, i:i + N] for i in range(5))
    return np.clip(img, 0.0, 1.0), mask.reshape(N, SS, N, SS).mean((1, 3))


def _vdc(i, base):
    """radical inverse of i: an evenly spreading sequence in [0, 1)"""
    f, x = 1.0, 0.0
    while i > 0:
        f /= base
        x += f * (i % base)
        i //= base
    return x


def sample(letter, rng, i):
    """Drawing number i (1, 2, ...) of a letter and its silhouette.  View angles and size follow evenly spreading
    sequences so that the drawings cover their ranges evenly; joints, place and lighting are random (rng.random())."""
    def u(lo, hi):
        return lo + (hi - lo) * rng.random()

    L = LETTERS[letter]
    V = L["view"]
    fingers = []
    for (flexs, abd) in L["f"]:
        amp = 6.0 if flexs[0] < 20 else 9.0
        fingers.append((max(flexs[0] + u(-amp, amp), 0.0), max(flexs[1] + u(-amp, amp), 0.0),
                        max(flexs[2] + u(-amp, amp), 0.0), abd + u(-4.0, 4.0)))
    thumb = [np.array(d, float) + np.array([u(-0.09, 0.09), u(-0.09, 0.09), u(-0.09, 0.09)]) for d in L["th"]]
    yaw = V["yaw"][0] + V["yaw"][1] * (2 * _vdc(i, 3) - 1)
    roll = V["roll"][0] + V["roll"][1] * (2 * _vdc(i, 5) - 1)
    pitch = V["pitch"][0] + V["pitch"][1] * (2 * _vdc(i, 11) - 1)
    S = hand_spheres(fingers, thumb, u(*V["dev"]), u(*V["flex"]))
    R = rot(yaw, pitch, roll)
    frac = V["frac"][0] + (V["frac"][1] - V["frac"][0]) * _vdc(i, 7)
    jx, jy = u(-1, 1), u(-1, 1)
    light = _unit((u(-0.8, 0.8), u(-0.2, 0.8), 1.0))
    alb = u(0.55, 1.0)
    amb = u(0.3, 0.55)
    bg = u(0.15, 0.95)
    while abs(bg - alb * (amb + (1 - amb) * 0.75)) < 0.12:
        bg = u(0.15, 0.95)
    bgrad = (u(-0.12, 0.12), u(-0.12, 0.12))
    img, sil = render(S, R, frac, jx, jy, light, alb, amb, bg, bgrad)
    if V["mirror"] and i % 2 == 1:
        img, sil = img[:, ::-1].copy(), sil[:, ::-1].copy()
    return img, sil


# ----------------------------------------------------------------------------------------------------------------
# The network
# ----------------------------------------------------------------------------------------------------------------
K = 8            # edge orientations over 180 degrees
BETA = 1.0       # orientation tuning: response falls to zero 45 degrees away from a filter's own orientation
CELLS = 13       # 26 x 26 gradient map pooled 2 x 2
GAMMA = 2.0      # weight of the neighbourhood energy in the contrast normaliser
FLOOR = 0.35     # edges much weaker than this (image scaled to unit spread) are treated as no edge
PAD = 2          # templates are tried at shifts of -2..2 cells (4 pixels) each way


def front_end():
    """Layers from the image to the (K, CELLS, CELLS) map of locally normalised, blurred edge-orientation energy."""
    layers, w = [], {}
    layers.append(dict(name="n0", op="norm", inputs=["image"]))
    sx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], float) / 32.0
    sy = sx.T
    g = np.zeros((2 * K, 1, 3, 3))
    for k in range(K):
        th = np.pi * k / K
        g[k, 0] = np.cos(th) * sx + np.sin(th) * sy
        g[K + k, 0] = -g[k, 0]
    layers.append(dict(name="g", op="conv", inputs=["n0"], out=2 * K, kernel=[3, 3], stride=1, padding=0))
    w["g.W"], w["g.b"] = g, np.zeros(2 * K)
    layers.append(dict(name="gr", op="relu", inputs=["g"]))
    e = np.zeros((K, 2 * K, 1, 1))
    for k in range(K):
        kp = (k + K // 2) % K
        e[k, k, 0, 0] = e[k, K + k, 0, 0] = 1.0                 # |d_k|
        e[k, kp, 0, 0] = e[k, K + kp, 0, 0] = -BETA             # minus |d| at right angles
    layers.append(dict(name="e", op="conv", inputs=["gr"], out=K, kernel=[1, 1], stride=1, padding=0))
    w["e.W"], w["e.b"] = e, np.zeros(K)
    layers.append(dict(name="er", op="relu", inputs=["e"]))
    layers.append(dict(name="p", op="avgpool", inputs=["er"], size=[2, 2], stride=[2, 2]))
    # local contrast normaliser.  For every cell a group of numbers is laid out: the cell's K energies v, the K
    # neighbourhood averages GAMMA*w, a constant FLOOR, and the negatives of all of these (so the group mean is 0).
    # 'norm' over the group divides by sqrt(mean square) = sqrt(sum v^2 + GAMMA^2 sum w^2 + FLOOR^2) / const.
    C2 = CELLS * CELLS
    G = 4 * K + 2
    d1 = np.zeros((C2 * G, K * C2))
    b1 = np.zeros(C2 * G)
    binom = np.array([1.0, 2.0, 1.0]) / 4.0
    for y in range(CELLS):
        for x in range(CELLS):
            p = y * CELLS + x
            for k in range(K):
                d1[p * G + k, k * C2 + p] = 1.0
                d1[p * G + K + k, k * C2 + p] = -1.0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < CELLS and 0 <= xx < CELLS:
                            q = yy * CELLS + xx
                            d1[p * G + 2 * K + k, k * C2 + q] = GAMMA * binom[dy + 1] * binom[dx + 1]
                            d1[p * G + 3 * K + k, k * C2 + q] = -GAMMA * binom[dy + 1] * binom[dx + 1]
            b1[p * G + 4 * K] = FLOOR
            b1[p * G + 4 * K + 1] = -FLOOR
    layers.append(dict(name="d1", op="dense", inputs=["p"], out=C2 * G))
    w["d1.W"], w["d1.b"] = d1, b1
    layers.append(dict(name="d1r", op="reshape", inputs=["d1"], shape=[C2, G, 1]))
    layers.append(dict(name="ln", op="norm", inputs=["d1r"]))
    d2 = np.zeros((K * C2, C2 * G))
    for p in range(C2):
        for k in range(K):
            d2[k * C2 + p, p * G + k] = 1.0
    layers.append(dict(name="d2", op="dense", inputs=["ln"], out=K * C2))
    w["d2.W"], w["d2.b"] = d2, np.zeros(K * C2)
    layers.append(dict(name="f", op="reshape", inputs=["d2"], shape=[K, CELLS, CELLS]))
    fb = np.zeros((K, K, 3, 3))
    for k in range(K):
        fb[k, k] = np.outer(binom, binom)
    layers.append(dict(name="fb", op="conv", inputs=["f"], out=K, kernel=[3, 3], stride=1, padding=1))
    w["fb.W"], w["fb.b"] = fb, np.zeros(K)
    return layers, w


def conv(x, W, b, stride, padding):
    n, c, h, wd = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (wd + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def run_front(layers, w, images):
    """The front end applied to a batch of drawings (N, 1, 28, 28); same arithmetic as the scorer's interpreter."""
    vals = {"image": images}
    for L in layers:
        x = vals[L["inputs"][0]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = conv(x, w[name + ".W"], w[name + ".b"], L["stride"], L["padding"])
        elif op == "dense":
            vals[name] = x.reshape(len(x), -1) @ w[name + ".W"].T + w[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(x, 0)
        elif op == "avgpool":
            (sh, sw), (th, tw) = L["size"], L["stride"]
            ho, wo = (x.shape[2] - sh) // th + 1, (x.shape[3] - sw) // tw + 1
            vals[name] = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0).mean(0)
        elif op == "reshape":
            vals[name] = x.reshape((len(x),) + tuple(L["shape"]))
        elif op == "norm":
            mu = x.mean(axis=(2, 3), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
    return vals[layers[-1]["name"]]


def drawings(per_class=PER_CLASS, seed=SEED):
    """(24 * per_class, 28, 28) drawings and their silhouettes, letter by letter."""
    rng = np.random.default_rng(seed)
    pairs = [sample(c, rng, i + 1) for c in NAMES for i in range(per_class)]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


def templates(layers, w, imgs, sils):
    """Each drawing becomes its own template: its edge map, made zero-sum by subtracting a constant over its OWN
    background cells only (edges where the drawing has background count against it, texture inside the hand is
    ignored), then divided by its own norm."""
    binom = np.array([1.0, 2.0, 1.0]) / 4.0
    out = []
    for i in range(0, len(imgs), 200):
        f = run_front(layers, w, imgs[i:i + 200][:, None].astype(np.float64))            # (n, K, CELLS, CELLS)
        s = sils[i:i + 200][:, 1:27, 1:27].reshape(-1, CELLS, 2, CELLS, 2).mean((2, 4))  # silhouette on the cell grid
        o = np.pad(1.0 - s, ((0, 0), (1, 1), (1, 1)))                                    # background, blurred like the features
        o = sum(binom[a] * binom[b] * o[:, a:a + CELLS, b:b + CELLS] for a in range(3) for b in range(3))
        area = o.sum((1, 2))
        lam = f.sum((1, 2, 3)) / (K * np.maximum(area, 1e-9))
        t = f - lam[:, None, None, None] * o[:, None]
        thin = area < 0.1 * CELLS * CELLS                # hardly any background in the drawing: plain mean instead
        t[thin] = f[thin] - f[thin].mean((1, 2, 3), keepdims=True)
        t = t / np.sqrt((t * t).sum((1, 2, 3), keepdims=True))
        out.append(t)
    return np.concatenate(out)


def build(per_class=PER_CLASS, seed=SEED):
    layers, w = front_end()
    T = templates(layers, w, *drawings(per_class, seed))
    outs = []
    for c in range(len(NAMES)):
        t = T[c * per_class:(c + 1) * per_class]
        n = len(t)
        s = 2 * PAD + 1
        layers.append(dict(name=f"t{c}", op="conv", inputs=["fb"], out=n, kernel=[CELLS, CELLS], stride=1, padding=PAD))
        w[f"t{c}.W"], w[f"t{c}.b"] = t, np.zeros(n)
        layers.append(dict(name=f"m{c}", op="maxpool", inputs=[f"t{c}"], size=[s, s], stride=[s, s]))      # best shift
        layers.append(dict(name=f"r{c}", op="reshape", inputs=[f"m{c}"], shape=[1, 1, n]))
        layers.append(dict(name=f"x{c}", op="maxpool", inputs=[f"r{c}"], size=[1, n], stride=[1, n]))      # best template
        one = np.zeros((len(NAMES), 1))
        one[c, 0] = 1.0
        layers.append(dict(name=f"o{c}", op="dense", inputs=[f"x{c}"], out=len(NAMES)))                    # into slot c
        w[f"o{c}.W"], w[f"o{c}.b"] = one, np.zeros(len(NAMES))
        outs.append(f"o{c}")
    layers.append(dict(name="scores", op="add", inputs=outs))
    return {"layers": layers}, {k: v.astype(np.float32) for k, v in w.items()}


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    net, weights = build()
    with open(os.path.join(out_dir, "network.json"), "w") as fh:
        json.dump(net, fh)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)
