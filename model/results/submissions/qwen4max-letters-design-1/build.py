"""Hand-designed ASL classifier: weights come only from an articulated hand model
written from general knowledge, and from analytic difference-of-Gaussian filters.
Reads no files.  Usage: python3 build.py OUT_DIR"""
"""Articulated 3D hand model + renderer, written from general knowledge of hand anatomy
and of the American Sign Language alphabet.  No data is read here.

Coordinates: x towards the thumb side, y towards the wrist (down), z out of the palm
towards the viewer.  Unit = palm width across the knuckles.
"""
import numpy as np

MCP_X = np.array([0.330, 0.100, -0.130, -0.340])
MCP_Y = np.array([-1.00, -1.055, -1.015, -0.915])
FLEN = np.array([0.98, 1.08, 0.98, 0.79])
FRAD = np.array([0.126, 0.126, 0.116, 0.100])
SEG = np.array([0.44, 0.30, 0.26])
THUMB_BASE = np.array([0.400, -0.250, 0.070])
THUMB_L = np.array([0.42, 0.34])
THUMB_R = np.array([0.160, 0.128])


def dirvec(theta, phi):
    t, p = np.radians(theta), np.radians(phi)
    return np.array([np.sin(t) * np.cos(p), -np.cos(t) * np.cos(p), np.sin(p)])


def rotmat(pitch, yaw, roll):
    a, b, c = np.radians([pitch, yaw, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Obj:
    """one painted object: a union of capsules / balls, with a single outline"""
    __slots__ = ("prims", "z", "shade", "name")

    def __init__(self, prims, z, shade, name=""):
        self.prims, self.z, self.shade, self.name = prims, z, shade, name


def cap(p, q, r0, r1=None):
    return (np.asarray(p, float), np.asarray(q, float), float(r0),
            float(r0 if r1 is None else r1))


def palm_prims():
    """a slab of overlapping balls: projects correctly under any 3D rotation"""
    pr = []
    r = 0.150
    for x in (-0.345, -0.175, 0.0, 0.175, 0.345):
        for y in (-0.880, -0.700, -0.520, -0.340, -0.150):
            for z in (-0.045, 0.045):
                # the palm is a rounded wedge: broad at the knuckles, narrow at the wrist
                t = (y + 0.880) / 0.730
                rr = r * (1.0 - 0.12 * t)
                xx = x * (1.0 - 0.22 * t)
                pr.append(cap((xx, y, z), (xx, y, z), rr))
    # thenar (thumb muscle) bulge and hypothenar bulge
    pr.append(cap((0.355, -0.330, 0.055), (0.300, -0.700, 0.045), 0.175, 0.150))
    pr.append(cap((-0.360, -0.330, 0.0), (-0.330, -0.660, 0.0), 0.130, 0.115))
    return pr


def build_hand(flex, spread, thumb, curl=(-1,) * 4, thumb_front=False,
               palm_shade=0.845, finger_shade=0.865, thumb_shade=0.855,
               wrist_shade=0.815, flen=1.0):
    objs = []
    objs.append(Obj(palm_prims(), 0.0, palm_shade, "palm"))
    objs.append(Obj([cap((0.0, -0.10, -0.02), (0.03, 0.55, -0.03), 0.330, 0.340),
                     cap((0.03, 0.50, -0.04), (0.09, 3.2, -0.05), 0.340, 0.400)],
                    -0.05, wrist_shade, "arm"))
    for j in range(4):
        s = np.radians(spread[j])
        u0 = np.array([np.sin(s), -np.cos(s), 0.0])
        sg = curl[j]
        p = np.array([MCP_X[j], MCP_Y[j], 0.0])
        ang, prims, zs = 0.0, [], []
        for k in range(3):
            ang += flex[j][k]
            a = np.radians(ang)
            d = u0 * np.cos(a) + np.array([0, 0, sg]) * np.sin(a)
            q = p + FLEN[j] * flen * SEG[k] * d
            prims.append(cap(p, q, FRAD[j] * (1 - 0.06 * k), FRAD[j] * (1 - 0.06 * (k + 1))))
            zs += [p[2], q[2]]
            p = q
        objs.append(Obj(prims, float(np.mean(zs)), finger_shade - 0.008 * j, "f%d" % j))
    t1, p1, t2, p2 = thumb
    p = THUMB_BASE.copy()
    prims, zs = [], []
    for k, (th, ph) in enumerate(((t1, p1), (t2, p2))):
        q = p + THUMB_L[k] * dirvec(th, ph)
        prims.append(cap(p, q, THUMB_R[k], THUMB_R[k] * 0.88))
        zs += [p[2], q[2]]
        p = q
    objs.append(Obj(prims, float(np.mean(zs)) + (0.32 if thumb_front else 0.0),
                    thumb_shade, "thumb"))
    return objs


def transform(objs, R, scale):
    out = []
    for o in objs:
        pr = [(R @ p * scale, R @ q * scale, r0 * scale, r1 * scale) for p, q, r0, r1 in o.prims]
        out.append(Obj(pr, o.z * scale, o.shade, o.name))
    return out


def bbox2d(objs):
    xs, ys = [], []
    for o in objs:
        if o.name == "arm":
            continue
        for p, q, r0, r1 in o.prims:
            r = max(r0, r1)
            xs += [p[0] - r, q[0] - r, p[0] + r, q[0] + r]
            ys += [p[1] - r, q[1] - r, p[1] + r, q[1] + r]
    return min(xs), min(ys), max(xs), max(ys)


def obj_field(X, Y, o):
    f = None
    for p, q, r0, r1 in o.prims:
        px, py = p[0], p[1]
        dx, dy = q[0] - px, q[1] - py
        L2 = dx * dx + dy * dy
        if L2 < 1e-10:
            d = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
            v = r0 - d
        else:
            t = np.clip(((X - px) * dx + (Y - py) * dy) / L2, 0.0, 1.0)
            cx, cy = px + t * dx, py + t * dy
            d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            v = (r0 + t * (r1 - r0)) - d
        f = v if f is None else np.maximum(f, v)
    return f


def rasterize(objs, side, cx, cy, n=28, ss=4, bg=0.60, crease=0.50, cw=2, soft=0.55,
              return_mask=False):
    """Photographic look: smooth shaded forms, dark creases only where two parts meet,
    never a drawn outline against the background."""
    m = n * ss
    g = (np.arange(m) + 0.5) / m * side - side / 2.0
    X = np.broadcast_to(cx + g[None, :], (m, m))
    Y = np.broadcast_to(cy + g[:, None], (m, m))
    img = np.full((m, m), bg, np.float32)
    img += (0.030 * (X - cx) / side - 0.045 * (Y - cy) / side).astype(np.float32)
    ident = np.full((m, m), -1, np.int16)
    hand = np.zeros((m, m), bool)
    for i, o in enumerate(sorted(objs, key=lambda a: a.z)):
        f = obj_field(X, Y, o)
        body = f > 0.0
        if not body.any():
            continue
        rel = np.clip(f / 0.15, 0.0, 1.0)
        shade = (o.shade + 0.070 * np.sqrt(rel)
                 + 0.060 * (-0.45 * (X - cx) - 0.60 * (Y - cy)) / side)
        img[body] = shade[body].astype(np.float32)
        ident[body] = i
        hand[body] = o.name != "arm"
    # creases: internal boundaries between two different visible parts
    seam = np.zeros((m, m), bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = ident[max(dy, 0):m + min(dy, 0), max(dx, 0):m + min(dx, 0)]
        b = ident[max(-dy, 0):m + min(-dy, 0), max(-dx, 0):m + min(-dx, 0)]
        d = (a != b) & (a >= 0) & (b >= 0)
        seam[max(dy, 0):m + min(dy, 0), max(dx, 0):m + min(dx, 0)] |= d
        seam[max(-dy, 0):m + min(-dy, 0), max(-dx, 0):m + min(-dx, 0)] |= d
    for _ in range(cw - 1):
        grown = seam.copy()
        grown[1:, :] |= seam[:-1, :]
        grown[:-1, :] |= seam[1:, :]
        grown[:, 1:] |= seam[:, :-1]
        grown[:, :-1] |= seam[:, 1:]
        seam = grown
    img[seam] = 0.45 * img[seam] + 0.55 * crease
    out = np.clip(img, 0.0, 1.0).reshape(n, ss, n, ss).mean(axis=(1, 3))
    if soft > 0:
        k = np.exp(-0.5 * (np.arange(-2, 3) / soft) ** 2)
        k /= k.sum()
        pad = np.pad(out, 2, mode="edge")
        out = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, pad)
        out = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, out)
    out = out.astype(np.float32)
    if return_mask:
        return out, hand.reshape(n, ss, n, ss).mean(axis=(1, 3)) > 0.3
    return out


def render(pose, roll=0.0, scale=1.0, n=28, ss=4, margin=1.10, wrist_pad=0.30,
           yaw=0.0, pitch=0.0, bg=0.60, dx=0.0, dy=0.0, zoom=1.0, return_mask=False):
    objs = transform(build_hand(**pose), rotmat(pitch, yaw, roll), scale)
    x0, y0, x1, y1 = bbox2d(objs)
    y1 += wrist_pad * scale
    side = max(x1 - x0, y1 - y0) * margin * zoom
    cx, cy = (x0 + x1) / 2 + dx * side, (y0 + y1) / 2 + dy * side
    return rasterize(objs, side, cx, cy, n=n, ss=ss, bg=bg, return_mask=return_mask)


# ----------------------------------------------------------------------------------
# helpers for posing: where a curled fingertip ends up, and thumb inverse kinematics
# ----------------------------------------------------------------------------------
def fingertip(j, flexj, spreadj, curlj):
    s = np.radians(spreadj)
    u0 = np.array([np.sin(s), -np.cos(s), 0.0])
    p = np.array([MCP_X[j], MCP_Y[j], 0.0])
    ang = 0.0
    for k in range(3):
        ang += flexj[k]
        a = np.radians(ang)
        p = p + FLEN[j] * SEG[k] * (u0 * np.cos(a) + np.array([0, 0, curlj]) * np.sin(a))
    return p


def thumb_ik(target, bend=(1.0, -0.15, 0.35)):
    """two link IK: return (theta1, phi1, theta2, phi2) putting the thumb tip at target"""
    p0 = THUMB_BASE
    L1, L2 = THUMB_L
    t = np.asarray(target, float)
    v = t - p0
    d = float(np.linalg.norm(v))
    d = min(max(d, abs(L1 - L2) + 1e-3), L1 + L2 - 1e-3)
    u = v / max(np.linalg.norm(v), 1e-9)
    a = (L1 * L1 - L2 * L2 + d * d) / (2 * d)
    h = np.sqrt(max(L1 * L1 - a * a, 0.0))
    n = np.asarray(bend, float)
    w = n - np.dot(n, u) * u
    nw = np.linalg.norm(w)
    w = w / nw if nw > 1e-6 else np.array([0.0, 0.0, 1.0])
    elbow = p0 + a * u + h * w
    out = []
    for dvec in (elbow - p0, t - elbow):
        dvec = dvec / max(np.linalg.norm(dvec), 1e-9)
        theta = np.degrees(np.arctan2(dvec[0], -dvec[1]))
        phi = np.degrees(np.arcsin(np.clip(dvec[2], -1, 1)))
        out += [float(theta), float(phi)]
    return tuple(out)


"""The 24 static ASL letters, described as joint angles / thumb targets from general
knowledge of the alphabet.  Nothing here is measured from data."""
import numpy as np

FIST = (58.0, 95.0, 45.0)      # closed finger, folded away from the camera
HOOK = (10.0, 85.0, 45.0)
FOLD = (72.0, 102.0, 35.0)     # finger folded over the front of the palm
STR = (0.0, 0.0, 0.0)
TIGHT = [5.0, 1.0, -3.5, -8.0]

BEND_F = (0.35, -0.25, 1.0)    # thumb knuckle bulges out in front of the palm
BEND_S = (1.0, -0.20, 0.40)    # thumb knuckle bulges out to the radial side
BEND_U = (0.15, -1.0, 0.20)    # thumb extended, knuckle upwards

P, O = {}, {}


def add(letter, fingers, spread, tip, bend, curl, roll, yaw=0.0, front=False, pitch=0.0):
    P[letter] = dict(flex=np.array(fingers, float), spread=np.array(spread, float),
                     thumb=thumb_ik(tip, bend), curl=tuple(curl), thumb_front=front)
    O[letter] = dict(roll=roll, yaw=yaw, pitch=pitch)


# A - fist, thumb straight up beside the index
add('A', [FIST] * 4, TIGHT, (0.56, -1.02, 0.16), BEND_S, [-1] * 4, -9.0)
# B - four fingers straight and together, thumb laid across the palm
add('B', [STR] * 4, [2.0, 0.5, -1.0, -2.5], (-0.24, -0.50, 0.30), BEND_F, [1] * 4, -3.0, front=True)
# C - the whole hand curved into a C, seen from the radial side
_cflex = (45.0, 55.0, 40.0)
_ctip = fingertip(0, _cflex, TIGHT[0], 1) + np.array([0.02, 0.62, 0.02])
add('C', [_cflex] * 4, TIGHT, _ctip, (0.5, -0.5, 0.7), [1] * 4, -12.0, yaw=62.0, front=True)
# D - index up, the other tips meeting the thumb in a loop
_dflex = (62.0, 72.0, 45.0)
_dtip = fingertip(1, _dflex, TIGHT[1], 1) + np.array([0.02, 0.04, -0.02])
add('D', [STR, _dflex, _dflex, _dflex], TIGHT, _dtip, BEND_S, [1] * 4, -4.0, yaw=52.0, front=True)
# E - fingertips curled down onto the tucked thumb
add('E', [(70.0, 100.0, 40.0)] * 4, TIGHT, (-0.10, -0.74, 0.26), BEND_F, [1] * 4, -6.0)
# F - index tip to thumb tip, three fingers up and spread
_fflex = (52.0, 68.0, 44.0)
_ftip = fingertip(0, _fflex, 8.0, 1)
add('F', [_fflex, STR, STR, STR], [8.0, 6.0, -4.0, -16.0], _ftip, BEND_S, [1] * 4, -5.0,
    yaw=46.0, front=True)
# G - index pointing sideways, thumb resting on the closed fingers (back of hand showing)
add('G', [STR, FIST, FIST, FIST], TIGHT, (0.04, -0.80, 0.30), BEND_F, [-1] * 4, 90.0,
    yaw=180.0, front=True)
# H - index and middle pointing sideways together
add('H', [STR, STR, FIST, FIST], [3.0, -1.0, -3.5, -8.0], (0.04, -0.80, 0.30), BEND_F,
    [-1] * 4, 90.0, yaw=180.0, front=True)
# I - little finger alone up
add('I', [FIST, FIST, FIST, STR], TIGHT, (0.04, -0.80, 0.30), BEND_F, [-1] * 4, -7.0, front=True)
# K - index and middle up in a V, thumb tip between them
add('K', [STR, STR, FIST, FIST], [16.0, -6.0, -3.5, -8.0], (0.24, -1.14, 0.28), BEND_S,
    [-1] * 4, -4.0, front=True)
# L - index up, thumb out sideways
add('L', [STR, FIST, FIST, FIST], TIGHT, (1.18, -0.52, 0.10), BEND_U, [-1] * 4, -4.0)
# M - three fingers folded over the tucked thumb
add('M', [FOLD, FOLD, FOLD, FIST], TIGHT, (0.00, -0.82, 0.16), BEND_S, [1, 1, 1, -1], -7.0)
# N - two fingers folded over the tucked thumb
add('N', [FOLD, FOLD, FIST, FIST], TIGHT, (0.16, -0.82, 0.16), BEND_S, [1, 1, -1, -1], -7.0)
# O - all the tips curved round onto the thumb, a closed ring
_oflex = (60.0, 70.0, 50.0)
_otip = fingertip(0, _oflex, TIGHT[0], 1) + np.array([0.0, 0.03, -0.03])
add('O', [_oflex] * 4, TIGHT, _otip, BEND_S, [1] * 4, -8.0, yaw=55.0, front=True)
# P - K turned to point downwards
add('P', [STR, (30.0, 55.0, 30.0), FIST, FIST], [16.0, -6.0, -3.5, -8.0],
    (0.24, -1.14, 0.28), BEND_S, [-1] * 4, 200.0, yaw=180.0, front=True)
# Q - G turned to point downwards, thumb pinching towards the index
add('Q', [STR, FIST, FIST, FIST], TIGHT, (0.66, -0.92, 0.22), BEND_S, [-1] * 4, 192.0,
    yaw=180.0, front=True)
# R - index and middle crossed
add('R', [STR, STR, FIST, FIST], [-10.0, 18.0, -3.5, -8.0], (0.04, -0.80, 0.30), BEND_F,
    [-1] * 4, -3.0, front=True)
# S - fist with the thumb laid across the front of the fingers
add('S', [FIST] * 4, TIGHT, (-0.26, -0.90, 0.32), BEND_F, [-1] * 4, -7.0, front=True)
# T - fist turned side on, thumb pushed up between index and middle
add('T', [FIST] * 4, TIGHT, (1.02, -0.62, 0.16), BEND_U, [-1] * 4, 90.0, yaw=180.0)
# U - index and middle up, pressed together
add('U', [STR, STR, FIST, FIST], [2.0, -2.0, -3.5, -8.0], (0.04, -0.80, 0.30), BEND_F,
    [-1] * 4, -3.0, front=True)
# V - index and middle up and spread
add('V', [STR, STR, FIST, FIST], [19.0, -5.0, -3.5, -8.0], (0.04, -0.80, 0.30), BEND_F,
    [-1] * 4, -3.0, front=True)
# W - index, middle and ring up and spread
add('W', [STR, STR, STR, FIST], [18.0, 3.0, -12.0, -8.0], (0.04, -0.80, 0.30), BEND_F,
    [-1] * 4, -3.0, front=True)
# X - index bent into a hook, seen side on
add('X', [HOOK, FIST, FIST, FIST], TIGHT, (0.04, -0.80, 0.30), BEND_F, [1, -1, -1, -1],
    -6.0, yaw=68.0, front=True)
# Y - thumb and little finger out, the rest closed
add('Y', [FIST, FIST, FIST, STR], [5.0, 1.0, -3.5, -22.0], (1.18, -0.52, 0.10), BEND_U,
    [-1] * 4, -4.0)

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"


# ==================================================================================
# feature pipeline + template bank + network assembly
# ==================================================================================
import json
import os
import sys

import numpy as np


def gauss1(sigma, r):
    t = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * (t / sigma) ** 2)
    return g / g.sum()


def gauss2(sigma, r):
    g = gauss1(sigma, r)
    return np.outer(g, g)


def zerosum(k):
    return k - k.mean()


def im2col(x, kh, kw, stride, padding):
    """x: (N,C,H,W) -> (N, C*kh*kw, ho*wo), same ordering as the scorer"""
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
                     for dy in range(kh) for dx in range(kw)], axis=2)
    return cols.reshape(n, c * kh * kw, ho * wo)


def convb(x, W, stride=1, padding=0):
    o = W.shape[0]
    kh, kw = W.shape[2], W.shape[3]
    cols = im2col(x, kh, kw, stride, padding)
    ho = int(np.sqrt(cols.shape[2]))
    return (cols.transpose(0, 2, 1) @ W.reshape(o, -1).T).transpose(0, 2, 1).reshape(
        x.shape[0], o, ho, ho)


def normb(x):
    mu = x.mean(axis=(2, 3), keepdims=True)
    v = ((x - mu) ** 2).mean(axis=(2, 3), keepdims=True)
    return (x - mu) / np.sqrt(v + 1e-5)


R1 = 8                       # radius of the first conv kernel (17 x 17)
SIG_F, SIG_M, SIG_C = 0.9, 2.0, 4.0
BLUR_R = 2                   # radius of the mixing kernel (5 x 5)
BLUR_SIG = 1.0


def make_filters():
    dog_f = zerosum(gauss2(SIG_F, R1) - gauss2(SIG_M, R1))
    dog_c = zerosum(gauss2(SIG_M, R1) - gauss2(SIG_C, R1))
    W1 = np.stack([dog_f, dog_c, -dog_f, -dog_c])[:, None]
    B = gauss2(BLUR_SIG, BLUR_R)
    mix = np.zeros((2, 4, 2 * BLUR_R + 1, 2 * BLUR_R + 1))
    edge = np.zeros_like(mix)
    for i, src in enumerate(((0, 2), (1, 3))):
        mix[i, src[0]] = B
        mix[i, src[1]] = -B
        edge[i, src[0]] = B
        edge[i, src[1]] = B
    return W1, mix, edge


W1, WMIX, WEDGE = make_filters()


def features(imgs):
    """the network's front end, batched: (N,28,28) -> signed (N,2,28,28), edge (N,2,28,28)"""
    x = normb(imgs[:, None].astype(np.float64))
    f = np.maximum(convb(x, W1, 1, R1), 0.0)
    s = normb(convb(f, WMIX, 1, BLUR_R))
    eraw = convb(f, WEDGE, 1, BLUR_R)          # non-negative edge energy
    return s, normb(eraw), eraw


# ----------------------------------------------------------------------------------
# template bank: every entry is one generated drawing, normalised by itself alone
# ----------------------------------------------------------------------------------
NPOSE = 2
ROLLS = (-12.0, -6.0, 0.0, 6.0, 12.0)
MARGINS = (1.04, 1.12, 1.22, 1.32)
CANVAS = 36                                  # render big, then take 28x28 windows
OFFS = (0, 2, 4, 6, 8)                       # window offsets -> shifts of -4..+4 px
NBASE = NPOSE * len(ROLLS) * len(MARGINS)
NPER = NBASE * len(OFFS) ** 2
W_SIGNED, W_EDGE = 1.0, 1.0
BASELINE = 0.0     # class score = best match over the class's own templates


def variants(letter):
    """two hand-set pose variants of the same letter"""
    base, opt = P[letter], O[letter]
    yield base, opt
    v = dict(base)
    v['flex'] = base['flex'] + 6.0
    v['spread'] = base['spread'] * 1.25
    v['flen'] = 0.92
    o = dict(opt)
    o['yaw'] = opt['yaw'] + 7.0
    o['pitch'] = opt['pitch'] + 5.0
    yield v, o


def hand_window(mask, grow=4, sig=1.0):
    """a soft window covering this drawing's own hand, a little dilated.  The wrist and
    sleeve carry no letter information and the background is not comparable between a
    drawing and a photograph, so neither takes part in the match."""
    m = mask.copy()
    for _ in range(grow):
        g = m.copy()
        g[1:, :] |= m[:-1, :]
        g[:-1, :] |= m[1:, :]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        m = g
    k = np.exp(-0.5 * (np.arange(-3, 4) / sig) ** 2)
    k /= k.sum()
    w = m.astype(np.float64)
    for ax in (0, 1):
        w = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), ax, w)
    return np.clip(w, 0.0, 1.0)


def l2(v):
    n = np.sqrt((v ** 2).sum(axis=(2, 3), keepdims=True))
    return v / np.maximum(n, 1e-9) / np.sqrt(v.shape[1])


def unit_signed(v):
    """Signed band-pass: a drawing's background is smooth, so its response there is
    already ~0.  Zero-mean over the frame, unit norm per channel -> the match is a pure
    cosine (the kernel covers the whole frame, so the input norm is the same for all)."""
    v = v - v.mean(axis=(2, 3), keepdims=True)
    return l2(v)


def unit_edge(v, w):
    """Edge energy is non-negative.  Subtracting its mean would make the whole background
    negative, and background x background would then score every sparse template highly.
    So keep it non-negative and cut it off outside this drawing's own hand: the match is
    then edge overlap, with a penalty wherever the drawing has an edge and the photo does
    not, and no reward at all from the background."""
    return l2(v * w[:, None])


def build_templates():
    S, E = [], []
    for letter in LETTERS:
        for v, o in variants(letter):
            for dr in ROLLS:
                for mg in MARGINS:
                    big, msk = render(v, roll=o['roll'] + dr, yaw=o['yaw'],
                                      pitch=o['pitch'], margin=mg, n=CANVAS,
                                      zoom=CANVAS / 28.0, return_mask=True)
                    wins = np.stack([big[oy:oy + 28, ox:ox + 28]
                                     for oy in OFFS for ox in OFFS])
                    wnds = np.stack([hand_window(msk[oy:oy + 28, ox:ox + 28])
                                     for oy in OFFS for ox in OFFS])
                    s, e, eraw = features(wins)
                    S.append(unit_signed(s) * W_SIGNED)
                    E.append(unit_edge(eraw, wnds) * W_EDGE)
        print("built", letter, flush=True)
    return np.concatenate(S), np.concatenate(E)


def main(outdir):
    S, E = build_templates()
    T = S.shape[0]
    assert T == 24 * NPER, (T, 24 * NPER)
    k = 28
    layers = [
        {"name": "n0", "op": "norm", "inputs": ["image"]},
        {"name": "f1", "op": "conv", "inputs": ["n0"], "out": 4,
         "kernel": [2 * R1 + 1] * 2, "stride": 1, "padding": R1},
        {"name": "r1", "op": "relu", "inputs": ["f1"]},
        {"name": "ms", "op": "conv", "inputs": ["r1"], "out": 2,
         "kernel": [2 * BLUR_R + 1] * 2, "stride": 1, "padding": BLUR_R},
        {"name": "me", "op": "conv", "inputs": ["r1"], "out": 2,
         "kernel": [2 * BLUR_R + 1] * 2, "stride": 1, "padding": BLUR_R},
        {"name": "ns", "op": "norm", "inputs": ["ms"]},
        {"name": "ne", "op": "norm", "inputs": ["me"]},
        {"name": "ts", "op": "conv", "inputs": ["ns"], "out": T,
         "kernel": [k, k], "stride": 1, "padding": 0},
        {"name": "te", "op": "conv", "inputs": ["ne"], "out": T,
         "kernel": [k, k], "stride": 1, "padding": 0},
        {"name": "rs", "op": "relu", "inputs": ["ts"]},
        {"name": "re", "op": "relu", "inputs": ["te"]},
        {"name": "sc", "op": "add", "inputs": ["rs", "re"]},
        {"name": "g1", "op": "reshape", "inputs": ["sc"], "shape": [24, NPER, 1]},
        {"name": "pmax", "op": "maxpool", "inputs": ["g1"], "size": [NPER, 1],
         "stride": [NPER, 1]},
        {"name": "pavg", "op": "avgpool", "inputs": ["g1"], "size": [NPER, 1],
         "stride": [NPER, 1]},
        {"name": "base", "op": "conv", "inputs": ["pavg"], "out": 24,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "fin", "op": "add", "inputs": ["pmax", "base"]},
        {"name": "out", "op": "reshape", "inputs": ["fin"], "shape": [24]},
    ]
    w = {
        "f1.W": W1, "f1.b": np.zeros(4),
        "ms.W": WMIX, "ms.b": np.zeros(2),
        "me.W": WEDGE, "me.b": np.zeros(2),
        "ts.W": S.reshape(T, 2, k, k), "ts.b": np.zeros(T),
        "te.W": E.reshape(T, 2, k, k), "te.b": np.zeros(T),
        "base.W": -BASELINE * np.eye(24).reshape(24, 24, 1, 1),
        "base.b": np.zeros(24),
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    np.savez(os.path.join(outdir, "weights.npz"),
             **{k: v.astype(np.float32) for k, v in w.items()})
    print("templates", T, "weights", sum(v.size for v in w.values()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
