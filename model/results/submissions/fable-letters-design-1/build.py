#!/usr/bin/env python3
"""Builds a hand-designed sign-language-letter network: python3 build.py OUT_DIR

No data is read.  Every weight comes from this file:
  * the first conv layers are Gaussian-derivative filters given by formulas,
  * the last dense layer holds templates.  Each template is the feature map of ONE image rendered by the
    3D hand model below (palm, fingers and thumb as chains of spheres, joint angles typed from general
    knowledge of hand anatomy and of the ASL manual alphabet), normalised by itself alone.
"""
import sys, os, json, math
import numpy as np

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

# ----------------------------------------------------------------------------------------------------------------
# 1.  HAND MODEL (centimetres).  Local frame of a right hand: x towards the thumb side, y towards the fingertips,
#     z out of the palm.  Wrist centre at the origin.
# ----------------------------------------------------------------------------------------------------------------
MCP = np.array([[3.2, 9.0, 0.0], [1.1, 9.6, 0.0], [-1.0, 9.2, 0.0], [-2.9, 8.3, 0.0]])      # index, middle, ring, pinky
FLEN = np.array([[4.2, 2.4, 1.9], [4.6, 2.8, 2.0], [4.3, 2.6, 2.0], [3.4, 1.9, 1.8]])     # phalanx lengths
FRAD = np.array([[1.06, 1.00, 0.92, 0.82], [1.06, 1.00, 0.92, 0.82], [1.00, 0.95, 0.87, 0.78], [0.90, 0.85, 0.78, 0.70]])
FN = (10, 7, 6)                                # spheres per phalanx
PALM0 = np.array([[1.8, 0.8, 0.0], [0.6, 0.6, 0.0], [-0.6, 0.6, 0.0], [-1.7, 0.8, 0.0]])     # wrist ends of metacarpals
PALMR = np.array([1.45, 1.45, 1.4, 1.3])
THUMB0 = np.array([2.0, 1.5, 1.0])             # thumb base (CMC joint)
TLEN = np.array([4.4, 3.2, 2.4])
TRAD = np.array([1.60, 1.10, 1.00, 0.88])
TN = (9, 8, 7)

CURL = (82.0, 103.0, 68.0)        # finger folded into a fist: (MCP, PIP, DIP) flexion in degrees
EXT = (4.0, 4.0, 3.0)             # straight finger
DRAPE = (68.0, 88.0, 42.0)        # finger folded forward and down over the thumb (M, N)

# thumb shapes: target points for the MCP joint, the IP joint and the tip (bone lengths are re-imposed in code)
TH_SIDE = [(4.4, 5.4, 1.8), (4.9, 8.5, 2.6), (4.7, 10.8, 3.0)]        # upright along the side of the fist (A)
TH_ACROSS = [(3.7, 5.0, 2.8), (2.4, 7.0, 4.7), (0.3, 7.6, 5.0)]       # across the front of the curled fingers (S)
TH_RING = [(3.6, 5.0, 2.8), (1.8, 6.8, 4.6), (-0.4, 7.4, 4.9)]        # across, holding the ring finger down (U, V, R)
TH_PINKY = [(3.4, 5.0, 2.6), (1.2, 6.4, 4.0), (-1.6, 6.9, 4.4)]       # across, holding the pinky down (W)
TH_PALM = [(3.4, 5.2, 2.2), (1.6, 6.6, 3.0), (-0.4, 7.2, 3.0)]        # folded flat across the palm (B, E)
TH_UNDER = [(3.2, 5.2, 2.4), (1.2, 6.6, 2.9), (-1.0, 7.2, 2.9)]       # tucked under draped fingers (M)
TH_UNDER2 = [(3.2, 5.2, 2.4), (1.6, 6.8, 3.0), (0.0, 7.6, 3.0)]       # tucked under two fingers (N)
TH_OUT = [(4.6, 4.6, 1.0), (7.6, 5.6, 1.0), (9.9, 6.2, 1.0)]          # stretched out sideways (L)
TH_OUTUP = [(4.6, 4.8, 1.0), (7.4, 6.4, 1.0), (9.4, 7.8, 1.0)]        # stretched out sideways and up (Y)
TH_C = [(3.0, 3.8, 3.2), (3.0, 5.4, 5.8), (2.8, 7.4, 7.2)]            # opposed, curved forward (C)
TH_O = [(3.35, 2.85, 5.0), (3.05, 5.1, 7.26), (2.9, 7.2, 6.1)]            # opposed, tip meeting the fingertips (O)
TH_D = [(3.3, 4.1, 4.3), (1.86, 5.54, 6.8), (1.3, 6.6, 5.5)]            # tip meeting the middle fingertip (D)
TH_F = [(3.2, 4.0, 3.6), (3.2, 5.6, 6.0), (3.1, 7.4, 6.4)]            # tip meeting the index fingertip (F)
TH_K = [(3.4, 5.2, 2.6), (2.8, 8.0, 4.2), (1.9, 10.4, 4.6)]           # tip on the middle finger (K, P)
TH_G = [(3.6, 5.0, 2.8), (3.6, 8.0, 4.2), (3.6, 10.4, 4.4)]           # parallel to the index, in front of the palm (G, Q)
TH_Q = [(4.5, 5.2, 1.8), (5.4, 8.2, 2.6), (5.6, 10.6, 3.0)]           # beside the index and pointing the same way (Q)
TH_QD = [(3.2, 4.0, 3.8), (3.4, 5.0, 6.8), (3.4, 5.6, 9.0)]          # pointing straight out of the palm (Q)
TH_T = [(4.2, 5.4, 1.6), (6.0, 7.6, 2.4), (7.6, 8.6, 3.0)]            # standing away from the index, at right angles to it (T)

# pose table: fingers = (spread towards the thumb side in degrees, MCP, PIP, DIP flexion) for index, middle, ring, pinky
#   base  = orientation of the hand: 'up' palm to the viewer, fingers up; 'side' (index pointing right, back of the
#           hand to the viewer); 'left' (fingers pointing left, palm down, thumb side to the viewer);
#           'down' (fingers pointing down, back of the hand to the viewer)
#   view  = (yaw, pitch, roll) added to the base, and their jitter half-widths, in degrees
#   arm   = direction the forearm leaves the wrist in the picture (x right, y up), blended with the hand's own axis
P = {}
def pose(name, fingers, thumb, base='up', view=(0, 0, -5), jit=(16, 12, 12), arm=(0.0, -1.0), armw=0.5):
    P[name] = dict(fingers=np.array(fingers, float), thumb=np.array(thumb, float), base=base, view=view, jit=jit, arm=arm, armw=armw)

C4 = lambda s=0.0: (s,) + CURL
E4 = lambda s=0.0: (s,) + EXT
FIST = dict(view=(-5, 0, -14), jit=(18, 12, 14), arm=(-0.25, -0.97))        # fists lean a little, the arm comes from below
TURN = dict(view=(-28, 0, -5), jit=(25, 12, 12))                            # hand turned so that the folded fingers bulge to the left
pose('A', [C4(), C4(), C4(), C4()], TH_SIDE, **FIST)
pose('B', [E4(-2), E4(0), E4(2), E4(5)], TH_PALM, view=(0, 0, -5), jit=(18, 12, 12))
pose('C', [(0, 42, 58, 38), (0, 42, 58, 38), (0, 42, 58, 38), (-2, 45, 58, 38)], TH_C, view=(55, 0, -25), jit=(20, 12, 12), arm=(-0.5, -0.87))
pose('D', [E4(0), (0, 70, 95, 50), (0, 72, 95, 50), (-2, 74, 95, 50)], TH_D, **TURN)
pose('E', [(0, 48, 108, 78), (0, 48, 108, 78), (0, 48, 108, 78), (0, 50, 108, 78)], TH_PALM, **FIST)
pose('F', [(0, 45, 70, 30), E4(5), E4(-14), E4(-33)], TH_F, view=(30, 0, -5), jit=(22, 12, 12))
pose('G', [E4(0), C4(), C4(), C4()], TH_G, base='side', view=(-42, 0, 12), jit=(22, 18, 14), arm=(-0.55, -0.83), armw=0.8)
pose('H', [E4(-3), E4(2), C4(), C4()], TH_G, base='side', view=(-42, 0, 12), jit=(22, 18, 14), arm=(-0.55, -0.83), armw=0.8)
pose('I', [C4(), C4(), C4(), E4(-8)], TH_PALM, view=(-10, 0, -8), jit=(22, 12, 14))
pose('K', [E4(3), (-5, 75, 8, 5), C4(), C4()], TH_K, view=(-35, 0, -5), jit=(25, 12, 12))
pose('L', [E4(0), C4(), C4(), C4()], TH_OUT, view=(0, 0, -5), jit=(18, 12, 12))
pose('M', [(0,) + DRAPE, (0,) + DRAPE, (0,) + DRAPE, C4()], TH_UNDER, **FIST)
pose('N', [(0,) + DRAPE, (0,) + DRAPE, C4(), C4()], TH_UNDER2, **FIST)
pose('O', [(0, 50, 75, 45), (0, 50, 75, 45), (0, 52, 75, 45), (-2, 56, 75, 45)], TH_O, view=(60, 0, -15), jit=(22, 12, 12), arm=(-0.3, -0.95))
pose('P', [(3, 30, 4, 3), (-5, 85, 8, 5), C4(), C4()], TH_K, base='left', view=(0, 45, -20), jit=(25, 22, 15), arm=(0.2, -0.98), armw=0.9)
pose('Q', [E4(0), C4(), C4(), C4()], TH_Q, base='down', view=(0, -35, -25), jit=(25, 20, 15), arm=(0.25, -0.97), armw=0.9)
pose('R', [(-7, 5, 4, 3), (9, 18, 5, 4), C4(), C4()], TH_RING, **TURN)
pose('S', [C4(), C4(), C4(), C4()], TH_ACROSS, **FIST)
pose('T', [E4(0), C4(), C4(), C4()], TH_T, base='side', view=(-25, 0, 0), jit=(22, 15, 14), arm=(-0.2, -0.98), armw=0.85)
pose('U', [E4(-3), E4(2), C4(), C4()], TH_RING, **TURN)
pose('V', [E4(13), E4(-11), C4(), C4()], TH_RING, view=(-15, 0, -5), jit=(22, 12, 12))
pose('W', [E4(16), E4(0), E4(-16), C4()], TH_PINKY, view=(-5, 0, -5), jit=(20, 12, 12))
pose('X', [(0, 15, 85, 55), C4(), C4(), C4()], TH_ACROSS, view=(55, 0, -18), jit=(25, 12, 14), arm=(-0.35, -0.94))
pose('Y', [C4(), C4(), C4(), E4(-22)], TH_OUTUP, view=(0, 0, -5), jit=(18, 12, 14))

LOOSE = set('IKLRUVWXYGHPQT')

BASES = {
    'up': np.eye(3),
    'side': np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),     # columns: images of local x, y, z
    'left': np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
    'down': np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
}


def rot_batch(yaw, pitch, roll):
    """R = Rz(roll) Rx(pitch) Ry(yaw); angles in radians, arrays (n,)."""
    n = len(yaw); cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.zeros((n, 3, 3)); Ry[:, 0, 0] = cy; Ry[:, 0, 2] = sy; Ry[:, 1, 1] = 1; Ry[:, 2, 0] = -sy; Ry[:, 2, 2] = cy
    Rx = np.zeros((n, 3, 3)); Rx[:, 0, 0] = 1; Rx[:, 1, 1] = cp; Rx[:, 1, 2] = -sp; Rx[:, 2, 1] = sp; Rx[:, 2, 2] = cp
    Rz = np.zeros((n, 3, 3)); Rz[:, 0, 0] = cr; Rz[:, 0, 1] = -sr; Rz[:, 1, 0] = sr; Rz[:, 1, 1] = cr; Rz[:, 2, 2] = 1
    return Rz @ Rx @ Ry


def chain(p0, p1, r0, r1, n, skip_first=False):
    """Spheres from p0 to p1 (n, 3) with radii going from r0 to r1.  Returns centres (B, n, 3), radii (B, n)."""
    t = np.linspace(0.0, 1.0, n)
    if skip_first: t = t[1:]
    c = p0[:, None, :] + t[None, :, None] * (p1 - p0)[:, None, :]
    r = r0[:, None] + t[None, :] * (r1 - r0)[:, None]
    return c, r


def unit(v):
    return v / np.sqrt((v * v).sum(-1, keepdims=True))


def hand_spheres(fing, thumb, R, armdir, flen_s, frad_s, palm_s):
    """fing (B, 4, 4) degrees; thumb (B, 3, 3) target points; R (B, 3, 3); armdir (B, 3) world direction of forearm;
    flen_s, frad_s, palm_s (B,) shape scales.  Returns world centres (B, M, 3), radii (B, M), core mask (M,) and the
    index where the small (finger) spheres end."""
    B = len(fing); one = np.ones(B)
    Cs, Rs = [], []
    xs = palm_s[:, None] * np.array([1.0, 0.0, 0.0])[None, :]           # scale palm width (x only)
    def px(p):                                                           # apply palm-width scale to a local point
        q = np.broadcast_to(np.asarray(p, float), (B, 3)).copy(); q[:, 0] *= palm_s; return q
    # fingers
    for f in range(4):
        a = np.deg2rad(fing[:, f, :])
        s, c = np.sin(a[:, 0]), np.cos(a[:, 0])
        p = px(MCP[f]); cum = np.zeros(B); mids = []
        for j in range(3):
            cum = cum + a[:, j + 1]
            d = np.stack([s * np.cos(cum), c * np.cos(cum), np.sin(cum)], 1)
            q = p + d * (FLEN[f, j] * flen_s)[:, None]
            cc, rr = chain(p, q, FRAD[f, j] * frad_s, FRAD[f, j + 1] * frad_s, FN[j], skip_first=(j > 0))
            Cs.append(cc); Rs.append(rr); mids.append((p + q) / 2); p = q
        cc, rr = chain(mids[0], mids[2], 0.9 * FRAD[f, 1] * frad_s, 0.9 * FRAD[f, 2] * frad_s, 5)
        Cs.append(cc); Rs.append(rr)
    # thumb: bone directions from the target points, bone lengths from anatomy
    p = px(THUMB0); prev = px(THUMB0)
    tp = thumb.copy(); tp[:, :, 0] *= palm_s[:, None]
    for j in range(3):
        d = unit(tp[:, j, :] - prev)
        q = p + d * TLEN[j] * (flen_s if j > 0 else one)[:, None]
        cc, rr = chain(p, q, TRAD[j] * (frad_s if j > 0 else one), TRAD[j + 1] * frad_s, TN[j], skip_first=(j > 0))
        if j == 0:
            big_c, big_r = [cc], [rr]                                    # thumb metacarpal / thenar counts as palm
        else:
            Cs.append(cc); Rs.append(rr)
        prev = tp[:, j, :]; p = q
    n_small = sum(c.shape[1] for c in Cs)
    # palm: four metacarpal slabs and the heel of the hand on the pinky side
    for f in range(7):
        i, j = f // 2, (f + 1) // 2
        cc, rr = chain(px((PALM0[i] + PALM0[j]) / 2), px((MCP[i] + MCP[j]) / 2), (PALMR[i] + PALMR[j]) / 2 * one, (PALMR[i] + PALMR[j]) / 2 * one, 12)
        big_c.append(cc); big_r.append(rr)
    cc, rr = chain(px([-2.1, 1.2, 0.4]), px([-3.0, 6.0, 0.3]), 1.45 * one, 1.35 * one, 7)
    big_c.append(cc); big_r.append(rr)
    Cl = np.concatenate(Cs + big_c, 1); Rl = np.concatenate(Rs + big_r, 1)
    Cw = np.einsum('bij,bmj->bmi', R, Cl)
    # forearm: three parallel chains leaving the wrist along armdir (an oval cross-section, wider than thick)
    xw = R[:, :, 0] + 0.05 * R[:, :, 1]                                 # the arm is broad across the hand's x axis ...
    xw = unit(xw - (xw * armdir).sum(1, keepdims=True) * armdir)        # ... and at right angles to its own direction
    arm_c, arm_r = [], []
    for sgn in (-1.0, 0.0, 1.0):
        a0 = np.einsum('bij,j->bi', R, np.array([0.0, 0.8, 0.0])) + sgn * 1.2 * palm_s[:, None] * xw
        a1 = a0 + 26.0 * armdir + sgn * 0.5 * xw
        cc, rr = chain(a0, a1, 1.9 * one, 2.7 * one, 22)
        arm_c.append(cc); arm_r.append(rr)
    Cw = np.concatenate([Cw] + arm_c, 1); Rl = np.concatenate([Rl] + arm_r, 1)
    core = np.ones(Cw.shape[1], bool)
    n_hand = Cl.shape[1]
    core[n_hand:] = False
    return Cw, Rl, core, n_small


def sample_params(letter, n, rng):
    """Random variations of one letter: joint angles, thumb, view, arm direction, hand proportions."""
    ps = P[letter]
    fing = np.broadcast_to(ps['fingers'], (n, 4, 4)).copy()
    if letter in LOOSE:
        u = rng.uniform(0, 1, n)
        for f in range(4):
            if tuple(ps['fingers'][f][1:]) == CURL:
                fing[:, f, 1] = CURL[0] - 34.0 * u
    fing[:, :, 0] += rng.normal(0, 2.5, (n, 4))
    fing[:, :, 1:] += rng.normal(0, 3.0, (n, 4, 3)) + rng.normal(0, 5.0, (n, 1, 3))      # mostly the fingers move together
    fing[:, :, 1:] = np.maximum(fing[:, :, 1:], 0.0)
    thumb = np.broadcast_to(ps['thumb'], (n, 3, 3)) + rng.normal(0, 0.35, (n, 3, 3)) + rng.normal(0, 0.3, (n, 1, 3))
    v, j = ps['view'], ps['jit']
    ang = np.deg2rad(np.array(v)[None, :] + rng.uniform(-1, 1, (n, 3)) * np.array(j)[None, :])
    R = rot_batch(ang[:, 0], ang[:, 1], ang[:, 2]) @ BASES[ps['base']][None]
    own = np.einsum('bij,j->bi', R, np.array([0.0, -1.0, 0.0]))
    tgt = np.array([ps['arm'][0], ps['arm'][1], 0.0])[None, :] + rng.normal(0, 0.18, (n, 3)) * np.array([1.0, 1.0, 0.5])
    armdir = unit((1 - ps['armw']) * own + ps['armw'] * unit(tgt))
    flen_s = rng.uniform(0.92, 1.08, n); frad_s = rng.uniform(0.95, 1.15, n); palm_s = rng.uniform(0.94, 1.06, n)
    return fing, thumb, R, armdir, flen_s, frad_s, palm_s


# ----------------------------------------------------------------------------------------------------------------
# 2.  RENDERER: orthographic depth buffer of the spheres, shaded from the depth gradient, on a flat background
# ----------------------------------------------------------------------------------------------------------------
RES = 56


def splat(zbuf, u, v, z, r, w):
    """u, v, z, r: (B, M) in pixel units; zbuf (B*RES*RES,).  Max-depth splat with window half-size w."""
    B, M = u.shape
    iu, iv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    off = np.arange(-w, w + 1)
    pu = iu[:, :, None] + off[None, None, :]                        # (B, M, K)
    pv = iv[:, :, None] + off[None, None, :]
    dx = (pu + 0.5 - u[:, :, None]); dy = (pv + 0.5 - v[:, :, None])
    h2 = (r * r)[:, :, None, None] - dy[:, :, :, None] ** 2 - dx[:, :, None, :] ** 2        # (B, M, Kv, Ku)
    ok = (h2 > 0) & ((pv >= 0) & (pv < RES))[:, :, :, None] & ((pu >= 0) & (pu < RES))[:, :, None, :]
    idx = (np.arange(B)[:, None, None, None] * RES + pv[:, :, :, None]) * RES + pu[:, :, None, :]
    zz = z[:, :, None, None] + np.sqrt(np.maximum(h2, 0.0))
    np.maximum.at(zbuf, idx[ok], zz[ok])


def box_blur(a, k):
    """Mean over a (2k+1)^2 window with zero padding, a: (B, H, W)."""
    B, H, W = a.shape
    p = np.zeros((B, H + 2 * k, W + 2 * k)); p[:, k:k + H, k:k + W] = a
    s = np.zeros((B, H + 2 * k, W))
    for d in range(2 * k + 1): s += p[:, :, d:d + W]
    o = np.zeros((B, H, W))
    for d in range(2 * k + 1): o += s[:, d:d + H, :]
    return o


def render(Cw, Rl, core, n_small, armdir, rng, full=False):
    """Returns images (B, 28, 28) in 0..1."""
    B, M, _ = Cw.shape
    # framing: a square around the hand (fingers, thumb, palm).  A long hand fills the frame and may lose its wrist
    # end; a compact hand (a fist) leaves some room.  The side of the hand away from the arm keeps a small margin,
    # so the wrist or arm runs out through the frame edge.
    xc, yc, rc = Cw[:, core, 0], Cw[:, core, 1], Rl[:, core]
    x0, x1 = (xc - rc).min(1), (xc + rc).max(1); y0, y1 = (yc - rc).min(1), (yc + rc).max(1)
    ext_x, ext_y = x1 - x0, y1 - y0
    frame = np.maximum(np.maximum(ext_x, ext_y) + 1.5, 13.0) * rng.uniform(0.88, 1.10, B)
    slack_x, slack_y = (frame - ext_x) / 2, (frame - ext_y) / 2
    marg = rng.uniform(0.02, 0.12, B) * frame
    an = np.sqrt(armdir[:, 0] ** 2 + armdir[:, 1] ** 2) + 1e-9
    ax_, ay_ = armdir[:, 0] / an, armdir[:, 1] / an
    hx = -ax_ * (slack_x - marg) + (1 - np.abs(ax_)) * rng.uniform(-1, 1, B) * (0.05 * frame + 0.3 * np.maximum(slack_x, 0))
    hy = -ay_ * (slack_y - marg) + (1 - np.abs(ay_)) * rng.uniform(-1, 1, B) * (0.05 * frame + 0.3 * np.maximum(slack_y, 0))
    cx, cy = (x0 + x1) / 2 - hx, (y0 + y1) / 2 - hy
    s = RES / frame                                                  # pixels per cm
    u = (Cw[:, :, 0] - cx[:, None]) * s[:, None] + RES / 2
    v = (cy[:, None] - Cw[:, :, 1]) * s[:, None] + RES / 2
    z = Cw[:, :, 2] * s[:, None]; r = Rl * s[:, None]
    zbuf = np.full(B * RES * RES, -1e9)
    splat(zbuf, u[:, :n_small], v[:, :n_small], z[:, :n_small], r[:, :n_small], int(math.ceil(r[:, :n_small].max())))
    splat(zbuf, u[:, n_small:], v[:, n_small:], z[:, n_small:], r[:, n_small:], int(math.ceil(r[:, n_small:].max())))
    zbuf = zbuf.reshape(B, RES, RES)
    mask = zbuf > -1e8
    zmin = np.where(mask, zbuf, 1e9).min(axis=(1, 2))
    zb = np.where(mask, zbuf, (zmin - 3.0 * s)[:, None, None])
    zp = np.pad(zb, ((0, 0), (1, 1), (1, 1)), mode='edge')
    zx = (zp[:, 1:-1, 2:] - zp[:, 1:-1, :-2]) / 2
    zv = (zp[:, 2:, 1:-1] - zp[:, :-2, 1:-1]) / 2
    # light from the viewer's side, somewhat off axis
    lx, ly = rng.uniform(-0.55, 0.55, B), rng.uniform(-0.35, 0.65, B)
    ln = np.sqrt(lx * lx + ly * ly + 1)
    lam = (-zx * (lx / ln)[:, None, None] + zv * (ly / ln)[:, None, None] + (1 / ln)[:, None, None]) / np.sqrt(1 + zx * zx + zv * zv)
    lam = np.maximum(lam, 0.0)
    # cavities (pixels lying deeper than their surroundings) are darker
    m = mask.astype(float)
    zs = box_blur(np.where(mask, zbuf, 0.0), 3) / np.maximum(box_blur(m, 3), 1e-9)
    cav = np.clip((zs - zbuf) / (1.2 * s)[:, None, None], 0.0, 1.0) * m
    amb = rng.uniform(0.45, 0.70, B); ao = rng.uniform(0.25, 0.60, B)
    bg = rng.uniform(0.35, 0.75, B)
    lit = rng.uniform(0.70, 0.97, B)
    dark = rng.uniform(0, 1, B) < 0.15                               # sometimes the hand is darker than the background
    lit = np.where(dark, np.maximum(bg - rng.uniform(0.15, 0.35, B), 0.12), lit)
    hand = lit[:, None, None] * (amb[:, None, None] + (1 - amb)[:, None, None] * lam) * (1 - ao[:, None, None] * cav)
    img = np.where(mask, hand, bg[:, None, None])
    if full: return img                                              # debugging view only
    img = img.reshape(B, 28, 2, 28, 2).mean(axis=(2, 4))
    # the photos are slightly soft: blur with a small Gaussian of random width
    sg = rng.uniform(0.45, 0.85, B)
    k = np.exp(-0.5 * (np.arange(-2, 3)[None, :] / sg[:, None]) ** 2); k /= k.sum(1, keepdims=True)
    ip = np.pad(img, ((0, 0), (2, 2), (2, 2)), mode='edge')
    t = sum(k[:, d, None, None] * ip[:, :, d:d + 28] for d in range(5))
    img = sum(k[:, d, None, None] * t[:, d:d + 28, :] for d in range(5))
    return np.clip(img, 0.0, 1.0)


def render_letter(letter, n, rng, batch=48):
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        prm = sample_params(letter, b, rng)
        Cw, Rl, core, n_small = hand_spheres(*prm)
        out.append(render(Cw, Rl, core, n_small, prm[3], rng))
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------------------------
# 3.  THE NETWORK.  Feature layers (all weights from formulas), then one dense layer of templates, then a maximum
#     over the templates of each class.
# ----------------------------------------------------------------------------------------------------------------
N_DIR = 12            # signed edge directions, 30 degrees apart (channel j and j + 6 are opposite)
N_ORI = 6             # line orientations, 30 degrees apart (ridge and valley sign for each)
SIGMA = 1.0           # scale of the Gaussian derivative filters, in pixels
CUT = math.cos(math.radians(37.5))        # an orientation channel is silent beyond 37.5 degrees from its own direction
KAPPA = 1.5           # contrast above KAPPA x (root-mean-square contrast of the image) is clipped
DEAD = 0.15           # contrast below DEAD x (root-mean-square contrast) is ignored (pixel noise)
K_PER_CLASS = 2885    # templates per letter
DELTA = 0.05          # a template votes for its letter if its score is within DELTA of the best score of all
POOL = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]) / 20.0     # triangular pooling window


def filters():
    """24 filters 5x5: first-derivative-of-Gaussian in 12 directions, then second-derivative-of-Gaussian across 6
    orientations with both signs.  Each has zero sum."""
    ax = np.arange(-2, 3, dtype=float)
    X, Y = np.meshgrid(ax, ax)                       # X to the right, Y down the rows
    G = np.exp(-(X * X + Y * Y) / (2 * SIGMA ** 2))
    W = np.zeros((24, 1, 5, 5))
    for j in range(N_DIR):
        t = math.radians(30.0 * j)
        u = X * math.cos(t) + Y * math.sin(t)
        f = u * G
        W[j, 0] = f - f.mean()
    for k in range(N_ORI):
        t = math.radians(30.0 * k)
        u = X * math.cos(t) + Y * math.sin(t)
        f = (1.0 / SIGMA ** 2 - u * u / SIGMA ** 4) * G     # positive for a bright line running across direction t
        f = f - f.mean()
        W[12 + k, 0] = f; W[18 + k, 0] = -f
    return W


def feature_layers():
    """Returns (layers, weights) of the feature part of the network; its output is (2, 72, 12)."""
    L, Wt = [], {}
    def add(name, op, inp, **kw):
        L.append(dict(name=name, op=op, inputs=[inp], **kw)); return name
    # oriented filters
    Wt['c1.W'] = filters(); Wt['c1.b'] = np.zeros(24)
    x = add('c1', 'conv', 'image', out=24, kernel=[5, 5], stride=1, padding=0)                   # (24, 24, 24)
    # divide the edge group and the line group each by its own root-mean-square over the picture
    # (every filter is present with both signs, so the mean of each group is exactly zero)
    x = add('g1', 'reshape', x, shape=[2, 12 * 24, 24]); x = add('n1', 'norm', x); x = add('g2', 'reshape', x, shape=[24, 24, 24])
    x = add('h1', 'relu', x)
    # sharpen the orientation tuning: subtract a fraction of the summed response of the group, and a small dead zone
    S = np.zeros((24, 24)); b = np.zeros(24)
    # edges: the 12 half-rectified directions of a gradient of size g sum to about 3.8 g
    a_o = CUT / 3.8
    S[:12, :12] = np.eye(12) - a_o
    # lines: a line of strength A gives A cos^2 in the 6 orientations of its own sign, which sum to 3 A
    a_e = CUT ** 2 / 3.0
    for g in (12, 18):
        S[g:g + 6, g:g + 6] = np.eye(6) - a_e
    # response of the best-matching channel per unit of (contrast / rms contrast): see the derivation in method.md
    gain_o = (1 - CUT) * math.sqrt(2.0)
    gain_e = (1 - CUT ** 2) * math.sqrt(8.0 / 3.0)
    b[:12] = -DEAD * gain_o; b[12:] = -DEAD * gain_e
    Wt['s1.W'] = S[:, :, None, None]; Wt['s1.b'] = b
    x = add('s1', 'conv', x, out=24, kernel=[1, 1], stride=1, padding=0); x = add('h2', 'relu', x)
    # clip at T:  min(y, T) = T - relu(T - y)
    T = np.concatenate([np.full(12, KAPPA * gain_o), np.full(12, KAPPA * gain_e)])
    Wt['u1.W'] = -np.eye(24)[:, :, None, None]; Wt['u1.b'] = T.copy()
    x = add('u1', 'conv', x, out=24, kernel=[1, 1], stride=1, padding=0); x = add('h3', 'relu', x)
    # undo the flip and add opposite signs together: 6 unsigned edge orientations, 6 unsigned line orientations,
    # each divided by its clip level so that both groups run from 0 to 2
    U = np.zeros((12, 24)); ub = np.zeros(12)
    for k in range(6):
        U[k, k] = U[k, k + 6] = -1.0 / T[0]; ub[k] = 2.0
        U[6 + k, 12 + k] = U[6 + k, 18 + k] = -1.0 / T[12]; ub[6 + k] = 2.0
    Wt['u2.W'] = U[:, :, None, None]; Wt['u2.b'] = ub
    x = add('u2', 'conv', x, out=12, kernel=[1, 1], stride=1, padding=0)                          # (12, 24, 24)
    # spatial pooling with a triangular window, stride 2
    Pw = np.zeros((12, 12, 8, 8))
    for k in range(12): Pw[k, k] = np.outer(POOL, POOL)
    Wt['pl.W'] = Pw; Wt['pl.b'] = np.zeros(12)
    x = add('pl', 'conv', x, out=12, kernel=[8, 8], stride=2, padding=3)                          # (12, 12, 12)
    # standardise the edge maps and the line maps of the picture (each group by its own mean and spread)
    x = add('g3', 'reshape', x, shape=[2, 72, 12]); x = add('n2', 'norm', x)
    return L, Wt


# --- a small interpreter for these layers (same arithmetic as the task statement), used to turn renders into templates
def _conv(x, W, b, stride, padding):
    n, c, h, w = x.shape; o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    if kh == 1 and kw == 1 and stride == 1:
        return np.einsum('nchw,oc->nohw', xp, W[:, :, 0, 0]) + b[None, :, None, None]
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def run_layers(layers, Wt, x):
    vals = {'image': x}
    for L in layers:
        a = vals[L['inputs'][0]]; op = L['op']; nm = L['name']
        if op == 'conv': y = _conv(a, Wt[nm + '.W'], Wt[nm + '.b'], L['stride'], L['padding'])
        elif op == 'relu': y = np.maximum(a, 0)
        elif op == 'reshape': y = a.reshape((len(a),) + tuple(L['shape']))
        elif op == 'norm':
            mu = a.mean(axis=(2, 3), keepdims=True)
            y = (a - mu) / np.sqrt(((a - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        else: raise ValueError(op)
        vals[nm] = y
    return y


def templates_for(letter, n, seed, layers, Wt):
    """n renders of one letter -> n templates.  Each template is the standardised feature map of one render, scaled
    by the number of entries (so that a score is the mean of the edge and the line correlation coefficients)."""
    rng = np.random.RandomState(seed)
    out = np.zeros((n, 1728), np.float32)
    for i in range(0, n, 170):
        b = min(170, n - i)
        img = render_letter(letter, b, rng)
        f = run_layers(layers, Wt, img[:, None, :, :]).reshape(b, -1)
        out[i:i + b] = (f / 1728.0).astype(np.float32)
    return out


def build(out_dir, k_per_class=K_PER_CLASS, verbose=True):
    layers, Wt = feature_layers()
    n = 24 * k_per_class
    T = np.zeros((n, 1728), np.float32)
    for c, letter in enumerate(LETTERS):
        T[c * k_per_class:(c + 1) * k_per_class] = templates_for(letter, k_per_class, 1000 + c, layers, Wt)
        if verbose: print('templates', letter, flush=True)
    layers = list(layers)
    # score of every template: mean of the edge and the line correlation coefficient with the picture
    layers.append(dict(name='fc', op='dense', inputs=['n2'], out=n))
    # best score over all templates of all letters
    layers.append(dict(name='g4', op='reshape', inputs=['fc'], shape=[1, 1, n]))
    layers.append(dict(name='mx', op='maxpool', inputs=['g4'], size=[1, n], stride=[1, n]))
    # vote of a template = max(score - best + DELTA, 0); a letter's score is the mean vote of its templates
    layers.append(dict(name='gb', op='dense', inputs=['mx'], out=n))
    layers.append(dict(name='sv', op='add', inputs=['fc', 'gb']))
    layers.append(dict(name='h4', op='relu', inputs=['sv']))
    layers.append(dict(name='g5', op='reshape', inputs=['h4'], shape=[1, 24, k_per_class]))
    layers.append(dict(name='vt', op='avgpool', inputs=['g5'], size=[1, k_per_class], stride=[1, k_per_class]))
    layers.append(dict(name='out', op='reshape', inputs=['vt'], shape=[24]))
    W = {k: v.astype(np.float32) for k, v in Wt.items()}
    W['fc.W'] = T; W['fc.b'] = np.zeros(n, np.float32)
    W['gb.W'] = -np.ones((n, 1), np.float32); W['gb.b'] = np.full(n, DELTA, np.float32)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'network.json'), 'w') as f:
        json.dump({'layers': layers}, f)
    np.savez(os.path.join(out_dir, 'weights.npz'), **W)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
