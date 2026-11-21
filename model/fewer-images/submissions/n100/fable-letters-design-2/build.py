"""Builds the network for the 24 static ASL letters without any data.

Idea: a small 3-D articulated hand (capsule fingers, slab palm and forearm) is posed for each letter from general
knowledge of the manual alphabet and of how each sign is usually held towards the viewer.  Every pose is drawn many
times with small random changes (joint angles, viewing direction, size, light).  Each drawing, on its own, becomes one
template: it is passed through the same fixed edge-orientation layers as the input image and standardised by itself.
The network compares the image with every template at 25 small shifts and gives each letter the score of its best
template.  No number here comes from the training images."""
import sys, json
import numpy as np

# ----------------------------------------------------------------------------- hand model

D2R = np.pi / 180.0
# finger: (knuckle x, knuckle y, (proximal, middle, distal length), (radius of each phalanx))
FINGER_GEOM = [
    (0.330, 1.05, (0.47, 0.28, 0.23), (0.122, 0.112, 0.102)),   # index
    (0.110, 1.10, (0.53, 0.32, 0.25), (0.124, 0.114, 0.104)),   # middle
    (-0.110, 1.04, (0.49, 0.30, 0.24), (0.119, 0.109, 0.100)),  # ring
    (-0.315, 0.93, (0.38, 0.22, 0.21), (0.108, 0.098, 0.090)),  # little
]
THUMB_CMC = (0.31, 0.22, 0.07)
THUMB_LEN = (0.48, 0.36, 0.29)
THUMB_RAD = (0.150, 0.125, 0.110)
PALM_R = 0.16
WIDTH = 1.10          # sideways scale of the palm (finger knuckles, rim, pads, thumb base)
PALM_POLY = [(0.26, 0.03), (0.315, 0.98), (0.12, 1.10), (-0.12, 1.05), (-0.305, 0.86), (-0.26, 0.03)]   # anticlockwise from +z
ARM_R = 0.215
# fleshy pads: ball of the thumb, edge of the hand below the little finger, heel of the hand
PADS = [((0.20, 0.14, 0.09), (0.29, 0.60, 0.11), 0.185), ((-0.21, 0.12, 0.05), (-0.27, 0.68, 0.03), 0.17),
        ((-0.19, 0.07, 0.03), (0.19, 0.07, 0.03), 0.195)]
ARM_LEN = 2.6


def build_hand(fing, thumb, arm, rs):
    """fing: (B,4,4) degrees (abduction, knuckle, middle, end flexion); thumb: (B,3,3) direction of each thumb segment;
    arm: (B,2) degrees (wrist flexion, wrist deviation); rs: (B,) finger thickness factor.
    Returns, in the hand frame: capsules A (B,NC,3), Bq (B,NC,3), radius (B,NC), in_box (NC,);
    slabs O (B,2,3), U (B,2,3), V (B,2,3), poly (2,6,2), radius (2,)."""
    B = fing.shape[0]
    A, Q, Rr, box = [], [], [], []
    zax = np.array([0.0, 0.0, 1.0])
    for i, (mx, my, L, rad) in enumerate(FINGER_GEOM):
        abd = fing[:, i, 0] * D2R
        d0 = np.stack([np.sin(abd), np.cos(abd), np.zeros(B)], 1)
        p = np.stack([mx * WIDTH * rs, np.full(B, my), np.zeros(B)], 1)
        ang = np.zeros(B)
        joints = [p]
        for j in range(3):
            ang = ang + fing[:, i, 1 + j] * D2R
            q = p + L[j] * (np.cos(ang)[:, None] * d0 + np.sin(ang)[:, None] * zax[None, :])
            A.append(p); Q.append(q); Rr.append(rad[j] * rs); box.append(True)
            p = q; joints.append(q)
        tight = np.clip((fing[:, i, 2] - 50.0) / 40.0, 0.02, 1.0)      # flesh fills the loop of a tightly bent finger
        A.append(joints[0]); Q.append(joints[2]); Rr.append(rad[0] * rs * tight); box.append(False)
        A.append(0.5 * (joints[0] + joints[1])); Q.append(joints[3]); Rr.append(0.9 * rad[1] * rs * tight); box.append(False)
    p = np.tile(np.array(THUMB_CMC) * np.array([WIDTH, 1.0, 1.0]), (B, 1))
    for j in range(3):
        d = thumb[:, j, :] / np.sqrt((thumb[:, j, :] ** 2).sum(1, keepdims=True))
        q = p + THUMB_LEN[j] * d
        A.append(p); Q.append(q); Rr.append(THUMB_RAD[j] * (rs if j > 0 else np.ones(B))); box.append(True)
        p = q
    poly = np.array(PALM_POLY) * np.array([WIDTH, 1.0])
    for k in range(len(poly)):                                   # rounded rim of the palm
        a, b = poly[k], poly[(k + 1) % len(poly)]
        A.append(np.stack([a[0] * rs, np.full(B, a[1]), np.zeros(B)], 1))
        Q.append(np.stack([b[0] * rs, np.full(B, b[1]), np.zeros(B)], 1))
        Rr.append(np.full(B, PALM_R)); box.append(True)
    for (pa, pb, pr) in PADS:
        sc = np.array([WIDTH, 1.0, 1.0])
        A.append(np.tile(np.array(pa) * sc, (B, 1))); Q.append(np.tile(np.array(pb) * sc, (B, 1))); Rr.append(np.full(B, pr)); box.append(True)
    # forearm: a rounded slab leaving the wrist
    fl, dv = arm[:, 0] * D2R, arm[:, 1] * D2R
    a = np.stack([np.cos(fl) * np.sin(dv), -np.cos(fl) * np.cos(dv), np.sin(fl)], 1)   # from the wrist towards the elbow
    w = np.stack([np.cos(dv), np.sin(dv), np.zeros(B)], 1)                               # wide axis of the forearm
    o = 0.05 * a
    apoly = np.array([(-0.17, 0.0), (0.17, 0.0), (0.27, ARM_LEN), (-0.27, ARM_LEN), (-0.27, ARM_LEN), (-0.27, ARM_LEN)])
    for (u0, v0, u1, v1, inb) in ((-0.17, 0.0, -0.27, ARM_LEN, False), (0.17, 0.0, 0.27, ARM_LEN, False),
                                  (-0.17, 0.0, 0.17, 0.0, False)):
        A.append(o + u0 * w + v0 * a); Q.append(o + u1 * w + v1 * a); Rr.append(np.full(B, ARM_R)); box.append(inb)
    A = np.stack(A, 1); Q = np.stack(Q, 1); Rr = np.stack(Rr, 1)
    xh = np.tile(np.array([1.0, 0, 0]), (B, 1)); yh = np.tile(np.array([0, 1.0, 0]), (B, 1))
    O = np.stack([np.zeros((B, 3)), o], 1)
    U = np.stack([xh, w], 1)
    V = np.stack([yh, a], 1)
    polys = np.tile(np.stack([poly, apoly])[None], (B, 1, 1, 1))
    polys[:, 0, :, 0] *= rs[:, None]
    return A, Q, Rr, np.array(box), O, U, V, polys, np.array([PALM_R, ARM_R]), a


def rot_mats(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    o, z = np.ones_like(yaw), np.zeros_like(yaw)
    Ry = np.stack([np.stack([cy, z, sy], -1), np.stack([z, o, z], -1), np.stack([-sy, z, cy], -1)], -2)
    Rx = np.stack([np.stack([o, z, z], -1), np.stack([z, cp, -sp], -1), np.stack([z, sp, cp], -1)], -2)
    Rz = np.stack([np.stack([cr, -sr, z], -1), np.stack([sr, cr, z], -1), np.stack([z, z, o], -1)], -2)
    return Rz @ (Rx @ Ry)


def place(hand, view, fill, shift, margin, cap, size=28.0):
    """rotate into the camera and frame the picture the way a square crop around a hand is framed: the hand (without
    the forearm) spans `fill` of the frame, but a compact hand (a fist) is not blown up beyond `cap`; if it fits it is
    centred, otherwise the side away from the forearm keeps a margin and the wrist side is cut off by the frame.
    Everything is returned in pixel units with y pointing down the image and z towards the viewer."""
    A, Q, Rr, box, O, U, V, poly, srad, adir = hand
    R = rot_mats(view[:, 0] * D2R, view[:, 1] * D2R, view[:, 2] * D2R)
    rot = lambda P: np.einsum('bij,bkj->bki', R, P)
    A, Q, O, U, V = rot(A), rot(Q), rot(O), rot(U), rot(V)
    ends = np.concatenate([A[:, box], Q[:, box]], 1)
    er = np.concatenate([Rr[:, box], Rr[:, box]], 1)
    lo = (ends[:, :, :2] - er[:, :, None]).min(1)
    hi = (ends[:, :, :2] + er[:, :, None]).max(1)
    s = np.minimum(size * fill / (hi - lo).max(1), cap)      # never zoomed in beyond `cap` pixels per palm width
    ctr = 0.5 * (lo + hi)
    adir = np.einsum('bij,bj->bi', R, adir) * np.array([1.0, -1.0, 1.0])     # where the forearm goes, image axes
    ext = (hi - lo) * s[:, None]
    off = np.zeros_like(ext)
    for ax in (0, 1):
        over = ext[:, ax] > size - 2 * margin
        c_far_low = margin + ext[:, ax] / 2 - size / 2
        off[:, ax] = np.where(over, np.where(adir[:, ax] > 0, c_far_low, -c_far_low), 0.0)
    shift = shift + off

    def pix(P):
        out = np.empty_like(P)
        out[:, :, 0] = (P[:, :, 0] - ctr[:, None, 0]) * s[:, None] + size / 2 + shift[:, None, 0]
        out[:, :, 1] = -(P[:, :, 1] - ctr[:, None, 1]) * s[:, None] + size / 2 + shift[:, None, 1]
        out[:, :, 2] = P[:, :, 2] * s[:, None]
        return out
    flip = np.array([1.0, -1.0, 1.0])
    return pix(A), pix(Q), Rr * s[:, None], pix(O), U * flip, V * flip, poly * s[:, None, None, None], srad[None] * s[:, None]


def render(geom, light, skin, bg, grad, amb=0.70, occ=0.60, size=28, ss=2):
    """z-buffer at ss x resolution, Lambert shading + darkening in creases, box-downsampled to size x size."""
    A, Q, Rr, O, U, V, poly, srad = geom
    B, NC = Rr.shape
    S = size * ss
    g = (np.arange(S) + 0.5) / ss
    gx = g[None, None, :]; gy = g[None, :, None]
    zb = np.full((B, S, S), -1e9); nx = np.zeros((B, S, S)); ny = np.zeros((B, S, S)); nz = np.ones((B, S, S))

    def put(z, ok, ax, ay, az):
        nonlocal zb, nx, ny, nz
        m = ok & (z > zb)
        zb = np.where(m, z, zb); nx = np.where(m, ax, nx); ny = np.where(m, ay, ny); nz = np.where(m, az, nz)

    for k in range(NC):
        a = A[:, k]; q = Q[:, k]; r = Rr[:, k][:, None, None]
        d = q - a
        L2 = (d ** 2).sum(1)[:, None, None]
        dxk, dyk, dzk = d[:, 0, None, None], d[:, 1, None, None], d[:, 2, None, None]
        px = gx - a[:, 0, None, None]; py = gy - a[:, 1, None, None]; az0 = a[:, 2, None, None]
        for (cx, cy, cz) in ((px, py, az0), (gx - q[:, 0, None, None], gy - q[:, 1, None, None], q[:, 2, None, None])):
            h2 = r * r - cx * cx - cy * cy                     # ball joint at each end
            ok = h2 > 0
            h = np.sqrt(np.where(ok, h2, 0.0))
            put(cz + h, ok, cx / r, cy / r, h / r)
        qd0 = px * dxk + py * dyk - az0 * dzk                  # (P - A).d with P at z = 0
        aa = 1.0 - dzk * dzk / L2
        bb = -az0 - qd0 * dzk / L2
        cc = px * px + py * py + az0 * az0 - qd0 * qd0 / L2 - r * r
        disc = bb * bb - aa * cc
        ok = (disc > 0) & (aa > 1e-6)
        z = (-bb + np.sqrt(np.where(ok, disc, 0.0))) / np.where(aa > 1e-6, aa, 1.0)
        t = (qd0 + z * dzk) / L2
        ok = ok & (t > 0) & (t < 1)
        put(z, ok, (px - t * dxk) / r, (py - t * dyk) / r, (z - az0 - t * dzk) / r)
    for k in range(O.shape[1]):                               # the two flat faces of each slab
        o = O[:, k]; u = U[:, k]; v = V[:, k]
        n = np.cross(u, v)
        sgn = np.where(n[:, 2] >= 0, 1.0, -1.0)
        nzk = np.maximum(np.abs(n[:, 2]), 1e-6)[:, None, None]
        px = gx - o[:, 0, None, None]; py = gy - o[:, 1, None, None]
        s3 = sgn[:, None, None]
        z = o[:, 2, None, None] + (srad[:, k, None, None] - s3 * (n[:, 0, None, None] * px + n[:, 1, None, None] * py)) / nzk
        pz = z - o[:, 2, None, None]
        uu = px * u[:, 0, None, None] + py * u[:, 1, None, None] + pz * u[:, 2, None, None]
        vv = px * v[:, 0, None, None] + py * v[:, 1, None, None] + pz * v[:, 2, None, None]
        ok = np.abs(n[:, 2])[:, None, None] > 1e-3
        pk = poly[:, k]
        m = pk.shape[1]
        for e in range(m):
            p0 = pk[:, e]; p1 = pk[:, (e + 1) % m]
            ex = (p1 - p0)[:, 0, None, None]; ey = (p1 - p0)[:, 1, None, None]
            ok = ok & (ex * (vv - p0[:, 1, None, None]) - ey * (uu - p0[:, 0, None, None]) >= 0)
        put(z, ok, s3 * n[:, 0, None, None], s3 * n[:, 1, None, None], s3 * n[:, 2, None, None])
    fg = zb > -1e8
    lam = np.maximum(nx * light[:, 0, None, None] - ny * light[:, 1, None, None] + nz * light[:, 2, None, None], 0.0)
    zf = np.where(fg, zb, np.nan)
    o = np.zeros_like(zb); cnt = 0
    for dd in (ss, 2 * ss):                                   # creases: how much nearer the neighbouring surface is
        for (sy, sx) in ((dd, 0), (-dd, 0), (0, dd), (0, -dd), (dd, dd), (dd, -dd), (-dd, dd), (-dd, -dd)):
            zn = np.full_like(zf, np.nan)
            ys = slice(max(sy, 0), S + min(sy, 0)); yt = slice(max(-sy, 0), S + min(-sy, 0))
            xs = slice(max(sx, 0), S + min(sx, 0)); xt = slice(max(-sx, 0), S + min(-sx, 0))
            zn[:, yt, xt] = zf[:, ys, xs]
            dz = np.nan_to_num(zn - zf, nan=0.0)
            o += np.clip(dz / (2.0 * dd) - 0.15, 0.0, 1.0); cnt += 1
    o = o / cnt
    shade = (amb + (1 - amb) * lam) * (1.0 - occ * np.clip(1.6 * o, 0.0, 1.0))
    gg = (g / size - 0.5)
    back = bg[:, None, None] + grad[:, 0, None, None] * gg[None, None, :] + grad[:, 1, None, None] * gg[None, :, None]
    img = np.where(fg, skin[:, None, None] * shade, back)
    img = img.reshape(B, size, ss, size, ss).mean((2, 4))
    return np.clip(img, 0.0, 1.0)


# ----------------------------------------------------------------------------- letters

FIST = (0, 85, 95, 60)
LOOSE = (0, 62, 95, 60)
UP = (0, 0, 0, 0)
TH_SIDE = [(0.40, 0.75, 0.50), (0.05, 1.0, 0.12), (-0.05, 1.0, 0.05)]       # thumb lying against the side of the fist
TH_CROSS = [(0.25, 0.50, 0.82), (-0.75, 0.45, 0.30), (-0.95, 0.20, -0.05)]  # thumb across the front of the folded fingers
TH_HOLD = [(0.20, 0.55, 0.80), (-0.60, 0.70, 0.30), (-0.80, 0.55, -0.05)]   # thumb holding down the folded ring and little fingers
TH_PALM = [(0.05, 0.70, 0.45), (-0.85, 0.45, -0.10), (-0.95, 0.30, -0.10)]  # thumb folded flat across the palm
TH_UNDER = [(0.10, 0.50, 0.65), (-0.80, 0.60, -0.10), (-1.0, 0.10, -0.10)]  # thumb across the palm, below the fingertips
TH_OUT = [(0.75, 0.60, 0.25), (1.0, 0.35, 0.10), (1.0, 0.30, 0.05)]         # thumb stuck out sideways

# fingers: index, middle, ring, little as (abduction, knuckle, middle joint, end joint)
# view: (yaw, pitch, roll); yaw > 0 turns the palm towards the viewer's right, roll > 0 is anticlockwise
# arm: (wrist flexion, wrist deviation)
POSES = {
    'A': dict(f=[FIST, FIST, FIST, FIST], t=TH_SIDE, view=(0, 0, -12), arm=(0, 0)),
    'B': dict(f=[(0, 0, 3, 0), (0, 0, 3, 0), (0, 0, 3, 0), (-1, 0, 3, 0)], t=TH_PALM, view=(0, 0, -6), arm=(0, 0), abd=1.2),
    'C': dict(f=[(0, 38, 45, 30), (0, 40, 45, 30), (0, 40, 45, 30), (0, 40, 45, 30)],
              t=[(0.25, 0.40, 0.80), (0.0, 0.55, 0.85), (-0.10, 0.75, 0.65)], view=(58, 0, -30), arm=(0, 0), abd=1.2),
    'D': dict(f=[UP, (0, 55, 75, 45), (0, 55, 75, 45), (0, 58, 75, 45)],
              t=[(0.05, 0.55, 0.75), (-0.25, 0.70, 0.60), (-0.40, 0.75, 0.40)], view=(20, 0, -6), arm=(0, 0)),
    'E': dict(f=[(0, 40, 105, 80)] * 4, t=TH_UNDER, view=(0, 0, -3), arm=(0, 0), abd=1.2),
    'F': dict(f=[(0, 50, 75, 40), (8, 5, 5, 0), (-8, 0, 5, 0), (-22, 0, 5, 0)],
              t=[(0.10, 0.55, 0.75), (-0.05, 0.75, 0.60), (-0.15, 0.80, 0.45)], view=(40, 0, -10), arm=(0, 0)),
    'G': dict(f=[(0, 5, 5, 0), LOOSE, LOOSE, LOOSE], t=[(0.10, 0.70, 0.60), (0.0, 0.85, 0.50), (0.0, 0.90, 0.40)],
              view=(180, 25, -85), arm=(0, -25)),
    'H': dict(f=[(0, 5, 5, 0), (0, 5, 5, 0), LOOSE, LOOSE], t=[(0.10, 0.70, 0.60), (0.0, 0.85, 0.50), (0.0, 0.90, 0.40)],
              view=(180, 25, -85), arm=(0, -25)),
    'I': dict(f=[FIST, FIST, FIST, (-10, 0, 0, 0)], t=TH_HOLD, view=(0, 0, -6), arm=(0, 0)),
    'K': dict(f=[UP, (-5, 65, 5, 5), FIST, FIST], t=[(0.0, 0.75, 0.60), (-0.15, 0.90, 0.35), (-0.20, 0.95, 0.10)],
              view=(-30, 0, -5), arm=(0, 0)),
    'L': dict(f=[UP, FIST, FIST, FIST], t=TH_OUT, view=(0, 0, -6), arm=(0, 0)),
    'M': dict(f=[(0, 75, 75, 35), (0, 75, 75, 35), (0, 75, 75, 35), FIST],
              t=[(0.0, 0.55, 0.60), (-0.85, 0.50, -0.10), (-1.0, 0.35, 0.05)], view=(0, 0, -15), arm=(0, 0)),
    'N': dict(f=[(0, 75, 75, 35), (0, 75, 75, 35), FIST, FIST],
              t=[(0.0, 0.55, 0.60), (-0.70, 0.60, 0.0), (-0.80, 0.60, 0.10)], view=(0, 0, -22), arm=(0, 0)),
    'O': dict(f=[(0, 50, 65, 45), (0, 50, 65, 45), (0, 52, 65, 45), (0, 55, 65, 45)],
              t=[(0.15, 0.45, 0.85), (-0.10, 0.70, 0.70), (-0.20, 0.90, 0.35)], view=(62, 0, -22), arm=(0, 0), abd=1.2,
              var=dict(joint=4.0, common=4.0, thumb=0.06)),
    'P': dict(f=[(0, 0, 5, 5), (-5, 65, 5, 5), FIST, FIST],
              t=[(0.0, 0.75, 0.60), (-0.15, 0.90, 0.35), (-0.20, 0.95, 0.10)], view=(-63, 48, 63), arm=(20, 0), var=dict(yaw=12.0, pitch=10.0)),
    'Q': dict(f=[(0, 25, 10, 5), LOOSE, LOOSE, LOOSE],
              t=[(0.75, 0.45, 0.45), (0.45, 0.80, 0.40), (0.15, 0.95, 0.30)], view=(180, 55, 162), arm=(70, 0), var=dict(yaw=12.0)),
    'R': dict(f=[(-10, 0, 5, 0), (10, 8, 5, 0), FIST, FIST], t=TH_HOLD, view=(0, 0, -6), arm=(0, 0)),
    'S': dict(f=[FIST, FIST, FIST, FIST], t=TH_CROSS, view=(0, 0, -8), arm=(0, 0)),
    'T': dict(f=[(0, 88, 15, 10), FIST, FIST, FIST],
              t=[(0.0, 0.90, 0.45), (-0.20, 1.0, 0.15), (0.0, 1.0, 0.0)], view=(65, 0, -5), arm=(0, 0)),
    'U': dict(f=[(0, 0, 3, 0), (0, 0, 3, 0), FIST, FIST], t=TH_HOLD, view=(0, 0, -6), arm=(0, 0), abd=1.2),
    'V': dict(f=[(14, 0, 3, 0), (-14, 0, 3, 0), FIST, FIST], t=TH_HOLD, view=(0, 0, -6), arm=(0, 0)),
    'W': dict(f=[(16, 0, 3, 0), (0, 0, 3, 0), (-16, 0, 3, 0), FIST],
              t=[(0.15, 0.55, 0.80), (-0.85, 0.50, 0.25), (-1.0, 0.30, -0.05)], view=(0, 0, -6), arm=(0, 0)),
    'X': dict(f=[(0, 10, 75, 55), FIST, FIST, FIST], t=TH_HOLD, view=(50, 0, -15), arm=(0, 0)),
    'Y': dict(f=[FIST, FIST, FIST, (-25, 0, 0, 0)],
              t=[(0.65, 0.70, 0.20), (0.80, 0.60, 0.10), (0.80, 0.60, 0.05)], view=(0, 0, -6), arm=(0, 0)),
}
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
# how much each pose is varied (standard deviations): joints, thumb direction, yaw, pitch, roll
VAR = dict(joint=7.0, common=6.0, abd=3.0, thumb=0.10, yaw=10.0, pitch=8.0, roll=9.0, arm=8.0, thick=0.07)


def sample(letter, n, rng, var=VAR):
    """n drawings of one letter: returns the sphere placement inputs."""
    P = POSES[letter]
    var = dict(var, **P.get('var', {}))
    f = np.tile(np.array(P['f'], float)[None], (n, 1, 1))
    common = rng.normal(0, var['common'], (n, 1, 1))                 # the whole hand a little more open or closed
    bent = (f[:, :, 1:] > 20)
    f[:, :, 1:] += rng.normal(0, var['joint'], (n, 4, 3)) * np.where(bent, 1.0, 0.5) + common * bent
    f[:, :, 1:] = np.maximum(f[:, :, 1:], -8.0)
    spread = rng.normal(0, P.get('abd', var['abd']), (n, 1))                       # fingers fan out or close up together
    f[:, :, 0] += spread * np.array([1.0, 0.33, -0.33, -1.0])[None] + rng.normal(0, 1.2, (n, 4))
    t = np.array(P['t'], float)
    t = t / np.sqrt((t ** 2).sum(1, keepdims=True))
    t = t[None] + rng.normal(0, var['thumb'], (n, 3, 3))
    view = np.array(P['view'], float)[None] + rng.normal(0, 1, (n, 3)) * np.array([var['yaw'], var['pitch'], var['roll']])
    arm = np.array(P['arm'], float)[None] + rng.normal(0, var['arm'], (n, 2))
    rs = np.exp(rng.normal(0, var['thick'], n))
    return f, t, arm, rs, view


def draw(letter, n, rng, size=28, ss=2, plain=False):
    f, t, arm, rs, view = sample(letter, n, rng)
    hand = build_hand(f, t, arm, rs)
    fill = rng.uniform(0.90, 1.16, n)
    cap = rng.uniform(15.5, 18.5, n) * (size / 28.0)
    shift = rng.uniform(-1.0, 1.0, (n, 2)) * (size / 28.0)
    margin = rng.uniform(1.0, 3.5, n) * (size / 28.0)
    geom = place(hand, view, fill, shift, margin, cap, size=float(size))
    lx = rng.normal(-0.25, 0.25, n); ly = rng.normal(0.30, 0.20, n)
    light = np.stack([lx, ly, np.ones(n)], 1); light /= np.sqrt((light ** 2).sum(1, keepdims=True))
    lighter = rng.random(n) < 0.6                      # the hand is often, but not always, lighter than the background
    hi_v = rng.uniform(0.62, 0.95, n); gap = rng.uniform(0.18, 0.45, n)
    skin = np.where(lighter, hi_v, hi_v - gap)
    bg = np.where(lighter, hi_v - gap, hi_v)
    grad = rng.normal(0, 0.05, (n, 2))
    if plain:
        light = np.tile(np.array([[-0.2, 0.3, 1.0]]) / np.sqrt(1.13), (n, 1)); skin = np.full(n, 0.8); bg = np.full(n, 0.3); grad[:] = 0
    out = []
    for i in range(0, n, 16):
        s = slice(i, i + 16)
        out.append(render(tuple(g[s] for g in geom), light[s], skin[s], bg[s], grad[s], size=size, ss=ss))
    return np.concatenate(out)


# ----------------------------------------------------------------------------- network
N_ORI = 6            # orientations, 30 degrees apart
N_CH = 2 * N_ORI     # 6 edge maps (sign of contrast ignored) + 6 maps of thin dark lines (creases between fingers)
SIGMA = 1.0          # scale of the derivative filters (pixels)
GAIN = 10.0
EDGE_VETO = 1.5      # a dark line must not be just the dark side of a step edge
DEAD = 0.50          # strength (in units of the image's RMS strength) below which an edge or line is ignored
SAT = 1.00           # ... and the further rise after which it counts fully
CELL, STEP = 4, 2    # maps are averaged over 4x4 cells every 2 pixels
PAD = 2              # templates are tried at shifts of -2..2 cells (= -4..4 pixels) in x and y
N_PER_LETTER = 1200
COARSE = 1.0         # weight of the blurred copy of each template (tolerance to small errors of the drawing)
ROW_WEIGHT = [1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.6, 0.45]   # the bottom of the picture (wrist, sleeve, table edge) counts less
SEED = 20240


def feature_layers():
    """Layers (and weights) that turn the image into 12 maps of 11x11 cells: edges and dark lines at 6 orientations."""
    L, W = [], {}
    r = np.arange(-2, 3, dtype=float)
    gy, gx = np.meshgrid(r, r, indexing='ij')
    g = np.exp(-(gx ** 2 + gy ** 2) / (2 * SIGMA ** 2))
    dx = gx * g; dx /= np.abs(dx[:, 3:]).sum()            # response 1 to a unit step edge
    dy = gy * g; dy /= np.abs(dy[3:, :]).sum()
    k = np.zeros((4 * N_ORI, 1, 5, 5))
    for j in range(2 * N_ORI):
        th = np.pi * j / N_ORI
        k[j, 0] = GAIN * (np.cos(th) * dx + np.sin(th) * dy)
    for o in range(N_ORI):                                # second derivative across a line whose normal is at angle th
        th = np.pi * o / N_ORI
        u = np.cos(th) * gx + np.sin(th) * gy
        d2 = (u ** 2 / SIGMA ** 2 - 1.0) * g
        d2 -= d2.mean()                                   # no response to a flat patch
        d2 /= d2[2, 2] * -1.0 if abs(d2[2, 2]) > 0 else 1.0
        line = np.zeros((5, 5)); line[np.abs(u) < 0.5] = -1.0     # a one pixel wide dark line of unit depth
        d2 /= max((d2 * line).sum(), 1e-9)                # response 1 to that line
        k[2 * N_ORI + o, 0] = GAIN * d2
        k[3 * N_ORI + o, 0] = -GAIN * d2
    L.append(dict(name='filt', op='conv', inputs=['image'], out=4 * N_ORI, kernel=[5, 5], stride=1, padding=0))
    W['filt.W'] = k; W['filt.b'] = np.zeros(4 * N_ORI)
    L.append(dict(name='filt_r', op='relu', inputs=['filt']))
    m = np.zeros((N_CH, 4 * N_ORI, 1, 1))
    for o in range(N_ORI):
        p = (o + N_ORI // 2) % N_ORI
        # edge: |gradient along o| - |gradient across o| (sharp orientation tuning, zero for edges 45 degrees off)
        m[o, o] = 1; m[o, o + N_ORI] = 1; m[o, p] = -1; m[o, p + N_ORI] = -1
        # dark line: valley across o, minus valley across the other direction (blobs), minus the edge strength
        m[N_ORI + o, 2 * N_ORI + o] = 1; m[N_ORI + o, 2 * N_ORI + p] = -1
        m[N_ORI + o, o] = -EDGE_VETO; m[N_ORI + o, o + N_ORI] = -EDGE_VETO
    L.append(dict(name='ori', op='conv', inputs=['filt_r'], out=N_CH, kernel=[1, 1], stride=1, padding=0))
    W['ori.W'] = m; W['ori.b'] = np.zeros(N_CH)
    L.append(dict(name='ori_r', op='relu', inputs=['ori']))
    # divide by the RMS strength over the whole image: stack +e and -e so that the mean is zero, then norm
    m = np.zeros((2 * N_CH, N_CH, 1, 1))
    for o in range(N_CH):
        m[o, o] = 1; m[o + N_CH, o] = -1
    L.append(dict(name='pm', op='conv', inputs=['ori_r'], out=2 * N_CH, kernel=[1, 1], stride=1, padding=0))
    W['pm.W'] = m; W['pm.b'] = np.zeros(2 * N_CH)
    L.append(dict(name='pm_flat', op='reshape', inputs=['pm'], shape=[1, 2 * N_CH * 24, 24]))
    L.append(dict(name='pm_norm', op='norm', inputs=['pm_flat']))
    L.append(dict(name='pm_back', op='reshape', inputs=['pm_norm'], shape=[2 * N_CH, 24, 24]))
    # dead zone and saturation: min(relu(x - DEAD), SAT) = relu(x - DEAD) - relu(x - DEAD - SAT)
    m = np.zeros((2 * N_CH, 2 * N_CH, 1, 1)); b = np.zeros(2 * N_CH)
    for o in range(N_CH):
        m[o, o] = 1; b[o] = -DEAD
        m[o + N_CH, o] = 1; b[o + N_CH] = -DEAD - SAT
    L.append(dict(name='lim', op='conv', inputs=['pm_back'], out=2 * N_CH, kernel=[1, 1], stride=1, padding=0))
    W['lim.W'] = m; W['lim.b'] = b
    L.append(dict(name='lim_r', op='relu', inputs=['lim']))
    m = np.zeros((N_CH, 2 * N_CH, 1, 1))
    for o in range(N_CH):
        m[o, o] = 1; m[o, o + N_CH] = -1
    L.append(dict(name='edge', op='conv', inputs=['lim_r'], out=N_CH, kernel=[1, 1], stride=1, padding=0))
    W['edge.W'] = m; W['edge.b'] = np.zeros(N_CH)
    L.append(dict(name='cells', op='avgpool', inputs=['edge'], size=[CELL, CELL], stride=[STEP, STEP]))
    return L, W


def features(img, W):
    """The same computation as the feature layers, for a batch of drawings (N,28,28) -> (N,12,11,11)."""
    n = img.shape[0]
    win = np.stack([img[:, dy:dy + 24, dx:dx + 24] for dy in range(5) for dx in range(5)], -1)      # (N,24,24,25)
    g = np.maximum(win @ W['filt.W'].reshape(4 * N_ORI, 25).T, 0.0)                                 # (N,24,24,24)
    e = np.maximum(g @ W['ori.W'].reshape(N_CH, 4 * N_ORI).T, 0.0)                                  # (N,24,24,12)
    rms = np.sqrt((e ** 2).mean((1, 2, 3), keepdims=True) + 1e-5)
    x = e / rms
    f = np.maximum(x - DEAD, 0.0) - np.maximum(x - DEAD - SAT, 0.0)
    f = f.transpose(0, 3, 1, 2)
    nc = (24 - CELL) // STEP + 1
    out = np.zeros((n, N_CH, nc, nc))
    for dy in range(CELL):
        for dx in range(CELL):
            out += f[:, :, dy:dy + STEP * nc:STEP, dx:dx + STEP * nc:STEP]
    return out / (CELL * CELL)


def blur_cells(z):
    p = np.pad(z, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='edge')
    v = 0.25 * p[:, :, :-2] + 0.5 * p[:, :, 1:-1] + 0.25 * p[:, :, 2:]
    return 0.25 * v[:, :, :, :-2] + 0.5 * v[:, :, :, 1:-1] + 0.25 * v[:, :, :, 2:]


def templates(feat):
    """Each drawing becomes one template, standardised by itself alone."""
    z = feat - feat.mean((1, 2, 3), keepdims=True)
    z = z + COARSE * blur_cells(z)
    z = z * np.array(ROW_WEIGHT)[None, None, :, None]
    return z / np.sqrt((z ** 2).sum((1, 2, 3), keepdims=True))


def build(n_per_letter=N_PER_LETTER, verbose=False):
    L, W = feature_layers()
    nc = (24 - CELL) // STEP + 1
    npos = 2 * PAD + 1
    heads = []
    for ci, letter in enumerate(LETTERS):
        rng = np.random.default_rng(SEED)        # the same draws of view, size, light for every letter
        img = draw(letter, n_per_letter, rng)
        t = templates(features(img, W))
        nm = 'm_' + letter
        L.append(dict(name=nm, op='conv', inputs=['cells'], out=n_per_letter, kernel=[nc, nc], stride=1, padding=PAD))
        W[nm + '.W'] = t; W[nm + '.b'] = np.zeros(n_per_letter)
        L.append(dict(name=nm + '_shift', op='maxpool', inputs=[nm], size=[npos, npos], stride=[npos, npos]))
        L.append(dict(name=nm + '_row', op='reshape', inputs=[nm + '_shift'], shape=[1, 1, n_per_letter]))
        L.append(dict(name=nm + '_best', op='maxpool', inputs=[nm + '_row'], size=[1, n_per_letter], stride=[1, n_per_letter]))
        L.append(dict(name=nm + '_out', op='dense', inputs=[nm + '_best'], out=len(LETTERS)))
        col = np.zeros((len(LETTERS), 1)); col[ci, 0] = 1.0
        W[nm + '_out.W'] = col; W[nm + '_out.b'] = np.zeros(len(LETTERS))
        heads.append(nm + '_out')
        if verbose:
            print(letter, 'done', flush=True)
    L.append(dict(name='scores', op='add', inputs=heads))
    return {'layers': L}, {k: v.astype(np.float32) for k, v in W.items()}


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    net, weights = build()
    with open(out + '/network.json', 'w') as fh:
        json.dump(net, fh)
    np.savez(out + '/weights.npz', **weights)
