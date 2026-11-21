"""Builds network.json and weights.npz for the ASL-letter classifier without any data.

Architecture: contrast-normalised image -> oriented Gaussian-derivative edge filters (12 signed directions) ->
polarity-free orientation energy (6 orientations, sharpened by subtracting the mean over orientations) -> clipping
-> binomial blur + 2x downsampling -> global normalisation -> template matching (each template is the feature map of
ONE rendered synthetic hand, normalised by itself) with +-2 cell shifts -> max over shifts -> per letter: best
template in each of 6 groups of 15, averaged over the groups (so one stray template cannot dominate a letter).

Templates come from a small 3-D hand model (union of spheres: palm slab, 4 three-joint fingers, three-segment thumb
with two-link inverse kinematics, wrist/forearm) posed for each letter from general knowledge of the ASL manual
alphabet, rendered in orthographic projection with Lambert shading, short-range ambient occlusion and a slight blur,
under random (fixed-seed) view, framing and background-level variations.  No data is read.
"""
import json, os, sys
import numpy as np

# ----------------------------------------------------------------------------------------------------------------
# geometry
E_R = np.array([1.0, 0.0, 0.0])    # radial (thumb side)    -> viewer's right in the canonical pose
E_F = np.array([0.0, -1.0, 0.0])   # distal (to fingertips) -> up in the image
E_N = np.array([0.0, 0.0, -1.0])   # palm normal            -> towards the viewer (camera looks along +z)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rotmat(axis, deg):
    axis = unit(axis); t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def view(f_dir, n_dir):
    """rotation taking the canonical pose (fingers up, palm to viewer) to fingers along f_dir, palm normal along n_dir"""
    f = unit(f_dir); n = np.asarray(n_dir, float); n = unit(n - (n @ f) * f)
    r = np.cross(f, n)
    return np.stack([r, f, n], 1) @ np.linalg.inv(np.stack([E_R, E_F, E_N], 1))


def hf(p):
    """hand-frame point (a radial, b distal, c palmar) -> canonical camera coordinates"""
    return p[0] * E_R + p[1] * E_F + p[2] * E_N


class Model:
    def __init__(self):
        self.c, self.r, self.alb, self.arm = [], [], [], []

    def ball(self, c, r, alb=1.0, arm=False):
        self.c.append(np.asarray(c, float)); self.r.append(float(r)); self.alb.append(alb); self.arm.append(arm)

    def capsule(self, p0, p1, r0, r1, alb=1.0, arm=False, step=0.3):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / (min(r0, r1) * step))) + 1)
        for t in np.linspace(0, 1, n):
            self.ball(p0 + t * (p1 - p0), r0 + (r1 - r0) * t, alb, arm)


MIN_CROP = 1.6   # the photo crop is a square around the hand, never smaller than 1.6 hand units (compact fists)
AO_RAD, AO_DEPTH, AO_STR = 0.15, 0.15, 0.7  # ambient occlusion: sample radius, depth scale (hand units), strength


def render(m, R, fill, off, size=28, ss=4, bg=0.4, light=(-0.35, -0.45, -1.0), amb=0.8, blur=0.55):
    """Orthographic z-buffer render of the sphere union after rotation R, Lambert shading, slight optical blur.
    The crop square (side = larger side of the hand's bounding box, forearm excluded, at least MIN_CROP) is scaled
    to size/fill and centred on the hand's box at size/2+off."""
    C = np.array(m.c) @ R.T; rad = np.array(m.r); alb = np.array(m.alb); hand = ~np.array(m.arm)
    lo = (C[hand, :2] - rad[hand, None]).min(0); hi = (C[hand, :2] + rad[hand, None]).max(0)
    scale = fill * size / max(max(hi - lo), MIN_CROP)
    P = (C[:, :2] - (lo + hi) / 2) * scale + size / 2 + np.asarray(off, float)
    Z = C[:, 2] * scale; Rr = rad * scale
    H = size * ss
    zb = np.full((H, H), np.inf); sh = np.zeros((H, H))
    g = (np.arange(H) + 0.5) / ss
    Lv = unit(light)
    for k in range(len(P)):
        cx, cy = P[k]; r = Rr[k]
        j0, j1 = max(0, int((cx - r) * ss)), min(H, int(np.ceil((cx + r) * ss)) + 1)
        i0, i1 = max(0, int((cy - r) * ss)), min(H, int(np.ceil((cy + r) * ss)) + 1)
        if j0 >= j1 or i0 >= i1:
            continue
        dx = np.broadcast_to(g[None, j0:j1] - cx, (i1 - i0, j1 - j0))
        dy = np.broadcast_to(g[i0:i1, None] - cy, (i1 - i0, j1 - j0))
        d2 = dx * dx + dy * dy
        dz = np.sqrt(np.maximum(r * r - d2, 0))
        z = Z[k] - dz
        sub = zb[i0:i1, j0:j1]
        upd = (d2 < r * r) & (z < sub)
        if not upd.any():
            continue
        sub[upd] = z[upd]
        lam = np.maximum(0, (dx * Lv[0] + dy * Lv[1] - dz * Lv[2]) / r)
        sh[i0:i1, j0:j1][upd] = (alb[k] * (amb + (1 - amb) * lam))[upd]
    fin = np.isfinite(zb)
    if AO_STR > 0:          # screen-space ambient occlusion: crevices and hollows behind nearer surfaces get less light
        occ = np.zeros((H, H)); cnt = 0
        rp = AO_RAD * scale * ss; dz0 = AO_DEPTH * scale
        zpad = np.pad(np.where(fin, zb, np.inf), int(np.ceil(rp)) + 1, constant_values=np.inf); o = int(np.ceil(rp)) + 1
        for fr in (0.5, 1.0):
            for ang in np.arange(8) * np.pi / 4 + fr * np.pi / 8:
                di, dj = int(round(fr * rp * np.sin(ang))), int(round(fr * rp * np.cos(ang)))
                zn = zpad[o + di:o + di + H, o + dj:o + dj + H]
                with np.errstate(invalid='ignore'):
                    d = np.where(fin & np.isfinite(zn), (zb - zn) / dz0, 0.0)
                occ += np.clip(d, 0, 1); cnt += 1
        sh = sh * (1 - AO_STR * occ / cnt)
    img = np.where(fin, sh, bg)
    if blur > 0:
        s = blur * ss; k = np.arange(-int(3 * s), int(3 * s) + 1); kern = np.exp(-k ** 2 / (2 * s * s)); kern /= kern.sum()
        pad = len(k) // 2
        img = np.pad(img, pad, mode='edge')
        img = np.apply_along_axis(lambda v: np.convolve(v, kern, 'valid'), 0, img)
        img = np.apply_along_axis(lambda v: np.convolve(v, kern, 'valid'), 1, img)
    return img.reshape(size, ss, size, ss).mean((1, 3))


# ----------------------------------------------------------------------------------------------------------------
# hand model: right hand, unit = wrist centre to middle knuckle.  Finger: knuckle (a, b, c), phalanx lengths, radii.
FING = {
    'i': ((0.30, 0.97, 0.0), (0.46, 0.28, 0.24), (0.125, 0.115, 0.106, 0.097)),
    'm': ((0.05, 1.00, 0.0), (0.50, 0.32, 0.26), (0.128, 0.117, 0.108, 0.099)),
    'r': ((-0.19, 0.97, 0.0), (0.47, 0.31, 0.25), (0.121, 0.111, 0.102, 0.093)),
    'p': ((-0.41, 0.89, 0.0), (0.37, 0.22, 0.22), (0.106, 0.098, 0.090, 0.082)),
}
THUMB_BASE = np.array([0.20, 0.16, 0.06])
THUMB_LEN = (0.46, 0.32, 0.27)
THUMB_RAD = (0.165, 0.13, 0.117, 0.105)
STRAIGHT = (0, 0, 0)


def finger_dirs(abd, flex):
    a = np.radians(abd)
    d0 = np.array([np.sin(a), np.cos(a), 0.0])
    out, th = [], 0.0
    for f in flex:
        th += np.radians(f)
        out.append(np.cos(th) * d0 + np.sin(th) * np.array([0, 0, 1.0]))
    return out


def finger_joints(f, spec):
    mcp, lens, _ = FING[f]
    p = np.array(mcp, float) + np.array([0, 0, spec[2] if len(spec) > 2 else 0.0])
    pts = [p]
    for k, d in enumerate(finger_dirs(spec[0], spec[1])):
        p = p + lens[k] * d; pts.append(p)
    return pts


def thumb_joints(spec, J):
    """spec: three segment directions, or dict(m=metacarpal direction, to=function of finger joints -> tip target,
    h=bend direction hint).  The last two segments reach for the target (two-link inverse kinematics)."""
    p0 = THUMB_BASE
    if not isinstance(spec, dict):
        pts = [p0]
        for k, d in enumerate(spec):
            pts.append(pts[-1] + THUMB_LEN[k] * unit(d))
        return pts
    p1 = p0 + THUMB_LEN[0] * unit(spec['m'])
    q = np.asarray(spec['to'](J), float)
    l1, l2 = THUMB_LEN[1], THUMB_LEN[2]
    v = q - p1; D = np.linalg.norm(v); u = v / D
    if D >= l1 + l2 - 1e-9:
        return [p0, p1, p1 + l1 * u, p1 + (l1 + l2) * u]
    D = max(D, abs(l1 - l2) + 1e-6)
    ca = (l1 * l1 + D * D - l2 * l2) / (2 * l1 * D); sa = np.sqrt(max(0.0, 1 - ca * ca))
    h = np.asarray(spec.get('h', (1.0, 0.0, -0.3)), float); h = unit(h - (h @ u) * u)
    e = p1 + l1 * (ca * u + sa * h)
    return [p0, p1, e, p1 + D * u]


def build_hand(pose, R, arm_dir=None, arm_alb=1.0):
    """returns Model in canonical camera coords; the forearm is added along arm_dir (camera coords, after R)."""
    m = Model()
    for b in np.linspace(0.14, 0.88, 9):
        t = (b - 0.14) / 0.74; w = np.sqrt(t)
        for a in np.linspace(-0.20 - 0.17 * w, 0.10 + 0.20 * w, 7):
            m.ball(hf((a, b, 0.0)), 0.15)
    m.capsule(hf((-0.28, 0.25, 0.03)), hf((-0.36, 0.70, 0.02)), 0.15, 0.14)
    J = {f: finger_joints(f, pose[f]) for f in 'imrp'}
    for f in 'imrp':
        rads = FING[f][2]
        for k in range(3):
            m.capsule(hf(J[f][k]), hf(J[f][k + 1]), rads[k], rads[k + 1])
    T = thumb_joints(pose['t'], J)
    for k in range(3):
        m.capsule(hf(T[k]), hf(T[k + 1]), THUMB_RAD[k], THUMB_RAD[k + 1])
    if arm_dir is not False:
        w0 = hf((-0.03, 0.30, -0.03))
        d = -E_F if arm_dir is None else R.T @ unit(arm_dir)
        w1 = w0 + 0.25 * d
        m.capsule(w0, w1, 0.34, 0.31, alb=arm_alb, arm=True)
        m.capsule(w1, w1 + 2.0 * d, 0.31, 0.36, alb=arm_alb, arm=True)
    return m


# ----------------------------------------------------------------------------------------------------------------
# the letters (right hand as seen by a person facing the signer)
UP, DOWN, LEFT, RIGHT, TOV, AWAY = (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)
FIST = dict(i=(0, (80, 100, 55)), m=(0, (85, 100, 55)), r=(0, (90, 100, 55)), p=(0, (95, 100, 55)))
V3 = lambda *v: np.array(v, float)
T_SIDE = dict(m=(0.35, 0.9, 0.2), to=lambda J: J['i'][1] + V3(0.15, -0.03, -0.17))         # A: up along the index
T_FIST = dict(m=(0.1, 0.55, 0.8), to=lambda J: V3(-0.15, 0.78, 0.80), h=(0, 1, 0.3))       # S: across the fingers
T_PALM = dict(m=(0.0, 0.5, 0.85), to=lambda J: V3(-0.35, 0.80, 0.28), h=(0, 1, 0.2))       # B: across the palm
T_OUT = [(0.80, 0.55, 0.15), (0.95, 0.30, 0.0), (1.0, 0.15, 0.0)]                          # L, Y: out sideways
T_FWD = [(0.55, 0.80, 0.25), (0.25, 0.97, 0.0), (0.12, 1.0, 0.0)]                          # G, Q: along the index
T_KP = dict(m=(0.2, 0.6, 0.75), to=lambda J: 0.5 * (J['m'][1] + J['m'][0]) + V3(0.12, 0, 0.12))  # K, P
T_TUCK = dict(m=(0.05, 0.55, 0.8), to=lambda J: V3(-0.25, 0.85, 0.35), h=(0, 1, 0.2))      # holding curled fingers


def P_(**kw):
    p = dict(FIST); p['t'] = T_TUCK; p.update(kw); return p


def all4(abd0, flex):
    return dict(i=(-abd0, flex), m=(0, flex), r=(abd0, flex), p=(3 * abd0, flex))


C_CURVE = (25, 45, 30); O_CURVE = (40, 70, 50); D_CURL = (45, 75, 40); E_CURL = (45, 95, 70); DRAPE = (60, 80, 30)
TIPS_IM = lambda J: 0.5 * (J['i'][3] + J['m'][3])
LETTERS = {  # name: list of (pose, finger direction, palm normal, forearm direction or None)
    'A': [(P_(t=T_SIDE), UP, TOV, None)],
    'B': [(P_(**all4(1, STRAIGHT), t=T_PALM), UP, TOV, None)],
    'C': [(P_(**all4(1, C_CURVE), t=dict(m=(0.3, 0.45, 0.85), to=lambda J: TIPS_IM(J) + V3(0, -0.45, 0.05))),
           UP, (0.85, 0, -0.5), None, dict(yaw=18))],
    'D': [(P_(i=(2, STRAIGHT), m=(0, D_CURL), r=(0, D_CURL), p=(0, D_CURL),
              t=dict(m=(0.05, 0.6, 0.8), to=lambda J: J['m'][3])), UP, (0.4, 0, -0.9), None)],
    'E': [(P_(**all4(0, E_CURL), t=dict(m=(0.0, 0.5, 0.85), to=lambda J: V3(-0.32, 0.92, 0.30), h=(0, 1, 0.2))),
           UP, TOV, None)],
    'F': [(P_(i=(0, (45, 60, 40)), m=(2, STRAIGHT), r=(-12, STRAIGHT), p=(-24, STRAIGHT),
              t=dict(m=(0.5, 0.6, 0.6), to=lambda J: J['i'][3])), UP, (0.5, 0, -0.85), None)],
    'G': [(P_(i=(0, STRAIGHT), t=T_FWD), (1, -0.1, 0), (0, 0.8, 0.6), (-0.6, 0.8, 0)),
          (P_(i=(0, STRAIGHT), t=T_FWD), (1, -0.1, 0), AWAY, (-0.6, 0.8, 0))],
    'H': [(P_(i=(0, STRAIGHT), m=(0, STRAIGHT)), (1, -0.1, 0), AWAY, (-0.6, 0.8, 0)),
          (P_(i=(0, STRAIGHT), m=(0, STRAIGHT)), (1, -0.1, 0), TOV, (-0.6, 0.8, 0))],
    'I': [(P_(p=(-3, STRAIGHT)), UP, TOV, None)],
    'K': [(P_(i=(4, STRAIGHT), m=(-22, (45, 0, 0)), t=T_KP), UP, TOV, None)],
    'L': [(P_(i=(0, STRAIGHT), t=T_OUT), UP, TOV, None)],
    'M': [(P_(i=(0, DRAPE), m=(0, DRAPE), r=(0, DRAPE), t=dict(m=(0.0, 0.5, 0.85), to=lambda J: V3(-0.34, 0.92, 0.33))),
           UP, TOV, None)],
    'N': [(P_(i=(0, DRAPE), m=(0, DRAPE), t=dict(m=(0.05, 0.5, 0.85), to=lambda J: V3(-0.08, 0.92, 0.36))), UP, TOV, None)],
    'O': [(P_(**all4(1, O_CURVE), t=dict(m=(0.3, 0.45, 0.85), to=TIPS_IM)), UP, (0.95, 0, -0.3), None, dict(yaw=18))],
    'P': [(P_(i=(0, STRAIGHT), m=(-10, (60, 10, 0)), t=T_KP), (-1, 0.2, 0), TOV, None),
          (P_(i=(0, STRAIGHT), m=(-10, (60, 10, 0)), t=T_KP), (-0.8, 0.6, 0), TOV, None)],
    'Q': [(P_(i=(0, (20, 0, 0)), t=T_FWD), (-0.6, 0.8, 0), TOV, None),
          (P_(i=(0, (20, 0, 0)), t=T_FWD), (-0.6, 0.8, 0), AWAY, None)],
    'R': [(P_(i=(-7, STRAIGHT, 0.0), m=(9, STRAIGHT, 0.1)), UP, TOV, None)],
    'S': [(P_(t=T_FIST), UP, TOV, None)],
    'T': [(P_(i=(0, (10, 10, 0)), t=[(0.75, 0.6, 0.1), (0.95, 0.25, 0.0), (1.0, 0.1, 0.0)]), (1, -0.1, 0), AWAY, (-0.6, 0.8, 0)),
          (P_(i=(0, (10, 10, 0)), t=[(0.75, 0.6, 0.1), (0.95, 0.25, 0.0), (1.0, 0.1, 0.0)]), (1, -0.1, 0), (0, 0.5, 0.85), (-0.6, 0.8, 0))],
    'U': [(P_(i=(-2, STRAIGHT), m=(2, STRAIGHT)), UP, TOV, None)],
    'V': [(P_(i=(16, STRAIGHT), m=(-16, STRAIGHT)), UP, TOV, None)],
    'W': [(P_(i=(18, STRAIGHT), m=(0, STRAIGHT), r=(-18, STRAIGHT)), UP, TOV, None)],
    'X': [(P_(i=(0, (10, 80, 60))), UP, (0.7, 0, -0.7), None, dict(yaw=18)),
          (P_(i=(0, (10, 80, 60))), UP, (0.4, 0, -0.9), None, dict(yaw=18))],
    'Y': [(P_(p=(-25, STRAIGHT), t=[(0.70, 0.70, 0.10), (0.75, 0.65, 0.0), (0.80, 0.60, 0.0)]), UP, TOV, None)],
}
NAMES = list('ABCDEFGHIKLMNOPQRSTUVWXY')


def jitter_pose(pose, rng, abd_sd=3.0, flex_sd=0.08):
    out = {}
    for f in 'imrp':
        spec = pose[f]
        out[f] = (spec[0] + rng.uniform(-abd_sd, abd_sd), tuple(v * (1 + rng.uniform(-flex_sd, flex_sd)) for v in spec[1])) + tuple(spec[2:])
    t = pose['t']
    if isinstance(t, dict):
        dj = rng.uniform(-0.06, 0.06, 3); fn = t['to']
        out['t'] = dict(t, m=unit(np.asarray(t['m']) + rng.uniform(-0.08, 0.08, 3)), to=lambda J, fn=fn, dj=dj: fn(J) + dj)
    else:
        out['t'] = [unit(np.asarray(d) + rng.uniform(-0.08, 0.08, 3)) for d in t]
    return out


def variants(name, n, rng):
    """n renders of one letter: random view perturbations around each base pose (fixed seed)."""
    bases = LETTERS[name]; imgs = []
    for k in range(n):
        pose, f, nrm, arm = bases[k % len(bases)][:4]
        jr = dict(roll=14, yaw=25, pitch=20); jr.update(bases[k % len(bases)][4] if len(bases[k % len(bases)]) > 4 else {})
        roll, yaw, pitch = rng.uniform(-jr['roll'], jr['roll']), rng.uniform(-jr['yaw'], jr['yaw']), rng.uniform(-jr['pitch'], jr['pitch'])
        R0 = view(f, nrm)
        Rg = rotmat((0, 0, 1), roll) @ rotmat((0, 1, 0), yaw) @ rotmat((1, 0, 0), pitch)
        R = Rg @ R0
        armd = Rg @ (R0 @ -E_F) if arm is None else rotmat((0, 0, 1), roll) @ unit(arm)
        use_arm = rng.uniform() < 0.85
        m = build_hand(jitter_pose(pose, rng), R, arm_dir=armd if use_arm else False,
                       arm_alb=1.0 if rng.uniform() < 0.7 else 0.45)
        imgs.append(render(m, R, fill=rng.uniform(0.88, 1.06), off=rng.uniform(-1.5, 1.5, 2), bg=rng.uniform(0.25, 0.55)))
    return imgs


# ----------------------------------------------------------------------------------------------------------------
# network front end (also mirrored in numpy below so the templates go through exactly the same computation)
SIGMA = 1.0      # Gaussian scale of the edge filters, pixels
NORI = 6         # unsigned edge orientations (every 30 degrees)
SHARP = 1.0      # orientation sharpening: subtract SHARP * mean energy over orientations
CLIP = 0.45      # sharpened orientation energy is clipped here (unit step edge on a bin gives 1 - 0.62 = 0.38)
SHIFT = 2        # template search range, +- cells (1 cell = 2 pixels)


def edge_filters():
    k = 7; c = np.arange(k) - k // 2
    yy, xx = np.meshgrid(c, c, indexing='ij')
    G = np.exp(-(xx ** 2 + yy ** 2) / (2 * SIGMA ** 2))
    W = np.zeros((2 * NORI, 1, k, k))
    for d in range(2 * NORI):
        th = d * np.pi / NORI
        f = (xx * np.cos(th) + yy * np.sin(th)) * G
        W[d, 0] = f / f[f > 0].sum()          # unit response to a unit step edge
    return W


def binomial4():
    v = np.array([1.0, 3.0, 3.0, 1.0]) / 8
    return np.outer(v, v)


def front_layers():
    L, Wt = [], {}
    L.append({"name": "n0", "op": "norm", "inputs": ["image"]})
    L.append({"name": "g", "op": "conv", "inputs": ["n0"], "out": 2 * NORI, "kernel": [7, 7], "stride": 1, "padding": 3})
    Wt["g.W"] = edge_filters(); Wt["g.b"] = np.zeros(2 * NORI)
    L.append({"name": "g_r", "op": "relu", "inputs": ["g"]})
    # |derivative| per orientation = relu(+) + relu(-); sharpened by subtracting SHARP * mean over orientations
    A = np.zeros((NORI, 2 * NORI))
    for o in range(NORI):
        A[o, o] = A[o, o + NORI] = 1.0
    A = (np.eye(NORI) - SHARP / NORI) @ A
    W = np.zeros((2 * NORI, 2 * NORI, 1, 1)); b = np.zeros(2 * NORI)
    W[:NORI, :, 0, 0] = A; W[NORI:, :, 0, 0] = A; b[NORI:] = -CLIP
    L.append({"name": "u", "op": "conv", "inputs": ["g_r"], "out": 2 * NORI, "kernel": [1, 1], "stride": 1, "padding": 0})
    Wt["u.W"], Wt["u.b"] = W, b
    L.append({"name": "u_r", "op": "relu", "inputs": ["u"]})
    W = np.zeros((NORI, 2 * NORI, 4, 4)); B4 = binomial4()
    for o in range(NORI):
        W[o, o] = B4; W[o, NORI + o] = -B4                       # min(energy, CLIP), blurred and halved
    L.append({"name": "p", "op": "conv", "inputs": ["u_r"], "out": NORI, "kernel": [4, 4], "stride": 2, "padding": 1})
    Wt["p.W"], Wt["p.b"] = W, np.zeros(NORI)
    L.append({"name": "p_flat", "op": "reshape", "inputs": ["p"], "shape": [1, NORI * 14, 14]})
    L.append({"name": "p_norm", "op": "norm", "inputs": ["p_flat"]})
    L.append({"name": "feat", "op": "reshape", "inputs": ["p_norm"], "shape": [NORI, 14, 14]})
    return L, Wt


# numpy mirror of the operations (same formulas as the scorer's interpreter)
def np_conv(x, W, b, stride, pad):
    c, h, w = x.shape; o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    ho, wo = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    cols = np.stack([xp[:, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], 1)
    out = np.einsum('ckp,ock->op', cols.reshape(c, kh * kw, ho * wo), W.reshape(o, c, kh * kw))
    return out.reshape(o, ho, wo) + b[:, None, None]


def np_norm(x):
    mu = x.mean(axis=(1, 2), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(1, 2), keepdims=True) + 1e-5)


def front_numpy(img, Wt):
    x = np_norm(img[None].astype(np.float64))
    x = np.maximum(np_conv(x, Wt["g.W"], Wt["g.b"], 1, 3), 0)
    x = np.maximum(np_conv(x, Wt["u.W"], Wt["u.b"], 1, 0), 0)
    x = np_conv(x, Wt["p.W"], Wt["p.b"], 2, 1)
    return np_norm(x.reshape(1, NORI * 14, 14)).reshape(NORI, 14, 14)


# ----------------------------------------------------------------------------------------------------------------
N_PER_CLASS = 90
GROUPS = 6       # class score = mean over 6 groups of 15 templates of the best match in the group


def make_templates(Wt, n=N_PER_CLASS, seed=12345):
    rng = np.random.default_rng(seed)
    T, imgs = [], []
    for name in NAMES:
        for im in variants(name, n, rng):
            f = front_numpy(im, Wt)
            f = f - f.mean(); f = f / np.linalg.norm(f)                  # this template normalised by itself
            T.append(f); imgs.append(im)
    return np.array(T), np.array(imgs)


def build(out_dir):
    layers, Wt = front_layers()
    T, _ = make_templates(Wt)
    nt = len(T); n = nt // 24
    s = 2 * SHIFT + 1
    layers.append({"name": "tm", "op": "conv", "inputs": ["feat"], "out": nt, "kernel": [14, 14], "stride": 1, "padding": SHIFT})
    Wt["tm.W"], Wt["tm.b"] = T, np.zeros(nt)
    layers.append({"name": "tm_max", "op": "maxpool", "inputs": ["tm"], "size": [s, s], "stride": [s, s]})
    layers.append({"name": "tm_grid", "op": "reshape", "inputs": ["tm_max"], "shape": [1, 24, n]})
    gs = n // GROUPS
    layers.append({"name": "grp_max", "op": "maxpool", "inputs": ["tm_grid"], "size": [1, gs], "stride": [1, gs]})
    layers.append({"name": "cls_avg", "op": "avgpool", "inputs": ["grp_max"], "size": [1, GROUPS], "stride": [1, GROUPS]})
    layers.append({"name": "scores", "op": "reshape", "inputs": ["cls_avg"], "shape": [24]})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as fh:
        json.dump({"layers": layers}, fh, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: np.asarray(v, np.float32) for k, v in Wt.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
