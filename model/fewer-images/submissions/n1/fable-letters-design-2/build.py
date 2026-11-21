"""Builds network.json and weights.npz for the sign-language letter classifier.

Everything here comes from general knowledge: an articulated hand model (palm, fingers and thumb as
capsules with anatomical proportions), the joint angles of the 24 static ASL letters, a simple
orthographic renderer, and a hand-chosen oriented-edge feature front end.  No file is read.

Usage: python3 build.py OUT_DIR
"""
import sys, os, json, math
import numpy as np

# ----------------------------------------------------------------------------------------------
# 1. Hand model (centimetres).  Hand frame: X toward the thumb side, Y toward the fingertips,
#    Z out of the palm.  Right hand.  Wrist centre at the origin.
# ----------------------------------------------------------------------------------------------
FINGER_MCP = np.array([[2.9, 9.0, 0.0], [0.9, 9.4, 0.0], [-1.05, 9.0, 0.0], [-2.9, 8.1, 0.0]])
FINGER_BASE = np.array([[2.2, 1.0, 0.0], [0.75, 1.0, 0.0], [-0.75, 1.0, 0.0], [-2.3, 1.0, 0.0]])
# phalanx lengths (proximal, middle, distal up to the centre of the tip sphere)
FINGER_LEN = np.array([[3.9, 2.3, 1.35], [4.3, 2.6, 1.4], [4.0, 2.5, 1.35], [3.2, 1.9, 1.25]])
# radii at MCP, PIP, DIP, tip
FINGER_RAD = np.array([[1.00, 0.95, 0.87, 0.78], [1.00, 0.95, 0.87, 0.79], [0.96, 0.90, 0.83, 0.75], [0.86, 0.79, 0.72, 0.65]])
PALM_RAD = (1.15, 1.05)
THUMB_CMC = np.array([2.2, 1.6, 0.6])
THUMB_LEN = np.array([4.4, 3.2, 1.85])
THUMB_RAD = np.array([1.45, 1.10, 0.95, 0.85])
WEB_RAD = 0.9
ARM_LEN = 26.0
ARM_RAD = (1.8, 2.5)
NCAP = 22          # 4 palm + 12 finger + 3 thumb + 1 web + 2 forearm
NHAND = 20         # capsules that belong to the hand itself

D2R = math.pi / 180.0


def _sincos(a):
    """sin and cos from +,-,* only (Taylor series at a/16, then four angle doublings), so that the
    geometry is bit-for-bit the same on every machine."""
    x = np.asarray(a, dtype=np.float64) / 16.0
    x2 = x * x
    s = x * (1.0 + x2 * (-1.0 / 6 + x2 * (1.0 / 120 + x2 * (-1.0 / 5040 + x2 * (1.0 / 362880 + x2 * (-1.0 / 39916800 + x2 / 6227020800.0))))))
    c = 1.0 + x2 * (-0.5 + x2 * (1.0 / 24 + x2 * (-1.0 / 720 + x2 * (1.0 / 40320 + x2 * (-1.0 / 3628800 + x2 * (1.0 / 479001600 - x2 / 87178291200.0))))))
    for _ in range(4):
        s, c = 2.0 * s * c, c * c - s * s
    return s, c


def _rot(axis, a):
    """Batch rotation matrices about a camera/hand axis; a: (B,) radians -> (B,3,3)."""
    s, c = _sincos(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    if axis == 'x':
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == 'y':
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.stack([np.stack(r, axis=-1) for r in m], axis=-2)


def _mm(A, B):
    """Batched 3x3 matrix product written out term by term (no BLAS, fixed order of operations)."""
    out = np.empty(np.broadcast_shapes(A.shape, B.shape))
    for i in range(3):
        for j in range(3):
            out[..., i, j] = A[..., i, 0] * B[..., 0, j] + A[..., i, 1] * B[..., 1, j] + A[..., i, 2] * B[..., 2, j]
    return out


def _mv(A, v):
    """Batched matrix (B,3,3) times vectors (B,3) or (B,K,3), term by term."""
    if v.ndim == 3:
        A = A[:, None]
    out = np.empty(v.shape)
    for i in range(3):
        out[..., i] = A[..., i, 0] * v[..., 0] + A[..., i, 1] * v[..., 1] + A[..., i, 2] * v[..., 2]
    return out


def _unit(v):
    return v / np.sqrt(v[..., 0:1] * v[..., 0:1] + v[..., 1:2] * v[..., 1:2] + v[..., 2:3] * v[..., 2:3])


def hand_capsules(fing, thumb, thick):
    """fing: (B,4,4) radians [abduction, MCP, PIP, DIP flexion] for index, middle, ring, little.
    thumb: (B,3,3) direction vectors (hand frame) of the thumb metacarpal, proximal and distal segments.
    thick: (B,) factor on all radii.
    Returns P0, P1 (B,NHAND,3) and R0, R1 (B,NHAND) in the hand frame."""
    B = fing.shape[0]
    P0 = np.zeros((B, NHAND, 3)); P1 = np.zeros((B, NHAND, 3))
    R0 = np.zeros((B, NHAND)); R1 = np.zeros((B, NHAND))
    for f in range(4):
        P0[:, f] = FINGER_BASE[f]; P1[:, f] = FINGER_MCP[f]
        R0[:, f] = PALM_RAD[0]; R1[:, f] = PALM_RAD[1]
        abd = fing[:, f, 0]
        sa, ca = _sincos(abd)
        d0 = np.stack([sa, ca, np.zeros(B)], axis=-1)
        p = np.broadcast_to(FINGER_MCP[f], (B, 3)).copy()
        cum = np.zeros(B)
        for k in range(3):
            cum = cum + fing[:, f, 1 + k]
            sc, cc = _sincos(cum)
            d = cc[:, None] * d0
            d[:, 2] = sc
            q = p + FINGER_LEN[f, k] * d
            i = 4 + 3 * f + k
            P0[:, i] = p; P1[:, i] = q
            R0[:, i] = FINGER_RAD[f, k]; R1[:, i] = FINGER_RAD[f, k + 1]
            p = q
    p = np.broadcast_to(THUMB_CMC, (B, 3)).copy()
    td = _unit(thumb)
    for k in range(3):
        q = p + THUMB_LEN[k] * td[:, k]
        P0[:, 16 + k] = p; P1[:, 16 + k] = q
        R0[:, 16 + k] = THUMB_RAD[k]; R1[:, 16 + k] = THUMB_RAD[k + 1]
        p = q
    # skin web between the thumb metacarpal and the index metacarpal
    P0[:, 19] = P0[:, 16] + 0.6 * (P1[:, 16] - P0[:, 16])
    P1[:, 19] = np.array([2.5, 6.2, 0.0])
    R0[:, 19] = WEB_RAD; R1[:, 19] = WEB_RAD
    return P0, P1, R0 * thick[:, None], R1 * thick[:, None]


# ----------------------------------------------------------------------------------------------
# 2. The 24 letters: joint angles (degrees) typed from knowledge of the ASL manual alphabet.
#    Finger tuples are (abduction toward the thumb side, MCP flexion, PIP flexion, DIP flexion).
# ----------------------------------------------------------------------------------------------
def EXT(abd=0.0, mcp=4.0):
    return (abd, mcp, 4.0, 3.0)


def FIST(abd=0.0):
    return (abd, 85.0, 100.0, 65.0)


TH_SIDE = [(0.60, 0.70, 0.38), (0.00, 0.97, 0.25), (-0.05, 1.00, 0.05)]      # alongside the index, pointing up (A)
TH_FRONT = [(0.36, 0.66, 0.66), (-0.66, 0.50, 0.56), (-0.95, 0.30, 0.00)]     # across the front of the fist (S)
TH_PALM = [(0.25, 0.92, 0.30), (-0.85, 0.50, 0.15), (-0.95, 0.30, -0.05)]     # folded flat across the palm (B)
TH_OUT = [(0.80, 0.55, 0.25), (0.96, 0.25, 0.10), (0.97, 0.22, 0.05)]         # stretched out sideways (L)
TH_OUTUP = [(0.75, 0.60, 0.25), (0.80, 0.58, 0.12), (0.75, 0.66, 0.05)]       # stretched out and up (Y)
TH_PAR = [(0.45, 0.72, 0.53), (0.05, 0.96, 0.27), (0.00, 1.00, 0.05)]         # parallel to the index (G, Q)
TH_GAP = [(0.55, 0.62, 0.56), (0.12, 0.92, 0.37), (0.00, 0.98, 0.20)]         # parallel to the index with a gap (Q)
TH_MID = [(0.42, 0.74, 0.52), (-0.35, 0.86, 0.36), (-0.30, 0.94, 0.15)]       # tip between index and middle (K, P)
TH_UNDER3 = [(0.30, 0.78, 0.55), (-0.90, 0.38, 0.20), (-0.96, 0.25, -0.10)]   # under three fingers (M)
TH_UNDER2 = [(0.32, 0.76, 0.56), (-0.80, 0.52, 0.30), (-0.70, 0.70, 0.10)]    # under two fingers (N)
TH_T = [(0.38, 0.76, 0.52), (-0.45, 0.80, 0.40), (-0.25, 0.95, 0.20)]         # up between index and middle (T)
TH_E = [(0.32, 0.72, 0.62), (-0.92, 0.25, 0.30), (-0.98, 0.15, -0.10)]        # across, below the fingertips (E)
TH_O = [(0.42, 0.58, 0.70), (-0.30, 0.72, 0.62), (-0.40, 0.90, 0.15)]         # tip meeting the fingertips (O, D)
TH_F = [(0.41, 0.60, 0.68), (-0.20, 0.65, 0.73), (-0.15, 0.92, 0.35)]         # tip meeting the index tip (F)
TH_C = [(0.40, 0.40, 0.82), (0.10, 0.40, 0.91), (0.00, 0.60, 0.80)]           # opposed, lower jaw of the C
TH_HOLD = [(0.30, 0.76, 0.58), (-0.88, 0.40, 0.25), (-0.92, 0.35, 0.00)]      # holding down curled fingers (U, V, W, R, I, X)

# Base orientations (hand frame -> camera frame; camera x right, y up, z toward the viewer).
#  front : palm toward the viewer, fingers up, thumb on the viewer's right
#  side  : turned so the palm faces the viewer's right (C, O, D, X)
#  point : fingers pointing to the viewer's right, back of the hand toward the viewer (G, H); the
#          'spin' angle turns the hand about the pointing axis toward palm-down
#  kside : seen from the thumb side, palm to the viewer's left (K turned; P is this rotated)
LETTERS = {
    'A': [dict(f=[FIST(), FIST(), FIST(), FIST()], th=TH_SIDE, view='front')],
    'B': [dict(f=[EXT(-2), EXT(0), EXT(2), EXT(5)], th=TH_PALM, view='front')],
    'C': [dict(f=[(2, 32, 48, 32), (0, 34, 50, 32), (-2, 34, 50, 32), (-5, 34, 48, 32)], th=TH_C, view='side', yaw=(40, 92))],
    'D': [dict(f=[EXT(0, 6), (0, 62, 74, 48), (-2, 64, 76, 50), (-4, 66, 78, 50)], th=TH_O, view='side', yaw=(35, 95))],
    'E': [dict(f=[(0, 42, 108, 82), (0, 42, 108, 82), (0, 44, 108, 82), (0, 46, 108, 82)], th=TH_E, view='front')],
    'F': [dict(f=[(4, 48, 70, 42), EXT(3, 6), EXT(-9, 6), EXT(-22, 8)], th=TH_F, view='front')],
    'G': [dict(f=[EXT(0, 8), FIST(), FIST(), FIST()], th=TH_PAR, view='point')],
    'H': [dict(f=[EXT(0, 8), EXT(0, 8), FIST(), FIST()], th=TH_HOLD, view='point')],
    'I': [dict(f=[FIST(), FIST(), FIST(), EXT(-8, 6)], th=TH_FRONT, view='front')],
    'K': [dict(f=[EXT(4, 6), (-6, 55, 8, 5), FIST(), FIST()], th=TH_MID, view='front', yaw=(-70, 10))],
    'L': [dict(f=[EXT(0, 5), FIST(), FIST(), FIST()], th=TH_OUT, view='front')],
    'M': [dict(f=[(0, 78, 86, 35), (0, 78, 86, 35), (0, 78, 88, 35), FIST()], th=TH_UNDER3, view='front', pitch=(-10, 35))],
    'N': [dict(f=[(0, 78, 86, 35), (0, 78, 86, 35), FIST(), FIST()], th=TH_UNDER2, view='front', pitch=(-10, 35))],
    'O': [dict(f=[(0, 58, 74, 48), (0, 60, 76, 48), (-2, 62, 76, 50), (-4, 64, 78, 50)], th=TH_O, view='side', yaw=(40, 92))],
    'P': [dict(f=[EXT(0, 12), (-4, 75, 12, 8), FIST(), FIST()], th=TH_MID, view='point', roll=(130, 215), spin=(0, 100), bendz=(-60, -5))],
    'Q': [dict(f=[EXT(0, 10), FIST(), FIST(), FIST()], th=TH_GAP, view='point', roll=(190, 260), spin=(-20, 90))],
    'R': [dict(f=[(-9, 4, 6, 4), (9, 12, 6, 4), FIST(), FIST()], th=TH_HOLD, view='front')],
    'S': [dict(f=[FIST(), FIST(), FIST(), FIST()], th=TH_FRONT, view='front')],
    'T': [dict(f=[(0, 70, 96, 50), FIST(), FIST(), FIST()], th=TH_T, view='front', yaw=(-25, 70))],
    'U': [dict(f=[EXT(-1, 4), EXT(1, 4), FIST(), FIST()], th=TH_HOLD, view='front')],
    'V': [dict(f=[EXT(13, 4), EXT(-11, 4), FIST(), FIST()], th=TH_HOLD, view='front')],
    'W': [dict(f=[EXT(16, 5), EXT(0, 5), EXT(-16, 5), FIST()], th=TH_HOLD, view='front')],
    'X': [dict(f=[(0, 18, 92, 62), FIST(), FIST(), FIST()], th=TH_HOLD, view='side', yaw=(25, 95))],
    'Y': [dict(f=[FIST(), FIST(), FIST(), EXT(-24, 6)], th=TH_OUTUP, view='front')],
}
CLASS_NAMES = 'ABCDEFGHIKLMNOPQRSTUVWXY'

BASE = {
    'front': np.eye(3),
    'side': np.eye(3),
    'point': np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),    # Y->+x, X->+y, Z->-z
    'kside': np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),    # X->+z, Y->+y, Z->-x
}
# default ranges in degrees: yaw (about camera y), pitch (about camera x), roll (in the image plane),
# spin (about the pointing axis, 'point' view only), forearm bend in the image plane and in depth
VIEW = {
    'front': dict(yaw=(-25, 25), pitch=(-20, 20), roll=(-15, 15), spin=(0, 0), bendz=(-18, 18), bendx=(-18, 18)),
    'side': dict(yaw=(50, 95), pitch=(-18, 18), roll=(-12, 22), spin=(0, 0), bendz=(-18, 18), bendx=(-18, 18)),
    'point': dict(yaw=(-20, 20), pitch=(-15, 15), roll=(-12, 25), spin=(-100, 10), bendz=(5, 60), bendx=(-20, 20)),
    'kside': dict(yaw=(-20, 20), pitch=(-15, 15), roll=(-15, 15), spin=(0, 0), bendz=(-20, 20), bendx=(-20, 20)),
}


GROUP = 8             # placements (crop scale and offset) rendered for every pose


def sample_poses(letter, n, rng, group=1):
    """n random poses of one letter, each repeated `group` times with different crops.
    Returns a dict of arrays with n * group entries (pose-major order)."""
    specs = LETTERS[letter]
    out = []
    for si, sp in enumerate(specs):
        m = n // len(specs) + (1 if si < n % len(specs) else 0)
        V = dict(VIEW[sp['view']])
        for k in ('yaw', 'pitch', 'roll', 'spin', 'bendz', 'bendx'):
            if k in sp:
                V[k] = sp[k]
        u = lambda lo, hi, size=None: lo + (hi - lo) * rng.random_sample((m,) if size is None else size)
        fing = np.tile(np.array(sp['f'], dtype=np.float64)[None], (m, 1, 1))
        common = u(-6, 6, (m, 1, 3))                       # whole-hand tighter / looser
        fing[:, :, 1:] += common + u(-6, 6, (m, 4, 3))
        fing[:, :, 0] += u(-3, 3, (m, 4))
        fing[:, :, 1:] = np.maximum(fing[:, :, 1:], 0.0)
        thumb = np.tile(np.array(sp['th'], dtype=np.float64)[None], (m, 1, 1)) + u(-0.09, 0.09, (m, 3, 3))
        ang = {k: u(*V[k]) * D2R for k in ('yaw', 'pitch', 'roll', 'spin', 'bendz', 'bendx')}
        d = dict(fing=fing * D2R, thumb=thumb, base=np.tile(BASE[sp['view']][None], (m, 1, 1)),
                 thick=u(0.92, 1.12), lx=u(-0.45, 0.45), ly=u(-0.7, -0.2), **ang)
        d = {k: np.repeat(v, group, axis=0) for k, v in d.items()}
        # crops: stratified over tight/loose, left/right and up/down when there are 8 per pose
        g = np.arange(m * group) % group
        r = lambda: rng.random_sample(m * group)
        if group == 8:
            b0, b1, b2 = (g & 1), ((g >> 1) & 1), ((g >> 2) & 1)
        else:
            b0 = b1 = b2 = None
        half = lambda lo, hi, bit: (lo + (hi - lo) * r()) if bit is None else (lo + (hi - lo) * 0.5 * (bit + r()))
        d['fill'] = half(0.84, 1.0, b0)
        d['jx'] = half(-0.3, 0.3, b1)
        d['jy'] = half(-0.3, 0.3, b2)
        d['allow'] = 3.0 * r()
        d['bias'] = 0.35 * r()
        out.append(d)
    return {k: np.concatenate([o[k] for o in out], axis=0) for k in out[0]}


FRAME_MIN = 15.0      # the crop is never much tighter than this many centimetres


def pose_to_scene(p, S):
    """Turn sampled poses into capsules in image coordinates of an S x S frame.
    Returns A, Bp (B,NCAP,3) with x right, y down, z toward the viewer (pixels) and radii R0, R1 (pixels), scale."""
    B = p['fing'].shape[0]
    P0, P1, R0, R1 = hand_capsules(p['fing'], p['thumb'], p['thick'])
    Rh = _mm(_rot('x', p['pitch']), _mm(_rot('y', p['yaw']), _mm(_rot('x', p['spin']), p['base'])))
    Rr = _rot('z', p['roll'])
    Rfull = _mm(Rr, Rh)
    C0 = _mv(Rfull, P0); C1 = _mv(Rfull, P1)
    # forearm: starts at the wrist, direction = hand's -Y bent at the wrist
    down = _mv(Rh, np.tile(np.array([0.0, -1.0, 0.0]), (B, 1)))
    side = _mv(Rh, np.tile(np.array([1.0, 0.0, 0.0]), (B, 1)))
    Rb = _mm(_rot('z', p['bendz']), _rot('x', p['bendx']))
    fd = _mv(Rr, _mv(Rb, down))
    fs = _mv(Rr, side)
    fs = _unit(fs - (fs[:, 0:1] * fd[:, 0:1] + fs[:, 1:2] * fd[:, 1:2] + fs[:, 2:3] * fd[:, 2:3]) * fd)
    wrist = _mv(Rfull, np.tile(np.array([0.0, 0.6, 0.0]), (B, 1)))
    A = np.zeros((B, NCAP, 3)); Bp = np.zeros((B, NCAP, 3))
    r0 = np.zeros((B, NCAP)); r1 = np.zeros((B, NCAP))
    A[:, :NHAND] = C0; Bp[:, :NHAND] = C1; r0[:, :NHAND] = R0; r1[:, :NHAND] = R1
    for j, sg in enumerate((-1.0, 1.0)):
        A[:, NHAND + j] = wrist + sg * 1.25 * fs
        Bp[:, NHAND + j] = wrist + sg * 1.75 * fs + ARM_LEN * fd
        r0[:, NHAND + j] = ARM_RAD[0] * p['thick']; r1[:, NHAND + j] = ARM_RAD[1] * p['thick']
    # bounding box of the hand itself
    lo = np.minimum(C0 - R0[..., None], C1 - R1[..., None]).min(axis=1)
    hi = np.maximum(C0 + R0[..., None], C1 + R1[..., None]).max(axis=1)
    w = hi[:, 0] - lo[:, 0]; h = hi[:, 1] - lo[:, 1]
    F = np.maximum(np.maximum(w, h) + p['allow'], FRAME_MIN) / p['fill']
    cx = 0.5 * (lo[:, 0] + hi[:, 0]); cy = 0.5 * (lo[:, 1] + hi[:, 1])
    # the hand sits away from the side the forearm leaves through
    ox = (F - w) * (-fd[:, 0] * p['bias'] + p['jx'])
    oy = (F - h) * (-fd[:, 1] * p['bias'] + p['jy'])
    s = S / F
    for X in (A, Bp):
        X[:, :, 0] = (X[:, :, 0] - cx[:, None] - ox[:, None]) * s[:, None] + S / 2.0
        X[:, :, 1] = S / 2.0 - (X[:, :, 1] - cy[:, None] - oy[:, None]) * s[:, None]
        X[:, :, 2] = X[:, :, 2] * s[:, None]
    return A, Bp, r0 * s[:, None], r1 * s[:, None], s


# ----------------------------------------------------------------------------------------------
# 3. Renderer: orthographic, z-buffered capsules, Lambert shading plus darkening in creases.
# ----------------------------------------------------------------------------------------------
def _gauss_matrix(n, sigma):
    i = np.arange(n)
    G = np.exp(-0.5 * ((i[:, None] - i[None, :]) / sigma) ** 2)
    return G / G.sum(axis=1, keepdims=True)


def _sep(X, G):
    """Separable filter of a batch of square maps with matrix G (rows = output samples), without BLAS."""
    Y = (X[:, :, None, :] * G[None, None, :, :]).sum(axis=-1)
    return (Y[:, None, :, :] * G[None, :, :, None]).sum(axis=2)


AMBIENT, DIFFUSE = 0.55, 0.45
AO_MAX, AO_START, AO_RANGE = 0.5, 0.15, 0.6     # crease darkening: depth below the local mean, in cm
AO_SIGMA = 1.3                                   # pixels of the final image
POST_BLUR = 0.6                                  # pixels, mimics the softness of the resized photos
_CACHE = {}


def render(A, Bp, r0, r1, s, lx, ly, S, ss=2):
    """Returns images (B,S,S): background 0, hand shaded in (0,1].
    Each capsule is drawn into a depth buffer inside its own bounding window (ss x ss samples per pixel)."""
    B = A.shape[0]
    N = S * ss
    key = (S, ss)
    if key not in _CACHE:
        _CACHE[key] = (_gauss_matrix(N, AO_SIGMA * ss), _gauss_matrix(S, POST_BLUR))
    Gao, Gp = _CACHE[key]
    A = A * ss; Bp = Bp * ss; r0 = r0 * ss; r1 = r1 * ss
    lz = np.sqrt(np.maximum(1.0 - lx * lx - ly * ly, 0.05))
    ln = np.sqrt(lx * lx + ly * ly + lz * lz)
    lx3 = (lx / ln)[:, None, None]; ly3 = (ly / ln)[:, None, None]; lz3 = (lz / ln)[:, None, None]
    zbuf = np.full((B, N * N), -1e9)
    sbuf = np.zeros((B, N * N))
    rm = np.maximum(r0, r1)
    for c in range(NCAP):
        win = []
        for k in (0, 1):
            lo = np.floor(np.clip(np.minimum(A[:, c, k], Bp[:, c, k]) - rm[:, c], 0.0, N))
            hi = np.clip(np.maximum(A[:, c, k], Bp[:, c, k]) + rm[:, c], 0.0, N)
            W = int(min(N, max(1.0, np.ceil((hi - lo).max()) + 1.0)))
            o = np.clip(lo, 0, N - W).astype(np.int64)
            win.append((o, W))
        (ox, W), (oy, Hh) = win
        px = ox[:, None] + np.arange(W)[None, :]
        py = oy[:, None] + np.arange(Hh)[None, :]
        ax = A[:, c, 0][:, None, None]; ay = A[:, c, 1][:, None, None]; az = A[:, c, 2][:, None, None]
        dx = Bp[:, c, 0][:, None, None] - ax; dy = Bp[:, c, 1][:, None, None] - ay; dz = Bp[:, c, 2][:, None, None] - az
        L2 = dx * dx + dy * dy + 1e-9
        kz = dz / np.sqrt(L2 * (L2 + dz * dz))
        ra = r0[:, c][:, None, None]; dr = (r1[:, c] - r0[:, c])[:, None, None]
        e0x = (px[:, None, :] + 0.5) - ax                                    # (B,1,W)
        e0y = (py[:, :, None] + 0.5) - ay                                    # (B,H,1)
        tu = (e0x * dx + e0y * dy) / L2                                      # (B,H,W)
        d2 = e0x * e0x + e0y * e0y - tu * tu * L2                            # squared distance from the axis line
        rr = ra + np.minimum(np.maximum(tu, 0.0), 1.0) * dr
        t = rr * rr - d2
        np.maximum(t, 0.0, out=t); np.sqrt(t, out=t)
        t *= kz; t += tu
        np.maximum(t, 0.0, out=t); np.minimum(t, 1.0, out=t)                 # axis point whose sphere is nearest the viewer
        ex = e0x - t * dx; ey = e0y - t * dy
        rr = ra + t * dr
        h = rr * rr - ex * ex - ey * ey
        outside = h <= 0.0
        np.maximum(h, 0.0, out=h); np.sqrt(h, out=h)                         # height of the surface over the axis
        z = az + t * dz + h
        np.copyto(z, -2e9, where=outside)
        sh = AMBIENT + DIFFUSE * np.maximum((ex * lx3 + ey * ly3 + h * lz3) / rr, 0.0)
        idx = (py[:, :, None] * N + px[:, None, :]).reshape(B, Hh * W)
        z = z.reshape(B, Hh * W); sh = sh.reshape(B, Hh * W)
        old = np.take_along_axis(zbuf, idx, 1)
        better = z > old
        np.put_along_axis(zbuf, idx, np.where(better, z, old), 1)
        np.put_along_axis(sbuf, idx, np.where(better, sh, np.take_along_axis(sbuf, idx, 1)), 1)
    mask = zbuf > -1e8
    # crease darkening: how far the surface lies behind the local average depth of the hand
    m = mask.astype(np.float64).reshape(B, N, N)
    zc = np.where(mask, zbuf, 0.0).reshape(B, N, N)
    zb = Gao @ zc @ Gao.T
    mb = Gao @ m @ Gao.T
    dcm = (zb / np.maximum(mb, 1e-6) - zc) / (s[:, None, None] * ss)
    ao = AO_MAX * np.clip((dcm - AO_START) / AO_RANGE, 0.0, 1.0)
    full = m * sbuf.reshape(B, N, N) * (1.0 - ao)
    small = full.reshape(B, S, ss, S, ss).mean(axis=(2, 4))
    return Gp @ small @ Gp.T


def render_letter(letter, n, rng, S=28, ss=2, group=1):
    p = sample_poses(letter, n, rng, group)
    A, Bp, r0, r1, s = pose_to_scene(p, S)
    return render(A, Bp, r0, r1, s, p['lx'], p['ly'], S, ss)


# ----------------------------------------------------------------------------------------------
# 4. Feature front end (the first layers of the network), also applied here to every render.
#    image -> norm -> 8 oriented derivative-of-Gaussian responses -> |.| -> sharpened orientation
#    tuning -> dead zone and saturation -> 4x4 average pooling (stride 2) -> norm over everything.
# ----------------------------------------------------------------------------------------------
IMG = 28
KS, SIGMA, NORI = 5, 1.0, 8
SHARP = math.tan(67.5 * D2R)        # response falls to zero 22.5 degrees away from the preferred orientation
TAU0, TAU1, TAU2 = 0.04, 0.20, 0.50 # dead zone, start of half slope, saturation (units: z-scored grey levels per pixel)
POOL, PSTRIDE = 4, 2
CONV_OUT = IMG - KS + 1             # 24
CELLS = (CONV_OUT - POOL) // PSTRIDE + 1   # 11
FDIM = NORI * CELLS * CELLS         # 968
NORM_EPS = 1e-5


def gradient_kernels():
    r = np.arange(KS) - KS // 2
    X = np.tile(r[None, :], (KS, 1)).astype(np.float64); Y = np.tile(r[:, None], (1, KS)).astype(np.float64)
    g = np.exp(-(X * X + Y * Y) / (2.0 * SIGMA * SIGMA))
    gx = X * g; gx = gx / (X * gx).sum()      # responds 1 to a ramp of slope 1
    gy = Y * g; gy = gy / (Y * gy).sum()
    return gx, gy


def front_end_weights():
    gx, gy = gradient_kernels()
    W1 = np.zeros((2 * NORI, 1, KS, KS))
    for k in range(NORI):
        th = k * math.pi / NORI
        f = math.cos(th) * gx + math.sin(th) * gy
        W1[2 * k, 0] = f; W1[2 * k + 1, 0] = -f
    W2 = np.zeros((NORI, 2 * NORI, 1, 1)); b2 = np.full(NORI, -TAU0)
    for k in range(NORI):
        kp = (k + NORI // 2) % NORI
        W2[k, 2 * k, 0, 0] = 1.0; W2[k, 2 * k + 1, 0, 0] = 1.0
        W2[k, 2 * kp, 0, 0] = -SHARP; W2[k, 2 * kp + 1, 0, 0] = -SHARP
    W3 = np.zeros((2 * NORI, NORI, 1, 1)); b3 = np.zeros(2 * NORI)
    W4 = np.zeros((NORI, 2 * NORI, 1, 1))
    for k in range(NORI):
        W3[2 * k, k, 0, 0] = 1.0; b3[2 * k] = -(TAU1 - TAU0)
        W3[2 * k + 1, k, 0, 0] = 1.0; b3[2 * k + 1] = -(TAU2 - TAU0)
        W4[k, 2 * k, 0, 0] = -0.5; W4[k, 2 * k + 1, 0, 0] = -0.5
    return W1, W2, b2, W3, b3, W4


def features(img):
    """img (B,28,28) -> (B,FDIM): exactly what the network computes up to its layer 'nf'."""
    B = img.shape[0]
    W1, W2, b2, W3, b3, W4 = front_end_weights()
    mu = img.mean(axis=(1, 2), keepdims=True)
    x = (img - mu) / np.sqrt(((img - mu) ** 2).mean(axis=(1, 2), keepdims=True) + NORM_EPS)
    n = CONV_OUT
    c1 = np.zeros((B, 2 * NORI, n, n))
    for dy in range(KS):
        for dx in range(KS):
            c1 += W1[None, :, 0, dy, dx, None, None] * x[:, None, dy:dy + n, dx:dx + n]
    r1 = np.maximum(c1, 0.0)
    a = r1[:, 0::2] + r1[:, 1::2]                                  # |oriented derivative|
    perp = [(k + NORI // 2) % NORI for k in range(NORI)]
    r2 = np.maximum(a - SHARP * a[:, perp] - TAU0, 0.0)
    m = r2 - 0.5 * np.maximum(r2 - (TAU1 - TAU0), 0.0) - 0.5 * np.maximum(r2 - (TAU2 - TAU0), 0.0)
    pooled = np.zeros((B, NORI, CELLS, CELLS))
    span = PSTRIDE * (CELLS - 1) + 1
    for dy in range(POOL):
        for dx in range(POOL):
            pooled += m[:, :, dy:dy + span:PSTRIDE, dx:dx + span:PSTRIDE]
    f = (pooled / (POOL * POOL)).reshape(B, FDIM)
    mu = f.mean(axis=1, keepdims=True)
    return (f - mu) / np.sqrt(((f - mu) ** 2).mean(axis=1, keepdims=True) + NORM_EPS)


# ----------------------------------------------------------------------------------------------
# 5. The network: front end -> one unit per rendered template (correlation with that template)
#    -> best crop of each pose (maxpool) -> standardise over all poses (norm) -> every pose that
#    stands out votes for its letter (relu thresholds, average per letter).
# ----------------------------------------------------------------------------------------------
NPOSE = 500                 # poses per letter
VOTE1, VOTE2 = 1.8, 2.6     # a pose votes when its score is this many standard deviations above the mean pose
SEED = 20240


def build_templates(nclass=24, npose=NPOSE, batch=64, log=None):
    T = nclass * npose * GROUP
    W = np.zeros((T, FDIM), dtype=np.float32)
    row = 0
    for ci in range(nclass):
        letter = CLASS_NAMES[ci]
        rng = np.random.RandomState(SEED + ci)
        p = sample_poses(letter, npose, rng, GROUP)
        ntot = npose * GROUP
        for i in range(0, ntot, batch):
            q = {k: v[i:i + batch] for k, v in p.items()}
            A, Bp, r0, r1, s = pose_to_scene(q, IMG)
            img = render(A, Bp, r0, r1, s, q['lx'], q['ly'], IMG, 2)
            f = features(img)
            # each render is its own template: its own feature map, standardised by itself alone
            W[row:row + f.shape[0]] = (f / math.sqrt(FDIM)).astype(np.float32)
            row += f.shape[0]
        if log:
            log('templates for %s done' % letter)
    return W


def build_network(nclass=24, npose=NPOSE):
    T = nclass * npose * GROUP
    layers = [
        dict(name='n0', op='norm', inputs=['image']),
        dict(name='c1', op='conv', inputs=['n0'], out=2 * NORI, kernel=[KS, KS], stride=1, padding=0),
        dict(name='r1', op='relu', inputs=['c1']),
        dict(name='c2', op='conv', inputs=['r1'], out=NORI, kernel=[1, 1], stride=1, padding=0),
        dict(name='r2', op='relu', inputs=['c2']),
        dict(name='c3', op='conv', inputs=['r2'], out=2 * NORI, kernel=[1, 1], stride=1, padding=0),
        dict(name='r3', op='relu', inputs=['c3']),
        dict(name='c4', op='conv', inputs=['r3'], out=NORI, kernel=[1, 1], stride=1, padding=0),
        dict(name='m', op='add', inputs=['r2', 'c4']),
        dict(name='pool', op='avgpool', inputs=['m'], size=[POOL, POOL], stride=[PSTRIDE, PSTRIDE]),
        dict(name='flat', op='reshape', inputs=['pool'], shape=[1, NORI * CELLS, CELLS]),
        dict(name='nf', op='norm', inputs=['flat']),
        dict(name='t', op='dense', inputs=['nf'], out=T),
        dict(name='tg', op='reshape', inputs=['t'], shape=[1, T // GROUP, GROUP]),
        dict(name='tm', op='maxpool', inputs=['tg'], size=[1, GROUP], stride=[1, GROUP]),
        dict(name='tz', op='norm', inputs=['tm']),
        dict(name='v1c', op='conv', inputs=['tz'], out=1, kernel=[1, 1], stride=1, padding=0),
        dict(name='v1', op='relu', inputs=['v1c']),
        dict(name='v2c', op='conv', inputs=['v1'], out=1, kernel=[1, 1], stride=1, padding=0),
        dict(name='v2', op='relu', inputs=['v2c']),
        dict(name='vs', op='add', inputs=['v1', 'v2', 'v2']),
        dict(name='vr', op='reshape', inputs=['vs'], shape=[nclass, npose, 1]),
        dict(name='vp', op='avgpool', inputs=['vr'], size=[npose, 1], stride=[npose, 1]),
        dict(name='scores', op='reshape', inputs=['vp'], shape=[nclass]),
    ]
    return dict(layers=layers)


def build_weights(templates):
    W1, W2, b2, W3, b3, W4 = front_end_weights()
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    w = {
        'c1.W': f32(W1), 'c1.b': np.zeros(2 * NORI, np.float32),
        'c2.W': f32(W2), 'c2.b': f32(b2),
        'c3.W': f32(W3), 'c3.b': f32(b3),
        'c4.W': f32(W4), 'c4.b': np.zeros(NORI, np.float32),
        't.W': templates, 't.b': np.zeros(templates.shape[0], np.float32),
        'v1c.W': np.ones((1, 1, 1, 1), np.float32), 'v1c.b': f32([-VOTE1]),
        'v2c.W': np.ones((1, 1, 1, 1), np.float32), 'v2c.b': f32([-(VOTE2 - VOTE1)]),
    }
    return w


def main(out_dir, npose=NPOSE):
    os.makedirs(out_dir, exist_ok=True)
    log = lambda s: (sys.stdout.write(s + '\n'), sys.stdout.flush())
    templates = build_templates(npose=npose, log=log)
    net = build_network(npose=npose)
    with open(os.path.join(out_dir, 'network.json'), 'w') as fh:
        json.dump(net, fh, indent=1)
    np.savez(os.path.join(out_dir, 'weights.npz'), **build_weights(templates))
    log('wrote %d templates x %d features' % templates.shape)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
