"""Hand model + renderer + letter poses (this text is merged into build.py)."""
import numpy as np

D2R = np.pi / 180.0
def Rx(a):
    c, s = np.cos(a * D2R), np.sin(a * D2R); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Ry(a):
    c, s = np.cos(a * D2R), np.sin(a * D2R); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a):
    c, s = np.cos(a * D2R), np.sin(a * D2R); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# ---------------------------------------------------------------------------------------------------------------------
# A right hand built from spheres.  Hand frame, centimetres: x = thumb side, y = towards the finger tips,
# z = out of the palm; the wrist is at the origin.  Sizes are ordinary adult hand proportions.
# ---------------------------------------------------------------------------------------------------------------------
FING = [  # knuckle position, phalanx lengths, radii at knuckle / middle joint / end joint / tip
    dict(mcp=(3.3, 8.8, 0.0), L=(4.2, 2.5, 2.1), r=(1.07, 1.03, 0.95, 0.83)),   # index
    dict(mcp=(1.1, 9.2, 0.0), L=(4.6, 2.9, 2.2), r=(1.07, 1.03, 0.95, 0.83)),   # middle
    dict(mcp=(-1.05, 8.8, 0.0), L=(4.3, 2.7, 2.1), r=(1.03, 0.98, 0.9, 0.78)),  # ring
    dict(mcp=(-3.1, 7.9, 0.0), L=(3.4, 2.0, 1.9), r=(0.93, 0.88, 0.8, 0.7)),    # little
]
CARP = [(2.5, 1.3, 0.0), (0.85, 1.0, 0.0), (-0.9, 1.0, 0.0), (-2.75, 1.3, 0.0)]  # heel of the palm
THUMB_CMC = np.array([2.5, 1.7, 0.9])
THUMB_L = (4.4, 3.2, 2.7)
THUMB_R = (1.3, 1.15, 1.08, 0.92)

def _seg(p0, p1, r0, r1, step=0.6):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / (step * min(r0, r1)))) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return p0 + t * (p1 - p0), (r0 + t[:, 0] * (r1 - r0))

def build_hand(fingers, thumb, arm, fl=1.0, fr=1.0, pw=1.0):
    """fingers: 4 x (spread, knuckle flex, middle flex, end flex, lift) in degrees / cm; thumb: 3 segment directions;
    arm: forearm direction; fl, fr, pw: finger length, finger thickness and palm width factors.
    Returns sphere centres (N,3), radii (N,), flags (N,) with 1 = forearm."""
    C, Rr, F = [], [], []
    def add(c, r, flag=0):
        C.append(c); Rr.append(r); F.append(np.full(len(r), flag))
    wid = np.array([pw, 1.0, 1.0])
    carp = np.array(CARP) * wid; mcp = np.array([g['mcp'] for g in FING]) * wid
    for u in np.linspace(0, 1, 6):                                   # palm slab
        row = carp + u * (mcp - carp)
        xs = np.linspace(0, 3, 6)
        pts = np.stack([np.interp(xs, np.arange(4), row[:, j]) for j in range(3)], 1)
        add(pts, np.full(len(pts), 1.35 + 0.1 * (1 - u)))
    add(*_seg(np.array((-2.75, 1.9, 0.4)) * wid, np.array((-3.3, 6.3, 0.3)) * wid, 1.5, 1.2))   # little-finger side heel
    th = [THUMB_CMC * wid]
    for L, d in zip(THUMB_L, thumb):
        d = np.asarray(d, float); th.append(th[-1] + L * d / np.linalg.norm(d))
    add(*_seg(np.array((2.1, 1.9, 0.7)) * wid, th[1], 1.75, 1.3))                              # ball of the thumb
    for i in range(3):
        add(*_seg(th[i], th[i + 1], THUMB_R[i] * fr, THUMB_R[i + 1] * fr))
    zax = np.array([0, 0, 1.0])
    for k in range(4):
        ab, f1, f2, f3, lift = fingers[k]
        g = FING[k]
        d0 = np.array([np.sin(ab * D2R), np.cos(ab * D2R), 0.0])
        p = [mcp[k] + np.array([0, 0, lift])]
        a = 0.0
        for L, f in zip(g['L'], (f1, f2, f3)):
            a += f
            p.append(p[-1] + fl * L * (np.cos(a * D2R) * d0 + np.sin(a * D2R) * zax))
        for i in range(3):
            add(*_seg(p[i], p[i + 1], g['r'][i] * fr, g['r'][i + 1] * fr))
    a = np.asarray(arm, float); a = a / np.linalg.norm(a)
    side = np.cross(a, zax); side = side / (np.linalg.norm(side) + 1e-9)
    for sgn in (-1, 1):                                              # forearm: two rows of spheres (wider than thick)
        p0 = np.array([0, 0.6, 0.0]) + sgn * 1.0 * side
        c, r = _seg(p0, p0 + 16.0 * a + sgn * 0.6 * side, 2.1, 2.8, step=0.5)
        add(c, r, 1)
    return np.concatenate(C), np.concatenate(Rr), np.concatenate(F)

def _box(a, k):
    p = np.pad(a, k)
    c = np.pad(np.cumsum(np.cumsum(p, 0), 1), ((1, 0), (1, 0)))
    n = 2 * k + 1
    return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]

def render(C, Rr, F, R, scale, cx, cy, res, light, amb, hand, bg, sleeve, occ):
    """Orthographic depth-buffer picture of the spheres. R: rotation hand frame -> image frame (x right, y up, z to
    the viewer). scale: pixels (of the 28-pixel frame) per cm; (cx, cy): where the wrist lands. Returns (res, res)."""
    f = res / 28.0
    P = C @ R.T
    u = (cx + scale * P[:, 0]) * f; v = (cy - scale * P[:, 1]) * f; zc = P[:, 2] * scale * f; rr = Rr * scale * f
    keep = (u + rr > 0) & (u - rr < res) & (v + rr > 0) & (v - rr < res)
    u, v, zc, rr, Fk = u[keep].astype(np.float32), v[keep].astype(np.float32), zc[keep].astype(np.float32), rr[keep].astype(np.float32), F[keep]
    g = (np.arange(res) + 0.5).astype(np.float32)
    dx = g[None, :] - u[:, None]
    dy = g[None, :] - v[:, None]
    h2 = (rr ** 2)[:, None, None] - (dy ** 2)[:, :, None] - (dx ** 2)[:, None, :]
    z = np.where(h2 > 0, zc[:, None, None] + np.sqrt(np.maximum(h2, 0)), np.float32(-1e9))
    ii = z.argmax(0)
    zmax = np.take_along_axis(z, ii[None], 0)[0]
    mask = zmax > -1e8
    yy, xx = np.meshgrid(g, g, indexing='ij')
    nz = (zmax - zc[ii]) / rr[ii]; nx = (xx - u[ii]) / rr[ii]; ny = -(yy - v[ii]) / rr[ii]
    l = np.asarray(light, float); l = l / np.linalg.norm(l)
    sh = np.sqrt(np.maximum(nx * l[0] + ny * l[1] + nz * l[2], 0))    # soft wrap-around light: only rims go dark
    col = hand * (amb + (1 - amb) * sh)
    # parts lying behind their surroundings (creases, hollows) get less light
    m = mask.astype(float); zz = np.where(mask, zmax, 0.0)
    k = max(1, int(round(scale * f)))
    zavg = _box(zz, k) / np.maximum(_box(m, k), 1e-6)
    dep = np.maximum(zavg - zmax, 0) / (scale * f)
    col = col * (1 - occ * (1 - np.exp(-dep / 0.8)))
    if sleeve is not None:
        col = np.where(Fk[ii] == 1, sleeve * (0.6 + 0.4 * sh), col)
    return np.where(mask, col, bg).astype(np.float64)

def _area_matrix(n_out, n_in):
    """row i averages the input pixels covered by output pixel i (box filter resampling)."""
    A = np.zeros((n_out, n_in)); k = n_in / n_out
    for i in range(n_out):
        lo, hi = i * k, (i + 1) * k
        for j in range(int(np.floor(lo)), int(np.ceil(hi))):
            A[i, j] = min(hi, j + 1) - max(lo, j)
    return A / k
RES = 42
A_DOWN = _area_matrix(28, RES)

def blur(img, sigma):
    r = int(np.ceil(3 * sigma)); x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    p = np.pad(img, r, mode='edge')
    t = sum(k[i] * p[:, i:i + img.shape[1]] for i in range(2 * r + 1))
    return sum(k[i] * t[i:i + img.shape[0], :] for i in range(2 * r + 1))

# ---------------------------------------------------------------------------------------------------------------------
# The 24 static letters.  fingers: index, middle, ring, little as (spread, knuckle flex, middle flex, end flex[, lift]).
# thumb: directions of its three segments.  view: (yaw, pitch, roll) with yaw 0 = palm to the viewer, +90 = palm to
# the viewer's right, 180 = back of the hand to the viewer, roll + = counter-clockwise; or ('axes', image direction of
# the thumb side, image direction of the fingers).  arm: forearm direction in the hand frame.
# ---------------------------------------------------------------------------------------------------------------------
CURL = (0, 85, 100, 60)
EXT = (0, 3, 4, 4)
T_SIDE = [(0.55, 0.75, 0.3), (0.12, 1, 0.12), (-0.05, 1, 0.05)]       # thumb along the side of the fist, pointing up
T_PALM = [(0.3, 0.85, 0.35), (-0.8, 0.5, 0.0), (-0.9, 0.3, -0.1)]     # folded flat across the palm
T_OUT = [(0.75, 0.6, 0.25), (0.95, 0.3, 0.1), (1, 0.12, 0.0)]         # stuck out sideways
T_FRONT = [(0.35, 0.6, 0.72), (-0.5, 0.6, 0.62), (-0.95, 0.3, 0.0)]   # wrapped over the front of the curled fingers
T_HOLD = [(0.3, 0.75, 0.6), (-0.7, 0.42, 0.58), (-0.9, 0.3, 0.3)]     # holding down the curled ring / little fingers
T_PINCH = [(0.3, 0.6, 0.75), (0.0, 0.8, 0.6), (-0.2, 0.9, 0.4)]       # reaching forward to touch a finger
DOWN = (0, -1, 0)
POSES = {
 'A': dict(fingers=[CURL, CURL, CURL, CURL], thumb=T_SIDE, view=(0, 0, -8), arm=DOWN),
 'B': dict(fingers=[(0, 2, 3, 3), (0, 2, 3, 3), (0, 2, 3, 3), (-2, 2, 3, 3)], thumb=T_PALM, view=(0, 0, 0), arm=DOWN, tight=1),
 'C': dict(fingers=[(0, 30, 55, 40), (0, 32, 55, 40), (0, 34, 55, 40), (0, 36, 55, 40)],
           thumb=[(0.3, 0.55, 0.78), (0.15, 0.7, 0.7), (0.0, 0.8, 0.6)], view=(50, 0, -26), arm=DOWN),
 'D': [dict(fingers=[EXT, (0, 75, 95, 50), (0, 78, 95, 50), (0, 80, 95, 50)],
            thumb=[(0.3, 0.7, 0.65), (-0.55, 0.5, 0.67), (-0.85, 0.4, 0.3)], view=(v, 0, 0), arm=DOWN) for v in (0, 0, 40)],
 'E': dict(fingers=[(0, 10, 90, 85)] * 4, thumb=[(0.25, 0.85, 0.45), (-0.8, 0.55, 0.15), (-0.95, 0.3, 0.0)], view=(0, 0, 0), arm=DOWN),
 'F': dict(fingers=[(5, 50, 70, 35), (12, 4, 4, 4), (-6, 2, 4, 4), (-28, 2, 4, 4)],
           thumb=[(0.25, 0.5, 0.83), (0.0, 0.8, 0.6), (-0.05, 0.95, 0.32)], view=(32, 0, 0), arm=DOWN),
 'G': dict(fingers=[EXT, (0, 65, 100, 70), (0, 67, 100, 70), (0, 70, 100, 70)],
           thumb=[(0.35, 0.6, 0.72), (0.15, 0.95, 0.3), (0.05, 1, 0.1)], view=(180, 0, -80), arm=(-0.75, -0.65, 0)),
 'H': dict(fingers=[(-1, 3, 4, 4), (1, 3, 4, 4), (0, 67, 100, 70), (0, 70, 100, 70)],
           thumb=T_HOLD, view=(180, 0, -80), arm=(-0.75, -0.65, 0), tight=1),
 'I': dict(fingers=[CURL, CURL, CURL, (-14, 0, 4, 4)], thumb=T_FRONT, view=(0, 0, 0), arm=DOWN),
 'K': dict(fingers=[EXT, (-4, 55, 10, 5), CURL, CURL], thumb=T_PINCH, view=(-28, 0, 0), arm=DOWN),
 'L': dict(fingers=[EXT, CURL, CURL, CURL], thumb=T_OUT, view=(0, 0, 0), arm=DOWN),
 'M': dict(fingers=[(0, 80, 80, 30), (0, 80, 80, 30), (0, 80, 80, 30), CURL],
           thumb=[(0.3, 0.8, 0.5), (-0.85, 0.45, 0.25), (-0.9, 0.3, 0.0)], view=(-20, 25, -8), arm=DOWN),
 'N': dict(fingers=[(0, 80, 80, 30), (0, 80, 80, 30), CURL, CURL],
           thumb=[(0.3, 0.8, 0.5), (-0.7, 0.6, 0.3), (-0.75, 0.6, 0.2)], view=(-20, 25, -8), arm=DOWN),
 'O': dict(fingers=[(0, 65, 80, 45), (0, 67, 80, 45), (0, 69, 80, 45), (0, 72, 80, 45)],
           thumb=[(0.2, 0.3, 0.93), (0.0, 0.9, 0.4), (-0.2, 0.9, -0.4)], view=(82, 0, -18), arm=DOWN),
 'P': dict(fingers=[(0, 25, 5, 5), (-4, 80, 20, 10), CURL, CURL], thumb=T_PINCH, view=(-28, 0, 65), arm=DOWN),
 'Q': dict(fingers=[EXT, (0, 90, 100, 60), (0, 90, 100, 60), (0, 90, 100, 60)],
           thumb=[(0.55, 0.5, 0.67), (0.6, 0.6, 0.5), (0.5, 0.75, 0.4)],
           view=('axes', (0.85, -0.1, 0.5), (-0.35, -0.75, 0.55)), arm=(-0.15, 0.1, 1.0)),
 'R': dict(fingers=[(-9, 0, 4, 4, 0.7), (9, 0, 4, 4, -0.2), CURL, CURL], thumb=T_HOLD, view=(0, 0, -5), arm=DOWN, tight=1),
 'S': dict(fingers=[CURL, CURL, CURL, CURL], thumb=T_FRONT, view=(-20, 0, -8), arm=DOWN),
 'T': dict(fingers=[(0, 10, 30, 20), (0, 55, 90, 50), (0, 60, 90, 50), (0, 65, 90, 50)],
           thumb=[(0.3, 0.9, 0.3), (0.5, 0.85, -0.1), (1.0, 0.1, -0.2)], view=(180, 0, -90), arm=(-0.75, -0.65, 0)),
 'U': dict(fingers=[(-1, 0, 3, 3), (1, 0, 3, 3), CURL, CURL], thumb=T_HOLD, view=(0, 0, -5), arm=DOWN, tight=1),
 'V': dict(fingers=[(16, 0, 3, 3), (-14, 0, 3, 3), CURL, CURL], thumb=T_HOLD, view=(0, 0, 0), arm=DOWN),
 'W': dict(fingers=[(17, 0, 3, 3), (0, 0, 3, 3), (-16, 0, 3, 3), CURL], thumb=T_PALM, view=(0, 0, 0), arm=DOWN),
 'X': dict(fingers=[(0, 15, 85, 55), CURL, CURL, CURL], thumb=T_FRONT, view=(38, 0, -8), arm=DOWN),
 'Y': dict(fingers=[CURL, CURL, CURL, (-28, 0, 3, 3)],
           thumb=[(0.8, 0.5, 0.3), (0.9, 0.42, 0.1), (0.95, 0.3, 0.0)], view=(0, 0, 0), arm=DOWN),
}
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

def base_rot(view):
    if view[0] == 'axes':
        x = np.asarray(view[1], float); y = np.asarray(view[2], float)
        y = y / np.linalg.norm(y); x = x - (x @ y) * y; x = x / np.linalg.norm(x)
        return np.stack([x, y, np.cross(x, y)], 1)
    yaw, pitch, roll = view
    return Rz(roll) @ Rx(pitch) @ Ry(yaw)

def sample(letter, rng, res=RES, jit=1.0):
    """One picture of the letter with random signer, pose, camera, framing and lighting variation."""
    p = POSES[letter]
    if isinstance(p, list):
        p = p[int(rng.integers(len(p)))]
    fingers = []
    for fp in p['fingers']:
        fp = list(fp) + [0.0] * (5 - len(fp))
        fp[0] += jit * rng.normal(0, 1.5 if p.get('tight') else 3.0)
        for j in (1, 2, 3):
            fp[j] += jit * rng.normal(0, 7.0)
        fingers.append(fp)
    thumb = [np.asarray(d, float) / np.linalg.norm(d) + jit * rng.normal(0, 0.10, 3) for d in p['thumb']]
    arm = np.asarray(p['arm'], float); arm = arm / np.linalg.norm(arm) + jit * rng.normal(0, 0.16, 3)
    C, Rr, F = build_hand(fingers, thumb, arm, fl=1 + jit * rng.uniform(-0.07, 0.07), fr=1 + jit * rng.uniform(-0.1, 0.1),
                          pw=1 + jit * rng.uniform(-0.06, 0.06))
    R = Rz(jit * rng.normal(0, 9)) @ Rx(jit * rng.normal(0, 9)) @ Ry(jit * rng.normal(0, 12)) @ base_rot(p['view'])
    # framing: a square crop around the hand.  Long hands span 0.86 - 1.08 of the frame; compact ones (fists) are not
    # blown up beyond 1.55 - 1.95 pixels per cm, so that some wrist shows.  The hand sits near the middle; when it is
    # smaller than the frame it stays close to the edge away from the forearm, which leaves through the other edge.
    P = C @ R.T; sel = F == 0
    x0, x1 = (P[sel, 0] - Rr[sel]).min(), (P[sel, 0] + Rr[sel]).max()
    y0, y1 = (P[sel, 1] - Rr[sel]).min(), (P[sel, 1] + Rr[sel]).max()
    s = min(rng.uniform(0.86, 1.08) * 28.0 / max(x1 - x0, y1 - y0), rng.uniform(1.55, 1.95))
    a = R @ (arm / np.linalg.norm(arm))
    w, h = s * (x1 - x0), s * (y1 - y0)
    margin = rng.uniform(1.0, 4.0); jx, jy = jit * rng.uniform(-2.0, 2.0, 2)
    left, top = (28 - w) / 2 + jx, (28 - h) / 2 + jy
    if abs(a[1]) >= abs(a[0]):
        if h < 28 - 2 * margin:
            top = margin if a[1] < 0 else 28 - margin - h
    elif w < 28 - 2 * margin:
        left = 28 - margin - w if a[0] < 0 else margin
    cx = left - s * x0; cy = top + s * y1
    img = render(C, Rr, F, R, s, cx, cy, res=res, light=(rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.6), 1.0),
                 amb=rng.uniform(0.5, 0.75), hand=rng.uniform(0.6, 0.95), bg=rng.uniform(0.15, 0.9),
                 sleeve=(rng.uniform(0.05, 0.9) if rng.random() < 0.3 else None), occ=rng.uniform(0.4, 0.8))
    A = A_DOWN if res == RES else _area_matrix(28, res)
    return blur(A @ img @ A.T, rng.uniform(0.5, 0.9))

# ---------------------------------------------------------------------------------------------------------------------
# The network.  Fixed, formula-made feature layers (oriented edges -> pooled cells -> local contrast normalisation),
# followed by a bank of templates, one per generated picture, a max over each class's templates and an average.
# ---------------------------------------------------------------------------------------------------------------------
N_ORI = 6            # edge orientations, 30 degrees apart
CELLS = 10           # 10 x 10 pooled cells
EPS_LOCAL = 0.01     # local-contrast floor (in squared cell-energy units): edges much weaker than this fade out
PER_CLASS = 4000     # generated pictures (= templates) per letter
GROUPS = 10          # each letter's templates are split into 10 sets; the class score is the mean of the 10 best matches
SEED = 20240

def make_feature_layers():
    layers, W = [], {}
    layers.append(dict(name="n0", op="norm", inputs=["image"]))
    # c1: first derivative of a Gaussian (sigma 1) along 12 directions (6 orientations x 2 signs), 5x5, no padding
    ax = np.arange(-2, 3)
    dy, dx = np.meshgrid(ax, ax, indexing='ij')
    g = np.exp(-(dx ** 2 + dy ** 2) / 2.0)
    unit = (np.where(dx > 0, 1.0, np.where(dx == 0, 0.5, 0.0)) * dx * g).sum() - (np.where(dx < 0, 1.0, 0.0) * 0).sum()
    unit = (dx * g * (dx > 0)).sum()                      # response of the 0-degree filter to a unit step edge
    k = np.zeros((2 * N_ORI, 1, 5, 5))
    for j in range(N_ORI):
        th = j * np.pi / N_ORI
        k[j, 0] = (dx * np.cos(th) + dy * np.sin(th)) * g / unit
        k[j + N_ORI, 0] = -k[j, 0]
    W["c1.W"], W["c1.b"] = k, np.zeros(2 * N_ORI)
    layers.append(dict(name="c1", op="conv", inputs=["n0"], out=2 * N_ORI, kernel=[5, 5], stride=1, padding=0))
    layers.append(dict(name="r1", op="relu", inputs=["c1"]))
    # c2: unsigned, orientation-sharpened edge energy: |r(theta)| - |r(theta + 90)|, kept where positive
    m = np.zeros((N_ORI, 2 * N_ORI, 1, 1))
    for j in range(N_ORI):
        jp = (j + N_ORI // 2) % N_ORI
        m[j, j, 0, 0] = m[j, j + N_ORI, 0, 0] = 1.0
        m[j, jp, 0, 0] = m[j, jp + N_ORI, 0, 0] = -1.0
    W["c2.W"], W["c2.b"] = m, np.zeros(N_ORI)
    layers.append(dict(name="c2", op="conv", inputs=["r1"], out=N_ORI, kernel=[1, 1], stride=1, padding=0))
    layers.append(dict(name="r2", op="relu", inputs=["c2"]))
    # p1: pool each orientation over overlapping 6x6 windows with a tent weight, stride 2 -> 10 x 10 cells
    t = np.array([1, 2, 3, 3, 2, 1.0]); t = t / t.sum()
    p = np.zeros((N_ORI, N_ORI, 6, 6))
    for j in range(N_ORI):
        p[j, j] = np.outer(t, t)
    W["p1.W"], W["p1.b"] = p, np.zeros(N_ORI)
    layers.append(dict(name="p1", op="conv", inputs=["r2"], out=N_ORI, kernel=[6, 6], stride=2, padding=0))
    # d1: lay the cells out so that each cell becomes one "channel" holding itself and its 3x3 neighbourhood, each
    # value once with + and once with - sign (so the channel mean is 0 and `norm` divides by the local RMS energy).
    # Scaling by sc turns norm's built-in 1e-5 into the floor EPS_LOCAL.
    nc = CELLS * CELLS; per = 9 * 2 * N_ORI
    sc = np.sqrt(1e-5 / EPS_LOCAL)
    d = np.zeros((nc * per, N_ORI * nc))
    offs = [(0, 0)] + [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    for cy in range(CELLS):
        for cx in range(CELLS):
            c = cy * CELLS + cx
            for nb, (a, b) in enumerate(offs):
                y, x = cy + a, cx + b
                if not (0 <= y < CELLS and 0 <= x < CELLS):
                    continue
                for sgn in (0, 1):
                    for o in range(N_ORI):
                        d[c * per + (2 * nb + sgn) * N_ORI + o, o * nc + y * CELLS + x] = sc * (1 - 2 * sgn)
    W["d1.W"], W["d1.b"] = d, np.zeros(nc * per)
    layers.append(dict(name="d1", op="dense", inputs=["p1"], out=nc * per))
    layers.append(dict(name="rs1", op="reshape", inputs=["d1"], shape=[nc, per, 1]))
    layers.append(dict(name="ln", op="norm", inputs=["rs1"]))
    layers.append(dict(name="rs2", op="reshape", inputs=["ln"], shape=[nc, 18, N_ORI]))
    layers.append(dict(name="sel", op="maxpool", inputs=["rs2"], size=[1, 1], stride=[18, 1]))   # keep row 0: the cell itself
    layers.append(dict(name="rs3", op="reshape", inputs=["sel"], shape=[1, nc, N_ORI]))
    layers.append(dict(name="gn", op="norm", inputs=["rs3"]))                                     # standardise the whole descriptor
    return layers, W

# --- a small numpy interpreter with the same semantics as the scorer's building blocks (used to make the templates) ---
def _conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, i:i + stride * ho:stride, j:j + stride * wo:stride] for i in range(kh) for j in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]

def _pool(x, size, stride, reduce):
    n, c, h, w = x.shape
    (sh, sw), (th, tw) = size, stride
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, i:i + th * ho:th, j:j + tw * wo:tw] for i in range(sh) for j in range(sw)], axis=0)
    return win.max(0) if reduce == "max" else win.mean(0)

def run_layers(layers, W, images):
    vals = {"image": images}
    for L in layers:
        ins = [vals[i] for i in L["inputs"]]; op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = _conv(ins[0], W[name + ".W"], W[name + ".b"], L.get("stride", 1), L.get("padding", 0))
        elif op == "dense":
            vals[name] = ins[0].reshape(len(ins[0]), -1) @ W[name + ".W"].T + W[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op in ("maxpool", "avgpool"):
            vals[name] = _pool(ins[0], L["size"], L["stride"], op[:3])
        elif op == "add":
            vals[name] = sum(ins)
        elif op == "reshape":
            vals[name] = ins[0].reshape((len(ins[0]),) + tuple(L["shape"]))
        elif op == "norm":
            x = ins[0]
            mu = x.mean(axis=(2, 3), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
    return vals[layers[-1]["name"]]

def build(out_dir, per_class=PER_CLASS, verbose=True):
    import json, os, time
    t0 = time.time()
    layers, W = make_feature_layers()
    nfeat = N_ORI * CELLS * CELLS
    rng = np.random.default_rng(SEED)
    T = np.zeros((24 * per_class, nfeat), np.float32)
    for k, letter in enumerate(LETTERS):
        imgs = np.stack([sample(letter, rng) for _ in range(per_class)])[:, None]
        imgs = np.clip(imgs, 0.0, 1.0)
        for i in range(0, per_class, 500):
            # each generated picture is its own template: its descriptor is already standardised by itself alone
            # (zero mean, unit variance over its own 600 numbers); dividing by 600 makes the dot product a correlation.
            f = run_layers(layers, W, imgs[i:i + 500]).reshape(-1, nfeat)
            T[k * per_class + i:k * per_class + i + len(f)] = f / nfeat
        if verbose:
            print(letter, round(time.time() - t0, 1), flush=True)
    W["tm.W"], W["tm.b"] = T, np.zeros(24 * per_class)
    layers.append(dict(name="tm", op="dense", inputs=["gn"], out=24 * per_class))
    gs = per_class // GROUPS
    layers.append(dict(name="rs4", op="reshape", inputs=["tm"], shape=[1, 24 * GROUPS, gs]))
    layers.append(dict(name="mx", op="maxpool", inputs=["rs4"], size=[1, gs], stride=[1, gs]))
    layers.append(dict(name="rs5", op="reshape", inputs=["mx"], shape=[1, 24, GROUPS]))
    layers.append(dict(name="av", op="avgpool", inputs=["rs5"], size=[1, GROUPS], stride=[1, GROUPS]))
    layers.append(dict(name="out", op="reshape", inputs=["av"], shape=[24]))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as fh:
        json.dump({"layers": layers}, fh, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})
    if verbose:
        print("done", round(time.time() - t0, 1), "s;", sum(v.size for v in W.values()), "weights", flush=True)

if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
