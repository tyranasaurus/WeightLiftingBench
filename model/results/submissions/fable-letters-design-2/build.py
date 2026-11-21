"""Parametric 3D hand (capsules + ellipsoid palm), orthographic z-buffer renderer.  No data is read here:
every number is typed from general knowledge of hand anatomy and of the ASL manual alphabet."""
import numpy as np

D2R = np.pi / 180.0
NAMES = "ABCDEFGHIKLMNOPQRSTUVWXY"

# ---------------------------------------------------------------- anatomy (centimetres, hand frame)
# hand frame: origin = wrist centre, +x = thumb side, +y = towards the finger tips, +z = out of the palm
MCP = np.array([[3.3, 9.3, 0.0], [1.1, 9.6, 0.0], [-1.1, 9.0, 0.0], [-3.2, 8.1, 0.0]])   # index, middle, ring, little
FLEN = np.array([[4.0, 2.4, 1.5], [4.4, 2.7, 1.6], [4.1, 2.6, 1.6], [3.3, 1.9, 1.4]])       # phalanx lengths
FRAD = np.array([[1.05, 0.96, 0.89, 0.80], [1.07, 0.98, 0.90, 0.82], [1.03, 0.94, 0.87, 0.78], [0.92, 0.83, 0.76, 0.69]])
CARP = np.array([[2.3, 1.5, 0.0], [0.8, 1.5, 0.0], [-0.8, 1.5, 0.0], [-2.4, 1.5, 0.0]])    # wrist ends of the metacarpals
THUMB0 = np.array([2.5, 1.7, 0.6])                                                         # thumb base joint
TLEN = np.array([4.6, 3.3, 2.3])
TRAD = np.array([1.55, 1.18, 1.06, 0.95])


def sph(az, el):
    """unit vector from azimuth (0 = +y, 90 = +x) and elevation towards +z, degrees."""
    az, el = az * D2R, el * D2R
    return np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], -1)


def hand_primitives(P):
    """P: dict of arrays with a leading batch axis B.
       fingers (B,4,4): spread, flex1, flex2, flex3 (degrees) for index, middle, ring, little
       thumb   (B,3,2): azimuth, elevation of the three thumb segments
       arm     (B,2):   azimuth, elevation of the forearm (0,0 = straight)
       size    (B,2):   finger length factor, thickness factor
    returns capsules a (B,N,3), b (B,N,3), ra (B,N), rb (B,N), kind (N,) and the ellipsoid palm."""
    F, T, A = P["fingers"], P["thumb"], P["arm"]
    B = F.shape[0]
    lf, tf = P["size"][:, 0], P["size"][:, 1]
    caps_a, caps_b, ra, rb, kind = [], [], [], [], []
    Z = np.array([0.0, 0.0, 1.0])

    def add(a, b, r0, r1, k):
        caps_a.append(a); caps_b.append(b); ra.append(r0 * tf); rb.append(r1 * tf); kind.append(k)

    one = np.ones(B)
    # forearm: two parallel capsules (the wrist is wider than it is thick)
    adir = sph(180.0 + A[:, 0], A[:, 1])
    side = np.cross(adir, Z); side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
    for s in (-1.0, 1.0):
        a = np.array([0.0, 0.9, 0.0]) + s * 1.7 * side
        b = a + 16.0 * adir + s * 0.5 * side
        add(a, b, 2.2 * one, 2.7 * one, 0)
    # palm: four metacarpals
    for k in range(4):
        add(np.tile(CARP[k], (B, 1)), np.tile(MCP[k], (B, 1)), 1.5 * one, (FRAD[k, 0] + 0.1) * one, 1)
    # thumb
    p = np.tile(THUMB0, (B, 1))
    for i in range(3):
        q = p + (TLEN[i] * (lf if i else one))[:, None] * sph(T[:, i, 0], T[:, i, 1])
        add(p, q, TRAD[i] * one, TRAD[i + 1] * one, 2 if i == 0 else 3)
        if i == 0:      # skin web between the thumb and the index metacarpal
            for w in (0.3, 0.6):
                s0 = (1 - w) * p + w * CARP[0]
                s1 = (1 - w) * (p + 0.8 * (q - p)) + w * (CARP[0] + 0.8 * (MCP[0] - CARP[0]))
                add(s0, s1, 1.2 * one, 1.0 * one, 2)
        p = q
    # fingers
    for k in range(4):
        d0 = sph(F[:, k, 0], 0 * F[:, k, 0])
        cum = np.cumsum(F[:, k, 1:], axis=1) * D2R
        p = np.tile(MCP[k], (B, 1))
        for i in range(3):
            d = np.cos(cum[:, i])[:, None] * d0 + np.sin(cum[:, i])[:, None] * Z
            q = p + (FLEN[k, i] * lf)[:, None] * d
            add(p, q, FRAD[k, i] * one, FRAD[k, i + 1] * one, 4 + k)
            p = q
    return (np.stack(caps_a, 1), np.stack(caps_b, 1), np.stack(ra, 1), np.stack(rb, 1), np.array(kind))


def rot(yaw, pitch, roll):
    """R = Rz(roll) Rx(pitch) Ry(yaw); arrays (B,) in degrees -> (B,3,3).
    yaw twists the hand about its own long axis, pitch tips the fingers towards the viewer, roll turns it in the image."""
    y, p, r = yaw * D2R, pitch * D2R, roll * D2R
    o, z = np.ones_like(y), np.zeros_like(y)
    Ry = np.stack([np.stack([np.cos(y), z, np.sin(y)], -1), np.stack([z, o, z], -1), np.stack([-np.sin(y), z, np.cos(y)], -1)], -2)
    Rx = np.stack([np.stack([o, z, z], -1), np.stack([z, np.cos(p), -np.sin(p)], -1), np.stack([z, np.sin(p), np.cos(p)], -1)], -2)
    Rz = np.stack([np.stack([np.cos(r), -np.sin(r), z], -1), np.stack([np.sin(r), np.cos(r), z], -1), np.stack([z, z, o], -1)], -2)
    return Rz @ Rx @ Ry


def render(P, S=56, out=28, full=False):
    """Render a batch of hands. Returns images (B,out,out) in [0,1] and a hand mask."""
    a, b, ra, rb, kind = hand_primitives(P)
    B, N = ra.shape
    R = rot(P["rot"][:, 0], P["rot"][:, 1], P["rot"][:, 2])
    a = np.einsum("bij,bnj->bni", R, a); b = np.einsum("bij,bnj->bni", R, b)
    # palm filler ellipsoid: centre and semi-axes in the hand frame
    pc = np.einsum("bij,j->bi", R, np.array([0.0, 5.2, 0.0]))
    pax = np.array([3.9, 4.6, 1.5])
    M = np.einsum("bij,j,bkj->bik", R, 1.0 / pax ** 2, R)
    # framing: bounding box of the hand proper (no forearm), then scale / place it in the frame
    hp = np.concatenate([a[:, 2:], b[:, 2:]], 1); hr = np.concatenate([ra[:, 2:], rb[:, 2:]], 1)
    x0, x1 = (hp[:, :, 0] - hr).min(1), (hp[:, :, 0] + hr).max(1)
    y0, y1 = (hp[:, :, 1] - hr).min(1), (hp[:, :, 1] + hr).max(1)
    sc = P["frame"][:, 0]                                   # pixels (of the 28 grid) per cm
    xm, ym = P["anchor"]                                    # how the hand's bounding box is placed in the frame
    fx, fy = P["frame"][:, 1], P["frame"][:, 2]
    # world x of the image centre column, world y of the image centre row
    if xm == "c":   wx = (x0 + x1) / 2 - fx / sc
    elif xm == "l": wx = x0 + (out / 2.0 - fx) / sc
    else:           wx = x1 - (out / 2.0 - fx) / sc
    if ym == "c":   wy = (y0 + y1) / 2 + fy / sc
    elif ym == "t": wy = y1 - (out / 2.0 - fy) / sc
    else:           wy = y0 + (out / 2.0 - fy) / sc
    k = S / out
    g = (np.arange(S) + 0.5) / k
    X = (g[None, None, :] - out / 2.0) / sc[:, None, None] + wx[:, None, None]      # cm, world x
    Y = -(g[None, :, None] - out / 2.0) / sc[:, None, None] + wy[:, None, None]     # cm, world y (up)
    X = np.broadcast_to(X, (B, S, S)); Y = np.broadcast_to(Y, (B, S, S))
    zbuf = np.full((B, S, S), -1e3); nx = np.zeros((B, S, S)); ny = np.zeros((B, S, S)); nz = np.zeros((B, S, S))
    kid = np.zeros((B, S, S), np.int8)

    def put(z, hit, ex, ey, ez, kk):
        nonlocal zbuf, nx, ny, nz, kid
        w = hit & (z > zbuf)
        zbuf = np.where(w, z, zbuf); nx = np.where(w, ex, nx); ny = np.where(w, ey, ny); nz = np.where(w, ez, nz)
        kid = np.where(w, kk, kid)

    for n in range(N):
        ax, ay, az = [a[:, n, i][:, None, None] for i in range(3)]
        dx, dy, dz = [(b[:, n, i] - a[:, n, i])[:, None, None] for i in range(3)]
        r0, r1 = ra[:, n][:, None, None], rb[:, n][:, None, None]
        l2 = dx * dx + dy * dy
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / (l2 + 1e-6), 0.0, 1.0)
        for tt in (t, 0.0 * t, 0.0 * t + 1.0):          # body, then the two end spheres (matters when foreshortened)
            ex, ey = X - (ax + tt * dx), Y - (ay + tt * dy)
            r = r0 + tt * (r1 - r0)
            q = 1.0 - (ex * ex + ey * ey) / (r * r)
            hit = q > 0
            ez = np.sqrt(np.maximum(q, 0.0))
            put(az + tt * dz + r * ez, hit, ex / r, ey / r, ez, kind[n])
    # ellipsoid palm
    ex, ey = X - pc[:, 0][:, None, None], Y - pc[:, 1][:, None, None]
    m = lambda i, j: M[:, i, j][:, None, None]
    qa = m(2, 2); qb = 2 * (m(0, 2) * ex + m(1, 2) * ey); qc = m(0, 0) * ex * ex + 2 * m(0, 1) * ex * ey + m(1, 1) * ey * ey - 1.0
    disc = qb * qb - 4 * qa * qc
    hit = disc > 0
    ez = (-qb + np.sqrt(np.maximum(disc, 0.0))) / (2 * qa)
    gx = m(0, 0) * ex + m(0, 1) * ey + m(0, 2) * ez; gy = m(1, 0) * ex + m(1, 1) * ey + m(1, 2) * ez; gz = m(2, 0) * ex + m(2, 1) * ey + m(2, 2) * ez
    gn = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9
    put(pc[:, 2][:, None, None] + ez, hit, gx / gn, gy / gn, gz / gn, 1)

    mask = zbuf > -500
    # shading: ambient + diffuse light, contact shadows where a neighbour is much nearer the viewer
    L = P["light"]                                           # (B,5): lx, ly, ambient, shadow strength, sleeve flag (unused)
    lv = np.stack([L[:, 0], L[:, 1], np.ones(B)], -1); lv /= np.linalg.norm(lv, axis=1, keepdims=True)
    dif = np.maximum(nx * lv[:, 0][:, None, None] + ny * lv[:, 1][:, None, None] + nz * lv[:, 2][:, None, None], 0.0)
    amb = L[:, 2][:, None, None]
    shade = amb + (1 - amb) * dif
    zz = np.where(mask, zbuf, -1e3)
    w = int(round(1.2 * k))
    zmax = zz.copy()
    for s in range(1, w + 1):
        for ax_ in (1, 2):
            for sg in (-s, s):
                zmax = np.maximum(zmax, np.roll(zz, sg, axis=ax_))
    occ = np.clip((zmax - zz - 0.8) / 2.5, 0.0, 1.0)
    shade = shade * (1.0 - L[:, 3][:, None, None] * occ)
    # no limb darkening along the outer outline (creases between touching parts stay dark)
    mb = mask * 1.0
    wb = int(round(1.0 * k))
    acc = np.zeros_like(mb); cnt = 0
    for sy in range(-wb, wb + 1):
        for sx in range(-wb, wb + 1):
            acc += np.roll(np.roll(mb, sy, 1), sx, 2); cnt += 1
    rim = np.clip((1.0 - acc / cnt) * 3.0, 0.0, 1.0)
    shade = shade + (1.0 - shade) * rim * (1.0 - occ)
    skin, bg = P["tone"][:, 0][:, None, None], P["tone"][:, 1][:, None, None]
    ds = lambda v: v.reshape(B, out, S // out, out, S // out).mean((2, 4))
    if not full:
        return ds(np.where(mask, skin * shade, bg)), ds(mask * 1.0)
    # extras for synthetic test scenes: hand shading, coverage, forearm coverage, distance from the wrist (cm)
    wr = np.einsum("bij,j->bi", R, np.array([0.0, 0.9, 0.0]))
    dist = np.sqrt((X - wr[:, 0][:, None, None]) ** 2 + (Y - wr[:, 1][:, None, None]) ** 2)
    return dict(shade=ds(np.where(mask, shade, 0.0)), mask=ds(mask * 1.0), arm=ds((mask & (kid == 0)) * 1.0),
                sleeve=[ds((mask & (kid == 0) & (dist > d)) * 1.0) for d in (2.5, 5.0, 8.0)])

"""Hand poses for the 24 static ASL letters, typed from knowledge of the manual alphabet.
fingers: (spread, mcp flex, pip flex, dip flex) for index, middle, ring, little; spread > 0 is towards the thumb.
thumb:   (azimuth, elevation) of metacarpal, proximal and distal segment (azimuth 0 = along the fingers, 90 = straight out
         sideways, negative = across the palm; elevation = out of the palm).
arm:     (deviation towards the little-finger side, flexion towards the palm) of the forearm at the wrist.
rot:     (yaw = twist about the hand's long axis, pitch = fingers tipped towards the viewer, roll = turn in the image, + = anticlockwise)
frame:   (pixels per centimetre, x placement, y placement); anchor says what the placements mean:
         x 'c' = box centre this far right of the middle, 'l' / 'r' = box edge this far from the left / right border;
         y 't' = top of the hand this many rows below the top border, 'c' = centre offset."""
import numpy as np

EXT = (0, 3, 4, 3)
CURL = (0, 80, 100, 70)
TH_SIDE = [(30, 25), (6, 14), (0, 5)]          # thumb lying against the side of the index (A)
TH_FRONT = [(20, 40), (-55, 25), (-75, 5)]     # thumb across the front of the curled fingers (S)
TH_PALM = [(15, 32), (-50, 16), (-75, 2)]      # thumb folded over the palm (B)
TH_OUT = [(52, 8), (70, 2), (76, 0)]           # thumb stuck out sideways (L, Y)
TH_ALONG = [(22, 28), (10, 12), (4, 2)]        # thumb pointing the same way as the index (G, Q)
TH_RING = [(22, 46), (-8, 40), (-28, 22)]      # thumb meeting curved finger tips (D, F, O)
TH_HOLD = [(12, 38), (-58, 24), (-80, 4)]      # thumb holding down curled fingers (U, V, W, R)

POSES = {
    "A": dict(fingers=[CURL, CURL, CURL, CURL], thumb=TH_SIDE, arm=(0, 0), rot=(0, 0, -10), frame=(2.0, 0.5, 2.5), anchor=('c', 't')),
    "B": dict(fingers=[(2, 2, 3, 2), (0, 2, 3, 2), (-1, 2, 3, 2), (-3, 2, 3, 2)], thumb=TH_PALM, arm=(0, 0), rot=(0, 0, -6), frame=(1.6, 0.5, 1.5), anchor=('c', 't')),
    "C": dict(fingers=[(0, 50, 25, 20), (0, 48, 25, 20), (0, 44, 28, 22), (0, 36, 32, 24)], thumb=[(30, 55), (10, 72), (0, 78)], arm=(0, 0), rot=(62, 0, -24), frame=(1.8, 0.0, 2.0), anchor=('c', 't')),
    "D": dict(fingers=[EXT, (0, 62, 72, 40), (0, 62, 72, 40), (0, 62, 72, 40)], thumb=TH_RING, arm=(0, 0), rot=(-12, 0, -6), frame=(1.6, 1.0, 1.5), anchor=('c', 't')),
    "E": dict(fingers=[(0, 68, 96, 62)] * 4, thumb=[(14, 34), (-60, 20), (-84, 0)], arm=(0, 0), rot=(0, 0, -8), frame=(1.85, 0.5, 2.5), anchor=('c', 't')),
    "F": dict(fingers=[(10, 48, 52, 28), (6, 4, 6, 4), (-12, 4, 6, 4), (-32, 4, 6, 4)], thumb=[(44, 28), (26, 34), (6, 28)], arm=(0, 0), rot=(0, 0, -8), frame=(1.55, 0.0, 1.5), anchor=('c', 't')),
    "G": dict(fingers=[EXT, CURL, CURL, CURL], thumb=TH_HOLD, arm=(48, 0), rot=(165, 0, -86), frame=(1.65, 1.5, 5.0), anchor=('r', 't')),
    "H": dict(fingers=[EXT, EXT, CURL, CURL], thumb=TH_HOLD, arm=(48, 0), rot=(165, 0, -86), frame=(1.65, 1.5, 5.0), anchor=('r', 't')),
    "I": dict(fingers=[CURL, CURL, CURL, (-8, 3, 4, 3)], thumb=TH_FRONT, arm=(0, 0), rot=(0, 0, -6), frame=(1.6, 1.0, 2.0), anchor=('c', 't')),
    "K": dict(fingers=[EXT, (0, 58, 6, 4), CURL, CURL], thumb=[(10, 42), (0, 30), (0, 18)], arm=(0, 0), rot=(-38, 0, -6), frame=(1.6, 1.0, 1.5), anchor=('c', 't')),
    "L": dict(fingers=[EXT, CURL, CURL, CURL], thumb=TH_OUT, arm=(0, 0), rot=(0, 0, -6), frame=(1.55, 0.0, 1.5), anchor=('c', 't')),
    "M": dict(fingers=[(0, 66, 86, 34), (0, 66, 86, 34), (0, 66, 86, 34), CURL], thumb=[(8, 36), (-62, 16), (-86, 0)], arm=(0, 0), rot=(0, 0, -14), frame=(1.8, 0.5, 2.5), anchor=('c', 't')),
    "N": dict(fingers=[(0, 66, 86, 34), (0, 66, 86, 34), CURL, CURL], thumb=[(10, 36), (-48, 18), (-70, 0)], arm=(0, 0), rot=(0, 0, -14), frame=(1.8, 0.5, 2.5), anchor=('c', 't')),
    "O": dict(fingers=[(0, 60, 60, 40), (0, 58, 58, 40), (0, 50, 62, 42), (0, 38, 66, 46)], thumb=[(25, 70), (0, 25), (-10, -20)], arm=(0, 0), rot=(72, 0, -24), frame=(1.95, 0.0, 2.0), anchor=('c', 't')),
    "P": dict(fingers=[(0, 20, 4, 3), (0, 84, 8, 6), CURL, CURL], thumb=[(12, 44), (4, 38), (0, 26)], arm=(0, 40), rot=(-45, 0, 70), frame=(1.8, 2.0, 4.0), anchor=('l', 't')),
    "Q": dict(fingers=[EXT, CURL, CURL, CURL], thumb=[(46, 14), (46, 6), (44, 0)], arm=(0, 60), rot=(180, 15, 160), frame=(1.8, 0.0, 2.5), anchor=('c', 't')),
    "R": dict(fingers=[(-13, 3, 4, 3), (11, 12, 4, 3), CURL, CURL], thumb=TH_HOLD, arm=(0, 0), rot=(0, 0, -6), frame=(1.6, 2.5, 1.5), anchor=('c', 't')),
    "S": dict(fingers=[CURL, CURL, CURL, CURL], thumb=TH_FRONT, arm=(0, 0), rot=(0, 0, -10), frame=(1.8, 0.5, 2.5), anchor=('c', 't')),
    "T": dict(fingers=[EXT, CURL, CURL, CURL], thumb=[(8, 12), (80, 5), (88, 0)], arm=(60, 0), rot=(170, 0, -88), frame=(1.55, 1.5, 2.0), anchor=('r', 't')),
    "U": dict(fingers=[(0, 3, 4, 3), (0, 3, 4, 3), CURL, CURL], thumb=TH_HOLD, arm=(0, 0), rot=(0, 0, -6), frame=(1.6, 2.5, 1.5), anchor=('c', 't')),
    "V": dict(fingers=[(14, 3, 4, 3), (-12, 3, 4, 3), CURL, CURL], thumb=TH_HOLD, arm=(0, 0), rot=(0, 0, -6), frame=(1.6, 2.5, 1.5), anchor=('c', 't')),
    "W": dict(fingers=[(18, 3, 4, 3), (0, 3, 4, 3), (-18, 3, 4, 3), CURL], thumb=[(6, 34), (-62, 22), (-82, 4)], arm=(0, 0), rot=(0, 0, -6), frame=(1.55, 1.0, 1.5), anchor=('c', 't')),
    "X": dict(fingers=[(0, 18, 92, 58), CURL, CURL, CURL], thumb=TH_FRONT, arm=(0, 0), rot=(62, 0, -26), frame=(1.8, 1.0, 2.0), anchor=('c', 't')),
    "Y": dict(fingers=[CURL, CURL, CURL, (-28, 3, 4, 3)], thumb=[(50, 6), (62, 2), (66, 0)], arm=(0, 0), rot=(0, 0, -6), frame=(1.45, 0.0, 3.0), anchor=('c', 't')),
}


def halton(n, d, start=1):
    """low-discrepancy points in [0,1)^d (radical inverse in the first d primes)."""
    out = np.zeros((n, d))
    for k, b in enumerate([2, 3, 5, 7, 11, 13][:d]):
        i = np.arange(start, start + n); f = 1.0; r = np.zeros(n)
        while i.max() > 0:
            f /= b; r += f * (i % b); i = i // b
        out[:, k] = r
    return out


def batch(name, n, rng=None, jit=1.0, light=(-0.25, 0.3, 0.65, 0.5), tone=(0.85, 0.55), harsh=False):
    """n variants of one letter: the first is the nominal pose, the rest are jittered (if rng is given)."""
    p = POSES[name]
    F = np.tile(np.array(p["fingers"], float), (n, 1, 1)); T = np.tile(np.array(p["thumb"], float), (n, 1, 1))
    A = np.tile(np.array(p["arm"], float), (n, 1)); R = np.tile(np.array(p["rot"], float), (n, 1))
    Fr = np.tile(np.array(p["frame"], float), (n, 1)); Sz = np.ones((n, 2)); Sz[:, 1] = 1.1
    Lt = np.tile(np.array(list(light) + [0.0]), (n, 1)); Tn = np.tile(np.array(tone, float), (n, 1))
    if rng is not None:
        j = jit * np.concatenate([np.zeros(1), np.ones(n - 1)])
        F[:, :, 0] += j[:, None] * rng.normal(0, 3.0, (n, 4))
        F[:, :, 1:] += j[:, None, None] * (rng.normal(0, 5.0, (n, 1, 3)) + rng.normal(0, 4.0, (n, 4, 3)))
        F[:, :, 1:] = np.maximum(F[:, :, 1:], 0.0)
        T += j[:, None, None] * (rng.normal(0, 5.0, (n, 1, 2)) + rng.normal(0, 4.0, (n, 3, 2)))
        A += j[:, None] * rng.normal(0, 9.0, (n, 2))
        R += j[:, None] * rng.normal(0, 1.0, (n, 3)) * np.array([10.0, 9.0, 4.0])
        h = halton(n, 4, int(rng.integers(1, 1000)))
        Fr[:, 0] *= np.exp(j * 0.22 * (2 * h[:, 0] - 1))
        Fr[:, 1] += j * 3.0 * (2 * h[:, 1] - 1)
        Fr[:, 2] += j * (6.0 * h[:, 2] - 2.5)
        R[:, 2] += j * 9.0 * (2 * h[:, 3] - 1)
        Sz[:, 0] *= np.exp(j * rng.normal(0, 0.04, n)); Sz[:, 1] *= np.exp(j * rng.normal(0, 0.07, n))
        Lt[:, 0] += j * rng.normal(0, 0.3, n); Lt[:, 1] += j * rng.normal(0, 0.3, n)
        Lt[:, 2] = np.clip(Lt[:, 2] + j * rng.normal(0, 0.12, n), 0.45, 0.9)
        Lt[:, 3] = np.clip(Lt[:, 3] + j * rng.normal(0, 0.15, n), 0.1, 0.9)
        if harsh:      # strong side light and deep shadows, for stress-test scenes
            Lt[:, 0] = rng.normal(0, 0.8, n); Lt[:, 1] = rng.normal(0.2, 0.6, n)
            Lt[:, 2] = rng.uniform(0.35, 0.95, n); Lt[:, 3] = rng.uniform(0.2, 0.95, n)
    return dict(fingers=F, thumb=T, arm=A, rot=R, frame=Fr, size=Sz, light=Lt, tone=Tn, anchor=p["anchor"])

"""Network layers (weights from formulas) and a local forward pass used on synthetic renders only."""
import numpy as np

K = 8            # unsigned edge orientations
FS = 5           # first filter size
SIG = 1.0        # Gaussian-derivative scale
THR_A, THR_B = 0.0, 2.0     # soft edge threshold and saturation (in standard deviations of the edge map)
CLIP1 = 2.5
POOL_K, POOL_S, POOL_P, POOL_SIG = 6, 2, 2, 1.5


def feature_layers():
    """returns (layers, weights) for image -> blurred, normalised oriented-edge maps (K, 12, 12)."""
    L, W = [], {}
    r = FS // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * SIG ** 2))
    gx, gy = -xx * g, -yy * g
    gx /= np.abs(gx).sum(); gy /= np.abs(gy).sum()
    w = np.zeros((2 * K, 1, FS, FS))
    for o in range(K):
        th = np.pi * o / K
        f = np.cos(th) * gx + np.sin(th) * gy
        w[o, 0], w[o + K, 0] = f, -f
    L.append(dict(name="edge", op="conv", inputs=["image"], out=2 * K, kernel=[FS, FS], stride=1, padding=0))
    W["edge.W"], W["edge.b"] = w, np.zeros(2 * K)
    L.append(dict(name="edge_r", op="relu", inputs=["edge"]))
    # oriented energy with cross-orientation inhibition: |r(th)| - |r(th + 90)|
    w = np.zeros((K, 2 * K, 1, 1))
    for o in range(K):
        p = (o + K // 2) % K
        w[o, o, 0, 0] = w[o, o + K, 0, 0] = 1.0
        w[o, p, 0, 0] = w[o, p + K, 0, 0] = -1.0
    L.append(dict(name="ori", op="conv", inputs=["edge_r"], out=K, kernel=[1, 1], stride=1, padding=0))
    W["ori.W"], W["ori.b"] = w, np.zeros(K)
    L.append(dict(name="ori_r", op="relu", inputs=["ori"]))
    n = 28 - FS + 1
    # contrast normalisation over all orientations and positions together, twice, clipping outliers in between
    L.append(dict(name="n1_in", op="reshape", inputs=["ori_r"], shape=[1, K * n, n]))
    L.append(dict(name="n1", op="norm", inputs=["n1_in"]))
    L.append(dict(name="clip", op="conv", inputs=["n1"], out=1, kernel=[1, 1], stride=1, padding=0))      # CLIP1 - z
    W["clip.W"], W["clip.b"] = -np.ones((1, 1, 1, 1)), np.array([CLIP1])
    L.append(dict(name="clip_r", op="relu", inputs=["clip"]))                                               # = CLIP1 - min(z, CLIP1)
    L.append(dict(name="n2", op="norm", inputs=["clip_r"]))                                                 # = -zscore(min(z, CLIP1))
    L.append(dict(name="n2_out", op="reshape", inputs=["n2"], shape=[K, n, n]))
    # soft threshold: clip(z - a, 0, b) = relu(z - a) - relu(z - a - b), with z = -n2
    w = np.zeros((2 * K, K, 1, 1)); b = np.zeros(2 * K)
    for o in range(K):
        w[o, o, 0, 0] = w[o + K, o, 0, 0] = -1.0
        b[o], b[o + K] = -THR_A, -THR_A - THR_B
    L.append(dict(name="thr", op="conv", inputs=["n2_out"], out=2 * K, kernel=[1, 1], stride=1, padding=0))
    W["thr.W"], W["thr.b"] = w, b
    L.append(dict(name="thr_r", op="relu", inputs=["thr"]))
    # spatial pooling: Gaussian blur, stride 2
    c = (POOL_K - 1) / 2.0
    yy, xx = np.mgrid[0:POOL_K, 0:POOL_K].astype(float)
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * POOL_SIG ** 2)); g /= g.sum()
    w = np.zeros((K, 2 * K, POOL_K, POOL_K))
    for o in range(K):
        w[o, o], w[o, o + K] = g, -g
    L.append(dict(name="pool", op="conv", inputs=["thr_r"], out=K, kernel=[POOL_K, POOL_K], stride=POOL_S, padding=POOL_P))
    W["pool.W"], W["pool.b"] = w, np.zeros(K)
    return L, W


# ---------------------------------------------------------------- local interpreter (same arithmetic as the scorer's)
def conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    cols = np.stack([xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c * kh * kw, ho * wo).transpose(0, 2, 1) @ W.reshape(o, -1).T
    return out.transpose(0, 2, 1).reshape(n, o, ho, wo) + b[None, :, None, None]


def pool(x, size, stride, reduce):
    (sh, sw), (th, tw) = size, stride
    n, c, h, w = x.shape
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    win = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if reduce == "max" else win.mean(0)


def run(layers, weights, images, upto=None):
    vals = {"image": images}
    for L in layers:
        ins = [vals[i] for i in L["inputs"]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = conv(ins[0], weights[name + ".W"], weights[name + ".b"], L.get("stride", 1), L.get("padding", 0))
        elif op == "dense":
            vals[name] = ins[0].reshape(len(ins[0]), -1) @ weights[name + ".W"].T + weights[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op in ("maxpool", "avgpool"):
            vals[name] = pool(ins[0], L["size"], L["stride"], op[:3])
        elif op == "add":
            vals[name] = sum(ins)
        elif op == "reshape":
            vals[name] = ins[0].reshape((len(ins[0]),) + tuple(L["shape"]))
        elif op == "norm":
            x = ins[0]
            mu = x.mean(axis=(2, 3), keepdims=True)
            vals[name] = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        if upto == name:
            return vals[name]
    return vals[layers[-1]["name"]]


# ---------------------------------------------------------------- template bank and output
def blur(img, s):
    r = 3; g = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); g /= g.sum()
    p = np.pad(img, ((0, 0), (r, r), (r, r)), mode='edge')
    a = sum(g[i] * p[:, i:i + img.shape[1], :] for i in range(2 * r + 1))
    return sum(g[i] * a[:, :, i:i + img.shape[2]] for i in range(2 * r + 1))


def build(out_dir, n_per=600, chunk=200):
    import json, os
    layers, W = feature_layers()
    dim = K * 12 * 12
    T = np.zeros((24 * n_per, dim), np.float32)
    for k, nm in enumerate(NAMES):
        rng = np.random.default_rng(k)
        P = batch(nm, n_per, rng)                       # rendered hands only: no data is read anywhere
        for i in range(0, n_per, chunk):
            sub = {key: (v[i:i + chunk] if isinstance(v, np.ndarray) else v) for key, v in P.items()}
            img, _ = render(sub)
            f = run(layers, W, blur(img, 0.7)[:, None]).reshape(len(img), -1)
            f -= f.mean(1, keepdims=True); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
            T[k * n_per + i:k * n_per + i + len(f)] = f
    layers.append(dict(name="match", op="dense", inputs=["pool"], out=len(T)))
    W["match.W"], W["match.b"] = T, np.zeros(len(T))
    layers.append(dict(name="bank", op="reshape", inputs=["match"], shape=[1, 24, n_per]))
    layers.append(dict(name="best", op="maxpool", inputs=["bank"], size=[1, n_per], stride=[1, n_per]))
    layers.append(dict(name="scores", op="reshape", inputs=["best"], shape=[24]))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as fh:
        json.dump({"layers": layers}, fh)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
