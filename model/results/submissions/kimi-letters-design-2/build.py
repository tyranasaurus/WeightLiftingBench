"""Build a data-free ASL shape-matching network.

All prototypes are rendered from a small anatomical hand model.  The model is
expressed in palm/finger coordinates and uses only general descriptions of the
static ASL alphabet.  No files are read by this program.
"""
import sys, json, math
from pathlib import Path
import numpy as np

NCLASS = 24
NVAR = 48
SS = 2

# A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def hand_spec(c, rng):
    """Return finger paths, a thumb path, and a global pose.

    Finger order is little, ring, middle, index.  Coordinates are in units of
    palm widths, with the wrist towards positive y.  A path is a polyline whose
    segments are rendered as rounded, tapered capsules.
    """
    anchors = [(-0.47, -0.46), (-0.16, -0.53), (0.16, -0.53), (0.45, -0.46)]
    lengths = [0.83, 1.04, 1.13, 1.00]
    # Directions are angles from vertically upwards; positive leans right.
    spread = [-0.16, -0.045, 0.045, 0.16]
    states = ["curl"] * 4
    thumb = "front"
    angle = rng.uniform(-8, 8)
    fist = c in (0, 3, 4, 8, 11, 12, 13, 16, 17, 18, 22)
    if c == 0:       # A: closed fist, thumb along its side
        thumb = "side"
    elif c == 1:     # B: four fingers together, thumb folded
        states = ["extend"] * 4; spread = [-0.045, -0.015, 0.015, 0.045]; thumb = "fold"
    elif c == 2:     # C: curved fingers and thumb make an open C
        states = ["C"] * 4; thumb = "C"; angle = rng.uniform(-12, 12)
    elif c == 3:     # D
        states[3] = "extend"; thumb = "front"
    elif c == 4:     # E: tightly bent digits and low folded thumb
        thumb = "low"
    elif c == 5:     # F: index-thumb circle, remaining digits extended
        states = ["extend", "extend", "extend", "ring"]
        spread = [-0.25, -0.08, 0.08, 0.0]; thumb = "ring"
    elif c == 6:     # G: index and thumb sideways
        states[3] = "extend"; thumb = "lateral"; angle = 90 + rng.uniform(-12, 12)
    elif c == 7:     # H: two fingers sideways
        states[2:] = ["extend", "extend"]; spread[2:] = [0.0, 0.025]
        thumb = "fold"; angle = 90 + rng.uniform(-12, 12)
    elif c == 8:     # I
        states[0] = "extend"; thumb = "front"
    elif c == 9:     # K: index/middle in a narrow V, thumb raised between
        states[2:] = ["extend", "extend"]; spread[2:] = [-0.13, 0.20]; thumb = "between"
    elif c == 10:    # L
        states[3] = "extend"; thumb = "lateral"
    elif c == 11:    # M: thumb beneath three curled fingers
        thumb = "hidden3"
    elif c == 12:    # N: thumb beneath two curled fingers
        thumb = "hidden2"
    elif c == 13:    # O
        states = ["O"] * 4; thumb = "O"
    elif c == 14:    # P: K hand pointing downwards
        states[2:] = ["extend", "extend"]; spread[2:] = [-0.10, 0.22]; thumb = "between"
        angle = -150 + rng.uniform(-12, 12)
    elif c == 15:    # Q: G hand pointing downwards
        states[3] = "extend"; thumb = "lateral"; angle = 180 + rng.uniform(-12, 12)
    elif c == 16:    # R: crossed index and middle fingers
        states[2:] = ["cross_middle", "cross_index"]; thumb = "front"
    elif c == 17:    # S: thumb across the front of the fist
        thumb = "front"
    elif c == 18:    # T: thumb tip between index and middle
        thumb = "tip"
    elif c == 19:    # U
        states[2:] = ["extend", "extend"]; spread[2:] = [-0.025, 0.025]; thumb = "fold"
    elif c == 20:    # V
        states[2:] = ["extend", "extend"]; spread[2:] = [-0.24, 0.25]; thumb = "front"
    elif c == 21:    # W
        states[1:] = ["extend", "extend", "extend"]; spread[1:] = [-0.25, 0.0, 0.25]; thumb = "front"
    elif c == 22:    # X: hooked index
        states[3] = "hook"; thumb = "side"
    elif c == 23:    # Y
        states[0] = "extend"; spread[0] = -0.34; thumb = "wide"

    paths = []
    for j, state in enumerate(states):
        x, y = anchors[j]
        le = lengths[j] * rng.uniform(0.91, 1.09)
        a = spread[j] + rng.uniform(-0.035, 0.035)
        if state == "extend":
            # Three phalanges; slight joint curvature avoids ruler-straight digits.
            p0 = np.array([x, y])
            p1 = p0 + le * 0.46 * np.array([math.sin(a), -math.cos(a)])
            a2 = a + rng.uniform(-0.055, 0.055)
            p2 = p1 + le * 0.33 * np.array([math.sin(a2), -math.cos(a2)])
            a3 = a2 + rng.uniform(-0.06, 0.06)
            p3 = p2 + le * 0.25 * np.array([math.sin(a3), -math.cos(a3)])
            paths.append([p0, p1, p2, p3])
        elif state == "curl":
            # The knuckle rises above the palm and the fingers fold down its face.
            drift = rng.uniform(-0.045, 0.045)
            paths.append([np.array([x, y]), np.array([x + drift, -0.64]),
                          np.array([x + drift * 1.5, -0.12]), np.array([x * 0.87, 0.28])])
        elif state == "ring":
            paths.append([np.array([x, y]), np.array([0.49, -0.08]),
                          np.array([0.29, 0.15]), np.array([0.02, -0.05])])
        elif state == "hook":
            paths.append([np.array([x, y]), np.array([0.46, -0.94]),
                          np.array([0.61, -1.16]), np.array([0.75, -0.91])])
        elif state == "cross_middle":
            paths.append([np.array([x, y]), np.array([0.24, -1.03]),
                          np.array([0.12, -1.48]), np.array([0.06, -1.69])])
        elif state == "cross_index":
            paths.append([np.array([x, y]), np.array([0.29, -1.02]),
                          np.array([0.30, -1.46]), np.array([0.38, -1.65])])
        else:
            paths.append([])
    return paths, thumb, angle, fist


def render(c, variant, ss=SS):
    """Render one general anatomical example, with a fixed-seed pose variation."""
    rng = np.random.default_rng(9137 + 1009 * c + 7919 * variant)
    paths, thumb, angle, fist = hand_spec(c, rng)
    mirror = (variant % 2) == 1
    if mirror:
        paths = [[np.array([-p[0], p[1]]) for p in path] for path in paths]
    scale = rng.uniform(6.0, 7.5) if not fist else rng.uniform(7.0, 8.5)
    if c in (2, 13): scale = rng.uniform(6.8, 8.1)
    theta = math.radians(angle)
    ct, st = math.cos(theta), math.sin(theta)
    center = np.array([14.0 + rng.uniform(-1.2, 1.2), 13.0 + rng.uniform(-1.0, 1.0)])
    n = 28 * ss
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    xx = (xx + 0.5) / ss; yy = (yy + 0.5) / ss
    dx, dy = xx - center[0], yy - center[1]
    # Inverse rotation and scaling into palm coordinates.
    X = (ct * dx + st * dy) / scale
    Y = (-st * dx + ct * dy) / scale
    if mirror:
        # Thumb geometry below is specified in the unmirrored hand frame.
        X_thumb = -X
    else:
        X_thumb = X

    light_hand = ((variant // 2) % 3) != 0
    hand = rng.uniform(0.61, 0.80) if light_hand else rng.uniform(0.25, 0.48)
    bg = rng.uniform(0.24, 0.60) if light_hand else rng.uniform(0.52, 0.78)
    # Smooth illumination, not a stored background or image patch.
    im = bg + rng.uniform(-0.13, 0.13) * (xx / 28 - 0.5) + rng.uniform(-0.10, 0.10) * (yy / 28 - 0.5)

    def paint(dist, color, edge=0.026):
        nonlocal im
        a = sigmoid(-dist / edge)
        im = im * (1 - a) + color * a

    def capsule(path, radius, color, Xuse=None, Yuse=None):
        Xuse = X if Xuse is None else Xuse
        Yuse = Y if Yuse is None else Yuse
        dist = np.full(X.shape, 1e3)
        for k in range(len(path) - 1):
            p, q = path[k], path[k + 1]
            vx, vy = q - p
            den = vx * vx + vy * vy + 1e-12
            t = np.clip(((Xuse - p[0]) * vx + (Yuse - p[1]) * vy) / den, 0, 1)
            d = np.sqrt((Xuse - (p[0] + t * vx)) ** 2 + (Yuse - (p[1] + t * vy)) ** 2)
            # A small taper from proximal to distal phalanx.
            rr = radius * (1.0 - 0.18 * (k + t) / max(1, len(path) - 1))
            dist = np.minimum(dist, d - rr)
        shade = color + 0.035 * np.clip(-Yuse, -1, 1)
        paint(dist, shade)

    def ellipse(cx, cy, rx, ry, color, inner=None, cut_c=False):
        z = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)
        dist = (z - 1) * min(rx, ry)
        if inner is not None:
            zi = np.sqrt(((X - cx) / inner[0]) ** 2 + ((Y - cy) / inner[1]) ** 2)
            dist = np.maximum(dist, (1 - zi) * min(inner))
        if cut_c:
            # Open the ring on the thumb/index side to form a C.
            dist = np.maximum(dist, np.minimum(0.16 - np.abs(Y - cy), X - (cx + 0.20)))
        paint(dist, color + 0.03 * np.clip(-Y, -1, 1))

    wrist_col = hand * rng.uniform(0.76, 0.98)
    capsule([np.array([0.0, 0.38]), np.array([0.0, 1.65])], 0.43, wrist_col)
    # Extended and hooked digits sit behind the palm.  Curled digits are
    # reserved for the front, where their rounded phalanges remain visible.
    curled = set((0, 4, 11, 12, 17, 18))
    front_curl = {
        3: {0, 1, 2}, 8: {1, 2, 3}, 9: {0, 1}, 16: {0, 1},
        19: {0, 1}, 20: {0, 1}, 21: {0}, 22: {0, 1, 2}, 23: {1, 2, 3}
    }
    front = {0, 1, 2, 3} if c in curled else front_curl.get(c, set())
    for j, path in enumerate(paths):
        if len(path) and j not in front and c != 5:
            capsule(path, 0.145 if j != 0 else 0.135, hand * rng.uniform(0.91, 1.04))
    if c == 5:
        for j in range(3):
            capsule(paths[j], 0.145 if j else 0.135, hand * rng.uniform(0.91, 1.04))
    ellipse(0, 0.03, 0.67, 0.76, hand)
    for j in sorted(front):
        capsule(paths[j], 0.145, hand * rng.uniform(0.77, 0.91))
    if c == 5:
        capsule(paths[3], 0.145, hand * 0.86)
    if c == 13:
        ellipse(0.0, -0.61, 0.63, 0.71, hand * 0.96, inner=(0.25, 0.32))
        # Finger divisions around the upper arc.
        for x in (-0.34, -0.10, 0.14, 0.36):
            capsule([np.array([x, -0.49]), np.array([x * 0.85, -1.10 + 0.35 * abs(x)])], 0.055, hand * 0.76)
    if c == 2:
        ellipse(-0.06, -0.40, 0.69, 0.89, hand * 0.96, inner=(0.31, 0.48), cut_c=True)

    def thumb_path(kind):
        if kind == "side": return [(-0.55, 0.43), (-0.73, -0.22), (-0.61, -0.73)]
        if kind in ("fold", "front"): return [(-0.57, 0.13), (-0.18, 0.22), (0.42, 0.02)]
        if kind == "low": return [(-0.50, 0.32), (-0.02, 0.37), (0.42, 0.25)]
        if kind == "lateral": return [(-0.53, 0.30), (-0.96, 0.03), (-1.25, -0.10)]
        if kind == "wide": return [(-0.53, 0.31), (-1.00, 0.30), (-1.32, 0.12)]
        if kind == "between": return [(-0.52, 0.27), (-0.22, -0.25), (0.02, -0.75)]
        if kind == "tip": return [(-0.53, 0.23), (-0.13, 0.04), (0.06, -0.60)]
        if kind == "hidden3": return [(-0.52, 0.35), (-0.05, 0.26), (0.38, 0.20)]
        if kind == "hidden2": return [(-0.52, 0.28), (-0.16, 0.05), (-0.02, -0.42)]
        if kind == "ring": return [(-0.55, 0.36), (-0.43, -0.03), (0.02, -0.05)]
        return []
    tp = thumb_path(thumb)
    if tp:
        tpath = [np.array(p) for p in tp]
        capsule(tpath, 0.18, hand * (0.91 if thumb != "front" else 0.83), X_thumb, Y)
    # Downsample the analytically antialiased rendering.
    im = im.reshape(28, ss, 28, ss).mean(axis=(1, 3))
    return np.clip(im, 0, 1).astype(np.float32)


def norm(x):
    return (x - x.mean(axis=(-2, -1), keepdims=True)) / np.sqrt(x.var(axis=(-2, -1), keepdims=True) + 1e-5)


def conv(x, w, b, pad):
    if pad: x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    o, ci, k, _ = w.shape
    cols = np.stack([x[:, :, dy:dy + 28, dx:dx + 28] for dy in range(k) for dx in range(k)], axis=2)
    z = cols.reshape(x.shape[0], ci*k*k, 28*28).transpose(0, 2, 1) @ w.reshape(o, -1).T
    return z.transpose(0, 2, 1).reshape(x.shape[0], o, 28, 28) + b[None, :, None, None]


def pool(x, s):
    return x.reshape(x.shape[0], x.shape[1], 28 // s, s, 28 // s, s).mean(axis=(3, 5))


def fixed_features(images, weights):
    n0 = norm(images)
    hp = conv(n0, weights['hp.W'], weights['hp.b'], 4)
    nh = norm(hp)
    gr = conv(nh, weights['gr.W'], weights['gr.b'], 2)
    ori = np.einsum('nchw,oc->nohw', gr, weights['ori.W'][:, :, 0, 0]) + weights['ori.b'][None, :, None, None]
    hn = norm(np.maximum(ori, 0))
    pos = norm(np.maximum(n0, 0))
    return [pool(hn, 2), pool(hn, 4), pool(n0, 4), pool(nh, 4), pool(pos, 4)]


def make_fixed_weights():
    w = {}
    # A Gaussian high-pass suppresses broad illumination and background gradients.
    y, x = np.mgrid[-4:5, -4:5]
    g = np.exp(-(x*x + y*y) / (2 * 1.65**2)); g /= g.sum()
    hp = -g; hp[4, 4] += 1
    w['hp.W'] = hp[None, None].astype(np.float32); w['hp.b'] = np.zeros(1, np.float32)
    # Five-point Sobel derivatives.
    y, x = np.mgrid[-2:3, -2:3]
    smooth = np.exp(-(x*x + y*y) / 3.2)
    gx = -x * smooth; gx -= gx.mean(); gx /= np.abs(gx).sum()
    gy = -y * smooth; gy -= gy.mean(); gy /= np.abs(gy).sum()
    w['gr.W'] = np.stack([gx, gy])[:, None].astype(np.float32); w['gr.b'] = np.zeros(2, np.float32)
    dirs = []
    for a in np.arange(8) * np.pi / 4:
        dirs.append([math.cos(a), math.sin(a)])
    w['ori.W'] = np.array(dirs, np.float32)[:, :, None, None]; w['ori.b'] = np.zeros(8, np.float32)
    return w


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    w = make_fixed_weights()
    layers = [
        {'name':'n0','op':'norm','inputs':['image']},
        {'name':'hp','op':'conv','inputs':['n0'],'out':1,'kernel':[9,9],'stride':1,'padding':4},
        {'name':'nh','op':'norm','inputs':['hp']},
        {'name':'gr','op':'conv','inputs':['nh'],'out':2,'kernel':[5,5],'stride':1,'padding':2},
        {'name':'ori','op':'conv','inputs':['gr'],'out':8,'kernel':[1,1],'stride':1,'padding':0},
        {'name':'or','op':'relu','inputs':['ori']},
        {'name':'hn','op':'norm','inputs':['or']},
        {'name':'hog14','op':'avgpool','inputs':['hn'],'size':[2,2],'stride':[2,2]},
        {'name':'hog7','op':'avgpool','inputs':['hn'],'size':[4,4],'stride':[4,4]},
        {'name':'raw7','op':'avgpool','inputs':['n0'],'size':[4,4],'stride':[4,4]},
        {'name':'hp7','op':'avgpool','inputs':['nh'],'size':[4,4],'stride':[4,4]},
        {'name':'pos','op':'relu','inputs':['n0']},
        {'name':'posn','op':'norm','inputs':['pos']},
        {'name':'pos7','op':'avgpool','inputs':['posn'],'size':[4,4],'stride':[4,4]},
    ]
    # Generate a fixed, balanced bank of anatomical pose models.
    templates = np.stack([render(c, v) for c in range(NCLASS) for v in range(NVAR)])[:, None]
    feats = fixed_features(templates, w)
    names = ['hog14', 'hog7', 'raw7', 'hp7', 'pos7']
    gains = [1.0, 0.30, 0.65, 0.30, 0.35]
    score_names = []
    for name, f, gain in zip(names, feats, gains):
        flat = f.reshape(len(f), -1).astype(np.float64)
        # Unit prototype vectors implement cosine-like shape matching.  The gain
        # only sets the relative importance of edge, silhouette and intensity cues.
        flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
        dn = 'score_' + name
        w[dn + '.W'] = (flat * gain).astype(np.float32)
        w[dn + '.b'] = np.zeros(NCLASS * NVAR, np.float32)
        layers.append({'name':dn,'op':'dense','inputs':[name],'out':NCLASS*NVAR})
        score_names.append(dn)
    layers += [
        {'name':'all_scores','op':'add','inputs':score_names},
        {'name':'by_class','op':'reshape','inputs':['all_scores'],'shape':[NCLASS,NVAR,1]},
        {'name':'best_pose','op':'maxpool','inputs':['by_class'],'size':[NVAR,1],'stride':[NVAR,1]},
        {'name':'classes','op':'reshape','inputs':['best_pose'],'shape':[NCLASS]},
    ]
    (out/'network.json').write_text(json.dumps({'layers':layers}, separators=(',',':')))
    np.savez(out/'weights.npz', **w)

if __name__ == '__main__':
    main()
