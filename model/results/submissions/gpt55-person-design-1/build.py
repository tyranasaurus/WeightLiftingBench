import json, sys
from pathlib import Path
import numpy as np

# Hand written, data-free network builder for the COCO person/no-person task.
# All numbers below are analytic colour/edge constants and geometric templates.

def layer(name, op, inputs, **kw):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(kw)
    return d

Y = np.array([0.299, 0.587, 0.114], np.float32)

def add_colour(W, o, coeff, bias, k):
    c = k // 2
    W[o, :, c, c] = np.array(coeff, np.float32)
    return bias

def add_sobel(W, o, mat, sign, k):
    c = k // 2
    m = np.array(mat, np.float32) * sign / 4.0
    for ch in range(3):
        W[o, ch, c-1:c+2, c-1:c+2] += Y[ch] * m

def identity(W, n):
    k = W.shape[2]; c = k // 2
    for i in range(min(n, W.shape[0], W.shape[1])):
        W[i, i, c, c] = 1.0

def grid(k):
    yy, xx = np.mgrid[0:k, 0:k].astype(np.float32)
    if k == 1:
        return xx, yy
    x = (xx - (k - 1) / 2) / ((k - 1) / 2)
    y = (yy - (k - 1) / 2) / ((k - 1) / 2)
    return x, y

def gauss(k, cx, cy, sx, sy):
    x, y = grid(k)
    m = np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2)).astype(np.float32)
    return m / (m.sum() + 1e-6)

def band(k, cx, cy, sx, sy):
    # Soft rectangular/line band from a Gaussian, normalized.
    return gauss(k, cx, cy, sx, sy)

def add_avg(W, o, chans, mask, coef):
    if not chans:
        return
    m = mask.astype(np.float32)
    s = float(m.sum()) + 1e-6
    for ch in chans:
        if ch < W.shape[1]:
            W[o, ch, :, :] += (coef / len(chans)) * m / s

def add_point(W, o, ch, mask, coef):
    if ch < W.shape[1]:
        m = mask.astype(np.float32); W[o, ch, :, :] += coef * m / (float(m.sum()) + 1e-6)

# Low-level channel convention. These channels are preserved by identity filters
# through all later stages.
SKIN = [0, 1, 2]
SKIN_REL = [3, 4]
DARK = [5, 6]
BRIGHT = [7, 8]
NOTBRIGHT = [9]
WARM = [10]
BLUE = [11]
GREEN = [12]
SAT = [13, 14, 15]
EDGE = list(range(16, 24))
VEDGE = [16, 17]
HEDGE = [18, 19]
DIAG = [20, 21, 22, 23]

# Pooled/local part channels made in c2 and then preserved.
EDGE_BLOB = 24
SKIN_BLOB = 25
DARK_BLOB = 26
WARM_BLOB = 27
VEDGE_BLOB = 28
HEDGE_BLOB = 29
PERSON_MIX = 30
SAT_BLOB = 31
TINY_FACE = 32
TINY_DARK_HEAD = 33
TINY_UPPER = 34
TINY_STAND = 35
TINY_SHOULDER = 36
TINY_LEGS = 37
TINY_CROWD = 38
TINY_PROFILE = 39
TINY_SNOW_DARK = 40
TINY_GREEN_PLAYER = 41
TINY_CLOSE = 42
TINY_WARM_OBJECT = 43
TINY_STRAIGHT_OBJECT = 44
TINY_TEXTURE = 45
TINY_BLUEGREEN = 46
TINY_BROWN_ANIMAL = 47


def make_c1():
    W = np.zeros((24, 3, 3, 3), np.float32)
    b = np.zeros(24, np.float32)
    # Warm flesh-like and brown/pink/yellow colour half-spaces.
    specs = [
        ([1.20, 0.80, -1.40], -0.18),   # red/yellow warmer than blue
        ([1.10, 0.45, -0.55], -0.42),   # pink/tan lightness
        ([0.75, 0.75, -1.10], -0.25),   # yellow/tan
        ([1.00, -1.00, 0.00], -0.02),   # R > G
        ([1.00, 0.00, -1.00], -0.04),   # R > B
        ([-Y[0], -Y[1], -Y[2]], 0.48),  # dark hair/clothing/shadows
        ([-Y[0], -Y[1], -Y[2]], 0.28),  # very dark
        ([Y[0], Y[1], Y[2]], -0.55),    # bright
        ([Y[0], Y[1], Y[2]], -0.25),    # not very dark
        ([-Y[0], -Y[1], -Y[2]], 0.78),  # non-bright
        ([1.20, 0.35, -0.80], -0.35),  # orange/red objects and skin
        ([-0.45, -0.35, 1.00], -0.08), # blue sky/water/clothes
        ([-0.45, 1.00, -0.45], -0.08), # green grass/foliage/clothes
        ([1.00, -0.50, -0.50], -0.08), # red saturation
        ([-0.50, -0.40, 1.00], -0.08), # blue saturation
        ([-0.40, 1.00, -0.50], -0.08), # green saturation
    ]
    for o, (co, bi) in enumerate(specs):
        b[o] = add_colour(W, o, co, bi, 3)
    sx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sy = [[-1,-2,-1],[0,0,0],[1,2,1]]
    sd1 = [[0,1,2],[-1,0,1],[-2,-1,0]]
    sd2 = [[2,1,0],[1,0,-1],[0,-1,-2]]
    for j, mat in enumerate([sx, sy, sd1, sd2]):
        add_sobel(W, 16 + 2*j, mat, 1.0, 3)
        add_sobel(W, 17 + 2*j, mat, -1.0, 3)
    return W, b


def make_c2():
    W = np.zeros((48, 24, 3, 3), np.float32)
    b = np.zeros(48, np.float32)
    identity(W, 24)
    all3 = np.ones((3,3), np.float32)
    top = np.array([[1,1,1],[0,0,0],[0,0,0]], np.float32)
    mid = np.array([[0,0,0],[1,1,1],[0,0,0]], np.float32)
    bot = np.array([[0,0,0],[0,0,0],[1,1,1]], np.float32)
    cen = np.array([[0,0,0],[0,1,0],[0,0,0]], np.float32)
    col = np.array([[0,1,0],[0,1,0],[0,1,0]], np.float32)
    sides = np.array([[1,0,1],[1,0,1],[1,0,1]], np.float32)
    # Local pooled primitives.
    add_avg(W, EDGE_BLOB, EDGE, all3, 1.15); b[EDGE_BLOB] = -0.075
    add_avg(W, SKIN_BLOB, SKIN + SKIN_REL, all3, 1.05); b[SKIN_BLOB] = -0.090
    add_avg(W, DARK_BLOB, DARK, all3, 1.00); b[DARK_BLOB] = -0.085
    add_avg(W, WARM_BLOB, WARM + SKIN, all3, 0.90); b[WARM_BLOB] = -0.080
    add_avg(W, VEDGE_BLOB, VEDGE + DIAG, all3, 1.10); b[VEDGE_BLOB] = -0.075
    add_avg(W, HEDGE_BLOB, HEDGE + DIAG, all3, 1.05); b[HEDGE_BLOB] = -0.075
    add_avg(W, PERSON_MIX, SKIN + DARK + EDGE, all3, 1.25); b[PERSON_MIX] = -0.130
    add_avg(W, SAT_BLOB, SAT + WARM, all3, 1.05); b[SAT_BLOB] = -0.100
    # Tiny part detectors (rough 6-12 pixel structures in the source image).
    add_avg(W, TINY_FACE, SKIN + SKIN_REL, all3, 0.95); add_avg(W, TINY_FACE, DARK, top+mid, 0.35); add_avg(W, TINY_FACE, HEDGE, mid, 0.28); add_avg(W, TINY_FACE, BLUE+GREEN, all3, -0.22); b[TINY_FACE] = -0.135
    add_avg(W, TINY_DARK_HEAD, DARK + EDGE, all3, 0.80); add_avg(W, TINY_DARK_HEAD, HEDGE, mid, 0.25); b[TINY_DARK_HEAD] = -0.145
    add_avg(W, TINY_UPPER, SKIN + DARK, top, 0.75); add_avg(W, TINY_UPPER, VEDGE + EDGE, mid+bot, 0.70); add_avg(W, TINY_UPPER, SAT, mid+bot, 0.25); b[TINY_UPPER] = -0.165
    add_avg(W, TINY_STAND, DARK + VEDGE + EDGE, col, 0.95); add_avg(W, TINY_STAND, SKIN + DARK, top, 0.35); add_avg(W, TINY_STAND, BLUE+GREEN, all3, -0.08); b[TINY_STAND] = -0.150
    add_avg(W, TINY_SHOULDER, SKIN + DARK, top, 0.55); add_avg(W, TINY_SHOULDER, HEDGE + EDGE, mid, 0.65); add_avg(W, TINY_SHOULDER, VEDGE, sides, 0.25); b[TINY_SHOULDER] = -0.145
    add_avg(W, TINY_LEGS, VEDGE + DARK + EDGE, sides+bot, 0.85); b[TINY_LEGS] = -0.145
    add_avg(W, TINY_CROWD, SKIN + DARK + EDGE, all3, 1.05); b[TINY_CROWD] = -0.185
    add_avg(W, TINY_PROFILE, SKIN + DARK, np.array([[0,1,1],[0,1,1],[0,0,1]], np.float32), 0.80); add_avg(W, TINY_PROFILE, VEDGE+HEDGE, all3, 0.35); b[TINY_PROFILE] = -0.150
    add_avg(W, TINY_SNOW_DARK, DARK+VEDGE+EDGE, cen+col, 0.90); add_avg(W, TINY_SNOW_DARK, BRIGHT, all3, 0.18); b[TINY_SNOW_DARK] = -0.150
    add_avg(W, TINY_GREEN_PLAYER, DARK+SKIN+EDGE, cen+col, 0.85); add_avg(W, TINY_GREEN_PLAYER, GREEN, all3, 0.12); b[TINY_GREEN_PLAYER] = -0.145
    add_avg(W, TINY_CLOSE, SKIN+DARK+HEDGE, all3, 0.95); b[TINY_CLOSE] = -0.135
    # A few counter-evidence channels for common no-person scenes.
    add_avg(W, TINY_WARM_OBJECT, WARM+SAT, all3, 1.05); add_avg(W, TINY_WARM_OBJECT, EDGE, all3, 0.20); b[TINY_WARM_OBJECT] = -0.160
    add_avg(W, TINY_STRAIGHT_OBJECT, VEDGE+HEDGE, all3, 0.95); add_avg(W, TINY_STRAIGHT_OBJECT, SKIN, all3, -0.18); b[TINY_STRAIGHT_OBJECT] = -0.145
    add_avg(W, TINY_TEXTURE, EDGE+SAT, all3, 1.00); b[TINY_TEXTURE] = -0.180
    add_avg(W, TINY_BLUEGREEN, BLUE+GREEN, all3, 1.00); b[TINY_BLUEGREEN] = -0.120
    add_avg(W, TINY_BROWN_ANIMAL, WARM+DARK+EDGE, all3, 0.95); add_avg(W, TINY_BROWN_ANIMAL, SKIN_REL, all3, -0.12); b[TINY_BROWN_ANIMAL] = -0.160
    return W, b


def add_face(W, o, k, dark=False, profile=False, close=False):
    x, y = grid(k)
    head = np.exp(-0.5*((x/0.55)**2 + ((y+0.05)/0.72)**2)).astype(np.float32)
    hair = np.exp(-0.5*((x/0.65)**2 + ((y+0.58)/0.25)**2)).astype(np.float32)
    eyes = np.exp(-0.5*((x/0.50)**2 + ((y+0.08)/0.13)**2)).astype(np.float32)
    mouth = np.exp(-0.5*((x/0.35)**2 + ((y-0.40)/0.12)**2)).astype(np.float32)
    if profile:
        head *= np.clip((x+1.0)/1.2, 0, 1)
    if dark:
        add_avg(W, o, DARK + [DARK_BLOB, EDGE_BLOB], head, 0.85)
        add_avg(W, o, BRIGHT + HEDGE, eyes+mouth, 0.30)
        add_avg(W, o, [TINY_DARK_HEAD, TINY_FACE], head, 0.55)
        W[o, :, :, :] += 0
        return -0.205 if not close else -0.175
    add_avg(W, o, SKIN + SKIN_REL + [SKIN_BLOB], head, 0.95)
    add_avg(W, o, DARK + [DARK_BLOB], hair + eyes, 0.42)
    add_avg(W, o, HEDGE + VEDGE + [EDGE_BLOB], eyes + mouth + head*0.25, 0.35)
    add_avg(W, o, BLUE + GREEN + [TINY_BLUEGREEN], head, -0.22)
    add_avg(W, o, [TINY_FACE, TINY_PROFILE, TINY_CLOSE], head, 0.65)
    return -0.210 if not close else -0.180


def add_head_shoulders(W, o, k):
    x, y = grid(k)
    head = np.exp(-0.5*((x/0.33)**2 + ((y+0.55)/0.25)**2)).astype(np.float32)
    shoulders = np.exp(-0.5*((x/0.70)**2 + ((y+0.05)/0.23)**2)).astype(np.float32)
    torso = np.exp(-0.5*((x/0.42)**2 + ((y-0.35)/0.40)**2)).astype(np.float32)
    add_avg(W, o, SKIN + DARK + [SKIN_BLOB, DARK_BLOB, TINY_FACE, TINY_DARK_HEAD], head, 0.75)
    add_avg(W, o, HEDGE + VEDGE + EDGE + [EDGE_BLOB, TINY_SHOULDER], shoulders, 0.62)
    add_avg(W, o, DARK + SAT + [PERSON_MIX, TINY_UPPER], torso, 0.35)
    add_avg(W, o, BLUE + GREEN, head, -0.08)
    return -0.235


def add_standing(W, o, k, on_green=False, on_bright=False, dark_only=False):
    x, y = grid(k)
    head = np.exp(-0.5*((x/0.24)**2 + ((y+0.72)/0.18)**2)).astype(np.float32)
    torso = np.exp(-0.5*((x/0.30)**2 + ((y+0.10)/0.43)**2)).astype(np.float32)
    side_l = np.exp(-0.5*(((x+0.32)/0.13)**2 + ((y+0.05)/0.55)**2)).astype(np.float32)
    side_r = np.exp(-0.5*(((x-0.32)/0.13)**2 + ((y+0.05)/0.55)**2)).astype(np.float32)
    legs = np.exp(-0.5*(((np.abs(x)-0.17)/0.12)**2 + ((y-0.55)/0.34)**2)).astype(np.float32)
    arms = np.exp(-0.5*(((np.abs(x)-0.48)/0.12)**2 + ((y+0.03)/0.37)**2)).astype(np.float32)
    add_avg(W, o, VEDGE + DIAG + [VEDGE_BLOB, EDGE_BLOB, TINY_STAND], side_l+side_r+legs+arms, 0.82)
    add_avg(W, o, HEDGE + [HEDGE_BLOB], head+legs*0.4, 0.22)
    if dark_only:
        add_avg(W, o, DARK + [DARK_BLOB, TINY_SNOW_DARK], head+torso+legs, 0.55)
    else:
        add_avg(W, o, SKIN + DARK + SAT + [PERSON_MIX, TINY_UPPER], head+torso+legs*0.45, 0.48)
    if on_green:
        add_avg(W, o, GREEN + [TINY_GREEN_PLAYER], np.ones((k,k), np.float32), 0.12)
    if on_bright:
        add_avg(W, o, BRIGHT + [TINY_SNOW_DARK], np.ones((k,k), np.float32), 0.10)
    return -0.255


def add_sitting(W, o, k):
    x, y = grid(k)
    head = gauss(k, 0.0, -0.55, 0.25, 0.22)
    body = gauss(k, -0.05, -0.05, 0.45, 0.35)
    legs = gauss(k, 0.25, 0.45, 0.55, 0.20)
    add_avg(W, o, SKIN + DARK + [SKIN_BLOB, DARK_BLOB, TINY_FACE], head, 0.62)
    add_avg(W, o, EDGE + VEDGE + HEDGE + [EDGE_BLOB, TINY_UPPER, TINY_LEGS], body+legs, 0.72)
    add_avg(W, o, SAT + DARK + [PERSON_MIX], body, 0.32)
    return -0.240


def add_group(W, o, k):
    x, y = grid(k)
    m1 = gauss(k, -0.38, -0.35, 0.22, 0.20)
    m2 = gauss(k,  0.00, -0.42, 0.22, 0.20)
    m3 = gauss(k,  0.38, -0.30, 0.22, 0.20)
    bodies = gauss(k, 0.0, 0.18, 0.78, 0.55)
    heads = m1 + m2 + m3
    add_avg(W, o, [TINY_FACE, TINY_DARK_HEAD, TINY_CROWD, SKIN_BLOB, DARK_BLOB], heads, 0.78)
    add_avg(W, o, EDGE + VEDGE + [EDGE_BLOB, PERSON_MIX, TINY_UPPER], bodies, 0.54)
    return -0.250


def add_closeup(W, o, k):
    x, y = grid(k)
    oval = np.exp(-0.5*((x/0.82)**2 + ((y+0.02)/0.86)**2)).astype(np.float32)
    eyes = np.exp(-0.5*((x/0.55)**2 + ((y+0.12)/0.12)**2)).astype(np.float32)
    hair = np.exp(-0.5*((x/0.85)**2 + ((y+0.70)/0.22)**2)).astype(np.float32)
    add_avg(W, o, SKIN + SKIN_REL + [SKIN_BLOB, TINY_FACE, TINY_CLOSE], oval, 0.82)
    add_avg(W, o, DARK + HEDGE + [DARK_BLOB, EDGE_BLOB], eyes+hair, 0.48)
    add_avg(W, o, BLUE+GREEN+[TINY_BLUEGREEN], oval, -0.16)
    return -0.220


def add_animal_or_food(W, o, k, kind):
    # Deliberately weak counter-evidence; person photos may also contain warm food or animals.
    allm = np.ones((k,k), np.float32)
    if kind == 'food':
        add_avg(W, o, [TINY_WARM_OBJECT, SAT_BLOB] + WARM + SAT, allm, 0.85)
        add_avg(W, o, [TINY_FACE, TINY_STAND, TINY_SHOULDER], allm, -0.35)
        return -0.210
    if kind == 'animal':
        add_avg(W, o, [TINY_BROWN_ANIMAL, TINY_TEXTURE] + WARM + DARK, allm, 0.78)
        add_avg(W, o, [TINY_FACE, TINY_UPPER, TINY_STAND], allm, -0.28)
        return -0.215
    add_avg(W, o, [TINY_STRAIGHT_OBJECT, TINY_BLUEGREEN] + BLUE + GREEN, allm, 0.75)
    add_avg(W, o, [TINY_FACE, TINY_STAND, TINY_UPPER], allm, -0.25)
    return -0.220


def template_bank(W, start, k, stage):
    # Adds 32 channels: a fixed bank of geometric person and weak non-person templates.
    bs = np.zeros(32, np.float32)
    funcs = []
    funcs += [(add_face, dict(dark=False, profile=False, close=False))]
    funcs += [(add_face, dict(dark=True, profile=False, close=False))]
    funcs += [(add_face, dict(dark=False, profile=True, close=False))]
    funcs += [(add_head_shoulders, {})]
    funcs += [(add_standing, dict(on_green=False, on_bright=False, dark_only=False))]
    funcs += [(add_standing, dict(on_green=True, on_bright=False, dark_only=False))]
    funcs += [(add_standing, dict(on_green=False, on_bright=True, dark_only=True))]
    funcs += [(add_sitting, {})]
    funcs += [(add_group, {})]
    funcs += [(add_closeup, {})]
    # repeat some shapes with slightly different emphasis by adding more instances later
    idx = 0
    for fn, kw in funcs:
        o = start + idx
        if fn is add_face:
            bs[idx] = fn(W, o, k, **kw)
        elif fn is add_standing:
            bs[idx] = fn(W, o, k, **kw)
        else:
            bs[idx] = fn(W, o, k)
        idx += 1
    # Extra variants, built by changing signs/locations/thresholds a bit.
    for j in range(idx, 24):
        o = start + j
        typ = j % 7
        if typ == 0:
            bs[j] = add_head_shoulders(W, o, k) + 0.015
        elif typ == 1:
            bs[j] = add_standing(W, o, k, on_green=(j % 2 == 1), on_bright=False, dark_only=(j % 3 == 0)) + 0.010
        elif typ == 2:
            bs[j] = add_sitting(W, o, k) + 0.010
        elif typ == 3:
            bs[j] = add_group(W, o, k) + 0.010
        elif typ == 4:
            bs[j] = add_face(W, o, k, dark=(j % 2 == 0), profile=(j % 3 == 0), close=False) + 0.012
        elif typ == 5:
            bs[j] = add_closeup(W, o, k) + 0.012
        else:
            bs[j] = add_standing(W, o, k, on_green=False, on_bright=True, dark_only=True) + 0.012
    for j, kind in enumerate(['food','animal','scene','food','animal','scene','food','animal']):
        bs[24+j] = add_animal_or_food(W, start+24+j, k, kind)
    # Later, coarser stages see larger objects; use slightly lower thresholds for
    # full bodies and groups because the max-pooled part evidence is sparser.
    if stage >= 2:
        bs[:24] += 0.015
    if stage >= 3:
        bs[:24] += 0.010
    return bs


def make_template_layer(in_ch, out_ch, k, start, stage):
    W = np.zeros((out_ch, in_ch, k, k), np.float32)
    b = np.zeros(out_ch, np.float32)
    identity(W, min(in_ch, out_ch))
    bs = template_bank(W, start, k, stage)
    b[start:start+32] = bs
    return W, b


def build(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = {"layers": [
        layer("c1", "conv", ["image"], out=24, kernel=[3,3], stride=1, padding=1),
        layer("r1", "relu", ["c1"]),
        layer("p1", "maxpool", ["r1"], size=[2,2], stride=[2,2]),
        layer("c2", "conv", ["p1"], out=48, kernel=[3,3], stride=1, padding=1),
        layer("r2", "relu", ["c2"]),
        layer("p2", "maxpool", ["r2"], size=[2,2], stride=[2,2]),
        layer("c3", "conv", ["p2"], out=80, kernel=[5,5], stride=1, padding=2),
        layer("r3", "relu", ["c3"]),
        layer("p3", "maxpool", ["r3"], size=[2,2], stride=[2,2]),
        layer("c4", "conv", ["p3"], out=112, kernel=[5,5], stride=1, padding=2),
        layer("r4", "relu", ["c4"]),
        layer("p4", "maxpool", ["r4"], size=[2,2], stride=[2,2]),
        layer("c5", "conv", ["p4"], out=144, kernel=[5,5], stride=1, padding=2),
        layer("r5", "relu", ["c5"]),
        layer("g", "maxpool", ["r5"], size=[6,6], stride=[6,6]),
        layer("out", "dense", ["g"], out=2),
    ]}
    W1, b1 = make_c1()
    W2, b2 = make_c2()
    W3, b3 = make_template_layer(48, 80, 5, 48, 1)
    W4, b4 = make_template_layer(80, 112, 5, 80, 2)
    W5, b5 = make_template_layer(112, 144, 5, 112, 3)
    # Dense readout: compare one scalar personness score to a no-person bias.
    D = np.zeros((2, 144), np.float32)
    db = np.array([12.0, 0.0], np.float32)
    # Low-level evidence is weak by itself.
    for ch in [SKIN_BLOB, DARK_BLOB, PERSON_MIX, TINY_FACE, TINY_DARK_HEAD, TINY_UPPER, TINY_STAND, TINY_SHOULDER, TINY_LEGS, TINY_CROWD, TINY_PROFILE, TINY_SNOW_DARK, TINY_GREEN_PLAYER, TINY_CLOSE]:
        D[1, ch] += 0.42
    # Tiny no-person counters.
    for ch in [TINY_WARM_OBJECT, TINY_STRAIGHT_OBJECT, TINY_TEXTURE, TINY_BLUEGREEN, TINY_BROWN_ANIMAL]:
        D[0, ch] += 0.16
    # Template banks: first 24 are person templates, last 8 are weak no-person counters.
    for base, scale in [(48, 0.92), (80, 1.10), (112, 1.22)]:
        for j in range(24):
            # Faces/head-shoulders/groups are strongest; generic shape variants a little lower.
            mult = 1.0
            if j in [0,1,2,3,8,9,18,19,22,23]: mult = 1.15
            if j in [4,5,6,11,13,16,20]: mult = 0.95
            D[1, base+j] += scale * mult
        for j in range(24,32):
            D[0, base+j] += 0.24 * scale
    # Penalize broad warm/animal/scene channels lightly; many person photos contain these too.
    D[0, TINY_WARM_OBJECT] += 0.12
    D[0, TINY_BROWN_ANIMAL] += 0.18
    D[0, TINY_BLUEGREEN] += 0.10
    weights = {
        "c1.W": W1, "c1.b": b1,
        "c2.W": W2, "c2.b": b2,
        "c3.W": W3, "c3.b": b3,
        "c4.W": W4, "c4.b": b4,
        "c5.W": W5, "c5.b": b5,
        "out.W": D, "out.b": db,
    }
    (out / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(out / "weights.npz", **{k: v.astype(np.float32) for k, v in weights.items()})

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
