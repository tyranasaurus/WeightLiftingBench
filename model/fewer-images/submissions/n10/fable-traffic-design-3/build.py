"""Hand-designed traffic sign network: no training data is read; every weight comes from the formulas and drawings below.

Pipeline:  image -> piecewise-linear log (1x1 conv + relu + 1x1 conv) -> log-opponent channels (brightness, red-vs-rest, blue-vs-rest)
           -> fixed 7x7 filters (difference of Gaussians on brightness, Gaussian on colour)
           -> dense bank of drawn sign templates (each drawing normalised by itself alone) -> max over each class's templates.
"""
import sys, json, os
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- feature extractor (from formulas)
EPS = 0.03
KN = np.array([0, .01, .02, .035, .055, .08, .12, .17, .25, .35, .5, .7, 1.0])   # knots of the piecewise-linear log
PIV = 8                                                                            # pivot knot 0.25 (zero padding == mid grey)
fk = np.log(KN + EPS)
sl = (fk[1:] - fk[:-1]) / (KN[1:] - KN[:-1])
nseg = len(sl)
units = []                                       # (sign, bias, coefficient): coefficient * relu(sign * x + bias)
for j in range(PIV, nseg):
    units.append((1.0, -KN[j], sl[j] if j == PIV else sl[j] - sl[j - 1]))
for j in range(PIV, 0, -1):
    units.append((-1.0, KN[j], -sl[PIV - 1] if j == PIV else -(sl[j - 1] - sl[j])))
NU = len(units)                                  # 12 units per colour channel
CW = 0.5                                         # weight of the colour channels relative to brightness
MIX = np.array([[1 / 3, 1 / 3, 1 / 3], [CW, -CW / 2, -CW / 2], [-CW / 2, -CW / 2, CW]])   # brightness, red-vs-rest, blue-vs-rest


def gauss2(sig, k=7):
    a = np.arange(k) - k // 2
    g = np.exp(-a ** 2 / (2 * sig ** 2)); g = np.outer(g, g)
    return g / g.sum()


DOG = gauss2(0.8) - gauss2(2.2)
GS = gauss2(1.0)
KER = [DOG, GS, GS]


def feat(x):
    """x: (n, 3, 32, 32) values 0..1 -> (n, 3, 32, 32) features; the same arithmetic the network performs."""
    g = np.zeros_like(x)
    for s, b, c in units:
        g += c * np.maximum(s * x + b, 0)
    m = np.einsum('oc,nchw->nohw', MIX, g)
    mp = np.pad(m, ((0, 0), (0, 0), (3, 3), (3, 3)))
    out = np.zeros_like(m)
    for o in range(3):
        for dy in range(7):
            for dx in range(7):
                out[:, o] += KER[o][dy, dx] * mp[:, o, dy:dy + 32, dx:dx + 32]
    return out


# ---------------------------------------------------------------- drawing tools (sign coordinates: sign box = [-1,1]^2, y down)
N, L = 200, 1.2
ax = (np.arange(N) + 0.5) / N * 2 * L - L
U, V = np.meshgrid(ax, ax)
BG = 0.3
RED, WHT, BLK, BLU, YEL, GRY = (.62, .13, .15), (.88, .88, .88), (.1, .1, .1), (.1, .24, .6), (.85, .62, .08), (.35, .35, .35)
img = np.zeros((N, N, 3)); alp = np.zeros((N, N))


def P(mask, col, base=False):
    img[mask] = col
    if base: alp[mask] = 1.0


def disc(cx, cy, r): return (U - cx) ** 2 + (V - cy) ** 2 <= r * r
def ell(cx, cy, rx, ry): return ((U - cx) / rx) ** 2 + ((V - cy) / ry) ** 2 <= 1
def rect(x0, x1, y0, y1): return (U >= x0) & (U <= x1) & (V >= y0) & (V <= y1)


def poly(pts):
    m = np.zeros(U.shape, bool)
    for i in range(len(pts)):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % len(pts)]
        if y1 == y2: continue
        m ^= ((y1 > V) != (y2 > V)) & (U < (x2 - x1) * (V - y1) / (y2 - y1) + x1)
    return m


def seg(p, q, w):
    dx, dy = q[0] - p[0], q[1] - p[1]; l2 = dx * dx + dy * dy
    t = np.clip(((U - p[0]) * dx + (V - p[1]) * dy) / l2, 0, 1) if l2 > 0 else 0.0
    return (U - p[0] - t * dx) ** 2 + (V - p[1] - t * dy) ** 2 <= (w / 2) ** 2


def path(pts, w):
    m = np.zeros(U.shape, bool)
    for a, b in zip(pts[:-1], pts[1:]): m |= seg(a, b, w)
    return m


def arc(cx, cy, rx, ry, a0, a1, n=16):
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]


def head(tip, base, hw):
    dx, dy = tip[0] - base[0], tip[1] - base[1]; d = np.hypot(dx, dy); nx, ny = -dy / d, dx / d
    return poly([tip, (base[0] + hw * nx, base[1] + hw * ny), (base[0] - hw * nx, base[1] - hw * ny)])


GL = {   # glyph centre lines in a box x 0..1, y 0..2 (DIN-like digits and letters, typed from memory)
    '0': [arc(.5, 1, .5, 1, 0, 360, 28)],
    '1': [[(.1, .5), (.6, 0), (.6, 2)]],
    '2': [arc(.5, .55, .5, .55, 180, 395, 14) + [(0, 2), (1, 2)]],
    '3': [arc(.45, .52, .5, .52, 200, 450, 14) + arc(.45, 1.5, .55, .5, -90, 160, 14)],
    '5': [[(.95, 0), (.12, 0), (.08, .85)] + arc(.48, 1.38, .52, .62, 215, 510, 16)],
    '6': [[(.85, 0), (.5, .25), (.2, .7), (.03, 1.2)], arc(.5, 1.4, .5, .6, 0, 360, 24)],
    '7': [[(0, 0), (1, 0), (.35, 2)]],
    '8': [arc(.5, .5, .44, .5, 0, 360, 20), arc(.5, 1.5, .5, .5, 0, 360, 20)],
    'S': [arc(.5, .5, .5, .5, -30, -270, 12) + arc(.5, 1.5, .5, .5, -90, 150, 12)],
    'T': [[(0, 0), (1, 0)], [(.5, 0), (.5, 2)]],
    'O': [arc(.5, 1, .5, 1, 0, 360, 28)],
    'P': [[(0, 2), (0, 0), (.5, 0)] + arc(.5, .55, .5, .55, -90, 90, 10) + [(0, 1.1)]],
}


def text(s, widths, h, gap, sw, col):
    x = -(sum(widths) + gap * (len(s) - 1)) / 2; y0 = -h / 2
    for ch, w in zip(s, widths):
        for pl in GL[ch]:
            P(path([(x + sw / 2 + gx * (w - sw), y0 + sw / 2 + gy / 2 * (h - sw)) for gx, gy in pl], sw), col)
        x += w + gap


def number(s, col=BLK):
    if len(s) == 2: text(s, [.5, .5], .84, .08, .13, col)
    else: text(s, [.24, .38, .38], .80, .06, .115, col)


def ring(): P(disc(0, 0, 1), RED, True); P(disc(0, 0, .72), WHT)
def endof(): P(disc(0, 0, 1), GRY, True); P(disc(0, 0, .93), WHT)
def band(w, col=GRY): P(seg((-.68, .68), (.68, -.68), w) & disc(0, 0, .93), col)
def blue(): P(disc(0, 0, 1), BLU, True)
def tri(): P(poly([(0, -1), (1, 1), (-1, 1)]), RED, True); P(poly([(0, -.493), (.62, .747), (-.62, .747)]), WHT)


def car(cx, cy, w, h, col):
    P(rect(cx - w / 2, cx + w / 2, cy, cy + h / 2), col)
    P(poly([(cx - .34 * w, cy), (cx - .22 * w, cy - h / 2), (cx + .22 * w, cy - h / 2), (cx + .34 * w, cy)]), col)
    P(rect(cx - .45 * w, cx - .22 * w, cy + h / 2, cy + .68 * h), col); P(rect(cx + .22 * w, cx + .45 * w, cy + h / 2, cy + .68 * h), col)


def truck(x0, x1, y0, y1, col):          # side view, cab on the left
    w, h = x1 - x0, y1 - y0
    P(rect(x0 + .3 * w, x1, y0, y0 + .75 * h), col); P(rect(x0, x0 + .26 * w, y0 + .3 * h, y0 + .75 * h), col)
    P(rect(x0, x1, y0 + .7 * h, y0 + .8 * h), col)
    P(disc(x0 + .17 * w, y0 + .86 * h, .14 * h), col); P(disc(x0 + .75 * w, y0 + .86 * h, .14 * h), col)


def boxtruck(x0, x1, y0, y1, col):       # lorry seen from behind
    P(rect(x0, x1, y0, y1), col); h = y1 - y0; w = x1 - x0
    P(rect(x0 + .05 * w, x0 + .3 * w, y1, y1 + .2 * h), col); P(rect(x1 - .3 * w, x1 - .05 * w, y1, y1 + .2 * h), col)


def person(cx, top, s):
    P(disc(cx, top + .08 * s, .085 * s), BLK); P(seg((cx, top + .2 * s), (cx, top + .5 * s), .14 * s), BLK)
    P(seg((cx, top + .5 * s), (cx - .17 * s, top + .82 * s), .08 * s), BLK); P(seg((cx, top + .5 * s), (cx + .17 * s, top + .82 * s), .08 * s), BLK)
    P(seg((cx, top + .24 * s), (cx - .18 * s, top + .45 * s), .06 * s), BLK); P(seg((cx, top + .24 * s), (cx + .18 * s, top + .45 * s), .06 * s), BLK)


SPEED = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 7: '100', 8: '120'}
MIRROR = {20: 19, 34: 33, 37: 36, 39: 38}


def draw(c):
    img[:] = BG; alp[:] = 0
    if c in SPEED: ring(); number(SPEED[c])
    elif c == 6: endof(); number('80', GRY); band(.2)
    elif c == 9: ring(); car(-.3, 0, .5, .45, RED); car(.3, 0, .5, .45, BLK)
    elif c == 10: ring(); boxtruck(-.6, -.1, -.26, .2, RED); car(.33, .03, .46, .4, BLK)
    elif c == 11:
        tri(); P(poly([(0, -.22), (.17, .12), (.08, .12), (.1, .68), (-.1, .68), (-.08, .12), (-.17, .12)]), BLK); P(rect(-.33, .33, .27, .4), BLK)
    elif c == 12:
        P(poly([(0, -1), (1, 0), (0, 1), (-1, 0)]), WHT, True); P(poly([(0, -.62), (.62, 0), (0, .62), (-.62, 0)]), YEL)
    elif c == 13:
        P(poly([(-1, -1), (1, -1), (0, 1)]), RED, True); P(poly([(-.6, -.733), (.6, -.733), (0, .467)]), WHT)
    elif c == 14:
        a = .414; o = [(-a, -1), (a, -1), (1, -a), (1, a), (a, 1), (-a, 1), (-1, a), (-1, -a)]
        P(poly(o), WHT, True); P(poly([(.92 * x, .92 * y) for x, y in o]), RED); text('STOP', [.32] * 4, .62, .07, .1, WHT)
    elif c == 15: ring()
    elif c == 16: ring(); truck(-.52, .52, -.3, .34, BLK)
    elif c == 17: P(disc(0, 0, 1), RED, True); P(rect(-.72, .72, -.17, .17), WHT)
    elif c == 18: tri(); P(poly([(-.08, -.2), (.08, -.2), (.05, .4), (-.05, .4)]), BLK); P(disc(0, .58, .075), BLK)
    elif c == 19: tri(); P(path([(.15, .7), (.15, .3), (-.12, .08)], .13), BLK); P(head((-.35, -.08), (-.12, .08), .14), BLK)
    elif c == 21: tri(); P(path([(.14, .7), (.14, .5), (-.12, .22), (-.12, .05)], .12), BLK); P(head((-.12, -.2), (-.12, .05), .13), BLK)
    elif c == 22: tri(); P(rect(-.45, .45, .55, .66), BLK); P(ell(-.2, .56, .15, .14), BLK); P(ell(.2, .56, .15, .14), BLK)
    elif c == 23:
        tri(); P(rect(-.17, .17, .02, .3), BLK); P(rect(-.12, .12, -.1, .02), BLK)
        P(path([(-.1, .35), (-.2, .45), (-.08, .56), (-.2, .68)], .06), BLK); P(path([(.1, .35), (0, .45), (.12, .56), (0, .68)], .06), BLK)
    elif c == 24: tri(); P(path([(-.14, -.08), (-.14, .68)], .1), BLK); P(path([(.1, -.08), (.1, .2), (.26, .42), (.26, .68)], .1), BLK)
    elif c == 25:
        tri(); P(disc(-.02, -.08, .085), BLK); P(seg((-.04, .03), (-.18, .38), .14), BLK)
        P(seg((-.18, .38), (-.36, .68), .09), BLK); P(seg((-.18, .38), (-.05, .68), .09), BLK)
        P(seg((-.02, .12), (.2, .36), .07), BLK); P(seg((.12, .2), (.3, .62), .05), BLK); P(poly([(.1, .68), (.27, .44), (.44, .68)]), BLK)
    elif c == 26:
        tri(); P(rect(-.12, .12, -.12, .7), BLK); P(disc(0, .02, .085), (.8, .15, .1)); P(disc(0, .29, .085), YEL); P(disc(0, .56, .085), (.1, .6, .25))
    elif c == 27: tri(); person(0, -.18, 1.05)
    elif c == 28: tri(); person(-.2, .08, .75); person(.17, -.1, .98)
    elif c == 29:
        tri()
        for cx in (-.25, .25): P(disc(cx, .52, .2) & ~disc(cx, .52, .13), BLK)
        P(path([(-.25, .52), (-.05, .25), (.18, .25), (.25, .52)], .05), BLK); P(path([(-.05, .25), (.02, .52), (-.25, .52)], .05), BLK)
        P(path([(.18, .25), (.14, .12), (.05, .12)], .05), BLK)
    elif c == 30:
        tri()
        for a in (90, 30, 150):
            dx, dy = .3 * np.cos(np.radians(a)), .3 * np.sin(np.radians(a))
            P(seg((-dx, .32 - dy), (dx, .32 + dy), .07), BLK); P(disc(-dx, .32 - dy, .06), BLK); P(disc(dx, .32 + dy, .06), BLK)
        P(disc(0, .32, .1), BLK)
    elif c == 31:
        tri(); P(seg((-.22, .42), (.12, .28), .17), BLK); P(seg((.12, .25), (.24, .05), .09), BLK); P(seg((.24, .05), (.36, .1), .07), BLK)
        P(seg((.22, 0), (.16, -.15), .04), BLK); P(seg((.25, -.02), (.3, -.14), .04), BLK)
        P(seg((-.25, .45), (-.42, .68), .06), BLK); P(seg((-.2, .45), (-.3, .7), .06), BLK)
        P(seg((.12, .33), (.3, .45), .05), BLK); P(seg((.08, .35), (.2, .55), .05), BLK)
    elif c == 32: endof(); band(.3)
    elif c == 33:
        blue(); P(path([(-.3, .65), (-.3, 0)] + arc(0, 0, .3, .3, 180, 270, 8) + [(.25, -.3)], .2), WHT); P(head((.68, -.3), (.22, -.3), .32), WHT)
    elif c == 35: blue(); P(seg((0, .7), (0, -.2), .22), WHT); P(head((0, -.78), (0, -.2), .38), WHT)
    elif c == 36:
        blue(); P(seg((-.25, .7), (-.25, -.3), .18), WHT); P(head((-.25, -.78), (-.25, -.3), .28), WHT)
        P(path([(-.25, .45), (-.05, .15), (.3, .15)], .16), WHT); P(head((.72, .15), (.3, .15), .26), WHT)
    elif c == 38: blue(); P(seg((-.5, -.5), (.2, .2), .24), WHT); P(head((.58, .58), (.15, .15), .36), WHT)
    elif c == 40:
        blue()
        for k in range(3):
            a0 = k * 120 + 20; P(path(arc(0, 0, .5, .5, a0, a0 + 75, 10), .16), WHT)
            a1 = np.radians(a0 + 75); a2 = np.radians(a0 + 108)
            P(poly([(.5 * np.cos(a2), .5 * np.sin(a2)), (.72 * np.cos(a1), .72 * np.sin(a1)), (.28 * np.cos(a1), .28 * np.sin(a1))]), WHT)
    elif c == 41: endof(); car(-.3, 0, .5, .45, GRY); car(.3, 0, .5, .45, GRY); band(.2, BLK)
    elif c == 42: endof(); boxtruck(-.6, -.1, -.26, .2, GRY); car(.33, .03, .46, .4, GRY); band(.2, BLK)
    return np.concatenate([img, alp[:, :, None]], axis=2)


canon = np.zeros((43, N, N, 4))
for c in range(43):
    if c not in MIRROR: canon[c] = draw(c)
for c, src in MIRROR.items(): canon[c] = canon[src][:, ::-1]

# ---------------------------------------------------------------- template bank: each drawing at one pose is one template
SC = [0.60, 0.66, 0.72, 0.78, 0.84]                 # sign box as a share of the frame (border rule: 10 percent, at least 5 px)
poses = [(SC[i], SC[j], th, dx, dy) for i in range(5) for j in range(5) if abs(i - j) <= 1
         for th in (-6.0, 0.0, 6.0) for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)]
K = len(poses)
sub = (np.arange(3) + 0.5) / 3
PX = (np.arange(32)[None, :, None, None] + sub[None, None, None, :]) + np.zeros((32, 1, 3, 1))
PY = (np.arange(32)[:, None, None, None] + sub[None, None, :, None]) + np.zeros((1, 32, 1, 3))
bgv = np.array([BG, BG, BG, 0.0])
W = np.zeros((43, K, 3072), np.float32)
for v, (sx, sy, th, dx, dy) in enumerate(poses):
    xc, yc = PX - 16 - dx, PY - 16 - dy
    ct, st = np.cos(np.radians(th)), np.sin(np.radians(th))
    u = (ct * xc + st * yc) / (16 * sx); w_ = (-st * xc + ct * yc) / (16 * sy)
    iu = np.floor((u + L) / (2 * L) * N).astype(int); iv = np.floor((w_ + L) / (2 * L) * N).astype(int)
    ok = (iu >= 0) & (iu < N) & (iv >= 0) & (iv < N)
    s = canon[:, np.clip(iv, 0, N - 1), np.clip(iu, 0, N - 1), :]
    s[:, ~ok] = bgv
    s = s.mean(axis=(3, 4))                           # (43, 32, 32, 4): the drawing at this pose, area-averaged
    F = feat(s[..., :3].transpose(0, 3, 1, 2))
    M = s[..., 3]
    tl = M * (F[:, 0] - ((M * F[:, 0]).sum(axis=(1, 2)) / M.sum(axis=(1, 2)))[:, None, None])   # brightness: sign area only, own mean removed
    tc = F[:, 1:] - F[:, 1:].mean(axis=(2, 3), keepdims=True)                                  # colour: whole frame, own mean removed
    t = np.concatenate([tl[:, None], tc], axis=1).reshape(43, -1)
    W[:, v] = (t / np.linalg.norm(t, axis=1, keepdims=True)).astype(np.float32)               # each template scaled by its own norm

# ---------------------------------------------------------------- network weights
kW = np.zeros((3 * NU, 3, 1, 1), np.float32); kb = np.zeros(3 * NU, np.float32)
mW = np.zeros((3, 3 * NU, 1, 1), np.float32)
for ch in range(3):
    for j, (s, b, c) in enumerate(units):
        kW[ch * NU + j, ch, 0, 0] = s; kb[ch * NU + j] = b
        mW[:, ch * NU + j, 0, 0] = MIX[:, ch] * c
fW = np.zeros((3, 3, 7, 7), np.float32)
for o in range(3): fW[o, o] = KER[o]
T = 43 * K
weights = {"k.W": kW, "k.b": kb, "mix.W": mW, "mix.b": np.zeros(3, np.float32), "filt.W": fW, "filt.b": np.zeros(3, np.float32),
           "t.W": W.reshape(T, 3072), "t.b": np.zeros(T, np.float32)}
net = {"layers": [
    {"name": "k", "op": "conv", "inputs": ["image"], "out": 3 * NU, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "kr", "op": "relu", "inputs": ["k"]},
    {"name": "mix", "op": "conv", "inputs": ["kr"], "out": 3, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "filt", "op": "conv", "inputs": ["mix"], "out": 3, "kernel": [7, 7], "stride": 1, "padding": 3},
    {"name": "t", "op": "dense", "inputs": ["filt"], "out": T},
    {"name": "r1", "op": "reshape", "inputs": ["t"], "shape": [43, 1, K]},
    {"name": "mp", "op": "maxpool", "inputs": ["r1"], "size": [1, K], "stride": [1, K]},
    {"name": "out", "op": "reshape", "inputs": ["mp"], "shape": [43]},
]}
with open(os.path.join(OUT, "network.json"), "w") as f: json.dump(net, f)
np.savez(os.path.join(OUT, "weights.npz"), **weights)
