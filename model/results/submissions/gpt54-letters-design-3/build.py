import json, math, sys
from pathlib import Path
import numpy as np

NAMES = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
HR = 112
S = HR / 28.0
EPS = 1e-8
SHIFT = 3
GRID = 2 * SHIFT + 1
FAMILIES = 3
THRESH = 3.0
YY, XX = np.meshgrid(np.arange(HR, dtype=np.float32) + 0.5, np.arange(HR, dtype=np.float32) + 0.5, indexing='ij')
Y28, X28 = np.meshgrid(np.arange(28, dtype=np.float32), np.arange(28, dtype=np.float32), indexing='ij')


def soft_union(*parts):
    out = np.zeros_like(parts[0])
    for p in parts:
        out = np.maximum(out, p)
    return np.clip(out, 0.0, 1.0)


def sub(mask, hole):
    return np.clip(mask * (1.0 - hole), 0.0, 1.0)


def circle(cx, cy, r, blur=1.4):
    cx *= S; cy *= S; r *= S; blur *= S
    d = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    return np.clip((r + blur - d) / blur, 0.0, 1.0)


def ellipse(cx, cy, rx, ry, angle=0.0, blur=1.4):
    a = math.radians(angle)
    x = (XX / S) - cx
    y = (YY / S) - cy
    ca, sa = math.cos(a), math.sin(a)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    q = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    t = blur / max(rx, ry)
    return np.clip((1.0 + t - q) / t, 0.0, 1.0)


def capsule(x0, y0, x1, y1, r, blur=1.4):
    x0 *= S; y0 *= S; x1 *= S; y1 *= S; r *= S; blur *= S
    vx, vy = x1 - x0, y1 - y0
    t = ((XX - x0) * vx + (YY - y0) * vy) / (vx * vx + vy * vy + 1e-6)
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    d = np.sqrt((XX - px) ** 2 + (YY - py) ** 2)
    return np.clip((r + blur - d) / blur, 0.0, 1.0)


def downsample28(mask):
    return mask.reshape(28, HR // 28, 28, HR // 28).mean((1, 3))


def shift2d(a, dx, dy):
    out = np.zeros_like(a)
    xs0 = max(0, -dx); xs1 = min(28, 28 - dx)
    ys0 = max(0, -dy); ys1 = min(28, 28 - dy)
    xd0 = max(0, dx); xd1 = xd0 + (xs1 - xs0)
    yd0 = max(0, dy); yd1 = yd0 + (ys1 - ys0)
    if xs1 > xs0 and ys1 > ys0:
        out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
    return out


def palm_basic():
    return soft_union(
        ellipse(14.0, 17.7, 5.25, 6.65, angle=-6),
        capsule(14.0, 22.5, 14.0, 28.4, 3.0),
    )


def open_fingers(count=4, spread=1.0, close=False, lengths=None, base_y=11.6, width=1.08):
    xs = [-3.8, -1.3, 1.2, 3.5][:count]
    if count == 3:
        xs = [-1.8, 0.5, 2.8]
    if count == 2:
        xs = [0.3, 2.8]
    if count == 1:
        xs = [2.8]
    if lengths is None:
        lengths = [9.8, 10.8, 10.4, 9.6][:count]
        if count == 3:
            lengths = [10.4, 10.9, 10.1]
        if count == 2:
            lengths = [10.6, 10.3]
        if count == 1:
            lengths = [10.8]
    parts = []
    for i, x in enumerate(xs):
        x += (i - (len(xs) - 1) / 2) * 0.7 * (spread - 1.0)
        ang = (-5 + 3 * i) if not close else (-2 + 1.2 * i)
        if count == 2 and spread > 1.2:
            ang = -10 + 20 * i
        if count == 3 and spread > 1.2:
            ang = -12 + 12 * i
        y0 = base_y + 1.5 + 0.3 * abs(i - (len(xs) - 1) / 2)
        x1 = 14 + x + math.sin(math.radians(ang)) * lengths[i] * 0.18
        y1 = y0 - lengths[i] * math.cos(math.radians(ang))
        parts.append(capsule(14 + x, y0, x1, y1, width))
    return soft_union(*parts) if parts else np.zeros((HR, HR), np.float32)


def thumb_side():
    return soft_union(capsule(18.6, 18.4, 20.7, 13.2, 1.35), circle(20.8, 13.0, 1.5))


def thumb_out():
    return capsule(18.5, 18.4, 24.4, 15.8, 1.15)


def curled_fingers(kind='fist'):
    parts = []
    bumps = [(-3.0, 12.8, 2.4, 1.8), (-0.5, 12.1, 2.6, 1.9), (2.0, 12.0, 2.5, 1.8), (4.1, 12.4, 2.0, 1.6)]
    for x, y, rx, ry in bumps:
        parts.append(ellipse(14 + x, y, rx, ry, angle=-6))
    if kind == 'e':
        parts.append(capsule(12.2, 14.1, 17.9, 12.7, 1.0))
    return soft_union(*parts)


def mask_A():
    return soft_union(palm_basic(), curled_fingers(), thumb_side())


def mask_B():
    return soft_union(
        palm_basic(),
        open_fingers(4, spread=1.0, close=True, width=1.02),
        capsule(18.4, 18.0, 11.0, 16.3, 1.0),
    )


def mask_C():
    outer = ellipse(13.8, 14.5, 7.0, 9.2, angle=-6)
    inner = ellipse(15.6, 14.7, 4.7, 6.1, angle=-6)
    wrist = capsule(11.3, 22.0, 10.0, 28.2, 2.8)
    opening = ellipse(19.8, 14.4, 3.4, 7.0)
    return soft_union(sub(soft_union(outer, wrist), inner), ellipse(10.8, 16.3, 3.0, 5.0, angle=-10)) * (1.0 - opening * 0.7)


def mask_D():
    palm = palm_basic()
    idx = open_fingers(count=1, width=1.0)
    loop = sub(ellipse(15.0, 16.7, 3.8, 4.1, angle=10), ellipse(15.4, 16.9, 1.7, 1.8, angle=10))
    others = soft_union(ellipse(11.2, 12.8, 2.2, 1.6), ellipse(13.4, 12.3, 2.4, 1.6), ellipse(15.4, 12.6, 2.2, 1.5))
    return soft_union(palm, idx, loop, others)


def mask_E():
    return soft_union(palm_basic(), curled_fingers('e'), capsule(18.5, 17.6, 11.4, 16.2, 1.2))


def mask_F():
    palm = palm_basic()
    fingers = open_fingers(3, spread=1.2, close=False, width=0.98)
    loop = sub(ellipse(16.2, 18.0, 3.7, 4.0, angle=12), ellipse(16.2, 18.0, 1.6, 1.7, angle=12))
    return soft_union(palm, fingers, loop)


def mask_G():
    palm = ellipse(11.0, 16.0, 4.5, 5.5, angle=80)
    wrist = capsule(6.4, 16.8, 0.2, 17.2, 3.0)
    idx = capsule(12.0, 14.2, 22.6, 14.0, 1.05)
    thumb = capsule(11.8, 18.3, 20.6, 17.0, 1.0)
    return soft_union(palm, wrist, idx, thumb)


def mask_H():
    palm = ellipse(10.8, 16.1, 4.4, 5.5, angle=84)
    wrist = capsule(6.0, 17.0, 0.2, 17.2, 3.0)
    idx = capsule(11.8, 13.1, 22.5, 12.9, 1.0)
    mid = capsule(11.8, 16.0, 22.0, 15.8, 1.0)
    thumb = capsule(12.0, 18.8, 16.4, 19.3, 0.9)
    return soft_union(palm, wrist, idx, mid, thumb)


def mask_I():
    return soft_union(palm_basic(), curled_fingers(), capsule(10.2, 13.5, 9.8, 3.2, 1.0), thumb_side())


def mask_K():
    palm = palm_basic()
    fingers = open_fingers(2, spread=1.55, close=False, width=1.0)
    thumb = capsule(18.4, 18.0, 14.9, 13.2, 1.0)
    return soft_union(palm, fingers, thumb)


def mask_L():
    return soft_union(palm_basic(), open_fingers(1, width=1.0), thumb_out())


def mask_M():
    palm = palm_basic()
    fingers = soft_union(
        ellipse(11.2, 12.8, 1.6, 2.0), ellipse(13.5, 12.2, 1.6, 2.0),
        ellipse(15.8, 12.2, 1.6, 2.0), ellipse(18.0, 12.8, 1.4, 1.8),
    )
    thumb = capsule(18.1, 17.5, 10.4, 14.2, 1.15)
    return soft_union(palm, fingers, thumb)


def mask_N():
    palm = palm_basic()
    fingers = soft_union(ellipse(11.4, 12.8, 1.5, 1.9), ellipse(14.2, 12.1, 1.6, 2.0), ellipse(16.8, 12.6, 1.5, 1.9))
    thumb = capsule(18.2, 17.3, 11.5, 14.1, 1.15)
    return soft_union(palm, fingers, thumb)


def mask_O():
    outer = soft_union(ellipse(14.1, 16.0, 5.1, 7.0), capsule(14.0, 22.8, 14.0, 28.2, 2.9))
    inner = ellipse(14.4, 16.3, 2.8, 4.0)
    return sub(outer, inner)


def mask_P():
    palm = ellipse(14.0, 13.8, 5.0, 4.6, angle=-10)
    wrist = capsule(14.8, 18.0, 14.9, 27.8, 2.6)
    idx = capsule(13.0, 14.3, 20.6, 21.2, 1.0)
    mid = capsule(15.6, 12.3, 18.9, 18.7, 0.95)
    thumb = capsule(11.2, 13.6, 15.8, 14.8, 1.0)
    return soft_union(palm, wrist, idx, mid, thumb)


def mask_Q():
    palm = ellipse(14.0, 14.3, 5.0, 4.8, angle=-12)
    wrist = capsule(14.6, 18.0, 15.0, 27.8, 2.6)
    idx = capsule(13.0, 13.8, 18.2, 21.8, 1.0)
    thumb = capsule(12.2, 15.7, 17.3, 16.6, 1.0)
    return soft_union(palm, wrist, idx, thumb)


def mask_R():
    palm = palm_basic()
    idx = capsule(14.0, 13.0, 13.0, 2.6, 1.0)
    mid = capsule(16.4, 13.1, 18.1, 2.8, 1.0)
    cross = capsule(14.2, 10.5, 17.0, 7.8, 0.7)
    thumb = capsule(18.2, 18.3, 15.4, 14.8, 0.95)
    return soft_union(palm, idx, mid, cross, thumb)


def mask_S():
    return soft_union(palm_basic(), curled_fingers(), capsule(18.6, 18.8, 13.0, 16.3, 1.15))


def mask_T():
    palm = palm_basic()
    fingers = soft_union(
        ellipse(11.4, 12.8, 1.6, 1.9), ellipse(13.9, 12.2, 1.8, 2.0),
        ellipse(16.4, 12.5, 1.6, 1.9), ellipse(18.2, 13.0, 1.3, 1.6),
    )
    thumb = capsule(18.6, 17.2, 12.0, 14.9, 1.1)
    return soft_union(palm, fingers, thumb, circle(11.7, 14.8, 1.2))


def mask_U():
    palm = palm_basic()
    fingers = open_fingers(2, spread=0.95, close=True, width=1.0)
    thumb = capsule(18.0, 18.2, 15.0, 15.1, 0.95)
    return soft_union(palm, fingers, thumb)


def mask_V():
    palm = palm_basic()
    fingers = open_fingers(2, spread=1.8, close=False, width=1.0)
    thumb = capsule(18.0, 18.2, 15.0, 15.1, 0.95)
    return soft_union(palm, fingers, thumb)


def mask_W():
    palm = palm_basic()
    fingers = open_fingers(3, spread=1.8, close=False, width=0.95)
    thumb = capsule(18.1, 18.0, 15.2, 15.2, 0.95)
    return soft_union(palm, fingers, thumb)


def mask_X():
    palm = palm_basic()
    hook = soft_union(capsule(14.5, 13.4, 14.3, 6.7, 0.95), capsule(14.3, 6.9, 16.8, 5.0, 0.8), circle(16.8, 5.0, 0.9))
    thumb = capsule(18.2, 18.2, 15.1, 15.0, 0.95)
    others = soft_union(ellipse(11.3, 12.9, 2.1, 1.7), ellipse(16.8, 12.7, 2.1, 1.7), ellipse(19.0, 13.0, 1.7, 1.5))
    return soft_union(palm, hook, thumb, others)


def mask_Y():
    palm = palm_basic()
    pinky = capsule(10.0, 13.6, 8.8, 3.1, 1.0)
    thumb = capsule(18.3, 18.0, 24.2, 14.8, 1.05)
    return soft_union(palm, curled_fingers(), pinky, thumb)


MASKS = [mask_A, mask_B, mask_C, mask_D, mask_E, mask_F, mask_G, mask_H, mask_I, mask_K, mask_L, mask_M,
         mask_N, mask_O, mask_P, mask_Q, mask_R, mask_S, mask_T, mask_U, mask_V, mask_W, mask_X, mask_Y]


def family_maps():
    q0 = np.ones((28, 28), np.float32)
    q0[23] *= 0.85
    q0[24] *= 0.70
    q0[25] *= 0.55
    q0[26] *= 0.40
    q0[27] *= 0.28
    q1 = q0 * (1.25 - 0.040 * Y28)
    q2 = q0 * (0.30 + np.exp(-((X28 - 13.5) ** 2) / 70.0 - ((Y28 - 14.0) ** 2) / 58.0))
    return [q0.astype(np.float32), q1.astype(np.float32), q2.astype(np.float32)]


def template_from_mask(mask, q):
    alpha = float((q * mask).sum() / (q.sum() + EPS))
    t = q * (mask - alpha)
    n = math.sqrt(float((t * t).sum()) + EPS)
    return (t / n).astype(np.float32)


def build_weights():
    masks28 = [downsample28(fn()).astype(np.float32) for fn in MASKS]
    qs = family_maps()
    templ = []
    for c in range(24):
        m = masks28[c]
        for q in qs:
            for dy in range(-SHIFT, SHIFT + 1):
                for dx in range(-SHIFT, SHIFT + 1):
                    ms = shift2d(m, dx, dy)
                    templ.append(template_from_mask(ms, q).reshape(-1))
    templ = np.stack(templ, axis=0).astype(np.float32)
    p = templ.shape[0]
    weights = {
        'pos.W': templ,
        'pos.b': np.full((p,), -THRESH, np.float32),
        'neg.W': -templ,
        'neg.b': np.full((p,), -THRESH, np.float32),
    }
    out_W = np.zeros((24, 24 * FAMILIES), np.float32)
    fam_w = [1.0, 0.85, 0.75]
    for c in range(24):
        for f in range(FAMILIES):
            out_W[c, c * FAMILIES + f] = fam_w[f]
    weights['out.W'] = out_W
    weights['out.b'] = np.zeros((24,), np.float32)
    return weights


def build_network():
    return {
        'layers': [
            {'name': 'n0', 'op': 'norm', 'inputs': ['image']},
            {'name': 'pos', 'op': 'dense', 'inputs': ['n0'], 'out': 24 * FAMILIES * GRID * GRID},
            {'name': 'neg', 'op': 'dense', 'inputs': ['n0'], 'out': 24 * FAMILIES * GRID * GRID},
            {'name': 'rpos', 'op': 'relu', 'inputs': ['pos']},
            {'name': 'rneg', 'op': 'relu', 'inputs': ['neg']},
            {'name': 'abs', 'op': 'add', 'inputs': ['rpos', 'rneg']},
            {'name': 'grid', 'op': 'reshape', 'inputs': ['abs'], 'shape': [24 * FAMILIES, GRID, GRID]},
            {'name': 'best', 'op': 'maxpool', 'inputs': ['grid'], 'size': [GRID, GRID], 'stride': [GRID, GRID]},
            {'name': 'feat', 'op': 'reshape', 'inputs': ['best'], 'shape': [24 * FAMILIES]},
            {'name': 'out', 'op': 'dense', 'inputs': ['feat'], 'out': 24},
        ]
    }


def main(dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    net = build_network()
    weights = build_weights()
    (dst / 'network.json').write_text(json.dumps(net, separators=(',', ':')))
    np.savez(dst / 'weights.npz', **weights)


if __name__ == '__main__':
    main(sys.argv[1])
