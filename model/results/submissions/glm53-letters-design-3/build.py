"""build.py -- writes network.json and weights.npz for a hand-designed ASL letter
classifier. Every weight derives from formulas and a parametric hand-shape model
typed from general knowledge of hands and the ASL alphabet (palm ellipse, finger
capsules at given angles and spacing, fist blobs, thumb positions, anatomical
crease lines). Thresholds, scales and variant poses were chosen by reasoning
about hand geometry, tested only on a self-made synthetic generator.
No training data is read. Usage: python3 build.py OUTPUT_DIR"""
import sys
import numpy as np

"""Port of /app/blocks.py semantics for local synthetic testing. Matches scorer exactly."""
NORM_EPS = 1e-5

def _conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    ho, wo = (h + 2*padding - kh)//stride + 1, (w + 2*padding - kw)//stride + 1
    cols = np.stack([xp[:, :, dy:dy+stride*ho:stride, dx:dx+stride*wo:stride] for dy in range(kh) for dx in range(kw)], axis=2)
    out = cols.reshape(n, c*kh*kw, ho*wo).transpose(0,2,1) @ W.reshape(o, c*kh*kw).reshape(o,-1).T
    return out.transpose(0,2,1).reshape(n,o,ho,wo) + b[None,:,None,None]

def _pool(x, size, stride, reduce):
    n,c,h,w = x.shape
    (sh,sw),(th,tw) = size, stride
    ho, wo = (h-sh)//th+1, (w-sw)//tw+1
    win = np.stack([x[:,:,dy:dy+th*ho:th, dx:dx+tw*wo:tw] for dy in range(sh) for dx in range(sw)], axis=0)
    return win.max(0) if reduce=="max" else win.mean(0)

def run(net, weights, images):
    vals = {"image": images.astype(np.float32)}
    for L in net["layers"]:
        ins = [vals[i] for i in L["inputs"]]
        op, name = L["op"], L["name"]
        if op == "conv":
            k = L["kernel"]
            vals[name] = conv(ins[0], weights[name+".W"], weights[name+".b"], L.get("stride",1), L.get("padding",0))
        elif op == "dense":
            vals[name] = ins[0].reshape(len(ins[0]),-1) @ weights[name+".W"].T + weights[name+".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op in ("maxpool","avgpool"):
            vals[name] = pool(ins[0], L["size"], L["stride"], op[:3])
        elif op == "add":
            vals[name] = sum(ins)
        elif op == "reshape":
            vals[name] = ins[0].reshape((len(ins[0]),)+tuple(L["shape"]))
        elif op == "norm":
            x = ins[0]
            mu = x.mean(axis=(2,3), keepdims=True)
            vals[name] = (x-mu)/np.sqrt(((x-mu)**2).mean(axis=(2,3), keepdims=True)+NORM_EPS)
        else: raise ValueError(op)
    return vals[net["layers"][-1]["name"]]

"""Parametric hand-shape renderer for the 24 static ASL letters.
All shapes are typed from general knowledge of hand proportions and ASL poses:
a palm ellipse, finger capsules at given angles/lengths, a fist blob, an extended thumb.
Right hand, palm toward viewer, upright, drawn in a 28x28 canvas; the caller
mirrors / rotates / scales / shifts for variants. Pure numpy, deterministic."""

SS = 4  # supersampling
N = 28

def _grid():
    g = np.mgrid[0:N*SS, 0:N*SS].astype(np.float64) / SS  # (2, H, W): y, x in 28-units
    return g[0], g[1]
GY, GX = _grid()

def _smooth(d, e=0.55):
    # d = signed distance-ish (negative inside); smoothstep over e
    t = np.clip(-d / e + 0.5, 0, 1)
    return t

def _capsule_mask(x0, y0, x1, y1, th):
    # distance to segment, in 28-units, on the supersampled grid
    p0 = np.array([y0, x0]); p1 = np.array([y1, x1])
    v = p1 - p0; L2 = float(v @ v)
    P = np.stack([GY, GX]) - p0[:, None, None]     # (2,H,W)
    if L2 > 0:
        t = np.clip((P[0] * v[0] + P[1] * v[1]) / L2, 0, 1)
    else:
        t = np.zeros_like(GY)
    proj0 = p0[0] + t * v[0]; proj1 = p0[1] + t * v[1]
    d = np.hypot(GY - proj0, GX - proj1) - th / 2.0
    return _smooth(d)

def _ellipse_mask(cx, cy, rx, ry, rot=0.0):
    ct, st = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    dy, dx = GY - cy, GX - cx
    u = ct * dx + st * dy; v = -st * dx + ct * dy
    d = np.sqrt((u / max(rx, 1e-6)) ** 2 + (v / max(ry, 1e-6)) ** 2) - 1.0
    # scaled distance
    d = d * min(rx, ry)
    return _smooth(d)

def _ring_mask(cx, cy, rx, ry, rix, riy, a0=None, a1=None, rot=0.0):
    m = _ellipse_mask(cx, cy, rx, ry, rot) * (1 - _ellipse_mask(cx, cy, rix, riy, rot))
    if a0 is not None:
        ang = np.degrees(np.arctan2(GY - cy, GX - cx)) % 360
        cut = np.zeros_like(m)
        a0, a1 = a0 % 360, a1 % 360
        if a0 < a1: cut = ((ang >= a0) & (ang <= a1)).astype(float)
        else: cut = ((ang >= a0) | (ang <= a1)).astype(float)
        # feather the cut edges by ~3.5 deg
        m = m * (1 - cut)
    return m

def _down(m):
    return m.reshape(N, SS, N, SS).mean(axis=(1, 3))

def _blur(m, sig=0.8):
    r = int(np.ceil(sig * 2.5)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
    m = np.apply_along_axis(lambda x: np.convolve(x, k, 'same'), 0, m)
    m = np.apply_along_axis(lambda x: np.convolve(x, k, 'same'), 1, m)
    return m

class Canvas:
    def __init__(self):
        self.m = np.zeros((N, N))
    def add(self, mask, val=1.0):
        self.m = np.maximum(self.m, val * _down(mask))
    def add_soft(self, mask, val=1.0):
        self.m = self.m + val * _down(mask); self.m = np.clip(self.m, 0, 1)
    def sub(self, mask, val=1.0):
        self.m = np.clip(self.m - val * _down(mask), 0, 1)

# ---- transforms: applied to shape coordinates before drawing ----
class Aff:
    """mirror x about 14, rotate about (cx,cy), scale, translate."""
    def __init__(self, mirror=False, rot=0.0, scale=1.0, dx=0.0, dy=0.0, cx=14.0, cy=15.5):
        self.M = np.eye(2)
        if mirror:
            M1 = np.array([[-1, 0], [0, 1]], float)
            self.M = M1
        th = np.deg2rad(rot)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        S = np.eye(2) * scale
        A = R @ S
        # combine: mirror first (about x=14), then rotate/scale about c, then translate
        self.A = A
        self.c = np.array([cy, cx]); self.t = np.array([dy, dx])
    def __call__(self, y, x):
        p = np.array([y, x], float)
        q = self.A @ (self.M @ (p - self.c)) + self.c + self.t
        return q[0], q[1]

def T_capsule(aff, x0, y0, x1, y1, th):
    y0t, x0t = aff(y0, x0); y1t, x1t = aff(y1, x1)
    return _capsule_mask(x0t, y0t, x1t, y1t, th * np.sqrt(abs(np.linalg.det(aff.A @ aff.M))) if False else th)

def T_ellipse(aff, cx, cy, rx, ry, rot=0.0):
    cyt, cxt = aff(cy, cx)
    A = aff.A @ aff.M
    # transform the ellipse axes
    ax = np.array([rx * np.cos(np.deg2rad(rot)), rx * np.sin(np.deg2rad(rot))])  # (y,x) of axis vec? keep simple: axis-aligned input
    u = A @ np.array([0.0, rx]); v = A @ np.array([ry, 0.0])
    # approximate: new rx,ry = lengths of transformed axes; rotation from u
    nrx = np.hypot(u[0], u[1]); nry = np.hypot(v[0], v[1])
    nrot = np.degrees(np.arctan2(u[0], u[1]))
    return _ellipse_mask(cxt, cyt, nrx, nry, nrot)

def T_ring(aff, cx, cy, rx, ry, rix, riy, a0=None, a1=None):
    cyt, cxt = aff(cy, cx)
    A = aff.A @ aff.M
    u = A @ np.array([0.0, rx]); v = A @ np.array([ry, 0.0])
    nrx = np.hypot(u[0], u[1]); nry = np.hypot(v[0], v[1])
    # angle of transformed +x axis in image coords (atan2(dy,dx) where dy=GY-cy)
    thx = np.degrees(np.arctan2((A @ np.array([0, 1.0]))[0], (A @ np.array([0, 1.0]))[1]))
    na0 = None if a0 is None else (a0 - thx)
    na1 = None if a1 is None else (a1 - thx)
    return _ring_mask(cxt, cyt, nrx, nry, rix * nrx / rx, riy * nry / ry, na0, na1)

def _fix_thickness():
    pass

def T_capsule2(aff, x0, y0, x1, y1, th):
    y0t, x0t = aff(y0, x0); y1t, x1t = aff(y1, x1)
    return _capsule_mask(x0t, y0t, x1t, y1t, th * aff.s)

def wrist(cv, aff, side='bottom', w=5.6):
    if side == 'bottom':
        cv.add(T_ellipse(aff, 14.2, 27.2, w, 4.0))
    elif side == 'left':
        cv.add(T_ellipse(aff, -1.5, 16.0, 4.0, w))
    elif side == 'right':
        cv.add(T_ellipse(aff, 29.5, 16.0, 4.0, w))
    elif side == 'top':
        cv.add(T_ellipse(aff, 14.2, 0.8, w, 4.0))

def fingers_up(cv, aff, xs, tip_ys, base_y, th, val=1.0):
    for x, ty in zip(xs, tip_ys):
        cv.add(T_capsule2(aff, x, base_y, x + (np.random if False else 0), ty, th), val)

# ---------------- class poses ----------------
# canonical: right hand, palm toward viewer, thumb on viewer's LEFT, upright.
def draw(cls, cv, aff, p=0.0, crc=None):
    """p in [-1,1]: pose jitter selector (spread/size perturbation)."""
    sp = 1.0 + 0.10 * p   # spread factor
    sz = 1.0 + 0.06 * p
    if cls == 0:  # A: fist, thumb up along the left side
        cv.add(T_ellipse(aff, 14.4, 17.0, 7.0*sz, 7.2*sz))
        for x in (10.2, 12.7, 15.2, 17.6):
            cv.add(T_ellipse(aff, x, 11.8, 1.7, 2.1))
        cv.add(T_capsule2(aff, 7.6, 20.0, 8.4, 12.6, 2.6))
        wrist(cv, aff)
    elif cls == 1:  # B: four fingers up with visible gaps, thumb folded across palm
        xs = [9.4 - 0.5*p, 13.1, 16.8, 20.5 + 0.5*p]
        tys = [5.6, 4.4, 4.8, 6.4]
        fingers_up(cv, aff, xs, tys, 16.0, 2.75)
        cv.add(T_ellipse(aff, 15.0, 20.6, 7.4*sz, 6.0))
        cv.add(T_capsule2(aff, 9.0, 23.0, 15.6, 20.8, 2.6))
        wrist(cv, aff)
    elif cls == 2:  # C: curved hand, opening to the left
        cv.add(T_ring(aff, 14.8, 14.6, 8.6*sz, 9.2*sz, 5.2, 5.6, a0=125-sp*8, a1=235+sp*8))
        wrist(cv, aff)
    elif cls == 3:  # D: index up, thumb-fingers circle below
        cv.add(T_capsule2(aff, 15.6, 18.0, 15.1, 3.6, 2.8))
        cv.add(T_ring(aff, 13.4, 20.2, 6.6*sz, 6.0, 2.6, 2.4))
        wrist(cv, aff)
    elif cls == 4:  # E: curled fingers, knuckle pads at top, thumb tucked
        cv.add(T_ellipse(aff, 14.4, 19.4, 7.0*sz, 7.2*sz))
        for x in (10.3, 13.0, 15.7, 18.4):
            cv.add(T_ellipse(aff, x + 0.15*p*(x-14), 11.8, 1.6, 2.7))
            cv.sub(T_capsule2(aff, x + 1.35, 14.2, x + 1.35, 11.0, 0.6), 0.55)
        wrist(cv, aff)
    elif cls == 5:  # F: thumb+index circle, three fingers up
        xs = [10.6 - 0.4*p, 14.3, 18.0 + 0.4*p]
        tys = [4.8, 4.1, 5.2]
        fingers_up(cv, aff, xs, tys, 15.8, 2.7)
        cv.add(T_ring(aff, 11.0, 19.8, 5.4*sz, 5.0, 2.1, 2.0))
        cv.add(T_ellipse(aff, 14.0, 20.4, 5.2, 5.2))
        wrist(cv, aff)
    elif cls == 6:  # G: hand horizontal, index and thumb pointing right
        cv.add(T_ellipse(aff, 7.6, 16.6, 5.4*sz, 6.2, rot=18))
        cv.add(T_capsule2(aff, 9.0, 14.2, 23.4, 13.6, 2.7))
        cv.add(T_capsule2(aff, 8.6, 18.6, 21.4, 17.8, 2.5))
        wrist(cv, aff, 'left')
    elif cls == 7:  # H: index+middle together, pointing right, with a seam
        cv.add(T_capsule2(aff, 9.0, 13.6, 23.4, 13.6, 5.4))
        cv.sub(T_capsule2(aff, 10.6, 13.7, 23.0, 13.5, 1.1*sp), 0.75)
        cv.add(T_ellipse(aff, 7.4, 17.2, 5.4*sz, 6.2))
        wrist(cv, aff, 'left')
    elif cls == 8:  # I: pinky up from fist
        cv.add(T_capsule2(aff, 17.2, 17.2, 17.9 + 0.5*p, 5.2, 2.4))
        cv.add(T_ellipse(aff, 14.2, 20.4, 6.6*sz, 6.2*sz))
        wrist(cv, aff)
    elif cls == 9:  # K: index up, middle angled out, thumb between
        cv.add(T_capsule2(aff, 12.4, 17.0, 11.7 - 0.5*p, 3.6, 2.8))
        cv.add(T_capsule2(aff, 16.6, 17.0, 20.6 + 0.7*p, 6.6, 2.8))
        cv.add(T_capsule2(aff, 14.2, 18.8, 21.6, 15.4, 2.5))
        cv.add(T_ellipse(aff, 15.6, 21.0, 6.0*sz, 5.6))
        wrist(cv, aff)
    elif cls == 10:  # L: thumb up, index out to the right
        cv.add(T_capsule2(aff, 13.9, 23.0, 13.2, 3.8, 3.0))
        cv.add(T_capsule2(aff, 14.6, 13.2, 24.0, 12.7, 2.8))
        cv.add(T_ellipse(aff, 11.4, 19.6, 4.6*sz, 5.6))
    elif cls == 11:  # M: three fingertips over thumb (palm view)
        for x in (9.8 - 0.4*p, 13.1, 16.4 + 0.4*p):
            cv.add(T_capsule2(aff, x, 17.4, x, 9.6, 2.8))
        cv.add(T_capsule2(aff, 19.6, 17.4, 19.8, 12.9, 2.4))
        cv.add(T_ellipse(aff, 14.8, 21.0, 6.8*sz, 6.0))
        wrist(cv, aff)
    elif cls == 12:  # N: two fingertips over thumb (palm view)
        for x in (10.6 - 0.4*p, 13.9):
            cv.add(T_capsule2(aff, x, 17.4, x, 9.8, 2.9))
        for x in (17.2, 19.5 + 0.4*p):
            cv.add(T_capsule2(aff, x, 17.4, x, 12.9, 2.4))
        cv.add(T_ellipse(aff, 14.8, 21.0, 6.8*sz, 6.0))
        wrist(cv, aff)
    elif cls == 13:  # O: all fingers meet thumb in an oval
        cv.add(T_ring(aff, 14.4, 14.2, 8.8*sz, 9.2*sz, 4.7, 4.9))
        wrist(cv, aff)
    elif cls == 14:  # P: like K pointing down
        cv.add(T_ellipse(aff, 15.6, 9.8, 6.0*sz, 5.6))
        cv.add(T_capsule2(aff, 16.2, 11.2, 15.6 + 0.5*p, 25.2, 2.8))
        cv.add(T_capsule2(aff, 12.0, 11.2, 8.4 - 0.5*p, 20.4, 2.8))
        cv.add(T_capsule2(aff, 14.8, 8.9, 7.8, 11.9, 2.5))
        wrist(cv, aff, 'top')
    elif cls == 15:  # Q: like G pointing down
        cv.add(T_ellipse(aff, 14.8, 9.9, 5.5*sz, 5.8))
        cv.add(T_capsule2(aff, 14.2, 11.7, 15.9, 25.8, 2.7))
        cv.add(T_capsule2(aff, 11.6, 11.3, 13.3, 23.2, 2.5))
        wrist(cv, aff, 'top')
    elif cls == 16:  # R: index and middle crossed
        cv.add(T_capsule2(aff, 12.6, 22.0, 11.2 - 0.5*p, 3.9, 2.8))
        cv.add(T_capsule2(aff, 16.0, 22.0, 17.4 + 0.5*p, 3.9, 2.8))
        cv.add(T_ellipse(aff, 14.3, 13.4, 3.0, 3.5))
        cv.add(T_ellipse(aff, 14.4, 21.2, 6.0*sz, 5.0))
        wrist(cv, aff)
    elif cls == 17:  # S: fist, thumb across the front
        cv.add(T_ellipse(aff, 14.5, 16.4, 7.5*sz, 7.6*sz))
        cv.add(T_capsule2(aff, 10.4, 15.9, 18.4, 15.2, 2.6))
        cv.add(T_ellipse(aff, 9.6, 15.5, 1.5, 2.4))
        wrist(cv, aff)
    elif cls == 18:  # T: thumb tip between index and middle knuckles
        cv.add(T_ellipse(aff, 14.5, 18.0, 7.2*sz, 7.3*sz))
        cv.add(T_capsule2(aff, 15.3, 16.2, 14.9, 11.0, 2.1))
        cv.add(T_ellipse(aff, 10.6, 12.9, 1.8, 2.3))
        cv.add(T_ellipse(aff, 18.8, 12.7, 1.8, 2.3))
        wrist(cv, aff)
    elif cls == 19:  # U: two fingers up together
        cv.add(T_capsule2(aff, 11.6 - 0.3*p, 22.2, 11.2, 4.0, 2.8))
        cv.add(T_capsule2(aff, 17.2 + 0.3*p, 22.2, 17.6, 4.2, 2.8))
        cv.sub(T_capsule2(aff, 14.4, 19.2, 14.4, 5.6, 0.85), 0.8)
        cv.add(T_ellipse(aff, 14.6, 20.8, 6.2*sz, 5.4))
        wrist(cv, aff)
    elif cls == 20:  # V: two fingers up, spread
        cv.add(T_capsule2(aff, 12.0, 22.0, 10.0 - 0.8*p, 3.8, 2.9))
        cv.add(T_capsule2(aff, 16.8, 22.0, 18.8 + 0.8*p, 3.8, 2.9))
        cv.add(T_ellipse(aff, 14.6, 21.0, 6.2*sz, 5.2))
        wrist(cv, aff)
    elif cls == 21:  # W: three fingers up, spread
        cv.add(T_capsule2(aff, 10.6, 21.6, 8.6 - 0.7*p, 4.2, 2.85))
        cv.add(T_capsule2(aff, 14.4, 21.6, 14.4, 3.4, 2.85))
        cv.add(T_capsule2(aff, 18.2, 21.6, 20.2 + 0.7*p, 4.2, 2.85))
        cv.add(T_ellipse(aff, 14.6, 20.6, 6.4*sz, 5.2))
        wrist(cv, aff)
    elif cls == 22:  # X: index hooked at the top
        cv.add(T_capsule2(aff, 15.0, 22.0, 14.6, 10.8, 2.8))
        cv.add(T_capsule2(aff, 14.6, 10.9, 17.9 + 0.5*p, 8.1, 2.6))
        cv.add(T_ellipse(aff, 14.4, 20.6, 6.7*sz, 5.9))
        wrist(cv, aff)
    else:  # 23 Y: thumb out down-left, pinky out up-right
        cv.add(T_ellipse(aff, 14.5, 15.6, 7.0*sz, 6.9*sz))
        cv.add(T_capsule2(aff, 11.8, 19.0, 5.6 - 0.5*p, 24.4, 2.8))
        cv.add(T_capsule2(aff, 17.0, 12.8, 22.2 + 0.5*p, 7.2, 2.5))
        wrist(cv, aff)
        if crc is not None:
            crc.add(T_capsule2(aff, 12.2, 17.2, 10.6, 20.4, 0.8), 0.7)
            crc.add(T_capsule2(aff, 16.0, 14.2, 18.4, 11.4, 0.8), 0.7)

def render_mask(cls, mirror=False, rot=0.0, scale=1.0, dx=0.0, dy=0.0, p=0.0, creases=True, base=1.22):
    sc = scale * base
    aff = Aff(mirror=mirror, rot=rot, scale=sc, dx=dx, dy=dy)
    aff.s = sc
    cv = Canvas(); crc = Canvas()
    draw(cls, cv, aff, p, crc=crc if creases else None)
    m = _blur(np.clip(cv.m, 0, 1), 0.75)
    cr = _blur(np.clip(cv.m, 0, 1) * np.clip(crc.m, 0, 1), 0.75) if creases else np.zeros_like(m)
    return m, cr

"""Constructs the hand-designed network: layer list + weights.
Feature pipeline: box-deviation contrast (2 scales, both polarities) + oriented
Gaussian-derivative edge energy (2 scales x 6 orientations), per-image standardized,
thresholded and capped; then per-class template matching against rendered hand-shape
models, max over pose variants; dense calibration on top."""

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

def _blur(m, sig):
    r = int(np.ceil(sig * 2.5)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
    m = np.apply_along_axis(lambda x: np.convolve(x, k, 'same'), 0, m)
    return np.apply_along_axis(lambda x: np.convolve(x, k, 'same'), 1, m)

NC = 24

def box_dev(k):
    W = np.full((k, k), -1.0 / (k * k)); W[k // 2, k // 2] += 1.0
    return W

def gderiv(sig, theta_deg):
    r = int(np.ceil(sig * 2.5))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    ct, st = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    u = xx * ct + yy * st  # coordinate along filter direction
    G = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
    d = -(u / sig ** 2) * G          # first derivative of Gaussian along u (zero DC)
    return d

def place(kern, K=21):
    k = kern.shape[0]
    if k % 2 == 0:
        kern = np.pad(kern, ((0, 1), (0, 0))); k += 1
    if k <= K:
        out = np.zeros((K, K)); o = (K - k) // 2
        out[o:o + k, o:o + k] = kern
        return out
    o = (k - K) // 2
    return kern[o:o + K, o:o + K]

THETAS = [0, 30, 60, 90, 120, 150]

def feat_bank(t9=1.1, t21=0.75, tg1=1.15, tg2=0.95, w9=2.2, w21=1.8, wg1=2.4, wg2=2.2,
              sig1=1.7, sig2=2.8, KS=25):
    """Smoothed feature kernels: blur-then-derive so pixel grain is suppressed.
    Channels (28): [+/- cb9 (G0.8*boxdev9), +/- cb21 (G2.2*boxdev21),
    +/- gabor sig1 x6, +/- gabor sig2 x6], all placed centrally in KS x KS."""
    def gblur(sig, k):
        r = int(np.ceil(sig * 3)); ar = np.arange(-r, r + 1)
        G = np.exp(-0.5 * (ar / sig) ** 2); G /= G.sum()
        K = np.outer(G, G)
        if K.shape[0] < k:
            K = np.pad(K, ((0, k - K.shape[0]), (0, k - K.shape[0])))
        return K / K.sum()
    def conv2(a, b):
        out = np.zeros((a.shape[0] + b.shape[0] - 1,) * 2)
        for i in range(a.shape[0]):
            for j in range(a.shape[0]):
                out[i:i + b.shape[0], j:j + b.shape[0]] += a[i, j] * b
        return out
    bd9 = box_dev(9)
    bd21 = box_dev(21)
    cb9 = conv2(bd9, gblur(0.8, 7))       # ~15x15
    cb21 = conv2(bd21, gblur(2.2, 13))    # ~33x33, centrally truncated to KS
    chans = [cb9, -cb9, cb21, -cb21]
    for sig in (sig1, sig2):
        for th in THETAS:
            chans.append(gderiv(sig, th))
        for th in THETAS:
            chans.append(-gderiv(sig, th))
    W = np.stack([place(c, KS) for c in chans])[:, None]
    t = np.zeros(28); cap = np.zeros(28)
    t[0:2] = t9; cap[0:2] = w9
    t[2:4] = t21; cap[2:4] = w21
    t[4:16] = tg1; cap[4:16] = wg1
    t[16:28] = tg2; cap[16:28] = wg2
    return W, t, cap

def mix_layout():
    """mix conv1x1: 56 -> 16 channel construction (capped P/N + edge energy)."""
    W = np.zeros((16, 56))
    # out 0..3: P9, N9, P21, N21  = capped pairs of sources 0..3
    for o, s in enumerate([0, 1, 2, 3]):
        W[o, 2 * s] = 1.0; W[o, 2 * s + 1] = -1.0
    # out 4..9: E sigma1 theta 0..5 ; out 10..15: E sigma2
    for s2 in (1, 2):
        base_out = 4 if s2 == 1 else 10
        base_src = 4 if s2 == 1 else 16
        for th in range(6):
            s = base_src + th; sn = base_src + 6 + th
            W[base_out + th, 2 * s] = 1.0; W[base_out + th, 2 * s + 1] = -1.0
            W[base_out + th, 2 * sn] = 1.0; W[base_out + th, 2 * sn + 1] = -1.0
    b = np.zeros(16)
    return W, b

def variant_list():
    """36 pose variants per class: mirror x rot x scale x pose-jitter."""
    vs = []
    for mirror in (False, True):
        for rot in (-4.0, 0.0, 4.0):
            for sc in (0.90, 1.0, 1.06):
                for p in (-0.6, 0.6):
                    vs.append(dict(mirror=mirror, rot=rot, scale=sc, p=p))
    return vs  # 36

def template_raster(cls, v, seed, delta=0.22, bblur=1.5):
    """Realistic pseudo-photo: soft hand boundary, shallow creases, flat bg + grain."""
    rng = np.random.default_rng(seed)
    m, cr = render_mask(cls, mirror=v["mirror"], rot=v["rot"], scale=v["scale"], p=v["p"])
    m = _blur(m, bblur)
    bg = 0.55 + 0.04 * rng.normal(0, 1, (28, 28))
    hv = 0.55 - delta
    img = bg * (1 - m) + hv * (1 - 0.35 * cr) * m
    img = img + rng.normal(0, 0.05, (28, 28))
    return np.clip(img, 0, 1).astype(np.float32)

def features_of(rasters, Wf, t, cap, Wmix, batch=256):
    """Run the feature pipeline (feat conv -> norm -> thr -> relu -> mix -> 2x2 maxpool)
    on a stack of rasters (N,1,28,28). Returns (N,16,14,14). Mirrors network ops."""
    out = []
    for i in range(0, len(rasters), batch):
        x = rasters[i:i + batch]
        # exactly as the network: norm -> feat conv -> norm -> thr -> relu -> mix -> 2x2 maxpool
        mu0 = x.mean(axis=(2, 3), keepdims=True)
        x = (x - mu0) / np.sqrt(((x - mu0) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        f = _conv(x, Wf, np.zeros(28), 1, (Wf.shape[-1] - 1) // 2)           # (N,28,28,28)
        mu = f.mean(axis=(2, 3), keepdims=True)
        f = (f - mu) / np.sqrt(((f - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        a = np.maximum(f - t[None, :, None, None], 0)
        b = np.maximum(f - t[None, :, None, None] - cap[None, :, None, None], 0)
        r = np.stack([a, b], axis=2).reshape(len(x), 56, 28, 28)
        mixed = np.einsum('nchw,oc->nohw', r, Wmix)
        p2 = _pool(mixed, [2, 2], [2, 2], 'max')
        out.append(p2)
    return np.concatenate(out)


def fullres_features(rasters, Wf, t, cap, Wmix, batch=128):
    """norm -> feat -> norm -> thr -> relu -> mix -> per-channel norm at 28x28. No pooling."""
    out = []
    for i in range(0, len(rasters), batch):
        x = rasters[i:i + batch]
        mu0 = x.mean(axis=(2, 3), keepdims=True)
        x = (x - mu0) / np.sqrt(((x - mu0) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        f = _conv(x, Wf, np.zeros(28), 1, (Wf.shape[-1] - 1) // 2)
        mu = f.mean(axis=(2, 3), keepdims=True)
        f = (f - mu) / np.sqrt(((f - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        a = np.maximum(f - t[None, :, None, None], 0)
        b = np.maximum(f - t[None, :, None, None] - cap[None, :, None, None], 0)
        r = np.stack([a, b], axis=2).reshape(len(x), 56, 28, 28)
        mixed = np.einsum('nchw,oc->nohw', r, Wmix)
        mu = mixed.mean(axis=(2, 3), keepdims=True)
        sd = np.sqrt(((mixed - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)
        out.append((mixed - mu) / sd)
    return np.concatenate(out)



def make_kernels(T):
    """Per-channel mean-centered, unit-L2 template kernels (so uniform feature
    energy anywhere contributes exactly zero to every class score)."""
    K = T - T.mean(axis=(2, 3), keepdims=True)
    Kn = K.reshape(len(K), -1)
    Kn = Kn / np.maximum(np.linalg.norm(Kn, axis=1, keepdims=True), 1e-9)
    return Kn.reshape(T.shape).astype(np.float32)


def build():
    thr_kw = dict(t9=1.1, t21=0.75, tg1=0.9, tg2=0.75)
    lamB = 0.35
    Wf, t, cap = feat_bank(**thr_kw)
    Wmix, bmix = mix_layout()
    vs = variant_list()
    NV = len(vs); V = 24 * NV
    rasters = np.array([template_raster(c, v, seed=1000 * (c + 1) + j)
                        for c in range(24) for j, v in enumerate(vs)])[:, None]
    TA = features_of(rasters, Wf, t, cap, Wmix)
    KA = make_kernels(TA)
    TB = fullres_features(rasters, Wf, t, cap, Wmix)
    KB = make_kernels(TB[:, :, 2:26, 2:26])
    layers = [
        {"name": "nrm", "op": "norm", "inputs": ["image"]},
        {"name": "feat", "op": "conv", "inputs": ["nrm"], "out": 28, "kernel": [25, 25], "stride": 1, "padding": 12},
        {"name": "fnrm", "op": "norm", "inputs": ["feat"]},
        {"name": "thr", "op": "conv", "inputs": ["fnrm"], "out": 56, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "thr_r", "op": "relu", "inputs": ["thr"]},
        {"name": "mix", "op": "conv", "inputs": ["thr_r"], "out": 16, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "p2", "op": "maxpool", "inputs": ["mix"], "size": [2, 2], "stride": [2, 2]},
        {"name": "matchA", "op": "conv", "inputs": ["p2"], "out": V, "kernel": [14, 14], "stride": 1, "padding": 2},
        {"name": "mpA", "op": "maxpool", "inputs": ["matchA"], "size": [5, 5], "stride": [5, 5]},
        {"name": "rshA", "op": "reshape", "inputs": ["mpA"], "shape": [24, NV, 1]},
        {"name": "vmaxA", "op": "maxpool", "inputs": ["rshA"], "size": [NV, 1], "stride": [NV, 1]},
        {"name": "a24", "op": "reshape", "inputs": ["vmaxA"], "shape": [24]},
        {"name": "matchB", "op": "conv", "inputs": ["mix"], "out": V, "kernel": [24, 24], "stride": 2, "padding": 2},
        {"name": "mpB", "op": "maxpool", "inputs": ["matchB"], "size": [5, 5], "stride": [5, 5]},
        {"name": "rshB", "op": "reshape", "inputs": ["mpB"], "shape": [24, NV, 1]},
        {"name": "vmaxB", "op": "maxpool", "inputs": ["rshB"], "size": [NV, 1], "stride": [NV, 1]},
        {"name": "b24", "op": "reshape", "inputs": ["vmaxB"], "shape": [24]},
        {"name": "sumAB", "op": "add", "inputs": ["a24", "b24"]},
        {"name": "out", "op": "dense", "inputs": ["sumAB"], "out": 24},
    ]
    Wthr = np.zeros((56, 28)); bthr = np.zeros(56)
    for i in range(28):
        Wthr[2 * i, i] = 1.0; Wthr[2 * i + 1, i] = 1.0
        bthr[2 * i] = -t[i]; bthr[2 * i + 1] = -t[i] - cap[i]
    weights = {
        "feat.W": Wf.astype(np.float32), "feat.b": np.zeros(28, np.float32),
        "thr.W": Wthr[:, :, None, None].astype(np.float32), "thr.b": bthr.astype(np.float32),
        "mix.W": Wmix[:, :, None, None].astype(np.float32), "mix.b": bmix.astype(np.float32),
        "matchA.W": KA.astype(np.float32), "matchA.b": np.zeros(V, np.float32),
        "matchB.W": (KB * lamB).astype(np.float32), "matchB.b": np.zeros(V, np.float32),
        "out.W": np.eye(24).astype(np.float32), "out.b": np.zeros(24, np.float32),
    }
    return {"layers": layers}, weights


if __name__ == "__main__":
    out_dir = sys.argv[1]
    import json, os
    net, weights = build()
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)
    print("wrote", out_dir, "weights:", sum(int(np.prod(v.shape)) for v in weights.values()))
