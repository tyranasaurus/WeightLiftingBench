"""build.py - writes network.json and weights.npz for the sign-language letter network.  Reads no files.

Part 1 draws hands.  Parametric 2-D drawings of the 24 static ASL letters as a viewer facing a right-handed signer
       sees them, typed from general knowledge of the hand shapes (a fist or palm block, finger capsules at given
       angles, a thumb, a forearm).  There are two sets: plain drawings, and drawings with a tapering wrist, creases
       between folded fingers and the shadows under fingertips and thumbs.  Every drawing gets random articulation
       and pose (fixed seed).
Part 2 is the fixed front-end of the network, built from formulas only: contrast normalisation, derivative-of-
       Gaussian filters at 6 orientations, rectification into unsigned orientation maps, a dead zone and a clip,
       Gaussian pooling at two resolutions (plus an orientation-free edge-density map), row weights, normalisation.
Part 3 turns every drawing into one template: the drawing is passed through the same front-end formulas and its own
       normalised feature vector becomes one row of a dense layer (no statistic is taken across drawings).  A letter's
       score in a bank is the best correlation between the image and that letter's templates (max-pool over them);
       the two banks' scores are added.
Usage: python3 build.py OUT_DIR
"""
import json
import numpy as np

SS = 2            # supersampling factor
BG = 0.50         # background brightness of the drawings
HAND = 0.85       # hand brightness
RIM_W = 0.5       # width (pixels) of that darkening
DARK = 0.50       # shade of creases, shadows under fingertips and dark gaps (relative to the hand)


class Scene:
    """Collects parts (drawn back to front) for a batch of B drawings; all coordinates may be scalars or (B,) arrays."""
    def __init__(self, B, rng, jscale=1.0, rim=0.3):
        self.B, self.rng, self.parts, self.jscale, self.open, self.rot = B, rng, [], jscale, None, None
        self.rim = rim        # darkening at the boundary of every part (gives creases between neighbouring parts)

    def j(self, sd):
        """a jitter value per drawing: normal, clipped at 2 sd"""
        if self.jscale == 0:
            return np.zeros(self.B)
        return np.clip(self.rng.normal(0.0, 1.0, self.B), -2, 2) * sd * self.jscale

    def set_rot(self, ang=None, px=0.0, py=0.0):
        """parts added from now on are turned by ang degrees (clockwise on screen) about (px, py); None switches it off"""
        self.rot = None if ang is None else (np.deg2rad(ang), px, py)

    def _pt(self, x, y):
        if self.rot is None:
            return x, y
        a, px, py = self.rot
        return px + (x - px) * np.cos(a) - (y - py) * np.sin(a), py + (x - px) * np.sin(a) + (y - py) * np.cos(a)

    def _add(self, prim, shade, rim):
        rim = self.rim if rim is None else rim
        if self.rot is not None:
            if prim[0] == "cap":
                x0, y0 = self._pt(prim[1], prim[2])
                x1, y1 = self._pt(prim[3], prim[4])
                prim = ("cap", x0, y0, x1, y1, prim[5], prim[6])
            elif prim[0] == "box":
                x0, y0 = self._pt(prim[1], prim[2])
                prim = ("box", x0, y0, prim[3], prim[4], prim[5], prim[6] + np.rad2deg(self.rot[0]))
            else:
                x0, y0 = self._pt(prim[1], prim[2])
                prim = ("ell", x0, y0, prim[3], prim[4], prim[5] + np.rad2deg(self.rot[0]), 0.0)
        if self.open is not None:
            self.open.append(prim)
        else:
            self.parts.append(([prim], shade, rim))

    def begin(self):
        """start a group: the parts added until end() are merged into one shape with one outline"""
        self.open = []

    def end(self, shade=1.0, rim=None):
        self.parts.append((self.open, shade, self.rim if rim is None else rim))
        self.open = None

    def cap(self, x0, y0, x1, y1, r0, r1=None, shade=1.0, rim=None):
        self._add(("cap", x0, y0, x1, y1, r0, r0 if r1 is None else r1), shade, rim)

    def box(self, cx, cy, hx, hy, rc, ang=0.0, shade=1.0, rim=None):
        self._add(("box", cx, cy, hx, hy, rc, ang), shade, rim)

    def ell(self, cx, cy, rx, ry, ang=0.0, shade=1.0, rim=None):
        self._add(("ell", cx, cy, rx, ry, ang, 0.0), shade, rim)

    def chain(self, pts, radii, shade=1.0, rim=None):
        own = self.open is None
        if own:
            self.begin()
        for (p, q, r, s) in zip(pts[:-1], pts[1:], radii[:-1], radii[1:]):
            self.cap(p[0], p[1], q[0], q[1], r, s)
        if own:
            self.end(shade, rim)

    def finger(self, x0, y0, ang, length, r, shade=1.0, sd_ang=6.0, sd_len=0.07, r1=None):
        """straight finger from its knuckle (x0, y0); ang in degrees from straight up, positive to the right"""
        a = np.deg2rad(ang + self.j(sd_ang))
        L = length * (1 + self.j(sd_len))
        self.cap(x0, y0, x0 + L * np.sin(a), y0 - L * np.cos(a), r, (0.9 * r) if r1 is None else r1, shade)
        return x0 + L * np.sin(a), y0 - L * np.cos(a)


def _col(v):
    return np.asarray(v, dtype=np.float64).reshape(-1, 1, 1) if np.ndim(v) else float(v)


def render(scene, A=None, t=None, size=28):
    """Rasterise a Scene.  A: (B, 2, 2) linear part and t: (B, 2) shift of the map canonical -> image
    (about the frame centre).  Returns (B, size, size) images."""
    B = scene.B
    n = size * SS
    c = (np.arange(n) + 0.5) / SS
    X, Y = np.meshgrid(c, c)
    X = X[None] - size / 2.0
    Y = Y[None] - size / 2.0
    if A is None:
        A = np.tile(np.eye(2), (B, 1, 1))
    if t is None:
        t = np.zeros((B, 2))
    Ai = np.linalg.inv(A)
    px = X - t[:, 0].reshape(-1, 1, 1)
    py = Y - t[:, 1].reshape(-1, 1, 1)
    U = Ai[:, 0, 0].reshape(-1, 1, 1) * px + Ai[:, 0, 1].reshape(-1, 1, 1) * py + size / 2.0
    V = Ai[:, 1, 0].reshape(-1, 1, 1) * px + Ai[:, 1, 1].reshape(-1, 1, 1) * py + size / 2.0
    sc = np.sqrt(np.abs(np.linalg.det(A))).reshape(-1, 1, 1)      # canonical length -> image pixels
    img = np.full((B, n, n), BG)
    for prims, shade, rim in scene.parts:
        d = None
        for kind, a0, a1, a2, a3, a4, a5 in prims:
            a0, a1, a2, a3, a4, a5 = _col(a0), _col(a1), _col(a2), _col(a3), _col(a4), _col(a5)
            if kind == "cap":
                dx, dy = a2 - a0, a3 - a1
                tt = np.clip(((U - a0) * dx + (V - a1) * dy) / (dx * dx + dy * dy + 1e-9), 0.0, 1.0)
                dd = np.sqrt((U - a0 - tt * dx) ** 2 + (V - a1 - tt * dy) ** 2) - (a4 + tt * (a5 - a4))
            elif kind == "box":
                ca, sa = np.cos(np.deg2rad(a5)), np.sin(np.deg2rad(a5))
                u = (U - a0) * ca + (V - a1) * sa
                v = -(U - a0) * sa + (V - a1) * ca
                qx, qy = np.abs(u) - (a2 - a4), np.abs(v) - (a3 - a4)
                dd = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - a4
            else:
                ca, sa = np.cos(np.deg2rad(a4)), np.sin(np.deg2rad(a4))
                u = (U - a0) * ca + (V - a1) * sa
                v = -(U - a0) * sa + (V - a1) * ca
                dd = (np.sqrt((u / a2) ** 2 + (v / a3) ** 2) - 1.0) * np.minimum(a2, a3)
            d = dd if d is None else np.minimum(d, dd)
        d = d * sc                                                  # signed distance in image pixels
        cov = np.clip(0.5 - d * SS, 0.0, 1.0)
        if shade is None:
            val = BG
        else:
            val = HAND * shade * (1.0 - rim * np.exp(np.minimum(d, 0.0) / RIM_W))
        img = img * (1.0 - cov) + val * cov
    return img.reshape(B, size, SS, size, SS).mean(axis=(2, 4))


# ----------------------------------------------------------------------------------------------------------
# SET 1: plain drawings (a block for the fist or palm, a straight forearm, finger capsules)
# ----------------------------------------------------------------------------------------------------------
def body(S, cx, cy, hx, hy, rc, ax=None, ay=None, ar=5.2, lean=0.0, ang=0.0):
    """fist / palm block merged with a forearm that leaves through the bottom edge"""
    S.begin()
    ax = cx if ax is None else ax
    ay = cy + hy - 2.0 if ay is None else ay
    S.cap(ax, ay, ax + lean + S.j(0.8), 42.0, ar + S.j(0.3), ar + 0.4)
    S.box(cx, cy, hx, hy, rc, ang)
    S.end()


def strips1(S, xs, y0, y1, r=1.6, dy=None, shade=1.0):
    """folded fingers seen from the front: short vertical strips side by side"""
    for i, x in enumerate(xs):
        o = 0.0 if dy is None else dy[i]
        S.cap(x, y0 + o, x + S.j(0.2), y1 + o + S.j(0.5), r, r, shade)


def p_A(S):
    cx, top = 14.2 + S.j(0.5), 4.5 + S.j(0.6)
    body(S, cx, top + 9.0, 7.3, 9.0, 4.5, ar=6.0)
    strips1(S, [cx - 5.4, cx - 1.85, cx + 1.7, cx + 5.2], top + 1.9, top + 7.6 + S.j(0.6), 1.75, dy=[0.7, 0.1, 0.0, 0.3])
    tx = cx + 7.7 + S.j(0.4)
    S.cap(tx - 0.4, top + 13.5, tx + S.j(0.5), top + 1.5 + S.j(0.9), 2.2, 1.9)           # thumb upright beside the fist


def p_E(S):
    cx, top = 14.5 + S.j(0.5), 4.0 + S.j(0.6)
    body(S, cx, top + 9.6, 7.5, 9.6, 6.0, ar=6.0)
    strips1(S, [cx - 5.4, cx - 1.85, cx + 1.7, cx + 5.2], top + 2.2, top + 6.2 + S.j(0.5), 1.75, dy=[1.0, 0.1, 0.0, 0.5])
    ty = top + 10.5 + S.j(0.6)
    S.cap(cx + 6.8, ty + 1.2, cx - 4.2 + S.j(0.8), ty + S.j(0.4), 2.0, 1.8)              # thumb lying under the fingertips


def p_M(S):
    cx, top = 14.8 + S.j(0.5), 4.6 + S.j(0.6)
    body(S, cx, top + 9.2, 7.6, 9.2, 5.0, ar=6.0)
    S.ell(cx - 4.0 + S.j(0.5), top + 11.2 + S.j(0.5), 1.7, 1.6)                          # thumb tip peeping out
    strips1(S, [cx - 5.5], top + 3.6, top + 7.6, 1.65)
    strips1(S, [cx - 1.9, cx + 1.7, cx + 5.2], top + 2.0, top + 9.6 + S.j(0.7), 1.75, dy=[0.8, 0.2, 0.0])


def p_N(S):
    cx, top = 14.8 + S.j(0.5), 4.6 + S.j(0.6)
    body(S, cx, top + 9.2, 7.6, 9.2, 5.0, ar=6.0)
    S.ell(cx - 0.5 + S.j(0.5), top + 11.0 + S.j(0.5), 1.7, 1.6)                          # thumb tip between middle and ring
    strips1(S, [cx - 5.5, cx - 2.0], top + 3.4, top + 7.2, 1.65, dy=[0.8, 0.0])
    strips1(S, [cx + 1.7, cx + 5.2], top + 2.0, top + 9.6 + S.j(0.7), 1.75, dy=[0.2, 0.0])


def p_S(S):
    cx, top = 14.5 + S.j(0.5), 4.6 + S.j(0.6)
    body(S, cx, top + 9.2, 7.5, 9.2, 4.6, ar=6.0)
    strips1(S, [cx - 5.4, cx - 1.85, cx + 1.7, cx + 5.2], top + 2.0, top + 6.4 + S.j(0.5), 1.75, dy=[1.6, 0.8, 0.2, 0.0])
    ty = top + 9.0 + S.j(0.6)
    S.cap(cx + 7.0, ty + 3.2, cx - 3.0 + S.j(0.8), ty + S.j(0.5), 2.1, 1.9)              # thumb across the fingers


def p_T(S):
    # as photographed: the fist is turned, the bent index juts out sideways and the thumb tip stands up beside it
    cx, top = 12.0 + S.j(0.6), 9.0 + S.j(0.7)
    body(S, cx, top + 8.0, 6.3, 8.0, 3.8, ax=cx - 1.0, ar=5.8, lean=-1.5)
    S.cap(cx + 1.0, top + 2.2, cx + 12.5 + S.j(0.9), top + 1.6 + S.j(0.9), 2.1, 1.7)
    S.cap(cx + 1.0, top + 1.5, cx + 1.6 + S.j(0.6), top - 5.2 + S.j(0.7), 1.9, 1.6)


def p_D(S):
    cx, top = 15.0 + S.j(0.5), 9.0 + S.j(0.7)
    body(S, cx, top + 7.2, 6.6, 7.2, 4.6, ar=5.6)
    strips1(S, [cx - 4.7, cx - 1.6, cx + 1.5], top + 1.7, top + 4.5, 1.6, dy=[0.6, 0.1, 0.0])
    S.cap(cx + 5.4, top + 10.0, cx + 0.5 + S.j(0.6), top + 6.4 + S.j(0.5), 1.9, 1.7)
    S.finger(cx + 4.9, top + 1.5, 2.0, 9.0, 1.55)


def p_R(S):
    cx, top = 14.0 + S.j(0.5), 10.5 + S.j(0.7)
    body(S, cx, top + 6.8, 6.2, 6.8, 4.2, ar=5.4)
    strips1(S, [cx - 4.4, cx - 1.5], top + 1.7, top + 4.6, 1.55, dy=[0.5, 0.0])
    S.cap(cx + 5.2, top + 9.4, cx + 0.2 + S.j(0.6), top + 6.2 + S.j(0.5), 1.9, 1.7)
    a = S.j(4.0)
    S.finger(cx + 2.0, top + 1.5, 8.0 + a, 9.8, 1.5, sd_ang=2.0)
    S.finger(cx + 4.3, top + 1.5, -7.0 + a, 9.4, 1.5, sd_ang=2.0)


def p_U(S):
    cx, top = 14.0 + S.j(0.5), 10.5 + S.j(0.7)
    body(S, cx, top + 6.8, 6.2, 6.8, 4.2, ar=5.4)
    strips1(S, [cx - 4.4, cx - 1.5], top + 1.7, top + 4.6, 1.55, dy=[0.5, 0.0])
    S.cap(cx + 5.2, top + 9.4, cx + 0.2 + S.j(0.6), top + 6.2 + S.j(0.5), 1.9, 1.7)
    a = S.j(4.0)
    S.finger(cx + 1.5, top + 1.5, 1.0 + a, 10.2, 1.55, sd_ang=1.5)
    S.finger(cx + 4.7, top + 1.5, 3.0 + a, 9.6, 1.55, sd_ang=1.5)


def p_V(S):
    cx, top = 14.0 + S.j(0.5), 10.8 + S.j(0.7)
    body(S, cx, top + 6.8, 6.2, 6.8, 4.2, ar=5.4)
    strips1(S, [cx - 4.4, cx - 1.5], top + 1.7, top + 4.6, 1.55, dy=[0.5, 0.0])
    S.cap(cx + 5.2, top + 9.4, cx + 0.2 + S.j(0.6), top + 6.2 + S.j(0.5), 1.9, 1.7)
    S.finger(cx + 1.3, top + 1.5, -9.0, 10.4, 1.5, sd_ang=5.0)
    S.finger(cx + 4.5, top + 1.5, 22.0, 10.0, 1.5, sd_ang=5.0)


def p_W(S):
    cx, top = 14.5 + S.j(0.5), 11.5 + S.j(0.7)
    body(S, cx, top + 6.6, 6.2, 6.6, 4.0, ar=5.4)
    S.cap(cx + 4.8, top + 9.4, cx - 2.8 + S.j(0.6), top + 6.0 + S.j(0.5), 1.9, 1.7)      # thumb holding the little finger
    S.finger(cx - 3.1, top + 1.6, -13.0, 9.6, 1.45, sd_ang=5.0)
    S.finger(cx + 0.3, top + 1.2, 2.0, 10.6, 1.5, sd_ang=4.0)
    S.finger(cx + 3.7, top + 1.6, 21.0, 9.9, 1.45, sd_ang=5.0)


def p_K(S):
    cx, top = 14.0 + S.j(0.5), 11.5 + S.j(0.7)
    body(S, cx, top + 7.0, 5.8, 7.0, 4.2, ar=5.2)
    strips1(S, [cx - 3.6, cx - 0.9], top + 2.8, top + 5.2, 1.5, dy=[0.4, 0.0])
    S.cap(cx + 4.6, top + 8.0, cx + 1.8 + S.j(0.5), top + 2.6 + S.j(0.6), 1.9, 1.5)      # thumb reaching up between the two fingers
    S.finger(cx + 0.6, top + 2.2, -15.0, 7.2, 1.5, sd_ang=7.0, sd_len=0.12)              # middle finger, leaning forward / left
    S.finger(cx + 3.4, top + 1.8, 4.0, 10.8, 1.5, sd_ang=5.0)                           # index up


def p_I(S):
    cx, top = 15.5 + S.j(0.5), 8.0 + S.j(0.7)
    body(S, cx, top + 8.0, 7.2, 8.0, 4.8, ar=5.8)
    strips1(S, [cx - 2.2, cx + 1.3, cx + 4.8], top + 1.9, top + 6.0, 1.7, dy=[0.2, 0.0, 0.4])
    S.cap(cx + 6.6, top + 10.5, cx - 0.4 + S.j(0.7), top + 7.9 + S.j(0.5), 2.0, 1.8)
    S.finger(cx - 5.6, top + 2.8, -4.0, 8.2, 1.35)


def p_Y(S):
    cx, top = 14.0 + S.j(0.5), 9.0 + S.j(0.7)
    body(S, cx, top + 7.3, 6.5, 7.3, 4.5, ar=5.6)
    strips1(S, [cx - 3.0, cx + 0.2, cx + 3.4], top + 1.8, top + 5.6, 1.6, dy=[0.2, 0.0, 0.3])
    S.finger(cx - 5.2, top + 2.8, -24.0, 8.4, 1.35, sd_ang=7.0)
    S.finger(cx + 5.2, top + 8.0, 47.0, 10.4, 2.1, sd_ang=8.0, r1=1.6)                   # thumb out to the side


def p_L(S):
    cx, top = 10.5 + S.j(0.6), 10.5 + S.j(0.7)
    body(S, cx, top + 7.2, 6.6, 7.2, 4.2, ax=cx - 0.8, ar=5.6, lean=-1.0)
    strips1(S, [cx - 4.9, cx - 2.0, cx + 0.9], top + 1.7, top + 5.8, 1.5, dy=[1.0, 0.4, 0.0])
    S.finger(cx + 4.6, top + 8.8, 64.0, 11.0, 2.3, sd_ang=8.0, r1=1.6)                   # thumb
    S.finger(cx + 4.2, top + 1.5, 3.0, 10.0, 1.5, sd_ang=5.0)                           # index


def p_F(S):
    cx, top = 12.5 + S.j(0.6), 12.0 + S.j(0.7)
    body(S, cx, top + 6.5, 5.8, 6.5, 3.5, ax=cx - 0.5, ar=5.2)
    S.finger(cx - 4.2, top + 1.8, -22.0, 8.8, 1.3, sd_ang=6.0)
    S.finger(cx - 1.4, top + 1.0, -4.0, 10.8, 1.45, sd_ang=5.0)
    S.finger(cx + 1.6, top + 1.0, 14.0, 11.0, 1.5, sd_ang=5.0)
    mx, my = cx + 10.0 + S.j(0.7), top + 2.5 + S.j(0.7)                                  # where index tip and thumb tip meet
    S.chain([(cx + 3.8, top + 2.0), (cx + 8.3 + S.j(0.5), top - 0.8 + S.j(0.5)), (mx, my)], [1.7, 1.55, 1.3])
    S.chain([(cx + 3.8, top + 9.0), (cx + 8.6 + S.j(0.5), top + 6.6 + S.j(0.5)), (mx, my)], [2.2, 1.9, 1.4])
    S.ell(cx + 6.6, top + 3.2 + S.j(0.4), 2.1 + S.j(0.3), 1.5 + S.j(0.2), -15.0, shade=None)


def p_B(S):
    cx, top = 15.0 + S.j(0.5), 12.5 + S.j(0.7)
    S.begin()
    S.cap(cx + 0.2, top + 11.0, cx + 0.2 + S.j(0.8), 42.0, 5.3 + S.j(0.3), 5.7)
    S.box(cx, top + 7.0, 6.1, 7.2, 3.0)
    S.ell(cx + 5.8, top + 6.4 + S.j(0.5), 2.0, 3.8)                                      # ball of the thumb
    S.end()
    a = S.j(3.0)
    S.finger(cx - 4.5, top + 1.0, -2.0 + a, 9.0, 1.4, sd_ang=2.0)
    S.finger(cx - 1.5, top + 0.4, -1.0 + a, 10.8, 1.5, sd_ang=1.5)
    S.finger(cx + 1.5, top + 0.3, 0.0 + a, 11.5, 1.52, sd_ang=1.5)
    S.finger(cx + 4.5, top + 0.6, 1.0 + a, 10.5, 1.5, sd_ang=2.0)
    S.cap(cx + 5.6, top + 8.8, cx + 0.6 + S.j(0.7), top + 4.6 + S.j(0.6), 2.0, 1.7)      # thumb folded over the palm


def p_C(S):
    # seen from the side: the back of the hand rises on the left, the fingers arch over to the right, the thumb reaches out below
    cx, top = 11.0 + S.j(0.6), 5.0 + S.j(0.7)
    ox, oy = S.j(0.9), S.j(0.8)
    tx, ty = S.j(0.9), S.j(0.9)
    S.begin()
    S.cap(cx - 2.0, top + 19.0, cx - 6.0 + S.j(1.0), 42.0, 6.0 + S.j(0.3), 6.4)
    S.cap(cx - 1.5, top + 17.0, cx, top + 6.0, 5.5, 5.0)
    S.chain([(cx, top + 6.0), (cx + 1.5, top + 1.6), (cx + 6.0 + 0.5 * ox, top + 0.2 + 0.5 * oy),
             (cx + 11.0 + ox, top + 0.8 + oy)], [5.0, 3.7, 3.1, 2.4])
    S.end()
    S.chain([(cx + 1.5, top + 15.0), (cx + 7.0 + 0.5 * tx, top + 12.8 + 0.5 * ty), (cx + 12.4 + tx, top + 9.6 + ty)],
            [3.0, 2.3, 1.8])


def p_O(S):
    # seen from the side like C, but the fingertips come down onto the thumb and close a ring
    cx, top = 10.8 + S.j(0.6), 4.8 + S.j(0.7)
    mx, my = cx + 10.2 + S.j(0.7), top + 7.0 + S.j(0.7)                                  # fingertips meet the thumb here
    S.begin()
    S.cap(cx + 0.7, top + 18.0, cx - 0.3 + S.j(1.0), 42.0, 6.2 + S.j(0.3), 6.6)
    S.cap(cx + 0.2, top + 17.2, cx - 0.3, top + 7.2, 5.6, 5.0)
    S.chain([(cx - 0.3, top + 7.2), (cx + 0.7, top + 2.2), (cx + 5.2, top + 0.0), (cx + 9.7 + S.j(0.5), top + 2.2 + S.j(0.5)),
             (mx, my)], [5.0, 3.6, 3.0, 2.6, 2.1])
    S.end()
    S.chain([(cx + 2.2, top + 15.7), (cx + 7.2 + S.j(0.5), top + 11.7 + S.j(0.5)), (mx, my)], [3.2, 2.5, 1.9])
    S.ell(cx + 5.7 + S.j(0.5), top + 5.8 + S.j(0.5), 2.6 + S.j(0.3), 2.1 + S.j(0.3), -20.0, shade=None)


def p_G(S):
    cx, cy = 9.0 + S.j(0.7), 15.0 + S.j(0.7)
    S.begin()
    S.cap(cx - 2.0, cy + 0.5, cx - 24.0, cy + 7.0 + S.j(1.5), 6.2, 6.6)                  # forearm from the left
    S.box(cx, cy, 6.8, 7.0, 4.0)
    S.end()
    for k, yy in enumerate([0.5, 3.2, 5.6]):                                            # folded fingers seen side on
        S.cap(cx + 1.5, cy + yy, cx + 6.2 - 0.6 * k + S.j(0.4), cy + yy + S.j(0.3), 1.4, 1.4)
    S.cap(cx + 1.0, cy - 2.6, cx + 7.5 + S.j(0.7), cy - 2.4 + S.j(0.4), 1.7, 1.5)        # thumb lying along the index
    a = np.deg2rad(S.j(6.0) - 4.0)
    L = 14.0 * (1 + S.j(0.07))
    S.cap(cx + 2.5, cy - 5.3, cx + 2.5 + L * np.cos(a), cy - 5.3 + L * np.sin(a), 1.75, 1.55)


def p_H(S):
    cx, cy = 9.0 + S.j(0.7), 15.0 + S.j(0.7)
    S.begin()
    S.cap(cx - 2.0, cy + 0.5, cx - 24.0, cy + 7.0 + S.j(1.5), 6.2, 6.6)
    S.box(cx, cy, 6.8, 7.0, 4.0)
    S.end()
    for k, yy in enumerate([0.9, 3.8]):
        S.cap(cx + 1.5, cy + yy, cx + 6.0 - 0.6 * k + S.j(0.4), cy + yy + S.j(0.3), 1.45, 1.45)
    a = np.deg2rad(S.j(6.0) - 4.0)
    L = 14.0 * (1 + S.j(0.07))
    S.cap(cx + 2.5, cy - 2.5, cx + 2.5 + 0.97 * L * np.cos(a), cy - 2.5 + 0.97 * L * np.sin(a), 1.7, 1.5)
    S.cap(cx + 2.5, cy - 5.6, cx + 2.5 + L * np.cos(a), cy - 5.6 + L * np.sin(a), 1.7, 1.5)


def p_P(S):
    # the hand hangs from a wrist at the right: index points left, middle finger points down, thumb touches the middle finger
    cx, cy = 17.5 + S.j(0.7), 13.5 + S.j(0.7)
    S.begin()
    S.cap(cx + 2.0, cy + 1.0, cx + 24.0, cy + 10.0 + S.j(1.5), 6.6, 7.0)                 # forearm from the right
    S.box(cx, cy, 6.8, 5.6, 3.8, 20.0)
    S.end()
    S.cap(cx - 4.0, cy + 0.8, cx - 6.6 + S.j(0.8), cy + 8.0 + S.j(0.8), 1.7, 1.5)        # middle finger down
    S.cap(cx + 0.6, cy + 5.0, cx - 5.0 + S.j(0.7), cy + 4.8 + S.j(0.6), 1.9, 1.6)        # thumb
    a = np.deg2rad(S.j(6.0) + 10.0)
    L = 13.0 * (1 + S.j(0.07))
    S.cap(cx - 3.4, cy - 4.8, cx - 3.4 - L * np.cos(a), cy - 4.8 - L * np.sin(a), 1.75, 1.5)   # index to the left


def p_Q(S):
    # the hand hangs down: back of the hand like a dome on top, index and thumb point down
    cx, cy = 14.0 + S.j(0.7), 10.0 + S.j(0.7)
    S.cap(cx + 2.0, cy + 6.0, cx + 4.0 + S.j(1.0), 42.0, 5.0, 5.4, shade=0.9)             # forearm, behind and below
    S.box(cx, cy, 9.2, 6.8, 6.0)
    a = np.deg2rad(S.j(7.0) - 14.0)
    L = 10.5 * (1 + S.j(0.08))
    S.cap(cx - 6.4, cy + 2.5, cx - 6.4 + L * np.sin(a), cy + 2.5 + L * np.cos(a), 1.9, 1.6)    # index down
    a = np.deg2rad(S.j(7.0) + 14.0)
    L = 7.5 * (1 + S.j(0.08))
    S.cap(cx + 6.2, cy + 3.0, cx + 6.2 + L * np.sin(a), cy + 3.0 + L * np.cos(a), 2.1, 1.8)    # thumb down


def p_X(S):
    cx, top = 13.0 + S.j(0.6), 9.5 + S.j(0.7)
    body(S, cx, top + 7.6, 6.6, 7.6, 4.2, ax=cx - 0.5, ar=5.8)
    strips1(S, [cx - 4.2, cx - 1.0, cx + 2.2], top + 2.2, top + 5.6, 1.6, dy=[0.5, 0.2, 0.0])
    S.cap(cx + 5.8, top + 9.8, cx + 0.4 + S.j(0.6), top + 7.6 + S.j(0.5), 2.0, 1.8)
    hx, hy = cx - 0.4 + S.j(0.6), top - 5.0 + S.j(0.6)
    S.chain([(cx - 0.6, top + 1.8), (hx, hy), (hx + 4.8 + S.j(0.6), hy - 0.2 + S.j(0.5)),
             (hx + 5.8 + S.j(0.6), hy + 2.6 + S.j(0.5))], [1.7, 1.6, 1.5, 1.4])              # the hooked index


# ----------------------------------------------------------------------------------------------------------
# SET 2: the same letters drawn with a tapering wrist, leaning fists for M, N, S, dark creases between folded
# fingers, shadows under fingertips and thumbs, and the dark gaps where thumb and fingers meet
# ----------------------------------------------------------------------------------------------------------
def fist(S, cx, top, hx, hy, rc, tilt=0.0, arm_dx=-0.6, arm_r=None, lean=0.0, thenar=True, wl=None):
    """Fist / palm block with its top edge at y = top, merged into one outline with the wrist and forearm that leave
    through the bottom edge.  The wrist (wl below the top edge) is narrower than the fist: the heel of the hand tapers
    down to it, most of all on the thumb side (right), where the ball of the thumb makes a long slanting edge.
    tilt leans the hand (degrees clockwise) about the wrist; the parts drawn afterwards lean with it until
    S.set_rot(None)."""
    wl = 2.35 * hx if wl is None else wl
    ar = 0.74 * hx if arm_r is None else arm_r
    px, py = cx + arm_dx, top + wl
    S.begin()
    S.cap(px, py - 2.0, px + lean + S.j(0.8), 42.0, ar + S.j(0.3), ar + 0.4)
    S.set_rot(tilt, px, py)
    S.box(cx, top + hy, hx, hy, rc)
    S.cap(cx, top + hy, px, py, min(hx - 0.3, ar + 1.2), ar)                             # middle of the palm
    S.cap(cx - hx + 3.2, top + hy + 1.0, px - ar + 2.8, py, 3.2, 2.8)                    # heel of the hand, little-finger side
    if thenar:
        S.cap(cx + hx - 3.4, top + hy - 1.0, px + ar - 2.8, py, 3.4, 2.8)                # ball of the thumb
    S.end()


def strips(S, xs, y0, y1, r=1.6, dy=None, gap=0.35, shadow=0.9):
    """folded fingers seen from the front: short vertical strips side by side, with dark creases between them
    and a shadow under the fingertips"""
    ends = []
    S.begin()
    for i, x in enumerate(xs):
        o = 0.0 if dy is None else dy[i]
        ends.append((x + S.j(0.2), y1 + o + S.j(0.5)))
        S.cap(x, y0 + o + 0.6, ends[i][0], ends[i][1] + shadow, r + 0.1, r + 0.1)
    S.end(shade=DARK, rim=0.0)
    for i, x in enumerate(xs):
        o = 0.0 if dy is None else dy[i]
        S.cap(x, y0 + o, ends[i][0], ends[i][1], r - gap, r - gap)


def thumb_over(S, x0, y0, x1, y1, r0, r1):
    """a thumb lying over the palm or the fingers, with the shadow it throws below it"""
    S.cap(x0 - 0.2, y0 + 0.7, x1 - 0.2, y1 + 0.7, r0 + 0.2, r1 + 0.2, shade=DARK, rim=0.0)
    S.cap(x0, y0, x1, y1, r0, r1)


def dark(S, x, y, rx, ry, ang=0.0):
    """a dark gap or shadow (between thumb and fingers, inside a bent finger, ...)"""
    S.ell(x + S.j(0.4), y + S.j(0.4), rx, ry, ang, shade=DARK, rim=0.0)


def letter_A(S):
    cx, top = 14.0 + S.j(0.5), 4.5 + S.j(0.6)
    fist(S, cx, top, 7.3, 6.0, 4.5, tilt=S.j(3.0))
    strips(S, [cx - 5.4, cx - 1.85, cx + 1.7, cx + 5.2], top + 1.9, top + 7.4 + S.j(0.6), 1.78, dy=[0.7, 0.1, 0.0, 0.3])
    tx = cx + 7.8 + S.j(0.4)
    S.cap(tx - 1.2, top + 12.5, tx + S.j(0.5), top + 1.5 + S.j(0.9), 2.2, 1.9)           # thumb upright beside the fist
    S.set_rot(None)


def letter_E(S):
    cx, top = 14.5 + S.j(0.5), 3.2 + S.j(0.6)
    fist(S, cx, top, 8.2, 7.2, 6.0, tilt=S.j(3.0), wl=19.5)
    strips(S, [cx - 5.9, cx - 2.0, cx + 1.9, cx + 5.8], top + 2.4, top + 6.6 + S.j(0.5), 1.95, dy=[1.0, 0.1, 0.0, 0.5],
           shadow=1.3)
    ty = top + 11.2 + S.j(0.6)
    thumb_over(S, cx + 7.2, ty + 1.2, cx - 4.6 + S.j(0.8), ty + S.j(0.4), 2.1, 1.9)      # thumb lying under the fingertips
    S.set_rot(None)


def letter_M(S):
    # photographed with the fist leaning over to the thumb side; three fingers lie over the thumb
    cx, top = 11.3 + S.j(0.6), 3.5 + S.j(0.6)
    fist(S, cx, top, 8.4, 6.8, 5.5, tilt=17.0 + S.j(5.0))
    dark(S, cx - 2.0, top + 11.6, 3.4, 1.5)                                              # hollow under the fingers
    S.ell(cx - 4.6 + S.j(0.5), top + 12.0 + S.j(0.5), 1.8, 1.7)                          # thumb tip peeping out
    strips(S, [cx - 6.1], top + 3.9, top + 8.2, 1.9)
    strips(S, [cx - 2.2, cx + 1.8, cx + 5.8], top + 2.3, top + 10.2 + S.j(0.7), 1.95, dy=[0.8, 0.2, 0.0])
    S.set_rot(None)


def letter_N(S):
    # like M but leaning further, two fingers over the thumb
    cx, top = 9.2 + S.j(0.6), 3.0 + S.j(0.6)
    fist(S, cx, top, 8.4, 6.8, 5.5, tilt=27.0 + S.j(5.0))
    dark(S, cx + 0.5, top + 11.6, 3.2, 1.5)                                              # hollow under the fingers
    S.ell(cx - 0.6 + S.j(0.5), top + 11.8 + S.j(0.5), 1.8, 1.7)                          # thumb tip between middle and ring
    strips(S, [cx - 6.1, cx - 2.2], top + 3.7, top + 7.8, 1.9, dy=[0.8, 0.0])
    strips(S, [cx + 1.8, cx + 5.8], top + 2.3, top + 10.2 + S.j(0.7), 1.95, dy=[0.2, 0.0])
    S.set_rot(None)


def letter_S(S):
    # fist leaning to the thumb side, thumb wrapped across the front of the fingers
    cx, top = 10.6 + S.j(0.6), 3.3 + S.j(0.6)
    fist(S, cx, top, 8.3, 6.6, 5.0, tilt=20.0 + S.j(5.0))
    strips(S, [cx - 6.0, cx - 2.05, cx + 1.9, cx + 5.8], top + 2.3, top + 7.2 + S.j(0.5), 1.95, dy=[1.2, 0.6, 0.2, 0.0])
    ty = top + 10.0 + S.j(0.6)
    thumb_over(S, cx + 7.8, ty + 3.4, cx - 3.4 + S.j(0.8), ty + S.j(0.5), 2.3, 2.0)      # thumb across the fingers
    S.set_rot(None)


def letter_T(S):
    # as photographed: the fist is turned side on, the folded fingers are stacked, the bent index juts out sideways
    # at the top and the thumb tip stands up beside it
    cx, top = 12.0 + S.j(0.6), 9.0 + S.j(0.7)
    fist(S, cx, top, 6.3, 6.6, 3.8, arm_dx=-1.0, lean=-1.5, thenar=False, wl=15.5)
    S.begin()
    for k, yy in enumerate([5.6, 8.6, 11.4]):
        S.cap(cx + 0.5, top + yy + 0.5, cx + 5.6 - 0.5 * k, top + yy + 0.8, 1.6, 1.6)
    S.end(shade=DARK, rim=0.0)
    for k, yy in enumerate([5.6, 8.6, 11.4]):
        S.cap(cx + 0.5, top + yy, cx + 5.4 - 0.5 * k + S.j(0.4), top + yy + 0.3 + S.j(0.3), 1.25, 1.25)
    S.cap(cx + 1.0, top + 2.2, cx + 12.5 + S.j(0.9), top + 1.6 + S.j(0.9), 2.1, 1.7)
    S.cap(cx + 1.0, top + 1.5, cx + 1.6 + S.j(0.6), top - 5.2 + S.j(0.7), 1.9, 1.6)
    S.set_rot(None)


def letter_D(S):
    cx, top = 15.0 + S.j(0.5), 9.2 + S.j(0.7)
    fist(S, cx, top, 6.5, 5.2, 4.6)
    strips(S, [cx - 4.6, cx - 1.5, cx + 1.6], top + 1.7, top + 3.6, 1.62, dy=[0.7, 0.1, 0.0], shadow=0.5)
    dark(S, cx - 1.0, top + 6.0, 2.6, 1.2)                                               # the ring of the D, seen as a dark gap
    thumb_over(S, cx + 5.4, top + 10.4, cx + 0.2 + S.j(0.6), top + 8.0 + S.j(0.5), 1.9, 1.7)
    S.finger(cx + 4.9, top + 1.6, 2.0, 9.2, 1.55)
    S.set_rot(None)


def two_up_body(S, cx, top):
    """body shared by R, U, V: ring and little finger folded, the thumb holding them down, a dark gap between"""
    fist(S, cx, top, 6.2, 5.0, 4.2)
    strips(S, [cx - 4.4, cx - 1.5], top + 1.4, top + 3.6, 1.6, dy=[0.4, 0.0], shadow=0.5)
    dark(S, cx - 1.6, top + 6.0, 1.8, 1.1)
    thumb_over(S, cx + 5.2, top + 9.8, cx - 0.4 + S.j(0.6), top + 7.6 + S.j(0.5), 1.9, 1.7)


def letter_R(S):
    cx, top = 14.2 + S.j(0.5), 10.2 + S.j(0.7)
    two_up_body(S, cx, top)
    a = S.j(4.0) + 2.0
    S.finger(cx + 2.0, top + 1.5, 8.0 + a, 9.8, 1.5, sd_ang=2.0)
    S.finger(cx + 4.3, top + 1.5, -7.0 + a, 9.4, 1.5, sd_ang=2.0)
    S.set_rot(None)


def letter_U(S):
    cx, top = 14.2 + S.j(0.5), 10.2 + S.j(0.7)
    two_up_body(S, cx, top)
    a = S.j(4.0) + 3.0
    S.finger(cx + 1.5, top + 1.5, 0.0 + a, 10.2, 1.6, sd_ang=1.5)
    S.finger(cx + 4.7, top + 1.5, 1.5 + a, 9.6, 1.6, sd_ang=1.5)
    S.set_rot(None)


def letter_V(S):
    cx, top = 14.0 + S.j(0.5), 10.6 + S.j(0.7)
    two_up_body(S, cx, top)
    S.finger(cx + 1.3, top + 1.5, -9.0, 10.4, 1.5, sd_ang=5.0)
    S.finger(cx + 4.5, top + 1.5, 22.0, 10.0, 1.5, sd_ang=5.0)
    S.set_rot(None)


def letter_W(S):
    cx, top = 14.5 + S.j(0.5), 11.3 + S.j(0.7)
    fist(S, cx, top, 6.2, 5.0, 4.0)
    dark(S, cx - 2.2, top + 5.4, 1.7, 1.1)
    thumb_over(S, cx + 4.8, top + 9.4, cx - 2.8 + S.j(0.6), top + 7.2 + S.j(0.5), 1.9, 1.7)   # thumb holding the little finger
    S.finger(cx - 3.1, top + 1.6, -13.0, 9.6, 1.45, sd_ang=5.0)
    S.finger(cx + 0.3, top + 1.2, 2.0, 10.6, 1.5, sd_ang=4.0)
    S.finger(cx + 3.7, top + 1.6, 21.0, 9.9, 1.45, sd_ang=5.0)
    S.set_rot(None)


def letter_K(S):
    cx, top = 13.8 + S.j(0.5), 11.3 + S.j(0.7)
    fist(S, cx, top, 5.8, 5.0, 4.2)
    strips(S, [cx - 3.6, cx - 0.9], top + 2.6, top + 4.6, 1.5, dy=[0.4, 0.0], shadow=0.5)
    dark(S, cx + 0.2, top + 6.6, 1.7, 1.2)
    S.cap(cx + 4.6, top + 8.0, cx + 1.8 + S.j(0.5), top + 2.6 + S.j(0.6), 1.9, 1.5)      # thumb reaching up between the two fingers
    S.finger(cx + 0.6, top + 2.2, -20.0, 7.6, 1.5, sd_ang=7.0, sd_len=0.10)              # middle finger, leaning forward / left
    S.finger(cx + 3.4, top + 1.8, 5.0, 10.6, 1.5, sd_ang=5.0)                           # index up
    S.set_rot(None)


def letter_I(S):
    cx, top = 15.3 + S.j(0.5), 7.8 + S.j(0.7)
    fist(S, cx, top, 7.0, 5.8, 4.8)
    strips(S, [cx - 2.1, cx + 1.3, cx + 4.7], top + 1.9, top + 6.0, 1.72, dy=[0.2, 0.0, 0.4])
    thumb_over(S, cx + 6.4, top + 11.2, cx - 0.4 + S.j(0.7), top + 9.2 + S.j(0.5), 2.0, 1.8)
    S.finger(cx - 5.5, top + 2.8, -4.0, 8.2, 1.35)
    S.set_rot(None)


def letter_Y(S):
    cx, top = 14.0 + S.j(0.5), 8.8 + S.j(0.7)
    fist(S, cx, top, 6.4, 5.4, 4.5)
    strips(S, [cx - 3.0, cx + 0.2, cx + 3.4], top + 1.8, top + 5.6, 1.62, dy=[0.2, 0.0, 0.3])
    S.finger(cx - 5.1, top + 2.8, -24.0, 8.4, 1.35, sd_ang=7.0)
    S.finger(cx + 5.0, top + 8.2, 47.0, 10.4, 2.1, sd_ang=8.0, r1=1.6)                   # thumb out to the side
    S.set_rot(None)


def letter_L(S):
    cx, top = 10.8 + S.j(0.6), 10.3 + S.j(0.7)
    fist(S, cx, top, 6.5, 5.4, 4.2, arm_dx=-1.0, lean=-1.0)
    strips(S, [cx - 4.8, cx - 1.9, cx + 1.0], top + 1.7, top + 5.8, 1.5, dy=[1.0, 0.4, 0.0])
    S.finger(cx + 4.6, top + 9.0, 64.0, 11.0, 2.3, sd_ang=8.0, r1=1.6)                   # thumb
    S.finger(cx + 4.2, top + 1.5, 3.0, 10.0, 1.5, sd_ang=5.0)                           # index
    S.set_rot(None)


def letter_F(S):
    cx, top = 12.5 + S.j(0.6), 11.8 + S.j(0.7)
    fist(S, cx, top, 5.8, 5.2, 3.5, arm_dx=-0.8, thenar=False)
    S.finger(cx - 4.2, top + 1.8, -22.0, 8.8, 1.3, sd_ang=6.0)
    S.finger(cx - 1.4, top + 1.0, -4.0, 10.8, 1.45, sd_ang=5.0)
    S.finger(cx + 1.6, top + 1.0, 14.0, 11.0, 1.5, sd_ang=5.0)
    mx, my = cx + 10.0 + S.j(0.7), top + 2.5 + S.j(0.7)                                  # where index tip and thumb tip meet
    S.chain([(cx + 3.8, top + 2.0), (cx + 8.3 + S.j(0.5), top - 0.8 + S.j(0.5)), (mx, my)], [1.7, 1.55, 1.3])
    S.chain([(cx + 3.8, top + 9.0), (cx + 8.6 + S.j(0.5), top + 6.6 + S.j(0.5)), (mx, my)], [2.2, 1.9, 1.4])
    S.ell(cx + 6.6, top + 3.2 + S.j(0.4), 2.1 + S.j(0.3), 1.5 + S.j(0.2), -15.0, shade=None)
    S.set_rot(None)


def letter_B(S):
    cx, top = 15.0 + S.j(0.5), 12.3 + S.j(0.7)
    fist(S, cx, top, 6.1, 5.6, 3.0, arm_r=4.9)
    a = S.j(3.0)
    S.begin()                                                                            # dark lines between the fingers
    S.cap(cx - 3.0, top + 0.5, cx - 3.0 + 9.0 * np.sin(np.deg2rad(a)), top - 7.0, 1.0, 0.8)
    S.cap(cx + 0.0, top + 0.5, cx + 0.0 + 9.0 * np.sin(np.deg2rad(a)), top - 9.0, 1.0, 0.8)
    S.cap(cx + 3.0, top + 0.5, cx + 3.0 + 9.0 * np.sin(np.deg2rad(a)), top - 8.5, 1.0, 0.8)
    S.end(shade=DARK, rim=0.0)
    S.finger(cx - 4.5, top + 1.0, -2.0 + a, 9.0, 1.3, sd_ang=1.0)
    S.finger(cx - 1.5, top + 0.4, -1.0 + a, 10.8, 1.36, sd_ang=1.0)
    S.finger(cx + 1.5, top + 0.3, 0.0 + a, 11.5, 1.36, sd_ang=1.0)
    S.finger(cx + 4.5, top + 0.6, 1.0 + a, 10.5, 1.36, sd_ang=1.0)
    thumb_over(S, cx + 5.6, top + 9.0, cx + 0.6 + S.j(0.7), top + 4.8 + S.j(0.6), 2.0, 1.7)   # thumb folded over the palm
    S.set_rot(None)


def letter_C(S):
    # seen from the side: the back of the hand rises on the left, the fingers arch over to the right, the thumb reaches out below
    cx, top = 11.0 + S.j(0.6), 5.0 + S.j(0.7)
    ox, oy = S.j(0.9), S.j(0.8)
    tx, ty = S.j(0.9), S.j(0.9)
    S.begin()
    S.cap(cx - 2.0, top + 19.0, cx - 6.0 + S.j(1.0), 42.0, 6.0 + S.j(0.3), 6.4)
    S.cap(cx - 1.5, top + 17.0, cx, top + 6.0, 5.5, 5.0)
    S.chain([(cx, top + 6.0), (cx + 1.5, top + 1.6), (cx + 6.0 + 0.5 * ox, top + 0.2 + 0.5 * oy),
             (cx + 11.0 + ox, top + 0.8 + oy)], [5.0, 3.7, 3.1, 2.4])
    S.end()
    S.chain([(cx + 1.5, top + 15.0), (cx + 7.0 + 0.5 * tx, top + 12.8 + 0.5 * ty), (cx + 12.4 + tx, top + 9.6 + ty)],
            [3.0, 2.3, 1.8])


def letter_O(S):
    # seen from the side like C, but the fingertips come down onto the thumb and close a ring
    cx, top = 10.8 + S.j(0.6), 4.8 + S.j(0.7)
    mx, my = cx + 10.2 + S.j(0.7), top + 7.0 + S.j(0.7)                                  # fingertips meet the thumb here
    S.begin()
    S.cap(cx + 0.7, top + 18.0, cx - 0.3 + S.j(1.0), 42.0, 6.2 + S.j(0.3), 6.6)
    S.cap(cx + 0.2, top + 17.2, cx - 0.3, top + 7.2, 5.6, 5.0)
    S.chain([(cx - 0.3, top + 7.2), (cx + 0.7, top + 2.2), (cx + 5.2, top + 0.0), (cx + 9.7 + S.j(0.5), top + 2.2 + S.j(0.5)),
             (mx, my)], [5.0, 3.6, 3.0, 2.6, 2.1])
    S.end()
    S.chain([(cx + 2.2, top + 15.7), (cx + 7.2 + S.j(0.5), top + 11.7 + S.j(0.5)), (mx, my)], [3.2, 2.5, 1.9])
    S.ell(cx + 5.7 + S.j(0.5), top + 5.8 + S.j(0.5), 2.6 + S.j(0.3), 2.1 + S.j(0.3), -20.0, shade=None)


def side_fingers(S, cx, cy, ys, x1):
    """folded fingers seen side on (G, H): stacked horizontal strips with dark creases"""
    S.begin()
    for k, yy in enumerate(ys):
        S.cap(cx + 1.5, cy + yy + 0.5, x1 - 0.6 * k + 0.5, cy + yy + 0.6, 1.6, 1.6)
    S.end(shade=DARK, rim=0.0)
    for k, yy in enumerate(ys):
        S.cap(cx + 1.5, cy + yy, x1 - 0.6 * k + S.j(0.4), cy + yy + S.j(0.3), 1.25, 1.25)


def letter_G(S):
    cx, cy = 9.0 + S.j(0.7), 15.0 + S.j(0.7)
    S.begin()
    S.cap(cx - 2.0, cy + 0.5, cx - 24.0, cy + 7.0 + S.j(1.5), 6.2, 6.6)                  # forearm from the left
    S.box(cx, cy, 6.8, 7.0, 4.0)
    S.end()
    side_fingers(S, cx, cy, [0.5, 3.2, 5.6], cx + 6.2)
    S.cap(cx + 1.0, cy - 2.6, cx + 7.5 + S.j(0.7), cy - 2.4 + S.j(0.4), 1.7, 1.5)        # thumb lying along the index
    a = np.deg2rad(S.j(6.0) - 4.0)
    L = 14.0 * (1 + S.j(0.07))
    S.cap(cx + 2.5, cy - 5.3, cx + 2.5 + L * np.cos(a), cy - 5.3 + L * np.sin(a), 1.75, 1.55)


def letter_H(S):
    cx, cy = 9.0 + S.j(0.7), 15.0 + S.j(0.7)
    S.begin()
    S.cap(cx - 2.0, cy + 0.5, cx - 24.0, cy + 7.0 + S.j(1.5), 6.2, 6.6)
    S.box(cx, cy, 6.8, 7.0, 4.0)
    S.end()
    side_fingers(S, cx, cy, [0.9, 3.8], cx + 6.0)
    a = np.deg2rad(S.j(6.0) - 4.0)
    L = 14.0 * (1 + S.j(0.07))
    S.cap(cx + 2.5, cy - 2.5, cx + 2.5 + 0.97 * L * np.cos(a), cy - 2.5 + 0.97 * L * np.sin(a), 1.7, 1.5)
    S.cap(cx + 2.5, cy - 5.6, cx + 2.5 + L * np.cos(a), cy - 5.6 + L * np.sin(a), 1.7, 1.5)


def letter_P(S):
    # the hand hangs from a wrist at the right: index points left, middle finger points down, thumb touches the middle finger
    cx, cy = 17.5 + S.j(0.7), 13.0 + S.j(0.7)
    S.begin()
    S.cap(cx + 2.0, cy + 1.0, cx + 24.0, cy + 10.0 + S.j(1.5), 6.8, 7.2)                 # forearm from the right
    S.box(cx, cy, 7.0, 6.0, 4.0, 20.0)
    S.end()
    dark(S, cx - 2.2, cy + 1.6, 1.9, 1.6)                                                # gap between thumb and fingers
    S.cap(cx - 4.0, cy + 1.0, cx - 6.4 + S.j(0.8), cy + 9.0 + S.j(0.8), 1.8, 1.5)        # middle finger down
    S.cap(cx + 0.6, cy + 5.4, cx - 5.0 + S.j(0.7), cy + 5.2 + S.j(0.6), 1.9, 1.6)        # thumb
    a = np.deg2rad(S.j(9.0) + 12.0)
    L = 13.0 * (1 + S.j(0.07))
    S.cap(cx - 3.4, cy - 5.0, cx - 3.4 - L * np.cos(a), cy - 5.0 - L * np.sin(a), 1.75, 1.5)   # index to the left


def letter_Q(S):
    # the hand hangs down from a raised wrist: the back of the hand is a wide dome on top, the index (long, on the
    # left) and the thumb (shorter, on the right) point down like two legs, the forearm shows below between them
    cx, cy = 13.5 + S.j(0.7), 9.8 + S.j(0.7)
    S.cap(cx + 1.0, cy + 5.0, cx + 1.0 + S.j(1.5), 42.0, 5.4 + S.j(0.4), 5.8, shade=0.92)   # forearm, behind and below
    S.box(cx, cy, 9.6, 6.6, 6.2)
    dark(S, cx + 1.0, cy + 4.6, 2.2, 1.4)                                                # shadow in the crook of thumb and index
    a = np.deg2rad(S.j(7.0) - 15.0)
    L = 11.0 * (1 + S.j(0.08))
    S.cap(cx - 6.6, cy + 2.0, cx - 6.6 + L * np.sin(a), cy + 2.0 + L * np.cos(a), 2.1, 1.7)    # index down
    a = np.deg2rad(S.j(7.0) + 12.0)
    L = 7.5 * (1 + S.j(0.08))
    S.cap(cx + 6.4, cy + 3.0, cx + 6.4 + L * np.sin(a), cy + 3.0 + L * np.cos(a), 2.2, 1.9)    # thumb down


def letter_X(S):
    cx, top = 13.0 + S.j(0.6), 9.5 + S.j(0.7)
    fist(S, cx, top, 6.6, 5.6, 4.2)
    strips(S, [cx - 4.2, cx - 1.0, cx + 2.2], top + 2.2, top + 5.6, 1.62, dy=[0.5, 0.2, 0.0])
    thumb_over(S, cx + 5.8, top + 10.6, cx + 0.4 + S.j(0.6), top + 8.8 + S.j(0.5), 2.0, 1.8)
    hx, hy = cx - 0.4 + S.j(0.6), top - 4.8 + S.j(0.6)
    S.chain([(cx - 0.6, top + 1.8), (hx, hy), (hx + 4.8 + S.j(0.6), hy - 0.2 + S.j(0.5)),
             (hx + 5.8 + S.j(0.6), hy + 2.8 + S.j(0.5))], [1.9, 1.8, 1.7, 1.5])              # the hooked index
    S.set_rot(None)


NAMES = "ABCDEFGHIKLMNOPQRSTUVWXY"
SET1 = [globals()["p_" + c] for c in NAMES]          # plain drawings, part outlines darkened by 0.30
SET2 = [globals()["letter_" + c] for c in NAMES]     # shaded drawings, part outlines darkened by 0.20
SETS = {1: (SET1, 0.30), 2: (SET2, 0.20)}


def draw(k, B, rng, jscale=1.0, pose=True, which=1):
    """B random drawings of letter k from drawing set `which`: articulation jitter plus a random affine pose."""
    letters, rim = SETS[which]
    S = Scene(B, rng, jscale, rim)
    letters[k](S)
    if pose and jscale > 0:
        rot = np.deg2rad(np.clip(rng.normal(0, 1, B), -2, 2) * 5.0)
        s = np.exp(np.clip(rng.normal(0, 1, B), -2, 2) * 0.07)
        asp = np.exp(np.clip(rng.normal(0, 1, B), -2, 2) * 0.05)
        sh = np.clip(rng.normal(0, 1, B), -2, 2) * 0.06
        A = np.zeros((B, 2, 2))
        c, s_ = np.cos(rot), np.sin(rot)
        # rotation * shear * anisotropic scale
        A[:, 0, 0] = s * asp * c
        A[:, 0, 1] = s / asp * (c * sh - s_)
        A[:, 1, 0] = s * asp * s_
        A[:, 1, 1] = s / asp * (s_ * sh + c)
        t = np.clip(rng.normal(0, 1, (B, 2)), -2, 2) * np.array([1.6, 1.4])
    else:
        A, t = None, None
    return render(S, A, t)


# ====================================================================================================
# the network
# ====================================================================================================
N_ORI = 6
SIGMA_G = 0.9      # scale of the derivative-of-Gaussian edge filters (pixels)
ALPHA = 1.73       # orientation sharpening: |a_k| - ALPHA * |a_perp|   (zero 30 degrees away from the preferred direction)
T0 = 0.06          # dead zone, in units of "step height / image standard deviation"
CMAX = 0.40        # clip level, same units
SIG_F, K_F, S_F, P_F = 1.5, 8, 2, 3     # fine pooling: Gaussian blur, stride 2  -> 12 x 12 cells
SIG_C, K_C, S_C, P_C = 3.0, 12, 4, 4    # coarse pooling: stride 4               -> 6 x 6 cells
ROW_W_F = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.75, 0.5, 0.3]   # weight of each row of cells (the wrist / sleeve rows count less)
ROW_W_C = [1, 1, 1, 1, 0.75, 0.4]
LAMBDA_C = 0.5     # weight of the coarse correlation in the score
LAMBDA_E = 0.5     # weight of the orientation-free edge-density correlation in the score (second template bank only)

def grad_filters():
    r = np.arange(-2, 3)
    X, Y = np.meshgrid(r, r)
    G = np.exp(-(X ** 2 + Y ** 2) / (2 * SIGMA_G ** 2))
    gx = X * G
    gx = gx / gx[X > 0].sum()          # an ideal unit step edge gives response 1
    gy = gx.T.copy()
    W = np.zeros((2 * N_ORI, 1, 5, 5))
    for k in range(N_ORI):
        th = np.pi * k / N_ORI
        f = np.cos(th) * gx + np.sin(th) * gy
        W[2 * k, 0], W[2 * k + 1, 0] = f, -f
    return W


def blur_kernel(sig, k):
    c = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-c ** 2 / (2 * sig ** 2))
    K = np.outer(g, g)
    return K / K.sum()


def frontend():
    L, Wt = [], {}
    def conv(name, inp, W, b, stride=1, padding=0):
        L.append({"name": name, "op": "conv", "inputs": [inp], "out": int(W.shape[0]), "kernel": [int(W.shape[2]), int(W.shape[3])],
                  "stride": stride, "padding": padding})
        Wt[name + ".W"], Wt[name + ".b"] = W, b
    L.append({"name": "n0", "op": "norm", "inputs": ["image"]})
    conv("g", "n0", grad_filters(), np.zeros(2 * N_ORI))
    L.append({"name": "gr", "op": "relu", "inputs": ["g"]})
    We = np.zeros((N_ORI, 2 * N_ORI, 1, 1))
    for k in range(N_ORI):
        kp = (k + N_ORI // 2) % N_ORI
        We[k, 2 * k, 0, 0] = We[k, 2 * k + 1, 0, 0] = 1.0
        We[k, 2 * kp, 0, 0] = We[k, 2 * kp + 1, 0, 0] = -ALPHA
    conv("e", "gr", We, np.full(N_ORI, -T0))
    L.append({"name": "er", "op": "relu", "inputs": ["e"]})
    I = np.eye(N_ORI).reshape(N_ORI, N_ORI, 1, 1)
    conv("c", "er", I, np.full(N_ORI, -CMAX))
    L.append({"name": "cr", "op": "relu", "inputs": ["c"]})
    conv("cn", "cr", -I, np.zeros(N_ORI))
    L.append({"name": "m", "op": "add", "inputs": ["er", "cn"]})          # = min(er, CMAX)
    for tag, sig, k, s, p, rw in (("f", SIG_F, K_F, S_F, P_F, ROW_W_F), ("c", SIG_C, K_C, S_C, P_C, ROW_W_C)):
        Wb = np.zeros((N_ORI, N_ORI, k, k))
        for o in range(N_ORI):
            Wb[o, o] = blur_kernel(sig, k)
        conv("p" + tag, "m", Wb, np.zeros(N_ORI), s, p)
        n = len(rw)
        d = np.tile(np.repeat(np.asarray(rw, float), n), N_ORI)            # (ori, row, col) flattened
        L.append({"name": "w" + tag, "op": "dense", "inputs": ["p" + tag], "out": int(d.size)})
        Wt["w" + tag + ".W"], Wt["w" + tag + ".b"] = np.diag(d), np.zeros(d.size)
        L.append({"name": "s" + tag, "op": "reshape", "inputs": ["w" + tag], "shape": [1, N_ORI * n, n]})
        L.append({"name": "z" + tag, "op": "norm", "inputs": ["s" + tag]})
    # edge density regardless of orientation ("how busy is this spot"), at the fine resolution
    We_ = np.zeros((1, N_ORI, K_F, K_F))
    We_[0, :] = blur_kernel(SIG_F, K_F)
    conv("pe", "m", We_, np.zeros(1), S_F, P_F)
    n = len(ROW_W_F)
    d = np.repeat(np.asarray(ROW_W_F, float), n)
    L.append({"name": "we", "op": "dense", "inputs": ["pe"], "out": int(d.size)})
    Wt["we.W"], Wt["we.b"] = np.diag(d), np.zeros(d.size)
    L.append({"name": "se", "op": "reshape", "inputs": ["we"], "shape": [1, n, n]})
    L.append({"name": "ze", "op": "norm", "inputs": ["se"]})
    return L, Wt


def conv_np(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    out = np.zeros((n, o, ho, wo))
    for dy in range(kh):
        for dx in range(kw):
            patch = xp[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
            out += np.einsum('nchw,oc->nohw', patch, W[:, :, dy, dx])
    return out + b[None, :, None, None]


def norm_np(x):
    mu = x.mean(axis=(2, 3), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)


def features(images, L=None, Wt=None):
    """images (N, 28, 28) in [0, 1] -> the normalised feature vectors the template layers read (zf, zc, ze).
    A plain numpy copy of what the network's front-end layers compute."""
    if L is None:
        L, Wt = frontend()
    vals = {"image": np.asarray(images, np.float64)[:, None]}
    for lay in L:
        ins = [vals[i] for i in lay["inputs"]]
        op, name = lay["op"], lay["name"]
        if op == "conv":
            W, b = Wt[name + ".W"], Wt[name + ".b"]
            if W.shape[2] == 1 and lay["stride"] == 1:
                vals[name] = np.einsum('nchw,oc->nohw', ins[0], W[:, :, 0, 0]) + b[None, :, None, None]
            else:
                vals[name] = conv_np(ins[0], W, b, lay["stride"], lay["padding"])
        elif op == "dense":
            W = Wt[name + ".W"]
            x = ins[0].reshape(len(ins[0]), -1)
            if W.shape[0] == W.shape[1] and np.count_nonzero(W - np.diag(np.diag(W))) == 0:
                vals[name] = x * np.diag(W)[None] + Wt[name + ".b"]
            else:
                vals[name] = x @ W.T + Wt[name + ".b"]
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op == "add":
            vals[name] = sum(ins)
        elif op == "reshape":
            vals[name] = ins[0].reshape((len(ins[0]),) + tuple(lay["shape"]))
        elif op == "norm":
            vals[name] = norm_np(ins[0])
        elif op == "maxpool":
            x = ins[0]
            (sh, sw), (th, tw) = lay["size"], lay["stride"]
            ho, wo = (x.shape[2] - sh) // th + 1, (x.shape[3] - sw) // tw + 1
            vals[name] = np.stack([x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw] for dy in range(sh) for dx in range(sw)]).max(0)
    n = len(images)
    return vals["zf"].reshape(n, -1), vals["zc"].reshape(n, -1), vals["ze"].reshape(n, -1), vals


N_PER_CLASS = 2000      # drawings (= templates) per letter in each bank
SEED = 20261120
BATCH = 250
BANK2_SCALE = 0.75      # bank 2 has three correlation terms (weights 1 + 0.5 + 0.5), bank 1 two (1 + 0.5): equal ceilings


def bank(which, seed, n_per_class, L, Wt, verbose, t0):
    """n_per_class drawings of every letter from drawing set `which`, each turned into one template"""
    import time
    rng = np.random.default_rng(seed)
    n_cls = len(NAMES)
    out = None
    for k in range(n_cls):
        for s in range(0, n_per_class, BATCH):
            b = min(BATCH, n_per_class - s)
            imgs = draw(k, b, rng, which=which)
            if s == 0:
                imgs[0] = draw(k, 1, rng, jscale=0.0, which=which)[0]      # the first template of a letter is its plain drawing
            z = features(imgs, L, Wt)[:3]
            if out is None:
                out = [np.zeros((n_cls * n_per_class, v.shape[1]), np.float32) for v in z]
            # every drawing becomes one template, normalised by itself alone: each feature vector z comes out of the
            # front-end already zero-mean with unit variance over its own entries; dividing by its length makes the
            # template layer's output the correlation between image and drawing
            for o, v in zip(out, z):
                o[k * n_per_class + s: k * n_per_class + s + b] = v / v.shape[1]
        if verbose:
            print("bank", which, "letter", NAMES[k], "done", round(time.time() - t0, 1), "s", flush=True)
    return out


def build(out_dir, n_per_class=N_PER_CLASS, verbose=False, two_banks=True):
    import os, time
    t0 = time.time()
    L, Wt = frontend()
    n_cls = len(NAMES)
    nt = n_cls * n_per_class
    Wt = dict(Wt)

    def head(tag, terms):
        """template layers of one bank -> the 24 best-match scores"""
        names = []
        for feat, T in terms:
            nm = tag + "_" + feat
            L.append({"name": nm, "op": "dense", "inputs": ["z" + feat], "out": nt})
            Wt[nm + ".W"], Wt[nm + ".b"] = T, np.zeros(nt, np.float32)
            names.append(nm)
        L.append({"name": tag, "op": "add", "inputs": names})
        L.append({"name": tag + "_r", "op": "reshape", "inputs": [tag], "shape": [1, n_cls, n_per_class]})
        L.append({"name": tag + "_best", "op": "maxpool", "inputs": [tag + "_r"], "size": [1, n_per_class], "stride": [1, n_per_class]})
        L.append({"name": tag + "_s", "op": "reshape", "inputs": [tag + "_best"], "shape": [n_cls]})
        return tag + "_s"

    f1, c1, _ = bank(1, SEED, n_per_class, L, Wt, verbose, t0)
    s1 = head("a", [("f", f1), ("c", LAMBDA_C * c1)])
    if two_banks:
        f2, c2, e2 = bank(2, SEED + 1, n_per_class, L, Wt, verbose, t0)
        s2 = head("b", [("f", BANK2_SCALE * f2), ("c", BANK2_SCALE * LAMBDA_C * c2), ("e", BANK2_SCALE * LAMBDA_E * e2)])
        L.append({"name": "scores", "op": "add", "inputs": [s1, s2]})
    else:
        # only bank 1 reads the features: drop the unused edge-density layers
        used = {"image"}
        keep = []
        for lay in reversed(L):
            if lay["name"] == s1 or lay["name"] in used:
                keep.append(lay)
                used.update(lay["inputs"])
        L[:] = keep[::-1]
        names = {l["name"] for l in L}
        Wt = {k: v for k, v in Wt.items() if k.rsplit(".", 1)[0] in names}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": L}, f, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: np.asarray(v, np.float32) for k, v in Wt.items()})
    if verbose:
        print("wrote", out_dir, "in", round(time.time() - t0, 1), "s")


TWO_BANKS = True

if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else ".", verbose=len(sys.argv) > 2, two_banks=TWO_BANKS)
