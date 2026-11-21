#!/usr/bin/env python3
"""Build a data-free, analytically specified ASL shape-matching network."""
import json
import math
import sys
from pathlib import Path
import numpy as np

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
K = 23


def derivative_filters():
    yy, xx = np.mgrid[-2:3, -2:3].astype(np.float64)
    g = np.exp(-(xx * xx + yy * yy) / (2 * 1.05 ** 2))
    fs = []
    for a in (0, 45, 90, 135):
        t = math.radians(a)
        f = (xx * math.cos(t) + yy * math.sin(t)) * g
        f -= f.mean()
        f /= np.sqrt(np.sum(f * f)) + 1e-12
        fs.append(f)
    return np.asarray(fs)


class Hand:
    """Small analytic rasterizer. Coordinates are independent of output pixels."""
    def __init__(self, size, angle=0.0, scale=1.0, aspect=1.0, extent=1.16):
        self.size = size
        q = extent
        y, x = np.mgrid[0:size, 0:size].astype(np.float64)
        x = (x + .5) / size * (2 * q) - q
        y = (y + .5) / size * (2 * q) - q
        a = math.radians(angle)
        ca, sa = math.cos(a), math.sin(a)
        # Inverse global transform of the query grid.
        xr = ca * x + sa * y
        yr = -sa * x + ca * y
        self.x = xr / (scale * aspect)
        self.y = yr / scale
        self.m = np.zeros((size, size), np.float64)
        self.g = np.zeros((size, size), np.float64)
        self.holes = np.zeros((size, size), np.float64)

    def ellipse_field(self, cx, cy, rx, ry, ang=0):
        a = math.radians(ang); ca, sa = math.cos(a), math.sin(a)
        x, y = self.x - cx, self.y - cy
        u, v = ca * x + sa * y, -sa * x + ca * y
        return ((u / rx) ** 2 + (v / ry) ** 2 <= 1).astype(np.float64)

    def ellipse(self, cx, cy, rx, ry, ang=0):
        self.m = np.maximum(self.m, self.ellipse_field(cx, cy, rx, ry, ang))

    def hole(self, cx, cy, rx, ry, ang=0):
        self.holes = np.maximum(self.holes, self.ellipse_field(cx, cy, rx, ry, ang))

    def capsule_field(self, x1, y1, x2, y2, r):
        vx, vy = x2 - x1, y2 - y1
        d = vx * vx + vy * vy + 1e-12
        t = np.clip(((self.x - x1) * vx + (self.y - y1) * vy) / d, 0, 1)
        dx, dy = self.x - (x1 + t * vx), self.y - (y1 + t * vy)
        return (dx * dx + dy * dy <= r * r).astype(np.float64)

    def cap(self, x1, y1, x2, y2, r):
        self.m = np.maximum(self.m, self.capsule_field(x1, y1, x2, y2, r))

    def groove(self, x1, y1, x2, y2, r=.025):
        self.g = np.maximum(self.g, self.capsule_field(x1, y1, x2, y2, r))

    def path(self, pts, r):
        for a, b in zip(pts[:-1], pts[1:]):
            self.cap(a[0], a[1], b[0], b[1], r)

    def groove_path(self, pts, r=.025):
        for a, b in zip(pts[:-1], pts[1:]):
            self.groove(a[0], a[1], b[0], b[1], r)

    def finish(self):
        m = self.m * (1 - self.holes)
        # Creases are dark marks within a lighter generic hand.
        return np.clip(m * (1 - .82 * self.g), 0, 1)


def wrist(h, x=0.0, slant=0.0, width=.24):
    h.cap(x, .54, x + slant, 1.28, width)


def palm(h, cx=0, cy=.30, rx=.40, ry=.51, wrist_x=0, slant=0):
    wrist(h, wrist_x, slant)
    h.ellipse(cx, cy, rx, ry)


def fist(h, kind=0):
    palm(h, 0, .28, .43, .48)
    # Rounded row of curled fingers and the separators visible on their fronts.
    for x, yy in zip((-.29, -.10, .09, .27), (-.06, -.10, -.10, -.05)):
        h.ellipse(x, yy, .145, .205, -5)
    for x in (-.20, -.01, .18):
        h.groove(x, -.22, x - .015, .12, .020)
    h.groove_path([(-.34, .12), (-.12, .18), (.12, .17), (.34, .10)], .021)


def add_finger(h, x0, y0, xt, yt, r=.105):
    h.cap(x0, y0, xt, yt, r)


def ring(h, cx, cy, rx, ry, r=.09, start=0, stop=360):
    ts = np.linspace(math.radians(start), math.radians(stop), 13)
    pts = [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in ts]
    h.path(pts, r)


def draw_sign(c, size, angle=0, scale=1, aspect=1, mirror=False):
    h = Hand(size, angle, scale, -aspect if mirror else aspect)
    if c == 0:                         # A: fist, thumb straight alongside
        fist(h)
        h.cap(.31, .40, .34, -.08, .115)
        h.groove_path([(.27, .43), (.23, .18), (.26, -.10)], .026)
    elif c == 1:                       # B: four straight fingers, folded thumb
        palm(h, 0, .34, .39, .48)
        xs = (-.29, -.10, .09, .27); tips = (-.59, -.76, -.78, -.66)
        for x, y in zip(xs, tips): add_finger(h, x, .16, x, y, .105)
        for x, y in zip((-.195, -.005, .18), (-.58, -.70, -.61)):
            h.groove(x, .12, x, y, .023)
        h.groove_path([(.36, .43), (.18, .27), (-.08, .24)], .032)
    elif c == 2:                       # C: curved fingers oppose the thumb
        palm(h, -.25, .35, .25, .53, wrist_x=-.18, slant=-.08)
        h.path([(-.28, .10), (-.42, -.28), (-.29, -.57), (.02, -.70), (.34, -.62), (.49, -.42)], .14)
        h.path([(-.22, .47), (.01, .27), (.29, .24), (.49, .32)], .105)
        h.groove_path([(-.31, -.02), (-.36, -.31), (-.20, -.52), (.12, -.60), (.38, -.48)], .022)
    elif c == 3:                       # D: index up, other fingers and thumb form ring
        palm(h, -.02, .34, .38, .48)
        add_finger(h, .06, .10, .06, -.90, .105)
        for x in (-.26, -.10, .22): h.ellipse(x, .00, .14, .18)
        ring(h, .08, .15, .27, .25, .09)
        h.hole(.08, .15, .14, .14)
        h.groove(.18, .10, .29, -.02, .022)
    elif c == 4:                       # E: all fingers curled down over thumb
        fist(h)
        for x in (-.27, -.09, .09, .27):
            h.groove(x, -.16, x - .04, .24, .027)
        h.groove_path([(-.34, .24), (-.10, .30), (.14, .27), (.35, .15)], .030)
        h.groove_path([(.34, .42), (.12, .33), (-.12, .29)], .025)
    elif c == 5:                       # F: three fingers up, index-thumb circle
        palm(h, -.05, .38, .39, .48)
        for x0, xt, yt in ((-.27, -.46, -.62), (-.10, -.18, -.82), (.07, .11, -.75)):
            add_finger(h, x0, .14, xt, yt, .095)
        ring(h, .31, .08, .25, .27, .09)
        h.hole(.31, .08, .12, .14)
        h.groove(-.18, .11, -.30, -.52, .020)
        h.groove(-.02, .10, .04, -.64, .020)
    elif c == 6:                       # G: horizontal index, thumb parallel below
        palm(h, -.30, .29, .35, .38, wrist_x=-.27, slant=-.17)
        h.cap(-.22, -.01, .86, -.28, .105)
        h.cap(-.03, .22, .57, .17, .105)
        h.groove(-.17, .05, .47, -.04, .025)
    elif c == 7:                       # H: two horizontal fingers
        palm(h, -.32, .30, .34, .38, wrist_x=-.28, slant=-.18)
        h.cap(-.22, -.06, .84, -.30, .095)
        h.cap(-.17, .10, .76, -.09, .095)
        h.groove(-.18, .02, .70, -.19, .023)
        h.groove(-.26, .23, .12, .18, .025)
    elif c == 8:                       # I: pinky up
        fist(h)
        h.cap(-.28, -.02, -.38, -.88, .095)
        h.groove(-.19, -.12, -.27, -.66, .022)
        h.groove_path([(.34, .39), (.10, .25), (-.10, .22)], .025)
    elif c == 9:                       # K: index and middle spread, thumb between
        palm(h, -.02, .37, .38, .46)
        h.cap(-.15, .08, -.35, -.83, .10)
        h.cap(.08, .08, .31, -.76, .10)
        h.cap(.27, .39, .06, -.14, .105)
        h.groove(-.04, .08, -.24, -.70, .021)
        h.groove(.09, .08, .24, -.62, .021)
    elif c == 10:                      # L: index vertical and thumb horizontal
        fist(h)
        h.cap(.02, -.02, .02, -.91, .105)
        h.cap(.27, .28, .83, .05, .105)
        h.groove(.12, -.02, .13, -.66, .022)
    elif c == 11:                      # M: thumb tucked under three fingers
        fist(h)
        h.groove_path([(.34, .34), (.13, .16), (-.21, .12)], .035)
        for x in (-.21, -.02, .17): h.groove(x, -.15, x - .08, .18, .025)
    elif c == 12:                      # N: thumb tucked under two fingers
        fist(h)
        h.groove_path([(.34, .35), (.12, .13), (-.05, .10)], .035)
        h.groove(-.17, -.16, -.23, .18, .026)
        h.groove(.04, -.17, -.03, .15, .026)
        h.groove(.21, -.14, .26, .09, .022)
    elif c == 13:                      # O: all fingertips meet thumb in oval
        palm(h, -.08, .43, .33, .43)
        ring(h, .03, -.05, .39, .47, .13)
        h.hole(.03, -.05, .20, .27)
        h.groove_path([(-.28, -.34), (-.03, -.48), (.22, -.39)], .022)
    elif c == 14:                      # P: K hand rotated downward
        h.ellipse(.18, -.03, .39, .37, -20)
        h.cap(.30, .17, .67, .91, .24)
        h.cap(-.02, -.04, -.78, -.28, .105)
        h.cap(.04, .11, -.52, .42, .105)
        h.cap(.30, .09, -.14, .09, .10)
        h.groove(-.04, .03, -.57, -.15, .022)
    elif c == 15:                      # Q: G hand directed down
        h.ellipse(.18, -.10, .40, .38, -15)
        h.cap(.33, .12, .65, .92, .24)
        h.cap(-.02, .02, -.48, .78, .105)
        h.cap(-.03, .12, -.63, .46, .10)
        h.groove(.02, .13, -.39, .63, .023)
    elif c == 16:                      # R: two adjacent fingers twisted together
        palm(h, 0, .38, .38, .46)
        h.path([(-.11, .10), (-.14, -.32), (.05, -.83)], .10)
        h.path([(.11, .10), (.13, -.32), (-.03, -.84)], .10)
        h.groove_path([(-.03, .05), (-.05, -.30), (.06, -.72)], .025)
        h.groove_path([(.10, .08), (.07, -.31), (-.02, -.73)], .020)
    elif c == 17:                      # S: fist with thumb across front
        fist(h)
        h.cap(.31, .30, -.12, .08, .115)
        h.groove_path([(.35, .38), (.14, .18), (-.15, .08)], .034)
        h.groove_path([(-.30, .18), (-.05, .25), (.19, .21)], .024)
    elif c == 18:                      # T: thumb emerges between index and middle
        fist(h)
        h.cap(.27, .38, .02, -.07, .105)
        h.groove_path([(.31, .40), (.10, .12), (.02, -.10)], .028)
        h.groove(-.08, -.18, -.10, .13, .025)
        h.groove(.15, -.17, .13, .07, .025)
    elif c == 19:                      # U: two long parallel fingers together
        palm(h, 0, .38, .37, .46)
        h.cap(-.09, .09, -.09, -.91, .095)
        h.cap(.09, .09, .09, -.89, .095)
        h.groove(0, .06, 0, -.77, .024)
        h.groove_path([(.31, .37), (.14, .19), (-.14, .16)], .025)
    elif c == 20:                      # V: two long fingers widely spread
        palm(h, 0, .38, .37, .46)
        h.cap(-.10, .10, -.43, -.80, .095)
        h.cap(.10, .09, .42, -.80, .095)
        h.groove(-.01, .08, -.32, -.69, .021)
        h.groove(.02, .08, .31, -.69, .021)
    elif c == 21:                      # W: three clearly separated fingers
        palm(h, 0, .39, .38, .46)
        h.cap(-.20, .12, -.50, -.69, .09)
        h.cap(-.02, .08, -.05, -.91, .09)
        h.cap(.16, .12, .43, -.72, .09)
        h.groove(-.11, .09, -.40, -.59, .020)
        h.groove(.07, .08, .33, -.62, .020)
    elif c == 22:                      # X: hooked index
        fist(h)
        h.path([(.02, -.03), (.03, -.58), (.30, -.64), (.40, -.47), (.36, -.29)], .105)
        h.groove_path([(.13, -.03), (.14, -.46), (.29, -.52)], .022)
    elif c == 23:                      # Y: pinky and thumb spread
        fist(h)
        h.cap(-.27, -.01, -.43, -.82, .10)
        h.cap(.28, .29, .87, .07, .105)
        h.groove(-.18, -.06, -.34, -.69, .021)
        h.groove(.16, .23, .69, .06, .021)
    return h.finish()


def conv_same(a, f):
    k = f.shape[0]; p = k // 2
    z = np.pad(a, p)
    out = np.zeros_like(a, dtype=np.float64)
    for iy in range(k):
        for ix in range(k):
            out += f[iy, ix] * z[iy:iy+a.shape[0], ix:ix+a.shape[1]]
    return out


def standard_kernel(a):
    a = a.astype(np.float64)
    a -= a.mean(axis=(-2, -1), keepdims=True)
    n = np.sqrt(np.sum(a * a)) + 1e-12
    return a / n


def build(dst):
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    fs = derivative_filters()
    # Broad, deterministic nuisance coverage: crop scale, slight camera rotation, hand aspect.
    transforms = []
    for sc in (.89, 1.0, 1.10):
        for ang in (-5.0, 0.0, 5.0): transforms.append((ang, sc, 1.0, False))
    transforms += [(0, .96, .88, False), (0, .96, 1.12, False),
                   (-3, 1.04, .91, False), (3, 1.04, 1.09, False)]

    edge_templates, edge_owner = [], []
    region_templates, region_owner = [], []
    for c in range(24):
        for ang, sc, asp, mirror in transforms:
            im = draw_sign(c, K, ang, sc, asp, mirror)
            ef = np.stack([np.abs(conv_same(im, f)) for f in fs])
            edge_templates.append(standard_kernel(ef)); edge_owner.append(c)

            # The low-frequency branch sees a 2x2 averaged 22-pixel rendering.
            hi = draw_sign(c, 22, ang, sc, asp, mirror)
            lo = hi.reshape(11, 2, 11, 2).mean((1, 3))
            r = standard_kernel(lo[None])[0]
            region_templates.extend([r, -r]); region_owner.extend([c, c])

    ET = np.asarray(edge_templates, np.float32)
    RT = np.asarray(region_templates, np.float32)[:, None]
    ne, nr = len(ET), len(RT)

    layers = [
        {"name":"inorm", "op":"norm", "inputs":["image"]},
        {"name":"edge", "op":"conv", "inputs":["inorm"], "out":8, "kernel":[5,5], "stride":1, "padding":2},
        {"name":"edge_relu", "op":"relu", "inputs":["edge"]},
        {"name":"edge_abs", "op":"conv", "inputs":["edge_relu"], "out":4, "kernel":[1,1], "stride":1, "padding":0},
        {"name":"edge_norm", "op":"norm", "inputs":["edge_abs"]},
        {"name":"edge_match", "op":"conv", "inputs":["edge_norm"], "out":ne, "kernel":[23,23], "stride":1, "padding":3},
        {"name":"edge_hit", "op":"relu", "inputs":["edge_match"]},
        {"name":"edge_best", "op":"maxpool", "inputs":["edge_hit"], "size":[12,12], "stride":[12,12]},
        {"name":"edge_scores", "op":"dense", "inputs":["edge_best"], "out":24},
        {"name":"coarse", "op":"avgpool", "inputs":["inorm"], "size":[2,2], "stride":[2,2]},
        {"name":"region_match", "op":"conv", "inputs":["coarse"], "out":nr, "kernel":[11,11], "stride":1, "padding":2},
        {"name":"region_hit", "op":"relu", "inputs":["region_match"]},
        {"name":"region_best", "op":"maxpool", "inputs":["region_hit"], "size":[8,8], "stride":[8,8]},
        {"name":"region_scores", "op":"dense", "inputs":["region_best"], "out":24},
        {"name":"scores", "op":"add", "inputs":["edge_scores", "region_scores"]}
    ]

    W = {}
    ew = np.zeros((8,1,5,5), np.float32)
    for j in range(4):
        ew[2*j,0] = fs[j]; ew[2*j+1,0] = -fs[j]
    W["edge.W"] = ew; W["edge.b"] = np.zeros(8, np.float32)
    aw = np.zeros((4,8,1,1), np.float32)
    for j in range(4): aw[j,2*j:2*j+2,0,0] = 1
    W["edge_abs.W"] = aw; W["edge_abs.b"] = np.zeros(4, np.float32)
    W["edge_match.W"] = ET
    W["edge_match.b"] = np.full(ne, -2.35, np.float32)
    ce = np.zeros((24,ne), np.float32)
    for j,c in enumerate(edge_owner): ce[c,j] = 1.0 / len(transforms)
    W["edge_scores.W"] = ce; W["edge_scores.b"] = np.zeros(24, np.float32)
    W["region_match.W"] = RT
    W["region_match.b"] = np.full(nr, -1.75, np.float32)
    cr = np.zeros((24,nr), np.float32)
    for j,c in enumerate(region_owner): cr[c,j] = .62 / (2 * len(transforms))
    W["region_scores.W"] = cr
    # Equalization constants chosen from general template geometry: very compact
    # downward poses otherwise produce more accidental background correlation.
    # They are fixed semantic-group corrections, not image statistics.
    rb = np.zeros(24, np.float32)
    rb[[15]] = -.18       # downward Q
    rb[[22]] = -.08       # compact hook X
    rb[[0,4,11,12,17,18]] = -.03  # compact fist family
    rb[[1,5,9,16,19,20,21]] = .025 # multi-finger upright family
    W["region_scores.b"] = rb

    (dst / "network.json").write_text(json.dumps({"layers":layers}, separators=(",", ":")))
    np.savez(dst / "weights.npz", **{k:v.astype(np.float32) for k,v in W.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
