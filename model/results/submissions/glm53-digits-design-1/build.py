"""Builds network.json + weights.npz for a hand-crafted digit classifier.

Everything here comes from general knowledge: filter formulas (Gaussian bar
detectors, rings, blobs), typed pen-path models of how people draw digits, and
hand-chosen thresholds.  No data files are read; nothing is fitted.

Pipeline:
  image -> oriented bar/ink/ring features (conv) -> threshold+gain -> 2x2 maxpool
  A parallel branch erodes thick strokes to their core and computes the SAME
  bar features on it, so stroke widths from ~1.5 to ~6px all produce solid,
  orientation-selective responses.
  Each digit class has several hand-typed prototype pen paths; dense layers
  dot the pooled features against each prototype's rendered "template"
  (positive deposits along strokes, negative weights where ink must be absent,
  distinctive strokes weighted higher).  The two branches' match scores are
  combined with an exact max, clamped below, z-normalised per image, and each
  class scores via an exact running max over its prototypes.
"""
import sys, json, math
import numpy as np
from pathlib import Path

# ---------------- constants (hand-chosen) ----------------
KS   = 11      # conv kernel size
PAD  = 5       # padding -> output stays 28x28
NORI = 8       # orientations, k * 22.5 deg
SIG_L = 2.0    # narrow bar: sigma along the stroke
SIG_P = 1.35   # narrow bar: sigma across (stroke widths ~1.4-3.6 px)
BAR_SCALE = 13.0
TH_BAR = 0.20  # bar threshold (soft; gain layer sharpens)
TH_INK = 0.20
TH_EMP = 0.50  # empty channel: value = TH_EMP - blur
TH_RING1 = 0.30
TH_RING2 = 0.24
TH_BLOB = 0.42
TH_BROAD = 0.18
R1, S1 = 4.4, 1.4   # small ring radius/sigma
R2, S2 = 7.0, 1.7   # large ring
ERO_T = 0.75        # erosion: 3x3 mean ink must exceed this (thick-stroke core)

GAIN  = [4.0] * 8 + [2.5, 2.0, 2.5, 2.5, 2.0, 2.5]   # bars, ink, emp, r1, r2, blob, broad
TH2   = [0.15] * 8 + [0.15, 0.15, 0.15, 0.15, 0.20, 0.10]

SELF_W1 = 2.4   # pen width for self-calibration, thin branch
SELF_W2 = 4.6   # pen width for self-calibration, thick (eroded) branch

# template constants
DEP_SA = 0.8    # deposit sigma ALONG the stroke (cells)
DEP_SX = 1.35   # deposit sigma ACROSS the stroke (cells)
ORI_W = [0.35, 1.0, 0.35]   # deposits to adjacent orientation channels
W_GEN = 0.09    # per-cell off-path ink penalty
W_ZONE = 0.28   # default per-cell must-be-empty zone penalty
END_BLOB = 3.0  # weight per stroke end on the blob channel (pre-normalisation)
EMP_POS = 0.35  # positive mass on the empty channel inside zones
EV_MIN, EV_MAX = 0.20, 0.90
ZGUARD = 4.6    # zone cells must be this far (px) from the path
NEG_D = 4.4     # generic off-path penalty distance
CLAMPF = -1.5   # floor for raw match scores before normalisation
EPS_RAW = 0.03  # weight of the linear (mean-template) tie-break branch
GRID = 14       # pooled grid (maxpool 2x2 stride 2)

# channels (bank and template space identical): 0..7 bars, 8 ink, 9 empty,
# 10 ring1, 11 ring2, 12 blob, 13 broad
NCH = 14
IDX_INK, IDX_EMP, IDX_R1, IDX_R2, IDX_BLOB, IDX_BROAD = 8, 9, 10, 11, 12, 13

# ---------------- filter construction ----------------
def grid_xy():
    m = (KS - 1) / 2.0
    ys, xs = np.mgrid[0:KS, 0:KS]
    return xs - m, ys - m

def gauss(sigma):
    x, y = grid_xy()
    k = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return k / k.sum()

def bar_kernel(theta, sl=SIG_L, sp=SIG_P, scale=BAR_SCALE):
    """Second-derivative-of-Gaussian bar detector: positive centre lobe,
    negative flanks - orientation-selective, silent on broad blobs.
    Hard-truncated across the stroke so a parallel wall ~4.5px away (the two
    sides of a small loop, e.g. the top of an '8') is not suppressed."""
    x, y = grid_xy()
    u = x * math.cos(theta) + y * math.sin(theta)
    v = -x * math.sin(theta) + y * math.cos(theta)
    k = (1.0 - v * v / (sp * sp)) * np.exp(-(u * u) / (2.0 * sl * sl) - (v * v) / (2.0 * sp * sp))
    k = k * (np.abs(v) <= 3.2) * (np.abs(u) <= 4.2)
    return k / scale

def ring_kernel(r, sigma):
    x, y = grid_xy()
    d = np.sqrt(x * x + y * y)
    k = np.exp(-((d - r) ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()

def feature_bank():
    W = np.zeros((NCH, 1, KS, KS), dtype=np.float64)
    b = np.zeros(NCH, dtype=np.float64)
    for k in range(NORI):
        W[k, 0] = bar_kernel(k * math.pi / NORI)
        b[k] = -TH_BAR
    W[IDX_INK, 0] = gauss(1.6);   b[IDX_INK] = -TH_INK
    W[IDX_EMP, 0] = -gauss(1.6);  b[IDX_EMP] = +TH_EMP
    W[IDX_R1, 0] = ring_kernel(R1, S1); b[IDX_R1] = -TH_RING1
    W[IDX_R2, 0] = ring_kernel(R2, S2); b[IDX_R2] = -TH_RING2
    W[IDX_BLOB, 0] = gauss(1.0);  b[IDX_BLOB] = -TH_BLOB
    W[IDX_BROAD, 0] = gauss(3.2); b[IDX_BROAD] = -TH_BROAD
    return W, b

def erode_bank():
    G = 4.0   # rescale core values back to ~0..1 ink range
    W = np.ones((1, 1, 3, 3), dtype=np.float64) * G / 9.0
    b = np.array([-G * ERO_T])
    return W, b

# ---------------- pen-path helpers ----------------
def to_px(pts):
    return [(x + 4.0, y + 4.0) for (x, y) in pts]

def shear_pts(pts, s, cy=10.0):
    return [(x + s * (cy - y), y) for (x, y) in pts]

def catmull(pts, step=0.55, idx=False):
    """Smooth polyline through control points, sampled every ~step units.
    With idx=True each sample is (x, y, left_control_point_index)."""
    p = [np.array(q, float) for q in pts]
    if len(p) == 2:
        p = [p[0], p[0], p[1], p[1]]
    elif len(p) < 4:
        p = [p[0]] + p + [p[-1]]
    else:
        p = [p[0]] + p + [p[-1]]
    out = []
    for i in range(len(p) - 3):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        L = max(np.linalg.norm(p2 - p1), 1e-6)
        n = max(2, int(L / step))
        for j in range(n):
            t = j / n
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t +
                       (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                       (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)
            out.append((q[0], q[1], i) if idx else (q[0], q[1]))
    out.append((p[-2][0], p[-2][1], len(p) - 4) if idx else (p[-2][0], p[-2][1]))
    return out

# ---------------- prototype library ----------------
"""Prototype pen-path library: hand-typed models of how people draw digits.

Sizes match how digits fill the 20x20 box (roughly 18-20 tall; widths vary by
class). Logical coords: u in [0,20) right, v in [0,20) down, centre (10,10).
Loops are stadiums (straight sides, rounded corners): tight arcs fire
oriented-bar detectors weakly, and real loops are drawn that way too.
"""
import math
import numpy as np

def stadium(cx, cy, w, h, r, nc=4, start='r'):
    """Rounded-rectangle loop perimeter (see build.py for the deposit logic)."""
    hw, hh = w / 2.0, h / 2.0
    def arc(ccx, ccy, a0, a1):
        return [(ccx + r * math.cos(a0 + (a1 - a0) * i / nc),
                 ccy + r * math.sin(a0 + (a1 - a0) * i / nc)) for i in range(1, nc + 1)]
    pts = [(cx + hw, cy)]
    pts += arc(cx + hw - r, cy - hh + r, 0.0, -math.pi / 2)
    pts += [(cx - hw + r, cy - hh)]
    pts += arc(cx - hw + r, cy - hh + r, -math.pi / 2, -math.pi)
    pts += [(cx - hw, cy)]
    pts += arc(cx - hw + r, cy + hh - r, math.pi, math.pi / 2)
    pts += [(cx - hw + r, cy + hh)]
    pts += arc(cx + hw - r, cy + hh - r, math.pi / 2, 0.0)
    pts += [(cx + hw, cy)]
    tgt = {'r': (cx + hw, cy), 't': (cx, cy - hh), 'b': (cx, cy + hh), 'l': (cx - hw, cy)}[start]
    i0 = int(np.argmin([(p[0] - tgt[0]) ** 2 + (p[1] - tgt[1]) ** 2 for p in pts]))
    pts = pts[i0:] + pts[:i0]
    if start != 'r':
        pts = pts + [pts[0]]
    return pts

def E(cx, cy, rx, ry, w=None):
    return ('e', cx, cy, rx, ry, w) if w else ('e', cx, cy, rx, ry)

PROTOS = {
0: [
  dict(name='0a', shear=0.0,
       strokes=[stadium(10.2, 10.2, 15.4, 17.6, 5.2)],
       empty=[E(10.2, 10.2, 3.2, 5.4)]),
  dict(name='0b', shear=0.12,
       strokes=[stadium(10.3, 10.1, 14.4, 18.0, 4.8)],
       empty=[E(10.3, 10.1, 2.8, 5.6)]),
  dict(name='0c', shear=-0.10,
       strokes=[stadium(10.4, 10.3, 16.4, 16.4, 5.6)],
       empty=[E(10.4, 10.3, 3.4, 4.8)]),
  dict(name='0d', shear=0.0,
       strokes=[stadium(10.2, 10.1, 13.4, 18.4, 4.4)],
       empty=[E(10.2, 10.1, 2.4, 5.8)]),
],
1: [
  dict(name='1a', shear=0.0,
       strokes=[[(10.4, 1.0), (10.4, 19.0)]],
       empty=[E(5.8, 10.5, 1.6, 7.5), E(15.0, 10.5, 1.6, 7.5)]),
  dict(name='1b', shear=0.0,
       strokes=[[(12.4, 1.2), (10.8, 7.0), (9.6, 13.0), (8.6, 19.0)]],
       empty=[E(4.8, 14.0, 1.4, 4.0), E(15.2, 10.5, 1.4, 7.0)]),
  dict(name='1c', shear=0.0,  # flag at top-left, no base serif
       strokes=[[(7.0, 4.8), (8.8, 3.4), (10.6, 2.0), (11.0, 2.2),
                 (10.9, 8.0), (10.7, 14.0), (10.4, 19.0)]],
       empty=[E(14.8, 9.5, 1.6, 7.0), E(6.6, 12.5, 1.6, 5.5)]),
  dict(name='1d', shear=0.0,  # flag + base serif
       strokes=[[(6.7, 5.4), (8.8, 3.4), (10.7, 1.9), (11.0, 2.4),
                 (10.8, 8.0), (10.6, 14.0), (10.3, 18.6)],
                [(7.2, 18.8), (10.2, 19.0), (13.8, 18.7)]],
       empty=[E(14.8, 9.5, 1.6, 7.0), E(6.2, 12.6, 1.5, 3.2)]),
  dict(name='1e', shear=0.0,  # strong slant, no serifs
       strokes=[[(14.4, 1.2), (13.0, 7.0), (11.4, 13.0), (9.4, 19.0)]],
       empty=[E(17.0, 6.0, 1.2, 3.2), E(5.4, 15.0, 1.6, 3.4)]),
],
2: [
  dict(name='2a', shear=0.0,  # round top, curved diagonal
       strokes=[[(4.2, 6.6), (5.2, 4.3), (7.8, 3.1), (10.8, 2.9), (13.6, 3.3),
                 (15.6, 4.8), (16.5, 7.1), (15.9, 9.5), (14.5, 11.7),
                 (12.3, 13.7), (9.7, 15.7), (6.9, 17.2), (4.6, 18.2),
                 (4.4, 18.6), (10.2, 18.9), (16.6, 18.6)]],
       empty=[E(11.6, 7.0, 1.6, 1.5), ('e', 17.8, 14.8, 1.2, 2.2)],
       crit=[(0,'ln',4.4,18.6,16.6,18.6,1.6,2.2),(0,'ln',12.0,13.9,5.5,17.5,2.0,1.8)]),
  dict(name='2b', shear=0.0,  # angular Z-like two
       strokes=[[(4.0, 7.0), (6.6, 3.4), (11.4, 3.4), (14.6, 4.2), (16.4, 7.3),
                 (15.8, 9.8), (14.4, 11.3), (12.0, 12.7), (9.0, 14.5),
                 (6.0, 16.5), (4.2, 17.8), (4.3, 18.4), (10.0, 18.8),
                 (16.6, 18.5)]],
       empty=[E(11.4, 6.8, 1.5, 1.4), ('e', 17.8, 14.6, 1.2, 2.2)],
       crit=[(0,'ln',4.3,18.4,16.6,18.5,1.6,2.2),(0,'ln',11.5,12.9,5.0,17.2,2.0,1.8)]),
  dict(name='2c', shear=0.10,  # loopy bottom
       strokes=[[(4.6, 6.4), (5.8, 4.2), (8.6, 3.2), (11.8, 3.3), (14.6, 4.0),
                 (16.3, 5.8), (16.5, 8.2), (15.6, 10.4), (13.8, 12.2),
                 (11.6, 14.0), (9.0, 15.8), (6.8, 17.2), (5.8, 16.6),
                 (7.2, 15.4), (9.4, 15.2), (11.2, 16.2), (12.8, 17.8),
                 (13.6, 18.8), (9.0, 19.1), (4.6, 18.7)]],
       empty=[E(11.8, 6.6, 1.5, 1.4)],
       crit=[(0,'ln',4.6,18.7,13.6,18.8,1.8,2.2),(0,'ln',11.6,14.0,6.0,17.3,2.2,1.8)]),
],
3: [
  dict(name='3a', shear=0.0,
       strokes=[[(4.4, 6.0), (6.2, 3.9), (9.4, 3.1), (12.6, 3.4), (15.2, 4.7),
                 (16.5, 6.9), (16.0, 9.1), (14.4, 10.4), (12.2, 10.9),
                 (10.7, 11.1), (11.8, 11.6), (14.4, 12.5), (16.1, 14.2),
                 (16.6, 16.2), (15.5, 18.2), (13.0, 19.3), (9.9, 19.6),
                 (7.0, 18.9), (5.6, 17.5)]],
       empty=[E(12.6, 6.6, 1.6, 1.4), E(13.0, 15.6, 1.6, 1.5),
              ('e', 6.0, 11.6, 2.0, 2.8, 0.35)],
       crit=[(0,'ln',15.6,4.9,16.1,8.6,1.5,1.5),(0,'ln',16.0,14.4,15.4,18.0,1.5,1.5)]),
  dict(name='3b', shear=0.08,  # angular, pointy waist
       strokes=[[(4.2, 6.2), (7.0, 3.4), (11.6, 3.6), (15.0, 5.2), (16.3, 7.8),
                 (15.9, 9.9), (14.6, 10.8), (12.4, 10.9), (10.9, 11.0),
                 (12.0, 11.6), (14.6, 12.8), (16.1, 14.8), (16.3, 16.9),
                 (15.0, 18.6), (12.4, 19.5), (9.2, 19.4), (6.6, 18.4)]],
       empty=[E(12.8, 6.4, 1.5, 1.3), E(13.2, 15.8, 1.5, 1.4),
              ('e', 5.8, 11.6, 2.0, 2.8, 0.35)],
       crit=[(0,'ln',15.4,5.4,15.9,9.4,1.5,1.5),(0,'ln',15.9,15.0,14.9,18.4,1.5,1.5)]),
  dict(name='3c', shear=0.0,  # flat-top three
       strokes=[[(4.0, 6.4), (6.8, 4.4), (10.4, 4.0), (13.8, 4.6), (15.8, 6.4),
                 (16.1, 8.6), (14.8, 10.0), (12.6, 10.7), (11.0, 10.9),
                 (12.0, 11.4), (14.4, 12.4), (16.0, 14.2), (16.4, 16.4),
                 (15.2, 18.4), (12.6, 19.5), (9.4, 19.4), (6.8, 18.4)]],
       empty=[E(12.8, 6.8, 1.5, 1.2), E(13.2, 15.8, 1.5, 1.4),
              ('e', 6.0, 11.6, 2.0, 2.8, 0.35)],
       crit=[(0,'ln',15.8,6.6,16.1,8.4,1.5,1.5),(0,'ln',15.8,14.4,15.1,18.2,1.5,1.5)]),
],
4: [
  dict(name='4a', shear=0.0,  # open four: left vertical, bar, right vertical
       strokes=[[(4.0, 6.4), (3.7, 10.4), (3.6, 15.0)],
                [(3.7, 14.9), (8.0, 15.4), (12.4, 15.3), (15.8, 14.8)],
                [(16.2, 4.6), (16.4, 10.0), (16.5, 19.2)]],
       empty=[('e', 10.0, 10.2, 3.0, 2.6), ('e', 10.0, 18.2, 3.4, 1.0)],
       crit=[(2,'ln',16.3,4.6,16.5,19.2,1.8,2.5),(0,'ln',3.8,6.4,3.6,15.0,1.8,2.0),(1,'ln',3.7,15.0,15.8,15.0,1.5,2.2)]),
  dict(name='4b', shear=0.0,  # closed triangle four
       strokes=[[(5.6, 4.6), (7.8, 8.0), (10.6, 11.2), (14.0, 12.9)],
                [(3.4, 13.1), (7.6, 13.2), (12.0, 13.2), (16.9, 13.3)],
                [(16.9, 4.4), (17.1, 9.0), (17.2, 19.4)]],
       empty=[('e', 10.4, 9.6, 2.4, 2.0), ('e', 10.2, 18.2, 3.6, 1.0)],
       crit=[(2,'ln',17.0,4.4,17.2,19.4,1.8,2.5),(1,'ln',3.4,13.2,16.9,13.3,1.5,2.2),(0,'ln',5.6,4.6,10.6,11.2,2.0,2.0)]),
  dict(name='4c', shear=0.0,  # long diagonal open four
       strokes=[[(3.6, 4.6), (5.6, 7.6), (8.6, 10.8), (12.4, 12.8), (15.4, 13.2)],
                [(3.4, 13.1), (7.6, 13.3), (12.0, 13.3), (16.9, 13.3)],
                [(16.9, 4.6), (17.1, 9.5), (17.2, 19.4)]],
       empty=[('e', 8.0, 8.2, 2.0, 2.0), ('e', 10.2, 18.2, 3.6, 1.0)],
       crit=[(2,'ln',17.0,4.6,17.2,19.4,1.8,2.5),(1,'ln',3.4,13.2,16.9,13.3,1.5,2.2),(0,'ln',3.6,4.6,8.6,10.8,2.0,2.0)]),
],
5: [
  dict(name='5a', shear=0.0,  # round belly, tail sweeps left
       strokes=[[(2.8, 4.5), (8.2, 4.3), (13.6, 4.2), (17.2, 4.3)],
                [(3.0, 4.8), (3.1, 7.4), (3.3, 9.8), (3.7, 10.6),
                 (6.6, 11.4), (10.4, 12.0), (13.4, 13.4), (15.2, 15.4),
                 (14.9, 17.4), (12.6, 18.8), (9.2, 19.3), (5.6, 19.4),
                 (3.8, 19.4)]],
       empty=[('e', 14.6, 8.0, 2.2, 1.4)],
       crit=[(0,'ln',2.8,4.4,17.2,4.3,1.5,1.6)]),
  dict(name='5b', shear=0.0,  # flat bottom bar
       strokes=[[(2.8, 4.5), (8.2, 4.3), (13.6, 4.2), (17.2, 4.3)],
                [(3.0, 4.8), (3.1, 7.2), (3.4, 9.8), (4.0, 10.8),
                 (7.2, 11.6), (11.0, 12.2), (13.8, 13.8), (15.0, 15.8),
                 (14.0, 17.4), (11.2, 18.2), (7.8, 18.6), (4.4, 18.8)],
                [(2.8, 18.9), (8.0, 19.0), (13.6, 18.8)]],
       empty=[('e', 14.6, 8.0, 2.2, 1.4)],
       crit=[(0,'ln',2.8,4.4,17.2,4.3,1.5,1.6)]),
  dict(name='5c', shear=0.10,  # angular S-like five
       strokes=[[(3.0, 4.8), (8.4, 4.5), (13.8, 4.4), (17.0, 4.5)],
                [(3.1, 5.0), (3.2, 8.0), (3.4, 10.4), (4.4, 10.9),
                 (7.8, 11.3), (11.4, 12.0), (14.0, 13.8), (14.6, 15.8),
                 (13.0, 17.4), (9.8, 18.2), (6.0, 18.6), (4.2, 18.8)]],
       empty=[('e', 14.8, 8.2, 2.0, 1.2)],
       crit=[(0,'ln',3.0,4.7,17.0,4.5,1.5,1.6)]),
],
6: [
  dict(name='6a', shear=0.0,
       strokes=[[(14.4, 3.2), (13.2, 5.4), (11.6, 7.6), (9.8, 9.6), (8.4, 11.6)],
                stadium(11.4, 15.3, 11.4, 7.6, 3.0, start='l')],
       empty=[E(11.5, 15.4, 1.6, 1.4), ('e', 17.2, 6.2, 1.6, 2.4),
              ('e', 5.4, 6.4, 1.2, 2.0, 0.35), ('e', 6.8, 6.4, 1.2, 1.8, 0.35)],
       crit=[(0,'ln',14.4,3.2,11.6,7.6,2.2,2.0)]),
  dict(name='6b', shear=0.0,  # straight neck
       strokes=[[(14.2, 3.2), (13.8, 6.0), (13.0, 8.8), (11.6, 11.2), (11.0, 12.8)],
                stadium(12.6, 15.6, 10.6, 7.2, 2.8, start='l')],
       empty=[E(12.7, 15.7, 1.2, 1.2), ('e', 17.4, 6.2, 1.4, 2.4),
              ('e', 5.6, 6.4, 1.2, 2.0, 0.35), ('e', 7.0, 6.4, 1.1, 1.7, 0.35)],
       crit=[(0,'ln',14.2,3.2,13.0,8.8,2.2,2.0)]),
  dict(name='6c', shear=-0.10,  # curved neck, narrower loop
       strokes=[[(14.8, 3.1), (13.0, 5.6), (11.0, 8.2), (9.2, 10.8), (8.4, 13.0)],
                stadium(11.4, 15.7, 10.8, 7.2, 2.8, start='l')],
       empty=[E(11.5, 15.8, 1.3, 1.2), ('e', 17.4, 6.0, 1.4, 2.2),
              ('e', 5.6, 6.4, 1.2, 2.0, 0.35)],
       crit=[(0,'ln',14.8,3.1,11.0,8.2,2.2,2.0)]),
],
7: [
  dict(name='7a', shear=0.0,
       strokes=[[(2.6, 4.7), (7.4, 4.5), (12.2, 4.4), (16.8, 4.5)],
                [(16.6, 4.8), (15.7, 7.2), (14.2, 10.4), (12.2, 13.8),
                 (10.0, 16.8), (8.0, 19.3)]],
       empty=[('e', 6.2, 9.4, 3.2, 2.2), ('e', 15.8, 15.8, 2.4, 2.2)],
       crit=[(0,'ln',2.6,4.6,16.8,4.5,1.5,2.2),(1,'ln',15.7,7.2,12.2,13.8,2.0,1.3)]),
  dict(name='7b', shear=0.0,  # straight diagonal
       strokes=[[(2.6, 4.7), (7.4, 4.5), (12.2, 4.4), (16.8, 4.5)],
                [(16.6, 4.8), (15.0, 8.4), (13.2, 12.0), (11.2, 15.6),
                 (9.4, 18.8)]],
       empty=[('e', 6.2, 9.4, 3.2, 2.2), ('e', 15.8, 15.8, 2.4, 2.2)],
       crit=[(0,'ln',2.6,4.6,16.8,4.5,1.5,2.2),(1,'ln',15.0,8.4,11.2,15.6,2.0,1.3)]),
  dict(name='7c', shear=0.0,  # crossed seven
       strokes=[[(2.6, 4.7), (7.4, 4.5), (12.2, 4.4), (16.8, 4.5)],
                [(16.6, 4.8), (15.4, 7.6), (14.0, 10.8), (12.2, 14.0),
                 (10.2, 16.9), (8.4, 19.3)],
                [(6.6, 13.1), (9.2, 12.6), (11.8, 12.3), (14.6, 12.2)]],
       empty=[('e', 6.0, 8.8, 2.8, 1.6), ('e', 16.4, 16.4, 1.6, 1.8)],
       crit=[(0,'ln',2.6,4.6,16.8,4.5,1.5,2.2),(1,'ln',15.4,7.6,12.2,14.0,2.0,1.3)]),
  dict(name='7d', shear=0.10,
       strokes=[[(4.0, 5.0), (8.4, 4.4), (12.8, 4.2), (17.2, 4.4)],
                [(17.0, 4.8), (15.4, 8.2), (13.5, 11.6), (11.3, 15.0),
                 (9.4, 18.6)]],
       empty=[('e', 6.6, 9.6, 3.0, 2.0), ('e', 16.0, 15.8, 2.2, 2.0)],
       crit=[(0,'ln',4.0,5.0,17.2,4.4,1.5,2.2),(1,'ln',15.4,8.2,11.3,15.0,2.0,1.3)]),
],
8: [
  dict(name='8a', shear=0.0,
       strokes=[stadium(10.5, 6.6, 7.2, 7.4, 2.7, start='b') +
                [(7.4, 12.2), (13.6, 10.4)] +
                stadium(10.4, 15.2, 10.6, 8.8, 3.4, start='t') +
                [(13.4, 12.4), (7.2, 10.6)]],
       empty=[('e', 4.2, 11.2, 1.2, 1.0, 0.30), E(11.0, 6.8, 1.3, 1.5, 0.35),
              E(10.4, 15.4, 1.2, 1.5, 0.30)],
       crit=[(0,'ln',6.8,12.8,9.8,10.6,2.0,3.0),(0,'ln',6.9,4.2,6.9,8.6,1.8,3.0),(0,'ln',5.1,11.6,5.1,18.8,1.8,2.5)]),
  dict(name='8b', shear=0.10,  # tall narrow
       strokes=[stadium(10.3, 6.4, 6.6, 7.8, 2.5, start='b') +
                [(7.0, 12.0), (13.2, 10.2)] +
                stadium(10.2, 15.2, 9.6, 9.0, 3.0, start='t') +
                [(13.0, 12.2), (6.9, 10.4)]],
       empty=[('e', 4.0, 11.0, 1.2, 1.0, 0.30), E(10.8, 6.6, 1.2, 1.4, 0.35),
              E(10.2, 15.4, 1.0, 1.4, 0.30)],
       crit=[(0,'ln',6.5,12.4,9.3,10.4,2.0,3.0),(0,'ln',7.0,4.0,7.0,8.4,1.8,3.0),(0,'ln',5.4,11.2,5.4,18.4,1.8,2.5)]),
  dict(name='8c', shear=-0.08,  # round, bottom-heavy
       strokes=[stadium(10.7, 6.8, 7.6, 7.4, 2.7, start='b') +
                [(7.6, 12.6), (13.9, 10.6)] +
                stadium(10.4, 15.3, 11.4, 9.0, 3.6, start='t') +
                [(13.7, 12.6), (7.4, 10.8)]],
       empty=[('e', 4.4, 11.4, 1.2, 1.0, 0.30), E(11.2, 7.0, 1.3, 1.5, 0.35),
              E(10.5, 15.5, 1.3, 1.6, 0.30)],
       crit=[(0,'ln',7.2,12.8,10.0,11.0,2.0,3.0),(0,'ln',7.1,4.4,7.1,8.8,1.8,3.0),(0,'ln',4.7,11.4,4.7,19.0,1.8,2.5)]),
],
9: [
  dict(name='9a', shear=0.0,  # round loop, gently curving tail
       strokes=[stadium(10.6, 7.2, 9.2, 7.8, 2.9, start='b') +
                [(12.8, 11.0), (14.0, 12.2), (14.2, 14.2), (13.6, 16.6),
                 (12.4, 18.6), (10.8, 19.6)]],
       empty=[('e', 5.0, 16.2, 2.2, 2.8, 0.35)],
       crit=[(0,'ln',6.0,5.8,6.0,8.6,1.8,2.5),(0,'ln',13.9,13.0,12.0,18.8,2.2,1.8)]),
  dict(name='9b', shear=0.0,  # straight tail
       strokes=[stadium(10.6, 7.2, 9.2, 7.8, 2.9, start='b') +
                [(13.0, 11.0), (14.0, 13.0), (14.0, 15.4), (13.9, 17.6),
                 (13.7, 19.4)]],
       empty=[('e', 5.0, 16.2, 2.2, 2.8, 0.35)],
       crit=[(0,'ln',6.0,5.8,6.0,8.6,1.8,2.5),(0,'ln',14.0,13.5,13.7,19.0,2.0,1.8)]),
  dict(name='9c', shear=0.10,  # big loop, tail hooks left
       strokes=[stadium(10.8, 7.4, 9.8, 8.0, 3.0, start='b') +
                [(13.0, 11.2), (14.2, 12.4), (14.0, 14.6), (12.8, 17.0),
                 (11.0, 18.8), (8.6, 19.6)]],
       empty=[('e', 4.6, 15.8, 1.8, 2.4, 0.35)],
       crit=[(0,'ln',5.9,5.8,5.9,9.0,1.8,2.5),(0,'ln',14.0,13.5,10.5,18.9,2.2,1.8)]),
  dict(name='9d', shear=-0.08,  # narrow loop, straight tail right of centre
       strokes=[stadium(10.5, 7.0, 8.2, 7.4, 2.6, start='b') +
                [(12.4, 10.6), (13.6, 12.4), (13.5, 15.0), (13.3, 17.4),
                 (13.1, 19.2)]],
       empty=[('e', 4.8, 16.0, 2.0, 2.6, 0.35)],
       crit=[(0,'ln',6.4,5.6,6.4,8.4,1.8,2.5),(0,'ln',13.5,12.5,13.0,18.8,2.0,1.8)]),
],
}

# --- slant variants: copies of the main variant at stronger shears ---
_SHEAR_BASES = {0: '0a', 1: '1a', 2: '2a', 3: '3a', 4: '4a', 5: '5a',
                6: '6a', 7: '7a', 8: '8a', 9: '9a'}
for _c, _nm in _SHEAR_BASES.items():
    _base = [v for v in PROTOS[_c] if v['name'] == _nm][0]
    for _sh, _tag in ((-0.20, 'm'), (0.24, 'p')):
        _v = dict(_base)
        _v['name'] = _nm + _tag
        _v['shear'] = _sh
        PROTOS[_c].append(_v)


# ---------------- template construction ----------------
def sample_strokes(var, shift=(0.0, 0.0)):
    """-> list of (x, y, theta, mult) in pixel coords; mult from var['crit']
    entries (stroke, 'bx', u0, v0, u1, v1, w)."""
    out = []
    sh = var.get('shear', 0.0)
    crit = var.get('crit', [])
    for si, stroke in enumerate(var['strokes']):
        pts = shear_pts(stroke, sh) if sh else list(stroke)
        sm = catmull(pts, step=0.5, idx=True)
        sm = [(x + 4.0 + shift[0], y + 4.0 + shift[1], i) for (x, y, i) in sm]
        rng = []
        for cr in crit:
            if cr[0] == si and cr[1] == 'bx':
                rng.append(('bx',) + tuple(cr[2:]))
            elif cr[0] == si and cr[1] == 'ln':
                rng.append(('ln',) + tuple(cr[2:]))
        for i in range(len(sm) - 1):
            (x0, y0, _), (x1, y1, _) = sm[i], sm[i + 1]
            th = math.atan2(y1 - y0, x1 - x0) % math.pi
            m = 1.0
            lu = (x0 + x1) / 2 - 4.0 - shift[0]
            lv = (y0 + y1) / 2 - 4.0 - shift[1]
            for rg in rng:
                if rg[0] == 'bx':
                    _, u0, v0, u1, v1, w = rg
                    if u0 <= lu <= u1 and v0 <= lv <= v1:
                        m = w
                else:
                    _, u0, v0, u1, v1, hw, w = rg
                    # distance from (lu,lv) to segment (u0,v0)-(u1,v1)
                    du, dv = u1 - u0, v1 - v0
                    L2 = du * du + dv * dv
                    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((lu - u0) * du + (lv - v0) * dv) / L2))
                    d = math.hypot(lu - (u0 + t * du), lv - (v0 + t * dv))
                    if d <= hw:
                        m = w
            out.append(((x0 + x1) / 2, (y0 + y1) / 2, th, m))
        out.append((sm[0][0], sm[0][1],
                    math.atan2(sm[1][1] - sm[0][1], sm[1][0] - sm[0][0]) % math.pi, 1.0))
        out.append((sm[-1][0], sm[-1][1],
                    math.atan2(sm[-1][1] - sm[-2][1], sm[-1][0] - sm[-2][0]) % math.pi, 1.0))
    return out

def deposit(pos, ch, r, c, w):
    r = min(max(r, 0), GRID - 1); c = min(max(c, 0), GRID - 1)
    pos[ch, r, c] += w

def render_ink(var, width, shift=(0.0, 0.0)):
    """Deterministic rendering of the variant's own pen paths (formula only)."""
    img = np.zeros((28, 28))
    r = width / 2.0
    sh = var.get('shear', 0.0)
    for stroke in var['strokes']:
        pts = shear_pts(stroke, sh) if sh else list(stroke)
        sm = to_px(catmull(pts, step=0.3))
        sm = [(x + shift[0], y + shift[1]) for (x, y) in sm]
        rad = int(r) + 2
        for (x, y) in sm:
            xi, yi = int(round(x)), int(round(y))
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    yy, xx = yi + dy, xi + dx
                    if 0 <= yy < 28 and 0 <= xx < 28:
                        d = math.hypot(xx - x, yy - y)
                        v = (r + 0.7 - d) / 1.4
                        if v > 0:
                            img[yy, xx] = max(img[yy, xx], min(v, 1.0))
    return img

def _conv2d(ink, W, b, ksize, pad):
    xp = np.pad(ink[None, :, :], ((0, 0), (pad, pad), (pad, pad)))
    o = W.shape[0]
    acc = np.zeros((o, 28, 28))
    for dy in range(ksize):
        for dx in range(ksize):
            acc += W[:, 0, dy, dx][:, None, None] * xp[0, dy:dy + 28, dx:dx + 28]
    return acc + b[:, None, None]

def feat_pool(ink, Wf, bf, erode=False):
    """features + relu + 2x2 maxpool + gain, with optional erosion first."""
    if erode:
        We, be = erode_bank()
        ink = np.maximum(_conv2d(ink, We, be, 3, 1)[0], 0.0)
    acc = np.maximum(_conv2d(ink, Wf, bf, KS, PAD), 0.0)
    pm = acc[:, 0::2, 0::2].copy()
    for dy in range(2):
        for dx in range(2):
            pm = np.maximum(pm, acc[:, dy::2, dx::2])
    g = np.array(GAIN)[:, None, None]
    t2 = np.array(TH2)[:, None, None]
    return np.maximum(g * pm - g * t2, 0.0)

def template_row(var, Wf, bf, thick=False):
    """Template weights in (NCH, GRID, GRID), self-calibrated in its branch."""
    ink0 = render_ink(var, SELF_W1 if not thick else SELF_W2)
    tot = ink0.sum()
    if tot > 0:
        cy = float((ink0.sum(1) * np.arange(28)).sum() / tot)
        cx = float((ink0.sum(0) * np.arange(28)).sum() / tot)
        shift = (14.0 - cx, 14.0 - cy)
    else:
        shift = (0.0, 0.0)
    pos = np.zeros((NCH, GRID, GRID))
    samples = sample_strokes(var, shift)
    ends = []
    sh = var.get('shear', 0.0)
    for stroke in var['strokes']:
        sm = to_px(catmull(shear_pts(stroke, sh) if sh else list(stroke), step=0.5))
        ends += [(sm[0][0] + shift[0], sm[0][1] + shift[1]),
                 (sm[-1][0] + shift[0], sm[-1][1] + shift[1])]
    for (x, y, th, mult) in samples:
        r, c = int(y // 2), int(x // 2)
        k0 = int(round(th / (math.pi / NORI))) % NORI
        ct, st = math.cos(th), math.sin(th)
        for dk, wk in zip((-1, 0, 1), ORI_W):
            for dr in (-3, -2, -1, 0, 1, 2, 3):
                for dc in (-3, -2, -1, 0, 1, 2, 3):
                    along = dr * st + dc * ct
                    across = dr * ct - dc * st
                    w = wk * math.exp(-(along * along / (2 * DEP_SA * DEP_SA) +
                                         across * across / (2 * DEP_SX * DEP_SX))) * mult
                    if w > 0.03:
                        deposit(pos, (k0 + dk) % NORI, r + dr, c + dc, w)
    for (x, y) in ends:
        deposit(pos, IDX_BLOB, int(y // 2), int(x // 2), END_BLOB)

    neg = np.zeros((GRID, GRID))
    dist = np.full((GRID, GRID), 99.0)
    sy = np.array([q[1] for q in samples]); sx = np.array([q[0] for q in samples])
    for r in range(GRID):
        for c in range(GRID):
            cy2, cx2 = 2 * r + 1.0, 2 * c + 1.0
            d = np.min(np.hypot(sx - cx2, sy - cy2))
            dist[r, c] = d
            if d > NEG_D:
                neg[r, c] += W_GEN
    emp = np.zeros((GRID, GRID))
    for z in var.get('empty', []):
        _, zx, zy, zrx, zry = z[:5]
        zw = z[5] if len(z) > 5 else W_ZONE
        zsh = var.get('shear', 0.0)
        zx = zx + zsh * (10.0 - zy)
        zx, zy = zx + 4.0 + shift[0], zy + 4.0 + shift[1]
        zrx, zry = zrx + 4.0 * abs(zsh), zry
        for r in range(GRID):
            for c in range(GRID):
                cy2, cx2 = 2 * r + 1.0, 2 * c + 1.0
                q = ((cx2 - zx) / zrx) ** 2 + ((cy2 - zy) / zry) ** 2
                if q <= 1.0 and dist[r, c] > ZGUARD:
                    emp[r, c] += zw
    if emp.sum() > 0:
        pos[IDX_EMP] += EMP_POS * pos.sum() * emp / emp.sum()

    f = feat_pool(render_ink(var, SELF_W2 if thick else SELF_W1, shift=shift),
                  Wf, bf, erode=thick)
    # prune deposits our own rendering cannot detect (short-stroke spread into
    # empty interiors); keep a fraction for shift tolerance
    for ch in range(NORI):
        cover = (f[ch] > 0.04).astype(np.float64)
        pos[ch] *= 0.15 + 0.85 * cover
    pos_sum = pos.sum()
    pos /= pos_sum
    EV = np.full(NCH, 0.25)
    for ch in range(NCH):
        m = pos[ch].sum()
        if m > 1e-6:
            ev = (pos[ch] * f[ch]).sum() / m
            EV[ch] = min(max(ev, EV_MIN), EV_MAX)
    row = pos / EV[:, None, None]
    ink_cells = dist <= 2.5
    if ink_cells.sum() > 3:
        ev_ink = min(max(f[IDX_INK][ink_cells].mean(), EV_MIN), EV_MAX)
    else:
        ev_ink = 0.25
    row[IDX_INK] -= (neg + emp) / ev_ink
    m_self = float((row * f).sum())
    row /= min(max(m_self, 0.75), 1.5)   # self-match ~1, mild boost only
    return row

def build_templates():
    Wf, bf = feature_bank()
    T1, T2, meta = [], [], []
    for c in range(10):
        for var in PROTOS[c]:
            T1.append(template_row(var, Wf, bf, thick=False))
            T2.append(template_row(var, Wf, bf, thick=True))
            meta.append((c, var['name']))
    return np.stack(T1), np.stack(T2), meta

# ---------------- network assembly ----------------
def build(out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Wf, bf = feature_bank()
    We, be = erode_bank()
    T1, T2, meta = build_templates()
    M = len(meta)
    slots = []
    for c in range(10):
        for j in range(len(PROTOS[c])):
            slots.append((c, j))
    rounds = max(len(PROTOS[c]) for c in range(10)) - 1
    layers = [
        {"name": "feat", "op": "conv", "inputs": ["image"], "out": NCH,
         "kernel": [KS, KS], "stride": 1, "padding": PAD},
        {"name": "fr", "op": "relu", "inputs": ["feat"]},
        {"name": "pool", "op": "maxpool", "inputs": ["fr"], "size": [2, 2], "stride": [2, 2]},
        {"name": "gain", "op": "conv", "inputs": ["pool"], "out": NCH,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "gr", "op": "relu", "inputs": ["gain"]},
        {"name": "ero", "op": "conv", "inputs": ["image"], "out": 1,
         "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "eror", "op": "relu", "inputs": ["ero"]},
        {"name": "feat2", "op": "conv", "inputs": ["eror"], "out": NCH,
         "kernel": [KS, KS], "stride": 1, "padding": PAD},
        {"name": "fr2", "op": "relu", "inputs": ["feat2"]},
        {"name": "pool2", "op": "maxpool", "inputs": ["fr2"], "size": [2, 2], "stride": [2, 2]},
        {"name": "gain2", "op": "conv", "inputs": ["pool2"], "out": NCH,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "gr2", "op": "relu", "inputs": ["gain2"]},
        {"name": "m1", "op": "dense", "inputs": ["gr"], "out": M},
        {"name": "m2", "op": "dense", "inputs": ["gr2"], "out": M},
        {"name": "neg1", "op": "dense", "inputs": ["m1"], "out": M},
        {"name": "dpre", "op": "add", "inputs": ["m2", "neg1"]},
        {"name": "dr", "op": "relu", "inputs": ["dpre"]},
        {"name": "mmax", "op": "add", "inputs": ["m1", "dr"]},
        {"name": "mfloor", "op": "dense", "inputs": ["mmax"], "out": M},
        {"name": "mfr", "op": "relu", "inputs": ["mfloor"]},
        {"name": "mclamp", "op": "add", "inputs": ["mmax", "mfr"]},
        {"name": "mz", "op": "reshape", "inputs": ["mclamp"], "shape": [1, M, 1]},
        {"name": "mnorm", "op": "norm", "inputs": ["mz"]},
        {"name": "mflat", "op": "reshape", "inputs": ["mnorm"], "shape": [M]},
    ]
    prev = "mflat"
    for r in range(rounds):
        dn, rn, an = f"mx{r}d", f"mx{r}r", f"mx{r}a"
        layers.append({"name": dn, "op": "dense", "inputs": [prev], "out": M})
        layers.append({"name": rn, "op": "relu", "inputs": [dn]})
        layers.append({"name": an, "op": "add", "inputs": [prev, rn]})
        prev = an
    layers.append({"name": "cls", "op": "dense", "inputs": [prev], "out": 10})
    layers.append({"name": "raw", "op": "dense", "inputs": ["pool"], "out": 10})
    layers.append({"name": "out", "op": "add", "inputs": ["cls", "raw"]})

    Wg = np.zeros((NCH, NCH, 1, 1), dtype=np.float64)
    bg = np.zeros(NCH, dtype=np.float64)
    for ch in range(NCH):
        Wg[ch, ch, 0, 0] = GAIN[ch]
        bg[ch] = -GAIN[ch] * TH2[ch]
    weights = {
        "feat.W": Wf.astype(np.float32), "feat.b": bf.astype(np.float32),
        "feat2.W": Wf.astype(np.float32), "feat2.b": bf.astype(np.float32),
        "ero.W": We.astype(np.float32), "ero.b": be.astype(np.float32),
        "gain.W": Wg.astype(np.float32), "gain.b": bg.astype(np.float32),
        "gain2.W": Wg.astype(np.float32), "gain2.b": bg.astype(np.float32),
        "m1.W": T1.reshape(M, -1).astype(np.float32),
        "m1.b": np.zeros(M, dtype=np.float32),
        "m2.W": T2.reshape(M, -1).astype(np.float32),
        "m2.b": np.zeros(M, dtype=np.float32),
        "neg1.W": (-np.eye(M)).astype(np.float32),
        "neg1.b": np.zeros(M, dtype=np.float32),
        "mfloor.W": (-np.eye(M)).astype(np.float32),
        "mfloor.b": np.full(M, CLAMPF, dtype=np.float32),
    }
    for r in range(rounds):
        W = np.zeros((M, M), dtype=np.float64)
        for i, (c, j) in enumerate(slots):
            if j == 0 and r + 1 < len(PROTOS[c]):
                jj = slots.index((c, r + 1))
                W[i, i] = -1.0
                W[i, jj] = 1.0
        weights[f"mx{r}d.W"] = W.astype(np.float32)
        weights[f"mx{r}d.b"] = np.zeros(M, dtype=np.float32)
    Wcls = np.zeros((10, M), dtype=np.float64)
    for c in range(10):
        Wcls[c, slots.index((c, 0))] = 1.0
    weights["cls.W"] = Wcls.astype(np.float32)
    weights["cls.b"] = np.zeros(10, dtype=np.float32)
    weights["raw.W"] = np.zeros((10, T1[0].size), dtype=np.float32)
    weights["raw.b"] = np.zeros(10, dtype=np.float32)
    for c in range(10):
        idxs = [i for i, (cc, _) in enumerate(slots) if cc == c]
        mean_row = T1[idxs].mean(0)
        weights["raw.W"][c] = (EPS_RAW * mean_row).ravel()

    (out_dir / "network.json").write_text(json.dumps({"layers": layers}))
    np.savez(out_dir / "weights.npz", **weights)
    return meta

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
