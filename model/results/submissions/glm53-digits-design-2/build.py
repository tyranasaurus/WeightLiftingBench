"""Builds a hand-designed digit classifier: 480 stroke templates + max matching.

No training, no data; every weight comes from typed pen paths and hand-set
constants.  Design:
  * Per class: 8 hand-typed handwriting styles (6 for "1"), each rendered at
    2-4 slants and 2 stroke widths => 48 templates per class, 480 in total.
  * Each style is rendered anti-aliased at 28x28, auto-scaled so its ink
    spans exactly 20 px (the data scales digits to fit a 20x20 box), then
    re-centred so its centre of mass is the frame centre (as in the data),
    zero-meaned and L2-normalised.
  * All 480 templates form one conv layer with padding 3 (alignment search
    +-3 px; every window still contains the whole digit).  Max-pooling over
    the 7x7 shifts, then over each class's 48 variants, gives 10 class
    scores s_c (best template match per class).
  * Hand-set rule table finish: the 10 scores are normalised (norm op),
    thresholded at +1 (relu), and classes that overlap strongly with another
    strong class (classic confusable pairs like 8/9, 5/6, 1/7) are penalised
    by 0.22 of the rival's excess, which sharpens the winner's margin.
Run: python3 build.py NEW_FOLDER  ->  network.json + weights.npz.
"""
import json
import sys
import numpy as np

S = 28          # image side
CEN = 14.0      # frame centre

# ---------------------------------------------------------------- geometry --
def ellipse(cx, cy, rx, ry, n=40):
    pts = [(cx + rx * np.cos(2 * np.pi * i / n), cy + ry * np.sin(2 * np.pi * i / n))
           for i in range(n)]
    pts.append(pts[0])
    return pts


def seg_coverage(XX, YY, ax, ay, bx, by, hw, soft):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 == 0:
        d = np.hypot(XX - ax, YY - ay)
    else:
        u = np.clip(((XX - ax) * vx + (YY - ay) * vy) / L2, 0.0, 1.0)
        d = np.hypot(XX - (ax + u * vx), YY - (ay + u * vy))
    return np.clip((hw - d) / soft + 1.0, 0.0, 1.0)


def transform(strokes, scale, shear):
    """Box coords -> image coords (before offset). Returns list of point lists."""
    return [[(scale * x + shear * (10.0 - scale * y), scale * y) for (x, y) in stroke]
            for stroke in strokes]


def ink_extent(strokes, width, shear, scale, soft=1.0):
    """Analytic ink bounding-box extents (x, y) of a transformed drawing."""
    r = width / 2.0 + 0.95 * soft          # radius where coverage > 0.05
    xs, ys = [], []
    for stroke in transform(strokes, scale, shear):
        for (x, y) in stroke:
            xs.append(x); ys.append(y)
    return (max(xs) - min(xs) + 2 * r, max(ys) - min(ys) + 2 * r)


def render(strokes, width=2.4, shear=0.0, scale=1.0, off=(0.0, 0.0), soft=1.0, sub=4):
    """strokes: list of point lists in box coords, x in [-10,10], y in [0,20]."""
    hw = width / 2.0
    segs = []
    for stroke in transform(strokes, scale, shear):
        pts = [(CEN + x + off[0], 4.0 + y + off[1]) for (x, y) in stroke]
        segs += [(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                 for i in range(len(pts) - 1)]
    n = S * sub
    t = (np.arange(n) + 0.5) / sub
    XX, YY = np.meshgrid(t, t)
    field = np.zeros((n, n))
    for (ax, ay, bx, by) in segs:
        field = np.maximum(field, seg_coverage(XX, YY, ax, ay, bx, by, hw, soft))
    return field.reshape(S, sub, S, sub).mean(axis=(1, 3))


def com(img):
    v = img.sum()
    if v <= 0:
        return CEN, CEN
    ii, jj = np.mgrid[0:S, 0:S]
    return ((img * (jj + 0.5)).sum() / v, (img * (ii + 0.5)).sum() / v)


def make_template(strokes, width, shear=0.0):
    """Render a drawing so its ink spans exactly 20 px (like the data's box
    fit), centred on its centre of mass, zero-meaned and unit-normed."""
    ex, ey = ink_extent(strokes, width, shear, 1.0)
    scale = 20.0 / max(ex, ey)
    off = (0.0, 0.0)
    for _ in range(2):
        img = render(strokes, width, shear, scale, off)
        cx, cy = com(img)
        off = (off[0] + CEN - cx, off[1] + CEN - cy)
    img = render(strokes, width, shear, scale, off)
    t = img - img.mean()
    return t / np.sqrt((t * t).sum())

# --------------------------------------------------------------- pen paths --
# Box coords: x in [-10, 10], y in [0, 20] (top to bottom). Every drawing is
# ~20 tall (or wide) because the data scales digits to fit a 20x20 box.

def ellipse_pts(cx, cy, rx, ry, n=40):
    pts = [(cx + rx * np.cos(2 * np.pi * i / n), cy + ry * np.sin(2 * np.pi * i / n))
           for i in range(n)]
    pts.append(pts[0])
    return pts


def eight(trx, brx):
    """Two round loops that cross narrowly at the waist (like a real pen 8)."""
    return [ellipse_pts(0, 5.4, trx, 4.2), ellipse_pts(0, 13.7, brx, 4.9)]

N9_STRAIGHT = [(4.8, 6.0), (4.2, 9.8), (3.2, 14.2), (1.4, 18.6)]
N9_DRIFTL = [(4.8, 6.0), (3.6, 10.0), (2.0, 14.4), (0.0, 18.6), (-1.6, 19.6)]
N9_CURVED = [(4.8, 6.0), (5.2, 10.4), (4.2, 15.0), (2.0, 18.2),
             (-0.8, 19.4), (-3.0, 18.6)]
N9_DRIFTR = [(4.8, 6.0), (5.2, 10.2), (5.6, 14.6), (6.4, 18.6)]

BASE = {
 1: [("bare", [[(0, 0.3), (0, 19.7)]]),
     ("flag", [[(-2.6, 1.6), (0, 0.3), (0, 19.7)]]),
     ("fser", [[(-2.8, 1.7), (0, 0.3), (0, 18.8)],
               [(-2.4, 19.6), (0, 18.9), (2.4, 19.6)]]),
     ("euro", [[(-4.2, 2.6), (-0.4, 0.5)], [(-0.4, 0.5), (0.2, 19.0)],
               [(-3.0, 19.6), (0.2, 18.9), (3.0, 19.6)]]),
     ("leanR", [[(-3.2, 1.5), (-0.7, 0.3), (0.8, 19.7)]]),
     ("leanL", [[(1.4, 0.3), (-1.0, 19.7)], [(1.4, 0.3), (3.4, 1.9)]])],
 2: [("open", [[(-4.2, 2.4), (-1.6, 0.7), (2.2, 0.4), (5.0, 2.2), (5.6, 5.0),
                (4.8, 8.2), (3.0, 11.2), (0.0, 14.2), (-3.2, 17.0), (-6.0, 18.8)],
               [(-6.0, 18.8), (-3.5, 19.5), (0.5, 19.8), (4.0, 19.4), (7.0, 18.2)]]),
     ("longside", [[(-3.6, 2.6), (-1.4, 0.8), (2.4, 0.5), (5.2, 2.6), (5.8, 5.8),
                    (5.4, 9.6), (3.6, 12.8), (0.8, 15.6), (-2.8, 18.0), (-5.6, 19.2)],
                   [(-5.6, 19.2), (-3.0, 19.6), (1.0, 19.8), (4.4, 19.4), (7.2, 18.2)]]),
     ("loop", [[(-3.4, 2.8), (-1.4, 0.8), (2.2, 0.4), (5.0, 2.4), (5.6, 5.6),
                (3.6, 9.4), (0.4, 12.4), (-3.2, 15.0), (-5.4, 16.4)],
               [(-5.4, 16.4), (-7.2, 17.8), (-7.4, 19.4), (-5.2, 19.9),
                (-3.0, 19.2), (-2.8, 17.4), (-4.4, 16.1)],
               [(-3.0, 19.2), (6.2, 19.5)]]),
     ("flattop", [[(-5.4, 1.6), (1.0, 0.5), (5.4, 1.4), (5.8, 3.6), (4.4, 7.2),
                   (2.4, 10.6), (-0.4, 13.8), (-3.6, 16.8), (-6.2, 18.8)],
                  [(-6.2, 18.8), (-3.5, 19.6), (0.5, 19.9), (4.0, 19.5), (7.2, 18.3)]]),
     ("zigzag", [[(-5.0, 1.2), (4.6, 0.7), (5.4, 3.4), (2.4, 8.6), (-1.2, 13.4),
                  (-5.0, 17.8), (-6.0, 19.3), (-6.2, 19.5), (7.0, 19.4)]]),
     ("curly", [[(-3.2, 3.0), (-1.2, 0.8), (2.2, 0.4), (4.8, 2.2), (5.2, 5.2),
                 (3.4, 9.0), (0.2, 12.0), (-3.4, 14.6), (-5.4, 16.0)],
                [(-5.4, 16.0), (-7.2, 17.4), (-7.5, 19.3), (-5.2, 19.9),
                 (-3.0, 19.2), (-2.8, 17.3), (-4.5, 15.8)],
                [(-3.0, 19.2), (6.4, 19.5)]]),
     ("smallarc", [[(-2.8, 3.2), (-0.8, 1.2), (2.0, 1.0), (4.4, 2.8), (5.0, 5.6),
                    (4.2, 9.4), (2.4, 12.8), (-0.4, 15.8), (-3.6, 18.2), (-6.0, 19.3)],
                   [(-6.0, 19.3), (-3.2, 19.7), (0.8, 19.9), (4.2, 19.5), (7.0, 18.3)]]),
     ("wideflat", [[(-6.6, 1.8), (0.0, 0.6), (6.0, 1.2), (6.6, 3.2), (5.0, 6.6),
                    (3.0, 10.2), (0.2, 13.6), (-3.2, 16.8), (-6.0, 18.9)],
                   [(-6.0, 18.9), (-3.2, 19.7), (0.8, 19.9), (4.2, 19.5), (7.4, 18.2)]])],
 3: [("round", [[(-5.6, 2.2), (-2.0, 0.4), (2.6, 0.6), (5.4, 2.8), (6.6, 5.6),
                 (5.4, 7.8), (2.4, 9.2), (0.4, 9.4)],
                [(0.4, 9.6), (3.8, 10.2), (6.4, 12.0), (7.0, 14.8), (5.6, 17.4),
                 (2.4, 19.2), (-1.6, 19.8), (-5.4, 19.4), (-7.8, 18.0), (-8.4, 16.6)]]),
     ("flattop", [[(-6.2, 1.4), (0.0, 0.4), (6.0, 1.6), (6.4, 4.4), (4.6, 7.4),
                   (1.6, 9.2), (0.2, 9.4)],
                  [(0.2, 9.6), (3.8, 10.2), (6.4, 12.0), (7.0, 14.8), (5.6, 17.4),
                   (2.4, 19.2), (-1.6, 19.8), (-5.4, 19.4), (-7.8, 18.0), (-8.4, 16.6)]]),
     ("smalltop", [[(-4.6, 2.6), (-1.4, 0.7), (2.4, 1.0), (4.8, 3.2), (5.4, 6.0),
                    (3.4, 8.4), (0.6, 9.3)],
                   [(0.6, 9.5), (4.2, 10.4), (6.8, 12.6), (7.2, 15.6), (5.4, 18.0),
                    (1.8, 19.6), (-2.4, 19.9), (-6.0, 19.4), (-8.2, 17.8), (-8.8, 16.2)]]),
     ("bigbottom", [[(-5.2, 2.0), (-1.8, 0.5), (2.4, 0.7), (5.0, 2.8), (6.0, 5.6),
                     (4.6, 7.9), (1.8, 9.2), (0.2, 9.4)],
                    [(0.2, 9.6), (4.4, 10.4), (7.2, 12.8), (7.4, 16.0), (5.2, 18.4),
                     (1.6, 19.7), (-2.6, 19.9), (-6.4, 19.2), (-8.6, 17.4), (-9.0, 15.8)]]),
     ("pointed", [[(-6.2, 2.8), (-2.4, 0.5), (2.4, 0.7), (5.6, 3.2), (5.8, 6.0),
                   (2.8, 8.8), (0.2, 9.2)],
                  [(0.2, 9.4), (4.4, 10.4), (7.0, 12.8), (6.6, 16.0), (4.0, 18.8),
                   (0.2, 19.8), (-4.0, 19.6), (-7.2, 17.8), (-8.0, 16.0)]]),
     ("looptop", [[(-4.8, 8.2), (-5.6, 5.0), (-3.0, 1.6), (1.4, 0.6), (4.8, 2.4),
                   (5.6, 6.0), (3.6, 8.8), (0.6, 9.4)],
                  [(0.6, 9.6), (4.0, 10.4), (6.6, 12.4), (7.0, 15.2), (5.4, 17.6),
                   (2.2, 19.4), (-1.8, 19.8), (-5.6, 19.3), (-7.8, 17.8), (-8.4, 16.4)]]),
     ("opengap", [[(-5.0, 2.2), (-1.6, 0.5), (2.6, 0.7), (5.2, 3.0), (6.0, 5.8),
                   (3.2, 8.6), (0.8, 9.0)],
                  [(0.8, 10.2), (4.2, 10.8), (6.6, 12.8), (6.8, 15.6), (4.8, 18.2),
                   (1.4, 19.6), (-2.6, 19.9), (-6.2, 19.2), (-8.0, 17.6), (-8.6, 16.0)]]),
     ("flatwide", [[(-6.6, 1.0), (1.2, 0.4), (6.4, 1.4), (6.8, 4.2), (5.0, 7.4),
                    (1.8, 9.2), (0.0, 9.4)],
                   [(0.0, 9.6), (3.8, 10.4), (6.6, 12.6), (7.0, 15.4), (5.2, 18.0),
                    (1.8, 19.6), (-2.4, 19.8), (-6.4, 19.2), (-8.4, 17.4), (-9.0, 15.8)]])],
 4: [("closed", [[(-5.6, 10.8), (2.6, 0.5)], [(-5.6, 10.8), (6.0, 11.2)],
                 [(2.6, 0.5), (2.6, 19.7)]]),
     ("open", [[(-5.4, 10.6), (3.8, 0.4)], [(-5.4, 10.6), (6.2, 10.9)],
               [(3.2, 11.0), (3.2, 19.7)]]),
     ("highcross", [[(-6.2, 9.4), (4.2, 0.5)], [(-6.2, 9.4), (6.6, 9.8)],
                    [(4.0, 9.9), (4.0, 19.7)]]),
     ("leftbar", [[(-8.6, 0.5), (-8.2, 11.0)], [(-8.4, 11.0), (5.4, 11.6)],
                  [(2.8, 0.5), (2.8, 19.7)]]),
     ("leftopen", [[(-8.6, 0.5), (-8.2, 11.4)], [(-8.4, 11.4), (5.8, 12.0)],
                   [(3.6, 11.9), (3.6, 19.7)]]),
     ("closedlow", [[(-5.6, 12.6), (2.6, 0.5)], [(-5.6, 12.6), (6.0, 13.0)],
                    [(2.6, 0.5), (2.6, 19.7)]]),
     ("detached", [[(-1.2, 3.8), (-7.4, 12.6)], [(-7.4, 12.6), (6.8, 13.2)],
                   [(4.6, 0.4), (4.6, 19.7)]]),
     ("narrowleft", [[(-4.8, 11.8), (0.6, 0.5)], [(-4.8, 11.8), (5.6, 12.2)],
                     [(0.6, 0.5), (0.6, 19.7)]])],
 5: [("print5", [[(-4.2, 0.9), (3.8, 1.1)], [(-4.2, 0.9), (-4.6, 7.2)],
                 [(-4.6, 7.2), (-5.0, 11.4), (-3.8, 16.0), (-0.8, 19.0),
                  (2.6, 19.6), (5.0, 17.8), (5.6, 14.2), (5.2, 10.6),
                  (4.0, 7.8), (2.2, 7.0)]]),
     ("closedbowl", [[(-4.6, 1.0), (4.4, 1.2)], [(-4.6, 1.0), (-5.0, 6.6)],
                     [(-5.0, 6.6), (-5.4, 10.6), (-4.0, 15.4), (-1.0, 18.6),
                      (2.4, 19.5), (5.0, 17.4), (5.6, 13.6), (5.2, 9.8),
                      (3.8, 7.0), (1.2, 6.3), (-2.0, 6.6), (-3.8, 8.0)]]),
     ("S5", [[(-5.2, 1.0), (4.8, 1.4)],
             [(-1.8, 2.0), (-0.4, 5.2), (1.6, 9.2), (3.8, 12.8), (5.2, 15.6),
              (4.2, 18.2), (1.0, 19.5), (-3.0, 19.8), (-6.6, 19.2), (-9.2, 17.8)]]),
     ("sigma5", [[(1.6, 0.7), (5.8, 1.3)],
                 [(5.0, 1.5), (3.0, 4.6), (0.6, 8.2), (-1.8, 11.4)],
                 [(-1.8, 11.4), (-4.4, 13.2), (-6.6, 15.6), (-6.0, 18.4),
                  (-2.8, 19.7), (1.2, 19.5), (4.2, 17.6), (5.2, 15.0), (3.8, 13.2)]]),
     ("wide5", [[(3.8, 3.0), (10.4, 3.6)],
                [(10.4, 3.6), (6.2, 7.2), (1.6, 10.8), (-2.8, 13.4)],
                [(-2.8, 13.4), (-5.8, 15.6), (-5.4, 17.9), (-2.2, 19.4),
                 (1.8, 19.8), (5.4, 18.4), (6.6, 15.6), (6.2, 13.2), (4.4, 12.0)]]),
     ("angular", [[(-4.2, 1.0), (4.0, 1.1)], [(-4.2, 1.0), (-4.8, 7.2)],
                  [(-4.8, 7.2), (-5.4, 12.4), (-3.8, 17.2), (-0.6, 19.4),
                   (3.0, 18.2), (4.8, 14.2), (4.4, 9.8), (2.4, 7.4)]]),
     ("flagged", [[(-4.8, 1.1), (-3.0, 0.6), (4.2, 1.1)],
                  [(-4.8, 1.1), (-5.2, 7.0)],
                  [(-5.2, 7.0), (-5.5, 11.2), (-4.2, 15.8), (-1.2, 18.8),
                   (2.2, 19.5), (4.8, 17.6), (5.5, 14.0), (5.2, 10.0), (4.0, 7.2)]]),
     ("tinybowl", [[(-3.8, 0.9), (3.6, 1.1)], [(-3.8, 0.9), (-4.2, 9.0)],
                   [(-4.2, 9.0), (-4.6, 13.0), (-3.4, 17.2), (-0.8, 19.2),
                    (2.0, 19.6), (4.0, 17.8), (4.5, 14.8), (4.2, 12.2), (3.2, 10.4)]])],
 6: [("round", [[(1.8, 0.4), (-1.2, 0.9), (-3.8, 2.6), (-5.4, 5.6), (-5.9, 9.8),
                 (-5.5, 13.8), (-3.8, 17.2), (-0.6, 19.1), (2.6, 19.0),
                 (4.8, 17.0), (5.2, 14.0), (4.0, 12.0), (1.4, 11.4),
                 (-1.4, 12.2), (-3.2, 14.0), (-4.6, 16.4)]]),
     ("open", [[(2.0, 0.4), (-1.4, 1.0), (-4.2, 3.2), (-5.6, 6.6), (-6.0, 11.0),
                (-5.2, 15.2), (-3.2, 18.2), (-0.2, 19.5), (3.0, 19.2),
                (5.0, 17.2)]]),
     ("straightstem", [[(2.4, 0.4), (0.4, 0.9), (-1.8, 2.6), (-3.0, 5.8),
                        (-3.5, 10.0), (-3.2, 14.0), (-1.8, 17.2), (0.8, 19.2),
                        (3.6, 18.8), (5.0, 16.4), (4.8, 13.8), (3.2, 12.0),
                        (0.8, 11.4), (-1.4, 12.2), (-2.8, 13.8)]]),
     ("bigloop", [[(1.0, 0.5), (-2.0, 1.4), (-4.4, 4.0), (-5.4, 7.6),
                   (-5.6, 11.6), (-4.6, 15.4), (-2.4, 18.2), (0.6, 19.4),
                   (3.4, 18.8), (5.0, 16.6), (5.2, 13.8), (4.2, 11.8),
                   (1.8, 11.0), (-0.8, 11.4), (-2.8, 12.8), (-4.0, 14.6)]]),
     ("tallloop", [[(0.8, 0.4), (-2.4, 1.6), (-4.6, 4.2), (-5.6, 8.0),
                    (-5.8, 12.0), (-4.6, 15.8), (-2.2, 18.4), (0.8, 19.4),
                    (3.6, 18.8), (5.2, 16.6), (5.4, 13.6), (4.2, 11.6),
                    (1.6, 10.9), (-1.0, 11.4), (-3.0, 13.0), (-4.2, 15.0)]]),
     ("narrow", [[(2.0, 0.4), (-0.8, 1.0), (-3.0, 2.8), (-4.2, 6.0), (-4.6, 10.0),
                  (-4.0, 13.8), (-2.4, 17.0), (0.2, 19.0), (3.0, 18.8),
                  (4.6, 16.8), (4.9, 14.0), (3.8, 12.0), (1.4, 11.2),
                  (-1.0, 11.9), (-2.6, 13.6), (-3.6, 15.4)]]),
     ("tinyloop", [[(1.6, 0.4), (-1.4, 1.2), (-4.0, 3.4), (-5.4, 6.8),
                    (-5.8, 11.0), (-5.4, 14.4), (-4.4, 16.6), (-2.6, 18.0),
                    (0.0, 18.6), (2.4, 18.0), (3.6, 16.2), (3.8, 14.2),
                    (3.0, 12.8), (1.2, 12.4), (-0.8, 13.2), (-2.0, 14.6)]]),
     ("opentop", [[(0.4, 0.5), (-2.2, 2.6), (-4.2, 6.0), (-5.2, 10.0),
                   (-5.4, 13.4), (-4.4, 16.4), (-2.4, 18.6), (0.4, 19.5),
                   (3.0, 19.0), (4.6, 17.0), (5.0, 14.4), (4.2, 12.4),
                   (2.0, 11.6), (-0.6, 12.2), (-2.4, 13.6)]])],
 7: [("plain", [[(-5.0, 1.0), (5.2, 0.7)],
                [(5.2, 0.7), (2.2, 6.0), (-0.4, 12.0), (-2.6, 19.5)]]),
     ("crossbar", [[(-5.0, 1.0), (5.2, 0.7)],
                   [(5.2, 0.7), (2.2, 6.0), (-0.4, 12.0), (-2.6, 19.5)],
                   [(-3.4, 10.8), (3.0, 9.6)]]),
     ("serif", [[(-5.4, 3.2), (-4.8, 1.2), (-0.6, 0.5), (4.8, 0.9)],
                [(4.8, 0.9), (1.6, 6.4), (-0.8, 12.4), (-2.8, 19.6)]]),
     ("serifcross", [[(-5.4, 3.2), (-4.8, 1.2), (-0.6, 0.5), (4.8, 0.9)],
                     [(4.8, 0.9), (1.6, 6.4), (-0.8, 12.4), (-2.8, 19.6)],
                     [(-3.2, 11.0), (3.2, 9.8)]]),
     ("steep", [[(-4.4, 1.2), (4.6, 0.9)], [(4.6, 0.9), (-1.6, 19.6)]]),
     ("shortbar", [[(-3.2, 1.1), (5.2, 0.7)],
                   [(5.2, 0.7), (2.0, 6.2), (-0.6, 12.2), (-2.8, 19.5)]]),
     ("curvetop", [[(-5.6, 4.0), (-4.4, 1.4), (-0.4, 0.5), (4.8, 0.8)],
                   [(4.8, 0.8), (1.8, 6.4), (-0.6, 12.4), (-2.8, 19.6)]]),
     ("longbar", [[(-6.4, 1.0), (6.0, 0.7)],
                  [(6.0, 0.7), (2.6, 6.2), (-0.2, 12.2), (-2.6, 19.5)]])],
}

SLANT = {1: (-0.28, -0.12, 0.04, 0.20), 2: (-0.15, 0.0, 0.18),
         3: (-0.15, 0.0, 0.18), 4: (-0.15, 0.0, 0.15), 5: (-0.15, 0.0, 0.15),
         6: (-0.15, 0.0, 0.15), 7: (-0.15, 0.0, 0.20), 8: (-0.15, 0.0, 0.18),
         9: (-0.15, 0.0, 0.20), 0: (-0.12, 0.0, 0.15)}
WIDTHS = {c: ((1.6, 2.2) if c == 1 else (1.8, 2.5)) for c in range(10)}

EIGHT = [(2.2, 3.4), (2.6, 5.0), (3.0, 4.6), (3.0, 5.3), (3.4, 4.9),
         (3.4, 5.6), (3.8, 5.2), (3.8, 5.9)]
N9_TAILS = {"S": N9_STRAIGHT, "L": N9_DRIFTL, "C": N9_CURVED, "R": N9_DRIFTR}
NINE = [(rx, k) for rx in (3.8, 4.4) for k in ("S", "L", "C", "R")]
ZERO_RX = (4.7, 5.1, 5.5, 5.9, 6.3, 6.7, 7.1, 7.5)


def styles(c):
    """Exactly 48 (strokes, shear, width) triples for class c."""
    out = []
    if c == 0:
        base = [[ellipse_pts(0, 10, rx, 9.7)] for rx in ZERO_RX]
    elif c == 8:
        base = [eight(trx, brx) for trx, brx in EIGHT]
    elif c == 9:
        base = [[ellipse_pts(0.1, 5.2, rx, 3.6), N9_TAILS[k]] for rx, k in NINE]
    else:
        base = [s for _, s in BASE[c]]
    for strokes in base:
        for sh in SLANT[c]:
            for w in WIDTHS[c]:
                out.append((strokes, sh, w))
    assert len(out) == 48, (c, len(out))
    return out

# Hand-set rule table: pairs of classes whose ink patterns overlap strongly
# (classic handwriting confusables).  The final dense layer subtracts a small
# fraction GAMMA of each overlapping class's best-match score, which sharpens
# the margin between such pairs.  GAMMA and the pair list are hand choices.
PAIRS = [(0, 5), (0, 8), (1, 4), (1, 7), (2, 7), (3, 5), (3, 8), (3, 9),
         (4, 9), (5, 6), (5, 8), (6, 8), (7, 9), (8, 9)]
GAMMA = 0.22     # suppression strength (hand choice)
TAU = 1.0        # threshold on the normalised class score (hand choice)

# ------------------------------------------------------------------ layers --
NPAD = 3          # shift search radius: all windows still contain all ink
NVAR = 48         # templates per class


def build_weights():
    temps = []
    for c in range(10):
        for strokes, sh, w in styles(c):
            temps.append(make_template(strokes, w, sh))
    return temps


def main(folder):
    import os
    temps = build_weights()
    T = len(temps)
    assert T == 10 * NVAR, T
    W = np.stack(temps)[:, None, :, :].astype(np.float32)
    b = np.zeros(T, dtype=np.float32)
    # threshold gate: u = relu(s_normalised - TAU) fires only for strong classes
    gateW = np.eye(10, dtype=np.float32)
    gateb = np.full(10, -TAU, dtype=np.float32)
    # suppression: subtract GAMMA * (strong excess) of overlapping classes
    suppW = np.zeros((10, 10), dtype=np.float32)
    for (i, j) in PAIRS:
        suppW[i, j] = -GAMMA
        suppW[j, i] = -GAMMA
    net = {"layers": [
        {"name": "match", "op": "conv", "inputs": ["image"],
         "out": T, "kernel": [S, S], "stride": 1, "padding": NPAD},
        {"name": "best", "op": "maxpool", "inputs": ["match"],
         "size": [2 * NPAD + 1, 2 * NPAD + 1], "stride": [2 * NPAD + 1, 2 * NPAD + 1]},
        {"name": "grouped", "op": "reshape", "inputs": ["best"], "shape": [1, 10, NVAR]},
        {"name": "clsmax", "op": "maxpool", "inputs": ["grouped"],
         "size": [1, NVAR], "stride": [1, NVAR]},
        {"name": "smax", "op": "reshape", "inputs": ["clsmax"], "shape": [10]},
        {"name": "sz", "op": "reshape", "inputs": ["smax"], "shape": [1, 1, 10]},
        {"name": "sn", "op": "norm", "inputs": ["sz"]},
        {"name": "snflat", "op": "reshape", "inputs": ["sn"], "shape": [10]},
        {"name": "gate", "op": "dense", "inputs": ["snflat"], "out": 10},
        {"name": "u", "op": "relu", "inputs": ["gate"]},
        {"name": "supp", "op": "dense", "inputs": ["u"], "out": 10},
        {"name": "scores", "op": "add", "inputs": ["snflat", "supp"]},
    ]}
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(folder, "weights.npz"),
             **{"match.W": W, "match.b": b, "gate.W": gateW, "gate.b": gateb,
                "supp.W": suppW, "supp.b": np.zeros(10, dtype=np.float32)})
    print(f"wrote {T} templates ({T * S * S} weights) to {folder}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
