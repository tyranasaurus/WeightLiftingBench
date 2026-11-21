"""Data-free pen model, template rasterizer, and source of the recognizer."""
import sys
from pathlib import Path
import numpy as np

# A little pen language: M lifts the pen; L, Q and C draw line segments,
# quadratic Beziers and cubic Beziers. Coordinates are in pen-height units
# (10 units is an ascender), NOT coordinates in a 28 x 28 image.
models = [
# Zero: oval, egg-shaped, written past the closure, slight gap, angular oval.
[(5, 'M 5 0 C 1 -1 -1 3 0 7 C 0 11 4 12 6.5 8 C 9 4 8 0 5 0'),
 (4, 'M 3.5 0 C 1 0 0 2 0 5 C 0 8 1 10 3.5 10 C 6 10 7 8 7 5 C 7 2 6 0 3.5 0'),
 (3, 'M 4.8 1.5 Q 3.5 -1 1.5 1 Q -1 4 0.6 8 Q 2 12 5.2 9 Q 8 6 6.5 2.5 Q 6 0.5 4 0'),
 (2, 'M 4 0 Q 1 0 0 5 Q -1 9 2 10 Q 6 12 7 5 Q 8 0 5 0'),
 (2, 'M 5.5 0.8 Q 3 -1 1 2 L 0 6 Q -0.5 11 3 10 L 6 8 Q 8 4 6.5 1.5 Q 5 -0.8 3.7 0.6'),
 (1, 'M 4 2 Q 3.2 -0.8 1.8 1.2 Q -1 5 0.5 8.3 Q 2 11.5 4.3 9.5 Q 7.2 7.5 6.3 3 Q 6 0 4 0 Q 3.1 0 2.5 0.7')],
# One: straight downstroke, slightly bent, lead-in flag, optional base.
[(7, 'M 3 0 L 3 10'),
 (5, 'M 3 0 Q 3.9 4.5 3 10'),
 (4, 'M 2.7 0 Q 2.1 3.2 2.8 6 Q 3.3 8.5 3 10'),
 (2, 'M 1.8 1.6 L 3 0 Q 3.3 5 3 10'),
 (2, 'M 0 3.2 Q 1.8 2 3 0 L 3 10'),
 (1, 'M 0 4 L 3.3 0.2 Q 3.7 -0.4 3.6 1 L 3.3 10'),
 (1, 'M 1 2 L 3.5 0 L 3.5 9.8 M 0.5 10 L 6.5 10'),
 (1, 'M 3 0 L 3 9 Q 2.9 10 4.5 9.9'),
 (1, 'M 2.7 0 L 2.7 9.9 M 1 10 L 5.4 10')],
# Two: arched top, diagonal and base; also a loop where the pen turns
# around at the baseline, and a continuous cursive two with a high entry.
[(5, 'M 0 2.1 C 1.2 -1 5.9 -0.9 6.1 2.1 C 6.5 4.4 2.8 7.1 0.4 10 Q 3 9.4 7 10'),
 (4, 'M 0 2.5 Q 1.4 -0.7 4.2 0 C 8 1.2 5.7 4.6 3.2 7 L 0.4 9.8 Q 3 10.3 7.4 9'),
 (3, 'M 0.3 0.8 Q 5 -1.8 6 1.8 Q 6.8 4.6 0.6 9.5 Q 0 10.7 2 10 L 7.4 9.7'),
 (3, 'M 0.3 3 Q 0.7 -0.2 3.6 0 Q 7.8 0.5 5.6 4.2 Q 3.5 7.2 0 9.4 Q 2.8 9 4.8 10 Q 6.6 10.9 8.3 9.4'),
 (4, 'M 0.6 2.8 Q 1 -0.4 3.8 0 Q 6.8 0.3 5.6 3.9 Q 4.6 6.8 1.1 9.2 Q -1.6 11.1 -0.5 8.6 Q 0.1 7.4 2.7 8.3 Q 6.7 10.9 8.2 9.3'),
 (2, 'M -0.4 1.3 Q 3 -1.3 5.2 0.9 Q 8 4 1.5 9.3 Q -0.6 10.8 -0.4 9 Q -0.3 7.6 2.4 8.6 L 7 10'),
 (2, 'M 1.6 4.6 Q -0.6 2.8 1.1 0.9 Q 3.3 -1.2 5.4 0.9 Q 8 4.2 0.9 9.7 Q 4.6 9 7.8 10.1'),
 (1, 'M 1.5 5 Q -0.4 2.7 1.1 0.8 Q 3.5 -1.4 5 1.3 Q 7.1 5.1 1 9.2 Q -1 10.6 -0.2 8.5 Q 0.7 7 3.6 8.6 Q 6 10.5 8 9.5'),
 (1, 'M 0 1 L 4 0 Q 7 -0.5 6 2.8 L 0 9.5 L 7 10')],
# Three: two rightward lobes (vary the centre cusp), or straight top
# cross stroke and diagonal turning into the lower bowl.
[(5, 'M 0 1 Q 3.4 -1.1 5.6 0.6 Q 8.4 3.1 2.2 4.9 Q 7.4 4.7 6.5 7.9 Q 5.4 11.6 0 9'),
 (3, 'M 0 1.2 Q 3.3 -1 5.5 0.6 C 8.4 2.8 4.8 4.4 2.8 5 Q 7.8 5.1 5.9 8.8 Q 4.2 11.5 0.3 9.8'),
 (3, 'M 0 2 Q 2.7 -1.5 5.5 0.4 Q 8 2.3 3.6 4.8 Q 2.2 5.8 2.6 4.9 Q 5.4 3.3 6.8 5.9 Q 8.6 9.8 3.5 10 Q 1 10.3 0 8'),
 (3, 'M 0 0.5 Q 3.2 -0.4 6.2 0 L 2.2 4.7 Q 6.8 3.8 6.8 6.9 Q 7 10.5 3.1 10 Q 1.4 10.2 0.1 9'),
 (2, 'M 0 1.6 L 5.2 0 Q 7.5 -0.3 5.1 2.6 L 2.7 4.7 Q 6.7 4 6.8 7.1 Q 7 11 1.2 10'),
 (2, 'M 0.6 0 Q 5.7 -1 6 2 Q 6.2 3.5 3.7 4.9 Q 7.2 5.8 6.1 8.6 Q 5.2 11.8 -0.3 9.4'),
 (2, 'M 0.5 1 Q 4.3 -1.1 6.3 0.9 Q 8.1 3.6 2.5 5 Q 1 5.6 1.7 4.8 Q 3.5 3.6 5.7 5.2 Q 8.6 7.5 5.4 9.6 Q 2.5 11.5 -0.3 8.5'),
 (1, 'M 0 0.2 Q 3.3 -0.5 5 0.8 Q 7.6 3.2 3.5 4.8 Q 6.8 6.1 5.8 8.3 Q 4.9 10.5 0.7 10')],
# Four: open top, a U-like or square left arm; also a closed triangle.
[(5, 'M 1.4 0 Q 0.2 2.3 0 4.8 Q -0.5 6.4 1.4 6.1 L 6.2 5.3 M 5.8 0.2 Q 5.1 4.8 5.5 10'),
 (4, 'M 2 0 L 0 5.8 L 7.6 5.5 M 5.5 1 L 5.4 10'),
 (3, 'M 0.7 0.3 Q -0.9 4.8 0.6 5.4 Q 2.4 6.4 5.6 2.6 M 5.9 0.5 Q 5 5.8 5.8 10'),
 (3, 'M 1.7 0 Q 0 3.2 0.1 5 Q 0.2 7.5 3 5.9 Q 5 4.8 5.9 1 Q 5.3 6.2 6.2 10'),
 (2, 'M 3 0.1 L -0.2 6.4 Q 2.4 6 7.2 5.6 M 5.4 2.1 L 5.8 10'),
 (2, 'M 5.3 0 L -0.3 6.2 L 7.3 5.8 M 5.3 0 L 5.3 10'),
 (2, 'M 4.8 0.5 Q 1.5 3 0 5.9 L 6.6 5.8 M 4.7 0 Q 5 4.3 5.5 10'),
 (1, 'M 1 0.5 L 0 4.9 Q -0.7 6.9 2.5 6.4 L 6.2 5.7 M 6.2 -0.2 L 5 9 Q 4.7 10.6 6.2 10.2'),
 (2, 'M 0 0 Q -0.5 2.8 0.4 4.5 Q 1.5 6.7 5.3 4.6 M 5 0.7 Q 5 6.4 5.7 10')],
# Five: flat roof, left upright and lower bowl. Roof often added with a
# separate pen stroke, and the upstroke or the end can curl.
[(5, 'M 6.5 0 Q 2.9 0.5 1 0 L 0.3 4.6 Q 3.5 2.8 5.4 4.9 Q 8.4 8.1 4.5 9.8 Q 2.2 11 0 9.1'),
 (4, 'M 6.7 -0.5 Q 3.6 0.8 1.1 0.4 L 0.2 4.2 Q 3.4 3.4 5.4 5.4 Q 8.4 8.7 3.9 10 Q 1.2 10.8 -0.6 9.2'),
 (3, 'M 1.1 0.3 Q 0.4 1.9 0.2 3.9 Q 3.8 3.7 5.7 6 Q 8 9.1 3.5 10.2 Q 0.8 10.8 -0.7 8.7 M 1.5 0.5 Q 4 0.9 7.6 -0.4'),
 (3, 'M 0.7 0 L 0.2 4.3 Q 6.2 3.6 6.3 6.9 Q 6.4 10.3 1.3 10 M 2 0 Q 4 0.6 6.4 0'),
 (3, 'M 7 0 Q 4 1 1.7 0.5 Q -0.3 1.9 0 3.8 Q 0 4.5 2.6 4.5 Q 7 5.6 5.8 8.3 Q 4.7 11.4 -0.8 9.9'),
 (2, 'M 6.4 0 Q 4.5 -0.4 1.3 0.8 Q -1.3 2.3 1 4.1 L 4.8 6 Q 8.3 9.2 3 10 Q 0.5 10.7 -0.5 8.6'),
 (2, 'M 6.4 0 L 1.1 0.2 L 0 4.7 Q 5.3 2.4 5.9 6.4 Q 6.7 10.6 1.7 10 Q 0 10 -0.5 8.5'),
 (1, 'M 1.6 1.3 L 0.3 4.8 C 6.9 2.1 8.4 8.5 4.3 9.9 Q 0.5 11.8 -0.3 8.6 M 0.5 0 Q 3.7 1.1 6.8 0'),
 (1, 'M 6.7 0.4 Q 1.2 -1 0.5 1.6 Q -0.7 3.6 3 5 Q 8.9 7.8 4.5 10 Q 2.2 11 -0.2 9.5')],
# Six: descending hook wrapping the bowl from below, either closing on
# the upright or crossing it; also a long straight entry and open bowls.
[(5, 'M 5.4 0 C 2.9 0.4 -0.5 4.1 0.4 7.8 Q 1.1 11.2 4.6 9.8 C 8 8.3 6.6 4.2 3.9 4.8 Q 1.8 5.1 0.4 7'),
 (4, 'M 4.7 0 Q -0.5 3.6 0.3 7.6 Q 0.6 10.4 3.4 10 Q 6.6 10 6.4 7.2 Q 6.2 4.8 3.5 5.4 Q 1.7 5.8 0.8 7.4'),
 (3, 'M 3.8 0 L 0.6 5.6 C -1.9 11.7 6.2 11.2 6 7.7 Q 5.7 4.6 2.9 5.9 Q 1.6 6.4 0.3 8'),
 (3, 'M 6.2 1.2 Q 4.5 -1.6 2.4 0.9 Q -0.6 4.2 0 7.6 Q 0.8 11.4 4.6 9.8 Q 7.5 8.4 6.4 6 Q 4.4 2.9 0.3 6.9'),
 (2, 'M 4 0 Q 0 3.7 0.1 7.6 Q 0.5 11.1 3.9 10 Q 7.7 8.8 6.2 6.3 Q 5 4.8 2.7 6.1'),
 (2, 'M 3.9 0 Q -0.5 3.9 0.4 8.1 Q 1.4 11.9 5.5 9.4 Q 8.4 7.1 5.7 5.5 Q 2.8 4.6 0.3 8.1 L -0.7 9.3'),
 (1, 'M 4.7 0 Q 0.7 -0.3 0.3 4.6 Q -0.8 10.6 3.2 10 Q 7 10.3 6.6 7.4 Q 6.3 4.3 2.9 5.2 Q 1 6.2 0.4 7.4'),
 (1, 'M 2.4 0 Q 0.8 3.2 0.3 6.7 Q -0.6 10.3 2.7 10 Q 7 10 6.8 7.3 Q 6.3 5.5 4.4 6.2 L 1.1 8.1')],
# Seven: straight or bowed top, sharp change into descending stroke;
# crossbar, entry tick, and curved ending are separate style families.
[(5, 'M 0 0.6 Q 2.4 -0.3 6.4 0.4 Q 3.9 4.5 2.4 10'),
 (4, 'M 0 0 Q 3.6 1.3 6.7 0.6 L 2.8 10'),
 (3, 'M 0 1.1 Q 2.6 -0.4 5.9 0.1 Q 7.1 0 6.2 1.7 L 2.5 10'),
 (2, 'M -0.2 2.3 L 0.2 0 Q 3.4 0.7 6.9 0.3 Q 3.6 6.7 3.1 10'),
 (2, 'M 0 0.3 Q 2.8 -0.2 6.5 0.2 L 2.7 10 M 1.3 5.2 L 6.3 5.1'),
 (2, 'M -0.3 1.1 Q 1.1 -0.5 3.8 0.2 L 6.7 0.8 Q 5.2 3.7 4.3 6.2 Q 3.3 9 2.1 10.2'),
 (1, 'M 0 0.5 L 5.8 0 L 2.3 10 M 1.6 5.2 Q 3.8 5.7 5.6 5.6'),
 (1, 'M 0 0.2 Q 1.7 1.5 4.8 0.5 L 6.7 0 Q 3.6 5.7 3.7 9.2 Q 3.7 10.4 2.7 9.7'),
 (1, 'M 0 1.4 Q 1 -0.7 3.4 0.6 L 5.5 1.2 Q 6.5 1.6 7.4 0.4 L 3.2 10')],
# Eight: two lobes sharing a waist, pen crossing in either direction;
# single-stroke S plus return diagonal, and ink overlap at the join.
[(4, 'M 3.5 4.9 C -1.4 2.8 0.3 -0.6 3.8 0 C 7.3 0.3 7.2 3 3.5 4.9 C -2.2 7.8 -0.2 11 3.7 10 C 8.2 9.8 7.8 7.2 3.5 4.9'),
 (4, 'M 5.6 0.8 Q 2 -1.6 0.9 1 Q -0.4 3.3 3.5 5.3 C 8.1 8 7.1 10.6 3.1 10 Q -1.7 9.6 1 6.9 L 5.5 2.9 Q 7.6 0.2 4.6 0'),
 (3, 'M 1.6 0.4 C -0.8 2.4 2.4 4.1 4.3 5.4 Q 8.8 9.9 4.2 10 Q -1.2 10.9 0.6 7.4 Q 1.3 5.9 4.4 3.9 Q 8 0.8 4.3 0 Q 2.5 -0.4 1.4 1.1'),
 (3, 'M 4.5 0.6 Q 1.2 -1.1 0.8 1.7 Q 0.7 3.2 3.3 5 Q 7.2 7.2 6.4 9.2 Q 5.4 11.1 2 9.8 Q -0.9 8.8 1.2 6.4 L 4.8 3.2 Q 7.6 0 3.1 0.3'),
 (2, 'M 2.7 4.7 Q -1.4 2.5 0.7 0.8 Q 3.2 -1.2 5.8 0.8 Q 8 2.9 3.1 4.9 Q -1.9 7.6 0.9 9.5 Q 4.3 11.6 6.3 9.2 Q 7.8 7.1 2.7 4.7'),
 (2, 'M 3.1 4.7 C -0.5 4.8 -0.7 0.1 3 0 C 6.8 -0.4 7.2 4.5 3.1 4.7 C -0.7 4.7 -1.7 9.7 3.4 10 C 8.2 10.4 7.5 5.1 3.1 4.7'),
 (2, 'M 4.8 0.3 Q 1.8 -0.9 1 1 Q 0.2 2.6 3.7 5.3 Q 9.4 10.4 4.2 10 Q -1 10.2 0.7 7.1 Q 1.7 5.8 4.3 3.8 Q 6.6 2 6 1.2'),
 (1, 'M 1.5 0.7 Q 6.5 -1.8 6.3 1.7 Q 5.7 3.4 2.3 5.3 Q -1.8 8.2 1.4 9.8 Q 5.5 11.6 6.7 8.4 Q 7.1 6.6 3.8 4.8 Q -0.5 2.7 1.5 0.7'),
 (1, 'M 1 0.6 Q 4.3 -0.5 5 1.3 Q 6.3 3 2.7 4.9 Q -2.6 8.8 1.6 10 Q 4.9 11.3 6.4 9.1 Q 8 7.3 3.2 4.3 Q 0.1 2.3 0.5 0.9')],
# Nine: a closed bowl above a long downstroke, a partly open cup,
# a tail that curls back, or a short connection to a separate stem.
[(5, 'M 5.9 1.3 Q 3.6 -1.4 1.4 0.8 C -1.2 2.7 -0.7 5.3 1.6 5.4 Q 4.1 5.4 6 1.1 Q 5.4 4.6 5.6 10'),
 (4, 'M 5.3 0.8 C 2.9 -1.3 -0.4 1.1 0.1 3.6 Q 0.5 6.2 3.2 4.8 Q 5.5 3.2 5.6 0.5 L 5.3 10'),
 (3, 'M 5.7 1.2 Q 3.6 -1 1.5 0.7 Q -1.1 2.6 0.6 4.7 Q 3 6.4 5.5 2.6 Q 6.8 0.1 6.3 2.3 Q 5.5 6.3 5 10'),
 (3, 'M 5.7 0.9 Q 2.6 -1.4 0.6 1.5 Q -1.2 4.6 1.8 5.1 Q 4.8 5.7 5.8 1.1 Q 7.4 7 4.9 9.5 Q 3.7 10.8 2.3 9.9'),
 (2, 'M 4.1 0.5 Q 2.2 -1 0.7 1.1 Q -1.2 3.9 1.1 5 Q 3.6 6.4 5.8 1 Q 5.7 6.1 6.7 10'),
 (2, 'M 4.8 0.4 Q 0.8 -1.1 0 2.2 Q -0.5 5.3 2.8 4.9 Q 6.1 4.5 5.7 1.8 Q 5.6 -0.1 4.6 0 M 5.7 1.3 Q 6.3 6 4.3 10'),
 (2, 'M 4.3 0 Q 0.9 0 0.1 3.2 Q -0.7 5.8 2 5.2 Q 4.8 4.3 6.2 0.8 L 4.9 10'),
 (1, 'M 5.8 1 Q 2.4 -1.4 0.4 1.9 Q -1.8 6.1 2.7 5.7 Q 5.4 4.9 5.7 1.4 Q 7 6.8 3.7 10'),
 (2, 'M 5.3 1.1 Q 3.2 -0.8 1.4 0.5 Q -1.2 2.5 0.3 4.8 Q 2 7.2 5.8 2 M 5.8 1 Q 5.8 6 6.4 10')]
]

# Draw each template independently. No font files or drawing libraries.
def decode(text):
    words = iter(text.split())
    result = []
    for cmd in words:
        pts = np.array([(float(next(words)), float(next(words)))
                        for j in range({'M':1, 'L':1, 'Q':2, 'C':3}[cmd])])
        if cmd == 'M':
            result.append([])
            cur = pts[0]
        else:
            pts = np.vstack([cur, pts])
            degree = len(pts)-1
            length = np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()
            t = np.linspace(0,1, max(3, int(length*4)+1))[:,None]
            a = 1-t
            if degree == 1:
                line = a*pts[0] + t*pts[1]
            elif degree == 2:
                line = a*a*pts[0] + 2*a*t*pts[1] + t*t*pts[2]
            else:
                line = a**3*pts[0]+3*a*a*t*pts[1]+3*a*t*t*pts[2]+t**3*pts[3]
            result[-1].append(line)
            cur = pts[-1]
    return [np.concatenate(stroke) for stroke in result]

# Fractional lattice used by the antialiased round pen nib.
dy, dx = np.mgrid[-4:5,-4:5].reshape(2,1,-1)
rng = np.random.default_rng(47021)

def draw(strokes):
    u = rng.uniform
    lean = u(-.62,.43)
    width = np.exp(u(-.46,.37))
    levels = 5 + u(-1.5,1.5)
    shift = u(-1.7,1.7)
    bend = u(-1.25,1.25)
    pinch = u(-.44,.44)
    nod = u(-.30,.30)
    # Smooth local displacements of a notional rubber writing surface.
    elastic = rng.normal(0,.46,(2,5,5))
    radius = np.exp(u(np.log(.39), np.log(1.22)))
    taper = u(.07,.37)
    phase = u(0,6.28)
    nib = u(.86,1.17)
    result=[]; radii=[]
    for points in strokes:
        # B-spline-like smooth perturbation, shared at crossing points.
        x, y = points.T
        gx = np.clip((x+2)/3,0,3.999)
        gy = np.clip((y+2)/3.6,0,3.999)
        ix, iy = gx.astype(int),gy.astype(int)
        gx, gy = gx-ix,gy-iy
        gx,gy = gx*gx*(3-2*gx),gy*gy*(3-2*gy)
        field = ((1-gy)*((1-gx)*elastic[:,iy,ix]+gx*elastic[:,iy,ix+1])
                +gy*((1-gx)*elastic[:,iy+1,ix]+gx*elastic[:,iy+1,ix+1]))
        # Change the relative sizes and alignment of the bowls, not just
        # the global aspect. Small stroke-wise offsets for detached pieces.
        t = np.clip(y/10,0,1)
        w = np.sin(np.pi*t)
        xx = width*((x-3.3)*(1+(2*t-1)*pinch)+w*bend)+field[0]
        yy = y + shift*np.where(y<levels,y/levels,(10-y)/(10-levels)) + nod*(x-3.3)+field[1]
        xx += lean*(yy-5)
        if len(strokes)>1:
            xx = xx + u(-.48,.48); yy = yy + u(-.46,.46)
        result.append(np.stack((xx,yy),axis=1))
        progress=np.linspace(0,1,len(x))
        rad = radius*(1+.12*np.sin(6*progress+phase)-taper*np.exp(-progress*13)-taper*np.exp((progress-1)*13))
        radii.append(rad)
    p = np.concatenate(result); r=np.concatenate(radii)
    scale=19.4/np.max(np.max(p+r[:,None],axis=0)-np.min(p-r[:,None],axis=0))
    p = (p-(p.max(axis=0)+p.min(axis=0))*.5)*scale + 13.5 + u(-.48,.48,(1,2))
    r=r[:,None]*scale
    x,y=p.T[:,:,None]
    xx = np.rint(x).astype(np.int32)+dx
    yy = np.rint(y).astype(np.int32)+dy
    d = np.sqrt(((xx-x)/nib)**2 + ((yy-y)*nib)**2)
    coverage = np.clip(.5+(r-d)/1.15,0,1).astype(np.float32)
    im=np.zeros(784,np.float32)
    np.maximum.at(im,(np.clip(yy,0,27)*28+np.clip(xx,0,27)).ravel(),coverage.ravel())
    return im.reshape(28,28)

# Shared input/template normalization & descriptors, all reductions are
# spatial reductions of a SINGLE image. First axis just vectorizes work.
features='''import numpy as np

axis = np.arange(28, dtype=np.float32)
rows = axis.reshape(1,28,1)
cols = axis.reshape(1,1,28)

def normalize(v):
    v = v.astype(np.float32)
    v = v / np.maximum(v.max((1,2),keepdims=True),.001)
    m = np.maximum(v.sum((1,2),keepdims=True),.001)
    cx = (cols*v).sum((1,2),keepdims=True)/m
    cy = (rows*v).sum((1,2),keepdims=True)/m
    ry = rows-cy
    rx = cols-cx
    lean = (v*rx*ry).sum((1,2),keepdims=True)/np.maximum(1,(v*ry**2).sum((1,2),keepdims=True))
    lean = np.clip(lean,-1.25,1.25)
    # Bounding rectangle after removing slant, excluding faint edge pixels.
    mark = v>.17
    xx = rx-lean*ry
    xl = np.min(np.where(mark,xx,40),axis=(1,2),keepdims=True)
    xr = np.max(np.where(mark,xx,-40),axis=(1,2),keepdims=True)
    yt = np.min(np.where(mark,ry,40),axis=(1,2),keepdims=True)
    yb = np.max(np.where(mark,ry,-40),axis=(1,2),keepdims=True)
    w = np.maximum(xr-xl+1,1)
    h = np.maximum(yb-yt+1,1)
    # Partial, not full, aspect normalization: a bare 1 stays narrow.
    scale = h/20*np.clip((w/h/.68)**.65,.61,1.57)
    sy = (rows-13.5)*h/20 + (yt+yb)*.42
    sx = (cols-13.5)*scale + (xl+xr)*.42 + lean*sy + cx
    sy = sy+cy
    ix = np.floor(sx).astype(np.int32)
    iy = np.floor(sy).astype(np.int32)
    fx = (sx-ix).astype(np.float32)
    fy = (sy-iy).astype(np.float32)
    b = np.arange(len(v))[:,None,None]
    l = np.clip(ix,0,27)
    r = np.clip(ix+1,0,27)
    a = np.clip(iy,0,27)
    d = np.clip(iy+1,0,27)
    n = ((1-fy)*((1-fx)*v[b,a,l]+fx*v[b,a,r])+fy*((1-fx)*v[b,d,l]+fx*v[b,d,r]))
    n *= (sx>-.5)&(sx<27.5)&(sy>-.5)&(sy<27.5)
    return n, w[:,0,0]/h[:,0,0]

def unit(v):
    return v/np.maximum(.0001,np.sqrt((v*v).sum(1,keepdims=True)))

def average(v, step):
    s = v.shape
    return v.reshape(s[0],s[1]//step,step,s[2]//step,step).mean((2,4))

def smooth(v):
    a = np.pad(v,((0,0),(1,1),(1,1)))
    h = (a[:,:,:-2] + 2*a[:,:,1:-1] + a[:,:,2:])*.25
    return (h[:,:-2] + 2*h[:,1:-1] + h[:,2:])*.25

def enclosed(n, level):
    # Fill from the outside with horizontal and vertical sweeps. Regions
    # that cannot see the exterior are enclosed counters (white spaces).
    wall = n>level
    ix = np.arange(28,dtype=np.int16).reshape(1,1,28)
    iy = ix.reshape(1,28,1)
    lw = np.maximum.accumulate(np.where(wall,ix,-2),axis=2)
    rw = np.minimum.accumulate(np.where(wall,ix,29)[:,:,::-1],axis=2)[:,:,::-1]
    tw = np.maximum.accumulate(np.where(wall,iy,-2),axis=1)
    bw = np.minimum.accumulate(np.where(wall,iy,29)[:,::-1,:],axis=1)[:,::-1,:]
    reach = (lw<0)|(rw>27)|(tw<0)|(bw>27)
    for turn in range(6):
        le = np.maximum.accumulate(np.where(reach,ix,-3),axis=2)
        ri = np.minimum.accumulate(np.where(reach,ix,30)[:,:,::-1],axis=2)[:,:,::-1]
        reach = (le>lw)|(ri<rw)
        up = np.maximum.accumulate(np.where(reach,iy,-3),axis=1)
        dn = np.minimum.accumulate(np.where(reach,iy,30)[:,::-1,:],axis=1)[:,::-1,:]
        reach = (up>tw)|(dn<bw)
    return (~reach)&(~wall)

def descriptors(ink):
    n, aspect = normalize(ink)
    a = smooth(n)
    b = smooth(smooth(a))
    mass = unit(average(b,2).reshape(len(n),196))
    # Signed edge directions: a dark counter has the opposite polarity
    # from a blob of ink. Soft bins avoid discontinuities on diagonals.
    gradx = np.roll(a,-1,2)-np.roll(a,1,2)
    grady = np.roll(a,-1,1)-np.roll(a,1,1)
    mag = np.sqrt(gradx**2+grady**2)
    dirs = []
    for i in range(8):
        angle = i*np.pi/4
        resp = np.maximum(0,gradx*np.float32(np.cos(angle))+grady*np.float32(np.sin(angle))-.55*mag)
        dirs.append(average(smooth(resp),4))
    hv = np.sqrt(np.stack(dirs,axis=-1)+.00002)
    hv = unit(hv.reshape(len(n),392))
    feature = np.concatenate((mass*np.float32(.62),hv*np.float32(.785)),axis=1)
    holes = (enclosed(n,.2).astype(np.float32)+enclosed(n,.48).astype(np.float32))*.5
    def region(mask):
        return (holes*mask).sum((1,2))
    top = region(rows<13.5)
    bot = region(rows>13.5)
    # Apertures have location as well as area. Saturate large areas: the
    # existence of a loop is better evidence than its precise area.
    topology = np.stack([1-np.exp(-top/13),1-np.exp(-bot/13),
                         1-np.exp(-(top+bot)/16),
                         1-np.exp(-region((rows>=11)&(rows<=16))/12)],axis=1)
    metadata = np.concatenate((np.log(np.maximum(aspect,.02))[:,None],topology),axis=1)
    return feature, metadata, n
'''

recognizer='''
# Kernels used for symmetric matching of local stroke directions. A line
# detector takes a second difference across the line, at two radii. Small
# direction and position errors are discounted rather than counted twice.
def fine_maps(v):
    a = smooth(v)
    halo = np.pad(a,((0,0),(4,4),(4,4)))
    def at(y,x):
        return halo[:,4+y:32+y,4+x:32+x]
    def diff(y,x):
        return a-(at(y,x)+at(-y,-x))*.5
    vertical = np.maximum(diff(0,2),diff(0,3)*.85)
    horiz = np.maximum(diff(2,0),diff(3,0)*.85)
    diag = np.maximum(diff(1,-1)*1.35,diff(2,-2))
    other = np.maximum(diff(1,1)*1.35,diff(2,2))
    oriented = [vertical,other,horiz,diag]
    responses = []
    fields = []
    for i in range(4):
        stroke = np.maximum(0,oriented[i]-.5*np.maximum(0,oriented[(i+2)%4]))
        responses.append(average(stroke,2))
    for i in range(4):
        p = np.maximum(responses[i],.75*np.maximum(responses[(i+1)%4],responses[(i+3)%4]))
        p = np.maximum(p,.46*responses[(i+2)%4])
        # A capped chamfer transform (quadratic penalty in the first cell,
        # linear tails). Distance to any good nearby line response.
        a = np.pad(p,((0,0),(3,3),(3,3)))
        field = p
        for y in range(-3,4):
            for x in range(-3,4):
                rr=x*x+y*y
                if rr>0 and rr<=10:
                    field=np.maximum(field,a[:,3+y:17+y,3+x:17+x]*np.float32(np.exp(-.24*rr)))
        fields.append(field)
    response = np.stack(responses,axis=1)
    field = np.stack(fields,axis=1)
    # Normalize each image by itself. Clipping above a typical strong
    # ridge keeps pen pressure from dominating coverage of the strokes.
    s = np.maximum(.015,response.max((1,2,3),keepdims=True)*.65)
    return np.minimum(response/s,1), np.minimum(field/s,1)
'''

inference='''
coarsebank = CONSTANTS['coarse'].astype(np.float32)/32767
meta = CONSTANTS['meta']
labels = CONSTANTS['labels']
count = len(labels)//10

def recognize(chunk):
    f, m, n = descriptors(chunk)
    # Fixed row count even for a singleton call: the matrix multiplication
    # uses the same kernel irrespective of surrounding images in the batch.
    q = np.zeros((32,f.shape[1]),np.float32)
    q[:len(f)] = f
    sims = (q @ coarsebank.T)[:len(f)]
    # Coarse hypothesis cost. Metadata is used gently: a break caused by
    # light pen pressure must not turn an eight irrevocably into a three.
    cost = 1-sims
    for i,weight in enumerate((.019,.011,.011,.014,.008)):
        delta = np.abs(m[:,i,None]-meta[None,:,i])
        cost += weight*np.minimum(delta,1.5)**1.45
    few = 8
    cc = cost.reshape(len(f),10,count)
    pick = np.argpartition(cc,few-1,axis=2)[:,:,:few] + np.arange(10)[None,:,None]*count
    testR,testF=fine_maps(n)
    testR=testR.reshape(len(n),784)
    testF=testF.reshape(len(n),784)
    result=np.zeros(len(f),dtype=np.int64)
    for j in range(len(f)):
        idx=pick[j].reshape(-1)
        r = CONSTANTS['ridge'][idx].astype(np.float32)*(1/255)
        g = CONSTANTS['near'][idx].astype(np.float32)*(1/255)
        # Distance from an input ridge to the template and vice versa;
        # subtract each image's own unmatched cost (a soft Dice distance).
        e = (r*(g-testF[j])+testR[j]*(testF[j]-g)).sum(1)
        d = e/np.maximum(1,r.sum(1)+testR[j].sum())
        total=(cost[j,idx]+.22*np.maximum(-.015,d)).reshape(10,few)
        ranked=np.sort(total,axis=1)
        # Similarity to several INDIVIDUAL independent renders, very
        # localized to the best explanation, not a synthesized prototype.
        value=.69*ranked[:,0]+.21*ranked[:,1]+.10*ranked[:,2]
        result[j]=np.argmin(value)
    return result

def predict(x):
    result=np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),32):
        result[start:start+32]=recognize(x[start:start+32,0])
    return result
'''

def main():
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(features+recognizer+inference)
    env={}
    exec(features+recognizer,env)
    per_class=4096
    total=10*per_class
    bank=np.empty((total,588),np.int16)
    meta=np.empty((total,5),np.float32)
    ridge=np.empty((total,784),np.uint8)
    near=np.empty_like(ridge)
    for label,collection in enumerate(models):
        patterns=[decode(s) for w,s in collection]
        choices=[j for j,(w,s) in enumerate(collection) for rep in range(w)]
        for start in range(0,per_class,64):
            lo=label*per_class+start
            section=slice(lo,lo+64)
            ims=np.stack([draw(patterns[choices[k%len(choices)]]) for k in range(start,start+64)])
            f,m,n=env['descriptors'](ims)
            bank[section]=np.rint(f*32767).astype(np.int16)
            meta[section]=m
            r,g=env['fine_maps'](n)
            ridge[section]=np.rint(r.reshape(64,784)*255).astype(np.uint8)
            near[section]=np.rint(g.reshape(64,784)*255).astype(np.uint8)
    np.savez(out/'constants.npz',coarse=bank,meta=meta,ridge=ridge,near=near,
             labels=np.repeat(np.arange(10,dtype=np.int8),per_class))

if __name__=='__main__':
    main()
