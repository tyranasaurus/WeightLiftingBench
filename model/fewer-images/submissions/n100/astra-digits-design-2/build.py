"""A geometric recognizer. No input files: syntax in this file is the pen alphabet.
Run: python3 build.py OUTPUT_DIRECTORY
"""
from pathlib import Path
import json
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows

# Paths are in pen-paper coordinates (x to right, y down). Q and C denote
# quadratic and cubic Beziers. New M commands lift and reposition the pen.
# Duplicate a style in this list by giving its sampling weight, not by
# measuring how often particular shapes occur.
PENS = [
    [ # 0: elliptical, oblong, cursive with a slight overlap, angular
        (6, 'M 52 3 C 23 -1 11 25 12 59 C 12 84 23 100 46 99 C 70 99 85 72 87 40 C 89 16 75 -1 52 3'),
        (3, 'M 52 2 C 33 0 22 14 18 36 L 13 68 C 10 85 20 98 38 99 C 60 102 75 87 81 60 L 88 30 C 92 11 71 -4 52 2'),
        (2, 'M 67 12 C 50 -3 34 7 22 25 C 7 49 4 83 23 94 C 42 110 72 97 83 71 C 97 40 89 5 68 2 Q 55 1 47 17'),
        (1, 'M 58 0 Q 31 2 17 39 L 8 76 Q 8 100 38 100 Q 69 100 84 63 L 93 26 Q 96 1 58 0'),
    ],
    [ # 1: plain downstroke, bowed, small and large flag, serif
        (7, 'M 51 0 L 49 100'),
        (3, 'M 53 0 Q 45 48 52 100'),
        (2, 'M 49 0 Q 60 43 49 100'),
        (3, 'M 35 17 L 53 0 L 50 100'),
        (2, 'M 15 33 Q 31 25 55 1 L 50 100'),
        (1, 'M 12 32 L 55 0 L 52 100 M 18 99 Q 49 98 83 100'),
        (1, 'M 45 0 L 48 97 Q 51 99 63 94'),
    ],
    [ # 2: arched head followed by diagonal, looped foot or straight foot
        (5, 'M 9 22 C 24 -2 58 -9 74 9 C 94 35 57 57 16 91 Q 7 100 25 98 L 92 96'),
        (3, 'M 12 29 C 15 0 55 -7 67 7 C 87 26 60 56 27 86 L 15 98 Q 42 90 69 97 Q 80 100 97 91'),
        (3, 'M 10 15 Q 34 -4 55 1 C 83 4 78 33 62 52 L 15 99 Q 45 88 68 95 L 92 97'),
        (3, 'M 13 28 C 5 13 32 -3 53 0 C 89 3 86 29 62 56 C 39 86 5 109 6 92 C 7 77 38 82 60 93 Q 81 103 98 92'),
        (2, 'M 15 19 Q 41 -9 60 3 C 85 21 72 44 46 64 C 14 84 0 97 18 100 C 38 105 50 83 68 85 Q 83 85 94 93'),
        (2, 'M 26 42 C 3 23 24 0 45 0 C 83 -1 81 38 57 61 L 21 98 Q 48 96 90 98'),
        (1, 'M 10 10 Q 36 -5 55 3 C 80 17 72 43 54 66 L 30 96 Q 44 98 59 91 L 99 69'),
        (1, 'M 30 29 C 13 41 9 18 23 7 C 45 -9 69 -1 72 22 Q 77 43 44 73 Q 23 91 15 100 Q 52 93 92 96'),
    ],
    [ # 3: two open bowls; sometimes an angular top or a retraced waist
        (5, 'M 13 14 C 33 -2 63 -7 76 6 C 91 24 67 43 40 49 C 72 39 99 58 83 82 C 70 103 36 108 12 90'),
        (3, 'M 11 10 Q 42 -4 68 2 Q 96 12 66 35 L 39 50 C 85 36 99 61 76 87 Q 52 110 13 94'),
        (2, 'M 9 14 Q 41 6 85 0 L 38 46 C 61 36 90 48 85 71 Q 77 106 28 99 L 12 92'),
        (2, 'M 16 15 C 41 -10 83 -4 79 18 C 76 37 39 58 32 51 C 29 44 64 39 79 50 C 106 75 63 108 29 99 Q 15 96 8 88'),
        (2, 'M 16 23 C 37 -5 79 -10 81 10 Q 83 30 47 49 Q 81 45 88 64 C 95 88 52 112 24 95 L 15 83'),
        (1, 'M 14 4 Q 55 -5 82 5 L 42 48 Q 91 41 87 63 Q 84 86 32 100'),
        (1, 'M 12 12 Q 42 -6 73 3 Q 96 17 49 44 Q 40 51 57 49 C 95 48 96 80 70 93 Q 48 104 18 97'),
    ],
    [ # 4: open chair, joined upright, and closed triangle
        (5, 'M 25 0 Q 17 28 9 55 Q 6 63 19 63 L 90 59 M 70 6 Q 66 46 66 100'),
        (4, 'M 27 0 Q 6 40 10 55 C 14 69 50 54 73 35 M 77 6 Q 67 47 66 100'),
        (3, 'M 29 0 Q 7 36 13 47 Q 23 61 70 43 M 76 3 L 65 100'),
        (2, 'M 19 3 L 12 52 Q 11 61 27 58 L 75 48 M 76 0 L 71 100'),
        (2, 'M 24 0 Q 14 52 22 58 Q 41 72 73 35 L 78 9 L 70 100'),
        (3, 'M 72 0 L 7 64 L 95 62 M 72 0 L 67 100'),
        (2, 'M 51 0 L 8 60 L 90 56 M 71 17 L 64 100'),
        (1, 'M 25 0 C 20 23 6 42 18 53 C 36 64 65 35 77 9 Q 73 57 67 100'),
    ],
    [ # 5: cap, left downstroke and lower bowl (separate cap is common)
        (5, 'M 85 2 Q 57 7 26 4 L 16 47 C 41 26 81 33 84 58 C 90 87 57 108 26 97 Q 13 93 8 84'),
        (3, 'M 87 1 Q 55 2 29 7 Q 18 25 19 44 C 51 29 83 43 81 66 C 79 93 41 105 13 94'),
        (3, 'M 27 8 Q 18 29 18 46 C 38 37 63 32 78 45 C 105 71 58 110 29 98 Q 8 94 10 77 M 18 5 Q 43 11 88 0'),
        (2, 'M 88 0 Q 55 5 25 2 L 13 46 C 53 26 88 39 84 68 C 81 94 44 104 17 98'),
        (2, 'M 87 3 Q 58 1 32 3 C 0 23 20 34 46 42 C 103 58 79 93 46 99 Q 26 103 12 94'),
        (2, 'M 26 5 L 14 48 Q 61 32 77 52 C 98 79 49 108 21 96 Q 9 91 13 81 M 28 0 Q 52 8 91 0'),
        (1, 'M 91 0 L 28 8 L 19 46 C 46 32 73 40 75 58 Q 80 83 30 100'),
    ],
    [ # 6: stem down the left, loop on right. Top can curl or be quite straight.
        (5, 'M 72 0 C 46 14 15 50 17 77 C 18 104 53 105 72 88 C 96 68 81 42 57 46 Q 33 48 17 74'),
        (4, 'M 65 0 Q 18 38 17 73 C 14 103 42 104 65 92 C 93 76 90 48 64 49 Q 40 48 22 74'),
        (3, 'M 78 5 Q 65 -7 49 6 C 27 24 10 65 21 86 C 34 110 69 103 81 81 C 95 55 67 38 44 53 Q 31 59 22 74'),
        (2, 'M 55 0 Q 20 39 20 75 C 20 105 57 107 76 88 C 93 68 79 49 62 51 Q 43 52 28 71'),
        (2, 'M 80 1 C 38 12 18 47 18 73 C 18 100 47 106 70 93 C 98 74 85 48 65 48 Q 49 48 31 63'),
        (1, 'M 59 0 Q 24 34 13 64 C -1 99 39 113 66 91 C 95 68 84 44 64 51 Q 52 55 44 65'),
    ],
    [ # 7: angled downstroke, optional bent/crossed top and midline crossbar
        (6, 'M 10 7 Q 40 -1 87 1 Q 61 46 34 100'),
        (3, 'M 11 4 Q 42 7 78 2 Q 86 0 81 12 Q 53 54 42 100'),
        (3, 'M 11 21 L 14 3 Q 49 -1 88 3 L 37 100'),
        (3, 'M 9 9 Q 40 -2 86 2 L 37 100 M 30 54 L 73 50'),
        (2, 'M 9 6 Q 40 4 69 0 Q 80 -1 77 14 L 49 100 M 40 54 Q 55 52 73 53'),
        (2, 'M 7 1 Q 43 13 81 3 Q 66 33 61 53 L 45 100'),
        (1, 'M 10 20 Q 3 -2 20 3 Q 58 13 82 2 L 38 99'),
    ],
    [ # 8: continuous figure eight, double bowls; vary crossing and lobe ratio
        (6, 'M 52 49 C 30 36 15 26 25 11 C 39 -10 81 0 81 16 C 82 31 54 46 36 61 C 13 81 13 98 40 100 C 71 105 94 79 76 62 Q 65 53 52 49'),
        (4, 'M 56 48 C 83 30 92 12 74 3 C 48 -9 17 7 21 24 C 24 39 62 51 75 66 C 96 90 63 107 35 97 C 6 87 24 62 56 48'),
        (3, 'M 58 44 C 39 37 20 24 29 9 C 43 -7 73 -2 81 11 C 93 30 62 43 43 52 C 15 65 5 90 25 98 C 54 111 90 88 84 68 Q 81 52 58 44'),
        (3, 'M 46 55 C 19 46 9 26 23 10 C 41 -9 79 -2 85 16 C 91 36 65 50 46 55 C 16 63 8 89 32 98 C 59 109 84 92 81 76 Q 80 61 46 55'),
        (2, 'M 53 47 C 21 52 13 31 23 14 C 38 -8 79 -4 85 16 C 90 35 67 48 53 47 C 29 48 10 73 20 90 C 35 111 77 99 83 78 C 88 60 73 44 53 47'),
        (1, 'M 65 4 C 42 -4 18 15 26 30 C 34 46 69 50 79 69 C 94 95 48 110 26 94 C 3 75 41 54 61 37 Q 86 13 73 4'),
    ],
    [ # 9: upper counter + descender. A loop can be narrow, angular, or open.
        (6, 'M 76 14 C 61 -8 30 -2 17 22 C 2 48 21 66 45 53 Q 66 42 78 13 Q 73 58 66 100'),
        (4, 'M 76 15 C 66 -6 32 -4 18 16 C -1 42 19 63 44 54 Q 68 47 79 19 L 60 100'),
        (3, 'M 78 8 C 57 -5 26 2 16 24 C 4 51 27 63 48 51 L 80 16 Q 67 70 72 100'),
        (2, 'M 71 3 Q 46 -5 26 11 Q 6 28 15 48 Q 37 65 62 39 L 80 7 Q 78 37 65 75 Q 58 95 46 100'),
        (2, 'M 75 18 C 59 -9 26 -1 16 22 C 5 45 22 57 46 50 Q 68 43 73 23 Q 77 59 81 100'),
        (2, 'M 76 16 Q 69 -7 43 3 Q 12 10 11 39 Q 12 59 36 54 Q 60 52 77 20 C 79 52 64 102 40 99 Q 27 99 21 91'),
        (2, 'M 61 2 Q 39 -2 22 21 Q 0 51 25 54 Q 50 58 78 13 L 66 100'),
        (1, 'M 74 3 L 48 0 Q 18 14 12 43 Q 13 59 39 50 L 78 11 L 68 100'),
    ],
]


# Extra allographs: e.g. a nib that makes a loop where a diagonal changes
# direction, or a cap made after a separate downstroke. These are pen paths,
# not coordinates from digit images.
PENS[0] += [
    (2, 'M 68 6 C 44 -13 11 29 11 67 C 9 99 32 108 56 94 C 81 80 95 39 84 16 Q 76 -1 59 2'),
    (1, 'M 58 3 C 28 -7 10 43 12 74 Q 15 105 46 98 C 80 90 91 46 83 16 Q 76 -1 56 8 L 48 19')]
PENS[2] += [
    (3, 'M 11 25 C 22 -2 47 -5 57 11 C 72 32 48 72 25 90 C 8 104 4 88 13 77 C 24 64 37 79 52 91 Q 72 106 100 81'),
    (2, 'M 12 18 C 30 -9 59 -2 69 23 Q 83 53 27 93 C 7 110 0 90 16 86 Q 29 80 46 85 Q 74 98 98 80'),
    (2, 'M 22 37 C 10 38 8 20 21 8 Q 48 -14 67 14 C 86 47 45 78 19 91 Q 47 103 96 91'),
    (2, 'M 14 12 Q 34 -7 51 5 C 85 33 45 80 21 99 C 47 85 68 84 96 87'),
    (1, 'M 5 12 C 36 -10 69 2 68 30 C 67 55 19 69 13 86 Q 5 100 18 99 Q 37 93 54 98 Q 76 108 99 95')]
PENS[3] += [
    (2, 'M 7 16 C 26 9 60 -9 80 5 Q 90 15 47 45 Q 30 56 57 51 C 99 41 92 76 66 92 Q 40 109 16 93'),
    (1, 'M 10 17 Q 36 -4 65 2 Q 86 8 70 25 L 42 42 C 73 31 85 46 80 67 Q 68 104 21 100'),
    (2, 'M 10 9 Q 45 -2 78 1 Q 78 20 43 46 C 68 39 91 52 85 69 Q 78 92 28 101')]
PENS[4] += [
    (2, 'M 29 0 Q 8 21 8 44 Q 9 57 28 54 L 84 39 M 70 4 Q 61 57 66 100'),
    (2, 'M 27 0 Q 16 32 13 48 Q 11 58 22 57 L 54 53 M 68 17 Q 62 57 60 100'),
    (2, 'M 29 0 Q 25 44 34 47 Q 43 50 74 15 M 79 0 Q 71 47 64 100')]
PENS[5] += [
    (2, 'M 37 7 Q 14 18 16 39 Q 65 29 82 54 Q 102 86 47 99 Q 23 103 11 91 M 28 4 Q 60 10 90 0'),
    (2, 'M 77 2 L 31 3 Q 10 28 16 35 Q 71 40 78 65 Q 86 92 42 98 Q 22 102 10 95'),
    (1, 'M 33 10 Q 10 24 18 49 Q 50 36 69 47 C 98 70 59 112 30 98 Q 14 88 17 78 M 21 1 Q 59 4 95 -5')]
PENS[6] += [
    (3, 'M 61 0 Q 17 56 22 85 Q 27 104 55 97 C 80 90 89 67 70 60 Q 47 54 25 77'),
    (2, 'M 69 0 Q 27 29 16 65 Q 5 98 38 99 C 68 102 88 79 79 58 Q 69 42 47 53 Q 28 62 17 81'),
    (1, 'M 67 0 C 52 12 23 43 21 74 Q 18 105 49 99 Q 73 92 78 73 Q 82 58 66 62 L 47 75')]
PENS[7] += [
    (2, 'M 8 16 Q 37 -7 78 4 Q 86 6 77 26 Q 57 63 55 100'),
    (1, 'M 8 13 Q 35 -8 83 5 Q 67 31 59 57 Q 39 114 29 98 Q 25 90 37 74')]
PENS[8] += [
    (2, 'M 47 40 C 24 31 17 18 32 6 C 58 -13 86 3 76 21 Q 67 33 47 40 C 27 51 8 76 18 91 C 35 115 81 95 83 72 Q 83 53 47 40'),
    (2, 'M 55 58 C 20 40 18 16 41 3 C 67 -9 92 5 80 26 Q 75 42 55 58 C 26 74 11 90 30 98 C 52 108 85 90 82 78 Q 82 69 55 58'),
    (2, 'M 59 6 C 39 -8 16 13 22 30 Q 26 46 65 64 C 99 84 59 108 31 98 C 3 87 36 59 57 41 Q 87 15 71 3'),
    (1, 'M 36 5 C 57 -7 89 1 83 20 Q 77 34 45 51 C 19 67 5 92 27 99 C 58 108 89 85 78 68 Q 67 53 46 45 Q 16 28 30 12')]
PENS[9] += [
    (2, 'M 69 12 C 52 -8 29 -2 19 17 Q 3 45 30 46 Q 56 46 72 15 Q 73 62 57 100'),
    (2, 'M 77 9 C 53 -8 25 5 15 32 Q 8 58 32 59 Q 56 59 76 23 L 73 100'),
    (1, 'M 69 7 C 37 -4 12 18 13 39 Q 18 61 44 49 Q 66 37 78 8 Q 67 59 77 100')]


def specify(text):
    """Sample each segment separately; a pen lift must leave no connecting line."""
    tokens = iter(text.split())
    ends, t = [], np.linspace(0, 1, 17)[:, None]
    def pt():
        return np.array((float(next(tokens)), float(next(tokens))))
    for command in tokens:
        if command == 'M':
            a = pt(); continue
        if command == 'L':
            b = pt(); curve = (1-t)*a+t*b
        elif command == 'Q':
            c,b = pt(),pt(); curve = (1-t)**2*a+2*t*(1-t)*c+t*t*b
        elif command == 'C':
            c,d,b = pt(),pt(),pt()
            curve = (1-t)**3*a+3*t*(1-t)**2*c+3*(1-t)*t*t*d+t**3*b
        else:
            raise ValueError(command)
        ends.append(np.stack((curve[:-1],curve[1:]),axis=1)); a=b
    return np.concatenate(ends).astype(float)/100


class Draftsman:
    """Fixed pen style variation; each rendering is centered by its own ink only."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.styles = [[(w,specify(s)) for w,s in variants] for variants in PENS]
        # A fine paper grid, with enough extra space for the initial placement.
        self.ss = 3
        self.page = 38*self.ss
        self.z = np.arange(self.page,dtype=float)
        self.dst = np.arange(28*self.ss,dtype=float)

    def image(self, digit):
        rng=self.rng
        def u(a,b): return rng.uniform(a,b)
        def bell(s): return np.clip(rng.normal(),-2.3,2.3)*s
        variants=self.styles[digit]
        r=rng.integers(sum(w for w,_ in variants))
        for w, seg in variants:
            if r<w: break
            r-=w
        xx,yy=seg[...,0]-.5, seg[...,1]-.5
        # Smooth pen-position displacements. In particular the oval/lobe sizes
        # change continuously; variation is not confined to rigid transforms.
        x = xx*np.exp(bell(.27))*(1+bell(.22)*yy*2)
        x += bell(.045)*np.cos(np.pi*2*yy)+bell(.042)*np.sin(np.pi*2*yy)
        x += bell(.035)*np.cos(np.pi*xx*2+u(0,6.28))*np.cos(np.pi*yy+u(0,6.28))
        y = yy + bell(.17)*xx+bell(.043)*np.cos(np.pi*2*yy)
        y += bell(.04)*np.cos(np.pi*2*xx+u(0,6.28))*(.7+.5*np.sin(np.pi*yy+u(0,6.28)))
        x += (bell(.29) - (.10 if digit!=7 else .035))*y
        y += bell(.055)*np.sin(np.pi*x*2+u(0,6.28))
        thick = u(1.45,4.6) if digit!=1 else u(1.65,5.0)
        # Scale down the whole stroke so its bounding dimension is about 20.
        box = u(19.5,21.0); scale=(box-thick)/max(np.ptp(x),np.ptp(y))
        x=((x-(x.min()+x.max())/2)*scale+19)*self.ss
        y=((y-(y.min()+y.max())/2)*scale+19)*self.ss
        # Rasterize by covering the center line with an overlapping round nib.
        dx,dy=x[:,1]-x[:,0],y[:,1]-y[:,0]
        n=np.maximum(1,np.ceil(np.hypot(dx,dy)/.72).astype(int))
        ids=np.repeat(np.arange(len(n)),n)
        t=(np.arange(len(ids))-np.repeat(np.cumsum(n)-n,n)+.5)/n[ids]
        cx=x[ids,0]+dx[ids]*t; cy=y[ids,0]+dy[ids]*t
        # Tilted nib pressure and a slowly varying stroke width.
        rad=thick*self.ss*.5
        radius=rad*(1+u(-.15,.15)*np.cos((cy-19*self.ss)*.10+u(0,6.28)) + u(-.10,.10)*dx[ids]/(1.e-4+np.hypot(dx[ids],dy[ids])))
        aspect=u(.85,1.15); bound=int(np.ceil(radius.max()*max(aspect,1/aspect)+1))
        j=np.arange(-bound,bound+1,dtype=int)
        ox = np.broadcast_to(j[None,:],(len(j),len(j))).ravel(); oy=ox.reshape(len(j),len(j)).T.ravel()
        # Occupancy (rather than adding ink repeatedly at self-intersections).
        ax=np.floor(cx).astype(int)[:,None]+ox; ay=np.floor(cy).astype(int)[:,None]+oy
        mask=((ax-cx[:,None])/aspect)**2+((ay-cy[:,None])*aspect)**2 <= radius[:,None]**2
        ink=np.zeros((self.page,self.page),np.float32); ink[ay[mask],ax[mask]]=1
        den=ink.sum()
        mx=np.dot(ink.sum(0),self.z)/den; my=np.dot(ink.sum(1),self.z)/den
        def lattice(m):
            return np.rint(self.dst - (len(self.dst)-1)/2 + m + u(-.55,.55)*self.ss).astype(int).clip(0,self.page-1)
        ink=ink[np.ix_(lattice(my),lattice(mx))]
        return ink.reshape(28,self.ss,28,self.ss).mean((1,3))


# First view of a shape: locally pooled signed directional edge energy. The
# second view gives the orientation of the stroke regardless of edge polarity.
# Both use the same eight Gaussian derivative axes; making all filters from
# formulas lets the input and a template be described in the same units.
def make_filters():
    u=np.arange(-2,3); y,x=np.meshgrid(u,u,indexing='ij')
    g=np.exp(-(x*x+y*y)/(2*1.04**2))
    bank=[]
    for k in range(8):
        theta=np.pi*k/4
        a=(np.cos(theta)*x+np.sin(theta)*y)*g
        a-=a.mean(); a/=np.sum(np.abs(a))/2
        bank.append(a)
    return np.asarray(bank,np.float32)


def pooling(a,k,stride,maximum=False):
    v=windows(a,(k,k),axis=(-2,-1))[...,::stride,::stride,:,:]
    return v.max(axis=(-2,-1)) if maximum else v.mean(axis=(-2,-1))


def describe(batch,filters):
    # No dimension indexed by example is ever reduced in the feature builder.
    # The small negative baseline of a descriptor counts matching empty space.
    a = np.pad(batch,((0,0),(2,2),(2,2)))
    v=windows(a,(5,5),axis=(1,2))
    v=np.ascontiguousarray(v.reshape(len(a),784,25))
    edges=(v@filters.reshape(8,25).T).transpose(0,2,1).reshape(-1,8,28,28)
    signed=pooling(np.maximum(edges,0),3,1,True)
    signed=pooling(signed,4,3)   # (8,8,8)
    # The third-magnitude-per-cell view broadens displacement tolerance. A
    # separate low-resolution ink channel keeps holes and thin strokes visible.
    unsigned=pooling(np.abs(edges[:,:4]),4,1,True)
    unsigned=pooling(unsigned,5,4)   # (4,6,6)
    coarse=pooling(batch[:,None],3,2) # (1,13,13)
    parts=[]
    for z in (signed,unsigned,coarse):
        z=z.reshape(len(z),-1)
        z=z-z.mean(1,keepdims=True)
        # Multiplying by the same normalization that the input will receive
        # makes a subsequent dense dot product a normalized cross-correlation.
        parts.append(z/(z.shape[1]*np.sqrt((z*z).mean(1,keepdims=True)+1.e-5)))
    return parts


class Network:
    def __init__(self):
        self.layers=[]; self.weights={}
    def layer(self,name,op,inputs,**settings):
        self.layers.append(dict(name=name,op=op,inputs=[inputs] if isinstance(inputs,str) else inputs,**settings));return name
    def affine(self,name,op,inputs,W,b=None,**settings):
        W=np.asarray(W,dtype=np.float32)
        self.weights[name+'.W']=W; self.weights[name+'.b']=np.zeros(len(W),np.float32) if b is None else np.asarray(b,np.float32)
        return self.layer(name,op,inputs,out=len(W),**settings)
    def pool(self,name,kind,src,k,st):
        return self.layer(name,kind+'pool',src,size=k,stride=st)
    def flatnorm(self,name,src,size):
        row=self.layer(name+'_row','reshape',src,shape=[1,1,size]);return self.layer(name,'norm',row)
    def conv(self,name,src,w,b=None,pad=0,stride=1):
        return self.affine(name,'conv',src,w,b,kernel=[w.shape[-1]]*2,stride=stride,padding=pad)


# A canonical frame per image, computed with fixed interpolation matrices and
# ReLU gates. No moments are shared between drawings. Correct slant first;
# scale horizontally second, leaving enough width cue to recognize a 1.
def warp_candidates():
    yy,xx=np.mgrid[:28,:28].astype(float)-13.5
    ey=yy.ravel();ex=xx.ravel()
    # Test inequalities on linear raw moments. For slant the tests correspond
    # to (sum xy)/(sum yy); for size to (sum xx)/(sum ink).
    slopes=np.array([-.90,-.55,-.28,0,.28,.55,.90]); sw=(slopes[:-1]+slopes[1:])*.5
    widths=np.array([2.6,3.35,4.3,5.4,6.7]); ww=((widths[:-1]+widths[1:])*.5)**2
    return [(slopes,np.array([ex*ey-t*ey**2 for t in sw]),np.array([ex.reshape(28,28)+ey.reshape(28,28)*t+13.5 for t in slopes])),
            (widths,np.array([ex*ex-t for t in ww]),np.array([xx*(t/4.1)**.8+13.5 for t in widths]))]


def resample_spec(xsrc):
    xlo=np.floor(xsrc).astype(int); frag=xsrc-xlo; xhi=xlo+1
    offs=np.arange(28)[:,None]*28
    return ((xlo.clip(0,27)+offs).ravel(),((1-frag)*((xlo>=0)&(xlo<28))).ravel().astype('f'),
            (xhi.clip(0,27)+offs).ravel(),(frag*((xhi>=0)&(xhi<28))).ravel().astype('f'))


def canonical_paper(batch,warps):
    b=batch.reshape(-1,784)
    for levels,tests,srcs in warps:
        idx=np.sum(b@tests.T>0,axis=1)
        nxt=np.zeros_like(b)
        for j in range(len(levels)):
            ii=np.where(idx==j)[0]
            lo,a,hi,c=resample_spec(srcs[j]); nxt[ii]=b[ii][:,lo]*a+b[ii][:,hi]*c
        b=nxt
    return b.reshape(-1,28,28)


def canonical_input(net,warps):
    z='image'
    for label,(levels,tests,srcs) in zip(('slant','width'),warps):
        K=len(levels)
        # Binary decisions approximated by a unit ramp of very small width in
        # raw moment units. Two candidates are smoothly gated at exact ties.
        zt=net.affine(label+'_moment_tests','dense',z,np.r_[tests,tests],np.r_[np.full(K-1,.5),np.full(K-1,-.5)])
        zt=net.layer(label+'_test_positive','relu',zt)
        # bias for candidate j = flag[j-1]-flag[j]-1, with flags for
        # the two outer boundaries fixed at 1 and 0 respectively.
        d=np.zeros((K,K-1));
        for j in range(K-1): d[j,j]=-1; d[j+1,j]=1
        gb=np.full(K,-1.); gb[0]=0
        gating=net.affine(label+'_gate','dense',zt,np.c_[d,-d],gb)
        biases=net.affine(label+'_gate_grid','dense',gating,np.repeat(np.eye(K),784,axis=0))
        mats=np.zeros((K,784,784),np.float32);ii=np.arange(784)
        for j,src in enumerate(srcs):
            lo,a,hi,b=resample_spec(src);mats[j,ii,lo]+=a;mats[j,ii,hi]+=b
        warped=net.affine(label+'_resample','dense',z,mats.reshape(-1,784));g=net.layer(label+'_masked','add',[warped,biases]);g=net.layer(label+'_accepted','relu',g)
        g=net.layer(label+'_candidate_grid','reshape',g,shape=[1,K,784]);g=net.pool(label+'_select','max',g,[K,1],[K,1]);z=net.layer(label+'_corrected','reshape',g,shape=[1,28,28])
    return z


def canonical_pattern(net,src,f):
    e=net.conv('canon_edge',src,f[:,None],pad=2);e=net.layer('canon_positive','relu',e);e=net.pool('canon_edge_radius','max',e,[3,3],[1,1]);e=net.pool('canon_edge_cells','avg',e,[4,4],[3,3]);e=net.flatnorm('canon_edge_pattern',e,512)
    c=net.pool('canon_ink_cells','avg',src,[3,3],[2,2]);return e,net.flatnorm('canon_ink_pattern',c,169)


def enclosure_masses(ink):
    """Four enclosure strengths at every cell: the least peak ink one has
    to cross on any straight escape within a quarter-plane. Work done only
    for an individual drawing; leading dimensions are independent examples.
    """
    h,w=ink.shape[-2:];rows,cols=np.mgrid[:h,:w]
    corners=[]
    for v,t in [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]:
        ray=ink.copy()
        for l in (1,2,4,8):
            sy=(rows+v*l);sx=(cols+t*l);inside=(sy>=0)&(sy<h)&(sx>=0)&(sx<w)
            other=ray[...,sy.clip(0,h-1),sx.clip(0,w-1)]*inside
            ray=np.maximum(ray,other)
        corners.append(ray)
    quadrants=[np.maximum(0,np.minimum(np.minimum(corners[i],corners[(i+1)%8]),corners[(i+2)%8])-ink) for i in (0,2,4,6)]
    value=np.stack(quadrants,axis=1); value=pooling(value,4,2);value=value.reshape(len(ink),-1)
    value-=value.mean(1,keepdims=True)
    return value/(value.shape[1]*np.sqrt((value*value).mean(1,keepdims=True)+1e-5))


def enclosure_input(n,ink):
    source=n.pool('canon_small','avg',ink,[2,2],[2,2]);sg=n.affine('white_cell','dense',source,-np.eye(196));index=np.arange(196);yy,xx=np.mgrid[:14,:14];rr=[]
    for i,(dy,dx) in enumerate([(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]):
        ray=n.layer('ray_start%d'%i,'reshape',source,shape=[196])
        for l in (1,2,4,8):
            sy=yy+dy*l;sx=xx+dx*l;inside=((sy>=0)&(sy<14)&(sx>=0)&(sx<14)).ravel()
            m=-np.eye(196);m[index[inside],(14*sy+sx).ravel()[inside]]+=1
            d=n.affine('ray_difference_%d_%d'%(i,l),'dense',ray,m);d=n.layer('ray_positive_%d_%d'%(i,l),'relu',d)
            ray=n.layer('ray_max_%d_%d'%(i,l),'add',[ray,d])
        rr.append(ray)
    negids=-np.eye(196); sheets=[]
    for i in (0,2,4,6):
        a=rr[i]
        for j in (1,2):
            b=rr[(i+j)%8];nb=n.affine('ray_neg_%d_%d'%(i,j),'dense',b,negids)
            d=n.layer('excess_%d_%d'%(i,j),'add',[a,nb]);d=n.layer('excess_positive_%d_%d'%(i,j),'relu',d)
            nd=n.affine('remove_excess_%d_%d'%(i,j),'dense',d,negids);a=n.layer('quadrant_bound_%d_%d'%(i,j),'add',[a,nd])
        q=n.layer('background_enclosure_%d'%i,'add',[a,sg]);q=n.layer('enclosed_%d'%i,'relu',q);q=n.layer('quarter_sheet_%d'%i,'reshape',q,shape=[1,14,14]);q=n.pool('quarter_cells_%d'%i,'avg',q,[4,4],[2,2]);
        mat=np.zeros((144,36)); mat[(i//2)*36:(i//2+1)*36]=np.eye(36);sheets.append(n.affine('quarter_place_%d'%i,'dense',q,mat))
    join=n.layer('enclosure_directions','add',sheets);return n.flatnorm('enclosure_pattern',join,144)


def topology(net):
    """Reliably enclosed background, ignoring white background connected to the
    frame. Propagate exterior access at two ink levels on a reduced lattice.
    This branch only uses a hand-specified anatomical rule table, no templates.
    """
    n=net
    grid=n.pool('small_ink','avg','image',[2,2],[2,2]) # 14x14
    # A soft binary indicator: 0 if no ink, 1 if definitely ink.
    # continuous clamp(x,0,1) = relu(x) - relu(x-1)
    cuts=np.array([.34,.53]); gain=9
    z=n.conv('ink_levels',grid,np.full((4,1,1,1),gain),-np.r_[cuts,cuts]*gain-np.array([0,0,1,1]))
    z=n.layer('ink_ramp','relu',z)
    wall=n.conv('barriers',z,np.array([[1,0,-1,0],[0,1,0,-1]])[:,:,None,None]);
    # Start the flood with a fixed one-cell-wide exterior border. Dense is
    # unnecessary: constant padding can be formed with a stencil on ones.
    one=n.conv('paper',grid,np.zeros((1,1,1,1)),[1])
    seed=np.zeros((2,1,3,3));seed[:,0,1,0]=seed[:,0,1,2]=seed[:,0,0,1]=seed[:,0,2,1]=-1
    interior=n.conv('interior_test',one,-seed,[-3,-3],pad=1)
    interior=n.layer('interior_only','relu',interior)
    e=n.conv('outer_edge',interior,-np.eye(2)[:,:,None,None],[1,1])
    # Propagate to neighboring cells using max pooling on a zero-padded
    # array. A 3x3 square (8-connectivity) prevents diagonal near-misses from
    # creating spurious holes. The wall is subtracted at each new location.
    neg=n.conv('stop_at_ink',wall,-np.eye(2)[:,:,None,None]); ident=np.eye(2)[:,:,None,None]
    for i in range(17):
        pad=n.conv('extend_%d'%i,e,ident,pad=1)
        near=n.pool('reach_%d'%i,'max',pad,[3,3],[1,1])
        out=n.layer('subtract_wall_%d'%i,'add',[near,neg]);e=n.layer('outside_%d'%i,'relu',out)
    return e,wall


def write_network(folder):
    folder.mkdir(parents=True,exist_ok=True)
    net=Network();f=make_filters()
    edges=net.conv('edge_derivatives','image',f[:,None],pad=2)
    r=net.layer('edge_strength','relu',edges)
    p=net.pool('edge_tolerance','max',r,[3,3],[1,1]);s=net.pool('edge_cells','avg',p,[4,4],[3,3]);s=net.flatnorm('edge_pattern',s,512)
    # abs(d) = relu(d) + relu(-d)
    unsigned=net.conv('unsigned_strength',r,(np.c_[np.eye(4),np.eye(4)])[:,:,None,None]);p=net.pool('stroke_tolerance','max',unsigned,[4,4],[1,1]);u=net.pool('stroke_cells','avg',p,[5,5],[4,4]);u=net.flatnorm('stroke_pattern',u,144)
    p=net.pool('ink_cells','avg','image',[3,3],[2,2]);c=net.flatnorm('ink_pattern',p,169)

    # Variants per label; pool within groups and then across groups. This
    # allows inexact variants without penalizing a rare style for being rare.
    groups=8; per_group=288; count=groups*per_group
    maker=Draftsman(20261109)
    warps=warp_candidates(); canonical=canonical_input(net,warps);ce,ci=canonical_pattern(net,canonical,f)
    enc=enclosure_input(net,canonical)
    rows=[np.empty((10*count,d),np.float32) for d in (512,144,169,512,169,144)]
    for digit in range(10):
        for j in range(0,count,24):
            imgs=np.stack([maker.image(digit) for _ in range(24)])
            sl=slice(digit*count+j,digit*count+j+24)
            a,b,c0=describe(imgs,f)
            aligned=canonical_paper(imgs,warps);a1,_,c1=describe(aligned,f)
            gaps=enclosure_masses(pooling(aligned,2,2))
            for W,part in zip(rows,(a,b,c0,a1,c1,gaps)): W[sl]=part
    # Fuse feature types per drawing, not per class: the loops and pen strokes
    # of the same interpretation should agree spatially.
    similarity=[]
    for name,z,W,gain in zip(('signed','axial','silhouette','canonical_edge','canonical_ink','enclosure'),(s,u,c,ce,ci,enc),rows,(.21,.10,.09,.32,.12,.16)):
        W*=gain
        similarity.append(net.affine('compare_'+name,'dense',z,W))
    sims=net.layer('drawing_match','add',similarity)
    layout=net.layer('by_digit_style','reshape',sims,shape=[10,groups,per_group]);a=net.pool('best_in_group','max',layout,[1,per_group],[1,per_group]);b=net.pool('best_overall','max',a,[groups,1],[groups,1]);b=net.layer('best_scores','reshape',b,shape=[10]);a=net.pool('near_matches','avg',a,[groups,1],[groups,1]);a=net.layer('group_scores','reshape',a,shape=[10]);a=net.affine('group_vote','dense',a,.30*np.eye(10));b=net.affine('best_vote','dense',b,.70*np.eye(10));base=net.layer('form_scores','add',[a,b]);

    e,wall=topology(net)
    inv=net.conv('unreached',e,-np.eye(2)[:,:,None,None],[1,1]);holes=net.layer('not_ink','add',[inv,'stop_at_ink']);holes=net.layer('counter_interiors','relu',holes)
    # Individual bounded zones distinguish a low loop (6), a high loop (9),
    # and hollow regions both above and below a crossing (8). Broad spatial
    # weights avoid any requirement for an exact waist location.
    y=np.arange(14)[:,None];x=np.arange(14)[None,:]
    bands=[np.maximum(0,1-np.abs(y-h)/r)*(np.abs(x-6.5)<6.5) for h,r in [(4,3.3),(9,3.3),(6.5,5.5)]]
    M=np.zeros((6,2,14,14));
    for j,band in enumerate(bands): M[j,:]=band*.50
    # Soft presence saturates, so a single large hole cannot give unlimited
    # evidence. Last three rows are the same hinges offset by the cap width.
    M[3:]=M[:3];h=net.affine('counter_mass','dense',holes,M.reshape(6,-1)/2.6,[-.32,-.32,-.35,-1.32,-1.32,-1.35]);h=net.layer('counter_mass_ramp','relu',h);h=net.affine('counter_presence','dense',h,np.c_[np.eye(3),-np.eye(3)])
    # No adjustment if no hole is detected. Absence of a hole need not rule
    # out a digit: an author may leave an almost-closed pen path.
    rules=np.array([[0,0,.034],[0,0,-.035],[0,.004,-.015],[0,0,-.030],[.004,0,-.008],[0,.010,-.015],[-.016,.055,.004],[0,0,-.030],[.022,.022,.001],[.055,-.015,.002]])
    structural=net.affine('loop_evidence','dense',h,rules)
    net.layer('digit_scores','add',[base,structural]);
    (folder/'network.json').write_text(json.dumps({'layers':net.layers},indent=1)+'\n');np.savez(folder/'weights.npz',**net.weights)

if __name__=='__main__': write_network(Path(sys.argv[1]))
