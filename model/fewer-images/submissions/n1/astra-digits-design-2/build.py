"""Compile hand-drawn digit grammars into an analytic template-matching network.

Usage: python3 build.py OUTPUT_DIRECTORY. No input files, fonts, models, or data.
The generator works on one hypothetical pen drawing at a time. No scoring or
statistics over generated drawings are involved in constructing the weights.
"""
from pathlib import Path
import json
import sys
import numpy as np


# A tiny pen language: M (start), L (straight), Q (quadratic), C (cubic).
# Coordinates are in an abstract writing square, from the cap line (0) to
# the baseline (1), not in image pixels. Slant, proportions and stroke weight
# are supplied independently of these instructions.

def pen(spec):
    tok = iter(spec.split())
    strokes, parts = [], []
    def xy():
        return np.array([float(next(tok)), float(next(tok))])
    for cmd in tok:
        if cmd == 'M':
            if parts:
                strokes.append(np.vstack(parts))
            a = xy()
            parts = [a[None, :]]
            continue
        if cmd == 'L':
            b = xy()
            points = [a, b]
        elif cmd == 'Q':
            b, c = xy(), xy()
            points = [a, b, c]
        elif cmd == 'C':
            b, c, d = xy(), xy(), xy()
            points = [a, b, c, d]
        else:
            raise ValueError(cmd)
        n = max(3, int(np.ceil(60 * sum(np.linalg.norm(j-i) for i,j in zip(points,points[1:])))))
        t = (np.arange(1,n+1) / n)[:,None]
        if cmd == 'L':
            p = (1-t)*a + t*b
        elif cmd == 'Q':
            p = (1-t)**2*a + 2*t*(1-t)*b + t**2*c
        else:
            p = (1-t)**3*a + 3*t*(1-t)**2*b + 3*t**2*(1-t)*c + t**3*d
        parts.append(p)
        a = points[-1]
    strokes.append(np.vstack(parts))
    return np.vstack(strokes)


# Each entry describes an allograph, its relative frequency in this drawing
# grammar and the width of its writing square relative to its height.
GLYPHS = [
    [ # 0: a round loop, a more vertical oval, or a loop with an overrun.
      (5,.72,'M .54 .015 C .21 -.03 0 .21 .03 .54 C .03 .83 .17 1 .48 1 C .80 1.02 .99 .81 .98 .47 C .98 .20 .87 0 .54 .015'),
      (3,.68,'M .68 .025 C .33 -.07 .05 .18 .04 .53 C -.02 .86 .19 1.03 .44 .99 C .76 .98 .96 .79 .98 .49 C 1 .25 .94 .04 .70 .01'),
      (2,.75,'M .65 .14 C .55 -.02 .33 -.03 .19 .11 C -.02 .32 -.02 .64 .12 .86 C .31 1.11 .66 1.02 .83 .85 C 1.12 .53 .98 .13 .66 .045 C .51 0 .37 .07 .30 .19'),
      (1,.68,'M .51 .05 Q .17 -.09 .06 .24 Q -.03 .46 .05 .74 Q .12 1.04 .42 1 Q .78 1.02 .92 .76 Q 1.05 .45 .91 .16 Q .76 -.08 .51 .05'),
    ],
    [ # 1: a bare downstroke is common; flags and feet are optional.
      (6,.63,'M .52 0 Q .47 .47 .49 1'),
      (3,.61,'M .57 0 C .57 .29 .40 .69 .44 .91 Q .45 .98 .49 1'),
      (2,.60,'M .37 .12 L .56 0 Q .53 .42 .46 1'),
      (2,.65,'M .12 .27 Q .30 .20 .52 0 Q .54 .47 .48 1'),
      (1,.62,'M .09 .31 L .59 0 Q .52 .55 .51 .99 M .15 1 Q .55 .96 .91 1'),
      (1,.64,'M .49 0 L .49 .99 M .23 .99 L .76 .99'),
    ],
    [ # 2: arch-diagonal-baseline, with angular, upturned or looped feet.
      (4,.82,'M .05 .24 C .14 .03 .46 -.06 .71 .02 C 1 .11 .89 .36 .72 .52 Q .42 .76 .02 .99 Q .48 .92 1 .94'),
      (3,.81,'M .04 .25 C .21 -.02 .60 -.08 .75 .09 C .95 .34 .54 .59 .22 .84 Q -.04 1.02 .14 1 C .37 .96 .66 1.02 1 .88'),
      (3,.77,'M .06 .22 Q .05 .06 .32 .01 C .60 -.07 .87 .05 .83 .28 C .79 .51 .34 .76 .07 .99 Q .47 .95 1 .99'),
      (3,.80,'M .04 .22 C .19 -.05 .57 -.06 .70 .08 C .93 .33 .59 .64 .27 .88 C .13 1.02 -.04 1.03 .03 .89 C .16 .69 .46 .82 .67 .91 Q .90 1.01 1 .84'),
      (2,.85,'M .01 .18 C .17 .01 .48 -.06 .69 .05 C .89 .19 .78 .40 .52 .60 L .03 .99 L .99 .96'),
      (1,.83,'M .10 .27 C -.03 .14 .13 -.04 .39 .01 C .73 .03 .80 .26 .62 .51 Q .36 .77 .07 .97 Q .41 .91 .64 .97 Q .88 1.03 1 .85'),
    ],
    [ # 3: two bowls, or an angular upper stroke above a rounded lower bowl.
      (4,.71,'M .06 .12 C .28 -.02 .64 -.06 .83 .07 C 1.03 .26 .70 .43 .33 .48 C .68 .41 .99 .53 .97 .72 C .96 .99 .46 1.11 .08 .87'),
      (3,.75,'M .03 .11 Q .41 -.01 .90 .03 Q .84 .21 .37 .46 C .70 .37 1.01 .50 .96 .72 C .89 1.04 .40 1.09 .07 .88'),
      (3,.72,'M .08 .08 C .34 -.03 .75 -.04 .82 .12 C .90 .27 .64 .43 .38 .49 C .78 .40 .99 .62 .90 .81 C .77 1.03 .32 1.04 .06 .93'),
      (2,.72,'M .04 .12 Q .52 -.11 .83 .07 Q 1.05 .24 .51 .46 Q .24 .57 .30 .46 Q .41 .36 .80 .49 C 1.15 .64 .91 .94 .54 .99 Q .25 1.06 .02 .91'),
      (2,.74,'M .04 .15 L .90 .02 L .38 .46 Q .96 .34 .97 .64 Q 1.02 .92 .61 .99 Q .27 1.05 .04 .90'),
      (1,.68,'M .03 .19 C .28 -.06 .75 -.04 .82 .13 Q .91 .33 .36 .49 Q .74 .38 .94 .59 C 1.06 .85 .73 1.03 .38 1 Q .08 .98 .03 .83'),
    ],
    [ # 4: open-topped chair, triangular 4, and cursively rounded chair.
      (4,.75,'M .17 .025 Q .11 .32 .035 .54 Q .02 .60 .19 .59 L .99 .54 M .77 .01 Q .74 .48 .74 1'),
      (4,.77,'M .37 0 Q .21 .28 .025 .55 Q .41 .57 .94 .49 M .79 .04 L .70 1'),
      (3,.73,'M .12 .065 C .13 .21 -.01 .50 .08 .56 C .23 .65 .57 .54 .77 .45 M .79 .015 Q .70 .49 .71 1'),
      (3,.76,'M .74 .015 L .015 .61 L 1 .55 M .75 .015 L .70 1'),
      (2,.76,'M .66 .02 Q .36 .31 .025 .59 L .91 .54 M .78 .14 Q .71 .52 .68 1'),
      (1,.71,'M .14 0 Q .08 .29 .035 .46 Q .03 .65 .35 .56 Q .64 .50 .78 .23 L .81 .09 Q .70 .60 .69 1'),
    ],
    [ # 5: cap and spine, an outward middle bow, then a left-going foot.
      (4,.76,'M .95 .025 Q .64 .05 .19 .04 L .10 .48 C .47 .29 .88 .40 .94 .62 C 1.07 .96 .50 1.11 .08 .91'),
      (4,.76,'M .99 .01 Q .55 .08 .24 .04 Q .21 .27 .12 .44 C .49 .34 .83 .46 .89 .66 C .99 .92 .49 1.10 .07 .91'),
      (3,.73,'M .96 .04 L .23 .015 Q .15 .17 .10 .39 Q .41 .38 .64 .49 C 1.06 .68 .84 .96 .50 1 Q .20 1.05 .04 .88'),
      (2,.77,'M .99 .01 Q .62 .12 .25 .09 M .23 .05 Q .17 .27 .12 .44 C .53 .35 .99 .53 .87 .77 C .75 1.05 .28 1.04 .02 .91'),
      (2,.76,'M 1 .03 L .15 .055 Q -.01 .30 .19 .40 C .43 .50 .95 .48 .86 .75 C .79 1.02 .28 1.05 .04 .86'),
      (1,.72,'M .96 .035 Q .64 .085 .33 .025 L .10 .46 Q .46 .32 .73 .45 C 1.10 .65 .92 .95 .56 .99 Q .15 1.04 .045 .82'),
    ],
    [ # 6: declining entry to a bottom loop (can lead in from far right),
      # or a broader outside hook with a compact counter.
      (4,.71,'M .81 .01 C .56 .11 .17 .38 .075 .65 C -.03 .91 .27 1.06 .54 .98 C .87 .94 1.03 .70 .84 .58 C .67 .45 .31 .52 .09 .72'),
      (4,.69,'M .72 0 Q .24 .31 .09 .60 C -.01 .82 .12 1.04 .44 1 C .78 .99 .99 .85 .93 .65 C .87 .45 .51 .45 .28 .58 Q .15 .66 .10 .78'),
      (3,.70,'M .88 .055 Q .62 -.09 .39 .16 C .14 .37 -.02 .69 .12 .88 C .33 1.11 .73 1.01 .88 .85 C 1.05 .64 .83 .48 .61 .51 Q .32 .50 .10 .73'),
      (2,.68,'M .68 0 C .46 .21 .09 .50 .10 .77 C .11 .99 .52 1.07 .77 .94 C 1.05 .78 .90 .58 .69 .56 Q .39 .49 .14 .71'),
      (2,.74,'M .83 .015 Q .34 .12 .12 .53 C -.02 .84 .20 1.05 .57 .98 Q .99 .91 .94 .70 Q .92 .52 .64 .54 Q .33 .53 .16 .76'),
      (1,.72,'M .82 .02 C .40 -.02 .035 .42 .075 .74 C .10 1.02 .51 1.08 .78 .93 C 1.03 .80 .95 .58 .76 .53 Q .53 .43 .26 .58'),
    ],
    [ # 7: angular or rounded cap, with an optional crossed stem.
      (4,.80,'M .025 .07 Q .42 .00 .94 .04 Q .68 .41 .36 1'),
      (3,.77,'M .03 .04 Q .44 .015 .92 .025 L .33 1'),
      (3,.82,'M .03 .14 Q .38 -.025 .75 .045 Q .86 .065 .95 .015 Q .66 .44 .37 1'),
      (2,.78,'M .02 .03 Q .51 .015 .94 .06 L .36 1 M .29 .52 L .76 .49'),
      (2,.79,'M .025 .15 L .04 .02 L .94 .045 Q .72 .34 .35 1'),
      (1,.77,'M .025 .06 Q .26 -.02 .53 .05 Q .74 .09 .95 .025 L .38 1 M .37 .51 L .76 .48'),
    ],
    [ # 8: continuous crossing paths and two bowls touching at the waist.
      (4,.73,'M .53 .025 C .25 -.045 -.01 .13 .14 .32 C .27 .48 .80 .56 .90 .73 C 1.07 .99 .54 1.09 .24 .94 C -.06 .76 .24 .55 .51 .44 C .80 .31 1.02 .11 .72 .03 Q .60 -.005 .48 .025'),
      (4,.70,'M .62 .025 C .31 -.07 .02 .14 .16 .31 C .36 .51 .79 .56 .91 .73 C 1.02 .92 .70 1.06 .38 .99 C .05 .94 .005 .76 .22 .58 C .48 .37 .91 .27 .85 .12 Q .82 0 .62 .025'),
      (2,.71,'M .48 .48 C .16 .38 .02 .21 .21 .08 C .40 -.06 .80 -.01 .87 .16 C .95 .30 .66 .46 .48 .48 C .15 .57 -.04 .77 .14 .91 C .36 1.09 .82 1.02 .92 .82 C 1.02 .65 .76 .53 .48 .48'),
      (2,.70,'M .60 .06 C .38 -.07 .095 .075 .12 .25 C .14 .42 .49 .47 .69 .59 C .98 .74 .98 .94 .64 1 C .32 1.06 .05 .91 .10 .74 C .16 .51 .64 .38 .80 .23 Q 1.01 -.005 .62 .03'),
      (1,.74,'M .52 .50 C .21 .47 .04 .28 .15 .13 C .30 -.06 .74 -.015 .84 .13 C .99 .32 .73 .47 .52 .50 M .52 .50 C .24 .50 .025 .67 .09 .84 C .17 1.07 .67 1.07 .88 .89 C 1.08 .69 .76 .51 .52 .50'),
      (1,.67,'M .52 .04 C .27 -.07 .08 .08 .15 .26 C .19 .42 .79 .61 .90 .81 Q .96 1.03 .50 .99 Q .07 1.02 .09 .78 C .13 .56 .59 .35 .76 .21 Q .97 -.04 .58 .015'),
    ],
    [ # 9: an upper counter followed by a downward stroke. Distinguish
      # the print-style outer arc, straight downstems, and left-going tails.
      (4,.69,'M .79 .12 C .64 -.045 .36 -.03 .17 .12 C -.10 .33 .07 .57 .34 .54 Q .62 .53 .84 .19 M .84 .08 Q .83 .54 .70 1'),
      (4,.70,'M .80 .13 C .64 -.055 .31 -.025 .12 .18 C -.06 .37 .09 .54 .33 .52 Q .61 .50 .81 .16 Q .86 .10 .84 .045 Q .80 .52 .61 1'),
      (3,.68,'M .82 .17 C .70 -.04 .36 -.07 .14 .15 C -.08 .37 .14 .53 .36 .50 Q .69 .49 .82 .23 C .90 .11 .79 .035 .74 .055 M .81 .14 C .91 .45 .69 .81 .48 1'),
      (2,.70,'M .79 .15 C .67 -.03 .37 -.03 .15 .12 Q -.09 .34 .09 .46 C .28 .63 .68 .42 .82 .17 Q .85 .56 .65 .90 Q .50 1.07 .34 .94'),
      (2,.70,'M .80 .10 C .60 -.08 .22 .02 .10 .21 C -.02 .43 .16 .61 .40 .54 Q .71 .47 .79 .18 L .73 .99'),
      (1,.71,'M .84 .17 Q .69 -.08 .39 .035 Q .09 .13 .06 .35 Q .035 .57 .29 .56 Q .57 .59 .84 .22 Q .84 .49 .61 .75 L .43 1'),
    ],
]

# A fixed, low-discrepancy sequence distributes the deformation parameters.
# It is a sequence of numbers, not a fit to an image distribution.
PRIMES = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103]
def parameters(count, seed):
    rr = np.random.default_rng(seed)
    params = np.empty((count,len(PRIMES)))
    for j,p in enumerate(PRIMES):
        k = np.arange(1,count+1,dtype=np.int64)
        u = np.zeros(count)
        den = 1.
        while np.any(k):
            den /= p
            u += (k % p)*den
            k //= p
        params[:,j] = (u+rr.random())%1.
    return params


def peaked(u):
    # More coverage near the middle than the extremes of an interval.
    a = 2*u - 1
    return a*(.57+.43*abs(a))

FOOT = np.arange(-3,4)
DY,DX = np.meshgrid(FOOT,FOOT,indexing='ij')
DX,DY = DX.ravel(),DY.ravel()
Y,X = np.mgrid[:28,:28]

def render(stroke, width, u, digit):
    # Smooth changes of pen trajectory. Nearby points (including joins) move
    # together; the lower half need not be a copy of the upper half.
    x,y = (stroke-np.array([.5,.5])).T
    xx = x*(1 + .48*peaked(u[4])*y*2) + .15*peaked(u[5])*(1-4*y*y)
    yy = y + .10*peaked(u[6])*(1-4*y*y) + .16*peaked(u[7])*x + .20*peaked(u[8])*x*y
    xx += .075*peaked(u[9])*np.sin(2*np.pi*y) + .055*peaked(u[10])*np.cos(np.pi*x)*np.sin(np.pi*(y+.5))
    yy += .07*peaked(u[11])*np.cos(np.pi*y)*np.sin(np.pi*x) + .045*peaked(u[12])*np.cos(2*np.pi*x)*np.sin(np.pi*(y+.5))
    aspect = width*np.exp(.56*peaked(u[1]))
    shear = .12 + .70*peaked(u[2])
    if digit==1:
        shear = .13 + .76*peaked(u[2])
    x = aspect*xx - shear*yy
    y = yy + .105*peaked(u[13])*xx
    # The nominal line width is in final-image pixels. Pressure varies
    # continuously with position, and the simulated nib may have some tilt.
    r = .74 + 1.65*u[3]
    size = 19.1 + 1.3*u[14]
    scale = (size-2*r)/max(np.ptp(x),np.ptp(y))
    x = (x-.5*(x.min()+x.max()))*scale + 13.5
    y = (y-.5*(y.min()+y.max()))*scale + 13.5
    radius = r*(1 + .18*peaked(u[15])*np.sin(.22*y+.16*x+6*u[16]) + .09*peaked(u[17])*np.cos(.31*y-.34*x))
    # Maximum of anti-aliased dabs along the pen path (does not accumulate
    # additional ink at slow-moving portions of a path or at intersections).
    ix,iy = np.rint(x).astype(int),np.rint(y).astype(int)
    dx = (ix-x)[:,None] + DX
    dy = (iy-y)[:,None] + DY
    dist = np.sqrt((dx*(1+.10*peaked(u[18]))+.16*peaked(u[19])*dy)**2 + dy*dy)
    vals = np.clip(radius[:,None]+.5-dist,0,1)
    # Use symmetric margins of 4 at raster time for unusually heavy nibs.
    a = np.zeros(36*36)
    indices = ((iy[:,None]+4+DY)*36 + ix[:,None]+4+DX).ravel()
    np.maximum.at(a,indices,vals.ravel())
    a = a.reshape(36,36)
    rows,cols = a.sum(axis=1),a.sum(axis=0)
    cy = np.dot(np.arange(36),rows)/rows.sum()
    cx = np.dot(np.arange(36),cols)/cols.sum()
    # Match the centering convention in the task, with subpixel placement
    # uncertainty. This uses only the mass of this one pen drawing.
    sy = cy - (13.5 + .85*peaked(u[20]))
    sx = cx - (13.5 + .85*peaked(u[21]))
    yi,xi = int(np.floor(sy)),int(np.floor(sx))
    fy,fx = sy-yi,sx-xi
    y0 = np.clip(Y+yi,0,35); y1 = np.clip(Y+yi+1,0,35)
    x0 = np.clip(X+xi,0,35); x1 = np.clip(X+xi+1,0,35)
    return (1-fy)*((1-fx)*a[y0,x0]+fx*a[y0,x1]) + fy*((1-fx)*a[y1,x0]+fx*a[y1,x1])


# Fixed feature maps: half-wave Gaussian derivatives (gradient directions),
# positive even-symmetric line ridges, plus low-pass ink occupancy.
# Generate the identical receptive fields for both sides of a match.
ORI=6
K=7
gy,gx = np.mgrid[-3:4,-3:4]
kernels=[]
for i in range(ORI):
    ang=np.pi*i/ORI
    n=gx*np.cos(ang)+gy*np.sin(ang)
    t=gy*np.cos(ang)-gx*np.sin(ang)
    # Real-valued derivative of an anisotropic Gaussian. The greater
    # tangential extent makes it selective for stroke boundaries.
    g=n*np.exp(-.5*((n/1.05)**2+(t/1.65)**2))
    kernels.append(g/g[g>0].sum())
for i in range(ORI):
    ang=np.pi*i/ORI
    n=gx*np.cos(ang)+gy*np.sin(ang)
    t=gy*np.cos(ang)-gx*np.sin(ang)
    g=(1-(n/1.18)**2)*np.exp(-.5*((n/1.18)**2+(t/1.65)**2))
    g-=g.mean()
    kernels.append(g/g[g>0].sum())
kernels=np.array(kernels)
# Project each 4x4 window of derivative samples to a smooth regional bin.
# Sampling spacing is two pixels; the 8x8 support lets contours move a bit.
F=7
# Pair the two gradient polarities for all six orientations; the bright
# ridge filters are even-symmetric, so only one polarity is needed for them.
signed_filters=np.concatenate([kernels[:ORI],-kernels[:ORI],kernels[ORI:]],axis=0)
NCH=len(signed_filters)
# A 3x3 local integration on the 14x14 grid followed by 2x2 averaging is a
# triangular-ish low-pass with weights [1,2,2,1] in each dimension. Write
# this directly as a regional summation matrix so template features require
# exactly the same sequence of operations as the network.
sm=np.zeros((F,14))
for i in range(F):
    for j,coeff in zip(range(2*i-1,2*i+3),[1,2,2,1]):
        if 0<=j<14: sm[i,j]=coeff/6


def local_features(image):
    # 7x7 convolutions at stride 2 then ReLU, fixed spatial integration.
    a=np.pad(image,3)
    patches=np.lib.stride_tricks.sliding_window_view(a,(7,7))[::2,::2].reshape(196,49)
    gradients=(kernels.reshape(12,49)@patches.T).reshape(12,14,14)
    responses=np.maximum(np.concatenate([gradients[:ORI],-gradients[:ORI],gradients[ORI:]]),0)
    return sm@responses@sm.T


# Graph assembly for the inference version of the filters and the hand rules.
class Graph:
    def __init__(self):
        self.layers=[]; self.w={}
    def layer(self,name,op,x,**kw):
        self.layers.append(dict(name=name,op=op,inputs=x if isinstance(x,list) else [x],**kw))
        return name
    def conv(self,name,x,w,b=0,stride=1,padding=0):
        w=np.asarray(w,dtype=np.float32); o,c,k,k2=w.shape
        self.w[name+'.W']=w
        self.w[name+'.b']=np.broadcast_to(np.float32(b),(o,)).copy()
        return self.layer(name,'conv',x,out=o,kernel=[k,k2],stride=stride,padding=padding)
    def dense(self,name,x,w,b=0):
        w=np.asarray(w,dtype=np.float32); o=w.shape[0]
        self.w[name+'.W']=w
        self.w[name+'.b']=np.broadcast_to(np.float32(b),(o,)).copy()
        return self.layer(name,'dense',x,out=o)
    def reshape(self,name,x,shape):return self.layer(name,'reshape',x,shape=shape)
    def pool(self,name,kind,x,size,stride):return self.layer(name,kind+'pool',x,size=size,stride=stride)


def topological_branch(net):
    # Two bands of grey-level thresholds make faint links less consequential.
    levels=[.27,.51]
    # Retain full 28x28 resolution to see small counters. Use the
    # cubical-complex inclusion/exclusion formula:
    # chi(union of closed ink tiles)= V-E+F, with a vertex present if ANY
    # incident ink tile is on, an edge if EITHER of its two tiles is on.
    # Max-pooling gives the exact OR at *every* threshold simultaneously.
    # The signed sum can also be weighted in height to locate the counter.
    slope=10.
    acts=np.ones((4,1,1,1))*slope
    a=net.conv('cuts','image',acts,[-slope*q+z for q in levels for z in (.5,-.5)])
    a=net.layer('cut_hinges','relu',a)
    w=np.zeros((2,4,1,1));w[0,:2,0,0]=[1,-1];w[1,2:,0,0]=[1,-1]
    cuts=net.conv('soft_ink',a,w)
    padded=net.conv('cell_border',cuts,np.eye(2)[:,:,None,None],padding=1)
    vertices=net.pool('vertices','max',padded,[2,2],[1,1])
    e_h=net.pool('horizontal_edges','max',padded,[2,1],[1,1])
    e_v=net.pool('vertical_edges','max',padded,[1,2],[1,1])
    # Localize Euler mass smoothly by the height of its cell. Symmetric
    # edge/face positions avoid introducing a preferred contour direction.
    # A Gaussian vertical region would give spurious curvature on sides;
    # use an upper+lower partition with a broad linear ramp at the waist.
    def reduce_cells(name,x,h,w,ybase,sign):
        y=(np.arange(h)+ybase)[:,None]
        upper=np.clip(.5+(13.5-y)/10,0,1)
        two=np.stack([np.broadcast_to(upper,(h,w)),np.broadcast_to(1-upper,(h,w))])
        co=sign*np.broadcast_to(two[:,None],(2,2,h,w)).reshape(2,-1)*.5
        return net.dense(name,x,co)
    terms=[reduce_cells('V',vertices,29,29,-.5,1),
           reduce_cells('Eh',e_h,29,30,-.5,-1),
           reduce_cells('Ev',e_v,30,29,-1,-1),
           reduce_cells('Fc',cuts,28,28,0,1)]
    euler=net.layer('euler_halves','add',terms)
    # Class penalties: robust distance to a target interval, not a hard
    # veto. Open 4s and 9s, plugged 8s, and fragmented 5s all occur.
    # For a single connected component chi=1. A loop in the upper half
    # removes roughly one upper unit; a lower loop does the reverse.
    # Overall topology is more reliable than its vertical partition.
    hinges=[];bs=[];coeff=[]
    def interval(k,vec,lo,hi,gain):
        for direction,offset in [(-1,lo),(1,-hi)]:
            hinges.append(np.array(vec)*direction);bs.append(offset)
            row=np.zeros(10);row[k]=-gain;coeff.append(row)
    # Numbers here are class-independent units of correlation per Euler unit.
    for k in range(10):
        lo,hi,g= {
            0:(-.10,.26,.085),1:(.85,1.25,.07),2:(.53,1.27,.045),
            3:(.60,1.32,.05),4:(-.10,1.17,.045),5:(.65,1.84,.05),
            6:(-.12,.38,.060),7:(.78,1.41,.054),8:(-1.15,-.52,.066),9:(-.12,.39,.059)
        }[k]
        interval(k,[1,1],lo,hi,g)
    interval(6,[-1,1],-2,.08,.048)
    interval(9,[1,-1],-2,.08,.048)
    interval(4,[1,-1],-2,.86,.028)
    interval(0,[1,-1],-.63,.63,.025)
    p=net.dense('euler_intervals',euler,hinges,bs)
    p=net.layer('euler_violations','relu',p)
    return net.dense('topology_adjustment',p,np.array(coeff).T)


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    graph=Graph()
    # Boundary orientation and ridge orientation. Spatial standardization is
    # joint over all orientations, not per orientation: preserve the fact
    # that the dominant directions of a 1 and those of a 0 differ.
    ed=graph.conv('pen_filters','image',signed_filters[:,None],stride=2,padding=3)
    ed=graph.layer('positive_responses','relu',ed)
    w=np.eye(NCH)[:,:,None,None]*np.ones((3,3))/9
    ed=graph.conv('integrate',ed,w,padding=1)
    ed=graph.pool('orientation_bins','avg',ed,[2,2],[2,2])
    ed=graph.reshape('all_directions',ed,[1,NCH*F,F])
    ed=graph.layer('normalized_directions','norm',ed)
    # A finer, isotropic correlation gives additional weight to the inside
    # of the stroke and to uninked space, irrespective of tangent direction.
    px=graph.pool('density_bins','avg','image',[2,2],[2,2])
    px=graph.layer('normalized_density','norm',px)

    # Group winners (followed by peak and consensus): a soft alternative to
    # a single nearest template. Different groups use the same grammar and
    # cover the same range of shapes with independent parameter sequences.
    groups,per_group=12,160
    n=groups*per_group
    Wd=np.empty((10*n,NCH*F*F),np.float32)
    Wi=np.empty((10*n,196),np.float32)
    def unit(a):
        a=a.ravel();a=a-a.mean()
        return a/(np.linalg.norm(a)+1.e-12)
    for k,grammar in enumerate(GLYPHS):
        drawings=[pen(spec) for _,_,spec in grammar]
        cum=np.cumsum([howmany for howmany,_,_ in grammar])
        for group in range(groups):
            params=parameters(per_group, 19391+k*761+group*5531)
            for j,u in enumerate(params):
                i=np.searchsorted(cum,u[0]*cum[-1],side='right')
                img=render(drawings[i],grammar[i][1],u,k)
                p=(k*groups+group)*per_group+j
                # Only statistics *within this very template* enter its row.
                # The sqrt of the descriptor dimension converts the test
                # image's unit standard deviation into unit length.
                Wd[p]=(.79/np.sqrt(NCH*F*F))*unit(local_features(img))
                Wi[p]=(.21/14)*unit(img.reshape(14,2,14,2).mean((1,3)))
    d=graph.dense('match_directions',ed,Wd)
    i=graph.dense('match_density',px,Wi)
    m=graph.layer('stroke_match','add',[d,i])
    m=graph.reshape('by_class_and_group',m,[10,groups,per_group])
    m=graph.pool('group_winners','max',m,[1,per_group],[1,per_group])
    peak=graph.pool('best_match','max',m,[groups,1],[groups,1])
    mean=graph.pool('consistent_match','avg',m,[groups,1],[groups,1])
    peak=graph.reshape('peak_vector',peak,[10]);mean=graph.reshape('consensus_vector',mean,[10])
    peak=graph.dense('peak_vote',peak,.70*np.eye(10))
    mean=graph.dense('consensus_vote',mean,.30*np.eye(10))
    adj=topological_branch(graph)
    graph.layer('class_scores','add',[peak,mean,adj])
    (out/'network.json').write_text(json.dumps({'layers':graph.layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**graph.w)

if __name__=='__main__':
    main(sys.argv[1])
