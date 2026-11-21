"""Construct independent pen-path templates; no input files or fonts are used."""
import math
import sys
from pathlib import Path
import numpy as np

# A little pen plotter. Coordinates run from left to right and top to bottom.
# Each entry is (relative frequency, strokes) in an abstract unit writing box.
# L, Q and C denote a line, a quadratic and a cubic Bezier respectively.
styles = [
    [],  # zero is drawn parametrically below
    [
        (17, 'M 55 0 L 44 100'),
        (9, 'M 48 0 Q 55 44 46 100'),
        (6, 'M 54 0 Q 44 69 50 100'),
        (7, 'M 39 18 Q 49 8 53 0 Q 56 21 46 100'),
        (6, 'M 24 32 L 55 0 L 48 100'),
        (3, 'M 10 41 Q 33 24 57 0 L 51 100'),
        (2, 'M 30 22 L 54 0 L 49 100 M 23 99 L 76 99'),
        (1, 'M 48 0 Q 54 59 44 96 Q 43 102 70 98'),
    ],
    [
        (8, 'M 5 24 C 23 -14 89 -7 85 22 C 85 43 40 65 9 98 Q 44 91 98 98'),
        (7, 'M 8 23 C 17 3 57 -9 72 8 C 93 31 49 61 13 91 Q 1 104 29 97 L 99 91'),
        (6, 'M 12 24 C 19 -3 66 -8 78 14 C 93 38 34 79 5 96 Q 47 102 99 92'),
        (5, 'M 12 29 C -2 -1 75 -14 77 18 C 79 43 28 80 10 91 C -4 104 15 106 40 97 Q 75 84 99 88'),
        (5, 'M 6 18 C 34 -8 79 -6 76 22 C 72 46 8 78 8 92 C 12 115 38 94 29 83 C 24 75 10 84 17 90 Q 44 105 99 86'),
        (4, 'M 9 24 C 15 -9 72 -10 73 19 C 75 45 26 82 5 93 C -5 110 24 100 41 87 Q 60 79 99 94'),
        (4, 'M 4 23 Q 25 -5 51 3 Q 99 11 67 48 L 9 94 Q 50 89 96 100'),
        (3, 'M 1 16 Q 31 -2 66 2 Q 94 6 69 36 L 9 98 L 98 94'),
        (3, 'M 13 29 C -3 2 44 -10 65 7 C 91 27 52 66 12 93 Q 58 80 98 97'),
        (2, 'M 2 8 Q 42 -1 81 4 Q 98 10 65 44 L 5 98 L 99 95'),
        (2, 'M 18 32 C 1 23 12 3 39 2 C 105 -6 96 42 54 68 L 9 98 Q 54 96 98 83'),
    ],
    [
        (8, 'M 7 14 C 39 -8 88 -4 79 19 Q 75 32 38 48 C 73 35 99 53 83 78 C 66 107 27 105 8 92'),
        (7, 'M 4 12 C 28 -3 84 -8 79 14 Q 72 34 33 47 C 68 40 89 50 86 71 C 84 93 48 105 14 97'),
        (5, 'M 6 10 Q 40 -3 84 3 Q 77 22 38 46 C 69 40 96 50 88 75 C 80 99 33 108 9 91'),
        (4, 'M 4 4 L 87 3 L 38 45 C 89 36 101 68 72 88 Q 43 110 5 94'),
        (6, 'M 16 18 C 34 -8 82 -3 79 18 C 77 33 60 41 42 47 C 67 45 90 58 78 83 C 66 108 26 106 10 87'),
        (4, 'M 2 11 C 28 -2 72 -7 73 12 C 71 34 32 52 30 45 C 27 37 78 38 89 57 C 109 89 58 110 14 96'),
        (3, 'M 18 10 C 45 -6 80 -2 76 19 Q 73 40 43 51 Q 81 44 88 59 C 102 84 53 109 8 94'),
        (2, 'M 9 22 Q 39 -8 66 5 Q 96 18 43 47 Q 97 42 82 76 Q 57 110 3 91'),
        (2, 'M 7 8 Q 46 -2 80 7 L 39 43 C 71 31 103 54 81 82 C 60 107 15 107 7 83'),
    ],
    [
        (8, 'M 28 3 L 5 56 Q 33 62 96 51 M 76 0 L 64 100'),
        (6, 'M 23 0 Q 9 30 8 50 Q 9 61 33 58 L 91 49 M 77 9 Q 69 46 68 100'),
        (6, 'M 30 1 Q 12 29 8 46 Q 2 67 39 57 L 75 48 M 77 3 L 68 100'),
        (5, 'M 43 2 L 6 57 L 98 55 M 75 6 L 69 100'),
        (4, 'M 13 3 Q 4 37 9 48 Q 17 62 71 47 L 79 4 L 67 100'),
        (3, 'M 20 2 L 17 47 Q 34 63 73 50 M 75 4 L 73 100'),
        (4, 'M 68 0 L 5 58 L 99 56 M 68 0 L 64 100'),
        (3, 'M 75 0 Q 33 25 9 55 L 85 55 M 74 1 L 63 100'),
        (3, 'M 50 6 L 7 53 Q 46 58 72 48 L 76 0 L 65 100'),
        (3, 'M 22 0 Q -4 42 10 50 Q 29 63 73 43 M 72 14 Q 81 42 65 100'),
        (2, 'M 21 0 Q 3 42 10 52 Q 25 65 67 45 Q 76 29 73 14 L 62 100'),
    ],
    [
        (8, 'M 93 1 Q 57 8 22 7 L 14 46 C 48 27 85 41 84 64 C 85 91 42 110 5 92'),
        (7, 'M 22 6 L 14 44 C 43 34 78 37 85 55 C 99 88 45 109 8 94 M 19 6 Q 51 11 95 0'),
        (5, 'M 94 0 Q 58 8 27 6 Q 16 23 12 45 C 48 31 83 44 82 66 C 83 90 41 109 7 92'),
        (5, 'M 90 2 L 18 4 L 16 46 Q 47 30 73 44 C 113 69 66 112 13 94'),
        (4, 'M 96 2 L 33 5 Q 24 22 20 39 C 47 35 89 48 83 71 C 77 95 27 108 5 88'),
        (5, 'M 95 5 C 57 5 23 -7 17 12 C -1 45 50 40 73 54 C 106 82 55 109 6 91'),
        (3, 'M 90 6 C 61 -7 23 0 13 20 C -9 49 53 40 75 54 C 105 82 60 108 8 94'),
        (3, 'M 19 9 Q 9 22 12 38 Q 51 28 76 54 C 98 82 42 110 9 90 M 31 5 Q 57 11 96 0'),
        (3, 'M 96 0 Q 66 10 25 5 Q 10 25 13 37 Q 49 41 64 53 C 108 85 48 111 2 92'),
        (2, 'M 88 1 L 17 7 L 15 46 C 40 34 85 38 84 64 C 85 87 45 110 9 94 Q -3 86 6 79'),
    ],
    [
        (7, 'M 79 0 C 52 9 19 41 14 68 C 6 95 34 109 62 95 C 89 82 93 55 71 51 Q 42 44 17 75'),
        (7, 'M 76 0 Q 30 18 16 55 C 0 98 40 112 70 92 C 93 76 87 53 66 51 Q 40 50 18 70'),
        (6, 'M 62 1 Q 22 30 18 69 C 12 97 41 107 65 97 C 89 86 96 57 70 53 Q 47 48 19 79'),
        (5, 'M 77 1 Q 41 25 15 68 C -1 94 38 111 69 94 C 99 76 88 52 68 52 Q 41 51 12 79'),
        (4, 'M 85 9 C 78 -6 56 0 41 14 Q 14 40 15 74 C 13 102 59 109 76 87 C 101 54 64 37 39 55 L 21 73'),
        (4, 'M 57 0 C 31 13 20 52 20 76 C 21 97 42 106 65 96 Q 91 81 78 57 Q 64 40 31 65'),
        (3, 'M 64 0 L 20 59 Q 1 90 36 98 Q 65 108 82 82 C 99 51 59 47 31 66 L 12 82'),
        (3, 'M 64 0 Q 27 32 18 68 C 14 102 48 110 71 92 C 89 80 90 54 72 50 Q 53 45 36 58'),
        (2, 'M 72 0 Q 23 28 15 69 C 11 103 52 112 76 88 C 92 69 74 46 53 53 Q 34 58 20 78 L 42 72'),
    ],
    [
        (9, 'M 7 5 Q 42 0 87 5 L 39 100'),
        (8, 'M 3 9 C 18 3 51 0 83 6 Q 63 43 38 100'),
        (6, 'M 8 4 Q 45 16 88 4 Q 58 51 42 100'),
        (5, 'M 8 19 L 11 3 Q 46 0 86 8 Q 62 45 38 100'),
        (5, 'M 5 6 Q 31 -2 57 5 Q 75 11 87 4 L 36 99'),
        (4, 'M 8 4 Q 43 10 85 4 Q 64 43 39 100 M 29 51 L 77 49'),
        (4, 'M 6 18 Q 9 0 26 5 Q 59 14 87 3 L 38 100 M 29 53 Q 52 56 76 51'),
        (3, 'M 5 0 Q 39 15 88 8 Q 61 48 46 100'),
        (3, 'M 9 13 Q 24 -2 45 3 L 85 6 Q 65 45 47 80 Q 36 102 30 100'),
        (2, 'M 9 19 L 9 3 L 87 4 L 43 94 Q 41 100 48 100'),
    ],
    [],  # a selection of single-line knots and separately drawn loops below
    [
        (9, 'M 79 13 C 61 -8 28 -1 15 19 C -9 54 35 72 62 50 Q 83 30 78 9 Q 81 49 64 100'),
        (7, 'M 75 7 C 55 -7 20 5 12 26 C 1 51 32 66 59 48 Q 77 33 78 11 L 66 100'),
        (6, 'M 76 12 C 57 -13 16 4 13 29 C 8 58 50 62 70 42 L 80 19 Q 77 59 57 100'),
        (5, 'M 76 18 C 73 -7 36 -3 21 12 C -8 40 21 67 54 52 Q 80 39 76 6 L 77 99'),
        (5, 'M 75 13 C 56 -9 22 3 13 24 C 2 55 40 65 66 44 Q 78 31 77 9 Q 92 80 56 97 Q 34 106 12 87'),
        (4, 'M 75 11 C 54 -10 20 5 14 24 C 7 48 33 66 60 49 Q 76 39 77 18 Q 84 77 53 96 Q 38 105 28 96'),
        (3, 'M 60 3 C 36 -3 12 14 11 36 C 13 62 47 58 66 36 L 79 7 L 66 100'),
        (3, 'M 67 4 Q 35 -8 13 25 Q -1 54 28 56 Q 58 62 79 15 Q 85 54 66 100'),
        (3, 'M 73 8 Q 46 -7 24 12 Q 3 28 11 46 Q 33 69 72 44 M 77 7 Q 77 49 60 100'),
        (2, 'M 80 18 Q 70 -6 42 3 Q 14 10 11 35 Q 13 61 45 53 Q 70 48 77 17 Q 78 72 47 100'),
    ],
]
styles[8] = [
    (7, 'M 54 45 C 28 37 9 25 23 10 C 42 -7 78 -2 79 13 C 85 29 55 43 41 55 C 12 74 4 91 25 98 C 47 109 82 93 82 73 C 85 57 65 49 54 45'),
    (7, 'M 63 3 C 32 -7 6 14 19 30 C 34 47 78 49 83 71 C 90 92 56 107 29 98 C 7 91 12 72 38 53 C 67 33 91 15 72 6 Q 61 -1 48 3'),
    (6, 'M 62 4 C 42 -7 11 7 17 23 C 21 41 66 49 78 65 C 98 94 52 110 24 97 C -2 82 27 61 52 45 C 84 24 92 1 63 2'),
    (5, 'M 54 46 C 27 41 12 21 25 8 C 42 -6 75 -1 78 15 C 83 30 66 41 48 48 C 19 57 3 77 16 91 C 31 110 73 102 83 84 C 99 60 74 47 50 46'),
    (4, 'M 60 3 C 38 -8 13 8 17 28 C 20 44 64 55 76 72 C 92 96 51 109 25 95 C 5 78 31 60 51 44 C 65 34 89 15 75 6'),
    (4, 'M 46 45 C 67 39 88 18 73 7 C 49 -8 16 7 17 25 C 17 39 39 46 49 49 C 74 57 94 74 77 91 C 57 109 20 103 14 89 C 3 69 30 52 46 45'),
    (3, 'M 51 44 C 29 44 14 31 20 16 C 27 -4 65 -9 76 9 C 89 27 69 46 51 44 M 50 45 C 18 50 6 72 15 88 C 27 108 64 104 80 88 C 97 65 79 50 50 45'),
    (3, 'M 48 49 C 15 36 6 16 30 4 C 48 -5 77 -1 81 12 C 92 30 55 48 37 60 C 5 85 14 102 42 100 C 80 102 97 77 72 58 Q 57 52 48 49'),
    (2, 'M 55 43 C 78 28 79 7 64 2 C 45 -5 21 4 19 18 C 15 36 54 46 71 63 C 100 88 67 108 35 98 C 0 88 19 65 55 43'),
]

def parse(s):
    tok = s.split()
    i, curves = 0, []
    while i < len(tok):
        cmd = tok[i]; i += 1
        n = {'M':1, 'L':1, 'Q':2, 'C':3}[cmd]
        pts = np.array(tok[i:i+2*n], dtype=float).reshape(n,2); i += 2*n
        if cmd == 'M':
            curves.append([])
        else:
            curves[-1].append(np.concatenate([prev[None,:], pts]))
        prev = pts[-1]
    return curves

styles = [[(w,parse(path)) for w,path in options] for options in styles]
rng = np.random.default_rng(371904)

def bounded():
    return np.clip(rng.standard_normal(), -2.3, 2.3)

def stroke_points(curves):
    result, ids = [], []
    for j,stroke in enumerate(curves):
        # Move each knot a little. The same knot is shared by adjoining spans.
        drift = rng.normal(size=(len(stroke)+1,2))*[3.4,2.5]
        if np.linalg.norm(stroke[0][0]-stroke[-1][-1]) < 5:
            drift[-1] = drift[0] + rng.normal(size=2)*1.5
        line = []
        for i, span in enumerate(stroke):
            n = len(span)-1
            frac = np.linspace(0,1,n+1)[:,None]
            p = span + drift[i]*(1-frac) + drift[i+1]*frac
            if n>1:
                p[1:-1] += rng.normal(size=(n-1,2))*[6.,4.5]
            length = np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
            num = max(3,int(length*.91))
            t = np.linspace(0,1,num)[:,None]
            v = sum(math.comb(n,k)*t**k*(1-t)**(n-k)*p[k] for k in range(n+1))
            line.append(v)
        line = np.concatenate(line)
        result.append(line); ids.append(np.full(len(line),j))
    return np.concatenate(result), np.concatenate(ids)

def zero():
    # Ovals with off-centre widest points, possible seam/gap and a small overshoot.
    a = rng.uniform(-2.6,-.35)
    if rng.random()<.13:
        a += rng.uniform(1.6,4.6)
    angle = rng.uniform(5.4,6.02) if rng.random()<.13 else rng.uniform(6.12,7.0)
    t = np.linspace(a,a-angle,280)
    power = rng.uniform(.79,1.24)
    co,si = np.cos(t),np.sin(t)
    x = 50 + 46*np.sign(co)*abs(co)**power*(1 + bounded()*.12*si)
    y = 50 + 50*np.sign(si)*abs(si)**power
    end = np.linspace(0,1,len(t)); d = rng.uniform(-6,9)*end**9
    x += d*co; y += d*si
    return np.stack((x,y),1), np.zeros(len(t),int)

# Use a finite collection of brush stamps to rasterize a continuous pen line.
aoff,boff = np.mgrid[-4:5,-4:5]
dx,dy = aoff.ravel()[None,:],boff.ravel()[None,:]

def render(c):
    if c==0:
        p,ids=zero()
    else:
        items=styles[c]; r=rng.random()*sum(w for w,s in items)
        for w,s in items:
            r-=w
            if r<=0: break
        p,ids=stroke_points(s)
    x,y=p[:,0]*.01-.5,p[:,1]*.01-.5
    # Vertical balance, curvature and tapered width of the writing box. These
    # continuous deformations are shared along every stroke (including joins).
    delta = np.clip(bounded()*.115,-.23,.23)
    y = y + delta*np.maximum(0,1-2*abs(y))
    x = x*(1.+bounded()*.25*y + bounded()*.09*np.cos(y*6))
    x = x + bounded()*.055*np.cos(y*5) + bounded()*.035*np.sin(y*6)
    y = y + x*bounded()*.15 + bounded()*.045*np.cos(x*5)
    if c==8 and rng.random()<.35: x=-x
    asp = (.70 if c!=1 else .85)*np.exp(.27*bounded())
    x = x*asp
    slope = -.13+bounded()*.24
    x = x+slope*y
    pressure = rng.uniform(1.25,4.0) if rng.random()<.86 else rng.uniform(.85,4.9)
    size = (20.-pressure)/max(np.ptp(x),np.ptp(y),.01)
    x = (x-.5*(x.max()+x.min()))*size + 13.5 + rng.uniform(-.45,.45)
    y = (y-.5*(y.max()+y.min()))*size + 13.5 + rng.uniform(-.45,.45)
    t=np.linspace(0,1,len(y))
    rad = pressure*.5*(1 + rng.uniform(-.19,.19)*np.cos(2*np.pi*t+rng.uniform(-3,3))
                        +rng.uniform(-.12,.12)*(2*t-1))
    if ids[-1]>0:
        rad *= np.where(ids!=0, rng.uniform(.81,1.17),1.)
    # Trim the extremes (pen lift) in a fraction of the drawings.
    if rng.random()<.62:
        rad *= 1 - rng.uniform(.12,.48)*(np.exp(-t/.037)+np.exp((t-1)/.038))
    ix=x.round().astype(np.int32)[:,None]+dx
    iy=y.round().astype(np.int32)[:,None]+dy
    sx=rng.uniform(.87,1.13)
    sq = ((x[:,None]-ix)*sx)**2+((y[:,None]-iy)/sx)**2
    amp = np.clip(.5+(rad[:,None]-np.sqrt(sq))/rng.uniform(.8,1.45),0,1)
    # Border clipping only affects points with zero coverage.
    idx = np.clip(ix,0,27)+28*np.clip(iy,0,27)
    raw = np.zeros(784,np.uint8)
    np.maximum.at(raw,idx.ravel(),np.rint(amp*255).astype(np.uint8).ravel())
    return raw.reshape(28,28)

# The restricted inference code is self contained; constants contain only
# separate drawings (their contiguous groups of equal size give the labels).
solution = r'''import numpy as np

axis = np.arange(28,dtype=np.float32)
wide = np.arange(-8,36,dtype=np.float32)

def quantiles(p):
    cs = np.cumsum(p,axis=1)
    def loc(f):
        v = cs[:,-1:] * f
        j = np.minimum(np.sum(cs<v,axis=1),p.shape[1]-1)
        n = np.arange(len(p))
        return (j + (v[:,0]-cs[n,j])/np.maximum(.0001,p[n,j])+.5).astype(np.float32)
    return loc(.008),loc(.992)

def canon(a):
    n = np.arange(len(a))
    a = a/np.maximum(.001,np.max(a,axis=(1,2),keepdims=True))
    rows = a.sum(axis=2); cols=a.sum(axis=1)
    mass = np.maximum(.001,rows.sum(1))
    cy = (rows*axis).sum(1)/mass
    cx = (cols*axis).sum(1)/mass
    yd = axis-cy[:,None]; xd = axis-cx[:,None]
    shear = np.clip((a*(yd[:,:,None]*xd[:,None,:])).sum(axis=(1,2))/np.maximum(.001,(rows*yd**2).sum(1)),-1.4,1.4)
    offset = yd*shear[:,None]
    grid = wide+offset[:,:,None]; j = np.floor(grid).astype(np.int32); f=grid-j
    yy=np.arange(28)
    proj = ((1-f)*a[n[:,None,None],yy[:,None],np.clip(j,0,27)] + f*a[n[:,None,None],yy[:,None],np.clip(j+1,0,27)]).sum(1).astype(np.float32)
    l,r=quantiles(proj); l-=8; r-=8
    top,bot=quantiles(rows)
    height=np.maximum(1.,bot-top); width=np.maximum(1.,r-l)
    ratio=np.clip(width/height,.06,3.5)
    nh = 21./np.maximum(1.,ratio/1.45)**.28
    nw = np.clip(nh*ratio*(.73/np.maximum(.15,ratio))**.64,2.7,24.)
    sy = (axis-13.5)*(height/nh)[:,None]+((top+bot)*.5)[:,None]
    sx = (axis-13.5)*(width/nw)[:,None]+((l+r)*.5)[:,None]
    sx = sx[:,None,:]+shear[:,None,None]*(sy-cy[:,None])[:,:,None]
    py=np.floor(sy).astype(np.int32); px=np.floor(sx).astype(np.int32)
    fy=(sy-py).astype(np.float32)[:,:,None]; fx=(sx-px).astype(np.float32)
    p0=np.clip(py,0,27)[:,:,None]; p1=np.clip(py+1,0,27)[:,:,None]
    q0=np.clip(px,0,27); q1=np.clip(px+1,0,27)
    im=(1-fy)*((1-fx)*a[n[:,None,None],p0,q0]+fx*a[n[:,None,None],p0,q1])+fy*((1-fx)*a[n[:,None,None],p1,q0]+fx*a[n[:,None,None],p1,q1])
    meta=np.stack((np.log(ratio),shear,np.log(np.clip(mass/(width*height),.025,2))),axis=1)
    return im,meta

def kernel(pos,sigma):
    v=np.exp(-.5*((np.asarray(pos,dtype=np.float32)[:,None]-axis)/sigma)**2)
    return v/v.sum(1,keepdims=True)

fineker=kernel(1.5+2*np.arange(13),1.10)
coarseker=kernel(1.5+4*np.arange(7),2.05)

def pool(k,f):
    return (k@f)@k.T

def trapped(a,level):
    space=a<level
    exterior=space.copy(); exterior[:,1:-1,1:-1]=False
    for i in range(58):
        v=exterior.copy()
        v[:,1:] |= exterior[:,:-1]; v[:,:-1] |= exterior[:,1:]
        v[:,:,1:] |= exterior[:,:,:-1]; v[:,:,:-1] |= exterior[:,:,1:]
        v &= space
        if i%8==7 and np.array_equal(v,exterior): break
        exterior=v
    return space & ~exterior

def features(raw):
    a,meta=canon(raw)
    v=np.pad(a,((0,0),(1,1),(1,1)))
    im=(4*a+v[:,1:-1,:-2]+v[:,1:-1,2:]+v[:,:-2,1:-1]+v[:,2:,1:-1])*.125
    v=np.pad(im,((0,0),(1,1),(1,1)))
    gx=v[:,1:-1,2:]-v[:,1:-1,:-2]; gy=v[:,2:,1:-1]-v[:,:-2,1:-1]
    floor=.66*np.sqrt(gx**2+gy**2)
    cparts=[]; fparts=[]
    for z in (gx,.70710678*(gx+gy),gy,.70710678*(gy-gx)):
        for sign in (1.,-1.):
            h=np.maximum(0.,(sign*z-floor)*2.94)
            cparts.append(np.sqrt(pool(coarseker,h)+.008)-.08944272)
            fparts.append(np.sqrt(pool(fineker,h)+.008)-.08944272)
    cparts.append(.55*pool(coarseker,a)); fparts.append(.55*pool(fineker,a))
    # A whitespace component closed to the exterior, or open to one side only.
    hole=.5*(trapped(a,.28).astype(np.float32)+trapped(a,.52))
    cparts.append(.50*pool(coarseker,hole))
    bars = (np.maximum.accumulate(a,2),np.maximum.accumulate(a[:,:,::-1],2)[:,:,::-1],
            np.maximum.accumulate(a,1),np.maximum.accumulate(a[:,::-1],1)[:,::-1])
    white=1-a
    m=np.minimum(np.minimum(bars[0],bars[1]),np.minimum(bars[2],bars[3]))
    cparts.append(.28*pool(coarseker,white*m))
    for j in range(4):
        b=np.minimum(np.minimum(bars[(j+1)%4],bars[(j+2)%4]),bars[(j+3)%4])
        cparts.append(.34*pool(coarseker,white*b*(1-bars[j])))
    coarse=np.stack(cparts,axis=1).reshape(len(a),-1)
    coarse/=np.maximum(.0001,np.sqrt((coarse**2).sum(1,keepdims=True)))
    fine=np.stack(fparts,axis=3)
    fine/=np.maximum(.0001,np.sqrt((fine**2).sum(axis=(1,2,3),keepdims=True)))
    return coarse,np.clip(np.rint(1900*fine),0,255).astype(np.uint8),meta

nper=CONSTANTS['drawings'].shape[1]
num=10*nper
rawbank=CONSTANTS['drawings'].reshape(num,28,28)
bank=np.empty((num,15*49),np.float32)
inkbank=np.empty((num,13,13,9),np.uint8)
measure=np.empty((num,3),np.float32)
for start in range(0,num,64):
    section=slice(start,min(start+64,num))
    bank[section],inkbank[section],measure[section]=features(rawbank[section].astype(np.float32)/255.)

def patchmean(e):
    # Compare overlapping neighbourhoods, not isolated gradient samples.
    p=np.pad(e,((0,0),(0,0),(1,1),(1,1)))
    v=.36*e+.12*(p[:,:,:-2,1:-1]+p[:,:,2:,1:-1]+p[:,:,1:-1,:-2]+p[:,:,1:-1,2:])
    return v+.04*(p[:,:,:-2,:-2]+p[:,:,2:,:-2]+p[:,:,:-2,2:]+p[:,:,2:,2:])

def elastic(q,ids):
    b=len(q); k=ids.shape[1]; area=13
    temp=inkbank[ids].astype(np.float32)*(1./1900.)
    pad=np.pad(q.astype(np.float32)*(1./1900.),((0,0),(1,1),(1,1),(0,0)))
    best=np.full((b,k,area,area),4.,np.float32); reverse=best.copy()
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            v=pad[:,1+dy:14+dy,1+dx:14+dx,:]
            dif=temp-v[:,None,:,:,:]
            err=np.einsum('bkhwc,bkhwc->bkhw',dif,dif)
            cost=patchmean(err)+(.028/(13*13))*(dx*dx+dy*dy)
            best=np.minimum(best,cost)
            ty=slice(max(0,dy),min(13,13+dy)); tx=slice(max(0,dx),min(13,13+dx))
            sy=slice(max(0,-dy),min(13,13-dy)); sx=slice(max(0,-dx),min(13,13-dx))
            reverse[:,:,ty,tx]=np.minimum(reverse[:,:,ty,tx],cost[:,:,sy,sx])
    return .5*(best+reverse).sum(axis=(2,3))

def classify(a):
    desc,q,meta=features(a)
    b=len(q)
    # Use a fixed shape for the matrix product, independent of the outer batch.
    query=np.zeros((48,desc.shape[1]),np.float32); query[:b]=desc
    dist=2.-2.*(query@bank.T)[:b]
    extra=(.032*np.minimum(1.6,abs(meta[:,0:1]-measure[:,0]))**2+
           .013*np.minimum(1.6,abs(meta[:,1:2]-measure[:,1]))**2+
           .021*np.minimum(1.4,abs(meta[:,2:3]-measure[:,2]))**2)
    dist+=extra
    d=dist.reshape(b,10,nper)
    # Keep several independent exemplars per promising class, rather than
    # allowing a common pen style to crowd out a rarer competing hypothesis.
    cls=np.argsort(d.min(2),axis=1,kind='stable')[:,:4]
    n=np.arange(b)[:,None]
    dv=d[n,cls]
    ind=np.argpartition(dv,7,axis=2)[:,:,:8]+nper*cls[:,:,None]
    ind=ind.reshape(b,-1)
    cost=dist[n,ind]*.50+extra[n,ind]*.50
    for st in range(0,b,8):
        ss=slice(st,st+8)
        cost[ss]+=elastic(q[ss],ind[ss])
    scores=np.sort(cost.reshape(b,4,8),axis=2)
    loss=.68*scores[:,:,0]+.22*scores[:,:,1]+.10*scores[:,:,2]
    return cls[np.arange(b),np.argmin(loss,axis=1)]

def predict(x):
    result=np.empty(len(x),np.int64)
    for st in range(0,len(x),48):
        result[st:st+48]=classify(x[st:st+48,0])
    return result
'''

def main():
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    n=4608
    bank=np.empty((10,n,28,28),np.uint8)
    for c in range(10):
        for j in range(n): bank[c,j]=render(c)
    np.savez(out/'constants.npz',drawings=bank)
    (out/'solution.py').write_text(solution)

if __name__=='__main__':
    main()
