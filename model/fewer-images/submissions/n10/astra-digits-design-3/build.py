"""Build a digit recognizer from pen geometry. There are no external inputs."""
import json
import sys
from pathlib import Path
import numpy as np

# Drawing coordinates: 0..1 in y from ascender to baseline; x is in
# units of the *nominal width*, not pixels. '|' means lift the pen.
# Each string is its own handwritten construction, not an outline from data.
PATHS = {
    0: [
        ('M .50 0 C .80 -.04 1 .17 1 .48 C 1 .78 .81 1 .51 1 C .18 1.01 0 .79 0 .51 C -.01 .23 .19 .02 .5 0', 7),
        ('M .63 .05 C .30 -.06 .04 .16 .02 .53 C 0 .82 .16 1.02 .43 1 C .75 1.02 1 .72 1 .39 C 1 .12 .86 -.02 .65 .01 L .51 .06', 3),
        ('M .36 .01 C .10 .04 .04 .28 .01 .52 C -.04 .83 .11 1 .38 1 C .73 1.01 .96 .75 .99 .47 C 1.04 .15 .88 .01 .61 0 Q .38 -.02 .30 .14', 2),
        ('M .52 .02 C .18 .02 .04 .22 .01 .63 Q -.02 1 .32 1 C .70 1.03 .98 .66 1 .34 Q 1.01 -.02 .64 .01 Q .53 .02 .46 .10', 2),
    ],
    1: [
        ('M .50 0 L .50 1', 12),
        ('M .58 0 Q .40 .55 .50 1', 4),
        ('M .49 .02 Q .53 -.04 .55 .03 Q .55 .36 .47 1', 3),
        ('M .23 .16 L .52 0 L .52 1', 4),
        ('M .00 .27 Q .29 .18 .52 0 L .52 1', 3),
        ('M .40 .09 L .53 0 L .48 .87 Q .47 .95 .61 1', 2),
        ('M .03 .23 L .47 0 L .49 1 | M .07 1 Q .53 .97 .92 1', 1.4),
        ('M .51 0 L .45 .98 L .80 1', .6),
    ],
    2: [
        ('M .01 .22 C .14 .04 .38 -.03 .61 .01 C .83 .05 .94 .15 .91 .29 C .88 .47 .53 .65 .29 .80 L .02 .98 Q .40 .93 1 .97', 4),
        ('M .02 .26 C .11 .05 .40 -.04 .62 .01 C .91 .05 .94 .27 .78 .45 C .61 .63 .22 .83 .03 .98 Q .55 1 1 .90', 3),
        ('M .10 .13 C .30 -.04 .67 -.03 .80 .12 C 1.01 .34 .48 .68 .10 .90 Q -.09 1.02 .24 .98 Q .69 .91 1 .98', 3),
        ('M .14 .25 C -.03 .19 .14 .02 .40 .00 C .66 -.02 .85 .12 .83 .31 C .80 .52 .49 .84 .18 .97 C -.10 1.07 -.05 .83 .16 .79 C .40 .73 .71 1.04 1 .90', 4),
        ('M .03 .25 C .00 .07 .27 -.01 .47 .01 C .81 .05 .84 .26 .71 .48 C .53 .72 .19 1.01 .05 .97 C -.11 .92 .05 .78 .22 .80 Q .67 .88 1 .95', 3),
        ('M .05 .12 Q .41 -.12 .72 .05 C 1.11 .29 .45 .65 .00 1 L .98 .99', 2),
        ('M .01 .28 C -.03 .05 .27 -.03 .53 .02 C .77 .05 .90 .19 .82 .38 Q .67 .64 .16 .97 Q .53 .89 .99 1', 2),
        ('M .15 .17 C .41 -.02 .68 -.03 .79 .15 C .91 .33 .63 .60 .33 .81 Q .08 1.02 .02 .95 Q -.01 .85 .17 .84 Q .41 .81 .69 .91 Q .83 .96 1 .91', 2),
    ],
    3: [
        ('M .04 .10 C .35 -.04 .82 -.03 .88 .13 C .98 .31 .61 .43 .36 .48 C .73 .41 1 .48 .98 .67 C .95 .91 .58 1.04 .28 .99 Q .10 .97 .01 .89', 4),
        ('M .07 .06 Q .53 -.01 .92 .03 Q .70 .28 .32 .46 C .74 .37 1.05 .51 .95 .72 C .81 1.01 .34 1.09 .01 .89', 4),
        ('M .01 .15 Q .46 -.06 .89 .04 L .35 .49 C .81 .38 1.05 .55 .91 .76 Q .62 1.12 .01 .95', 3),
        ('M .01 .19 C .22 -.04 .73 -.04 .87 .13 C 1.01 .31 .68 .49 .39 .50 C .71 .42 1.02 .53 .97 .76 C .90 .99 .39 1.09 .01 .88', 5),
        ('M .04 .06 Q .49 -.02 .95 .02 L .46 .44 Q .35 .52 .59 .47 C .91 .40 1 .58 .91 .77 C .78 1.02 .31 1.05 .03 .94', 2),
        ('M .06 .07 C .43 -.06 .81 -.03 .84 .15 C .87 .31 .61 .44 .30 .51 Q .11 .55 .15 .48 Q .29 .39 .60 .48 C 1.32 .64 .95 1.02 .49 1.00 Q .17 1 .01 .84', 2),
        ('M .11 .10 Q .58 -.10 .86 .06 Q 1.05 .23 .39 .46 Q 1 .46 .95 .72 C .91 .99 .42 1.11 .02 .87', 2),
    ],
    4: [
        ('M .28 .00 Q .13 .28 .04 .48 Q -.06 .66 .13 .65 L .98 .60 | M .79 .01 L .75 1', 5),
        ('M .12 .10 L .03 .55 Q -.01 .65 .18 .62 L .96 .52 | M .84 .00 L .81 1', 3),
        ('M .63 .00 L .01 .62 L 1 .60 | M .83 .16 L .78 1', 4),
        ('M .74 .00 L .01 .61 L 1 .60 | M .79 .03 L .77 1', 3),
        ('M .78 1 L .80 .00 L .03 .62 L 1 .59', 4),
        ('M .30 .02 Q .06 .34 .03 .47 C -.06 .76 .43 .68 .74 .42 L .92 .13 L .78 1', 4),
        ('M .09 .03 Q .02 .33 .04 .50 Q .05 .68 .28 .62 Q .61 .57 .84 .27 L .90 .09 L .77 1', 3),
        ('M .30 .04 Q .11 .32 .08 .55 Q .10 .65 .31 .59 L .85 .49 | M .83 .00 Q .69 .60 .73 1', 3),
        ('M .01 .06 L .04 .52 Q .04 .63 .30 .59 L .86 .48 L .82 .01 L .77 1', 1.5),
        ('M .68 0 L .05 .57 L .89 .51 | M .68 .24 L .70 1', 2),
    ],
    5: [
        ('M .98 .00 Q .59 .07 .17 .04 L .09 .47 C .37 .32 .78 .38 .89 .58 C 1.05 .85 .59 1.04 .26 .98 Q .10 .95 .01 .86', 5),
        ('M .99 .02 L .22 .04 L .07 .43 C .46 .32 .85 .43 .90 .65 C .93 .92 .42 1.09 .02 .91', 4),
        ('M .23 .04 Q .17 .26 .10 .40 C .39 .33 .83 .37 .91 .61 C 1.01 .93 .39 1.09 .03 .91 | M .11 .06 Q .54 .05 .98 .00', 4),
        ('M .98 .01 Q .53 .07 .18 .07 Q .05 .25 .06 .37 C .14 .47 .57 .39 .78 .53 C 1.23 .81 .59 1.06 .23 .98 Q .08 .94 .01 .84', 3),
        ('M .93 .03 C .56 -.03 .19 .00 .15 .19 Q .10 .34 .47 .42 C .74 .47 .94 .58 .88 .75 C .78 .96 .31 1.07 .02 .92', 3),
        ('M .92 .00 Q .48 .08 .23 .07 L .12 .32 Q .20 .38 .51 .43 C 1.18 .60 .88 .94 .43 .99 Q .15 1.01 .02 .92', 2),
        ('M .30 .06 L .12 .43 Q .37 .41 .60 .47 C 1.04 .57 1.02 .85 .64 .97 Q .28 1.07 .00 .87 | M .18 .06 Q .62 .12 1 .01', 3),
        ('M .95 .00 Q .63 .04 .25 .03 L .08 .47 C .30 .37 .56 .36 .76 .46 C 1.12 .66 .86 .91 .55 .98 Q .22 1.05 .03 .90', 2),
    ],
    6: [
        ('M .79 .00 C .44 .09 .13 .34 .06 .61 C -.04 .89 .21 1.04 .52 .99 C .80 .96 1 .82 .93 .67 C .83 .47 .48 .48 .26 .59 Q .10 .67 .11 .76', 5),
        ('M .65 .00 Q .26 .25 .09 .53 C -.12 .87 .15 1.06 .52 .99 C .84 .94 1.04 .72 .88 .60 C .74 .48 .45 .55 .26 .69 L .14 .81', 4),
        ('M .85 .01 C .63 -.07 .32 .14 .16 .40 C -.07 .75 .05 .99 .43 1 C .74 1.02 1 .82 .90 .66 C .79 .48 .41 .51 .13 .68', 4),
        ('M .66 .00 C .32 .19 .10 .52 .09 .74 C .07 .94 .35 1.05 .60 .98 C .92 .89 .99 .71 .83 .62 C .57 .49 .23 .69 .26 .80', 3),
        ('M .63 .01 Q .22 .32 .06 .70 Q -.02 .90 .26 .98 C .61 1.11 1.06 .86 .89 .63 Q .75 .44 .46 .50', 2),
        ('M .74 .00 Q .35 .24 .08 .64 C -.08 .93 .21 1.07 .59 .97 C .93 .89 1.07 .66 .89 .56 C .64 .39 .22 .63 .16 .78', 3),
        ('M .85 .03 C .53 -.09 .14 .21 .08 .57 C -.01 .93 .30 1.07 .61 .98 C .95 .89 1.02 .65 .86 .52 C .66 .36 .30 .47 .16 .64', 2),
    ],
    7: [
        ('M .01 .05 Q .49 -.01 .98 .02 Q .59 .44 .32 1', 7),
        ('M .01 .13 Q .44 -.02 .97 .01 L .31 1', 5),
        ('M .02 .01 Q .43 .05 .95 .01 C .74 .28 .46 .67 .37 1', 4),
        ('M .05 .23 L .02 .06 Q .51 -.02 .98 .01 L .28 1', 3),
        ('M .00 .09 Q .41 -.02 .95 .02 L .34 1 | M .23 .48 L .88 .46', 4),
        ('M .03 .04 Q .51 -.02 .96 .03 Q .56 .56 .41 1 | M .43 .52 L .78 .49', 2),
        ('M .01 .12 Q .04 -.01 .25 .02 Q .58 .08 .93 .02 Q .64 .41 .37 1', 2),
    ],
    8: [
        ('M .53 .48 C .25 .38 -.02 .25 .12 .10 C .25 -.06 .64 -.04 .81 .08 C 1 .25 .70 .40 .45 .54 C .11 .70 .01 .83 .18 .95 C .40 1.10 .88 .99 .90 .82 C .94 .68 .71 .57 .53 .48', 6),
        ('M .58 .03 C .32 -.06 .06 .07 .11 .23 C .15 .39 .50 .48 .72 .62 C 1.04 .82 .90 .99 .59 1 C .30 1.03 .01 .92 .09 .76 C .14 .61 .46 .46 .70 .32 C .94 .15 .84 -.01 .57 .02', 4),
        ('M .65 .02 C .36 -.04 .07 .08 .09 .24 C .13 .41 .60 .52 .78 .66 C .98 .82 .90 .99 .60 1 C .19 1.04 -.02 .87 .12 .69 Q .25 .54 .56 .44 C .92 .32 1 .08 .72 .04', 3),
        ('M .50 .45 C .12 .43 .02 .22 .18 .09 C .42 -.08 .86 -.01 .87 .17 Q .87 .37 .55 .46 C .83 .50 1.06 .69 .88 .87 C .71 1.05 .29 1.07 .12 .89 C -.02 .73 .16 .52 .50 .45', 3),
        ('M .49 .48 C .77 .37 .92 .20 .78 .07 C .59 -.08 .22 -.02 .15 .15 C .06 .30 .30 .43 .52 .49 C .86 .57 .99 .76 .82 .92 C .63 1.06 .27 1.03 .12 .87 C -.03 .71 .19 .56 .49 .48', 3),
        ('M .69 .10 C .53 -.06 .16 -.01 .10 .17 C .01 .38 .49 .52 .72 .68 C 1.02 .87 .77 1.06 .43 .99 C .09 .93 .04 .78 .25 .60 C .51 .40 .97 .29 .86 .10 Q .79 -.02 .56 .02', 2),
        ('M .50 .46 C .21 .43 .07 .30 .12 .16 C .20 -.05 .71 -.05 .83 .12 C .97 .30 .74 .45 .51 .47 | M .51 .45 C .17 .45 .01 .65 .10 .83 C .23 1.08 .69 1.05 .86 .88 C 1.02 .68 .86 .49 .51 .45', 1.5),
    ],
    9: [
        ('M .85 .13 C .77 -.05 .36 -.02 .16 .13 C -.07 .30 .03 .52 .33 .54 C .58 .55 .80 .39 .87 .17 Q .85 .53 .72 1', 5),
        ('M .79 .09 C .62 -.05 .26 .00 .10 .18 C -.06 .36 .10 .52 .35 .51 Q .66 .51 .86 .22 L .88 .03 L .73 1', 4),
        ('M .88 .05 C .64 -.05 .29 .02 .12 .18 C -.08 .36 .05 .52 .30 .53 Q .64 .53 .92 .17 L .60 1', 4),
        ('M .83 .17 C .77 -.01 .43 -.05 .22 .10 C -.02 .25 -.01 .44 .20 .51 C .42 .61 .73 .45 .84 .27 C .95 .51 .82 .87 .57 1', 4),
        ('M .89 .08 C .71 -.07 .31 -.01 .13 .14 C -.04 .30 .01 .50 .29 .52 Q .62 .56 .85 .27 Q .86 .58 .80 .83 Q .78 .97 .90 1', 3),
        ('M .89 .05 Q .60 -.07 .27 .07 Q .01 .22 .03 .41 Q .07 .54 .31 .51 Q .63 .48 .85 .15 L .65 .82 Q .62 .95 .49 1', 3),
        ('M .75 .02 C .46 -.07 .11 .10 .07 .26 C -.02 .46 .20 .57 .44 .47 Q .68 .39 .83 .11 Q .87 .58 .69 1', 3),
        ('M .81 .13 C .72 -.05 .35 -.02 .16 .14 C -.02 .30 .05 .51 .32 .52 C .57 .55 .84 .37 .87 .20 C .99 .46 .87 .81 .59 .95 Q .37 1.06 .15 .93', 1.5),
    ],
}


def decode(s):
    paths=[]
    for part in s.split('|'):
        tokens=iter(part.split())
        def point():
            return np.array([float(next(tokens)),float(next(tokens))])
        seg=[]
        t=np.linspace(0,1,31)[1:,None]; u=1-t
        for cmd in tokens:
            if cmd=='M':
                p=point(); seg.append(p[None,:]); continue
            if cmd=='L':
                q=point(); v=u*p+t*q
            elif cmd=='Q':
                a=point(); q=point(); v=u*u*p+2*u*t*a+t*t*q
            elif cmd=='C':
                a=point(); b=point(); q=point(); v=u**3*p+3*u*u*t*a+3*u*t*t*b+t**3*q
            else: raise ValueError(cmd)
            seg.append(v); p=q
        paths.append(np.concatenate(seg))
    return paths

# Reusable analytic filter and descriptor definitions. These constants do
# not depend on the contents of any drawing, or on any group of drawings.
YY,XX=np.mgrid[-3:4,-3:4]
GAUSS=np.exp(-(XX**2+YY**2)/(2*1.25**2)); GAUSS/=GAUSS.sum()
GX=XX*np.exp(-(XX**2+YY**2)/(2*1.20**2)); GX/=np.maximum(GX,0).sum()
GY=GX.T
ANGLES=np.arange(8)*np.pi/4
EDGES=np.array([np.cos(t)*GX+np.sin(t)*GY for t in ANGLES])
# 2x2 stride for the ink descriptor and for the sampling of edge responses.
# The latter is followed by non-overlapping 2x2 cell averages.

def sample2(a):
    return (a[...,::2,::2]+a[...,1::2,::2]+a[...,::2,1::2]+a[...,1::2,1::2])*.25


# For each oriented line, gather the two rays starting at the site. The
# direction table and all indices are computed from grid geometry alone.
SIDE=14

def ray_indices():
    y,x=np.mgrid[:SIDE,:SIDE]; y=y.reshape(-1,1); x=x.reshape(-1,1)
    r=np.arange(SIDE)[None,:]
    rows=[]
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        rays=[]
        for sign in [-1,1]:
            yy=y+dy*r*sign; xx=x+dx*r*sign
            # Repeat the ray origin after exiting the frame (neutral for max).
            idx=np.where((yy>=0)&(yy<SIDE)&(xx>=0)&(xx<SIDE),yy*SIDE+xx,y*SIDE+x)
            rays.append(idx)
        rows.append(np.stack(rays,axis=1))
    return np.stack(rows)

RAYS=ray_indices()

def gap_descriptor(a):
    u=sample2(a)
    both=u.ravel()[RAYS].max(axis=-1).min(axis=-1).reshape(4,SIDE,SIDE)
    return sample2(np.concatenate((both-u,1.20*u[None]),axis=0))


def projections(a):
    # Sample after convolution at y,x = 0,2,...,26 (padding 3).
    # For the two first derivatives the filters are separable. Gaussian
    # ink is kept at full resolution before 2x2 averaging.
    def sep(row, col, stride):
        pad=np.pad(a,((0,0),(3,3)))
        v=sum(e*pad[:,j:j+28:stride] for j,e in enumerate(row))
        pad=np.pad(v,((3,3),(0,0)))
        return sum(e*pad[j:j+28:stride] for j,e in enumerate(col))
    g=np.exp(-(np.arange(-3,4)**2)/(2*1.25**2)); g/=g.sum()
    h=np.exp(-(np.arange(-3,4)**2)/(2*1.20**2)); d=np.arange(-3,4)*h
    dx=sep(d,h,2)/(h.sum()*np.maximum(d,0).sum()); dy=sep(h,d,2)/(h.sum()*np.maximum(d,0).sum())
    act=np.maximum(0,np.cos(ANGLES)[:,None,None]*dx+np.sin(ANGLES)[:,None,None]*dy)
    # Include the last location in a symmetric cell by using 2x2 cells
    # on the 14x14 lattice, yielding exactly 7x7.
    edges=sample2(act)
    return sample2(sep(g,g,1)),edges


def unit(v):
    v=v.ravel(); v=v-v.mean()
    return v/(np.sqrt(v@v)+1.e-12)


def make_image(k,paths,rng):
    # Sample a common set of deformation priors. All choices below are
    # fixed a priori. Compute geometry/size/centroid for this drawing only.
    def bell(): return np.clip(rng.normal(),-2.3,2.3)
    ratio=(.70,.61,.79,.76,.77,.77,.69,.81,.65,.73)[k]*np.exp(.245*bell())
    slant=-.14+.255*bell()
    # One is primarily a stem; its inclination creates most of its width.
    # Expressing slope relative to height makes nearly vertical ones possible.
    if k==1: slant=-.11+.205*bell()
    cross=.13*bell()
    # A spare degree of freedom near each end and at the waist. Same
    # smooth warp on all strokes so that most intersections stay connected.
    xwaist=.085*bell(); xend=.075*bell(); ywaist=.098*bell()
    taper=.23*bell(); turn=.14*bell(); bulge=.11*bell()
    stepheight=.09*bell(); twist=.085*bell()
    f=[]
    for j,path in enumerate(paths):
        x=path[:,0]-.5; y=path[:,1]-.5
        sx=(x*(1+2*taper*y+bulge*np.cos(np.pi*y*2))+xwaist*np.cos(np.pi*y)+xend*np.cos(2*np.pi*y)+twist*np.sin(3.5*y+1.2*x))
        sy=y+ywaist*np.cos(np.pi*y)+stepheight*np.cos(2*np.pi*y)+cross*x+turn*x*y
        if j: sx+=.044*bell(); sy+=.037*bell()
        f.append(np.stack((ratio*sx+slant*sy,sy),axis=1))
    p=np.concatenate(f); span=np.ptp(p,axis=0).max()
    radius=np.clip(1.37+.36*bell(),.62,2.25)
    sz=np.clip(20+.38*bell(),18.8,21.2)
    scale=(sz-2*radius)/max(span,.01)
    # Rasterize a polyline as a chain of tightly overlapping, anti-aliased
    # round pen dabs; all calculations here involve only one template.
    phase=rng.uniform(0,2*np.pi); ripple=.14*bell(); slope=.13*bell()
    chunks=[]; radii=[]
    for stroke in f:
        xy=stroke*scale
        arc=np.r_[0.,np.cumsum(np.sqrt((np.diff(xy,axis=0)**2).sum(1)))]
        u=np.linspace(0,arc[-1],max(2,int(arc[-1]/.24)+1))
        pt=np.stack([np.interp(u,arc,col) for col in xy.T],axis=1)
        s=u/max(1.e-9,arc[-1]); r=radius*(1+ripple*np.sin(2*np.pi*s+phase)+slope*(2*s-1))
        chunks.append(pt); radii.append(r)
    pts=np.concatenate(chunks); r=np.concatenate(radii)
    center=(pts.max(0)+pts.min(0))/2
    # Local square footprint around each dab; clipping out of frame can only
    # affect the outermost border. Coordinates use pixel centers at integers.
    offsets=np.arange(-4,5); jj,ii=np.meshgrid(offsets,offsets,indexing='ij'); jj=jj.ravel(); ii=ii.ravel()
    def draw(c):
        v=pts+c
        xx=v[:,0]; yy=v[:,1]
        ix=np.rint(xx).astype(int)[:,None]+ii; iy=np.rint(yy).astype(int)[:,None]+jj
        dist=np.sqrt((ix-xx[:,None])**2+(iy-yy[:,None])**2)
        ink=np.clip(.5+(r[:,None]-dist)/1.12,0.,1.)
        arr=np.zeros(784)
        np.maximum.at(arr,(28*np.clip(iy,0,27)+np.clip(ix,0,27)).ravel(),ink.ravel())
        return arr.reshape(28,28)
    c=13.5-center; a=draw(c); den=a.sum()
    loc=np.arange(28); cm=np.array([a.sum(0)@loc,a.sum(1)@loc])/den
    target=13.5+rng.uniform(-.53,.53,2)
    return draw(c+target-cm)


def build(out):
    out.mkdir(parents=True,exist_ok=True)
    layers=[]; w={}
    def layer(name,op,inp,**settings):
        layers.append(dict(name=name,op=op,inputs=[inp] if isinstance(inp,str) else inp,**settings)); return name
    def conv(name,inp,kernels,stride=1,pad=0,b=None):
        W=np.array(kernels,dtype=np.float32)
        w[name+'.W']=W; w[name+'.b']=np.zeros(len(W),np.float32) if b is None else np.array(b,np.float32)
        return layer(name,'conv',inp,out=len(W),kernel=list(W.shape[-2:]),stride=stride,padding=pad)
    def dense(name,inp,W,b=None):
        w[name+'.W']=np.asarray(W,dtype=np.float32)
        w[name+'.b']=np.zeros(len(W),np.float32) if b is None else np.asarray(b,dtype=np.float32)
        return layer(name,'dense',inp,out=len(W))
    def pool(name,inp,size,stride,op='maxpool'):
        return layer(name,op,inp,size=size,stride=stride)
    def reshape(name,inp,shape): return layer(name,'reshape',inp,shape=shape)
    def add(name,*inp): return layer(name,'add',list(inp))
    def normflat(name,inp,n): return layer(name,'norm',reshape(name+'_vector',inp,[1,1,n]))

    blur=conv('gaussian_ink','image',GAUSS[None,None],pad=3)
    ink=pool('ink_cells',blur,[2,2],[2,2],'avgpool')
    ink=layer('ink_contrast','norm',ink)
    edge=conv('compass_derivatives','image',EDGES[:,None],2,3)
    edge=layer('positive_derivatives','relu',edge)
    edge=pool('contour_cells',edge,[2,2],[2,2],'avgpool')
    edge=normflat('contour_contrast',edge,8*49)
    # Four maps showing clear space bracketed by ink along a line. Gather
    # each ray, max over it, then min the opposite pair by negating twice.
    # Append an ink map as a fifth channel for context, with fixed gain.
    coarse=pool('ray_lattice','image',[2,2],[2,2],'avgpool')
    idx=RAYS.ravel(); raymat=np.zeros((len(idx),SIDE*SIDE),np.float32)
    raymat[np.arange(len(idx)),idx]=1
    q=dense('ray_gather',coarse,raymat); q=reshape('ray_strips',q,[4,2*SIDE*SIDE,SIDE]); q=pool('ray_max',q,[1,SIDE],[1,SIDE]); q=reshape('opposite_rays',q,[4,SIDE*SIDE,2])
    flip=-np.eye(4)[:,:,None,None]
    q=conv('neg_rays',q,flip); q=pool('min_via_negation',q,[1,2],[1,2]); q=reshape('negative_brackets',q,[4,SIDE,SIDE]); q=conv('bracket_height',q,np.vstack((flip,np.zeros((1,4,1,1)))))
    subtract=conv('remove_local_ink',coarse,np.array([-1]*4+[1.20])[:,None,None,None]); q=add('free_intervals',q,subtract)
    q=pool('interval_cells',q,[2,2],[2,2],'avgpool'); gap=normflat('interval_contrast',q,5*49)
    rng=np.random.default_rng(18764321)
    # Group size controls only the reducing tree. Eight independent subbanks
    # give a small vote to support beyond the single winning template.
    per=3072; groups=8; pergroup=per//groups
    ni,ne,ng=196,392,245
    wi=np.empty((10*per,ni),np.float32); we=np.empty((10*per,ne),np.float32); wg=np.empty((10*per,ng),np.float32)
    for k in range(10):
        styles=[decode(s) for s,_ in PATHS[k]]
        # These probabilities apply to *strings* before rendering; they are
        # not derived from images, comparisons, or counts of generated shapes.
        prob=np.array([p for _,p in PATHS[k]]); prob=prob/prob.sum()
        for i in range(per):
            a=make_image(k,styles[rng.choice(len(styles),p=prob)],rng)
            v,e=projections(a)
            wi[k*per+i]=unit(v)*(.33/np.sqrt(ni))
            we[k*per+i]=unit(e)*(.44/np.sqrt(ne))
            wg[k*per+i]=unit(gap_descriptor(a))*(.23/np.sqrt(ng))
    a=dense('match_ink',ink,wi); b=dense('match_contour',edge,we); c=dense('match_intervals',gap,wg)
    m=add('joint_matches',a,b,c)
    m=reshape('template_groups',m,[10,groups,pergroup]); m=pool('group_best',m,[1,pergroup],[1,pergroup])
    mx=pool('closest',m,[groups,1],[groups,1]); av=pool('support',m,[groups,1],[groups,1],'avgpool')
    mx=reshape('closest_vector',mx,[10]); av=reshape('support_vector',av,[10]); av=dense('support_weight',av,.34*np.eye(10))
    shape_score=add('shape_scores',mx,av)

    # A geometric second opinion: enclosed background cannot reach the
    # border through free space. Use grayscale morphological reconstruction
    # (min is a negated max) on a coarse lattice, propagating through gaps
    # by the eight neighbors of each cell. This does not refer to the templates.
    # First make an ink barrier with a small closing effect on thin strokes.
    barrier=pool('barriers','image',[2,2],[2,2]); G=14
    border=np.ones((G,G)); border[1:-1,1:-1]=0
    first=dense('exterior_seed',barrier,np.zeros((G*G,G*G)),border.ravel()); first=reshape('initial_exterior',first,[1,G,G])
    # Take the max of the neighbors including center, then cap at 1-barrier.
    # Pad each map with zeros using an identity 3x3 filter with padding 2.
    ident=np.zeros((1,1,3,3)); ident[0,0,1,1]=1
    z=first
    # z represents the water level above zero. With B the ink opacity,
    # min(z,1-B) = z - relu(z+B-1). At the end, the free-space capacity
    # minus the final water level measures the trapped background.
    free=conv('free_space',barrier,[[[[-1.]]]],b=[1]); negfree=conv('negative_free',barrier,[[[[1.]]]],b=[-1])
    for i in range(22):
        p=conv('ex_pad%d'%i,z,ident,pad=2)
        # A 3x3 step uses 8-connectivity of the background. The contour
        # barrier was max-pooled above to avoid leaking diagonally through it.
        u=pool('reach%d'%i,p,[3,3],[1,1]); t=add('excess%d'%i,u,negfree)
        t=layer('overflow%d'%i,'relu',t); t=conv('subtract%d'%i,t,[[[[-1.]]]])
        z=add('exterior%d'%i,u,t)
    n=conv('negative_water',z,[[[[-1.]]]]); holes=add('trapped_space',free,n)
    # A large ring has many interior pixels, a small one far fewer. Measure
    # the clearest trapped-background pixel as evidence, rather than area.
    # Four spatial readings: anywhere, upper, lower, and upper PLUS lower
    # with an actual bar of ink between them. Requiring a bar helps avoid
    # counting a single tall counter in a zero as two holes.
    # Use mask weights to read each coarse pixel once. Mask biases are
    # constants; keep margins wide because digits were centered by mass.
    def regionpeak(name,src,mask):
        h,wid=mask.shape
        # Gather only the requested sites; no weights depend on ink.
        ids=np.flatnonzero(mask); mat=np.zeros((len(ids),h*wid)); mat[np.arange(len(ids)),ids]=1
        q=dense(name+'_sites',src,mat); q=reshape(name+'_strip',q,[1,1,len(ids)])
        q=pool(name+'_peak',q,[1,len(ids)],[1,len(ids)]); return reshape(name,q,[1])
    y,x=np.mgrid[:G,:G]; central=(x>=2)&(x<=11)
    anyhole=regionpeak('a_hole',holes,central & (y>=1)&(y<=12))
    high=regionpeak('upper_hole',holes,central & (y>=1)&(y<=6))
    low=regionpeak('lower_hole',holes,central & (y>=7)&(y<=12))
    # Two diagonally allowed holes x0,x1; enforce the vertical order
    # hole -> ink band -> hole. At the intermediate height require ink
    # across the straight interpolation of their centers. Min is realized
    # by negation + maxpool. It is weak evidence alone for concave loops,
    # hence most of the decision remains the template matcher.
    pairs=[]
    for ya in range(2,7):
        for yb in range(7,12):
            if not 3<=yb-ya<=8: continue
            for xa in range(3,11):
                for xb in range(max(2,xa-3),min(12,xa+4)):
                    pairs.append((ya,xa,yb,xb))
    # Fixed one-hot gathers put candidate pairs in a 1-D strip. A third
    # gather reads the ink barrier at each pair's midpoint.
    hflat=reshape('holes_flat',holes,[G*G]); bflat=reshape('ink_flat',barrier,[G*G])
    def gather(name,src,idx):
        mat=np.zeros((len(idx),G*G),np.float32); mat[np.arange(len(idx)),idx]=-1
        return reshape(name+'_array',dense(name,src,mat),[1,1,len(idx)])
    ya,xa,yb,xb=np.array(pairs).T
    ho1=gather('neg_upper',hflat,ya*G+xa); ho2=gather('neg_lower',hflat,yb*G+xb)
    # Construct min from a-b using a-ReLU(a-b); construct max by symmetry.
    def max2(name,a,b):
        nb=conv(name+'_neg_b',b,[[[[-1.]]]]); diff=add(name+'_diff',a,nb); pos=layer(name+'_pos','relu',diff)
        return add(name,b,pos)
    ym=(ya+yb)//2; xm=(xa+xb+1)//2
    # A half-cell minimum on the middle ink is preferred to maximum on an
    # adjacent sidewall; use the pair midpoint directly.
    bar=gather('neg_separator',bflat,ym*G+xm)
    two_neg=max2('negative_pair',max2('negative_holes',ho1,ho2),bar)
    two=conv('pair_presence',two_neg,[[[[-1.]]]]); two=pool('best_pair',two,[1,len(pairs)],[1,len(pairs)]); two=reshape('divided_hole',two,[1])
    # Hand-specified, small additive evidence in similarity-score units.
    # A closed hole is positive evidence for 0,6,8,9, not a hard condition.
    # 2 can have a cursive loop and 4 a triangular counter, so avoid vetoes.
    def vote(name,src,coeff): return dense(name,src,np.array(coeff)[:,None])
    topo=vote('enclosure_vote',anyhole,[.045,-.028,-.018,-.030,-.003,-.022,.016,-.030,.026,.016])
    tv=vote('top_counter_vote',high,[0,0,0,0,0,0,-.014,0,0,.018]); bv=vote('bottom_counter_vote',low,[0,0,0,0,0,0,.021,0,0,-.014]); dv=vote('two_counter_vote',two,[-.048,-.005,-.008,-.010,-.010,-.010,-.026,-.010,.110,-.026])
    add('scores',shape_score,topo,tv,bv,dv)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
    np.savez(out/'weights.npz',**w)

if __name__=='__main__':
    build(Path(sys.argv[1]))
