"""Build a digit shape matcher. Only Python, numpy, and the specifications below are needed."""
import json
import sys
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# A stroke recipe is a start point followed by line (L), quadratic (Q), or cubic
# (C) Bezier segments. M lifts the pen. Coordinates are in a nominal 10x10 em.
# These are writing instructions, not outlines sampled from character images.
# Integer multiplicities describe common versus more unusual ways of writing.
GLYPHS = [
  [ # zero: elliptical, upright-sided, pointed, and retraced at the join
    (5, 'M 5 0 C 8 0 10 1.8 10 5 C 10 8 8.2 10 5 10 C 2 10 0 8.3 0 5 C 0 2.2 2 0 5 0'),
    (3, 'M 5.4 0 C 9 -.4 10 1.5 10 4.8 C 10 7.6 8.3 10 4.4 10 C 1.4 10 0 8.8 0 6 C 0 2.5 3.2 -.1 6 0'),
    (2, 'M 5.8 0 C 9.1 0 9.9 1.5 9.7 5.1 C 9.6 8.5 8.5 10 5 10 C 1.8 10 .5 9.3 .4 6.7 C .1 3.7 1.5 .8 5.8 0'),
    (2, 'M 7.5 .8 Q 5 -1.2 3.1 .8 C .2 2.8 -.7 7.1 1 9.1 C 2.5 10.8 5.8 10 7.7 8 C 10 5.8 10.7 2.2 7.5 .8'),
    (1, 'M 5 .8 C 1 .6 -.6 5.2 .6 8.3 C 1.2 10.5 4.7 10.5 7 8.7 C 9.9 6.3 10.7 2.2 8 .4 Q 5.4 -.8 3 1.9'),
  ],
  [ # one: plain downstroke, small lead-in, a full flag, or a foot
    (9, 'M 5 0 L 5 10'),
    (4, 'M 5.1 0 Q 4.4 5.8 5.3 10'),
    (3, 'M 5 0 Q 5.9 4.9 4.9 10'),
    (4, 'M 4 1.3 L 5.3 0 Q 5 5 5.1 10'),
    (3, 'M 1.4 2.7 L 5.9 0 L 5.3 10'),
    (2, 'M .8 3.9 Q 3.7 2.8 6 .2 L 5.6 10'),
    (1, 'M 2 1.9 L 5.5 0 L 5.1 10 M .6 10 Q 5 9.4 9.3 9.9'),
    (1, 'M 4.8 0 L 4.9 9.9 M 2.4 9.9 L 7.9 9.9'),
  ],
  [ # two: curved cap, diagonal, baseline; sometimes a loop at the foot
    (4, 'M .2 2.1 C 1.4 -.1 6.6 -1.2 8.6 1.4 C 10.2 3.6 6.8 5.5 1 9.5 Q 4.5 9.3 9.9 10'),
    (3, 'M 1.1 3.1 C .7 .6 4.9 -.9 7.5 .7 C 11.2 3.1 4.4 7.1 .5 9.4 Q 3.6 10 9.8 9.3'),
    (3, 'M .5 2.3 Q 4 -1.6 7 .7 C 9.4 2.6 6 5.5 1.2 8.7 Q -1.1 10.6 3.4 9.7 L 10 8.9'),
    (2, 'M 1.7 3.3 C .3 1.8 2.5 -.1 5.8 .1 C 10.8 .5 7.6 4.7 3.6 7.6 Q .2 10.3 1.4 10 Q 4.4 9 9.9 9.6'),
    (3, 'M 1.5 1.9 C 4.5 -1.2 8.2 -.4 7.7 2.4 C 7.3 4.6 4 8.8 1.4 9.7 C -1 10.6 -.6 7.8 1.3 7.8 C 4.2 7.3 6.6 10.9 10 9.3'),
    (2, 'M 1.4 2.9 C 1.1 -.2 6.7 -1.1 7.4 1.8 C 8.1 4.3 5.1 8.1 2.4 9.5 C -.6 11.1 -.7 7.1 1.7 7.4 Q 3.8 7.5 6.3 9 Q 8.4 10.1 10 8.9'),
    (1, 'M .3 2 L 4.6 .1 Q 7.8 -1 8.5 1.4 Q 9.3 3.9 1.1 9.6 L 9.8 9.5'),
    (1, 'M 2.3 3.4 C 1.5 .6 3.6 -.8 5.8 .2 C 9.8 2.3 5.2 7 .6 9.5 Q 3.3 9.7 5.7 8.9 Q 8.1 8.4 10 9.5'),
  ],
  [ # three: round or angular upper lobe, often a little loop at the waist
    (4, 'M .7 1.4 C 3.7 -.8 9.8 -.2 9 2.2 Q 8.5 4 3.9 4.8 C 7.8 3.9 10.6 5.6 9.2 8 Q 7.3 11.3 .8 9.1'),
    (3, 'M .7 1.7 Q 5.3 -1.6 8.7 .9 Q 11.2 3.3 4.2 5 C 8.7 4.4 10.9 7.1 8.1 9.1 Q 4.5 11.4 .4 8.5'),
    (3, 'M .4 .8 L 8.8 .1 Q 9.9 .4 4.1 4.7 C 10 3.8 11 7.6 7.6 9.4 Q 4.8 10.9 .9 9'),
    (2, 'M 1.4 1.1 C 4 -.5 9.6 -.5 8.8 1.8 C 8.1 3.7 4.9 5.7 3.8 5 Q 2.9 3.9 5.6 4.5 C 11.1 5.3 10.6 8.5 7.8 9.5 Q 4.6 10.9 .2 9.1'),
    (2, 'M .3 1.8 C 3 -.4 8.5 -.8 8.7 1.4 C 8.9 3.1 5.8 4.6 4.8 4.9 C 10.8 4.5 10.2 8.5 7.8 9.5 C 5.7 10.9 1.5 10.6 .4 8.1'),
    (1, 'M .6 .4 Q 4.2 -.3 8.8 .5 L 4.3 4.6 Q 8.7 3.5 9.3 6.3 Q 10.1 9.7 2.3 10'),
    (2, 'M .5 1.2 Q 4.1 -.8 7.6 .3 Q 11.8 2.6 5.4 4.9 Q 9.7 5.1 9.1 7.6 Q 8.3 10.7 1.6 10'),
  ],
  [ # four: open chair, joined triangular top, or a flowing round left stroke
    (5, 'M 2.4 0 L .5 5.7 Q 3.2 6.1 9.4 5.4 M 7.6 .7 L 7.3 10'),
    (4, 'M 2 0 Q -.6 5.3 .6 5.9 Q 3.7 6.8 8 4.8 M 8.1 .3 Q 7.4 5.2 7.8 10'),
    (3, 'M 2.8 .3 L .5 5.9 L 7.9 5.7 M 8 2 L 7.7 10'),
    (3, 'M 7.4 .1 L .1 6.1 L 10 5.9 M 7.4 .1 L 7.6 10'),
    (2, 'M 6.2 0 L .1 5.9 Q 4.1 6.4 9.9 5.6 M 7.8 .4 L 7.1 10'),
    (3, 'M 1.3 0 C .9 2.5 -.7 5.4 1.1 6 C 3 6.7 6.9 4.7 8.2 3 M 8.3 .6 L 7.7 10'),
    (2, 'M 1.8 .2 Q .2 3.4 .8 5.4 Q 2.1 7.4 7.5 4.6 L 7.9 1.6 L 7.3 10'),
    (1, 'M 3.7 0 L .2 6.1 L 9.9 5.4 M 7.6 3.1 L 7.4 10'),
  ],
  [ # five: separate top bar, shorthand S-like forms, and a square shoulder
    (5, 'M 9.3 .1 Q 5.7 .6 2.1 .3 L 1.1 4.8 C 4.2 3.2 9 3.3 9.3 6.5 C 9.8 9.5 4.2 11.6 .5 8.7'),
    (4, 'M 2 .4 L 1 4.2 Q 5.7 3.3 8.1 5.2 C 11 7.8 7.1 10.3 3.6 10 Q 1.7 10 .4 9 M 1.3 .7 Q 6.4 .9 9.8 -.1'),
    (3, 'M 9.9 .2 L 2.7 .4 Q 1.8 1.8 1.4 3.5 Q 7.4 4.1 8.6 6.2 C 10.1 9 4.5 11.3 .3 9'),
    (2, 'M 8.9 .2 C 5.5 -.3 1.1 .6 1.3 2.7 C 1.6 4.4 7.5 4.4 8.5 6.7 C 9.7 9.6 3.4 11.3 .3 8.9'),
    (2, 'M 9.6 .2 Q 5.6 1.1 2.5 .5 L 1.4 4.6 C 4.7 3.6 8.4 4.3 8.7 6.7 Q 9 9.8 1.9 10'),
    (2, 'M 2.2 .4 Q .3 3.2 1.1 4.6 C 6.7 2.9 10.3 5.4 8.3 8.1 Q 5.8 11.1 .1 9.9 M 1.8 .7 L 9.7 0'),
    (1, 'M 9.6 .2 L 3 .1 L .9 3.6 C 3.7 4.3 8.7 5.1 8.7 7.5 Q 8.2 10.3 1.8 10'),
  ],
  [ # six: a long lead-in followed by a counterclockwise bottom bowl
    (5, 'M 7.4 .1 C 4.4 1 1.2 3.8 .9 7.1 C .5 10.1 4.7 10.8 7.5 9.2 C 10.2 7.3 8.1 4.6 5.4 5 Q 2.6 5.2 1.1 7.7'),
    (4, 'M 6.5 .1 C 3.3 1.5 .1 5.2 1.2 8.2 C 2.3 11.1 7.2 10.4 8.7 8.3 C 10.6 5.3 6.9 4.4 4.4 5.5 Q 2.3 6.5 2 8.4'),
    (3, 'M 6.4 .1 C 3.3 1.7 -.6 6.5 1.8 8.8 C 4.2 11.1 8.7 10 9.1 7.4 C 9.6 4 4.8 3.8 2 6.8'),
    (2, 'M 7.4 -.1 Q 1.4 3.8 1.1 7 C .7 10.7 6.4 10.7 8.7 8.7 Q 11.3 5.8 7.3 5.3 Q 4 5.5 1.6 8'),
    (2, 'M 8.4 .7 Q 6.7 -.9 4.5 1 C 1.7 3.2 -.2 7 2 9.1 C 4.6 11.7 9.5 9.3 9.1 6.8 Q 8.5 3 3 5.6 Q 2 6.3 1.5 7.8'),
    (2, 'M 5.8 .1 C 3.4 2.4 -.3 6.2 2 8.9 Q 4.5 11 8 9 Q 10.7 7 7.1 5 Q 4.5 4 3.1 5.9'),
    (1, 'M 7.2 .1 C 3.2 .5 -.2 5.9 1.7 8.8 C 3.1 10.7 7.9 10.7 8.8 8.3 Q 9.7 6 5.4 5.4'),
  ],
  [ # seven: cap then diagonal, with optional cap hook, crossbar or a foot
    (6, 'M .2 .6 Q 4.6 -.2 9 .3 L 3.8 10'),
    (4, 'M .3 .2 Q 3.4 1.2 8.8 .3 Q 6.4 4.1 4 10'),
    (3, 'M .4 2 L .2 .4 Q 4.7 -.2 8.8 .3 L 4.1 10'),
    (3, 'M .4 1 Q 3 -.8 5.5 .2 Q 7.6 .7 9 .4 L 3.9 10'),
    (3, 'M .2 .5 L 9.1 .1 Q 6.4 5.1 4 10 M 2.9 5.1 L 8 4.8'),
    (2, 'M .5 2.1 L .4 .3 L 9 .5 L 3.5 10 M 3.3 5 L 8.5 5.1'),
    (1, 'M .3 .5 Q 4 .9 8.8 .1 Q 3.3 8.6 4 10'),
    (1, 'M .6 1.2 L 8.1 .1 Q 8.8 .5 7.7 2.4 L 3.7 9.9 L 5 9.7'),
  ],
  [ # eight: two bowls, continuous crossed ribbon, small/open joins
    (4, 'M 5.2 4.6 C 1.6 3.8 .2 1.7 2.8 .5 C 5.8 -.9 9.7 .4 8.6 2.7 C 7.4 4.9 1.7 5.7 1.1 8 C .1 10.8 6.5 10.6 8.5 9 C 10.8 7.1 8.1 5.5 5.2 4.6'),
    (3, 'M 4.2 4.8 C 1.1 3.3 1.1 1.3 3.5 .4 C 6.9 -.9 10.2 .9 8.6 3 Q 7.3 4.3 4.8 5.1 C 2.1 6.1 -.4 8.2 2.3 9.6 C 5.4 11.3 9.9 9.1 8.6 7 Q 7.7 5.7 4.2 4.8'),
    (3, 'M 5 4.8 C .9 4.6 -.1 .6 4 .1 C 8.4 -.8 10.4 2.8 7.6 4.3 Q 6.3 5 5 4.8 C .5 5.3 -.3 9.3 3.5 10 C 8 10.9 10.8 7.4 8.3 5.8 Q 7 4.9 5 4.8'),
    (2, 'M 6.2 .3 C 2 -.5 .5 1.8 2.7 3.5 C 4.8 5.1 9.1 5.8 9.1 7.8 C 9.2 10.7 3.3 10.9 1.6 9.2 C -.6 6.6 5.9 4.7 7.8 3.1 Q 10.2 .5 6.2 .3'),
    (2, 'M 6 .4 C 3.9 -.4 1.2 .8 1.5 2.3 C 1.8 4.4 7.8 5.2 8.6 7.6 C 9.9 10.6 4.5 11.1 2.1 9.3 C -.1 7.4 3.3 5.3 5.4 4 Q 9.9 1.1 7.7 .4'),
    (2, 'M 4.7 4.5 C 1.5 3.6 1.2 1.3 3.6 .4 C 7.1 -1.1 10.3 1.1 8.1 3.4 Q 6.3 5.1 4.7 4.5 C .8 5.6 -.8 8.6 2.3 9.7 Q 6.1 11.2 8.3 9 Q 10.8 6.7 4.7 4.5'),
    (1, 'M 8.1 1 Q 6.6 -.7 3.8 .4 C -.3 2.2 3.7 4.1 5.4 5.3 C 9 7.5 9.3 10 5.6 10 C 1.4 10.5 .3 8.5 2.3 6.4 C 3.9 4.8 8.7 3.8 8.4 1.3'),
  ],
  [ # nine: circle and stem, or continuous loop with long descending stroke
    (5, 'M 8.6 1.2 C 6.3 -.9 2.1 -.1 1.2 2.1 C -.2 5 3.1 6.1 5.6 4.9 Q 7.8 3.9 8.7 1.2 L 7.4 10'),
    (4, 'M 8.2 1.4 C 7.9 -.3 4.5 -.5 2.4 .7 C -.7 2.6 .7 5.5 3.2 5.3 C 5.8 5.2 8 2.6 8.8 .6 Q 7.4 5.9 7.7 10'),
    (3, 'M 8.9 2.3 C 8.3 .1 4.7 -.7 2.6 .5 C -.2 2.1 .3 5.1 2.9 5.3 Q 6.9 5.9 8.9 2.3 Q 9 6.2 5.5 10'),
    (3, 'M 8.2 1.2 Q 6.2 -.8 3.4 .3 C .5 1.4 -.8 4.7 1.7 5.3 C 4.5 6 7.8 3.4 8.9 1 L 6.3 10'),
    (3, 'M 8.5 1.5 C 6.8 -.5 2.7 -.7 1.2 1.8 C -.7 4.9 3.4 6.6 6.6 4.6 Q 8 3.7 8.5 1.5 C 9.2 5.6 7.5 9.4 3.8 10'),
    (2, 'M 8.2 1.3 C 6.2 -.6 3.4 -.1 1.7 1.3 C -.8 3.8 1.1 5.9 3.5 5.1 Q 6.7 4.3 8.4 .4 Q 7.9 6.5 9.3 10'),
    (1, 'M 8.5 1.2 C 7.9 -.6 3.4 -.6 1.6 1.4 C -1.1 4.7 3.5 6.3 6.7 4.4 Q 8.8 3 8.5 1.2 C 10.1 6 7.8 10.5 4.9 10 Q 2.9 9.9 2.3 8.6'),
  ]
]

def parse(spec):
    tokens=iter(spec.split()); strokes=[]; ts=np.linspace(0,1,21)[1:, None]; t1=1-ts
    def point(): return np.array([float(next(tokens)),float(next(tokens))])
    for cmd in tokens:
        if cmd=='M':
            p=point(); strokes.append([p[None]]); continue
        if cmd=='L':
            q=point(); seg=t1*p + ts*q
        elif cmd=='Q':
            c,q=point(),point(); seg=t1**2*p + 2*t1*ts*c + ts**2*q
        elif cmd=='C':
            a,b,q=point(),point(),point(); seg=t1**3*p+3*t1*t1*ts*a+3*t1*ts*ts*b+ts**3*q
        else: raise ValueError(cmd)
        strokes[-1].append(seg); p=q
    return [np.vstack(s) for s in strokes]

# Smooth, bounded random warps of the *writing motion*: scale, italic angle,
# moving the waist, bending the stroke, and small independent pen drifts.
WIDTHS=[(.66,1.44),(.66,1.55),(.68,1.40),(.66,1.38),(.66,1.39),
        (.66,1.45),(.64,1.4),(.63,1.36),(.65,1.35),(.65,1.35)]

def path_points(paths, k, rng):
    u=rng.random(18); n=np.clip(rng.standard_normal(20),-2.3,2.3)
    lo,hi=WIDTHS[k]; width=lo*(hi/lo)**u[0]
    shear=(u[1]-.40)*1.22 + .085*n[0]; tilt=.16*n[1]
    widthtop=.19*n[2]; waist=.105*n[3]
    xb=.089*n[4]; yb=.045*n[5]; ripple=.035*n[6]
    width_px=1.35 + 3.65*(.50*u[2]+.50*u[3])
    if u[4]<.12: width_px=1.3+3.9*u[2]
    if k==1: width_px+=.35
    warp_paths=[]; rad_paths=[]
    for p in paths:
        x=(p[:,0]-5)*.1; y=(p[:,1]-5)*.1
        a=.52+.48*np.sin(np.pi*(y+.5)); b=np.cos(np.pi*x)
        v=(1-4*y*y).clip(-.3,1)
        drift=rng.uniform(-1,1,(5,2)); t=np.linspace(0,1,len(p))
        d=np.stack([np.interp(t,np.linspace(0,1,5),drift[:,i]) for i in range(2)],axis=-1)
        # Reduce discontinuity in a closed path, without eliminating pen drift.
        if np.linalg.norm(p[-1]-p[0])<.6: d-=t[:,None]*(d[-1]-d[0])*.88
        xp=x*width*(1+widthtop*2*y) + xb*v + ripple*np.sin(2*np.pi*y) + .043*d[:,0]
        yp=y + waist*v*b + yb*(1-4*x*x)*a + .029*d[:,1]
        xs=xp-shear*yp; ys=yp+tilt*xp
        warp_paths.append(np.column_stack((xs,ys)))
        # Slowly varying pen pressure; mild thinning near pen lift and touchdown.
        r=1 + .12*np.sin(t*np.pi*2+6.28*u[5])*n[7]+.08*d[:,1]
        if np.linalg.norm(p[-1]-p[0])>1:
            r-=.25*(np.exp(-t*18)+np.exp(-(1-t)*18))*u[6]
        rad_paths.append(np.clip(r,.53,1.48))
    ps=np.vstack(warp_paths); scale=(20-width_px)/max(np.ptp(ps,axis=0))
    mid=(ps.min(0)+ps.max(0))*.5
    # Sample the swept pen densely along arc length for area rasterization.
    samples=[]
    for p,r in zip(warp_paths,rad_paths):
        p=(p-mid)*scale
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        s=np.linspace(0,length[-1],max(2,1+int(np.ceil(length[-1]/.28))))
        xx=np.interp(s,length,p[:,0]); yy=np.interp(s,length,p[:,1]); rr=np.interp(s,length,r)*width_px*.5
        samples.append(np.stack((xx,yy,rr),axis=1))
    return np.vstack(samples), (.73+.58*u[7]), (.43+.37*u[8]), ((u[9]-.5)*.94,(u[10]-.5)*.94)

Y,X=np.mgrid[:28,:28]; DY,DX=np.mgrid[-4:5,-4:5]; DY=DY.ravel(); DX=DX.ravel()

def render(paths,k,rng):
    pts,axis,soft,jitter=path_points(paths,k,rng)
    px,py,r=pts.T
    # First pass in scratch coordinates for individual ink centroid; second
    # pass in the canonical frame. The first pass is NOT a separate exemplar.
    def ink(cx,cy):
        x=px+cx; y=py+cy
        ix=np.rint(x).astype(int)[:,None]+DX; iy=np.rint(y).astype(int)[:,None]+DY
        dist=np.sqrt(((ix-x[:,None])/axis)**2 + (iy-y[:,None])**2)
        coverage=np.clip(.5+(r[:,None]-dist)/soft,0,1)
        im=np.zeros(784); np.maximum.at(im,(iy*28+ix).clip(0,783),
                                        np.where((ix>=0)&(ix<28)&(iy>=0)&(iy<28),coverage,0))
        return im.reshape(28,28)
    im=ink(13.5,13.5); mass=im.sum()
    return ink(13.5+(13.5-(im*X).sum()/mass)+jitter[0],13.5+(13.5-(im*Y).sum()/mass)+jitter[1])

# Spatial filters. Signed first derivatives are half-wave rectified (edge
# normals distinguish the inner and outer borders of a loop). Positive ridges
# are unoriented as to direction of writing; they represent stroke tangents.
F=7; yf,xf=np.mgrid[-3:4,-3:4]
def filters():
    K=[]
    for i in range(8):
        a=i*np.pi/4; d=xf*np.cos(a)+yf*np.sin(a); t=-xf*np.sin(a)+yf*np.cos(a)
        h=d*np.exp(-.5*((d/1.05)**2+(t/1.4)**2)); h-=h.mean(); K.append(h/np.maximum(h,0).sum())
    for i in range(4):
        a=i*np.pi/4; d=xf*np.cos(a)+yf*np.sin(a); t=-xf*np.sin(a)+yf*np.cos(a)
        h=(1-(d/1.36)**2)*np.exp(-.5*((d/1.36)**2+(t/1.85)**2)); h-=h.mean(); K.append(h/np.maximum(h,0).sum())
    return np.array(K,dtype=np.float32)

# Individual template descriptors mirror the fixed input representation. Norm
# on the network side is per-channel AFTER stacking all channels into one plane,
# so the templates can use a dot product as a centered cosine similarity.
def signature(batch,K):
    b=np.asarray(batch,dtype=np.float32); m=len(b)
    ink=b.reshape(m,14,2,14,2).mean((2,4))
    win=sliding_window_view(np.pad(b,((0,0),(3,3),(3,3))),(F,F),axis=(1,2))
    raw=(np.ascontiguousarray(win).reshape(-1,F*F) @ K.reshape(-1,F*F).T).reshape(m,28,28,12)
    feat=np.maximum(raw,0).reshape(m,7,4,7,4,12).mean((2,4)).transpose(0,3,1,2)
    return [ink.reshape(m,-1),feat[:,:8].reshape(m,-1),feat[:,8:].reshape(m,-1)]

def per_row_norm(x):
    x=x-x.mean(axis=1,keepdims=True); return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)

# An independent geometric cue: gray-level persistence of enclosed pockets.
# Propagate exterior paper by dilation, bounded by the local paper value.
# The difference is the contrast of a surrounding barrier, rather than just
# a dark spot. Use a half-scale image to bound intermediate tensor sizes.
def topology(lay,weighted):
    def conv(name,source,w,b,p=0):
        w=np.asarray(w,np.float32); return weighted(name,'conv',source,w,b,kernel=list(w.shape[2:]),stride=1,padding=p)
    def scale(name,src,a,b=0): return conv(name,src,[[[[a]]]], [b])
    def pool(name,source,size,stride):return lay(name,'maxpool',source,size=size,stride=stride)
    paper=scale('paper','inkTiles',-1,1)
    # Seed the frame. One on a side and two in a corner is sufficient after
    # clamping the first dilation at the actual paper intensity.
    unit=scale('unit','inkTiles',0,1); kernel=np.zeros((1,1,3,3)); kernel[0,0,0,1]=kernel[0,0,1,0]=kernel[0,0,2,1]=kernel[0,0,1,2]=-1
    border=conv('frame',unit,kernel,[4],1)
    last=border
    for j in range(14):
        # A 3x3 maximum includes diagonals. min(spread,paper) equals
        # paper - relu(paper-spread); no binary threshold is needed.
        pad=conv('halo%d'%j,last,[[[[1]]]], [0],1)
        spread=pool('reach%d'%j,pad,[3,3],[1,1]); neg=scale('negReach%d'%j,spread,-1)
        rem=lay('diff%d'%j,'add',[paper,neg]); excess=lay('barrier%d'%j,'relu',rem)
        if j!=13:
            negex=scale('negBarrier%d'%j,excess,-1); last=lay('external%d'%j,'add',[paper,negex])
    # Suppress low-persistence texture. Absence of a hole is not a veto: a
    # writer may leave even a 0, 8, 9, or 6 open.
    d=scale('holeMargin',excess,1,-.33); d=lay('persistent','relu',d)
    avg=lay('holeMass','avgpool',d,size=[2,2],stride=[1,1])
    # A maximum over valid centers is robust to slant and width. Exclude the
    # middle row from upper/lower evidence but include it in general evidence.
    allh=pool('anyHole',avg,[13,13],[13,13]); bands=pool('whichHalf',avg,[6,13],[7,13]);
    # Add weak, one-sided evidence: presence is meaningful, absence often isn't.
    general=np.array([.042,-.07,-.054,-.055,.0,-.055,.034,-.055,.043,.034]);
    upper=np.array([.004,-.004,-.006,-.008,.012,-.012,-.028,-.008,.036,.043]);
    lower=np.array([.004,-.004,-.007,-.009,-.036,-.010,.045,-.009,.036,-.035]);
    g=weighted('cavityEvidence','dense',allh,general[:,None],np.zeros(10));
    h=weighted('cavityPosition','dense',bands,np.stack((upper,lower),axis=1),np.zeros(10));
    # Count Euler holes in superlevel sets at two contrast levels. For
    # 4-connected ink, chi=(Q1-Q3+2*Qdiagonal)/4 over all binary 2x2
    # neighborhoods (including the black exterior). Each minterm is one
    # threshold unit. The ramps keep gray-level transitions continuous.
    # A broken or disconnected stroke can lower the inferred hole count, so
    # use this only as positive evidence, never as an exclusion rule.
    ramps=conv('contrastRamps','image',np.full((4,1,1,1),16),[-16*.26,-16*.26-1,-16*.52,-16*.52-1]);
    ramps=lay('contrastOn','relu',ramps)
    clip=np.array([[1,-1,0,0],[0,0,1,-1]])[:,:,None,None]; binary=conv('softInk',ramps,clip,[0,0]);
    patterns=[]; bias=[]; sign=[]
    for chan in range(2):
        for mask in range(1,15):
            bits=np.array([(mask>>j)&1 for j in range(4)]); cnt=bits.sum()
            if cnt==2 and mask not in (6,9): continue
            w=np.zeros((2,2,2)); w[chan]=(2*bits-1).reshape(2,2); patterns.append(w); bias.append(1-cnt)
            sign.append((chan,{1:-.25,2:-.5,3:.25}[cnt]))
    pat=conv('eulerPatterns',binary,patterns,bias,1); pat=lay('matchedPatterns','relu',pat)
    # Sum local Euler contributions via global average and the known area.
    pat=lay('patternFrequency','avgpool',pat,size=[29,29],stride=[29,29]);
    w=np.zeros((2,len(sign)))
    for i,(c,a) in enumerate(sign): w[c,i]=a*29*29
    # A one-piece digit with two holes has chi=-1. Give 8 a bonus
    # proportional to supra-margin evidence from the two gray levels.
    double=weighted('extraPocket','dense',pat,w,np.full(2,-.30)); double=lay('twoLoops','relu',double)
    # Cap each contribution to keep texture in a noisy drawing from dominating.
    double=lay('twoLoopPlanes','reshape',double,shape=[1,1,2]); inv=scale('upperBound',double,-1,.7); inv=lay('roomLeft','relu',inv);
    capped=scale('doubleEvidence',inv,-1,.7);
    gains=np.array([-.036,-.035,-.035,-.035,-.032,-.035,-.032,-.035,.12,-.032]);
    two=weighted('doubleLoopVote','dense',capped,np.tile(gains[:,None]/2,(1,2)),np.zeros(10));
    return [g,h,two]

def build(where):
    where.mkdir(parents=True,exist_ok=True); layers=[]; weights={}
    def lay(name,op,src,**kw):
        layers.append(dict(name=name,op=op,inputs=src if isinstance(src,list) else [src],**kw));return name
    def weighted(name,op,src,w,b,**kw):
        weights[name+'.W']=np.asarray(w,np.float32);weights[name+'.b']=np.asarray(b,np.float32)
        return lay(name,op,src,out=len(w),**kw)
    K=filters(); base=lay('inkTiles','avgpool','image',size=[2,2],stride=[2,2]); sources=[base]
    for tag,kernels in [('edges',K[:8]),('ridges',K[8:])]:
        f=weighted(tag,'conv','image',kernels[:,None],np.zeros(len(kernels)),kernel=[F,F],padding=3,stride=1)
        f=lay(tag+'Positive','relu',f); sources.append(lay(tag+'Tiles','avgpool',f,size=[4,4],stride=[4,4]))
    dims=(196,392,196); names=('ink','contour','stroke'); gains=(.30,.43,.27)
    n=6144; chunk=32; rng=np.random.default_rng(20261120)
    banks=[np.empty((10*n,d),np.float32) for d in dims]
    for k,specs in enumerate(GLYPHS):
        styles=[]
        for freq,spec in specs: styles += [parse(spec)]*freq
        for off in range(0,n,chunk):
            b=[render(styles[rng.integers(len(styles))],k,rng) for j in range(chunk)]
            sig=signature(b,K)
            for W,v,d,g in zip(banks,sig,dims,gains):
                W[k*n+off:k*n+off+chunk]=per_row_norm(v)*(g/np.sqrt(d))
    sims=[]
    for src,d,W,name in zip(sources,dims,banks,names):
        flat=lay(name+'Plane','reshape',src,shape=[1,1,d]); z=lay(name+'Norm','norm',flat)
        sims.append(weighted(name+'Match','dense',z,W,np.zeros(10*n)))
    s=lay('correlation','add',sims); grid=lay('byClass','reshape',s,shape=[10,1,n]);
    # A softened maximum: average the winners in independent fixed groups,
    # combined with the very closest match. No statistics of the generated
    # population enter the weights. Pool hierarchically for batch memory.
    top=lay('winners16','maxpool',grid,size=[1,16],stride=[1,16]);
    top=lay('winners256','maxpool',top,size=[1,16],stride=[1,16]);
    near=lay('nearMatches','avgpool',top,size=[1,n//256],stride=[1,n//256]);
    best=lay('bestMatch','maxpool',top,size=[1,n//256],stride=[1,n//256]);
    near=weighted('support','dense',near,.36*np.eye(10),np.zeros(10));
    best=weighted('peak','dense',best,.64*np.eye(10),np.zeros(10));
    cues=topology(lay,weighted); lay('scores','add',[near,best]+cues)
    (where/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n');np.savez(where/'weights.npz',**weights)

if __name__=='__main__': build(Path(sys.argv[1]))
