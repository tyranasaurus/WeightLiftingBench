"""Construct a digit reader from pen paths. No data or other input files."""
import json
import sys
from pathlib import Path
import numpy as np

# Each entry: relative frequency, horizontal scale, pen strokes in a ten-unit em.
# A bang denotes a sharp corner; z closes a smooth contour. | lifts the pen.
# Coordinates are original construction instructions, not digitized observations.
PENS = [
 [ # 0: egg, oval, round rectangle, and slightly open/overtraced ovals
 (6,.70,'5,0 8.4,.9 10,4.2 9,7.7 6.3,9.8 3.4,10 1,8.3 0,5.5 1,2.5 2.8,.6 z'),
 (4,.70,'6,0 9.2,1.4 10,5.1 8.1,8.8 5,10 1.6,9 0,6.5 .4,3.3 2.7,.9 z'),
 (2,.64,'5,.2 8.8,.4 9.7,2.6 9.6,7 8.2,9.4 5.1,10 1.6,9.5 .2,7 .5,3 2,1 z'),
 (2,.70,'6.3,1.6 4.7,.4 2.6,1 .4,4 0,7.2 2,9.4 5.5,10 8.6,8.4 10,5 9.4,1.8 7.7,.2 5.4,.3 3.7,1.4'),
 (1,.67,'7,.6 4.2,.2 1.3,2.7 0,6.7 1.7,9.5 4.9,10 8,8.2 9.7,5 9.1,1.8 7.6,.9')],
 [ # 1: plain downstroke, bent or hooked entry, optional foot
 (10,.72,'5.4,0 5.1,3.2 4.8,6.6 4.6,10'),
 (5,.72,'5.7,0 5.1,4.9 4.8,8.4 5.2,10'),
 (3,.67,'4,1.1 5.8,0! 5.3,4.5 4.9,10'),
 (2,.68,'1.8,2.7 5.9,0! 5.4,4 5.2,10'),
 (1,.65,'1,2.8 5.4,0! 5.3,5.3 5,10 | .5,10 4.8,9.9 9.2,10'),
 (.6,.58,'3,1.4 5.6,0! 5.4,5.4 5.1,9.8! 8.5,10')],
 [ # 2: arched head, diagonal body, flat, curling or looped foot
 (5,.78,'.1,2.5 1.5,.9 4,0 6.8,.2 8.5,1.5 8.5,3.2 6.5,5.4 3,7.7 .6,9.7! 4.9,9.6 9.9,9.8'),
 (4,.80,'.5,2.1 3,.5 5.6,0 7.7,1.1 8.1,3.4 6.4,5.6 3.6,7.8 1,9.8 3.8,9.5 6.1,9.2 8.1,9.4 10,9.1'),
 (3,.85,'1.2,1.4 3.7,.1 6,.5 7.2,2.3 6.9,4.7 4.9,7.5 2.5,9.6 .6,10 -.1,9.1 .8,8 3,7.8 6.2,8.8 10.3,9.7'),
 (3,.78,'0,2.8 1.8,.6 4.4,0 6.6,1 7.3,3.1 6.4,5.9 4.2,8.3 1.4,9.8 .3,9.2 1.8,8.6 4.9,8.8 7.7,9.4 10,9'),
 (2,.76,'0,2.6 2.5,.7 5.5,0 8.8,.6 9.4,2.1 7.3,4.2 3.7,6.9 1.3,9.5! 5.4,9.6 9.6,8.9'),
 (1,.74,'.4,1.2 2.8,0 5.1,.6 6.5,2.5 6.1,4.6 3.6,7.6 2,9.8! 5.7,9.1 9.9,9.5'),
 (1,.79,'.6,3.8 .2,2.2 1.7,.6 4.4,0 7.4,.9 8.4,2.8 7.3,4.9 4.8,6.6 1.1,9.8! 5.5,9.3 10,9.1')],
 [ # 3: two right-hand bowls; also the flat or angular head
 (5,.71,'0,1.9 2.5,.5 5.8,0 8.4,.6 9.1,1.9 8.3,3.2 5.8,4.6 3.9,5.1! 6.4,4.8 8.8,5.6 9.6,7.3 8.4,9.1 5.8,10 2.7,9.8 .1,8.5'),
 (3,.70,'1.2,1 4.5,0 7.7,.3 9.2,1.5 8.5,3.1 5.2,4.9! 8.2,5.3 10,7 9.1,8.7 6.3,9.9 3.2,10 0,9.1'),
 (3,.77,'0,.9 4.3,.5 9.5,0! 6.3,2.7 3.7,4.9! 6.6,4.6 8.7,5.3 9.6,6.8 8.8,8.7 6.1,9.9 2.7,9.9 .2,8.5'),
 (2,.73,'0,1.5 3.3,.2 7.2,.1 8.6,.9 7.9,2.4 4.9,4.1 2.7,5 3.2,5.4 5.2,5.1 7.2,5 9,6 9.4,7.6 7.3,9.5 3.4,10 .3,8.3'),
 (2,.72,'.6,.5 4.7,0 8.5,.6 8.9,2 6.9,3.5 4.5,4.7! 7.5,5.6 9.6,7.3 9.1,8.9 6.6,9.9 3.2,10 .2,9.4'),
 (1,.74,'0,1.3 3.5,.2 7.2,.2 8.9,1.2 7.6,3.1 4.8,4.7! 7.5,5.5 9.1,7.6 8.7,9.3 6.7,10 3.8,9.7 1.1,8.1 .4,7.1')],
 [ # 4: open chair forms, angular and rounded, or closed triangle
 (5,.77,'1.3,.4 .5,2.6 0,4.6 .5,5.9 3.4,6.1 7,5.9 10,5.7 | 7.8,.1 7.6,3.6 7.7,6.8 7.9,10'),
 (4,.78,'4.3,0 2.2,2.5 0,5.8! 4.1,5.8 9.7,5.6 | 7.6,.2 7.4,5.5 7.6,10'),
 (3,.74,'1.3,.4 .3,3 0,5 1.2,6 3.7,5.9 6.5,5.1 8,4.1 | 8.3,0 7.7,4.3 7.6,7.1 8,10'),
 (3,.77,'7.7,0 4.4,2.5 0,6.1! 5,6 10,5.8 | 7.7,0 7.5,4.4 7.5,7.5 7.7,10'),
 (2,.78,'1.3,0 .8,2.6 .5,5 .8,6.2! 4,5.9 7.5,5.3! 8.1,.1! 7.7,4.7 7.5,10'),
 (2,.68,'6.2,0 3.3,2.4 0,5.8! 3.5,6 8.9,5.6 | 6.9,.8 6.6,5.9 7,10'),
 (1,.73,'1.3,.4 .2,3.2 .5,5.4 2.9,5.9 5.8,5.4 8.5,.1! 7.8,5 7.8,10')],
 [ # 5: top bar plus hanging bowl, angular and flowing versions
 (4,.73,'9.6,.1 6.2,.3 2,.5! 1.6,2.5 1,4.8! 3.7,4.1 6.6,4.1 8.7,5.2 9.3,7 8.2,8.8 5.6,9.9 2.6,9.9 0,8.7'),
 (4,.73,'2.8,.6 2,2.5 1.3,4.3! 4.5,4.2 7.1,5 8.8,6.7 8.3,8.4 5.6,9.8 2.6,10 0,9.1 | 2.1,.5 4.8,.4 7.2,.5 10,0'),
 (3,.77,'9.6,.2 6.3,.1 3.4,.4 1.2,1.3 .8,2.8 2.1,4.1 5.6,4.9 8.1,6.1 8.8,7.6 7.5,9.1 5,10 2.1,9.8 0,8.8'),
 (2,.77,'2,0 1.2,2.2 .5,4.3! 3.1,4 6.3,4.8 8.8,6.4 9.2,8 7.5,9.4 4.7,10 2,9.4 1,8.2 | 2.1,.4 5.8,.3 10,.1'),
 (2,.75,'9.4,0 6.1,.7 2.6,.5! 1.8,2.7 1.7,4.2! 4.4,4.1 6.8,5 8.5,6.8 8.1,8.6 5.8,9.8 2.9,10 .6,9.1 .5,7.9 2.1,7.6'),
 (2,.75,'10,.2 7.8,.8 5,.6 2.8,.3! 1.6,2.5 .8,4.3 3,4.7 5.6,5.6 7.1,7.1 6.7,8.8 4.5,10 1.9,9.7 .1,8.3'),
 (1,.76,'9,.2 6.4,.2 2.6,.8 1,2.2 1.5,3.9 4.5,5.1 7.2,6.4 8.3,8 6.4,9.6 3.2,10 0,9.1')],
 [ # 6: long approach stroke flowing into a low loop
 (5,.68,'7.9,0 5.2,1.2 2.3,3.6 .6,6.2 1,8.4 3.4,9.9 6.7,9.8 9,8.3 9,6.6 7.2,5.4 4.8,5.2 2.3,6.1 1.2,7.4'),
 (3,.68,'7.4,0 4.8,1.9 2.1,4.7 .7,7.5 1.7,9.2 4.1,10 7.1,9.5 9,7.9 8.5,6.1 6.3,5.1 3.9,5.3 1.6,6.8'),
 (3,.72,'8.9,1.3 7.4,.2 5.2,.3 2.8,1.9 1.1,4.4 .8,7.1 2,9.1 4.8,10 7.8,9.2 9.2,7.5 8.5,5.7 6.2,4.9 3.8,5.5 1.6,7.1'),
 (2,.71,'5.8,0 3.3,1.9 1.7,4.3 1,7.2 2.2,9.3 4.9,10 8.1,9.3 9.5,7.7 8.8,6.1 6.6,5.2 4.1,5.9 1.2,8'),
 (2,.74,'7.8,0 4.4,2.3 1.9,5.4 1.1,7.9 2.7,9.7 5.7,10 8.5,9 9.6,7.4 8.5,6 6.7,5.4 4.8,5.7'),
 (1,.70,'8.2,.2 6.1,.3 3.6,1.3 1.5,3.3 .8,6 .9,7.7 2.6,9.5 5,10 7.6,9.4 9,7.9 8.8,6.1 7.1,4.7 4.9,4.4 2.3,5.5 .9,6.6')],
 [ # 7: a roof and descending stroke, optionally a serif or a cross stroke
 (6,.81,'0,.4 3.2,.1 6.6,.2 9.8,.3! 7.3,3.4 5.3,6.4 3.4,10'),
 (4,.79,'0,.1 2.8,.4 6.3,.3 9.2,0! 7.2,3 5.8,6.1 5,10'),
 (3,.80,'.3,1.7 .6,.3! 3.5,0 6.6,.2 9.6,.8! 7.7,3.8 5.6,6.8 3.1,10'),
 (3,.77,'.4,.5 3,.1 6.4,.2 9.2,.4! 7,3.5 5.3,6.7 4.1,10 | 2.4,5.4 5.9,5.3 8.7,5.1'),
 (2,.79,'0,.2 2.4,.8 5.3,.7 9.1,.1! 8.3,2.5 6.5,5.5 4.1,8.3 3.1,10'),
 (1,.81,'0,1.4 .2,.1! 3.2,.3 6.3,.2 9.3,0! 7.4,3.1 5.6,6.1 5,10 | 3.4,5.3 6,5.2 8.3,5.4')],
 [ # 8: continuous figure eights, S and return, or stacked ovals
 (5,.66,'4.9,4.7 2,3.5 1.1,2.1 2.2,.6 5,0 7.9,.5 8.6,1.9 7.5,3.4 4.9,4.7 2.3,6.1 .8,7.9 1.7,9.4 4.6,10 7.5,9.5 9.1,8 8.6,6.7 7.1,5.7 z'),
 (4,.69,'5,0 2.5,.5 1.3,1.9 2.1,3.5 5.2,5 8.3,6.6 9.2,8.2 7.7,9.7 4.7,10 1.7,9.2 .8,7.5 2.4,6 5.6,4.5 8,2.9 8.7,1.3 7.4,.3 z'),
 (3,.63,'6.3,.6 3.7,.1 1.5,1.3 1.6,3 3.9,4.5 6.7,5.8 8.7,7.5 8.3,9.1 5.7,10 2.4,9.6 .8,8.3 1.6,6.6 4.4,4.8 7.4,3.2 8.4,1.6 7.4,.4 5.9,.3'),
 (2,.72,'5,4.7 2,4 .7,2.1 1.8,.7 4.8,0 7.9,.6 8.9,2.2 7.6,3.9 z | 5,4.7 8.1,5.7 9.4,7.6 8.1,9.4 5,10 1.9,9.3 .7,7.6 2.2,5.6 z'),
 (2,.65,'7.3,1.1 5.5,.1 2.8,.4 1.4,1.8 2.4,3.4 5,5.1 7.8,6.9 8.6,8.5 7,9.9 4.2,10 1.4,9 .8,7.4 2.5,5.6 5.5,4.1 7.7,2.6 8,1.3'),
 (1,.72,'4.7,4.4 2.2,3.1 2,1.4 3.6,.2 6.1,.1 8.1,1.2 7.9,2.7 5.8,4.4 2.9,5.9 .9,7.5 1.5,9.1 4.1,10 7.5,9.8 9.1,8.2 8.8,6.6 6.8,5.3 z')],
 [ # 9: high loop with a right downstroke; sometimes a hooked descender
 (5,.71,'8.4,1.6 6.9,.3 4.3,.1 1.8,1.2 .7,3.1 1.2,4.7 3.3,5.6 5.8,5.1 7.8,3.4 8.6,1.1! 8.2,4 8.2,6.7 8.5,10'),
 (4,.69,'8.2,1.2 6.2,.1 3.7,.4 1.4,1.9 .8,3.9 2.4,5.2 4.9,5.3 7.6,3.9 8.7,1.5! 8.1,4.6 7.4,7.4 6.2,10'),
 (3,.71,'8.4,3.4 8.3,1.1 6.5,.1 3.7,.3 1.4,2 .9,3.8 2.1,5 4.8,5.6 7.2,4.7 8.7,2.8! 8.9,5.5 8.2,8.2 6.8,9.6 5.1,10'),
 (3,.69,'8.3,2.6 7.3,.7 5.2,0 2.8,.6 1,2.2 .9,3.9 2.8,5.3 5.6,5.3 7.7,4.3 z | 8.2,1.5 8,4.3 8.2,7.6 8.8,10'),
 (2,.74,'7.6,.7 5.2,.1 2.4,1.1 .9,2.9 1.1,4.5 3.2,5.4 5.6,4.8 7.8,3.1 8.9,.4! 8,3.6 7.4,6.7 7.6,10'),
 (2,.71,'7.9,1.7 6.6,.4 4.3,.2 1.9,1.2 .8,3.1 1.3,4.7 3.6,5.4 6.4,4.8 8.3,3! 8,6 7,8.6 5.2,9.9 3.6,9.7'),
 (1,.70,'6.5,.8 4.1,.3 1.7,1.4 .8,3.4 1.8,5 4.2,5.6 6.7,4.6 8.7,1.9! 8.1,5 8.4,7.6 9.2,10')]
]


def spline(spec):
    """Construct cubic Hermite interpolation rows once for each pen style."""
    knots, rows, phases = [], [], []
    for stroke in spec.split('|'):
        tok = stroke.split(); closed = tok[-1]=='z'
        if closed: tok.pop()
        pts = np.array([list(map(float,t.strip('!').split(','))) for t in tok])
        corner = [t.endswith('!') for t in tok]; n = len(pts); base=len(knots)
        knots.extend(pts); ns = n if closed else n-1
        for j in range(ns):
            k=(j+1)%n; a=(j-1)%n; b=(k+1)%n
            steps=max(5, int(np.ceil(np.linalg.norm(pts[k]-pts[j])*6)))
            ts=np.arange(steps)/steps
            if j==ns-1 and not closed: ts=np.append(ts,1.)
            for t in ts:
                row=np.zeros(n); t2=t*t; t3=t2*t
                row[j]+=2*t3-3*t2+1; row[k]+=-2*t3+3*t2
                h=t3-2*t2+t; g=t3-t2
                if corner[j] or (j==0 and not closed): row[k]+=h; row[j]-=h
                else: row[k]+=.5*h; row[a]-=.5*h
                if corner[k] or (k==n-1 and not closed): row[k]+=g; row[j]-=g
                else: row[b]+=.5*g; row[j]-=.5*g
                rows.append((base,row)); phases.append(1. if closed else max(0.,np.sin(np.pi*(j+t)/ns))**.42)
    W=np.zeros((len(rows),len(knots)))
    for i,(b,r) in enumerate(rows): W[i,b:b+len(r)]=r
    return np.array(knots), W, np.array(phases)

Y,X=np.mgrid[:28,:28]; OY,OX=np.mgrid[-4:5,-4:5]; OY=OY.ravel(); OX=OX.ravel()

def draw(style, rng):
    """Render one independent model. No number is estimated across renderings."""
    aspect, knots, basis, taper = style
    x,y=(basis @ (knots+rng.normal(0,rng.uniform(.10,.36),knots.shape))).T
    u=(x-5)/5; v=(y-5)/5
    h=rng.normal(size=9)
    # Smooth changes of waist height, relative bowl widths, bow, and keystone.
    x=x+.80*h[0]*(1-v*v)+.62*h[1]*u*v+.43*h[2]*np.sin(np.pi*v)+.25*h[3]*(1-u*u)
    y=y+.83*h[4]*(1-v*v)+.44*h[5]*u*v+.47*h[6]*u+.33*h[7]*(1-u*u)+.22*h[8]*np.sin(np.pi*v)
    x=(x-5)*aspect*np.clip(np.exp(rng.normal(0,.205)),.54,1.68); y=y-5
    # Shear and rotation have different effects on crossbars and loops.
    x=x-y*np.clip(rng.normal(.115,.245),-.60,.78)
    a=np.clip(rng.normal(0,.105),-.30,.30); c=np.cos(a); s=np.sin(a)
    x,y=c*x-s*y,s*x+c*y
    # Choose pressure in output-pixel units. Individual strokes can taper.
    width=np.clip(rng.normal(3.05,.86),1.25,5.65)
    r=.5*width*(1+rng.uniform(-.21,.21)*v+rng.uniform(-.12,.12)*u+rng.uniform(-.13,.13)*np.sin(np.pi*(v+.35*u)))
    r=r*(1-rng.uniform(.04,.36)*(1-taper)); r=np.clip(r,.38,3.45)
    scale=(rng.uniform(19.5,20.45)-width)/max(np.ptp(x),np.ptp(y),.01)
    x=x*scale; y=y*scale
    x+=13.5-(x.min()+x.max())*.5; y+=13.5-(y.min()+y.max())*.5
    ix=np.rint(x).astype(int)[:,None]+OX; iy=np.rint(y).astype(int)[:,None]+OY
    d=np.sqrt((ix-x[:,None])**2+(iy-y[:,None])**2)
    alpha=np.clip(.5+(r[:,None]-d)/rng.uniform(.87,1.37),0,1)
    im=np.zeros(784); np.maximum.at(im,(iy*28+ix).clip(0,783).ravel(),alpha.ravel()); im=im.reshape(28,28)
    # Replicate the stated centering convention on this rendering by itself.
    mass=im.sum(); cx=(im*X).sum()/mass; cy=(im*Y).sum()/mass
    shiftx=13.75+rng.uniform(-.64,.64)-cx; shifty=13.75+rng.uniform(-.64,.64)-cy
    sx=X-shiftx; sy=Y-shifty; xx=np.floor(sx).astype(int); yy=np.floor(sy).astype(int)
    fx=sx-xx; fy=sy-yy; xx=xx.clip(0,26); yy=yy.clip(0,26)
    return (1-fy)*((1-fx)*im[yy,xx]+fx*im[yy,xx+1])+fy*((1-fx)*im[yy+1,xx]+fx*im[yy+1,xx+1])

# Analytic image operators. Their coefficients are independent of the pen models.
F=np.float32
ug=np.arange(6)-2.5; g=np.exp(-ug**2/(2*1.38**2)); blur=np.outer(g,g)/g.sum()**2
vy,vx=np.mgrid[-2:3,-2:3]; envelope=np.exp(-(vx*vx+vy*vy)/(2*1.03**2))
edges=np.array([envelope*(np.cos(a)*vx+np.sin(a)*vy) for a in np.arange(8)*np.pi/4])
smooth=np.outer([1,3,3,1],[1,3,3,1])/64.
dirs=[(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
yn,xn=np.mgrid[:14,:14]
rays=np.zeros((8,14,14,196),F)
for i,(dy,dx) in enumerate(dirs):
    for j in range(14):
        ys=yn+dy*j; xs=xn+dx*j; ok=(ys>=0)&(ys<14)&(xs>=0)&(xs<14)
        rays[i,yn[ok],xn[ok],ys[ok]*14+xs[ok]]=1.35

def patches(a,k,stride,p):
    ap=np.pad(a,[(0,0)]*(a.ndim-2)+[(p,p),(p,p)])
    return np.lib.stride_tricks.sliding_window_view(ap,(k,k),axis=(-2,-1))[...,::stride,::stride,:,:]

def avg2(a):
    return (a[...,::2,::2]+a[...,1::2,::2]+a[...,::2,1::2]+a[...,1::2,1::2])*.25

DIMS=(196,392,392,49,3); D=sum(DIMS)

def describe(a):
    """Apply the fixed feature maps separately to each drawing (batch dimension only for speed)."""
    a=a.astype(F); n=len(a)
    low=patches(a,6,2,2).reshape(n,196,36) @ blur.astype(F).ravel()
    ed=patches(a,5,1,2).reshape(n,784,25) @ edges.reshape(8,25).astype(F).T
    ed=avg2(np.maximum(ed.transpose(0,2,1).reshape(n,8,28,28),0))
    ed=patches(ed,4,2,1).reshape(n,8,49,16) @ smooth.astype(F).ravel()
    small=avg2(a); vis=np.empty((n,8,14,14),F)
    for i,(dy,dx) in enumerate(dirs):
        acc=small.copy()
        for t in range(1,14):
            yy0=max(0,-t*dy); yy1=min(14,14-t*dy); xx0=max(0,-t*dx); xx1=min(14,14-t*dx)
            acc[:,yy0:yy1,xx0:xx1]+=small[:,yy0+t*dy:yy1+t*dy,xx0+t*dx:xx1+t*dx]
        vis[:,i]=np.maximum(0,1-1.35*acc)
    vis=avg2(vis)
    ink=a.reshape(n,14,2,14,2).max(axis=(2,4)); cap=1-ink; block=np.ones_like(ink)
    for _ in range(13):
        outside=1-np.pad(block,((0,0),(1,1),(1,1)))
        spill=np.maximum.reduce([outside[:,i:i+14,j:j+14] for i in range(3) for j in range(3)])
        hole=np.maximum(cap-spill,0); block=ink+hole
    holes=avg2(hole); bands=np.stack([holes[:,j:j+3,:].max(axis=(1,2)) for j in (0,2,4)],axis=1)
    parts=[]
    for arr,coef in ((low,.38),(ed,.44),(vis,.18)):
        arr=arr.reshape(n,-1); centered=arr-arr.mean(axis=1,keepdims=True)
        standardized=centered/np.sqrt((centered**2).mean(axis=1,keepdims=True)+1e-5)
        parts.append(standardized*(coef/arr.shape[1]))
    # Per-template negative squared cavity distances (query-only term cancels).
    bias=np.zeros(n,F)
    for arr,coef in ((holes,.98/49),(bands,.015)):
        arr=arr.reshape(n,-1); parts.append(2*coef*arr); bias-=coef*(arr**2).sum(axis=1)
    return np.concatenate(parts,axis=1),bias


def network(out):
    weights={}; layers=[]
    def layer(name,op,inp,**settings):
        layers.append(dict(name=name,op=op,inputs=inp if isinstance(inp,list) else [inp],**settings)); return name
    def affine(name,op,inp,w,b=0,**settings):
        w=np.asarray(w,F); weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(np.asarray(b,F),(w.shape[0],)).copy()
        return layer(name,op,inp,out=w.shape[0],**settings)
    def conv(name,inp,w,b=0,s=1,p=0):
        w=np.array(w,F,ndmin=4); return affine(name,'conv',inp,w,b,kernel=list(w.shape[-2:]),stride=s,padding=p)
    def pool(name,op,inp,k,s): return layer(name,op,inp,size=k,stride=s)
    def avg(name,inp): return pool(name,'avgpool',inp,[2,2],[2,2])
    def view(name,inp,*dims): return layer(name,'reshape',inp,shape=list(dims))
    def norm(name,inp): return layer(name,'norm',inp)
    def relu(name,inp): return layer(name,'relu',inp)
    def add(name,*inputs): return layer(name,'add',list(inputs))

    low=norm('ink_contrast',conv('ink_scale_space','image',blur,s=2,p=2))
    ed=relu('positive_edges',conv('steered_derivatives','image',edges[:,None],p=2))
    ed=avg('edge_cells',ed)
    ed=conv('overlapping_histograms',ed,np.eye(8)[:,:,None,None]*smooth,s=2,p=1)
    ed=norm('histogram_contrast',view('histogram_plane',ed,1,56,7))
    small=avg('area_coverage','image')
    vis=relu('clear_sightlines',affine('sightlines','dense',small,-rays.reshape(-1,196),1))
    vis=avg('silhouette_cells',view('compass_directions',vis,8,14,14))
    vis=norm('silhouette_contrast',view('silhouette_plane',vis,1,56,7))
    ink=pool('ink_barrier','maxpool','image',[2,2],[2,2]); cap=conv('free_space',ink,-1,1)
    block=conv('unreached',ink,0,1)
    for i in range(13):
        prefix='flood%02d_'%i
        outside=conv(prefix+'boundary',block,-1,1,p=1)
        spill=pool(prefix+'spread','maxpool',outside,[3,3],[1,1]); neg=conv(prefix+'negative',spill,-1)
        hole=relu(prefix+'enclosed',add(prefix+'residual',cap,neg))
        if i!=12: block=add(prefix+'barrier',ink,hole)
    holes=avg('cavity_cells',hole); bands=pool('cavity_zones','maxpool',holes,[3,7],[2,1])
    # Embed into disjoint coordinate ranges to concatenate with the available add op.
    pieces=[]; start=0
    for i,(size,src) in enumerate(zip(DIMS,(low,ed,vis,holes,bands))):
        w=np.zeros((D,size),F); w[start:start+size]=np.eye(size); start+=size
        pieces.append(affine('embed_%d'%i,'dense',src,w))
    features=add('shape_coordinates',*pieces)

    # Each score compares against ONE rendered pen trajectory. Group maxima
    # temper a lucky one-off match without merging drawings into prototypes.
    rng=np.random.default_rng(735129); groups=4; per_group=1024; total=groups*per_group
    ws=[np.empty((10,per_group,D),F) for _ in range(groups)]; bs=[np.empty((10,per_group),F) for _ in range(groups)]
    for digit,lib in enumerate(PENS):
        styles=[(aspect,*spline(spec)) for _,aspect,spec in lib]; freq=np.array([freq for freq,_,_ in lib]); freq=freq/freq.sum()
        for group in range(groups):
            for j in range(0,per_group,32):
                drawings=np.stack([draw(styles[rng.choice(len(styles),p=freq)],rng) for _ in range(32)])
                ws[group][digit,j:j+32],bs[group][digit,j:j+32]=describe(drawings)
    maxima=[]
    for i in range(groups):
        sc=affine('individual_models_%d'%i,'dense',features,ws[i].reshape(-1,D),bs[i].ravel())
        group=view('class_by_model_%d'%i,sc,10,1,per_group)
        maxima.append(pool('best_in_group_%d'%i,'maxpool',group,[1,per_group],[1,per_group]))
    # Compare best overall and the mean of four within-group best scores.
    cols=[]
    for i,m in enumerate(maxima):
        w=np.zeros((40,10),F); w[np.arange(10)*4+i,np.arange(10)]=1
        cols.append(affine('group_slot_%d'%i,'dense',m,w))
    summary=view('four_matches',add('group_scores',*cols),10,1,4)
    mx=pool('closest','maxpool',summary,[1,4],[1,4]); av=pool('consensus','avgpool',summary,[1,4],[1,4])
    add('digit_scores',affine('closest_weight','dense',mx,.68*np.eye(10)),affine('consensus_weight','dense',av,.32*np.eye(10)))
    out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
    np.savez(out/'weights.npz',**weights)

if __name__=='__main__': network(Path(sys.argv[1]))
