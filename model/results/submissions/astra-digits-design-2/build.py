"""A handwritten geometric shape grammar and its matched-filter network.

No files are used as sources. All coordinates below are pen-design coordinates
(on an arbitrary rough 7 by 10 writing grid), not image measurements.
"""
import sys, json
from pathlib import Path
import numpy as np

# Each list entry is one pen style: smooth curves through knots (S), straight
# line segments (L), pen lifts (/). Repeated entries need not be stored to
# adjust a style's frequency; the integer before the path is its multiplicity.
STYLES = [
 [ # zero: oval, somewhat square oval, pointed turns, a cursive overlap
 (6,'S 3.7,0 1.4,.75 .15,3.7 .1,7.1 1.1,9.35 3.1,10 5.6,9.0 6.85,6.25 6.9,3.1 6.1,.9 3.7,0 2.7,.22'),
 (3,'S 4.6,.15 2.1,.55 .55,2.2 0,5.2 .3,8.4 1.4,9.85 3.9,9.9 6.2,8.7 7,6.3 7,3.4 6.4,.95 4.6,.15'),
 (2,'S 4.5,0 2.4,.9 .55,3.4 0,6.9 .6,9.1 2.2,10 4.85,9 6.65,6.7 7,3.9 6.2,1.5 4.5,0 3.75,1.25'),
 (1,'S 4.8,1.7 4.5,.1 2.8,.3 1.0,2.45 .15,5.6 .3,8.4 1.75,9.9 4.4,9.45 6.15,7.5 7,4.2 6.55,1.9 5.3,.75 3.9,.8')],
 [ # one: plain, short upstroke, flag and foot
 (10,'S 3.6,0 3.45,3.45 3.55,6.7 3.35,10'),
 (4,'S 3.95,0 3.8,2.5 3.5,5.3 3.25,7.95 3.5,10'),
 (4,'L 1.8,1.9 3.8,0 3.45,10'),
 (2,'L .85,3.7 4.0,0 3.75,10'),
 (1,'L 1.1,2.9 3.8,0 3.6,9.7 / L .8,9.9 6.2,9.75'),
 (1,'S 2.35,.6 3.5,.1 3.65,2.5 3.35,7.55 3.4,9.95 4.35,9.5')],
 [ # two: arched top and baseline (pointed, swept or looped)
 (6,'S .15,2.4 1.4,.6 3.9,0 5.8,.6 6.5,2.3 5.8,4.35 3.35,6.7 .65,9.2 / S .65,9.2 1.55,9.55 4.2,9.6 7.25,9.7'),
 (4,'S .45,2.25 1.85,.55 3.7,.05 5.3,.75 5.85,2.95 4.7,5.65 2.4,8.35 .3,9.9 / S .3,9.9 2.5,9.25 4.8,9.2 7.35,9.65'),
 (3,'S .6,1.4 2.2,.15 3.75,.4 4.9,2.25 4.85,4.9 3.8,7.65 1.8,9.65 .3,9.6 .8,8.2 2.6,7.95 4.7,8.95 7.7,9.8'),
 (2,'S .8,3.4 .65,1.65 1.9,.2 4,.15 5.8,1.2 6.1,3.5 4.9,5.5 2.4,7.5 .1,9.55 / S .1,9.55 2.35,9.8 5.15,9.55 7.5,8.95'),
 (2,'S 0,1.3 2.45,.05 4.3,.3 5.5,1.8 5.35,4.05 3.45,6.55 1.3,8.75 .9,9.65 3.25,9.3 5.2,9.55 7.25,10.0'),
 (1,'S .45,.8 2.25,.05 3.7,.8 4.85,3.05 4.95,5.15 3.55,7.65 1.2,9.6 .1,9.65 .55,8.25 2.45,7.55 5.3,8.2 7.65,8.3')],
 [ # three: paired lobes, angular Z-top, crossed central spur
 (5,'S .15,1.45 2.2,.15 4.8,.1 6.25,1.05 6.05,2.65 4.5,4.0 2.75,4.75 / S 2.75,4.75 4.95,4.8 6.45,5.9 6.55,7.7 5.45,9.3 3.45,10 1.5,9.7 .05,8.55'),
 (3,'S .6,.6 2.6,.05 4.95,.3 6.3,1.65 5.45,3.35 3.9,4.5 2.8,4.9 3.4,4.35 5.5,4.65 6.8,6.15 6.3,8.45 4.15,9.85 1.85,9.7 .35,8.7'),
 (3,'L .35,.65 6.25,.15 2.45,4.65 / S 2.45,4.65 4.85,4.4 6.65,5.55 6.75,7.7 5.0,9.5 2.65,10 .1,9.15'),
 (2,'S .45,.4 3.4,0 5.5,1.1 5.9,2.9 4.5,4.35 2.7,5.05 / S 2.7,5.05 4.8,5.15 6.15,6.65 5.9,8.45 4.1,9.85 1.6,9.95 .25,9.5'),
 (1,'S .1,1.4 2,.1 4.45,.05 5.9,1.1 5.6,3.05 3.95,4.55 1.85,5.1 2.35,4.45 4.75,4.8 6.4,6.3 6.65,8.45 5.1,9.85 2.85,9.85 .45,8.7')],
 [ # four: open two-stroke 4, continuous round elbow, closed triangle
 (6,'L 2.1,.15 .45,5.8 6.75,5.25 / L 5.45,.2 5.1,10'),
 (3,'S 1.85,.05 .95,2.65 .25,5.05 .65,5.8 3.0,5.75 6.25,5.05 / S 5.7,.55 5.4,3.3 5.45,6.75 5.2,10'),
 (3,'L 4.85,.05 .05,6.15 7.1,5.75 / L 4.85,.05 5.0,10'),
 (3,'S 1.5,.1 .65,2.9 .5,4.95 1.5,5.8 3.2,5.55 5.7,4.2 / L 5.8,.45 5.3,10'),
 (2,'L 3.3,.1 .2,6.25 6.75,5.75 / L 5.45,1.15 5.1,10'),
 (1,'L 5.2,.4 .2,6 7.1,5.9 / L 5.0,2.2 5.35,10')],
 [ # five: cap, left stem and right-facing bottom bowl. Both pen orders.
 (5,'S 6.55,.1 3.85,.35 1.0,.25 / L 1.0,.25 .65,4.6 / S .65,4.6 2.8,3.95 4.95,4.35 6.4,6.05 6.25,8.1 4.55,9.7 2.3,9.9 .05,8.95'),
 (3,'S 6.95,.35 4.2,.2 1.0,.15 / S 1.0,.15 .75,2.5 .85,4.1 2.5,4.3 4.95,5.2 6.25,6.8 5.9,8.5 4.1,9.8 1.85,10 0,9.35'),
 (3,'S 1.8,.15 .7,2.4 .35,3.95 1.05,4.5 3.5,4.65 5.65,5.65 6.4,7.45 5.25,9.25 2.9,10 .3,9.45 / S 1.55,.15 3.75,.4 6.7,.05'),
 (2,'S 6.8,.0 4.0,.5 1.95,.3 / L 1.95,.3 .25,5.35 / S .25,5.35 2.15,4.2 4.3,4.05 5.95,5.5 6.0,7.85 4.15,9.6 1.9,9.95 .2,9.45'),
 (2,'S 2.3,.25 .9,1.55 .6,3.3 1.8,4.65 4.15,5.65 5.75,7.35 5.2,9 3.0,9.95 .3,9.8 / S 1.25,.45 4,.15 7.45,.5')],
 [ # six: sweeping descending stroke and lower loop, sometimes with a hood
 (5,'S 5.7,0 3.35,1.2 1.4,3.7 .4,6.4 .65,8.6 2.35,9.95 4.65,9.7 6.15,8.3 6.4,6.4 5.1,5.15 3.2,5.2 1.4,6.4 .65,7.3'),
 (4,'S 4.85,0 3.05,1.55 1.45,4.1 .65,6.65 1.1,8.7 2.95,9.95 5.45,9.5 6.55,7.75 6.1,5.9 4.55,5.05 2.6,5.65 1.4,7.25'),
 (2,'S 6.45,1.1 5.15,.0 3.4,.55 1.55,2.75 .45,5.9 .7,8.05 2.0,9.65 4.4,9.9 6.0,8.65 6.55,6.6 5.6,4.95 3.65,4.65 1.55,5.85 .7,7.35'),
 (2,'S 4.2,.0 2.25,2.7 .95,5.4 .6,7.85 1.7,9.55 4.05,9.9 6.0,8.9 6.45,7.3 5.65,6.1 4.1,5.9 2.5,6.55 1.7,7.8'),
 (1,'S 3.65,0 2.1,2.6 1.2,5.85 1.65,8.6 3.2,10 5.4,9.65 6.65,8.0 6.0,6.1 4.45,5.3 3.0,5.7 2.0,6.65')],
 [ # seven: straight, waved cap, serif, crossed
 (6,'L .1,.4 6.7,.15 2.95,10'),
 (3,'S 0,.65 2.5,.25 4.55,.6 6.8,.25 / S 6.8,.25 5.35,3.3 3.65,6.75 2.95,10'),
 (3,'L .15,2.1 .4,.3 6.8,.25 2.8,10'),
 (2,'L 0,.15 6.7,.5 2.8,10 / L 2.25,5.1 6.45,4.95'),
 (1,'S .1,2 .6,.3 2.45,.15 4.6,.7 6.9,.45 / L 6.9,.45 2.95,10 / L 2.3,4.9 6.65,5.1')],
 [ # eight: single figure eight in both pen directions, double bowls
 (5,'S 3.7,4.75 1.45,3.25 .85,1.6 2.15,.3 4.35,.1 5.95,.95 6.0,2.4 4.5,3.95 2.25,5.55 .7,7.15 .65,8.8 2.2,9.9 4.25,9.85 5.85,8.8 6.15,7.25 5.3,6.0 3.7,4.75'),
 (4,'S 4.15,.3 2.1,.2 .9,1.45 1.35,3.2 3.2,4.75 5.35,6.1 6.2,7.6 5.6,9.2 3.9,10 1.8,9.7 .6,8.4 .65,6.8 2.05,5.4 4.45,3.8 5.9,2.2 5.75,.85 4.1,.2 2.85,.35'),
 (3,'S 4.7,.35 2.5,.15 1.1,1.35 1.45,3.1 3.35,4.65 5.3,6.25 6.0,8.0 5.05,9.6 2.95,9.95 1.05,9.0 .65,7.5 1.5,5.95 3.65,4.4 5.45,2.95 5.95,1.4 4.85,.45'),
 (2,'S 3.45,4.7 1.55,4.0 .75,2.4 1.4,.85 3.55,.05 5.55,.6 6.15,2.35 5.35,3.9 3.45,4.7 / S 3.45,4.7 1.5,5.5 .55,7.25 1.05,9.05 3.3,10 5.45,9.35 6.35,7.7 5.55,5.85 3.45,4.7')],
 [ # nine: oval and stem, cursive single stroke, angular bowl, curved tail
 (5,'S 5.65,1.05 4.1,.15 2.15,.4 .85,1.8 .55,3.7 1.55,5.1 3.35,5.3 4.75,4.45 5.7,2.75 5.85,1.0 / S 5.85,1.0 5.65,3.85 5.35,6.9 5.55,10'),
 (4,'S 5.95,1.35 4.15,.05 2.35,.15 .85,1.85 .65,3.85 1.8,5.1 3.55,5 5.2,3.25 6.0,.9 / S 6.0,.9 5.95,3.95 5.55,7.2 4.8,10'),
 (2,'S 5.8,1.3 4.05,0 2.05,.4 .6,2.05 .55,3.75 1.8,4.95 3.7,4.9 5.4,3.5 5.95,1.55 5.55,.75 / S 5.75,1.35 6.25,4.3 5.85,7.3 4.4,9.55 2.25,10'),
 (2,'S 4.45,.1 2.55,.7 1.0,2.3 .4,4.55 1.6,5.3 3.7,4.45 5.45,2.25 5.45,.8 4.5,.15 / L 5.55,1.45 5.15,10'),
 (1,'S 5.0,.15 3.35,0 1.75,1.1 .65,3.45 1.2,5.2 3.5,5.7 5.35,4.45 6.0,2.25 5.3,.5 / S 5.85,1.5 5.85,4.3 5.6,7.35 6.25,9.95')]
]

# Additional pen conventions: broken or overlapping joins and substantial
# variation in the relative size of the bowls. In an eight the two diagonals
# can be written continuously even when the last stroke misses the first.
STYLES[8].extend([
 (4,'S 4.7,.8 3.45,.1 2,.25 1.1,1.3 1.45,2.9 3.15,4.8 5.25,6.85 5.85,8.5 5.0,9.75 3.3,9.95 1.7,9.15 1.3,7.5 2.25,5.5 4.3,3.7 5.85,2.0 5.65,.7 4.8,.15'),
 (2,'S 4.5,.1 2.55,.15 1.65,1.2 2.0,2.65 3.55,3.95 5.7,5.95 6.3,7.85 5.5,9.4 3.45,10 1.35,9.3 .75,7.6 1.55,5.7 3.45,3.9 5.0,2.55 5.3,1.2 4.5,.1'),
 (2,'S 3.5,.2 1.95,.25 1.05,1.8 1.45,3.5 3.15,5.25 5.15,6.85 5.65,8.3 4.95,9.65 3.35,10 1.85,9.15 1.55,7.8 2.5,6.05 4.65,4.1 5.9,2.55 5.65,1.05 4.5,.45'),
 (2,'S 3.35,.2 1.8,.55 1.2,2 2.1,3.55 4.1,5.1 5.55,6.95 5.55,8.6 4.05,9.85 2.25,9.8 1.15,8.6 1.3,7.0 2.5,5.4 4.1,3.95 5.4,2.25 5.2,.9 3.9,.2 2.9,.4'),
 (2,'S 4.35,.5 3.1,.05 1.9,.65 1.65,2.05 2.6,3.85 4.5,5.85 5.55,7.75 5.1,9.3 3.6,10 2.05,9.5 1.4,8.3 1.8,6.6 3.3,4.7 4.75,2.95 5,1.45 4.35,.5'),
 (1,'S 5.65,1.55 4.55,.15 2.45,.15 1.3,1.35 1.0,2.95 2.3,4.45 4.4,5.7 5.75,7.45 5.4,9.05 3.7,9.95 1.85,9.55 .8,8.0 1.3,6.35 3.1,4.55 5.0,3.0 5.65,1.55')])
STYLES[9].extend([
 (3,'S 5.65,1.3 4.1,0 2.05,.35 .65,1.9 .5,4.15 1.75,5.4 3.6,5.25 5.2,3.5 5.65,1.3 / S 5.8,1.1 6.25,3.9 6.3,6.55 5.35,8.75 3.45,10'),
 (2,'S 5.3,1.1 4,.1 2,.65 .95,2.7 1.25,4.5 2.9,5.65 4.65,5.05 5.7,3.35 5.5,1.55 4.75,.75 / S 5.75,2.25 6.05,5.5 6.75,8.5 7.7,10'),
 (2,'S 5.75,2.55 4.4,.4 2.9,.15 1.35,1.15 .65,3.15 1.1,4.6 2.7,4.85 4.45,3.95 5.5,2.3 / L 5.65,1.35 4.95,10'),
 (1,'S 5.75,1.25 4.25,.05 2.25,.25 .7,1.55 .55,3.8 2.05,5.25 3.95,5.1 5.8,3.25 / S 5.5,.6 6.15,3.4 6.3,6.15 5.5,8.4 3.75,9.75 2.35,9.95')])
STYLES[2].extend([
 (3,'S .15,.85 1.9,.05 3.7,.65 4.8,2.2 5.0,4.55 4.05,6.9 2.45,8.75 .9,9.7 .4,9.05 1.3,8.2 3.6,8.15 5.45,8.95 7.4,9.75'),
 (2,'S .2,2.8 .8,1.15 2.65,.15 4.25,.55 5.15,1.85 5.2,3.65 4.15,5.8 2.25,8.05 .8,9.65 / S .8,9.65 2.65,9.1 5.25,9.15 7.25,8.5'),
 (1,'S .65,1.65 1.55,.25 3.15,.15 4.5,1.95 4.65,4.85 3.8,7.45 1.8,9.5 .2,9.45 .4,7.9 1.85,7.3 4.4,7.7 7.4,8.0')])

def compile_path(spec):
    subpaths=[]
    for stroke in spec.split('/'):
        kind,*coords=stroke.split()
        pts=np.array([v.split(',') for v in coords],float)
        before=np.vstack([pts[0]*2-pts[1],pts[:-2]])[:,None,:]
        after=np.vstack([pts[2:],pts[-1]*2-pts[-2]])[:,None,:]
        a,b=pts[:-1,None,:],pts[1:,None,:]
        # Approximate each spline by short line segments, then sample by length.
        t=np.arange(12)[None,:,None]/12
        if kind=='L':
            samples=a+(b-a)*t
        else:
            samples=a+0.5*t*((b-before)+t*((2*before-5*a+4*b-after)+t*(3*a-3*b+after-before)))
        samples=np.vstack([samples.reshape(-1,2),pts[-1]])
        dist=np.r_[0.,np.cumsum(np.sqrt(np.sum(np.diff(samples,axis=0)**2,axis=1)))]
        q=np.linspace(0,dist[-1],max(2,round(dist[-1]*6)+1))
        subpaths.append(np.stack([np.interp(q,dist,samples[:,i]) for i in range(2)],axis=1))
    return np.concatenate(subpaths)

rng=np.random.default_rng(261107)
# Rasterize a brush swept along an analytic path. The stamp is antialiased and
# may change its radius over the path; the cap and joint are both rounded.
oy,ox=np.mgrid[-4:5,-4:5].reshape(2,1,-1)
yy,xx=np.mgrid[:28,:28]

def write_glyph(path,k):
    u=rng.uniform
    # For a single-stroke loop the writer can lift the pen before closing
    # the curve; shorten an end of the stroke, not its central crossing.
    if k in (0,6,8) and u(0,1)<(.47 if k==8 else .14):
        end=max(1,int(len(path)*u(.024,.11)))
        path=path[end:] if (k!=6 and u(0,1)<.5) else path[:-end]
    x,y=(path-[3.5,5.]).T/5.
    x=x*u(.55 if k==8 else .67,1.34)
    # Handwriting axes, a moving waist, changes of bowl size, and smooth bends.
    dx=(u(-.58,.36)*y + u(-.18,.18)*(1-y*y) + u(-.15,.15)*np.sin(np.pi*y)
        + u(-.27,.27)*x*y + u(-.095,.095)*np.cos(np.pi*(x+y)*.85+u(-3,3)))
    dy=(u(-.24,.22)*x + u(-.24,.24)*(1-y*y) + u(-.10,.10)*np.sin(np.pi*y)
        + u(-.12,.12)*x*y + u(-.075,.075)*np.cos(np.pi*(x-y)+u(-3,3)))
    x=x+dx; y=y+dy
    # Brush diameter in final pixels. Most drawings are made with a medium pen.
    radius=u(.8,2.3) if k!=1 else u(.85,2.25)
    r=radius*(1.+u(-.19,.19)*np.sin(np.linspace(0,u(3.,9.),len(x))+u(-3.,3.)) + u(-.1,.1)*y)
    scale=(u(19.4,20.4)-2*radius)/max(np.ptp(x),np.ptp(y),.5)
    x=scale*(x-(x.min()+x.max())/2)+13.5
    y=scale*(y-(y.min()+y.max())/2)+13.5
    def draw(x,y):
        ix=np.rint(x).astype(int)[:,None]+ox; iy=np.rint(y).astype(int)[:,None]+oy
        d=(ix-x[:,None])**2+(iy-y[:,None])**2
        v=np.clip((r[:,None]+.6-np.sqrt(d))/1.2,0,1).astype(np.float32)
        idx=np.clip(iy,0,27)*28+np.clip(ix,0,27)
        buf=np.zeros(784,np.float32); np.maximum.at(buf,idx.ravel(),v.ravel()); return buf.reshape(28,28)
    im=draw(x,y); mass=im.sum()
    return draw(x+14-(xx*im).sum()/mass+u(-.65,.65),y+14-(yy*im).sum()/mass+u(-.65,.65))

# Spatial descriptors: response to edges in 8 directions (half-wave rectified),
# locally integrated by a Gaussian window; also a smoothed ink density map.
f32=lambda a: np.asarray(a,dtype=np.float32)
def gaussian(r,sigma):
    y,x=np.mgrid[-r:r+1,-r:r+1]; g=np.exp(-(x*x+y*y)/(2*sigma**2)); return g/g.sum()

y,x=np.mgrid[-2:3,-2:3]
edge_kernel=[]
for theta in np.arange(8)*np.pi/4:
    e=(np.cos(theta)*x+np.sin(theta)*y)*gaussian(2,1.05)
    edge_kernel.append(e/np.abs(e).sum()*2.)
edge_kernel=f32(edge_kernel)
window=f32(gaussian(3,1.55)); density=f32(gaussian(3,1.45))

def conv_math(a,k,s,p):
    a=np.pad(a,((0,0),(p,p),(p,p))); w=np.lib.stride_tricks.sliding_window_view(a,k.shape[-2:],axis=(1,2))[:,::s,::s]
    return np.einsum('nxyij,oij->noxy',w,k,optimize=True)

def unit(a):
    a=a.reshape(len(a),-1); a=a-a.mean(axis=1,keepdims=True); return a/np.sqrt(np.sum(a*a,axis=1,keepdims=True)+1.e-7)

def edge_map(images):
    e=np.maximum(conv_math(images,edge_kernel,1,2),0)
    return conv_math(e.reshape(-1,28,28),window[None],3,2).reshape(len(images),8,9,9)

# The banks below use soft grayscale flood filling. At a given gray level, a
# hole is a background component with no path to the boundary. Propagating
# the minimum barrier from the boundary at *all* levels simultaneously gives
# evidence even for a tiny hole, without choosing a binarization threshold.
def basins(images):
    box=lambda a: a.reshape(-1,14,2,14,2).mean(axis=(2,4))
    win=lambda a: np.lib.stride_tricks.sliding_window_view(a,(3,3),axis=(1,2))
    # Grayscale morphological closing bridges small pen-lift gaps. Blend
    # original and closed ink, retaining contrast for genuinely tiny holes.
    thick=win(np.pad(images,((0,0),(2,2),(2,2)))).max(axis=(-2,-1))
    a=.5*(box(images)+box(win(thick).min(axis=(-2,-1))))
    filled=np.ones_like(a)
    for j in range(21):
        p=np.pad(filled,((0,0),(1,1),(1,1)))
        filled=np.maximum(a,np.minimum.reduce([p[:,dy:dy+14,dx:dx+14] for dy,dx in [(1,1),(0,1),(1,0),(1,2),(2,1)]]))
    return filled-a

# Fuzzy OR on overlapping sectors; mostly insensitive to a loop's exact area.
def basin_sectors(h):
    w=np.lib.stride_tricks.sliding_window_view(h,(4,4),axis=(1,2))[:,::2,::2]
    return w.max(axis=(-2,-1)).reshape(-1,36)

# Overlapping upper/lower portions. Pairs of candidate registrations near the
# middle must agree on the same original pen path, but a crooked cap or tail
# need not have exactly the same lateral registration as the entire digit.
part_rows=[np.arange(6),np.arange(3,9)]

def partmodel(a,rows,dy,dx):
    y=rows-dy; x=np.arange(9)-dx
    b=a[:,:,np.clip(y,0,8)][:,:,:,np.clip(x,0,8)].copy()
    b[:,:,(y<0)|(y>8),:]=0; b[:,:,:,(x<0)|(x>8)]=0
    return unit(b)

def ink_desc(images):
    return unit(conv_math(images,density[None],2,2))

layers=[];weights={}
def layer(name,op,source,**kw):
    layers.append(dict(name=name,op=op,inputs=[source] if isinstance(source,str) else source,**kw));return name

def conv(name,source,k,s,p):
    k=f32(k); weights[name+'.W']=k; weights[name+'.b']=np.zeros(len(k),np.float32)
    return layer(name,'conv',source,out=len(k),kernel=list(k.shape[-2:]),stride=s,padding=p)

def dense(name,source,w,b=None):
    w=f32(w); weights[name+'.W']=w;weights[name+'.b']=np.zeros(len(w),np.float32) if b is None else f32(b)
    return layer(name,'dense',source,out=len(w))

def normalized(name,source,shape):
    a=layer(name+'_field','reshape',source,shape=[1,1,int(np.prod(shape))]);return layer(name,'norm',a)

def main(dest):
    dest=Path(dest); dest.mkdir(exist_ok=True,parents=True)
    edge=conv('ink_derivatives','image',edge_kernel[:,None],1,2)
    pos=layer('edge_polarities','relu',edge)
    # Integrate separately for each edge direction.
    edge=conv('edge_histograms',pos,np.eye(8)[:,:,None,None]*window,3,2)
    local=[]
    for p,ys in enumerate(part_rows):
        mask=np.eye(648,dtype=np.float32).reshape(8,9,9,648)[:,ys].reshape(-1,648)
        sub=dense('select_part%d'%p,edge,mask); local.append(normalized('local_contrast%d'%p,sub,[8,6,9]))
    edge=normalized('edge_contrast',edge,[8,9,9]);ink=conv('soft_ink','image',density[None,None],2,2)
    ink=normalized('ink_contrast',ink,[13,13])
    ip=conv('ink_margin','image',[[[[1]]]],1,2)
    fat=layer('swell_pen','maxpool',ip,size=[3,3],stride=[1,1]);fat=conv('invert_fat',fat,[[[[-1]]]],1,0)
    thin=layer('restore_width','maxpool',fat,size=[3,3],stride=[1,1]);thin=conv('closed_pen',thin,[[[[-1]]]],1,0)
    both=layer('two_pen_widths','add',['image',thin]);both=conv('soft_close',both,[[[[.5]]]],1,0)
    ink14=layer('small_image','avgpool',both,size=[2,2],stride=[2,2]);opposite=conv('minus_ink',ink14,[[[[-1]]]],1,0)
    # Use M=-filled, then max on the neighbours is minus their minimum.
    # Zero padding here represents unobstructed access from outside the image.
    offsets=[(1,1),(0,1),(1,0),(1,2),(2,1)]
    cross=np.zeros((5,1,3,3),np.float32)
    for j,(y,x) in enumerate(offsets): cross[j,0,y,x]=1
    m=conv('initial_fill',ink14,[[[[0]]]],1,0);weights['initial_fill.b'][:]=-1
    for j in range(21):
        n=conv('adjacent%d'%j,m,cross,1,1)
        n=layer('neighbours%d'%j,'reshape',n,shape=[1,5,196]);n=layer('escape%d'%j,'maxpool',n,size=[5,1],stride=[5,1]);n=layer('barrier%d'%j,'reshape',n,shape=[1,14,14])
        z=layer('flood_residual%d'%j,'add',[n,ink14]);z=layer('barrier_excess%d'%j,'relu',z)
        # min(n,-ink) = n - relu(n+ink).
        z=conv('neg_excess%d'%j,z,[[[[-1]]]],1,0);m=layer('inundation%d'%j,'add',[n,z])
    filled=conv('filled_basins',m,[[[[-1]]]],1,0);h=layer('hole_depth','add',[filled,opposite]);h=layer('loop_sectors','maxpool',h,size=[4,4],stride=[2,2])
    # Fuzzy intersection via 'layer cake': min(u,v) = integral from 0 to 1
    # of [u>t]*[v>t]. Use a piecewise linear quadrature with three intervals.
    q=conv('loop_levels',h,np.ones((3,1,1,1)),1,0);weights['loop_levels.b'][:]=f32([0.,-1/3,-2/3])
    q=layer('loop_level_responses','relu',q)

    groups=16; per_group=256; count=groups*per_group
    we=np.empty((10*count,648),np.float32);wi=np.empty((10*count,169),np.float32)
    wh=np.empty((10*count,108),np.float32);bh=np.empty(10*count,np.float32)
    partcount=1024; shifts=[(0,0),(-1,0),(1,0),(0,-1),(0,1)]
    wp=np.empty((2,10,partcount,len(shifts),432),np.float32);ph=np.empty((10,partcount,108),np.float32);pb=np.empty((10,partcount),np.float32)
    for k,styles in enumerate(STYLES):
        paths=[compile_path(s) for weight,s in styles]; choices=np.repeat(np.arange(len(paths)),[m for m,s in styles])
        for offset in range(0,count,32):
            images=np.stack([write_glyph(paths[rng.choice(choices)],k) for j in range(32)])
            rows=slice(k*count+offset,k*count+offset+32)
            e=edge_map(images); we[rows]=unit(e)*(.76/np.sqrt(648));wi[rows]=ink_desc(images)*(.24/13.)
            hs=basin_sectors(basins(images))
            # Minus L1 distance (up to an input-only term); exact at the
            # quadrature boundaries. Both extra and missing holes cost points.
            slopes=np.clip(hs[:,None,:]*3-np.arange(3)[None,:,None],0,1)
            # A missing loop is not a veto: an open join still often belongs
            # to a looped digit. Keep this smaller than the stroke terms.
            scale=.011
            wh[rows]=(2*scale*np.diff(slopes,axis=1,prepend=0)).reshape(32,108)
            bh[rows]=-scale*hs.sum(axis=1)
            if offset<partcount:
                sub=slice(offset,offset+32);ph[k,sub]=wh[rows];pb[k,sub]=bh[rows]
                for p,ys in enumerate(part_rows):
                    for si,(dy,dx) in enumerate(shifts):wp[p,k,sub,si]=partmodel(e,ys,dy,dx)*(.5/np.sqrt(432))
    a=dense('shape_edge_matches',edge,we); b=dense('shape_ink_matches',ink,wi);c=dense('shape_loop_matches',q,wh,bh)
    def vote(z,tag,n):
        z=layer(tag+'hypotheses','reshape',z,shape=[10,groups,n//groups]);z=layer(tag+'best_in_families','maxpool',z,size=[1,n//groups],stride=[1,n//groups])
        top=layer(tag+'best_pen','maxpool',z,size=[groups,1],stride=[groups,1]);avg=layer(tag+'consensus','avgpool',z,size=[groups,1],stride=[groups,1]);
        top=layer(tag+'best_score','reshape',top,shape=[10]);avg=dense(tag+'support_score',avg,np.eye(10)*.4)
        return layer(tag+'vote','add',[top,avg])
    whole=vote(layer('pen_agreement','add',[a,b,c]),'whole_',count)
    parts=[]
    for p in range(2):
        s=str(p);b=dense('part_metric'+s,local[p],wp[p].reshape(-1,432),np.tile([0.,-.012,-.012,-.012,-.012],10*partcount))
        b=layer('placements'+s,'reshape',b,shape=[10,partcount,len(shifts)]);b=layer('part_deform'+s,'maxpool',b,size=[1,len(shifts)],stride=[1,len(shifts)])
        parts.append(layer('part_score'+s,'reshape',b,shape=[10*partcount]))
    parts.append(dense('part_loop_matches',q,ph.reshape(-1,108),pb.ravel()))
    part=vote(layer('part_agreement','add',parts),'articulated_',partcount)
    part=dense('articulated_vote_strength',part,np.eye(10)*.85)
    layer('digit_scores','add',[whole,part])
    (dest/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(dest/'weights.npz',**weights)

if __name__=='__main__':main(sys.argv[1])
