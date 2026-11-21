"""Make a digit reader from pen paths. No data, external fonts, or fitted parameters."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows


# Coordinates in a typographer's em square, x rightwards, y downwards.
# Each tuple is (relative frequency in the drawing grammar, pen trajectory).
# Multiple M commands lift the pen. Controls need not fit inside the em square.
INK = [
    [  # 0: oval, squarer oval, oval with an imperfect join
        (5, 'M .51 .02 C .18 -.04 -.01 .23 .03 .56 C .04 .91 .24 1.04 .53 .98 C .86 .93 .99 .68 .97 .40 C .95 .13 .78 -.02 .51 .02'),
        (2, 'M .57 .02 C .24 -.03 .04 .18 .04 .48 C .04 .81 .02 .99 .39 1 C .72 1.04 .93 .9 .96 .59 C 1 .29 .95 .04 .66 .02 L .48 .03'),
        (2, 'M .65 .15 C .5 -.06 .23 .04 .12 .28 C -.03 .62 .06 1.01 .43 .99 C .72 1.01 .98 .74 .97 .41 C .97 .15 .75 -.06 .47 .04'),
        (1, 'M .38 .15 C .12 .25 -.03 .62 .12 .89 C .31 1.13 .73 .95 .86 .7 C 1.06 .39 .94 .00 .67 .01 C .49 -.01 .32 .12 .29 .23'),
    ],
    [  # 1: staff with optional entry flag, hook, and foot
        (6, 'M .48 0 Q .5 .52 .5 1'),
        (2, 'M .50 0 C .54 .28 .43 .72 .54 1'),
        (3, 'M .28 .22 Q .39 .15 .52 .01 Q .49 .53 .48 1'),
        (1.6, 'M .03 .32 L .52 .01 Q .53 .46 .48 1'),
        (1, 'M .21 .22 L .51 0 L .48 .99 M .10 1 Q .46 .96 .86 .99'),
        (.7, 'M .40 .06 Q .54 -.04 .53 .06 L .47 .91 Q .47 1.01 .58 .97'),
    ],
    [  # 2: arc to diagonal to baseline, sometimes looped at the pen reversal
        (4, 'M .05 .25 C .17 .04 .42 -.05 .68 .02 C .98 .09 .94 .30 .72 .49 Q .41 .75 .05 .97 Q .45 .90 .98 .96'),
        (2, 'M .06 .32 C -.02 .08 .33 -.07 .60 .04 C .89 .16 .78 .41 .56 .62 L .11 .96 Q .52 .91 .97 1'),
        (2.2, 'M .09 .22 Q .42 -.16 .68 .09 C .95 .39 .6 .77 .22 .94 C -.09 1.07 -.01 .81 .22 .82 C .46 .83 .71 1.03 1 .91'),
        (2, 'M .04 .2 C .10 -.01 .38 -.05 .55 .13 C .76 .41 .63 .73 .30 .94 C .06 1.10 -.04 .88 .14 .79 C .38 .68 .70 .92 .99 .97'),
        (1, 'M .02 .13 C .23 -.02 .75 -.07 .86 .09 C 1.04 .31 .56 .58 .14 .87 Q .04 .97 .15 .97 L 1 .95'),
        (1, 'M .09 .1 Q .37 -.06 .52 .09 C .72 .27 .63 .57 .39 .76 Q .24 .88 .1 .95 Q .64 .84 1 .78'),
    ],
    [  # 3: paired lobes with a leftward cusp; also the angular top
        (4, 'M .04 .17 C .31 -.04 .82 -.06 .89 .12 C .99 .28 .65 .42 .35 .48 C .71 .38 1.05 .56 .93 .78 C .83 1.03 .32 1.11 .03 .86'),
        (2, 'M .03 .11 Q .43 -.04 .92 .04 Q .95 .27 .39 .48 C .76 .39 1 .55 .91 .76 Q .73 1.03 .16 .99'),
        (2, 'M .03 .09 L .94 .01 L .37 .46 C .75 .36 1.04 .5 .93 .72 C .84 .97 .31 1.14 .04 .9'),
        (1.4, 'M .11 .03 C .46 -.06 .93 .03 .83 .23 Q .72 .39 .41 .45 C 1.12 .42 1.02 .76 .8 .89 C .61 1.05 .27 1.03 .04 .93'),
        (1.6, 'M .08 .15 C .54 -.14 1.16 .02 .86 .25 C .68 .40 .14 .51 .32 .52 C .89 .42 1.01 .69 .80 .88 Q .53 1.11 .03 .94'),
        (1, 'M .04 .14 Q .42 -.03 .77 .03 Q .98 .16 .54 .42 Q .34 .52 .52 .52 C 1.03 .5 .99 .78 .72 .94 Q .43 1.07 .15 .95'),
    ],
    [  # 4: open two-stroke form, triangular form, rounded/written-through form
        (4, 'M .32 .03 Q .17 .27 .04 .53 Q .02 .64 .23 .61 L .98 .54 M .77 .03 Q .72 .58 .70 1'),
        (2.5, 'M .12 .01 Q .09 .25 .07 .49 Q .04 .63 .27 .62 Q .51 .61 .78 .44 M .81 .09 L .76 1'),
        (2.5, 'M .75 .01 L .03 .62 L 1 .58 M .75 .01 L .71 1'),
        (1.5, 'M .46 .02 L .03 .61 L .90 .58 M .72 .13 L .72 1'),
        (1.3, 'M .18 .04 C .11 .26 -.02 .51 .12 .54 Q .43 .62 .80 .19 M .82 .07 Q .72 .52 .69 1'),
        (1, 'M .23 .04 Q .04 .41 .09 .57 Q .4 .56 .82 .51 L .79 .06 L .76 .99'),
        (.8, 'M .68 .08 L .03 .58 L .97 .56 M .74 .24 Q .70 .57 .72 1'),
    ],
    [  # 5: cap, short downstroke, bowl open to the left
        (4, 'M .94 .03 Q .52 .08 .20 .05 L .10 .47 C .45 .27 .92 .35 .93 .59 C 1 .89 .53 1.14 .05 .91'),
        (2.4, 'M .90 .02 L .24 .03 Q .10 .23 .12 .41 C .47 .35 .93 .43 .90 .69 C .88 .93 .32 1.09 .02 .89'),
        (2, 'M .18 .05 Q .24 .18 .12 .40 C .44 .41 .95 .53 .86 .74 Q .63 1.08 .02 .93 M .15 .06 Q .54 .09 .98 .01'),
        (2, 'M .89 .04 Q .35 -.02 .16 .14 C -.06 .35 .25 .45 .57 .51 C 1.06 .62 .93 .88 .56 .97 Q .26 1.06 .03 .91'),
        (1.4, 'M .96 .07 L .23 .02 L .12 .37 Q .52 .34 .79 .49 C 1.08 .71 .73 .99 .22 1 L .05 .99'),
        (1, 'M .95 .01 C .65 .06 .3 .01 .22 .1 C .06 .24 .29 .37 .51 .48 Q 1.00 .7 .66 .89 Q .37 1.05 .01 .95'),
    ],
    [  # 6: hook or straight entry into bottom loop
        (4, 'M .76 .02 C .46 .12 .12 .43 .08 .68 C .02 .99 .42 1.06 .70 .93 C 1.02 .78 .95 .52 .69 .49 C .48 .45 .25 .54 .10 .72'),
        (3, 'M .80 .01 C .36 .13 -.02 .59 .12 .84 C .32 1.14 .94 .98 .94 .73 C .98 .45 .43 .43 .17 .65'),
        (2, 'M .77 .12 C .66 -.08 .38 .03 .24 .26 C .04 .56 -.01 .93 .31 .98 C .66 1.09 1.03 .89 .91 .66 C .79 .45 .41 .49 .14 .68'),
        (1.7, 'M .65 .01 Q .18 .39 .11 .71 C .05 .98 .35 1.05 .63 .96 C 1.03 .84 .97 .55 .77 .54 Q .47 .52 .17 .8'),
        (1.2, 'M .69 .01 Q .30 .19 .13 .55 Q -.01 .95 .39 .99 C .78 1.05 1.04 .85 .88 .69 Q .65 .48 .29 .65'),
        (.7, 'M .73 .02 C .36 .21 .05 .60 .15 .85 C .27 1.05 .76 1.04 .88 .82 Q 1.01 .65 .73 .55'),
    ],
    [  # 7: cap and diagonal; entry flick and crossing bar optional
        (4, 'M .03 .07 Q .48 .00 .96 .04 Q .62 .50 .39 1'),
        (2.7, 'M .04 .02 Q .50 .10 .95 .07 L .30 1'),
        (2, 'M .04 .19 L .07 .045 Q .51 -.01 .96 .05 Q .59 .49 .43 1'),
        (1.5, 'M .04 .035 Q .47 .02 .96 .055 L .34 1 M .24 .51 L .82 .50'),
        (1.2, 'M .03 .11 Q .15 -.05 .48 .04 Q .75 .12 .95 .07 Q .73 .34 .56 .65 L .48 1'),
        (1, 'M .03 .02 L .93 .04 L .30 1 M .32 .53 Q .57 .50 .75 .53'),
    ],
    [  # 8: figure eight in both directions; top lobe can be tiny, open, or broad
        (4, 'M .48 .48 C .13 .34 .09 .15 .32 .055 C .60 -.08 .98 .06 .86 .25 C .71 .43 .37 .52 .16 .71 C -.09 .98 .39 1.1 .72 .94 C 1.10 .75 .79 .59 .48 .48'),
        (3, 'M .47 .48 C .76 .37 .96 .18 .75 .05 C .47 -.10 .07 .06 .15 .26 C .25 .45 .74 .57 .86 .76 C 1 .98 .63 1.07 .32 .96 C -.03 .85 .02 .65 .47 .48'),
        (1.8, 'M .50 .44 C .15 .42 .01 .19 .25 .06 C .52 -.09 .91 .06 .84 .25 Q .77 .4 .48 .49 C .06 .63 -.04 .92 .31 .99 C .68 1.1 1.03 .9 .86 .69 Q .78 .54 .50 .44'),
        (1.6, 'M .63 .10 C .39 -.09 .08 .09 .17 .26 C .24 .43 .76 .59 .90 .78 C 1.05 1.05 .50 1.05 .26 .92 C .0 .78 .22 .58 .47 .43 C .86 .21 .88 .05 .58 .02 Q .43 .0 .30 .08'),
        (1.3, 'M .65 .03 C .30 -.05 -.05 .17 .20 .34 C .43 .51 .84 .61 .89 .83 C .94 1.06 .41 1.04 .21 .90 C -.04 .73 .36 .47 .55 .36 Q .86 .15 .68 .05'),
        (1, 'M .53 .46 C .17 .40 .13 .23 .22 .10 C .41 -.08 .82 .0 .82 .18 Q .82 .36 .49 .47 C .87 .46 1.01 .77 .78 .93 C .53 1.1 .13 1.0 .09 .81 Q .04 .57 .53 .46'),
    ],
    [  # 9: bowl at cap height, then a staff or a sweeping descender
        (4, 'M .81 .19 C .73 -.02 .41 -.05 .19 .12 C -.06 .31 .04 .52 .30 .51 Q .61 .54 .84 .12 Q .78 .54 .68 1'),
        (2.7, 'M .83 .14 C .57 -.11 .13 .04 .07 .28 C -.02 .52 .33 .58 .59 .43 Q .78 .32 .86 .10 Q .84 .60 .51 1'),
        (2, 'M .82 .19 C .67 -.07 .31 -.01 .13 .17 C -.05 .41 .18 .58 .42 .50 Q .70 .42 .80 .19 Q .76 .60 .93 .99'),
        (1.7, 'M .83 .22 C .67 -.02 .24 -.06 .09 .19 C -.09 .47 .21 .58 .51 .43 Q .76 .33 .88 .14 C .93 .43 .73 .81 .43 .96 Q .29 1.04 .16 .95'),
        (1.5, 'M .78 .15 C .66 -.08 .22 -.01 .09 .26 C .00 .50 .21 .62 .44 .53 Q .7 .42 .79 .14 Q .74 .58 .74 .98'),
        (1, 'M .75 .08 C .43 -.05 .10 .14 .09 .36 C .04 .60 .36 .56 .55 .40 L .82 .09 L .63 1'),
        (.65, 'M .83 .20 C .58 -.07 .13 .0 .08 .22 C .01 .46 .28 .51 .54 .35 Q .73 .26 .80 .14 Q 1 .54 .69 .90 Q .50 1.08 .24 .93'),
    ],
]


def parse_path(s):
    words = iter(s.split())
    def xy():
        return np.array([float(next(words)), float(next(words))])
    t = np.arange(1, 33)[:, None] / 32.
    u = 1-t
    strokes = []
    for op in words:
        if op == 'M':
            a = xy(); strokes.append([a[None, :]])
        elif op == 'L':
            b = xy(); strokes[-1].append(u*a+t*b); a = b
        elif op == 'Q':
            b, c = xy(), xy(); strokes[-1].append(u*u*a+2*u*t*b+t*t*c); a = c
        elif op == 'C':
            b, c, d = xy(), xy(), xy()
            strokes[-1].append(u**3*a+3*u*u*t*b+3*u*t*t*c+t**3*d); a = d
        else:
            raise ValueError(op)
    return [np.concatenate(p) for p in strokes]


def sample_spacing(p, step):
    if len(p) == 1: return p
    s = np.r_[0, np.cumsum(np.sqrt(np.sum(np.diff(p, axis=0)**2, axis=1)))]
    q = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1]/step))+1))
    return np.stack([np.interp(q, s, p[:, n]) for n in (0, 1)], axis=1)


# Smooth deformation modes. The coefficients below specify a handwriting prior,
# not measurements of images. The same map is applied at intersecting strokes.
def distort(p, a, b):
    x, y = p.T; X, Y = x-.5, y-.5
    arch = (1-4*Y**2); bulge = 1-4*X**2
    dx = a[0]*arch + a[1]*arch*2*X + a[2]*4*X*Y + a[3]*np.sin(2*np.pi*y) + a[4]*bulge
    dy = b[0]*arch + b[1]*arch*2*X + b[2]*4*X*Y + b[3]*np.sin(2*np.pi*x) + b[4]*bulge
    return p + np.stack([dx,dy], axis=1)


# A soft circular pen. The line is evenly sampled at less than half a pixel,
# and rasterized by taking the union (max coverage) of the pen footprints.
gy, gx = np.indices((28,28))
gy44, gx44 = np.indices((44,44))
offy, offx = np.mgrid[-5:6, -5:6].reshape(2,-1)


def render(rng, k, strokes):
    z = lambda: np.clip(rng.normal(size=5),-2.1,2.1)
    amt = rng.uniform(.35, 1.3)
    a = z()*[.085,.13,.092,.027,.043]*amt
    b = z()*[.12,.093,.053,.020,.085]*amt
    raw = np.concatenate(strokes)
    p = distort(raw,a,b)
    # The diagonal of a 1 is chiefly controlled by shear, not stem width.
    width = rng.triangular(.32, .68, 1.24) if k != 1 else rng.uniform(.46,.97)
    if k == 0: width *= rng.uniform(.73,1.1)
    p[:,0] *= width
    angle = np.clip(rng.normal(0,.12),-.32,.32)
    shear = rng.triangular(-.87, -.16, .62)
    x,y = p.T.copy()
    p[:,0] = x + (shear-angle)*y
    p[:,1] = y + angle*x
    # Write in a 20px bounding box including the pen width, preserve aspect.
    diameter = rng.triangular(1.22, 2.65, 5.30)
    p -= p.min(0)
    p *= (20-diameter)/max(p.max(0))
    p += 10
    start = 0
    parts = []
    for st in strokes:
        stop = start + len(st)
        parts.append(sample_spacing(p[start:stop],.40))
        start = stop
    p = np.concatenate(parts)
    # Per-stroke pressure can progress or taper at either end.
    phase = rng.uniform(0, 2*np.pi)
    amp = rng.uniform(.04,.23)
    ends = np.concatenate([np.sin(np.linspace(0,np.pi,len(s))) for s in parts])
    rad = diameter/2 * (1 + amp*np.sin(.19*p[:,0]+.24*p[:,1]+phase) - rng.uniform(0,.19)*(1-ends)**5)
    ix = np.floor(p[:,0]).astype(int)[:,None] + offx
    iy = np.floor(p[:,1]).astype(int)[:,None] + offy
    dist = np.sqrt((ix-p[:,0,None])**2 + (iy-p[:,1,None])**2)
    raster = np.zeros((44,44), dtype=np.float32)
    np.maximum.at(raster.ravel(), (44*iy+ix).ravel(), np.clip(rad[:,None]+.5-dist,0,1).astype('f').ravel())
    # Translate by the individual drawing's own ink centroid, as in the task.
    m = raster.sum(dtype='d')
    cx = (raster*gx44).sum()/m; cy = (raster*gy44).sum()/m
    tgt = rng.uniform(-.58,.58,size=2) + 13.65
    dx, dy = cx-tgt[0], cy-tgt[1]
    i,j = int(np.floor(dx)), int(np.floor(dy)); f,g = dx-i, dy-j
    def shift(d,e):
        return raster[(gy+j+e).clip(0,43),(gx+i+d).clip(0,43)]
    return ((1-f)*(1-g)*shift(0,0)+f*(1-g)*shift(1,0)+(1-f)*g*shift(0,1)+f*g*shift(1,1)).astype('f')


# Network construction utilities. All feature extraction weights are analytic.
layers = []; weights = {}
def layer(name, op, inputs, **kw):
    layers.append(dict(name=name, op=op, inputs=inputs if isinstance(inputs,list) else [inputs], **kw))
    return name

def conv(name, inp, w, b=None, st=1, pad=0):
    w = np.asarray(w,np.float32); n,_,k,_ = w.shape
    weights[name+'.W'] = w
    weights[name+'.b'] = np.zeros(n,'f') if b is None else np.asarray(b,'f')
    return layer(name,'conv',inp,out=n,kernel=[k,k],stride=st,padding=pad)

def pool(name,inp,h,w,sy,sx,op='maxpool'):
    return layer(name,op,inp,size=[h,w],stride=[sy,sx])

def reshape(name,inp,*shape): return layer(name,'reshape',inp,shape=list(shape))
def plus(name,*inputs): return layer(name,'add',list(inputs))


def conv_np(x, w, st=1, pad=0):
    # Build-time feature map of ONE rendering -- no dataset or batch moments.
    if pad: x = np.pad(x,((0,0),(pad,pad),(pad,pad)))
    n,c,k,_ = w.shape
    cols = windows(x,(k,k),axis=(1,2))[:,::st,::st].transpose(1,2,0,3,4)
    h,u = cols.shape[:2]
    return (cols.reshape(h*u,-1) @ w.reshape(n,-1).T).T.reshape(n,h,u)


def unit(v):
    return v / max(1e-8, float(np.sqrt(np.sum(v*v, dtype=np.float64))))


def gaussian(k,s):
    y,x = np.mgrid[:k,:k] - (k-1)*.5
    w = np.exp(-.5*(x*x+y*y)/(s*s)); return w/w.sum()


def template_reader():
    # Six half-wave rectified Gaussian derivatives, three even ridge detectors,
    # and a low-frequency ink channel. Positive and negative edges are separate.
    y,x = np.mgrid[-3:4,-3:4]
    kernels = []
    for a in np.arange(6)*np.pi/3:
        n = np.cos(a)*x + np.sin(a)*y
        t = -np.sin(a)*x + np.cos(a)*y
        w = -n*np.exp(-.5*((n/1.04)**2+(t/1.65)**2))
        w -= w.mean(); w /= (.5*abs(w).sum()); kernels.append(w)
    for a in np.arange(3)*np.pi/3:
        n = np.cos(a)*x + np.sin(a)*y
        t = -np.sin(a)*x + np.cos(a)*y
        w = (1-(n/1.32)**2)*np.exp(-.5*((n/1.32)**2+(t/1.95)**2))
        w -= w.mean(); w /= (.5*abs(w).sum()); kernels.append(w*.84)
    kernels.append(gaussian(7,1.13)*1.5)
    w = np.asarray(kernels,np.float32)[:,None]
    e = layer('edge_energy','relu',conv('edge_and_ridge','image',w,pad=3))
    # Integrate in overlapping spatial cells, sampled every four pixels.
    g = gaussian(8,1.93).astype('f')
    intw = np.eye(10,dtype='f')[:,:,None,None]*g
    f = conv('spatial_histograms',e,intw,st=4,pad=2)
    f = reshape('feature_vector',f,1,10,49)
    f = layer('standardized_histograms','norm',f)
    size = 4096  # number of individual renderings for each symbol
    W = np.empty((10,size,490),'f')
    rng = np.random.Generator(np.random.PCG64(709203))
    for k,grammar in enumerate(INK):
        prior = np.array([r for r,_ in grammar]); prior = prior/prior.sum()
        styles = [parse_path(s) for _,s in grammar]
        for i in range(size):
            pic = render(rng,k,styles[rng.choice(len(styles),p=prior)])
            act = np.maximum(conv_np(pic[None],w,pad=3),0)
            hh = windows(np.pad(act,((0,0),(2,2),(2,2))), (8,8),axis=(1,2))[:,::4,::4]
            hist = np.einsum('cyxij,ij->cyx',hh,g).ravel()
            hist -= hist.mean(); W[k,i] = unit(hist) / math.sqrt(490)
        print('Wrote strokes for',k,flush=True)
    nm = 'individual_stroke_templates'
    weights[nm+'.W'] = W.reshape(10*size,-1)
    weights[nm+'.b'] = np.zeros(10*size,'f')
    corr = layer(nm,'dense',f,out=10*size)
    # Max within groups and then combine the very best match with a few
    # independent best matches. No template-to-template comparison is needed.
    corr = reshape('style_groups',corr,10,8,size//8)
    corr = pool('best_in_group',corr,1,size//8,1,size//8)
    best = pool('best_overall',corr,8,1,8,1)
    consensus = pool('mean_of_group_bests',corr,8,1,8,1,'avgpool')
    return conv('group_vote',plus('template_evidence',best,consensus),np.eye(10)[:,:,None,None]*.5)


def topology_reader():
    # Compute soft ink barriers on a 14x14 lattice. Geodesic dilation on the
    # free space, seeded only from its border, distinguishes a bounded cavity
    # from an ordinary gap between strokes. Cycling horizontal and vertical
    # propagation prevents the flood from jumping across a diagonal pen line.
    barrier_k = gaussian(4,.94).reshape(1,1,4,4)
    m = conv('coarse_ink','image',barrier_k,st=2,pad=1)
    o = layer('barrier_start','relu',conv('barrier_ramp',m,[[[[7]]]],[-.63]))
    excess = layer('barrier_excess','relu',conv('above_one',o,[[[[1]]]],[-1]))
    o = plus('barrier',o,conv('remove_excess',excess,[[[[-1]]]]))
    free = conv('free_space',o,[[[[-1]]]], [1])
    neg_o = conv('negative_barrier',o,[[[[-1]]]])
    # Reshape a single broadcast affine bias into a fixed border seed mask.
    name = 'seed_mask'; M = np.zeros((14,14),'f'); M[[0,-1],:]=1; M[:,[0,-1]]=1
    one = pool('field_mean',m,14,14,14,14,'avgpool')
    layer(name,'dense',one,out=196); weights[name+'.W'] = np.zeros((196,1),'f'); weights[name+'.b'] = M.ravel()
    v = reshape('border',name,1,14,14)
    for i in range(16):
        # Extract zero-padded horizontal or vertical triples into three lanes,
        # then pool over those lanes. Alternate axes (4-connected flood).
        w = np.zeros((3,1,3,3),'f')
        if i%2: w[:,0,:,1] = np.eye(3)
        else: w[:,0,1,:] = np.eye(3)
        n = str(i)
        # Soft flood update: ReLU(max over neighbors - ink barrier).
        shifted = conv('flood_shifts_'+n,v,w,pad=1)
        rows = reshape('flood_planes_'+n,shifted,1,3,196)
        hi = pool('flood_max_'+n,rows,3,1,3,1)
        hi = reshape('flood_neighborhood_'+n,hi,1,14,14)
        v = layer('reachable_'+n,'relu',plus('flood_step_'+n,hi,neg_o))
    hole = layer('cavities','relu',plus('unreached_space',free,conv('negative_exterior',v,[[[[-1]]]])))
    # A suppressed (non-binary) response on the ink fringe is not evidence
    # for a hole. Only the high-confidence white interior votes.
    hole = layer('clear_interiors','relu',conv('interior_gate',hole,[[[[2.5]]]],[-1.5]))
    yy,xx = np.mgrid[:14,:14] * 2 + .5
    # Total, upper, and lower enclosed area. Lower bowls suggest 6,
    # while upper bowls suggest 9.
    top = np.clip((15-yy)/5,0,1); bot = np.clip((yy-12)/5,0,1)
    base = [np.ones_like(top),top,bot]
    n = 'cavity_areas'; weights[n+'.W']=np.asarray(base,'f').reshape(3,196); weights[n+'.b']=np.zeros(3,'f')
    h = layer(n,'dense',hole,out=3)
    # Capped linear signals for the three kinds of evidence. Do not punish
    # broken loops strongly -- 6s, 8s, and 9s are often open in handwriting.
    n='cavity_ramps'; weights[n+'.W']=np.vstack([np.eye(3),np.eye(3)]).astype('f')*.48
    weights[n+'.b']=np.array([-.24]*3+[-1.24]*3,'f')
    h=layer('positive_areas','relu',layer(n,'dense',h,out=6))
    # If there is a hole, strongly favor the classes normally drawn with
    # one, and modestly favor the ones putting it on the correct side.
    # Zero: one centered loop. Two and four: occasionally a small loop.
    table=np.array([[.033,.005,.005],[-.042,0,0],[-.017,-.014,.008],[-.037,0,0],
                    [.002,.012,-.021],[-.028,0,.003],[.020,-.022,.031],[-.043,0,0],
                    [.025,.008,.008],[.015,.034,-.024]],'f')
    n='cavity_votes'; weights[n+'.W']=np.hstack([table,-table]); weights[n+'.b']=np.array([-.026,0,.003,.006,.001,.004,-.013,0,-.017,-.012],'f')
    return reshape('structural_scores',layer(n,'dense',h,out=10),10,1,1)


def main(folder):
    app = template_reader(); topology = topology_reader()
    reshape('digit_scores',plus('combine_evidence',app,topology),10)
    folder.mkdir(exist_ok=True,parents=True)
    (folder/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
    np.savez(folder/'weights.npz',**weights)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
