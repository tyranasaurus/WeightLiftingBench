"""A data-free compiler for a pen-stroke shape matcher. Only needs NumPy."""
import sys
import json
from pathlib import Path
import numpy as np


# Pen programs use a coordinate frame of 10 by 10 (x rightward, y downward).
# Several pen lifts / ways of writing the same digit are represented explicitly.
# A string is one stroke; | is a pen lift. Smooth at ordinary knots, not at a
# knot marked !. The numbers here describe stroke order and are not digit data.
STYLES = [
    [ # 0: an oval, sometimes egg-shaped, with occasional overlap or gap
        (5, '5,0 8.6,1.4 10,5 8.6,8.6 5,10 1.4,8.6 0,5 1.4,1.4 5,0 7,.45'),
        (3, '6,.9 4.4,.15 1.4,2.5 0,6.3 1.3,9.4 4.7,10 8.2,8.1 9.9,4.6 9,.8 6,0 4.7,1.2'),
        (2, '7.5,2.4 5.5,0 2.6,.8 .3,4.6 .4,8.2 3.3,10 6.9,9.6 9.5,6.1 9.8,2.4 7.3,.6 5.5,.8'),
        (1, '4.4,.3 1.7,2 0,5.5 .8,8.8 4.3,10 8.5,8.4 10,5.1 9.3,1.6 6.4,0 5.3,.2'),
    ],
    [ # 1: a downstroke, a short starting pennant, or a serifed one
        (6, '5,0 5,3.7 5,7.1 5,10'),
        (3, '4.3,1.1 5,.05! 5,4.2 5,7.1 4.9,10'),
        (2, '2.8,2.6 5,0! 5,4.2 5,7.5 5,10'),
        (1.5, '4.7,0 5.2,2.5 5.1,6.2 4.6,10'),
        (.8, '1,3.6 5,0! 5,5.5 5,10'),
        (.38, '1.4,3.1 5,0! 5,4.9 5,10! | 1.6,10 5.1,9.9 8.4,10'),
        (.32, '4.9,0 5,5.3 4.8,9.8! 7,9.8'),
    ],
    [ # 2: arched head and diagonal, or a cursive foot that crosses itself
        (3, '0,2.2 2,.5 5.4,0 7.6,1.2 7.9,3.2 6,5.6 3.2,7.9 .8,10! 5.3,9.5 10,9.6'),
        (2, '.3,3.3 .4,1.6 3,0 6.2,.4 8.3,2.3 7.3,4.9 3.9,7.8 .5,9.8! 5.4,9.5 10,9.4'),
        (3, '.8,2.2 2.6,.2 5.4,.1 6.9,2.1 6.1,4.7 3.6,7.9 1.1,9.8 .4,8.8 2.3,7.9 5.2,8.6 8.3,9.7 10,9.3'),
        (1.7, '1.8,2.2 2,.7 4,0 6.8,1 7.2,3.5 5.3,6.6 2,9.5 .1,9.6 .5,8 3.7,7 7.3,7.9 10,8.7'),
        (1.3, '0,.8 3.3,0 6.8,.4 8,1.8 6.8,4.1 3.2,7.3 .7,9.6! 4.3,9.8 8.2,9.9 10,9.2'),
        (1, '1.2,3.8 1,2.1 2.7,.3 5.7,.1 8,2.1 7.2,4.7 4.5,7.2 .3,10! 3.8,10 7.1,9.2 9.8,8'),
        (.7, '0,4 1.4,1.6 3.9,.1 6.5,.3 8.6,1.8 8.5,3.4 5.6,6.4 1.8,9.9! 6.2,9.5 10,9.7'),
    ],
    [ # 3: two arcs sharing a waist; there are also flat-topped forms
        (3, '.2,1.5 3.2,.2 6.6,.1 9,1.4 8.3,3.1 4.6,4.9! 7.3,5 9.6,6.5 9.3,8.3 7.1,9.8 3.8,10 .3,8.9'),
        (2.6, '1,.8 4.6,.1 8,.5 9.4,1.8 7.7,3.5 3.1,5.1! 6.5,4.9 9.2,6 9.7,7.7 8,9.4 4.6,10 1.2,9.3 0,8.1'),
        (2, '.2,.8 3.7,.1 9,0! 3.8,4.6! 6.9,4.8 9.4,6.1 9.8,8.1 7.6,9.7 4.1,10 .2,8.8'),
        (1.5, '1.4,.4 4.9,0 8.5,.9 8.9,2.5 7.1,4 4.3,4.9! 6.8,5 9.4,6.4 9.5,8.5 7.7,9.8 4.2,10 .3,9.7'),
        (1.4, '.1,2.2 2.7,.4 6,0 8.6,1 8.6,2.7 6.4,4.5 4.2,5.1! 7.7,5.5 9.5,7.1 8.5,9.2 5.3,10 1.5,9.1 .2,7.4'),
        (.7, '.3,.5 3.5,.1 6.8,.3 8.5,1.6 7.6,3.5 4.8,5 .9,5.4 3.3,4.8 6.8,5 9.1,6.5 9.2,8.4 6.9,9.7 3.5,10 .3,9.1'),
    ],
    [ # 4: open-topped chair / fork, closed triangle, and a continuous 4
        (4, '2,0 1.1,2.6 .3,5.6! 4.9,5.8 9.6,5.6 | 7.6,.5 7.6,4.2 7.7,7.4 7.9,10'),
        (2.8, '2.1,.1 .5,3.8 .3,5.8 2.3,6.3 5.5,5.5 8.5,4.4 | 8.2,.8 7.8,4.9 7.7,7.6 7.8,10'),
        (2, '5.3,0 2.3,3 .1,5.8! 5.3,5.8 10,5.4 | 7.1,.6 7.1,4.4 7.2,7.2 7.4,10'),
        (1.6, '7.4,.2 3.9,2.9 .1,6! 5.6,6.1 10,5.9 | 7.4,0 7.3,3.9 7.3,7.2 7.5,10'),
        (1.7, '1.9,.1 .5,3.1 .3,5.3 2,5.9 5.3,5.1 7.9,2.2! 7.3,5.6 7.1,8 7.2,10'),
        (1, '1.3,0 .4,2.4 .3,4.2 2.5,5.4 5.2,5.6 8.3,5.2 | 7,1.2 7,5 7.2,8 7.6,10'),
    ],
    [ # 5: cap (often a separate up-sloping stroke), shoulder and belly
        (3.3, '9.3,.1 5.6,.1 1.8,.3! 1.4,2.1 1.1,4.5! 4.1,4 7.5,4.5 9.1,6.3 8.5,8.4 5.9,9.8 2.8,10 0,8.8'),
        (2.2, '2.1,.8 1.3,2.9 .9,4.7! 4.1,3.9 7.1,4.5 8.7,6.4 8,8.4 5.2,9.8 2.1,9.9 .1,8.6 | 1.1,1.2 4.5,1.1 7.7,.5 10,0'),
        (2.1, '1.3,1.5 .9,3.2 1.8,4.2 5.2,4.9 8.1,6.3 8.3,7.9 6.5,9.4 3.4,10 .1,9.4 | .9,.9 4.6,1.1 8.1,.6 10,0'),
        (1.2, '9.7,0 6.1,.6 2.1,.7! 1.5,2.8 1.3,5! 3.7,4.2 6.6,4.3 8.7,5.8 8.7,7.6 6.7,9.3 3.9,10 .2,9.8'),
        (1.6, '8.8,.3 5.7,.2 2.4,.3 .8,1.4 .9,3.2 3.6,4.4 7,5.5 8.8,7 8.2,8.6 5.6,9.8 2.3,10 .1,9'),
        (.9, '10,0 6.5,.5 2.1,.8! 1.2,2.2 .8,4! 3.3,4 6.5,5.1 8.2,6.8 7.7,8.7 5,10 2.1,9.8 .1,8.1'),
    ],
    [ # 6: curl from above into a lower bowl or a longer, straight entry
        (3, '7.9,0 4.8,1.1 2.2,3.8 .6,6.4 1,8.6 3.2,9.9 6.6,9.9 8.8,8.3 8.5,6.2 6.6,5.2 4,5.4 1.5,7.3'),
        (2.5, '6.2,0 4.2,2.2 2,4.8 .9,7.2 1.9,9.2 4.7,10 7.7,9.2 9.3,7.4 8.6,5.7 6,5.2 3.5,6.3 1.4,8'),
        (2, '6.6,0 5,1.6 2.5,4.6 1,7 .9,8.8 3.1,9.9 6.2,10 8.9,8.8 9.6,6.8 8.2,5.3 5.7,5.1 3.2,6.3 .8,8.6'),
        (1.3, '9.2,1 7.2,.1 4.6,.6 2.3,2.7 .9,5.4 .9,8 2.9,9.6 6,10 8.6,8.8 9.2,6.6 7.5,5.2 4.7,5.4 1,7.2'),
        (1.1, '4.1,0 2.8,2.6 1.1,5.8 1.1,8.7 3.2,9.9 6.2,9.9 8.9,8.2 9,6.2 7.3,4.9 5,5.1 2.7,6.6'),
        (.8, '7.8,0 4.1,2.1 1.6,4.9 .7,7.2 1.9,9.2 4.6,10 7.5,9.5 9,8.1 8.8,6.4 7,5.6 4.8,5.9'),
    ],
    [ # 7: bar and descending diagonal, sometimes crossed or with a flag
        (3, '0,.4 3.2,.1 6.6,.3 9.8,.6! 7.9,3.5 5.8,6.6 4,10'),
        (2.6, '0,.1 2.2,.5 5.7,.6 9.5,.2! 7.6,3.2 5.5,6.7 4.4,10'),
        (2.1, '0,1 3.3,.3 6.6,.1 9.7,0! 7.6,3.5 5.2,6.9 3.4,10'),
        (1.4, '.2,2.6 .4,.3! 3.7,.3 6.8,.4 9.5,.2! 7.3,3.9 5.3,7 3.8,10'),
        (1.5, '.2,.5 3.4,.2 6.7,.3 9.6,.3! 7.5,3.6 5.4,7 3.6,10 | 2.5,5.1 5.8,5.1 8.7,5.2'),
        (.7, '.4,1.3 2.3,.3 5.7,.1 9,.5! 7.8,3.2 6.6,6.2 6.2,10'),
    ],
    [ # 8: figure-eight in either direction, a stacked double loop, or a knot
        (3, '5,4.7 2.1,3.5 1.3,1.6 3.2,.2 6.6,.1 8.6,1.4 8,3 5.1,4.8 2,6.4 1,8.3 2.9,9.8 6.3,10 8.8,8.9 9.2,7.2 7.8,5.8 4.7,4.5'),
        (2.4, '5.8,0 2.8,.1 .9,1.7 1.8,3.5 4.6,4.9 7.5,6.3 8.6,8 7.2,9.6 4,10 1.6,9 .9,7.5 2.6,5.8 5.9,4.2 8.4,2.7 8.9,1.2 6.6,.1 4.1,.3'),
        (1.7, '3.6,.6 6.4,0 8.7,1.2 8.9,2.9 7.4,4 4.3,5.2 1.6,6.3 .7,7.9 2,9.4 5.1,10 7.8,9.2 8.5,7.6 6.8,5.8 3.8,4 1.5,2.3 1.7,.9 3.6,.2'),
        (1.6, '4.8,4.7 2.1,4 1.4,2.3 2.6,.6 5.4,0 8.1,.8 8.9,2.4 7.9,4 5,4.8 2.3,5.7 .9,7.6 2,9.4 5.1,10 8,9.1 9,7.1 7.7,5.4 4.8,4.7'),
        (1.3, '6.9,1.5 4.8,.1 2.3,.6 1.1,2.3 2.8,4.1 5.9,5.5 8.4,7.1 8.8,8.8 6.5,10 3.4,9.7 1.5,8.2 1.9,6.5 4.6,4.6 7.5,2.9 8.7,1.3 7.2,.3 5.4,.4'),
        (.8, '2.6,1.4 4.3,.2 7.1,.3 8.6,1.7 7.7,3.4 4.4,5.6 1.7,7.4 1.4,9 3.6,10 6.5,9.6 8.2,8.1 7.8,6.3 5.2,4.5 2.4,3 1.4,1.5 2.5,.3'),
    ],
    [ # 9: upper loop and stem; the stem can descend or curl left
        (3, '8.3,2.2 6.6,.5 3.9,.1 1.3,1.4 .6,3.4 2.1,4.9 4.8,5.1 7.2,3.8 8.6,1! 8.4,4.7 8.4,7.3 8.5,10'),
        (2.2, '8.3,1.7 6.4,.2 3.6,.2 1.2,1.5 .6,3.7 2.1,5.4 4.9,5.2 7.1,3.6 8.7,.8! 8.2,4.1 7.1,7.3 5.7,10'),
        (1.7, '8.1,1.2 5.6,0 2.5,.6 .7,2.2 1,4.2 3.5,5.3 6.2,4.7 8.2,2.4 8.6,.5! 8.8,4.2 8.8,7.6 9.5,10'),
        (1.4, '8.5,2 7.3,.5 4.6,.1 2,1 .7,3.1 1.8,4.7 4.5,5.2 7,4 8.5,1.2! 8.6,4.7 8.1,7.2 6.6,9.2 4.1,10'),
        (1, '8.7,1.5 6.7,.1 3.9,.1 1.3,1.6 .7,3.4 2.6,5.1 5.6,5 8.1,3.5 8.6,1.7 | 8.8,.7 8.5,3.8 8.1,7.1 7.5,10'),
        (1, '7.4,1.3 5.5,.1 3.1,.4 1.1,2.1 1,4 2.9,5.5 5.1,5.1 7.2,3.5 8.4,1.5! 8.1,4.3 7.9,7.5 8.9,10'),
        (.7, '8.2,1.2 6.2,.2 3.6,.3 1.5,1.5 1,3.3 2.7,4.8 4.9,4.8 6.8,3.8 | 8.2,.8 7.9,3.9 7.3,7.2 7.1,10'),
    ],
]


def spline(program):
    parts = []
    for stroke in program.split('|'):
        toks = stroke.split()
        p = np.array([list(map(float,t.rstrip('!').split(','))) for t in toks])
        pieces = []
        for i in range(len(p)-1):
            a,b=p[i:i+2]; h=np.linalg.norm(b-a)
            # Cardinal cubic with length-limited tangents at uneven knot spacing.
            def tangent(v):
                return v * h / max(np.linalg.norm(v), 1.e-8)
            va = tangent(b-p[max(i-1,0)]) if '!' not in toks[i] else b-a
            vb = tangent(p[min(len(p)-1,i+2)]-a) if '!' not in toks[i+1] else b-a
            t=np.linspace(0,1,max(4,int(4*h)),endpoint=False)[:,None]
            pieces.append((1-t)**2*((1+2*t)*a+t*va) + t**2*((3-2*t)*b-(1-t)*vb))
        parts.append(np.concatenate(pieces+[p[-1:]]))
    return np.concatenate(parts)


# Each call renders ONE image. The fit-to-box and translation normalize only
# that pen drawing, imitating the construction of the original input frames.
NG = 44
OFF = (NG-28)//2
R = np.arange(-4,5)
DY,DX = (v.ravel() for v in np.meshgrid(R,R,indexing='ij'))
Y,X = np.mgrid[:NG,:NG]

def render(path, digit, g):
    u,v=path.T
    # Nonuniform horizontal and vertical deformations, coherent over strokes.
    x=(u-5)/5; y=(v-5)/5
    def vary(s): return g.uniform(-s,s)
    a,b,c,d,e,f = g.uniform(-1,1,6)
    x1 = x + .18*a*(1-y*y) + .20*b*x*y + .15*c*np.sin(np.pi*y) + .10*d*np.cos(np.pi*x/2)*np.sin(np.pi*(y+e)/1.5)
    y1 = y + .22*f*(1-y*y) + vary(.16)*x + vary(.17)*x*y + vary(.10)*np.sin(np.pi*x+.6*y)
    # Simple handwriting priors: nib width in pixels, aspect, lean and roll.
    # The 1 programs encode their own slender aspect by geometry.
    aspects=(.78,.82,.87,.84,.88,.86,.80,.91,.76,.82)
    width=aspects[digit]*np.exp(vary(.44))
    xx=width*x1 - (.14+vary(.55))*y1
    yy=y1 + vary(.15)*xx
    rad = g.uniform(.78,2.1)
    scale = (g.uniform(19.6,20.5)-2*rad)/max(np.ptp(xx),np.ptp(yy))
    xx=(xx-(xx.max()+xx.min())*.5)*scale+(NG-1)*.5
    yy=(yy-(yy.max()+yy.min())*.5)*scale+(NG-1)*.5
    # Distance to a varying-width pen trajectory, antialiased at the boundary.
    r=rad*(1+vary(.19)*np.sin(np.pi*(y+.4*x)+vary(3)) +vary(.12)*x)
    ix=np.floor(xx+.5).astype(int)[:,None]+DX
    iy=np.floor(yy+.5).astype(int)[:,None]+DY
    vals=np.clip((r[:,None]+.5-np.sqrt((xx[:,None]-ix)**2+(yy[:,None]-iy)**2)),0,1)
    canvas=np.zeros(NG*NG)
    np.maximum.at(canvas, (iy*NG+ix).ravel(), vals.ravel())
    canvas=canvas.reshape(NG,NG)
    mass=canvas.sum()
    cx=(canvas*X).sum()/mass; cy=(canvas*Y).sum()/mass
    # Center of mass quantization and subpixel placement.
    sx=cx-13.65+vary(.57); sy=cy-13.65+vary(.57)
    ix=int(np.floor(sx)); iy=int(np.floor(sy)); tx=sx-ix; ty=sy-iy
    def cut(j,i): return canvas[j:j+28,i:i+28]
    im=(1-tx)*((1-ty)*cut(iy,ix)+ty*cut(iy+1,ix))+tx*((1-ty)*cut(iy,ix+1)+ty*cut(iy+1,ix+1))
    return im.astype(np.float32)


# Small analytic filter bank. Use smoother images of the local signed gradient
# channels: differentiating after Gaussian smoothing resists pixel noise;
# pooling after half-wave rectification permits small displacements of edges.
ax=np.arange(-2,3,dtype=np.float64)
ky,kx=np.meshgrid(ax,ax,indexing='ij')
ga=np.exp(-(kx*kx+ky*ky)/(2*.95**2)); ga/=ga.sum()
F = []
for a in np.arange(8)*np.pi/4:
    k=ga*(np.cos(a)*kx+np.sin(a)*ky)
    F.append(k / np.maximum(k,0).sum())
F.append(ga)
F=np.asarray(F,dtype=np.float32)
sm=np.exp(-(kx*kx+ky*ky)/(2*1.40**2)); sm=(sm/sm.sum()).astype(np.float32)

# Parameter-free network utilities
L=[]
W={}
def layer(name, op, source, **settings):
    L.append(dict(name=name,op=op,inputs=source if isinstance(source,list) else [source],**settings))
    return name

def par(name,weight,bias):
    W[name+'.W']=np.asarray(weight,np.float32)
    W[name+'.b']=np.asarray(bias,np.float32)

def conv(name,src,k,bias=None,stride=1,pad=0):
    k=np.asarray(k,np.float32)
    if bias is None: bias=np.zeros(k.shape[0])
    par(name,k,bias)
    return layer(name,'conv',src,kernel=list(k.shape[-2:]),out=k.shape[0],stride=stride,padding=pad)

def dense(name,src,w,b=None):
    if b is None: b=np.zeros(len(w),np.float32)
    par(name,w,b)
    return layer(name,'dense',src,out=len(w))

def fold(name,src,shape): return layer(name,'reshape',src,shape=shape)
def pool(name,src,size,st,mode='max'): return layer(name,mode+'pool',src,size=size,stride=st)
def relu(name,src): return layer(name,'relu',src)


def feature_graph():
    r=conv('directions','image',F[:,None],stride=2,pad=2)
    r=relu('edge_energy',r)
    # Grouped spatial smoothing using the allowed ordinary convolution.
    w=np.eye(9,dtype=np.float32)[:,:,None,None]*sm
    r=conv('cells',r,w,stride=2,pad=2) # 9 x 7 x 7
    # One global normalization for the orientation field (relative evidence for
    # different angles matters). A fixed weaker weight on luminance, which has
    # no rectification losses, keeps shading from swamping the edges.
    p=np.eye(9,dtype=np.float32); p[8,8]=.65
    r=conv('cell_scales',r,p[:,:,None,None])
    r=fold('one_field',r,[1,9*7,7]); return layer('descriptor','norm',r)


def describe(images):
    """Compute descriptor of an individual drawing, vectorized over drawings.
    In particular, no reduction here or in 'templates' mixes two drawings.
    """
    n=len(images)
    def patches(x):
        padded=np.pad(x,((0,0),)*(x.ndim-2)+((2,2),(2,2)))
        return np.lib.stride_tricks.sliding_window_view(padded,(5,5),axis=(-2,-1))[...,::2,::2,:,:]
    e=patches(images).reshape(n,14*14,25)@F.reshape(9,25).T
    e=np.maximum(0,e).transpose(0,2,1).reshape(n,9,14,14)
    q=patches(e).reshape(n,9,7,7,25)@sm.ravel()
    q[:,8]*=.65
    q=q.reshape(n,-1)
    q-=q.mean(1,keepdims=True)
    q/=np.maximum(1.e-8,np.linalg.norm(q,axis=1,keepdims=True))
    return q


def templates(src, dest):
    # A maximum within each group, then average the independent maxima. This
    # avoids relying entirely on a single pen path with a fortuitous likeness.
    ng,group=4,1024
    each=ng*group
    w=np.empty((10*each,9*7*7),np.float32)
    for d,styles in enumerate(STYLES):
        print('Pen programs for',d,flush=True)
        g=np.random.default_rng(20261001+d)
        paths=[spline(s) for _,s in styles]
        prob=np.array([p for p,_ in styles]); prob=prob/prob.sum()
        for i in range(0,each,32):
            imgs=np.stack([render(paths[g.choice(len(paths),p=prob)],d,g) for _ in range(32)])
            w[d*each+i:d*each+i+32]=describe(imgs)*(1/np.sqrt(9*7*7))
    r=dense('individual_pen_templates',src,w)
    r=fold('digit_family_member',r,[10,ng,group])
    # Tree of small max-pools (efficient with a window-stacking interpreter).
    for k in (4,4,4,4,4):
        r=pool('best_'+str(len(L)),r,[1,k],[1,k])
    r=pool('vote_of_families',r,[ng,1],[ng,1],mode='avg')
    return fold('shape_match',r,[10])


# A soft, threshold-decomposed flood fill on a half-resolution grid.
# This branch needs no templates. White space not reachable from an edge of
# the image must be enclosed by the pen. The graph propagates 4-connectivity
# over a continuum of image thresholds by min/max on real-valued intensities.

def holes_graph():
    # Valid 2x2 patches from the 28x28 input.
    s=pool('low_resolution_ink','image',[2,2],[2,2],mode='avg')
    s=conv('clearance',s,[[[[-1.]]]],bias=[1.])
    # Seed the whole boundary with its value, all other sites with zero.
    n=14; m=n*n; edge=np.zeros((n,n)); edge[[0,-1],:]=1; edge[:,[0,-1]]=1
    seed=dense('edge_of_frame',s,np.diag(edge.ravel()))
    v=fold('outside_0',seed,[1,n,n])
    # Dilation of reachability (or, equivalently, parallel widest paths).
    # min(clear,neighbor max) = clear - relu(clear - neighbor max).
    cross=np.zeros((5,1,3,3));
    for i,(y,x) in enumerate(((0,1),(1,0),(1,1),(1,2),(2,1))): cross[i,0,y,x]=1
    for i in range(15):
        a=conv('neighbors_'+str(i),v,cross,pad=1)
        a=fold('stencil_'+str(i),a,[1,5,m]); a=pool('step_'+str(i),a,[5,1],[5,1]); a=fold('arrival_'+str(i),a,[1,n,n])
        a=conv('neg_arrival_'+str(i),a,[[[[-1.]]]])
        gap=layer('not_reached_'+str(i),'add',[s,a]); h=relu('cavity_'+str(i),gap)
        if i<14:
            nh=conv('neg_cavity_'+str(i),h,[[[[-1.]]]])
            v=layer('outside_'+str(i+1),'add',[s,nh])
    # Dense piecewise-linear tests of upper/lower/full hole, using smooth
    # mass rather than pixel counts. Slight leaks in a loop give some credit.
    yy,xx=np.mgrid[:n,:n]; yy=(2*yy+.5); xx=(2*xx+.5)
    center=np.exp(-.5*((xx-13.5)/7.)**2)
    upper=center*np.clip((14.3-yy)/3.,0,1)
    lower=center*np.clip((yy-12.7)/3.,0,1)
    total=center
    # 0 and 8 enclose area, 6 has a lower bowl, 9 an upper bowl (and a
    # triangular 4 can have an upper hole too). Moderate bonuses only: a
    # closed cursive 2 is quite legitimate, as is a 6 with an open bowl.
    mw=np.stack([upper,lower,total]); masses=dense('loop_mass',h,mw.reshape(3,m))
    thresholds=np.array([1.2,1.2,1.6]); lengths=np.array([4,4,6]);
    clipped=relu('above_loop_threshold',dense('test_loop_mass',masses,np.vstack((np.eye(3),np.eye(3))),-np.r_[thresholds,thresholds+lengths]))
    mat=np.array([[.012,.012,.005],[-.008,-.008,-.005],[-.014,-.003,-.002],[-.009,-.009,-.004],
                  [.004,-.014,-.002],[-.012,-.006,-.003],[-.010,.031,.002],[-.012,-.012,-.003],
                  [.014,.014,.001],[.029,-.014,.002]])
    # Bias encodes a weak expectation of some closure only for the looped
    # digits. All values chosen a priori, in cosine-similarity units.
    bias=np.array([-.020,0,0,0,0,0,-.012,0,-.017,-.012])
    return dense('loop_evidence',clipped,np.hstack((mat/lengths,-mat/lengths)),bias)


def main(dest):
    dest.mkdir(parents=True,exist_ok=True)
    feat=feature_graph(); shape=templates(feat,dest); topology=holes_graph()
    layer('scores','add',[shape,topology])
    (dest/'network.json').write_text(json.dumps({'layers':L},indent=1)+'\n')
    np.savez(dest/'weights.npz',**W)

if __name__=='__main__': main(Path(sys.argv[1]))
