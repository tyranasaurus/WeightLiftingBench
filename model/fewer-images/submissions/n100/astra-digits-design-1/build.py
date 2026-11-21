"""Build a digit recognizer from a pen-path grammar; no input files.

Coordinates in the grammar are in an arbitrary writing space, nominally
-5..5 across and -8..8 down. Uppercase commands are move, line, quadratic
and cubic Beziers. A semicolon lifts the pen. The paths and style recipes
are hand specified. Each call to draw creates exactly one matching template.
"""
import sys
import json
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows

F = np.float32
rng = np.random.default_rng(913704)

# A style's integer multiplicity describes how to draw the class, not a vote
# or a measured prior. All ten classes receive the same number of templates.
PENS = [
 [ # 0: oval, a little overshoot at closure, eggs and a cursive join
  (6,'M 1 -8 C -2 -9 -5 -5 -5 0 C -6 5 -4 8 -1 8 C 3 9 5 5 5 0 C 6 -5 4 -8 1 -8'),
  (3,'M 3 -6 C 1 -9 -3 -8 -4 -4 C -7 2 -5 8 -1 8 C 3 9 5 4 5 -2 Q 5 -9 1 -8 Q -1 -8 -2 -6'),
  (2,'M 0 -8 C -3 -8 -5 -4 -5 1 Q -5 8 -1 8 C 3 8 5 5 5 1 Q 5 -5 1 -8 L 0 -8'),
  (2,'M 2 -7 Q -1 -10 -4 -4 C -6 0 -5 8 -2 8 C 2 9 5 5 5 0 Q 5 -8 1 -7 L 0 -5'),
  (1,'M 0 -8 Q -5 -7 -5 -1 L -5 4 Q -5 8 -1 8 Q 4 9 5 4 L 5 -3 Q 5 -8 0 -8'),
 ],
 [ # 1: single downstroke, entry tick, an angled flag, and the serifed form
  (6,'M 0 -8 Q 0.3 0 0 8'),
  (3,'M 0.4 -8 Q -0.8 -1 -0.3 4 Q -0.3 6 0.2 8'),
  (2,'M -1.4 -6 Q -0.5 -6.8 0 -8 Q 0.4 -1 0 8'),
  (2,'M -3 -4 L 0 -8 Q 0 0 0 8'),
  (1,'M -4 -3 Q -2 -5 0 -8 L 0 8 ; M -3 8 L 3 8'),
  (1,'M -2 -6 L 0 -8 L 0 8 Q 1 8 2 7.8'),
 ],
 [ # 2: curled crown into an oblique downstroke, then a foot or a loop
  (5,'M -5 -4 C -3 -9 3 -9 4 -6 C 6 -2 0 2 -5 7 Q -5.5 8 -4 8 Q 0 7 5 8'),
  (3,'M -5 -5 Q -3 -9 0 -8 Q 6 -8 4 -3 Q 3 0 -5 7 L -5 8 Q 0 7 5 7'),
  (3,'M -4 -3 C -6 -7 -2 -9 1 -8 C 7 -6 3 -1 -1 3 Q -7 10 -5 8 Q -1 6 5 8'),
  (2,'M -4 -5 Q -5 -8 -1 -8 Q 4 -9 4 -4 Q 4 -1 0 3 C -6 9 -7 9 -5 6 Q -4 4 0 6 Q 3 8 5 7'),
  (2,'M -5 -5 Q -3 -8 0 -8 Q 5 -9 4 -5 Q 4 -1 -2 5 Q -7 10 -6 6 C -5 3 -1 6 2 7 L 6 7'),
  (2,'M -5 -4 Q 0 -10 3 -7 Q 7 -4 1 1 L -5 7 Q -1 8 5 6'),
  (1,'M -4 -7 Q -1 -9 2 -7 Q 5 -5 3 -2 L -5 7 Q -4 8 5 8'),
  (1,'M -4 -4 C -7 -9 0 -9 3 -6 Q 6 -3 0 2 L -5 8 Q 0 8 5 5'),
 ],
 [ # 3: open lobes, angular upper bowls, small retraced cusp at the waist
  (5,'M -4 -6 C 0 -9 5 -8 4 -5 Q 4 -2 -1 -0.5 C 2 -2 6 1 4 5 Q 2 10 -5 7'),
  (3,'M -4 -6 Q 0 -9 3 -7 C 6 -4 2 -1 -2 0 Q 3 -1 4 2 C 6 7 1 10 -4 6'),
  (2,'M -5 -7 Q 0 -8 4 -7 Q 2 -4 -2 0 C 1 -1 5 0 5 4 Q 4 10 -5 7'),
  (2,'M -5 -6 Q 0 -9 4 -7 L -1 0 Q 5 -1 5 3 C 6 8 0 10 -4 7'),
  (2,'M -4 -7 C 2 -10 7 -6 2 -2 Q -3 1 -2 -0.3 C 0 -2 5 -1 4 4 Q 3 10 -5 7'),
  (2,'M -4 -5 Q 0 -10 3 -7 C 5 -4 2 -1 -1 0 Q 4 -1 4 3 Q 5 8 0 8 Q -2 8 -4 5'),
  (1,'M -4 -7 L 4 -7 L -2 -0.5 Q 4 -1 4 3 Q 5 9 -3 8 L -5 6'),
 ],
 [ # 4: open and closed, angular and rounded; long and short crossbars
  (4,'M -3 -7 L -5 1 Q -5 2 -4 2 L 5 1 ; M 3 -8 Q 2 0 2 8'),
  (4,'M -3 -7 Q -4 -4 -5 1 Q -4 4 3 0 ; M 3 -7 L 2 8'),
  (3,'M -3 -7 Q -6 -1 -4 1 Q -2 3 3 -1 ; M 3 -7 Q 2 0 2 8'),
  (3,'M -4 -7 L -5 1 L 3 0 L 3 -7 L 3 8'),
  (2,'M 2 -8 L -5 1 L 5 1 ; M 2 -8 L 2 8'),
  (2,'M 3 -8 L -5 2 Q -1 2 5 1 ; M 3 -6 Q 2 0 2 8'),
  (2,'M -2 -8 L -5 1 L 2 1 ; M 2 -4 L 2 8'),
  (1,'M -4 -7 Q -6 4 -3 2 L 3 -2 L 4 -7 Q 2 1 2 8'),
 ],
 [ # 5: left stem, a bowl pointing right, and a separately written cap
  (5,'M 5 -8 Q 1 -7 -3 -8 L -4 -1 C 0 -4 5 -1 4 3 Q 4 10 -4 7'),
  (3,'M -3 -7 L -4 -1 Q 1 -3 4 1 C 7 6 1 10 -5 6 ; M -3 -7 Q 0 -7 5 -8'),
  (2,'M 5 -8 L -3 -7 Q -5 -4 -4 0 Q -2 -2 1 -1 C 7 1 5 8 -1 8 Q -4 8 -5 6'),
  (3,'M -3 -7 Q -4 -4 -4 -2 C -3 -1 2 -1 4 2 C 7 7 0 10 -4 7 ; M -4 -7 Q 1 -6 5 -8'),
  (2,'M 5 -7 Q 0 -8 -3 -7 Q -6 -5 -4 -2 Q -2 0 1 1 C 8 5 1 10 -5 7'),
  (2,'M 4 -8 Q 0 -7 -3 -7 Q -5 -7 -4 -4 Q -4 -1 0 0 Q 6 1 4 5 Q 2 10 -5 7'),
  (2,'M -3 -7 L -4 -1 Q 2 -2 4 2 Q 6 8 -3 8 L -5 7 ; M -2 -7 L 5 -8'),
  (1,'M 3 -7 Q -2 -9 -4 -6 Q -7 -2 0 0 C 8 3 4 10 -4 7'),
 ],
 [ # 6: a downstroke curving into a lower loop, sometimes a long top hook
  (6,'M 3 -8 C -1 -7 -5 -1 -5 4 Q -5 9 0 8 C 5 8 6 1 3 0 Q -1 -2 -4 3'),
  (4,'M 2 -8 Q -4 -4 -4 2 C -5 7 -2 9 1 8 Q 6 7 4 2 Q 2 -2 -3 1 L -4 3'),
  (3,'M 4 -7 Q 2 -10 -1 -6 C -5 -1 -6 6 -2 8 Q 3 10 5 5 C 7 0 1 -3 -4 2'),
  (2,'M 2 -8 Q -2 -4 -4 1 Q -7 7 -1 8 Q 4 10 4 5 Q 4 -1 1 0 Q -2 0 -4 4'),
  (2,'M 1 -8 Q -5 -3 -4 3 Q -4 9 1 8 Q 6 7 4 2 Q 2 -2 -1 0 Q -4 2 -3 5'),
  (1,'M 3 -8 C -2 -6 -7 4 -4 7 Q -1 10 3 7 Q 6 2 2 0 Q 0 -1 -2 1'),
 ],
 [ # 7: top bar and descending slash, bent crowns or a European crossbar
  (6,'M -5 -7 Q -1 -8 5 -7 Q 1 -1 -2 8'),
  (4,'M -5 -8 Q 0 -6 5 -7 L -2 8'),
  (3,'M -5 -5 L -5 -7 Q -2 -8 5 -7 Q 1 -1 -2 8'),
  (3,'M -5 -7 Q 0 -8 5 -7 Q 2 -2 -2 8 ; M -2 0 L 4 0'),
  (2,'M -4 -7 L 4 -7 L -2 8 ; M -2 0 L 3 -1'),
  (2,'M -5 -6 Q -4 -8 -2 -7 Q 2 -6 5 -7 L -1 8'),
  (1,'M -5 -4 L -4 -8 L 5 -7 Q 1 -1 -2 8 ; M -2 0 L 4 1'),
 ],
 [ # 8: both directions of a figure eight, stacked loops and open joints
  (5,'M 2 -7 C -1 -10 -6 -6 -3 -3 C -2 -1 3 1 4 4 C 6 9 -3 10 -4 6 C -6 3 -1 0 1 -2 C 6 -6 4 -8 1 -8'),
  (4,'M 0 -8 C -5 -8 -5 -3 -1 0 C 3 2 6 5 3 7 C -1 11 -7 7 -4 3 Q -3 1 0 -1 C 6 -5 5 -8 0 -8'),
  (3,'M 1 -8 C 5 -8 6 -3 1 -1 C -4 1 -6 5 -3 7 C 0 10 6 7 4 3 Q 3 1 -1 -1 C -6 -4 -4 -8 1 -8'),
  (3,'M -1 0 C -6 -2 -5 -8 0 -8 C 5 -8 6 -3 1 -1 L -1 0 C -5 2 -6 8 0 8 C 5 9 6 3 2 1 L -1 0'),
  (2,'M 2 -6 C 0 -10 -5 -7 -4 -4 C -4 -1 2 0 4 3 Q 7 9 0 8 C -7 8 -5 3 -2 1 C 0 -1 5 -4 3 -7 L 2 -8'),
  (2,'M 0 -8 C -5 -8 -6 -3 -1 0 C 5 3 6 7 1 8 C -5 10 -7 4 -2 1 Q 5 -3 4 -6 Q 3 -9 -1 -7'),
  (1,'M 1 -8 C -4 -9 -6 -4 -2 -1 Q 2 2 4 4 C 6 9 -4 10 -4 6 Q -5 3 0 0 Q 5 -4 3 -7'),
 ],
 [ # 9: a top loop and descender. The bottom can be straight or curled.
  (5,'M 3 -6 C 1 -9 -3 -8 -4 -5 Q -6 0 -2 1 Q 1 2 4 -5 L 4 -7 Q 3 0 2 8'),
  (4,'M 4 -6 Q 1 -10 -3 -6 Q -7 -1 -3 0 Q 1 2 4 -4 ; M 4 -7 Q 3 0 2 8'),
  (3,'M 4 -6 C 1 -9 -3 -9 -4 -4 Q -5 2 0 0 Q 4 -2 4 -6 Q 4 0 2 8'),
  (2,'M 3 -6 Q -1 -9 -4 -5 C -7 1 -1 4 3 -2 L 4 -6 Q 4 0 2 5 Q 1 9 -2 8'),
  (2,'M 3 -6 C -1 -11 -7 -5 -4 -1 Q -1 2 4 -3 Q 4 4 0 8'),
  (2,'M 4 -7 Q 0 -9 -3 -6 Q -6 -2 -3 0 Q 0 2 4 -4 L 4 -7 L 3 7 Q 3 8 4 8'),
  (1,'M 3 -8 C -1 -9 -5 -5 -4 -2 Q -3 2 1 0 L 4 -5 Q 4 3 1 7 Q -1 9 -4 7'),
 ]]


def compile_path(s):
    t = np.linspace(0, 1, 57, dtype=np.float64)[:, None]
    result = []
    for part in s.split(';'):
        words = iter(part.split()); segments = []; progress = 0
        for cmd in words:
            pts = np.array([[float(next(words)), float(next(words))]
                            for j in range({'M':1,'L':1,'Q':2,'C':3}[cmd])])
            if cmd == 'M':
                cur = pts[0]; continue
            if cmd == 'L': xy = (1-t)*cur + t*pts[0]
            if cmd == 'Q': xy = (1-t)**2*cur + 2*(1-t)*t*pts[0] + t*t*pts[1]
            if cmd == 'C': xy = (1-t)**3*cur + 3*(1-t)**2*t*pts[0] + 3*(1-t)*t*t*pts[1] + t**3*pts[2]
            segments.append(xy if not progress else xy[1:]); cur = pts[-1]; progress += 1
        xy = np.concatenate(segments)
        d = np.r_[0, np.cumsum(np.sqrt(np.sum(np.diff(xy,axis=0)**2,axis=1)))]
        u = np.linspace(0,d[-1], max(4,int(d[-1]/.15)+1))
        result.append(np.stack([np.interp(u,d,xy[:,j]) for j in (0,1)],axis=1))
    return result


STYLES = [[(n,compile_path(s)) for n,s in ps] for ps in PENS]
WRITING_WIDTH = [1.00,1.00,1.06,1.06,1.10,1.08,.98,1.12,1.00,1.03]
Y,X = np.mgrid[:28,:28].astype(float)


def draw(digit,i):
    """Independently warp one pen trajectory and paint round, variable-width ink."""
    opts = STYLES[digit]; which = i % sum(n for n,_ in opts)
    for n,shape in opts:
        if which < n: break
        which -= n
    points = np.concatenate(shape).copy(); x = points[:,0]; y = points[:,1]
    U = rng.uniform; N = rng.normal
    waist = U(.27,.73) if digit in (3,8) else U(.37,.65)
    mid = U(-.9,.9)
    upper = np.clip((-y+3)/6,0,1)
    if digit in (2,3,5,8):
        x *= np.exp(upper*U(-.28,.23) + (1-upper)*U(-.16,.18))
    if digit in (4,9,6):
        # Vary the size of the bowl relative to the stem, about the junction.
        waist = U(.33,.68)
    y0=y.copy(); y[:] = np.where(y0<mid,(y0+8)*waist*2-8,(y0-mid)*(1-waist)*2+(mid+8)*waist*2-8)
    pen = U(.75,2.04) + U(-.12,.12)
    # Smooth displacement field shared at crossings; a slight drift per pen lift.
    xn=x/5; yn=y/8
    delta = np.stack([xn*yn, yn**2-.4, xn**2-.4, np.sin(yn*np.pi), np.cos(yn*np.pi), np.sin(xn*np.pi*.75)*np.cos(yn*np.pi*.68), np.sin(yn*np.pi*1.6+U(-2,2))*np.cos(xn*2+U(-2,2))],axis=1) @ (N(size=(7,2))*np.array([.48,.65,.40,.43,.34,.38,.25])[:,None])
    x += delta[:,0]; y += delta[:,1]
    k = 0
    for stroke in shape:
        if k: points[k:k+len(stroke)] += N(size=2)*[.40,.55]
        k += len(stroke)
    x *= WRITING_WIDTH[digit]*np.exp(np.clip(N(0,.23),-.58,.56))
    # Slant and baseline angle are separate. Positive slant leans right at the top.
    slant = np.clip(N(.19,.27),-.54,.90)
    if digit == 1: slant = np.clip(N(.20,.25),-.55,.90)
    y += x*np.clip(N(0,.13),-.38,.38)
    x -= y*slant
    rad = (pen/2)*np.exp(N(0,.09)*np.sin(yn*2.5+U(-3,3)) + N(0,.12)*np.sin(xn*1.8+yn+U(-3,3)))
    if digit == 1: rad *= U(1.05,1.70)
    # A modest taper at one or both ends of a stroke.
    start = 0
    for stroke in shape:
        phase=np.linspace(0,1,len(stroke)); taper=1-U(0,.40)*np.exp(-phase*18)-U(0,.30)*np.exp((phase-1)*18)
        rad[start:start+len(stroke)] *= taper; start+=len(stroke)
    lo=(points-rad[:,None]).min(0); hi=(points+rad[:,None]).max(0)
    scale = U(19.4,20.5)/max(hi-lo)
    points = (points-(hi+lo)/2)*scale+13.5; rad*=scale
    # Paint samples of the continuous centerline in a local neighborhood.
    # Do this before and after centering, to avoid resampling/blur at the end.
    def paint(xy):
        r = int(np.ceil(rad.max()+.8)); s=np.arange(-r,r+1)
        by=np.floor(xy[:,1]).astype(np.int32); bx=np.floor(xy[:,0]).astype(np.int32)
        xx=bx[:,None,None]+s[None,None,:]; yy=by[:,None,None]+s[None,:,None]
        dist=np.sqrt((xx-xy[:,0,None,None])**2+(yy-xy[:,1,None,None])**2)
        ink=np.clip((rad[:,None,None]-dist)/1.0+.5,0,1)
        im=np.zeros(28*28,dtype=F)
        np.maximum.at(im,(np.clip(yy,0,27)*28+np.clip(xx,0,27)).ravel(),ink.ravel())
        return im.reshape(28,28)
    im=paint(points); mass=im.sum(dtype=np.float64)
    target=13.75 + U(-.75,.75,size=2)
    points += target - [(im*X).sum()/mass,(im*Y).sum()/mass]
    return paint(points)


# All features are the result of fixed operators defined below. They can be
# evaluated on a single drawing when writing that drawing's single weight row.
YC,XC=np.mgrid[-3:4,-3:4].astype(float)

def gauss(s):
    a=np.exp(-.5*(XC*XC+YC*YC)/(s*s)); return a/a.sum()

K0=np.stack([gauss(.85),gauss(1.7)]).astype(F)
E=[]
for theta in np.arange(8)*np.pi/4:
    across=XC*np.cos(theta)+YC*np.sin(theta); along=-XC*np.sin(theta)+YC*np.cos(theta)
    e=across*np.exp(-.5*((across/1.04)**2+(along/1.52)**2)); e-=e.mean(); E.append(e/e[e>0].sum())
K1=np.array(E,dtype=F)

def rays():
    # Ray patterns through a padded image sampled every two pixels. The max
    # on each ray is the strength of a foreground boundary in that direction.
    q=12; yy,xx=np.mgrid[:q,:q]*2+3
    result=[]
    y,x=np.mgrid[:28,:28]; y=y[None,None,:,:]-yy[:,:,None,None]; x=x[None,None,:,:]-xx[:,:,None,None]
    for k in range(8):
        theta=k*np.pi/4; ux=round(np.cos(theta),10); uy=round(np.sin(theta),10)
        dx=x*ux+y*uy; dy=-x*uy+y*ux
        result.append((dx>=-.01)&(np.abs(dy)<.8))
    return np.stack(result)

MASK=rays()
RAY_SUM=MASK.reshape(8*144,784).astype(F)
# The soft ray hit is min(1, sum of opacity along its narrow strip).
# Gap is (hit on the left + hit on the right - own ink - 1)_+ .
# Ring takes the intersection (a thresholded sum) over those four gap axes.
# This is a directional analogue of bounded background / loop detection.

def shell(im):
    hits=np.minimum(1, np.einsum('ij,j->i',RAY_SUM,im.ravel(), optimize=False)).reshape(8,12,12)
    pix=im[3:26:2,3:26:2]; gaps=np.maximum(0,hits[:4]+hits[4:]-pix[None,:,:]-1)
    hole=np.maximum(0,gaps.sum(0)-3)
    return gaps,hole


def whiten_self(a):
    # Single channel, or independent channels of ONE template. This is just
    # normalized correlation, not a covariance estimate or learned transform.
    a=a-a.mean(axis=(-2,-1),keepdims=True)
    return a/np.sqrt(np.mean(a*a,axis=(-2,-1),keepdims=True)+1.e-5)


def pooled_directional(im):
    v=windows(np.pad(im,((3,3),(3,3))), (7,7))[0::2,0::2]
    edge=np.maximum(0,np.einsum('xyij,kij->kxy',v,K1,optimize=False))
    return windows(edge,(3,3),axis=(1,2))[:,::2,::2].max(axis=(-2,-1))


def descriptors(im):
    win=windows(np.pad(im,((3,3),(3,3))),(7,7))[::2,::2]
    blur=np.einsum('xyij,kij->kxy',win,K0,optimize=False)
    eb=pooled_directional(im); gaps,hole=shell(im)
    # Global orientation magnitudes are useful too: normalize the entire edge
    # tensor jointly, not each orientation separately. Same for ray axes.
    edge=whiten_self(eb.reshape(1,48,6))
    gaps=windows(gaps,(2,2),axis=(1,2))[:,::2,::2].mean(axis=(-2,-1))
    dirs=whiten_self(gaps.reshape(1,24,6))
    smooth_hole=windows(hole,(2,2))[::2,::2].mean(axis=(-2,-1))
    # Cosine-like similarity for loop maps, with an extra 'empty map' token.
    # Encoding a constant reference token keeps no-hole/no-hole a valid match.
    hv=np.r_[smooth_hole.ravel(),F(1.25)].reshape(1,1,37)
    return whiten_self(blur), edge, dirs, whiten_self(hv)


class Network:
    def __init__(self): self.layers=[]; self.w={}
    def layer(self,op,name,inp,**args):
        self.layers.append(dict(name=name,op=op,inputs=inp if isinstance(inp,list) else [inp],**args)); return name
    def affine(self,op,name,inp,w,b,**args):
        w=np.asarray(w,dtype=F); self.w[name+'.W']=w; self.w[name+'.b']=np.broadcast_to(np.asarray(b,dtype=F),(len(w),)).copy()
        return self.layer(op,name,inp,out=len(w),**args)
    def conv(self,name,inp,w,b=0,stride=1,padding=0):
        k=w.shape[-1]; return self.affine('conv',name,inp,w,b,kernel=[k,k],stride=stride,padding=padding)
    def dense(self,name,inp,w,b=0): return self.affine('dense',name,inp,w,b)
    def reshape(self,name,inp,*s): return self.layer('reshape',name,inp,shape=list(s))
    def pool(self,op,name,inp,size,stride): return self.layer(op,name,inp,size=size,stride=stride)
    def norm(self,name,inp): return self.layer('norm',name,inp)
    def relu(self,name,inp): return self.layer('relu',name,inp)
    def add(self,name,*inputs): return self.layer('add',name,list(inputs))


def make_features(net):
    blur=net.norm('ink_normalized',net.conv('ink_scales','image',K0[:,None],stride=2,padding=3))
    deriv=net.relu('edge_strength',net.conv('directional_derivatives','image',K1[:,None],stride=2,padding=3))
    edge=net.pool('maxpool','edge_neighborhood',deriv,[3,3],[2,2])
    edge=net.norm('edge_normalized',net.reshape('edge_array',edge,1,48,6))
    # w = max(0, 1 - ray integral) = how much that ray misses.
    miss=net.relu('ray_miss',net.dense('ray_transmittance','image',-RAY_SUM,1))
    miss=net.reshape('ray_planes',miss,8,12,12)
    mix=np.zeros((4,8,1,1),dtype=F)
    for k in range(4): mix[k,[k,k+4],0,0]=-1
    both=net.conv('both_sides',miss,mix,1)
    sel=np.zeros((144,784),dtype=F)
    y,x=np.mgrid[:12,:12]*2+3; sel[np.arange(144),(y*28+x).ravel()]=-1
    bg=net.reshape('negative_ink_grid',net.dense('negative_ink_samples','image',sel),1,12,12)
    bg=net.conv('foreground_exclusion',bg,np.ones((4,1,1,1),dtype=F))
    gaps=net.relu('bounded_spaces',net.add('opposed_ray_test',both,bg))
    dg=net.pool('avgpool','space_cells',gaps,[2,2],[2,2]); dg=net.norm('space_normalized',net.reshape('space_array',dg,1,24,6))
    hole=net.relu('enclosure',net.conv('all_four_axes',gaps,np.ones((1,4,1,1),dtype=F),-3))
    hole=net.pool('avgpool','loop_cells',hole,[2,2],[2,2])
    hole=net.dense('loop_reference_token',hole,np.vstack((np.eye(36),np.zeros((1,36)))),np.r_[np.zeros(36),1.25])
    hole=net.norm('loop_normalized',net.reshape('loop_array',hole,1,1,37))
    return blur,edge,dg,hole


def main(folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    net=Network(); features=make_features(net)
    # Uniformly many rows per class. Keep groups of equally diverse rows,
    # pool within each group, then combine the overall maximum and the group mean.
    per=3200; groups=8; total=10*per
    widths=(392,288,144,37)
    banks=[np.empty((total,w),dtype=F) for w in widths]
    for d in range(10):
        for i in range(per):
            ds=descriptors(draw(d,i)); j=d*per+i
            for bank,desc in zip(banks,ds): bank[j]=desc.ravel()
    # A sum of block correlations; the coefficients are design constants,
    # not determined on generated glyphs, training images, or labels.
    # Division by number of positions turns each dot product into a correlation.
    banks[0][:,:196] *= .20/196; banks[0][:,196:] *= .17/196
    banks[1] *= .36/288; banks[2] *= .18/144; banks[3] *= .09/37
    names=['ink_match','edge_match','space_match','loop_match']
    partial=[net.dense(name,feat,bank) for name,feat,bank in zip(names,features,banks)]
    dots=net.add('individual_glyph_scores',*partial)
    m=net.reshape('class_and_style',dots,10,groups,per//groups)
    m=net.pool('maxpool','neighborhood_winners',m,[1,per//groups],[1,per//groups])
    a=net.pool('maxpool','best_match',m,[groups,1],[groups,1]); a=net.reshape('strongest',a,10)
    b=net.pool('avgpool','average_of_winners',m,[groups,1],[groups,1]); b=net.reshape('corroboration',b,10)
    best=net.dense('best_match_weight',a,np.eye(10,dtype=F)*.67)
    other=net.dense('corroboration_weight',b,np.eye(10,dtype=F)*.33)
    net.add('digit_scores',best,other)
    (folder/'network.json').write_text(json.dumps({'layers':net.layers},indent=2)+'\n')
    np.savez(folder/'weights.npz',**net.w)

if __name__=='__main__': main(sys.argv[1])
