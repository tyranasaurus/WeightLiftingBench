"""Data-free handwritten digit recognizer.

All curves are generic pen paths written in this source.  No data are read,
no parameters are estimated, and no optimization/training is performed.
The weight matrix implements normalized correlation with rendered pen paths.
"""
import json
import sys
from pathlib import Path
import numpy as np


VARIANTS = 14336


def halton(n, base):
    f, v = 1.0, 0.0
    while n:
        f /= base
        v += f * (n % base)
        n //= base
    return v


def line(*points):
    points = np.array(points, dtype=np.float64)
    pieces = []
    for a, b in zip(points[:-1], points[1:]):
        t = np.linspace(0, 1, max(3, int(np.linalg.norm(b-a)*28)+1))
        pieces.append(a[None, :] + t[:, None]*(b-a)[None, :])
    return np.concatenate(pieces)


def bez(start, *segments):
    result = []
    p = np.array(start, dtype=np.float64)
    t = np.linspace(0, 1, 25)[:, None]
    for a, b, c in segments:
        a, b, c = np.array([a, b, c], dtype=np.float64)
        result.append((1-t)**3*p + 3*(1-t)**2*t*a + 3*(1-t)*t*t*b + t**3*c)
        p = c
    return np.concatenate(result)


def oval(cx, cy, rx, ry, power=1., phase=0.):
    t = np.linspace(0, 2*np.pi, 121) + phase
    c, s = np.cos(t), np.sin(t)
    return np.stack([cx + rx*np.sign(c)*np.abs(c)**power,
                     cy + ry*np.sign(s)*np.abs(s)**power], 1)


def pen_paths(digit, style, q):
    """Coordinates are abstract, with the top at -1 and bottom at +1."""
    a, b, c = q[:3]
    if digit == 0:
        # Round, narrow, flattened, or slightly pointed closed oval.
        power = .78 + .48*a
        p = oval(0, 0, 1, 1, power)
        p[:, 0] *= 1 + (.25*b-.125)*p[:, 1]
        return [p]
    if digit == 1:
        bend = (a-.5)*.26
        stem = bez((.08,-1), ((bend,-.45),(-bend,.45),(-.08,1)))
        if style < 3:
            return [stem]
        flag = line((-.35-.48*b,-.48-.20*c),(.08,-1))
        if style < 6:
            return [flag, stem]
        foot = line((-.50-.25*b,1),(.45+.25*c,1))
        return [stem, flag, foot] if style == 6 else [stem, foot]
    if digit == 2:
        if style < 5:
            return [bez((-.92,-.53),
                ((-.68,-1.15),(.68,-1.24),(.93,-.66)),
                ((1.08,-.17),(-.55,.43),(-.96,.97)),
                ((-.25,.83),(.48,1.13),(.98,.84+.25*a)))]
        if style < 7:
            # Cursive two, including its little lower-left loop.
            return [bez((-.85,-.57),
                ((-.54,-1.18),(.92,-1.17),(.91,-.55)),
                ((.87,-.05),(-.40,.72),(-.88,.92)),
                ((-1.22,.87),(-.99,.41),(-.55,.63)),
                ((-.04,.91),(.49,1.12),(1.02,.82)))]
        return [bez((-.90,-.80),
                ((-.25,-1.11),(.77,-1.11),(.97,-.69)),
                ((.98,-.32),(-.40,.35),(-.92,.98))),
                line((-.92,.98),(1.,.98))]
    if digit == 3:
        middle = -.12 + .22*a
        if style < 5:
            return [bez((-.92,-.75),
                ((-.35,-1.12),(.88,-1.17),(.92,-.61)),
                ((.97,-.27),(.39,middle),(-.15,middle)),
                ((.43,middle-.08),(1.07,.11),(.93,.57)),
                ((.83,1.14),(-.35,1.15),(-.94,.69+.16*b)))]
        # Angular upper half, often taught as a top bar and diagonal.
        return [line((-.88,-.94),(.90,-.94),(-.16,middle)),
                bez((-.16,middle),
                    ((.45,middle-.13),(1.08,.16),(.91,.63)),
                    ((.73,1.14),(-.31,1.10),(-.91,.72)))]
    if digit == 4:
        cross = -.05 + .42*a
        right = .35 + .28*b
        if style < 5:
            left_top = -.60 + .70*c
            return [line((left_top,-1),(-.87,cross),(1.,cross)),
                    line((right,-.97),(right-.15,1))]
        if style < 7:
            return [line((right,-1),(-.94,cross),(.94,cross)),
                    line((right,-1),(right-.14,1))]
        return [bez((-.36,-1),((-.53,-.57),(-.96,.15),(-.73,cross))),
                line((-.73,cross),(1,cross)),
                line((right,-.88),(right-.10,1))]
    if digit == 5:
        mid = -.08 + .23*a
        bar = line((.92,-1),(-.67,-1))
        if style < 5:
            body = bez((-.67,-1),
                ((-.71,-.67),(-.84,mid-.08),(-.81,mid+.02)),
                ((-.05,mid-.40),(1.07,mid-.09),(.91,.58)),
                ((.78,1.18),(-.40,1.12),(-.94,.73)))
        elif style < 7:
            body = bez((-.67,-1),
                ((-.78,-.64),(-.67,-.37),(-.49,mid-.06)),
                ((.10,mid+.04),(1.13,.13),(.87,.65)),
                ((.60,1.15),(-.43,1.13),(-.94,.70)))
        else:
            body = line((-.67,-1),(-.88,mid),(.25,mid))
            body = np.concatenate([body, bez((.25,mid),
                ((1.09,mid),(.99,.94),(.15,1)),
                ((-.35,1.04),(-.72,.90),(-.94,.74)))])
        return [bar, body]
    if digit == 6:
        topx = .32 + .61*a
        loopy = .15 + .28*b
        if style < 6:
            return [bez((topx,-.98),
                ((-.10,-1.05),(-1.04,-.20),(-.88,.53)),
                ((-.78,1.13),(.75,1.20),(.95,.55)),
                ((1.18,loopy-.51),(-.52,loopy-.49),(-.88,loopy+.15)))]
        return [bez((topx,-1),
                ((-.20,-.85),(-.95,-.07),(-.88,.54)),
                ((-.77,1.13),(.75,1.15),(.94,.53)),
                ((1.11,-.06),(-.42,-.16),(-.88,.39)))]
    if digit == 7:
        if style < 4:
            paths = [line((-.97,-.97),(.97,-.97)),
                     bez((.97,-.97),((.51,-.29),(.04,.35),(-.39,1)))]
        elif style < 6:
            paths = [bez((-.95,-.71),
                         ((-.91,-1.13),(.26,-.93),(.98,-.96))),
                     bez((.98,-.96),((.53,-.26),(-.02,.31),(-.38,1)))]
        else:
            paths = [line((-.99,-.98),(.99,-.98),(-.34,1))]
        if style in (3,5,7):
            paths.append(line((-.40,-.04),(.56,-.04-.11*a)))
        return paths
    if digit == 8:
        join = -.18 + .34*a
        rt = .66 + .25*b
        if style < 4:
            t = np.linspace(0, 2*np.pi, 201)
            y = -np.cos(t)
            x = np.sin(2*t)*(.91-(1-rt)*np.cos(t)*.5)
            y += join*(1-y*y)
            return [np.stack([x,y],1)]
        top = oval(-.04,(-1+join)/2,rt,(1+join)/2)
        bottom = oval(.04,(1+join)/2,.96,(1-join)/2)
        if style < 7:
            return [top,bottom]
        # Slight overlap of the bowls in a fast, continuous double loop.
        return [oval(-.05,-.47,.79,.57),oval(.05,.45,.94,.60)]
    if digit == 9:
        # Upper bowl and descending stem. Bowl depth is a genuine writing
        # variation, independent of the overall aspect ratio.
        join = -.38 + .60*a
        right = .83
        cy, ry = (-1+join)/2, (1+join)/2
        loop = oval(-.03,cy,.86,ry,power=.85+.30*b)
        if style < 4:
            stem = bez((right,cy),
                       ((right+.06,-.02),(.51,.58),(.26,1)))
        elif style < 7:
            stem = bez((right,cy),
                       ((.85,.12),(.67,1.20),(-.48,.93)))
        else:
            stem = line((right,cy),(.58,1))
        return [loop,stem]
    raise ValueError(digit)


def curve_template(digit, index, grid):
    style = index % 8
    # Bases coprime to the eight-style cycle avoid coupling geometry to style.
    j = index + 1
    q = np.array([halton(j,p) for p in (11,13,17,19,23,29,31,37,41,43,47,53)])
    paths = pen_paths(digit, style, q[6:])
    # General handwriting variation; these are design intervals, not estimates.
    width_ranges = ((.48,.95),(.46,.70),(.49,.98),(.43,.88),(.53,.94),
                    (.48,.95),(.45,.87),(.48,.92),(.44,.85),(.45,.88))
    lo, hi = width_ranges[digit]
    # Letter aspect ratios refer to the extents of each path, not to its
    # nominal control-point box (Bezier controls need not lie on the curve).
    raw = np.concatenate(paths)
    span = np.ptp(raw,axis=0)
    if digit != 1:
        midpoint = (raw.max(axis=0)+raw.min(axis=0))/2
        factor=np.array([2/span[0],2/span[1]])
        paths = [(p-midpoint)*factor for p in paths]
    width = lo + (hi-lo)*q[0]
    slant = -.55 + 1.00*q[1]
    thickness = .50 + 1.35*halton(j,59)
    size = 20.0 - 2*thickness + (q[2]-.5)*1.6
    bow = (q[3]-.5)*.46
    taper = (q[4]-.5)*.40
    vertical = (q[5]-.5)*.36
    def transform(p):
        x,y = p[:,0].copy(),p[:,1].copy()
        x = width*(x*(1+taper*y) + bow*(1-y*y))
        y = y + vertical*(1-y*y)
        x = x + slant*y
        return np.stack([x,y],1)
    transformed = [transform(p) for p in paths]
    allp = np.concatenate(transformed)
    extent = np.ptp(allp,axis=0).max()
    transformed = [p*(size/extent) for p in transformed]
    # Resample paths by arc length, giving each unit of pen travel equal mass.
    samples, masses = [], []
    for p in transformed:
        lengths = np.linalg.norm(np.diff(p,axis=0),axis=1)
        good = np.r_[True,lengths>1.e-9]
        p = p[good]
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        n = max(3,int(np.ceil(arc[-1]/.28)))
        u = (np.arange(n)+.5)*(arc[-1]/n)
        s = np.stack([np.interp(u,arc,p[:,0]),np.interp(u,arc,p[:,1])],1)
        samples.append(s)
        masses.append(np.full(n,arc[-1]/n))
    points = np.concatenate(samples)
    mass = np.concatenate(masses)
    mass /= mass.sum()
    com = (points*mass[:,None]).sum(axis=0)
    # MNIST's stated center-of-mass placement, not any measured training center.
    target=np.array([13.5+1.50*(halton(j,61)-.5),
                     13.5+1.50*(halton(j,67)-.5)])
    points += target-com
    p = points.astype(np.float32)
    dist2 = ((grid[:,None,:]-p[None,:,:])**2).sum(axis=2).min(axis=1)
    distance = np.sqrt(dist2)
    silhouette = 1/(1+np.exp(np.clip((distance-thickness)/.32,-40,40)))
    # Center the rendered union of strokes, not their (overlap-counted) pen
    # lengths. This is a deterministic rendering convention, not data fitting.
    raster_com=(silhouette[:,None]*grid).sum(axis=0)/silhouette.sum()
    shift=target-raster_com
    img=silhouette.reshape(28,28)
    yy,xx=np.mgrid[:28,:28]
    sx,sy=xx-shift[0],yy-shift[1]
    ix,iy=np.floor(sx).astype(int),np.floor(sy).astype(int)
    fx,fy=sx-ix,sy-iy
    centered=np.zeros((28,28),dtype=np.float64)
    for dx,dy,w in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),
                    (0,1,(1-fx)*fy),(1,1,fx*fy)):
        xx1,yy1=ix+dx,iy+dy
        good=(xx1>=0)&(xx1<28)&(yy1>=0)&(yy1<28)
        centered[good] += w[good]*img[yy1[good],xx1[good]]
    # Gaussian smoothing broadens tolerance without discarding negative space.
    kernel=np.array([1,4,6,4,1],dtype=np.float64)/16
    padded=np.pad(centered,2)
    smoothed=np.zeros((28,28),dtype=np.float64)
    for dy in range(5):
        for dx in range(5):
            smoothed += kernel[dy]*kernel[dx]*padded[dy:dy+28,dx:dx+28]
    result=smoothed.ravel()
    result -= result.mean()
    result /= np.sqrt(np.mean(result**2)+1.e-5)*784
    return result.astype(np.float32)


def build(out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    layers, weights = [], {}
    def op(name,kind,inp,**kw):
        layers.append(dict(name=name,op=kind,inputs=inp if isinstance(inp,list) else [inp],**kw))
        return name
    def conv(name,inp,w,b,pad=0):
        w=np.array(w,dtype=np.float32)
        weights[name+'.W']=w
        weights[name+'.b']=np.array(b,dtype=np.float32)
        return op(name,'conv',inp,out=w.shape[0],kernel=list(w.shape[-2:]),stride=1,padding=pad)
    def dense(name,inp,w,b):
        weights[name+'.W']=np.asarray(w,dtype=np.float32)
        weights[name+'.b']=np.asarray(b,dtype=np.float32)
        return op(name,'dense',inp,out=len(b))
    # The same fixed binomial Gaussian is applied to input and analytic models.
    kernel=np.array([1,4,6,4,1],dtype=np.float32)/16
    smooth=np.outer(kernel,kernel)
    conv('soft_ink','image',smooth.reshape(1,1,5,5),[0.],pad=2)
    op('standard_silhouette','norm','soft_ink')
    count=10*VARIANTS
    silhouette=np.empty((count,784),dtype=np.float32)
    yy,xx=np.mgrid[:28,:28]
    grid=np.stack([xx.ravel(),yy.ravel()],1).astype(np.float32)
    for digit in range(10):
        for j in range(VARIANTS):
            silhouette[digit*VARIANTS+j]=curve_template(digit,j,grid)
        print('Constructed analytic models for',digit,flush=True)
    dense('model_scores','standard_silhouette',silhouette,np.zeros(count))
    op('class_models','reshape','model_scores',shape=[10,VARIANTS,1])
    op('best_model','maxpool','class_models',size=[VARIANTS,1],stride=[VARIANTS,1])
    op('scores','reshape','best_model',shape=[10])
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**weights)
    print('Weight entries:',sum(a.size for a in weights.values()),flush=True)


if __name__=='__main__':
    build(sys.argv[1])
