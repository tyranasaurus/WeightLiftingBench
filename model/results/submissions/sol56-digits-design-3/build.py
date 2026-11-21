#!/usr/bin/env python3
"""Build a fixed, analytic handwritten-digit recognizer without data files.

The model is a nearest-shape classifier in a small bank of analytic edge
features.  All reference shapes below are generated from generic pen paths,
with a fixed-seed range of ordinary handwriting deformations.
"""
import json
import math
import os
import sys
import numpy as np

SEED = 1729
VARIANTS = 1800


def catmull(points, steps=9, closed=False):
    """Sample a centripetal-looking Catmull-Rom pen path."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 2:
        t = np.linspace(0.0, 1.0, steps + 1)
        return p[0][None] * (1-t[:, None]) + p[1][None] * t[:, None]
    if closed:
        q = np.concatenate([p[-1:], p, p[:2]], axis=0)
        count = len(p)
    else:
        q = np.concatenate([p[:1], p, p[-1:]], axis=0)
        count = len(p)-1
    ans = []
    for i in range(count):
        a,b,c,d = q[i:i+4]
        t = np.linspace(0.0, 1.0, steps, endpoint=False)[:,None]
        # Standard Catmull-Rom; handwriting jitter makes exact parameterization unimportant.
        z = 0.5*((2*b) + (-a+c)*t + (2*a-5*b+4*c-d)*t*t + (-a+3*b-3*c+d)*t*t*t)
        ans.append(z)
    ans.append((p[:1] if closed else p[-1:]))
    return np.concatenate(ans, axis=0)


def ellipse_path(rx, ry, phase=0.0, n=64, pinch=0.0):
    a = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
    # Harmonic perturbations turn perfect typography into a pen-drawn loop.
    x = rx*np.sin(a)*(1.0 + pinch*np.cos(a))
    y = -ry*np.cos(a)
    return np.stack([x,y], axis=1)


def base_strokes(d, rng):
    """Generic pen-center paths in coordinates whose y axis points down."""
    u = rng.random()
    if d == 0:
        rx = rng.uniform(.60,.82); ry = 1.0
        p = ellipse_path(rx, ry, rng.uniform(-.15,.15), 72, rng.uniform(-.10,.10))
        return [p]
    if d == 1:
        if u < .30:
            main = np.array([[.02,-1.0],[-.10,-.45],[-.08,.15],[-.02,1.0]])
        else:
            main = np.array([[-.35,-.55],[-.02,-1.0],[-.02,-.35],[-.03,.35],[.00,1.0]])
        out = [catmull(main, 10)]
        if rng.random() < .28:
            out.append(catmull([[-.45,1.0],[.45,1.0]],8))
        return out
    if d == 2:
        if u < .18:
            p=[[-.66,-.63],[-.38,-.96],[.22,-1.0],[.67,-.67],[.60,-.25],[.20,.12],[-.24,.45],[-.65,.88],[-.15,.96],[.70,.92]]
        elif u < .50:
            # Rounded top followed by a mostly diagonal descent and flat foot.
            p=[[-.58,-.62],[-.34,-.95],[.18,-.98],[.62,-.70],[.58,-.32],[.22,.05],[-.17,.36],[-.55,.70],[-.66,.91],[-.05,.94],[.68,.91]]
        else:
            p=[[-.65,-.62],[-.38,-.96],[.20,-1.0],[.68,-.72],[.65,-.30],[.28,.10],[-.18,.40],[-.62,.78],[-.68,.94],[.05,.95],[.70,.92]]
        return [catmull(p,7)]
    if d == 3:
        if u < .18:
            p=[[-.52,-.82],[-.05,-1.0],[.52,-.82],[.62,-.43],[.28,-.08],[-.10,.00],[.31,.05],[.67,.36],[.60,.78],[.10,1.0],[-.53,.84]]
        else:
            p=[[-.55,-.83],[-.05,-1.0],[.52,-.83],[.62,-.45],[.34,-.12],[-.08,.00],[.32,.10],[.67,.43],[.55,.83],[.06,1.0],[-.55,.82]]
        return [catmull(p,7)]
    if d == 4:
        if u < .72:
            # Common open-top four, usually drawn in two pen strokes.
            s1=catmull([[.32,-1.0],[.04,-.58],[-.62,.22],[-.58,.38],[.68,.36]],8)
            s2=catmull([[.32,-1.0],[.31,-.42],[.27,.24],[.22,1.0]],9)
            return [s1,s2]
        else:
            # Closed triangular four: explicit top-left-to-crossbar and right stem.
            s1=catmull([[-.46,-.88],[-.58,.20],[-.52,.36],[.68,.35]],8)
            s2=catmull([[-.46,-.88],[-.05,-.28],[.27,.34]],8)
            s3=catmull([[.31,-1.0],[.30,-.35],[.27,.35],[.22,1.0]],9)
            return [s1,s2,s3]
    if d == 5:
        p=[[.65,-.92],[.10,-.98],[-.52,-.92],[-.55,-.15],[-.12,-.22],[.38,-.12],[.67,.20],[.64,.58],[.35,.88],[-.05,1.0],[-.55,.82]]
        if u < .22:
            p=[[.62,-.94],[-.48,-.94],[-.52,-.10],[.05,-.20],[.53,-.04],[.67,.40],[.48,.80],[.03,1.0],[-.55,.82]]
        elif u < .48:
            p=[[.62,-.93],[-.10,-.94],[-.52,-.87],[-.48,-.15],[-.15,-.09],[.30,-.12],[.60,.12],[.66,.52],[.42,.84],[.02,1.0],[-.50,.84]]
        return [catmull(p,7)]
    if d == 6:
        if u < .25:
            p=[[.52,-.86],[.12,-1.0],[-.35,-.70],[-.62,-.18],[-.63,.45],[-.38,.86],[.03,1.0],[.48,.83],[.66,.43],[.53,.08],[.12,-.03],[-.32,.12],[-.58,.45]]
        else:
            p=[[.50,-.90],[.05,-1.0],[-.40,-.62],[-.65,-.08],[-.62,.55],[-.30,.92],[.10,1.0],[.52,.78],[.62,.35],[.42,.04],[.02,-.05],[-.38,.12],[-.57,.46]]
        return [catmull(p,7)]
    if d == 7:
        p=[[-.68,-.90],[-.10,-.94],[.70,-.90],[.46,-.51],[.18,-.10],[-.05,.35],[-.26,1.0]]
        out=[catmull(p,8)]
        if rng.random() < .28:
            out.append(catmull([[-.38,.05],[.40,-.02]],7))
        return out
    if d == 8:
        if u < .25:
            # Two separately penned loops.
            top=ellipse_path(rng.uniform(.42,.56),.53,0,44,rng.uniform(-.08,.08)); top[:,1]-=.49
            bot=ellipse_path(rng.uniform(.48,.62),.57,0,48,rng.uniform(-.08,.08)); bot[:,1]+=.46
            return [top,bot]
        p=[[.00,.00],[-.43,-.26],[-.49,-.62],[-.27,-.93],[.05,-1.0],[.42,-.78],[.45,-.40],[.22,-.12],[.00,.00],[-.35,.22],[-.52,.58],[-.34,.90],[.02,1.0],[.44,.83],[.54,.45],[.33,.15],[.00,.00]]
        return [catmull(p,6)]
    if d == 9:
        if u < .20:
            p=[[.47,.00],[.12,.18],[-.30,.08],[-.55,-.27],[-.47,-.70],[-.10,-1.0],[.35,-.90],[.57,-.55],[.52,-.08],[.46,.42],[.18,.83],[-.14,1.0]]
        else:
            p=[[.48,.00],[.14,.16],[-.30,.08],[-.55,-.29],[-.43,-.75],[-.05,-1.0],[.39,-.84],[.56,-.44],[.50,.02],[.43,.50],[.20,.90],[-.18,1.0]]
        return [catmull(p,7)]
    raise ValueError(d)


def deform(strokes, d, rng):
    """Apply smooth writer-like variations, then fit a generic 20-pixel box."""
    aspect = rng.uniform(.82,1.18)
    if d == 1: aspect = rng.uniform(.70,1.25)
    slant = rng.uniform(-.25,.28)
    rot = rng.normal(0.0, .045)
    bendx = rng.normal(0,.045)
    bendy = rng.normal(0,.025)
    ca,sa = math.cos(rot),math.sin(rot)
    ans=[]
    # Shared low-order deformation preserves deliberate intersections.
    for p0 in strokes:
        p=np.array(p0,dtype=np.float64,copy=True)
        y=p[:,1].copy(); x=p[:,0].copy()
        p[:,0] = aspect*x + slant*y + bendx*(y*y-.35)
        p[:,1] = y + bendy*x*y
        x=p[:,0].copy(); y=p[:,1].copy()
        p[:,0]=ca*x-sa*y; p[:,1]=sa*x+ca*y
        # Very small smooth path perturbation, independent of any real image.
        ph=rng.uniform(0,2*np.pi)
        p[:,0] += rng.uniform(.00,.045)*np.sin(np.linspace(ph,ph+2*np.pi,len(p)))
        p[:,1] += rng.uniform(.00,.035)*np.sin(np.linspace(ph+.7,ph+2*np.pi+.7,len(p)))
        ans.append(p)
    width=rng.uniform(1.20,3.15)
    target=rng.uniform(18.8,20.5)
    allp=np.concatenate(ans)
    span=allp.max(0)-allp.min(0)
    scale=(target-width)/max(span)
    center=(allp.max(0)+allp.min(0))/2
    ans=[(p-center)*scale + np.array([13.5,13.5]) for p in ans]
    return ans,width


def render(strokes, width):
    yy,xx=np.mgrid[0:28,0:28].astype(np.float64)
    best=np.full((28,28),1e9,dtype=np.float64)
    for p in strokes:
        # Dense sampled paths make point distance a close stroke-distance approximation.
        dx=xx[...,None]-p[:,0]
        dy=yy[...,None]-p[:,1]
        best=np.minimum(best, np.sqrt(np.min(dx*dx+dy*dy,axis=2)))
    radius=width/2
    # One-pixel analytic antialiasing around an opaque pen core.
    im=np.clip(radius+.55-best,0.0,1.0)
    # Align the intensity center of mass exactly as in the stated preprocessing.
    mass=im.sum()
    if mass > 0:
        cx=(im*xx).sum()/mass; cy=(im*yy).sum()/mass
        shift=np.array([13.5-cx,13.5-cy])
        if np.max(np.abs(shift)) > .025:
            moved=[p+shift for p in strokes]
            best.fill(1e9)
            for p in moved:
                dx=xx[...,None]-p[:,0]; dy=yy[...,None]-p[:,1]
                best=np.minimum(best,np.sqrt(np.min(dx*dx+dy*dy,axis=2)))
            im=np.clip(radius+.55-best,0.0,1.0)
    return im.astype(np.float32)


def filters():
    y,x=np.mgrid[-2:3,-2:3].astype(np.float64)
    g=np.exp(-(x*x+y*y)/(2*.90**2)); g/=g.sum()
    fs=[1.10*g]
    for a in np.arange(8)*np.pi/4:
        q=x*np.cos(a)+y*np.sin(a)
        f=-q*g
        f-=f.mean()
        f/=np.abs(f).sum()/2
        fs.append(.34*f)
    return np.asarray(fs,dtype=np.float32)[:,None]


def feature(images, W):
    """Exact numpy equivalent of norm, 5x5 conv, ReLU and 2x2 average pool."""
    a=np.asarray(images,dtype=np.float32)
    mu=a.mean((1,2),keepdims=True)
    a=(a-mu)/np.sqrt(((a-mu)**2).mean((1,2),keepdims=True)+1e-5)
    ap=np.pad(a,((0,0),(2,2),(2,2)))
    out=np.zeros((len(a),9,28,28),dtype=np.float32)
    for ky in range(5):
        for kx in range(5):
            out += a.dtype.type(1.0)*ap[:,None,ky:ky+28,kx:kx+28]*W[None,:,0,ky,kx,None,None] if False else 0
    # Written explicitly by output channel to keep build memory bounded.
    for o in range(9):
        v=np.zeros((len(a),28,28),dtype=np.float32)
        for ky in range(5):
            for kx in range(5): v += ap[:,ky:ky+28,kx:kx+28]*W[o,0,ky,kx]
        out[:,o]=np.maximum(v,0)
    out=(out[:,:,0::2,0::2]+out[:,:,0::2,1::2]+out[:,:,1::2,0::2]+out[:,:,1::2,1::2])*.25
    return out.reshape(len(a),-1)


def build(dest):
    os.makedirs(dest,exist_ok=True)
    F=filters()
    total=10*VARIANTS
    D=9*14*14
    proto=np.empty((total,D),dtype=np.float32)
    rng=np.random.default_rng(SEED)
    row=0
    batch=[]
    rows=[]
    for d in range(10):
        for j in range(VARIANTS):
            strokes=base_strokes(d,rng)
            strokes,width=deform(strokes,d,rng)
            batch.append(render(strokes,width)); rows.append(row); row+=1
            if len(batch)==64:
                proto[np.asarray(rows)]=feature(np.asarray(batch),F)
                batch=[]; rows=[]
    if batch: proto[np.asarray(rows)]=feature(np.asarray(batch),F)
    # Correlation minus half prototype energy is nearest squared Euclidean distance;
    # the omitted input energy is common to all classes.
    bias=(-.5*np.sum(proto*proto,axis=1)).astype(np.float32)
    # Conservative handwritten priors correct generic-shape volume imbalance.
    # These are class-level design constants, not statistics from images.
    class_prior=np.array([0.0,0.0,1.2,0.0,2.2,1.0,0.0,0.0,0.0,-1.4],dtype=np.float32)
    bias += np.repeat(class_prior,VARIANTS)
    layers=[
      {"name":"input_norm","op":"norm","inputs":["image"]},
      {"name":"analytic_edges","op":"conv","inputs":["input_norm"],"out":9,"kernel":[5,5],"stride":1,"padding":2},
      {"name":"edge_relu","op":"relu","inputs":["analytic_edges"]},
      {"name":"small_grid","op":"avgpool","inputs":["edge_relu"],"size":[2,2],"stride":[2,2]},
      {"name":"shape_scores","op":"dense","inputs":["small_grid"],"out":total},
      {"name":"by_class","op":"reshape","inputs":["shape_scores"],"shape":[10,VARIANTS,1]},
      {"name":"best_shape","op":"maxpool","inputs":["by_class"],"size":[VARIANTS,1],"stride":[VARIANTS,1]},
      {"name":"scores","op":"reshape","inputs":["best_shape"],"shape":[10]}
    ]
    with open(os.path.join(dest,"network.json"),"w") as f: json.dump({"layers":layers},f,separators=(",",":"))
    np.savez(os.path.join(dest,"weights.npz"),
             **{"analytic_edges.W":F.astype(np.float32),
                "analytic_edges.b":np.zeros(9,dtype=np.float32),
                "shape_scores.W":proto,
                "shape_scores.b":bias})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv)>1 else ".")
