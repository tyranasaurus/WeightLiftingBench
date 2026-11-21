#!/usr/bin/env python3
"""Build a data-free handwritten digit classifier from procedural pen-stroke models."""
import sys, json
from pathlib import Path
import numpy as np

# All prototypes below are drawings made from generic pen paths.  No data files are read.
SEED = 314159
PER_CLASS = 420


def gauss_filters():
    q = np.arange(-2, 3, dtype=np.float64)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    g = np.exp(-(xx*xx + yy*yy) / (2.0 * 0.90**2))
    g /= g.sum()
    fs = [g]
    # Eight signed, oriented first derivatives.  ReLU makes these ink-edge channels.
    for a in np.arange(8) * np.pi / 4.0:
        u = xx*np.cos(a) + yy*np.sin(a)
        f = -u * np.exp(-(xx*xx + yy*yy) / (2.0 * 1.05**2))
        f -= f.mean()
        f /= np.sqrt((f*f).sum())
        fs.append(f)
    return np.asarray(fs, np.float32)[:, None, :, :]


def bezier(p0, p1, p2, p3, n=32):
    t = np.linspace(0., 1., n)
    s = 1.-t
    return (s**3)[:,None]*p0 + (3*s*s*t)[:,None]*p1 + (3*s*t*t)[:,None]*p2 + (t**3)[:,None]*p3


def smooth_path(nodes, closed=False, n=12):
    """Catmull-Rom interpolation through typed pen-guide points."""
    a = np.asarray(nodes, np.float64)
    if closed:
        ext = np.vstack((a[-1], a, a[0], a[1]))
        count = len(a)
    else:
        ext = np.vstack((a[0], a, a[-1]))
        count = len(a)-1
    pieces = []
    for j in range(count):
        p0,p1,p2,p3 = ext[j:j+4]
        t = np.linspace(0.,1.,n,endpoint=(j==count-1))[:,None]
        z = .5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t)
        pieces.append(z)
    return np.vstack(pieces)


def jitter_nodes(nodes, rng, amount=.035):
    a = np.asarray(nodes, np.float64).copy()
    a += rng.normal(0., amount, a.shape)
    return a


def ellipse(cx, cy, rx, ry, rng, phase=0., wobble=.025, n=100):
    t = np.linspace(phase, phase+2*np.pi, n)
    # Low-frequency deviations produce natural, non-geometric loops.
    w1,w2,w3,w4 = rng.normal(0.,wobble,4)
    x = cx + rx*np.cos(t) + w1*np.cos(2*t) + w2*np.sin(3*t)
    y = cy + ry*np.sin(t) + w3*np.sin(2*t) + w4*np.cos(3*t)
    return np.column_stack((x,y))


def glyph(k, rng):
    """Return generic normalized pen paths for one decimal digit."""
    paths=[]
    if k == 0:
        rx=rng.uniform(.38,.52); ry=rng.uniform(.82,1.02)
        paths=[ellipse(0,0,rx,ry,rng,wobble=.035)]
    elif k == 1:
        lean=rng.uniform(-.14,.15)
        main=np.array([[lean-.04,-.92],[lean, -.48],[lean+.01,.03],[lean+.04,.94]])
        paths=[smooth_path(jitter_nodes(main,rng,.025),False,18)]
        if rng.random()<.58:
            top=np.array([[-.30,-.66],[lean-.10,-.92],[lean+.01,-.98]])
            paths.insert(0,smooth_path(jitter_nodes(top,rng,.025),False,16))
        if rng.random()<.25:
            foot=np.array([[lean-.27,.94],[lean+.04,.94],[lean+.32,.94]])
            paths.append(smooth_path(jitter_nodes(foot,rng,.02),False,14))
    elif k == 2:
        n=[[-.43,-.66],[-.20,-.96],[.20,-.96],[.48,-.70],[.43,-.34],[.12,.02],[-.20,.34],[-.48,.75],[-.43,.91],[.04,.92],[.49,.90]]
        a=jitter_nodes(n,rng,.035)
        a[:,0]*=rng.uniform(.88,1.08)
        paths=[smooth_path(a,False,11)]
    elif k == 3:
        n=[[-.40,-.78],[-.08,-.96],[.34,-.86],[.48,-.55],[.35,-.20],[-.07,-.02],[.33,.10],[.50,.45],[.39,.78],[.06,.96],[-.40,.79]]
        a=jitter_nodes(n,rng,.035)
        a[5,0]+=rng.uniform(-.08,.08)
        paths=[smooth_path(a,False,12)]
    elif k == 4:
        right=rng.uniform(.16,.34)
        # The common open-top handwritten four.
        diag=[ [right-.03,-.98],[-.12,-.48],[-.48,.18],[-.43,.30],[.49,.28] ]
        vert=[[right,-.98],[right-.01,-.26],[right,.36],[right+.01,.98]]
        paths=[smooth_path(jitter_nodes(diag,rng,.025),False,14),
               smooth_path(jitter_nodes(vert,rng,.018),False,16)]
        if rng.random()<.48: # closed triangular-top four, another common hand
            paths[0]=smooth_path(jitter_nodes([[-.42,.23],[-.27,-.30],[right,-.96],[right,.28],[-.42,.23]],rng,.025),False,14)
        elif rng.random()<.30: # more angular, wide open-top variant
            paths[0]=smooth_path(jitter_nodes([[right-.02,-.98],[-.43,.19],[-.43,.31],[.50,.29]],rng,.022),False,16)
    elif k == 5:
        n=[[.43,-.90],[-.05,-.94],[-.42,-.87],[-.43,-.37],[-.39,-.05],[-.03,-.13],[.33,-.02],[.48,.29],[.40,.67],[.10,.92],[-.28,.89],[-.46,.72]]
        a=jitter_nodes(n,rng,.035)
        paths=[smooth_path(a,False,11)]
    elif k == 6:
        n=[[.34,-.82],[.03,-.92],[-.31,-.63],[-.47,-.16],[-.46,.38],[-.29,.79],[.05,.96],[.37,.77],[.48,.40],[.34,.06],[.04,-.05],[-.29,.08],[-.44,.35]]
        a=jitter_nodes(n,rng,.035)
        paths=[smooth_path(a,False,11)]
    elif k == 7:
        top=jitter_nodes([[-.48,-.88],[-.03,-.93],[.49,-.88]],rng,.025)
        fall=jitter_nodes([[.49,-.88],[.30,-.51],[.06,-.08],[-.10,.36],[-.22,.96]],rng,.025)
        paths=[smooth_path(top,False,16),smooth_path(fall,False,16)]
        if rng.random()<.18:
            cross=jitter_nodes([[-.25,.00],[.23,-.01]],rng,.02)
            paths.append(smooth_path(cross,False,15))
    elif k == 8:
        # Two touching, slightly asymmetric loops are a robust model of a written eight.
        rt=rng.uniform(.29,.43); rb=rng.uniform(.34,.47)
        paths=[ellipse(rng.uniform(-.05,.05),-.47,rt,rng.uniform(.44,.56),rng,wobble=.035),
               ellipse(rng.uniform(-.05,.05),.46,rb,rng.uniform(.47,.59),rng,phase=np.pi,wobble=.035)]
        u=rng.random()
        if u<.36: # continuous figure-eight stroke variant
            t=np.linspace(0,2*np.pi,150)
            paths=[np.column_stack((rng.uniform(.35,.45)*np.sin(t),-.92*np.sin(2*t)/2))]
        elif u<.56: # S-shaped crossing with rounded end turns
            n=[[-.28,-.88],[.13,-.98],[.42,-.67],[.30,-.30],[-.10,.02],[-.39,.32],[-.36,.68],[-.05,.94],[.35,.82],[.39,.47],[.09,.06],[-.24,-.26],[-.35,-.58],[-.28,-.88]]
            paths=[smooth_path(jitter_nodes(n,rng,.028),False,9)]
    elif k == 9:
        rx=rng.uniform(.32,.45); ry=rng.uniform(.43,.59)
        loop=ellipse(-.05,-.43,rx,ry,rng,wobble=.030)
        side=rng.uniform(.20,.37)
        u=rng.random()
        if u<.55:
            stem=np.array([[side,-.62],[side,-.18],[side-.02,.22],[side-.10,.62],[.02,.92],[-.20,.98]])
        elif u<.82: # straighter descender
            stem=np.array([[side,-.60],[side,-.08],[side,.42],[side-.03,.94]])
        else: # open, handwritten q-like loop and descending side
            loop=smooth_path(jitter_nodes([[-.36,-.64],[-.25,-.92],[.10,-.98],[.36,-.77],[.39,-.39],[.23,-.08],[-.08,.00],[-.35,-.20],[-.39,-.55]],rng,.025),False,12)
            stem=np.array([[side,-.62],[side,-.05],[side-.02,.46],[.02,.96]])
        paths=[loop,smooth_path(jitter_nodes(stem,rng,.025),False,14)]
    return paths


def render(k, rng):
    paths=glyph(k,rng)
    # Writer-level affine variation: width, slant, and small rotation.
    sx=rng.uniform(.92,1.36); sy=rng.uniform(.94,1.07)
    sl=rng.uniform(-.22,.18)
    ang=rng.normal(0.,.045)
    ca,sa=np.cos(ang),np.sin(ang)
    A=np.array([[ca,-sa],[sa,ca]]) @ np.array([[sx,sl],[0.,sy]])
    paths=[p@A.T for p in paths]
    allp=np.vstack(paths)
    span=np.ptp(allp,axis=0).max()
    sigma=rng.uniform(.58,1.38)
    # MNIST-like normalization: ink occupies approximately a 20 pixel box.
    target=19.4-3.5*sigma
    scale=target/span
    paths=[p*scale for p in paths]
    allp=np.vstack(paths)
    center=(allp.min(0)+allp.max(0))/2
    paths=[p-center+13.5 for p in paths]
    yy,xx=np.mgrid[0:28,0:28]
    def paint(ps):
        d2=np.full((28,28),1e9)
        for p in ps:
            d2=np.minimum(d2,(xx-p[0])**2+(yy-p[1])**2)
        return np.exp(-d2/(2*sigma*sigma))
    im=np.zeros((28,28),np.float64)
    for p in paths: im=np.maximum(im,paint(p))
    # Shift the rendered center of mass to frame center, as in the task.
    mass=im.sum()
    cy=(im*yy).sum()/mass; cx=(im*xx).sum()/mass
    shift=np.array([13.5-cx,13.5-cy]) + rng.normal(0.,.16,2)
    im=np.zeros((28,28),np.float64)
    for p in paths: im=np.maximum(im,paint(p+shift))
    # A little opacity variation broadens grey-level style without data dependence.
    im=np.clip(im*rng.uniform(.88,1.12),0,1)
    return im.astype(np.float32)


def features(im, filters):
    x=im.astype(np.float64)
    x=(x-x.mean())/np.sqrt(x.var()+1e-5)
    xp=np.pad(x,2)
    z=np.empty((len(filters),28,28),np.float64)
    for o,f in enumerate(filters[:,0]):
        q=np.zeros((28,28),np.float64)
        for dy in range(5):
            for dx in range(5): q += float(f[dy,dx])*xp[dy:dy+28,dx:dx+28]
        z[o]=np.maximum(q,0)
    # 2x2 max pooling, exactly matching the submitted network.
    z=np.maximum.reduce((z[:,0::2,0::2],z[:,1::2,0::2],z[:,0::2,1::2],z[:,1::2,1::2]))
    mu=z.mean((1,2),keepdims=True)
    z=(z-mu)/np.sqrt(((z-mu)**2).mean((1,2),keepdims=True)+1e-5)
    # Ink correlation and directional edge channels contribute equally after
    # independent normalization; this emphasizes topology over exact greyscale.
    z[0] *= 1.00
    z[1:] *= 1.00
    return z.reshape(-1).astype(np.float32)


def build(dest):
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    filters=gauss_filters()
    rng=np.random.default_rng(SEED)
    total=10*PER_CLASS
    W=np.empty((total,9*14*14),np.float32)
    row=0
    for k in range(10):
        for j in range(PER_CLASS):
            W[row]=features(render(k,rng),filters)/(14*14)
            row+=1
    layers=[
      {"name":"input_norm","op":"norm","inputs":["image"]},
      {"name":"features","op":"conv","inputs":["input_norm"],"out":9,"kernel":[5,5],"stride":1,"padding":2},
      {"name":"positive","op":"relu","inputs":["features"]},
      {"name":"local_tolerance","op":"maxpool","inputs":["positive"],"size":[2,2],"stride":[2,2]},
      {"name":"feature_norm","op":"norm","inputs":["local_tolerance"]},
      {"name":"prototype_scores","op":"dense","inputs":["feature_norm"],"out":total},
      {"name":"by_class","op":"reshape","inputs":["prototype_scores"],"shape":[10,PER_CLASS,1]},
      {"name":"best_style","op":"maxpool","inputs":["by_class"],"size":[PER_CLASS,1],"stride":[PER_CLASS,1]},
      {"name":"scores","op":"reshape","inputs":["best_style"],"shape":[10]}
    ]
    weights={"features.W":filters.astype(np.float32),"features.b":np.zeros(9,np.float32),
             "prototype_scores.W":W,"prototype_scores.b":np.zeros(total,np.float32)}
    (dest/'network.json').write_text(json.dumps({"layers":layers},separators=(',',':')))
    np.savez(dest/'weights.npz',**weights)

if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv)>1 else '.')
