#!/usr/bin/env python3
"""Build a data-free, synthetic-stroke nearest-prototype network for MNIST-like digits."""
import json
import os
import sys
import numpy as np

# Every prototype below is rendered from generic pen paths.  No data files are read.
SEED = 20261119
NPROTO = 625                 # per class at 14x14 and 7x7
NHI = 300                    # per class at 28x28


def bezier(p0, p1, p2, p3, n=28):
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
    u = 1.0 - t
    return u**3*p0 + 3*u*u*t*p1 + 3*u*t*t*p2 + t**3*p3


def line(a, b, n=18, bend=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b-a
    q1 = a + d/3 + bend*np.array([-d[1], d[0]])
    q2 = a + 2*d/3 + bend*np.array([-d[1], d[0]])
    return bezier(a, q1, q2, b, n)


def chain(points, closed=False, nseg=14, smooth=0.72):
    """A Catmull-Rom-like smooth pen line through hand-chosen waypoints."""
    p = np.asarray(points, float)
    if closed:
        q = np.vstack([p[-1], p, p[0], p[1]])
        count = len(p)
    else:
        q = np.vstack([p[0], p, p[-1]])
        count = len(p)-1
    ans = []
    for i in range(count):
        p0, p1, p2, p3 = q[i:i+4]
        c1 = p1 + smooth*(p2-p0)/6.0
        c2 = p2 - smooth*(p3-p1)/6.0
        ans.append(bezier(p1, c1, c2, p2, nseg))
    return np.vstack(ans)


def loop(rx, ry, rng, top=0.0, skew=0.0, n=100):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    # Small low-frequency variations make round, oval, and handwritten loops.
    a = rng.uniform(-0.055, 0.055)
    b = rng.uniform(-0.035, 0.035)
    phase = rng.uniform(0, 2*np.pi)
    rr = 1.0 + a*np.cos(2*t+phase) + b*np.sin(3*t-phase)
    x = rx*np.sin(t)*rr + skew*np.cos(t)
    y = top - ry*np.cos(t)*rr
    return np.stack([x, y], 1)


def generic_paths(k, rng):
    """Return generic normalized pen paths for one decimal digit."""
    j = lambda s: rng.normal(0.0, s)
    paths = []
    if k == 0:
        rx = rng.uniform(.31, .47); ry = rng.uniform(.49, .56)
        if rng.random() < .82:
            paths = [loop(rx, ry, rng, skew=rng.uniform(-.04,.04))]
        else:  # occasionally boxier or pinched, but still a single closed loop
            paths = [chain([(-rx*.70,-ry),(-rx,-ry*.32),(-rx*.92,ry*.45),
                            (-rx*.45,ry),(rx*.45,ry),(rx*.94,ry*.42),
                            (rx,-ry*.40),(rx*.45,-ry),(-rx*.70,-ry)],
                           nseg=12,smooth=rng.uniform(.45,.75))]

    elif k == 1:
        style = rng.random()
        x = rng.uniform(-.05,.10)
        main = chain([(x+j(.025),-.50),(x+j(.035),-.12),(x+j(.03),.22),(x+j(.035),.53)],
                     nseg=18, smooth=.45)
        paths = [main]
        if style < .72:
            paths.append(chain([(-.22+j(.04),-.28),(x+j(.025),-.52),(x+j(.025),-.31)],
                               nseg=16, smooth=.35))
        if style < .22 or style > .92:
            paths.append(chain([(-.23+j(.04),.51),(x,.53),(.28+j(.04),.51)],
                               nseg=13, smooth=.25))

    elif k == 2:
        # Rounded cap, descending sweep, and a baseline.
        pts = [(-.37+j(.035),-.33+j(.025)),(-.25+j(.04),-.50+j(.02)),
               (.05+j(.04),-.54+j(.02)),(.34+j(.03),-.40+j(.03)),
               (.37+j(.025),-.18+j(.035)),(.17+j(.035),.05+j(.04)),
               (-.11+j(.04),.28+j(.04)),(-.36+j(.03),.49+j(.025)),
               (-.05+j(.04),.51+j(.02)),(.38+j(.035),.50+j(.02))]
        paths = [chain(pts, nseg=11, smooth=rng.uniform(.55,.88))]

    elif k == 3:
        pts = [(-.33+j(.035),-.42),(-.08+j(.04),-.54),(.24+j(.035),-.47),
               (.37+j(.025),-.27),(.25+j(.035),-.08),(-.05+j(.04),.00),
               (.22+j(.035),.06),(.38+j(.025),.25),(.30+j(.03),.45),
               (.04+j(.04),.55),(-.31+j(.04),.42)]
        paths = [chain(pts, nseg=10, smooth=rng.uniform(.62,.92))]

    elif k == 4:
        style = rng.random()
        xr = rng.uniform(.12,.25)
        if style < .72:  # the common open-top 4
            paths = [chain([(xr+j(.025),-.54),(-.38+j(.035),.14+j(.03)),
                            (.39+j(.03),.14+j(.03))], nseg=20, smooth=.25),
                     chain([(xr+j(.02),-.53),(xr+j(.025),-.05),
                            (xr+j(.025),.54)], nseg=20, smooth=.35)]
        elif style < .92:  # a closed triangular 4
            paths = [chain([(xr,-.54),(-.38+j(.03),.15),(.36+j(.03),.15),
                            (xr+j(.02),-.54)], nseg=18, smooth=.15),
                     line((xr,.0),(xr+j(.02),.54),22)]
        else:              # vertical-left schoolbook form
            paths = [chain([(-.24+j(.03),-.49),(-.28+j(.03),.12),
                            (.38+j(.03),.12)], nseg=20, smooth=.22),
                     chain([(xr,-.51),(xr,.08),(xr+j(.02),.54)], nseg=20, smooth=.25)]

    elif k == 5:
        # Keep the cap angular; round only the lower bowl.  Most handwritten
        # fives are one continuous stroke, but separated caps are also common.
        bowl = [(-.33+j(.025),-.05),(-.08+j(.04),-.10),(.23+j(.035),-.02),
                (.38+j(.025),.20),(.32+j(.03),.40),(.08+j(.04),.54),
                (-.19+j(.04),.50),(-.36+j(.03),.37)]
        if rng.random() < .78:
            paths = [chain([(.36+j(.03),-.49),(-.32+j(.03),-.49),
                            (-.34+j(.025),-.05)]+bowl[1:],
                           nseg=10, smooth=rng.uniform(.38,.72))]
        else:
            paths = [line((-.34+j(.03),-.48),(.36+j(.03),-.48),22,
                          bend=rng.uniform(-.03,.03)),
                     chain([(-.31+j(.03),-.43)]+bowl,nseg=10,
                           smooth=rng.uniform(.55,.82))]

    elif k == 6:
        pts = [(.24+j(.035),-.49),(.00+j(.04),-.53),(-.24+j(.035),-.35),
               (-.37+j(.025),-.05),(-.36+j(.025),.28),(-.22+j(.03),.48),
               (.04+j(.035),.55),(.30+j(.03),.43),(.37+j(.025),.20),
               (.26+j(.035),.01),(.03+j(.04),-.05),(-.22+j(.035),.04),
               (-.35+j(.025),.25)]
        paths = [chain(pts, nseg=9, smooth=rng.uniform(.67,.92))]

    elif k == 7:
        paths = [chain([(-.39+j(.03),-.48),(.02+j(.04),-.51),
                        (.40+j(.03),-.47)], nseg=16, smooth=.2),
                 chain([(.37+j(.025),-.45),(.17+j(.035),-.12),
                        (-.02+j(.035),.20),(-.17+j(.035),.54)],
                       nseg=16, smooth=.48)]
        if rng.random() < .32:
            paths.append(line((-.20+j(.03),.02),(.20+j(.03),.02+j(.025)),18,
                              bend=rng.uniform(-.04,.04)))

    elif k == 8:
        style = rng.random()
        if style < .84:
            pts = [(0+j(.025),-.55),(-.27+j(.03),-.49),(-.34+j(.025),-.27),
                   (-.25+j(.03),-.08),(0+j(.03),.02),(.27+j(.03),.14),
                   (.34+j(.025),.37),(.18+j(.03),.54),(-.08+j(.03),.55),
                   (-.32+j(.025),.39),(-.27+j(.03),.17),(0+j(.03),.01),
                   (.26+j(.03),-.12),(.31+j(.025),-.36),(.18+j(.03),-.52),
                   (0+j(.025),-.55)]
            paths = [chain(pts, nseg=8, smooth=rng.uniform(.68,.90))]
        else:
            top = loop(rng.uniform(.25,.34), rng.uniform(.25,.31), rng,
                       top=-.26+j(.015), skew=rng.uniform(-.025,.025), n=70)
            bot = loop(rng.uniform(.28,.37), rng.uniform(.28,.34), rng,
                       top=.27+j(.015), skew=rng.uniform(-.025,.025), n=75)
            paths = [top, bot]

    elif k == 9:
        rx = rng.uniform(.27,.38); ry = rng.uniform(.28,.38)
        cy = rng.uniform(-.25,-.15)
        ring = loop(rx, ry, rng, top=cy, skew=rng.uniform(-.025,.025), n=82)
        side = rng.random()
        if side < .94:
            x0 = rx*.86 + j(.02)
            tail = chain([(x0,cy-.03),(x0+j(.025),.12),(.13+j(.04),.39),
                          (-.01+j(.05),.54)], nseg=16, smooth=.62)
        else:
            tail = chain([(-rx*.82,cy),(-.27+j(.03),.12),(-.10+j(.04),.40),
                          (.05+j(.04),.54)], nseg=16, smooth=.55)
        paths = [ring, tail]
    return paths


def transform_paths(paths, rng):
    # General handwriting variation: aspect, slant, tiny rotation, and target size.
    sx = rng.uniform(.88,1.12)
    sy = rng.uniform(.94,1.07)
    slant = rng.uniform(-.18,.23)
    angle = rng.uniform(-.065,.065)
    ca, sa = np.cos(angle), np.sin(angle)
    out=[]
    for p in paths:
        q=p.copy()
        q[:,0] *= sx; q[:,1] *= sy
        q[:,0] += -slant*q[:,1]
        x=q[:,0]*ca-q[:,1]*sa; y=q[:,0]*sa+q[:,1]*ca
        out.append(np.stack([x,y],1))
    return out


def raster(paths, width, rng):
    allp=np.vstack(paths)
    span=np.ptp(allp,axis=0)
    target=rng.uniform(18.8,20.4)
    radius=width/2
    scale=max(6.0,(target-2*radius)/max(span))
    pp=[p*scale for p in paths]
    allp=np.vstack(pp)
    ctr=(allp.min(0)+allp.max(0))/2
    pp=[p-ctr+13.5 for p in pp]
    yy,xx=np.mgrid[0:28,0:28]
    def draw(ps):
        d2=(xx[:,:,None]-ps[:,0])**2+(yy[:,:,None]-ps[:,1])**2
        d=np.sqrt(d2.min(2))
        # A one-pixel antialiasing ramp around a solid pen stroke.
        return np.clip(radius+.58-d,0.0,1.0)
    im=np.maximum.reduce([draw(p) for p in pp])
    # Match the standard centering convention using the synthetic ink's own COM.
    mass=im.sum()
    cy=(im*yy).sum()/mass; cx=(im*xx).sum()/mass
    residual=rng.normal(0,.22,2)
    delta=np.array([13.5+residual[0]-cx,13.5+residual[1]-cy])
    pp=[p+delta for p in pp]
    im=np.maximum.reduce([draw(p) for p in pp])
    return im.astype(np.float32)


def make_digit(k, rng):
    paths=generic_paths(k,rng)
    paths=transform_paths(paths,rng)
    # Most MNIST pens are around 2 pixels wide, with a useful broad tail.
    width=np.clip(rng.normal(2.25,.48),1.25,3.35)
    return raster(paths,width,rng)


def norm_image(a):
    a=a.astype(np.float64)
    return ((a-a.mean())/np.sqrt(((a-a.mean())**2).mean()+1e-5)).astype(np.float32)


def pool2(a):
    h,w=a.shape
    return a[:h//2*2,:w//2*2].reshape(h//2,2,w//2,2).mean((1,3))


def build(outdir):
    os.makedirs(outdir,exist_ok=True)
    rng=np.random.default_rng(SEED)
    t28=[]; t14=[]; t7=[]
    for k in range(10):
        for n in range(NPROTO):
            im=make_digit(k,rng)
            lo=pool2(im); tiny=pool2(lo)
            if n < NHI: t28.append(norm_image(im))
            t14.append(norm_image(lo)); t7.append(norm_image(tiny))
    # The class-major order permits max pooling over each class's prototype bank.
    # Convert normalized inner products to correlations and weight the scales.
    W28=np.stack(t28).reshape(10*NHI,-1)*(0.34/(28*28))
    W14=np.stack(t14).reshape(10*NPROTO,-1)*(1.00/(14*14))
    W7 =np.stack(t7 ).reshape(10*NPROTO,-1)*(0.42/(7*7))

    layers=[
      {"name":"n28","op":"norm","inputs":["image"]},
      {"name":"corr28","op":"dense","inputs":["n28"],"out":10*NHI},
      {"name":"grid28","op":"reshape","inputs":["corr28"],"shape":[10,15,20]},
      {"name":"best28","op":"maxpool","inputs":["grid28"],"size":[15,20],"stride":[15,20]},
      {"name":"score28","op":"reshape","inputs":["best28"],"shape":[10]},
      {"name":"p14","op":"avgpool","inputs":["image"],"size":[2,2],"stride":[2,2]},
      {"name":"n14","op":"norm","inputs":["p14"]},
      {"name":"corr14","op":"dense","inputs":["n14"],"out":10*NPROTO},
      {"name":"grid14","op":"reshape","inputs":["corr14"],"shape":[10,25,25]},
      {"name":"best14","op":"maxpool","inputs":["grid14"],"size":[25,25],"stride":[25,25]},
      {"name":"score14","op":"reshape","inputs":["best14"],"shape":[10]},
      {"name":"p7","op":"avgpool","inputs":["p14"],"size":[2,2],"stride":[2,2]},
      {"name":"n7","op":"norm","inputs":["p7"]},
      {"name":"corr7","op":"dense","inputs":["n7"],"out":10*NPROTO},
      {"name":"grid7","op":"reshape","inputs":["corr7"],"shape":[10,25,25]},
      {"name":"best7","op":"maxpool","inputs":["grid7"],"size":[25,25],"stride":[25,25]},
      {"name":"score7","op":"reshape","inputs":["best7"],"shape":[10]},
      {"name":"calibrated","op":"dense","inputs":["score14"],"out":10},
      {"name":"scores","op":"add","inputs":["score28","calibrated","score7"]}
    ]
    with open(os.path.join(outdir,"network.json"),"w") as f:
        json.dump({"layers":layers},f,separators=(",",":"))
    # Identity readout with hand-chosen priors counters the greater visual
    # diversity of 2, 3, 5, 6, and 8 and the overly generic bar/loop classes.
    prior=np.array([.01,-.10,.05,.04,-.03,.08,.03,-.07,.06,-.04],np.float32)
    np.savez(os.path.join(outdir,"weights.npz"),
             **{"corr28.W":W28.astype(np.float32),"corr28.b":np.zeros(10*NHI,np.float32),
                "corr14.W":W14.astype(np.float32),"corr14.b":np.zeros(10*NPROTO,np.float32),
                "corr7.W":W7.astype(np.float32),"corr7.b":np.zeros(10*NPROTO,np.float32),
                "calibrated.W":np.eye(10,dtype=np.float32),"calibrated.b":prior})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv)>1 else ".")
