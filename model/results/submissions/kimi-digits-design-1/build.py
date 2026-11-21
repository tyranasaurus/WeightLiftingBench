"""Build a data-free, generative stroke-template MNIST classifier.

Every template is rendered from the typed pen paths below.  Randomness has a
fixed seed and describes ordinary handwriting variation (slant, aspect,
curvature, and pen width), not any observed dataset example.
"""
import sys, json
from pathlib import Path
import numpy as np

M = 4000                        # templates for every class
SEED = 20261119


def line(*points):
    return np.asarray(points, dtype=np.float64)


def bezier(p0, p1, p2, p3, n=28):
    t = np.linspace(0.0, 1.0, n)[:, None]
    a, b, c, d = [np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3)]
    return (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t**2*c + t**3*d


def ellipse(cx, cy, rx, ry, start=0.0, stop=2*np.pi, n=72, wobble=0.0):
    t = np.linspace(start, stop, n)
    r = 1.0 + wobble*np.sin(3*t+0.4) + 0.35*wobble*np.cos(5*t)
    return np.column_stack((cx+rx*r*np.cos(t), cy+ry*r*np.sin(t)))


def paths_for(d, family, r):
    """A small vocabulary of conventional ways to write each digit."""
    j = lambda a=0.06: float(r.uniform(-a, a))
    if d == 0:
        rx = r.uniform(.50, .82)
        ry = r.uniform(.86, 1.10)
        if family == 0:
            return [ellipse(0, 0, rx, ry, wobble=.06)]
        if family == 1:
            return [ellipse(j(), 0, rx, ry, start=.04, stop=2*np.pi+.04, wobble=.09)]
        return [ellipse(0, 0, rx, ry, wobble=.12)]
    if d == 1:
        top = (.02+j(), -1.0)
        bot = (-.03+j(), .98)
        if family == 0:
            return [line(top, (.07+j(), -.15), bot)]
        if family == 1:
            return [line((-.48+j(), -.57), (-.12, -.87), top, bot)]
        if family == 2:
            return [line((-.48+j(), -.57), (-.10, -.87), top, bot),
                    line((-.52, .98), (.48, .98))]
        return [line((-.45+j(), -.65), top, (-.02+j(), .12), bot),
                line((-.42, .98), (.43, .98))]
    if d == 2:
        if family == 0:
            a = bezier((-.64,-.43),(-.58,-1.08),(.52,-1.13),(.62,-.46))
            b = bezier((.62,-.46),(.75,.03),(-.36,.48),(-.63,.86))
            c = bezier((-.63,.86),(-.15,.78),(.28,.84),(.68,.84))
            return [a,b,c]
        if family == 1:
            a = bezier((-.67,-.48),(-.52,-1.04),(.53,-1.10),(.63,-.43))
            b = bezier((.63,-.43),(.38,.08),(-.25,.48),(-.61,.84))
            c = line((-.61,.84),(.67,.84))
            return [a,b,c]
        if family == 2:
            a = bezier((-.62,-.44),(-.55,-1.04),(.54,-1.08),(.61,-.42))
            b = bezier((.61,-.42),(.36,.02),(-.12,.27),(-.28,.42))
            c = bezier((-.28,.42),(-.73,.56),(-.67,.99),(-.25,.80))
            e = bezier((-.25,.80),(.02,.70),(.36,.84),(.67,.84))
            return [a,b,c,e]
        a = line((-.65,-.64),(-.25,-.91),(.26,-.91),(.62,-.56))
        b = bezier((.62,-.56),(.57,.05),(-.25,.51),(-.62,.84))
        return [a,b,line((-.62,.84),(.67,.84))]
    if d == 3:
        if family == 0:
            a = bezier((-.57,-.72),(-.12,-1.07),(.57,-.99),(.59,-.54))
            b = bezier((.59,-.54),(.61,-.20),(.27,-.10),(.04,-.01))
            c = bezier((.04,-.01),(.58,-.09),(.79,.25),(.57,.65))
            e = bezier((.57,.65),(.34,1.06),(-.30,1.03),(-.64,.72))
            return [a,b,c,e]
        if family == 1:
            a = line((-.60,-.85),(.48,-.85),(.06,-.04))
            b = bezier((.06,-.04),(.72,-.16),(.77,.43),(.43,.76))
            c = bezier((.43,.76),(.12,1.03),(-.37,.96),(-.64,.70))
            return [a,b,c]
        a = bezier((-.62,-.66),(-.10,-1.04),(.55,-.91),(.52,-.48))
        b = line((.52,-.48),(-.03,-.02),(.47,.02))
        c = bezier((.47,.02),(.85,.35),(.42,1.03),(-.04,.91))
        e = bezier((-.04,.91),(-.32,.89),(-.53,.79),(-.65,.65))
        return [a,b,c,e]
    if d == 4:
        if family == 0:
            return [line((.28,-1.0),(-.65,.29),(.70,.29)),
                    line((.36,-.96),(.35,.99))]
        if family == 1:
            return [line((-.28,-.98),(-.46,.28),(.68,.28)),
                    line((.31,-1.0),(.35,.99))]
        if family == 2:
            return [line((-.62,.30),(.17,-1.0),(.18,.99)),
                    line((-.62,.30),(.70,.30))]
        return [line((.12,-.98),(-.61,-.02),(-.62,.30),(.69,.30)),
                line((.38,-.91),(.34,.99))]
    if d == 5:
        if family == 0:
            a = line((.59,-.89),(-.48,-.89),(-.56,-.10))
            b = bezier((-.56,-.10),(-.13,-.34),(.57,-.32),(.66,.12))
            c = bezier((.66,.12),(.79,.60),(.26,1.04),(-.15,.86))
            e = bezier((-.15,.86),(-.36,.82),(-.52,.76),(-.62,.66))
            return [a,b,c,e]
        if family == 1:
            a = line((.60,-.88),(-.46,-.88),(-.53,-.08),(.05,-.20))
            b = bezier((.05,-.20),(.71,-.30),(.80,.40),(.47,.72))
            c = bezier((.47,.72),(.17,1.00),(-.35,.96),(-.62,.69))
            return [a,b,c]
        a = line((.57,-.90),(-.49,-.90),(-.56,-.09),(.30,-.22),(.63,.11))
        b = line((.63,.11),(.57,.60),(.18,.88),(-.26,.88),(-.62,.68))
        return [a,b]
    if d == 6:
        loop = ellipse(r.uniform(-.10,.10),r.uniform(.34,.50),r.uniform(.48,.74),r.uniform(.43,.64),wobble=.075)
        if family == 0:
            stem = bezier((.53,-.76),(-.06,-1.13),(-.65,-.74),(-.62,-.03))
            stem2 = bezier((-.62,-.03),(-.64,.24),(-.60,.51),(-.48,.66))
            return [stem,stem2,loop]
        if family == 1:
            stem = bezier((.58,-.72),(-.18,-1.15),(-.70,-.54),(-.57,.28))
            b = bezier((-.57,.28),(-.53,.94),(.31,1.12),(.60,.54))
            c = bezier((.60,.54),(.77,.02),(.13,-.16),(-.45,.20))
            return [stem,b,c]
        stem = line((.47,-.87),(-.40,-.55),(-.57,.34))
        return [stem,loop]
    if d == 7:
        if family == 0:
            return [line((-.67,-.84),(.66,-.84),(-.15,.99))]
        if family == 1:
            return [line((-.67,-.84),(.66,-.84),(-.15,.99)),
                    line((-.39,.04),(.34,.04))]
        if family == 2:
            a = bezier((-.67,-.80),(-.15,-.94),(.34,-.75),(.65,-.85))
            b = bezier((.65,-.85),(.31,-.25),(.14,.39),(-.15,.99))
            return [a,b]
        a = line((-.65,-.84),(.65,-.84),(.30,-.18),(-.13,.99))
        return [a,line((-.35,.08),(.32,.08))]
    if d == 8:
        if family == 0:
            return [ellipse(r.uniform(-.07,.07),-.49,r.uniform(.43,.61),r.uniform(.40,.54),wobble=.065),
                    ellipse(r.uniform(-.08,.08),.48,r.uniform(.52,.74),r.uniform(.43,.62),wobble=.075)]
        if family == 1:
            return [ellipse(r.uniform(-.08,.08),-.48,r.uniform(.48,.66),r.uniform(.41,.56),wobble=.08),
                    ellipse(r.uniform(-.08,.08),.47,r.uniform(.49,.71),r.uniform(.42,.61),wobble=.08)]
        a = bezier((.02,-.02),(-.75,-.28),(-.53,-1.08),(.03,-.99))
        b = bezier((.03,-.99),(.67,-1.04),(.69,-.25),(.02,-.02))
        c = bezier((.02,-.02),(-.69,.18),(-.74,.99),(-.02,.98))
        e = bezier((-.02,.98),(.72,1.06),(.79,.22),(.02,-.02))
        return [a,b,c,e]
    # 9: a closed upper bowl and a descending right-hand tail.
    loop = ellipse(r.uniform(-.10,.08),r.uniform(-.50,-.35),r.uniform(.48,.73),r.uniform(.45,.66),wobble=.075)
    if family == 0:
        tail = bezier((.56,-.53),(.67,.03),(.59,.67),(.19,.96))
        return [loop,tail]
    if family == 1:
        return [loop,line((.57,-.62),(.51,.10),(.38,.98))]
    if family == 2:
        tail = bezier((.57,-.54),(.67,.13),(.28,.89),(-.27,.98))
        return [loop,tail]
    tail = bezier((.50,-.83),(.76,-.13),(.51,.71),(-.12,.96))
    return [loop,tail]


def render(paths, width):
    """Antialiased distance-to-pen rendering on a 28 by 28 canvas."""
    yy, xx = np.mgrid[0:28,0:28]
    dist = np.full((28,28), 100.0)
    for path in paths:
        p, q = path[:-1], path[1:]
        vx, vy = q[:,0]-p[:,0], q[:,1]-p[:,1]
        den = vx*vx+vy*vy
        den = np.maximum(den, 1e-12)
        # Process a whole polyline in a vectorized (pixel, segment) array.
        dx = xx.ravel()[:,None]-p[None,:,0]
        dy = yy.ravel()[:,None]-p[None,:,1]
        t = np.clip((dx*vx+dy*vy)/den, 0.0, 1.0)
        d2 = (dx-t*vx)**2+(dy-t*vy)**2
        dist = np.minimum(dist, np.sqrt(d2.min(axis=1)).reshape(28,28))
    return np.clip(width*.5+.52-dist, 0.0, 1.0)


def make_template(d, index, r):
    families = [3,4,4,3,4,3,3,4,3,4][d]
    paths = paths_for(d, index % families, r)
    # Smooth, independent geometric handwriting variation.
    shear = r.uniform(-.34,.34)
    aspect = r.uniform(.72,1.28)
    rot = r.uniform(-.14,.14)
    wave = r.uniform(-.13,.13)
    bow = r.uniform(-.10,.10)
    # Low-frequency elastic modes let separate parts of a stroke move without
    # introducing sharp kinks.  Their ranges describe pen movement generally.
    modes = r.uniform(-.11,.11,size=6)
    cr, sr = np.cos(rot), np.sin(rot)
    transformed = []
    for path in paths:
        x0, y0 = path[:,0].copy(), path[:,1].copy()
        x = aspect*(x0 + shear*y0 + wave*np.sin(np.pi*y0) + bow*(1-y0*y0))
        x += modes[0]*np.sin(np.pi*(y0+1)/2) + modes[1]*(1-y0*y0)*x0
        y = y0 + modes[2]*np.sin(np.pi*x0) + modes[3]*(1-x0*x0)
        y += modes[4]*x0*y0 + modes[5]*np.sin(np.pi*(x0+y0)/2)
        transformed.append(np.column_stack((cr*x-sr*y, sr*x+cr*y)))
    paths = transformed
    allp = np.concatenate(paths)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    extent = np.maximum(hi-lo, .15)
    scale = min(18.7/extent[0],18.7/extent[1])
    center = (lo+hi)*.5
    width = r.uniform(1.20,3.25)
    mapped = [np.column_stack((13.5+(p[:,0]-center[0])*scale,
                               13.5+(p[:,1]-center[1])*scale)) for p in paths]
    im = render(mapped,width)
    # The data description specifies a centered centre of mass.  Apply that
    # normalization to the generated pen drawing too, with subpixel redraw.
    yy,xx = np.mgrid[0:28,0:28]
    mass = im.sum()+1e-12
    dx,dy = 13.5-(im*xx).sum()/mass,13.5-(im*yy).sum()/mass
    # Integer image normalization and pen rasterization leave subpixel/one-pixel
    # placement differences even when the nominal centre of mass is fixed.
    dx += r.uniform(-1.0,1.0); dy += r.uniform(-1.0,1.0)
    mapped = [p+np.array([dx,dy]) for p in mapped]
    return render(mapped,width)


def blur(im, sigma=1.35):
    """Gaussian positional uncertainty of a typed pen path."""
    radius = 4
    x = np.arange(-radius,radius+1)
    k = np.exp(-.5*(x/sigma)**2)
    k /= k.sum()
    a = np.pad(im,radius)
    # Separable convolution with a zero background outside the frame.
    h = sum(k[i]*a[radius:radius+28,i:i+28] for i in range(2*radius+1))
    a = np.pad(h,radius)
    v = sum(k[i]*a[i:i+28,radius:radius+28] for i in range(2*radius+1))
    return v


def build(out):
    r = np.random.default_rng(SEED)
    templates = []
    for d in range(10):
        for i in range(M):
            # Mix a broad background model with the blurred ideal pen ink.
            p = .07+.88*blur(make_template(d,i,r))
            p = np.clip(p,.07,.95)
            templates.append(p)
    p = np.asarray(templates,dtype=np.float64)
    ps = p.reshape(10*M,14,2,14,2).mean(axis=(2,4))
    pt = p.reshape(10*M,7,4,7,4).mean(axis=(2,4))
    # For Bernoulli ink probabilities p, log P(image) is the affine form
    # image @ logit(p) + sum(log(1-p)).  Division only keeps scores small.
    w = (np.log(p)-np.log1p(-p)).reshape(10*M,784)/784
    b = np.log1p(-p).sum(axis=(1,2))/784
    ws = (np.log(ps)-np.log1p(-ps)).reshape(10*M,196)/196
    bs = np.log1p(-ps).sum(axis=(1,2))/196
    wt = (np.log(pt)-np.log1p(-pt)).reshape(10*M,49)/49
    bt = np.log1p(-pt).sum(axis=(1,2))/49
    weights = {'likelihood.W':(.50*w).astype(np.float32),
               'likelihood.b':(.50*b).astype(np.float32),
               'small_likelihood.W':(.35*ws).astype(np.float32),
               'small_likelihood.b':(.35*bs).astype(np.float32),
               'tiny_likelihood.W':(.15*wt).astype(np.float32),
               'tiny_likelihood.b':(.15*bt).astype(np.float32)}
    layers = [
        {'name':'pool14','op':'avgpool','inputs':['image'],'size':[2,2],'stride':[2,2]},
        {'name':'pool7','op':'avgpool','inputs':['image'],'size':[4,4],'stride':[4,4]},
        {'name':'likelihood','op':'dense','inputs':['image'],'out':10*M},
        {'name':'small_likelihood','op':'dense','inputs':['pool14'],'out':10*M},
        {'name':'tiny_likelihood','op':'dense','inputs':['pool7'],'out':10*M},
        {'name':'combined','op':'add','inputs':['likelihood','small_likelihood','tiny_likelihood']},
        {'name':'by_class','op':'reshape','inputs':['combined'],'shape':[10,1,M]},
        {'name':'best','op':'maxpool','inputs':['by_class'],'size':[1,M],'stride':[1,M]},
        {'name':'scores','op':'reshape','inputs':['best'],'shape':[10]},
    ]
    out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':')))
    np.savez(out/'weights.npz',**weights)


if __name__ == '__main__':
    build(Path(sys.argv[1]))
