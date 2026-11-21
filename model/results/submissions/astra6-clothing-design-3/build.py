"""Data-free, analytic clothing classifier. No files are read by this program.

Geometric primitives and generic garment proportions specify every template.
The fixed random seed samples geometric variations, not examples from a dataset.
No objective is optimized, and there is no training step, even on synthetic data.
"""
import sys
import json
from pathlib import Path
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def op(self, name, op, source, **settings):
        sources = [source] if isinstance(source, str) else source
        self.layers.append(dict(name=name, op=op, inputs=sources, **settings))
        return name

    def dense(self, name, source, W, b=None):
        W = np.asarray(W, np.float32)
        self.weights[name + '.W'] = W
        self.weights[name + '.b'] = np.zeros(len(W), np.float32) if b is None else np.asarray(b, np.float32)
        return self.op(name, 'dense', source, out=len(W))

    def conv(self, name, source, W, b=None, padding=0):
        W = np.asarray(W, np.float32)
        self.weights[name + '.W'] = W
        self.weights[name + '.b'] = np.zeros(len(W), np.float32) if b is None else np.asarray(b, np.float32)
        return self.op(name, 'conv', source, out=len(W), kernel=list(W.shape[-2:]), stride=1, padding=padding)

    def relu(self, name, source):
        return self.op(name, 'relu', source)

    def reshape(self, name, source, shape):
        return self.op(name, 'reshape', source, shape=shape)

    def pool(self, name, source, size, stride=None, kind='maxpool'):
        return self.op(name, kind, source, size=size, stride=size if stride is None else stride)


class Drawing:
    """Union/subtraction of polygons, ellipses and stroked polylines."""
    def __init__(self):
        self.parts = []
        self.bounds = []

    def polygon(self, pts, subtract=False):
        pts = np.asarray(pts, float)
        self.parts.append(('polygon', pts, subtract))
        if not subtract:
            self.bounds.extend(pts)

    def ellipse(self, x, y, rx, ry, subtract=False):
        self.parts.append(('ellipse', (x, y, rx, ry), subtract))
        if not subtract:
            self.bounds.extend([(x-rx, y-ry), (x+rx, y+ry)])

    def stroke(self, pts, width):
        pts = np.asarray(pts, float)
        for a, b in zip(pts[:-1], pts[1:]):
            d = b-a
            p = np.array([-d[1], d[0]]) / (np.linalg.norm(d)+1e-12) * width/2
            self.polygon([a+p, b+p, b-p, a-p])
        for x, y in pts:
            self.ellipse(x, y, width/2, width/2)

    def raster(self, scale=0.97, dx=0, dy=0, mirror=False):
        bound = np.asarray(self.bounds)
        lo, hi = bound.min(0), bound.max(0)
        center = (lo+hi)/2
        extent = (hi-lo).max()/scale
        p = (np.arange(56)+0.5)/56 - 0.5
        X, Y = np.meshgrid(p, p)
        X = (X-dx)*extent*( -1 if mirror else 1)+center[0]
        Y = (Y-dy)*extent+center[1]
        result = np.zeros((56, 56), bool)
        for typ, args, subtract in self.parts:
            if typ == 'ellipse':
                x,y,rx,ry = args
                inside = ((X-x)/rx)**2+((Y-y)/ry)**2 <= 1
            else:
                inside = np.zeros_like(result)
                for a,b in zip(args, np.roll(args, -1, axis=0)):
                    if a[1] == b[1]:
                        continue
                    cross = ((a[1]>Y)!=(b[1]>Y)) & (X < (b[0]-a[0])*(Y-a[1])/(b[1]-a[1])+a[0])
                    inside ^= cross
            if subtract:
                result &= ~inside
            else:
                result |= inside
        return result.astype(float).reshape(28,2,28,2).mean((1,3))


def neckline(d, rng, width, depth, kind):
    if kind == 'v':
        d.polygon([(-width/2,-0.04),(width/2,-0.04),(0,depth)], True)
    else:
        d.ellipse(0, -0.015, width/2, depth+0.015, True)


def garment(k, rng):
    d = Drawing()
    # Coordinates here are garment units: torso length is one, not image pixels.
    if k == 0:
        bw = rng.uniform(.52,.78)
        shoulder = bw*rng.uniform(.9,1.12)
        hem = bw*rng.uniform(.85,1.13)
        waist = bw*rng.uniform(.83,1.02)
        arm = rng.choice(['none','short','short','short'])
        end = rng.uniform(.26,.43)
        gap = rng.uniform(.08,.19)
        sleeve = rng.uniform(.16,.24)
        neckw = shoulder*rng.uniform(.22,.4)
        neckd = rng.uniform(.06,.17)
        neckkind = rng.choice(['round','round','v'])
    elif k == 2:
        bw = rng.uniform(.62,.90)
        shoulder = bw*rng.uniform(.86,1.05)
        waist = bw*rng.uniform(.9,1.03)
        hem = bw*rng.uniform(.84,1.04)
        arm = 'long'
        end = rng.uniform(.86,1.08)
        gap = rng.uniform(.04,.18)
        sleeve = rng.uniform(.12,.21)
        neckw = shoulder*rng.uniform(.2,.35)
        neckd = rng.uniform(.025,.13)
        neckkind = rng.choice(['round','round','round','v'])
    elif k == 4:
        bw = rng.uniform(.51,.80)
        shoulder = bw*rng.uniform(.88,1.10)
        waist = bw*rng.uniform(.86,1.04)
        hem = bw*rng.uniform(.95,1.20)
        arm = 'long'
        end = rng.uniform(.70,1.02)
        gap = rng.uniform(.045,.15)
        sleeve = rng.uniform(.11,.20)
        neckw = shoulder*rng.uniform(.17,.34)
        neckd = rng.uniform(.08,.28)
        neckkind = rng.choice(['v','v','round'])
    else:
        bw = rng.uniform(.53,.80)
        shoulder = bw*rng.uniform(.9,1.1)
        waist = bw*rng.uniform(.8,1.03)
        hem = bw*rng.uniform(.88,1.15)
        arm = rng.choice(['long','long','long','short','none'])
        end = rng.uniform(.83,1.06) if arm == 'long' else rng.uniform(.27,.45)
        gap = rng.uniform(.03,.16) if arm == 'long' else rng.uniform(.13,.27)
        sleeve = rng.uniform(.10,.19) if arm == 'long' else rng.uniform(.15,.23)
        neckw = shoulder*rng.uniform(.22,.40)
        neckd = rng.uniform(.09,.23)
        neckkind = rng.choice(['v','v','round'])
    bottom_round = rng.uniform(.015,.075) if k == 6 else rng.uniform(0,.025)
    d.polygon([(-shoulder/2,.065),(-bw/2,.28),(-waist/2,.64),
               (-hem/2,1-bottom_round),(-hem*.25,1),(hem*.25,1),
               (hem/2,1-bottom_round),(waist/2,.64),(bw/2,.28),
               (shoulder/2,.065),(shoulder*.22,0),(-shoulder*.22,0)])
    if arm == 'none':
        # Armholes in a tank top: subtract generic oval cutouts at the shoulders.
        for s in [-1,1]:
            d.ellipse(s*bw*.53,.12,bw*.15,.18,True)
    if arm != 'none':
        for s in [-1,1]:
            cuff = bw/2+gap
            pts = [(shoulder/2,.065),(shoulder/2+.09,.13),
                   (cuff+sleeve/2,end-.025),(cuff-sleeve/2,end+.025),
                   (bw*.43,.29)]
            d.polygon([(s*x,y) for x,y in pts])
    neckline(d,rng,neckw,neckd,neckkind)
    if k == 4 and rng.random() < .20:
        # Unfastened coat fronts: a narrow opening widening towards the hem.
        w = bw*rng.uniform(.03,.13)
        d.polygon([(0,.45),(-w,1.02),(w,1.02)],True)
    return d


def trousers(rng):
    d=Drawing()
    waist=rng.uniform(.27,.46)
    hip=waist*rng.uniform(1,1.13)
    crotch=rng.uniform(.27,.42)
    cuff=rng.uniform(.055,.13)
    # Straight/tapered legs descend from the two halves of the pelvis.
    # A small minority are genuinely wide-legged rather than all flaring out.
    center=waist*.25+rng.uniform(-.015,.04)
    if rng.random()<.15:
        cuff=rng.uniform(.12,.19)
        center+=.025
    d.polygon([(-waist/2,0),(waist/2,0),(hip/2,.20),
               (center+cuff/2,1),(center-cuff/2,1),
               (0,crotch),(-center+cuff/2,1),(-center-cuff/2,1),(-hip/2,.20)])
    return d


def dress(rng):
    d=Drawing()
    shoulder=rng.uniform(.25,.48)
    bust=shoulder*rng.uniform(.85,1.13)
    waist=bust*rng.uniform(.62,.91)
    waist_y=rng.uniform(.32,.53)
    hem=rng.uniform(max(.35,waist*1.1),.82)
    d.polygon([(-shoulder/2,.04),(-bust/2,.21),(-waist/2,waist_y),
               (-hem/2,.98),(-hem*.3,1),(hem*.3,1),(hem/2,.98),
               (waist/2,waist_y),(bust/2,.21),(shoulder/2,.04),
               (shoulder*.2,0),(-shoulder*.2,0)])
    ar=rng.choice(['none','none','short','short','long'])
    if ar!='none':
        end=rng.uniform(.22,.32) if ar=='short' else rng.uniform(.52,.74)
        gap=rng.uniform(.04,.14)
        width=rng.uniform(.07,.12)
        for s in [-1,1]:
            d.polygon([(s*shoulder/2,.04),(s*(shoulder/2+.07),.10),
                       (s*(bust/2+gap+width/2),end),
                       (s*(bust/2+gap-width/2),end+.01),(s*bust*.44,.23)])
    neckline(d,rng,shoulder*rng.uniform(.25,.5),rng.uniform(.04,.13),rng.choice(['v','round']))
    return d


def shoe(k,rng):
    d=Drawing()
    if k==7:
        h=rng.uniform(.29,.54)
        high=rng.random()<.13
        if high:
            h=rng.uniform(.48,.68)
        ankle=rng.uniform(.59,.78)
        top=rng.uniform(.17,.32)*h
        d.polygon([(0,.78*h),(.06,.63*h),(.23,.55*h),(.48,.28*h),
                   (ankle,0),(ankle+.08,.19*h),(.92,top),(1,.69*h),
                   (.99,.95*h),(.82,h),(.13,h),(0,.93*h)])
        if high:
            d.polygon([(.55,.53*h),(.6,-.12*h),(.92,-.10*h),(.96,.61*h)])
    elif k==9:
        h=rng.uniform(.44,1.05)
        shaft=rng.uniform(.40,.63)
        back=rng.uniform(.88,1)
        heel=rng.random()<.5
        sole=.80 if heel else .98
        d.polygon([(0,.80*h),(.06,.69*h),(.29,.61*h),(shaft,.46*h),
                   (shaft,.025*h),(back,0),(back+.025,.35*h),(1,.78*h),
                   (.97,sole*h),(.60,sole*h),(.22,.99*h),(.03,.99*h)])
        if heel:
            w=rng.uniform(.075,.20)
            d.polygon([(1-w,.76*h),(1,.76*h),(.99,h),(1-w*.85,h)])
        else:
            d.polygon([(.05,.91*h),(.97,.84*h),(.98,h),(.1,h)])
    else:
        h=rng.uniform(.22,.82)
        heel=rng.random()<.5
        heelh=rng.uniform(.18,.36) if heel else .04
        # A thin sole, an optional heel, and a sparse collection of straps.
        sole=[(0,h),(.25,h),(.60,h-heelh),(.97,h-heelh)]
        d.stroke(sole,rng.uniform(.035,.07))
        if heel:
            w=rng.uniform(.045,.16)
            d.polygon([(.95-w,h-heelh),(.99,h-heelh),(.99,h),(.97-w,h)])
        style=rng.integers(0,4)
        width=rng.uniform(.035,.13)
        if style==0:
            d.stroke([(.13,h-.02),(.29,h-.20),(.56,h-.16),(.62,h-heelh)],width)
            d.stroke([(.51,h-heelh),(.72,0),(.9,.08),(.92,h-heelh)],width)
        elif style==1:
            d.stroke([(.1,h),(.35,h*.36),(.84,0),(.92,h-heelh)],width)
            d.stroke([(.22,h*.45),(.62,h*.62),(.88,0)],width)
        elif style==2:
            d.stroke([(.1,h),(.3,h-.18),(.47,h-.16),(.55,h-heelh)],width*1.4)
            d.stroke([(.57,h-heelh),(.72,h*.2),(.92,h*.3),(.94,h-heelh)],width)
        else:
            for x in [.25,.48,.72]:
                y=h-heelh*max(0,(x-.25)/.35)
                d.stroke([(x-.13,y),(x,y-.22),(x+.13,y)],width)
    return d


def bag(rng):
    d=Drawing()
    kind=rng.choice(['handle','handle','handle','tote','clutch','rucksack'])
    w=rng.uniform(.66,1.18)
    h=rng.uniform(.43,.83)
    top=rng.uniform(.8,1.08)*w
    if kind=='rucksack':
        d.ellipse(0,.27,w/2,.40)
        d.polygon([(-w/2,.24),(-w/2,h),(w/2,h),(w/2,.24)])
        d.ellipse(0,-.1,w*.12,.12)
        d.ellipse(0,-.11,w*.065,.07,True)
    else:
        rounding=rng.uniform(.035,.12)
        d.polygon([(-top/2,0),(top/2,0),(w/2,h-rounding),
                   (w/2-rounding,h),(-w/2+rounding,h),(-w/2,h-rounding)])
        if kind!='clutch':
            handlew=top*rng.uniform(.35,.75)
            handleh=rng.uniform(.22,.60)
            thickness=rng.uniform(.035,.075)
            theta=np.linspace(np.pi,0,25)
            pts=[(handlew/2*np.cos(t),-handleh*np.sin(t)) for t in theta]
            d.stroke(pts,thickness)
    return d


def avg2(a):
    return a.reshape(14,2,14,2).mean((1,3))


def dilate(a):
    p=np.pad(a,1)
    return np.maximum.reduce([p[y:y+28,x:x+28] for y in range(3) for x in range(3)])


def template_arrays(count):
    detail=[]
    filled=[]
    for k in range(10):
        for j in range(count):
            rng=np.random.default_rng(20261119+100003*k+j)
            if k in (0,2,4,6):
                d=garment(k,rng)
            elif k==1:
                d=trousers(rng)
            elif k==3:
                d=dress(rng)
            elif k==8:
                d=bag(rng)
            else:
                d=shoe(k,rng)
            # Product photographs are centered and scaled by their longest side.
            a=d.raster(scale=rng.uniform(.94,1.0),dx=rng.uniform(-.018,.018),
                       dy=rng.uniform(-.018,.018),mirror=(j%2==0))
            detail.append(avg2(a)*.92)
            filled.append(avg2(dilate(a))*.98)
    return np.asarray(detail,np.float32).reshape(-1,196),np.asarray(filled,np.float32).reshape(-1,196)


def mean_rect(x0,x1,y0,y1):
    y,x=np.mgrid[0:28,0:28]
    x=(x+.5)/28
    y=(y+.5)/28
    m=((x>=x0)&(x<x1)&(y>=y0)&(y<y1)).astype(float)
    return m/max(1,m.sum())


def seam_cue(n):
    W=[]
    # Either a light placket or a dark front opening, sustained over three bands.
    for cx in [.5-1/28,.5,.5+1/28]:
        for width in [1/28,2/28]:
            for sign in [-1,1]:
                for a,b in [(.24,.42),(.42,.60),(.60,.78)]:
                    core=mean_rect(cx-width/2,cx+width/2,a,b)
                    sides=.5*(mean_rect(cx-width*2.5,cx-width/2,a,b)+
                              mean_rect(cx+width/2,cx+width*2.5,a,b))
                    W.append(sign*(core-sides).ravel())
    z=n.dense('front_line_responses','standardized',W,np.full(len(W),-.08))
    z=n.relu('front_line_positive',z)
    z=n.dense('front_line_negative',z,-np.eye(len(W)))
    z=n.reshape('front_line_bands',z,[len(W)//3,3,1])
    z=n.pool('front_line_consistency',z,[3,1])
    z=n.dense('front_line_strength',z,-np.eye(len(W)//3))
    z=n.reshape('front_line_candidates',z,[1,len(W)//3,1])
    return n.pool('front_line_best',z,[len(W)//3,1])


def collar_cue(n):
    y,x=np.mgrid[0:28,0:28]
    x=(x+.5)/28
    y=(y+.5)/28
    W=[]
    for cx in [.5-1/28,.5,.5+1/28]:
        for depth in [.22,.30,.38]:
            # A symmetric diagonal ridge surrounding the front of the neck.
            start=.07
            v=np.abs(x-cx)-.15*(depth-y)/(depth-start)
            region=(y>start)&(y<depth)
            ridge=np.exp(-.5*(v/.026)**2)*region
            surround=np.exp(-.5*(v/.065)**2)*region
            ridge/=max(ridge.sum(),1e-12)
            surround/=max(surround.sum(),1e-12)
            for sign in [-1,1]:
                W.append(sign*(ridge-surround).ravel())
    z=n.dense('collar_ridges','standardized',W,np.full(len(W),-.10))
    z=n.relu('collar_ridges_positive',z)
    z=n.reshape('collar_candidates',z,[1,len(W),1])
    return n.pool('collar_best',z,[len(W),1])


def texture_cue(n):
    W=np.zeros((4,1,3,3),np.float32)
    W[0,0,1,:]=[-.5,1,-.5]
    W[1]=-W[0]
    W[2,0,:,1]=[-.5,1,-.5]
    W[3]=-W[2]
    z=n.conv('fabric_curvature','standardized',W,padding=1)
    z=n.relu('fabric_curvature_positive',z)
    region=mean_rect(.33,.67,.28,.78)
    M=np.zeros((2,4,28,28),np.float32)
    M[0,0]=M[0,1]=region
    M[1,2]=M[1,3]=region
    z=n.dense('fabric_two_axes',z,M.reshape(2,-1))
    z=n.dense('fabric_axes_negated',z,-np.eye(2))
    z=n.reshape('fabric_axes_pool',z,[1,2,1])
    z=n.pool('fabric_weave_minimum',z,[2,1])
    return n.dense('fabric_weave',z,[[-1]])


def shoe_air_cue(n):
    # Morphological closing fills INTERNAL holes, unlike dilation alone,
    # which also counts the exterior border of a perfectly solid sneaker.
    z=n.conv('air_border','foreground',np.ones((1,1,1,1)),padding=4)
    z=n.pool('strap_neighborhood',z,[9,9],[1,1])
    z=n.conv('air_negated',z,-np.ones((1,1,1,1)),padding=4)
    z=n.pool('air_eroded_negated',z,[9,9],[1,1])
    region=mean_rect(.12,.88,.28,.78)
    a=n.dense('upper_coverage','foreground',region.reshape(1,-1))
    b=n.dense('upper_envelope',z,-region.reshape(1,-1))
    z=n.dense('negative_coverage',a,[[-1]])
    return n.op('open_upper','add',[z,b])


def fork_cue(n):
    # A pair of bright leg regions surrounding a vertically sustained dark gap.
    # This describes a fork, rather than merely a thin object like a dress.
    W=[]
    for cx in [.5-1/28,.5,.5+1/28]:
        for spread in [.07,.11,.15]:
            for y0,y1 in [(.50,.68),(.68,.86),(.86,.96)]:
                gap=mean_rect(cx-.022,cx+.022,y0,y1)
                sides=.5*(mean_rect(cx-spread-.03,cx-spread+.03,y0,y1)+
                          mean_rect(cx+spread-.03,cx+spread+.03,y0,y1))
                W.append((sides-gap).ravel())
    z=n.dense('leg_gap_responses','foreground',W)
    z=n.relu('leg_gap_positive',z)
    z=n.dense('leg_gap_negative',z,-np.eye(len(W)))
    z=n.reshape('leg_gap_bands',z,[len(W)//3,3,1])
    z=n.pool('leg_gap_consistency',z,[3,1])
    z=n.dense('leg_gap_strength',z,-np.eye(len(W)//3))
    z=n.reshape('leg_gap_candidates',z,[1,len(W)//3,1])
    return n.pool('leg_gap_best',z,[len(W)//3,1])


def plaid_cue(n):
    # Woven checks contain coherent lines in BOTH axes. Local curvature alone
    # also responds to knitwear, so require lines extending across the torso.
    W=[]
    for axis in [0,1]:
        for width in [1/28,2/28]:
            for center in np.linspace(.37,.63,7):
                if axis==0:
                    core=mean_rect(center-width/2,center+width/2,.30,.72)
                    sides=.5*(mean_rect(center-2*width,center-width/2,.30,.72)+
                              mean_rect(center+width/2,center+2*width,.30,.72))
                else:
                    core=mean_rect(.34,.66,center-width/2,center+width/2)
                    sides=.5*(mean_rect(.34,.66,center-2*width,center-width/2)+
                              mean_rect(.34,.66,center+width/2,center+2*width))
                for sign in [-1,1]:
                    W.append(sign*(core-sides).ravel())
    z=n.dense('plaid_lines','standardized',W,np.full(len(W),-.10))
    z=n.relu('plaid_lines_positive',z)
    z=n.reshape('plaid_orientations',z,[2,len(W)//2,1])
    z=n.pool('plaid_strongest_axes',z,[len(W)//2,1])
    z=n.dense('plaid_axes_negative',z,-np.eye(2))
    z=n.reshape('plaid_both_axes',z,[1,2,1])
    z=n.pool('plaid_axes_minimum',z,[2,1])
    return n.dense('plaid_strength',z,[[-1]])


def sleeve_cue(n):
    W=[]
    for x0,x1 in [(.02,.23),(.77,.98)]:
        for y0,y1 in [(.43,.63),(.63,.83)]:
            W.append(mean_rect(x0,x1,y0,y1).ravel())
    z=n.dense('lower_sleeve_presence','foreground',W)
    z=n.dense('sleeve_negative',z,-np.eye(4))
    z=n.reshape('sleeve_bands',z,[1,4,1])
    z=n.pool('sleeve_consistency',z,[4,1])
    return n.dense('sleeve_strength',z,[[-1]])


def opening_beyond_texture(n,seam,texture):
    a=n.dense('opening_signal',seam,[[1]])
    b=n.dense('opening_background',texture,[[-.65]])
    z=n.op('opening_contrast','add',[a,b])
    return n.relu('opening_distinct',z)


def bounded_cue(n,name,source,threshold,width,class_coeff):
    z=n.dense(name+'_hinges',source,[[1/width],[1/width]],[-threshold/width,-threshold/width-1])
    z=n.relu(name+'_positive',z)
    return n.dense(name+'_scores',z,np.asarray(class_coeff)[:,None]*np.array([[1,-1]]))


def build(destination):
    n=Network()
    # A two-threshold, piecewise-linear foreground probability; thresholds are
    # generic grayscale noise/visibility tolerances, not data measurements.
    z=n.conv('visibility_hinges','image',np.array([1/.075,1/.075,1/.15,1/.15]).reshape(4,1,1,1),
             [-.025/.075,-.025/.075-1,-.075/.15,-.075/.15-1])
    z=n.relu('visibility_positive',z)
    absolute=n.conv('absolute_foreground',z,np.array([.75,-.75,.25,-.25]).reshape(1,4,1,1))
    n.op('standardized','norm','image')
    # Estimate a black reference from all four corners, then use a generic
    # noise tolerance in units of within-image contrast. This is adaptive
    # arithmetic in the network, not an image-derived weight statistic.
    corner=(mean_rect(0,.14,0,.14)+mean_rect(.86,1,0,.14)+
            mean_rect(0,.14,.86,1)+mean_rect(.86,1,.86,1))/4
    W=np.eye(784)-np.ones((784,1))*corner.reshape(1,-1)
    rel=n.dense('above_black','standardized',W)
    rel=n.reshape('relative_image',rel,[1,28,28])
    rel=n.conv('relative_visibility',rel,np.array([1/.20,1/.20]).reshape(2,1,1,1),[-.06/.20,-.06/.20-1])
    rel=n.relu('relative_visibility_positive',rel)
    rel=n.conv('relative_foreground',rel,np.array([1,-1]).reshape(1,2,1,1))
    inv=n.conv('absolute_negative',absolute,-np.ones((1,1,1,1)))
    extra=n.op('relative_excess','add',[rel,inv])
    extra=n.relu('relative_excess_positive',extra)
    extra=n.conv('relative_blend',extra,np.full((1,1,1,1),.65))
    z=n.op('foreground','add',[absolute,extra])
    detail=n.pool('silhouette_detail',z,[2,2],kind='avgpool')
    padded=n.conv('foreground_border',z,np.ones((1,1,1,1)),padding=1)
    expanded=n.pool('foreground_closed',padded,[3,3],[1,1])
    filled=n.pool('silhouette_solid',expanded,[2,2],kind='avgpool')
    count=1024
    D,F=template_arrays(count)
    sd=.35*12/196
    sf=.65*12/196
    a=n.dense('detail_models',detail,2*sd*D,-sd*(D*D).sum(1))
    b=n.dense('solid_models',filled,2*sf*F,-sf*(F*F).sum(1))
    z=n.op('geometric_models','add',[a,b])
    z=n.reshape('models_by_class',z,[10,count,1])
    z=n.pool('best_geometry',z,[count,1])
    priors=np.array([0,-.10,0,0,-.06,0,-.10,0,0,0],np.float32)
    geometry=n.dense('geometry_scores',z,np.eye(10),priors)
    seam=seam_cue(n)
    collar=collar_cue(n)
    texture=texture_cue(n)
    distinct=opening_beyond_texture(n,seam,texture)
    a=bounded_cue(n,'opening',distinct,.035,.55,[0,0,-.25,0,.26,0,.20,0,0,0])
    b=bounded_cue(n,'lapels',collar,.02,.45,[-.08,0,-.10,0,.10,0,.16,0,0,0])
    plaid=plaid_cue(n)
    c=bounded_cue(n,'woven_pattern',plaid,.06,.45,[-.06,0,-.18,0,-.10,0,.40,0,0,0])
    air=shoe_air_cue(n)
    d=bounded_cue(n,'sandal_straps',air,.045,.19,[0,0,0,0,0,.85,0,-.20,0,-.20])
    fork=fork_cue(n)
    e=bounded_cue(n,'separate_legs',fork,.10,.50,[0,.40,0,-.10,0,0,0,0,0,0])
    sleeves=sleeve_cue(n)
    f=bounded_cue(n,'long_arms',sleeves,.055,.20,[-.30,0,.12,-.10,.12,0,.12,0,0,0])
    n.op('scores','add',[geometry,a,b,c,d,e,f])
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'network.json').write_text(json.dumps({'layers':n.layers},indent=2)+'\n')
    np.savez(destination/'weights.npz',**n.weights)
    print('Wrote',len(n.layers),'layers and',sum(a.size for a in n.weights.values()),'analytic weights.')


if __name__=='__main__':
    build(sys.argv[1])
