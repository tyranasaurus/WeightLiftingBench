#!/usr/bin/env python3
"""Build a data-free, analytically specified multiscale human-shape detector."""
import json, math, os, sys
from pathlib import Path
import numpy as np


def layer(name, op, inputs, **settings):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(settings)
    return d


def first_filters():
    """Oriented Gaussian derivatives and broad colour-opponent fields."""
    n, k = 16, 5
    W = np.zeros((n, 3, k, k), np.float32)
    b = np.zeros(n, np.float32)
    yy, xx = np.mgrid[-2:3, -2:3].astype(np.float64)
    g = np.exp(-(xx * xx + yy * yy) / (2 * 1.05 ** 2))
    g /= g.sum()
    rgb_gray = np.array([0.299, 0.587, 0.114])
    # Eight signed directional derivatives. After norm and ReLU, opposite
    # channels together are a contrast-polarity-independent oriented edge.
    for j in range(8):
        a = j * math.pi / 4
        q = xx * math.cos(a) + yy * math.sin(a)
        f = q * g
        f -= f.mean()
        f /= math.sqrt(np.sum(f * f))
        W[j] = rgb_gray[:, None, None] * f[None]
    # Smooth fields: luminance and opponent colours. Norm turns these into
    # within-photo relative brightness/warmth cues, robust to illumination.
    fields = [
        (0.299, 0.587, 0.114),
        (-0.299, -0.587, -0.114),
        (1.00, 0.15, -0.85),       # warm / skin-like against blue
        (-1.00, -0.15, 0.85),
        (0.80, -0.55, -0.25),      # red relative to green
        (-0.80, 0.55, 0.25),
        (0.35, 0.65, -1.00),       # yellow-green against blue
        (-0.35, -0.65, 1.00),
    ]
    for o, col in enumerate(fields, 8):
        W[o] = np.asarray(col)[:, None, None] * g[None]
    return W, b


def add_oriented_point(A, x, y, normal, amount=1.0, spread=0.48):
    """Paint an edge point into the two polarity channels nearest its normal."""
    k = A.shape[1]
    # Linear interpolation between the two nearest unsigned orientations.
    u = (normal % math.pi) / (math.pi / 4)
    j0 = int(math.floor(u)) % 4
    t = u - math.floor(u)
    chans = [(j0, 1-t), ((j0+1) % 4, t)]
    rad = 1
    for iy in range(max(0, int(math.floor(y))-rad), min(k, int(math.floor(y))+rad+2)):
        for ix in range(max(0, int(math.floor(x))-rad), min(k, int(math.floor(x))+rad+2)):
            z = math.exp(-((ix-x)**2 + (iy-y)**2) / (2*spread*spread))
            for c, q in chans:
                # c and c+4 are opposite signed derivatives.
                A[c, iy, ix] += amount * q * z
                A[c+4, iy, ix] += amount * q * z


def add_curve(A, points, amount=1.0):
    """Rasterize a polyline; the expected image gradient is normal to it."""
    for p, q in zip(points[:-1], points[1:]):
        x0, y0 = p; x1, y1 = q
        length = math.hypot(x1-x0, y1-y0)
        steps = max(2, int(math.ceil(length * 3)))
        normal = math.atan2(y1-y0, x1-x0) + math.pi/2
        for s in range(steps):
            t = (s + .5) / steps
            add_oriented_point(A, x0*(1-t)+x1*t, y0*(1-t)+y1*t,
                               normal, amount/steps*max(1.0, length))


def ellipse(A, cx, cy, rx, ry, amount=1.0, lo=0.0, hi=2*math.pi):
    steps = max(12, int(2*math.pi*max(rx, ry)*3))
    for i in range(steps):
        a = lo + (hi-lo)*(i+.5)/steps
        x, y = cx + rx*math.cos(a), cy + ry*math.sin(a)
        # Ellipse outward normal.
        normal = math.atan2(math.sin(a)/max(ry, .01), math.cos(a)/max(rx, .01))
        add_oriented_point(A, x, y, normal, amount * (hi-lo)/steps)


def uv(k, x, y):
    """Map coordinates in [-1,1] to a filter window."""
    return ((x+1)*(k-1)/2, (y+1)*(k-1)/2)


def poly(k, seq):
    return [uv(k, x, y) for x, y in seq]


def body_template(k, width=.28, arm=0, leg=0, skin=False):
    A = np.zeros((16, k, k), np.float64)
    cx = (k-1)/2
    # Head and neck/shoulder outline.
    hx, hy = uv(k, 0, -.69)
    ellipse(A, hx, hy, width*.47*(k-1)/2, .145*(k-1)/2, 1.18)
    # Outer torso, deliberately slightly tapered.
    left = [(-.08,-.54), (-width,-.45), (-width*.82,.12), (-width*.66,.30)]
    right = [(-x,y) for x,y in left]
    add_curve(A, poly(k,left), 1.0); add_curve(A, poly(k,right), 1.0)
    add_curve(A, poly(k,[(-width,-.45),(width,-.45)]), .58)
    # Arms: several canonical everyday/sports poses.
    if arm == 0:       # hanging / walking
        la=[(-width,-.42),(-width-.10,-.02),(-width-.04,.32)]
        ra=[(width,-.42),(width+.10,-.02),(width+.04,.32)]
    elif arm == 1:     # wide/down
        la=[(-width,-.42),(-.58,-.15),(-.69,.13)]
        ra=[(width,-.42),(.58,-.15),(.69,.13)]
    elif arm == 2:     # raised or reaching
        la=[(-width,-.42),(-.49,-.69),(-.64,-.84)]
        ra=[(width,-.42),(.49,-.69),(.64,-.84)]
    elif arm == 3:     # asymmetric
        la=[(-width,-.42),(-.55,-.12),(-.64,.15)]
        ra=[(width,-.42),(.49,-.60),(.56,-.82)]
    else:              # bent, hands near torso
        la=[(-width,-.42),(-.48,-.10),(-.22,.10)]
        ra=[(width,-.42),(.48,-.10),(.22,.10)]
    add_curve(A,poly(k,la),.72); add_curve(A,poly(k,ra),.72)
    # Hip line and leg axes/outer boundaries.
    add_curve(A,poly(k,[(-width*.66,.30),(width*.66,.30)]),.48)
    if leg == 0:
        legs=[[(-.12,.30),(-.13,.63),(-.15,.91)],[(.12,.30),(.13,.63),(.15,.91)]]
    elif leg == 1:
        legs=[[(-.10,.30),(-.24,.60),(-.42,.88)],[(.10,.30),(.23,.62),(.38,.91)]]
    elif leg == 2:
        legs=[[(-.11,.30),(-.06,.62),(-.29,.88)],[(.11,.30),(.21,.58),(.10,.90)]]
    else:
        legs=[[(-.11,.30),(-.31,.50),(-.50,.52)],[(.11,.30),(.28,.61),(.43,.84)]]
    for z in legs: add_curve(A,poly(k,z),.94)
    # A small warm head region helps a subset without excluding covered or
    # dark faces. Channel 10 is relative warmth, channel 12 relative red.
    if skin:
        for iy in range(k):
            for ix in range(k):
                dx=(ix-hx)/(max(1.,width*.45*(k-1)/2)); dy=(iy-hy)/(max(1.,.14*(k-1)/2))
                if dx*dx+dy*dy < .72:
                    A[10,iy,ix] += .20
                    A[12,iy,ix] += .08
    return finish_template(A, kind='body')


def upper_template(k, width=.48, turn=0, skin=False):
    A=np.zeros((16,k,k),np.float64)
    shift = turn*.09
    hx,hy=uv(k,shift,-.42)
    rx=.19*(k-1)/2; ry=.25*(k-1)/2
    ellipse(A,hx,hy,rx,ry,1.2)
    # Neck, shoulders, and sides of a bust.
    add_curve(A,poly(k,[(shift-.10,-.18),(-.20,-.08),(-width,.12),(-.72,.58),(-.74,.90)]),1.0)
    add_curve(A,poly(k,[(shift+.10,-.18),(.20,-.08),(width,.12),(.72,.58),(.74,.90)]),1.0)
    add_curve(A,poly(k,[(-width,.12),(0,.28),(width,.12)]),.46)
    if skin:
        for iy in range(k):
            for ix in range(k):
                d=((ix-hx)/(rx*.88))**2+((iy-hy)/(ry*.88))**2
                if d<1:
                    A[10,iy,ix]+=.23; A[12,iy,ix]+=.08
        # Hair/eye darkness is only a soft cue.
        for xn,yn in [(-.08,-.47),(.10,-.47),(shift,-.28),(shift,-.62)]:
            x,y=uv(k,xn+shift,yn)
            iy=int(round(y)); ix=int(round(x))
            if 0<=iy<k and 0<=ix<k: A[9,iy,ix]+=.16
    return finish_template(A,kind='upper')


def face_template(k, aspect=0, skin=True, profile=0):
    A=np.zeros((16,k,k),np.float64)
    sx=profile*.10
    hx,hy=uv(k,sx,-.03)
    rx=(.43+.05*aspect)*(k-1)/2; ry=(.61-.04*aspect)*(k-1)/2
    ellipse(A,hx,hy,rx,ry,1.12)
    # Relative warmth through the oval, with eyes/mouth darker than the face.
    if skin:
        for iy in range(k):
            for ix in range(k):
                d=((ix-hx)/(rx*.90))**2+((iy-hy)/(ry*.90))**2
                if d<1:
                    A[10,iy,ix]+=.30*(1-d*.35)
                    A[12,iy,ix]+=.10*(1-d*.35)
                elif d<1.45:
                    A[10,iy,ix]-=.035
        for xn,yn,w in [(-.17,-.14,.24),(.17,-.14,.24),(0,.28,.22),(0,-.49,.13)]:
            x,y=uv(k,xn+sx,yn)
            for iy in range(k):
                for ix in range(k):
                    z=math.exp(-((ix-x)**2+(iy-y)**2)/(2*(.07*k)**2))
                    A[9,iy,ix]+=w*z
    return finish_template(A,kind='face')


def crowd_template(k, rows=2):
    A=np.zeros((16,k,k),np.float64)
    centers=[(-.45,-.35),(0,-.43),(.45,-.33)]
    if rows==2: centers += [(-.25,.35),(.28,.32)]
    for x,y in centers:
        px,py=uv(k,x,y)
        ellipse(A,px,py,.12*(k-1)/2,.15*(k-1)/2,.72)
        # little shoulders under each head
        add_curve(A,poly(k,[(x-.20,y+.30),(x,y+.18),(x+.20,y+.30)]),.46)
    return finish_template(A,kind='crowd')


def nonperson_template(k, kind, variant=0):
    """Conservative counter-templates for common non-human COCO subjects."""
    A=np.zeros((16,k,k),np.float64)
    if kind == 'quadruped':
        # Horizontal trunk, head at one end, tail and four downward legs.
        y=-.08 + .05*(variant%2)
        add_curve(A,poly(k,[(-.52,y-.22),(.32,y-.22),(.48,y-.05),(.30,y+.22),(-.50,y+.22),(-.62,y)]),.80)
        hx,hy=uv(k,.58,y-.18)
        ellipse(A,hx,hy,.18*(k-1)/2,.20*(k-1)/2,.68)
        for j,x in enumerate((-.42,-.20,.12,.34)):
            slant=(.08 if (variant+j)%3==0 else (-.05 if (variant+j)%3==1 else 0))
            add_curve(A,poly(k,[(x,y+.18),(x+slant,.48),(x+slant*.7,.82)]),.62)
        add_curve(A,poly(k,[(-.52,y-.12),(-.73,-.35-.10*(variant%2)),(-.82,-.55)]),.45)
    elif kind == 'plate':
        # Two complete concentric rims are unlike the single open oval of a face.
        asp=(.62,.78,.92)[variant%3]
        cx,cy=uv(k,0,.02)
        ellipse(A,cx,cy,.76*(k-1)/2,asp*(k-1)/2,.90)
        ellipse(A,cx,cy,.52*(k-1)/2,asp*.66*(k-1)/2,.72)
        # Food commonly gives several warm, irregular regions within the rim.
        for x,y in [(-.20,-.15),(.18,-.08),(-.05,.23)]:
            px,py=uv(k,x,y)
            for iy in range(k):
                for ix in range(k):
                    z=math.exp(-((ix-px)**2+(iy-py)**2)/(2*(.10*k)**2))
                    A[10,iy,ix]+=.07*z
    elif kind == 'animalface':
        # Broad muzzle plus two pointed ears; useful against dog/cat close-ups.
        cx,cy=uv(k,0,.08)
        ellipse(A,cx,cy,.50*(k-1)/2,.48*(k-1)/2,.76,lo=.05*math.pi,hi=.95*math.pi)
        ellipse(A,cx,cy,.50*(k-1)/2,.48*(k-1)/2,.76,lo=1.05*math.pi,hi=1.95*math.pi)
        add_curve(A,poly(k,[(-.45,-.15),(-.52,-.75),(-.12,-.43)]),.88)
        add_curve(A,poly(k,[(.45,-.15),(.52,-.75),(.12,-.43)]),.88)
        mx,my=uv(k,0,.30)
        ellipse(A,mx,my,.27*(k-1)/2,.20*(k-1)/2,.58)
        for x in (-.19,.19):
            px,py=uv(k,x,-.03); A[9,int(round(py)),int(round(px))]+=.18
    elif kind == 'vehicle':
        # A long cabin/body over two round wheels.
        h=.06*(variant%2)
        add_curve(A,poly(k,[(-.82,.12),(-.58,-.20),(.28,-.24),(.62,.02),(.82,.14),(.76,.42),(-.76,.42),(-.82,.12)]),.72)
        for x in (-.48,.50):
            px,py=uv(k,x,.43)
            ellipse(A,px,py,.15*(k-1)/2,.18*(k-1)/2,.72)
    return finish_template(A,kind='counter')


def finish_template(A, kind):
    # Light edge-density penalty away from the hypothesised outline suppresses
    # textured foliage, shelves, and food without requiring a blank scene.
    posmask=A[:8].sum(0)>1e-8
    if kind in ('body','upper'):
        neg=np.ones_like(posmask,dtype=float); neg[posmask]=0
        A[:8] -= neg[None] * (0.010 if kind=='body' else .008)
    # Normalize positive mass. Because both edge polarities are represented,
    # roughly half of the edge mass fires on a real boundary.
    ps=A[A>0].sum()
    target={'body':2.1,'upper':2.0,'face':1.8,'crowd':2.0,'counter':2.0}[kind]
    if ps>0: A[A>0]*=target/ps
    ns=-A[A<0].sum()
    maxneg={'body':.30,'upper':.25,'face':.20,'crowd':.20,'counter':.20}[kind]
    if ns>maxneg: A[A<0]*=maxneg/ns
    return A


def detector_bank(k):
    arr=[]; kinds=[]
    # Upright and articulated full bodies: shape comes from generic anatomy,
    # not from any photograph.
    for width in (.22,.29,.37):
        for arm in range(5):
            for leg in (0,1,2):
                arr.append(body_template(k,width,arm,leg,skin=False)); kinds.append('body')
    # A few skin-assisted versions of common poses.
    for width,arm,leg in [(.25,0,0),(.31,0,1),(.30,1,1),(.35,4,0),(.24,3,2)]:
        arr.append(body_template(k,width,arm,leg,skin=True)); kinds.append('bodyskin')
    # Busts and close portraits.
    for width in (.40,.52,.64):
        for turn in (-1,0,1):
            arr.append(upper_template(k,width,turn,skin=False)); kinds.append('upper')
            arr.append(upper_template(k,width,turn,skin=True)); kinds.append('upperskin')
    for asp in (-1,0,1):
        for prof in (-1,0,1):
            arr.append(face_template(k,asp,True,prof)); kinds.append('face')
    arr.append(crowd_template(k,1)); kinds.append('crowd')
    arr.append(crowd_template(k,2)); kinds.append('crowd')
    # Counter-evidence is generated from general object geometry. It receives
    # its own vote rather than changing any human template.
    for kind, count in [('quadruped',6),('plate',3),('animalface',3),('vehicle',2)]:
        for v in range(count):
            arr.append(nonperson_template(k,kind,v)); kinds.append('nonperson_'+kind)
    W=np.stack(arr).astype(np.float32)
    # Thresholds are based on normalized-filter response magnitudes. These are
    # fixed analytic constants, shared across all scales and photographs.
    bias=[]
    for kind in kinds:
        if kind.startswith('nonperson_'):
            bias.append(-.72)
        else:
            bias.append({'body':-.78,'bodyskin':-.76,'upper':-.76,
                         'upperskin':-.73,'face':-.68,'crowd':-.70}[kind])
    return W,np.asarray(bias,np.float32),kinds


def build(dst):
    dst=Path(dst); dst.mkdir(parents=True,exist_ok=True)
    W1,b1=first_filters()
    banks={}
    kinds=None
    for k in (9,11,13):
        w,b,ks=detector_bank(k); banks[k]=(w,b)
        if kinds is None: kinds=ks
    D=len(kinds)
    layers=[
      layer('cues','conv',['image'],out=16,kernel=[5,5],stride=2,padding=2),
      layer('cue_norm','norm',['cues']),
      layer('cue_relu','relu',['cue_norm']),
      layer('small_det','conv',['cue_relu'],out=D,kernel=[9,9],stride=2,padding=0),
      layer('small_relu','relu',['small_det']),
      layer('small_max','maxpool',['small_relu'],size=[20,20],stride=[20,20]),
      layer('small_vec','reshape',['small_max'],shape=[D]),
      layer('small_vote','dense',['small_vec'],out=2),
      layer('medium_det','conv',['cue_relu'],out=D,kernel=[13,13],stride=2,padding=0),
      layer('medium_relu','relu',['medium_det']),
      layer('medium_max','maxpool',['medium_relu'],size=[18,18],stride=[18,18]),
      layer('medium_vec','reshape',['medium_max'],shape=[D]),
      layer('medium_vote','dense',['medium_vec'],out=2),
      layer('large_in','maxpool',['cue_relu'],size=[2,2],stride=[2,2]),
      layer('large_det','conv',['large_in'],out=D,kernel=[11,11],stride=1,padding=0),
      layer('large_relu','relu',['large_det']),
      layer('large_max','maxpool',['large_relu'],size=[14,14],stride=[14,14]),
      layer('large_vec','reshape',['large_max'],shape=[D]),
      layer('large_vote','dense',['large_vec'],out=2),
      layer('huge_in','maxpool',['cue_relu'],size=[3,3],stride=[3,3]),
      layer('huge_det','conv',['huge_in'],out=D,kernel=[11,11],stride=1,padding=0),
      layer('huge_relu','relu',['huge_det']),
      layer('huge_max','maxpool',['huge_relu'],size=[6,6],stride=[6,6]),
      layer('huge_vec','reshape',['huge_max'],shape=[D]),
      layer('huge_vote','dense',['huge_vec'],out=2),
      layer('scores','add',['small_vote','medium_vote','large_vote','huge_vote'])]
    net={'layers':layers}
    weights={'cues.W':W1,'cues.b':b1}
    for name,k in [('small_det',9),('medium_det',13),('large_det',11),('huge_det',11)]:
        weights[name+'.W']=banks[k][0]; weights[name+'.b']=banks[k][1]
    # Evidence is the amount by which a generic shape template clears its
    # conservative threshold. Correlated templates provide consensus.
    votes=np.zeros((2,D),np.float32)
    for j,kind in enumerate(kinds):
        if kind.startswith('nonperson_'):
            votes[0,j]={'nonperson_quadruped':.10,'nonperson_plate':.085,
                        'nonperson_animalface':.09,'nonperson_vehicle':.08}[kind]
        else:
            votes[1,j]={'body':.105,'bodyskin':.12,'upper':.095,
                        'upperskin':.11,'face':.075,'crowd':.10}[kind]
    for name in ('small_vote','medium_vote','large_vote','huge_vote'):
        weights[name+'.W']=votes.copy()
        weights[name+'.b']=np.array([7.875,0.0],np.float32)
    (dst/'network.json').write_text(json.dumps(net,separators=(',',':')))
    np.savez(dst/'weights.npz',**weights)


if __name__=='__main__':
    build(sys.argv[1] if len(sys.argv)>1 else '.')
